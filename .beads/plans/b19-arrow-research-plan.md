# B19 Research Plan: Cross-Process Shared Hat via Memory-Mapped Arrow IPC

Ticket: FoRecoML-dzb (P4, research-only). This document is the experimental methodology.

## Research Question

Can `mirai` daemons share a read-only view of `hat` (and `base`) across process boundaries via memory-mapped Arrow IPC files such that:
- Per-daemon RAM cost drops from `K × hat_bytes` to `1 × hat_bytes` (effectively shared via kernel page cache)?
- Per-iteration access latency overhead remains tolerable (< 10% wall-clock)?
- Output remains BYTE-IDENTICAL to status quo (no numerical drift)?
- Cleanup is robust to daemon crash, file truncation, abrupt session exit?
- Cross-platform: Linux + macOS at minimum; Windows nice-to-have?

Decision outcomes: **PROCEED** → file implementation ticket | **DEFER** → revisit at user request | **REJECT** → document why, close.

## I. Hypotheses (ultrathink)

### H1 (Architecture)
Arrow IPC `write_ipc_file()` + `read_ipc_file(mmap = TRUE)` provides cross-process zero-copy access:
- Main writes once → file in tmpfs / /tmp
- Mirai daemons receive only the file path (string, cheap NNG serialization)
- Each daemon `mmap`-opens the same file
- Kernel page cache deduplicates pages across processes → effective shared memory
- Read-only access; no write contention

### H2 (Per-iter cost)
Per `loop_body_kset` iteration, the dominant cost is `input2rtw_partial_from_parts(parts_hat, kset, cols)`. With Arrow:
- Status quo: parts_hat in daemon RAM as R list; per-iter does ~5 column slices each ~10–100 ns
- Arrow: parts_hat as Arrow Table; per-iter does ~5 Arrow column selects each ~1–10 µs (10–100× slower per slice)
- For p=2432 iterations × 5 slices × 10 µs = 122 ms total CPU added per rml() call
- Acceptable if hat is large enough that the RAM savings dominate

### H3 (Memory savings)
For n_workers = K daemons and raw hat bytes H:
- Status quo: K × H total RAM (each daemon serializes its own copy)
- Arrow IPC mmap: ~1 × H total RAM (kernel page cache shared)
- Savings ≈ (K − 1) × H

For user's XL case (H ≈ 3.3 GB, K = 4): savings ≈ 9.9 GB. For B19 enabled at K = 8: ≈ 23 GB.

### H4 (Breakeven)
B19 wins when memory pressure dominates CPU overhead:
- Setup cost: arrow conversion + IPC write ≈ 100 ms/GB
- Per-iter overhead: ~10 µs × n_slices × p iterations / K daemons
- Memory benefit: (K − 1) × H bytes

Crude breakeven (heuristic, to be empirically verified): `H × (K − 1) > 1 GB AND K ≥ 2`. Below this: native faster. Above: Arrow wins (especially under OOM pressure where the alternative is swap thrashing or crash).

### H5 (Failure modes)
1. **tmpfs full**: arrow::write_ipc_file fails → fallback to native dispatch (warn user)
2. **Daemon crash mid-mmap**: file remains; cleanup via main's `on.exit(unlink)` handles
3. **SIGBUS on truncated file**: prevent by ensuring main owns the file lifetime + uses tempdir
4. **Arrow version mismatch main vs daemon R**: mirai's `everywhere(library(FoRecoML))` ensures FoRecoML's dependency arrow loads consistently; daemons spawned from same install
5. **Concurrent reads**: kernel handles cleanly; read-only mmap has no contention

## II. Experimental Setup

### Hardware
- Primary: 16 GB RAM Linux (matches user's reported scenario)
- Secondary: 64 GB Linux + 32 GB macOS (cross-validation)
- Windows: deferred unless macOS shows blockers

### Software
- R ≥ 4.3
- arrow R package ≥ 14.0 (latest stable)
- mirai ≥ 2.0
- Linux: kernel ≥ 5.10 (for stable mmap+page-cache behavior)

### Fixtures
Five hat sizes spanning breakeven:
- Small: hat = 1 MB (n=100, h=4, kt=28) — should favor native
- Medium: hat = 50 MB (n=1000, h=24, kt=28)
- Large: hat = 500 MB (n=2000, h=48, kt=60)
- XL: hat = 3.3 GB (n=2432, h=288, kt=600) — matches user case
- XXL: hat = 10 GB (synthetic, single-machine stress)

n_workers ∈ {1, 2, 4, 8}. p ∈ {100, 500, 2432}. ML backend: lightgbm (fast, deterministic with nthread=1).

## III. Feasibility Experiments (must complete first)

### E1: mirai + arrow + mmap proof of concept
```r
# Main process
hat <- matrix(rnorm(n_rows * n_cols), nrow = n_rows)
# Explicit float64 schema to prevent silent int32 coercion (breaks B4 byte-identity)
schema_f64 <- arrow::schema(!!!setNames(rep(list(arrow::float64()), ncol(hat)), 
                                         paste0("V", seq_len(ncol(hat)))))
tbl <- arrow::Table$create(as.data.frame(hat), schema = schema_f64)
ipc_path <- tempfile(fileext = ".arrow")
arrow::write_ipc_file(tbl, ipc_path)

# Spawn daemons; pass ipc_path
mirai::daemons(K)
mirai::everywhere({ library(arrow); library(FoRecoML) })
results <- mirai::mirai_map(seq_len(p), function(i, path, cols) {
  tbl_view <- arrow::read_ipc_file(path, as_data_frame = FALSE)
  mat_slice <- as.matrix(as.data.frame(tbl_view[, cols, drop = FALSE]))
  # ... per-iter work on mat_slice ...
  sum(mat_slice)  # dummy
}, .args = list(path = ipc_path, cols = some_cols))[]
```

**Pass criteria**: 
- `mirai::everywhere(library(arrow))` succeeds on all daemons (distinct from RR3 serialization concern)
- Results identical to native dispatch (each daemon `hat[, cols]` from full R matrix)
- `mirai::status()$connections` returns to 0 after

### E2: Page-cache deduplication verification
Run E1 with hat = 1 GB, K = 4 daemons. Measure:
- Per-process RSS via `ps -p $(pgrep R) -o rss` (overcounts shared pages)
- **Definitive metric**: PSS (Proportional Set Size) from `/proc/<pid>/smaps_rollup` — divides shared pages proportionally across mappers, so K=4 daemons sharing 1 GB show PSS ≈ 250 MB each, totaling 1 GB (not 4 GB).
- Expected: Sum(PSS over all daemons + main) ≈ baseline + 1 GB (NOT baseline + 4 GB)
- Also report `Shared_Clean` from smaps_rollup as cross-check
- If sum exceeds 1 GB significantly: mmap NOT shared; B19 infeasible on this OS/arrow config

### E3: Cross-platform smoke test
Repeat E2 on macOS. Document any divergence in page-cache semantics.

### E4: Daemon crash recovery
Run E1 with `mirai_map` where one task force-exits mid-mmap (`Sys.kill(Sys.getpid(), 9)` from inside). Verify:
- Main process detects daemon failure cleanly via `errorValue` 19 (`Connection reset`)
- Other daemons continue
- on.exit cleanup runs (file removed)
- No SIGBUS or file-handle leak

**Nuance to test separately**:
- Mid-computation crash (daemon accepted task, died during work) → errorValue returned
- Pre-accept crash (daemon died before NNG socket handoff) → task re-dispatched if `dispatcher=TRUE`
Document both observable patterns in F2's writeup.

## IV. Benchmark Methodology

### B1: Setup overhead
Per hat size, time:
- `arrow::Table$create` + `arrow::write_ipc_file` (one-time per rml() call)
- vs native: serialize hat to mirai .args (status quo)

Tools: `bench::mark` with 10 iterations, GC-cleared between, median + 95% CI.

### B2: Per-iter access latency
For each kset level access pattern (small contiguous slice, scatter, full):
- Native: `hat[, cols]` direct R matrix slice
- Arrow: `tbl[, cols]` → as.matrix conversion
- Arrow zero-copy: `tbl$select(cols)$Slice(...)` without conversion (operate on arrow column refs)

Report ratio per slice + overhead per iteration.

### B3: End-to-end wall-clock
Full `ctrml(..., n_workers = K)` call timed for each (hat_size, K, p) combination. Compare:
- Native (status quo)
- B19 Arrow mmap

Tool: `bench::mark(..., iterations = 5, gc = TRUE)`. Measure both wall-clock and peak RSS via `system2("/usr/bin/time", "-v", ...)`.

### B4: Output byte-identity verification
For every (hat_size, K, p, ML backend) tuple: `expect_equal(r_arrow$bts, r_native$bts, tolerance = 0)`. Any drift = REJECT.

## V. Breakeven Derivation (empirical)

Fit two-variable surface from B3 results:
- x = hat_bytes × (K − 1)
- y = wall-clock ratio (Arrow / native)

Find the contour where y = 1.0. Above contour: Arrow wins. Below: native wins. Derive a simple linear or piecewise threshold:
- e.g., `hat_bytes × (K − 1) > T_breakeven AND K ≥ 2` → Arrow recommended

Tabulate T_breakeven for each platform. Compare to H4 heuristic (1 GB).

## VI. Failure Mode Tests

Each scenario in section H5 → reproducible test case in `dev/b19-failure-tests/`. All scripts ephemeral (discarded on REJECT/DEFER outcome).

1. **tmpfs full simulation**: bind-mount tmpfs of 10 MB, attempt 100 MB hat write, verify graceful fallback
2. **Daemon SIGKILL mid-iter**: trigger via signal, verify recovery via errorValue 19. Test both mid-computation and pre-accept crash variants per E4 nuance.
3. **Manual file deletion mid-run**: `unlink(ipc_path)` between mirai_map dispatch and collect, verify SIGBUS containment
4. **Arrow version mismatch (EPHEMERAL test script)**: install older arrow in daemon-load via temporary library path, document failure behavior; script deleted on DEFER/REJECT
5. **Truncated file**: write half hat, verify daemon errors cleanly (not crash)

## VII. Decision Rubric

| Outcome | Criteria |
|---|---|
| **PROCEED** | E1+E2+E3 pass; B4 output byte-identity holds across ≥ 95% of tuples; B3 shows Arrow wins at hat_bytes × (K−1) > 1 GB; failure modes mitigable with documented fallbacks |
| **DEFER** | E1 passes but E2 fails on Linux (page cache not shared); revisit when arrow/OS evolves |
| **REJECT** | E1 fails (arrow + mirai incompatible); OR B4 byte-identity fails (numerical drift); OR all benchmark scenarios show Arrow ≥ 10% slower with no memory benefit |

## VIII. Deliverables

1. **`docs/research/b19-arrow-shared-hat.md`** — full report with:
   - Hypothesis evaluation (H1–H5)
   - Benchmark tables (B1–B4)
   - Breakeven contour plot
   - Failure mode test results
   - Cross-platform notes
   - PROCEED/DEFER/REJECT decision with rationale
2. **`dev/b19-bench/`** — reproducible benchmark scripts
3. **`dev/b19-failure-tests/`** — failure mode reproducers
4. If PROCEED: **`dev/b19-design-doc.md`** — implementation design specifying:
   - `shared_hat = "auto" | TRUE | FALSE` resolver semantics
   - Refactor scope for `input2rtw_partial_from_parts` to accept arrow Tables
   - Fallback chain (Arrow fails → native)
   - cleanup contract (tempdir lifecycle, on.exit ordering)
   - mirai .args delta (path string instead of parts_hat list)
   - Test plan for implementation ticket

## IX. Out of Scope

- **Implementation** (held until PROCEED decision)
- **Wrapper API changes** (would happen in implementation phase)
- **GPU offload** (separate research thread)
- **bigmemory / POSIX shm direct alternatives** — Arrow IPC chosen as primary research vector (cross-platform R package, active maintenance, mmap-aware)
- **Spark/Dask integration** (different problem class)
- **Streaming arrow Flight** (cross-machine, not cross-process)

## X. Risks to the Research Itself

| ID | Risk | Mitigation |
|---|---|---|
| RR1 | Arrow R version skew across CI / dev machines | Pin arrow version in DESCRIPTION:Suggests for research period; document tested versions |
| RR2 | Benchmark variance from concurrent system load | Run on dedicated remote bench host (per RTK note); 5+ iterations with median; explicit cgroup if available |
| RR3 | mirai may serialize Arrow Tables incorrectly | E1 directly tests this; pivot to manual path-string passing if needed |
| RR4 | Byte-identity may break for arrow→R conversion (float precision) | B4 uses `tolerance = 0` strict check; if fails, document why and assess if user-acceptable |
| RR5 | Cleanup chain interaction with B3 auto-promote tempdir + B2 user pool | Test combined workflows in failure-mode suite (F1) |
| RR6 | macOS hardware availability for E3 cross-platform smoke; may slip timeline | Pre-procure / schedule access during week 1; fallback: defer cross-platform to follow-on if blocking |

## XI. Timeline & Effort

Estimate: 2–4 weeks of focused research (one engineer):
- Week 1: E1–E4 feasibility + cross-platform smoke
- Week 2: B1–B4 benchmark fixtures + execution
- Week 3: Failure mode tests + breakeven derivation
- Week 4: Report writeup + decision

Out-of-scope until PROCEED.

## XII. Success Definition

Research is COMPLETE when:
- [ ] All 4 feasibility experiments executed; results documented
- [ ] All 4 benchmark methods executed across 5 hat sizes × 4 n_workers × 3 p values
- [ ] Empirical breakeven derived with platform variance noted
- [ ] All 5 failure mode tests executed; mitigations validated
- [ ] PROCEED / DEFER / REJECT decision rendered with evidence
- [ ] If PROCEED: design doc drafted for next implementation ticket
- [ ] All deliverables committed to docs/research/

NO PRODUCTION CODE written during research.
