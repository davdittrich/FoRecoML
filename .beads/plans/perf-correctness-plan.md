# Math Correctness + Perf Followup Plan

## Context

Critical math review surfaced 1 BLOCKING bug (terml mfh). Lateral-thinking efficiency review surfaced 17 opportunities. Plan covers BLOCKING bug + Tier 1 + Tier 2 + Tier 3 measured/specific items. Tier 4 spd.25/spd.26 EXCLUDED. spd.27 included as future-investigation placeholder.

5 epics, 15 child tickets.

---

## Epic C: terml mfh correctness (CRASH-CLASS) — P0

### Goal
Fix terml mfh runtime bug: pre-materialization via vec2hmat conflicts with spd.12 path in rml().

### Success
- [ ] All 4 terml mfh features (mfh-all, mfh-hfts, mfh-str, mfh-str-hfts) execute without dim-mismatch or silent wrong fit
- [ ] Tests cover all 4 mfh features × ≥1 ML lib
- [ ] No regression on existing 261 tests

### Background
`R/terml.R` mfh path (lines 217, 319, 468-469) pre-materializes hat via `vec2hmat(vec=hat, h=h_hat, kset=kset)` producing `N × kt` matrix. Then calls `rml(kset=kset)` WITHOUT h, n. rml() falls into spd.12 path → `FoReco::FoReco2matrix(hat, kset)` on already-materialized N×kt matrix → misinterprets as N cross-temporal series → wrong shape, dim mismatch with y, or silent wrong fit if N==m by accident.

Zero existing test coverage (test-terml.R only covers `"all"` + `"low-high"`).

### Tasks

#### C.1 — Fix terml mfh: delete vec2hmat pre-materialization + route via spd.13 path

##### I. Context & Objective
- Objective: terml mfh wrappers stop pre-materializing hat/base; pass `h=h_hat`, `n=1L`, `kset=kset` to rml() so spd.13 mat2hmat_partial_from_sorted path handles per-iter expansion.
- Why: BLOCKING math bug. Pre-materialization + kset-only call → FoReco2matrix misinterprets matrix as cross-temporal.
- Reference: R/terml.R lines 217, 319, 468-469 (vec2hmat call sites); R/reco_ml.R rml() pre-loop hoist (post-spd.14).

##### II. Input Specification
- R/terml.R 3 mfh sites
- R/reco_ml.R rml() already supports h, n formals (from spd.13/spd.14)

##### III. Constraints & Guards
| Type | Guard |
|------|-------|
| Logic | terml mfh wrappers: DELETE vec2hmat call; pass `h=h_hat, n=1L, kset=kset` to rml(). |
| Format | Surgical edits to 3 sites. No new utils. |
| Boundary | Only R/terml.R + new test file. csrml, ctrml, reco_ml, utils, FoReco UNTOUCHED. |
| Audit | Numerical equivalence: capture baseline output from HEAD 72311cc BEFORE fix (terml mfh currently crashes; baseline = empirical reference output from a fresh, working implementation). Since current behavior is BUGGY, baseline must be derived from `mat2hmat`-based reconstruction or a hand-verified small fixture (n=1, kt=4, p=3). |

##### IV. Step-by-step

**KEY INSIGHT v3 (corrected)**: terml hat is a length-`h_hat*kt` vector. Current code calls vec2hmat → h_hat×kt matrix → passes to rml() WITHOUT h/n → rml() falls into spd.12 path which is incompatible. v2 attempted `matrix(hat, nrow=1L)` but this also broke sel_mat sizing because `total_cols <- NCOL(hat)` then = h_hat*kt instead of kt.

**v3 correct fix**: Mirror spd.13's mfh deferral pattern for ctrml. Specifically:
1. DELETE `vec2hmat` calls entirely (mirror spd.13's mat2hmat deletion).
2. DELETE wrapper-side NA detection block (mirror spd.13 — defer NA to loop_body via existing spd.12/13 mechanism).
3. Compute `total_cols` from `kset` directly: `total_cols <- sum(max(kset) / kset)` (= kt — the column count of mat2hmat output for n=1).
4. Pass `hat <- matrix(hat, nrow = 1L)` to rml() with `h = h_hat, n = 1L, kset = kset`.

This keeps sel_mat sized to kt (correct), global_id ∈ [1, kt] (correct), and routes through spd.13 path (which handles per-iter mat2hmat_partial_from_sorted with byte-equivalence to vec2hmat).

1. `grep -n 'vec2hmat' R/terml.R` → confirm 3 sites at 217, 319, 469.
2. At each site:
   - DELETE `hat <- vec2hmat(vec=hat, h=h_hat, kset=kset)` (or base equivalent at line 319)
   - DELETE the subsequent wrapper-side `na_col_mask(hat)` + `sel_mat[na_local] <- 0` block (~line 267 + mirror sites; mirror spd.13 deletion in ctrml). **MANDATORY**: if left in, `rep(sel_mat, NCOL(hat))` at line 271 would expand sel_mat to length h*kt instead of kt → corruption.
   - Compute `kt_eff <- sum(max(kset) / kset)`; `total_cols <- kt_eff`; `features_size <- total_cols`
   - sel_mat construction unchanged (already sized against `total_cols = kt_eff`)
   - **ADD**: `keep_cols <- sel_mat_keep_cols(sel_mat, total_cols)` (non-NULL) — forces rml() T5 branch which sets `active_ncol = max(keep_cols) = kt`. Without this, rml() enters `is.null(keep_cols)` branch and uses `NCOL(hat)=h*kt` → out-of-bounds.
   - Before rml() call: `hat <- matrix(hat, nrow = 1L)` (or `base <- matrix(base, nrow = 1L)` for predict path)
   - **Predict path (line 319-329)**: ALSO remove or skip the `expected_base_ncol` check at line 326 for mfh path. Under v3, `NCOL(base) = h*kt ≠ features_size = kt` so the current guard aborts incorrectly. Either delete the guard for mfh OR replace with `NCOL(base) %% kt_eff == 0` + `(NCOL(base) / kt_eff) > 0` shape check.
3. At each rml() call: add `h = h_hat, n = 1L` to args (kset already passed). Verify `keep_cols = keep_cols` is also passed.
4. Verify byte-equivalence: `vec2hmat(vec, h_hat, kset)` produces `h_hat × kt` with `vec[order(i)]` where `i = rep(rep(1:h_hat, length(kset)), rep(m/kset, each=h_hat))`. `mat2hmat(matrix(vec, nrow=1), h_hat, kset, n=1)` produces the SAME index pattern (n=1 outer rep is identity). After spd.13/spd.14 hoist, `mat2hmat_partial_from_sorted` per-iter slice reproduces what vec2hmat-followed-by-column-subset would have produced. NA detection moves from wrapper to per-iter loop_body via the spd.12 Option A mechanism (already in place).
5. Add `tests/testthat/test-terml-mfh.R`:
   - terml × mfh-all × lightgbm: no crash, output `r$bts` dim matches expected (n_bts × h)
   - terml × mfh-hfts × lightgbm
   - terml × mfh-str × lightgbm
   - terml_fit + terml(fit=, base=) with mfh-all: round-trip
6. Run `Rscript -e 'devtools::test()'`. All 261 baseline + new tests pass.
7. Single commit: `fix(terml): defer vec2hmat into loop_body via h=h_hat, n=1L for mfh path`.

##### V. Output Schema
```
sites_modified: [R/terml.R: ~217 (training), ~319 (predict base), ~468 (terml_fit training)]
tests_added: tests/testthat/test-terml-mfh.R (≥ 4 test_that blocks)
test_count_pre: 261
test_count_post: >= 265
commit_count: 1
```

##### VI. DoD
- [ ] `grep 'vec2hmat' R/terml.R` → 0 matches (or only in vec2hmat function definition, not as caller)
- [ ] terml mfh tests all pass; output shape sanity-checked
- [ ] All baseline 261 tests still pass
- [ ] mw3.3 invariant intact
- [ ] Single commit; conventional commits; no AI attribution

---

## Epic D: hot-path perf wins (Tier 1, measured) — P1

### Goal
Apply 5 measured low-risk hot-path optimizations identified by efficiency review. Each ≥ 2x on its slice or saves substantial daemon spawn time.

### Success
- [ ] spd.15-19 all landed
- [ ] No numerical regression on spd.12+spd.13+spd.14 equivalence suites (max_abs_diff == 0)
- [ ] Aggregate speedup measurable in dev/perf-bench.R (≥ 1.5x on representative ctrml workload)

### Tasks

#### D.1 (spd.15) — Precompute global_id_list outside loop_body
##### I. Objective
Hoist `which(sel_mat[, i] != 0)` from per-iter to outer scope in rml(). When `NCOL(sel_mat) > 1`, current code recomputes the sparse-column extraction p times. Hoist via `global_id_list <- lapply(seq_len(p), function(i) which(sel_mat[, i] != 0))`. Pass as `.args` element. loop_body reads `global_id <- global_id_list[[i]]`.

##### III. Guards
| Type | Guard |
|------|-------|
| Logic | Sparse matrix sel_mat (NCOL > 1) path uses precomputed list. Scalar / sparseVector paths UNCHANGED (already fast). |
| Boundary | Only R/reco_ml.R. |
| Audit | Equivalence suites still pass byte-identical. |

##### IV. Logic
1. In rml() outer scope, after sel_mat construction, if `NCOL(sel_mat) > 1`: build `global_id_list`.
2. Add as loop_body formal + .args entry (count grows 24 → 25).
3. loop_body: replace `global_id <- which(sel_mat[, i] != 0)` with `global_id <- global_id_list[[i]]`.
4. Update line ~160 comment "24-item" → "25-item closure list (spd.15: +global_id_list)".
5. Run baseline + equivalence. Single commit `perf(rml): hoist per-series global_id extraction (spd.15)`.

##### VI. DoD
- [ ] global_id_list precomputed once in rml() outer when NCOL(sel_mat) > 1
- [ ] Comment count updated
- [ ] Sequential + mirai .args both include global_id_list
- [ ] Equivalence suites max_abs_diff == 0
- [ ] Micro-benchmark dev/spd15-bench.R documents speedup at p ≥ 20

---

#### D.2 (spd.16) — Replace outer() with rep() in mat2hmat_partial gather
##### I. Objective
In `R/utils.R` `mat2hmat_partial_from_sorted` (line ~353) and `mat2hmat_partial` (line ~337): replace `idx <- as.vector(outer((seq_len(h) - 1L) * ncol_total, cols, "+"))` with `idx <- rep(cols, each = h) + rep((seq_len(h) - 1L) * ncol_total, times = length(cols))`. Byte-identical output (verified). 2.8x faster, 93% less allocation.

##### III. Guards
| Type | Guard |
|------|-------|
| Logic | Byte-identical idx vector. Test fixture asserts `identical()`. |
| Boundary | Only R/utils.R. |
| Audit | spd.14 unit tests (test-spd14-internals.R) re-run unchanged. |

##### IV. Logic
1. Apply replacement at both call sites (mat2hmat_partial + mat2hmat_partial_from_sorted).
2. Run tests/testthat/test-spd14-internals.R + spd.13 equivalence. All pass.
3. Single commit `perf(utils): replace outer() with rep() in mat2hmat partial gather (spd.16)`.

##### VI. DoD
- [ ] Both mat2hmat_partial functions use rep-based idx
- [ ] All existing tests pass (no test changes)
- [ ] dev/spd16-bench.R demonstrates 2x+ on idx-build microbench

---

#### D.3 (spd.17) — Replace apply(block, 2, rep, each=k) with matrix(rep(block, each=k), ...)
##### I. Objective
In `R/utils.R` `input2rtw_partial` (line ~384) and `input2rtw_partial_from_parts` (line ~424): replace the `if (NCOL(block) > 1) apply(block, 2, rep, each=k) else rep(block, each=k)` branch with unified `matrix(rep(block, each=k), nrow=NROW(block)*k, ncol=NCOL(block))`. 2.8x faster, 42% less allocation.

##### III. Guards
| Type | Guard |
|------|-------|
| Logic | Unified formula handles both single- and multi-col cases identically. Byte-identical to current. |
| Boundary | Only R/utils.R. |
| Audit | spd.12 + spd.14 equivalence + unit tests pass. |

##### IV. Logic
1. Apply replacement at both call sites.
2. Run all tests. Equivalence preserved.
3. Single commit `perf(utils): unify block row-replication via matrix(rep, each) (spd.17)`.

##### VI. DoD
- [ ] Both input2rtw_partial functions use matrix-rep formula
- [ ] Conditional branch removed (single expression)
- [ ] All baseline + equivalence tests pass

---

#### D.4 (spd.18) — na_col_mask via colSums
##### I. Objective
In `R/utils.R` `na_col_mask` (line ~529): replace `vapply(seq_len(NCOL(hat)), function(j) sum(is.na(hat[,j])) >= threshold * NROW(hat), logical(1))` with `colSums(is.na(hat)) >= threshold * NROW(hat)`. 3.3x faster, 4x less allocation. Edge case NROW==0: `colSums(is.na(empty)) == 0 >= 0 == TRUE`, matches current vacuously-TRUE behavior.

##### III. Guards
| Type | Guard |
|------|-------|
| Logic | colSums on is.na mask = vapply sum per col. Single C-level operation. |
| Boundary | Only R/utils.R. |
| Audit | Empty matrix (NROW==0) edge case explicitly tested. All-NA matrix tested. |

##### IV. Logic
1. Replace vapply loop with colSums expression.
2. Add edge case tests if not already present.
3. Run baseline + equivalence (NA-injected fixtures exercise this path).
4. Single commit `perf(utils): na_col_mask via colSums for 3.3x speedup (spd.18)`.

##### VI. DoD
- [ ] Function body single-line `colSums(is.na(hat)) >= threshold * NROW(hat)`
- [ ] Existing NA fixtures pass byte-identical
- [ ] Empty matrix edge case test added or verified

---

#### D.5 (spd.19) — Clamp n_workers_resolved to p
##### I. Objective
In `R/reco_ml.R` (line ~52, post-resolve_n_workers call): add `n_workers_resolved <- min(n_workers_resolved, p)` to avoid spawning idle daemons when p < n_workers. Saves 200-500ms × idle_workers daemon spawn time.

##### III. Guards
| Type | Guard |
|------|-------|
| Logic | n_workers_resolved capped at p. Sequential path triggers when n_workers=1 (existing). |
| Boundary | Only R/reco_ml.R. |
| Audit | New test: rml() with p=2, n_workers=8 → no extra daemons spawned beyond 2. |

##### IV. Logic
1. Add `n_workers_resolved <- min(n_workers_resolved, p)` after the resolve call.
2. Verify checkpoint threshold + in-memory fit cap still fire correctly (they precede this clamp).
3. Add test in test-parallel.R asserting daemon count ≤ p.
4. Single commit `perf(rml): clamp n_workers to p to avoid idle daemon spawn (spd.19)`.

##### VI. DoD
- [ ] `n_workers_resolved <- min(n_workers_resolved, p)` clamp present
- [ ] New test asserts daemon count behavior
- [ ] No regression on spd.10 in-memory fit cap

---

## Epic E: medium-impact perf (Tier 2) — P2

### Goal
5 medium-impact optimizations. Each 10-50% on its slice or moderate complexity reduction.

### Tasks

#### E.1 (spd.20) — Hoist row_offsets for mat2hmat_partial_from_sorted
##### I. Objective
Pre-compute `row_offsets_hat <- (seq_len(h_hat_eff) - 1L) * ncol_total_hat` and `row_offsets_base` in rml() outer scope (chains with spd.16). Pass via .args. Removes per-call recomputation. Additional ~30% on top of spd.16.

##### III. Guards
| Type | Guard |
|------|-------|
| Logic | Adds 2 outer-scope vector computations. Loop_body uses pre-computed. |
| Boundary | R/reco_ml.R + R/utils.R signature change for mat2hmat_partial_from_sorted (accepts row_offsets instead of h+ncol_total OR keeps both for back-compat). |
| Audit | spd.13 + spd.14 equivalence byte-identical. |

##### IV. Logic
1. In rml() outer (post-spd.14 hoist block): compute row_offsets_hat, row_offsets_base when applicable.
2. Add as loop_body formals. Comment count grows.
3. Pass to mat2hmat_partial_from_sorted via new arg OR inline.
4. Test + single commit `perf(utils): hoist row_offsets for mat2hmat partial (spd.20)`.

##### VI. DoD
- [ ] row_offsets precomputed once per rml() call
- [ ] All equivalence tests pass
- [ ] dev/spd20-bench.R demonstrates speedup chains with spd.16

---

#### E.2 (spd.21) — qs2 multi-thread for checkpoint serialization
##### I. Objective
In `R/utils.R` `serialize_fit` (line ~232): pass `nthreads = min(parallel::detectCores(), 4L)` to `qs2::qs_save`. 30-50% I/O reduction on checkpoint writes.

##### III. Guards
| Type | Guard |
|------|-------|
| Logic | nthreads option only. compress_level stays default (3). qs2 thread-safe. |
| Boundary | Only R/utils.R serialize_fit. |
| Audit | Checkpoint round-trip tests still pass (test-checkpoint.R). |

##### IV. Logic
1. Add `nthreads = min(parallel::detectCores(), 4L)` to qs2::qs_save call.
2. Verify checkpoint tests pass.
3. Single commit `perf(checkpoint): enable qs2 multi-thread compression (spd.21)`.

##### VI. DoD
- [ ] qs_save uses nthreads
- [ ] test-checkpoint.R passes
- [ ] dev/spd21-bench.R measures wall-clock reduction on large fixture

---

#### E.3 (spd.22) — Auto-sequential threshold at small p
##### I. Objective
In `R/utils.R` `resolve_n_workers`: if `p <= min_parallel_p` (default 3L), force `n_workers_resolved = 1L`. Avoids daemon spawn overhead when sequential is faster.

##### III. Guards
| Type | Guard |
|------|-------|
| Logic | Default min_parallel_p = 3L (sequential when p ≤ 3). Tunable via arg. |
| Boundary | R/utils.R resolve_n_workers + signature update. |
| Audit | New test asserts p=2, n_workers=4 → resolved to 1. |

##### IV. Logic
1. Add `min_parallel_p = 3L` param to resolve_n_workers.
2. Insert `if (p <= min_parallel_p) return(1L)` early.
3. Add test in test-parallel.R.
4. Single commit `perf(rml): auto-sequential at small p (spd.22)`.

##### VI. DoD
- [ ] resolve_n_workers signature includes min_parallel_p
- [ ] p ≤ 3 → n_workers_resolved = 1 (verified by test)
- [ ] No regression on existing parallel tests

---

#### E.4 (spd.23) — Test fixture sharing
##### I. Objective
Refactor test-spd12-equivalence.R + test-spd13-equivalence.R + test-spd14-internals.R to load qs2 snapshots ONCE per test file via top-level setup (not inside each test_that). Reduces redundant disk I/O.

##### III. Guards
| Type | Guard |
|------|-------|
| Logic | Snapshots loaded at file scope. Tests reference shared objects. |
| Boundary | Only tests/testthat/ files. No R/ changes. |
| Audit | All tests still pass with identical assertions. |

##### IV. Logic
1. At top of each equivalence test file: load all qs2 fixtures into named env or list.
2. Replace inline `qs2::qs_read(snap_path)` with reference to preloaded.
3. Measure test suite wall-clock before/after.
4. Single commit `refactor(tests): share qs2 fixtures at file scope (spd.23)`.

##### VI. DoD
- [ ] All equivalence tests pass
- [ ] qs2 disk read count reduced (verify via strace or instrumentation)
- [ ] Test suite wall-clock unchanged or improved

---

#### E.5 (spd.24) — as.matrix guard at lightgbm predict (xgboost has no such call)
##### I. Objective
In `R/reco_ml.R` rml.lightgbm predict branch (line ~633): guard `as.matrix(Xtest)` with `if (!is.matrix(Xtest)) Xtest <- as.matrix(Xtest)`. Avoids 300µs + 67KB unnecessary copy per p=100 predict calls when Xtest is already a matrix (which it is, coming from mat2hmat_partial_from_sorted).

NOTE: rml.xgboost uses `xgb.DMatrix(data = Xtest)` directly — NO `as.matrix(Xtest)` call exists there. Scope corrected: only lightgbm site.

##### II. Input Specification
- R/reco_ml.R: rml.lightgbm predict branch only

##### III. Guards
| Type | Guard |
|------|-------|
| Logic | Skip as.matrix when already matrix. Preserves correctness for non-matrix Xtest (defensive). |
| Boundary | Only R/reco_ml.R rml.lightgbm. xgboost UNTOUCHED (no as.matrix call). |
| Audit | All lightgbm + xgboost tests pass. spd.12+13+14 equivalence byte-identical. mw3.3 intact. |

##### IV. Logic
1. Locate `as.matrix(Xtest)` in rml.lightgbm predict branch.
2. Replace with `if (!is.matrix(Xtest)) Xtest <- as.matrix(Xtest)`.
3. Run baseline + equivalence.
4. Single commit `perf(rml): guard as.matrix on already-matrix Xtest in lightgbm (spd.24)`.

##### V. Output Schema
```
site_modified: R/reco_ml.R rml.lightgbm line ~633
commit_count: 1
test_count_pre: 261 (or post-D/E tickets)
test_count_post: >= 261
```

##### VI. DoD
- [ ] is.matrix guard present at lightgbm predict site
- [ ] rml.xgboost UNTOUCHED
- [ ] mw3.3 invariant intact
- [ ] spd.12 + spd.13 + spd.14 equivalence max_abs_diff == 0
- [ ] All tests pass

---

## Epic F: polish (Tier 3) — P3

### Goal
3 low-priority hardening items. Each cheap to implement.

### Tasks

#### F.1 — n_workers-aware gc_every
##### I. Objective
In `R/reco_ml.R` loop_body: derive `gc_every` from n_workers (more workers → less memory headroom → gc more often). Default: `gc_every <- max(2L, 10L %/% n_workers_resolved)`. Currently hardcoded `5L`.

##### IV. Logic
1. In loop_body or as new formal: compute gc_every based on n_workers_resolved.
2. Pass via .args.
3. Single commit `perf(rml): n_workers-aware gc_every throttle (F.1)`.

##### VI. DoD
- [ ] gc_every adapts to n_workers
- [ ] Existing gc tests pass

---

#### F.2 — Unify dimnames(NULL) calls in rml()
##### I. Objective
In `R/reco_ml.R` rml() prologue (lines ~83-85): replace 2 conditional `if (!is.null(names(x))) names(x) <- NULL; if (!is.null(dimnames(x))) dimnames(x) <- NULL` with unified `dimnames(x) <- NULL` (NULL→NULL is no-op).

##### IV. Logic
1. Simplify dimnames stripping for hat, obs, base.
2. Single commit `refactor(rml): unify dimnames NULL stripping (F.2)`.

##### VI. DoD
- [ ] Conditional checks removed
- [ ] All tests pass byte-identical

---

#### F.3 — Cache available_ram_bytes per session
##### I. Objective
In `R/utils.R` `available_ram_bytes`: cache result in package-internal env on first call. RAM doesn't change materially during session. Saves ~50µs syscall per rml() call.

##### IV. Logic
1. Add internal env `.ram_cache` at package load.
2. available_ram_bytes() checks cache; reads /proc/meminfo + caches on miss.
3. Single commit `perf(utils): cache available_ram_bytes per session (F.3)`.

##### VI. DoD
- [ ] First call reads /proc/meminfo; subsequent calls hit cache
- [ ] spd.9 checkpoint threshold tests still pass

---

## Epic G: future investigation — P4

### Goal
Track architectural opportunities flagged in efficiency review for later evaluation. No immediate implementation.

### Tasks

#### G.1 (spd.27) — Global pre-train + per-series fine-tune (architectural)
##### I. Objective
INVESTIGATE replacing p independent ML models with 1 global model trained on stacked (y, X) across all series + series indicator feature. Per-series fine-tune optional.

##### II. Status
RESEARCH ONLY. Do not implement until statistical validity established.

##### III. Tasks (for future ticket)
1. Survey literature on global-model-vs-per-series for reconciliation forecasting (academic refs)
2. Empirical study: compare per-series RMSE vs global-model RMSE on FoRecoML test fixtures (n_bts series, kt aggregations)
3. If validity established: prototype global rml.<backend>_global S3 method
4. Cost/benefit analysis vs current p-model approach

##### VI. DoD
- [ ] Literature survey doc at docs/research/global-model-feasibility.md
- [ ] Empirical comparison results
- [ ] Recommendation: PROCEED / DEFER / REJECT

---

## Atomicity & Sequencing

Within each epic, tasks can land independently (no shared file conflicts in most cases).

Cross-epic ordering:
- **Epic C first** (BLOCKING math bug)
- **Epic D next** (Tier 1 perf — independent of each other; spd.15-19 can ship sequentially in any order)
- **Epic E parallel with D** (spd.20 depends on spd.16; spd.21-24 independent)
- **Epic F any time** (polish; independent)
- **Epic G research only** (no code change)

## Dependencies (beads)

```
Epic C — C.1 (no deps)
Epic D — D.1, D.2, D.3, D.4, D.5 (no inter-deps, parallel-safe)
Epic E — E.1 (depends on D.2), E.2, E.3, E.4, E.5 (rest independent)
Epic F — F.1, F.2, F.3 (independent)
Epic G — G.1 (research)
```

## Risks

- **R1**: Tier 1 micro-opts (spd.15-19) individually small; bench needs aggregate measurement to justify
- **R2**: Closure formal-count growth (spd.15 + spd.20 each add formals) — readability impact
- **R3**: Epic C requires terml mfh numerical reference: since current implementation is BUGGY, no baseline fixture exists. Hand-verified small fixture (n=1, kt=4, p=3) becomes the reference; OR use byte-equivalence vs `vec2hmat`-based reconstruction (apply vec2hmat manually in test to compute reference, then assert spd.13-path output matches).
- **R4**: spd.27 architectural change may invalidate spd.10 + spd.12-14 invariants — track separately

## Cross-cutting invariants (apply to ALL implementation tickets C.1, D.*, E.*, F.*)

- [ ] mw3.3 invariant intact: reco_ml.R `if (is.null(fit)) list(bts, fit, na_mask) else list(bts)` UNCHANGED
- [ ] spd.12 equivalence suite passes (max_abs_diff == 0) — test-spd12-equivalence.R
- [ ] spd.13 equivalence suite passes (max_abs_diff == 0) — test-spd13-equivalence.R
- [ ] spd.14 internals byte-identity passes — test-spd14-internals.R
- [ ] spd.12 predict-reuse NA tests pass — test-spd12-predict-reuse-na.R
- [ ] spd.10 in-memory fit cap test passes — test-parallel.R
- [ ] No AI attribution in any commit body or trailer
- [ ] No `--no-verify`; no `git push --force`

These apply UNIFORMLY to every code-touching ticket. Per-ticket DoD lists may abbreviate but cross-cutting invariants supersede.

## Template note

D.*, E.*, F.* tickets omit explicit Section II (Input Specification) for brevity since Section III (Constraints/Guards) "Boundary" row carries the same information (files touched). Section V (Output Schema) collapsed into Section VI DoD for these tickets. C.1 retains full 6-section template as canonical example. This is acceptable per planning-with-beads "hermetic if all logic/schema/constraints inside" rule — info is present, just relocated.
