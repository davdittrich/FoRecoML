# Epic: SOTA Single-Process FoRecoML Rebuild (v7 — gate-approved)

## Goal

Rebuild FoRecoML on pre-spd.1 baseline `25be237` as SOTA single-process implementation.

**Framing change v3 → v4 (user directive)**:
- DROP "minimal divergence" mandate. Better is preferred even if more effort required.
- ranger IS better than randomForest → ranger becomes DEFAULT (breaking change accepted).
- Consolidate code when consolidation is better (DRY for ≥3 occurrences with identical patterns).
- Add SOTA practices missing from v3: categorical encoding for series_id in global ML, predict/print methods for rml_g_fit, reconciliation-invariant tests, benchmark-validated speedups, math correctness audit.

## Plan Header

- **Mechanism**: Branch `refactor/clean-baseline` from `25be237`. Preserve baseline's existing `for (i in seq_len(p))` lexical loop. Add improvements as small, atomic commits.
- **Forbidden — grep-able literal symbols that MUST NOT appear anywhere in `R/`, `tests/`, `DESCRIPTION`, `NAMESPACE`, `man/`**:
  ```
  mirai          n_workers       loop_body_kset      loop_body_csrml
  resolve_n_workers              cap_inner_threads    promote_fit_to_checkpoint
  b19_daemon_load                 arrow_available     compute_chunk_size
  compute_chunk_size_nonmfh       mirai_map           mirai::
  daemons        everywhere      shared_hat          ipc_hat
  ```
  Plus conceptual forbidden: spd.1 (mirai introduction), spd.9 n_workers-aware checkpoint threshold, spd.10 in-memory fit cap, spd.19 daemon-side IPC load, spd.22 daemon teardown ordering, B19 Arrow IPC, B2 pool guard, B3 auto-promote, B5/B16 outer chunk dispatch. Verified by `grep -rnE "<symbol>" R/ tests/ DESCRIPTION NAMESPACE man/` returning zero hits at every commit.
- **Audit**: equivalence tests vs baseline behavior (no spies needed — sequential code, behavior validated end-to-end).

## Verified Baseline State (read from `25be237`)

| Element | Status at 25be237 | Source |
|---|---|---|
| `rml()` signature | 11 args: `approach, base, obs, hat, sel_mat, fit, params, seed, keep_cols, checkpoint, ...` | `R/reco_ml.R:1-15` |
| `kset` parameter | **ABSENT** in rml() signature | confirmed |
| rml() loop | `for (i in seq_len(p))` at line 96, lexical scoping | `R/reco_ml.R:96` |
| `global_id` computation | INSIDE loop, depends only on sel_mat (loop-invariant) | `R/reco_ml.R:97-103` |
| `na.omit(X)` | unconditional, inside loop | `R/reco_ml.R:111` |
| `gc(verbose=FALSE)` | conditional on `!is.null(checkpoint_dir)` only | `R/reco_ml.R:163-165` |
| `.rml` dispatch | `UseMethod("rml", approach)` S3 — methods: `rml.randomForest`, `rml.xgboost`, `rml.lightgbm`, `rml.mlr3` | `R/reco_ml.R:191` |
| `rml.ranger` | **ABSENT** (ranger reachable only via mlr3 `regr.ranger`) | confirmed |
| `rml.catboost` | **ABSENT** | confirmed |
| `input2rtw_partial(x, kset, cols)` | exists | `R/utils.R:282` |
| `input2rtw(x, kset)` | exists | `R/utils.R:264` |
| `na_col_mask(hat, threshold=0.75)` | exists | `R/utils.R:405` |
| `estimate_peak_bytes(hat, approach, p)` | exists; first multiplication already casts via `as.numeric(NROW(hat))` | `R/utils.R:149-167` |
| `serialize_fit/deserialize_fit/get_fit_i` | exist | `R/utils.R:232/244/255` |
| `mat2hmat_partial`, `mat2hmat_partial_from_sorted`, `compute_sorted_vec_direct`, `qs_nthreads_adaptive`, `FoReco2matrix`-as-local-helper | **ABSENT** at baseline | confirmed |
| `ctrml()` hat expansion | calls `mat2hmat(hat, h, kset, n)` BEFORE rml() | `R/ctrml.R:307` |
| `terml()` mfh path | calls `vec2hmat(hat, h_hat, kset)` (C.1 target) | `R/terml.R:216` |
| `terml()` non-mfh path | already slice-first via `input2rtw_partial(hat, kset, cols=keep_cols)` | `R/terml.R:257` |
| `csrml()` hat handling | **Does NOT expand hat** — passes hat directly to `rml(...)` at `R/csrml.R:262` and `R/csrml.R:382`. Cross-sectional only — no temporal kset expansion needed. → **T3 SKIPS csrml** (no work required). | `R/csrml.R:262,382` |
| Baseline NAMESPACE | exports: `csrml(_fit)`, `ctrml(_fit)`, `terml(_fit)`, `S3method(print,rml_fit)` | `NAMESPACE` |
| Baseline DESCRIPTION deps to audit | Imports/Suggests: must verify `lightgbm` version supports `lgb.train(init_model=)` (required by T7.4); must verify `qs2::qs_save(..., nthreads=)` is present in pinned qs2 version (required by T5/B7). | T1 records exact versions. |
| Test fixtures | (T1 measures) | TBD |

This inventory is load-bearing for T0-T11. Discrepancies vs. inventory found during execution → halt and re-plan.

---

## Lateral-Thinking SOTA Coverage Map

Dimensions audited (efficiency / speed / maintainability / mathematical correctness / SOTA / UX). v3 vs v4 deltas:

| Dimension | v3 coverage | v4 addition |
|---|---|---|
| Loop micro-ops (spd.2/3/8/15) | T2 | unchanged |
| Hat-expansion deferral (spd.12/13/14, C.1, B8) | T3 | unchanged |
| Vectorize utils (spd.16/17/18) | T4 | unchanged |
| qs2 nthreads | T5 (inline) | unchanged |
| Lazy NA cols (B10) | T5 | unchanged |
| Overflow guard | T5 | unchanged |
| ranger backend | T5 (opt-in) | **T5 makes ranger DEFAULT (user directive)** |
| catboost backend | T6 | unchanged |
| Global ML (H.1-5) | T7.1-T7.3 | unchanged |
| G.2 chunked | T7.4 | unchanged |
| Math correctness audit | absent | **T0.1 read-only audit (sel_mat, mat2hmat, input2rtw, FoReco2matrix, reconciliation algebra)** |
| Benchmark baseline | absent | **T0.2 `bench::mark()` on representative workloads** |
| series_id encoding SOTA | absent | **T7.2: categorical_feature/cat_features/enable_categorical per backend** |
| predict methods for new backends | implicit | **T5+T6 explicitly update `predict.rml_fit` dispatch** |
| predict + print + summary for rml_g_fit | absent | **T7.5 method dispatch** |
| Hierarchical-sum invariant tests | absent | **T8 reconciliation-invariant test suite** |
| Benchmark validation post-rebuild | absent | **T9 re-run benches; record speedup** |
| Documentation/vignette | absent | **T10 vignette + roxygen sweep** |
| Wrapper consolidation | implicit | **T7.2 internal `.stack_series()` helper for rml_g methods** |
| Seed reproducibility audit | absent | **T5 audit + test per backend** |

Items deliberately OUT OF SCOPE (require additional explicit ask):
- GPU support (lightgbm/catboost `device='gpu'`).
- Probabilistic / quantile-loss forecasts.
- `Matrix::sparseMatrix` support for `sel_mat` (baseline only supports `sparseVector`).
- Cross-validation framework on top of global models.
- `randomForest` deprecation (kept for back-compat; ranger is the new default).
- Distributed / multi-host training.

---

## Task T0.1: Mathematical Correctness Audit (READ-ONLY)

**Status:** READY (no dep)

### I. Objective
Read-only audit producing a report verifying mathematical correctness of baseline's reconciliation algebra. Findings feed downstream tasks (any bug discovered becomes a new ticket and gates T2+).

### II. Logic
1. Read `R/utils.R` `input2rtw`, `input2rtw_partial`, `mat2hmat`, `vec2hmat`, `sel_mat_keep_cols`.
2. Read each baseline `rml.<backend>` body in `R/reco_ml.R` (randomForest, xgboost, lightgbm, mlr3) for: row/column conventions, response coercion, NA handling at backend boundary, seed propagation contract.
3. Document each transformation: input shape → output shape; algebraic semantics.
4. Verify `FoReco::FoReco2matrix` external API usage matches FoReco documentation (WebFetch authoritative source). Note: `mat2hmat`/`vec2hmat` are INTERNAL FoRecoML functions (not FoReco exports) — flag in audit.
5. Validate reconciliation algebra: for sample hierarchical data, manually compute `sel_mat × hat` slicing and verify per-series feature matrix.
6. Audit numerical stability: condition numbers of typical hat matrices; overflow risk in matrix products; underflow in normalize_stack scale factors.
7. **v6 G11 fix — extended audit scope**: also audit T5/T6/T7.2-introduced methods POST-implementation (this part of T0.1 runs after T7.5): `.stack_series` row-binding correctness; `rml_g.<backend>` stacking/unstacking algebra; `predict.rml_g_fit` unstacking shape; backend categorical encoding paths.
8. Document findings in `tasks/math-audit.md` (read-only artifact; updated at two checkpoints: baseline audit pre-T2, extended audit post-T7.5).
9. Any discovered defect → file new bd ticket, gate further work until resolved.

### III. DoD
- [ ] Audit report committed.
- [ ] All transformations have documented shape semantics.
- [ ] Zero defects found OR all defects ticketed with severity.

---

## Task T0.2: Benchmark Baseline (`bench::mark()`)

**Status:** READY (depends T0.1, T1)

### I. Objective
Capture quantitative baseline performance numbers on representative workloads. Re-run after T11 to prove speedup.

### II. Logic
1. Add `benchmarks/run-baseline.R`. Three workloads:
   - small: `p=24, T_obs=100, hat 100×500`.
   - medium: `p=200, T_obs=200, hat 200×5000`.
   - large: `p=500, T_obs=72, hat 72×15000` (mimics user's real workload).
2. For each: time `ctrml`, `terml`, `csrml` end-to-end via `bench::mark()` with iterations≥10, gc=TRUE, memory profiling=TRUE.
3. Record: median wall time, peak RSS, allocations.
4. Output: `benchmarks/baseline-25be237.qs2` (frozen result) + markdown summary.
5. Commit: `chore(bench): capture baseline 25be237 performance numbers`.

### III. DoD
- [ ] 3 workloads × 3 wrappers = 9 measurements recorded.
- [ ] Frozen result file committed.

---

## Task T1: Branch + Baseline Inventory

**Status:** READY

### I. Objective
Create branch from `25be237`. Verify baseline test suite passes. Record inventory facts that T2-T7 depend on.

### II. Input
- Commit `25be237658d1a82b06069f82bb0d647ada266529` (verified to exist).

### III. Guards
- Boundary: READ-ONLY of baseline; branch creation only.
- No code edits in T1.

### IV. Logic
1. `git checkout -b refactor/clean-baseline 25be237`.
2. `Rscript -e 'devtools::load_all(); devtools::test()'` → record pass/fail counts → log `baseline_tests_pass`, `baseline_tests_fail`.
3. `grep -rn "mirai" R/ DESCRIPTION NAMESPACE` → must return 0 lines.
4. Verify rml() signature at line 1-15 has 11 args, no `kset`, no `n_workers`, no `pool`.
5. Verify `for (i in seq_len(p))` at `R/reco_ml.R:96`.
6. Verify `na.omit(X)` unconditional at `R/reco_ml.R:111`.
7. Verify `gc(verbose=FALSE)` only if `!is.null(checkpoint_dir)` at `R/reco_ml.R:163-165`.
8. `grep -nE "^[a-zA-Z_].*<- function" R/utils.R` → confirm presence of `input2rtw_partial`, `na_col_mask`, `estimate_peak_bytes`, `serialize_fit`; confirm ABSENCE of `mat2hmat_partial`, `qs_nthreads_adaptive`, `compute_sorted_vec_direct`.
9. Inspect `R/csrml.R:262` and `R/csrml.R:382` to confirm csrml passes hat directly to `rml(...)` without `mat2hmat`/`vec2hmat`/`input2rtw` expansion. Record finding.
10. Read `DESCRIPTION` Imports/Suggests. Resolve installed versions: `Rscript -e 'packageVersion("lightgbm"); packageVersion("qs2")'`. Verify (a) `lightgbm` ≥ version that supports `lgb.train(init_model=)` argument — check `?lightgbm::lgb.train` help text or release notes; (b) `qs2::qs_save` accepts `nthreads` arg via `Rscript -e 'args(qs2::qs_save)'`. If either fails: T7.4 / T5 must bump DESCRIPTION min version.
11. Forbidden-symbol grep gate: `grep -rnE "(mirai|n_workers|loop_body_kset|loop_body_csrml|resolve_n_workers|cap_inner_threads|promote_fit_to_checkpoint|b19_daemon_load|arrow_available|compute_chunk_size|shared_hat|ipc_hat)" R/ tests/ DESCRIPTION NAMESPACE man/` → MUST return zero. Record. This gate re-runs at every commit boundary in T2-T7.
12. Write findings to `bd remember "<inventory>"`.

### V. Output
```
branch: refactor/clean-baseline
baseline_sha: 25be237
baseline_tests_pass: <N>
baseline_tests_fail: <M>  # MUST be 0 to proceed
forbidden_symbol_hits: 0  # from step 11 grep
helpers_present: [input2rtw_partial, na_col_mask, estimate_peak_bytes, serialize_fit, deserialize_fit, get_fit_i, input2rtw, sel_mat_keep_cols, available_ram_bytes]
helpers_absent: [mat2hmat_partial, mat2hmat_partial_from_sorted, compute_sorted_vec_direct, qs_nthreads_adaptive]
csrml_hat_expansion: none  # confirmed direct pass-through
lightgbm_version: <X.Y.Z>
lightgbm_init_model_supported: true|false
qs2_version: <X.Y.Z>
qs2_nthreads_arg_supported: true|false
```

### VI. DoD
- [ ] Branch exists.
- [ ] `baseline_tests_fail == 0`.
- [ ] Inventory matches Verified Baseline State table; any mismatch halts.
- [ ] No commit made by T1.

---

## Task T2: Loop-Body Micro-Optimizations (spd.2, spd.3/8 gc throttle, spd.15 global_id hoist)

**Status:** READY (depends T1)

**Scope-rule:** T2 touches ONLY the rml() inner loop body. No changes to hat shape, no wrapper changes, no new helpers. This keeps T2 independent of T3.

### I. Objective
Three loop-body memory/CPU wins, all using existing lexical scope:
- spd.2: skip `na.omit` when `!anyNA(X)`.
- spd.3/8: throttle `gc()` to every 10th iteration.
- spd.15: hoist invariant `global_id` computation out of the loop.

### II. Input
- `R/reco_ml.R` rml() at `25be237`.

### III. Guards
- spd.2: behavior preserved when X has no NAs (no `attr(X,"na.action")` set → already takes no-strip branch, identical outcome).
- spd.3/8: existing `!is.null(checkpoint_dir)` guard preserved; AND-gate with `i %% 10L == 0L`. Final iteration `i == p` forces gc regardless of modulo to keep prior cleanup semantics (last-iter cleanup happens at function exit anyway, so this is optional — choose simple modulo, document).
- spd.15: hoist must produce a list of length `p`; per-iter access is O(1).
- No new helpers, no new params, no new file.

### IV. Logic
1. Before the `for` loop in `R/reco_ml.R`, compute:
   ```r
   global_id_list <- if (length(sel_mat) == 1) {
     rep(list(seq_len(active_ncol)), p)
   } else if (is(sel_mat, "sparseVector") || NCOL(sel_mat) == 1) {
     v <- which(as.numeric(if (is(sel_mat, "sparseVector")) sel_mat else sel_mat[, 1]) != 0)
     rep(list(v), p)
   } else {
     lapply(seq_len(p), function(j) which(sel_mat[, j] != 0))
   }
   ```
2. Inside the loop, replace the inline `global_id <- if (...) ...` block with `global_id <- global_id_list[[i]]`.
3. Replace `X <- na.omit(X); if (length(attr(X, "na.action")) > 0) { ... }` with:
   ```r
   if (anyNA(X)) {
     X <- na.omit(X)
     if (length(attr(X, "na.action")) > 0) { ... existing strip-y branch ... }
   }
   ```
4. Replace `if (!is.null(checkpoint_dir)) gc(verbose = FALSE)` with `if (!is.null(checkpoint_dir) && i %% 10L == 0L) gc(verbose = FALSE)`.
5. **v5 — Progress reporting (gate non-blocking gap)**: wrap the for-loop in `cli::cli_progress_along(seq_len(p), name = "Training models")` so users running p=500 get visible progress. Auto-disabled when not interactive (`getOption("forecoml.progress", interactive())`). Negligible runtime overhead.
6. `devtools::test()` → all baseline tests still pass.
7. Single commit: `perf(rml): hoist global_id; anyNA guard; gc throttle; cli progress (spd.2/3/8/15)`.

### V. Output
```
files_changed: [R/reco_ml.R]
loc_delta: ~+15/-7
tests_pass_delta: 0
commit: 1
```

### VI. DoD
- [ ] Three optimizations + cli progress applied.
- [ ] All baseline tests pass unchanged.
- [ ] cli progress active under interactive(); silent under tests.
- [ ] DESCRIPTION: `cli` already in Imports at baseline (verify in T1; if not, add).
- [ ] No new exports, no new helpers, no new files.
- [ ] Single commit.

---

## Task T3: Deferred hat Expansion + parts Hoist + terml mfh dim()<- (spd.12, spd.13, spd.14, C.1, B8)

**Status:** READY (depends T2)

**Scope-rule:** T3 changes hat shape contract between wrappers and rml(). It MUST add `kset = NULL` to rml(), MUST keep slice-first semantics, and MUST make parts hoist consistent with deferred expansion. **csrml is OUT OF SCOPE** for T3: per Verified Baseline State, csrml passes hat directly to rml() without expansion.

### I. Objective
- spd.13: ctrml() stops pre-expanding hat for BOTH non-mfh AND mfh paths. **v5 fix (gate G1)**: v4 only deferred non-mfh; user's documented OOM root cause was the 11-fold horizontal cbind in the mfh `mat2hmat(hat, h, kset, n)` call. v5 defers mfh as well.
- spd.12: rml() expands per-series slice inside the loop (smaller peak RSS).
- spd.14: hoist invariant per-fold parts decomposition outside the loop.
- C.1 + B8 (merged — same line, single edit): terml() mfh path replaces `vec2hmat(hat, h_hat, kset)` with in-place reshape via `dim(hat) <- c(1L, length(hat))` (zero new allocation); pass `h = h_hat`, `n = 1L`, `kset` through to rml(). **B8 is folded into T3 to avoid touching the same site twice (was T5/B8 in v2 plan).**

### II. Input
- `R/reco_ml.R` (post-T2).
- `R/ctrml.R`, `R/terml.R`.
- `R/utils.R` — uses existing `input2rtw_partial(x, kset, cols)`.

### III. Guards
- rml() signature: add `kset = NULL` as a named param (placed after `keep_cols`, before `checkpoint`).
- When `kset` is `NULL`: behavior IDENTICAL to current — caller passed pre-expanded hat. Defer-path is OPT-IN via non-NULL kset.
- ctrml() non-mfh path: stop calling `mat2hmat(hat, h, kset, n)`; instead pass raw hat + kset to rml().
- terml() mfh path: `hat <- matrix(hat, nrow = 1L)`; call rml(..., kset = kset).
- parts hoist (spd.14): computed lexically before loop in rml() ONLY when `!is.null(kset)`; loop body uses `parts_hat[[i]]`.
- YAGNI: do NOT introduce `input2rtw_partial_from_parts`. Inside loop use existing `input2rtw_partial(raw_hat, kset, cols = global_id_list[[i]])`. If profiling later shows duplicated per-iter parts decomposition is the bottleneck, add a helper THEN.

### IV. Logic
1. Add `kset = NULL` to `rml()` signature in `R/reco_ml.R`.
2. Inside rml(), when `!is.null(kset) && !is.null(hat)`:
   ```r
   # parts_hat[[i]] = per-series slice of expanded hat, computed lazily per iteration
   ```
   Inside loop, when `!is.null(kset)`:
   ```r
   X <- input2rtw_partial(hat, kset, cols = global_id_list[[i]])
   ```
   When `kset` is NULL (legacy callers): preserve existing `X <- hat[, id, drop = FALSE]` path.
3. **v6 fix G9 (base symmetric defer)**: baseline pre-expands BOTH `hat` AND `base` symmetrically. Verified sites:
   - ctrml non-mfh: `R/ctrml.R:307` (hat) + `R/ctrml.R:463` (base) — both `mat2hmat`.
   - ctrml mfh: `R/ctrml.R:608` (hat second mfh entry point) + symmetric base site.
   - terml non-mfh: `R/terml.R:257` (hat) + `R/terml.R:329` (base) — `input2rtw_partial` slice-first already.
   - terml mfh: `R/terml.R:216` (hat) + `R/terml.R:331` (base) — both `vec2hmat`.
   T3 defers ALL these sites symmetrically. In ctrml() non-mfh: pass raw `hat` AND raw `base` + `kset` to rml(). rml() loop expands per-iter for both via `input2rtw_partial(hat|base, kset, cols=global_id_list[[i]])`.
3b. **v6 fix G1 — ctrml() mfh (BOTH entry points at lines 307 AND 608)**: drop the eager `mat2hmat(hat|base, h, kset, n)` calls. Pass raw inputs + `kset + h + n` to rml(). New internal helper `mat2hmat_cols(x, h, kset, n, cols)` in `R/utils.R` materializes only the requested column range per-iter. YAGNI suspended — gap is user-visible OOM root cause. Helper authored vectorized from the start (T4's spd.17 sweep can't vectorize what doesn't exist yet at T4 dependency-time).
4. In terml() mfh path: replace `hat <- vec2hmat(vec = hat, h = h_hat, kset = kset)` AND `base <- vec2hmat(...)` with `dim(hat) <- c(1L, length(hat))` and `dim(base) <- c(1L, length(base))` respectively (in-place reshape — zero new allocation). Call `rml(..., kset = kset)`.
5. Equivalence test set (NEW, file `tests/testthat/test-defer-equivalence.R`):
   - (a) ctrml non-mfh: defer-path vs legacy-path, byte-identical.
   - (b) ctrml non-mfh with sparseVector `sel_mat`: byte-identical (exercises global_id_list sparseVector branch from T2).
   - (c) ctrml non-mfh with single-column `sel_mat` (NCOL==1): byte-identical.
   - (d) ctrml non-mfh with scalar `sel_mat` (length==1): byte-identical (exercises seq_len(active_ncol) branch).
   - (e) terml mfh: defer-path vs legacy-path, byte-identical (covers C.1 + B8).
   - (f) terml non-mfh: defer-path vs legacy-path, byte-identical.
   - (g) terml with `kset` length 1: defer-path matches legacy.
   - (h) csrml unchanged: smoke test verifying csrml output equals baseline output (regression guard).
   - (i) **v5 — ctrml mfh defer**: byte-identical between defer-path (raw hat + kset → rml() materializes per-iter) and legacy-path (caller pre-mat2hmat).
   - (j) **v5 — ctrml mfh memory**: peak RSS during ctrml mfh with `p=200, hat 200×5000` is at least 2× lower than baseline pre-mat2hmat. Verified via `bench::mark()` memory profiling.
   - (k) **v6 G9 — base defer symmetry**: ctrml + terml with non-NULL `base`, both mfh and non-mfh, defer-path vs legacy-path byte-identical (4 sub-cases).
6. `devtools::test()` → all pass.
7. Single commit: `perf(rml): defer hat expansion; per-iter slice; C.1 terml mfh dim<- reshape (spd.12/13/14, C.1, B8)`.

### V. Output
```
files_changed: [R/reco_ml.R, R/utils.R, R/ctrml.R, R/terml.R, tests/testthat/test-defer-equivalence.R]
loc_delta: ~+85/-20
new_tests: 11 (a-k enumerated above) — k expands to 4 sub-cases
commit: 1
```

### VI. DoD
- [ ] `kset = NULL` added to rml().
- [ ] ctrml + terml defer hat expansion for BOTH mfh AND non-mfh paths.
- [ ] terml mfh uses `dim(hat) <- c(1L, length(hat))` (in-place, NOT `matrix(hat, nrow=1L)`).
- [ ] ctrml mfh: new `mat2hmat_cols()` helper added to R/utils.R; per-iter materialization.
- [ ] 10 equivalence tests pass.
- [ ] csrml regression smoke test passes (unchanged behavior).
- [ ] All baseline tests still pass.
- [ ] Forbidden-symbol grep gate (T1 step 11) returns 0.

---

## Task T4: utils.R Algorithmic Wins (spd.16, spd.17, spd.18)

**Status:** READY (depends T1)

**Scope change vs prior plan**: Drop spd.20 and B6 from T4. spd.20 / B6 require `mat2hmat_partial_from_sorted` / `compute_sorted_vec_direct` — which DO NOT exist at baseline. Adding them is a from-scratch design, not a port. If profiling after T3 shows mat2hmat is the hot path, file a new ticket; do not speculatively add helpers. YAGNI.

### I. Objective
Three pure-vectorization wins in existing utils.R functions:
- spd.16: replace `outer` broadcast in `input2rtw_partial` with `rep`/`rep`.
- spd.17: replace `apply(block, 2, rep, each=k)` with `matrix(rep(block, each=k), ...)`.
- spd.18: replace `vapply(seq_len(NCOL(hat)), function(j) sum(is.na(hat[,j]))>=thresh*NROW(hat), logical(1))` in `na_col_mask` with `colSums(is.na(hat)) >= thresh * NROW(hat)`.

### II. Input
- `R/utils.R` (baseline).

### III. Guards
- Each replacement byte-identical on representative inputs (verify in equivalence test).
- No signature changes.

### IV. Logic
1. spd.16 in `input2rtw_partial`: identify `outer(...)` site, replace.
2. spd.17 wherever `apply(.,2,rep,each=k)` appears (search all R/).
3. spd.18 in `na_col_mask`.
4. Equivalence tests (NEW): byte-identical before/after for each function on small synthetic input.
5. `devtools::test()`.
6. Single commit: `perf(utils): vectorize input2rtw_partial; na_col_mask; rep broadcast (spd.16/17/18)`.

### V. Output
```
files_changed: [R/utils.R, tests/testthat/test-utils-vectorize.R]
loc_delta: ~+25/-15
new_tests: ~3
commit: 1
```

### VI. DoD
- [ ] Three vectorizations applied.
- [ ] Equivalence tests byte-identical.
- [ ] All baseline tests pass.

---

## Task T5: ranger DEFAULT backend + predict dispatch + robustness (B4 ranger, B7 qs2, B10 lazy NA, overflow, seed audit)

**Status:** READY (depends T3)

**Scope-rule v4**:
- T5/B4: ranger becomes DEFAULT (user directive: "ranger is better implementation than randomForest"). All three wrappers ctrml/terml/csrml switch `approach = "ranger"`. randomForest backend retained for back-compat with `lifecycle::deprecate_soft("2.0.0", "approach='randomForest'", details="ranger is faster and statistically equivalent")`.
- T5/B7: `qs2::qs_save(..., nthreads = min(parallel::detectCores(), 4L))` inlined at call site. No helper (YAGNI).
- T5/B8: in T3 already.
- T5 NEW: `predict.rml_fit` dispatcher updated to handle `rml.ranger` (and `rml.catboost` after T6). Add `predict.rml_<backend>` S3 methods.
- T5 NEW: seed reproducibility audit per backend. ranger/randomForest/lightgbm/xgboost each consume seed differently; verify each method respects the `seed` parameter.

### I. Objective
- B4: ranger as **DEFAULT** backend across ctrml/terml/csrml + soft-deprecate randomForest.
- B7: inline qs2 nthreads cap.
- B10: defer NA-column-mask materialization in rml().
- Overflow guard in `estimate_peak_bytes`.
- predict.rml_fit dispatch update.
- seed-reproducibility audit + per-backend test.

### II. Input
- Post-T3 state.

### III. Guards
- ranger: ABSENT from baseline DESCRIPTION entirely (verified — feasibility iter 2 note). T5 ADDS `ranger` to Imports (not promotes — it was nowhere at baseline). If absent at runtime: clean cli_abort with install hint.
- nthreads cap of 4 hard-coded inline.
- B10 lazy: only compute na_cols_list inside the branch that consumes it.
- Default approach CHANGES from `"randomForest"` → `"ranger"` in ctrml/terml/csrml. BREAKING change. NEWS.md prominent entry. DESCRIPTION Version bumped to 2.0.0 per semver.
- randomForest backend retained but warns via `lifecycle::deprecate_soft` when used.
- predict.rml_fit must dispatch to per-backend predict methods (S3) — verify ranger's `predict.ranger` returns same shape as randomForest.
- seed: each backend uses different RNG state; rml() must `set.seed(seed)` before calling each fit method.

### IV. Logic
1. Add `rml.ranger` S3 method to `R/reco_ml.R`. Wraps `ranger::ranger(num.threads = 1L, seed = seed, ...)` (num.threads=1 since single-process + per-series; ranger's threading helps only with large trees, not tiny per-series boosters).
2. Add `predict.rml_fit` master dispatcher + per-backend `predict.rml_ranger_fit` (and update existing predict.rml_<backend> for randomForest/xgboost/lightgbm/mlr3 if not already present at baseline). Audit baseline predict path first.
3. Change `approach = "randomForest"` → `approach = "ranger"` in ctrml/terml/csrml signatures. Add `lifecycle::deprecate_soft` in rml() if `approach == "randomForest"`.
4. Update `serialize_fit` in `R/utils.R`: `qs2::qs_save(model, path, nthreads = min(parallel::detectCores(), 4L))`.
5. In rml() loop, gate na_cols_list computation behind branch that uses it.
6. `estimate_peak_bytes` audit; verify all multiplications cast numeric. Update per-model multiplier for ranger (lower than randomForest's 5×; suggest 3× as starting point — calibrate against actual measurement on representative input).
7. Per-backend seed test: same seed → same fit + same prediction across two calls.
8. Tests (`tests/testthat/test-ranger.R`, `test-checkpoint.R`, `test-seed-repro.R`):
   - (a) ranger smoke: `ctrml(..., approach="ranger")` runs end-to-end.
   - (b) ranger vs randomForest: RMSE within 1.10× tolerance.
   - (c) ranger checkpoint round-trip: serialize → deserialize → predict equality.
   - (d) ranger install-hint abort (mocked package-absent path).
   - (e) overflow regression: hat with `NROW × NCOL > 2^31` → finite output.
   - (f) deprecate_soft fires when `approach="randomForest"` (catch via `expect_warning`).
   - (g) predict.rml_fit dispatches correctly to ranger/lightgbm/xgboost/mlr3/randomForest backends (5 backend × predict equality).
   - (h-k) seed reproducibility: ranger / randomForest / lightgbm / xgboost — same seed produces same fit + same predictions across 2 calls (4 tests).
9. Commits (atomic, separate):
   - `feat(rml): ranger backend + make ranger default (B4); deprecate_soft randomForest [BREAKING]`
   - `feat(rml): predict.rml_fit master dispatcher; predict.rml_<backend> per-backend methods`
   - `perf(checkpoint): qs2 nthreads cap inline (B7)`
   - `perf(rml): lazy na_cols_list (B10)`
   - `fix(utils): estimate_peak_bytes overflow audit; ranger multiplier`
   - `test(rml): seed reproducibility audit per backend`

### V. Output
```
files_changed: [R/reco_ml.R, R/utils.R, R/ctrml.R, R/terml.R, R/csrml.R, DESCRIPTION, NEWS.md, tests/testthat/test-ranger.R, tests/testthat/test-checkpoint.R, tests/testthat/test-predict-dispatch.R, tests/testthat/test-seed-repro.R]
new_tests: 11 (a-k)
commits: 6
```

### VI. DoD
- [ ] `rml.ranger` registered via `@exportS3Method`.
- [ ] **Default `approach="ranger"` in ctrml/terml/csrml** (BREAKING change).
- [ ] `lifecycle::deprecate_soft` warns when `approach="randomForest"` used.
- [ ] predict.rml_fit + per-backend predict methods cover all 5 backends.
- [ ] qs2 nthreads inlined.
- [ ] Lazy NA cols.
- [ ] Overflow audited; ranger multiplier set; regression test passes.
- [ ] DESCRIPTION: `ranger` Imports (promoted); `lifecycle` Imports.
- [ ] NEWS.md: prominent "BREAKING: default backend changed randomForest → ranger" entry.
- [ ] 11 new tests pass.
- [ ] Forbidden-symbol grep gate returns 0.

---

## Task T6: Catboost Per-Series Backend (I.1)

**Status:** READY (depends T1)

### I. Objective
Add `approach = "catboost"` as a first-class per-series backend via `rml.catboost`. Persist with `.cbm` checkpoint format.

### II. Input
- Post-T1.

### III. Guards
- catboost is Suggests-only; abort with install hint if missing.
- `serialize_fit.catboost`: use `catboost::catboost.save_model(model, file.path(dir, paste0("fit_", i, ".cbm")))`.
- `deserialize_fit.catboost`: use `catboost::catboost.load_model(path)`.
- S3 method dispatch on `class_base` (matches existing serialize_fit pattern).

### IV. Logic
1. Add `rml.catboost` S3 method.
2. Add `predict.rml_catboost_fit` (calls `catboost::catboost.predict`) wired into T5's `predict.rml_fit` master dispatcher.
3. Add `serialize_fit.catboost` (uses `catboost::catboost.save_model(model, file.path(dir, paste0("fit_", i, ".cbm")))`) and `deserialize_fit.catboost` (uses `catboost::catboost.load_model(path)`).
4. Register S3 methods via `@exportS3Method` roxygen tags; regenerate NAMESPACE.
5. Add ctrml/terml/csrml docs entry for `approach = "catboost"`.
6. NEWS.md: "Add `approach='catboost'` backend (opt-in)".
7. Tests (all gated by `skip_if_not_installed("catboost")`):
   - (a) catboost smoke: `ctrml(..., approach="catboost")` runs end-to-end.
   - (b) catboost serialize round-trip: serialize → deserialize → predict equality.
   - (c) catboost install-hint abort.
   - (d) catboost seed reproducibility.
   - (e) catboost predict dispatch via `predict.rml_fit`.
8. Single commit: `feat(rml): catboost per-series backend + predict dispatch (I.1)`.

### V. Output
```
files_changed: [R/reco_ml.R, R/utils.R, NAMESPACE, DESCRIPTION, NEWS.md, tests/testthat/test-catboost.R]
new_tests: 5
commit: 1
```

### VI. DoD
- [ ] `rml.catboost` + `predict.rml_catboost_fit` registered via `@exportS3Method`.
- [ ] `serialize_fit.catboost` and `deserialize_fit.catboost` registered.
- [ ] `.cbm` round-trip test passes when catboost installed.
- [ ] catboost wired into `predict.rml_fit` master dispatcher (T5).
- [ ] DESCRIPTION: `catboost` in Suggests.
- [ ] NEWS.md entry added.
- [ ] 5 tests pass when catboost installed; cleanly skipped when not.
- [ ] Forbidden-symbol grep gate returns 0.

---

## Task T7: Global ML Stack (split into 4 atomic sub-tasks)

**Status:** READY (depends T5, T6)

**Scope-rule:** "no bundling" per CLAUDE.md → T7 is split into T7.1-T7.4. Each is one ticket, one commit.

### T7.1: `normalize_stack()` + robscale-driven scale estimator (H.1, H.2)

#### I. Objective
Add `normalize_stack(X, method = c("zscore","robust"), scale_fn = "gmd")` utility that produces stacked-series normalization for global ML. Default `scale_fn = "gmd"`; support all robscale exports: `gmd, mad_scaled, qn, sn, iqr_scaled, sd_c4`.

#### II. Logic
1. Add `normalize_stack()` to `R/utils.R` with mandatory **zero-scale guard**: after computing scale `sigma <- .robscale_fn(name)(x)`, force `sigma[!is.finite(sigma) | sigma < .Machine$double.eps] <- 1`. Constant series (gmd=0, mad=0) get no normalization (sigma=1) instead of division-by-zero. Comment the guard with rationale.
2. Add `.robscale_fn(name)` private dispatcher returning the chosen robscale function (6 supported names → 6 mapped exports).
3. Tests:
   - (a) each scale_fn returns a positive scalar on known input.
   - (b) unknown name → cli_abort with allowed list.
   - (c) **v5 — zero-scale guard**: constant input series → normalize_stack does NOT produce NaN/Inf; returns unscaled values (sigma=1 applied).
4. NEWS.md: "Add `normalize_stack()` utility and `robscale` dependency for global-ML stacking normalization."
5. Commit: `feat(utils): normalize_stack + robscale scale estimators + zero-scale guard (H.1/H.2)`.

#### III. DoD
- [ ] `normalize_stack()` works for both `method` values, all 6 `scale_fn` options.
- [ ] Unknown scale_fn aborts with allowed list.
- [ ] Zero-scale guard prevents div-by-zero on constant series (test c).
- [ ] DESCRIPTION: `robscale` in Imports.
- [ ] NEWS.md entry added.
- [ ] Forbidden-symbol grep gate returns 0.

---

### T7.2: `rml_g` core + per-backend methods + SOTA categorical encoding + early stopping (H.3, H.4)

#### I. Objective
Add `rml_g` S3 generic + 5 methods (`rml_g.lightgbm`, `rml_g.xgboost`, `rml_g.ranger`, `rml_g.mlr3`, `rml_g.catboost`) trained on stacked series.

Critical SOTA additions:
1. series_id MUST be treated as CATEGORICAL by each backend (categorical_feature/cat_features/enable_categorical/factor). Treating as numeric imposes false ordering — well-known statistical error that hurts accuracy.
2. **v5 — Early stopping (gate G2)**: gradient-boosting backends (lightgbm, xgboost, catboost) support `early_stopping_rounds` + `valids`/`watchlist`/`eval_set`. Add `early_stopping_rounds` + optional `validation_split` (default 0.0 = disabled) parameters to rml_g. When enabled, hold out `validation_split` fraction of stacked rows; pass to backend's early-stopping API. Universal SOTA practice.
3. Feature importance: after training, store backend-native importance scores in fit object; expose via `summary.rml_g_fit` (T7.5).

#### II. Logic
1. Add `.stack_series(X_list, y_list, scale_fn, normalize_method, validation_split = 0, min_validation_rows = 10L, seed = NULL)` private helper in `R/utils.R`:
   - Stacks X_i row-bind with series_id column appended.
   - series_id encoded as `factor` with levels computed ONCE over the full population BEFORE any subsequent batching (G5 fix — frozen schema across batches in T7.4). **v7 — Deterministic level ordering**: `levels = sort(unique(as.character(series_id)))` — alphabetical sort guarantees reproducibility regardless of input order. Documented in @details.
   - **v7 — Min validation guard**: if `floor(validation_split * N_total) < min_validation_rows`, `cli_warn("Validation set would have only {n} rows (< {min_validation_rows}); disabling validation. Increase validation_split or reduce min_validation_rows.")` and set `valid_idx = integer(0)`. Default `min_validation_rows = 10L` is conservative; user can override.
   - **v6 G6 fix (lightgbm integer encoding)**: also stores `series_id_int <- as.integer(series_id_factor)` (1-indexed). lightgbm methods consume the integer column; xgboost/catboost/ranger/mlr3 consume the factor. Both columns are in the returned stacked matrix; backend method selects.
   - **v6 G4 fix (GLOBAL validation holdout)**: when `validation_split > 0`, compute `valid_idx <- sample(seq_len(N_total), floor(validation_split * N_total))` ONCE over the full stacked dataset (using `seed`). Returns `train_idx` and `valid_idx` indices. ALL subsequent batches (in T7.4) split their slice of training rows but evaluate against the SAME globally-held-out validation rows. No leakage across batches; consistent early-stopping signal.
   - Returns list: `X_stacked`, `y_stacked`, `series_id_factor`, `series_id_int`, `series_id_levels`, `norm_params`, `train_idx`, `valid_idx`.
   - Used by ALL 5 backend methods → genuine DRY (5 occurrences identical pattern).
2. Add `rml_g` generic + 5 methods. Each method:
   - Calls `.stack_series()` with `validation_split` and `seed`. Validation indices are GLOBAL (G4 fix).
   - Backend-specific encoding + early stopping:
     - `rml_g.lightgbm`: column `series_id_int` (G6 fix — integer, not factor); pass `categorical_feature = "series_id_int"` (or positional index) to `lgb.Dataset`. `valids = list(valid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=...))` constructed from GLOBAL `valid_idx`. `early_stopping_rounds` passed to `lgb.train`.
     - `rml_g.catboost`: column `series_id_factor`; pass `cat_features = "series_id"` to `catboost.load_pool`. `eval_set` Pool constructed from GLOBAL `valid_idx`. Early stopping translated: user `early_stopping_rounds=N` → `params$od_type='Iter'; params$od_wait=N`.
     - `rml_g.xgboost`: build DMatrix with `enable_categorical = TRUE` (xgboost ≥1.5); factor column `series_id_factor` preserved. `evals = list(valid = xgb_dvalid)` from GLOBAL `valid_idx`. **v6 G5 fix (frozen schema)**: DMatrix construction must use `levels(series_id_factor)` from `.stack_series` (frozen population-wide levels), not per-batch factor recomputation. Pass `early_stopping_rounds` to `xgb.train`.
     - `rml_g.ranger`: factor `series_id_factor` handled natively. **Ranger importance default**: set `importance = "impurity"` when constructing `ranger::ranger()` call so `ranger::importance(fit)` returns non-NULL. No early stopping (RF non-iterative); param emits `cli_inform` notice.
     - `rml_g.mlr3`: factor handled natively via task feature roles. Early stopping via mlr3tuning if `early_stopping_rounds > 0`; else fit once.
   - Store `feature_importance` in returned fit object via backend-native call: `ranger::importance(fit)` (now populated by importance='impurity'), `lgb.importance(fit)`, `xgb.importance(model=fit)`, `catboost::catboost.get_feature_importance(fit)`, mlr3 `fit$importance()` if learner supports.
3. `@export` for the generic; `@exportS3Method rml_g <backend>` for each method.
4. Tests:
   - (a-e) smoke + prediction round-trip per backend (5 backends, gated by `skip_if_not_installed`).
   - (f) categorical encoding test: train rml_g.lightgbm with categorical_feature; verify model parameters reflect the setting.
   - (g) numeric-vs-categorical comparison: categorical produces lower held-out RMSE than numeric series_id.
   - (h) `.stack_series` unit test: stacks correctly across varying p, T_obs.
   - (i) early stopping fires when `early_stopping_rounds=10, validation_split=0.2`: trained model has fewer iterations than `nrounds` when validation loss plateaus. Backends: lightgbm + xgboost + catboost.
   - (j) ranger early stopping no-op: passing `early_stopping_rounds > 0` to rml_g.ranger emits `cli_inform` and proceeds.
   - (k) feature importance stored: each backend's fit object has non-empty `$feature_importance` field.
   - (l) **v6 G4 — global holdout**: with `validation_split=0.2, seed=42`, the SAME row indices are held out regardless of how data is sliced/permuted for batching. Verify `valid_idx` is identical across two calls.
   - (m) **v6 G5 — xgboost frozen schema**: train rml_g.xgboost on dataset with `p=10` series, then incrementally on subset with only `p=5` series (subset of original levels). Verify (i) no error, (ii) factor levels in batch-2 DMatrix match population-wide levels from batch 1 (`length(levels) == 10` not 5).
   - (n) **v6 G6 — lightgbm integer encoding**: verify the column passed to lgb.Dataset is `is.integer()` (not factor); verify `categorical_feature` index resolves to the integer column.
5. Commit: `feat(rml): rml_g global ML with SOTA categorical encoding + early stopping + feature importance + global holdout + schema invariance (H.3/H.4)`.

#### III. DoD
- [ ] All 5 methods registered in NAMESPACE.
- [ ] `.stack_series` private helper used by all 5 methods (DRY ≥3 occurrences proven).
- [ ] `.stack_series` produces frozen `series_id_levels` AND both `series_id_factor` + `series_id_int` columns.
- [ ] `.stack_series` computes `train_idx`/`valid_idx` GLOBALLY (G4).
- [ ] series_id passed as CATEGORICAL to each backend (lightgbm: integer; others: factor).
- [ ] xgboost DMatrix uses frozen levels across all batches (G5).
- [ ] lightgbm receives integer-encoded series_id_int (G6).
- [ ] Numeric-vs-categorical comparison test confirms accuracy gain.
- [ ] Early stopping works for lightgbm/xgboost/catboost; ranger emits cli_inform; mlr3 via tuner.
- [ ] ranger constructed with `importance="impurity"` so feature_importance is non-NULL.
- [ ] feature_importance stored in fit object for every backend.
- [ ] 14 tests pass for available backends.
- [ ] Forbidden-symbol grep gate returns 0.

---

### T7.3: `ctrml_g/terml_g/csrml_g` user wrappers + @param (H.5)

#### I. Objective
User-facing wrappers that:
- Stack series via `normalize_stack`.
- Call `rml_g`.
- Reconcile output.
- Accept `normalize = c("none","zscore","robust")` and `scale_fn` parameter (passed to `normalize_stack`).

#### II. Logic
1. `ctrml_g`, `terml_g`, `csrml_g` in respective files.
2. Roxygen `@param normalize`, `@param scale_fn`; full docs for each new wrapper.
3. `@export` each wrapper; regenerate NAMESPACE via `devtools::document()`.
4. Tests: equivalence — `ctrml_g(..., normalize="none")` matches stacking-by-hand baseline output; one equivalence test per wrapper (3 total).
5. NEWS.md: "Add `ctrml_g()`, `terml_g()`, `csrml_g()` global-ML wrappers with `normalize` and `scale_fn` parameters."
6. Commit: `feat(global): ctrml_g/terml_g/csrml_g wrappers (H.5)`.

#### III. DoD
- [ ] Three wrappers exported (NAMESPACE).
- [ ] 3 equivalence tests pass.
- [ ] NEWS.md entry added.
- [ ] Forbidden-symbol grep gate returns 0.

---

### T7.4: Chunked incremental training (G.2)

#### I. Objective
Add `batch_size` parameter to `ctrml_g/terml_g/csrml_g`. Default `batch_size = "auto"` computed from `available_ram_bytes()`. Implements warm-start incremental training via `train_incremental_global.<backend>`.

#### II. Guards
- **v5 fix B1 (xgboost API name)**: xgboost warm-start uses `xgb.train(params, data, nrounds, xgb_model = prev_booster, ...)`. Parameter is `xgb_model=`, NOT `init_model`. Analyst confirmed v4 plan referenced wrong param. Verified via `args(xgboost::xgb.train)` locally.
- **v5 fix B2 (catboost has NO warm-start in R)**: verified against catboost R docs — `catboost.train` has no `init_model`/`init_score`/`continue_from` param. `baseline` parameter in `catboost.load_pool` sets initial RAW SCORES (not model continuation). **catboost is REMOVED from chunked path**. catboost_g always trains single-batch; `batch_size != p` with `approach="catboost"` aborts with `cli_abort` directing user to chunk via lightgbm/xgboost.
- `lightgbm`: `lgb.train(init_model = model, ..., free_raw_data = FALSE)` — `free_raw_data=TRUE` crashes on warm-start (analyst-verified).
- `xgboost`: `xgb.train(xgb_model = prev_booster, ...)`. Carry prev booster across batches.
- `series_id` append: done ONCE inside `train_incremental_global.*`, NOT in caller (avoids double-append).
- Auto formula: `batch_size = max(1L, floor(0.5 * available_ram_bytes() / per_series_bytes))`. Guard with `as.numeric()` casts; integer overflow risk on `T_obs * NCOL(stacked_X) * 8L`.
- `T_obs` calc per scope: ctrml_g uses `NCOL(obs)`, terml_g uses `length(obs)/nb`, csrml_g uses `NROW(obs)`.
- Per-batch early stopping: each batch trains with `nrounds_per_batch` (default 50); user-facing `early_stopping_rounds` applies WITHIN each batch. Validation set is GLOBAL (T7.2 G4); same `valid_idx` rows used as eval across every batch.
- **v6 G10 — `per_series_bytes` formula defined**: `per_series_bytes <- as.numeric(T_obs) * as.numeric(NCOL(stacked_X)) * 8 + as.numeric(NCOL(stacked_X)) * 64`. First term = stacked design-matrix row bytes for one series (double precision); second term = ~64 bytes per-column overhead (R header + factor levels + cache locality). `as.numeric()` mandatory to prevent integer overflow at user scale. `T_obs` per scope: ctrml_g `NCOL(obs)`, terml_g `length(obs)/nb`, csrml_g `NROW(obs)`.
- **v7 — Deterministic chunking**: new param `chunk_strategy = c("sequential", "random")`. Default `sequential` partitions `seq_len(p)` into adjacent blocks (reproducible without seed). `random` permutes via `set.seed(seed); sample(seq_len(p))` then blocks (reproducible given seed). Document chunk_strategy in @param; record actual partition in `fit$batch_indices` for user introspection.
- **v7 — OOM fallback**: wrap each backend fit call in `tryCatch` for `bad_alloc` / `std::bad_alloc` / `cudaErrorMemoryAllocation`-class errors. On catch: halve `batch_size` (`max(1L, batch_size %/% 2L)`), restart current batch with smaller slice. Log retry via `cli_alert_warning`. Hard cap of 3 halvings; beyond that, propagate the error.
- **v7 — Checkpoint + resume**: new param `batch_checkpoint_dir = NULL`. When non-NULL: after each batch, serialize the carry-state booster via `qs2::qs_save(list(booster=current_booster, batch_idx=k, best_iter_history=hist), file.path(dir, "batch_state.qs2"))`. On a fresh call with the same `batch_checkpoint_dir`, detect existing state, resume from `batch_idx + 1`. Long-job crash resilience for user-scale workloads.
- **v7 — best_iter_history**: fit object exposes `$best_iter_history` (length = n_batches; each entry is the early-stopped iteration count for that batch). NULL when early stopping disabled. Surfaced via `print.rml_g_fit` and `summary.rml_g_fit` (T7.5).

#### III. Logic
1. Add `train_incremental_global.<backend>` methods for **lightgbm + xgboost only** (the two backends with documented R-side warm-start API). ranger/mlr3 fall back to single-batch (no incremental boosting concept). catboost ABORTS with explicit error when `batch_size != p` directing user to lightgbm/xgboost.
2. Wire `batch_size` into ctrml_g/terml_g/csrml_g.
3. Tests (`tests/testthat/test-g2-chunking.R`):
   - (a) lightgbm: `batch_size = p` byte-identical to baseline single-batch.
   - (b) lightgbm: `batch_size = "auto"` resolves to numeric > 0; if `auto >= p` produces identical output to (a).
   - (c) lightgbm quality: RMSE(`batch_size = "auto"` multi-batch) ≤ 1.05 × RMSE(`batch_size = p`).
   - (d) **v5 — xgboost incremental**: `xgb_model=` warm-start; batch_size=p byte-identical; quality test for multi-batch.
   - (e) lightgbm regression guard: `free_raw_data=FALSE` enforced; explicit warm-start chain that would crash if `free_raw_data=TRUE` succeeds.
   - (f) **v5 — catboost chunk-abort**: `csrml_g(..., approach="catboost", batch_size=10)` with `p=100` returns `cli_abort` with message naming lightgbm/xgboost as alternatives. Test catches the abort.
   - (g) **v6 G7 — long warm-start chain drift**: force `batch_size = p/10` producing 10+ batches; train lightgbm chain and xgboost chain; RMSE(10-batch chain) ≤ 1.10× RMSE(single batch). Verifies accumulated gradient drift is bounded.
   - (h) **v6 G10 — overflow on `per_series_bytes`**: synthetic dims `T_obs=10000, NCOL=20000` produce numeric (not integer) byte count; auto resolves to positive batch_size.
   - (i) **v7 — checkpoint+resume**: simulate crash after batch 5 of 10 (interrupt + delete in-memory state); resume from same `batch_checkpoint_dir`; final model byte-identical to no-crash run.
   - (j) **v7 — deterministic chunking**: `chunk_strategy="sequential"` produces same `fit$batch_indices` across two runs (no seed needed). `chunk_strategy="random", seed=42` reproducible.
   - (k) **v7 — OOM fallback**: mock backend to throw `bad_alloc` on first call, succeed on second. Verify batch_size halved + cli_alert + train continues.
   - (l) **v7 — best_iter_history**: after 10-batch chain with early_stopping_rounds=10, `fit$best_iter_history` is length 10 with all entries ≤ nrounds_per_batch.
   - (m) **v7 — min_validation_rows guard**: `validation_split=0.001, N_total=200` → cli_warn + valid_idx empty + early stopping silently disabled.
   - (n) **v7 — deterministic factor levels**: input series_id passed in different orders ("A,C,B" vs "B,A,C") produces identical `series_id_levels` ordering (alphabetical).
   - 14 tests total.
4. NEWS.md: "Add `batch_size` parameter to `ctrml_g/terml_g/csrml_g` for chunked incremental training (lightgbm + xgboost). catboost not supported for chunked path (catboost R API lacks warm-start)."
5. Commit: `feat(global): chunked incremental training; batch_size auto; lightgbm+xgboost only (G.2)`.

#### IV. DoD
- [ ] `batch_size`, `chunk_strategy`, `batch_checkpoint_dir`, `seed` parameters present in 3 wrappers.
- [ ] `per_series_bytes` formula spelled out in code with `as.numeric()` casts (overflow-safe).
- [ ] 14 G.2 tests pass.
- [ ] `free_raw_data=FALSE` for lightgbm — enforced by test (e).
- [ ] `xgb_model=` (NOT `init_model`) used for xgboost warm-start.
- [ ] catboost chunked path aborts cleanly with helpful message.
- [ ] 10+ batch warm-start chain produces RMSE ≤ 1.10× single-batch (test g).
- [ ] Checkpoint + resume round-trip is byte-identical (test i).
- [ ] Deterministic chunking under both strategies (test j).
- [ ] OOM fallback halves and retries up to 3 times (test k).
- [ ] `best_iter_history` populated per batch (test l).
- [ ] min_validation_rows guard warns + disables (test m).
- [ ] Factor levels alphabetically deterministic (test n).
- [ ] Validation set is GLOBAL across all batches (from T7.2 `.stack_series`).
- [ ] DESCRIPTION min versions confirmed against T1 inventory.
- [ ] NEWS.md entry added.
- [ ] Forbidden-symbol grep gate returns 0.

---

### T7.5: predict / print / summary methods for `rml_g_fit` (SOTA UX)

**Status:** READY (depends T7.2, T7.4)

#### I. Objective
Match S3 contract of `rml_fit` for the global-ML object: `predict.rml_g_fit`, `print.rml_g_fit`, `summary.rml_g_fit`.

#### II. Logic
1. Add `predict.rml_g_fit(object, newdata, ...)` that:
   - Stacks `newdata` using stored `series_id_levels` from `.stack_series`.
   - **v6 G8 fix**: when newdata contains a `series_id` NOT in `object$series_id_levels`, `cli_abort("Unknown series_id in newdata: {.val {missing}}. Known levels: {.val {head(object$series_id_levels, 10)}}{if (length(object$series_id_levels) > 10) ' ...' else ''}.")`. Strict failure mode — never silently impute or coerce to NA. Documented in @param.
   - Calls backend predict; unstacks per-series.
   - Returns per-series predictions matching shape of `predict.rml_fit`.
2. Add `print.rml_g_fit`: compact summary (backend, p, T_obs, scale_fn, norm_method, **n_batches + best_iter_history range when present**).
3. Add `summary.rml_g_fit`: per-series in-sample fit metrics (RMSE, MAE), aggregate metrics, model size on disk if checkpointed, **and top-N feature importance** from the stored `$feature_importance` (set in T7.2). **v6 fix**: if `$feature_importance` is NULL/empty, summary prints `cli_inform("Feature importance not available for this backend/configuration.")` and proceeds.
4. `@exportS3Method` for each.
5. Tests:
   - (a) predict.rml_g_fit returns correct shape; equality with manual unstack.
   - (b) predict matches per-series rml_fit predict shape (cross-paradigm consistency).
   - (c) print/summary produce non-empty output, no errors.
   - (d) **v6 G8 — missing series_id level**: train rml_g on 5 series; predict on newdata containing a 6th unseen series_id → cli_abort with helpful message.
   - (e) **v6 — summary handles missing importance**: fit object with `feature_importance=NULL` → summary still runs, emits cli_inform.
6. Commit: `feat(rml_g): predict/print/summary methods + strict unseen-level handling`.

#### III. DoD
- [ ] 3 S3 methods registered + exported.
- [ ] predict shape matches `predict.rml_fit` (consistency).
- [ ] Strict cli_abort on unseen series_id (G8).
- [ ] Summary handles missing feature_importance gracefully.
- [ ] 5 tests pass.
- [ ] NEWS.md entry: "S3 print/summary/predict methods for rml_g_fit; strict unseen series_id handling."
- [ ] Forbidden-symbol grep gate returns 0.

---

## Task T8: Reconciliation Invariant Tests (mathematical correctness)

**Status:** READY (depends T5, T6, T7.3, T7.5)

### I. Objective
End-to-end mathematical correctness: post-reconciliation forecasts MUST satisfy hierarchical aggregation constraints. Sum of children = parent. Currently no tests enforce this invariant.

### II. Logic
1. New test file `tests/testthat/test-reconciliation-invariants.R`.
2. For each wrapper (ctrml, terml, csrml, ctrml_g, terml_g, csrml_g):
   - Construct toy hierarchy with known aggregation matrix.
   - Run reconciliation.
   - Assert: for every parent series, `sum(reconciled[children]) == reconciled[parent]` within numerical tolerance (1e-10).
3. For cross-temporal (ctrml/ctrml_g): assert both cross-sectional AND temporal aggregation constraints.
4. Total: 6 invariant tests + 2 cross-paradigm tests (ctrml: 1 cross-sec + 1 temporal) = 8.
5. If any wrapper fails: gate halts; file bug ticket; fix BEFORE T9 benchmark.
6. Commit: `test(reco): hierarchical aggregation invariant suite`.

### III. DoD
- [ ] 8 invariant tests pass with tol ≤ 1e-10.
- [ ] Any tolerance violation halts and is ticketed.
- [ ] Forbidden-symbol grep gate returns 0.

---

## Task T9: Benchmark Re-run vs T0.2 Baseline (proven speedup)

**Status:** READY (depends T8)

**v5 fix (gate non-blocking)**: split into two benchmark passes to isolate refactor speedup from algorithm swap.

### I. Objective
Quantitative validation. Two passes:
- **Pass A (apples-to-apples)**: Force `approach="randomForest"` on refactored branch; compare to T0.2 baseline (also randomForest). Isolates the refactor's effect on wall time / peak RSS / allocations.
- **Pass B (user-experience)**: Use default (now `approach="ranger"`); compare to T0.2 baseline (randomForest). Captures user-facing perf, includes algorithm swap.

### II. Logic
1. Re-run `benchmarks/run-baseline.R` on the refactored branch twice: once with `force_approach="randomForest"`, once with default.
2. Produce `benchmarks/speedup-summary.md` with both passes' tables.
3. SLAs:
   - Pass A per workload: wall_time ≤ 1.00× baseline + peak_rss ≤ 1.00× baseline (refactor must NOT regress anything).
   - Pass B per workload: SLA ≤ 1.05× baseline (algorithm swap acceptable within 5%; expect speedup on most workloads since ranger is generally faster).
4. NEWS.md "Performance" section: paste both tables.
5. Commit: `chore(bench): post-refactor speedup validation (refactor-only + default paths)`.

### III. DoD
- [ ] All 9 workloads × 2 passes = 18 measurements.
- [ ] Pass A: zero regressions vs baseline.
- [ ] Pass B: ≤1.05× baseline acceptable.
- [ ] Both speedup tables committed.
- [ ] NEWS.md "Performance" section updated.

---

## Task T10: Documentation Sweep + Vignette

**Status:** READY (depends T9)

### I. Objective
Per CLAUDE.md §4 "Update all docs in SAME commit as code changes" — sweep at end since v4 plan adds many user-visible features. Single coherent doc pass keeps docs consistent.

### II. Logic
1. `roxygen2::roxygenise()` regenerate all man/ pages from current source.
2. Audit each new function's @param, @return, @examples for completeness.
3. Add or update `vignettes/forecoml.Rmd`:
   - Migration note: randomForest → ranger default.
   - Choosing per-series vs global ML.
   - Normalization options & robscale scale_fn choices (zero-scale guard explained).
   - Chunked incremental training (batch_size).
   - **v6 G12 — catboost-chunked caveat**: vignette explicitly notes catboost R API has no warm-start, so chunked path is unsupported for catboost; users must pick lightgbm or xgboost for incremental training.
   - Reconciliation math: sel_mat semantics, agg_mat contract (referenced from T0.1 audit).
   - Per-backend tuning notes: when to use which backend; categorical encoding behavior.
4. README.md: update install instructions, add quick-start example using ranger default.
5. pkgdown rebuild (if configured).
6. Commit: `docs: vignette + roxygen sweep for v2.0 features`.

### III. DoD
- [ ] `devtools::check()` reports no doc warnings.
- [ ] Vignette renders without error.
- [ ] README quickstart works (manual smoke-run).
- [ ] All new functions have @examples block.
- [ ] Forbidden-symbol grep gate returns 0.

---

## Cross-task: Final Verification (gated as a real task, not free-text)

After T7.4 completes, perform ALL steps below. Each is a DoD checkbox.

### Logic
1. `devtools::document()` runs cleanly (no diagnostics).
2. `devtools::check()` 0 errors, 0 warnings (Notes acceptable).
3. Forbidden-symbol grep gate one last time: `grep -rnE "(mirai|n_workers|loop_body_kset|loop_body_csrml|resolve_n_workers|cap_inner_threads|promote_fit_to_checkpoint|b19_daemon_load|arrow_available|compute_chunk_size|shared_hat|ipc_hat)" R/ tests/ DESCRIPTION NAMESPACE man/` → 0 hits.
4. `git diff 25be237..HEAD --stat` — record total LOC delta and files changed. No upper bound (v4 dropped "minimal divergence" mandate); the diff is "what SOTA implementation requires." Record for audit only.
5. DESCRIPTION `Version:` field bump from baseline `1.0.0.9000` → **`2.0.0`** (semver MAJOR, since T5/B4 changes default backend = breaking).
6. NEWS.md: confirm structured entries:
   - **BREAKING**: default backend `randomForest` → `ranger` (T5).
   - **New backends**: ranger (now default), catboost (opt-in).
   - **New functions**: ctrml_g, terml_g, csrml_g, normalize_stack.
   - **New parameters**: batch_size in _g wrappers; normalize, scale_fn.
   - **Performance**: headline speedup numbers from T9.
   - **Deprecated**: randomForest backend emits soft deprecation warning.
7. Run full test suite: `Rscript -e 'devtools::test()'` → all pass; record final count = baseline_tests + ~50 new tests (8 T3 + 11 T5 + 5 T6 + 3 T7.1 + 8 T7.2 + 3 T7.3 + 5 T7.4 + 3 T7.5 + 8 T8 = 54 new).
8. Session-close protocol (per project CLAUDE.md):
   ```bash
   git status                          # MUST show clean tree
   git pull --rebase                   # ensure synced
   bd dolt push                        # sync beads tickets
   git push                            # push code
   git status                          # MUST show "up to date with origin"
   ```

### DoD (all required)
- [ ] `devtools::document()` clean.
- [ ] `devtools::check()` 0E/0W.
- [ ] Forbidden-symbol grep returns 0 hits.
- [ ] `git diff 25be237..HEAD --stat` recorded for audit.
- [ ] DESCRIPTION `Version:` bumped to `2.0.0`.
- [ ] NEWS.md has all required entries including BREAKING section.
- [ ] Test suite passes; final count documented (~baseline+54).
- [ ] T9 benchmark speedup table linked in NEWS.md.
- [ ] T10 vignette renders.
- [ ] `bd dolt push` succeeded.
- [ ] `git push` succeeded; `git status` shows up-to-date with origin.

## Success Criteria (Epic)

- All 7 task groups complete.
- `mirai` does not appear in DESCRIPTION, R/, NAMESPACE, or tests.
- `rml()` retains baseline lexical for-loop structure.
- No `loop_body_*` explicit-formal closures.
- Test count = baseline_tests + (T2..T7.4 new tests). No subtraction (baseline at 25be237 has no parallel tests to subtract).
- Final code is SIMPLER than mirai-era code at every comparable point.

## Risks

- **R1**: T3 ordering — must come AFTER T2; T3 introduces `kset` and shape contract change; T2's spd.15 hoist is signature-stable.
- **R2 (intentional, not mitigated)**: T5 ranger default is BREAKING. Mitigation = (a) prominent NEWS.md entry, (b) `lifecycle::deprecate_soft` keeps randomForest functional with warning, (c) DESCRIPTION Version 2.0.0 signals major bump. Per v4 user directive, breaking change is accepted.
- **R3**: T6 catboost not on CRAN — install path varies. Mitigation: Suggests-only; `skip_if_not_installed` gates.
- **R4**: T7.4 auto batch_size formula must NOT overflow on user-scale inputs. `as.numeric()` casts mandatory.
- **R5**: T7.2 series_id categorical encoding has different APIs per backend (categorical_feature/cat_features/enable_categorical). Mitigation: per-backend test (T7.2 test (f)+(g)) validates the encoding actually fires.
- **R6**: T9 benchmark might reveal regression on a workload not anticipated. Mitigation: T8 invariant tests + T9 SLA ≤1.05× baseline. If regression: gate halts; file fix ticket.
- **R7**: T0.1 math audit might surface latent bug in baseline. Mitigation: if found, ticket and fix BEFORE T2-T11 proceed.
- **R8 (analyst-confirmed)**: xgboost warm-start in v4 plan used wrong API name (`init_model` vs correct `xgb_model`). v5 corrects. Future API changes in xgboost could break again; mitigation: regression test (T7.4 (d)) trains a chain of 3+ batches and verifies prediction stability — would catch silent breakage.
- **R9 (analyst-confirmed)**: catboost R API has no model-continuation warm-start. v5 explicitly excludes catboost from chunked path with helpful abort. Future catboost upstream addition of warm-start → new ticket; not v5 scope.

## Changes from v1 → v2

| v1 | v2 |
|---|---|
| T1: branch + record test count | T1: branch + full inventory; uses inventory table to gate T2-T7 |
| T2 included spd.14 parts hoist | spd.14 moved into T3 (hat-shape concept inseparable from defer) |
| T3 vague on `input2rtw_partial_from_parts` | T3 uses EXISTING `input2rtw_partial` — YAGNI; no new helper |
| T4 included spd.20 + B6 | dropped (depended on non-existent helpers) |
| T7 single ticket | split into T7.1-T7.4, each atomic |
| Forbidden list "do not introduce X" implied | explicit "MUST NOT be re-introduced" |
| Mechanism: "lexical replaces explicit dispatch" | "preserve baseline lexical; never add explicit closures" |
| Test count formula included subtraction | corrected: no subtraction at 25be237 |
| Inventory implied | explicit Verified Baseline State table with file:line refs |

## Changes from v6 → v7 (gate iter 3 — 6 new SOTA gaps from "new angles" scrutiny)

| Gap | Fix |
|---|---|
| Per-batch checkpoint+resume for long jobs | T7.4 `batch_checkpoint_dir` param + per-batch qs2 save + auto-detect-resume protocol. Test (i) crash-resume byte-identical. |
| Deterministic batch ordering | T7.4 `chunk_strategy = c("sequential", "random")` + `seed`; record `fit$batch_indices`. Test (j) reproducibility. |
| OOM fallback (bad_alloc retry) | T7.4 tryCatch around fit calls; halve batch_size up to 3 times. Test (k) mock-throw verifies retry. |
| Per-batch best_iter_history | T7.4 fit stores `$best_iter_history`; T7.5 print/summary surface it. Test (l). |
| Min validation-rows guard | T7.2 `.stack_series` min_validation_rows=10 default; cli_warn + disable when below. Test (m). |
| Deterministic factor-level ordering | T7.2 `.stack_series` uses `levels = sort(unique(as.character(...)))`. Test (n). |

## Changes from v5 → v6 (gate iter 2 fixes)

| Gate gap | Fix |
|---|---|
| **G4** per-batch validation leakage | T7.2 `.stack_series` computes `train_idx`/`valid_idx` GLOBALLY ONCE; all batches in T7.4 reuse same `valid_idx`. New test (l). |
| **G5** xgboost frozen schema across batches | T7.2 `.stack_series` computes population-wide `series_id_levels` ONCE; xgboost DMatrix in every batch uses frozen levels. New test (m). |
| **G6** lightgbm needs integer-encoded categorical | T7.2 `.stack_series` returns BOTH `series_id_factor` AND `series_id_int`; lightgbm consumes integer. New test (n). |
| **G7** drift test ≥10 batches | T7.4 new test (g) forces `batch_size = p/10` for 10+ batches; bounds RMSE ≤1.10× single batch. |
| **G8** predict.rml_g_fit missing-level handling | T7.5 strict `cli_abort` with helpful message listing known levels. New test (d). |
| **G9** `base` defer symmetric with `hat` | T3 step 3 explicitly enumerates 4 baseline sites (ctrml:307/463/608 + terml:216/257/329/331); defers base symmetrically. New test (k) — 4 sub-cases. |
| **G10** `per_series_bytes` formula undefined | T7.4 guards define formula exactly: `T_obs × NCOL(stacked_X) × 8 + NCOL(stacked_X) × 64`. All `as.numeric()` casts mandatory. New test (h). |
| G11 (non-blocking) T0.1 audit scope | T0.1 extended: post-T7.5 audit pass covers `.stack_series`, `rml_g.<backend>` algebra, predict unstacking. Two-checkpoint audit. |
| G12 (non-blocking) Vignette catboost note | T10 vignette adds explicit catboost-chunked-unsupported note + reconciliation math + per-backend tuning. |
| G13 (non-blocking) Importance graceful-missing | T7.5 summary emits cli_inform when `$feature_importance` is empty. T7.2 sets ranger `importance="impurity"` default. New test (e). |
| Feasibility note | T5 prose corrected: ranger is ABSENT at baseline DESCRIPTION (not Suggests); T5 ADDS to Imports. |

## Changes from v4 → v5 (gate iter 1 fixes + analyst bug reports)

| Source | Change |
|---|---|
| **Analyst bug B1** | xgboost warm-start uses `xgb.train(xgb_model=prev_booster, ...)`, NOT `init_model`. v4 plan referenced wrong API name; v5 corrects in T7.4. Verified via `args(xgboost::xgb.train)` locally. |
| **Analyst bug B2** | catboost R API has NO model-continuation warm-start (verified against official catboost R docs). `baseline` param in `catboost.load_pool` sets initial RAW scores, not model state. v5 REMOVES catboost from G.2 chunked path; T7.4 aborts cleanly with helpful error. |
| Gate G1 (Completeness) | ctrml mfh path was silently NOT deferred in v4. v5 T3 adds mfh defer + new `mat2hmat_cols()` helper. Tests (i) defer-equivalence, (j) ≥2× peak RSS reduction. Addresses user's documented OOM root cause. |
| Gate G2 (Completeness) | No early stopping for gradient boosters in v4. v5 T7.2 adds `early_stopping_rounds` + `validation_split` parameters; lightgbm uses `valids+early_stopping_rounds`; xgboost uses `evals+early_stopping_rounds`; catboost uses `od_type='Iter'+od_wait`; ranger no-op; mlr3 via tuner. Tests (i) early stopping fires, (j) ranger no-op. |
| Gate G3 (Completeness) | Wrapper consolidation (ctrml/terml/csrml) — Scope reviewer judged non-consolidation correct (genuinely different hat semantics — ctrml expands cross-temporal, terml mfh reshapes vector, csrml passes through). Two reviewer disagreement → retain v4 decision: NO consolidation; document in plan. |
| Gate non-blocking | Zero-scale guard added to T7.1 normalize_stack (prevents div-by-zero on constant series). |
| Gate non-blocking | Feature importance stored per backend in T7.2 fit object; surfaced via T7.5 summary. |
| Gate non-blocking | cli progress bar added to rml() loop in T2. |
| Gate non-blocking | mlr3 seed-repro test deferred to follow-up ticket (mlr3 wrapping ranger has multi-layer RNG state; non-trivial; out of v5 critical path). |
| Gate non-blocking | T9 SLA conflation (ranger-vs-randomForest measures algo swap, not refactor): T9 v5 splits benchmark into (i) refactor-only path with `approach="randomForest"` for ALL workloads — direct apples-to-apples speedup vs T0.2 baseline; (ii) default path with ranger — captures user-experience perf. |

## Changes from v3 → v4 (user directive: SOTA, not minimal)

| Reason | Change |
|---|---|
| "ranger is better than randomForest" | T5 makes ranger DEFAULT in ctrml/terml/csrml; randomForest soft-deprecated; DESCRIPTION Version 2.0.0 |
| "consolidate code changes if better" | T7.2 adds `.stack_series()` private helper for 5 rml_g methods (DRY ≥3 occurrences proven) |
| "Verify SOTA" / lateral-thinking | T7.2 adds SOTA series_id categorical encoding (categorical_feature/cat_features/enable_categorical/factor) per backend; comparison test proves accuracy gain |
| "all beneficial changes" — predict methods | T5 adds predict.rml_fit master dispatcher + per-backend predict.rml_<backend>; T6 wires catboost predict; T7.5 adds predict.rml_g_fit |
| "all beneficial changes" — print/summary | T7.5 adds print.rml_g_fit + summary.rml_g_fit |
| "all beneficial changes" — math correctness | T0.1 read-only math audit (sel_mat, mat2hmat, input2rtw, FoReco2matrix, reconciliation algebra); T8 hierarchical-invariant test suite |
| "all beneficial changes" — speed validation | T0.2 baseline benchmark + T9 post-refactor speedup table + NEWS Performance section |
| "all beneficial changes" — maintainability | T10 vignette + roxygen sweep; per-backend seed-reproducibility audit (T5 tests h-k) |
| "not restricted to minimal changes" | DROP LOC ≤1500 ceiling from Cross-task verification |

## Changes from v2 → v3 (post plan-review-gate iter 1)

| Issue (reviewer) | Fix |
|---|---|
| T5 promoted ranger to default = scope creep (Scope) | DROP default change; default stays `randomForest`; ranger is opt-in only |
| T5 `qs_nthreads_adaptive()` helper = YAGNI (Scope) | INLINE `min(parallel::detectCores(), 4L)` at the `qs2::qs_save` site |
| DESCRIPTION `Version:` never bumped (Scope) | Added to Cross-task Final Verification DoD with semver justification |
| csrml not inventoried for T3 (Completeness) | Verified Baseline State now records csrml passes hat directly to rml(); T3 explicitly skips csrml |
| C.1 + B8 touched same line in two commits (Completeness) | Folded B8 into T3 — `dim(hat)<-` used FROM THE START. B8 removed from T5 |
| T3 equivalence tests incomplete on global_id_list branches (Completeness) | Test set expanded to 8 enumerated tests covering kset length 1, sparseVector sel_mat, single-column sel_mat, scalar sel_mat, mfh, non-mfh, csrml regression |
| T5 ranger smoke-only inadequate (Completeness) | Tests expanded: smoke + RF equivalence + checkpoint round-trip + install-hint abort + overflow regression (5 tests) |
| T6 catboost round-trip test absent (Completeness) | Added explicit serialize→deserialize→predict equality test (b); install-hint abort test (c) |
| T7.4 G.2 backend matrix arithmetic ambiguous (Completeness) | Enumerated 5 named tests (a-e) covering lightgbm byte-identical + auto + quality + free_raw_data=FALSE regression + catboost byte-identical |
| NEWS.md only mentioned for T5 (Completeness) | NEWS.md entries added to DoD of T5 (ranger), T6 (catboost), T7.1 (normalize_stack+robscale), T7.3 (3 wrappers), T7.4 (batch_size param) |
| DESCRIPTION dep version audit missed lightgbm/qs2 (Completeness) | T1 step 10 added: verify lightgbm supports `init_model`, qs2 supports `nthreads`; record versions; T7.4 DoD references and bumps if needed |
| Forbidden list conceptual not grep-able (Completeness) | Forbidden list rewritten as literal symbols suitable for `grep -rnE`; grep gate runs at every commit boundary |
| Final-verification push elided (Completeness) | "Cross-task: Final Verification" now a real gated task with 9 DoD checkboxes including `bd dolt push` + `git push` + `git status` per session-close protocol |
