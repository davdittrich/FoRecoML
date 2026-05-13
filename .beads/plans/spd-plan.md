# Active Plan: spd (parallelization + per-iter waste)
<!-- approved: 2026-05-13 -->
<!-- gate-iterations: 3 (final PASS unanimous) -->
<!-- key revisions: v1->v2 RNG-aware DoD + plan selection + mlr3 cap; v2->v3 future API fix + cap generalization to all approaches + ordering enforcement -->
<!-- status: planned, ready for execution -->

# Epic: Speedup (hybrid parallelization + per-iter waste removal)

## Goal
Cut ctrml/csrml/terml wall-clock by parallelizing the per-series loop with sane defaults, and eliminating two per-iter waste patterns (na.omit when no NAs; gc every iter under checkpoint).

## Success Criteria
- [ ] spd.1: hybrid outer-loop parallelization with cap = floor(cores/inner) - 1
- [ ] spd.2: na.omit() skipped when anyNA(X) == FALSE
- [ ] spd.3: gc() under checkpoint runs every K iterations (not every i)
- [ ] All 85+ existing tests pass; numerical output unchanged per RNG-aware policy
- [ ] Public API: 1 new optional argument (`n_workers` on 6 entry points)

## Execution ordering (MANDATORY)
spd.1 lands FIRST. spd.2 and spd.3 land AFTER spd.1 — both modify the loop body which spd.1 restructures into `loop_body <- function(i) { ... }`. Landing spd.2/spd.3 before spd.1 → spd.1 must rebase against modified body (extra merge work). Strict order: spd.1 → spd.2 → spd.3 (or spd.3 → spd.2; both independent after spd.1).

## Context & Background
Post all memory-reduction work (T1-T8 + mw3 + tfg + vu2 + tvy = 16 tickets), per-series fit dominates wall-clock for ctrml + lightgbm. Hybrid formula: outer workers = `floor(detectCores() / inner_threads_per_fit) - 1` (minimum 1). RNG-aware DoD distinguishes deterministic learners (xgboost/lightgbm — bit-equivalence holds) from stochastic (randomForest/ranger — sequential and parallel paths use different RNG kinds, equivalence not claimed).

## Sub-Agent Strategy
3 tickets in strict order. spd.1 → sonnet (medium complexity, multi-file). spd.2 + spd.3 → haiku (single-file mechanical).
# spd.1: Hybrid outer-loop parallelization with auto-cap (v3)
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** Add `n_workers` argument to all 6 entry points. Outer-loop parallel via `future.apply::future_lapply` in `rml()`. Default `"auto"` computes workers = `max(1, floor(detectCores() / inner_threads) - 1)`. Auto-select `multicore` plan on POSIX-non-RStudio; `multisession` elsewhere. Auto-cap inner threads to 1 when outer parallel active (all approaches).
* **Why:** ctrml fits p = nb sequential models. Outer scales near-linearly; lightgbm scales sub-linearly past 8 cores. Hybrid maximizes cores without oversubscription.
* **Mechanism:** New optional argument. Cap formula deterministic. Future plan auto-selected by OS. Inner threads auto-capped via single helper covering randomForest/xgboost/lightgbm/mlr3 paths. Numerical: bit-equivalent (≤1e-12) only for deterministic learners (xgboost/lightgbm); stochastic learners (randomForest/ranger) reproducible across parallel runs but NOT vs sequential due to RNG kind change.
* **Forbidden:**
  - Calling `future::multicore()` or `future::multisession()` directly — they are NOT constructors. Must use `future::plan(future::multicore, workers = N)` or `future::tweak(future::multicore, workers = N)`.
  - Detecting RStudio via `getOption("rstudio.console")` — that option does not exist. Use only `nzchar(Sys.getenv("RSTUDIO"))`.
  - Auto-overriding user-set inner thread params (cap helpers MUST use `is.null` guard).
  - Claiming bit-equivalence between sequential and parallel for randomForest/ranger.

## I.a Numerical equivalence policy (RNG-aware)
- **Sequential (n_workers=1)**: bare `lapply`, R's global RNG (Mersenne-Twister). Byte-identical to pre-patch.
- **Parallel (n_workers>1)**: `future_lapply(future.seed = TRUE)`, L'Ecuyer-CMRG per-substream.
- **Deterministic learners** (xgboost, lightgbm): outputs identical (≤1e-12) across n_workers — RNG state does not affect predictions given fixed inputs.
- **Stochastic learners** (randomForest, mlr3+ranger): outputs differ between sequential and parallel paths (different RNG kinds). Parallel reproducible across runs given same `set.seed()`. Tests assert: (a) sequential byte-identical to pre-patch; (b) parallel reproducible across runs; (c) parallel predictions finite + reasonable shape.

## I.b Plan selection
When `n_workers > 1` AND current `future::plan()` is `sequential`:
- **POSIX (`.Platform$OS.type == "unix"`) AND not RStudio (`Sys.getenv("RSTUDIO") != "1"`)**: use `multicore` (fork; zero-copy via COW; large hat shared without serialization).
- **Otherwise** (Windows, or RStudio on any platform): use `multisession` (subprocess; copies captured environment including hat/obs/base to each worker).

If user already set non-sequential plan: respect it (no override).

## I.c Inner-thread oversubscription guard (all approaches)
When `n_workers > 1`, auto-cap inner threads to 1 via params unless user explicitly set them. Covers all approaches in a single helper:

```r
cap_inner_threads <- function(params, n_workers) {
  if (n_workers <= 1L) return(params)
  if (is.null(params)) params <- list()
  # xgboost direct OR xgboost-as-mlr3
  if (is.null(params$nthread))     params$nthread     <- 1L
  # lightgbm direct OR lightgbm-as-mlr3 OR ranger num_threads (mlr3 underscore form)
  if (is.null(params$num_threads)) params$num_threads <- 1L
  # ranger (mlr3 dot form)
  if (is.null(params$num.threads)) params$num.threads <- 1L
  params
}
```

User override: setting any of these in `params` prevents auto-cap (the `is.null` guard).

## Reference Data (verified HEAD 25be237)
- R/reco_ml.R `rml()` per-series for-loop (post-mw3.3): `out <- vector("list", p); for (i in seq_len(p)) { ... out[[i]] <- ... }`.
- Checkpoint serialize_fit + gc call inside for-loop.
- 6 entry points pass through `params`; `n_workers` piggybacks.

## II. Input Specification
- R/reco_ml.R (rml() body)
- R/utils.R (3 helpers: resolve_n_workers, choose_parallel_plan, cap_inner_threads)
- R/csrml.R, R/terml.R, R/ctrml.R (6 entry points)
- DESCRIPTION (add `future`, `future.apply` to Imports)
- man/{csrml,terml,ctrml}.Rd (roxygen regen)
- tests/testthat/test-parallel.R (NEW)

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **Logic (det. learners)** | n_workers=1 byte-identical to pre-patch. n_workers≥2 vs n_workers=1 within ≤1e-12 for xgboost, lightgbm. |
| **Logic (stoch. learners)** | n_workers=1 byte-identical. n_workers≥2 reproducible across runs (same seed → same output), NOT bit-equal to sequential. |
| **Format** | Match snake_case, 2-space indent. |
| **Boundary** | 6 R files + DESCRIPTION + 3 .Rd + 1 new test. |
| **API** | One new optional argument `n_workers = "auto"`. Backward compatible. |
| **Plan strategy** | `future::plan(future::multicore, workers = N)` on POSIX-non-RStudio; `future::plan(future::multisession, workers = N)` else. NEVER call `future::multicore(...)` or `future::multisession(...)` directly. |
| **RStudio detection** | `Sys.getenv("RSTUDIO") == "1"`. Do NOT use `getOption("rstudio.console")` (phantom option). |
| **Inner-thread cap** | When n_workers>1, auto-set `params$nthread`, `params$num_threads`, `params$num.threads` to 1L if currently NULL. User override via setting any of these in params. |
| **Plan restore** | `prev_plan <- future::plan(); on.exit(future::plan(prev_plan), add = TRUE)` |
| **Checkpoint compat** | Workers write distinct fit_<i>.<ext> to shared filesystem. Path strings returned via lapply result. No race. |

## IV. Step-by-Step Logic
1. Add 3 helpers to R/utils.R:
   ```r
   # Resolve n_workers argument into positive integer.
   resolve_n_workers <- function(n_workers, approach, params) {
     if (is.numeric(n_workers) && length(n_workers) == 1) {
       return(max(1L, as.integer(n_workers)))
     }
     if (identical(n_workers, "auto")) {
       inner <- 1L
       if (!is.null(params)) {
         if (!is.null(params$num_threads))      inner <- max(1L, as.integer(params$num_threads))
         else if (!is.null(params$nthread))     inner <- max(1L, as.integer(params$nthread))
         else if (!is.null(params$num.threads)) inner <- max(1L, as.integer(params$num.threads))
       }
       cores <- parallel::detectCores(logical = TRUE)
       if (!is.finite(cores)) cores <- 1L
       return(max(1L, as.integer(floor(cores / inner) - 1L)))
     }
     cli_abort("`n_workers` must be a positive integer or 'auto'.")
   }

   # Choose parallel plan strategy class (not call it).
   # Returns a function suitable for future::plan(strategy, workers = N).
   choose_parallel_strategy <- function() {
     posix <- .Platform$OS.type == "unix"
     rstudio <- identical(Sys.getenv("RSTUDIO"), "1")
     if (posix && !rstudio) future::multicore else future::multisession
   }

   # Auto-cap inner threads when outer parallel active. Respect user override.
   cap_inner_threads <- function(params, n_workers) {
     if (n_workers <= 1L) return(params)
     if (is.null(params)) params <- list()
     if (is.null(params$nthread))      params$nthread      <- 1L
     if (is.null(params$num_threads))  params$num_threads  <- 1L
     if (is.null(params$num.threads))  params$num.threads  <- 1L
     params
   }
   ```

2. Add `n_workers = "auto"` parameter to all 6 entry points. Pass to `rml()`.

3. In `rml()` after p resolved:
   ```r
   n_workers_resolved <- resolve_n_workers(n_workers, class_base, params)
   if (n_workers_resolved > 1L) {
     params <- cap_inner_threads(params, n_workers_resolved)
   }
   ```

4. Restructure per-series body into `loop_body <- function(i) { ... }`. Body returns `list(bts=..., fit=...)` on training or `list(bts=...)` on reuse (per mw3.3).

5. Dispatch:
   ```r
   out <- if (n_workers_resolved == 1L) {
     lapply(seq_len(p), loop_body)
   } else {
     prev_plan <- future::plan()
     if (inherits(prev_plan, "sequential")) {
       strategy <- choose_parallel_strategy()
       future::plan(strategy, workers = n_workers_resolved)
       on.exit(future::plan(prev_plan), add = TRUE)
     }
     future.apply::future_lapply(seq_len(p), loop_body, future.seed = TRUE)
   }
   ```

6. Roxygen `@param n_workers` on all 6 entry points:
   ```
   @param n_workers Number of parallel workers for the per-series fit loop.
     Default `"auto"` sets workers = `max(1, floor(detectCores() / inner_threads) - 1)`,
     where `inner_threads` is inferred from `params$num_threads` (lightgbm) /
     `params$nthread` (xgboost) / `params$num.threads` (ranger), else 1. Set
     `n_workers = 1` for sequential. When parallel, package auto-selects
     `multicore` (POSIX-non-RStudio; zero-copy fork) or `multisession` (Windows
     / RStudio; copies training data to each worker — higher memory cost).
     Inner thread params (`nthread`/`num_threads`/`num.threads`) are
     auto-capped at 1 when `n_workers > 1` to prevent oversubscription;
     override by setting any of those in `params` explicitly. Reproducibility:
     stochastic learners (randomForest, mlr3+ranger) under parallel use
     L'Ecuyer-CMRG and produce different outputs from sequential (each path
     reproducible within itself); deterministic learners (xgboost, lightgbm)
     match sequential to <=1e-12.
   ```

7. Add `future` and `future.apply` to DESCRIPTION Imports (alphabetical).
8. Run `devtools::document()`.

9. New test `tests/testthat/test-parallel.R` — 7 test_that blocks:
   - "n_workers=1 byte-identical to no-arg default (deterministic fixture)"
   - "n_workers=2 ≤1e-12 vs n_workers=1 for xgboost"
   - "n_workers=2 ≤1e-12 vs n_workers=1 for lightgbm"
   - "n_workers=2 reproducible across runs for randomForest" (NOT vs sequential)
   - "resolve_n_workers formula correctness" (auto with various params combinations)
   - "resolve_n_workers rejects invalid input"
   - "cap_inner_threads sets unset thread params to 1 when n_workers>1; respects user override" (white-box)
   - "choose_parallel_strategy returns multicore on POSIX-non-RStudio else multisession" (white-box; mock Sys.getenv via withr)
   - "checkpoint=path + n_workers=2 (xgboost) produces same output as n_workers=1"

10. Run `Rscript -e 'devtools::test()'` → 85 + 9 new = 94 expected.

11. Commit: "spd.1: hybrid outer-loop parallelization + inner-thread auto-cap"

## V. Output Schema
```toon
task_id: spd.1
success: bool
files_changed: [DESCRIPTION, R/reco_ml.R, R/utils.R, R/csrml.R, R/terml.R, R/ctrml.R, tests/testthat/test-parallel.R, man/csrml.Rd, man/terml.Rd, man/ctrml.Rd]
new_deps: [future, future.apply]
test_result: { passed: int, failed: int }
n_workers_resolution: { auto_no_params: int, auto_lightgbm_4: int, explicit_1: 1, explicit_8: 8 }
plan_strategy_selection: { posix_non_rstudio: multicore, windows_or_rstudio: multisession }
inner_thread_cap:
  unset_capped_to_1: bool
  user_override_respected: bool
numerical_check:
  det_seq_vs_par_xgboost: 0.0
  det_seq_vs_par_lightgbm: 0.0
  stoch_par_reproducibility_rf: 0.0
  stoch_seq_vs_par_rf: "different by design (RNG kind)"
checkpoint_compat: bool
```

## VI. Definition of Done
- [ ] `n_workers` arg on all 6 entry points; default "auto"
- [ ] 3 helpers in utils.R: resolve_n_workers, choose_parallel_strategy, cap_inner_threads
- [ ] `future::plan(strategy, workers = N)` form used (NOT direct constructor call)
- [ ] RStudio detection uses ONLY `Sys.getenv("RSTUDIO") == "1"`
- [ ] cap_inner_threads sets nthread / num_threads / num.threads (all 3) with is.null guards
- [ ] n_workers=1 byte-identical to pre-patch
- [ ] n_workers≥2 deterministic learners ≤1e-12; stochastic reproducible across runs
- [ ] Plan auto-selected by OS+RStudio; user-set plan respected; restored on exit
- [ ] Checkpoint mode compatible under parallel
- [ ] 9 new tests + 85 existing = 94 pass
- [ ] Roxygen documents RNG caveat + inner-thread auto-cap
- [ ] DESCRIPTION Imports has future + future.apply
# spd.2: Skip na.omit(X) when anyNA(X) is FALSE
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** In `rml()` per-series loop body, guard `X <- na.omit(X)` with `if (anyNA(X)) { ... }`. When data is clean (common case), skip the unconditional full-matrix copy.
* **Why:** `na.omit()` on a matrix always allocates a new matrix + `na.action` attribute regardless of NA presence. For clean inputs, this is a per-iter O(NROW × NCOL) copy of X. Original finding #9.
* **Mechanism:** Single conditional wrap.
* **Forbidden:** Changing na.omit behavior when NAs present (must remain identical). Skipping the `y <- y[-attr(X, "na.action")]` line — keep it inside the same guard.
* **Reference Data (verified HEAD 25be237):**
  - R/reco_ml.R rml() lapply/for body (post-mw3.3, mw3.2). Look for the `X <- na.omit(X)` line.

## II. Input Specification
R/reco_ml.R only.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **Logic** | When `anyNA(X)` is FALSE: X unchanged, y unchanged. When TRUE: identical to current behavior (na.omit removes rows + y dropped at na.action indices). |
| **Boundary** | R/reco_ml.R only. |
| **API** | None. |

## IV. Step-by-Step Logic
1. Locate the `X <- na.omit(X)` line in rml() loop body (~around mid-loop).
2. Wrap with `if (anyNA(X)) { ... }`. Move BOTH `X <- na.omit(X)` AND `y <- y[-attr(X, "na.action")]` inside the guard:
   ```r
   if (anyNA(X)) {
     X <- na.omit(X)
     if (!is.null(attr(X, "na.action"))) y <- y[-attr(X, "na.action")]
   }
   ```
3. Run `Rscript -e 'devtools::test()'` → 85/85 pass.
4. Numerical equivalence: csrml × randomForest/xgboost/lightgbm on clean fixture (no NAs). max_abs_diff = 0.0 (since na.omit is no-op when clean). Also: fixture WITH NAs — verify identical behavior (use a fixture with hat[1, 1] = NA).
5. Commit: "spd.2: skip na.omit(X) when anyNA(X) is FALSE"

## V. Output Schema
```toon
task_id: spd.2
success: bool
files_changed: [R/reco_ml.R]
test_result: { passed: int, failed: int }
numerical_check:
  clean_fixture: 0.0
  na_fixture: 0.0    # identical to legacy when NAs present
```

## VI. Definition of Done
- [ ] na.omit guarded by anyNA check
- [ ] Both X-mutation and y-mutation inside same guard
- [ ] 85/85 tests pass
- [ ] Numerical identity on clean + NA fixtures
# spd.3: Reduce gc() frequency under checkpoint mode
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** In `rml()` per-series loop, when `checkpoint_dir` is set, gc() currently runs EVERY iteration. Replace with periodic gc — every `gc_every` iterations (default 5).
* **Why:** gc() takes 50-200ms on modern systems. For p=100 series with cheap fits (xgboost/lightgbm), gc overhead = 5-20s per ctrml call. Cutting gc frequency by 5× reduces overhead proportionally without sacrificing memory benefit (the in-memory model from each iter still gets dropped via mw3.3 conditional store; gc just delays reclamation by a few iters).
* **Mechanism:** Modular counter: `if (i %% gc_every == 0) gc(verbose = FALSE)`. Final gc() after the loop to clean up any residual.
* **Forbidden:** Changing gc behavior when checkpoint_dir is NULL (still no gc). Skipping gc entirely under checkpoint (defeats memory cap).
* **Reference Data (verified HEAD 25be237):**
  - R/reco_ml.R `if (!is.null(checkpoint_dir)) { gc(verbose = FALSE) }` inside the per-series loop body.

## II. Input Specification
R/reco_ml.R only.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **Logic** | Peak memory under checkpoint mode bounded within 5× single-model footprint (was 1×). Acceptable tradeoff for ~5× gc speedup. |
| **Boundary** | R/reco_ml.R only. |
| **API** | None (internal gc_every constant). |
| **Hardcoded constant** | `gc_every <- 5L` as a local in rml(). Not exposed via API. |

## IV. Step-by-Step Logic
1. Locate the per-iter `gc(verbose = FALSE)` call in rml() loop body.
2. Add `gc_every <- 5L` near top of rml() (next to other locals).
3. Replace per-iter gc with conditional:
   ```r
   if (!is.null(checkpoint_dir) && i %% gc_every == 0L) {
     gc(verbose = FALSE)
   }
   ```
4. After the loop, add one final gc to release any residual:
   ```r
   if (!is.null(checkpoint_dir)) gc(verbose = FALSE)
   ```
5. Run `Rscript -e 'devtools::test()'` → 85/85.
6. Numerical equivalence: csrml × randomForest with checkpoint=tempfile() on p≥5 fixture; max_abs_diff = 0.0 vs sequential no-checkpoint baseline.
7. Commit: "spd.3: gc() every 5 iters under checkpoint mode (was every iter)"

## V. Output Schema
```toon
task_id: spd.3
success: bool
files_changed: [R/reco_ml.R]
test_result: { passed: int, failed: int }
gc_every_constant: 5
numerical_check:
  checkpoint_path: 0.0
```

## VI. Definition of Done
- [ ] gc_every = 5 constant introduced
- [ ] Per-iter gc gated on (i %% gc_every == 0)
- [ ] Final post-loop gc remains
- [ ] 85/85 tests pass
- [ ] Numerical output unchanged with checkpoint enabled
