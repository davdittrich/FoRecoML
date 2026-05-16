# Active Plan: spd (mirai-based parallelization + per-iter waste)
<!-- approved: 2026-05-13 -->
<!-- gate iterations: 3 (max). Pivot iter applied; feasibility + scope PASS in final iter; completeness fix applied post-gate to spd.3 (gc_every scope) -->
<!-- key revisions: future -> mirai pivot due to fork+thread UB observed in v1 future::multicore attempt; v3 added complete 13-formal closure list + dots threading + daemons(seed=) wiring; spd.2/spd.3 coordinated with spd.1 loop_body refactor -->

# Epic: Speedup (hybrid parallelization via mirai + per-iter waste removal)

## Goal
Cut ctrml/csrml/terml wall-clock by parallelizing per-series loop via mirai (spawn-based daemons, avoids fork+thread UB with xgboost/lightgbm/ranger), plus 2 micro-optimizations (na.omit skip; gc throttle).

## Success Criteria
- [ ] spd.1: mirai-based outer parallelization, n_workers="auto", inner-thread cap, deterministic daemon teardown
- [ ] spd.2: na.omit skip when no NAs
- [ ] spd.3: gc every K iters under checkpoint
- [ ] 94 tests pass (85 + 9 new); RNG-aware numerical policy
- [ ] Public API: 1 new optional argument (n_workers)

## Why mirai (not future)
Initial spd.1 attempt used `future::multicore`. Hang observed: fork() after OpenMP/threaded init in xgboost/lightgbm/ranger is undefined behavior — child processes idle indefinitely. `future::multisession` would work but serializes hat/obs/base to each worker (cancels T5/T6 memory wins on POSIX) and has per-task subprocess startup overhead.

mirai (Charlie Gao, NNG-based async/parallel): persistent spawn-based daemons (not fork), single cross-platform API, no L'Ecuyer-CMRG overhead, lighter than future, adopted by purrr ≥1.1 / targets / crew. Sidesteps fork+thread hang outright.

## Execution ordering (MANDATORY)
spd.1 first. spd.2 + spd.3 modify the loop body which spd.1 restructures.

## Sub-Agent Strategy
3 tickets. spd.1 → sonnet (multi-file, integration). spd.2 + spd.3 → haiku (single-file mechanical).
# spd.1: Hybrid outer-loop parallelization via mirai (v3 — complete closure list + seed wiring)
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** Add `n_workers = "auto"` argument to all 6 entry points. Parallelize `rml()` per-series loop via `mirai::mirai_map`. Pass ALL closure-captured objects explicitly via `.args=list(...)` (spawn daemons don't inherit host env). Wire L'Ecuyer-CMRG seed via `mirai::daemons(n, seed = ...)` for stochastic reproducibility.
* **Why:** future::multicore hung due to fork+thread UB. mirai = spawn (NNG), avoids issue. Persistent daemons via NNG sockets.

## I.a mirai API (verified live, mirai 2.6.1)
- `mirai::daemons(n, seed = NULL)` — REQUIRES n. seed integer → L'Ecuyer-CMRG per-task substream.
- `mirai::daemons(0)` — teardown.
- `mirai::status()$connections` — query daemon count.
- `mirai::mirai_map(.x, .f, ..., .args = list())` — .args is NAMED list passed to .f.
- `result[]` or `mirai::collect_mirai(result)` — collect.
- `mirai::everywhere({ expr })` — eval on all daemons (load packages, define globals).
- Daemons are SPAWN-FRESH R sessions: no host env inherited.

## I.b Closure list (CRITICAL — verified by reading current rml() lines 96-165 post-mw3.3)
loop_body MUST take ALL of these as formal parameters:
- `i` (iteration index)
- `hat`, `obs`, `base` (training/predict matrices)
- `sel_mat`, `col_map` (selection structures from T4)
- `class_base` (raw approach string)
- `approach` (S3-classed dispatch object — `class(approach) <- c(approach, class_base)` set before loop)
- `active_ncol` (computed from sel_mat/hat outside loop, used in scalar-sel_mat path)
- `params` (post-cap_inner_threads)
- `fit` (input fit object; NULL on training, non-NULL on reuse)
- `checkpoint_dir` (NULL or path string)
- `dots` (captured as `dots <- list(...)` in rml() — carries tuning, block_sampling for mlr3)

Inside loop_body, dispatch via:
```r
tmp <- do.call(.rml, c(
  list(approach = approach, y = y, X = X, Xtest = Xtest, fit = fit_i, params = params),
  dots
))
```

## I.c Seed wiring (for stochastic reproducibility)
At spawn time, derive integer seed from current RNG state so user's `set.seed(42)` propagates:
```r
mirai_seed <- sample.int(.Machine$integer.max, 1L)
mirai::daemons(n_workers_resolved, seed = mirai_seed)
```
User flow:
```r
set.seed(42); r1 <- csrml(..., n_workers = 2)   # both runs sample same int, daemons get same seed
set.seed(42); r2 <- csrml(..., n_workers = 2)
identical(as.numeric(r1), as.numeric(r2))  # TRUE
```

## I.d Numerical equivalence policy (unchanged from v2)
- Deterministic: ≤1e-12 vs sequential.
- Stochastic: reproducible across parallel runs (given same user set.seed); NOT vs sequential.

## I.e Daemon lifecycle
```r
prev <- mirai::status()$connections
if (prev == 0L) {
  mirai_seed <- sample.int(.Machine$integer.max, 1L)
  mirai::daemons(n_workers_resolved, seed = mirai_seed)
  mirai::everywhere({ library(FoRecoML) })
  on.exit(mirai::daemons(0), add = TRUE)
}
```
User pre-configured pool (`status()$connections > 0`): respect it (no spawn, no teardown).

## I.f Inner-thread cap (unchanged)
cap_inner_threads sets nthread/num_threads/num.threads to 1L conditionally when n_workers>1.

## I.g Forbidden
- `future` package.
- `mirai::daemons()$connections` (wrong API; use `mirai::status()$connections`).
- Implicit closure capture across daemon boundary.
- Calling `mirai::daemons(n)` without `seed = <int>` when stochastic reproducibility is required.

## II. Input Specification
- R/reco_ml.R (rml() body + helpers)
- R/utils.R (resolve_n_workers + cap_inner_threads)
- R/csrml.R, R/terml.R, R/ctrml.R (6 entry points)
- DESCRIPTION (add `mirai` to Imports)
- man/{csrml,terml,ctrml}.Rd (roxygen regen)
- tests/testthat/test-parallel.R (NEW)

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **Logic (det.)** | n_workers=1 byte-identical pre-patch. n_workers≥2 ≤1e-12 vs sequential. |
| **Logic (stoch.)** | n_workers=1 byte-identical. n_workers≥2 reproducible across runs given same set.seed before call. NOT compared to sequential. |
| **Boundary** | 6 R files + DESCRIPTION + 3 .Rd + 1 new test. |
| **API** | One new optional arg `n_workers = "auto"`. Backward compatible. |
| **Closure list complete** | loop_body formals MUST include: i, hat, obs, base, sel_mat, col_map, class_base, approach, active_ncol, params, fit, checkpoint_dir, dots. (13 args. Verify against rml() loop body before commit.) |
| **dots threading** | `dots <- list(...)` captured in rml() before loop; passed as `.args=list(..., dots=dots)`; inside loop_body, `do.call(.rml, c(list(approach=...), dots))`. |
| **seed wiring** | `mirai_seed <- sample.int(.Machine$integer.max, 1L)` derived from user's RNG state. Passed as `daemons(n, seed = mirai_seed)`. Stochastic reproducibility test must use `set.seed()` BEFORE the csrml() call. |
| **mirai API** | `mirai::status()$connections` (not daemons()$connections). `daemons(n, seed=)` with seed. `everywhere({ library(FoRecoML) })`. |
| **Lifecycle** | on.exit(mirai::daemons(0)) when we spawned. Respect caller pool. |

## IV. Step-by-Step Logic
1. Add 2 helpers to R/utils.R: `resolve_n_workers`, `cap_inner_threads` (same impls).
2. Add `n_workers = "auto"` to all 6 entry points; pass to rml().
3. In rml() after p resolved:
   ```r
   n_workers_resolved <- resolve_n_workers(n_workers, class_base, params)
   if (n_workers_resolved > 1L) {
     params <- cap_inner_threads(params, n_workers_resolved)
   }
   dots <- list(...)
   ```
4. Restructure per-series body into:
   ```r
   loop_body <- function(i, hat, obs, base, sel_mat, col_map,
                         class_base, approach, active_ncol,
                         params, fit, checkpoint_dir, dots) {
     # ... existing per-iter body, using formal args instead of closures
     # On dispatch:
     #   tmp <- do.call(.rml, c(list(approach=approach, y=y, X=X, Xtest=Xtest,
     #                               fit=fit_i, params=params), dots))
     # Return per mw3.3:
     #   if (is.null(fit)) list(bts = tmp$bts, fit = tmp$fit)
     #   else             list(bts = tmp$bts)
   }
   ```
5. Dispatch:
   ```r
   out <- if (n_workers_resolved == 1L) {
     lapply(seq_len(p), function(i) loop_body(
       i, hat = hat, obs = obs, base = base, sel_mat = sel_mat,
       col_map = col_map, class_base = class_base, approach = approach,
       active_ncol = active_ncol, params = params, fit = fit,
       checkpoint_dir = checkpoint_dir, dots = dots
     ))
   } else {
     prev <- mirai::status()$connections
     if (prev == 0L) {
       mirai_seed <- sample.int(.Machine$integer.max, 1L)
       mirai::daemons(n_workers_resolved, seed = mirai_seed)
       mirai::everywhere({ library(FoRecoML) })
       on.exit(mirai::daemons(0), add = TRUE)
     }
     mirai::mirai_map(
       seq_len(p), loop_body,
       .args = list(
         hat = hat, obs = obs, base = base, sel_mat = sel_mat,
         col_map = col_map, class_base = class_base, approach = approach,
         active_ncol = active_ncol, params = params, fit = fit,
         checkpoint_dir = checkpoint_dir, dots = dots
       )
     )[]
   }
   ```
6. Roxygen @param n_workers on ctrml.R (others inherit). Covers mirai mechanism + RNG caveat + inner-thread cap.
7. Add `mirai` to DESCRIPTION Imports (alphabetical).
8. Run `devtools::document()`.
9. New test `tests/testthat/test-parallel.R` (9 blocks):
   - n_workers=1 byte-identical (xgboost deterministic fixture)
   - n_workers=2 ≤1e-12 vs n_workers=1 (xgboost, nthread=1)
   - n_workers=2 ≤1e-12 vs n_workers=1 (lightgbm, num_threads=1)
   - n_workers=2 reproducible across 2 runs (rF, set.seed BEFORE each call)
   - resolve_n_workers formula
   - resolve_n_workers rejects invalid (NA, list, bad string)
   - cap_inner_threads caps unset, preserves user-set
   - mirai daemon pool torn down: `mirai::status()$connections == 0` after csrml() returns
   - checkpoint=path + n_workers=2 (xgboost) ≤1e-12 vs n_workers=1
10. Run `Rscript -e 'devtools::test()'` → 85 + 9 = 94 expected. Use sequential testing (no concurrent test execution) to avoid daemon pool contention.
11. Commit: "spd.1: hybrid parallelization via mirai (spawn-based daemons + full closure list + seed wiring)"

## V. Output Schema
```toon
task_id: spd.1
success: bool
files_changed: [DESCRIPTION, R/reco_ml.R, R/utils.R, R/csrml.R, R/terml.R, R/ctrml.R, tests/testthat/test-parallel.R, man/csrml.Rd, man/terml.Rd, man/ctrml.Rd]
new_deps: [mirai]
test_result: { passed: int, failed: int }
loop_body_formals_count: 13   # i + 12 captured objects
mirai_lifecycle:
  status_api_used: bool
  seed_wired: bool
  everywhere_load_pkg: bool
  daemons_post_call: 0
numerical_check:
  det_seq_vs_par_xgboost: 0.0
  det_seq_vs_par_lightgbm: 0.0
  stoch_par_reproducibility_rf: 0.0
checkpoint_compat: bool
```

## VI. Definition of Done
- [ ] n_workers arg on all 6 entry points
- [ ] resolve_n_workers + cap_inner_threads in R/utils.R
- [ ] loop_body takes 13 formal params (verify each against rml() captures)
- [ ] dots captured as `list(...)` in rml() and passed via `.args`
- [ ] mirai::status()$connections used (NOT daemons())
- [ ] mirai::daemons(n, seed = sample.int(.Machine$integer.max, 1L)) wires reproducibility
- [ ] mirai::everywhere({ library(FoRecoML) }) called after daemons(n)
- [ ] on.exit teardown when spawned; respect caller pool
- [ ] n_workers=1 byte-identical to pre-patch
- [ ] n_workers≥2 deterministic ≤1e-12
- [ ] n_workers≥2 stochastic reproducible across runs (same set.seed before call)
- [ ] Zombie test: status()$connections == 0 post-call
- [ ] 94 tests pass
- [ ] Roxygen documents mirai mechanism + reproducibility caveat
- [ ] DESCRIPTION has mirai (NO future, NO future.apply)
# spd.2: Skip na.omit(X) when anyNA(X) is FALSE (v2 — coordinated with spd.1)
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** Guard `X <- na.omit(X)` with `if (anyNA(X)) { ... }` inside `loop_body` (post-spd.1).
* **Why:** na.omit allocates a new matrix unconditionally. Skip when X has no NAs.

## II. Input Specification
R/reco_ml.R only.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **Logic** | When anyNA(X) is FALSE: X unchanged, y unchanged. When TRUE: identical to current behavior. |
| **Boundary** | R/reco_ml.R only. |
| **API** | None. |
| **attr guard form** | Use `!is.null(attr(X, "na.action"))` (matches current code at reco_ml.R:112 which uses `length() > 0` — these are equivalent for na.omit's na.action since na.action is either NULL or non-empty integer vec; either form correct; pick `!is.null` for clarity). |

## IV. Step-by-Step Logic
1. Inside `loop_body` (post-spd.1) where `X <- na.omit(X)` lives, wrap:
   ```r
   if (anyNA(X)) {
     X <- na.omit(X)
     if (!is.null(attr(X, "na.action"))) y <- y[-attr(X, "na.action")]
   }
   ```
2. Run `Rscript -e 'devtools::test()'` → 94+.
3. Numerical equivalence: csrml on clean fixture (no NAs) AND on fixture with hat[1,1] = NA. max_abs_diff = 0.0 both cases.
4. Commit: "spd.2: skip na.omit when anyNA(X)==FALSE"

## V. Output Schema
```toon
task_id: spd.2
success: bool
files_changed: [R/reco_ml.R]
test_result: { passed: int, failed: int }
numerical_check:
  clean_fixture: 0.0
  na_fixture: 0.0
```

## VI. Definition of Done
- [ ] na.omit guarded by anyNA check (inside loop_body)
- [ ] X-mutation and y-mutation both inside guard
- [ ] 94+ tests pass
- [ ] Numerical identity on clean + NA fixtures
# spd.3: Reduce gc() frequency under checkpoint mode (v2 — coordinated with spd.1 loop_body)
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** Per-iter `gc()` under checkpoint mode runs every iteration. Replace with periodic gc (every 5 iters). Define `gc_every` INSIDE `loop_body` (so it's accessible after spd.1 restructures the for-loop into a parallelizable function).
* **Why:** gc() takes 50-200ms. For p=100 cheap fits, gc overhead = 5-20s. Cut by 5×.
* **Forbidden:** Defining `gc_every` in `rml()` outer scope only — that variable won't be visible inside `loop_body` after spd.1 refactor (spawn daemons don't inherit). Must be either inside loop_body or in .args.

## II. Input Specification
R/reco_ml.R only.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **Logic** | Memory peak bounded within 5× single-model footprint under checkpoint. |
| **Boundary** | R/reco_ml.R only. |
| **API** | None. |
| **Scope-safe under spd.1** | `gc_every` defined INSIDE `loop_body` (function-local), NOT outer `rml()` scope. Each daemon's loop_body has its own constant. |

## IV. Step-by-Step Logic
1. Inside `loop_body` (post-spd.1), at the top of the function:
   ```r
   loop_body <- function(i, hat, obs, base, sel_mat, col_map, class_base,
                         approach, active_ncol, params, fit, checkpoint_dir, dots) {
     gc_every <- 5L
     # ... existing body ...
   }
   ```
2. Replace the per-iter gc() call:
   ```r
   if (!is.null(checkpoint_dir)) {
     gc(verbose = FALSE)
   }
   ```
   with:
   ```r
   if (!is.null(checkpoint_dir) && i %% gc_every == 0L) {
     gc(verbose = FALSE)
   }
   ```
3. Note: with spd.1 in place, each daemon evaluates `loop_body` independently. The `i %% gc_every` counter is on the global iteration index `i` (1..p) passed via mirai_map's `.x`. Daemons hit i values 1, 2, 3, ... in arbitrary order; each daemon will trigger gc when its local i is divisible by 5. Equivalent total gc count across full run.
4. No post-loop gc in rml() (each daemon manages its own GC; orchestrator's gc is automatic).
5. Run `Rscript -e 'devtools::test()'` → 94+ pass (after spd.1 + spd.3).
6. Numerical equivalence: csrml × randomForest × checkpoint=tempfile() on p>=10 fixture. max_abs_diff = 0.0 vs baseline.
7. Commit: "spd.3: gc() every 5 iters under checkpoint mode (inside loop_body for spd.1 compat)"

## V. Output Schema
```toon
task_id: spd.3
success: bool
files_changed: [R/reco_ml.R]
test_result: { passed: int, failed: int }
gc_every_constant: 5
gc_every_scope: "inside loop_body"   # NOT outer rml()
numerical_check: { checkpoint_path: 0.0 }
```

## VI. Definition of Done
- [ ] `gc_every <- 5L` defined INSIDE loop_body (function-local)
- [ ] Per-iter gc gated on (`i %% gc_every == 0L`)
- [ ] No `gc_every` reference in outer rml() scope
- [ ] 94+ tests pass
- [ ] Numerical output byte-identical with checkpoint enabled
