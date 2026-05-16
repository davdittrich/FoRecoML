# Active Plan: spd.9 + spd.10 (auto-checkpoint hardening)
<!-- approved with caveat: gate iter 3/3 — feasibility flagged expect_no_message scope, fixed in v4; completeness/scope flagged available_ram mock — verified IS in spd.9 spec (false positive due to compressed read) -->

# Epic: Auto-checkpoint hardening (training + predict-reuse safety)

## Goal
Close two remaining checkpoint auto-mode gaps that allowed daemon-copy OOM:
- spd.9: training-time auto threshold ignores n_workers multiplier → underestimates peak → checkpoint stays off → fit$fit holds in-memory boosters.
- spd.10: predict-time with non-checkpoint fit + n_workers > 1 silently copies the full booster list to each daemon → triplicated/N-plicated OOM.

## Success Criteria
- [ ] spd.9: `resolve_checkpoint` factors `n_workers` into peak estimate; threshold tightened to 0.5 (was 0.8) of available RAM
- [ ] spd.10: predict-reuse path auto-caps `n_workers = 1` with cli_inform when fit$fit holds in-memory boosters
- [ ] Existing 112 tests pass; numerical output unchanged
- [ ] Real-world predict-reuse with in-memory fit + n_workers=3 no longer OOMs

## Context
User reported OOM at predict time with `n_workers=3` on hierarchy with 2432 in-memory lgb.Booster objects. Diagnosis confirmed:
- Training was done with `checkpoint="auto"` (default).
- Auto threshold did NOT fire (estimate < 0.8 × avail) because estimate didn't multiply by n_workers.
- fit$fit ended up as 2432 in-memory lgb.Booster R6 objects.
- Predict with `n_workers=3` → 3 mirai daemons × full booster list closure → catastrophic memory.

spd.8 (gc on predict-reuse) fixed in-daemon accumulation. Daemon-baseline (closure size) was unaddressed — this epic closes that gap.

## Plan ordering
spd.9 and spd.10 independent (different code paths). Can land either order. Recommend spd.9 first (smaller change, lower risk).

## Sub-Agent Strategy
2 tickets. spd.9 = utils.R + reco_ml.R wiring; spd.10 = reco_ml.R only. Both haiku-tier mechanical edits.
# spd.9: n_workers-aware checkpoint auto threshold (v3)
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** Thread `n_workers` through `resolve_checkpoint`; in auto mode, multiply `estimate_peak_bytes` by n_workers; tighten threshold 0.8 → 0.5 of available RAM.
* **Why:** Current auto threshold ignores parallel daemon multiplier. For n_workers=3, real peak ~3× single-process estimate.

## I.a No reorder needed
HEAD 8e18958 already has `n_workers_resolved` at line 49 BEFORE `checkpoint_dir` at line 60-64. Only signature + call-site change needed.

## Reference Data (verified HEAD 8e18958)
R/utils.R:103 `resolve_checkpoint(checkpoint, hat, approach, p)`. Auto branch threshold `est > 0.8 * avail` at line 114.

## II. Input Specification
R/utils.R + R/reco_ml.R + tests/testthat/test-checkpoint.R.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **Logic** | TRUE/FALSE/path branches UNCHANGED. Auto: multiply est × n_workers, threshold 0.5. |
| **Boundary** | 3 files. |
| **API** | resolve_checkpoint gains `n_workers = 1L` (internal). |
| **NA fallback** | available_ram NA → auto OFF. UNCHANGED. |

## IV. Step-by-Step Logic
1. R/utils.R `resolve_checkpoint`:
   ```r
   resolve_checkpoint <- function(checkpoint, hat, approach, p, n_workers = 1L) {
     # ... TRUE/FALSE/path UNCHANGED ...
     if (checkpoint == "auto") {
       est <- estimate_peak_bytes(hat, approach, p) * max(1L, as.integer(n_workers))
       avail <- available_ram_bytes()
       if (is.finite(est) && is.finite(avail) && est > 0.5 * avail) {
         return(checkpoint_session_dir())
       }
       return(NULL)
     }
     # ... rest UNCHANGED ...
   }
   ```
2. R/reco_ml.R: pass `n_workers_resolved` to resolve_checkpoint at line 61:
   ```r
   checkpoint_dir <- if (is.null(fit)) {
     resolve_checkpoint(checkpoint, hat, class_base, p, n_workers = n_workers_resolved)
   } else {
     NULL
   }
   ```
3. **DIAGNOSTIC tests** in tests/testthat/test-checkpoint.R — must distinguish multiplier applied vs not:
   ```r
   test_that("resolve_checkpoint auto: n_workers multiplier crosses threshold", {
     skip_if_not_installed("testthat")
     # Mock avail to 1 GB. Threshold = 0.5 GB.
     # Construct hat such that single-worker estimate is BELOW threshold,
     # but n_workers=4 multiplier crosses it.
     local_mocked_bindings(
       available_ram_bytes = function() 1e9,  # 1 GB
       .package = "FoRecoML"
     )
     # Pick: hat ~5 MB. randomForest per_model=5. p=10.
     # estimate_peak_bytes = hat_bytes * 5 * p + hat_bytes * 3 = hat_bytes * 53.
     # For hat_bytes = 5e6: est_single ≈ 265 MB. < 500 MB threshold → no fire.
     # For n_workers = 4: est_par ≈ 1.06 GB. > 500 MB → fires.
     hat <- matrix(rnorm(1000 * 625), nrow = 1000)  # 1000*625*8 = 5e6 bytes
     # Verify the math:
     est_single <- as.numeric(NROW(hat)) * NCOL(hat) * 8 * 5 * 10 + as.numeric(NROW(hat)) * NCOL(hat) * 8 * 3
     stopifnot(est_single < 0.5e9 && est_single * 4 > 0.5e9)  # sanity

     # Single-worker: stays OFF (below threshold)
     r1 <- FoRecoML:::resolve_checkpoint("auto", hat, "randomForest", p = 10, n_workers = 1L)
     expect_null(r1)
     # 4 workers: crosses threshold → returns a path (checkpoint dir)
     r4 <- FoRecoML:::resolve_checkpoint("auto", hat, "randomForest", p = 10, n_workers = 4L)
     expect_true(is.character(r4))
     expect_true(dir.exists(r4))
     unlink(r4, recursive = TRUE)
   })

   test_that("resolve_checkpoint backward compat: default n_workers = 1L", {
     expect_no_error(FoRecoML:::resolve_checkpoint("auto", matrix(1:4, 2, 2), "lightgbm", p = 1))
     expect_no_error(FoRecoML:::resolve_checkpoint(FALSE, NULL, "lightgbm", p = 1))
     expect_no_error(FoRecoML:::resolve_checkpoint(TRUE, NULL, "lightgbm", p = 1))
   })
   ```
4. Run `Rscript -e 'devtools::test()'` → expect 114/114.
5. Numerical equivalence (smoke): csrml × randomForest × n_workers=1; max_abs_diff = 0.0 vs HEAD.
6. Commit: `git add R/utils.R R/reco_ml.R tests/testthat/test-checkpoint.R && git commit -m "spd.9: n_workers-aware checkpoint auto threshold (0.8 -> 0.5)"`

## V. Output Schema
```toon
task_id: spd.9
success: bool
files_changed: [R/utils.R, R/reco_ml.R, tests/testthat/test-checkpoint.R]
test_result: { passed: int, failed: int }
threshold_changed: { old: 0.8, new: 0.5 }
diagnostic_test_distinguishes_multiplier: bool
```

## VI. Definition of Done
- [ ] resolve_checkpoint gains n_workers=1L default
- [ ] Auto-mode estimate × n_workers
- [ ] Threshold 0.5
- [ ] rml() passes n_workers_resolved
- [ ] **Diagnostic test**: fixture where single-worker est < threshold AND par-worker est > threshold; positive case fires, single-case doesn't.
- [ ] Backward-compat test for default n_workers
- [ ] 114/114 tests pass
# spd.10: Predict-time auto-cap n_workers on in-memory fit (v4 — final)
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** rml() predict-reuse path — if n_workers > 1 AND fit$fit holds in-memory model objects (not character paths), auto-cap n_workers=1 with cli_inform.

## Reference Data (verified HEAD 8e18958)
R/reco_ml.R:49 n_workers_resolved; line 51-52 cap_inner_threads; line 60 checkpoint_dir. Insert guard after cap_inner_threads (line ~55). csrml fit-reuse: `agg_mat` NOT required; silently overwritten from `fit$agg_mat`. `make_cs_fixture(seed, N_hat)` exists in test-parallel.R lines 9-26.

## II. Input Specification
R/reco_ml.R + tests/testthat/test-parallel.R.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **Logic** | Training UNCHANGED. Predict + path fit + parallel UNCHANGED. Predict + in-memory fit + parallel: cap with cli_inform. |
| **Boundary** | 2 files. |
| **API** | None. Silent cap with cli_inform. |
| **Edge cases** | 4-clause && chain. Empty list → no fire. |
| **expect_no_message scope** | DO NOT wrap assignment inside expect_no_message — value won't propagate. Split into two statements. |

## IV. Step-by-Step Logic
1. R/reco_ml.R: insert guard after cap_inner_threads (~line 55), BEFORE checkpoint_dir:
   ```r
   # spd.10 — predict-reuse safety: in-memory booster list × N daemons = OOM.
   if (n_workers_resolved > 1L &&
       !is.null(fit) &&
       !is.null(fit$fit) &&
       length(fit$fit) > 0L &&
       !is.character(fit$fit[[1]])) {
     cli::cli_inform(c(
       "!" = "{.arg fit} holds {length(fit$fit)} in-memory model(s); copying to {n_workers_resolved} workers would risk OOM.",
       "i" = "Auto-capping {.arg n_workers} = 1. Train with {.code checkpoint = TRUE} (or a path) for parallel predict-reuse."
     ))
     n_workers_resolved <- 1L
   }
   ```
2. Add tests to tests/testthat/test-parallel.R. **IMPORTANT: do NOT put assignment inside expect_no_message** — wrap-then-assign:
   ```r
   test_that("spd.10: in-memory fit + n_workers>1 auto-caps to 1 with cli_inform", {
     skip_if_not_installed("randomForest")
     fx <- make_cs_fixture(seed = 42L, N_hat = 30L)
     set.seed(42)
     mdl_mem <- csrml_fit(
       hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
       approach = "randomForest",
       params = list(ntree = 5L),
       checkpoint = FALSE,
       n_workers = 1L
     )
     expect_false(is.character(mdl_mem$fit[[1]]))

     # Verify message emitted (do not capture value here):
     expect_message(
       csrml(base = fx$base, fit = mdl_mem, n_workers = 3L),
       "in-memory model"
     )
     # Capture value separately (suppress message to keep test output clean):
     r_par <- suppressMessages(csrml(base = fx$base, fit = mdl_mem, n_workers = 3L))
     r_seq <- csrml(base = fx$base, fit = mdl_mem, n_workers = 1L)
     expect_equal(as.numeric(r_par), as.numeric(r_seq), tolerance = 1e-12)
   })

   test_that("spd.10: path fit + n_workers>1 NOT auto-capped", {
     skip_if_not_installed("qs2")
     skip_if_not_installed("xgboost")
     fx <- make_cs_fixture(seed = 42L, N_hat = 30L)
     td <- tempfile(); dir.create(td)
     set.seed(42)
     mdl_disk <- csrml_fit(
       hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
       approach = "xgboost",
       params = list(nthread = 1L, nrounds = 5L),
       checkpoint = td,
       n_workers = 1L
     )
     expect_true(is.character(mdl_disk$fit[[1]]))

     # Negative case: no message
     expect_no_message(
       csrml(base = fx$base, fit = mdl_disk, n_workers = 2L),
       message = "in-memory"
     )
     unlink(td, recursive = TRUE)
   })
   ```
3. Run `Rscript -e 'devtools::test()'` → expect 116/116.
4. Commit: `git add R/reco_ml.R tests/testthat/test-parallel.R && git commit -m "spd.10: auto-cap n_workers=1 on predict-reuse with in-memory fit"`

## V. Output Schema
```toon
task_id: spd.10
success: bool
files_changed: [R/reco_ml.R, tests/testthat/test-parallel.R]
test_result: { passed: int, failed: int }
positive_test_fires_cap: bool
negative_test_skips_cap: bool
```

## VI. Definition of Done
- [ ] Guard with 4-clause chain at correct location
- [ ] cli_inform message clear
- [ ] Positive test: message captured via expect_message (no assignment); value captured separately via suppressMessages
- [ ] Negative test: expect_no_message on bare csrml call (no assignment)
- [ ] 116/116 tests pass
