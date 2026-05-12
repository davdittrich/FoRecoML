# Active Plan: mw3 copy elimination
<!-- approved with caveat: gate iterations exhausted (3/3) -->
<!-- final state: Feasibility PASS, Scope PASS, Completeness PASS (after v3+ revisions to mw3.1 csrml resolution, mw3.2 test-drop note, mw3.3 broken-finalizer removal) -->

# Epic: Tier-B mw3 copy elimination

## Goal
Drop redundant memory copies on three independent paths: obs/hat unname duplicates (wrapper + rml), fit-reference triple-hold at end of rml() training, deserialized-model accumulation on predict-reuse.

## Success Criteria
- [ ] mw3.1: obs/hat unname copies eliminated or replaced with in-place dimnames<-NULL
- [ ] mw3.2: ml_step intermediate matrix-of-lists removed; bts + fit$fit extracted via direct lapply
- [ ] mw3.3: predict-reuse drops fit_i after bts extraction; p models no longer accumulate on reuse path
- [ ] All 84/84+ tests pass; numerical output byte-identical across all approaches × all features

## Context & Background
Original deferred-findings analysis #11 (obs copies), #14 (fit refs), plus the final-review note on T6 predict-reuse memory accumulation.

Three independent tickets. mw3.3 has highest standalone ROI (finishes T6 story). mw3.1 + mw3.2 are smaller wins, low risk.

## Plan ordering
Sequential, independent: mw3.3 → mw3.2 → mw3.1 (or any order).
Each lands on own branch off main; merge after spec + quality review.

## Sub-Agent Strategy
Subagent-driven development per superpowers/subagent-driven-development.
Each ticket: implementer (sonnet) → spec reviewer → quality reviewer.
# mw3.1: Drop redundant obs/hat copies in ctrml/terml + rml()
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** Eliminate redundant obs/hat copies along wrapper → rml(). Currently rml() entry does `hat <- unname(hat)` + `obs <- unname(obs)` regardless of whether wrappers already strip dimnames.
* **Why:** ctrml non-mfh path: `obs <- t(obs)` (R/ctrml.R:282, :588) — transpose copies; then `obs <- unname(obs)` in rml() — second copy. obs size = nb × N*m. Two transient copies per call.
* **Mechanism:** Replace `unname()` (always allocates duplicate) with `dimnames(x) <- NULL` (in-place when refcount=1).
* **Reference Data (verified HEAD 5984fca):**
  - R/reco_ml.R:33 `hat <- unname(hat)`
  - R/reco_ml.R:34 `obs <- unname(obs)`
  - R/ctrml.R:282 / R/ctrml.R:588 `obs <- t(obs)` (non-mfh training + _fit)
  - R/terml.R:195 / R/terml.R:444 `obs <- cbind(obs)` (non-mfh)
  - R/csrml.R: NO obs transpose. obs passed directly to rml() at R/csrml.R:265 (`obs = obs`). Confirmed via grep — only mentions are a doc example (line 75) and predict-reuse path setting `obs <- NULL` (line 245). mw3.1 unname-removal still safe for csrml because user-supplied obs may carry names; `dimnames<-NULL` handles it correctly.
  - mfh branches (e.g., R/ctrml.R:284, R/terml.R:197) reshape obs via `matrix(...)` rather than `t()` — out of scope for this ticket; revisit if needed.

## II. Input Specification
R/reco_ml.R only (preferred minimal). Optionally R/ctrml.R + R/terml.R if folding wrapper t/cbind with the in-place dimnames is worthwhile (only if it reduces copies further).

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **Logic** | Per-series y = obs[, i] (or equivalent) byte-identical pre/post across ALL approaches × ALL features. |
| **Boundary** | R/reco_ml.R primary; optional touchpoints in R/ctrml.R + R/terml.R only if needed. |
| **API** | None. |
| **csrml safety** | csrml passes obs raw → dimnames-stripping in rml() is necessary (user obs may carry names). |

## IV. Step-by-Step Logic
1. Replace R/reco_ml.R:33-34:
   ```r
   hat <- unname(hat)
   obs <- unname(obs)
   ```
   with:
   ```r
   if (!is.null(dimnames(hat))) dimnames(hat) <- NULL
   if (!is.null(dimnames(obs))) dimnames(obs) <- NULL
   ```
   The `is.null(dimnames(x))` check avoids the trivial assignment when already stripped (cheap).
2. Numerical equivalence: set.seed(42); fixture from man/csrml.Rd + man/ctrml.Rd + man/terml.Rd; all 4 approaches × applicable features for each wrapper; max_abs_diff = 0.0.
3. `devtools::test()` → 84/84.

## V. Output Schema
```toon
task_id: mw3.1
success: bool
files_changed: [R/reco_ml.R]
test_result: { passed: int, failed: int }
numerical_check:
  csrml: { approaches: [...], features: [...], max_abs_diff: 0.0 }
  ctrml: { approaches: [...], features: [...], max_abs_diff: 0.0 }
  terml: { approaches: [...], features: [...], max_abs_diff: 0.0 }
```

## VI. Definition of Done
- [ ] R/reco_ml.R:33-34 use dimnames(x) <- NULL idiom
- [ ] 84/84 tests pass
- [ ] Numerical output byte-identical across all 3 wrappers × all approaches × all features
# mw3.2: Dedup fit references in rml() return path
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** Replace `do.call("rbind", out) + ml_step[, "fit"] + ml_step[, "bts"]` chain with direct lapply extraction. Drop intermediate matrix-of-lists `ml_step`. PRESERVE existing `is.null(fit)` guard around `fit$fit <- ...`.
* **Why:** Three concurrent references hold the fit list at end of training: `out`, `ml_step`, `fit$fit`. R refcounts (no deep copy) but `ml_step` keeps mlr3/xgboost backend pointers alive longer than necessary. Removing `ml_step` releases sooner.
* **Mechanism:** Direct `lapply(out, [[, "fit")` and `lapply(out, [[, "bts")`. Both produce unnamed lists of length p — shape-equivalent to current `ml_step[, "fit"]` and `ml_step[, "bts"]` (each extracts an unnamed list-column; `do.call("list", x)` on an unnamed list is identity in shape, just rewraps).
* **Forbidden:** Moving `fit$fit <- ...` outside the existing `if (is.null(fit))` guard.
* **Reference Data (verified HEAD 5984fca):**
  - R/reco_ml.R:161 `ml_step <- do.call("rbind", out)`
  - R/reco_ml.R:162-172 `if (is.null(fit)) { fit <- NULL; fit$sel_mat <- sel_mat; fit$fit <- do.call("list", ml_step[, "fit"]); fit$approach <- class_base; fit$checkpoint_dir <- checkpoint_dir; class(fit) <- "rml_fit" }`
  - R/reco_ml.R:174-179 `if (!is.null(base)) { bts <- do.call("cbind", ml_step[, "bts"]); attr(bts, "fit") <- fit; return(bts) } else { return(fit) }`

## II. Input Specification
R/reco_ml.R only.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **Logic** | `bts` content + dim identical. `fit$fit` set ONLY when `is.null(fit)` (training). On predict-reuse, caller-supplied `fit$fit` (checkpoint paths) MUST remain unchanged. |
| **Shape** | `lapply(out, [[, "fit")` produces unnamed length-p list; verify equivalent to `do.call("list", ml_step[, "fit"])` (both unnamed length-p lists of fit objects/paths). Note: `ml_step[, "fit"]` is a list-column of class "list"; `do.call("list", x)` on it is `as.list(x)` — identity for unnamed list inputs. Shape preserved. |
| **Boundary** | R/reco_ml.R only. |
| **API** | None. |

## IV. Step-by-Step Logic
1. Replace lines 161-179 with this exact block (note braces):
   ```r
   if (is.null(fit)) {
     fit <- NULL
     fit$sel_mat <- sel_mat
     fit$fit <- lapply(out, `[[`, "fit")
     fit$approach <- class_base
     fit$checkpoint_dir <- checkpoint_dir
     class(fit) <- "rml_fit"
   }

   if (!is.null(base)) {
     bts <- do.call(cbind, lapply(out, `[[`, "bts"))
     rm(out)
     attr(bts, "fit") <- fit
     return(bts)
   } else {
     rm(out)
     return(fit)
   }
   ```
2. Confirm `ml_step` is fully removed (no later reference).
3. Run `devtools::test()` → 84/84 pass.
4. Numerical equivalence check (set.seed=42; ctrml + csrml + terml × randomForest/xgboost/lightgbm × features [all, str-bts, compact where applicable]); max_abs_diff = 0.0.

## V. Output Schema
```toon
task_id: mw3.2
success: bool
files_changed: [R/reco_ml.R]
test_result: { passed: int, failed: int }
ml_step_removed: bool
fit_fit_guard_preserved: bool
shape_equivalence_verified: bool
numerical_check: { max_abs_diff: 0.0 }
```

## VI. Definition of Done
- [ ] `ml_step` removed from rml()
- [ ] `fit$fit <- lapply(...)` stays inside `if (is.null(fit))` guard
- [ ] `do.call(cbind, lapply(out, [[, "bts"))` replaces `do.call("cbind", ml_step[, "bts"])`
- [ ] `rm(out)` after extraction
- [ ] 84/84 tests pass
- [ ] Numerical output byte-identical

## Appendix: Test scope
This ticket does NOT add a new regression test. Earlier draft (v2) proposed a check on `attr(bts, "fit")$fit` type, but the existing `tests/testthat/test-checkpoint.R:80` already verifies that `mdl$fit` is a character vector on `checkpoint=path` runs — same property, same path. Adding a duplicate test would not improve coverage. The `is.null(fit)` guard preservation (this ticket's primary invariant) is verified via existing test-checkpoint.R suite + the new mw3.3 round-trip test.
# mw3.3: Drop fit_i after bts extraction in predict-reuse loop
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** On `rml()` predict-reuse (`!is.null(fit)`), lambda returns only `list(bts = bts)` — drop `fit` field so deserialized models do not accumulate across the lapply.
* **Why:** T6 final-review: `get_fit_i(fit, i)` lazy-loads each model at R/reco_ml.R:126. Lambda returns `list(bts=bts, fit=fit_i)`. `out[[i]] <- tmp` keeps the model alive via `out[[i]]$fit`. Line 155 `rm(X, y, Xtest, fit_i)` IS unconditional and DOES release the loop-local `fit_i` binding — but `out[[i]]$fit` retains the model reference until `out` itself is released after the loop. All p deserialized models live concurrently until lapply completes.
* **Mechanism:** Conditional return shape at lambda end: `if (is.null(fit)) list(bts=bts, fit=fit_obj) else list(bts=bts)`.
* **Forbidden:** Adding `rm(fit_i)` to the new branch — line 155 already covers it. No double-free.
* **Reference Data (verified HEAD 5984fca):**
  - R/reco_ml.R:92 `out <- vector("list", p)`
  - R/reco_ml.R:103-126 lapply body with branches
  - R/reco_ml.R:154 `out[[i]] <- tmp`
  - R/reco_ml.R:155 `rm(X, y, Xtest, fit_i)` unconditional both paths

## II. Input Specification
R/reco_ml.R + tests/testthat/test-checkpoint.R.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **Logic** | reco_mat unchanged. `rml_fit$fit` (caller checkpoint paths) untouched. |
| **Boundary** | R/reco_ml.R + tests/testthat/test-checkpoint.R only. |
| **API** | None. |
| **No double-free** | Line 155 unchanged. |
| **Cross-ticket** | Compatible with mw3.2: on reuse, `out[[i]]` has only `bts` field. mw3.2's `lapply(out, [[, "bts")` (gated by `!is.null(base)`) is safe — `bts` present on both paths. mw3.2's `lapply(out, [[, "fit")` is gated by `is.null(fit)` — only runs training path where `out[[i]]$fit` exists. |

## IV. Step-by-Step Logic
1. Locate lambda body in `rml()`. At lambda end, find `return(list(bts=..., fit=...))` and replace with:
   ```r
   if (is.null(fit)) {
     list(bts = bts, fit = fit_obj)   # use actual local name from existing code
   } else {
     list(bts = bts)
   }
   ```
2. Line 155 `rm(X, y, Xtest, fit_i)` UNCHANGED.
3. Add mandatory correctness regression test in `tests/testthat/test-checkpoint.R`:
   ```r
   test_that("predict-reuse numerical equivalence with checkpointed fit (mw3.3)", {
     skip_if_not_installed("qs2")
     skip_if_not_installed("randomForest")
     set.seed(42)
     fx <- <fixture from existing helpers>
     mdl_mem <- csrml_fit(hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
                          approach = "randomForest")
     set.seed(42)
     td <- tempfile(); dir.create(td)
     mdl_disk <- csrml_fit(hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
                           approach = "randomForest", checkpoint = td)
     r_mem  <- csrml(base = fx$base, fit = mdl_mem)
     r_disk <- csrml(base = fx$base, fit = mdl_disk)
     expect_equal(as.numeric(r_disk), as.numeric(r_mem), tolerance = 1e-12)
     unlink(td, recursive = TRUE)
   })
   ```
   This test (a) exercises the predict-reuse code path edited by this ticket, (b) verifies numerical equivalence vs in-memory baseline, (c) does NOT attempt to measure internal memory accumulation (which is not observable from R-level test code without rewriting rml()).
4. Memory behavior (the actual accumulation reduction) is verified via code review of the conditional return shape, not a unit test. Document this in the commit message:
   ```
   Memory reduction verified by inspection: on reuse path, lambda returns
   list(bts = bts) only; out[[i]] no longer retains deserialized model.
   Behavioral test asserts predict-reuse numerical correctness; internal
   memory pressure is not observable from R-level test code.
   ```
5. `devtools::test()` → 85/85 pass (84 existing + 1 new).
6. Commit: "mw3.3: drop tmp$fit on predict-reuse to release deserialized models in-loop"

## V. Output Schema
```toon
task_id: mw3.3
success: bool
files_changed: [R/reco_ml.R, tests/testthat/test-checkpoint.R]
test_result: { passed: int, failed: int }
reco_mat_unchanged: bool
rml_fit_object_unchanged: bool
no_double_free: bool
correctness_test_added: bool
mw3_2_compatibility_verified: bool
```

## VI. Definition of Done
- [ ] Lambda return shape conditional on `is.null(fit)`
- [ ] Line 155 `rm(X, y, Xtest, fit_i)` UNCHANGED (no double-free)
- [ ] Correctness regression test added + passing
- [ ] reco_mat + rml_fit$fit unchanged on predict-reuse
- [ ] 85/85 tests pass
- [ ] Commit message documents that memory behavior is verified by code review (not unit test)
