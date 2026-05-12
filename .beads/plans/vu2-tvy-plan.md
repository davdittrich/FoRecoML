# Active Plan: vu2 + tvy
<!-- approved with caveat: gate iterations 3/3, 1 real fix applied post-cycle -->
<!-- Feasibility PASS, Scope PASS, Completeness 3 false positives (compressed-reviewer-view artifacts) + 1 real fixed (tvy.1a per-site arg names) -->

# Epic: Tier-B vu2 mlr3 path tweaks

## Goal
Eliminate redundant task construction inside `rml.mlr3()` when block_sampling is supplied.

## Success Criteria
- [ ] vu2.1: rml.mlr3 builds tsk_i once per call (no double allocation when block_sampling set)
- [ ] 85/85 tests pass; mlr3 numerical output unchanged.

## Context & Background
Finding #7. Current `rml.mlr3()` always builds the non-block tsk_i first (lines 217-218), then if `block_sampling != NULL` rebuilds it with an id column (lines 239-240). First task is GC-eligible only after reassignment; mlr3 backend deep-copies training data on construction. Doubles training-set transient cost when block_sampling active.

## Sub-Agent Strategy
Single hermetic ticket. R/reco_ml.R only.
# vu2.1: Build tsk_i once in rml.mlr3 block_sampling branch
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** In `rml.mlr3()`, build `tsk_i` exactly ONCE per call. Currently builds unconditionally first (lines 217-218), then rebuilds inside `block_sampling != NULL` branch (lines 239-240) — discarding the first.
* **Why:** mlr3 `as_task_regr()` deep-copies its data into DataBackendDataTable. Double construction = double training-set transient copy.
* **Mechanism:** Gate the initial build on `is.null(block_sampling)`. When block_sampling set, build the id-augmented form directly (single construction). DROP `rownames(X) <- NULL` (no-op on matrix; redundant per tvy.1b).

## Reference Data (verified HEAD 6c32efe)
R/reco_ml.R lines 215-244 (rml.mlr3 training path), relevant block:
```r
if (is.null(fit)) {
  if (is.null(y) && is.null(X)) cli_abort(...)
  tsk_i <- data.frame(y = y, X, check.names = FALSE)                        # 217 — unconditional first build
  tsk_i <- mlr3::as_task_regr(tsk_i, target = "y")                          # 218
  fit <- do.call(lrn, params)
  if (!is.null(tuning)) {
    ...
    if (!is.null(block_sampling)) {
      rownames(X) <- NULL                                                    # 238 — no-op on matrix
      tsk_i <- data.frame(y = y, X, id = rep(1:NROW(X), each = block_sampling), check.names = FALSE)
      tsk_i <- mlr3::as_task_regr(tsk_i, target = "y")
      tsk_i$col_roles$group <- "id"
      tsk_i$col_roles$feature <- setdiff(tsk_i$col_roles$feature, "id")
      tuning$resampling$instantiate(tsk_i)
    }
    ...
  }
  fit$train(tsk_i)
}
```

## II. Input Specification
R/reco_ml.R ONLY.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **Logic** | mlr3 task semantics unchanged: id column has `col_role = "group"`, features exclude id. fit$train(tsk_i) sees identical input. |
| **Boundary** | R/reco_ml.R only. |
| **API** | None. |
| **No double-build** | tsk_i constructed exactly once per training call. |
| **rownames<-NULL drop** | vu2.1 MUST emit a restructured branch WITHOUT `rownames(X) <- NULL`. This is non-negotiable regardless of tvy.1b execution order (see §VII). |

## IV. Step-by-Step Logic
1. Restructure lines 217-240 region so tsk_i is built once:
   ```r
   if (is.null(fit)) {
     if (is.null(y) && is.null(X)) cli_abort(...)

     if (is.null(block_sampling)) {
       tsk_i <- data.frame(y = y, X, check.names = FALSE)
       tsk_i <- mlr3::as_task_regr(tsk_i, target = "y")
     } else {
       tsk_i <- data.frame(
         y = y, X,
         id = rep(seq_len(NROW(X)), each = block_sampling),
         check.names = FALSE
       )
       tsk_i <- mlr3::as_task_regr(tsk_i, target = "y")
       tsk_i$col_roles$group <- "id"
       tsk_i$col_roles$feature <- setdiff(tsk_i$col_roles$feature, "id")
     }

     fit <- do.call(lrn, params)
     if (!is.null(tuning)) {
       # existing tuning setup …
       if (!is.null(block_sampling)) {
         tuning$resampling$instantiate(tsk_i)
       }
       # autotuner …
     }
     fit$train(tsk_i)
   }
   ```
   Note absence of `rownames(X) <- NULL` line.
2. `1:NROW(X)` → `seq_len(NROW(X))` (robustness; safe when NROW=0).
3. col_roles$group + col_roles$feature set INSIDE the new else branch, BEFORE tuning$resampling$instantiate (preserves invariant).
4. Run `Rscript -e 'devtools::test()'` → must show 85/85.
5. Numerical equivalence (mlr3 path, both modes): set.seed(42); max_abs_diff = 0.0 vs baseline HEAD.
6. Commit: "vu2.1: build mlr3 tsk_i once (drop double construct + rownames no-op)"

## V. Output Schema
```toon
task_id: vu2.1
success: bool
files_changed: [R/reco_ml.R]
test_result: { passed: int, failed: int }
tsk_i_builds_per_call: 1
rownames_null_removed: true
numerical_check:
  block_sampling_null: { max_abs_diff: 0.0 }
  block_sampling_set: { max_abs_diff: 0.0 }
```

## VI. Definition of Done
- [ ] tsk_i constructed exactly once per training call
- [ ] block_sampling branch attaches col_roles correctly
- [ ] No `rownames(X) <- NULL` in the restructured branch (`grep -c "rownames(X) <- NULL" R/reco_ml.R` → 0 after this ticket)
- [ ] 85/85 tests pass
- [ ] Numerical mlr3 output unchanged (both block_sampling modes)

## VII. Cross-ticket interaction with tvy.1b
* **vu2.1 unilaterally drops `rownames(X) <- NULL`.** Whether tvy.1b lands before or after vu2.1, the final state has 0 occurrences. Post-vu2.1 grep MUST return 0 hits.
* If tvy.1b lands first: the line is already gone; vu2.1's restructure naturally produces 0 hits.
* If vu2.1 lands first: vu2.1's restructure produces 0 hits; tvy.1b becomes a no-op (its target line is gone) — that ticket may be closed-as-superseded.
# Epic: Tier-C tvy cosmetic no-ops

## Goal
Remove no-op wrappers identified in original memory analysis.

## Success Criteria
- [ ] tvy.1: as.vector(predict(...)) wrappers dropped where predict returns numeric vector
- [ ] tvy.1: rownames(X) <- NULL removed when X is matrix (no-op)
- [ ] tvy.1: unname(base) replaced with in-place dimnames<-NULL + names<-NULL (per mw3.1 pattern)
- [ ] 85/85 tests pass; numerical output unchanged.

## Context & Background
Original findings #15, #16, #18. C-tier micro-optimizations. Each individually small; bundled because all trivial + same review pass.

## Sub-Agent Strategy
Single bundled ticket. R/reco_ml.R only.
# tvy.1a: Drop as.vector(predict(...)) no-op wrapper (3 sites)
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** Remove `as.vector(...)` wrapping `predict()` calls at 3 sites in R/reco_ml.R. predict() for randomForest/xgboost/lightgbm regression already returns numeric vector; `as.vector()` is identity-copy (xgboost/lightgbm) or names-stripping (randomForest, names unused downstream).
* **Why:** Micro-allocation per per-series predict. Original finding #16.
* **Reference Data (verified HEAD 6c32efe):**
  - R/reco_ml.R:328 `bts <- as.vector(predict(fit, Xtest))` — rml.randomForest predict; arg = `Xtest`
  - R/reco_ml.R:393 `bts <- as.vector(predict(fit, test))` — rml.xgboost predict; arg = `test` (xgb.DMatrix)
  - R/reco_ml.R:460 `bts <- as.vector(predict(fit, Xtest))` — rml.lightgbm predict; arg = `Xtest`
  Commented-out at 502/549/595 UNCHANGED.

## II. Input Specification
R/reco_ml.R only.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **Logic** | `bts` numeric content + length unchanged. randomForest may drop row-name attribute (names not used downstream — verified). |
| **Boundary** | R/reco_ml.R only. |
| **API** | None. |
| **Per-site arg names** | Preserve the existing argument name at each site (`Xtest` at 328 + 460; `test` at 393). DO NOT rename. |

## IV. Step-by-Step Logic
1. At each site, unwrap `as.vector(...)`. Per-site edits:
   - **R/reco_ml.R:328** (rml.randomForest):  
     `bts <- as.vector(predict(fit, Xtest))` → `bts <- predict(fit, Xtest)`
   - **R/reco_ml.R:393** (rml.xgboost):  
     `bts <- as.vector(predict(fit, test))` → `bts <- predict(fit, test)`  
     (Note: arg is `test`, an `xgb.DMatrix` built earlier as `test <- xgb.DMatrix(data = Xtest)`. DO NOT rename `test` to `Xtest`.)
   - **R/reco_ml.R:460** (rml.lightgbm):  
     `bts <- as.vector(predict(fit, Xtest))` → `bts <- predict(fit, Xtest)`
2. Run `Rscript -e 'devtools::test()'` → 85/85.
3. Numerical equivalence: set.seed(42); csrml × randomForest/xgboost/lightgbm; max_abs_diff = 0.0 vs baseline HEAD 6c32efe.
4. Commit: "tvy.1a: drop as.vector(predict(...)) wrapper at 3 sites"

## V. Output Schema
```toon
task_id: tvy.1a
success: bool
files_changed: [R/reco_ml.R]
sites_changed: 3
per_site_arg_preserved:
  line_328: Xtest
  line_393: test
  line_460: Xtest
test_result: { passed: int, failed: int }
numerical_check: { max_abs_diff: 0.0 }
```

## VI. Definition of Done
- [ ] 3 sites unwrapped with per-site argument name preserved
- [ ] No `as.vector(predict(` remains in live code (`grep -n "as.vector(predict(" R/reco_ml.R | grep -v "^[0-9]*:#"` → 0 hits)
- [ ] 85/85 tests pass
- [ ] Numerical output byte-identical
# tvy.1b: Drop rownames(X) <- NULL no-op in rml.mlr3 block_sampling
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** Remove R/reco_ml.R:238 `rownames(X) <- NULL`. Post-T1, X is numeric matrix throughout rml.mlr3; rownames already NULL; assignment is no-op.
* **Why:** Original finding #18. Pre-T1 vestige (X was data.frame; rownames mattered for mlr3 task ID column).
* **Reference Data (verified HEAD 6c32efe):**
  - R/reco_ml.R:238 `rownames(X) <- NULL` inside `if (!is.null(block_sampling)) { ... }`

## II. Input Specification
R/reco_ml.R only.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **Logic** | X stays a numeric matrix; rownames stay NULL implicitly. mlr3 task built immediately after retains the same structure. |
| **Boundary** | R/reco_ml.R only. |
| **API** | None. |
| **Cross-ticket** | If vu2.1 lands AFTER tvy.1b, vu2.1 implementation MUST NOT re-introduce this line (vu2.1 has a corresponding guard). If vu2.1 lands FIRST, tvy.1b removes whatever residual rownames<-NULL remains. |

## IV. Step-by-Step Logic
1. Delete R/reco_ml.R:238 (the entire `rownames(X) <- NULL` line).
2. Run `devtools::test()` → 85/85.
3. Numerical equivalence on mlr3 block_sampling path: set.seed(42); max_abs_diff = 0.0.
4. Commit: "tvy.1b: drop rownames(X) <- NULL no-op (X is matrix post-T1)"

## V. Output Schema
```toon
task_id: tvy.1b
success: bool
files_changed: [R/reco_ml.R]
test_result: { passed: int, failed: int }
line_removed: 238
numerical_check: { max_abs_diff: 0.0 }
```

## VI. Definition of Done
- [ ] Line removed
- [ ] 85/85 tests pass
- [ ] mlr3 block_sampling numerical output byte-identical
# tvy.1c: Replace unname(base) with in-place dimnames<-NULL + names<-NULL
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** R/reco_ml.R:57 `base <- unname(base)` → conditional in-place pattern matching mw3.1's hat/obs fix at lines 33-36.
* **Why:** `unname()` always allocates a duplicate. `dimnames<-NULL` is in-place when refcount permits. Original finding #15 — mw3.1 explicitly left base untouched.
* **Reference Data (verified HEAD 6c32efe):**
  - R/reco_ml.R:57 `base <- unname(base)` (inside `if (!is.null(base))` block starting around line 55)
  - mw3.1 idiom already at lines 33-36 (hat + obs)

## II. Input Specification
R/reco_ml.R only.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **Logic** | base content + dim unchanged; names/dimnames stripped equivalently. |
| **Boundary** | R/reco_ml.R only. |
| **API** | None. |

## IV. Step-by-Step Logic
1. Replace R/reco_ml.R:57 with:
   ```r
   if (!is.null(names(base))) names(base) <- NULL
   if (!is.null(dimnames(base))) dimnames(base) <- NULL
   ```
2. Run `devtools::test()` → 85/85.
3. Numerical equivalence: predict-reuse paths (where base is supplied); set.seed(42); max_abs_diff = 0.0.
4. Commit: "tvy.1c: in-place dimnames<-NULL + names<-NULL for base (parity with mw3.1)"

## V. Output Schema
```toon
task_id: tvy.1c
success: bool
files_changed: [R/reco_ml.R]
test_result: { passed: int, failed: int }
numerical_check: { max_abs_diff: 0.0 }
```

## VI. Definition of Done
- [ ] base unname replaced with conditional in-place pattern
- [ ] 85/85 tests pass
- [ ] Numerical output byte-identical
