# Active Plan
<!-- approved: 2026-05-12 -->
<!-- gate-iterations: 3 -->
<!-- post-approval revisions: T6 auto/true/false memory-aware semantics -->
<!-- user-approved: pending -->
<!-- status: in-progress (T1-T4 landed) -->

# Epic: Reduce ctrml/csrml/terml runtime memory consumption

## Goal
Cut peak runtime memory of FoRecoML reconciliation functions by ≥5x for medium hierarchies (nb~50, m=12), without changing public API (except 1 opt-in arg) or numerical results.

## Success Criteria
- [ ] Peak RSS during ctrml() training reduced ≥5x on benchmark (nb=50, m=12, N=200, randomForest)
- [ ] Peak RSS during ctrml() training reduced ≥5x on benchmark with xgboost
- [ ] Peak RSS during ctrml() training reduced ≥10x with checkpoint_dir + randomForest nb=100
- [ ] All existing tests in tests/testthat/ pass unchanged
- [ ] Numerical output of reco_mat byte-identical (or within 1e-12) to baseline for fixed seed
- [ ] No regression in csrml() / terml() peak memory
- [ ] `features="compact"` path reduces hat storage proportional to compactness

## Context & Background
Source: R/{ctrml,csrml,terml,reco_ml,utils}.R. Total ~2200 LOC.

Memory analysis identified 6 high-impact opportunities ranked by ROI:

| # | Patch | Sub-Epic | Risk | Impact |
|---|---|---|---|---|
| 1 | Remove unconditional `as.data.frame(hat)` in rml() | Easy refactors | Low | 1-2x hat |
| 2 | Column-wise NA detection (drop `is.na(hat)` full alloc) | Easy refactors | Low | 1x hat transient |
| 3 | Flip `store_benchmark_result` default FALSE | Easy refactors | Low | 100s MB on tuning |
| 4 | Precompute keep_cols, slice hat once before lapply | Compact pruning | Med | (p-1)/p of hat for compact |
| 5 | Slice-first input2rtw (no row-replication of unused cols) | Input virtualization | High | (p-1)/p of training matrix |
| 6 | Optional disk checkpoint of fits (qs2 + lgb.save dispatch) | Fit streaming | Med-High | O(p) accumulated models |

Baseline: ctrml/csrml/terml → wrapper sel_mat build → rml() engine → per-series lapply(p) {build X, train model} → rbind out → return list of p fits + bts.

p = nb (cs/ct) or 1 (te). For ctrml with nb=100 + randomForest ntree=500, p models + training-data backends pinned by mlr3/xgboost → multi-GB peak.

## Dependencies (between tickets)
- T1 blocks T4, T5, T6 (matrix slicing must be cheap)
- T4 blocks T5 (T5 builds on keep_cols mechanism)
- T6 independent of T4/T5 (only depends on T1)

## New dependency
- T6 adds `qs2` to DESCRIPTION Imports. No other new deps.

## Sub-Epic Structure
- **EP-A Easy Refactors**: patches 1, 2, 3. Zero API change. Land first.
- **EP-B Compact Column Pruning**: patch 4. Internal API refactor.
- **EP-C Input Virtualization**: patch 5. Touches utils.R input2rtw + 6 call sites.
- **EP-D Fit Streaming**: patch 6. Adds opt-in `checkpoint_dir` arg + qs2 dep.

## Sub-Agent Strategy
Each task ticket is hermetic. Sub-agent reads ticket only.
Required reading per task: file:line citations + reference snippets included in ticket.

Global guards:
- No public API breaking change EXCEPT EP-D adds one optional argument.
- Numerical equivalence check: snapshot reco_mat with set.seed(42) before+after.
- Run `devtools::test()` after each patch. All pass.
- Match existing styling (snake_case, FoReco cli_abort idiom, 80-col).
- Touch ONLY files listed in ticket. No drive-by edits.
# Epic: EP-A Easy Refactors (zero API change)

## Goal
Land 3 low-risk memory wins: drop unconditional data.frame coercion, column-wise NA detection, flip mlr3 tuning benchmark default to FALSE.

## Success Criteria
- [ ] Patch 1: `rml()` no longer calls `as.data.frame(hat)` unconditionally
- [ ] Patch 2: `na_var` computed without materializing `is.na(hat)`
- [ ] Patch 3: `tuning$store_benchmark_result` defaults to FALSE
- [ ] All existing tests pass
- [ ] No public API change

## Context & Background
Files: R/reco_ml.R (patches 1, 3), R/{ctrml,csrml,terml,ctrml_fit,csrml_fit,terml_fit}.R (patch 2).

Each finding is independent and can land in any order. Lowest risk first.

## Sub-Agent Strategy
3 atomic tasks. Each names exact file:line. Each ends with `devtools::test()` pass + targeted bench script.
# Epic: EP-B Compact-mode column pruning

## Goal
For `features="compact"` (and other sparse-sel modes), avoid holding full N*m x n*p training matrix when each per-series model uses only a small column subset.

## Success Criteria
- [ ] `keep_cols` union computed from `sel_mat` once before per-series lapply
- [ ] `hat` (and `base`) subset to keep_cols globally; per-i `id` remapped to local positions
- [ ] Memory profile for `ctrml(features="compact")` drops by factor approx (p-1)/p
- [ ] All existing tests pass; numerical output unchanged

## Context & Background
Currently `rml()` holds full hat across the lapply closure even when sel_mat selects only a few columns per series. With compact for ctrml, sel_mat covers ~n columns per HFBTS out of n*p total.

Touches: R/reco_ml.R rml() entry block (lines 35-50 region) plus the lapply id resolution branch.

## Sub-Agent Strategy
Single task. Hermetic. Includes baseline snippet and target snippet.
# Epic: EP-C input2rtw virtualization

## Goal
Replace dense row-replication in `input2rtw()` with a row-index mapping that materializes only the columns each per-series model needs. Eliminates p-fold blow-up of training matrix for ctrml/terml.

## Success Criteria
- [ ] `input2rtw()` returns a structure (or list with $hat + $row_idx + $col_groups) that does NOT materialize full N*m rows × n*p cols
- [ ] OR: drop input2rtw entirely; build per-series X via on-the-fly row-replication during lapply
- [ ] `ctrml()` and `terml()` peak memory drops ≥ 3x for typical (m=12, p=6)
- [ ] All existing tests pass; numerical output unchanged

## Context & Background
File: R/utils.R `input2rtw()`. Called in ctrml.R, terml.R (training + base paths).
Current impl: FoReco::FoReco2matrix splits by kset; then `apply(x[[i]], 2, rep, each=kset[i])` row-replicates each level; finally `do.call(cbind, rev(x))` horizontally stacks.

Output rows = N*m. Output cols = sum across levels of n_k. For ctrml: cols = n*p.

The row-replication is deterministic and can be expressed as integer index map. Same result via `mat[row_idx[[i]], ]` views.

Dependent on EP-B landing first (compact slicing uses keep_cols indexing).

## Sub-Agent Strategy
Single task. Most invasive of the planned set. Requires before/after numerical equivalence check on a fixed-seed simulation case from man/csrml.Rd example.
# Epic: EP-D Fit streaming / checkpointing (user-reported worst offender)

## Goal
Eliminate peak memory dominated by all p fitted models held simultaneously by adding an opt-in disk checkpoint mode. After each per-series fit, serialize via approach-specific dispatcher (qs2 for R-native + xgboost raw bytes; lightgbm's native lgb.save for lgb.Booster), free in-memory model, gc, continue.

## Success Criteria
- [ ] New optional argument `checkpoint_dir` (default NULL = current behavior) on all 6 entry points
- [ ] When set: per-series fit serialized to `checkpoint_dir/fit_<i>.{qs2|lgb}` then dropped; lazy-load on predict
- [ ] Approach-specific serializer dispatch:
      * randomForest → qs2::qs_save / qs_read (.qs2)
      * xgboost → xgb.save.raw → qs2 (.qs2)
      * lightgbm → lgb.save / lgb.load (.lgb)
      * mlr3 → qs2 if round-trip clean, else cli_abort guard (Outcome A vs B in T6)
- [ ] Returned `rml_fit` object stores file paths; predict path lazy-loads
- [ ] `extract_reconciled_ml()` unchanged
- [ ] Peak memory for ctrml(randomForest, nb=100, ntree=500) reduced ≥10x
- [ ] All existing tests pass with checkpoint_dir = NULL
- [ ] New test-checkpoint.R covers all 4 approaches via outcome dispatch
- [ ] qs2 added to DESCRIPTION Imports

## Context & Background
User-identified as worst offender. Per-series models accumulate in `out` across lapply → returned object. p = nb (cs/ct) × randomForest ntree=500 → multi-GB peak.

qs2 chosen over saveRDS for serialization speed (~4–10x faster on R-native objects) and smaller files (zstd, ~2x smaller). lightgbm cannot use qs2 (external C++ pointer requires lgb.save). xgboost's xgb.Booster is wrapped via xgb.save.raw → raw vector → qs2.

API surface: 1 new optional argument. No behavioral change when NULL.

## Sub-Epic Strategy
Single task T6. Largest scope. Touches:
- DESCRIPTION (Imports: qs2)
- R/reco_ml.R rml() per-series loop + return path + predict branch
- R/utils.R new helpers: serialize_fit, deserialize_fit, get_fit_i
- R/{csrml,terml,ctrml}.R wrappers (6 entry points, pass-through arg)
- man/{csrml,terml,ctrml}.Rd (roxygen regen)
- tests/testthat/test-checkpoint.R (new)
# T1: Drop unconditional as.data.frame(hat) in rml()
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** Remove unconditional `hat <- as.data.frame(hat)` coercion in `rml()`; coerce per-approach only where required (randomForest, mlr3); leave xgboost/lightgbm operating on the original numeric matrix.
* **Why:** data.frame coercion duplicates training matrix (~1.5–2x) and breaks contiguity. xgboost/lightgbm then coerce back via `as.matrix(X)`. Net 3x peak. Primary, lowest-risk memory win.
* **Reference Data:** R/reco_ml.R lines ~33–53 (rml entry block) + per-approach handlers rml.randomForest, rml.mlr3, rml.xgboost, rml.lightgbm.
* **Philosophy:** Sub-agent = Goldfish Memory. All info here.

Current code (R/reco_ml.R approx lines 33–53):
```r
if (is.null(fit)) {
    hat <- unname(hat)
    hat <- as.data.frame(hat)
    obs <- unname(obs)
    p <- NCOL(obs)
} else {
    p <- length(fit$fit)
}

if (!is.null(base)) {
    base <- unname(base)
    base <- as.data.frame(base)
}
```

Per-approach `X` ingestion:
- rml.randomForest: `randomForest(y=y, x=X, ...)` — accepts matrix or df.
- rml.xgboost: `train <- xgb.DMatrix(data = as.matrix(X), label = y)` — wants matrix.
- rml.lightgbm: `train <- lgb.Dataset(data = as.matrix(X), label = y)` — wants matrix.
- rml.mlr3: `tsk_i <- cbind(y = y, X); mlr3::as_task_regr(tsk_i, target = "y")` — needs df-like.

## II. Input Specification
* **Expected Input:** R source file R/reco_ml.R only.
* **Format:** R code edits via Edit tool.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **Logic** | Numerical output of all 4 approaches MUST be identical (set.seed fixed) before/after. |
| **Format** | Match existing 2-space indent, snake_case, FoReco cli idiom. |
| **Boundary** | Touch ONLY R/reco_ml.R. No changes to wrappers or utils.R. |
| **Tone** | Code only. No new comments beyond what was there. |
| **API** | No public API change. |

## IV. Step-by-Step Logic
1. In `rml()` entry block, change:
   - Keep `hat <- unname(hat)` and `obs <- unname(obs)`.
   - Remove `hat <- as.data.frame(hat)`.
   - Remove `base <- as.data.frame(base)` block.
2. In `rml()` lapply body, at the `X <- hat[, id, drop = FALSE]` line: X is now a numeric matrix subset. Keep that as matrix.
3. In rml.mlr3 handler: replace `tsk_i <- cbind(y = y, X)` with `tsk_i <- data.frame(y = y, X, check.names = FALSE)` so the task receives a data.frame directly. Same for the block_sampling branch.
4. In rml.randomForest: pass `x = X` as matrix (already supported by randomForest pkg).
5. In rml.xgboost / rml.lightgbm: drop the `as.matrix(X)` wrapper; X is already numeric matrix.
6. In `na.omit(X)` inside lapply: `na.omit()` accepts matrix; preserves `na.action` attribute. Verify by inspection.
7. Run `devtools::test()`. All 3 test files must pass.
8. Run reproducibility check (see V.).

## V. Output Schema (Strict)
Sub-agent MUST return:
```toon
task_id: T1
success: bool
files_changed:
  - path: R/reco_ml.R
    lines_removed: int
    lines_added: int
diff: |
  <unified diff block>
test_result:
  passed: int
  failed: int
  output: <last 20 lines>
repro_check:
  baseline_seed: 42
  approaches_tested: [randomForest, xgboost, lightgbm, mlr3]
  max_abs_diff: float    # must be < 1e-12
error_log: null | msg
```

## VI. Definition of Done
- [ ] No unconditional `as.data.frame(hat)` in rml()
- [ ] No `as.matrix(X)` wrappers in xgboost/lightgbm handlers
- [ ] mlr3 handler still produces valid task object
- [ ] `devtools::test()` 100% pass
- [ ] Numerical equivalence verified for all 4 approaches via fixed seed
- [ ] No new lint warnings
# T2: Column-wise NA detection
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** Replace `na_var <- colSums(is.na(hat)) >= 0.75 * NROW(hat)` with column-wise iteration that does NOT allocate a logical matrix the size of `hat`.
* **Why:** `is.na(hat)` materializes a logical matrix the size of (already row-expanded) `hat`. For ctrml this is 1× of the large training matrix transiently held.
* **Reference Data:** Pattern appears 6 times: R/csrml.R (csrml + csrml_fit), R/terml.R (terml + terml_fit), R/ctrml.R (ctrml + ctrml_fit).
* **Philosophy:** Sub-agent = Goldfish Memory.

Current pattern (identical in all 6 places):
```r
# Remove NA variables from sel_mat
na_var <- colSums(is.na(hat)) >= 0.75 * NROW(hat)
if (any(na_var)) { ... }
```

Target pattern:
```r
na_var <- vapply(
  seq_len(NCOL(hat)),
  function(j) mean(is.na(hat[, j])) >= 0.75,
  logical(1)
)
if (any(na_var)) { ... }
```

For matrix input, `hat[, j]` returns numeric vector (1 col). For Matrix-package sparse Matrix this still works; isolate one column at a time. Peak transient = 1 column instead of full hat.

## II. Input Specification
* **Expected Input:** R source files R/csrml.R, R/terml.R, R/ctrml.R.
* **Format:** R code edits via Edit tool, applied to 6 occurrences.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **Logic** | Resulting `na_var` logical vector must be identical to current behavior (verify on test fixture with NAs). |
| **Format** | Match existing 2-space indent, snake_case. |
| **Boundary** | Touch ONLY R/csrml.R, R/terml.R, R/ctrml.R. |
| **Tone** | Surgical. No drive-by edits. |
| **API** | No public API change. |

## IV. Step-by-Step Logic
1. Grep each file: `grep -n "colSums(is.na(hat))" R/{csrml,terml,ctrml}.R`. Expect 2 hits per file.
2. Replace each with `vapply` form above.
3. Run `devtools::test()`. All pass.
4. Add a small regression check: build a fixture with one all-NA column; verify `na_var[that_col]` is TRUE post-patch. (Add as expectation inside an existing test file at end, or as a manual inline `testthat::test_that` block.)
5. Verify no other use of `is.na(hat)` introduced.

## V. Output Schema (Strict)
```toon
task_id: T2
success: bool
files_changed:
  - path: R/csrml.R
    sites: 2
  - path: R/terml.R
    sites: 2
  - path: R/ctrml.R
    sites: 2
diff: |
  <unified diff>
test_result:
  passed: int
  failed: int
regression_check:
  na_col_detected: bool
error_log: null | msg
```

## VI. Definition of Done
- [ ] 6 occurrences replaced
- [ ] No `colSums(is.na(hat))` remaining in package
- [ ] devtools::test() passes
- [ ] Regression fixture catches all-NA column
# T3: Flip store_benchmark_result default to FALSE in rml.mlr3
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** Change default of `tuning$store_benchmark_result` in `rml.mlr3()` from TRUE to FALSE. User can still override via tuning list.
* **Why:** With tuning enabled (e.g., 20 evals × 5 CV folds), keeping all benchmark archives in memory holds hundreds of MB per series. Default should be memory-frugal.
* **Reference Data:** R/reco_ml.R rml.mlr3 handler, default-setting block.
* **Philosophy:** Sub-agent = Goldfish Memory.

Current code (R/reco_ml.R approx lines 152-156):
```r
if (is.null(tuning$store_benchmark_result)) {
  tuning$store_benchmark_result <- TRUE
}
if (is.null(tuning$store_models)) {
  tuning$store_models <- FALSE
}
```

Target:
```r
if (is.null(tuning$store_benchmark_result)) {
  tuning$store_benchmark_result <- FALSE
}
if (is.null(tuning$store_models)) {
  tuning$store_models <- FALSE
}
```

## II. Input Specification
* **Expected Input:** R source file R/reco_ml.R only.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **Logic** | User-supplied `tuning$store_benchmark_result = TRUE` MUST still be honored. |
| **Format** | One-character semantic edit (TRUE → FALSE). |
| **Boundary** | Only R/reco_ml.R rml.mlr3 default block. No other defaults change. |
| **API** | No public API change (default-only). |

## IV. Step-by-Step Logic
1. Edit `R/reco_ml.R`: change `tuning$store_benchmark_result <- TRUE` to `... <- FALSE` inside the `is.null(tuning$store_benchmark_result)` branch.
2. Verify tuning override still works: add inline expectation in mlr3 test (if absent) that user-set TRUE stays TRUE.
3. Run devtools::test().

## V. Output Schema (Strict)
```toon
task_id: T3
success: bool
files_changed:
  - path: R/reco_ml.R
    line: <line_number>
diff: |
  <unified diff>
test_result:
  passed: int
  failed: int
override_check: bool
error_log: null | msg
```

## VI. Definition of Done
- [ ] Default value is FALSE
- [ ] User override path still works
- [ ] devtools::test() passes
# T4: Precompute keep_cols and slice hat once before lapply
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** In `rml()`, before the per-series lapply, compute the union of columns referenced by `sel_mat` and subset the active feature matrix (`hat` during training, `base` during predict) to that union exactly once. Remap each per-series `id` to local positions.
* **Why:** For `features="compact"` and any sparse sel_mat, only a small fraction of feature columns are ever needed. Slicing once before iteration drops (p−1)/p of training matrix for compact.
* **Mechanism:** Single shared col_map built against the active feature matrix; predict-mode handled explicitly.
* **Forbidden:** No alternative sparse-matrix backend; no S3 wrapper class.
* **Reference Data:** R/reco_ml.R rml() lines 30–105. T1 already removed data.frame coercion so matrix slicing is cheap.

Current id-resolution in lapply (R/reco_ml.R lines 50–56):
```r
if (length(sel_mat) == 1) {
    id <- seq_len(max(NCOL(hat), NCOL(base)))
} else if (is(sel_mat, "sparseVector") | NCOL(sel_mat) == 1) {
    id <- which(sel_mat == 1)
} else {
    id <- which(sel_mat[, i] == 1)
}
```

Predict-mode reminder (R/reco_ml.R lines 30–42): when `!is.null(fit)`, `hat` is NULL. The wrapper supplied `base` and `sel_mat <- fit$sel_mat`. Per-series body reads only `base[, id]`.

## II. Input Specification
* **Expected Input:** R/reco_ml.R only.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **Logic** | Per-series X (training) and Xtest (predict) MUST be byte-identical (column order included) to current behavior. |
| **Format** | Match current style. |
| **Boundary** | Only R/reco_ml.R. sel_mat structure stored in returned object unchanged. |
| **API** | No public API change. |
| **Dependency** | Land AFTER T1. |

## IV. Step-by-Step Logic
1. Define helper inside `rml()` (after `p` is set, before lapply):
   ```r
   active_ncol <- if (!is.null(hat)) NCOL(hat) else NCOL(base)
   keep_cols <- if (length(sel_mat) == 1) {
       seq_len(active_ncol)
   } else if (is(sel_mat, "sparseVector")) {
       which(as.numeric(sel_mat) != 0)
   } else if (NCOL(sel_mat) == 1) {
       which(as.numeric(sel_mat[, 1]) != 0)
   } else {
       which(Matrix::rowSums(sel_mat != 0) > 0)
   }
   ```
2. Short-circuit: if `length(keep_cols) == active_ncol` AND `length(sel_mat) == 1`, skip slicing (no benefit).
3. Else, slice:
   ```r
   col_map <- rep(NA_integer_, active_ncol)
   col_map[keep_cols] <- seq_along(keep_cols)
   if (!is.null(hat))  hat  <- hat[,  keep_cols, drop = FALSE]
   if (!is.null(base)) base <- base[, keep_cols, drop = FALSE]
   ```
4. In lapply body, replace id-resolution:
   ```r
   global_id <- if (length(sel_mat) == 1) {
       keep_cols
   } else if (is(sel_mat, "sparseVector") || NCOL(sel_mat) == 1) {
       which(as.numeric(if (is(sel_mat, "sparseVector")) sel_mat else sel_mat[, 1]) != 0)
   } else {
       which(sel_mat[, i] != 0)
   }
   id <- col_map[global_id]
   id <- id[!is.na(id)]
   ```
5. Predict-mode branch: when `is.null(hat)`, `hat[, keep_cols]` is skipped per step 3 conditional; base sliced. Verify the per-series `Xtest <- base[, id, drop=FALSE]` path produces identical content as pre-patch.
6. Run `devtools::test()`.
7. Run numerical equivalence check (Step V).

## V. Output Schema (Strict)
```toon
task_id: T4
success: bool
files_changed:
  - path: R/reco_ml.R
    lines_added: int
    lines_removed: int
diff: |
  <unified diff>
test_result: { passed: int, failed: int }
numerical_check:
  wrapper: csrml
  approaches: [randomForest, xgboost, lightgbm, mlr3]
  features: [all, bts, str, str-bts]
  max_abs_diff: float    # < 1e-12 each
numerical_check_terml:
  wrapper: terml
  approaches: [randomForest, xgboost, lightgbm, mlr3]
  features: [all, mfh-all, mfh-str, mfh-str-hfts, low-high]
  max_abs_diff: float    # < 1e-12 each
numerical_check_ctrml:
  wrapper: ctrml
  approaches: [randomForest, xgboost, lightgbm, mlr3]
  features: [all, mfh-all, mfh-str-bts, mfh-str-hfbts, compact]
  max_abs_diff: float    # < 1e-12 each
predict_reuse_check:
  trained_then_predict_new_base: bool
  max_abs_diff: float    # < 1e-12
error_log: null | msg
```

## VI. Definition of Done
- [ ] keep_cols + col_map computed once outside lapply
- [ ] hat sliced once when non-NULL; base sliced once when non-NULL
- [ ] Predict-mode (`hat=NULL`) path verified: no NULL-slicing crash; Xtest content identical
- [ ] Per-series X and Xtest byte-identical to baseline across all approaches × all feature modes for all 3 wrappers
- [ ] `csrml(fit = mdl, base = base_new)` reuse path produces identical reconciled output before/after
- [ ] devtools::test() passes
- [ ] Memory profile: ctrml(features="compact") shows ≥ 3x reduction on benchmark fixture
# T5: Slice-first input2rtw (no row-replication of unused columns)
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** Eliminate the dense row-replicated N*m × n*p training matrix produced by `input2rtw()` when only a subset of columns is ever used. Implement a single concrete mechanism: `input2rtw_partial(x, kset, cols)` materializes ONLY the columns whose global index is in `cols`. Wrappers compute `keep_cols` from `sel_mat` and pass it down.
* **Why:** input2rtw row-replication inflates training matrix ≈ p-fold. With T4's keep_cols typically << total cols (especially under "compact"), building only those columns drops the largest single allocation.
* **Mechanism:** Slice-first. No S3 virtualization, no lazy views, no `materialize` flag.
* **Forbidden:** `virtual_rtw` S3 class; `materialize` parameter; lazy view of input2rtw return.
* **Reference Data:**
  - R/utils.R:95-107 `input2rtw()` definition.
  - **All 6 call sites (verified via grep):**
    - R/ctrml.R:275 `hat  <- input2rtw(hat,  tmp$set)`  — ctrml() training path
    - R/ctrml.R:398 `base <- input2rtw(base, tmp$set)`  — ctrml() base-input path (also used in predict-reuse)
    - R/ctrml.R:533 `hat  <- input2rtw(hat,  tmp$set)`  — ctrml_fit() training path
    - R/terml.R:209 `hat  <- input2rtw(hat,  kset)`     — terml() training path
    - R/terml.R:297 `base <- input2rtw(base, kset)`     — terml() base-input path
    - R/terml.R:418 `hat  <- input2rtw(hat,  kset)`     — terml_fit() training path

Current input2rtw:
```r
input2rtw <- function(x, kset) {
  x <- FoReco::FoReco2matrix(x, kset)
  x <- lapply(1:length(kset), function(i) {
    if (NCOL(x[[i]]) > 1) {
      tmp <- apply(x[[i]], 2, rep, each = kset[i])
    } else {
      tmp <- rep(x[[i]], each = kset[i])
    }
    tmp
  })
  do.call(cbind, rev(x))
}
```

## II. Input Specification
* **Expected Input:** R/utils.R, R/reco_ml.R, R/ctrml.R, R/terml.R.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **Logic** | Full-cols call `input2rtw_partial(x, kset, cols = seq_len(total_cols))` MUST equal legacy `input2rtw(x, kset)` byte-identically. |
| **Format** | Match existing style. |
| **Boundary** | utils.R + reco_ml.R + ctrml.R + terml.R only. csrml.R UNTOUCHED (no input2rtw use; verified via grep). |
| **API** | Public API unchanged. Legacy `input2rtw()` retained as 1-line wrapper for any external callers (none in this pkg). New internal helper `input2rtw_partial()` added. |
| **Site coverage** | All 6 call sites MUST be migrated. Listed in §I above. |
| **Dependency** | Land AFTER T1 + T4. Uses T4's keep_cols logic. |

## IV. Step-by-Step Logic
1. Add to R/utils.R:
   ```r
   # cols: integer vector of column indices in the FULL row-replicated output (1..total_cols).
   input2rtw_partial <- function(x, kset, cols) {
     parts <- FoReco::FoReco2matrix(x, kset)
     ncol_per_level_rev <- vapply(rev(parts), NCOL, integer(1))
     col_offsets <- c(0L, cumsum(ncol_per_level_rev))
     out_blocks <- lapply(seq_along(ncol_per_level_rev), function(j) {
       lvl_idx_full <- length(kset) - j + 1L
       in_range <- cols > col_offsets[j] & cols <= col_offsets[j + 1]
       if (!any(in_range)) return(NULL)
       local <- cols[in_range] - col_offsets[j]
       block <- parts[[lvl_idx_full]][, local, drop = FALSE]
       k <- kset[lvl_idx_full]
       expanded <- if (NCOL(block) > 1) apply(block, 2, rep, each = k)
                   else matrix(rep(block, each = k), ncol = 1)
       list(mat = expanded, global_cols = cols[in_range])
     })
     out_blocks <- Filter(Negate(is.null), out_blocks)
     mat <- do.call(cbind, lapply(out_blocks, `[[`, "mat"))
     global_cols <- unlist(lapply(out_blocks, `[[`, "global_cols"))
     mat[, order(match(global_cols, cols)), drop = FALSE]
   }

   # Helper: compute keep_cols from sel_mat. Mirrors T4's logic; shared with rml().
   sel_mat_keep_cols <- function(sel_mat, ncol_full) {
     if (length(sel_mat) == 1) return(seq_len(ncol_full))
     if (is(sel_mat, "sparseVector")) return(which(as.numeric(sel_mat) != 0))
     if (NCOL(sel_mat) == 1) return(which(as.numeric(sel_mat[, 1]) != 0))
     which(Matrix::rowSums(sel_mat != 0) > 0)
   }
   ```
2. Restructure ctrml.R wrapper paths so sel_mat is built BEFORE input2rtw:
   - **ctrml() training path (around lines 264-345)**: move sel_mat switch ABOVE `hat <- input2rtw(...)` at line 275. Then compute `keep_cols <- sel_mat_keep_cols(sel_mat, total_cols)` where `total_cols = tmp$dim[["n"]] * tmp$dim[["p"]]`. Call `hat <- input2rtw_partial(hat, tmp$set, cols = keep_cols)`.
   - **ctrml() base path (line 398)**: in predict-reuse branch, `sel_mat <- fit$sel_mat` is already set. Compute keep_cols from it and call `base <- input2rtw_partial(base, tmp$set, cols = keep_cols)`.
   - **ctrml_fit() training path (line 533)**: same as ctrml() training. Move sel_mat construction above input2rtw call; use input2rtw_partial.
3. Same restructure for terml.R:
   - **terml() training path (line 209)**: sel_mat above input2rtw; input2rtw_partial with keep_cols.
   - **terml() base path (line 297)**: sel_mat from fit; input2rtw_partial.
   - **terml_fit() training path (line 418)**: same as training.
4. In `rml()` (modified by T4): when hat is already sliced upstream, T4's col_map indirection still works because keep_cols passed to rml() will equal `seq_len(NCOL(hat))` (already trimmed). T4's short-circuit covers this.
5. NA-detection in ctrml.R/terml.R operates on the partially materialized hat: when columns are missing globally from the partial hat, sel_mat already excludes them (they were filtered out by keep_cols). Behavior consistent.
6. Run `devtools::test()`.
7. Numerical equivalence check (Step V).

## V. Output Schema (Strict)
```toon
task_id: T5
success: bool
files_changed:
  - path: R/utils.R
  - path: R/reco_ml.R       # only if T4 col_map adjustment needed
  - path: R/ctrml.R
  - path: R/terml.R
diff: |
  <unified diff>
test_result: { passed: int, failed: int }
identity_check:
  full_cols_equivalence: bool      # input2rtw_partial(x, kset, 1:total) == input2rtw(x, kset)
  max_abs_diff: 0.0
sites_migrated:
  ctrml.R: [275, 398, 533]
  terml.R: [209, 297, 418]
  count: 6
numerical_check_ctrml:
  approaches: [randomForest, xgboost, lightgbm, mlr3]
  features: [all, mfh-all, mfh-str-bts, mfh-str-hfbts, compact]
  max_abs_diff: float    # < 1e-12
numerical_check_terml:
  approaches: [randomForest, xgboost, lightgbm, mlr3]
  features: [all, mfh-all, mfh-str, mfh-str-hfts, low-high]
  max_abs_diff: float    # < 1e-12
predict_reuse_check:
  ctrml_fit_then_predict_new_base: bool
  terml_fit_then_predict_new_base: bool
  max_abs_diff: float    # < 1e-12
mem_check:
  ctrml_compact_peak_mb_before: float
  ctrml_compact_peak_mb_after: float
  reduction_factor: float    # ≥ 3.0
error_log: null | msg
```

## VI. Definition of Done
- [ ] `input2rtw_partial()` defined; full-cols invocation byte-identical to legacy
- [ ] `sel_mat_keep_cols()` helper defined
- [ ] All 6 input2rtw call sites migrated (ctrml.R:275/398/533 + terml.R:209/297/418)
- [ ] sel_mat construction relocated above input2rtw call in all training+base paths
- [ ] Predict-reuse (`fit=` arg) paths produce identical reconciled output
- [ ] All approaches × all feature modes numerically equivalent for ctrml + terml
- [ ] devtools::test() passes
- [ ] Peak memory reduction ≥ 3x on ctrml(features="compact") benchmark
# T6: Auto/forced/disabled disk checkpoint for per-series fits (qs2 + memory-aware default)
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** Add `checkpoint = "auto"` argument to all 6 entry points with tri-mode semantics:
  - `"auto"` (default): enable checkpoint when estimated peak memory > 80% of available physical RAM; use a session-scoped tempdir under `tempdir()`.
  - `"false"` / `FALSE`: never enable; keep all fits in memory (legacy behavior).
  - `"true"` / `TRUE`: always enable; use session-scoped tempdir under `tempdir()`.
  - character path: always enable; use that path (persistent, user-controlled).
* **Why:** User-identified worst offender = all p fits held simultaneously. Auto mode protects the common case (large hierarchies on average machines) without forcing I/O for small hierarchies. Path mode preserves the `_fit`/`fit=` reuse workflow.
* **Mechanism:** Single `checkpoint` argument; resolved at rml() entry into an effective directory path or NULL; per-approach serializer dispatch (qs2 + lgb.save); lazy-load accessor `get_fit_i()`.
* **Forbidden:** Multiple boolean arguments (use single `checkpoint`); naked `saveRDS` for xgboost/lightgbm; silent no-op on unsupported approaches.
* **Reference Data:** R/reco_ml.R rml() per-series loop + return path; R/utils.R helpers; R/{csrml,terml,ctrml}.R 6 entry points (csrml.R:156, csrml.R:308; terml.R:153, terml.R:372; ctrml.R:216, ctrml.R:479).
* **New dependency:** `qs2` (CRAN) added to `DESCRIPTION` `Imports:`.

## I.a Argument resolution semantics
```r
resolve_checkpoint <- function(checkpoint, hat, approach, p) {
  # Returns: NULL (disabled) | character(1) (absolute path)
  if (identical(checkpoint, FALSE) || identical(checkpoint, "false")) {
    return(NULL)
  }
  if (identical(checkpoint, TRUE) || identical(checkpoint, "true")) {
    return(checkpoint_session_dir())
  }
  if (is.character(checkpoint) && length(checkpoint) == 1) {
    if (checkpoint == "auto") {
      est <- estimate_peak_bytes(hat, approach, p)
      avail <- available_ram_bytes()
      if (is.finite(est) && is.finite(avail) && est > 0.8 * avail) {
        return(checkpoint_session_dir())
      }
      return(NULL)
    }
    # treated as path
    return(normalizePath(checkpoint, mustWork = FALSE))
  }
  cli_abort("`checkpoint` must be 'auto' (default), 'true'/TRUE, 'false'/FALSE, or a directory path.")
}

checkpoint_session_dir <- function() {
  d <- file.path(tempdir(), sprintf("foreco_ckpt_%s", paste0(sample(letters, 6), collapse = "")))
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  d
}
```

## I.b Memory estimation
Conservative peak estimator (over-estimates → more aggressive auto-checkpointing; acceptable):
```r
estimate_peak_bytes <- function(hat, approach, p) {
  if (is.null(hat)) return(0)   # predict mode — fits already trained
  hat_bytes <- as.numeric(NROW(hat)) * NCOL(hat) * 8
  per_model <- switch(approach,
    "randomForest" = 5,
    "mlr3"         = 3,
    "xgboost"      = 0.5,
    "lightgbm"     = 0.3,
    1
  )
  models <- hat_bytes * per_model * p
  copies <- hat_bytes * 3   # training matrix copies during fit
  models + copies
}
```

Available RAM detection (cross-platform, no new deps):
```r
available_ram_bytes <- function() {
  os <- Sys.info()[["sysname"]]
  if (os == "Linux") {
    info <- tryCatch(readLines("/proc/meminfo", n = 5), error = function(e) NULL)
    if (is.null(info)) return(NA_real_)
    line <- grep("^MemAvailable:", info, value = TRUE)
    if (length(line) == 0) line <- grep("^MemFree:", info, value = TRUE)
    if (length(line) == 0) return(NA_real_)
    kb <- as.numeric(regmatches(line, regexpr("\\d+", line)))
    return(kb * 1024)
  }
  if (os == "Darwin") {
    total <- tryCatch(as.numeric(system("sysctl -n hw.memsize", intern = TRUE)), error = function(e) NA_real_)
    return(total)   # worst-case proxy
  }
  if (os == "Windows") {
    out <- tryCatch(system("wmic OS get FreePhysicalMemory /Value", intern = TRUE),
                    error = function(e) NULL, warning = function(w) NULL)
    if (is.null(out)) return(NA_real_)
    line <- grep("FreePhysicalMemory=", out, value = TRUE)
    if (length(line) == 0) return(NA_real_)
    kb <- as.numeric(sub(".*=", "", line))
    return(kb * 1024)
  }
  NA_real_
}
```

When `available_ram_bytes()` returns NA (unknown OS), `"auto"` falls back to OFF (NULL), since the comparison `est > 0.8 * NA` is NA which is treated as "not > threshold" by the guard.

## I.c Alternatives considered (preserved + augmented)
| Alternative | Outcome | Reason |
| --- | --- | --- |
| Single `checkpoint_dir = NULL`/path | REJECTED | Forces user to estimate memory themselves; new requirement (this revision) is auto-detection. |
| Two booleans (`stream`, `path`) | REJECTED | Two args for one concept; harder to reason about. |
| Auto-only (no override) | REJECTED | Users with shared filesystems or restricted disk space need OFF; users with predictable workloads need always-ON. |
| **CHOSEN: tri-mode `checkpoint` with "auto" default** | KEPT | Auto for safety; user override for control; persistent path for `_fit`/`fit=` reuse. |

## I.d mlr3 + checkpoint policy
Same as prior revision. Round-trip via qs2 on mlr3 trained learner is learner-class-dependent. T6 test §IV.13 determines outcome:
- **Outcome A**: qs_save/qs_read on the mlr3 learner produces predictions ≤1e-12 from in-memory mode → mlr3 included in approach list.
- **Outcome B**: Round-trip fails → `cli_abort("checkpoint not supported for approach='mlr3' (learner serialization is class-dependent). Use 'randomForest','xgboost', or 'lightgbm'.")` guard.

## II. Input Specification
R/reco_ml.R, R/utils.R, R/csrml.R, R/terml.R, R/ctrml.R, DESCRIPTION, tests/testthat/test-checkpoint.R (new), regenerated man/{csrml,terml,ctrml}.Rd.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **Logic** | `checkpoint=FALSE`: bit-identical to pre-patch. `checkpoint=TRUE`/path: predictions identical ≤1e-12 across all supported approaches. `checkpoint="auto"`: behavior matches "false" or "true" branch depending on memory estimate; produces identical predictions to in-memory mode. |
| **Format** | Match roxygen idiom. |
| **Boundary** | 5 source files + DESCRIPTION + 1 new test + 3 .Rd. |
| **API** | One optional argument `checkpoint = "auto"`. Backward compatible: prior behavior reachable via `checkpoint = FALSE`. |
| **External pointers** | xgboost: `xgb.save.raw` → raw vec → qs2. lightgbm: native `lgb.save`/`lgb.load`. |
| **New dep** | qs2 listed in DESCRIPTION Imports. |
| **OS fallback** | Unknown OS (available_ram_bytes returns NA) → "auto" falls back to OFF. |
| **Dependency** | After T1. |

## IV. Step-by-Step Logic
1. Add `qs2` to `DESCRIPTION` `Imports:` (alphabetical).
2. Add `checkpoint = "auto"` parameter to all 6 entry points; pass to `rml()`.
3. In `rml()` entry: `checkpoint_dir <- resolve_checkpoint(checkpoint, hat, approach, p)`. Continue with `checkpoint_dir` (NULL or path) as before.
4. If `!is.null(checkpoint_dir) && approach == "mlr3"`: per §I.d.
5. Per-iteration: train → if `checkpoint_dir` set: `serialize_fit()` → replace in-memory model with path; `rm(); gc()`.
6. Add `serialize_fit(model, dir, i, approach)` to R/utils.R:
   ```r
   serialize_fit <- function(model, dir, i, approach) {
     ext <- if (approach == "lightgbm") ".lgb" else ".qs2"
     path <- file.path(dir, sprintf("fit_%d%s", i, ext))
     switch(approach,
       "xgboost"  = qs2::qs_save(xgboost::xgb.save.raw(model), path),
       "lightgbm" = lightgbm::lgb.save(model, filename = path),
       qs2::qs_save(model, path)
     )
     path
   }
   ```
7. Add `deserialize_fit(path, approach)`:
   ```r
   deserialize_fit <- function(path, approach) {
     switch(approach,
       "xgboost"  = xgboost::xgb.load.raw(qs2::qs_read(path)),
       "lightgbm" = lightgbm::lgb.load(filename = path),
       qs2::qs_read(path)
     )
   }
   ```
8. Add `get_fit_i(obj, i)` to R/utils.R; use `obj$approach`.
9. `rml()` predict branch: `fit_i <- get_fit_i(fit, i)`.
10. `new_rml_fit()`: store paths-or-objects. Add `checkpoint_dir` field (NULL or path).
11. **DO NOT modify `print.rml_fit()`.**
12. Roxygen on all 6 entry points:
    ```
    @param checkpoint One of `"auto"` (default), `"true"`/`TRUE`, `"false"`/`FALSE`,
      or a directory path. Controls whether per-series fits are serialized to disk
      and lazy-loaded at predict (caps peak memory at one model). `"auto"`
      enables checkpointing when estimated peak memory exceeds 80% of available
      physical RAM, falling back to off on unknown platforms. `TRUE`/`"true"`
      forces checkpointing to a session-scoped temp directory. `FALSE`/`"false"`
      disables it. A path forces checkpointing to that directory (persistent across
      sessions). Serializers: `qs2::qs_save` for R-native (randomForest); raw bytes
      via `xgboost::xgb.save.raw` + `qs2` for xgboost; `lightgbm::lgb.save` for
      lightgbm. See Note for `approach="mlr3"`.
    ```
13. Tests in `tests/testthat/test-checkpoint.R`:
    - Round-trip for randomForest, xgboost, lightgbm via `checkpoint=TRUE` (forced)
    - mlr3 outcome dispatch (Outcome A vs B)
    - `checkpoint=FALSE` byte-identical to pre-patch (omitting checkpoint arg)
    - `checkpoint="auto"` returns NULL when estimated peak < threshold (use a tiny fixture)
    - `checkpoint="auto"` returns a path when fake-RAM forced low (mock `available_ram_bytes` if testable, else skip)
    - Argument-resolution unit tests on `resolve_checkpoint()` itself
    - File extension dispatch: .qs2 vs .lgb
    - predict-reuse path with `fit=` works on a checkpointed fit
14. Run `devtools::document()` + `devtools::test()`.

## V. Output Schema (Strict)
```toon
task_id: T6
success: bool
files_changed:
  - DESCRIPTION
  - R/reco_ml.R
  - R/utils.R
  - R/csrml.R
  - R/terml.R
  - R/ctrml.R
  - tests/testthat/test-checkpoint.R
  - man/csrml.Rd
  - man/terml.Rd
  - man/ctrml.Rd
new_dep: { pkg: qs2, added_to: Imports }
checkpoint_modes_verified: [auto, true, false, path]
checkpoint_round_trip:
  approaches_supported: [randomForest, xgboost, lightgbm, ...]
  mlr3_outcome: A | B
  max_abs_diff_per_approach: float    # < 1e-12
  predict_reuse_after_checkpoint: bool
file_extension_check:
  randomForest: .qs2
  xgboost: .qs2
  lightgbm: .lgb
auto_detection_check:
  small_fixture_auto_returns_off: bool
  threshold_calculation_unit_tested: bool
mem_check:
  ctrml_rf_peak_mb_before: float
  ctrml_rf_peak_mb_after_forced: float
  reduction_factor: float    # ≥ 10 with checkpoint=TRUE
test_result: { passed: int, failed: int }
error_log: null | msg
```

## VI. Definition of Done
- [ ] qs2 in DESCRIPTION Imports
- [ ] `checkpoint` argument with tri-mode semantics (auto/true/false/path)
- [ ] `resolve_checkpoint()` helper unit-tested
- [ ] `estimate_peak_bytes()` + `available_ram_bytes()` helpers implemented; OS fallback to NA → "auto" off
- [ ] checkpoint=FALSE: bit-identical to pre-patch
- [ ] checkpoint=TRUE/path: predictions identical (≤1e-12) for randomForest, xgboost, lightgbm
- [ ] mlr3 outcome A or B documented
- [ ] File extensions correct (.qs2 / .lgb)
- [ ] All existing tests pass
- [ ] New test-checkpoint.R covers all modes + 4 approaches
- [ ] Roxygen regenerated
- [ ] Memory reduction ≥10x on ctrml randomForest nb=100 ntree=500 with checkpoint=TRUE
- [ ] `print.rml_fit()` unchanged
