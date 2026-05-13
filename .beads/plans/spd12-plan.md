## I. Objective

In: rml() expands per-series hat/base inside loop_body via input2rtw_partial. Daemons receive raw hat (small), not expanded hat (huge).
Out: orchestrator-side hat row-expansion eliminated for **non-mfh path** (compact, all) in ctrml + terml. Column-slicing via sel_mat_keep_cols stays in wrapper (preserves features_size storage). Only `input2rtw_partial` (row-replication) is deferred.
Scope: csrml UNCHANGED. mfh-* features UNCHANGED in spd.12 (mat2hmat path; spd.13).

## II. Input

- R/reco_ml.R rml() (11 named formals) + loop_body (13 formals) + line 158 unconditional Xtest. `# 13-item closure list spec` comment at line 114.
- R/ctrml.R sites: 392, 474, 694 (input2rtw_partial); NA-zeroing block at 397-426. Verify via fresh grep.
- R/terml.R sites: 258, 330, 502 (input2rtw_partial); mirror NA block. Verify via fresh grep.
- R/csrml.R: 0 input2rtw_partial calls; 0 references to kset. UNTOUCHED.

## III. Guards

- **Mechanism:** rml() gains `kset = NULL` formal; loop_body gains 14th formal `kset`; branch on `is.null(kset)`. ONLY `input2rtw_partial` ROW-EXPANSION moves into loop_body; `sel_mat_keep_cols` COLUMN-SLICING remains in wrapper.
- **Forbidden:** touching mfh path; touching csrml; mutating closure-captured sel_mat; touching `sel_mat_keep_cols` invocation; touching `keep_cols`/`features_size`/`new_rml_fit`.
- **Audit:** mandatory numerical equivalence gate (max_abs_diff = 0.0) vs baseline qs2 snapshots from HEAD a90fd30. Matrix includes 1 NA-injected fixture.
- **Predict-reuse Xtest (B1 fix):** rewrite reco_ml.R:158 to dispatch:
  - `is.null(kset)`: `Xtest <- base[, id, drop = FALSE]` (csrml + mfh fallthrough)
  - else: `Xtest <- input2rtw_partial(base, kset, cols = global_id_post_na)` (ctrml/terml non-mfh)
- **NA semantics (B1+B2 fix — Option A: per-series local NA drop):**
  - Pre-spd.12 wrapper: detects all-NA expanded columns, zeros sel_mat ROWS at those global indices, applies GLOBALLY across all series.
  - Post-spd.12 loop_body: detects all-NA expanded columns IN PER-ITER X (after kset-path input2rtw_partial), drops them from `global_id_post_na` LOCAL TO SERIES i.
  - **Equivalence proof:** if column j of expanded hat is all-NA, every series i whose `global_id` contains j detects it and drops it. Series whose `global_id` does NOT contain j are unaffected (they would not have selected j anyway). Resulting trained per-series feature set is identical: `{j ∈ global_id_i : col_j(expanded_hat) not all-NA} == old behavior`. No cross-series contamination possible because NA-column membership is a property of expanded_hat columns, not of sel_mat.
  - **DoD enforces equivalence:** synthetic NA-injected fixture (DoD step 2.c) catches any semantic deviation.

## IV. Logic

### 1. R/reco_ml.R — rml() + loop_body

a. rml() signature: insert `kset = NULL` after existing named formals, before `...`.

b. loop_body formals (14): `function(i, hat, obs, base, sel_mat, col_map, class_base, approach, active_ncol, params, fit, checkpoint_dir, kset, dots)`.

c. Update comment at line 114: replace `# 13-item closure list spec` → `# 14-item closure list spec (kset added in spd.12)`.

d. Replace existing X/Xtest derivation:

```r
global_id <- ...   # unchanged (sel_mat-derived)
id <- if (is.null(col_map)) global_id else { x <- col_map[global_id]; x[!is.na(x)] }

global_id_post_na <- global_id   # default; trimmed below if kset path detects column NAs

if (is.null(fit)) {
  y <- obs[, i]
  if (is.null(kset)) {
    # csrml + mfh fallthrough: hat is pre-expanded; slice by local id.
    X <- hat[, id, drop = FALSE]
  } else {
    # ctrml/terml non-mfh: defer row-expansion; expand only this series' columns.
    X <- input2rtw_partial(hat, kset, cols = global_id)
    na_cols <- FoRecoML:::na_col_mask(X)
    if (any(na_cols)) {
      X <- X[, !na_cols, drop = FALSE]
      global_id_post_na <- global_id[!na_cols]
    }
  }
  fit_i <- NULL
  if (anyNA(X)) {
    X <- stats::na.omit(X)
    if (length(attr(X, "na.action")) > 0L) {
      if (NROW(X) == 0L) cli::cli_abort(...)  # existing message preserved
      y <- y[-attr(X, "na.action")]
    }
  }
} else {
  y <- X <- NULL
  fit_i <- FoRecoML:::get_fit_i(fit, i)
  # global_id_post_na stays = global_id (no NA filter on fit-reuse path)
}

if (!is.null(base)) {
  if (is.null(kset)) {
    Xtest <- base[, id, drop = FALSE]
  } else {
    Xtest <- input2rtw_partial(base, kset, cols = global_id_post_na)
  }
} else {
  Xtest <- NULL
}
```

**CRITICAL:** reco_ml.R:158 (`Xtest <- base[, id, drop = FALSE]` unconditional) is DELETED; replacement is the dispatch above. No code path may construct Xtest with stale `id` when kset is non-NULL.

e. Sequential dispatch (n_workers==1, lines 191-196): append `kset = kset` to the explicit-arg call.

f. mirai_map (.args, lines 210-215): append `kset = kset`.

### 2. R/ctrml.R

For each non-mfh site (3 total: ~392, ~474, ~694 — fresh grep mandatory):

- Compute `keep_cols <- sel_mat_keep_cols(sel_mat, total_cols)` → KEEP. (Needed for `features_size <- total_cols` and fit-reuse validation.)
- DELETE `hat <- input2rtw_partial(hat, tmp$set, cols = keep_cols)` (or per-site equivalent).
- DELETE the full NA-on-expanded-hat block at ctrml.R lines 397-426 (and the analogous blocks at the other two sites). Replicate the locations via fresh grep for `na_col_mask(hat)` inside `!grepl("mfh", features)` branches.
- Add `kset = tmp$set` to the rml() call.
- mfh branch: leave mat2hmat + NA block + sel_mat zeroing intact.

### 3. R/terml.R

Mirror of ctrml. 3 non-mfh sites (~258, ~330, ~502). Variable name `kset` already in scope (per `kset <- tmp$set`). Pass `kset = kset` to rml().

### 4. R/csrml.R

UNTOUCHED. Confirmed 0 input2rtw_partial calls; csrml never passes kset → rml() default `kset = NULL` triggers col_map fallthrough branch.

## V. Schema

```
sites:
  - file: R/reco_ml.R
    fn: rml
    delta: +kset=NULL formal; +kset=kset in sequential .args; +kset=kset in mirai_map .args
  - file: R/reco_ml.R
    fn: loop_body
    delta: +kset formal (14th); X dispatch on kset; per-iter na_col_mask on kset path; Xtest dispatch on kset
  - file: R/reco_ml.R
    line: 114
    delta: comment "# 13-item closure list spec" -> "# 14-item closure list spec"
  - file: R/reco_ml.R
    line: 158
    delta: DELETE unconditional Xtest; replaced by IV.1.d dispatch
  - file: R/ctrml.R
    sites: [~392, ~474, ~694]
    delta: per-site: keep sel_mat_keep_cols; DELETE input2rtw_partial; DELETE NA-on-expanded-hat block (lines ~397-426 at site 1; mirror at other 2 sites); add kset=tmp$set to rml()
  - file: R/terml.R
    sites: [~258, ~330, ~502]
    delta: same as ctrml; variable is `kset`
  - file: R/csrml.R
    delta: NONE
```

## VI. DoD

1. `devtools::test()` → 112/112 pass (current HEAD baseline).
2. **MANDATORY numerical equivalence gate** (B2 blocker fix; promoted from optional, NA fixture added):
   - **2.a Snapshot baseline** (pre-spd.12, on HEAD a90fd30): for each config below, save `r$bts` via qs2 to `tests/testthat/_snaps/spd12/<config>.qs2`:
     - ctrml × {compact, all} × {randomForest, xgboost, lightgbm} = 6 cases
     - terml × {low-high, all} × {randomForest, xgboost, lightgbm} = 6 cases
     - csrml × compact × lightgbm = 1 control
   - **2.b Numerical equivalence** (post-spd.12): `max(abs(new - old)) == 0` for each. Snapshot via qs2.
   - **2.c NA injection fixture**: ctrml × compact × lightgbm with hat synthetically set to `NA` for one full column of the expanded form (inject via base-forecast NA at one level). Pre- and post-spd.12 must produce byte-identical `r$bts`. Validates Option A NA semantics.
   - Encode as `tests/testthat/test-spd12-equivalence.R`. FAIL on any nonzero diff. No "close enough."
3. mw3.3 invariant intact: reco_ml.R:183 `if (is.null(fit)) list(bts, fit) else list(bts)` unchanged.
4. spd.10 in-memory fit cap unaffected. `test-parallel.R::"in-memory fit auto-caps n_workers"` passes.
5. spd.9 auto threshold unchanged. test-checkpoint.R diagnostic tests pass.
6. Closure-size verification: assertion `as.numeric(utils::object.size(hat_in_args_post)) < as.numeric(utils::object.size(hat_in_args_pre))` for a fixture with kt=4 expansion. Uses base R (no extra dep). Encoded as test-spd12-closure-size.R, skip-on-CRAN.
7. mfh features: zero behavior change (mat2hmat path untouched). Existing tests cover invariant.
8. Single ticket FoRecoML-zdc (epic: defer input2rtw_partial expansion). Single commit, conventional commits, no AI attribution, no co-author trailer.
9. Comment update: reco_ml.R:114 `13-item` → `14-item`.

## Iteration history

- v1: F=PASS, C=FAIL (B1 Xtest orphan, B2 mfh tests gap), S=PASS
- v2: F=PASS, C=FAIL (B1 NA semantics underspecified, B2 NA fixture missing), S=PASS (adv: T5 clarity, stale comment)
- v3 changes:
  - **B1 (NA semantics)**: Option A explicit — per-series local NA drop via na_col_mask(X) on kset path; `global_id_post_na` propagates to Xtest. Equivalence proof in Guards section: NA-column membership is property of expanded_hat, not sel_mat → per-series and global-zero yield identical effective per-series feature sets.
  - **B2 (NA fixture)**: DoD step 2.c adds NA-injected fixture (ctrml × compact × lightgbm with synthetic all-NA column post-expansion). Validates NA path post-spd.12.
  - **Scope advisory A**: T5 clarity — Guards explicitly state row-expansion (input2rtw_partial) defers, column-slicing (sel_mat_keep_cols) stays. Schema entries reflect this.
  - **Scope advisory B**: Comment update at line 114 added to Schema + DoD step 9.
