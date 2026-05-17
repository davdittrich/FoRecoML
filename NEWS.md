# FoRecoML 2.0.0

## New features

* `csrml_g()`, `terml_g()`, `ctrml_g()`, and `rml_g()` accept `obs_mask` to exclude
  structurally missing observations from training. Series with all-zero (or `NA`)
  observations across training time can cause catboost to error and LightGBM/XGBoost
  to learn degenerate zero predictors. `obs_mask = NULL` (default) preserves current
  behavior. `obs_mask = "auto"` auto-detects rows where `obs == 0`; supply an explicit
  logical matrix for precise control. Masked series still receive predictions at test
  time via ML extrapolation from unmasked neighbors.

* `csrml_g()`, `terml_g()`, and `ctrml_g()` accept `nonneg_method` to select the
  non-negative reconciliation approach. `"sntz"` (default, near-optimal per
  Girolimetto 2025) clips negatives after bottom-up. `"bpv"`, `"nfca"`, `"nnic"`,
  `"osqp"` use FoReco's projection-based non-negative methods (`nn=` argument in
  `csrec`/`terec`/`ctrec`). On sparse hierarchies with >30% structural-zero bottom
  series, degenerate face risk may arise — a warning fires automatically.

* New `csrml_g()`, `terml_g()`, `ctrml_g()` wrappers for global ML
  reconciliation. `csrml_g()` and `ctrml_g()` return reconciled forecast
  matrices with `attr(., 'FoReco')`. `terml_g()` returns a named numeric vector
  matching `FoReco::tebu()` output. Access the underlying fit via
  `extract_reconciled_ml(result)`. Supports `normalize = c("none","zscore","robust")`
  pre-normalization of the shared feature matrix before fitting.

* New `normalize_stack()` function for pre-normalizing stacked feature matrices
  before global ML training. Supports zscore and robust normalization with 6
  scale estimators: gmd, mad_scaled, qn (robscale), sn (robscale), iqr_scaled, sd_c4.

* `terml_g()` and `ctrml_g()` accept `level_id = TRUE` (default `FALSE`) to add an ordered-integer temporal-aggregation-level feature to the stacked training matrix (1 = finest granularity, max = coarsest). Improves global ML correction accuracy on hierarchies with strong level-specific variance. `csrml_g()` rejects this argument with an informative error. See the "Feature engineering for global ML" article.

* `ctrml_g()` and `terml_g()` accept `input_format = "wide_ct"` for FoReco's
  canonical CT wide matrix layout. When `"wide_ct"`: `hat` is
  `n_series × (n_folds × kt)` (series as rows), `obs` is
  `n_bottom × T_monthly`, and `base` is `n_series × kt`. Internally FoRecoML
  builds one training observation per (series, fold) pair with the series's own
  CT feature vector — enabling per-series base forecasts as ML features.
  `hat_wide` should have column names for LightGBM compatibility.
  Default `"tall"` preserves current behavior.

* `ctrml_g()` gains `cs_level = FALSE`. When `cs_level = TRUE` and
  `input_format = "wide_ct"`, a `cs_level` column is appended to the stacked
  training matrix: `0` for upper (aggregate) series, `1` for bottom (leaf)
  series. This gives tree-based learners an explicit cross-sectional depth
  signal. Requires `input_format = "wide_ct"`; passing `cs_level = TRUE` with
  `input_format = "tall"` raises an informative error.

* `csrml_g()`, `terml_g()`, and `ctrml_g()` accept `method = "rec"` to use
  FoReco's optimal combination reconciliation (`csrec`, `terec`, `ctrec`)
  instead of bottom-up. The `comb = "ols"` (default) works without residuals;
  `comb = "shr"` / `"sam"` for `csrml_g()` use internally-computed validation
  residuals (requires `validation_split > 0`). The `res = NULL` argument allows
  an optional pre-computed residual matrix override.

## Bug fixes

* `csrml_g()`, `terml_g()`, `ctrml_g()` no longer silently ignore the `base`
  argument. Previously they returned bare `rml_g_fit` objects; now they perform
  the full fit + predict + reconcile pipeline.

* `sntz`, `round` (all three wrappers) and `tew` (terml_g/ctrml_g) parameters
  restored for parity with the per-series csrml/terml/ctrml functions.

## Breaking changes

* The default ML backend for `csrml()`, `terml()`, `ctrml()` and their
  `*_fit()` variants changed from `"randomForest"` to `"ranger"`. Update
  existing scripts that relied on the old default by passing
  `approach = "randomForest"` explicitly. The randomForest backend now
  emits a soft deprecation warning (visible via
  `options(lifecycle_verbosity = "warning")`) and will be removed in a
  future major version. ranger is faster on the per-series workload and
  statistically equivalent.

## New features

* New `approach = "catboost"` per-series backend. Install via the catboost R
  package (not on CRAN; see catboost documentation). Supports checkpoint
  serialization to `.cbm` files via `catboost::catboost.save_model()`.
* New `approach = "ranger"` backend (now the default). Uses the
  `ranger` package with `num.threads = 1L` (per-series models are too
  small to benefit from intra-tree threading; the outer rml() loop is
  the natural parallel boundary).
* New `predict.rml_fit()` S3 method providing a uniform predict
  interface that dispatches back to the corresponding framework
  function (`csrml`, `terml`, or `ctrml`) using the stored fit.
* `rml()` now respects the `seed` argument: calling `set.seed(seed)`
  before the per-series training loop. Prior versions accepted `seed`
  but the actual call to `set.seed()` was commented out, making the
  argument a silent no-op.

## Performance

* Checkpoint serialization uses `qs2` with adaptive `nthreads` (capped
  at 4 cores) for faster disk I/O during streaming fit checkpointing.
* `estimate_peak_bytes()` audited for integer overflow: all
  multiplicands now cast to double before multiplication, preventing
  overflow on hierarchies where `NROW(hat) * NCOL(hat) * p` exceeds
  `.Machine$integer.max`.

## Deprecated

* `approach = "randomForest"` is soft-deprecated; use
  `approach = "ranger"` instead.

# FoRecoML (development version)

## New features

* `csrml()`, `terml()`, `ctrml()` and their `*_fit()` variants gain a
  `checkpoint` argument (default `"auto"`) for streaming per-series fits
  to disk. Set `checkpoint = TRUE` to force checkpointing to a
  session-scoped temporary directory, `checkpoint = FALSE` to keep all
  fits in memory (legacy behaviour), or pass a directory path for
  persistent, reusable storage. The default `"auto"` enables
  checkpointing when the estimated peak memory exceeds 80% of
  available RAM. Approach-specific serializers are used:
  `qs2::qs_save()` for randomForest and mlr3, `xgboost::xgb.save.raw()`
  wrapped in `qs2` for xgboost, and `lightgbm::lgb.save()` for lightgbm.

## User-visible changes

* The default value of `tuning$store_benchmark_result` for
  `approach = "mlr3"` is now `FALSE` (was `TRUE`). This is a
  memory-frugal default; users who relied on benchmark archives should
  pass `tuning = list(store_benchmark_result = TRUE)` to restore the
  previous behaviour.

* Internal NA-column detection now treats an empty input
  (`NROW(hat) == 0`) as "all columns NA". Previously this case produced
  `NaN` and propagated downstream; the new behaviour matches the
  intent of the threshold check at the zero-row edge.

# FoRecoML 1.0.0

* Cross-sectional, temporal and cross-temporal forecast reconciliation with machine learning
* Initial CRAN submission.
