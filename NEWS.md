# FoRecoML 2.0.0

## New features

* New `csrml_g()`, `terml_g()`, `ctrml_g()` wrappers for global ML
  reconciliation. `csrml_g()` and `ctrml_g()` return reconciled forecast
  matrices with `attr(., 'FoReco')`. `terml_g()` returns a named numeric vector
  matching `FoReco::tebu()` output. Access the underlying fit via
  `extract_reconciled_ml(result)`. Supports `normalize = c("none","zscore","robust")`
  pre-normalization of the shared feature matrix before fitting.

* New `normalize_stack()` function for pre-normalizing stacked feature matrices
  before global ML training. Supports zscore and robust normalization with 6
  scale estimators: gmd, mad_scaled, qn (robscale), sn (robscale), iqr_scaled, sd_c4.

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
