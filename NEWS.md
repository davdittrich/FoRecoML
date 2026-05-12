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
