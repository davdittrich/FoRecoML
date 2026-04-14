# Temporal Reconciliation with Machine Learning

This function performs machine-learning–based temporal forecast
reconciliation for linearly constrained multiple time series based on
the cross-temporal approach proposed by Rombouts et al. (2024).
Reconciled forecasts are obtained by fitting non-linear models that map
base forecasts across both temporal dimensions to high-frequency series.
Fully coherent forecasts are then derived by temporal bottom-up.

## Usage

``` r
# Reconciled forecasts
terml(base, hat, obs, agg_order, tew = "sum", features = "all",
      approach = "randomForest", params = NULL, tuning = NULL,
      sntz = FALSE, round = FALSE, fit = NULL)

# Pre-trained reconciled ML models
terml_fit(hat, obs, agg_order, tew = "sum", features = "all",
          approach = "randomForest", params = NULL, tuning = NULL)
```

## Arguments

- base:

  A (\\h(k^\ast + m) \times 1\\) numeric vector containing the base
  forecasts to be reconciled, ordered from lowest to highest frequency;
  \\m\\ is the maximum aggregation order, \\k^\ast\\ is the sum of a
  chosen subset of the \\p - 1\\ factors of \\m\\ (excluding \\m\\
  itself) and \\h\\ is the forecast horizon for the lowest frequency
  time series.

- hat:

  A (\\N(k^\ast + m) \times 1\\) numeric vector containing the base
  forecasts ordered from lowest to highest frequency; \\N\\ is the
  training length for the lowest frequency time series. These forecasts
  are used to train the ML approach.

- obs:

  A (\\Nm \times 1\\) numeric vector containing (observed) values for
  the highest frequency series (\\k = 1\\). These values are used to
  train the ML approach.

- agg_order:

  Highest available sampling frequency per seasonal cycle (max. order of
  temporal aggregation, \\m\\), or a vector representing a subset of
  \\p\\ factors of \\m\\.

- tew:

  A string specifying the type of temporal aggregation. Options include:
  "`sum`" (simple summation, *default*), "`avg`" (average), "`first`"
  (first value of the period), and "`last`" (last value of the period).

- features:

  Character string specifying which features are used for model
  training. Options include "`all`" (see Rombouts et al. 2025,
  *default*) and "`low-high`" (only the lowest- and highest-frequency
  base forecasts as features).

- approach:

  Character string specifying the machine learning method used for
  reconciliation. Options are:

  - "`randomForest`" (*default*): Random Forest algorithm (see the
    randomForest package).

  - "`xgboost`": Extreme Gradient Boosting (see the xgboost package).

  - "`lightgbm`": Light Gradient Boosting Machine (see the lightgbm
    package).

  - "`mlr3`": Any regression learner available in the mlr3 package. The
    learner must be specified via `params`, e.g.
    `params = list(.key = "regr.ranger")`.

- params:

  Optional list of additional parameters passed to the chosen ML
  approach These may include algorithm-specific hyperparameters for
  randomForest, xgboost, lightgbm, or learner options for mlr3. When
  `approach = "mlr3"`, the list must include `.key` to select the
  learner (e.g. `.key = "regr.ranger"`, *default*).

- tuning:

  Optional list specifying tuning options when using the
  [mlr3tuning::mlr3tuning](https://mlr3tuning.mlr-org.com/reference/mlr3tuning-package.html)
  framework (e.g., terminators, search spaces). The argument format
  follows
  [mlr3tuning::auto_tuner](https://mlr3tuning.mlr-org.com/reference/auto_tuner.html),
  except that the learner is set through `params`.

- sntz:

  Logical. If `TRUE`, enforces non-negativity on reconciled forecasts
  using the heuristic "set-negative-to-zero" (Di Fonzo and Girolimetto,
  2023). *Default* is `FALSE`.

- round:

  Logical. If `TRUE`, reconciled forecasts are rounded to integer values
  and coherence is ensured via a bottom-up adjustment. *Default* is
  `FALSE`.

- fit:

  A pre-trained ML reconciliation model (see,
  [extract_reconciled_ml](https://danigiro.github.io/FoRecoML/reference/extract_reconciled_ml.md)).
  If supplied, training data (`hat`, `obs`) are not required.

## Value

- terml returns a temporal reconciled forecast vector with the same
  dimensions, along with attributes containing the fitted model and
  reconciliation settings (see,
  [FoReco::recoinfo](https://danigiro.github.io/FoReco/reference/recoinfo.html)
  and
  [extract_reconciled_ml](https://danigiro.github.io/FoRecoML/reference/extract_reconciled_ml.md)).

&nbsp;

- terml_fit returns a fitted object that can be reused for
  reconciliation on new base forecasts.

## References

Di Fonzo, T. and Girolimetto, D. (2023), Spatio-temporal reconciliation
of solar forecasts, *Solar Energy*, 251, 13–29.
[doi:10.1016/j.solener.2023.01.003](https://doi.org/10.1016/j.solener.2023.01.003)

Girolimetto, D. and Di Fonzo, T. (2023), Point and probabilistic
forecast reconciliation for general linearly constrained multiple time
series, *Statistical Methods & Applications*, 33, 581-607.
[doi:10.1007/s10260-023-00738-6](https://doi.org/10.1007/s10260-023-00738-6)
.

Rombouts, J., Ternes, M., and Wilms, I. (2025). Cross-temporal forecast
reconciliation at digital platforms with machine learning.
*International Journal of Forecasting*, 41(1), 321-344.
[doi:10.1016/j.ijforecast.2024.05.008](https://doi.org/10.1016/j.ijforecast.2024.05.008)

## Examples

``` r
# m: quarterly temporal aggregation order
m <- 4
te_set <- tetools(m)$set

# te_fh: minimum forecast horizon per temporal aggregate
te_fh <- m/te_set

# N_hat: dimension for the lowest frequency (k = m) training set
N_hat <- 16

# bts_mean: mean for the Normal draws used to simulate data
bts_mean <- 5

# hat: a training (base forecasts) feautures vector
hat <- rnorm(sum(te_fh)*N_hat, rep(te_set*bts_mean,  N_hat*te_fh))

# obs: (observed) values for the highest frequency series (k = 1)
obs <- rnorm(m*N_hat, bts_mean)

# h: base forecast horizon at the lowest-frequency series (k = m)
h <- 2

# base: base forecasts matrix
base <- rnorm(sum(te_fh)*h, rep(te_set*bts_mean,  h*te_fh))

##########################################################################
# Different ML approaches
##########################################################################
# XGBoost Reconciliation (xgboost pkg)
reco <- terml(base = base, hat = hat, obs = obs, agg_order = m,
              approach = "xgboost")

# XGBoost Reconciliation with Tweedie loss function (xgboost pkg)
reco <- terml(base = base, hat = hat, obs = obs, agg_order = m,
              approach = "xgboost",
              params =  list(
                eta = 0.3, colsample_bytree = 1, min_child_weight = 1,
                max_depth = 6, gamma = 0, subsample = 1,
                objective = "reg:tweedie", # Tweedie regression objective
                tweedie_variance_power = 1.5 # Tweedie power parameter
              ))

# LightGBM Reconciliation (lightgbm pkg)
reco <- terml(base = base, hat = hat, obs = obs, agg_order = m,
              approach = "lightgbm")

# Random Forest Reconciliation (randomForest pkg)
reco <- terml(base = base, hat = hat, obs = obs, agg_order = m,
              approach = "randomForest")

# Using the mlr3 pkg:
# With 'params = list(.key = mlr_learners)' we can specify different
# mlr_learners implemented in mlr3 such as "regr.ranger" for Random Forest,
# "regr.xgboost" for XGBoost, and others.
reco <- terml(base = base, hat = hat, obs = obs, agg_order = m,
              approach = "mlr3",
              # choose mlr3 learner (here Random Forest via ranger)
              params = list(.key = "regr.ranger"))

# \donttest{
# With mlr3 we can also tune our parameters: e.g. explore mtry in [1,4].
# We can reduce excessive logging by calling:
# if(requireNamespace("lgr", quietly = TRUE)){
#   lgr::get_logger("mlr3")$set_threshold("warn")
#   lgr::get_logger("bbotk")$set_threshold("warn")
# }
reco <- terml(base = base, hat = hat, obs = obs, agg_order = m,
              approach = "mlr3",
              params = list(
                .key = "regr.ranger",
                # number of features tried at each split
                mtry = paradox::to_tune(paradox::p_int(1, 2))
              ),
              tuning = list(
                # stop after 10 evaluations
                terminator = mlr3tuning::trm("evals", n_evals = 10)
              ))
# }
##########################################################################
# Usage with pre-trained models
##########################################################################
# Pre-trained machine learning models (e.g., omit the base param)
mdl <- terml_fit(hat = hat, obs = obs, agg_order = m,
                 approach = "lightgbm")

# Pre-trained machine learning models with base param
reco <- terml(base = base, hat = hat, obs = obs, agg_order = m,
              approach = "lightgbm")
mdl2 <- extract_reconciled_ml(reco)

# New base forecasts matrix
base_new <- rnorm(sum(te_fh)*h, rep(te_set*bts_mean,  h*te_fh))
reco_new <- terml(base = base_new, fit = mdl2, agg_order = m)
```
