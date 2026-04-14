# Cross-temporal Reconciliation with Machine Learning

This function performs machine-learning–based cross-temporal forecast
reconciliation for linearly constrained multiple time series (Rombouts
et al., 2024). Reconciled forecasts are obtained by fitting non-linear
models that map base forecasts across both temporal and cross-sectional
dimensions to bottom-level high-frequency series. Fully coherent
forecasts across all temporal and cross-sectional linear combinations
are then derived by cross-temporal bottom-up. While the approach is
designed for hierarchical and grouped structures, in the case of general
linearly constrained time series it can be applied within the broader
reconciliation framework described by Girolimetto and Di Fonzo (2024).

## Usage

``` r
# Reconciled forecasts
ctrml(base, hat, obs, agg_mat, agg_order, tew = "sum", features = "all",
      approach = "randomForest", params = NULL, tuning = NULL,
      sntz = FALSE, round = FALSE, fit = NULL)

# Pre-trained reconciled ML models
ctrml_fit(hat, obs, agg_mat, agg_order, tew = "sum", features = "all",
          approach = "randomForest", params = NULL, tuning = NULL)
```

## Arguments

- base:

  A (\\n \times h(k^\ast+m)\\) numeric matrix containing the base
  forecasts to be reconciled; \\n\\ is the total number of variables,
  \\m\\ is the maximum aggregation order, and \\k^\ast\\ is the sum of a
  chosen subset of the \\p - 1\\ factors of \\m\\ (excluding \\m\\
  itself), and \\h\\ is the forecast horizon for the lowest frequency
  time series. The row identifies a time series, and the forecasts in
  each row are ordered from the lowest frequency (most temporally
  aggregated) to the highest frequency.

- hat:

  A (\\n \times N(k^\ast+m)\\) numeric matrix containing the base
  forecasts ordered from lowest to highest frequency; \\N\\ is the
  training length for the lowest frequency time series. The row
  identifies a time series, and the forecasts in each row are ordered
  from the lowest frequency (most temporally aggregated) to the highest
  frequency. These forecasts are used to train the ML approach.

- obs:

  A (\\n_b \times Nm\\) numeric matrix containing (observed) values for
  the highest frequency series (\\k = 1\\); \\n_b\\ is the total number
  of high-frequency bottom variables. These values are used to train the
  ML approach.

- agg_mat:

  A (\\n_a \times n_b\\) numeric matrix representing the cross-sectional
  aggregation matrix. It maps the \\n_b\\ bottom-level (free) variables
  into the \\n_a\\ upper (constrained) variables.

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
  training. Options include "`all`" (see Rombouts et al. 2025), and
  "`compact`" (see Rombouts et al. 2025, *default*).

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

- ctrml returns a cross-temporal reconciled forecast matrix with the
  same dimensions, along with attributes containing the fitted model and
  reconciliation settings (see,
  [FoReco::recoinfo](https://danigiro.github.io/FoReco/reference/recoinfo.html)
  and
  [extract_reconciled_ml](https://danigiro.github.io/FoRecoML/reference/extract_reconciled_ml.md)).

&nbsp;

- ctrml_fit returns a fitted object that can be reused for
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

# agg_mat: simple aggregation matrix, A = B + C
agg_mat <- t(c(1,1))
dimnames(agg_mat) <- list("A", c("B", "C"))

# te_fh: minimum forecast horizon per temporal aggregate
te_fh <- m/te_set

# N_hat: dimension for the lowest-frequency (k = m) training set
N_hat <- 16

# bts_mean: mean for the Normal draws used to simulate data
bts_mean <- 5

# hat: a training (base forecasts) feautures matrix
hat <- rbind(
  rnorm(sum(te_fh)*N_hat, rep(2*te_set*bts_mean, N_hat*te_fh)),  # Series A
  rnorm(sum(te_fh)*N_hat, rep(te_set*bts_mean,  N_hat*te_fh)),   # Series B
  rnorm(sum(te_fh)*N_hat, rep(te_set*bts_mean,  N_hat*te_fh))    # Series C
)
rownames(hat) <- c("A", "B", "C")

# obs: (observed) values for the highest-frequency bottom-level series
# (B and C with k = 1)
obs <- rbind(
  rnorm(m*N_hat, bts_mean),  # Observed for series B
  rnorm(m*N_hat, bts_mean)   # Observed for series C
)
rownames(obs) <- c("B", "C")


# h: base forecast horizon at the lowest-frequency series (k = m)
h <- 2

# base: base forecasts matrix
base <- rbind(
  rnorm(sum(te_fh)*h, rep(2*te_set*bts_mean, h*te_fh)),  # Base for A
  rnorm(sum(te_fh)*h, rep(te_set*bts_mean,  h*te_fh)),   # Base for B
  rnorm(sum(te_fh)*h, rep(te_set*bts_mean,  h*te_fh))    # Base for C
)
rownames(base) <- c("A", "B", "C")

##########################################################################
# Different ML approaches
##########################################################################
# XGBoost Reconciliation (xgboost pkg)
reco <- ctrml(base = base, hat = hat, obs = obs, agg_order = m,
              agg_mat = agg_mat, approach = "xgboost")

# XGBoost Reconciliation with Tweedie loss function (xgboost pkg)
reco <- ctrml(base = base, hat = hat, obs = obs, agg_order = m,
              agg_mat = agg_mat, approach = "xgboost",
              params =  list(
                eta = 0.3, colsample_bytree = 1, min_child_weight = 1,
                max_depth = 6, gamma = 0, subsample = 1,
                objective = "reg:tweedie", # Tweedie regression objective
                tweedie_variance_power = 1.5 # Tweedie power parameter
              ))

# LightGBM Reconciliation (lightgbm pkg)
reco <- ctrml(base = base, hat = hat, obs = obs, agg_order = m,
              agg_mat = agg_mat, approach = "lightgbm")

# Random Forest Reconciliation (randomForest pkg)
reco <- ctrml(base = base, hat = hat, obs = obs, agg_order = m,
              agg_mat = agg_mat, approach = "randomForest")

# Using the mlr3 pkg:
# With 'params = list(.key = mlr_learners)' we can specify different
# mlr_learners implemented in mlr3 such as "regr.ranger" for Random Forest,
# "regr.xgboost" for XGBoost, and others.
reco <- ctrml(base = base, hat = hat, obs = obs, agg_order = m,
              agg_mat = agg_mat, approach = "mlr3",
              # choose mlr3 learner (here Random Forest via ranger)
              params = list(.key = "regr.ranger"))
# \donttest{
# With mlr3 we can also tune our parameters: e.g. explore mtry in [1,4].
# We can reduce excessive logging by calling:
# if(requireNamespace("lgr", quietly = TRUE)){
#   lgr::get_logger("mlr3")$set_threshold("warn")
#   lgr::get_logger("bbotk")$set_threshold("warn")
# }
reco <- ctrml(base = base, hat = hat, obs = obs, agg_order = m,
              agg_mat = agg_mat, approach = "mlr3",
              params = list(
                .key = "regr.ranger",
                # number of features tried at each split
                mtry = paradox::to_tune(paradox::p_int(1, 4))
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
mdl <- ctrml_fit(hat = hat, obs = obs, agg_order = m, agg_mat = agg_mat,
                 approach = "xgboost")

# Pre-trained machine learning models with base param
reco <- ctrml(base = base, hat = hat, obs = obs, agg_order = m,
              agg_mat = agg_mat, approach = "xgboost")
mdl2 <- extract_reconciled_ml(reco)

# New base forecasts matrix
base_new <- rbind(
  rnorm(sum(te_fh)*h, rep(2*te_set*bts_mean, h*te_fh)),  # Base for A
  rnorm(sum(te_fh)*h, rep(te_set*bts_mean,  h*te_fh)),   # Base for B
  rnorm(sum(te_fh)*h, rep(te_set*bts_mean,  h*te_fh))    # Base for C
)
reco_new <- ctrml(base = base_new, fit = mdl, agg_order = m,
                  agg_mat = agg_mat)
```
