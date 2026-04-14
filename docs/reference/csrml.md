# Cross-sectional Reconciliation with Machine Learning

This function performs machine-learning–based cross-sectional forecast
reconciliation for linearly constrained (e.g., hierarchical/grouped)
multiple time series (Spiliotis et al., 2021). Reconciled forecasts are
obtained by training non-linear predictive models (e.g., random forests,
gradient boosting) that learn mappings from base forecasts across all
series to bottom-level series values. Coherent forecasts for the entire
hierarchy are then derived by aggregating the reconciled bottom-level
forecasts through the summing constraints. While the approach is
designed for hierarchical and grouped structures, in the case of general
linearly constrained time series it can be applied within the broader
reconciliation framework described by Girolimetto and Di Fonzo (2024).

## Usage

``` r
# Reconciled forecasts
csrml(base, hat, obs, agg_mat, features = "all", approach = "randomForest",
      params = NULL, tuning = NULL, sntz = FALSE, round = FALSE, fit = NULL)

# Pre-trained reconciled ML models
csrml_fit(hat, obs, agg_mat, features = "all", approach = "randomForest",
          params = NULL, tuning = NULL)
```

## Arguments

- base:

  A (\\h \times n\\) numeric matrix or multivariate time series (`mts`
  class) containing the base forecasts to be reconciled; \\h\\ is the
  forecast horizon, and \\n\\ is the total number of time series (\\n =
  n_a + n_b\\).

- hat:

  A (\\N \times n\\) numeric matrix containing the base forecasts to
  train the ML approach; \\N\\ is the training length.

- obs:

  A (\\N \times n_b\\) numeric matrix containing (observed) values to
  train the ML approach; \\n_b\\ is the total number of bottom
  variables.

- agg_mat:

  A (\\n_a \times n_b\\) numeric matrix representing the cross-sectional
  aggregation matrix. It maps the \\n_b\\ bottom-level (free) variables
  into the \\n_a\\ upper (constrained) variables.

- features:

  Character string specifying which features are used for model
  training. Options include "`bts`" (only bottom-level series as
  features), `str` (features based on the structural matrix),
  "`str-bts`" (`bts` + `str` features), and "`all`" (all series as
  features, *default*).

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

- csrml returns a cross-sectional reconciled forecast matrix with the
  same dimensions, along with attributes containing the fitted model and
  reconciliation settings (see,
  [FoReco::recoinfo](https://danigiro.github.io/FoReco/reference/recoinfo.html)
  and
  [extract_reconciled_ml](https://danigiro.github.io/FoRecoML/reference/extract_reconciled_ml.md)).

&nbsp;

- csrml_fit returns a fitted object that can be reused for
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

Spiliotis, E., Abolghasemi, M., Hyndman, R. J., Petropoulos, F., and
Assimakopoulos, V. (2021). Hierarchical forecast reconciliation with
machine learning. *Applied Soft Computing*, 112, 107756.
[doi:10.1016/j.asoc.2021.107756](https://doi.org/10.1016/j.asoc.2021.107756)

## Examples

``` r
# agg_mat: simple aggregation matrix, A = B + C
agg_mat <- t(c(1,1))
dimnames(agg_mat) <- list("A", c("B", "C"))

# N_hat: dimension for the most aggregated training set
N_hat <- 100

# ts_mean: mean for the Normal draws used to simulate data
ts_mean <- c(20, 10, 10)

# hat: a training (base forecasts) feautures matrix
hat <- matrix(rnorm(length(ts_mean)*N_hat, mean = ts_mean),
              N_hat, byrow = TRUE)
colnames(hat) <- unlist(dimnames(agg_mat))

# obs: (observed) values for bottom-level series (B, C)
obs <- matrix(rnorm(length(ts_mean[-1])*N_hat, mean = ts_mean[-1]),
              N_hat, byrow = TRUE)
colnames(obs) <- colnames(agg_mat)

# h: base forecast horizon
h <- 2

# base: base forecasts matrix
base <- matrix(rnorm(length(ts_mean)*h, mean = ts_mean),
               h, byrow = TRUE)
colnames(base) <- unlist(dimnames(agg_mat))

##########################################################################
# Different ML approaches
##########################################################################
# XGBoost Reconciliation (xgboost pkg)
reco <- csrml(base = base, hat = hat, obs = obs, agg_mat = agg_mat,
              approach = "xgboost", features = "all")

# XGBoost Reconciliation with Tweedie loss function (xgboost pkg)
reco <- csrml(base = base, hat = hat, obs = obs, agg_mat = agg_mat,
              approach = "xgboost", features = "all",
              params =  list(
                eta = 0.3, colsample_bytree = 1, min_child_weight = 1,
                max_depth = 6, gamma = 0, subsample = 1,
                objective = "reg:tweedie", # Tweedie regression objective
                tweedie_variance_power = 1.5 # Tweedie power parameter
              ))

# LightGBM Reconciliation (lightgbm pkg)
reco <- csrml(base = base, hat = hat, obs = obs, agg_mat = agg_mat,
              approach = "lightgbm", features = "all")

# Random Forest Reconciliation (randomForest pkg)
reco <- csrml(base = base, hat = hat, obs = obs, agg_mat = agg_mat,
              approach = "randomForest", features = "all")

# Using the mlr3 pkg:
# With 'params = list(.key = mlr_learners)' we can specify different
# mlr_learners implemented in mlr3 such as "regr.ranger" for Random Forest,
# "regr.xgboost" for XGBoost, and others.
reco <- csrml(base = base, hat = hat, obs = obs, agg_mat = agg_mat,
              approach = "mlr3", features = "all",
              # choose mlr3 learner (here Random Forest via ranger)
              params = list(.key = "regr.ranger"))

# \donttest{
# With mlr3 we can also tune our parameters: e.g. explore mtry in [1,2].
# We can reduce excessive logging by calling:
# if(requireNamespace("lgr", quietly = TRUE)){
#   lgr::get_logger("mlr3")$set_threshold("warn")
#   lgr::get_logger("bbotk")$set_threshold("warn")
# }
reco <- csrml(base = base, hat = hat, obs = obs, agg_mat = agg_mat,
              approach = "mlr3", features = "all",
              params = list(
                .key = "regr.ranger",
                # number of features tried at each split
                mtry = paradox::to_tune(paradox::p_int(1, 2))
              ),
              tuning = list(
                # stop after 10 evaluations
                terminator = mlr3tuning::trm("evals", n_evals = 20)
              ))
#> Warning: package ‘future’ was built under R version 4.5.2
#> Warning: package ‘mlr3’ was built under R version 4.5.2
# }
##########################################################################
# Usage with pre-trained models
##########################################################################
# Pre-trained machine learning models (e.g., omit the base param)
mdl <- csrml_fit(hat = hat, obs = obs, agg_mat = agg_mat,
                 approach = "xgboost", features = "all")

# Pre-trained machine learning models with base param
reco <- csrml(base = base, hat = hat, obs = obs, agg_mat = agg_mat,
              approach = "xgboost", features = "all")
mdl2 <- extract_reconciled_ml(reco)

# New base forecasts matrix
base_new <- matrix(rnorm(length(ts_mean)*h, mean = ts_mean), h, byrow = TRUE)
reco_new <- csrml(base = base_new, fit = mdl, agg_mat = agg_mat)
```
