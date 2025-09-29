#' Cross-sectional Reconciliation with Machine Learning
#'
#' This function performs machine-learning–based cross-sectional forecast
#' reconciliation for linearly constrained (e.g., hierarchical/grouped)
#' multiple time series (Spiliotis et al., 2021). Reconciled forecasts are
#' obtained by training non-linear predictive models (e.g., random forests,
#' gradient boosting) that learn mappings from base forecasts across all
#' series to bottom-level series values. Coherent forecasts for the entire
#' hierarchy are then derived by aggregating the reconciled bottom-level
#' forecasts through the summing constraints. While the approach is designed
#' for hierarchical and grouped structures, in the case of general linearly
#' constrained time series it can be applied within the broader reconciliation
#' framework described by Girolimetto and Di Fonzo (2024).
#'
#' @usage
#' csrml(base, hat, obs, agg_mat, features = "all", approach = "randomForest",
#'       params = NULL, tuning = NULL, fit = NULL, sntz = FALSE, round = TRUE,
#'       seed = NULL)
#'
#' @param base A (\eqn{h \times n}) numeric matrix or multivariate time series
#'   (\code{mts} class) containing the base forecasts to be reconciled; \eqn{h}
#'   is the forecast horizon, and \eqn{n} is the total number of time series
#'   (\eqn{n = n_a + n_b}).
#' @param hat A (\eqn{N \times n}) numeric matrix containing the base forecasts
#'   to train the ML approach; \eqn{N} is the training length.
#' @param obs A (\eqn{N \times n_b}) numeric matrix containing (observed) values
#'   to train the ML approach; \eqn{n_b} is the total number of bottom
#'   variables.
#' @param features Character string specifying which features are used for model
#'   training. Options include "\code{bts}", "\code{str}", "\code{str-bts}", and
#'   "\code{all}" (\emph{default}).
#' @inheritParams ctrml
#'
#' @returns If \code{base} is provided, returns a cross-sectional reconciled
#'   forecast matrix with the same dimensions, along with attributes containing
#'   the fitted model and reconciliation settings (see, [FoReco::recoinfo] and
#'   [extract_reconciled_ml]). If only models are trained (omitting
#'   \code{base}), returns a fitted object that can be reused for reconciliation
#'   on new base forecasts (see, [extract_reconciled_ml]).
#'
#' @references
#' Di Fonzo, T. and Girolimetto, D. (2023), Spatio-temporal reconciliation of
#' solar forecasts, \emph{Solar Energy}, 251, 13–29.
#' \doi{10.1016/j.solener.2023.01.003}
#'
#' Girolimetto, D. and Di Fonzo, T. (2023), Point and probabilistic forecast
#' reconciliation for general linearly constrained multiple time series,
#' \emph{Statistical Methods & Applications}, 33, 581-607.
#' \doi{10.1007/s10260-023-00738-6}.
#'
#' Spiliotis, E., Abolghasemi, M., Hyndman, R. J., Petropoulos, F., and
#' Assimakopoulos, V. (2021). Hierarchical forecast reconciliation with machine
#' learning. \emph{Applied Soft Computing}, 112, 107756.
#' \doi{10.1016/j.asoc.2021.107756}
#'
#' @examples
#' # agg_mat: simple aggregation matrix, A = B + C
#' agg_mat <- t(c(1,1))
#' dimnames(agg_mat) <- list("A", c("B", "C"))
#'
#' # N_hat: dimension for the most aggregated training set
#' N_hat <- 100
#'
#' # ts_mean: mean for the Normal draws used to simulate data
#' ts_mean <- c(20, 10, 10)
#'
#' # hat: a training (base forecasts) feautures matrix
#' hat <- matrix(
#'   rnorm(length(ts_mean)*N_hat, mean = ts_mean),
#'   N_hat, byrow = TRUE)
#' colnames(hat) <- unlist(dimnames(agg_mat))
#'
#' # obs: (observed) values for bottom-level series (B, C)
#' obs <- matrix(
#'   rnorm(length(ts_mean[-1])*N_hat, mean = ts_mean[-1]),
#'   N_hat, byrow = TRUE)
#' colnames(obs) <- colnames(agg_mat)
#'
#' # h: base forecast horizon
#' h <- 2
#'
#' # base: base forecasts matrix
#' base <- matrix(
#'   rnorm(length(ts_mean)*h, mean = ts_mean),
#'   h, byrow = TRUE)
#' colnames(base) <- unlist(dimnames(agg_mat))
#'
#' ##########################################################################
#' # Different ML approaches
#' ##########################################################################
#' # XGBoost Reconciliation (xgboost pkg)
#' reco <- csrml(base = base, hat = hat, obs = obs, agg_mat = agg_mat,
#'               approach = "xgboost", seed = 123, features = "all")
#'
#' # XGBoost Reconciliation with Tweedie loss function (xgboost pkg)
#' reco <- csrml(base = base, hat = hat, obs = obs, agg_mat = agg_mat,
#'               approach = "xgboost", seed = 123, features = "all",
#'               params =  list(
#'                 eta = 0.3, colsample_bytree = 1, min_child_weight = 1,
#'                 max_depth = 6, gamma = 0, subsample = 1,
#'                 objective = "reg:tweedie", # Tweedie regression objective
#'                 tweedie_variance_power = 1.5 # Tweedie power parameter
#'               ))
#'
#' # LightGBM Reconciliation (lightgbm pkg)
#' reco <- csrml(base = base, hat = hat, obs = obs, agg_mat = agg_mat,
#'               approach = "lightgbm", seed = 123, features = "all")
#'
#' # Random Forest Reconciliation (randomForest pkg)
#' reco <- csrml(base = base, hat = hat, obs = obs, agg_mat = agg_mat,
#'               approach = "randomForest", seed = 123, features = "all")
#'
#' # Using the mlr3 pkg:
#' # With 'params = list(.key = mlr_learners)' we can specify different
#' # mlr_learners implemented in mlr3 such as "regr.ranger" for Random Forest,
#' # "regr.xgboost" for XGBoost, and others.
#' reco <- csrml(base = base, hat = hat, obs = obs, agg_mat = agg_mat,
#'               approach = "mlr3", seed = 123, features = "all",
#'               # choose mlr3 learner (here Random Forest via ranger)
#'               params = list(.key = "regr.ranger"))
#'
#' \dontrun{
#' # With mlr3 we can also tune our parameters: e.g. explore mtry in [1,2].
#' # We can reduce excessive logging by calling:
#' if (requireNamespace("lgr", quietly = TRUE)) {
#'   lgr::get_logger("mlr3")$set_threshold("warn")
#'   lgr::get_logger("bbotk")$set_threshold("warn")
#' }
#' reco <- csrml(base = base, hat = hat, obs = obs, agg_mat = agg_mat,
#'               approach = "mlr3", seed = 123, features = "all",
#'               params = list(
#'                 .key = "regr.ranger",
#'                 # number of features tried at each split
#'                 mtry = paradox::to_tune(paradox::p_int(1, 2))
#'               ),
#'               tuning = list(
#'                 # stop after 10 evaluations
#'                 terminator = mlr3tuning::trm("evals", n_evals = 20)
#'               ))
#' }
#' ##########################################################################
#' # Usage with pre-trained models
#' ##########################################################################
#' # Pre-trained machine learning models (e.g., omit the base param)
#' mdl <- csrml(hat = hat, obs = obs, agg_mat = agg_mat,
#'              approach = "xgboost", seed = 123, features = "all")
#'
#' # Pre-trained machine learning models with base param
#' reco <- csrml(base = base, hat = hat, obs = obs, agg_mat = agg_mat,
#'               approach = "xgboost", seed = 123, features = "all")
#' mdl2 <- extract_reconciled_ml(reco)
#'
#' # New base forecasts matrix
#' base_new <- matrix(
#'   rnorm(length(ts_mean)*h, mean = ts_mean),
#'   h, byrow = TRUE)
#' reco_new <- csrml(base = base_new, fit = mdl, agg_mat = agg_mat)
#'
#' @export
csrml <- function(
  base,
  hat,
  obs,
  agg_mat,
  features = "all",
  approach = "randomForest",
  params = NULL,
  tuning = NULL,
  fit = NULL,
  sntz = FALSE,
  round = FALSE,
  seed = NULL
) {
  features <- match.arg(features, c("all", "bts", "str", "str-bts"))

  if (missing(agg_mat)) {
    cli_abort(
      "Argument {.arg agg_mat} is missing, with no default.",
      call = NULL
    )
  }

  tmp <- cstools(agg_mat = agg_mat)
  n <- tmp$dim[["n"]]
  nb <- tmp$dim[["nb"]]
  strc_mat <- tmp$strc_mat
  agg_mat <- tmp$agg_mat
  id_bts <- c(rep(0, n - nb), rep(1, nb))

  if (is.null(fit)) {
    if (missing(obs)) {
      cli_abort("Argument {.arg obs} is missing, with no default.", call = NULL)
    } else if (NCOL(obs) != nb) {
      cli_abort("Incorrect {.arg obs} columns dimension.", call = NULL)
    }

    if (missing(hat)) {
      cli_abort("Argument {.arg hat} is missing, with no default.", call = NULL)
    }

    if (missing(base)) {
      base <- NULL
    }

    switch(
      features,
      "bts" = {
        sel_mat <- Matrix(rep(id_bts, nb), ncol = nb, sparse = TRUE)
      },
      "str" = {
        sel_mat <- strc_mat
      },
      "str-bts" = {
        sel_mat <- strc_mat + Matrix(rep(id_bts, nb), ncol = nb, sparse = TRUE)
        sel_mat[sel_mat != 0] <- 1
      },
      "all" = {
        sel_mat <- Matrix(1, nrow = n, ncol = nb, sparse = TRUE)
      }
    )
    attr(sel_mat, "sel_method") <- features
  } else {
    hat <- NULL
    obs <- NULL
    sel_mat <- NULL
    approach <- fit$approach
    features <- attr(fit$sel_mat, "sel_method")

    if (missing(base)) {
      cli_abort(
        "Argument {.arg base} is missing, with no default.",
        call = NULL
      )
    } else if (NCOL(base) != n) {
      cli_abort("Incorrect {.arg base} columns dimension.", call = NULL)
    }
  }

  # if(is.null(fit)){
  #   if(missing(obs)){
  #     cli_abort("Argument {.arg obs} is missing, with no default.", call = NULL)
  #   }else if(NCOL(obs) != nb){
  #     cli_abort("Incorrect {.arg obs} columns dimension.", call = NULL)
  #   }
  #
  #   if(missing(hat)){
  #     cli_abort("Argument {.arg hat} is missing, with no default.", call = NULL)
  #   }
  #
  #   if(is.list(hat) && is.list(base)){
  #     p <- length(base)
  #
  #     ina <- sapply(base, function(bmat){
  #       is.na(colSums(bmat))
  #     })
  #
  #     hat <- lapply(hat, rbind)
  #     hat <- do.call(cbind, hat)
  #
  #     base <- lapply(base, rbind)
  #     base <- do.call(cbind, base)
  #
  #     if(NCOL(hat) != n*p){
  #       cli_abort("Incorrect {.arg hat} elements' columns dimension.", call = NULL)
  #     }
  #
  #     if(NCOL(base) != n*p){
  #       cli_abort("Incorrect {.arg base} elements' columns dimension.", call = NULL)
  #     }
  #
  #     hat <- hat[, !as.vector(ina), drop = FALSE]
  #     base <- base[, !as.vector(ina), drop = FALSE]
  #
  #     strc_mat <- do.call(rbind, rep(list(strc_mat), p))[!as.vector(ina), , drop = FALSE]
  #     id_bts <- rep(id_bts, p)[!as.vector(ina)]
  #
  #   }else if(!is.list(hat) && !is.list(base)){
  #     if(NCOL(hat) != n){
  #       cli_abort("Incorrect {.arg hat} columns dimension.", call = NULL)
  #     }
  #
  #     if(NCOL(base) != n){
  #       cli_abort("Incorrect {.arg base} columns dimension.", call = NULL)
  #     }
  #   }else{
  #     cli_abort("Incorrect {.arg base} or {.arg hat} arguments.", call = NULL)
  #   }
  #
  #   switch(features,
  #          "bts" = {
  #            sel_mat <- Matrix(rep(id_bts, nb), ncol = nb, sparse = TRUE)
  #          },
  #          "hier" = {
  #            sel_mat <- strc_mat
  #          },
  #          "hier-bts" = {
  #            sel_mat <- strc_mat + Matrix(rep(id_bts, nb), ncol = nb, sparse = TRUE)
  #            sel_mat[sel_mat != 0] <- 1
  #          },
  #          "all" = {
  #            sel_mat <- Matrix(1, nrow = NCOL(base), ncol = nb, sparse = TRUE)
  #          }
  #   )
  #   attr(sel_mat, "sel_method") <- features
  # }else{
  #   hat <- NULL
  #   obs <- NULL
  #   sel_mat <- NULL
  #
  #   if(is.list(base)){
  #     p <- length(base)
  #
  #     ina <- sapply(base, function(bmat){
  #       is.na(colSums(bmat))
  #     })
  #
  #     base <- lapply(base, rbind)
  #     base <- do.call(cbind, base)
  #
  #     if(NCOL(base) != n*p){
  #       cli_abort("Incorrect {.arg base} elements' columns dimension.", call = NULL)
  #     }
  #
  #     base <- base[, !as.vector(ina), drop = FALSE]
  #
  #     strc_mat <- do.call(rbind, rep(list(strc_mat), p))[!as.vector(ina), , drop = FALSE]
  #     id_bts <- rep(id_bts, p)[!as.vector(ina)]
  #
  #   }else if(NCOL(base) != n){
  #     cli_abort("Incorrect {.arg base} columns dimension.", call = NULL)
  #   }
  # }

  reco_mat <- rml(
    base = base,
    hat = hat,
    obs = obs,
    sel_mat = sel_mat,
    approach = approach,
    params = params,
    seed = seed,
    fit = fit,
    tuning = tuning
  )

  if (!is.null(base)) {
    fit <- attr(reco_mat, "fit")
    fit$approach <- approach
    attr(reco_mat, "fit") <- NULL
    reco_mat <- csbu(reco_mat, agg_mat = agg_mat, round = round, sntz = sntz)

    attr(reco_mat, "FoReco") <- list2env(list(
      fit = fit,
      framework = "Cross-sectional",
      forecast_horizon = NROW(reco_mat),
      cs_n = n,
      rfun = "csrml",
      ml = approach
    ))
    return(reco_mat)
  } else {
    reco_mat$approach <- approach
    return(reco_mat)
  }
}
