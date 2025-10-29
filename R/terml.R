#' Temporal Reconciliation with Machine Learning
#'
#' This function performs machine-learning–based temporal forecast
#' reconciliation for linearly constrained multiple time series based on the
#' cross-temporal approach proposed by Rombouts et al. (2024). Reconciled
#' forecasts are obtained by fitting non-linear models that map base forecasts
#' across both temporal dimensions to high-frequency series. Fully coherent
#' forecasts are then derived by temporal bottom-up.
#'
#' @usage
#' # Reconciled forecasts
#' terml(base, hat, obs, agg_order, tew = "sum", features = "rtw",
#'       approach = "randomForest", params = NULL, tuning = NULL,
#'       fit = NULL, sntz = FALSE, round = FALSE)
#'
#' @param base A (\eqn{N(k^\ast + m) \times 1}) numeric vector containing the
#'   base forecasts to be reconciled, ordered from lowest to highest frequency;
#'   \eqn{N} is the training length for the lowest frequency time series,
#'   \eqn{m} is the maximum aggregation order, and \eqn{k^\ast} is the sum of a
#'   chosen subset of the \eqn{p - 1} factors of \eqn{m} (excluding \eqn{m}
#'   itself).
#' @param hat A (\eqn{N(k^\ast + m) \times 1}) numeric vector containing the
#'   base forecasts ordered from lowest to highest frequency; \eqn{N} is the
#'   training length for the lowest frequency time series. These
#'   forecasts are used to train the ML approach.
#' @param obs A (\eqn{Nm \times 1}) numeric vector containing (observed) values
#'   for the highest frequency series (\eqn{k = 1}). These values are used to
#'   train the ML approach.
#' @param features Character string specifying which features are used for
#'   model training. Options include "\code{rtw}" (see Rombouts et al. 2025,
#'   \emph{default}) and "\code{rtw-top}" (only the lowest- and
#'   highest-frequency base forecasts as features).
#'
#' @inheritParams ctrml
#'
#' @returns
#'   - [terml] returns a temporal reconciled forecast vector with the same
#'   dimensions, along with attributes containing the fitted model and
#'   reconciliation settings (see, [FoReco::recoinfo] and
#'   [extract_reconciled_ml]).
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
#' Rombouts, J., Ternes, M., and Wilms, I. (2025). Cross-temporal forecast
#' reconciliation at digital platforms with machine learning.
#' \emph{International Journal of Forecasting}, 41(1), 321-344.
#' \doi{10.1016/j.ijforecast.2024.05.008}
#'
#' @examples
#' # m: quarterly temporal aggregation order
#' m <- 4
#' te_set <- tetools(m)$set
#'
#' # te_fh: minimum forecast horizon per temporal aggregate
#' te_fh <- m/te_set
#'
#' # N_hat: dimension for the lowest frequency (k = m) training set
#' N_hat <- 16
#'
#' # bts_mean: mean for the Normal draws used to simulate data
#' bts_mean <- 5
#'
#' # hat: a training (base forecasts) feautures vector
#' hat <- rnorm(sum(te_fh)*N_hat, rep(te_set*bts_mean,  N_hat*te_fh))
#'
#' # obs: (observed) values for the highest frequency series (k = 1)
#' obs <- rnorm(m*N_hat, bts_mean)
#'
#' # h: base forecast horizon at the lowest-frequency series (k = m)
#' h <- 2
#'
#' # base: base forecasts matrix
#' base <- rnorm(sum(te_fh)*h, rep(te_set*bts_mean,  h*te_fh))
#'
#' ##########################################################################
#' # Different ML approaches
#' ##########################################################################
#' # XGBoost Reconciliation (xgboost pkg)
#' reco <- terml(base = base, hat = hat, obs = obs, agg_order = m,
#'               approach = "xgboost", features = "rtw")
#'
#' # XGBoost Reconciliation with Tweedie loss function (xgboost pkg)
#' reco <- terml(base = base, hat = hat, obs = obs, agg_order = m,
#'               approach = "xgboost", features = "rtw",
#'               params =  list(
#'                 eta = 0.3, colsample_bytree = 1, min_child_weight = 1,
#'                 max_depth = 6, gamma = 0, subsample = 1,
#'                 objective = "reg:tweedie", # Tweedie regression objective
#'                 tweedie_variance_power = 1.5 # Tweedie power parameter
#'               ))
#'
#' # LightGBM Reconciliation (lightgbm pkg)
#' reco <- terml(base = base, hat = hat, obs = obs, agg_order = m,
#'               approach = "lightgbm", features = "rtw")
#'
#' # Random Forest Reconciliation (randomForest pkg)
#' reco <- terml(base = base, hat = hat, obs = obs, agg_order = m,
#'               approach = "randomForest", features = "rtw")
#'
#' # Using the mlr3 pkg:
#' # With 'params = list(.key = mlr_learners)' we can specify different
#' # mlr_learners implemented in mlr3 such as "regr.ranger" for Random Forest,
#' # "regr.xgboost" for XGBoost, and others.
#' reco <- terml(base = base, hat = hat, obs = obs, agg_order = m,
#'               approach = "mlr3", features = "rtw",
#'               # choose mlr3 learner (here Random Forest via ranger)
#'               params = list(.key = "regr.ranger"))
#'
#' \donttest{
#' # With mlr3 we can also tune our parameters: e.g. explore mtry in [1,4].
#' # We can reduce excessive logging by calling:
#' # if(requireNamespace("lgr", quietly = TRUE)){
#' #   lgr::get_logger("mlr3")$set_threshold("warn")
#' #   lgr::get_logger("bbotk")$set_threshold("warn")
#' # }
#' reco <- terml(base = base, hat = hat, obs = obs, agg_order = m,
#'               approach = "mlr3", features = "rtw",
#'               params = list(
#'                 .key = "regr.ranger",
#'                 # number of features tried at each split
#'                 mtry = paradox::to_tune(paradox::p_int(1, 2))
#'               ),
#'               tuning = list(
#'                 # stop after 10 evaluations
#'                 terminator = mlr3tuning::trm("evals", n_evals = 10)
#'               ))
#' }
#' ##########################################################################
#' # Usage with pre-trained models
#' ##########################################################################
#' # Pre-trained machine learning models (e.g., omit the base param)
#' mdl <- terml_fit(hat = hat, obs = obs, agg_order = m,
#'                  approach = "lightgbm", features = "rtw")
#'
#' # Pre-trained machine learning models with base param
#' reco <- terml(base = base, hat = hat, obs = obs, agg_order = m,
#'               approach = "lightgbm", features = "rtw")
#' mdl2 <- extract_reconciled_ml(reco)
#'
#' # New base forecasts matrix
#' base_new <- rnorm(sum(te_fh)*h, rep(te_set*bts_mean,  h*te_fh))
#' reco_new <- terml(base = base_new, fit = mdl2, agg_order = m)
#'
#' @export
terml <- function(
  base,
  hat,
  obs,
  agg_order,
  tew = "sum",
  features = "rtw",
  approach = "randomForest",
  params = NULL,
  tuning = NULL,
  fit = NULL,
  sntz = FALSE,
  round = FALSE
) {
  if (is.null(fit)) {
    if (missing(agg_order)) {
      cli_abort(
        "Argument {.arg agg_order} is missing, with no default.",
        call = NULL
      )
    }

    tmp <- tetools(agg_order = agg_order, tew = tew)
    kset <- tmp$set
    m <- tmp$dim[["m"]]
    kt <- tmp$dim[["kt"]]
    id_hfts <- c(rep(0, tmp$dim[["ks"]]), rep(1, m))
    strc_mat <- tmp$strc_mat
    agg_mat <- tmp$agg_mat

    block_sampling <- NULL # block_sampling for the block tuning rtw option on mlr3

    if (missing(obs)) {
      cli_abort(
        "Argument {.arg obs} is missing, with no default.",
        call = NULL
      )
    } else if (length(obs) %% m != 0) {
      cli_abort("Incorrect {.arg obs} length.", call = NULL)
    } else {
      if (grepl("rtw", features)) {
        obs <- cbind(obs)
      } else {
        obs <- matrix(obs, ncol = m, byrow = TRUE)
      }
    }

    if (missing(hat)) {
      cli_abort(
        "Argument {.arg hat} is missing, with no default.",
        call = NULL
      )
    } else if (length(hat) %% kt != 0) {
      cli_abort("Incorrect {.arg hat} length.", call = NULL)
    } else {
      if (grepl("rtw", features)) {
        hat <- input2rtw(hat, kset)
      } else {
        h_hat <- length(hat) / kt
        hat <- vec2hmat(vec = hat, h = h_hat, kset = kset)
      }
    }
    features_size <- NCOL(hat)

    switch(
      features,
      "hfts" = {
        sel_mat <- as(id_hfts, "sparseVector")
      },
      "str" = {
        sel_mat <- 1 * (strc_mat != 0)
      },
      "str-hfts" = {
        sel_mat <- 1 * (strc_mat != 0)
        sel_mat <- sel_mat +
          Matrix(rep(id_hfts, m), ncol = m, sparse = TRUE)
        sel_mat[sel_mat != 0] <- 1
      },
      "all" = {
        sel_mat <- 1 #Matrix(1, nrow = kt, ncol = m, sparse = TRUE)
      },
      "rtw" = {
        sel_mat <- 1
        block_sampling <- tmp$dim[["m"]]
      },
      "rtw-top" = {
        sel_mat <- as(id_hfts, "sparseVector")
        sel_mat[1] <- 1
        block_sampling <- tmp$dim[["m"]]
      },
      {
        cli_abort("Unknown {.arg features} option.", call = NULL)
      }
    )
    attr(sel_mat, "sel_method") <- features
  } else {
    if (!inherits(fit, "rml_fit")) {
      cli_abort("Incorrect {.arg fit} object.", call = NULL)
    }

    if (fit$framework != "te") {
      cli_abort("Incompatible {.arg fit} framework.", call = NULL)
    }

    agg_order <- fit$agg_order
    tew <- fit$tew
    tmp <- tetools(agg_order = agg_order, tew = tew)
    kset <- tmp$set
    kt <- tmp$dim[["kt"]]

    hat <- NULL
    obs <- NULL
    sel_mat <- fit$sel_mat
    approach <- fit$approach
    features <- attr(fit$sel_mat, "sel_method")
    features_size <- fit$features_size
    block_sampling <- fit$block_sampling
  }

  if (missing(base)) {
    cli_abort(
      "Argument {.arg base} is missing, with no default.",
      call = NULL
    )
  } else if (length(base) %% kt != 0) {
    cli_abort("Incorrect {.arg base} length.", call = NULL)
  } else {
    h <- length(base) / kt
    if (grepl("rtw", features)) {
      base <- input2rtw(base, kset)
    } else {
      base <- vec2hmat(vec = base, h = h, kset = kset)
    }
  }

  if (NCOL(base) != features_size) {
    cli_abort(
      paste0(
        "The number of columns of {.arg base} ",
        "must be equal to the number of ",
        "features used during fitting."
      ),
      call = NULL
    )
  }

  reco_mat <- rml(
    base = base,
    hat = hat,
    obs = obs,
    sel_mat = sel_mat,
    approach = approach,
    params = params,
    fit = fit,
    tuning = tuning,
    block_sampling = block_sampling
  )

  obj <- attr(reco_mat, "fit")
  obj <- new_rml_fit(
    fit = obj$fit,
    agg_order = agg_order,
    tew = tew,
    sel_mat = obj$sel_mat,
    approach = approach,
    framework = "te",
    features = features,
    features_size = features_size,
    block_sampling = block_sampling
  )

  attr(reco_mat, "fit") <- NULL

  reco_mat <- tebu(
    as.vector(t(reco_mat)),
    agg_order = agg_order,
    sntz = sntz,
    round = round,
    tew = tew
  )

  attr(reco_mat, "FoReco") <- new_foreco_info(list(
    fit = obj,
    framework = "Temporal",
    forecast_horizon = h,
    te_set = tmp$set,
    rfun = "terml",
    ml = approach
  ))
  return(reco_mat)
}

#' @usage
#' # Pre-trained reconciled ML models
#' terml_fit(hat, obs, agg_order, tew = "sum", features = "rtw",
#'           approach = "randomForest", params = NULL, tuning = NULL)
#'
#' @return
#'   - [terml_fit] returns a fitted object that can be reused for
#'     reconciliation on new base forecasts.
#'
#' @rdname terml
#'
#' @export
terml_fit <- function(
  hat,
  obs,
  agg_order,
  tew = "sum",
  features = "rtw",
  approach = "randomForest",
  params = NULL,
  tuning = NULL
) {
  # Check if 'agg_order' is provided
  if (missing(agg_order)) {
    cli_abort(
      "Argument {.arg agg_order} is missing, with no default.",
      call = NULL
    )
  }

  tmp <- tetools(agg_order = agg_order, tew = tew)
  kset <- tmp$set
  m <- tmp$dim[["m"]]
  kt <- tmp$dim[["kt"]]
  id_hfts <- c(rep(0, tmp$dim[["ks"]]), rep(1, m))
  strc_mat <- tmp$strc_mat
  agg_mat <- tmp$agg_mat

  block_sampling <- NULL # block_sampling for the block tuning rtw option on mlr3

  if (missing(obs)) {
    cli_abort("Argument {.arg obs} is missing, with no default.", call = NULL)
  } else if (length(obs) %% m != 0) {
    cli_abort("Incorrect {.arg obs} length.", call = NULL)
  } else {
    if (grepl("rtw", features)) {
      obs <- cbind(obs)
    } else {
      obs <- matrix(obs, ncol = m, byrow = TRUE)
    }
  }

  if (missing(hat)) {
    cli_abort("Argument {.arg hat} is missing, with no default.", call = NULL)
  } else if (length(hat) %% kt != 0) {
    cli_abort("Incorrect {.arg hat} length.", call = NULL)
  } else {
    if (grepl("rtw", features)) {
      hat <- input2rtw(hat, kset)
    } else {
      h <- length(hat) / kt
      hat <- vec2hmat(vec = hat, h = h, kset = kset)
    }
  }

  switch(
    features,
    "hfts" = {
      sel_mat <- as(id_hfts, "sparseVector")
    },
    "str" = {
      sel_mat <- 1 * (strc_mat != 0)
    },
    "str-hfts" = {
      sel_mat <- 1 * (strc_mat != 0)
      sel_mat <- sel_mat + Matrix(rep(id_hfts, m), ncol = m, sparse = TRUE)
      sel_mat[sel_mat != 0] <- 1
    },
    "all" = {
      sel_mat <- 1 #Matrix(1, nrow = kt, ncol = m, sparse = TRUE)
    },
    "rtw" = {
      sel_mat <- 1
      block_sampling <- tmp$dim[["m"]]
    },
    "rtw-top" = {
      sel_mat <- as(id_hfts, "sparseVector")
      sel_mat[1] <- 1
      block_sampling <- tmp$dim[["m"]]
    },
    {
      cli_abort("Unknown {.arg features} option.", call = NULL)
    }
  )
  attr(sel_mat, "sel_method") <- features

  obj <- rml(
    base = NULL,
    hat = hat,
    obs = obs,
    sel_mat = sel_mat,
    approach = approach,
    params = params,
    fit = NULL,
    tuning = tuning,
    block_sampling = block_sampling
  )

  obj <- new_rml_fit(
    fit = obj$fit,
    agg_order = agg_order,
    tew = tew,
    sel_mat = obj$sel_mat,
    approach = approach,
    framework = "te",
    features = features,
    features_size = NCOL(hat),
    block_sampling = block_sampling
  )

  return(obj)
}
