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
#' terml(base, hat, obs, agg_order, tew = "sum", features = "all",
#'       approach = "randomForest", params = NULL, tuning = NULL,
#'       sntz = FALSE, round = FALSE, fit = NULL, checkpoint = "auto")
#'
#' @param base A (\eqn{h(k^\ast + m) \times 1}) numeric vector containing the
#'   base forecasts to be reconciled, ordered from lowest to highest frequency;
#'   \eqn{m} is the maximum aggregation order, \eqn{k^\ast} is the sum of a
#'   chosen subset of the \eqn{p - 1} factors of \eqn{m} (excluding \eqn{m}
#'   itself) and \eqn{h} is the forecast horizon for the lowest frequency
#'   time series.
#' @param hat A (\eqn{N(k^\ast + m) \times 1}) numeric vector containing the
#'   base forecasts ordered from lowest to highest frequency; \eqn{N} is the
#'   training length for the lowest frequency time series. These
#'   forecasts are used to train the ML approach.
#' @param obs A (\eqn{Nm \times 1}) numeric vector containing (observed) values
#'   for the highest frequency series (\eqn{k = 1}). These values are used to
#'   train the ML approach.
#' @param features Character string specifying which features are used for
#'   model training. Options include "\code{all}" (see Rombouts et al. 2025,
#'   \emph{default}) and "\code{low-high}" (only the lowest- and
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
#'               approach = "xgboost")
#'
#' # XGBoost Reconciliation with Tweedie loss function (xgboost pkg)
#' reco <- terml(base = base, hat = hat, obs = obs, agg_order = m,
#'               approach = "xgboost",
#'               params =  list(
#'                 eta = 0.3, colsample_bytree = 1, min_child_weight = 1,
#'                 max_depth = 6, gamma = 0, subsample = 1,
#'                 objective = "reg:tweedie", # Tweedie regression objective
#'                 tweedie_variance_power = 1.5 # Tweedie power parameter
#'               ))
#'
#' # LightGBM Reconciliation (lightgbm pkg)
#' reco <- terml(base = base, hat = hat, obs = obs, agg_order = m,
#'               approach = "lightgbm")
#'
#' # Random Forest Reconciliation (randomForest pkg)
#' reco <- terml(base = base, hat = hat, obs = obs, agg_order = m,
#'               approach = "randomForest")
#'
#' # Using the mlr3 pkg:
#' # With 'params = list(.key = mlr_learners)' we can specify different
#' # mlr_learners implemented in mlr3 such as "regr.ranger" for Random Forest,
#' # "regr.xgboost" for XGBoost, and others.
#' reco <- terml(base = base, hat = hat, obs = obs, agg_order = m,
#'               approach = "mlr3",
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
#'               approach = "mlr3",
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
#'                  approach = "lightgbm")
#'
#' # Pre-trained machine learning models with base param
#' reco <- terml(base = base, hat = hat, obs = obs, agg_order = m,
#'               approach = "lightgbm")
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
  features = "all",
  approach = "randomForest",
  params = NULL,
  tuning = NULL,
  sntz = FALSE,
  round = FALSE,
  fit = NULL,
  checkpoint = "auto",
  n_workers = "auto"
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
      if (!grepl("mfh", features)) {
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
    }

    # T5: build sel_mat BEFORE materializing hat for the non-mfh slice-first path.
    # Non-mfh: input2rtw produces ncol = length(kset). mfh: vec2hmat path stays full.
    if (!grepl("mfh", features)) {
      total_cols <- length(kset)
    } else {
      h_hat <- length(hat) / kt
      hat <- vec2hmat(vec = hat, h = h_hat, kset = kset)
      total_cols <- NCOL(hat)
    }
    features_size <- total_cols

    switch(
      features,
      "mfh-hfts" = {
        sel_mat <- as(id_hfts, "sparseVector")
      },
      "mfh-str" = {
        sel_mat <- 1 * (strc_mat != 0)
      },
      "mfh-str-hfts" = {
        sel_mat <- 1 * (strc_mat != 0)
        sel_mat <- sel_mat +
          sparse_col_replicate(id_hfts, m)
        sel_mat[sel_mat != 0] <- 1
      },
      "mfh-all" = {
        sel_mat <- 1 #Matrix(1, nrow = kt, ncol = m, sparse = TRUE)
      },
      "all" = {
        sel_mat <- 1
        block_sampling <- tmp$dim[["m"]]
      },
      "low-high" = {
        sel_mat <- rep(0, length(kset))
        sel_mat[c(1, length(kset))] <- 1
        sel_mat <- as(sel_mat, "sparseVector")
        block_sampling <- tmp$dim[["m"]]
      },
      {
        cli_abort("Unknown {.arg features} option.", call = NULL)
      }
    )
    attr(sel_mat, "sel_method") <- features

    # T5: compute keep_cols for features_size; hat row-expansion deferred to loop_body.
    if (!grepl("mfh", features)) {
      keep_cols <- sel_mat_keep_cols(sel_mat, total_cols)
    } else {
      keep_cols <- NULL
    }

    # Remove NA variables from sel_mat. For non-mfh (spd.12), NA detection is
    # deferred to loop_body per-series via kset. For mfh, NA detection runs on
    # the fully-materialized hat.
    if (is.null(keep_cols)) {
      # mfh path: hat is the full materialization; sel_mat is sized to it.
      na_local <- na_col_mask(hat)
      if (any(na_local)) {
        if (NCOL(sel_mat) == 1) {
          if (length(sel_mat) == 1) {
            sel_mat <- rep(sel_mat, NCOL(hat))
          }
          sel_mat[na_local] <- 0
          sel_mat <- as(sel_mat, "sparseVector")
        } else {
          sel_mat[na_local, ] <- 0
        }
      }
    }
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
    if (!grepl("mfh", features)) {
      keep_cols <- sel_mat_keep_cols(sel_mat, features_size)
    } else {
      keep_cols <- NULL
    }
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
    if (grepl("mfh", features)) {
      base <- vec2hmat(vec = base, h = h, kset = kset)
    }
    # non-mfh: base row-expansion deferred to loop_body via kset
  }

  # Validate base column count for mfh path (post-materialization check).
  # Non-mfh: raw base dimensions already validated by length(base) %% kt check above.
  if (is.null(keep_cols)) {
    expected_base_ncol <- features_size
    if (NCOL(base) != expected_base_ncol) {
      cli_abort(
        paste0(
          "The number of columns of {.arg base} ",
          "must be equal to the number of ",
          "features used during fitting."
        ),
        call = NULL
      )
    }
  } else if (!is.null(fit) && !grepl("mfh", features)) {
    # Horizon mismatch guard: compare predict-time h against training h_train.
    # h_train is stored at fit time; NULL guard preserves back-compat for old fits.
    if (!is.null(fit$h_train) && h != fit$h_train) {
      cli::cli_abort(c(
        "`base` horizon mismatch with training fit.",
        "i" = "Training fit was built with h = {fit$h_train}.",
        "x" = "Got base with implied h = {h} (= length(base) / kt)."
      ), call = NULL)
    }
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
    block_sampling = block_sampling,
    keep_cols = keep_cols,
    checkpoint = checkpoint,
    n_workers = n_workers,
    kset = kset
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
    block_sampling = block_sampling,
    checkpoint_dir = obj$checkpoint_dir,
    na_cols_list = obj$na_cols_list,
    h_train = h
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
#' terml_fit(hat, obs, agg_order, tew = "sum", features = "all",
#'           approach = "randomForest", params = NULL, tuning = NULL,
#'           checkpoint = "auto")
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
  features = "all",
  approach = "randomForest",
  params = NULL,
  tuning = NULL,
  checkpoint = "auto",
  n_workers = "auto"
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
    if (!grepl("mfh", features)) {
      obs <- cbind(obs)
    } else {
      obs <- matrix(obs, ncol = m, byrow = TRUE)
    }
  }

  if (missing(hat)) {
    cli_abort("Argument {.arg hat} is missing, with no default.", call = NULL)
  } else if (length(hat) %% kt != 0) {
    cli_abort("Incorrect {.arg hat} length.", call = NULL)
  }

  # T5: build sel_mat BEFORE materializing hat (non-mfh slice-first).
  if (!grepl("mfh", features)) {
    total_cols <- length(kset)
  } else {
    h <- length(hat) / kt
    hat <- vec2hmat(vec = hat, h = h, kset = kset)
    total_cols <- NCOL(hat)
  }

  switch(
    features,
    "mfh-hfts" = {
      sel_mat <- as(id_hfts, "sparseVector")
    },
    "mfh-str" = {
      sel_mat <- 1 * (strc_mat != 0)
    },
    "mfh-str-hfts" = {
      sel_mat <- 1 * (strc_mat != 0)
      sel_mat <- sel_mat + sparse_col_replicate(id_hfts, m)
      sel_mat[sel_mat != 0] <- 1
    },
    "mfh-all" = {
      sel_mat <- 1 #Matrix(1, nrow = kt, ncol = m, sparse = TRUE)
    },
    "all" = {
      sel_mat <- 1
      block_sampling <- tmp$dim[["m"]]
    },
    "low-high" = {
      sel_mat <- as(id_hfts, "sparseVector")
      sel_mat[1] <- 1
      block_sampling <- tmp$dim[["m"]]
    },
    {
      cli_abort("Unknown {.arg features} option.", call = NULL)
    }
  )
  attr(sel_mat, "sel_method") <- features

  # T5: compute keep_cols for features_size; hat row-expansion deferred to loop_body.
  if (!grepl("mfh", features)) {
    keep_cols <- sel_mat_keep_cols(sel_mat, total_cols)
  } else {
    keep_cols <- NULL
  }

  # Remove NA variables from sel_mat. For non-mfh (spd.12), NA detection is
  # deferred to loop_body per-series via kset. For mfh, NA detection runs on
  # the fully-materialized hat.
  if (is.null(keep_cols)) {
    # mfh path: hat is the full materialization; sel_mat is sized to it.
    na_local <- na_col_mask(hat)
    if (any(na_local)) {
      if (NCOL(sel_mat) == 1) {
        if (length(sel_mat) == 1) {
          sel_mat <- rep(sel_mat, NCOL(hat))
        }
        sel_mat[na_local] <- 0
        sel_mat <- as(sel_mat, "sparseVector")
      } else {
        sel_mat[na_local, ] <- 0
      }
    }
  }

  obj <- rml(
    base = NULL,
    hat = hat,
    obs = obs,
    sel_mat = sel_mat,
    approach = approach,
    params = params,
    fit = NULL,
    tuning = tuning,
    block_sampling = block_sampling,
    keep_cols = keep_cols,
    checkpoint = checkpoint,
    n_workers = n_workers,
    kset = kset
  )

  # h_train is the forecast horizon used at fit time.
  # For mfh path h is already computed from hat.
  # For non-mfh path, terml_fit has no base → no forecast horizon → leave NULL.
  h_train <- if (grepl("mfh", features)) h else NULL

  obj <- new_rml_fit(
    fit = obj$fit,
    agg_order = agg_order,
    tew = tew,
    sel_mat = obj$sel_mat,
    approach = approach,
    framework = "te",
    features = features,
    features_size = total_cols,
    block_sampling = block_sampling,
    checkpoint_dir = obj$checkpoint_dir,
    na_cols_list = obj$na_cols_list,
    h_train = h_train
  )

  return(obj)
}
