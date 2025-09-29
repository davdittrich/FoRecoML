#' Cross-temporal Reconciliation with Machine Learning
#'
#' This function performs machine-learning–based cross-temporal forecast
#' reconciliation for linearly constrained multiple time series (Rombouts et
#' al., 2024). Reconciled forecasts are obtained by fitting non-linear models
#' that map base forecasts across both temporal and cross-sectional dimensions
#' to bottom-level high-frequency series. Fully coherent forecasts across all
#' temporal and cross-sectional linear combinations are then derived by
#' cross-temporal bottom-up. While the approach is designed for hierarchical
#' and grouped structures, in the case of general linearly constrained time
#' series it can be applied within the broader reconciliation framework
#' described by Girolimetto and Di Fonzo (2024).
#'
#' @usage
#' ctrml(base, hat, obs, agg_mat, agg_order, features = "all",
#'       approach = "randomForest", params = NULL, tuning = NULL,
#'       fit = NULL, tew = "sum", sntz = FALSE, round = TRUE, seed = NULL)
#'
#' @param base A (\eqn{n \times h(k^\ast+m)}) numeric matrix containing the base
#'   forecasts to be reconciled; \eqn{n} is the total number of variables,
#'   \eqn{m} is the maximum aggregation order, and \eqn{k^\ast} is the sum of a
#'   chosen subset of the \eqn{p - 1} factors of \eqn{m} (excluding \eqn{m}
#'   itself), and \eqn{h} is the forecast horizon for the lowest frequency time
#'   series. The row identifies a time series, and the forecasts in each row are
#'   ordered from the lowest frequency (most temporally aggregated) to the
#'   highest frequency.
#' @param hat A (\eqn{n \times N(k^\ast+m)}) numeric matrix containing the base
#'   forecasts ordered from lowest to highest frequency; \eqn{N} is the training
#'   length for the lowest frequency time series. The row identifies a time
#'   series, and the forecasts in each row are ordered from the lowest frequency
#'   (most temporally aggregated) to the highest frequency. These forecasts are
#'   used to train the ML approach.
#' @param obs A (\eqn{n_b \times Nm}) numeric matrix containing (observed)
#'   values for the highest frequency series (\eqn{k = 1}); \eqn{n_b} is the
#'   total number of high-frequency bottom variables. These values are used to
#'   train the ML approach.
#' @param features Character string specifying which features are used for model
#'   training. Options include "\code{hfbts}", "\code{hfts}", "\code{bts}",
#'   "\code{str}", "\code{str-hfbts}", "\code{str-bts}", "\code{all}"
#'   (\emph{default}), "\code{rtw-full}", and "\code{rtw-comp}".
#' @param fit A pre-trained ML reconciliation model (see,
#'   [extract_reconciled_ml]). If supplied, training data (\code{hat},
#'   \code{obs}) are not required.
#' @param approach Character string specifying the machine learning method used
#'   for reconciliation. Options are:
#'   \itemize{
#'   \item "\code{randomForest}" (\emph{default}): Random Forest algorithm
#'   (see the \pkg{randomForest} package).
#'   \item "\code{xgboost}": Extreme Gradient Boosting (see the \pkg{xgboost}
#'   package).
#'   \item "\code{lightgbm}": Light Gradient Boosting Machine (see the
#'   \pkg{lightgbm} package).
#'   \item "\code{mlr3}": Any regression learner available in the \pkg{mlr3}
#'   package. The learner must be specified via \code{params}, e.g.
#'   \code{params = list(.key = "regr.ranger")}.
#'   }
#' @param params Optional list of additional parameters passed to the chosen
#'   ML approach These may include algorithm-specific hyperparameters for
#'   \pkg{randomForest}, \pkg{xgboost}, \pkg{lightgbm}, or learner options for
#'   \pkg{mlr3}. When \code{approach = "mlr3"}, the list must include
#'   \code{.key} to select the learner (e.g. \code{.key = "regr.xgboost"},
#'   \emph{default}).
#' @param sntz Logical. If \code{TRUE}, enforces non-negativity on reconciled
#'   forecasts using the heuristic "set-negative-to-zero" (Di Fonzo and
#'   Girolimetto, 2023). \emph{Default} is \code{FALSE}.
#' @param round Logical. If \code{TRUE}, reconciled forecasts are rounded to
#'   integer values and coherence is ensured via a bottom-up adjustment.
#'   \emph{Default} is \code{FALSE}.
#' @param seed Optional integer seed for reproducibility.
#' @param tuning Optional list specifying tuning options when using the
#'   [mlr3tuning] framework (e.g., terminators, search spaces). The argument
#'   format follows [mlr3tuning::auto_tuner], except that the learner is set
#'   through `params`.
#' @inheritParams FoReco::ctrec
#'
#' @returns If \code{base} is provided, returns a cross-temporal reconciled
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
#' # agg_mat: simple aggregation matrix, A = B + C
#' agg_mat <- t(c(1,1))
#' dimnames(agg_mat) <- list("A", c("B", "C"))
#'
#' # te_fh: minimum forecast horizon per temporal aggregate
#' te_fh <- m/te_set
#'
#' # N_hat: dimension for the lowest-frequency (k = m) training set
#' N_hat <- 16
#'
#' # bts_mean: mean for the Normal draws used to simulate data
#' bts_mean <- 5
#'
#' # hat: a training (base forecasts) feautures matrix
#' hat <- rbind(
#'   rnorm(sum(te_fh)*N_hat, rep(2*te_set*bts_mean, N_hat*te_fh)),  # Series A
#'   rnorm(sum(te_fh)*N_hat, rep(te_set*bts_mean,  N_hat*te_fh)),   # Series B
#'   rnorm(sum(te_fh)*N_hat, rep(te_set*bts_mean,  N_hat*te_fh))    # Series C
#' )
#' rownames(hat) <- c("A", "B", "C")
#'
#' # obs: (observed) values for the highest-frequency bottom-level series
#' # (B and C with k = 1)
#' obs <- rbind(
#'   rnorm(m*N_hat, bts_mean),  # Observed for series B
#'   rnorm(m*N_hat, bts_mean)   # Observed for series C
#' )
#' rownames(obs) <- c("B", "C")
#'
#'
#' # h: base forecast horizon at the lowest-frequency series (k = m)
#' h <- 2
#'
#' # base: base forecasts matrix
#' base <- rbind(
#'   rnorm(sum(te_fh)*h, rep(2*te_set*bts_mean, h*te_fh)),  # Base for A
#'   rnorm(sum(te_fh)*h, rep(te_set*bts_mean,  h*te_fh)),   # Base for B
#'   rnorm(sum(te_fh)*h, rep(te_set*bts_mean,  h*te_fh))    # Base for C
#' )
#' rownames(base) <- c("A", "B", "C")
#'
#' ##########################################################################
#' # Different ML approaches
#' ##########################################################################
#' # XGBoost Reconciliation (xgboost pkg)
#' reco <- ctrml(base = base, hat = hat, obs = obs, agg_order = m,
#'               agg_mat = agg_mat, approach = "xgboost", seed = 123,
#'               features = "rtw-full")
#'
#' # XGBoost Reconciliation with Tweedie loss function (xgboost pkg)
#' reco <- ctrml(base = base, hat = hat, obs = obs, agg_order = m,
#'               agg_mat = agg_mat, approach = "xgboost", seed = 123,
#'               params =  list(
#'                 eta = 0.3, colsample_bytree = 1, min_child_weight = 1,
#'                 max_depth = 6, gamma = 0, subsample = 1,
#'                 objective = "reg:tweedie", # Tweedie regression objective
#'                 tweedie_variance_power = 1.5 # Tweedie power parameter
#'               ),
#'               features = "rtw-full")
#'
#' # LightGBM Reconciliation (lightgbm pkg)
#' reco <- ctrml(base = base, hat = hat, obs = obs, agg_order = m,
#'               agg_mat = agg_mat, approach = "lightgbm", seed = 123,
#'               features = "rtw-full")
#'
#' # Random Forest Reconciliation (randomForest pkg)
#' reco <- ctrml(base = base, hat = hat, obs = obs, agg_order = m,
#'               agg_mat = agg_mat, approach = "randomForest", seed = 123,
#'               features = "rtw-full")
#'
#' # Using the mlr3 pkg:
#' # With 'params = list(.key = mlr_learners)' we can specify different
#' # mlr_learners implemented in mlr3 such as "regr.ranger" for Random Forest,
#' # "regr.xgboost" for XGBoost, and others.
#' reco <- ctrml(base = base, hat = hat, obs = obs, agg_order = m,
#'               agg_mat = agg_mat, approach = "mlr3", seed = 123,
#'               # choose mlr3 learner (here Random Forest via ranger)
#'               params = list(.key = "regr.ranger"),
#'               features = "rtw-full")
#' \dontrun{
#' # With mlr3 we can also tune our parameters: e.g. explore mtry in [1,4].
#' # We can reduce excessive logging by calling:
#' if(requireNamespace("lgr", quietly = TRUE)){
#'   lgr::get_logger("mlr3")$set_threshold("warn")
#'   lgr::get_logger("bbotk")$set_threshold("warn")
#' }
#' reco <- ctrml(base = base, hat = hat, obs = obs, agg_order = m,
#'               agg_mat = agg_mat, approach = "mlr3", seed = 123,
#'               params = list(
#'                 .key = "regr.ranger",
#'                 # number of features tried at each split
#'                 mtry = paradox::to_tune(paradox::p_int(1, 4))
#'               ),
#'               tuning = list(
#'                 # stop after 10 evaluations
#'                 terminator = mlr3tuning::trm("evals", n_evals = 10)
#'               ),
#'               features = "rtw-full")
#' }
#' ##########################################################################
#' # Usage with pre-trained models
#' ##########################################################################
#' # Pre-trained machine learning models (e.g., omit the base param)
#' mdl <- ctrml(hat = hat, obs = obs,
#'              agg_order = m, agg_mat = agg_mat, approach = "lightgbm",
#'              seed = 123, features = "rtw-full")
#'
#' # Pre-trained machine learning models with base param
#' reco <- ctrml(base = base, hat = hat, obs = obs,
#'               agg_order = m, agg_mat = agg_mat, approach = "lightgbm",
#'               seed = 123, features = "rtw-full")
#' mdl2 <- extract_reconciled_ml(reco)
#'
#' # New base forecasts matrix
#' base_new <- rbind(
#'   rnorm(sum(te_fh)*h, rep(2*te_set*bts_mean, h*te_fh)),  # Base for A
#'   rnorm(sum(te_fh)*h, rep(te_set*bts_mean,  h*te_fh)),   # Base for B
#'   rnorm(sum(te_fh)*h, rep(te_set*bts_mean,  h*te_fh))    # Base for C
#' )
#' reco_new <- ctrml(base = base_new, fit = mdl, agg_order = m,
#'                   agg_mat = agg_mat)
#'
#' @export
ctrml <- function(
  base,
  hat,
  obs,
  agg_mat,
  agg_order,
  features = "all",
  approach = "randomForest",
  params = NULL,
  tuning = NULL,
  fit = NULL,
  tew = "sum",
  sntz = FALSE,
  round = FALSE,
  seed = NULL
) {
  features <- match.arg(
    features,
    c(
      "hfbts",
      "hfts",
      "bts",
      "str",
      "str-hfbts",
      "str-bts",
      "all",
      "rtw-full",
      "rtw-comp"
    )
  )

  # Check if 'agg_order' is provided
  if (missing(agg_order)) {
    cli_abort(
      "Argument {.arg agg_order} is missing, with no default.",
      call = NULL
    )
  }

  tmp <- cttools(agg_mat = agg_mat, agg_order = agg_order, tew = tew)
  strc_mat <- tmp$strc_mat

  id_bts <- c(rep(0, tmp$dim[["na"]]), rep(1, tmp$dim[["nb"]]))
  id_hfts <- c(rep(0, tmp$dim[["ks"]]), rep(1, tmp$dim[["m"]]))
  id_hfbts <- as.numeric(kronecker(id_bts, id_hfts))

  # block_sampling for the block tuning rtw option on mlr3
  block_sampling <- NULL

  if (is.null(fit)) {
    if (missing(obs)) {
      cli_abort("Argument {.arg obs} is missing, with no default.", call = NULL)
    } else if (NCOL(obs) %% tmp$dim[["m"]] != 0) {
      cli_abort("Incorrect {.arg obs} columns dimension.", call = NULL)
    } else if (NROW(obs) != tmp$dim[["nb"]]) {
      cli_abort("Incorrect {.arg obs} rows dimension.", call = NULL)
    } else {
      if (grepl("rtw", features)) {
        obs <- t(obs)
      } else {
        obs <- matrix(
          as.vector(t(obs)),
          ncol = tmp$dim[["m"]] * tmp$dim[["nb"]]
        )
      }
    }

    if (missing(hat)) {
      cli_abort("Argument {.arg hat} is missing, with no default.", call = NULL)
    } else if (NCOL(hat) %% tmp$dim[["kt"]] != 0) {
      cli_abort("Incorrect {.arg hat} columns dimension.", call = NULL)
    } else if (NROW(hat) != tmp$dim[["n"]]) {
      cli_abort("Incorrect {.arg hat} rows dimension.", call = NULL)
    } else {
      if (grepl("rtw", features)) {
        hat <- input2rtw(hat, tmp$set)
      } else {
        h <- NCOL(hat) / tmp$dim[["kt"]]
        hat <- mat2hmat(hat, h = h, kset = tmp$set, n = tmp$dim[["n"]])
      }
    }

    if (missing(base)) {
      base <- NULL
    } else if (NCOL(base) %% tmp$dim[["kt"]] != 0) {
      cli_abort("Incorrect {.arg base} columns dimension.", call = NULL)
    } else if (NROW(base) != tmp$dim[["n"]]) {
      cli_abort("Incorrect {.arg base} rows dimension.", call = NULL)
    } else {
      h <- NCOL(base) / tmp$dim[["kt"]]
      if (grepl("rtw", features)) {
        base <- input2rtw(base, tmp$set)
      } else {
        # Calculate 'h' and 'base_hmat'
        base <- mat2hmat(base, h = h, kset = tmp$set, n = tmp$dim[["n"]])
      }
    }

    switch(
      features,
      "hfbts" = {
        sel_mat <- as(id_hfbts, "sparseVector")
      },
      "hfts" = {
        sel_mat <- as(rep(id_hfts, tmp$dim[["n"]]), "sparseVector")
      },
      "bts" = {
        sel_mat <- as(rep(id_bts, each = tmp$dim[["kt"]]), "sparseVector")
      },
      "str" = {
        sel_mat <- strc_mat
      },
      "str-hfbts" = {
        sel_mat <- strc_mat +
          Matrix(
            rep(id_hfbts, tmp$dim[["nb"]] * tmp$dim[["m"]]),
            ncol = tmp$dim[["nb"]] * tmp$dim[["m"]],
            sparse = TRUE
          )
        sel_mat[sel_mat != 0] <- 1
      },
      "str-bts" = {
        sel_mat <- strc_mat +
          Matrix(
            rep(
              rep(id_bts, each = tmp$dim[["kt"]]),
              tmp$dim[["nb"]] * tmp$dim[["m"]]
            ),
            ncol = tmp$dim[["nb"]] * tmp$dim[["m"]],
            sparse = TRUE
          )
        sel_mat[sel_mat != 0] <- 1
      },
      "all" = {
        sel_mat <- 1
      },
      "rtw-full" = {
        sel_mat <- 1
        block_sampling <- tmp$dim[["m"]]
      },
      "rtw-comp" = {
        pos <- seq(
          tmp$dim[["na"]],
          by = tmp$dim[["n"]],
          length.out = tmp$dim[["p"]]
        )
        sel_mat <- Matrix::bandSparse(
          tmp$dim[["nb"]],
          tmp$dim[["n"]] * tmp$dim[["p"]],
          pos
        )
        sel_mat <- 1 * t(sel_mat))
        sel_mat[1:tmp$dim[["n"]], ] <- 1
        block_sampling <- tmp$dim[["m"]]
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
    } else if (NCOL(base) %% tmp$dim[["kt"]] != 0) {
      cli_abort("Incorrect {.arg base} columns dimension.", call = NULL)
    } else if (NROW(base) != tmp$dim[["n"]]) {
      cli_abort("Incorrect {.arg base} rows dimension.", call = NULL)
    } else {
      h <- NCOL(base) / tmp$dim[["kt"]]
      if (grepl("rtw", features)) {
        base <- input2rtw(base, tmp$set)
      } else {
        base <- mat2hmat(base, h = h, kset = tmp$set, n = tmp$dim[["n"]])
      }
    }
  }

  reco_mat <- rml(
    base = base,
    hat = hat,
    obs = obs,
    sel_mat = sel_mat,
    approach = approach,
    params = params,
    seed = seed,
    fit = fit,
    tuning = tuning,
    block_sampling = block_sampling
  )

  if (!is.null(base)) {
    fit <- attr(reco_mat, "fit")
    fit$approach <- approach
    attr(reco_mat, "fit") <- NULL
    if (!grepl("rtw", features)) {
      reco_mat <- matrix(as.vector(reco_mat), ncol = tmp$dim[["nb"]])
    }

    reco_mat <- ctbu(
      t(reco_mat),
      agg_order = agg_order,
      agg_mat = agg_mat,
      sntz = sntz,
      round = round,
      tew = tew
    )

    attr(reco_mat, "FoReco") <- list2env(list(
      fit = fit,
      framework = "Cross-temporal",
      forecast_horizon = h,
      te_set = tmp$set,
      cs_n = tmp$dim[["n"]],
      rfun = "ctrml",
      ml = approach
    ))
    return(reco_mat)
  } else {
    reco_mat$approach <- approach
    return(reco_mat)
  }
}
