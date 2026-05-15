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
#' # Reconciled forecasts
#' ctrml(base, hat, obs, agg_mat, agg_order, tew = "sum", features = "all",
#'       approach = "ranger", params = NULL, tuning = NULL,
#'       sntz = FALSE, round = FALSE, fit = NULL, checkpoint = "auto")
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
#'   training. Options include "\code{all}" (see Rombouts et al. 2025), and
#'   "\code{compact}" (see Rombouts et al. 2025, \emph{default}).
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
#'   \code{.key} to select the learner (e.g. \code{.key = "regr.ranger"},
#'   \emph{default}).
#' @param sntz Logical. If \code{TRUE}, enforces non-negativity on reconciled
#'   forecasts using the heuristic "set-negative-to-zero" (Di Fonzo and
#'   Girolimetto, 2023). \emph{Default} is \code{FALSE}.
#' @param round Logical. If \code{TRUE}, reconciled forecasts are rounded to
#'   integer values and coherence is ensured via a bottom-up adjustment.
#'   \emph{Default} is \code{FALSE}.
#' @param tuning Optional list specifying tuning options when using the
#'   [mlr3tuning] framework (e.g., terminators, search spaces). The argument
#'   format follows [mlr3tuning::auto_tuner], except that the learner is set
#'   through `params`.
#' @param checkpoint Controls disk-backed checkpointing of fitted per-series
#'   models. With \eqn{p} bottom-level models trained simultaneously, the
#'   default behaviour can exceed available RAM for large hierarchies. Tri-mode
#'   semantics:
#'   \itemize{
#'   \item \code{"auto"} (\emph{default}): enable checkpointing when the
#'   estimated peak memory exceeds 80\% of available physical RAM, using a
#'   session-scoped sub-directory of \code{tempdir()}; otherwise keep all fits
#'   in memory. On platforms where available memory cannot be detected, this
#'   falls back to OFF.
#'   \item \code{TRUE} or \code{"true"}: always enable, using a session-scoped
#'   sub-directory of \code{tempdir()} (removed at end of the R session).
#'   \item \code{FALSE} or \code{"false"}: never enable; keep all fits in
#'   memory (legacy behaviour, byte-identical to pre-checkpoint).
#'   \item character path: always enable, storing fits in the given directory
#'   (created if missing). Use this for persistent storage suitable for
#'   reusing a fit across R sessions via the \code{fit} argument.
#'   }
#'   Serialization uses \pkg{qs2} for in-memory R objects (\code{randomForest},
#'   \code{mlr3}) and \pkg{xgboost} raw bytes, and the native \code{lgb.save}
#'   for \pkg{lightgbm} (its C++ external pointer cannot be serialized by
#'   \pkg{qs2}). Predictions from a checkpointed fit are identical
#'   (\eqn{\le 10^{-12}}) to the in-memory fit.
#' @inheritParams FoReco::ctrec
#'
#' @returns
#'   - [ctrml] returns a cross-temporal reconciled forecast matrix with the
#'   same dimensions, along with attributes containing the fitted model and
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
#'               agg_mat = agg_mat, approach = "xgboost")
#'
#' # XGBoost Reconciliation with Tweedie loss function (xgboost pkg)
#' reco <- ctrml(base = base, hat = hat, obs = obs, agg_order = m,
#'               agg_mat = agg_mat, approach = "xgboost",
#'               params =  list(
#'                 eta = 0.3, colsample_bytree = 1, min_child_weight = 1,
#'                 max_depth = 6, gamma = 0, subsample = 1,
#'                 objective = "reg:tweedie", # Tweedie regression objective
#'                 tweedie_variance_power = 1.5 # Tweedie power parameter
#'               ))
#'
#' # LightGBM Reconciliation (lightgbm pkg)
#' reco <- ctrml(base = base, hat = hat, obs = obs, agg_order = m,
#'               agg_mat = agg_mat, approach = "lightgbm")
#'
#' # Random Forest Reconciliation (randomForest pkg)
#' reco <- ctrml(base = base, hat = hat, obs = obs, agg_order = m,
#'               agg_mat = agg_mat, approach = "randomForest")
#'
#' # Using the mlr3 pkg:
#' # With 'params = list(.key = mlr_learners)' we can specify different
#' # mlr_learners implemented in mlr3 such as "regr.ranger" for Random Forest,
#' # "regr.xgboost" for XGBoost, and others.
#' reco <- ctrml(base = base, hat = hat, obs = obs, agg_order = m,
#'               agg_mat = agg_mat, approach = "mlr3",
#'               # choose mlr3 learner (here Random Forest via ranger)
#'               params = list(.key = "regr.ranger"))
#' \donttest{
#' # With mlr3 we can also tune our parameters: e.g. explore mtry in [1,4].
#' # We can reduce excessive logging by calling:
#' # if(requireNamespace("lgr", quietly = TRUE)){
#' #   lgr::get_logger("mlr3")$set_threshold("warn")
#' #   lgr::get_logger("bbotk")$set_threshold("warn")
#' # }
#' reco <- ctrml(base = base, hat = hat, obs = obs, agg_order = m,
#'               agg_mat = agg_mat, approach = "mlr3",
#'               params = list(
#'                 .key = "regr.ranger",
#'                 # number of features tried at each split
#'                 mtry = paradox::to_tune(paradox::p_int(1, 4))
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
#' mdl <- ctrml_fit(hat = hat, obs = obs, agg_order = m, agg_mat = agg_mat,
#'                  approach = "xgboost")
#'
#' # Pre-trained machine learning models with base param
#' reco <- ctrml(base = base, hat = hat, obs = obs, agg_order = m,
#'               agg_mat = agg_mat, approach = "xgboost")
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
  tew = "sum",
  features = "all",
  approach = "ranger",
  params = NULL,
  tuning = NULL,
  sntz = FALSE,
  round = FALSE,
  fit = NULL,
  checkpoint = "auto"
) {
  if (identical(approach, "randomForest")) {
    lifecycle::deprecate_soft(
      when    = "2.0.0",
      what    = "ctrml(approach = 'randomForest')",
      details = "Use approach = 'ranger' instead; ranger is faster and statistically equivalent."
    )
  }
  if (is.null(fit)) {
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

    if (missing(obs)) {
      cli_abort("Argument {.arg obs} is missing, with no default.", call = NULL)
    } else if (NCOL(obs) %% tmp$dim[["m"]] != 0) {
      cli_abort("Incorrect {.arg obs} columns dimension.", call = NULL)
    } else if (NROW(obs) != tmp$dim[["nb"]]) {
      cli_abort("Incorrect {.arg obs} rows dimension.", call = NULL)
    } else {
      if (!grepl("mfh", features)) {
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
    }

    # T5: build sel_mat BEFORE materializing hat so we can slice-first.
    # For the non-mfh path, the full row-replicated training matrix has
    # `tmp$dim[["n"]] * tmp$dim[["p"]]` columns. For mfh, the full mat2hmat
    # output has `n * kt` columns; T3 defers materialization to per-iter in
    # rml() so we compute total_cols from dims without expanding hat.
    if (!grepl("mfh", features)) {
      total_cols <- tmp$dim[["n"]] * tmp$dim[["p"]]
      h_train <- NULL
    } else {
      h_train <- NCOL(hat) / tmp$dim[["kt"]]
      total_cols <- tmp$dim[["n"]] * tmp$dim[["kt"]]
    }
    features_size <- total_cols

    switch(
      features,
      "mfh-hfbts" = {
        sel_mat <- as(id_hfbts, "sparseVector")
      },
      "mfh-hfts" = {
        sel_mat <- as(rep(id_hfts, tmp$dim[["n"]]), "sparseVector")
      },
      "mfh-bts" = {
        sel_mat <- as(rep(id_bts, each = tmp$dim[["kt"]]), "sparseVector")
      },
      "mfh-str" = {
        sel_mat <- 1 * (sel_mat != 0)
      },
      "mfh-str-hfbts" = {
        sel_mat <- 1 * (sel_mat != 0)
        sel_mat <- sel_mat +
          sparse_col_replicate(id_hfbts, tmp$dim[["nb"]] * tmp$dim[["m"]])
        sel_mat[sel_mat != 0] <- 1
      },
      "mfh-str-bts" = {
        sel_mat <- 1 * (sel_mat != 0)
        idx_local <- rep(id_bts, each = tmp$dim[["kt"]])
        sel_mat <- sel_mat +
          sparse_col_replicate(idx_local, tmp$dim[["nb"]] * tmp$dim[["m"]])
        sel_mat[sel_mat != 0] <- 1
      },
      "mfh-all" = {
        sel_mat <- 1
      },
      "all" = {
        sel_mat <- 1
        block_sampling <- tmp$dim[["m"]]
      },
      "compact" = {
        n_  <- tmp$dim[["n"]]
        nb_ <- tmp$dim[["nb"]]
        na_ <- tmp$dim[["na"]]
        p_  <- tmp$dim[["p"]]
        i_top <- rep(seq_len(n_), times = nb_)
        j_top <- rep(seq_len(nb_), each = n_)
        if (p_ > 1) {
          row_offsets <- seq(from = na_ + n_, by = n_, length.out = p_ - 1)
          i_band <- as.vector(outer(seq_len(nb_), row_offsets, `+`))
          j_band <- rep(seq_len(nb_), times = p_ - 1)
        } else {
          i_band <- integer(0)
          j_band <- integer(0)
        }
        sel_mat <- Matrix::sparseMatrix(
          i = c(i_top, i_band),
          j = c(j_top, j_band),
          x = 1,
          dims = c(n_ * p_, nb_)
        )
        block_sampling <- tmp$dim[["m"]]
      },
      {
        cli_abort("Unknown {.arg features} option.", call = NULL)
      }
    )
    attr(sel_mat, "sel_method") <- features

    # T5: slice-first materialization for non-mfh path.
    if (!grepl("mfh", features)) {
      keep_cols <- sel_mat_keep_cols(sel_mat, total_cols)
      hat <- input2rtw_partial(hat, tmp$set, cols = keep_cols)
    } else {
      keep_cols <- NULL
    }

    # Remove NA variables from sel_mat. For the slice-first (non-mfh) path,
    # NA detection runs on the already-sliced hat and we map local NA columns
    # back to global sel_mat row indices via keep_cols. T3: for mfh defer,
    # raw hat is held; we only materialize once for na_col_mask when NAs are
    # actually present (rare in the OOM case). When no raw NAs, skip entirely.
    if (grepl("mfh", features)) {
      na_local <- if (anyNA(hat)) {
        na_col_mask(
          mat2hmat(hat, h = h_train, kset = tmp$set, n = tmp$dim[["n"]])
        )
      } else {
        logical(total_cols)
      }
    } else {
      na_local <- na_col_mask(hat)
    }
    if (any(na_local)) {
      if (is.null(keep_cols)) {
        # mfh path: sel_mat sized to total_cols (= n*kt for ctrml mfh).
        if (NCOL(sel_mat) == 1) {
          if (length(sel_mat) == 1) {
            sel_mat <- rep(sel_mat, total_cols)
          }
          sel_mat[na_local] <- 0
          sel_mat <- as(sel_mat, "sparseVector")
        } else {
          sel_mat[na_local, ] <- 0
        }
      } else {
        # T5: map local NA -> global indices via keep_cols.
        na_global <- keep_cols[na_local]
        if (NCOL(sel_mat) == 1) {
          if (length(sel_mat) == 1) {
            sel_mat <- rep(sel_mat, total_cols)
          }
          sel_mat[na_global] <- 0
          sel_mat <- as(sel_mat, "sparseVector")
        } else {
          sel_mat[na_global, ] <- 0
        }
        # Drop NA columns from pre-sliced hat and refresh keep_cols.
        hat <- hat[, !na_local, drop = FALSE]
        keep_cols <- keep_cols[!na_local]
      }
    }
  } else {
    if (!inherits(fit, "rml_fit")) {
      cli_abort("Incorrect {.arg fit} object.", call = NULL)
    }

    if (fit$framework != "ct") {
      cli_abort("Incompatible {.arg fit} framework.", call = NULL)
    }

    agg_order <- fit$agg_order
    agg_mat <- fit$agg_mat
    tew <- fit$tew
    tmp <- cttools(agg_order = agg_order, agg_mat = agg_mat, tew = tew)
    kset <- tmp$set
    kt <- tmp$dim[["kt"]]

    hat <- NULL
    obs <- NULL
    sel_mat <- fit$sel_mat
    approach <- fit$approach
    features <- attr(fit$sel_mat, "sel_method")
    features_size <- fit$features_size
    block_sampling <- fit$block_sampling
    # T5: in fit-reuse, derive keep_cols from stored (full) sel_mat for the
    # non-mfh path so we can slice-materialize base. mfh path stays full.
    if (!grepl("mfh", features)) {
      keep_cols <- sel_mat_keep_cols(sel_mat, features_size)
    } else {
      keep_cols <- NULL
    }
    # T3: predict-only path -> no training hat, h_train is unused.
    h_train <- NULL
  }

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
    if (!grepl("mfh", features)) {
      base <- input2rtw_partial(base, tmp$set, cols = keep_cols)
    }
    # T3: mfh path keeps `base` raw (n x h*kt); rml() materializes per-iter.
  }

  # T5: post-materialization width = length(keep_cols) for non-mfh slice-first
  # path. mfh defers expansion so base remains raw (NCOL = h * kt); skip the
  # post-expansion ncol assert for that path.
  if (!grepl("mfh", features)) {
    expected_base_ncol <- length(keep_cols)
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
    kset   = if (grepl("mfh", features)) tmp$set else NULL,
    h      = if (grepl("mfh", features)) h_train else NULL,
    h_base = if (grepl("mfh", features)) h       else NULL,
    n      = if (grepl("mfh", features)) tmp$dim[["n"]] else NULL,
    checkpoint = checkpoint
  )

  obj <- attr(reco_mat, "fit")
  obj <- new_rml_fit(
    fit = obj$fit,
    agg_mat = agg_mat,
    agg_order = agg_order,
    tew = tew,
    sel_mat = obj$sel_mat,
    approach = approach,
    framework = "ct",
    features = features,
    features_size = features_size,
    block_sampling = block_sampling,
    checkpoint_dir = obj$checkpoint_dir
  )
  attr(reco_mat, "fit") <- NULL
  if (!grepl("mfh", features)) {
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

  attr(reco_mat, "FoReco") <- new_foreco_info(list(
    fit = obj,
    framework = "Cross-temporal",
    forecast_horizon = h,
    te_set = tmp$set,
    cs_n = tmp$dim[["n"]],
    rfun = "ctrml",
    ml = approach
  ))
  return(reco_mat)
}

#' @usage
#' # Pre-trained reconciled ML models
#' ctrml_fit(hat, obs, agg_mat, agg_order, tew = "sum", features = "all",
#'           approach = "ranger", params = NULL, tuning = NULL,
#'           checkpoint = "auto")
#'
#' @return
#'   - [ctrml_fit] returns a fitted object that can be reused for
#'     reconciliation on new base forecasts.
#'
#' @rdname ctrml
#'
#' @export
ctrml_fit <- function(
  hat,
  obs,
  agg_mat,
  agg_order,
  tew = "sum",
  features = "all",
  approach = "ranger",
  params = NULL,
  tuning = NULL,
  checkpoint = "auto"
) {
  if (identical(approach, "randomForest")) {
    lifecycle::deprecate_soft(
      when    = "2.0.0",
      what    = "ctrml_fit(approach = 'randomForest')",
      details = "Use approach = 'ranger' instead; ranger is faster and statistically equivalent."
    )
  }
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

  if (missing(obs)) {
    cli_abort("Argument {.arg obs} is missing, with no default.", call = NULL)
  } else if (NCOL(obs) %% tmp$dim[["m"]] != 0) {
    cli_abort("Incorrect {.arg obs} columns dimension.", call = NULL)
  } else if (NROW(obs) != tmp$dim[["nb"]]) {
    cli_abort("Incorrect {.arg obs} rows dimension.", call = NULL)
  } else {
    if (!grepl("mfh", features)) {
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
  }

  # T5: build sel_mat BEFORE materializing hat (non-mfh slice-first).
  # T3: mfh keeps `hat` raw and defers mat2hmat to per-iter inside rml().
  if (!grepl("mfh", features)) {
    total_cols <- tmp$dim[["n"]] * tmp$dim[["p"]]
  } else {
    h <- NCOL(hat) / tmp$dim[["kt"]]
    total_cols <- tmp$dim[["n"]] * tmp$dim[["kt"]]
  }

  switch(
    features,
    "mfh-hfbts" = {
      sel_mat <- as(id_hfbts, "sparseVector")
    },
    "mfh-hfts" = {
      sel_mat <- as(rep(id_hfts, tmp$dim[["n"]]), "sparseVector")
    },
    "mfh-bts" = {
      sel_mat <- as(rep(id_bts, each = tmp$dim[["kt"]]), "sparseVector")
    },
    "mfh-str" = {
      sel_mat <- 1 * (sel_mat != 0)
    },
    "mfh-str-hfbts" = {
      sel_mat <- 1 * (sel_mat != 0)
      sel_mat <- sel_mat +
        sparse_col_replicate(id_hfbts, tmp$dim[["nb"]] * tmp$dim[["m"]])
      sel_mat[sel_mat != 0] <- 1
    },
    "mfh-str-bts" = {
      sel_mat <- 1 * (sel_mat != 0)
      idx_local <- rep(id_bts, each = tmp$dim[["kt"]])
      sel_mat <- sel_mat +
        sparse_col_replicate(idx_local, tmp$dim[["nb"]] * tmp$dim[["m"]])
      sel_mat[sel_mat != 0] <- 1
    },
    "mfh-all" = {
      sel_mat <- 1
    },
    "all" = {
      sel_mat <- 1
      block_sampling <- tmp$dim[["m"]]
    },
    "compact" = {
      n_  <- tmp$dim[["n"]]
      nb_ <- tmp$dim[["nb"]]
      na_ <- tmp$dim[["na"]]
      p_  <- tmp$dim[["p"]]
      i_top <- rep(seq_len(n_), times = nb_)
      j_top <- rep(seq_len(nb_), each = n_)
      if (p_ > 1) {
        row_offsets <- seq(from = na_ + n_, by = n_, length.out = p_ - 1)
        i_band <- as.vector(outer(seq_len(nb_), row_offsets, `+`))
        j_band <- rep(seq_len(nb_), times = p_ - 1)
      } else {
        i_band <- integer(0)
        j_band <- integer(0)
      }
      sel_mat <- Matrix::sparseMatrix(
        i = c(i_top, i_band),
        j = c(j_top, j_band),
        x = 1,
        dims = c(n_ * p_, nb_)
      )
      block_sampling <- tmp$dim[["m"]]
    },
    {
      cli_abort("Unknown {.arg features} option.", call = NULL)
    }
  )
  attr(sel_mat, "sel_method") <- features

  # T5: slice-first materialization for non-mfh.
  if (!grepl("mfh", features)) {
    keep_cols <- sel_mat_keep_cols(sel_mat, total_cols)
    hat <- input2rtw_partial(hat, tmp$set, cols = keep_cols)
  } else {
    keep_cols <- NULL
  }

  # Remove NA variables from sel_mat (slice-first aware). T3: mfh defers
  # expansion; only materialize for na_col_mask when raw hat actually has
  # NAs (rare in the OOM case targeted by T3).
  if (grepl("mfh", features)) {
    na_local <- if (anyNA(hat)) {
      na_col_mask(
        mat2hmat(hat, h = h, kset = tmp$set, n = tmp$dim[["n"]])
      )
    } else {
      logical(total_cols)
    }
  } else {
    na_local <- na_col_mask(hat)
  }
  if (any(na_local)) {
    if (is.null(keep_cols)) {
      if (NCOL(sel_mat) == 1) {
        if (length(sel_mat) == 1) {
          sel_mat <- rep(sel_mat, total_cols)
        }
        sel_mat[na_local] <- 0
        sel_mat <- as(sel_mat, "sparseVector")
      } else {
        sel_mat[na_local, ] <- 0
      }
    } else {
      na_global <- keep_cols[na_local]
      if (NCOL(sel_mat) == 1) {
        if (length(sel_mat) == 1) {
          sel_mat <- rep(sel_mat, total_cols)
        }
        sel_mat[na_global] <- 0
        sel_mat <- as(sel_mat, "sparseVector")
      } else {
        sel_mat[na_global, ] <- 0
      }
      hat <- hat[, !na_local, drop = FALSE]
      keep_cols <- keep_cols[!na_local]
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
    kset = if (grepl("mfh", features)) tmp$set else NULL,
    h    = if (grepl("mfh", features)) h        else NULL,
    n    = if (grepl("mfh", features)) tmp$dim[["n"]] else NULL,
    checkpoint = checkpoint
  )

  obj <- new_rml_fit(
    fit = obj$fit,
    agg_mat = agg_mat,
    agg_order = agg_order,
    tew = tew,
    sel_mat = obj$sel_mat,
    approach = approach,
    framework = "ct",
    features = features,
    features_size = total_cols,
    block_sampling = block_sampling,
    checkpoint_dir = obj$checkpoint_dir
  )
  return(obj)
}
