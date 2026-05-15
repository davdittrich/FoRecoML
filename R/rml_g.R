# T7.2: Global ML across all bottom-series.
# Stacks `obs` (T_obs x p) into one long training matrix with series_id as a
# categorical feature, fits a single global model, returns an rml_g_fit object.
# The rml_g_fit S3 methods (predict/print/...) land in T7.5.

# Private helper: stack series into long-format training matrix.
#
# Returns:
#   X_stacked        : (T_obs*p) x ncol(hat) numeric matrix (no series_id column)
#   y_stacked        : length T_obs*p numeric vector
#   series_id_factor : factor with FROZEN, alphabetically-sorted levels
#   series_id_int    : 1-based integer (1..p) for backends that need raw ints
#   series_id_levels : character vector of the frozen factor levels
#   norm_params      : NULL (normalization is applied OUTSIDE by wrappers, T7.3)
#   train_idx        : integer rows used for training (global, computed once)
#   valid_idx        : integer rows held out for validation (may be length 0)
#
# Series order in the stack: column-major over `obs`, i.e. series 1 occupies the
# first T_obs rows, series 2 the next T_obs rows, and so on.
.stack_series <- function(hat, obs, kset = NULL,
                          validation_split = 0,
                          seed = NULL,
                          min_validation_rows = 10L) {
  # `kset` is reserved for T7.4 (temporal aggregation orders); v1 ignores it.

  p <- NCOL(obs)
  T_obs <- NROW(obs)

  if (NROW(hat) != T_obs) {
    cli_abort(
      "{.arg hat} has {NROW(hat)} rows, {.arg obs} has {T_obs} rows; must match.",
      call = NULL
    )
  }

  series_names <- if (!is.null(colnames(obs))) {
    colnames(obs)
  } else {
    paste0("S", seq_len(p))
  }
  # G5: alphabetically-sorted, frozen levels — keeps factor coding stable across
  # train/predict and across batches in T7.4.
  series_id_levels <- sort(unique(as.character(series_names)))

  hat_mat <- if (is.matrix(hat)) hat else as.matrix(hat)
  obs_mat <- if (is.matrix(obs)) obs else as.matrix(obs)

  # rbind hat p times; the same `hat` features are shared across series.
  X_stacked <- do.call(rbind, rep(list(hat_mat), p))
  y_stacked <- as.numeric(obs_mat)  # column-major: series 1, series 2, ...

  series_id_char <- rep(series_names, each = T_obs)
  series_id_factor <- factor(series_id_char, levels = series_id_levels)
  series_id_int    <- as.integer(series_id_factor)

  # G4: global train/valid split computed ONCE; reused across batches in T7.4.
  n_total <- T_obs * p
  if (validation_split > 0) {
    n_valid <- max(round(validation_split * n_total), 1L)
    if (n_valid < min_validation_rows) {
      cli_warn(
        c(
          "Validation set has only {n_valid} row{?s} (< {min_validation_rows}).",
          "i" = "Early stopping disabled."
        ),
        call = NULL
      )
      valid_idx <- integer(0L)
      train_idx <- seq_len(n_total)
    } else {
      if (!is.null(seed)) set.seed(seed)
      valid_idx <- sort(sample.int(n_total, n_valid))
      train_idx <- setdiff(seq_len(n_total), valid_idx)
    }
  } else {
    train_idx <- seq_len(n_total)
    valid_idx <- integer(0L)
  }

  list(
    X_stacked        = X_stacked,
    y_stacked        = y_stacked,
    series_id_factor = series_id_factor,
    series_id_int    = series_id_int,
    series_id_levels = series_id_levels,
    norm_params      = NULL,
    train_idx        = train_idx,
    valid_idx        = valid_idx
  )
}

#' Fit a global ML model across all bottom-series
#'
#' Stacks all series into a single training set with `series_id` as a
#' categorical feature, fits one model across all series, and returns an
#' `rml_g_fit` object. Per-series predictions are produced by the
#' (forthcoming) `predict.rml_g_fit` method (T7.5).
#'
#' @param approach character; one of `"lightgbm"`, `"xgboost"`, `"ranger"`,
#'   `"mlr3"`, `"catboost"`.
#' @param hat numeric matrix of features, dimension `T_obs x ncol_hat`.
#'   Shared across series.
#' @param obs numeric matrix of observations, dimension `T_obs x p`, where
#'   `p` is the number of bottom-series.
#' @param params named list of backend-specific hyperparameters.
#' @param seed integer seed for reproducibility.
#' @param early_stopping_rounds integer; `0` disables early stopping.
#' @param validation_split fraction of stacked rows reserved for validation
#'   (`0` disables).
#' @param ... passed to backend.
#' @return list of class `rml_g_fit` with elements `fit`, `approach`,
#'   `series_id_levels`, `feature_importance`, `ncol_hat`.
#' @examples
#' \dontrun{
#' agg_mat <- t(c(1, 1))
#' dimnames(agg_mat) <- list("A", c("B", "C"))
#' N_hat <- 50
#' ts_mean <- c(20, 10, 10)
#' hat <- matrix(rnorm(length(ts_mean) * N_hat, mean = ts_mean),
#'               N_hat, byrow = TRUE)
#' colnames(hat) <- unlist(dimnames(agg_mat))
#' obs <- matrix(rnorm(length(ts_mean[-1]) * N_hat, mean = ts_mean[-1]),
#'               N_hat, byrow = TRUE)
#' colnames(obs) <- colnames(agg_mat)
#' fit <- rml_g(approach = "ranger", hat = hat, obs = obs, seed = 42L)
#' }
#' @export
rml_g <- function(approach, hat, obs, params = NULL, seed = NULL,
                  early_stopping_rounds = 0L,
                  validation_split = 0,
                  ...) {
  class(approach) <- c(approach, class(approach))
  UseMethod("rml_g", approach)
}

#' @export
#' @method rml_g lightgbm
rml_g.lightgbm <- function(approach, hat, obs, params = NULL, seed = NULL,
                           early_stopping_rounds = 0L,
                           validation_split = 0,
                           ...) {
  stack <- .stack_series(hat, obs,
                         validation_split = validation_split,
                         seed = seed)

  # G6: integer-encoded categorical feature; lightgbm wants the column index
  # marked via `categorical_feature`.
  X_train <- cbind(stack$X_stacked[stack$train_idx, , drop = FALSE],
                   series_id = stack$series_id_int[stack$train_idx])
  y_train <- stack$y_stacked[stack$train_idx]

  cat_col_idx <- ncol(X_train)  # 1-based; the last column is series_id.

  lgb_params <- list(
    objective   = "regression",
    metric      = "rmse",
    num_threads = 1L,
    verbose     = -1L
  )
  if (!is.null(seed))   lgb_params$seed <- as.integer(seed)
  if (!is.null(params)) lgb_params <- utils::modifyList(lgb_params, params)
  nrounds <- if (!is.null(params$num_iteration)) params$num_iteration else 100L

  dtrain <- lightgbm::lgb.Dataset(
    data                = X_train,
    label               = y_train,
    categorical_feature = cat_col_idx,
    free_raw_data       = FALSE
  )

  valids <- list()
  es_rounds <- NULL
  if (length(stack$valid_idx) > 0L && early_stopping_rounds > 0L) {
    X_valid <- cbind(stack$X_stacked[stack$valid_idx, , drop = FALSE],
                     series_id = stack$series_id_int[stack$valid_idx])
    dvalid <- lightgbm::lgb.Dataset.create.valid(
      dtrain,
      data  = X_valid,
      label = stack$y_stacked[stack$valid_idx]
    )
    valids   <- list(valid = dvalid)
    es_rounds <- as.integer(early_stopping_rounds)
  }

  fit <- lightgbm::lgb.train(
    params                = lgb_params,
    data                  = dtrain,
    nrounds               = nrounds,
    valids                = valids,
    early_stopping_rounds = es_rounds,
    verbose               = -1L
  )

  feature_importance <- tryCatch(
    lightgbm::lgb.importance(fit, percentage = TRUE),
    error = function(e) NULL
  )

  structure(
    list(
      fit                = fit,
      approach           = "lightgbm",
      series_id_levels   = stack$series_id_levels,
      feature_importance = feature_importance,
      ncol_hat           = ncol(hat)
    ),
    class = "rml_g_fit"
  )
}

#' @export
#' @method rml_g xgboost
rml_g.xgboost <- function(approach, hat, obs, params = NULL, seed = NULL,
                          early_stopping_rounds = 0L,
                          validation_split = 0,
                          ...) {
  stack <- .stack_series(hat, obs,
                         validation_split = validation_split,
                         seed = seed)

  # xgboost has no native categorical handling in this version; use the
  # G5-frozen factor-as-integer encoding directly as a numeric column.
  X_train_mat <- cbind(stack$X_stacked[stack$train_idx, , drop = FALSE],
                       series_id = stack$series_id_int[stack$train_idx])

  dtrain <- xgboost::xgb.DMatrix(
    data  = X_train_mat,
    label = stack$y_stacked[stack$train_idx]
  )

  xgb_params <- list(
    objective = "reg:squarederror",
    nthread   = 1L,
    verbosity = 0L
  )
  if (!is.null(seed))   xgb_params$seed <- as.integer(seed)
  if (!is.null(params)) xgb_params <- utils::modifyList(xgb_params, params)
  nrounds <- if (!is.null(params$nrounds)) params$nrounds else 100L

  evals <- list()
  if (length(stack$valid_idx) > 0L && early_stopping_rounds > 0L) {
    X_valid_mat <- cbind(stack$X_stacked[stack$valid_idx, , drop = FALSE],
                         series_id = stack$series_id_int[stack$valid_idx])
    dvalid <- xgboost::xgb.DMatrix(
      data  = X_valid_mat,
      label = stack$y_stacked[stack$valid_idx]
    )
    evals <- list(valid = dvalid)
  }

  fit <- xgboost::xgb.train(
    params                = xgb_params,
    data                  = dtrain,
    nrounds               = nrounds,
    evals                 = evals,
    early_stopping_rounds = if (length(evals) > 0L) early_stopping_rounds else NULL,
    verbose               = 0L
  )

  feature_importance <- tryCatch(
    xgboost::xgb.importance(model = fit),
    error = function(e) NULL
  )

  structure(
    list(
      fit                = fit,
      approach           = "xgboost",
      series_id_levels   = stack$series_id_levels,
      feature_importance = feature_importance,
      ncol_hat           = ncol(hat)
    ),
    class = "rml_g_fit"
  )
}

#' @export
#' @method rml_g ranger
rml_g.ranger <- function(approach, hat, obs, params = NULL, seed = NULL,
                         early_stopping_rounds = 0L,
                         validation_split = 0,
                         ...) {
  if (early_stopping_rounds > 0L) {
    cli_inform(
      "ranger does not support early stopping; {.arg early_stopping_rounds} ignored.",
      call = NULL
    )
  }

  stack <- .stack_series(hat, obs,
                         validation_split = validation_split,
                         seed = seed)

  df_train <- data.frame(
    stack$X_stacked[stack$train_idx, , drop = FALSE],
    series_id   = stack$series_id_factor[stack$train_idx],
    .y          = stack$y_stacked[stack$train_idx],
    check.names = TRUE
  )

  ranger_params <- list(num.trees = 500L, num.threads = 1L, importance = "impurity")
  if (!is.null(seed))   ranger_params$seed <- as.integer(seed)
  if (!is.null(params)) ranger_params <- utils::modifyList(ranger_params, params)

  fit <- do.call(ranger::ranger, c(
    list(formula = .y ~ ., data = df_train, verbose = FALSE),
    ranger_params
  ))

  feature_importance <- tryCatch(fit$variable.importance, error = function(e) NULL)

  structure(
    list(
      fit                = fit,
      approach           = "ranger",
      series_id_levels   = stack$series_id_levels,
      feature_importance = feature_importance,
      ncol_hat           = ncol(hat)
    ),
    class = "rml_g_fit"
  )
}

#' @export
#' @method rml_g mlr3
rml_g.mlr3 <- function(approach, hat, obs, params = NULL, seed = NULL,
                       early_stopping_rounds = 0L,
                       validation_split = 0,
                       ...) {
  stack <- .stack_series(hat, obs,
                         validation_split = validation_split,
                         seed = seed)

  df_train <- data.frame(
    stack$X_stacked[stack$train_idx, , drop = FALSE],
    series_id   = stack$series_id_factor[stack$train_idx],
    .y          = stack$y_stacked[stack$train_idx],
    check.names = TRUE
  )

  task <- mlr3::TaskRegr$new(id = "rml_g", backend = df_train, target = ".y")

  learner_id <- if (!is.null(params$learner)) params$learner else "regr.ranger"
  learner    <- mlr3::lrn(learner_id, num.threads = 1L)
  if (!is.null(seed)) {
    try(learner$param_set$set_values(seed = as.integer(seed)), silent = TRUE)
  }

  learner$train(task)

  feature_importance <- tryCatch(learner$importance(), error = function(e) NULL)

  structure(
    list(
      fit                = learner,
      approach           = "mlr3",
      series_id_levels   = stack$series_id_levels,
      feature_importance = feature_importance,
      ncol_hat           = ncol(hat)
    ),
    class = "rml_g_fit"
  )
}

# =========================================================================
# T7.4: Chunked incremental training (warm-start + OOM fallback).
# =========================================================================

# Compute auto batch_size based on available RAM and per-series memory estimate.
# Returns an integer in [1, p].
.auto_batch_size <- function(T_obs, ncol_hat, approach = "lightgbm", p) {
  avail_bytes <- tryCatch({
    a <- available_ram_bytes()
    if (is.na(a)) 4e9 else a * 0.5
  }, error = function(e) 4e9)
  per_series_bytes <- as.numeric(T_obs) * as.numeric(ncol_hat + 1L) * 8 +
                      as.numeric(ncol_hat + 1L) * 64
  batch <- as.integer(floor(avail_bytes / per_series_bytes))
  max(1L, min(batch, as.integer(p)))
}

# Internal lightgbm batch fit with warm-start.
.rml_g_lightgbm_batch <- function(hat, obs_chunk, params, seed,
                                  nrounds, prev_model, ...) {
  stack <- .stack_series(hat, obs_chunk)
  X_train <- cbind(stack$X_stacked[stack$train_idx, , drop = FALSE],
                   series_id = stack$series_id_int[stack$train_idx])
  y_train <- stack$y_stacked[stack$train_idx]
  cat_col_idx <- ncol(X_train)

  lgb_params <- list(
    objective   = "regression",
    metric      = "rmse",
    num_threads = 1L,
    verbose     = -1L
  )
  if (!is.null(seed))   lgb_params$seed <- as.integer(seed)
  if (!is.null(params)) lgb_params <- utils::modifyList(lgb_params, params)

  dtrain <- lightgbm::lgb.Dataset(
    data                = X_train,
    label               = y_train,
    categorical_feature = cat_col_idx,
    free_raw_data       = FALSE
  )

  fit <- lightgbm::lgb.train(
    params     = lgb_params,
    data       = dtrain,
    nrounds    = nrounds,
    init_model = prev_model,
    verbose    = -1L
  )
  list(fit = fit)
}

# Internal xgboost batch fit with warm-start (xgb_model =).
.rml_g_xgboost_batch <- function(hat, obs_chunk, params, seed,
                                 nrounds, prev_model, ...) {
  stack <- .stack_series(hat, obs_chunk)
  X_train_mat <- cbind(stack$X_stacked[stack$train_idx, , drop = FALSE],
                       series_id = stack$series_id_int[stack$train_idx])
  dtrain <- xgboost::xgb.DMatrix(
    data  = X_train_mat,
    label = stack$y_stacked[stack$train_idx]
  )
  xgb_params <- list(
    objective = "reg:squarederror",
    nthread   = 1L,
    verbosity = 0L
  )
  if (!is.null(seed))   xgb_params$seed <- as.integer(seed)
  if (!is.null(params)) xgb_params <- utils::modifyList(xgb_params, params)

  fit <- xgboost::xgb.train(
    params    = xgb_params,
    data      = dtrain,
    nrounds   = nrounds,
    xgb_model = prev_model,
    verbose   = 0L
  )
  list(fit = fit)
}

# Run rml_g in chunks over series with warm-start (lightgbm/xgboost),
# OOM fallback (halve batch_size up to 3 retries), and per-batch checkpoints.
.run_chunked_rml_g <- function(approach, hat, obs, params, seed,
                               early_stopping_rounds, validation_split,
                               batch_size, chunk_strategy,
                               batch_checkpoint_dir, nrounds_per_batch, ...) {
  p        <- NCOL(obs)
  T_obs    <- NROW(obs)
  ncol_hat <- NCOL(hat)

  bs <- if (identical(batch_size, "auto")) {
    .auto_batch_size(T_obs, ncol_hat, approach, p)
  } else {
    as.integer(batch_size)
  }

  # catboost has no model-continuation in the R API.
  if (approach == "catboost" && bs < p) {
    cli_abort(
      paste0(
        "Chunked training (batch_size < p) is not supported for ",
        "{.val catboost}: the catboost R API has no model-continuation ",
        "(warm-start). Use {.val lightgbm} or {.val xgboost} for ",
        "incremental training."
      ),
      call = NULL
    )
  }

  # Single-batch: delegate to rml_g (no warm-start needed).
  if (bs >= p) {
    return(rml_g(approach = approach, hat = hat, obs = obs,
                 params = params, seed = seed,
                 early_stopping_rounds = early_stopping_rounds,
                 validation_split = validation_split, ...))
  }

  chunk_strategy <- match.arg(chunk_strategy, c("sequential", "random"))
  series_idx <- seq_len(p)
  if (chunk_strategy == "random") {
    if (!is.null(seed)) set.seed(seed)
    series_idx <- sample(series_idx)
  }
  chunks <- split(series_idx, ceiling(seq_along(series_idx) / bs))

  # Per-batch early stopping with global validation split: not yet supported.
  if (validation_split > 0 && early_stopping_rounds > 0L) {
    cli_inform(
      paste0(
        "Per-batch early stopping with global validation_split is not yet ",
        "supported in chunked mode. Early stopping disabled."
      )
    )
    early_stopping_rounds <- 0L
    validation_split      <- 0
  }

  if (!is.null(batch_checkpoint_dir)) {
    dir.create(batch_checkpoint_dir, showWarnings = FALSE, recursive = TRUE)
  }

  prev_model        <- NULL
  best_iter_history <- vector("list", length(chunks))
  batch_indices     <- chunks
  retry_bs          <- bs

  for (chunk_i in seq_along(chunks)) {
    idx       <- chunks[[chunk_i]]
    obs_chunk <- obs[, idx, drop = FALSE]

    ckpt_path <- if (!is.null(batch_checkpoint_dir)) {
      file.path(batch_checkpoint_dir, paste0("batch_", chunk_i, ".qs2"))
    } else {
      NULL
    }

    if (!is.null(ckpt_path) && file.exists(ckpt_path)) {
      prev_model <- qs2::qs_read(ckpt_path)
      next
    }

    fit_chunk <- NULL
    for (attempt in 1:4) {
      fit_chunk <- tryCatch({
        if (approach == "lightgbm") {
          .rml_g_lightgbm_batch(hat, obs_chunk, params, seed,
                                nrounds_per_batch, prev_model = prev_model, ...)
        } else if (approach == "xgboost") {
          .rml_g_xgboost_batch(hat, obs_chunk, params, seed,
                               nrounds_per_batch, prev_model = prev_model, ...)
        } else {
          # ranger / mlr3: no warm-start. Fit fresh on chunk.
          rml_g(approach = approach, hat = hat, obs = obs_chunk,
                params = params, seed = seed, ...)
        }
      }, error = function(e) {
        if (grepl("bad_alloc|cannot allocate|out of memory",
                  conditionMessage(e), ignore.case = TRUE) &&
            attempt <= 3L) {
          retry_bs <<- max(1L, as.integer(retry_bs / 2L))
          cli_alert_warning(
            "OOM in batch {chunk_i}, halving batch_size to {retry_bs} (attempt {attempt}/3)."
          )
          NULL
        } else {
          stop(e)
        }
      })
      if (!is.null(fit_chunk)) break
    }

    if (is.null(fit_chunk)) {
      cli_abort(
        "OOM: failed after 3 retries. Reduce batch_size or use a machine with more RAM."
      )
    }

    bi <- NULL
    if (!is.null(fit_chunk$fit$best_iter)) {
      bi <- fit_chunk$fit$best_iter
    } else if (!is.null(fit_chunk$fit$best_iteration)) {
      bi <- fit_chunk$fit$best_iteration
    }
    best_iter_history[[chunk_i]] <- bi

    if (!is.null(ckpt_path)) {
      qs2::qs_save(
        fit_chunk$fit, ckpt_path,
        nthreads = min(parallel::detectCores(), 4L)
      )
    }

    prev_model <- fit_chunk$fit
  }

  structure(
    list(
      fit                = prev_model,
      approach           = approach,
      series_id_levels   = colnames(obs),
      feature_importance = NULL,
      ncol_hat           = ncol_hat,
      best_iter_history  = best_iter_history,
      batch_indices      = batch_indices
    ),
    class = "rml_g_fit"
  )
}

#' Cross-sectional reconciliation with a global ML model
#'
#' Normalizes `hat` (optionally), fits a single global ML model across all
#' bottom-series via [rml_g], stores reconciliation metadata on the returned
#' `rml_g_fit` object, and returns it. Reconciled forecasts are produced by
#' calling [predict.rml_g_fit] (T7.5) on the returned object.
#'
#' @param base base forecasts matrix (`n x h`); rows = all series (upper +
#'   bottom), columns = forecast horizons. Required for API consistency with
#'   the series-level wrappers; not used for fitting.
#' @param hat numeric feature matrix (`T_obs x ncol_hat`), shared across series.
#' @param obs numeric observation matrix (`T_obs x p`), one column per
#'   bottom-level series.
#' @param agg_mat cross-sectional aggregation matrix (`n_agg x p`).
#' @param approach character; ML backend. One of `"lightgbm"`, `"xgboost"`,
#'   `"ranger"`, `"mlr3"`, `"catboost"`.
#' @param normalize pre-normalization applied to `hat` before fitting:
#'   `"none"` (default), `"zscore"`, or `"robust"`.
#' @param scale_fn scale estimator for `normalize = "robust"`. See
#'   [normalize_stack].
#' @param params named list of backend hyperparameters.
#' @param seed integer for reproducibility.
#' @param early_stopping_rounds integer; `0` disables early stopping.
#' @param validation_split fraction of stacked rows reserved for validation
#'   (`0` disables).
#' @param batch_size integer or `"auto"`. When non-`NULL`, series are chunked
#'   into batches of this size and the model is updated incrementally (warm
#'   start). `"auto"` uses a memory-based heuristic. Not supported for
#'   catboost.
#' @param chunk_strategy `"sequential"` (default) or `"random"`. Controls how
#'   series are assigned to batches.
#' @param batch_checkpoint_dir character path for saving intermediate batch
#'   model checkpoints. `NULL` disables batch checkpointing.
#' @param nrounds_per_batch integer; additional boosting rounds added per batch
#'   when using incremental training. Default 50.
#' @param ... passed to [rml_g].
#' @return `rml_g_fit` object with additional fields `agg_mat`, `norm_params`,
#'   and `framework = "cs"`.
#' @examples
#' \dontrun{
#' agg_mat <- t(c(1, 1))
#' dimnames(agg_mat) <- list("A", c("B", "C"))
#' N_hat <- 50; h <- 2
#' ts_mean <- c(20, 10, 10)
#' hat <- matrix(rnorm(length(ts_mean) * N_hat, mean = ts_mean),
#'               N_hat, byrow = TRUE)
#' colnames(hat) <- unlist(dimnames(agg_mat))
#' obs <- matrix(rnorm(length(ts_mean[-1]) * N_hat, mean = ts_mean[-1]),
#'               N_hat, byrow = TRUE)
#' colnames(obs) <- colnames(agg_mat)
#' base <- matrix(rnorm(length(ts_mean) * h, mean = ts_mean), h, byrow = TRUE)
#' colnames(base) <- unlist(dimnames(agg_mat))
#' fit <- csrml_g(base = base, hat = hat, obs = obs, agg_mat = agg_mat,
#'                approach = "lightgbm", seed = 42L)
#' }
#' @export
csrml_g <- function(base, hat, obs, agg_mat,
                    approach = "lightgbm",
                    normalize = c("none", "zscore", "robust"),
                    scale_fn = "gmd",
                    params = NULL, seed = NULL,
                    early_stopping_rounds = 0L,
                    validation_split = 0,
                    batch_size = NULL,
                    chunk_strategy = c("sequential", "random"),
                    batch_checkpoint_dir = NULL,
                    nrounds_per_batch = 50L,
                    ...) {
  normalize <- match.arg(normalize)
  hat_norm <- hat
  norm_params <- NULL
  if (normalize != "none") {
    nr <- normalize_stack(hat, method = normalize, scale_fn = scale_fn)
    hat_norm    <- nr$X_norm
    norm_params <- nr
  }
  fit_obj <- if (!is.null(batch_size)) {
    .run_chunked_rml_g(approach = approach, hat = hat_norm, obs = obs,
                       params = params, seed = seed,
                       early_stopping_rounds = early_stopping_rounds,
                       validation_split = validation_split,
                       batch_size = batch_size,
                       chunk_strategy = chunk_strategy,
                       batch_checkpoint_dir = batch_checkpoint_dir,
                       nrounds_per_batch = nrounds_per_batch, ...)
  } else {
    rml_g(approach = approach, hat = hat_norm, obs = obs,
          params = params, seed = seed,
          early_stopping_rounds = early_stopping_rounds,
          validation_split = validation_split, ...)
  }
  fit_obj$agg_mat     <- agg_mat
  fit_obj$norm_params <- norm_params
  fit_obj$framework   <- "cs"
  fit_obj
}

#' Temporal reconciliation with a global ML model
#'
#' Normalizes `hat` (optionally), fits a single global ML model across all
#' temporal aggregation levels via [rml_g], stores reconciliation metadata on
#' the returned `rml_g_fit` object, and returns it. Reconciled forecasts are
#' produced by calling [predict.rml_g_fit] (T7.5) on the returned object.
#'
#' @param base base forecasts matrix (`n x h`); rows = all temporal levels,
#'   columns = forecast horizons. Required for API consistency; not used for
#'   fitting.
#' @param hat numeric feature matrix (`T_obs x ncol_hat`), shared across series.
#' @param obs numeric observation matrix (`T_obs x p`).
#' @param agg_order integer vector of temporal aggregation orders (e.g.
#'   `c(4L, 2L, 1L)` for annual, semi-annual, quarterly data).
#' @param approach character; ML backend. One of `"lightgbm"`, `"xgboost"`,
#'   `"ranger"`, `"mlr3"`, `"catboost"`.
#' @param normalize pre-normalization applied to `hat` before fitting:
#'   `"none"` (default), `"zscore"`, or `"robust"`.
#' @param scale_fn scale estimator for `normalize = "robust"`. See
#'   [normalize_stack].
#' @param params named list of backend hyperparameters.
#' @param seed integer for reproducibility.
#' @param early_stopping_rounds integer; `0` disables early stopping.
#' @param validation_split fraction of stacked rows reserved for validation
#'   (`0` disables).
#' @param batch_size integer or `"auto"`. When non-`NULL`, temporal levels are
#'   chunked into batches for incremental training. `"auto"` uses a
#'   memory-based heuristic. Not supported for catboost.
#' @param chunk_strategy `"sequential"` (default) or `"random"`.
#' @param batch_checkpoint_dir character path for batch model checkpoints.
#'   `NULL` disables.
#' @param nrounds_per_batch integer; boosting rounds added per batch. Default 50.
#' @param ... passed to [rml_g].
#' @return `rml_g_fit` object with additional fields `agg_order`, `norm_params`,
#'   and `framework = "te"`.
#' @examples
#' \dontrun{
#' agg_order <- c(4L, 2L, 1L)  # annual, semi-annual, quarterly
#' n_levels <- sum(agg_order)
#' N_hat <- 40; h <- 1
#' hat <- matrix(rnorm(n_levels * N_hat), N_hat, n_levels)
#' obs <- matrix(rnorm(N_hat), N_hat, 1L)  # one bottom-level series
#' base <- matrix(rnorm(n_levels * h), h, n_levels)
#' fit <- terml_g(base = base, hat = hat, obs = obs,
#'                agg_order = agg_order, approach = "lightgbm", seed = 1L)
#' }
#' @export
terml_g <- function(base, hat, obs, agg_order,
                    approach = "lightgbm",
                    normalize = c("none", "zscore", "robust"),
                    scale_fn = "gmd",
                    params = NULL, seed = NULL,
                    early_stopping_rounds = 0L,
                    validation_split = 0,
                    batch_size = NULL,
                    chunk_strategy = c("sequential", "random"),
                    batch_checkpoint_dir = NULL,
                    nrounds_per_batch = 50L,
                    ...) {
  normalize <- match.arg(normalize)
  hat_norm <- hat
  norm_params <- NULL
  if (normalize != "none") {
    nr <- normalize_stack(hat, method = normalize, scale_fn = scale_fn)
    hat_norm    <- nr$X_norm
    norm_params <- nr
  }
  fit_obj <- if (!is.null(batch_size)) {
    .run_chunked_rml_g(approach = approach, hat = hat_norm, obs = obs,
                       params = params, seed = seed,
                       early_stopping_rounds = early_stopping_rounds,
                       validation_split = validation_split,
                       batch_size = batch_size,
                       chunk_strategy = chunk_strategy,
                       batch_checkpoint_dir = batch_checkpoint_dir,
                       nrounds_per_batch = nrounds_per_batch, ...)
  } else {
    rml_g(approach = approach, hat = hat_norm, obs = obs,
          params = params, seed = seed,
          early_stopping_rounds = early_stopping_rounds,
          validation_split = validation_split, ...)
  }
  fit_obj$agg_order   <- agg_order
  fit_obj$norm_params <- norm_params
  fit_obj$framework   <- "te"
  fit_obj
}

#' Cross-temporal reconciliation with a global ML model
#'
#' Normalizes `hat` (optionally), fits a single global ML model across all
#' cross-temporal series via [rml_g], stores reconciliation metadata on the
#' returned `rml_g_fit` object, and returns it. Reconciled forecasts are
#' produced by calling [predict.rml_g_fit] (T7.5) on the returned object.
#'
#' @param base base forecasts matrix (`n x h`); rows = all series/levels,
#'   columns = forecast horizons. Required for API consistency; not used for
#'   fitting.
#' @param hat numeric feature matrix (`T_obs x ncol_hat`), shared across series.
#' @param obs numeric observation matrix (`T_obs x p`).
#' @param agg_mat cross-sectional aggregation matrix (`n_agg x p`).
#' @param agg_order integer vector of temporal aggregation orders.
#' @param approach character; ML backend. One of `"lightgbm"`, `"xgboost"`,
#'   `"ranger"`, `"mlr3"`, `"catboost"`.
#' @param normalize pre-normalization applied to `hat` before fitting:
#'   `"none"` (default), `"zscore"`, or `"robust"`.
#' @param scale_fn scale estimator for `normalize = "robust"`. See
#'   [normalize_stack].
#' @param params named list of backend hyperparameters.
#' @param seed integer for reproducibility.
#' @param early_stopping_rounds integer; `0` disables early stopping.
#' @param validation_split fraction of stacked rows reserved for validation
#'   (`0` disables).
#' @param batch_size integer or `"auto"`. When non-`NULL`, series are chunked
#'   for incremental training. `"auto"` uses a memory-based heuristic. Not
#'   supported for catboost.
#' @param chunk_strategy `"sequential"` (default) or `"random"`.
#' @param batch_checkpoint_dir character path for batch model checkpoints.
#'   `NULL` disables.
#' @param nrounds_per_batch integer; boosting rounds added per batch. Default 50.
#' @param ... passed to [rml_g].
#' @return `rml_g_fit` object with additional fields `agg_mat`, `agg_order`,
#'   `norm_params`, and `framework = "ct"`.
#' @examples
#' \dontrun{
#' agg_mat <- t(c(1, 1))
#' dimnames(agg_mat) <- list("A", c("B", "C"))
#' agg_order <- c(4L, 1L)
#' p <- NCOL(agg_mat)
#' n_cs <- nrow(agg_mat) + p
#' n_te <- sum(agg_order)
#' N_hat <- 40; h <- 1
#' hat <- matrix(rnorm(n_te * N_hat), N_hat, n_te)
#' obs <- matrix(rnorm(p * N_hat), N_hat, p)
#' colnames(obs) <- colnames(agg_mat)
#' base <- matrix(rnorm(n_cs * n_te * h), h, n_cs * n_te)
#' fit <- ctrml_g(base = base, hat = hat, obs = obs,
#'                agg_mat = agg_mat, agg_order = agg_order,
#'                approach = "lightgbm", seed = 1L)
#' }
#' @export
ctrml_g <- function(base, hat, obs, agg_mat, agg_order,
                    approach = "lightgbm",
                    normalize = c("none", "zscore", "robust"),
                    scale_fn = "gmd",
                    params = NULL, seed = NULL,
                    early_stopping_rounds = 0L,
                    validation_split = 0,
                    batch_size = NULL,
                    chunk_strategy = c("sequential", "random"),
                    batch_checkpoint_dir = NULL,
                    nrounds_per_batch = 50L,
                    ...) {
  normalize <- match.arg(normalize)
  hat_norm <- hat
  norm_params <- NULL
  if (normalize != "none") {
    nr <- normalize_stack(hat, method = normalize, scale_fn = scale_fn)
    hat_norm    <- nr$X_norm
    norm_params <- nr
  }
  fit_obj <- if (!is.null(batch_size)) {
    .run_chunked_rml_g(approach = approach, hat = hat_norm, obs = obs,
                       params = params, seed = seed,
                       early_stopping_rounds = early_stopping_rounds,
                       validation_split = validation_split,
                       batch_size = batch_size,
                       chunk_strategy = chunk_strategy,
                       batch_checkpoint_dir = batch_checkpoint_dir,
                       nrounds_per_batch = nrounds_per_batch, ...)
  } else {
    rml_g(approach = approach, hat = hat_norm, obs = obs,
          params = params, seed = seed,
          early_stopping_rounds = early_stopping_rounds,
          validation_split = validation_split, ...)
  }
  fit_obj$agg_mat     <- agg_mat
  fit_obj$agg_order   <- agg_order
  fit_obj$norm_params <- norm_params
  fit_obj$framework   <- "ct"
  fit_obj
}

#' @export
#' @method rml_g catboost
rml_g.catboost <- function(approach, hat, obs, params = NULL, seed = NULL,
                           early_stopping_rounds = 0L,
                           validation_split = 0,
                           ...) {
  if (!requireNamespace("catboost", quietly = TRUE)) {
    cli_abort(
      "Package {.pkg catboost} required for {.code approach = \"catboost\"}.",
      call = NULL
    )
  }

  stack <- .stack_series(hat, obs,
                         validation_split = validation_split,
                         seed = seed)

  X_train <- cbind(stack$X_stacked[stack$train_idx, , drop = FALSE],
                   series_id = stack$series_id_int[stack$train_idx])

  cat_feat_idx <- ncol(X_train) - 1L  # 0-based for catboost.

  pool_train <- catboost::catboost.load_pool(
    data         = X_train,
    label        = stack$y_stacked[stack$train_idx],
    cat_features = cat_feat_idx
  )

  cb_params <- list(
    loss_function = "RMSE",
    iterations    = if (!is.null(params$iterations)) params$iterations else 100L,
    thread_count  = 1L,
    logging_level = "Silent"
  )
  if (!is.null(seed))   cb_params$random_seed <- as.integer(seed)
  if (!is.null(params)) cb_params <- utils::modifyList(cb_params, params)

  pool_valid <- NULL
  if (length(stack$valid_idx) > 0L && early_stopping_rounds > 0L) {
    X_valid <- cbind(stack$X_stacked[stack$valid_idx, , drop = FALSE],
                     series_id = stack$series_id_int[stack$valid_idx])
    pool_valid <- catboost::catboost.load_pool(
      data         = X_valid,
      label        = stack$y_stacked[stack$valid_idx],
      cat_features = cat_feat_idx
    )
    cb_params$od_type <- "Iter"
    cb_params$od_wait <- as.integer(early_stopping_rounds)
  }

  fit <- catboost::catboost.train(pool_train, pool_valid, params = cb_params)

  feature_importance <- tryCatch(
    catboost::catboost.get_feature_importance(fit),
    error = function(e) NULL
  )

  structure(
    list(
      fit                = fit,
      approach           = "catboost",
      series_id_levels   = stack$series_id_levels,
      feature_importance = feature_importance,
      ncol_hat           = ncol(hat)
    ),
    class = "rml_g_fit"
  )
}

# =============================================================================
# T7.5: S3 methods for rml_g_fit
# =============================================================================

`%||%` <- function(x, y) if (is.null(x)) y else x

#' Predict from a global ML fit object
#'
#' @param object An `rml_g_fit` object returned by [rml_g], [csrml_g],
#'   [terml_g], or [ctrml_g].
#' @param newdata Numeric matrix (`n_rows x ncol_hat`) of features.  Must
#'   have the same column layout as the `hat` matrix used during training
#'   (i.e. `ncol_hat` columns, no `series_id` column).
#' @param series_id Character or factor vector of length `n_rows` giving the
#'   series identifier for each row of `newdata`.  All values must appear in
#'   `object$series_id_levels`.  If `NULL`, all training-level series are
#'   cycled in order (one row of `newdata` replicated for every series level).
#' @param ... Ignored.
#' @return Numeric vector of predictions, one per row of the expanded
#'   `newdata` (after `series_id` replication when `series_id = NULL`).
#' @export
#' @method predict rml_g_fit
predict.rml_g_fit <- function(object, newdata, series_id = NULL, ...) {
  if (missing(newdata) || is.null(newdata)) {
    cli_abort("{.arg newdata} is required.", call = NULL)
  }
  newdata <- as.matrix(newdata)

  # --- series_id resolution -------------------------------------------------
  if (is.null(series_id)) {
    # Broadcast: replicate each row of newdata once per training series level.
    p <- length(object$series_id_levels)
    series_id_chr <- rep(object$series_id_levels, each = NROW(newdata))
    newdata <- newdata[rep(seq_len(NROW(newdata)), times = p), , drop = FALSE]
  } else {
    series_id_chr <- as.character(series_id)
    unknown <- setdiff(series_id_chr, as.character(object$series_id_levels))
    if (length(unknown) > 0L) {
      cli_abort(
        c(
          "Unknown {.arg series_id} level{?s}: {.val {unknown}}.",
          "i" = "Model was trained on: {.val {as.character(object$series_id_levels)}}."
        ),
        call = NULL
      )
    }
  }

  series_id_factor <- factor(series_id_chr, levels = object$series_id_levels)
  series_id_int    <- as.integer(series_id_factor)

  # --- backend dispatch -----------------------------------------------------
  switch(
    object$approach,
    "lightgbm" = {
      X_pred <- cbind(newdata, series_id = series_id_int)
      predict(object$fit, X_pred)
    },
    "xgboost" = {
      X_pred  <- cbind(newdata, series_id = series_id_int)
      dpred   <- xgboost::xgb.DMatrix(X_pred)
      predict(object$fit, dpred)
    },
    "ranger" = {
      df_pred <- data.frame(
        newdata,
        series_id   = series_id_factor,
        check.names = TRUE
      )
      predict(object$fit, data = df_pred, num.threads = 1L)$predictions
    },
    "mlr3" = {
      df_pred <- data.frame(
        newdata,
        series_id   = series_id_factor,
        check.names = TRUE
      )
      object$fit$predict_newdata(df_pred)$response
    },
    "catboost" = {
      if (!requireNamespace("catboost", quietly = TRUE)) {
        cli_abort("Package {.pkg catboost} is required for this backend.",
                  call = NULL)
      }
      X_pred    <- cbind(newdata, series_id = series_id_int)
      cat_idx   <- ncol(X_pred)  # 1-based; catboost expects 0-based
      pool_pred <- catboost::catboost.load_pool(
        data         = X_pred,
        cat_features = cat_idx - 1L
      )
      catboost::catboost.predict(object$fit, pool_pred)
    },
    cli_abort("Unknown approach: {.val {object$approach}}", call = NULL)
  )
}

#' Print an rml_g_fit object
#'
#' @param x An `rml_g_fit` object.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
#' @method print rml_g_fit
print.rml_g_fit <- function(x, ...) {
  cli_h1("Global ML Fit ({.cls rml_g_fit})")
  top5 <- utils::head(as.character(x$series_id_levels), 5L)
  suffix <- if (length(x$series_id_levels) > 5L) ", ..." else ""
  cli_bullets(c(
    "*" = "Approach:  {.val {x$approach}}",
    "*" = "Framework: {.val {x$framework %||% 'unknown'}}",
    "*" = paste0("Series:    ", length(x$series_id_levels),
                 " (levels: ", paste(top5, collapse = ", "), suffix, ")"),
    "*" = "Features:  {x$ncol_hat} (+ 1 series_id column)"
  ))
  if (!is.null(x$best_iter_history) && length(x$best_iter_history) > 0L) {
    iters <- unlist(x$best_iter_history)
    iters <- iters[!is.null(iters)]
    if (length(iters) > 0L) {
      cli_bullets(c("*" = "Best iters per batch: {iters}"))
    }
  }
  invisible(x)
}

#' Summarise an rml_g_fit object
#'
#' Prints the object header (via [print.rml_g_fit]) and, when available,
#' the top 10 most important features.
#'
#' @param object An `rml_g_fit` object.
#' @param ... Ignored.
#' @return `object`, invisibly.
#' @export
#' @method summary rml_g_fit
summary.rml_g_fit <- function(object, ...) {
  print(object)
  fi <- object$feature_importance
  if (is.null(fi) || (is.data.frame(fi) && NROW(fi) == 0L) ||
      (is.numeric(fi) && length(fi) == 0L)) {
    cli_inform(
      "Feature importance not available for this backend/configuration.",
      call = NULL
    )
  } else {
    cli_h2("Feature Importance (top 10)")
    if (is.data.frame(fi)) {
      print(utils::head(fi, 10L))
    } else {
      top10 <- sort(fi, decreasing = TRUE)[seq_len(min(10L, length(fi)))]
      print(top10)
    }
  }
  invisible(object)
}
