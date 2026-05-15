rml <- function(
  approach,
  base = NULL,
  obs = NULL,
  hat = NULL,
  sel_mat = NULL,
  fit = NULL,
  params = NULL,
  seed = NULL,
  keep_cols = NULL,
  kset = NULL,
  h = NULL,
  h_base = NULL,
  n = NULL,
  checkpoint = "auto",
  ...
) {
  class_base <- approach
  class(approach) <- c(class(approach), class_base)

  # check input
  if (is.null(obs) | is.null(hat) | is.null(sel_mat)) {
    if (is.null(fit) | is.null(base)) {
      cli_abort(
        c(
          "Mandatory arguments:",
          "1. {.arg obs}, {.arg hat}, and {.arg sel_mat};",
          "2. {.arg obs}, {.arg hat}, {.arg base} and {.arg sel_mat};",
          "3. {.arg fit}, {.arg base}."
        ),
        call = NULL
      )
    }
  }

  if (is.null(fit)) {
    if (!is.null(names(hat))) names(hat) <- NULL
    if (!is.null(dimnames(hat))) dimnames(hat) <- NULL
    if (!is.null(names(obs))) names(obs) <- NULL
    if (!is.null(dimnames(obs))) dimnames(obs) <- NULL
    p <- NCOL(obs)

    # T5: set.seed once per rml() call (was a silent no-op prior to 2.0.0).
    # Placed inside the training branch so predict-reuse calls do not perturb
    # the user's RNG state.
    if (!is.null(seed)) {
      set.seed(seed)
    }
  } else {
    # sel_mat <- fit$sel_mat
    p <- length(fit$fit)
  }

  # T6 — Resolve tri-mode `checkpoint` into NULL (disabled) or a directory.
  # Only meaningful during training (fit=NULL). On fit reuse, the stored fit
  # already encodes whether models are paths or in-memory; we never re-decide.
  checkpoint_dir <- if (is.null(fit)) {
    resolve_checkpoint(checkpoint, hat, class_base, p)
  } else {
    NULL
  }

  if (!is.null(base)) {
    if (!is.null(names(base))) names(base) <- NULL
    if (!is.null(dimnames(base))) dimnames(base) <- NULL
  }

  # T4/T5: precompute keep_cols + col_map once before lapply.
  # If `keep_cols` is supplied by the caller (T5 path), hat/base are ALREADY
  # column-sliced and full feature width must be derived from sel_mat dims.
  # Otherwise (legacy/csrml path) derive keep_cols from sel_mat and slice here.
  # T3: when `kset` is provided (mfh defer), hat is raw (NCOL(hat) != feature
  # width). Compute feature width from kset+n (ctrml mfh) or length(kset) is
  # wrong — for mfh the feature width is `n*kt` (ctrml) or `kt` (terml).
  defer_ctrml_mfh_pre <- !is.null(kset) && !is.null(n)
  defer_terml_mfh_pre <- !is.null(kset) && is.null(n)
  if (is.null(keep_cols)) {
    active_ncol <- if (defer_ctrml_mfh_pre) {
      # mat2hmat output has h x (n*kt) cols; kt = sum(max(kset)/kset).
      kt_local <- sum(max(kset) / kset)
      n * kt_local
    } else if (defer_terml_mfh_pre) {
      sum(max(kset) / kset) # kt
    } else if (!is.null(hat)) {
      NCOL(hat)
    } else {
      NCOL(base)
    }
    keep_cols <- sel_mat_keep_cols(sel_mat, active_ncol)
    slice <- length(keep_cols) < active_ncol
    if (slice && !defer_ctrml_mfh_pre && !defer_terml_mfh_pre) {
      col_map <- rep(NA_integer_, active_ncol)
      col_map[keep_cols] <- seq_along(keep_cols)
      if (!is.null(hat))  hat  <- hat[,  keep_cols, drop = FALSE]
      if (!is.null(base)) base <- base[, keep_cols, drop = FALSE]
    } else {
      col_map <- NULL
    }
    # T3: when deferring, never slice the raw hat with keep_cols (column
    # semantics differ); keep_cols set above is used only to derive global_id
    # via the standard sel_mat path. Reset to NULL so the loop body treats id
    # as global indices into the deferred-expanded feature space.
    if (defer_ctrml_mfh_pre || defer_terml_mfh_pre) {
      keep_cols <- NULL
    }
  } else {
    # T5: hat/base pre-sliced. active_ncol = full feature count from sel_mat.
    active_ncol <- if (length(sel_mat) == 1) {
      if (!is.null(hat)) NCOL(hat) else NCOL(base)
    } else if (is(sel_mat, "sparseVector")) {
      length(sel_mat)
    } else {
      NROW(sel_mat)
    }
    col_map <- rep(NA_integer_, active_ncol)
    col_map[keep_cols] <- seq_along(keep_cols)
  }

  # T6 — for-loop replaces lapply so we can serialize each `tmp$fit` to disk
  # immediately (when checkpoint_dir != NULL) and replace it with a path
  # string, then drop the in-memory copy and explicit-gc before the next i.
  # Net effect: peak RSS = 1 live model + (p-1) path strings, instead of p
  # live models. Predict (fit reuse) uses get_fit_i() for lazy reload.
  # spd.15: hoist global_id computation before the loop (loop-invariant for
  # scalar/vector sel_mat; per-column only for matrix sel_mat).
  global_id_list <- if (length(sel_mat) == 1) {
    rep(list(seq_len(active_ncol)), p)
  } else if (is(sel_mat, "sparseVector") || NCOL(sel_mat) == 1) {
    v <- which(as.numeric(if (is(sel_mat, "sparseVector")) sel_mat else sel_mat[, 1]) != 0)
    rep(list(v), p)
  } else {
    lapply(seq_len(p), function(j) which(sel_mat[, j] != 0))
  }

  # cli progress: respect forecoml.progress option; default to interactive().
  loop_seq <- if (isTRUE(getOption("forecoml.progress", interactive()))) {
    cli::cli_progress_along(seq_len(p), name = "Training per-series models", clear = TRUE)
  } else {
    seq_len(p)
  }

  # T3: deferred mfh expansion. When `kset != NULL`, hat/base are passed in
  # raw (un-expanded) form and the per-series feature block is materialized
  # only inside the loop body, then discarded. Peak memory drops from
  # O(full hmat across iterations) to O(one series block per iter).
  # - kset + h + n  -> ctrml mfh: hat is n x (h*kt) cross-temporal matrix.
  # - kset + h      -> terml mfh: hat is a length h*kt vector (or 1 x L matrix).
  defer_kset <- !is.null(kset)
  defer_ctrml_mfh <- defer_kset && !is.null(n)
  defer_terml_mfh <- defer_kset && is.null(n)
  # T3: in ctrml/terml, training uses h (derived from hat) and prediction uses
  # h_base (derived from base) which differ when forecast horizon != train h.
  # If one is NULL fall back to the other (single-h calls; predict-only path).
  h_train_eff <- if (!is.null(h)) h else h_base
  h_base_eff  <- if (!is.null(h_base)) h_base else h
  hat_vec_terml <- if (defer_terml_mfh && !is.null(hat)) as.vector(hat) else NULL
  base_vec_terml <- if (defer_terml_mfh && !is.null(base)) as.vector(base) else NULL

  out <- vector("list", p)
  for (i in loop_seq) {
    global_id <- global_id_list[[i]]
    id <- if (is.null(col_map)) global_id else { x <- col_map[global_id]; x[!is.na(x)] }

    if (is.null(fit)) {
      y <- obs[, i]
      X <- if (defer_ctrml_mfh) {
        mat2hmat_cols(hat, h = h_train_eff, kset = kset, n = n, cols = id)
      } else if (defer_terml_mfh) {
        vec2hmat_cols(hat_vec_terml, h = h_train_eff, kset = kset, cols = id)
      } else {
        hat[, id, drop = FALSE]
      }
      fit_i <- NULL

      # spd.2: skip na.omit allocation when no NAs present.
      if (anyNA(X)) {
        X <- na.omit(X)
        if (length(attr(X, "na.action")) > 0) {
          if (NROW(X) == 0) {
            cli_abort(
              paste0(
                "All the predictor variables for series {.val {i}} contain ",
                "{.code NA} values after applying {.fn na.omit}. ",
                "Please check your {.arg hat} input or consider using ",
                "another {.arg features} option."
              ),
              call = NULL
            )
          }
          y <- y[-attr(X, "na.action")]
        }
      }
    } else {
      y <- X <- NULL
      # T6 — lazy-load: fit$fit[[i]] may be a path (checkpointed) or model.
      fit_i <- get_fit_i(fit, i)
    }

    if (!is.null(base)) {
      Xtest <- if (defer_ctrml_mfh) {
        mat2hmat_cols(base, h = h_base_eff, kset = kset, n = n, cols = id)
      } else if (defer_terml_mfh) {
        vec2hmat_cols(base_vec_terml, h = h_base_eff, kset = kset, cols = id)
      } else {
        base[, id, drop = FALSE]
      }
    } else {
      Xtest <- NULL
    }

    tmp <- .rml(
      approach = approach,
      y = y,
      X = X,
      Xtest = Xtest,
      fit = fit_i,
      params = params,
      ...
    )

    # T6 — Serialize freshly-trained fit to disk, replace with path. Skip
    # when no training happened (fit reuse: tmp$fit IS the loaded model and
    # we don't want to re-serialize it; the persistent path is already in
    # fit$fit[[i]]).
    if (!is.null(checkpoint_dir) && is.null(fit)) {
      path_i <- serialize_fit(tmp$fit, checkpoint_dir, i, class_base)
      tmp$fit <- path_i
    }

    # mw3.3: on predict-reuse, store only bts so the deserialized model is
    # released when `tmp` goes out of scope at end of this iteration, rather
    # than being retained in out[[i]]$fit until the loop finishes.
    out[[i]] <- if (is.null(fit)) tmp else list(bts = tmp$bts)
    rm(X, y, Xtest, fit_i)
    # spd.3/8: throttle gc to every 10th iteration to reduce GC overhead.
    if (!is.null(checkpoint_dir) && i %% 10L == 0L) {
      gc(verbose = FALSE)
    }
  }

  if (is.null(fit)) {
    fit <- NULL
    fit$sel_mat <- sel_mat
    fit$fit <- lapply(out, `[[`, "fit")
    # T6 — stash approach + checkpoint_dir so the outer entry point can pass
    # them through to new_rml_fit() (predict reuse needs `approach` for the
    # serializer dispatch in get_fit_i()).
    fit$approach <- class_base
    fit$checkpoint_dir <- checkpoint_dir
    class(fit) <- "rml_fit"
  }

  if (!is.null(base)) {
    # Point reconciled forecasts
    bts <- do.call(cbind, lapply(out, `[[`, "bts"))
    rm(out)
    attr(bts, "fit") <- fit
    return(bts)
  } else {
    rm(out)
    return(fit)
  }
}

.rml <- function(approach, ...) {
  UseMethod("rml", approach)
}

#' Predict from a reconciled-ML fit object
#'
#' Dispatches back to the framework-specific reconciliation function
#' ([csrml], [terml], or [ctrml]) using the stored fit so callers
#' have a single uniform predict interface.
#'
#' @param object an object of class \code{rml_fit} returned by
#'   [csrml_fit], [terml_fit], or [ctrml_fit].
#' @param newdata base forecasts matrix/vector (the \code{base} argument
#'   to the corresponding reconciliation function).
#' @param ... additional arguments forwarded to the framework-specific
#'   reconciliation call (e.g., \code{sntz}, \code{round}).
#'
#' @return The reconciled forecast matrix returned by the corresponding
#'   framework function.
#'
#' @export
#' @method predict rml_fit
predict.rml_fit <- function(object, newdata, ...) {
  if (missing(newdata) || is.null(newdata)) {
    cli_abort(
      "{.arg newdata} (base forecasts) is required for {.fn predict.rml_fit}.",
      call = NULL
    )
  }
  switch(
    object$framework,
    "cs" = csrml(base = newdata, fit = object, agg_mat = object$agg_mat, ...),
    "te" = terml(
      base = newdata, fit = object,
      agg_order = object$agg_order, tew = object$tew, ...
    ),
    "ct" = ctrml(
      base = newdata, fit = object,
      agg_mat = object$agg_mat, agg_order = object$agg_order,
      tew = object$tew, ...
    ),
    cli_abort(
      "Unknown framework {.val {object$framework}} in {.cls rml_fit}.",
      call = NULL
    )
  )
}

rml.mlr3 <- function(
  y = NULL,
  X = NULL,
  Xtest = NULL,
  fit = NULL,
  params = NULL,
  tuning = NULL,
  block_sampling = NULL,
  ...
) {
  if (is.null(fit)) {
    if (is.null(y) && is.null(X)) {
      cli_abort(
        c(
          "Mandatory arguments:",
          "1. {.arg y} and {.arg X};",
          "2. {.arg fit} and {.arg Xtest}."
        ),
        call = NULL
      )
    }

    params$.key <- ifelse(is.null(params$.key), "regr.ranger", params$.key)

    if (!is.null(block_sampling) && !is.null(tuning)) {
      tsk_i <- data.frame(
        y = y, X,
        id = rep(seq_len(NROW(X)), each = block_sampling),
        check.names = FALSE
      )
      tsk_i <- mlr3::as_task_regr(tsk_i, target = "y")
      tsk_i$col_roles$group <- "id"
      tsk_i$col_roles$feature <- setdiff(tsk_i$col_roles$feature, "id")
    } else {
      tsk_i <- data.frame(y = y, X, check.names = FALSE)
      tsk_i <- mlr3::as_task_regr(tsk_i, target = "y")
    }

    fit <- do.call(lrn, params)
    if (!is.null(tuning)) {
      if (is.null(tuning$tuner)) {
        tuning$tuner <- mlr3tuning::tnr("random_search", batch_size = 2)
      }
      if (is.null(tuning$resampling)) {
        tuning$resampling <- mlr3::rsmp("cv", folds = 5)
      }
      if (is.null(tuning$store_benchmark_result)) {
        tuning$store_benchmark_result <- FALSE
      }
      if (is.null(tuning$store_models)) {
        tuning$store_models <- FALSE
      }
      if (is.null(tuning$check_values)) {
        tuning$check_values <- FALSE
      }

      if (!is.null(block_sampling)) {
        tuning$resampling$instantiate(tsk_i)
      }

      fit <- mlr3tuning::auto_tuner(
        tuner = tuning$tuner,
        #task = tsk_i,
        learner = fit,
        resampling = tuning$resampling,
        measure = tuning$measure,
        term_evals = tuning$term_evals,
        term_time = tuning$term_time,
        terminator = tuning$terminator,
        search_space = tuning$search_space,
        store_benchmark_result = tuning$store_benchmark_result,
        store_models = tuning$store_models,
        check_values = tuning$check_values,
        callbacks = tuning$callbacks,
        rush = tuning$rush
      )
    }

    fit$train(tsk_i)
  }

  bts <- NULL
  if (!is.null(Xtest)) {
    bts <- fit$predict_newdata(data.frame(Xtest, check.names = FALSE))$response
  }

  if (is.null(bts) && is.null(fit)) {
    cli_abort(
      c(
        "Mandatory arguments:",
        "1. {.arg y} and {.arg X};",
        "2. {.arg fit} and {.arg Xtest}."
      ),
      call = NULL
    )
  }
  return(
    list(bts = bts, fit = fit)
  )
}

rml.randomForest <- function(
  y = NULL,
  X = NULL,
  Xtest = NULL,
  fit = NULL,
  params = NULL,
  ...
) {
  if (is.null(fit)) {
    # T5: soft-deprecate randomForest; ranger is the new default.
    lifecycle::deprecate_soft(
      when = "2.0.0",
      what = I("`approach = \"randomForest\"`"),
      details = paste0(
        "Use `approach = \"ranger\"` instead; ranger is faster and ",
        "statistically equivalent."
      )
    )
    if (is.null(y) && is.null(X)) {
      cli_abort(
        c(
          "Mandatory arguments:",
          "1. {.arg y} and {.arg X};",
          "2. {.arg fit} and {.arg Xtest}."
        ),
        call = NULL
      )
    }

    mtry <- ifelse(
      is.null(params$mtry),
      max(floor(ncol(X) / 3), 1),
      params$mtry
    )
    nodesize <- ifelse(is.null(params$nodesize), 5, params$nodesize)
    ntree <- ifelse(is.null(params$ntree), 500, params$ntree)

    fit <- randomForest(
      y = y,
      x = X,
      mtry = mtry,
      nodesize = nodesize,
      ntree = ntree,
      importance = FALSE
    )
  }

  bts <- NULL
  if (!is.null(Xtest)) {
    bts <- predict(fit, Xtest)
  }

  if (is.null(bts) && is.null(fit)) {
    cli_abort(
      c(
        "Mandatory arguments:",
        "1. {.arg y} and {.arg X};",
        "2. {.arg fit} and {.arg Xtest}."
      ),
      call = NULL
    )
  }
  return(
    list(bts = bts, fit = fit)
  )
}

# T6: catboost backend. Works with matrices directly (no formula interface).
# thread_count = 1L: same rationale as ranger — per-series models are tiny.
# random_seed is optional; omitted when NULL so catboost uses its own default.
# Checkpoint format: native .cbm (catboost binary model), not qs2.
rml.catboost <- function(
  y = NULL,
  X = NULL,
  Xtest = NULL,
  fit = NULL,
  params = NULL,
  ...
) {
  if (is.null(fit)) {
    if (is.null(y) && is.null(X)) {
      cli_abort(
        c(
          "Mandatory arguments:",
          "1. {.arg y} and {.arg X};",
          "2. {.arg fit} and {.arg Xtest}."
        ),
        call = NULL
      )
    }
    if (!requireNamespace("catboost", quietly = TRUE)) {
      cli_abort(
        paste0(
          "Package {.pkg catboost} required for {.code approach = \"catboost\"}. ",
          "See https://catboost.ai/en/docs/installation/r-package-install"
        ),
        call = NULL
      )
    }
    iterations <- if (is.null(params$iterations)) 500L else as.integer(params$iterations)
    depth      <- if (is.null(params$depth))      6L   else as.integer(params$depth)
    seed_i     <- params$random_seed

    pool <- catboost::catboost.load_pool(data = as.matrix(X), label = y)
    cb_params <- list(
      loss_function = "RMSE",
      iterations    = iterations,
      depth         = depth,
      thread_count  = 1L,
      logging_level = "Silent"
    )
    if (!is.null(seed_i)) cb_params$random_seed <- as.integer(seed_i)
    fit <- catboost::catboost.train(pool, NULL, params = cb_params)
  }

  bts <- NULL
  if (!is.null(Xtest)) {
    test_pool <- catboost::catboost.load_pool(data = as.matrix(Xtest))
    bts <- catboost::catboost.predict(fit, test_pool)
  }

  return(list(bts = bts, fit = fit))
}

# T5: ranger backend. Mirrors rml.randomForest contract but uses ranger's
# data.frame interface (no matrix-mode equivalent that handles factor +
# numeric mixes cleanly). num.threads = 1L: per-series trees are tiny and
# threading overhead exceeds any benefit at this granularity.
rml.ranger <- function(
  y = NULL,
  X = NULL,
  Xtest = NULL,
  fit = NULL,
  params = NULL,
  ...
) {
  if (is.null(fit)) {
    if (is.null(y) && is.null(X)) {
      cli_abort(
        c(
          "Mandatory arguments:",
          "1. {.arg y} and {.arg X};",
          "2. {.arg fit} and {.arg Xtest}."
        ),
        call = NULL
      )
    }

    if (!requireNamespace("ranger", quietly = TRUE)) {
      cli_abort(
        paste0(
          "Package {.pkg ranger} is required for {.code approach = \"ranger\"}. ",
          "Install it via {.code install.packages(\"ranger\")}."
        ),
        call = NULL
      )
    }

    mtry <- if (is.null(params$mtry)) max(floor(ncol(X) / 3), 1) else params$mtry
    min.node.size <- if (is.null(params$min.node.size)) 5 else params$min.node.size
    num.trees <- if (is.null(params$num.trees)) 500 else params$num.trees
    seed_i <- if (is.null(params$seed)) NULL else params$seed

    # ranger requires a data.frame; column names must be syntactically valid.
    df <- data.frame(as.matrix(X), check.names = TRUE)
    df$.y <- y

    fit <- ranger::ranger(
      formula = .y ~ .,
      data = df,
      num.trees = num.trees,
      mtry = mtry,
      min.node.size = min.node.size,
      num.threads = 1L,
      seed = seed_i,
      importance = "none",
      verbose = FALSE
    )
  }

  bts <- NULL
  if (!is.null(Xtest)) {
    df_test <- data.frame(as.matrix(Xtest), check.names = TRUE)
    bts <- predict(fit, data = df_test, num.threads = 1L)$predictions
  }

  if (is.null(bts) && is.null(fit)) {
    cli_abort(
      c(
        "Mandatory arguments:",
        "1. {.arg y} and {.arg X};",
        "2. {.arg fit} and {.arg Xtest}."
      ),
      call = NULL
    )
  }
  return(
    list(bts = bts, fit = fit)
  )
}

rml.xgboost <- function(
  y = NULL,
  X = NULL,
  Xtest = NULL,
  fit = NULL,
  params = NULL,
  ...
) {
  if (is.null(fit)) {
    if (is.null(y) && is.null(X)) {
      cli_abort(
        c(
          "Mandatory arguments:",
          "1. {.arg y} and {.arg X};",
          "2. {.arg fit} and {.arg Xtest}."
        ),
        call = NULL
      )
    }

    if (is.null(params)) {
      params <- list(
        eta = 0.3,
        colsample_bytree = 1,
        min_child_weight = 1,
        max_depth = 6,
        gamma = 0,
        subsample = 1,
        objective = "reg:squarederror"
      )
      nrounds = 100
    } else {
      nrounds <- ifelse(is.null(params$nrounds), 100, params$nrounds)
    }

    train <- xgb.DMatrix(data = X, label = y)
    fit <- xgb.train(
      data = train,
      nrounds = nrounds,
      params = params,
      verbose = 0
    )
  }

  bts <- NULL
  if (!is.null(Xtest)) {
    test <- xgb.DMatrix(data = Xtest)
    bts <- predict(fit, test)
  }

  if (is.null(bts) && is.null(fit)) {
    cli_abort(
      c(
        "Mandatory arguments:",
        "1. {.arg y} and {.arg X};",
        "2. {.arg fit} and {.arg Xtest}."
      ),
      call = NULL
    )
  }

  return(
    list(bts = bts, fit = fit)
  )
}

rml.lightgbm <- function(
  y = NULL,
  X = NULL,
  Xtest = NULL,
  fit = NULL,
  params = NULL,
  ...
) {
  if (is.null(fit)) {
    if (is.null(y) && is.null(X)) {
      cli_abort(
        c(
          "Mandatory arguments:",
          "1. {.arg y} and {.arg X};",
          "2. {.arg fit} and {.arg Xtest}."
        ),
        call = NULL
      )
    }

    if (is.null(params)) {
      params <- list(
        eta = 0.1,
        num_leaves = 31,
        subsample = 1,
        colsample_bytree = 1,
        min_child_weight = 1e-3,
        max_depth = -1,
        lambda_l1 = 0,
        objective = "regression"
      )
      nrounds = 100
    } else {
      nrounds <- ifelse(is.null(params$nrounds), 100, params$nrounds)
    }
    X <- as.matrix(X)
    train <- lgb.Dataset(data = X, label = y)
    fit <- lgb.train(
      data = train,
      params = params,
      nrounds = nrounds,
      verbose = -1
    )
  }

  bts <- NULL
  if (!is.null(Xtest)) {
    Xtest <- as.matrix(Xtest)
    bts <- predict(fit, Xtest)
  }

  if (is.null(bts) && is.null(fit)) {
    cli_abort(
      c(
        "Mandatory arguments:",
        "1. {.arg y} and {.arg X};",
        "2. {.arg fit} and {.arg Xtest}."
      ),
      call = NULL
    )
  }
  return(
    list(bts = bts, fit = fit)
  )
}

# rml_fit_rf <- function(y = NULL, X = NULL, Xtest = NULL, fit = NULL,
#                        params = NULL, ...){
#   if(is.null(fit)){
#     if(is.null(y) && is.null(X)){
#       cli_abort(c("Mandatory arguments:",
#                   "1. {.arg y} and {.arg X};",
#                   "2. {.arg fit} and {.arg Xtest}."),
#                 call = NULL)
#     }
#
#     mtry <- ifelse(is.null(params$mtry), max(floor(ncol(X)/3), 1), params$mtry)
#     nodesize <- ifelse(is.null(params$nodesize), 5, params$nodesize)
#     ntree <- ifelse(is.null(params$ntree), 500, params$ntree)
#
#     fit <- randomForest(y = y,
#                         x = X,
#                         mtry = mtry,
#                         nodesize = nodesize,
#                         ntree = ntree,
#                         importance = FALSE)
#   }
#
#   bts <- NULL
#   if(!is.null(Xtest)){
#     bts <- as.vector(predict(fit, Xtest))
#   }
#
#   if(is.null(bts) && is.null(fit)){
#     cli_abort(c("Mandatory arguments:",
#                 "1. {.arg y} and {.arg X};",
#                 "2. {.arg fit} and {.arg Xtest}."),
#               call = NULL)
#   }
#   return(
#     list(bts = bts, fit = fit)
#   )
# }
#
# rml_fit_xgboost <- function(y = NULL, X = NULL, Xtest = NULL, fit = NULL,
#                             params = NULL, ...){
#   if(is.null(fit)){
#     if(is.null(y) && is.null(X)){
#       cli_abort(c("Mandatory arguments:",
#                   "1. {.arg y} and {.arg X};",
#                   "2. {.arg fit} and {.arg Xtest}."),
#                 call = NULL)
#     }
#
#     if(is.null(params)){
#       params <- list(
#         eta = 0.3,
#         colsample_bytree = 1,
#         min_child_weight = 1,
#         max_depth = 6,
#         gamma = 0,
#         subsample = 1,
#         objective = "reg:squarederror"
#       )
#       nrounds = 100
#     }else{
#       nrounds <- ifelse(is.null(params$nrounds), 100, params$nrounds)
#     }
#
#     train <- xgb.DMatrix(data = as.matrix(X), label = y)
#     fit <- xgb.train(data = train, nrounds = nrounds,
#                      params = params, verbose = 0)
#   }
#
#   bts <- NULL
#   if(!is.null(Xtest)){
#     test <- xgb.DMatrix(data = as.matrix(Xtest))
#     bts <- as.vector(predict(fit, test))
#   }
#
#   if(is.null(bts) && is.null(fit)){
#     cli_abort(c("Mandatory arguments:",
#                 "1. {.arg y} and {.arg X};",
#                 "2. {.arg fit} and {.arg Xtest}."),
#               call = NULL)
#   }
#
#   return(
#     list(bts = bts, fit = fit)
#   )
# }
#
# rml_fit_lightgbm <- function(y = NULL, X = NULL, Xtest = NULL, fit = NULL,
#                              params = NULL, ...){
#   if(is.null(fit)){
#     if(is.null(y) && is.null(X)){
#       cli_abort(c("Mandatory arguments:",
#                   "1. {.arg y} and {.arg X};",
#                   "2. {.arg fit} and {.arg Xtest}."),
#                 call = NULL)
#     }
#
#     if(is.null(params)){
#       params <- list(
#         eta = 0.1,
#         num_leaves = 31,
#         subsample = 1,
#         colsample_bytree = 1,
#         min_child_weight = 1e-3,
#         max_depth = -1,
#         lambda_l1 = 0,
#         objective = "regression"
#       )
#       nrounds = 100
#     }else{
#       nrounds <- ifelse(is.null(params$nrounds), 100, params$nrounds)
#     }
#     train <- lgb.Dataset(data = as.matrix(X), label = y)
#     fit <- lgb.train(data = train, params = params, nrounds = nrounds, verbose = -1)
#   }
#
#   bts <- NULL
#   if(!is.null(Xtest)){
#     bts <- as.vector(predict(fit, as.matrix(Xtest)))
#   }
#
#   if(is.null(bts) && is.null(fit)){
#     cli_abort(c("Mandatory arguments:",
#                 "1. {.arg y} and {.arg X};",
#                 "2. {.arg fit} and {.arg Xtest}."),
#               call = NULL)
#   }
#   return(
#     list(bts = bts, fit = fit)
#   )
# }
#
# rml_fit_mlr3 <- function(y = NULL, X = NULL, Xtest = NULL, fit = NULL,
#                          params = NULL, tuning = NULL, block_sampling = NULL,
#                          ...){
#   require("mlr3learners")
#   require("mlr3")
#   if(is.null(fit)){
#     if(is.null(y) && is.null(X)){
#       cli_abort(c("Mandatory arguments:",
#                   "1. {.arg y} and {.arg X};",
#                   "2. {.arg fit} and {.arg Xtest}."),
#                 call = NULL)
#     }
#
#     params$.key <- ifelse(is.null(params$.key), "regr.ranger", params$.key)
#     tsk_i <- as_task_regr(cbind(y = y, X), target = "y")
#     fit <- do.call(lrn, params)
#     if(!is.null(tuning)){
#       if(is.null(tuning$tuner)){
#         tuning$tuner <- tnr("random_search", batch_size = 2)
#       }
#       if(is.null(tuning$resampling)){
#         tuning$resampling <- rsmp("cv", folds = 5)
#       }
#       if(is.null(tuning$store_benchmark_result)){
#         tuning$store_benchmark_result <- TRUE
#       }
#       if(is.null(tuning$store_models)){
#         tuning$store_models <- FALSE
#       }
#       if(is.null(tuning$check_values)){
#         tuning$check_values <- FALSE
#       }
#
#       if(!is.null(block_sampling)){
#         tsk_i <- as_task_regr(cbind(y = y, X, id = rep(1:NROW(X), each = block_sampling)),
#                               target = "y")
#         tsk_i$col_roles$group <- "id"
#         tsk_i$col_roles$feature <- setdiff(tsk_i$col_roles$feature, "id")
#         tuning$resampling$instantiate(tsk_i)
#       }
#
#       fit <- auto_tuner(
#         tuner = tuning$tuner,
#         #task = tsk_i,
#         learner = fit,
#         resampling = tuning$resampling,
#         measure = tuning$measure,
#         term_evals = tuning$term_evals,
#         term_time = tuning$term_time,
#         terminator = tuning$terminator,
#         search_space = tuning$search_space,
#         store_benchmark_result = tuning$store_benchmark_result,
#         store_models = tuning$store_models,
#         check_values = tuning$check_values,
#         callbacks = tuning$callbacks,
#         rush = tuning$rush
#       )
#       #fit$param_set$values = instance$result_learner_param_vals
#     }
#
#     fit$train(tsk_i)
#   }
#
#   bts <- NULL
#   if(!is.null(Xtest)){
#     bts <- fit$predict_newdata(Xtest)$response
#   }
#
#   if(is.null(bts) && is.null(fit)){
#     cli_abort(c("Mandatory arguments:",
#                 "1. {.arg y} and {.arg X};",
#                 "2. {.arg fit} and {.arg Xtest}."),
#               call = NULL)
#   }
#   return(
#     list(bts = bts, fit = fit)
#   )
# }
