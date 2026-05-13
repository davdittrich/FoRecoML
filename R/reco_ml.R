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
  checkpoint = "auto",
  n_workers = "auto",
  kset = NULL,
  h = NULL,
  n = NULL,
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

    # if (!is.null(seed)) {
    #   set.seed(seed)
    # }
  } else {
    # sel_mat <- fit$sel_mat
    p <- length(fit$fit)
  }

  # Resolve parallelism: cap inner threads when outer workers > 1.
  n_workers_resolved <- resolve_n_workers(n_workers, class_base, params)
  if (n_workers_resolved > 1L) {
    params <- cap_inner_threads(params, n_workers_resolved, approach = class_base)
  }

  # spd.10 — predict-reuse safety: in-memory booster list copied to N daemons = OOM.
  if (n_workers_resolved > 1L &&
      !is.null(fit) &&
      !is.null(fit$fit) &&
      length(fit$fit) > 0L &&
      !is.character(fit$fit[[1]])) {
    cli::cli_inform(c(
      "!" = "{.arg fit} holds {length(fit$fit)} in-memory model(s); copying to {n_workers_resolved} workers would risk OOM.",
      "i" = "Auto-capping {.arg n_workers} = 1. Train with {.code checkpoint = TRUE} (or a path) for parallel predict-reuse."
    ))
    n_workers_resolved <- 1L
  }

  # Capture dots BEFORE the loop so they can be forwarded via .args.
  dots <- list(...)

  # T6 — Resolve tri-mode `checkpoint` into NULL (disabled) or a directory.
  # Only meaningful during training (fit=NULL). On fit reuse, the stored fit
  # already encodes whether models are paths or in-memory; we never re-decide.
  checkpoint_dir <- if (is.null(fit)) {
    resolve_checkpoint(checkpoint, hat, class_base, p, n_workers = n_workers_resolved)
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
  if (is.null(keep_cols)) {
    active_ncol <- if (!is.null(hat)) NCOL(hat) else NCOL(base)
    keep_cols <- sel_mat_keep_cols(sel_mat, active_ncol)
    slice <- length(keep_cols) < active_ncol
    if (slice) {
      col_map <- rep(NA_integer_, active_ncol)
      col_map[keep_cols] <- seq_along(keep_cols)
      if (!is.null(hat))  hat  <- hat[,  keep_cols, drop = FALSE]
      if (!is.null(base)) base <- base[, keep_cols, drop = FALSE]
    } else {
      col_map <- NULL
    }
  } else {
    # T5: active_ncol = full feature count from sel_mat.
    # spd.12: when kset is non-NULL, hat/base are raw (not pre-expanded); derive
    # active_ncol from sel_mat or keep_cols (not from NCOL(hat/base)).
    active_ncol <- if (length(sel_mat) == 1) {
      max(keep_cols)  # total_cols, works for raw and pre-expanded hat/base
    } else if (is(sel_mat, "sparseVector")) {
      length(sel_mat)
    } else {
      NROW(sel_mat)
    }
    col_map <- rep(NA_integer_, active_ncol)
    col_map[keep_cols] <- seq_along(keep_cols)
  }

  # Per-series loop body with all closure-captured objects as explicit formals.
  # This signature must stay in sync with the 16-item closure list spec.
  loop_body <- function(i, hat, obs, base, sel_mat, col_map,
                        class_base, approach, active_ncol,
                        params, fit, checkpoint_dir, kset, dots, h, n) {
    gc_every <- 5L

    global_id <- if (length(sel_mat) == 1) {
      seq_len(active_ncol)
    } else if (is(sel_mat, "sparseVector") || NCOL(sel_mat) == 1) {
      which(as.numeric(if (is(sel_mat, "sparseVector")) sel_mat else sel_mat[, 1]) != 0)
    } else {
      which(sel_mat[, i] != 0)
    }
    id <- if (is.null(col_map)) global_id else { x <- col_map[global_id]; x[!is.na(x)] }
    global_id_post_na <- global_id

    na_mask <- NULL  # per-series NA column mask; persisted into fit for predict-reuse

    if (is.null(fit)) {
      y <- obs[, i]
      if (is.null(kset)) {
        X <- hat[, id, drop = FALSE]
      } else {
        # Per-iter expansion: 3-way dispatch on (kset, h).
        # kset only (non-mfh, spd.12): input2rtw_partial.
        # kset + h + n (mfh, spd.13): mat2hmat_partial.
        # For mfh: h_hat_eff and h_base_eff are derived from NCOL(hat/base)
        # and kt (= sum(max(kset)/kset)), not from the nominal h parameter,
        # because training and prediction horizons can differ.
        # Future opt: pre-compute parts once in rml() (spd.14 TBD).
        if (!is.null(h)) {
          kt_eff     <- sum(max(kset) / kset)
          h_hat_eff  <- NCOL(hat) / kt_eff
          h_base_eff <- if (!is.null(base)) NCOL(base) / kt_eff else NULL
        }
        X <- if (is.null(h)) {
          FoRecoML:::input2rtw_partial(hat, kset, cols = global_id)
        } else {
          FoRecoML:::mat2hmat_partial(hat, h_hat_eff, kset, n, cols = global_id)
        }
        # NA filter — applies uniformly to BOTH spd.12 and spd.13 expansion outputs.
        na_cols <- FoRecoML:::na_col_mask(X)
        if (any(na_cols)) {
          X <- X[, !na_cols, drop = FALSE]
          global_id_post_na <- global_id[!na_cols]
          na_mask <- na_cols  # persist: which expanded cols were NA-dropped for this series
        }
      }
      fit_i <- NULL

      if (anyNA(X)) {
        X <- stats::na.omit(X)
        if (length(attr(X, "na.action")) > 0L) {
          if (NROW(X) == 0L) {
            cli::cli_abort(
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
      fit_i <- FoRecoML:::get_fit_i(fit, i)
      # Replay train-time NA column mask for kset path (CRITICAL: spd.12 fix-up).
      # Without this, global_id_post_na == global_id (wider than fit was trained on)
      # → dim-mismatch or silent wrong prediction. Back-compat: NULL means no NA mask.
      if (!is.null(kset) && !is.null(fit$na_cols_list)) {
        stored_na <- fit$na_cols_list[[i]]
        if (!is.null(stored_na) && any(stored_na)) {
          global_id_post_na <- global_id[!stored_na]
        }
      }
    }

    if (!is.null(base)) {
      if (is.null(kset)) {
        Xtest <- base[, id, drop = FALSE]
      } else if (is.null(h)) {
        Xtest <- FoRecoML:::input2rtw_partial(base, kset, cols = global_id_post_na)
      } else {
        # mfh Xtest: always derive h_base from base dimensions.
        # kt = sum(max(kset)/kset); predict horizon can differ from training horizon.
        h_base_kt <- sum(max(kset) / kset)
        h_base_eff_xtest <- NCOL(base) / h_base_kt
        Xtest <- FoRecoML:::mat2hmat_partial(base, h_base_eff_xtest, kset, n, cols = global_id_post_na)
      }
    } else {
      Xtest <- NULL
    }

    tmp <- do.call(FoRecoML:::.rml, c(
      list(approach = approach, y = y, X = X, Xtest = Xtest,
           fit = fit_i, params = params),
      dots
    ))

    # T6 — Serialize freshly-trained fit to disk, replace with path.
    if (!is.null(checkpoint_dir) && is.null(fit)) {
      path_i <- FoRecoML:::serialize_fit(tmp$fit, checkpoint_dir, i, class_base)
      tmp$fit <- path_i
    }

    # T6 + spd.8 — gc() periodically under checkpoint mode OR on predict-reuse
    # (where deserialized/in-memory boosters + lightgbm C++ scratch accumulate
    # without explicit cleanup).
    if ((!is.null(checkpoint_dir) || !is.null(fit)) && i %% gc_every == 0L) {
      gc(verbose = FALSE)
    }

    # mw3.3 invariant: on predict-reuse, store only bts. na_mask is NULL on
    # predict-reuse branch (ignored by the outer collector lapply).
    if (is.null(fit)) list(bts = tmp$bts, fit = tmp$fit, na_mask = na_mask) else list(bts = tmp$bts)
  }

  # Dispatch: sequential (n_workers == 1) or parallel via mirai.
  out <- if (n_workers_resolved == 1L) {
    # T6 — sequential path: for-loop so we can gc() after each checkpoint.
    result <- vector("list", p)
    for (i in seq_len(p)) {
      result[[i]] <- loop_body(
        i, hat = hat, obs = obs, base = base, sel_mat = sel_mat,
        col_map = col_map, class_base = class_base, approach = approach,
        active_ncol = active_ncol, params = params, fit = fit,
        checkpoint_dir = checkpoint_dir, kset = kset, dots = dots,
        h = h, n = n
      )
    }
    result
  } else {
    # Parallel path via mirai (spawn-based, avoids fork+OpenMP issues).
    prev <- mirai::status()$connections
    if (prev == 0L) {
      mirai_seed <- sample.int(.Machine$integer.max, 1L)
      mirai::daemons(n_workers_resolved, seed = mirai_seed)
      on.exit(mirai::daemons(0), add = TRUE)
    }
    mirai::everywhere({ library(FoRecoML) })
    mirai::mirai_map(
      seq_len(p), loop_body,
      .args = list(
        hat = hat, obs = obs, base = base, sel_mat = sel_mat,
        col_map = col_map, class_base = class_base, approach = approach,
        active_ncol = active_ncol, params = params, fit = fit,
        checkpoint_dir = checkpoint_dir, kset = kset, dots = dots,
        h = h, n = n
      )
    )[]
  }

  if (is.null(fit)) {
    fit <- NULL
    fit$sel_mat <- sel_mat
    fit$fit <- lapply(out, `[[`, "fit")
    # spd.12 fix-up: persist per-series NA column masks so predict-reuse on the
    # kset path can reconstruct the correct global_id_post_na for each series.
    # NULL entries (no NA columns) are preserved; the list is always length == p.
    fit$na_cols_list <- lapply(out, `[[`, "na_mask")
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
