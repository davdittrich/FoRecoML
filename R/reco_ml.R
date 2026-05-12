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
    hat <- unname(hat)
    obs <- unname(obs)
    p <- NCOL(obs)

    # if (!is.null(seed)) {
    #   set.seed(seed)
    # }
  } else {
    # sel_mat <- fit$sel_mat
    p <- length(fit$fit)
  }

  if (!is.null(base)) {
    base <- unname(base)
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

  out <- lapply(1:p, function(i) {
    global_id <- if (length(sel_mat) == 1) {
      seq_len(active_ncol)
    } else if (is(sel_mat, "sparseVector") || NCOL(sel_mat) == 1) {
      which(as.numeric(if (is(sel_mat, "sparseVector")) sel_mat else sel_mat[, 1]) != 0)
    } else {
      which(sel_mat[, i] != 0)
    }
    id <- if (is.null(col_map)) global_id else { x <- col_map[global_id]; x[!is.na(x)] }

    if (is.null(fit)) {
      y <- obs[, i]
      X <- hat[, id, drop = FALSE]
      fit_i <- NULL

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
    } else {
      y <- X <- NULL
      fit_i <- fit$fit[[i]]
    }

    if (!is.null(base)) {
      Xtest <- base[, id, drop = FALSE]
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
    return(tmp)
  })

  ml_step <- do.call("rbind", out)
  if (is.null(fit)) {
    fit <- NULL
    fit$sel_mat <- sel_mat
    fit$fit <- do.call("list", ml_step[, "fit"])
    class(fit) <- "rml_fit"
  }

  if (!is.null(base)) {
    # Point reconciled forecasts
    bts <- do.call("cbind", ml_step[, "bts"])
    attr(bts, "fit") <- fit
    return(bts)
  } else {
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
    tsk_i <- data.frame(y = y, X, check.names = FALSE)
    tsk_i <- mlr3::as_task_regr(tsk_i, target = "y")
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
        rownames(X) <- NULL
        tsk_i <- data.frame(y = y, X, id = rep(1:NROW(X), each = block_sampling), check.names = FALSE)
        tsk_i <- mlr3::as_task_regr(tsk_i, target = "y")
        # tsk_i$encapsulate("evaluate", fallback = lrn("regr.featureless"))
        tsk_i$col_roles$group <- "id"
        tsk_i$col_roles$feature <- setdiff(tsk_i$col_roles$feature, "id")
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
    bts <- as.vector(predict(fit, Xtest))
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
    bts <- as.vector(predict(fit, test))
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
    bts <- as.vector(predict(fit, Xtest))
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
