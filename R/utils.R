#' Extract the reconciled model from a reconciliation result
#'
#' @description
#' Extract the fitted reconciled model(s) from a reconciliation
#' function's output (e.g., [csrml], [terml] and [ctrml]).
#' The model can be reused for forecast reconciliation in the
#' reconciliation functions.
#'
#' @param reco An object returned by a reconciliation function
#' (e.g., the result of [csrml], [terml] and [ctrml]).
#'
#' @return A named list with reconciliation information:
#'   \item{\code{sel_mat}}{Features used (e.g., the selected feature
#'   matrix or indices).}
#'   \item{\code{fit}}{List of reconciled models.}
#'   \item{\code{approach}}{The learning approach used (e.g., \code{"xgboost"},
#'   \code{"lightgbm"}, \code{"randomForest"}, \code{"mlr3"}).}
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
#' # `reco` is the result of a reconciliation call:
#' reco <- csrml(base = base, hat = hat, obs = obs, agg_mat = agg_mat)
#'
#' mdl <- extract_reconciled_ml(reco)
#' mdl
#'
#' @export
extract_reconciled_ml <- function(reco) {
  if (inherits(reco, "rml_fit")) {
    cli_inform(
      "Input {.arg reco} is already an {.cls rml_fit}; returning it unchanged."
    )
    return(reco)
  }

  info <- tryCatch(
    suppressWarnings(recoinfo(reco, verbose = FALSE)),
    error = function(e) {
      cli_warn(
        "Failed to retrieve reconciliation info: {conditionMessage(e)}",
        call = NULL
      )
      return(NULL)
    }
  )

  if (is.null(info) || is.null(info$fit)) {
    cli::cli_warn("No reconciled model information available.", call = NULL)
    return(invisible(NULL))
  }

  return(info$fit)
}

#' @export
#' @method print rml_fit
print.rml_fit <- function(x, ...) {
  cat("----- Reconciled models -----\n")
  cat("Framework:", x$framework, "\n")
  cat("Features:", x$features, "\n")
  cat("Approach:", x$approach, "\n")
  cat("  Models:", length(x$fit), "\n")
}

# Rombouts et al. (2025) matrix-form
input2rtw <- function(x, kset) {
  x <- FoReco::FoReco2matrix(x, kset)
  x <- lapply(1:length(kset), function(i) {
    if (NCOL(x[[i]]) > 1) {
      tmp <- apply(x[[i]], 2, rep, each = kset[i])
      #colnames(tmp) <- paste0(colnames(tmp), "_", kset[i])
    } else {
      tmp <- rep(x[[i]], each = kset[i])
    }
    tmp
  })
  do.call(cbind, rev(x))
}

# Slice-first variant of input2rtw: materializes ONLY the columns whose global
# index (in the full row-replicated output) is in `cols`. Output column order
# matches `cols`. Full-cols invocation is byte-identical to input2rtw().
input2rtw_partial <- function(x, kset, cols) {
  parts <- FoReco::FoReco2matrix(x, kset)
  # do.call(cbind, rev(parts)) ordering => reversed level order
  ncol_per_level_rev <- vapply(rev(parts), NCOL, integer(1))
  col_offsets <- c(0L, cumsum(ncol_per_level_rev)) # length = length(kset)+1
  out_blocks <- lapply(seq_along(ncol_per_level_rev), function(j) {
    lvl_idx_full <- length(kset) - j + 1L # original (non-reversed) level index
    in_range <- cols > col_offsets[j] & cols <= col_offsets[j + 1]
    if (!any(in_range)) {
      return(NULL)
    }
    local <- cols[in_range] - col_offsets[j]
    block <- if (NCOL(parts[[lvl_idx_full]]) > 1) {
      parts[[lvl_idx_full]][, local, drop = FALSE]
    } else {
      parts[[lvl_idx_full]]
    }
    k <- kset[lvl_idx_full]
    expanded <- if (NCOL(block) > 1) {
      apply(block, 2, rep, each = k)
    } else {
      rep(block, each = k)
    }
    list(mat = expanded, global_cols = cols[in_range])
  })
  out_blocks <- Filter(Negate(is.null), out_blocks)
  mat <- do.call(cbind, lapply(out_blocks, `[[`, "mat"))
  global_cols <- unlist(lapply(out_blocks, `[[`, "global_cols"))
  mat[, order(match(global_cols, cols)), drop = FALSE]
}

# Compute keep_cols (global column indices of features actually used) from
# sel_mat. Mirrors the T4 keep_cols logic in rml().
sel_mat_keep_cols <- function(sel_mat, ncol_full) {
  if (length(sel_mat) == 1) {
    return(seq_len(ncol_full))
  }
  if (is(sel_mat, "sparseVector")) {
    return(which(as.numeric(sel_mat) != 0))
  }
  if (NCOL(sel_mat) == 1) {
    return(which(as.numeric(sel_mat[, 1]) != 0))
  }
  which(Matrix::rowSums(sel_mat != 0) > 0)
}

# Reconcile using machine learning models class
#
# This function creates an object of class \code{reconcile_ml} that contains the
# necessary components to perform forecast reconciliation using machine learning
# models.
#
# @param features Character string specifying which features are used for model
#   training.
# @param features_size Optional numeric vector specifying the size of the
#   feature set to be used for model training.
# @param framework Character string specifying the reconciliation framework to
#   be used. Options include "\code{cs}" for cross-sectional, "\code{te}" for
#   temporal, and "\code{ct}" for cross-temporal.
# @param sel_mat Selection matrix/vector to be used to select the features
#   for each variable. It's strickly related to the \code{features} argument.
# @inheritParams ctrml
#
# @returns Returns a fitted object ([reconcile_ml] class) that can be used
#   for reconciliation.
#
# @export
new_rml_fit <- function(
  fit,
  agg_mat = NULL,
  agg_order = NULL,
  tew = NULL,
  sel_mat = NULL,
  approach = NULL,
  framework = NULL,
  features = NULL,
  features_size = NULL,
  block_sampling = NULL
) {
  framework <- match.arg(framework, choices = c("cs", "te", "ct"))
  structure(
    list(
      agg_mat = agg_mat,
      agg_order = agg_order,
      tew = tew,
      fit = fit,
      sel_mat = sel_mat,
      approach = approach,
      framework = framework,
      features = features,
      features_size = features_size,
      block_sampling = block_sampling
    ),
    class = "rml_fit"
  )
}
