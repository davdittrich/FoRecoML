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

print.rml_fit <- function(x, ...) {
  cat("----- Reconciled models -----\n")
  cat("Features:", attr(x$sel_mat, "sel_method"), "\n")
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
