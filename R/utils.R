#' Extract the reconciled model from a reconciliation result
#'
#' @description
#' Safely extracts the fitted reconciled model(s) from a reconciliation
#' function's output (e.g., [csrml], [terml] and [ctrml]).
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
#' \dontrun{
#' # Suppose `reco` is the result of a reconciliation call:
#' # reco <- ctrml(...)
#'
#' mdl <- extract_reconciled_ml(reco)
#' if(!is.null(mdl)){
#'   print(mdl)
#' }
#'
#' # If already an rml_fit:
#' mdl2 <- extract_reconciled_ml(mdl)
#' }
#' @export
extract_reconciled_ml <- function(reco){
  if(inherits(reco, "rml_fit")) {
    cli_inform("Input {.arg reco} is already an {.cls rml_fit}; returning it unchanged.")
    return(reco)
  }

  info <- tryCatch(
    suppressWarnings(recoinfo(reco, verbose = FALSE)),
    error = function(e) {
      cli_warn("Failed to retrieve reconciliation info: {conditionMessage(e)}", call = NULL)
      return(NULL)
    }
  )

  if (is.null(info) || is.null(info$fit)) {
    cli::cli_warn("No reconciled model information available.", call = NULL)
    return(invisible(NULL))
  }

  return(info$fit)
}

print.rml_fit <- function(x, ...){
  cat("----- Reconciled models -----\n")
  cat("Features:", attr(x$sel_mat, "sel_method"), "\n")
  cat("Approach:", x$approach, "\n")
  cat("  Models:", length(x$fit), "\n")
}

# Rombouts et al. (2025) matrix-form
input2rtw <- function(x, kset){
  x <- FoReco::FoReco2matrix(x, kset)
  x <- lapply(1:length(kset), function(i){
    if(NCOL(x[[i]])>1){
      tmp <- apply(x[[i]], 2, rep, each = kset[i])
      #colnames(tmp) <- paste0(colnames(tmp), "_", kset[i])
    }else{
      tmp <- rep(x[[i]], each = kset[i])
    }
    tmp
  })
  do.call(cbind, rev(x))
}
