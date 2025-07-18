#' Cross-sectional Reconciliation with Machine Learning
#'
#' @param hat todo
#' @param obs todo
#' @param base todo
#' @param agg_mat todo
#' @param params todo
#' @param features todo
#' @param fit todo
#' @param approach todo
#' @param sntz todo
#' @param seed todo
#' @param tuning todo (see [mlr3tuning::auto_tuner], except 'learner')
#'
#' @return Reconciled forecasts or Reconciled models
#'
#' @examples
#' # (2 x 3) base forecasts matrix (simulated), Z = X + Y
#' base <- matrix(rnorm(6, mean = c(20, 10, 10)), 2, byrow = TRUE)
#' # (10 x 3) in-sample residuals matrix (simulated)
#' fitted <- matrix(rnorm(3*100, mean = c(20, 10, 10)), 100, byrow = TRUE)
#' obs <- matrix(rnorm(2*100, mean = c(10, 10)), 100, byrow = TRUE)
#' agg_mat <- t(c(1,1))
#'
#' # randomForest (default)
#' ## Option 1: one step
#' rml <- csrml(hat = fitted, obs = obs, base = base,
#'              agg_mat = agg_mat, seed = 1996) # Reconciled forecasts
#' model_fit <- recoinfo(rml, verbose = FALSE)$fit # Reconciled models
#'
#' ## Option 2: two steps
#' model_fit <- csrml(hat = fitted, obs = obs,
#'                    agg_mat = agg_mat, seed = 1996) # Reconciled models
#' rml <- csrml(base = base, agg_mat = agg_mat, fit = model_fit) # Reconciled forecasts
#'
#' # randomForest with mlr3
#' ## Option 1: one step
#' rml_mlr3 <- csrml(hat = fitted, obs = obs, base = base, agg_mat = agg_mat, seed = 1996,
#'                   approach = "mlr3") # Reconciled forecasts
#' model_fit_mlr3 <- recoinfo(rml, verbose = FALSE)$fit # Reconciled models
#'
#' ## Option 2: two steps
#' model_fit_mlr3 <- csrml(hat = fitted, obs = obs, agg_mat = agg_mat, seed = 1996,
#'                         approach = "mlr3") # Reconciled models
#'
#' rml_refit <- csrml(base = base, agg_mat = agg_mat, fit = model_fit_mlr3) # Reconciled forecasts
#'
#' @export
csrml <- function(hat, obs, base, agg_mat, features = "all", approach = "randomForest",
                  params = NULL, tuning = NULL, fit = NULL, sntz = FALSE, seed = NULL){
  features <- match.arg(features, c("all", "bts", "str", "str-bts"))

  if(missing(agg_mat)){
    cli_abort("Argument {.arg agg_mat} is missing, with no default.", call = NULL)
  }

  tmp <- cstools(agg_mat = agg_mat)
  n <- tmp$dim[["n"]]
  nb <- tmp$dim[["nb"]]
  strc_mat <- tmp$strc_mat
  agg_mat <- tmp$agg_mat
  id_bts <- c(rep(0, n-nb), rep(1, nb))

  if(is.null(fit)){
    if(missing(obs)){
      cli_abort("Argument {.arg obs} is missing, with no default.", call = NULL)
    }else if(NCOL(obs) != nb){
      cli_abort("Incorrect {.arg obs} columns dimension.", call = NULL)
    }

    if(missing(hat)){
      cli_abort("Argument {.arg hat} is missing, with no default.", call = NULL)
    }

    if(missing(base)){
      base <- NULL
    }

    switch(features,
           "bts" = {
             sel_mat <- Matrix(rep(id_bts, nb), ncol = nb, sparse = TRUE)
           },
           "str" = {
             sel_mat <- strc_mat
           },
           "str-bts" = {
             sel_mat <- strc_mat + Matrix(rep(id_bts, nb), ncol = nb, sparse = TRUE)
             sel_mat[sel_mat != 0] <- 1
           },
           "all" = {
             sel_mat <- Matrix(1, nrow = n, ncol = nb, sparse = TRUE)
           }
    )
    attr(sel_mat, "sel_method") <- features
  }else{
    hat <- NULL
    obs <- NULL
    sel_mat <- NULL
    approach <- fit$approach

    if(missing(base)){
      cli_abort("Argument {.arg base} is missing, with no default.", call = NULL)
    }else if(NCOL(base) != n){
      cli_abort("Incorrect {.arg base} columns dimension.", call = NULL)
    }
  }

  # if(is.null(fit)){
  #   if(missing(obs)){
  #     cli_abort("Argument {.arg obs} is missing, with no default.", call = NULL)
  #   }else if(NCOL(obs) != nb){
  #     cli_abort("Incorrect {.arg obs} columns dimension.", call = NULL)
  #   }
  #
  #   if(missing(hat)){
  #     cli_abort("Argument {.arg hat} is missing, with no default.", call = NULL)
  #   }
  #
  #   if(is.list(hat) && is.list(base)){
  #     p <- length(base)
  #
  #     ina <- sapply(base, function(bmat){
  #       is.na(colSums(bmat))
  #     })
  #
  #     hat <- lapply(hat, rbind)
  #     hat <- do.call(cbind, hat)
  #
  #     base <- lapply(base, rbind)
  #     base <- do.call(cbind, base)
  #
  #     if(NCOL(hat) != n*p){
  #       cli_abort("Incorrect {.arg hat} elements' columns dimension.", call = NULL)
  #     }
  #
  #     if(NCOL(base) != n*p){
  #       cli_abort("Incorrect {.arg base} elements' columns dimension.", call = NULL)
  #     }
  #
  #     hat <- hat[, !as.vector(ina), drop = FALSE]
  #     base <- base[, !as.vector(ina), drop = FALSE]
  #
  #     strc_mat <- do.call(rbind, rep(list(strc_mat), p))[!as.vector(ina), , drop = FALSE]
  #     id_bts <- rep(id_bts, p)[!as.vector(ina)]
  #
  #   }else if(!is.list(hat) && !is.list(base)){
  #     if(NCOL(hat) != n){
  #       cli_abort("Incorrect {.arg hat} columns dimension.", call = NULL)
  #     }
  #
  #     if(NCOL(base) != n){
  #       cli_abort("Incorrect {.arg base} columns dimension.", call = NULL)
  #     }
  #   }else{
  #     cli_abort("Incorrect {.arg base} or {.arg hat} arguments.", call = NULL)
  #   }
  #
  #   switch(features,
  #          "bts" = {
  #            sel_mat <- Matrix(rep(id_bts, nb), ncol = nb, sparse = TRUE)
  #          },
  #          "hier" = {
  #            sel_mat <- strc_mat
  #          },
  #          "hier-bts" = {
  #            sel_mat <- strc_mat + Matrix(rep(id_bts, nb), ncol = nb, sparse = TRUE)
  #            sel_mat[sel_mat != 0] <- 1
  #          },
  #          "all" = {
  #            sel_mat <- Matrix(1, nrow = NCOL(base), ncol = nb, sparse = TRUE)
  #          }
  #   )
  #   attr(sel_mat, "sel_method") <- features
  # }else{
  #   hat <- NULL
  #   obs <- NULL
  #   sel_mat <- NULL
  #
  #   if(is.list(base)){
  #     p <- length(base)
  #
  #     ina <- sapply(base, function(bmat){
  #       is.na(colSums(bmat))
  #     })
  #
  #     base <- lapply(base, rbind)
  #     base <- do.call(cbind, base)
  #
  #     if(NCOL(base) != n*p){
  #       cli_abort("Incorrect {.arg base} elements' columns dimension.", call = NULL)
  #     }
  #
  #     base <- base[, !as.vector(ina), drop = FALSE]
  #
  #     strc_mat <- do.call(rbind, rep(list(strc_mat), p))[!as.vector(ina), , drop = FALSE]
  #     id_bts <- rep(id_bts, p)[!as.vector(ina)]
  #
  #   }else if(NCOL(base) != n){
  #     cli_abort("Incorrect {.arg base} columns dimension.", call = NULL)
  #   }
  # }

  reco_mat <- rml(base = base,
                  hat = hat,
                  obs = obs,
                  sel_mat = sel_mat,
                  approach = approach,
                  params = params,
                  seed = seed,
                  fit = fit,
                  tuning = tuning)

  if(!is.null(base)){
    fit <- attr(reco_mat, "fit")
    fit$approach <- approach
    attr(reco_mat, "fit") <- NULL
    reco_mat <- csbu(reco_mat, agg_mat = agg_mat, sntz = sntz)

    attr(reco_mat, "FoReco") <- list2env(list(fit = fit,
                                              framework = "Cross-sectional",
                                              forecast_horizon = NROW(reco_mat),
                                              cs_n = n,
                                              rfun = "csrml",
                                              ml = approach))
    return(reco_mat)
  }else{
    reco_mat$approach <- approach
    return(reco_mat)
  }
}
