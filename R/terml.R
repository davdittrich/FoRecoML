#' Temporal Reconciliation with Machine Learning
#'
#' @param hat todo
#' @param obs todo
#' @param base todo
#' @param agg_order todo
#' @param tew todo
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
#' set.seed(123)
#' # (7 x 1) base forecasts vector (simulated), m = 4
#'
#' hat <- rnorm(7*10, rep(c(20, 10, 5), 10*c(1, 2, 4)))
#' obs <- rnorm(4*10, 5)
#' base <- rnorm(7*2, rep(c(20, 10, 5), 2*c(1, 2, 4)))
#'
#' m <- 4 # from quarterly to annual temporal aggregation
#' reco <- terml(hat = hat, obs = obs, base = base, agg_order = m,
#'               features = "all")
#'
#' @export
terml <- function(hat, obs, base, agg_order, features = "all", approach = "randomForest",
                  params = NULL, tuning = NULL, fit = NULL, tew = "sum", sntz = FALSE,
                  seed = NULL){

  features <- match.arg(features, c("all", "hfts", "str", "str-hfts", "rtw"))

  # Check if 'agg_order' is provided
  if(missing(agg_order)){
    cli_abort("Argument {.arg agg_order} is missing, with no default.", call = NULL)
  }

  tmp <- tetools(agg_order = agg_order, tew = tew)
  kset <- tmp$set
  m <- tmp$dim[["m"]]
  kt <- tmp$dim[["kt"]]
  id_hfts <- c(rep(0, tmp$dim[["ks"]]), rep(1, m))
  strc_mat <- tmp$strc_mat
  agg_mat <- tmp$agg_mat

  if(is.null(fit)){
    if(missing(obs)){
      cli_abort("Argument {.arg obs} is missing, with no default.", call = NULL)
    }else if(length(obs) %% m != 0){
      cli_abort("Incorrect {.arg obs} length.", call = NULL)
    }else{
      if(grepl("rtw", features)){
        obs <- cbind(obs)
      }else{
        obs <- matrix(obs, ncol = m, byrow = TRUE)
      }
    }

    if(missing(hat)){
      cli_abort("Argument {.arg hat} is missing, with no default.", call = NULL)
    }else if(length(hat) %% kt != 0){
      cli_abort("Incorrect {.arg hat} length.", call = NULL)
    }else{
      if(grepl("rtw", features)){
        hat <- input2rtw(hat, kset)
      }else{
        h <- length(hat) / kt
        hat <- vec2hmat(vec = hat, h = h, kset = kset)
      }
    }

    if(missing(base)){
      base <- NULL
    }else if(length(base) %% kt != 0){
      cli_abort("Incorrect {.arg base} length.", call = NULL)
    } else {
      h <- length(base) / kt
      if(grepl("rtw", features)){
        base <- input2rtw(base, kset)
      }else{
        base <- vec2hmat(vec = base, h = h, kset = kset)
      }
    }

    switch(features,
           "hfts" = {
             sel_mat <- Matrix(rep(id_hfts, m), ncol = m, sparse = TRUE)
           },
           "str" = {
             sel_mat <- strc_mat
           },
           "str-hfts" = {
             sel_mat <- strc_mat + Matrix(rep(id_hfts, m), ncol = m, sparse = TRUE)
             sel_mat[sel_mat != 0] <- 1
           },
           "all" = {
             sel_mat <- Matrix(1, nrow = kt, ncol = m, sparse = TRUE)
           },
           "rtw" = {
             sel_mat <- 1
             block_sampling <- h
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
    }else if(length(base) %% kt != 0){
      cli_abort("Incorrect {.arg base} length.", call = NULL)
    } else {
      h <- length(base) / kt
      base <- vec2hmat(vec = base, h = h, kset = kset)
    }
  }

  reco_mat <- rml(base = base,
                  hat = hat,
                  obs = obs,
                  sel_mat = sel_mat,
                  approach = approach,
                  params = params,
                  seed = seed,
                  fit = fit,
                  tuning = tuning,
                  block_sampling = block_sampling)

  if(!is.null(base)){
    fit <- attr(reco_mat, "fit")
    fit$approach <- approach
    attr(reco_mat, "fit") <- NULL
    reco_mat <- tebu(as.vector(t(reco_mat)), agg_order = agg_order, sntz = sntz, tew = tew)

    attr(reco_mat, "FoReco") <- list2env(list(fit = fit,
                                              framework = "Temporal",
                                              forecast_horizon = h,
                                              te_set = tmp$set,
                                              rfun = "terml",
                                              ml = approach))
    return(reco_mat)
  }else{
    reco_mat$approach <- approach
    return(reco_mat)
  }
}
