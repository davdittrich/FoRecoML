#' Cross-temporal Reconciliation with Machine Learning
#'
#' @param hat todo
#' @param obs todo
#' @param base todo
#' @param agg_mat todo
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
#' @returns todo
#' @export
#'
#' @examples
#' hat <- rnorm(7*10, rep(c(20, 10, 5), 10*c(1, 2, 4)))
#' obs <- rnorm(4*10, 5)
#' base <- rnorm(7*3, rep(c(20, 10, 5), 3*c(1, 2, 4)))
#'
#' base <- rbind(base+100, base*10, base)
#' obs <- rbind(obs*10, obs)
#' hat <- rbind(hat+100, hat*10, hat)
#' m <- 4 # from quarterly to annual temporal aggregation
#' reco <- ctrml(hat = hat, obs = obs, base = base, agg_order = m, tew = "sum",
#'               agg_mat = t(c(1,1)), approach = "xgboost", seed = 123,
#'               #params = list(.key = "regr.xgboost"),
#'               features = "str")
#'
#' reco <- ctrml(hat = hat, obs = obs, base = base, agg_order = m, tew = "sum",
#'               agg_mat = t(c(1,1)), approach = "lightgbm", seed = 123,
#'               #params = list(.key = "regr.xgboost"),
#'               features = "str")
#'
#' # tweedie xgboost
#' reco <- ctrml(hat = hat, obs = obs, base = base, agg_order = m, tew = "sum",
#'               agg_mat = t(c(1,1)), approach = "xgboost", seed = 123,
#'               params =  list(
#'                 eta = 0.3,
#'                 colsample_bytree = 1,
#'                 min_child_weight = 1,
#'                 max_depth = 6,
#'                 gamma = 0,
#'                 subsample = 1,
#'                 objective = "reg:tweedie",
#'                 tweedie_variance_power = 1.5
#'               ),
#'               features = "str")
#'
#' # Tuning mlr3
#' reco <- ctrml(hat = hat, obs = obs, base = base, agg_order = m, tew = "sum",
#'               agg_mat = t(c(1,1)), approach = "mlr3", seed = 123,
#'               params = list(.key = "regr.ranger",
#'                             mtry = paradox::to_tune(paradox::p_int(1, 4))),
#'               tuning = list(terminator = mlr3tuning::trm("evals", n_evals = 10)),
#'               features = "rtw-full")
#'
ctrml <- function(hat, obs, base, agg_mat, agg_order, features = "all",
                  approach = "randomForest", params = NULL, tuning = NULL,
                  fit = NULL, tew = "sum", sntz = FALSE, seed = NULL){

  features <- match.arg(features, c("all", "hfts", "str", "str-hfts", "rtw-full", "rtw-comp"))

  # Check if 'agg_order' is provided
  if(missing(agg_order)){
    cli_abort("Argument {.arg agg_order} is missing, with no default.", call = NULL)
  }

  tmp <- cttools(agg_mat = agg_mat, agg_order = agg_order, tew = tew)
  strc_mat <- tmp$strc_mat

  id_bts <- c(rep(0, tmp$dim[["na"]]), rep(1, tmp$dim[["nb"]]))
  id_hfts <- c(rep(0, tmp$dim[["ks"]]), rep(1, tmp$dim[["m"]]))
  id_hfbts <- as.numeric(kronecker(id_bts, id_hfts))

  block_sampling <- NULL # block_sampling for the block tuning rtw option on mlr3

  if(is.null(fit)){

    if(missing(obs)){
      cli_abort("Argument {.arg obs} is missing, with no default.", call = NULL)
    }else if(NCOL(obs) %% tmp$dim[["m"]] != 0){
      cli_abort("Incorrect {.arg obs} columns dimension.", call = NULL)
    }else if(NROW(obs) != tmp$dim[["nb"]]){
      cli_abort("Incorrect {.arg obs} rows dimension.", call = NULL)
    }else{
      if(grepl("rtw", features)){
        obs <- t(obs)
      }else{
        obs <- matrix(as.vector(t(obs)), ncol = tmp$dim[["m"]]*tmp$dim[["nb"]])
      }

    }

    if(missing(hat)){
      cli_abort("Argument {.arg hat} is missing, with no default.", call = NULL)
    }else if(NCOL(hat) %% tmp$dim[["kt"]] != 0){
      cli_abort("Incorrect {.arg hat} columns dimension.", call = NULL)
    }else if(NROW(hat) != tmp$dim[["n"]]){
      cli_abort("Incorrect {.arg hat} rows dimension.", call = NULL)
    }else{
      if(grepl("rtw", features)){
        hat <- input2rtw(hat, tmp$set)
      }else{
        h <- NCOL(hat) / tmp$dim[["kt"]]
        hat <- mat2hmat(hat, h = h, kset = tmp$set, n = tmp$dim[["n"]])
      }
    }
    if(missing(base)){
      base <- NULL
    }else if(NCOL(base) %% tmp$dim[["kt"]] != 0){
      cli_abort("Incorrect {.arg base} columns dimension.", call = NULL)
    }else if(NROW(base) != tmp$dim[["n"]]){
      cli_abort("Incorrect {.arg base} rows dimension.", call = NULL)
    }else{
      h <- NCOL(base) / tmp$dim[["kt"]]
      if(grepl("rtw", features)){
        base <- input2rtw(base, tmp$set)
      }else{
        # Calculate 'h' and 'base_hmat'
        base <- mat2hmat(base, h = h, kset = tmp$set, n = tmp$dim[["n"]])
      }
    }

    switch(features,
           "hfbts" = {
             sel_mat <- as(id_hfbts, "sparseVector")
           },
           "hfts" = {
             sel_mat <- as(rep(id_hfts, tmp$dim[["n"]]), "sparseVector")
           },
           "bts" = {
             sel_mat <- as(rep(id_bts, each = tmp$dim[["kt"]]), "sparseVector")
           },
           "str" = {
             sel_mat <- strc_mat
           },
           "str-hfbts" = {
             sel_mat <- strc_mat + Matrix(rep(id_hfbts, tmp$dim[["nb"]]*tmp$dim[["m"]]),
                                          ncol = tmp$dim[["nb"]]*tmp$dim[["m"]], sparse = TRUE)
             sel_mat[sel_mat != 0] <- 1
           },
           "str-bts" = {
             sel_mat <- strc_mat + Matrix(rep(rep(id_bts, each = tmp$dim[["kt"]]),
                                              tmp$dim[["nb"]]*tmp$dim[["m"]]),
                                          ncol = tmp$dim[["nb"]]*tmp$dim[["m"]], sparse = TRUE)
             sel_mat[sel_mat != 0] <- 1
           },
           "all" = {
             sel_mat <- 1
           },
           "rtw-full" = {
             sel_mat <- 1
             block_sampling <- h
           },
           "rtw-comp" = {
             pos <- seq(tmp$dim[["na"]], by = tmp$dim[["n"]], length.out = tmp$dim[["p"]])
             sel_mat <- t(Matrix::bandSparse(tmp$dim[["nb"]], tmp$dim[["n"]]*tmp$dim[["p"]], pos)*1)
             sel_mat[1:tmp$dim[["n"]],] <- 1
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
    }else if(NCOL(base) %% tmp$dim[["kt"]] != 0){
      cli_abort("Incorrect {.arg base} columns dimension.", call = NULL)
    }else if(NROW(base) != tmp$dim[["n"]]){
      cli_abort("Incorrect {.arg base} rows dimension.", call = NULL)
    }else{
      if(grepl("rtw", features)){
        base <- input2rtw(base, tmp$set)
      }else{
        # Calculate 'h' and 'base_hmat'
        h <- NCOL(base) / tmp$dim[["kt"]]
        base <- mat2hmat(base, h = h, kset = tmp$set, n = tmp$dim[["n"]])
      }
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
    if(!grepl("rtw", features)){
      reco_mat <- matrix(as.vector(reco_mat), ncol = tmp$dim[["nb"]])
    }
    reco_mat <- ctbu(t(reco_mat), agg_order = agg_order, agg_mat = agg_mat,
                     sntz = sntz, tew = tew)

    attr(reco_mat, "FoReco") <- list2env(list(fit = fit,
                                         framework = "Cross-temporal",
                                         forecast_horizon = h,
                                         te_set = tmp$set,
                                         cs_n = tmp$dim[["n"]],
                                         rfun = "ctrml",
                                         ml = approach))
    return(reco_mat)
  }else{
    reco_mat$approach <- approach
    return(reco_mat)
  }
}

input2rtw <- function(x, kset){
  x <- FoReco::FoReco2matrix(x, kset)
  x <- lapply(1:length(kset), function(i){
    if(NCOL(x[[i]])>1){
      tmp <- apply(x[[i]], 2, rep, each = kset[i])
      colnames(tmp) <- paste0(colnames(tmp), "_", kset[i])
    }else{
      tmp <- rep(x[[i]], each = kset[i])
    }
    tmp
  })
  do.call(cbind, rev(x))
}
