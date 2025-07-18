#' @keywords package
#' @aliases mlReco-package
"_PACKAGE"

#' @import Matrix FoReco
#' @importFrom randomForest randomForest
#' @importFrom xgboost xgb.DMatrix xgb.train
#' @importFrom lightgbm lgb.Dataset lgb.train
#' @importFrom mlr3 lrn as_task_regr rsmp
#' @importFrom mlr3tuning tnr auto_tuner
#' @importFrom stats na.omit setNames predict
#' @importFrom methods as is
#' @importFrom cli cli_abort
NULL
