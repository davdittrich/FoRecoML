#' @keywords package
#' @aliases FoRecoML-package
"_PACKAGE"

#' @import Matrix FoReco
#' @importFrom randomForest randomForest
#' @importFrom xgboost xgb.DMatrix xgb.train
#' @importFrom lightgbm lgb.Dataset lgb.train
#' @importFrom mlr3 lrn as_task_regr rsmp
#' @importFrom mlr3tuning tnr auto_tuner
#' @importFrom stats na.omit setNames predict
#' @importFrom methods as is
#' @importFrom cli cli_abort cli_inform cli_warn
#' @importFrom paradox to_tune
#' @importFrom qs2 qs_save qs_read
#' @import mlr3learners
NULL
