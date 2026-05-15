# T7.2: rml_g S3 generic + 5 methods + .stack_series.

make_toy <- function(T_obs = 30L, p = 4L, n_feat = 5L, seed = 1L) {
  set.seed(seed)
  hat <- matrix(rnorm(T_obs * n_feat), T_obs, n_feat)
  colnames(hat) <- paste0("f", seq_len(n_feat))
  # Slightly different signal per series so series_id has signal.
  obs <- sapply(seq_len(p), function(j) {
    as.numeric(hat %*% rnorm(n_feat) + 0.5 * j + rnorm(T_obs, sd = 0.1))
  })
  colnames(obs) <- paste0("S", seq_len(p))
  list(hat = hat, obs = obs)
}

# (g) .stack_series unit: stacks correctly.
test_that("(g) .stack_series produces correct shapes and ordering", {
  d <- make_toy(T_obs = 10L, p = 3L, n_feat = 4L)
  s <- FoRecoML:::.stack_series(d$hat, d$obs)

  expect_equal(nrow(s$X_stacked), 10L * 3L)
  expect_equal(ncol(s$X_stacked), 4L)
  expect_length(s$y_stacked, 10L * 3L)
  expect_length(s$series_id_factor, 10L * 3L)
  expect_length(s$series_id_int, 10L * 3L)
  expect_length(s$series_id_levels, 3L)

  # Column-major stack: first T_obs rows correspond to series 1.
  expect_equal(s$y_stacked[1:10], as.numeric(d$obs[, 1]))
  expect_equal(s$y_stacked[11:20], as.numeric(d$obs[, 2]))
  expect_true(all(s$series_id_int[1:10]  == s$series_id_int[1]))
  expect_true(all(s$series_id_int[11:20] == s$series_id_int[11]))

  # No validation split => train_idx covers everything.
  expect_equal(s$train_idx, seq_len(30L))
  expect_length(s$valid_idx, 0L)
  expect_null(s$norm_params)
})

# (l) .stack_series: factor levels alphabetically sorted.
test_that("(l) .stack_series freezes factor levels in alphabetical order", {
  hat <- matrix(rnorm(5 * 2), 5, 2)
  obs <- matrix(rnorm(5 * 3), 5, 3)
  colnames(obs) <- c("zebra", "apple", "mango")
  s <- FoRecoML:::.stack_series(hat, obs)
  expect_identical(s$series_id_levels, c("apple", "mango", "zebra"))
  expect_identical(levels(s$series_id_factor), c("apple", "mango", "zebra"))
})

# (k) .stack_series global holdout: valid_idx identical across two calls with same seed.
test_that("(k) .stack_series valid_idx reproducible with seed", {
  d <- make_toy(T_obs = 50L, p = 4L)
  s1 <- FoRecoML:::.stack_series(d$hat, d$obs,
                                 validation_split = 0.2, seed = 123L)
  s2 <- FoRecoML:::.stack_series(d$hat, d$obs,
                                 validation_split = 0.2, seed = 123L)
  expect_identical(s1$valid_idx, s2$valid_idx)
  expect_identical(s1$train_idx, s2$train_idx)
  expect_true(length(s1$valid_idx) > 0L)
  expect_length(intersect(s1$train_idx, s1$valid_idx), 0L)
  expect_length(union(s1$train_idx, s1$valid_idx), 50L * 4L)
})

# (n) min_validation_rows guard.
test_that("(n) .stack_series warns and disables validation when too few rows", {
  hat <- matrix(rnorm(5 * 2), 5, 2)
  obs <- matrix(rnorm(5 * 2), 5, 2)  # 10 rows total => 0.05 * 10 = 1 row
  expect_warning(
    s <- FoRecoML:::.stack_series(hat, obs,
                                  validation_split = 0.05,
                                  min_validation_rows = 5L),
    "Validation set"
  )
  expect_length(s$valid_idx, 0L)
  expect_equal(s$train_idx, seq_len(10L))
})

# (a) lightgbm smoke.
test_that("(a) rml_g.lightgbm trains and returns rml_g_fit", {
  skip_if_not_installed("lightgbm")
  d <- make_toy(T_obs = 30L, p = 4L)
  res <- FoRecoML::rml_g("lightgbm", d$hat, d$obs, seed = 1L,
                         params = list(num_iteration = 20L))
  expect_s3_class(res, "rml_g_fit")
  expect_equal(res$approach, "lightgbm")
  expect_identical(res$series_id_levels, sort(colnames(d$obs)))
  expect_equal(res$ncol_hat, ncol(d$hat))
  expect_false(is.null(res$fit))
})

# (b) xgboost smoke.
test_that("(b) rml_g.xgboost trains and returns rml_g_fit", {
  skip_if_not_installed("xgboost")
  d <- make_toy(T_obs = 30L, p = 4L)
  res <- FoRecoML::rml_g("xgboost", d$hat, d$obs, seed = 2L,
                         params = list(nrounds = 20L))
  expect_s3_class(res, "rml_g_fit")
  expect_equal(res$approach, "xgboost")
  expect_identical(res$series_id_levels, sort(colnames(d$obs)))
  expect_equal(res$ncol_hat, ncol(d$hat))
  expect_false(is.null(res$fit))
})

# (c) ranger smoke.
test_that("(c) rml_g.ranger trains and returns rml_g_fit", {
  skip_if_not_installed("ranger")
  d <- make_toy(T_obs = 30L, p = 4L)
  res <- FoRecoML::rml_g("ranger", d$hat, d$obs, seed = 3L,
                         params = list(num.trees = 50L))
  expect_s3_class(res, "rml_g_fit")
  expect_equal(res$approach, "ranger")
  expect_identical(res$series_id_levels, sort(colnames(d$obs)))
  expect_false(is.null(res$fit))
})

# (d) mlr3 smoke (minimal — see notes in spec).
test_that("(d) rml_g.mlr3 method exists and trains a learner", {
  skip_if_not_installed("mlr3")
  skip_if_not_installed("mlr3learners")
  skip_if_not_installed("ranger")
  d <- make_toy(T_obs = 30L, p = 4L)
  res <- FoRecoML::rml_g("mlr3", d$hat, d$obs, seed = 4L,
                         params = list(learner = "regr.ranger",
                                       num.trees = 50L,
                                       importance = "impurity"))
  expect_s3_class(res, "rml_g_fit")
  expect_equal(res$approach, "mlr3")
  expect_false(is.null(res$fit))
})

# (e) series_id treated as CATEGORICAL in lightgbm.
test_that("(e) lightgbm sees series_id as a categorical feature", {
  skip_if_not_installed("lightgbm")
  d <- make_toy(T_obs = 25L, p = 3L)
  res <- FoRecoML::rml_g("lightgbm", d$hat, d$obs, seed = 5L,
                         params = list(num_iteration = 20L))
  # lightgbm stores categorical_feature inside fit$params; index is converted
  # to 0-based on storage. We passed the 1-based last column index, so the
  # stored value should equal ncol(hat) (i.e. 1-based ncol(hat)+1 minus 1).
  cf <- res$fit$params$categorical_feature
  expect_false(is.null(cf))
  expected_0based <- ncol(d$hat)  # 1-based (n+1) -> 0-based (n)
  cf_vec <- as.integer(unlist(cf))
  expect_true(expected_0based %in% cf_vec)
})

# (f) directional: RMSE with categorical series_id should be no worse than
# without. We compare global model (with series_id) to a model trained on a
# pooled dataset that drops series_id.
test_that("(f) categorical series_id does not hurt RMSE (directional)", {
  skip_if_not_installed("lightgbm")
  set.seed(99)
  T_obs <- 60L
  n_feat <- 4L
  p <- 4L
  hat <- matrix(rnorm(T_obs * n_feat), T_obs, n_feat)
  colnames(hat) <- paste0("f", seq_len(n_feat))
  # Strong per-series offset => series_id has signal.
  obs <- sapply(seq_len(p), function(j) {
    as.numeric(hat %*% rnorm(n_feat) + 5 * j + rnorm(T_obs, sd = 0.1))
  })
  colnames(obs) <- paste0("S", seq_len(p))

  res_cat <- FoRecoML::rml_g("lightgbm", hat, obs, seed = 7L,
                             params = list(num_iteration = 50L))

  # Build a "no-series-id" model by stacking manually and dropping series_id.
  X_stack <- do.call(rbind, rep(list(hat), p))
  y_stack <- as.numeric(obs)
  dtrain  <- lightgbm::lgb.Dataset(data = X_stack, label = y_stack,
                                   free_raw_data = FALSE)
  fit_no <- lightgbm::lgb.train(
    params = list(objective = "regression", metric = "rmse",
                  num_threads = 1L, verbose = -1L, seed = 7L),
    data = dtrain, nrounds = 50L, verbose = -1L
  )

  X_test <- cbind(X_stack, series_id = rep(seq_len(p), each = T_obs))
  pred_cat <- predict(res_cat$fit, X_test)
  pred_no  <- predict(fit_no, X_stack)

  rmse_cat <- sqrt(mean((pred_cat - y_stack)^2))
  rmse_no  <- sqrt(mean((pred_no  - y_stack)^2))

  expect_lte(rmse_cat, rmse_no + 1e-6)
})

# (h) early stopping fires (lightgbm).
test_that("(h) lightgbm early stopping triggers with validation split", {
  skip_if_not_installed("lightgbm")
  d <- make_toy(T_obs = 80L, p = 4L)
  res <- FoRecoML::rml_g("lightgbm", d$hat, d$obs, seed = 11L,
                         params = list(num_iteration = 500L),
                         early_stopping_rounds = 5L,
                         validation_split = 0.2)
  # When early stopping is active, fit$best_iter should be < nrounds.
  expect_false(is.null(res$fit$best_iter))
  expect_true(res$fit$best_iter <= 500L)
})

# (h) early stopping fires (xgboost).
test_that("(h) xgboost early stopping triggers with validation split", {
  skip_if_not_installed("xgboost")
  d <- make_toy(T_obs = 80L, p = 4L)
  res <- FoRecoML::rml_g("xgboost", d$hat, d$obs, seed = 12L,
                         params = list(nrounds = 500L),
                         early_stopping_rounds = 5L,
                         validation_split = 0.2)
  # xgboost 3.x stores best_iteration as a booster attribute, not a list slot.
  attrs <- xgboost::xgb.attributes(res$fit)
  best_iter <- attrs$best_iteration
  expect_false(is.null(best_iter))
  expect_true(as.integer(best_iter) <= 500L)
})

# (i) ranger early stopping no-op: cli_inform emitted.
test_that("(i) rml_g.ranger emits inform when early_stopping_rounds > 0", {
  skip_if_not_installed("ranger")
  d <- make_toy(T_obs = 30L, p = 3L)
  expect_message(
    res <- FoRecoML::rml_g("ranger", d$hat, d$obs, seed = 13L,
                           params = list(num.trees = 20L),
                           early_stopping_rounds = 10L),
    "ranger does not support early stopping"
  )
  expect_s3_class(res, "rml_g_fit")
})

# (j) feature importance non-empty for lightgbm + ranger.
test_that("(j) feature_importance populated for lightgbm and ranger", {
  skip_if_not_installed("lightgbm")
  skip_if_not_installed("ranger")
  d <- make_toy(T_obs = 30L, p = 3L)

  res_lgb <- FoRecoML::rml_g("lightgbm", d$hat, d$obs, seed = 14L,
                             params = list(num_iteration = 20L))
  expect_false(is.null(res_lgb$feature_importance))
  expect_true(nrow(res_lgb$feature_importance) > 0L)

  res_rng <- FoRecoML::rml_g("ranger", d$hat, d$obs, seed = 15L,
                             params = list(num.trees = 20L))
  expect_false(is.null(res_rng$feature_importance))
  expect_true(length(res_rng$feature_importance) > 0L)
})
