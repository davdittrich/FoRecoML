# T7.5 — predict / print / summary S3 methods for rml_g_fit

make_s3_fixture <- function(p = 4L, T_obs = 20L, ncol_hat = 5L, seed = 77L) {
  set.seed(seed)
  obs <- matrix(rnorm(T_obs * p), T_obs, p,
                dimnames = list(NULL, paste0("S", seq_len(p))))
  hat <- matrix(rnorm(T_obs * ncol_hat), T_obs, ncol_hat,
                dimnames = list(NULL, paste0("f", seq_len(ncol_hat))))
  agg_mat <- matrix(1, 1, p)
  base    <- matrix(rnorm((1L + p) * 2L), 1L + p, 2L)
  list(obs = obs, hat = hat, agg_mat = agg_mat, base = base)
}

test_that("(a) predict.rml_g_fit returns numeric vector", {
  skip_if_not_installed("ranger")
  fx  <- make_s3_fixture()
  fit <- csrml_g(base = fx$base, hat = fx$hat, obs = fx$obs,
                 agg_mat = fx$agg_mat, approach = "ranger", seed = 1L)

  # One row of features; provide explicit series_id for a known level.
  newdata <- matrix(rnorm(ncol(fx$hat)), 1L, ncol(fx$hat),
                    dimnames = list(NULL, paste0("f", seq_len(ncol(fx$hat)))))
  preds <- predict(fit, newdata = newdata, series_id = "S1")

  expect_true(is.numeric(preds))
  expect_equal(length(preds), 1L)
})

test_that("(b) print.rml_g_fit produces non-empty output mentioning backend", {
  skip_if_not_installed("ranger")
  fx  <- make_s3_fixture()
  fit <- csrml_g(base = fx$base, hat = fx$hat, obs = fx$obs,
                 agg_mat = fx$agg_mat, approach = "ranger", seed = 1L)

  # cli writes to stderr; capture via type = "message"
  output <- capture.output(print(fit), type = "message")
  expect_true(length(output) > 0L)
  expect_true(any(grepl("ranger", output, ignore.case = TRUE)))
  expect_invisible(print(fit))
})

test_that("(c) summary.rml_g_fit runs without error and returns invisibly", {
  skip_if_not_installed("ranger")
  fx  <- make_s3_fixture()
  fit <- csrml_g(base = fx$base, hat = fx$hat, obs = fx$obs,
                 agg_mat = fx$agg_mat, approach = "ranger", seed = 1L)

  expect_no_error(summary(fit))
  expect_invisible(summary(fit))
})

test_that("(d) predict: unseen series_id level triggers cli_abort", {
  skip_if_not_installed("ranger")
  fx  <- make_s3_fixture()
  fit <- csrml_g(base = fx$base, hat = fx$hat, obs = fx$obs,
                 agg_mat = fx$agg_mat, approach = "ranger", seed = 1L)

  newdata <- matrix(rnorm(ncol(fx$hat)), 1L, ncol(fx$hat),
                    dimnames = list(NULL, paste0("f", seq_len(ncol(fx$hat)))))
  expect_error(
    predict(fit, newdata = newdata, series_id = "S_UNSEEN"),
    regexp = "[Uu]nknown.*series_id|series_id.*level",
    ignore.case = TRUE
  )
})

test_that("(e) summary handles NULL feature_importance gracefully", {
  skip_if_not_installed("ranger")
  fx  <- make_s3_fixture()
  fit <- csrml_g(base = fx$base, hat = fx$hat, obs = fx$obs,
                 agg_mat = fx$agg_mat, approach = "ranger", seed = 1L)

  fit$feature_importance <- NULL
  expect_no_error(summary(fit))   # cli_inform, no error
})
