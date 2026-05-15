# Regression tests for restored *_g fit+predict+reconcile pipeline (Epic ak9)

test_that("csrml_g returns reconciled matrix with FoReco attr + coherency", {
  skip_if_not_installed("lightgbm")
  fx <- make_g_fixture_cs()
  result <- csrml_g(base = fx$base, hat = fx$hat, obs = fx$obs,
                    agg_mat = fx$agg_mat, approach = "lightgbm", seed = 1L)
  expect_true(is.matrix(result))
  expect_false(is.null(attr(result, "FoReco")))
  expect_s3_class(attr(result, "FoReco"), "foreco_info")
  # result is h x n (rows=time, cols=series); check column coherency
  # upper cols = agg_mat %*% bottom cols (for each time step)
  na <- nrow(fx$agg_mat); p <- ncol(fx$agg_mat)
  upper_idx  <- seq_len(na)
  bottom_idx <- seq.int(na + 1L, na + p)
  upper  <- t(result[, upper_idx,  drop = FALSE])   # na x h
  bottom <- t(result[, bottom_idx, drop = FALSE])   # p  x h
  expect_equal(as.numeric(fx$agg_mat %*% bottom),
               as.numeric(upper),
               tolerance = 1e-10)
})

test_that("terml_g returns named vector with correct length", {
  skip_if_not_installed("lightgbm")
  fx <- make_g_fixture_te()
  result <- terml_g(base = fx$base, hat = fx$hat, obs = fx$obs,
                    agg_order = fx$agg_order, approach = "lightgbm", seed = 1L)
  expect_type(result, "double")
  expect_false(is.null(names(result)))
  expect_false(is.null(attr(result, "FoReco")))
  m  <- max(fx$agg_order); kt <- sum(m / fx$agg_order); h <- nrow(fx$base) / m
  expect_equal(length(result), h * kt)
})

test_that("ctrml_g returns n x (h*kt) reconciled matrix", {
  skip_if_not_installed("lightgbm")
  fx <- make_g_fixture_ct()
  result <- ctrml_g(base = fx$base, hat = fx$hat, obs = fx$obs,
                    agg_mat = fx$agg_mat, agg_order = fx$agg_order,
                    approach = "lightgbm", seed = 1L)
  expect_true(is.matrix(result))
  expect_false(is.null(attr(result, "FoReco")))
  m   <- max(fx$agg_order); kt <- sum(m / fx$agg_order)
  na  <- nrow(fx$agg_mat); p <- ncol(fx$agg_mat); n <- na + p
  h   <- nrow(fx$base) / m
  expect_equal(dim(result), c(n, h * kt))
})

test_that("extract_reconciled_ml returns rml_g_fit for *_g results", {
  skip_if_not_installed("lightgbm")
  fx <- make_g_fixture_cs()
  result <- csrml_g(base = fx$base, hat = fx$hat, obs = fx$obs,
                    agg_mat = fx$agg_mat, approach = "lightgbm", seed = 1L)
  fit <- extract_reconciled_ml(result)
  expect_s3_class(fit, "rml_g_fit")
})

test_that("csrml_g respects sntz=TRUE (non-negative reconciliation)", {
  skip_if_not_installed("lightgbm")
  fx <- make_g_fixture_cs()
  result <- csrml_g(base = fx$base, hat = fx$hat, obs = fx$obs,
                    agg_mat = fx$agg_mat, approach = "lightgbm",
                    sntz = TRUE, seed = 1L)
  expect_true(all(result >= 0))
})
