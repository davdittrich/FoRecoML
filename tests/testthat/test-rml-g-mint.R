# Regression tests for method="rec" — FoReco optimal combination path (P1)

make_cs_fixture_small <- function() {
  set.seed(99)
  agg_mat <- t(c(1, 1))
  dimnames(agg_mat) <- list("A", c("B", "C"))
  n <- 3L; p <- 2L; T_obs <- 50L; h <- 2L
  hat  <- matrix(rnorm(T_obs * n), T_obs, n,
                 dimnames = list(NULL, c("A", "B", "C")))
  obs  <- matrix(rnorm(T_obs * p), T_obs, p,
                 dimnames = list(NULL, c("B", "C")))
  base <- matrix(rnorm(h * n), h, n,
                 dimnames = list(NULL, c("A", "B", "C")))
  list(agg_mat = agg_mat, hat = hat, obs = obs, base = base)
}

test_that("T4.1 csrml_g method=rec comb=ols produces reconciled matrix", {
  skip_if_not_installed("lightgbm")
  fx <- make_cs_fixture_small()
  result <- csrml_g(base = fx$base, hat = fx$hat, obs = fx$obs,
                    agg_mat = fx$agg_mat, method = "rec", comb = "ols", seed = 1L)
  expect_true(is.matrix(result))
  expect_false(is.null(attr(result, "FoReco")))
  expect_equal(attr(result, "FoReco")$rfun, "csrml_g")
})

test_that("T4.2 csrml_g method=rec comb=shr with validation_split=0.2", {
  skip_if_not_installed("lightgbm")
  fx <- make_cs_fixture_small()
  result <- csrml_g(base = fx$base, hat = fx$hat, obs = fx$obs,
                    agg_mat = fx$agg_mat, method = "rec", comb = "shr",
                    validation_split = 0.2, seed = 1L)
  expect_true(is.matrix(result))
  expect_true(all(is.finite(result)))
})

test_that("T4.3 terml_g method=rec comb=ols produces named vector", {
  skip_if_not_installed("lightgbm")
  fx <- make_g_fixture_te()
  result <- terml_g(base = fx$base, hat = fx$hat, obs = fx$obs,
                    agg_order = fx$agg_order,
                    method = "rec", comb = "ols", seed = 1L)
  expect_type(result, "double")
  expect_false(is.null(attr(result, "FoReco")))
})

test_that("T4.4 ctrml_g method=rec comb=ols produces reconciled matrix", {
  skip_if_not_installed("lightgbm")
  fx <- make_g_fixture_ct()
  result <- ctrml_g(base = fx$base, hat = fx$hat, obs = fx$obs,
                    agg_mat = fx$agg_mat, agg_order = fx$agg_order,
                    method = "rec", comb = "ols", seed = 1L)
  expect_true(is.matrix(result))
  expect_false(is.null(attr(result, "FoReco")))
})

test_that("T4.5 unsupported comb on terml_g aborts with informative message", {
  skip_if_not_installed("lightgbm")
  fx <- make_g_fixture_te()
  expect_error(
    terml_g(base = fx$base, hat = fx$hat, obs = fx$obs,
            agg_order = fx$agg_order,
            method = "rec", comb = "shr", seed = 1L),
    "only comb='ols' is supported"
  )
})

test_that("T4.6 method=bu default produces BU output (unchanged behavior)", {
  skip_if_not_installed("lightgbm")
  fx <- make_cs_fixture_small()
  r_bu  <- csrml_g(base = fx$base, hat = fx$hat, obs = fx$obs,
                   agg_mat = fx$agg_mat, method = "bu", seed = 1L)
  r_def <- csrml_g(base = fx$base, hat = fx$hat, obs = fx$obs,
                   agg_mat = fx$agg_mat, seed = 1L)
  expect_identical(r_bu, r_def)
})
