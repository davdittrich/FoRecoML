# Tests for ctrml_g/terml_g/csrml_g wrappers (T7.3)
# Fixture constructors live in helper-g-fixtures.R

test_that("(a) csrml_g returns reconciled matrix with FoReco attr", {
  skip_if_not_installed("ranger")
  fx <- make_g_fixture_cs()
  result <- csrml_g(
    base     = fx$base,
    hat      = fx$hat,
    obs      = fx$obs,
    agg_mat  = fx$agg_mat,
    approach = "ranger",
    seed     = 1L
  )
  expect_true(is.matrix(result))
  expect_false(is.null(attr(result, "FoReco")))
  expect_equal(attr(result, "FoReco")$framework, "Cross-sectional")
  expect_equal(attr(result, "FoReco")$fit$agg_mat, fx$agg_mat)
  expect_null(attr(result, "FoReco")$fit$norm_params)
})

test_that("(b) terml_g returns named vector with FoReco attr", {
  skip_if_not_installed("ranger")
  fx <- make_g_fixture_te()
  result <- terml_g(
    base      = fx$base,
    hat       = fx$hat,
    obs       = fx$obs,
    agg_order = fx$agg_order,
    approach  = "ranger",
    seed      = 1L
  )
  expect_type(result, "double")
  expect_false(is.null(names(result)))
  expect_false(is.null(attr(result, "FoReco")))
  expect_equal(attr(result, "FoReco")$framework, "Temporal")
  expect_equal(attr(result, "FoReco")$fit$agg_order, fx$agg_order)
  expect_null(attr(result, "FoReco")$fit$norm_params)
})

test_that("(c) normalize='zscore' changes hat before fitting", {
  skip_if_not_installed("ranger")
  fx <- make_g_fixture_cs()
  result_none <- csrml_g(
    base      = fx$base,
    hat       = fx$hat,
    obs       = fx$obs,
    agg_mat   = fx$agg_mat,
    approach  = "ranger",
    normalize = "none",
    seed      = 42L
  )
  result_z <- csrml_g(
    base      = fx$base,
    hat       = fx$hat,
    obs       = fx$obs,
    agg_mat   = fx$agg_mat,
    approach  = "ranger",
    normalize = "zscore",
    seed      = 42L
  )
  expect_null(attr(result_none, "FoReco")$fit$norm_params)
  expect_false(is.null(attr(result_z, "FoReco")$fit$norm_params))
  expect_false(is.null(attr(result_z, "FoReco")$fit$norm_params$center))
})
