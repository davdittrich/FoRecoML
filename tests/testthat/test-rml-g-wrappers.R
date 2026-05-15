# Tests for ctrml_g/terml_g/csrml_g wrappers (T7.3)

make_g_fixture <- function(p = 4L, T_obs = 20L, ncol_hat = 8L) {
  set.seed(99)
  agg_mat <- matrix(c(1, 1, 0, 0, 0, 0, 1, 1), nrow = 2, ncol = p)
  rownames(agg_mat) <- c("G1", "G2")
  obs <- matrix(rnorm(T_obs * p), T_obs, p,
                dimnames = list(NULL, paste0("S", seq_len(p))))
  hat <- matrix(rnorm(T_obs * ncol_hat), T_obs, ncol_hat)
  base <- matrix(rnorm((nrow(agg_mat) + p) * 2), nrow(agg_mat) + p, 2)
  rownames(base) <- c(rownames(agg_mat), paste0("S", seq_len(p)))
  list(agg_mat = agg_mat, obs = obs, hat = hat, base = base)
}

test_that("(a) csrml_g returns rml_g_fit with framework='cs'", {
  skip_if_not_installed("ranger")
  fx <- make_g_fixture()
  result <- csrml_g(
    base     = fx$base,
    hat      = fx$hat,
    obs      = fx$obs,
    agg_mat  = fx$agg_mat,
    approach = "ranger",
    seed     = 1L
  )
  expect_s3_class(result, "rml_g_fit")
  expect_equal(result$framework, "cs")
  expect_false(is.null(result$agg_mat))
  expect_equal(result$agg_mat, fx$agg_mat)
  expect_null(result$norm_params)
})

test_that("(b) terml_g returns rml_g_fit with framework='te'", {
  skip_if_not_installed("ranger")
  fx <- make_g_fixture()
  result <- terml_g(
    base      = fx$base,
    hat       = fx$hat,
    obs       = fx$obs,
    agg_order = c(2L, 1L),
    approach  = "ranger",
    seed      = 1L
  )
  expect_s3_class(result, "rml_g_fit")
  expect_equal(result$framework, "te")
  expect_equal(result$agg_order, c(2L, 1L))
  expect_null(result$norm_params)
})

test_that("(c) normalize='zscore' changes hat before fitting", {
  skip_if_not_installed("ranger")
  fx <- make_g_fixture()
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
  expect_null(result_none$norm_params)
  expect_false(is.null(result_z$norm_params))
  expect_false(is.null(result_z$norm_params$center))
})
