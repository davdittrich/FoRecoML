# Tests for ctrml_g/terml_g/csrml_g wrappers (T7.3)

make_g_fixture_cs <- function(p = 4L, T_obs = 20L, h = 2L) {
  set.seed(99)
  na <- 2L
  n <- na + p
  agg_mat <- matrix(c(1,1,0,0,0,0,1,1), nrow = na, ncol = p,
                    dimnames = list(c("G1","G2"), paste0("S", seq_len(p))))
  obs <- matrix(rnorm(T_obs * p), T_obs, p,
                dimnames = list(NULL, paste0("S", seq_len(p))))
  hat <- matrix(rnorm(T_obs * n), T_obs, n,
                dimnames = list(NULL, c(rownames(agg_mat), colnames(obs))))
  base <- matrix(rnorm(h * n), h, n,
                 dimnames = list(NULL, colnames(hat)))
  list(agg_mat = agg_mat, obs = obs, hat = hat, base = base)
}

make_g_fixture_te <- function(T_obs = 60L, h = 2L, agg_order = c(4L, 2L, 1L)) {
  set.seed(99)
  m  <- max(agg_order)
  kt <- sum(m / agg_order)
  hat  <- matrix(rnorm(T_obs * kt), T_obs, kt,
                 dimnames = list(NULL, paste0("L", seq_len(kt))))
  obs  <- matrix(rnorm(T_obs),  T_obs, 1L,
                 dimnames = list(NULL, "S1"))
  base <- matrix(rnorm(h * m * kt), h * m, kt,
                 dimnames = list(NULL, colnames(hat)))
  list(agg_order = agg_order, obs = obs, hat = hat, base = base)
}

make_g_fixture_ct <- function(p = 2L, T_obs = 60L, h = 2L, agg_order = c(4L, 2L, 1L)) {
  set.seed(99)
  m   <- max(agg_order)
  kt  <- sum(m / agg_order)
  na  <- 1L
  n   <- na + p
  ncf <- n * kt
  agg_mat <- matrix(c(1, 1), nrow = na, ncol = p,
                    dimnames = list("G1", paste0("S", seq_len(p))))
  obs  <- matrix(rnorm(T_obs * p), T_obs, p,
                 dimnames = list(NULL, paste0("S", seq_len(p))))
  hat  <- matrix(rnorm(T_obs * ncf), T_obs, ncf,
                 dimnames = list(NULL, paste0("F", seq_len(ncf))))
  base <- matrix(rnorm(h * m * ncf), h * m, ncf,
                 dimnames = list(NULL, colnames(hat)))
  list(agg_mat = agg_mat, agg_order = agg_order, obs = obs, hat = hat, base = base)
}

test_that("(a) csrml_g returns rml_g_fit with framework='cs'", {
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
  expect_s3_class(result, "rml_g_fit")
  expect_equal(result$framework, "cs")
  expect_false(is.null(result$agg_mat))
  expect_equal(result$agg_mat, fx$agg_mat)
  expect_null(result$norm_params)
})

test_that("(b) terml_g returns rml_g_fit with framework='te'", {
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
  expect_s3_class(result, "rml_g_fit")
  expect_equal(result$framework, "te")
  expect_equal(result$agg_order, c(2L, 1L))
  expect_null(result$norm_params)
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
  expect_null(result_none$norm_params)
  expect_false(is.null(result_z$norm_params))
  expect_false(is.null(result_z$norm_params$center))
})
