# Tests for input_format = "wide_ct" in ctrml_g() and terml_g() (T3).
# Covers convert_wide_ct() shapes, round-trip regression, backward compat,
# dimension validation, and cs_level feature column.

make_wide_ct_fixture <- function(n_folds = 5L, agg_order = c(4L, 2L, 1L)) {
  set.seed(42)
  m  <- max(agg_order)
  kt <- sum(m / agg_order)
  n_s <- 3L  # U (upper), A, B (bottom)
  n_b <- 2L  # bottom series
  agg_mat <- matrix(c(1, 1), 1, 2, dimnames = list("U", c("A", "B")))
  # hat_wide: n_series × (n_folds × kt)  — colnames required for LightGBM
  hat_wide <- matrix(
    rnorm(n_s * n_folds * kt), n_s, n_folds * kt,
    dimnames = list(c("U", "A", "B"), paste0("f", seq_len(n_folds * kt)))
  )
  # obs_wide: n_bottom × T_monthly
  obs_wide <- matrix(
    rnorm(n_b * n_folds * m), n_b, n_folds * m,
    dimnames = list(c("A", "B"), NULL)
  )
  # base_wide: n_series × kt
  base_wide <- matrix(
    rnorm(n_s * kt), n_s, kt,
    dimnames = list(c("U", "A", "B"), paste0("f", seq_len(kt)))
  )
  list(
    agg_mat   = agg_mat,
    agg_order = agg_order,
    hat_wide  = hat_wide,
    obs_wide  = obs_wide,
    base_wide = base_wide,
    kt        = kt,
    n_folds   = n_folds,
    n_s       = n_s
  )
}

test_that("convert_wide_ct: output shapes correct", {
  fx  <- make_wide_ct_fixture()
  wct <- FoRecoML:::convert_wide_ct(
    fx$hat_wide, fx$obs_wide, fx$base_wide,
    fx$agg_mat, fx$agg_order
  )
  expect_equal(dim(wct$X_stacked), c(fx$n_s * fx$n_folds, fx$kt))
  expect_equal(length(wct$y_stacked), fx$n_s * fx$n_folds)
  expect_setequal(wct$series_id_levels, c("A", "B", "U"))
  expect_equal(dim(wct$base_tall), c(fx$n_s, fx$kt))
})

test_that("ctrml_g input_format='wide_ct' produces reconciled matrix", {
  skip_if_not_installed("lightgbm")
  fx <- make_wide_ct_fixture()
  r  <- ctrml_g(
    base       = fx$base_wide,
    hat        = fx$hat_wide,
    obs        = fx$obs_wide,
    agg_mat    = fx$agg_mat,
    agg_order  = fx$agg_order,
    input_format = "wide_ct",
    seed       = 1L
  )
  expect_true(is.matrix(r))
  expect_false(is.null(attr(r, "FoReco")))
})

test_that("ctrml_g input_format='tall' (default): backward compat", {
  skip_if_not_installed("lightgbm")
  agg_mat   <- t(c(1, 1))
  dimnames(agg_mat) <- list("A", c("B", "C"))
  agg_order <- c(4L, 2L, 1L)
  m  <- 4L; kt <- 7L; T_obs <- 60L; h <- 2L
  hat  <- matrix(rnorm(T_obs * 21), T_obs, 21,
                 dimnames = list(NULL, paste0("F", seq_len(21))))
  obs  <- matrix(rnorm(T_obs * 2), T_obs, 2,
                 dimnames = list(NULL, c("B", "C")))
  base <- matrix(rnorm(h * m * 21), h * m, 21,
                 dimnames = list(NULL, paste0("F", seq_len(21))))
  r <- ctrml_g(
    base      = base, hat = hat, obs = obs,
    agg_mat   = agg_mat, agg_order = agg_order,
    input_format = "tall", seed = 1L
  )
  expect_true(is.matrix(r))
})

test_that("wrong dims with wide_ct -> cli_abort mentioning 'kt'", {
  fx       <- make_wide_ct_fixture()
  bad_base <- matrix(rnorm(fx$n_s * 3), fx$n_s, 3)
  expect_error(
    FoRecoML:::convert_wide_ct(
      fx$hat_wide, fx$obs_wide, bad_base,
      fx$agg_mat, fx$agg_order
    ),
    "kt"
  )
})

test_that("terml_g input_format='wide_ct' works", {
  skip_if_not_installed("lightgbm")
  agg_order <- c(4L, 2L, 1L)
  m         <- max(agg_order)
  kt        <- sum(m / agg_order)
  n_folds   <- 5L
  hat_wide  <- matrix(rnorm(kt * n_folds), 1, kt * n_folds)
  obs_wide  <- matrix(rnorm(1 * n_folds * m), 1, n_folds * m)
  base_wide <- matrix(rnorm(kt), 1, kt)
  r <- terml_g(
    base        = base_wide,
    hat         = hat_wide,
    obs         = obs_wide,
    agg_order   = agg_order,
    input_format = "wide_ct",
    seed        = 1L
  )
  expect_type(r, "double")
})
