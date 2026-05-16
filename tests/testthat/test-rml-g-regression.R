# Regression tests locking in B1/B2 code-review fixes and T2 temporal split

make_reg_fixture <- function(T_obs = 20L, p = 3L) {
  set.seed(42)
  kt <- 7L
  hat <- matrix(rnorm(T_obs * kt), T_obs, kt,
                dimnames = list(NULL, paste0("L", seq_len(kt))))
  obs <- matrix(rnorm(T_obs * p), T_obs, p,
                dimnames = list(NULL, paste0("S", seq_len(p))))
  list(hat = hat, obs = obs, T_obs = T_obs, p = p, kt = kt)
}

# -------- Group A: B1/B2 fixes --------

test_that("A1: series_id_int_valid stored; length == length(valid_idx)", {
  skip_if_not_installed("lightgbm")
  fx <- make_reg_fixture()
  fit <- rml_g("lightgbm", hat = fx$hat, obs = fx$obs,
                validation_split = 0.2, seed = 1L)
  expect_equal(length(fit$series_id_int_valid), length(fit$valid_idx))
  expect_true(length(fit$series_id_int_valid) > 0L)
})

test_that("A2: X_valid has ncol(hat) cols when level_id=FALSE", {
  skip_if_not_installed("lightgbm")
  fx <- make_reg_fixture()
  fit <- rml_g("lightgbm", hat = fx$hat, obs = fx$obs,
                validation_split = 0.2, seed = 1L)
  expect_equal(ncol(fit$X_valid), fx$kt)
})

test_that("A3: X_valid has ncol(hat) cols when level_id=TRUE (level_id_col stripped)", {
  skip_if_not_installed("lightgbm")
  fx <- make_reg_fixture()
  fit <- rml_g("lightgbm", hat = fx$hat, obs = fx$obs,
                validation_split = 0.2, level_id = TRUE,
                kset = c(4L, 2L, 1L), seed = 1L)
  # ncol(hat) = kt, NOT kt+1 (level_id_col must be stripped)
  expect_equal(ncol(fit$X_valid), fx$kt)
})

test_that("A4: predict with raw features + level_id=TRUE: no abort, correct length", {
  skip_if_not_installed("lightgbm")
  fx <- make_reg_fixture(T_obs = 40L, p = 2L)
  fit <- rml_g("lightgbm", hat = fx$hat, obs = fx$obs,
                validation_split = 0.2, level_id = TRUE,
                kset = c(4L, 2L, 1L), seed = 1L)
  newdata <- fx$hat[1:4, , drop = FALSE]  # ncol(hat) cols, no level_id_col
  preds <- predict(fit, newdata = newdata, series_id = "S1")
  expect_length(preds, 4L)
})

test_that("A5: compute_rec_residuals aborts when validation_split=0", {
  skip_if_not_installed("lightgbm")
  fx <- make_reg_fixture()
  fit <- rml_g("lightgbm", hat = fx$hat, obs = fx$obs, seed = 1L)
  expect_error(
    FoRecoML:::compute_rec_residuals(fit),
    "validation_split > 0"
  )
})

# -------- Group B: T2 temporal split --------

test_that("B6: valid_idx length == n_per_series * p (exact)", {
  skip_if_not_installed("lightgbm")
  fx <- make_reg_fixture()
  n_per_series <- max(1L, round(0.2 * fx$T_obs))   # round(0.2*20) = 4
  fit <- rml_g("lightgbm", hat = fx$hat, obs = fx$obs,
                validation_split = 0.2, seed = 1L)
  expect_equal(length(fit$valid_idx), n_per_series * fx$p)
})

test_that("B7: valid time indices are last n_per_series rows of each series block", {
  skip_if_not_installed("lightgbm")
  fx <- make_reg_fixture()
  n_per_series <- max(1L, round(0.2 * fx$T_obs))   # = 4
  fit <- rml_g("lightgbm", hat = fx$hat, obs = fx$obs,
                validation_split = 0.2, seed = 1L)
  # series 1: rows 17-20; series 2: rows 37-40; series 3: rows 57-60
  expected <- unlist(lapply(seq_len(fx$p), function(j) {
    be <- j * fx$T_obs
    (be - n_per_series + 1L):be
  }))
  expect_equal(fit$valid_idx, expected)
})

test_that("B8: valid_idx identical regardless of seed (deterministic split)", {
  skip_if_not_installed("lightgbm")
  fx <- make_reg_fixture()
  fit1 <- rml_g("lightgbm", hat = fx$hat, obs = fx$obs, validation_split = 0.2, seed = 1L)
  fit2 <- rml_g("lightgbm", hat = fx$hat, obs = fx$obs, validation_split = 0.2, seed = 99L)
  expect_identical(fit1$valid_idx, fit2$valid_idx)
})

test_that("B9: min_validation_rows guard fires when n_valid_total < 10", {
  skip_if_not_installed("lightgbm")
  # T_obs=5, p=2, validation_split=0.1 -> n_per_series=max(1,round(0.5))=1, total=2 < 10
  hat_s <- matrix(rnorm(5 * 3), 5, 3, dimnames = list(NULL, paste0("L", 1:3)))
  obs_s <- matrix(rnorm(5 * 2), 5, 2, dimnames = list(NULL, c("A", "B")))
  expect_warning(
    rml_g("lightgbm", hat = hat_s, obs = obs_s, validation_split = 0.1, seed = 1L),
    "Validation set has only"
  )
})

test_that("B10: cli_abort fires when validation_split too large", {
  skip_if_not_installed("lightgbm")
  fx <- make_reg_fixture()
  expect_error(
    rml_g("lightgbm", hat = fx$hat, obs = fx$obs, validation_split = 0.99, seed = 1L),
    "too large"
  )
})

# -------- T4: S4 structural test --------

test_that("T4: cli_warn present in all 5 feature_importance handlers", {
  # Verify the fix was applied — structural check on source file
  # devtools::test() sets cwd to tests/testthat; climb two levels to package root
  dev_path <- file.path(getwd(), "..", "..", "R", "rml_g.R")
  if (file.exists(dev_path)) {
    src <- readLines(dev_path)
    n_warn <- sum(grepl("feature_importance extraction failed", src, fixed = TRUE))
    expect_equal(n_warn, 5L)
  } else {
    skip("Not in development tree; skip structural test")
  }
})
