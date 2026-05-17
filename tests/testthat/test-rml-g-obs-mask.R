# Regression tests for obs_mask — structural zero filtering (FoRecoML-8r1)

make_obs_mask_fixture <- function(T_obs = 30L, p = 3L) {
  set.seed(42)
  kt <- 5L
  hat <- matrix(rnorm(T_obs * kt), T_obs, kt,
                dimnames = list(NULL, paste0("L", seq_len(kt))))
  # Series 1 (B) is all-zero (structural missing)
  obs <- matrix(rnorm(T_obs * p), T_obs, p,
                dimnames = list(NULL, paste0("S", seq_len(p))))
  obs[, 1L] <- 0
  list(hat = hat, obs = obs, T_obs = T_obs, p = p)
}

test_that("obs_mask=NULL: no change to train_idx (backward compat)", {
  skip_if_not_installed("lightgbm")
  fx <- make_obs_mask_fixture()
  s_with    <- FoRecoML:::.stack_series(fx$hat, fx$obs, obs_mask = NULL)
  s_without <- FoRecoML:::.stack_series(fx$hat, fx$obs)
  expect_identical(s_with$train_idx, s_without$train_idx)
})

test_that("obs_mask=logical matrix: masked rows excluded from train_idx", {
  skip_if_not_installed("lightgbm")
  fx <- make_obs_mask_fixture()
  mask <- fx$obs != 0   # S1 is all FALSE
  s <- FoRecoML:::.stack_series(fx$hat, fx$obs, obs_mask = mask)
  # S1 rows are 1:T_obs in stacked; all should be absent from train_idx
  s1_rows <- seq_len(fx$T_obs)
  expect_true(length(intersect(s$train_idx, s1_rows)) == 0L)
  # S2 and S3 rows should mostly be in train_idx
  s2_rows <- (fx$T_obs + 1L):(2L * fx$T_obs)
  expect_true(length(intersect(s$train_idx, s2_rows)) > 0L)
})

test_that("obs_mask: series_id_levels preserved for masked series", {
  skip_if_not_installed("lightgbm")
  fx <- make_obs_mask_fixture()
  mask <- fx$obs != 0
  s <- FoRecoML:::.stack_series(fx$hat, fx$obs, obs_mask = mask)
  expect_setequal(s$series_id_levels, paste0("S", seq_len(fx$p)))
})

test_that("obs_mask='auto': warning fires; obs==0 rows filtered", {
  skip_if_not_installed("lightgbm")
  fx <- make_obs_mask_fixture()
  expect_warning(
    s <- FoRecoML:::.stack_series(fx$hat, fx$obs, obs_mask = "auto"),
    "obs_mask='auto'"
  )
  # Same filtering as explicit mask
  mask <- fx$obs != 0
  s_explicit <- FoRecoML:::.stack_series(fx$hat, fx$obs, obs_mask = mask)
  expect_identical(s$train_idx, s_explicit$train_idx)
})

test_that("csrml_g with obs_mask: does not error on all-zero series", {
  skip_if_not_installed("lightgbm")
  set.seed(1)
  agg_mat <- t(c(1,1)); dimnames(agg_mat) <- list("A",c("B","C"))
  hat <- matrix(rnorm(150),50,3,dimnames=list(NULL,c("A","B","C")))
  obs <- matrix(c(rep(0,50), rnorm(50)),50,2,dimnames=list(NULL,c("B","C")))
  # Without mask: B is all-zero (degenerate)
  base <- matrix(rnorm(6),2,3,dimnames=list(NULL,c("A","B","C")))
  mask <- obs != 0
  # Should succeed — B filtered from training but still predicted
  r <- csrml_g(base=base, hat=hat, obs=obs, agg_mat=agg_mat,
               obs_mask=mask, seed=1L)
  expect_true(is.matrix(r))
  expect_false(is.null(attr(r, "FoReco")))
})

test_that("obs_mask_valid stored and NA in residuals for masked rows", {
  skip_if_not_installed("lightgbm")
  set.seed(1)
  hat <- matrix(rnorm(350), 50, 7, dimnames=list(NULL,paste0("L",1:7)))
  obs <- matrix(c(rep(0,50), rnorm(50)), 50, 2, dimnames=list(NULL,c("A","B")))
  # A is all-zero
  mask <- obs != 0
  fit <- rml_g("lightgbm", hat=hat, obs=obs, obs_mask=mask, validation_split=0.2, seed=1L)
  expect_false(is.null(fit$obs_mask_valid))
  # compute residuals — A's validation rows should be NA
  resid <- FoRecoML:::compute_rec_residuals(fit)
  expect_true(all(is.na(resid[, "A"])))
  expect_true(all(is.finite(resid[, "B"])))
})
