# Extracted from test-rml-g-s3.R:46

# setup ------------------------------------------------------------------------
library(testthat)
test_env <- simulate_test_env(package = "FoRecoML", path = "..")
attach(test_env, warn.conflicts = FALSE)

# prequel ----------------------------------------------------------------------
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

# test -------------------------------------------------------------------------
skip_if_not_installed("ranger")
fx  <- make_s3_fixture()
fit <- csrml_g(base = fx$base, hat = fx$hat, obs = fx$obs,
                 agg_mat = fx$agg_mat, approach = "ranger", seed = 1L)
