# Extracted from test-rml-g-wrappers.R:43

# setup ------------------------------------------------------------------------
library(testthat)
test_env <- simulate_test_env(package = "FoRecoML", path = "..")
attach(test_env, warn.conflicts = FALSE)

# prequel ----------------------------------------------------------------------
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

# test -------------------------------------------------------------------------
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
