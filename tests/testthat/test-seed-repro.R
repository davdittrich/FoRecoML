# T5 — seed reproducibility audit per backend.
# Confirms the set.seed(seed) un-comment in rml() and that ranger/randomForest
# both honour upstream set.seed() before the fit loop.

if (require(testthat)) {
  make_fx <- function() {
    set.seed(99)
    agg_mat <- t(c(1, 1))
    dimnames(agg_mat) <- list("A", c("B", "C"))
    N_hat <- 60
    ts_mean <- c(20, 10, 10)
    hat <- matrix(
      rnorm(length(ts_mean) * N_hat, mean = ts_mean),
      N_hat, byrow = TRUE
    )
    obs <- matrix(
      rnorm(length(ts_mean[-1]) * N_hat, mean = ts_mean[-1]),
      N_hat, byrow = TRUE
    )
    base <- matrix(
      rnorm(length(ts_mean) * 2, mean = ts_mean),
      2, byrow = TRUE
    )
    list(agg_mat = agg_mat, hat = hat, obs = obs, base = base)
  }

  test_that("(h) ranger seed reproducibility via set.seed(seed)", {
    skip_if_not_installed("ranger")
    fx <- make_fx()
    set.seed(42)
    r1 <- csrml(
      base = fx$base, hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
      approach = "ranger", features = "all"
    )
    set.seed(42)
    r2 <- csrml(
      base = fx$base, hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
      approach = "ranger", features = "all"
    )
    expect_equal(as.numeric(r1), as.numeric(r2), tolerance = 0)
  })

  test_that("(i) randomForest seed reproducibility via set.seed(seed)", {
    skip_if_not_installed("randomForest")
    fx <- make_fx()
    set.seed(42)
    r1 <- suppressWarnings(csrml(
      base = fx$base, hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
      approach = "randomForest", features = "all"
    ))
    set.seed(42)
    r2 <- suppressWarnings(csrml(
      base = fx$base, hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
      approach = "randomForest", features = "all"
    ))
    expect_equal(as.numeric(r1), as.numeric(r2), tolerance = 0)
  })

  test_that("(j) different seeds produce different ranger predictions", {
    skip_if_not_installed("ranger")
    fx <- make_fx()
    set.seed(42)
    r1 <- csrml(
      base = fx$base, hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
      approach = "ranger", features = "all"
    )
    set.seed(7)
    r2 <- csrml(
      base = fx$base, hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
      approach = "ranger", features = "all"
    )
    # Should be non-identical (stochastic fit).
    expect_false(isTRUE(all.equal(as.numeric(r1), as.numeric(r2))))
  })
}
