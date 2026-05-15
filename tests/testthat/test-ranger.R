# T5 — ranger backend tests.
# Fixture mirrors test-csrml.R so we exercise the same code paths the existing
# tests cover, but with approach="ranger" (the new default) and explicit
# fallback paths (deprecation, install hint, overflow).

if (require(testthat)) {
  make_small_fixture <- function() {
    set.seed(42)
    agg_mat <- t(c(1, 1))
    dimnames(agg_mat) <- list("A", c("B", "C"))
    N_hat <- 60
    ts_mean <- c(20, 10, 10)
    hat <- matrix(
      rnorm(length(ts_mean) * N_hat, mean = ts_mean),
      N_hat,
      byrow = TRUE
    )
    obs <- matrix(
      rnorm(length(ts_mean[-1]) * N_hat, mean = ts_mean[-1]),
      N_hat,
      byrow = TRUE
    )
    base <- matrix(
      rnorm(length(ts_mean) * 2, mean = ts_mean),
      2,
      byrow = TRUE
    )
    list(agg_mat = agg_mat, hat = hat, obs = obs, base = base)
  }

  test_that("(a) ranger smoke: csrml runs end-to-end with ranger default", {
    skip_if_not_installed("ranger")
    fx <- make_small_fixture()
    # No `approach` arg -> uses new default "ranger".
    r <- csrml(
      base = fx$base, hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
      features = "all"
    )
    expect_true(is.matrix(r))
    expect_equal(NCOL(r), 3L) # n_total = 1 agg + 2 bottom
    expect_true(all(is.finite(r)))
  })

  test_that("(a2) ranger ctrml + terml smoke", {
    skip_if_not_installed("ranger")
    m <- 4
    te_set <- FoReco::tetools(m)$set
    te_fh <- m / te_set
    N_hat <- 24
    hat_te <- rnorm(sum(te_fh) * N_hat, rep(te_set * 5, N_hat * te_fh))
    obs_te <- rnorm(m * N_hat, 5)
    base_te <- rnorm(sum(te_fh), rep(te_set * 5, te_fh))
    r_te <- terml(
      base = base_te, hat = hat_te, obs = obs_te, agg_order = m,
      approach = "ranger"
    )
    expect_true(all(is.finite(as.numeric(r_te))))
  })

  test_that("(b) ranger vs randomForest RMSE within 1.20x", {
    skip_if_not_installed("ranger")
    skip_if_not_installed("randomForest")
    fx <- make_small_fixture()
    set.seed(1)
    r_rf <- suppressWarnings(csrml(
      base = fx$base, hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
      approach = "randomForest", features = "all"
    ))
    set.seed(1)
    r_rg <- csrml(
      base = fx$base, hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
      approach = "ranger", features = "all"
    )
    rmse <- function(x) sqrt(mean(x^2))
    # Both should produce finite output of the same shape.
    expect_identical(dim(r_rf), dim(r_rg))
    # Relative RMSE on the predictions: ranger should be within 1.2x of rf.
    ratio <- rmse(r_rg - r_rf) / max(rmse(r_rf), 1e-12)
    expect_lt(ratio, 1.20)
  })

  test_that("(c) ranger checkpoint round-trip", {
    skip_if_not_installed("ranger")
    skip_if_not_installed("qs2")
    fx <- make_small_fixture()
    set.seed(7)
    r_mem <- csrml(
      base = fx$base, hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
      approach = "ranger", features = "all", checkpoint = FALSE
    )
    set.seed(7)
    r_ckpt <- csrml(
      base = fx$base, hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
      approach = "ranger", features = "all", checkpoint = TRUE
    )
    expect_equal(as.numeric(r_mem), as.numeric(r_ckpt), tolerance = 1e-10)
  })

  test_that("(e) overflow: estimate_peak_bytes returns finite for large dims", {
    # The as.numeric() casts inside estimate_peak_bytes() prevent integer
    # overflow when NROW(hat)*NCOL(hat)*p exceeds .Machine$integer.max
    # (~2.147e9). A small matrix combined with a huge p exercises the same
    # multiplication path without requiring large allocation.
    real_hat <- matrix(0, nrow = 100L, ncol = 50L)
    # NROW*NCOL*p = 100*50*5e6 = 2.5e10  >> integer.max
    result <- FoRecoML:::estimate_peak_bytes(real_hat, "ranger", p = 5e6L)
    expect_true(is.finite(result))
    expect_true(result > 0)
    expect_true(result > .Machine$integer.max)  # would overflow without cast

    # NULL hat early-return.
    expect_equal(FoRecoML:::estimate_peak_bytes(NULL, "ranger", p = 100L), 0)

    # All known approaches resolve in the switch.
    for (apr in c("randomForest", "ranger", "mlr3", "xgboost", "lightgbm")) {
      r <- FoRecoML:::estimate_peak_bytes(real_hat, apr, p = 10L)
      expect_true(is.finite(r), info = apr)
      expect_true(r > 0, info = apr)
    }
  })

  test_that("(f) deprecate_soft fires for randomForest", {
    skip_if_not_installed("randomForest")
    fx <- make_small_fixture()
    # testthat already sets lifecycle_verbosity = "warning" by default; we
    # just verify a warning is emitted matching the deprecation message.
    old <- options(lifecycle_verbosity = "warning")
    on.exit(options(old), add = TRUE)
    expect_warning(
      csrml(
        base = fx$base, hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
        approach = "randomForest", features = "all"
      ),
      regexp = "ranger|deprecated|randomForest",
      ignore.case = TRUE
    )
  })

  test_that("(g) predict.rml_fit dispatches correctly for cs framework", {
    skip_if_not_installed("ranger")
    fx <- make_small_fixture()
    set.seed(11)
    mdl <- csrml_fit(
      hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
      approach = "ranger", features = "all"
    )
    expect_s3_class(mdl, "rml_fit")
    r_predict <- predict(mdl, newdata = fx$base)
    r_direct <- csrml(base = fx$base, fit = mdl, agg_mat = fx$agg_mat)
    expect_equal(as.numeric(r_predict), as.numeric(r_direct), tolerance = 1e-10)
  })

  test_that("(g2) predict.rml_fit errors on missing newdata", {
    skip_if_not_installed("ranger")
    fx <- make_small_fixture()
    mdl <- csrml_fit(
      hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
      approach = "ranger", features = "all"
    )
    expect_error(predict(mdl), "newdata")
  })
}
