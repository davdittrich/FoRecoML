# test cross-sectional reconciliation
if (require(testthat)) {
  # agg_mat: simple aggregation matrix, A = B + C
  agg_mat <- t(c(1, 1))
  dimnames(agg_mat) <- list("A", c("B", "C"))

  # N_hat: dimension for the most aggregated training set
  N_hat <- 100

  # ts_mean: mean for the Normal draws used to simulate data
  ts_mean <- c(20, 10, 10)

  # hat: a training (base forecasts) feautures matrix
  hat <- matrix(
    rnorm(length(ts_mean) * N_hat, mean = ts_mean),
    N_hat,
    byrow = TRUE
  )
  colnames(hat) <- unlist(dimnames(agg_mat))

  # obs: (observed) values for bottom-level series (B, C)
  obs <- matrix(
    rnorm(length(ts_mean[-1]) * N_hat, mean = ts_mean[-1]),
    N_hat,
    byrow = TRUE
  )
  colnames(obs) <- colnames(agg_mat)

  # h: base forecast horizon
  h <- 2

  # base: base forecasts matrix
  base <- matrix(
    rnorm(length(ts_mean) * h, mean = ts_mean),
    h,
    byrow = TRUE
  )
  colnames(base) <- unlist(dimnames(agg_mat))

  test_that("Approach and features", {
    for (i in c("xgboost", "mlr3", "lightgbm", "randomForest")) {
      for (j in c("all", "bts", "str", "str-bts")) {
        expect_no_error(csrml(
          hat = hat,
          obs = obs,
          base = base,
          agg_mat = agg_mat,
          approach = i,
          seed = 123,
          features = j
        ))
      }
    }
  })

  test_that("Two step", {
    mdl <- csrml(
      hat = hat,
      obs = obs,
      agg_mat = agg_mat,
      approach = "lightgbm",
      seed = 123,
      features = "all"
    )
    r1 <- csrml(
      hat = hat,
      obs = obs,
      base = base,
      agg_mat = agg_mat,
      approach = "lightgbm",
      seed = 123,
      features = "all"
    )
    mdl2 <- extract_reconciled_ml(r1)

    r2 <- csrml(base = base, fit = mdl, agg_mat = agg_mat)

    r3 <- csrml(base = base, fit = mdl2, agg_mat = agg_mat)

    expect_equal(r1, r2, ignore_attr = TRUE)
    expect_equal(r2, r3, ignore_attr = TRUE)
  })

  test_that("Errors", {
    expect_error(csrml(hat = hat, obs = obs))
    expect_error(csrml(hat = hat, agg_mat = agg_mat))
    expect_error(csrml(obs = obs, agg_mat = agg_mat))
    mdl <- csrml(
      hat = hat,
      obs = obs,
      agg_mat = agg_mat,
      approach = "lightgbm",
      seed = 123,
      features = "all"
    )
    expect_error(csrml(fit = mdl, agg_order = m))
  })
}
