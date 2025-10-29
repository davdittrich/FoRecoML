# test cross-sectional reconciliation
if (require(testthat)) {
  # m: quarterly temporal aggregation order
  m <- 4
  te_set <- tetools(m)$set

  # te_fh: minimum forecast horizon per temporal aggregate
  te_fh <- m / te_set

  # N_hat: dimension for the lowest-frequency (k = m) training set
  N_hat <- 16

  # bts_mean: mean for the Normal draws used to simulate data
  bts_mean <- 5

  # hat: a training (base forecasts) feautures vector
  hat <- rnorm(sum(te_fh) * N_hat, rep(te_set * bts_mean, N_hat * te_fh))

  # obs: (observed) values for the highest-frequency series (k = 1)
  obs <- rnorm(m * N_hat, bts_mean)

  # h: base forecast horizon at the lowest-frequency series (k = m)
  h <- 2

  # base: base forecasts matrix
  base <- rnorm(sum(te_fh) * h, rep(te_set * bts_mean, h * te_fh))

  test_that("Approach and features", {
    for (i in c("xgboost", "mlr3", "lightgbm", "randomForest")) {
      for (j in c("rtw")) {
        expect_no_error(terml(
          hat = hat,
          obs = obs,
          base = base,
          agg_order = m,
          approach = i,
          features = j
        ))
      }
    }
  })

  test_that("Two step", {
    mdl <- terml_fit(
      hat = hat,
      obs = obs,
      agg_order = m,
      approach = "lightgbm",
      features = "rtw"
    )
    r1 <- terml(
      hat = hat,
      obs = obs,
      base = base,
      agg_order = m,
      approach = "lightgbm",
      features = "rtw"
    )
    mdl2 <- extract_reconciled_ml(r1)

    r2 <- terml(base = base, fit = mdl, agg_order = m)

    r3 <- terml(base = base, fit = mdl2, agg_order = m)

    expect_equal(r1, r2, ignore_attr = TRUE)
    expect_equal(r2, r3, ignore_attr = TRUE)
  })

  test_that("Errors", {
    expect_error(terml_fit(hat = hat, obs = obs))
    expect_error(terml_fit(hat = hat, agg_order = m))
    expect_error(terml_fit(obs = obs, agg_order = m))
    expect_error(terml(hat = hat, obs = obs))
    expect_error(terml(hat = hat, agg_order = m))
    expect_error(terml(obs = obs, agg_order = m))
    mdl <- terml_fit(
      hat = hat,
      obs = obs,
      agg_order = m,
      approach = "lightgbm",
      features = "rtw"
    )
    expect_error(terml(fit = mdl, agg_order = m))
  })
}
