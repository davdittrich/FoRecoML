# test cross-sectional reconciliation
if (require(testthat)) {
  # m: quarterly temporal aggregation order
  m <- 4
  te_set <- tetools(m)$set

  # agg_mat: simple aggregation matrix, A = B + C
  agg_mat <- t(c(1, 1))
  dimnames(agg_mat) <- list("A", c("B", "C"))

  # te_fh: ,inimum forecast horizon per temporal aggregate
  te_fh <- m / te_set

  # h_hat: number of the most aggregate ts values to train the ML approach
  h_hat <- 16

  # bts_mean: base mean for the Normal draws used to simulate data
  bts_mean <- 5

  # hat: a (3 x 112) matrix to train the ML approach
  hat <- rbind(
    rnorm(sum(te_fh) * h_hat, rep(2 * te_set * bts_mean, h_hat * te_fh)), # Series A
    rnorm(sum(te_fh) * h_hat, rep(te_set * bts_mean, h_hat * te_fh)), # Series B
    rnorm(sum(te_fh) * h_hat, rep(te_set * bts_mean, h_hat * te_fh)) # Series C
  )
  rownames(hat) <- c("A", "B", "C")

  # obs: (observed) values for the high-frequency bottom-level series
  # (B and C with k = 1)
  obs <- rbind(
    rnorm(m * h_hat, bts_mean), # Observed for series B
    rnorm(m * h_hat, bts_mean) # Observed for series C
  )
  rownames(obs) <- c("B", "C")

  # h: base forecast horizon (e.g., short-term) at the most aggregated series
  h <- 2

  # base: base forecasts matrix
  base <- rbind(
    rnorm(sum(te_fh) * h, rep(2 * te_set * bts_mean, h * te_fh)), # Base for A
    rnorm(sum(te_fh) * h, rep(te_set * bts_mean, h * te_fh)), # Base for B
    rnorm(sum(te_fh) * h, rep(te_set * bts_mean, h * te_fh)) # Base for C
  )

  test_that("Approach and features", {
    for (i in c("xgboost", "mlr3", "lightgbm", "randomForest")) {
      for (j in c(
        "all",
        "compact"
      )) {
        expect_no_error(ctrml(
          hat = hat,
          obs = obs,
          base = base,
          agg_order = m,
          agg_mat = agg_mat,
          approach = i,
          features = j
        ))
      }
    }
  })

  test_that("Two step", {
    mdl <- ctrml_fit(
      hat = hat,
      obs = obs,
      agg_order = m,
      agg_mat = agg_mat,
      approach = "lightgbm",
      features = "all"
    )
    r1 <- ctrml(
      hat = hat,
      obs = obs,
      base = base,
      agg_order = m,
      agg_mat = agg_mat,
      approach = "lightgbm",
      features = "all"
    )
    mdl2 <- extract_reconciled_ml(r1)

    r2 <- ctrml(base = base, fit = mdl, agg_order = m, agg_mat = agg_mat)

    r3 <- ctrml(base = base, fit = mdl2, agg_order = m, agg_mat = agg_mat)

    expect_equal(r1, r2, ignore_attr = TRUE)
    expect_equal(r2, r3, ignore_attr = TRUE)
  })

  test_that("Errors", {
    expect_error(ctrml_fit(hat = hat, obs = obs, agg_order = m))
    expect_error(ctrml_fit(hat = hat, obs = obs, agg_mat = agg_mat))
    expect_error(ctrml_fit(hat = hat, agg_order = m, agg_mat = agg_mat))
    expect_error(ctrml_fit(obs = obs, agg_order = m, agg_mat = agg_mat))
    expect_error(ctrml(hat = hat, obs = obs, agg_order = m))
    expect_error(ctrml(hat = hat, obs = obs, agg_mat = agg_mat))
    expect_error(ctrml(hat = hat, agg_order = m, agg_mat = agg_mat))
    expect_error(ctrml(obs = obs, agg_order = m, agg_mat = agg_mat))
    mdl <- ctrml_fit(
      hat = hat,
      obs = obs,
      agg_order = m,
      agg_mat = agg_mat,
      approach = "lightgbm",
      features = "all"
    )
    expect_error(ctrml(fit = mdl, agg_order = m, agg_mat = agg_mat))
  })
}
