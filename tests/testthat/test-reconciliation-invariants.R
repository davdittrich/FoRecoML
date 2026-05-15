# T8 — Reconciliation invariant test suite
# Post-reconciliation forecasts must satisfy hierarchical constraints.
# Tolerance: 1e-10 (max absolute deviation).
#
# API facts (verified from source):
#   csrml() returns an h x n matrix (not a list$recf).
#   base: h x n (horizon rows, series cols).
#   hat:  N x n (training rows, all series cols — col names required).
#   obs:  N x nb (training rows, bottom-level series cols — col names required).
#   terml() returns a numeric vector of length h * sum(m / kset).

# Shared fixture builder for cross-sectional tests.
# p: number of bottom-level series; T_obs: training length; h: forecast horizon.
make_cs_fixture <- function(p = 4L, T_obs = 30L, h = 1L, seed = 13L) {
  set.seed(seed)
  # Simple 2-level hierarchy: 1 total + p bottom series.
  n <- p + 1L
  series_names <- c("Total", paste0("S", seq_len(p)))
  agg_mat <- matrix(1, nrow = 1L, ncol = p,
                    dimnames = list("Total", paste0("S", seq_len(p))))

  # hat: N x n (all series as features).
  hat <- matrix(
    rnorm(T_obs * n, mean = rep(c(p * 100, rep(100, p)), each = T_obs)),
    T_obs, n,
    dimnames = list(NULL, series_names)
  )

  # obs: N x p (bottom-level series only).
  obs <- matrix(
    rnorm(T_obs * p, mean = 100),
    T_obs, p,
    dimnames = list(NULL, paste0("S", seq_len(p)))
  )

  # base: h x n.
  base <- matrix(
    rnorm(h * n, mean = rep(c(p * 100, rep(100, p)), each = h)),
    h, n,
    dimnames = list(NULL, series_names)
  )

  list(agg_mat = agg_mat, obs = obs, hat = hat, base = base, p = p, n = n, h = h)
}

# Check cross-sectional invariant: agg_mat %*% bottom_cols == top_cols for all h.
# recf: h x n matrix; col order: top series first, then bottom series.
check_cs_invariant <- function(recf, agg_mat, tol = 1e-10) {
  n_agg  <- nrow(agg_mat)
  p      <- ncol(agg_mat)
  n      <- n_agg + p
  # columns: first n_agg are aggregates, remaining p are bottom-level.
  top_cols    <- seq_len(n_agg)
  bottom_cols <- (n_agg + 1L):n
  max_dev <- 0
  for (i in seq_len(nrow(recf))) {
    # recf[i, bottom_cols] is a 1×p row; transpose to p×1 for agg_mat (na×p) %*% (p×1).
    bottom_vec   <- as.matrix(recf[i, bottom_cols, drop = FALSE])  # 1 x p
    computed_top <- as.vector(agg_mat %*% t(bottom_vec))           # na × 1 -> vector
    actual_top   <- as.vector(recf[i, top_cols, drop = FALSE])
    dev <- max(abs(computed_top - actual_top))
    if (dev > max_dev) max_dev <- dev
  }
  list(ok = max_dev <= tol, max_dev = max_dev)
}

# --- Cross-sectional (csrml) invariant tests ---

test_that("(a) csrml (ranger): cross-sectional invariant holds", {
  skip_if_not_installed("ranger")
  fx <- make_cs_fixture(p = 4L, seed = 1L)
  result <- csrml(
    base    = fx$base,
    hat     = fx$hat,
    obs     = fx$obs,
    agg_mat = fx$agg_mat,
    approach = "ranger",
    features = "all"
  )
  check <- check_cs_invariant(result, fx$agg_mat)
  expect_true(check$ok,
    info = sprintf("CS invariant violated: max_dev = %.2e", check$max_dev))
})

test_that("(b) csrml (lightgbm): cross-sectional invariant holds", {
  skip_if_not_installed("lightgbm")
  fx <- make_cs_fixture(p = 4L, seed = 2L)
  result <- csrml(
    base     = fx$base,
    hat      = fx$hat,
    obs      = fx$obs,
    agg_mat  = fx$agg_mat,
    approach = "lightgbm",
    features = "all"
  )
  check <- check_cs_invariant(result, fx$agg_mat)
  expect_true(check$ok, info = sprintf("max_dev = %.2e", check$max_dev))
})

test_that("(c) csrml (xgboost): cross-sectional invariant holds", {
  skip_if_not_installed("xgboost")
  fx <- make_cs_fixture(p = 4L, seed = 3L)
  result <- csrml(
    base     = fx$base,
    hat      = fx$hat,
    obs      = fx$obs,
    agg_mat  = fx$agg_mat,
    approach = "xgboost",
    features = "all"
  )
  check <- check_cs_invariant(result, fx$agg_mat)
  expect_true(check$ok, info = sprintf("max_dev = %.2e", check$max_dev))
})

test_that("(d) csrml multi-horizon: CS invariant holds for all h", {
  skip_if_not_installed("ranger")
  fx <- make_cs_fixture(p = 3L, T_obs = 25L, h = 3L, seed = 4L)
  result <- csrml(
    base     = fx$base,
    hat      = fx$hat,
    obs      = fx$obs,
    agg_mat  = fx$agg_mat,
    approach = "ranger",
    features = "all"
  )
  check <- check_cs_invariant(result, fx$agg_mat)
  expect_true(check$ok, info = sprintf("multi-horizon max_dev = %.2e", check$max_dev))
})

test_that("(e) csrml: invariant is at machine precision (well below 1e-10)", {
  skip_if_not_installed("ranger")
  fx <- make_cs_fixture(p = 4L, seed = 5L)
  result <- csrml(
    base     = fx$base,
    hat      = fx$hat,
    obs      = fx$obs,
    agg_mat  = fx$agg_mat,
    approach = "ranger",
    features = "all"
  )
  n_agg  <- nrow(fx$agg_mat)
  p      <- ncol(fx$agg_mat)
  # result is h×n; top cols are first n_agg, bottom cols follow.
  top    <- result[, seq_len(n_agg), drop = FALSE]      # h × n_agg
  bottom <- result[, (n_agg + 1L):(n_agg + p), drop = FALSE]  # h × p
  # For each horizon row: agg_mat (na×p) %*% bottom_row (p×1) = top_row (na×1).
  computed_top <- t(fx$agg_mat %*% t(bottom))           # h × n_agg
  max_dev <- max(abs(computed_top - top))
  expect_lte(max_dev, 1e-10)
})

test_that("(f) csrml sntz=TRUE: all reconciled forecasts non-negative", {
  skip_if_not_installed("ranger")
  # Force strongly negative base to exercise sntz path.
  fx <- make_cs_fixture(p = 4L, seed = 6L)
  fx$base <- -abs(fx$base)
  result <- csrml(
    base     = fx$base,
    hat      = fx$hat,
    obs      = fx$obs,
    agg_mat  = fx$agg_mat,
    approach = "ranger",
    features = "all",
    sntz     = TRUE
  )
  expect_true(all(result >= 0),
    info = "sntz=TRUE must produce non-negative forecasts")
})

test_that("(g) terml (ranger): temporal constraints satisfied at tol 1e-10", {
  skip_if_not_installed("ranger")
  # Quarterly aggregation: kset = {4, 2, 1}, te_fh = {1, 2, 4}
  m <- 4L
  te_set <- tetools(m)$set   # c(4, 2, 1)
  te_fh  <- m / te_set       # c(1, 2, 4)
  N_hat  <- 16L
  bts_mean <- 5
  h <- 1L                    # single horizon step

  set.seed(7L)
  hat  <- rnorm(sum(te_fh) * N_hat, rep(te_set * bts_mean, N_hat * te_fh))
  obs  <- rnorm(m * N_hat, bts_mean)
  base <- rnorm(sum(te_fh) * h, rep(te_set * bts_mean, h * te_fh))

  result <- terml(
    base      = base,
    hat       = hat,
    obs       = obs,
    agg_order = m,
    approach  = "ranger",
    features  = "all"
  )

  # Output layout for h=1, m=4 (kset={4,2,1}, te_fh={1,2,4}):
  #   [1]       = annual value (k=4)
  #   [2:3]     = semi-annual values (k=2)
  #   [4:7]     = quarterly values (k=1)
  annual  <- result[1L]
  semi    <- result[2L:3L]
  quarter <- result[4L:7L]

  # Temporal constraints:
  expect_lte(abs(annual - sum(semi)),    1e-10,
    label = "annual == sum(semi-annual)")
  expect_lte(abs(annual - sum(quarter)), 1e-10,
    label = "annual == sum(quarterly)")
  expect_lte(abs(semi[1L] - (quarter[1L] + quarter[2L])), 1e-10,
    label = "semi[1] == q[1]+q[2]")
  expect_lte(abs(semi[2L] - (quarter[3L] + quarter[4L])), 1e-10,
    label = "semi[2] == q[3]+q[4]")
})

test_that("(h) csrml: invariant robust to varying p (2, 5, 10)", {
  skip_if_not_installed("ranger")
  for (p in c(2L, 5L, 10L)) {
    fx <- make_cs_fixture(p = p, T_obs = 20L, h = 1L, seed = p * 10L)
    result <- csrml(
      base     = fx$base,
      hat      = fx$hat,
      obs      = fx$obs,
      agg_mat  = fx$agg_mat,
      approach = "ranger",
      features = "all"
    )
    check <- check_cs_invariant(result, fx$agg_mat)
    expect_true(check$ok,
      info = sprintf("p=%d: max_dev = %.2e", p, check$max_dev))
  }
})
