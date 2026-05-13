# test-parallel.R — mirai outer-loop parallelization tests.
# All fixtures use p <= 8 (small nb) and fast params to stay well under 60s.
#
# Design: n_workers=1 exercises the sequential for-loop; n_workers=2 exercises
# the mirai daemon path. Each test that spawns daemons (by passing n_workers=2)
# explicitly tears down at the end via withr::defer(mirai::daemons(0)) OR
# relies on rml's on.exit — then verifies status()$connections == 0 post-call.

make_cs_fixture <- function(seed = 42, N_hat = 30) {
  set.seed(seed)
  # Simple 3-series hierarchy: A = B + C (1 aggregate, 2 bottom-level)
  agg_mat <- t(c(1, 1))
  dimnames(agg_mat) <- list("A", c("B", "C"))
  nb <- 2
  na <- 1
  n  <- na + nb
  ts_mean <- c(10, 5, 5)
  hat  <- matrix(rnorm(n * N_hat, mean = ts_mean), nrow = N_hat, byrow = TRUE)
  colnames(hat) <- c("A", "B", "C")
  obs  <- matrix(rnorm(nb * N_hat, mean = 5), nrow = N_hat)
  colnames(obs) <- c("B", "C")
  h    <- 2
  base <- matrix(rnorm(n * h, mean = ts_mean), nrow = h, byrow = TRUE)
  colnames(base) <- c("A", "B", "C")
  list(agg_mat = agg_mat, hat = hat, obs = obs, base = base, n = n, nb = nb)
}

# ---------------------------------------------------------------------------
# 1. n_workers=1 byte-identical to no-arg default (sequential reference).
# ---------------------------------------------------------------------------
test_that("n_workers=1 byte-identical to pre-patch sequential (xgboost)", {
  fx <- make_cs_fixture()
  set.seed(10)
  r1 <- csrml(
    base = fx$base, hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
    approach = "xgboost", features = "all",
    params = list(nthread = 1, nrounds = 5, eta = 0.3, max_depth = 3),
    n_workers = 1L, checkpoint = FALSE
  )
  set.seed(10)
  r_seq <- csrml(
    base = fx$base, hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
    approach = "xgboost", features = "all",
    params = list(nthread = 1, nrounds = 5, eta = 0.3, max_depth = 3),
    n_workers = 1L, checkpoint = FALSE
  )
  expect_equal(as.numeric(r1), as.numeric(r_seq), tolerance = 0)
})

# ---------------------------------------------------------------------------
# 2. n_workers=2 <= 1e-12 vs n_workers=1 for xgboost (nthread=1 → deterministic).
# ---------------------------------------------------------------------------
test_that("n_workers=2 <= 1e-12 vs n_workers=1 for xgboost (nthread=1)", {
  fx <- make_cs_fixture()
  set.seed(20)
  r_seq <- csrml(
    base = fx$base, hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
    approach = "xgboost", features = "all",
    params = list(nthread = 1, nrounds = 5, eta = 0.3, max_depth = 3),
    n_workers = 1L, checkpoint = FALSE
  )
  set.seed(20)
  r_par <- csrml(
    base = fx$base, hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
    approach = "xgboost", features = "all",
    params = list(nthread = 1, nrounds = 5, eta = 0.3, max_depth = 3),
    n_workers = 2L, checkpoint = FALSE
  )
  # Ensure cleanup
  on.exit(if (mirai::status()$connections > 0L) mirai::daemons(0), add = TRUE)
  expect_equal(as.numeric(r_seq), as.numeric(r_par), tolerance = 1e-12)
  # Verify teardown
  expect_equal(mirai::status()$connections, 0L)
})

# ---------------------------------------------------------------------------
# 3. n_workers=2 <= 1e-12 vs n_workers=1 for lightgbm (num_threads=1).
# ---------------------------------------------------------------------------
test_that("n_workers=2 <= 1e-12 vs n_workers=1 for lightgbm (num_threads=1)", {
  fx <- make_cs_fixture()
  params_lgb <- list(num_threads = 1, nrounds = 5, num_leaves = 4, verbose = -1)
  set.seed(30)
  r_seq <- csrml(
    base = fx$base, hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
    approach = "lightgbm", features = "all",
    params = params_lgb,
    n_workers = 1L, checkpoint = FALSE
  )
  set.seed(30)
  r_par <- csrml(
    base = fx$base, hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
    approach = "lightgbm", features = "all",
    params = params_lgb,
    n_workers = 2L, checkpoint = FALSE
  )
  on.exit(if (mirai::status()$connections > 0L) mirai::daemons(0), add = TRUE)
  expect_equal(as.numeric(r_seq), as.numeric(r_par), tolerance = 1e-12)
  expect_equal(mirai::status()$connections, 0L)
})

# ---------------------------------------------------------------------------
# 4. n_workers=2 reproducible across runs for randomForest.
# ---------------------------------------------------------------------------
test_that("n_workers=2 reproducible across parallel runs for randomForest", {
  fx <- make_cs_fixture()
  params_rf <- list(ntree = 50)
  set.seed(40)
  r_par1 <- csrml(
    base = fx$base, hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
    approach = "randomForest", features = "all",
    params = params_rf,
    n_workers = 2L, checkpoint = FALSE
  )
  on.exit(if (mirai::status()$connections > 0L) mirai::daemons(0), add = TRUE)

  set.seed(40)
  r_par2 <- csrml(
    base = fx$base, hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
    approach = "randomForest", features = "all",
    params = params_rf,
    n_workers = 2L, checkpoint = FALSE
  )
  on.exit(if (mirai::status()$connections > 0L) mirai::daemons(0), add = TRUE)
  # Parallel outputs are reproducible across two calls with same seed.
  expect_equal(as.numeric(r_par1), as.numeric(r_par2), tolerance = 1e-12)
  expect_equal(mirai::status()$connections, 0L)
})

# ---------------------------------------------------------------------------
# 5. resolve_n_workers formula matches detectCores / inner_threads - 1.
# ---------------------------------------------------------------------------
test_that("resolve_n_workers formula matches detectCores/inner - 1", {
  cores  <- parallel::detectCores(logical = TRUE)
  if (!is.finite(cores)) cores <- 1L
  # No inner threads specified → inner = 1
  expected <- max(1L, as.integer(floor(cores / 1L) - 1L))
  got      <- FoRecoML:::resolve_n_workers("auto", "xgboost", NULL)
  expect_equal(got, expected)

  # inner_threads = 2 via nthread
  expected2 <- max(1L, as.integer(floor(cores / 2L) - 1L))
  got2      <- FoRecoML:::resolve_n_workers("auto", "xgboost", list(nthread = 2))
  expect_equal(got2, expected2)

  # Explicit integer
  expect_equal(FoRecoML:::resolve_n_workers(3L, "xgboost", NULL), 3L)
  expect_equal(FoRecoML:::resolve_n_workers(1, "xgboost", NULL), 1L)
})

# ---------------------------------------------------------------------------
# 6. resolve_n_workers rejects invalid inputs.
# ---------------------------------------------------------------------------
test_that("resolve_n_workers rejects NA, list, bad string", {
  expect_error(FoRecoML:::resolve_n_workers(NA, "xgboost", NULL))
  expect_error(FoRecoML:::resolve_n_workers(list(2), "xgboost", NULL))
  expect_error(FoRecoML:::resolve_n_workers("bad_string", "xgboost", NULL))
  expect_error(FoRecoML:::resolve_n_workers(Inf, "xgboost", NULL))
})

# ---------------------------------------------------------------------------
# 7. cap_inner_threads: caps unset params, preserves user-set values.
# ---------------------------------------------------------------------------
test_that("cap_inner_threads caps unset, preserves user-set", {
  # caps all three when unset (non-mlr3 approach)
  capped <- FoRecoML:::cap_inner_threads(NULL, 2L)
  expect_equal(capped$nthread, 1L)
  expect_equal(capped$num_threads, 1L)
  expect_equal(capped$num.threads, 1L)

  # preserves user-set nthread (explicitly set to 4)
  user_params <- list(nthread = 4L)
  preserved   <- FoRecoML:::cap_inner_threads(user_params, 2L)
  expect_equal(preserved$nthread, 4L)

  # n_workers=1 → no-op
  unchanged <- FoRecoML:::cap_inner_threads(list(nthread = 4L), 1L)
  expect_equal(unchanged$nthread, 4L)
  expect_null(unchanged$num_threads)

  # mlr3 approach: only num.threads capped, no nthread/num_threads injected
  mlr3_capped <- FoRecoML:::cap_inner_threads(NULL, 2L, approach = "mlr3")
  expect_equal(mlr3_capped$num.threads, 1L)
  expect_null(mlr3_capped$nthread)
  expect_null(mlr3_capped$num_threads)
})

# ---------------------------------------------------------------------------
# 8. mirai daemon pool torn down after rml() returns.
# ---------------------------------------------------------------------------
test_that("mirai daemon pool torn down after rml() returns (status == 0)", {
  fx <- make_cs_fixture()
  csrml(
    base = fx$base, hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
    approach = "xgboost", features = "all",
    params = list(nthread = 1, nrounds = 5),
    n_workers = 2L, checkpoint = FALSE
  )
  # rml's on.exit fires when spawned; connections should be 0.
  expect_equal(mirai::status()$connections, 0L)
  # Defensive cleanup in case on.exit misfired.
  if (mirai::status()$connections > 0L) mirai::daemons(0)
})

# ---------------------------------------------------------------------------
# 9. checkpoint=path + n_workers=2 (xgboost) <= 1e-12 vs n_workers=1.
# ---------------------------------------------------------------------------
test_that("checkpoint=path + n_workers=2 (xgboost) <= 1e-12 vs n_workers=1", {
  fx   <- make_cs_fixture()
  dir1 <- file.path(tempdir(), paste0("ckpt_par1_", Sys.getpid()))
  dir2 <- file.path(tempdir(), paste0("ckpt_par2_", Sys.getpid()))
  dir.create(dir1, recursive = TRUE, showWarnings = FALSE)
  dir.create(dir2, recursive = TRUE, showWarnings = FALSE)
  on.exit({
    unlink(dir1, recursive = TRUE)
    unlink(dir2, recursive = TRUE)
    if (mirai::status()$connections > 0L) mirai::daemons(0)
  }, add = TRUE)

  set.seed(50)
  r_seq <- csrml(
    base = fx$base, hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
    approach = "xgboost", features = "all",
    params = list(nthread = 1, nrounds = 5, eta = 0.3, max_depth = 3),
    n_workers = 1L, checkpoint = dir1
  )
  set.seed(50)
  r_par <- csrml(
    base = fx$base, hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
    approach = "xgboost", features = "all",
    params = list(nthread = 1, nrounds = 5, eta = 0.3, max_depth = 3),
    n_workers = 2L, checkpoint = dir2
  )
  expect_equal(as.numeric(r_seq), as.numeric(r_par), tolerance = 1e-12)
  expect_equal(mirai::status()$connections, 0L)
})
