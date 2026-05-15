# T7.4 — chunked incremental training tests (G.2, reduced scope: 4 tests).

make_chunk_fixture <- function(p = 10L, T_obs = 20L, ncol_hat = 5L) {
  set.seed(7)
  obs <- matrix(rnorm(T_obs * p), T_obs, p,
                dimnames = list(NULL, paste0("S", seq_len(p))))
  hat <- matrix(rnorm(T_obs * ncol_hat), T_obs, ncol_hat,
                dimnames = list(NULL, paste0("f", seq_len(ncol_hat))))
  agg_mat <- matrix(1, 1, p)
  rownames(agg_mat) <- "G1"
  base <- matrix(rnorm(p + 1), p + 1, 1)
  rownames(base) <- c("G1", paste0("S", seq_len(p)))
  list(obs = obs, hat = hat, agg_mat = agg_mat, base = base, p = p)
}

test_that("(a) lightgbm single-batch path matches non-chunked rml_g_fit class", {
  skip_if_not_installed("lightgbm")
  fx <- make_chunk_fixture(p = 6L)
  r1 <- csrml_g(base = fx$base, hat = fx$hat, obs = fx$obs,
                agg_mat = fx$agg_mat[, 1:6, drop = FALSE],
                approach = "lightgbm", seed = 42L)
  r2 <- csrml_g(base = fx$base, hat = fx$hat, obs = fx$obs,
                agg_mat = fx$agg_mat[, 1:6, drop = FALSE],
                approach = "lightgbm", seed = 42L,
                batch_size = fx$p)  # single batch → delegates to rml_g
  expect_s3_class(r1, "rml_g_fit")
  expect_s3_class(r2, "rml_g_fit")
})

test_that("(b) .auto_batch_size resolves to integer in [1, p]", {
  fx <- make_chunk_fixture()
  bs <- FoRecoML:::.auto_batch_size(
    T_obs = nrow(fx$obs), ncol_hat = ncol(fx$hat),
    approach = "lightgbm", p = fx$p
  )
  expect_true(is.integer(bs))
  expect_true(bs >= 1L)
  expect_true(bs <= fx$p)
})

test_that("(c) catboost chunked aborts with helpful message", {
  fx <- make_chunk_fixture(p = 6L)
  expect_error(
    csrml_g(base = fx$base, hat = fx$hat, obs = fx$obs,
            agg_mat = fx$agg_mat[, 1:6, drop = FALSE],
            approach = "catboost", batch_size = 2L),
    regexp = "catboost",
    ignore.case = TRUE
  )
})

test_that("(d) chunk_strategy='sequential' batch_indices reproducible", {
  skip_if_not_installed("lightgbm")
  fx <- make_chunk_fixture(p = 6L)
  agg6 <- fx$agg_mat[, 1:6, drop = FALSE]
  r1 <- csrml_g(base = fx$base, hat = fx$hat, obs = fx$obs, agg_mat = agg6,
                approach = "lightgbm", seed = 1L, batch_size = 3L,
                chunk_strategy = "sequential")
  r2 <- csrml_g(base = fx$base, hat = fx$hat, obs = fx$obs, agg_mat = agg6,
                approach = "lightgbm", seed = 1L, batch_size = 3L,
                chunk_strategy = "sequential")
  expect_equal(r1$batch_indices, r2$batch_indices)
})
