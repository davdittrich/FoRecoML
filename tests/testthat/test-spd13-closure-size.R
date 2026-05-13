test_that("mfh raw hat and mat2hmat expanded hat have identical byte counts", {
  # Sanity: mat2hmat is a reshape, not row-replication. n*h*kt doubles either way.
  skip_on_cran()

  set.seed(99)
  m_ct    <- 4
  te_set_ct <- FoReco::tetools(m_ct)$set
  te_fh_ct  <- m_ct / te_set_ct

  h_hat  <- 8
  n      <- 3

  hat_raw <- rbind(
    rnorm(sum(te_fh_ct) * h_hat),
    rnorm(sum(te_fh_ct) * h_hat),
    rnorm(sum(te_fh_ct) * h_hat)
  )
  # hat_raw is n × (h * kt)
  h_from_ncol <- NCOL(hat_raw) / sum(m_ct / te_set_ct)
  hat_expanded <- FoRecoML:::mat2hmat(hat_raw, h = h_from_ncol, kset = te_set_ct, n = n)
  # hat_expanded is h × (n * kt)

  expect_equal(
    as.numeric(object.size(hat_raw)),
    as.numeric(object.size(hat_expanded)),
    label = "raw and expanded hat byte counts are equal"
  )
})

test_that("per-iter X is smaller than h * total_cols for sparse mfh feature mode", {
  # For sparse mfh (e.g., mfh-bts), each per-iter X has only |global_id_i| columns
  # which is strictly less than total_cols = n * kt when sel_mat is sparse.
  skip_on_cran()

  set.seed(42)
  m_ct    <- 4
  te_set_ct <- FoReco::tetools(m_ct)$set
  te_fh_ct  <- m_ct / te_set_ct
  agg_mat_ct <- t(c(1, 1))
  dimnames(agg_mat_ct) <- list("A", c("B", "C"))

  h_hat_ct  <- 8
  bts_mean_ct <- 5

  n   <- 3   # total series
  nb  <- 2   # bottom-level series

  hat_ct <- rbind(
    rnorm(sum(te_fh_ct) * h_hat_ct, rep(2 * te_set_ct * bts_mean_ct, h_hat_ct * te_fh_ct)),
    rnorm(sum(te_fh_ct) * h_hat_ct, rep(    te_set_ct * bts_mean_ct, h_hat_ct * te_fh_ct)),
    rnorm(sum(te_fh_ct) * h_hat_ct, rep(    te_set_ct * bts_mean_ct, h_hat_ct * te_fh_ct))
  )
  rownames(hat_ct) <- c("A", "B", "C")
  obs_ct <- rbind(
    rnorm(m_ct * h_hat_ct, bts_mean_ct),
    rnorm(m_ct * h_hat_ct, bts_mean_ct)
  )
  rownames(obs_ct) <- c("B", "C")

  kt         <- sum(m_ct / te_set_ct)
  total_cols <- n * kt  # n*kt in post-spd.13 world

  # mfh-bts: sel_mat selects only the kt columns corresponding to bts series.
  # For n=3, nb=2: bts fraction = nb/n = 2/3 → fewer than total_cols columns active.
  id_bts <- c(0, 1, 1)  # A is aggregate, B/C are bottom
  bts_active <- sum(rep(id_bts, each = kt) != 0)
  expect_true(bts_active < total_cols,
              label = "mfh-bts active cols < total_cols confirms sparsity")

  h <- NCOL(hat_ct) / kt
  # Build the global_id that loop_body would compute for mfh-bts
  sel_vec <- as(rep(id_bts, each = kt), "sparseVector")
  global_id <- which(as.numeric(sel_vec) != 0)
  X_sample  <- FoRecoML:::mat2hmat_partial(hat_ct, h = h, kset = te_set_ct, n = n, cols = global_id)

  # Per-iter X dimensions: h rows × |global_id| cols
  expect_equal(NROW(X_sample), h, label = "X rows = h")
  expect_equal(NCOL(X_sample), length(global_id), label = "X cols = |global_id|")
  expect_true(NCOL(X_sample) < total_cols, label = "sparse mfh per-iter X < total_cols")
})
