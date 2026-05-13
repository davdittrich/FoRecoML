test_that("spd.13 mfh ctrml produces correct results (numerical equivalence to reference)", {
  # Reference: compute mfh result by manually applying mat2hmat in wrapper,
  # then passing the materialized hat/base to rml() WITHOUT kset (T4 path).
  # This replicates the correct mfh behavior that spd.13 preserves via
  # mat2hmat_partial in loop_body.
  #
  # Note: pre-spd.13 code (post-spd.12) was broken for mfh because loop_body
  # called input2rtw_partial on the already-materialized hat. The spd.13 baseline
  # fixtures contain subscriptOutOfBoundsError — do NOT compare against them.

  set.seed(42)
  m_ct    <- 4
  te_set_ct <- FoReco::tetools(m_ct)$set
  te_fh_ct  <- m_ct / te_set_ct
  agg_mat_ct <- t(c(1, 1))
  dimnames(agg_mat_ct) <- list("A", c("B", "C"))
  h_hat_ct  <- 16
  bts_mean_ct <- 5

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
  h_ct   <- 2
  base_ct <- rbind(
    rnorm(sum(te_fh_ct) * h_ct, rep(2 * te_set_ct * bts_mean_ct, h_ct * te_fh_ct)),
    rnorm(sum(te_fh_ct) * h_ct, rep(    te_set_ct * bts_mean_ct, h_ct * te_fh_ct)),
    rnorm(sum(te_fh_ct) * h_ct, rep(    te_set_ct * bts_mean_ct, h_ct * te_fh_ct))
  )
  rownames(base_ct) <- c("A", "B", "C")

  n  <- 3L  # total series (1 agg + 2 bottom)
  nb <- 2L  # bottom series
  kt <- sum(m_ct / te_set_ct)  # temporal aggregate count

  # ── mfh-all with 3 learners (basic smoke test: no error, correct dimensions) ──
  for (appr in c("randomForest", "xgboost", "lightgbm")) {
    set.seed(1001)
    result <- expect_no_error(ctrml(
      hat = hat_ct, obs = obs_ct, base = base_ct,
      agg_order = m_ct, agg_mat = agg_mat_ct,
      approach = appr, features = "mfh-all"
    ))
    expect_true(is.matrix(result))
    # ctbu returns n × h*kt matrix (n series, h*kt temporal cols)
    expect_equal(NROW(result), n)
    expect_equal(NCOL(result), h_ct * kt)
  }

  # ── mfh-bts with xgboost ─────────────────────────────────────────────────────
  set.seed(3002)
  result_bts <- expect_no_error(ctrml(
    hat = hat_ct, obs = obs_ct, base = base_ct,
    agg_order = m_ct, agg_mat = agg_mat_ct,
    approach = "xgboost", features = "mfh-bts"
  ))
  expect_true(is.matrix(result_bts))

  # ── mfh-bts with lightgbm ─────────────────────────────────────────────────────
  # Note: mfh-str-bts and mfh-str cases reference sel_mat before initialization;
  # these are pre-existing bugs outside spd.13 scope. Test mfh-bts instead.
  set.seed(3003)
  expect_no_error(ctrml(
    hat = hat_ct, obs = obs_ct, base = base_ct,
    agg_order = m_ct, agg_mat = agg_mat_ct,
    approach = "lightgbm", features = "mfh-bts"
  ))

  # ── NA-injection: mfh-all with NA in hat col 1 ───────────────────────────────
  hat_na <- hat_ct
  hat_na[, 1] <- NA

  set.seed(4001)
  result_na <- expect_no_error(ctrml(
    hat = hat_na, obs = obs_ct, base = base_ct,
    agg_order = m_ct, agg_mat = agg_mat_ct,
    approach = "lightgbm", features = "mfh-all"
  ))
  expect_true(is.matrix(result_na))
  expect_false(anyNA(result_na))

  # ── Predict-reuse: train then predict ────────────────────────────────────────
  set.seed(4002)
  mdl <- expect_no_error(ctrml_fit(
    hat = hat_na, obs = obs_ct,
    agg_order = m_ct, agg_mat = agg_mat_ct,
    approach = "lightgbm", features = "mfh-all"
  ))
  expect_s3_class(mdl, "rml_fit")

  result_reuse <- expect_no_error(
    ctrml(base = base_ct, fit = mdl, agg_order = m_ct, agg_mat = agg_mat_ct)
  )
  expect_true(is.matrix(result_reuse))

  # ── mat2hmat_partial byte-equivalence: spd.13 matches explicit expansion ─────
  # Reference: materialize hat/base explicitly via mat2hmat, pass to rml()
  # without kset (T4 path, hat already in h × (n*kt) form). This is equivalent
  # to the correct pre-spd.12 mfh behavior.
  h_hat   <- NCOL(hat_ct) / kt
  h_base  <- NCOL(base_ct) / kt

  hat_expanded  <- FoRecoML:::mat2hmat(hat_ct,  h = h_hat,  kset = te_set_ct, n = n)
  base_expanded <- FoRecoML:::mat2hmat(base_ct, h = h_base, kset = te_set_ct, n = n)
  obs_mfh <- matrix(as.vector(t(obs_ct)), ncol = m_ct * nb)

  keep_cols_ref <- seq_len(n * kt)

  # Use n_workers = 1 for deterministic comparison (parallel would advance RNG).
  set.seed(1001)
  result_spd13 <- ctrml(
    hat = hat_ct, obs = obs_ct, base = base_ct,
    agg_order = m_ct, agg_mat = agg_mat_ct,
    approach = "randomForest", features = "mfh-all",
    n_workers = 1L
  )

  set.seed(1001)
  ref_rml <- FoRecoML:::rml(
    approach   = "randomForest",
    base       = base_expanded,
    hat        = hat_expanded,
    obs        = obs_mfh,
    sel_mat    = 1,
    keep_cols  = keep_cols_ref,
    kset       = NULL,   # no kset: T4 path (hat already expanded)
    h          = NULL,   # no h: not mfh partial expansion
    checkpoint = FALSE,
    n_workers  = 1L
  )

  # ref_rml is h_base × (m*nb); rebuild the same ctbu_base as ctrml does internally.
  h_base_ref <- NROW(ref_rml)
  ctbu_base_ref <- matrix(NA_real_, nrow = nb, ncol = h_base_ref * m_ct)
  for (s in seq_len(nb)) {
    for (lv in seq_len(m_ct)) {
      obs_col   <- (s - 1L) * m_ct + lv
      ctbu_cols <- (lv - 1L) * h_base_ref + seq_len(h_base_ref)
      ctbu_base_ref[s, ctbu_cols] <- ref_rml[, obs_col]
    }
  }
  ref_mat <- FoReco::ctbu(
    ctbu_base_ref,
    agg_order = m_ct,
    agg_mat   = agg_mat_ct
  )

  expect_equal(
    max(abs(as.vector(result_spd13) - as.vector(ref_mat))), 0,
    label = "spd.13 mfh-all output matches explicit mat2hmat reference"
  )
})

test_that("spd.13 mfh ctrml predict-reuse with different horizon completes", {
  # For mfh, h_train stored in ctrml_fit is NULL (h from hat = N training obs,
  # not the forecast horizon). ctrml predict-reuse with a different h should
  # succeed without error.
  set.seed(42)
  m_ct    <- 4
  te_set_ct <- FoReco::tetools(m_ct)$set
  te_fh_ct  <- m_ct / te_set_ct
  agg_mat_ct <- t(c(1, 1))
  dimnames(agg_mat_ct) <- list("A", c("B", "C"))
  h_hat_ct  <- 16
  bts_mean_ct <- 5

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

  set.seed(4003)
  mdl <- ctrml_fit(
    hat = hat_ct, obs = obs_ct,
    agg_order = m_ct, agg_mat = agg_mat_ct,
    approach = "lightgbm", features = "mfh-all"
  )
  # h_train must be NULL for mfh: h from hat is N (obs count), not forecast horizon.
  expect_null(mdl$h_train)

  # h=1 forecast horizon; model trained on h_hat=16 observation blocks.
  # No h_train guard for mfh → should succeed.
  base_diff_h <- rbind(
    rnorm(sum(te_fh_ct) * 1, rep(2 * te_set_ct * bts_mean_ct, 1 * te_fh_ct)),
    rnorm(sum(te_fh_ct) * 1, rep(    te_set_ct * bts_mean_ct, 1 * te_fh_ct)),
    rnorm(sum(te_fh_ct) * 1, rep(    te_set_ct * bts_mean_ct, 1 * te_fh_ct))
  )
  rownames(base_diff_h) <- c("A", "B", "C")

  expect_no_error(
    ctrml(base = base_diff_h, fit = mdl, agg_order = m_ct, agg_mat = agg_mat_ct)
  )
})
