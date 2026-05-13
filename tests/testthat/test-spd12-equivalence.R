# test-spd12-equivalence.R — numerical equivalence gate for spd.12
# Verifies that deferring input2rtw_partial from ctrml/terml wrapper into
# loop_body produces bit-identical bts for all non-mfh feature paths.
#
# Snapshots in _snaps/spd12/ were captured on HEAD a90fd30 (pre-spd.12)
# via dev/spd12-baseline.R. Same RNG sequence must be reproduced here.
#
# NOTE: set.seed(42) is called ONCE. Fixture generation order (ctrml, terml,
# csrml) must match dev/spd12-baseline.R exactly — the RNG state flows through.

skip_on_cran()

if (!requireNamespace("qs2", quietly = TRUE)) {
  skip("qs2 not available")
}

snap_dir <- testthat::test_path("fixtures/spd12")
if (!dir.exists(snap_dir)) {
  skip("Snapshots not found — run dev/spd12-baseline.R on pre-spd.12 HEAD first")
}

library(qs2)

# ── Fixtures (order mirrors dev/spd12-baseline.R) ────────────────────────────

set.seed(42)

# ctrml fixtures
m_ct <- 4
te_set_ct <- tetools(m_ct)$set
te_fh_ct  <- m_ct / te_set_ct
agg_mat_ct <- t(c(1, 1))
dimnames(agg_mat_ct) <- list("A", c("B", "C"))
h_hat_ct <- 16L
bts_mean_ct <- 5
hat_ct <- rbind(
  rnorm(sum(te_fh_ct) * h_hat_ct, rep(2 * te_set_ct * bts_mean_ct, h_hat_ct * te_fh_ct)),
  rnorm(sum(te_fh_ct) * h_hat_ct, rep(te_set_ct * bts_mean_ct, h_hat_ct * te_fh_ct)),
  rnorm(sum(te_fh_ct) * h_hat_ct, rep(te_set_ct * bts_mean_ct, h_hat_ct * te_fh_ct))
)
obs_ct <- rbind(
  rnorm(m_ct * h_hat_ct, bts_mean_ct),
  rnorm(m_ct * h_hat_ct, bts_mean_ct)
)
h_ct <- 2L
base_ct <- rbind(
  rnorm(sum(te_fh_ct) * h_ct, rep(2 * te_set_ct * bts_mean_ct, h_ct * te_fh_ct)),
  rnorm(sum(te_fh_ct) * h_ct, rep(te_set_ct * bts_mean_ct, h_ct * te_fh_ct)),
  rnorm(sum(te_fh_ct) * h_ct, rep(te_set_ct * bts_mean_ct, h_ct * te_fh_ct))
)

# terml fixtures (after ctrml consumes RNG)
m_te <- 4
te_set_te <- tetools(m_te)$set
te_fh_te  <- m_te / te_set_te
N_hat_te <- 16L
bts_mean_te <- 5
hat_te <- rnorm(sum(te_fh_te) * N_hat_te, rep(te_set_te * bts_mean_te, N_hat_te * te_fh_te))
obs_te <- rnorm(m_te * N_hat_te, bts_mean_te)
h_te <- 2L
base_te <- rnorm(sum(te_fh_te) * h_te, rep(te_set_te * bts_mean_te, h_te * te_fh_te))

# csrml fixtures (after terml consumes RNG)
agg_mat_cs <- t(c(1, 1))
dimnames(agg_mat_cs) <- list("A", c("B", "C"))
N_hat_cs <- 100L
ts_mean_cs <- c(20, 10, 10)
hat_cs <- matrix(rnorm(3L * N_hat_cs, mean = ts_mean_cs), N_hat_cs, byrow = TRUE)
colnames(hat_cs) <- unlist(dimnames(agg_mat_cs))
obs_cs <- matrix(rnorm(2L * N_hat_cs, mean = ts_mean_cs[-1]), N_hat_cs, byrow = TRUE)
colnames(obs_cs) <- colnames(agg_mat_cs)
h_cs <- 2L
base_cs <- matrix(rnorm(3L * h_cs, mean = ts_mean_cs), h_cs, byrow = TRUE)
colnames(base_cs) <- unlist(dimnames(agg_mat_cs))

# ── ctrml × {compact, all} × {randomForest, xgboost, lightgbm} ─────────────

for (feat in c("compact", "all")) {
  for (appr in c("randomForest", "xgboost", "lightgbm")) {
    label <- paste0("ctrml_", feat, "_", appr)
    snap_path <- file.path(snap_dir, paste0(label, ".qs2"))
    test_that(paste0(label, ": max_abs_diff == 0"), {
      skip_if_not(file.exists(snap_path), paste("Snapshot missing:", label))
      set.seed(1001)
      r <- ctrml(
        hat = hat_ct, obs = obs_ct, base = base_ct,
        agg_order = m_ct, agg_mat = agg_mat_ct,
        approach = appr, features = feat
      )
      old <- qs_read(snap_path)
      expect_equal(max(abs(r - old)), 0)
    })
  }
}

# ── terml × {low-high, all} × {randomForest, xgboost, lightgbm} ────────────

for (feat in c("low-high", "all")) {
  for (appr in c("randomForest", "xgboost", "lightgbm")) {
    feat_clean <- gsub("-", "_", feat)
    label <- paste0("terml_", feat_clean, "_", appr)
    snap_path <- file.path(snap_dir, paste0(label, ".qs2"))
    test_that(paste0(label, ": max_abs_diff == 0"), {
      skip_if_not(file.exists(snap_path), paste("Snapshot missing:", label))
      set.seed(2001)
      r <- terml(
        hat = hat_te, obs = obs_te, base = base_te,
        agg_order = m_te, approach = appr, features = feat
      )
      old <- qs_read(snap_path)
      expect_equal(max(abs(r - old)), 0)
    })
  }
}

# ── csrml × compact × lightgbm (control — csrml is UNTOUCHED by spd.12) ─────

test_that("csrml_compact_lightgbm: max_abs_diff == 0", {
  snap_path <- file.path(snap_dir, "csrml_compact_lightgbm.qs2")
  skip_if_not(file.exists(snap_path), "Snapshot missing: csrml_compact_lightgbm")
  set.seed(3001)
  r <- csrml(
    hat = hat_cs, obs = obs_cs, base = base_cs,
    agg_mat = agg_mat_cs, approach = "lightgbm", features = "bts"
  )
  old <- qs_read(snap_path)
  expect_equal(max(abs(r - old)), 0)
})

# ── ctrml × compact × lightgbm with synthetic NA injection ───────────────────

test_that("ctrml_compact_lightgbm_na: NA-path max_abs_diff == 0", {
  snap_path <- file.path(snap_dir, "ctrml_compact_lightgbm_na.qs2")
  skip_if_not(file.exists(snap_path), "Snapshot missing: ctrml_compact_lightgbm_na")
  hat_na <- hat_ct
  hat_na[, 1] <- NA  # inject all-NA into column 1 — triggers NA path in loop_body
  set.seed(4001)
  r <- ctrml(
    hat = hat_na, obs = obs_ct, base = base_ct,
    agg_order = m_ct, agg_mat = agg_mat_ct,
    approach = "lightgbm", features = "compact"
  )
  old <- qs_read(snap_path)
  expect_equal(max(abs(r - old)), 0)
})
