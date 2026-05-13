#!/usr/bin/env Rscript
# spd13-baseline.R — capture pre-spd.13 bts snapshots for numerical equivalence gate
# Run BEFORE applying spd.13 changes (on HEAD 4efeef9).
# Outputs: tests/testthat/fixtures/spd13/<config>.qs2
#
# IMPORTANT: set.seed(42) is called ONCE; RNG state flows through all calls.
# Any change to call order or count breaks snapshot reproducibility. Do NOT reorder.

devtools::load_all(quiet = TRUE)
library(qs2)

snap_dir <- "tests/testthat/fixtures/spd13"
dir.create(snap_dir, showWarnings = FALSE, recursive = TRUE)

set.seed(42)

# ── Cross-temporal setup ───────────────────────────────────────────────────────
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

# ── Helper ─────────────────────────────────────────────────────────────────────
run_and_save <- function(label, seed, expr) {
  message("Capturing: ", label)
  set.seed(seed)
  result <- tryCatch(eval(expr), error = function(e) e)
  path   <- file.path(snap_dir, paste0(label, ".qs2"))
  qs2::qs_save(result, path)
  message("  saved -> ", path)
}

# ── 9 primary mfh cases: {mfh-all, mfh-str-bts, mfh-bts} x {randomForest, xgboost, lightgbm} ──
for (feat in c("mfh-all", "mfh-str-bts", "mfh-bts")) {
  for (appr in c("randomForest", "xgboost", "lightgbm")) {
    label <- paste0("ctrml_", gsub("-", "_", feat), "_", appr)
    seed  <- switch(paste(feat, appr),
      "mfh-all randomForest"  = 1001,
      "mfh-all xgboost"       = 1002,
      "mfh-all lightgbm"      = 1003,
      "mfh-str-bts randomForest" = 2001,
      "mfh-str-bts xgboost"      = 2002,
      "mfh-str-bts lightgbm"     = 2003,
      "mfh-bts randomForest"  = 3001,
      "mfh-bts xgboost"       = 3002,
      "mfh-bts lightgbm"      = 3003
    )
    run_and_save(label, seed = seed, quote(ctrml(
      hat = hat_ct, obs = obs_ct, base = base_ct,
      agg_order = m_ct, agg_mat = agg_mat_ct,
      approach = appr, features = feat
    )))
  }
}

# ── NA-injection: mfh-all x lightgbm with NA in hat col 1 ─────────────────────
{
  hat_na    <- hat_ct
  hat_na[, 1] <- NA
  run_and_save("ctrml_mfh_all_lightgbm_na", seed = 4001, quote(ctrml(
    hat = hat_na, obs = obs_ct, base = base_ct,
    agg_order = m_ct, agg_mat = agg_mat_ct,
    approach = "lightgbm", features = "mfh-all"
  )))
}

# ── Predict-reuse: mfh-all x lightgbm train then predict ──────────────────────
{
  hat_na    <- hat_ct
  hat_na[, 1] <- NA
  run_and_save("ctrml_mfh_all_lightgbm_na_predict_reuse", seed = 4002, quote({
    mdl <- ctrml_fit(
      hat = hat_na, obs = obs_ct,
      agg_order = m_ct, agg_mat = agg_mat_ct,
      approach = "lightgbm", features = "mfh-all"
    )
    ctrml(base = base_ct, fit = mdl, agg_order = m_ct, agg_mat = agg_mat_ct)
  }))
}

# ── Horizon mismatch (expect error): mfh-all x lightgbm with wrong-h base ─────
{
  run_and_save("ctrml_mfh_all_lightgbm_hmismatch", seed = 4003, quote(tryCatch({
    mdl <- ctrml_fit(
      hat = hat_ct, obs = obs_ct,
      agg_order = m_ct, agg_mat = agg_mat_ct,
      approach = "lightgbm", features = "mfh-all"
    )
    # base with h=1 vs training h_train derived from hat
    base_wrong_h <- rbind(
      rnorm(sum(te_fh_ct) * 1, rep(2 * te_set_ct * bts_mean_ct, 1 * te_fh_ct)),
      rnorm(sum(te_fh_ct) * 1, rep(    te_set_ct * bts_mean_ct, 1 * te_fh_ct)),
      rnorm(sum(te_fh_ct) * 1, rep(    te_set_ct * bts_mean_ct, 1 * te_fh_ct))
    )
    rownames(base_wrong_h) <- c("A", "B", "C")
    ctrml(base = base_wrong_h, fit = mdl, agg_order = m_ct, agg_mat = agg_mat_ct)
  }, error = function(e) structure(list(message = conditionMessage(e)), class = "captured_error"))))
}

message("\nDone. ", length(list.files(snap_dir, pattern = "*.qs2")), " snapshots saved.")
