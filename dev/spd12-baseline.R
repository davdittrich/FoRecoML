#!/usr/bin/env Rscript
# spd12-baseline.R — capture pre-spd.12 bts snapshots for numerical equivalence gate
# Run BEFORE applying spd.12 changes (on HEAD a90fd30).
# Outputs: tests/testthat/_snaps/spd12/<config>.qs2
#
# IMPORTANT: set.seed(42) is called ONCE at the top. The RNG state flows through
# ctrml, terml, and csrml fixture generation in order. Any change to this order
# breaks snapshot reproducibility. Do NOT reorder.

devtools::load_all(quiet = TRUE)
library(qs2)

snap_dir <- "tests/testthat/fixtures/spd12"
dir.create(snap_dir, recursive = TRUE, showWarnings = FALSE)

# ── Shared seed and fixtures (ORDER MATTERS for RNG state) ──────────────────

set.seed(42)

# --- ctrml fixture (generated FIRST from seed 42) ---
m_ct <- 4
te_set_ct <- tetools(m_ct)$set
te_fh_ct  <- m_ct / te_set_ct
agg_mat_ct <- t(c(1, 1))
dimnames(agg_mat_ct) <- list("A", c("B", "C"))

h_hat_ct <- 16
bts_mean_ct <- 5
hat_ct <- rbind(
  rnorm(sum(te_fh_ct) * h_hat_ct, rep(2 * te_set_ct * bts_mean_ct, h_hat_ct * te_fh_ct)),
  rnorm(sum(te_fh_ct) * h_hat_ct, rep(te_set_ct * bts_mean_ct, h_hat_ct * te_fh_ct)),
  rnorm(sum(te_fh_ct) * h_hat_ct, rep(te_set_ct * bts_mean_ct, h_hat_ct * te_fh_ct))
)
rownames(hat_ct) <- c("A", "B", "C")

obs_ct <- rbind(
  rnorm(m_ct * h_hat_ct, bts_mean_ct),
  rnorm(m_ct * h_hat_ct, bts_mean_ct)
)
rownames(obs_ct) <- c("B", "C")

h_ct <- 2
base_ct <- rbind(
  rnorm(sum(te_fh_ct) * h_ct, rep(2 * te_set_ct * bts_mean_ct, h_ct * te_fh_ct)),
  rnorm(sum(te_fh_ct) * h_ct, rep(te_set_ct * bts_mean_ct, h_ct * te_fh_ct)),
  rnorm(sum(te_fh_ct) * h_ct, rep(te_set_ct * bts_mean_ct, h_ct * te_fh_ct))
)

# --- terml fixture (generated SECOND, after ctrml consumes RNG) ---
m_te <- 4
te_set_te <- tetools(m_te)$set
te_fh_te  <- m_te / te_set_te

N_hat_te <- 16
bts_mean_te <- 5
hat_te <- rnorm(sum(te_fh_te) * N_hat_te, rep(te_set_te * bts_mean_te, N_hat_te * te_fh_te))
obs_te <- rnorm(m_te * N_hat_te, bts_mean_te)
h_te <- 2
base_te <- rnorm(sum(te_fh_te) * h_te, rep(te_set_te * bts_mean_te, h_te * te_fh_te))

# --- csrml fixture (generated THIRD) ---
agg_mat_cs <- t(c(1, 1))
dimnames(agg_mat_cs) <- list("A", c("B", "C"))
N_hat_cs <- 100
ts_mean_cs <- c(20, 10, 10)
hat_cs <- matrix(rnorm(length(ts_mean_cs) * N_hat_cs, mean = ts_mean_cs), N_hat_cs, byrow = TRUE)
colnames(hat_cs) <- unlist(dimnames(agg_mat_cs))
obs_cs <- matrix(rnorm(length(ts_mean_cs[-1]) * N_hat_cs, mean = ts_mean_cs[-1]), N_hat_cs, byrow = TRUE)
colnames(obs_cs) <- colnames(agg_mat_cs)
h_cs <- 2
base_cs <- matrix(rnorm(length(ts_mean_cs) * h_cs, mean = ts_mean_cs), h_cs, byrow = TRUE)
colnames(base_cs) <- unlist(dimnames(agg_mat_cs))

# ── Helper ───────────────────────────────────────────────────────────────────

run_and_save <- function(label, seed, expr) {
  message("  Running: ", label)
  set.seed(seed)
  r <- eval(expr)
  path <- file.path(snap_dir, paste0(label, ".qs2"))
  qs_save(r, path)
  dims <- if (is.matrix(r)) paste(dim(r), collapse = "x") else ""
  message("  Saved: ", path, " [", dims, "]")
}

# ── ctrml × {compact, all} × {randomForest, xgboost, lightgbm} ─────────────

message("=== ctrml snapshots ===")
for (feat in c("compact", "all")) {
  for (appr in c("randomForest", "xgboost", "lightgbm")) {
    label <- paste0("ctrml_", feat, "_", appr)
    run_and_save(label, seed = 1001, quote(ctrml(
      hat = hat_ct, obs = obs_ct, base = base_ct,
      agg_order = m_ct, agg_mat = agg_mat_ct,
      approach = appr, features = feat
    )))
  }
}

# ── terml × {low-high, all} × {randomForest, xgboost, lightgbm} ────────────

message("=== terml snapshots ===")
for (feat in c("low-high", "all")) {
  for (appr in c("randomForest", "xgboost", "lightgbm")) {
    feat_clean <- gsub("-", "_", feat)
    label <- paste0("terml_", feat_clean, "_", appr)
    run_and_save(label, seed = 2001, quote(terml(
      hat = hat_te, obs = obs_te, base = base_te,
      agg_order = m_te, approach = appr, features = feat
    )))
  }
}

# ── csrml × compact × lightgbm (control) ────────────────────────────────────

message("=== csrml snapshot ===")
run_and_save("csrml_compact_lightgbm", seed = 3001, quote(csrml(
  hat = hat_cs, obs = obs_cs, base = base_cs,
  agg_mat = agg_mat_cs, approach = "lightgbm", features = "bts"
)))

# ── ctrml × compact × lightgbm with synthetic NA-injected hat ───────────────

message("=== NA-injection snapshot ===")
{
  hat_na <- hat_ct
  # Column 1 of the raw hat corresponds to the first temporal feature level.
  # Setting it to all-NA creates an all-NA expanded column, triggering the
  # NA-path in loop_body (post-spd.12) or the NA block in wrapper (pre-spd.12).
  hat_na[, 1] <- NA

  run_and_save("ctrml_compact_lightgbm_na", seed = 4001, quote(ctrml(
    hat = hat_na, obs = obs_ct, base = base_ct,
    agg_order = m_ct, agg_mat = agg_mat_ct,
    approach = "lightgbm", features = "compact"
  )))
}

message("\nDone. ", length(list.files(snap_dir, "*.qs2")), " snapshots saved.")
