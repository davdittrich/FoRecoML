# FoRecoML baseline benchmark — T0.2
# Scaled workloads (see plan T0.2 for original sizes: p=500, hat 72x15000)
# Scaling rationale: original workload would take hours per iteration with
# randomForest. Proportional scaling preserves regression-detection value.
# Run from /tmp/forecoml-rebuild with: Rscript benchmarks/run-baseline.R

library(FoRecoML)
library(FoReco)
library(bench)
library(qs2)

set.seed(42)

# ---------------------------------------------------------------------------
# Dimension helpers
# ---------------------------------------------------------------------------
# ctrml: base = n x (h*kt), hat = n x (N*kt), obs = nb x (N*m)
#   where n = na+nb, kt = ks+m, from cttools(agg_mat, agg_order)
# terml: base = vector h*kt, hat = vector N*kt, obs = vector N*m
#   (single series; from tetools(agg_order))
# csrml: base = h x n, hat = T_obs x n, obs = T_obs x nb
#   where n = na+nb from cstools(agg_mat)
# ---------------------------------------------------------------------------

make_ctrml_data <- function(p, T_obs, N_kt_cols) {
  # p bottom series, 1 aggregate row, agg_order = c(1,2) -> m=2, kt=3
  agg_mat   <- matrix(rep(1, p), nrow = 1L)
  agg_order <- c(1L, 2L)
  tmp <- cttools(agg_mat = agg_mat, agg_order = agg_order)
  n  <- tmp$dim[["n"]]
  nb <- tmp$dim[["nb"]]
  kt <- tmp$dim[["kt"]]
  m  <- tmp$dim[["m"]]

  # N = training periods at lowest frequency; ncol(hat) = N * kt
  # We choose N so that N*kt ≈ N_kt_cols (round down to multiple of kt)
  N <- max(1L, N_kt_cols %/% kt)

  base <- matrix(rnorm(n * 1L * kt), nrow = n)              # h=1
  hat  <- matrix(rnorm(n * N  * kt), nrow = n)
  obs  <- matrix(rnorm(nb * N * m),  nrow = nb)

  list(
    base = base, hat = hat, obs = obs,
    agg_mat = agg_mat, agg_order = agg_order,
    meta = list(n = n, nb = nb, kt = kt, m = m, N = N,
                hat_ncol = ncol(hat), obs_ncol = ncol(obs))
  )
}

make_terml_data <- function(T_obs, N_kt_cols) {
  # Single series; agg_order = c(1,2) -> m=2, kt=3
  agg_order <- c(1L, 2L)
  tmp <- tetools(agg_order = agg_order)
  kt <- tmp$dim[["kt"]]
  m  <- tmp$dim[["m"]]
  N  <- max(1L, N_kt_cols %/% kt)

  base <- rnorm(1L * kt)        # h=1
  hat  <- rnorm(N  * kt)
  obs  <- rnorm(N  * m)

  list(
    base = base, hat = hat, obs = obs,
    agg_order = agg_order,
    meta = list(kt = kt, m = m, N = N)
  )
}

make_csrml_data <- function(p, T_obs) {
  # p bottom series, 1 aggregate
  agg_mat <- matrix(rep(1, p), nrow = 1L)
  tmp <- cstools(agg_mat = agg_mat)
  n  <- tmp$dim[["n"]]
  nb <- tmp$dim[["nb"]]

  base <- matrix(rnorm(1L * n),          nrow = 1L, ncol = n)
  hat  <- matrix(rnorm(T_obs * n),  nrow = T_obs, ncol = n)
  obs  <- matrix(rnorm(T_obs * nb), nrow = T_obs, ncol = nb)

  list(
    base = base, hat = hat, obs = obs,
    agg_mat = agg_mat,
    meta = list(n = n, nb = nb, T_obs = T_obs)
  )
}

# ---------------------------------------------------------------------------
# Workload definitions (scaled from original p=500, hat 72x15000)
# ---------------------------------------------------------------------------
workloads <- list(
  small  = list(p = 12, T_obs = 50,  ncol_hat = 100,  iter = 5),
  medium = list(p = 24, T_obs = 72,  ncol_hat = 500,  iter = 3),
  large  = list(p = 50, T_obs = 72,  ncol_hat = 2000, iter = 2)
)

results <- list()

for (wname in names(workloads)) {
  w <- workloads[[wname]]
  cat(sprintf(
    "\n=== Workload: %s (p=%d, T_obs=%d, ncol_hat=%d) ===\n",
    wname, w$p, w$T_obs, w$ncol_hat
  ))

  # ---- ctrml ---------------------------------------------------------------
  tryCatch({
    d <- make_ctrml_data(w$p, w$T_obs, w$ncol_hat)
    cat(sprintf(
      "  ctrml data: base %dx%d, hat %dx%d, obs %dx%d\n",
      nrow(d$base), ncol(d$base),
      nrow(d$hat),  ncol(d$hat),
      nrow(d$obs),  ncol(d$obs)
    ))
    bm <- bench::mark(
      ctrml(
        base      = d$base,
        hat       = d$hat,
        obs       = d$obs,
        agg_mat   = d$agg_mat,
        agg_order = d$agg_order,
        approach  = "randomForest"
      ),
      iterations = w$iter,
      memory     = TRUE
    )
    key <- paste(wname, "ctrml", sep = "_")
    results[[key]] <- list(
      median_ms   = as.numeric(bm$median)    * 1e3,
      peak_rss_mb = as.numeric(bm$mem_alloc) / 1024^2,
      n_itr       = bm$n_itr
    )
    cat(sprintf(
      "  ctrml: %.1f ms, %.2f MB\n",
      results[[key]]$median_ms,
      results[[key]]$peak_rss_mb
    ))
  }, error = function(e) {
    msg <- conditionMessage(e)
    cat(sprintf("  ctrml ERROR: %s\n", msg))
    results[[paste(wname, "ctrml", sep = "_")]] <<- list(error = msg)
  })

  # ---- terml ---------------------------------------------------------------
  tryCatch({
    d <- make_terml_data(w$T_obs, w$ncol_hat)
    cat(sprintf(
      "  terml data: base len %d, hat len %d, obs len %d\n",
      length(d$base), length(d$hat), length(d$obs)
    ))
    bm <- bench::mark(
      terml(
        base      = d$base,
        hat       = d$hat,
        obs       = d$obs,
        agg_order = d$agg_order,
        approach  = "randomForest"
      ),
      iterations = w$iter,
      memory     = TRUE
    )
    key <- paste(wname, "terml", sep = "_")
    results[[key]] <- list(
      median_ms   = as.numeric(bm$median)    * 1e3,
      peak_rss_mb = as.numeric(bm$mem_alloc) / 1024^2,
      n_itr       = bm$n_itr
    )
    cat(sprintf(
      "  terml: %.1f ms, %.2f MB\n",
      results[[key]]$median_ms,
      results[[key]]$peak_rss_mb
    ))
  }, error = function(e) {
    msg <- conditionMessage(e)
    cat(sprintf("  terml ERROR: %s\n", msg))
    results[[paste(wname, "terml", sep = "_")]] <<- list(error = msg)
  })

  # ---- csrml ---------------------------------------------------------------
  tryCatch({
    d <- make_csrml_data(w$p, w$T_obs)
    cat(sprintf(
      "  csrml data: base %dx%d, hat %dx%d, obs %dx%d\n",
      nrow(d$base), ncol(d$base),
      nrow(d$hat),  ncol(d$hat),
      nrow(d$obs),  ncol(d$obs)
    ))
    bm <- bench::mark(
      csrml(
        base     = d$base,
        hat      = d$hat,
        obs      = d$obs,
        agg_mat  = d$agg_mat,
        approach = "randomForest"
      ),
      iterations = w$iter,
      memory     = TRUE
    )
    key <- paste(wname, "csrml", sep = "_")
    results[[key]] <- list(
      median_ms   = as.numeric(bm$median)    * 1e3,
      peak_rss_mb = as.numeric(bm$mem_alloc) / 1024^2,
      n_itr       = bm$n_itr
    )
    cat(sprintf(
      "  csrml: %.1f ms, %.2f MB\n",
      results[[key]]$median_ms,
      results[[key]]$peak_rss_mb
    ))
  }, error = function(e) {
    msg <- conditionMessage(e)
    cat(sprintf("  csrml ERROR: %s\n", msg))
    results[[paste(wname, "csrml", sep = "_")]] <<- list(error = msg)
  })
}

# ---------------------------------------------------------------------------
# Save frozen result
# ---------------------------------------------------------------------------
outfile <- "benchmarks/baseline-25be237.qs2"
qs2::qs_save(
  list(
    results   = results,
    workloads = workloads,
    date      = Sys.time(),
    branch    = "25be237",
    R_version = R.version.string,
    note      = paste(
      "Scaled workloads: original p=500, hat 72x15000.",
      "Scaled to small(p=12,T=50,ncol=100),",
      "medium(p=24,T=72,ncol=500),",
      "large(p=50,T=72,ncol=2000) for tractable CI benchmarks."
    )
  ),
  outfile
)
cat(sprintf("\nSaved: %s\n", outfile))
