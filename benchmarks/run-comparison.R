# T9 benchmark comparison — post-rebuild vs baseline 25be237
# Pass A: approach="randomForest" (apples-to-apples refactor speedup)
# Pass B: approach="ranger" (user-experience perf)

library(FoRecoML)
library(bench)

set.seed(42)

# Helper: build minimal valid csrml fixture
# csrml needs: base (n×h), hat (T_obs×n), obs (T_obs×nb), agg_mat
# n = number of series (p bottom + 1 upper), nb = p bottom-level series
make_bench_fixture <- function(p = 6L, T_obs = 20L) {
  agg_mat <- matrix(1, 1, p, dimnames = list("T", paste0("S", seq_len(p))))
  nb <- p          # bottom-level series count
  n  <- p + 1L     # total series (matches cstools(agg_mat)$dim[["n"]])
  obs <- matrix(rnorm(T_obs * nb, 100), T_obs, nb,
                dimnames = list(NULL, paste0("S", seq_len(nb))))
  # hat: T_obs rows × n cols  (training length × total series)
  hat <- matrix(rnorm(T_obs * n, 100), T_obs, n,
                dimnames = list(NULL, c("T", paste0("S", seq_len(nb)))))
  # base: h × n  (forecast horizon h=1 × total series n)
  base <- matrix(c(nb * 100, rnorm(nb, 100)), 1L, n,
                 dimnames = list(NULL, c("T", paste0("S", seq_len(nb)))))
  list(base = base, hat = hat, obs = obs, agg_mat = agg_mat)
}

# Use SMALL workload for reasonable timing
fx_small <- make_bench_fixture(p = 6L, T_obs = 20L)

cat("\n=== Pass A: randomForest (apples-to-apples) ===\n")
bm_a <- bench::mark(
  csrml_randomForest = csrml(
    base = fx_small$base, hat = fx_small$hat, obs = fx_small$obs,
    agg_mat = fx_small$agg_mat, approach = "randomForest", features = "all"
  ),
  iterations = 2L, memory = TRUE, check = FALSE
)
cat(sprintf("  csrml(randomForest): median=%.1f ms, mem=%.1f MB\n",
            as.numeric(bm_a$median) * 1000,
            as.numeric(bm_a$mem_alloc) / 1024^2))

cat("\n=== Pass B: ranger (new default) ===\n")
bm_b <- bench::mark(
  csrml_ranger = csrml(
    base = fx_small$base, hat = fx_small$hat, obs = fx_small$obs,
    agg_mat = fx_small$agg_mat, approach = "ranger", features = "all"
  ),
  iterations = 2L, memory = TRUE, check = FALSE
)
cat(sprintf("  csrml(ranger): median=%.1f ms, mem=%.1f MB\n",
            as.numeric(bm_b$median) * 1000,
            as.numeric(bm_b$mem_alloc) / 1024^2))

# Save results
qs2::qs_save(
  list(pass_a = bm_a, pass_b = bm_b,
       fixture = list(p = 6L, T_obs = 20L, n = 7L, nb = 6L),
       date = Sys.time(), branch = "refactor/clean-baseline"),
  "benchmarks/comparison-results.qs2"
)

# Compute speedup Pass A vs baseline (placeholder — baseline not captured)
cat("\n=== Summary ===\n")
cat(sprintf("Pass A (randomForest): %.1f ms\n", as.numeric(bm_a$median) * 1000))
cat(sprintf("Pass B (ranger):       %.1f ms\n", as.numeric(bm_b$median) * 1000))
cat(sprintf("B/A ratio:             %.2fx\n",
            as.numeric(bm_b$median) / as.numeric(bm_a$median)))
cat("\nNote: T0.2 baseline did not capture full timing (API shape mismatch).\n")
cat("Pass A vs baseline: N/A (see baseline-summary.md)\n")
cat("Pass B vs Pass A:   above ratio\n")
cat("SLA: Pass B/A <= 1.05 (ranger should be similar to randomForest)\n")
