# Shared fixture constructors for *_g tests (ak9)

make_g_fixture_cs <- function(p = 4L, T_obs = 20L, h = 2L) {
  set.seed(99)
  na <- 2L
  n <- na + p
  agg_mat <- matrix(c(1,1,0,0,0,0,1,1), nrow = na, ncol = p,
                    dimnames = list(c("G1","G2"), paste0("S", seq_len(p))))
  obs <- matrix(rnorm(T_obs * p), T_obs, p,
                dimnames = list(NULL, paste0("S", seq_len(p))))
  hat <- matrix(rnorm(T_obs * n), T_obs, n,
                dimnames = list(NULL, c(rownames(agg_mat), colnames(obs))))
  base <- matrix(rnorm(h * n), h, n,
                 dimnames = list(NULL, colnames(hat)))
  list(agg_mat = agg_mat, obs = obs, hat = hat, base = base)
}

make_g_fixture_te <- function(T_obs = 60L, h = 2L, agg_order = c(4L, 2L, 1L)) {
  set.seed(99)
  m  <- max(agg_order)
  kt <- sum(m / agg_order)
  hat  <- matrix(rnorm(T_obs * kt), T_obs, kt,
                 dimnames = list(NULL, paste0("L", seq_len(kt))))
  obs  <- matrix(rnorm(T_obs),  T_obs, 1L,
                 dimnames = list(NULL, "S1"))
  base <- matrix(rnorm(h * m * kt), h * m, kt,
                 dimnames = list(NULL, colnames(hat)))
  list(agg_order = agg_order, obs = obs, hat = hat, base = base)
}

make_g_fixture_ct <- function(p = 2L, T_obs = 60L, h = 2L, agg_order = c(4L, 2L, 1L)) {
  set.seed(99)
  m   <- max(agg_order)
  kt  <- sum(m / agg_order)
  na  <- 1L
  n   <- na + p
  ncf <- n * kt
  agg_mat <- matrix(c(1, 1), nrow = na, ncol = p,
                    dimnames = list("G1", paste0("S", seq_len(p))))
  obs  <- matrix(rnorm(T_obs * p), T_obs, p,
                 dimnames = list(NULL, paste0("S", seq_len(p))))
  hat  <- matrix(rnorm(T_obs * ncf), T_obs, ncf,
                 dimnames = list(NULL, paste0("F", seq_len(ncf))))
  base <- matrix(rnorm(h * m * ncf), h * m, ncf,
                 dimnames = list(NULL, colnames(hat)))
  list(agg_mat = agg_mat, agg_order = agg_order, obs = obs, hat = hat, base = base)
}
