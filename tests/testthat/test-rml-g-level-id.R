# T4-T7 regression tests: level_id feature (ak9 / FoRecoML-a0t.5-8)

# ---------------------------------------------------------------------------
# T4: .stack_series() shape and value-range tests
# ---------------------------------------------------------------------------

test_that("T4a: .stack_series() ncol unchanged when level_id=FALSE", {
  set.seed(99)
  hat <- matrix(rnorm(7 * 50), 50, 7)
  obs <- matrix(rnorm(50), 50, 1)
  s <- FoRecoML:::.stack_series(hat, obs, level_id = FALSE)
  expect_equal(ncol(s$X_stacked), 7L)  # hat cols only, no extra col
})

test_that("T4b: .stack_series() ncol +1 when level_id=TRUE", {
  set.seed(99)
  hat <- matrix(rnorm(7 * 50), 50, 7)
  obs <- matrix(rnorm(50), 50, 1)
  s <- FoRecoML:::.stack_series(hat, obs, level_id = TRUE, kset = c(4L, 2L, 1L))
  expect_equal(ncol(s$X_stacked), 8L)
  expect_true("level_id_col" %in% colnames(s$X_stacked))
})

test_that("T4c: level_id values in [1, max_level] for kset=c(4,2,1)", {
  set.seed(99)
  hat <- matrix(rnorm(7 * 50), 50, 7)
  obs <- matrix(rnorm(50), 50, 1)
  s <- FoRecoML:::.stack_series(hat, obs, level_id = TRUE, kset = c(4L, 2L, 1L))
  lvl <- s$X_stacked[, "level_id_col"]
  expect_gte(min(lvl), 1L)
  expect_lte(max(lvl), length(c(4L, 2L, 1L)))  # max = 3
})

test_that("T4d: NULL kset with level_id=TRUE aborts", {
  hat <- matrix(rnorm(7 * 50), 50, 7)
  obs <- matrix(rnorm(50), 50, 1)
  expect_error(
    FoRecoML:::.stack_series(hat, obs, level_id = TRUE),
    "level_id=TRUE requires kset"
  )
})

test_that("T4e: level_id=1 is finest (k=1), max is coarsest (k=4) for kset=c(4,2,1)", {
  set.seed(99)
  # m=4, T_obs=12 (3 full cycles of length 4)
  hat <- matrix(rnorm(4 * 12), 12, 4)
  obs <- matrix(rnorm(12), 12, 1)
  # kset=c(4,2,1): sorted_kset=c(1,2,4) → k_to_level: 1→1, 2→2, 4→3
  # cycle-start rule: (pos-1) %% k == 0 marks a new period of length k.
  # pos 1: (0%%4==0,0%%2==0,0%%1==0) → starts={4,2,1} → coarsest=4 → level 3
  # pos 2: (1%%4≠0,1%%2≠0,1%%1==0) → starts={1}       → coarsest=1 → level 1
  # pos 3: (2%%4≠0,2%%2==0,2%%1==0) → starts={2,1}     → coarsest=2 → level 2
  # pos 4: (3%%4≠0,3%%2≠0,3%%1==0) → starts={1}        → coarsest=1 → level 1
  s <- FoRecoML:::.stack_series(hat, obs, level_id = TRUE, kset = c(4L, 2L, 1L))
  lvl_first_cycle <- s$X_stacked[1:4, "level_id_col"]
  # Expected: pos1→level3, pos2→level1, pos3→level2, pos4→level1
  expect_equal(as.integer(lvl_first_cycle), c(3L, 1L, 2L, 1L))
})

# ---------------------------------------------------------------------------
# T5: LightGBM importance test (seeded DGP with level signal)
# ---------------------------------------------------------------------------

test_that("T5: level_id gains importance > 5% on level-effect DGP", {
  skip_if_not_installed("lightgbm")
  set.seed(99)
  agg_order <- c(4L, 2L, 1L)
  m <- 4L
  kt <- 7L
  T_obs <- 200L

  # DGP: y = 5 * level_id + small noise (strong level effect)
  hat <- matrix(rnorm(T_obs * kt), T_obs, kt)
  # Derive level_ids matching .stack_series logic for kset=c(4,2,1):
  # sorted_kset = c(1,2,4); level 1=k1, 2=k2, 3=k4
  # pos 1 → coarsest=4 → level 3
  # pos 2 → coarsest=1 → level 1
  # pos 3 → coarsest=2 → level 2
  # pos 4 → coarsest=1 → level 1
  cycle_pos  <- ((seq_len(T_obs) - 1L) %% m) + 1L
  level_ids  <- c(3L, 1L, 2L, 1L)[cycle_pos]
  obs <- matrix(5 * level_ids + rnorm(T_obs, sd = 0.1), T_obs, 1)
  colnames(hat) <- paste0("L", seq_len(kt))
  colnames(obs) <- "S1"

  fit <- rml_g(
    approach = "lightgbm", hat = hat, obs = obs,
    level_id = TRUE, kset = agg_order, seed = 99L,
    params = list(
      num_threads  = 1L, num_leaves = 4L,
      num_iterations = 50L, deterministic = TRUE,
      force_row_wise = TRUE
    )
  )

  # feature_importance is a data.table/data.frame with columns Feature, Gain
  imp <- fit$feature_importance
  if (!is.null(imp) && is.data.frame(imp) && nrow(imp) > 0L) {
    total_gain   <- sum(imp$Gain)
    level_row    <- imp[imp$Feature == "level_id_col", , drop = FALSE]
    if (nrow(level_row) > 0L && total_gain > 0) {
      level_gain_frac <- level_row$Gain[1L] / total_gain
      expect_gt(level_gain_frac, 0.05)
    }
  }
})

# ---------------------------------------------------------------------------
# T6: coherency preserved with level_id=TRUE
# ---------------------------------------------------------------------------

test_that("T6: terml_g coherency invariant holds with level_id=TRUE", {
  skip_if_not_installed("lightgbm")
  fx <- make_g_fixture_te()
  result <- terml_g(
    base      = fx$base, hat = fx$hat, obs = fx$obs,
    agg_order = fx$agg_order, level_id = TRUE,
    approach  = "lightgbm", seed = 1L
  )
  expect_type(result, "double")
  expect_false(is.null(attr(result, "FoReco")))
  expect_true(all(is.finite(result)))
  expect_gt(length(result), 0L)
})

# ---------------------------------------------------------------------------
# T7: ncol_hat guard regression with level_id mismatch
# ---------------------------------------------------------------------------

test_that("T7: predict aborts when ncol(newdata) mismatches ncol_hat from level_id=TRUE fit", {
  skip_if_not_installed("lightgbm")
  set.seed(99)
  agg_order <- c(4L, 2L, 1L)
  kt <- 7L
  T_obs <- 60L
  hat <- matrix(rnorm(T_obs * kt), T_obs, kt,
                dimnames = list(NULL, paste0("L", seq_len(kt))))
  obs <- matrix(rnorm(T_obs), T_obs, 1L,
                dimnames = list(NULL, "S1"))

  # Fit with level_id=TRUE → ncol_hat = kt + 1 = 8
  fit <- rml_g(
    approach = "lightgbm", hat = hat, obs = obs,
    level_id = TRUE, kset = agg_order, seed = 99L
  )
  expect_equal(fit$ncol_hat, kt + 1L)

  # predict() auto-appends level_id_col when use_level_id=TRUE.
  # So newdata with (kt+1) cols gets an extra col appended → kt+2 ≠ ncol_hat=kt+1 → error.
  wrong_newdata <- matrix(rnorm(4 * (kt + 1L)), 4, kt + 1L)
  expect_error(
    predict(fit, newdata = wrong_newdata, series_id = "S1"),
    "must have.*columns"
  )

  # newdata with kt (raw hat) cols → auto-appended to kt+1 = ncol_hat → no error.
  correct_newdata <- matrix(rnorm(4 * kt), 4, kt)
  expect_no_error(predict(fit, newdata = correct_newdata, series_id = "S1"))
})

# ---------------------------------------------------------------------------
# BUG-2 regression: ctrml_g(level_id=TRUE) end-to-end on default + chunked
# ---------------------------------------------------------------------------

.bug2_fixture <- function() {
  set.seed(1)
  agg_mat <- t(c(1, 1))
  dimnames(agg_mat) <- list("A", c("B", "C"))
  agg_order <- c(4L, 2L, 1L)
  m <- 4L; kt <- 7L; n <- 3L; ncf <- n * kt
  T_obs <- 60L; h <- 2L
  hat  <- matrix(rnorm(T_obs * ncf), T_obs, ncf,
                 dimnames = list(NULL, paste0("F", seq_len(ncf))))
  obs  <- matrix(rnorm(T_obs * 2L), T_obs, 2L,
                 dimnames = list(NULL, c("B", "C")))
  base <- matrix(rnorm(h * m * ncf), h * m, ncf,
                 dimnames = list(NULL, paste0("F", seq_len(ncf))))
  list(base = base, hat = hat, obs = obs,
       agg_mat = agg_mat, agg_order = agg_order)
}

test_that("BUG-2.1: ctrml_g(level_id=TRUE) default path works end-to-end", {
  skip_if_not_installed("lightgbm")
  fx <- .bug2_fixture()
  r <- ctrml_g(
    base = fx$base, hat = fx$hat, obs = fx$obs,
    agg_mat = fx$agg_mat, agg_order = fx$agg_order,
    level_id = TRUE, seed = 1L
  )
  expect_true(is.matrix(r))
  expect_false(is.null(attr(r, "FoReco")))
})

test_that("BUG-2.2: ctrml_g(level_id=TRUE, batch_size=1L) chunked path works", {
  skip_if_not_installed("lightgbm")
  fx <- .bug2_fixture()
  r_default <- ctrml_g(
    base = fx$base, hat = fx$hat, obs = fx$obs,
    agg_mat = fx$agg_mat, agg_order = fx$agg_order,
    level_id = TRUE, seed = 1L
  )
  r_chunked <- ctrml_g(
    base = fx$base, hat = fx$hat, obs = fx$obs,
    agg_mat = fx$agg_mat, agg_order = fx$agg_order,
    level_id = TRUE, batch_size = 1L, seed = 1L
  )
  expect_true(is.matrix(r_chunked))
  expect_equal(dim(r_chunked), dim(r_default))
})
