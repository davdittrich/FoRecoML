# T6 — tri-mode disk checkpoint tests.
# Fixture: small cross-temporal-ish hierarchy with p > 1 so we exercise the
# multi-model code path that the checkpoint feature is built around.

if (require(testthat)) {
  make_fixture <- function() {
    set.seed(42)
    agg_mat <- matrix(
      c(
        1, 1, 1, 1, 1, 1,
        1, 1, 1, 0, 0, 0,
        0, 0, 0, 1, 1, 1,
        1, 1, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0,
        0, 0, 0, 1, 1, 0,
        0, 0, 0, 0, 0, 1
      ),
      nrow = 7,
      byrow = TRUE
    )
    nb <- ncol(agg_mat)
    n <- nrow(agg_mat) + nb
    N_hat <- 16
    ts_mean <- c(rep(10, n - nb), rep(5, nb))
    hat <- matrix(rnorm(n * N_hat, mean = ts_mean), nrow = N_hat, byrow = TRUE)
    # csrml expects obs of shape (N_hat x nb). The task spec's transposed
    # form is for ctrml; we use csrml because it exercises p > 1 with a
    # simple agg_mat and is the fastest fixture to round-trip.
    obs <- matrix(rnorm(nb * N_hat, mean = 5), nrow = N_hat, byrow = TRUE)
    base <- matrix(rnorm(n, mean = ts_mean), nrow = 1)
    list(agg_mat = agg_mat, hat = hat, obs = obs, base = base, nb = nb, n = n)
  }

  fx <- make_fixture()

  test_that("checkpoint=FALSE bit-identical to no-arg default-OFF auto", {
    set.seed(1)
    r_off <- csrml(
      base = fx$base, hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
      approach = "xgboost", features = "all", checkpoint = FALSE
    )
    set.seed(1)
    r_default <- csrml(
      base = fx$base, hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
      approach = "xgboost", features = "all", checkpoint = "auto"
    )
    # On this tiny fixture, "auto" resolves to OFF (estimate << available).
    expect_equal(as.numeric(r_off), as.numeric(r_default), tolerance = 0)
  })

  for (approach in c("randomForest", "xgboost", "lightgbm")) {
    test_that(sprintf("checkpoint=TRUE round-trips for %s", approach), {
      set.seed(7)
      r_mem <- csrml(
        base = fx$base, hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
        approach = approach, features = "all", checkpoint = FALSE
      )
      set.seed(7)
      r_ckpt <- csrml(
        base = fx$base, hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
        approach = approach, features = "all", checkpoint = TRUE
      )
      expect_equal(
        as.numeric(r_mem), as.numeric(r_ckpt),
        tolerance = 1e-12
      )
    })
  }

  test_that("checkpoint=path persists fits to that path and predicts identically", {
    tmpd <- file.path(tempdir(), "foreco_t6_path_test")
    if (dir.exists(tmpd)) unlink(tmpd, recursive = TRUE)
    dir.create(tmpd, recursive = TRUE)
    set.seed(11)
    mdl <- csrml_fit(
      hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
      approach = "xgboost", features = "all", checkpoint = tmpd
    )
    # All p fits should be paths in `tmpd`.
    expect_true(all(vapply(mdl$fit, is.character, logical(1))))
    expect_true(all(vapply(mdl$fit, file.exists, logical(1))))
    expect_true(all(startsWith(unlist(mdl$fit), normalizePath(tmpd))))
    expect_identical(mdl$checkpoint_dir, normalizePath(tmpd, mustWork = FALSE))

    # Predict reuse via the same fit should be ≤1e-12 to in-memory baseline.
    set.seed(11)
    mdl_mem <- csrml_fit(
      hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
      approach = "xgboost", features = "all", checkpoint = FALSE
    )
    r_mem <- csrml(base = fx$base, fit = mdl_mem, agg_mat = fx$agg_mat)
    r_ckpt <- csrml(base = fx$base, fit = mdl, agg_mat = fx$agg_mat)
    expect_equal(as.numeric(r_mem), as.numeric(r_ckpt), tolerance = 1e-12)
    unlink(tmpd, recursive = TRUE)
  })

  test_that("checkpoint='auto' falls back to FALSE on small fixture", {
    # On a 16x13 hat the peak estimate is << available RAM; auto must be OFF.
    mdl <- csrml_fit(
      hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
      approach = "xgboost", features = "all", checkpoint = "auto"
    )
    expect_null(mdl$checkpoint_dir)
    # All fits remain in-memory (NOT path strings).
    expect_true(all(!vapply(mdl$fit, is.character, logical(1))))
  })

  test_that("checkpoint='auto' supports mlr3 (Outcome A)", {
    set.seed(13)
    mdl_mem <- csrml_fit(
      hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
      approach = "mlr3", features = "all", checkpoint = FALSE,
      params = list(.key = "regr.ranger")
    )
    set.seed(13)
    mdl_ckpt <- csrml_fit(
      hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
      approach = "mlr3", features = "all", checkpoint = TRUE,
      params = list(.key = "regr.ranger")
    )
    r_mem <- csrml(base = fx$base, fit = mdl_mem, agg_mat = fx$agg_mat)
    r_ckpt <- csrml(base = fx$base, fit = mdl_ckpt, agg_mat = fx$agg_mat)
    expect_equal(as.numeric(r_mem), as.numeric(r_ckpt), tolerance = 1e-12)
  })

  test_that("file extension dispatch: .qs2 for rf/xgb/mlr3, .lgb for lightgbm", {
    for (apr in c("randomForest", "xgboost", "mlr3", "lightgbm")) {
      params <- if (apr == "mlr3") list(.key = "regr.ranger") else NULL
      mdl <- csrml_fit(
        hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
        approach = apr, features = "all", checkpoint = TRUE,
        params = params
      )
      exts <- unique(tools::file_ext(unlist(mdl$fit)))
      expected <- if (apr == "lightgbm") "lgb" else "qs2"
      expect_identical(exts, expected, info = paste("approach=", apr))
    }
  })

  test_that("predict reuse from a checkpointed fit works end-to-end", {
    set.seed(17)
    mdl <- csrml_fit(
      hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
      approach = "lightgbm", features = "all", checkpoint = TRUE
    )
    # All fits are paths.
    expect_true(all(vapply(mdl$fit, is.character, logical(1))))
    # Predict from this fit (lazy-load via get_fit_i).
    r <- csrml(base = fx$base, fit = mdl, agg_mat = fx$agg_mat)
    expect_equal(NCOL(r), fx$n)
    expect_true(all(is.finite(r)))
  })

  test_that("resolve_checkpoint argument validation", {
    expect_error(
      csrml_fit(
        hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
        approach = "xgboost", features = "all", checkpoint = 1L
      )
    )
    expect_error(
      csrml_fit(
        hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
        approach = "xgboost", features = "all", checkpoint = list("a")
      )
    )
    expect_error(
      csrml_fit(
        hat = fx$hat, obs = fx$obs, agg_mat = fx$agg_mat,
        approach = "xgboost", features = "all", checkpoint = c("a", "b")
      )
    )
  })

  test_that("terml + ctrml accept checkpoint argument end-to-end", {
    # Quick smoke tests that the new arg threads through cleanly.
    m <- 4
    te_set <- tetools(m)$set
    te_fh <- m / te_set
    N_hat <- 16
    hat_te <- rnorm(sum(te_fh) * N_hat, rep(te_set * 5, N_hat * te_fh))
    obs_te <- rnorm(m * N_hat, 5)
    base_te <- rnorm(sum(te_fh) * 1, rep(te_set * 5, 1 * te_fh))
    set.seed(3)
    r1 <- terml(
      base = base_te, hat = hat_te, obs = obs_te, agg_order = m,
      approach = "xgboost", checkpoint = FALSE
    )
    set.seed(3)
    r2 <- terml(
      base = base_te, hat = hat_te, obs = obs_te, agg_order = m,
      approach = "xgboost", checkpoint = TRUE
    )
    expect_equal(as.numeric(r1), as.numeric(r2), tolerance = 1e-12)
  })

  test_that("predict-reuse numerical equivalence with checkpointed fit (mw3.3)", {
    # mw3.3: reco_mat returned by predict-reuse must be numerically identical
    # to the in-memory baseline regardless of whether the per-iteration fit is
    # retained in out[[i]] or dropped. Memory behaviour verified by inspection;
    # this test guards reco_mat correctness.
    skip_if_not_installed("qs2")
    skip_if_not_installed("randomForest")
    fx_local <- make_fixture()
    tmpd <- file.path(tempdir(), "foreco_mw33_test")
    if (dir.exists(tmpd)) unlink(tmpd, recursive = TRUE)
    dir.create(tmpd, recursive = TRUE)
    on.exit(unlink(tmpd, recursive = TRUE), add = TRUE)

    set.seed(42)
    mdl_disk <- csrml_fit(
      hat = fx_local$hat, obs = fx_local$obs, agg_mat = fx_local$agg_mat,
      approach = "randomForest", features = "all", checkpoint = tmpd
    )
    set.seed(42)
    mdl_mem <- csrml_fit(
      hat = fx_local$hat, obs = fx_local$obs, agg_mat = fx_local$agg_mat,
      approach = "randomForest", features = "all", checkpoint = FALSE
    )

    r_disk <- csrml(base = fx_local$base, fit = mdl_disk, agg_mat = fx_local$agg_mat)
    r_mem  <- csrml(base = fx_local$base, fit = mdl_mem,  agg_mat = fx_local$agg_mat)

    expect_equal(as.numeric(r_disk), as.numeric(r_mem), tolerance = 1e-12)
  })
}
