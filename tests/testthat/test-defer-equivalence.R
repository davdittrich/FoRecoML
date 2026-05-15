# T3: verify deferred hat/base expansion (mfh path) produces identical
# reconciliations to the legacy eager-expansion behavior.
#
# Strategy: drive rml() directly with the same data in two modes (eager and
# deferred) and assert byte-equivalence on per-series bts output. Doing this
# at the rml() boundary (not at ctrml/terml) avoids dependence on FoReco's
# downstream ctbu()/tebu() reshape conventions that are unrelated to T3 and
# only test what T3 actually changed: hat/base column materialization.

if (require(testthat)) {
  set.seed(123)

  # ---- ctrml mfh equivalence: rml(eager) vs rml(deferred) ----------------
  test_that("rml mfh defer (ctrml form): same bts as eager pre-expansion", {
    skip_if_not_installed("lightgbm")
    m <- 4
    kset <- c(4L, 2L, 1L)
    kt <- sum(max(kset) / kset) # 4 + 2 + 1 = 7
    n <- 3L
    h_train <- 8L
    h_predict <- 2L

    # raw hat: n x (h_train * kt)
    set.seed(1)
    hat_raw <- matrix(rnorm(n * h_train * kt, mean = 5),
                       nrow = n, ncol = h_train * kt)
    base_raw <- matrix(rnorm(n * h_predict * kt, mean = 5),
                       nrow = n, ncol = h_predict * kt)

    # eager-expanded form: h x (n*kt)
    hat_eager <- FoRecoML:::mat2hmat(hat_raw, h = h_train, kset = kset, n = n)
    base_eager <- FoRecoML:::mat2hmat(base_raw, h = h_predict, kset = kset, n = n)

    # obs: h_train x p, where p = NCOL(hat_eager) for mfh-all-style sel_mat = 1
    p <- NCOL(hat_eager) # n * kt = 21
    obs_mat <- matrix(rnorm(h_train * p, mean = 5), nrow = h_train, ncol = p)

    sel_mat <- 1

    # Eager path: rml gets pre-expanded matrices, no kset.
    set.seed(42)
    out_eager <- FoRecoML:::rml(
      approach = "lightgbm",
      hat = hat_eager,
      obs = obs_mat,
      base = base_eager,
      sel_mat = sel_mat,
      seed = 42,
      checkpoint = FALSE
    )

    # Deferred path: rml gets raw hat/base + kset/h/h_base/n.
    set.seed(42)
    out_defer <- FoRecoML:::rml(
      approach = "lightgbm",
      hat = hat_raw,
      obs = obs_mat,
      base = base_raw,
      sel_mat = sel_mat,
      seed = 42,
      kset = kset,
      h = h_train,
      h_base = h_predict,
      n = n,
      checkpoint = FALSE
    )

    # Both return reconciled bts; with sel_mat=1 we train p=21 models and
    # cbind their 2-row predictions -> h_predict x p matrix.
    expect_equal(dim(out_defer), dim(out_eager))
    expect_equal(as.vector(out_defer), as.vector(out_eager), tolerance = 1e-10)
  })

  # ---- terml mfh equivalence -----------------------------------------------
  test_that("rml mfh defer (terml form): same bts as eager vec2hmat", {
    skip_if_not_installed("lightgbm")
    kset <- c(4L, 2L, 1L)
    kt <- sum(max(kset) / kset)
    h_train <- 8L
    h_predict <- 2L

    # raw hat: length(h_train * kt) vector (terml shape)
    set.seed(2)
    hat_vec <- rnorm(h_train * kt, mean = 5)
    base_vec <- rnorm(h_predict * kt, mean = 5)

    hat_eager <- FoRecoML:::vec2hmat(hat_vec, h = h_train, kset = kset)
    base_eager <- FoRecoML:::vec2hmat(base_vec, h = h_predict, kset = kset)

    p <- NCOL(hat_eager) # = kt = 7
    obs_mat <- matrix(rnorm(h_train * p, mean = 5), nrow = h_train, ncol = p)

    sel_mat <- 1

    set.seed(42)
    out_eager <- FoRecoML:::rml(
      approach = "lightgbm",
      hat = hat_eager,
      obs = obs_mat,
      base = base_eager,
      sel_mat = sel_mat,
      seed = 42,
      checkpoint = FALSE
    )

    # Deferred: hat/base stashed as 1xL matrices, kset + h + h_base set, n = NULL.
    hat_raw <- hat_vec
    dim(hat_raw) <- c(1L, length(hat_raw))
    base_raw <- base_vec
    dim(base_raw) <- c(1L, length(base_raw))

    set.seed(42)
    out_defer <- FoRecoML:::rml(
      approach = "lightgbm",
      hat = hat_raw,
      obs = obs_mat,
      base = base_raw,
      sel_mat = sel_mat,
      seed = 42,
      kset = kset,
      h = h_train,
      h_base = h_predict,
      n = NULL,
      checkpoint = FALSE
    )

    expect_equal(dim(out_defer), dim(out_eager))
    expect_equal(as.vector(out_defer), as.vector(out_eager), tolerance = 1e-10)
  })

  # ---- helper sanity ------------------------------------------------------
  test_that("mat2hmat_cols matches mat2hmat[, cols]", {
    set.seed(1)
    kset <- c(4L, 2L, 1L)
    mat <- matrix(rnorm(3 * 14), nrow = 3) # n=3, h=2, kt=7 -> ncol=14
    full <- FoRecoML:::mat2hmat(mat, h = 2, kset = kset, n = 3)
    partial <- FoRecoML:::mat2hmat_cols(
      mat, h = 2, kset = kset, n = 3, cols = c(1L, 4L, 9L)
    )
    expect_equal(partial, full[, c(1L, 4L, 9L), drop = FALSE])
  })

  test_that("vec2hmat_cols matches vec2hmat[, cols]", {
    set.seed(2)
    kset <- c(4L, 2L, 1L)
    vec <- rnorm(2 * 7) # h=2, kt=7
    full <- FoRecoML:::vec2hmat(vec, h = 2, kset = kset)
    partial <- FoRecoML:::vec2hmat_cols(
      vec, h = 2, kset = kset, cols = c(2L, 5L)
    )
    expect_equal(partial, full[, c(2L, 5L), drop = FALSE])
  })

  # ---- csrml regression smoke test ---------------------------------------
  # csrml is the cross-sectional path, unrelated to mfh. T3 added optional
  # kset/h/h_base/n params to rml(); this test guards that csrml's caller
  # path (which omits all four) continues to work.
  test_that("csrml unaffected by rml signature changes", {
    skip_if_not_installed("lightgbm")
    set.seed(789)
    n <- 3
    nb <- 2
    p_obs <- 16
    h_cs <- 2
    cs_agg <- t(c(1, 1))
    dimnames(cs_agg) <- list("A", c("B", "C"))
    # csrml expects hat: (h_train x n), obs: (h_train x nb), base: (h_predict x n)
    hat_cs <- matrix(rnorm(p_obs * n, 5), nrow = p_obs, ncol = n)
    colnames(hat_cs) <- c("A", "B", "C")
    obs_cs <- matrix(rnorm(p_obs * nb, 5), nrow = p_obs, ncol = nb)
    colnames(obs_cs) <- c("B", "C")
    base_cs <- matrix(rnorm(h_cs * n, 5), nrow = h_cs, ncol = n)
    colnames(base_cs) <- c("A", "B", "C")
    expect_no_error(
      csrml(
        hat = hat_cs,
        obs = obs_cs,
        base = base_cs,
        agg_mat = cs_agg,
        approach = "lightgbm"
      )
    )
  })
}
