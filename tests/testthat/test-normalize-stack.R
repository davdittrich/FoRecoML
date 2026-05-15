test_that("(a) zscore: center=0, scale=1 after normalization", {
  set.seed(1)
  X <- matrix(rnorm(50 * 5, mean = 10, sd = 3), 50, 5)
  result <- normalize_stack(X, method = "zscore")
  expect_equal(colMeans(result$X_norm), rep(0, 5), tolerance = 1e-10)
  expect_equal(apply(result$X_norm, 2, sd), rep(1, 5), tolerance = 1e-10)
  expect_length(result$center, 5)
  expect_length(result$scale, 5)
})

test_that("(b) robust method: 6 scale_fn options work", {
  set.seed(2)
  X <- matrix(rnorm(30 * 3), 30, 3)
  for (fn in c("gmd", "mad_scaled", "iqr_scaled", "sd_c4")) {
    result <- normalize_stack(X, method = "robust", scale_fn = fn)
    expect_true(is.matrix(result$X_norm), info = fn)
    expect_equal(dim(result$X_norm), dim(X), info = fn)
  }
  # qn and sn require robscale -- skip if not installed
  skip_if_not_installed("robscale")
  for (fn in c("qn", "sn")) {
    result <- normalize_stack(X, method = "robust", scale_fn = fn)
    expect_true(is.matrix(result$X_norm), info = fn)
  }
})

test_that("(c) zero-scale guard: constant column gets scale=1, X_norm finite", {
  X <- cbind(matrix(rnorm(20 * 3), 20, 3), rep(5, 20))  # col 4 is constant
  result <- normalize_stack(X, method = "zscore")
  expect_equal(result$scale[4], 1)  # zero-scale replaced with 1
  expect_true(all(is.finite(result$X_norm[, 4])))
  expect_true(all(result$X_norm[, 4] == 0))  # constant col centers to 0
})
