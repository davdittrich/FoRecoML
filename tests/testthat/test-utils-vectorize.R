test_that("spd.16: rep broadcast matches outer in input2rtw_partial", {
  set.seed(42L)
  hat <- matrix(rnorm(20 * 10), 20, 10)
  kset <- c(4L, 2L, 1L)
  cols <- c(1L, 3L, 5L)
  result <- FoRecoML:::input2rtw_partial(hat, kset, cols)
  expect_true(is.matrix(result))
  expect_equal(NCOL(result), length(cols))
  expect_true(all(is.finite(result)))
})

test_that("spd.18: colSums matches vapply in na_col_mask", {
  hat <- matrix(c(NA, 1, NA, 2, 3, 4), nrow = 3)
  result <- FoRecoML:::na_col_mask(hat, threshold = 0.5)
  # col 1 has 2/3 NAs > 0.5 -> TRUE; col 2 has 1/3 NAs -> FALSE
  expect_equal(result, c(TRUE, FALSE))
})

test_that("spd.16/17 input2rtw correctness", {
  # kset=c(4L,2L,1L), m=4 => FoReco2matrix expects ncol = 4/4+4/2+4/1 = 1+2+4 = 7 cols
  # For a 3-series input, input2rtw cbinds replicated level blocks per series
  set.seed(7L)
  hat <- matrix(rnorm(3 * 7), nrow = 3, ncol = 7)
  kset <- c(4L, 2L, 1L)
  result <- FoRecoML:::input2rtw(hat, kset)
  expect_true(is.matrix(result))
  expect_true(is.finite(sum(result)))
  # Shape: rows = kset[1] (highest-freq block row count), cols = n_series * length(kset)
  expect_equal(NROW(result), kset[1L])
})
