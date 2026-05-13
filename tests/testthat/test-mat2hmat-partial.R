test_that("mat2hmat_partial is byte-identical to mat2hmat column subset", {
  # Test across a variety of (n, h, kset, cols) combinations.
  # Correctness condition: identical(mat2hmat_partial(m, h, k, n, cols),
  #                                   mat2hmat(m, h, k, n)[, cols, drop = FALSE])

  check_equiv <- function(n, h, kset, cols, label = "") {
    m  <- max(kset)
    kt <- sum(m / kset)  # number of temporal aggregation periods (sum of m/k)
    mat <- matrix(rnorm(n * h * kt), nrow = n)
    full <- FoRecoML:::mat2hmat(mat, h = h, kset = kset, n = n)
    expected <- full[, cols, drop = FALSE]
    got      <- FoRecoML:::mat2hmat_partial(mat, h = h, kset = kset, n = n, cols = cols)
    expect_identical(got, expected, label = label)
  }

  set.seed(123)

  # --- n = 1, h = 1, kset = c(1) ---
  check_equiv(n = 1, h = 1, kset = c(1), cols = 1L,               label = "n1h1k1 full")
  check_equiv(n = 1, h = 1, kset = c(1), cols = integer(0),        label = "n1h1k1 empty")

  # --- n = 3, h = 4, kset = c(1) ---
  check_equiv(n = 3, h = 4, kset = c(1), cols = 1:3,              label = "n3h4k1 full")
  check_equiv(n = 3, h = 4, kset = c(1), cols = c(1L),             label = "n3h4k1 single col")
  check_equiv(n = 3, h = 4, kset = c(1), cols = c(2L, 3L),         label = "n3h4k1 two cols")

  # --- n = 1, h = 4, kset = c(1, 2) ---
  check_equiv(n = 1, h = 4, kset = c(1, 2), cols = 1:3,           label = "n1h4k12 full")
  check_equiv(n = 1, h = 4, kset = c(1, 2), cols = c(2L),          label = "n1h4k12 single")
  check_equiv(n = 1, h = 4, kset = c(1, 2), cols = integer(0),     label = "n1h4k12 empty")

  # --- n = 3, h = 4, kset = c(1, 2) ---
  total <- 3 * 3  # n * kt; kt = 1+2=3
  cols_full   <- seq_len(total)
  cols_single <- c(1L)
  cols_range  <- 2:(total - 1)
  cols_rand   <- sort(sample(seq_len(total), size = 4))
  check_equiv(n = 3, h = 4, kset = c(1, 2), cols = cols_full,    label = "n3h4k12 full")
  check_equiv(n = 3, h = 4, kset = c(1, 2), cols = cols_single,  label = "n3h4k12 single")
  check_equiv(n = 3, h = 4, kset = c(1, 2), cols = cols_range,   label = "n3h4k12 range")
  check_equiv(n = 3, h = 4, kset = c(1, 2), cols = cols_rand,    label = "n3h4k12 random")

  # --- n = 7, h = 8, kset = c(1, 2, 4) ---
  total <- 7 * (1 + 2 + 4)  # n * kt; kt = 1+2+4 = 7
  cols_full   <- seq_len(total)
  cols_single <- c(1L)
  cols_empty  <- integer(0)
  cols_range  <- 5:(total - 5)
  cols_rand   <- sort(sample(seq_len(total), size = 10))
  check_equiv(n = 7, h = 8, kset = c(1, 2, 4), cols = cols_full,  label = "n7h8k124 full")
  check_equiv(n = 7, h = 8, kset = c(1, 2, 4), cols = cols_single, label = "n7h8k124 single")
  check_equiv(n = 7, h = 8, kset = c(1, 2, 4), cols = cols_empty,  label = "n7h8k124 empty")
  check_equiv(n = 7, h = 8, kset = c(1, 2, 4), cols = cols_range,  label = "n7h8k124 range")
  check_equiv(n = 7, h = 8, kset = c(1, 2, 4), cols = cols_rand,   label = "n7h8k124 random")

  # --- n = 3, h = 4, kset = c(1, 2, 4) ---
  total <- 3 * 7
  cols_full  <- seq_len(total)
  cols_rand1 <- sort(sample(seq_len(total), size = 7))
  cols_rand2 <- sort(sample(seq_len(total), size = 3))
  check_equiv(n = 3, h = 4, kset = c(1, 2, 4), cols = cols_full,  label = "n3h4k124 full")
  check_equiv(n = 3, h = 4, kset = c(1, 2, 4), cols = cols_rand1, label = "n3h4k124 rand1")
  check_equiv(n = 3, h = 4, kset = c(1, 2, 4), cols = cols_rand2, label = "n3h4k124 rand2")

  # --- n = 1, h = 1, kset = c(1, 3, 6, 12) ---
  # kt = 12/1 + 12/3 + 12/6 + 12/12 = 12+4+2+1 = 19
  total <- 1 * 19
  cols_full  <- seq_len(total)
  cols_rand  <- sort(sample(seq_len(total), size = 8))
  check_equiv(n = 1, h = 1, kset = c(1, 3, 6, 12), cols = cols_full, label = "n1h1k1_3_6_12 full")
  check_equiv(n = 1, h = 1, kset = c(1, 3, 6, 12), cols = cols_rand, label = "n1h1k1_3_6_12 rand")

  # --- n = 3, h = 1, kset = c(1, 3, 6, 12) ---
  total <- 3 * 19
  cols_full  <- seq_len(total)
  cols_rand  <- sort(sample(seq_len(total), size = 15))
  check_equiv(n = 3, h = 1, kset = c(1, 3, 6, 12), cols = cols_full, label = "n3h1k1_3_6_12 full")
  check_equiv(n = 3, h = 1, kset = c(1, 3, 6, 12), cols = cols_rand, label = "n3h1k1_3_6_12 rand")

  # --- n = 7, h = 4, kset = c(1, 3, 6, 12) ---
  total <- 7 * 19
  cols_rand <- sort(sample(seq_len(total), size = 20))
  check_equiv(n = 7, h = 4, kset = c(1, 3, 6, 12), cols = cols_rand, label = "n7h4k1_3_6_12 rand20")

  # --- last column only ---
  check_equiv(n = 3, h = 4, kset = c(1, 2, 4), cols = c(21L), label = "n3h4k124 last-col")

  # --- n = 7, h = 1, kset = c(1, 2, 4) ---
  total <- 7 * 7
  check_equiv(n = 7, h = 1, kset = c(1, 2, 4), cols = seq_len(total), label = "n7h1k124 full")
})

test_that("mat2hmat_partial rejects out-of-range cols", {
  mat <- matrix(rnorm(3 * 7), nrow = 3)
  expect_error(
    FoRecoML:::mat2hmat_partial(mat, h = 1, kset = c(1, 2, 4), n = 3, cols = 0L),
    regexp = "out of range"
  )
  expect_error(
    FoRecoML:::mat2hmat_partial(mat, h = 1, kset = c(1, 2, 4), n = 3, cols = 22L),
    regexp = "out of range"
  )
})
