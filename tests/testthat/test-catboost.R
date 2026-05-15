skip_if_not_installed("catboost")

test_that("(a) catboost: rml.catboost trains and predicts", {
  set.seed(42)
  n_obs  <- 20L
  n_feat <- 5L
  X      <- matrix(rnorm(n_obs * n_feat), n_obs, n_feat)
  y      <- rnorm(n_obs)
  Xtest  <- matrix(rnorm(5L * n_feat), 5L, n_feat)

  result <- FoRecoML:::rml.catboost(y = y, X = X, Xtest = Xtest,
                                    params = list(iterations = 50L, random_seed = 1L))

  expect_type(result$bts, "double")
  expect_length(result$bts, 5L)
  expect_false(is.null(result$fit))
})

test_that("(b) catboost: predict from stored fit only (no Xtest at train time)", {
  skip_if_not_installed("catboost")
  set.seed(7)
  n_obs  <- 15L
  n_feat <- 3L
  X      <- matrix(rnorm(n_obs * n_feat), n_obs, n_feat)
  y      <- rnorm(n_obs)
  Xtest  <- matrix(rnorm(4L * n_feat), 4L, n_feat)

  # Train without Xtest
  trained <- FoRecoML:::rml.catboost(y = y, X = X,
                                     params = list(iterations = 30L, random_seed = 2L))
  expect_null(trained$bts)

  # Predict passing stored fit
  predicted <- FoRecoML:::rml.catboost(fit = trained$fit, Xtest = Xtest)
  expect_type(predicted$bts, "double")
  expect_length(predicted$bts, 4L)
})

test_that("(c) catboost: serialize/deserialize round-trip (.cbm)", {
  skip_if_not_installed("catboost")
  set.seed(99)
  n_obs  <- 20L
  n_feat <- 4L
  X      <- matrix(rnorm(n_obs * n_feat), n_obs, n_feat)
  y      <- rnorm(n_obs)
  Xtest  <- matrix(rnorm(3L * n_feat), 3L, n_feat)

  result <- FoRecoML:::rml.catboost(y = y, X = X, Xtest = Xtest,
                                    params = list(iterations = 30L, random_seed = 3L))

  tmp_dir <- tempfile()
  dir.create(tmp_dir)
  path_saved <- FoRecoML:::serialize_fit(result$fit, tmp_dir, 1L, "catboost")

  expect_true(file.exists(path_saved))
  expect_true(grepl("\\.cbm$", path_saved))

  fit2  <- FoRecoML:::deserialize_fit(path_saved, "catboost")
  bts2  <- FoRecoML:::rml.catboost(fit = fit2, Xtest = Xtest)$bts

  expect_equal(result$bts, bts2, tolerance = 1e-6)
})

test_that("(d) catboost: missing y+X with no fit raises informative error", {
  skip_if_not_installed("catboost")

  expect_error(
    FoRecoML:::rml.catboost(),
    regexp = "Mandatory arguments"
  )
})

test_that("(e) catboost: missing package error is informative", {
  # Only runs when catboost is NOT installed (graceful skip otherwise).
  skip_if(requireNamespace("catboost", quietly = TRUE), "catboost is installed")

  expect_error(
    FoRecoML:::rml.catboost(
      y = rnorm(10), X = matrix(rnorm(10), 10, 1),
      Xtest = matrix(0, 1, 1)
    ),
    regexp = "catboost"
  )
})
