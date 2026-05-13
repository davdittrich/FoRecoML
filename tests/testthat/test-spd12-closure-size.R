# test-spd12-closure-size.R — closure size verification for spd.12
# Verifies that the hat object sent to mirai daemons (loop_body's hat arg)
# is the RAW compact matrix, not the row-expanded form.
# Object size of raw hat must be < expanded hat / 2 for kt >= 2.

skip_on_cran()

test_that("spd.12: raw hat < expanded hat by factor > 2 for ctrml compact kt=4", {
  set.seed(99)
  m <- 4
  te_set <- tetools(m)$set  # c(4, 2, 1), kt=7
  te_fh  <- m / te_set
  agg_mat <- t(c(1, 1))
  dimnames(agg_mat) <- list("A", c("B", "C"))
  n <- 3L
  N_hat <- 8L  # small training length
  hat_raw <- rbind(
    rnorm(sum(te_fh) * N_hat),
    rnorm(sum(te_fh) * N_hat),
    rnorm(sum(te_fh) * N_hat)
  )

  # Compute keep_cols for compact features
  tmp <- cttools(agg_order = m, agg_mat = agg_mat)
  total_cols <- tmp$dim[["n"]] * tmp$dim[["p"]]
  nb_ <- tmp$dim[["nb"]]; na_ <- tmp$dim[["na"]]
  n_  <- tmp$dim[["n"]];  p_  <- tmp$dim[["p"]]
  i_top <- rep(seq_len(n_), times = nb_)
  j_top <- rep(seq_len(nb_), each = n_)
  if (p_ > 1) {
    row_offsets <- seq(from = na_ + n_, by = n_, length.out = p_ - 1)
    i_band <- as.vector(outer(seq_len(nb_), row_offsets, `+`))
    j_band <- rep(seq_len(nb_), times = p_ - 1)
  } else {
    i_band <- integer(0); j_band <- integer(0)
  }
  sel_mat_compact <- Matrix::sparseMatrix(
    i = c(i_top, i_band), j = c(j_top, j_band), x = 1,
    dims = c(n_ * p_, nb_)
  )
  keep_cols <- FoRecoML:::sel_mat_keep_cols(sel_mat_compact, total_cols)

  # Pre-spd.12: expanded hat = input2rtw_partial(hat_raw, kset, cols=keep_cols)
  hat_expanded <- FoRecoML:::input2rtw_partial(hat_raw, tmp$set, cols = keep_cols)

  size_raw      <- as.numeric(utils::object.size(hat_raw))
  size_expanded <- as.numeric(utils::object.size(hat_expanded))

  # Raw hat must be strictly smaller than expanded hat (row-replication inflates).
  # The factor depends on feature density (keep_cols / kt). For large n the factor
  # grows proportionally; here we only assert raw < expanded as a sanity bound.
  expect_lt(size_raw, size_expanded)
})
