#' Extract the reconciled model from a reconciliation result
#'
#' @description
#' Extract the fitted reconciled model(s) from a reconciliation
#' function's output (e.g., [csrml], [terml] and [ctrml]).
#' The model can be reused for forecast reconciliation in the
#' reconciliation functions.
#'
#' @param reco An object returned by a reconciliation function
#' (e.g., the result of [csrml], [terml] and [ctrml]).
#'
#' @return A named list with reconciliation information:
#'   \item{\code{sel_mat}}{Features used (e.g., the selected feature
#'   matrix or indices).}
#'   \item{\code{fit}}{List of reconciled models.}
#'   \item{\code{approach}}{The learning approach used (e.g., \code{"xgboost"},
#'   \code{"lightgbm"}, \code{"randomForest"}, \code{"mlr3"}).}
#'
#' @examples
#' # agg_mat: simple aggregation matrix, A = B + C
#' agg_mat <- t(c(1,1))
#' dimnames(agg_mat) <- list("A", c("B", "C"))
#'
#' # N_hat: dimension for the most aggregated training set
#' N_hat <- 100
#'
#' # ts_mean: mean for the Normal draws used to simulate data
#' ts_mean <- c(20, 10, 10)
#'
#' # hat: a training (base forecasts) feautures matrix
#' hat <- matrix(
#'   rnorm(length(ts_mean)*N_hat, mean = ts_mean),
#'   N_hat, byrow = TRUE)
#' colnames(hat) <- unlist(dimnames(agg_mat))
#'
#' # obs: (observed) values for bottom-level series (B, C)
#' obs <- matrix(
#'   rnorm(length(ts_mean[-1])*N_hat, mean = ts_mean[-1]),
#'   N_hat, byrow = TRUE)
#' colnames(obs) <- colnames(agg_mat)
#'
#' # h: base forecast horizon
#' h <- 2
#'
#' # base: base forecasts matrix
#' base <- matrix(
#'   rnorm(length(ts_mean)*h, mean = ts_mean),
#'   h, byrow = TRUE)
#' colnames(base) <- unlist(dimnames(agg_mat))
#' # `reco` is the result of a reconciliation call:
#' reco <- csrml(base = base, hat = hat, obs = obs, agg_mat = agg_mat)
#'
#' mdl <- extract_reconciled_ml(reco)
#' mdl
#'
#' @export
extract_reconciled_ml <- function(reco) {
  foreco_attr <- attr(reco, "FoReco")
  if (!is.null(foreco_attr) && inherits(foreco_attr$fit, "rml_g_fit")) {
    return(foreco_attr$fit)
  }

  if (inherits(reco, "rml_fit")) {
    cli_inform(
      "Input {.arg reco} is already an {.cls rml_fit}; returning it unchanged."
    )
    return(reco)
  }

  info <- tryCatch(
    suppressWarnings(recoinfo(reco, verbose = FALSE)),
    error = function(e) {
      cli_warn(
        "Failed to retrieve reconciliation info: {conditionMessage(e)}",
        call = NULL
      )
      return(NULL)
    }
  )

  if (is.null(info) || is.null(info$fit)) {
    cli::cli_warn("No reconciled model information available.", call = NULL)
    return(invisible(NULL))
  }

  return(info$fit)
}

#' @export
#' @method print rml_fit
print.rml_fit <- function(x, ...) {
  cat("----- Reconciled models -----\n")
  cat("Framework:", x$framework, "\n")
  cat("Features:", x$features, "\n")
  cat("Approach:", x$approach, "\n")
  cat("  Models:", length(x$fit), "\n")
}

# T6 — Tri-mode disk checkpoint helpers.
# `checkpoint` argument semantics (resolved here into NULL or a directory path):
#   - FALSE / "false": never enable; keep fits in memory (legacy).
#   - TRUE / "true": always enable; session-scoped tempdir.
#   - "auto" (default): enable when estimated peak memory > 80% of available
#     physical RAM; session-scoped tempdir. Falls back to OFF on platforms
#     where available_ram_bytes() returns NA.
#   - character path: always enable; uses that exact directory (persistent,
#     suitable for fit reuse across sessions).
resolve_checkpoint <- function(checkpoint, hat, approach, p) {
  if (identical(checkpoint, FALSE) || identical(checkpoint, "false")) {
    return(NULL)
  }
  if (identical(checkpoint, TRUE) || identical(checkpoint, "true")) {
    return(checkpoint_session_dir())
  }
  if (is.character(checkpoint) && length(checkpoint) == 1) {
    if (checkpoint == "auto") {
      est <- estimate_peak_bytes(hat, approach, p)
      avail <- available_ram_bytes()
      if (is.finite(est) && is.finite(avail) && est > 0.8 * avail) {
        return(checkpoint_session_dir())
      }
      return(NULL)
    }
    return(normalizePath(checkpoint, mustWork = FALSE))
  }
  cli_abort(
    paste0(
      "`checkpoint` must be 'auto' (default), TRUE/'true', ",
      "FALSE/'false', or a directory path."
    ),
    call = NULL
  )
}

checkpoint_session_dir <- function() {
  # Use tempfile() (which does NOT consume the user's RNG state) to build a
  # unique session-scoped directory name. We MUST NOT call sample() or any
  # other R-RNG function here, because that would advance .Random.seed and
  # cause checkpoint=TRUE predictions to diverge from checkpoint=FALSE for
  # stochastic learners (randomForest, mlr3+ranger) even under identical
  # upstream set.seed().
  d <- tempfile(pattern = "foreco_ckpt_", tmpdir = tempdir())
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  if (!dir.exists(d)) {
    cli_abort("Failed to create checkpoint directory {.path {d}}.")
  }
  d
}

# Cheap conservative estimate of simultaneous peak bytes when all p models are
# held in memory. `per_model` multipliers are order-of-magnitude correct:
# randomForest stores full forests (large); mlr3 wraps learners + a copy of
# training data; xgboost/lightgbm trees are compact relative to hat bytes.
estimate_peak_bytes <- function(hat, approach, p) {
  if (is.null(hat)) {
    return(0)
  }
  # T5 overflow audit: cast every multiplicand to double to prevent integer
  # overflow on large hat dims (NROW*NCOL > .Machine$integer.max occurs at
  # ~46k x 46k). p is also cast — p can reach 100k+ on dense ct hierarchies.
  hat_bytes <- as.numeric(NROW(hat)) * as.numeric(NCOL(hat)) * 8
  per_model <- switch(
    approach,
    "randomForest" = 5,
    "ranger"       = 3,
    "mlr3"         = 3,
    "xgboost"      = 0.5,
    "lightgbm"     = 0.3,
    1
  )
  models <- hat_bytes * per_model * as.numeric(p)
  copies <- hat_bytes * 3
  models + copies
}

# Returns available RAM in bytes, or NA_real_ on unknown OS / parse failure.
# "auto" mode falls back to OFF when this returns NA.
available_ram_bytes <- function() {
  os <- Sys.info()[["sysname"]]
  if (os == "Linux") {
    info <- tryCatch(
      readLines("/proc/meminfo"),
      error = function(e) NULL
    )
    if (is.null(info)) {
      return(NA_real_)
    }
    line <- grep("^MemAvailable:", info, value = TRUE)
    if (length(line) == 0) {
      line <- grep("^MemFree:", info, value = TRUE)
    }
    if (length(line) == 0) {
      return(NA_real_)
    }
    kb <- as.numeric(regmatches(line, regexpr("\\d+", line)))
    return(kb * 1024)
  }
  if (os == "Darwin") {
    out <- tryCatch(system("vm_stat", intern = TRUE),
                    error = function(e) NULL, warning = function(w) NULL)
    if (is.null(out)) return(NA_real_)
    # Page size (typically 4096 or 16384 on Apple Silicon)
    ps_line <- grep("page size of", out, value = TRUE)
    ps <- if (length(ps_line)) {
      as.numeric(regmatches(ps_line, regexpr("\\d+", ps_line)))
    } else 4096
    get_pages <- function(label) {
      l <- grep(label, out, value = TRUE, fixed = TRUE)
      if (!length(l)) return(0)
      as.numeric(sub("\\.$", "", regmatches(l, regexpr("\\d+", l))))
    }
    free  <- get_pages("Pages free:")
    inact <- get_pages("Pages inactive:")
    spec  <- get_pages("Pages speculative:")
    return((free + inact + spec) * ps)
  }
  if (os == "Windows") {
    out <- tryCatch(
      system("wmic OS get FreePhysicalMemory /Value", intern = TRUE),
      error = function(e) NULL,
      warning = function(w) NULL
    )
    if (is.null(out)) {
      return(NA_real_)
    }
    line <- grep("FreePhysicalMemory=", out, value = TRUE)
    if (length(line) == 0) {
      return(NA_real_)
    }
    kb <- as.numeric(sub(".*=", "", line))
    return(kb * 1024)
  }
  NA_real_
}

# Per-approach serializer dispatch.
# xgboost: serialize raw bytes via xgb.save.raw + qs_save (qs_save on a live
#   xgb.Booster fails because of the external C++ pointer).
# lightgbm: use native lgb.save (the same C++ pointer constraint applies).
# everything else (randomForest, mlr3): qs2 round-trip is safe (verified).
serialize_fit <- function(model, dir, i, approach) {
  ext <- switch(
    approach,
    "lightgbm" = ".lgb",
    "catboost"  = ".cbm",
    ".qs2"
  )
  path <- file.path(dir, sprintf("fit_%d%s", i, ext))
  # B7: cap qs2 nthreads at min(detectCores(), 4L). Capping at 4 avoids
  # diminishing returns from contention on small per-series fits while still
  # giving the parallel compressor enough lanes to saturate disk I/O.
  nth <- min(parallel::detectCores(logical = TRUE), 4L)
  if (!is.finite(nth) || nth < 1L) nth <- 1L
  switch(
    approach,
    "xgboost"  = qs2::qs_save(xgboost::xgb.save.raw(model), path, nthreads = nth),
    "lightgbm" = lightgbm::lgb.save(model, filename = path),
    "catboost"  = catboost::catboost.save_model(model, path),
    qs2::qs_save(model, path, nthreads = nth)
  )
  path
}

deserialize_fit <- function(path, approach) {
  switch(
    approach,
    "xgboost"  = xgboost::xgb.load.raw(qs2::qs_read(path)),
    "lightgbm" = lightgbm::lgb.load(filename = path),
    "catboost"  = catboost::catboost.load_model(path),
    qs2::qs_read(path)
  )
}

# Lazy-load accessor used by rml() at predict time. If `fit[[i]]` is a stored
# path (string), reload from disk; otherwise return the in-memory model.
get_fit_i <- function(obj, i) {
  f <- obj$fit[[i]]
  if (is.character(f) && length(f) == 1) {
    return(deserialize_fit(f, obj$approach))
  }
  f
}

# Rombouts et al. (2025) matrix-form
input2rtw <- function(x, kset) {
  x <- FoReco::FoReco2matrix(x, kset)
  x <- lapply(1:length(kset), function(i) {
    if (NCOL(x[[i]]) > 1) {
      tmp <- matrix(rep(as.vector(x[[i]]), each = kset[i]), nrow = NROW(x[[i]]) * kset[i], ncol = NCOL(x[[i]]))
      #colnames(tmp) <- paste0(colnames(tmp), "_", kset[i])
    } else {
      tmp <- rep(x[[i]], each = kset[i])
    }
    tmp
  })
  do.call(cbind, rev(x))
}

# Slice-first variant of input2rtw: materializes ONLY the columns whose global
# index (in the full row-replicated output) is in `cols`. Output column order
# matches `cols`. Full-cols invocation is byte-identical to input2rtw().
# `cols` must be unique strictly-increasing integers (use `which()` output).
input2rtw_partial <- function(x, kset, cols) {
  if (length(cols) == 0L) {
    n_rows <- NROW(FoReco::FoReco2matrix(x, kset)[[1]]) * kset[1]
    return(matrix(numeric(0), nrow = n_rows, ncol = 0))
  }
  parts <- FoReco::FoReco2matrix(x, kset)
  # do.call(cbind, rev(parts)) ordering => reversed level order
  ncol_per_level_rev <- vapply(rev(parts), NCOL, integer(1))
  col_offsets <- c(0L, cumsum(ncol_per_level_rev)) # length = length(kset)+1
  out_blocks <- lapply(seq_along(ncol_per_level_rev), function(j) {
    lvl_idx_full <- length(kset) - j + 1L # original (non-reversed) level index
    in_range <- cols > col_offsets[j] & cols <= col_offsets[j + 1]
    if (!any(in_range)) {
      return(NULL)
    }
    local <- cols[in_range] - col_offsets[j]
    block <- if (NCOL(parts[[lvl_idx_full]]) > 1) {
      parts[[lvl_idx_full]][, local, drop = FALSE]
    } else {
      parts[[lvl_idx_full]]
    }
    k <- kset[lvl_idx_full]
    expanded <- if (NCOL(block) > 1) {
      matrix(rep(as.vector(block), each = k), nrow = NROW(block) * k, ncol = NCOL(block))
    } else {
      rep(block, each = k)
    }
    list(mat = expanded, global_cols = cols[in_range])
  })
  out_blocks <- Filter(Negate(is.null), out_blocks)
  mat <- do.call(cbind, lapply(out_blocks, `[[`, "mat"))
  global_cols <- unlist(lapply(out_blocks, `[[`, "global_cols"))
  mat[, order(match(global_cols, cols)), drop = FALSE]
}

# T3 (spd.12/13): per-iteration column-slice helpers for deferred hat/base
# expansion in mfh paths. Materializes the full hmat once per call but returns
# only the requested columns; the full expansion is never held across the rml
# training loop. Memory peak drops from O(full_hmat + p models) to
# O(1 series block + 1 model) per iteration.
#
# mat2hmat_cols: cross-temporal (ctrml mfh). x is n x (h*kt). Returns h x len(cols).
mat2hmat_cols <- function(x, h, kset, n, cols) {
  mat2hmat(x, h = h, kset = kset, n = n)[, cols, drop = FALSE]
}

# vec2hmat_cols: temporal (terml mfh). vec is length h*kt. Returns h x len(cols).
vec2hmat_cols <- function(vec, h, kset, cols) {
  vec2hmat(vec = vec, h = h, kset = kset)[, cols, drop = FALSE]
}

# Build a column-replicated sparse 0/1 indicator matrix:
# `length(idx) x n_cols` sparse matrix where each column equals `idx`.
#
# Cheaper than `Matrix(rep(idx, n_cols), ncol = n_cols, sparse = TRUE)`
# when `idx` is sparse (nnz << length(idx)): allocates two integer
# index vectors of size `nnz * n_cols`, vs the legacy form's
# `length(idx) * n_cols` doubles in the data vector.
sparse_col_replicate <- function(idx, n_cols) {
  nz <- which(idx != 0)
  Matrix::sparseMatrix(
    i = rep(nz, times = n_cols),
    j = rep(seq_len(n_cols), each = length(nz)),
    x = 1,
    dims = c(length(idx), n_cols)
  )
}

# Compute keep_cols (global column indices of features actually used) from
# sel_mat. Mirrors the T4 keep_cols logic in rml().
sel_mat_keep_cols <- function(sel_mat, ncol_full) {
  if (length(sel_mat) == 1) {
    return(seq_len(ncol_full))
  }
  if (is(sel_mat, "sparseVector")) {
    return(which(as.numeric(sel_mat) != 0))
  }
  if (NCOL(sel_mat) == 1) {
    return(which(as.numeric(sel_mat[, 1]) != 0))
  }
  which(Matrix::rowSums(sel_mat != 0) > 0)
}

# Reconcile using machine learning models class
#
# This function creates an object of class \code{reconcile_ml} that contains the
# necessary components to perform forecast reconciliation using machine learning
# models.
#
# @param features Character string specifying which features are used for model
#   training.
# @param features_size Optional numeric vector specifying the size of the
#   feature set to be used for model training.
# @param framework Character string specifying the reconciliation framework to
#   be used. Options include "\code{cs}" for cross-sectional, "\code{te}" for
#   temporal, and "\code{ct}" for cross-temporal.
# @param sel_mat Selection matrix/vector to be used to select the features
#   for each variable. It's strickly related to the \code{features} argument.
# @inheritParams ctrml
#
# @returns Returns a fitted object ([reconcile_ml] class) that can be used
#   for reconciliation.
#
# @export
new_rml_fit <- function(
  fit,
  agg_mat = NULL,
  agg_order = NULL,
  tew = NULL,
  sel_mat = NULL,
  approach = NULL,
  framework = NULL,
  features = NULL,
  features_size = NULL,
  block_sampling = NULL,
  checkpoint_dir = NULL
) {
  framework <- match.arg(framework, choices = c("cs", "te", "ct"))
  structure(
    list(
      agg_mat = agg_mat,
      agg_order = agg_order,
      tew = tew,
      fit = fit,
      sel_mat = sel_mat,
      approach = approach,
      framework = framework,
      features = features,
      features_size = features_size,
      block_sampling = block_sampling,
      checkpoint_dir = checkpoint_dir
    ),
    class = "rml_fit"
  )
}

# Identify feature columns of `hat` with NA rate >= threshold.
# Returns logical vector of length NCOL(hat).
# Uses sum/threshold*NROW form to preserve TRUE-everywhere semantics at NROW=0.
na_col_mask <- function(hat, threshold = 0.75) {
  colSums(is.na(hat)) >= threshold * NROW(hat)
}

# T7.1 — Global ML stacking normalization helpers -------------------------

#' Normalize a stacked feature matrix
#'
#' @param X numeric matrix (rows = observations, cols = features)
#' @param method normalization method: "zscore" (default) or "robust"
#' @param scale_fn scale estimator when method="robust". One of
#'   "gmd", "mad_scaled", "qn", "sn", "iqr_scaled", "sd_c4"
#' @return list with components:
#'   \item{X_norm}{normalized matrix, same dims as X}
#'   \item{center}{numeric vector, per-column center (mean or median)}
#'   \item{scale}{numeric vector, per-column scale; 1 where scale < .Machine$double.eps}
#' @examples
#' X <- matrix(rnorm(60), nrow = 20, ncol = 3)
#' result <- normalize_stack(X, method = "zscore")
#' stopifnot(ncol(result$X_norm) == 3L)
#'
#' # Robust normalization with Gini Mean Difference scale
#' result_r <- normalize_stack(X, method = "robust", scale_fn = "gmd")
#' @export
normalize_stack <- function(X, method = c("zscore", "robust"), scale_fn = "gmd") {
  method <- match.arg(method)

  if (method == "zscore") {
    center <- colMeans(X, na.rm = TRUE)
    scale  <- apply(X, 2, sd, na.rm = TRUE)
  } else {
    # robust method
    scale_fn <- match.arg(scale_fn, c("gmd", "mad_scaled", "qn", "sn", "iqr_scaled", "sd_c4"))
    center <- apply(X, 2, median, na.rm = TRUE)
    scale  <- apply(X, 2, .robscale_fn(scale_fn), na.rm = TRUE)
  }

  # Zero-scale guard: constant series -> divide by 1 (leave untouched)
  zero_mask <- scale < .Machine$double.eps
  scale[zero_mask] <- 1

  X_norm <- sweep(sweep(X, 2, center, "-"), 2, scale, "/")
  list(X_norm = X_norm, center = center, scale = scale)
}

apply_norm_params <- function(X_new, norm_params) {
  if (is.null(norm_params)) return(X_new)
  sc <- norm_params$scale
  sc[!is.finite(sc) | sc < .Machine$double.eps] <- 1
  X_new <- sweep(X_new, 2, norm_params$center, "-")
  X_new <- sweep(X_new, 2, sc, "/")
  X_new
}

# Returns a function that computes the robust scale for a single column vector.
.robscale_fn <- function(scale_fn) {
  switch(scale_fn,
    "gmd" = function(x, na.rm) {
      # Gini mean difference via sorted-order trick: O(n log n), O(n) memory.
      # outer(x, x) is O(n^2) and OOMs for large columns.
      x <- if (na.rm) x[!is.na(x)] else x
      n <- length(x)
      if (n < 2L) return(0)
      x <- sort.int(x)
      w <- 2L * seq_len(n) - n - 1L
      # sorted order: sum_{i<j}(x_j - x_i) = sum_j x_j*(2j-n-1), all terms non-negative
      sum(w * x) / (n * (n - 1L) / 2L)
    },
    "mad_scaled" = function(x, na.rm) {
      mad(x, na.rm = na.rm)   # mad() already has constant=1.4826
    },
    "qn" = function(x, na.rm) {
      if (!requireNamespace("robscale", quietly = TRUE)) {
        cli_abort("{.pkg robscale} required for scale_fn='qn'")
      }
      robscale::qn(if (na.rm) x[!is.na(x)] else x)
    },
    "sn" = function(x, na.rm) {
      if (!requireNamespace("robscale", quietly = TRUE)) {
        cli_abort("{.pkg robscale} required for scale_fn='sn'")
      }
      robscale::sn(if (na.rm) x[!is.na(x)] else x)
    },
    "iqr_scaled" = function(x, na.rm) {
      IQR(x, na.rm = na.rm) / 1.3490  # consistent with normal
    },
    "sd_c4" = function(x, na.rm) {
      n <- sum(!is.na(x))
      sd(x, na.rm = na.rm) / (1 - 1 / (4 * n))  # c4 bias correction
    }
  )
}

# Compute validation residuals from a fitted rml_g_fit object.
# Returns a T_valid × p matrix (time × bottom-series) of residuals.
# Aborts when valid_idx is empty (validation_split was 0).
compute_rec_residuals <- function(fit_obj) {
  if (length(fit_obj$valid_idx) == 0L) {
    cli_abort(
      paste0("method = 'rec' with comb requiring residuals needs",
             " validation_split > 0."),
      call = NULL
    )
  }
  # Pass series_id explicitly to prevent the broadcast path from replicating
  # each row p times (which would give preds of length T_valid*p, not T_valid).
  # Convert stored integer indices back to character level names for predict().
  series_ids <- fit_obj$series_id_levels[fit_obj$series_id_int_valid]
  preds     <- predict(fit_obj, newdata = fit_obj$X_valid,
                       series_id = series_ids)
  resid_vec <- fit_obj$y_valid - preds
  if (!is.null(fit_obj$obs_mask_valid) && any(!fit_obj$obs_mask_valid)) {
    resid_vec[!fit_obj$obs_mask_valid] <- NA_real_
  }
  p         <- length(fit_obj$series_id_levels)
  n_valid_per_series <- length(fit_obj$valid_idx) %/% p
  T_valid <- n_valid_per_series
  if (T_valid * p != length(resid_vec)) {
    cli_abort("Validation residual count {length(resid_vec)} not divisible by p={p}.",
              call = NULL)
  }
  # Route each block of residuals to the correct sorted-level column.
  # valid_idx lapply iterates in series_names (original colnames) order, which may
  # differ from series_id_levels (sorted) order. Using series_id_int_valid to
  # assign each block to its correct sorted-index column prevents silent mislabeling
  # when colnames(obs) are not already alphabetically sorted.
  resid_mat <- matrix(NA_real_, nrow = T_valid, ncol = p,
                      dimnames = list(NULL, fit_obj$series_id_levels))
  for (j in seq_len(p)) {
    block_rows  <- seq_len(T_valid) + (j - 1L) * T_valid
    sorted_col  <- fit_obj$series_id_int_valid[block_rows[1L]]
    resid_mat[, sorted_col] <- resid_vec[block_rows]
  }
  resid_mat
}

# Convert FoReco wide CT matrix format to internal stacked representation.
# hat_wide:  n_series × (n_folds × kt)  — row = series, cols = CT features per fold
# obs_wide:  n_bottom × T_monthly       — row = bottom series, cols = monthly obs
# base_wide: n_series × kt              — row = series, cols = CT test features
# agg_mat:   n_agg × n_bottom           — summation matrix (from cstools/cttools)
# agg_order: integer vector             — temporal aggregation orders
#
# Returns list with X_stacked, y_stacked, series_id_int, series_id_levels,
# base_tall, n_folds, n_series, n_bottom.
convert_wide_ct <- function(hat_wide, obs_wide, base_wide, agg_mat, agg_order) {
  m  <- max(agg_order)
  kt <- sum(m / agg_order)   # total CT columns per fold
  n_series <- nrow(hat_wide)
  n_bottom <- nrow(obs_wide)
  n_folds  <- ncol(hat_wide) %/% kt
  T_monthly <- ncol(obs_wide)
  months_per_fold <- T_monthly %/% n_folds

  if (ncol(hat_wide) != n_folds * kt)
    cli_abort("`hat_wide` must have ncol = n_folds × kt = {n_folds * kt}; got {ncol(hat_wide)}.", call = NULL)
  if (ncol(base_wide) != kt)
    cli_abort("`base_wide` must have ncol = kt = {kt}; got {ncol(base_wide)}.", call = NULL)
  if (nrow(agg_mat) != n_series - n_bottom)
    cli_abort("`agg_mat` rows ({nrow(agg_mat)}) must equal n_series - n_bottom = {n_series - n_bottom}.", call = NULL)

  # Derive full observations for all series (upper via aggregation)
  obs_upper <- agg_mat %*% obs_wide   # n_agg × T_monthly
  obs_full  <- rbind(obs_upper, obs_wide)  # n_series × T_monthly
  if (!is.null(rownames(hat_wide))) rownames(obs_full) <- rownames(hat_wide)

  # Series labels (sorted for stable factor coding)
  snames     <- if (!is.null(rownames(hat_wide))) rownames(hat_wide) else paste0("S", seq_len(n_series))
  sid_levels <- sort(unique(snames))

  # Column names for X_stacked: use the first kt colnames of hat_wide (one fold's
  # worth) so that LightGBM sees consistent feature names at train and predict time.
  # Auto-generate if hat_wide has no colnames to avoid empty-string colnames from cbind.
  x_colnames <- if (!is.null(colnames(hat_wide))) {
    colnames(hat_wide)[seq_len(kt)]
  } else {
    paste0("V", seq_len(kt))
  }

  # Build stacked training matrix: one row per (series, fold) pair
  n_rows  <- n_series * n_folds
  X_stack <- matrix(0, n_rows, kt, dimnames = list(NULL, x_colnames))
  y_stack <- numeric(n_rows)
  sid_int <- integer(n_rows)
  row_i   <- 1L
  for (t in seq_len(n_folds)) {
    fold_cols   <- ((t - 1L) * kt + 1L):(t * kt)
    fold_months <- ((t - 1L) * months_per_fold + 1L):(t * months_per_fold)
    for (s in seq_len(n_series)) {
      X_stack[row_i, ] <- hat_wide[s, fold_cols]
      y_stack[row_i]   <- mean(obs_full[s, fold_months])
      sid_int[row_i]   <- match(snames[s], sid_levels)
      row_i <- row_i + 1L
    }
  }

  list(
    X_stacked        = X_stack,
    y_stacked        = y_stack,
    series_id_int    = sid_int,
    series_id_levels = sid_levels,
    n_folds          = n_folds,
    n_series         = n_series,
    n_bottom         = n_bottom,
    # base_tall: n_series × kt, colnames synced with X_stacked for LightGBM compat.
    base_tall        = `colnames<-`(base_wide, x_colnames)
  )
}
