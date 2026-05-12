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
  hat_bytes <- as.numeric(NROW(hat)) * NCOL(hat) * 8
  per_model <- switch(
    approach,
    "randomForest" = 5,
    "mlr3" = 3,
    "xgboost" = 0.5,
    "lightgbm" = 0.3,
    1
  )
  models <- hat_bytes * per_model * p
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
  ext <- if (approach == "lightgbm") ".lgb" else ".qs2"
  path <- file.path(dir, sprintf("fit_%d%s", i, ext))
  switch(
    approach,
    "xgboost" = qs2::qs_save(xgboost::xgb.save.raw(model), path),
    "lightgbm" = lightgbm::lgb.save(model, filename = path),
    qs2::qs_save(model, path)
  )
  path
}

deserialize_fit <- function(path, approach) {
  switch(
    approach,
    "xgboost" = xgboost::xgb.load.raw(qs2::qs_read(path)),
    "lightgbm" = lightgbm::lgb.load(filename = path),
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
      tmp <- apply(x[[i]], 2, rep, each = kset[i])
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
      apply(block, 2, rep, each = k)
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
