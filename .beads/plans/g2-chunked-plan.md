# G.2 (spd.28): Chunked Incremental Global Training for *_g Functions

Ticket: FoRecoML-1cn (P1)

## I. Context & Objective

**Problem (analyst diagnosis)**: ctrml_g stacks ALL p series at once before training.
For user's case (p=2432, T_obs=72, compact cols=15,597): stacked matrix =
175,104 × 15,597 × 8B = **21.8 GB** → OOM on 16 GB RAM before lgb.train() is called.

**Root cause asymmetry**:
- ctrml (per-series): N=72 rows/booster → LightGBM never engages num_threads internally.
  Parallelism must come from n_workers (cross-daemon), which adds serialization overhead.
- ctrml_g (global): num_threads=9 engages on 175K-row booster. BUT OOMs at user scale.

**Solution**: Train on batches of `batch_size` series at a time (e.g., 200 series →
14.4K rows, 1.8 GB), warm-starting each batch from the previous model via
`init_model` (lightgbm/xgboost) or equivalent (catboost). Final model sees all
data incrementally; num_threads engages on each batch.

**Out**: `batch_size` param added to `csrml_g/terml_g/ctrml_g`. Auto-computes
from available RAM. Incremental warm-start for lightgbm/xgboost/catboost.
ranger/mlr3 abort when stacked matrix exceeds RAM threshold (no warm-start API).

## II. Input Specification

- R/csrml.R: csrml_g() + csrml_g_fit() if needed
- R/terml.R: terml_g()
- R/ctrml.R: ctrml_g()
- R/reco_ml.R: train_incremental_global() new internal helper + rml_g.* update for nrounds splitting

## III. Constraints & Guards

| Type | Guard |
|------|-------|
| **Architecture** | Training in k batches of batch_size series each. Total nrounds split: each batch gets `ceiling(nrounds / k_batches)` rounds. Final model has ≈ nrounds trees total. lightgbm/xgboost/catboost only. |
| **batch_size="auto"** | `floor(0.2 × available_ram_bytes() / (T_obs × active_ncol × 8))`, clamped to [1, p]. T_obs = NROW(obs) / p (obs rows per series). |
| **Single batch (batch_size ≥ p)** | Reduces to current behavior. Output BYTE-IDENTICAL to unchunked ctrml_g (single call, no warm-start needed). |
| **Multi-batch** | Output is NOT byte-identical to unchunked (different data-presentation order in boosting). Test: RMSE within 1.05× of unchunked on same fixture. |
| **ranger/mlr3** | No warm-start API → cli_abort if `p × T_obs × active_ncol × 8 > 0.5 × available_ram_bytes()`. Document: use lightgbm/xgboost for large-scale global training. |
| **Normalization** | Per-series norm_params computed per-batch (each series appears in exactly one batch). Accumulated into a list of p params across batches. Returned in fit$norm_params. Predict-reuse reads existing fit$norm_params unchanged. |
| **nrounds splitting invariant** | `n_per_batch = ceiling(nrounds / n_batches)`. Total trees after k batches = k × n_per_batch ≈ nrounds. Document that multi-batch nrounds may slightly exceed target due to ceiling. |
| **Boundary** | Only R/csrml.R, R/terml.R, R/ctrml.R, R/reco_ml.R (new helper). rml_g.* methods untouched (warm-start lives in the wrappers, not dispatch methods). |
| **Audit** | Single-batch: byte-identical to current ctrml_g. Multi-batch: RMSE ≤1.05× of unchunked on small fixture. |

## IV. Step-by-Step Logic

### Step 1: New internal helper train_incremental_global() in R/reco_ml.R

```r
# Trains (or warm-starts) a global model on one batch of stacked data.
# Returns updated global_model.
# Only called when batch_size < p; otherwise rml_g() is called directly (no helper).
train_incremental_global <- function(approach, X_batch, y_batch, series_id_batch,
                                      global_model, n_rounds, params, seed) {
  class(approach) <- c(class(approach), approach)
  UseMethod("train_incremental_global", approach)
}

train_incremental_global.lightgbm <- function(approach, X_batch, y_batch,
                                               series_id_batch, global_model,
                                               n_rounds, params, seed) {
  X_mat <- cbind(X_batch, series_id = as.integer(series_id_batch))
  colnames(X_mat) <- c(paste0("f", seq_len(ncol(X_batch))), "series_id")
  # CRITICAL: free_raw_data = FALSE required for init_model warm-start.
  # free_raw_data=TRUE causes "cannot set predictor after free raw data" error.
  dset <- lgb.Dataset(X_mat, label = y_batch, free_raw_data = FALSE)
  lgb.train(params = params, data = dset, nrounds = n_rounds,
             init_model = global_model)
}

train_incremental_global.xgboost <- function(approach, X_batch, y_batch,
                                              series_id_batch, global_model,
                                              n_rounds, params, seed) {
  X_mat <- cbind(X_batch, series_id = as.integer(series_id_batch))
  colnames(X_mat) <- c(paste0("f", seq_len(ncol(X_batch))), "series_id")
  dm <- xgb.DMatrix(X_mat, label = y_batch)
  xgb.train(params = params, data = dm, nrounds = n_rounds,
             xgb_model = global_model)
}

train_incremental_global.catboost <- function(approach, X_batch, y_batch,
                                               series_id_batch, global_model,
                                               n_rounds, params, seed) {
  X_mat <- cbind(X_batch, series_id = as.integer(series_id_batch))
  colnames(X_mat) <- c(paste0("f", seq_len(ncol(X_batch))), "series_id")
  cat_idx <- ncol(X_mat) - 1L  # 0-based
  pool <- catboost::catboost.load_pool(X_mat, label = y_batch,
                                        cat_features = cat_idx)
  catboost::catboost.train(pool,
    params = modifyList(params, list(iterations = n_rounds)),
    init_model = global_model)
}

train_incremental_global.default <- function(approach, ...) {
  cli::cli_abort("Incremental global training not supported for this approach.")
}
```

### Step 2: Add batch_size param to ctrml_g/terml_g/csrml_g signatures

```r
ctrml_g <- function(..., batch_size = "auto", ...) { ... }
```

Add `@param batch_size Integer or "auto". Number of series per training batch.
"auto" computes from available RAM (target 20% per batch). Single batch (batch_size >= p)
is byte-identical to previous ctrml_g behavior. Multi-batch uses incremental warm-start;
only supported for lightgbm, xgboost, catboost.`

### Step 3: Chunked training logic in each *_g wrapper

Insert between feature-matrix construction and the rml_g() call:

```r
# After: X_list built (per-series feature matrices)
# Before: normalize_stack + rml_g call

# --- Resolve batch_size ---
if (identical(batch_size, "auto")) {
  # IMPORTANT: obs orientation varies per wrapper.
  # ctrml_g: obs enters as nb×N, is transposed internally → T_obs = NCOL(obs) = N*m
  # terml_g: obs is N_obs vector length nb*N → T_obs = length(obs)/nb
  # csrml_g: obs is N×nb → T_obs = NROW(obs)
  # Use NCOL(obs) for ctrml_g (original pre-transpose has N*m cols = training obs per series).
  # Each wrapper computes T_obs from its own local obs shape BEFORE any transpose.
  T_obs <- NCOL(obs)  # ctrml_g: orig obs is nb×(N*m), so NCOL = N*m = T_obs
  bytes_per_series <- T_obs * active_ncol * 8
  ram_budget <- 0.2 * available_ram_bytes()
  batch_size <- max(1L, min(nb, as.integer(ram_budget / bytes_per_series)))
}

# ranger/mlr3 OOM guard
if (approach %in% c("ranger", "mlr3")) {
  stacked_bytes <- nb * T_obs * active_ncol * 8
  if (stacked_bytes > 0.5 * available_ram_bytes()) {
    cli::cli_abort(c(
      "Stacked matrix ({format(stacked_bytes/1e9, digits=2)} GB) exceeds 50% of available RAM.",
      "i" = "Use approach = 'lightgbm' or 'xgboost' for large-scale global training.",
      "i" = "Alternatively, reduce features (features='compact') or p."
    ))
  }
}

if (batch_size >= nb) {
  # --- Single batch: current behavior ---
  X_stack <- do.call(rbind, X_list)
  y_stack <- as.vector(obs)
  series_id_vec <- rep(seq_len(nb), each = T_obs)
  norm_out <- normalize_stack(X_stack, series_id_vec, normalize, scale_estimator)
  g_result <- rml_g(approach, norm_out$X_norm, y_stack, series_id_vec,
                     Xtest_stack, fine_tune_rounds, fit, seed,
                     norm_params = norm_out$params, params = params)

} else {
  # --- Multi-batch incremental training ---
  batches <- split(seq_len(nb), ceiling(seq_len(nb) / batch_size))
  n_batches <- length(batches)
  n_per_batch <- ceiling(nrounds / n_batches)  # nrounds from params

  global_model <- NULL
  all_norm_params <- vector("list", nb)
  names(all_norm_params) <- as.character(seq_len(nb))

  cli::cli_inform("ctrml_g: incremental training over {n_batches} batches of {batch_size} series.")

  for (b_idx in seq_along(batches)) {
    batch_series <- batches[[b_idx]]
    X_batch <- do.call(rbind, X_list[batch_series])
    T_obs_vec <- rep(batch_series, each = T_obs)
    y_batch <- as.vector(obs[, batch_series, drop = FALSE])

    norm_out <- normalize_stack(X_batch, T_obs_vec, normalize, scale_estimator)
    for (s in batch_series) all_norm_params[[as.character(s)]] <- norm_out$params[[as.character(s)]]

    global_model <- train_incremental_global(
      approach, norm_out$X_norm, y_batch, T_obs_vec,
      global_model, n_per_batch, params_lgb_or_xgb, seed)
  }

  # Prediction (same as single-batch but uses accumulated global_model)
  Xtest_stack <- ...  # build Xtest_stack from base features
  g_result <- rml_g(approach, X_stack = NULL, y_stack = NULL, series_id_vec,
                     Xtest_stack, fine_tune_rounds = 0L,
                     fit = list(global_model = global_model, series_models = NULL,
                                norm_params = all_norm_params),
                     seed, norm_params = all_norm_params, params = params)
}
```

Note: `nrounds` must be extracted from `params` (e.g., `params$nrounds` for lightgbm) before the dispatch. The `train_incremental_global` methods receive `n_per_batch`, not the full nrounds, to avoid over-training.

### Step 3b: Chunked loop pseudocode for terml_g and csrml_g

**terml_g**: obs is a length-N*nb vector (temporal stacked). T_obs = length(obs) / nb.
```r
# terml_g batch_size resolution:
T_obs <- length(obs) / nb   # temporal obs per series
bytes_per_series <- T_obs * active_ncol * 8
# ...same batch loop as ctrml_g but feature engineering uses tetools/input2rtw...
```
Feature stacking: per-series X_i comes from temporal input2rtw on hat slice.
Post-reconciliation: tebu (same as existing terml).

**csrml_g**: obs is N×nb matrix. T_obs = NROW(obs).
```r
# csrml_g batch_size resolution:
T_obs <- NROW(obs)   # obs per series (cross-sectional)
bytes_per_series <- T_obs * active_ncol * 8
# ...same batch loop, feature engineering via direct hat column slicing...
```
Post-reconciliation: csbu (same as existing csrml).

Both follow the **exact same batch loop structure** as ctrml_g (Step 3): `split(seq_len(nb), ceiling(seq_len(nb) / batch_size))` → per-batch normalize + train_incremental_global + accumulate norm_params.

### Step 4: Tests (tests/testthat/test-g2-chunked.R)

1. **Single batch byte-identical** (lightgbm): `ctrml_g(batch_size=nb)` → `expect_equal(bts, bts_no_chunk, tolerance=0)`. withr::defer cache cleanup.

2. **Multi-batch RMSE quality**: small fixture (nb=4, T=20, m=4). `batch_size=2` (2 batches of 2 series). RMSE ratio ≤1.05 vs single-batch.
   ```r
   r_chunk <- ctrml_g(batch_size=2, approach="lightgbm", params=list(nrounds=20L, nthread=1L))
   r_full  <- ctrml_g(batch_size=nb, ...)
   rmse_chunk <- sqrt(mean((r_chunk$bts - truth)^2))
   rmse_full  <- sqrt(mean((r_full$bts  - truth)^2))
   expect_lte(rmse_chunk / rmse_full, 1.05)
   ```

3. **Auto batch_size < nb at tight RAM**: inject `.rml_cache$ram_bytes = 1L` → `batch_size` resolves to 1. withr::defer cleanup. `expect_no_error(ctrml_g(batch_size="auto", ...))`.

4. **ranger OOM abort**: `expect_error(ctrml_g(approach="ranger", batch_size="auto", ...), regexp="Stacked matrix.*exceeds 50%")`. Inject small RAM via cache_swap.

5. **Warm-start tree count** (lightgbm; requires lightgbm installed):
   ```r
   test_that("G.2: lightgbm warm-start accumulates trees across batches", {
     skip_if_not_installed("lightgbm"); skip_on_cran()
     # nb=4, batch_size=2 → 2 batches; nrounds=10 → 5 rounds/batch → ~10 total
     r <- ctrml_g(batch_size=2, params=list(nrounds=10L, nthread=1L), ...)
     n_trees <- r$fit[[1]]$global_model$num_trees()
     expect_gte(n_trees, 8L)   # at least ~80% of target nrounds
     expect_lte(n_trees, 12L)  # at most ~120% (ceiling rounding)
   })
   ```

### Step 5: devtools::document() + Run tests

All 459 baseline pass + 5+ new G.2 tests.

### Step 6: Commit

`perf(rml): chunked incremental global training for *_g (G.2 spd.28 FoRecoML-1cn)`

Body: mechanism, batch_size auto formula, supported backends, test strategy.

## V. Output Schema

```
files_modified: [R/reco_ml.R, R/ctrml.R, R/terml.R, R/csrml.R]
new_helper: train_incremental_global + methods (lightgbm/xgboost/catboost/default)
new_param: batch_size = "auto" in all 3 wrappers
test_file: tests/testthat/test-g2-chunked.R
test_count_pre: 459
test_count_post: >= 464
commit_count: 1
```

## VI. Definition of Done

- [ ] train_incremental_global S3 generic + 4 methods (lightgbm, xgboost, catboost, default) in reco_ml.R
- [ ] free_raw_data = FALSE in train_incremental_global.lightgbm (prevents init_model crash)
- [ ] batch_size = "auto" param in ctrml_g/terml_g/csrml_g with @param doc
- [ ] T_obs computed correctly per wrapper: NCOL(obs) for ctrml_g; length(obs)/nb for terml_g; NROW(obs) for csrml_g
- [ ] Auto batch_size: floor(0.2 × available_ram_bytes() / (T_obs × active_ncol × 8)) clamped [1, nb]
- [ ] ranger/mlr3 OOM guard: cli_abort when stacked > 0.5 × RAM
- [ ] Single-batch path byte-identical to pre-G.2 (test 1)
- [ ] Multi-batch path RMSE ≤1.05× of single-batch (test 2)
- [ ] norm_params accumulated across batches; fit$norm_params covers all p series
- [ ] nrounds splitting: ceiling(nrounds / k_batches) rounds per batch
- [ ] cli_inform message when multi-batch active (shows n_batches + batch_size)
- [ ] Warm-start tree count verified: n_trees ≈ nrounds (test 5, within ±20%)
- [ ] terml_g chunked loop implemented (T_obs=length(obs)/nb; tebu post-reconciliation)
- [ ] csrml_g chunked loop implemented (T_obs=NROW(obs); csbu post-reconciliation)
- [ ] 459 baseline tests unchanged; test_count_post ≥ 464
- [ ] Single commit; conventional commits; no AI attribution

## Risks

- **R1**: nrounds extraction from params differs per backend (lgb: params$nrounds; xgb: params$nrounds; catboost: params$iterations). Extract in wrapper before calling helper. The helper receives `n_per_batch` NOT the full params.
- **R2**: lgb.Dataset `free_raw_data = FALSE` required for `init_model` warm-start. `free_raw_data = TRUE` causes "cannot set predictor after free raw data" error on second batch. This is a VERIFIED bug — always use `free_raw_data = FALSE` in train_incremental_global.lightgbm.
- **R3**: ranger output is NOT byte-identical to unchunked even for single-batch (different internal RNG state). Document: ranger global training is deterministic only with fixed seed; the single-batch byte-identical test (test 1) is lightgbm-specific.
- **R4**: predict-reuse with multi-batch model: rml_g receives fit=existing (the global_model from incremental training). The predict path in rml_g.lightgbm etc. is unchanged — single model predicts on Xtest regardless of how many batches trained it.
