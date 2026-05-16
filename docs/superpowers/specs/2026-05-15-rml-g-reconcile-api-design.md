# Design: Global ML Reconciliation API (`*_g` Wrappers)

**Date:** 2026-05-15
**Branch:** fix/rml-g-reconcile-ak9
**Epic:** FoRecoML-ak9

## Problem

The rebuild's `csrml_g()`, `terml_g()`, `ctrml_g()` accept `base` but silently ignore it and return `rml_g_fit` instead of reconciled forecasts. NEWS.md and vignettes promise reconciled output. `predict.rml_g_fit` returns a raw numeric vector with no `csbu`/`tebu`/`ctbu` step — so there is currently no path to reconciled forecasts in the global ML API.

Root cause discovered during prior iteration: the rebuild author treated `hat` as a generic feature matrix and `base` as a separate reconciliation target. The test fixture (`ncol_hat=8`, `n=6`) reflects this. But this leaves `base` semantically homeless and the reconciliation path undefined.

## Design Decisions (locked)

1. **`hat = base forecasts`** (column space = n series). hat at training time and base at test time share the same column layout. This matches per-series `csrml` semantics. Test fixtures using `ncol_hat ≠ n` are buggy and need fixing.
2. **Architecture B**: wrappers own reconciliation. `*_g` returns a reconciled forecast matrix with `attr(., "FoReco")`. `predict.rml_g_fit` stays as a raw prediction utility.
3. **Temporal base shape**: `base = h_hf × kt` where `h_hf = h × max(agg_order)`. Each row = one high-freq prediction period. No internal expansion.

## API

### csrml_g

```r
csrml_g(base, hat, obs, agg_mat, approach = "lightgbm",
        normalize = c("none", "zscore", "robust"),
        scale_fn = "gmd", params = NULL, seed = NULL,
        early_stopping_rounds = 0L, validation_split = 0,
        batch_size = NULL, chunk_strategy = c("sequential", "random"),
        batch_checkpoint_dir = NULL, nrounds_per_batch = 50L, ...)
```

- `hat`: `T_obs × n` numeric matrix. Base forecasts for all n series during training period.
- `obs`: `T_obs × nb` numeric matrix. Bottom-level training observations.
- `base`: `h × n` numeric matrix. Base forecasts at h forecast horizons.
- `agg_mat`: cross-sectional aggregation matrix (`na × nb`).
- **Returns:** `n × h` reconciled forecast matrix with `attr(., "FoReco")` of class `foreco_info`.

### terml_g

```r
terml_g(base, hat, obs, agg_order, ...)  # same param family as csrml_g
```

- `hat`: `T_obs × kt` numeric matrix. Base forecasts at all temporal aggregation levels during training period. `kt = sum(max(agg_order) / agg_order)`.
- `obs`: `T_obs × 1` numeric matrix (single bottom-level series at highest frequency).
- `base`: `h_hf × kt` matrix where `h_hf = h × max(agg_order)`. Base forecasts at h_hf high-freq prediction periods, all temporal levels per row.
- `agg_order`: integer vector of temporal aggregation orders (e.g., `c(4L, 2L, 1L)`).
- **Returns:** named numeric `FoReco::tebu` output (length `h × kt`, names `"k-<order> h-<horizon>"`) with `attr(., "FoReco")`.

### ctrml_g

```r
ctrml_g(base, hat, obs, agg_mat, agg_order, ...)
```

- `hat`: `T_obs × (n × kt)` numeric matrix. Cross-temporal features per training period.
- `obs`: `T_obs × nb` numeric matrix (nb cross-temporal bottom-level series).
- `base`: `h_hf × (n × kt)` numeric matrix.
- **Returns:** `n × (h × kt)` reconciled cross-temporal matrix where `kt = sum(max(agg_order)/agg_order)`. (Note: `h_hf = h × max(agg_order)` is the input row count to `ctbu`, not the output column count.) Carries `attr(., "FoReco")`.

## Internal Pipeline (per-framework, explicit)

Each wrapper implements its own pipeline. No shared switch — the reshape/reconcile shapes differ per framework and conflating them invites bugs.

### csrml_g pipeline
```r
# Pre-conditions
if (missing(base)) cli_abort("Argument {.arg base} is missing, with no default.", call = NULL)
base <- as.matrix(base)
if (!is.numeric(base)) cli_abort("{.arg base} must be numeric.", call = NULL)
if (ncol(base) != ncol(hat))
  cli_abort("{.arg base} must have {ncol(hat)} columns (matching {.arg hat}); got {ncol(base)}.", call = NULL)

# Fit (existing code, unchanged)
fit_obj <- rml_g(approach, hat = hat_norm, obs = obs, ...)
fit_obj$norm_params <- norm_params
fit_obj$agg_mat     <- agg_mat
fit_obj$framework   <- "cs"
fit_obj$ncol_hat    <- ncol(hat)   # NEW: stored for predict.rml_g_fit guard

# Predict + reconcile
base_features <- apply_norm_params(base, fit_obj$norm_params)
bts_vec <- predict(fit_obj, newdata = base_features)
h  <- nrow(base)
nb <- length(fit_obj$series_id_levels)
bts_mat  <- matrix(bts_vec, nrow = h, ncol = nb)   # h × nb (column j = series j)
reco_mat <- FoReco::csbu(bts_mat, agg_mat = fit_obj$agg_mat)  # h × nb in → n × h out (or h × n — verify against FoReco docs)
attr(reco_mat, "FoReco") <- new_foreco_info(list(
  fit = fit_obj, framework = "Cross-sectional",
  forecast_horizon = h, rfun = "csrml_g", ml = approach))
reco_mat
```

### terml_g pipeline
```r
# Pre-conditions
if (missing(base)) cli_abort(...)
base <- as.matrix(base)
if (ncol(base) != ncol(hat))
  cli_abort("{.arg base} must have {ncol(hat)} columns; got {ncol(base)}.", call = NULL)
m <- max(agg_order)
if (nrow(base) %% m != 0)
  cli_abort("{.arg base} must have rows divisible by max(agg_order) = {m}; got {nrow(base)}.", call = NULL)

# terml_g invariant: obs has exactly 1 column (single bottom-level temporal series).
# Multi-series temporal is the cross-temporal case (ctrml_g).
if (ncol(obs) != 1L)
  cli_abort("{.arg obs} for terml_g must have exactly 1 column (single bottom series).", call = NULL)

# Fit
fit_obj <- rml_g(approach, hat = hat_norm, obs = obs, ...)
fit_obj$norm_params <- norm_params
fit_obj$agg_order   <- agg_order
fit_obj$framework   <- "te"
fit_obj$ncol_hat    <- ncol(hat)

# Predict — nb=1 invariant: bts_vec has length nrow(base) = h_hf
base_features <- apply_norm_params(base, fit_obj$norm_params)
bts_vec <- predict(fit_obj, newdata = base_features)   # length h_hf

# tebu accepts a vector of length h_hf and returns a NAMED numeric vector of length h*kt
# (kt = sum(m/agg_order)). Names encode "k-<order> h-<horizon>".
reco_vec <- FoReco::tebu(bts_vec, agg_order = fit_obj$agg_order)
attr(reco_vec, "FoReco") <- new_foreco_info(list(
  fit = fit_obj, framework = "Temporal",
  forecast_horizon = nrow(base) / m,  # number of low-freq horizons
  rfun = "terml_g", ml = approach))
reco_vec   # NAMED numeric vector, length h*kt
```

### ctrml_g pipeline
```r
# Pre-conditions (same as terml_g + agg_mat check)
if (missing(base)) cli_abort(...)
base <- as.matrix(base)
if (ncol(base) != ncol(hat)) cli_abort(...)
m <- max(agg_order)
if (nrow(base) %% m != 0) cli_abort(...)

# Fit
fit_obj <- rml_g(approach, hat = hat_norm, obs = obs, ...)
fit_obj$norm_params <- norm_params
fit_obj$agg_mat     <- agg_mat
fit_obj$agg_order   <- agg_order
fit_obj$framework   <- "ct"
fit_obj$ncol_hat    <- ncol(hat)

# Predict + reshape to nb × hm
base_features <- apply_norm_params(base, fit_obj$norm_params)
bts_vec <- predict(fit_obj, newdata = base_features)
h_hf <- nrow(base)
nb   <- length(fit_obj$series_id_levels)
bts_mat <- matrix(bts_vec, nrow = h_hf, ncol = nb)   # h_hf × nb
reco_mat <- FoReco::ctbu(t(bts_mat), agg_mat = fit_obj$agg_mat,
                          agg_order = fit_obj$agg_order)   # nb × h_hf in
attr(reco_mat, "FoReco") <- new_foreco_info(list(
  fit = fit_obj, framework = "Cross-temporal",
  forecast_horizon = h_hf / m, rfun = "ctrml_g", ml = approach))
reco_mat
```

## predict.rml_g_fit shape guard

Add at top of predict.rml_g_fit (after object/newdata extraction):
```r
if (!is.null(object$ncol_hat) && ncol(newdata) != object$ncol_hat)
  cli_abort("{.arg newdata} must have {object$ncol_hat} columns (matching training {.arg hat}); got {ncol(newdata)}.", call = NULL)
```
Backwards-compatible: skipped when `object$ncol_hat` is absent (legacy fits).

## Validation Guards

| Framework | Required check | cli_abort message |
|-----------|---------------|--------------------|
| csrml_g | `ncol(base) == ncol(hat)` | "`base` must have the same number of columns as `hat` ({ncol(hat)})." |
| terml_g | `ncol(base) == ncol(hat)` AND `nrow(base) %% max(agg_order) == 0` | "`base` rows must be a multiple of `max(agg_order)`." |
| ctrml_g | `ncol(base) == ncol(hat)` AND `nrow(base) %% max(agg_order) == 0` | Same as terml_g. |

## predict.rml_g_fit

Unchanged. Stays as raw-prediction utility returning `numeric` vector. Already verified: `series_id=NULL` broadcast produces series-major order `[s1_h1..hh, s2_h1..hh, ..., snb_h1..hh]`, so `matrix(bts_vec, nrow=h, ncol=nb)` fills correctly.

For advanced users who want to call predict on a fitted model without re-running the wrapper, the fit is accessible via `attr(reco_result, "FoReco")$fit` — and a separate `extract_reconciled_ml()` helper continues to work.

## Test Fixtures (3 frameworks, distinct)

`tests/testthat/test-rml-g-wrappers.R::make_g_fixture()` currently produces `ncol_hat=8, n=6` (incompatible — broken). Replace with three distinct fixtures matching each framework's actual API:

```r
# CS fixture — n total series, h horizons
make_g_fixture_cs <- function(p = 4L, T_obs = 20L, h = 2L) {
  set.seed(99)
  na <- 2L
  n <- na + p
  agg_mat <- matrix(c(1,1,0,0,0,0,1,1), nrow = na, ncol = p,
                    dimnames = list(c("G1","G2"), paste0("S", seq_len(p))))
  obs <- matrix(rnorm(T_obs * p), T_obs, p,
                dimnames = list(NULL, paste0("S", seq_len(p))))
  hat <- matrix(rnorm(T_obs * n), T_obs, n,
                dimnames = list(NULL, c(rownames(agg_mat), colnames(obs))))
  base <- matrix(rnorm(h * n), h, n,
                 dimnames = list(NULL, colnames(hat)))
  list(agg_mat = agg_mat, obs = obs, hat = hat, base = base)
}

# TE fixture — single bottom series, all temporal levels as features
make_g_fixture_te <- function(T_obs = 60L, h = 2L, agg_order = c(4L, 2L, 1L)) {
  set.seed(99)
  m  <- max(agg_order)
  kt <- sum(m / agg_order)           # total temporal levels
  hat  <- matrix(rnorm(T_obs * kt), T_obs, kt,
                 dimnames = list(NULL, paste0("L", seq_len(kt))))
  obs  <- matrix(rnorm(T_obs * 1L),  T_obs, 1L,
                 dimnames = list(NULL, "S1"))
  base <- matrix(rnorm(h * m * kt), h * m, kt,    # h_hf × kt
                 dimnames = list(NULL, colnames(hat)))
  list(agg_order = agg_order, obs = obs, hat = hat, base = base)
}

# CT fixture — nb bottom cross-temporal series
make_g_fixture_ct <- function(p = 2L, T_obs = 60L, h = 2L, agg_order = c(4L, 2L, 1L)) {
  set.seed(99)
  m   <- max(agg_order)
  kt  <- sum(m / agg_order)
  na  <- 1L
  n   <- na + p                       # total cross-sectional series
  ncf <- n * kt                       # cross-temporal feature width
  agg_mat <- matrix(c(1, 1), nrow = na, ncol = p,
                    dimnames = list("G1", paste0("S", seq_len(p))))
  obs  <- matrix(rnorm(T_obs * p), T_obs, p,
                 dimnames = list(NULL, paste0("S", seq_len(p))))
  hat  <- matrix(rnorm(T_obs * ncf), T_obs, ncf,
                 dimnames = list(NULL, paste0("F", seq_len(ncf))))
  base <- matrix(rnorm(h * m * ncf), h * m, ncf,
                 dimnames = list(NULL, colnames(hat)))
  list(agg_mat = agg_mat, agg_order = agg_order, obs = obs, hat = hat, base = base)
}
```

## Additional Required Work (added by design review iter 1)

### Roxygen `@return` rewrites (ak9.2/3/4 scope)
Current docs say `rml_g_fit object with additional fields`. Must change to match new return type:
- `csrml_g`/`ctrml_g`: `numeric matrix with attr("FoReco") of class "foreco_info"`
- `terml_g`: `named numeric vector with attr("FoReco") of class "foreco_info"`

### extract_reconciled_ml() dispatch (ak9.4.5 — NEW TICKET, blocked-by ak9.2/3/4)
`R/utils.R::extract_reconciled_ml()` currently checks `inherits(reco, "rml_fit")` and calls `recoinfo(reco)`. For matrix/vector return with `attr("FoReco")$fit` of class `rml_g_fit`, add a parallel branch at function entry:
```r
foreco_attr <- attr(reco, "FoReco")
if (!is.null(foreco_attr) && inherits(foreco_attr$fit, "rml_g_fit")) {
  return(foreco_attr$fit)
}
```

### DESCRIPTION FoReco version pin (ak9.5.5 — NEW TICKET, can co-land with ak9.5)
Add `FoReco (>= 1.2.1)` to Depends in DESCRIPTION. Pins against silent API breakage if FoReco rewrites csbu/tebu/ctbu.

### NEWS.md correction (extend ak9.7 scope)
Current NEWS.md line 5-8 says: *"Return `rml_g_fit` objects; use `predict.rml_g_fit()` for reconciled forecasts."* — this is now FALSE under Architecture B. Rewrite to:
> "`csrml_g()`, `ctrml_g()` return reconciled forecast matrices with `attr(., 'FoReco')`. `terml_g()` returns a named numeric vector (matching `FoReco::tebu()` output). Access the underlying fit via `extract_reconciled_ml(result)`."

### Vignette update (extend ak9.7 scope)
Audit `vignettes/forecoml.Rmd` for any usage of `result_g$framework`, `result_g$agg_mat`, or similar list-style accessors that assume `rml_g_fit` return. Replace with `extract_reconciled_ml(result_g)` or `attr(result_g, "FoReco")$fit$framework`.

### Ticket split: ak9.8a (fixture rewrite) — NEW TICKET
Per CLAUDE.md "every Task gets own ticket", split off the fixture rewrite as its own ticket. Becomes a precondition for ak9.2/3/4.

## Out of Scope

- `fit=` argument on `*_g` for predict-only path. File as separate ticket post-merge if needed.
- `sntz`/`round` restoration on `*_g`: deferred to ticket ak9.6 decision.
- `tew=` restoration on terml_g/ctrml_g: deferred to ak9.6.

## Breaking Changes (already approved)

- `normalize` argument: logical → character enum (`"none"`/`"zscore"`/`"robust"`).
- `scale_estimator` renamed to `scale_fn`.
- `*_g` return type: `rml_g_fit` → reconciled matrix with `attr("FoReco")`.

Documented in NEWS.md per ticket ak9.7.

## Test Coverage (ak9.8)

1. **Fixture rewrite**: replace `make_g_fixture()` with `make_g_fixture_cs/te/ct()` per spec above.
2. **Update ALL assertions in test-rml-g-wrappers.R**:
   - Line 26 (csrml_g): `expect_s3_class(result, "rml_g_fit")` → `expect_true(is.matrix(result))` + `expect_false(is.null(attr(result, "FoReco")))`
   - Line 44 (terml_g): result is a NAMED numeric vector, not a matrix → `expect_type(result, "double")` + `expect_false(is.null(names(result)))` + FoReco attr check
   - **Lines 50-74 (normalize test block)**: `result$norm_params` no longer exists on a matrix — change to `attr(result, "FoReco")$fit$norm_params`
   - Any ctrml_g test: matrix check + FoReco attr
3. **New file test-rml-g-reconcile.R** with explicit shape + coherency assertions:
   - `csrml_g`: returns matrix, coherency `agg_mat %*% bottom == top` (tol=1e-10)
   - `terml_g`: returns NAMED numeric vector, length == `h * sum(max(agg_order)/agg_order)`, coherency (sum of high-freq blocks == low-freq)
   - `ctrml_g`: returns matrix, dim == `c(n, h * kt)` where `kt = sum(max(agg_order)/agg_order)`, cross-temporal coherency

## Implementation Order

1. **ak9.9** (DONE): `apply_norm_params()` helper
2. **ak9.1** (DONE): verify `predict.rml_g_fit` broadcast order
3. **ak9.8a** (NEW PRECONDITION): Fix `make_g_fixture()` so hat/base share column space
4. **ak9.2**: csrml_g pipeline
5. **ak9.3**: terml_g pipeline
6. **ak9.4**: ctrml_g pipeline
7. **ak9.5**: mlr3tuning/paradox audit (independent)
8. **ak9.6**: sntz/round/tew decision (drives ak9.7)
9. **ak9.7**: NEWS.md
10. **ak9.8b**: New reconcile regression tests
