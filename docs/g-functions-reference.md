# `*_g` Quick Reference Manual

Global ML reconciliation wrappers in FoRecoML.

| Function   | Hierarchy        | Returns                                                              |
|------------|------------------|----------------------------------------------------------------------|
| `csrml_g`  | Cross-sectional  | `n × h` reconciled matrix with `attr(., "FoReco")`                   |
| `terml_g`  | Temporal         | Named numeric vector length `h × kt` with `attr(., "FoReco")`        |
| `ctrml_g`  | Cross-temporal   | `n × (h × kt)` reconciled matrix with `attr(., "FoReco")`            |

Where:
- `n` = total series (upper + bottom)
- `nb` = bottom-only series
- `h` = forecast horizons (cycles ahead)
- `m = max(agg_order)` (finest temporal frequency, e.g. 12 for monthly)
- `kt = sum(m / agg_order)` (total CT columns per cycle; for `agg_order=c(12,6,4,3,2,1)` → `kt=28`)
- `h_hf = h × m` (high-frequency horizon count)
- `T_obs` = training observations (rows of `hat`/`obs` in tall mode)

All three return objects carry `attr(., "FoReco")` of class `foreco_info`. Extract the underlying fit via `extract_reconciled_ml(result)`.

---

## 1. `csrml_g` — cross-sectional only

```r
csrml_g(base, hat, obs, agg_mat,
        approach = c("lightgbm", "xgboost", "ranger", "mlr3", "catboost"),
        normalize = c("none", "zscore", "robust"),
        scale_fn  = "gmd",
        params, seed, early_stopping_rounds = 0L,
        validation_split = 0,
        method = c("bu", "rec"), comb = "ols",
        nonneg_method = "sntz", sntz = FALSE, round = FALSE,
        obs_mask = NULL,
        batch_size = NULL, ...)
```

### Inputs (shapes locked)

| Arg       | Shape          | Rows                 | Cols                 |
|-----------|----------------|----------------------|----------------------|
| `hat`     | `T_obs × n`    | training time obs    | one per series (upper + bottom) |
| `obs`     | `T_obs × nb`   | training time obs    | one per bottom series |
| `base`    | `h × n`        | forecast horizons    | one per series (same column order as `hat`) |
| `agg_mat` | `n_agg × nb`   | upper series         | bottom series        |

**Hard rule:** `ncol(base) == ncol(hat)` AND both share column space (same series in same order).

### Output

`n × h` reconciled matrix. Row order: upper series first (matching `rownames(agg_mat)`), then bottom (matching `colnames(agg_mat)`).

### Minimal example

```r
library(FoRecoML); set.seed(1)
agg_mat <- t(c(1, 1)); dimnames(agg_mat) <- list("A", c("B", "C"))
T_obs <- 50L; h <- 2L; n <- 3L; nb <- 2L
hat  <- matrix(rnorm(T_obs * n), T_obs, n, dimnames = list(NULL, c("A","B","C")))
obs  <- matrix(rnorm(T_obs * nb), T_obs, nb, dimnames = list(NULL, c("B","C")))
base <- matrix(rnorm(h * n), h, n, dimnames = list(NULL, c("A","B","C")))

r <- csrml_g(base = base, hat = hat, obs = obs, agg_mat = agg_mat,
             approach = "lightgbm", seed = 1L)
dim(r)  # n × h
```

---

## 2. `terml_g` — temporal only

```r
terml_g(base, hat, obs, agg_order,
        approach, normalize, scale_fn, params, seed,
        early_stopping_rounds, validation_split,
        method, comb, nonneg_method, sntz, round, tew,
        obs_mask = NULL,
        level_id = FALSE, kset = NULL,
        input_format = c("tall", "wide_ct"), ...)
```

### Inputs — TALL (default)

| Arg       | Shape       | Notes                                               |
|-----------|-------------|-----------------------------------------------------|
| `hat`     | `T_obs × kt` | All temporal levels stacked as columns             |
| `obs`     | `T_obs × 1` | **Single** bottom-level series (highest freq). Multi-series temporal → use `ctrml_g`. |
| `base`    | `h_hf × kt` | `nrow(base) %% m == 0` enforced                    |
| `agg_order` | integer vector | e.g. `c(4L, 2L, 1L)` for quarterly→monthly      |

### Inputs — WIDE_CT

| Arg       | Shape                | Notes                                       |
|-----------|----------------------|---------------------------------------------|
| `hat`     | `1 × (n_folds × kt)` | Single series, folds of CT features         |
| `obs`     | `1 × T_monthly`      | Single bottom series, monthly observations  |
| `base`    | `1 × kt`             | Single test CT feature vector               |

### Output

**Named numeric vector** of length `h × kt`. Names encode `"k-<order> h-<horizon>"`.

### Minimal example

```r
agg_order <- c(4L, 2L, 1L); m <- max(agg_order); kt <- sum(m / agg_order)  # 7
T_obs <- 60L; h <- 2L
hat  <- matrix(rnorm(T_obs * kt), T_obs, kt, dimnames = list(NULL, paste0("L", 1:kt)))
obs  <- matrix(rnorm(T_obs), T_obs, 1L, dimnames = list(NULL, "S1"))
base <- matrix(rnorm(h * m * kt), h * m, kt, dimnames = list(NULL, paste0("L", 1:kt)))

r <- terml_g(base = base, hat = hat, obs = obs, agg_order = agg_order,
             approach = "lightgbm", seed = 1L)
length(r)         # h * kt
head(names(r))    # "k-4 h-1", "k-4 h-2", ...
```

---

## 3. `ctrml_g` — cross-temporal (primary use case)

```r
ctrml_g(base, hat, obs, agg_mat, agg_order,
        approach, normalize, scale_fn, params, seed,
        early_stopping_rounds, validation_split,
        method, comb, nonneg_method, sntz, round, tew,
        obs_mask = NULL,
        level_id = FALSE, kset = NULL,
        input_format = c("tall", "wide_ct"),
        cs_level = FALSE,
        batch_size = NULL, ...)
```

### Inputs — TALL (default)

| Arg       | Shape              | Notes                                                          |
|-----------|--------------------|----------------------------------------------------------------|
| `hat`     | `T_obs × (n × kt)` | CT feature matrix, time as rows                                |
| `obs`     | `T_obs × nb`       | Bottom-level monthly obs, one column per bottom series         |
| `base`    | `h_hf × (n × kt)`  | `nrow(base) %% m == 0`; same column space as `hat`             |
| `agg_mat` | `n_agg × nb`       | Cross-sectional aggregation                                    |
| `agg_order` | integer vector   | Temporal aggregation                                           |

### Inputs — WIDE_CT (FoReco canonical layout)

| Arg       | Shape                | Notes                                                           |
|-----------|----------------------|-----------------------------------------------------------------|
| `hat`     | `n × (n_folds × kt)` | **Rows = series** (upper + bottom). Cols = CT features per fold. **Colnames required** for LightGBM/ranger (auto-generated `V1..Vkt` if absent). |
| `obs`     | `nb × T_monthly`     | Rows = bottom series. Cols = monthly observations.              |
| `base`    | `n × kt`             | Rows = series (same order as `hat`). One test CT vector per series. |
| `agg_mat` | `n_agg × nb`         | Cross-sectional aggregation                                     |
| `agg_order` | integer vector     | e.g. `c(12L, 6L, 4L, 3L, 2L, 1L)`                              |

**Hard rules in wide_ct:**
- `ncol(hat) == n_folds × kt` (integer division; aborts otherwise)
- `ncol(base) == kt`
- `nrow(agg_mat) == n - nb` (upper count consistent)
- `rownames(hat)` and `rownames(base)` MUST exist and match
- `rownames(obs)` MUST be a subset of `rownames(hat)` (bottom series identified by name)

### Output

`n × kt` (for `h=1`) or `n × (h × kt)` reconciled matrix. Row order matches `series_id_levels` (sorted).

### Minimal example — TALL

```r
agg_mat <- t(c(1, 1)); dimnames(agg_mat) <- list("A", c("B", "C"))
agg_order <- c(4L, 2L, 1L); m <- max(agg_order); kt <- sum(m / agg_order)  # 7
n <- 3L; nb <- 2L; ncf <- n * kt  # 21
T_obs <- 60L; h <- 2L

hat  <- matrix(rnorm(T_obs * ncf), T_obs, ncf, dimnames = list(NULL, paste0("F", 1:ncf)))
obs  <- matrix(rnorm(T_obs * nb),  T_obs, nb, dimnames = list(NULL, c("B", "C")))
base <- matrix(rnorm(h * m * ncf), h * m, ncf, dimnames = list(NULL, paste0("F", 1:ncf)))

r <- ctrml_g(base = base, hat = hat, obs = obs,
             agg_mat = agg_mat, agg_order = agg_order,
             approach = "lightgbm", seed = 1L)
dim(r)  # n × (h × kt)
```

### Minimal example — WIDE_CT (production layout)

```r
agg_order <- c(12L, 6L, 4L, 3L, 2L, 1L); m <- 12L; kt <- sum(m / agg_order)  # 28
n <- 3L; nb <- 2L; n_folds <- 11L; T_monthly <- n_folds * m

agg_mat <- matrix(c(1, 1), 1, 2, dimnames = list("U", c("A", "B")))

hat  <- matrix(rnorm(n * n_folds * kt), n, n_folds * kt,
               dimnames = list(c("U", "A", "B"), paste0("f", seq_len(n_folds * kt))))
obs  <- matrix(rnorm(nb * T_monthly), nb, T_monthly,
               dimnames = list(c("A", "B"), NULL))
base <- matrix(rnorm(n * kt), n, kt,
               dimnames = list(c("U", "A", "B"), paste0("f", seq_len(kt))))

r <- ctrml_g(base = base, hat = hat, obs = obs,
             agg_mat = agg_mat, agg_order = agg_order,
             approach = "lightgbm",
             input_format = "wide_ct", seed = 1L)
dim(r)  # n × kt
```

---

## Optional features (all default OFF — backward compatible)

### `obs_mask` — exclude structurally missing rows

```r
mask <- obs != 0   # or supplied logical matrix
ctrml_g(..., obs_mask = mask)
ctrml_g(..., obs_mask = "auto")   # auto-detect obs==0; warns
```

- Same shape as `obs` (tall: `T_obs × nb`; wide_ct: `nb × T_monthly`).
- Masked rows excluded from `X_train`/`y_train`.
- `series_id_levels` preserved → masked series still get predictions via ML extrapolation.
- Masked validation residuals → `NA` in `compute_rec_residuals` (no zero-imputation, no MinT poisoning).

### `level_id` — temporal aggregation level as feature

```r
terml_g(..., level_id = TRUE, kset = c(4L, 2L, 1L))
ctrml_g(..., level_id = TRUE)   # kset derived from agg_order
```

- Ordered integer: `1` = finest (k=1, monthly), `max(level_id)` = coarsest.
- `csrml_g(level_id = TRUE)` → `cli_abort` (CS has no temporal axis).

### `cs_level` — cross-sectional depth as feature

```r
ctrml_g(..., input_format = "wide_ct", cs_level = TRUE)
```

- Requires `input_format = "wide_ct"`. Tall mode aborts (bottom-only stacking makes `cs_level` constant).
- `0` = upper series (aggregate). `1` = bottom series (leaf).
- Encoded as ordered integer column appended to `X_stacked`.

### `method = "rec"` — FoReco optimal combination

```r
csrml_g(..., method = "rec", comb = "shr")            # cov shrinkage
terml_g(..., method = "rec", comb = "ols")            # ols only (no residuals API)
ctrml_g(..., method = "rec", comb = "ols")            # ols only
```

- `comb = "shr"`/`"sam"`: uses validation residuals from the global ML fit. Requires `validation_split > 0`.
- `comb = "ols"`: no residuals needed.

### `nonneg_method` — non-negativity post-reconciliation

```r
ctrml_g(..., nonneg_method = c("sntz", "bpv", "nfca", "nnic"))
```

- `"sntz"` (default): clip negatives after BU. Near-optimal per Girolimetto 2025.
- `"bpv"`/`"nfca"`/`"nnic"`: passed to `FoReco::ctrec(nn = ...)`. Warns if base predictions >30% negative (degenerate face risk).

---

## Reproducibility checklist

Before reporting "X doesn't work":

```bash
# 1. Reinstall (NOT load_all)
R CMD INSTALL --preclean .

# 2. Run full suite
Rscript -e 'devtools::test()'   # expect: FAIL 0 | PASS 326+

# 3. Confirm package version + commit
Rscript -e 'cat(as.character(packageVersion("FoRecoML")), "\n")'
git rev-parse HEAD

# 4. Minimal repro on fresh fixture (use the wide_ct example above)
```

`devtools::load_all()` is sufficient for iterative development but is NOT equivalent to a fresh install when the dispatch table (S3 methods, `series_id_levels` factor coding, namespace) has changed. **Always `R CMD INSTALL --preclean .` before benchmarking, profiling, or reporting bugs.**

---

## See also

- `docs/research/2026-05-17-analyst-report-triage.md` — triage of the 2026-05-17 analyst report
- `vignettes/articles/feature-engineering.Rmd` — `level_id` and feature-engineering deep dive
- `?csrml_g`, `?terml_g`, `?ctrml_g` — full roxygen documentation
- `extract_reconciled_ml(result)` — recover the underlying `rml_g_fit` from a reconciliation result
