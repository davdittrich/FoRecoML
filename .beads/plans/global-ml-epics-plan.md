# Plan: Global ML Reconciliation Functions + CatBoost

## Context

G.1 research (PROCEED verdict, confidence 88/100) established: global pre-training on stacked (y, X, series_id) is accurate (RMSE ≤1.05×) and 4-5× faster for lightgbm/xgboost. User requests:

1. **New core functions** alongside csrml/terml/ctrml: `csrml_g()`, `terml_g()`, `ctrml_g()` — same signature pattern as existing wrappers but with global training path.
2. **Backends**: lightgbm, xgboost, ranger, mlr3, catboost (new).
3. **Normalization**: optional z-score; scale estimator switchable via `robscale` package; **GMD (Gini Mean Difference) as default**.
4. **Extra epic**: add `approach = "catboost"` to EXISTING ctrml/terml/csrml.

## Architecture

### Global function pattern
```r
ctrml_g(base, hat, obs, agg_mat, agg_order, tew = "sum",
         features = "all", approach = "lightgbm",
         normalize = TRUE, scale_estimator = "gmd",
         params = NULL, tuning = NULL,
         sntz = FALSE, round = FALSE,
         fit = NULL, checkpoint = "auto",
         n_workers = 1L, pool = FALSE, shared_hat = FALSE, ...)
```

Key differences from ctrml():
- `normalize = TRUE` (default): z-score features before stacking; denormalize predictions
- `scale_estimator = "gmd"` (default): uses `robscale::scale_gmd()`; supported: any `robscale::` scale_* function name + "sd"
- NO `global_id_list` hoist (different dispatch path)
- Single model trained on all p series stacked
- Only gradient boosters (lightgbm, xgboost, catboost) recommended; ranger/mlr3 supported but documented as slow

### S3 dispatch: `rml_g.<backend>`
New S3 generic `rml_g()` and methods:
- `rml_g.lightgbm(X_stack, y_stack, series_id, ...)`
- `rml_g.xgboost(X_stack, y_stack, series_id, ...)`
- `rml_g.catboost(X_stack, y_stack, series_id, ...)` — native categorical series_id
- `rml_g.ranger(X_stack, y_stack, series_id, ...)` — supported, documented slow
- `rml_g.mlr3(X_stack, y_stack, series_id, ...)` — supported, documented slow

### Normalization
```r
normalize_stack <- function(X_stack, series_id, normalize, scale_estimator) {
  if (!normalize) return(list(X_norm = X_stack, params = NULL))
  # robscale 0.5.4 exports: gmd, mad_scaled, qn, sn, iqr_scaled, sd_c4
  # NOT scale_gmd etc. — switch dispatch on actual function names.
  scale_fn <- switch(scale_estimator,
    "sd"    = function(x) stats::sd(x, na.rm = TRUE),
    "gmd"   = function(x) robscale::gmd(x),
    "mad"   = function(x) robscale::mad_scaled(x),
    "qn"    = function(x) robscale::qn(x),
    "sn"    = function(x) robscale::sn(x),
    "iqr"   = function(x) robscale::iqr_scaled(x),
    "sd_c4" = function(x) robscale::sd_c4(x),
    cli::cli_abort("Unknown scale_estimator: '{scale_estimator}'. Choose: sd, gmd, mad, qn, sn, iqr, sd_c4.")
  )
  # Per-series normalization (each series independently)
  params <- tapply(seq_len(nrow(X_stack)), series_id, function(rows) {
    mu <- colMeans(X_stack[rows, , drop = FALSE])
    sigma <- apply(X_stack[rows, , drop = FALSE], 2, scale_fn)
    sigma[sigma < .Machine$double.eps] <- 1  # guard near-zero scale
    list(mu = mu, sigma = sigma)
  })
  X_norm <- X_stack
  for (s in unique(series_id)) {
    rows <- which(series_id == s)
    X_norm[rows, ] <- sweep(sweep(X_stack[rows, , drop=FALSE], 2, params[[s]]$mu), 2, params[[s]]$sigma, "/")
  }
  list(X_norm = X_norm, params = params)
}
```

Denormalization at predict time: reverse sweep using stored params.

---

## Epic H: Global ML Reconciliation Functions (P1)

### Goal
Add `csrml_g()`, `terml_g()`, `ctrml_g()` as first-class functions with global training path, optional robscale normalization (GMD default), and catboost support.

### Success Criteria
- [ ] All 3 wrapper functions exist with same signature pattern as ctrml/terml/csrml
- [ ] Backends: lightgbm, xgboost, catboost, ranger, mlr3 all functional
- [ ] normalize=TRUE + scale_estimator="gmd" default path works
- [ ] All robscale scale estimators supported (gmd, mad, Qn, Sn, sd)
- [ ] Numerical equivalence: normalized global ≤1.05× per-series RMSE on test fixtures
- [ ] Tests cover: all backends × normalization on/off × all 3 wrapper types

### Sub-Agent Strategy
Sequential tasks H.1 → H.2 → H.3 → H.4. Each hermetic.

---

### Task H.1: Core stacking engine + normalize_stack()

**Status:** `READY_FOR_EXECUTION`

#### I. Context & Objective
- **Objective:** Add `normalize_stack()` + `denormalize_predictions()` internal helpers to R/utils.R; add `rml_g()` generic + dispatcher to R/rml.R.
- **Why:** Foundation for all 3 global wrappers.

#### II. Input
- R/utils.R (after existing normalize/serialize helpers)
- R/rml.R (after existing rml.* methods)
- DESCRIPTION: add `robscale` to Suggests

#### III. Guards
| Type | Guard |
|------|-------|
| Logic | scale_estimator must be one of: "sd", "gmd", "mad", "qn", "sn", "iqr", "sd_c4". These are the ACTUAL robscale 0.5.4 exported function names: gmd(), mad_scaled(), qn(), sn(), iqr_scaled(), sd_c4(). dispatch via switch(), NOT getExportedValue("robscale", paste0("scale_", ...)). |
| Format | Per-series normalization: each series' rows normalized independently using that series' training mu/sigma. Sigma near-zero (< .Machine$double.eps) → set to 1 (no scaling). |
| Boundary | robscale in Suggests; guard with requireNamespace("robscale"); fall back to sd with cli_warn if absent. |
| Audit | `normalize_stack(X, series_id) |> denormalize_predictions(pred)` must be byte-identical to raw predictions when normalize=FALSE. |

#### IV. Logic
1. Add to DESCRIPTION:Suggests: `robscale`
2. Add `normalize_stack(X_stack, series_id, normalize, scale_estimator)` to R/utils.R
3. Add `denormalize_predictions(pred_stack, norm_params, series_id)` to R/utils.R
4. Add `rml_g <- function(approach, X_stack, y_stack, series_id, Xtest_stack, fit, ...) UseMethod("rml_g", approach)` to R/rml.R
5. Add `rml_g.default` that errors with informative message listing supported backends
6. Tests: normalize_stack round-trip byte-identical; GMD default fires; sd fallback when robscale absent

#### VI. DoD
- [ ] normalize_stack + denormalize_predictions in utils.R
- [ ] rml_g generic in rml.R
- [ ] robscale in Suggests
- [ ] Tests: round-trip identity when normalize=FALSE; GMD default; sd fallback when robscale absent
- [ ] norm_params structure documented: `list(mu=numeric(n_feat), sigma=numeric(n_feat))` per series
- [ ] Single commit

---

### Task H.2: rml_g methods for lightgbm + xgboost

**Status:** `READY_FOR_EXECUTION`

#### I. Context & Objective
- **Objective:** Implement `rml_g.lightgbm` and `rml_g.xgboost` in R/rml.R. Global training on stacked X + integer series_id; predict per-series; optional fine-tune.
- **Why:** Two fastest backends per G.1 benchmarks (4.8× / 5.5× speedup).

#### II. Input
- R/rml.R (after H.1 rml_g generic)
- G.1 prototype at dev/g1-bench/g1-global-prototype.R (reference implementation)

#### III. Guards
| Type | Guard |
|------|-------|
| Logic | series_id must be integer column; lightgbm: NOT declared as categorical (poor perf at p>100); xgboost: treated as continuous numeric. |
| Fine-tune | `fine_tune_rounds > 0`: additional gradient steps per-series after global model; default 0 (off). |
| Boundary | Both methods return same structure as rml.*: list(bts=..., fit=...) |
| Audit | RMSE ratio (global/per-series) ≤1.05 on itagdp fixture; verified in test. |

#### IV. Logic
1. `rml_g.lightgbm(X_stack, y_stack, series_id, Xtest_stack, params, fine_tune_rounds=0, fit=NULL, seed=NULL, ...)`
   - Train: `lgb.train(params, lgb.Dataset(X_stack, label=y_stack))`
   - Predict: loop per series, cbind Xtest_i + series_id, `predict(model, ...)`
   - Fine-tune: if fine_tune_rounds > 0, `lgb.train(params, per_series_data, init_model=global_model, nrounds=fine_tune_rounds)`
2. `rml_g.xgboost(...)` — same pattern with xgb.DMatrix
3. Both: return list(bts=matrix(ncol=p), fit=list(global_model=, series_models=if_finetune, norm_params=list_of_p_params))
   norm_params stored here for predict-reuse denormalization. NULL if normalize=FALSE.
   Predict-reuse path (fit=existing): reads fit$norm_params → calls denormalize_predictions(pred, fit$norm_params, series_id).
4. Tests: RMSE ratio test on itagdp; both backends run without error; fine_tune_rounds=10 improves or maintains RMSE.

#### VI. DoD
- [ ] rml_g.lightgbm and rml_g.xgboost in rml.R
- [ ] Both handle fit=NULL (training) and fit=existing (predict-reuse)
- [ ] fine_tune_rounds optional; default 0
- [ ] RMSE ratio test on small fixture
- [ ] fit$norm_params present in returned list; NULL when normalize=FALSE
- [ ] Predict-reuse test: train with normalize=TRUE → store fit → predict from fit → output matches direct global call at tolerance=0

---

### Task H.3: rml_g methods for ranger + mlr3

**Status:** `READY_FOR_EXECUTION`

#### I. Context & Objective
- **Objective:** Implement `rml_g.ranger` and `rml_g.mlr3` in R/rml.R. Functional but documented as slow (no fine-tune support).

#### III. Guards
| Type | Guard |
|------|-------|
| Logic | No fine-tune support. fine_tune_rounds > 0 with ranger/mlr3 → cli_warn("fine_tune not supported for {approach}; skipping.") |
| Boundary | series_id declared as factor for ranger (native categorical), integer for mlr3. |

#### IV. Logic
1. `rml_g.ranger`: `ranger::ranger(y ~ ., data=as.data.frame(cbind(X_stack, y=y_stack)), num.trees=params$num.trees, seed=seed)`; series_id as factor column
2. `rml_g.mlr3`: `mlr3::TaskRegr$new(...)` + learner dispatch; reuse existing mlr3 logic from rml.mlr3 where possible
3. Tests: both run on small fixture; wall-clock warning in docs

#### VI. DoD
- [ ] rml_g.ranger + rml_g.mlr3 in rml.R
- [ ] fine_tune_rounds warning fires for both
- [ ] fit$norm_params echoed in return list (NULL if normalize=FALSE); same contract as H.2
- [ ] Tests run clean

---

### Task H.4: rml_g.catboost (global)

**Status:** `READY_FOR_EXECUTION`

#### I. Context & Objective
- **Objective:** Implement `rml_g.catboost` using catboost R package. Native categorical series_id (no encoding needed). In Suggests (not on CRAN — install from GitHub).

#### III. Guards
| Type | Guard |
|------|-------|
| Logic | catboost not on CRAN; install from GitHub:catboost/catboost. Add to Suggests with DESCRIPTION note. Guard with requireNamespace("catboost"). |
| Feature | series_id declared as categorical via `catboost::catboost.load_pool(cat_features = ncol(X)+1)`. |
| Fine-tune | Supported: `catboost::catboost.train(..., init_model=global_model)`. |

#### IV. Logic
1. `rml_g.catboost(X_stack, y_stack, series_id, Xtest_stack, params, fine_tune_rounds=0, fit=NULL, ...)`
2. Pool construction: `catboost.load_pool(data=cbind(X_stack, series_id), label=y_stack, cat_features=which(colnames(...)=="series_id"))`
3. Train: `catboost.train(pool, params=list(loss_function="RMSE", ...))`
4. Fine-tune: `catboost.train(pool, init_model=model, ...)`
5. Tests: skip_if_not_installed("catboost"); runs on small fixture.

#### VI. DoD
- [ ] rml_g.catboost in rml.R
- [ ] catboost in DESCRIPTION:Suggests
- [ ] Tests gated on skip_if_not_installed("catboost")

---

### Task H.5: csrml_g / terml_g / ctrml_g wrappers

**Status:** `READY_FOR_EXECUTION`

#### I. Context & Objective
- **Objective:** Add `csrml_g()`, `terml_g()`, `ctrml_g()` wrapper functions to R/csrml.R, R/terml.R, R/ctrml.R. Same signature as existing wrappers plus `normalize`, `scale_estimator`, `fine_tune_rounds`.
- **Why:** User-facing API matching existing pattern.

#### III. Guards
| Type | Guard |
|------|-------|
| Logic | Wrappers build X_stack, y_stack, call rml_g(), then pass bts through ctbu/tebu/csbu. |
| Boundary | csrml_g: kset=NULL path; terml_g: temporal-only; ctrml_g: cross-temporal. Same geometric setup as existing siblings. |
| Normalization | Passed through to normalize_stack(). If normalize=TRUE and robscale absent: cli_warn + fall to sd. |
| Audit | `csrml_g(..., normalize=FALSE, approach="lightgbm")` ≈ `csrml(..., approach="lightgbm")` RMSE within 2% (different training but equivalent problem). |

#### IV. Logic
1. `csrml_g()`: copy csrml() structure; replace loop_body_kset dispatch with stacking + rml_g() call
2. `terml_g()`: same for terml
3. `ctrml_g()`: same for ctrml
4. Roxygen: @param normalize, @param scale_estimator, @param fine_tune_rounds
5. Tests: smoke tests all 3 functions × lightgbm backend × normalize on/off

#### VI. DoD
- [ ] csrml_g/terml_g/ctrml_g in respective R files
- [ ] All params documented in roxygen (@param normalize, @param scale_estimator, @param fine_tune_rounds)
- [ ] Returned fit object includes norm_params field (or NULL if normalize=FALSE)
- [ ] Predict-reuse path: reads fit$norm_params → denormalizes predictions
- [ ] Tests: smoke-test all 3 wrappers; normalize=TRUE predict-reuse round-trip
- [ ] devtools::document() generates man pages

---

## Epic I: CatBoost in Existing Functions (P1, extra epic)

### Goal
Add `approach = "catboost"` to existing ctrml/terml/csrml (per-series path) alongside lightgbm/xgboost/randomForest/ranger/mlr3.

### Success Criteria
- [ ] `rml.catboost` S3 method added to R/rml.R
- [ ] catboost in DESCRIPTION:Suggests
- [ ] ctrml/terml/csrml accept `approach = "catboost"` without error
- [ ] Checkpoint: catboost model serialized via catboost::catboost.save_model + catboost::catboost.load_model
- [ ] Tests: skip_if_not_installed("catboost"); fixture runs; checkpoint round-trip
- [ ] Wall-clock benchmark vs lightgbm on small fixture (document in NEWS or doc)

### Tasks

#### Task I.1: rml.catboost + serialize/deserialize

**Status:** `READY_FOR_EXECUTION`

##### I. Context & Objective
- **Objective:** Add `rml.catboost(X, y, Xtest, params, seed, fit, ...)` to R/rml.R. Per-series training (not global — that's H.4). Integrate with serialize_fit/deserialize_fit.

##### III. Guards
| Type | Guard |
|------|-------|
| Logic | catboost in Suggests; requireNamespace guard; error with install instructions if absent. |
| Checkpoint | catboost models serialized via tempfile + catboost::catboost.save_model/load_model; NOT via qs2. |
| Boundary | Only R/rml.R + R/utils.R (serialize_fit extension). |

##### IV. Logic
1. Add catboost case to serialize_fit() / deserialize_fit() / get_fit_i() in utils.R
2. Implement `rml.catboost(X, y, Xtest, params, seed, fit, ...)`:
   - Train: `catboost.load_pool(X, label=y)` + `catboost.train(pool, params=list(loss="RMSE", ...))`
   - Predict: `catboost.predict(model, catboost.load_pool(Xtest))`
   - Return: list(bts=pred, fit=serialized_model)
3. Tests: skip_if_not_installed; smoke test on itagdp fixture; checkpoint round-trip

##### VI. DoD
- [ ] rml.catboost in rml.R
- [ ] serialize/deserialize support for catboost in utils.R
- [ ] catboost in DESCRIPTION:Suggests (if not already from H.4)
- [ ] Tests pass (or skip if catboost not installed)

---

## Cross-cutting invariants

- mw3.3 invariant intact
- spd.12+13+14+B1-B19 regression: existing 383 tests pass unchanged
- Both epics: catboost always in Suggests (not Imports); all catboost calls gated on requireNamespace
- GMD default: robscale::scale_gmd() documentation link in @param scale_estimator
- Single commits per task; conventional commits; no AI attribution

## Sequencing

```
Epic H: H.1 → H.2 → H.3 → H.4 → H.5
Epic I: I.1 (can parallel with H.2+)
```

H.1 must land first (foundation). H.4 and I.1 both add catboost to DESCRIPTION — coordinate or do I.1 first so H.4 inherits.

## Risks

- R1: catboost not on CRAN — install from GitHub; tests must skip_if_not_installed
- R2: robscale not on CRAN — install from CRAN (robscale IS on CRAN since v1.0); add to Suggests
- R3: ranger + mlr3 global (H.3) may be too slow for CI — cap ntrees=50, nrounds=10 for test fixtures
- R4: csrml_g denormalization for predict-reuse: per-series normalization params must be stored in fit object and reused at predict time
