# Research Project: Global ML Cross-Temporal Reconciliation — SOTA

**Date:** 2026-05-16
**PI:** assistant
**Deliverable:** `docs/research/2026-05-16-global-ml-ctrml-sota.md`

## Task List

- [x] **1. Scope + key terms** — done inline
- [x] **2. Literature retrieval (academic)** — DONE (14 papers, 5 categories)
- [x] **3. Implementation survey** — DONE (12 implementations characterized)
- [x] **4. Synthesize SOTA patterns** — DONE → `docs/research/2026-05-16-global-ml-ctrml-sota.md` §2-5
- [x] **5. Cross-temporal deep dive** — DONE → §4 of the report
- [x] **6. Design recommendations for ctrml_g** — DONE → §6 of the report (3-tier ranked)

## Scope (Phase 1)

### Key terms
- **Hierarchical reconciliation**: post-forecast adjustment so child sums equal parent
- **Cross-sectional (cs)**: hierarchy over series (countries → states → cities)
- **Temporal (te)**: hierarchy over time (yearly → quarterly → monthly)
- **Cross-temporal (ct)**: joint cs + te
- **Bottom-up (bu)**: forecast leaves, sum upward (csbu/tebu/ctbu in FoReco)
- **MinT**: optimal Minimum Trace reconciliation (Hyndman et al.)
- **Global model**: one ML model trained on stacked series + series_id
- **Per-series model**: independent model per leaf
- **End-to-end coherent**: model outputs already-coherent forecasts (no post-hoc reconciliation)
- **Post-hoc reconciliation**: model outputs raw forecasts; csbu/tebu/ctbu applied after

### Anchors
- Rombouts, Girolimetto, Wilms 2024 (Int J Forecasting) — current FoRecoML reference
- Spiliotis et al. 2021 (Appl Soft Comp) — cross-sectional ML reconciliation
- Hyndman et al. (MinT) — analytic reconciliation
- Athanasopoulos et al. — temporal hierarchies

### Out of scope
- Probabilistic reconciliation (separate topic)
- Pure deep learning encoders (DeepAR, TFT) without coherency
- Application-specific case studies (retail, energy)

---

## Phase 2 — Literature findings

**Retrieved:** 2026-05-16 | **Method:** WebSearch + WebFetch across Semantic Scholar, arXiv, awesome-forecast-reconciliation, ICML/AAAI proceedings.

---

### Category A — Cross-temporal ML reconciliation (highest relevance)

**A1.** Rombouts, J., Ternes, M., & Wilms, I. (2024). Cross-temporal forecast reconciliation at digital platforms with machine learning. *International Journal of Forecasting*, 41(1), 321–344. DOI: 10.1016/j.ijforecast.2024.05.008 | arXiv:2402.09033
- **TL;DR:** First ML-based cross-temporal reconciliation method; non-linear post-hoc reconciliation via ML (gradient boosted trees) applied jointly across cross-sectional × temporal hierarchy levels. Tested on delivery-platform and NYC bike-share data.
- **Relevance:** HIGH — this is the primary anchor for `ctrml_g`. The global ML approach applied post-hoc to cross-temporal hierarchies is the exact pattern the package generalises.
- **Key insight:** ML reconciliation can be applied in a single automated pass across the full ct-hierarchy without requiring separate cs/te steps; LightGBM/XGBoost work well at scale.

**A2.** Girolimetto, D., & Di Fonzo, T. (2025). Cross-temporal forecast reconciliation: insights on sequential, iterative, and optimal approaches. *Statistical Methods & Applications*. DOI: 10.1007/s10260-025-00822-z | arXiv:2410.19407
- **TL;DR:** Proves when sequential (cs-then-te or te-then-cs) equals iterative equals optimal reconciliation; delivers significant memory/compute savings for high-dimensional ct-hierarchies.
- **Relevance:** HIGH — defines the theoretical space that `ctrml_g`'s sequential/iterative/optimal modes must respect.
- **Key insight:** Iterative cross-temporal reconciliation converges to the optimal regardless of application order; sequential can match optimal under specific covariance conditions.

**A3.** Girolimetto, D., Athanasopoulos, G., Di Fonzo, T., & Hyndman, R.J. (2024). Cross-temporal probabilistic forecast reconciliation. *International Journal of Forecasting*, 40(3), 1134–1151. DOI: 10.1016/j.ijforecast.2023.10.003
- **TL;DR:** Linear probabilistic cross-temporal reconciliation framework unifying point and distributional coherence; complements Rombouts et al. on the linear side.
- **Relevance:** HIGH — baseline comparator for `ctrml_g`; probabilistic extension directly applicable.
- **Key insight:** Optimal ct-reconciliation requires joint cs×te covariance estimation; scalable approximations (diagonal, shrinkage) critical in practice.

---

### Category B — Global ML + post-hoc reconciliation (Spiliotis paradigm)

**B1.** Spiliotis, E., Abolghasemi, M., Hyndman, R.J., Petropoulos, F., & Assimakopoulos, V. (2021). Hierarchical forecast reconciliation with machine learning. *Applied Soft Computing*, 112, 107756. DOI: 10.1016/j.asoc.2021.107756
- **TL;DR:** Replaces linear MinT reconciliation with ML (Random Forest, XGBoost) that non-linearly combines base forecasts across hierarchy levels; cross-sectional only.
- **Relevance:** HIGH — foundational method that Rombouts et al. extends to the cross-temporal case.
- **Key insight:** Non-linear ML reconciliation consistently outperforms linear MinT; feature engineering (level indicators, lags) is the key driver.

**B2.** Yingjie, Z., & Abolghasemi, M. (2024). Local vs. Global Models for Hierarchical Forecasting. arXiv:2411.06394
- **TL;DR:** Systematic comparison of per-series local models vs. global ML (LightGBM MCB) across hierarchy levels; GFMs dominate on accuracy and compute.
- **Relevance:** HIGH — directly answers the local-vs-global question for `ctrml_g`; provides empirical evidence that single global model + reconciliation beats many local models.
- **Key insight:** LightGBM GFMs achieve superior accuracy at lower model complexity; cross-hierarchy features (series_id, level_id) are critical for the global model to learn hierarchical structure.

**B3.** Sprangers, O., Wadman, W., Schelter, S., & de Rijke, M. (2023). Hierarchical Forecasting at Scale. arXiv:2310.12809 | *International Journal of Forecasting*, 2024. DOI: 10.1016/j.ijforecast.2024.00116
- **TL;DR:** Sparse loss function enabling coherent forecasts for millions of series using a single bottom-level model; no post-hoc reconciliation needed. Validated on M5 (+10% RMSE) and bol.com production (+5–10% across cs-hierarchy).
- **Relevance:** HIGH — end-to-end coherence via loss design rather than post-hoc; critical architecture alternative for `ctrml_g`.
- **Key insight:** Training-time hierarchical loss is competitive with post-hoc reconciliation and scales to millions of series without reconciliation overhead.

**B4.** Abolghasemi, M., Hyndman, R.J., Tarr, G., & Bergmeir, C. (2022). Model selection in reconciling hierarchical time series. *Machine Learning*, 111, 3031–3056.
- **TL;DR:** Framework for selecting which ML reconciliation method to apply per hierarchy level based on cross-validation; shows no single method wins across all levels.
- **Relevance:** MEDIUM — relevant for adaptive reconciliation strategy in `ctrml_g`.
- **Key insight:** Per-level model selection via CV yields consistent gains over fixed reconciliation method.

---

### Category C — End-to-end coherent neural approaches

**C1.** Rangapuram, S.S., Werner, L.D., Benidis, K., Mercado, P., Gasthaus, J., & Januschowski, T. (2021). End-to-end learning of coherent probabilistic forecasts for hierarchical time series. *ICML 2021*, PMLR 139:8832–8843.
- **TL;DR:** Reparameterisation trick casts reconciliation as a closed-form optimisation step embedded in training; produces coherent probabilistic forecasts without post-hoc reconciliation. Supports grouped and temporal hierarchies.
- **Relevance:** HIGH — canonical end-to-end coherent model; baseline for neural architectures in `ctrml_g` evaluation.
- **Key insight:** Reconciliation can be embedded as a differentiable layer; the reparameterisation trick makes gradients flow through the constraint.

**C2.** Tsiourvas, A., Sun, W., Perakis, G., Chen, P.-Y., & Zhu, Y. (2024). Learning Optimal Projection for Forecast Reconciliation of Hierarchical Time Series. *ICML 2024*, PMLR 235:48713–48727.
- **TL;DR:** Reconciliation as a learnable oblique projection layer trained end-to-end with the neural forecaster; oblique projection assigns adaptive weights per series, outperforming fixed-weight (MinT) approaches.
- **Relevance:** HIGH — directly addresses differentiable reconciliation layers; the oblique projection generalises MinT and is learnable from data.
- **Key insight:** Oblique vs. orthogonal projection is the key degree of freedom; learning the projection jointly with the forecaster yields significant accuracy gains.

**C3.** Cini, A., Mandic, D., & Alippi, C. (2024). Graph-based Time Series Clustering for End-to-End Hierarchical Forecasting. *ICML 2024*. arXiv:2305.19183
- **TL;DR:** Pyramidal graph structure where each layer is a hierarchy level; trainable graph pooling learns cluster-to-hierarchy alignment; differentiable reconciliation enforces constraints as both architectural bias and regulariser.
- **Relevance:** MEDIUM — differentiable reconciliation + GNN; applicable when the hierarchy graph is learned from data rather than pre-specified.
- **Key insight:** Hierarchical structure need not be pre-specified; graph pooling can learn it, with differentiable reconciliation ensuring coherence throughout training.

**C4.** Wang, S., Zhou, F., Sun, Y., Ma, L., Zhang, J., Zheng, Y., Lei, L., & Hu, Y. (2022). End-to-End Modeling of Hierarchical Time Series Using Autoregressive Transformer and Conditional Normalizing Flow based Reconciliation. *IEEE ICDM Workshops 2022*. arXiv:2212.13706
- **TL;DR:** Combines autoregressive transformer (base forecasts) with conditional normalising flow (reconciliation) in one model; handles non-Gaussian distributions and complex aggregation constraints.
- **Relevance:** MEDIUM — useful for probabilistic end-to-end design; normalising flow reconciliation handles non-Gaussian residuals better than Gaussian MinT.
- **Key insight:** Normalising flows can parameterise the reconciliation distribution, enabling exact likelihood training with coherence guarantees.

**C5.** Zhou, F., Pan, C., Ma, L., et al. (2024). GMP-AR: Granularity Message Passing and Adaptive Reconciliation for Temporal Hierarchy Forecasting. *AAAI 2024*. arXiv:2406.12242
- **TL;DR:** Granularity message-passing across temporal hierarchy levels enriches base forecasts; adaptive reconciliation (node-dependent weighting) maintains coherence without accuracy loss. Applied at Alipay.
- **Relevance:** HIGH — temporal-hierarchy-specific end-to-end neural approach; directly extends to cross-temporal setting.
- **Key insight:** Hierarchy-aware message passing is a stronger inductive bias than simple global-feature concatenation; adaptive reconciliation weights per node outperform uniform MinT weights.

---

### Category D — Regularisation / training-time coherence

**D1.** Kamarthi, H., Kong, L., Rodríguez, A., Zhang, C., & Prakash, B.A. (2022). PROFHIT: Probabilistic Robust Forecasting for Hierarchical Time-series. arXiv:2206.07940
- **TL;DR:** Bayesian refinement module + Distributional Coherency regularisation that penalises incoherence during training; robust to up to 10% missing series; 41–88% CRPS improvement.
- **Relevance:** MEDIUM — probabilistic focus; the coherency regularisation loss is applicable to non-probabilistic global ML training.
- **Key insight:** Soft coherency regularisation during training (not hard post-hoc reconciliation) is more robust to missing data and distribution shift.

**D2.** Paria, B., Sen, R., Ahmed, A., & Das, A. (2021). Hierarchically Regularized Deep Forecasting. arXiv:2106.07630
- **TL;DR:** Decomposes time series into global basis + hierarchy-constrained coefficients using time-varying autoregressive model; approximate coherence constraints applied as regularisation during training.
- **Relevance:** MEDIUM — basis decomposition approach provides an alternative to explicit reconciliation; scalable (inference per-series).
- **Key insight:** Basis decomposition + regularisation achieves near-coherence without the quadratic cost of full reconciliation; useful for very large hierarchies.

---

### Category E — Cross-temporal foundations (non-ML, needed as baselines)

**E1.** Di Fonzo, T., & Girolimetto, D. (2023). Cross-temporal forecast reconciliation: Optimal combination method and heuristic alternatives. *International Journal of Forecasting*, 39(1), 39–57.
- **TL;DR:** Defines optimal cross-temporal reconciliation (ct-OLS, ct-WLS, ct-MinT) and heuristic sequential alternatives; establishes the linear baseline that ML methods must beat.
- **Relevance:** HIGH — mandatory baseline for `ctrml_g` evaluation; defines the reconciliation target.
- **Key insight:** Sequential cs-then-te reconciliation is a tractable heuristic that approaches optimal ct-reconciliation at large scale.

**E2.** Punia, S., Singh, S.P., & Madaan, J.K. (2020). A cross-temporal hierarchical framework and deep learning for supply chain forecasting. *Computers & Industrial Engineering*, 145, 106506.
- **TL;DR:** First LSTM-based cross-temporal framework; forecasts at all temporal aggregation levels, reconciles te first then cs; applied to retail supply chain (141 series).
- **Relevance:** MEDIUM — early cross-temporal + DL work; establishes sequential te→cs as a practical pattern.
- **Key insight:** Sequential te-then-cs reconciliation with LSTM base forecasts is feasible and outperforms independent-level approaches in supply chain settings.

**E3.** Neubauer, L., & Filzmoser, P. (2024). Rediscovering Bottom-Up: Effective Forecasting in Temporal Hierarchies. arXiv:2407.02367
- **TL;DR:** First theoretical proof that bottom-up is the optimal reconciliation for temporally aggregated ARIMA models; validated on simulations and real data.
- **Relevance:** MEDIUM — sets the theoretical bar: ctbu is optimal under ARIMA assumptions; ML must demonstrate improvement over ctbu, not just over MinT.
- **Key insight:** Bottom-up is theoretically optimal for ARIMA temporal hierarchies; ML reconciliation gains come primarily from departures from ARIMA structure (non-linearity, cross-series features).

---

### Gaps identified

1. **No paper found combining global LightGBM/XGBoost as base forecaster + ML cross-temporal reconciler in one pipeline** — Rombouts et al. use ML for the reconciliation step but do not address global base forecasting jointly. This is the gap `ctrml_g` fills.
2. **Cross-temporal GMP-AR (C5) is temporal-only** — no paper found doing cross-sectional × temporal message passing end-to-end in a neural model.
3. **M5 competition winners using global models + cross-temporal reconciliation** — no dedicated paper found; M5 winners used LightGBM globally but reconciled by bottom-up (cs only), not cross-temporal.
4. **SHARQ** — found reference in survey but no standalone arXiv or published paper identified with that name; likely a conference workshop contribution (not retrieved with confidence). Excluded to avoid fabrication.

---

## Phase 3 — Implementation survey

**Retrieved:** 2026-05-16 | **Method:** WebSearch + WebFetch + GitHub API across R CRAN, PyPI, GitHub.

---

### R ecosystem

#### FoReco (Girolimetto & Di Fonzo)
- **Repo:** https://github.com/danigiro/FoReco | **CRAN:** v1.2.1.9000 (dev HEAD)
- **Last meaningful release:** v1.2.0 (adds probabilistic methods, non-negative constraints) — 2024/2025; v1.2.1 is a bug-fix patch
- **License:** GPL-3
- **Approach:** Post-hoc reconciliation only (no base forecasting); applies to any base-forecast source. Covers all three frameworks (cs, te, ct).
- **Cross-temporal support:** Full (`ctrec()`, `ctlcc()`, `ctbu()`, `cttd()`, `ctmo()`, `ctmvn()`, `ctsmp()`)
- **Series_id handling:** N/A — operates on pre-computed forecast matrices, not raw series
- **End-to-end coherent:** No — always post-hoc
- **Notable design decisions:**
  - v1.2.0 added **four non-negative reconciliation algorithms** (bpv, nfca, nnic, sntz) following Girolimetto (2025) — critical for count/demand hierarchies
  - **Probabilistic: Gaussian MVN** (`*mvn()`) and sample-based (`*smp()`) reconciliation added across all three frameworks
  - New conversion utilities (`as_ctmatrix()`, `as_horizon_stacked_ctmatrix()`) for reshaping ct-forecast arrays
  - Companion package **FoRecoML** (v1.0.0, released 2026-04-21) handles all non-linear ML reconciliation via `csrml()`, `terml()`, `ctrml()`, and `extract_reconciled_ml()` — uses `mlr3` backend with RF, LightGBM, XGBoost
- **Worth borrowing for ctrml_g:**
  1. **`extract_reconciled_ml()` pattern** — allows reuse of already-fitted reconciliation models across datasets without re-fitting; directly applicable to `ctrml_g` for production settings
  2. **Non-negative constraint algorithms** (bpv, nnic) — FoRecoML does not yet implement these on top of ML reconciliation; adding them would be a meaningful gap-fill

#### hts (Hyndman et al.)
- **Repo:** https://github.com/earowang/hts | **CRAN:** v6.0.3 (2024-07-30)
- **License:** GPL-2 | GPL-3
- **Approach:** Bottom-up, top-down, MinT (OLS, WLS), optimal combination
- **Cross-temporal support:** No — cross-sectional only
- **Status:** **Retired** in favour of fable/fabletools; kept on CRAN for backward compatibility
- **Worth borrowing:** Nothing new — fully superseded

#### thief (Hyndman)
- **Repo:** https://github.com/robjhyndman/thief | **CRAN:** v0.3 (2018-01-24)
- **License:** GPL-3
- **Approach:** Temporal hierarchies only; BU + optimal combination
- **Cross-temporal support:** Temporal only
- **Status:** **Archived/inactive** — superseded by fable + FoReco for temporal work
- **Worth borrowing:** Nothing new

#### fable / fabletools (tidyverts)
- **Repo:** https://github.com/tidyverts/fabletools | **Last update:** 2026 (active)
- **License:** GPL-3
- **Approach:** Tidy interface over any `fable`-model; reconciliation via `reconcile()` verb with `min_trace()`, `bottom_up()`, `top_down()`; temporal aggregation via `aggregate_index()`
- **Cross-temporal support:** Partial — temporal aggregation supported (`aggregate_index()`); cross-temporal joint reconciliation not directly exposed; cross-sectional via nested/crossed key structure
- **Series_id handling:** tbl_ts keys; hierarchical nesting via tsibble
- **End-to-end coherent:** No — post-hoc only
- **Notable design decisions:**
  - `__total` reserved key prevents naming collisions in hierarchy specification
  - Tight integration with tidyverse/tsibble ecosystem; composable grammar (`model() |> reconcile() |> forecast()`)
  - No native ML reconciliation; fable models can feed FoReco/FoRecoML as external post-step
- **Worth borrowing:** `__total` reserved-key convention; composable verb-chain API design

---

### Python ecosystem

#### HierarchicalForecast (Nixtla)
- **Repo:** https://github.com/Nixtla/hierarchicalforecast | **Last release:** v1.5.1 (2026-03-04) | 522 commits
- **License:** Apache 2.0
- **Approach:** Post-hoc reconciliation library; plugs into StatsForecast / MLForecast / NeuralForecast output. All methods assume external base forecasts.
- **Reconciliation methods:**
  - *Point:* BottomUp, TopDown, MiddleOut, MinTrace (OLS/WLS/shrunk/sample), ERM (L1-regularised)
  - *Probabilistic:* Normality (MinTrace-Gaussian), Bootstrap (Gamakumara), PERMBU (copula + BU), Conformal (distribution-free PI)
- **Cross-temporal support:** Yes — all methods work with temporal hierarchies; full cross-temporal (cs × te) documented
- **Series_id handling:** `unique_id` column as categorical identifier; summation matrix `S_df` passed separately — clean separation of identity vs. structure
- **End-to-end coherent:** No — always post-hoc; however, referenced paper FlowRec (arXiv:2505.03955) appears as an upcoming extension
- **Notable design decisions:**
  - **Unified `HierarchicalReconciliation` wrapper** with list-of-reconcilers API: `HierarchicalReconciliation(reconcilers=[MinTrace(), BottomUp()]).reconcile(Y_df, S_df, tags)` — single call reconciles all methods in one pass
  - **ERM** (Empirical Risk Minimisation with L1 regularisation) — an ML-flavoured linear reconciliation alternative to MinT; trains a regularised projection matrix from residuals
  - Clean integration: StatsForecast → MLForecast → HierarchicalForecast → evaluation (utilsforecast) as a pipeline
- **Worth borrowing for ctrml_g:**
  1. **ERM reconciler** — L1-regularised projection matrix acts as a sparse ML reconciler; intermediate between full MinT and full non-linear ML; could serve as a regularised-linear baseline in `ctrml_g`
  2. **List-of-reconcilers API** — running multiple reconciliation methods in one call enables fair same-split comparison; easy to add to the `ctrml_g` output structure

#### GluonTS — DeepVAR-Hierarchical (AWS / Rangapuram et al. ICML 2021)
- **Repo:** https://github.com/awslabs/gluonts | **License:** Apache 2.0 | **Status:** Active
- **Approach:** `DeepVARHierarchical` — single unified DeepVAR network with **reconciliation embedded as a differentiable projection layer** during training (not post-hoc)
- **Reconciliation mechanism:** Projects samples onto the coherence subspace via `P = S(SᵀS)⁻¹Sᵀ` (or weighted variant); gradients flow through the projection
- **Coherency during training:** Yes — `coherent_train_samples=True` enforces coherence at every training step; `coherent_pred_samples=True` at inference
- **Cross-temporal support:** Grouped and temporal hierarchies supported; full cross-temporal not explicitly documented
- **Series_id handling:** Bottom-level series + aggregation matrix from CSV; aggregated series constructed automatically — **user does not need to provide aggregate series**
- **End-to-end coherent:** **Yes** — canonical end-to-end model; training objective flows through reconciliation layer
- **Notable design decisions:**
  - Coherency error metric: `max(|forecast_parent − sum(forecast_children)| / |forecast_parent|)` computed at evaluation
  - Projection used: standard orthogonal (`P = S(SᵀS)⁻¹Sᵀ`) or user-supplied weighted (`P = S(SᵀDS)⁻¹SᵀD`)
- **Worth borrowing for ctrml_g:**
  - **Coherency error diagnostic** — cheap metric for measuring post-reconciliation residual incoherence; trivial to add to `ctrml_g` output diagnostics

#### NeuralForecast / HINT (Nixtla)
- **Repo:** https://github.com/Nixtla/neuralforecast | **License:** Apache 2.0 | **Status:** Active
- **Approach:** HINT (Hierarchical Neural Time Series) = NHITS backbone + reconciliation layer (BottomUp default); **reconciliation is a parameter at model init**, not applied post-hoc after training
- **Reconciliation method:** BottomUp (default); other methods pluggable. GMM (Gaussian Mixture) loss for probabilistic output.
- **Cross-temporal support:** Not explicit in tutorial — hierarchy S matrix passed as `unique_id`-keyed DataFrame; temporal is via NHITS multi-scale interpolation, not explicit te-reconciliation
- **Series_id handling:** `unique_id` column + separate `S` matrix; identical convention to HierarchicalForecast
- **End-to-end coherent:** **Partially** — reconciliation (BU) is applied to training outputs so loss is computed on reconciled forecasts; gradient flows through BU matrix multiply. Not truly differentiable reconciliation (BU is fixed linear).
- **Notable design decisions:**
  - Training on reconciled output (GMM loss on reconciled samples) is the key innovation over post-hoc — the model learns to produce base forecasts that reconcile well
  - Probabilistic output via CRPS + quantile GMM loss
- **Worth borrowing:** Training with a reconciled-output loss rather than raw-output loss — teaches the base model to produce reconciliation-friendly residuals without changing the reconciliation algorithm

#### MLForecast (Nixtla)
- **Repo:** https://github.com/Nixtla/mlforecast | **Last release:** v1.0.31 (2026-03) | 1.2k stars
- **License:** Apache 2.0
- **Approach:** Global ML (sklearn-compatible) forecasting framework; no built-in reconciliation
- **Series_id handling:** `unique_id` column; lags, rolling stats, calendar features engineered automatically
- **Cross-temporal support:** No
- **End-to-end coherent:** No
- **Notable design decisions:** Clean sklearn interface; pairs with HierarchicalForecast for reconciliation step. No special hierarchy awareness during training.
- **Worth borrowing:** The `unique_id` → categorical-feature encoding pattern (identical convention used across all Nixtla libraries enables one-line pipeline composition)

#### DARTS (Unit8)
- **Repo:** https://github.com/unit8co/darts | **License:** Apache 2.0 | **Status:** Active
- **Approach:** Unified time-series framework; reconciliation via `darts.dataprocessing.transformers.reconciliation` (transformer pattern)
- **Reconciliation methods:** BottomUp, TopDown, MinT variants (OLS, WLS-value `wls_val`, etc.); **cross-temporal not supported** (confirmed by search)
- **Series_id handling:** `TimeSeries` with component hierarchy; series identity stored in component structure
- **End-to-end coherent:** No — reconciliation applied as a post-processing transformer
- **Notable design decisions:**
  - Reconciliation as a **data transformer** (fits into the same pipeline abstraction as any other data preprocessing step); this allows easy composition with any DARTS model
  - `MinTReconciliator(method='wls_val')` — clean API; method string selects covariance estimator
- **Worth borrowing:** Reconciler-as-transformer pattern — treating reconciliation as a composable pipeline step (not a monolithic function) that can be chained with preprocessing and prediction

#### skforecast
- **Repo:** https://github.com/JoaquinAmatRodrigo/skforecast | **Status:** Active (2024–2025)
- **License:** BSD-3
- **Approach:** Global ML (`ForecasterRecursiveMultiSeries`) via sklearn estimators; no built-in reconciliation
- **Series_id handling:** Series identity encoded as **categorical feature** directly in the feature matrix (`series_id` as ordinal/one-hot); the global model sees hierarchy position as an input feature
- **Cross-temporal support:** No
- **End-to-end coherent:** No
- **Notable design decisions:**
  - `reshape_series_wide_to_long()` utility — explicit long-format data handling for multi-series global models; relevant to FoRecoML's stacking convention
  - Series_id as a direct input feature to the learner (not a separate embedding or summation-constraint structure) — simplest possible global model architecture
- **Worth borrowing:** Long-format stacking with categorical `series_id` as model feature — matches the `ctrml_g` global design; confirms this is the community standard

#### sktime
- **Repo:** https://github.com/sktime/sktime | **License:** BSD-3 | **Status:** Active (very large)
- **Approach:** Hierarchical forecasting via `HierarchyEnsembleForecaster`; global models possible but not the primary design
- **Series_id handling:** MultiIndex DataFrame (pd.DataFrame with `(series_id, timepoint)` MultiIndex); `__total` reserved for aggregate level
- **Cross-temporal support:** Not directly — cross-sectional hierarchy supported; temporal via separate composition
- **End-to-end coherent:** No — reconciliation via sklearn-style transformers
- **Notable design decisions:**
  - `__total` reserved key (shared with fabletools) prevents naming collision; strong evidence this is a good convention
  - `HierarchyEnsembleForecaster` dispatches per-level forecasters and reconciles; does not natively support ML-based (non-linear) reconciliation
- **Worth borrowing:** `__total` convention; MultiIndex hierarchy specification

#### GluonTS ListNet / hierarchical loss (AWS)
- **Status:** Searched but no standalone dedicated hierarchical-loss module confirmed in GluonTS beyond `DeepVARHierarchical`. The hierarchical summing-constraint loss mentioned in search results is specific to `DeepVARHierarchical`, not a general utility. **No separate GluonTS "ListWiseLoss" module confirmed** — this claim was not verified and is excluded.

---

### Industrial implementations

#### M5 Competition (LightGBM global + cs-BU reconciliation)
- **2nd place:** https://github.com/matthiasanderer/m5-accuracy-competition — LightGBM global model with hierarchical alignment multipliers (not formal MinT; BU-style post-processing)
- **Approach:** Train single global LightGBM on all series; multiply leaf forecasts up the cs-hierarchy; no te-reconciliation
- **Cross-temporal:** No — cs-BU only
- **Worth borrowing:** Hierarchical-alignment multiplier as a fast, interpretable alternative to MinT; could serve as a lightweight `ctrml_g` baseline

#### FlowRec (arXiv:2505.03955, May 2025)
- **Status:** Academic paper only; no open-source release confirmed
- **Approach:** Reformulates reconciliation as network flow optimisation on general DAG hierarchies (beyond trees); polynomial-time for ℓₚ>0 norms; extends MinT to non-tree structures
- **Cross-temporal:** Implied (general network structures subsume ct-hierarchies)
- **ML integration:** Post-hoc on top of any base forecasts
- **Worth borrowing:** General network/DAG hierarchy structure — FoRecoML currently assumes tree hierarchy; FlowRec's DAG formulation is relevant if grouped (non-tree) cs-hierarchies are needed in `ctrml_g`

#### Amazon end-to-end coherent hierarchical (AWS Science blog)
- **Status:** Described in blog post; no public code confirmed
- **Approach:** Single unified neural network; reparameterisation + projection trick (as in Rangapuram ICML 2021 — this appears to be the same work); produces coherent probabilistic forecasts end-to-end
- **Worth borrowing:** Same patterns as GluonTS DeepVARHierarchical above

---

### Summary table

| Implementation | Lang | Active | CS | TE | CT | E2E | ML reconciler | Series_id |
|---|---|---|---|---|---|---|---|---|
| **FoReco** (Girolimetto) | R | Yes | ✓ | ✓ | ✓ | No | No (FoRecoML companion) | matrix inputs |
| **FoRecoML** (Girolimetto) | R | Yes (2026-04) | ✓ | ✓ | ✓ | No | RF/LGB/XGB via mlr3 | matrix inputs |
| **fable/fabletools** | R | Yes | ✓ | partial | No | No | No | tsibble keys |
| **hts** | R | Retired | ✓ | No | No | No | No | — |
| **thief** | R | Archived | No | ✓ | No | No | No | — |
| **HierarchicalForecast** | Py | Yes (v1.5.1) | ✓ | ✓ | ✓ | No | ERM (linear, L1) | unique_id + S_df |
| **GluonTS DeepVARH** | Py | Yes | ✓ | partial | No | **Yes** | projection layer | auto from S |
| **NeuralForecast / HINT** | Py | Yes | ✓ | No | No | Partial | BU-trained | unique_id + S |
| **MLForecast** | Py | Yes | No | No | No | No | No | unique_id |
| **DARTS** | Py | Yes | ✓ | No | No | No | No | component hierarchy |
| **skforecast** | Py | Yes | No | No | No | No | No | cat. feature |
| **sktime** | Py | Yes | ✓ | No | No | No | No | MultiIndex |

*E2E = gradient flows through reconciliation during training. CT = cross-temporal. ERM = Empirical Risk Minimisation.*
