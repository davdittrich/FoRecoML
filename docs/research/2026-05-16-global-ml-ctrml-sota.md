# Global ML Cross-Temporal Reconciliation — SOTA Review v2

**Date:** 2026-05-16
**Version:** v2 (amended with full Girolimetto bibliography synthesis)
**Audience:** FoRecoML maintainers
**Trigger:** Inform future improvements to `ctrml_g()` after Epic FoRecoML-ak9 restoration
**Sources:** OpenAlex, arXiv, Semantic Scholar, CRAN, GitHub — full Girolimetto bibliography (34 works)

---

## TL;DR

- The Rombouts/Wilms/Girolimetto 2024 paper is the **only published method** combining ML reconciliation with cross-temporal hierarchies. `ctrml_g()` implements it. No direct competitor exists.
- **`ctrml_g()` already occupies a unique niche.** Closest peers (Spiliotis 2021, Sprangers 2023) are cross-sectional only.
- **5 SOTA patterns** the package does NOT yet borrow:
  1. Differentiable / learnable reconciliation projection (Tsiourvas ICML 2024 — *oblique projection*)
  2. End-to-end coherent training loss (Sprangers 2023 *sparse hierarchical loss*; GluonTS *DeepVARHierarchical*)
  3. ERM (L1-regularised projection) as a middle tier between MinT and full nonlinear ML (HierarchicalForecast)
  4. **Non-negative ML reconciliation** — bpv/nfca/nnic methods (Girolimetto 2025) not yet threaded through `*_g`; only `sntz` exposed
  5. **Validation-based covariance estimation** — using out-of-sample validation errors rather than in-sample residuals for the reconciliation weight matrix (Girolimetto & Di Fonzo 2024)
- **Theoretical bar:** (a) bottom-up is optimal for ARIMA temporal hierarchies (Neubauer & Filzmoser 2024); (b) iterative cross-temporal reconciliation converges to optimal *regardless of application order* under mild covariance conditions (Girolimetto & Di Fonzo 2025); (c) non-negativity of reconciled forecasts can be guaranteed optimally at near-zero extra cost (Girolimetto 2025).
- **New non-linear frontier:** Girolimetto et al. 2025 (arXiv:2510.21249) extends reconciliation to non-linear constraints (ratios, mortality rates) — signals a future where `ctrml_g` might need to enforce multiplicative constraints.

---

## 1. Problem definition

Given a cross-temporal hierarchy with:
- `n` cross-sectional series (`na` aggregate + `nb` bottom)
- temporal aggregation orders `agg_order = (k₁, k₂, …)`; `m = max(agg_order)`
- `h` low-frequency forecast horizons → `h_hf = h × m` high-frequency periods

Produce reconciled forecasts at all `n × (k* + m)` combinations such that:
- cs constraint: `agg_mat %*% bottom_cs == upper_cs` per temporal period
- te constraint: `sum(high-freq within period) == low-freq` per series

`csbu`/`tebu`/`ctbu` are bottom-up: forecast leaves, sum upward. MinT/OLS/WLS are post-hoc weighted combinations. ML reconciliation (Spiliotis/Rombouts) generalises the combination function from linear to nonlinear.

---

## 2. Method taxonomy

| Family | Coherency mechanism | Example | FoRecoML support |
|--------|--------------------|---------|------------------|
| Bottom-up (BU) | Sum from leaves | `FoReco::ctbu` | ✅ current `*_g` post-fit |
| MinT / OLS / WLS | Linear projection (analytic Σ⁻¹) | `FoReco::ctrec` | ✅ via series-level non-`_g` path |
| ERM (L1 projection) | Linear projection, learned | `HierarchicalForecast.ERM` | ❌ gap |
| Spiliotis ML reconciler | Nonlinear combine per-series | `csrml`/`terml`/`ctrml` | ✅ series-level |
| Rombouts global ML | Single model, all series stacked + `series_id`, then BU | `csrml_g`/`terml_g`/`ctrml_g` | ✅ as of Epic ak9 |
| End-to-end coherent NN | Differentiable reconciliation embedded in loss | Rangapuram ICML21; Tsiourvas ICML24 | ❌ gap |
| Hierarchical-loss bottom-only | Single bottom-level model trained with hierarchy-aware loss | Sprangers 2023; M5 winners | ❌ gap |
| Non-negative projection | Post-projection non-negativity guarantees | Girolimetto 2025 (bpv/nfca/nnic/sntz) | ⚠️ sntz only |
| Non-linear constrained reconciliation | Projection onto non-linear constraint surface | Girolimetto et al. 2025 (NLCR) | ❌ gap |
| Coherent forecast combination | Combination-and-reconciliation joint step | Girolimetto & Di Fonzo 2024 (FoCo2) | ❌ gap |

---

## 3. Where `ctrml_g()` stands

**Strengths**
- Only published implementation combining global ML training with full cross-temporal reconciliation in a single function call
- Post-fit reconcile path (csbu/tebu/ctbu) is mathematically well-founded and — per Girolimetto & Di Fonzo (2025) — provably equivalent to optimal under standard covariance assumptions
- `extract_reconciled_ml()` enables fit reuse across new base forecasts
- `sntz`/`round`/`tew` restored matches series-level parity
- Test fixtures now respect `ncol(hat) == ncol(base)` invariant

**Gaps vs SOTA**
- **No end-to-end training loss** — base ML model is fit with vanilla MSE/quantile loss; reconciliation happens after. SOTA neural methods train with coherency in the loss.
- **No learnable projection** — Tsiourvas 2024 shows oblique projection beats fixed (MinT) by significant margin. `ctbu` is a fixed orthogonal projection.
- **No ERM tier** — sits between MinT (closed-form) and full nonlinear ML; sparse interpretable; valuable for moderately-sized hierarchies.
- **No DAG / non-tree hierarchy support** — FlowRec 2025 generalises beyond trees; FoReco/`ctrml_g` assume tree structure via `agg_mat`/`agg_order`.
- **Non-negative ML reconciliation incomplete** — `sntz` exposed but bpv/nfca/nnic not threaded through `*_g`. Girolimetto (2025) shows sntz achieves near-optimal non-negative performance at negligible computation cost — relevant for energy/demand hierarchies.
- **In-sample covariance only** — reconciliation weight matrix estimated from training residuals; Girolimetto & Di Fonzo (2024 arXiv:2412.11153) shows validation-based covariance estimation improves decision costs.
- **No coherent forecast combination** — FoCo2 (arXiv:2412.03429) enables simultaneous combination of multiple base forecasts under linear constraints; relevant when multiple ML backends produce independent forecasts.

---

## 4. Cross-temporal-specific findings

### Sequential vs iterative vs optimal (Girolimetto & Di Fonzo 2025)
**Published:** Stat Methods Appl, DOI:10.1007/s10260-025-00822-z

Proves: iterative cross-temporal reconciliation **converges to optimal regardless of application order** (cs-then-te or te-then-cs) when the error covariance matrix has specific block-diagonal patterns. Sequential reconciliation matches optimal under stronger (but common) covariance conditions. First full theoretical treatment unifying the three approaches.

→ **For `ctrml_g`**: current single-call ctbu is already optimal under the bottom-up assumption. No need to add sequential/iterative options unless users explicitly want non-BU. The convergence result also means we can safely offer `method = "iterative"` as a computationally cheaper alternative to full joint optimal in high-dimensional settings.

### Regression-based insights (Girolimetto & Di Fonzo 2024, arXiv:2410.19407)
Comprehensive framework for understanding sequential/iterative/optimal equivalence, with emphasis on **computation time and memory** savings achievable in high-dimensional cross-temporal hierarchies. Empirical validation on hourly photovoltaic power generation: "significant improvement in terms of memory space and computation time."

→ **For `ctrml_g`**: in large hierarchies (n × kt > 10⁴), iterative application of 1D reconciliation is viable and theoretically grounded. A `method = "iterative"` option exploits this directly.

### Decision-based vs validation-based reconciliation (Girolimetto, Di Fonzo et al. 2024, arXiv:2412.11153)
Introduces two innovations: (1) **validation-based covariance estimation** using out-of-sample validation errors rather than in-sample residuals; (2) **decision-based hierarchy construction** where aggregation levels align with operational decision horizons rather than statistical groupings. Applied to wind power forecasting; validation-based approach "offers a more balanced compromise between accuracy and decision cost."

→ **For `ctrml_g`**: the current `ctbu` step uses in-sample residuals for any analytic weight matrices. A `cov_method = "validation"` option for MinT-class methods would be a meaningful improvement, directly applicable to energy market use cases.

### Non-negative reconciliation (Girolimetto 2025, DOI:10.3390/forecast7040064)
**Key result:** Set-negative-to-zero (`sntz`) achieves **near-optimal non-negative forecast performance** with negligible computation time compared to constrained optimization approaches (bpv, nfca, nnic). The paper establishes optimality theory for all four methods and provides empirical ranking.

→ **For `ctrml_g`**: `sntz` already exposed — this is the right default. The formal result justifies the current implementation. The bpv/nfca/nnic methods should still be threaded through as expert options, since specific distributional assumptions may favour them in particular datasets.

### Non-linear constrained reconciliation (Girolimetto, Panagiotelis, Di Fonzo, Li 2025, arXiv:2510.21249)
Extends reconciliation to non-linear constraints (ratios: mortality rates, unemployment rates). Proposes NLCR — projection onto a non-linear constraint surface via constrained optimisation. Provides sufficient theoretical conditions for NLCR to improve forecast accuracy. Empirical demonstration on demographic and economic datasets.

→ **For `ctrml_g`**: not immediately applicable (FoReco assumes linear constraints via `agg_mat`). Signals a future API direction: `constraint_type = "nonlinear"` with user-supplied constraint function. Monitor for CRAN package release.

### Temporal-hierarchy message passing (Zhou et al. 2024 GMP-AR, AAAI)
Granularity message passing across temporal levels enriches base forecasts before reconciliation; adaptive node-dependent weighting beats uniform MinT. Applied at Alipay.

→ **For `ctrml_g`**: applicable in cross-temporal setting (extension of temporal-only GMP-AR). Architectural change — not a small patch.

### Bottom-up is optimal for ARIMA temporal hierarchies (Neubauer & Filzmoser 2024)
First theoretical proof. Implies: ML cross-temporal reconciliation gains must come from **nonlinearity** and **cross-series features**, not from departures from BU per se.

→ **For `ctrml_g`**: current ctbu post-fit step is theoretically optimal under linear ARIMA. Improvements must target the base ML stage (richer features, hierarchy-aware loss) or replace ctbu with a learnable projection.

### Local vs global empirical (Yingjie & Abolghasemi 2024)
Systematic comparison: global LightGBM dominates per-series local models on accuracy + compute across cs-hierarchy levels. Cross-hierarchy features (`series_id`, `level_id`) are critical.

→ **For `ctrml_g`**: current `.stack_series()` adds `series_id`. Adding `level_id` (which temporal aggregation level a row corresponds to) as an additional categorical feature is a low-effort improvement.

### Coherent forecast combination (Girolimetto & Di Fonzo 2024, arXiv:2412.03429; FoCo2 package)
Proposes **coherent forecast combination** — combines multiple independent base forecasts (from different experts/models) while simultaneously enforcing linear constraints. Closed-form optimal solutions exist. Demonstrated superior accuracy vs single-task combination and single-expert reconciliation on Australian electricity data.

→ **For `ctrml_g`**: when users run multiple ML backends (e.g., XGBoost + LightGBM + RF), currently they must pick one. FoCo2-style combination of these base forecasts before or instead of the BU step would enable ensemble-within-reconciliation. A `base_combine = "foco2"` option.

### Cross-temporal probabilistic reconciliation (Girolimetto, Athanasopoulos, Di Fonzo, Hyndman 2023, IJF)
**Published:** IJF, DOI:10.1016/j.ijforecast.2023.10.003

Extends probabilistic reconciliation (Panagiotelis 2023, Corani 2021) to the cross-temporal joint framework. Key innovations: (1) **multi-step residuals** for temporal covariance estimation (one-step residuals fail in the temporal dimension); (2) **four covariance matrix alternatives** exploiting the block structure; (3) **overlapping residuals** technique. Validated with CRPS and Energy Score on Australian GDP and Tourism datasets.

→ **For `ctrml_g`**: currently no probabilistic variant. This paper provides the complete methodology for a `ctrml_g_prob()` function. The multi-step residuals insight is directly relevant to the existing BU post-fit step.

---

## 5. Girolimetto research program synthesis

### Core thesis
Daniele Girolimetto's work from 2020–2026 constitutes a systematic mathematical treatment of forecast reconciliation as a general statistical estimation problem. His unifying thesis is: **reconciliation is a constrained projection problem, and the right choice of projection (point vs probabilistic, linear vs non-linear, scalar vs matrix-weighted) determines both accuracy and structural validity of forecasts**. The program moves from establishing the cross-temporal optimal combination (2020) → extending to general linear constraints (2023) → probabilistic extensions (2023) → non-negative projections (2025) → non-linear constraints (2025) → ML integration as package design (2026).

### Main theoretical contributions

1. **Cross-temporal optimal combination (Di Fonzo & Girolimetto 2021, IJF):** First closed-form optimal solution for simultaneous cross-sectional + temporal reconciliation. Supersedes Kourentzes & Athanasopoulos (2019) two-step heuristic.

2. **Equivalence of sequential/iterative/optimal (arXiv:2410.19407 → Stat Methods Appl 2025):** Proves that iterative reconciliation converges to the joint optimal under specific (mild) covariance block-diagonal conditions, regardless of application order. Eliminates the perceived need to always solve the full joint problem.

3. **General linear constraints reconciliation (Stat Methods Appl 2023, arXiv:2305.05330):** Extends reconciliation beyond strict hierarchies to arbitrary linear constraint structures via a structural-like projection separating free from constrained variables. Unifies hierarchical, grouped, and accounting-identity settings under one formula.

4. **Probabilistic cross-temporal reconciliation (IJF 2023, arXiv:2303.17277):** Complete extension of probabilistic reconciliation to the cross-temporal joint case. The multi-step residuals insight is a genuine contribution to the estimation literature, not just to reconciliation.

5. **Non-negative reconciliation optimality (Forecasting 2025):** First formal proof that `sntz` achieves near-optimal non-negative performance. Justifies the default implementation in FoReco.

6. **Non-linear constrained reconciliation (arXiv:2510.21249, 2025):** Opens the non-linear constraint frontier with NLCR — the first general framework for ratio and multiplicative constraints in forecast reconciliation.

### Main methodological contributions

1. **FoReco R package (2020–present):** Comprehensive implementation of all linear reconciliation methods; becomes the reference implementation for the field.

2. **FoRecoML R package (2026):** Adds ML-based nonlinear reconciliation (XGBoost, LightGBM, RF via mlr3) to FoReco, covering cs/te/ct dimensions with global model option.

3. **FoCo2 R package (2024):** Implements coherent forecast combination — simultaneous combination-and-reconciliation of multiple base forecast sources.

4. **Validation-based covariance estimation (arXiv:2412.11153):** Practical improvement to the reconciliation weight matrix using out-of-sample rather than in-sample residuals.

5. **Decision-based hierarchy construction (arXiv:2412.11153):** Aligns aggregation levels with operational decision horizons rather than purely statistical groupings.

6. **Applications programme:** Spatio-temporal solar (Solar Energy 2023), intraday realized volatility (JJFE 2024), Italian electricity generation (Applied Energy 2025), portfolio variance via GARCH (2026) — establish reconciliation as a general forecasting improvement tool across domains.

### Open problems identified by Girolimetto

1. **Non-linear constraint reconciliation in practice** — NLCR (arXiv:2510.21249) is theoretical; no CRAN package yet; empirical performance on larger datasets unknown.
2. **ML + probabilistic cross-temporal joint framework** — no published method combines ML base forecasts with cross-temporal probabilistic reconciliation; the arXiv:2303.17277 framework assumes parametric Gaussian base forecasts.
3. **Scalability of coherent forecast combination** — FoCo2 closed form requires inverting large matrices; no lazy/approximate variant exists for very large hierarchies.
4. **Decision-based hierarchy learning** — current decision-based approach requires user-specified decision horizons; data-driven selection not yet addressed.
5. **ML reconciliation with non-Gaussian losses** — current `ctrml_g` uses squared-error; reconciling quantile forecasts under cross-temporal constraints remains open.

---

## 6. Implementations to learn from

### High signal
- **HierarchicalForecast (Python, Nixtla)** — `pip install hierarchicalforecast`. v1.5.1 March 2026. Apache 2.0. Most comprehensive post-hoc reconciliation library. **ERM** (L1-regularised projection) is the closest match for a missing FoRecoML tier.
- **GluonTS DeepVARHierarchical (AWS)** — only mature end-to-end coherent implementation. `coherent_train_samples=True` flag enforces coherence via differentiable projection during training. Cross-sectional only.
- **FoCo2 (R, Girolimetto/Di Fonzo 2024)** — coherent forecast combination; directly relevant for multi-backend ensemble within FoRecoML.
- **NeuralForecast HINT (Nixtla)** — partial end-to-end via reconciliation-trained NN losses.
- **DARTS (Unit8)** — recent hierarchical support; reconciliation as composable transformer.

### Medium signal
- **fable/fabletools (R)** — tidyverts reconciliation verb pattern; `__total` naming convention worth borrowing.
- **MLForecast (Python, Nixtla)** — global ML interface pattern (`unique_id` + separate S matrix). Clean separation of identity from structure.

### Low signal / superseded
- **hts**, **thief** — retired/archived. Nothing new.
- **statsforecast / sktime** — non-ML or limited hierarchical scope.

---

## 7. Recommendations for `ctrml_g`

Ranked by **(impact / effort)** ratio. New recommendations added in v2 are marked ★.

### Tier 1 — Quick wins (1-5 tickets each)

1. **Add `level_id` feature** to `.stack_series()` for global models
   - One additional categorical column distinguishing temporal aggregation level
   - Low effort, high upside per Yingjie/Abolghasemi 2024
   - Backwards-compatible default

2. **Add `method = "erm"` option** to `*_g` functions
   - Calls `FoReco`-or-custom L1-regularised linear projection between MinT (analytic) and full ML
   - Borrows HierarchicalForecast.ERM logic
   - Lightweight alternative when full ML is overkill

3. **Thread `bpv`/`nfca`/`nnic` non-negative methods** through `*_g`
   - FoReco 1.2.0 already implements them at the linear level
   - `sntz` already exposed; Girolimetto (2025) proves `sntz` near-optimal — the three alternatives are expert options for specific use cases
   - Wins users in demand/count/energy hierarchies where forecasts must be non-negative

4. ★ **Add `cov_method = "validation"` option for weight matrix estimation**
   - Replace in-sample residuals with out-of-sample validation residuals in MinT weight matrix
   - Girolimetto & Di Fonzo (arXiv:2412.11153): validation-based approach reduces decision costs, better generalisation
   - Low-effort API extension; requires user to provide a validation split

5. ★ **Document iterative equivalence in `ctrml_g` help page**
   - Girolimetto & Di Fonzo (2025) Stat Methods Appl proves iterative ≡ optimal under standard covariance conditions
   - Users asking "why BU and not MinT?" now have a theorem to cite
   - Pure documentation win, zero code change

### Tier 2 — Medium projects (1-3 epics)

4. **Hierarchy-aware loss for base ML** (Sprangers 2023 pattern)
   - LightGBM/XGBoost custom objective that penalises hierarchical inconsistency at training time
   - Reduces post-hoc adjustment magnitude → smaller reconciliation step → better generalisation
   - Requires writing custom `objective=` for each backend

5. **Learnable projection layer** (Tsiourvas ICML 2024 pattern)
   - Adds `method = "oblique"` returning learned oblique projection
   - Trains a small linear layer jointly with base forecast residuals
   - Likely a new package or sub-module, not an in-place change

6. ★ **FoCo2 ensemble combination within `*_g` functions**
   - When `n_models > 1` base ML fits are available, apply coherent forecast combination (FoCo2) instead of picking the best single model
   - Girolimetto & Di Fonzo (arXiv:2412.03429): simultaneous combination + reconciliation strictly dominates sequential combination then reconciliation
   - `base_combine = "foco2"` flag; requires matrix inversion — may need approximate variant for n × kt > 10⁴
   - Ticket: depends on FoCo2 CRAN stability

7. ★ **Probabilistic `ctrml_g_prob()` function**
   - Girolimetto et al. (IJF 2023, arXiv:2303.17277) provides the complete methodology: parametric Gaussian + bootstrap sampling, multi-step residuals, four covariance structures
   - Would extend FoRecoML to quantile/interval forecasts under cross-temporal constraints
   - Large scope but well-specified; the maths is done

8. ★ **Decision-based hierarchy specification**
   - Let users define `agg_order` via operational decision horizons (e.g., "daily operations" → daily/weekly/monthly) rather than only statistical groupings
   - Girolimetto & Di Fonzo (arXiv:2412.11153): decision-based hierarchies reduce ancillary service penalties in energy market applications
   - API change: `agg_order` accepts named vector with decision labels

### Tier 3 — Research projects (multi-epic)

6. **End-to-end coherent neural variant** — separate package `FoRecoML.deep`, leverages Rangapuram ICML21 / Tsiourvas ICML24 / GluonTS DeepVARHierarchical
7. **Non-tree DAG hierarchy support** (FlowRec 2025) — requires `agg_mat`/`agg_order` API generalisation; affects all of FoReco, not just `_g`
8. ★ **Non-linear constraint reconciliation (NLCR)** — Girolimetto et al. (arXiv:2510.21249); relevant for ratio-constrained hierarchies (mortality rates, unemployment rates, proportional energy mix). Monitor for companion CRAN package; interface: `constraint_type = "nonlinear"` with user constraint function.

---

## 8. Open R&D questions

- Empirical question: how much does `ctrml_g + ctbu` lose vs a hypothetical `ctrml_g + learned oblique projection`? No published comparison exists.
- For very large hierarchies (n × kt > 10⁵), is `ctbu` still the right post-fit step, or does ERM become preferable?
- Does Sprangers-style sparse hierarchical loss extend cleanly to cross-temporal? Public examples are cs-only.
- ★ Can FoCo2 coherent combination be made approximate (e.g., via sketching) for high-dimensional `ctrml_g`?
- ★ What is the correct probabilistic reconciliation approach when ML base forecasts are non-Gaussian (e.g., quantile regression forests)?
- ★ Does the NLCR convergence guarantee (arXiv:2510.21249) extend to cross-temporal non-linear constraints, or is the current proof cross-sectional only?

---

## 9. Girolimetto bibliography — annotated

### Core cross-temporal theory

**[G1]** Di Fonzo, T. & Girolimetto, D. (2023). *Cross-temporal forecast reconciliation: Optimal combination method and heuristic alternatives.* International Journal of Forecasting, 39(1), 39–57. DOI:10.1016/j.ijforecast.2021.08.004. arXiv:2006.08570.
- **Method TL;DR:** First closed-form optimal for simultaneous cs+te reconciliation. Derives iterative procedure improving Kourentzes & Athanasopoulos (2019) two-step heuristic.
- **Innovation:** Unified notation for both dimensions; closed-form LS optimal satisfying both constraint sets simultaneously.
- **Math core:** Optimal: $\tilde{Y} = S_H (S_H' W^{-1} S_H)^{-1} S_H' W^{-1} \hat{Y}$ where $S_H$ is the cross-temporal summing matrix and $W$ the error covariance.
- **Data:** Australian GDP, 95 quarterly series.
- **Result:** Iterative procedure converges; optimal closed-form achieves lower RMSE than sequential alternatives.
- **Relevance to ctrml_g:** High — mathematical foundation of the BU step in `ctrml_g`.
- **Code:** FoReco package.

**[G2]** Girolimetto, D. & Di Fonzo, T. (2025). *Cross-temporal forecast reconciliation: Insights on sequential, iterative, and optimal approaches.* Statistical Methods & Applications. DOI:10.1007/s10260-025-00822-z. arXiv:2410.19407.
- **Method TL;DR:** Proves equivalence conditions between sequential, iterative, and joint-optimal cross-temporal reconciliation. Focuses on computational savings.
- **Innovation:** Convergence theorem: iterative reconciliation reaches optimal regardless of application order under block-diagonal covariance conditions.
- **Math core:** Conditions on $W = \text{blockdiag}(W_{cs}, W_{te})$ under which $\lim_{k\to\infty} \tilde{Y}^{(k)} = \tilde{Y}^*_{opt}$.
- **Data:** Hourly photovoltaic power generation hierarchy.
- **Result:** Significant improvement in memory and computation time vs joint optimal; no accuracy loss under stated conditions.
- **Relevance to ctrml_g:** High — justifies current BU design; motivates `method = "iterative"` option for large hierarchies.
- **Code:** FoReco package.

**[G3]** Girolimetto, D., Athanasopoulos, G., Di Fonzo, T. & Hyndman, R.J. (2023). *Cross-temporal probabilistic forecast reconciliation: Methodological and practical issues.* International Journal of Forecasting. DOI:10.1016/j.ijforecast.2023.10.003. arXiv:2303.17277.
- **Method TL;DR:** Extends probabilistic reconciliation to joint cs+te setting using Gaussian and bootstrap sampling. Introduces multi-step residuals and four covariance alternatives.
- **Innovation:** Multi-step residuals for temporal covariance estimation (one-step residuals fail in the temporal dimension); overlapping residuals technique for dimensionality reduction.
- **Math core:** Sample from $\hat{f}(\hat{Y}) \sim \mathcal{N}(\hat{\mu}, \hat{\Sigma})$ then apply linear reconciliation projection; or resample from bootstrap and project.
- **Data:** Australian GDP, Tourism Demand; evaluated via CRPS and Energy Score.
- **Result:** Joint probabilistic reconciliation outperforms sequential probabilistic approaches.
- **Relevance to ctrml_g:** High — complete blueprint for future `ctrml_g_prob()`.
- **Code:** FoReco package.

### General linear constraints

**[G4]** Girolimetto, D. & Di Fonzo, T. (2023). *Point and probabilistic forecast reconciliation for general linearly constrained multiple time series.* Statistical Methods & Applications. DOI:10.1007/s10260-023-00738-6. arXiv:2305.05330.
- **Method TL;DR:** Extends reconciliation from strict hierarchies to arbitrary linear constraint structures. Structural-like projection separates free from constrained variables.
- **Innovation:** Unified formula for hierarchical, grouped, and accounting-identity constraint systems; probabilistic extension included.
- **Math core:** $\tilde{Y} = \hat{Y} + \Sigma_{Y} C' (C \Sigma_Y C')^{-1} (d - C\hat{Y})$ for general linear constraint $CY = d$.
- **Data:** Australian and European Area GDP (income + expenditure + output × 19 countries).
- **Result:** Fully reconciled point and probabilistic forecasts for complex multi-dimensional accounting constraints.
- **Relevance to ctrml_g:** Medium — FoReco already handles this via `agg_mat`; important for non-standard hierarchy users.
- **Code:** FoReco package.

### Non-negative and non-linear extensions

**[G5]** Girolimetto, D. (2025). *Non-Negative Forecast Reconciliation: Optimal Methods and Operational Solutions.* Forecasting, 7(4), 64. DOI:10.3390/forecast7040064.
- **Method TL;DR:** Derives four non-negative reconciliation methods (bpv, nfca, nnic, sntz); proves sntz is near-optimal with negligible computation cost.
- **Innovation:** First formal optimality proof for non-negative reconciliation; clear ranking of methods for practitioners.
- **Math core:** Constrained LS: $\min_{\tilde{Y}} \|\tilde{Y} - \hat{Y}\|^2_W$ s.t. $S\tilde{b} = \tilde{Y}$, $\tilde{b} \geq 0$. sntz = project then clip negatives to 0 (not guaranteed optimal but empirically near-optimal).
- **Data:** Energy/demand hierarchies (specific dataset from abstract not confirmed).
- **Result:** sntz achieves near-optimal non-negative performance; bpv/nfca/nnic for expert use in specific distributional settings.
- **Relevance to ctrml_g:** High — FoReco already has all four; only sntz exposed in `*_g`.
- **Code:** FoReco 1.2.0+.

**[G6]** Girolimetto, D., Panagiotelis, A., Di Fonzo, T. & Li, H. (2025). *Forecast reconciliation with non-linear constraints.* arXiv:2510.21249.
- **Method TL;DR:** NLCR — projection onto a non-linear constraint surface via constrained optimisation. Sufficient conditions for accuracy improvement derived.
- **Innovation:** First general framework for ratio/multiplicative constraints in reconciliation; extends beyond the linear algebra that underpins all prior work.
- **Math core:** $\min_{\tilde{Y}} \|\tilde{Y} - \hat{Y}\|^2_W$ s.t. $g(\tilde{Y}) = 0$ for non-linear $g$; solved via iterative projected gradient.
- **Data:** Demography (mortality rates) and economics (unemployment rates).
- **Result:** Significant forecast accuracy improvement vs unconstrained benchmarks.
- **Relevance to ctrml_g:** Low (current API assumes linear constraints); high future relevance for energy mix proportions.
- **Code:** None yet.

### Forecast combination innovations

**[G7]** Girolimetto, D. & Di Fonzo, T. (2024). *Coherent forecast combination for linearly constrained multiple time series.* arXiv:2412.03429. [FoCo2 CRAN package.]
- **Method TL;DR:** Joint combination-and-reconciliation of multiple independent base forecast sources. Closed-form solution outperforms sequential combination-then-reconciliation.
- **Innovation:** Unifies forecast combination and reconciliation into a single optimization step; handles unequal numbers of forecasts per variable.
- **Math core:** Generalized least squares combining $K$ independent base forecast vectors subject to $CY = d$; closed-form exists.
- **Data:** Australian electricity generation.
- **Result:** Superior accuracy vs single-task combination and single-expert reconciliation.
- **Relevance to ctrml_g:** Medium-High — directly applicable when users have multiple ML backends.
- **Code:** FoCo2 CRAN package.

**[G8]** Di Fonzo, T. & Girolimetto, D. (2022). *Forecast combination-based forecast reconciliation: Insights and extensions.* International Journal of Forecasting. DOI:10.1016/j.ijforecast.2022.07.001. arXiv:2106.05653.
- **Method TL;DR:** Extends Hollyman et al. (2021) LCC/CCC to endogenous constraints; elucidates reconciliation as constrained quadratic minimization.
- **Innovation:** Shows LCC = exogenous quadratic minimization; endogenous variant revises both upper and bottom series coherently.
- **Data:** Australian Tourism Demand (Visitor Nights).
- **Result:** CCC (average of LCC + BU) achieves competitive accuracy vs state-of-the-art.
- **Relevance to ctrml_g:** Medium — theoretical grounding for combination-reconciliation duality.
- **Code:** FoReco package.

### Decision cost and validation covariance

**[G9]** Girolimetto, D. & Di Fonzo, T. (2024). *Balancing Accuracy and Costs in Cross-Temporal Hierarchies: Investigating Decision-Based and Validation-Based Reconciliation.* arXiv:2412.11153.
- **Method TL;DR:** Two innovations: validation-based covariance estimation (out-of-sample residuals); decision-based hierarchy alignment with operational decision horizons.
- **Innovation:** First paper to consider decision-cost alongside statistical accuracy in reconciliation evaluation; practical for energy/ancillary service applications.
- **Math core:** Replace $\hat{W} = \text{Cov}(\hat{\epsilon}_{train})$ with $\hat{W}_{val} = \text{Cov}(\hat{\epsilon}_{val})$ in the MinT weight matrix.
- **Data:** Wind power forecasting.
- **Result:** Decision-based reconciliation offers better accuracy-cost balance; validation-based covariance reduces revenue losses.
- **Relevance to ctrml_g:** High — directly actionable as `cov_method = "validation"` parameter.
- **Code:** None yet.

### Applications (energy, finance, demographics)

**[G10]** Di Fonzo, T. & Girolimetto, D. (2022). *Enhancements in cross-temporal forecast reconciliation, with an application to solar irradiance forecasts.* arXiv:2209.07146.
- Applies cross-temporal reconciliation to hierarchical PV generation data; establishes relationships between two-step, iterative, and simultaneous procedures for the first time; handles non-negativity.
- **Relevance:** Medium — precursor to [G2]; establishes solar as canonical domain.

**[G11]** Caporin, M., Di Fonzo, T. & Girolimetto, D. (2024). *Exploiting Intraday Decompositions in Realized Volatility Forecasting.* Journal of Financial Econometrics. DOI:10.1093/jjfinec/nbae014. arXiv:2306.02952.
- Applies reconciliation to Dow Jones realized variance hierarchy (index + constituents); improves forecast accuracy via hierarchical structure exploitation.
- **Relevance:** Low for ctrml_g (financial domain; cross-sectional hierarchy).

**[G12]** Di Fonzo, T. & Girolimetto, D. (2023). *Spatio-temporal reconciliation of solar forecasts.* Solar Energy. DOI:10.1016/j.solener.2023.01.003.
- Cross-temporal + spatial reconciliation for PV generation; full non-negativity; beats sequential procedures at all hierarchy levels.
- **Relevance:** Medium — extends [G10]; confirms cross-temporal advantage over sequential for spatial-temporal hierarchies.

**[G13]** Girolimetto, D. & Di Fonzo, T. (2025). *Improving cross-temporal forecasts reconciliation accuracy and utility in energy market.* Applied Energy. DOI:10.1016/j.apenergy.2025.126053.
- Applies cross-temporal reconciliation in energy market context; journal-validated application of [G1]/[G2].
- **Relevance:** Medium — confirms applicability for `ctrml_g` in energy domain use cases.

**[G14]** Girolimetto, D. & Di Fonzo, T. (2025). *Forecasting Italian daily electricity generation disaggregated by geographical zones and energy sources using coherent forecast combination.* arXiv:2502.11878.
- Multi-task coherent combination for hierarchical Italian electricity data; superior to single-task and single-expert approaches.
- **Relevance:** Medium — demonstrates FoCo2 in practice; motivates recommendation #6 (FoCo2 ensemble).

**[G15]** Girolimetto, D. & Di Fonzo, T. (2025). *Energy load forecasting using Terna public data: a free lunch multi-task combination approach.* arXiv:2502.11873.
- Stacked-regression combining Terna operator forecasts with random-walk naives; "significantly more accurate."
- **Relevance:** Low for ctrml_g (combination rather than reconciliation focus).

**[G16]** Caporin, M., Girolimetto, D. & Lopetuso, E. (2026). *Multivariate GARCH and portfolio variance prediction: A forecast reconciliation perspective.* arXiv:2603.17463.
- Reconciliation improves multivariate GARCH under misspecification; noise in covariance proxy key determinant of gain.
- **Relevance:** Low for ctrml_g (financial domain; different problem).

### Software packages

**[G17]** Girolimetto, D. et al. (2026). *FoReco and FoRecoML: A Unified Toolbox for Forecast Reconciliation in R.* arXiv:2604.27696.
- Companion paper for both packages; documents unified design philosophy; dual-focus: accessible defaults + expert control.
- **Relevance:** High — primary reference for FoRecoML architecture decisions.

**[G18]** Girolimetto, D. (2020–present). *FoReco: Forecast Reconciliation* [R package]. CRAN. DOI:10.32614/cran.package.foreco.
**[G19]** Girolimetto, D. et al. (2026). *FoRecoML: Forecast Reconciliation with Machine Learning* [R package]. CRAN. DOI:10.32614/cran.package.forecoml.
**[G20]** Girolimetto, D. & Di Fonzo, T. (2025). *FoCo2: Coherent Forecast Combination for Linearly Constrained Multiple Time Series* [R package]. CRAN.

### Peripheral / non-reconciliation

**[G21]** Girolimetto, D. et al. (2024). *Exploring the impacts of COVID-19 on births in Italy, 2020–2022.* Population Space and Place. DOI:10.1002/psp.2807. [Not open access; not read.]
- Demographic application, no reconciliation methodology. **Not relevant to ctrml_g.**

---

## 10. Anchor citations

### Girolimetto-authored (all open access unless noted)
- **[G1]** Di Fonzo & Girolimetto (2023) IJF 39(1):39–57, arXiv:2006.08570 — cross-temporal optimal combination
- **[G2]** Girolimetto & Di Fonzo (2025) Stat Methods Appl, arXiv:2410.19407 — sequential/iterative/optimal equivalence
- **[G3]** Girolimetto, Athanasopoulos, Di Fonzo & Hyndman (2023) IJF, arXiv:2303.17277 — probabilistic ct reconciliation
- **[G4]** Girolimetto & Di Fonzo (2023) Stat Methods Appl, arXiv:2305.05330 — general linear constraints
- **[G5]** Girolimetto (2025) Forecasting 7(4):64, DOI:10.3390/forecast7040064 — non-negative methods
- **[G6]** Girolimetto, Panagiotelis, Di Fonzo & Li (2025) arXiv:2510.21249 — non-linear constraints
- **[G7]** Girolimetto & Di Fonzo (2024) arXiv:2412.03429 — coherent forecast combination (FoCo2)
- **[G9]** Girolimetto & Di Fonzo (2024) arXiv:2412.11153 — decision/validation-based reconciliation
- **[G17]** Girolimetto, Rombouts, Wilms & Yang (2026) arXiv:2604.27696 — unified FoReco/FoRecoML toolbox

### External SOTA
- Rombouts, Ternes, Wilms (2024) IJF 41(1):321–344 — primary anchor for `ctrml_g`
- Spiliotis et al. (2021) Appl Soft Comp 112:107756 — cs ML reconciliation origin
- Tsiourvas et al. (2024) ICML 235:48713–48727 — learnable oblique projection
- Rangapuram et al. (2021) ICML PMLR 139:8832–8843 — end-to-end coherent NN
- Sprangers et al. (2024) IJF — sparse hierarchical loss at scale
- Neubauer & Filzmoser (2024) arXiv:2407.02367 — BU optimality for ARIMA
- Yingjie & Abolghasemi (2024) arXiv:2411.06394 — local vs global empirical
- Zhou et al. (2024) AAAI — GMP-AR temporal hierarchy message passing

Full bibliography: `research-tasks.md` Phase 2+.
