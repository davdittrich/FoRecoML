# Global ML Cross-Temporal Reconciliation — SOTA Review

**Date:** 2026-05-16
**Audience:** FoRecoML maintainers
**Trigger:** Inform future improvements to `ctrml_g()` after Epic FoRecoML-ak9 restoration
**Sources:** see `research-tasks.md` Phases 2+3 (Semantic Scholar, arXiv, GitHub, ICML/AAAI proceedings)

## TL;DR

- The Rombouts/Wilms/Girolimetto 2024 paper is the **only published method** combining ML reconciliation with cross-temporal hierarchies. `ctrml_g()` implements it. No direct competitor exists.
- **`ctrml_g()` already occupies a unique niche.** Closest peers (Spiliotis 2021, Sprangers 2023) are cross-sectional only.
- **3 SOTA patterns** the package does NOT yet borrow:
  1. Differentiable / learnable reconciliation projection (Tsiourvas ICML 2024 — *oblique projection*)
  2. End-to-end coherent training loss (Sprangers 2023 *sparse hierarchical loss*; GluonTS *DeepVARHierarchical*)
  3. ERM (L1-regularised projection) as a middle tier between MinT and full nonlinear ML (HierarchicalForecast)
- **Theoretical bar:** bottom-up is optimal for ARIMA temporal hierarchies (Neubauer & Filzmoser 2024). ML gains must come from departures from ARIMA structure (nonlinearity, cross-series features).

## 1. Problem definition

Given a cross-temporal hierarchy with:
- `n` cross-sectional series (`na` aggregate + `nb` bottom)
- temporal aggregation orders `agg_order = (k₁, k₂, …)`; `m = max(agg_order)`
- `h` low-frequency forecast horizons → `h_hf = h × m` high-frequency periods

Produce reconciled forecasts at all `n × (k* + m)` combinations such that:
- cs constraint: `agg_mat %*% bottom_cs == upper_cs` per temporal period
- te constraint: `sum(high-freq within period) == low-freq` per series

`csbu`/`tebu`/`ctbu` are bottom-up: forecast leaves, sum upward. MinT/OLS/WLS are post-hoc weighted combinations. ML reconciliation (Spiliotis/Rombouts) generalises the combination function from linear to nonlinear.

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

## 3. Where `ctrml_g()` stands

**Strengths**
- Only published implementation combining global ML training with full cross-temporal reconciliation in a single function call
- Post-fit reconcile path (csbu/tebu/ctbu) is mathematically well-founded
- `extract_reconciled_ml()` enables fit reuse across new base forecasts
- `sntz`/`round`/`tew` restored matches series-level parity
- Test fixtures now respect `ncol(hat) == ncol(base)` invariant

**Gaps vs SOTA**
- **No end-to-end training loss** — base ML model is fit with vanilla MSE/quantile loss; reconciliation happens after. SOTA neural methods train with coherency in the loss.
- **No learnable projection** — Tsiourvas 2024 shows oblique projection beats fixed (MinT) by significant margin. `ctbu` is a fixed orthogonal projection.
- **No ERM tier** — sits between MinT (closed-form) and full nonlinear ML; sparse interpretable; valuable for moderately-sized hierarchies.
- **No DAG / non-tree hierarchy support** — FlowRec 2025 generalises beyond trees; FoReco/`ctrml_g` assume tree structure via `agg_mat`/`agg_order`.
- **No non-negative ML reconciliation** — FoReco 1.2.0 added 4 non-negative algorithms (bpv, nfca, nnic, sntz) but only `sntz` is exposed in `*_g`. The post-hoc post-projection algorithms (bpv, nfca, nnic) are not threaded through.

## 4. Cross-temporal-specific findings

### Sequential vs iterative vs optimal (Girolimetto & Di Fonzo 2025)
Proves: iterative cross-temporal reconciliation **converges to optimal regardless of application order** (cs-then-te or te-then-cs); sequential matches optimal under specific covariance conditions.

→ **For `ctrml_g`**: current single-call ctbu is already optimal under the bottom-up assumption. No need to add sequential/iterative options unless users explicitly want non-BU.

### Temporal-hierarchy message passing (Zhou et al. 2024 GMP-AR, AAAI)
Granularity message passing across temporal levels enriches base forecasts before reconciliation; adaptive node-dependent weighting beats uniform MinT. Applied at Alipay.

→ **For `ctrml_g`**: applicable in cross-temporal setting (extension of temporal-only GMP-AR). Architectural change — not a small patch.

### Bottom-up is optimal for ARIMA temporal hierarchies (Neubauer & Filzmoser 2024)
First theoretical proof. Implies: ML cross-temporal reconciliation gains must come from **nonlinearity** and **cross-series features**, not from departures from BU per se.

→ **For `ctrml_g`**: current ctbu post-fit step is theoretically optimal under linear ARIMA. Improvements must target the base ML stage (richer features, hierarchy-aware loss) or replace ctbu with a learnable projection.

### Local vs global empirical (Yingjie & Abolghasemi 2024)
Systematic comparison: global LightGBM dominates per-series local models on accuracy + compute across cs-hierarchy levels. Cross-hierarchy features (`series_id`, `level_id`) are critical.

→ **For `ctrml_g`**: current `.stack_series()` adds `series_id`. Adding `level_id` (which temporal aggregation level a row corresponds to) as an additional categorical feature is a low-effort improvement.

## 5. Implementations to learn from

### High signal
- **HierarchicalForecast (Python, Nixtla)** — `pip install hierarchicalforecast`. v1.5.1 March 2026. Apache 2.0. Most comprehensive post-hoc reconciliation library. **ERM** (L1-regularised projection) is the closest match for a missing FoRecoML tier.
- **GluonTS DeepVARHierarchical (AWS)** — only mature end-to-end coherent implementation. `coherent_train_samples=True` flag enforces coherence via differentiable projection during training. Cross-sectional only.
- **NeuralForecast HINT (Nixtla)** — partial end-to-end via reconciliation-trained NN losses.
- **DARTS (Unit8)** — recent hierarchical support; reconciliation as composable transformer.

### Medium signal
- **fable/fabletools (R)** — tidyverts reconciliation verb pattern; `__total` naming convention worth borrowing.
- **MLForecast (Python, Nixtla)** — global ML interface pattern (`unique_id` + separate S matrix). Clean separation of identity from structure.

### Low signal / superseded
- **hts**, **thief** — retired/archived. Nothing new.
- **statsforecast / sktime** — non-ML or limited hierarchical scope.

## 6. Recommendations for `ctrml_g`

Ranked by **(impact / effort)** ratio:

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
   - `sntz` already exposed; the other three are similar dispatch
   - Wins users in demand/count hierarchies

### Tier 2 — Medium projects (1-3 epics)

4. **Hierarchy-aware loss for base ML** (Sprangers 2023 pattern)
   - LightGBM/XGBoost custom objective that penalises hierarchical inconsistency at training time
   - Reduces post-hoc adjustment magnitude → smaller reconciliation step → better generalisation
   - Requires writing custom `objective=` for each backend

5. **Learnable projection layer** (Tsiourvas ICML 2024 pattern)
   - Adds `method = "oblique"` returning learned oblique projection
   - Trains a small linear layer jointly with base forecast residuals
   - Likely a new package or sub-module, not an in-place change

### Tier 3 — Research projects (multi-epic)

6. **End-to-end coherent neural variant** — separate package `FoRecoML.deep`, leverages Rangapuram ICML21 / Tsiourvas ICML24 / GluonTS DeepVARHierarchical
7. **Non-tree DAG hierarchy support** (FlowRec 2025) — requires `agg_mat`/`agg_order` API generalisation; affects all of FoReco, not just `_g`

## 7. Open R&D questions

- Empirical question: how much does `ctrml_g + ctbu` lose vs a hypothetical `ctrml_g + learned oblique projection`? No published comparison exists.
- For very large hierarchies (n × kt > 10⁵), is `ctbu` still the right post-fit step, or does ERM become preferable?
- Does Sprangers-style sparse hierarchical loss extend cleanly to cross-temporal? Public examples are cs-only.

## 8. Anchor citations

- Rombouts, Ternes, Wilms (2024) IJF 41(1):321–344 — primary anchor for `ctrml_g`
- Girolimetto & Di Fonzo (2025) Stat Methods Appl — sequential/iterative/optimal proof
- Spiliotis et al. (2021) Appl Soft Comp 112:107756 — cs ML reconciliation origin
- Tsiourvas et al. (2024) ICML 235:48713–48727 — learnable oblique projection
- Rangapuram et al. (2021) ICML PMLR 139:8832–8843 — end-to-end coherent NN
- Sprangers et al. (2024) IJF — sparse hierarchical loss at scale
- Neubauer & Filzmoser (2024) arXiv:2407.02367 — BU optimality for ARIMA
- Yingjie & Abolghasemi (2024) arXiv:2411.06394 — local vs global empirical
- Zhou et al. (2024) AAAI — GMP-AR temporal hierarchy message passing

Full bibliography: `research-tasks.md` Phase 2.
