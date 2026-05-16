# `ctrml_g` Recommendations — Prioritized v2 (analyst-reconciled + gate-revised)

**Date:** 2026-05-16
**Version:** v2 (post design-review-gate iter 1)
**Status:** Re-submitted to design-review-gate
**Inputs:**
- `docs/research/2026-05-16-global-ml-ctrml-sota.md` (SOTA v2 from Girolimetto bibliography)
- Analyst doc `ctrml_g_hat_redesign.md` (production constraints from a 3437-series CT hierarchy, ticket `vakanzeiten-u7ow`)
- Design-review-gate iter 1 findings (5 specialists; 4 of 5 NEEDS_REVISION, 1 APPROVED)

## Changes in v2

- **ncol_hat guard added to `predict.rml_g_fit`** (already landed — R/rml_g.R:1077-1082). Was specified in Epic ak9 design but missed in implementation; fixed now.
- **`cs_level` DEFERRED from P0 to backlog** — architectural blocker: `.stack_series()` stacks bottom-only series, so `cs_level` would be constant across all training rows (information-free). True usefulness requires upper-aggregate stacking, a larger change.
- **`is_bottom` DROPPED** — perfectly collinear with `cs_level`; tree models can derive any threshold split from `cs_level` alone.
- **`level_id` derivation pinned**: requires wiring `kset`/`agg_order` into `.stack_series()` (currently reserved but unused per rml_g.R:24).
- **Open questions Q1–Q4 RESOLVED** in §6.
- **NA threshold X = 30%** for cov warning.
- **Caller-side pipeline epic stub added** to track LOOO CV persistence.
- **Backwards-compat now programmatic**, not just documented (ncol_hat guard enforces).

## Production constraints (re-stated for completeness)

1. **Sparse bottom layer**: ~40% structural zeros → Σ_ii → 0 catastrophic if imputed.
2. **Parallel reconcilers**: VD (`tew="sum"`) + V (`tew="avg"`) symmetric.
3. **High fit volume**: 144 LOOO + 6 production fits per run.
4. **Downstream pass**: `csrec(comb="ols")` after `ctrml_g`.

## Reconciled priority table (v2)

| Rank | Recommendation | v1 Tier | v2 Position | Effort |
|------|---------------|---------|-------------|--------|
| 1 | `level_id` in `.stack_series()` (terml_g/ctrml_g) | T1 | **P0 — ship next** | Low |
| 2 | Docs (NEWS + roxygen + pkgdown article) co-landed | docs | **P0 — co-land with #1** | Trivial |
| 3 | `cov_method="validation"` + NA-safe Σ | T1 | **P1** (separate epic, requires caller-side LOOO persistence) | Medium |
| 4 | Thread `bpv`/`nfca`/`nnic` (expert opt-in) | T1 | **P3** (backlog, behind `nonneg_method=` flag) | Low |
| 5 | `cs_level` — multi-level stacked global training | (new in v1 prio) | **Backlog → requires architectural change** | High |
| 6 | ERM (L1-projection) `method="erm"` | T1 | **P2 — backlog** | Medium |
| 7 | Probabilistic CT reconciliation | T2 | **Backlog** | Medium |
| 8 | GMP-AR temporal message passing | T2 | **Backlog** | Medium |
| 9 | Learnable oblique projection (Tsiourvas) | T2 | **Backlog → `FoRecoML.deep`** | Research |
| 10 | End-to-end coherent NN | T3 | **Backlog → `FoRecoML.deep`** | Research |
| 11 | DAG / non-tree hierarchy (FlowRec) | T3 | **Backlog** | Research |

**DROPPED:**
- Hierarchy-aware training loss (Sprangers 2023) — architectural conflict with post-fit reconciliation; gradient undefined on suppressed rows.
- FoCo2 inside `_g` — wrong layer; relocated to caller-side recommendations (§7).

## P0 spec — `level_id`

### Why
Yingjie & Abolghasemi (2024 arXiv:2411.06394) + analyst production evidence: global LightGBM without cross-hierarchy features fails to differentiate level-specific correction magnitude (annual aggregates near-deterministic; monthly bottom series high variance).

### Encoding
Ordered integer, NOT one-hot:
- `1 = month` (finest granularity, highest frequency)
- `2 = bi-month`
- `3 = quarter`
- `…`
- `max = annual` (coarsest, lowest frequency)

Convention aligns with FoReco `agg_order` semantics (`agg_order` values are aggregation multipliers; smaller value = finer granularity period in original units, but the `m = max(agg_order)` invariant means `level_id=1` corresponds to high-frequency leaves). Q4 resolution: **finer-to-coarser ascending**.

### Mechanism
`.stack_series()` currently ignores `kset` (rml_g.R:24 — "reserved for T7.4; v1 ignores it"). P0 wires it:

1. Caller (terml_g/ctrml_g) passes `kset` (or derives from `agg_order`) into `.stack_series()`.
2. `.stack_series()` derives a per-column `level_id` mapping by splitting hat's `ncol_hat` columns into temporal aggregation blocks (size determined by `agg_order` proportions per the FoReco layout).
3. When `level_id = TRUE`, the per-column mapping is broadcast to a per-row feature: each stacked row inherits the level_id of the time column its (series, time) cell came from. NOTE: this assumes hat is laid out by temporal level within row — confirm against FoReco's `FoReco2matrix` convention before implementation.
4. Resulting stacked feature matrix gains one extra column.

### Scope per framework
- **csrml_g**: NOT applicable (no temporal levels). Accepting `level_id = TRUE` returns a clean `cli_abort` ("level_id is not applicable to csrml_g; cs hierarchy has no temporal axis").
- **terml_g**: APPLICABLE.
- **ctrml_g**: APPLICABLE.

### Default
`level_id = FALSE` (backwards compatible).

### DoD
- `R CMD INSTALL --preclean .` clean
- Unit test: `.stack_series()` with `level_id = TRUE` produces expected column count and value range
- LightGBM feature importance on a synthetic DGP with strong level-effect: `level_id` importance > 5% of total (deterministic since DGP designed with level signal — see §8 test spec)
- Coherency invariant preserved: `agg_mat %*% bottom == top`, tol 1e-10
- Backwards compat: existing tests pass without modification
- Programmatic guard: fit with `level_id=TRUE` then predict on newdata with wrong ncol → `cli_abort` (already enforced by ncol_hat guard)
- NEWS.md entry under existing 2.0.0 section
- pkgdown article `vignettes/articles/feature-engineering.Rmd` (or similar) with two sections: **Practitioners** (sparsity + flag rationale) and **Researchers** (Yingjie/Abolghasemi citation, ordinality vs one-hot)

## P1 spec — `cov_method="validation"` + NA-safe Σ

### Scope clarification
This is a **future** feature, NOT in the current `*_g` BU code path. Current `csrml_g`/`terml_g`/`ctrml_g` pipelines:
1. fit global ML on (hat, obs, series_id)
2. predict on `apply_norm_params(base, norm_params)`
3. apply `FoReco::csbu/tebu/ctbu` — these are bottom-up, **no Σ used**

P1 introduces a NEW reconciliation path that uses validation residuals as Σ input to a MinT-style projection. This means adding a `method = c("bu", "mint")` argument (or similar) to `*_g`. Out of scope for P0.

### Why
Girolimetto & Di Fonzo (2024 arXiv:2412.11153): validation residuals beat training residuals for MinT weight matrix. Decision-cost-aware.

### API
```r
ctrml_g(..., method = c("bu", "mint"),
            cov_method = c("insample", "validation"),
            val_residuals = NULL,
            cov_na_action = c("exclude", "pairwise"))
```

Q3 resolution: **default `cov_na_action = "exclude"`** when validation residuals have >30% NA; warn-and-default to `"pairwise"` otherwise. Production system has 40% NA → "exclude" is the safer default.

### NA contract
- `val_residuals` may contain `NA` for suppressed cells
- Σ estimator MUST use pairwise complete obs OR series-level exclusion — **NEVER impute to zero**
- Warning when >30% of cells in `val_residuals` are NA: "Validation residuals contain {pct}% NA — consider `cov_na_action = 'exclude'`."

### Q2 resolution
`val_residuals` accepts a **matrix only** for P1 (n_series × T_val). Nixtla-style data.frame deferred to a separate ticket if user demand warrants.

### Dependency
P1 requires a caller-side ticket (§7) that persists LOOO CV residuals. Cannot ship P1 end-to-end without that infrastructure.

### DoD
- `method = "mint"` path produces reconciled output with same coherency invariant
- NA injection tests: 30%/50% NA → finite Σ; >60% NA → explicit warning AND `cli_abort` if `cov_na_action = "pairwise"`
- Symmetry: `tew="sum"` and `tew="avg"` produce different but both-finite Σ-based outputs

## Resolved open questions (Q1–Q4)

| Q | Question | Resolution |
|---|----------|------------|
| Q1 | Auto-derive `cs_level` or caller-supplied? | **Moot — `cs_level` DEFERRED.** With bottom-only stacking, `cs_level` is information-free. Backlog item, requires architectural change. |
| Q2 | `val_residuals` matrix vs data.frame? | **Matrix only for P1.** Data.frame deferred. |
| Q3 | `cov_na_action` default? | **"exclude"** (production 40% NA rate makes pairwise near-singular). Auto-detect via NA percentage. |
| Q4 | `level_id` ordering? | **1=finest (month) → max=coarsest (annual)**. Aligns with `m = max(agg_order)` invariant. |

## Explicit rejections (carried from v1)

### REJECTED: Hierarchy-aware training loss (Sprangers 2023)
- Architectural conflict with post-fit reconciliation
- Gradient undefined on suppressed rows
- Sprangers assumes dense cs-only hierarchy
- Coherence-in-loss path = Tsiourvas-style E2E differentiable (separate module, `FoRecoML.deep`)

### REJECTED for `_g`: FoCo2 inside reconciler
- Wrong layer (belongs at orchestration)
- Stays a pipeline-layer recommendation (§7)

### DEFERRED: Tsiourvas oblique projection
- Separate architecture, separate module

### DEMOTED: bpv/nfca/nnic
- `sntz` near-optimal (Girolimetto 2025 *Forecasting* 7(4):64)
- Degenerate-face risk on 40% structural-zero hierarchies
- Thread as P3 expert opt-in only

## Backwards-compat contract (now programmatic)

1. All new params default to `FALSE` / `NULL` / first option in `c()`
2. Existing post-Epic ak9 tests pass unchanged
3. **Programmatic enforcement** of column-count consistency:
   - `fit_obj$ncol_hat` stored at fit time (already done, all 4 backends)
   - `predict.rml_g_fit` guard checks `ncol(newdata) == object$ncol_hat` (LANDED in R/rml_g.R:1077-1082 as part of this revision)
   - When `level_id=TRUE` adds a feature column, `ncol_hat` reflects the post-encoding width
   - Test: fit with `level_id=TRUE`, predict on newdata without `level_id` → clear `cli_abort`

## Caller-side pipeline recommendations (separate epic stub)

These are OUT of `ctrml_g` scope but should be filed as a parallel caller-side beads epic. Tracked here for reference:

- **LOOO CV persistence** (prereq for P1)
- **FoCo2 ensemble** at model-selection layer (Girolimetto & Di Fonzo 2024 arXiv:2412.03429)
- **Decision-cost-driven model selection** (Girolimetto & Di Fonzo 2024 arXiv:2412.11153)

Beads ticket to file: `caller(vakanzeiten-u7ow): pipeline infrastructure for ctrml_g P1`.

## Test plan (TDD per CLAUDE.md)

### P0 tests (RED → GREEN → REFACTOR per category)

**T1: `.stack_series()` level_id column shape**
- RED: write `expect_equal(ncol(.stack_series(hat, obs, level_id=TRUE)$X_stacked), ncol(hat) + 1L)`
- GREEN: implement level_id wiring in `.stack_series()`
- REFACTOR: extract level_id-mapping helper

**T2: `.stack_series()` level_id value range (1 ≤ x ≤ max(agg_order))**
- RED → GREEN → REFACTOR same pattern

**T3: csrml_g rejects `level_id=TRUE`**
- RED: `expect_error(csrml_g(level_id=TRUE, ...), "level_id is not applicable")`
- GREEN: add cli_abort

**T4: predict ncol_hat guard fires on flag mismatch**
- Already landed; add regression test:
- `fit <- terml_g(..., level_id=TRUE); expect_error(predict(fit, newdata=wrong_ncol), "must have .* columns")`

**T5: LightGBM importance on synthetic level-effect DGP**
- DGP: simulate base forecasts with strong level-dependent bias
- Assertion: `level_id` importance > 5% of total

**T6: Coherency preserved (existing invariant test extended for level_id=TRUE)**

### P1 tests (deferred until P0 lands and pipeline infra ready)

(Specs in §4 above.)

## Anchor citations

- Yingjie & Abolghasemi (2024) arXiv:2411.06394 — local vs global, level_id critical
- Girolimetto & Di Fonzo (2024) arXiv:2412.11153 — validation-based covariance
- Girolimetto (2025) Forecasting 7(4):64 — sntz near-optimal
- Sprangers et al. (2024) IJF — hierarchy-aware loss (rejected)
- Tsiourvas et al. (2024) ICML 235:48713 — learnable projection (deferred)
- Girolimetto & Di Fonzo (2024) arXiv:2412.03429 — FoCo2 (caller-side)

## Implementation epic outline

Single epic with one ticket (one functional change keeps it atomic but P0 has multiple files):
- **Epic: P0 — level_id feature for global ML wrappers**
  - T1: implement `.stack_series()` level_id wiring + value mapping
  - T2: csrml_g rejection guard + terml_g/ctrml_g signature param
  - T3: NEWS + roxygen + pkgdown article
  - T4: regression tests (T1–T6 above)

P1 epic to be filed when P1 caller-side prereqs are met.
