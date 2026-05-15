# G.1 (spd.27) Research Plan: Global Pre-train + Per-series Fine-tune

Ticket: FoRecoML-hvf.1 (P4, research-only).

## Context

Current architecture trains p independent ML models (one per bottom series). At large p (2432+) with randomForest: 240 GB cumulative model storage; with lightgbm: still 5–10 GB. The proposed global model replaces p models with 1 global model trained on all-series stacked features, with optional per-series fine-tuning.

**Memory savings if valid**: p models → 1 model. 99.9% reduction at p=2432.
**Accuracy risk**: global model may underfit heterogeneous series if per-series dynamics differ substantially.
**Interaction with B19**: if global model eliminates p-model checkpoint overhead, the motivation for B19 (daemon hat sharing) becomes orthogonal — B19 solves the hat-transfer problem, G.1 solves the model-storage problem.

## I. Research Hypotheses

### H1 (Statistical validity)
A single ML model trained on stacked (y, X, series_id) across all p series produces reconciled forecasts with RMSE ≤ 1.05× the per-series baseline (within 5% degradation tolerance).

### H2 (Memory reduction)
Global model storage ≈ constant (independent of p) vs per-series ∝ p. At p=2432, savings ≥ 100×.

### H3 (Aggregation compatibility)
Global model + temporal bottom-up (tebu/ctbu/csbu) produces coherent forecasts satisfying the linear aggregation constraints. Coherence is guaranteed post-reconciliation regardless of ML model structure.

### H4 (Feature heterogeneity)
Per-series feature scales vary. Series indicator (one-hot or integer) is insufficient when series span multiple magnitudes. Normalization strategy required for valid global training.

### H5 (Transfer / fine-tune benefit)
Per-series fine-tuning of the global model (few-shot gradient steps on series-specific data) recovers any global-model accuracy gap. Benefit depends on per-series sample size.

## II. Experimental Setup

### Literature Survey (E1)
1. Search: "global model forecasting", "pooled forecasting", "meta-learning time series", "N-BEATS cross-learning".
2. Key references: Montero-Manso & Hyndman (2021) "Principles and Algorithms for Forecasting Groups of Time Series"; Oreshkin et al. (2020) N-BEATS.
3. Focus on: cross-series pooling benefits, when local outperforms global, reconciliation-specific evidence.
4. Output: `docs/research/g1-literature-survey.md` with ≥5 relevant citations.

### Fixture Specification (E2)
Use the FoRecoML standard fixtures for consistency. ALL fixtures use **80% train / 20% holdout** temporal split; baseline and global model evaluated on identical holdout (prevents in-sample overfitting conflation).

- Small: itagdp hierarchy (n_bts=8, p=8, kt=7, N_obs=30) — from existing tests
- Medium: synthetic (n_bts=50, p=50, kt=28, N_obs=100) — monthly aggregation
- Large: synthetic (n_bts=200, p=200, kt=28, N_obs=200) — same
- Stress-test: synthetic (n_bts=500, p=500, kt=28, N_obs=200) — upper memory limit

**User-scale (p=2432, 5-min) is EXCLUDED from in-memory research.** Stacked matrix = 5.9M rows × 2432 cols = 115 GB — infeasible for in-memory stacking. If PROCEED verdict issued, implementation would require chunked streaming training (separate implementation ticket).

For each fixture: ground truth is per-series lightgbm RMSE on the holdout (current architecture). Global model target: stacked (X_all, y_all, series_id) trained on the same feature engineering pipeline.

### Feasibility Experiment (E3)
Implement minimal global rml prototype as a standalone research script (NOT production code):
```r
# Research script only — dev/g1-global-prototype.R
# Stack all series
X_stack <- do.call(rbind, lapply(seq_len(p), function(i) cbind(X_i, series_id = i)))
y_stack <- c(obs[, 1], obs[, 2], ..., obs[, p])

# Train global model
global_model <- lightgbm::lgb.train(...)

# Per-series predict
for (i in seq_len(p)) {
  Xtest_i <- ...  # per-series test features
  bts_i <- predict(global_model, cbind(Xtest_i, series_id = i))
}
```
Stacking strategy options to benchmark:
- A: No normalization, raw series values + series_id integer
- B: Per-series z-score normalization before stacking + denormalize after
- C: Log transform if positive-valued series
- D: Per-series z-score + categorical embedding (lgb native)

### Benchmark Protocol (E4)
For each (fixture_size × stacking_strategy) pair:
1. Compute `r_perseries <- ctrml(...)` (baseline)
2. Compute `r_global <- g1_global(...)` (prototype)
3. RMSE comparison: `mean((r_perseries$bts - truth)^2)` vs `mean((r_global$bts - truth)^2)`
4. Wall-clock: global train time vs sum of p per-series train times
5. Memory: `pryr::mem_used()` peak during training (global vs serial per-series)

Tools: `bench::mark`, GC-cleared, 3+ iterations. Report: ratio_rmse, ratio_wallclock, ratio_memory.

### Fine-tune Experiment (E5)
After global training, apply N gradient steps on each series:
- lightgbm: `lgb.train(... init_model = global_model, nrounds = 10)`
- xgboost: `xgb.train(... xgb_model = global_model, nrounds = 10)`
Measure RMSE recovery vs pure global model.

## III. Decision Rubric

| Outcome | Criteria |
|---|---|
| **PROCEED** | E1 finds supporting evidence; H1 holds for ≥ 3 fixture sizes (RMSE ratio ≤ 1.05); H3 confirmed (coherent output); H5 shows fine-tune closes gap if any |
| **DEFER** | RMSE gap > 5% but fine-tune (E5) closes it; requires architectural support for per-series fine-tune in production (new ticket) |
| **REJECT** | RMSE gap > 10% across all strategies AND fine-tune fails; OR H4 insurmountable without per-series normalization that's incompatible with production pipeline |

## IV. Deliverables

1. `docs/research/g1-literature-survey.md` — ≥5 citations + synthesis
2. `dev/g1-bench/g1-global-prototype.R` — standalone research script (NOT production)
3. `dev/g1-bench/g1-benchmark.R` — benchmark script across fixtures × strategies
4. `docs/research/g1-findings.md` — benchmark tables, RMSE ratios, wall-clock, memory
5. `docs/research/g1-decision.md` — PROCEED / DEFER / REJECT memo with evidence (separate from findings)
6. `docs/research/g1-limitations.md` — scope limitations (user-scale exclusion, xgboost strategy-D exclusion, randomForest fine-tune exclusion)
7. If PROCEED: `docs/research/g1-design-doc.md` with:
   - rml.<backend>_global S3 method spec
   - stacking strategy recommendation
   - series normalization approach
   - interaction with existing loop_body + checkpoint machinery
   - DoD for implementation ticket

## V. Out of Scope

- Production code changes (until PROCEED)
- Modification of existing ML backend S3 methods
- Any changes to rml() / loop_body_kset / wrappers
- GPU offload (B20, separate ticket)
- Fine-tune for non-gradient backends (randomForest has no incremental fit)

## VI. Risks to Research

| ID | Risk | Mitigation |
|---|---|---|
| RR1 | Stacking explodes memory at p=2432 (stack = 2432 × N_obs rows) | Use chunked stacking; measure peak memory before OOM |
| RR2 | lightgbm categorical series_id with 2432 categories may underfit due to grouping | Test with integer encoding; compare strategy A vs D; xgboost strategy D excluded (no native categorical). Document scope. |
| RR3 | "Truth" for RMSE comparison — 80/20 split canonicalized in E2; OOF comparison prevents overfitting inflation | Pre-specified; per-series and global both evaluated on identical holdout |
| RR4 | Wall-clock comparison unfair (global model trains once vs p serial; not parallel) | Compare against p-serial baseline AND parallel-p-worker baseline |
| RR5 | User-scale (p=2432, 5-min): stacked matrix = 115 GB, infeasible in-memory | Excluded from research scope. Noted in deliverable 6. Chunked streaming left for implementation phase. |

## VII. Success Criteria

- [ ] Literature survey complete (≥5 citations synthesized)
- [ ] Prototype script runs on itagdp fixture without error
- [ ] Benchmark results for ≥3 fixture sizes (small/medium/large)
- [ ] RMSE ratio documented for ≥2 stacking strategies
- [ ] PROCEED / DEFER / REJECT issued with evidence
- [ ] If PROCEED: design doc drafted

## VIII. Non-research note

This research directly addresses the largest remaining memory bottleneck (p-model storage) that B1-B19 cannot solve. If PROCEED: implementation eliminates the need for per-series checkpoint serialization and potentially makes B4 (ranger default) and B3 (auto-promote checkpoint) less critical at large p.
