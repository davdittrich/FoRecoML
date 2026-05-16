# Active Plan: Bugfix Epic FoRecoML-eo2

<!-- approved: 2026-05-17 -->
<!-- gate-iterations: 4 (T4 P2 overridden) -->
<!-- status: ready-to-execute -->

## Execution order
T1 → T2 → T3; T4 independent

## Tickets
- FoRecoML-eo2.1 (T1): sort series_id_levels in .run_chunked_rml_g() — 1 line
- FoRecoML-eo2.3 (T2): temporal block validation split in .stack_series()
- FoRecoML-eo2.4 (T3): 10 regression tests (B1/B2 + T2 temporal split)
- FoRecoML-eo2.5 (T4, P2, override): cli_warn on feature_importance errors — grep-verify only

## Key design decisions
- T2: temporal block = last n_per_series rows per series; min_validation_rows guard preserved
- T2: seed removed from validation path (deterministic); seed for chunk shuffle untouched
- T3: Group A (B1/B2 fixes) + Group B (temporal split behavior)
- T4: no automated test; verify post-implementation by grep count
