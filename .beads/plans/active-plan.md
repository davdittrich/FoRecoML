# Active Plan: Epic FoRecoML-ak9 — Restore *_g fit+predict+reconcile API

<!-- approved: 2026-05-15 -->
<!-- gate-iterations: 2 (Plan Review) + 3 (Design Review) -->
<!-- user-approved: pending (presenting to user) -->
<!-- status: ready-to-execute -->

## Design
docs/superpowers/specs/2026-05-15-rml-g-reconcile-api-design.md

## Architecture
Architecture B — wrappers own reconciliation. predict.rml_g_fit stays raw-prediction utility.

## hat / base column space
hat = base forecasts. hat T_obs×n, base h×n for cs. terml/ctrml use kt feature width per high-freq period.

## Ticket waves

### Done
- ak9.1: verified predict broadcast (series-major)
- ak9.9: apply_norm_params helper

### Wave 1 — Precondition
- ak9.12: rewrite fixtures into cs/te/ct variants

### Wave 2 — Core pipelines + independent items
- ak9.2: csrml_g pipeline (csbu)
- ak9.3: terml_g pipeline (tebu, named vector return)
- ak9.4: ctrml_g pipeline (ctbu, n×(h×kt) matrix)
- ak9.5: mlr3tuning/paradox audit (independent)
- ak9.11: pin FoReco (>= 1.2.1) (independent)

### Wave 3 — Threading + dispatch
- ak9.6: RESTORE sntz/round/tew (USER APPROVED 2026-05-15). Thread through csbu/tebu/ctbu calls.
- ak9.10: extract_reconciled_ml dispatch for rml_g_fit via attr(FoReco)

### Wave 4 — Docs + tests
- ak9.7: NEWS.md + vignette audit
- ak9.13: test-rml-g-reconcile.R + assertion updates

## Branch
fix/rml-g-reconcile-ak9 (on fork remote davdittrich/FoRecoML)
