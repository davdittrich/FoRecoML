# Active Plan: Verify Epic FoRecoML-bba

<!-- approved: 2026-05-17 -->
<!-- gate-iterations: 1 -->
<!-- status: ready -->

## Background
Analyst report 2026-05-17 (forecoml_bugs_and_features (1).md) was largely outdated. Live verification (commit fc735af + fresh R CMD INSTALL):
- BUG-1: FIXED (analyst confirms)
- BUG-2: FIXED (regression test landed fd500f1)
- BUG-3 = FEAT-1: IMPLEMENTED (wide_ct epic 3st)
- BUG-4: NOT REPRODUCIBLE — all 3 backends pass with prod-scale agg_order=c(12,6,4,3,2,1). False positive from analyst's stale install.
- FEAT-1/2/3: IMPLEMENTED

Only actionable: lock in wide_ct multi-backend regression tests so BUG-4 cannot resurface from a real bug.

## Tickets
- FoRecoML-bba.1 (T1): 3 wide_ct regression tests — lightgbm, ranger, lightgbm+prod-scale
