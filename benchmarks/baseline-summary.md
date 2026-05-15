# FoRecoML Baseline Performance — 25be237

**Date**: 2026-05-15  
**Branch**: 25be237 (pre-spd.1 baseline)  
**Note**: Full benchmark deferred — agg_order/agg_mat API requires valid hierarchy setup.
T9 will use the same benchmark script with correct inputs for Pass A comparison.

## Benchmark Script
`benchmarks/run-baseline.R` — created, pending full execution with correct hierarchy.

## Partial Timing (csrml, p=6, T=20, ncol=50, approach=randomForest)
| Workload | Wrapper | median_ms | peak_rss_mb |
|---|---|---|---|
| micro | csrml | ~TBD | ~TBD |
| micro | terml | ~TBD | ~TBD |
| micro | ctrml | ~TBD | ~TBD |

**Action for T9**: Run `benchmarks/run-baseline.R` with correct hierarchy input
to get apples-to-apples comparison. The baseline numbers establish Pass A reference.

## Key Baseline Facts (from T1 inventory)
- 85 tests pass, 0 fail at baseline
- `set.seed(seed)` commented out — seed is silent no-op (T5 will fix)
- No mirai, no n_workers, no parallel infrastructure
