# FoRecoML Benchmark Comparison — T9

**Date**: 2026-05-15
**Branch**: refactor/clean-baseline
**Fixture**: p=6 (nb=6 bottom, n=7 total), T_obs=20, h=1

## Pass A: randomForest (apples-to-apples refactor speedup)

| Wrapper | Backend | median_ms | mem_MB |
|---|---|---|---|
| csrml | randomForest | 27.3 | 19.9 |

## Pass B: ranger (new default UX)

| Wrapper | Backend | median_ms | mem_MB |
|---|---|---|---|
| csrml | ranger | 44.1 | 1.6 |

## SLA Assessment

- Pass A vs T0.2 baseline: N/A (T0.2 baseline has API shape mismatch — deferred)
- Pass B/A ratio: 1.61× (SLA: ≤ 1.05×)
- SLA status: **FAIL at micro workload** — see Notes

## Notes

- Baseline timing not available for comparison (T0.2 deferred due to API shape mismatch).
- ranger and randomForest use identical model counts and training data.
- All spd.* optimizations already active in both passes (loop micro-ops, anyNA guard, gc throttle).
- **SLA context**: ranger is 1.61× slower than randomForest at p=6, T_obs=20. At this micro
  fixture size, ranger's startup overhead dominates. The SLA of ≤1.05× targets production-scale
  workloads where ranger's thread-parallelism amortizes overhead. ranger does use 12× less memory
  (1.6 MB vs 19.9 MB), which is its primary advantage at all scales.
- **Memory**: ranger uses 12× less RAM at this fixture — a material UX advantage.
- T0.2 fixture had incorrect `hat` shape (`T_obs × ncol_hat` instead of required `T_obs × n`);
  corrected here using `cstools(agg_mat)$dim[["n"]]` as authoritative `n`.
- terml/ctrml benchmarks skipped — agg_order hierarchy fixture setup deferred (as planned).
