# FoRecoML — Project Context

## What this package does

Nonlinear forecast reconciliation using ML. Three main frameworks:

- **Cross-sectional** (`csrml`): hierarchy reconciliation via ML (Spiliotis et al. 2021)
- **Temporal** (`terml`): temporal aggregation reconciliation
- **Cross-temporal** (`ctrml`): joint cross-sectional + temporal (Rombouts et al. 2024)

Entry point for all three is `reco_ml()`.

## Architecture

```
R/
  reco_ml.R      # dispatcher — routes to cs/te/ct variants
  csrml.R        # cross-sectional ML reconciliation
  terml.R        # temporal ML reconciliation
  ctrml.R        # cross-temporal ML reconciliation
  FoReco.R       # re-exports from FoReco dependency
  utils.R        # shared helpers
```

## ML Backend System

- `mlr3` + `mlr3learners` + `mlr3tuning` + `paradox` for unified learner API
- Supported backends: `randomForest`, `lightgbm`, `xgboost`, `ranger`, `catboost`
- Parallelism via `mirai` (async futures)
- Serialization via `qs2` (fast binary, checkpoint/resume)

## Performance-Critical Paths

- Chunked training loops (b5-chunk-size, b16-nonmfh-chunk tests)
- Shared hat matrix caching (b19-shared-hat test)
- RAM-cache management (test-available-ram-cache)
- Lazy NA column dropping (b10-lazy-na-cols)
- Dispatch closure construction (b1-dispatch-closure)

## Test Topology

- `tests/testthat/fixtures/` — shared test fixtures
- `test-b*.R` — bug regression tests (numbered by issue)
- `test-g*.R` — global ML scenario tests
- `test-csrml.R`, `test-ctrml.R` — main functional tests
- `test-checkpoint.R` — qs2 serialization tests
- `test-catboost.R` — optional backend (Suggests only)

## Benchmarking

- Benchmarks live in `dev/` (`b19-bench/`, `g1-bench/`, `spd*.R`)
- Remote host: `192.168.1.43` via `remote-bench` skill
- Use `r-benchmarking` skill for new timing experiments
- Always interleaved before/after in single `bench::mark()` call

## CI Matrix

GitHub Actions: mac-release, win-release, ubuntu-devel, ubuntu-release, ubuntu-oldrel-1

## Key Constraints

- `FoReco` is a hard dependency (upstream reconciliation math)
- `mirai` workers must not inherit global state
- `qs2` checkpoint files are NOT committed to git
- `catboost` is Suggests-only (not on CRAN) — guard with `skip_if_not_installed()`
