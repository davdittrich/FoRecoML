# Plan: Deprecate Outer Parallelism

## I. Objective

Remove mirai-based outer parallelism (n_workers > 1) from rml()/ctrml()/terml()/csrml(). Retain all single-process memory/CPU optimizations. Replace the 28-formal explicit-closure dispatch with a plain for-loop using lexical scoping. Net effect: simpler code, identical performance for n_workers=1 (which is the empirically optimal config for typical use cases).

**Reference**: Analyst confirmed n_workers=3 was 4× SLOWER than n_workers=1 at T_obs=72. Outer parallelism is a net negative for gradient boosters on small-N data.

## II. What Is Removed

- `mirai` from DESCRIPTION:Imports → move to nowhere (no longer needed)
- `mirai::daemons()`, `mirai::mirai_map()`, `mirai::everywhere()` calls in rml()
- `.args` list construction (all 28 formals shipped to daemons)
- `loop_body_kset` and `loop_body_csrml` as standalone closures with explicit formals. Replaced by inline helper with lexical scoping OR retained as helpers but called directly (no daemon transport needed).
- B2 (pool= parameter) — remove from all signatures
- B3 (auto-promote in-memory fits to tempdir checkpoint for parallel) — remove the promote path; keep checkpoint logic for single-process large fits
- B5/B16 (outer chunk dispatch loop for multi-daemon) — remove the ENTIRE chunk loop structure (`chunk_starts`, `for (cs in chunk_starts)` wrapper). The inner for-loop `for (i in seq_len(p))` is the replacement. batch_size kept only in ctrml_g global path.
- B19 (Arrow IPC shared hat) — remove entirely
- `n_workers` parameter — deprecate with `lifecycle::deprecate_warn()` or silent no-op; effectively always 1
- `pool` parameter — remove

## III. What Is Retained

| Component | Why kept |
|---|---|
| loop_body_* memory logic | spd.2 anyNA, spd.3/8 gc throttle, spd.12 defer input2rtw, spd.13 defer mat2hmat, spd.14 hoist FoReco2matrix, spd.15 global_id_list, spd.16-20 per-iter opts |
| Checkpoint machinery | qs2 serialize_fit/deserialize_fit/get_fit_i — valuable for large p even single-process |
| spd.9 n_workers-aware checkpoint threshold | Simplify: remove n_workers factor; keep base threshold |
| spd.10 in-memory fit cap | Now irrelevant (no parallel), remove the auto-cap; keep the detection for user warning |
| B4 ranger default + ML-aware threshold | Keep |
| B6 direct index formula | Keep (CPU optimization) |
| B7 qs2 nthreads | Keep (checkpoint perf) |
| B8 dim()<- | Keep |
| B9 drop sel_mat from .args | Now irrelevant (no .args); keep the resolve_global_id_only logic itself |
| B10 lazy NULL na_cols_list | Keep |
| C.1 terml mfh fix | Keep |
| ctrml_g/terml_g/csrml_g + G.2 chunking | Keep (single-process, new architecture) |
| normalize_stack, robscale | Keep |
| catboost backends | Keep |

## IV. Loop Body Refactor

### Pre-removal (complex)
```r
loop_body_kset <- function(i, hat, obs, base, sel_mat, col_map, class_base,
                            approach, active_ncol, params, fit, checkpoint_dir,
                            kset, dots, h, n, parts_hat, parts_base, ..., gc_every) { ... }

# dispatched via:
if (n_workers_resolved > 1L) {
  out <- mirai::mirai_map(seq_len(p), loop_body_kset, .args = list(
    hat=hat, obs=obs, ..., gc_every=gc_every))[]
} else {
  for (i in seq_len(p)) out[[i]] <- loop_body_kset(i, hat=hat, obs=obs, ...)
}
```

### Post-removal (simple)
```r
# loop_body_kset becomes an internal closure with lexical access to rml() scope.
# No explicit formal list needed — R's lexical scoping provides everything.
# Either inline OR a nested function:

loop_body_inner <- function(i) {
  # accesses hat, obs, parts_hat, sel_mat, etc. from rml() scope
  # same logic as before, no argument passing
}

for (i in seq_len(p)) out[[i]] <- loop_body_inner(i)
```

**Benefit**: Eliminates entire .args construction, dramatically simplifies code, removes all "pass N formals explicitly" overhead. loop_body_csrml and loop_body_kset merge back into one inner function (csrml takes the `is.null(kset)` branch, kset path takes the other — same as before B1).

## V. n_workers Deprecation Strategy

Option A (clean): Remove from all signatures. Emit `lifecycle::deprecate_warn("0.x.0", "ctrml(n_workers=)")` if passed.
Option B (silent): Keep in signature as `n_workers = NULL`, silently ignore any value > 1 with a single `cli_warn` if n_workers was explicitly set > 1.

Recommend **Option A** for clarity. Users get an explicit signal to remove the argument.

## VI. Files Affected

| File | Change |
|------|--------|
| R/reco_ml.R | Remove mirai dispatch, pool/B19/B2/B3 code, simplify loop, merge loop_body back to lexical |
| R/utils.R | Remove promote_fit_to_checkpoint(), B19 helpers (b19_daemon_load, normalize_stack for IPC — keep normalize_stack for global path), qs_nthreads_adaptive kept, available_ram_bytes kept |
| R/ctrml.R | Remove n_workers/pool/shared_hat params; simplify batch_size (only for ctrml_g) |
| R/terml.R | Same |
| R/csrml.R | Same |
| DESCRIPTION | Remove mirai from Imports |
| tests/ | Remove parallel-path tests (test-parallel.R, test-b1-*, test-b5-*, test-b9-*, test-b16-*, test-b19-*) or gut to smoke-only |

## VII. Step-by-Step Logic

### Step 1: Remove mirai and B2/B3/B5/B16/B19 code from rml() (R/reco_ml.R)
- Remove `mirai::daemons()` block + `on.exit(daemons(0))` + `mirai::everywhere()`
- Remove `if (n_workers_resolved > 1L) { mirai_map(...) } else { for-loop }` → keep ONLY the for-loop
- Remove B19 block (ipc_hat, col_remap, use_b19, arrow IPC)
- Remove B2 block (pool= param, pool_provided guard)
- Remove B3 block (promote_fit_to_checkpoint call)
- Remove B5/B16 chunked dispatch (chunk_size, chunk loop) — keep ctrml_g's batch_size separately
- Merge loop_body_kset + loop_body_csrml into a single inner function with lexical scoping (or keep as separate named inner functions for readability — both work)
- Remove all .args construction, global_id_list from .args, gc_every from .args (GC still happens but can be simplified)

### Step 2: Remove n_workers/pool from all wrapper signatures
- ctrml.R: remove n_workers, pool, shared_hat from ctrml() + ctrml_fit()
- terml.R: same
- csrml.R: same
- Add `lifecycle::deprecate_warn` or cli_warn if n_workers was passed by user (via ...)
- batch_size stays in ctrml_g/terml_g/csrml_g ONLY

### Step 3: Remove now-dead utilities from R/utils.R
- promote_fit_to_checkpoint() — remove
- b19_daemon_load() — remove
- arrow_available() — remove
- qs_nthreads_adaptive() — keep (still used by serialize_fit checkpoint)
- resolve_n_workers() — remove entirely (no longer needed)
- cap_inner_threads() — remove (no multi-worker to cap for)
- compute_chunk_size() / compute_chunk_size_nonmfh — remove (outer dispatch chunking gone; G.2 has its own batch_size logic)

### Step 4: DESCRIPTION cleanup
- Remove `mirai` from Imports (check all files for mirai:: calls first)
- Keep: qs2, arrow (Suggests for B19 removal? → yes, remove from Suggests too since B19 gone)

### Step 5: Test cleanup
- Remove or gut test-parallel.R (all daemon-path tests become invalid)
- Remove test-b1-dispatch-closure.R (B1 .args test)
- Remove test-b5-chunk-size.R, test-b16-nonmfh-chunk.R
- Remove test-b19-shared-hat.R
- Keep: all equivalence tests (spd.12/13/14), all new _g tests, all ML backend tests, checkpoint tests, normalization tests

### Step 6: devtools::document() + run tests

### Step 7: Commit
Single commit: `refactor(rml): remove outer parallelism; single-process sequential path`
Body: explains the architectural decision (inner threading > process-level for small N_obs gradient boosters), references analyst finding.

## VIII. Output Schema
```
files_modified: [R/reco_ml.R, R/utils.R, R/ctrml.R, R/terml.R, R/csrml.R, DESCRIPTION, man/, tests/]
functions_removed: [promote_fit_to_checkpoint, b19_daemon_load, arrow_available, resolve_n_workers, cap_inner_threads, compute_chunk_size]
params_removed: [n_workers, pool, shared_hat] from all non-_g wrappers
test_count_delta: ~ -48 (parallel tests removed: test-parallel ~22, test-b1-dispatch ~5, test-b5 ~7, test-b16 ~5, test-b19 ~9) + 0 (no new tests)
commit_count: 1
```

## IX. DoD
- [ ] `mirai` completely removed from DESCRIPTION and all R/ files
- [ ] rml() uses simple for-loop only (no mirai_map)
- [ ] loop_body as lexical inner function (no 25-formal explicit dispatch)
- [ ] n_workers/pool/shared_hat removed from ctrml/terml/csrml (warning if passed)
- [ ] B19/B2/B3/B5/B16 code removed from rml()
- [ ] resolve_n_workers/cap_inner_threads/promote_fit_to_checkpoint removed
- [ ] All spd.*/B* single-process optimizations intact
- [ ] Checkpoint still works for single-process (serialize_fit, qs2)
- [ ] ctrml_g/terml_g/csrml_g + G.2 batch_size unchanged
- [ ] devtools::check() passes (no mirai references)
- [ ] All test_count_post ≥ (test_count_pre - parallel_tests_removed)
- [ ] Single commit; conventional commits; no AI attribution

## X. Risks

- **R1**: Test suite has many parallel-path tests. Removing them reduces coverage count but not meaningful coverage (the removed tests test removed code).
- **R2**: Any external user calling `ctrml(n_workers=3)` will get deprecation warning and sequential execution — correct behavior.
- **R3**: checkpoint threshold (spd.9's n_workers-aware 0.5 factor) becomes irrelevant — revert to single threshold (0.8 × avail_ram). Simpler.
- **R4**: loop_body_* simplification from explicit 25-formal to lexical: lexical closures capture EVERYTHING in scope including large objects (hat, obs, etc.). In a for-loop this is fine (no serialization needed). Performance identical to before.
