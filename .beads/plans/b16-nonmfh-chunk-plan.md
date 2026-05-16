# Plan: Extend Series Chunking to Non-mfh Paths

## Context

B5 (FoRecoML-drq, merged) added outer-loop chunking for mfh dispatch paths (h non-NULL). Non-mfh paths (kset non-NULL, h NULL) use `chunk_size = p` (no chunking). Developer diagnosis: OOM inside targets/crew worker with n_workers=3 is caused by 3 concurrent daemons each holding parts_hat + building per-series X via input2rtw_partial_from_parts simultaneously, PLUS crew worker already holding all_folds data.

Per-daemon peak memory: `parts_hat + X_for_one_series (sequential per daemon)`. With n_workers=3 concurrent: `3 × (parts_hat + X_current_series)`.

Outer chunking allows more frequent gc() between batches, bounds max concurrent series across daemon pool to chunk_size, and reduces peak series-concurrent X allocation when chunk_size < p.

## Ticket B16: Extend chunking to non-mfh dispatch paths

### I. Objective
Out: non-mfh dispatch (kset non-NULL, h NULL) also uses RAM-adaptive `compute_chunk_size`; chunk_size < p triggers chunked outer loop with gc() between chunks, same as mfh. Both mfh and non-mfh get bounded peak.

### II. Input
- R/reco_ml.R: `chunk_size` computation block (currently: `if (!is.null(h)) compute_chunk_size(...) else p`)
- R/utils.R: `compute_chunk_size` — currently only has mfh formula

### III. Guards

| Type | Guard |
|------|-------|
| Logic | Non-mfh per-iter X estimate: `NROW(hat) * total_cols * 8` where `total_cols = max(keep_cols)` (from spd.12 T5 branch) OR `active_ncol`. Formula gives conservative upper bound per-series X bytes. |
| Format | `compute_chunk_size` gains `h` param (or pass-through remains). OR: add separate `compute_chunk_size_nonmfh(...)`. Prefer extending existing function for DRY. |
| Boundary | R/reco_ml.R + R/utils.R. Wrappers UNTOUCHED. |
| Audit | BYTE-IDENTICAL output (chunked == unchunked, tolerance=0) for non-mfh fixture — same test structure as B5 byte-identity test. |

### IV. Step-by-Step

1. Read current `compute_chunk_size` in R/utils.R. Currently parameterized by `(p, h, n, kt_eff, available_bytes, n_workers)` — designed for mfh.

2. Add non-mfh variant logic. Two options:
   - **Option A** (extend `compute_chunk_size`): add `hat_rows = NULL, cols = NULL` params; when `!is.null(hat_rows)` use non-mfh formula.
   - **Option B** (inline in rml()): compute non-mfh chunk_size inline using `active_ncol` and `NROW(hat)` already in scope.

   Prefer Option B (simpler, reuses existing rml() locals):
   ```r
   chunk_size <- if (!is.null(h)) {
     compute_chunk_size(p, h, n, ncol_total_hat, available_ram_bytes(), n_workers_resolved)
   } else if (!is.null(kset)) {
     # non-mfh: per-iter X ≈ NROW(hat) * max(kset) * active_ncol * 8 bytes
     # NOTE: max(kset) is a CONSERVATIVE OVERESTIMATE. True per-iter X row count = NROW(hat)
     # (input2rtw expands then contracts back to NROW(hat) per kset level).
     # Overestimation intentional: yields safely smaller chunk_size without correctness risk.
     hat_nrow <- if (!is.null(hat)) NROW(hat) else if (!is.null(base)) NROW(base) else 1L
     row_rep_factor <- if (!is.null(kset)) max(kset) else 1L
     per_iter_bytes_nonmfh <- hat_nrow * row_rep_factor * active_ncol * 8L
     target_nonmfh <- 0.2 * available_ram_bytes()
     max_c <- target_nonmfh %/% (per_iter_bytes_nonmfh * max(1L, n_workers_resolved))
     min(p, max(1L, as.integer(max_c)))
   } else {
     p   # csrml (kset=NULL): no parts_hat expansion, no chunking needed
   }
   ```

3. `available_ram_bytes()` uses F.3 cache — same as B5. No double-read.

4. Dispatch loop remains unchanged from B5's implementation (already chunks both sequential + parallel dispatch). Only the `chunk_size` value changes for non-mfh.

5. When `active_ncol` is 0 or formula produces `max_c >= p`: chunk_size = p (no chunking, identical to pre-B16).

6. Tests:
   - `test-b16-nonmfh-chunk.R`: inject small `.rml_cache$ram_bytes` to force chunk_size < p on ctrml compact fixture; assert byte-identical bts vs unchunked. Use `withr::defer` for cleanup.
   - Unit test: non-mfh chunk_size formula produces 1 at tight RAM, p at ample RAM.

7. Run `Rscript -e 'devtools::test()' 2>&1 | tail -5`. All 367+ pass.

8. Single commit: `perf(rml): extend series chunking to non-mfh dispatch paths (B16)`.

### V. Output Schema
```
sites_modified: R/reco_ml.R (chunk_size computation, ~5 lines)
new_tests: tests/testthat/test-b16-nonmfh-chunk.R
test_count_pre: 367
test_count_post: >= 370
commit_count: 1
```

### VI. DoD
- [ ] Non-mfh dispatch (kset!=NULL, h=NULL) uses adaptive chunk_size
- [ ] csrml (kset=NULL) remains chunk_size = p (no chunking — csrml doesn't use input2rtw_partial)
- [ ] BYTE-IDENTICAL test for non-mfh ctrml compact: chunked == unchunked at tolerance=0
- [ ] withr::defer used for cache cleanup in test
- [ ] spd.12+13+14 equivalence byte-identical
- [ ] mw3.3 invariant intact
- [ ] Single commit

## Risks

- **R1**: `active_ncol` is computed in rml() outer scope but the exact value at the non-mfh chunk_size computation point must be verified (comes after T5 keep_cols/col_map setup).
- **R2**: Conservative estimate (sel_mat=1 full-density) may produce over-conservative chunk_size for sparse features (compact, mfh-str-bts). This is acceptable — erring toward smaller chunks is safe.
- **R3**: For predict-reuse (hat=NULL), use `NROW(base)` instead. When both NULL (no Xtest), skip chunking for per-iter X (no training data peak).
- **R4**: This addresses GC frequency and series-concurrency but NOT parts_hat replication across daemons. The root OOM in the targets/crew scenario has two drivers: parts_hat (n × h_train*kt bytes per daemon) and per-iter X. B16 addresses #2; B1 addressed #1 partially. Full fix requires B19 (Arrow shared hat, research).

## Out of scope
- csrml chunking (csrml uses direct hat slicing, not input2rtw_partial — per-iter X is hat[, id] which is much smaller)
- Changing n_workers semantics inside targets/crew (separate docs recommendation)
- B19 Arrow shared hat
