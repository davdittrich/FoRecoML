## I. Objective

In: ctrml mfh path materializes `mat2hmat(hat, h, kset, n)` output `h × (n*kt)` in wrapper at lines 321 (training), 459 (predict base), 622 (ctrml_fit training).
Out: defer mat2hmat into loop_body via new `mat2hmat_partial(mat, h, kset, n, cols)`. Per-iter materialization of only `length(global_id)` output columns.
Scope: ctrml mfh only. terml mfh (vec2hmat → h × kt, no n-factor) UNTOUCHED. csrml UNTOUCHED. ctrml non-mfh (spd.12 territory) UNTOUCHED.

## I-bis. Benefit analysis (corrected from v2)

**Raw mfh hat**: `n × (h*kt)` = `n*h*kt` doubles.
**Expanded mfh hat (mat2hmat output)**: `h × (n*kt)` = `h*n*kt` doubles.
**Byte count: IDENTICAL.** mat2hmat is a reshape, not a row-replication.

**Real benefits of spd.13:**
1. **Main-process peak transient saved**: pre-spd.13 main holds BOTH raw hat AND expanded hat momentarily during mat2hmat. Post-spd.13 only raw hat. Saves transient `n*h*kt` doubles in main process. Real for users on near-OOM boundary at orchestrator.
2. **Per-iter daemon X shrinks for SPARSE mfh features** (mfh-hfbts, mfh-hfts, mfh-bts, mfh-str, mfh-str-bts, mfh-str-hfbts, mfh-str-hfts). Pre-spd.13: per-iter X slice from pre-expanded hat is `h × |id_i|` where id_i is local-col map of selected features. Post-spd.13: per-iter X is `h × |global_id_i|`. For sparse sel_mat, |global_id_i| << total_cols, so per-iter peak memory drops.
3. **Per-iter daemon X for mfh-all** (sel_mat=1, dense): NO savings. global_id_i = all cols.
4. **Daemon mirai .args closure size**: UNCHANGED. Both raw and expanded hat have same byte count. `n_workers × closure_bytes` floor unchanged. spd.13 does NOT fix mfh OOM at high n_workers.

**Honesty in commit message**: "Defers mat2hmat materialization from wrapper into per-iter loop_body. Saves main-process peak transient (one full hat copy) and per-iter daemon X memory for sparse mfh feature modes. Daemon closure size unchanged. Symmetry with spd.12 non-mfh deferral."

## II. Input (post-spd.12 state)

- R/FoReco.R:30-35 `mat2hmat(mat, h, kset, n)` — current impl: output `h × (n*sum(m/kset))` via `order(i)` + `matrix(byrow=TRUE)`
- R/ctrml.R mfh sites: 321 (training), 459 (predict base), 622 (ctrml_fit training) — fresh grep confirmed; all 3 inside `else` of `if (!grepl("mfh", features))`
- R/ctrml.R wrapper-side mfh NA block: lines 397-426 (and mirror at site 3 inside `is.null(keep_cols)` path) — must be deleted
- R/reco_ml.R: rml() has `kset = NULL`; loop_body has 14 formals; line 117 comment "14-item closure list spec"; spd.12 `na_cols_list` + `h_train` plumbed; mw3.3 invariant at line 214; predict-reuse NA retrieval at lines 177-182
- R/utils.R: `new_rml_fit()` has `na_cols_list`, `h_train` (spd.12)
- Branch: spd.13 builds atop `spd.12-defer-input2rtw` HEAD 4efeef9

## III. Guards

- **Mechanism:** new `mat2hmat_partial(mat, h, kset, n, cols)` in R/utils.R. rml() gains `h = NULL, n = NULL` formals. loop_body 14 → 16 formals. 3-way X dispatch on `(kset, h)`: (a) `is.null(kset)` → `hat[, id]` (csrml); (b) `!is.null(kset) && is.null(h)` → `input2rtw_partial(hat, kset, cols=global_id)` (spd.12 non-mfh); (c) `!is.null(kset) && !is.null(h)` → `mat2hmat_partial(hat, h, kset, n, cols=global_id)` (spd.13 mfh). Xtest dispatch mirrors.
- **Forbidden:** modifying mat2hmat (FoReco.R:30) or input2rtw_partial (utils.R); modifying mw3.3 invariant; modifying spd.12 NA persistence machinery (na_cols_list / h_train); touching terml / csrml / ctrml non-mfh.
- **Audit:** mat2hmat_partial byte-equivalence unit test (DoD §2). End-to-end max_abs_diff == 0 equivalence (DoD §3) covering all mfh modes × ML libs + NA + predict-reuse + horizon-mismatch.
- **NA semantics**: identical Option A as spd.12. Per-series local `na_col_mask(X)` post-expansion in loop_body. **mfh loop_body branch MUST set `na_mask <- na_cols`** (symmetric to non-mfh kset branch) so `na_cols_list` populates for predict-reuse. The existing NA block at reco_ml.R lines 143-149 must extend to also fire on mfh branch (or be relocated outside the kset-non-mfh sub-branch so it applies to BOTH expansion paths uniformly).
- **Predict-reuse**: reuses spd.12 mechanism unchanged. Line 177 guard `!is.null(kset) && !is.null(fit$na_cols_list)` fires for mfh (kset non-NULL).
- **h_train guard**: ctrml mfh wrapper currently derives `h <- NCOL(base) / kt` at predict site. spd.12 fix-up #2 stores `h_train` in fit; guard compares at predict. spd.13 preserves this. No new validation.

### B2 fix — keep_cols / active_ncol coherence

ctrml mfh wrapper MUST pass NON-NULL `keep_cols` to rml() so that the existing T5 branch (rml() lines 100-110, post-spd.12) takes effect (active_ncol derived from `max(keep_cols)` which equals `total_cols` when sel_mat is dense, or correctly bounded by sel_mat-derived indices when sparse).

Per-site change in ctrml.R mfh:
```r
total_cols <- tmp$dim[["n"]] * tmp$dim[["kt"]]   # B4 fix — explicit formula
features_size <- total_cols
# ... sel_mat construction unchanged ...
keep_cols <- sel_mat_keep_cols(sel_mat, total_cols)   # B2 fix — non-NULL for mfh
```

For mfh-all (sel_mat == 1): `sel_mat_keep_cols(1, total_cols) = seq_len(total_cols)` = `seq_len(n*kt)`. Length = n*kt. Pass to rml() with keep_cols non-NULL. T5 branch fires.

For sparse mfh (e.g., mfh-hfbts where sel_mat is sparseVector): `sel_mat_keep_cols` returns `which(sel_mat != 0)`. Indices in `[1, n*kt]`. Pass to rml().

In rml() with non-NULL keep_cols + h, n non-NULL:
- `active_ncol <- max(keep_cols)` (T5 branch, already in code)
- col_map derived from sel_mat_keep_cols indices (already in code)
- loop_body called with kset, h, n → 3-way dispatch to mat2hmat_partial

NO new rml() branch logic needed. The `is.null(keep_cols)` path is for non-spd.12-aware callers only and stays unchanged.

### B3 fix — wrapper mfh NA block deletion

Currently at ctrml.R lines 397-426 (and mirror site 3 ~line 700ish), the mfh path runs:
```r
na_local <- na_col_mask(hat)
if (any(na_local)) sel_mat[na_local, ] <- 0
```
on the materialized hat. Post-spd.13 hat is RAW (n × h*kt), so `na_col_mask(hat)` operates on wrong-shaped matrix.

DELETE both wrapper mfh NA blocks. Per-series local NA detection moves into loop_body's mfh branch (symmetric to spd.12 non-mfh; see Guards above).

### B5 fix — mfh loop_body branch populates na_mask

In loop_body, post-spd.13 the NA block must apply to BOTH non-mfh and mfh expansion outputs. Cleanest: structure as

```r
if (is.null(kset)) {
  X <- hat[, id, drop = FALSE]
} else {
  X <- if (is.null(h)) {
    input2rtw_partial(hat, kset, cols = global_id)
  } else {
    mat2hmat_partial(hat, h, kset, n, cols = global_id)
  }
  # NA filter — applies to BOTH spd.12 and spd.13 paths
  na_cols <- FoRecoML:::na_col_mask(X)
  if (any(na_cols)) {
    X <- X[, !na_cols, drop = FALSE]
    global_id_post_na <- global_id[!na_cols]
    na_mask <- na_cols
  }
}
```

Existing `na_mask <- NULL` initializer at function entry (spd.12) stays. `na_mask` assignment within the unified post-expansion NA block ensures `na_cols_list[[i]]` populates correctly for BOTH non-mfh and mfh predict-reuse.

## IV. Logic

### 1. R/utils.R — `mat2hmat_partial`

Algorithm (verified by hand on n=2,h=2,kset=c(1,2) fixture):

```r
mat2hmat_partial <- function(mat, h, kset, n, cols) {
  if (length(cols) == 0L) return(matrix(numeric(0), nrow = h, ncol = 0))
  m <- max(kset)
  i <- rep(rep(rep(seq_len(h), length(kset)), rep(m / kset, each = h)), n)
  vec <- as.vector(t(mat))
  ord <- order(i)
  ncol_total <- length(vec) %/% h
  if (any(cols < 1L | cols > ncol_total)) {
    cli::cli_abort("`cols` out of range [1, {ncol_total}].")
  }
  idx <- as.vector(outer((seq_len(h) - 1L) * ncol_total, cols, "+"))
  matrix(vec[ord[idx]], nrow = h, ncol = length(cols))
}
```

CORRECTNESS: `identical(mat2hmat_partial(m, h, k, n, cols), mat2hmat(m, h, k, n)[, cols, drop = FALSE])` MUST hold for all (m, h, k, n, cols) combinations. Mandatory unit test (DoD §2).

### 2. R/reco_ml.R — rml() + loop_body

a. rml() signature: add `h = NULL, n = NULL` after kset, before `...`.

b. loop_body formals (16): `function(i, hat, obs, base, sel_mat, col_map, class_base, approach, active_ncol, params, fit, checkpoint_dir, kset, h, n, dots)`. Line 117 comment "14-item" → "16-item (h, n added in spd.13 for mfh deferred expansion)".

c. Inside loop_body, X branch (replace lines 137-150):

```r
if (is.null(kset)) {
  X <- hat[, id, drop = FALSE]
} else {
  X <- if (is.null(h)) {
    input2rtw_partial(hat, kset, cols = global_id)
  } else {
    mat2hmat_partial(hat, h, kset, n, cols = global_id)
  }
  na_cols <- FoRecoML:::na_col_mask(X)
  if (any(na_cols)) {
    X <- X[, !na_cols, drop = FALSE]
    global_id_post_na <- global_id[!na_cols]
    na_mask <- na_cols
  }
}
```

d. Xtest branch (replace lines 185-192):

```r
if (is.null(base)) {
  Xtest <- NULL
} else if (is.null(kset)) {
  Xtest <- base[, id, drop = FALSE]
} else if (is.null(h)) {
  Xtest <- input2rtw_partial(base, kset, cols = global_id_post_na)
} else {
  Xtest <- mat2hmat_partial(base, h, kset, n, cols = global_id_post_na)
}
```

e. Sequential dispatch (reco_ml.R ~line 224-229): append `h = h, n = n`.
f. mirai_map .args (reco_ml.R ~line 243-248): append `h = h, n = n`.

### 3. R/ctrml.R — 3 mfh sites

Per site (lines 321, 459, 622 — verify with fresh grep):

**Site 1 (training, ~321):**
- DELETE `hat <- mat2hmat(hat, h = h, kset = tmp$set, n = tmp$dim[["n"]])`
- KEEP `h <- NCOL(hat) / tmp$dim[["kt"]]` (still derivable from raw hat: NCOL=h*kt)
- REPLACE `total_cols <- NCOL(hat)` with `total_cols <- tmp$dim[["n"]] * tmp$dim[["kt"]]` (B4 fix)
- KEEP sel_mat construction (sized to total_cols)
- ADD `keep_cols <- sel_mat_keep_cols(sel_mat, total_cols)` (B2 fix — make non-NULL for mfh too)
- DELETE wrapper mfh NA block (`na_local <- na_col_mask(hat); sel_mat[na_local] <- 0` etc.) — B3 fix
- Pass `kset = tmp$set, h = h, n = tmp$dim[["n"]]` to rml() at the rml() call site for this training path

**Site 2 (predict base, ~459):**
- DELETE `base <- mat2hmat(base, h = h, kset = tmp$set, n = tmp$dim[["n"]])`
- KEEP `h <- NCOL(base) / tmp$dim[["kt"]]` derivation (h_train guard depends on this)
- Pass `kset = tmp$set, h = h, n = tmp$dim[["n"]]` to rml() at predict-reuse call site

**Site 3 (ctrml_fit training, ~622):**
- Mirror of site 1.

### 4. UNTOUCHED

R/terml.R, R/csrml.R, R/ctrml.R non-mfh path, R/FoReco.R, R/utils.R::input2rtw_partial.

## V. Schema

```
sites:
  - file: R/utils.R
    delta: +mat2hmat_partial(mat, h, kset, n, cols)
  - file: R/reco_ml.R
    fn: rml
    delta: +h=NULL, +n=NULL formals; propagate to sequential .args + mirai_map .args
  - file: R/reco_ml.R
    fn: loop_body
    delta: +h, +n formals (16); X 3-way dispatch; Xtest 3-way dispatch; NA block unified across kset paths; na_mask populated symmetrically
  - file: R/reco_ml.R
    line: ~117
    delta: comment "14-item" -> "16-item"
  - file: R/ctrml.R
    sites: [~321, ~459, ~622]
    delta: per-site: DELETE mat2hmat; DELETE wrapper mfh NA block; explicit total_cols = n*kt; ADD non-NULL keep_cols; pass h, n, kset to rml()
  - file: R/terml.R, R/csrml.R, FoReco.R
    delta: NONE
```

## VI. DoD

1. `devtools::test()` baseline 156/156 still passes.
2. **mat2hmat_partial byte-equivalence unit test** at `tests/testthat/test-mat2hmat-partial.R`. Random fixtures (set.seed) with n ∈ {1, 3, 7}, h ∈ {1, 4, 8}, kset ∈ {c(1), c(1,2), c(1,2,4), c(1,3,6,12)}, cols ∈ {full, single, empty, contiguous range, random subset}. Assert `identical(mat2hmat_partial(m, h, k, n, cols), mat2hmat(m, h, k, n)[, cols, drop = FALSE])`. ≥ 30 assertions across the cross product.
3. **End-to-end numerical equivalence gate (MANDATORY)**:
   - **3.a Baseline snapshot script** `dev/spd13-baseline.R` (analogous to dev/spd12-baseline.R). Captures `r$bts` via qs2 to `tests/testthat/fixtures/spd13/<config>.qs2` for:
     - ctrml × {mfh-all, mfh-bts, mfh-hfts, mfh-hfbts, mfh-str, mfh-str-bts, mfh-str-hfbts, mfh-str-hfts} × {randomForest, xgboost, lightgbm} (8 features × 3 ML = 24 cases, but condensed to 9: pick 3 representative features: mfh-all, mfh-str-bts, mfh-bts)
     - ctrml × mfh-all × lightgbm with NA-injected hat (one synthetic all-NA expanded col) — 1 case
     - ctrml × mfh-all × lightgbm with NA-injected hat + predict-reuse workflow (ctrml_fit then ctrml(fit=) with NA-bearing base) — 1 case
     - ctrml × mfh-all × lightgbm horizon-mismatch (expect_error) — 1 case
     - **Total: 9 + 3 = 12 fixtures**
   - Script must run on `spd.12-defer-input2rtw` HEAD 4efeef9 (pre-spd.13).
   - **3.b** `tests/testthat/test-spd13-equivalence.R`: each config → `max(abs(new - old)) == 0`. Horizon-mismatch → `expect_error` with regex match. FAIL on any nonzero diff.
4. **spd.12 regression suite re-run**: `test-spd12-equivalence.R`, `test-spd12-predict-reuse-na.R`, `test-spd12-closure-size.R`. All pass unchanged.
5. **Closure-size assertion (mfh, corrected)**: `tests/testthat/test-spd13-closure-size.R`. Build fixture (n=10, h=4, kset=c(1,2,4)). Capture wrapper output `hat` BEFORE deferred expansion (= raw, post-spd.13) and what wrapper used to output (= expanded, pre-spd.13: manually call `mat2hmat(hat, h, kset, n)`). Assertion: `object.size(hat_raw) == object.size(hat_expanded)` (sanity: same byte count). Document this in the test as "byte counts identical; main savings is transient peak avoidance + per-iter X shrink for sparse mfh." Add a SECOND assertion: build sparse mfh fixture (mfh-hfbts, n=10, h=4, kt=7); per-iter X size = `h × |global_id_i|` strictly LESS than `h × total_cols` for sparse cases. skip-on-CRAN.
6. **mw3.3 invariant intact**.
7. spd.10 / spd.9 unaffected.
8. terml / csrml: zero behavior change. Existing tests pass.
9. Single ticket. Fix-up commits permitted per spd.12 precedent. Conventional commits. No AI attribution.
10. Commit body MUST explicitly state: "Daemon mirai closure size unchanged for mfh (raw vs expanded have identical byte counts). Real benefits: main-process peak transient avoided; per-iter X memory drops for sparse mfh feature modes."

## Iteration history

- v1 (pre-spd.12 land): stale
- v2: refreshed against post-spd.12; claimed factor-h closure shrinkage (WRONG); B2/B3/B4/B5 ambiguities
- v3 changes:
  - **Section I-bis**: explicit honest benefit analysis (no factor-h closure savings; real wins = main transient + per-iter X for sparse mfh)
  - **B2 fix**: ctrml mfh wrapper passes non-NULL `keep_cols = sel_mat_keep_cols(sel_mat, total_cols)`; routes through rml() T5 branch which already handles raw hat via max(keep_cols)
  - **B3 fix**: explicit deletion of wrapper-side mfh NA block at all sites
  - **B4 fix**: explicit `total_cols <- tmp$dim[["n"]] * tmp$dim[["kt"]]` formula (no longer derivable from NCOL of expanded hat)
  - **B5 fix**: loop_body NA block unified across non-mfh and mfh kset branches; `na_mask <- na_cols` set symmetrically; na_cols_list populates correctly for mfh predict-reuse
  - **DoD §5 corrected**: assertion no longer claims factor-h closure shrinkage; tests per-iter X savings for sparse mfh
  - **Commit body §10**: honest framing of benefits

## Risks

- **R1 (mat2hmat_partial correctness)**: byte-equivalence verified algorithmically; DoD §2 enforces exhaustive equivalence.
- **R2 (limited real-world benefit)**: spd.13 does NOT shrink daemon closure. mfh users at high n_workers may still OOM. Documented honestly.
- **R3 (per-iter sort cost)**: `order()` on length n*h*kt per call × p iters. For typical (n=10, h=4, kt=21, p=2432): ~840 elements per sort × 2432 = 2M ops. Trivial.
- **R4 (mfh NA semantics)**: per-series NA detection on raw-then-partial-expanded hat produces identical effective per-series feature sets as wrapper-side global NA-zeroing (proof identical to spd.12). DoD §3 NA-injection fixtures validate.
