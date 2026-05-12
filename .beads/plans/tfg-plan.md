# Active Plan: tfg sparse matrix construction
<!-- approved: 2026-05-12 -->
<!-- gate-iterations: 3 (PASS at iter 3) -->
<!-- status: planned -->

# Epic: Tier-B tfg sparse matrix construction

## Goal
Eliminate dense intermediate allocations in sel_mat construction across csrml/terml/ctrml. Two related fixes: replace `Matrix(rep(...), sparse=TRUE)` pattern with direct sparseMatrix(i,j,x) triples (tfg.1), and collapse ctrml compact's 3-step sparse construction into a single sparseMatrix() call (tfg.2).

## Success Criteria
- [ ] tfg.1: 6 `Matrix(rep(idx, n), ncol=n, sparse=TRUE)` sites in csrml/terml use sparseMatrix(i,j,x,dims) directly; no transient dense rep() allocation.
- [ ] tfg.2: ctrml compact branch builds sel_mat via single sparseMatrix() call (instead of bandSparse → t() → indexed-assignment).
- [ ] sel_mat byte-identical for all touched features × all wrappers.
- [ ] 85/85 tests pass; numerical reco_mat byte-identical.

## Context & Background
Original analysis findings #10 + #12 (B-tier). Both involve sel_mat construction in the wrapper feature switches.

#10 (tfg.1): `Matrix(rep(id_bts, nb), ncol=nb, sparse=TRUE)` allocates a length-(n*nb) dense vector via rep() before sparse coercion. For nb=100, n=200: 20000 doubles allocated transiently. For ctrml-like dims with id_bts of length n*p and ncol=nb*m: much larger.

#12 (tfg.2): ctrml compact branch:
```r
sel_mat <- Matrix::bandSparse(nb, n*p, pos)   # dgCMatrix copy 1
sel_mat <- 1 * t(sel_mat)                      # dgCMatrix copy 2 (transpose + numeric coerce)
sel_mat[1:n, ] <- 1                            # dgCMatrix copy 3 (indexed assignment)
```
Three successive sparse matrix copies via R reassignment semantics. For large hierarchies (nb=100, n*p=2400), each copy holds the sparse structure metadata + final value vector.

## Plan ordering
Independent. tfg.1 + tfg.2 can land in any order. Recommend tfg.2 first (single file, ctrml.R; smaller blast radius) → tfg.1 (3 files).

## Sub-Agent Strategy
Subagent-driven development. Each ticket: implementer (sonnet) → spec reviewer → quality reviewer.
# tfg.1: sel_mat via sparseMatrix() triples (no dense rep) — 10 sites
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** Replace `Matrix(rep(idx, n_cols), ncol = n_cols, sparse = TRUE)` pattern with `Matrix::sparseMatrix(i, j, x, dims=...)` via a new helper `sparse_col_replicate(idx, n_cols)` in R/utils.R. Total: **10 sites** across csrml/terml/ctrml.
* **Why:** `rep(idx, n_cols)` allocates `length(idx) * n_cols` doubles in one dense vector before R's sparse compression. For ctrml with `length(idx) ~ n*p` and `n_cols = nb*m`, this is the dominant transient.
* **Mechanism:** Single internal helper in R/utils.R (alongside existing cross-file helpers: `sel_mat_keep_cols`, `na_col_mask`, `input2rtw`, `input2rtw_partial`). 10 call sites updated to use the helper.
* **Forbidden:**
  - `Matrix(1, nrow=n, ncol=nb, sparse=TRUE)` "all" sites (csrml.R:214, csrml.R:360) — scalar broadcast, different pattern.
  - `as(rep(...), "sparseVector")` sites in ctrml.R (lines 318, 321, 620, 623) — produces sparseVector, not matrix; different optimization.
  - ctrml compact branch (tfg.2's domain).

## Reference Data (verified HEAD adc43a8)

### 10 target sites:
- **csrml.R** (4 sites, single-line):
  - 203 — `"bts"` feature (csrml)
  - 210 — `"str-bts"` feature (csrml)
  - 349 — `"bts"` (csrml_fit)
  - 356 — `"str-bts"` (csrml_fit)
  All pattern: `Matrix(rep(id_bts, nb), ncol = nb, sparse = TRUE)`.
- **terml.R** (2 sites, single-line, feature `"mfh-str-hfts"` NOT `"mfh-str"`):
  - 232 — terml `"mfh-str-hfts"`
  - 475 — terml_fit `"mfh-str-hfts"`
  Pattern: `Matrix(rep(id_hfts, m), ncol = m, sparse = TRUE)`.
- **ctrml.R** (4 sites, multi-line):
  - 329-333 — `"mfh-str-hfbts"` (ctrml): `Matrix(\n rep(id_hfbts, nb*m), \n ncol=nb*m, sparse=TRUE\n)`
  - 339-346 — `"mfh-str-bts"` (ctrml): `Matrix(\n rep(rep(id_bts, each=kt), nb*m), \n ncol=nb*m, sparse=TRUE\n)` (nested rep)
  - 631-635 — `"mfh-str-hfbts"` (ctrml_fit)
  - 641-648 — `"mfh-str-bts"` (ctrml_fit)

### R/utils.R existing helpers (helper home — verified):
- `sel_mat_keep_cols`, `na_col_mask`, `input2rtw`, `input2rtw_partial`, `new_rml_fit`, `serialize_fit`, `deserialize_fit`, `get_fit_i` — all non-exported, called from csrml/terml/ctrml via namespace visibility. `sparse_col_replicate` joins this family.

## II. Input Specification
R/utils.R (helper definition) + R/csrml.R + R/terml.R + R/ctrml.R (call sites).

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **Logic** | New sel_mat byte-identical to legacy at each site via `identical(as.matrix(old), as.matrix(new))`. |
| **Format** | Match snake_case, 2-space indent. |
| **Boundary** | R/utils.R + R/csrml.R + R/terml.R + R/ctrml.R only. NO reco_ml.R, tests, NAMESPACE, DESCRIPTION. |
| **API** | Helper internal (no `@export`). NAMESPACE unchanged. |
| **Strict scope** | Only the 10 listed sites. `Matrix(1, ...)`, `as(..., "sparseVector")`, ctrml compact UNCHANGED. |

## IV. Step-by-Step Logic
1. Add to R/utils.R (next to existing helpers like `sel_mat_keep_cols`):
   ```r
   # Build a column-replicated sparse 0/1 indicator matrix:
   # sparse `length(idx) × n_cols` matrix where each column equals `idx`.
   # Avoids the dense `rep(idx, n_cols)` allocation done by `Matrix(rep(...), sparse=TRUE)`.
   sparse_col_replicate <- function(idx, n_cols) {
     nz <- which(idx != 0)
     Matrix::sparseMatrix(
       i = rep(nz, times = n_cols),
       j = rep(seq_len(n_cols), each = length(nz)),
       x = 1,
       dims = c(length(idx), n_cols)
     )
   }
   ```
2. Replace 10 sites:
   - csrml.R:203 / :210 / :349 / :356 — `sparse_col_replicate(id_bts, nb)` replaces `Matrix(rep(id_bts, nb), ncol=nb, sparse=TRUE)`.
   - terml.R:232 / :475 — `sparse_col_replicate(id_hfts, m)` replaces `Matrix(rep(id_hfts, m), ncol=m, sparse=TRUE)`.
   - ctrml.R:329-333 / :631-635 — `sparse_col_replicate(id_hfbts, tmp$dim[["nb"]] * tmp$dim[["m"]])` replaces the multi-line `Matrix(...)`.
   - ctrml.R:339-346 / :641-648 — precompute `idx_local <- rep(id_bts, each = tmp$dim[["kt"]])` then `sparse_col_replicate(idx_local, tmp$dim[["nb"]] * tmp$dim[["m"]])`.
3. Per-site identity check (inline during development, not committed):
   ```r
   stopifnot(identical(
     as.matrix(Matrix(rep(idx, n_cols), ncol = n_cols, sparse = TRUE)),
     as.matrix(sparse_col_replicate(idx, n_cols))
   ))
   ```
4. Run `Rscript -e 'devtools::test()'` → must show 85/85.
5. Numerical equivalence (set.seed=42, baseline HEAD adc43a8 via git worktree):
   - csrml × {bts, str-bts} × {randomForest, xgboost, lightgbm}: max_abs_diff = 0.0
   - terml × {mfh-str-hfts}: max_abs_diff = 0.0
   - ctrml × {mfh-str-hfbts, mfh-str-bts}: max_abs_diff = 0.0
6. Commit: `git add R/utils.R R/csrml.R R/terml.R R/ctrml.R && git commit -m "tfg.1: sel_mat via sparseMatrix triples helper at 10 sites (drop dense rep)"`

## V. Output Schema
```toon
task_id: tfg.1
success: bool
files_changed: [R/utils.R, R/csrml.R, R/terml.R, R/ctrml.R]
sites_migrated: 10
helper_location: R/utils.R
test_result: { passed: int, failed: int }
identity_check: bool        # all 10 sel_mats byte-identical
numerical_check:
  csrml: { features: [bts, str-bts], approaches: [randomForest, xgboost, lightgbm], max_abs_diff: 0.0 }
  terml: { features: [mfh-str-hfts], approaches: [...], max_abs_diff: 0.0 }
  ctrml: { features: [mfh-str-hfbts, mfh-str-bts], approaches: [...], max_abs_diff: 0.0 }
grep_check:
  matrix_rep_remaining: 0    # `grep -nE "Matrix\\(rep|Matrix\\(\$" R/csrml.R R/terml.R R/ctrml.R` → 0 hits
```

## VI. Definition of Done
- [ ] `sparse_col_replicate()` defined in R/utils.R
- [ ] 10 sites migrated
- [ ] sel_mat byte-identical at each site (verified inline)
- [ ] 85/85 tests pass
- [ ] Numerical reco_mat byte-identical across all 6 wrapper×feature combinations
- [ ] No `Matrix(rep(...)` or multi-line `Matrix(\n rep(...)` pattern remains in csrml/terml/ctrml
# tfg.2: ctrml compact branch — collapse 3 sparse copies into one sparseMatrix call
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** Replace ctrml `features="compact"` sel_mat construction (bandSparse → 1*t() → indexed assignment) with a single `Matrix::sparseMatrix(i, j, x, dims=...)` call. Eliminates 2 intermediate sparse copies.
* **Why:** Each R reassignment to a sparse Matrix creates a new dgCMatrix object. For ctrml compact with large hierarchies (nb=100, n*p=2400), each transient copy holds ~p*nb non-zeros + structure overhead.
* **Mechanism:** Derive final (i, j) triples directly from the geometric structure.
* **Forbidden:** Touching csrml/terml/utils/reco_ml; touching ctrml non-compact branches; changing the `block_sampling <- tmp$dim[["m"]]` assignment.
* **Reference Data (verified HEAD adc43a8):**
  - R/ctrml.R:358-369 (ctrml() compact branch):
    ```r
    pos <- seq(
      tmp$dim[["na"]],
      by = tmp$dim[["n"]],
      length.out = tmp$dim[["p"]]
    )
    sel_mat <- Matrix::bandSparse(
      tmp$dim[["nb"]],
      tmp$dim[["n"]] * tmp$dim[["p"]],
      pos
    )
    sel_mat <- 1 * t(sel_mat)
    sel_mat[1:tmp$dim[["n"]], ] <- 1
    block_sampling <- tmp$dim[["m"]]
    ```
  - R/ctrml.R:660-671 (ctrml_fit() compact branch): identical structure, dup site.

## II. Input Specification
R/ctrml.R only.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **Logic** | Final sel_mat byte-identical via `identical(as.matrix(old), as.matrix(new))`. Class `dgCMatrix` preserved (or `dgTMatrix` then `as(. , "CsparseMatrix")` if needed). |
| **Format** | Match style. |
| **Boundary** | R/ctrml.R ONLY. Both compact sites (line 358-369 + 660-671). |
| **API** | None. |
| **block_sampling** | Assignment `block_sampling <- tmp$dim[["m"]]` UNCHANGED, immediately follows new sparseMatrix. |

## IV. Step-by-Step Logic
1. Derive triples analytically:
   - `n   <- tmp$dim[["n"]]`, `nb  <- tmp$dim[["nb"]]`, `na  <- tmp$dim[["na"]]`, `p   <- tmp$dim[["p"]]`.
   - Final dims: `n*p × nb` (after t()).
   - Top n rows × all nb cols: all 1 (from `sel_mat[1:n, ] <- 1`).
     ```r
     i_top <- rep(seq_len(n), times = nb)
     j_top <- rep(seq_len(nb), each = n)
     ```
   - Additional band entries for r in 2..p (the diagonal band rows from bandSparse, after transpose, that land in rows > n):
     For r in 2..p, k in 1..nb: row index = na + (r-1)*n + k, col = k.
     ```r
     if (p > 1) {
       row_offsets <- seq(from = na + n, by = n, length.out = p - 1)   # n+na, 2n+na, ..., (p-1)n+na
       i_band <- as.vector(outer(seq_len(nb), row_offsets, `+`))         # length nb*(p-1)
       j_band <- rep(seq_len(nb), times = p - 1)
     } else {
       i_band <- integer(0); j_band <- integer(0)
     }
     ```
   - Combine + build:
     ```r
     sel_mat <- Matrix::sparseMatrix(
       i = c(i_top, i_band),
       j = c(j_top, j_band),
       x = 1,
       dims = c(n * p, nb)
     )
     ```
2. **MANDATORY** inline byte-identity check during development (NOT committed):
   Build sel_mat both ways for a representative fixture (e.g., nb=8, n=11, na=3, p=4). Compare via `identical(as.matrix(old), as.matrix(new))` and `identical(Matrix::nnzero(old), Matrix::nnzero(new))`. MUST be TRUE before committing.
3. Replace BOTH compact sites in R/ctrml.R with the new construction. Keep `pos <- ...` block removed (no longer needed) but verify nothing else in the file references `pos`.
4. Run `Rscript -e 'devtools::test()'` → 85/85 pass.
5. Numerical equivalence: ctrml(features="compact") × randomForest/xgboost/lightgbm. set.seed(42); max_abs_diff = 0.0 vs baseline HEAD adc43a8.
6. Commit: "tfg.2: ctrml compact sel_mat via single sparseMatrix call (drop 3-copy chain)"

## V. Output Schema
```toon
task_id: tfg.2
success: bool
files_changed: [R/ctrml.R]
sites_migrated: 2
test_result: { passed: int, failed: int }
identity_check: bool       # final sel_mat byte-identical to legacy
numerical_check:
  approaches: [randomForest, xgboost, lightgbm]
  features: [compact]
  max_abs_diff: 0.0
pos_var_removed: bool      # `pos <- seq(...)` no longer needed
```

## VI. Definition of Done
- [ ] Both compact branches (ctrml() + ctrml_fit()) migrated
- [ ] sel_mat byte-identical (verified inline)
- [ ] `Matrix::bandSparse` and `1 * t(...)` and `sel_mat[1:n, ] <- 1` removed from compact branches
- [ ] 85/85 tests pass
- [ ] Numerical reco_mat byte-identical for ctrml compact
