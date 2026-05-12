# Active Plan
<!-- followups approved: 2026-05-12 -->
<!-- gate-iterations: 2 -->
<!-- status: planned (T1-T6 landed; T7/T8 ready) -->

# T7 (w30.9): Extract na_col_mask helper for DRY consolidation
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** Extract the 6 NA-detection blocks across R/csrml.R / R/terml.R / R/ctrml.R into a single helper `na_col_mask(hat, threshold = 0.75)` in R/utils.R. Replace the `vapply(...)` RHS only; KEEP each site's existing LHS variable name (`na_var` in csrml; `na_local` in ctrml + terml) so downstream references stay intact.
* **Why:** Post-T2, the `vapply` body is byte-identical across 6 sites. T5 augmented 4 of those sites (ctrml + terml) with a `keep_cols` mapping path that uses a different LHS name (`na_local`). The vapply RHS is the duplicated portion; the surrounding downstream block diverges by file. CLAUDE.md §3 Pragmatic DRY: 3+ proven-identical occurrences → abstract. Centralizes the n=0 sum/0.75*NROW form.
* **Mechanism:** Single helper in utils.R. Replace ONLY the `vapply(...)` RHS at each site. Preserve LHS name + all downstream references.
* **Forbidden:** Standardizing LHS to one name globally (would touch 36 downstream references across 6 sites, expanding scope unnecessarily); changing threshold defaults; modifying any `if (any(...))` block or downstream `keep_cols`/`sel_mat` indexing logic.
* **Reference Data (verified via grep at HEAD 727ac91):**
  - csrml.R sites use `na_var`:
    - csrml.R:223 (`na_var <- vapply(...)`) — downstream refs at lines 228, 233, 236
    - csrml.R:373 — downstream refs at 378, 383, 386
  - ctrml.R sites use `na_local`:
    - ctrml.R:388 — downstream refs at 393, 400, 403, 407, 418, 419
    - ctrml.R:692 — downstream refs at 697, 703, 706, 709, 719, 720
  - terml.R sites use `na_local`:
    - terml.R:263 — downstream refs at 268, 274, 277, 280, 290, 291
    - terml.R:509 — downstream refs at 514, 520, 523, 526, 536, 537

  All 6 vapply bodies are byte-identical:
  ```r
  vapply(
    seq_len(NCOL(hat)),
    function(j) sum(is.na(hat[, j])) >= 0.75 * NROW(hat),
    logical(1)
  )
  ```

## II. Input Specification
* **Expected Input:** R/utils.R + R/csrml.R + R/terml.R + R/ctrml.R.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **Logic** | Output of `na_col_mask(hat)` MUST equal current inline `vapply(...)` result for every input. Verify via byte-identical reco_mat (set.seed=42) across all approaches × all feature modes. |
| **Format** | Match existing snake_case, 2-space indent. |
| **Boundary** | ONLY R/utils.R + R/csrml.R + R/terml.R + R/ctrml.R. No changes to reco_ml.R, NAMESPACE, tests, .Rd files. |
| **LHS preservation** | At csrml.R sites: keep `na_var <- ...`. At ctrml/terml sites: keep `na_local <- ...`. NO standardization. |
| **API** | Helper internal (not exported). NAMESPACE untouched. |

## IV. Step-by-Step Logic
1. Add to R/utils.R (after the existing helpers, end of file or near other internals):
   ```r
   # Identify feature columns of `hat` with NA rate >= threshold.
   # Returns logical vector of length NCOL(hat).
   # Uses sum/threshold*NROW form to preserve TRUE-everywhere semantics at NROW=0.
   na_col_mask <- function(hat, threshold = 0.75) {
     vapply(
       seq_len(NCOL(hat)),
       function(j) sum(is.na(hat[, j])) >= threshold * NROW(hat),
       logical(1)
     )
   }
   ```
2. Replace at csrml.R:223 — block:
   ```r
   na_var <- vapply(
     seq_len(NCOL(hat)),
     function(j) sum(is.na(hat[, j])) >= 0.75 * NROW(hat),
     logical(1)
   )
   ```
   with:
   ```r
   na_var <- na_col_mask(hat)
   ```
3. Same at csrml.R:373 (LHS = `na_var`).
4. At ctrml.R:388, replace the equivalent block with:
   ```r
   na_local <- na_col_mask(hat)
   ```
   (note LHS = `na_local`, NOT `na_var`).
5. Same at ctrml.R:692 (LHS = `na_local`).
6. At terml.R:263 — `na_local <- na_col_mask(hat)`.
7. At terml.R:509 — `na_local <- na_col_mask(hat)`.
8. Verify via grep:
   - `grep -n "vapply" R/csrml.R R/terml.R R/ctrml.R` → 0 hits (all 6 extracted)
   - `grep -c "na_col_mask" R/utils.R` → 1 (the definition; helper body uses vapply but `na_col_mask` token appears once at function head)
   - `grep -c "na_col_mask(hat)" R/csrml.R R/terml.R R/ctrml.R` → 6 (one per site)
   - `grep -n "na_var\\|na_local" R/csrml.R R/terml.R R/ctrml.R | wc -l` → 36 lines (unchanged — downstream refs intact)
9. Run `Rscript -e 'devtools::test()'` → 84/84 pass.
10. Reproducibility check (mandatory): build fixture from man/csrml.Rd; for approach ∈ {randomForest, xgboost, lightgbm}, features ∈ {all, str-bts} → capture reco_mat; max_abs_diff vs pre-patch HEAD must equal 0.0.
11. Commit: "T7 (w30.9): extract na_col_mask helper; replace 6 inline vapply sites (DRY)"

## V. Output Schema (Strict)
```toon
task_id: T7
success: bool
files_changed:
  - R/utils.R       # +1 helper (~7 lines)
  - R/csrml.R       # 2 sites: -10 lines, +2 lines
  - R/ctrml.R       # 2 sites: -10 lines, +2 lines  (LHS=na_local)
  - R/terml.R       # 2 sites: -10 lines, +2 lines  (LHS=na_local)
diff: |
  <unified diff>
test_result: { passed: int, failed: int }
grep_check:
  vapply_in_wrappers: 0
  na_col_mask_calls: 6
  na_var_refs_csrml: 8        # unchanged
  na_local_refs_ctrml: 14     # unchanged
  na_local_refs_terml: 14     # unchanged
numerical_check:
  approaches: [randomForest, xgboost, lightgbm]
  features: [all, str-bts]
  max_abs_diff: 0.0
error_log: null | msg
```

## VI. Definition of Done
- [ ] `na_col_mask(hat, threshold = 0.75)` defined in R/utils.R
- [ ] All 6 `vapply(...)` RHS sites replaced with `na_col_mask(hat)` call
- [ ] LHS preserved per-site: `na_var` in csrml (2 sites); `na_local` in ctrml + terml (4 sites)
- [ ] All 36 downstream references to na_var/na_local unchanged
- [ ] 84/84 tests pass
- [ ] Numerical equivalence: max_abs_diff = 0 on fixture
# T8 (w30.10): NEWS.md — document checkpoint arg + tuning default flip + n=0 edge
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** Add NEWS.md entries under the existing `# FoRecoML (development version)` heading documenting three user-visible changes from the memory-reduction branch:
  1. New `checkpoint = "auto"` argument on csrml/terml/ctrml + `_fit` variants (T6).
  2. Default flip of `tuning$store_benchmark_result` to FALSE in approach="mlr3" (T3).
  3. `NROW(hat) == 0` edge: NA-detection now returns TRUE for all columns instead of NaN-propagating (T2 — sum/0.75*NROW form). Vacuous case but explicit for completeness.
* **Why:** Final review of the 6-ticket branch flagged NEWS.md not updated. Plan review gate flagged the T2 n=0 case as a silent behavioral change deserving either NEWS coverage or written justification of omission; including it is cheaper than justifying.
* **Mechanism:** Augment existing `# FoRecoML (development version)` section (NEWS.md line 1) with two subsections.
* **Forbidden:** Prepending a duplicate `# FoRecoML (development version)` heading; rewriting `# FoRecoML 1.0.0` section; bumping DESCRIPTION Version field; editing .Rd or code files.
* **Reference Data:** Current NEWS.md (6 lines):
  ```markdown
  # FoRecoML (development version)

  # FoRecoML 1.0.0

  * Cross-sectional, temporal and cross-temporal forecast reconciliation with machine learning
  * Initial CRAN submission.
  ```

## II. Input Specification
* **Expected Input:** NEWS.md only.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **Logic** | Wording technically accurate per code at HEAD 727ac91 (e.g., 0.8 × available RAM, not 0.75; auto trigger, not always-on). |
| **Format** | Match existing convention: top-level `#` per version. Subsection style: `##` headers (consistent with conventional R-package NEWS style). |
| **Boundary** | ONLY NEWS.md. No DESCRIPTION, .Rd, code, tests. |
| **API** | Documentation only. |
| **Tone** | User-facing; no internal ticket references ("T3", "w30.x"). |
| **Augment, do not prepend** | Existing `# FoRecoML (development version)` line at NEWS.md:1 MUST be preserved as the only dev-version heading; insert new content below it. |

## IV. Step-by-Step Logic
1. Read NEWS.md. Confirm line 1 = `# FoRecoML (development version)` and line 2 is blank.
2. Insert these sections directly below line 1 (and above the blank line + `# FoRecoML 1.0.0`):
   ```markdown

   ## New features

   * `csrml()`, `terml()`, `ctrml()` and their `*_fit()` variants gain a
     `checkpoint` argument (default `"auto"`) for streaming per-series fits
     to disk. Set `checkpoint = TRUE` to force checkpointing to a
     session-scoped temporary directory, `checkpoint = FALSE` to keep all
     fits in memory (legacy behaviour), or pass a directory path for
     persistent, reusable storage. The default `"auto"` enables
     checkpointing when the estimated peak memory exceeds 80% of
     available RAM. Approach-specific serializers are used:
     `qs2::qs_save()` for randomForest and mlr3, `xgboost::xgb.save.raw()`
     wrapped in `qs2` for xgboost, and `lightgbm::lgb.save()` for lightgbm.

   ## User-visible changes

   * The default value of `tuning$store_benchmark_result` for
     `approach = "mlr3"` is now `FALSE` (was `TRUE`). This is a
     memory-frugal default; users who relied on benchmark archives should
     pass `tuning = list(store_benchmark_result = TRUE)` to restore the
     previous behaviour.

   * Internal NA-column detection now treats an empty input
     (`NROW(hat) == 0`) as "all columns NA". Previously this case produced
     `NaN` and propagated downstream; the new behaviour matches the
     intent of the threshold check at the zero-row edge.
   ```
3. Preserve all existing content below (blank line + `# FoRecoML 1.0.0` block) UNCHANGED.
4. Verify NEWS.md final shape:
   - Line 1: `# FoRecoML (development version)` (unchanged)
   - Lines 2–N: new content
   - Lines N+1 onward: pre-existing `# FoRecoML 1.0.0` block intact
   - No duplicate `# FoRecoML (development version)` headings
5. Commit: "T8 (w30.10): NEWS.md - checkpoint arg, store_benchmark_result default, n=0 NA edge"

## V. Output Schema (Strict)
```toon
task_id: T8
success: bool
files_changed: [NEWS.md]
diff: |
  <unified diff>
sections_added:
  - "## New features"
  - "## User-visible changes"
bullets_added: 3
prior_dev_heading_preserved: bool      # only one # FoRecoML (development version)
prior_v1_section_intact: bool          # # FoRecoML 1.0.0 block byte-identical
error_log: null | msg
```

## VI. Definition of Done
- [ ] NEWS.md augmented with 3 bullets across 2 subsections under existing dev-version heading
- [ ] Existing `# FoRecoML (development version)` line preserved as the only dev-version heading
- [ ] `# FoRecoML 1.0.0` block unchanged
- [ ] Wording technically accurate
- [ ] No code, .Rd, or DESCRIPTION changes
