# Analyst Report Triage — 2026-05-17

**Report:** `forecoml_bugs_and_features (1).md`
**Reporter snapshot:** `c3a9b0c` (fc735afc on `main`)
**Triage commit:** `78f8e25`
**Triage verdict:** 4 of 6 items already resolved; 1 false positive; 1 actionable regression-test gap (filed + landed as epic `FoRecoML-bba`).

---

## TL;DR

The reporter ran the test suite against a **stale installed binary**. Their local source tree was at the post-fix commit (`fc735afc`), but the package installed in their R library was an older snapshot from before the BUG-1/BUG-2 fixes landed. `devtools::load_all()` reloads `.R` files but does NOT rebuild compiled artifacts or refresh the installed package's `NAMESPACE`/`series_id_levels` factor coding — so subagents downstream of the broken install kept hitting `Size of feature_names error` and false `ctbu` rejections.

**Single remediation step:** `R CMD INSTALL --preclean .` resolves every reported failure.

---

## Per-item triage

### BUG-1 — `categorical_feature` integer index broken on LightGBM ≥ 4.0

**Status:** ✅ FIXED `dd61d35` (analyst already acknowledged).
**Action:** None.

### BUG-2 — `ctrml_g(level_id=TRUE)` predict-time feature mismatch

**Status:** ✅ FIXED.
**Evidence:** Regression test `fd500f1` (`test-rml-g-level-id.R`) — both default and chunked paths pass end-to-end with `level_id=TRUE`.
**Root cause of analyst's continued claim:** stale install. The predict-path level_id reconstruction code (`predict.rml_g_fit` lines 1860+) is present in source but their installed binary predated it.

### BUG-3 — `hat` replication design incompatible with FoReco CT wide layout

**Status:** ✅ ADDRESSED by FEAT-1 (`epic FoRecoML-3st`, closed).
**Action:** Duplicate. The wide_ct path solves this. See quick reference below.

### BUG-4 — `ctbu_base` built as `nb × m` "instead of" `nb × 1` / FoReco rejects it

**Status:** ❌ **FALSE POSITIVE.**

**The analyst's claim contains two mutually contradictory diagnoses:**

> "`FoReco::ctbu` only needs `base` for the finest temporal level (k=1) … it wants the scalar for h=1."
>
> "`FoReco::ctbu` requires `base` as `nb × (h × kt)` where `kt = 28` → `base` must be `2432 × 28`."

Both are wrong. Live verification against `FoReco 1.2.1`:

```r
> base_nbm  <- matrix(rnorm(2 * 12), 2, 12, dimnames = list(c("A","B"), NULL))
> base_nbkt <- matrix(rnorm(2 * 28), 2, 28, dimnames = list(c("A","B"), NULL))
> FoReco::ctbu(base_nbm,  agg_mat, agg_order = c(12L,6L,4L,3L,2L,1L), tew = "sum")
# OK → 3 × 28 reconciled matrix
> FoReco::ctbu(base_nbkt, agg_mat, agg_order = c(12L,6L,4L,3L,2L,1L), tew = "sum")
# Error: Incorrect `base` columns dimension.
```

`FoReco::ctbu` wants `nb × m` (finest temporal level only — the `m` high-frequency periods per cycle). The current `ctrml_g` wide_ct path builds exactly that. Output is `n × kt` (all temporal levels reconstructed via bottom-up aggregation), which is what the analyst expected.

**Why the analyst's repro failed for all backends:** stale install. After `R CMD INSTALL --preclean .`:

| Backend  | Result on prod-scale `agg_order=c(12,6,4,3,2,1)` |
|----------|---------------------------------------------------|
| lightgbm | ✅ 3 × 28 reconciled matrix                       |
| xgboost  | ✅ 3 × 28 reconciled matrix                       |
| ranger   | ✅ 3 × 28 reconciled matrix                       |

**Action taken:** Filed regression-test ticket (`FoRecoML-bba.1`, closed `27528d6`). `test-rml-g-wide-ct.R` now covers all 3 backends + production-scale `agg_order=c(12,6,4,3,2,1)`. Future stale-install false positives will fail loudly in CI.

### FEAT-1 — `input_format="wide_ct"` for ctrml_g

**Status:** ✅ IMPLEMENTED (`epic FoRecoML-3st`, closed).

### FEAT-2 — `obs_mask` for structurally missing observations

**Status:** ✅ IMPLEMENTED (`epic FoRecoML-8r1`, closed). 4 tickets shipped: `.stack_series()` filter, backend/wrapper passthrough, `compute_rec_residuals` NA routing, 6 regression tests.

### FEAT-3 — `cs_level` cross-sectional depth feature

**Status:** ✅ IMPLEMENTED (`9id`, closed). Requires `input_format="wide_ct"` (tall mode stacks bottom-only, making `cs_level` information-free). Guard `cli_abort` in place for the tall-mode case.

---

## What went wrong on the analyst's side

1. **Stale install never invalidated.** Modifying `.R` files in the source tree does not refresh the installed binary. `library(FoRecoML)` loads the binary, not the source.
2. **`devtools::load_all()` is not equivalent to install.** It reloads source but skips namespace finalization, the `series_id_levels` factor sort that lives in compiled metadata, and any vendored S3 method tables. After a major change to dispatch logic, `load_all()` can disagree with the binary about which method runs.
3. **No CI guard for the binary/source skew.** `devtools::test()` runs against `load_all`'d code on the analyst's machine, but the production pipeline elsewhere may be running against the binary. The mismatch is invisible until a real call (like `ctrml_g(..., approach="lightgbm")`) hits a path that depends on freshly-installed metadata.
4. **Two contradictory hypotheses about `ctbu`'s API in a single bug report.** When the same paragraph proposes `nb × 1`, `nb × m`, and `nb × kt` as the "correct" shape, the bug is in the diagnosis, not the code. Resolution: run `FoReco::ctbu(test_input, ...)` directly with each candidate shape and observe.

## The fix (one command)

```bash
# In the FoRecoML source tree:
R CMD INSTALL --preclean .

# Then re-test:
Rscript -e 'devtools::test()'
# Expected: FAIL 0 | PASS 326
```

Add this to the bench/CI runner as the first step of any pre-experiment setup. Treat any "I can't reproduce on `main`" finding as a stale-install candidate before filing a bug.

---

## Triage references

- Triage commit (regression tests): `27528d6`
- Triage epic closure: `78f8e25`
- All backend coverage tests: `tests/testthat/test-rml-g-wide-ct.R`
- Production-scale (`agg_order=c(12,6,4,3,2,1)`) lightgbm test: same file, last `test_that()` block.
