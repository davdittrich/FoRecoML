# Follow-up Epics from spd.* Critical Review

## Context

Adversarial code review of spd.1-spd.13 surfaced 1 BLOCKING + 2 REQUIRED + 1 known TODO (spd.14). All spd-introduced work landed on main (HEAD 5db37bf). Items below are surfaced regressions, latent bugs, and follow-up perf work, organized into 2 epics.

---

## Epic A: ctrml mfh-str* correctness fixes (CRASH-CLASS)

### Goal
Fix runtime crash when user calls `ctrml(features = "mfh-str" | "mfh-str-hfbts" | "mfh-str-bts")`.

### Success Criteria
- [ ] `ctrml()` + `ctrml_fit()` execute without "object 'sel_mat' not found" error for all 3 mfh-str* features
- [ ] Numerical equivalence vs terml mfh-str pattern (where sel_mat = `1 * (strc_mat != 0)`)
- [ ] testthat tests cover all 3 mfh-str* feature modes through ctrml end-to-end
- [ ] No regression on existing 215 tests

### Context & Background
**Root cause**: R/ctrml.R lines 337-352 (and mirror 626-636 in ctrml_fit) initialize `sel_mat` from `(sel_mat != 0)` — but `sel_mat` is never assigned earlier in scope. The terml.R analog at line 228 correctly uses `strc_mat != 0`. ctrml has `strc_mat <- tmp$strc_mat` at lines 279 and 571 (in scope at the switch site).

**Pre-existing in upstream danigiro/FoRecoML**: `git show origin/main:R/ctrml.R` shows identical bug. Not introduced by spd.* tickets. Surfaced only because spd.13 added end-to-end mfh test coverage that didn't exercise mfh-str* variants.

**Severity rationale**: Hard crash on any user invocation of these 3 features. No silent corruption, no data loss — but feature is non-functional. P0 because mfh-str* are documented features (`man/ctrml.Rd` lists them as valid `features` arg values).

### Sub-Agent Strategy
Single-task epic. Single atomic implementer commit + spec/quality review. Surgical 6-line replacement + tests.

### Tasks

#### A.1 — Replace unbound `sel_mat` with `strc_mat` in ctrml mfh-str* branches

**Status**: `READY_FOR_EXECUTION`

##### I. Context & Objective
- **Objective**: Fix runtime crash in `ctrml()` and `ctrml_fit()` when features ∈ {mfh-str, mfh-str-hfbts, mfh-str-bts}.
- **Why**: Pre-existing upstream bug surfaced by spd.13 test additions; `sel_mat` referenced before assignment.
- **Reference Data**: 
  - ctrml.R sites: lines 338, 341, 347 (training path); lines 627, 630, 636 (ctrml_fit path)
  - terml.R reference pattern: line 228 `sel_mat <- 1 * (strc_mat != 0)` and line 231 (mfh-str-hfts)
  - `strc_mat` defined at ctrml.R:279 and ctrml.R:571

##### II. Input Specification
- R/ctrml.R (6 sites)
- R/terml.R (reference pattern only — UNTOUCHED)

##### III. Constraints & Guards

| Type | Guard |
| :--- | :--- |
| Logic | At each site, RHS `sel_mat != 0` → `strc_mat != 0`. LHS `sel_mat <- ...` unchanged. |
| Format | 6 surgical line edits. No structural refactoring. |
| Boundary | Only ctrml.R touched. terml + csrml + reco_ml + utils UNTOUCHED. |
| Tone | Conventional commit. Body explains pre-existing upstream bug. |

##### IV. Step-by-Step Logic

1. `grep -n "sel_mat <- 1 \\* (sel_mat != 0)" R/ctrml.R` → identify the 6 RHS-bug sites (3 in ctrml at lines 338, 341, 347; 3 in ctrml_fit at lines 627, 630, 636). NOTE: `grep "sel_mat != 0"` also matches 4 LEGITIMATE LHS-index expressions (`sel_mat[sel_mat != 0] <- 1` at lines 344, 351, 633, 640) — those are NOT the bug. Use the more specific pattern above.
2. At each of the 6 sites, replace ONLY the RHS `(sel_mat != 0)` with `(strc_mat != 0)`. Preserve LHS `sel_mat <- 1 * ...` form. Do NOT touch the `sel_mat[sel_mat != 0] <- 1` lines — those reference the just-assigned `sel_mat` and are correct.
3. Verify `strc_mat` is in scope at each site by reading function structure.
4. Add tests at `tests/testthat/test-ctrml-mfh-str.R`:
   - ctrml × mfh-str × lightgbm: no crash, produces non-trivial output
   - ctrml × mfh-str-hfbts × lightgbm: no crash, output shape matches mfh-hfbts shape semantics
   - ctrml × mfh-str-bts × lightgbm: no crash
   - ctrml_fit + ctrml(fit=, base=) workflow with mfh-str: no crash, round-trip
5. Run `devtools::test()`. All 215 baseline + new tests pass.
6. Single commit: `fix(ctrml): replace unbound sel_mat with strc_mat in mfh-str branches`.

##### V. Output Schema

```
sites_fixed: [ctrml.R:338, ctrml.R:341, ctrml.R:347, ctrml.R:627, ctrml.R:630, ctrml.R:636]
tests_added: tests/testthat/test-ctrml-mfh-str.R
test_count_pre: 215
test_count_post: >= 219
commit_count: 1
```

##### VI. Definition of Done
- [ ] `grep "sel_mat <- 1 \\* (sel_mat != 0)" R/ctrml.R` returns 0 matches (the 6 bug sites)
- [ ] `grep "sel_mat <- 1 \\* (strc_mat != 0)" R/ctrml.R` returns 6 matches (the fixed sites)
- [ ] `grep "sel_mat\\[sel_mat != 0\\] <- 1" R/ctrml.R` returns 4 matches (legitimate LHS-index lines untouched)
- [ ] All baseline + new mfh-str tests pass
- [ ] mw3.3, spd.12, spd.13 invariants intact
- [ ] Single commit; conventional commits; no AI attribution

---

## Epic B: spd.* maintenance + perf followups

### Goal
Address review findings flagged as REQUIRED/SUGGESTION on spd.1/spd.12/spd.13: mirai daemon teardown leak, misleading test snapshot label, and spd.14 perf hoist (FoReco2matrix per-iter regression).

### Success Criteria
- [ ] mirai daemons always torn down or never claimed (no leak on pre-existing pool)
- [ ] Test snapshot file naming matches actual fixture features
- [ ] FoReco2matrix hoisted out of per-iter loop_body, reducing O(p) overhead to O(1) per rml() call
- [ ] No numerical regression on spd.12 + spd.13 equivalence suites

### Context & Background
**Item B.1 (mirai teardown)**: reco_ml.R:254-258 registers `on.exit(mirai::daemons(0))` only when `prev == 0L`. If caller has existing daemon pool, teardown is skipped. After rml() returns, caller's pool may be in a state mutated by `mirai::everywhere(library(FoRecoML))`. Edge case — but real.

**Item B.2 (snapshot label)**: `tests/testthat/fixtures/spd12/csrml_compact_lightgbm.qs2` was generated with `features = "bts"` (see dev/spd12-baseline.R). Self-consistent today but landmine for future regeneration.

**Item B.3 (spd.14 — FoReco2matrix hoist)**: Both `input2rtw_partial` and `mat2hmat_partial` call `FoReco::FoReco2matrix(hat, kset)` (or equivalent) PER iteration. For p=2432 series, that's 2432 redundant transforms of the same raw hat. Plan: compute parts ONCE in rml() outer scope, pass to loop_body as additional formal, slice inside.

### Sub-Agent Strategy
3 atomic tasks. Sequential — B.3 (spd.14) is the heaviest and should land last so B.1 + B.2 don't conflict.

### Tasks

#### B.1 — Fix mirai daemon teardown on pre-existing pool

**Status**: `READY_FOR_EXECUTION`

##### I. Context & Objective
- **Objective**: Always restore caller's mirai daemon state on rml() exit.
- **Why**: Current `on.exit(mirai::daemons(0))` is conditional on `prev == 0L`; user with pre-existing pool gets `library(FoRecoML)` injected into their pool but no teardown of injected state.
- **Reference Data**: reco_ml.R:254-260

##### II. Input Specification
- R/reco_ml.R (single function: `rml`)

##### III. Constraints & Guards

| Type | Guard |
| :--- | :--- |
| Logic | If `prev == 0L`, register teardown to `mirai::daemons(0)` (current). If `prev > 0L`, error explicitly OR document reuse contract + assert daemons are loaded with FoRecoML. |
| Format | Surgical change to rml() preamble. |
| Boundary | Only reco_ml.R touched. Tests in test-parallel.R get one new case. |
| Tone | Conventional commit. Body explains pool ownership semantics. |

##### IV. Step-by-Step Logic

1. Read reco_ml.R:240-280 (mirai setup block).
2. Decision: explicit error when `prev > 0L`. Less invasive than reuse-with-cleanup. Documented in commit body.
3. Replace conditional teardown with:
   ```r
   prev <- mirai::status()$connections
   if (prev > 0L) {
     cli::cli_abort(c(
       "Existing mirai daemon pool ({prev} connections) detected.",
       "i" = "rml() requires exclusive daemon ownership.",
       "x" = "Tear down with `mirai::daemons(0)` before calling rml()."
     ))
   }
   mirai::daemons(n_workers_resolved, seed = mirai_seed)
   on.exit(mirai::daemons(0), add = TRUE)
   mirai::everywhere({ library(FoRecoML) })
   ```
4. Add test: `test-parallel.R::"rml errors on pre-existing daemon pool"`. Setup: `mirai::daemons(1)`. Expect: `expect_error(rml(...), "Existing mirai daemon pool")`. Teardown: `mirai::daemons(0)`.
5. `devtools::test()` → all pass.
6. Single commit: `fix(rml): error on pre-existing mirai daemon pool to prevent leakage`.

##### V. Output Schema

```
site_modified: R/reco_ml.R lines 254-260
test_added: test-parallel.R::"rml errors on pre-existing daemon pool"
commit_count: 1
test_count_post: >= 216
```

##### VI. Definition of Done
- [ ] `prev > 0` case explicit `cli_abort` with actionable message
- [ ] Test exercises pre-existing pool and asserts error
- [ ] No leak after test (`mirai::status()$connections == 0` post-test)
- [ ] Existing 215 tests still pass

---

#### B.2 — Rename misleading csrml snapshot label

**Status**: `READY_FOR_EXECUTION`

##### I. Context & Objective
- **Objective**: Rename `csrml_compact_lightgbm.qs2` → `csrml_bts_lightgbm.qs2`. Update generator + test references.
- **Why**: Snapshot was generated with `features = "bts"`. Filename "compact" is wrong. Future regenerator will use wrong features.
- **Reference Data**: 
  - File: `tests/testthat/fixtures/spd12/csrml_compact_lightgbm.qs2`
  - Generator: `dev/spd12-baseline.R` (references same filename)
  - Test: `tests/testthat/test-spd12-equivalence.R:118-127`

##### II. Input Specification
- 1 qs2 file (rename)
- 2 R files (string update)

##### III. Constraints & Guards

| Type | Guard |
| :--- | :--- |
| Logic | Rename only. Snapshot content NOT regenerated (would invalidate equivalence). |
| Format | `git mv` for rename to preserve history. |
| Boundary | Only spd12 csrml fixture. spd.13 fixtures UNTOUCHED. |
| Tone | Conventional commit. `refactor(tests):` not `fix:`. |

##### IV. Step-by-Step Logic

1. `git mv tests/testthat/fixtures/spd12/csrml_compact_lightgbm.qs2 tests/testthat/fixtures/spd12/csrml_bts_lightgbm.qs2`
2. Update `dev/spd12-baseline.R`: replace `csrml_compact_lightgbm` with `csrml_bts_lightgbm` (verify it's the only csrml fixture name).
3. Update `tests/testthat/test-spd12-equivalence.R`: replace the same string.
4. Run `devtools::test()`. csrml equivalence test must still pass byte-identical (rename should not change snapshot bytes).
5. Single commit: `refactor(tests): rename csrml snapshot to match actual fixture features (bts not compact)`.

##### V. Output Schema

```
file_renamed: tests/testthat/fixtures/spd12/csrml_compact_lightgbm.qs2 -> csrml_bts_lightgbm.qs2
files_updated: [dev/spd12-baseline.R, tests/testthat/test-spd12-equivalence.R]
commit_count: 1
test_pass_count: 215 (unchanged)
```

##### VI. Definition of Done
- [ ] `find tests/testthat/fixtures/spd12 -name 'csrml_compact_*'` → 0 matches
- [ ] `find tests/testthat/fixtures/spd12 -name 'csrml_bts_*'` → 1 match
- [ ] grep for `csrml_compact_lightgbm` in repo → 0 matches
- [ ] Tests all green

---

#### B.3 — spd.14: Hoist FoReco2matrix out of per-iter loop_body

**Status**: `READY_FOR_EXECUTION`

##### I. Context & Objective
- **Objective**: Pre-compute `FoReco::FoReco2matrix(hat, kset)` parts (and analogous transform for `mat2hmat`) ONCE in rml() outer scope. Pass to loop_body. Per-iter call materializes only requested columns from pre-computed parts.
- **Why**: spd.12 commit body acknowledged "trades per-iter FoReco2matrix cost for closure-size reduction" — this is the perf followup to eliminate the trade. Both `input2rtw_partial` and `mat2hmat_partial` currently invoke FoReco2matrix per-iter on the SAME raw hat. For p iterations × n_workers daemons = redundant O(p) transform work.
- **Reference Data**: 
  - utils.R: `input2rtw_partial` (line 343), `mat2hmat_partial` (line 322)
  - reco_ml.R: loop_body kset branch (lines 137-165), Xtest branch (lines 195-212)
  - spd.12 commit body reference + spd.13 reco_ml.R:140-142 TODO comment

##### II. Input Specification
- R/reco_ml.R rml() outer scope + loop_body
- R/utils.R input2rtw_partial + mat2hmat_partial signatures may evolve to accept pre-computed parts

##### III. Constraints & Guards

| Type | Guard |
| :--- | :--- |
| Logic | Numerical output byte-identical to current state. spd.12 + spd.13 equivalence suites MUST stay green at max_abs_diff == 0. |
| Format | Two new variants: `input2rtw_partial_from_parts(parts, kset, cols)` and `mat2hmat_partial_from_sorted(sorted_vec, h, ncol_total, cols)`. Originals kept for back-compat. |
| Boundary | reco_ml.R + utils.R only. Wrappers (ctrml/terml/csrml) UNTOUCHED. |
| Audit | spd.12 + spd.13 equivalence tests stay green; new microbenchmark fixture demonstrates per-iter speedup. |

##### IV. Step-by-Step Logic

1. Read current loop_body kset branch. Identify the per-iter call: `input2rtw_partial(hat, kset, cols=global_id)` (line 141 area) and `mat2hmat_partial(hat, h, kset, n, cols=global_id)` (line 143 area).
2. In rml() pre-loop scope: if `!is.null(kset)` and `is.null(h)`, compute `parts <- FoReco::FoReco2matrix(hat, kset)` once. If `!is.null(h)`, compute `sorted_vec <- vec[order(i)]; ncol_total <- ...` once.
3. Pass `parts` (or `sorted_vec` + `ncol_total`) to loop_body as additional formal. Same for predict-time base — pre-compute analogous structures.
4. Add new internal utils:
   - `input2rtw_partial_from_parts(parts, kset, cols)` — sibling to input2rtw_partial; takes pre-computed parts list.
   - `mat2hmat_partial_from_sorted(sorted_vec, h, ncol_total, cols)` — sibling to mat2hmat_partial; takes pre-sorted vec + dim.
5. loop_body kset branch calls `*_from_parts` / `*_from_sorted` variants. Originals (input2rtw_partial, mat2hmat_partial) UNTOUCHED for back-compat.
6. Update mirai_map `.args` and sequential dispatch `.args` to include `parts` / `sorted_vec` / `ncol_total`.
7. Numerical equivalence: re-run spd.12 + spd.13 equivalence suites. All max_abs_diff == 0.
8. Micro-benchmark fixture: `dev/spd14-bench.R` — time rml() with p=100 series, kt=21 expansion, before vs after. Document in commit body.
9. Single commit: `perf(rml): hoist FoReco2matrix and mat2hmat sort out of loop_body (spd.14)`.

##### V. Output Schema

```
files_modified: [R/reco_ml.R, R/utils.R]
files_unchanged: [R/ctrml.R, R/terml.R, R/csrml.R, R/FoReco.R]
new_internals: [input2rtw_partial_from_parts, mat2hmat_partial_from_sorted]
equivalence_max_abs_diff: 0
test_count_pre: 215 (or 216 if B.1 landed)
test_count_post: >= 215
benchmark_speedup_factor: >= 2x at p=100 series (empirical)
```

##### VI. Definition of Done
- [ ] spd.12 + spd.13 equivalence suites pass with max_abs_diff == 0
- [ ] mw3.3 invariant intact
- [ ] mirai .args extended with pre-computed parts
- [ ] Micro-benchmark documents speedup
- [ ] Wrappers untouched
- [ ] Single commit; conventional commits; honest framing

---

## Atomicity & Sequencing

- A.1 is independent of B.* — can land first or in parallel.
- B.1 + B.2 are independent. Order arbitrary.
- B.3 (spd.14) should land LAST because it modifies loop_body extensively and reduces conflict surface for B.1.

## Dependencies (beads)

```
Epic A (FoRecoML-???)
└── A.1 (FoRecoML-???.1)

Epic B (FoRecoML-???)
├── B.1 (FoRecoML-???.1)
├── B.2 (FoRecoML-???.2)
└── B.3 (FoRecoML-???.3)   # depends on B.1 to land first (lower conflict)
```

## Risks

- **R1 (A.1 scope creep)**: tempting to also fix `mfh-str-hfbts` mfh-extension semantics (sparse arithmetic combination). Resist — that's a separate ticket.
- **R2 (B.3 perf claim)**: micro-benchmark on small p may not show the gain. Need realistic p (100+ series) and kt (≥7) to surface the difference. Plan documents benchmark fixture.
- **R3 (B.1 user breakage)**: if any caller currently invokes rml() with their own mirai pool, the new error breaks them. Documented in commit body with workaround.
