# Memory Bottleneck Epic Plan

## Scope
10 implementation tickets (B1-B10) organized into 3 epics. B19 (Arrow shared hat) stays research-only at P4 — not in this plan. B11/B12-B17/B18/B20 skipped per prior review.

---

## Epic H — Daemon closure shrink (P1)

### Goal
Eliminate redundant data in mirai `.args` closure shipped to each daemon. Three independent surgical changes that compound for ~50% closure-byte reduction on ctrml/terml kset paths.

### Tickets
| ID | Title | Soundness mechanism |
|---|---|---|
| FoRecoML-6u7 | B1: split loop_body csrml + kset variants | Drops raw hat/base from kset closure (~50% savings) |
| FoRecoML-wup | B6: direct index formula bypasses order() | Eliminates main-process 2.5× hat transient |
| (B9 ID) | B9: drop sel_mat from .args when global_id_list populated | Drops redundant sparse sel_mat (~60 MB at XL) |

### Success Criteria
- [ ] All 3 child tickets merged
- [ ] Daemon closure size (verified via `object.size(.args)`) reduced by ≥40% on representative ctrml fixture
- [ ] spd.12 + spd.13 + spd.14 equivalence suites pass byte-identical
- [ ] mw3.3 invariant intact
- [ ] No CPU regression > 5% on micro-benchmark

### Sequencing
B1, B6, B9 are independent (different code paths). Land in any order. Recommend B1 first (largest single win + sets architectural pattern for cleaner formals; B9 can chain off the new loop_body_kset signature). B6 standalone.

### Risks
- B1 may surface latent assumption in loop_body that raw hat is always present. Mitigated by spec-reviewer audit step.
- B6 byte-equivalence depends on correct algebraic derivation; exhaustive unit-test cross product mandated.

---

## Epic I — Parallel-path architecture (P1)

### Goal
Coordinate three changes to the parallel/checkpoint architecture: caller-owned pools, transparent in-memory→checkpoint promotion, and bounded peak via series chunking.

### Tickets
| ID | Title | Soundness mechanism |
|---|---|---|
| FoRecoML-4pm | B2: persistent mirai pool opt-in (pool= arg) | Amortizes daemon spawn across multi-call workflows |
| FoRecoML-x2e | B3: auto-promote in-memory fits to checkpoint | Preserves user parallel intent without OOM |
| FoRecoML-drq | B5: series chunking for mfh-all | Bounds per-iter X peak to `chunk_size × X_per_iter` |

### Success Criteria
- [ ] User-owned mirai pool works across rolling-window calls
- [ ] In-memory fit + parallel predict-reuse no longer auto-caps to n_workers=1 (auto-promote instead)
- [ ] mfh-all ctrml on large fixture runs without exhausting available RAM (verify with mocked low RAM)
- [ ] All baseline + spd.12/13/14 equivalence preserved
- [ ] mw3.3 invariant intact

### Sequencing
B5 first (lowest cross-coupling). Then B2 OR B3 in either order. B3 should land before B2 if B3's tempdir lifecycle interacts with pool lifecycle; verify in feasibility gate.

### Risks
- B2 inverts B.1 strict guard; existing tests for "pool detected → abort" need updating to test pool=NULL default still aborts and pool=TRUE accepts.
- B3 tempdir cleanup on exit: ensure on.exit chain doesn't conflict with B2's user-owned pool teardown.
- B5 chunking + B2 pool: chunk dispatch in mirai_map must respect existing-pool lifetime (no daemon restart between chunks if pool is provided).

---

## Epic J — ML + checkpoint + polish (P1-P2)

### Goal
Backend + storage refinements: smaller default model lib, RAM-adaptive checkpoint compression, lazy storage of derived data, in-place terml reshape.

### Tickets
| ID | Title | Priority | Soundness mechanism |
|---|---|---|---|
| FoRecoML-8md | B4: ranger default + ML-aware checkpoint threshold | P1 | 5-10× smaller models for RF; auto-checkpoint sooner for tree ensembles |
| (B7 ID) | B7: RAM-adaptive qs2 nthreads | P2 | Bounds compression buffer to available RAM |
| (B8 ID) | B8: dim()<- for terml mfh reshape | P2 | In-place where possible vs matrix() |
| (B10 ID) | B10: lazy NULL na_cols_list | P2 | NULL when no NAs anywhere (cleaner semantics) |

### Success Criteria
- [ ] Default approach = "ranger" (or randomForest-equivalent smaller backend)
- [ ] resolve_checkpoint factors `approach` into threshold
- [ ] qs2 nthreads bounded by RAM headroom (mock-tested)
- [ ] terml mfh reshape uses dim()<- (3 sites)
- [ ] fit$na_cols_list is NULL when no series had NAs
- [ ] All baseline + spd.12/13/14 equivalence preserved
- [ ] mw3.3 invariant intact

### Sequencing
All 4 tickets independent. Recommend B4 first (largest user-visible change, validates ML backend default). B7, B8, B10 in parallel after.

### Risks
- B4: ranger is statistically equivalent to randomForest but NOT byte-identical (different RNG, different default node handling). Test fixtures using default approach need explicit `approach = "randomForest"` to preserve equivalence; fixtures testing ranger as default need new baseline.
- B4: resolve_checkpoint signature change must be back-compat (default arg).
- B7: RAM-adaptive must round-trip identically — qs2 byte-output should be independent of nthreads (verify).

---

## Cross-epic dependencies & sequencing

```
Epic H (closure shrink) ────┐
                            │
Epic I (parallel arch) ─────┼──→ All P1 tickets land
                            │
Epic J (ML + polish) ───────┘
        ↓
    Epic G (research, B19) — deferred independently
```

No hard ordering between H, I, J. All can proceed in parallel via separate branches. Recommend interleaved merges to avoid stale equivalence fixtures (each landed perf ticket regenerates baseline if needed).

## Cross-cutting invariants (apply to ALL 10 tickets)

- [ ] mw3.3 invariant in reco_ml.R UNCHANGED
- [ ] spd.12 equivalence suite passes byte-identical (max_abs_diff == 0)
- [ ] spd.13 equivalence suite passes byte-identical
- [ ] spd.14 internals byte-identity preserved
- [ ] spd.12 predict-reuse NA tests pass
- [ ] spd.10 in-memory fit cap test passes (B3 changes its assertion semantics, but test must be updated to reflect auto-promote — not deleted)
- [ ] No AI attribution in any commit
- [ ] No --no-verify
- [ ] Single commit per ticket (fix-up commits allowed if reviewer feedback)

## Out of scope

- B11 (T3 already mitigated)
- B12-B17 (negligible, no tickets)
- B18 (G.1 research, separate epic)
- B19 (B19 research, P4, deferred)
- B20 (skipped per user)

## Risks (epic-level)

- **R1**: Multiple tickets touch loop_body / .args. Conflict-prone if merged out-of-order. Mitigate: sequence Epic H tickets carefully; each rebase + retest before merge.
- **R2**: B4 ranger swap changes test baseline. Mitigate: explicit `approach = "randomForest"` in equivalence-dependent fixtures.
- **R3**: B2 + B3 interaction: pool ownership × tempdir lifecycle. Mitigate: test combined workflow (user-owned pool + auto-promote checkpoint) explicitly.

## Ticket-level gaps to address before implementation (added in v2)

- **B4 gap**: `ranger` is currently in `Suggests` (DESCRIPTION:49). Switching default `approach` requires moving to `Imports` (or adding fallback if ranger unavailable). Ticket B4 must include DESCRIPTION update step.
- **B2 gap**: existing test in `test-parallel.R` ("rml errors on pre-existing daemon pool") asserts specific abort message `"Existing mirai daemon pool"`. B2 changes the message wording (adds pool=TRUE opt-in hint). Ticket B2 must include explicit update to that test's regex.
- **B5 annotation**: ticket should add explicit sentence asserting chunked output byte-identical to unchunked output (covered by cross-cutting equivalence, but in-ticket clarity improves implementer confidence).
