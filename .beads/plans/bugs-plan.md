# Active Plan: Bugs Epic FoRecoML-t9r

<!-- approved: 2026-05-17 -->
<!-- gate-iterations: 3 -->
<!-- status: ready-to-execute -->

## Tickets
- FoRecoML-t9r.1 (BUG-1, P0): Fix categorical_feature to string in 4 sites (2 DELETE, 2 replace)
- FoRecoML-t9r.2 (BUG-2, P1): Verify + regression test ctrml_g(level_id=TRUE) default + chunked

## Key design decisions
- BUG-1: colnames(X_train) already has "series_id" from cbind; just change categorical_feature arg
- BUG-2: chunked path confirmed to pass level_id/kset through (grep verified)
- BUG-3/FEAT-1: duplicate of existing FoRecoML-3st epic
- FEAT-2: existing FoRecoML-8r1 epic (updated with analyst input)
- FEAT-3: existing FoRecoML-9id backlog (updated with implementation hint)
