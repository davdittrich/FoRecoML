# FoRecoML-rwj: Implement Arrow IPC Prefetch-per-daemon Shared Hat (B19)

**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective

- **Objective:** Replace `parts_hat` in mirai daemon `.args` closure with an Arrow IPC file path. Each daemon materializes only its needed column subset once at startup, then operates at native R speed. Reduces daemon hat memory from K×hat_bytes to ≈subset_fraction×K×hat_bytes for sparse feature modes.
- **Why:** OOM with n_workers=3 + targets/crew worker already holding all_folds. Developer diagnosis: hat duplicated across K mirai daemons via parts_hat serialization. B19 phase-1/2 research (docs/research/b19-arrow-shared-hat.md) confirmed: (1) kernel page-cache sharing via Arrow IPC works (E1/E2 PASS), (2) per-iter Arrow access is 22×+ slower than native but prefetch-once pattern is native-speed after first iteration (B2 phase-2).
- **Scope:** Non-mfh kset path ONLY (h=NULL). mfh uses sorted_vec_hat, different ticket.
- **Philosophy:** Sub-agent = Goldfish Memory. All logic is here. No reading prior session.

## II. Input Specification

- R/reco_ml.R: rml() pre-loop scope + loop_body_kset function (post-B1 split, 24 formals + b19_active to add)
- R/utils.R: add b19_daemon_load() internal helper next to existing serialize_fit
- R/ctrml.R, R/terml.R, R/csrml.R + their _fit variants: add shared_hat formal + pass-through
- DESCRIPTION: add arrow to Suggests
- tests/testthat/: 6 new test scenarios (see §IV step 9)

## III. Constraints & Guards

| Type | Guard |
|------|-------|
| **Scope** | Non-mfh ONLY: `!is.null(kset) && is.null(h)`. mfh (h non-NULL) → skip B19. |
| **Activation** | B19 activates iff ALL 7 conditions: non-mfh, parallel (n_workers>1), arrow available, matrix sel_mat (!is.null(global_id_list)), subset_frac<0.7, RAM pressure threshold, shared_hat!=FALSE |
| **Dense fallback** | subset_frac = length(all_needed_cols)/active_ncol ≥ 0.7 → skip B19. Prevents IPC overhead when no subset savings. |
| **Security** | Sys.chmod(ipc_hat, "0600") immediately after write_ipc_file and BEFORE daemon spawn. |
| **on.exit order** | unlink(ipc_hat) registered AFTER daemon spawn with `add=TRUE, after=TRUE` so execution order = daemons(0) FIRST then unlink. |
| **Format** | arrow in Suggests (not Imports). All arrow:: calls inside requireNamespace guard. |
| **Correctness** | BYTE-IDENTICAL to non-B19 path (max_abs_diff == 0) for compact AND dense edge case. |
| **Fallback** | arrow absent + shared_hat="auto" → silent fallback FALSE. shared_hat=TRUE → cli_abort with specified message. |
| **Pool** | Sequential pool=TRUE reuse supported via ipc_hat string comparison. Concurrent pool reuse unsupported; document in @details. |
| **Cleanup** | on.exit(unlink(ipc_hat)) fires even on error (add=TRUE). |

## IV. Step-by-Step Logic

### Step 1 — DESCRIPTION
Add `arrow` to Suggests field.

### Step 2 — shared_hat param: 6 wrappers + rml()
Add `shared_hat = FALSE` to signatures of rml(), ctrml(), ctrml_fit(), terml(), terml_fit(), csrml(), csrml_fit(). Each wrapper passes `pool = pool, shared_hat = shared_hat` to rml().

Add roxygen @param shared_hat with this exact text (consistent across all files; terml/csrml inherit via @inheritParams ctrml):
```r
#' @param shared_hat Character or logical. Controls Arrow IPC memory sharing for
#'   the \code{hat} matrix across parallel workers. One of \code{FALSE} (default),
#'   \code{TRUE}, or \code{"auto"}.
#'
#'   - \code{FALSE} (default): Current behavior. \code{hat} serialized per daemon.
#'   - \code{TRUE}: Force Arrow IPC. Requires \pkg{arrow}. Error if absent.
#'   - \code{"auto"}: Enable when ALL hold: (1) parallel (n_workers > 1),
#'     (2) arrow available (else silent fallback), (3) sparse features
#'     (column coverage < 70%), (4) RAM pressure
#'     (\code{ncol_subset * NROW(hat) * 8 * (n_workers - 1) > 0.5 * available_ram_bytes()}).
#'
#'   Memory savings when active: \code{(1 - subset_frac) * (n_workers - 1) * hat_bytes}.
#'   \pkg{arrow} in Suggests: \code{install.packages("arrow")}.
#'
#'   \strong{Pool interaction (\code{pool = TRUE}):} Sequential reuse supported
#'   (new IPC path per call). Concurrent \code{rml()} on same pool NOT supported
#'   with \code{shared_hat != FALSE}.
#'
#'   \strong{Note:} hat written to a 0600 tempfile; deleted on exit. Avoid on
#'   shared-\code{/tmp} HPC systems; use \code{XDG_RUNTIME_DIR} if available.
```

### Step 3 — b19_daemon_load() helper in R/utils.R
Place after serialize_fit:
```r
b19_daemon_load <- function(ipc_hat, kset, envir = .GlobalEnv) {
  tbl_d <- arrow::read_ipc_file(ipc_hat, as_data_frame = FALSE, mmap = TRUE)
  local_hat <- as.matrix(as.data.frame(tbl_d))
  dimnames(local_hat) <- NULL
  parts_local <- FoReco::FoReco2matrix(local_hat, kset)
  assign(".b19_hat",      local_hat,   envir = envir)
  assign(".b19_parts",    parts_local, envir = envir)
  assign(".b19_hat_path", ipc_hat,     envir = envir)
  invisible(list(local_hat = local_hat, parts_local = parts_local))
}
```

### Step 4 — rml() pre-loop: compute use_b19 + write IPC

After computing global_id_list (spd.15 block) and before daemon spawn:

```r
use_b19 <- !isFALSE(shared_hat) &&
  !is.null(kset) && is.null(h) &&
  n_workers_resolved > 1L &&
  requireNamespace("arrow", quietly = TRUE) &&
  !is.null(global_id_list) &&
  {
    subset_frac <- length(unique(unlist(global_id_list))) / active_ncol
    subset_frac < 0.7
  } &&
  {
    hat_nrow <- if (!is.null(hat)) NROW(hat) else if (!is.null(base)) NROW(base) else 1L
    all_needed_ncol <- round(subset_frac * active_ncol)
    (hat_nrow * all_needed_ncol * 8L * (n_workers_resolved - 1L)) >
      0.5 * available_ram_bytes()
  }

# Error for explicit TRUE + no arrow
if (isTRUE(shared_hat) && !requireNamespace("arrow", quietly = TRUE)) {
  cli::cli_abort(c(
    "shared_hat = TRUE requires the {.pkg arrow} package.",
    "i" = "Install with {.run install.packages('arrow')}.",
    "x" = "Set {.code shared_hat = FALSE} to disable."
  ))
}

ipc_hat <- NULL
col_remap <- NULL

if (use_b19) {
  all_needed_cols <- sort(unique(unlist(global_id_list)))
  hat_subset <- hat[, all_needed_cols, drop = FALSE]
  schema_f64 <- arrow::schema(!!!setNames(
    rep(list(arrow::float64()), ncol(hat_subset)),
    paste0("C", seq_len(ncol(hat_subset)))))
  ipc_hat <- tempfile("FoRecoML_B19_", fileext = ".arrow")
  arrow::write_ipc_file(
    arrow::Table$create(as.data.frame(hat_subset), schema = schema_f64), ipc_hat)
  Sys.chmod(ipc_hat, "0600")  # SECURITY: before daemon spawn
  col_remap <- match(seq_len(active_ncol), all_needed_cols)
  cli::cli_inform(c(
    "i" = "Arrow IPC hat sharing activated (shared_hat = 'auto').",
    " " = "{ncol(hat_subset)} / {active_ncol} cols ({round(100*subset_frac)}%) written to {.path {ipc_hat}}."
  ))
}
```

### Step 5 — Register IPC cleanup AFTER daemon spawn

In the existing daemon spawn block (after on.exit(daemons(0))):
```r
# EXISTING:
mirai::daemons(n_workers_resolved, seed = mirai_seed)
on.exit(mirai::daemons(0), add = TRUE, after = TRUE)
mirai::everywhere({ library(FoRecoML) })

# ADD (after the above):
if (use_b19) on.exit(unlink(ipc_hat, force = TRUE), add = TRUE, after = TRUE)
# Execution order: daemons(0) fires first, then unlink — correct
```

### Step 6 — Update .args construction

In both sequential + mirai_map dispatch:
```r
# Current (kset path):
# parts_hat = parts_hat, ...

# B19 path replaces parts_hat:
# When use_b19=TRUE:
#   ipc_hat = ipc_hat, col_remap = col_remap, b19_active = TRUE
#   parts_hat omitted from .args (saves hat_bytes per daemon)
# When use_b19=FALSE:
#   b19_active = FALSE (or omit; loop_body_kset defaults to FALSE)
#   parts_hat = parts_hat (existing)
```

### Step 7 — Update loop_body_kset

Add `b19_active = FALSE` to function signature (24→25 formals). Update comment.

Replace X construction block:
```r
# B19 lazy-load
b19_env <- .GlobalEnv
if (isTRUE(b19_active)) {
  if (!exists(".b19_hat", envir = b19_env, inherits = FALSE) ||
      !identical(get(".b19_hat_path", envir = b19_env, inherits = FALSE), ipc_hat)) {
    FoRecoML:::b19_daemon_load(ipc_hat, kset, envir = b19_env)
  }
}

# X construction: B19 or current
if (isTRUE(b19_active)) {
  local_cols <- col_remap[global_id]  # remap original → local col index
  X <- FoRecoML:::input2rtw_partial_from_parts(
    get(".b19_parts", envir = b19_env, inherits = FALSE), kset, cols = local_cols)
} else {
  X <- FoRecoML:::input2rtw_partial_from_parts(parts_hat, kset, cols = global_id)
}
```

Update defensive guard:
```r
# OLD:
if (is.null(fit) && is.null(parts_hat) && is.null(sorted_vec_hat)) cli_abort(...)
# NEW:
if (!isTRUE(b19_active) && is.null(fit) && is.null(parts_hat) && is.null(sorted_vec_hat)) cli_abort(...)
```

### Step 8 — devtools::document()

Run to regenerate man pages. Verify @param shared_hat appears in ctrml.Rd, terml.Rd, csrml.Rd.

### Step 9 — Tests: tests/testthat/test-b19-shared-hat.R

6 test scenarios (all `skip_on_cran()` + `skip_if_not_installed("arrow")`):

1. **b19_daemon_load direct (no mirai)**: Write temp IPC; call b19_daemon_load(ipc_hat, kset, envir=e); assert e$.b19_parts is non-null; e$.b19_hat dimensions correct; identical(as.matrix(as.data.frame(arrow::read_ipc_file(ipc_hat, as_data_frame=FALSE))), e$.b19_hat) TRUE.

2. **Byte-identical compact**: Inject tiny RAM (cache_swap(1L)) → force use_b19=TRUE; ctrml compact lightgbm; compare bts vs non-B19 (cache_swap(1e12)) at tolerance=0.

3. **Dense edge case (global_id_list=NULL)**: ctrml mfh-all (sel_mat=1 → global_id_list=NULL → use_b19=FALSE by guard); verify same output regardless of shared_hat setting. No IPC created.

4. **IPC existence and cleanup**: cache_swap(1L) + withr::defer cleanup; call ctrml with n_workers=2; assert IPC file EXISTED (capture path from cli_inform message OR check tempdir pre/post); assert !file.exists(ipc_path) after call returns.

5. **shared_hat=TRUE + arrow absent**: mock requireNamespace("arrow")=FALSE; expect_error(ctrml(..., shared_hat=TRUE), regexp="requires the .+ arrow").

6. **Dense fallback (subset_frac >= 0.7)**: ctrml features="all" (dense sel_mat); inject RAM pressure; assert B19 NOT activated (no cli_inform "Arrow IPC", no IPC file written).

### Step 10 — Run tests

`Rscript -e 'devtools::test()' 2>&1 | tail -5` — all 370+ pass + 6 new.

### Step 11 — Commit

`feat(rml): Arrow IPC prefetch-per-daemon shared hat (B19 FoRecoML-rwj)`

Body must include: mechanism summary, security (0600 chmod), fallback chain, byte-identical verified, dense guard explanation. No AI attribution.

## V. Output Schema

```
files_modified: [R/reco_ml.R, R/utils.R, R/ctrml.R, R/terml.R, R/csrml.R, DESCRIPTION]
new_test_file: tests/testthat/test-b19-shared-hat.R
man_pages_regenerated: [ctrml.Rd, terml.Rd, csrml.Rd]
test_count_pre: 370
test_count_post: >= 376
commit_count: 1
```

## VI. Definition of Done

- [ ] arrow in DESCRIPTION:Suggests
- [ ] shared_hat = FALSE param in rml() + 6 wrappers; @param text matches spec
- [ ] b19_daemon_load() in utils.R (testable without mirai)
- [ ] use_b19 7-condition gate; dense fallback at subset_frac >= 0.7
- [ ] Sys.chmod("0600") before daemon spawn
- [ ] on.exit(unlink) registered AFTER daemon spawn (after=TRUE)
- [ ] b19_active=FALSE formal in loop_body_kset; defensive guard updated
- [ ] X construction branches: B19 (cached parts_local) vs current (parts_hat)
- [ ] BYTE-IDENTICAL: compact at tolerance=0 AND dense (global_id_list=NULL) edge case
- [ ] IPC file existence verified pre-cleanup in test
- [ ] shared_hat=TRUE + arrow absent → correct cli_abort
- [ ] Dense (subset_frac>=0.7): B19 NOT activated
- [ ] devtools::document() → man pages updated
- [ ] All 370+ baseline tests pass (shared_hat=FALSE default unchanged)
- [ ] mw3.3 invariant; spd.12+13+14 equivalence preserved
- [ ] Single commit; conventional commits; no AI attribution
- [ ] @details in rml.Rd: pool=TRUE + concurrent rml() marked unsupported with B19
- [ ] R3 cross-filesystem guard: daemon lazy-load wraps read in tryCatch; on file-not-found → cli_warn + fall back to current path (no abort)

## Risk R3: Cross-filesystem IPC (cross-mount / HPC guard in daemon)
In b19_daemon_load(), add fallback for inaccessible IPC file:
```r
b19_daemon_load <- function(ipc_hat, kset, envir = .GlobalEnv) {
  if (!file.exists(ipc_hat)) {
    cli::cli_warn(c(
      "Arrow IPC file not accessible in daemon: {.path {ipc_hat}}.",
      "i" = "Falling back to standard serialization path."
    ))
    return(invisible(NULL))  # NULL triggers caller to use parts_hat fallback
  }
  tbl_d <- arrow::read_ipc_file(ipc_hat, as_data_frame = FALSE, mmap = TRUE)
  ...rest unchanged...
}
```
In loop_body_kset, after b19_daemon_load() call: if return is NULL → fall through to current parts_hat path.
This handles cross-mount /tmp on HPC without crashing.
