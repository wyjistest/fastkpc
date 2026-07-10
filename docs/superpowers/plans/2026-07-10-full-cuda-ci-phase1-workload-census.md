# Full-CUDA CI Phase 1 Workload Census Implementation Plan

> **Status: SUPERSEDED. Do not execute this plan.** Expert review identified
> specification errors in the Phase 0 input contract, trace lineage, formula
> enum, route-metric gates, metadata schema, key serialization, risk contract,
> legacy-layout parity, and roadmap terminology. The amended design in
> `docs/superpowers/specs/2026-07-10-full-cuda-ci-phase1-workload-census-design.md`
> must be approved before this file is replaced with the required six-stage
> implementation plan.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete, restartable 351x48 logical-CI and legacy-mgcv workload/risk census required by Phase 1 of `goal-5.6.md`.

**Architecture:** Derive the canonical logical-test and global `target|S` request sets offline from the immutable Phase 0 oracle. Fit every globally unique key once through authoritative `mgcv::gam` in deterministic, resumable shards, then merge the metadata into a standard fail-closed artifact without executing or instrumenting a skeleton.

**Tech Stack:** R 4.4, mgcv, digest, jsonlite, base `parallel`, Phase 0 full-CUDA CI oracle artifacts.

---

## File Map

```text
Create  fastkpc/R/full_cuda_ci_workload_census.R
Create  fastkpc/tools/run_full_cuda_ci_workload_census.R
Create  fastkpc/tests/test_full_cuda_ci_workload_census.R
Create  fastkpc/tests/test_full_cuda_ci_workload_census_real_subset.R
Modify  goal-5.6.md
```

`full_cuda_ci_workload_census.R` owns normalization, request expansion,
metadata extraction, sharding, merge validation, and artifact writing. The tool
contains only argument/environment handling and exit status. The focused test
uses synthetic data; the real-subset test uses a bounded set of canonical keys.

---

### Task 1: Canonical Logical-Test Census

**Files:**
- Create: `fastkpc/R/full_cuda_ci_workload_census.R`
- Create: `fastkpc/tests/test_full_cuda_ci_workload_census.R`

- [ ] **Step 1: Write the failing logical-census test**

Create a four-row logical trace with one unconditional row, full-smooth rows at
`|S|=1` and `|S|=2`, and an additive row at `|S|=3`. Assert exact formula route,
alpha distance, residual keys, and selected-sepset marking.

```r
source("fastkpc/R/full_cuda_ci_workload_census.R")

trace <- data.frame(
  logical_sequence_id = 1:4,
  source_task_index = c(1L, 4L, 9L, 15L),
  level = 0:3,
  x = c(1L, 1L, 2L, 3L),
  y = c(2L, 3L, 4L, 5L),
  S_key = c("", "4", "1|5", "1|2|6"),
  p_value = c(0.02, 0.11, 0.08, 0.20),
  deletes_edge = c(FALSE, TRUE, FALSE, TRUE),
  stringsAsFactors = FALSE
)
deletions <- data.frame(
  level = c(1L, 3L),
  edge_x = c(1L, 3L),
  edge_y = c(3L, 5L),
  S_key = c("4", "1|2|6"),
  stringsAsFactors = FALSE
)

rows <- fastkpc_full_cuda_census_logical_tests(trace, deletions, alpha = 0.1)
stopifnot(
  identical(rows$formula_route,
            c("unconditional", "full_smooth", "full_smooth",
              "additive_smooth")),
  identical(rows$selected_sepset, c(FALSE, TRUE, FALSE, TRUE)),
  identical(rows$residual_key_x,
            c("", "1|4", "2|1|5", "3|1|2|6")),
  identical(rows$residual_key_y,
            c("", "3|4", "4|1|5", "5|1|2|6"))
)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
Rscript fastkpc/tests/test_full_cuda_ci_workload_census.R
```

Expected: FAIL because `fastkpc_full_cuda_census_logical_tests` is undefined.

- [ ] **Step 3: Implement logical normalization**

Add these functions:

```r
source("fastkpc/R/full_cuda_ci_gate.R")

fastkpc_full_cuda_census_formula_route <- function(S_size) {
  ifelse(S_size == 0L, "unconditional",
         ifelse(S_size <= 2L, "full_smooth", "additive_smooth"))
}

fastkpc_full_cuda_census_residual_key <- function(target, S_key) {
  if (!nzchar(S_key)) return("")
  paste(as.integer(target), S_key, sep = "|")
}

fastkpc_full_cuda_census_logical_tests <- function(trace, deletions, alpha) {
  trace <- as.data.frame(trace, stringsAsFactors = FALSE)
  S_size <- vapply(trace$S_key, function(value) {
    length(fastkpc_full_cuda_parse_s_key(value))
  }, integer(1))
  deletion_key <- paste(deletions$level, deletions$edge_x,
                        deletions$edge_y, deletions$S_key, sep = "|")
  row_key <- paste(trace$level, pmin(trace$x, trace$y),
                   pmax(trace$x, trace$y), trace$S_key, sep = "|")
  p_floor <- .Machine$double.xmin
  data.frame(
    logical_sequence_id = as.integer(trace$logical_sequence_id),
    source_task_index = as.integer(trace$source_task_index),
    level = as.integer(trace$level),
    x = as.integer(trace$x),
    y = as.integer(trace$y),
    S_key = as.character(trace$S_key),
    S_size = S_size,
    formula_route = fastkpc_full_cuda_census_formula_route(S_size),
    reference_p_value = as.numeric(trace$p_value),
    alpha = as.numeric(alpha),
    deletes_edge = as.logical(trace$deletes_edge),
    selected_sepset = row_key %in% deletion_key,
    absolute_distance_from_alpha = abs(as.numeric(trace$p_value) - alpha),
    log_distance_from_alpha = abs(log(pmax(as.numeric(trace$p_value),
                                           p_floor)) - log(alpha)),
    residual_key_x = mapply(fastkpc_full_cuda_census_residual_key,
                            trace$x, trace$S_key, USE.NAMES = FALSE),
    residual_key_y = mapply(fastkpc_full_cuda_census_residual_key,
                            trace$y, trace$S_key, USE.NAMES = FALSE),
    stringsAsFactors = FALSE
  )
}
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the same test. Expected: `PASS full CUDA CI workload census`.

- [ ] **Step 5: Commit Task 1**

```bash
git add fastkpc/R/full_cuda_ci_workload_census.R \
        fastkpc/tests/test_full_cuda_ci_workload_census.R
git commit -m "feat: add canonical full CUDA CI logical census"
```

---

### Task 2: Residual Request Expansion and Global Deduplication

**Files:**
- Modify: `fastkpc/R/full_cuda_ci_workload_census.R`
- Modify: `fastkpc/tests/test_full_cuda_ci_workload_census.R`

- [ ] **Step 1: Extend the test with request assertions**

```r
requests <- fastkpc_full_cuda_census_residual_requests(rows)
stopifnot(
  nrow(requests$expanded) == 6L,
  nrow(requests$unique) == 6L,
  identical(sort(unique(requests$unique$S_key)),
            c("1|2|6", "1|5", "4")),
  all(requests$unique$request_multiplicity == 1L),
  all(requests$unique$same_S_group_size == 2L)
)

duplicated_rows <- rbind(rows, rows[2L, , drop = FALSE])
duplicated_rows$logical_sequence_id[nrow(duplicated_rows)] <- 5L
duplicated <- fastkpc_full_cuda_census_residual_requests(duplicated_rows)
stopifnot(
  nrow(duplicated$expanded) == 8L,
  nrow(duplicated$unique) == 6L,
  duplicated$unique$request_multiplicity[
    duplicated$unique$residual_key == "1|4"
  ] == 2L
)
```

- [ ] **Step 2: Run and verify RED**

Expected: FAIL because the residual request function is undefined.

- [ ] **Step 3: Implement expansion and deduplication**

```r
fastkpc_full_cuda_census_residual_requests <- function(logical_tests) {
  conditional <- logical_tests[logical_tests$S_size > 0L, , drop = FALSE]
  expanded <- rbind(
    data.frame(
      logical_sequence_id = conditional$logical_sequence_id,
      target_side = "x",
      target = conditional$x,
      residual_key = conditional$residual_key_x,
      S_key = conditional$S_key,
      S_size = conditional$S_size,
      formula_route = conditional$formula_route,
      stringsAsFactors = FALSE
    ),
    data.frame(
      logical_sequence_id = conditional$logical_sequence_id,
      target_side = "y",
      target = conditional$y,
      residual_key = conditional$residual_key_y,
      S_key = conditional$S_key,
      S_size = conditional$S_size,
      formula_route = conditional$formula_route,
      stringsAsFactors = FALSE
    )
  )
  expanded <- expanded[order(expanded$logical_sequence_id,
                             expanded$target_side), , drop = FALSE]
  split_rows <- split(expanded, expanded$residual_key, drop = TRUE)
  unique_rows <- do.call(rbind, lapply(split_rows, function(group) {
    row <- group[1L, , drop = FALSE]
    row$request_multiplicity <- nrow(group)
    row$first_logical_sequence_id <- min(group$logical_sequence_id)
    row$last_logical_sequence_id <- max(group$logical_sequence_id)
    row
  }))
  group_sizes <- table(unique_rows$S_key)
  S_levels <- sort(unique(unique_rows$S_key))
  unique_rows$same_S_group_id <- match(unique_rows$S_key, S_levels)
  unique_rows$same_S_group_size <- as.integer(group_sizes[unique_rows$S_key])
  rownames(expanded) <- NULL
  rownames(unique_rows) <- NULL
  list(expanded = expanded, unique = unique_rows)
}
```

- [ ] **Step 4: Run and verify GREEN**

Expected: focused test passes with six canonical unique keys.

- [ ] **Step 5: Commit Task 2**

```bash
git add fastkpc/R/full_cuda_ci_workload_census.R \
        fastkpc/tests/test_full_cuda_ci_workload_census.R
git commit -m "feat: derive canonical target-S census"
```

---

### Task 3: Legacy mgcv Metadata Extraction

**Files:**
- Modify: `fastkpc/R/full_cuda_ci_workload_census.R`
- Modify: `fastkpc/tests/test_full_cuda_ci_workload_census.R`

- [ ] **Step 1: Add one-key metadata tests for `|S|=1,2,3`**

Use deterministic nonlinear data with six columns. Assert formula routing,
selected-sp length, penalty count, matrix dimensions, finite fit time, hashes,
and explicit convergence/error fields.

```r
set.seed(5601)
n <- 80L
z <- matrix(stats::rnorm(n * 6L), nrow = n)
data <- cbind(
  sin(z[, 2L]) + stats::rnorm(n, sd = 0.05),
  z[, 1L] + z[, 3L]^2,
  z[, 3:6, drop = FALSE]
)
colnames(data) <- paste0("V", seq_len(ncol(data)))

for (key in c("1|4", "2|1|5", "3|1|2|6")) {
  request <- requests$unique[requests$unique$residual_key == key,
                             , drop = FALSE]
  metadata <- fastkpc_full_cuda_census_mgcv_metadata(data, request)
  stopifnot(
    metadata$fit_status == "ok",
    metadata$fit_error == "",
    metadata$fit_time_ms >= 0,
    metadata$model_matrix_nrow == n,
    metadata$model_matrix_ncol > 0L,
    metadata$penalty_count >= 1L,
    nzchar(metadata$residual_hash),
    nzchar(metadata$fitted_hash),
    nzchar(metadata$model_matrix_hash)
  )
}
```

- [ ] **Step 2: Run and verify RED**

Expected: FAIL because the mgcv metadata extractor is undefined.

- [ ] **Step 3: Implement exact formula and metadata helpers**

Implement these named helpers in the census module:

```r
fastkpc_full_cuda_census_formula <- function(S_size) {
  predictors <- paste0("s", seq_len(as.integer(S_size)))
  rhs <- if (S_size <= 2L) {
    paste0("s(", paste(predictors, collapse = ","), ")")
  } else {
    paste0("s(", predictors, ")", collapse = " + ")
  }
  stats::as.formula(paste("target ~", rhs), env = parent.frame())
}

fastkpc_full_cuda_census_matrix_rank <- function(value) {
  value <- as.matrix(value)
  if (nrow(value) == 0L || ncol(value) == 0L) return(0L)
  as.integer(qr(value)$rank)
}

fastkpc_full_cuda_census_matrix_condition <- function(value) {
  value <- as.matrix(value)
  if (nrow(value) == 0L || ncol(value) == 0L) return(NA_real_)
  if (fastkpc_full_cuda_census_matrix_rank(value) < ncol(value)) return(Inf)
  as.numeric(kappa(value, exact = FALSE))
}

fastkpc_full_cuda_census_penalty_metadata <- function(fit) {
  penalties <- unlist(lapply(fit$smooth, function(smooth) smooth$S),
                      recursive = FALSE)
  constraints <- lapply(fit$smooth, function(smooth) {
    value <- smooth$C
    if (is.null(value)) matrix(numeric(), nrow = 0L, ncol = 0L) else
      as.matrix(value)
  })
  list(
    penalty_count = length(penalties),
    penalty_block_dimensions = paste(vapply(penalties, function(value) {
      paste(dim(as.matrix(value)), collapse = "x")
    }, character(1)), collapse = "|"),
    penalty_ranks = paste(vapply(penalties, function(value) {
      fastkpc_full_cuda_census_matrix_rank(value)
    }, integer(1)), collapse = "|"),
    constraint_dimensions = paste(vapply(constraints, function(value) {
      paste(dim(value), collapse = "x")
    }, character(1)), collapse = "|")
  )
}

fastkpc_full_cuda_census_hash_numeric <- function(value) {
  digest::digest(round(as.numeric(value), digits = 14L), algo = "sha256",
                 serialize = TRUE)
}
```

The extractor must:

```r
S <- fastkpc_full_cuda_parse_s_key(request$S_key[[1L]])
target <- as.integer(request$target[[1L]])
fit_data <- data.frame(target = data[, target, drop = TRUE],
                       data[, S, drop = FALSE], check.names = FALSE)
names(fit_data) <- c("target", paste0("s", seq_along(S)))
rhs <- if (length(S) <= 2L) {
  paste0("s(", paste(names(fit_data)[-1L], collapse = ","), ")")
} else {
  paste0("s(", names(fit_data)[-1L], ")", collapse = " + ")
}
formula <- stats::as.formula(paste("target ~", rhs), env = environment())
fit <- mgcv::gam(formula, data = fit_data)
model_matrix <- stats::predict(fit, type = "lpmatrix")
```

Return exactly one data-frame row for success or failure. Failure rows retain
`residual_key`, target/S metadata, `fit_status="error"`, and the condition
message; they must not be dropped.

Use `digest::digest(value, algo="sha256", serialize=TRUE)` over values rounded to
14 digits for residual, fitted, model-matrix, and coefficient hashes.

Implement `fastkpc_full_cuda_census_mgcv_metadata()` as a `tryCatch` around the
exact fit. The success branch returns this complete row schema:

```r
penalty <- fastkpc_full_cuda_census_penalty_metadata(fit)
conditioning <- scale(as.matrix(fit_data[-1L]), center = TRUE, scale = FALSE)
conditioning_sd <- vapply(fit_data[-1L], stats::sd, numeric(1))
selected_sp <- if (length(fit$sp)) fit$sp else fit$full.sp
data.frame(
  residual_key = request$residual_key[[1L]],
  target = target,
  S_key = request$S_key[[1L]],
  S_size = length(S),
  formula_route = request$formula_route[[1L]],
  fit_status = "ok",
  fit_error = "",
  fit_time_ms = fit_time_ms,
  formula = paste(deparse(formula), collapse = ""),
  method = as.character(fit$method),
  optimizer = paste(as.character(fit$optimizer), collapse = "|"),
  family = as.character(fit$family$family),
  link = as.character(fit$family$link),
  converged = isTRUE(fit$converged),
  outer_iterations = if (is.null(fit$outer.info$iter)) NA_integer_ else
    as.integer(fit$outer.info$iter),
  outer_convergence = if (is.null(fit$outer.info$conv)) "" else
    paste(as.character(fit$outer.info$conv), collapse = "|"),
  outer_gradient_max = if (is.null(fit$outer.info$grad)) NA_real_ else
    max(abs(as.numeric(fit$outer.info$grad))),
  selected_sp = paste(signif(selected_sp, 16L), collapse = "|"),
  selected_sp_count = length(selected_sp),
  GCV_Cp_score = as.numeric(fit$gcv.ubre),
  EDF = sum(as.numeric(fit$edf)),
  coefficient_rank = as.integer(fit$rank),
  model_matrix_nrow = nrow(model_matrix),
  model_matrix_ncol = ncol(model_matrix),
  model_matrix_rank = fastkpc_full_cuda_census_matrix_rank(model_matrix),
  model_matrix_condition =
    fastkpc_full_cuda_census_matrix_condition(model_matrix),
  penalty_count = penalty$penalty_count,
  penalty_block_dimensions = penalty$penalty_block_dimensions,
  penalty_ranks = penalty$penalty_ranks,
  constraint_dimensions = penalty$constraint_dimensions,
  conditioning_rank = fastkpc_full_cuda_census_matrix_rank(conditioning),
  conditioning_condition =
    fastkpc_full_cuda_census_matrix_condition(conditioning),
  conditioning_rank_deficient =
    fastkpc_full_cuda_census_matrix_rank(conditioning) < length(S),
  near_constant_conditioning_count =
    sum(!is.finite(conditioning_sd) |
          conditioning_sd <= sqrt(.Machine$double.eps)),
  target_sd = stats::sd(fit_data$target),
  target_near_constant = !is.finite(stats::sd(fit_data$target)) ||
    stats::sd(fit_data$target) <= sqrt(.Machine$double.eps),
  residual_hash = fastkpc_full_cuda_census_hash_numeric(stats::residuals(fit)),
  fitted_hash = fastkpc_full_cuda_census_hash_numeric(stats::fitted(fit)),
  model_matrix_hash =
    fastkpc_full_cuda_census_hash_numeric(model_matrix),
  coefficient_hash =
    fastkpc_full_cuda_census_hash_numeric(stats::coef(fit)),
  mgcv_version = as.character(utils::packageVersion("mgcv")),
  R_version = R.version.string,
  stringsAsFactors = FALSE
)
```

The error branch returns the same columns with identity fields preserved,
`fit_status="error"`, `fit_error=conditionMessage(error)`, scalar diagnostics
set to `NA`, and hashes set to empty strings.

Also add `fastkpc_full_cuda_census_empty_metadata()` by evaluating the error-row
constructor with zero rows. The structure-only runner passes this typed frame to
the artifact writer, so every table/summary operation sees the complete schema
even when no numerical shard has run.

- [ ] **Step 4: Run and verify GREEN**

Expected: focused test passes all three formula classes with no fit errors.

- [ ] **Step 5: Add rank-deficient and near-constant tests**

Create duplicated conditioning columns and a constant target. Assert the rows
remain present with `conditioning_rank_deficient=TRUE` and
`target_near_constant=TRUE`; no filtering is allowed.

- [ ] **Step 6: Commit Task 3**

```bash
git add fastkpc/R/full_cuda_ci_workload_census.R \
        fastkpc/tests/test_full_cuda_ci_workload_census.R
git commit -m "feat: collect legacy mgcv workload metadata"
```

---

### Task 4: Deterministic Shards and Resume Validation

**Files:**
- Modify: `fastkpc/R/full_cuda_ci_workload_census.R`
- Modify: `fastkpc/tests/test_full_cuda_ci_workload_census.R`

- [ ] **Step 1: Write shard assignment and resume tests**

Assert stable assignments, complete coverage, no duplicate keys, atomic shard
files, and rejection when `mgcv_version`, data hash, schema version, or shard
count changes.

```r
assigned_a <- fastkpc_full_cuda_census_assign_shards(
  requests$unique$residual_key, shard_count = 3L
)
assigned_b <- fastkpc_full_cuda_census_assign_shards(
  requests$unique$residual_key, shard_count = 3L
)
stopifnot(identical(assigned_a, assigned_b), all(assigned_a %in% 0:2))

manifest <- fastkpc_full_cuda_census_shard_manifest(
  data_hash = "data", oracle_hash = "oracle", shard_count = 3L,
  shard_id = 0L, schema_version = "phase1-v1"
)
stopifnot(fastkpc_full_cuda_census_shard_reusable(manifest, manifest))
changed <- manifest
changed$shard_count <- 4L
stopifnot(!fastkpc_full_cuda_census_shard_reusable(manifest, changed))
```

- [ ] **Step 2: Run and verify RED**

Expected: FAIL because shard helpers are undefined.

- [ ] **Step 3: Implement shard helpers and atomic write**

Use the first seven hex digits of `digest::digest(key, algo="xxhash32",
serialize=FALSE)` to avoid signed integer overflow:

```r
fastkpc_full_cuda_census_assign_shards <- function(keys, shard_count) {
  vapply(keys, function(key) {
    value <- digest::digest(key, algo = "xxhash32", serialize = FALSE)
    strtoi(substr(value, 1L, 7L), base = 16L) %% shard_count
  }, integer(1))
}
```

Write each shard to `<path>.tmp-<pid>`, then rename to the final RDS only after
both metadata rows and manifest validate.

- [ ] **Step 4: Implement shard execution**

```r
fastkpc_full_cuda_census_run_shard <- function(
    data, requests, shard_id, shard_count, shard_dir, expected_manifest) {
  path <- file.path(shard_dir, sprintf("shard_%03d.rds", shard_id))
  if (file.exists(path)) {
    existing <- readRDS(path)
    if (fastkpc_full_cuda_census_shard_reusable(
          existing$manifest, expected_manifest)) {
      return(path)
    }
  }
  assigned <- fastkpc_full_cuda_census_assign_shards(
    requests$residual_key, shard_count
  ) == shard_id
  shard_requests <- requests[assigned, , drop = FALSE]
  rows <- do.call(rbind, lapply(seq_len(nrow(shard_requests)), function(i) {
    fastkpc_full_cuda_census_mgcv_metadata(
      data, shard_requests[i, , drop = FALSE]
    )
  }))
  payload <- list(manifest = expected_manifest, rows = rows)
  dir.create(shard_dir, recursive = TRUE, showWarnings = FALSE)
  temporary <- paste0(path, ".tmp-", Sys.getpid())
  saveRDS(payload, temporary)
  if (!file.rename(temporary, path)) {
    unlink(temporary)
    stop("failed to publish census shard: ", path, call. = FALSE)
  }
  path
}
```

The parent runner uses `parallel::mclapply` only on Unix and records the worker
count. Non-Unix execution is serial and explicit in the manifest.

- [ ] **Step 5: Run and verify GREEN**

Expected: focused test creates and reuses all three shards.

- [ ] **Step 6: Commit Task 4**

```bash
git add fastkpc/R/full_cuda_ci_workload_census.R \
        fastkpc/tests/test_full_cuda_ci_workload_census.R
git commit -m "feat: shard full CUDA CI workload census"
```

---

### Task 5: Merge Validation and Standard Artifact Writer

**Files:**
- Modify: `fastkpc/R/full_cuda_ci_workload_census.R`
- Modify: `fastkpc/tests/test_full_cuda_ci_workload_census.R`

- [ ] **Step 1: Add merge failure tests**

Create valid shard fixtures, then separately remove one key and duplicate one
key. Assert both merges produce `pass=FALSE` and a persisted
`first_divergence.json` naming the missing or duplicate residual key.

- [ ] **Step 2: Run and verify RED**

Expected: FAIL because merge/artifact functions are undefined.

- [ ] **Step 3: Implement validated merge**

```r
fastkpc_full_cuda_census_merge_shards <- function(
    shard_paths, expected_requests) {
  # require all manifests to match
  # rbind rows in residual_key order
  # reject missing, unexpected, or duplicated residual keys
  # return rows plus first-divergence payload
}
```

- [ ] **Step 4: Implement standard artifact outputs**

Add:

```r
fastkpc_full_cuda_census_write_artifact <- function(
    oracle, logical_tests, requests, metadata, output_dir, commands,
    stage_timing, executed_fit_count = 273284L) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  paths <- fastkpc_full_cuda_artifact_paths(output_dir)
  saveRDS(logical_tests, file.path(output_dir, "logical_ci_tests.rds"))
  utils::write.csv(logical_tests,
                   file.path(output_dir, "logical_ci_tests.csv"),
                   row.names = FALSE)
  saveRDS(requests, file.path(output_dir, "residual_requests.rds"))
  utils::write.csv(requests,
                   file.path(output_dir, "residual_requests.csv"),
                   row.names = FALSE)
  saveRDS(metadata, file.path(output_dir, "residual_metadata.rds"))
  utils::write.csv(metadata,
                   file.path(output_dir, "residual_metadata.csv"),
                   row.names = FALSE)

  summary <- list(
    run_status = "ok",
    timeout = FALSE,
    source_commit = fastkpc_full_cuda_source_commit(),
    oracle_artifact = oracle$output_dir,
    candidate_route = "phase1-offline-legacy-mgcv-census",
    edge_count_reference = oracle$summary$edge_count_reference,
    edge_count_candidate = oracle$summary$edge_count_candidate,
    SHD = oracle$summary$SHD,
    adjacency_identical = oracle$summary$adjacency_identical,
    sepsets_identical = oracle$summary$sepsets_identical,
    n_edgetests_identical = oracle$summary$n_edgetests_identical,
    deletions_identical = oracle$summary$deletions_identical,
    unknown_fallback_count = 0L,
    approximate_backend_count = 0L,
    backend_fallback_error_count = 0L,
    logical_test_count = nrow(logical_tests),
    conditional_residual_request_count =
      sum(logical_tests$S_size > 0L) * 2L,
    canonical_unique_target_s_count = nrow(requests),
    unique_s_count = length(unique(requests$S_key)),
    s_affinity_executed_mgcv_fit_count = executed_fit_count,
    metadata_row_count = nrow(metadata),
    metadata_error_count = sum(metadata$fit_status != "ok"),
    pass = FALSE
  )
  summary$pass <- identical(summary$logical_test_count, 240489L) &&
    identical(summary$conditional_residual_request_count, 476552L) &&
    identical(summary$canonical_unique_target_s_count, 110617L) &&
    identical(summary$unique_s_count, 8634L) &&
    identical(summary$metadata_row_count, 110617L) &&
    identical(summary$metadata_error_count, 0L) &&
    isTRUE(summary$adjacency_identical) && isTRUE(summary$sepsets_identical) &&
    isTRUE(summary$n_edgetests_identical) &&
    isTRUE(summary$deletions_identical) && identical(summary$SHD, 0L)

  manifest <- list(
    schema_version = "full-cuda-ci-workload-census-v1",
    generated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
    source_commit = summary$source_commit,
    oracle_artifact = oracle$output_dir,
    oracle_data_hash = oracle$manifest$data_hash,
    oracle_source_commit = oracle$manifest$source_commit,
    logical_trace_hash = fastkpc_full_cuda_file_hash(
      oracle$paths$logical_ci_trace_rds
    ),
    p_value_floor = .Machine$double.xmin,
    formula_semantics_version = "legacy-regrXonS-v1",
    metadata_schema_version = "phase1-residual-metadata-v1"
  )
  fastkpc_full_cuda_write_json(manifest, paths$manifest_json)
  fastkpc_full_cuda_write_json(summary, paths$summary_json)
  utils::write.csv(as.data.frame(summary, stringsAsFactors = FALSE),
                   paths$summary_csv, row.names = FALSE)
  fastkpc_full_cuda_write_summary_md(
    summary, paths$summary_md, "Full CUDA CI Phase 1 workload census"
  )
  writeLines(commands, paths$commands_txt)
  writeLines(fastkpc_full_cuda_environment_lines(), paths$environment_txt)
  utils::write.csv(stage_timing, paths$stage_timing_csv, row.names = FALSE)
  file.copy(oracle$paths$graph_agreement_csv, paths$graph_agreement_csv,
            overwrite = TRUE)
  file.copy(oracle$paths$sepset_agreement_csv, paths$sepset_agreement_csv,
            overwrite = TRUE)
  file.copy(oracle$paths$n_edgetests_csv, paths$n_edgetests_csv,
            overwrite = TRUE)
  fastkpc_full_cuda_write_json(
    fastkpc_full_cuda_empty_first_divergence(),
    paths$first_divergence_json
  )
  utils::write.csv(data.frame(
    type = c("unknown", "approximate", "backend"),
    key = c("unknown_fallback_count", "approximate_backend_count",
            "backend_fallback_error_count"),
    reason = "Phase 1 summary counter",
    count = 0L,
    stringsAsFactors = FALSE
  ), paths$fallbacks_csv, row.names = FALSE)
  utils::write.csv(data.frame(
    run = 1L,
    elapsed_sec = sum(stage_timing$elapsed_ms) / 1000,
    pass = summary$pass
  ), paths$raw_runs_csv, row.names = FALSE)

  utils::write.csv(as.data.frame(table(S_size = requests$S_size)),
                   file.path(output_dir, "counts_by_s_size.csv"),
                   row.names = FALSE)
  utils::write.csv(as.data.frame(table(
    penalty_count = metadata$penalty_count
  )), file.path(output_dir, "counts_by_penalty_count.csv"),
  row.names = FALSE)
  utils::write.csv(as.data.frame(table(
    model_matrix_ncol = metadata$model_matrix_ncol
  )), file.path(output_dir, "counts_by_model_dimension.csv"),
  row.names = FALSE)
  condition_bucket <- cut(
    metadata$model_matrix_condition,
    breaks = c(-Inf, 1e4, 1e8, 1e12, Inf),
    labels = c("lt_1e4", "1e4_1e8", "1e8_1e12", "ge_1e12")
  )
  utils::write.csv(as.data.frame(table(condition_bucket,
                                       useNA = "ifany")),
                   file.path(output_dir, "counts_by_condition_bucket.csv"),
                   row.names = FALSE)
  utils::write.csv(as.data.frame(table(
    same_S_group_size = requests$same_S_group_size
  )), file.path(output_dir, "same_s_group_distribution.csv"),
  row.names = FALSE)
  utils::write.csv(
    logical_tests[order(logical_tests$absolute_distance_from_alpha),
                  , drop = FALSE],
    file.path(output_dir, "near_alpha_tests.csv"), row.names = FALSE
  )
  unsupported <- metadata[
    metadata$fit_status != "ok" |
      !is.finite(metadata$model_matrix_condition) |
      metadata$conditioning_rank_deficient |
      metadata$target_near_constant,
    , drop = FALSE
  ]
  utils::write.csv(unsupported,
                   file.path(output_dir, "unsupported_envelope.csv"),
                   row.names = FALSE)
  runtime_weighted <- stats::aggregate(
    metadata$fit_time_ms,
    by = list(S_size = metadata$S_size,
              penalty_count = metadata$penalty_count),
    FUN = sum
  )
  names(runtime_weighted)[[3L]] <- "fit_time_ms"
  utils::write.csv(
    runtime_weighted,
    file.path(output_dir, "runtime_by_s_size_penalty_count.csv"),
    row.names = FALSE
  )
  summary
}
```

Reuse `fastkpc_full_cuda_artifact_paths()` for the standard files. Write all
Phase 1 files listed in the design spec. Summary `pass` is true only when:

```text
logical_test_count = 240489
conditional_residual_request_count = 476552
canonical_unique_target_s_count = 110617
unique_s_count = 8634
metadata_row_count = 110617
metadata_error_count = 0
Phase 0 graph fields remain pass=true
fallback/error counts = 0
```

- [ ] **Step 5: Run and verify GREEN**

Expected: synthetic artifact contains every standard schema file and all
phase-specific RDS/CSV outputs.

- [ ] **Step 6: Commit Task 5**

```bash
git add fastkpc/R/full_cuda_ci_workload_census.R \
        fastkpc/tests/test_full_cuda_ci_workload_census.R
git commit -m "feat: write full CUDA CI workload census artifact"
```

---

### Task 6: Standard Runner and Real-Subset Gate

**Files:**
- Create: `fastkpc/tools/run_full_cuda_ci_workload_census.R`
- Create: `fastkpc/tests/test_full_cuda_ci_workload_census_real_subset.R`
- Modify: `fastkpc/tests/test_full_cuda_ci_workload_census.R`

- [ ] **Step 1: Write runner existence and fail-closed tests**

Assert the tool exists, rejects a mismatched Phase 0 source commit/data hash,
and writes `pass=false` rather than leaving a stale artifact.

- [ ] **Step 2: Implement runner arguments and environment**

Support:

```text
FASTKPC_FULL_CUDA_CI_ORACLE_DIR
FASTKPC_FULL_CUDA_CI_DATA_PATH
FASTKPC_FULL_CUDA_CI_CENSUS_OUTPUT_DIR
FASTKPC_FULL_CUDA_CI_CENSUS_SHARD_COUNT
FASTKPC_FULL_CUDA_CI_CENSUS_WORKERS
FASTKPC_FULL_CUDA_CI_CENSUS_SHARD_ID
FASTKPC_FULL_CUDA_CI_CENSUS_STRUCTURE_ONLY
FASTKPC_FULL_CUDA_CI_S_AFFINITY_RESULT_PATH
```

`SHARD_ID` runs one shard. Without it, the tool builds Stage A/B, runs or
reuses all shards, merges, writes the artifact, prints key counts, and exits 1
when `summary$pass` is not true.

- [ ] **Step 3: Add the real-subset test**

Load the canonical Phase 0 logical trace, select at least two keys for each of
`|S|=1`, `|S|=2`, and `|S|>2`, run metadata extraction, and assert:

```text
fit_status = ok for every row
formula route matches S size
selected sp count equals penalty count
residual/fitted/model hashes are nonempty
all output keys equal the requested keys
```

- [ ] **Step 4: Run focused and real-subset tests**

```bash
Rscript fastkpc/tests/test_full_cuda_ci_workload_census.R
Rscript fastkpc/tests/test_full_cuda_ci_workload_census_real_subset.R
git diff --check
```

Expected: both tests print PASS and diff check is silent.

- [ ] **Step 5: Commit Task 6**

```bash
git add fastkpc/tools/run_full_cuda_ci_workload_census.R \
        fastkpc/tests/test_full_cuda_ci_workload_census.R \
        fastkpc/tests/test_full_cuda_ci_workload_census_real_subset.R
git commit -m "feat: run full CUDA CI workload census"
```

---

### Task 7: Generate and Verify the Structural 351x48 Census

**Files:**
- Artifact only: `fastkpc/artifacts/full_cuda_ci/workload_census_351x48_v1/`

- [ ] **Step 1: Run structure-only census**

```bash
FASTKPC_FULL_CUDA_CI_CENSUS_STRUCTURE_ONLY=1 \
Rscript fastkpc/tools/run_full_cuda_ci_workload_census.R
```

- [ ] **Step 2: Verify canonical counts**

```bash
jq '{logical_test_count, conditional_residual_request_count,
     canonical_unique_target_s_count, unique_s_count,
     s_affinity_executed_mgcv_fit_count, pass}' \
  fastkpc/artifacts/full_cuda_ci/workload_census_351x48_v1/summary.json
```

Expected:

```json
{
  "logical_test_count": 240489,
  "conditional_residual_request_count": 476552,
  "canonical_unique_target_s_count": 110617,
  "unique_s_count": 8634,
  "s_affinity_executed_mgcv_fit_count": 273284,
  "pass": false
}
```

Structure-only remains `pass=false` because numerical metadata is incomplete.

- [ ] **Step 3: Verify no graph drift**

Confirm copied Phase 0 graph/sepset/test/deletion fields are all true and SHD is
zero.

---

### Task 8: Run Full Metadata Shards and Close Phase 1

**Files:**
- Modify: `goal-5.6.md`
- Artifact: `fastkpc/artifacts/full_cuda_ci/workload_census_351x48_v1/`

- [ ] **Step 1: Run all metadata shards**

Start conservatively with 64 deterministic shards and 20 Unix workers:

```bash
FASTKPC_FULL_CUDA_CI_CENSUS_SHARD_COUNT=64 \
FASTKPC_FULL_CUDA_CI_CENSUS_WORKERS=20 \
Rscript fastkpc/tools/run_full_cuda_ci_workload_census.R
```

The command may be rerun; completed matching shards must be reused.

- [ ] **Step 2: Verify the full summary**

```bash
jq '{pass, logical_test_count, conditional_residual_request_count,
     canonical_unique_target_s_count, unique_s_count, metadata_row_count,
     metadata_error_count, rank_deficient_count, near_constant_count,
     multi_penalty_count, elapsed_sec}' \
  fastkpc/artifacts/full_cuda_ci/workload_census_351x48_v1/summary.json
```

Required:

```text
pass = true
metadata_row_count = 110617
metadata_error_count = 0
all canonical structural counts exact
```

- [ ] **Step 3: Verify distributions and unsupported envelope**

Check every required summary CSV is nonempty where applicable, all counts sum
to 110,617, and every unsupported/high-risk key is present in
`residual_metadata`.

- [ ] **Step 4: Rerun Phase 0 gate**

```bash
Rscript fastkpc/tools/run_full_cuda_ci_oracle.R
```

Expected: `pass: TRUE`, `SHD: 0`, all graph-semantic fields true.

- [ ] **Step 5: Update the roadmap**

Change Phase 1 status in `goal-5.6.md` to complete and record artifact counts,
metadata error count, runtime, hashes, and the corrected distinction between
110,617 canonical unique keys and 273,284 S-affinity executed fits.

- [ ] **Step 6: Run final verification**

```bash
Rscript fastkpc/tests/test_full_cuda_ci_workload_census.R
Rscript fastkpc/tests/test_full_cuda_ci_workload_census_real_subset.R
Rscript fastkpc/tests/test_full_cuda_ci_oracle_gate.R
Rscript fastkpc/tools/run_full_cuda_ci_oracle.R
git diff --check
git status --short
```

- [ ] **Step 7: Commit and push Phase 1**

```bash
git add fastkpc/R/full_cuda_ci_workload_census.R \
        fastkpc/tools/run_full_cuda_ci_workload_census.R \
        fastkpc/tests/test_full_cuda_ci_workload_census.R \
        fastkpc/tests/test_full_cuda_ci_workload_census_real_subset.R \
        goal-5.6.md
git commit -m "feat: add full CUDA CI workload census"
git -c http.version=HTTP/1.1 \
    -c http.proxy=http://localhost:7890 \
    push --verbose origin HEAD:refs/heads/main
```

Phase 1 is complete only after the full 110,617-row metadata artifact passes.
