# Full-CUDA CI Phase 2 Prepared-S Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build and qualify the complete response-independent PreparedSSetup and target-specific TargetState corpus required by Phase 2 of goal-5.6.md.

**Architecture:** Authenticate the completed Phase 1 artifact, build one zero-response and no-fixed-sp normalized setup for each of 8,634 same-S groups, batch-project all 110,617 target states, and persist the result in restartable deterministic shards. Use a response-free mgcv C_magic fixed-sp adapter as the Phase 2 numerical authority. Run a class-complete 44-setup/270-key/44-dCov iteration gate during development and the extended 6,143-key qualification only at artifact gates. Production CUDA resources and stable GPU solves remain Phase 3 work.

**Tech Stack:** R 4.4.1, mgcv 1.9.1, kpcalg, digest, jsonlite, Rcpp/RcppArmadillo, base parallel, authenticated Phase 0/1 artifacts.

---

## Scope and Ordering

Implement exactly these nine independently reviewable tasks:

~~~text
1. Authenticate Phase 1 inputs and freeze PreparedSKey serialization
2. Build and validate response-independent PreparedSSetup objects
3. Build and validate batched TargetState objects
4. Add the response-free mgcv C_magic fixed-sp reference adapter
5. Select and execute the deterministic qualification subset
6. Add deterministic setup shards and restart qualification
7. Add the standard runner and artifact writer
8. Run focused tests and a scaled real dry run
9. Run the full Phase 2 artifact and close the roadmap phase
~~~

Do not modify legacy_runner.R, scheduler routing, the live skeleton, CUDA
kernels, or production backend selection in this plan. Do not start Task 9
until Tasks 1-8 pass.

## File Map

~~~text
Create  fastkpc/R/full_cuda_ci_prepared_s_contract.R
Create  fastkpc/tools/run_full_cuda_ci_prepared_s_contract.R
Create  fastkpc/tests/test_full_cuda_ci_prepared_s_contract.R
Create  fastkpc/tests/test_full_cuda_ci_target_retarget.R
Create  fastkpc/tests/test_full_cuda_ci_prepared_s_contract_restart.R
Create  fastkpc/tests/test_full_cuda_ci_prepared_s_contract_real_subset.R
Modify  goal-5.6.md only after the complete artifact passes
Artifact fastkpc/artifacts/full_cuda_ci/prepared_s_contract_v1/
~~~

full_cuda_ci_prepared_s_contract.R owns Phase 1 input authentication,
PreparedSKey serialization, PreparedSSetup and TargetState construction,
fixed-sp reference retargeting, qualification selection and parity, sharding,
merge validation, and artifact writing.

run_full_cuda_ci_prepared_s_contract.R owns environment parsing, worker
orchestration, stage timing, failure summaries, and command provenance.

---

### Task 1: Phase 1 Input Authentication and PreparedSKey

**Files:**
- Create: fastkpc/R/full_cuda_ci_prepared_s_contract.R
- Create: fastkpc/tests/test_full_cuda_ci_prepared_s_contract.R

- [ ] **Step 1: Write the failing input and key test**

Create fastkpc/tests/test_full_cuda_ci_prepared_s_contract.R:

~~~r
fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_error <- function(expression, pattern, message) {
  error <- tryCatch({ force(expression); NULL }, error = identity)
  assert_true(
    inherits(error, "error") &&
      grepl(pattern, conditionMessage(error), fixed = TRUE),
    message
  )
}

source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")

contract <- fastkpc_full_cuda_prepared_s_input_contract()
assert_true(
  identical(contract$schema_version, "full-cuda-ci-phase2-input-v1"),
  "Phase 2 input schema must be versioned"
)
assert_true(
  identical(contract$phase1_source_commit,
            "1560068ba8d635e806612554e11bbed92c0b8843c"),
  "Phase 2 must pin the artifact-producing Phase 1 commit"
)
assert_true(
  identical(unname(contract$file_hashes[["manifest.json"]]),
            "b0990cfc932a5fcabc09ad25e352e7babb67fc8a127f11c7d2b88887c4940574"),
  "Phase 2 must pin the Phase 1 manifest bytes"
)

inputs <- fastkpc_full_cuda_prepared_s_load_inputs(
  census_dir =
    "fastkpc/artifacts/full_cuda_ci/workload_census_351x48_v1",
  data_path = paste0(
    "fastkpc/artifacts/kpc_tprs_real_zhu/",
    "cancer_RD-causalDiscoveryInput.rds"
  )
)
assert_true(
  nrow(inputs$logical_tests) == 240489L &&
    nrow(inputs$residual_requests) == 110617L &&
    nrow(inputs$same_s_setup_metadata) == 8634L &&
    nrow(inputs$target_fit_metadata) == 110617L,
  "authenticated Phase 1 tables must retain canonical counts"
)

setup_row <- inputs$same_s_setup_metadata[1L, , drop = FALSE]
key <- fastkpc_full_cuda_prepared_s_key(
  setup_row = setup_row,
  dataset_sha256 = inputs$dataset_sha256,
  R_version = inputs$manifest$R_version,
  mgcv_version = inputs$manifest$mgcv_version
)
assert_true(
  grepl("schema_version=full-cuda-ci-prepared-s-key-v1\n",
        key$payload, fixed = TRUE) &&
    substr(key$payload, nchar(key$payload), nchar(key$payload)) == "\n" &&
    grepl("^[0-9a-f]{64}$", key$sha256),
  "PreparedSKey must use canonical LF-terminated UTF-8 and SHA-256"
)

collision <- tryCatch(
  fastkpc_full_cuda_prepared_s_validate_key_mapping(
    payload = c(key$payload, paste0(key$payload, "x")),
    hash = c(key$sha256, key$sha256)
  ),
  error = identity
)
assert_true(
  inherits(collision, "error") &&
    grepl("PreparedSKey hash collision", conditionMessage(collision),
          fixed = TRUE),
  "PreparedSKey collisions must fail closed"
)

cat("PASS full CUDA CI Prepared-S input and key contract\n")
~~~

- [ ] **Step 2: Run the test and verify RED**

Run:

~~~bash
Rscript fastkpc/tests/test_full_cuda_ci_prepared_s_contract.R
~~~

Expected: failure because
fastkpc/R/full_cuda_ci_prepared_s_contract.R does not exist.

- [ ] **Step 3: Implement the pinned input contract**

Create fastkpc/R/full_cuda_ci_prepared_s_contract.R with these sources and
contract:

~~~r
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/mgcv_compat_contract.R")
source("fastkpc/R/mgcv_extract_oracle.R")
source("fastkpc/R/native.R")

fastkpc_full_cuda_prepared_s_input_contract <- function() {
  list(
    schema_version = "full-cuda-ci-phase2-input-v1",
    phase1_source_commit =
      "1560068ba8d635e806612554e11bbed92c0b8843c",
    metadata_schema_version = "full-cuda-ci-metadata-v4",
    dataset_file_sha256 =
      "e03cbfafed3336f3da7725878e92af24556b5ebdc6227b94ea84469e62e94036",
    dataset_matrix_sha256 =
      "971f1e0784817c6febfc84154f19540be02a92f6e4acad784dc7f92b979f1df7",
    canonical_logical_census_hash =
      "c9b48074dd59a439fceb9d5e64806adda5620cc4abe32095371abc447ef98634",
    canonical_key_corpus_hash =
      "b843630969f116da63f7fad095c54de2ff471540159ff97ca56c3871d6b2e1fa",
    file_hashes = c(
      "manifest.json" =
        "b0990cfc932a5fcabc09ad25e352e7babb67fc8a127f11c7d2b88887c4940574",
      "summary.json" =
        "4f71d1bbbbdd2436e3576b728363120bb9b911897b9dee3ecf6f8a5d3379eb24",
      "logical_ci_tests.rds" =
        "17781421df868ae7822c022ac58ad6322292ad51053d5da686a3d6f79b40d7c8",
      "residual_requests.rds" =
        "d7a995f12f6bc118a39009b0b685cb5c28068d418c5a569d552cf26e8748ec8b",
      "same_s_setup_metadata.rds" =
        "8b35a463b17a64512d653da949f5ac74f7cc21223f346a304ac52fdfe8434a3f",
      "target_fit_metadata.rds" =
        "af09b5dc4c6a34d7ec126e1fe7f3f1f9c3d7fcb6316ada759a293abe76d8323c",
      "risk_cases.rds" =
        "1e0951e9856bea3c9a1b7ba83ec03b79a678e7aa60464d7f6808397ab8d9a7bc"
    ),
    named_metadata_hashes = c(
      "setup_observation_metadata" =
        "5282820451b2658c636132e579859c8c2c8e6497a926b8b6d9c393e0043e667a",
      "same_s_setup_metadata" =
        "07830db88c62aa7658d44373e86d897b254e453773b4c0070460dc20fce91113",
      "target_fit_metadata" =
        "361672b87cd056a689f578a5eb7660a55d056395ac270d3e28dbbe24738bab40",
      "target_risk_metadata" =
        "95eba27f5ea7904761ae4afbc203c58a84566fc3cc308d1b9c90acca69cc96f2",
      "risk_cases" =
        "4a2748ba469e039143c482fd4cf0367324886cc526552e9529117cab7c596d91"
    )
  )
}
~~~

- [ ] **Step 4: Implement the fail-closed loader**

Implement
fastkpc_full_cuda_prepared_s_load_inputs(census_dir, data_path, contract).
The function must:

~~~r
paths <- file.path(census_dir, names(contract$file_hashes))
actual <- vapply(paths, fastkpc_full_cuda_census_file_hash, character(1L))
if (!identical(unname(actual), unname(contract$file_hashes))) {
  stop("Phase 1 input file hash mismatch", call. = FALSE)
}
if (!identical(fastkpc_full_cuda_census_file_hash(data_path),
               contract$dataset_file_sha256)) {
  stop("canonical dataset file hash mismatch", call. = FALSE)
}

manifest <- jsonlite::read_json(
  file.path(census_dir, "manifest.json"), simplifyVector = TRUE
)
summary <- jsonlite::read_json(
  file.path(census_dir, "summary.json"), simplifyVector = TRUE
)
logical_tests <- readRDS(file.path(census_dir, "logical_ci_tests.rds"))
residual_requests <- readRDS(
  file.path(census_dir, "residual_requests.rds")
)
same_s_setup_metadata <- readRDS(
  file.path(census_dir, "same_s_setup_metadata.rds")
)
target_fit_metadata <- readRDS(
  file.path(census_dir, "target_fit_metadata.rds")
)
risk_cases <- readRDS(file.path(census_dir, "risk_cases.rds"))
data <- as.matrix(readRDS(data_path))
storage.mode(data) <- "double"
~~~

Require the exact counts and hashes from the design. Recompute named metadata
hashes with fastkpc_full_cuda_census_frame_hash(). Require
manifest$phase1_complete, summary$phase1_complete, summary$pass, exact Phase 1
source commit, exact R/mgcv versions, exact dataset matrix hash, and zero fit,
warning-classification, nonfinite-classification, fallback, and approximation
errors.

Load all 64 Phase 1 shard RDS/summary pairs, validate each with
fastkpc_full_cuda_census_validate_shard_payload(), merge setup_observations and
target_risks in residual-key order, and require their named hashes to match the
contract. For every same-S group compare every setup-observation field except
representative_residual_key_sha256. List fields are compared through
fastkpc_full_cuda_census_named_metadata_hash(); atomic fields are compared
after canonical type normalization. Stop on the first non-invariant field and
report group ID plus field name.

Return a named list containing every loaded table, data, manifest, summary,
dataset hashes, an input_hashes data frame, and a deterministic
phase1_input_bundle_hash built from sorted logical path plus SHA-256 rows.

- [ ] **Step 5: Implement canonical PreparedSKey serialization**

Add:

~~~r
fastkpc_full_cuda_prepared_s_key <- function(
    setup_row, dataset_sha256, R_version, mgcv_version,
    hash_fun = fastkpc_full_cuda_census_hash_utf8) {
  setup_row <- as.data.frame(setup_row, stringsAsFactors = FALSE)
  if (nrow(setup_row) != 1L) {
    stop("PreparedSKey requires one same-S row", call. = FALSE)
  }
  S <- fastkpc_full_cuda_census_parse_s(setup_row$S_key[[1L]])
  fields <- c(
    "schema_version=full-cuda-ci-prepared-s-key-v1",
    paste0("dataset_sha256=", dataset_sha256),
    paste0("same_S_group_id=", setup_row$same_S_group_id[[1L]]),
    paste0("sorted_S=", paste(S, collapse = ",")),
    paste0("formula_class=", setup_row$formula_class[[1L]]),
    "formula_semantics_version=kpcalg_regrXonS_v1",
    "mgcv_semantics_version=mgcv-gam-gcv-cp-v1",
    "family=gaussian",
    "link=identity",
    "method=GCV.Cp",
    "optimizer=mgcv-default",
    "bs=tp",
    "k=mgcv-default",
    "select=false",
    "scale=mgcv-default",
    "na_action=na.fail",
    paste0("R_version=", R_version),
    paste0("mgcv_version=", mgcv_version)
  )
  payload <- paste0(paste(fields, collapse = "\n"), "\n")
  list(payload = payload, sha256 = hash_fun(payload))
}

fastkpc_full_cuda_prepared_s_validate_key_mapping <- function(payload, hash) {
  if (length(payload) != length(hash) || anyNA(payload) || anyNA(hash)) {
    stop("PreparedSKey mapping is incomplete", call. = FALSE)
  }
  by_hash <- split(payload, hash)
  if (any(vapply(by_hash, function(x) length(unique(x)) != 1L,
                 logical(1L)))) {
    stop("PreparedSKey hash collision", call. = FALSE)
  }
  by_payload <- split(hash, payload)
  if (any(vapply(by_payload, function(x) length(unique(x)) != 1L,
                 logical(1L)))) {
    stop("PreparedSKey serialization mismatch", call. = FALSE)
  }
  invisible(TRUE)
}
~~~

- [ ] **Step 6: Run the focused contract test**

~~~bash
Rscript fastkpc/tests/test_full_cuda_ci_prepared_s_contract.R
git diff --check
~~~

Expected: PASS for authenticated inputs and key serialization.

- [ ] **Step 7: Commit Task 1**

~~~bash
git add fastkpc/R/full_cuda_ci_prepared_s_contract.R fastkpc/tests/test_full_cuda_ci_prepared_s_contract.R
git commit -m "feat: authenticate Phase 2 prepared-S inputs"
~~~

---

### Task 2: PreparedSSetup Builder and Validator

**Files:**
- Modify: fastkpc/R/full_cuda_ci_prepared_s_contract.R
- Modify: fastkpc/tests/test_full_cuda_ci_prepared_s_contract.R

- [ ] **Step 1: Add failing response-independence tests**

Append a canonical representative from each penalty count:

~~~r
penalty_counts <- c(1L, 3L, 4L, 5L, 6L, 7L)
representatives <- do.call(rbind, lapply(penalty_counts, function(count) {
  rows <- inputs$same_s_setup_metadata[
    inputs$same_s_setup_metadata$penalty_count == count, , drop = FALSE
  ]
  rows[order(rows$same_S_group_id, method = "radix")[1L], , drop = FALSE]
}))

prepared <- lapply(seq_len(nrow(representatives)), function(index) {
  fastkpc_full_cuda_build_prepared_s_setup(
    inputs = inputs,
    setup_row = representatives[index, , drop = FALSE]
  )
})

for (index in seq_along(prepared)) {
  fastkpc_full_cuda_validate_prepared_s_setup(
    prepared[[index]],
    setup_row = representatives[index, , drop = FALSE],
    dataset_sha256 = inputs$dataset_sha256
  )
}
assert_true(
  identical(vapply(prepared, function(x) length(x$penalty_blocks),
                   integer(1L)), penalty_counts),
  "PreparedSSetup must retain every canonical penalty block"
)

leaked <- prepared[[1L]]
leaked$nested <- list(y = inputs$data[, 1L])
assert_error(
  fastkpc_full_cuda_validate_prepared_s_setup(
    leaked, representatives[1L, , drop = FALSE], inputs$dataset_sha256
  ),
  "response-bearing field",
  "PreparedSSetup must reject nested response leakage"
)

leaked_environment <- prepared[[1L]]
leaked_environment$formula_environment <- new.env(parent = emptyenv())
assert_error(
  fastkpc_full_cuda_validate_prepared_s_setup(
    leaked_environment,
    representatives[1L, , drop = FALSE],
    inputs$dataset_sha256
  ),
  "executable object",
  "PreparedSSetup must reject retained environments"
)
~~~

- [ ] **Step 2: Run the test and verify RED**

Expected failure: PreparedSSetup builders do not exist.

- [ ] **Step 3: Implement zero-response canonical layout**

Add:

~~~r
fastkpc_full_cuda_prepared_s_layout <- function(data, S) {
  frame <- data.frame(
    cbind(rep(0, nrow(data)), data[, S, drop = FALSE]),
    check.names = FALSE
  )
  names(frame) <- paste0("x", seq_len(ncol(frame)))
  formula_fun <- fastkpc_full_cuda_census_formula_function(length(S))
  list(
    data = frame,
    formula = formula_fun(1L, 2L:(1L + length(S)))
  )
}
~~~

- [ ] **Step 4: Implement PreparedSSetup construction**

Build one target-free raw provider setup and whitelist-project it:

~~~r
S <- fastkpc_full_cuda_census_parse_s(setup_row$S_key[[1L]])
layout <- fastkpc_full_cuda_prepared_s_layout(inputs$data, S)
raw <- mgcv::gam(
  formula = layout$formula,
  data = layout$data,
  method = "GCV.Cp",
  fit = FALSE
)
key <- fastkpc_full_cuda_prepared_s_key(
  setup_row, inputs$dataset_sha256,
  inputs$manifest$R_version, inputs$manifest$mgcv_version
)
~~~

Return a list with schema
full-cuda-ci-prepared-s-setup-v1 and every field in the approved design.
Normalize neutral state exactly:

~~~r
constraint <- if (is.null(raw$C)) {
  matrix(numeric(), nrow = 0L, ncol = ncol(raw$X))
} else {
  as.matrix(raw$C)
}
constraint_mode <- if (nrow(constraint) == 0L) "identity" else "explicit"
Z <- if (constraint_mode == "identity") {
  NULL
} else {
  fastkpc_constraint_nullspace(constraint, ncol(raw$X))
}
gram <- crossprod(raw$X)
null_gram <- if (constraint_mode == "identity") {
  NULL
} else {
  crossprod(raw$X %*% Z)
}
~~~

Store raw$S as penalty_blocks, raw$off as penalty_offsets, raw$rank as
mgcv_penalty_rank_metadata, raw$L/lsp0/min.sp as response-independent smoothing
mapping metadata, raw$H only when nonempty, raw$w only when nonunit, and
raw$offset only when nonzero.

Whitelist coefficient labels, intercept, assign, cmX, n/min.edf scoring
constants, and plain smooth descriptors:

~~~r
smooth_descriptors <- lapply(raw$smooth, function(smooth) {
  list(
    class = class(smooth)[[1L]],
    label = as.character(smooth$label),
    term = as.character(smooth$term),
    by = as.character(smooth$by),
    bs_dim = as.integer(smooth$bs.dim),
    first_para = as.integer(smooth$first.para),
    last_para = as.integer(smooth$last.para),
    first_sp = as.integer(smooth$first.sp),
    last_sp = as.integer(smooth$last.sp),
    S_scale = as.numeric(smooth$S.scale),
    shift = as.numeric(smooth$shift),
    p_order = as.numeric(smooth$p.order),
    null_space_dim = as.integer(smooth$null.space.dim),
    rank = as.integer(smooth$rank),
    side_constrain = isTRUE(smooth$side.constrain),
    reparameterized = isTRUE(smooth$repara)
  )
})
~~~

Do not retain raw, raw$y, raw$smooth, formula, formula environment, call,
family closure, zero response, target, or selected sp.

Compute a provider fingerprint over R/mgcv versions, formula-helper and
regrXonS function-body hashes, contrasts, NA action, and extractor schema.
Compute a representation fingerprint including PreparedSKey and exact
plain-data hashes. Compute a semantic fingerprint excluding PreparedSKey,
target, backend identity, and construction response.

- [ ] **Step 5: Implement recursive leakage and lineage validation**

Add recursive forbidden-name and forbidden-type scanners:

~~~r
fastkpc_full_cuda_prepared_s_response_field_names <- function() {
  c("G", "y", "target", "sp", "lsp0", "Xty", "Xty_null",
    "target_fit_fingerprint", "residual_hash", "fitted_hash")
}

fastkpc_full_cuda_prepared_s_find_response_fields <- function(
    value, path = "setup") {
  if (!is.list(value)) return(character())
  fields <- names(value)
  hits <- character()
  if (!is.null(fields)) {
    bad <- fields %in%
      fastkpc_full_cuda_prepared_s_response_field_names()
    hits <- paste0(path, "$", fields[bad])
    for (index in which(!bad)) {
      hits <- c(
        hits,
        fastkpc_full_cuda_prepared_s_find_response_fields(
          value[[index]], paste0(path, "$", fields[[index]])
        )
      )
    }
  }
  unique(hits)
}

fastkpc_full_cuda_prepared_s_find_executable_objects <- function(
    value, path = "setup") {
  if (is.environment(value) || is.function(value) ||
      inherits(value, "formula") || is.call(value) ||
      typeof(value) == "externalptr" || isS4(value) ||
      inherits(value, "gam") ||
      any(grepl("smooth", class(value), fixed = TRUE))) {
    return(path)
  }
  hits <- character()
  if (is.list(value)) {
    fields <- names(value)
    if (is.null(fields)) fields <- as.character(seq_along(value))
    hits <- unlist(lapply(seq_along(value), function(index) {
      fastkpc_full_cuda_prepared_s_find_executable_objects(
        value[[index]], paste0(path, "$", fields[[index]])
      )
    }), use.names = FALSE)
  }
  attrs <- attributes(value)
  if (!is.null(attrs)) {
    attr_names <- names(attrs)
    hits <- c(hits, unlist(lapply(seq_along(attrs), function(index) {
      fastkpc_full_cuda_prepared_s_find_executable_objects(
        attrs[[index]], paste0(path, "@", attr_names[[index]])
      )
    }), use.names = FALSE))
  }
  unique(hits)
}
~~~

Validator gates:

~~~text
schema and PreparedSKey exact
no response-bearing or executable objects, including attributes
X finite with exact provider-representation hash
Phase 1 dimensions/rank exact and model-space diagnostics compatible
penalty count/order/dimensions/ranks/offsets/hashes exact
constraint dimensions/rank/hash exact
H, weights, and offset policies exact
provider, formula-helper, contrasts, NA-policy, and version lineage exact
gram dimensions and finite values exact
representation and semantic fingerprints recompute exactly
Phase 1 fitted-lpmatrix raw hash is retained as lineage, not compared to
the provider-X raw hash
~~~

- [ ] **Step 6: Run the contract test**

~~~bash
Rscript fastkpc/tests/test_full_cuda_ci_prepared_s_contract.R
git diff --check
~~~

Expected: six canonical penalty-count representatives pass and injected
response leakage fails.

- [ ] **Step 7: Commit Task 2**

~~~bash
git add fastkpc/R/full_cuda_ci_prepared_s_contract.R fastkpc/tests/test_full_cuda_ci_prepared_s_contract.R
git commit -m "feat: define response-independent PreparedSSetup"
~~~

---

### Task 3: Batched TargetState Construction

**Files:**
- Modify: fastkpc/R/full_cuda_ci_prepared_s_contract.R
- Modify: fastkpc/tests/test_full_cuda_ci_prepared_s_contract.R

- [ ] **Step 1: Add failing same-S target-state tests**

Use the largest single-penalty group and the largest seven-penalty group. For
each group build all target states and assert:

~~~r
states <- fastkpc_full_cuda_build_target_states(
  inputs = inputs,
  prepared_setup = prepared_setup
)
assert_true(
  nrow(states) ==
    sum(inputs$target_fit_metadata$same_S_group_id ==
          prepared_setup$same_S_group_id),
  "TargetState count must equal canonical same-S target count"
)
assert_true(
  all(states$prepared_s_key_sha256 ==
        prepared_setup$prepared_s_key_sha256),
  "all same-S targets must reference one PreparedSSetup"
)
assert_true(
  all(vapply(states$selected_sp, length, integer(1L)) ==
        length(prepared_setup$penalty_blocks)),
  "selected sp must remain target-specific in penalty order"
)
assert_true(
  all(vapply(states$projected_rhs, length, integer(1L)) ==
        ncol(prepared_setup$X)),
  "TargetState must retain full projected RHS"
)
assert_true(
  !any(c("y", "numeric_y") %in% names(states)),
  "persistent TargetState rows must not duplicate the target vector"
)
~~~

Materialize one row and require its y hash to match. Corrupt the target index
and require materialization to fail.

- [ ] **Step 2: Run the test and verify RED**

Expected failure: TargetState functions do not exist.

- [ ] **Step 3: Implement same-S batch projection**

Resolve all target rows for the group in residual-key radix order. Build Y
from canonical data columns:

~~~r
target_rows <- inputs$target_fit_metadata[
  inputs$target_fit_metadata$same_S_group_id ==
    prepared_setup$same_S_group_id,
  ,
  drop = FALSE
]
target_rows <- target_rows[
  order(target_rows$residual_key_sha256, method = "radix"),
  ,
  drop = FALSE
]
Y <- inputs$data[, target_rows$target, drop = FALSE]
if (is.null(prepared_setup$weights)) {
  projected <- crossprod(prepared_setup$X, Y)
} else {
  projected <- crossprod(
    prepared_setup$X,
    Y * as.numeric(prepared_setup$weights)
  )
}
null_projected <- if (prepared_setup$constraint_mode == "identity") {
  projected
} else {
  crossprod(prepared_setup$constraint_nullspace, projected)
}
~~~

Emit one data-frame row per target with list columns for projected RHS,
nullspace projected RHS, selected sp, selected-sp names, convergence fields,
warning classes, and warning messages.

Set y_source to a list with dataset_sha256 and target_column. Compute y_hash
with fastkpc_full_cuda_census_metadata_hash(as.numeric(Y[, index])).

Compute target_state_fingerprint from the canonical residual key, prepared key,
y hash, RHS hashes, selected-sp hash, GCV/Cp, EDF, convergence/warnings, and
Phase 1 target hashes.

- [ ] **Step 4: Implement TargetState validation and materialization**

Validator gates:

~~~text
exact residual-key and same-S lineage
exact PreparedSKey and Phase 1 setup fingerprint
target index within canonical data
y_source dataset hash exact
y_hash recomputes from canonical data
selected-sp values finite, positive, and penalty-count length
selected-sp names and hash exact to Phase 1
projected RHS lengths and hashes exact
GCV/Cp, EDF, convergence, warnings, and target hashes exact to Phase 1
target-state fingerprint recomputes exactly
~~~

Materializer:

~~~r
fastkpc_full_cuda_materialize_target_state <- function(
    state_row, data, dataset_sha256) {
  if (!identical(state_row$y_source[[1L]]$dataset_sha256,
                 dataset_sha256)) {
    stop("TargetState dataset identity mismatch", call. = FALSE)
  }
  target <- as.integer(state_row$target[[1L]])
  y <- as.numeric(data[, target])
  if (!identical(fastkpc_full_cuda_census_metadata_hash(y),
                 state_row$y_hash[[1L]])) {
    stop("TargetState y hash mismatch", call. = FALSE)
  }
  list(row = state_row, y = y)
}
~~~

- [ ] **Step 5: Run the contract test**

~~~bash
Rscript fastkpc/tests/test_full_cuda_ci_prepared_s_contract.R
git diff --check
~~~

Expected: batched states pass for single- and seven-penalty groups.

- [ ] **Step 6: Commit Task 3**

~~~bash
git add fastkpc/R/full_cuda_ci_prepared_s_contract.R fastkpc/tests/test_full_cuda_ci_prepared_s_contract.R
git commit -m "feat: add canonical Prepared-S target states"
~~~

---

### Task 4: Response-Free Fixed-Sp mgcv Reference Adapter

**Files:**
- Modify: fastkpc/R/full_cuda_ci_prepared_s_contract.R
- Create: fastkpc/tests/test_full_cuda_ci_target_retarget.R

- [ ] **Step 1: Write failing canonical retarget parity tests**

Create fastkpc/tests/test_full_cuda_ci_target_retarget.R. Load authenticated
inputs and select:

~~~text
one finite representative for penalty counts 1, 3, 4, 5, 6, 7
one rank-deficient target
one nonconverged target
one target used by the closest conditional near-alpha test
~~~

Derive every row through metadata filters and radix ordering. Do not hardcode
row numbers.

For each key:

~~~r
prepared <- fastkpc_full_cuda_build_prepared_s_setup(inputs, setup_row)
states <- fastkpc_full_cuda_build_target_states(inputs, prepared)
state <- states[
  states$residual_key_sha256 == selected_key, , drop = FALSE
]
materialized <- fastkpc_full_cuda_materialize_target_state(
  state, inputs$data, inputs$dataset_sha256
)
solved <- fastkpc_mgcv_magic_fixed_sp_from_prepared(
  prepared, materialized
)
assert_true(
  identical(
    fastkpc_full_cuda_census_metadata_hash(solved$coefficients),
    state$coefficient_hash[[1L]]
  ) &&
    identical(
      fastkpc_full_cuda_census_metadata_hash(solved$fitted),
      state$fitted_hash[[1L]]
    ) &&
    identical(
      fastkpc_full_cuda_census_metadata_hash(solved$residuals),
      state$residual_hash[[1L]]
    ),
  "Prepared fixed-sp solve must reproduce Phase 1 hashes exactly"
)
~~~

- [ ] **Step 2: Run the test and verify RED**

Expected failure: fixed-sp Prepared adapter does not exist.

- [ ] **Step 3: Extract an mgcv C_magic core that accepts normalized fields**

Add fastkpc_mgcv_magic_fixed_sp_from_prepared(). Validate schemas and selected
sp, then create only the minimal transient all-fixed mapping:

~~~r
sp <- as.numeric(target_state$row$selected_sp[[1L]])
penalty_count <- length(prepared_setup$penalty_blocks)
minimal <- list(
  G = list(
    L = matrix(numeric(), nrow = penalty_count, ncol = 0L),
    lsp0 = log(sp)
  ),
  X = prepared_setup$X,
  y = target_state$y,
  S = prepared_setup$penalty_blocks,
  off = prepared_setup$penalty_offsets,
  rank = prepared_setup$mgcv_penalty_rank_metadata,
  H = prepared_setup$H,
  C = prepared_setup$constraint,
  w = prepared_setup$weights,
  sp = sp,
  setup_fingerprint = list(
    fingerprint = prepared_setup$prepared_setup_fingerprint
  )
)
beta <- fastkpc_mgcv_magic_kernel_fixed_sp_coefficients(minimal, sp = sp)
fitted <- as.numeric(prepared_setup$X %*% beta)
residuals <- as.numeric(target_state$y - fitted)
~~~

Return explicit identity fields:

~~~r
list(
  backend_family = "mgcvExtractCPU",
  mode = "prepared-s-fixed-sp-mgcv-reference",
  solve_source = "mgcv-C-magic-from-prepared-s",
  authoritative = TRUE,
  coefficients = beta,
  fitted = fitted,
  residuals = residuals,
  sp = sp,
  prepared_s_key_sha256 = prepared_setup$prepared_s_key_sha256,
  residual_key_sha256 =
    target_state$row$residual_key_sha256[[1L]]
)
~~~

Do not retain minimal after the call and do not expose raw G in the result.

- [ ] **Step 4: Add semantic diagnostics**

Implement a comparator that reports:

~~~text
model matrix hash equality
model matrix rank equality
maximum column-space principal angle
constraint rank and projector action
penalty-order equality
coefficient/fitted/residual hash equality
~~~

Use:

~~~r
semantic_angle_tolerance <- function(X) {
  64 * .Machine$double.eps * max(nrow(X), ncol(X))
}
~~~

The angle is diagnostic. Exact coefficient/fitted/residual hashes remain the
canonical gate.

- [ ] **Step 5: Run retarget and existing mgcv tests**

~~~bash
Rscript fastkpc/tests/test_full_cuda_ci_target_retarget.R
Rscript fastkpc/tests/test_mgcv_extract_setup_contract.R
Rscript fastkpc/tests/test_mgcv_extract_same_setup_fixed_sp_batch_cpp.R
git diff --check
~~~

Expected: all canonical retarget cases pass exact hashes; existing prototypes
remain unchanged.

- [ ] **Step 6: Commit Task 4**

~~~bash
git add fastkpc/R/full_cuda_ci_prepared_s_contract.R fastkpc/tests/test_full_cuda_ci_target_retarget.R
git commit -m "feat: add Prepared-S fixed-sp mgcv reference"
~~~

---

### Task 5: Deterministic Iteration/Qualification Selection and dCov Parity

**Files:**
- Modify: fastkpc/R/full_cuda_ci_prepared_s_contract.R
- Create: fastkpc/tests/test_full_cuda_ci_prepared_s_contract_real_subset.R

- [ ] **Step 1: Write the failing selection-count test**

Create fastkpc/tests/test_full_cuda_ci_prepared_s_contract_real_subset.R:

~~~r
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
inputs <- fastkpc_full_cuda_prepared_s_load_inputs(
  "fastkpc/artifacts/full_cuda_ci/workload_census_351x48_v1",
  paste0(
    "fastkpc/artifacts/kpc_tprs_real_zhu/",
    "cancer_RD-causalDiscoveryInput.rds"
  )
)
iteration <- fastkpc_full_cuda_select_prepared_s_iteration_subset(inputs)
assert_true(nrow(iteration$setup_groups) == 44L,
            "iteration setup count")
assert_true(nrow(iteration$target_keys) == 270L,
            "iteration target count")
assert_true(nrow(iteration$logical_tests) == 44L,
            "iteration dCov count")

qualification <-
  fastkpc_full_cuda_select_prepared_s_qualification_subset(inputs)
assert_true(nrow(qualification$seed_target_keys) == 2356L,
            "qualification seed count")
assert_true(nrow(qualification$target_keys) == 6143L,
            "qualification expanded target count")
assert_true(nrow(qualification$logical_tests) == 3808L,
            "qualification logical-test count")
assert_true(nrow(qualification$setup_groups) == 2061L,
            "qualification same-S group count")
assert_true(sum(qualification$logical_tests$near_alpha) == 1478L,
            "all conditional near-alpha tests must be selected")
~~~

Assert penalty-count target distribution exactly
3327, 872, 837, 730, 312, 65 for penalty counts 1, 3, 4, 5, 6, 7.
Assert every rank-deficient, nonfinite-metadata, nonconverged, and
single-penalty high-condition key is present.

- [ ] **Step 2: Run the test and verify RED**

Expected failure: iteration and qualification selectors do not exist.

- [ ] **Step 3: Recompute the complete target-risk table**

Join target metadata to setup penalty count and target risk rows. Treat missing
risk rows as all-false only after reconstructing all risk semantics from
target/setup metadata. Recompute condition buckets with the Phase 1 frozen
boundaries. Verify reconstructed risk rows agree with every filtered
risk_cases.rds target row.

Rank-deficient `Inf` conditions remain explicit nonfinite metadata while
coefficient/fitted/residual outputs must remain finite.

- [ ] **Step 4: Implement the 44/270/44 iteration selector**

Implement the exact class-complete algorithm from the approved design:

~~~text
all six conditional tests within 1e-3 log-alpha distance
closest near-alpha test for each observed S_size by decision
one ordinary lower-median residual pair for every S_size 1..7
one target for every high/rank condition stratum by penalty count
one target for every convergence-signature/S-size/condition stratum
same-S fan-out anchors 2/9/47
same-S logical-load anchors 2/16/3092
consumer closure, endpoint closure, and three-target setup closure
~~~

Record sorted selection reasons and require exact 44 setup groups, 270 target
keys, and 44 logical tests. Hash setup IDs, target keys, logical IDs, and
reason rows into iteration_subset_hash.

- [ ] **Step 5: Implement the extended qualification selector**

Seed all rows satisfying:

~~~r
rare <- target$rank_deficient |
  target$nonfinite_metadata |
  target$mgcv_nonconverged |
  (target$high_condition & target$penalty_count == 1L)
~~~

For each penalty-count by condition-bucket stratum select minimum,
lower-median, maximum condition, and lexical minimum key. For each selected-sp
component select minimum, lower-median, and maximum. For each penalty count and
type-1 multiplicity quantile 0, .25, .5, .75, 1, select the closest group with
same-S ID tie-break and choose lexical minimum, lower-median, and maximum
target.

Store a list-column selection_reasons and collapse duplicate reasons in sorted
order. Require exactly 2,356 seed keys.

Implement logical-test expansion:

Build a two-column long consumer table from residual_key_x and residual_key_y
for conditional tests. Add:

~~~text
all conditional near-alpha logical tests
first canonical consumer for every seed key
first test for each S_size by reference_decision combination
both residual keys for every selected logical test
~~~

Require exact counts 1,478, 3,808, 6,143, and 2,061 and exact penalty-count
distribution. Emit coverage rows for risk classes, condition buckets, penalty
counts, S sizes, reference decisions, and setup multiplicity witnesses.

- [ ] **Step 6: Implement target and dCov parity executors**

Target parity:

~~~r
fastkpc_full_cuda_run_prepared_s_target_parity <- function(
    inputs, prepared_by_group, target_keys) {
  rows <- vector("list", nrow(target_keys))
  residuals <- new.env(hash = TRUE, parent = emptyenv())
  for (index in seq_len(nrow(target_keys))) {
    key <- target_keys$residual_key_sha256[[index]]
    target_row <- inputs$target_fit_metadata[
      match(key, inputs$target_fit_metadata$residual_key_sha256),
      ,
      drop = FALSE
    ]
    prepared <- prepared_by_group[[target_row$same_S_group_id[[1L]]]]
    states <- fastkpc_full_cuda_build_target_states(inputs, prepared)
    state <- states[states$residual_key_sha256 == key, , drop = FALSE]
    solved <- fastkpc_mgcv_magic_fixed_sp_from_prepared(
      prepared,
      fastkpc_full_cuda_materialize_target_state(
        state, inputs$data, inputs$dataset_sha256
      )
    )
    assign(key, solved$residuals, envir = residuals)
    rows[[index]] <- data.frame(
      residual_key_sha256 = key,
      coefficient_hash_exact =
        identical(fastkpc_full_cuda_census_metadata_hash(
          solved$coefficients), state$coefficient_hash[[1L]]),
      fitted_hash_exact =
        identical(fastkpc_full_cuda_census_metadata_hash(
          solved$fitted), state$fitted_hash[[1L]]),
      residual_hash_exact =
        identical(fastkpc_full_cuda_census_metadata_hash(
          solved$residuals), state$residual_hash[[1L]]),
      stringsAsFactors = FALSE
    )
  }
  list(rows = do.call(rbind, rows), residuals = residuals)
}
~~~

Optimize the implementation by caching TargetState tables per same-S group so
each group batch projection runs once.

dCov parity must set and restore
FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK=spectra, call
fastkpc_legacy_dcov_gamma_cpp_oracle(), compare to reference_p_value with
absolute difference at most 1e-12, and require identical p >= alpha decisions.
Record exact-p-value equality separately.

- [ ] **Step 7: Execute the complete iteration gate in the real-subset test**

Build all 44 selected setups, solve all 270 selected targets, and run all 44
dCov pairs. Require exact coefficient/fitted/residual hashes, dCov max drift at
most 1e-12, and zero decision flips. Also validate the extended qualification
counts without solving the full 6,143-key set.

- [ ] **Step 8: Run subset and retarget tests**

~~~bash
Rscript fastkpc/tests/test_full_cuda_ci_prepared_s_contract_real_subset.R
Rscript fastkpc/tests/test_full_cuda_ci_target_retarget.R
git diff --check
~~~

Expected: deterministic iteration and qualification counts pass; the complete
44/270/44 iteration gate has zero hash or decision failures.

- [ ] **Step 9: Commit Task 5**

~~~bash
git add fastkpc/R/full_cuda_ci_prepared_s_contract.R fastkpc/tests/test_full_cuda_ci_prepared_s_contract_real_subset.R
git commit -m "feat: add Prepared-S qualification corpus"
~~~

---

### Task 6: Deterministic Shards and Restart Qualification

**Files:**
- Modify: fastkpc/R/full_cuda_ci_prepared_s_contract.R
- Create: fastkpc/tests/test_full_cuda_ci_prepared_s_contract_restart.R

- [ ] **Step 1: Write failing shard and failure-injection tests**

Create a six-group fixture with hexadecimal PreparedSKeys in scrambled order.
Require radix sorted-rank modulo assignment across three shards. Include:

~~~text
empty trailing shard
duplicate PreparedSKey
wrong Phase 1 input bundle hash
wrong PreparedSKey corpus hash
wrong schema/tolerance/selection config
corrupt payload hash
missing completion JSON
response-bearing y inserted into a setup
interruption after temporary RDS write and before rename
~~~

Run a valid shard twice and assert the second result reports reused with zero
builder calls.

- [ ] **Step 2: Run the restart test and verify RED**

~~~bash
Rscript fastkpc/tests/test_full_cuda_ci_prepared_s_contract_restart.R
~~~

Expected failure: Phase 2 shard helpers do not exist.

- [ ] **Step 3: Implement setup assignment and manifest context**

Create an index with one row per PreparedSKey and assign:

~~~r
order_id <- order(index$prepared_s_key_sha256, method = "radix")
index <- index[order_id, , drop = FALSE]
index$sorted_rank <- seq_len(nrow(index))
index$shard_id <- (index$sorted_rank - 1L) %% shard_count
~~~

Manifest fields:

~~~text
phase1_input_bundle_hash
dataset file and matrix hashes
canonical logical and target key corpus hashes
PreparedSKey corpus hash
source commit
R and mgcv versions
BLAS/LAPACK identity and thread count
PreparedSSetup and TargetState schema versions
semantic tolerance config hash
qualification selection config hash
expected setup count/hash for shard
expected target count/hash for shard
shard count and ID
~~~

- [ ] **Step 4: Implement payload authentication**

Each payload contains manifest, ordered setup keys, PreparedSSetup objects, and
TargetState rows. Hash:

~~~text
manifest hash
setup-key set hash
target-key set hash
per-setup normalized object hashes
TargetState frame hash with semantic nested names
combined payload hash
~~~

Validation reruns every PreparedSSetup and TargetState validator, including the
recursive response-leakage scan.

- [ ] **Step 5: Implement atomic write and resume**

Follow the proven Phase 1 pattern:

~~~r
rds_tmp <- tempfile(".prepared-s-shard-rds-", tmpdir = output_dir)
json_tmp <- tempfile(".prepared-s-shard-json-", tmpdir = output_dir)
saveRDS(payload, rds_tmp, version = 2)
fastkpc_full_cuda_write_json(summary, json_tmp)
validated_payload <- readRDS(rds_tmp)
validated_summary <- jsonlite::read_json(json_tmp, simplifyVector = TRUE)
fastkpc_full_cuda_validate_prepared_s_shard(
  validated_payload, validated_summary, expected_manifest
)
if (!file.rename(rds_tmp, final_rds)) {
  stop("failed to publish Prepared-S shard RDS", call. = FALSE)
}
if (!file.rename(json_tmp, final_json)) {
  unlink(final_rds, force = TRUE)
  stop("failed to publish Prepared-S shard summary", call. = FALSE)
}
~~~

On resume, reuse only when both files exist and full validation passes. A
corrupt or stale pair stops; it is not silently overwritten.

- [ ] **Step 6: Run restart and contract tests**

~~~bash
Rscript fastkpc/tests/test_full_cuda_ci_prepared_s_contract_restart.R
Rscript fastkpc/tests/test_full_cuda_ci_prepared_s_contract.R
git diff --check
~~~

Expected: all interruption, corruption, stale-context, and leakage injections
fail closed; valid resume is byte-identical.

- [ ] **Step 7: Commit Task 6**

~~~bash
git add fastkpc/R/full_cuda_ci_prepared_s_contract.R fastkpc/tests/test_full_cuda_ci_prepared_s_contract_restart.R
git commit -m "feat: add restartable Prepared-S shards"
~~~

---

### Task 7: Standard Runner and Artifact Writer

**Files:**
- Modify: fastkpc/R/full_cuda_ci_prepared_s_contract.R
- Create: fastkpc/tools/run_full_cuda_ci_prepared_s_contract.R
- Modify: fastkpc/tests/test_full_cuda_ci_prepared_s_contract_real_subset.R

- [ ] **Step 1: Add a failing scaled-runner test**

Run the future runner into a temporary directory with:

~~~text
FASTKPC_FULL_CUDA_PREPARED_S_MAX_GROUPS=64
FASTKPC_FULL_CUDA_PREPARED_S_PARITY_SCOPE=iteration
FASTKPC_FULL_CUDA_PREPARED_S_WORKERS=1
FASTKPC_FULL_CUDA_PREPARED_S_SHARD_COUNT=2
FASTKPC_FULL_CUDA_PREPARED_S_RESUME=1
~~~

Require every standard artifact path, pass=TRUE,
run_scope=scaled_iteration, phase2_complete=FALSE, exact selected group/target
lineage, complete 44/270/44 iteration parity, and inherited graph scope. The
64-group selection contains all 44 iteration groups and fills the remainder by
PreparedSKey radix order. Run again and require
zero executed groups, two reused shards, and byte-identical merged RDS files.

- [ ] **Step 2: Run the real-subset test and verify RED**

Expected failure: runner does not exist.

- [ ] **Step 3: Implement artifact paths and writer**

Add fastkpc_full_cuda_prepared_s_artifact_paths() for every file named in the
design. Write:

~~~text
input_hashes.csv
Prepared setup and target indexes as RDS/CSV
iteration setup/target/logical RDS/CSV
iteration_coverage.csv
qualification setup/target/logical RDS/CSV
qualification_coverage.csv
setup_semantic_parity.csv
target_retarget_parity.csv
dcov_parity.csv
unsupported_envelope.csv
fallbacks.csv
stage_timing.csv
commands.txt
environment.txt
manifest.json
summary.json
summary.md
~~~

CSV target rows replace numeric RHS list columns with RHS length and hash
columns. The RDS keeps exact vectors.

Summary phase2_complete is TRUE only when:

~~~r
full_counts <- selected_group_count == 8634L &&
  target_state_count == 110617L
iteration_counts <- iteration_setup_group_count == 44L &&
  iteration_target_key_count == 270L &&
  iteration_logical_test_count == 44L
qualification_counts <- seed_target_key_count == 2356L &&
  qualification_target_key_count == 6143L &&
  qualification_logical_test_count == 3808L &&
  qualification_same_S_group_count == 2061L
phase2_complete <- full_counts &&
  iteration_counts &&
  qualification_counts &&
  all(target_parity$coefficient_hash_exact) &&
  all(target_parity$fitted_hash_exact) &&
  all(target_parity$residual_hash_exact) &&
  max(dcov_parity$absolute_p_diff) <= 1e-12 &&
  sum(!dcov_parity$decision_exact) == 0L &&
  unsupported_count == 0L &&
  fallback_count == 0L
~~~

- [ ] **Step 4: Implement environment parsing and orchestration**

Create run_full_cuda_ci_prepared_s_contract.R with:

~~~text
FASTKPC_FULL_CUDA_PREPARED_S_CENSUS_DIR
FASTKPC_FULL_CUDA_PREPARED_S_DATA_PATH
FASTKPC_FULL_CUDA_PREPARED_S_OUTPUT_DIR
FASTKPC_FULL_CUDA_PREPARED_S_MAX_GROUPS
FASTKPC_FULL_CUDA_PREPARED_S_PARITY_SCOPE=none|iteration|qualification
FASTKPC_FULL_CUDA_PREPARED_S_WORKERS
FASTKPC_FULL_CUDA_PREPARED_S_SHARD_COUNT
FASTKPC_FULL_CUDA_PREPARED_S_RESUME
~~~

Defaults:

~~~text
census_dir    = workload_census_351x48_v1
data_path     = canonical cancer dataset
output_dir    = prepared_s_contract_v1
max_groups    = 0, meaning all groups
parity_scope  = qualification
workers       = 1
shard_count   = 64
resume        = 1
~~~

Use parallel::mclapply only on Unix with requested workers greater than one.
Each worker builds complete group-owned shards. Parent merge validates all
payloads before qualification parity and artifact publication.

On failure write a summary.json with pass=FALSE, phase2_complete=FALSE, stage,
error class/message, and elapsed seconds. Preserve a previously completed
artifact when a pure validation gate fails after publication.

- [ ] **Step 5: Run the scaled runner twice**

~~~bash
Rscript fastkpc/tests/test_full_cuda_ci_prepared_s_contract_real_subset.R
git diff --check
~~~

Expected: first run writes two shards; second run reuses both and merged RDS
hashes are unchanged.

- [ ] **Step 6: Commit Task 7**

~~~bash
git add fastkpc/R/full_cuda_ci_prepared_s_contract.R fastkpc/tools/run_full_cuda_ci_prepared_s_contract.R fastkpc/tests/test_full_cuda_ci_prepared_s_contract_real_subset.R
git commit -m "feat: add full CUDA CI Prepared-S artifact runner"
~~~

---

### Task 8: Focused Verification and Scaled Real Dry Run

**Files:**
- Modify only files needed to fix defects exposed by this task

- [ ] **Step 1: Run all Phase 2 focused tests**

~~~bash
Rscript fastkpc/tests/test_full_cuda_ci_prepared_s_contract.R
Rscript fastkpc/tests/test_full_cuda_ci_target_retarget.R
Rscript fastkpc/tests/test_full_cuda_ci_prepared_s_contract_restart.R
Rscript fastkpc/tests/test_full_cuda_ci_prepared_s_contract_real_subset.R
~~~

Expected: four PASS lines and no skipped canonical test.

- [ ] **Step 2: Run regression tests for reused substrate**

~~~bash
Rscript fastkpc/tests/test_full_cuda_ci_workload_census.R
Rscript fastkpc/tests/test_full_cuda_ci_workload_census_parity.R
Rscript fastkpc/tests/test_full_cuda_ci_workload_census_metadata.R
Rscript fastkpc/tests/test_mgcv_extract_setup_contract.R
Rscript fastkpc/tests/test_mgcv_fixed_sp_cpp_solver.R
Rscript fastkpc/tests/test_legacy_dcov_gamma_cpp_oracle.R
~~~

Expected: all PASS. The Phase 2 work must not change existing prototype or
oracle behavior.

- [ ] **Step 3: Run a 64-group scaled artifact**

~~~bash
output=/tmp/fastkpc_prepared_s_scaled_64
rm -rf "$output"
env FASTKPC_FULL_CUDA_PREPARED_S_OUTPUT_DIR="$output" FASTKPC_FULL_CUDA_PREPARED_S_MAX_GROUPS=64 FASTKPC_FULL_CUDA_PREPARED_S_PARITY_SCOPE=iteration FASTKPC_FULL_CUDA_PREPARED_S_WORKERS=4 FASTKPC_FULL_CUDA_PREPARED_S_SHARD_COUNT=8 FASTKPC_FULL_CUDA_PREPARED_S_RESUME=1 /usr/bin/time -v Rscript fastkpc/tools/run_full_cuda_ci_prepared_s_contract.R
~~~

Require:

~~~text
pass = TRUE
phase2_complete = FALSE
selected group count = 64
response leakage = 0
unsupported = 0
fallback = 0
iteration coefficient/fitted/residual hashes exact
iteration dCov max p drift <= 1e-12
iteration dCov decision flips = 0
~~~

Record wall time, maximum RSS, artifact bytes, setup build throughput, target
projection throughput, and maximum shard size in a local run note.

- [ ] **Step 4: Run deterministic resume**

Repeat the exact command without deleting output. Require:

~~~text
executed group count = 0
reused shard count = 8
merged setup index hash unchanged
merged TargetState index hash unchanged
qualification selection hash unchanged
~~~

- [ ] **Step 5: Fix defects with RED-GREEN cycles**

For each defect, first add a focused regression assertion to the owning Phase
2 test, run it to reproduce failure, apply the smallest implementation fix,
and rerun the owning test plus the scaled artifact. Do not weaken hashes,
counts, dCov tolerance, or failure policy.

- [ ] **Step 6: Run static checks**

~~~bash
git diff --check
git status --short
~~~

Expected: only intended source/test/runner changes and the shared untracked
fastkpc/artifacts symlink.

- [ ] **Step 7: Commit scaled qualification fixes**

If Task 8 required code changes:

~~~bash
git add fastkpc/R/full_cuda_ci_prepared_s_contract.R fastkpc/tools/run_full_cuda_ci_prepared_s_contract.R fastkpc/tests/test_full_cuda_ci_prepared_s_contract.R fastkpc/tests/test_full_cuda_ci_target_retarget.R fastkpc/tests/test_full_cuda_ci_prepared_s_contract_restart.R fastkpc/tests/test_full_cuda_ci_prepared_s_contract_real_subset.R
git commit -m "fix: qualify Prepared-S scaled artifact"
~~~

If no files changed, record the commands and evidence for Task 9 without
creating an empty commit.

---

### Task 9: Complete Artifact, Closure, and Roadmap Update

**Files:**
- Modify: goal-5.6.md
- Modify implementation/tests only for defects proven by the complete run
- Artifact: fastkpc/artifacts/full_cuda_ci/prepared_s_contract_v1/

- [ ] **Step 1: Run the complete restartable Phase 2 artifact**

~~~bash
env FASTKPC_FULL_CUDA_PREPARED_S_WORKERS=20 FASTKPC_FULL_CUDA_PREPARED_S_SHARD_COUNT=64 FASTKPC_FULL_CUDA_PREPARED_S_RESUME=1 /usr/bin/time -v Rscript fastkpc/tools/run_full_cuda_ci_prepared_s_contract.R
~~~

Do not use MAX_GROUPS. Keep PARITY_SCOPE at its default qualification value.
If interrupted, rerun the same command and require valid shards to be reused.

- [ ] **Step 2: Verify the complete artifact independently**

Load summary.json, manifest.json, all 64 shard summaries, setup index,
TargetState index, qualification tables, target parity, dCov parity,
unsupported envelope, and fallbacks.

Require:

~~~text
pass                                      = TRUE
phase2_complete                           = TRUE
prepared_s_setup_count                    = 8,634
target_state_count                        = 110,617
iteration_setup_group_count               = 44
iteration_target_key_count                = 270
iteration_logical_test_count              = 44
seed_target_key_count                     = 2,356
qualification_target_key_count            = 6,143
qualification_logical_test_count          = 3,808
qualification_same_S_group_count          = 2,061
conditional_near_alpha_test_count         = 1,478
response_leakage_count                    = 0
PreparedSKey corpus exact                 = TRUE
target key corpus exact                   = TRUE
setup and target lineage exact            = TRUE
setup fingerprint collision count         = 0
target fingerprint collision count        = 0
fixed-sp coefficient hashes exact         = TRUE
fixed-sp fitted hashes exact              = TRUE
fixed-sp residual hashes exact            = TRUE
dCov maximum absolute p-value drift       <= 1e-12
dCov decision flip count                  = 0
unsupported canonical setup count         = 0
unknown fallback count                    = 0
approximate backend count                 = 0
oracle inherited graph gate               = TRUE
new candidate graph gate                  = NOT_APPLICABLE
~~~

Recompute each artifact file SHA-256, the PreparedSKey corpus hash, target key
corpus hash, named setup/target index hashes, and every shard payload hash.

- [ ] **Step 3: Rerun all Phase 2 tests after the complete artifact**

~~~bash
Rscript fastkpc/tests/test_full_cuda_ci_prepared_s_contract.R
Rscript fastkpc/tests/test_full_cuda_ci_target_retarget.R
Rscript fastkpc/tests/test_full_cuda_ci_prepared_s_contract_restart.R
Rscript fastkpc/tests/test_full_cuda_ci_prepared_s_contract_real_subset.R
git diff --check
~~~

Expected: all PASS against the complete authenticated artifact.

- [ ] **Step 4: Update goal-5.6.md**

Change Task 4 from active to complete and record:

~~~text
PreparedSSetup rows             = 8,634
TargetState rows                = 110,617
iteration setup groups          = 44
iteration target keys           = 270
iteration logical tests         = 44
qualification target keys       = 6,143
qualification logical tests     = 3,808
fixed-sp residual hashes exact  = TRUE
dCov decision flips             = 0
unsupported/fallback count      = 0
artifact path                   =
  fastkpc/artifacts/full_cuda_ci/prepared_s_contract_v1/
~~~

Set the active next task to Phase 3 persistent fixed-sp CUDA resources and
stable multi-target solve. Do not claim Phase 3 CUDA authority in the Phase 2
closure.

- [ ] **Step 5: Commit Phase 2 closure**

~~~bash
git add fastkpc/R/full_cuda_ci_prepared_s_contract.R fastkpc/tools/run_full_cuda_ci_prepared_s_contract.R fastkpc/tests/test_full_cuda_ci_prepared_s_contract.R fastkpc/tests/test_full_cuda_ci_target_retarget.R fastkpc/tests/test_full_cuda_ci_prepared_s_contract_restart.R fastkpc/tests/test_full_cuda_ci_prepared_s_contract_real_subset.R goal-5.6.md
git commit -m "docs: record full CUDA CI Phase 2 closure"
~~~

- [ ] **Step 6: Push through the required proxy**

~~~bash
git -c http.version=HTTP/1.1 -c http.proxy=http://localhost:7890 push --verbose origin HEAD:refs/heads/main
~~~

- [ ] **Step 7: Independent final review**

Request a review focused on:

~~~text
response leakage
Phase 1 input authentication
PreparedSKey and TargetState lineage
multi-penalty order
rank-deficient fixed-sp parity
qualification subset completeness
dCov tolerance and decision gates
restart corruption handling
artifact closure claims
~~~

Address every confirmed issue with a failing regression test, focused fix,
verification, commit, and proxy push.

## Plan Completion Gate

This plan is complete only when the Phase 2 artifact proves all 8,634 setup
objects and all 110,617 target states, the 6,143-key qualification corpus
passes exact fixed-sp hashes and zero dCov decision flips, all restart and
failure-injection tests pass, goal-5.6.md records Phase 2 closure, and remote
main contains the verified implementation.
