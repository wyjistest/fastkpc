# Full-CUDA CI Phase 1 Workload Census Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete, restartable 351x48 logical-CI and legacy-mgcv workload/risk census required by Phase 1 of `goal-5.6.md`.

**Architecture:** Derive the canonical logical-test and conditional `target|S` corpus offline from the immutable Phase 0 oracle. Prove one-target census fitting is bit-exact with the real two-target `regrXonS` layout, fit each of the 110,617 canonical keys once through authoritative `mgcv::gam`, split response-independent same-S setup metadata from target-specific fit metadata, and merge deterministic restartable shards into a fail-closed artifact. Phase 1 never executes a new skeleton; graph evidence is explicitly inherited from Phase 0.

**Tech Stack:** R 4.4.1, mgcv 1.9.1, kpcalg, digest, jsonlite, base `parallel`, Phase 0 full-CUDA CI oracle artifacts.

---

## Scope and Ordering

Implement exactly six independently reviewable stages:

```text
1. Structural census, no mgcv fits
2. Legacy regrXonS layout parity subset
3. Setup/fit metadata split and risk schema
4. Deterministic shards and restart qualification
5. Scaled dry runs and standard runner
6. Full 110,617-key census and Phase 1 closure
```

Do not start Stage 6 until Stages 1-5 pass. Do not add production caching,
CUDA solves, fixed-sp fitting, scheduler instrumentation, or a new skeleton
comparison in this plan.

## File Map

```text
Create  fastkpc/R/full_cuda_ci_workload_census.R
Create  fastkpc/R/full_cuda_ci_oracle_contract.R
Create  fastkpc/tools/run_full_cuda_ci_workload_census.R
Create  fastkpc/tests/test_full_cuda_ci_workload_census.R
Create  fastkpc/tests/test_full_cuda_ci_workload_census_parity.R
Create  fastkpc/tests/test_full_cuda_ci_workload_census_metadata.R
Create  fastkpc/tests/test_full_cuda_ci_workload_census_restart.R
Create  fastkpc/tests/test_full_cuda_ci_workload_census_real_subset.R
Modify  goal-5.6.md only after the full artifact passes
Artifact fastkpc/artifacts/full_cuda_ci/workload_census_351x48_v1/
```

`full_cuda_ci_oracle_contract.R` owns the frozen Phase 0 input contract and
fail-closed loader shared by the Phase 0 validator and Phase 1 census.
`full_cuda_ci_workload_census.R` owns structural tables, canonical
serialization, parity helpers, metadata extraction, risk classification,
sharding, merge validation, and artifact writing. Public helpers remain
prefixed by `fastkpc_full_cuda_census_`.

---

### Task 1: Structural Census and Canonical Key Corpus

**Files:**
- Create: `fastkpc/R/full_cuda_ci_oracle_contract.R`
- Create: `fastkpc/R/full_cuda_ci_workload_census.R`
- Create: `fastkpc/tests/test_full_cuda_ci_workload_census.R`

- [ ] **Step 1: Write the failing structural test**

Create `fastkpc/tests/test_full_cuda_ci_workload_census.R` with base-R
assertions. The first fixture must contain level 0 and conditional rows,
explicit source lineage, a p-value exactly at alpha, and two deleting rows:

```r
fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_error <- function(expression, pattern, message) {
  error <- tryCatch({ force(expression); NULL }, error = identity)
  assert_true(inherits(error, "error") &&
                grepl(pattern, conditionMessage(error)), message)
}

source("fastkpc/R/full_cuda_ci_workload_census.R")

trace <- data.frame(
  logical_sequence_id = 1:5,
  source_sequence_id = c(11L, 12L, 13L, 14L, 15L),
  source_task_index = c(1L, 4L, 9L, 15L, 22L),
  level = c(0L, 1L, 2L, 3L, 1L),
  x = c(1L, 1L, 2L, 3L, 4L),
  y = c(2L, 3L, 4L, 5L, 5L),
  S_key = c("", "4", "1|5", "1|2|6", "2"),
  p_value = c(0.02, 0.1, 0.08, 0.20, 0),
  deletes_edge = c(FALSE, TRUE, FALSE, TRUE, FALSE),
  stringsAsFactors = FALSE
)
deletions <- data.frame(
  level = c(1L, 3L),
  edge_x = c(3L, 3L),
  edge_y = c(1L, 5L),
  S_key = c("4", "1|2|6"),
  stringsAsFactors = FALSE
)

rows <- fastkpc_full_cuda_census_logical_tests(
  trace = trace,
  deletions = deletions,
  alpha = 0.1,
  data_hash = paste(rep("a", 64L), collapse = ""),
  n = 20L,
  p = 6L
)

assert_true(identical(rows$source_sequence_id, trace$source_sequence_id),
            "logical census must retain source_sequence_id")
assert_true(identical(rows$formula_class,
                      c("direct-ci", "full-smooth", "full-smooth",
                        "additive-smooth", "full-smooth")),
            "logical census must reuse the compatibility formula enum")
assert_true(all(is.na(rows[rows$level == 0L, c("residual_key_x",
                                               "residual_key_y")])),
            "level 0 must not create GAM residual keys")
assert_true(identical(rows$reference_independent,
                      trace$deletes_edge),
            "reference decisions must be explicit")
assert_true(identical(rows$selected_sepset,
                      trace$deletes_edge),
            "selected sepsets must follow deleting logical rows")
assert_true(rows$signed_distance_from_alpha[[2L]] == 0,
            "signed alpha distance must retain the boundary")
assert_true(rows$reference_decision[[2L]] == "independent",
            "legacy replay uses p >= alpha")

requests <- fastkpc_full_cuda_census_residual_requests(rows)
assert_true(sum(requests$request_multiplicity) == 8L,
            "four conditional tests must expand to eight requests")
assert_true(!anyDuplicated(requests$residual_key_payload),
            "canonical residual payloads must be unique")
assert_true(!anyDuplicated(requests$residual_key_sha256),
            "canonical residual hashes must be unique")
```

Add fail-closed checks for a mismatched deletion tuple, an unsorted S key, a
non-finite p-value, a forced hash collision through an injected hash function,
and one payload with CRLF instead of LF.

- [ ] **Step 2: Run the structural test and verify RED**

Run:

```bash
Rscript fastkpc/tests/test_full_cuda_ci_workload_census.R
```

Expected: failure because
`fastkpc/R/full_cuda_ci_workload_census.R` does not exist.

- [ ] **Step 3: Implement the frozen input and hashing contracts**

Create `fastkpc/R/full_cuda_ci_workload_census.R`. Source the existing Phase 0
and mgcv compatibility helpers, then add these functions:

```r
source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/mgcv_compat_contract.R")

fastkpc_full_cuda_census_hash_raw <- function(raw) {
  fastkpc_full_cuda_require_namespace("digest")
  digest::digest(raw, algo = "sha256", serialize = FALSE)
}

fastkpc_full_cuda_census_hash_utf8 <- function(value) {
  fastkpc_full_cuda_census_hash_raw(charToRaw(enc2utf8(value)))
}

fastkpc_full_cuda_census_file_hash <- function(path) {
  fastkpc_full_cuda_require_namespace("digest")
  digest::digest(file = path, algo = "sha256", serialize = FALSE)
}

fastkpc_full_cuda_census_input_contract <- function() {
  list(
    schema_version = "full-cuda-ci-phase1-input-v1",
    phase0_source_commit =
      "93ae8430aa24ef4458f6ae62451982fb04bab804",
    dataset_matrix_sha256 =
      "971f1e0784817c6febfc84154f19540be02a92f6e4acad784dc7f92b979f1df7",
    oracle_input_bundle_sha256 =
      "7700bc78240984c36f8ae5ca281362a0afb8d7dedd34a5711ce4ab76a2ebee0e",
    canonical_key_corpus_hash =
      "b843630969f116da63f7fad095c54de2ff471540159ff97ca56c3871d6b2e1fa",
    file_hashes = c(
      "dataset/cancer_RD-causalDiscoveryInput.rds" =
        "e03cbfafed3336f3da7725878e92af24556b5ebdc6227b94ea84469e62e94036",
      "oracle/adjacency.rds" =
        "6701a033e821f8433842ae825f67715b6c2349e3c36515b45296f125ff7e1d4e",
      "oracle/deletion_trace.csv" =
        "00eeb3fe843e9f868b133cef9f573430ea9d49fe7b0dd53ba3be51e2e9e94486",
      "oracle/fallbacks.csv" =
        "dc45430a89ad1c4fb85cf9f4c63b4babe89df8cf2b505759e7389f7b5462beca",
      "oracle/first_divergence.json" =
        "b29373f20a56d99ce76ef4c21f2d508a13ed22ed009a30c902a1e6d23e46fef6",
      "oracle/graph_agreement.csv" =
        "f1eabdbc578607ee0b124a6dcd3fd8250077493472605583889eb532cd3b85d5",
      "oracle/logical_ci_trace.rds" =
        "b777c5dc1b9acad08c133ccd668eae5e9444a89d481c7f8adf9b2c2a0dd6cda5",
      "oracle/manifest.json" =
        "f907559586c4b766f483bdc01b4074d93ce2c8b80972c3199c4493848b2b8750",
      "oracle/n_edgetests.csv" =
        "6c0e1ccb14c9721e7056aa91e851065877965ab33869867dd130c7ca3d503058",
      "oracle/pmax.rds" =
        "2fafe1f5084dcb86114adfb86d06855350d36872e59795b6ee604bf4c6e19df5",
      "oracle/sepset_agreement.csv" =
        "9e57978d03fa0e62526b884e0566d2671d2c74417f117fd420b0fb4afb85a256",
      "oracle/sepsets.rds" =
        "69853449f95e1486ef237a2b1bd7c3a99d94cac4c0f202d7c509c890a49e1ca6",
      "oracle/summary.json" =
        "eec6724d9fd69671399783b565c2dd8bbdbc3a4e553ba742ac781d483354ade7"
    )
  )
}

fastkpc_full_cuda_census_bundle_payload <- function(file_hashes) {
  file_hashes <- file_hashes[order(names(file_hashes), method = "radix")]
  paste0(paste0(names(file_hashes), "\t", unname(file_hashes),
                collapse = "\n"), "\n")
}
```

Implement `fastkpc_full_cuda_census_validate_input_hashes()` so it maps the
oracle directory and data path to the logical names above, compares every file
hash, recomputes the bundle hash, and returns a data frame for
`oracle_input_hashes.csv`. Missing, extra, or mismatched files must stop before
RDS/JSON parsing.

- [ ] **Step 4: Implement the unified Phase 0 loader**

Add `fastkpc_full_cuda_census_load_inputs(oracle_dir, data_path, contract)`.
It must:

```r
hashes <- fastkpc_full_cuda_census_validate_input_hashes(
  oracle_dir, data_path, contract
)
oracle <- fastkpc_load_full_cuda_ci_oracle(oracle_dir)
data <- readRDS(data_path)
canonical <- fastkpc_full_cuda_canonical_contract()
stopifnot(
  identical(fastkpc_full_cuda_data_hash(data), canonical$data_hash),
  identical(digest::digest(oracle$reference$adjacency,
                           algo = "sha256", serialize = TRUE),
            canonical$adjacency_hash),
  identical(digest::digest(
    fastkpc_full_cuda_normalize_sepsets(oracle$reference,
                                        canonical$column_order),
    algo = "sha256", serialize = TRUE
  ), canonical$sepset_hash),
  identical(as.integer(oracle$reference$n.edgetests),
            as.integer(canonical$n_edgetests))
)
deletion_trace <- utils::read.csv(
  file.path(oracle_dir, "deletion_trace.csv"), stringsAsFactors = FALSE
)
deletion_trace$p_value <- as.numeric(deletion_trace$p_value)
stopifnot(identical(
  digest::digest(deletion_trace, algo = "sha256", serialize = TRUE),
  canonical$deletion_trace_hash
))
```

Then independently read and validate `graph_agreement.csv`,
`sepset_agreement.csv`, `n_edgetests.csv`, `fallbacks.csv`, `summary.json`, and
`first_divergence.json`. Require SHD zero, all inherited agreement flags true,
all fallback/error counts zero, the manifest source commit equal to the pinned
commit, and no first divergence. Return:

```r
list(
  data = as.matrix(data),
  oracle = oracle,
  oracle_input_hashes = hashes,
  oracle_input_bundle_sha256 = contract$oracle_input_bundle_sha256,
  oracle_inherited_graph_gate = TRUE,
  new_candidate_graph_gate = "NOT_APPLICABLE"
)
```

- [ ] **Step 5: Implement logical-row normalization and decision validation**

Add exact S parsing and decision helpers:

```r
fastkpc_full_cuda_census_parse_s <- function(S_key) {
  if (is.na(S_key) || !nzchar(S_key)) return(integer())
  values <- as.integer(strsplit(S_key, "|", fixed = TRUE)[[1L]])
  if (anyNA(values) || !identical(values, sort(unique(values)))) {
    stop("conditioning set key is not canonical sorted unique", call. = FALSE)
  }
  values
}

fastkpc_full_cuda_census_reference_decision <- function(p, alpha, deleted) {
  if (!is.finite(p)) stop("canonical reference p-value is non-finite",
                          call. = FALSE)
  independent <- p >= alpha
  if (!identical(independent, isTRUE(deleted))) {
    stop("reference alpha decision disagrees with deletes_edge",
         call. = FALSE)
  }
  if (independent) "independent" else "dependent"
}
```

Implement `fastkpc_full_cuda_census_logical_tests()` with every Stage A field
from the approved design. Set `selected_sepset = deletes_edge`, then prove a
one-to-one match between deleting rows and deletion rows by
`level|unordered edge|S_key`. Level 0 receives `direct-ci` and `NA_character_`
residual keys. Use `.Machine$double.xmin` for the signed log ratio.

- [ ] **Step 6: Implement residual and same-S payloads**

Add payload builders that emit the exact LF-terminated fields from the design:

```r
fastkpc_full_cuda_census_residual_payload <- function(
    target, S, formula_class, data_hash, n, p) {
  fields <- c(
    "schema_version=full-cuda-ci-residual-key-v1",
    paste0("dataset_sha256=", data_hash),
    paste0("n=", as.integer(n)),
    paste0("p=", as.integer(p)),
    paste0("target_index=", as.integer(target)),
    paste0("sorted_S=", paste(as.integer(S), collapse = ",")),
    paste0("formula_class=", formula_class),
    "family=gaussian",
    "link=identity",
    "method=GCV.Cp",
    "optimizer=mgcv-default",
    "gamma=1",
    "select=false",
    "scale=mgcv-default",
    "weights=none",
    "offset=none",
    "formula_semantics_version=kpcalg_regrXonS_v1",
    "mgcv_semantics_version=legacy-mgcv-gam-default-selection-v1"
  )
  paste0(paste(fields, collapse = "\n"), "\n")
}
```

Implement the same-S payload without `target_index`. Hash raw UTF-8 bytes with
SHA-256. Keep payload and hash columns. Inject `hash_fun` in tests so collision
handling can be tested without finding a real SHA-256 collision.

- [ ] **Step 7: Implement expansion, deduplication, and canonical gates**

`fastkpc_full_cuda_census_residual_requests(logical_tests)` must:

1. select only `S_size > 0` rows;
2. expand x and y requests;
3. attach payload, SHA-256, and same-S SHA-256;
4. aggregate multiplicity and first/last logical IDs and levels;
5. compute same-S group size as unique target count;
6. sort final rows by `residual_key_sha256` using radix order;
7. detect payload/hash inconsistencies both directions;
8. compute corpus hash from sorted residual hashes plus final LF.

Add `fastkpc_full_cuda_census_validate_structural(structural, canonical=TRUE)`
with these hard gates:

```r
stopifnot(
  nrow(structural$logical_tests) == 240489L,
  sum(structural$logical_tests$S_size > 0L) == 238276L,
  sum(structural$residual_requests$request_multiplicity) == 476552L,
  nrow(structural$residual_requests) == 110617L,
  length(unique(structural$residual_requests$same_S_group_id)) == 8634L,
  identical(structural$canonical_key_corpus_hash,
            "b843630969f116da63f7fad095c54de2ff471540159ff97ca56c3871d6b2e1fa")
)
```

Keep `273284` only in a nested historical metric with
`metric_role = "historical_route_metric"`, `hard_gate = FALSE`, and
`provenance_status = "no_hash_protected_route_trace"`.

- [ ] **Step 8: Run focused and canonical structural tests**

Add a canonical section to the same test:

```r
inputs <- fastkpc_full_cuda_census_load_inputs(
  "fastkpc/artifacts/full_cuda_ci/oracle_351x48_v1",
  "fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"
)
structural <- fastkpc_full_cuda_census_structural(inputs)
fastkpc_full_cuda_census_validate_structural(structural, canonical = TRUE)
```

Run:

```bash
Rscript fastkpc/tests/test_full_cuda_ci_workload_census.R
git diff --check
```

Expected: `PASS full CUDA CI structural census`, including the canonical
110,617-key corpus hash.

- [ ] **Step 9: Commit Stage 1**

```bash
git add fastkpc/R/full_cuda_ci_oracle_contract.R \
        fastkpc/R/full_cuda_ci_workload_census.R \
        fastkpc/tests/test_full_cuda_ci_workload_census.R
git commit -m "feat: add canonical full CUDA CI structural census"
```

---

### Task 2: Legacy `regrXonS` Layout Parity Subset

**Files:**
- Modify: `fastkpc/R/full_cuda_ci_workload_census.R`
- Create: `fastkpc/tests/test_full_cuda_ci_workload_census_parity.R`

- [ ] **Step 1: Write the failing exact-parity test**

Create the parity test and load the canonical structural corpus. Select:

```r
conditional <- structural$logical_tests[
  structural$logical_tests$S_size > 0L, , drop = FALSE
]
first_for_size <- function(size) {
  conditional$logical_sequence_id[
    which(conditional$S_size == as.integer(size))[[1L]]
  ]
}
case_ids <- unique(c(
  first_for_size(1L),
  first_for_size(2L),
  first_for_size(3L),
  conditional$logical_sequence_id[
    which(conditional$S_size == max(conditional$S_size))[[1L]]
  ],
  conditional$logical_sequence_id[
    which.min(conditional$absolute_log_distance_from_alpha)
  ]
))
cases <- conditional[
  match(case_ids, conditional$logical_sequence_id), , drop = FALSE
]
```

Implement this selection with deterministic `which()` calls, not literal row
numbers. Add two deterministic synthetic cases with `set.seed(20260710)`:

```r
z <- stats::rnorm(351L)
rank_deficient <- list(
  X = cbind(z + stats::rnorm(351L) * 0.2,
            -z + stats::rnorm(351L) * 0.2),
  S = cbind(z, 2 * z)
)
near_constant <- list(
  X = cbind(stats::rnorm(351L), stats::rnorm(351L)),
  S = cbind(1 + stats::rnorm(351L) * 1e-10)
)
```

For every case assert exact equality for residual/fitted hashes, ordered
numeric selected sp, GCV/Cp, EDF, dCov p-value, and `p >= alpha` decision.

- [ ] **Step 2: Run the parity test and verify RED**

```bash
Rscript fastkpc/tests/test_full_cuda_ci_workload_census_parity.R
```

Expected: failure because parity helpers do not exist.

- [ ] **Step 3: Implement exact legacy pair-layout fitting**

Add:

```r
fastkpc_full_cuda_census_pair_layout_fits <- function(X, S_data) {
  X <- as.matrix(X)
  S_data <- as.matrix(S_data)
  data <- data.frame(cbind(X, S_data))
  names(data) <- paste0("x", seq_len(ncol(data)))
  formula_fun <- if (ncol(S_data) > 2L) {
    kpcalg:::frml.additive.smooth
  } else {
    kpcalg:::frml.full.smooth
  }
  predictors <- (ncol(X) + 1L):(ncol(X) + ncol(S_data))
  lapply(seq_len(ncol(X)), function(i) {
    mgcv::gam(formula_fun(i, predictors), data = data)
  })
}
```

Call the real `kpcalg:::regrXonS(X, S_data)` separately so the test covers the
actual function, not only a reproduction.

- [ ] **Step 4: Implement the offline one-target authority**

Add:

```r
fastkpc_full_cuda_census_single_target_fit <- function(y, S_data) {
  S_data <- as.matrix(S_data)
  data <- data.frame(cbind(as.numeric(y), S_data))
  names(data) <- paste0("x", seq_len(ncol(data)))
  formula_fun <- if (ncol(S_data) > 2L) {
    kpcalg:::frml.additive.smooth
  } else {
    kpcalg:::frml.full.smooth
  }
  formula <- formula_fun(1L, 2L:(1L + ncol(S_data)))
  mgcv::gam(formula, data = data)
}
```

Capture warnings with `withCallingHandlers`, preserve class/message order, and
muffle after capture. Return the fit plus warnings and elapsed milliseconds.

- [ ] **Step 5: Implement canonical numeric hashing and comparison**

Add `fastkpc_full_cuda_census_metadata_hash()` using normalized portable R
serialization version 2 and SHA-256. For vector parity, remove names and hash
the exact double vector. Compare selected sp numerically in mgcv penalty order
and preserve names in a separate field.

Add `fastkpc_full_cuda_census_parity_case()` that computes pair-layout fits,
single-target fits, real `regrXonS` residuals, and `kpcalg:::dcov.gamma` with
`index = 1`, `numCol = 35`. Return one result row per case.

- [ ] **Step 6: Write parity artifact helpers**

Implement `fastkpc_full_cuda_census_parity_cases()` and
`fastkpc_full_cuda_census_write_parity()` to produce:

```text
legacy_layout_parity_cases.rds
legacy_layout_parity_results.csv
```

The result must contain exact-equality booleans and absolute differences. Any
false equality fails before metadata shards can run.

- [ ] **Step 7: Run parity tests**

```bash
Rscript fastkpc/tests/test_full_cuda_ci_workload_census.R
Rscript fastkpc/tests/test_full_cuda_ci_workload_census_parity.R
git diff --check
```

Expected: `PASS full CUDA CI legacy layout parity` with canonical
`|S|=1,2,3,7`, nearest-alpha, rank-deficient, and near-constant cases exact.

- [ ] **Step 8: Commit Stage 2**

```bash
git add fastkpc/R/full_cuda_ci_workload_census.R \
        fastkpc/tests/test_full_cuda_ci_workload_census_parity.R
git commit -m "test: qualify legacy layout for full CUDA CI census"
```

---

### Task 3: Setup/Fit Schema and Numerical Risk Census

**Files:**
- Modify: `fastkpc/R/full_cuda_ci_workload_census.R`
- Create: `fastkpc/tests/test_full_cuda_ci_workload_census_metadata.R`

- [ ] **Step 1: Write failing metadata-schema tests**

Use a small canonical subset containing two targets from each of one
`|S|=1`, `2`, `3`, and maximum-size S group. Fit each key and assert:

```r
required_setup_fields <- c(
  "same_S_group_id", "S_key", "S_size", "formula_class",
  "model_matrix_hash", "model_matrix_rank", "model_matrix_condition",
  "penalty_count", "penalty_hashes", "penalty_nullity",
  "constraint_rank", "constraint_nullspace_dimension",
  "constraint_hash", "H_hash", "weights_policy", "offset_policy",
  "setup_fingerprint"
)
required_target_fields <- c(
  "residual_key_sha256", "same_S_group_id", "setup_fingerprint",
  "target", "fit_status", "fit_time_ms", "selected_sp",
  "GCV_Cp_score", "EDF", "convergence_fields", "warning_classes",
  "penalized_system_condition_at_selected_sp", "target_sd",
  "coefficient_hash", "fitted_hash", "residual_hash",
  "target_fit_fingerprint"
)
```

Deliberately alter one setup observation's model-matrix hash and assert that
compression fails with `same-S setup invariant violation`.

Add direct tests for empty, all-zero, rank-deficient, finite high-condition,
and non-finite matrices under the frozen SVD rules. Add boundary tests for all
near-alpha buckets and vectors of length 0, 1, constant, near-constant, and
ordinary variance.

- [ ] **Step 2: Run the metadata test and verify RED**

```bash
Rscript fastkpc/tests/test_full_cuda_ci_workload_census_metadata.R
```

Expected: failure because metadata/risk functions are absent.

- [ ] **Step 3: Implement portable metadata hashes and setup fingerprint**

Add normalized object hashing:

```r
fastkpc_full_cuda_census_metadata_hash <- function(value) {
  normalize <- function(x) {
    if (is.numeric(x)) storage.mode(x) <- "double"
    names(x) <- NULL
    if (!is.null(dim(x))) dimnames(x) <- NULL
    if (is.list(x)) x <- lapply(x, normalize)
    x
  }
  raw <- serialize(normalize(value), NULL, version = 2)
  fastkpc_full_cuda_census_hash_raw(raw)
}
```

Implement the exact `full-cuda-ci-same-s-setup-fingerprint-v1` payload from
the design. Use `canonical_input_p = 48`; never use the local fit-frame column
count. Exclude target and response fields.

- [ ] **Step 4: Implement frozen rank, condition, nullspace, and risk rules**

Add:

```r
fastkpc_full_cuda_census_svd_diagnostics <- function(A,
    expected_rank = min(dim(as.matrix(A)))) {
  A <- as.matrix(A)
  storage.mode(A) <- "double"
  dimnames(A) <- NULL
  if (any(dim(A) == 0L)) {
    return(list(rank = 0L, condition = NA_real_,
                bucket = "not_applicable_empty", tolerance = NA_real_))
  }
  if (any(!is.finite(A))) {
    return(list(rank = NA_integer_, condition = NA_real_,
                bucket = "nonfinite_unknown", tolerance = NA_real_))
  }
  singular <- La.svd(A, nu = 0L, nv = 0L)$d
  if (any(!is.finite(singular))) {
    return(list(rank = NA_integer_, condition = NA_real_,
                bucket = "nonfinite_unknown", tolerance = NA_real_))
  }
  smax <- max(singular)
  if (identical(smax, 0)) {
    return(list(rank = 0L, condition = Inf,
                bucket = "rank_deficient_inf", tolerance = 0))
  }
  tolerance <- max(dim(A)) * smax * .Machine$double.eps
  rank <- sum(singular > tolerance)
  condition <- if (rank < expected_rank) Inf else smax / min(singular)
  list(rank = as.integer(rank), condition = condition,
       bucket = fastkpc_full_cuda_census_condition_bucket(condition, rank,
                                                          expected_rank),
       tolerance = tolerance)
}
```

Add the exact bucket and scalar-risk helpers:

```r
fastkpc_full_cuda_census_condition_bucket <- function(condition, rank,
                                                       expected_rank) {
  if (is.na(condition) && identical(rank, 0L) && expected_rank == 0L) {
    return("not_applicable_empty")
  }
  if (is.na(condition) || is.na(rank)) return("nonfinite_unknown")
  if (rank < expected_rank || is.infinite(condition)) {
    return("rank_deficient_inf")
  }
  if (condition < 1e4) return("finite_lt_1e4")
  if (condition < 1e8) return("finite_1e4_to_lt_1e8")
  if (condition < 1e12) return("finite_1e8_to_lt_1e12")
  "finite_ge_1e12"
}

fastkpc_full_cuda_census_near_constant <- function(value) {
  value <- as.double(value)
  sd_value <- if (length(value) < 2L || any(!is.finite(value))) {
    NA_real_
  } else {
    sqrt(sum((value - mean(value))^2) / (length(value) - 1L))
  }
  list(
    sd = sd_value,
    near_constant = !is.finite(sd_value) ||
      sd_value <= sqrt(.Machine$double.eps)
  )
}

fastkpc_full_cuda_census_near_alpha_bucket <- function(distance) {
  if (!is.finite(distance)) return("nonfinite_unknown")
  if (distance == 0) return("exact_boundary")
  limits <- c(1e-12, 1e-9, 1e-6, 1e-3,
              log(1.01), log(1.1), log(2))
  labels <- c("le_1e_minus_12", "le_1e_minus_9", "le_1e_minus_6",
              "le_1e_minus_3", "le_log_1_01", "le_log_1_1",
              "le_log_2")
  index <- which(distance <= limits)[1L]
  if (is.na(index)) "farther" else labels[[index]]
}
```

Implement deterministic SVD nullspace and field-level non-finite
classification using these helpers and the approved design formulas.

- [ ] **Step 5: Extract setup observations from the authoritative fit**

From each single-target `mgcv::gam` fit:

1. obtain exact model matrix with `predict(fit, type = "lpmatrix")`;
2. extract each smooth penalty block, rank, `first.para`, `last.para`,
   `first.sp`, and `last.sp` in mgcv order;
3. embed penalty blocks into coefficient space;
4. record absent constraints as a zero-row matrix and absent H as NONE;
5. record prior weights and offset policies;
6. calculate model, conditioning, penalty, constraint, and setup hashes;
7. calculate the response-independent setup fingerprint.

Expose this as
`fastkpc_full_cuda_census_setup_observation(fit, request_row, data)`.

- [ ] **Step 6: Extract target-specific fit metadata**

Implement `fastkpc_full_cuda_census_fit_key(data, request_row, risk_config)`.
It calls the qualified single-target helper once and returns:

```r
list(
  setup_observation = setup_row,
  target_fit = target_row,
  risk_cases = risk_rows
)
```

Build `P_unit`, `P(selected_sp)`, and
`A(selected_sp) = Z' (X' W X + P(selected_sp) + H) Z` exactly as specified.
Record raw convergence slots with provenance rather than inventing a combined
value. Keep error rows with `fit_status = "error"`.

- [ ] **Step 7: Implement same-S compression and field coverage**

`fastkpc_full_cuda_census_compress_setups(observations)` must group by
`same_S_group_id`, require exactly one model-matrix hash, penalty-hash vector,
constraint hash, and setup fingerprint, then emit one row per group.

`fastkpc_full_cuda_census_field_coverage()` must report field, table, total,
present, finite where applicable, required, and ratio. Required coverage below
100% fails the final merge.

`fastkpc_full_cuda_census_risk_cases()` must emit one row per flagged target
key or logical test with all approved boolean flags.

- [ ] **Step 8: Run metadata tests**

```bash
Rscript fastkpc/tests/test_full_cuda_ci_workload_census.R
Rscript fastkpc/tests/test_full_cuda_ci_workload_census_parity.R
Rscript fastkpc/tests/test_full_cuda_ci_workload_census_metadata.R
git diff --check
```

Expected: all PASS, including deliberate invariant and non-finite failures.

- [ ] **Step 9: Commit Stage 3**

```bash
git add fastkpc/R/full_cuda_ci_workload_census.R \
        fastkpc/tests/test_full_cuda_ci_workload_census_metadata.R
git commit -m "feat: add full CUDA CI mgcv metadata schema"
```

---

### Task 4: Deterministic Shards and Restart Qualification

**Files:**
- Modify: `fastkpc/R/full_cuda_ci_workload_census.R`
- Create: `fastkpc/tests/test_full_cuda_ci_workload_census_restart.R`

- [ ] **Step 1: Write failing shard/restart tests**

Use a 12-key synthetic request table and an injected deterministic `fit_fun`
that returns valid setup/target rows without calling mgcv. Test:

```text
sorted-rank modulo assignment
expected key count/hash per shard
atomic temporary-file rename
completed shard reuse
interrupted run leaves no completed partial shard
wrong corpus hash rejection
wrong risk-config hash rejection
wrong R/mgcv/source/BLAS identity rejection
duplicate key rejection
duplicate shard rejection
missing shard rejection
merge order determinism
```

- [ ] **Step 2: Run the restart test and verify RED**

```bash
Rscript fastkpc/tests/test_full_cuda_ci_workload_census_restart.R
```

Expected: failure because shard helpers are absent.

- [ ] **Step 3: Implement sorted-rank shard assignment**

Add:

```r
fastkpc_full_cuda_census_assign_shards <- function(requests, shard_count) {
  order_id <- order(requests$residual_key_sha256, method = "radix")
  requests <- requests[order_id, , drop = FALSE]
  requests$sorted_rank <- seq_len(nrow(requests))
  requests$shard_id <- (requests$sorted_rank - 1L) %% as.integer(shard_count)
  requests
}
```

Compute each shard hash from its sorted residual hashes joined by LF with a
final LF.

- [ ] **Step 4: Implement complete shard manifests**

`fastkpc_full_cuda_census_shard_manifest()` must include:

```text
canonical_key_corpus_hash
expected_key_count_for_shard
expected_key_hash_for_shard
dataset_sha256
oracle_input_bundle_sha256
source_commit
R_version
mgcv_version
BLAS_identity
LAPACK_identity
BLAS_thread_count
formula_semantics_version
mgcv_semantics_version
risk_threshold_config_hash
metadata_schema_version
shard_count
shard_id
```

Use `sessionInfo()`, `extSoftVersion()`, and the existing environment helpers
for identities. Require exact manifest equality on resume.

- [ ] **Step 5: Implement atomic shard writing**

Write RDS and JSON to random temporary names in the destination directory,
flush/close them, validate they can be read, then `file.rename()` to final
paths. Delete temporary files on error. Write the summary JSON last so its
presence marks completion.

- [ ] **Step 6: Implement shard execution and fail-closed merge**

Add:

```r
fastkpc_full_cuda_census_run_shard(
  assigned_requests, shard_id, context, output_dir,
  fit_fun = fastkpc_full_cuda_census_fit_key
)

fastkpc_full_cuda_census_merge_shards(
  requests, shard_count, context, shard_dir
)
```

The merge must validate every manifest, require the exact request-key set,
retain error rows, reject duplicates/missing rows, sort target fits by residual
SHA, compress setup observations, and calculate risk/coverage tables.

- [ ] **Step 7: Run restart qualification**

```bash
Rscript fastkpc/tests/test_full_cuda_ci_workload_census_restart.R
Rscript fastkpc/tests/test_full_cuda_ci_workload_census_metadata.R
git diff --check
```

Expected: `PASS full CUDA CI census restart qualification`.

- [ ] **Step 8: Commit Stage 4**

```bash
git add fastkpc/R/full_cuda_ci_workload_census.R \
        fastkpc/tests/test_full_cuda_ci_workload_census_restart.R
git commit -m "feat: add restartable full CUDA CI census shards"
```

---

### Task 5: Standard Runner and Scaled Dry Runs

**Files:**
- Modify: `fastkpc/R/full_cuda_ci_workload_census.R`
- Create: `fastkpc/tools/run_full_cuda_ci_workload_census.R`
- Create: `fastkpc/tests/test_full_cuda_ci_workload_census_real_subset.R`

- [ ] **Step 1: Write the failing runner/real-subset test**

The test must invoke the runner with a temporary output directory and:

```text
FASTKPC_FULL_CUDA_CENSUS_MODE=metadata
FASTKPC_FULL_CUDA_CENSUS_MAX_KEYS=8
FASTKPC_FULL_CUDA_CENSUS_WORKERS=1
FASTKPC_FULL_CUDA_CENSUS_SHARD_COUNT=2
```

Assert the standard files exist, eight target rows are present, the selected
keys are the first eight lexicographically sorted SHA values, parity passed,
inherited graph gate is true, new candidate gate is NOT_APPLICABLE, and rerun
with resume produces byte-identical merged RDS hashes.

- [ ] **Step 2: Run the real-subset test and verify RED**

```bash
Rscript fastkpc/tests/test_full_cuda_ci_workload_census_real_subset.R
```

Expected: failure because the runner does not exist.

- [ ] **Step 3: Implement standard artifact paths and writer**

Extend the core module with paths for every approved artifact file. Write
structural RDS/CSV, parity evidence, setup/fit metadata, risk cases, coverage,
distributions, inherited gate CSVs, timing, raw runs, manifest, summary JSON,
summary markdown, commands, and environment.

The manifest must include `p_floor`, the full risk configuration and its hash,
metadata/hash schema versions, canonical dataset matrix hash, dataset file
hash, oracle bundle hash, canonical key corpus hash, R/mgcv/BLAS/LAPACK
identities, requested/actual worker counts, and shard count.

Summary `pass` is true only when the selected run scope is internally complete.
For scaled runs include:

```text
run_scope = scaled_prefix
phase1_complete = FALSE
selected_key_count = N
canonical_key_count = 110617
```

Only the full run may set `phase1_complete = TRUE`.

- [ ] **Step 4: Implement the runner**

Create `fastkpc/tools/run_full_cuda_ci_workload_census.R` with these env/arg
controls:

```text
FASTKPC_FULL_CUDA_CENSUS_ORACLE_DIR
FASTKPC_FULL_CUDA_CENSUS_DATA_PATH
FASTKPC_FULL_CUDA_CENSUS_OUTPUT_DIR
FASTKPC_FULL_CUDA_CENSUS_MODE=structural|parity|metadata
FASTKPC_FULL_CUDA_CENSUS_MAX_KEYS=0
FASTKPC_FULL_CUDA_CENSUS_WORKERS=1
FASTKPC_FULL_CUDA_CENSUS_SHARD_COUNT=64
FASTKPC_FULL_CUDA_CENSUS_RESUME=1
```

`MAX_KEYS > 0` selects a fixed prefix after SHA sort. `MAX_KEYS = 0` means all
110,617 keys. Refuse full metadata mode unless parity passes in the same run or
an exact matching parity artifact is loaded.

Use `parallel::mclapply` only on Unix and only when workers > 1. Otherwise run
sequentially. Record requested and actual workers.

- [ ] **Step 5: Run the real-subset test GREEN**

```bash
Rscript fastkpc/tests/test_full_cuda_ci_workload_census_real_subset.R
```

Expected: `PASS full CUDA CI census real subset`.

- [ ] **Step 6: Run fixed scaled dry runs**

Run three deterministic prefixes:

```bash
FASTKPC_FULL_CUDA_CENSUS_MAX_KEYS=64 \
FASTKPC_FULL_CUDA_CENSUS_WORKERS=2 \
FASTKPC_FULL_CUDA_CENSUS_SHARD_COUNT=4 \
FASTKPC_FULL_CUDA_CENSUS_OUTPUT_DIR=fastkpc/artifacts/full_cuda_ci/workload_census_dry_64_v1 \
Rscript fastkpc/tools/run_full_cuda_ci_workload_census.R

FASTKPC_FULL_CUDA_CENSUS_MAX_KEYS=512 \
FASTKPC_FULL_CUDA_CENSUS_WORKERS=4 \
FASTKPC_FULL_CUDA_CENSUS_SHARD_COUNT=8 \
FASTKPC_FULL_CUDA_CENSUS_OUTPUT_DIR=fastkpc/artifacts/full_cuda_ci/workload_census_dry_512_v1 \
Rscript fastkpc/tools/run_full_cuda_ci_workload_census.R

FASTKPC_FULL_CUDA_CENSUS_MAX_KEYS=4096 \
FASTKPC_FULL_CUDA_CENSUS_WORKERS=8 \
FASTKPC_FULL_CUDA_CENSUS_SHARD_COUNT=16 \
FASTKPC_FULL_CUDA_CENSUS_OUTPUT_DIR=fastkpc/artifacts/full_cuda_ci/workload_census_dry_4096_v1 \
/usr/bin/time -v Rscript fastkpc/tools/run_full_cuda_ci_workload_census.R
```

For each run verify no fit errors, deterministic key hashes, exact same-S
invariants, 100% required field coverage, bounded peak RSS/disk, no unclassified
warnings/non-finite fields, and merge determinism. Record throughput and
estimated full runtime in `summary.md`.

- [ ] **Step 7: Run all focused tests**

```bash
Rscript fastkpc/tests/test_full_cuda_ci_workload_census.R
Rscript fastkpc/tests/test_full_cuda_ci_workload_census_parity.R
Rscript fastkpc/tests/test_full_cuda_ci_workload_census_metadata.R
Rscript fastkpc/tests/test_full_cuda_ci_workload_census_restart.R
Rscript fastkpc/tests/test_full_cuda_ci_workload_census_real_subset.R
git diff --check
```

Expected: all PASS.

- [ ] **Step 8: Commit Stage 5**

```bash
git add fastkpc/R/full_cuda_ci_workload_census.R \
        fastkpc/tools/run_full_cuda_ci_workload_census.R \
        fastkpc/tests/test_full_cuda_ci_workload_census_real_subset.R
git commit -m "feat: run scaled full CUDA CI workload census"
```

---

### Task 6: Full 110,617-Key Census and Phase 1 Closure

**Files:**
- Modify: `goal-5.6.md`
- Artifact: `fastkpc/artifacts/full_cuda_ci/workload_census_351x48_v1/`

- [ ] **Step 1: Run preflight verification**

```bash
Rscript fastkpc/tests/test_full_cuda_ci_workload_census.R
Rscript fastkpc/tests/test_full_cuda_ci_workload_census_parity.R
Rscript fastkpc/tests/test_full_cuda_ci_workload_census_metadata.R
Rscript fastkpc/tests/test_full_cuda_ci_workload_census_restart.R
Rscript fastkpc/tests/test_full_cuda_ci_workload_census_real_subset.R
FASTKPC_FULL_CUDA_CI_ORACLE_MODE=validate \
Rscript fastkpc/tools/run_full_cuda_ci_oracle.R
git diff --check
```

Expected: all tests PASS and Phase 0 remains SHD 0 with all semantic fields
identical.

- [ ] **Step 2: Run the full census**

Use 20 workers and 64 shards. If the mandatory 4096-key dry run proves this
configuration exceeds available memory or regresses throughput, stop and amend
this plan with the measured replacement before starting the full pass:

```bash
FASTKPC_FULL_CUDA_CENSUS_MODE=metadata \
FASTKPC_FULL_CUDA_CENSUS_MAX_KEYS=0 \
FASTKPC_FULL_CUDA_CENSUS_WORKERS=20 \
FASTKPC_FULL_CUDA_CENSUS_SHARD_COUNT=64 \
FASTKPC_FULL_CUDA_CENSUS_RESUME=1 \
FASTKPC_FULL_CUDA_CENSUS_OUTPUT_DIR=fastkpc/artifacts/full_cuda_ci/workload_census_351x48_v1 \
/usr/bin/time -v Rscript fastkpc/tools/run_full_cuda_ci_workload_census.R
```

If interrupted, rerun the exact command. Resume must reuse only exact matching
completed shards.

- [ ] **Step 3: Verify canonical structural and numerical gates**

Run:

```bash
Rscript -e '
s <- jsonlite::read_json(
  "fastkpc/artifacts/full_cuda_ci/workload_census_351x48_v1/summary.json",
  simplifyVector = TRUE
)
stopifnot(
  isTRUE(s$pass),
  isTRUE(s$phase1_complete),
  s$logical_test_count == 240489L,
  s$conditional_logical_test_count == 238276L,
  s$conditional_residual_request_count == 476552L,
  s$canonical_global_unique_conditional_target_s_count == 110617L,
  s$unique_conditional_S_count == 8634L,
  s$same_s_setup_metadata_rows == 8634L,
  s$target_fit_metadata_rows == 110617L,
  s$mgcv_fit_error_count == 0L,
  s$same_s_invariant_violation_count == 0L,
  s$required_field_coverage == 1,
  isTRUE(s$legacy_layout_parity_pass),
  isTRUE(s$oracle_inherited_graph_gate),
  identical(s$new_candidate_graph_gate, "NOT_APPLICABLE"),
  identical(s$canonical_key_corpus_hash,
    "b843630969f116da63f7fad095c54de2ff471540159ff97ca56c3871d6b2e1fa")
)
'
```

Inspect `risk_cases.csv`, `field_coverage.csv`, all distribution files,
warnings, non-finite classifications, unsupported envelope, shard manifests,
peak RSS, disk use, and full runtime. No required row may be absent.

- [ ] **Step 4: Re-run inherited Phase 0 gate after the full pass**

```bash
FASTKPC_FULL_CUDA_CI_ORACLE_MODE=validate \
Rscript fastkpc/tools/run_full_cuda_ci_oracle.R
```

Expected:

```text
edge_count: 110 / 110
SHD: 0
adjacency_identical: TRUE
sepsets_identical: TRUE
n_edgetests_identical: TRUE
deletions_identical: TRUE
pass: TRUE
```

- [ ] **Step 5: Update Phase 1 roadmap evidence**

In `goal-5.6.md`, change Phase 1 status from PARTIAL to COMPLETE and record:

```text
artifact path
source commit
oracle input bundle hash
canonical key corpus hash
240,489 / 238,276 / 476,552 / 110,617 / 8,634 counts
8,634 setup rows
110,617 target-fit rows
fit errors and invariant violations = 0
risk-case counts
runtime, peak RSS, disk use, workers, and shard count
273,284 retained only as historical S-affinity route metric
```

Do not mark later phases complete.

- [ ] **Step 6: Run final Phase 1 verification**

```bash
Rscript fastkpc/tests/test_full_cuda_ci_workload_census.R
Rscript fastkpc/tests/test_full_cuda_ci_workload_census_parity.R
Rscript fastkpc/tests/test_full_cuda_ci_workload_census_metadata.R
Rscript fastkpc/tests/test_full_cuda_ci_workload_census_restart.R
Rscript fastkpc/tests/test_full_cuda_ci_workload_census_real_subset.R
FASTKPC_FULL_CUDA_CI_ORACLE_MODE=validate \
Rscript fastkpc/tools/run_full_cuda_ci_oracle.R
git diff --check
git status --short
```

Expected: all tests and gates PASS. The ignored artifact exists locally and is
fully reproducible from the recorded command and shard manifests.

- [ ] **Step 7: Commit and push Phase 1 closure**

```bash
git add fastkpc/R/full_cuda_ci_oracle_contract.R \
        fastkpc/R/full_cuda_ci_workload_census.R \
        fastkpc/tools/run_full_cuda_ci_workload_census.R \
        fastkpc/tests/test_full_cuda_ci_workload_census.R \
        fastkpc/tests/test_full_cuda_ci_workload_census_parity.R \
        fastkpc/tests/test_full_cuda_ci_workload_census_metadata.R \
        fastkpc/tests/test_full_cuda_ci_workload_census_restart.R \
        fastkpc/tests/test_full_cuda_ci_workload_census_real_subset.R \
        goal-5.6.md
git commit -m "feat: complete full CUDA CI Phase 1 workload census"
git -c http.version=HTTP/1.1 \
    -c http.proxy=http://localhost:7890 \
    push --verbose origin HEAD:refs/heads/main
```

Phase 1 is complete only after the full 110,617-key artifact passes every gate.
The overall `goal-5.6.md` goal remains active and proceeds to Phase 2.
