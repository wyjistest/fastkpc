# Full-CUDA CI Phase 3A Persistent Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the authenticated Phase 3 catalog/DTO and a process-persistent CUDA fixed-sp runtime that uploads one Prepared-S setup once, computes RHS on CUDA, solves safe Cholesky targets through a batch-size-one adapter, returns a leased device-resident residual token, and fails closed for every target that requires the Phase 3C stable path.

**Architecture:** Add a read-only R catalog over the completed Phase 1/2 artifacts, then translate one setup and its targets into a versioned native DTO. A shared C++ `CudaRuntimeContext` owns one explicitly deterministic stream/cuBLAS/cuSOLVER configuration and pre-reserved arenas; a `PreparedSGpuHandle` owns response-independent setup buffers; a separately leased `TransientResidualSlot` owns target outputs. CUDA computes `X0' * Y` internally, and `DeviceResidualBatch` holds an exclusive slot lease until explicit release. Phase 3A supports only the authenticated `< 1e8`, full-rank Cholesky envelope and returns `ERR_STABLE_PATH_NOT_IMPLEMENTED` for QR/SVD targets.

**Tech Stack:** R 4.4.1, Rcpp `.Call`, C++17, CUDA C++17, CUDA Runtime, cuBLAS, cuSOLVER, existing Phase 0/1/2 artifact validators, existing `mgcv` fixed-sp `C_magic` reference.

---

## Scope and Preconditions

Implement against:

```text
design spec:
  docs/superpowers/specs/2026-07-13-full-cuda-ci-phase3-fixed-sp-runtime-design.md

implementation branch base:
  main >= 37ce594 plus the Phase 3 review-amendment commit

production code baseline:
  42ef3ef

worktree:
  /home/amax/.config/superpowers/worktrees/kpcalg/phase1-structural-census

GPU:
  NVIDIA GeForce RTX 4090, compute capability 8.9
```

Do not stage `fastkpc/artifacts`; it is a shared untracked artifact symlink.
Push reviewed commits with the configured `localhost:7890` proxy only after
the task-level tests and review pass.

Phase 3A is not Phase 3 completion. Its canonical route outcome is:

```text
iteration targets:
  OK Cholesky                         = 172
  ERR_STABLE_PATH_NOT_IMPLEMENTED     = 98

qualification/full artifacts:
  not permitted until Phase 3C
```

## File Structure

Create:

```text
fastkpc/R/full_cuda_ci_fixed_sp_runtime.R
  Authenticated Phase 3 catalog, route contract, DTO builder, and iteration
  runner helpers. No CUDA implementation details beyond wrapper calls.

fastkpc/src/cuda/mgcv_fixed_sp_runtime_types.hpp
  Versioned enums and plain C++ request/result/diagnostic structs.

fastkpc/src/cuda/mgcv_fixed_sp_runtime.hpp
  Opaque runtime, prepared-handle, residual-token API.

fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu
  RAII resources, reserve arenas, setup upload, Phase 3A Cholesky solve,
  output-slot leases, generation checks, explicit shadow materialization.

fastkpc/tests/test_full_cuda_ci_fixed_sp_catalog.R
fastkpc/tests/test_full_cuda_ci_fixed_sp_native_dto.R
fastkpc/tests/test_mgcv_fixed_sp_cuda_runtime_lifecycle.R
fastkpc/tests/test_mgcv_fixed_sp_cuda_prepared_handle.R
fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3a_solve.R
fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3a_iteration.R
fastkpc/tests/test_mgcv_fixed_sp_cuda_runtime_misuse.R
```

Modify:

```text
fastkpc/tools/build_cuda_native.sh
  Compile/link the new runtime object.

fastkpc/src/r_api_cuda.cpp
  Tagged external-pointer/finalizer wrappers and `.Call` registration.

fastkpc/R/cuda_native.R
  Thin R wrappers over the new `.Call` entries.

goal-5.6.md
  Record Phase 3A only after all Phase 3A verification passes.
```

## Spec Coverage for This Subplan

```text
authenticated Phase 0/1/2 catalog and route metadata    Tasks 1-3
versioned zero-based DTO and hash boundary             Task 3
persistent deterministic stream/handles/workspaces      Task 4
one-time Prepared-S upload and setup ownership          Task 5
leased device token and explicit stable-path failure    Task 6
CUDA RHS, safe Cholesky, and shadow materialization     Task 7
real iteration evidence and prototype speed gate        Task 8
PID/device/tag/lease/generation misuse hardening         Task 9
clean build, regression suite, and roadmap record       Task 10
```

True multi-target Cholesky is intentionally assigned to the Phase 3B plan.
Penalty roots, augmented QR/SVD, qualification, and both full artifacts are
intentionally assigned to the Phase 3C and closure plans. Phase 3A does not
claim those requirements.

## Task 1: Freeze the Phase 3 Route Contract in R

**Files:**
- Create: `fastkpc/R/full_cuda_ci_fixed_sp_runtime.R`
- Create: `fastkpc/tests/test_full_cuda_ci_fixed_sp_catalog.R`

- [ ] **Step 1: Write the failing route-contract test**

Create `fastkpc/tests/test_full_cuda_ci_fixed_sp_catalog.R` with the initial
contract assertions:

```r
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

phase1_dir <- file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "workload_census_351x48_v1"
)
targets <- readRDS(file.path(phase1_dir, "target_fit_metadata.rds"))
setups <- readRDS(file.path(phase1_dir, "same_s_setup_metadata.rds"))
setup_index <- match(targets$same_S_group_id, setups$same_S_group_id)
assert_true(!anyNA(setup_index), "canonical target/setup null-dim join")
target_null_dim <- as.integer(
  setups$constraint_nullspace_dimension[setup_index]
)
routes <- fastkpc_full_cuda_fixed_sp_route(
  condition = targets$penalized_system_condition_at_selected_sp,
  coefficient_rank = targets$coefficient_rank,
  null_dim = target_null_dim,
  authenticated = rep(TRUE, nrow(targets))
)

planned_route_counts <- table(factor(
  routes,
  levels = c("CHOLESKY_BATCHED", "AUGMENTED_QR", "AUGMENTED_SVD")
))
assert_true(identical(
  as.integer(planned_route_counts), c(73158L, 4210L, 33249L)
), "canonical Phase 3 planned route counts")
rank_deficient <- targets$coefficient_rank < target_null_dim
assert_true(sum(rank_deficient) == 1L &&
              all(routes[rank_deficient] == "AUGMENTED_SVD"),
            "canonical rank-deficient targets route to SVD")
nonfinite_condition <-
  !is.finite(targets$penalized_system_condition_at_selected_sp)
assert_true(sum(nonfinite_condition) == 1162L &&
              all(routes[nonfinite_condition] == "AUGMENTED_SVD"),
            "canonical nonfinite-condition targets route to SVD")

assert_true(identical(
  fastkpc_full_cuda_fixed_sp_route(
    condition = NA_real_, coefficient_rank = 10L, null_dim = 10L,
    authenticated = FALSE
  ),
  "AUGMENTED_SVD"
), "unauthenticated conditions must route to SVD")
assert_true(identical(
  fastkpc_full_cuda_fixed_sp_route(
    condition = 1, coefficient_rank = 9L, null_dim = 10L,
    authenticated = TRUE
  ),
  "AUGMENTED_SVD"
), "finite rank-deficient targets must route to SVD")

contract <- fastkpc_full_cuda_fixed_sp_contract()
assert_true(identical(contract$schema_version,
                      "full-cuda-ci-fixed-sp-runtime-v1"),
            "runtime schema version")
assert_true(identical(contract$cholesky_condition_max, 1e8),
            "Cholesky condition threshold")
assert_true(identical(contract$svd_condition_min, 1e12),
            "SVD condition threshold")

cat("PASS Phase 3 fixed-sp route contract\n")
```

- [ ] **Step 2: Run the test and verify the missing API failure**

Run:

```bash
Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_catalog.R
```

Expected: FAIL while sourcing the missing
`fastkpc/R/full_cuda_ci_fixed_sp_runtime.R` or with
`could not find function "fastkpc_full_cuda_fixed_sp_route"`.

- [ ] **Step 3: Implement the route contract**

Create `fastkpc/R/full_cuda_ci_fixed_sp_runtime.R` with:

```r
fastkpc_full_cuda_fixed_sp_contract <- function() {
  list(
    schema_version = "full-cuda-ci-fixed-sp-runtime-v1",
    native_dto_schema_version = "full-cuda-ci-prepared-s-native-dto-v1",
    cholesky_condition_max = 1e8,
    svd_condition_min = 1e12,
    route_levels = c(
      "CHOLESKY_BATCHED", "AUGMENTED_QR", "AUGMENTED_SVD"
    ),
    target_status_levels = c(
      "OK_CHOLESKY_BATCHED", "OK_CHOLESKY_SINGLE",
      "OK_AUGMENTED_QR", "OK_AUGMENTED_SVD",
      "ERR_NONFINITE_INPUT", "ERR_SP_SHAPE_OR_ORDER",
      "ERR_ROUTE_METADATA", "ERR_STABLE_PATH_NOT_IMPLEMENTED",
      "ERR_QR_FAILED", "ERR_SVD_FAILED", "ERR_NONFINITE_OUTPUT",
      "ERR_INTERNAL_CUDA"
    ),
    canonical_capacities = list(
      n = 351L, null_dim = 64L, target_count = 47L,
      penalty_count = 7L, augmented_rows = 407L
    )
  )
}

fastkpc_full_cuda_fixed_sp_route <- function(
    condition, coefficient_rank, null_dim, authenticated) {
  n <- max(length(condition), length(coefficient_rank), length(null_dim),
           length(authenticated))
  condition <- rep_len(as.numeric(condition), n)
  coefficient_rank <- rep_len(as.integer(coefficient_rank), n)
  null_dim <- rep_len(as.integer(null_dim), n)
  authenticated <- rep_len(as.logical(authenticated), n)
  out <- rep("AUGMENTED_SVD", n)
  trusted <- !is.na(authenticated) & authenticated &
    is.finite(condition) & !is.na(coefficient_rank) & !is.na(null_dim) &
    coefficient_rank == null_dim
  out[trusted & condition < 1e8] <- "CHOLESKY_BATCHED"
  out[trusted & condition >= 1e8 & condition < 1e12] <- "AUGMENTED_QR"
  if (n == 1L) out[[1L]] else out
}
```

- [ ] **Step 4: Run the test and verify exact planned-route counts**

Run:

```bash
Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_catalog.R
```

Expected: `PASS Phase 3 fixed-sp route contract`.

- [ ] **Step 5: Commit the route contract**

```bash
git add fastkpc/R/full_cuda_ci_fixed_sp_runtime.R \
  fastkpc/tests/test_full_cuda_ci_fixed_sp_catalog.R
git commit -m "feat: add Phase 3 fixed-sp route contract"
```

## Task 2: Add the Authenticated Subset Catalog

**Files:**
- Modify: `fastkpc/R/full_cuda_ci_fixed_sp_runtime.R`
- Modify: `fastkpc/tests/test_full_cuda_ci_fixed_sp_catalog.R`

- [ ] **Step 1: Extend the test with iteration catalog assertions**

Append:

```r
catalog <- fastkpc_full_cuda_open_fixed_sp_catalog(
  phase0_dir = file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "oracle_351x48_v1"
  ),
  phase1_dir = phase1_dir,
  phase2_dir = file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1"
  ),
  data_path = file.path(
    "fastkpc", "artifacts", "kpc_tprs_real_zhu",
    "cancer_RD-causalDiscoveryInput.rds"
  ),
  require_full = TRUE
)
iteration <- fastkpc_full_cuda_fixed_sp_scope(catalog, "iteration")

assert_true(nrow(iteration$setup_rows) == 44L,
            "iteration setup count")
assert_true(nrow(iteration$target_rows) == 270L,
            "iteration target count")
assert_true(length(iteration$shard_ids) <= 44L,
            "subset loader must not load all setup payloads")
iteration_planned_route_counts <- table(factor(
  iteration$target_rows$planned_route,
  levels = c("CHOLESKY_BATCHED", "AUGMENTED_QR", "AUGMENTED_SVD")
))
assert_true(identical(
  as.integer(iteration_planned_route_counts), c(172L, 31L, 67L)
), "iteration planned route counts")

batches <- fastkpc_full_cuda_fixed_sp_batches(catalog, iteration)
assert_true(length(batches) == 44L, "one batch per iteration setup")
assert_true(sum(vapply(batches, function(x) ncol(x$Y), integer(1))) == 270L,
            "all iteration targets materialized")
assert_true(all(vapply(batches, function(x) {
  identical(nrow(x$SP), length(x$setup$penalty_blocks)) &&
    identical(ncol(x$SP), ncol(x$Y))
}, logical(1))), "SP matrix dimensions")

cat("PASS authenticated Phase 3 iteration catalog\n")
```

- [ ] **Step 2: Run the test and verify the missing catalog failure**

Run:

```bash
Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_catalog.R
```

Expected: FAIL with
`could not find function "fastkpc_full_cuda_open_fixed_sp_catalog"`.

- [ ] **Step 3: Implement catalog open and scope selection**

Add these public functions and private helpers to
`fastkpc/R/full_cuda_ci_fixed_sp_runtime.R`:

```r
fastkpc_full_cuda_fixed_sp_read_json <- function(path) {
  if (!file.exists(path)) stop("required artifact file is missing: ", path,
                               call. = FALSE)
  jsonlite::read_json(path, simplifyVector = TRUE)
}

fastkpc_full_cuda_open_fixed_sp_catalog <- function(
    phase0_dir, phase1_dir, phase2_dir, data_path, require_full = TRUE) {
  oracle <- fastkpc_load_full_cuda_ci_oracle(phase0_dir)
  phase1_inputs <- fastkpc_full_cuda_prepared_s_load_inputs(
    phase1_dir, data_path
  )
  phase2_summary <- fastkpc_full_cuda_fixed_sp_read_json(
    file.path(phase2_dir, "summary.json")
  )
  if (!isTRUE(oracle$summary$pass) || !isTRUE(phase1_inputs$summary$pass) ||
      !isTRUE(phase2_summary$pass) ||
      (isTRUE(require_full) && !isTRUE(phase2_summary$phase2_complete))) {
    stop("Phase 0/1/2 input gate is incomplete", call. = FALSE)
  }
  phase2_manifest <- fastkpc_full_cuda_fixed_sp_read_json(
    file.path(phase2_dir, "manifest.json")
  )
  required_identity <- isTRUE(phase2_manifest$phase2_complete) &&
    identical(as.integer(phase2_manifest$selected_group_count), 8634L) &&
    identical(as.integer(phase2_manifest$target_state_count), 110617L) &&
    identical(as.integer(phase2_manifest$shard_count), 64L) &&
    identical(
      as.character(phase2_manifest$phase1_input_bundle_hash),
      as.character(phase1_inputs$phase1_input_bundle_hash)
    ) && identical(
      as.character(phase2_manifest$dataset_matrix_sha256),
      as.character(phase1_inputs$dataset_sha256)
    )
  if (!required_identity) {
    stop("Phase 3 Phase 2 artifact identity mismatch", call. = FALSE)
  }
  semantic <- phase2_manifest$semantic_file_sha256
  required_semantic <- c(
    "prepared_s_setup_index_csv", "iteration_setup_groups_rds",
    "iteration_target_keys_rds", "qualification_setup_groups_rds",
    "qualification_target_keys_rds"
  )
  if (length(setdiff(required_semantic, names(semantic))) > 0L) {
    stop("Phase 3 Phase 2 semantic hash set is incomplete", call. = FALSE)
  }
  semantic_paths <- c(
    prepared_s_setup_index_csv = "prepared_s_setup_index.csv",
    iteration_setup_groups_rds = "iteration_setup_groups.rds",
    iteration_target_keys_rds = "iteration_target_keys.rds",
    qualification_setup_groups_rds = "qualification_setup_groups.rds",
    qualification_target_keys_rds = "qualification_target_keys.rds"
  )
  actual <- vapply(semantic_paths, function(name) {
    fastkpc_full_cuda_file_hash(file.path(phase2_dir, name))
  }, character(1L))
  if (!identical(unname(actual),
                 unname(as.character(semantic[names(actual)])))) {
    stop("Phase 3 Phase 2 semantic file hash mismatch", call. = FALSE)
  }
  setup_summary <- utils::read.csv(
    file.path(phase2_dir, "prepared_s_setup_index.csv"),
    stringsAsFactors = FALSE, check.names = FALSE
  )
  list(
    phase0_dir = phase0_dir, phase1_dir = phase1_dir,
    phase2_dir = phase2_dir, data_path = data_path,
    data = phase1_inputs$data, oracle = oracle,
    phase1_inputs = phase1_inputs,
    phase0_summary = oracle$summary,
    phase1_summary = phase1_inputs$summary,
    phase2_summary = phase2_summary, phase2_manifest = phase2_manifest,
    setup_summary = setup_summary
  )
}

fastkpc_full_cuda_fixed_sp_scope <- function(catalog, scope) {
  scope <- match.arg(scope, c("iteration", "qualification", "full"))
  if (identical(scope, "full")) {
    stop("full scope streaming is introduced in the closure plan",
         call. = FALSE)
  }
  setup_rows <- readRDS(file.path(
    catalog$phase2_dir, paste0(scope, "_setup_groups.rds")
  ))
  target_rows <- readRDS(file.path(
    catalog$phase2_dir, paste0(scope, "_target_keys.rds")
  ))
  mapping <- catalog$setup_summary[, c(
    "prepared_s_key_sha256", "same_S_group_id"
  ), drop = FALSE]
  setup_index <- match(setup_rows$same_S_group_id, mapping$same_S_group_id)
  target_setup_index <- match(
    target_rows$same_S_group_id, mapping$same_S_group_id
  )
  if (anyNA(setup_index) || anyNA(target_setup_index)) {
    stop("Phase 3 scope PreparedSKey mapping is incomplete", call. = FALSE)
  }
  setup_rows$prepared_s_key_sha256 <-
    mapping$prepared_s_key_sha256[setup_index]
  target_rows$prepared_s_key_sha256 <-
    mapping$prepared_s_key_sha256[target_setup_index]
  fit_index <- match(
    target_rows$residual_key_sha256,
    catalog$phase1_inputs$target_fit_metadata$residual_key_sha256
  )
  if (anyNA(fit_index)) {
    stop("Phase 3 scope target metadata mapping is incomplete", call. = FALSE)
  }
  fit_rows <- catalog$phase1_inputs$target_fit_metadata[
    fit_index, , drop = FALSE
  ]
  setup_null_dim <- as.integer(setup_rows$model_matrix_ncol) -
    as.integer(setup_rows$constraint_rank)
  target_null_dim <- setup_null_dim[match(
    target_rows$same_S_group_id, setup_rows$same_S_group_id
  )]
  if (anyNA(target_null_dim)) {
    stop("Phase 3 scope nullspace dimension mapping is incomplete",
         call. = FALSE)
  }
  target_rows$condition <-
    as.numeric(fit_rows$penalized_system_condition_at_selected_sp)
  target_rows$coefficient_rank <- as.integer(fit_rows$coefficient_rank)
  target_rows$null_dim <- target_null_dim
  target_rows$planned_route <- fastkpc_full_cuda_fixed_sp_route(
    condition = target_rows$condition,
    coefficient_rank = target_rows$coefficient_rank,
    null_dim = target_rows$null_dim,
    authenticated = rep(TRUE, nrow(target_rows))
  )
  sorted_keys <- sort(mapping$prepared_s_key_sha256, method = "radix")
  ranks <- match(setup_rows$prepared_s_key_sha256, sorted_keys)
  shard_ids <- sort(unique((ranks - 1L) %% 64L))
  list(scope = scope, setup_rows = setup_rows, target_rows = target_rows,
       shard_ids = as.integer(shard_ids))
}
```

The implementation must use the existing Phase 2 shard reader/validator when
loading each selected shard. Do not call `readRDS()` directly without
validating the matching `.summary.json`.

- [ ] **Step 4: Implement deterministic batch materialization**

Add:

```r
fastkpc_full_cuda_fixed_sp_batches <- function(catalog, selected_scope) {
  loaded <- lapply(selected_scope$shard_ids, function(shard_id) {
    fastkpc_full_cuda_prepared_s_read_shard(
      shard_dir = file.path(catalog$phase2_dir, "shards"),
      inputs = catalog$phase1_inputs,
      shard_count = 64L,
      shard_id = shard_id
    )
  })
  setups <- unlist(lapply(loaded, function(x) x$payload$prepared_s_setups),
                   recursive = FALSE)
  states <- do.call(rbind, lapply(loaded, function(x) x$payload$target_states))
  setup_order <- order(
    selected_scope$setup_rows$prepared_s_key_sha256, method = "radix"
  )
  lapply(setup_order, function(i) {
    key <- selected_scope$setup_rows$prepared_s_key_sha256[[i]]
    setup <- setups[[key]]
    wanted <- selected_scope$target_rows$residual_key_sha256[
      selected_scope$target_rows$prepared_s_key_sha256 == key
    ]
    state_index <- match(wanted, states$residual_key_sha256)
    if (is.null(setup) || anyNA(state_index)) {
      stop("Phase 3 selected shard payload is incomplete", call. = FALSE)
    }
    state_rows <- states[state_index, , drop = FALSE]
    materialized <- lapply(seq_len(nrow(state_rows)), function(j) {
      target <- fastkpc_full_cuda_materialize_target_state(
        state_rows[j, , drop = FALSE], catalog$data,
        setup$dataset_sha256
      )
      fastkpc_full_cuda_validate_materialized_target_for_prepared(
        setup, target
      )
    })
    Y <- do.call(cbind, lapply(materialized, `[[`, "y"))
    SP <- do.call(cbind, lapply(materialized, `[[`, "sp"))
    oracle_rhs <- do.call(cbind, lapply(
      materialized, `[[`, "nullspace_projected_rhs"
    ))
    metadata <- selected_scope$target_rows[
      match(wanted, selected_scope$target_rows$residual_key_sha256), ,
      drop = FALSE
    ]
    list(setup = setup, target_rows = state_rows, Y = Y, SP = SP,
         oracle_nullspace_rhs = oracle_rhs,
         planned_route = metadata$planned_route,
         condition = metadata$condition,
         prepared_s_key_sha256 = key)
  })
}
```

- [ ] **Step 5: Run the catalog test**

Run:

```bash
Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_catalog.R
```

Expected: both `PASS` lines, with 44 batches and 270 targets.

- [ ] **Step 6: Commit the catalog**

```bash
git add fastkpc/R/full_cuda_ci_fixed_sp_runtime.R \
  fastkpc/tests/test_full_cuda_ci_fixed_sp_catalog.R
git commit -m "feat: add authenticated Phase 3 Prepared-S catalog"
```

## Task 3: Build and Validate the Native DTO

**Files:**
- Modify: `fastkpc/R/full_cuda_ci_fixed_sp_runtime.R`
- Create: `fastkpc/tests/test_full_cuda_ci_fixed_sp_native_dto.R`

- [ ] **Step 1: Write the failing DTO test**

Create `fastkpc/tests/test_full_cuda_ci_fixed_sp_native_dto.R`:

```r
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

catalog <- fastkpc_full_cuda_open_fixed_sp_catalog(
  phase0_dir = file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "oracle_351x48_v1"
  ),
  phase1_dir = file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "workload_census_351x48_v1"
  ),
  phase2_dir = file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1"
  ),
  data_path = file.path(
    "fastkpc", "artifacts", "kpc_tprs_real_zhu",
    "cancer_RD-causalDiscoveryInput.rds"
  )
)
scope <- fastkpc_full_cuda_fixed_sp_scope(catalog, "iteration")
batches <- fastkpc_full_cuda_fixed_sp_batches(catalog, scope)

batch_index <- which(vapply(
  batches, function(x) length(x$setup$penalty_blocks) > 1L, logical(1L)
))[[1L]]
batch <- batches[[batch_index]]
dto <- fastkpc_full_cuda_fixed_sp_native_dto(batch$setup)
native_batch <- fastkpc_full_cuda_fixed_sp_native_batch(batch, dto)

assert_true(identical(dto$schema_version,
                      "full-cuda-ci-prepared-s-native-dto-v1"),
            "native DTO schema")
assert_true(identical(dto$prepared_s_key_sha256,
                      batch$prepared_s_key_sha256),
            "native DTO PreparedSKey")
assert_true(identical(dto$n, 351L), "native DTO row count")
assert_true(identical(native_batch$target_count, ncol(batch$Y)),
            "native target count")
assert_true(identical(dim(native_batch$SP),
                      c(dto$penalty_count, native_batch$target_count)),
            "native SP shape")
assert_true(dto$penalty_offsets_zero_based[[1L]] >= 0L,
            "penalty offsets are zero based")
assert_true(all(dto$penalty_sp_indices_zero_based >= 0L &
                  dto$penalty_sp_indices_zero_based < dto$penalty_count),
            "penalty SP indices are zero based and in range")
assert_true(identical(
  dto$penalty_sp_indices_zero_based,
  seq.int(0L, dto$penalty_count - 1L)
), "Phase 3 v1 penalty-to-SP mapping is identity")
assert_true(is.null(native_batch$nullspace_rhs),
            "production native batch must not carry CPU RHS")

reconstruct_penalty <- function(block, zero_offset, p) {
  out <- matrix(0, p, p)
  index <- zero_offset + seq_len(nrow(block))
  out[index, index] <- block
  out
}
check_indices <- unique(c(1L, dto$penalty_count))
for (index in check_indices) {
  from_dto <- reconstruct_penalty(
    dto$penalty_blocks[[index]],
    dto$penalty_offsets_zero_based[[index]], dto$coefficient_dim
  )
  from_phase2 <- reconstruct_penalty(
    batch$setup$penalty_blocks[[index]],
    batch$setup$penalty_offsets[[index]] - 1L, dto$coefficient_dim
  )
  assert_true(identical(from_dto, from_phase2),
              "zero-based penalty reconstruction")
}

zero_sp <- native_batch$SP
zero_sp[1L, 1L] <- 0
fastkpc_full_cuda_fixed_sp_validate_numeric_inputs(native_batch$Y, zero_sp)
assert_true(zero_sp[1L, 1L] == 0,
            "zero selected SP is a valid boundary")

negative_sp <- native_batch$SP
negative_sp[1L, 1L] <- -1
error_negative_sp <- tryCatch(
  fastkpc_full_cuda_fixed_sp_validate_numeric_inputs(
    native_batch$Y, negative_sp
  ),
  error = identity
)
assert_true(inherits(error_negative_sp, "error"),
            "negative selected SP must fail closed")

bad_y_hash <- batch
bad_y_hash$Y[1L, 1L] <- bad_y_hash$Y[1L, 1L] + 1e-6
error_y_hash <- tryCatch(
  fastkpc_full_cuda_fixed_sp_native_batch(bad_y_hash, dto), error = identity
)
assert_true(inherits(error_y_hash, "error") &&
              grepl("Y hash", conditionMessage(error_y_hash), fixed = TRUE),
            "R adapter validates Y hash")

bad_sp_hash <- batch
bad_sp_hash$SP[1L, 1L] <- bad_sp_hash$SP[1L, 1L] + 1e-6
error_sp_hash <- tryCatch(
  fastkpc_full_cuda_fixed_sp_native_batch(bad_sp_hash, dto), error = identity
)
assert_true(inherits(error_sp_hash, "error") &&
              grepl("selected-SP hash", conditionMessage(error_sp_hash),
                    fixed = TRUE),
            "R adapter validates selected-SP hash")

bad_order <- batch
bad_order$target_rows$selected_sp_names[[1L]] <-
  rev(bad_order$target_rows$selected_sp_names[[1L]])
error_order <- tryCatch(
  fastkpc_full_cuda_fixed_sp_native_batch(bad_order, dto),
  error = identity
)
assert_true(inherits(error_order, "error") &&
              grepl("malformed", conditionMessage(error_order), fixed = TRUE),
            "SP name order must fail closed")

bad <- batch$setup
bad$weights_policy <- "unsupported-weights"
error <- tryCatch(
  fastkpc_full_cuda_fixed_sp_native_dto(bad),
  error = identity
)
assert_true(inherits(error, "error") &&
              grepl("weights policy", conditionMessage(error), fixed = TRUE),
            "unsupported weights must fail closed")

bad_h <- batch$setup
bad_h$H <- diag(ncol(bad_h$X))
error_h <- tryCatch(
  fastkpc_full_cuda_fixed_sp_native_dto(bad_h), error = identity
)
assert_true(inherits(error_h, "error") &&
              grepl("non-null H", conditionMessage(error_h), fixed = TRUE),
            "Phase 3A non-null H must fail closed")

bad_mapping <- batch$setup
bad_mapping$sp_mapping <- diag(length(bad_mapping$penalty_blocks))
error_mapping <- tryCatch(
  fastkpc_full_cuda_fixed_sp_native_dto(bad_mapping), error = identity
)
assert_true(inherits(error_mapping, "error") &&
              grepl("smoothing mapping", conditionMessage(error_mapping),
                    fixed = TRUE),
            "Phase 3A smoothing mapping must fail closed")

bad_offset <- batch$setup
bad_offset$penalty_offsets[[1L]] <- -1L
error_offset <- tryCatch(
  fastkpc_full_cuda_fixed_sp_native_dto(bad_offset), error = identity
)
assert_true(inherits(error_offset, "error") &&
              grepl("penalty offset", conditionMessage(error_offset),
                    fixed = TRUE),
            "out-of-range penalty offset must fail closed")

bad_sp_index <- batch$setup
bad_sp_index$penalty_sp_indices[[1L]] <-
  length(bad_sp_index$penalty_blocks) + 1L
error_sp_index <- tryCatch(
  fastkpc_full_cuda_fixed_sp_native_dto(bad_sp_index), error = identity
)
assert_true(inherits(error_sp_index, "error") &&
              grepl("penalty SP index", conditionMessage(error_sp_index),
                    fixed = TRUE),
            "out-of-range penalty SP index must fail closed")

bad_sp_order <- batch$setup
bad_sp_order$penalty_sp_indices[1:2] <-
  rev(bad_sp_order$penalty_sp_indices[1:2])
error_sp_order <- tryCatch(
  fastkpc_full_cuda_fixed_sp_native_dto(bad_sp_order), error = identity
)
assert_true(inherits(error_sp_order, "error") &&
              grepl("identity penalty-to-SP mapping",
                    conditionMessage(error_sp_order), fixed = TRUE),
            "non-identity penalty-to-SP mapping must fail closed")

cat("PASS Phase 3 native DTO\n")
```

- [ ] **Step 2: Run the test and verify the missing DTO failure**

Run:

```bash
Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_native_dto.R
```

Expected: FAIL with missing `fastkpc_full_cuda_fixed_sp_native_dto`.

- [ ] **Step 3: Implement the setup DTO**

Add:

```r
fastkpc_full_cuda_fixed_sp_native_dto <- function(setup) {
  if (!is.list(setup) ||
      length(setdiff(c("weights_policy", "offset_policy", "H",
                       "sp_mapping", "min_sp"), names(setup))) > 0L) {
    stop("Phase 3 PreparedSSetup policy fields are incomplete",
         call. = FALSE)
  }
  if (!identical(setup$weights_policy, "none-or-unit")) {
    stop("Phase 3 unsupported weights policy", call. = FALSE)
  }
  if (!identical(setup$offset_policy, "none-or-zero")) {
    stop("Phase 3 unsupported offset policy", call. = FALSE)
  }
  if (!is.null(setup$H)) {
    stop("Phase 3A non-null H is not implemented", call. = FALSE)
  }
  if (!is.null(setup$sp_mapping) || !is.null(setup$min_sp)) {
    stop("Phase 3A smoothing mapping is not implemented", call. = FALSE)
  }
  for (index in seq_along(setup$penalty_blocks)) {
    block_size <- nrow(setup$penalty_blocks[[index]])
    start <- as.integer(setup$penalty_offsets[[index]])
    if (start < 1L || start + block_size - 1L > ncol(setup$X)) {
      stop("Phase 3 penalty offset is out of range", call. = FALSE)
    }
  }
  if (any(setup$penalty_sp_indices < 1L |
          setup$penalty_sp_indices > length(setup$penalty_blocks))) {
    stop("Phase 3 penalty SP index is out of range", call. = FALSE)
  }
  if (!identical(
    as.integer(setup$penalty_sp_indices),
    seq_len(length(setup$penalty_blocks))
  )) {
    stop("Phase 3 v1 requires identity penalty-to-SP mapping",
         call. = FALSE)
  }
  fastkpc_full_cuda_validate_prepared_s_for_adapter(setup)
  list(
    schema_version = "full-cuda-ci-prepared-s-native-dto-v1",
    dataset_sha256 = setup$dataset_sha256,
    prepared_s_key_sha256 = setup$prepared_s_key_sha256,
    same_S_group_id = setup$same_S_group_id,
    phase1_setup_fingerprint = setup$phase1_setup_fingerprint,
    provider_fingerprint = setup$provider_fingerprint,
    semantic_fingerprint = setup$semantic_fingerprint,
    representation_fingerprint = setup$representation_fingerprint,
    n = as.integer(nrow(setup$X)),
    coefficient_dim = as.integer(ncol(setup$X)),
    null_dim = as.integer(setup$constraint_nullspace_dimension),
    penalty_count = as.integer(length(setup$penalty_blocks)),
    X = unclass(setup$X),
    constraint_mode = setup$constraint_mode,
    constraint_nullspace = setup$constraint_nullspace,
    gram_matrix = unclass(setup$gram_matrix),
    nullspace_gram_matrix = setup$nullspace_gram_matrix,
    penalty_blocks = lapply(setup$penalty_blocks, unclass),
    penalty_offsets_zero_based = as.integer(setup$penalty_offsets) - 1L,
    penalty_ranks = as.integer(setup$penalty_ranks),
    penalty_sp_indices_zero_based =
      as.integer(setup$penalty_sp_indices) - 1L,
    penalty_sp_labels = as.character(setup$penalty_sp_labels),
    H = setup$H,
    weights_policy = setup$weights_policy,
    offset_policy = setup$offset_policy
  )
}
```

Before returning the DTO, reconstruct every full coefficient-space penalty
using `penalty_offsets_zero_based + 1L` only inside the R test helper and
compare it with the Phase 2 reconstruction. The test must cover the first and
last blocks of a multi-penalty setup and reject negative offsets, blocks that
run past `coefficient_dim`, and SP indices outside `[0, penalty_count)`. The R
adapter also validates dataset lineage, target/Y hashes, and oracle selected-SP
hashes; the native layer is responsible only for schema, shape, order,
finite/range, route, and device status checks.

- [ ] **Step 4: Implement target batch validation**

Add:

```r
fastkpc_full_cuda_fixed_sp_validate_numeric_inputs <- function(Y, SP) {
  if (!is.matrix(Y) || !is.double(Y) || any(!is.finite(Y)) ||
      !is.matrix(SP) || !is.double(SP) ||
      any(!is.finite(SP)) || any(SP < 0)) {
    stop("Phase 3 native target batch numeric inputs are malformed",
         call. = FALSE)
  }
  invisible(TRUE)
}

fastkpc_full_cuda_fixed_sp_native_batch <- function(batch, dto) {
  Y <- unclass(as.matrix(batch$Y))
  SP <- unclass(as.matrix(batch$SP))
  storage.mode(Y) <- "double"
  storage.mode(SP) <- "double"
  fastkpc_full_cuda_fixed_sp_validate_numeric_inputs(Y, SP)
  target_count <- ncol(Y)
  sp_name_order_exact <- all(vapply(
    batch$target_rows$selected_sp_names,
    function(value) identical(
      as.character(value), as.character(dto$penalty_sp_labels)
    ),
    logical(1L)
  ))
  if (!identical(nrow(Y), dto$n) ||
      !identical(dim(SP), c(dto$penalty_count, target_count)) ||
      !sp_name_order_exact) {
    stop("Phase 3 native target batch is malformed", call. = FALSE)
  }

  expected_y_hashes <- vapply(
    seq_len(target_count),
    function(index) fastkpc_full_cuda_census_metadata_hash(
      as.numeric(Y[, index])
    ),
    character(1L)
  )
  expected_sp_hashes <- vapply(
    seq_len(target_count),
    function(index) fastkpc_full_cuda_census_metadata_hash(
      stats::setNames(as.numeric(SP[, index]), dto$penalty_sp_labels)
    ),
    character(1L)
  )
  if (!identical(expected_y_hashes, as.character(batch$target_rows$y_hash))) {
    stop("Phase 3 Y hash mismatch", call. = FALSE)
  }
  if (!identical(
    expected_sp_hashes, as.character(batch$target_rows$selected_sp_hash)
  )) {
    stop("Phase 3 selected-SP hash mismatch", call. = FALSE)
  }
  list(
    Y = Y, SP = SP,
    planned_route = as.character(batch$planned_route),
    target_keys = as.character(batch$target_rows$residual_key_sha256),
    target_count = as.integer(target_count)
  )
}
```

- [ ] **Step 5: Run DTO and catalog tests**

Run:

```bash
Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_native_dto.R
Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_catalog.R
```

Expected: both PASS.

- [ ] **Step 6: Commit the DTO**

```bash
git add fastkpc/R/full_cuda_ci_fixed_sp_runtime.R \
  fastkpc/tests/test_full_cuda_ci_fixed_sp_native_dto.R
git commit -m "feat: add Phase 3 Prepared-S native DTO"
```

## Task 4: Add Persistent CUDA Runtime Lifecycle

**Files:**
- Create: `fastkpc/src/cuda/mgcv_fixed_sp_runtime_types.hpp`
- Create: `fastkpc/src/cuda/mgcv_fixed_sp_runtime.hpp`
- Create: `fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu`
- Modify: `fastkpc/tools/build_cuda_native.sh`
- Modify: `fastkpc/src/r_api_cuda.cpp`
- Modify: `fastkpc/R/cuda_native.R`
- Create: `fastkpc/tests/test_mgcv_fixed_sp_cuda_runtime_lifecycle.R`

- [ ] **Step 1: Write the failing CUDA lifecycle test**

Create `fastkpc/tests/test_mgcv_fixed_sp_cuda_runtime_lifecycle.R`:

```r
source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP Phase 3 CUDA runtime lifecycle\n")
  quit(save = "no", status = 0)
}

build_fastkpc_cuda_native(rebuild = TRUE)
assert_true(fastkpc_cuda_available(), "CUDA must be available")

runtime <- fixed_sp_cuda_runtime_create(device_id = 0L)
on.exit(try(fixed_sp_cuda_runtime_free(runtime), silent = TRUE), add = TRUE)
before <- fixed_sp_cuda_runtime_info(runtime)
assert_true(before$stream_create_count == 1L, "one stream")
assert_true(before$cublas_handle_create_count == 1L, "one cuBLAS handle")
assert_true(before$cusolver_handle_create_count == 1L, "one cuSOLVER handle")
assert_true(identical(before$cusolver_deterministic_mode, "enabled"),
            "cuSOLVER deterministic mode")
assert_true(identical(before$cublas_math_mode, "pedantic"),
            "cuBLAS pedantic math")
assert_true(identical(before$cublas_atomics_mode, "not_allowed"),
            "cuBLAS atomics disabled")

fixed_sp_cuda_runtime_reserve(
  runtime, n = 351L, null_dim = 64L, target_count = 47L,
  penalty_count = 7L, augmented_rows = 407L
)
after <- fixed_sp_cuda_runtime_info(runtime)
assert_true(after$workspace_reserve_count == 1L, "one reserve")
assert_true(after$workspace_bytes > 0, "workspace allocated")
assert_true(isTRUE(after$cublas_user_workspace_installed),
            "user cuBLAS workspace installed")
assert_true(after$cublas_workspace_bytes >= 16L * 1024L * 1024L,
            "cuBLAS workspace size")
assert_true(after$cublas_workspace_alignment >= 256L,
            "cuBLAS workspace alignment")
assert_true(after$compute_capability_major == 8L &&
              after$compute_capability_minor == 9L &&
              after$sm_count > 0L,
            "declared GPU identity")

fixed_sp_cuda_runtime_free(runtime)
error <- tryCatch(fixed_sp_cuda_runtime_info(runtime), error = identity)
assert_true(inherits(error, "error") &&
              grepl("freed", conditionMessage(error), fixed = TRUE),
            "freed runtime must reject use")

cat("PASS Phase 3 CUDA runtime lifecycle\n")
```

- [ ] **Step 2: Run the test and verify the missing wrapper failure**

Run:

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_runtime_lifecycle.R
```

Expected: FAIL with missing `fixed_sp_cuda_runtime_create`.

- [ ] **Step 3: Define runtime types**

Create `fastkpc/src/cuda/mgcv_fixed_sp_runtime_types.hpp` with the exact enum
and plain diagnostic types:

```cpp
#ifndef FASTKPC_MGCV_FIXED_SP_RUNTIME_TYPES_HPP
#define FASTKPC_MGCV_FIXED_SP_RUNTIME_TYPES_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fastkpc {

enum class FixedSpRoute : int {
  CholeskyBatched = 0,
  AugmentedQr = 1,
  AugmentedSvd = 2
};

enum class FixedSpStatus : int {
  OkCholeskyBatched = 0,
  OkCholeskySingle = 1,
  OkAugmentedQr = 2,
  OkAugmentedSvd = 3,
  ErrNonfiniteInput = 10,
  ErrSpShapeOrOrder = 11,
  ErrRouteMetadata = 12,
  ErrStablePathNotImplemented = 13,
  ErrQrFailed = 14,
  ErrSvdFailed = 15,
  ErrNonfiniteOutput = 16,
  ErrInternalCuda = 17
};

struct FixedSpCapacities {
  int n = 0;
  int null_dim = 0;
  int target_count = 0;
  int penalty_count = 0;
  int augmented_rows = 0;
};

struct FixedSpRuntimeInfo {
  int device_id = -1;
  std::int64_t creator_pid = -1;
  std::uint64_t generation = 0;
  int runtime_context_create_count = 0;
  int stream_create_count = 0;
  int cublas_handle_create_count = 0;
  int cusolver_handle_create_count = 0;
  int workspace_reserve_count = 0;
  int workspace_grow_count = 0;
  int cuda_device_synchronize_count = 0;
  int cholesky_factor_checkpoint_record_count = 0;
  int cholesky_factor_checkpoint_wait_count = 0;
  int cholesky_solve_checkpoint_record_count = 0;
  int cholesky_solve_checkpoint_wait_count = 0;
  std::size_t workspace_bytes = 0;
  std::size_t cublas_workspace_bytes = 0;
  std::size_t cublas_workspace_alignment = 0;
  int cuda_toolkit_version = 0;
  int cuda_driver_version = 0;
  int compute_capability_major = 0;
  int compute_capability_minor = 0;
  int sm_count = 0;
  bool cusolver_deterministic_mode_enabled = false;
  bool cublas_pedantic_math_enabled = false;
  bool cublas_atomics_not_allowed = false;
  bool cublas_user_workspace_installed = false;
  bool freed = false;
};

const char* fixed_sp_status_name(FixedSpStatus status);
const char* fixed_sp_route_name(FixedSpRoute route);

}  // namespace fastkpc

#endif
```

`C_fixed_sp_cuda_runtime_info` maps the queried booleans to the exact R fields
used by the test:

```text
cusolver_deterministic_mode = "enabled"
cublas_math_mode            = "pedantic"
cublas_atomics_mode         = "not_allowed"
cublas_user_workspace_installed = TRUE
```

- [ ] **Step 4: Define the opaque runtime API**

Create `fastkpc/src/cuda/mgcv_fixed_sp_runtime.hpp`:

```cpp
#ifndef FASTKPC_MGCV_FIXED_SP_RUNTIME_HPP
#define FASTKPC_MGCV_FIXED_SP_RUNTIME_HPP

#include "mgcv_fixed_sp_runtime_types.hpp"

#include <memory>

namespace fastkpc {

class CudaRuntimeContext;
class PreparedSGpuHandle;
class DeviceResidualBatch;

std::shared_ptr<CudaRuntimeContext> create_fixed_sp_runtime(int device_id);
void reserve_fixed_sp_runtime(
  const std::shared_ptr<CudaRuntimeContext>& context,
  const FixedSpCapacities& capacities);
FixedSpRuntimeInfo fixed_sp_runtime_info(
  const std::shared_ptr<CudaRuntimeContext>& context);
void free_fixed_sp_runtime(std::shared_ptr<CudaRuntimeContext>* context);

}  // namespace fastkpc

#endif
```

- [ ] **Step 5: Implement RAII context and reserve arenas**

In `fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu`, implement:

```cpp
#include "mgcv_fixed_sp_runtime.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <unistd.h>

#include <algorithm>
#include <mutex>
#include <stdexcept>

namespace fastkpc {

class CudaRuntimeContext {
 public:
  explicit CudaRuntimeContext(int requested_device);
  ~CudaRuntimeContext();
  void reserve(const FixedSpCapacities& capacities);
  FixedSpRuntimeInfo info() const;
  void require_usable() const;

  int device_id = -1;
  std::int64_t creator_pid = -1;
  std::uint64_t generation = 1;
  cudaStream_t stream = nullptr;
  cublasHandle_t blas = nullptr;
  cusolverDnHandle_t solver = nullptr;
  cudaEvent_t cholesky_factor_checkpoint_event = nullptr;
  cudaEvent_t cholesky_solve_checkpoint_event = nullptr;
  double* double_arena = nullptr;
  int* int_arena = nullptr;
  int* host_status_arena = nullptr;
  void** pointer_arena = nullptr;
  void* cublas_workspace = nullptr;
  std::size_t double_capacity = 0;
  std::size_t int_capacity = 0;
  std::size_t host_status_capacity = 0;
  std::size_t pointer_capacity = 0;
  std::size_t cublas_workspace_bytes = 16U * 1024U * 1024U;
  int potrf_lwork = 0;
  FixedSpCapacities capacities;
  FixedSpRuntimeInfo diagnostics;
  bool freed = false;
  mutable std::mutex mutex;
};

// Use local check_cuda/check_cublas/check_cusolver helpers that throw with the
// stage name. Constructor order is device -> stream -> cuBLAS -> cuSOLVER.
// Bind both handles to the stream, set pedantic math, disallow atomics, and
// enable/query deterministic cuSOLVER results. Reserve installs the aligned
// user cuBLAS workspace after cublasSetStream. Destructor frees arenas and
// workspace, destroys handles, then destroys the stream, each at most once.

}  // namespace fastkpc
```

The arena-size calculation must reserve all Phase 3A shared per-batch scratch
for the requested capacities, including `Y`, `SP`, CUDA-built RHS, one system matrix per
target, theta, cuSOLVER work, info, and pointer arrays. Prepared-handle output
buffers are replaced by the leased transient slot in Tasks 5-6. Allocate the
three device slabs, one `cudaMallocHost` compact status arena, and a 16 MiB
`cudaMalloc`-aligned cuBLAS workspace once.
Call `cublasSetStream` before `cublasSetWorkspace`; query and record
cuSOLVER deterministic mode, cuBLAS math/atomics modes, workspace install
success, toolkit/driver versions, compute capability, and SM count. A second
reserve with equal or smaller capacities does not
allocate and does not increment `workspace_grow_count`.

The checked setup calls are:

```cpp
check_cublas(cublasSetStream(blas, stream), "bind cuBLAS stream");
check_cublas(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH),
             "set cuBLAS pedantic math");
check_cublas(cublasSetAtomicsMode(blas, CUBLAS_ATOMICS_NOT_ALLOWED),
             "disable cuBLAS atomics");
check_cusolver(cusolverDnSetStream(solver, stream),
               "bind cuSOLVER stream");
check_cusolver(cusolverDnSetDeterministicMode(
  solver, CUSOLVER_DETERMINISTIC_RESULTS
), "enable deterministic cuSOLVER");
// During reserve, after cudaMalloc(cublas_workspace):
check_cublas(cublasSetStream(blas, stream), "rebind cuBLAS stream");
check_cublas(cublasSetWorkspace(
  blas, cublas_workspace, cublas_workspace_bytes
), "install cuBLAS workspace");
```

Immediately query `cusolverDnGetDeterministicMode`, `cublasGetMathMode`, and
`cublasGetAtomicsMode`; mismatch with the requested values fails runtime
creation. `cudaMalloc` alignment is recorded and must be at least 256 bytes.

Use these checked counts:

```cpp
const std::size_t targets = static_cast<std::size_t>(capacities.target_count);
const std::size_t n = static_cast<std::size_t>(capacities.n);
const std::size_t q = static_cast<std::size_t>(capacities.null_dim);
const std::size_t penalties =
  static_cast<std::size_t>(capacities.penalty_count);
const std::size_t y_count = n * targets;
const std::size_t sp_count = penalties * targets;
const std::size_t rhs_count = q * targets;
const std::size_t system_count = q * q * targets;
const std::size_t theta_count = q * targets;
const std::size_t double_required = y_count + sp_count + rhs_count +
  system_count + theta_count + static_cast<std::size_t>(potrf_lwork);
const std::size_t int_required = targets + 1U;
const std::size_t pointer_required = targets * 2;
```

Query `potrf_lwork` during reserve with a reserve-time `q x q` probe buffer,
then free the probe before allocating the final slabs. Count those operations
as reserve allocations; they are outside the post-warm-up solve gate.

- [ ] **Step 6: Add `.Call` external-pointer wrappers**

In `fastkpc/src/r_api_cuda.cpp`, follow the existing legacy dCov external
pointer pattern. Add a tag symbol, finalizer, extractor, and these entries:

```cpp
extern "C" SEXP C_fixed_sp_cuda_runtime_create(SEXP device_s);
extern "C" SEXP C_fixed_sp_cuda_runtime_reserve(
  SEXP runtime_s, SEXP n_s, SEXP q_s, SEXP targets_s,
  SEXP penalties_s, SEXP augmented_rows_s);
extern "C" SEXP C_fixed_sp_cuda_runtime_info(SEXP runtime_s);
extern "C" SEXP C_fixed_sp_cuda_runtime_free(SEXP runtime_s);
```

Store `new std::shared_ptr<fastkpc::CudaRuntimeContext>(context)` in the
external pointer. The finalizer calls `free_fixed_sp_runtime`, deletes the
heap-allocated `shared_ptr`, clears the pointer, and is idempotent.

Register the four `.Call` functions with argument counts `1`, `6`, `1`, and
`1` in the existing registration table.

- [ ] **Step 7: Add R wrappers**

Append to `fastkpc/R/cuda_native.R`:

```r
fixed_sp_cuda_runtime_create <- function(device_id = 0L) {
  load_fastkpc_cuda_native()
  .Call("C_fixed_sp_cuda_runtime_create", as.integer(device_id),
        PACKAGE = "fastkpc_cuda")
}

fixed_sp_cuda_runtime_reserve <- function(
    runtime, n, null_dim, target_count, penalty_count, augmented_rows) {
  invisible(.Call(
    "C_fixed_sp_cuda_runtime_reserve", runtime, as.integer(n),
    as.integer(null_dim), as.integer(target_count), as.integer(penalty_count),
    as.integer(augmented_rows), PACKAGE = "fastkpc_cuda"
  ))
}

fixed_sp_cuda_runtime_info <- function(runtime) {
  load_fastkpc_cuda_native()
  .Call("C_fixed_sp_cuda_runtime_info", runtime, PACKAGE = "fastkpc_cuda")
}

fixed_sp_cuda_runtime_free <- function(runtime) {
  load_fastkpc_cuda_native()
  invisible(.Call("C_fixed_sp_cuda_runtime_free", runtime,
                  PACKAGE = "fastkpc_cuda"))
}
```

- [ ] **Step 8: Add the CUDA object to the build**

In `fastkpc/tools/build_cuda_native.sh`, add an NVCC compile stanza immediately
before the existing `mgcv_extract_fixed_sp_cuda.cu` stanza:

```sh
"$NVCC" -O3 -arch=sm_89 -Xcompiler -fPIC -std=c++17 \
  $COMMON_INC -c "$ROOT/src/cuda/mgcv_fixed_sp_runtime.cu" \
  -o "$BUILD/mgcv_fixed_sp_runtime.o"
```

Add `"$BUILD/mgcv_fixed_sp_runtime.o"` to the final link list before
`mgcv_extract_fixed_sp_cuda.o`.

- [ ] **Step 9: Clean-build and run the lifecycle test**

Run:

```bash
rm -f fastkpc/build/mgcv_fixed_sp_runtime.o \
  fastkpc/build/mgcv_extract_fixed_sp_cuda.o \
  fastkpc/build/r_api_cuda.o \
  fastkpc/build/fastkpc_cuda.so
bash fastkpc/tools/build_cuda_native.sh
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_runtime_lifecycle.R
```

Expected: native build succeeds and lifecycle test prints PASS.

- [ ] **Step 10: Commit persistent runtime lifecycle**

```bash
git add fastkpc/src/cuda/mgcv_fixed_sp_runtime_types.hpp \
  fastkpc/src/cuda/mgcv_fixed_sp_runtime.hpp \
  fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu \
  fastkpc/src/r_api_cuda.cpp fastkpc/R/cuda_native.R \
  fastkpc/tools/build_cuda_native.sh \
  fastkpc/tests/test_mgcv_fixed_sp_cuda_runtime_lifecycle.R
git commit -m "feat: add persistent fixed-sp CUDA runtime resources"
```

## Task 5: Upload One Prepared-S Setup Once

**Files:**
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_runtime_types.hpp`
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_runtime.hpp`
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu`
- Modify: `fastkpc/src/r_api_cuda.cpp`
- Modify: `fastkpc/R/cuda_native.R`
- Create: `fastkpc/tests/test_mgcv_fixed_sp_cuda_prepared_handle.R`

- [ ] **Step 1: Write the failing prepared-handle test**

Create `fastkpc/tests/test_mgcv_fixed_sp_cuda_prepared_handle.R`:

```r
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP Phase 3 prepared handle\n")
  quit(save = "no", status = 0)
}

build_fastkpc_cuda_native(rebuild = TRUE)
catalog <- fastkpc_full_cuda_open_fixed_sp_catalog(
  file.path("fastkpc", "artifacts", "full_cuda_ci", "oracle_351x48_v1"),
  file.path("fastkpc", "artifacts", "full_cuda_ci",
            "workload_census_351x48_v1"),
  file.path("fastkpc", "artifacts", "full_cuda_ci",
            "prepared_s_contract_v1"),
  file.path("fastkpc", "artifacts", "kpc_tprs_real_zhu",
            "cancer_RD-causalDiscoveryInput.rds")
)
scope <- fastkpc_full_cuda_fixed_sp_scope(catalog, "iteration")
batch <- fastkpc_full_cuda_fixed_sp_batches(catalog, scope)[[1L]]
dto <- fastkpc_full_cuda_fixed_sp_native_dto(batch$setup)

runtime <- fixed_sp_cuda_runtime_create(0L)
on.exit(try(fixed_sp_cuda_runtime_free(runtime), silent = TRUE), add = TRUE)
fixed_sp_cuda_runtime_reserve(runtime, 351L, 64L, 47L, 7L, 407L)
handle <- fixed_sp_cuda_prepared_create(runtime, dto)
on.exit(try(fixed_sp_cuda_prepared_free(handle), silent = TRUE), add = TRUE)
info <- fixed_sp_cuda_prepared_info(handle)

assert_true(identical(info$prepared_s_key_sha256,
                      dto$prepared_s_key_sha256), "PreparedSKey")
assert_true(info$setup_h2d_upload_count == 1L, "one setup upload")
assert_true(info$setup_h2d_bytes > 0, "setup bytes")
assert_true(info$n == dto$n && info$null_dim == dto$null_dim,
            "setup dimensions")
assert_true(!isTRUE(info$output_slot_leased),
            "new prepared handle has a free output slot")

fixed_sp_cuda_prepared_free(handle)
error <- tryCatch(fixed_sp_cuda_prepared_info(handle), error = identity)
assert_true(inherits(error, "error"), "freed prepared handle rejects use")

cat("PASS Phase 3 prepared handle\n")
```

- [ ] **Step 2: Run the test and verify the missing prepared API failure**

Run:

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_prepared_handle.R
```

Expected: FAIL with missing `fixed_sp_cuda_prepared_create`.

- [ ] **Step 3: Add prepared setup request/info structs and API**

Add to the native headers:

```cpp
struct PreparedSHostView {
  std::string dataset_sha256;
  std::string prepared_s_key_sha256;
  std::string same_s_group_id;
  std::string semantic_fingerprint;
  std::string representation_fingerprint;
  int n = 0;
  int coefficient_dim = 0;
  int null_dim = 0;
  int penalty_count = 0;
  const double* X = nullptr;
  const double* Z = nullptr;
  const double* gram = nullptr;
  std::vector<const double*> penalty_blocks;
  std::vector<int> penalty_dimensions;
  std::vector<int> penalty_offsets_zero_based;
  std::vector<int> penalty_ranks;
  std::vector<int> penalty_sp_indices_zero_based;
};

struct PreparedSInfo {
  std::string prepared_s_key_sha256;
  int n = 0;
  int coefficient_dim = 0;
  int null_dim = 0;
  int penalty_count = 0;
  int setup_h2d_upload_count = 0;
  std::size_t setup_h2d_bytes = 0;
  std::uint64_t generation = 0;
  bool output_slot_leased = false;
};

std::shared_ptr<PreparedSGpuHandle> create_prepared_s_gpu(
  const std::shared_ptr<CudaRuntimeContext>& context,
  const PreparedSHostView& setup);
PreparedSInfo prepared_s_gpu_info(
  const std::shared_ptr<PreparedSGpuHandle>& handle);
void free_prepared_s_gpu(std::shared_ptr<PreparedSGpuHandle>* handle);
```

- [ ] **Step 4: Implement setup upload and ownership**

In `mgcv_fixed_sp_runtime.cu`, make `PreparedSGpuHandle` retain a shared
context and allocate/upload only setup state:

```cpp
struct TransientResidualSlot {
  double* coefficients = nullptr;
  double* fitted = nullptr;
  double* residuals = nullptr;
  double* rss = nullptr;
  cudaEvent_t solve_completion_event = nullptr;
  cudaEvent_t consumer_completion_event = nullptr;
  std::uint64_t generation = 0;
  bool leased = false;
  bool consumer_event_registered = false;
};

class PreparedSGpuHandle {
 public:
  std::shared_ptr<CudaRuntimeContext> context;
  std::int64_t creator_pid = -1;
  int device_id = -1;
  std::uint64_t generation = 1;
  std::string prepared_s_key_sha256;
  int n = 0;
  int p = 0;
  int q = 0;
  int penalty_count = 0;
  double* d_X = nullptr;
  double* d_Z = nullptr;
  double* d_X_null = nullptr;
  double* d_gram = nullptr;
  double* d_projected_penalties = nullptr;
  std::shared_ptr<TransientResidualSlot> residual_slot;
  std::size_t setup_h2d_bytes = 0;
  bool freed = false;
};
```

Assemble each full coefficient-space penalty from its Phase 2 block/offset on
the host view, project it through `Z` when constraints are non-identity, and
upload the contiguous `penalty_count x q x q` array. Under the canonical
identity constraint, alias `X_null = X`, omit `d_Z`, and use the Phase 2 Gram
directly. Create one separate `TransientResidualSlot` at handle creation with
coefficients, fitted, residual, and RSS buffers sized to the reserved target
capacity, solve and optional consumer events, generation zero, and lease state
`FREE`. This is a reusable transient-output resource, not part of the static
Prepared-S payload and not a per-target hot-path allocation. Phase 3A does not
build penalty roots yet. Handle creation before a successful runtime reserve
fails closed.

- [ ] **Step 5: Add prepared external-pointer wrappers**

Add `create/info/free` `.Call` wrappers following the runtime pointer pattern.
The create wrapper validates exact DTO field names, finite matrices, dimensions,
SHA-256 strings, identity/constraint semantics, and canonical weights/offset
policies before constructing `PreparedSHostView`. It also requires
`penalty_sp_indices_zero_based[i] == i` for every penalty; native numerical
builders may use penalty ordinal as the SP row only after this validation.

Add R wrappers:

```r
fixed_sp_cuda_prepared_create <- function(runtime, dto) {
  load_fastkpc_cuda_native()
  .Call("C_fixed_sp_cuda_prepared_create", runtime, dto,
        PACKAGE = "fastkpc_cuda")
}

fixed_sp_cuda_prepared_info <- function(handle) {
  .Call("C_fixed_sp_cuda_prepared_info", handle, PACKAGE = "fastkpc_cuda")
}

fixed_sp_cuda_prepared_free <- function(handle) {
  invisible(.Call("C_fixed_sp_cuda_prepared_free", handle,
                  PACKAGE = "fastkpc_cuda"))
}
```

- [ ] **Step 6: Run prepared-handle and lifecycle tests**

Run:

```bash
bash fastkpc/tools/build_cuda_native.sh
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_prepared_handle.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_runtime_lifecycle.R
```

Expected: both PASS.

- [ ] **Step 7: Commit setup upload**

```bash
git add fastkpc/src/cuda/mgcv_fixed_sp_runtime_types.hpp \
  fastkpc/src/cuda/mgcv_fixed_sp_runtime.hpp \
  fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu \
  fastkpc/src/r_api_cuda.cpp fastkpc/R/cuda_native.R \
  fastkpc/tests/test_mgcv_fixed_sp_cuda_prepared_handle.R
git commit -m "feat: upload Prepared-S state to persistent CUDA handles"
```

## Task 6: Fail Closed for Stable-Route Targets

**Files:**
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_runtime_types.hpp`
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_runtime.hpp`
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu`
- Modify: `fastkpc/src/r_api_cuda.cpp`
- Modify: `fastkpc/R/cuda_native.R`
- Create: `fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3a_solve.R`

- [ ] **Step 1: Write a failing route-status test**

Create `fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3a_solve.R`. Open the
iteration catalog exactly as in Task 3, then select one stable target and
assert the milestone-only fail-closed status:

```r
subset_target <- function(batch, index) {
  list(
    setup = batch$setup,
    target_rows = batch$target_rows[index, , drop = FALSE],
    Y = batch$Y[, index, drop = FALSE],
    SP = batch$SP[, index, drop = FALSE],
    oracle_nullspace_rhs =
      batch$oracle_nullspace_rhs[, index, drop = FALSE],
    planned_route = batch$planned_route[[index]],
    prepared_s_key_sha256 = batch$prepared_s_key_sha256
  )
}

stable_batch_index <- which(vapply(batches, function(x) {
  any(x$planned_route != "CHOLESKY_BATCHED")
}, logical(1L)))[[1L]]
stable_target_index <- which(
  batches[[stable_batch_index]]$planned_route != "CHOLESKY_BATCHED"
)[[1L]]
stable_batch <- subset_target(
  batches[[stable_batch_index]], stable_target_index
)
stable_dto <- fastkpc_full_cuda_fixed_sp_native_dto(stable_batch$setup)
stable_native_batch <- fastkpc_full_cuda_fixed_sp_native_batch(
  stable_batch, stable_dto
)

runtime <- fixed_sp_cuda_runtime_create(0L)
on.exit(try(fixed_sp_cuda_runtime_free(runtime), silent = TRUE), add = TRUE)
fixed_sp_cuda_runtime_reserve(runtime, 351L, 64L, 47L, 7L, 407L)
stable_handle <- fixed_sp_cuda_prepared_create(runtime, stable_dto)
on.exit(try(fixed_sp_cuda_prepared_free(stable_handle), silent = TRUE),
        add = TRUE)

stable_result <- fixed_sp_cuda_solve_batch(
  stable_handle, stable_native_batch$Y, stable_native_batch$SP,
  stable_native_batch$planned_route, stable_native_batch$target_keys,
  outputs = c("residuals")
)
stable_info <- fixed_sp_cuda_residual_info(stable_result)
assert_true(all(stable_info$solver_status ==
                  "ERR_STABLE_PATH_NOT_IMPLEMENTED"),
            "Phase 3A stable targets fail closed")
assert_true(stable_info$invalid_output_init_count == 1L,
            "stable-only batch initializes outputs invalid")
assert_true(stable_info$cpu_fallback_count == 0L,
            "no CPU fallback")
fixed_sp_cuda_residual_free(stable_result)
```

- [ ] **Step 2: Run the test and verify missing solve wrappers**

Run:

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3a_solve.R
```

Expected: FAIL with missing `fixed_sp_cuda_solve_batch`.

- [ ] **Step 3: Add batch request/result/token types**

Add:

```cpp
struct FixedSpBatchHostView {
  const double* Y = nullptr;
  const double* SP = nullptr;
  int n = 0;
  int null_dim = 0;
  int penalty_count = 0;
  int target_count = 0;
  std::uint32_t output_mask = 0;
  std::vector<FixedSpRoute> planned_routes;
  std::vector<std::string> target_keys;
};

struct DeviceResidualInfo {
  int n = 0;
  int target_count = 0;
  std::vector<std::string> target_keys;
  std::vector<FixedSpRoute> planned_routes;
  std::vector<FixedSpRoute> executed_routes;
  std::vector<std::string> reroute_reasons;
  std::vector<FixedSpStatus> solver_statuses;
  bool native_batch_call = false;
  bool true_batched_kernel = false;
  int true_batched_target_count = 0;
  int stable_reroute_count = 0;
  int planned_cholesky_target_count = 0;
  int planned_qr_target_count = 0;
  int planned_svd_target_count = 0;
  int executed_cholesky_target_count = 0;
  int executed_qr_target_count = 0;
  int executed_svd_target_count = 0;
  int cholesky_to_svd_count = 0;
  int qr_to_svd_count = 0;
  int output_slot_acquire_count = 0;
  int output_slot_release_count = 0;
  int output_slot_busy_count = 0;
  int stale_token_reject_count = 0;
  int invalid_output_init_count = 0;
  int cpu_fallback_count = 0;
  int unknown_fallback_count = 0;
  int per_target_allocation_count_after_warmup = 0;
  int per_target_handle_create_count = 0;
  int implicit_residual_d2h_count = 0;
  int rhs_device_build_count = 0;
  std::string rhs_authority = "cuda-x0-transpose-y";
  bool full_cuda_data_plane = true;
  int shadow_materialize_call_count = 0;
  int shadow_materialize_target_count = 0;
  std::size_t shadow_d2h_bytes = 0;
  std::uint64_t owner_generation = 0;
  std::uint64_t slot_generation = 0;
};

std::shared_ptr<DeviceResidualBatch> solve_fixed_sp_batch(
  const std::shared_ptr<PreparedSGpuHandle>& handle,
  const FixedSpBatchHostView& batch);
DeviceResidualInfo device_residual_info(
  const std::shared_ptr<DeviceResidualBatch>& token);
void release_device_residual(
  const std::shared_ptr<DeviceResidualBatch>& token);
void register_device_residual_consumer_event(
  const std::shared_ptr<DeviceResidualBatch>& token,
  cudaEvent_t consumer_completion_event);
void free_device_residual(std::shared_ptr<DeviceResidualBatch>* token);
```

In `mgcv_fixed_sp_runtime.cu`, define the private token class as:

```cpp
class DeviceResidualBatch {
 public:
  std::shared_ptr<PreparedSGpuHandle> owner;
  std::int64_t creator_pid = -1;
  int device_id = -1;
  std::uint64_t owner_generation = 0;
  std::uint64_t slot_generation = 0;
  int n = 0;
  int target_count = 0;
  std::vector<std::string> target_keys;
  std::shared_ptr<TransientResidualSlot> slot;
  std::vector<FixedSpRoute> planned_routes;
  std::vector<FixedSpRoute> executed_routes;
  std::vector<std::string> reroute_reasons;
  std::vector<FixedSpStatus> solver_statuses;
  DeviceResidualInfo diagnostics;
  bool lease_released = false;
  bool freed = false;
};
```

- [ ] **Step 4: Implement Phase 3A route gate and token lifecycle**

The solve validates all shapes, requires finite `Y`, and requires every `SP`
value to be finite and `>= 0`, then acquires the handle's
transient output slot. If the prior token has not released its lease or its
registered consumer event is incomplete, fail before any write with
`ERR_OUTPUT_SLOT_BUSY`. On acquisition increment the slot generation and
create a token holding the exclusive lease. For each non-Cholesky planned
route set:

```cpp
status = FixedSpStatus::ErrStablePathNotImplemented;
```

Set `planned_route` from authenticated metadata, leave `executed_route`
unset for the milestone error, and set `reroute_reason = ""`. `release` marks
the lease released only after any registered consumer event has completed;
`free` calls `release` if necessary, then clears the external pointer.

Immediately after acquiring a slot, launch one fixed-order initialization
kernel that writes IEEE quiet NaN to every public coefficient, fitted,
residual, and RSS output element for the requested target count. Increment
`invalid_output_init_count`. Successful routes overwrite their canonical
columns; non-OK columns remain invalid. Record the token completion event even
when the batch contains only Phase 3A stable-path errors.

```cpp
__global__ void initialize_fixed_sp_outputs_invalid(
    double* coefficients, std::size_t coefficient_count,
    double* fitted, std::size_t fitted_count,
    double* residuals, std::size_t residual_count,
    double* rss, std::size_t rss_count) {
  const std::size_t index =
    static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const double invalid = __longlong_as_double(0x7ff8000000000000ULL);
  if (index < coefficient_count) coefficients[index] = invalid;
  if (index < fitted_count) fitted[index] = invalid;
  if (index < residual_count) residuals[index] = invalid;
  if (index < rss_count) rss[index] = invalid;
}
```

Launch enough threads for the maximum of the four requested counts. For a
stable-only Task 6 batch, record the slot completion event after this kernel
and before returning the error-status token.

If this intermediate Task 6 entry point receives a Cholesky target, throw
`Phase 3A Cholesky solve is not implemented` before returning a token. Do not
encode an unfinished safe solve as `ERR_INTERNAL_CUDA`. Do not allocate per
target. Task 6 records completion after invalid initialization for stable-only
batches; Task 7 moves the safe-route completion record after numerical
finalization.

- [ ] **Step 5: Add `.Call` and R wrappers**

Add:

```r
fixed_sp_cuda_solve_batch <- function(
    handle, Y, SP, planned_route, target_keys, outputs = c("residuals")) {
  .Call("C_fixed_sp_cuda_solve_batch", handle, Y, SP,
        as.character(planned_route), as.character(target_keys),
        as.character(outputs),
        PACKAGE = "fastkpc_cuda")
}

fixed_sp_cuda_residual_info <- function(token) {
  .Call("C_fixed_sp_cuda_residual_info", token, PACKAGE = "fastkpc_cuda")
}

fixed_sp_cuda_residual_release <- function(token) {
  invisible(.Call("C_fixed_sp_cuda_residual_release", token,
                  PACKAGE = "fastkpc_cuda"))
}

fixed_sp_cuda_residual_free <- function(token) {
  invisible(.Call("C_fixed_sp_cuda_residual_free", token,
                  PACKAGE = "fastkpc_cuda"))
}
```

`C_fixed_sp_cuda_residual_info` exposes the per-target vectors as
`planned_route`, `executed_route`, `reroute_reason`, and `solver_status`; it
must not publish an ambiguous field named only `route` or `status`.

- [ ] **Step 6: Build and run the route-status test**

Run the same test command. Expected: PASS for the explicit Phase 3A statuses.

- [ ] **Step 7: Commit fail-closed routing**

```bash
git add fastkpc/src/cuda/mgcv_fixed_sp_runtime_types.hpp \
  fastkpc/src/cuda/mgcv_fixed_sp_runtime.hpp \
  fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu \
  fastkpc/src/r_api_cuda.cpp fastkpc/R/cuda_native.R \
  fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3a_solve.R
git commit -m "feat: fail closed for Phase 3A stable fixed-sp routes"
```

## Task 7: Implement Persistent Safe Cholesky and Shadow Materialization

**Files:**
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu`
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_runtime.hpp`
- Modify: `fastkpc/src/r_api_cuda.cpp`
- Modify: `fastkpc/R/cuda_native.R`
- Modify: `fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3a_solve.R`

- [ ] **Step 1: Replace the temporary safe-status assertion with parity gates**

Extend the test's `subset_target()` helper. Select one real iteration
Cholesky target, create its DTO/native batch/handle, then materialize the Phase
2 oracle and assert:

```r
safe_batch_index <- which(vapply(batches, function(x) {
  any(x$planned_route == "CHOLESKY_BATCHED") &&
    any(x$planned_route != "CHOLESKY_BATCHED")
}, logical(1L)))[[1L]]
safe_target_index <- which(
  batches[[safe_batch_index]]$planned_route == "CHOLESKY_BATCHED"
)[[1L]]
safe_batch <- subset_target(batches[[safe_batch_index]], safe_target_index)
safe_dto <- fastkpc_full_cuda_fixed_sp_native_dto(safe_batch$setup)
native_batch <- fastkpc_full_cuda_fixed_sp_native_batch(safe_batch, safe_dto)
handle <- fixed_sp_cuda_prepared_create(runtime, safe_dto)
on.exit(try(fixed_sp_cuda_prepared_free(handle), silent = TRUE), add = TRUE)

token <- fixed_sp_cuda_solve_batch(
  handle, native_batch$Y, native_batch$SP,
  native_batch$planned_route, native_batch$target_keys,
  outputs = c("coefficients", "fitted", "residuals", "rss", "rhs")
)
shadow <- fixed_sp_cuda_materialize_shadow(
  token, outputs = c("coefficients", "fitted", "residuals", "rss", "rhs")
)
info <- fixed_sp_cuda_residual_info(token)
oracle <- fastkpc_mgcv_magic_fixed_sp_from_prepared(
  safe_batch$setup,
  fastkpc_full_cuda_materialize_target_state(
    safe_batch$target_rows, catalog$data,
    safe_batch$setup$dataset_sha256
  )
)
relative_l2 <- function(candidate, reference) {
  sqrt(sum((candidate - reference)^2)) /
    max(sqrt(sum(reference^2)), 1e-300)
}

assert_true(info$solver_status[[1L]] == "OK_CHOLESKY_SINGLE",
            "safe target status")
assert_true(!isTRUE(info$true_batched_kernel) &&
              info$true_batched_target_count == 0L,
            "one target must not claim true batching")
assert_true(max(abs(shadow$residuals[, 1L] - oracle$residuals)) < 1e-7,
            "safe residual max abs parity")
assert_true(max(abs(shadow$fitted[, 1L] - oracle$fitted)) < 1e-7,
            "safe fitted max abs parity")
assert_true(relative_l2(shadow$residuals[, 1L], oracle$residuals) < 1e-7,
            "safe residual relative L2 parity")
assert_true(relative_l2(shadow$fitted[, 1L], oracle$fitted) < 1e-7,
            "safe fitted relative L2 parity")
assert_true(info$implicit_residual_d2h_count == 0L,
            "solve does not implicitly download residuals")
assert_true(info$shadow_materialize_call_count == 1L,
            "explicit shadow materialization counted")
runtime_after_solve <- fixed_sp_cuda_runtime_info(runtime)
assert_true(runtime_after_solve$cholesky_factor_checkpoint_record_count == 1L &&
              runtime_after_solve$cholesky_factor_checkpoint_wait_count == 1L,
            "separate potrf checkpoint")
assert_true(runtime_after_solve$cholesky_solve_checkpoint_record_count == 1L &&
              runtime_after_solve$cholesky_solve_checkpoint_wait_count == 1L,
            "separate potrs scalar checkpoint")
assert_true(identical(info$rhs_authority, "cuda-x0-transpose-y") &&
              isTRUE(info$full_cuda_data_plane),
            "CUDA owns production RHS")
assert_true(max(abs(
  shadow$cuda_nullspace_rhs[, 1L] - safe_batch$oracle_nullspace_rhs[, 1L]
)) < 1e-12, "CUDA RHS shadow parity")
```

Release the first token before calling the same solve again. Assert that the
second solve succeeds and runtime
`workspace_grow_count`, per-target allocation count, stream/handle create
counts, and setup upload count do not increase. Materialize both runs and
require exact same-environment hashes for coefficients, fitted, residuals,
RSS, CUDA RHS, planned/executed routes, reroute reasons, and solver statuses:

```r
first_hash <- fastkpc_full_cuda_census_metadata_hash(list(
  coefficients = shadow$coefficients,
  fitted = shadow$fitted,
  residuals = shadow$residuals,
  rss = shadow$rss,
  rhs = shadow$cuda_nullspace_rhs,
  planned_route = info$planned_route,
  executed_route = info$executed_route,
  reroute_reason = info$reroute_reason,
  solver_status = info$solver_status
))
fixed_sp_cuda_residual_release(token)
same_handle_stable_index <- which(
  batches[[safe_batch_index]]$planned_route != "CHOLESKY_BATCHED"
)[[1L]]
same_handle_stable_batch <- subset_target(
  batches[[safe_batch_index]], same_handle_stable_index
)
same_handle_stable_native <- fastkpc_full_cuda_fixed_sp_native_batch(
  same_handle_stable_batch, safe_dto
)
same_handle_stable_token <- fixed_sp_cuda_solve_batch(
  handle, same_handle_stable_native$Y, same_handle_stable_native$SP,
  same_handle_stable_native$planned_route,
  same_handle_stable_native$target_keys,
  outputs = c("coefficients", "fitted", "residuals", "rss", "rhs")
)
same_handle_stable_shadow <- fixed_sp_cuda_materialize_shadow(
  same_handle_stable_token,
  outputs = c("coefficients", "fitted", "residuals", "rss", "rhs")
)
assert_true(all(is.na(same_handle_stable_shadow$coefficients)) &&
              all(is.na(same_handle_stable_shadow$fitted)) &&
              all(is.na(same_handle_stable_shadow$residuals)) &&
              all(is.na(same_handle_stable_shadow$rss)) &&
              all(is.na(same_handle_stable_shadow$cuda_nullspace_rhs)),
            "stable error token cannot expose prior safe output")
fixed_sp_cuda_residual_release(same_handle_stable_token)
fixed_sp_cuda_residual_free(same_handle_stable_token)
second_token <- fixed_sp_cuda_solve_batch(
  handle, native_batch$Y, native_batch$SP,
  native_batch$planned_route, native_batch$target_keys,
  outputs = c("coefficients", "fitted", "residuals", "rss", "rhs")
)
second_shadow <- fixed_sp_cuda_materialize_shadow(
  second_token,
  outputs = c("coefficients", "fitted", "residuals", "rss", "rhs")
)
second_info <- fixed_sp_cuda_residual_info(second_token)
second_hash <- fastkpc_full_cuda_census_metadata_hash(list(
  coefficients = second_shadow$coefficients,
  fitted = second_shadow$fitted,
  residuals = second_shadow$residuals,
  rss = second_shadow$rss,
  rhs = second_shadow$cuda_nullspace_rhs,
  planned_route = second_info$planned_route,
  executed_route = second_info$executed_route,
  reroute_reason = second_info$reroute_reason,
  solver_status = second_info$solver_status
))
assert_true(identical(first_hash, second_hash),
            "deterministic output and diagnostic hash")
fixed_sp_cuda_residual_release(second_token)
fixed_sp_cuda_residual_free(second_token)
fixed_sp_cuda_residual_free(token)
```

- [ ] **Step 2: Run the test and verify numerical failure**

Expected: FAIL because the safe target still has `ERR_INTERNAL_CUDA` or the
materializer is missing.

- [ ] **Step 3: Implement Phase 3A single-target Cholesky**

Use handle/context arena slices instead of `cudaMalloc`:

```text
d_Y
d_SP
d_rhs                    # computed on CUDA, never copied from host
d_A
d_theta
d_beta
d_fitted
d_residual
d_rss
d_potrf_info
d_potrs_info
```

After the two H2D copies, compute the production RHS from resident `X_null`:

```cpp
const double one = 1.0;
const double zero = 0.0;
check_cublas(cublasDgemm(
  context->blas, CUBLAS_OP_T, CUBLAS_OP_N,
  q, target_count, n, &one,
  handle->d_X_null, n, d_Y, n, &zero, d_rhs, q
), "Phase 3A build CUDA RHS");
```

The canonical Phase 3A corpus has unit/no weights and zero/no offsets. If a
future authenticated setup uses another policy, preprocessing must happen on
CUDA before this GEMM; CPU projected RHS is never a production input. The
shadow materializer may optionally expose `cuda_nullspace_rhs` only for oracle
comparison and must count those D2H bytes separately.

Launch a system-build kernel that computes:

```cpp
A[idx] = gram[idx];
for (int penalty = 0; penalty < penalty_count; ++penalty) {
  A[idx] += sp[penalty] * projected_penalty[penalty * q * q + idx];
}
```

Direct ordinal indexing is valid only because both adapters have already
proved `penalty_sp_indices_zero_based[penalty] == penalty`. Do not silently
accept a non-identity mapping.

Then call the context's existing cuSOLVER handle:

```cpp
int* h_potrf_info = context->host_status_arena;
int* h_potrs_info = context->host_status_arena + 1;
check_cuda(cudaMemsetAsync(
  d_potrf_info, 0, sizeof(int), context->stream
), "zero Phase 3A potrf info");
check_cusolver(
  cusolverDnDpotrf(
    context->solver, CUBLAS_FILL_MODE_UPPER, q, d_A, q,
    d_potrf_work, context->potrf_lwork, d_potrf_info
  ),
  "Phase 3A fixed-sp potrf"
);
check_cuda(cudaMemcpyAsync(
  h_potrf_info, d_potrf_info, sizeof(int), cudaMemcpyDeviceToHost,
  context->stream
), "copy Phase 3A potrf info");
check_cuda(cudaEventRecord(
  context->cholesky_factor_checkpoint_event, context->stream
), "record Phase 3A potrf checkpoint");
check_cuda(cudaEventSynchronize(
  context->cholesky_factor_checkpoint_event
), "wait Phase 3A potrf checkpoint");
if (*h_potrf_info == 0) {
  check_cuda(cudaMemsetAsync(
    d_potrs_info, 0, sizeof(int), context->stream
  ), "zero Phase 3A potrs info");
  check_cusolver(
    cusolverDnDpotrs(
      context->solver, CUBLAS_FILL_MODE_UPPER, q, 1, d_A, q,
      d_theta, q, d_potrs_info
    ),
    "Phase 3A fixed-sp potrs"
  );
  check_cuda(cudaMemcpyAsync(
    h_potrs_info, d_potrs_info, sizeof(int), cudaMemcpyDeviceToHost,
    context->stream
  ), "copy Phase 3A potrs info");
  check_cuda(cudaEventRecord(
    context->cholesky_solve_checkpoint_event, context->stream
  ), "record Phase 3A potrs checkpoint");
  check_cuda(cudaEventSynchronize(
    context->cholesky_solve_checkpoint_event
  ), "wait Phase 3A potrs checkpoint");
}
```

Query `potrf_lwork` during `fixed_sp_cuda_runtime_reserve()` and carve
`d_potrf_work` from the reserved double arena. Do not create or destroy solver
handles. Do not allocate work after reserve.
Use separate compact status cells. Check `potrf` before issuing `potrs`, so a
positive factorization result cannot be overwritten by the solve call. Record
one factor checkpoint and one scalar-solve checkpoint; neither is
`cudaDeviceSynchronize`. After `potrs_info == 0`, launch the batched-shaped
beta/fitted/residual/RSS kernels, then record the transient slot's solve
completion event without waiting. The shadow materializer or future device
consumer performs the wait. Task 3B will widen the grid.

Immediately after the factor checkpoint:

```text
potrf_info > 0:
  planned_route stays CHOLESKY_BATCHED
  executed_route is unset in Phase 3A
  reroute_reason = CHOLESKY_NON_POSITIVE_PIVOT
  solver_status = ERR_STABLE_PATH_NOT_IMPLEMENTED
  cholesky_to_svd_count += 1
  record the token completion event and return without potrs

potrf_info < 0:
  restore the output slot to FREE and throw a batch/API error

potrf_info == 0:
  issue potrs with its separate scalar info cell

potrs_info != 0:
  restore the output slot to FREE and throw a batch/API error

potrs_info == 0:
  launch finalization kernels and record the token completion event
```

Set `solver_status = OK_CHOLESKY_SINGLE`,
`executed_route = CHOLESKY_BATCHED`, and an empty reroute reason. A positive
`potrf` info declares a
Cholesky-to-SVD reroute and returns `ERR_STABLE_PATH_NOT_IMPLEMENTED` in 3A.
A nonzero scalar `potrs` info is an API/batch failure and throws; it is not a
target numerical failure and must not increment a reroute counter.

- [ ] **Step 4: Implement explicit shadow materialization**

Add this result type and native API:

```cpp
struct FixedSpShadowResult {
  int n = 0;
  int coefficient_dim = 0;
  int target_count = 0;
  std::vector<double> coefficients;
  std::vector<double> fitted;
  std::vector<double> residuals;
  std::vector<double> rss;
  std::vector<double> cuda_nullspace_rhs;
};

FixedSpShadowResult materialize_fixed_sp_shadow(
  const std::shared_ptr<DeviceResidualBatch>& token,
  bool include_coefficients,
  bool include_fitted,
  bool include_residuals,
  bool include_rss,
  bool include_rhs);
```

The native API and `.Call` wrapper:

1. validates token tag, PID/device, owner generation, slot generation, and
   unreleased lease;
2. waits on the token event;
3. copies only requested matrices/vectors to R;
4. for every non-OK target, synthesizes explicit `NA_real_` for all requested
   fields, including shadow RHS, without reading that target's device column;
5. increments shadow D2H counters;
6. never changes solve status or ownership.

Add R wrapper:

```r
fixed_sp_cuda_materialize_shadow <- function(
    token, outputs = c("residuals")) {
  allowed <- c("coefficients", "fitted", "residuals", "rss", "rhs")
  if (length(setdiff(outputs, allowed)) > 0L) {
    stop("unknown Phase 3 shadow output", call. = FALSE)
  }
  .Call("C_fixed_sp_cuda_materialize_shadow", token,
        as.character(outputs), PACKAGE = "fastkpc_cuda")
}
```

- [ ] **Step 5: Run the solve, lifecycle, and prepared-handle tests**

Run:

```bash
bash fastkpc/tools/build_cuda_native.sh
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3a_solve.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_runtime_lifecycle.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_prepared_handle.R
```

Expected: all PASS.

- [ ] **Step 6: Commit safe persistent solving**

```bash
git add fastkpc/src/cuda/mgcv_fixed_sp_runtime.hpp \
  fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu \
  fastkpc/src/r_api_cuda.cpp fastkpc/R/cuda_native.R \
  fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3a_solve.R
git commit -m "feat: solve safe fixed-sp targets with persistent CUDA state"
```

## Task 8: Add the Real Iteration-Corpus Phase 3A Gate

**Files:**
- Modify: `fastkpc/R/full_cuda_ci_fixed_sp_runtime.R`
- Create: `fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3a_iteration.R`

- [ ] **Step 1: Write the failing iteration gate**

Create the test to open the iteration catalog, reserve one runtime, stream all
44 setup batches, solve each Cholesky target individually through its prepared
handle, and inspect stable statuses without solving them. Build this summary:

```r
expected <- list(
  setup_count = 44L,
  target_count = 270L,
  cholesky_ok_count = 172L,
  stable_not_implemented_count = 98L,
  residual_max_abs_diff_max = 1e-7,
  residual_relative_l2_diff_max = 1e-7,
  fitted_max_abs_diff_max = 1e-7,
  fitted_relative_l2_diff_max = 1e-7,
  setup_h2d_upload_count = 44L,
  runtime_context_create_count = 1L,
  deterministic_runtime_config_exact = TRUE,
  rhs_authority = "cuda-x0-transpose-y",
  full_cuda_data_plane = TRUE,
  post_warmup_workspace_grow_count = 0L,
  cuda_device_synchronize_count = 0L,
  per_target_allocation_count_after_warmup = 0L,
  cpu_fallback_count = 0L,
  unknown_fallback_count = 0L,
  all_output_slot_leases_released = TRUE,
  invalid_output_init_matches_batch_calls = TRUE,
  persistent_faster_than_repeated_prototype = TRUE
)
```

For each safe target, call the Phase 2 fixed-sp oracle and compare residual and
fitted max absolute and relative-L2 errors using the design-spec denominator.
The test must fail if any non-Cholesky target returns an `OK_*` status in 3A.
For the same ordered 172-target safe corpus, run one untimed warm-up and then
measure three repetitions of the persistent path and the old repeated
single-target CUDA prototype. Compare medians and require:

```text
persistent_safe_elapsed_ms < repeated_single_target_prototype_elapsed_ms
persistent_speedup = prototype / persistent > 1
```

Record both raw repetitions, medians, and GPU identity so this local gate does
not hide timing noise.

- [ ] **Step 2: Run and verify the missing runner failure**

Run:

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3a_iteration.R
```

Expected: FAIL with missing
`fastkpc_run_full_cuda_fixed_sp_phase3a_iteration`.

- [ ] **Step 3: Implement the iteration runner**

Add `fastkpc_run_full_cuda_fixed_sp_phase3a_iteration()` to the R runtime file.
It must:

```text
authenticate/open catalog once
select iteration scope
create/reserve one runtime once
for each batch in PreparedSKey radix order:
  create one prepared handle
  split targets by route
  solve each Cholesky target with target_count=1
  explicitly shadow-materialize and compare to C_magic
  submit all QR/SVD targets once and require ERR_STABLE_PATH_NOT_IMPLEMENTED
  release tokens and prepared handle
return per-target rows plus recomputed summary
```

The runner releases every token before reusing the handle's output slot. It
also runs the frozen performance comparison against the existing repeated
single-target CUDA prototype after numerical parity has passed.

The summary is derived from per-target rows and runtime/prepared info, not from
caller-supplied counters.

- [ ] **Step 4: Run the iteration gate**

Run the test command. Expected: PASS with exact `172/98` status counts and
numeric maxima below `1e-7`.

- [ ] **Step 5: Commit the iteration gate**

```bash
git add fastkpc/R/full_cuda_ci_fixed_sp_runtime.R \
  fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3a_iteration.R
git commit -m "test: gate persistent fixed-sp CUDA on iteration corpus"
```

## Task 9: Harden Lifetime, Generation, PID, and Device Misuse

**Files:**
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu`
- Modify: `fastkpc/src/r_api_cuda.cpp`
- Create: `fastkpc/tests/test_mgcv_fixed_sp_cuda_runtime_misuse.R`

- [ ] **Step 1: Write misuse tests**

Create the test by opening the iteration catalog, selecting one safe target
with the `subset_target()` helper from Task 6, creating `native_batch`, and
reserving one active runtime/handle. Then add these assertions:

```r
expect_error_contains <- function(expr, text) {
  error <- tryCatch(force(expr), error = identity)
  assert_true(inherits(error, "error") &&
                grepl(text, conditionMessage(error), fixed = TRUE), text)
}

freed_runtime <- fixed_sp_cuda_runtime_create(0L)
fixed_sp_cuda_runtime_free(freed_runtime)
expect_error_contains(fixed_sp_cuda_runtime_info(freed_runtime), "freed")

freed_handle <- fixed_sp_cuda_prepared_create(runtime, safe_dto)
fixed_sp_cuda_prepared_free(freed_handle)
expect_error_contains(fixed_sp_cuda_prepared_info(freed_handle), "freed")

first_token <- fixed_sp_cuda_solve_batch(
  handle, native_batch$Y, native_batch$SP,
  native_batch$planned_route, native_batch$target_keys
)
expect_error_contains(
  fixed_sp_cuda_solve_batch(
    handle, native_batch$Y, native_batch$SP,
    native_batch$planned_route, native_batch$target_keys
  ),
  "ERR_OUTPUT_SLOT_BUSY"
)
fixed_sp_cuda_residual_release(first_token)
second_token <- fixed_sp_cuda_solve_batch(
  handle, native_batch$Y, native_batch$SP,
  native_batch$planned_route, native_batch$target_keys
)
expect_error_contains(
  fixed_sp_cuda_materialize_shadow(first_token), "STALE_TOKEN"
)
fixed_sp_cuda_residual_free(first_token)
fixed_sp_cuda_residual_release(second_token)
fixed_sp_cuda_residual_free(second_token)

consumer_token <- fixed_sp_cuda_solve_batch(
  handle, native_batch$Y, native_batch$SP,
  native_batch$planned_route, native_batch$target_keys
)
.Call("C_fixed_sp_cuda_test_register_blocked_consumer", consumer_token,
      PACKAGE = "fastkpc_cuda")
expect_error_contains(
  fixed_sp_cuda_residual_release(consumer_token), "ERR_OUTPUT_SLOT_BUSY"
)
expect_error_contains(
  fixed_sp_cuda_solve_batch(
    handle, native_batch$Y, native_batch$SP,
    native_batch$planned_route, native_batch$target_keys
  ),
  "ERR_OUTPUT_SLOT_BUSY"
)
.Call("C_fixed_sp_cuda_test_complete_consumer", consumer_token,
      PACKAGE = "fastkpc_cuda")
fixed_sp_cuda_residual_release(consumer_token)
fixed_sp_cuda_residual_free(consumer_token)

wrong_tag <- legacy_dcov_spectra_matvec_cuda_handle(diag(2))
on.exit(try(legacy_dcov_spectra_matvec_cuda_handle_free(wrong_tag),
            silent = TRUE), add = TRUE)
expect_error_contains(
  fixed_sp_cuda_runtime_info(wrong_tag),
  "wrong fixed-sp external pointer tag"
)

device_count <- .Call(
  "C_fixed_sp_cuda_test_device_count", PACKAGE = "fastkpc_cuda"
)
if (device_count >= 2L) {
  .Call("C_fixed_sp_cuda_test_set_device", 1L, PACKAGE = "fastkpc_cuda")
  expect_error_contains(fixed_sp_cuda_runtime_info(runtime), "wrong device")
  .Call("C_fixed_sp_cuda_test_set_device", 0L, PACKAGE = "fastkpc_cuda")
}

if (.Platform$OS.type == "unix") {
  child <- parallel::mcparallel(
    tryCatch(fixed_sp_cuda_runtime_info(runtime), error = conditionMessage),
    mc.set.seed = FALSE
  )
  child_result <- parallel::mccollect(child)[[1L]]
  assert_true(grepl("creator PID", child_result, fixed = TRUE),
              "forked CUDA handle must fail before CUDA use")
}
```

Add the two internal device test `.Call` entries shown above. Also add
`C_fixed_sp_cuda_test_register_blocked_consumer` and
`C_fixed_sp_cuda_test_complete_consumer`: the first creates an unrecorded
event and registers it through `register_device_residual_consumer_event`; the
second records that event on the runtime stream and waits for completion.
Every test entry uses the `C_fixed_sp_cuda_test_` prefix and is unavailable
through the production R wrapper surface.

- [ ] **Step 2: Run and verify at least one misuse case fails**

Run:

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_runtime_misuse.R
```

Expected: FAIL before lease, generation, PID, and device checks are implemented.

- [ ] **Step 3: Implement checks on every public entry point**

Before any CUDA call, verify:

```cpp
creator_pid == static_cast<std::int64_t>(getpid())
current_device == stored_device_id
!freed
external pointer tag matches
handle context generation matches
token owner generation matches
token slot generation matches current slot generation for device access
active output-slot lease is absent before solve
registered consumer event is complete before lease release
```

Use exact error substrings asserted by the test. No check may call
`cudaSetDevice` in a forked child before rejecting the PID.

In destructors/finalizers, if `getpid()` differs from `creator_pid`, clear only
the child process's host-side external pointer/shared-pointer wrapper. Do not
call CUDA destroy/free APIs in the forked child; the parent process retains
and later destroys the real CUDA resources.

Add a test-only incomplete consumer event case: register an unrecorded or
not-yet-complete event on the first token, require `residual_release` and the
next solve to preserve `ERR_OUTPUT_SLOT_BUSY`, complete the event, then release
and solve successfully. Generation checks supplement this lease protocol; they
are not the mechanism that protects an executing CUDA consumer.

- [ ] **Step 4: Run misuse and iteration tests**

Run both tests. Expected: PASS.

- [ ] **Step 5: Commit misuse hardening**

```bash
git add fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu \
  fastkpc/src/r_api_cuda.cpp \
  fastkpc/tests/test_mgcv_fixed_sp_cuda_runtime_misuse.R
git commit -m "fix: harden fixed-sp CUDA handle ownership"
```

## Task 10: Phase 3A Verification and Roadmap Record

**Files:**
- Modify: `goal-5.6.md`

- [ ] **Step 1: Run all non-CUDA Phase 3A contract tests**

```bash
Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_catalog.R
Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_native_dto.R
Rscript fastkpc/tests/test_full_cuda_ci_prepared_s_contract.R
Rscript fastkpc/tests/test_full_cuda_ci_target_retarget.R
```

Expected: all PASS.

- [ ] **Step 2: Perform a clean CUDA build**

```bash
rm -f fastkpc/build/*.o fastkpc/build/fastkpc_cuda.so
bash fastkpc/tools/build_cuda_native.sh
```

Expected: command exits zero and its final line ends with
`fastkpc/build/fastkpc_cuda.so`.

- [ ] **Step 3: Run all Phase 3A CUDA tests**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_runtime_lifecycle.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_prepared_handle.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3a_solve.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_runtime_misuse.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3a_iteration.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_extract_gpu_native_fixed_sp.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_extract_gpu_native_batch_bridge.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_extract_gpu_same_setup_batch.R
```

Expected: all PASS; the iteration gate reports exactly 172 safe solves and 98
declared stable-path failures.

- [ ] **Step 4: Run repository hygiene checks**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only the intentional tracked changes and the
pre-existing untracked `fastkpc/artifacts` symlink.

- [ ] **Step 5: Update the roadmap without claiming Phase 3 completion**

In `goal-5.6.md`, record:

```text
Phase 3A complete:
  authenticated Phase 3 catalog and native DTO
  one persistent CUDA stream/cuBLAS/cuSOLVER context
  explicit deterministic cuSOLVER and pedantic/no-atomics cuBLAS config
  one-time Prepared-S uploads
  CUDA-built X0'Y RHS and safe single-target Cholesky
  leased device-resident residual token and explicit shadow materializer
  post-warm-up per-target allocation/handle creation = 0
  persistent path faster than repeated single-target CUDA prototype
  iteration safe targets = 172 with parity < 1e-7
  stable targets = 98 explicit ERR_STABLE_PATH_NOT_IMPLEMENTED

Active next task:
  Phase 3B true same-S multi-target Cholesky
```

- [ ] **Step 6: Commit the Phase 3A closure record**

```bash
git add goal-5.6.md
git commit -m "docs: record full CUDA CI Phase 3A milestone"
```

- [ ] **Step 7: Request code and spec-conformance review**

Run a code-quality review against:

```text
docs/superpowers/specs/2026-07-13-full-cuda-ci-phase3-fixed-sp-runtime-design.md
docs/superpowers/plans/2026-07-13-full-cuda-ci-phase3a-persistent-runtime.md
```

Do not start Phase 3B until review findings are resolved and the verification
commands above are rerun.
