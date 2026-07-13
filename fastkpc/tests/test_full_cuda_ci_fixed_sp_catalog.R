source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_error <- function(expression, message, pattern = NULL) {
  error <- tryCatch({ force(expression); NULL }, error = identity)
  assert_true(inherits(error, "error"), message)
  if (!is.null(pattern)) {
    assert_true(
      grepl(pattern, conditionMessage(error), fixed = TRUE),
      paste0(message, ": unexpected error message")
    )
  }
}
write_json_checked <- function(value, path, message) {
  jsonlite::write_json(value, path, auto_unbox = TRUE, pretty = TRUE)
  assert_true(
    file.exists(path) && !dir.exists(path) && file.info(path)$size > 0,
    message
  )
}

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
    condition = 1, coefficient_rank = 10L, null_dim = 10L,
    authenticated = FALSE
  ),
  "AUGMENTED_SVD"
), "finite unauthenticated conditions must route to SVD")
assert_true(identical(
  fastkpc_full_cuda_fixed_sp_route(
    condition = 1, coefficient_rank = 9L, null_dim = 10L,
    authenticated = TRUE
  ),
  "AUGMENTED_SVD"
), "finite rank-deficient targets must route to SVD")

assert_error(
  fastkpc_full_cuda_fixed_sp_route(
    condition = factor("1"), coefficient_rank = 10L, null_dim = 10L,
    authenticated = TRUE
  ), "factor condition must be rejected",
  "condition must be a bare integer/double vector")
assert_error(
  fastkpc_full_cuda_fixed_sp_route(
    condition = as.Date("2020-01-01"), coefficient_rank = 10L,
    null_dim = 10L, authenticated = TRUE
  ), "Date condition must be rejected",
  "condition must be a bare integer/double vector")
assert_error(
  fastkpc_full_cuda_fixed_sp_route(
    condition = matrix(1, nrow = 1L), coefficient_rank = 10L,
    null_dim = 10L, authenticated = TRUE
  ), "matrix condition must be rejected",
  "condition must be a bare integer/double vector")
assert_error(
  fastkpc_full_cuda_fixed_sp_route(
    condition = "1", coefficient_rank = 10L, null_dim = 10L,
    authenticated = TRUE
  ), "character condition must be rejected",
  "condition must be a bare integer/double vector")
assert_error(
  fastkpc_full_cuda_fixed_sp_route(
    condition = 1, coefficient_rank = 10.5, null_dim = 10L,
    authenticated = TRUE
  ), "fractional coefficient rank must be rejected",
  "coefficient_rank must contain finite nonnegative integer-valued values")
assert_error(
  fastkpc_full_cuda_fixed_sp_route(
    condition = -1, coefficient_rank = 10L, null_dim = 10L,
    authenticated = TRUE
  ), "negative condition must be rejected",
  "condition must have nonnegative finite values")
assert_error(
  fastkpc_full_cuda_fixed_sp_route(
    condition = 1, coefficient_rank = -1L, null_dim = 10L,
    authenticated = TRUE
  ), "negative coefficient rank must be rejected",
  "coefficient_rank must contain finite nonnegative integer-valued values")
assert_error(
  fastkpc_full_cuda_fixed_sp_route(
    condition = 1, coefficient_rank = 10L, null_dim = 10L,
    authenticated = 1
  ), "nonlogical authentication must be rejected",
  "authenticated must be a bare logical vector")
assert_error(
  fastkpc_full_cuda_fixed_sp_route(
    condition = 1, coefficient_rank = 10L, null_dim = 10L,
    authenticated = matrix(TRUE, nrow = 1L)
  ), "matrix authentication must be rejected",
  "authenticated must be a bare logical vector")
assert_error(
  fastkpc_full_cuda_fixed_sp_route(
    condition = c(1, 2), coefficient_rank = c(10L, 10L, 10L),
    null_dim = 10L, authenticated = TRUE
  ), "incompatible non-scalar lengths must be rejected",
  "inputs must be non-empty and scalar or common-length vectors")

contract <- fastkpc_full_cuda_fixed_sp_contract()
assert_true(identical(names(contract), c(
  "schema_version", "native_dto_schema_version", "cholesky_condition_max",
  "svd_condition_min", "route_levels", "target_status_levels",
  "canonical_capacities"
)), "Task 1 fixed-sp contract names remain exact")
assert_true(identical(contract$schema_version,
                      "full-cuda-ci-fixed-sp-runtime-v1"),
            "runtime schema version")
assert_true(identical(contract$native_dto_schema_version,
                      "full-cuda-ci-prepared-s-native-dto-v1"),
            "native DTO schema version")
catalog_contract <- fastkpc_full_cuda_fixed_sp_catalog_contract()
assert_true(identical(
  catalog_contract$schema_version,
  "full-cuda-ci-fixed-sp-catalog-v1"
), "catalog contract schema version")
assert_true(is.character(catalog_contract$phase2_manifest_fields) &&
              is.character(catalog_contract$phase2_summary_fields),
            "catalog contract freezes Phase 2 identity schemas")
assert_true(identical(
  catalog_contract$phase2_source_commit,
  "42ef3efa08327056ffe5c9aad7a8953ff6864c7e"
), "Phase 2 source lineage is frozen")
expected_inherited_setup_fields <- c(
  "same_S_group_id", "S_key", "S_size", "formula_class",
  "representative_residual_key_sha256", "formula_semantics_version",
  "model_matrix_nrow", "model_matrix_ncol", "model_matrix_hash",
  "model_matrix_rank", "model_matrix_condition", "penalty_count",
  "penalty_block_dimensions", "penalty_ranks", "penalty_offsets",
  "penalty_hashes", "penalty_nullity", "constraint_dimensions",
  "constraint_rank", "constraint_nullspace_dimension", "constraint_hash",
  "H_dimensions", "H_hash", "weights_policy", "offset_policy",
  "smooth_classes", "basis_dimensions", "conditioning_rank",
  "conditioning_condition", "near_constant_conditioning_count",
  "setup_fingerprint", "mgcv_version", "R_version"
)
expected_inherited_target_fields <- c(
  "residual_key_sha256", "same_S_group_id", "target", "selected_sp",
  "selected_sp_names", "selected_sp_hash", "fit_time_ms",
  "coefficient_all_finite", "fitted_all_finite", "residual_all_finite",
  "coefficient_hash", "fitted_hash", "residual_hash",
  "target_fit_fingerprint", "setup_fingerprint"
)
expected_setup_scope_fields <- c(
  expected_inherited_setup_fields, "setup_target_count",
  "setup_logical_request_count", "selection_reasons"
)
expected_target_scope_fields <- c(
  "case_type", "residual_key_sha256", "logical_sequence_id",
  "same_S_group_id", "high_condition", "rank_deficient",
  "near_constant_target", "near_constant_conditioner", "multi_penalty",
  "near_alpha", "mgcv_warning", "mgcv_nonconverged",
  "nonfinite_metadata", "condition_bucket", "near_alpha_bucket", "target",
  "S_size", "penalty_count", "condition", "request_multiplicity",
  "same_S_group_size", "setup_target_count", "setup_logical_request_count",
  "representative_residual_key_sha256", "convergence_signature",
  "optimizer_iterations", "selected_sp", "selected_sp_names",
  "selected_sp_hash", "fit_time_ms", "coefficient_all_finite",
  "fitted_all_finite", "residual_all_finite", "coefficient_hash",
  "fitted_hash", "residual_hash", "target_fit_fingerprint",
  "setup_fingerprint", "selection_reasons"
)
assert_true(identical(
  catalog_contract$inherited_setup_fields,
  expected_inherited_setup_fields
), "catalog contract freezes all inherited setup fields")
assert_true(identical(
  catalog_contract$inherited_target_fields,
  expected_inherited_target_fields
), "catalog contract freezes all inherited target fields")
assert_true(identical(
  catalog_contract$setup_scope_fields,
  expected_setup_scope_fields
), "catalog contract freezes the exact setup-scope schema")
assert_true(identical(
  catalog_contract$target_scope_fields,
  expected_target_scope_fields
), "catalog contract freezes the exact target-scope schema")
assert_true(identical(catalog_contract$phase2_file_sha256, c(
  "manifest.json" =
    "755a07e2386279a3de4b72a663a30fced69f91eef7c2ca1f1c05c88288d74c84",
  "summary.json" =
    "27ec14c3a91ce5d6b8fb600ea1da02e1aa131140a24628336a079ea74b45466a",
  "prepared_s_setup_index.csv" =
    "68c8904a39a15423ded120f652ca35e843ade88cd1b0c32f54d7cf3d372855eb",
  "iteration_setup_groups.rds" =
    "2ca2199ad9926517bdafc809be992dccfc95cbc9a1da88a875c66a85a1dba8e8",
  "iteration_target_keys.rds" =
    "599ab3619415a6e3ff50dc6a8ae9aff473e2b31e67005323ebe88e9015bfabe0",
  "qualification_setup_groups.rds" =
    "df2518f9c790997ec31184388fd1b201c3b852c5818241bd37ccc2b7e7334375",
  "qualification_target_keys.rds" =
    "2e9f05d4f5d9d5538cd404f5bb53702c541d340995172b26d544f7949e96bf3f"
)), "Phase 2 immutable file hashes are frozen")
assert_true(identical(contract$cholesky_condition_max, 1e8),
            "Cholesky condition threshold")
assert_true(identical(contract$svd_condition_min, 1e12),
            "SVD condition threshold")
assert_true(identical(
  fastkpc_full_cuda_fixed_sp_route(
    condition = contract$cholesky_condition_max,
    coefficient_rank = 10L, null_dim = 10L, authenticated = TRUE
  ),
  "AUGMENTED_QR"
), "Cholesky threshold belongs to QR interval")
assert_true(identical(
  fastkpc_full_cuda_fixed_sp_route(
    condition = contract$svd_condition_min,
    coefficient_rank = 10L, null_dim = 10L, authenticated = TRUE
  ),
  "AUGMENTED_SVD"
), "SVD threshold belongs to SVD interval")
assert_true(identical(
  contract$route_levels,
  c("CHOLESKY_BATCHED", "AUGMENTED_QR", "AUGMENTED_SVD")
), "route levels")
assert_true(identical(
  contract$target_status_levels,
  c(
    "OK_CHOLESKY_BATCHED", "OK_CHOLESKY_SINGLE",
    "OK_AUGMENTED_QR", "OK_AUGMENTED_SVD",
    "ERR_NONFINITE_INPUT", "ERR_SP_SHAPE_OR_ORDER",
    "ERR_ROUTE_METADATA", "ERR_STABLE_PATH_NOT_IMPLEMENTED",
    "ERR_QR_FAILED", "ERR_SVD_FAILED", "ERR_NONFINITE_OUTPUT",
    "ERR_INTERNAL_CUDA"
  )
), "target status levels")
assert_true(identical(
  contract$canonical_capacities,
  list(
    n = 351L, null_dim = 64L, target_count = 47L,
    penalty_count = 7L, augmented_rows = 407L
  )
), "canonical capacities")

cat("PASS Phase 3 fixed-sp route contract\n")

phase0_dir <- file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "oracle_351x48_v1"
)
phase1_dir <- file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "workload_census_351x48_v1"
)
phase2_dir <- file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1"
)
data_path <- file.path(
  "fastkpc", "artifacts", "kpc_tprs_real_zhu",
  "cancer_RD-causalDiscoveryInput.rds"
)

catalog <- fastkpc_full_cuda_open_fixed_sp_catalog(
  phase0_dir, phase1_dir, phase2_dir, data_path, require_full = TRUE
)
iteration <- fastkpc_full_cuda_fixed_sp_scope(catalog, "iteration")
assert_true(nrow(iteration$setup_rows) == 44L, "iteration setup count")
assert_true(nrow(iteration$target_rows) == 270L, "iteration target count")
assert_true(length(iteration$shard_ids) <= 44L &&
              length(iteration$shard_ids) < 64L,
            "iteration selects a bounded proper shard subset")
assert_true(identical(
  as.integer(table(factor(
    iteration$target_rows$planned_route,
    levels = c("CHOLESKY_BATCHED", "AUGMENTED_QR", "AUGMENTED_SVD")
  ))), c(172L, 31L, 67L)
), "iteration planned routes are in contract order")

qualification <- fastkpc_full_cuda_fixed_sp_scope(catalog, "qualification")
assert_true(nrow(qualification$setup_rows) == 2061L,
            "qualification setup count")
assert_true(nrow(qualification$target_rows) == 6143L,
            "qualification target count")
assert_true(typeof(qualification$shard_ids) == "integer" &&
              length(qualification$shard_ids) > 0L &&
              !anyDuplicated(qualification$shard_ids) &&
              all(qualification$shard_ids >= 0L) &&
              all(qualification$shard_ids < catalog$catalog_contract$shard_count),
            "qualification shard ids are valid")

bad_manifest <- catalog$phase2_manifest
bad_manifest$source_commit <- paste(rep("0", 40L), collapse = "")
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_phase2_identity(
    catalog$phase2_summary, bad_manifest, catalog$inputs
  ), "focused helper must reject a different valid source commit",
  "Phase 2 Prepared-S identity does not match authenticated inputs"
)
bad_manifest <- catalog$phase2_manifest
bad_manifest$selected_group_count <- 8634.5
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_phase2_identity(
    catalog$phase2_summary, bad_manifest, catalog$inputs
  ), "focused helper must reject a fractional selected-group count",
  "Phase 2 Prepared-S identity does not match authenticated inputs"
)
bad_summary <- catalog$phase2_summary
bad_summary$selected_group_count <- 1L
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_phase2_identity(
    bad_summary, catalog$phase2_manifest, catalog$inputs
  ), "focused helper must reject a summary count disagreement",
  "Phase 2 Prepared-S identity does not match authenticated inputs"
)
bad_manifest <- catalog$phase2_manifest
bad_manifest$semantic_file_sha256$iteration_target_keys_rds <-
  paste(rep("0", 64L), collapse = "")
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_phase2_identity(
    catalog$phase2_summary, bad_manifest, catalog$inputs
  ), "focused helper must reject a self-issued semantic hash",
  "Phase 2 Prepared-S semantic manifest is invalid"
)
bad_manifest <- catalog$phase2_manifest
bad_manifest$unexpected <- TRUE
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_phase2_identity(
    catalog$phase2_summary, bad_manifest, catalog$inputs
  ), "focused helper must reject an unexpected manifest field",
  "Phase 2 Prepared-S identity does not match authenticated inputs"
)
bad_manifest <- catalog$phase2_manifest
bad_manifest$run_scope <- factor("full")
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_phase2_identity(
    catalog$phase2_summary, bad_manifest, catalog$inputs
  ), "focused helper must reject a factor run scope",
  "Phase 2 Prepared-S identity does not match authenticated inputs"
)
bad_manifest <- catalog$phase2_manifest
bad_manifest$dataset_file_sha256 <- paste(rep("0", 64L), collapse = "")
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_phase2_identity(
    catalog$phase2_summary, bad_manifest, catalog$inputs
  ), "focused helper must reject a replaced dataset file hash",
  "Phase 2 Prepared-S identity does not match authenticated inputs"
)
bad_manifest <- structure(
  catalog$phase2_manifest, class = "forged_phase2_manifest"
)
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_phase2_identity(
    catalog$phase2_summary, bad_manifest, catalog$inputs
  ), "focused helper must reject a classed manifest list",
  "Phase 2 Prepared-S identity does not match authenticated inputs"
)
bad_summary <- structure(
  catalog$phase2_summary, class = "forged_phase2_summary"
)
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_phase2_identity(
    bad_summary, catalog$phase2_manifest, catalog$inputs
  ), "focused helper must reject a classed summary list",
  "Phase 2 Prepared-S identity does not match authenticated inputs"
)
bad_summary <- catalog$phase2_summary
bad_summary$run_scope <- factor("full")
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_phase2_identity(
    bad_summary, catalog$phase2_manifest, catalog$inputs
  ), "focused helper must reject a factor summary run scope",
  "Phase 2 Prepared-S identity does not match authenticated inputs"
)
bad_summary <- catalog$phase2_summary
bad_summary$parity_scope <- factor("qualification")
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_phase2_identity(
    bad_summary, catalog$phase2_manifest, catalog$inputs
  ), "focused helper must reject a factor summary parity scope",
  "Phase 2 Prepared-S identity does not match authenticated inputs"
)
bad_inputs <- catalog$inputs
class(bad_inputs$manifest$canonical_logical_census_hash) <-
  "forged_canonical_census_sha"
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_phase2_identity(
    catalog$phase2_summary, catalog$phase2_manifest, bad_inputs
  ), "focused helper must reject a classed canonical census SHA",
  "Phase 2 Prepared-S identity does not match authenticated inputs"
)
bad_manifest <- catalog$phase2_manifest
class(bad_manifest$R_version) <- "forged_R_version"
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_phase2_identity(
    catalog$phase2_summary, bad_manifest, catalog$inputs
  ), "focused helper must reject a classed R version",
  "Phase 2 Prepared-S identity does not match authenticated inputs"
)
bad_summary <- catalog$phase2_summary
bad_summary$shard_rds_size_bytes$shard_0 <- list(1L)
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_phase2_identity(
    bad_summary, catalog$phase2_manifest, catalog$inputs
  ), "focused helper must reject a nested shard-size value",
  "Phase 2 Prepared-S identity does not match authenticated inputs"
)
bad_summary <- catalog$phase2_summary
names(bad_summary$shard_rds_size_bytes)[[1L]] <- "wrong_shard"
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_phase2_identity(
    bad_summary, catalog$phase2_manifest, catalog$inputs
  ), "focused helper must reject a wrong shard-size name",
  "Phase 2 Prepared-S identity does not match authenticated inputs"
)
bad_summary <- catalog$phase2_summary
bad_summary$shard_rds_size_bytes$shard_63 <- NULL
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_phase2_identity(
    bad_summary, catalog$phase2_manifest, catalog$inputs
  ), "focused helper must reject a missing shard-size name",
  "Phase 2 Prepared-S identity does not match authenticated inputs"
)
bad_summary <- catalog$phase2_summary
bad_summary$shard_rds_size_bytes$shard_64 <- 1L
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_phase2_identity(
    bad_summary, catalog$phase2_manifest, catalog$inputs
  ), "focused helper must reject an extra shard-size name",
  "Phase 2 Prepared-S identity does not match authenticated inputs"
)
bad_summary <- catalog$phase2_summary
bad_summary$shard_rds_size_bytes <-
  rev(bad_summary$shard_rds_size_bytes)
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_phase2_identity(
    bad_summary, catalog$phase2_manifest, catalog$inputs
  ), "focused helper must reject reordered shard-size names",
  "Phase 2 Prepared-S identity does not match authenticated inputs"
)
bad_summary <- catalog$phase2_summary
class(bad_summary$stage_timing_total_seconds) <- "forged_timing"
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_phase2_identity(
    bad_summary, catalog$phase2_manifest, catalog$inputs
  ), "focused helper must reject a classed stage timing",
  "Phase 2 Prepared-S identity does not match authenticated inputs"
)
bad_summary <- catalog$phase2_summary
bad_summary$elapsed_seconds <- as.character(bad_summary$elapsed_seconds)
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_phase2_identity(
    bad_summary, catalog$phase2_manifest, catalog$inputs
  ), "focused helper must reject a character elapsed timing",
  "Phase 2 Prepared-S identity does not match authenticated inputs"
)
bad_summary <- catalog$phase2_summary
bad_summary$stage_timing_total_seconds <- -1
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_phase2_identity(
    bad_summary, catalog$phase2_manifest, catalog$inputs
  ), "focused helper must reject a negative stage timing",
  "Phase 2 Prepared-S identity does not match authenticated inputs"
)
bad_summary <- catalog$phase2_summary
bad_summary$elapsed_seconds <- Inf
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_phase2_identity(
    bad_summary, catalog$phase2_manifest, catalog$inputs
  ), "focused helper must reject a nonfinite elapsed timing",
  "Phase 2 Prepared-S identity does not match authenticated inputs"
)
captured <- fastkpc_full_cuda_fixed_sp_capture_phase2_files(phase2_dir)
captured$source_path <- file.path(tempdir(), "missing-phase2")
captured_parsed <- fastkpc_full_cuda_fixed_sp_parse_phase2_files(captured)
assert_true(identical(captured_parsed$manifest, catalog$phase2_manifest) &&
              identical(captured_parsed$scopes$iteration$setup_rows,
                        catalog$scopes$iteration$setup_rows),
            "captured Phase 2 bytes parse without reopening their paths")

local({
  temp_phase2 <- tempfile("fixed-sp-file-contract-")
  assert_true(
    dir.create(temp_phase2),
    "temporary Phase 2 fixture directory must be created"
  )
  on.exit(unlink(temp_phase2, recursive = TRUE, force = TRUE), add = TRUE)
  phase2_names <- names(catalog_contract$phase2_file_sha256)
  copied <- file.copy(
    file.path(phase2_dir, phase2_names),
    file.path(temp_phase2, phase2_names)
  )
  assert_true(
    length(copied) == length(phase2_names) && all(copied %in% TRUE),
    "all consumed Phase 2 fixture files must copy"
  )
  manifest_path <- file.path(temp_phase2, "manifest.json")
  tampered_manifest <- fastkpc_full_cuda_fixed_sp_read_json(manifest_path)
  tampered_manifest$source_commit <- paste(rep("0", 40L), collapse = "")
  write_json_checked(
    tampered_manifest, manifest_path,
    "tampered Phase 2 manifest write must succeed"
  )
  assert_error(
    fastkpc_full_cuda_fixed_sp_validate_phase2_files(temp_phase2),
    "immutable Phase 2 contract must reject a rewritten manifest",
    "Phase 2 immutable file hash mismatch"
  )
})

catalog_without_path <- catalog
catalog_without_path$phase2_dir <- file.path(tempdir(), "missing-phase2")
assert_true(identical(
  fastkpc_full_cuda_fixed_sp_scope(catalog_without_path, "iteration")$setup_rows,
  iteration$setup_rows
), "scope must consume authenticated in-memory semantic objects")

corrupt_catalog <- catalog
corrupt_catalog$scopes$iteration$setup_rows$constraint_nullspace_dimension[[1L]] <-
  1.5
assert_error(
  fastkpc_full_cuda_fixed_sp_scope(corrupt_catalog, "iteration"),
  "scope must reject a fractional authenticated null dimension",
  "fixed-sp scope selection is malformed"
)
corrupt_catalog <- catalog
corrupt_catalog$scopes$iteration$setup_rows$setup_fingerprint[[1L]] <-
  paste(rep("0", 64L), collapse = "")
assert_error(
  fastkpc_full_cuda_fixed_sp_scope(corrupt_catalog, "iteration"),
  "scope must reject a mismatched setup fingerprint",
  "fixed-sp setup lineage is inconsistent"
)
corrupt_catalog <- catalog
corrupt_catalog$scopes$iteration$setup_rows$model_matrix_condition <-
  as.character(
    corrupt_catalog$scopes$iteration$setup_rows$model_matrix_condition
  )
assert_error(
  fastkpc_full_cuda_fixed_sp_scope(corrupt_catalog, "iteration"),
  "scope must reject character model-matrix condition lineage",
  "fixed-sp setup lineage is inconsistent"
)
corrupt_catalog <- catalog
corrupt_catalog$scopes$iteration$setup_rows$H_hash[[1L]] <-
  paste(rep("0", 64L), collapse = "")
assert_error(
  fastkpc_full_cuda_fixed_sp_scope(corrupt_catalog, "iteration"),
  "scope must reject a forged inherited setup field",
  "fixed-sp setup lineage is inconsistent"
)
corrupt_catalog <- catalog
corrupt_catalog$scopes$iteration$target_rows$residual_key_sha256 <- factor(
  corrupt_catalog$scopes$iteration$target_rows$residual_key_sha256
)
assert_error(
  fastkpc_full_cuda_fixed_sp_scope(corrupt_catalog, "iteration"),
  "scope must reject a factor residual key SHA",
  "fixed-sp scope selection is malformed"
)
corrupt_catalog <- catalog
corrupt_catalog$scopes$iteration$target_rows$selected_sp_hash[[1L]] <-
  paste(rep("0", 64L), collapse = "")
assert_error(
  fastkpc_full_cuda_fixed_sp_scope(corrupt_catalog, "iteration"),
  "scope must reject a forged inherited target field",
  "fixed-sp target lineage is inconsistent"
)
corrupt_catalog <- catalog
orphan_setup_group <-
  corrupt_catalog$scopes$iteration$setup_rows$same_S_group_id[[1L]]
corrupt_catalog$scopes$iteration$target_rows <-
  corrupt_catalog$scopes$iteration$target_rows[
    corrupt_catalog$scopes$iteration$target_rows$same_S_group_id !=
      orphan_setup_group,
    , drop = FALSE
  ]
assert_error(
  fastkpc_full_cuda_fixed_sp_scope(corrupt_catalog, "iteration"),
  "scope must reject an orphan setup with no target",
  "fixed-sp target lineage is inconsistent"
)
corrupt_catalog <- catalog
orphan_target_group <-
  corrupt_catalog$scopes$iteration$target_rows$same_S_group_id[[1L]]
corrupt_catalog$scopes$iteration$setup_rows <-
  corrupt_catalog$scopes$iteration$setup_rows[
    corrupt_catalog$scopes$iteration$setup_rows$same_S_group_id !=
      orphan_target_group,
    , drop = FALSE
  ]
assert_error(
  fastkpc_full_cuda_fixed_sp_scope(corrupt_catalog, "iteration"),
  "scope must reject an orphan target with no setup",
  "fixed-sp target lineage is inconsistent"
)
corrupt_catalog <- catalog
corrupt_catalog$scopes$iteration$setup_rows$unexpected <- TRUE
assert_error(
  fastkpc_full_cuda_fixed_sp_scope(corrupt_catalog, "iteration"),
  "scope must reject an extra setup-scope column",
  "fixed-sp scope selection is malformed"
)
corrupt_catalog <- catalog
corrupt_catalog$scopes$iteration$target_rows <-
  corrupt_catalog$scopes$iteration$target_rows[
    rev(names(corrupt_catalog$scopes$iteration$target_rows))
  ]
assert_error(
  fastkpc_full_cuda_fixed_sp_scope(corrupt_catalog, "iteration"),
  "scope must reject reordered target-scope columns",
  "fixed-sp scope selection is malformed"
)
corrupt_catalog <- catalog
corrupt_target_index <- match(
  iteration$target_rows$residual_key_sha256[[1L]],
  corrupt_catalog$inputs$target_fit_metadata$residual_key_sha256
)
corrupt_catalog$inputs$target_fit_metadata$same_S_group_id[
  corrupt_target_index
] <- paste(rep("0", 64L), collapse = "")
assert_error(
  fastkpc_full_cuda_fixed_sp_scope(corrupt_catalog, "iteration"),
  "scope must reject inconsistent matched Phase 1 target lineage",
  "fixed-sp target lineage is inconsistent"
)

local({
  temp_phase0 <- tempfile("fixed-sp-catalog-phase0-")
  assert_true(
    dir.create(temp_phase0),
    "temporary Phase 0 fixture directory must be created"
  )
  on.exit(unlink(temp_phase0, recursive = TRUE, force = TRUE), add = TRUE)
  phase0_inputs <- fastkpc_full_cuda_census_input_paths(phase0_dir, data_path)
  phase0_paths <- unname(phase0_inputs[startsWith(
    names(phase0_inputs), "oracle/"
  )])
  phase0_names <- basename(phase0_paths)
  copied <- file.copy(
    phase0_paths, file.path(temp_phase0, phase0_names)
  )
  assert_true(
    length(copied) == length(phase0_paths) && all(copied %in% TRUE),
    "all consumed Phase 0 fixture files must copy"
  )
  manifest_path <- file.path(temp_phase0, "manifest.json")
  tampered_phase0_manifest <- fastkpc_full_cuda_fixed_sp_read_json(
    manifest_path
  )
  tampered_phase0_manifest$source_commit <- paste(rep("0", 40L), collapse = "")
  write_json_checked(
    tampered_phase0_manifest, manifest_path,
    "tampered Phase 0 manifest write must succeed"
  )
  assert_error(
    fastkpc_full_cuda_open_fixed_sp_catalog(
      temp_phase0, phase1_dir, phase2_dir, data_path, require_full = TRUE
    ), "catalog must authenticate the Phase 0 manifest",
    "Phase 1 input hash mismatch: oracle/manifest.json"
  )
})
assert_error(
  fastkpc_full_cuda_open_fixed_sp_catalog("", "", "", "", require_full = FALSE),
  "v1 catalog must reject non-full artifacts",
  "fixed-sp catalog only supports the full Phase 2 artifact"
)
assert_error(
  fastkpc_full_cuda_open_fixed_sp_catalog("", "", "", "",
                                          require_full = matrix(TRUE, 1L, 1L)),
  "v1 catalog must reject matrix require_full",
  "require_full must be a logical scalar"
)

if (identical(Sys.getenv("FASTKPC_CATALOG_FOCUSED_ONLY"), "1")) {
  cat("PASS focused Phase 3 catalog validation probes\n")
  quit(save = "no", status = 0L)
}

batches <- fastkpc_full_cuda_fixed_sp_batches(catalog, iteration)
assert_true(length(batches) == 44L, "iteration batch count")
assert_true(identical(names(batches), iteration$setup_rows$prepared_s_key_sha256),
            "iteration batch setup order")
assert_true(sum(vapply(batches, function(batch) ncol(batch$Y), integer(1))) ==
              270L,
            "iteration batch target-column count")
assert_true(all(vapply(batches, function(batch) {
  nrow(batch$SP) == length(batch$setup$penalty_blocks) &&
    ncol(batch$SP) == ncol(batch$Y) &&
    ncol(batch$oracle_nullspace_rhs) == ncol(batch$Y) &&
    identical(colnames(batch$Y), batch$target_rows$residual_key_sha256) &&
    identical(colnames(batch$SP), batch$target_rows$residual_key_sha256) &&
    identical(colnames(batch$oracle_nullspace_rhs),
              batch$target_rows$residual_key_sha256)
}, logical(1))), "iteration SP and RHS dimensions")

qualification_one_setup <- qualification
qualification_one_setup$setup_rows <- qualification$setup_rows[1L, , drop = FALSE]
qualification_one_setup$target_rows <- qualification$target_rows[
  qualification$target_rows$prepared_s_key_sha256 ==
    qualification_one_setup$setup_rows$prepared_s_key_sha256[[1L]],
  , drop = FALSE
]
qualification_rank <- match(
  qualification_one_setup$setup_rows$prepared_s_key_sha256,
  catalog$setup_index$prepared_s_key_sha256
)
qualification_one_setup$shard_ids <- as.integer(
  (qualification_rank - 1L) %% catalog$catalog_contract$shard_count
)
qualification_batches <- fastkpc_full_cuda_fixed_sp_batches(
  catalog, qualification_one_setup
)
assert_true(length(qualification_batches) == 1L &&
              identical(names(qualification_batches),
                        qualification_one_setup$setup_rows$prepared_s_key_sha256),
            "one-setup qualification batch count and name")
assert_true(identical(
  colnames(qualification_batches[[1L]]$Y),
  qualification_one_setup$target_rows$residual_key_sha256
) && identical(
  colnames(qualification_batches[[1L]]$SP),
  qualification_one_setup$target_rows$residual_key_sha256
) && identical(
  colnames(qualification_batches[[1L]]$oracle_nullspace_rhs),
  qualification_one_setup$target_rows$residual_key_sha256
), "one-setup qualification target column order")
assert_error(
  fastkpc_full_cuda_fixed_sp_scope(catalog, "full"),
  "full scope must be explicitly rejected",
  "full scope streaming is introduced in the closure plan"
)

cat("PASS authenticated Phase 3 iteration catalog\n")
