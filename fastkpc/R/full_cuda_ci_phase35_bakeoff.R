.fastkpc_full_cuda_phase35_exact_batch_is_sha256 <- function(value) {
  is.character(value) && length(value) == 1L && !is.na(value) &&
    grepl("^[0-9a-f]{64}$", value)
}

fastkpc_full_cuda_phase35_exact_batch_request <- function(
    expected_prepared_s_key_sha256,
    target_keys,
    logical_sequence_ids,
    left_target_ordinals,
    right_target_ordinals,
    alpha = 0.1,
    component_capacity = NULL) {
  if (!exists("fastkpc_full_cuda_phase35_sha256_utf8", mode = "function",
              inherits = TRUE)) {
    stop("source full_cuda_ci_phase35_contracts.R before building a batch request",
         call. = FALSE)
  }
  if (!.fastkpc_full_cuda_phase35_exact_batch_is_sha256(
        expected_prepared_s_key_sha256)) {
    stop("expected PreparedSKey must be a lowercase SHA-256", call. = FALSE)
  }
  if (!is.character(target_keys) || length(target_keys) < 2L ||
      anyNA(target_keys) || anyDuplicated(target_keys) ||
      any(!vapply(target_keys,
                  .fastkpc_full_cuda_phase35_exact_batch_is_sha256,
                  logical(1L)))) {
    stop("target_keys must be distinct lowercase SHA-256 strings",
         call. = FALSE)
  }
  pair_count <- length(logical_sequence_ids)
  if (!is.double(logical_sequence_ids) || pair_count < 1L ||
      !is.null(attributes(logical_sequence_ids)) ||
      anyNA(logical_sequence_ids) || any(!is.finite(logical_sequence_ids)) ||
      any(logical_sequence_ids < 1 | logical_sequence_ids > 2^53 - 1) ||
      any(logical_sequence_ids != floor(logical_sequence_ids)) ||
      any(diff(logical_sequence_ids) <= 0)) {
    stop("logical_sequence_ids must be strictly increasing safe-53-bit doubles",
         call. = FALSE)
  }
  if (!is.integer(left_target_ordinals) ||
      !is.integer(right_target_ordinals) ||
      length(left_target_ordinals) != pair_count ||
      length(right_target_ordinals) != pair_count ||
      !is.null(attributes(left_target_ordinals)) ||
      !is.null(attributes(right_target_ordinals)) ||
      anyNA(left_target_ordinals) || anyNA(right_target_ordinals) ||
      any(left_target_ordinals < 1L | left_target_ordinals > length(target_keys)) ||
      any(right_target_ordinals < 1L |
          right_target_ordinals > length(target_keys)) ||
      any(left_target_ordinals == right_target_ordinals)) {
    stop("exact batch target ordinals are invalid", call. = FALSE)
  }
  if (!is.double(alpha) || length(alpha) != 1L || is.na(alpha) ||
      !is.finite(alpha) || !identical(alpha, 0.1) ||
      !is.null(attributes(alpha))) {
    stop("exact batch requires bare canonical alpha 0.1", call. = FALSE)
  }
  referenced <- sort(unique(c(left_target_ordinals, right_target_ordinals)),
                     method = "radix")
  if (is.null(component_capacity)) {
    component_capacity <- as.integer(length(referenced))
  }
  if (!is.integer(component_capacity) || length(component_capacity) != 1L ||
      is.na(component_capacity) || !is.null(attributes(component_capacity)) ||
      component_capacity < length(referenced) ||
      component_capacity > length(target_keys)) {
    stop("component_capacity cannot hold the referenced target set",
         call. = FALSE)
  }

  payload <- paste(c(
    "schema_version=full-cuda-ci-phase35-exact-batch-request-v1",
    paste0("expected_prepared_s_key_sha256=",
           expected_prepared_s_key_sha256),
    paste0("component_capacity=", component_capacity),
    paste0("target_count=", length(target_keys)),
    paste0("target_key=", target_keys),
    paste0("pair_count=", pair_count),
    paste0(
      "pair=", sprintf("%.0f", logical_sequence_ids), "|",
      target_keys[left_target_ordinals], "|",
      target_keys[right_target_ordinals], "|0.1"
    )
  ), collapse = "\n")
  list(
    schema_version = "full-cuda-ci-phase35-exact-batch-request-v1",
    expected_prepared_s_key_sha256 = expected_prepared_s_key_sha256,
    request_identity_sha256 =
      fastkpc_full_cuda_phase35_sha256_utf8(payload),
    logical_sequence_ids = logical_sequence_ids,
    left_target_ordinals = left_target_ordinals,
    right_target_ordinals = right_target_ordinals,
    alpha = alpha,
    component_capacity = component_capacity
  )
}

fastkpc_full_cuda_phase35_exact_batch_request_from_logical <- function(
    expected_prepared_s_key_sha256, target_keys, logical_rows,
    component_capacity = NULL) {
  required <- c(
    "logical_sequence_id", "residual_key_x", "residual_key_y", "alpha"
  )
  if (!is.data.frame(logical_rows) || nrow(logical_rows) < 1L ||
      length(setdiff(required, names(logical_rows))) > 0L ||
      anyNA(logical_rows[required]) ||
      !identical(logical_rows$logical_sequence_id,
                 sort(logical_rows$logical_sequence_id, method = "radix")) ||
      anyDuplicated(logical_rows$logical_sequence_id) ||
      any(logical_rows$alpha != 0.1)) {
    stop("logical_rows are not an ordered canonical exact-batch slice",
         call. = FALSE)
  }
  left <- match(logical_rows$residual_key_x, target_keys)
  right <- match(logical_rows$residual_key_y, target_keys)
  if (anyNA(left) || anyNA(right)) {
    stop("logical pair endpoints are absent from target_keys", call. = FALSE)
  }
  fastkpc_full_cuda_phase35_exact_batch_request(
    expected_prepared_s_key_sha256 = expected_prepared_s_key_sha256,
    target_keys = target_keys,
    logical_sequence_ids = as.double(logical_rows$logical_sequence_id),
    left_target_ordinals = as.integer(left),
    right_target_ordinals = as.integer(right),
    alpha = 0.1,
    component_capacity = component_capacity
  )
}

fastkpc_full_cuda_phase35_exact_batch_ci <- function(
    prepared_s, Y, SP, planned_route, target_keys, request) {
  load_fastkpc_cuda_native()
  Y <- matrix(as.double(Y), nrow = nrow(Y), ncol = ncol(Y))
  SP <- matrix(as.double(SP), nrow = nrow(SP), ncol = ncol(SP))
  .Call(
    "C_full_cuda_ci_phase35_exact_batch",
    prepared_s,
    Y,
    SP,
    as.character(planned_route),
    as.character(target_keys),
    request,
    PACKAGE = "fastkpc_cuda"
  )
}

fastkpc_full_cuda_phase35_validate_exact_batch_result <- function(
    result, request, target_keys) {
  compact_fields <- c(
    "logical_sequence_id", "p_value", "status", "solver_route",
    "optimizer_status", "dcov_status", "diagnostic_flags"
  )
  numerical_fields <- c(
    "statistic", "mean", "variance", "gamma_shape", "gamma_scale",
    "gamma_iterations"
  )
  if (!is.list(result) || !identical(names(result), c(
        "schema_version", "request_identity_sha256",
        "prepared_s_key_sha256", "target_keys", "records", "numerical",
        "diagnostics"
      )) ||
      !identical(result$schema_version,
                 "full-cuda-ci-phase35-exact-batch-result-v1") ||
      !identical(result$request_identity_sha256,
                 request$request_identity_sha256) ||
      !identical(result$prepared_s_key_sha256,
                 request$expected_prepared_s_key_sha256) ||
      !identical(result$target_keys, target_keys) ||
      !is.data.frame(result$records) ||
      !identical(names(result$records), compact_fields) ||
      !is.data.frame(result$numerical) ||
      !identical(names(result$numerical), numerical_fields) ||
      nrow(result$records) != length(request$logical_sequence_ids) ||
      nrow(result$numerical) != nrow(result$records) ||
      !identical(result$records$logical_sequence_id,
                 request$logical_sequence_ids) ||
      any(result$records$status != "OK") ||
      any(result$records$dcov_status != "OK_EXACT_CUDA_GAMMA") ||
      any(!is.finite(result$records$p_value)) ||
      any(result$records$p_value < 0 | result$records$p_value > 1) ||
      any(!is.finite(as.matrix(result$numerical)))) {
    stop("exact batch result payload is malformed", call. = FALSE)
  }
  diagnostics <- result$diagnostics
  flags <- c(
    "request_identity_authenticated", "prepared_identity_authenticated",
    "target_identity_authenticated", "residuals_device_resident",
    "components_device_resident", "compact_result_only_d2h",
    "deterministic_logical_order", "component_capacity_respected",
    "bounded_allocation", "leak_free_teardown", "caller_device_restored"
  )
  if (!is.list(diagnostics) || length(setdiff(flags, names(diagnostics))) > 0L ||
      !all(unlist(diagnostics[flags], use.names = FALSE)) ||
      diagnostics$pair_count != nrow(result$records) ||
      diagnostics$pair_evaluation_count != nrow(result$records) ||
      diagnostics$component_capacity != request$component_capacity ||
      diagnostics$component_cache_lookup_count != 2L * nrow(result$records) ||
      diagnostics$component_cache_miss_count !=
        diagnostics$referenced_component_count ||
      diagnostics$component_build_count !=
        diagnostics$referenced_component_count ||
      diagnostics$component_cache_hit_count +
        diagnostics$component_cache_miss_count !=
        diagnostics$component_cache_lookup_count ||
      diagnostics$component_cache_eviction_count != 0L ||
      diagnostics$residual_d2h_bytes != 0 ||
      diagnostics$component_d2h_bytes != 0 ||
      diagnostics$compact_result_d2h_count != 1L ||
      diagnostics$compact_result_d2h_bytes !=
        72 * nrow(result$records) ||
      diagnostics$cpu_dcov_component_count != 0L ||
      diagnostics$cpu_dcov_pair_statistic_count != 0L ||
      diagnostics$cpu_gamma_p_value_count != 0L ||
      diagnostics$device_allocation_count !=
        diagnostics$device_free_count ||
      diagnostics$device_allocation_count != 9L) {
    stop("exact batch structural diagnostics failed", call. = FALSE)
  }
  invisible(TRUE)
}

fastkpc_full_cuda_phase35_legacy_eig_batch_request <- function(
    expected_prepared_s_key_sha256,
    target_keys,
    logical_sequence_ids,
    left_target_ordinals,
    right_target_ordinals,
    alpha = 0.1,
    component_capacity = NULL,
    num_col = 35L) {
  validated <- fastkpc_full_cuda_phase35_exact_batch_request(
    expected_prepared_s_key_sha256 = expected_prepared_s_key_sha256,
    target_keys = target_keys,
    logical_sequence_ids = logical_sequence_ids,
    left_target_ordinals = left_target_ordinals,
    right_target_ordinals = right_target_ordinals,
    alpha = alpha,
    component_capacity = component_capacity
  )
  if (!is.integer(num_col) || length(num_col) != 1L || is.na(num_col) ||
      !is.null(attributes(num_col)) || num_col != 35L) {
    stop("legacy eig batch requires bare canonical num_col 35L",
         call. = FALSE)
  }
  payload <- paste(c(
    "schema_version=full-cuda-ci-phase35-legacy-eig-batch-request-v1",
    paste0("expected_prepared_s_key_sha256=",
           expected_prepared_s_key_sha256),
    paste0("component_capacity=", validated$component_capacity),
    paste0("num_col=", num_col),
    paste0("target_count=", length(target_keys)),
    paste0("target_key=", target_keys),
    paste0("pair_count=", length(logical_sequence_ids)),
    paste0(
      "pair=", sprintf("%.0f", logical_sequence_ids), "|",
      target_keys[left_target_ordinals], "|",
      target_keys[right_target_ordinals], "|0.1"
    )
  ), collapse = "\n")
  list(
    schema_version =
      "full-cuda-ci-phase35-legacy-eig-batch-request-v1",
    expected_prepared_s_key_sha256 = expected_prepared_s_key_sha256,
    request_identity_sha256 =
      fastkpc_full_cuda_phase35_sha256_utf8(payload),
    logical_sequence_ids = logical_sequence_ids,
    left_target_ordinals = left_target_ordinals,
    right_target_ordinals = right_target_ordinals,
    alpha = alpha,
    component_capacity = validated$component_capacity,
    num_col = num_col
  )
}

fastkpc_full_cuda_phase35_legacy_eig_batch_request_from_logical <- function(
    expected_prepared_s_key_sha256, target_keys, logical_rows,
    component_capacity = NULL, num_col = 35L) {
  exact <- fastkpc_full_cuda_phase35_exact_batch_request_from_logical(
    expected_prepared_s_key_sha256 = expected_prepared_s_key_sha256,
    target_keys = target_keys,
    logical_rows = logical_rows,
    component_capacity = component_capacity
  )
  fastkpc_full_cuda_phase35_legacy_eig_batch_request(
    expected_prepared_s_key_sha256 = expected_prepared_s_key_sha256,
    target_keys = target_keys,
    logical_sequence_ids = exact$logical_sequence_ids,
    left_target_ordinals = exact$left_target_ordinals,
    right_target_ordinals = exact$right_target_ordinals,
    alpha = exact$alpha,
    component_capacity = exact$component_capacity,
    num_col = num_col
  )
}

fastkpc_full_cuda_phase35_legacy_eig_batch_ci <- function(
    prepared_s, Y, SP, planned_route, target_keys, request) {
  load_fastkpc_cuda_native()
  Y <- matrix(as.double(Y), nrow = nrow(Y), ncol = ncol(Y))
  SP <- matrix(as.double(SP), nrow = nrow(SP), ncol = ncol(SP))
  .Call(
    "C_full_cuda_ci_phase35_legacy_eig_batch",
    prepared_s,
    Y,
    SP,
    as.character(planned_route),
    as.character(target_keys),
    request,
    PACKAGE = "fastkpc_cuda"
  )
}

fastkpc_full_cuda_phase35_validate_legacy_eig_batch_result <- function(
    result, request, target_keys) {
  compact_fields <- c(
    "logical_sequence_id", "p_value", "status", "solver_route",
    "optimizer_status", "dcov_status", "diagnostic_flags"
  )
  numerical_fields <- c(
    "statistic", "mean", "variance", "gamma_shape", "gamma_scale",
    "gamma_iterations"
  )
  pair_count <- length(request$logical_sequence_ids)
  referenced_count <- length(unique(c(
    request$left_target_ordinals, request$right_target_ordinals
  )))
  malformed <-
    !is.list(result) || !identical(names(result), c(
      "schema_version", "request_identity_sha256",
      "prepared_s_key_sha256", "target_keys", "records", "numerical",
      "diagnostics"
    )) ||
    !identical(result$schema_version,
               "full-cuda-ci-phase35-legacy-eig-batch-result-v1") ||
    !identical(result$request_identity_sha256,
               request$request_identity_sha256) ||
    !identical(result$prepared_s_key_sha256,
               request$expected_prepared_s_key_sha256) ||
    !identical(result$target_keys, target_keys) ||
    !is.data.frame(result$records) ||
    !identical(names(result$records), compact_fields) ||
    !is.data.frame(result$numerical) ||
    !identical(names(result$numerical), numerical_fields) ||
    nrow(result$records) != pair_count ||
    nrow(result$numerical) != pair_count ||
    !identical(result$records$logical_sequence_id,
               request$logical_sequence_ids) ||
    any(result$records$status != "OK") ||
    any(result$records$dcov_status != "OK_EXACT_CUDA_GAMMA") ||
    any(result$records$diagnostic_flags != 4L) ||
    any(!is.finite(result$records$p_value)) ||
    any(result$records$p_value < 0 | result$records$p_value > 1) ||
    any(!is.finite(as.matrix(result$numerical)))
  if (isTRUE(malformed)) {
    stop("legacy eig batch result payload is malformed", call. = FALSE)
  }

  diagnostics <- result$diagnostics
  flags <- c(
    "request_identity_authenticated", "prepared_identity_authenticated",
    "target_identity_authenticated", "residuals_device_resident",
    "components_device_resident", "compact_result_only_d2h",
    "deterministic_logical_order", "component_capacity_respected",
    "bounded_allocation", "leak_free_teardown", "caller_device_restored"
  )
  timing_fields <- c(
    "residual_solve_host_ms", "metadata_h2d_cuda_ms",
    "distance_build_cuda_ms", "full_eig_cuda_ms",
    "component_finalize_cuda_ms", "component_build_cuda_ms",
    "pair_evaluation_cuda_ms", "compact_d2h_cuda_ms",
    "dcov_host_boundary_ms", "teardown_host_ms", "total_host_ms"
  )
  structural_failure <-
    !is.list(diagnostics) ||
    length(setdiff(c(flags, timing_fields), names(diagnostics))) > 0L ||
    !all(unlist(diagnostics[flags], use.names = FALSE)) ||
    !identical(diagnostics$component_semantic_version,
      "full-cuda-ci-legacy-raw-distance-full-eig-numcol35-v1") ||
    diagnostics$pair_count != pair_count ||
    diagnostics$pair_evaluation_count != pair_count ||
    diagnostics$referenced_component_count != referenced_count ||
    diagnostics$component_build_count != referenced_count ||
    diagnostics$component_capacity != request$component_capacity ||
    diagnostics$num_col != request$num_col ||
    diagnostics$solver_failure_count != 0L ||
    diagnostics$residual_d2h_bytes != 0 ||
    diagnostics$component_d2h_bytes != 0 ||
    diagnostics$compact_result_d2h_count != 1L ||
    diagnostics$compact_result_d2h_bytes != 72 * pair_count ||
    diagnostics$compact_status_d2h_count != 1L ||
    diagnostics$compact_status_d2h_bytes != 4 * referenced_count ||
    diagnostics$metadata_h2d_count != 7L ||
    diagnostics$metadata_h2d_bytes !=
      4 * referenced_count + 40 * pair_count ||
    diagnostics$cpu_dcov_component_count != 0L ||
    diagnostics$cpu_dcov_eigen_count != 0L ||
    diagnostics$cpu_dcov_pair_statistic_count != 0L ||
    diagnostics$cpu_gamma_p_value_count != 0L ||
    diagnostics$cuda_full_eig_count != referenced_count ||
    diagnostics$cuda_pair_count != pair_count ||
    diagnostics$cuda_gamma_count != pair_count ||
    diagnostics$consumer_event_registration_count != 1L ||
    diagnostics$explicit_host_wait_count != 2L ||
    diagnostics$device_allocation_count != 19L ||
    diagnostics$device_allocation_count != diagnostics$device_free_count ||
    any(!is.finite(unlist(diagnostics[timing_fields], use.names = FALSE))) ||
    any(unlist(diagnostics[timing_fields], use.names = FALSE) < 0) ||
    abs(diagnostics$component_build_cuda_ms - sum(unlist(
      diagnostics[c(
        "distance_build_cuda_ms", "full_eig_cuda_ms",
        "component_finalize_cuda_ms"
      )], use.names = FALSE
    ))) > 1e-9
  if (isTRUE(structural_failure)) {
    stop("legacy eig batch structural diagnostics failed", call. = FALSE)
  }
  invisible(TRUE)
}

.fastkpc_full_cuda_phase35_bakeoff_require <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
  invisible(TRUE)
}

.fastkpc_full_cuda_phase35_bakeoff_required_functions <- function() {
  c(
    "fastkpc_full_cuda_census_file_hash",
    "fastkpc_full_cuda_census_key_set_hash",
    "fastkpc_full_cuda_census_named_metadata_hash",
    "fastkpc_full_cuda_fixed_sp_catalog_contract",
    "fastkpc_full_cuda_fixed_sp_native_dto",
    "fastkpc_full_cuda_fixed_sp_read_json",
    "fastkpc_full_cuda_prepared_s_input_contract",
    "fastkpc_full_cuda_prepared_s_validate_target_state_frame_schema",
    "fixed_sp_cuda_runtime_create",
    "fixed_sp_cuda_runtime_reserve",
    "fixed_sp_cuda_runtime_free",
    "fixed_sp_cuda_prepared_create",
    "fixed_sp_cuda_prepared_free",
    "fixed_sp_cuda_prepared_info"
  )
}

.fastkpc_full_cuda_phase35_bakeoff_check_dependencies <- function() {
  required <- .fastkpc_full_cuda_phase35_bakeoff_required_functions()
  missing <- required[!vapply(
    required, exists, logical(1L), mode = "function", inherits = TRUE
  )]
  if (length(missing) > 0L) {
    stop("Phase 3.5 bake-off dependency is missing: ", missing[[1L]],
         call. = FALSE)
  }
  invisible(TRUE)
}

fastkpc_full_cuda_phase35_bakeoff_input_paths <- function() {
  prepared_dir <- file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1"
  )
  census_dir <- file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "workload_census_351x48_v1"
  )
  oracle_dir <- file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "fixed_sp_cuda_oracle_sp_v1"
  )
  list(
    prepared_dir = prepared_dir,
    census_dir = census_dir,
    oracle_dir = oracle_dir,
    data = file.path(
      "fastkpc", "artifacts", "kpc_tprs_real_zhu",
      "cancer_RD-causalDiscoveryInput.rds"
    ),
    prepared_manifest = file.path(prepared_dir, "manifest.json"),
    prepared_setups = file.path(prepared_dir, "prepared_s_setup_index.rds"),
    target_states = file.path(prepared_dir, "target_state_index.rds"),
    qualification_setups = file.path(
      prepared_dir, "qualification_setup_groups.rds"
    ),
    qualification_targets = file.path(
      prepared_dir, "qualification_target_keys.rds"
    ),
    qualification_logical = file.path(
      prepared_dir, "qualification_logical_tests.rds"
    ),
    qualification_coverage = file.path(
      prepared_dir, "qualification_coverage.csv"
    ),
    canonical_logical = file.path(census_dir, "logical_ci_tests.rds"),
    oracle_manifest = file.path(oracle_dir, "manifest.json"),
    target_parity = file.path(oracle_dir, "target_parity.rds")
  )
}

.fastkpc_full_cuda_phase35_bakeoff_verify_file <- function(
    path, expected_sha256, label) {
  actual <- fastkpc_full_cuda_census_file_hash(path)
  .fastkpc_full_cuda_phase35_bakeoff_require(
    is.character(expected_sha256) && length(expected_sha256) == 1L &&
      !is.na(expected_sha256) &&
      grepl("^[0-9a-f]{64}$", expected_sha256) &&
      identical(actual, expected_sha256),
    paste0("Phase 3.5 bake-off ", label, " SHA-256 mismatch")
  )
  actual
}

.fastkpc_full_cuda_phase35_plain_frame <- function(value) {
  result <- as.data.frame(value, stringsAsFactors = FALSE)
  attributes(result) <- list(
    names = names(result),
    class = "data.frame",
    row.names = .set_row_names(nrow(result))
  )
  result
}

fastkpc_full_cuda_phase35_load_bakeoff_corpus <- function(
    include_canonical_logical = TRUE) {
  .fastkpc_full_cuda_phase35_bakeoff_check_dependencies()
  paths <- fastkpc_full_cuda_phase35_bakeoff_input_paths()
  if (!all(file.exists(unlist(paths, use.names = FALSE)))) {
    stop("Phase 3.5 bake-off input file set is incomplete", call. = FALSE)
  }

  catalog_contract <- fastkpc_full_cuda_fixed_sp_catalog_contract()
  input_contract <- fastkpc_full_cuda_prepared_s_input_contract()
  prepared_manifest <- fastkpc_full_cuda_fixed_sp_read_json(
    paths$prepared_manifest
  )
  oracle_manifest <- fastkpc_full_cuda_fixed_sp_read_json(
    paths$oracle_manifest
  )
  hashes <- c(
    prepared_s_setup_index_rds =
      .fastkpc_full_cuda_phase35_bakeoff_verify_file(
        paths$prepared_setups,
        catalog_contract$prepared_s_setup_index_rds_sha256,
        "Prepared-S setup index"
      ),
    target_state_index_rds =
      .fastkpc_full_cuda_phase35_bakeoff_verify_file(
        paths$target_states,
        catalog_contract$target_state_index_rds_sha256,
        "target-state index"
      ),
    qualification_setup_groups_rds =
      .fastkpc_full_cuda_phase35_bakeoff_verify_file(
        paths$qualification_setups,
        unname(catalog_contract$phase2_semantic_file_sha256[[
          "qualification_setup_groups_rds"
        ]]),
        "qualification setup corpus"
      ),
    qualification_target_keys_rds =
      .fastkpc_full_cuda_phase35_bakeoff_verify_file(
        paths$qualification_targets,
        unname(catalog_contract$phase2_semantic_file_sha256[[
          "qualification_target_keys_rds"
        ]]),
        "qualification target corpus"
      ),
    qualification_logical_tests_rds =
      .fastkpc_full_cuda_phase35_bakeoff_verify_file(
        paths$qualification_logical,
        unname(catalog_contract$phase2_semantic_file_sha256[[
          "qualification_logical_tests_rds"
        ]]),
        "qualification logical corpus"
      ),
    qualification_coverage_csv =
      .fastkpc_full_cuda_phase35_bakeoff_verify_file(
        paths$qualification_coverage,
        unname(catalog_contract$phase2_semantic_file_sha256[[
          "qualification_coverage_csv"
        ]]),
        "qualification coverage"
      ),
    target_parity_rds =
      .fastkpc_full_cuda_phase35_bakeoff_verify_file(
        paths$target_parity,
        as.character(oracle_manifest$payload_file_sha256[[
          "target_parity.rds"
        ]]),
        "Phase 3 route authority"
      ),
    dataset_rds = .fastkpc_full_cuda_phase35_bakeoff_verify_file(
      paths$data, input_contract$dataset_file_sha256, "dataset"
    )
  )
  if (isTRUE(include_canonical_logical)) {
    hashes <- c(
      hashes,
      canonical_logical_tests_rds =
        .fastkpc_full_cuda_phase35_bakeoff_verify_file(
          paths$canonical_logical,
          unname(input_contract$file_hashes[["logical_ci_tests.rds"]]),
          "canonical logical trace"
        )
    )
  }
  .fastkpc_full_cuda_phase35_bakeoff_require(
    identical(
      as.character(prepared_manifest$prepared_s_setup_index_rds_sha256),
      hashes[["prepared_s_setup_index_rds"]]
    ) && identical(
      as.character(prepared_manifest$target_state_index_rds_sha256),
      hashes[["target_state_index_rds"]]
    ),
    "Phase 3.5 bake-off Prepared-S manifest/index linkage failed"
  )

  prepared_setups <- readRDS(paths$prepared_setups)
  target_states <- readRDS(paths$target_states)
  qualification_setups <- readRDS(paths$qualification_setups)
  qualification_targets <- readRDS(paths$qualification_targets)
  qualification_logical <- readRDS(paths$qualification_logical)
  qualification_coverage <- utils::read.csv(
    paths$qualification_coverage, stringsAsFactors = FALSE,
    check.names = FALSE
  )
  target_parity <- readRDS(paths$target_parity)
  data <- readRDS(paths$data)
  canonical_logical <- if (isTRUE(include_canonical_logical)) {
    readRDS(paths$canonical_logical)
  } else {
    NULL
  }
  fastkpc_full_cuda_prepared_s_validate_target_state_frame_schema(
    target_states
  )

  setup_keys <- names(prepared_setups)
  target_keys <- as.character(target_states$residual_key_sha256)
  qualification_endpoint_keys <- sort(unique(c(
    qualification_logical$residual_key_x,
    qualification_logical$residual_key_y
  )), method = "radix")
  gates <-
    is.list(prepared_setups) && length(prepared_setups) == 8634L &&
    length(setup_keys) == 8634L && !anyNA(setup_keys) &&
    !anyDuplicated(setup_keys) &&
    all(grepl("^[0-9a-f]{64}$", setup_keys)) &&
    identical(setup_keys, sort(setup_keys, method = "radix")) &&
    nrow(target_states) == 110617L && !anyNA(target_keys) &&
    !anyDuplicated(target_keys) &&
    identical(
      fastkpc_full_cuda_census_key_set_hash(sort(
        setup_keys, method = "radix"
      )),
      catalog_contract$full_canonical_prepared_s_key_corpus_hash
    ) && identical(
      fastkpc_full_cuda_census_key_set_hash(sort(
        target_keys, method = "radix"
      )),
      prepared_manifest$full_canonical_target_key_corpus_hash
    ) &&
    is.data.frame(qualification_setups) &&
    nrow(qualification_setups) == 2061L &&
    is.data.frame(qualification_targets) &&
    nrow(qualification_targets) == 6143L &&
    is.data.frame(qualification_logical) &&
    nrow(qualification_logical) == 3808L &&
    identical(sum(qualification_logical$near_alpha), 1478L) &&
    identical(qualification_endpoint_keys,
              sort(qualification_targets$residual_key_sha256,
                   method = "radix")) &&
    is.data.frame(target_parity) && nrow(target_parity) == 110617L &&
    !anyNA(target_parity$residual_key_sha256) &&
    !anyDuplicated(target_parity$residual_key_sha256) &&
    all(target_parity$full_cuda_data_plane) &&
    all(target_parity$cpu_fallback_count == 0L) &&
    all(target_parity$unknown_fallback_count == 0L) &&
    all(!target_parity$approximate_backend) &&
    is.matrix(data) && identical(dim(data), c(351L, 48L)) &&
    all(is.finite(data))
  if (isTRUE(include_canonical_logical)) {
    gates <- gates && is.data.frame(canonical_logical) &&
      nrow(canonical_logical) == 240489L &&
      identical(
        attr(canonical_logical, "dataset_sha256", exact = TRUE),
        input_contract$dataset_matrix_sha256
      )
  }
  .fastkpc_full_cuda_phase35_bakeoff_require(
    gates, "Phase 3.5 bake-off authenticated index corpus is malformed"
  )

  setup_group_ids <- vapply(
    prepared_setups, `[[`, character(1L), "same_S_group_id"
  )
  qualification_setup_index <- match(
    qualification_setups$same_S_group_id, setup_group_ids
  )
  .fastkpc_full_cuda_phase35_bakeoff_require(
    !anyNA(setup_group_ids) && !anyDuplicated(setup_group_ids) &&
      !anyNA(qualification_setup_index) &&
      !anyDuplicated(qualification_setup_index),
    "Phase 3.5 bake-off Prepared-S group index is malformed"
  )
  # Bind every executable object to its own key without rehydrating shards.
  selected_setup_keys <- setup_keys[qualification_setup_index]
  .fastkpc_full_cuda_phase35_bakeoff_require(
    !anyNA(selected_setup_keys) && length(selected_setup_keys) == 2061L &&
      all(vapply(seq_along(selected_setup_keys), function(index) {
        setup <- prepared_setups[[selected_setup_keys[[index]]]]
        identical(setup$prepared_s_key_sha256,
                  selected_setup_keys[[index]]) &&
          identical(setup$same_S_group_id,
                    qualification_setups$same_S_group_id[[index]]) &&
          identical(setup$dataset_sha256,
                    input_contract$dataset_matrix_sha256)
      }, logical(1L))),
    "Phase 3.5 bake-off selected Prepared-S lineage failed"
  )

  list(
    schema_version = "full-cuda-ci-phase35-bakeoff-index-corpus-v1",
    paths = paths,
    file_sha256 = hashes,
    prepared_manifest = prepared_manifest,
    oracle_manifest = oracle_manifest,
    catalog_contract = catalog_contract,
    input_contract = input_contract,
    prepared_setups = prepared_setups,
    setup_group_ids = setup_group_ids,
    target_states = target_states,
    target_parity = target_parity,
    qualification_setups = qualification_setups,
    qualification_targets = qualification_targets,
    qualification_logical = qualification_logical,
    qualification_coverage = qualification_coverage,
    canonical_logical = canonical_logical,
    data = data
  )
}

.fastkpc_full_cuda_phase35_group_logical_rows <- function(
    logical_rows, target_states) {
  left <- match(logical_rows$residual_key_x,
                target_states$residual_key_sha256)
  right <- match(logical_rows$residual_key_y,
                 target_states$residual_key_sha256)
  if (anyNA(left) || anyNA(right)) {
    stop("Phase 3.5 scale has an unresolved residual endpoint", call. = FALSE)
  }
  left_setup <- target_states$prepared_s_key_sha256[left]
  right_setup <- target_states$prepared_s_key_sha256[right]
  if (!identical(left_setup, right_setup)) {
    stop("Phase 3.5 scale pair endpoints cross Prepared-S groups",
         call. = FALSE)
  }
  groups <- split(
    seq_len(nrow(logical_rows)),
    factor(left_setup, levels = sort(unique(left_setup), method = "radix"))
  )
  groups[lengths(groups) > 0L]
}

.fastkpc_full_cuda_phase35_scale_group_table <- function(
    logical_rows, groups) {
  rows <- lapply(names(groups), function(prepared_key) {
    index <- groups[[prepared_key]]
    endpoints <- unique(c(
      logical_rows$residual_key_x[index],
      logical_rows$residual_key_y[index]
    ))
    data.frame(
      prepared_s_key_sha256 = prepared_key,
      level = as.integer(unique(logical_rows$level[index])),
      pair_count = as.integer(length(index)),
      component_count = as.integer(length(endpoints)),
      reuse_ratio = as.numeric(2 * length(index) / length(endpoints)),
      minimum_logical_sequence_id = as.integer(min(
        logical_rows$logical_sequence_id[index]
      )),
      stringsAsFactors = FALSE
    )
  })
  result <- do.call(rbind, rows)
  rownames(result) <- NULL
  result
}

fastkpc_full_cuda_phase35_qualification_scale <- function(corpus) {
  logical_rows <- .fastkpc_full_cuda_phase35_plain_frame(
    corpus$qualification_logical
  )
  logical_rows <- logical_rows[order(
    logical_rows$logical_sequence_id, method = "radix"
  ), , drop = FALSE]
  groups <- .fastkpc_full_cuda_phase35_group_logical_rows(
    logical_rows, corpus$target_states
  )
  group_table <- .fastkpc_full_cuda_phase35_scale_group_table(
    logical_rows, groups
  )
  .fastkpc_full_cuda_phase35_bakeoff_require(
    nrow(logical_rows) == 3808L && length(groups) == 2061L &&
      sum(group_table$component_count) == 6143L &&
      sum(logical_rows$near_alpha) == 1478L,
    "Phase 3.5 Scale A counts do not match the tracked corpus"
  )
  list(
    schema_version = "full-cuda-ci-phase35-scale-v1",
    scale_id = "A_qualification_complete",
    selection_policy = "complete-authenticated-development-qualification",
    logical_rows = logical_rows,
    groups = groups,
    group_table = group_table,
    selection_sha256 = fastkpc_full_cuda_census_named_metadata_hash(list(
      scale_id = "A_qualification_complete",
      logical_sequence_ids = logical_rows$logical_sequence_id,
      prepared_s_keys = names(groups)
    ))
  )
}

.fastkpc_full_cuda_phase35_evenly_spaced_indices <- function(count, take) {
  if (count < take || take < 1L) {
    stop("Phase 3.5 stratified selection is undersized", call. = FALSE)
  }
  selected <- unique(as.integer(round(seq(1, count, length.out = take))))
  if (length(selected) != take) {
    stop("Phase 3.5 stratified selection produced duplicate ranks",
         call. = FALSE)
  }
  selected
}

fastkpc_full_cuda_phase35_campaign_slice_scale <- function(
    corpus, groups_per_quartile = 48L) {
  if (!is.integer(groups_per_quartile) ||
      length(groups_per_quartile) != 1L || is.na(groups_per_quartile) ||
      groups_per_quartile < 1L) {
    stop("groups_per_quartile must be one positive integer", call. = FALSE)
  }
  logical_rows <- corpus$canonical_logical
  if (is.null(logical_rows)) {
    stop("canonical logical trace is required for Scale B", call. = FALSE)
  }
  logical_rows <- .fastkpc_full_cuda_phase35_plain_frame(
    logical_rows[logical_rows$level == 2L, , drop = FALSE]
  )
  all_groups <- .fastkpc_full_cuda_phase35_group_logical_rows(
    logical_rows, corpus$target_states
  )
  group_table <- .fastkpc_full_cuda_phase35_scale_group_table(
    logical_rows, all_groups
  )
  group_table <- group_table[order(
    group_table$reuse_ratio, group_table$prepared_s_key_sha256,
    method = "radix"
  ), , drop = FALSE]
  rownames(group_table) <- NULL
  group_count <- nrow(group_table)
  breaks <- c(
    0L, floor(group_count * 0.25), floor(group_count * 0.50),
    floor(group_count * 0.75), group_count
  )
  quartile <- integer(group_count)
  for (index in seq_len(4L)) {
    quartile[seq.int(breaks[[index]] + 1L, breaks[[index + 1L]])] <- index
  }
  group_table$reuse_quartile <- quartile
  selected_rows <- unlist(lapply(seq_len(4L), function(index) {
    candidates <- which(group_table$reuse_quartile == index)
    candidates[
      .fastkpc_full_cuda_phase35_evenly_spaced_indices(
        length(candidates), groups_per_quartile
      )
    ]
  }), use.names = FALSE)
  selected_groups <- group_table[selected_rows, , drop = FALSE]
  selected_keys <- selected_groups$prepared_s_key_sha256
  selected_index <- unlist(all_groups[selected_keys], use.names = FALSE)
  selected_logical <- logical_rows[selected_index, , drop = FALSE]
  selected_logical <- selected_logical[order(
    selected_logical$logical_sequence_id, method = "radix"
  ), , drop = FALSE]
  selected_logical$near_alpha <-
    abs(selected_logical$signed_log_ratio_from_alpha) <= log(2)
  groups <- .fastkpc_full_cuda_phase35_group_logical_rows(
    selected_logical, corpus$target_states
  )
  selected_groups <- .fastkpc_full_cuda_phase35_scale_group_table(
    selected_logical, groups
  )
  selected_groups$reuse_quartile <- group_table$reuse_quartile[
    match(selected_groups$prepared_s_key_sha256,
          group_table$prepared_s_key_sha256)
  ]
  expected_groups <- as.integer(4L * groups_per_quartile)
  .fastkpc_full_cuda_phase35_bakeoff_require(
    group_count == 1126L && nrow(selected_groups) == expected_groups &&
      nrow(selected_logical) > 20000L &&
      sum(selected_groups$component_count) > 7000L &&
      identical(sort(unique(selected_groups$reuse_quartile)), 1:4),
    "Phase 3.5 Scale B density/reuse selection gate failed"
  )
  list(
    schema_version = "full-cuda-ci-phase35-scale-v1",
    scale_id = "B_level2_reuse_quartiles_192",
    selection_policy = paste0(
      "level-2;rank-by-reuse-then-key;four-rank-quartiles;",
      groups_per_quartile, "-evenly-spaced-groups-per-quartile"
    ),
    logical_rows = selected_logical,
    groups = groups,
    group_table = selected_groups,
    population_group_table = group_table,
    selection_sha256 = fastkpc_full_cuda_census_named_metadata_hash(list(
      scale_id = "B_level2_reuse_quartiles_192",
      groups_per_quartile = groups_per_quartile,
      logical_sequence_ids = selected_logical$logical_sequence_id,
      prepared_s_keys = names(groups),
      reuse_quartile = selected_groups$reuse_quartile
    ))
  )
}

fastkpc_full_cuda_phase35_full_conditional_scale <- function(corpus) {
  if (is.null(corpus$canonical_logical)) {
    stop("canonical logical trace is required for the full conditional scale",
         call. = FALSE)
  }
  logical_rows <- .fastkpc_full_cuda_phase35_plain_frame(
    corpus$canonical_logical[
      corpus$canonical_logical$level > 0L, , drop = FALSE
    ]
  )
  logical_rows <- logical_rows[order(
    logical_rows$logical_sequence_id, method = "radix"
  ), , drop = FALSE]
  logical_rows$near_alpha <-
    abs(logical_rows$signed_log_ratio_from_alpha) <= log(2)
  groups <- .fastkpc_full_cuda_phase35_group_logical_rows(
    logical_rows, corpus$target_states
  )
  group_table <- .fastkpc_full_cuda_phase35_scale_group_table(
    logical_rows, groups
  )
  .fastkpc_full_cuda_phase35_bakeoff_require(
    nrow(logical_rows) == 238276L && length(groups) == 8634L &&
      sum(group_table$component_count) == 110617L,
    "Phase 3.5 full conditional scale counts are malformed"
  )
  list(
    schema_version = "full-cuda-ci-phase35-scale-v1",
    scale_id = "FULL_CONDITIONAL_LEVELS_1_TO_7",
    selection_policy =
      "complete-authenticated-canonical-trace-excluding-raw-level-zero",
    logical_rows = logical_rows,
    groups = groups,
    group_table = group_table,
    selection_sha256 = fastkpc_full_cuda_census_named_metadata_hash(list(
      scale_id = "FULL_CONDITIONAL_LEVELS_1_TO_7",
      logical_sequence_ids = logical_rows$logical_sequence_id,
      prepared_s_keys = names(groups)
    ))
  )
}

.fastkpc_full_cuda_phase35_bakeoff_empty_pair_frame <- function() {
  data.frame(
    scale_id = character(),
    prepared_s_key_sha256 = character(),
    logical_sequence_id = integer(),
    level = integer(),
    x = integer(),
    y = integer(),
    S_key = character(),
    residual_key_x = character(),
    residual_key_y = character(),
    alpha = double(),
    legacy_reference_p_value = double(),
    candidate_p_value = double(),
    p_value_difference_from_legacy = double(),
    candidate_independent = logical(),
    legacy_reference_independent = logical(),
    decision_flip = logical(),
    near_alpha = logical(),
    deletes_edge = logical(),
    candidate_statistic = double(),
    candidate_mean = double(),
    candidate_variance = double(),
    gamma_shape = double(),
    gamma_scale = double(),
    gamma_iterations = integer(),
    solver_route = character(),
    dcov_status = character(),
    stringsAsFactors = FALSE
  )
}

.fastkpc_full_cuda_phase35_bakeoff_group_diagnostics <- function(
    scale_id, prepared_key, result, setup_elapsed_ms,
    prepared_free_elapsed_ms, output_slot_released) {
  diagnostics <- result$diagnostics
  data.frame(
    scale_id = scale_id,
    prepared_s_key_sha256 = prepared_key,
    target_count = as.integer(diagnostics$target_count),
    pair_count = as.integer(diagnostics$pair_count),
    component_count = as.integer(diagnostics$referenced_component_count),
    component_cache_hit_count =
      as.integer(diagnostics$component_cache_hit_count),
    component_cache_miss_count =
      as.integer(diagnostics$component_cache_miss_count),
    residual_solve_host_ms = diagnostics$residual_solve_host_ms,
    metadata_h2d_cuda_ms = diagnostics$metadata_h2d_cuda_ms,
    component_build_cuda_ms = diagnostics$component_build_cuda_ms,
    pair_evaluation_cuda_ms = diagnostics$pair_evaluation_cuda_ms,
    compact_d2h_cuda_ms = diagnostics$compact_d2h_cuda_ms,
    dcov_host_boundary_ms = diagnostics$dcov_host_boundary_ms,
    teardown_host_ms = diagnostics$teardown_host_ms,
    total_host_ms = diagnostics$total_host_ms,
    prepared_create_host_ms = setup_elapsed_ms,
    prepared_free_host_ms = prepared_free_elapsed_ms,
    metadata_h2d_bytes = diagnostics$metadata_h2d_bytes,
    compact_result_d2h_bytes = diagnostics$compact_result_d2h_bytes,
    peak_component_bytes = diagnostics$peak_component_bytes,
    peak_live_device_bytes = diagnostics$peak_live_device_bytes,
    device_allocation_count =
      as.integer(diagnostics$device_allocation_count),
    device_free_count = as.integer(diagnostics$device_free_count),
    explicit_host_wait_count =
      as.integer(diagnostics$explicit_host_wait_count),
    residual_d2h_bytes = diagnostics$residual_d2h_bytes,
    component_d2h_bytes = diagnostics$component_d2h_bytes,
    cpu_dcov_component_count =
      as.integer(diagnostics$cpu_dcov_component_count),
    cpu_dcov_pair_statistic_count =
      as.integer(diagnostics$cpu_dcov_pair_statistic_count),
    cpu_gamma_p_value_count =
      as.integer(diagnostics$cpu_gamma_p_value_count),
    bounded_allocation = diagnostics$bounded_allocation,
    leak_free_teardown = diagnostics$leak_free_teardown,
    caller_device_restored = diagnostics$caller_device_restored,
    output_slot_released = output_slot_released,
    stringsAsFactors = FALSE
  )
}

.fastkpc_full_cuda_phase35_legacy_eig_group_diagnostics <- function(
    scale_id, prepared_key, result, setup_elapsed_ms,
    prepared_free_elapsed_ms, output_slot_released) {
  diagnostics <- result$diagnostics
  data.frame(
    scale_id = scale_id,
    prepared_s_key_sha256 = prepared_key,
    target_count = as.integer(diagnostics$target_count),
    pair_count = as.integer(diagnostics$pair_count),
    component_count = as.integer(diagnostics$referenced_component_count),
    residual_solve_host_ms = diagnostics$residual_solve_host_ms,
    metadata_h2d_cuda_ms = diagnostics$metadata_h2d_cuda_ms,
    distance_build_cuda_ms = diagnostics$distance_build_cuda_ms,
    full_eig_cuda_ms = diagnostics$full_eig_cuda_ms,
    component_finalize_cuda_ms = diagnostics$component_finalize_cuda_ms,
    component_build_cuda_ms = diagnostics$component_build_cuda_ms,
    pair_evaluation_cuda_ms = diagnostics$pair_evaluation_cuda_ms,
    compact_d2h_cuda_ms = diagnostics$compact_d2h_cuda_ms,
    dcov_host_boundary_ms = diagnostics$dcov_host_boundary_ms,
    teardown_host_ms = diagnostics$teardown_host_ms,
    total_host_ms = diagnostics$total_host_ms,
    prepared_create_host_ms = setup_elapsed_ms,
    prepared_free_host_ms = prepared_free_elapsed_ms,
    metadata_h2d_bytes = diagnostics$metadata_h2d_bytes,
    compact_result_d2h_bytes = diagnostics$compact_result_d2h_bytes,
    compact_status_d2h_bytes = diagnostics$compact_status_d2h_bytes,
    persistent_component_bytes = diagnostics$persistent_component_bytes,
    eig_workspace_bytes = diagnostics$eig_workspace_bytes,
    pair_workspace_bytes = diagnostics$pair_workspace_bytes,
    peak_live_device_bytes = diagnostics$peak_live_device_bytes,
    device_allocation_count =
      as.integer(diagnostics$device_allocation_count),
    device_free_count = as.integer(diagnostics$device_free_count),
    explicit_host_wait_count =
      as.integer(diagnostics$explicit_host_wait_count),
    residual_d2h_bytes = diagnostics$residual_d2h_bytes,
    component_d2h_bytes = diagnostics$component_d2h_bytes,
    cpu_dcov_component_count =
      as.integer(diagnostics$cpu_dcov_component_count),
    cpu_dcov_eigen_count = as.integer(diagnostics$cpu_dcov_eigen_count),
    cpu_dcov_pair_statistic_count =
      as.integer(diagnostics$cpu_dcov_pair_statistic_count),
    cpu_gamma_p_value_count =
      as.integer(diagnostics$cpu_gamma_p_value_count),
    cuda_full_eig_count = as.integer(diagnostics$cuda_full_eig_count),
    cuda_pair_count = as.integer(diagnostics$cuda_pair_count),
    cuda_gamma_count = as.integer(diagnostics$cuda_gamma_count),
    solver_failure_count = as.integer(diagnostics$solver_failure_count),
    bounded_allocation = diagnostics$bounded_allocation,
    leak_free_teardown = diagnostics$leak_free_teardown,
    caller_device_restored = diagnostics$caller_device_restored,
    output_slot_released = output_slot_released,
    stringsAsFactors = FALSE
  )
}

.fastkpc_full_cuda_phase35_resolve_group_execution <- function(
    corpus, prepared_key, logical_rows) {
  target_keys <- sort(unique(c(
    logical_rows$residual_key_x, logical_rows$residual_key_y
  )), method = "radix")
  state_index <- match(
    target_keys, corpus$target_states$residual_key_sha256
  )
  parity_index <- match(
    target_keys, corpus$target_parity$residual_key_sha256
  )
  if (anyNA(state_index) || anyNA(parity_index)) {
    stop("Phase 3.5 group endpoint authority is incomplete", call. = FALSE)
  }
  states <- corpus$target_states[state_index, , drop = FALSE]
  parity <- corpus$target_parity[parity_index, , drop = FALSE]
  setup <- corpus$prepared_setups[[prepared_key]]
  route_clean <-
    !is.null(setup) &&
    identical(states$residual_key_sha256, target_keys) &&
    all(states$prepared_s_key_sha256 == prepared_key) &&
    identical(parity$residual_key_sha256, target_keys) &&
    all(parity$prepared_s_key_sha256 == prepared_key) &&
    identical(parity$planned_route, parity$authenticated_planned_route) &&
    identical(parity$planned_route, parity$executed_route) &&
    all(parity$solver_status %in% c(
      "OK_CHOLESKY_BATCHED", "OK_CHOLESKY_SINGLE",
      "OK_AUGMENTED_QR", "OK_AUGMENTED_SVD"
    )) &&
    all(parity$cpu_fallback_count == 0L) &&
    all(parity$unknown_fallback_count == 0L) &&
    all(!parity$approximate_backend)
  if (!isTRUE(route_clean)) {
    stop("Phase 3.5 group route authentication failed", call. = FALSE)
  }
  list(
    target_keys = target_keys,
    states = states,
    parity = parity,
    setup = setup,
    dto = fastkpc_full_cuda_fixed_sp_native_dto(setup),
    Y = corpus$data[, states$target, drop = FALSE],
    SP = do.call(cbind, lapply(states$selected_sp, as.numeric))
  )
}

.fastkpc_full_cuda_phase35_monotonic_ms <- function(start) {
  as.numeric((proc.time()[["elapsed"]] - start) * 1000)
}

fastkpc_full_cuda_phase35_run_exact_scale <- function(
    corpus, scale, runtime, progress = interactive(),
    require_decision_parity = TRUE) {
  if (!is.list(scale) || !identical(
        scale$schema_version, "full-cuda-ci-phase35-scale-v1"
      )) {
    stop("Phase 3.5 exact scale is malformed", call. = FALSE)
  }
  pair_chunks <- vector("list", length(scale$groups))
  group_chunks <- vector("list", length(scale$groups))
  resources_before <-
    fastkpc_full_cuda_phase35_vertical_resource_snapshot()
  group_keys <- names(scale$groups)
  for (group_ordinal in seq_along(group_keys)) {
    prepared_key <- group_keys[[group_ordinal]]
    logical_rows <- scale$logical_rows[
      scale$groups[[prepared_key]], , drop = FALSE
    ]
    logical_rows <- logical_rows[order(
      logical_rows$logical_sequence_id, method = "radix"
    ), , drop = FALSE]
    target_keys <- sort(unique(c(
      logical_rows$residual_key_x, logical_rows$residual_key_y
    )), method = "radix")
    state_index <- match(
      target_keys, corpus$target_states$residual_key_sha256
    )
    parity_index <- match(
      target_keys, corpus$target_parity$residual_key_sha256
    )
    if (anyNA(state_index) || anyNA(parity_index)) {
      stop("Phase 3.5 exact group endpoint authority is incomplete",
           call. = FALSE)
    }
    states <- corpus$target_states[state_index, , drop = FALSE]
    parity <- corpus$target_parity[parity_index, , drop = FALSE]
    setup <- corpus$prepared_setups[[prepared_key]]
    route_clean <-
      !is.null(setup) &&
      identical(states$residual_key_sha256, target_keys) &&
      all(states$prepared_s_key_sha256 == prepared_key) &&
      identical(parity$residual_key_sha256, target_keys) &&
      all(parity$prepared_s_key_sha256 == prepared_key) &&
      identical(parity$planned_route, parity$authenticated_planned_route) &&
      identical(parity$planned_route, parity$executed_route) &&
      all(parity$solver_status %in% c(
        "OK_CHOLESKY_BATCHED", "OK_CHOLESKY_SINGLE",
        "OK_AUGMENTED_QR", "OK_AUGMENTED_SVD"
      )) &&
      all(parity$cpu_fallback_count == 0L) &&
      all(parity$unknown_fallback_count == 0L) &&
      all(!parity$approximate_backend)
    if (!isTRUE(route_clean)) {
      stop("Phase 3.5 exact group route authentication failed",
           call. = FALSE)
    }

    dto <- fastkpc_full_cuda_fixed_sp_native_dto(setup)
    Y <- corpus$data[, states$target, drop = FALSE]
    SP <- do.call(cbind, lapply(states$selected_sp, as.numeric))
    request <- fastkpc_full_cuda_phase35_exact_batch_request_from_logical(
      expected_prepared_s_key_sha256 = prepared_key,
      target_keys = target_keys,
      logical_rows = logical_rows
    )
    handle <- NULL
    create_start <- proc.time()[["elapsed"]]
    handle <- fixed_sp_cuda_prepared_create(runtime, dto)
    create_ms <- .fastkpc_full_cuda_phase35_monotonic_ms(create_start)
    result <- tryCatch(
      fastkpc_full_cuda_phase35_exact_batch_ci(
        handle, Y, SP, parity$planned_route, target_keys, request
      ),
      error = function(error) {
        try(fixed_sp_cuda_prepared_free(handle), silent = TRUE)
        stop(error)
      }
    )
    fastkpc_full_cuda_phase35_validate_exact_batch_result(
      result, request, target_keys
    )
    output_slot_released <- identical(
      fixed_sp_cuda_prepared_info(handle)$output_slot_state, "free"
    )
    free_start <- proc.time()[["elapsed"]]
    fixed_sp_cuda_prepared_free(handle)
    handle <- NULL
    free_ms <- .fastkpc_full_cuda_phase35_monotonic_ms(free_start)

    pair_chunks[[group_ordinal]] <- data.frame(
      scale_id = scale$scale_id,
      prepared_s_key_sha256 = prepared_key,
      logical_sequence_id = as.integer(logical_rows$logical_sequence_id),
      level = as.integer(logical_rows$level),
      x = as.integer(logical_rows$x),
      y = as.integer(logical_rows$y),
      S_key = as.character(logical_rows$S_key),
      residual_key_x = as.character(logical_rows$residual_key_x),
      residual_key_y = as.character(logical_rows$residual_key_y),
      alpha = as.numeric(logical_rows$alpha),
      legacy_reference_p_value =
        as.numeric(logical_rows$reference_p_value),
      candidate_p_value = as.numeric(result$records$p_value),
      p_value_difference_from_legacy =
        as.numeric(result$records$p_value) -
          as.numeric(logical_rows$reference_p_value),
      candidate_independent = result$records$p_value >= logical_rows$alpha,
      legacy_reference_independent =
        as.logical(logical_rows$reference_independent),
      decision_flip =
        (result$records$p_value >= logical_rows$alpha) !=
          logical_rows$reference_independent,
      near_alpha = as.logical(logical_rows$near_alpha),
      deletes_edge = as.logical(logical_rows$deletes_edge),
      candidate_statistic = as.numeric(result$numerical$statistic),
      candidate_mean = as.numeric(result$numerical$mean),
      candidate_variance = as.numeric(result$numerical$variance),
      gamma_shape = as.numeric(result$numerical$gamma_shape),
      gamma_scale = as.numeric(result$numerical$gamma_scale),
      gamma_iterations = as.integer(result$numerical$gamma_iterations),
      solver_route = as.character(result$records$solver_route),
      dcov_status = as.character(result$records$dcov_status),
      stringsAsFactors = FALSE
    )
    group_chunks[[group_ordinal]] <-
      .fastkpc_full_cuda_phase35_bakeoff_group_diagnostics(
        scale$scale_id, prepared_key, result, create_ms, free_ms,
        output_slot_released
      )
    if (isTRUE(progress) &&
        (group_ordinal == 1L || group_ordinal %% 100L == 0L ||
         group_ordinal == length(group_keys))) {
      cat(
        scale$scale_id, ": group ", group_ordinal, "/",
        length(group_keys), "; pairs=",
        sum(vapply(pair_chunks[seq_len(group_ordinal)], nrow, integer(1L))),
        "\n", sep = ""
      )
    }
  }
  pairs <- if (length(pair_chunks) == 0L) {
    .fastkpc_full_cuda_phase35_bakeoff_empty_pair_frame()
  } else {
    do.call(rbind, pair_chunks)
  }
  groups <- do.call(rbind, group_chunks)
  rownames(pairs) <- NULL
  rownames(groups) <- NULL
  resources_after <-
    fastkpc_full_cuda_phase35_vertical_resource_snapshot()
  live_fields <- c(
    "live_device_allocations", "live_device_bytes", "live_streams",
    "live_events"
  )
  gate_values <- c(
    pair_count_exact = nrow(pairs) == nrow(scale$logical_rows),
    group_count_exact = nrow(groups) == length(scale$groups),
    logical_ids_exact = identical(
      sort(pairs$logical_sequence_id),
      sort(as.integer(scale$logical_rows$logical_sequence_id))
    ),
    decision_parity = !any(pairs$decision_flip),
    status_ok = all(pairs$dcov_status == "OK_EXACT_CUDA_GAMMA"),
    numerical_finite = all(is.finite(as.matrix(pairs[c(
      "candidate_p_value", "candidate_statistic", "candidate_mean",
      "candidate_variance", "gamma_shape", "gamma_scale"
    )]))),
    residual_d2h_zero = all(groups$residual_d2h_bytes == 0),
    component_d2h_zero = all(groups$component_d2h_bytes == 0),
    cpu_component_zero = all(groups$cpu_dcov_component_count == 0L),
    cpu_pair_zero = all(groups$cpu_dcov_pair_statistic_count == 0L),
    cpu_gamma_zero = all(groups$cpu_gamma_p_value_count == 0L),
    bounded_allocation = all(groups$bounded_allocation),
    leak_free_teardown = all(groups$leak_free_teardown),
    caller_device_restored = all(groups$caller_device_restored),
    output_slot_released = all(groups$output_slot_released),
    live_resources_restored = identical(
      resources_after[live_fields], resources_before[live_fields]
    )
  )
  required_gates <- gate_values
  if (!isTRUE(require_decision_parity)) {
    required_gates <- required_gates[names(required_gates) !=
                                      "decision_parity"]
  }
  failed_gates <- names(required_gates)[!required_gates]
  .fastkpc_full_cuda_phase35_bakeoff_require(
    length(failed_gates) == 0L,
    paste0(
      "Phase 3.5 exact ", scale$scale_id, " gate failed: ",
      paste(failed_gates, collapse = ","),
      "; decision_flips=", sum(pairs$decision_flip),
      "; near_alpha_flips=", sum(pairs$decision_flip & pairs$near_alpha)
    )
  )
  list(
    schema_version = "full-cuda-ci-phase35-exact-scale-result-v1",
    scale_id = scale$scale_id,
    selection_sha256 = scale$selection_sha256,
    pairs = pairs,
    groups = groups,
    gates = gate_values,
    resources_before = resources_before,
    resources_after = resources_after
  )
}

fastkpc_full_cuda_phase35_run_guarded_hybrid_scale <- function(
    corpus, scale, runtime, exact_result = NULL,
    guard_lower = 0.05, guard_upper = 0.15,
    progress = interactive()) {
  if (!is.list(scale) || !identical(
        scale$schema_version, "full-cuda-ci-phase35-scale-v1"
      ) ||
      !is.double(guard_lower) || length(guard_lower) != 1L ||
      !is.double(guard_upper) || length(guard_upper) != 1L ||
      !is.finite(guard_lower) || !is.finite(guard_upper) ||
      guard_lower < 0 || guard_lower >= 0.1 ||
      guard_upper <= 0.1 || guard_upper > 1) {
    stop("Phase 3.5 guarded hybrid configuration is malformed",
         call. = FALSE)
  }
  if (is.null(exact_result)) {
    exact_result <- fastkpc_full_cuda_phase35_run_exact_scale(
      corpus, scale, runtime, progress = progress,
      require_decision_parity = FALSE
    )
  }
  exact_gate_names <- setdiff(names(exact_result$gates), "decision_parity")
  exact_valid <-
    is.list(exact_result) &&
    identical(exact_result$schema_version,
              "full-cuda-ci-phase35-exact-scale-result-v1") &&
    identical(exact_result$scale_id, scale$scale_id) &&
    identical(exact_result$selection_sha256, scale$selection_sha256) &&
    nrow(exact_result$pairs) == nrow(scale$logical_rows) &&
    !anyDuplicated(exact_result$pairs$logical_sequence_id) &&
    all(exact_result$gates[exact_gate_names])
  .fastkpc_full_cuda_phase35_bakeoff_require(
    exact_valid,
    "Phase 3.5 guarded hybrid exact-screen authority is malformed"
  )

  pairs <- exact_result$pairs
  pairs$screen_p_value <- pairs$candidate_p_value
  pairs$screen_p_value_difference_from_legacy <-
    pairs$p_value_difference_from_legacy
  pairs$screen_independent <- pairs$candidate_independent
  pairs$screen_decision_flip <- pairs$decision_flip
  pairs$screen_statistic <- pairs$candidate_statistic
  pairs$screen_mean <- pairs$candidate_mean
  pairs$screen_variance <- pairs$candidate_variance
  pairs$refined <-
    pairs$screen_p_value >= guard_lower &
      pairs$screen_p_value <= guard_upper
  pairs$final_backend <- ifelse(
    pairs$refined,
    "guarded-legacy-full-eig-cuda",
    "exact-cuda-certified-outside-guard"
  )

  guarded_ids <- pairs$logical_sequence_id[pairs$refined]
  logical_index <- match(
    guarded_ids, as.integer(scale$logical_rows$logical_sequence_id)
  )
  .fastkpc_full_cuda_phase35_bakeoff_require(
    !anyNA(logical_index),
    "Phase 3.5 guarded hybrid logical selection is unresolved"
  )
  guarded_logical <- scale$logical_rows[logical_index, , drop = FALSE]
  guarded_logical <- guarded_logical[order(
    guarded_logical$logical_sequence_id, method = "radix"
  ), , drop = FALSE]
  guarded_groups <- .fastkpc_full_cuda_phase35_group_logical_rows(
    guarded_logical, corpus$target_states
  )
  refinement_pair_chunks <- vector("list", length(guarded_groups))
  refinement_group_chunks <- vector("list", length(guarded_groups))
  resources_before <-
    fastkpc_full_cuda_phase35_vertical_resource_snapshot()
  group_keys <- names(guarded_groups)

  for (group_ordinal in seq_along(group_keys)) {
    prepared_key <- group_keys[[group_ordinal]]
    logical_rows <- guarded_logical[
      guarded_groups[[prepared_key]], , drop = FALSE
    ]
    logical_rows <- logical_rows[order(
      logical_rows$logical_sequence_id, method = "radix"
    ), , drop = FALSE]
    context <- .fastkpc_full_cuda_phase35_resolve_group_execution(
      corpus, prepared_key, logical_rows
    )
    request <-
      fastkpc_full_cuda_phase35_legacy_eig_batch_request_from_logical(
        expected_prepared_s_key_sha256 = prepared_key,
        target_keys = context$target_keys,
        logical_rows = logical_rows
      )
    handle <- NULL
    create_start <- proc.time()[["elapsed"]]
    handle <- fixed_sp_cuda_prepared_create(runtime, context$dto)
    create_ms <- .fastkpc_full_cuda_phase35_monotonic_ms(create_start)
    result <- tryCatch(
      fastkpc_full_cuda_phase35_legacy_eig_batch_ci(
        handle, context$Y, context$SP, context$parity$planned_route,
        context$target_keys, request
      ),
      error = function(error) {
        try(fixed_sp_cuda_prepared_free(handle), silent = TRUE)
        stop(error)
      }
    )
    fastkpc_full_cuda_phase35_validate_legacy_eig_batch_result(
      result, request, context$target_keys
    )
    output_slot_released <- identical(
      fixed_sp_cuda_prepared_info(handle)$output_slot_state, "free"
    )
    free_start <- proc.time()[["elapsed"]]
    fixed_sp_cuda_prepared_free(handle)
    handle <- NULL
    free_ms <- .fastkpc_full_cuda_phase35_monotonic_ms(free_start)

    result_index <- match(
      as.integer(logical_rows$logical_sequence_id),
      as.integer(result$records$logical_sequence_id)
    )
    if (anyNA(result_index)) {
      stop("Phase 3.5 guarded hybrid refinement result is incomplete",
           call. = FALSE)
    }
    result_records <- result$records[result_index, , drop = FALSE]
    result_numerical <- result$numerical[result_index, , drop = FALSE]
    refinement_pair_chunks[[group_ordinal]] <- data.frame(
      scale_id = scale$scale_id,
      prepared_s_key_sha256 = prepared_key,
      logical_sequence_id = as.integer(logical_rows$logical_sequence_id),
      legacy_reference_p_value =
        as.numeric(logical_rows$reference_p_value),
      refined_p_value = as.numeric(result_records$p_value),
      refined_p_value_difference_from_legacy =
        as.numeric(result_records$p_value) -
          as.numeric(logical_rows$reference_p_value),
      refined_independent =
        result_records$p_value >= logical_rows$alpha,
      legacy_reference_independent =
        as.logical(logical_rows$reference_independent),
      refined_decision_flip =
        (result_records$p_value >= logical_rows$alpha) !=
          logical_rows$reference_independent,
      near_alpha = as.logical(logical_rows$near_alpha),
      refined_statistic = as.numeric(result_numerical$statistic),
      refined_mean = as.numeric(result_numerical$mean),
      refined_variance = as.numeric(result_numerical$variance),
      refined_gamma_shape = as.numeric(result_numerical$gamma_shape),
      refined_gamma_scale = as.numeric(result_numerical$gamma_scale),
      refined_gamma_iterations =
        as.integer(result_numerical$gamma_iterations),
      solver_route = as.character(result_records$solver_route),
      dcov_status = as.character(result_records$dcov_status),
      stringsAsFactors = FALSE
    )
    refinement_group_chunks[[group_ordinal]] <-
      .fastkpc_full_cuda_phase35_legacy_eig_group_diagnostics(
        scale$scale_id, prepared_key, result, create_ms, free_ms,
        output_slot_released
      )
    if (isTRUE(progress) &&
        (group_ordinal == 1L || group_ordinal %% 50L == 0L ||
         group_ordinal == length(group_keys))) {
      cat(
        scale$scale_id, " refinement: group ", group_ordinal, "/",
        length(group_keys), "; pairs=",
        sum(vapply(
          refinement_pair_chunks[seq_len(group_ordinal)], nrow, integer(1L)
        )), "\n", sep = ""
      )
    }
  }

  refinement_pairs <- if (length(refinement_pair_chunks) == 0L) {
    data.frame(
      scale_id = character(),
      prepared_s_key_sha256 = character(),
      logical_sequence_id = integer(),
      legacy_reference_p_value = double(),
      refined_p_value = double(),
      refined_p_value_difference_from_legacy = double(),
      refined_independent = logical(),
      legacy_reference_independent = logical(),
      refined_decision_flip = logical(),
      near_alpha = logical(),
      refined_statistic = double(),
      refined_mean = double(),
      refined_variance = double(),
      refined_gamma_shape = double(),
      refined_gamma_scale = double(),
      refined_gamma_iterations = integer(),
      solver_route = character(),
      dcov_status = character(),
      stringsAsFactors = FALSE
    )
  } else {
    do.call(rbind, refinement_pair_chunks)
  }
  refinement_groups <- if (length(refinement_group_chunks) == 0L) {
    NULL
  } else {
    do.call(rbind, refinement_group_chunks)
  }
  rownames(refinement_pairs) <- NULL
  if (!is.null(refinement_groups)) rownames(refinement_groups) <- NULL
  resources_after <-
    fastkpc_full_cuda_phase35_vertical_resource_snapshot()

  final_index <- match(
    refinement_pairs$logical_sequence_id, pairs$logical_sequence_id
  )
  .fastkpc_full_cuda_phase35_bakeoff_require(
    !anyNA(final_index) && all(pairs$refined[final_index]),
    "Phase 3.5 guarded hybrid refinement replay is malformed"
  )
  if (length(final_index) > 0L) {
    pairs$candidate_p_value[final_index] <- refinement_pairs$refined_p_value
    pairs$p_value_difference_from_legacy[final_index] <-
      refinement_pairs$refined_p_value_difference_from_legacy
    pairs$candidate_independent[final_index] <-
      refinement_pairs$refined_independent
    pairs$decision_flip[final_index] <-
      refinement_pairs$refined_decision_flip
    pairs$candidate_statistic[final_index] <-
      refinement_pairs$refined_statistic
    pairs$candidate_mean[final_index] <- refinement_pairs$refined_mean
    pairs$candidate_variance[final_index] <-
      refinement_pairs$refined_variance
    pairs$gamma_shape[final_index] <- refinement_pairs$refined_gamma_shape
    pairs$gamma_scale[final_index] <- refinement_pairs$refined_gamma_scale
    pairs$gamma_iterations[final_index] <-
      refinement_pairs$refined_gamma_iterations
    pairs$solver_route[final_index] <- refinement_pairs$solver_route
    pairs$dcov_status[final_index] <- refinement_pairs$dcov_status
  }

  live_fields <- c(
    "live_device_allocations", "live_device_bytes", "live_streams",
    "live_events"
  )
  structural_refinement <- if (is.null(refinement_groups)) {
    FALSE
  } else {
    all(refinement_groups$residual_d2h_bytes == 0) &&
      all(refinement_groups$component_d2h_bytes == 0) &&
      all(refinement_groups$cpu_dcov_component_count == 0L) &&
      all(refinement_groups$cpu_dcov_eigen_count == 0L) &&
      all(refinement_groups$cpu_dcov_pair_statistic_count == 0L) &&
      all(refinement_groups$cpu_gamma_p_value_count == 0L) &&
      all(refinement_groups$cuda_full_eig_count ==
            refinement_groups$component_count) &&
      all(refinement_groups$cuda_pair_count == refinement_groups$pair_count) &&
      all(refinement_groups$cuda_gamma_count ==
            refinement_groups$pair_count) &&
      all(refinement_groups$solver_failure_count == 0L) &&
      all(refinement_groups$bounded_allocation) &&
      all(refinement_groups$leak_free_teardown) &&
      all(refinement_groups$caller_device_restored) &&
      all(refinement_groups$output_slot_released)
  }
  gate_values <- c(
    exact_screen_structural = all(exact_result$gates[exact_gate_names]),
    guard_selection_exact = identical(
      pairs$refined,
      pairs$screen_p_value >= guard_lower &
        pairs$screen_p_value <= guard_upper
    ),
    refinement_nonempty = nrow(refinement_pairs) > 0L,
    refinement_pair_count_exact =
      nrow(refinement_pairs) == sum(pairs$refined),
    refinement_group_count_exact =
      !is.null(refinement_groups) &&
        nrow(refinement_groups) == length(guarded_groups),
    screen_flips_guarded =
      all(!pairs$screen_decision_flip | pairs$refined),
    final_decision_parity = !any(pairs$decision_flip),
    complete_near_alpha_decision_parity =
      !any(pairs$decision_flip & pairs$near_alpha),
    refined_p_value_parity =
      max(abs(refinement_pairs$refined_p_value_difference_from_legacy)) <=
        1e-10,
    refined_status_ok =
      all(refinement_pairs$dcov_status == "OK_EXACT_CUDA_GAMMA"),
    refined_numerical_finite = all(is.finite(as.matrix(
      refinement_pairs[c(
        "refined_p_value", "refined_statistic", "refined_mean",
        "refined_variance", "refined_gamma_shape", "refined_gamma_scale"
      )]
    ))),
    refinement_cuda_authority = structural_refinement,
    live_resources_restored = identical(
      resources_after[live_fields], resources_before[live_fields]
    )
  )
  failed_gates <- names(gate_values)[!gate_values]
  .fastkpc_full_cuda_phase35_bakeoff_require(
    length(failed_gates) == 0L,
    paste0(
      "Phase 3.5 guarded hybrid ", scale$scale_id, " gate failed: ",
      paste(failed_gates, collapse = ","),
      "; screen_flips=", sum(pairs$screen_decision_flip),
      "; final_flips=", sum(pairs$decision_flip),
      "; refined_pairs=", sum(pairs$refined)
    )
  )

  exact_groups <- exact_result$groups
  timing <- data.frame(
    scale_id = scale$scale_id,
    screen_component_cuda_ms = sum(exact_groups$component_build_cuda_ms),
    screen_pair_gamma_cuda_ms =
      sum(exact_groups$pair_evaluation_cuda_ms),
    screen_dcov_host_boundary_ms = sum(exact_groups$dcov_host_boundary_ms),
    refinement_component_cuda_ms =
      sum(refinement_groups$component_build_cuda_ms),
    refinement_pair_gamma_cuda_ms =
      sum(refinement_groups$pair_evaluation_cuda_ms),
    refinement_dcov_host_boundary_ms =
      sum(refinement_groups$dcov_host_boundary_ms),
    total_component_cuda_ms =
      sum(exact_groups$component_build_cuda_ms) +
        sum(refinement_groups$component_build_cuda_ms),
    total_pair_gamma_cuda_ms =
      sum(exact_groups$pair_evaluation_cuda_ms) +
        sum(refinement_groups$pair_evaluation_cuda_ms),
    total_dcov_host_boundary_ms =
      sum(exact_groups$dcov_host_boundary_ms) +
        sum(refinement_groups$dcov_host_boundary_ms),
    stringsAsFactors = FALSE
  )
  list(
    schema_version =
      "full-cuda-ci-phase35-guarded-hybrid-scale-result-v1",
    candidate_id = "candidate-c-exact-screen-plus-guarded-a-full-eig-v1",
    scale_id = scale$scale_id,
    selection_sha256 = scale$selection_sha256,
    guard = list(
      lower_inclusive = guard_lower,
      upper_inclusive = guard_upper,
      alpha = 0.1,
      policy = "refine-exact-screen-p-in-closed-interval"
    ),
    pairs = pairs,
    refinement_pairs = refinement_pairs,
    exact_groups = exact_groups,
    refinement_groups = refinement_groups,
    timing = timing,
    gates = gate_values,
    resources_before = resources_before,
    resources_after = resources_after,
    exact_screen_result = exact_result
  )
}

fastkpc_full_cuda_phase35_run_two_scale_exact <- function(
    corpus = NULL, progress = interactive()) {
  if (is.null(corpus)) {
    corpus <- fastkpc_full_cuda_phase35_load_bakeoff_corpus(
      include_canonical_logical = TRUE
    )
  }
  scale_a <- fastkpc_full_cuda_phase35_qualification_scale(corpus)
  scale_b <- fastkpc_full_cuda_phase35_campaign_slice_scale(corpus)
  load_fastkpc_cuda_native()
  runtime <- fixed_sp_cuda_runtime_create(0L)
  on.exit({
    if (!is.null(runtime)) {
      try(fixed_sp_cuda_runtime_free(runtime), silent = TRUE)
    }
  }, add = TRUE)
  fixed_sp_cuda_runtime_reserve(
    runtime, 351L, 64L, 47L, 7L, 415L
  )
  runtime_info <- fixed_sp_cuda_runtime_info(runtime)
  result_a <- fastkpc_full_cuda_phase35_run_exact_scale(
    corpus, scale_a, runtime, progress = progress,
    require_decision_parity = FALSE
  )
  result_b <- fastkpc_full_cuda_phase35_run_exact_scale(
    corpus, scale_b, runtime, progress = progress,
    require_decision_parity = FALSE
  )
  fixed_sp_cuda_runtime_free(runtime)
  runtime <- NULL
  list(
    schema_version = "full-cuda-ci-phase35-two-scale-exact-run-v1",
    corpus_file_sha256 = corpus$file_sha256,
    scale_a = scale_a,
    scale_b = scale_b,
    result_a = result_a,
    result_b = result_b,
    runtime_info = runtime_info
  )
}

fastkpc_full_cuda_phase35_run_two_scale_guarded_hybrid <- function(
    corpus = NULL, progress = interactive(),
    guard_lower = 0.05, guard_upper = 0.15) {
  if (is.null(corpus)) {
    corpus <- fastkpc_full_cuda_phase35_load_bakeoff_corpus(
      include_canonical_logical = TRUE
    )
  }
  scale_a <- fastkpc_full_cuda_phase35_qualification_scale(corpus)
  scale_b <- fastkpc_full_cuda_phase35_campaign_slice_scale(corpus)
  load_fastkpc_cuda_native()
  runtime <- fixed_sp_cuda_runtime_create(0L)
  on.exit({
    if (!is.null(runtime)) {
      try(fixed_sp_cuda_runtime_free(runtime), silent = TRUE)
    }
  }, add = TRUE)
  fixed_sp_cuda_runtime_reserve(
    runtime, 351L, 64L, 47L, 7L, 415L
  )
  runtime_info <- fixed_sp_cuda_runtime_info(runtime)
  exact_a <- fastkpc_full_cuda_phase35_run_exact_scale(
    corpus, scale_a, runtime, progress = progress,
    require_decision_parity = FALSE
  )
  hybrid_a <- fastkpc_full_cuda_phase35_run_guarded_hybrid_scale(
    corpus, scale_a, runtime, exact_result = exact_a,
    guard_lower = guard_lower, guard_upper = guard_upper,
    progress = progress
  )
  exact_b <- fastkpc_full_cuda_phase35_run_exact_scale(
    corpus, scale_b, runtime, progress = progress,
    require_decision_parity = FALSE
  )
  hybrid_b <- fastkpc_full_cuda_phase35_run_guarded_hybrid_scale(
    corpus, scale_b, runtime, exact_result = exact_b,
    guard_lower = guard_lower, guard_upper = guard_upper,
    progress = progress
  )
  fixed_sp_cuda_runtime_free(runtime)
  runtime <- NULL
  list(
    schema_version =
      "full-cuda-ci-phase35-two-scale-guarded-hybrid-run-v1",
    candidate_id =
      "candidate-c-exact-screen-plus-guarded-a-full-eig-v1",
    corpus_file_sha256 = corpus$file_sha256,
    scale_a = scale_a,
    scale_b = scale_b,
    exact_a = exact_a,
    hybrid_a = hybrid_a,
    exact_b = exact_b,
    hybrid_b = hybrid_b,
    runtime_info = runtime_info
  )
}
