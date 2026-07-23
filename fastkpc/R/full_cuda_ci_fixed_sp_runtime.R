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

fastkpc_full_cuda_fixed_sp_runtime_abi <- function() {
  abi <- list(
    schema_version = "full-cuda-ci-fixed-sp-runtime-v1",
    native_dto_schema_version =
      "full-cuda-ci-prepared-s-native-dto-v1",
    phase3c_runtime_schema_version =
      "full-cuda-ci-fixed-sp-phase3c-runtime-v1",
    resource_instrumentation_version = 1L
  )
  c(
    abi,
    list(
      sha256 = fastkpc_full_cuda_census_named_metadata_hash(abi)
    )
  )
}

fastkpc_full_cuda_fixed_sp_catalog_contract <- function() {
  inherited_setup_fields <- c(
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
  inherited_target_fields <- c(
    "residual_key_sha256", "same_S_group_id", "target", "selected_sp",
    "selected_sp_names", "selected_sp_hash", "fit_time_ms",
    "coefficient_all_finite", "fitted_all_finite", "residual_all_finite",
    "coefficient_hash", "fitted_hash", "residual_hash",
    "target_fit_fingerprint", "setup_fingerprint"
  )
  list(
    schema_version = "full-cuda-ci-fixed-sp-catalog-v1",
    phase2_artifact_schema_version = "full-cuda-ci-prepared-s-artifact-v1",
    phase2_input_schema_version = "full-cuda-ci-phase2-input-v1",
    run_scope = "full",
    parity_scope = "qualification",
    selected_group_count = 8634L,
    target_state_count = 110617L,
    shard_count = 64L,
    phase2_source_commit = "42ef3efa08327056ffe5c9aad7a8953ff6864c7e",
    prepared_s_setup_schema_version = "full-cuda-ci-prepared-s-setup-v1",
    target_state_schema_version = "full-cuda-ci-target-state-v1",
    census_dir = file.path(
      "fastkpc", "artifacts", "full_cuda_ci", "workload_census_351x48_v1"
    ),
    data_path = file.path(
      "fastkpc", "artifacts", "kpc_tprs_real_zhu",
      "cancer_RD-causalDiscoveryInput.rds"
    ),
    unsupported_scope = "canonical",
    new_candidate_graph_gate = "NOT_APPLICABLE",
    full_canonical_prepared_s_key_corpus_hash =
      "acf6d8a93884467ca35da3f9a24966151a041ac84efd76edffde83a348a2cd79",
    iteration_subset_hash =
      "d69956655e2ee0186aceb4bf2d545b831051c389aed139769d9e5902f433fb96",
    qualification_subset_hash =
      "0adea2bac7b31615421f180b6caa5aeef5567bafa0e45a319b358136bf429c61",
    semantic_tolerance_config_hash =
      "9f4d9222e20b7e20cb04f2cb48ff66e763a85812cc79986f0cf95f9fbb0ea3c8",
    qualification_selection_config_hash =
      "5faf92e8555f04339345fdf298bbe0cde388ff7263fbbf27e56b1656a3f4bf59",
    prepared_s_setup_index_rds_sha256 =
      "c87a07ccefe57d55f4175f8745ba8d261721f276c1912d1072adf448fea1b687",
    target_state_index_rds_sha256 =
      "0673c42efc129f3de9bff64247ff1cb0626a7ff7552838c9f385a2c555287215",
    phase2_semantic_file_sha256 = c(
      input_hashes_csv =
        "68b32df8fe5a072b497cf20cdd24697561aac786a6f3393ca60003fb30d5e989",
      prepared_s_setup_index_rds =
        "c87a07ccefe57d55f4175f8745ba8d261721f276c1912d1072adf448fea1b687",
      prepared_s_setup_index_csv =
        "68c8904a39a15423ded120f652ca35e843ade88cd1b0c32f54d7cf3d372855eb",
      target_state_index_rds =
        "0673c42efc129f3de9bff64247ff1cb0626a7ff7552838c9f385a2c555287215",
      target_state_index_csv =
        "4ea5134ec0e306eea22ee184ad074a14c4ceb620eafa91c373093dbf86437822",
      iteration_setup_groups_rds =
        "2ca2199ad9926517bdafc809be992dccfc95cbc9a1da88a875c66a85a1dba8e8",
      iteration_setup_groups_csv =
        "22d90ca1dfcbe8496a3838ff81f93db9ff54c2ce0b56124ed79b9dd5e813f86b",
      iteration_target_keys_rds =
        "599ab3619415a6e3ff50dc6a8ae9aff473e2b31e67005323ebe88e9015bfabe0",
      iteration_target_keys_csv =
        "dd49b98d5a88467c9a155fa66827e4ec9dff851738c05c9923973883364c9507",
      iteration_logical_tests_rds =
        "d0c98f690f2b2c031ca3e224f041f8de522eacc606fc604ab0d46d0346256159",
      iteration_logical_tests_csv =
        "2a96dcb0cf04599827daa27132aa1745ee29c7fb5d07d51a5da9c279f14c393f",
      iteration_coverage_csv =
        "f7bd1e0419bfd160522cbfdcc9f98ce19897d07025bb4119259aebdb9747817e",
      qualification_setup_groups_rds =
        "df2518f9c790997ec31184388fd1b201c3b852c5818241bd37ccc2b7e7334375",
      qualification_setup_groups_csv =
        "210351936a53c36714275221675bb5233f4a9d0763d4da6c293857e2b325255d",
      qualification_target_keys_rds =
        "2e9f05d4f5d9d5538cd404f5bb53702c541d340995172b26d544f7949e96bf3f",
      qualification_target_keys_csv =
        "af0fe348fae7622d89c98aa7b05b7048181ee82451dd7bef5c77b461114fe218",
      qualification_logical_tests_rds =
        "da7bfb2e13606f00523c8fcbaca87848bf89c784a319c8e98be7b0856653aff0",
      qualification_logical_tests_csv =
        "2d68e81228830bf4f0f2813cd20f95fe8911f5a85be3328e6f13f0f8d2028897",
      qualification_coverage_csv =
        "eee0a9530c2f4e09ed9d9bd72c92a6354893e44e1cf9c062f9790546be046543",
      setup_semantic_parity_csv =
        "26368e05e939bd7a60ec506e1a07b192739d4b5c56ad38bbd8121b3583570c17",
      target_retarget_parity_csv =
        "22bcea5b8c4fffa540136082f7eec799fac51670b1142933215ba6de7f16faae",
      dcov_parity_csv =
        "faf0f9233fb457f7974163a09cb3d41543bd34b07fc123ed476865eb0b2f8ab9",
      unsupported_envelope_csv =
        "275470c02f595c600837d2d3bdb487f38475985d976be43a89cfd426fca477b0",
      fallbacks_csv =
        "8a9025b288f20e3af6b04eb59b7a3590ab1a5811a3cd17ededb29e38f38ec150"
    ),
    inherited_setup_fields = inherited_setup_fields,
    inherited_target_fields = inherited_target_fields,
    setup_scope_fields = c(
      inherited_setup_fields, "setup_target_count",
      "setup_logical_request_count", "selection_reasons"
    ),
    target_scope_fields = c(
      "case_type", "residual_key_sha256", "logical_sequence_id",
      "same_S_group_id", "high_condition", "rank_deficient",
      "near_constant_target", "near_constant_conditioner", "multi_penalty",
      "near_alpha", "mgcv_warning", "mgcv_nonconverged",
      "nonfinite_metadata", "condition_bucket", "near_alpha_bucket",
      "target", "S_size", "penalty_count", "condition",
      "request_multiplicity", "same_S_group_size", "setup_target_count",
      "setup_logical_request_count", "representative_residual_key_sha256",
      "convergence_signature", "optimizer_iterations", "selected_sp",
      "selected_sp_names", "selected_sp_hash", "fit_time_ms",
      "coefficient_all_finite", "fitted_all_finite", "residual_all_finite",
      "coefficient_hash", "fitted_hash", "residual_hash",
      "target_fit_fingerprint", "setup_fingerprint", "selection_reasons"
    ),
    phase2_manifest_fields = c(
      "schema_version", "phase2_input_schema_version", "run_scope", "parity_scope",
      "phase2_complete", "phase1_input_bundle_hash", "dataset_file_sha256",
      "dataset_matrix_sha256", "canonical_logical_census_hash",
      "canonical_target_key_corpus_hash", "full_canonical_target_key_corpus_hash",
      "full_canonical_prepared_s_key_corpus_hash", "selected_prepared_s_key_corpus_hash",
      "selected_target_key_corpus_hash", "iteration_subset_hash", "qualification_subset_hash",
      "source_commit", "R_version", "mgcv_version", "BLAS_identity", "LAPACK_identity",
      "BLAS_thread_count", "prepared_s_setup_schema_version", "target_state_schema_version",
      "semantic_tolerance_config_hash", "qualification_selection_config_hash", "census_dir",
      "data_path", "selected_group_count", "target_state_count",
      "unsupported_selected_setup_count", "unsupported_canonical_setup_count",
      "unsupported_canonical_evaluated", "unsupported_scope", "unknown_fallback_count",
      "approximate_backend_count", "dcov_fallback_count", "requested_workers",
      "actual_workers", "shard_count", "resume", "max_groups", "written_shard_count",
      "reused_shard_count", "executed_group_count", "reused_group_count",
      "semantic_file_sha256", "prepared_s_setup_index_rds_sha256",
      "target_state_index_rds_sha256", "oracle_inherited_graph_gate",
      "new_candidate_graph_gate"
    ),
    phase2_summary_fields = c(
      "pass", "phase2_complete", "run_scope", "parity_scope",
      "phase2_input_authenticated", "selected_group_count", "target_state_count",
      "prepared_s_key_corpus_exact", "target_key_corpus_exact", "setup_lineage_exact",
      "target_lineage_exact", "response_leakage_count",
      "prepared_setup_fingerprint_collision_count", "target_state_fingerprint_collision_count",
      "unsupported_selected_setup_count", "unsupported_canonical_setup_count",
      "unsupported_canonical_evaluated", "unsupported_scope", "unknown_fallback_count",
      "approximate_backend_count", "dcov_fallback_count", "iteration_setup_group_count",
      "iteration_target_key_count", "iteration_logical_test_count", "seed_target_key_count",
      "qualification_target_key_count", "qualification_logical_test_count",
      "qualification_same_S_group_count", "conditional_near_alpha_test_count",
      "setup_semantic_parity_count", "fixed_sp_coefficient_hash_exact_count",
      "fixed_sp_fitted_hash_exact_count", "fixed_sp_residual_hash_exact_count",
      "fixed_sp_coefficient_hash_exact", "fixed_sp_fitted_hash_exact",
      "fixed_sp_residual_hash_exact", "legacy_dcov_max_abs_p_value_diff",
      "legacy_dcov_p_value_exact_count", "legacy_dcov_decision_flip_count",
      "legacy_dcov_spectra_no_fallback", "oracle_inherited_graph_gate",
      "new_candidate_graph_gate", "requested_workers", "actual_workers", "shard_count",
      "resume", "max_groups", "written_shard_count", "reused_shard_count",
      "executed_group_count", "reused_group_count", "parity_evidence_reused",
      "artifact_payload_size_bytes", "shard_rds_size_bytes", "stage_timing_total_seconds",
      "elapsed_seconds", "max_rss_kb"
    ),
    phase2_file_sha256 = c(
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
    )
  )
}

fastkpc_full_cuda_fixed_sp_is_exact_named_bare_list <- function(
    value, fields) {
  typeof(value) == "list" && !is.object(value) &&
    identical(attributes(value), list(names = fields))
}

fastkpc_full_cuda_fixed_sp_is_bare_scalar <- function(value, type) {
  typeof(value) == type && length(value) == 1L && !is.object(value) &&
    is.null(attributes(value)) && !is.na(value)
}

fastkpc_full_cuda_fixed_sp_is_bare_nonnegative_finite_numeric <- function(
    value) {
  typeof(value) %in% c("integer", "double") && length(value) == 1L &&
    !is.object(value) && is.null(attributes(value)) && !is.na(value) &&
    is.finite(value) && value >= 0
}

fastkpc_full_cuda_fixed_sp_is_bare_sha256 <- function(value) {
  fastkpc_full_cuda_fixed_sp_is_bare_scalar(value, "character") &&
    grepl("^[0-9a-f]{64}$", value)
}

fastkpc_full_cuda_fixed_sp_is_bare_sha256_vector <- function(value) {
  typeof(value) == "character" && !is.object(value) &&
    is.null(attributes(value)) && !anyNA(value) &&
    all(grepl("^[0-9a-f]{64}$", value))
}

fastkpc_full_cuda_fixed_sp_validate_bare_vector <- function(
    value, name, allowed_types) {
  if (!(typeof(value) %in% allowed_types) || is.object(value) ||
      !is.null(attributes(value))) {
    stop(name, " must be a bare ", paste(allowed_types, collapse = "/"),
         " vector", call. = FALSE)
  }
  invisible(value)
}

fastkpc_full_cuda_fixed_sp_route <- function(
    condition, coefficient_rank, null_dim, authenticated) {
  validate_numeric <- function(value, name) {
    fastkpc_full_cuda_fixed_sp_validate_bare_vector(
      value, name, c("integer", "double")
    )
  }
  validate_integer_metadata <- function(value, name) {
    validate_numeric(value, name)
    non_na <- !is.na(value)
    if (any(non_na & (!is.finite(value) | value < 0 |
                     value != floor(value) |
                     value > .Machine$integer.max))) {
      stop(name, " must contain finite nonnegative integer-valued values",
           call. = FALSE)
    }
    invisible(value)
  }

  validate_numeric(condition, "condition")
  if (any(is.finite(condition) & condition < 0)) {
    stop("condition must have nonnegative finite values", call. = FALSE)
  }
  validate_integer_metadata(coefficient_rank, "coefficient_rank")
  validate_integer_metadata(null_dim, "null_dim")
  fastkpc_full_cuda_fixed_sp_validate_bare_vector(
    authenticated, "authenticated", "logical"
  )

  lengths <- c(length(condition), length(coefficient_rank),
               length(null_dim), length(authenticated))
  n <- max(lengths)
  if (n == 0L || any(!(lengths %in% c(1L, n)))) {
    stop("inputs must be non-empty and scalar or common-length vectors",
         call. = FALSE)
  }

  condition <- rep_len(condition, n)
  coefficient_rank <- rep_len(as.integer(coefficient_rank), n)
  null_dim <- rep_len(as.integer(null_dim), n)
  authenticated <- rep_len(authenticated, n)
  contract <- fastkpc_full_cuda_fixed_sp_contract()
  out <- rep(contract$route_levels[[3L]], n)
  trusted <- !is.na(authenticated) & authenticated &
    is.finite(condition) & !is.na(coefficient_rank) & !is.na(null_dim) &
    coefficient_rank == null_dim
  out[trusted & condition < contract$cholesky_condition_max] <-
    contract$route_levels[[1L]]
  out[trusted & condition >= contract$cholesky_condition_max &
        condition < contract$svd_condition_min] <- contract$route_levels[[2L]]
  if (n == 1L) out[[1L]] else out
}

fastkpc_full_cuda_fixed_sp_read_json <- function(path) {
  if (!is.character(path) || length(path) != 1L || is.na(path) ||
      !nzchar(path) || !file.exists(path) || dir.exists(path)) {
    stop("fixed-sp JSON file is missing", call. = FALSE)
  }
  fastkpc_full_cuda_require_namespace("jsonlite")
  jsonlite::read_json(path, simplifyVector = TRUE)
}

fastkpc_full_cuda_fixed_sp_require <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
  invisible(value)
}

fastkpc_full_cuda_fixed_sp_phase2_file_paths <- function(phase2_dir, contract) {
  file.path(phase2_dir, names(contract$phase2_file_sha256))
}

fastkpc_full_cuda_fixed_sp_capture_phase2_files <- function(
    phase2_dir, contract = fastkpc_full_cuda_fixed_sp_catalog_contract()) {
  paths <- fastkpc_full_cuda_fixed_sp_phase2_file_paths(phase2_dir, contract)
  names(paths) <- names(contract$phase2_file_sha256)
  read_one <- function(path) {
    connection <- file(path, open = "rb")
    on.exit(close(connection), add = TRUE)
    chunks <- list()
    repeat {
      chunk <- readBin(connection, what = "raw", n = 1024L * 1024L)
      if (!length(chunk)) break
      chunks[[length(chunks) + 1L]] <- chunk
    }
    do.call(c, chunks)
  }
  fastkpc_full_cuda_fixed_sp_require(
    all(file.exists(paths)) && !any(dir.exists(paths)),
    "Phase 2 immutable file set is incomplete"
  )
  raw_files <- lapply(paths, read_one)
  hashes <- vapply(raw_files, fastkpc_full_cuda_census_hash_raw, character(1L))
  fastkpc_full_cuda_fixed_sp_require(
    identical(hashes, contract$phase2_file_sha256),
    "Phase 2 immutable file hash mismatch"
  )
  list(raw = raw_files, hashes = hashes)
}

fastkpc_full_cuda_fixed_sp_parse_phase2_files <- function(captured) {
  raw_connection <- function(value) rawConnection(value, open = "rb")
  parse_json <- function(value) {
    jsonlite::fromJSON(rawToChar(value), simplifyVector = TRUE)
  }
  parse_rds <- function(value) {
    connection <- gzcon(raw_connection(value))
    on.exit(close(connection), add = TRUE)
    readRDS(connection)
  }
  parse_csv <- function(value) {
    connection <- textConnection(rawToChar(value), open = "r")
    on.exit(close(connection), add = TRUE)
    utils::read.csv(connection, stringsAsFactors = FALSE)
  }
  list(
    summary = parse_json(captured$raw[["summary.json"]]),
    manifest = parse_json(captured$raw[["manifest.json"]]),
    setup_index = parse_csv(captured$raw[["prepared_s_setup_index.csv"]]),
    scopes = list(
      iteration = list(
        setup_rows = parse_rds(captured$raw[["iteration_setup_groups.rds"]]),
        target_rows = parse_rds(captured$raw[["iteration_target_keys.rds"]])
      ),
      qualification = list(
        setup_rows = parse_rds(captured$raw[["qualification_setup_groups.rds"]]),
        target_rows = parse_rds(captured$raw[["qualification_target_keys.rds"]])
      )
    )
  )
}

fastkpc_full_cuda_fixed_sp_validate_phase2_files <- function(
    phase2_dir, contract = fastkpc_full_cuda_fixed_sp_catalog_contract()) {
  fastkpc_full_cuda_fixed_sp_capture_phase2_files(phase2_dir, contract)$hashes
}

fastkpc_full_cuda_fixed_sp_validate_phase2_identity <- function(
    summary, manifest, inputs,
    contract = fastkpc_full_cuda_fixed_sp_catalog_contract()) {
  manifest_fields <- contract$phase2_manifest_fields
  summary_fields <- contract$phase2_summary_fields
  structure_clean <-
    fastkpc_full_cuda_fixed_sp_is_exact_named_bare_list(
      manifest, manifest_fields
    ) && fastkpc_full_cuda_fixed_sp_is_exact_named_bare_list(
      summary, summary_fields
    ) && typeof(inputs) == "list" && !is.object(inputs) &&
    typeof(inputs$manifest) == "list" && !is.object(inputs$manifest)
  fastkpc_full_cuda_fixed_sp_require(
    structure_clean,
    "Phase 2 Prepared-S identity does not match authenticated inputs"
  )

  manifest_flag_values <- list(
    phase2_complete = TRUE,
    unsupported_canonical_evaluated = TRUE,
    resume = TRUE,
    oracle_inherited_graph_gate = TRUE
  )
  summary_flag_values <- list(
    pass = TRUE,
    phase2_complete = TRUE,
    phase2_input_authenticated = TRUE,
    prepared_s_key_corpus_exact = TRUE,
    target_key_corpus_exact = TRUE,
    setup_lineage_exact = TRUE,
    target_lineage_exact = TRUE,
    unsupported_canonical_evaluated = TRUE,
    fixed_sp_coefficient_hash_exact = TRUE,
    fixed_sp_fitted_hash_exact = TRUE,
    fixed_sp_residual_hash_exact = TRUE,
    legacy_dcov_spectra_no_fallback = TRUE,
    oracle_inherited_graph_gate = TRUE,
    resume = TRUE,
    parity_evidence_reused = FALSE
  )
  manifest_count_values <- list(
    BLAS_thread_count = 1L,
    selected_group_count = contract$selected_group_count,
    target_state_count = contract$target_state_count,
    unsupported_selected_setup_count = 0L,
    unsupported_canonical_setup_count = 0L,
    unknown_fallback_count = 0L,
    approximate_backend_count = 0L,
    dcov_fallback_count = 0L,
    requested_workers = 20L,
    actual_workers = 20L,
    shard_count = contract$shard_count,
    max_groups = 0L,
    written_shard_count = 64L,
    reused_shard_count = 0L,
    executed_group_count = contract$selected_group_count,
    reused_group_count = 0L
  )
  summary_count_values <- list(
    selected_group_count = contract$selected_group_count,
    target_state_count = contract$target_state_count,
    response_leakage_count = 0L,
    prepared_setup_fingerprint_collision_count = 0L,
    target_state_fingerprint_collision_count = 0L,
    unsupported_selected_setup_count = 0L,
    unsupported_canonical_setup_count = 0L,
    unknown_fallback_count = 0L,
    approximate_backend_count = 0L,
    dcov_fallback_count = 0L,
    iteration_setup_group_count = 44L,
    iteration_target_key_count = 270L,
    iteration_logical_test_count = 44L,
    seed_target_key_count = 2356L,
    qualification_target_key_count = 6143L,
    qualification_logical_test_count = 3808L,
    qualification_same_S_group_count = 2061L,
    conditional_near_alpha_test_count = 1478L,
    setup_semantic_parity_count = 2061L,
    fixed_sp_coefficient_hash_exact_count = 6143L,
    fixed_sp_fitted_hash_exact_count = 6143L,
    fixed_sp_residual_hash_exact_count = 6143L,
    legacy_dcov_max_abs_p_value_diff = 0L,
    legacy_dcov_p_value_exact_count = 3808L,
    legacy_dcov_decision_flip_count = 0L,
    requested_workers = 20L,
    actual_workers = 20L,
    shard_count = contract$shard_count,
    max_groups = 0L,
    written_shard_count = 64L,
    reused_shard_count = 0L,
    executed_group_count = contract$selected_group_count,
    reused_group_count = 0L,
    artifact_payload_size_bytes = 1945880953L,
    max_rss_kb = 4933736L
  )
  manifest_flags <- names(manifest_flag_values)
  summary_flags <- names(summary_flag_values)
  manifest_counts <- names(manifest_count_values)
  summary_counts <- names(summary_count_values)
  manifest_character_fields <- setdiff(
    manifest_fields,
    c(manifest_flags, manifest_counts, "semantic_file_sha256")
  )
  summary_character_fields <- c(
    "run_scope", "parity_scope", "unsupported_scope",
    "new_candidate_graph_gate"
  )
  exact_values <- function(values, expected) {
    all(vapply(names(expected), function(name) {
      identical(values[[name]], expected[[name]])
    }, logical(1L)))
  }
  bare_values <- function(values, fields, type) {
    all(vapply(fields, function(name) {
      fastkpc_full_cuda_fixed_sp_is_bare_scalar(values[[name]], type)
    }, logical(1L)))
  }

  semantic_names <- fastkpc_full_cuda_prepared_s_semantic_artifact_path_names()
  semantic_hashes <- manifest$semantic_file_sha256
  semantic_clean <-
    identical(semantic_names, names(contract$phase2_semantic_file_sha256)) &&
    fastkpc_full_cuda_fixed_sp_is_exact_named_bare_list(
      semantic_hashes, semantic_names
    ) && all(vapply(
      semantic_hashes,
      fastkpc_full_cuda_fixed_sp_is_bare_sha256,
      logical(1L)
    )) && all(vapply(semantic_names, function(name) {
      identical(
        semantic_hashes[[name]],
        contract$phase2_semantic_file_sha256[[name]]
      )
    }, logical(1L)))
  fastkpc_full_cuda_fixed_sp_require(
    semantic_clean, "Phase 2 Prepared-S semantic manifest is invalid"
  )

  manifest_sha_fields <- c(
    "phase1_input_bundle_hash", "dataset_file_sha256",
    "dataset_matrix_sha256", "canonical_logical_census_hash",
    "canonical_target_key_corpus_hash",
    "full_canonical_target_key_corpus_hash",
    "full_canonical_prepared_s_key_corpus_hash",
    "selected_prepared_s_key_corpus_hash",
    "selected_target_key_corpus_hash", "iteration_subset_hash",
    "qualification_subset_hash", "semantic_tolerance_config_hash",
    "qualification_selection_config_hash",
    "prepared_s_setup_index_rds_sha256",
    "target_state_index_rds_sha256"
  )
  input_sha_values <- list(
    phase1_input_bundle_hash = inputs$phase1_input_bundle_hash,
    dataset_file_sha256 = inputs$dataset_file_sha256,
    dataset_matrix_sha256 = inputs$dataset_sha256,
    phase1_manifest_dataset_file_sha256 =
      inputs$manifest$dataset_file_sha256,
    phase1_manifest_dataset_matrix_sha256 =
      inputs$manifest$dataset_matrix_sha256,
    canonical_logical_census_hash =
      inputs$manifest$canonical_logical_census_hash,
    canonical_target_key_corpus_hash =
      inputs$manifest$canonical_key_corpus_hash
  )
  environment_fields <- c(
    "R_version", "mgcv_version", "BLAS_identity", "LAPACK_identity"
  )
  shard_names <- paste0("shard_", seq_len(contract$shard_count) - 1L)
  shard_sizes <- summary$shard_rds_size_bytes
  shard_sizes_clean <-
    fastkpc_full_cuda_fixed_sp_is_exact_named_bare_list(
      shard_sizes, shard_names
    ) && all(vapply(shard_sizes, function(value) {
      fastkpc_full_cuda_fixed_sp_is_bare_scalar(value, "integer") &&
        value > 0L
    }, logical(1L)))
  timing_fields <- c("stage_timing_total_seconds", "elapsed_seconds")
  timings_clean <- all(vapply(timing_fields, function(name) {
    fastkpc_full_cuda_fixed_sp_is_bare_nonnegative_finite_numeric(
      summary[[name]]
    )
  }, logical(1L)))
  shared_counts <- intersect(manifest_counts, summary_counts)
  identity_clean <-
    shard_sizes_clean && timings_clean &&
    bare_values(manifest, manifest_character_fields, "character") &&
    all(vapply(manifest_character_fields, function(name) {
      nzchar(manifest[[name]])
    }, logical(1L))) &&
    bare_values(summary, summary_character_fields, "character") &&
    all(vapply(summary_character_fields, function(name) {
      nzchar(summary[[name]])
    }, logical(1L))) &&
    bare_values(manifest, manifest_flags, "logical") &&
    bare_values(summary, summary_flags, "logical") &&
    bare_values(manifest, manifest_counts, "integer") &&
    bare_values(summary, summary_counts, "integer") &&
    exact_values(manifest, manifest_flag_values) &&
    exact_values(summary, summary_flag_values) &&
    exact_values(manifest, manifest_count_values) &&
    exact_values(summary, summary_count_values) &&
    all(vapply(shared_counts, function(name) {
      identical(manifest[[name]], summary[[name]])
    }, logical(1L))) &&
    all(vapply(manifest_sha_fields, function(name) {
      fastkpc_full_cuda_fixed_sp_is_bare_sha256(manifest[[name]])
    }, logical(1L))) &&
    all(vapply(input_sha_values,
               fastkpc_full_cuda_fixed_sp_is_bare_sha256, logical(1L))) &&
    all(vapply(environment_fields, function(name) {
      fastkpc_full_cuda_fixed_sp_is_bare_scalar(
        inputs$manifest[[name]], "character"
      ) && nzchar(inputs$manifest[[name]]) &&
        identical(manifest[[name]], inputs$manifest[[name]])
    }, logical(1L))) &&
    identical(manifest$schema_version,
              contract$phase2_artifact_schema_version) &&
    identical(manifest$phase2_input_schema_version,
              contract$phase2_input_schema_version) &&
    identical(manifest$prepared_s_setup_schema_version,
              contract$prepared_s_setup_schema_version) &&
    identical(manifest$target_state_schema_version,
              contract$target_state_schema_version) &&
    identical(manifest$run_scope, contract$run_scope) &&
    identical(summary$run_scope, contract$run_scope) &&
    identical(manifest$parity_scope, contract$parity_scope) &&
    identical(summary$parity_scope, contract$parity_scope) &&
    identical(manifest$unsupported_scope, contract$unsupported_scope) &&
    identical(summary$unsupported_scope, contract$unsupported_scope) &&
    identical(manifest$new_candidate_graph_gate,
              contract$new_candidate_graph_gate) &&
    identical(summary$new_candidate_graph_gate,
              contract$new_candidate_graph_gate) &&
    identical(manifest$census_dir, contract$census_dir) &&
    identical(manifest$data_path, contract$data_path) &&
    identical(manifest$source_commit, contract$phase2_source_commit) &&
    identical(manifest$phase1_input_bundle_hash,
              inputs$phase1_input_bundle_hash) &&
    identical(manifest$dataset_file_sha256,
              inputs$dataset_file_sha256) &&
    identical(manifest$dataset_file_sha256,
              inputs$manifest$dataset_file_sha256) &&
    identical(manifest$dataset_matrix_sha256, inputs$dataset_sha256) &&
    identical(manifest$dataset_matrix_sha256,
              inputs$manifest$dataset_matrix_sha256) &&
    identical(manifest$canonical_logical_census_hash,
              inputs$manifest$canonical_logical_census_hash) &&
    identical(manifest$canonical_target_key_corpus_hash,
              inputs$manifest$canonical_key_corpus_hash) &&
    identical(manifest$full_canonical_target_key_corpus_hash,
              manifest$canonical_target_key_corpus_hash) &&
    identical(manifest$selected_target_key_corpus_hash,
              manifest$canonical_target_key_corpus_hash) &&
    identical(manifest$full_canonical_prepared_s_key_corpus_hash,
              contract$full_canonical_prepared_s_key_corpus_hash) &&
    identical(manifest$selected_prepared_s_key_corpus_hash,
              manifest$full_canonical_prepared_s_key_corpus_hash) &&
    identical(manifest$iteration_subset_hash,
              contract$iteration_subset_hash) &&
    identical(manifest$qualification_subset_hash,
              contract$qualification_subset_hash) &&
    identical(manifest$semantic_tolerance_config_hash,
              contract$semantic_tolerance_config_hash) &&
    identical(manifest$qualification_selection_config_hash,
              contract$qualification_selection_config_hash) &&
    identical(manifest$prepared_s_setup_index_rds_sha256,
              contract$prepared_s_setup_index_rds_sha256) &&
    identical(manifest$target_state_index_rds_sha256,
              contract$target_state_index_rds_sha256)
  fastkpc_full_cuda_fixed_sp_require(
    identity_clean, "Phase 2 Prepared-S identity does not match authenticated inputs"
  )
  invisible(TRUE)
}

.fastkpc_full_cuda_phase3_catalog_authority_registry <-
  new.env(parent = emptyenv())

.fastkpc_full_cuda_phase3_catalog_authority_registry_max_entries <- function() {
  32L
}

.fastkpc_full_cuda_phase3_catalog_authority_fields <- function() {
  c(
    "schema_version", "phase0_dir", "phase1_dir", "phase2_dir", "data_path",
    "phase0_manifest_hash", "phase1_manifest_hash", "phase2_manifest_hash",
    "dataset_file_sha256", "dataset_matrix_sha256",
    "canonical_setup_corpus_hash", "canonical_target_corpus_hash",
    "phase0_source_commit", "phase1_source_commit", "phase2_source_commit",
    "phase2_R_version", "phase2_mgcv_version"
  )
}

.fastkpc_full_cuda_phase3_catalog_authority_hash <- function(authority) {
  fastkpc_full_cuda_census_named_metadata_hash(
    authority[.fastkpc_full_cuda_phase3_catalog_authority_fields()]
  )
}

.fastkpc_full_cuda_phase3_validate_catalog_authority <- function(authority) {
  fields <- .fastkpc_full_cuda_phase3_catalog_authority_fields()
  if (!is.list(authority) || is.object(authority) ||
      !identical(names(authority), c(fields, "sha256")) ||
      anyDuplicated(names(authority))) {
    stop("Phase 3 catalog authority is malformed", call. = FALSE)
  }
  for (field in c(
    "phase0_manifest_hash", "phase1_manifest_hash", "phase2_manifest_hash",
    "dataset_file_sha256", "dataset_matrix_sha256",
    "canonical_setup_corpus_hash", "canonical_target_corpus_hash", "sha256"
  )) {
    fastkpc_full_cuda_fixed_sp_require(
      fastkpc_full_cuda_fixed_sp_is_bare_sha256(authority[[field]]),
      paste0("Phase 3 catalog authority hash is malformed: ", field)
    )
  }
  for (field in c(
    "schema_version", "phase0_dir", "phase1_dir", "phase2_dir", "data_path",
    "phase0_source_commit", "phase1_source_commit", "phase2_source_commit",
    "phase2_R_version", "phase2_mgcv_version"
  )) {
    fastkpc_full_cuda_fixed_sp_require(
      fastkpc_full_cuda_fixed_sp_is_bare_scalar(authority[[field]], "character") &&
        nzchar(authority[[field]]),
      paste0("Phase 3 catalog authority value is malformed: ", field)
    )
  }
  fastkpc_full_cuda_fixed_sp_require(
    identical(
      authority$schema_version,
      "full-cuda-ci-phase3-catalog-authority-v1"
    ),
    "Phase 3 catalog authority schema is unsupported"
  )
  fastkpc_full_cuda_fixed_sp_require(
    identical(
      authority$sha256,
      .fastkpc_full_cuda_phase3_catalog_authority_hash(authority)
    ),
    "Phase 3 catalog authority canonical hash mismatch"
  )
  authority
}

.fastkpc_full_cuda_phase3_catalog_authority_file_paths <- function(
    authority) {
  authority <- .fastkpc_full_cuda_phase3_validate_catalog_authority(authority)
  paths <- c(
    phase0_manifest = file.path(authority$phase0_dir, "manifest.json"),
    phase1_manifest = file.path(authority$phase1_dir, "manifest.json"),
    phase2_manifest = file.path(authority$phase2_dir, "manifest.json"),
    data = authority$data_path
  )
  normalized <- unname(vapply(
    paths, normalizePath, character(1L), winslash = "/", mustWork = TRUE
  ))
  stats::setNames(normalized, names(paths))
}

.fastkpc_full_cuda_phase3_catalog_authority_file_records <- function(
    authority) {
  paths <- .fastkpc_full_cuda_phase3_catalog_authority_file_paths(authority)
  fastkpc_full_cuda_fixed_sp_require(
    all(file.exists(paths)) && !any(dir.exists(paths)),
    "Phase 3 catalog authority file identity is unavailable"
  )
  info <- file.info(paths, extra_cols = FALSE)
  fastkpc_full_cuda_fixed_sp_require(
    is.data.frame(info) && nrow(info) == length(paths) &&
      !anyNA(info$size) && !anyNA(info$mtime) && !anyNA(info$ctime),
    "Phase 3 catalog authority file identity is malformed"
  )
  manifest_rows <- names(paths) != "data"
  sha256 <- rep(NA_character_, length(paths))
  sha256[manifest_rows] <- unname(vapply(
    paths[manifest_rows],
    fastkpc_full_cuda_fixed_sp_sha256_file,
    character(1L)
  ))
  data.frame(
    logical_path = names(paths),
    path = unname(paths),
    size = as.character(info$size),
    mtime = sprintf("%.6f", as.numeric(info$mtime)),
    ctime = sprintf("%.6f", as.numeric(info$ctime)),
    sha256 = sha256,
    stringsAsFactors = FALSE
  )
}

.fastkpc_full_cuda_phase3_catalog_authority_attach_registry_metadata <-
  function(authority) {
    authority <- .fastkpc_full_cuda_phase3_validate_catalog_authority(authority)
    records <- .fastkpc_full_cuda_phase3_catalog_authority_file_records(
      authority
    )
    attr(authority, "file_records") <- records
    attr(authority, "registered_at") <- as.numeric(Sys.time())
    authority
  }

.fastkpc_full_cuda_phase3_catalog_authority_revalidate_files <- function(
    authority) {
  authority <- .fastkpc_full_cuda_phase3_validate_catalog_authority(authority)
  records <- attr(authority, "file_records", exact = TRUE)
  fastkpc_full_cuda_fixed_sp_require(
    !is.null(records),
    "Phase 3 catalog authority file identity is missing"
  )
  current <- .fastkpc_full_cuda_phase3_catalog_authority_file_records(
    authority
  )
  fastkpc_full_cuda_fixed_sp_require(
    identical(records, current),
    "Phase 3 catalog authority file identity changed"
  )
  invisible(TRUE)
}

.fastkpc_full_cuda_phase3_prune_catalog_authority_registry <- function(
    protect = character()) {
  registry <- .fastkpc_full_cuda_phase3_catalog_authority_registry
  limit <- .fastkpc_full_cuda_phase3_catalog_authority_registry_max_entries()
  keys <- ls(registry, all.names = TRUE)
  if (length(keys) <= limit) return(invisible(keys))
  ages <- vapply(keys, function(key) {
    value <- get(key, envir = registry, inherits = FALSE)
    registered_at <- attr(value, "registered_at", exact = TRUE)
    if (is.null(registered_at) || length(registered_at) != 1L ||
        !is.finite(registered_at)) {
      -Inf
    } else registered_at
  }, numeric(1L))
  removable <- keys[order(ages, method = "radix")]
  removable <- removable[!removable %in% protect]
  while (length(ls(registry, all.names = TRUE)) > limit &&
         length(removable) > 0L) {
    rm(list = removable[[1L]], envir = registry)
    removable <- removable[-1L]
  }
  invisible(ls(registry, all.names = TRUE))
}

.fastkpc_full_cuda_phase3_catalog_manifest_hash <- function(
    hashes, candidates, label) {
  if (is.data.frame(hashes)) {
    required_fields <- c("logical_path", "actual_sha256")
    fastkpc_full_cuda_fixed_sp_require(
      all(required_fields %in% names(hashes)) &&
        nrow(hashes) > 0L &&
        !anyNA(hashes$logical_path) &&
        !anyDuplicated(hashes$logical_path) &&
        fastkpc_full_cuda_fixed_sp_is_bare_sha256_vector(
          as.character(hashes$actual_sha256)
        ),
      paste0(label, " hash surface is malformed")
    )
    present <- candidates[candidates %in% hashes$logical_path]
    fastkpc_full_cuda_fixed_sp_require(
      length(present) == 1L,
      paste0(label, " hash is unavailable from authenticated file validation")
    )
    row <- match(present[[1L]], hashes$logical_path)
    return(as.character(hashes$actual_sha256[[row]]))
  }
  fastkpc_full_cuda_fixed_sp_require(
    typeof(hashes) == "character" && !is.object(hashes) &&
      !is.null(names(hashes)) && !anyDuplicated(names(hashes)) &&
      !anyNA(hashes) && all(grepl("^[0-9a-f]{64}$", hashes)),
    paste0(label, " hash surface is malformed")
  )
  present <- candidates[candidates %in% names(hashes)]
  fastkpc_full_cuda_fixed_sp_require(
    length(present) == 1L,
    paste0(label, " hash is unavailable from authenticated file validation")
  )
  unname(hashes[[present[[1L]]]])
}

.fastkpc_full_cuda_phase3_build_catalog_authority <- function(
    phase0_dir, phase1_dir, phase2_dir, data_path, require_full = TRUE) {
  require_flag <- is.logical(require_full) && length(require_full) == 1L &&
    !is.object(require_full) && is.null(attributes(require_full)) &&
    !is.na(require_full)
  fastkpc_full_cuda_fixed_sp_require(
    require_flag, "require_full must be a logical scalar"
  )
  fastkpc_full_cuda_fixed_sp_require(
    isTRUE(require_full),
    "fixed-sp catalog only supports the full Phase 2 artifact"
  )
  phase0_dir <- normalizePath(phase0_dir, winslash = "/", mustWork = TRUE)
  phase1_dir <- normalizePath(phase1_dir, winslash = "/", mustWork = TRUE)
  phase2_dir <- normalizePath(phase2_dir, winslash = "/", mustWork = TRUE)
  data_path <- normalizePath(data_path, winslash = "/", mustWork = TRUE)

  phase0_inputs <- fastkpc_full_cuda_census_load_inputs(
    phase0_dir, data_path
  )
  phase0 <- phase0_inputs$oracle
  fastkpc_full_cuda_fixed_sp_require(
    isTRUE(phase0$summary$pass) &&
      fastkpc_full_cuda_fixed_sp_is_bare_scalar(
        phase0$manifest$schema_version, "character"
      ) && identical(
        phase0$manifest$schema_version, "full-cuda-ci-oracle-v1"
      ),
    "Phase 0 oracle gate is not authenticated"
  )
  inputs <- fastkpc_full_cuda_prepared_s_load_inputs(phase1_dir, data_path)
  fastkpc_full_cuda_fixed_sp_require(
    isTRUE(inputs$summary$pass) &&
      fastkpc_full_cuda_fixed_sp_is_bare_scalar(
        inputs$manifest$schema_version, "character"
      ) && identical(
        inputs$manifest$schema_version,
        "full-cuda-ci-workload-census-artifact-v1"
      ),
    "Phase 1 workload census gate is not authenticated"
  )
  fastkpc_full_cuda_fixed_sp_require(
    fastkpc_full_cuda_fixed_sp_is_bare_sha256(
      phase0_inputs$oracle_input_bundle_sha256
    ) && fastkpc_full_cuda_fixed_sp_is_bare_sha256(
      inputs$manifest$oracle_input_bundle_sha256
    ) && fastkpc_full_cuda_fixed_sp_is_bare_scalar(
      phase0$manifest$source_commit, "character"
    ) && fastkpc_full_cuda_fixed_sp_is_bare_scalar(
      inputs$manifest$phase0_source_commit, "character"
    ) && identical(
      phase0_inputs$oracle_input_bundle_sha256,
      inputs$manifest$oracle_input_bundle_sha256
    ) && identical(
      phase0$manifest$source_commit,
      inputs$manifest$phase0_source_commit
    ),
    "Phase 0 oracle identity does not match authenticated Phase 1 inputs"
  )

  contract <- fastkpc_full_cuda_fixed_sp_catalog_contract()
  captured <- fastkpc_full_cuda_fixed_sp_capture_phase2_files(
    phase2_dir, contract
  )
  phase2_objects <- fastkpc_full_cuda_fixed_sp_parse_phase2_files(captured)
  summary <- phase2_objects$summary
  manifest <- phase2_objects$manifest
  fastkpc_full_cuda_fixed_sp_validate_phase2_identity(
    summary, manifest, inputs, contract
  )
  setup_index <- phase2_objects$setup_index
  fastkpc_full_cuda_fixed_sp_require(
    all(c("same_S_group_id", "prepared_s_key_sha256") %in%
          names(setup_index)) && nrow(setup_index) == contract$selected_group_count &&
      fastkpc_full_cuda_fixed_sp_is_bare_sha256_vector(
        setup_index$same_S_group_id
      ) &&
      !anyDuplicated(setup_index$same_S_group_id) &&
      fastkpc_full_cuda_fixed_sp_is_bare_sha256_vector(
        setup_index$prepared_s_key_sha256
      ) &&
      !anyDuplicated(setup_index$prepared_s_key_sha256),
    "Phase 2 Prepared-S setup index is invalid"
  )
  setup_index <- setup_index[order(
    setup_index$prepared_s_key_sha256, method = "radix"
  ), , drop = FALSE]
  rownames(setup_index) <- NULL

  phase0_manifest_hash <- .fastkpc_full_cuda_phase3_catalog_manifest_hash(
    phase0_inputs$oracle_input_hashes,
    c("oracle/manifest.json", "manifest.json"),
    "Phase 0 manifest"
  )
  phase1_manifest_hash <- .fastkpc_full_cuda_phase3_catalog_manifest_hash(
    inputs$input_hashes,
    c("phase1/manifest.json", "manifest.json"),
    "Phase 1 manifest"
  )
  phase2_manifest_hash <- .fastkpc_full_cuda_phase3_catalog_manifest_hash(
    captured$hashes,
    c("manifest.json"),
    "Phase 2 manifest"
  )
  authority <- list(
    schema_version = "full-cuda-ci-phase3-catalog-authority-v1",
    phase0_dir = phase0_dir,
    phase1_dir = phase1_dir,
    phase2_dir = phase2_dir,
    data_path = data_path,
    phase0_manifest_hash = phase0_manifest_hash,
    phase1_manifest_hash = phase1_manifest_hash,
    phase2_manifest_hash = phase2_manifest_hash,
    dataset_file_sha256 = as.character(inputs$dataset_file_sha256),
    dataset_matrix_sha256 = as.character(inputs$dataset_sha256),
    canonical_setup_corpus_hash = as.character(
      manifest$full_canonical_prepared_s_key_corpus_hash
    ),
    canonical_target_corpus_hash = as.character(
      manifest$full_canonical_target_key_corpus_hash
    ),
    phase0_source_commit = as.character(phase0$manifest$source_commit),
    phase1_source_commit = as.character(inputs$manifest$source_commit),
    phase2_source_commit = as.character(manifest$source_commit),
    phase2_R_version = as.character(manifest$R_version),
    phase2_mgcv_version = as.character(manifest$mgcv_version)
  )
  authority$sha256 <- .fastkpc_full_cuda_phase3_catalog_authority_hash(
    authority
  )
  authority <- .fastkpc_full_cuda_phase3_validate_catalog_authority(authority)
  list(
    catalog = list(
      phase0 = phase0,
      inputs = inputs,
      phase0_dir = phase0_dir,
      phase1_dir = phase1_dir,
      phase2_dir = phase2_dir,
      data_path = data_path,
      phase0_manifest_hash = phase0_manifest_hash,
      phase1_manifest_hash = phase1_manifest_hash,
      phase2_file_hashes = captured$hashes,
      phase2_summary = summary,
      phase2_manifest = manifest,
      catalog_contract = contract,
      setup_index = setup_index,
      scopes = phase2_objects$scopes,
      semantic_files = names(contract$phase2_file_sha256)
    ),
    authority = authority
  )
}

.fastkpc_full_cuda_phase3_catalog_authority_token <- function(authority_sha256) {
  token <- new.env(parent = emptyenv())
  assign(
    "schema_version",
    "full-cuda-ci-phase3-catalog-authority-token-v1",
    envir = token
  )
  assign("authority_sha256", authority_sha256, envir = token)
  lockBinding("schema_version", token)
  lockBinding("authority_sha256", token)
  lockEnvironment(token, bindings = TRUE)
  token
}

.fastkpc_full_cuda_phase3_register_catalog_authority <- function(authority) {
  authority <- .fastkpc_full_cuda_phase3_validate_catalog_authority(authority)
  authority <- .fastkpc_full_cuda_phase3_catalog_authority_attach_registry_metadata(
    authority
  )
  registry <- .fastkpc_full_cuda_phase3_catalog_authority_registry
  registry[[authority$sha256]] <- authority
  .fastkpc_full_cuda_phase3_prune_catalog_authority_registry(
    protect = authority$sha256
  )
  .fastkpc_full_cuda_phase3_catalog_authority_token(authority$sha256)
}

.fastkpc_full_cuda_phase3_extract_catalog_authority <- function(catalog) {
  token <- catalog$phase3_catalog_authority_token
  fastkpc_full_cuda_fixed_sp_require(
    is.environment(token) && environmentIsLocked(token) &&
      identical(ls(token, all.names = TRUE), c("authority_sha256", "schema_version")) &&
      identical(token$schema_version,
                "full-cuda-ci-phase3-catalog-authority-token-v1") &&
      fastkpc_full_cuda_fixed_sp_is_bare_sha256(token$authority_sha256),
    "Phase 3 catalog authority token is missing or malformed"
  )
  registry <- .fastkpc_full_cuda_phase3_catalog_authority_registry
  fastkpc_full_cuda_fixed_sp_require(
    exists(token$authority_sha256, envir = registry, inherits = FALSE),
    "Phase 3 catalog authority registry entry is missing"
  )
  authority <- get(token$authority_sha256, envir = registry, inherits = FALSE)
  authority <- .fastkpc_full_cuda_phase3_validate_catalog_authority(authority)
  attr(authority, "registered_at") <- as.numeric(Sys.time())
  assign(token$authority_sha256, authority, envir = registry)
  authority
}

.fastkpc_full_cuda_phase3_catalog_authority_snapshot <- function(catalog) {
  fastkpc_full_cuda_fixed_sp_require(
    is.list(catalog) && !is.object(catalog) &&
      is.list(catalog$phase0) && is.list(catalog$inputs) &&
      is.list(catalog$phase2_manifest) &&
      fastkpc_full_cuda_fixed_sp_is_bare_sha256(catalog$phase0_manifest_hash) &&
      fastkpc_full_cuda_fixed_sp_is_bare_sha256(catalog$phase1_manifest_hash) &&
      typeof(catalog$phase2_file_hashes) == "character" &&
      !is.object(catalog$phase2_file_hashes) &&
      !is.null(names(catalog$phase2_file_hashes)) &&
      "manifest.json" %in% names(catalog$phase2_file_hashes),
    "Phase 3 catalog does not expose authenticated authority fields"
  )
  authority <- list(
    schema_version = "full-cuda-ci-phase3-catalog-authority-v1",
    phase0_dir = normalizePath(catalog$phase0_dir, winslash = "/", mustWork = TRUE),
    phase1_dir = normalizePath(catalog$phase1_dir, winslash = "/", mustWork = TRUE),
    phase2_dir = normalizePath(catalog$phase2_dir, winslash = "/", mustWork = TRUE),
    data_path = normalizePath(catalog$data_path, winslash = "/", mustWork = TRUE),
    phase0_manifest_hash = unname(catalog$phase0_manifest_hash),
    phase1_manifest_hash = unname(catalog$phase1_manifest_hash),
    phase2_manifest_hash = unname(as.character(
      catalog$phase2_file_hashes[["manifest.json"]]
    )),
    dataset_file_sha256 = as.character(catalog$inputs$dataset_file_sha256),
    dataset_matrix_sha256 = as.character(catalog$inputs$dataset_sha256),
    canonical_setup_corpus_hash = as.character(
      catalog$phase2_manifest$full_canonical_prepared_s_key_corpus_hash
    ),
    canonical_target_corpus_hash = as.character(
      catalog$phase2_manifest$full_canonical_target_key_corpus_hash
    ),
    phase0_source_commit = as.character(catalog$phase0$manifest$source_commit),
    phase1_source_commit = as.character(catalog$inputs$manifest$source_commit),
    phase2_source_commit = as.character(catalog$phase2_manifest$source_commit),
    phase2_R_version = as.character(catalog$phase2_manifest$R_version),
    phase2_mgcv_version = as.character(catalog$phase2_manifest$mgcv_version)
  )
  authority$sha256 <- .fastkpc_full_cuda_phase3_catalog_authority_hash(
    authority
  )
  .fastkpc_full_cuda_phase3_validate_catalog_authority(authority)
}

fastkpc_full_cuda_phase3_discover_catalog_authority <- function(catalog) {
  stored <- .fastkpc_full_cuda_phase3_extract_catalog_authority(catalog)
  .fastkpc_full_cuda_phase3_catalog_authority_revalidate_files(stored)
  current <- .fastkpc_full_cuda_phase3_catalog_authority_snapshot(catalog)
  fastkpc_full_cuda_fixed_sp_require(
    identical(
      current[c(.fastkpc_full_cuda_phase3_catalog_authority_fields(), "sha256")],
      stored[c(.fastkpc_full_cuda_phase3_catalog_authority_fields(), "sha256")]
    ),
    "Phase 3 catalog authority does not match authenticated catalog state"
  )
  lineage <- list(
    authenticated = TRUE,
    phase0_manifest_hash = stored$phase0_manifest_hash,
    phase1_manifest_hash = stored$phase1_manifest_hash,
    phase2_manifest_hash = stored$phase2_manifest_hash,
    dataset_file_sha256 = stored$dataset_file_sha256,
    dataset_matrix_sha256 = stored$dataset_matrix_sha256,
    canonical_setup_corpus_hash = stored$canonical_setup_corpus_hash,
    canonical_target_corpus_hash = stored$canonical_target_corpus_hash,
    phase0_source_commit = stored$phase0_source_commit,
    phase1_source_commit = stored$phase1_source_commit,
    phase2_source_commit = stored$phase2_source_commit,
    phase2_R_version = stored$phase2_R_version,
    phase2_mgcv_version = stored$phase2_mgcv_version
  )
  list(
    schema_version = stored$schema_version,
    authority_sha256 = stored$sha256,
    lineage = lineage
  )
}

fastkpc_full_cuda_phase3_deep_revalidate_catalog_authority <- function(
    catalog) {
  stored <- .fastkpc_full_cuda_phase3_extract_catalog_authority(catalog)
  rebuilt <- .fastkpc_full_cuda_phase3_build_catalog_authority(
    stored$phase0_dir,
    stored$phase1_dir,
    stored$phase2_dir,
    stored$data_path,
    require_full = TRUE
  )$authority
  fastkpc_full_cuda_fixed_sp_require(
    identical(
      rebuilt[c(.fastkpc_full_cuda_phase3_catalog_authority_fields(), "sha256")],
      stored[c(.fastkpc_full_cuda_phase3_catalog_authority_fields(), "sha256")]
    ),
    "Phase 3 catalog authority registry entry is stale"
  )
  current <- .fastkpc_full_cuda_phase3_catalog_authority_snapshot(catalog)
  fastkpc_full_cuda_fixed_sp_require(
    identical(
      current[c(.fastkpc_full_cuda_phase3_catalog_authority_fields(), "sha256")],
      stored[c(.fastkpc_full_cuda_phase3_catalog_authority_fields(), "sha256")]
    ),
    "Phase 3 catalog authority does not match authenticated catalog state"
  )
  invisible(TRUE)
}

fastkpc_full_cuda_open_fixed_sp_catalog <- function(
    phase0_dir, phase1_dir, phase2_dir, data_path, require_full = TRUE) {
  built <- .fastkpc_full_cuda_phase3_build_catalog_authority(
    phase0_dir, phase1_dir, phase2_dir, data_path, require_full = require_full
  )
  built$catalog$phase3_catalog_authority_token <-
    .fastkpc_full_cuda_phase3_register_catalog_authority(built$authority)
  built$catalog$phase3_catalog_authority_sha256 <- built$authority$sha256
  built$catalog
}

fastkpc_full_cuda_fixed_sp_scope <- function(catalog, scope) {
  fastkpc_full_cuda_fixed_sp_require(
    is.list(catalog) && !is.null(catalog$inputs) &&
      !is.null(catalog$setup_index) && !is.null(catalog$scopes) &&
      !is.null(catalog$catalog_contract),
    "fixed-sp catalog is malformed"
  )
  scope <- match.arg(scope, c("iteration", "qualification", "full"))
  if (identical(scope, "full")) {
    setup_rows <- catalog$inputs$same_s_setup_metadata
    target_rows <- catalog$inputs$target_fit_metadata
    setup_index <- catalog$setup_index
    setup_match <- match(
      setup_rows$same_S_group_id, setup_index$same_S_group_id
    )
    target_setup_match <- match(
      target_rows$same_S_group_id, setup_rows$same_S_group_id
    )
    clean <- is.data.frame(setup_rows) && is.data.frame(target_rows) &&
      is.data.frame(setup_index) && nrow(setup_rows) == 8634L &&
      nrow(target_rows) == 110617L && !anyNA(setup_match) &&
      !anyNA(target_setup_match) &&
      !anyDuplicated(setup_rows$same_S_group_id) &&
      !anyDuplicated(target_rows$residual_key_sha256)
    if (!isTRUE(clean)) {
      stop("fixed-sp full scope canonical lineage is malformed",
           call. = FALSE)
    }
    setup_rows$prepared_s_key_sha256 <-
      setup_index$prepared_s_key_sha256[setup_match]
    target_rows$prepared_s_key_sha256 <-
      setup_rows$prepared_s_key_sha256[target_setup_match]
    target_rows$condition <-
      target_rows$penalized_system_condition_at_selected_sp
    target_rows$null_dim <- as.integer(
      setup_rows$constraint_nullspace_dimension[target_setup_match]
    )
    target_rows$planned_route <- fastkpc_full_cuda_fixed_sp_route(
      condition = target_rows$condition,
      coefficient_rank = target_rows$coefficient_rank,
      null_dim = target_rows$null_dim,
      authenticated = rep(TRUE, nrow(target_rows))
    )
    setup_rows <- setup_rows[order(
      setup_rows$prepared_s_key_sha256, method = "radix"
    ), , drop = FALSE]
    target_rows <- target_rows[order(
      target_rows$prepared_s_key_sha256,
      target_rows$residual_key_sha256, method = "radix"
    ), , drop = FALSE]
    rownames(setup_rows) <- rownames(target_rows) <- NULL
    setup_hash <- fastkpc_full_cuda_census_key_set_hash(
      setup_rows$prepared_s_key_sha256
    )
    target_hash <- fastkpc_full_cuda_census_key_set_hash(
      target_rows$residual_key_sha256
    )
    if (!identical(
          setup_hash,
          catalog$phase2_manifest$full_canonical_prepared_s_key_corpus_hash
        ) || !identical(
          target_hash,
          catalog$phase2_manifest$full_canonical_target_key_corpus_hash
        )) {
      stop("fixed-sp full scope corpus authentication failed",
           call. = FALSE)
    }
    return(list(
      scope = scope, setup_rows = setup_rows, target_rows = target_rows,
      shard_ids = as.integer(seq.int(
        0L, catalog$catalog_contract$shard_count - 1L
      ))
    ))
  }
  scope_objects <- catalog$scopes[[scope]]
  setup_rows <- scope_objects$setup_rows
  selected_targets <- scope_objects$target_rows
  setup_fields <- catalog$catalog_contract$inherited_setup_fields
  target_fields <- catalog$catalog_contract$inherited_target_fields
  integer_setup_fields <- c(
    "model_matrix_nrow", "model_matrix_ncol", "constraint_rank",
    "constraint_nullspace_dimension", "model_matrix_rank", "penalty_count"
  )
  bare_nonnegative_integer <- function(value) {
    typeof(value) == "integer" && !is.object(value) &&
      is.null(attributes(value)) && !anyNA(value) && all(value >= 0L)
  }
  phase1_setup_rows <- catalog$inputs$same_s_setup_metadata
  phase1_target_rows <- catalog$inputs$target_fit_metadata
  fastkpc_full_cuda_fixed_sp_require(
    is.data.frame(setup_rows) && is.data.frame(selected_targets) &&
      identical(
        names(setup_rows), catalog$catalog_contract$setup_scope_fields
      ) && identical(
        names(selected_targets), catalog$catalog_contract$target_scope_fields
      ) &&
      fastkpc_full_cuda_fixed_sp_is_bare_sha256_vector(
        setup_rows$same_S_group_id
      ) && fastkpc_full_cuda_fixed_sp_is_bare_sha256_vector(
        selected_targets$same_S_group_id
      ) && fastkpc_full_cuda_fixed_sp_is_bare_sha256_vector(
        selected_targets$residual_key_sha256
      ) &&
      !anyDuplicated(setup_rows$same_S_group_id) &&
      !anyDuplicated(selected_targets$residual_key_sha256) &&
      all(vapply(setup_rows[integer_setup_fields],
                 bare_nonnegative_integer, logical(1L))),
    "fixed-sp scope selection is malformed"
  )
  fastkpc_full_cuda_fixed_sp_require(
    is.data.frame(phase1_setup_rows) && is.data.frame(phase1_target_rows) &&
      all(setup_fields %in% names(phase1_setup_rows)) &&
      all(target_fields %in% names(phase1_target_rows)) &&
      fastkpc_full_cuda_fixed_sp_is_bare_sha256_vector(
        phase1_setup_rows$same_S_group_id
      ) && fastkpc_full_cuda_fixed_sp_is_bare_sha256_vector(
        phase1_target_rows$same_S_group_id
      ) && fastkpc_full_cuda_fixed_sp_is_bare_sha256_vector(
        phase1_target_rows$residual_key_sha256
      ) &&
      !anyDuplicated(phase1_setup_rows$same_S_group_id) &&
      !anyDuplicated(phase1_target_rows$residual_key_sha256),
    "fixed-sp canonical lineage keys are malformed"
  )
  phase1_setup_match <- match(
    setup_rows$same_S_group_id,
    phase1_setup_rows$same_S_group_id
  )
  matched_phase1_setups <- phase1_setup_rows[
    phase1_setup_match, setup_fields, drop = FALSE
  ]
  fastkpc_full_cuda_fixed_sp_require(
    !anyNA(phase1_setup_match) &&
      all(vapply(setup_fields, function(field) {
        identical(setup_rows[[field]], matched_phase1_setups[[field]])
      }, logical(1L))),
    "fixed-sp setup lineage is inconsistent"
  )
  fastkpc_full_cuda_fixed_sp_require(
    fastkpc_full_cuda_fixed_sp_is_bare_sha256_vector(
      catalog$setup_index$same_S_group_id
    ) && fastkpc_full_cuda_fixed_sp_is_bare_sha256_vector(
      catalog$setup_index$prepared_s_key_sha256
    ) && !anyDuplicated(catalog$setup_index$same_S_group_id),
    "fixed-sp scope setup index keys are malformed"
  )
  setup_match <- match(setup_rows$same_S_group_id,
                       catalog$setup_index$same_S_group_id)
  fastkpc_full_cuda_fixed_sp_require(
    !anyNA(setup_match), "fixed-sp scope setup mapping is incomplete"
  )
  setup_rows$prepared_s_key_sha256 <-
    catalog$setup_index$prepared_s_key_sha256[setup_match]
  target_match <- match(
    selected_targets$residual_key_sha256,
    phase1_target_rows$residual_key_sha256
  )
  fastkpc_full_cuda_fixed_sp_require(
    !anyNA(target_match),
    "fixed-sp scope target mapping is incomplete"
  )
  target_rows <- phase1_target_rows[target_match, , drop = FALSE]
  fastkpc_full_cuda_fixed_sp_require(
    all(vapply(target_fields, function(field) {
      identical(selected_targets[[field]], target_rows[[field]])
    }, logical(1L))) &&
      setequal(setup_rows$same_S_group_id,
               selected_targets$same_S_group_id),
    "fixed-sp target lineage is inconsistent"
  )
  target_rows$prepared_s_key_sha256 <- catalog$setup_index$prepared_s_key_sha256[
    match(target_rows$same_S_group_id, catalog$setup_index$same_S_group_id)
  ]
  setup_null_dim <- setup_rows$constraint_nullspace_dimension[
    match(target_rows$same_S_group_id, setup_rows$same_S_group_id)
  ]
  fastkpc_full_cuda_fixed_sp_require(
    !anyNA(target_rows$prepared_s_key_sha256) && !anyNA(setup_null_dim),
    "fixed-sp scope lineage mapping is incomplete"
  )
  target_rows$condition <-
    target_rows$penalized_system_condition_at_selected_sp
  target_rows$null_dim <- setup_null_dim
  target_rows$planned_route <- fastkpc_full_cuda_fixed_sp_route(
    condition = target_rows$condition,
    coefficient_rank = target_rows$coefficient_rank,
    null_dim = target_rows$null_dim,
    authenticated = rep(TRUE, nrow(target_rows))
  )
  setup_rows <- setup_rows[order(
    setup_rows$prepared_s_key_sha256, method = "radix"
  ), , drop = FALSE]
  target_rows <- target_rows[order(
    target_rows$prepared_s_key_sha256, target_rows$residual_key_sha256,
    method = "radix"
  ), , drop = FALSE]
  rownames(setup_rows) <- NULL
  rownames(target_rows) <- NULL
  selected_rank <- match(
    setup_rows$prepared_s_key_sha256,
    catalog$setup_index$prepared_s_key_sha256
  )
  shard_ids <- sort(unique(as.integer(
    (selected_rank - 1L) %% catalog$catalog_contract$shard_count
  )))
  list(
    scope = scope,
    setup_rows = setup_rows,
    target_rows = target_rows,
    shard_ids = shard_ids
  )
}

fastkpc_full_cuda_fixed_sp_load_oracle_phase2_shards <- function(
    catalog, setup_keys, target_rows,
    shard_loader = fastkpc_full_cuda_prepared_s_read_selected_shards) {
  setup_index <- catalog$setup_index
  shard_count <- catalog$catalog_contract$shard_count
  clean <- is.data.frame(setup_index) &&
    "prepared_s_key_sha256" %in% names(setup_index) &&
    typeof(setup_index$prepared_s_key_sha256) == "character" &&
    !anyNA(setup_index$prepared_s_key_sha256) &&
    !anyDuplicated(setup_index$prepared_s_key_sha256) &&
    typeof(shard_count) == "integer" && length(shard_count) == 1L &&
    !is.na(shard_count) && shard_count > 0L && is.function(shard_loader) &&
    fastkpc_full_cuda_fixed_sp_is_bare_sha256_vector(setup_keys) &&
    length(setup_keys) > 0L && !anyDuplicated(setup_keys) &&
    identical(setup_keys, sort(setup_keys, method = "radix")) &&
    is.data.frame(target_rows) && nrow(target_rows) > 0L &&
    all(c(
      "prepared_s_key_sha256", "residual_key_sha256"
    ) %in% names(target_rows))
  if (!isTRUE(clean)) {
    stop("fixed-sp oracle Phase 2 shard request is malformed",
         call. = FALSE)
  }
  setup_rank <- match(setup_keys, setup_index$prepared_s_key_sha256)
  target_setup_rank <- match(
    as.character(target_rows$prepared_s_key_sha256), setup_keys
  )
  target_keys <- as.character(target_rows$residual_key_sha256)
  expected_target_order <- order(
    target_setup_rank, target_keys, method = "radix"
  )
  if (anyNA(setup_rank) || anyNA(target_setup_rank) ||
      !fastkpc_full_cuda_fixed_sp_is_bare_sha256_vector(target_keys) ||
      anyDuplicated(target_keys) ||
      !identical(expected_target_order, seq_len(nrow(target_rows))) ||
      !identical(
        sort(unique(as.character(target_rows$prepared_s_key_sha256)),
             method = "radix"),
        setup_keys
      )) {
    stop("fixed-sp oracle Phase 2 shard request identity is incomplete",
         call. = FALSE)
  }
  shard_ids <- sort(unique(as.integer(
    (setup_rank - 1L) %% shard_count
  )))
  started <- proc.time()[["elapsed"]]
  loaded <- shard_loader(
    shard_dir = file.path(catalog$phase2_dir, "shards"),
    inputs = catalog$inputs,
    shard_count = shard_count,
    shard_ids = shard_ids,
    setup_keys = setup_keys,
    target_keys = target_keys,
    expected_source_commit = catalog$phase2_manifest$source_commit
  )
  elapsed_ms <- as.double((proc.time()[["elapsed"]] - started) * 1000)
  loaded_clean <- is.list(loaded) && all(c(
    "prepared_s_setups", "target_states", "shard_ids"
  ) %in% names(loaded)) && is.list(loaded$prepared_s_setups) &&
    identical(names(loaded$prepared_s_setups), setup_keys) &&
    is.data.frame(loaded$target_states) &&
    all(c(
      "prepared_s_key_sha256", "residual_key_sha256"
    ) %in% names(loaded$target_states)) &&
    identical(
      as.character(loaded$target_states$residual_key_sha256), target_keys
    ) && identical(
      as.character(loaded$target_states$prepared_s_key_sha256),
      as.character(target_rows$prepared_s_key_sha256)
    ) && identical(as.integer(loaded$shard_ids), shard_ids) &&
    is.finite(elapsed_ms) && elapsed_ms >= 0
  if (!isTRUE(loaded_clean)) {
    stop("fixed-sp oracle authenticated Phase 2 shard payload is incomplete",
         call. = FALSE)
  }
  list(
    loaded = loaded,
    phase2_shard_ids = shard_ids,
    phase2_shard_load_count = as.integer(length(shard_ids)),
    phase2_shard_authentication_count = as.integer(length(shard_ids)),
    phase2_shard_load_elapsed_ms = elapsed_ms
  )
}

fastkpc_full_cuda_fixed_sp_batches_from_loaded <- function(
    catalog, selected_scope, loaded) {
  fastkpc_full_cuda_fixed_sp_require(
    is.list(selected_scope) && !is.null(selected_scope$setup_rows) &&
      !is.null(selected_scope$target_rows) && !is.null(selected_scope$shard_ids),
    "fixed-sp selected scope is malformed"
  )
  setup_keys <- selected_scope$setup_rows$prepared_s_key_sha256
  target_keys <- selected_scope$target_rows$residual_key_sha256
  fastkpc_full_cuda_fixed_sp_require(
    is.list(loaded) && all(c(
      "prepared_s_setups", "target_states", "shard_ids"
    ) %in% names(loaded)),
    "fixed-sp loaded shard payload is malformed"
  )
  setups <- loaded$prepared_s_setups
  states <- loaded$target_states
  fastkpc_full_cuda_fixed_sp_require(
    length(setups) == nrow(selected_scope$setup_rows) &&
      nrow(states) == nrow(selected_scope$target_rows) &&
      identical(names(setups), setup_keys) &&
      identical(as.character(states$residual_key_sha256), target_keys),
    "fixed-sp selected shard payload is incomplete"
  )
  batches <- lapply(setup_keys, function(setup_key) {
    setup <- setups[[setup_key]]
    selected <- states[states$prepared_s_key_sha256 == setup_key, , drop = FALSE]
    metadata <- selected_scope$target_rows[
      selected_scope$target_rows$prepared_s_key_sha256 == setup_key,
      , drop = FALSE
    ]
    fastkpc_full_cuda_fixed_sp_require(
      nrow(selected) == nrow(metadata) &&
        identical(selected$residual_key_sha256, metadata$residual_key_sha256) &&
        identical(as.character(selected$residual_key_sha256),
                  as.character(metadata$residual_key_sha256)),
      "fixed-sp target state selection is inconsistent"
    )
    materialized <- lapply(seq_len(nrow(selected)), function(index) {
      target <- fastkpc_full_cuda_materialize_target_state(
        selected[index, , drop = FALSE], catalog$inputs$data,
        catalog$inputs$dataset_sha256
      )
      fastkpc_full_cuda_validate_materialized_target_for_prepared(setup, target)
    })
    Y <- do.call(cbind, lapply(materialized, `[[`, "y"))
    SP <- do.call(cbind, lapply(materialized, `[[`, "sp"))
    rhs <- do.call(cbind, lapply(materialized, `[[`, "nullspace_projected_rhs"))
    colnames(Y) <- colnames(SP) <- colnames(rhs) <- selected$residual_key_sha256
    fastkpc_full_cuda_fixed_sp_require(
      nrow(Y) == nrow(setup$X) && ncol(Y) == nrow(selected) &&
        nrow(SP) == length(setup$penalty_blocks) &&
        ncol(SP) == ncol(Y) && nrow(rhs) ==
          setup$constraint_nullspace_dimension && ncol(rhs) == ncol(Y),
      "fixed-sp batch matrix dimensions are invalid"
    )
    list(
      setup = setup,
      target_rows = selected,
      Y = Y,
      SP = SP,
      oracle_nullspace_rhs = rhs,
      planned_route = metadata$planned_route,
      condition = metadata$condition,
      prepared_s_key_sha256 = setup_key
    )
  })
  names(batches) <- setup_keys
  fastkpc_full_cuda_fixed_sp_require(
    identical(names(batches), setup_keys) && all(vapply(
      seq_along(batches), function(index) {
        keys <- batches[[index]]$target_rows$residual_key_sha256
        identical(colnames(batches[[index]]$Y), keys) &&
          identical(colnames(batches[[index]]$SP), keys) &&
          identical(colnames(batches[[index]]$oracle_nullspace_rhs), keys)
      }, logical(1L)
    )), "fixed-sp batch order is inconsistent"
  )
  batches
}

fastkpc_full_cuda_fixed_sp_batches <- function(catalog, selected_scope) {
  fastkpc_full_cuda_fixed_sp_require(
    is.list(selected_scope) && !is.null(selected_scope$setup_rows) &&
      !is.null(selected_scope$target_rows) && !is.null(selected_scope$shard_ids),
    "fixed-sp selected scope is malformed"
  )
  setup_keys <- selected_scope$setup_rows$prepared_s_key_sha256
  target_keys <- selected_scope$target_rows$residual_key_sha256
  loaded <- fastkpc_full_cuda_prepared_s_read_selected_shards(
    shard_dir = file.path(catalog$phase2_dir, "shards"),
    inputs = catalog$inputs,
    shard_count = catalog$catalog_contract$shard_count,
    shard_ids = selected_scope$shard_ids,
    setup_keys = setup_keys,
    target_keys = target_keys,
    expected_source_commit = catalog$phase2_manifest$source_commit
  )
  fastkpc_full_cuda_fixed_sp_batches_from_loaded(
    catalog, selected_scope, loaded
  )
}

fastkpc_full_cuda_fixed_sp_native_dto_fields <- function() {
  c(
    "schema_version", "dataset_sha256", "prepared_s_key_sha256",
    "same_S_group_id", "phase1_setup_fingerprint", "provider_fingerprint",
    "semantic_fingerprint", "representation_fingerprint",
    "prepared_s_setup_schema_version", "native_dto_schema_version",
    "data_p", "n", "coefficient_dim", "null_dim", "penalty_count", "X",
    "constraint_mode", "constraint_nullspace", "gram_matrix",
    "nullspace_gram_matrix", "penalty_blocks",
    "penalty_offsets_zero_based", "penalty_ranks",
    "penalty_sp_indices_zero_based", "penalty_sp_labels", "H",
    "weights_policy", "offset_policy"
  )
}

fastkpc_full_cuda_fixed_sp_is_finite_double_matrix <- function(value) {
  attribute_names <- names(attributes(value))
  matrix_attributes_clean <- identical(attribute_names, "dim") ||
    (length(attribute_names) == 2L && !anyDuplicated(attribute_names) &&
       setequal(attribute_names, c("dim", "dimnames")))
  is.matrix(value) && typeof(value) == "double" && !is.object(value) &&
    matrix_attributes_clean && all(is.finite(value))
}

fastkpc_full_cuda_fixed_sp_is_bare_integer_vector <- function(value) {
  typeof(value) == "integer" && !is.object(value) &&
    is.null(attributes(value)) && !anyNA(value)
}

fastkpc_full_cuda_fixed_sp_is_bare_character_vector <- function(value) {
  typeof(value) == "character" && !is.object(value) &&
    is.null(attributes(value)) && !anyNA(value)
}

fastkpc_full_cuda_fixed_sp_native_dto <- function(setup) {
  setup_fields <- fastkpc_full_cuda_prepared_s_setup_field_names()
  if (!fastkpc_full_cuda_fixed_sp_is_exact_named_bare_list(
    setup, setup_fields
  )) {
    stop("Phase 3 PreparedSSetup is malformed", call. = FALSE)
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
  if (!fastkpc_full_cuda_fixed_sp_is_finite_double_matrix(setup$X)) {
    stop("Phase 3 X must be a finite double matrix", call. = FALSE)
  }

  penalty_count <- length(setup$penalty_blocks)
  block_clean <- typeof(setup$penalty_blocks) == "list" &&
    !is.object(setup$penalty_blocks) && penalty_count > 0L &&
    all(vapply(setup$penalty_blocks, function(block) {
      fastkpc_full_cuda_fixed_sp_is_finite_double_matrix(block) &&
        nrow(block) == ncol(block)
    }, logical(1L)))
  if (!block_clean) {
    stop(
      "Phase 3 penalty blocks must be finite square double matrices",
      call. = FALSE
    )
  }
  if (!fastkpc_full_cuda_fixed_sp_is_bare_integer_vector(
    setup$penalty_offsets
  )) {
    stop("Phase 3 penalty offsets must be a bare integer vector",
         call. = FALSE)
  }
  if (!fastkpc_full_cuda_fixed_sp_is_bare_integer_vector(
    setup$penalty_sp_indices
  )) {
    stop("Phase 3 penalty SP indices must be a bare integer vector",
         call. = FALSE)
  }
  if (!fastkpc_full_cuda_fixed_sp_is_bare_integer_vector(
    setup$penalty_ranks
  ) || length(setup$penalty_ranks) != penalty_count) {
    stop("Phase 3 penalty ranks are malformed", call. = FALSE)
  }
  if (!fastkpc_full_cuda_fixed_sp_is_bare_character_vector(
    setup$penalty_sp_labels
  ) || length(setup$penalty_sp_labels) != penalty_count ||
      any(!nzchar(setup$penalty_sp_labels))) {
    stop("Phase 3 penalty SP labels are malformed", call. = FALSE)
  }
  if (length(setup$penalty_offsets) != penalty_count) {
    stop("Phase 3 penalty offset count is malformed", call. = FALSE)
  }
  if (length(setup$penalty_sp_indices) != penalty_count) {
    stop("Phase 3 penalty SP index count is malformed", call. = FALSE)
  }

  coefficient_dim <- ncol(setup$X)
  for (index in seq_len(penalty_count)) {
    block_size <- nrow(setup$penalty_blocks[[index]])
    start <- setup$penalty_offsets[[index]]
    if (start < 1L || block_size > coefficient_dim ||
        start > coefficient_dim - block_size + 1L) {
      stop("Phase 3 penalty offset is out of range", call. = FALSE)
    }
  }
  if (any(setup$penalty_sp_indices < 1L |
          setup$penalty_sp_indices > penalty_count)) {
    stop("Phase 3 penalty SP index is out of range", call. = FALSE)
  }
  if (!identical(setup$penalty_sp_indices, seq_len(penalty_count))) {
    stop("Phase 3 v1 requires identity penalty-to-SP mapping",
         call. = FALSE)
  }

  lineage_fields <- c(
    "dataset_sha256", "prepared_s_key_sha256", "same_S_group_id",
    "phase1_setup_fingerprint", "provider_fingerprint",
    "semantic_fingerprint", "representation_fingerprint"
  )
  if (!all(vapply(setup[lineage_fields], function(value) {
    fastkpc_full_cuda_fixed_sp_is_bare_scalar(value, "character")
  }, logical(1L)))) {
    stop("Phase 3 PreparedSSetup lineage is malformed", call. = FALSE)
  }
  if (!fastkpc_full_cuda_fixed_sp_is_bare_scalar(
    setup$constraint_mode, "character"
  ) || !fastkpc_full_cuda_fixed_sp_is_finite_double_matrix(
    setup$gram_matrix
  )) {
    stop("Phase 3 constraint/Gram data are malformed", call. = FALSE)
  }
  if (!is.null(setup$constraint_nullspace) &&
      !fastkpc_full_cuda_fixed_sp_is_finite_double_matrix(
        setup$constraint_nullspace
      )) {
    stop("Phase 3 constraint/Gram data are malformed", call. = FALSE)
  }
  if (!is.null(setup$nullspace_gram_matrix) &&
      !fastkpc_full_cuda_fixed_sp_is_finite_double_matrix(
        setup$nullspace_gram_matrix
      )) {
    stop("Phase 3 constraint/Gram data are malformed", call. = FALSE)
  }

  fastkpc_full_cuda_validate_prepared_s_for_adapter(setup)
  contract <- fastkpc_full_cuda_fixed_sp_contract()
  canonical <- fastkpc_full_cuda_canonical_contract()
  setup_n <- as.integer(nrow(setup$X))
  canonical_identity_clean <-
    fastkpc_full_cuda_fixed_sp_is_bare_scalar(canonical$n, "integer") &&
    fastkpc_full_cuda_fixed_sp_is_bare_scalar(canonical$p, "integer") &&
    identical(canonical$p, 48L) &&
    fastkpc_full_cuda_fixed_sp_is_bare_sha256(canonical$data_hash) &&
    identical(setup$dataset_sha256, canonical$data_hash) &&
    identical(setup_n, canonical$n)
  if (!canonical_identity_clean) {
    stop("Phase 3 canonical dataset identity mismatch", call. = FALSE)
  }
  data_p <- canonical$p
  list(
    schema_version = contract$native_dto_schema_version,
    dataset_sha256 = setup$dataset_sha256,
    prepared_s_key_sha256 = setup$prepared_s_key_sha256,
    same_S_group_id = setup$same_S_group_id,
    phase1_setup_fingerprint = setup$phase1_setup_fingerprint,
    provider_fingerprint = setup$provider_fingerprint,
    semantic_fingerprint = setup$semantic_fingerprint,
    representation_fingerprint = setup$representation_fingerprint,
    prepared_s_setup_schema_version = setup$schema_version,
    native_dto_schema_version = contract$native_dto_schema_version,
    data_p = data_p,
    n = setup_n,
    coefficient_dim = as.integer(coefficient_dim),
    null_dim = as.integer(setup$constraint_nullspace_dimension),
    penalty_count = as.integer(penalty_count),
    X = setup$X,
    constraint_mode = setup$constraint_mode,
    constraint_nullspace = setup$constraint_nullspace,
    gram_matrix = setup$gram_matrix,
    nullspace_gram_matrix = setup$nullspace_gram_matrix,
    penalty_blocks = setup$penalty_blocks,
    penalty_offsets_zero_based = setup$penalty_offsets - 1L,
    penalty_ranks = setup$penalty_ranks,
    penalty_sp_indices_zero_based = setup$penalty_sp_indices - 1L,
    penalty_sp_labels = setup$penalty_sp_labels,
    H = setup$H,
    weights_policy = setup$weights_policy,
    offset_policy = setup$offset_policy
  )
}

fastkpc_full_cuda_fixed_sp_validate_numeric_inputs <- function(Y, SP) {
  if (!fastkpc_full_cuda_fixed_sp_is_finite_double_matrix(Y) ||
      !fastkpc_full_cuda_fixed_sp_is_finite_double_matrix(SP) ||
      any(SP < 0)) {
    stop("Phase 3 native target batch numeric inputs are malformed",
         call. = FALSE)
  }
  invisible(TRUE)
}

fastkpc_full_cuda_fixed_sp_native_batch <- function(batch, dto) {
  batch_fields <- c(
    "setup", "target_rows", "Y", "SP", "oracle_nullspace_rhs",
    "planned_route", "condition", "prepared_s_key_sha256"
  )
  if (!fastkpc_full_cuda_fixed_sp_is_exact_named_bare_list(
    batch, batch_fields
  )) {
    stop("Phase 3 native source batch is malformed", call. = FALSE)
  }
  if (!fastkpc_full_cuda_fixed_sp_is_exact_named_bare_list(
    dto, fastkpc_full_cuda_fixed_sp_native_dto_fields()
  )) {
    stop("Phase 3 native DTO is malformed", call. = FALSE)
  }
  expected_dto <- fastkpc_full_cuda_fixed_sp_native_dto(batch$setup)
  if (!identical(dto, expected_dto) ||
      !fastkpc_full_cuda_fixed_sp_is_bare_sha256(
        batch$prepared_s_key_sha256
      ) || !identical(
        batch$prepared_s_key_sha256, dto$prepared_s_key_sha256
      )) {
    stop("Phase 3 native batch lineage mismatch", call. = FALSE)
  }

  Y <- batch$Y
  SP <- batch$SP
  fastkpc_full_cuda_fixed_sp_validate_numeric_inputs(Y, SP)
  target_count <- ncol(Y)
  if (target_count < 1L || !identical(nrow(Y), dto$n) ||
      !identical(dim(SP), c(dto$penalty_count, target_count)) ||
      !is.data.frame(batch$target_rows) ||
      nrow(batch$target_rows) != target_count) {
    stop("Phase 3 native target batch is malformed", call. = FALSE)
  }

  target_keys <- batch$target_rows$residual_key_sha256
  if (!fastkpc_full_cuda_fixed_sp_is_bare_sha256_vector(target_keys) ||
      length(target_keys) != target_count) {
    stop("Phase 3 target keys are malformed", call. = FALSE)
  }
  if (!identical(colnames(Y), target_keys) ||
      !identical(colnames(SP), target_keys)) {
    stop("Phase 3 native target order mismatch", call. = FALSE)
  }
  if (anyDuplicated(target_keys)) {
    stop("Phase 3 duplicate target identity", call. = FALSE)
  }

  residual_payloads <- batch$target_rows$residual_key_payload
  targets <- batch$target_rows$target
  sorted_S <- batch$setup$sorted_S
  residual_metadata_clean <-
    fastkpc_full_cuda_fixed_sp_is_bare_scalar(dto$data_p, "integer") &&
    dto$data_p >= 1L &&
    fastkpc_full_cuda_fixed_sp_is_bare_character_vector(
      residual_payloads
    ) && length(residual_payloads) == target_count &&
    all(nzchar(residual_payloads)) &&
    fastkpc_full_cuda_fixed_sp_is_bare_integer_vector(targets) &&
    length(targets) == target_count &&
    all(targets >= 1L & targets <= dto$data_p) &&
    fastkpc_full_cuda_fixed_sp_is_bare_integer_vector(sorted_S) &&
    all(sorted_S >= 1L & sorted_S <= dto$data_p)
  if (!residual_metadata_clean) {
    stop("Phase 3 residual key metadata is malformed", call. = FALSE)
  }
  expected_same_s_payload <- fastkpc_full_cuda_census_same_s_payload(
    S = sorted_S,
    formula_class = batch$setup$formula_class,
    data_hash = dto$dataset_sha256,
    n = dto$n,
    p = dto$data_p
  )
  expected_same_s_group_id <- fastkpc_full_cuda_census_hash_utf8(
    expected_same_s_payload
  )
  same_s_group_ids <- batch$target_rows$same_S_group_id
  same_s_identity_exact <-
    identical(dto$same_S_group_id, expected_same_s_group_id) &&
    fastkpc_full_cuda_fixed_sp_is_bare_sha256_vector(same_s_group_ids) &&
    length(same_s_group_ids) == target_count &&
    identical(
      same_s_group_ids, rep(expected_same_s_group_id, target_count)
    )
  if (!same_s_identity_exact) {
    stop("Phase 3 same-S group identity mismatch", call. = FALSE)
  }
  residual_semantics_exact <- all(vapply(
    seq_len(target_count), function(index) {
      payload <- residual_payloads[[index]]
      payload_p <- fastkpc_full_cuda_prepared_s_payload_integer(payload, "p")
      expected_payload <- fastkpc_full_cuda_census_residual_payload(
        target = targets[[index]],
        S = sorted_S,
        formula_class = batch$setup$formula_class,
        data_hash = dto$dataset_sha256,
        n = dto$n,
        p = dto$data_p
      )
      identical(payload_p, dto$data_p) &&
        identical(payload, expected_payload) && identical(
        target_keys[[index]],
        fastkpc_full_cuda_census_hash_utf8(expected_payload)
      )
    }, logical(1L)
  ))
  if (!residual_semantics_exact) {
    stop("Phase 3 residual key serialization mismatch", call. = FALSE)
  }
  if (!identical(target_keys, sort(target_keys, method = "radix"))) {
    stop("Phase 3 canonical target order mismatch", call. = FALSE)
  }

  lineage_clean <-
    fastkpc_full_cuda_fixed_sp_is_bare_sha256_vector(
      batch$target_rows$prepared_s_key_sha256
    ) && identical(
      batch$target_rows$prepared_s_key_sha256,
      rep(dto$prepared_s_key_sha256, target_count)
    ) && fastkpc_full_cuda_fixed_sp_is_bare_sha256_vector(
      batch$target_rows$same_S_group_id
    ) && identical(
      batch$target_rows$same_S_group_id,
      rep(dto$same_S_group_id, target_count)
    ) && fastkpc_full_cuda_fixed_sp_is_bare_sha256_vector(
      batch$target_rows$phase1_setup_fingerprint
    ) && identical(
      batch$target_rows$phase1_setup_fingerprint,
      rep(dto$phase1_setup_fingerprint, target_count)
    )
  if (!lineage_clean) {
    stop("Phase 3 native batch lineage mismatch", call. = FALSE)
  }

  sp_name_order_exact <- is.list(batch$target_rows$selected_sp_names) &&
    length(batch$target_rows$selected_sp_names) == target_count &&
    all(vapply(batch$target_rows$selected_sp_names, function(value) {
      fastkpc_full_cuda_fixed_sp_is_bare_character_vector(value) &&
        identical(value, dto$penalty_sp_labels)
    }, logical(1L)))
  if (!sp_name_order_exact) {
    stop("Phase 3 SP name order mismatch", call. = FALSE)
  }

  planned_route <- batch$planned_route
  if (!fastkpc_full_cuda_fixed_sp_is_bare_character_vector(planned_route) ||
      length(planned_route) != target_count ||
      any(!planned_route %in%
            fastkpc_full_cuda_fixed_sp_contract()$route_levels)) {
    stop("Phase 3 route metadata is malformed", call. = FALSE)
  }
  condition <- batch$condition
  coefficient_rank <- batch$target_rows$coefficient_rank
  route_inputs_clean <-
    typeof(condition) %in% c("integer", "double") &&
    !is.object(condition) && is.null(attributes(condition)) &&
    length(condition) == target_count &&
    !any(is.finite(condition) & condition < 0) &&
    fastkpc_full_cuda_fixed_sp_is_bare_integer_vector(coefficient_rank) &&
    length(coefficient_rank) == target_count &&
    all(coefficient_rank >= 0L) &&
    fastkpc_full_cuda_fixed_sp_is_bare_scalar(dto$null_dim, "integer") &&
    dto$null_dim >= 0L
  if (!route_inputs_clean) {
    stop("Phase 3 route metadata is malformed", call. = FALSE)
  }
  expected_routes <- fastkpc_full_cuda_fixed_sp_route(
    condition = condition,
    coefficient_rank = coefficient_rank,
    null_dim = dto$null_dim,
    authenticated = rep(TRUE, target_count)
  )
  if (!identical(planned_route, expected_routes)) {
    stop("Phase 3 route metadata is malformed", call. = FALSE)
  }

  y_hashes <- batch$target_rows$y_hash
  if (!fastkpc_full_cuda_fixed_sp_is_bare_sha256_vector(y_hashes) ||
      length(y_hashes) != target_count) {
    stop("Phase 3 Y hash metadata is malformed", call. = FALSE)
  }
  expected_y_hashes <- vapply(seq_len(target_count), function(index) {
    fastkpc_full_cuda_census_metadata_hash(Y[, index])
  }, character(1L))
  if (!identical(expected_y_hashes, y_hashes)) {
    stop("Phase 3 Y hash mismatch", call. = FALSE)
  }

  selected_sp_hashes <- batch$target_rows$selected_sp_hash
  if (!fastkpc_full_cuda_fixed_sp_is_bare_sha256_vector(
    selected_sp_hashes
  ) || length(selected_sp_hashes) != target_count) {
    stop("Phase 3 selected-SP hash metadata is malformed", call. = FALSE)
  }
  expected_sp_hashes <- vapply(seq_len(target_count), function(index) {
    fastkpc_full_cuda_census_metadata_hash(stats::setNames(
      SP[, index], dto$penalty_sp_labels
    ))
  }, character(1L))
  if (!identical(expected_sp_hashes, selected_sp_hashes)) {
    stop("Phase 3 selected-SP hash mismatch", call. = FALSE)
  }

  target_state_fingerprints <- batch$target_rows$target_state_fingerprint
  authenticated_target_rows <-
    fastkpc_full_cuda_fixed_sp_is_bare_sha256_vector(
      target_state_fingerprints
    ) && length(target_state_fingerprints) == target_count &&
    all(vapply(seq_len(target_count), function(index) {
      identical(
        target_state_fingerprints[[index]],
        fastkpc_full_cuda_target_state_fingerprint(
          batch$target_rows[index, , drop = FALSE]
        )
      )
    }, logical(1L)))
  if (!authenticated_target_rows) {
    stop("Phase 3 authenticated target state mismatch", call. = FALSE)
  }

  dimnames(Y) <- NULL
  dimnames(SP) <- NULL

  list(
    Y = Y,
    SP = SP,
    planned_route = planned_route,
    target_keys = target_keys,
    target_count = as.integer(target_count)
  )
}

fastkpc_full_cuda_fixed_sp_phase3a_runtime_record <- function(stage, info) {
  data.frame(
    stage = stage,
    device_id = as.integer(info$device_id),
    gpu_name = as.character(info$gpu_name),
    runtime_context_create_count =
      as.integer(info$runtime_context_create_count),
    cuda_device_allocation_count =
      as.integer(info$cuda_device_allocation_count),
    cuda_host_allocation_count =
      as.integer(info$cuda_host_allocation_count),
    stream_create_count = as.integer(info$stream_create_count),
    event_create_count = as.integer(info$event_create_count),
    cublas_handle_create_count =
      as.integer(info$cublas_handle_create_count),
    cusolver_handle_create_count =
      as.integer(info$cusolver_handle_create_count),
    workspace_grow_count = as.integer(info$workspace_grow_count),
    cuda_device_synchronize_count =
      as.integer(info$cuda_device_synchronize_count),
    compute_capability_major =
      as.integer(info$compute_capability_major),
    compute_capability_minor =
      as.integer(info$compute_capability_minor),
    sm_count = as.integer(info$sm_count),
    cuda_toolkit_version = as.integer(info$cuda_toolkit_version),
    cuda_driver_version = as.integer(info$cuda_driver_version),
    cusolver_deterministic_mode =
      as.character(info$cusolver_deterministic_mode),
    cublas_math_mode = as.character(info$cublas_math_mode),
    cublas_atomics_mode = as.character(info$cublas_atomics_mode),
    cublas_user_workspace_installed =
      isTRUE(info$cublas_user_workspace_installed),
    cublas_workspace_alignment =
      as.numeric(info$cublas_workspace_alignment),
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_fixed_sp_phase3a_counter_delta <- function(
    before, after, field) {
  delta <- as.integer(after[[field]] - before[[field]])
  if (length(delta) != 1L || is.na(delta) || delta < 0L) {
    stop(paste("fixed-sp CUDA resource counter regressed:", field),
         call. = FALSE)
  }
  delta
}

fastkpc_full_cuda_fixed_sp_phase3a_relative_l2_diff <- function(
    actual, reference) {
  difference <- as.numeric(actual) - as.numeric(reference)
  sqrt(sum(difference^2)) /
    max(sqrt(sum(as.numeric(reference)^2)), 1e-300)
}

fastkpc_full_cuda_fixed_sp_phase3a_validate_parity <- function(
    target_records) {
  error_fields <- c(
    "residual_max_abs_diff", "residual_relative_l2_diff",
    "fitted_max_abs_diff", "fitted_relative_l2_diff"
  )
  required_fields <- c(
    "residual_key_sha256", "planned_route",
    "authenticated_planned_route", "solver_status", error_fields
  )
  parity_ok <- is.data.frame(target_records) &&
    all(required_fields %in% names(target_records)) &&
    nrow(target_records) == 270L
  if (isTRUE(parity_ok)) {
    keys <- target_records$residual_key_sha256
    parity_ok <- is.character(keys) && !anyNA(keys) &&
      !anyDuplicated(keys) && all(grepl("^[0-9a-f]{64}$", keys)) &&
      is.character(target_records$planned_route) &&
      is.character(target_records$authenticated_planned_route) &&
      !anyNA(target_records$planned_route) &&
      !anyNA(target_records$authenticated_planned_route) &&
      !anyNA(target_records$solver_status) &&
      identical(target_records$planned_route,
                target_records$authenticated_planned_route)
  }
  if (isTRUE(parity_ok)) {
    safe <- target_records$authenticated_planned_route ==
      "CHOLESKY_BATCHED"
    stable <- !safe
    parity_ok <- sum(safe) == 172L && sum(stable) == 98L &&
      all(target_records$authenticated_planned_route[stable] %in%
          c("AUGMENTED_QR", "AUGMENTED_SVD")) &&
      all(target_records$solver_status[safe] == "OK_CHOLESKY_SINGLE") &&
      all(target_records$solver_status[stable] ==
          "ERR_STABLE_PATH_NOT_IMPLEMENTED") &&
      all(vapply(error_fields, function(field) {
        values <- target_records[[field]]
        is.numeric(values) && all(is.finite(values[safe])) &&
          all(values[safe] >= 0 & values[safe] < 1e-7) &&
          all(is.na(values[stable]) & !is.nan(values[stable]))
      }, logical(1L)))
  }
  if (!isTRUE(parity_ok)) {
    stop("Phase 3A iteration status/numerical parity failed",
         call. = FALSE)
  }
  list(safe = safe, stable = stable)
}

fastkpc_full_cuda_fixed_sp_phase3a_benchmark_identity <- function(
    descriptors, canonical_target_keys) {
  canonical_target_keys <- as.character(canonical_target_keys)
  descriptor_keys <- if (is.list(descriptors)) {
    vapply(descriptors, function(descriptor) {
      if (!is.list(descriptor) || !is.list(descriptor$native) ||
          !is.character(descriptor$native$target_keys) ||
          length(descriptor$native$target_keys) != 1L ||
          is.na(descriptor$native$target_keys) ||
          !nzchar(descriptor$native$target_keys)) {
        return(NA_character_)
      }
      descriptor$native$target_keys[[1L]]
    }, character(1L), USE.NAMES = FALSE)
  } else {
    character()
  }
  identity_ok <- length(canonical_target_keys) == 172L &&
    !anyNA(canonical_target_keys) &&
    !anyDuplicated(canonical_target_keys) &&
    all(grepl("^[0-9a-f]{64}$", canonical_target_keys)) &&
    identical(descriptor_keys, canonical_target_keys)
  if (!isTRUE(identity_ok)) {
    stop("Phase 3A benchmark target identity mismatch", call. = FALSE)
  }
  list(
    benchmark_target_count = as.integer(length(canonical_target_keys)),
    ordered_target_keys = canonical_target_keys,
    target_key_corpus_hash =
      fastkpc_full_cuda_census_key_set_hash(canonical_target_keys)
  )
}

fastkpc_full_cuda_fixed_sp_phase3a_prototype_payload_fields <- function() {
  c("target_key", "X", "y", "Z", "XtX_null", "penalty_null", "Xty_null")
}

fastkpc_full_cuda_fixed_sp_phase3a_prototype_payload_identity <- function(
    prototype) {
  fields <- fastkpc_full_cuda_fixed_sp_phase3a_prototype_payload_fields()
  payload_ok <- is.list(prototype) && identical(names(prototype), fields) &&
    is.character(prototype$target_key) && length(prototype$target_key) == 1L &&
    !is.na(prototype$target_key) &&
    grepl("^[0-9a-f]{64}$", prototype$target_key) &&
    all(vapply(prototype[fields[-1L]], function(value) {
      is.numeric(value) && length(value) > 0L && all(is.finite(value))
    }, logical(1L)))
  if (!isTRUE(payload_ok)) {
    stop("Phase 3A prototype payload is malformed", call. = FALSE)
  }
  field_hashes <- vapply(prototype[fields[-1L]],
                         fastkpc_full_cuda_census_metadata_hash,
                         character(1L))
  names(field_hashes) <- paste0(fields[-1L], "_hash")
  c(
    list(target_key = prototype$target_key),
    as.list(field_hashes),
    list(payload_hash = fastkpc_full_cuda_census_named_metadata_hash(prototype))
  )
}

fastkpc_full_cuda_fixed_sp_phase3a_prototype_expected_identity <- function(
    dto, target_key, target_row, y, sp, canonical_nullspace_rhs,
    planned_route) {
  if (!is.character(target_key) || length(target_key) != 1L ||
      is.na(target_key) || !grepl("^[0-9a-f]{64}$", target_key) ||
      !is.data.frame(target_row) || nrow(target_row) != 1L ||
      !"residual_key_sha256" %in% names(target_row) ||
      !identical(as.character(target_row$residual_key_sha256[[1L]]), target_key) ||
      !all(c("y_hash", "selected_sp_hash") %in% names(target_row)) ||
      !is.character(target_row$y_hash) || length(target_row$y_hash) != 1L ||
      !grepl("^[0-9a-f]{64}$", target_row$y_hash) ||
      !is.character(target_row$selected_sp_hash) ||
      length(target_row$selected_sp_hash) != 1L ||
      !grepl("^[0-9a-f]{64}$", target_row$selected_sp_hash) ||
      !is.character(planned_route) || length(planned_route) != 1L ||
      !planned_route %in% fastkpc_full_cuda_fixed_sp_contract()$route_levels ||
      !is.numeric(y) || !is.numeric(sp) ||
      !is.numeric(canonical_nullspace_rhs) || any(!is.finite(y)) ||
      any(!is.finite(sp)) || any(sp < 0) ||
      any(!is.finite(canonical_nullspace_rhs)) ||
      !is.character(dto$penalty_sp_labels) ||
      length(dto$penalty_sp_labels) != length(sp) ||
      !is.character(dto$prepared_s_key_sha256) ||
      length(dto$prepared_s_key_sha256) != 1L ||
      !grepl("^[0-9a-f]{64}$", dto$prepared_s_key_sha256)) {
    stop("Phase 3A prototype expected source is malformed", call. = FALSE)
  }
  candidate_y_hash <- fastkpc_full_cuda_census_metadata_hash(as.numeric(y))
  candidate_sp_hash <- fastkpc_full_cuda_census_metadata_hash(
    stats::setNames(as.numeric(sp), dto$penalty_sp_labels)
  )
  if (!identical(candidate_y_hash, target_row$y_hash[[1L]]) ||
      !identical(candidate_sp_hash, target_row$selected_sp_hash[[1L]])) {
    stop("Phase 3A prototype expected source is malformed", call. = FALSE)
  }
  target_state_fingerprint <- if (
    "target_state_fingerprint" %in% names(target_row)
  ) {
    value <- target_row$target_state_fingerprint[[1L]]
    if (!is.character(value) || length(value) != 1L ||
        !grepl("^[0-9a-f]{64}$", value)) {
      stop("Phase 3A prototype expected source is malformed", call. = FALSE)
    }
    value
  } else {
    NULL
  }
  # Keep this derivation independent from the timed prototype payload.
  Z <- if (identical(dto$constraint_mode, "identity")) {
    diag(dto$coefficient_dim)
  } else {
    dto$constraint_nullspace
  }
  X_null <- if (identical(dto$constraint_mode, "identity")) {
    dto$X
  } else {
    dto$X %*% Z
  }
  XtX_null <- if (identical(dto$constraint_mode, "identity")) {
    dto$gram_matrix
  } else {
    dto$nullspace_gram_matrix
  }
  penalty_null <- matrix(0, dto$null_dim, dto$null_dim)
  for (index in seq_len(dto$penalty_count)) {
    full_penalty <- matrix(0, dto$coefficient_dim, dto$coefficient_dim)
    block <- dto$penalty_blocks[[index]]
    block_indices <- dto$penalty_offsets_zero_based[[index]] +
      seq_len(nrow(block))
    full_penalty[block_indices, block_indices] <- block
    projected <- if (identical(dto$constraint_mode, "identity")) {
      full_penalty
    } else {
      crossprod(Z, full_penalty %*% Z)
    }
    penalty_null <- penalty_null + as.numeric(sp[[index]]) * projected
  }
  expected_payload <- list(
    target_key = target_key,
    X = dto$X,
    y = as.numeric(y),
    Z = Z,
    XtX_null = XtX_null,
    penalty_null = penalty_null,
    Xty_null = as.numeric(canonical_nullspace_rhs)
  )
  expected_payload_identity <-
    fastkpc_full_cuda_fixed_sp_phase3a_prototype_payload_identity(
      expected_payload
    )
  list(
    payload = expected_payload_identity,
    source = list(
      target_key = target_key,
      target_row_hash =
        fastkpc_full_cuda_census_named_metadata_hash(as.list(target_row)),
      prepared_setup_hash = fastkpc_full_cuda_census_named_metadata_hash(dto),
      prepared_s_key_sha256 = dto$prepared_s_key_sha256,
      authenticated_y_hash = target_row$y_hash[[1L]],
      authenticated_selected_sp_hash = target_row$selected_sp_hash[[1L]],
      penalty_sp_labels = dto$penalty_sp_labels,
      target_state_fingerprint = target_state_fingerprint,
      canonical_nullspace_rhs_hash =
        fastkpc_full_cuda_census_metadata_hash(
          as.numeric(canonical_nullspace_rhs)
        ),
      planned_route = planned_route
    )
  )
}

fastkpc_full_cuda_fixed_sp_phase3a_prototype_benchmark_identity <- function(
    descriptors, canonical_target_keys) {
  key_identity <- fastkpc_full_cuda_fixed_sp_phase3a_benchmark_identity(
    descriptors, canonical_target_keys
  )
  actual <- lapply(descriptors, function(descriptor) {
    if (!is.list(descriptor) || is.null(descriptor$prototype)) {
      stop("Phase 3A prototype payload identity mismatch", call. = FALSE)
    }
    fastkpc_full_cuda_fixed_sp_phase3a_prototype_payload_identity(
      descriptor$prototype
    )
  })
  expected <- lapply(descriptors, function(descriptor) {
    if (!is.list(descriptor) || !is.list(descriptor$prototype_expected)) {
      stop("Phase 3A prototype payload identity mismatch", call. = FALSE)
    }
    descriptor$prototype_expected
  })
  expected_payload <- lapply(expected, `[[`, "payload")
  expected_source <- lapply(expected, `[[`, "source")
  actual_keys <- vapply(actual, `[[`, character(1L), "target_key")
  actual_hashes <- vapply(actual, `[[`, character(1L), "payload_hash")
  source_matches_native <- vapply(seq_along(descriptors), function(index) {
    native <- descriptors[[index]]$native
    source <- expected_source[[index]]
    is.list(native) && is.list(source) &&
      identical(source$target_key, native$target_keys) &&
      identical(
        source$authenticated_y_hash,
        fastkpc_full_cuda_census_metadata_hash(as.numeric(native$Y))
      ) &&
      identical(
        source$authenticated_selected_sp_hash,
        fastkpc_full_cuda_census_metadata_hash(
          stats::setNames(as.numeric(native$SP), source$penalty_sp_labels)
        )
      ) &&
      identical(source$planned_route, native$planned_route)
  }, logical(1L))
  if (!identical(actual, expected_payload) || !all(source_matches_native) ||
      !identical(actual_keys, key_identity$ordered_target_keys)) {
    stop("Phase 3A prototype payload identity mismatch", call. = FALSE)
  }
  c(
    key_identity,
    list(
      ordered_payload_hashes = actual_hashes,
      payload_corpus_hash = fastkpc_full_cuda_census_key_set_hash(actual_hashes)
    )
  )
}

fastkpc_full_cuda_fixed_sp_phase3a_benchmark_path_identities <- function(
    persistent_descriptors, prototype_descriptors, canonical_target_keys) {
  identities <- list(
    persistent = fastkpc_full_cuda_fixed_sp_phase3a_benchmark_identity(
      persistent_descriptors, canonical_target_keys
    ),
    prototype = fastkpc_full_cuda_fixed_sp_phase3a_prototype_benchmark_identity(
      prototype_descriptors, canonical_target_keys
    )
  )
  if (!identical(
    identities$persistent[c(
      "benchmark_target_count", "ordered_target_keys", "target_key_corpus_hash"
    )],
    identities$prototype[c(
      "benchmark_target_count", "ordered_target_keys", "target_key_corpus_hash"
    )]
  )) {
    stop("Phase 3A benchmark path identity mismatch", call. = FALSE)
  }
  identities
}

fastkpc_full_cuda_fixed_sp_phase3a_prototype_handle <- function(
    dto, y, sp) {
  Z <- if (identical(dto$constraint_mode, "identity")) {
    diag(dto$coefficient_dim)
  } else {
    dto$constraint_nullspace
  }
  X_null <- if (identical(dto$constraint_mode, "identity")) {
    dto$X
  } else {
    dto$X %*% Z
  }
  XtX_null <- if (identical(dto$constraint_mode, "identity")) {
    dto$gram_matrix
  } else {
    dto$nullspace_gram_matrix
  }
  penalty_null <- matrix(0, dto$null_dim, dto$null_dim)
  for (index in seq_len(dto$penalty_count)) {
    full_penalty <- matrix(
      0, dto$coefficient_dim, dto$coefficient_dim
    )
    block <- dto$penalty_blocks[[index]]
    block_indices <- dto$penalty_offsets_zero_based[[index]] +
      seq_len(nrow(block))
    full_penalty[block_indices, block_indices] <- block
    projected <- if (identical(dto$constraint_mode, "identity")) {
      full_penalty
    } else {
      crossprod(Z, full_penalty %*% Z)
    }
    penalty_null <- penalty_null + as.numeric(sp[[index]]) * projected
  }
  list(
    X = dto$X,
    y = as.numeric(y),
    Z = Z,
    XtX_null = XtX_null,
    penalty_null = penalty_null,
    Xty_null = as.numeric(crossprod(X_null, y))
  )
}

fastkpc_run_full_cuda_fixed_sp_phase3a_iteration <- function(
    phase2_dir, census_dir, prepared_dir, data_path, device_id = 0L) {
  required_functions <- c(
    "fixed_sp_cuda_runtime_create", "fixed_sp_cuda_runtime_reserve",
    "fixed_sp_cuda_runtime_info", "fixed_sp_cuda_runtime_free",
    "fixed_sp_cuda_prepared_create", "fixed_sp_cuda_prepared_info",
    "fixed_sp_cuda_prepared_free", "fixed_sp_cuda_solve_batch",
    "fixed_sp_cuda_residual_info", "fixed_sp_cuda_materialize_shadow",
    "fixed_sp_cuda_residual_release", "fixed_sp_cuda_residual_free",
    "mgcv_extract_gpu_solve_handle_fixed_sp_cuda"
  )
  missing <- required_functions[!vapply(required_functions, exists, logical(1L),
                                        mode = "function", inherits = TRUE)]
  if (length(missing) != 0L) {
    stop(paste("Phase 3A CUDA API is unavailable:", paste(missing,
                                                           collapse = ", ")),
         call. = FALSE)
  }
  path_values <- c(phase2_dir, census_dir, prepared_dir, data_path)
  if (length(path_values) != 4L || anyNA(path_values) ||
      any(!nzchar(path_values))) {
    stop("Phase 3A iteration paths must be non-empty", call. = FALSE)
  }
  if (typeof(device_id) != "integer" || length(device_id) != 1L ||
      is.na(device_id) || device_id < 0L) {
    stop("Phase 3A device_id must be one non-negative integer",
         call. = FALSE)
  }

  catalog <- fastkpc_full_cuda_open_fixed_sp_catalog(
    phase2_dir, census_dir, prepared_dir, data_path
  )
  iteration <- fastkpc_full_cuda_fixed_sp_scope(catalog, "iteration")
  batches <- fastkpc_full_cuda_fixed_sp_batches(catalog, iteration)
  setup_keys <- names(batches)
  if (!identical(setup_keys, sort(setup_keys, method = "radix"))) {
    stop("Phase 3A iteration PreparedSKey order is not canonical",
         call. = FALSE)
  }
  catalog_records <- data.frame(
    scope = "iteration",
    authenticated = TRUE,
    setup_count = as.integer(length(batches)),
    target_count = as.integer(sum(vapply(
      batches, function(batch) nrow(batch$target_rows), integer(1L)
    ))),
    stringsAsFactors = FALSE
  )

  dtos <- lapply(batches, function(batch) {
    fastkpc_full_cuda_fixed_sp_native_dto(batch$setup)
  })
  max_n <- max(vapply(dtos, `[[`, integer(1L), "n"))
  max_q <- max(vapply(dtos, `[[`, integer(1L), "null_dim"))
  max_penalties <- max(vapply(dtos, `[[`, integer(1L), "penalty_count"))
  max_augmented_rows <- max(vapply(dtos, function(dto) {
    as.integer(dto$n + sum(dto$penalty_ranks))
  }, integer(1L)))

  runtime <- fixed_sp_cuda_runtime_create(device_id)
  runtime_freed <- FALSE
  handles <- setNames(vector("list", length(setup_keys)), setup_keys)
  on.exit({
    for (handle in rev(handles)) {
      if (!is.null(handle)) {
        try(fixed_sp_cuda_prepared_free(handle), silent = TRUE)
      }
    }
    if (!runtime_freed) {
      try(fixed_sp_cuda_runtime_free(runtime), silent = TRUE)
    }
  }, add = TRUE)

  runtime_rows <- list(fastkpc_full_cuda_fixed_sp_phase3a_runtime_record(
    "runtime-created", fixed_sp_cuda_runtime_info(runtime)
  ))
  fixed_sp_cuda_runtime_reserve(
    runtime, max_n, max_q, 1L, max_penalties, max_augmented_rows
  )
  runtime_rows[[length(runtime_rows) + 1L]] <-
    fastkpc_full_cuda_fixed_sp_phase3a_runtime_record(
      "workspace-reserved", fixed_sp_cuda_runtime_info(runtime)
    )

  max_abs_diff <- function(actual, expected) {
    max(abs(as.numeric(actual) - as.numeric(expected)))
  }
  run_checked_target <- function(handle, target, batch, target_index) {
    token <- fixed_sp_cuda_solve_batch(
      handle, target$Y, target$SP, target$planned_route,
      target$target_keys, outputs = c("fitted", "residuals")
    )
    released <- FALSE
    freed <- FALSE
    on.exit({
      if (!released) {
        try(fixed_sp_cuda_residual_release(token), silent = TRUE)
      }
      if (!freed) {
        try(fixed_sp_cuda_residual_free(token), silent = TRUE)
      }
    }, add = TRUE)

    info <- fixed_sp_cuda_residual_info(token)
    residual_max_abs <- NA_real_
    residual_relative_l2 <- NA_real_
    fitted_max_abs <- NA_real_
    fitted_relative_l2 <- NA_real_
    if (identical(target$planned_route, "CHOLESKY_BATCHED")) {
      shadow <- fixed_sp_cuda_materialize_shadow(
        token, outputs = c("fitted", "residuals")
      )
      info <- fixed_sp_cuda_residual_info(token)
      oracle <- fastkpc_mgcv_magic_fixed_sp_from_prepared(
        prepared_setup = batch$setup,
        target_state = list(
          row = batch$target_rows[target_index, , drop = FALSE],
          y = as.numeric(target$Y[, 1L])
        )
      )
      residual_max_abs <- max_abs_diff(shadow$residuals, oracle$residuals)
      residual_relative_l2 <-
        fastkpc_full_cuda_fixed_sp_phase3a_relative_l2_diff(
          shadow$residuals, oracle$residuals
        )
      fitted_max_abs <- max_abs_diff(shadow$fitted, oracle$fitted)
      fitted_relative_l2 <-
        fastkpc_full_cuda_fixed_sp_phase3a_relative_l2_diff(
          shadow$fitted, oracle$fitted
        )
    }

    fixed_sp_cuda_residual_release(token)
    released <- TRUE
    prepared_after_release <- fixed_sp_cuda_prepared_info(handle)
    fixed_sp_cuda_residual_free(token)
    freed <- TRUE
    approximate_fallback_count <- as.integer(any(grepl(
      "APPROX", c(info$executed_route, info$reroute_reason,
                   info$solver_status), ignore.case = TRUE
    ), na.rm = TRUE))

    data.frame(
      prepared_s_key_sha256 = batch$prepared_s_key_sha256,
      residual_key_sha256 = as.character(target$target_keys),
      target = as.integer(batch$target_rows$target[[target_index]]),
      target_count = as.integer(info$target_count),
      planned_route = as.character(info$planned_route),
      authenticated_planned_route = as.character(target$planned_route),
      executed_route = as.character(info$executed_route),
      reroute_reason = as.character(info$reroute_reason),
      solver_status = as.character(info$solver_status),
      native_batch_call = isTRUE(info$native_batch_call),
      rhs_authority = as.character(info$rhs_authority),
      full_cuda_data_plane = isTRUE(info$full_cuda_data_plane),
      invalid_output_init_count =
        as.integer(info$invalid_output_init_count),
      shadow_materialize_call_count =
        as.integer(info$shadow_materialize_call_count),
      cpu_fallback_count = as.integer(info$cpu_fallback_count),
      unknown_fallback_count = as.integer(info$unknown_fallback_count),
      approximate_fallback_count = approximate_fallback_count,
      resource_snapshot_captured =
        isTRUE(info$resource_snapshot_captured),
      resource_instrumentation_version =
        as.integer(info$resource_instrumentation_version),
      resource_allocation_count_before_solve =
        as.integer(info$resource_allocation_count_before_solve),
      resource_allocation_count_after_solve =
        as.integer(info$resource_allocation_count_after_solve),
      resource_handle_create_count_before_solve =
        as.integer(info$resource_handle_create_count_before_solve),
      resource_handle_create_count_after_solve =
        as.integer(info$resource_handle_create_count_after_solve),
      cuda_device_allocation_count_during_solve =
        as.integer(info$cuda_device_allocation_count_during_solve),
      cuda_host_allocation_count_during_solve =
        as.integer(info$cuda_host_allocation_count_during_solve),
      stream_create_count_during_solve =
        as.integer(info$stream_create_count_during_solve),
      event_create_count_during_solve =
        as.integer(info$event_create_count_during_solve),
      cublas_handle_create_count_during_solve =
        as.integer(info$cublas_handle_create_count_during_solve),
      cusolver_handle_create_count_during_solve =
        as.integer(info$cusolver_handle_create_count_during_solve),
      per_target_allocation_count_after_warmup =
        as.integer(info$per_target_allocation_count_after_warmup),
      per_target_handle_create_count =
        as.integer(info$per_target_handle_create_count),
      residual_max_abs_diff = residual_max_abs,
      residual_relative_l2_diff = residual_relative_l2,
      fitted_max_abs_diff = fitted_max_abs,
      fitted_relative_l2_diff = fitted_relative_l2,
      lease_released_before_reuse = released,
      output_slot_leased_after_release =
        isTRUE(prepared_after_release$output_slot_leased),
      stringsAsFactors = FALSE
    )
  }

  prepared_rows <- vector("list", length(setup_keys))
  target_rows <- vector("list", catalog_records$target_count[[1L]])
  native_batches <- setNames(vector("list", length(setup_keys)), setup_keys)
  target_row_index <- 0L
  for (setup_index in seq_along(setup_keys)) {
    setup_key <- setup_keys[[setup_index]]
    batch <- batches[[setup_key]]
    dto <- dtos[[setup_key]]
    native_batch <- fastkpc_full_cuda_fixed_sp_native_batch(batch, dto)
    native_batches[[setup_key]] <- native_batch
    resources_before_setup <- fixed_sp_cuda_runtime_info(runtime)
    handle <- fixed_sp_cuda_prepared_create(runtime, dto)
    handles[[setup_key]] <- handle
    prepared_info <- fixed_sp_cuda_prepared_info(handle)
    resources_after_setup <- fixed_sp_cuda_runtime_info(runtime)
    prepared_rows[[setup_index]] <- data.frame(
      prepared_s_key_sha256 = setup_key,
      setup_h2d_upload_count =
        as.integer(prepared_info$setup_h2d_upload_count),
      setup_h2d_bytes = as.numeric(prepared_info$setup_h2d_bytes),
      cuda_device_allocation_count_during_setup =
        fastkpc_full_cuda_fixed_sp_phase3a_counter_delta(
          resources_before_setup, resources_after_setup,
          "cuda_device_allocation_count"
        ),
      cuda_host_allocation_count_during_setup =
        fastkpc_full_cuda_fixed_sp_phase3a_counter_delta(
          resources_before_setup, resources_after_setup,
          "cuda_host_allocation_count"
        ),
      event_create_count_during_setup =
        fastkpc_full_cuda_fixed_sp_phase3a_counter_delta(
          resources_before_setup, resources_after_setup,
          "event_create_count"
        ),
      output_slot_leased_at_end = NA,
      stringsAsFactors = FALSE
    )

    for (target_index in seq_len(native_batch$target_count)) {
      target <- list(
        Y = native_batch$Y[, target_index, drop = FALSE],
        SP = native_batch$SP[, target_index, drop = FALSE],
        planned_route = native_batch$planned_route[[target_index]],
        target_keys = native_batch$target_keys[[target_index]],
        target_count = 1L
      )
      target_row_index <- target_row_index + 1L
      target_rows[[target_row_index]] <- run_checked_target(
        handle, target, batch, target_index
      )
    }
    prepared_rows[[setup_index]]$output_slot_leased_at_end <-
      isTRUE(fixed_sp_cuda_prepared_info(handle)$output_slot_leased)
  }
  prepared_records <- do.call(rbind, prepared_rows)
  rownames(prepared_records) <- NULL
  target_records <- do.call(rbind, target_rows)
  rownames(target_records) <- NULL
  parity <- fastkpc_full_cuda_fixed_sp_phase3a_validate_parity(target_records)
  safe <- parity$safe
  stable <- parity$stable

  safe_record_indices <- which(safe)
  safe_descriptor_inputs <- vector("list", length(safe_record_indices))
  for (descriptor_index in seq_along(safe_record_indices)) {
    target_record <- target_records[
      safe_record_indices[[descriptor_index]], , drop = FALSE
    ]
    setup_key <- target_record$prepared_s_key_sha256[[1L]]
    batch <- batches[[setup_key]]
    dto <- dtos[[setup_key]]
    native_batch <- native_batches[[setup_key]]
    target_index <- match(
      target_record$residual_key_sha256[[1L]], native_batch$target_keys
    )
    source_is_authenticated <- !is.null(handles[[setup_key]]) &&
      !is.null(batch) && !is.null(dto) && !is.null(native_batch) &&
      length(target_index) == 1L && !is.na(target_index) &&
      identical(
        native_batch$target_keys[[target_index]],
        target_record$residual_key_sha256[[1L]]
      ) && identical(
        batch$target_rows$residual_key_sha256[[target_index]],
        target_record$residual_key_sha256[[1L]]
      )
    if (!isTRUE(source_is_authenticated)) {
      stop("Phase 3A authenticated benchmark descriptor source is malformed",
           call. = FALSE)
    }
    native <- list(
      Y = native_batch$Y[, target_index, drop = FALSE],
      SP = native_batch$SP[, target_index, drop = FALSE],
      planned_route = native_batch$planned_route[[target_index]],
      target_keys = native_batch$target_keys[[target_index]],
      target_count = 1L
    )
    safe_descriptor_inputs[[descriptor_index]] <- list(
      handle = handles[[setup_key]],
      native = native,
      dto = dto,
      canonical_target_row = batch$target_rows[target_index, , drop = FALSE],
      canonical_nullspace_rhs = as.numeric(
        batch$oracle_nullspace_rhs[, target_index]
      )
    )
  }
  safe_descriptors <- list()
  for (descriptor in safe_descriptor_inputs) {
    prototype_expected <-
      fastkpc_full_cuda_fixed_sp_phase3a_prototype_expected_identity(
        dto = descriptor$dto,
        target_key = descriptor$native$target_keys,
        target_row = descriptor$canonical_target_row,
        y = descriptor$native$Y[, 1L],
        sp = descriptor$native$SP[, 1L],
        canonical_nullspace_rhs = descriptor$canonical_nullspace_rhs,
        planned_route = descriptor$native$planned_route
      )
    prototype <- fastkpc_full_cuda_fixed_sp_phase3a_prototype_handle(
      descriptor$dto, descriptor$native$Y[, 1L], descriptor$native$SP[, 1L]
    )
    prototype <- c(list(target_key = descriptor$native$target_keys), prototype)
    safe_descriptors[[length(safe_descriptors) + 1L]] <- list(
      handle = descriptor$handle,
      native = descriptor$native,
      prototype = prototype,
      prototype_expected = prototype_expected
    )
  }
  persistent_descriptors <- lapply(safe_descriptors, function(descriptor) {
    descriptor[c("handle", "native")]
  })
  prototype_descriptors <- lapply(safe_descriptors, function(descriptor) {
    descriptor[c("native", "prototype", "prototype_expected")]
  })
  benchmark_path_identities <-
    fastkpc_full_cuda_fixed_sp_phase3a_benchmark_path_identities(
      persistent_descriptors, prototype_descriptors,
      target_records$residual_key_sha256[safe]
    )
  benchmark_identity <- benchmark_path_identities$persistent

  timed_persistent_solve <- function(descriptor) {
    .Call(
      "C_fixed_sp_cuda_solve_batch", descriptor$handle,
      descriptor$native$Y, descriptor$native$SP,
      as.character(descriptor$native$planned_route),
      as.character(descriptor$native$target_keys), "residuals",
      PACKAGE = "fastkpc_cuda"
    )
  }
  timed_persistent_free <- function(token) {
    invisible(.Call(
      "C_fixed_sp_cuda_residual_free", token, PACKAGE = "fastkpc_cuda"
    ))
  }
  timed_prototype_solve <- function(descriptor) {
    handle <- descriptor$prototype
    .Call(
      "C_mgcv_extract_gpu_solve_handle_fixed_sp",
      handle$X, handle$y, handle$Z, handle$XtX_null,
      handle$penalty_null, handle$Xty_null, PACKAGE = "fastkpc_cuda"
    )
  }
  run_persistent_corpus <- function(profile = FALSE) {
    resources_before <- fixed_sp_cuda_runtime_info(runtime)
    solve_ms <- 0
    release_ms <- 0
    free_ms <- 0
    started <- proc.time()[["elapsed"]]
    for (descriptor in persistent_descriptors) {
      solve_started <- if (profile) proc.time()[["elapsed"]] else 0
      token <- timed_persistent_solve(descriptor)
      if (profile) {
        solve_ms <- solve_ms +
          1000 * (proc.time()[["elapsed"]] - solve_started)
      }
      freed <- FALSE
      tryCatch({
        free_started <- if (profile) proc.time()[["elapsed"]] else 0
        timed_persistent_free(token)
        freed <- TRUE
        if (profile) {
          free_ms <- free_ms +
            1000 * (proc.time()[["elapsed"]] - free_started)
        }
      }, finally = {
        if (!freed) {
          try(timed_persistent_free(token), silent = TRUE)
        }
      })
    }
    elapsed_ms <- 1000 * (proc.time()[["elapsed"]] - started)
    resources_after <- fixed_sp_cuda_runtime_info(runtime)
    allocation_count <-
      fastkpc_full_cuda_fixed_sp_phase3a_counter_delta(
        resources_before, resources_after, "cuda_device_allocation_count"
      ) + fastkpc_full_cuda_fixed_sp_phase3a_counter_delta(
        resources_before, resources_after, "cuda_host_allocation_count"
      )
    handle_create_count <- sum(vapply(
      c("stream_create_count", "event_create_count",
        "cublas_handle_create_count", "cusolver_handle_create_count"),
      function(field) {
        fastkpc_full_cuda_fixed_sp_phase3a_counter_delta(
          resources_before, resources_after, field
        )
      }, integer(1L)
    ))
    list(
      elapsed_ms = as.numeric(elapsed_ms),
      allocation_count = as.integer(allocation_count),
      handle_create_count = as.integer(handle_create_count),
      solve_ms = as.numeric(solve_ms),
      release_ms = as.numeric(release_ms),
      free_ms = as.numeric(free_ms)
    )
  }
  run_prototype_corpus <- function() {
    started <- proc.time()[["elapsed"]]
    for (descriptor in prototype_descriptors) {
      timed_prototype_solve(descriptor)
    }
    as.numeric(1000 * (proc.time()[["elapsed"]] - started))
  }

  persistent_resource_rows <- list()
  persistent_warmup <- run_persistent_corpus(profile = TRUE)
  persistent_resource_rows[[1L]] <- data.frame(
    repetition = 0L,
    allocation_count = persistent_warmup$allocation_count,
    handle_create_count = persistent_warmup$handle_create_count
  )
  run_prototype_corpus()
  runtime_rows[[length(runtime_rows) + 1L]] <-
    fastkpc_full_cuda_fixed_sp_phase3a_runtime_record(
      "post-warmup", fixed_sp_cuda_runtime_info(runtime)
    )

  persistent_raw_ms <- numeric(3L)
  prototype_raw_ms <- numeric(3L)
  for (repetition in seq_len(3L)) {
    invisible(gc(FALSE))
    persistent_repetition <- run_persistent_corpus()
    persistent_raw_ms[[repetition]] <-
      persistent_repetition$elapsed_ms
    persistent_resource_rows[[repetition + 1L]] <- data.frame(
      repetition = as.integer(repetition),
      allocation_count = persistent_repetition$allocation_count,
      handle_create_count = persistent_repetition$handle_create_count
    )
    invisible(gc(FALSE))
    prototype_raw_ms[[repetition]] <- run_prototype_corpus()
  }
  runtime_rows[[length(runtime_rows) + 1L]] <-
    fastkpc_full_cuda_fixed_sp_phase3a_runtime_record(
      "final", fixed_sp_cuda_runtime_info(runtime)
    )
  runtime_records <- do.call(rbind, runtime_rows)
  rownames(runtime_records) <- NULL

  for (setup_index in seq_along(setup_keys)) {
    prepared_records$output_slot_leased_at_end[[setup_index]] <-
      isTRUE(fixed_sp_cuda_prepared_info(
        handles[[setup_keys[[setup_index]]]]
      )$output_slot_leased)
  }
  persistent_median_ms <- as.numeric(stats::median(persistent_raw_ms))
  prototype_median_ms <- as.numeric(stats::median(prototype_raw_ms))
  speedup <- prototype_median_ms / persistent_median_ms
  final_runtime <- runtime_records[
    runtime_records$stage == "final", , drop = FALSE
  ]
  post_warmup_runtime <- runtime_records[
    runtime_records$stage == "post-warmup", , drop = FALSE
  ]
  timing <- list(
    warmup_count = c(persistent = 1L, prototype = 1L),
    benchmark_target_count = benchmark_identity$benchmark_target_count,
    ordered_target_keys = benchmark_identity$ordered_target_keys,
    target_key_corpus_hash = benchmark_identity$target_key_corpus_hash,
    persistent_workload_identity = benchmark_path_identities$persistent,
    prototype_workload_identity = benchmark_path_identities$prototype,
    persistent_raw_ms = persistent_raw_ms,
    prototype_raw_ms = prototype_raw_ms,
    persistent_median_ms = persistent_median_ms,
    prototype_median_ms = prototype_median_ms,
    speedup = as.numeric(speedup),
    persistent_warmup_profile_ms = c(
      solve = persistent_warmup$solve_ms,
      release = persistent_warmup$release_ms,
      free = persistent_warmup$free_ms
    ),
    gpu_identity = list(
      device_id = as.integer(final_runtime$device_id[[1L]]),
      name = final_runtime$gpu_name[[1L]],
      compute_capability_major =
        as.integer(final_runtime$compute_capability_major[[1L]]),
      compute_capability_minor =
        as.integer(final_runtime$compute_capability_minor[[1L]]),
      sm_count = as.integer(final_runtime$sm_count[[1L]]),
      cuda_toolkit_version =
        as.integer(final_runtime$cuda_toolkit_version[[1L]]),
      cuda_driver_version =
        as.integer(final_runtime$cuda_driver_version[[1L]])
    ),
    persistent_resource_records = do.call(
      rbind, persistent_resource_rows
    )
  )

  summary <- list(
    catalog_open_count = as.integer(nrow(catalog_records)),
    setup_count = as.integer(nrow(prepared_records)),
    target_count = as.integer(nrow(target_records)),
    cholesky_ok_count = as.integer(sum(
      target_records$solver_status == "OK_CHOLESKY_SINGLE"
    )),
    stable_not_implemented_count = as.integer(sum(
      target_records$solver_status == "ERR_STABLE_PATH_NOT_IMPLEMENTED"
    )),
    residual_max_abs_diff_max = max(
      target_records$residual_max_abs_diff[safe]
    ),
    residual_relative_l2_diff_max = max(
      target_records$residual_relative_l2_diff[safe]
    ),
    fitted_max_abs_diff_max = max(
      target_records$fitted_max_abs_diff[safe]
    ),
    fitted_relative_l2_diff_max = max(
      target_records$fitted_relative_l2_diff[safe]
    ),
    setup_h2d_upload_count = as.integer(sum(
      prepared_records$setup_h2d_upload_count
    )),
    runtime_context_create_count =
      as.integer(final_runtime$runtime_context_create_count[[1L]]),
    deterministic_runtime_config_exact =
      identical(final_runtime$cusolver_deterministic_mode[[1L]], "enabled") &&
      identical(final_runtime$cublas_math_mode[[1L]], "pedantic") &&
      identical(final_runtime$cublas_atomics_mode[[1L]], "not_allowed") &&
      isTRUE(final_runtime$cublas_user_workspace_installed[[1L]]) &&
      final_runtime$cublas_workspace_alignment[[1L]] >= 256,
    rhs_authority = if (length(unique(target_records$rhs_authority)) == 1L) {
      unique(target_records$rhs_authority)
    } else {
      "mixed"
    },
    full_cuda_data_plane = all(target_records$full_cuda_data_plane),
    post_warmup_workspace_grow_count = as.integer(
      final_runtime$workspace_grow_count[[1L]] -
        post_warmup_runtime$workspace_grow_count[[1L]]
    ),
    cuda_device_synchronize_count =
      as.integer(final_runtime$cuda_device_synchronize_count[[1L]]),
    per_target_allocation_count_after_warmup = as.integer(max(c(
      target_records$per_target_allocation_count_after_warmup,
      timing$persistent_resource_records$allocation_count
    ))),
    per_target_handle_create_count = as.integer(max(c(
      target_records$per_target_handle_create_count,
      timing$persistent_resource_records$handle_create_count
    ))),
    invalid_output_init_count = as.integer(sum(
      target_records$invalid_output_init_count
    )),
    cpu_fallback_count = as.integer(sum(target_records$cpu_fallback_count)),
    unknown_fallback_count = as.integer(sum(
      target_records$unknown_fallback_count
    )),
    approximate_fallback_count = as.integer(sum(
      target_records$approximate_fallback_count
    )),
    all_output_slot_leases_released =
      all(target_records$lease_released_before_reuse) &&
      all(!target_records$output_slot_leased_after_release) &&
      all(!prepared_records$output_slot_leased_at_end),
    invalid_output_init_matches_batch_calls =
      sum(target_records$invalid_output_init_count) ==
        nrow(target_records),
    no_non_cholesky_target_ok =
      !any(startsWith(target_records$solver_status[stable], "OK_")),
    persistent_faster_than_repeated_prototype =
      persistent_median_ms < prototype_median_ms && speedup > 1,
    persistent_speedup = as.numeric(speedup)
  )

  exact_gate <- summary$catalog_open_count == 1L &&
    summary$setup_count == 44L && summary$target_count == 270L &&
    summary$cholesky_ok_count == 172L &&
    summary$stable_not_implemented_count == 98L &&
    summary$residual_max_abs_diff_max < 1e-7 &&
    summary$residual_relative_l2_diff_max < 1e-7 &&
    summary$fitted_max_abs_diff_max < 1e-7 &&
    summary$fitted_relative_l2_diff_max < 1e-7 &&
    summary$setup_h2d_upload_count == 44L &&
    summary$runtime_context_create_count == 1L &&
    isTRUE(summary$deterministic_runtime_config_exact) &&
    identical(summary$rhs_authority, "cuda-x0-transpose-y") &&
    isTRUE(summary$full_cuda_data_plane) &&
    summary$post_warmup_workspace_grow_count == 0L &&
    summary$cuda_device_synchronize_count == 0L &&
    all(target_records$resource_snapshot_captured) &&
    all(target_records$resource_instrumentation_version == 1L) &&
    summary$per_target_allocation_count_after_warmup == 0L &&
    summary$per_target_handle_create_count == 0L &&
    summary$cpu_fallback_count == 0L &&
    summary$unknown_fallback_count == 0L &&
    summary$approximate_fallback_count == 0L &&
    isTRUE(summary$all_output_slot_leases_released) &&
    isTRUE(summary$invalid_output_init_matches_batch_calls) &&
    isTRUE(summary$no_non_cholesky_target_ok)
  if (!isTRUE(exact_gate)) {
    stop("Phase 3A iteration correctness/resource gate failed",
         call. = FALSE)
  }
  if (!isTRUE(summary$persistent_faster_than_repeated_prototype)) {
    stop(sprintf(
      paste0(
        "Phase 3A persistent median gate failed: persistent raw ms=%s; ",
        "prototype raw ms=%s; medians=%.6f/%.6f; speedup=%.6f; ",
        "warmup solve/release/free ms=%s"
      ),
      paste(format(persistent_raw_ms, digits = 10), collapse = ","),
      paste(format(prototype_raw_ms, digits = 10), collapse = ","),
      persistent_median_ms, prototype_median_ms, speedup,
      paste(format(timing$persistent_warmup_profile_ms, digits = 10),
            collapse = ",")
    ), call. = FALSE)
  }

  list(
    catalog_records = catalog_records,
    runtime_records = runtime_records,
    prepared_records = prepared_records,
    target_records = target_records,
    timing = timing,
    summary = summary
  )
}

fastkpc_full_cuda_fixed_sp_phase3b_validate_label <- function(
    value, label) {
  clean <- typeof(value) == "character" && length(value) == 1L &&
    !is.object(value) && is.null(attributes(value)) && !is.na(value) &&
    nzchar(value)
  if (!isTRUE(clean)) {
    stop(label, " must be one bare non-empty character scalar",
         call. = FALSE)
  }
  value
}

fastkpc_full_cuda_fixed_sp_phase3b_require_fields <- function(
    value, fields, context) {
  fastkpc_full_cuda_fixed_sp_phase3b_validate_label(context, "context")
  fields_clean <- typeof(fields) == "character" && length(fields) > 0L &&
    !is.object(fields) && is.null(attributes(fields)) && !anyNA(fields) &&
    all(nzchar(fields)) && !anyDuplicated(fields)
  if (!isTRUE(fields_clean)) {
    stop("fields must be bare non-empty unique character names",
         call. = FALSE)
  }
  value_names <- names(value)
  value_clean <- typeof(value) == "list" && !is.object(value) &&
    is.character(value_names) && length(value_names) > 0L &&
    !anyNA(value_names) && all(nzchar(value_names)) &&
    !anyDuplicated(value_names) &&
    identical(attributes(value), list(names = value_names))
  if (!isTRUE(value_clean)) {
    stop(context, " must be a bare named list", call. = FALSE)
  }
  missing <- setdiff(fields, value_names)
  if (length(missing) != 0L) {
    stop(
      context, " fields are malformed; missing=",
      paste(missing, collapse = ","), call. = FALSE
    )
  }
  invisible(value)
}

fastkpc_full_cuda_fixed_sp_phase3b_integer_scalar <- function(
    value, name) {
  fastkpc_full_cuda_fixed_sp_phase3b_validate_label(name, "name")
  clean <- typeof(value) == "integer" && length(value) == 1L &&
    !is.object(value) && is.null(attributes(value)) && !is.na(value) &&
    value >= 0L
  if (!isTRUE(clean)) {
    stop(name, " must be one non-negative integer", call. = FALSE)
  }
  value
}

fastkpc_full_cuda_fixed_sp_phase3b_double_scalar <- function(value, name) {
  fastkpc_full_cuda_fixed_sp_phase3b_validate_label(name, "name")
  clean <- typeof(value) == "double" && length(value) == 1L &&
    !is.object(value) && is.null(attributes(value)) && !is.na(value) &&
    is.finite(value) && value >= 0
  if (!isTRUE(clean)) {
    stop(name, " must be one non-negative finite double", call. = FALSE)
  }
  value
}

fastkpc_full_cuda_fixed_sp_phase3b_logical_scalar <- function(value, name) {
  fastkpc_full_cuda_fixed_sp_phase3b_validate_label(name, "name")
  clean <- typeof(value) == "logical" && length(value) == 1L &&
    !is.object(value) && is.null(attributes(value)) && !is.na(value)
  if (!isTRUE(clean)) {
    stop(name, " must be one non-NA logical", call. = FALSE)
  }
  value
}

fastkpc_full_cuda_fixed_sp_phase3b_character_scalar <- function(
    value, name, allow_empty = FALSE) {
  fastkpc_full_cuda_fixed_sp_phase3b_validate_label(name, "name")
  allow_empty_clean <- typeof(allow_empty) == "logical" &&
    length(allow_empty) == 1L && !is.object(allow_empty) &&
    is.null(attributes(allow_empty)) && !is.na(allow_empty)
  if (!isTRUE(allow_empty_clean)) {
    stop("allow_empty must be one non-NA logical scalar", call. = FALSE)
  }
  clean <- typeof(value) == "character" && length(value) == 1L &&
    !is.object(value) && is.null(attributes(value)) && !is.na(value) &&
    (isTRUE(allow_empty) || nzchar(value))
  if (!isTRUE(clean)) {
    stop(name, " must be one valid character scalar", call. = FALSE)
  }
  value
}

fastkpc_full_cuda_fixed_sp_phase3b_character_vector <- function(
    value, size, name, allow_na = FALSE) {
  size_clean <- typeof(size) == "integer" && length(size) == 1L &&
    !is.object(size) && is.null(attributes(size)) && !is.na(size) &&
    size >= 0L
  if (!isTRUE(size_clean)) {
    stop("size must be one non-negative integer", call. = FALSE)
  }
  fastkpc_full_cuda_fixed_sp_phase3b_validate_label(name, "name")
  allow_na_clean <- typeof(allow_na) == "logical" &&
    length(allow_na) == 1L && !is.object(allow_na) &&
    is.null(attributes(allow_na)) && !is.na(allow_na)
  if (!isTRUE(allow_na_clean)) {
    stop("allow_na must be one non-NA logical scalar", call. = FALSE)
  }
  clean <- typeof(value) == "character" && length(value) == size &&
    !is.object(value) && is.null(attributes(value)) &&
    (isTRUE(allow_na) || !anyNA(value))
  if (!isTRUE(clean)) {
    stop(name, " must be one valid character vector", call. = FALSE)
  }
  value
}

fastkpc_full_cuda_fixed_sp_phase3b_validate_paths <- function(
    phase2_dir, census_dir, prepared_dir, data_path) {
  paths <- list(
    phase2_dir = phase2_dir,
    census_dir = census_dir,
    prepared_dir = prepared_dir,
    data_path = data_path
  )
  for (field in names(paths)) {
    value <- paths[[field]]
    clean <- typeof(value) == "character" && length(value) == 1L &&
      !is.object(value) && is.null(attributes(value)) && !is.na(value) &&
      nzchar(value)
    if (!isTRUE(clean)) {
      stop(field, " must be one bare non-empty character scalar",
           call. = FALSE)
    }
  }
  paths
}

fastkpc_full_cuda_fixed_sp_phase3b_counter_delta <- function(
    before, after, field, context) {
  fastkpc_full_cuda_fixed_sp_phase3b_require_fields(
    before, field, paste(context, "before")
  )
  fastkpc_full_cuda_fixed_sp_phase3b_require_fields(
    after, field, paste(context, "after")
  )
  before_value <- fastkpc_full_cuda_fixed_sp_phase3b_integer_scalar(
    before[[field]], paste(context, field, "before")
  )
  after_value <- fastkpc_full_cuda_fixed_sp_phase3b_integer_scalar(
    after[[field]], paste(context, field, "after")
  )
  delta <- as.double(after_value) - as.double(before_value)
  if (length(delta) != 1L || !is.finite(delta) || delta < 0 ||
      delta != floor(delta) || delta > .Machine$integer.max) {
    stop(context, " counter regressed: ", field, call. = FALSE)
  }
  as.integer(delta)
}

fastkpc_full_cuda_fixed_sp_phase3b_validate_runtime_info <- function(
    info, context) {
  integer_fields <- c(
    "device_id", "runtime_context_create_count",
    "cuda_device_allocation_count", "cuda_host_allocation_count",
    "stream_create_count", "event_create_count",
    "cublas_handle_create_count", "cusolver_handle_create_count",
    "workspace_reserve_count", "workspace_grow_count",
    "cuda_device_synchronize_count",
    "cholesky_factor_checkpoint_record_count",
    "cholesky_factor_checkpoint_wait_count",
    "cholesky_solve_checkpoint_record_count",
    "cholesky_solve_checkpoint_wait_count", "cuda_toolkit_version",
    "cuda_driver_version", "compute_capability_major",
    "compute_capability_minor", "sm_count"
  )
  double_fields <- c(
    "creator_pid", "generation", "workspace_bytes",
    "cublas_workspace_bytes", "cublas_workspace_alignment"
  )
  character_fields <- c(
    "gpu_name", "cusolver_deterministic_mode", "cublas_math_mode",
    "cublas_atomics_mode"
  )
  required <- c(
    integer_fields, double_fields, character_fields,
    "cublas_user_workspace_installed", "freed"
  )
  fastkpc_full_cuda_fixed_sp_phase3b_require_fields(info, required, context)
  for (field in integer_fields) {
    fastkpc_full_cuda_fixed_sp_phase3b_integer_scalar(
      info[[field]], paste(context, field)
    )
  }
  for (field in double_fields) {
    fastkpc_full_cuda_fixed_sp_phase3b_double_scalar(
      info[[field]], paste(context, field)
    )
  }
  for (field in character_fields) {
    fastkpc_full_cuda_fixed_sp_phase3b_character_scalar(
      info[[field]], paste(context, field)
    )
  }
  fastkpc_full_cuda_fixed_sp_phase3b_logical_scalar(
    info$cublas_user_workspace_installed,
    paste(context, "cublas_user_workspace_installed")
  )
  fastkpc_full_cuda_fixed_sp_phase3b_logical_scalar(
    info$freed, paste(context, "freed")
  )
  config_exact <- info$runtime_context_create_count == 1L &&
    identical(info$cusolver_deterministic_mode, "enabled") &&
    identical(info$cublas_math_mode, "pedantic") &&
    identical(info$cublas_atomics_mode, "not_allowed") &&
    !isTRUE(info$freed)
  if (!isTRUE(config_exact)) {
    stop(context, " deterministic runtime configuration is invalid",
         call. = FALSE)
  }
  invisible(info)
}

fastkpc_full_cuda_fixed_sp_phase3b_validate_runtime_identity <- function(
    created, reserved, final, requested_device_id) {
  identity_fields <- c(
    "device_id", "creator_pid", "generation", "gpu_name",
    "compute_capability_major", "compute_capability_minor", "sm_count",
    "cuda_toolkit_version", "cuda_driver_version",
    "cusolver_deterministic_mode", "cublas_math_mode",
    "cublas_atomics_mode"
  )
  requested_device_id <-
    fastkpc_full_cuda_fixed_sp_phase3b_integer_scalar(
      requested_device_id, "requested_device_id"
    )
  snapshots <- list(created = created, reserved = reserved, final = final)
  for (snapshot_name in names(snapshots)) {
    snapshot <- snapshots[[snapshot_name]]
    fastkpc_full_cuda_fixed_sp_phase3b_require_fields(
      snapshot, identity_fields,
      paste("Phase 3B", snapshot_name, "runtime identity")
    )
    fastkpc_full_cuda_fixed_sp_phase3b_integer_scalar(
      snapshot$device_id, paste(snapshot_name, "device_id")
    )
    fastkpc_full_cuda_fixed_sp_phase3b_double_scalar(
      snapshot$creator_pid, paste(snapshot_name, "creator_pid")
    )
    fastkpc_full_cuda_fixed_sp_phase3b_double_scalar(
      snapshot$generation, paste(snapshot_name, "generation")
    )
    fastkpc_full_cuda_fixed_sp_phase3b_character_scalar(
      snapshot$gpu_name, paste(snapshot_name, "gpu_name")
    )
    for (field in c(
      "compute_capability_major", "compute_capability_minor", "sm_count",
      "cuda_toolkit_version", "cuda_driver_version"
    )) {
      fastkpc_full_cuda_fixed_sp_phase3b_integer_scalar(
        snapshot[[field]], paste(snapshot_name, field)
      )
    }
    for (field in c(
      "cusolver_deterministic_mode", "cublas_math_mode",
      "cublas_atomics_mode"
    )) {
      fastkpc_full_cuda_fixed_sp_phase3b_character_scalar(
        snapshot[[field]], paste(snapshot_name, field)
      )
    }
  }
  identity_exact <- identical(created$device_id, requested_device_id) &&
    identical(created[identity_fields], reserved[identity_fields]) &&
    identical(created[identity_fields], final[identity_fields])
  if (!isTRUE(identity_exact)) {
    stop("Phase 3B runtime immutable identity changed", call. = FALSE)
  }
  invisible(TRUE)
}

fastkpc_full_cuda_fixed_sp_phase3b_runtime_record <- function(stage, info) {
  fastkpc_full_cuda_fixed_sp_phase3b_validate_runtime_info(
    info, paste("Phase 3B", stage, "runtime info")
  )
  record <- fastkpc_full_cuda_fixed_sp_phase3a_runtime_record(stage, info)
  record$workspace_reserve_count <- info$workspace_reserve_count
  record$freed <- info$freed
  record$creator_pid <- info$creator_pid
  record$generation <- info$generation
  record
}

fastkpc_full_cuda_fixed_sp_phase3b_validate_prepared_info <- function(
    info, dto, context) {
  integer_fields <- c(
    "n", "coefficient_dim", "null_dim", "penalty_count",
    "setup_h2d_upload_count"
  )
  double_fields <- c(
    "setup_h2d_bytes", "coefficient_output_capacity", "generation"
  )
  character_fields <- c(
    "prepared_s_key_sha256", "output_slot_state",
    "output_slot_poison_reason"
  )
  required <- c(
    integer_fields, double_fields, character_fields, "output_slot_leased"
  )
  fastkpc_full_cuda_fixed_sp_phase3b_require_fields(info, required, context)
  for (field in integer_fields) {
    fastkpc_full_cuda_fixed_sp_phase3b_integer_scalar(
      info[[field]], paste(context, field)
    )
  }
  for (field in double_fields) {
    fastkpc_full_cuda_fixed_sp_phase3b_double_scalar(
      info[[field]], paste(context, field)
    )
  }
  fastkpc_full_cuda_fixed_sp_phase3b_character_scalar(
    info$prepared_s_key_sha256,
    paste(context, "prepared_s_key_sha256")
  )
  fastkpc_full_cuda_fixed_sp_phase3b_character_scalar(
    info$output_slot_state, paste(context, "output_slot_state")
  )
  fastkpc_full_cuda_fixed_sp_phase3b_character_scalar(
    info$output_slot_poison_reason,
    paste(context, "output_slot_poison_reason"), allow_empty = TRUE
  )
  fastkpc_full_cuda_fixed_sp_phase3b_logical_scalar(
    info$output_slot_leased, paste(context, "output_slot_leased")
  )
  identity_exact <- identical(info$prepared_s_key_sha256,
                              dto$prepared_s_key_sha256) &&
    identical(info$n, dto$n) &&
    identical(info$coefficient_dim, dto$coefficient_dim) &&
    identical(info$null_dim, dto$null_dim) &&
    identical(info$penalty_count, dto$penalty_count)
  if (!isTRUE(identity_exact)) {
    stop(context, " prepared identity is inconsistent", call. = FALSE)
  }
  invisible(info)
}

fastkpc_full_cuda_fixed_sp_phase3b_validate_prepared_snapshots <- function(
    before, leased, released, expected_key, required_capacity) {
  fields <- c(
    "prepared_s_key_sha256", "setup_h2d_upload_count", "setup_h2d_bytes",
    "coefficient_output_capacity", "generation", "output_slot_leased",
    "output_slot_state", "output_slot_poison_reason"
  )
  expected_key <- fastkpc_full_cuda_fixed_sp_phase3b_character_scalar(
    expected_key, "expected_key"
  )
  if (!grepl("^[0-9a-f]{64}$", expected_key)) {
    stop("expected_key must be one lowercase SHA-256", call. = FALSE)
  }
  required_capacity <- fastkpc_full_cuda_fixed_sp_phase3b_double_scalar(
    required_capacity, "required_capacity"
  )
  snapshots <- list(before = before, leased = leased, released = released)
  for (snapshot_name in names(snapshots)) {
    snapshot <- snapshots[[snapshot_name]]
    fastkpc_full_cuda_fixed_sp_phase3b_require_fields(
      snapshot, fields, paste("Phase 3B prepared", snapshot_name)
    )
    fastkpc_full_cuda_fixed_sp_phase3b_character_scalar(
      snapshot$prepared_s_key_sha256,
      paste("prepared", snapshot_name, "key")
    )
    fastkpc_full_cuda_fixed_sp_phase3b_integer_scalar(
      snapshot$setup_h2d_upload_count,
      paste("prepared", snapshot_name, "setup upload count")
    )
    for (field in c(
      "setup_h2d_bytes", "coefficient_output_capacity", "generation"
    )) {
      fastkpc_full_cuda_fixed_sp_phase3b_double_scalar(
        snapshot[[field]], paste("prepared", snapshot_name, field)
      )
    }
    fastkpc_full_cuda_fixed_sp_phase3b_logical_scalar(
      snapshot$output_slot_leased,
      paste("prepared", snapshot_name, "leased")
    )
    fastkpc_full_cuda_fixed_sp_phase3b_character_scalar(
      snapshot$output_slot_state,
      paste("prepared", snapshot_name, "state")
    )
    fastkpc_full_cuda_fixed_sp_phase3b_character_scalar(
      snapshot$output_slot_poison_reason,
      paste("prepared", snapshot_name, "poison reason"),
      allow_empty = TRUE
    )
  }
  identity_fields <- c(
    "prepared_s_key_sha256", "setup_h2d_upload_count", "setup_h2d_bytes",
    "coefficient_output_capacity", "generation"
  )
  identity_exact <- identical(before$prepared_s_key_sha256, expected_key) &&
    identical(before[identity_fields], leased[identity_fields]) &&
    identical(before[identity_fields], released[identity_fields]) &&
    before$coefficient_output_capacity >= required_capacity
  lifecycle_exact <-
    identical(before$output_slot_leased, FALSE) &&
    identical(before$output_slot_state, "free") &&
    identical(leased$output_slot_leased, TRUE) &&
    identical(leased$output_slot_state, "leased") &&
    identical(released$output_slot_leased, FALSE) &&
    identical(released$output_slot_state, "free") &&
    identical(before$output_slot_poison_reason, "") &&
    identical(leased$output_slot_poison_reason, "") &&
    identical(released$output_slot_poison_reason, "")
  if (!isTRUE(identity_exact)) {
    stop("Phase 3B prepared snapshot identity changed", call. = FALSE)
  }
  if (!isTRUE(lifecycle_exact)) {
    stop("Phase 3B prepared output-slot lifecycle is invalid",
         call. = FALSE)
  }
  invisible(TRUE)
}

fastkpc_full_cuda_fixed_sp_phase3b_cleanup_failure_condition <- function(
    body_condition, cleanup_condition, context) {
  fastkpc_full_cuda_fixed_sp_phase3b_validate_label(context, "context")
  if (!inherits(body_condition, "condition") ||
      !inherits(cleanup_condition, "condition")) {
    stop("cleanup failure composition requires two conditions",
         call. = FALSE)
  }
  body_classes <- setdiff(class(body_condition), "condition")
  structure(
    list(
      message = paste0(
        conditionMessage(body_condition), "; ",
        conditionMessage(cleanup_condition)
      ),
      call = conditionCall(body_condition),
      parent = body_condition,
      cleanup = cleanup_condition
    ),
    class = unique(c(
      "fastkpc_phase3b_cleanup_failure", body_classes,
      "error", "condition"
    ))
  )
}

fastkpc_full_cuda_fixed_sp_phase3b_cleanup_condition_is_retryable <-
    function(condition) {
  inherits(condition, "condition") &&
    (inherits(condition, "fastkpc_phase3b_cleanup_retryable") ||
       grepl(
         "retryable teardown work", conditionMessage(condition),
         fixed = TRUE
       ))
}

fastkpc_full_cuda_fixed_sp_phase3b_cleanup_operations <- function(
    operations, body_error = NULL, max_attempts = 2L,
    context = "Phase 3B cleanup") {
  suspendInterrupts({
    fastkpc_full_cuda_fixed_sp_phase3b_validate_label(context, "context")
    max_attempts_clean <- typeof(max_attempts) == "integer" &&
      length(max_attempts) == 1L && !is.object(max_attempts) &&
      is.null(attributes(max_attempts)) && !is.na(max_attempts) &&
      max_attempts >= 1L && max_attempts <= 3L
    if (!isTRUE(max_attempts_clean)) {
      stop("max_attempts must be an integer from one to three",
           call. = FALSE)
    }
    if (!is.null(body_error) && !inherits(body_error, "condition")) {
      stop("body_error must be NULL or a condition", call. = FALSE)
    }
    operation_names <- c(
      "token_release", "token_free", "handle_free", "runtime_free"
    )
    fastkpc_full_cuda_fixed_sp_phase3b_require_fields(
      operations, operation_names, paste(context, "operations")
    )
    if (!identical(names(operations), operation_names)) {
      stop(context, " operations are not in canonical order", call. = FALSE)
    }
    for (operation_name in operation_names) {
      operation <- operations[[operation_name]]
      fastkpc_full_cuda_fixed_sp_phase3b_require_fields(
        operation, c("needed", "run"),
        paste(context, operation_name, "operation")
      )
      if (!identical(names(operation), c("needed", "run")) ||
          typeof(operation$needed) != "closure" ||
          typeof(operation$run) != "closure") {
        stop(context, " cleanup operation is malformed: ", operation_name,
             call. = FALSE)
      }
    }

    condition_handler <- function(condition) condition
    failures <- character()
    for (operation_name in operation_names) {
      operation <- operations[[operation_name]]
      last_error <- NULL
      for (attempt in seq_len(max_attempts)) {
        needed <- tryCatch(
          operation$needed(),
          error = condition_handler,
          interrupt = condition_handler
        )
        if (inherits(needed, "condition")) {
          last_error <- needed
          if (!fastkpc_full_cuda_fixed_sp_phase3b_cleanup_condition_is_retryable(
                needed
              ) || attempt == max_attempts) {
            break
          }
          next
        } else if (typeof(needed) != "logical" || length(needed) != 1L ||
                   is.object(needed) || !is.null(attributes(needed)) ||
                   is.na(needed)) {
          last_error <- simpleError(
            "needed() did not return a logical scalar"
          )
          break
        } else if (!needed) {
          last_error <- NULL
          break
        } else {
          run_error <- tryCatch({
            operation$run()
            NULL
          }, error = condition_handler, interrupt = condition_handler)
          if (!is.null(run_error)) {
            last_error <- run_error
            if (!fastkpc_full_cuda_fixed_sp_phase3b_cleanup_condition_is_retryable(
                  run_error
                ) || attempt == max_attempts) {
              break
            }
            next
          } else {
            still_needed <- tryCatch(
              operation$needed(),
              error = condition_handler,
              interrupt = condition_handler
            )
            if (inherits(still_needed, "condition")) {
              last_error <- still_needed
              if (!fastkpc_full_cuda_fixed_sp_phase3b_cleanup_condition_is_retryable(
                    still_needed
                  ) || attempt == max_attempts) {
                break
              }
              next
            } else if (!identical(still_needed, FALSE)) {
              last_error <- simpleError(
                "operation did not clear ownership state"
              )
              break
            } else {
              last_error <- NULL
              break
            }
          }
        }
      }
      if (!is.null(last_error)) {
        failures <- c(
          failures,
          paste0(operation_name, ": ", conditionMessage(last_error))
        )
      }
    }
    if (length(failures) != 0L) {
      cleanup_error <- simpleError(paste0(
        context, " failures: ", paste(failures, collapse = "; ")
      ))
      if (!is.null(body_error)) {
        stop(
          fastkpc_full_cuda_fixed_sp_phase3b_cleanup_failure_condition(
            body_error, cleanup_error, context
          )
        )
      }
      stop(cleanup_error)
    }
    if (!is.null(body_error)) stop(body_error)
    invisible(TRUE)
  })
}

fastkpc_full_cuda_fixed_sp_phase3b_run_cleanup_scope <- function(
    body, operations, context) {
  fastkpc_full_cuda_fixed_sp_phase3b_validate_label(context, "context")
  condition_handler <- function(condition) {
    list(value = NULL, condition = condition)
  }
  capture_condition <- function(expression) {
    tryCatch(
      list(value = force(expression), condition = NULL),
      error = condition_handler,
      interrupt = condition_handler
    )
  }
  run_cleanup <- function() {
    capture_condition(
      fastkpc_full_cuda_fixed_sp_phase3b_cleanup_operations(
        operations, context = context
      )
    )
  }

  cleanup_complete <- FALSE
  primary_condition <- NULL
  on.exit({
    if (!cleanup_complete) {
      fallback_outcome <- run_cleanup()
      if (!is.null(fallback_outcome$condition)) {
        fallback_condition <- fallback_outcome$condition
        if (!is.null(primary_condition)) {
          fallback_condition <-
            fastkpc_full_cuda_fixed_sp_phase3b_cleanup_failure_condition(
              primary_condition, fallback_condition,
              paste(context, "on.exit fallback")
            )
        }
        stop(fallback_condition)
      }
    }
  }, add = TRUE)

  body_outcome <- capture_condition(force(body))
  if (!is.null(body_outcome$condition)) {
    primary_condition <- body_outcome$condition
    cleanup_outcome <- run_cleanup()
    if (is.null(cleanup_outcome$condition)) {
      cleanup_complete <- TRUE
      stop(primary_condition)
    }
    primary_condition <-
      fastkpc_full_cuda_fixed_sp_phase3b_cleanup_failure_condition(
        primary_condition, cleanup_outcome$condition, context
      )
    stop(primary_condition)
  }

  cleanup_outcome <- run_cleanup()
  if (!is.null(cleanup_outcome$condition)) {
    primary_condition <- cleanup_outcome$condition
    stop(primary_condition)
  }
  cleanup_complete <- TRUE
  body_outcome$value
}

fastkpc_full_cuda_fixed_sp_phase3b_validate_batch_info <- function(
    info, native, dto, expected_shadow_calls,
    expected_shadow_targets, expected_shadow_bytes,
    expected_release_count, context) {
  integer_fields <- c(
    "n", "coefficient_dim", "target_count", "batch_call_count",
    "true_batched_subgroup_count", "true_batched_attempted_target_count",
    "true_batched_target_count", "cholesky_single_target_count",
    "potrf_batched_call_count", "potrs_batched_call_count",
    "target_batch_h2d_call_count", "target_h2d_copy_count",
    "coefficient_batch_finalize_call_count",
    "fitted_batch_finalize_call_count",
    "residual_rss_batch_finalize_call_count",
    "per_target_output_finalize_call_count",
    "batch_output_finalized_target_count", "stable_reroute_count",
    "planned_cholesky_target_count", "planned_qr_target_count",
    "planned_svd_target_count", "executed_cholesky_target_count",
    "executed_qr_target_count", "executed_svd_target_count",
    "cholesky_to_svd_count", "qr_to_svd_count",
    "output_slot_acquire_count", "output_slot_release_count",
    "output_slot_busy_count", "stale_token_reject_count",
    "invalid_output_init_count", "nonfinite_output_count",
    "cpu_fallback_count", "unknown_fallback_count",
    "resource_instrumentation_version",
    "resource_allocation_count_before_solve",
    "resource_allocation_count_after_solve",
    "resource_handle_create_count_before_solve",
    "resource_handle_create_count_after_solve",
    "cuda_device_allocation_count_during_solve",
    "cuda_host_allocation_count_during_solve",
    "stream_create_count_during_solve", "event_create_count_during_solve",
    "cublas_handle_create_count_during_solve",
    "cusolver_handle_create_count_during_solve",
    "per_target_allocation_count_after_warmup",
    "per_target_handle_create_count", "implicit_residual_d2h_count",
    "rhs_device_build_count", "shadow_materialize_call_count",
    "shadow_materialize_target_count"
  )
  double_fields <- c(
    "target_h2d_bytes", "shadow_d2h_bytes", "owner_generation",
    "slot_generation"
  )
  logical_fields <- c(
    "native_batch_call", "true_batched_kernel",
    "canonical_output_order_exact", "resource_snapshot_captured",
    "full_cuda_data_plane"
  )
  vector_fields <- c(
    "target_keys", "planned_route", "executed_route", "reroute_reason",
    "solver_status"
  )
  required <- c(
    integer_fields, double_fields, logical_fields, vector_fields,
    "rhs_authority"
  )
  fastkpc_full_cuda_fixed_sp_phase3b_require_fields(info, required, context)
  for (field in integer_fields) {
    fastkpc_full_cuda_fixed_sp_phase3b_integer_scalar(
      info[[field]], paste(context, field)
    )
  }
  for (field in double_fields) {
    fastkpc_full_cuda_fixed_sp_phase3b_double_scalar(
      info[[field]], paste(context, field)
    )
  }
  for (field in logical_fields) {
    fastkpc_full_cuda_fixed_sp_phase3b_logical_scalar(
      info[[field]], paste(context, field)
    )
  }
  target_count <- native$target_count
  fastkpc_full_cuda_fixed_sp_phase3b_character_vector(
    info$target_keys, target_count, paste(context, "target_keys")
  )
  fastkpc_full_cuda_fixed_sp_phase3b_character_vector(
    info$planned_route, target_count, paste(context, "planned_route")
  )
  fastkpc_full_cuda_fixed_sp_phase3b_character_vector(
    info$executed_route, target_count, paste(context, "executed_route"),
    allow_na = TRUE
  )
  fastkpc_full_cuda_fixed_sp_phase3b_character_vector(
    info$reroute_reason, target_count, paste(context, "reroute_reason")
  )
  fastkpc_full_cuda_fixed_sp_phase3b_character_vector(
    info$solver_status, target_count, paste(context, "solver_status")
  )
  fastkpc_full_cuda_fixed_sp_phase3b_character_scalar(
    info$rhs_authority, paste(context, "rhs_authority")
  )
  expected_shadow_calls <-
    fastkpc_full_cuda_fixed_sp_phase3b_integer_scalar(
      expected_shadow_calls, paste(context, "expected shadow calls")
    )
  expected_shadow_targets <-
    fastkpc_full_cuda_fixed_sp_phase3b_integer_scalar(
      expected_shadow_targets, paste(context, "expected shadow targets")
    )
  expected_shadow_bytes <-
    fastkpc_full_cuda_fixed_sp_phase3b_double_scalar(
      expected_shadow_bytes, paste(context, "expected shadow bytes")
    )
  expected_release_count <-
    fastkpc_full_cuda_fixed_sp_phase3b_integer_scalar(
      expected_release_count, paste(context, "expected release count")
    )

  safe <- native$planned_route == "CHOLESKY_BATCHED"
  stable <- !safe
  safe_count <- as.integer(sum(safe))
  batched_subgroup_count <- as.integer(safe_count >= 2L)
  batched_target_count <- if (safe_count >= 2L) safe_count else 0L
  single_target_count <- if (safe_count == 1L) 1L else 0L
  expected_status <- rep("ERR_STABLE_PATH_NOT_IMPLEMENTED", target_count)
  expected_status[safe] <- if (safe_count >= 2L) {
    "OK_CHOLESKY_BATCHED"
  } else {
    "OK_CHOLESKY_SINGLE"
  }
  expected_executed <- rep(NA_character_, target_count)
  expected_executed[safe] <- "CHOLESKY_BATCHED"
  planned_counts <- c(
    CHOLESKY_BATCHED = sum(native$planned_route == "CHOLESKY_BATCHED"),
    AUGMENTED_QR = sum(native$planned_route == "AUGMENTED_QR"),
    AUGMENTED_SVD = sum(native$planned_route == "AUGMENTED_SVD")
  )
  resource_allocation_delta <-
    info$resource_allocation_count_after_solve -
      info$resource_allocation_count_before_solve
  resource_handle_delta <-
    info$resource_handle_create_count_after_solve -
      info$resource_handle_create_count_before_solve
  route_status_conservation_exact <-
    identical(info$target_keys, native$target_keys) &&
    identical(info$planned_route, native$planned_route) &&
    identical(info$executed_route, expected_executed) &&
    identical(info$reroute_reason, rep("", target_count)) &&
    identical(info$solver_status, expected_status) &&
    isTRUE(info$canonical_output_order_exact) &&
    info$true_batched_subgroup_count == batched_subgroup_count &&
    info$true_batched_attempted_target_count == batched_target_count &&
    info$true_batched_target_count == batched_target_count &&
    info$cholesky_single_target_count == single_target_count &&
    info$potrf_batched_call_count == batched_subgroup_count &&
    info$potrs_batched_call_count == batched_subgroup_count &&
    identical(
      info$true_batched_kernel,
      isTRUE(all(safe) && safe_count >= 2L)
    ) &&
    info$planned_cholesky_target_count ==
      planned_counts[["CHOLESKY_BATCHED"]] &&
    info$planned_qr_target_count == planned_counts[["AUGMENTED_QR"]] &&
    info$planned_svd_target_count == planned_counts[["AUGMENTED_SVD"]] &&
    info$executed_cholesky_target_count == safe_count &&
    info$executed_qr_target_count == 0L &&
    info$executed_svd_target_count == 0L &&
    info$stable_reroute_count == 0L &&
    info$cholesky_to_svd_count == 0L && info$qr_to_svd_count == 0L
  semantics_exact <- isTRUE(route_status_conservation_exact) &&
    identical(info$n, dto$n) &&
    identical(info$coefficient_dim, dto$coefficient_dim) &&
    identical(info$target_count, target_count) &&
    isTRUE(info$native_batch_call) && info$batch_call_count == 1L &&
    info$output_slot_acquire_count == 1L &&
    info$output_slot_release_count == expected_release_count &&
    info$output_slot_busy_count == 0L &&
    info$stale_token_reject_count == 0L &&
    info$invalid_output_init_count == 1L &&
    info$nonfinite_output_count == 0L &&
    info$target_batch_h2d_call_count == 1L &&
    info$target_h2d_copy_count == 2L &&
    identical(
      info$target_h2d_bytes,
      8 * as.double(length(native$Y) + length(native$SP))
    ) &&
    info$rhs_device_build_count == 1L &&
    identical(info$rhs_authority, "cuda-x0-transpose-y") &&
    isTRUE(info$full_cuda_data_plane) &&
    info$coefficient_batch_finalize_call_count == 0L &&
    info$fitted_batch_finalize_call_count == as.integer(safe_count > 0L) &&
    info$residual_rss_batch_finalize_call_count ==
      as.integer(safe_count > 0L) &&
    info$per_target_output_finalize_call_count == 0L &&
    info$batch_output_finalized_target_count == safe_count &&
    isTRUE(info$resource_snapshot_captured) &&
    info$resource_instrumentation_version == 1L &&
    resource_allocation_delta ==
      info$cuda_device_allocation_count_during_solve +
        info$cuda_host_allocation_count_during_solve &&
    resource_handle_delta ==
      info$stream_create_count_during_solve +
        info$event_create_count_during_solve +
        info$cublas_handle_create_count_during_solve +
        info$cusolver_handle_create_count_during_solve &&
    all(c(
      info$cuda_device_allocation_count_during_solve,
      info$cuda_host_allocation_count_during_solve,
      info$stream_create_count_during_solve,
      info$event_create_count_during_solve,
      info$cublas_handle_create_count_during_solve,
      info$cusolver_handle_create_count_during_solve,
      info$per_target_allocation_count_after_warmup,
      info$per_target_handle_create_count,
      info$implicit_residual_d2h_count,
      info$cpu_fallback_count, info$unknown_fallback_count
    ) == 0L) &&
    info$shadow_materialize_call_count == expected_shadow_calls &&
    info$shadow_materialize_target_count == expected_shadow_targets &&
    identical(info$shadow_d2h_bytes, expected_shadow_bytes) &&
    all(info$solver_status[stable] ==
        "ERR_STABLE_PATH_NOT_IMPLEMENTED") &&
    all(is.na(info$executed_route[stable]))
  if (!isTRUE(semantics_exact)) {
    stop(context, " route/status/resource conservation failed",
         call. = FALSE)
  }
  list(
    safe = safe,
    route_status_conservation_exact = route_status_conservation_exact
  )
}

fastkpc_full_cuda_fixed_sp_phase3b_is_double_vector <- function(value) {
  attributes_clean <- is.null(attributes(value)) || {
    value_names <- names(value)
    is.character(value_names) && length(value_names) == length(value) &&
      !anyNA(value_names) && identical(
        attributes(value), list(names = value_names)
      )
  }
  typeof(value) == "double" && !is.object(value) && length(value) > 0L &&
    is.null(dim(value)) && isTRUE(attributes_clean) && all(is.finite(value))
}

fastkpc_full_cuda_fixed_sp_phase3b_saturating_product <- function(a, b) {
  if (a == 0 || b == 0) return(0)
  if (a > .Machine$double.xmax / b) return(.Machine$double.xmax)
  a * b
}

fastkpc_full_cuda_fixed_sp_phase3b_scaled_l2_components <- function(value) {
  scale <- max(abs(value))
  if (scale == 0) return(c(scale = 0, factor = 0))
  scaled <- value / scale
  c(scale = scale, factor = sqrt(sum(scaled * scaled)))
}

fastkpc_full_cuda_fixed_sp_phase3b_log_ratio <- function(
    numerator_scale, numerator_factor, denominator_log) {
  if (numerator_scale == 0 || numerator_factor == 0) return(0)
  log_ratio <- log(numerator_scale) + log(numerator_factor) - denominator_log
  if (log_ratio >= log(.Machine$double.xmax)) {
    return(.Machine$double.xmax)
  }
  exp(log_ratio)
}

fastkpc_full_cuda_fixed_sp_phase3b_numeric_errors <- function(
    actual, reference, context) {
  fastkpc_full_cuda_fixed_sp_phase3b_validate_label(context, "context")
  clean <- fastkpc_full_cuda_fixed_sp_phase3b_is_double_vector(actual) &&
    fastkpc_full_cuda_fixed_sp_phase3b_is_double_vector(reference) &&
    identical(length(actual), length(reference))
  if (!isTRUE(clean)) {
    stop(
      context,
      " requires finite non-empty bare double vectors with equal shape",
      call. = FALSE
    )
  }

  actual <- unname(actual)
  reference <- unname(reference)
  joint_scale <- max(abs(actual), abs(reference))
  difference_scaled <- if (joint_scale == 0) {
    numeric(length(actual))
  } else {
    actual / joint_scale - reference / joint_scale
  }
  difference_components <-
    fastkpc_full_cuda_fixed_sp_phase3b_scaled_l2_components(
      difference_scaled
    )
  difference_factor <- difference_components[["scale"]] *
    difference_components[["factor"]]
  max_abs <- fastkpc_full_cuda_fixed_sp_phase3b_saturating_product(
    joint_scale, max(abs(difference_scaled))
  )

  reference_components <-
    fastkpc_full_cuda_fixed_sp_phase3b_scaled_l2_components(reference)
  reference_log <- if (reference_components[["scale"]] == 0) {
    -Inf
  } else {
    log(reference_components[["scale"]]) +
      log(reference_components[["factor"]])
  }
  denominator_log <- max(reference_log, log(1e-300))
  relative_l2 <- if (reference_log > log(1e-300)) {
    scale_ratio <- joint_scale / reference_components[["scale"]]
    factor_ratio <- difference_factor / reference_components[["factor"]]
    if (is.finite(scale_ratio) && is.finite(factor_ratio)) {
      fastkpc_full_cuda_fixed_sp_phase3b_saturating_product(
        scale_ratio, factor_ratio
      )
    } else {
      fastkpc_full_cuda_fixed_sp_phase3b_log_ratio(
        joint_scale, difference_factor, denominator_log
      )
    }
  } else {
    fastkpc_full_cuda_fixed_sp_phase3b_log_ratio(
      joint_scale, difference_factor, denominator_log
    )
  }
  if (!is.finite(max_abs) || !is.finite(relative_l2) || max_abs < 0 ||
      relative_l2 < 0) {
    stop(context, " produced invalid numeric error evidence", call. = FALSE)
  }
  c(max_abs = max_abs, relative_l2 = relative_l2)
}

fastkpc_full_cuda_fixed_sp_phase3b_require_record_fields <- function(
    records, fields, context) {
  fastkpc_full_cuda_fixed_sp_phase3b_validate_label(context, "context")
  fields_clean <- typeof(fields) == "character" && length(fields) > 0L &&
    !is.object(fields) && is.null(attributes(fields)) && !anyNA(fields) &&
    all(nzchar(fields)) && !anyDuplicated(fields)
  if (!isTRUE(fields_clean)) {
    stop("record fields must be bare non-empty unique character names",
         call. = FALSE)
  }
  record_names <- names(records)
  records_clean <- is.data.frame(records) && nrow(records) > 0L &&
    is.character(record_names) && !anyNA(record_names) &&
    all(nzchar(record_names)) && !anyDuplicated(record_names)
  if (!isTRUE(records_clean)) {
    stop(context, " must be one non-empty data frame", call. = FALSE)
  }
  missing <- setdiff(fields, record_names)
  if (length(missing) != 0L) {
    stop(
      context, " fields are malformed; missing=",
      paste(missing, collapse = ","), call. = FALSE
    )
  }
  row_count <- nrow(records)
  for (field in fields) {
    if (!identical(length(records[[field]]), row_count)) {
      stop(context, " field length is invalid: ", field, call. = FALSE)
    }
  }
  invisible(records)
}

fastkpc_full_cuda_fixed_sp_phase3b_is_sha_vector <- function(
    value, size, allow_na = FALSE) {
  clean <- typeof(value) == "character" && length(value) == size &&
    !is.object(value) && is.null(attributes(value)) &&
    (isTRUE(allow_na) || !anyNA(value))
  if (!isTRUE(clean)) return(FALSE)
  present <- value[!is.na(value)]
  length(present) == 0L || all(grepl("^[0-9a-f]{64}$", present))
}

fastkpc_full_cuda_fixed_sp_phase3b_validate_result_evidence <- function(
    catalog_records, batch_records, target_records,
    expected_iteration_subset_hash, tolerance = 1e-7) {
  expected_iteration_subset_hash <-
    fastkpc_full_cuda_fixed_sp_phase3b_character_scalar(
      expected_iteration_subset_hash, "expected_iteration_subset_hash"
    )
  if (!grepl("^[0-9a-f]{64}$", expected_iteration_subset_hash)) {
    stop("expected iteration subset hash is invalid", call. = FALSE)
  }
  tolerance_clean <- typeof(tolerance) == "double" &&
    length(tolerance) == 1L && !is.object(tolerance) &&
    is.null(attributes(tolerance)) && !is.na(tolerance) &&
    is.finite(tolerance) && tolerance > 0
  if (!isTRUE(tolerance_clean)) {
    stop("tolerance must be one positive finite double", call. = FALSE)
  }

  catalog_fields <- c(
    "iteration_subset_hash", "ordered_setup_key_digest",
    "ordered_target_key_digest"
  )
  batch_fields <- c(
    "prepared_s_key_sha256", "target_count",
    "coefficient_batch_finalize_call_count",
    "fitted_batch_finalize_call_count",
    "residual_rss_batch_finalize_call_count",
    "per_target_output_finalize_call_count",
    "batch_output_finalized_target_count",
    "route_status_conservation_exact"
  )
  target_fields <- c(
    "prepared_s_key_sha256", "residual_key_sha256", "target",
    "authenticated_planned_route", "executed_route", "solver_status",
    "residual_max_abs_diff", "residual_relative_l2_diff",
    "fitted_max_abs_diff", "fitted_relative_l2_diff",
    "oracle_call_count", "oracle_fitted_hash", "authenticated_fitted_hash",
    "oracle_residual_hash", "authenticated_residual_hash",
    "oracle_fitted_hash_exact", "oracle_residual_hash_exact"
  )
  fastkpc_full_cuda_fixed_sp_phase3b_require_record_fields(
    catalog_records, catalog_fields, "Phase 3B catalog evidence"
  )
  fastkpc_full_cuda_fixed_sp_phase3b_require_record_fields(
    batch_records, batch_fields, "Phase 3B batch evidence"
  )
  fastkpc_full_cuda_fixed_sp_phase3b_require_record_fields(
    target_records, target_fields, "Phase 3B target evidence"
  )
  if (nrow(catalog_records) != 1L) {
    stop("Phase 3B catalog digest evidence is invalid", call. = FALSE)
  }

  setup_count <- nrow(batch_records)
  target_count <- nrow(target_records)
  setup_keys <- batch_records$prepared_s_key_sha256
  target_keys <- target_records$residual_key_sha256
  key_evidence_exact <-
    fastkpc_full_cuda_fixed_sp_phase3b_is_sha_vector(
      setup_keys, setup_count
    ) && !anyDuplicated(setup_keys) &&
    fastkpc_full_cuda_fixed_sp_phase3b_is_sha_vector(
      target_records$prepared_s_key_sha256, target_count
    ) &&
    fastkpc_full_cuda_fixed_sp_phase3b_is_sha_vector(
      target_keys, target_count
    ) && !anyDuplicated(target_keys) &&
    typeof(target_records$target) == "integer" &&
    !is.object(target_records$target) &&
    is.null(attributes(target_records$target)) &&
    !anyNA(target_records$target) &&
    all(target_records$target >= 1L & target_records$target <= 48L)
  if (!isTRUE(key_evidence_exact)) {
    stop("Phase 3B target key evidence is invalid", call. = FALSE)
  }

  catalog_values <- unlist(catalog_records[1L, catalog_fields], use.names = TRUE)
  catalog_evidence_exact <- all(vapply(
    catalog_values, function(value) {
      typeof(value) == "character" && length(value) == 1L &&
        !is.na(value) && grepl("^[0-9a-f]{64}$", value)
    }, logical(1L)
  )) && identical(
    catalog_records$iteration_subset_hash[[1L]],
    expected_iteration_subset_hash
  ) && identical(
    catalog_records$ordered_setup_key_digest[[1L]],
    fastkpc_full_cuda_census_key_set_hash(setup_keys)
  ) && identical(
    catalog_records$ordered_target_key_digest[[1L]],
    fastkpc_full_cuda_census_key_set_hash(target_keys)
  )
  if (!isTRUE(catalog_evidence_exact)) {
    stop("Phase 3B catalog digest evidence is invalid", call. = FALSE)
  }

  route_fields <- c(
    "authenticated_planned_route", "executed_route", "solver_status"
  )
  route_types_clean <- all(vapply(route_fields, function(field) {
    value <- target_records[[field]]
    typeof(value) == "character" && !is.object(value) &&
      is.null(attributes(value)) && length(value) == target_count
  }, logical(1L))) && !anyNA(target_records$authenticated_planned_route) &&
    !anyNA(target_records$solver_status)
  if (!isTRUE(route_types_clean)) {
    stop("Phase 3B target route evidence is invalid", call. = FALSE)
  }
  safe <- target_records$authenticated_planned_route == "CHOLESKY_BATCHED"
  stable <- !safe
  if (!any(safe) || !any(stable)) {
    stop("Phase 3B target route evidence is invalid", call. = FALSE)
  }
  route_evidence_exact <-
    !anyNA(target_records$executed_route[safe]) &&
    all(target_records$executed_route[safe] == "CHOLESKY_BATCHED") &&
    all(is.na(target_records$executed_route[stable])) &&
    all(target_records$solver_status[safe] %in%
        c("OK_CHOLESKY_BATCHED", "OK_CHOLESKY_SINGLE")) &&
    all(target_records$solver_status[stable] ==
        "ERR_STABLE_PATH_NOT_IMPLEMENTED")
  if (!isTRUE(route_evidence_exact)) {
    stop("Phase 3B target route evidence is invalid", call. = FALSE)
  }

  error_fields <- c(
    "residual_max_abs_diff", "residual_relative_l2_diff",
    "fitted_max_abs_diff", "fitted_relative_l2_diff"
  )
  numeric_evidence_exact <- all(vapply(error_fields, function(field) {
    value <- target_records[[field]]
    typeof(value) == "double" && !is.object(value) &&
      is.null(attributes(value)) && length(value) == target_count &&
      all(is.finite(value[safe])) &&
      all(value[safe] >= 0 & value[safe] < tolerance) &&
      all(is.na(value[stable]) & !is.nan(value[stable]))
  }, logical(1L)))
  if (!isTRUE(numeric_evidence_exact)) {
    stop("Phase 3B target numeric evidence is invalid", call. = FALSE)
  }

  oracle_count <- target_records$oracle_call_count
  oracle_hash_fields <- c(
    "oracle_fitted_hash", "authenticated_fitted_hash",
    "oracle_residual_hash", "authenticated_residual_hash"
  )
  oracle_flag_fields <- c(
    "oracle_fitted_hash_exact", "oracle_residual_hash_exact"
  )
  oracle_types_clean <- typeof(oracle_count) == "integer" &&
    !is.object(oracle_count) && is.null(attributes(oracle_count)) &&
    !anyNA(oracle_count) && all(vapply(oracle_hash_fields, function(field) {
      value <- target_records[[field]]
      typeof(value) == "character" && !is.object(value) &&
        is.null(attributes(value)) && length(value) == target_count
    }, logical(1L))) && all(vapply(oracle_flag_fields, function(field) {
      value <- target_records[[field]]
      typeof(value) == "logical" && !is.object(value) &&
        is.null(attributes(value)) && length(value) == target_count
    }, logical(1L)))
  oracle_evidence_exact <- isTRUE(oracle_types_clean) &&
    all(oracle_count[safe] == 1L) && all(oracle_count[stable] == 0L) &&
    fastkpc_full_cuda_fixed_sp_phase3b_is_sha_vector(
      target_records$authenticated_fitted_hash, target_count
    ) && fastkpc_full_cuda_fixed_sp_phase3b_is_sha_vector(
      target_records$authenticated_residual_hash, target_count
    ) && fastkpc_full_cuda_fixed_sp_phase3b_is_sha_vector(
      target_records$oracle_fitted_hash[safe], as.integer(sum(safe))
    ) && fastkpc_full_cuda_fixed_sp_phase3b_is_sha_vector(
      target_records$oracle_residual_hash[safe], as.integer(sum(safe))
    ) && all(is.na(target_records$oracle_fitted_hash[stable])) &&
    all(is.na(target_records$oracle_residual_hash[stable])) &&
    all(target_records$oracle_fitted_hash_exact[safe] %in% TRUE) &&
    all(target_records$oracle_residual_hash_exact[safe] %in% TRUE) &&
    all(is.na(target_records$oracle_fitted_hash_exact[stable])) &&
    all(is.na(target_records$oracle_residual_hash_exact[stable])) &&
    identical(
      target_records$oracle_fitted_hash[safe],
      target_records$authenticated_fitted_hash[safe]
    ) && identical(
      target_records$oracle_residual_hash[safe],
      target_records$authenticated_residual_hash[safe]
    )
  if (!isTRUE(oracle_evidence_exact)) {
    stop("Phase 3B target oracle evidence is invalid", call. = FALSE)
  }

  integer_batch_fields <- setdiff(
    batch_fields,
    c("prepared_s_key_sha256", "route_status_conservation_exact")
  )
  batch_types_clean <- all(vapply(integer_batch_fields, function(field) {
    value <- batch_records[[field]]
    typeof(value) == "integer" && !is.object(value) &&
      is.null(attributes(value)) && !anyNA(value) && all(value >= 0L)
  }, logical(1L))) &&
    typeof(batch_records$route_status_conservation_exact) == "logical" &&
    !is.object(batch_records$route_status_conservation_exact) &&
    is.null(attributes(batch_records$route_status_conservation_exact)) &&
    !anyNA(batch_records$route_status_conservation_exact) &&
    all(batch_records$route_status_conservation_exact)
  if (!isTRUE(batch_types_clean)) {
    stop("Phase 3B batch finalize evidence is invalid", call. = FALSE)
  }
  target_indices <- lapply(setup_keys, function(setup_key) {
    which(target_records$prepared_s_key_sha256 == setup_key)
  })
  owned_target_count <- vapply(target_indices, length, integer(1L))
  safe_count <- vapply(target_indices, function(indices) {
    as.integer(sum(safe[indices]))
  }, integer(1L))
  any_safe <- as.integer(safe_count > 0L)
  finalize_evidence_exact <-
    identical(batch_records$target_count, owned_target_count) &&
    all(batch_records$coefficient_batch_finalize_call_count == 0L) &&
    identical(batch_records$fitted_batch_finalize_call_count, any_safe) &&
    identical(
      batch_records$residual_rss_batch_finalize_call_count, any_safe
    ) &&
    all(batch_records$per_target_output_finalize_call_count == 0L) &&
    identical(batch_records$batch_output_finalized_target_count, safe_count)
  if (!isTRUE(finalize_evidence_exact)) {
    stop("Phase 3B batch finalize evidence is invalid", call. = FALSE)
  }

  list(
    iteration_subset_hash = expected_iteration_subset_hash,
    ordered_setup_key_digest =
      catalog_records$ordered_setup_key_digest[[1L]],
    ordered_target_key_digest =
      catalog_records$ordered_target_key_digest[[1L]],
    oracle_call_count = as.integer(sum(oracle_count)),
    oracle_fitted_hash_exact_count = as.integer(sum(
      target_records$oracle_fitted_hash_exact[safe]
    )),
    oracle_residual_hash_exact_count = as.integer(sum(
      target_records$oracle_residual_hash_exact[safe]
    )),
    coefficient_batch_finalize_call_count = as.integer(sum(
      batch_records$coefficient_batch_finalize_call_count
    )),
    fitted_batch_finalize_call_count = as.integer(sum(
      batch_records$fitted_batch_finalize_call_count
    )),
    residual_rss_batch_finalize_call_count = as.integer(sum(
      batch_records$residual_rss_batch_finalize_call_count
    )),
    per_target_output_finalize_call_count = as.integer(sum(
      batch_records$per_target_output_finalize_call_count
    )),
    batch_output_finalized_target_count = as.integer(sum(
      batch_records$batch_output_finalized_target_count
    ))
  )
}

fastkpc_full_cuda_fixed_sp_phase3b_summarize <- function(
    catalog_records, batch_records, target_records) {
  if (!is.data.frame(catalog_records) || !is.data.frame(batch_records) ||
      !is.data.frame(target_records) || nrow(catalog_records) != 1L ||
      nrow(batch_records) < 1L || nrow(target_records) < 1L) {
    stop("Phase 3B iteration records are malformed", call. = FALSE)
  }
  setup_keys <- batch_records$prepared_s_key_sha256
  target_indices <- lapply(setup_keys, function(setup_key) {
    which(target_records$prepared_s_key_sha256 == setup_key)
  })
  target_count_by_setup <- vapply(target_indices, length, integer(1L))
  if (any(target_count_by_setup < 1L) ||
      sum(target_count_by_setup) != nrow(target_records) ||
      any(!target_records$prepared_s_key_sha256 %in% setup_keys)) {
    stop("Phase 3B iteration target ownership is malformed", call. = FALSE)
  }
  safe_count_by_setup <- vapply(target_indices, function(indices) {
    as.integer(sum(
      target_records$authenticated_planned_route[indices] ==
        "CHOLESKY_BATCHED"
    ))
  }, integer(1L))
  all_safe <- safe_count_by_setup == target_count_by_setup
  all_stable <- safe_count_by_setup == 0L
  mixed <- !all_safe & !all_stable
  evidence <- fastkpc_full_cuda_fixed_sp_phase3b_validate_result_evidence(
    catalog_records, batch_records, target_records,
    catalog_records$iteration_subset_hash[[1L]]
  )
  list(
    iteration_subset_hash = evidence$iteration_subset_hash,
    ordered_setup_key_digest = evidence$ordered_setup_key_digest,
    ordered_target_key_digest = evidence$ordered_target_key_digest,
    setup_count = as.integer(nrow(batch_records)),
    target_count = as.integer(nrow(target_records)),
    batch_call_count = as.integer(sum(batch_records$batch_call_count)),
    all_safe_batch_count = as.integer(sum(all_safe)),
    mixed_batch_count = as.integer(sum(mixed)),
    all_stable_batch_count = as.integer(sum(all_stable)),
    cholesky_ok_count = as.integer(sum(
      target_records$solver_status %in%
        c("OK_CHOLESKY_BATCHED", "OK_CHOLESKY_SINGLE")
    )),
    true_batched_subgroup_count = as.integer(sum(
      batch_records$true_batched_subgroup_count
    )),
    true_batched_target_count = as.integer(sum(
      target_records$solver_status == "OK_CHOLESKY_BATCHED"
    )),
    cholesky_single_target_count = as.integer(sum(
      target_records$solver_status == "OK_CHOLESKY_SINGLE"
    )),
    whole_batch_true_batched_count = as.integer(sum(
      batch_records$true_batched_kernel
    )),
    stable_not_implemented_count = as.integer(sum(
      target_records$solver_status == "ERR_STABLE_PATH_NOT_IMPLEMENTED"
    )),
    oracle_call_count = evidence$oracle_call_count,
    oracle_fitted_hash_exact_count =
      evidence$oracle_fitted_hash_exact_count,
    oracle_residual_hash_exact_count =
      evidence$oracle_residual_hash_exact_count,
    setup_h2d_upload_count = as.integer(sum(
      batch_records$setup_h2d_upload_count
    )),
    target_batch_h2d_call_count = as.integer(sum(
      batch_records$target_batch_h2d_call_count
    )),
    target_h2d_copy_count = as.integer(sum(
      batch_records$target_h2d_copy_count
    )),
    rhs_device_build_count = as.integer(sum(
      batch_records$rhs_device_build_count
    )),
    full_cuda_data_plane = all(batch_records$full_cuda_data_plane),
    invalid_output_init_count = as.integer(sum(
      batch_records$invalid_output_init_count
    )),
    coefficient_batch_finalize_call_count =
      evidence$coefficient_batch_finalize_call_count,
    fitted_batch_finalize_call_count =
      evidence$fitted_batch_finalize_call_count,
    residual_rss_batch_finalize_call_count =
      evidence$residual_rss_batch_finalize_call_count,
    per_target_output_finalize_call_count =
      evidence$per_target_output_finalize_call_count,
    batch_output_finalized_target_count =
      evidence$batch_output_finalized_target_count,
    workspace_grow_count_after_warmup = as.integer(sum(
      batch_records$workspace_grow_count_after_warmup
    )),
    per_target_allocation_count_after_warmup = as.integer(sum(
      batch_records$per_target_allocation_count_after_warmup
    )),
    per_target_handle_create_count = as.integer(sum(
      batch_records$per_target_handle_create_count
    )),
    cuda_device_synchronize_count = as.integer(sum(
      batch_records$cuda_device_synchronize_count
    )),
    implicit_residual_d2h_count = as.integer(sum(
      batch_records$implicit_residual_d2h_count
    )),
    all_output_slot_leases_released = all(
      batch_records$output_slot_release_count == 1L &
        !batch_records$output_slot_leased_after_release
    ),
    cpu_fallback_count = as.integer(sum(batch_records$cpu_fallback_count)),
    unknown_fallback_count = as.integer(sum(
      batch_records$unknown_fallback_count
    ))
  )
}

fastkpc_run_full_cuda_fixed_sp_phase3b_iteration <- function(
    phase2_dir, census_dir, prepared_dir, data_path, device_id = 0L) {
  required_functions <- c(
    "fixed_sp_cuda_runtime_create", "fixed_sp_cuda_runtime_reserve",
    "fixed_sp_cuda_runtime_info", "fixed_sp_cuda_runtime_free",
    "fixed_sp_cuda_prepared_create", "fixed_sp_cuda_prepared_info",
    "fixed_sp_cuda_prepared_free", "fixed_sp_cuda_solve_batch",
    "fixed_sp_cuda_residual_info", "fixed_sp_cuda_materialize_shadow",
    "fixed_sp_cuda_residual_release", "fixed_sp_cuda_residual_free",
    "fastkpc_mgcv_magic_fixed_sp_from_prepared"
  )
  missing <- required_functions[!vapply(
    required_functions, exists, logical(1L), mode = "function",
    inherits = TRUE
  )]
  if (length(missing) != 0L) {
    stop(
      "Phase 3B CUDA API is unavailable: ", paste(missing, collapse = ", "),
      call. = FALSE
    )
  }
  validated_paths <- fastkpc_full_cuda_fixed_sp_phase3b_validate_paths(
    phase2_dir, census_dir, prepared_dir, data_path
  )
  phase2_dir <- validated_paths$phase2_dir
  census_dir <- validated_paths$census_dir
  prepared_dir <- validated_paths$prepared_dir
  data_path <- validated_paths$data_path
  if (typeof(device_id) != "integer" || length(device_id) != 1L ||
      is.object(device_id) || !is.null(attributes(device_id)) ||
      is.na(device_id) || device_id < 0L) {
    stop("Phase 3B device_id must be one non-negative integer",
         call. = FALSE)
  }

  catalog <- fastkpc_full_cuda_open_fixed_sp_catalog(
    phase2_dir, census_dir, prepared_dir, data_path
  )
  iteration <- fastkpc_full_cuda_fixed_sp_scope(catalog, "iteration")
  batches <- fastkpc_full_cuda_fixed_sp_batches(catalog, iteration)
  setup_keys <- names(batches)
  if (length(setup_keys) < 1L ||
      !identical(setup_keys, sort(setup_keys, method = "radix")) ||
      anyDuplicated(setup_keys)) {
    stop("Phase 3B iteration PreparedSKey order is not canonical",
         call. = FALSE)
  }
  ordered_target_keys <- unlist(lapply(batches, function(batch) {
    as.character(batch$target_rows$residual_key_sha256)
  }), use.names = FALSE)
  iteration_subset_hash <- catalog$catalog_contract$iteration_subset_hash
  catalog_records <- data.frame(
    scope = "iteration",
    authenticated = TRUE,
    catalog_open_count = 1L,
    setup_count = as.integer(length(batches)),
    target_count = as.integer(sum(vapply(
      batches, function(batch) nrow(batch$target_rows), integer(1L)
    ))),
    iteration_subset_hash = iteration_subset_hash,
    ordered_setup_key_digest =
      fastkpc_full_cuda_census_key_set_hash(setup_keys),
    ordered_target_key_digest =
      fastkpc_full_cuda_census_key_set_hash(ordered_target_keys),
    stringsAsFactors = FALSE
  )

  dtos <- lapply(batches, function(batch) {
    fastkpc_full_cuda_fixed_sp_native_dto(batch$setup)
  })
  native_batches <- lapply(seq_along(batches), function(index) {
    fastkpc_full_cuda_fixed_sp_native_batch(
      batches[[index]], dtos[[index]]
    )
  })
  names(native_batches) <- names(dtos) <- setup_keys
  max_n <- max(vapply(dtos, `[[`, integer(1L), "n"))
  max_q <- max(vapply(dtos, `[[`, integer(1L), "null_dim"))
  max_targets <- max(vapply(
    native_batches, `[[`, integer(1L), "target_count"
  ))
  max_penalties <- max(vapply(dtos, `[[`, integer(1L), "penalty_count"))
  max_augmented_rows <- max(vapply(dtos, function(dto) {
    as.integer(dto$n + sum(dto$penalty_ranks))
  }, integer(1L)))

  runtime <- NULL
  runtime_freed <- FALSE
  runtime_cleanup_operations <- list(
    token_release = list(
      needed = function() FALSE,
      run = function() invisible(NULL)
    ),
    token_free = list(
      needed = function() FALSE,
      run = function() invisible(NULL)
    ),
    handle_free = list(
      needed = function() FALSE,
      run = function() invisible(NULL)
    ),
    runtime_free = list(
      needed = function() !is.null(runtime) && !runtime_freed,
      run = function() {
        fixed_sp_cuda_runtime_free(runtime)
        runtime_freed <<- TRUE
        runtime <<- NULL
      }
    )
  )
  iteration_result <-
    fastkpc_full_cuda_fixed_sp_phase3b_run_cleanup_scope(
      body = {
    runtime <- fixed_sp_cuda_runtime_create(device_id)
    runtime_created <- fixed_sp_cuda_runtime_info(runtime)
  fastkpc_full_cuda_fixed_sp_phase3b_validate_runtime_info(
    runtime_created, "Phase 3B runtime-created info"
  )
  fixed_sp_cuda_runtime_reserve(
    runtime, max_n, max_q, max_targets, max_penalties, max_augmented_rows
  )
  runtime_reserved <- fixed_sp_cuda_runtime_info(runtime)
  fastkpc_full_cuda_fixed_sp_phase3b_validate_runtime_info(
    runtime_reserved, "Phase 3B workspace-reserved info"
  )
  reserve_count_delta <-
    fastkpc_full_cuda_fixed_sp_phase3b_counter_delta(
      runtime_created, runtime_reserved, "workspace_reserve_count",
      "Phase 3B runtime reserve"
    )
  if (runtime_created$workspace_reserve_count != 0L ||
      reserve_count_delta != 1L ||
      runtime_created$cuda_device_synchronize_count != 0L ||
      runtime_reserved$cuda_device_synchronize_count != 0L ||
      !isTRUE(runtime_reserved$cublas_user_workspace_installed) ||
      runtime_reserved$cublas_workspace_alignment < 256) {
    stop("Phase 3B runtime create/reserve lifecycle is invalid",
         call. = FALSE)
  }

  run_batch <- function(batch_index) {
    setup_key <- setup_keys[[batch_index]]
    batch <- batches[[setup_key]]
    dto <- dtos[[setup_key]]
    native <- native_batches[[setup_key]]
    runtime_before_batch <- fixed_sp_cuda_runtime_info(runtime)
    fastkpc_full_cuda_fixed_sp_phase3b_validate_runtime_info(
      runtime_before_batch,
      paste("Phase 3B batch", batch_index, "runtime before")
    )
    handle <- NULL
    token <- NULL
    token_released <- FALSE
    token_freed <- FALSE
    handle_freed <- FALSE
    prepared_info <- NULL
    prepared_after_solve <- NULL
    prepared_after_release <- NULL
    pre_shadow_info <- NULL
    post_shadow_info <- NULL
    released_info <- NULL
    safe <- NULL
    route_status_conservation_exact <- FALSE
    shadow <- NULL
    target_count <- native$target_count
    expected_shadow_calls <- 0L
    expected_shadow_targets <- 0L
    expected_shadow_bytes <- 0

    cleanup_operations <- list(
      token_release = list(
        needed = function() !is.null(token) && !token_released,
        run = function() {
          fixed_sp_cuda_residual_release(token)
          token_released <<- TRUE
          released_info <<- fixed_sp_cuda_residual_info(token)
          if (!is.null(post_shadow_info)) {
            fastkpc_full_cuda_fixed_sp_phase3b_validate_batch_info(
              released_info, native, dto, expected_shadow_calls,
              expected_shadow_targets, expected_shadow_bytes, 1L,
              paste("Phase 3B batch", batch_index, "released info")
            )
            release_fields <- setdiff(
              names(post_shadow_info), "output_slot_release_count"
            )
            if (!identical(post_shadow_info[release_fields],
                           released_info[release_fields])) {
              stop("Phase 3B token release changed solve diagnostics",
                   call. = FALSE)
            }
          }
          prepared_after_release <<- fixed_sp_cuda_prepared_info(handle)
          fastkpc_full_cuda_fixed_sp_phase3b_validate_prepared_info(
            prepared_after_release, dto,
            paste("Phase 3B batch", batch_index, "released prepared info")
          )
          if (!is.null(prepared_info) && !is.null(prepared_after_solve)) {
            fastkpc_full_cuda_fixed_sp_phase3b_validate_prepared_snapshots(
              prepared_info, prepared_after_solve, prepared_after_release,
              setup_key,
              as.double(dto$coefficient_dim) * as.double(target_count)
            )
          }
        }
      ),
      token_free = list(
        needed = function() !is.null(token) && !token_freed,
        run = function() {
          fixed_sp_cuda_residual_free(token)
          token_freed <<- TRUE
          token_released <<- TRUE
          token <<- NULL
        }
      ),
      handle_free = list(
        needed = function() !is.null(handle) && !handle_freed,
        run = function() {
          fixed_sp_cuda_prepared_free(handle)
          handle_freed <<- TRUE
          handle <<- NULL
        }
      ),
      runtime_free = list(
        needed = function() FALSE,
        run = function() invisible(NULL)
      )
    )

    fastkpc_full_cuda_fixed_sp_phase3b_run_cleanup_scope(
      body = {
      handle <- fixed_sp_cuda_prepared_create(runtime, dto)
      prepared_info <- fixed_sp_cuda_prepared_info(handle)
      fastkpc_full_cuda_fixed_sp_phase3b_validate_prepared_info(
        prepared_info, dto,
        paste("Phase 3B batch", batch_index, "prepared info")
      )
      if (prepared_info$setup_h2d_upload_count != 1L ||
          isTRUE(prepared_info$output_slot_leased)) {
        stop("Phase 3B prepared setup/upload lifecycle is invalid",
             call. = FALSE)
      }

      token <- fixed_sp_cuda_solve_batch(
        handle, native$Y, native$SP, native$planned_route,
        native$target_keys, outputs = c("fitted", "residuals")
      )
      pre_shadow_info <- fixed_sp_cuda_residual_info(token)
      pre_validation <-
        fastkpc_full_cuda_fixed_sp_phase3b_validate_batch_info(
          pre_shadow_info, native, dto, 0L, 0L, 0, 0L,
          paste("Phase 3B batch", batch_index, "pre-shadow info")
        )
      safe <- pre_validation$safe
      route_status_conservation_exact <-
        pre_validation$route_status_conservation_exact
      prepared_after_solve <- fixed_sp_cuda_prepared_info(handle)
      fastkpc_full_cuda_fixed_sp_phase3b_validate_prepared_info(
        prepared_after_solve, dto,
        paste("Phase 3B batch", batch_index, "post-solve prepared info")
      )
      if (!isTRUE(prepared_after_solve$output_slot_leased)) {
        stop("Phase 3B solve did not retain its output-slot lease",
             call. = FALSE)
      }

      expected_shadow_calls <- as.integer(any(safe))
      expected_shadow_targets <- if (any(safe)) target_count else 0L
      expected_shadow_bytes <- if (any(safe)) {
        8 * as.double(sum(safe)) * as.double(2L * dto$n)
      } else {
        0
      }
      if (any(safe)) {
        shadow <- fixed_sp_cuda_materialize_shadow(
          token, outputs = c("fitted", "residuals")
        )
        shadow_clean <- is.list(shadow) &&
          identical(names(shadow), c("fitted", "residuals")) &&
          is.matrix(shadow$fitted) && is.double(shadow$fitted) &&
          identical(dim(shadow$fitted), c(dto$n, target_count)) &&
          is.matrix(shadow$residuals) && is.double(shadow$residuals) &&
          identical(dim(shadow$residuals), c(dto$n, target_count))
        if (!isTRUE(shadow_clean)) {
          stop("Phase 3B explicit oracle shadow is malformed",
               call. = FALSE)
        }
      }

      post_shadow_info <- fixed_sp_cuda_residual_info(token)
      fastkpc_full_cuda_fixed_sp_phase3b_validate_batch_info(
        post_shadow_info, native, dto, expected_shadow_calls,
        expected_shadow_targets, expected_shadow_bytes, 0L,
        paste("Phase 3B batch", batch_index, "post-shadow info")
      )
      shadow_fields <- c(
        "shadow_materialize_call_count", "shadow_materialize_target_count",
        "shadow_d2h_bytes"
      )
      unchanged_fields <- setdiff(names(pre_shadow_info), shadow_fields)
      if (!identical(pre_shadow_info[unchanged_fields],
                     post_shadow_info[unchanged_fields])) {
        stop("Phase 3B explicit shadow changed non-shadow diagnostics",
             call. = FALSE)
      }
        NULL
      },
      operations = cleanup_operations,
      context = paste("Phase 3B batch", batch_index, "cleanup")
    )

    residual_max_abs <- rep(NA_real_, target_count)
    residual_relative_l2 <- rep(NA_real_, target_count)
    fitted_max_abs <- rep(NA_real_, target_count)
    fitted_relative_l2 <- rep(NA_real_, target_count)
    oracle_call_count <- integer(target_count)
    oracle_fitted_hash <- rep(NA_character_, target_count)
    oracle_residual_hash <- rep(NA_character_, target_count)
    authenticated_fitted_hash <- as.character(batch$target_rows$fitted_hash)
    authenticated_residual_hash <- as.character(batch$target_rows$residual_hash)
    oracle_fitted_hash_exact <- rep(NA, target_count)
    oracle_residual_hash_exact <- rep(NA, target_count)
    for (target_index in seq_len(target_count)) {
      if (safe[[target_index]]) {
        oracle_call_count[[target_index]] <- 1L
        oracle <- fastkpc_mgcv_magic_fixed_sp_from_prepared(
          prepared_setup = batch$setup,
          target_state = list(
            row = batch$target_rows[target_index, , drop = FALSE],
            y = as.numeric(native$Y[, target_index])
          )
        )
        residual_errors <-
          fastkpc_full_cuda_fixed_sp_phase3b_numeric_errors(
            shadow$residuals[, target_index], oracle$residuals,
            paste("Phase 3B residual target", native$target_keys[[target_index]])
          )
        fitted_errors <-
          fastkpc_full_cuda_fixed_sp_phase3b_numeric_errors(
            shadow$fitted[, target_index], oracle$fitted,
            paste("Phase 3B fitted target", native$target_keys[[target_index]])
          )
        errors <- c(residual_errors, fitted_errors)
        if (any(!is.finite(errors)) || any(errors >= 1e-7)) {
          stop(
            "Phase 3B safe target failed numeric oracle parity: ",
            native$target_keys[[target_index]], call. = FALSE
          )
        }
        residual_max_abs[[target_index]] <- residual_errors[["max_abs"]]
        residual_relative_l2[[target_index]] <-
          residual_errors[["relative_l2"]]
        fitted_max_abs[[target_index]] <- fitted_errors[["max_abs"]]
        fitted_relative_l2[[target_index]] <-
          fitted_errors[["relative_l2"]]
        oracle_fitted_hash[[target_index]] <-
          fastkpc_full_cuda_census_metadata_hash(oracle$fitted)
        oracle_residual_hash[[target_index]] <-
          fastkpc_full_cuda_census_metadata_hash(oracle$residuals)
        oracle_fitted_hash_exact[[target_index]] <- identical(
          oracle_fitted_hash[[target_index]],
          authenticated_fitted_hash[[target_index]]
        )
        oracle_residual_hash_exact[[target_index]] <- identical(
          oracle_residual_hash[[target_index]],
          authenticated_residual_hash[[target_index]]
        )
        if (!isTRUE(oracle_fitted_hash_exact[[target_index]]) ||
            !isTRUE(oracle_residual_hash_exact[[target_index]])) {
          stop("Phase 3B safe target oracle hash mismatch: ",
               native$target_keys[[target_index]], call. = FALSE)
        }
      } else if (!is.null(shadow)) {
        stable_na <-
          all(is.na(shadow$fitted[, target_index]) &
              !is.nan(shadow$fitted[, target_index])) &&
          all(is.na(shadow$residuals[, target_index]) &
              !is.nan(shadow$residuals[, target_index]))
        if (!isTRUE(stable_na)) {
          stop("Phase 3B stable output was presented as executed",
               call. = FALSE)
        }
      }
    }

    runtime_after_batch <- fixed_sp_cuda_runtime_info(runtime)
    fastkpc_full_cuda_fixed_sp_phase3b_validate_runtime_info(
      runtime_after_batch,
      paste("Phase 3B batch", batch_index, "runtime after")
    )
    workspace_grow_delta <-
      fastkpc_full_cuda_fixed_sp_phase3b_counter_delta(
        runtime_before_batch, runtime_after_batch, "workspace_grow_count",
        paste("Phase 3B batch", batch_index)
      )
    device_synchronize_delta <-
      fastkpc_full_cuda_fixed_sp_phase3b_counter_delta(
        runtime_before_batch, runtime_after_batch,
        "cuda_device_synchronize_count",
        paste("Phase 3B batch", batch_index)
      )
    if (workspace_grow_delta != 0L || device_synchronize_delta != 0L) {
      stop("Phase 3B batch created an unreserved resource or synchronized",
           call. = FALSE)
    }

    target_records <- data.frame(
      prepared_s_key_sha256 = setup_key,
      batch_ordinal = as.integer(batch_index),
      target_ordinal = seq_len(target_count),
      residual_key_sha256 = native$target_keys,
      target = as.integer(batch$target_rows$target),
      planned_route = pre_shadow_info$planned_route,
      authenticated_planned_route = native$planned_route,
      executed_route = pre_shadow_info$executed_route,
      reroute_reason = pre_shadow_info$reroute_reason,
      solver_status = pre_shadow_info$solver_status,
      residual_max_abs_diff = residual_max_abs,
      residual_relative_l2_diff = residual_relative_l2,
      fitted_max_abs_diff = fitted_max_abs,
      fitted_relative_l2_diff = fitted_relative_l2,
      explicit_oracle_shadow_observation = safe,
      oracle_call_count = oracle_call_count,
      oracle_fitted_hash = oracle_fitted_hash,
      authenticated_fitted_hash = authenticated_fitted_hash,
      oracle_residual_hash = oracle_residual_hash,
      authenticated_residual_hash = authenticated_residual_hash,
      oracle_fitted_hash_exact = oracle_fitted_hash_exact,
      oracle_residual_hash_exact = oracle_residual_hash_exact,
      stringsAsFactors = FALSE
    )
    prepared_record <- data.frame(
      prepared_s_key_sha256 = setup_key,
      batch_ordinal = as.integer(batch_index),
      prepared_handle_create_count = 1L,
      prepared_handle_free_count = 1L,
      setup_h2d_upload_count = prepared_info$setup_h2d_upload_count,
      setup_h2d_bytes = prepared_info$setup_h2d_bytes,
      coefficient_output_capacity =
        prepared_info$coefficient_output_capacity,
      prepared_generation = prepared_info$generation,
      output_slot_state_before_solve = prepared_info$output_slot_state,
      output_slot_state_after_solve =
        prepared_after_solve$output_slot_state,
      output_slot_state_after_release =
        prepared_after_release$output_slot_state,
      output_slot_poison_reason_empty = identical(
        c(
          prepared_info$output_slot_poison_reason,
          prepared_after_solve$output_slot_poison_reason,
          prepared_after_release$output_slot_poison_reason
        ),
        rep("", 3L)
      ),
      output_slot_leased_after_release =
        prepared_after_release$output_slot_leased,
      stringsAsFactors = FALSE
    )
    batch_record <- data.frame(
      prepared_s_key_sha256 = setup_key,
      batch_ordinal = as.integer(batch_index),
      target_count = pre_shadow_info$target_count,
      batch_call_count = pre_shadow_info$batch_call_count,
      native_batch_call = pre_shadow_info$native_batch_call,
      true_batched_kernel = pre_shadow_info$true_batched_kernel,
      true_batched_subgroup_count =
        pre_shadow_info$true_batched_subgroup_count,
      true_batched_attempted_target_count =
        pre_shadow_info$true_batched_attempted_target_count,
      true_batched_target_count = pre_shadow_info$true_batched_target_count,
      cholesky_single_target_count =
        pre_shadow_info$cholesky_single_target_count,
      potrf_batched_call_count = pre_shadow_info$potrf_batched_call_count,
      potrs_batched_call_count = pre_shadow_info$potrs_batched_call_count,
      planned_cholesky_target_count =
        pre_shadow_info$planned_cholesky_target_count,
      planned_qr_target_count = pre_shadow_info$planned_qr_target_count,
      planned_svd_target_count = pre_shadow_info$planned_svd_target_count,
      executed_cholesky_target_count =
        pre_shadow_info$executed_cholesky_target_count,
      executed_qr_target_count = pre_shadow_info$executed_qr_target_count,
      executed_svd_target_count = pre_shadow_info$executed_svd_target_count,
      stable_reroute_count = pre_shadow_info$stable_reroute_count,
      cholesky_to_svd_count = pre_shadow_info$cholesky_to_svd_count,
      qr_to_svd_count = pre_shadow_info$qr_to_svd_count,
      setup_h2d_upload_count = prepared_info$setup_h2d_upload_count,
      target_batch_h2d_call_count =
        pre_shadow_info$target_batch_h2d_call_count,
      target_h2d_copy_count = pre_shadow_info$target_h2d_copy_count,
      target_h2d_bytes = pre_shadow_info$target_h2d_bytes,
      rhs_device_build_count = pre_shadow_info$rhs_device_build_count,
      full_cuda_data_plane = pre_shadow_info$full_cuda_data_plane,
      invalid_output_init_count = pre_shadow_info$invalid_output_init_count,
      coefficient_batch_finalize_call_count =
        pre_shadow_info$coefficient_batch_finalize_call_count,
      fitted_batch_finalize_call_count =
        pre_shadow_info$fitted_batch_finalize_call_count,
      residual_rss_batch_finalize_call_count =
        pre_shadow_info$residual_rss_batch_finalize_call_count,
      per_target_output_finalize_call_count =
        pre_shadow_info$per_target_output_finalize_call_count,
      batch_output_finalized_target_count =
        pre_shadow_info$batch_output_finalized_target_count,
      canonical_output_order_exact =
        pre_shadow_info$canonical_output_order_exact,
      target_keys_exact = identical(
        pre_shadow_info$target_keys, native$target_keys
      ),
      route_status_conservation_exact = route_status_conservation_exact,
      resource_snapshot_captured =
        pre_shadow_info$resource_snapshot_captured,
      resource_instrumentation_version =
        pre_shadow_info$resource_instrumentation_version,
      resource_allocation_count_before_solve =
        pre_shadow_info$resource_allocation_count_before_solve,
      resource_allocation_count_after_solve =
        pre_shadow_info$resource_allocation_count_after_solve,
      resource_handle_create_count_before_solve =
        pre_shadow_info$resource_handle_create_count_before_solve,
      resource_handle_create_count_after_solve =
        pre_shadow_info$resource_handle_create_count_after_solve,
      cuda_device_allocation_count_during_solve =
        pre_shadow_info$cuda_device_allocation_count_during_solve,
      cuda_host_allocation_count_during_solve =
        pre_shadow_info$cuda_host_allocation_count_during_solve,
      stream_create_count_during_solve =
        pre_shadow_info$stream_create_count_during_solve,
      event_create_count_during_solve =
        pre_shadow_info$event_create_count_during_solve,
      cublas_handle_create_count_during_solve =
        pre_shadow_info$cublas_handle_create_count_during_solve,
      cusolver_handle_create_count_during_solve =
        pre_shadow_info$cusolver_handle_create_count_during_solve,
      per_target_allocation_count_after_warmup =
        pre_shadow_info$per_target_allocation_count_after_warmup,
      per_target_handle_create_count =
        pre_shadow_info$per_target_handle_create_count,
      workspace_grow_count_after_warmup = workspace_grow_delta,
      cuda_device_synchronize_count = device_synchronize_delta,
      implicit_residual_d2h_count =
        released_info$implicit_residual_d2h_count,
      cpu_fallback_count = pre_shadow_info$cpu_fallback_count,
      unknown_fallback_count = pre_shadow_info$unknown_fallback_count,
      pre_shadow_materialize_call_count =
        pre_shadow_info$shadow_materialize_call_count,
      pre_shadow_materialize_target_count =
        pre_shadow_info$shadow_materialize_target_count,
      pre_shadow_d2h_bytes = pre_shadow_info$shadow_d2h_bytes,
      explicit_oracle_shadow_observation = any(safe),
      shadow_observation_purpose = if (any(safe)) {
        "explicit-oracle-comparison"
      } else {
        "none"
      },
      post_shadow_materialize_call_count =
        post_shadow_info$shadow_materialize_call_count,
      post_shadow_materialize_target_count =
        post_shadow_info$shadow_materialize_target_count,
      post_shadow_d2h_bytes = post_shadow_info$shadow_d2h_bytes,
      output_slot_release_count = released_info$output_slot_release_count,
      output_slot_leased_after_release =
        prepared_after_release$output_slot_leased,
      stringsAsFactors = FALSE
    )
    list(
      prepared_record = prepared_record,
      batch_record = batch_record,
      target_records = target_records
    )
  }

  batch_results <- lapply(seq_along(setup_keys), run_batch)
  prepared_records <- do.call(
    rbind, lapply(batch_results, `[[`, "prepared_record")
  )
  batch_records <- do.call(
    rbind, lapply(batch_results, `[[`, "batch_record")
  )
  target_records <- do.call(
    rbind, lapply(batch_results, `[[`, "target_records")
  )
  rownames(prepared_records) <- NULL
  rownames(batch_records) <- NULL
  rownames(target_records) <- NULL

  runtime_final <- fixed_sp_cuda_runtime_info(runtime)
  fastkpc_full_cuda_fixed_sp_phase3b_validate_runtime_info(
    runtime_final, "Phase 3B final runtime info"
  )
  fastkpc_full_cuda_fixed_sp_phase3b_validate_runtime_identity(
    runtime_created, runtime_reserved, runtime_final, device_id
  )
  final_workspace_delta <-
    fastkpc_full_cuda_fixed_sp_phase3b_counter_delta(
      runtime_reserved, runtime_final, "workspace_grow_count",
      "Phase 3B final runtime"
    )
  final_synchronize_delta <-
    fastkpc_full_cuda_fixed_sp_phase3b_counter_delta(
      runtime_reserved, runtime_final, "cuda_device_synchronize_count",
      "Phase 3B final runtime"
    )
  lifecycle_exact <- runtime_final$workspace_reserve_count ==
    runtime_reserved$workspace_reserve_count &&
    isTRUE(runtime_final$cublas_user_workspace_installed) &&
    runtime_final$cublas_workspace_alignment >= 256 &&
    final_workspace_delta ==
      sum(batch_records$workspace_grow_count_after_warmup) &&
    final_synchronize_delta ==
      sum(batch_records$cuda_device_synchronize_count)
  if (!isTRUE(lifecycle_exact)) {
    stop("Phase 3B runtime lifecycle does not match batch records",
         call. = FALSE)
  }
  runtime_records <- do.call(rbind, list(
    fastkpc_full_cuda_fixed_sp_phase3b_runtime_record(
      "runtime-created", runtime_created
    ),
    fastkpc_full_cuda_fixed_sp_phase3b_runtime_record(
      "workspace-reserved", runtime_reserved
    ),
    fastkpc_full_cuda_fixed_sp_phase3b_runtime_record("final", runtime_final)
  ))
  rownames(runtime_records) <- NULL
  summary <- fastkpc_full_cuda_fixed_sp_phase3b_summarize(
    catalog_records, batch_records, target_records
  )

        list(
          catalog_records = catalog_records,
          runtime_records = runtime_records,
          prepared_records = prepared_records,
          batch_records = batch_records,
          target_records = target_records,
          summary = summary
        )
      },
      operations = runtime_cleanup_operations,
      context = "Phase 3B iteration cleanup"
    )
  iteration_result
}

fastkpc_full_cuda_fixed_sp_phase3c_validate_runtime_info <- function(
    info, context) {
  fastkpc_full_cuda_fixed_sp_phase3b_validate_runtime_info(info, context)
  integer_fields <- c(
    "gesvdj_info_create_count", "gesvdj_info_destroy_count",
    "stable_workspace_grow_count", "qr_checkpoint_record_count",
    "qr_checkpoint_wait_count", "svd_checkpoint_record_count",
    "svd_checkpoint_wait_count"
  )
  double_fields <- c(
    "eigen_workspace_bytes", "qr_workspace_bytes", "svd_workspace_bytes",
    "augmented_workspace_bytes", "aggregate_factor_workspace_bytes"
  )
  fastkpc_full_cuda_fixed_sp_phase3b_require_fields(
    info, c(integer_fields, double_fields), context
  )
  for (field in integer_fields) {
    fastkpc_full_cuda_fixed_sp_phase3b_integer_scalar(
      info[[field]], paste(context, field)
    )
  }
  for (field in double_fields) {
    fastkpc_full_cuda_fixed_sp_phase3b_double_scalar(
      info[[field]], paste(context, field)
    )
  }
  invisible(info)
}

fastkpc_full_cuda_fixed_sp_phase3c_runtime_record <- function(stage, info) {
  fastkpc_full_cuda_fixed_sp_phase3c_validate_runtime_info(
    info, paste("Phase 3C", stage, "runtime info")
  )
  record <- fastkpc_full_cuda_fixed_sp_phase3b_runtime_record(stage, info)
  integer_fields <- c(
    "gesvdj_info_create_count", "gesvdj_info_destroy_count",
    "stable_workspace_grow_count",
    "cholesky_factor_checkpoint_record_count",
    "cholesky_factor_checkpoint_wait_count",
    "cholesky_solve_checkpoint_record_count",
    "cholesky_solve_checkpoint_wait_count",
    "qr_checkpoint_record_count", "qr_checkpoint_wait_count",
    "svd_checkpoint_record_count", "svd_checkpoint_wait_count"
  )
  double_fields <- c(
    "workspace_bytes", "cublas_workspace_bytes", "eigen_workspace_bytes",
    "qr_workspace_bytes", "svd_workspace_bytes", "augmented_workspace_bytes",
    "aggregate_factor_workspace_bytes"
  )
  for (field in integer_fields) record[[field]] <- as.integer(info[[field]])
  for (field in double_fields) record[[field]] <- as.double(info[[field]])
  record
}

fastkpc_full_cuda_fixed_sp_phase3c_expected_X_null <- function(dto) {
  if (identical(dto$constraint_mode, "identity")) {
    dto$X
  } else {
    dto$X %*% dto$constraint_nullspace
  }
}

fastkpc_full_cuda_fixed_sp_phase3c_expected_Z <- function(dto) {
  if (identical(dto$constraint_mode, "identity")) {
    diag(dto$coefficient_dim)
  } else {
    dto$constraint_nullspace
  }
}

fastkpc_full_cuda_fixed_sp_phase3c_projected_penalty <- function(
    dto, penalty_index, Z) {
  block <- dto$penalty_blocks[[penalty_index]]
  offset <- dto$penalty_offsets_zero_based[[penalty_index]] + 1L
  indices <- offset:(offset + nrow(block) - 1L)
  full <- matrix(0, dto$coefficient_dim, dto$coefficient_dim)
  full[indices, indices] <- block
  if (identical(dto$constraint_mode, "identity")) {
    full
  } else {
    crossprod(Z, full %*% Z)
  }
}

fastkpc_full_cuda_fixed_sp_phase3c_cpu_aggregate_shadow <- function(
    dto, sp) {
  if (!is.list(dto) || !is.double(sp) ||
      length(sp) != dto$penalty_count || any(!is.finite(sp)) ||
      any(sp < 0)) {
    stop("Phase 3C CPU aggregate-shadow inputs are malformed",
         call. = FALSE)
  }
  q <- dto$null_dim
  Z <- fastkpc_full_cuda_fixed_sp_phase3c_expected_Z(dto)
  aggregate_penalty <- if (is.null(dto$H)) {
    matrix(0, q, q)
  } else if (identical(dto$constraint_mode, "identity")) {
    dto$H
  } else {
    crossprod(Z, dto$H %*% Z)
  }
  for (penalty_index in seq_len(dto$penalty_count)) {
    aggregate_penalty <- aggregate_penalty +
      sp[[penalty_index]] *
        fastkpc_full_cuda_fixed_sp_phase3c_projected_penalty(
          dto, penalty_index, Z
        )
  }
  factor <- suppressWarnings(chol(
    aggregate_penalty, pivot = TRUE, tol = -1
  ))
  rank <- as.integer(attr(factor, "rank"))
  pivot <- as.integer(attr(factor, "pivot"))
  root <- matrix(0, nrow = rank, ncol = q)
  if (rank > 0L) {
    root[, pivot] <- factor[seq_len(rank), , drop = FALSE]
  }
  B <- rbind(
    fastkpc_full_cuda_fixed_sp_phase3c_expected_X_null(dto),
    root,
    matrix(0, nrow = q - rank, ncol = q)
  )
  if (!identical(dim(B), c(dto$n + q, q))) {
    stop("Phase 3C CPU aggregate padded B is malformed", call. = FALSE)
  }
  singular_values <- svd(B, nu = 0L, nv = 0L)$d
  sigma_max <- max(singular_values)
  threshold <- sigma_max * sqrt(.Machine$double.eps)
  retained <- singular_values > 0 & singular_values >= threshold
  list(
    root_rank = rank,
    root_pivot = pivot,
    effective_rank = as.integer(sum(retained)),
    effective_rank_threshold = as.double(threshold),
    sigma_max = as.double(sigma_max)
  )
}

fastkpc_full_cuda_fixed_sp_phase3c_validate_prepared_info <- function(
    info, dto, expected_rank_shadow_count, context) {
  fastkpc_full_cuda_fixed_sp_phase3b_validate_prepared_info(
    info, dto, context
  )
  integer_fields <- c(
    "penalty_root_build_count", "penalty_root_rank_mismatch_count",
    "penalty_root_matrix_count", "penalty_root_row_count",
    "H_root_matrix_count", "H_root_rank", "setup_shadow_d2h_count",
    "augmented_test_shadow_d2h_count"
  )
  double_fields <- c(
    "penalty_root_bytes", "penalty_root_build_ms",
    "setup_shadow_d2h_bytes", "augmented_test_shadow_d2h_bytes"
  )
  fastkpc_full_cuda_fixed_sp_phase3b_require_fields(
    info, c(integer_fields, double_fields), context
  )
  for (field in integer_fields) {
    fastkpc_full_cuda_fixed_sp_phase3b_integer_scalar(
      info[[field]], paste(context, field)
    )
  }
  for (field in double_fields) {
    fastkpc_full_cuda_fixed_sp_phase3b_double_scalar(
      info[[field]], paste(context, field)
    )
  }
  expected_rank_shadow_count <-
    fastkpc_full_cuda_fixed_sp_phase3b_integer_scalar(
      expected_rank_shadow_count,
      paste(context, "expected rank-reference shadow count")
    )
  expected_H_count <- as.integer(!is.null(dto$H))
  metrics_exact <- info$penalty_root_build_count == 1L &&
    info$penalty_root_matrix_count == dto$penalty_count &&
    info$penalty_root_row_count == sum(dto$penalty_ranks) &&
    info$H_root_matrix_count == expected_H_count &&
    info$setup_shadow_d2h_count == expected_rank_shadow_count &&
    info$augmented_test_shadow_d2h_count == 0L &&
    info$augmented_test_shadow_d2h_bytes == 0 &&
    info$penalty_root_bytes >= 0 && info$penalty_root_build_ms >= 0 &&
    if (expected_rank_shadow_count == 0L) {
      info$setup_shadow_d2h_bytes == 0
    } else {
      info$setup_shadow_d2h_bytes > 0
    }
  if (!isTRUE(metrics_exact)) {
    stop(context, " root/setup-shadow diagnostics are inconsistent",
         call. = FALSE)
  }
  invisible(info)
}

fastkpc_full_cuda_fixed_sp_phase3c_route_counts <- function(
    planned_route, executed_route) {
  target_count <- length(planned_route)
  clean <- is.character(planned_route) && is.character(executed_route) &&
    length(executed_route) == target_count && target_count > 0L &&
    !anyNA(planned_route) && !anyNA(executed_route) &&
    all(planned_route %in% c(
      "CHOLESKY_BATCHED", "AUGMENTED_QR", "AUGMENTED_SVD"
    )) && all(executed_route %in% c(
      "CHOLESKY_BATCHED", "AUGMENTED_QR", "AUGMENTED_SVD"
    ))
  if (!isTRUE(clean)) {
    stop("Phase 3C route evidence is malformed", call. = FALSE)
  }
  planned_cholesky <- sum(planned_route == "CHOLESKY_BATCHED")
  planned_qr <- sum(planned_route == "AUGMENTED_QR")
  planned_svd <- sum(planned_route == "AUGMENTED_SVD")
  executed_cholesky <- sum(executed_route == "CHOLESKY_BATCHED")
  executed_qr <- sum(executed_route == "AUGMENTED_QR")
  executed_svd <- sum(executed_route == "AUGMENTED_SVD")
  cholesky_to_svd <- sum(
    planned_route == "CHOLESKY_BATCHED" &
      executed_route == "AUGMENTED_SVD"
  )
  qr_to_svd <- sum(
    planned_route == "AUGMENTED_QR" &
      executed_route == "AUGMENTED_SVD"
  )
  stable_reroute <- sum(planned_route != executed_route)
  conserved <-
    planned_cholesky == executed_cholesky + cholesky_to_svd &&
    planned_qr == executed_qr + qr_to_svd &&
    executed_svd == planned_svd + cholesky_to_svd + qr_to_svd &&
    stable_reroute == cholesky_to_svd + qr_to_svd
  list(
    planned_cholesky = as.integer(planned_cholesky),
    planned_qr = as.integer(planned_qr),
    planned_svd = as.integer(planned_svd),
    executed_cholesky = as.integer(executed_cholesky),
    executed_qr = as.integer(executed_qr),
    executed_svd = as.integer(executed_svd),
    cholesky_to_svd = as.integer(cholesky_to_svd),
    qr_to_svd = as.integer(qr_to_svd),
    stable_reroute = as.integer(stable_reroute),
    conserved = isTRUE(conserved)
  )
}

fastkpc_full_cuda_fixed_sp_phase3c_validate_batch_info <- function(
    info, native, dto, expected_shadow_calls,
    expected_shadow_targets, expected_shadow_bytes,
    expected_release_count, context, expected_coefficients = FALSE) {
  integer_fields <- c(
    "n", "coefficient_dim", "target_count", "batch_call_count",
    "true_batched_subgroup_count", "true_batched_attempted_target_count",
    "true_batched_target_count", "cholesky_single_target_count",
    "potrf_batched_call_count", "potrs_batched_call_count",
    "target_batch_h2d_call_count", "target_h2d_copy_count",
    "coefficient_batch_finalize_call_count",
    "fitted_batch_finalize_call_count",
    "residual_rss_batch_finalize_call_count",
    "per_target_output_finalize_call_count",
    "batch_output_finalized_target_count", "stable_reroute_count",
    "planned_cholesky_target_count", "planned_qr_target_count",
    "planned_svd_target_count", "executed_cholesky_target_count",
    "executed_qr_target_count", "executed_svd_target_count",
    "cholesky_to_svd_count", "qr_to_svd_count",
    "aggregate_penalty_factor_count", "aggregate_svd_b_build_count",
    "aggregate_penalty_root_d2h_count",
    "output_slot_acquire_count", "output_slot_release_count",
    "output_slot_busy_count", "stale_token_reject_count",
    "invalid_output_init_count", "nonfinite_output_count",
    "cpu_fallback_count", "unknown_fallback_count",
    "resource_instrumentation_version",
    "resource_allocation_count_before_solve",
    "resource_allocation_count_after_solve",
    "resource_handle_create_count_before_solve",
    "resource_handle_create_count_after_solve",
    "cuda_device_allocation_count_during_solve",
    "cuda_host_allocation_count_during_solve",
    "stream_create_count_during_solve", "event_create_count_during_solve",
    "cublas_handle_create_count_during_solve",
    "cusolver_handle_create_count_during_solve",
    "per_target_allocation_count_after_warmup",
    "per_target_handle_create_count", "implicit_residual_d2h_count",
    "rhs_device_build_count", "shadow_materialize_call_count",
    "shadow_materialize_target_count"
  )
  double_fields <- c(
    "target_h2d_bytes", "shadow_d2h_bytes", "owner_generation",
    "slot_generation", "aggregate_penalty_root_d2h_bytes"
  )
  logical_fields <- c(
    "native_batch_call", "true_batched_kernel",
    "canonical_output_order_exact", "resource_snapshot_captured",
    "full_cuda_data_plane"
  )
  character_vectors <- c(
    "target_keys", "planned_route", "executed_route", "reroute_reason",
    "solver_status"
  )
  integer_vectors <- c(
    "qr_rank", "geqrf_info", "ormqr_info", "effective_rank", "svd_info",
    "aggregate_factor_call_count", "aggregate_b_build_count"
  )
  double_vectors <- c(
    "sigma_max", "smallest_retained_sigma"
  )
  required <- c(
    integer_fields, double_fields, logical_fields, character_vectors,
    integer_vectors, double_vectors, "target_true_batched", "rhs_authority",
    "aggregate_penalty_root_rank", "aggregate_penalty_root_pivot",
    "aggregate_dstop"
  )
  fastkpc_full_cuda_fixed_sp_phase3b_require_fields(info, required, context)
  for (field in integer_fields) {
    fastkpc_full_cuda_fixed_sp_phase3b_integer_scalar(
      info[[field]], paste(context, field)
    )
  }
  for (field in double_fields) {
    fastkpc_full_cuda_fixed_sp_phase3b_double_scalar(
      info[[field]], paste(context, field)
    )
  }
  for (field in logical_fields) {
    fastkpc_full_cuda_fixed_sp_phase3b_logical_scalar(
      info[[field]], paste(context, field)
    )
  }
  target_count <- native$target_count
  for (field in character_vectors) {
    fastkpc_full_cuda_fixed_sp_phase3b_character_vector(
      info[[field]], target_count, paste(context, field)
    )
  }
  for (field in integer_vectors) {
    value <- info[[field]]
    if (typeof(value) != "integer" || length(value) != target_count ||
        is.object(value) || !is.null(attributes(value)) || anyNA(value)) {
      stop(context, " ", field, " is malformed", call. = FALSE)
    }
  }
  for (field in double_vectors) {
    value <- info[[field]]
    if (typeof(value) != "double" || length(value) != target_count ||
        is.object(value) || !is.null(attributes(value))) {
      stop(context, " ", field, " is malformed", call. = FALSE)
    }
  }
  target_true_batched <- info$target_true_batched
  if (typeof(target_true_batched) != "logical" ||
      length(target_true_batched) != target_count ||
      is.object(target_true_batched) ||
      !is.null(attributes(target_true_batched)) ||
      anyNA(target_true_batched)) {
    stop(context, " target_true_batched is malformed", call. = FALSE)
  }
  aggregate_root_rank <- info$aggregate_penalty_root_rank
  aggregate_root_pivot <- info$aggregate_penalty_root_pivot
  aggregate_dstop <- info$aggregate_dstop
  if (typeof(aggregate_root_rank) != "integer" ||
      length(aggregate_root_rank) != target_count ||
      is.object(aggregate_root_rank) ||
      !is.null(attributes(aggregate_root_rank)) ||
      !is.list(aggregate_root_pivot) ||
      length(aggregate_root_pivot) != target_count ||
      typeof(aggregate_dstop) != "double" ||
      length(aggregate_dstop) != target_count ||
      is.object(aggregate_dstop) ||
      !is.null(attributes(aggregate_dstop))) {
    stop(context, " aggregate SVD diagnostics are malformed",
         call. = FALSE)
  }
  fastkpc_full_cuda_fixed_sp_phase3b_character_scalar(
    info$rhs_authority, paste(context, "rhs_authority")
  )
  expected_shadow_calls <-
    fastkpc_full_cuda_fixed_sp_phase3b_integer_scalar(
      expected_shadow_calls, paste(context, "expected shadow calls")
    )
  expected_shadow_targets <-
    fastkpc_full_cuda_fixed_sp_phase3b_integer_scalar(
      expected_shadow_targets, paste(context, "expected shadow targets")
    )
  expected_shadow_bytes <-
    fastkpc_full_cuda_fixed_sp_phase3b_double_scalar(
      expected_shadow_bytes, paste(context, "expected shadow bytes")
    )
  expected_release_count <-
    fastkpc_full_cuda_fixed_sp_phase3b_integer_scalar(
      expected_release_count, paste(context, "expected release count")
    )
  expected_coefficients <-
    fastkpc_full_cuda_fixed_sp_phase3b_logical_scalar(
      expected_coefficients, paste(context, "expected coefficients")
    )

  routes <- fastkpc_full_cuda_fixed_sp_phase3c_route_counts(
    info$planned_route, info$executed_route
  )
  same_route <- info$planned_route == info$executed_route
  cholesky_to_svd <- info$planned_route == "CHOLESKY_BATCHED" &
    info$executed_route == "AUGMENTED_SVD"
  qr_to_svd <- info$planned_route == "AUGMENTED_QR" &
    info$executed_route == "AUGMENTED_SVD"
  reroute_reason_exact <- all(
    (same_route & info$reroute_reason == "") |
      (cholesky_to_svd &
         info$reroute_reason == "CHOLESKY_NON_POSITIVE_PIVOT") |
      (qr_to_svd & info$reroute_reason == "QR_RANK_GUARD_REJECTED")
  )
  expected_status <- ifelse(
    info$executed_route == "CHOLESKY_BATCHED",
    ifelse(info$target_true_batched,
           "OK_CHOLESKY_BATCHED", "OK_CHOLESKY_SINGLE"),
    ifelse(info$executed_route == "AUGMENTED_QR",
           "OK_AUGMENTED_QR", "OK_AUGMENTED_SVD")
  )
  planned_cholesky <- info$planned_route == "CHOLESKY_BATCHED"
  planned_cholesky_count <- as.integer(sum(planned_cholesky))
  expected_true_batched <- planned_cholesky &
    info$executed_route == "CHOLESKY_BATCHED" &
    planned_cholesky_count >= 2L
  expected_subgroups <- as.integer(planned_cholesky_count >= 2L)
  expected_attempted <- if (planned_cholesky_count >= 2L) {
    planned_cholesky_count
  } else {
    0L
  }
  qr <- info$executed_route == "AUGMENTED_QR"
  svd <- info$executed_route == "AUGMENTED_SVD"
  expected_aggregate_factor_calls <- as.integer(svd)
  expected_aggregate_b_builds <- 2L * expected_aggregate_factor_calls
  aggregate_row_shapes_exact <- all(vapply(
    seq_len(target_count), function(index) {
      pivot <- aggregate_root_pivot[[index]]
      if (!svd[[index]]) {
        return(
          is.na(aggregate_root_rank[[index]]) &&
            is.integer(pivot) && identical(pivot, integer()) &&
            is.na(aggregate_dstop[[index]])
        )
      }
      !is.na(aggregate_root_rank[[index]]) &&
        aggregate_root_rank[[index]] >= 0L &&
        aggregate_root_rank[[index]] <= dto$null_dim &&
        is.integer(pivot) && length(pivot) == dto$null_dim &&
        identical(sort(pivot), seq_len(dto$null_dim)) &&
        is.finite(aggregate_dstop[[index]]) &&
        aggregate_dstop[[index]] >= 0
    }, logical(1L)
  ))
  aggregate_lifecycle_exact <-
    isTRUE(aggregate_row_shapes_exact) &&
    identical(
      info$aggregate_factor_call_count,
      expected_aggregate_factor_calls
    ) &&
    identical(info$aggregate_b_build_count, expected_aggregate_b_builds) &&
    info$aggregate_penalty_factor_count ==
      sum(expected_aggregate_factor_calls) &&
    info$aggregate_svd_b_build_count == sum(expected_aggregate_b_builds) &&
    info$aggregate_penalty_root_d2h_count == 0L &&
    info$aggregate_penalty_root_d2h_bytes == 0
  route_diagnostics_exact <-
    all(info$qr_rank[qr] == dto$null_dim) &&
    all(info$geqrf_info[qr] == 0L) && all(info$ormqr_info[qr] == 0L) &&
    all(info$effective_rank[svd] >= 0L) &&
    all(info$effective_rank[svd] <= dto$null_dim) &&
    all(info$svd_info[svd] == 0L) &&
    all(is.finite(info$sigma_max[svd])) && all(info$sigma_max[svd] > 0) &&
    all(is.finite(info$smallest_retained_sigma[svd])) &&
    all(info$smallest_retained_sigma[svd] > 0) &&
    all(info$smallest_retained_sigma[svd] <= info$sigma_max[svd])
  route_status_exact <-
    identical(info$target_keys, native$target_keys) &&
    identical(info$planned_route, native$planned_route) &&
    identical(info$solver_status, unname(expected_status)) &&
    identical(info$target_true_batched, expected_true_batched) &&
    all(startsWith(info$solver_status, "OK_")) &&
    isTRUE(reroute_reason_exact) && isTRUE(routes$conserved) &&
    isTRUE(route_diagnostics_exact) &&
    info$planned_cholesky_target_count == routes$planned_cholesky &&
    info$planned_qr_target_count == routes$planned_qr &&
    info$planned_svd_target_count == routes$planned_svd &&
    info$executed_cholesky_target_count == routes$executed_cholesky &&
    info$executed_qr_target_count == routes$executed_qr &&
    info$executed_svd_target_count == routes$executed_svd &&
    info$cholesky_to_svd_count == routes$cholesky_to_svd &&
    info$qr_to_svd_count == routes$qr_to_svd &&
    info$stable_reroute_count == routes$stable_reroute &&
    info$true_batched_subgroup_count == expected_subgroups &&
    info$true_batched_attempted_target_count == expected_attempted &&
    info$true_batched_target_count == sum(expected_true_batched) &&
    info$cholesky_single_target_count ==
      sum(info$solver_status == "OK_CHOLESKY_SINGLE") &&
    info$potrf_batched_call_count == expected_subgroups &&
    info$potrs_batched_call_count == expected_subgroups &&
    identical(
      info$true_batched_kernel,
      target_count >= 2L && all(expected_true_batched)
    ) && isTRUE(info$canonical_output_order_exact)

  allocation_delta <- info$resource_allocation_count_after_solve -
    info$resource_allocation_count_before_solve
  handle_delta <- info$resource_handle_create_count_after_solve -
    info$resource_handle_create_count_before_solve
  resource_exact <- isTRUE(info$resource_snapshot_captured) &&
    info$resource_instrumentation_version == 1L &&
    allocation_delta ==
      info$cuda_device_allocation_count_during_solve +
        info$cuda_host_allocation_count_during_solve &&
    handle_delta ==
      info$stream_create_count_during_solve +
        info$event_create_count_during_solve +
        info$cublas_handle_create_count_during_solve +
        info$cusolver_handle_create_count_during_solve &&
    all(c(
      info$cuda_device_allocation_count_during_solve,
      info$cuda_host_allocation_count_during_solve,
      info$stream_create_count_during_solve,
      info$event_create_count_during_solve,
      info$cublas_handle_create_count_during_solve,
      info$cusolver_handle_create_count_during_solve,
      info$per_target_allocation_count_after_warmup,
      info$per_target_handle_create_count
    ) == 0L)
  expected_batch_finalize_calls <- as.integer(
    routes$executed_cholesky > 0L
  )
  expected_coefficient_finalize_calls <- as.integer(
    expected_coefficients && routes$executed_cholesky > 0L
  )
  semantics_exact <- isTRUE(route_status_exact) && isTRUE(resource_exact) &&
    isTRUE(aggregate_lifecycle_exact) &&
    identical(info$n, dto$n) &&
    identical(info$coefficient_dim, dto$coefficient_dim) &&
    identical(info$target_count, target_count) &&
    isTRUE(info$native_batch_call) && info$batch_call_count == 1L &&
    info$output_slot_acquire_count == 1L &&
    info$output_slot_release_count == expected_release_count &&
    info$output_slot_busy_count == 0L &&
    info$stale_token_reject_count == 0L &&
    info$invalid_output_init_count == 1L &&
    info$nonfinite_output_count == 0L &&
    info$target_batch_h2d_call_count == 1L &&
    info$target_h2d_copy_count == 2L &&
    identical(
      info$target_h2d_bytes,
      8 * as.double(length(native$Y) + length(native$SP))
    ) && info$rhs_device_build_count == 1L &&
    identical(info$rhs_authority, "cuda-x0-transpose-y") &&
    isTRUE(info$full_cuda_data_plane) &&
    info$coefficient_batch_finalize_call_count ==
      expected_coefficient_finalize_calls &&
    info$fitted_batch_finalize_call_count ==
      expected_batch_finalize_calls &&
    info$residual_rss_batch_finalize_call_count ==
      expected_batch_finalize_calls &&
    info$per_target_output_finalize_call_count ==
      routes$executed_qr + routes$executed_svd &&
    info$batch_output_finalized_target_count == target_count &&
    info$implicit_residual_d2h_count == 0L &&
    info$cpu_fallback_count == 0L && info$unknown_fallback_count == 0L &&
    info$shadow_materialize_call_count == expected_shadow_calls &&
    info$shadow_materialize_target_count == expected_shadow_targets &&
    identical(info$shadow_d2h_bytes, expected_shadow_bytes)
  if (!isTRUE(semantics_exact)) {
    stop(context, " route/status/resource conservation failed",
         call. = FALSE)
  }
  list(routes = routes, route_status_conservation_exact = TRUE)
}

fastkpc_full_cuda_fixed_sp_phase3c_expected_counts <- function(scope) {
  scope <- match.arg(scope, c("iteration", "qualification"))
  if (identical(scope, "iteration")) {
    return(list(
      setup_count = 44L, target_count = 270L,
      penalty_root_matrix_count = 159L, penalty_root_row_count = 1424L,
      H_root_matrix_count = 0L,
      planned_cholesky_count = 172L, planned_qr_count = 31L,
      planned_svd_count = 67L, executed_cholesky_count = 172L,
      executed_qr_count = 31L, executed_svd_count = 67L,
      cholesky_to_svd_count = 0L, qr_to_svd_count = 0L,
      true_batched_target_count = 160L, cholesky_single_target_count = 12L,
      whole_batch_true_batched_count = 5L,
      stable_not_implemented_count = 0L, stable_reroute_count = 0L,
      non_ok_status_count = 0L, root_rank_mismatch_count = 0L,
      aggregate_penalty_factor_count = 67L,
      aggregate_svd_b_build_count = 134L,
      aggregate_penalty_root_rank_mismatch_count = 0L,
      aggregate_penalty_root_pivot_mismatch_count = 0L,
      aggregate_penalty_root_d2h_count = 0L,
      aggregate_penalty_root_d2h_bytes = 0,
      workspace_grow_count_after_warmup = 0L,
      stable_workspace_grow_count_after_warmup = 0L,
      per_target_allocation_count_after_warmup = 0L,
      per_target_handle_create_count = 0L,
      cuda_device_synchronize_count = 0L,
      target_level_stable_sync_count = 0L,
      implicit_residual_d2h_count = 0L,
      all_output_slot_leases_released = TRUE,
      invalid_output_init_count = 44L, cpu_fallback_count = 0L,
      unknown_fallback_count = 0L, approximate_backend_count = 0L
    ))
  }
  list(
    setup_count = 2061L, target_count = 6143L,
    penalty_root_matrix_count = 6272L, penalty_root_row_count = 63552L,
    H_root_matrix_count = 0L,
    planned_cholesky_count = 3889L, planned_qr_count = 190L,
    planned_svd_count = 2064L, executed_cholesky_count = 3889L,
    executed_qr_count = 190L, executed_svd_count = 2064L,
    cholesky_to_svd_count = 0L, qr_to_svd_count = 0L,
    svd_finite_high_count = 902L, svd_nonfinite_count = 1162L,
    all_safe_batch_count = 723L, mixed_batch_count = 652L,
    all_stable_batch_count = 686L, true_batched_subgroup_count = 806L,
    true_batched_target_count = 3320L,
    cholesky_single_target_count = 569L,
    whole_batch_true_batched_count = 723L,
    non_ok_status_count = 0L, stable_not_implemented_count = 0L,
    stable_reroute_count = 0L, root_rank_mismatch_count = 0L,
    aggregate_penalty_factor_count = 2064L,
    aggregate_svd_b_build_count = 4128L,
    aggregate_penalty_root_rank_mismatch_count = 0L,
    aggregate_penalty_root_pivot_mismatch_count = 0L,
    aggregate_penalty_root_d2h_count = 0L,
    aggregate_penalty_root_d2h_bytes = 0,
    workspace_grow_count_after_warmup = 0L,
    stable_workspace_grow_count_after_warmup = 0L,
    per_target_allocation_count_after_warmup = 0L,
    per_target_handle_create_count = 0L,
    cuda_device_synchronize_count = 0L,
    target_level_stable_sync_count = 0L,
    implicit_residual_d2h_count = 0L,
    all_output_slot_leases_released = TRUE,
    invalid_output_init_count = 2061L, cpu_fallback_count = 0L,
    unknown_fallback_count = 0L, approximate_backend_count = 0L
  )
}

fastkpc_full_cuda_fixed_sp_phase3c_summarize <- function(
    catalog_records, runtime_records, setup_records,
    batch_records, target_records, scope = "iteration") {
  scope <- match.arg(scope, c("iteration", "qualification"))
  records <- list(
    catalog = catalog_records, runtime = runtime_records,
    setup = setup_records, batch = batch_records, target = target_records
  )
  if (any(!vapply(records, is.data.frame, logical(1L))) ||
      nrow(catalog_records) != 1L || nrow(runtime_records) != 3L ||
      nrow(setup_records) < 1L || nrow(batch_records) < 1L ||
      nrow(target_records) < 1L) {
    stop("Phase 3C iteration records are malformed", call. = FALSE)
  }
  setup_keys <- setup_records$prepared_s_key_sha256
  if (!identical(batch_records$prepared_s_key_sha256, setup_keys)) {
    stop("Phase 3C setup/batch ownership is malformed", call. = FALSE)
  }
  target_indices <- lapply(setup_keys, function(setup_key) {
    which(target_records$prepared_s_key_sha256 == setup_key)
  })
  target_count_by_setup <- vapply(target_indices, length, integer(1L))
  if (any(target_count_by_setup < 1L) ||
      sum(target_count_by_setup) != nrow(target_records) ||
      !identical(batch_records$target_count, target_count_by_setup)) {
    stop("Phase 3C target ownership is malformed", call. = FALSE)
  }
  safe_count_by_setup <- vapply(target_indices, function(indices) {
    as.integer(sum(
      target_records$planned_route[indices] == "CHOLESKY_BATCHED"
    ))
  }, integer(1L))
  all_safe <- safe_count_by_setup == target_count_by_setup
  all_stable <- safe_count_by_setup == 0L
  mixed <- !all_safe & !all_stable
  routes <- fastkpc_full_cuda_fixed_sp_phase3c_route_counts(
    target_records$planned_route, target_records$executed_route
  )
  per_batch_routes <- lapply(target_indices, function(indices) {
    fastkpc_full_cuda_fixed_sp_phase3c_route_counts(
      target_records$planned_route[indices],
      target_records$executed_route[indices]
    )
  })
  if (!isTRUE(routes$conserved) ||
      !all(vapply(per_batch_routes, `[[`, logical(1L), "conserved"))) {
    stop("Phase 3C route conservation failed", call. = FALSE)
  }
  route_field_map <- c(
    planned_cholesky = "planned_cholesky_target_count",
    planned_qr = "planned_qr_target_count",
    planned_svd = "planned_svd_target_count",
    executed_cholesky = "executed_cholesky_target_count",
    executed_qr = "executed_qr_target_count",
    executed_svd = "executed_svd_target_count",
    cholesky_to_svd = "cholesky_to_svd_count",
    qr_to_svd = "qr_to_svd_count",
    stable_reroute = "stable_reroute_count"
  )
  batch_routes_exact <- all(vapply(seq_along(per_batch_routes), function(i) {
    all(vapply(names(route_field_map), function(metric) {
      identical(
        per_batch_routes[[i]][[metric]],
        batch_records[[route_field_map[[metric]]]][[i]]
      )
    }, logical(1L)))
  }, logical(1L)))
  if (!isTRUE(batch_routes_exact)) {
    stop("Phase 3C batch route counters are not derived from targets",
         call. = FALSE)
  }
  batch_aggregate_factor_counts <- vapply(
    target_indices, function(indices) {
      as.integer(sum(
        target_records$aggregate_factor_call_count[indices]
      ))
    }, integer(1L)
  )
  batch_aggregate_b_build_counts <- vapply(
    target_indices, function(indices) {
      as.integer(sum(target_records$aggregate_b_build_count[indices]))
    }, integer(1L)
  )
  batch_aggregate_exact <- identical(
    batch_records$aggregate_penalty_factor_count,
    batch_aggregate_factor_counts
  ) && identical(
    batch_records$aggregate_svd_b_build_count,
    batch_aggregate_b_build_counts
  ) && all(batch_records$aggregate_penalty_root_d2h_count == 0L) &&
    all(batch_records$aggregate_penalty_root_d2h_bytes == 0)
  if (!isTRUE(batch_aggregate_exact)) {
    stop("Phase 3C batch aggregate counters are not derived from targets",
         call. = FALSE)
  }

  reserved_index <- match("workspace-reserved", runtime_records$stage)
  final_index <- match("final", runtime_records$stage)
  if (is.na(reserved_index) || is.na(final_index)) {
    stop("Phase 3C runtime stages are malformed", call. = FALSE)
  }
  workspace_delta <- as.integer(
    runtime_records$workspace_grow_count[[final_index]] -
      runtime_records$workspace_grow_count[[reserved_index]]
  )
  stable_workspace_delta <- as.integer(
    runtime_records$stable_workspace_grow_count[[final_index]] -
      runtime_records$stable_workspace_grow_count[[reserved_index]]
  )
  synchronize_delta <- as.integer(
    runtime_records$cuda_device_synchronize_count[[final_index]] -
      runtime_records$cuda_device_synchronize_count[[reserved_index]]
  )
  qr_affected <- batch_records$planned_qr_target_count > 0L
  svd_affected <- batch_records$executed_svd_target_count > 0L
  checkpoints_bounded <-
    all(batch_records$qr_checkpoint_record_count >= 0L) &&
    all(batch_records$qr_checkpoint_wait_count >= 0L) &&
    all(batch_records$svd_checkpoint_record_count >= 0L) &&
    all(batch_records$svd_checkpoint_wait_count >= 0L) &&
    all(batch_records$qr_checkpoint_record_count <= as.integer(qr_affected)) &&
    all(batch_records$qr_checkpoint_wait_count <= as.integer(qr_affected)) &&
    all(batch_records$svd_checkpoint_record_count <= as.integer(svd_affected)) &&
    all(batch_records$svd_checkpoint_wait_count <= as.integer(svd_affected))
  if (!isTRUE(checkpoints_bounded)) {
    stop("Phase 3C stable checkpoint waits scaled beyond public batches",
         call. = FALSE)
  }
  target_level_stable_sync_count <- as.integer(sum(
    pmax(
      batch_records$qr_checkpoint_wait_count - as.integer(qr_affected), 0L
    ) + pmax(
      batch_records$svd_checkpoint_wait_count - as.integer(svd_affected), 0L
    )
  ))
  whole_batch_true_batched_count <- as.integer(sum(vapply(
    target_indices, function(indices) {
      length(indices) >= 2L && all(target_records$target_true_batched[indices])
    }, logical(1L)
  )))
  error_fields <- c(
    "residual_max_abs_diff", "residual_relative_l2_diff",
    "fitted_max_abs_diff", "fitted_relative_l2_diff"
  )
  errors_exact <- all(vapply(error_fields, function(field) {
    value <- target_records[[field]]
    is.double(value) && length(value) == nrow(target_records) &&
      all(is.finite(value)) && all(value >= 0) && all(value < 1e-7)
  }, logical(1L)))
  svd <- target_records$executed_route == "AUGMENTED_SVD"
  expected_aggregate_factor_calls <- as.integer(svd)
  expected_aggregate_b_builds <- 2L * expected_aggregate_factor_calls
  aggregate_exact <-
    identical(
      target_records$aggregate_factor_call_count,
      expected_aggregate_factor_calls
    ) &&
    identical(
      target_records$aggregate_b_build_count,
      expected_aggregate_b_builds
    ) &&
    all(target_records$aggregate_penalty_root_rank_exact[svd]) &&
    all(target_records$aggregate_penalty_root_pivot_exact[svd]) &&
    all(target_records$aggregate_effective_rank_exact[svd]) &&
    identical(
      target_records$effective_rank[svd],
      target_records$cpu_aggregate_effective_rank[svd]
    )
  target_evidence_exact <- errors_exact &&
    all(target_records$outputs_all_finite) &&
    all(startsWith(target_records$solver_status, "OK_")) &&
    all(target_records$numeric_reference == "mgcv-fixed-sp") &&
    identical(
      target_records$oracle_call_count,
      rep(1L, nrow(target_records))
    ) &&
    all(target_records$oracle_fitted_hash_exact) &&
    all(target_records$oracle_residual_hash_exact) &&
    !any(target_records$approximate_backend) && isTRUE(aggregate_exact)
  if (!isTRUE(target_evidence_exact)) {
    stop("Phase 3C target numerical/rank evidence failed", call. = FALSE)
  }
  route_status_hash <- fastkpc_full_cuda_census_metadata_hash(list(
    target_records$residual_key_sha256,
    target_records$planned_route,
    target_records$executed_route,
    target_records$reroute_reason,
    target_records$solver_status
  ))
  numeric_hash <- fastkpc_full_cuda_census_metadata_hash(list(
    target_records$residual_key_sha256,
    target_records$fitted_numeric_hash,
    target_records$residual_numeric_hash
  ))
  summary <- list(
    scope = scope,
    scope_subset_hash = catalog_records$scope_subset_hash[[1L]],
    ordered_setup_key_digest =
      catalog_records$ordered_setup_key_digest[[1L]],
    ordered_target_key_digest =
      catalog_records$ordered_target_key_digest[[1L]],
    route_status_hash = route_status_hash,
    numeric_hash = numeric_hash,
    setup_count = as.integer(nrow(setup_records)),
    target_count = as.integer(nrow(target_records)),
    penalty_root_matrix_count = as.integer(sum(
      setup_records$penalty_root_matrix_count
    )),
    penalty_root_row_count = as.integer(sum(
      setup_records$penalty_root_row_count
    )),
    H_root_matrix_count = as.integer(sum(
      setup_records$H_root_matrix_count
    )),
    planned_cholesky_count = routes$planned_cholesky,
    planned_qr_count = routes$planned_qr,
    planned_svd_count = routes$planned_svd,
    executed_cholesky_count = routes$executed_cholesky,
    executed_qr_count = routes$executed_qr,
    executed_svd_count = routes$executed_svd,
    cholesky_to_svd_count = routes$cholesky_to_svd,
    qr_to_svd_count = routes$qr_to_svd,
    svd_finite_high_count = as.integer(sum(
      svd & is.finite(target_records$phase1_condition)
    )),
    svd_nonfinite_count = as.integer(sum(
      svd & !is.finite(target_records$phase1_condition)
    )),
    all_safe_batch_count = as.integer(sum(all_safe)),
    mixed_batch_count = as.integer(sum(mixed)),
    all_stable_batch_count = as.integer(sum(all_stable)),
    true_batched_subgroup_count = as.integer(sum(
      batch_records$true_batched_subgroup_count
    )),
    true_batched_target_count = as.integer(sum(
      target_records$target_true_batched
    )),
    cholesky_single_target_count = as.integer(sum(
      target_records$solver_status == "OK_CHOLESKY_SINGLE"
    )),
    whole_batch_true_batched_count = whole_batch_true_batched_count,
    stable_not_implemented_count = as.integer(sum(
      target_records$solver_status == "ERR_STABLE_PATH_NOT_IMPLEMENTED"
    )),
    stable_reroute_count = routes$stable_reroute,
    non_ok_status_count = as.integer(sum(
      !startsWith(target_records$solver_status, "OK_")
    )),
    root_rank_mismatch_count = as.integer(sum(
      setup_records$penalty_root_rank_mismatch_count
    )),
    aggregate_penalty_factor_count = as.integer(sum(
      target_records$aggregate_factor_call_count
    )),
    aggregate_svd_b_build_count = as.integer(sum(
      target_records$aggregate_b_build_count
    )),
    aggregate_penalty_root_rank_mismatch_count = as.integer(sum(
      !target_records$aggregate_penalty_root_rank_exact[svd]
    )),
    aggregate_penalty_root_pivot_mismatch_count = as.integer(sum(
      !target_records$aggregate_penalty_root_pivot_exact[svd]
    )),
    aggregate_penalty_root_d2h_count = as.integer(sum(
      batch_records$aggregate_penalty_root_d2h_count
    )),
    aggregate_penalty_root_d2h_bytes = as.double(sum(
      batch_records$aggregate_penalty_root_d2h_bytes
    )),
    workspace_grow_count_after_warmup = workspace_delta,
    stable_workspace_grow_count_after_warmup = stable_workspace_delta,
    per_target_allocation_count_after_warmup = as.integer(sum(
      batch_records$per_target_allocation_count_after_warmup
    )),
    per_target_handle_create_count = as.integer(sum(
      batch_records$per_target_handle_create_count
    )),
    cuda_device_synchronize_count = synchronize_delta,
    target_level_stable_sync_count = target_level_stable_sync_count,
    implicit_residual_d2h_count = as.integer(sum(
      batch_records$implicit_residual_d2h_count
    )),
    all_output_slot_leases_released = all(
      setup_records$output_slot_state_after_release == "free" &
        !setup_records$output_slot_leased_after_release &
        batch_records$output_slot_release_count == 1L &
        !batch_records$output_slot_leased_after_release
    ),
    invalid_output_init_count = as.integer(sum(
      batch_records$invalid_output_init_count
    )),
    cpu_fallback_count = as.integer(sum(batch_records$cpu_fallback_count)),
    unknown_fallback_count = as.integer(sum(
      batch_records$unknown_fallback_count
    )),
    approximate_backend_count = as.integer(sum(
      target_records$approximate_backend
    )),
    qr_checkpoint_record_count = as.integer(sum(
      batch_records$qr_checkpoint_record_count
    )),
    qr_checkpoint_wait_count = as.integer(sum(
      batch_records$qr_checkpoint_wait_count
    )),
    svd_checkpoint_record_count = as.integer(sum(
      batch_records$svd_checkpoint_record_count
    )),
    svd_checkpoint_wait_count = as.integer(sum(
      batch_records$svd_checkpoint_wait_count
    )),
    max_residual_abs_diff = max(target_records$residual_max_abs_diff),
    max_residual_relative_l2_diff =
      max(target_records$residual_relative_l2_diff),
    max_fitted_abs_diff = max(target_records$fitted_max_abs_diff),
    max_fitted_relative_l2_diff =
      max(target_records$fitted_relative_l2_diff)
  )
  subset_name <- paste0(scope, "_subset_hash")
  summary[[subset_name]] <- summary$scope_subset_hash
  expected <- fastkpc_full_cuda_fixed_sp_phase3c_expected_counts(scope)
  if (!identical(summary[names(expected)], expected)) {
    stop("Phase 3C ", scope, " exact aggregate gate failed", call. = FALSE)
  }
  summary
}

fastkpc_full_cuda_fixed_sp_residual_registry_state <- function(registry) {
  state <- attr(
    registry, "fastkpc_fixed_sp_residual_registry_state", exact = TRUE
  )
  if (!is.environment(registry) || !is.environment(state) ||
      !identical(parent.env(registry), emptyenv()) ||
      !identical(parent.env(state), emptyenv()) ||
      !identical(state$schema_version,
                 "full-cuda-ci-fixed-sp-residual-registry-v1") ||
      !is.logical(state$cleared) || length(state$cleared) != 1L ||
      is.na(state$cleared)) {
    stop("qualification residual registry is malformed", call. = FALSE)
  }
  keys_clean <- typeof(state$expected_keys) == "character" &&
    !is.object(state$expected_keys) &&
    is.null(attributes(state$expected_keys)) &&
    !anyNA(state$expected_keys) && !anyDuplicated(state$expected_keys) &&
    all(grepl("^[0-9a-f]{64}$", state$expected_keys))
  owners_clean <- typeof(state$expected_owners) == "character" &&
    !is.object(state$expected_owners) &&
    is.null(attributes(state$expected_owners)) &&
    length(state$expected_owners) == length(state$expected_keys) &&
    !anyNA(state$expected_owners) &&
    all(grepl("^[0-9a-f]{64}$", state$expected_owners))
  hashes_clean <- typeof(state$captured_hashes) == "character" &&
    !is.object(state$captured_hashes) &&
    is.null(attributes(state$captured_hashes)) &&
    length(state$captured_hashes) == length(state$expected_keys) &&
    all(is.na(state$captured_hashes) |
          grepl("^[0-9a-f]{64}$", state$captured_hashes))
  counters_clean <- typeof(state$n) == "integer" &&
    length(state$n) == 1L && !is.object(state$n) &&
    is.null(attributes(state$n)) && !is.na(state$n) &&
    typeof(state$next_index) == "integer" &&
    length(state$next_index) == 1L && !is.object(state$next_index) &&
    is.null(attributes(state$next_index)) && !is.na(state$next_index)
  if (!isTRUE(keys_clean) || !isTRUE(owners_clean) ||
      !isTRUE(hashes_clean) || !isTRUE(counters_clean)) {
    stop("qualification residual registry is malformed", call. = FALSE)
  }
  if (isTRUE(state$cleared)) {
    cleared_exact <- identical(state$expected_keys, character()) &&
      identical(state$expected_owners, character()) &&
      identical(state$captured_hashes, character()) &&
      identical(state$n, 0L) && identical(state$next_index, 0L)
    if (!isTRUE(cleared_exact)) {
      stop("qualification residual registry is malformed", call. = FALSE)
    }
    return(state)
  }
  expected_count <- length(state$expected_keys)
  captured_count <- state$next_index - 1L
  captured_prefix_exact <- captured_count == 0L || all(!is.na(
    state$captured_hashes[seq_len(captured_count)]
  ))
  uncaptured_suffix_exact <- state$next_index > expected_count || all(is.na(
    state$captured_hashes[state$next_index:expected_count]
  ))
  active_exact <- expected_count > 0L && state$n > 0L &&
    state$next_index >= 1L &&
    state$next_index <= expected_count + 1L &&
    identical(
      order(state$expected_owners, state$expected_keys, method = "radix"),
      seq_along(state$expected_keys)
    ) && captured_prefix_exact && uncaptured_suffix_exact
  if (!isTRUE(active_exact)) {
    stop("qualification residual registry is malformed", call. = FALSE)
  }
  state
}

fastkpc_full_cuda_fixed_sp_residual_registry_create <- function(
    expected_keys, expected_owners, n) {
  keys_clean <- typeof(expected_keys) == "character" &&
    !is.object(expected_keys) && is.null(attributes(expected_keys)) &&
    length(expected_keys) > 0L && !anyNA(expected_keys) &&
    !anyDuplicated(expected_keys) &&
    all(grepl("^[0-9a-f]{64}$", expected_keys))
  owners_clean <- typeof(expected_owners) == "character" &&
    !is.object(expected_owners) && is.null(attributes(expected_owners)) &&
    length(expected_owners) == length(expected_keys) &&
    !anyNA(expected_owners) &&
    all(grepl("^[0-9a-f]{64}$", expected_owners))
  n_clean <- typeof(n) == "integer" && length(n) == 1L &&
    !is.object(n) && is.null(attributes(n)) && !is.na(n) && n > 0L
  order_clean <- isTRUE(keys_clean) && isTRUE(owners_clean) && identical(
    order(expected_owners, expected_keys, method = "radix"),
    seq_along(expected_keys)
  )
  if (!isTRUE(keys_clean) || !isTRUE(owners_clean) || !isTRUE(n_clean) ||
      !isTRUE(order_clean)) {
    stop("qualification residual registry corpus is malformed",
         call. = FALSE)
  }
  registry <- new.env(hash = TRUE, parent = emptyenv())
  state <- new.env(hash = FALSE, parent = emptyenv())
  state$schema_version <- "full-cuda-ci-fixed-sp-residual-registry-v1"
  state$expected_keys <- expected_keys
  state$expected_owners <- expected_owners
  state$captured_hashes <- rep(NA_character_, length(expected_keys))
  state$n <- n
  state$next_index <- 1L
  state$cleared <- FALSE
  attr(registry, "fastkpc_fixed_sp_residual_registry_state") <- state
  registry
}

fastkpc_full_cuda_fixed_sp_residual_registry_capture <- function(
    registry, owner, target_keys, residuals) {
  state <- fastkpc_full_cuda_fixed_sp_residual_registry_state(registry)
  if (isTRUE(state$cleared)) {
    stop("qualification residual registry was cleared", call. = FALSE)
  }
  owner_clean <- typeof(owner) == "character" && length(owner) == 1L &&
    !is.object(owner) && is.null(attributes(owner)) && !anyNA(owner) &&
    grepl("^[0-9a-f]{64}$", owner)
  keys_clean <- typeof(target_keys) == "character" &&
    !is.object(target_keys) && is.null(attributes(target_keys)) &&
    length(target_keys) > 0L && !anyNA(target_keys) &&
    !anyDuplicated(target_keys) &&
    all(grepl("^[0-9a-f]{64}$", target_keys))
  matrix_clean <- is.matrix(residuals) && is.double(residuals) &&
    !is.object(residuals) && identical(
      dim(residuals), c(state$n, as.integer(length(target_keys)))
    )
  if (!isTRUE(owner_clean) || !isTRUE(keys_clean)) {
    stop("qualification residual registry capture is malformed",
         call. = FALSE)
  }
  if (!isTRUE(matrix_clean)) {
    stop("qualification residual length or shape is invalid",
         call. = FALSE)
  }
  if (any(!is.finite(residuals))) {
    stop("qualification residual registry received nonfinite values",
         call. = FALSE)
  }
  duplicate <- target_keys[vapply(target_keys, exists, logical(1L),
                                  envir = registry, inherits = FALSE)]
  if (length(duplicate) > 0L) {
    stop("qualification residual registry duplicate residual key: ",
         duplicate[[1L]], call. = FALSE)
  }
  positions <- match(target_keys, state$expected_keys)
  if (anyNA(positions)) {
    stop("qualification residual registry unexpected residual key: ",
         target_keys[[which(is.na(positions))[[1L]]]], call. = FALSE)
  }
  if (any(state$expected_owners[positions] != owner)) {
    stop("qualification residual owner does not match authenticated owner",
         call. = FALSE)
  }
  end_index <- state$next_index + length(target_keys) - 1L
  if (end_index > length(state$expected_keys) || !identical(
        target_keys,
        state$expected_keys[state$next_index:end_index]
      )) {
    stop("qualification canonical residual order mismatch", call. = FALSE)
  }
  values <- lapply(seq_along(target_keys), function(column) {
    as.numeric(residuals[, column])
  })
  hashes <- vapply(
    values, fastkpc_full_cuda_census_metadata_hash, character(1L)
  )
  for (column in seq_along(target_keys)) {
    key <- target_keys[[column]]
    assign(key, values[[column]], envir = registry)
    lockBinding(key, registry)
    state$captured_hashes[[state$next_index + column - 1L]] <-
      hashes[[column]]
  }
  state$next_index <- as.integer(end_index + 1L)
  invisible(registry)
}

fastkpc_full_cuda_fixed_sp_residual_registry_validate <- function(
    registry, target_records = NULL) {
  state <- fastkpc_full_cuda_fixed_sp_residual_registry_state(registry)
  if (isTRUE(state$cleared)) {
    stop("qualification residual registry was cleared", call. = FALSE)
  }
  actual <- ls(registry, all.names = TRUE, sorted = TRUE)
  expected_sorted <- sort(state$expected_keys, method = "radix")
  extra <- setdiff(actual, expected_sorted)
  if (length(extra) > 0L) {
    stop("qualification residual registry has extra vectors: ",
         extra[[1L]], call. = FALSE)
  }
  missing <- setdiff(expected_sorted, actual)
  if (length(missing) > 0L ||
      state$next_index != length(state$expected_keys) + 1L) {
    stop("qualification residual registry has missing residual vectors",
         call. = FALSE)
  }
  valid <- all(vapply(state$expected_keys, function(key) {
    value <- get(key, envir = registry, inherits = FALSE)
    is.double(value) && length(value) == state$n &&
      !is.object(value) && is.null(attributes(value)) &&
      all(is.finite(value))
  }, logical(1L)))
  if (!isTRUE(valid)) {
    stop("qualification residual registry contains an invalid vector",
         call. = FALSE)
  }
  actual_hashes <- vapply(state$expected_keys, function(key) {
    fastkpc_full_cuda_census_metadata_hash(
      get(key, envir = registry, inherits = FALSE)
    )
  }, character(1L))
  names(actual_hashes) <- NULL
  if (!identical(actual_hashes, state$captured_hashes)) {
    stop("qualification residual registry residual hash mismatch",
         call. = FALSE)
  }
  bindings_locked <- all(vapply(
    state$expected_keys, bindingIsLocked, logical(1L), env = registry
  ))
  if (!isTRUE(bindings_locked)) {
    stop("qualification residual registry binding is unlocked",
         call. = FALSE)
  }
  if (!is.null(target_records)) {
    required_fields <- c("residual_key_sha256", "residual_numeric_hash")
    target_count <- length(state$expected_keys)
    target_clean <- is.data.frame(target_records) &&
      nrow(target_records) == target_count &&
      all(required_fields %in% names(target_records)) &&
      all(vapply(required_fields, function(field) {
        value <- target_records[[field]]
        typeof(value) == "character" && length(value) == target_count &&
          !is.object(value) && is.null(attributes(value)) &&
          !anyNA(value) && all(grepl("^[0-9a-f]{64}$", value))
      }, logical(1L))) &&
      !anyDuplicated(target_records$residual_key_sha256)
    if (!isTRUE(target_clean)) {
      stop("qualification Task 7 residual hash authority is malformed",
           call. = FALSE)
    }
    if (!identical(
          target_records$residual_key_sha256, state$expected_keys
        )) {
      stop("qualification Task 7 residual key corpus mismatch",
           call. = FALSE)
    }
    if (!identical(
          target_records$residual_numeric_hash, state$captured_hashes
        )) {
      stop("qualification Task 7 residual numeric hash mismatch",
           call. = FALSE)
    }
  }
  state$expected_keys
}

fastkpc_full_cuda_fixed_sp_residual_registry_clear <- function(registry) {
  state <- fastkpc_full_cuda_fixed_sp_residual_registry_state(registry)
  entries <- ls(registry, all.names = TRUE, sorted = FALSE)
  for (key in entries) {
    if (bindingIsLocked(key, registry)) unlockBinding(key, registry)
  }
  if (length(entries) > 0L) rm(list = entries, envir = registry)
  state$expected_keys <- character()
  state$expected_owners <- character()
  state$captured_hashes <- character()
  state$n <- 0L
  state$next_index <- 0L
  state$cleared <- TRUE
  invisible(TRUE)
}

fastkpc_full_cuda_fixed_sp_with_legacy_dcov_backend <- function(fun) {
  if (!is.function(fun)) {
    stop("qualification dCov backend callback is invalid", call. = FALSE)
  }
  variables <- c(
    "FASTKPC_LEGACY_DCOV_GAMMA_BACKEND",
    "FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK"
  )
  prior <- Sys.getenv(variables, unset = NA_character_)
  on.exit({
    for (index in seq_along(variables)) {
      name <- variables[[index]]
      value <- prior[[index]]
      if (is.na(value)) {
        Sys.unsetenv(name)
      } else {
        do.call(Sys.setenv, setNames(list(value), name))
      }
    }
  }, add = TRUE)
  Sys.setenv(
    FASTKPC_LEGACY_DCOV_GAMMA_BACKEND = "cpp",
    FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = "spectra"
  )
  fun()
}

fastkpc_full_cuda_fixed_sp_qualification_dcov_core <- function() {
  core_name <- ".fastkpc_full_cuda_run_prepared_s_dcov_parity_core"
  if (!exists(core_name, mode = "function", inherits = TRUE)) {
    stop("qualified Prepared-S dCov core is unavailable", call. = FALSE)
  }
  core <- get(core_name, mode = "function", inherits = TRUE)
  expected_formals <- c(
    "inputs", "logical_tests", "residuals", "oracle_manifest",
    "oracle_fun", "scope"
  )
  if (!identical(names(formals(core)), expected_formals)) {
    stop("qualified Prepared-S dCov core interface changed",
         call. = FALSE)
  }
  numeric_replacements <- 0L
  message_replacements <- 0L
  old_message <- "Prepared-S dCov p-value drift exceeds 1e-12: "
  new_message <- "Prepared-S dCov p-value drift exceeds 1e-10: "
  patch_node <- function(node) {
    if (is.double(node) && length(node) == 1L &&
        !is.object(node) && is.null(attributes(node)) &&
        identical(node, 1e-12)) {
      numeric_replacements <<- numeric_replacements + 1L
      return(1e-10)
    }
    if (is.character(node) && length(node) == 1L &&
        !is.object(node) && is.null(attributes(node)) &&
        identical(node, old_message)) {
      message_replacements <<- message_replacements + 1L
      return(new_message)
    }
    if (is.call(node)) {
      result <- node
      for (index in seq_along(node)) {
        result[index] <- list(patch_node(node[[index]]))
      }
      return(result)
    }
    if (is.pairlist(node)) {
      return(as.pairlist(lapply(node, patch_node)))
    }
    if (is.expression(node)) {
      return(as.expression(lapply(node, patch_node)))
    }
    node
  }
  patched_body <- patch_node(body(core))
  if (numeric_replacements != 1L || message_replacements != 1L) {
    stop("qualified Prepared-S dCov core tolerance shape changed",
         call. = FALSE)
  }
  patched <- core
  body(patched) <- patched_body
  if (!identical(formals(patched), formals(core)) ||
      !identical(environment(patched), environment(core))) {
    stop("qualified Prepared-S dCov core clone is malformed",
         call. = FALSE)
  }
  patched
}

fastkpc_full_cuda_fixed_sp_qualification_dcov_diagnostic_schema <- function() {
  names <- c(
    "n", "numCol", "index", "lowrank_mode",
    "lowrank_full_eig_count", "lowrank_spectra_count",
    "lowrank_spectra_converged_count", "lowrank_spectra_failed_count",
    "lowrank_spectra_fallback_full_eig_count",
    "lowrank_spectra_iterations", "lowrank_spectra_nconv",
    "lowrank_spectra_ncv", "lowrank_spectra_tol",
    "lowrank_spectra_matvec_count"
  )
  types <- setNames(rep.int("integer", length(names)), names)
  types[c("index", "lowrank_spectra_tol")] <- "double"
  types[["lowrank_mode"]] <- "character"
  list(names = names, types = types)
}

fastkpc_full_cuda_fixed_sp_qualification_dcov_schema <- function() {
  logical_names <- c(
    "logical_sequence_id", "source_sequence_id", "source_task_index",
    "level", "x", "y", "S_key", "S_size", "formula_class",
    "reference_p_value", "alpha", "reference_decision",
    "reference_independent", "deletes_edge", "selected_sepset",
    "signed_distance_from_alpha", "absolute_distance_from_alpha",
    "signed_log_ratio_from_alpha", "absolute_log_distance_from_alpha",
    "residual_key_x", "residual_key_y", "near_alpha",
    "selection_reasons"
  )
  names <- c(
    "parity_scope", logical_names, "index", "numCol", "backend",
    "low_rank_backend", "p_value", "p_value_difference",
    "absolute_p_value_difference", "p_value_exact",
    "signed_alpha_distance", "decision", "independent",
    "decision_flip", "backend_error", "spectra_fallback", "diagnostics"
  )
  integer <- c(
    "logical_sequence_id", "source_sequence_id", "source_task_index",
    "level", "x", "y", "S_size", "index", "numCol"
  )
  double <- c(
    "reference_p_value", "alpha", "signed_distance_from_alpha",
    "absolute_distance_from_alpha", "signed_log_ratio_from_alpha",
    "absolute_log_distance_from_alpha", "p_value",
    "p_value_difference", "absolute_p_value_difference",
    "signed_alpha_distance"
  )
  logical <- c(
    "reference_independent", "deletes_edge", "selected_sepset",
    "near_alpha", "p_value_exact", "independent", "decision_flip",
    "backend_error", "spectra_fallback"
  )
  list_fields <- c("selection_reasons", "diagnostics")
  character <- setdiff(names, c(integer, double, logical, list_fields))
  types <- setNames(rep.int(NA_character_, length(names)), names)
  types[integer] <- "integer"
  types[double] <- "double"
  types[logical] <- "logical"
  types[list_fields] <- "list"
  types[character] <- "character"
  if (anyNA(types) || anyDuplicated(names) ||
      !identical(names(types), names)) {
    stop("internal qualification dCov schema is malformed", call. = FALSE)
  }
  list(
    names = names, types = types, list_fields = list_fields,
    logical_names = logical_names
  )
}

fastkpc_full_cuda_fixed_sp_sanitize_dcov_diagnostic <- function(value) {
  schema <-
    fastkpc_full_cuda_fixed_sp_qualification_dcov_diagnostic_schema()
  if (!is.list(value) || is.object(value) ||
      length(setdiff(schema$names, names(value))) > 0L) {
    stop("qualification dCov diagnostic is malformed", call. = FALSE)
  }
  fastkpc_full_cuda_prepared_s_validate_dcov_spectra_diagnostics(
    value, numCol = 35L
  )
  result <- value[schema$names]
  clean <- identical(names(result), schema$names) &&
    all(vapply(schema$names, function(field) {
      field_value <- result[[field]]
      typeof(field_value) == schema$types[[field]] &&
        length(field_value) == 1L && !is.object(field_value) &&
        is.null(attributes(field_value)) && !anyNA(field_value) &&
        if (typeof(field_value) %in% c("integer", "double")) {
          is.finite(field_value)
        } else {
          nzchar(field_value)
        }
    }, logical(1L))) && identical(result$n, 351L) &&
    identical(result$numCol, 35L) && identical(result$index, 1) &&
    identical(result$lowrank_mode, "spectra")
  if (!isTRUE(clean)) {
    stop("qualification dCov diagnostic is malformed", call. = FALSE)
  }
  result
}

fastkpc_full_cuda_fixed_sp_validate_qualification_dcov_frame <- function(
    value) {
  schema <- fastkpc_full_cuda_fixed_sp_qualification_dcov_schema()
  if (!is.data.frame(value) || nrow(value) < 1L ||
      !identical(names(value), schema$names)) {
    stop("qualification dCov row schema mismatch", call. = FALSE)
  }
  clean <- all(vapply(schema$names, function(field) {
    column <- value[[field]]
    if (field %in% schema$list_fields) {
      typeof(column) == "list" && length(column) == nrow(value) &&
        is.object(column) &&
        identical(attributes(column), list(class = "AsIs"))
    } else {
      typeof(column) == schema$types[[field]] &&
        length(column) == nrow(value) && !is.object(column) &&
        is.null(attributes(column)) && !anyNA(column)
    }
  }, logical(1L))) && all(vapply(
    value$selection_reasons,
    function(element) {
      typeof(element) == "character" && !is.object(element) &&
        is.null(attributes(element)) && !anyNA(element)
    }, logical(1L)
  )) && all(vapply(
    value$diagnostics,
    function(element) isTRUE(tryCatch({
      identical(
        fastkpc_full_cuda_fixed_sp_sanitize_dcov_diagnostic(element),
        element
      )
    }, error = function(error) FALSE)),
    logical(1L)
  ))
  if (!isTRUE(clean)) {
    stop("qualification dCov row schema mismatch", call. = FALSE)
  }
  TRUE
}

fastkpc_full_cuda_fixed_sp_build_qualification_dcov_records <- function(
    logical_tests, parity_rows) {
  schema <- fastkpc_full_cuda_fixed_sp_qualification_dcov_schema()
  parity_names <- c(
    "parity_scope", "logical_sequence_id", "residual_key_x",
    "residual_key_y", "index", "numCol", "alpha",
    "reference_p_value", "p_value", "p_value_drift",
    "absolute_p_value_drift", "p_value_exact",
    "reference_signed_alpha_distance", "signed_alpha_distance",
    "reference_decision", "reference_independent", "decision",
    "decision_identical", "spectra_no_fallback", "diagnostics"
  )
  if (!is.data.frame(logical_tests) || nrow(logical_tests) < 1L ||
      !identical(names(logical_tests), schema$logical_names) ||
      !is.data.frame(parity_rows) ||
      nrow(parity_rows) != nrow(logical_tests) ||
      !identical(names(parity_rows), parity_names)) {
    stop("qualification dCov build inputs are malformed", call. = FALSE)
  }
  logical_ids <- logical_tests$logical_sequence_id
  logical_clean <- typeof(logical_ids) == "integer" &&
    !anyNA(logical_ids) && !anyDuplicated(logical_ids) &&
    identical(logical_ids, sort(logical_ids, method = "radix")) &&
    all(grepl("^[0-9a-f]{64}$", logical_tests$residual_key_x)) &&
    all(grepl("^[0-9a-f]{64}$", logical_tests$residual_key_y)) &&
    identical(parity_rows$parity_scope,
              rep("qualification", nrow(logical_tests))) &&
    identical(parity_rows$logical_sequence_id, logical_ids) &&
    identical(parity_rows$residual_key_x,
              logical_tests$residual_key_x) &&
    identical(parity_rows$residual_key_y,
              logical_tests$residual_key_y) &&
    identical(parity_rows$alpha, logical_tests$alpha) &&
    identical(parity_rows$reference_p_value,
              logical_tests$reference_p_value) &&
    identical(parity_rows$reference_decision,
              logical_tests$reference_decision) &&
    identical(parity_rows$reference_independent,
              logical_tests$reference_independent) &&
    identical(parity_rows$index, rep(1L, nrow(logical_tests))) &&
    identical(parity_rows$numCol, rep(35L, nrow(logical_tests)))
  if (!isTRUE(logical_clean)) {
    stop("qualification dCov canonical lineage mismatch", call. = FALSE)
  }
  diagnostics <- lapply(
    parity_rows$diagnostics,
    fastkpc_full_cuda_fixed_sp_sanitize_dcov_diagnostic
  )
  p_value <- as.double(parity_rows$p_value)
  difference <- p_value - logical_tests$reference_p_value
  independent <- p_value >= logical_tests$alpha
  decision <- ifelse(independent, "independent", "dependent")
  decision_flip <- independent != logical_tests$reference_independent
  values <- c(
    list(parity_scope = rep("qualification", nrow(logical_tests))),
    unclass(logical_tests),
    list(
      index = rep(1L, nrow(logical_tests)),
      numCol = rep(35L, nrow(logical_tests)),
      backend = rep("cpp", nrow(logical_tests)),
      low_rank_backend = rep("spectra", nrow(logical_tests)),
      p_value = p_value,
      p_value_difference = difference,
      absolute_p_value_difference = abs(difference),
      p_value_exact = vapply(seq_along(p_value), function(index) {
        identical(
          p_value[[index]], logical_tests$reference_p_value[[index]]
        )
      }, logical(1L)),
      signed_alpha_distance = p_value - logical_tests$alpha,
      decision = decision,
      independent = independent,
      decision_flip = decision_flip,
      backend_error = rep(FALSE, nrow(logical_tests)),
      spectra_fallback = !as.logical(parity_rows$spectra_no_fallback),
      diagnostics = I(diagnostics)
    )
  )
  records <- structure(
    values[schema$names], class = "data.frame",
    row.names = .set_row_names(nrow(logical_tests))
  )
  fastkpc_full_cuda_fixed_sp_validate_qualification_dcov_frame(records)
  records
}

fastkpc_full_cuda_fixed_sp_summarize_qualification_dcov_records <- function(
    records, logical_tests, target_keys) {
  schema <- fastkpc_full_cuda_fixed_sp_qualification_dcov_schema()
  fastkpc_full_cuda_fixed_sp_validate_qualification_dcov_frame(records)
  target_clean <- typeof(target_keys) == "character" &&
    !is.object(target_keys) && is.null(attributes(target_keys)) &&
    length(target_keys) > 0L && !anyNA(target_keys) &&
    !anyDuplicated(target_keys) &&
    all(grepl("^[0-9a-f]{64}$", target_keys))
  if (!isTRUE(target_clean) || !is.data.frame(logical_tests) ||
      !identical(names(logical_tests), schema$logical_names) ||
      nrow(logical_tests) != nrow(records)) {
    stop("qualification dCov summary inputs are malformed", call. = FALSE)
  }
  canonical <- records[schema$logical_names]
  canonical_exact <- identical(names(canonical), names(logical_tests)) &&
    all(vapply(names(canonical), function(field) {
      identical(canonical[[field]], logical_tests[[field]])
    }, logical(1L)))
  endpoints <- sort(unique(c(
    logical_tests$residual_key_x, logical_tests$residual_key_y
  )), method = "radix")
  target_set <- sort(target_keys, method = "radix")
  p_value <- records$p_value
  difference <- p_value - logical_tests$reference_p_value
  independent <- p_value >= logical_tests$alpha
  decision <- ifelse(independent, "independent", "dependent")
  flip <- independent != logical_tests$reference_independent
  fallback <- vapply(records$diagnostics, function(diagnostic) {
    diagnostic$lowrank_spectra_failed_count != 0L ||
      diagnostic$lowrank_spectra_fallback_full_eig_count != 0L ||
      diagnostic$lowrank_full_eig_count != 0L
  }, logical(1L))
  exact <- canonical_exact &&
    identical(endpoints, target_set) &&
    identical(records$parity_scope,
              rep("qualification", nrow(records))) &&
    identical(records$index, rep(1L, nrow(records))) &&
    identical(records$numCol, rep(35L, nrow(records))) &&
    identical(records$backend, rep("cpp", nrow(records))) &&
    identical(records$low_rank_backend,
              rep("spectra", nrow(records))) &&
    identical(records$p_value_difference, difference) &&
    identical(records$absolute_p_value_difference, abs(difference)) &&
    identical(records$signed_alpha_distance,
              p_value - logical_tests$alpha) &&
    identical(records$decision, decision) &&
    identical(records$independent, independent) &&
    identical(records$decision_flip, flip) &&
    identical(records$spectra_fallback, fallback) &&
    !any(records$backend_error)
  if (!isTRUE(exact) || any(!is.finite(p_value)) ||
      any(!is.finite(records$absolute_p_value_difference))) {
    stop("qualification dCov rows do not match canonical evidence",
         call. = FALSE)
  }
  list(
    qualification_dcov_logical_test_count = as.integer(nrow(records)),
    qualification_dcov_near_alpha_count = as.integer(sum(
      records$near_alpha
    )),
    qualification_dcov_unique_residual_key_count =
      as.integer(length(endpoints)),
    qualification_dcov_max_absolute_p_value_difference =
      as.double(max(records$absolute_p_value_difference)),
    qualification_dcov_decision_flip_count = as.integer(sum(
      records$decision_flip
    )),
    qualification_dcov_near_alpha_decision_flip_count = as.integer(sum(
      records$near_alpha & records$decision_flip
    )),
    qualification_dcov_backend_error_count = as.integer(sum(
      records$backend_error
    )),
    qualification_dcov_spectra_fallback_count = as.integer(sum(
      records$spectra_fallback
    )),
    qualification_dcov_logical_ids_hash =
      fastkpc_full_cuda_census_metadata_hash(
        records$logical_sequence_id
      ),
    qualification_dcov_residual_key_hash =
      fastkpc_full_cuda_census_key_set_hash(endpoints),
    qualification_dcov_rows_hash =
      fastkpc_full_cuda_census_frame_hash(records)
  )
}

fastkpc_full_cuda_fixed_sp_load_qualification_logical_tests <- function(
    census_dir, prepared_dir, data_path) {
  scalar_path <- function(value, label, directory = FALSE) {
    if (typeof(value) != "character" || length(value) != 1L ||
        is.object(value) || !is.null(attributes(value)) || anyNA(value) ||
        !nzchar(value)) {
      stop(label, " is malformed", call. = FALSE)
    }
    normalized <- normalizePath(value, winslash = "/", mustWork = TRUE)
    if (!identical(dir.exists(normalized), directory)) {
      stop(label, " has the wrong file type", call. = FALSE)
    }
    normalized
  }
  census_dir <- scalar_path(census_dir, "qualification census_dir", TRUE)
  prepared_dir <- scalar_path(
    prepared_dir, "qualification prepared_dir", TRUE
  )
  data_path <- scalar_path(data_path, "qualification data_path", FALSE)
  contract <- fastkpc_full_cuda_fixed_sp_catalog_contract()
  logical_path <- file.path(
    prepared_dir, "qualification_logical_tests.rds"
  )
  manifest_path <- file.path(prepared_dir, "manifest.json")
  logical_sha256 <- fastkpc_full_cuda_fixed_sp_sha256_file(logical_path)
  expected_sha256 <- unname(contract$phase2_semantic_file_sha256[[
    "qualification_logical_tests_rds"
  ]])
  manifest <- fastkpc_full_cuda_fixed_sp_read_json(manifest_path)
  manifest_sha256 <- manifest$semantic_file_sha256[[
    "qualification_logical_tests_rds"
  ]]
  if (!fastkpc_full_cuda_fixed_sp_is_bare_sha256(expected_sha256) ||
      !fastkpc_full_cuda_fixed_sp_is_bare_sha256(manifest_sha256) ||
      !identical(logical_sha256, expected_sha256) ||
      !identical(manifest_sha256, expected_sha256)) {
    stop("qualification logical-test artifact hash mismatch",
         call. = FALSE)
  }
  logical_tests <- readRDS(logical_path)
  inputs <- fastkpc_full_cuda_prepared_s_load_inputs(
    census_dir, data_path
  )
  expected <- fastkpc_full_cuda_prepared_s_selection_for_scope(
    inputs, "qualification"
  )$logical_tests
  schema <- fastkpc_full_cuda_fixed_sp_qualification_dcov_schema()
  fields_exact <- is.data.frame(logical_tests) &&
    is.data.frame(expected) &&
    identical(names(logical_tests), schema$logical_names) &&
    identical(names(expected), schema$logical_names) &&
    nrow(logical_tests) == nrow(expected) &&
    all(vapply(schema$logical_names, function(field) {
      identical(logical_tests[[field]], expected[[field]])
    }, logical(1L)))
  endpoint_keys <- if (isTRUE(fields_exact)) {
    sort(unique(c(
      logical_tests$residual_key_x, logical_tests$residual_key_y
    )), method = "radix")
  } else {
    character()
  }
  gates <- fields_exact && nrow(logical_tests) == 3808L &&
    identical(sum(logical_tests$near_alpha), 1478L) &&
    identical(
      logical_tests$logical_sequence_id,
      sort(logical_tests$logical_sequence_id, method = "radix")
    ) && !anyDuplicated(logical_tests$logical_sequence_id) &&
    length(endpoint_keys) == 6143L && !anyDuplicated(endpoint_keys) &&
    all(grepl("^[0-9a-f]{64}$", endpoint_keys))
  if (!isTRUE(gates)) {
    stop("qualification logical-test canonical evidence mismatch",
         call. = FALSE)
  }
  list(
    inputs = inputs,
    logical_tests = logical_tests,
    logical_tests_sha256 = logical_sha256,
    logical_ids_hash = fastkpc_full_cuda_census_metadata_hash(
      logical_tests$logical_sequence_id
    ),
    endpoint_keys = endpoint_keys,
    endpoint_key_hash =
      fastkpc_full_cuda_census_key_set_hash(endpoint_keys)
  )
}

fastkpc_full_cuda_fixed_sp_run_qualification_dcov_parity <- function(
    census_dir, prepared_dir, data_path, phase0_dir, residual_registry,
    target_records) {
  phase0_dir <- normalizePath(
    phase0_dir, winslash = "/", mustWork = TRUE
  )
  if (!dir.exists(phase0_dir)) {
    stop("qualification Phase 0 directory is malformed", call. = FALSE)
  }
  cleared <- FALSE
  on.exit({
    if (!cleared) {
      fastkpc_full_cuda_fixed_sp_residual_registry_clear(
        residual_registry
      )
    }
  }, add = TRUE)
  registry_keys <-
    fastkpc_full_cuda_fixed_sp_residual_registry_validate(
      residual_registry, target_records = target_records
    )
  authenticated <-
    fastkpc_full_cuda_fixed_sp_load_qualification_logical_tests(
      census_dir = census_dir,
      prepared_dir = prepared_dir,
      data_path = data_path
    )
  if (!identical(
        sort(registry_keys, method = "radix"),
        authenticated$endpoint_keys
      )) {
    stop("qualification residual registry does not match logical endpoints",
         call. = FALSE)
  }
  parity <- fastkpc_full_cuda_fixed_sp_with_legacy_dcov_backend(function() {
    qualification_core <-
      fastkpc_full_cuda_fixed_sp_qualification_dcov_core()
    fastkpc_full_cuda_fixed_sp_residual_registry_validate(
      residual_registry, target_records = target_records
    )
    qualification_core(
      inputs = authenticated$inputs,
      logical_tests = authenticated$logical_tests,
      residuals = residual_registry,
      oracle_manifest = file.path(phase0_dir, "manifest.json"),
      oracle_fun = fastkpc_cuda_legacy_dcov_gamma_cpp_oracle,
      scope = "qualification"
    )
  })
  if (!is.list(parity) || !is.data.frame(parity$rows)) {
    stop("qualification dCov parity core returned malformed evidence",
         call. = FALSE)
  }
  records <-
    fastkpc_full_cuda_fixed_sp_build_qualification_dcov_records(
      authenticated$logical_tests, parity$rows
    )
  summary <-
    fastkpc_full_cuda_fixed_sp_summarize_qualification_dcov_records(
      records, authenticated$logical_tests, registry_keys
    )
  canonical <-
    identical(summary$qualification_dcov_logical_test_count, 3808L) &&
    identical(summary$qualification_dcov_near_alpha_count, 1478L) &&
    identical(
      summary$qualification_dcov_unique_residual_key_count, 6143L
    ) && is.finite(
      summary$qualification_dcov_max_absolute_p_value_difference
    ) && summary$qualification_dcov_max_absolute_p_value_difference <
      1e-10 &&
    identical(summary$qualification_dcov_decision_flip_count, 0L) &&
    identical(
      summary$qualification_dcov_near_alpha_decision_flip_count, 0L
    ) &&
    identical(summary$qualification_dcov_backend_error_count, 0L) &&
    identical(summary$qualification_dcov_spectra_fallback_count, 0L) &&
    identical(
      summary$qualification_dcov_logical_ids_hash,
      authenticated$logical_ids_hash
    ) && identical(
      summary$qualification_dcov_residual_key_hash,
      authenticated$endpoint_key_hash
    )
  if (!isTRUE(canonical)) {
    stop("qualification dCov canonical gate failed", call. = FALSE)
  }
  fastkpc_full_cuda_fixed_sp_residual_registry_clear(residual_registry)
  cleared <- TRUE
  list(
    records = records,
    summary = summary,
    logical_tests_sha256 = authenticated$logical_tests_sha256
  )
}

fastkpc_full_cuda_fixed_sp_validate_repeat_exact <- function(first, second) {
  route_fields <- c(
    "residual_key_sha256", "planned_route", "executed_route",
    "reroute_reason", "solver_status"
  )
  numeric_fields <- c(
    "residual_key_sha256", "fitted_numeric_hash", "residual_numeric_hash"
  )
  result_clean <- function(value) {
    is.list(value) && !is.object(value) &&
      is.data.frame(value$target_records) &&
      all(c(route_fields, numeric_fields) %in% names(value$target_records)) &&
      is.list(value$summary) && !is.object(value$summary) &&
      all(c("route_status_hash", "numeric_hash") %in% names(value$summary)) &&
      all(vapply(c(route_fields, numeric_fields), function(field) {
        column <- value$target_records[[field]]
        typeof(column) == "character" && !is.object(column) &&
          is.null(attributes(column)) && !anyNA(column)
      }, logical(1L))) &&
      all(vapply(c("route_status_hash", "numeric_hash"), function(field) {
        fastkpc_full_cuda_fixed_sp_is_bare_sha256(value$summary[[field]])
      }, logical(1L)))
  }
  if (!isTRUE(result_clean(first)) || !isTRUE(result_clean(second))) {
    stop("Phase 3 repeat evidence is malformed", call. = FALSE)
  }
  if (!identical(
    first$target_records[route_fields], second$target_records[route_fields]
  )) {
    stop("Phase 3 repeat route/status evidence changed", call. = FALSE)
  }
  if (!identical(
    first$target_records[numeric_fields], second$target_records[numeric_fields]
  )) {
    stop("Phase 3 repeat numeric evidence changed", call. = FALSE)
  }
  recompute <- function(value) list(
    route_status_hash = fastkpc_full_cuda_census_metadata_hash(list(
      value$target_records$residual_key_sha256,
      value$target_records$planned_route,
      value$target_records$executed_route,
      value$target_records$reroute_reason,
      value$target_records$solver_status
    )),
    numeric_hash = fastkpc_full_cuda_census_metadata_hash(list(
      value$target_records$residual_key_sha256,
      value$target_records$fitted_numeric_hash,
      value$target_records$residual_numeric_hash
    ))
  )
  first_hashes <- recompute(first)
  second_hashes <- recompute(second)
  if (!identical(first$summary[names(first_hashes)], first_hashes) ||
      !identical(second$summary[names(second_hashes)], second_hashes)) {
    stop("Phase 3 repeat summary hashes are not row-derived", call. = FALSE)
  }
  if (!identical(first_hashes, second_hashes)) {
    stop("Phase 3 repeat summary hashes changed", call. = FALSE)
  }
  TRUE
}

fastkpc_run_full_cuda_fixed_sp_phase3c_iteration <- function(
    phase2_dir, census_dir, prepared_dir, data_path, device_id = 0L,
    scope = "iteration", return_residual_registry = FALSE) {
  scope <- match.arg(scope, c("iteration", "qualification"))
  registry_flag_clean <- typeof(return_residual_registry) == "logical" &&
    length(return_residual_registry) == 1L &&
    !is.object(return_residual_registry) &&
    is.null(attributes(return_residual_registry)) &&
    !is.na(return_residual_registry)
  if (!isTRUE(registry_flag_clean) ||
      (isTRUE(return_residual_registry) &&
       !identical(scope, "qualification"))) {
    stop("Phase 3C residual registry is qualification-only",
         call. = FALSE)
  }
  required_functions <- c(
    "fixed_sp_cuda_runtime_create", "fixed_sp_cuda_runtime_reserve",
    "fixed_sp_cuda_runtime_info", "fixed_sp_cuda_runtime_free",
    "fixed_sp_cuda_prepared_create", "fixed_sp_cuda_prepared_info",
    "fixed_sp_cuda_prepared_free", "fixed_sp_cuda_solve_batch",
    "fixed_sp_cuda_residual_info", "fixed_sp_cuda_materialize_shadow",
    "fixed_sp_cuda_residual_release", "fixed_sp_cuda_residual_free",
    "fastkpc_mgcv_magic_fixed_sp_from_prepared"
  )
  missing <- required_functions[!vapply(
    required_functions, exists, logical(1L), mode = "function",
    inherits = TRUE
  )]
  if (length(missing) != 0L) {
    stop(
      "Phase 3C CUDA API is unavailable: ", paste(missing, collapse = ", "),
      call. = FALSE
    )
  }
  validated_paths <- fastkpc_full_cuda_fixed_sp_phase3b_validate_paths(
    phase2_dir, census_dir, prepared_dir, data_path
  )
  phase2_dir <- validated_paths$phase2_dir
  census_dir <- validated_paths$census_dir
  prepared_dir <- validated_paths$prepared_dir
  data_path <- validated_paths$data_path
  if (typeof(device_id) != "integer" || length(device_id) != 1L ||
      is.object(device_id) || !is.null(attributes(device_id)) ||
      is.na(device_id) || device_id < 0L) {
    stop("Phase 3C device_id must be one non-negative integer",
         call. = FALSE)
  }

  catalog <- fastkpc_full_cuda_open_fixed_sp_catalog(
    phase2_dir, census_dir, prepared_dir, data_path
  )
  selected_scope <- fastkpc_full_cuda_fixed_sp_scope(catalog, scope)
  batches <- fastkpc_full_cuda_fixed_sp_batches(catalog, selected_scope)
  setup_keys <- names(batches)
  if (length(setup_keys) < 1L ||
      !identical(setup_keys, sort(setup_keys, method = "radix")) ||
      anyDuplicated(setup_keys)) {
    stop("Phase 3C ", scope, " PreparedSKey order is not canonical",
         call. = FALSE)
  }
  ordered_target_keys <- unlist(lapply(batches, function(batch) {
    as.character(batch$target_rows$residual_key_sha256)
  }), use.names = FALSE)
  ordered_target_owners <- unlist(lapply(seq_along(batches), function(index) {
    rep.int(setup_keys[[index]], nrow(batches[[index]]$target_rows))
  }), use.names = FALSE)
  if (anyDuplicated(ordered_target_keys)) {
    stop("Phase 3C ", scope, " target-key order is not canonical",
         call. = FALSE)
  }
  scope_target_rows <- catalog$scopes[[scope]]$target_rows
  condition_bucket_by_target <- setNames(
    as.character(scope_target_rows$condition_bucket),
    as.character(scope_target_rows$residual_key_sha256)
  )
  if (anyNA(condition_bucket_by_target) ||
      !setequal(names(condition_bucket_by_target), ordered_target_keys)) {
    stop("Phase 3C ", scope, " condition-bucket lineage is malformed",
         call. = FALSE)
  }
  scope_subset_hash <- catalog$catalog_contract[[paste0(
    scope, "_subset_hash"
  )]]
  catalog_records <- data.frame(
    scope = scope,
    authenticated = TRUE,
    catalog_open_count = 1L,
    setup_count = as.integer(length(batches)),
    target_count = as.integer(sum(vapply(
      batches, function(batch) nrow(batch$target_rows), integer(1L)
    ))),
    scope_subset_hash = scope_subset_hash,
    iteration_subset_hash = scope_subset_hash,
    ordered_setup_key_digest =
      fastkpc_full_cuda_census_key_set_hash(setup_keys),
    ordered_target_key_digest =
      fastkpc_full_cuda_census_key_set_hash(ordered_target_keys),
    stringsAsFactors = FALSE
  )

  dtos <- lapply(batches, function(batch) {
    fastkpc_full_cuda_fixed_sp_native_dto(batch$setup)
  })
  native_batches <- lapply(seq_along(batches), function(index) {
    fastkpc_full_cuda_fixed_sp_native_batch(
      batches[[index]], dtos[[index]]
    )
  })
  names(dtos) <- names(native_batches) <- setup_keys
  max_n <- max(vapply(dtos, `[[`, integer(1L), "n"))
  max_q <- max(vapply(dtos, `[[`, integer(1L), "null_dim"))
  max_targets <- max(vapply(
    native_batches, `[[`, integer(1L), "target_count"
  ))
  max_penalties <- max(vapply(dtos, `[[`, integer(1L), "penalty_count"))
  max_augmented_rows <- max(vapply(dtos, function(dto) {
    as.integer(dto$n + sum(dto$penalty_ranks))
  }, integer(1L)))
  if (!identical(max_augmented_rows, 407L)) {
    stop("Phase 3C logical augmented-row reserve must equal 407",
         call. = FALSE)
  }
  registry_n <- unique(vapply(dtos, `[[`, integer(1L), "n"))
  if (!identical(registry_n, 351L)) {
    stop("Phase 3C qualification residual length must equal 351",
         call. = FALSE)
  }
  residual_registry <- if (isTRUE(return_residual_registry)) {
    fastkpc_full_cuda_fixed_sp_residual_registry_create(
      expected_keys = ordered_target_keys,
      expected_owners = ordered_target_owners,
      n = registry_n
    )
  } else {
    NULL
  }

  runtime <- NULL
  runtime_freed <- FALSE
  runtime_cleanup_operations <- list(
    token_release = list(
      needed = function() FALSE,
      run = function() invisible(NULL)
    ),
    token_free = list(
      needed = function() FALSE,
      run = function() invisible(NULL)
    ),
    handle_free = list(
      needed = function() FALSE,
      run = function() invisible(NULL)
    ),
    runtime_free = list(
      needed = function() !is.null(runtime) && !runtime_freed,
      run = function() {
        fixed_sp_cuda_runtime_free(runtime)
        runtime_freed <<- TRUE
        runtime <<- NULL
      }
    )
  )

  iteration_result <-
    fastkpc_full_cuda_fixed_sp_phase3b_run_cleanup_scope(
      body = {
        runtime <- fixed_sp_cuda_runtime_create(device_id)
        runtime_created <- fixed_sp_cuda_runtime_info(runtime)
        fastkpc_full_cuda_fixed_sp_phase3c_validate_runtime_info(
          runtime_created, "Phase 3C runtime-created info"
        )
        fixed_sp_cuda_runtime_reserve(
          runtime, max_n, max_q, max_targets, max_penalties,
          max_augmented_rows
        )
        runtime_reserved <- fixed_sp_cuda_runtime_info(runtime)
        fastkpc_full_cuda_fixed_sp_phase3c_validate_runtime_info(
          runtime_reserved, "Phase 3C workspace-reserved info"
        )
        reserve_count_delta <-
          fastkpc_full_cuda_fixed_sp_phase3b_counter_delta(
            runtime_created, runtime_reserved, "workspace_reserve_count",
            "Phase 3C runtime reserve"
          )
        if (runtime_created$workspace_reserve_count != 0L ||
            reserve_count_delta != 1L ||
            runtime_created$cuda_device_synchronize_count != 0L ||
            runtime_reserved$cuda_device_synchronize_count != 0L ||
            !isTRUE(runtime_reserved$cublas_user_workspace_installed) ||
            runtime_reserved$cublas_workspace_alignment < 256 ||
            runtime_reserved$augmented_workspace_bytes !=
              as.double(8 * 415L * 64L) ||
            runtime_reserved$aggregate_factor_workspace_bytes !=
              as.double(8 * (64L * 64L + 2L * 64L))) {
          stop("Phase 3C runtime create/reserve lifecycle is invalid",
               call. = FALSE)
        }

        run_batch <- function(batch_index) {
          setup_key <- setup_keys[[batch_index]]
          batch <- batches[[setup_key]]
          dto <- dtos[[setup_key]]
          native <- native_batches[[setup_key]]
          target_count <- native$target_count
          handle <- NULL
          token <- NULL
          token_released <- FALSE
          token_freed <- FALSE
          handle_freed <- FALSE
          prepared_created <- NULL
          prepared_before_solve <- NULL
          prepared_after_solve <- NULL
          prepared_after_release <- NULL
          pre_shadow_info <- NULL
          post_shadow_info <- NULL
          released_info <- NULL
          shadow <- NULL
          rank_reference_materialize_elapsed_ms <- 0
          solve_elapsed_ms <- NA_real_
          shadow_materialize_elapsed_ms <- NA_real_
          cpu_aggregate_penalty_root_rank <-
            rep(NA_integer_, target_count)
          cpu_aggregate_penalty_root_pivot <-
            rep(list(integer()), target_count)
          cpu_aggregate_effective_rank <- rep(NA_integer_, target_count)
          cpu_aggregate_effective_rank_threshold <-
            rep(NA_real_, target_count)
          cpu_aggregate_sigma_max <- rep(NA_real_, target_count)
          expected_shadow_calls <- 0L
          expected_shadow_targets <- 0L
          expected_shadow_bytes <- 0
          route_status_conservation_exact <- FALSE

          cleanup_operations <- list(
            token_release = list(
              needed = function() !is.null(token) && !token_released,
              run = function() {
                fixed_sp_cuda_residual_release(token)
                token_released <<- TRUE
                released_info <<- fixed_sp_cuda_residual_info(token)
                if (!is.null(post_shadow_info)) {
                  fastkpc_full_cuda_fixed_sp_phase3c_validate_batch_info(
                    released_info, native, dto, expected_shadow_calls,
                    expected_shadow_targets, expected_shadow_bytes, 1L,
                    paste("Phase 3C batch", batch_index, "released info")
                  )
                  release_fields <- setdiff(
                    names(post_shadow_info), "output_slot_release_count"
                  )
                  if (!identical(
                        post_shadow_info[release_fields],
                        released_info[release_fields]
                      )) {
                    stop("Phase 3C token release changed solve diagnostics",
                         call. = FALSE)
                  }
                }
                prepared_after_release <<- fixed_sp_cuda_prepared_info(handle)
                fastkpc_full_cuda_fixed_sp_phase3c_validate_prepared_info(
                  prepared_after_release, dto, 0L,
                  paste(
                    "Phase 3C batch", batch_index,
                    "released prepared info"
                  )
                )
                if (!is.null(prepared_before_solve) &&
                    !is.null(prepared_after_solve)) {
                  fastkpc_full_cuda_fixed_sp_phase3b_validate_prepared_snapshots(
                    prepared_before_solve, prepared_after_solve,
                    prepared_after_release, setup_key,
                    as.double(dto$coefficient_dim) * as.double(target_count)
                  )
                }
              }
            ),
            token_free = list(
              needed = function() !is.null(token) && !token_freed,
              run = function() {
                fixed_sp_cuda_residual_free(token)
                token_freed <<- TRUE
                token_released <<- TRUE
                token <<- NULL
              }
            ),
            handle_free = list(
              needed = function() !is.null(handle) && !handle_freed,
              run = function() {
                fixed_sp_cuda_prepared_free(handle)
                handle_freed <<- TRUE
                handle <<- NULL
              }
            ),
            runtime_free = list(
              needed = function() FALSE,
              run = function() invisible(NULL)
            )
          )

          runtime_before_solve <- NULL
          runtime_after_solve <- NULL
          fastkpc_full_cuda_fixed_sp_phase3b_run_cleanup_scope(
            body = {
              handle <- fixed_sp_cuda_prepared_create(runtime, dto)
              prepared_created <- fixed_sp_cuda_prepared_info(handle)
              fastkpc_full_cuda_fixed_sp_phase3c_validate_prepared_info(
                prepared_created, dto, 0L,
                paste("Phase 3C batch", batch_index, "created prepared info")
              )
              if (prepared_created$setup_h2d_upload_count != 1L ||
                  isTRUE(prepared_created$output_slot_leased)) {
                stop("Phase 3C prepared setup/upload lifecycle is invalid",
                     call. = FALSE)
              }

              prepared_before_solve <- fixed_sp_cuda_prepared_info(handle)
              fastkpc_full_cuda_fixed_sp_phase3c_validate_prepared_info(
                prepared_before_solve, dto, 0L,
                paste("Phase 3C batch", batch_index, "pre-solve prepared info")
              )

              runtime_before_solve <- fixed_sp_cuda_runtime_info(runtime)
              fastkpc_full_cuda_fixed_sp_phase3c_validate_runtime_info(
                runtime_before_solve,
                paste("Phase 3C batch", batch_index, "runtime before")
              )
              solve_start <- proc.time()[["elapsed"]]
              token <- fixed_sp_cuda_solve_batch(
                handle, native$Y, native$SP, native$planned_route,
                native$target_keys, outputs = c("fitted", "residuals")
              )
              solve_elapsed_ms <- as.double(
                (proc.time()[["elapsed"]] - solve_start) * 1000
              )
              pre_shadow_info <- fixed_sp_cuda_residual_info(token)
              pre_validation <-
                fastkpc_full_cuda_fixed_sp_phase3c_validate_batch_info(
                  pre_shadow_info, native, dto, 0L, 0L, 0, 0L,
                  paste("Phase 3C batch", batch_index, "pre-shadow info")
                )
              route_status_conservation_exact <-
                pre_validation$route_status_conservation_exact
              runtime_after_solve <- fixed_sp_cuda_runtime_info(runtime)
              fastkpc_full_cuda_fixed_sp_phase3c_validate_runtime_info(
                runtime_after_solve,
                paste("Phase 3C batch", batch_index, "runtime after")
              )
              prepared_after_solve <- fixed_sp_cuda_prepared_info(handle)
              fastkpc_full_cuda_fixed_sp_phase3c_validate_prepared_info(
                prepared_after_solve, dto, 0L,
                paste("Phase 3C batch", batch_index, "post-solve prepared info")
              )
              if (!isTRUE(prepared_after_solve$output_slot_leased)) {
                stop("Phase 3C solve did not retain its output-slot lease",
                     call. = FALSE)
              }

              svd_indices <- which(
                pre_shadow_info$executed_route == "AUGMENTED_SVD"
              )
              for (target_index in svd_indices) {
                aggregate_reference <-
                  fastkpc_full_cuda_fixed_sp_phase3c_cpu_aggregate_shadow(
                    dto, native$SP[, target_index]
                  )
                cpu_aggregate_penalty_root_rank[[target_index]] <-
                  aggregate_reference$root_rank
                cpu_aggregate_penalty_root_pivot[[target_index]] <-
                  aggregate_reference$root_pivot
                cpu_aggregate_effective_rank[[target_index]] <-
                  aggregate_reference$effective_rank
                cpu_aggregate_effective_rank_threshold[[target_index]] <-
                  aggregate_reference$effective_rank_threshold
                cpu_aggregate_sigma_max[[target_index]] <-
                  aggregate_reference$sigma_max
              }

              expected_shadow_calls <- 1L
              expected_shadow_targets <- target_count
              expected_shadow_bytes <-
                8 * as.double(target_count) * as.double(2L * dto$n)
              shadow_start <- proc.time()[["elapsed"]]
              shadow <- fixed_sp_cuda_materialize_shadow(
                token, outputs = c("fitted", "residuals")
              )
              shadow_materialize_elapsed_ms <- as.double(
                (proc.time()[["elapsed"]] - shadow_start) * 1000
              )
              shadow_clean <- is.list(shadow) &&
                identical(names(shadow), c("fitted", "residuals")) &&
                is.matrix(shadow$fitted) && is.double(shadow$fitted) &&
                identical(dim(shadow$fitted), c(dto$n, target_count)) &&
                all(is.finite(shadow$fitted)) &&
                is.matrix(shadow$residuals) && is.double(shadow$residuals) &&
                identical(dim(shadow$residuals), c(dto$n, target_count)) &&
                all(is.finite(shadow$residuals))
              if (!isTRUE(shadow_clean)) {
                stop("Phase 3C explicit oracle shadow is malformed",
                     call. = FALSE)
              }
              if (isTRUE(return_residual_registry)) {
                fastkpc_full_cuda_fixed_sp_residual_registry_capture(
                  registry = residual_registry,
                  owner = setup_key,
                  target_keys = native$target_keys,
                  residuals = shadow$residuals
                )
              }
              post_shadow_info <- fixed_sp_cuda_residual_info(token)
              fastkpc_full_cuda_fixed_sp_phase3c_validate_batch_info(
                post_shadow_info, native, dto, expected_shadow_calls,
                expected_shadow_targets, expected_shadow_bytes, 0L,
                paste("Phase 3C batch", batch_index, "post-shadow info")
              )
              shadow_fields <- c(
                "shadow_materialize_call_count",
                "shadow_materialize_target_count", "shadow_d2h_bytes"
              )
              unchanged_fields <- setdiff(names(pre_shadow_info), shadow_fields)
              if (!identical(
                    pre_shadow_info[unchanged_fields],
                    post_shadow_info[unchanged_fields]
                  )) {
                stop("Phase 3C explicit shadow changed solve diagnostics",
                     call. = FALSE)
              }
              NULL
            },
            operations = cleanup_operations,
            context = paste("Phase 3C batch", batch_index, "cleanup")
          )

          if (is.null(runtime_before_solve) || is.null(runtime_after_solve) ||
              is.null(released_info) || is.null(prepared_after_release)) {
            stop("Phase 3C batch lifecycle evidence is incomplete",
                 call. = FALSE)
          }
          workspace_grow_delta <-
            fastkpc_full_cuda_fixed_sp_phase3b_counter_delta(
              runtime_before_solve, runtime_after_solve,
              "workspace_grow_count", paste("Phase 3C batch", batch_index)
            )
          stable_workspace_grow_delta <-
            fastkpc_full_cuda_fixed_sp_phase3b_counter_delta(
              runtime_before_solve, runtime_after_solve,
              "stable_workspace_grow_count",
              paste("Phase 3C batch", batch_index)
            )
          synchronize_delta <-
            fastkpc_full_cuda_fixed_sp_phase3b_counter_delta(
              runtime_before_solve, runtime_after_solve,
              "cuda_device_synchronize_count",
              paste("Phase 3C batch", batch_index)
            )
          checkpoint_fields <- c(
            "cholesky_factor_checkpoint_record_count",
            "cholesky_factor_checkpoint_wait_count",
            "cholesky_solve_checkpoint_record_count",
            "cholesky_solve_checkpoint_wait_count",
            "qr_checkpoint_record_count", "qr_checkpoint_wait_count",
            "svd_checkpoint_record_count", "svd_checkpoint_wait_count"
          )
          checkpoint_deltas <- vapply(checkpoint_fields, function(field) {
            fastkpc_full_cuda_fixed_sp_phase3b_counter_delta(
              runtime_before_solve, runtime_after_solve, field,
              paste("Phase 3C batch", batch_index)
            )
          }, integer(1L))
          if (workspace_grow_delta != 0L ||
              stable_workspace_grow_delta != 0L ||
              synchronize_delta != 0L) {
            stop("Phase 3C batch grew workspace or synchronized",
                 call. = FALSE)
          }

          residual_max_abs <- double(target_count)
          residual_relative_l2 <- double(target_count)
          fitted_max_abs <- double(target_count)
          fitted_relative_l2 <- double(target_count)
          fitted_numeric_hash <- character(target_count)
          residual_numeric_hash <- character(target_count)
          oracle_fitted_hash <- character(target_count)
          oracle_residual_hash <- character(target_count)
          authenticated_fitted_hash <-
            as.character(batch$target_rows$fitted_hash)
          authenticated_residual_hash <-
            as.character(batch$target_rows$residual_hash)
          oracle_fitted_hash_exact <- logical(target_count)
          oracle_residual_hash_exact <- logical(target_count)
          outputs_all_finite <- logical(target_count)
          approximate_backend <- logical(target_count)
          oracle_call_count <- integer(target_count)
          numeric_reference <- rep("mgcv-fixed-sp", target_count)
          for (target_index in seq_len(target_count)) {
            outputs_all_finite[[target_index]] <-
              all(is.finite(shadow$fitted[, target_index])) &&
              all(is.finite(shadow$residuals[, target_index]))
            if (!isTRUE(outputs_all_finite[[target_index]])) {
              stop("Phase 3C target output is nonfinite: ",
                   native$target_keys[[target_index]], call. = FALSE)
            }
            oracle_call_count[[target_index]] <- 1L
            oracle <- fastkpc_mgcv_magic_fixed_sp_from_prepared(
              prepared_setup = batch$setup,
              target_state = list(
                row = batch$target_rows[target_index, , drop = FALSE],
                y = as.numeric(native$Y[, target_index])
              )
            )
            approximate_backend[[target_index]] <- !(
              identical(oracle$backend_family, "mgcvExtractCPU") &&
                identical(
                  oracle$mode, "prepared-s-fixed-sp-mgcv-reference"
                ) && isTRUE(oracle$authoritative)
            )
            residual_errors <-
              fastkpc_full_cuda_fixed_sp_phase3b_numeric_errors(
                shadow$residuals[, target_index], oracle$residuals,
                paste(
                  "Phase 3C residual target",
                  native$target_keys[[target_index]]
                )
              )
            fitted_errors <-
              fastkpc_full_cuda_fixed_sp_phase3b_numeric_errors(
                shadow$fitted[, target_index], oracle$fitted,
                paste(
                  "Phase 3C fitted target",
                  native$target_keys[[target_index]]
                )
              )
            errors <- c(residual_errors, fitted_errors)
            if (any(!is.finite(errors)) || any(errors >= 1e-7)) {
              stop(
                "Phase 3C target failed route numeric parity: ",
                native$target_keys[[target_index]], "; reference=",
                numeric_reference[[target_index]], "; errors=",
                paste(format(errors, digits = 17L), collapse = ","),
                call. = FALSE
              )
            }
            residual_max_abs[[target_index]] <-
              residual_errors[["max_abs"]]
            residual_relative_l2[[target_index]] <-
              residual_errors[["relative_l2"]]
            fitted_max_abs[[target_index]] <- fitted_errors[["max_abs"]]
            fitted_relative_l2[[target_index]] <-
              fitted_errors[["relative_l2"]]
            fitted_numeric_hash[[target_index]] <-
              fastkpc_full_cuda_census_metadata_hash(
                shadow$fitted[, target_index]
              )
            residual_numeric_hash[[target_index]] <-
              fastkpc_full_cuda_census_metadata_hash(
                shadow$residuals[, target_index]
              )
            oracle_fitted_hash[[target_index]] <-
              fastkpc_full_cuda_census_metadata_hash(oracle$fitted)
            oracle_residual_hash[[target_index]] <-
              fastkpc_full_cuda_census_metadata_hash(oracle$residuals)
            oracle_fitted_hash_exact[[target_index]] <- identical(
              oracle_fitted_hash[[target_index]],
              authenticated_fitted_hash[[target_index]]
            )
            oracle_residual_hash_exact[[target_index]] <- identical(
              oracle_residual_hash[[target_index]],
              authenticated_residual_hash[[target_index]]
            )
            if (!isTRUE(oracle_fitted_hash_exact[[target_index]]) ||
                !isTRUE(oracle_residual_hash_exact[[target_index]]) ||
                isTRUE(approximate_backend[[target_index]])) {
              stop("Phase 3C oracle authentication failed: ",
                   native$target_keys[[target_index]], call. = FALSE)
            }
          }

          svd_executed <-
            pre_shadow_info$executed_route == "AUGMENTED_SVD"
          aggregate_penalty_root_rank_exact <- rep(NA, target_count)
          aggregate_penalty_root_pivot_exact <- rep(NA, target_count)
          aggregate_effective_rank_exact <- rep(NA, target_count)
          aggregate_penalty_root_rank_exact[svd_executed] <-
            pre_shadow_info$aggregate_penalty_root_rank[svd_executed] ==
              cpu_aggregate_penalty_root_rank[svd_executed]
          aggregate_penalty_root_pivot_exact[svd_executed] <- vapply(
            which(svd_executed), function(target_index) {
              identical(
                pre_shadow_info$aggregate_penalty_root_pivot[[target_index]],
                cpu_aggregate_penalty_root_pivot[[target_index]]
              )
            }, logical(1L)
          )
          aggregate_effective_rank_exact[svd_executed] <-
            pre_shadow_info$effective_rank[svd_executed] ==
              cpu_aggregate_effective_rank[svd_executed]
          if (any(!aggregate_penalty_root_rank_exact[svd_executed]) ||
              any(!aggregate_penalty_root_pivot_exact[svd_executed]) ||
              any(!aggregate_effective_rank_exact[svd_executed])) {
            stop("Phase 3C GPU/CPU aggregate SVD diagnostics mismatch",
                 call. = FALSE)
          }

          target_records <- data.frame(
            prepared_s_key_sha256 = setup_key,
            batch_ordinal = as.integer(batch_index),
            target_ordinal = seq_len(target_count),
            residual_key_sha256 = native$target_keys,
            target = as.integer(batch$target_rows$target),
            null_dim = rep.int(dto$null_dim, target_count),
            phase1_condition = as.double(batch$condition),
            condition_bucket = unname(
              condition_bucket_by_target[native$target_keys]
            ),
            phase1_coefficient_rank =
              as.integer(batch$target_rows$coefficient_rank),
            planned_route = pre_shadow_info$planned_route,
            authenticated_planned_route = native$planned_route,
            executed_route = pre_shadow_info$executed_route,
            reroute_reason = pre_shadow_info$reroute_reason,
            solver_status = pre_shadow_info$solver_status,
            target_true_batched = pre_shadow_info$target_true_batched,
            qr_rank = pre_shadow_info$qr_rank,
            geqrf_info = pre_shadow_info$geqrf_info,
            ormqr_info = pre_shadow_info$ormqr_info,
            effective_rank = pre_shadow_info$effective_rank,
            sigma_max = pre_shadow_info$sigma_max,
            smallest_retained_sigma =
              pre_shadow_info$smallest_retained_sigma,
            svd_info = pre_shadow_info$svd_info,
            aggregate_penalty_root_rank =
              pre_shadow_info$aggregate_penalty_root_rank,
            aggregate_penalty_root_pivot = I(lapply(
              pre_shadow_info$aggregate_penalty_root_pivot, as.integer
            )),
            aggregate_factor_call_count =
              pre_shadow_info$aggregate_factor_call_count,
            aggregate_b_build_count =
              pre_shadow_info$aggregate_b_build_count,
            aggregate_dstop = pre_shadow_info$aggregate_dstop,
            cpu_aggregate_penalty_root_rank =
              cpu_aggregate_penalty_root_rank,
            cpu_aggregate_penalty_root_pivot = I(lapply(
              cpu_aggregate_penalty_root_pivot, as.integer
            )),
            cpu_aggregate_effective_rank = cpu_aggregate_effective_rank,
            cpu_aggregate_effective_rank_threshold =
              cpu_aggregate_effective_rank_threshold,
            cpu_aggregate_sigma_max = cpu_aggregate_sigma_max,
            aggregate_penalty_root_rank_exact =
              aggregate_penalty_root_rank_exact,
            aggregate_penalty_root_pivot_exact =
              aggregate_penalty_root_pivot_exact,
            aggregate_effective_rank_exact = aggregate_effective_rank_exact,
            numeric_reference = numeric_reference,
            outputs_all_finite = outputs_all_finite,
            residual_max_abs_diff = residual_max_abs,
            residual_relative_l2_diff = residual_relative_l2,
            fitted_max_abs_diff = fitted_max_abs,
            fitted_relative_l2_diff = fitted_relative_l2,
            fitted_numeric_hash = fitted_numeric_hash,
            residual_numeric_hash = residual_numeric_hash,
            oracle_call_count = oracle_call_count,
            oracle_fitted_hash = oracle_fitted_hash,
            oracle_residual_hash = oracle_residual_hash,
            authenticated_fitted_hash = authenticated_fitted_hash,
            authenticated_residual_hash = authenticated_residual_hash,
            oracle_fitted_hash_exact = oracle_fitted_hash_exact,
            oracle_residual_hash_exact = oracle_residual_hash_exact,
            approximate_backend = approximate_backend,
            stringsAsFactors = FALSE
          )
          setup_record <- data.frame(
            prepared_s_key_sha256 = setup_key,
            batch_ordinal = as.integer(batch_index),
            prepared_handle_create_count = 1L,
            prepared_handle_free_count = 1L,
            setup_h2d_upload_count = prepared_created$setup_h2d_upload_count,
            setup_h2d_bytes = prepared_created$setup_h2d_bytes,
            penalty_root_build_count =
              prepared_created$penalty_root_build_count,
            penalty_root_rank_mismatch_count =
              prepared_created$penalty_root_rank_mismatch_count,
            penalty_root_bytes = prepared_created$penalty_root_bytes,
            penalty_root_build_ms = prepared_created$penalty_root_build_ms,
            penalty_root_matrix_count =
              prepared_created$penalty_root_matrix_count,
            penalty_root_row_count =
              prepared_created$penalty_root_row_count,
            H_root_matrix_count = prepared_created$H_root_matrix_count,
            H_root_rank = prepared_created$H_root_rank,
            rank_reference_materialize_call_count = 0L,
            rank_reference_materialize_elapsed_ms =
              rank_reference_materialize_elapsed_ms,
            setup_shadow_d2h_count =
              prepared_before_solve$setup_shadow_d2h_count,
            setup_shadow_d2h_bytes =
              prepared_before_solve$setup_shadow_d2h_bytes,
            coefficient_output_capacity =
              prepared_created$coefficient_output_capacity,
            prepared_generation = prepared_created$generation,
            output_slot_state_before_solve =
              prepared_before_solve$output_slot_state,
            output_slot_state_after_solve =
              prepared_after_solve$output_slot_state,
            output_slot_state_after_release =
              prepared_after_release$output_slot_state,
            output_slot_leased_after_release =
              prepared_after_release$output_slot_leased,
            output_slot_poison_reason_empty = identical(
              c(
                prepared_before_solve$output_slot_poison_reason,
                prepared_after_solve$output_slot_poison_reason,
                prepared_after_release$output_slot_poison_reason
              ),
              rep("", 3L)
            ),
            stringsAsFactors = FALSE
          )
          batch_record <- data.frame(
            prepared_s_key_sha256 = setup_key,
            batch_ordinal = as.integer(batch_index),
            target_count = pre_shadow_info$target_count,
            batch_call_count = pre_shadow_info$batch_call_count,
            native_batch_call = pre_shadow_info$native_batch_call,
            true_batched_kernel = pre_shadow_info$true_batched_kernel,
            true_batched_subgroup_count =
              pre_shadow_info$true_batched_subgroup_count,
            true_batched_attempted_target_count =
              pre_shadow_info$true_batched_attempted_target_count,
            true_batched_target_count =
              pre_shadow_info$true_batched_target_count,
            cholesky_single_target_count =
              pre_shadow_info$cholesky_single_target_count,
            potrf_batched_call_count =
              pre_shadow_info$potrf_batched_call_count,
            potrs_batched_call_count =
              pre_shadow_info$potrs_batched_call_count,
            planned_cholesky_target_count =
              pre_shadow_info$planned_cholesky_target_count,
            planned_qr_target_count =
              pre_shadow_info$planned_qr_target_count,
            planned_svd_target_count =
              pre_shadow_info$planned_svd_target_count,
            executed_cholesky_target_count =
              pre_shadow_info$executed_cholesky_target_count,
            executed_qr_target_count =
              pre_shadow_info$executed_qr_target_count,
            executed_svd_target_count =
              pre_shadow_info$executed_svd_target_count,
            stable_reroute_count = pre_shadow_info$stable_reroute_count,
            cholesky_to_svd_count = pre_shadow_info$cholesky_to_svd_count,
            qr_to_svd_count = pre_shadow_info$qr_to_svd_count,
            aggregate_penalty_factor_count =
              pre_shadow_info$aggregate_penalty_factor_count,
            aggregate_svd_b_build_count =
              pre_shadow_info$aggregate_svd_b_build_count,
            aggregate_penalty_root_d2h_count =
              pre_shadow_info$aggregate_penalty_root_d2h_count,
            aggregate_penalty_root_d2h_bytes =
              pre_shadow_info$aggregate_penalty_root_d2h_bytes,
            target_batch_h2d_call_count =
              pre_shadow_info$target_batch_h2d_call_count,
            target_h2d_copy_count = pre_shadow_info$target_h2d_copy_count,
            target_h2d_bytes = pre_shadow_info$target_h2d_bytes,
            rhs_device_build_count = pre_shadow_info$rhs_device_build_count,
            full_cuda_data_plane = pre_shadow_info$full_cuda_data_plane,
            invalid_output_init_count =
              pre_shadow_info$invalid_output_init_count,
            coefficient_batch_finalize_call_count =
              pre_shadow_info$coefficient_batch_finalize_call_count,
            fitted_batch_finalize_call_count =
              pre_shadow_info$fitted_batch_finalize_call_count,
            residual_rss_batch_finalize_call_count =
              pre_shadow_info$residual_rss_batch_finalize_call_count,
            per_target_output_finalize_call_count =
              pre_shadow_info$per_target_output_finalize_call_count,
            batch_output_finalized_target_count =
              pre_shadow_info$batch_output_finalized_target_count,
            canonical_output_order_exact =
              pre_shadow_info$canonical_output_order_exact,
            target_keys_exact = identical(
              pre_shadow_info$target_keys, native$target_keys
            ),
            route_status_conservation_exact =
              route_status_conservation_exact,
            resource_snapshot_captured =
              pre_shadow_info$resource_snapshot_captured,
            resource_instrumentation_version =
              pre_shadow_info$resource_instrumentation_version,
            resource_allocation_count_before_solve =
              pre_shadow_info$resource_allocation_count_before_solve,
            resource_allocation_count_after_solve =
              pre_shadow_info$resource_allocation_count_after_solve,
            resource_handle_create_count_before_solve =
              pre_shadow_info$resource_handle_create_count_before_solve,
            resource_handle_create_count_after_solve =
              pre_shadow_info$resource_handle_create_count_after_solve,
            cuda_device_allocation_count_during_solve =
              pre_shadow_info$cuda_device_allocation_count_during_solve,
            cuda_host_allocation_count_during_solve =
              pre_shadow_info$cuda_host_allocation_count_during_solve,
            stream_create_count_during_solve =
              pre_shadow_info$stream_create_count_during_solve,
            event_create_count_during_solve =
              pre_shadow_info$event_create_count_during_solve,
            cublas_handle_create_count_during_solve =
              pre_shadow_info$cublas_handle_create_count_during_solve,
            cusolver_handle_create_count_during_solve =
              pre_shadow_info$cusolver_handle_create_count_during_solve,
            per_target_allocation_count_after_warmup =
              pre_shadow_info$per_target_allocation_count_after_warmup,
            per_target_handle_create_count =
              pre_shadow_info$per_target_handle_create_count,
            workspace_grow_count_after_warmup = workspace_grow_delta,
            stable_workspace_grow_count_after_warmup =
              stable_workspace_grow_delta,
            cuda_device_synchronize_count = synchronize_delta,
            cholesky_factor_checkpoint_record_count = checkpoint_deltas[[
              "cholesky_factor_checkpoint_record_count"
            ]],
            cholesky_factor_checkpoint_wait_count = checkpoint_deltas[[
              "cholesky_factor_checkpoint_wait_count"
            ]],
            cholesky_solve_checkpoint_record_count = checkpoint_deltas[[
              "cholesky_solve_checkpoint_record_count"
            ]],
            cholesky_solve_checkpoint_wait_count = checkpoint_deltas[[
              "cholesky_solve_checkpoint_wait_count"
            ]],
            qr_checkpoint_record_count = checkpoint_deltas[[
              "qr_checkpoint_record_count"
            ]],
            qr_checkpoint_wait_count = checkpoint_deltas[[
              "qr_checkpoint_wait_count"
            ]],
            svd_checkpoint_record_count = checkpoint_deltas[[
              "svd_checkpoint_record_count"
            ]],
            svd_checkpoint_wait_count = checkpoint_deltas[[
              "svd_checkpoint_wait_count"
            ]],
            implicit_residual_d2h_count =
              released_info$implicit_residual_d2h_count,
            cpu_fallback_count = pre_shadow_info$cpu_fallback_count,
            unknown_fallback_count = pre_shadow_info$unknown_fallback_count,
            solve_elapsed_ms = solve_elapsed_ms,
            pre_shadow_materialize_call_count =
              pre_shadow_info$shadow_materialize_call_count,
            pre_shadow_materialize_target_count =
              pre_shadow_info$shadow_materialize_target_count,
            pre_shadow_d2h_bytes = pre_shadow_info$shadow_d2h_bytes,
            shadow_materialize_elapsed_ms = shadow_materialize_elapsed_ms,
            post_shadow_materialize_call_count =
              post_shadow_info$shadow_materialize_call_count,
            post_shadow_materialize_target_count =
              post_shadow_info$shadow_materialize_target_count,
            post_shadow_d2h_bytes = post_shadow_info$shadow_d2h_bytes,
            output_slot_release_count =
              released_info$output_slot_release_count,
            output_slot_leased_after_release =
              prepared_after_release$output_slot_leased,
            stringsAsFactors = FALSE
          )
          list(
            setup_record = setup_record,
            batch_record = batch_record,
            target_records = target_records
          )
        }

        batch_results <- lapply(seq_along(setup_keys), run_batch)
        setup_records <- do.call(
          rbind, lapply(batch_results, `[[`, "setup_record")
        )
        batch_records <- do.call(
          rbind, lapply(batch_results, `[[`, "batch_record")
        )
        target_records <- do.call(
          rbind, lapply(batch_results, `[[`, "target_records")
        )
        rownames(setup_records) <- NULL
        rownames(batch_records) <- NULL
        rownames(target_records) <- NULL

        runtime_final <- fixed_sp_cuda_runtime_info(runtime)
        fastkpc_full_cuda_fixed_sp_phase3c_validate_runtime_info(
          runtime_final, "Phase 3C final runtime info"
        )
        fastkpc_full_cuda_fixed_sp_phase3b_validate_runtime_identity(
          runtime_created, runtime_reserved, runtime_final, device_id
        )
        final_workspace_delta <-
          fastkpc_full_cuda_fixed_sp_phase3b_counter_delta(
            runtime_reserved, runtime_final, "workspace_grow_count",
            "Phase 3C final runtime"
          )
        final_stable_workspace_delta <-
          fastkpc_full_cuda_fixed_sp_phase3b_counter_delta(
            runtime_reserved, runtime_final, "stable_workspace_grow_count",
            "Phase 3C final runtime"
          )
        final_synchronize_delta <-
          fastkpc_full_cuda_fixed_sp_phase3b_counter_delta(
            runtime_reserved, runtime_final,
            "cuda_device_synchronize_count", "Phase 3C final runtime"
          )
        checkpoint_fields <- c(
          "cholesky_factor_checkpoint_record_count",
          "cholesky_factor_checkpoint_wait_count",
          "cholesky_solve_checkpoint_record_count",
          "cholesky_solve_checkpoint_wait_count",
          "qr_checkpoint_record_count", "qr_checkpoint_wait_count",
          "svd_checkpoint_record_count", "svd_checkpoint_wait_count"
        )
        checkpoint_batch_fields <- checkpoint_fields
        checkpoint_lifecycle_exact <- all(vapply(
          seq_along(checkpoint_fields), function(index) {
            field <- checkpoint_fields[[index]]
            delta <- fastkpc_full_cuda_fixed_sp_phase3b_counter_delta(
              runtime_reserved, runtime_final, field,
              "Phase 3C final runtime"
            )
            identical(
              delta,
              as.integer(sum(batch_records[[checkpoint_batch_fields[[index]]]]))
            )
          }, logical(1L)
        ))
        lifecycle_exact <-
          runtime_final$workspace_reserve_count ==
            runtime_reserved$workspace_reserve_count &&
          isTRUE(runtime_final$cublas_user_workspace_installed) &&
          runtime_final$cublas_workspace_alignment >= 256 &&
          runtime_final$augmented_workspace_bytes ==
            as.double(8 * 415L * 64L) &&
          runtime_final$aggregate_factor_workspace_bytes ==
            as.double(8 * (64L * 64L + 2L * 64L)) &&
          final_workspace_delta ==
            sum(batch_records$workspace_grow_count_after_warmup) &&
          final_stable_workspace_delta ==
            sum(batch_records$stable_workspace_grow_count_after_warmup) &&
          final_synchronize_delta ==
            sum(batch_records$cuda_device_synchronize_count) &&
          isTRUE(checkpoint_lifecycle_exact)
        if (!isTRUE(lifecycle_exact)) {
          stop("Phase 3C runtime lifecycle does not match batch records",
               call. = FALSE)
        }
        runtime_records <- do.call(rbind, list(
          fastkpc_full_cuda_fixed_sp_phase3c_runtime_record(
            "runtime-created", runtime_created
          ),
          fastkpc_full_cuda_fixed_sp_phase3c_runtime_record(
            "workspace-reserved", runtime_reserved
          ),
          fastkpc_full_cuda_fixed_sp_phase3c_runtime_record(
            "final", runtime_final
          )
        ))
        rownames(runtime_records) <- NULL
        summary <- fastkpc_full_cuda_fixed_sp_phase3c_summarize(
          catalog_records, runtime_records, setup_records,
          batch_records, target_records, scope = scope
        )

        list(
          catalog_records = catalog_records,
          runtime_records = runtime_records,
          setup_records = setup_records,
          batch_records = batch_records,
          target_records = target_records,
          summary = summary
        )
      },
      operations = runtime_cleanup_operations,
      context = paste("Phase 3C", scope, "cleanup")
    )
  if (!isTRUE(return_residual_registry)) return(iteration_result)
  captured_keys <-
    fastkpc_full_cuda_fixed_sp_residual_registry_validate(
      residual_registry
    )
  if (!identical(captured_keys, ordered_target_keys)) {
    stop("Phase 3C qualification residual registry corpus mismatch",
         call. = FALSE)
  }
  list(result = iteration_result, residual_registry = residual_registry)
}

fastkpc_run_full_cuda_fixed_sp_phase3c_qualification <- function(
    phase2_dir, census_dir, prepared_dir, data_path, device_id = 0L,
    return_residual_registry = FALSE) {
  fastkpc_run_full_cuda_fixed_sp_phase3c_iteration(
    phase2_dir = phase2_dir,
    census_dir = census_dir,
    prepared_dir = prepared_dir,
    data_path = data_path,
    device_id = device_id,
    scope = "qualification",
    return_residual_registry = return_residual_registry
  )
}

fastkpc_full_cuda_fixed_sp_oracle_row_schemas <- function() {
  list(
    setup_results = c(
      prepared_s_key_sha256 = "character", shard_id = "integer",
      setup_ordinal = "integer", canonical_setup_rank = "integer",
      phase2_shard_id = "integer",
      phase2_shard_load_count = "integer",
      phase2_shard_authentication_count = "integer", n = "integer",
      coefficient_dim = "integer", null_dim = "integer",
      penalty_count = "integer", target_count = "integer",
      target_key_set_sha256 = "character",
      prepared_handle_create_count = "integer",
      prepared_handle_destroy_count = "integer",
      setup_h2d_upload_count = "integer", setup_h2d_bytes = "double",
      penalty_root_build_count = "integer",
      penalty_root_rank_mismatch_count = "integer",
      penalty_root_matrix_count = "integer",
      penalty_root_row_count = "integer",
      planned_cholesky_target_count = "integer",
      planned_qr_target_count = "integer",
      planned_svd_target_count = "integer",
      executed_cholesky_target_count = "integer",
      executed_qr_target_count = "integer",
      executed_svd_target_count = "integer",
      cholesky_to_svd_count = "integer", qr_to_svd_count = "integer",
      stable_reroute_count = "integer",
      true_batched_target_count = "integer",
      output_slot_leased_after_release = "logical",
      setup_load_elapsed_ms = "double", total_elapsed_ms = "double"
    ),
    target_parity = c(
      prepared_s_key_sha256 = "character", shard_id = "integer",
      setup_ordinal = "integer", canonical_setup_rank = "integer",
      target_ordinal = "integer", canonical_target_rank = "integer",
      residual_key_sha256 = "character", target = "integer",
      null_dim = "integer", condition = "double",
      condition_bucket = "character",
      phase1_coefficient_rank = "integer",
      planned_route = "character",
      authenticated_planned_route = "character",
      executed_route = "character", reroute_reason = "character",
      solver_status = "character", target_true_batched = "logical",
      true_batched_kernel = "logical",
      true_batched_target_count = "integer",
      cholesky_to_svd_count = "integer", qr_to_svd_count = "integer",
      stable_reroute_count = "integer", qr_rank = "integer",
      geqrf_info = "integer", ormqr_info = "integer",
      effective_rank = "integer", sigma_max = "double",
      smallest_retained_sigma = "double", svd_info = "integer",
      aggregate_penalty_root_rank = "integer",
      aggregate_penalty_root_pivot_sha256 = "character",
      aggregate_factor_call_count = "integer",
      aggregate_b_build_count = "integer", aggregate_dstop = "double",
      numeric_reference = "character",
      coefficient_all_finite = "logical", fitted_all_finite = "logical",
      residual_all_finite = "logical", rss_all_finite = "logical",
      rhs_all_finite = "logical", output_all_finite = "logical",
      coefficient_max_abs_diff = "double",
      coefficient_relative_l2 = "double",
      fitted_max_abs_diff = "double", fitted_relative_l2 = "double",
      residual_max_abs_diff = "double",
      residual_relative_l2 = "double", rss_max_abs_diff = "double",
      rss_relative_l2 = "double", rhs_max_abs_diff = "double",
      rhs_relative_l2 = "double",
      coefficient_candidate_sha256 = "character",
      coefficient_oracle_sha256 = "character",
      coefficient_phase2_sha256 = "character",
      coefficient_oracle_phase2_exact = "logical",
      fitted_candidate_sha256 = "character",
      fitted_oracle_sha256 = "character",
      fitted_phase2_sha256 = "character",
      fitted_oracle_phase2_exact = "logical",
      residual_candidate_sha256 = "character",
      residual_oracle_sha256 = "character",
      residual_phase2_sha256 = "character",
      residual_oracle_phase2_exact = "logical",
      rss_candidate_sha256 = "character", rss_oracle_sha256 = "character",
      rhs_candidate_sha256 = "character", rhs_oracle_sha256 = "character",
      selected_sp_sha256 = "character",
      target_fit_fingerprint = "character", y_sha256 = "character",
      target_state_fingerprint = "character", oracle_call_count = "integer",
      rhs_authority = "character", full_cuda_data_plane = "logical",
      cpu_fallback_count = "integer", unknown_fallback_count = "integer",
      approximate_backend = "logical", fallback_type = "character",
      error_code = "character", error_message_sha256 = "character"
    ),
    resource_metrics = c(
      prepared_s_key_sha256 = "character", shard_id = "integer",
      setup_ordinal = "integer", canonical_setup_rank = "integer",
      target_count = "integer", phase2_shard_load_count = "integer",
      phase2_shard_authentication_count = "integer",
      prepared_handle_create_count = "integer",
      prepared_handle_destroy_count = "integer",
      residual_token_acquire_count = "integer",
      residual_token_release_count = "integer",
      output_slot_acquire_count = "integer",
      output_slot_release_count = "integer",
      output_slot_leased_after_release = "logical",
      setup_h2d_upload_count = "integer", setup_h2d_bytes = "double",
      target_batch_h2d_call_count = "integer",
      target_h2d_copy_count = "integer", target_h2d_bytes = "double",
      rhs_device_build_count = "integer", rhs_authority = "character",
      full_cuda_data_plane = "logical",
      coefficient_batch_finalize_call_count = "integer",
      fitted_batch_finalize_call_count = "integer",
      residual_rss_batch_finalize_call_count = "integer",
      per_target_output_finalize_call_count = "integer",
      batch_output_finalized_target_count = "integer",
      true_batched_subgroup_count = "integer",
      true_batched_attempted_target_count = "integer",
      true_batched_target_count = "integer",
      planned_cholesky_target_count = "integer",
      planned_qr_target_count = "integer",
      planned_svd_target_count = "integer",
      executed_cholesky_target_count = "integer",
      executed_qr_target_count = "integer",
      executed_svd_target_count = "integer",
      cholesky_to_svd_count = "integer", qr_to_svd_count = "integer",
      stable_reroute_count = "integer",
      aggregate_penalty_factor_count = "integer",
      aggregate_svd_b_build_count = "integer",
      aggregate_penalty_root_d2h_count = "integer",
      aggregate_penalty_root_d2h_bytes = "double",
      resource_allocation_count_before_solve = "integer",
      resource_allocation_count_after_solve = "integer",
      resource_handle_create_count_before_solve = "integer",
      resource_handle_create_count_after_solve = "integer",
      cuda_device_allocation_count_during_solve = "integer",
      cuda_host_allocation_count_during_solve = "integer",
      stream_create_count_during_solve = "integer",
      event_create_count_during_solve = "integer",
      cublas_handle_create_count_during_solve = "integer",
      cusolver_handle_create_count_during_solve = "integer",
      per_target_allocation_count_after_warmup = "integer",
      per_target_handle_create_count = "integer",
      workspace_grow_count_after_warmup = "integer",
      stable_workspace_grow_count_after_warmup = "integer",
      cuda_device_synchronize_count = "integer",
      cholesky_factor_checkpoint_record_count = "integer",
      cholesky_factor_checkpoint_wait_count = "integer",
      cholesky_solve_checkpoint_record_count = "integer",
      cholesky_solve_checkpoint_wait_count = "integer",
      qr_checkpoint_record_count = "integer",
      qr_checkpoint_wait_count = "integer",
      svd_checkpoint_record_count = "integer",
      svd_checkpoint_wait_count = "integer",
      implicit_residual_d2h_count = "integer",
      implicit_residual_d2h_bytes = "double",
      shadow_materialize_call_count = "integer",
      shadow_materialize_target_count = "integer",
      shadow_d2h_bytes = "double", invalid_output_init_count = "integer",
      nonfinite_output_count = "integer", cpu_fallback_count = "integer",
      unknown_fallback_count = "integer",
      approximate_backend_count = "integer",
      cusolver_deterministic_mode = "character",
      cublas_math_mode = "character", cublas_atomics_mode = "character",
      cublas_user_workspace_installed = "logical",
      cublas_workspace_bytes = "double",
      cublas_workspace_alignment = "double"
    ),
    stage_timing = c(
      prepared_s_key_sha256 = "character", shard_id = "integer",
      setup_ordinal = "integer", stage = "character",
      elapsed_ms = "double"
    )
  )
}

fastkpc_full_cuda_fixed_sp_oracle_empty_frame <- function(name) {
  schemas <- fastkpc_full_cuda_fixed_sp_oracle_row_schemas()
  if (typeof(name) != "character" || length(name) != 1L || is.na(name) ||
      !name %in% names(schemas)) {
    stop("unknown fixed-sp oracle row schema", call. = FALSE)
  }
  schema <- schemas[[name]]
  columns <- lapply(unname(schema), function(type) switch(
    type,
    character = character(), integer = integer(), double = double(),
    logical = logical(), stop("unsupported oracle row type", call. = FALSE)
  ))
  names(columns) <- names(schema)
  as.data.frame(columns, stringsAsFactors = FALSE, optional = TRUE)
}

fastkpc_full_cuda_fixed_sp_validate_oracle_frame <- function(value, name) {
  schemas <- fastkpc_full_cuda_fixed_sp_oracle_row_schemas()
  if (!name %in% names(schemas) || !is.data.frame(value) ||
      !identical(names(value), names(schemas[[name]])) ||
      any(!vapply(names(value), function(field) {
        identical(typeof(value[[field]]), unname(schemas[[name]][[field]])) &&
          !is.object(value[[field]])
      }, logical(1L)))) {
    stop("fixed-sp oracle ", name, " row schema mismatch", call. = FALSE)
  }
  value
}

fastkpc_full_cuda_fixed_sp_oracle_setup_batch <- function(
    catalog, setup_key, target_rows, batch = NULL) {
  if (!fastkpc_full_cuda_fixed_sp_is_bare_sha256(setup_key) ||
      !is.data.frame(target_rows) || nrow(target_rows) < 1L ||
      !all(c(
        "prepared_s_key_sha256", "residual_key_sha256", "coefficient_rank",
        "condition", "planned_route"
      ) %in% names(target_rows))) {
    stop("fixed-sp oracle setup selection is malformed", call. = FALSE)
  }
  setup_index <- catalog$setup_index
  setup_rank <- match(setup_key, setup_index$prepared_s_key_sha256)
  if (is.na(setup_rank) ||
      !identical(
        as.character(target_rows$prepared_s_key_sha256),
        rep(setup_key, nrow(target_rows))
      )) {
    stop("fixed-sp oracle setup identity mismatch", call. = FALSE)
  }
  target_keys <- as.character(target_rows$residual_key_sha256)
  if (anyNA(target_keys) || anyDuplicated(target_keys) ||
      !identical(target_keys, sort(target_keys, method = "radix"))) {
    stop("fixed-sp oracle target order is not canonical", call. = FALSE)
  }
  phase1_targets <- catalog$inputs$target_fit_metadata
  phase2_shard_id <- as.integer(
    (setup_rank - 1L) %% catalog$catalog_contract$shard_count
  )
  if (is.null(batch)) {
    selected_scope <- list(
      setup_rows = data.frame(
        prepared_s_key_sha256 = setup_key, stringsAsFactors = FALSE
      ),
      target_rows = target_rows,
      shard_ids = phase2_shard_id
    )
    batches <- fastkpc_full_cuda_fixed_sp_batches(catalog, selected_scope)
    if (!identical(names(batches), setup_key) || length(batches) != 1L) {
      stop("fixed-sp oracle Phase 2 shard selection is incomplete",
           call. = FALSE)
    }
    batch <- batches[[1L]]
  } else if (!is.list(batch) ||
             !identical(batch$prepared_s_key_sha256, setup_key) ||
             !is.data.frame(batch$target_rows) ||
             !identical(
               as.character(batch$target_rows$residual_key_sha256),
               target_keys
             )) {
    stop("fixed-sp oracle preloaded setup batch is inconsistent",
         call. = FALSE)
  }
  dto <- fastkpc_full_cuda_fixed_sp_native_dto(batch$setup)
  native <- fastkpc_full_cuda_fixed_sp_native_batch(batch, dto)
  canonical_target_keys <- sort(
    as.character(phase1_targets$residual_key_sha256), method = "radix"
  )
  canonical_target_rank <- match(native$target_keys, canonical_target_keys)
  if (anyNA(canonical_target_rank)) {
    stop("fixed-sp oracle canonical target rank is incomplete", call. = FALSE)
  }
  list(
    batch = batch, dto = dto, native = native,
    canonical_setup_rank = as.integer(setup_rank),
    canonical_target_rank = as.integer(canonical_target_rank),
    phase2_shard_id = phase2_shard_id
  )
}

fastkpc_full_cuda_fixed_sp_execute_oracle_setup <- function(
    context, catalog, setup_key, target_rows, shard_id, setup_ordinal,
    selected = NULL, phase2_shard_load_count = NULL,
    phase2_shard_authentication_count = NULL,
    phase2_shard_load_elapsed_ms = NULL, shadow_callback = NULL) {
  if (!is.null(shadow_callback) && !is.function(shadow_callback)) {
    stop("fixed-sp oracle shadow_callback must be NULL or a function",
         call. = FALSE)
  }
  shard_id <- fastkpc_full_cuda_fixed_sp_phase3b_integer_scalar(
    shard_id, "fixed-sp oracle shard_id"
  )
  setup_ordinal <- fastkpc_full_cuda_fixed_sp_phase3b_integer_scalar(
    setup_ordinal, "fixed-sp oracle setup_ordinal"
  )
  if (setup_ordinal < 1L) {
    stop("fixed-sp oracle setup_ordinal must be positive", call. = FALSE)
  }
  total_start <- proc.time()[["elapsed"]]
  load_start <- total_start
  if (is.null(selected)) {
    selected <- fastkpc_full_cuda_fixed_sp_oracle_setup_batch(
      catalog, setup_key, target_rows
    )
    setup_load_elapsed_ms <- as.double(
      (proc.time()[["elapsed"]] - load_start) * 1000
    )
    phase2_shard_load_count <- 1L
    phase2_shard_authentication_count <- 1L
  } else {
    setup_load_elapsed_ms <- phase2_shard_load_elapsed_ms
  }
  phase2_shard_load_count <-
    fastkpc_full_cuda_fixed_sp_phase3b_integer_scalar(
      phase2_shard_load_count, "fixed-sp oracle Phase 2 shard load count"
    )
  phase2_shard_authentication_count <-
    fastkpc_full_cuda_fixed_sp_phase3b_integer_scalar(
      phase2_shard_authentication_count,
      "fixed-sp oracle Phase 2 shard authentication count"
    )
  setup_load_elapsed_ms <-
    fastkpc_full_cuda_fixed_sp_phase3b_double_scalar(
      setup_load_elapsed_ms, "fixed-sp oracle Phase 2 shard load elapsed"
    )
  if (phase2_shard_load_count < 0L ||
      phase2_shard_authentication_count < 0L ||
      phase2_shard_load_count != phase2_shard_authentication_count ||
      setup_load_elapsed_ms < 0) {
    stop("fixed-sp oracle Phase 2 shard load evidence is invalid",
         call. = FALSE)
  }
  batch <- selected$batch
  dto <- selected$dto
  native <- selected$native
  target_count <- native$target_count
  full_outputs <- c("coefficients", "fitted", "residuals", "rss", "rhs")
  expected_shadow_bytes <- 8 * as.double(target_count) * as.double(
    dto$coefficient_dim + 2L * dto$n + 1L + dto$null_dim
  )

  runtime_initial <- fixed_sp_cuda_runtime_info(context)
  fastkpc_full_cuda_fixed_sp_phase3c_validate_runtime_info(
    runtime_initial, paste("fixed-sp oracle setup", setup_ordinal, "runtime")
  )
  handle <- NULL
  token <- NULL
  handle_freed <- FALSE
  token_released <- FALSE
  token_freed <- FALSE
  prepared_created <- NULL
  prepared_before_solve <- NULL
  prepared_after_solve <- NULL
  prepared_after_release <- NULL
  pre_shadow_info <- NULL
  post_shadow_info <- NULL
  released_info <- NULL
  runtime_before_solve <- NULL
  runtime_after_solve <- NULL
  comparison <- NULL
  shadow_callback_result <- NULL
  handle_create_elapsed_ms <- NA_real_
  solve_elapsed_ms <- NA_real_
  shadow_elapsed_ms <- NA_real_
  oracle_elapsed_ms <- NA_real_
  cleanup_start <- NA_real_

  cleanup_operations <- list(
    token_release = list(
      needed = function() !is.null(token) && !token_released,
      run = function() {
        fixed_sp_cuda_residual_release(token)
        token_released <<- TRUE
        released_info <<- fixed_sp_cuda_residual_info(token)
        fastkpc_full_cuda_fixed_sp_phase3c_validate_batch_info(
          released_info, native, dto, 1L, target_count,
          expected_shadow_bytes, 1L,
          paste("fixed-sp oracle setup", setup_ordinal, "released batch"),
          expected_coefficients = TRUE
        )
        prepared_after_release <<- fixed_sp_cuda_prepared_info(handle)
        fastkpc_full_cuda_fixed_sp_phase3c_validate_prepared_info(
          prepared_after_release, dto, 0L,
          paste("fixed-sp oracle setup", setup_ordinal, "released handle")
        )
        if (isTRUE(prepared_after_release$output_slot_leased)) {
          stop("fixed-sp oracle output-slot lease was not released",
               call. = FALSE)
        }
      }
    ),
    token_free = list(
      needed = function() !is.null(token) && !token_freed,
      run = function() {
        fixed_sp_cuda_residual_free(token)
        token_freed <<- TRUE
        token_released <<- TRUE
        token <<- NULL
      }
    ),
    handle_free = list(
      needed = function() !is.null(handle) && !handle_freed,
      run = function() {
        fixed_sp_cuda_prepared_free(handle)
        handle_freed <<- TRUE
        handle <<- NULL
      }
    ),
    runtime_free = list(
      needed = function() FALSE,
      run = function() invisible(NULL)
    )
  )

  fastkpc_full_cuda_fixed_sp_phase3b_run_cleanup_scope(
    body = {
      handle_start <- proc.time()[["elapsed"]]
      handle <- fixed_sp_cuda_prepared_create(context, dto)
      handle_create_elapsed_ms <- as.double(
        (proc.time()[["elapsed"]] - handle_start) * 1000
      )
      prepared_created <- fixed_sp_cuda_prepared_info(handle)
      fastkpc_full_cuda_fixed_sp_phase3c_validate_prepared_info(
        prepared_created, dto, 0L,
        paste("fixed-sp oracle setup", setup_ordinal, "created handle")
      )
      if (prepared_created$setup_h2d_upload_count != 1L ||
          isTRUE(prepared_created$output_slot_leased)) {
        stop("fixed-sp oracle prepared setup lifecycle is invalid",
             call. = FALSE)
      }
      prepared_before_solve <- fixed_sp_cuda_prepared_info(handle)
      runtime_before_solve <- fixed_sp_cuda_runtime_info(context)
      solve_start <- proc.time()[["elapsed"]]
      token <- fixed_sp_cuda_solve_batch(
        handle, native$Y, native$SP, native$planned_route,
        native$target_keys, outputs = full_outputs
      )
      solve_elapsed_ms <- as.double(
        (proc.time()[["elapsed"]] - solve_start) * 1000
      )
      pre_shadow_info <- fixed_sp_cuda_residual_info(token)
      fastkpc_full_cuda_fixed_sp_phase3c_validate_batch_info(
        pre_shadow_info, native, dto, 0L, 0L, 0, 0L,
        paste("fixed-sp oracle setup", setup_ordinal, "pre-shadow batch"),
        expected_coefficients = TRUE
      )
      prepared_after_solve <- fixed_sp_cuda_prepared_info(handle)
      if (!isTRUE(prepared_after_solve$output_slot_leased)) {
        stop("fixed-sp oracle solve did not retain its output-slot lease",
             call. = FALSE)
      }
      runtime_after_solve <- fixed_sp_cuda_runtime_info(context)
      shadow_start <- proc.time()[["elapsed"]]
      shadow <- fixed_sp_cuda_materialize_shadow(
        token, outputs = full_outputs
      )
      shadow_elapsed_ms <- as.double(
        (proc.time()[["elapsed"]] - shadow_start) * 1000
      )
      shadow_clean <- is.list(shadow) && identical(
        names(shadow),
        c("coefficients", "fitted", "residuals", "rss", "cuda_nullspace_rhs")
      ) && is.matrix(shadow$coefficients) &&
        identical(dim(shadow$coefficients), c(dto$coefficient_dim, target_count)) &&
        is.matrix(shadow$fitted) &&
        identical(dim(shadow$fitted), c(dto$n, target_count)) &&
        is.matrix(shadow$residuals) &&
        identical(dim(shadow$residuals), c(dto$n, target_count)) &&
        typeof(shadow$rss) == "double" && length(shadow$rss) == target_count &&
        is.matrix(shadow$cuda_nullspace_rhs) && identical(
          dim(shadow$cuda_nullspace_rhs), c(dto$null_dim, target_count)
        )
      if (!isTRUE(shadow_clean)) {
        stop("fixed-sp oracle explicit shadow is malformed", call. = FALSE)
      }
      post_shadow_info <- fixed_sp_cuda_residual_info(token)
      fastkpc_full_cuda_fixed_sp_phase3c_validate_batch_info(
        post_shadow_info, native, dto, 1L, target_count,
        expected_shadow_bytes, 0L,
        paste("fixed-sp oracle setup", setup_ordinal, "post-shadow batch"),
        expected_coefficients = TRUE
      )

      output_names <- c("coefficient", "fitted", "residual", "rss", "rhs")
      max_abs <- relative_l2 <- setNames(
        lapply(output_names, function(name) double(target_count)), output_names
      )
      candidate_hash <- oracle_hash <- setNames(
        lapply(output_names, function(name) character(target_count)),
        output_names
      )
      finite <- setNames(
        lapply(output_names, function(name) logical(target_count)), output_names
      )
      oracle_call_count <- integer(target_count)
      approximate_backend <- logical(target_count)
      coefficient_phase2_exact <- fitted_phase2_exact <-
        residual_phase2_exact <- logical(target_count)
      oracle_start <- proc.time()[["elapsed"]]
      for (target_index in seq_len(target_count)) {
        candidate_values <- list(
          coefficient = as.numeric(shadow$coefficients[, target_index]),
          fitted = as.numeric(shadow$fitted[, target_index]),
          residual = as.numeric(shadow$residuals[, target_index]),
          rss = as.double(shadow$rss[[target_index]]),
          rhs = as.numeric(shadow$cuda_nullspace_rhs[, target_index])
        )
        oracle_call_count[[target_index]] <- 1L
        oracle <- fastkpc_mgcv_magic_fixed_sp_from_prepared(
          prepared_setup = batch$setup,
          target_state = list(
            row = batch$target_rows[target_index, , drop = FALSE],
            y = as.numeric(native$Y[, target_index])
          )
        )
        approximate_backend[[target_index]] <- !(
          identical(oracle$backend_family, "mgcvExtractCPU") &&
            identical(oracle$mode, "prepared-s-fixed-sp-mgcv-reference") &&
            identical(oracle$solve_source, "mgcv-C-magic-from-prepared-s") &&
            isTRUE(oracle$authoritative)
        )
        oracle_values <- list(
          coefficient = as.numeric(oracle$coefficients),
          fitted = as.numeric(oracle$fitted),
          residual = as.numeric(oracle$residuals),
          rss = as.double(sum(oracle$residuals^2)),
          rhs = as.numeric(batch$oracle_nullspace_rhs[, target_index])
        )
        for (name in output_names) {
          finite[[name]][[target_index]] <-
            all(is.finite(candidate_values[[name]])) &&
            all(is.finite(oracle_values[[name]]))
          errors <- fastkpc_full_cuda_fixed_sp_phase3b_numeric_errors(
            candidate_values[[name]], oracle_values[[name]],
            paste("fixed-sp oracle", name, native$target_keys[[target_index]])
          )
          max_abs[[name]][[target_index]] <- errors[["max_abs"]]
          relative_l2[[name]][[target_index]] <- errors[["relative_l2"]]
          candidate_hash[[name]][[target_index]] <-
            fastkpc_full_cuda_census_metadata_hash(candidate_values[[name]])
          oracle_hash[[name]][[target_index]] <-
            fastkpc_full_cuda_census_metadata_hash(oracle_values[[name]])
        }
        coefficient_phase2_exact[[target_index]] <- identical(
          oracle_hash$coefficient[[target_index]],
          as.character(batch$target_rows$coefficient_hash[[target_index]])
        )
        fitted_phase2_exact[[target_index]] <- identical(
          oracle_hash$fitted[[target_index]],
          as.character(batch$target_rows$fitted_hash[[target_index]])
        )
        residual_phase2_exact[[target_index]] <- identical(
          oracle_hash$residual[[target_index]],
          as.character(batch$target_rows$residual_hash[[target_index]])
        )
        if (!all(vapply(finite, `[[`, logical(1L), target_index)) ||
            max_abs$fitted[[target_index]] >= 1e-7 ||
            relative_l2$fitted[[target_index]] >= 1e-7 ||
            max_abs$residual[[target_index]] >= 1e-7 ||
            relative_l2$residual[[target_index]] >= 1e-7 ||
            max_abs$rhs[[target_index]] >= 1e-12 ||
            relative_l2$rhs[[target_index]] >= 1e-12 ||
            !coefficient_phase2_exact[[target_index]] ||
            !fitted_phase2_exact[[target_index]] ||
            !residual_phase2_exact[[target_index]] ||
            approximate_backend[[target_index]]) {
          stop("fixed-sp oracle target comparison failed: ",
               native$target_keys[[target_index]], call. = FALSE)
        }
      }
      oracle_elapsed_ms <- as.double(
        (proc.time()[["elapsed"]] - oracle_start) * 1000
      )
      comparison <- list(
        max_abs = max_abs, relative_l2 = relative_l2,
        candidate_hash = candidate_hash, oracle_hash = oracle_hash,
        finite = finite, oracle_call_count = oracle_call_count,
        approximate_backend = approximate_backend,
        coefficient_phase2_exact = coefficient_phase2_exact,
        fitted_phase2_exact = fitted_phase2_exact,
        residual_phase2_exact = residual_phase2_exact
      )
      if (!is.null(shadow_callback)) {
        callback_started <- proc.time()[["elapsed"]]
        shadow_callback_result <- shadow_callback(
          setup_key = setup_key, target_keys = native$target_keys,
          residuals = shadow$residuals
        )
        compact_callback_result <- is.null(shadow_callback_result) ||
          (is.data.frame(shadow_callback_result) &&
             as.numeric(object.size(shadow_callback_result)) <= 67108864)
        if (!isTRUE(compact_callback_result)) {
          stop("fixed-sp oracle shadow callback result is not compact",
               call. = FALSE)
        }
        oracle_elapsed_ms <- oracle_elapsed_ms + as.double(
          (proc.time()[["elapsed"]] - callback_started) * 1000
        )
      }
      rm(shadow)
      cleanup_start <- proc.time()[["elapsed"]]
      invisible(NULL)
    },
    operations = cleanup_operations,
    context = paste("fixed-sp oracle setup", setup_ordinal, "cleanup")
  )

  if (!isTRUE(handle_freed) || !isTRUE(token_freed) ||
      !isTRUE(token_released) || is.null(comparison) ||
      is.null(prepared_after_release) || is.null(released_info)) {
    stop("fixed-sp oracle setup cleanup evidence is incomplete",
         call. = FALSE)
  }
  cleanup_elapsed_ms <- as.double(
    (proc.time()[["elapsed"]] - cleanup_start) * 1000
  )
  runtime_final <- fixed_sp_cuda_runtime_info(context)
  fastkpc_full_cuda_fixed_sp_phase3c_validate_runtime_info(
    runtime_final, paste("fixed-sp oracle setup", setup_ordinal, "final runtime")
  )
  counter_fields <- c(
    "workspace_grow_count", "stable_workspace_grow_count",
    "cuda_device_synchronize_count",
    "cholesky_factor_checkpoint_record_count",
    "cholesky_factor_checkpoint_wait_count",
    "cholesky_solve_checkpoint_record_count",
    "cholesky_solve_checkpoint_wait_count",
    "qr_checkpoint_record_count", "qr_checkpoint_wait_count",
    "svd_checkpoint_record_count", "svd_checkpoint_wait_count"
  )
  counter_deltas <- setNames(vapply(counter_fields, function(field) {
    fastkpc_full_cuda_fixed_sp_phase3b_counter_delta(
      runtime_before_solve, runtime_after_solve, field,
      paste("fixed-sp oracle setup", setup_ordinal)
    )
  }, integer(1L)), counter_fields)
  if (counter_deltas[["workspace_grow_count"]] != 0L ||
      counter_deltas[["stable_workspace_grow_count"]] != 0L ||
      counter_deltas[["cuda_device_synchronize_count"]] != 0L) {
    stop("fixed-sp oracle setup grew workspace or synchronized",
         call. = FALSE)
  }

  condition_bucket <- vapply(seq_len(target_count), function(index) {
    fastkpc_full_cuda_census_condition_bucket(
      batch$condition[[index]], batch$target_rows$coefficient_rank[[index]],
      dto$null_dim
    )
  }, character(1L))
  cholesky_to_svd <- as.integer(
    pre_shadow_info$planned_route == "CHOLESKY_BATCHED" &
      pre_shadow_info$executed_route == "AUGMENTED_SVD"
  )
  qr_to_svd <- as.integer(
    pre_shadow_info$planned_route == "AUGMENTED_QR" &
      pre_shadow_info$executed_route == "AUGMENTED_SVD"
  )
  stable_reroute <- as.integer(
    pre_shadow_info$planned_route != pre_shadow_info$executed_route
  )
  all_finite <- Reduce(`&`, comparison$finite)
  empty_error_hash <- fastkpc_full_cuda_census_hash_utf8("")
  pivot_hash <- vapply(
    pre_shadow_info$aggregate_penalty_root_pivot,
    fastkpc_full_cuda_census_metadata_hash, character(1L)
  )
  target_parity <- data.frame(
    prepared_s_key_sha256 = rep(setup_key, target_count),
    shard_id = rep.int(shard_id, target_count),
    setup_ordinal = rep.int(setup_ordinal, target_count),
    canonical_setup_rank = rep.int(
      selected$canonical_setup_rank, target_count
    ),
    target_ordinal = seq_len(target_count),
    canonical_target_rank = selected$canonical_target_rank,
    residual_key_sha256 = native$target_keys,
    target = as.integer(batch$target_rows$target),
    null_dim = rep.int(dto$null_dim, target_count),
    condition = as.double(batch$condition),
    condition_bucket = condition_bucket,
    phase1_coefficient_rank = as.integer(batch$target_rows$coefficient_rank),
    planned_route = pre_shadow_info$planned_route,
    authenticated_planned_route = native$planned_route,
    executed_route = pre_shadow_info$executed_route,
    reroute_reason = pre_shadow_info$reroute_reason,
    solver_status = pre_shadow_info$solver_status,
    target_true_batched = pre_shadow_info$target_true_batched,
    true_batched_kernel = rep.int(
      pre_shadow_info$true_batched_kernel, target_count
    ),
    true_batched_target_count = rep.int(
      pre_shadow_info$true_batched_target_count, target_count
    ),
    cholesky_to_svd_count = cholesky_to_svd,
    qr_to_svd_count = qr_to_svd,
    stable_reroute_count = stable_reroute,
    qr_rank = pre_shadow_info$qr_rank,
    geqrf_info = pre_shadow_info$geqrf_info,
    ormqr_info = pre_shadow_info$ormqr_info,
    effective_rank = pre_shadow_info$effective_rank,
    sigma_max = pre_shadow_info$sigma_max,
    smallest_retained_sigma = pre_shadow_info$smallest_retained_sigma,
    svd_info = pre_shadow_info$svd_info,
    aggregate_penalty_root_rank =
      pre_shadow_info$aggregate_penalty_root_rank,
    aggregate_penalty_root_pivot_sha256 = pivot_hash,
    aggregate_factor_call_count =
      pre_shadow_info$aggregate_factor_call_count,
    aggregate_b_build_count = pre_shadow_info$aggregate_b_build_count,
    aggregate_dstop = pre_shadow_info$aggregate_dstop,
    numeric_reference = rep("mgcv-fixed-sp", target_count),
    coefficient_all_finite = comparison$finite$coefficient,
    fitted_all_finite = comparison$finite$fitted,
    residual_all_finite = comparison$finite$residual,
    rss_all_finite = comparison$finite$rss,
    rhs_all_finite = comparison$finite$rhs,
    output_all_finite = all_finite,
    coefficient_max_abs_diff = comparison$max_abs$coefficient,
    coefficient_relative_l2 = comparison$relative_l2$coefficient,
    fitted_max_abs_diff = comparison$max_abs$fitted,
    fitted_relative_l2 = comparison$relative_l2$fitted,
    residual_max_abs_diff = comparison$max_abs$residual,
    residual_relative_l2 = comparison$relative_l2$residual,
    rss_max_abs_diff = comparison$max_abs$rss,
    rss_relative_l2 = comparison$relative_l2$rss,
    rhs_max_abs_diff = comparison$max_abs$rhs,
    rhs_relative_l2 = comparison$relative_l2$rhs,
    coefficient_candidate_sha256 = comparison$candidate_hash$coefficient,
    coefficient_oracle_sha256 = comparison$oracle_hash$coefficient,
    coefficient_phase2_sha256 = as.character(
      batch$target_rows$coefficient_hash
    ),
    coefficient_oracle_phase2_exact = comparison$coefficient_phase2_exact,
    fitted_candidate_sha256 = comparison$candidate_hash$fitted,
    fitted_oracle_sha256 = comparison$oracle_hash$fitted,
    fitted_phase2_sha256 = as.character(batch$target_rows$fitted_hash),
    fitted_oracle_phase2_exact = comparison$fitted_phase2_exact,
    residual_candidate_sha256 = comparison$candidate_hash$residual,
    residual_oracle_sha256 = comparison$oracle_hash$residual,
    residual_phase2_sha256 = as.character(batch$target_rows$residual_hash),
    residual_oracle_phase2_exact = comparison$residual_phase2_exact,
    rss_candidate_sha256 = comparison$candidate_hash$rss,
    rss_oracle_sha256 = comparison$oracle_hash$rss,
    rhs_candidate_sha256 = comparison$candidate_hash$rhs,
    rhs_oracle_sha256 = comparison$oracle_hash$rhs,
    selected_sp_sha256 = as.character(batch$target_rows$selected_sp_hash),
    target_fit_fingerprint = as.character(
      batch$target_rows$target_fit_fingerprint
    ),
    y_sha256 = as.character(batch$target_rows$y_hash),
    target_state_fingerprint = as.character(
      batch$target_rows$target_state_fingerprint
    ),
    oracle_call_count = comparison$oracle_call_count,
    rhs_authority = rep(pre_shadow_info$rhs_authority, target_count),
    full_cuda_data_plane = rep.int(
      pre_shadow_info$full_cuda_data_plane, target_count
    ),
    cpu_fallback_count = integer(target_count),
    unknown_fallback_count = integer(target_count),
    approximate_backend = comparison$approximate_backend,
    fallback_type = rep("NONE", target_count),
    error_code = rep("NONE", target_count),
    error_message_sha256 = rep(empty_error_hash, target_count),
    stringsAsFactors = FALSE
  )
  target_parity <- fastkpc_full_cuda_fixed_sp_validate_oracle_frame(
    target_parity, "target_parity"
  )

  total_elapsed_ms <- as.double(
    (proc.time()[["elapsed"]] - total_start) * 1000
  )
  setup_results <- data.frame(
    prepared_s_key_sha256 = setup_key, shard_id = shard_id,
    setup_ordinal = setup_ordinal,
    canonical_setup_rank = selected$canonical_setup_rank,
    phase2_shard_id = selected$phase2_shard_id,
    phase2_shard_load_count = phase2_shard_load_count,
    phase2_shard_authentication_count =
      phase2_shard_authentication_count,
    n = dto$n, coefficient_dim = dto$coefficient_dim,
    null_dim = dto$null_dim, penalty_count = dto$penalty_count,
    target_count = target_count,
    target_key_set_sha256 = fastkpc_full_cuda_census_key_set_hash(
      native$target_keys
    ),
    prepared_handle_create_count = 1L,
    prepared_handle_destroy_count = 1L,
    setup_h2d_upload_count = prepared_created$setup_h2d_upload_count,
    setup_h2d_bytes = prepared_created$setup_h2d_bytes,
    penalty_root_build_count = prepared_created$penalty_root_build_count,
    penalty_root_rank_mismatch_count =
      prepared_created$penalty_root_rank_mismatch_count,
    penalty_root_matrix_count = prepared_created$penalty_root_matrix_count,
    penalty_root_row_count = prepared_created$penalty_root_row_count,
    planned_cholesky_target_count =
      pre_shadow_info$planned_cholesky_target_count,
    planned_qr_target_count = pre_shadow_info$planned_qr_target_count,
    planned_svd_target_count = pre_shadow_info$planned_svd_target_count,
    executed_cholesky_target_count =
      pre_shadow_info$executed_cholesky_target_count,
    executed_qr_target_count = pre_shadow_info$executed_qr_target_count,
    executed_svd_target_count = pre_shadow_info$executed_svd_target_count,
    cholesky_to_svd_count = pre_shadow_info$cholesky_to_svd_count,
    qr_to_svd_count = pre_shadow_info$qr_to_svd_count,
    stable_reroute_count = pre_shadow_info$stable_reroute_count,
    true_batched_target_count = pre_shadow_info$true_batched_target_count,
    output_slot_leased_after_release =
      prepared_after_release$output_slot_leased,
    setup_load_elapsed_ms = setup_load_elapsed_ms,
    total_elapsed_ms = total_elapsed_ms,
    stringsAsFactors = FALSE
  )
  setup_results <- fastkpc_full_cuda_fixed_sp_validate_oracle_frame(
    setup_results, "setup_results"
  )

  resource_metrics <- data.frame(
    prepared_s_key_sha256 = setup_key, shard_id = shard_id,
    setup_ordinal = setup_ordinal,
    canonical_setup_rank = selected$canonical_setup_rank,
    target_count = target_count,
    phase2_shard_load_count = phase2_shard_load_count,
    phase2_shard_authentication_count =
      phase2_shard_authentication_count,
    prepared_handle_create_count = 1L,
    prepared_handle_destroy_count = 1L,
    residual_token_acquire_count = 1L,
    residual_token_release_count = released_info$output_slot_release_count,
    output_slot_acquire_count = pre_shadow_info$output_slot_acquire_count,
    output_slot_release_count = released_info$output_slot_release_count,
    output_slot_leased_after_release =
      prepared_after_release$output_slot_leased,
    setup_h2d_upload_count = prepared_created$setup_h2d_upload_count,
    setup_h2d_bytes = prepared_created$setup_h2d_bytes,
    target_batch_h2d_call_count =
      pre_shadow_info$target_batch_h2d_call_count,
    target_h2d_copy_count = pre_shadow_info$target_h2d_copy_count,
    target_h2d_bytes = pre_shadow_info$target_h2d_bytes,
    rhs_device_build_count = pre_shadow_info$rhs_device_build_count,
    rhs_authority = pre_shadow_info$rhs_authority,
    full_cuda_data_plane = pre_shadow_info$full_cuda_data_plane,
    coefficient_batch_finalize_call_count =
      pre_shadow_info$coefficient_batch_finalize_call_count,
    fitted_batch_finalize_call_count =
      pre_shadow_info$fitted_batch_finalize_call_count,
    residual_rss_batch_finalize_call_count =
      pre_shadow_info$residual_rss_batch_finalize_call_count,
    per_target_output_finalize_call_count =
      pre_shadow_info$per_target_output_finalize_call_count,
    batch_output_finalized_target_count =
      pre_shadow_info$batch_output_finalized_target_count,
    true_batched_subgroup_count =
      pre_shadow_info$true_batched_subgroup_count,
    true_batched_attempted_target_count =
      pre_shadow_info$true_batched_attempted_target_count,
    true_batched_target_count = pre_shadow_info$true_batched_target_count,
    planned_cholesky_target_count =
      pre_shadow_info$planned_cholesky_target_count,
    planned_qr_target_count = pre_shadow_info$planned_qr_target_count,
    planned_svd_target_count = pre_shadow_info$planned_svd_target_count,
    executed_cholesky_target_count =
      pre_shadow_info$executed_cholesky_target_count,
    executed_qr_target_count = pre_shadow_info$executed_qr_target_count,
    executed_svd_target_count = pre_shadow_info$executed_svd_target_count,
    cholesky_to_svd_count = pre_shadow_info$cholesky_to_svd_count,
    qr_to_svd_count = pre_shadow_info$qr_to_svd_count,
    stable_reroute_count = pre_shadow_info$stable_reroute_count,
    aggregate_penalty_factor_count =
      pre_shadow_info$aggregate_penalty_factor_count,
    aggregate_svd_b_build_count =
      pre_shadow_info$aggregate_svd_b_build_count,
    aggregate_penalty_root_d2h_count =
      pre_shadow_info$aggregate_penalty_root_d2h_count,
    aggregate_penalty_root_d2h_bytes =
      pre_shadow_info$aggregate_penalty_root_d2h_bytes,
    resource_allocation_count_before_solve =
      pre_shadow_info$resource_allocation_count_before_solve,
    resource_allocation_count_after_solve =
      pre_shadow_info$resource_allocation_count_after_solve,
    resource_handle_create_count_before_solve =
      pre_shadow_info$resource_handle_create_count_before_solve,
    resource_handle_create_count_after_solve =
      pre_shadow_info$resource_handle_create_count_after_solve,
    cuda_device_allocation_count_during_solve =
      pre_shadow_info$cuda_device_allocation_count_during_solve,
    cuda_host_allocation_count_during_solve =
      pre_shadow_info$cuda_host_allocation_count_during_solve,
    stream_create_count_during_solve =
      pre_shadow_info$stream_create_count_during_solve,
    event_create_count_during_solve =
      pre_shadow_info$event_create_count_during_solve,
    cublas_handle_create_count_during_solve =
      pre_shadow_info$cublas_handle_create_count_during_solve,
    cusolver_handle_create_count_during_solve =
      pre_shadow_info$cusolver_handle_create_count_during_solve,
    per_target_allocation_count_after_warmup =
      pre_shadow_info$per_target_allocation_count_after_warmup,
    per_target_handle_create_count =
      pre_shadow_info$per_target_handle_create_count,
    workspace_grow_count_after_warmup =
      counter_deltas[["workspace_grow_count"]],
    stable_workspace_grow_count_after_warmup =
      counter_deltas[["stable_workspace_grow_count"]],
    cuda_device_synchronize_count =
      counter_deltas[["cuda_device_synchronize_count"]],
    cholesky_factor_checkpoint_record_count =
      counter_deltas[["cholesky_factor_checkpoint_record_count"]],
    cholesky_factor_checkpoint_wait_count =
      counter_deltas[["cholesky_factor_checkpoint_wait_count"]],
    cholesky_solve_checkpoint_record_count =
      counter_deltas[["cholesky_solve_checkpoint_record_count"]],
    cholesky_solve_checkpoint_wait_count =
      counter_deltas[["cholesky_solve_checkpoint_wait_count"]],
    qr_checkpoint_record_count =
      counter_deltas[["qr_checkpoint_record_count"]],
    qr_checkpoint_wait_count = counter_deltas[["qr_checkpoint_wait_count"]],
    svd_checkpoint_record_count =
      counter_deltas[["svd_checkpoint_record_count"]],
    svd_checkpoint_wait_count =
      counter_deltas[["svd_checkpoint_wait_count"]],
    implicit_residual_d2h_count =
      released_info$implicit_residual_d2h_count,
    implicit_residual_d2h_bytes = 0,
    shadow_materialize_call_count =
      post_shadow_info$shadow_materialize_call_count,
    shadow_materialize_target_count =
      post_shadow_info$shadow_materialize_target_count,
    shadow_d2h_bytes = post_shadow_info$shadow_d2h_bytes,
    invalid_output_init_count = pre_shadow_info$invalid_output_init_count,
    nonfinite_output_count = pre_shadow_info$nonfinite_output_count,
    cpu_fallback_count = pre_shadow_info$cpu_fallback_count,
    unknown_fallback_count = pre_shadow_info$unknown_fallback_count,
    approximate_backend_count = as.integer(sum(
      comparison$approximate_backend
    )),
    cusolver_deterministic_mode = runtime_final$cusolver_deterministic_mode,
    cublas_math_mode = runtime_final$cublas_math_mode,
    cublas_atomics_mode = runtime_final$cublas_atomics_mode,
    cublas_user_workspace_installed =
      runtime_final$cublas_user_workspace_installed,
    cublas_workspace_bytes = runtime_final$cublas_workspace_bytes,
    cublas_workspace_alignment = runtime_final$cublas_workspace_alignment,
    stringsAsFactors = FALSE
  )
  resource_metrics <- fastkpc_full_cuda_fixed_sp_validate_oracle_frame(
    resource_metrics, "resource_metrics"
  )

  stage_timing <- data.frame(
    prepared_s_key_sha256 = rep(setup_key, 6L),
    shard_id = rep.int(shard_id, 6L),
    setup_ordinal = rep.int(setup_ordinal, 6L),
    stage = c(
      "phase2_shard_load", "prepared_handle_create", "solve",
      "shadow_materialize", "cmagic_oracle", "release_and_free"
    ),
    elapsed_ms = as.double(c(
      setup_load_elapsed_ms, handle_create_elapsed_ms, solve_elapsed_ms,
      shadow_elapsed_ms, oracle_elapsed_ms, cleanup_elapsed_ms
    )),
    stringsAsFactors = FALSE
  )
  stage_timing <- fastkpc_full_cuda_fixed_sp_validate_oracle_frame(
    stage_timing, "stage_timing"
  )
  list(
    setup_results = setup_results, target_parity = target_parity,
    resource_metrics = resource_metrics, stage_timing = stage_timing,
    shadow_callback_result = shadow_callback_result
  )
}

fastkpc_full_cuda_fixed_sp_qualification_payload_names <- function() {
  c(
    "target_parity.rds", "target_parity.csv",
    "batch_metrics.rds", "batch_metrics.csv",
    "setup_metrics.rds", "setup_metrics.csv",
    "qualification_dcov_parity.rds",
    "qualification_dcov_parity.csv",
    "runtime_metrics.csv", "stage_timing.csv", "fallbacks.csv",
    "failures.csv", "native_build_dependencies.csv",
    "native_build_exclusions.csv", "native_build_trace.txt",
    "commands.txt", "environment.txt"
  )
}

fastkpc_full_cuda_fixed_sp_sha256_file <- function(path) {
  if (!is.character(path) || length(path) != 1L || is.na(path) ||
      !nzchar(path) || !file.exists(path) || dir.exists(path)) {
    stop("execution source hash path is invalid", call. = FALSE)
  }
  if (!requireNamespace("digest", quietly = TRUE)) {
    stop("execution source hashing requires digest", call. = FALSE)
  }
  unname(digest::digest(
    file = path, algo = "sha256", serialize = FALSE
  ))
}

fastkpc_full_cuda_fixed_sp_git_source_state <- function(path) {
  status <- suppressWarnings(system2(
    "git",
    c("status", "--porcelain=v1", "--untracked-files=all", "--", path),
    stdout = TRUE, stderr = TRUE
  ))
  exit_status <- attr(status, "status")
  if (!is.null(exit_status) && exit_status != 0L) {
    stop("failed to inspect execution source git state", call. = FALSE)
  }
  if (length(status) == 0L) return("clean")
  if (any(startsWith(status, "??"))) return("untracked")
  "tracked-dirty"
}

fastkpc_full_cuda_fixed_sp_source_call <- function(expression) {
  if (!is.call(expression)) return(NULL)
  head <- expression[[1L]]
  direct <- is.symbol(head) && identical(as.character(head), "source")
  namespaced <- is.call(head) && length(head) == 3L &&
    is.symbol(head[[1L]]) && as.character(head[[1L]]) %in% c("::", ":::") &&
    is.symbol(head[[2L]]) && identical(as.character(head[[2L]]), "base") &&
    is.symbol(head[[3L]]) && identical(as.character(head[[3L]]), "source")
  if (!direct && !namespaced) return(NULL)

  arguments <- as.list(expression)[-1L]
  argument_names <- names(arguments)
  if (is.null(argument_names)) argument_names <- rep.int("", length(arguments))
  named_file <- which(argument_names == "file")
  unnamed <- which(!nzchar(argument_names))
  if (length(named_file) > 1L ||
      (length(named_file) == 1L && length(unnamed) > 0L) ||
      (length(named_file) == 0L && length(unnamed) < 1L)) {
    stop("execution source has an unsupported source() signature",
         call. = FALSE)
  }
  file_argument <- if (length(named_file) == 1L) {
    arguments[[named_file]]
  } else {
    arguments[[unnamed[[1L]]]]
  }
  if (typeof(file_argument) != "character" || length(file_argument) != 1L ||
      is.object(file_argument) || !is.null(attributes(file_argument)) ||
      is.na(file_argument) || !nzchar(file_argument)) {
    stop("execution source contains a dynamic source() call", call. = FALSE)
  }
  as.character(file_argument)
}

fastkpc_full_cuda_fixed_sp_scan_source_ast <- function(expressions) {
  dependencies <- character()
  walk <- function(node, load_time) {
    if (is.call(node)) {
      source_path <- fastkpc_full_cuda_fixed_sp_source_call(node)
      if (!is.null(source_path)) {
        if (load_time) dependencies <<- c(dependencies, source_path)
        return(invisible(NULL))
      }
      head <- node[[1L]]
      if (is.symbol(head) && identical(as.character(head), "function")) {
        formals <- node[[2L]]
        for (formal_name in names(formals)) {
          if (!identical(formals[[formal_name]], quote(expr = ))) {
            walk(formals[[formal_name]], FALSE)
          }
        }
        walk(node[[3L]], FALSE)
        return(invisible(NULL))
      }
      for (index in seq_along(node)) {
        if (!identical(node[[index]], quote(expr = ))) {
          walk(node[[index]], load_time)
        }
      }
    } else if (is.expression(node) || is.pairlist(node)) {
      for (index in seq_along(node)) {
        if (!identical(node[[index]], quote(expr = ))) {
          walk(node[[index]], load_time)
        }
      }
    }
    invisible(NULL)
  }
  walk(expressions, TRUE)
  sort(unique(dependencies), method = "radix")
}

fastkpc_full_cuda_fixed_sp_source_identity <- function(path, project_root) {
  scalar <- typeof(path) == "character" && length(path) == 1L &&
    !is.object(path) && is.null(attributes(path)) && !anyNA(path) &&
    nzchar(path) && !grepl("[\r\n\\]", path)
  if (!isTRUE(scalar)) {
    stop("execution source path identity is malformed", call. = FALSE)
  }
  root <- normalizePath(project_root, winslash = "/", mustWork = TRUE)
  absolute_input <- startsWith(path, "/")
  candidate <- if (absolute_input) path else file.path(root, path)
  if (!file.exists(candidate) || dir.exists(candidate)) {
    stop("execution source path does not exist: ", path, call. = FALSE)
  }
  normalized <- normalizePath(candidate, winslash = "/", mustWork = TRUE)
  root_prefix <- paste0(root, "/")
  if (!startsWith(normalized, root_prefix)) {
    stop("execution source escapes the project root", call. = FALSE)
  }
  source_id <- substring(normalized, nchar(root_prefix) + 1L)
  canonical_input <- if (absolute_input) normalized else path
  expected_input <- if (absolute_input) normalized else source_id
  if (!identical(canonical_input, expected_input) ||
      !nzchar(source_id) || startsWith(source_id, "../") ||
      grepl("(^|/)\\.{1,2}(/|$)", source_id)) {
    stop("execution source identity is ambiguous or noncanonical",
         call. = FALSE)
  }
  list(id = source_id, path = normalized, project_root = root)
}

fastkpc_full_cuda_fixed_sp_validate_source_closure <- function(closure) {
  expected_names <- c(
    "source_closure_schema_version", "source_discovery_semantics",
    "source_project_root", "direct_source_ids", "source_ids",
    "source_file_paths", "source_dependency_map", "source_closure_count"
  )
  scalar_character <- function(value) {
    typeof(value) == "character" && length(value) == 1L &&
      !is.object(value) && is.null(attributes(value)) && !anyNA(value) &&
      nzchar(value) && !grepl("[\r\n]", value)
  }
  source_ids <- closure$source_ids
  named_character <- function(value, expected_names) {
    typeof(value) == "character" && !is.object(value) &&
      identical(names(value), expected_names) &&
      identical(names(attributes(value)), "names") && !anyNA(value) &&
      all(nzchar(value)) && !any(grepl("[\r\n]", value))
  }
  clean <- is.list(closure) && !is.object(closure) &&
    identical(names(closure), expected_names) &&
    identical(
      closure$source_closure_schema_version,
      "full-cuda-ci-execution-source-closure-v1"
    ) && identical(
      closure$source_discovery_semantics,
      "parsed-r-ast-load-time-literal-source-v1"
    ) && scalar_character(closure$source_project_root) &&
    typeof(source_ids) == "character" && !is.object(source_ids) &&
    is.null(attributes(source_ids)) && length(source_ids) > 0L &&
    !anyNA(source_ids) && all(nzchar(source_ids)) &&
    !any(grepl("[\r\n]", source_ids)) && !anyDuplicated(source_ids) &&
    identical(source_ids, sort(source_ids, method = "radix")) &&
    named_character(closure$direct_source_ids,
                    names(closure$direct_source_ids)) &&
    length(closure$direct_source_ids) > 0L &&
    !anyDuplicated(names(closure$direct_source_ids)) &&
    all(closure$direct_source_ids %in% source_ids) &&
    named_character(closure$source_file_paths, source_ids) &&
    all(file.exists(closure$source_file_paths)) &&
    !any(dir.exists(closure$source_file_paths)) &&
    is.list(closure$source_dependency_map) &&
    !is.object(closure$source_dependency_map) &&
    identical(names(closure$source_dependency_map), source_ids) &&
    all(vapply(closure$source_dependency_map, function(dependencies) {
      typeof(dependencies) == "character" && !is.object(dependencies) &&
        is.null(attributes(dependencies)) && !anyNA(dependencies) &&
        !anyDuplicated(dependencies) &&
        identical(dependencies, sort(dependencies, method = "radix")) &&
        all(dependencies %in% source_ids)
    }, logical(1L))) &&
    typeof(closure$source_closure_count) == "integer" &&
    length(closure$source_closure_count) == 1L &&
    !is.object(closure$source_closure_count) &&
    is.null(attributes(closure$source_closure_count)) &&
    identical(closure$source_closure_count, as.integer(length(source_ids)))
  if (!isTRUE(clean)) {
    stop("execution source closure is malformed", call. = FALSE)
  }
  TRUE
}

fastkpc_full_cuda_fixed_sp_discover_execution_source_closure <- function(
    root_sources, project_root = ".") {
  root_clean <- typeof(root_sources) == "character" &&
    !is.object(root_sources) && length(root_sources) > 0L &&
    !is.null(names(root_sources)) && !anyNA(root_sources) &&
    all(nzchar(root_sources)) && !anyDuplicated(names(root_sources)) &&
    all(nzchar(names(root_sources)))
  if (!isTRUE(root_clean)) {
    stop("execution source roots are malformed", call. = FALSE)
  }
  root <- normalizePath(project_root, winslash = "/", mustWork = TRUE)
  roots <- lapply(root_sources, function(path) {
    fastkpc_full_cuda_fixed_sp_source_identity(path, root)
  })
  root_ids <- vapply(roots, `[[`, character(1L), "id")
  if (anyDuplicated(root_ids)) {
    stop("execution source root identity is ambiguous", call. = FALSE)
  }
  direct_source_ids <- setNames(unname(root_ids), names(root_sources))
  states <- new.env(parent = emptyenv())
  path_by_id <- new.env(parent = emptyenv())
  dependencies_by_id <- new.env(parent = emptyenv())

  visit <- function(source_id, source_path) {
    state <- states[[source_id]]
    if (identical(state, "visiting")) {
      stop("execution source cycle detected at ", source_id, call. = FALSE)
    }
    if (identical(state, "complete")) {
      existing <- path_by_id[[source_id]]
      if (!identical(existing, source_path)) {
        stop("execution source identity ambiguity detected", call. = FALSE)
      }
      return(invisible(NULL))
    }
    states[[source_id]] <- "visiting"
    path_by_id[[source_id]] <- source_path
    parsed <- tryCatch(
      parse(file = source_path, keep.source = FALSE),
      error = function(error) stop(
        "execution source parse failed for ", source_id, ": ",
        conditionMessage(error), call. = FALSE
      )
    )
    dependency_literals <-
      fastkpc_full_cuda_fixed_sp_scan_source_ast(parsed)
    dependency_ids <- character()
    for (literal in dependency_literals) {
      identity <- fastkpc_full_cuda_fixed_sp_source_identity(literal, root)
      dependency_ids <- c(dependency_ids, identity$id)
      visit(identity$id, identity$path)
    }
    dependency_ids <- sort(unique(dependency_ids), method = "radix")
    dependencies_by_id[[source_id]] <- dependency_ids
    states[[source_id]] <- "complete"
    invisible(NULL)
  }
  for (index in seq_along(roots)) {
    visit(root_ids[[index]], roots[[index]]$path)
  }
  source_ids <- sort(ls(path_by_id, all.names = TRUE), method = "radix")
  source_file_paths <- setNames(vapply(
    source_ids, function(source_id) path_by_id[[source_id]], character(1L)
  ), source_ids)
  source_dependency_map <- setNames(lapply(
    source_ids, function(source_id) dependencies_by_id[[source_id]]
  ), source_ids)
  closure <- list(
    source_closure_schema_version =
      "full-cuda-ci-execution-source-closure-v1",
    source_discovery_semantics =
      "parsed-r-ast-load-time-literal-source-v1",
    source_project_root = root,
    direct_source_ids = direct_source_ids,
    source_ids = unname(source_ids),
    source_file_paths = source_file_paths,
    source_dependency_map = source_dependency_map,
    source_closure_count = as.integer(length(source_ids))
  )
  fastkpc_full_cuda_fixed_sp_validate_source_closure(closure)
  closure
}

fastkpc_full_cuda_fixed_sp_source_closure_hash <- function(
    source_closure, source_file_sha256) {
  fastkpc_full_cuda_fixed_sp_validate_source_closure(source_closure)
  source_ids <- source_closure$source_ids
  hashes_clean <- typeof(source_file_sha256) == "character" &&
    !is.object(source_file_sha256) &&
    identical(names(source_file_sha256), source_ids) &&
    identical(names(attributes(source_file_sha256)), "names") &&
    !anyNA(source_file_sha256) &&
    all(grepl("^[0-9a-f]{64}$", source_file_sha256))
  if (!isTRUE(hashes_clean)) {
    stop("execution source closure hash map is malformed", call. = FALSE)
  }
  lines <- c(
    paste0("closure.schema=",
           source_closure$source_closure_schema_version),
    paste0("closure.discovery=",
           source_closure$source_discovery_semantics),
    paste0("closure.project_root=", source_closure$source_project_root),
    paste0("closure.count=", source_closure$source_closure_count)
  )
  for (role in names(source_closure$direct_source_ids)) {
    lines <- c(lines, paste0(
      "closure.root.", role, "=",
      source_closure$direct_source_ids[[role]]
    ))
  }
  for (source_id in source_ids) {
    lines <- c(
      lines,
      paste0("closure.source.", source_id, ".path=",
             source_closure$source_file_paths[[source_id]]),
      paste0("closure.source.", source_id, ".sha256=",
             source_file_sha256[[source_id]]),
      paste0("closure.source.", source_id, ".dependencies=",
             paste(source_closure$source_dependency_map[[source_id]],
                   collapse = ","))
    )
  }
  payload <- charToRaw(enc2utf8(paste0(paste(lines, collapse = "\n"), "\n")))
  unname(digest::digest(payload, algo = "sha256", serialize = FALSE))
}

fastkpc_full_cuda_fixed_sp_native_build_input_hash <- function(
    paths, sha256) {
  clean <- typeof(paths) == "character" && !is.object(paths) &&
    !is.null(names(paths)) && length(paths) > 0L && !anyNA(paths) &&
    all(nzchar(paths)) && !any(grepl("[\r\n]", paths)) &&
    identical(names(paths), sort(names(paths), method = "radix")) &&
    !anyDuplicated(names(paths)) && typeof(sha256) == "character" &&
    !is.object(sha256) && identical(names(sha256), names(paths)) &&
    identical(names(attributes(paths)), "names") &&
    identical(names(attributes(sha256)), "names") && !anyNA(sha256) &&
    all(grepl("^[0-9a-f]{64}$", sha256))
  if (!isTRUE(clean)) {
    stop("native build input identity is malformed", call. = FALSE)
  }
  lines <- unlist(lapply(names(paths), function(input_id) c(
    paste0("native_build_input.", input_id, ".path=", paths[[input_id]]),
    paste0("native_build_input.", input_id, ".sha256=", sha256[[input_id]])
  )), use.names = FALSE)
  payload <- charToRaw(enc2utf8(paste0(paste(lines, collapse = "\n"), "\n")))
  unname(digest::digest(payload, algo = "sha256", serialize = FALSE))
}

fastkpc_full_cuda_fixed_sp_native_build_environment_names <- function() {
  sort(c(
    "AR", "AS", "CC", "CFLAGS", "COMPILER_PATH", "CPATH", "CPPFLAGS",
    "CUDAFLAGS", "CUDAHOSTCXX", "CUDA_HOME", "CUDA_PATH", "CXX",
    "CXXFLAGS", "GCC_EXEC_PREFIX", "LANG", "LC_ALL", "LDFLAGS",
    "LD_LIBRARY_PATH", "LIBRARY_PATH", "MAKEFLAGS", "NVCCFLAGS",
    "NVCC_APPEND_FLAGS", "NVCC_CCBIN", "NVCC_PREPEND_FLAGS", "PATH",
    "PKG_CPPFLAGS", "PKG_CXXFLAGS", "PKG_LIBS", "R_ARCH", "R_HOME",
    "R_LIBS", "R_LIBS_SITE", "R_LIBS_USER", "R_MAKEVARS_SITE",
    "R_MAKEVARS_USER", "SHLIB_CXXLD", "SHLIB_CXXLDFLAGS",
    "SOURCE_DATE_EPOCH"
  ), method = "radix")
}

fastkpc_full_cuda_fixed_sp_native_build_environment <- function(
    trace_invocation) {
  if (typeof(trace_invocation) != "character" ||
      length(trace_invocation) != 1L || is.object(trace_invocation) ||
      !is.null(attributes(trace_invocation)) || anyNA(trace_invocation) ||
      !nzchar(trace_invocation) || grepl("[\r\n]", trace_invocation)) {
    stop("native build trace invocation is malformed", call. = FALSE)
  }
  names <- fastkpc_full_cuda_fixed_sp_native_build_environment_names()
  values <- Sys.getenv(names, unset = NA_character_)
  if (grepl("(^|[[:space:]])LC_ALL=C([[:space:]]|$)",
            trace_invocation, perl = TRUE)) {
    values[[match("LC_ALL", names)]] <- "C"
  }
  is_set <- !is.na(values)
  values[!is_set] <- ""
  if (any(grepl("[\r\n]", values))) {
    stop("native build environment contains a newline", call. = FALSE)
  }
  data.frame(
    name = names, is_set = unname(is_set), value = unname(values),
    stringsAsFactors = FALSE, row.names = as.character(seq_along(names))
  )
}

fastkpc_full_cuda_fixed_sp_validate_native_build_commands <- function(
    value, require_phase3 = FALSE) {
  scalar_character <- function(element) {
    typeof(element) == "character" && length(element) == 1L &&
      !is.object(element) && is.null(attributes(element)) &&
      !anyNA(element) && nzchar(element) && !grepl("[\r\n]", element)
  }
  require_phase3 <- identical(require_phase3, TRUE)
  commands <- value$commands
  environment <- value$build_environment
  expected_environment_names <-
    fastkpc_full_cuda_fixed_sp_native_build_environment_names()
  clean <- scalar_character(value$command_projection_schema_version) &&
    identical(
      value$command_projection_schema_version,
      "full-cuda-ci-native-build-command-projection-v1"
    ) && typeof(value$command_count) == "integer" &&
    length(value$command_count) == 1L && !is.object(value$command_count) &&
    is.null(attributes(value$command_count)) &&
    !is.na(value$command_count) && value$command_count >= 0L &&
    is.list(commands) && !is.object(commands) &&
    is.null(attributes(commands)) &&
    length(commands) == value$command_count &&
    scalar_character(value$build_environment_schema_version) &&
    identical(
      value$build_environment_schema_version,
      "full-cuda-ci-native-build-environment-v1"
    ) && is.data.frame(environment) && !is.object(environment$name) &&
    !is.object(environment$is_set) && !is.object(environment$value) &&
    identical(names(environment), c("name", "is_set", "value")) &&
    identical(
      rownames(environment), as.character(seq_len(nrow(environment)))
    ) && identical(environment$name, expected_environment_names) &&
    typeof(environment$is_set) == "logical" &&
    is.null(attributes(environment$is_set)) &&
    !anyNA(environment$is_set) &&
    typeof(environment$value) == "character" &&
    is.null(attributes(environment$value)) && !anyNA(environment$value) &&
    !any(grepl("[\r\n]", environment$value)) &&
    all(environment$is_set | environment$value == "")
  if (!isTRUE(clean)) {
    stop("native build command/environment projection is malformed",
         call. = FALSE)
  }
  roles <- character(length(commands))
  command_clean <- vapply(seq_along(commands), function(index) {
    command <- commands[[index]]
    exact <- is.list(command) && !is.object(command) && identical(
      names(command),
      c("role", "executable_path", "executable_sha256", "argv")
    ) && scalar_character(command$role) &&
      command$role %in% c("cxx_compile", "cuda_compile", "link") &&
      scalar_character(command$executable_path) &&
      startsWith(command$executable_path, "/") &&
      scalar_character(command$executable_sha256) &&
      grepl("^[0-9a-f]{64}$", command$executable_sha256) &&
      typeof(command$argv) == "character" && !is.object(command$argv) &&
      is.null(attributes(command$argv)) && length(command$argv) >= 4L &&
      !anyNA(command$argv) && all(nzchar(command$argv)) &&
      !any(grepl("[\r\n]", command$argv)) &&
      identical(command$argv[[1L]], "<EXECUTABLE>")
    if (!isTRUE(exact)) return(FALSE)
    roles[[index]] <<- command$role
    output_flags <- which(command$argv == "-o")
    output_markers <- which(command$argv == "<OUTPUT>")
    output_exact <- length(output_flags) == 1L &&
      output_flags[[1L]] < length(command$argv) &&
      identical(output_markers, output_flags + 1L)
    source_argument <- any(grepl(
      "\\.(c|cc|cpp|cxx|cu)$", command$argv, perl = TRUE
    ))
    role_exact <- switch(
      command$role,
      cxx_compile = "-c" %in% command$argv && source_argument,
      cuda_compile = "-c" %in% command$argv && source_argument,
      link = "-shared" %in% command$argv &&
        !"-c" %in% command$argv
    )
    file_index <- if (is.data.frame(value$files) &&
                      all(c("path", "sha256") %in% names(value$files))) {
      match(command$executable_path, value$files$path)
    } else {
      NA_integer_
    }
    tool_exact <- !is.na(file_index) && identical(
      command$executable_sha256, value$files$sha256[[file_index]]
    )
    isTRUE(output_exact) && isTRUE(role_exact) && isTRUE(tool_exact)
  }, logical(1L))
  if (!all(command_clean) ||
      (require_phase3 && !identical(
        sort(unique(roles), method = "radix"),
        c("cuda_compile", "cxx_compile", "link")
      ))) {
    stop("native build command projection is incomplete or malformed",
         call. = FALSE)
  }
  TRUE
}

.fastkpc_full_cuda_fixed_sp_native_build_command_lines <- function(value) {
  lines <- c(
    paste0("command_projection.schema=",
           value$command_projection_schema_version),
    paste0("command_count=", value$command_count)
  )
  for (index in seq_along(value$commands)) {
    command <- value$commands[[index]]
    lines <- c(
      lines,
      paste0("command.", index, ".role=", command$role),
      paste0("command.", index, ".executable_path=",
             command$executable_path),
      paste0("command.", index, ".executable_sha256=",
             command$executable_sha256),
      paste0("command.", index, ".argc=", length(command$argv))
    )
    for (argument_index in seq_along(command$argv)) {
      argument <- command$argv[[argument_index]]
      lines <- c(lines, paste0(
        "command.", index, ".argv.", argument_index, "=",
        nchar(argument, type = "bytes"), ":", argument
      ))
    }
  }
  environment <- value$build_environment
  lines <- c(
    lines,
    paste0("build_environment.schema=",
           value$build_environment_schema_version),
    paste0("build_environment.count=", nrow(environment))
  )
  for (index in seq_len(nrow(environment))) {
    lines <- c(
      lines,
      paste0("build_environment.", index, ".name=",
             environment$name[[index]]),
      paste0("build_environment.", index, ".is_set=",
             if (environment$is_set[[index]]) "true" else "false"),
      paste0(
        "build_environment.", index, ".value=",
        nchar(environment$value[[index]], type = "bytes"), ":",
        environment$value[[index]]
      )
    )
  }
  lines
}

fastkpc_full_cuda_fixed_sp_native_build_dependency_hash <- function(value) {
  files <- value$files
  exclusions <- value$exclusions
  lines <- c(
    paste0("schema_version=", value$schema_version),
    paste0("trace_semantics=", value$trace_semantics),
    paste0("trace_invocation=", value$trace_invocation),
    paste0("build_working_dir=", value$build_working_dir),
    paste0("trace.sha256=", value$trace_sha256),
    paste0("tracer.path=", value$tracer_path),
    paste0("tracer.sha256=", value$tracer_sha256),
    paste0("dependency_count=", value$dependency_count)
  )
  for (index in seq_len(nrow(files))) {
    lines <- c(
      lines,
      paste0("dependency.", index, ".path=", files$path[[index]]),
      paste0("dependency.", index, ".sha256=", files$sha256[[index]])
    )
  }
  lines <- c(lines, paste0("exclusion_count=", value$exclusion_count))
  for (index in seq_len(nrow(exclusions))) {
    lines <- c(
      lines,
      paste0("exclusion.", index, ".path=", exclusions$path[[index]]),
      paste0("exclusion.", index, ".reason=", exclusions$reason[[index]])
    )
  }
  lines <- c(
    lines,
    .fastkpc_full_cuda_fixed_sp_native_build_command_lines(value)
  )
  payload <- charToRaw(enc2utf8(paste0(paste(lines, collapse = "\n"), "\n")))
  unname(digest::digest(payload, algo = "sha256", serialize = FALSE))
}

fastkpc_full_cuda_fixed_sp_native_build_attestation_hash <- function(value) {
  files <- value$files
  exclusions <- value$exclusions
  scalar_text <- function(element) {
    typeof(element) == "character" && length(element) == 1L &&
      !is.object(element) && is.null(attributes(element)) &&
      !anyNA(element) && nzchar(element) && !grepl("[\r\n]", element)
  }
  clean <- is.list(value) && !is.object(value) &&
    all(c(
      "schema_version", "trace_semantics", "build_working_dir",
      "tracer_path", "tracer_sha256", "dependency_count", "files",
      "exclusion_count", "exclusions", "command_projection_schema_version",
      "command_count", "commands", "build_environment_schema_version",
      "build_environment"
    ) %in% names(value)) &&
    scalar_text(value$schema_version) && scalar_text(value$trace_semantics) &&
    scalar_text(value$build_working_dir) && scalar_text(value$tracer_path) &&
    scalar_text(value$tracer_sha256) &&
    grepl("^[0-9a-f]{64}$", value$tracer_sha256) &&
    typeof(value$dependency_count) == "integer" &&
    length(value$dependency_count) == 1L && !is.na(value$dependency_count) &&
    value$dependency_count > 0L && is.data.frame(files) &&
    identical(names(files), c("path", "sha256")) &&
    nrow(files) == value$dependency_count &&
    typeof(files$path) == "character" && !anyNA(files$path) &&
    identical(files$path, sort(files$path, method = "radix")) &&
    !anyDuplicated(files$path) && typeof(files$sha256) == "character" &&
    !anyNA(files$sha256) && all(grepl("^[0-9a-f]{64}$", files$sha256)) &&
    typeof(value$exclusion_count) == "integer" &&
    length(value$exclusion_count) == 1L && !is.na(value$exclusion_count) &&
    value$exclusion_count >= 0L && is.data.frame(exclusions) &&
    identical(names(exclusions), c("path", "reason")) &&
    nrow(exclusions) == value$exclusion_count &&
    typeof(exclusions$path) == "character" && !anyNA(exclusions$path) &&
    identical(exclusions$path, sort(exclusions$path, method = "radix")) &&
    !anyDuplicated(exclusions$path) &&
    typeof(exclusions$reason) == "character" && !anyNA(exclusions$reason) &&
    all(exclusions$reason %in% c(
      "generated_output", "pseudo_fs", "non_regular"
    ))
  if (!isTRUE(clean) || !isTRUE(tryCatch(
        fastkpc_full_cuda_fixed_sp_validate_native_build_commands(
          value, require_phase3 = TRUE
        ),
        error = function(error) FALSE
      ))) {
    stop("native build trace attestation input is malformed", call. = FALSE)
  }
  lines <- c(
    "attestation.schema=full-cuda-ci-native-build-trace-attestation-v2",
    paste0("dependency.schema=", value$schema_version),
    paste0("trace.semantics=", value$trace_semantics),
    paste0("build.working_dir=", value$build_working_dir),
    paste0("tracer.path=", value$tracer_path),
    paste0("tracer.sha256=", value$tracer_sha256),
    paste0("dependency_count=", value$dependency_count)
  )
  for (index in seq_len(nrow(files))) {
    lines <- c(
      lines,
      paste0("dependency.", index, ".path=", files$path[[index]]),
      paste0("dependency.", index, ".sha256=", files$sha256[[index]])
    )
  }
  lines <- c(lines, paste0("exclusion_count=", value$exclusion_count))
  for (reason in c("generated_output", "pseudo_fs", "non_regular")) {
    lines <- c(lines, paste0(
      "exclusion.", reason, ".count=",
      sum(exclusions$reason == reason)
    ))
  }
  lines <- c(
    lines,
    .fastkpc_full_cuda_fixed_sp_native_build_command_lines(value)
  )
  payload <- charToRaw(enc2utf8(paste0(paste(lines, collapse = "\n"), "\n")))
  unname(digest::digest(payload, algo = "sha256", serialize = FALSE))
}

fastkpc_full_cuda_fixed_sp_regular_file_mask <- function(paths) {
  if (typeof(paths) != "character" || is.object(paths) || anyNA(paths) ||
      any(!nzchar(paths)) || any(grepl("[\r\n]", paths))) {
    stop("regular-file identity paths are malformed", call. = FALSE)
  }
  if (length(paths) == 0L) return(logical())
  bash <- unname(Sys.which("bash"))
  if (length(bash) != 1L || !nzchar(bash) || !file.exists(bash) ||
      dir.exists(bash)) {
    stop("bash is required for regular-file identity checks", call. = FALSE)
  }
  bash <- normalizePath(bash, winslash = "/", mustWork = TRUE)
  script <- paste0(
    'for path do if test -f "$path"; then printf "1\\n"; ',
    'else printf "0\\n"; fi; done'
  )
  chunks <- split(paths, ceiling(seq_along(paths) / 256L))
  values <- unlist(lapply(chunks, function(chunk) {
    output <- suppressWarnings(system2(
      bash,
      c(
        "-c", shQuote(script), "fastkpc-regular-file-check",
        vapply(chunk, shQuote, character(1L))
      ),
      stdout = TRUE, stderr = TRUE
    ))
    status <- attr(output, "status")
    if ((!is.null(status) && status != 0L) ||
        length(output) != length(chunk) ||
        any(!output %in% c("0", "1"))) {
      stop("regular-file identity check failed", call. = FALSE)
    }
    output == "1"
  }), use.names = FALSE)
  if (length(values) != length(paths)) {
    stop("regular-file identity check is incomplete", call. = FALSE)
  }
  values
}

fastkpc_full_cuda_fixed_sp_validate_native_build_dependencies <- function(
    value) {
  scalar_character <- function(element) {
    typeof(element) == "character" && length(element) == 1L &&
      !is.object(element) && is.null(attributes(element)) &&
      !anyNA(element) && nzchar(element) && !grepl("[\r\n]", element)
  }
  files <- value$files
  exclusions <- value$exclusions
  normalized_build_working_dir <- tryCatch(
    normalizePath(
      value$build_working_dir, winslash = "/", mustWork = TRUE
    ),
    error = function(error) ""
  )
  normalized_trace_path <- tryCatch(
    normalizePath(value$trace_path, winslash = "/", mustWork = TRUE),
    error = function(error) ""
  )
  trace_regular <- tryCatch(
    identical(
      fastkpc_full_cuda_fixed_sp_regular_file_mask(value$trace_path), TRUE
    ),
    error = function(error) FALSE
  )
  current_trace_sha256 <- tryCatch(
    fastkpc_full_cuda_fixed_sp_sha256_file(value$trace_path),
    error = function(error) ""
  )
  normalized_file_paths <- tryCatch({
    if (!is.data.frame(files) || !identical(names(files), c("path", "sha256")) ||
        typeof(files$path) != "character" || anyNA(files$path)) {
      character()
    } else {
      unname(normalizePath(
        files$path, winslash = "/", mustWork = TRUE
      ))
    }
  }, error = function(error) character())
  regular_file_paths <- tryCatch({
    if (!is.data.frame(files) || typeof(files$path) != "character" ||
        anyNA(files$path)) {
      logical()
    } else {
      fastkpc_full_cuda_fixed_sp_regular_file_mask(files$path)
    }
  }, error = function(error) logical())
  tracer_regular <- tryCatch(
    identical(
      fastkpc_full_cuda_fixed_sp_regular_file_mask(value$tracer_path),
      TRUE
    ),
    error = function(error) FALSE
  )
  clean <- is.list(value) && !is.object(value) && identical(
    names(value),
    c(
      "schema_version", "trace_semantics", "trace_invocation",
      "build_working_dir", "trace_path", "trace_sha256", "tracer_path",
      "tracer_sha256", "dependency_count", "files", "exclusion_count",
      "exclusions", "command_projection_schema_version", "command_count",
      "commands", "build_environment_schema_version", "build_environment",
      "aggregate_sha256"
    )
  ) && scalar_character(value$schema_version) && identical(
    value$schema_version, "full-cuda-ci-native-build-dependencies-v3"
  ) && scalar_character(value$trace_semantics) && identical(
    value$trace_semantics,
    "linux-strace-successful-read-exec-evidence-v3"
  ) && scalar_character(value$trace_invocation) &&
    scalar_character(value$build_working_dir) &&
    startsWith(value$build_working_dir, "/") &&
    dir.exists(value$build_working_dir) &&
    identical(value$build_working_dir, normalized_build_working_dir) &&
    scalar_character(value$trace_path) && startsWith(value$trace_path, "/") &&
    identical(value$trace_path, normalized_trace_path) && trace_regular &&
    scalar_character(value$trace_sha256) &&
    grepl("^[0-9a-f]{64}$", value$trace_sha256) &&
    identical(value$trace_sha256, current_trace_sha256) &&
    scalar_character(value$tracer_path) && startsWith(value$tracer_path, "/") &&
    identical(
      value$tracer_path,
      tryCatch(
        normalizePath(value$tracer_path, winslash = "/", mustWork = TRUE),
        error = function(error) ""
      )
    ) && tracer_regular &&
    scalar_character(value$tracer_sha256) &&
    grepl("^[0-9a-f]{64}$", value$tracer_sha256) &&
    typeof(value$dependency_count) == "integer" &&
    length(value$dependency_count) == 1L &&
    !is.object(value$dependency_count) &&
    is.null(attributes(value$dependency_count)) &&
    !is.na(value$dependency_count) && value$dependency_count > 0L &&
    is.data.frame(files) && !is.object(files$path) &&
    !is.object(files$sha256) && identical(names(files), c("path", "sha256")) &&
    identical(rownames(files), as.character(seq_len(nrow(files)))) &&
    nrow(files) == value$dependency_count &&
    typeof(files$path) == "character" &&
    is.null(attributes(files$path)) && !anyNA(files$path) &&
    all(nzchar(files$path)) && !any(grepl("[\r\n]", files$path)) &&
    identical(files$path, sort(files$path, method = "radix")) &&
    !anyDuplicated(files$path) && all(startsWith(files$path, "/")) &&
    identical(regular_file_paths, rep.int(TRUE, nrow(files))) &&
    identical(normalized_file_paths, files$path) &&
    typeof(files$sha256) == "character" &&
    is.null(attributes(files$sha256)) && !anyNA(files$sha256) &&
    all(grepl("^[0-9a-f]{64}$", files$sha256)) &&
    value$tracer_path %in% files$path &&
    identical(
      files$sha256[[match(value$tracer_path, files$path)]],
      value$tracer_sha256
    ) && scalar_character(value$aggregate_sha256) &&
    grepl("^[0-9a-f]{64}$", value$aggregate_sha256) &&
    typeof(value$exclusion_count) == "integer" &&
    length(value$exclusion_count) == 1L &&
    !is.object(value$exclusion_count) &&
    is.null(attributes(value$exclusion_count)) &&
    !is.na(value$exclusion_count) && value$exclusion_count >= 0L &&
    is.data.frame(exclusions) && !is.object(exclusions$path) &&
    !is.object(exclusions$reason) &&
    identical(names(exclusions), c("path", "reason")) &&
    identical(rownames(exclusions), as.character(seq_len(nrow(exclusions)))) &&
    nrow(exclusions) == value$exclusion_count &&
    typeof(exclusions$path) == "character" &&
    is.null(attributes(exclusions$path)) && !anyNA(exclusions$path) &&
    all(nzchar(exclusions$path)) &&
    !any(grepl("[\r\n]", exclusions$path)) &&
    all(startsWith(exclusions$path, "/")) &&
    identical(exclusions$path, sort(exclusions$path, method = "radix")) &&
    !anyDuplicated(exclusions$path) &&
    typeof(exclusions$reason) == "character" &&
    is.null(attributes(exclusions$reason)) && !anyNA(exclusions$reason) &&
    all(exclusions$reason %in% c(
      "generated_output", "pseudo_fs", "non_regular"
    )) && !any(exclusions$path %in% files$path)
  command_projection_clean <- tryCatch(
    fastkpc_full_cuda_fixed_sp_validate_native_build_commands(value),
    error = function(error) FALSE
  )
  if (!isTRUE(clean) || !isTRUE(command_projection_clean) || !identical(
        value$aggregate_sha256,
        fastkpc_full_cuda_fixed_sp_native_build_dependency_hash(value)
      )) {
    stop("native build dependency identity is malformed", call. = FALSE)
  }
  TRUE
}

fastkpc_full_cuda_fixed_sp_decode_strace_path <- function(value) {
  if (typeof(value) != "character" || length(value) != 1L ||
      is.object(value) || !is.null(attributes(value)) || anyNA(value)) {
    stop("native build trace is malformed", call. = FALSE)
  }
  parsed <- tryCatch(
    parse(text = paste0('"', value, '"'), keep.source = FALSE),
    error = function(error) NULL
  )
  if (is.null(parsed) || length(parsed) != 1L ||
      typeof(parsed[[1L]]) != "character" || length(parsed[[1L]]) != 1L ||
      anyNA(parsed[[1L]]) || grepl("[\r\n]", parsed[[1L]])) {
    stop("native build trace is malformed", call. = FALSE)
  }
  parsed[[1L]]
}

fastkpc_full_cuda_fixed_sp_open_flag_tokens <- function(lines, syscalls) {
  malformed <- function() {
    stop("native build trace is malformed", call. = FALSE)
  }
  if (typeof(lines) != "character" || typeof(syscalls) != "character" ||
      length(lines) != length(syscalls) || anyNA(lines) || anyNA(syscalls) ||
      any(!syscalls %in% c("open", "openat", "openat2"))) {
    malformed()
  }
  quoted <- '"(?:[^"\\\\]|\\\\.)*"'
  flag_expressions <- vapply(
    seq_along(lines),
    function(index) {
      pattern <- switch(
        syscalls[[index]],
        open = paste0(
          "open\\([[:space:]]*", quoted,
          "[[:space:]]*,[[:space:]]*([^,)]*)"
        ),
        openat = paste0(
          "openat\\([^,]+,[[:space:]]*", quoted,
          "[[:space:]]*,[[:space:]]*([^,)]*)"
        ),
        openat2 = paste0(
          "openat2\\([^,]+,[[:space:]]*", quoted,
          "[[:space:]]*,[[:space:]]*\\{[[:space:]]*",
          "flags[[:space:]]*=[[:space:]]*([^,}]*)"
        )
      )
      matched <- regmatches(
        lines[[index]], regexec(pattern, lines[[index]], perl = TRUE)
      )[[1L]]
      if (length(matched) != 2L) malformed()
      trimws(matched[[2L]])
    },
    character(1L)
  )
  unname(lapply(flag_expressions, function(expression) {
    unname(regmatches(
      expression,
      gregexpr(
        "(?<![[:alnum:]_])O_[A-Z0-9_]+(?![[:alnum:]_])",
        expression, perl = TRUE
      )
    )[[1L]])
  }))
}

fastkpc_full_cuda_fixed_sp_decode_strace_argv <- function(lines, syscalls) {
  malformed <- function() {
    stop("native build trace command projection is malformed",
         call. = FALSE)
  }
  if (typeof(lines) != "character" || typeof(syscalls) != "character" ||
      length(lines) != length(syscalls) || anyNA(lines) || anyNA(syscalls) ||
      any(!syscalls %in% c("execve", "execveat"))) {
    malformed()
  }
  quoted <- '"(?:[^"\\\\]|\\\\.)*"'
  lapply(seq_along(lines), function(index) {
    pattern <- switch(
      syscalls[[index]],
      execve = paste0(
        "execve\\(", quoted, ",[[:space:]]*\\[((?:", quoted,
        "(?:[[:space:]]*,[[:space:]]*", quoted,
        ")*)?)\\][[:space:]]*,"
      ),
      execveat = paste0(
        "execveat\\([^,]+,[[:space:]]*", quoted,
        ",[[:space:]]*\\[((?:", quoted,
        "(?:[[:space:]]*,[[:space:]]*", quoted,
        ")*)?)\\][[:space:]]*,"
      )
    )
    matched <- regmatches(
      lines[[index]], regexec(pattern, lines[[index]], perl = TRUE)
    )[[1L]]
    if (length(matched) != 2L || grepl("(^|[[:space:],])\\.\\.",
                                      matched[[2L]], perl = TRUE)) {
      malformed()
    }
    body <- matched[[2L]]
    if (!nzchar(body)) return(character())
    tokens <- regmatches(body, gregexpr(quoted, body, perl = TRUE))[[1L]]
    remainder <- gsub(quoted, "", body, perl = TRUE)
    remainder <- gsub("[[:space:],]", "", remainder, perl = TRUE)
    if (length(tokens) == 0L || nzchar(remainder)) malformed()
    vapply(tokens, function(token) {
      fastkpc_full_cuda_fixed_sp_decode_strace_path(
        substring(token, 2L, nchar(token, type = "chars") - 1L)
      )
    }, character(1L), USE.NAMES = FALSE)
  })
}

fastkpc_full_cuda_fixed_sp_native_build_commands <- function(
    exec_lines, exec_syscalls, exec_paths, build_working_dir, files) {
  if (typeof(exec_lines) != "character" ||
      typeof(exec_syscalls) != "character" ||
      typeof(exec_paths) != "character" ||
      length(exec_lines) != length(exec_syscalls) ||
      length(exec_lines) != length(exec_paths) || anyNA(exec_lines) ||
      anyNA(exec_syscalls) || anyNA(exec_paths) ||
      !is.data.frame(files) || !identical(names(files), c("path", "sha256"))) {
    stop("native build command inputs are malformed", call. = FALSE)
  }
  argv <- fastkpc_full_cuda_fixed_sp_decode_strace_argv(
    exec_lines, exec_syscalls
  )
  root_prefix <- paste0(build_working_dir, "/")
  records <- list()
  for (index in seq_along(argv)) {
    arguments <- argv[[index]]
    if (length(arguments) == 0L) next
    source_arguments <- arguments[
      grepl("\\.(c|cc|cpp|cxx|cu)$", arguments, perl = TRUE) &
        startsWith(arguments, root_prefix)
    ]
    executable_name <- tolower(basename(exec_paths[[index]]))
    cxx_driver <- grepl(
      "(^|-)(g\\+\\+|c\\+\\+|clang\\+\\+)(-[0-9.]+)?$",
      executable_name, perl = TRUE
    )
    nvcc_driver <- grepl("(^|-)nvcc(-[0-9.]+)?$", executable_name,
                         perl = TRUE)
    role <- NULL
    if ("-c" %in% arguments && length(source_arguments) > 0L) {
      cuda_source <- any(grepl("\\.cu$", source_arguments, perl = TRUE))
      role <- if (cuda_source && nvcc_driver) {
        "cuda_compile"
      } else if (!cuda_source && cxx_driver) {
        "cxx_compile"
      } else {
        NULL
      }
    } else if (cxx_driver && "-shared" %in% arguments &&
               "-o" %in% arguments) {
      role <- "link"
    }
    if (is.null(role)) next
    output_flags <- which(arguments == "-o")
    if (length(output_flags) != 1L ||
        output_flags[[1L]] >= length(arguments)) {
      stop("native build command output projection is incomplete",
           call. = FALSE)
    }
    canonical_argv <- arguments
    canonical_argv[[1L]] <- "<EXECUTABLE>"
    canonical_argv[[output_flags + 1L]] <- "<OUTPUT>"
    executable_path <- normalizePath(
      exec_paths[[index]], winslash = "/", mustWork = TRUE
    )
    file_index <- match(executable_path, files$path)
    if (is.na(file_index)) {
      stop("native build command executable is absent from dependencies",
           call. = FALSE)
    }
    records[[length(records) + 1L]] <- list(
      role = role,
      executable_path = executable_path,
      executable_sha256 = files$sha256[[file_index]],
      argv = unname(canonical_argv)
    )
  }
  records
}

fastkpc_full_cuda_fixed_sp_reconstruct_native_build_dependencies <- function(
    trace_path, build_working_dir, tracer_path, trace_invocation) {
  scalar_path <- function(value, label) {
    if (typeof(value) != "character" || length(value) != 1L ||
        is.object(value) || !is.null(attributes(value)) || anyNA(value) ||
        !nzchar(value)) {
      stop(label, " is malformed", call. = FALSE)
    }
    normalizePath(value, winslash = "/", mustWork = TRUE)
  }
  trace_path <- scalar_path(trace_path, "native build trace path")
  working_dir <- scalar_path(build_working_dir, "native build working directory")
  tracer_path <- scalar_path(tracer_path, "native build tracer path")
  if (!file_test("-f", trace_path) || !file_test("-f", tracer_path) ||
      typeof(trace_invocation) != "character" ||
      length(trace_invocation) != 1L || is.object(trace_invocation) ||
      !is.null(attributes(trace_invocation)) || anyNA(trace_invocation) ||
      !nzchar(trace_invocation) || grepl("[\r\n]", trace_invocation)) {
    stop("native build trace metadata is malformed", call. = FALSE)
  }
  trace_sha256 <- fastkpc_full_cuda_fixed_sp_sha256_file(trace_path)
  lines <- readLines(trace_path, warn = FALSE, encoding = "bytes")
  if (length(lines) == 0L || all(!nzchar(lines))) {
    stop("native build trace is empty", call. = FALSE)
  }
  syscall_pattern <- paste0(
    "^[[:space:]]*(?:[0-9]+|\\[pid[[:space:]]+[0-9]+\\])",
    "[[:space:]]+(open|openat|openat2|execve|execveat)\\("
  )
  traced <- grepl(syscall_pattern, lines, perl = TRUE)
  if (!any(traced)) {
    stop("native build trace is malformed", call. = FALSE)
  }
  traced_lines <- lines[traced]
  syscalls <- sub(
    paste0(syscall_pattern, ".*$"), "\\1", traced_lines, perl = TRUE
  )
  has_result <- grepl("[[:space:]]=", traced_lines)
  failed <- grepl("[[:space:]]= -[0-9]+", traced_lines)
  succeeded <- grepl(
    "[[:space:]]= (0|[1-9][0-9]*)", traced_lines
  )
  if (any(!has_result) || any(!(failed | succeeded))) {
    stop("native build trace is malformed", call. = FALSE)
  }
  traced_lines <- traced_lines[succeeded]
  syscalls <- syscalls[succeeded]
  open_call <- syscalls %in% c("open", "openat", "openat2")
  quoted <- '"((?:[^"\\\\]|\\\\.)*)"'
  extract_paths <- function(group, pattern) {
    if (length(group) == 0L) return(character())
    matches <- regmatches(group, regexec(pattern, group, perl = TRUE))
    if (any(lengths(matches) != 2L)) {
      stop("native build trace is malformed", call. = FALSE)
    }
    vapply(matches, `[[`, character(1L), 2L)
  }
  resolved_pattern <-
    "^.*[[:space:]]= [0-9]+<(/[^>]*)>[[:space:]]*$"
  extract_open_paths <- function(open_lines, open_syscalls) {
    if (length(open_lines) == 0L) return(character())
    resolved <- grepl(resolved_pattern, open_lines, perl = TRUE)
    encoded <- character(length(open_lines))
    encoded[resolved] <- sub(
      resolved_pattern, "\\1", open_lines[resolved], perl = TRUE
    )
    if (any(!resolved)) {
      unresolved_lines <- open_lines[!resolved]
      unresolved_syscalls <- open_syscalls[!resolved]
      direct <- unresolved_syscalls == "open"
      unresolved_encoded <- character(length(unresolved_lines))
      unresolved_encoded[direct] <- extract_paths(
        unresolved_lines[direct], paste0("open\\(", quoted)
      )
      unresolved_encoded[!direct] <- extract_paths(
        unresolved_lines[!direct],
        paste0("(?:openat|openat2)\\([^,]+,[[:space:]]*", quoted)
      )
      encoded[!resolved] <- unresolved_encoded
    }
    decoded <- unname(vapply(
      encoded,
      fastkpc_full_cuda_fixed_sp_decode_strace_path,
      character(1L)
    ))
    if (any(!startsWith(decoded, "/"))) {
      stop("relative native build path lacks resolved strace identity",
           call. = FALSE)
    }
    decoded
  }
  open_lines <- traced_lines[open_call]
  open_syscalls <- syscalls[open_call]
  open_flag_tokens <- fastkpc_full_cuda_fixed_sp_open_flag_tokens(
    open_lines, open_syscalls
  )
  has_open_flag <- function(flag) {
    vapply(open_flag_tokens, function(tokens) flag %in% tokens, logical(1L))
  }
  read_open <- (has_open_flag("O_RDONLY") | has_open_flag("O_RDWR")) &
    !has_open_flag("O_WRONLY")
  generation_open <- has_open_flag("O_TRUNC") |
    has_open_flag("O_TMPFILE") |
    (has_open_flag("O_CREAT") & has_open_flag("O_EXCL"))
  directory_open <- read_open & has_open_flag("O_DIRECTORY")
  open_paths <- extract_open_paths(open_lines, open_syscalls)
  exec_lines <- traced_lines[!open_call]
  exec_syscalls <- syscalls[!open_call]
  exec_direct <- exec_syscalls == "execve"
  exec_paths <- if (length(exec_lines) == 0L) {
    character()
  } else {
    encoded_exec_paths <- character(length(exec_lines))
    encoded_exec_paths[exec_direct] <- extract_paths(
      exec_lines[exec_direct], paste0("execve\\(", quoted)
    )
    encoded_exec_paths[!exec_direct] <- extract_paths(
      exec_lines[!exec_direct],
      paste0("execveat\\([^,]+,[[:space:]]*", quoted)
    )
    unname(vapply(
      encoded_exec_paths,
      fastkpc_full_cuda_fixed_sp_decode_strace_path,
      character(1L)
    ))
  }
  if (any(!startsWith(exec_paths, "/"))) {
    stop("relative native build path lacks resolved strace identity",
         call. = FALSE)
  }
  event_paths <- character(length(traced_lines))
  event_paths[open_call] <- open_paths
  event_paths[!open_call] <- exec_paths
  access_event <- !open_call
  access_event[open_call] <- read_open
  generation_event <- rep.int(FALSE, length(traced_lines))
  generation_event[open_call] <- generation_open
  directory_event <- rep.int(FALSE, length(traced_lines))
  directory_event[open_call] <- directory_open
  paths <- unique(event_paths[access_event])
  access_paths <- event_paths
  access_paths[!access_event] <- NA_character_
  first_access <- match(paths, access_paths)
  generation_paths <- event_paths
  generation_paths[!generation_event] <- NA_character_
  first_generation <- match(paths, generation_paths)
  reasons <- rep.int(NA_character_, length(paths))
  generated <- !is.na(first_generation) & first_generation <= first_access
  reasons[generated] <- "generated_output"
  reasons[grepl("^/(?:proc|sys|dev)(?:/|$)", paths, perl = TRUE)] <-
    "pseudo_fs"
  reasons[is.na(reasons) & directory_event[first_access]] <- "non_regular"
  unresolved <- which(is.na(reasons))
  unresolved_exist <- file.exists(paths[unresolved])
  missing_non_pseudo <- sort(paths[unresolved[!unresolved_exist]],
                             method = "radix")
  if (length(missing_non_pseudo) > 0L) {
    stop(
      "native build successful read/exec path disappeared before hash ",
      "capture: ", missing_non_pseudo[[1L]], call. = FALSE
    )
  }
  surviving <- unresolved[unresolved_exist]
  regular <- fastkpc_full_cuda_fixed_sp_regular_file_mask(paths[surviving])
  reasons[surviving[!regular]] <- "non_regular"
  dependency_paths <- paths[surviving[regular]]
  if (length(dependency_paths) > 0L) {
    dependency_paths <- vapply(
      dependency_paths, normalizePath, character(1L),
      winslash = "/", mustWork = TRUE
    )
  }
  if (length(dependency_paths) == 0L) {
    stop("native build trace has no surviving regular dependencies",
         call. = FALSE)
  }
  exclusion_paths <- paths[!is.na(reasons)]
  exclusions <- data.frame(
    path = exclusion_paths,
    reason = reasons[!is.na(reasons)],
    stringsAsFactors = FALSE
  )
  exclusions <- exclusions[
    order(exclusions$path, method = "radix"), , drop = FALSE
  ]
  rownames(exclusions) <- as.character(seq_len(nrow(exclusions)))
  paths <- sort(
    unique(c(unname(dependency_paths), tracer_path)), method = "radix"
  )
  hashes <- unname(vapply(
    paths, fastkpc_full_cuda_fixed_sp_sha256_file, character(1L)
  ))
  files <- data.frame(path = paths, sha256 = hashes, stringsAsFactors = FALSE)
  rownames(files) <- as.character(seq_len(nrow(files)))
  tracer_sha256 <- hashes[[match(tracer_path, paths)]]
  commands <- fastkpc_full_cuda_fixed_sp_native_build_commands(
    exec_lines = exec_lines, exec_syscalls = exec_syscalls,
    exec_paths = exec_paths, build_working_dir = working_dir,
    files = files
  )
  build_environment <-
    fastkpc_full_cuda_fixed_sp_native_build_environment(trace_invocation)
  if (!identical(
        fastkpc_full_cuda_fixed_sp_sha256_file(trace_path), trace_sha256
      )) {
    stop("native build trace changed during dependency capture",
         call. = FALSE)
  }
  result <- list(
    schema_version = "full-cuda-ci-native-build-dependencies-v3",
    trace_semantics =
      "linux-strace-successful-read-exec-evidence-v3",
    trace_invocation = trace_invocation,
    build_working_dir = working_dir,
    trace_path = trace_path,
    trace_sha256 = trace_sha256,
    tracer_path = tracer_path,
    tracer_sha256 = tracer_sha256,
    dependency_count = as.integer(nrow(files)),
    files = files,
    exclusion_count = as.integer(nrow(exclusions)),
    exclusions = exclusions,
    command_projection_schema_version =
      "full-cuda-ci-native-build-command-projection-v1",
    command_count = as.integer(length(commands)),
    commands = commands,
    build_environment_schema_version =
      "full-cuda-ci-native-build-environment-v1",
    build_environment = build_environment,
    aggregate_sha256 = strrep("0", 64L)
  )
  result$aggregate_sha256 <-
    fastkpc_full_cuda_fixed_sp_native_build_dependency_hash(result)
  result
}

fastkpc_full_cuda_fixed_sp_capture_native_build_dependencies <- function(
    trace_path, build_working_dir, tracer_path, trace_invocation) {
  result <- fastkpc_full_cuda_fixed_sp_reconstruct_native_build_dependencies(
    trace_path = trace_path,
    build_working_dir = build_working_dir,
    tracer_path = tracer_path,
    trace_invocation = trace_invocation
  )
  fastkpc_full_cuda_fixed_sp_validate_native_build_dependencies(result)
  result
}

fastkpc_full_cuda_fixed_sp_verify_native_build_dependencies <- function(
    value) {
  trace_current <- tryCatch(
    fastkpc_full_cuda_fixed_sp_sha256_file(value$trace_path),
    error = function(error) ""
  )
  if (typeof(value$trace_sha256) == "character" &&
      length(value$trace_sha256) == 1L &&
      !identical(trace_current, value$trace_sha256)) {
    stop("native build trace changed during qualification", call. = FALSE)
  }
  fastkpc_full_cuda_fixed_sp_validate_native_build_dependencies(value)
  current <- unname(vapply(
    value$files$path,
    fastkpc_full_cuda_fixed_sp_sha256_file,
    character(1L)
  ))
  if (!identical(current, value$files$sha256)) {
    stop("native build dependency changed during qualification",
         call. = FALSE)
  }
  reconstructed <-
    fastkpc_full_cuda_fixed_sp_capture_native_build_dependencies(
      trace_path = value$trace_path,
      build_working_dir = value$build_working_dir,
      tracer_path = value$tracer_path,
      trace_invocation = value$trace_invocation
    )
  if (!identical(reconstructed, value)) {
    stop(
      "native build dependency evidence does not reconstruct from retained trace",
      call. = FALSE
    )
  }
  TRUE
}

fastkpc_full_cuda_fixed_sp_loaded_native_paths <- function() {
  unname(vapply(getLoadedDLLs(), function(dll) {
    normalizePath(
      dll[["path"]], winslash = "/", mustWork = FALSE
    )
  }, character(1L)))
}

fastkpc_full_cuda_fixed_sp_verify_loaded_native_library <- function(
    path, expected_sha256,
    loaded_dlls = getLoadedDLLs,
    mapped_records = .fastkpc_cuda_mapped_object_records,
    symbol_info = getNativeSymbolInfo,
    file_identity = .fastkpc_cuda_posix_file_identity,
    hash_file = fastkpc_full_cuda_fixed_sp_sha256_file,
    loaded_paths = NULL,
    mapped_paths = NULL) {
  if (!is.null(loaded_paths)) {
    loaded_dlls <- function() {
      paths <- loaded_paths()
      if (typeof(paths) != "character" || anyNA(paths)) {
        stop("loaded native library path snapshot is malformed", call. = FALSE)
      }
      normalized <- unname(vapply(
        paths, normalizePath, character(1L),
        winslash = "/", mustWork = FALSE
      ))
      target <- normalizePath(path, winslash = "/", mustWork = FALSE)
      if (!target %in% normalized) {
        return(list())
      }
      list(fastkpc_cuda = list(path = target))
    }
  }
  if (!is.null(mapped_paths)) {
    mapped_records <- function() {
      paths <- mapped_paths()
      if (typeof(paths) != "character" || anyNA(paths)) {
        stop("mapped native library path snapshot is malformed", call. = FALSE)
      }
      normalized <- unname(vapply(
        paths, normalizePath, character(1L),
        winslash = "/", mustWork = FALSE
      ))
      data.frame(
        path = normalized,
        live_path = normalized,
        deleted = FALSE,
        device_major_hex = NA_character_,
        device_minor_hex = NA_character_,
        inode = NA_character_,
        stringsAsFactors = FALSE
      )
    }
  }
  .fastkpc_cuda_verify_registered_library_identity(
    path,
    expected_sha256,
    loaded_dlls = loaded_dlls,
    mapped_records = mapped_records,
    symbol_info = symbol_info,
    file_identity = file_identity,
    hash_file = hash_file
  )
}

fastkpc_full_cuda_fixed_sp_execution_snapshot_hash <- function(provenance) {
  source_ids <- names(provenance$source_file_paths)
  native_input_ids <- names(provenance$native_build_input_paths)
  scalar_character <- function(value) {
    typeof(value) == "character" && length(value) == 1L &&
      !is.object(value) && is.null(attributes(value)) && !anyNA(value) &&
      nzchar(value) && !grepl("[\r\n]", value)
  }
  named_character_map <- function(value, expected_names) {
    typeof(value) == "character" && !is.object(value) &&
      identical(names(value), expected_names) &&
      identical(names(attributes(value)), "names") && !anyNA(value) &&
      all(nzchar(value)) && !any(grepl("[\r\n]", value))
  }
  closure <- list(
    source_closure_schema_version =
      provenance$source_closure_schema_version,
    source_discovery_semantics = provenance$source_discovery_semantics,
    source_project_root = provenance$source_project_root,
    direct_source_ids = provenance$direct_source_ids,
    source_ids = unname(source_ids),
    source_file_paths = provenance$source_file_paths,
    source_dependency_map = provenance$source_dependency_map,
    source_closure_count = provenance$source_closure_count
  )
  clean <- is.list(provenance) && !is.object(provenance) &&
    scalar_character(provenance$provenance_schema_version) &&
    identical(
      provenance$provenance_schema_version,
      "full-cuda-ci-execution-source-snapshot-v6"
    ) && scalar_character(provenance$provenance_mode) &&
    identical(
      provenance$provenance_mode, "working-tree-execution-snapshot-v1"
    ) && scalar_character(provenance$head_base_commit) &&
    grepl("^[0-9a-f]{40}$", provenance$head_base_commit) &&
    isTRUE(tryCatch(
      fastkpc_full_cuda_fixed_sp_validate_source_closure(closure),
      error = function(error) FALSE
    )) && scalar_character(provenance$source_closure_sha256) &&
    grepl("^[0-9a-f]{64}$", provenance$source_closure_sha256) &&
    named_character_map(provenance$source_file_sha256, source_ids) &&
    all(grepl("^[0-9a-f]{64}$", provenance$source_file_sha256)) &&
    identical(
      provenance$source_closure_sha256,
      fastkpc_full_cuda_fixed_sp_source_closure_hash(
        closure, provenance$source_file_sha256
      )
    ) && named_character_map(provenance$source_file_git_state, source_ids) &&
    all(provenance$source_file_git_state %in% c(
      "clean", "tracked-dirty", "untracked"
    )) && named_character_map(
      provenance$native_build_input_paths, native_input_ids
    ) && length(native_input_ids) > 0L &&
    identical(native_input_ids, sort(native_input_ids, method = "radix")) &&
    all(file.exists(provenance$native_build_input_paths)) &&
    !any(dir.exists(provenance$native_build_input_paths)) &&
    named_character_map(
      provenance$native_build_input_sha256, native_input_ids
    ) && all(grepl(
      "^[0-9a-f]{64}$", provenance$native_build_input_sha256
    )) && named_character_map(
      provenance$native_build_input_git_state, native_input_ids
    ) && all(provenance$native_build_input_git_state %in% c(
      "clean", "tracked-dirty", "untracked"
    )) && scalar_character(provenance$native_build_inputs_sha256) &&
    grepl("^[0-9a-f]{64}$", provenance$native_build_inputs_sha256) &&
    identical(
      provenance$native_build_inputs_sha256,
      fastkpc_full_cuda_fixed_sp_native_build_input_hash(
        provenance$native_build_input_paths,
        provenance$native_build_input_sha256
      )
    ) && isTRUE(tryCatch(
      fastkpc_full_cuda_fixed_sp_validate_native_build_dependencies(
        provenance$native_build_dependencies
      ),
      error = function(error) FALSE
    )) && typeof(provenance$relevant_sources_dirty_or_untracked) ==
      "logical" &&
    length(provenance$relevant_sources_dirty_or_untracked) == 1L &&
    !is.object(provenance$relevant_sources_dirty_or_untracked) &&
    is.null(attributes(
      provenance$relevant_sources_dirty_or_untracked
    )) && !anyNA(provenance$relevant_sources_dirty_or_untracked) &&
    identical(
      provenance$relevant_sources_dirty_or_untracked,
      any(c(
        provenance$source_file_git_state,
        provenance$native_build_input_git_state
      ) != "clean")
    ) && scalar_character(provenance$native_library_identity) &&
    identical(
      provenance$native_library_identity,
      "qualified-pinned-inode-sha-exact-registered-mapped-path-v3"
    ) && scalar_character(provenance$native_library_path) &&
    scalar_character(provenance$native_library_sha256) &&
    grepl("^[0-9a-f]{64}$", provenance$native_library_sha256)
  if (!isTRUE(clean)) {
    stop("execution source provenance is malformed", call. = FALSE)
  }
  lines <- c(
    paste0("schema_version=", provenance$provenance_schema_version),
    paste0("provenance_mode=", provenance$provenance_mode),
    paste0("head_base_commit=", provenance$head_base_commit),
    paste0("source_closure.schema=",
           provenance$source_closure_schema_version),
    paste0("source_closure.discovery=",
           provenance$source_discovery_semantics),
    paste0("source_closure.project_root=", provenance$source_project_root),
    paste0("source_closure.count=", provenance$source_closure_count),
    paste0("source_closure.sha256=", provenance$source_closure_sha256)
  )
  for (role in names(provenance$direct_source_ids)) {
    lines <- c(lines, paste0(
      "source_closure.root.", role, "=",
      provenance$direct_source_ids[[role]]
    ))
  }
  for (source_id in source_ids) {
    lines <- c(
      lines,
      paste0("source.", source_id, ".path=",
             provenance$source_file_paths[[source_id]]),
      paste0("source.", source_id, ".sha256=",
             provenance$source_file_sha256[[source_id]]),
      paste0("source.", source_id, ".git_state=",
             provenance$source_file_git_state[[source_id]]),
      paste0("source.", source_id, ".dependencies=",
             paste(provenance$source_dependency_map[[source_id]],
                   collapse = ","))
    )
  }
  for (input_id in native_input_ids) {
    lines <- c(
      lines,
      paste0("native_build_input.", input_id, ".path=",
             provenance$native_build_input_paths[[input_id]]),
      paste0("native_build_input.", input_id, ".sha256=",
             provenance$native_build_input_sha256[[input_id]]),
      paste0("native_build_input.", input_id, ".git_state=",
             provenance$native_build_input_git_state[[input_id]])
    )
  }
  lines <- c(
    lines,
    paste0("native_build_inputs.sha256=",
           provenance$native_build_inputs_sha256),
    paste0(
      "native_build_dependencies.schema=",
      provenance$native_build_dependencies$schema_version
    ),
    paste0(
      "native_build_dependencies.trace_semantics=",
      provenance$native_build_dependencies$trace_semantics
    ),
    paste0(
      "native_build_dependencies.trace_invocation=",
      provenance$native_build_dependencies$trace_invocation
    ),
    paste0(
      "native_build_dependencies.build_working_dir=",
      provenance$native_build_dependencies$build_working_dir
    ),
    paste0(
      "native_build_dependencies.trace_sha256=",
      provenance$native_build_dependencies$trace_sha256
    ),
    paste0(
      "native_build_dependencies.tracer_path=",
      provenance$native_build_dependencies$tracer_path
    ),
    paste0(
      "native_build_dependencies.tracer_sha256=",
      provenance$native_build_dependencies$tracer_sha256
    ),
    paste0(
      "native_build_dependencies.count=",
      provenance$native_build_dependencies$dependency_count
    ),
    paste0(
      "native_build_dependencies.exclusion_count=",
      provenance$native_build_dependencies$exclusion_count
    ),
    paste0(
      "native_build_dependencies.sha256=",
      provenance$native_build_dependencies$aggregate_sha256
    ),
    paste0(
      "relevant_sources_dirty_or_untracked=",
      if (provenance$relevant_sources_dirty_or_untracked) "true" else "false"
    ),
    paste0("native.identity=", provenance$native_library_identity),
    paste0("native.path=", provenance$native_library_path),
    paste0("native.sha256=", provenance$native_library_sha256)
  )
  payload <- charToRaw(enc2utf8(paste0(paste(lines, collapse = "\n"), "\n")))
  unname(digest::digest(payload, algo = "sha256", serialize = FALSE))
}

fastkpc_full_cuda_fixed_sp_capture_execution_provenance <- function(
    source_closure, expected_source_sha256, native_library_path,
    native_build_input_paths, expected_native_build_input_sha256,
    native_build_dependencies, expected_native_library_sha256,
    loaded_paths = fastkpc_full_cuda_fixed_sp_loaded_native_paths,
    mapped_records = .fastkpc_cuda_mapped_object_records) {
  fastkpc_full_cuda_fixed_sp_validate_source_closure(source_closure)
  source_ids <- source_closure$source_ids
  if (typeof(expected_source_sha256) != "character" ||
      is.object(expected_source_sha256) ||
      !identical(names(expected_source_sha256), source_ids) ||
      !identical(names(attributes(expected_source_sha256)), "names") ||
      anyNA(expected_source_sha256) ||
      !all(grepl("^[0-9a-f]{64}$", expected_source_sha256))) {
    stop("execution source preload snapshot is malformed", call. = FALSE)
  }
  current_hashes <- vapply(
    source_closure$source_file_paths,
    fastkpc_full_cuda_fixed_sp_sha256_file,
    character(1L)
  )
  if (!identical(unname(current_hashes),
                 unname(expected_source_sha256))) {
    stop("execution source changed between preload hash and capture",
         call. = FALSE)
  }
  git_states <- vapply(
    source_closure$source_file_paths,
    fastkpc_full_cuda_fixed_sp_git_source_state,
    character(1L)
  )
  native_paths <- vapply(
    native_build_input_paths,
    normalizePath, character(1L), winslash = "/", mustWork = TRUE
  )
  names(native_paths) <- names(native_build_input_paths)
  native_input_ids <- names(native_paths)
  native_inputs_clean <- typeof(native_build_input_paths) == "character" &&
    !is.object(native_build_input_paths) && length(native_input_ids) > 0L &&
    identical(native_input_ids, sort(native_input_ids, method = "radix")) &&
    !anyDuplicated(native_input_ids) &&
    typeof(expected_native_build_input_sha256) == "character" &&
    !is.object(expected_native_build_input_sha256) &&
    identical(names(expected_native_build_input_sha256), native_input_ids) &&
    identical(names(attributes(expected_native_build_input_sha256)), "names") &&
    !anyNA(expected_native_build_input_sha256) &&
    all(grepl("^[0-9a-f]{64}$", expected_native_build_input_sha256))
  if (!isTRUE(native_inputs_clean)) {
    stop("native build input preload snapshot is malformed", call. = FALSE)
  }
  current_native_input_hashes <- vapply(
    native_paths, fastkpc_full_cuda_fixed_sp_sha256_file, character(1L)
  )
  if (!identical(
    unname(current_native_input_hashes),
    unname(expected_native_build_input_sha256)
  )) {
    stop("native build input changed between preload hash and capture",
         call. = FALSE)
  }
  native_input_git_states <- vapply(
    native_paths, fastkpc_full_cuda_fixed_sp_git_source_state, character(1L)
  )
  fastkpc_full_cuda_fixed_sp_validate_native_build_dependencies(
    native_build_dependencies
  )
  fastkpc_full_cuda_fixed_sp_verify_native_build_dependencies(
    native_build_dependencies
  )
  native_path <- normalizePath(
    native_library_path, winslash = "/", mustWork = TRUE
  )
  fastkpc_full_cuda_fixed_sp_verify_loaded_native_library(
    native_path, expected_native_library_sha256,
    loaded_paths = loaded_paths, mapped_records = mapped_records
  )
  head_base_commit <- system2(
    "git", c("rev-parse", "HEAD"), stdout = TRUE, stderr = FALSE
  )
  if (length(head_base_commit) != 1L ||
      !grepl("^[0-9a-f]{40}$", head_base_commit)) {
    stop("execution source base commit is unavailable", call. = FALSE)
  }
  provenance <- list(
    provenance_schema_version =
      "full-cuda-ci-execution-source-snapshot-v6",
    provenance_mode = "working-tree-execution-snapshot-v1",
    head_base_commit = unname(head_base_commit),
    source_closure_schema_version =
      source_closure$source_closure_schema_version,
    source_discovery_semantics =
      source_closure$source_discovery_semantics,
    source_project_root = source_closure$source_project_root,
    source_closure_count = source_closure$source_closure_count,
    source_closure_sha256 =
      fastkpc_full_cuda_fixed_sp_source_closure_hash(
        source_closure, current_hashes
      ),
    direct_source_ids = source_closure$direct_source_ids,
    source_dependency_map = source_closure$source_dependency_map,
    source_file_paths = source_closure$source_file_paths,
    source_file_sha256 = current_hashes,
    source_file_git_state = git_states,
    native_build_input_paths = native_paths,
    native_build_input_sha256 = current_native_input_hashes,
    native_build_input_git_state = native_input_git_states,
    native_build_inputs_sha256 =
      fastkpc_full_cuda_fixed_sp_native_build_input_hash(
        native_paths, current_native_input_hashes
      ),
    native_build_dependencies = native_build_dependencies,
    relevant_sources_dirty_or_untracked = any(c(
      git_states, native_input_git_states
    ) != "clean"),
    native_library_identity =
      "qualified-pinned-inode-sha-exact-registered-mapped-path-v3",
    native_library_path = native_path,
    native_library_sha256 = expected_native_library_sha256
  )
  provenance$execution_snapshot_sha256 <-
    fastkpc_full_cuda_fixed_sp_execution_snapshot_hash(provenance)
  provenance$execution_sources_unchanged_after_run <- FALSE
  provenance
}

fastkpc_full_cuda_fixed_sp_verify_execution_provenance <- function(
    provenance,
    loaded_paths = fastkpc_full_cuda_fixed_sp_loaded_native_paths,
    mapped_records = .fastkpc_cuda_mapped_object_records) {
  expected_names <- c(
    "provenance_schema_version", "provenance_mode", "head_base_commit",
    "source_closure_schema_version", "source_discovery_semantics",
    "source_project_root", "source_closure_count", "source_closure_sha256",
    "direct_source_ids", "source_dependency_map", "source_file_paths",
    "source_file_sha256", "source_file_git_state",
    "native_build_input_paths", "native_build_input_sha256",
    "native_build_input_git_state", "native_build_inputs_sha256",
    "native_build_dependencies",
    "relevant_sources_dirty_or_untracked", "native_library_identity",
    "native_library_path", "native_library_sha256",
    "execution_snapshot_sha256", "execution_sources_unchanged_after_run"
  )
  if (!is.list(provenance) || !identical(names(provenance), expected_names) ||
      typeof(provenance$execution_snapshot_sha256) != "character" ||
      length(provenance$execution_snapshot_sha256) != 1L ||
      is.object(provenance$execution_snapshot_sha256) ||
      !is.null(attributes(provenance$execution_snapshot_sha256)) ||
      !grepl("^[0-9a-f]{64}$", provenance$execution_snapshot_sha256) ||
      !identical(
        provenance$execution_snapshot_sha256,
        fastkpc_full_cuda_fixed_sp_execution_snapshot_hash(provenance)
      ) || typeof(provenance$execution_sources_unchanged_after_run) !=
        "logical" ||
      length(provenance$execution_sources_unchanged_after_run) != 1L ||
      !is.null(attributes(
        provenance$execution_sources_unchanged_after_run
      ))) {
    stop("execution source provenance verification input is malformed",
         call. = FALSE)
  }
  source_and_library_unchanged <- tryCatch({
    current_closure <-
      fastkpc_full_cuda_fixed_sp_discover_execution_source_closure(
        root_sources = provenance$direct_source_ids,
        project_root = provenance$source_project_root
      )
    current_source_hashes <- vapply(
      current_closure$source_file_paths,
      fastkpc_full_cuda_fixed_sp_sha256_file,
      character(1L)
    )
    current_head <- system2(
      "git", c("rev-parse", "HEAD"), stdout = TRUE, stderr = FALSE
    )
    current_native_hash <- fastkpc_full_cuda_fixed_sp_sha256_file(
      provenance$native_library_path
    )
    identical(unname(current_head), provenance$head_base_commit) &&
      identical(current_closure$source_ids,
                names(provenance$source_file_paths)) &&
      identical(current_closure$direct_source_ids,
                provenance$direct_source_ids) &&
      identical(current_closure$source_dependency_map,
                provenance$source_dependency_map) &&
      identical(unname(current_source_hashes),
                unname(provenance$source_file_sha256)) &&
      identical(
        fastkpc_full_cuda_fixed_sp_source_closure_hash(
          current_closure, current_source_hashes
        ),
        provenance$source_closure_sha256
      ) && identical(current_native_hash, provenance$native_library_sha256) &&
      isTRUE(fastkpc_full_cuda_fixed_sp_verify_loaded_native_library(
        provenance$native_library_path,
        provenance$native_library_sha256,
        loaded_paths = loaded_paths, mapped_records = mapped_records
      ))
  }, error = function(error) FALSE)
  if (!isTRUE(source_and_library_unchanged)) {
    stop("execution source snapshot changed during qualification",
         call. = FALSE)
  }
  native_inputs_unchanged <- tryCatch({
    current_native_input_hashes <- vapply(
      provenance$native_build_input_paths,
      fastkpc_full_cuda_fixed_sp_sha256_file,
      character(1L)
    )
    identical(
      unname(current_native_input_hashes),
      unname(provenance$native_build_input_sha256)
    ) && identical(
      fastkpc_full_cuda_fixed_sp_native_build_input_hash(
        provenance$native_build_input_paths, current_native_input_hashes
      ),
      provenance$native_build_inputs_sha256
    )
  }, error = function(error) FALSE)
  if (!isTRUE(native_inputs_unchanged)) {
    stop("native build input snapshot changed during qualification",
         call. = FALSE)
  }
  fastkpc_full_cuda_fixed_sp_verify_native_build_dependencies(
    provenance$native_build_dependencies
  )
  provenance$execution_sources_unchanged_after_run <- TRUE
  provenance
}

fastkpc_full_cuda_fixed_sp_guarded_source <- function(
    file, source_closure, ...) {
  fastkpc_full_cuda_fixed_sp_validate_source_closure(source_closure)
  identity <- fastkpc_full_cuda_fixed_sp_source_identity(
    file, source_closure$source_project_root
  )
  if (!identity$id %in% source_closure$source_ids) {
    stop("runtime source() escaped the authenticated execution closure: ",
         identity$id, call. = FALSE)
  }
  base_source <- get("source", envir = baseenv(), inherits = FALSE)
  base_source(source_closure$source_file_paths[[identity$id]], ...)
}

fastkpc_full_cuda_fixed_sp_qualification_csv_frame <- function(value) {
  if (!is.data.frame(value)) {
    stop("qualification CSV payload must be a data frame", call. = FALSE)
  }
  result <- value
  list_fields <- names(result)[vapply(result, is.list, logical(1L))]
  for (field in list_fields) {
    result[[field]] <- vapply(result[[field]], function(element) {
      if (length(element) == 0L) "" else paste(element, collapse = ";")
    }, character(1L))
  }
  result
}

fastkpc_full_cuda_fixed_sp_write_qualification_json <- function(value, path) {
  if (!is.character(path) || length(path) != 1L || is.na(path) ||
      !nzchar(path) || dir.exists(path)) {
    stop("qualification JSON output path is invalid", call. = FALSE)
  }
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("qualification JSON output requires jsonlite", call. = FALSE)
  }
  jsonlite::write_json(
    value, path, auto_unbox = TRUE, pretty = TRUE,
    null = "null", na = "null", digits = 17L, always_decimal = TRUE
  )
  invisible(path)
}

fastkpc_full_cuda_fixed_sp_qualification_validate_summary_counts <- function(
    summary) {
  if (!is.list(summary) || is.object(summary) || is.null(names(summary)) ||
      anyDuplicated(names(summary)) || "pass" %in% names(summary)) {
    stop("qualification summary count types are invalid", call. = FALSE)
  }
  expected <-
    fastkpc_full_cuda_fixed_sp_phase3c_expected_counts("qualification")
  if (!all(names(expected) %in% names(summary))) {
    stop("qualification summary count types are incomplete", call. = FALSE)
  }
  bare_exact <- all(vapply(names(expected), function(field) {
    value <- summary[[field]]
    reference <- expected[[field]]
    typeof(value) == typeof(reference) && length(value) == 1L &&
      !is.object(value) && is.null(attributes(value)) && !anyNA(value) &&
      identical(value, reference)
  }, logical(1L)))
  if (!isTRUE(bare_exact) ||
      !identical(summary[names(expected)], expected)) {
    stop("qualification summary count types or values are invalid",
         call. = FALSE)
  }
  TRUE
}

fastkpc_full_cuda_fixed_sp_qualification_summary_schema <- function() {
  input_names <- c(
    "scope", "scope_subset_hash", "ordered_setup_key_digest",
    "ordered_target_key_digest", "route_status_hash", "numeric_hash",
    "setup_count", "target_count", "penalty_root_matrix_count",
    "penalty_root_row_count", "H_root_matrix_count", "planned_cholesky_count",
    "planned_qr_count", "planned_svd_count", "executed_cholesky_count",
    "executed_qr_count", "executed_svd_count", "cholesky_to_svd_count",
    "qr_to_svd_count", "svd_finite_high_count", "svd_nonfinite_count",
    "all_safe_batch_count", "mixed_batch_count", "all_stable_batch_count",
    "true_batched_subgroup_count", "true_batched_target_count",
    "cholesky_single_target_count", "whole_batch_true_batched_count",
    "stable_not_implemented_count", "stable_reroute_count",
    "non_ok_status_count", "root_rank_mismatch_count",
    "aggregate_penalty_factor_count", "aggregate_svd_b_build_count",
    "aggregate_penalty_root_rank_mismatch_count",
    "aggregate_penalty_root_pivot_mismatch_count",
    "aggregate_penalty_root_d2h_count", "aggregate_penalty_root_d2h_bytes",
    "workspace_grow_count_after_warmup",
    "stable_workspace_grow_count_after_warmup",
    "per_target_allocation_count_after_warmup",
    "per_target_handle_create_count", "cuda_device_synchronize_count",
    "target_level_stable_sync_count", "implicit_residual_d2h_count",
    "all_output_slot_leases_released", "invalid_output_init_count",
    "cpu_fallback_count", "unknown_fallback_count", "approximate_backend_count",
    "qr_checkpoint_record_count", "qr_checkpoint_wait_count",
    "svd_checkpoint_record_count", "svd_checkpoint_wait_count",
    "max_residual_abs_diff", "max_residual_relative_l2_diff",
    "max_fitted_abs_diff", "max_fitted_relative_l2_diff",
    "qualification_subset_hash"
  )
  publication_names <- c(
    "artifact_schema_version", "catalog_authenticated", "provenance_mode",
    "head_base_commit", "source_closure_schema_version",
    "source_closure_count", "source_closure_sha256",
    "execution_snapshot_sha256", "relevant_sources_dirty_or_untracked",
    "native_build_inputs_sha256", "native_build_dependency_count",
    "native_build_dependencies_sha256", "native_build_trace_sha256",
    "native_build_tracer_sha256", "native_library_sha256",
    "execution_sources_unchanged_after_run",
    "elapsed_seconds", "stage_timing_total_seconds", "payload_file_count"
  )
  dcov_names <- c(
    "qualification_dcov_logical_test_count",
    "qualification_dcov_near_alpha_count",
    "qualification_dcov_unique_residual_key_count",
    "qualification_dcov_max_absolute_p_value_difference",
    "qualification_dcov_decision_flip_count",
    "qualification_dcov_near_alpha_decision_flip_count",
    "qualification_dcov_backend_error_count",
    "qualification_dcov_spectra_fallback_count",
    "qualification_dcov_logical_ids_hash",
    "qualification_dcov_residual_key_hash",
    "qualification_dcov_rows_hash"
  )
  make_types <- function(names, character, double, logical) {
    types <- setNames(rep.int("integer", length(names)), names)
    types[character] <- "character"
    types[double] <- "double"
    types[logical] <- "logical"
    types
  }
  input_types <- make_types(
    input_names,
    character = c(
      "scope", "scope_subset_hash", "ordered_setup_key_digest",
      "ordered_target_key_digest", "route_status_hash", "numeric_hash",
      "qualification_subset_hash"
    ),
    double = c(
      "aggregate_penalty_root_d2h_bytes", "max_residual_abs_diff",
      "max_residual_relative_l2_diff", "max_fitted_abs_diff",
      "max_fitted_relative_l2_diff"
    ),
    logical = "all_output_slot_leases_released"
  )
  publication_types <- make_types(
    publication_names,
    character = c(
      "artifact_schema_version", "provenance_mode", "head_base_commit",
      "source_closure_schema_version", "source_closure_sha256",
      "execution_snapshot_sha256", "native_build_inputs_sha256",
      "native_build_dependencies_sha256", "native_build_trace_sha256",
      "native_build_tracer_sha256", "native_library_sha256"
    ),
    double = c("elapsed_seconds", "stage_timing_total_seconds"),
    logical = c(
      "catalog_authenticated", "relevant_sources_dirty_or_untracked",
      "execution_sources_unchanged_after_run"
    )
  )
  dcov_types <- make_types(
    dcov_names,
    character = c(
      "qualification_dcov_logical_ids_hash",
      "qualification_dcov_residual_key_hash",
      "qualification_dcov_rows_hash"
    ),
    double = "qualification_dcov_max_absolute_p_value_difference",
    logical = character()
  )
  final_names <- c(input_names, dcov_names, publication_names)
  final_types <- c(input_types, dcov_types, publication_types)
  clean <- !anyDuplicated(input_names) && !anyDuplicated(dcov_names) &&
    !anyDuplicated(publication_names) &&
    !anyDuplicated(final_names) && identical(names(input_types), input_names) &&
    identical(names(dcov_types), dcov_names) &&
    identical(names(publication_types), publication_names) &&
    identical(names(final_types), final_names) &&
    all(input_types %in% c("character", "integer", "double", "logical")) &&
    all(publication_types %in%
        c("character", "integer", "double", "logical"))
  if (!isTRUE(clean)) {
    stop("internal qualification summary schema is malformed", call. = FALSE)
  }
  list(
    input_names = input_names,
    input_types = input_types,
    dcov_names = dcov_names,
    dcov_types = dcov_types,
    publication_names = publication_names,
    publication_types = publication_types,
    final_names = final_names,
    final_types = final_types
  )
}

fastkpc_full_cuda_fixed_sp_qualification_summary_has_schema <- function(
    summary, expected_names, expected_types) {
  is.list(summary) && !is.object(summary) &&
    identical(names(summary), expected_names) &&
    !anyDuplicated(names(summary)) &&
    identical(names(expected_types), expected_names) &&
    all(vapply(expected_names, function(field) {
      value <- summary[[field]]
      typeof(value) == expected_types[[field]] && length(value) == 1L &&
        !is.object(value) && is.null(attributes(value)) && !anyNA(value)
    }, logical(1L)))
}

fastkpc_full_cuda_fixed_sp_qualification_validate_input_summary <- function(
    summary) {
  schema <- fastkpc_full_cuda_fixed_sp_qualification_summary_schema()
  clean <- fastkpc_full_cuda_fixed_sp_qualification_summary_has_schema(
    summary, schema$input_names, schema$input_types
  ) && identical(summary$scope, "qualification")
  if (!isTRUE(clean)) {
    stop("qualification input summary schema is invalid", call. = FALSE)
  }
  fastkpc_full_cuda_fixed_sp_qualification_validate_summary_counts(summary)
  TRUE
}

fastkpc_full_cuda_fixed_sp_qualification_validate_dcov_summary <- function(
    summary) {
  schema <- fastkpc_full_cuda_fixed_sp_qualification_summary_schema()
  if (!is.list(summary) ||
      length(setdiff(schema$dcov_names, names(summary))) > 0L) {
    stop("qualification dCov summary schema is invalid", call. = FALSE)
  }
  values <- summary[schema$dcov_names]
  clean <-
    fastkpc_full_cuda_fixed_sp_qualification_summary_has_schema(
      values, schema$dcov_names, schema$dcov_types
    ) && identical(
      values$qualification_dcov_logical_test_count, 3808L
    ) && identical(
      values$qualification_dcov_near_alpha_count, 1478L
    ) && identical(
      values$qualification_dcov_unique_residual_key_count, 6143L
    ) && is.finite(
      values$qualification_dcov_max_absolute_p_value_difference
    ) && values$qualification_dcov_max_absolute_p_value_difference <
      1e-10 && identical(
      values$qualification_dcov_decision_flip_count, 0L
    ) && identical(
      values$qualification_dcov_near_alpha_decision_flip_count, 0L
    ) && identical(
      values$qualification_dcov_backend_error_count, 0L
    ) && identical(
      values$qualification_dcov_spectra_fallback_count, 0L
    ) && all(vapply(
      schema$dcov_names[seq.int(length(schema$dcov_names) - 2L,
                                length(schema$dcov_names))],
      function(field) fastkpc_full_cuda_fixed_sp_is_bare_sha256(
        values[[field]]
      ), logical(1L)
    ))
  if (!isTRUE(clean)) {
    stop("qualification dCov summary schema is invalid", call. = FALSE)
  }
  TRUE
}

fastkpc_full_cuda_fixed_sp_qualification_validate_published_summary <- function(
    summary) {
  schema <- fastkpc_full_cuda_fixed_sp_qualification_summary_schema()
  clean <- fastkpc_full_cuda_fixed_sp_qualification_summary_has_schema(
    summary, schema$final_names, schema$final_types
  ) && identical(summary$scope, "qualification") && identical(
    summary$artifact_schema_version,
    "full-cuda-ci-fixed-sp-qualification-v6"
  ) && identical(summary$catalog_authenticated, TRUE) &&
    identical(summary$execution_sources_unchanged_after_run, TRUE) &&
    is.finite(summary$elapsed_seconds) && summary$elapsed_seconds >= 0 &&
    is.finite(summary$stage_timing_total_seconds) &&
    summary$stage_timing_total_seconds >= 0 &&
    identical(
      summary$payload_file_count,
      as.integer(length(
        fastkpc_full_cuda_fixed_sp_qualification_payload_names()
      ))
    )
  if (!isTRUE(clean)) {
    stop("qualification published summary schema is invalid", call. = FALSE)
  }
  fastkpc_full_cuda_fixed_sp_qualification_validate_summary_counts(summary)
  fastkpc_full_cuda_fixed_sp_qualification_validate_dcov_summary(summary)
  TRUE
}

fastkpc_full_cuda_fixed_sp_qualification_validate_catalog_evidence <- function(
    catalog_records, summary, authoritative) {
  required <- c(
    "scope", "authenticated", "catalog_open_count", "setup_count",
    "target_count", "scope_subset_hash", "ordered_setup_key_digest",
    "ordered_target_key_digest"
  )
  bare_scalar <- function(value, type) {
    typeof(value) == type && length(value) == 1L && !is.object(value) &&
      is.null(attributes(value)) && !anyNA(value)
  }
  authoritative_names <- c(
    "qualification_subset_hash", "ordered_setup_key_digest",
    "ordered_target_key_digest", "setup_count", "target_count"
  )
  authoritative_clean <- is.list(authoritative) &&
    !is.object(authoritative) &&
    identical(names(authoritative), authoritative_names) &&
    all(vapply(authoritative_names[seq_len(3L)], function(field) {
      bare_scalar(authoritative[[field]], "character") &&
        fastkpc_full_cuda_fixed_sp_is_bare_sha256(authoritative[[field]])
    }, logical(1L))) &&
    bare_scalar(authoritative$setup_count, "integer") &&
    bare_scalar(authoritative$target_count, "integer") &&
    identical(
      authoritative$qualification_subset_hash,
      fastkpc_full_cuda_fixed_sp_catalog_contract()$qualification_subset_hash
    ) && identical(authoritative$setup_count, 2061L) &&
    identical(authoritative$target_count, 6143L)
  summary_clean <- is.list(summary) && !is.object(summary) &&
    all(authoritative_names[seq_len(3L)] %in% names(summary)) &&
    all(vapply(authoritative_names[seq_len(3L)], function(field) {
      bare_scalar(summary[[field]], "character") &&
        fastkpc_full_cuda_fixed_sp_is_bare_sha256(summary[[field]])
    }, logical(1L)))
  records_clean <- is.data.frame(catalog_records) &&
    nrow(catalog_records) == 1L &&
    all(required %in% names(catalog_records)) &&
    bare_scalar(catalog_records$scope, "character") &&
    identical(catalog_records$scope, "qualification") &&
    bare_scalar(catalog_records$authenticated, "logical") &&
    identical(catalog_records$authenticated, TRUE) &&
    all(vapply(c(
      "catalog_open_count", "setup_count", "target_count"
    ), function(field) {
      bare_scalar(catalog_records[[field]], "integer")
    }, logical(1L))) &&
    identical(catalog_records$catalog_open_count, 1L) &&
    identical(catalog_records$setup_count, 2061L) &&
    identical(catalog_records$target_count, 6143L) &&
    bare_scalar(catalog_records$scope_subset_hash, "character") &&
    fastkpc_full_cuda_fixed_sp_is_bare_sha256(
      catalog_records$scope_subset_hash
    ) && all(vapply(c(
      "ordered_setup_key_digest", "ordered_target_key_digest"
    ), function(field) {
      bare_scalar(catalog_records[[field]], "character") &&
        fastkpc_full_cuda_fixed_sp_is_bare_sha256(catalog_records[[field]])
    }, logical(1L)))
  exact <- authoritative_clean && summary_clean && records_clean &&
    identical(
      catalog_records$scope_subset_hash,
      authoritative$qualification_subset_hash
    ) && identical(
      summary$qualification_subset_hash,
      authoritative$qualification_subset_hash
    ) && identical(
      summary$ordered_setup_key_digest,
      authoritative$ordered_setup_key_digest
    ) && identical(
      summary$ordered_target_key_digest,
      authoritative$ordered_target_key_digest
    ) && identical(
      catalog_records$ordered_setup_key_digest,
      authoritative$ordered_setup_key_digest
    ) && identical(
      catalog_records$ordered_target_key_digest,
      authoritative$ordered_target_key_digest
    ) && identical(catalog_records$setup_count, authoritative$setup_count) &&
    identical(catalog_records$target_count, authoritative$target_count)
  if (!isTRUE(exact)) {
    stop("qualification catalog evidence does not match canonical authentication",
         call. = FALSE)
  }
  TRUE
}

fastkpc_full_cuda_fixed_sp_qualification_reopen_catalog_evidence <- function(
    phase0_dir, phase1_dir, phase2_dir, data_path) {
  catalog <- fastkpc_full_cuda_open_fixed_sp_catalog(
    phase0_dir = phase0_dir,
    phase1_dir = phase1_dir,
    phase2_dir = phase2_dir,
    data_path = data_path
  )
  qualification <- fastkpc_full_cuda_fixed_sp_scope(
    catalog, "qualification"
  )
  setup_keys <- as.character(
    qualification$setup_rows$prepared_s_key_sha256
  )
  target_keys <- as.character(
    qualification$target_rows$residual_key_sha256
  )
  target_prepared_s_keys <- as.character(
    qualification$target_rows$prepared_s_key_sha256
  )
  if (length(setup_keys) != 2061L || length(target_keys) != 6143L ||
      length(target_prepared_s_keys) != 6143L ||
      anyNA(setup_keys) || anyNA(target_keys) ||
      anyNA(target_prepared_s_keys) ||
      anyDuplicated(setup_keys) || anyDuplicated(target_keys) ||
      !all(grepl("^[0-9a-f]{64}$", setup_keys)) ||
      !all(grepl("^[0-9a-f]{64}$", target_keys)) ||
      !all(grepl("^[0-9a-f]{64}$", target_prepared_s_keys)) ||
      !identical(setup_keys, sort(setup_keys, method = "radix")) ||
      !all(target_prepared_s_keys %in% setup_keys) ||
      !identical(
        order(target_prepared_s_keys, target_keys, method = "radix"),
        seq_along(target_keys)
      )) {
    stop("canonical qualification catalog ordering is malformed",
         call. = FALSE)
  }
  list(
    qualification_subset_hash =
      catalog$catalog_contract$qualification_subset_hash,
    ordered_setup_key_digest =
      fastkpc_full_cuda_census_key_set_hash(setup_keys),
    ordered_target_key_digest =
      fastkpc_full_cuda_census_key_set_hash(target_keys),
    setup_count = as.integer(length(setup_keys)),
    target_count = as.integer(length(target_keys)),
    setup_keys = setup_keys,
    target_keys = target_keys,
    target_prepared_s_keys = target_prepared_s_keys
  )
}

fastkpc_full_cuda_fixed_sp_qualification_validate_record_identity <- function(
    catalog_records, summary, authoritative, setup_records, batch_records,
    target_records) {
  authoritative_names <- c(
    "qualification_subset_hash", "ordered_setup_key_digest",
    "ordered_target_key_digest", "setup_count", "target_count",
    "setup_keys", "target_keys", "target_prepared_s_keys"
  )
  bare_sha_vector <- function(value, size, unique = FALSE) {
    clean <- typeof(value) == "character" && length(value) == size &&
      !is.object(value) && is.null(attributes(value)) && !anyNA(value) &&
      all(grepl("^[0-9a-f]{64}$", value))
    isTRUE(clean) && (!isTRUE(unique) || !anyDuplicated(value))
  }
  authoritative_clean <- is.list(authoritative) &&
    !is.object(authoritative) &&
    identical(names(authoritative), authoritative_names) &&
    bare_sha_vector(authoritative$setup_keys, 2061L, unique = TRUE) &&
    bare_sha_vector(authoritative$target_keys, 6143L, unique = TRUE) &&
    bare_sha_vector(authoritative$target_prepared_s_keys, 6143L) &&
    identical(
      authoritative$setup_keys,
      sort(authoritative$setup_keys, method = "radix")
    ) && all(
      authoritative$target_prepared_s_keys %in% authoritative$setup_keys
    ) && identical(
      order(
        authoritative$target_prepared_s_keys,
        authoritative$target_keys,
        method = "radix"
      ),
      seq_along(authoritative$target_keys)
    )
  if (!isTRUE(authoritative_clean)) {
    stop("qualification published rows do not match canonical catalog identity",
         call. = FALSE)
  }
  fastkpc_full_cuda_fixed_sp_qualification_validate_catalog_evidence(
    catalog_records = catalog_records,
    summary = summary,
    authoritative = authoritative[seq_len(5L)]
  )

  summary_identity_names <- c(
    "scope_subset_hash", "qualification_subset_hash",
    "ordered_setup_key_digest", "ordered_target_key_digest",
    "route_status_hash", "numeric_hash", "setup_count", "target_count"
  )
  bare_scalar <- function(value, type) {
    typeof(value) == type && length(value) == 1L && !is.object(value) &&
      is.null(attributes(value)) && !anyNA(value)
  }
  summary_clean <- is.list(summary) && !is.object(summary) &&
    all(summary_identity_names %in% names(summary)) &&
    all(vapply(summary_identity_names[seq_len(6L)], function(field) {
      bare_scalar(summary[[field]], "character") &&
        fastkpc_full_cuda_fixed_sp_is_bare_sha256(summary[[field]])
    }, logical(1L))) &&
    bare_scalar(summary$setup_count, "integer") &&
    bare_scalar(summary$target_count, "integer")
  record_keys_clean <- is.data.frame(setup_records) &&
    is.data.frame(batch_records) && is.data.frame(target_records) &&
    all(c("prepared_s_key_sha256") %in% names(setup_records)) &&
    all(c("prepared_s_key_sha256") %in% names(batch_records)) &&
    all(c(
      "prepared_s_key_sha256", "residual_key_sha256", "planned_route",
      "executed_route", "reroute_reason", "solver_status",
      "fitted_numeric_hash", "residual_numeric_hash"
    ) %in% names(target_records)) &&
    identical(
      setup_records$prepared_s_key_sha256, authoritative$setup_keys
    ) && identical(
      batch_records$prepared_s_key_sha256, authoritative$setup_keys
    ) && identical(
      target_records$residual_key_sha256, authoritative$target_keys
    ) && identical(
      target_records$prepared_s_key_sha256,
      authoritative$target_prepared_s_keys
    )
  if (!isTRUE(summary_clean) || !isTRUE(record_keys_clean)) {
    stop("qualification published rows do not match canonical catalog identity",
         call. = FALSE)
  }

  setup_digest <- fastkpc_full_cuda_census_key_set_hash(
    setup_records$prepared_s_key_sha256
  )
  target_digest <- fastkpc_full_cuda_census_key_set_hash(
    target_records$residual_key_sha256
  )
  route_status_hash <- fastkpc_full_cuda_census_metadata_hash(list(
    target_records$residual_key_sha256,
    target_records$planned_route,
    target_records$executed_route,
    target_records$reroute_reason,
    target_records$solver_status
  ))
  numeric_hash <- fastkpc_full_cuda_census_metadata_hash(list(
    target_records$residual_key_sha256,
    target_records$fitted_numeric_hash,
    target_records$residual_numeric_hash
  ))
  exact <- identical(
    setup_digest, authoritative$ordered_setup_key_digest
  ) && identical(
    target_digest, authoritative$ordered_target_key_digest
  ) && identical(
    setup_digest, catalog_records$ordered_setup_key_digest[[1L]]
  ) && identical(
    target_digest, catalog_records$ordered_target_key_digest[[1L]]
  ) && identical(summary$ordered_setup_key_digest, setup_digest) &&
    identical(summary$ordered_target_key_digest, target_digest) &&
    identical(
      summary$scope_subset_hash, authoritative$qualification_subset_hash
    ) && identical(
      summary$qualification_subset_hash,
      authoritative$qualification_subset_hash
    ) && identical(summary$route_status_hash, route_status_hash) &&
    identical(summary$numeric_hash, numeric_hash) &&
    identical(summary$setup_count, authoritative$setup_count) &&
    identical(summary$target_count, authoritative$target_count)
  if (!isTRUE(exact)) {
    stop("qualification published rows do not match canonical catalog identity",
         call. = FALSE)
  }
  list(
    catalog_authenticated = TRUE,
    qualification_subset_hash = authoritative$qualification_subset_hash,
    ordered_setup_key_digest = setup_digest,
    ordered_target_key_digest = target_digest,
    route_status_hash = route_status_hash,
    numeric_hash = numeric_hash,
    setup_count = authoritative$setup_count,
    target_count = authoritative$target_count
  )
}

fastkpc_full_cuda_fixed_sp_publish_qualification_staging <- function(
    staging_dir, output_dir, publication_order, expected_sha256,
    move_file = file.rename) {
  scalar_path <- function(value) {
    typeof(value) == "character" && length(value) == 1L &&
      !is.object(value) && is.null(attributes(value)) && !anyNA(value) &&
      nzchar(value)
  }
  order_clean <- typeof(publication_order) == "character" &&
    !is.object(publication_order) && is.null(attributes(publication_order)) &&
    length(publication_order) >= 3L && !anyNA(publication_order) &&
    all(nzchar(publication_order)) && !anyDuplicated(publication_order) &&
    identical(tail(publication_order, 2L),
              c("manifest.json", "summary.json"))
  hashes_clean <- typeof(expected_sha256) == "character" &&
    !is.object(expected_sha256) &&
    identical(names(expected_sha256), publication_order) &&
    identical(names(attributes(expected_sha256)), "names") &&
    !anyNA(expected_sha256) &&
    all(grepl("^[0-9a-f]{64}$", expected_sha256))
  if (!scalar_path(staging_dir) || !scalar_path(output_dir) ||
      !isTRUE(order_clean) || !isTRUE(hashes_clean) ||
      !dir.exists(staging_dir) || !is.function(move_file)) {
    stop("qualification publication input is malformed", call. = FALSE)
  }
  staging_dir <- normalizePath(
    staging_dir, winslash = "/", mustWork = TRUE
  )
  output_parent <- normalizePath(
    dirname(output_dir), winslash = "/", mustWork = TRUE
  )
  output_dir <- file.path(output_parent, basename(output_dir))
  staged_names <- list.files(staging_dir, all.files = FALSE)
  staged_paths <- file.path(staging_dir, publication_order)
  if (!setequal(staged_names, publication_order) ||
      length(staged_names) != length(publication_order) ||
      !all(file.exists(staged_paths)) || any(dir.exists(staged_paths))) {
    stop("qualification staging artifact is incomplete", call. = FALSE)
  }
  staged_hashes <- vapply(
    staged_paths, fastkpc_full_cuda_fixed_sp_sha256_file, character(1L)
  )
  if (!identical(unname(staged_hashes), unname(expected_sha256))) {
    stop("qualification staging artifact hash mismatch", call. = FALSE)
  }

  reserved <- FALSE
  published <- FALSE
  on.exit({
    if (!published && reserved && dir.exists(output_dir)) {
      for (name in rev(publication_order)) {
        owned_path <- file.path(output_dir, name)
        if (file.exists(owned_path) && !dir.exists(owned_path)) {
          current_hash <- tryCatch(
            fastkpc_full_cuda_fixed_sp_sha256_file(owned_path),
            error = function(error) NA_character_
          )
          if (identical(current_hash, expected_sha256[[name]])) {
            unlink(owned_path, force = TRUE)
          }
        }
      }
      if (dir.exists(output_dir) &&
          length(list.files(output_dir, all.files = TRUE,
                            no.. = TRUE)) == 0L) {
        suppressWarnings(file.remove(output_dir))
      }
    }
  }, add = TRUE)
  suspendInterrupts({
    reserved_now <- suppressWarnings(dir.create(
      output_dir, recursive = FALSE, showWarnings = FALSE
    ))
    if (!isTRUE(reserved_now)) {
      stop("exclusive qualification output reservation failed", call. = FALSE)
    }
    reserved <- TRUE
  })

  for (name in publication_order) {
    suspendInterrupts({
      destination <- file.path(output_dir, name)
      if (file.exists(destination) || dir.exists(destination) ||
          !isTRUE(move_file(file.path(staging_dir, name), destination))) {
        stop("qualification payload move failed for ", name, call. = FALSE)
      }
    })
  }
  published_names <- list.files(output_dir, all.files = FALSE)
  if (length(published_names) != length(publication_order) ||
      !setequal(published_names, publication_order) ||
      !file.exists(file.path(output_dir, "summary.json"))) {
    stop("qualification publication surface is incomplete", call. = FALSE)
  }
  published_hashes <- vapply(
    file.path(output_dir, publication_order),
    fastkpc_full_cuda_fixed_sp_sha256_file,
    character(1L)
  )
  if (!identical(unname(published_hashes), unname(expected_sha256))) {
    stop("qualification publication hash changed during move",
         call. = FALSE)
  }
  suspendInterrupts({
    if (length(list.files(staging_dir, all.files = TRUE, no.. = TRUE)) != 0L ||
        unlink(staging_dir, recursive = TRUE, force = TRUE) != 0L ||
        dir.exists(staging_dir)) {
      stop("qualification staging cleanup failed", call. = FALSE)
    }
    published <- TRUE
  })
  normalizePath(output_dir, winslash = "/", mustWork = TRUE)
}

fastkpc_full_cuda_fixed_sp_qualification_result_schema <- function() {
  schema <- function(names, character = character(), integer = character(),
                     double = character(), logical = character(),
                     list = character()) {
    types <- setNames(rep.int(NA_character_, length(names)), names)
    for (type in c("character", "integer", "double", "logical", "list")) {
      fields <- get(type, inherits = FALSE)
      types[fields] <- type
    }
    if (anyNA(types) || anyDuplicated(names) ||
        !setequal(names(types), names)) {
      stop("internal qualification schema definition is malformed",
           call. = FALSE)
    }
    list(names = names, types = types, list_fields = list)
  }
  target_names <- c(
    "prepared_s_key_sha256", "batch_ordinal", "target_ordinal",
    "residual_key_sha256", "target", "null_dim", "phase1_condition",
    "condition_bucket", "phase1_coefficient_rank", "planned_route",
    "authenticated_planned_route", "executed_route", "reroute_reason",
    "solver_status", "target_true_batched", "qr_rank", "geqrf_info",
    "ormqr_info", "effective_rank", "sigma_max", "smallest_retained_sigma",
    "svd_info", "aggregate_penalty_root_rank",
    "aggregate_penalty_root_pivot", "aggregate_factor_call_count",
    "aggregate_b_build_count", "aggregate_dstop",
    "cpu_aggregate_penalty_root_rank", "cpu_aggregate_penalty_root_pivot",
    "cpu_aggregate_effective_rank",
    "cpu_aggregate_effective_rank_threshold", "cpu_aggregate_sigma_max",
    "aggregate_penalty_root_rank_exact",
    "aggregate_penalty_root_pivot_exact", "aggregate_effective_rank_exact",
    "numeric_reference", "outputs_all_finite", "residual_max_abs_diff",
    "residual_relative_l2_diff", "fitted_max_abs_diff",
    "fitted_relative_l2_diff", "fitted_numeric_hash",
    "residual_numeric_hash", "oracle_call_count", "oracle_fitted_hash",
    "oracle_residual_hash", "authenticated_fitted_hash",
    "authenticated_residual_hash", "oracle_fitted_hash_exact",
    "oracle_residual_hash_exact", "approximate_backend"
  )
  target <- schema(
    target_names,
    character = c(
      "prepared_s_key_sha256", "residual_key_sha256", "condition_bucket",
      "planned_route", "authenticated_planned_route", "executed_route",
      "reroute_reason", "solver_status", "numeric_reference",
      "fitted_numeric_hash", "residual_numeric_hash", "oracle_fitted_hash",
      "oracle_residual_hash", "authenticated_fitted_hash",
      "authenticated_residual_hash"
    ),
    integer = c(
      "batch_ordinal", "target_ordinal", "target", "null_dim",
      "phase1_coefficient_rank", "qr_rank", "geqrf_info", "ormqr_info",
      "effective_rank", "svd_info", "aggregate_penalty_root_rank",
      "aggregate_factor_call_count", "aggregate_b_build_count",
      "cpu_aggregate_penalty_root_rank", "cpu_aggregate_effective_rank",
      "oracle_call_count"
    ),
    double = c(
      "phase1_condition", "sigma_max", "smallest_retained_sigma",
      "aggregate_dstop", "cpu_aggregate_effective_rank_threshold",
      "cpu_aggregate_sigma_max", "residual_max_abs_diff",
      "residual_relative_l2_diff", "fitted_max_abs_diff",
      "fitted_relative_l2_diff"
    ),
    logical = c(
      "target_true_batched", "aggregate_penalty_root_rank_exact",
      "aggregate_penalty_root_pivot_exact", "aggregate_effective_rank_exact",
      "outputs_all_finite", "oracle_fitted_hash_exact",
      "oracle_residual_hash_exact", "approximate_backend"
    ),
    list = c(
      "aggregate_penalty_root_pivot", "cpu_aggregate_penalty_root_pivot"
    )
  )
  batch_names <- c(
    "prepared_s_key_sha256", "batch_ordinal", "target_count",
    "batch_call_count", "native_batch_call", "true_batched_kernel",
    "true_batched_subgroup_count", "true_batched_attempted_target_count",
    "true_batched_target_count", "cholesky_single_target_count",
    "potrf_batched_call_count", "potrs_batched_call_count",
    "planned_cholesky_target_count", "planned_qr_target_count",
    "planned_svd_target_count", "executed_cholesky_target_count",
    "executed_qr_target_count", "executed_svd_target_count",
    "stable_reroute_count", "cholesky_to_svd_count", "qr_to_svd_count",
    "aggregate_penalty_factor_count", "aggregate_svd_b_build_count",
    "aggregate_penalty_root_d2h_count", "aggregate_penalty_root_d2h_bytes",
    "target_batch_h2d_call_count", "target_h2d_copy_count",
    "target_h2d_bytes", "rhs_device_build_count", "full_cuda_data_plane",
    "invalid_output_init_count", "coefficient_batch_finalize_call_count",
    "fitted_batch_finalize_call_count",
    "residual_rss_batch_finalize_call_count",
    "per_target_output_finalize_call_count",
    "batch_output_finalized_target_count", "canonical_output_order_exact",
    "target_keys_exact", "route_status_conservation_exact",
    "resource_snapshot_captured", "resource_instrumentation_version",
    "resource_allocation_count_before_solve",
    "resource_allocation_count_after_solve",
    "resource_handle_create_count_before_solve",
    "resource_handle_create_count_after_solve",
    "cuda_device_allocation_count_during_solve",
    "cuda_host_allocation_count_during_solve",
    "stream_create_count_during_solve", "event_create_count_during_solve",
    "cublas_handle_create_count_during_solve",
    "cusolver_handle_create_count_during_solve",
    "per_target_allocation_count_after_warmup",
    "per_target_handle_create_count", "workspace_grow_count_after_warmup",
    "stable_workspace_grow_count_after_warmup",
    "cuda_device_synchronize_count",
    "cholesky_factor_checkpoint_record_count",
    "cholesky_factor_checkpoint_wait_count",
    "cholesky_solve_checkpoint_record_count",
    "cholesky_solve_checkpoint_wait_count", "qr_checkpoint_record_count",
    "qr_checkpoint_wait_count", "svd_checkpoint_record_count",
    "svd_checkpoint_wait_count", "implicit_residual_d2h_count",
    "cpu_fallback_count", "unknown_fallback_count", "solve_elapsed_ms",
    "pre_shadow_materialize_call_count",
    "pre_shadow_materialize_target_count", "pre_shadow_d2h_bytes",
    "shadow_materialize_elapsed_ms", "post_shadow_materialize_call_count",
    "post_shadow_materialize_target_count", "post_shadow_d2h_bytes",
    "output_slot_release_count", "output_slot_leased_after_release"
  )
  batch_logical <- c(
    "native_batch_call", "true_batched_kernel", "full_cuda_data_plane",
    "canonical_output_order_exact", "target_keys_exact",
    "route_status_conservation_exact", "resource_snapshot_captured",
    "output_slot_leased_after_release"
  )
  batch_double <- c(
    "aggregate_penalty_root_d2h_bytes", "target_h2d_bytes",
    "solve_elapsed_ms", "pre_shadow_d2h_bytes",
    "shadow_materialize_elapsed_ms", "post_shadow_d2h_bytes"
  )
  batch <- schema(
    batch_names,
    character = "prepared_s_key_sha256",
    integer = setdiff(
      batch_names,
      c("prepared_s_key_sha256", batch_logical, batch_double)
    ),
    double = batch_double,
    logical = batch_logical
  )
  setup_names <- c(
    "prepared_s_key_sha256", "batch_ordinal", "prepared_handle_create_count",
    "prepared_handle_free_count", "setup_h2d_upload_count", "setup_h2d_bytes",
    "penalty_root_build_count", "penalty_root_rank_mismatch_count",
    "penalty_root_bytes", "penalty_root_build_ms", "penalty_root_matrix_count",
    "penalty_root_row_count", "H_root_matrix_count", "H_root_rank",
    "rank_reference_materialize_call_count",
    "rank_reference_materialize_elapsed_ms", "setup_shadow_d2h_count",
    "setup_shadow_d2h_bytes", "coefficient_output_capacity",
    "prepared_generation", "output_slot_state_before_solve",
    "output_slot_state_after_solve", "output_slot_state_after_release",
    "output_slot_leased_after_release", "output_slot_poison_reason_empty"
  )
  setup_character <- c(
    "prepared_s_key_sha256", "output_slot_state_before_solve",
    "output_slot_state_after_solve", "output_slot_state_after_release"
  )
  setup_logical <- c(
    "output_slot_leased_after_release", "output_slot_poison_reason_empty"
  )
  setup_double <- c(
    "setup_h2d_bytes", "penalty_root_bytes", "penalty_root_build_ms",
    "rank_reference_materialize_elapsed_ms", "setup_shadow_d2h_bytes",
    "coefficient_output_capacity", "prepared_generation"
  )
  setup <- schema(
    setup_names,
    character = setup_character,
    integer = setdiff(
      setup_names, c(setup_character, setup_logical, setup_double)
    ),
    double = setup_double,
    logical = setup_logical
  )
  runtime_names <- c(
    "stage", "device_id", "gpu_name", "runtime_context_create_count",
    "cuda_device_allocation_count", "cuda_host_allocation_count",
    "stream_create_count", "event_create_count", "cublas_handle_create_count",
    "cusolver_handle_create_count", "workspace_grow_count",
    "cuda_device_synchronize_count", "compute_capability_major",
    "compute_capability_minor", "sm_count", "cuda_toolkit_version",
    "cuda_driver_version", "cusolver_deterministic_mode", "cublas_math_mode",
    "cublas_atomics_mode", "cublas_user_workspace_installed",
    "cublas_workspace_alignment", "workspace_reserve_count", "freed",
    "creator_pid", "generation", "gesvdj_info_create_count",
    "gesvdj_info_destroy_count", "stable_workspace_grow_count",
    "cholesky_factor_checkpoint_record_count",
    "cholesky_factor_checkpoint_wait_count",
    "cholesky_solve_checkpoint_record_count",
    "cholesky_solve_checkpoint_wait_count", "qr_checkpoint_record_count",
    "qr_checkpoint_wait_count", "svd_checkpoint_record_count",
    "svd_checkpoint_wait_count", "workspace_bytes", "cublas_workspace_bytes",
    "eigen_workspace_bytes", "qr_workspace_bytes", "svd_workspace_bytes",
    "augmented_workspace_bytes", "aggregate_factor_workspace_bytes"
  )
  runtime_character <- c(
    "stage", "gpu_name", "cusolver_deterministic_mode", "cublas_math_mode",
    "cublas_atomics_mode"
  )
  runtime_logical <- c("cublas_user_workspace_installed", "freed")
  runtime_double <- c(
    "cublas_workspace_alignment", "creator_pid", "generation",
    "workspace_bytes", "cublas_workspace_bytes", "eigen_workspace_bytes",
    "qr_workspace_bytes", "svd_workspace_bytes", "augmented_workspace_bytes",
    "aggregate_factor_workspace_bytes"
  )
  runtime <- schema(
    runtime_names,
    character = runtime_character,
    integer = setdiff(
      runtime_names, c(runtime_character, runtime_logical, runtime_double)
    ),
    double = runtime_double,
    logical = runtime_logical
  )
  list(target = target, batch = batch, setup = setup, runtime = runtime)
}

fastkpc_full_cuda_fixed_sp_qualification_validate_result_schema <- function(
    target_records, batch_records, setup_records, runtime_records) {
  schemas <- fastkpc_full_cuda_fixed_sp_qualification_result_schema()
  frames <- list(
    target = target_records, batch = batch_records,
    setup = setup_records, runtime = runtime_records
  )
  row_counts <- c(target = 6143L, batch = 2061L, setup = 2061L, runtime = 3L)
  valid_frame <- function(frame, schema, row_count) {
    if (!is.data.frame(frame) || nrow(frame) != row_count ||
        !identical(names(frame), schema$names)) return(FALSE)
    all(vapply(schema$names, function(field) {
      value <- frame[[field]]
      if (field %in% schema$list_fields) {
        typeof(value) == "list" && length(value) == row_count &&
          is.object(value) && identical(attributes(value), list(class = "AsIs")) &&
          all(vapply(value, function(element) {
            typeof(element) == "integer" && !is.object(element) &&
              is.null(attributes(element)) && !anyNA(element)
          }, logical(1L)))
      } else {
        typeof(value) == schema$types[[field]] && length(value) == row_count &&
          !is.object(value) && is.null(attributes(value))
      }
    }, logical(1L)))
  }
  clean <- all(vapply(names(frames), function(name) {
    valid_frame(frames[[name]], schemas[[name]], row_counts[[name]])
  }, logical(1L))) &&
    identical(setup_records$prepared_s_key_sha256,
              batch_records$prepared_s_key_sha256) &&
    identical(
      setup_records$prepared_s_key_sha256,
      sort(setup_records$prepared_s_key_sha256, method = "radix")
    ) && identical(
      order(
        target_records$prepared_s_key_sha256,
        target_records$residual_key_sha256,
        method = "radix"
      ),
      seq_len(nrow(target_records))
    ) && identical(
      runtime_records$stage,
      c("runtime-created", "workspace-reserved", "final")
    )
  if (!isTRUE(clean)) {
    stop("qualification result schema mismatch", call. = FALSE)
  }
  TRUE
}

fastkpc_full_cuda_write_fixed_sp_qualification_artifact <- function(
    result, output_dir, phase0_dir, phase1_dir, phase2_dir, data_path,
    device_id, stage_timing, elapsed_seconds, command_lines,
    environment_lines, execution_provenance, dcov_records) {
  if (is.list(result) && is.list(result$summary) &&
      "pass" %in% names(result$summary)) {
    stop("qualification writer rejects caller-supplied pass fields",
         call. = FALSE)
  }
  required_result_names <- c(
    "catalog_records", "runtime_records", "setup_records",
    "batch_records", "target_records", "summary"
  )
  if (!is.list(result) || !identical(names(result), required_result_names) ||
      !is.data.frame(stage_timing) || nrow(stage_timing) < 1L ||
      !identical(names(stage_timing), c("stage", "elapsed_seconds")) ||
      any(!is.finite(stage_timing$elapsed_seconds)) ||
      any(stage_timing$elapsed_seconds < 0)) {
    stop("qualification artifact evidence is malformed", call. = FALSE)
  }
  fastkpc_full_cuda_fixed_sp_qualification_validate_input_summary(
    result$summary
  )
  fastkpc_full_cuda_fixed_sp_qualification_validate_result_schema(
    target_records = result$target_records,
    batch_records = result$batch_records,
    setup_records = result$setup_records,
    runtime_records = result$runtime_records
  )
  if (!is.list(execution_provenance) ||
      !isTRUE(
        execution_provenance$execution_sources_unchanged_after_run
      )) {
    stop("qualification execution provenance is not post-run verified",
         call. = FALSE)
  }
  execution_provenance <-
    fastkpc_full_cuda_fixed_sp_verify_execution_provenance(
      execution_provenance
    )
  scalar_path <- function(value, label, must_work = TRUE) {
    if (!is.character(value) || length(value) != 1L || is.na(value) ||
        !nzchar(value)) {
      stop(label, " must be one nonempty path", call. = FALSE)
    }
    normalizePath(value, winslash = "/", mustWork = must_work)
  }
  output_dir <- scalar_path(output_dir, "output_dir", must_work = FALSE)
  phase0_dir <- scalar_path(phase0_dir, "phase0_dir")
  phase1_dir <- scalar_path(phase1_dir, "phase1_dir")
  phase2_dir <- scalar_path(phase2_dir, "phase2_dir")
  data_path <- scalar_path(data_path, "data_path")
  if (!requireNamespace("jsonlite", quietly = TRUE) ||
      !requireNamespace("digest", quietly = TRUE)) {
    stop("qualification artifact requires jsonlite and digest",
         call. = FALSE)
  }
  authoritative_catalog <-
    fastkpc_full_cuda_fixed_sp_qualification_reopen_catalog_evidence(
      phase0_dir = phase0_dir,
      phase1_dir = phase1_dir,
      phase2_dir = phase2_dir,
      data_path = data_path
    )
  validated_catalog_identity <-
    fastkpc_full_cuda_fixed_sp_qualification_validate_record_identity(
      catalog_records = result$catalog_records,
      summary = result$summary,
      authoritative = authoritative_catalog,
      setup_records = result$setup_records,
      batch_records = result$batch_records,
      target_records = result$target_records
    )
  catalog_authenticated <- validated_catalog_identity$catalog_authenticated
  authenticated_dcov <-
    fastkpc_full_cuda_fixed_sp_load_qualification_logical_tests(
      census_dir = phase1_dir,
      prepared_dir = phase2_dir,
      data_path = data_path
    )
  validated_dcov_summary <-
    fastkpc_full_cuda_fixed_sp_summarize_qualification_dcov_records(
      records = dcov_records,
      logical_tests = authenticated_dcov$logical_tests,
      target_keys = result$target_records$residual_key_sha256
    )
  fastkpc_full_cuda_fixed_sp_qualification_validate_dcov_summary(
    validated_dcov_summary
  )
  if (!identical(
        validated_dcov_summary$qualification_dcov_logical_ids_hash,
        authenticated_dcov$logical_ids_hash
      ) || !identical(
        validated_dcov_summary$qualification_dcov_residual_key_hash,
        authenticated_dcov$endpoint_key_hash
      )) {
    stop("qualification dCov published identity mismatch", call. = FALSE)
  }
  if (typeof(device_id) != "integer" || length(device_id) != 1L ||
      is.na(device_id) || device_id < 0L ||
      !is.double(elapsed_seconds) || length(elapsed_seconds) != 1L ||
      !is.finite(elapsed_seconds) || elapsed_seconds < 0 ||
      !is.character(command_lines) || !is.character(environment_lines) ||
      anyNA(command_lines) || anyNA(environment_lines)) {
    stop("qualification artifact scalar metadata is malformed",
         call. = FALSE)
  }

  parent_dir <- dirname(output_dir)
  dir.create(parent_dir, recursive = TRUE, showWarnings = FALSE)
  staging_dir <- tempfile(
    pattern = paste0(".", basename(output_dir), ".staging-"),
    tmpdir = parent_dir
  )
  if (!dir.create(staging_dir, recursive = FALSE, showWarnings = FALSE)) {
    stop("failed to create qualification staging directory",
         call. = FALSE)
  }
  published <- FALSE
  on.exit({
    if (!published && dir.exists(staging_dir)) {
      unlink(staging_dir, recursive = TRUE, force = TRUE)
    }
  }, add = TRUE)
  path <- function(name) file.path(staging_dir, name)
  write_csv <- function(value, name) {
    utils::write.csv(
      fastkpc_full_cuda_fixed_sp_qualification_csv_frame(value),
      path(name), row.names = FALSE, na = ""
    )
  }

  saveRDS(result$target_records, path("target_parity.rds"), version = 3L)
  write_csv(result$target_records, "target_parity.csv")
  saveRDS(result$batch_records, path("batch_metrics.rds"), version = 3L)
  write_csv(result$batch_records, "batch_metrics.csv")
  saveRDS(result$setup_records, path("setup_metrics.rds"), version = 3L)
  write_csv(result$setup_records, "setup_metrics.csv")
  saveRDS(
    dcov_records, path("qualification_dcov_parity.rds"), version = 3L
  )
  write_csv(dcov_records, "qualification_dcov_parity.csv")
  write_csv(result$runtime_records, "runtime_metrics.csv")
  write_csv(stage_timing, "stage_timing.csv")
  write_csv(data.frame(
    residual_key_sha256 = character(),
    fallback_type = character(),
    reason = character(),
    stringsAsFactors = FALSE
  ), "fallbacks.csv")
  write_csv(data.frame(
    stage = character(), prepared_s_key_sha256 = character(),
    residual_key_sha256 = character(), error_class = character(),
    error_message = character(), stringsAsFactors = FALSE
  ), "failures.csv")
  writeLines(command_lines, path("commands.txt"), useBytes = TRUE)
  writeLines(environment_lines, path("environment.txt"), useBytes = TRUE)
  write_csv(
    execution_provenance$native_build_dependencies$files,
    "native_build_dependencies.csv"
  )
  write_csv(
    execution_provenance$native_build_dependencies$exclusions,
    "native_build_exclusions.csv"
  )
  trace_payload_path <- path("native_build_trace.txt")
  trace_copied <- file.copy(
    execution_provenance$native_build_dependencies$trace_path,
    trace_payload_path,
    overwrite = FALSE, copy.mode = FALSE, copy.date = FALSE
  )
  if (!identical(trace_copied, TRUE) || !identical(
        fastkpc_full_cuda_fixed_sp_sha256_file(trace_payload_path),
        execution_provenance$native_build_dependencies$trace_sha256
      )) {
    stop("qualification native build trace copy is not byte exact",
         call. = FALSE)
  }

  payload_names <- fastkpc_full_cuda_fixed_sp_qualification_payload_names()
  payload_hashes <- vapply(payload_names, function(name) {
    digest::digest(file = path(name), algo = "sha256", serialize = FALSE)
  }, character(1L))
  validated_result_summary <- result$summary
  validated_result_summary$scope_subset_hash <-
    validated_catalog_identity$qualification_subset_hash
  validated_result_summary$qualification_subset_hash <-
    validated_catalog_identity$qualification_subset_hash
  for (field in c(
    "ordered_setup_key_digest", "ordered_target_key_digest",
    "route_status_hash", "numeric_hash", "setup_count", "target_count"
  )) {
    validated_result_summary[[field]] <- validated_catalog_identity[[field]]
  }
  publication_summary <- list(
    artifact_schema_version =
      "full-cuda-ci-fixed-sp-qualification-v6",
    catalog_authenticated = catalog_authenticated,
    provenance_mode = execution_provenance$provenance_mode,
    head_base_commit = execution_provenance$head_base_commit,
    source_closure_schema_version =
      execution_provenance$source_closure_schema_version,
    source_closure_count = execution_provenance$source_closure_count,
    source_closure_sha256 = execution_provenance$source_closure_sha256,
    execution_snapshot_sha256 =
      execution_provenance$execution_snapshot_sha256,
    relevant_sources_dirty_or_untracked =
      execution_provenance$relevant_sources_dirty_or_untracked,
    native_build_inputs_sha256 =
      execution_provenance$native_build_inputs_sha256,
    native_build_dependency_count =
      execution_provenance$native_build_dependencies$dependency_count,
    native_build_dependencies_sha256 =
      execution_provenance$native_build_dependencies$aggregate_sha256,
    native_build_trace_sha256 =
      execution_provenance$native_build_dependencies$trace_sha256,
    native_build_tracer_sha256 =
      execution_provenance$native_build_dependencies$tracer_sha256,
    native_library_sha256 = execution_provenance$native_library_sha256,
    execution_sources_unchanged_after_run =
      execution_provenance$execution_sources_unchanged_after_run,
    elapsed_seconds = as.double(elapsed_seconds),
    stage_timing_total_seconds = as.double(sum(stage_timing$elapsed_seconds)),
    payload_file_count = as.integer(length(payload_names))
  )
  summary_schema <- fastkpc_full_cuda_fixed_sp_qualification_summary_schema()
  if (!identical(
    names(validated_result_summary), summary_schema$input_names
  ) || !identical(
    names(validated_dcov_summary), summary_schema$dcov_names
  ) || !identical(
    names(publication_summary), summary_schema$publication_names
  ) || anyDuplicated(c(
    names(validated_result_summary), names(validated_dcov_summary),
    names(publication_summary)
  ))) {
    stop("qualification summary composition namespace is invalid",
         call. = FALSE)
  }
  summary <- c(
    validated_result_summary, validated_dcov_summary,
    publication_summary
  )
  fastkpc_full_cuda_fixed_sp_qualification_validate_published_summary(summary)
  manifest <- list(
    schema_version = "full-cuda-ci-fixed-sp-qualification-v6",
    scope = "qualification",
    catalog_authenticated = catalog_authenticated,
    provenance_schema_version =
      execution_provenance$provenance_schema_version,
    provenance_mode = execution_provenance$provenance_mode,
    head_base_commit = execution_provenance$head_base_commit,
    source_closure_schema_version =
      execution_provenance$source_closure_schema_version,
    source_discovery_semantics =
      execution_provenance$source_discovery_semantics,
    source_project_root = execution_provenance$source_project_root,
    source_closure_count = execution_provenance$source_closure_count,
    source_closure_sha256 = execution_provenance$source_closure_sha256,
    direct_source_ids = as.list(execution_provenance$direct_source_ids),
    source_dependency_map = lapply(
      execution_provenance$source_dependency_map, as.list
    ),
    source_file_paths = as.list(execution_provenance$source_file_paths),
    source_file_sha256 = as.list(execution_provenance$source_file_sha256),
    source_file_git_state =
      as.list(execution_provenance$source_file_git_state),
    native_build_input_paths =
      as.list(execution_provenance$native_build_input_paths),
    native_build_input_sha256 =
      as.list(execution_provenance$native_build_input_sha256),
    native_build_input_git_state =
      as.list(execution_provenance$native_build_input_git_state),
    native_build_inputs_sha256 =
      execution_provenance$native_build_inputs_sha256,
    native_build_dependencies_schema_version =
      execution_provenance$native_build_dependencies$schema_version,
    native_build_dependency_trace_semantics =
      execution_provenance$native_build_dependencies$trace_semantics,
    native_build_dependency_trace_invocation =
      execution_provenance$native_build_dependencies$trace_invocation,
    native_build_working_dir =
      execution_provenance$native_build_dependencies$build_working_dir,
    native_build_trace_path = "native_build_trace.txt",
    native_build_trace_sha256 =
      execution_provenance$native_build_dependencies$trace_sha256,
    native_build_tracer_path =
      execution_provenance$native_build_dependencies$tracer_path,
    native_build_tracer_sha256 =
      execution_provenance$native_build_dependencies$tracer_sha256,
    native_build_dependency_count =
      execution_provenance$native_build_dependencies$dependency_count,
    native_build_exclusion_count =
      execution_provenance$native_build_dependencies$exclusion_count,
    native_build_dependencies_sha256 =
      execution_provenance$native_build_dependencies$aggregate_sha256,
    relevant_sources_dirty_or_untracked =
      execution_provenance$relevant_sources_dirty_or_untracked,
    native_library_identity =
      execution_provenance$native_library_identity,
    native_library_path = execution_provenance$native_library_path,
    native_library_sha256 = execution_provenance$native_library_sha256,
    execution_snapshot_sha256 =
      execution_provenance$execution_snapshot_sha256,
    execution_sources_unchanged_after_run =
      execution_provenance$execution_sources_unchanged_after_run,
    device_id = device_id,
    phase0_dir = phase0_dir,
    phase1_dir = phase1_dir,
    phase2_dir = phase2_dir,
    data_path = data_path,
    qualification_subset_hash =
      validated_catalog_identity$qualification_subset_hash,
    ordered_setup_key_digest =
      validated_catalog_identity$ordered_setup_key_digest,
    ordered_target_key_digest =
      validated_catalog_identity$ordered_target_key_digest,
    route_status_hash = validated_catalog_identity$route_status_hash,
    numeric_hash = validated_catalog_identity$numeric_hash,
    qualification_logical_tests_sha256 =
      authenticated_dcov$logical_tests_sha256,
    qualification_dcov_logical_ids_hash =
      validated_dcov_summary$qualification_dcov_logical_ids_hash,
    qualification_dcov_residual_key_hash =
      validated_dcov_summary$qualification_dcov_residual_key_hash,
    qualification_dcov_rows_hash =
      validated_dcov_summary$qualification_dcov_rows_hash,
    payload_file_sha256 = as.list(payload_hashes),
    publication_order = as.list(c(
      payload_names, "manifest.json", "summary.json"
    ))
  )
  fastkpc_full_cuda_fixed_sp_write_qualification_json(
    manifest, path("manifest.json")
  )
  fastkpc_full_cuda_fixed_sp_qualification_validate_published_summary(summary)
  fastkpc_full_cuda_fixed_sp_write_qualification_json(
    summary, path("summary.json")
  )

  publication_order <- c(payload_names, "manifest.json", "summary.json")
  publication_hashes <- c(
    payload_hashes,
    manifest.json = fastkpc_full_cuda_fixed_sp_sha256_file(
      path("manifest.json")
    ),
    summary.json = fastkpc_full_cuda_fixed_sp_sha256_file(
      path("summary.json")
    )
  )
  fastkpc_full_cuda_fixed_sp_publish_qualification_staging(
    staging_dir = staging_dir,
    output_dir = output_dir,
    publication_order = publication_order,
    expected_sha256 = publication_hashes
  )
  published <- TRUE
  list(
    output_dir = output_dir,
    manifest = manifest,
    summary = summary,
    payload_file_sha256 = payload_hashes
  )
}
