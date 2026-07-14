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

fastkpc_full_cuda_open_fixed_sp_catalog <- function(
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
  list(
    phase0 = phase0,
    inputs = inputs,
    phase2_dir = phase2_dir,
    phase2_file_hashes = captured$hashes,
    phase2_summary = summary,
    phase2_manifest = manifest,
    catalog_contract = contract,
    setup_index = setup_index,
    scopes = phase2_objects$scopes,
    semantic_files = names(contract$phase2_file_sha256)
  )
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
    stop("full scope streaming is introduced in the closure plan", call. = FALSE)
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

fastkpc_full_cuda_fixed_sp_phase3a_prototype_payload_identity <- function(
    prototype) {
  fields <- c(
    "target_key", "X", "y", "Z", "XtX_null", "penalty_null", "Xty_null"
  )
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
  list(
    target_key = prototype$target_key,
    payload_hash = fastkpc_full_cuda_census_named_metadata_hash(prototype)
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
  actual_keys <- vapply(actual, `[[`, character(1L), "target_key")
  actual_hashes <- vapply(actual, `[[`, character(1L), "payload_hash")
  if (!identical(actual, expected) ||
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
  target_row_index <- 0L
  safe_descriptor_inputs <- list()
  for (setup_index in seq_along(setup_keys)) {
    setup_key <- setup_keys[[setup_index]]
    batch <- batches[[setup_key]]
    dto <- dtos[[setup_key]]
    native_batch <- fastkpc_full_cuda_fixed_sp_native_batch(batch, dto)
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
      if (identical(target$planned_route, "CHOLESKY_BATCHED")) {
        safe_descriptor_inputs[[length(safe_descriptor_inputs) + 1L]] <-
          list(handle = handle, native = target, dto = dto)
      }
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

  safe_descriptors <- list()
  for (descriptor in safe_descriptor_inputs) {
    prototype <- fastkpc_full_cuda_fixed_sp_phase3a_prototype_handle(
      descriptor$dto, descriptor$native$Y[, 1L], descriptor$native$SP[, 1L]
    )
    prototype <- c(list(target_key = descriptor$native$target_keys), prototype)
    safe_descriptors[[length(safe_descriptors) + 1L]] <- list(
      handle = descriptor$handle,
      native = descriptor$native,
      prototype = prototype,
      prototype_expected =
        fastkpc_full_cuda_fixed_sp_phase3a_prototype_payload_identity(prototype)
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
