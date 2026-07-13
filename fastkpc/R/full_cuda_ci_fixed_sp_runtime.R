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
