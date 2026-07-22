fastkpc_full_cuda_phase3_oracle_schema_version <- function() {
  "full-cuda-ci-fixed-sp-oracle-sp-artifact-v1"
}

fastkpc_full_cuda_phase3_shadow_schema_version <- function() {
  "full-cuda-ci-fixed-sp-shadow-artifact-v1"
}

fastkpc_full_cuda_phase3_artifact_paths <- function(
    output_dir, kind = "oracle_sp") {
  scalar_character <- function(value, name) {
    if (typeof(value) != "character" || length(value) != 1L ||
        is.object(value) || !is.null(attributes(value)) || is.na(value) ||
        !nzchar(value) || grepl("[[:cntrl:]]", value, perl = TRUE)) {
      stop(name, " must be one nonempty bare character path", call. = FALSE)
    }
    value
  }

  output_dir <- scalar_character(output_dir, "output_dir")
  kind <- scalar_character(kind, "kind")
  if (!kind %in% c("oracle_sp", "full_shadow")) {
    stop("unknown Phase 3 artifact kind", call. = FALSE)
  }

  common_names <- c(
    "manifest_json", "summary_json", "commands_txt", "environment_txt",
    "input_hashes_csv", "route_config_json", "runtime_lifecycle_csv",
    "resource_metrics_csv", "stage_timing_csv", "fallbacks_csv",
    "failures_csv"
  )
  payload_names <- switch(
    kind,
    oracle_sp = c(
      "setup_results_csv", "setup_results_rds", "target_parity_csv",
      "target_parity_rds", "risk_cases_csv", "risk_cases_rds",
      "qualification_dcov_csv", "qualification_dcov_rds"
    ),
    full_shadow = c(
      "logical_ci_parity_csv", "logical_ci_parity_rds",
      "deletion_trace_csv", "sepset_agreement_csv", "n_edgetests_csv",
      "adjacency_rds", "first_divergence_json", "direct_ci_rds",
      "direct_ci_summary_json"
    )
  )
  file_names <- c(
    manifest_json = "manifest.json",
    summary_json = "summary.json",
    commands_txt = "commands.txt",
    environment_txt = "environment.txt",
    input_hashes_csv = "input_hashes.csv",
    route_config_json = "route_config.json",
    runtime_lifecycle_csv = "runtime_lifecycle.csv",
    resource_metrics_csv = "resource_metrics.csv",
    stage_timing_csv = "stage_timing.csv",
    fallbacks_csv = "fallbacks.csv",
    failures_csv = "failures.csv"
  )
  payload_file_names <- c(
    setup_results_csv = "setup_results.csv",
    setup_results_rds = "setup_results.rds",
    target_parity_csv = "target_parity.csv",
    target_parity_rds = "target_parity.rds",
    risk_cases_csv = "risk_cases.csv",
    risk_cases_rds = "risk_cases.rds",
    qualification_dcov_csv = "qualification_dcov_parity.csv",
    qualification_dcov_rds = "qualification_dcov_parity.rds",
    logical_ci_parity_csv = "logical_ci_parity.csv",
    logical_ci_parity_rds = "logical_ci_parity.rds",
    deletion_trace_csv = "deletion_trace.csv",
    sepset_agreement_csv = "sepset_agreement.csv",
    n_edgetests_csv = "n_edgetests.csv",
    adjacency_rds = "adjacency.rds",
    first_divergence_json = "first_divergence.json",
    direct_ci_rds = "direct_ci.rds",
    direct_ci_summary_json = "direct_ci.summary.json"
  )
  selected_names <- c(common_names, payload_names, "shards_dir", "sessions_dir")
  selected_file_names <- c(
    file_names[common_names],
    payload_file_names[payload_names],
    shards_dir = "shards",
    sessions_dir = "sessions"
  )
  if (anyDuplicated(selected_names) || anyDuplicated(selected_file_names)) {
    stop("Phase 3 artifact path names are duplicated", call. = FALSE)
  }
  setNames(
    as.list(file.path(output_dir, unname(selected_file_names))),
    selected_names
  )
}

.fastkpc_full_cuda_phase3_bare_scalar <- function(value, type) {
  typeof(value) == type && length(value) == 1L && !is.object(value) &&
    is.null(attributes(value)) && !is.na(value)
}

.fastkpc_full_cuda_phase3_bare_integer <- function(value, minimum = NULL) {
  clean <- .fastkpc_full_cuda_phase3_bare_scalar(value, "integer")
  if (!isTRUE(clean)) return(FALSE)
  if (is.null(minimum)) return(TRUE)
  value >= minimum
}

.fastkpc_full_cuda_phase3_bare_number <- function(value, minimum = NULL) {
  clean <- typeof(value) %in% c("integer", "double") && length(value) == 1L &&
    !is.object(value) && is.null(attributes(value)) && !is.na(value) &&
    is.finite(value)
  if (!isTRUE(clean)) return(FALSE)
  is.null(minimum) || value >= minimum
}

.fastkpc_full_cuda_phase3_sha256 <- function(value) {
  .fastkpc_full_cuda_phase3_bare_scalar(value, "character") &&
    grepl("^[0-9a-f]{64}$", value)
}

.fastkpc_full_cuda_phase3_commit <- function(value) {
  .fastkpc_full_cuda_phase3_bare_scalar(value, "character") &&
    grepl("^[0-9a-f]{40}$", value)
}

.fastkpc_full_cuda_phase3_named_hash <- function(value) {
  if (!exists("fastkpc_full_cuda_census_named_metadata_hash",
             mode = "function")) {
    stop("Phase 3 canonical hash helper is unavailable", call. = FALSE)
  }
  fastkpc_full_cuda_census_named_metadata_hash(value)
}

fastkpc_full_cuda_phase3_route_config <- function() {
  route <- list(
    schema_version = "full-cuda-ci-fixed-sp-route-v1",
    condition_lt_1e8 = "CHOLESKY_BATCHED",
    condition_1e8_to_lt_1e12 = "AUGMENTED_QR",
    condition_ge_1e12 = "AUGMENTED_SVD",
    rank_deficient = "AUGMENTED_SVD",
    unauthenticated = "AUGMENTED_SVD",
    svd_rank_tolerance =
      "sigma_max*sqrt(double_epsilon)",
    residual_tolerance = 1e-7,
    fitted_tolerance = 1e-7,
    qualification_dcov_p_tolerance = 1e-10,
    dcov_backend = "legacy-cpp-spectra",
    reroute_policy = "declared-cuda-svd-reroute-with-conservation",
    cpu_fallback_allowed = FALSE,
    approximate_backend_allowed = FALSE,
    shard_count = 64L
  )
  route$sha256 <- .fastkpc_full_cuda_phase3_named_hash(route)
  route
}

fastkpc_full_cuda_phase3_route_config_hash <- function(
    route = NULL, route_config = NULL) {
  if (is.null(route)) {
    route <- if (is.null(route_config)) {
      fastkpc_full_cuda_phase3_route_config()
    } else {
      route_config
    }
  }
  if (!is.list(route) || is.object(route) || is.null(names(route)) ||
      !"sha256" %in% names(route)) {
    stop("Phase 3 route configuration is malformed", call. = FALSE)
  }
  expected <- .fastkpc_full_cuda_phase3_named_hash(
    route[setdiff(names(route), "sha256")]
  )
  if (!identical(route$sha256, expected)) {
    stop("Phase 3 route configuration hash mismatch", call. = FALSE)
  }
  expected
}

.fastkpc_full_cuda_phase3_identity_fields <- function() {
  c(
    "schema_version", "phase0_manifest_hash", "phase1_manifest_hash",
    "phase2_manifest_hash", "dataset_file_sha256", "dataset_matrix_sha256",
    "canonical_setup_corpus_hash", "canonical_target_corpus_hash",
    "phase0_source_commit", "phase1_source_commit", "phase2_source_commit",
    "phase2_R_version", "phase2_mgcv_version",
    "runtime_abi", "runtime_abi_hash",
    "runtime_policy_schema_version", "route_config_hash",
    "source_commit", "phase3_source_commit", "R_version",
    "phase3_R_version", "mgcv_version", "phase3_mgcv_version",
    "provenance_schema_version", "provenance_mode",
    "source_closure_schema_version", "source_discovery_semantics",
    "source_closure_count", "source_closure_sha256",
    "execution_snapshot_sha256", "relevant_sources_dirty_or_untracked",
    "execution_sources_unchanged_after_run", "execution_provenance_state",
    "native_library_identity", "native_library_path",
    "native_library_device_major_hex", "native_library_device_minor_hex",
    "native_library_inode", "native_library_sha256",
    "native_build_inputs_sha256",
    "native_build_dependencies_schema_version",
    "native_build_attestation_schema_version",
    "native_build_attestation_sha256",
    "native_build_trace_semantics", "native_build_trace_invocation",
    "native_build_tracer_path", "native_build_dependency_count",
    "native_build_exclusion_count", "native_build_dependencies_sha256",
    "native_build_trace_sha256", "native_build_tracer_sha256",
    "cuda_toolkit_version", "cuda_driver_version", "gpu_name", "gpu_uuid",
    "compute_capability_major", "compute_capability_minor",
    "compute_capability", "sm_count", "device_id",
    "cusolver_deterministic_mode_required", "cublas_math_mode_required",
    "cublas_atomics_mode_required", "cublas_user_workspace_required",
    "cublas_workspace_bytes_required",
    "cublas_workspace_min_alignment_required", "shard_count",
    "authenticated"
  )
}

.fastkpc_full_cuda_phase3_session_identity_fields <- function() {
  c(
    "execution_snapshot_sha256",
    "native_library_path",
    "native_library_device_major_hex",
    "native_library_device_minor_hex",
    "native_library_inode"
  )
}

.fastkpc_full_cuda_phase3_stable_identity_fields <- function() {
  setdiff(
    .fastkpc_full_cuda_phase3_identity_fields(),
    c(
      .fastkpc_full_cuda_phase3_session_identity_fields(),
      "native_library_sha256",
      "native_build_trace_invocation", "native_build_dependencies_sha256",
      "native_build_trace_sha256"
    )
  )
}

.fastkpc_full_cuda_phase3_identity_hash <- function(identity) {
  fields <- .fastkpc_full_cuda_phase3_stable_identity_fields()
  .fastkpc_full_cuda_phase3_named_hash(identity[fields])
}

.fastkpc_full_cuda_phase3_scalar_text <- function(value, name) {
  if (!.fastkpc_full_cuda_phase3_bare_scalar(value, "character") ||
      !nzchar(value) || grepl("[[:cntrl:]]", value, perl = TRUE)) {
    stop(name, " is malformed", call. = FALSE)
  }
  value
}

.fastkpc_full_cuda_phase3_lineage_alias <- function(value, aliases, name) {
  present <- aliases[aliases %in% names(value)]
  if (length(present) == 0L) return(NULL)
  selected <- value[[present[[1L]]]]
  if (length(present) > 1L && any(vapply(
        present[-1L], function(field) !identical(value[[field]], selected),
        logical(1L)
      ))) {
    stop("Phase 3 catalog lineage aliases disagree: ", name,
         call. = FALSE)
  }
  selected
}

.fastkpc_full_cuda_phase3_validate_lineage <- function(value) {
  if (!is.list(value) || is.object(value) || is.null(names(value)) ||
      anyDuplicated(names(value))) {
    stop("Phase 3 catalog lineage evidence is malformed", call. = FALSE)
  }
  aliases <- list(
    phase0_manifest_hash = c(
      "phase0_manifest_hash", "phase0_manifest_sha256",
      "phase0_manifest_file_sha256"
    ),
    phase1_manifest_hash = c(
      "phase1_manifest_hash", "phase1_manifest_sha256",
      "phase1_manifest_file_sha256"
    ),
    phase2_manifest_hash = c(
      "phase2_manifest_hash", "phase2_manifest_sha256",
      "phase2_manifest_file_sha256"
    ),
    dataset_file_sha256 = c("dataset_file_sha256", "dataset_file_hash"),
    dataset_matrix_sha256 = c(
      "dataset_matrix_sha256", "dataset_sha256", "dataset_matrix_hash"
    ),
    canonical_setup_corpus_hash = c(
      "canonical_setup_corpus_hash", "canonical_setup_key_corpus_hash",
      "canonical_prepared_s_key_corpus_hash",
      "full_canonical_prepared_s_key_corpus_hash"
    ),
    canonical_target_corpus_hash = c(
      "canonical_target_corpus_hash", "canonical_target_key_corpus_hash",
      "full_canonical_target_key_corpus_hash"
    ),
    phase0_source_commit = c(
      "phase0_source_commit", "phase0_source_sha1"
    ),
    phase1_source_commit = c(
      "phase1_source_commit", "phase1_source_sha1"
    ),
    phase2_source_commit = c(
      "phase2_source_commit", "phase2_source_sha1", "source_commit"
    ),
    phase2_R_version = c("phase2_R_version", "phase2_r_version"),
    phase2_mgcv_version = c(
      "phase2_mgcv_version", "phase2_MGCV_version", "mgcv_version"
    )
  )
  for (field in names(aliases)) {
    selected <- .fastkpc_full_cuda_phase3_lineage_alias(
      value, aliases[[field]], field
    )
    if (!is.null(selected)) value[[field]] <- selected
  }
  required <- c(
    "authenticated", "phase0_manifest_hash", "phase1_manifest_hash",
    "phase2_manifest_hash", "dataset_file_sha256", "dataset_matrix_sha256",
    "canonical_setup_corpus_hash", "canonical_target_corpus_hash",
    "phase0_source_commit", "phase1_source_commit", "phase2_source_commit",
    "phase2_R_version", "phase2_mgcv_version"
  )
  missing <- setdiff(required, names(value))
  if (length(missing) > 0L) {
    stop("Phase 3 catalog lineage is missing: ",
         paste(missing, collapse = ","), call. = FALSE)
  }
  if (!.fastkpc_full_cuda_phase3_bare_scalar(value$authenticated, "logical") ||
      !isTRUE(value$authenticated)) {
    stop("Phase 3 catalog lineage is not authenticated", call. = FALSE)
  }
  for (field in setdiff(required, c(
    "authenticated", "phase0_source_commit", "phase1_source_commit",
    "phase2_source_commit", "phase2_R_version", "phase2_mgcv_version"
  ))) {
    if (!.fastkpc_full_cuda_phase3_sha256(value[[field]])) {
      stop("Phase 3 catalog lineage hash is malformed: ", field,
           call. = FALSE)
    }
  }
  for (field in c(
    "phase0_source_commit", "phase1_source_commit", "phase2_source_commit"
  )) {
    if (!.fastkpc_full_cuda_phase3_commit(value[[field]])) {
      stop("Phase 3 catalog lineage source commit is malformed: ", field,
           call. = FALSE)
    }
  }
  .fastkpc_full_cuda_phase3_scalar_text(
    value$phase2_R_version, "phase2_R_version"
  )
  .fastkpc_full_cuda_phase3_scalar_text(
    value$phase2_mgcv_version, "phase2_mgcv_version"
  )
  list(
    authenticated = TRUE,
    phase0_manifest_hash = value$phase0_manifest_hash,
    phase1_manifest_hash = value$phase1_manifest_hash,
    phase2_manifest_hash = value$phase2_manifest_hash,
    dataset_file_sha256 = value$dataset_file_sha256,
    dataset_matrix_sha256 = value$dataset_matrix_sha256,
    canonical_setup_corpus_hash = value$canonical_setup_corpus_hash,
    canonical_target_corpus_hash = value$canonical_target_corpus_hash,
    phase0_source_commit = value$phase0_source_commit,
    phase1_source_commit = value$phase1_source_commit,
    phase2_source_commit = value$phase2_source_commit,
    phase2_R_version = value$phase2_R_version,
    phase2_mgcv_version = value$phase2_mgcv_version
  )
}

.fastkpc_full_cuda_phase3_catalog_value <- function(catalog, candidates) {
  if (!is.list(catalog)) return(NULL)
  for (name in candidates) {
    if (!is.null(catalog[[name]])) return(catalog[[name]])
  }
  NULL
}

.fastkpc_full_cuda_phase3_catalog_hash <- function(
    catalog, candidates, fallback, label) {
  present <- candidates[vapply(
    candidates, function(name) !is.null(catalog[[name]]), logical(1L)
  )]
  if (length(present) == 0L) return(unname(fallback))
  values <- vapply(present, function(name) {
    value <- catalog[[name]]
    if (!.fastkpc_full_cuda_phase3_sha256(value)) {
      stop(label, " is malformed", call. = FALSE)
    }
    unname(value)
  }, character(1L))
  if (length(values) > 1L && any(values[-1L] != values[[1L]])) {
    stop(label, " aliases disagree", call. = FALSE)
  }
  values[[1L]]
}

.fastkpc_full_cuda_phase3_validate_execution_evidence <- function(value) {
  if (!is.list(value) || is.object(value) || is.null(names(value)) ||
      anyDuplicated(names(value))) {
    stop("Phase 3 execution provenance is malformed", call. = FALSE)
  }
  required <- c(
    "authenticated", "source_commit", "phase3_source_commit", "R_version",
    "phase3_R_version", "mgcv_version", "phase3_mgcv_version",
    "provenance_schema_version", "provenance_mode",
    "source_closure_schema_version", "source_discovery_semantics",
    "source_closure_count", "source_closure_sha256",
    "execution_snapshot_sha256", "relevant_sources_dirty_or_untracked",
    "execution_sources_unchanged_after_run", "execution_provenance_state",
    "native_library_identity", "native_library_path",
    "native_library_device_major_hex", "native_library_device_minor_hex",
    "native_library_inode", "native_library_sha256",
    "native_build_inputs_sha256",
    "native_build_dependencies_schema_version",
    "native_build_attestation_schema_version",
    "native_build_attestation_sha256",
    "native_build_trace_semantics", "native_build_trace_invocation",
    "native_build_tracer_path", "native_build_dependency_count",
    "native_build_exclusion_count", "native_build_dependencies_sha256",
    "native_build_trace_sha256", "native_build_tracer_sha256"
  )
  missing <- setdiff(required, names(value))
  if (length(missing) > 0L) {
    stop("Phase 3 execution provenance is missing: ",
         paste(missing, collapse = ","), call. = FALSE)
  }
  if (!.fastkpc_full_cuda_phase3_bare_scalar(value$authenticated, "logical") ||
      !isTRUE(value$authenticated)) {
    stop("Phase 3 execution provenance is not authenticated", call. = FALSE)
  }
  for (field in c("source_commit", "phase3_source_commit")) {
    if (!.fastkpc_full_cuda_phase3_commit(value[[field]])) {
      stop("Phase 3 execution source commit is malformed: ", field,
           call. = FALSE)
    }
  }
  if (!identical(value$source_commit, value$phase3_source_commit)) {
    stop("Phase 3 execution source commit aliases disagree", call. = FALSE)
  }
  if (!identical(
        value$source_commit,
        .fastkpc_full_cuda_phase3_current_source_commit()
      )) {
    stop("Phase 3 execution source commit is not the authenticated HEAD",
         call. = FALSE)
  }
  for (field in c(
    "R_version", "phase3_R_version", "mgcv_version", "phase3_mgcv_version",
    "provenance_schema_version", "provenance_mode",
    "source_closure_schema_version", "source_discovery_semantics",
    "native_library_identity", "native_library_path",
    "native_library_device_major_hex", "native_library_device_minor_hex",
    "native_library_inode", "native_build_dependencies_schema_version",
    "native_build_attestation_schema_version",
    "native_build_trace_semantics", "native_build_trace_invocation",
    "native_build_tracer_path"
  )) {
    .fastkpc_full_cuda_phase3_scalar_text(value[[field]], field)
  }
  if (!identical(value$R_version, value$phase3_R_version) ||
      !identical(value$mgcv_version, value$phase3_mgcv_version)) {
    stop("Phase 3 execution version aliases disagree", call. = FALSE)
  }
  if (!identical(
        value$provenance_schema_version,
        "full-cuda-ci-execution-source-snapshot-v6"
      ) || !identical(
        value$provenance_mode, "working-tree-execution-snapshot-v1"
      ) || !identical(
        value$source_closure_schema_version,
        "full-cuda-ci-execution-source-closure-v1"
      ) || !identical(
        value$source_discovery_semantics,
        "parsed-r-ast-load-time-literal-source-v1"
      ) || !identical(
        value$native_library_identity,
        "qualified-pinned-inode-sha-exact-registered-mapped-path-v3"
      ) || !startsWith(value$native_library_path, "/") ||
      grepl("[\r\n]", value$native_library_path) ||
      !grepl("^[0-9a-f]+$", value$native_library_device_major_hex) ||
      !grepl("^[0-9a-f]+$", value$native_library_device_minor_hex) ||
      !grepl("^[0-9]+$", value$native_library_inode) ||
      !identical(
        value$native_build_dependencies_schema_version,
        "full-cuda-ci-native-build-dependencies-v3"
      ) || !identical(
        value$native_build_attestation_schema_version,
        "full-cuda-ci-native-build-trace-attestation-v1"
      ) || !identical(
        value$native_build_trace_semantics,
        "linux-strace-successful-read-exec-evidence-v3"
      )) {
    stop("Phase 3 execution provenance schema is unsupported", call. = FALSE)
  }
  if (!.fastkpc_full_cuda_phase3_bare_integer(value$source_closure_count, 1L) ||
      !.fastkpc_full_cuda_phase3_sha256(value$source_closure_sha256) ||
      !.fastkpc_full_cuda_phase3_sha256(value$execution_snapshot_sha256) ||
      !.fastkpc_full_cuda_phase3_sha256(value$native_library_sha256) ||
      !.fastkpc_full_cuda_phase3_sha256(value$native_build_inputs_sha256) ||
      !.fastkpc_full_cuda_phase3_sha256(
        value$native_build_attestation_sha256
      ) ||
      !.fastkpc_full_cuda_phase3_sha256(
        value$native_build_dependencies_sha256
      ) || !.fastkpc_full_cuda_phase3_sha256(
        value$native_build_trace_sha256
      ) || !.fastkpc_full_cuda_phase3_sha256(
        value$native_build_tracer_sha256
      ) || !.fastkpc_full_cuda_phase3_bare_integer(
        value$native_build_dependency_count, 1L
      ) || !.fastkpc_full_cuda_phase3_bare_integer(
        value$native_build_exclusion_count, 0L
      )) {
    stop("Phase 3 execution provenance hash/count is malformed", call. = FALSE)
  }
  for (field in c(
    "relevant_sources_dirty_or_untracked",
    "execution_sources_unchanged_after_run"
  )) {
    if (!.fastkpc_full_cuda_phase3_bare_scalar(value[[field]], "logical")) {
      stop("Phase 3 execution provenance flag is malformed: ", field,
           call. = FALSE)
    }
  }
  .fastkpc_full_cuda_phase3_scalar_text(
    value$execution_provenance_state, "execution_provenance_state"
  )
  state_consistent <- value$execution_provenance_state %in% c(
    "pre-run-capture", "post-run-verified"
  ) && (
    identical(value$execution_provenance_state, "pre-run-capture") &&
      identical(value$execution_sources_unchanged_after_run, FALSE) ||
    identical(value$execution_provenance_state, "post-run-verified") &&
      identical(value$execution_sources_unchanged_after_run, TRUE)
  )
  if (!isTRUE(state_consistent)) {
    stop("Phase 3 execution provenance state/verification flag disagrees",
         call. = FALSE)
  }
  authoritative <- .fastkpc_full_cuda_phase3_execution_projection(
    fastkpc_full_cuda_phase3_discover_qualified_native_evidence()$provenance
  )
  bound_fields <- c(
    "source_commit", "phase3_source_commit", "R_version",
    "phase3_R_version", "mgcv_version", "phase3_mgcv_version",
    "provenance_schema_version", "provenance_mode",
    "source_closure_schema_version", "source_discovery_semantics",
    "source_closure_count", "source_closure_sha256",
    "execution_snapshot_sha256", "relevant_sources_dirty_or_untracked",
    "native_library_identity", "native_library_path",
    "native_library_device_major_hex", "native_library_device_minor_hex",
    "native_library_inode", "native_library_sha256",
    "native_build_inputs_sha256",
    "native_build_dependencies_schema_version",
    "native_build_attestation_schema_version",
    "native_build_attestation_sha256",
    "native_build_trace_semantics", "native_build_trace_invocation",
    "native_build_tracer_path", "native_build_dependency_count",
    "native_build_exclusion_count", "native_build_dependencies_sha256",
    "native_build_trace_sha256", "native_build_tracer_sha256"
  )
  mismatched <- bound_fields[!vapply(
    bound_fields,
    function(field) .fastkpc_full_cuda_phase3_identity_value_equal(
      value[[field]], authoritative[[field]]
    ),
    logical(1L)
  )]
  if (length(mismatched) > 0L) {
    stop(
      "Phase 3 execution provenance does not match authenticated qualified build: ",
      mismatched[[1L]],
      call. = FALSE
    )
  }
  value
}

.fastkpc_full_cuda_phase3_execution_roots <- function() {
  candidates <- c(
    phase3_artifacts = "fastkpc/R/full_cuda_ci_phase3_artifacts.R",
    fixed_sp_runtime = "fastkpc/R/full_cuda_ci_fixed_sp_runtime.R",
    cuda_gate = "fastkpc/R/full_cuda_ci_gate.R",
    oracle_contract = "fastkpc/R/full_cuda_ci_oracle_contract.R",
    prepared_s_contract = "fastkpc/R/full_cuda_ci_prepared_s_contract.R",
    workload_census = "fastkpc/R/full_cuda_ci_workload_census.R",
    cuda_native = "fastkpc/R/cuda_native.R"
  )
  candidates[file.exists(candidates)]
}

.fastkpc_full_cuda_phase3_current_source_commit <- function() {
  if (!exists("fastkpc_full_cuda_source_commit", mode = "function")) {
    stop("authenticated current Phase 3 source helper is unavailable",
         call. = FALSE)
  }
  commit <- tryCatch(
    fastkpc_full_cuda_source_commit(),
    error = function(error) {
      stop("authenticated current Phase 3 source commit query failed: ",
           conditionMessage(error), call. = FALSE)
    }
  )
  if (!.fastkpc_full_cuda_phase3_commit(commit)) {
    stop("authenticated current Phase 3 source commit is malformed",
         call. = FALSE)
  }
  commit
}

.fastkpc_full_cuda_phase3_qualified_native_cache <-
  new.env(parent = emptyenv())

.fastkpc_full_cuda_phase3_native_build_inputs <- function() {
  ids <- sort(c(
    "fastkpc/tools/build_cuda_native.sh",
    list.files(
      "fastkpc/src", pattern = "\\.(c|cc|cpp|cu|cuh|h|hpp)$",
      recursive = TRUE, full.names = TRUE
    )
  ), method = "radix")
  if (length(ids) == 0L || anyDuplicated(ids) || any(!file.exists(ids)) ||
      any(dir.exists(ids))) {
    stop("Phase 3 native build input closure is malformed", call. = FALSE)
  }
  setNames(vapply(
    ids, normalizePath, character(1L), winslash = "/", mustWork = TRUE
  ), ids)
}

.fastkpc_full_cuda_phase3_require_qualified_helpers <- function() {
  required <- c(
    "load_fastkpc_cuda_native_qualified",
    "fastkpc_full_cuda_fixed_sp_discover_execution_source_closure",
    "fastkpc_full_cuda_fixed_sp_sha256_file",
    "fastkpc_full_cuda_fixed_sp_capture_native_build_dependencies",
    "fastkpc_full_cuda_fixed_sp_capture_execution_provenance",
    "fastkpc_full_cuda_fixed_sp_verify_execution_provenance"
  )
  missing <- required[!vapply(required, exists, logical(1L), mode = "function")]
  if (length(missing) > 0L) {
    stop("authenticated Phase 3 qualified-build helpers are unavailable: ",
         paste(missing, collapse = ","), call. = FALSE)
  }
  invisible(TRUE)
}

.fastkpc_full_cuda_phase3_capture_qualified_native <- function() {
  .fastkpc_full_cuda_phase3_require_qualified_helpers()
  roots <- .fastkpc_full_cuda_phase3_execution_roots()
  if (length(roots) == 0L) {
    stop("Phase 3 execution source roots are unavailable", call. = FALSE)
  }
  project_root <- normalizePath(".", winslash = "/", mustWork = TRUE)
  closure <- fastkpc_full_cuda_fixed_sp_discover_execution_source_closure(
    root_sources = roots, project_root = project_root
  )
  source_hashes <- vapply(
    closure$source_file_paths,
    fastkpc_full_cuda_fixed_sp_sha256_file,
    character(1L)
  )
  native_inputs <- .fastkpc_full_cuda_phase3_native_build_inputs()
  native_input_hashes <- vapply(
    native_inputs, fastkpc_full_cuda_fixed_sp_sha256_file, character(1L)
  )
  trace_path <- tempfile(
    "fastkpc-phase3-input-native-build-", tmpdir = tempdir(),
    fileext = ".strace"
  )
  native_load <- NULL
  tryCatch({
    native_load <- load_fastkpc_cuda_native_qualified(trace_path = trace_path)
    dependencies <-
      fastkpc_full_cuda_fixed_sp_capture_native_build_dependencies(
        trace_path = native_load$trace_path,
        build_working_dir = project_root,
        tracer_path = native_load$tracer_path,
        trace_invocation = native_load$trace_invocation
      )
    provenance <- fastkpc_full_cuda_fixed_sp_capture_execution_provenance(
      source_closure = closure,
      expected_source_sha256 = source_hashes,
      native_library_path = native_load$native_library_path,
      native_build_input_paths = native_inputs,
      expected_native_build_input_sha256 = native_input_hashes,
      native_build_dependencies = dependencies,
      expected_native_library_sha256 = native_load$native_library_sha256
    )
    provenance <- fastkpc_full_cuda_fixed_sp_verify_execution_provenance(
      provenance
    )
    if (!isTRUE(provenance$execution_sources_unchanged_after_run)) {
      stop("Phase 3 qualified native provenance did not verify",
           call. = FALSE)
    }
    list(
      schema_version = "full-cuda-ci-phase3-qualified-native-cache-v1",
      provenance = provenance
    )
  }, error = function(error) {
    if (!is.null(native_load) &&
        is.character(native_load$native_library_path)) {
      rollback <- tryCatch(
        .fastkpc_cuda_rollback_qualified_native_load(native_load),
        error = identity
      )
      if (inherits(rollback, "error")) {
        stop(
          conditionMessage(error),
          "; qualified native rollback failed: ",
          conditionMessage(rollback),
          call. = FALSE
        )
      }
    }
    stop(conditionMessage(error), call. = FALSE)
  })
}

.fastkpc_full_cuda_phase3_verify_qualified_native <- function(value) {
  .fastkpc_full_cuda_phase3_require_qualified_helpers()
  if (!is.list(value) || is.object(value) || !identical(
        names(value), c("schema_version", "provenance")
      ) || !identical(
        value$schema_version, "full-cuda-ci-phase3-qualified-native-cache-v1"
      )) {
    stop("Phase 3 qualified native cache is malformed", call. = FALSE)
  }
  verified <- fastkpc_full_cuda_fixed_sp_verify_execution_provenance(
    value$provenance
  )
  if (!isTRUE(verified$execution_sources_unchanged_after_run)) {
    stop("Phase 3 qualified native cache failed verification", call. = FALSE)
  }
  value$provenance <- verified
  value
}

fastkpc_full_cuda_phase3_discover_qualified_native_evidence <- function() {
  cache <- .fastkpc_full_cuda_phase3_qualified_native_cache
  if (!exists("value", envir = cache, inherits = FALSE)) {
    cache$value <- .fastkpc_full_cuda_phase3_capture_qualified_native()
  }
  cache$value <- .fastkpc_full_cuda_phase3_verify_qualified_native(cache$value)
  cache$value
}

.fastkpc_full_cuda_phase3_execution_projection <- function(provenance) {
  dependencies <- provenance$native_build_dependencies
  if (!requireNamespace("mgcv", quietly = TRUE)) {
    stop("Phase 3 execution requires installed mgcv", call. = FALSE)
  }
  native_identity <- if (all(c(
    "native_library_path", "native_library_device_major_hex",
    "native_library_device_minor_hex", "native_library_inode"
  ) %in% names(provenance))) {
    list(
      path = provenance$native_library_path,
      device_major_hex = provenance$native_library_device_major_hex,
      device_minor_hex = provenance$native_library_device_minor_hex,
      inode = provenance$native_library_inode
    )
  } else {
    .fastkpc_cuda_posix_file_identity(provenance$native_library_path)
  }
  native_identity$device_major_hex <- .fastkpc_cuda_normalize_hex_identity(
    native_identity$device_major_hex
  )
  native_identity$device_minor_hex <- .fastkpc_cuda_normalize_hex_identity(
    native_identity$device_minor_hex
  )
  complete_dependency_projection <- all(c(
    "build_working_dir", "files", "exclusions"
  ) %in% names(dependencies))
  native_build_attestation_sha256 <- if (complete_dependency_projection) {
    fastkpc_full_cuda_fixed_sp_native_build_attestation_hash(dependencies)
  } else {
    .fastkpc_full_cuda_phase3_named_hash(list(
      schema_version = "full-cuda-ci-native-build-trace-attestation-v1",
      trace_semantics = dependencies$trace_semantics,
      native_build_inputs_sha256 = provenance$native_build_inputs_sha256,
      native_library_sha256 = provenance$native_library_sha256,
      tracer_path = dependencies$tracer_path,
      tracer_sha256 = dependencies$tracer_sha256,
      dependency_count = dependencies$dependency_count,
      exclusion_count = dependencies$exclusion_count
    ))
  }
  list(
    authenticated = TRUE,
    source_commit = provenance$head_base_commit,
    phase3_source_commit = provenance$head_base_commit,
    R_version = R.version.string,
    phase3_R_version = R.version.string,
    mgcv_version = as.character(utils::packageVersion("mgcv")),
    phase3_mgcv_version = as.character(utils::packageVersion("mgcv")),
    provenance_schema_version = provenance$provenance_schema_version,
    provenance_mode = provenance$provenance_mode,
    source_closure_schema_version = provenance$source_closure_schema_version,
    source_discovery_semantics = provenance$source_discovery_semantics,
    source_closure_count = provenance$source_closure_count,
    source_closure_sha256 = provenance$source_closure_sha256,
    execution_snapshot_sha256 = provenance$execution_snapshot_sha256,
    relevant_sources_dirty_or_untracked =
      provenance$relevant_sources_dirty_or_untracked,
    execution_sources_unchanged_after_run = FALSE,
    execution_provenance_state = "pre-run-capture",
    native_library_identity = provenance$native_library_identity,
    native_library_path = native_identity$path,
    native_library_device_major_hex = native_identity$device_major_hex,
    native_library_device_minor_hex = native_identity$device_minor_hex,
    native_library_inode = as.character(native_identity$inode),
    native_library_sha256 = provenance$native_library_sha256,
    native_build_inputs_sha256 = provenance$native_build_inputs_sha256,
    native_build_dependencies_schema_version = dependencies$schema_version,
    native_build_attestation_schema_version =
      "full-cuda-ci-native-build-trace-attestation-v1",
    native_build_attestation_sha256 = native_build_attestation_sha256,
    native_build_trace_semantics = dependencies$trace_semantics,
    native_build_trace_invocation = dependencies$trace_invocation,
    native_build_tracer_path = dependencies$tracer_path,
    native_build_dependency_count = dependencies$dependency_count,
    native_build_exclusion_count = dependencies$exclusion_count,
    native_build_dependencies_sha256 = dependencies$aggregate_sha256,
    native_build_trace_sha256 = dependencies$trace_sha256,
    native_build_tracer_sha256 = dependencies$tracer_sha256
  )
}

.fastkpc_full_cuda_phase3_default_execution_evidence <- function(
    catalog, device_id) {
  supplied_names <- c(
    "phase3_execution_evidence", "phase3_execution_provenance",
    "execution_evidence", "execution_provenance"
  )
  if (any(vapply(
        supplied_names, function(name) !is.null(catalog[[name]]), logical(1L)
      ))) {
    stop("catalog-supplied Phase 3 execution evidence is not authoritative",
         call. = FALSE)
  }
  qualified <- fastkpc_full_cuda_phase3_discover_qualified_native_evidence()
  .fastkpc_full_cuda_phase3_validate_execution_evidence(
    .fastkpc_full_cuda_phase3_execution_projection(qualified$provenance)
  )
}

fastkpc_full_cuda_phase3_discover_execution_evidence <- function(
    catalog, device_id) {
  .fastkpc_full_cuda_phase3_default_execution_evidence(catalog, device_id)
}

fastkpc_full_cuda_phase3_discover_runtime_evidence <- function(
    catalog, device_id) {
  .fastkpc_full_cuda_phase3_static_environment_evidence(device_id)
}

fastkpc_full_cuda_phase3_discover_device_evidence <- function(
    catalog, device_id) {
  .fastkpc_full_cuda_phase3_static_environment_evidence(device_id)
}

.fastkpc_full_cuda_phase3_static_environment_fields <- function() {
  c(
    "schema_version", "runtime_abi_schema_version",
    "configuration_schema_version", "device_id", "cuda_toolkit_version",
    "cuda_driver_version", "gpu_name", "gpu_uuid",
    "compute_capability_major", "compute_capability_minor", "sm_count",
    "cusolver_deterministic_mode_required", "cublas_math_mode_required",
    "cublas_atomics_mode_required", "cublas_user_workspace_required",
    "cublas_workspace_bytes_required",
    "cublas_workspace_min_alignment_required"
  )
}

.fastkpc_full_cuda_phase3_policy_contract <- function() {
  list(
    schema_version = "full-cuda-ci-phase3-environment-policy-v1",
    configuration_schema_version =
      "full-cuda-ci-fixed-sp-environment-policy-v1",
    cusolver_deterministic_mode_required = "enabled",
    cublas_math_mode_required = "pedantic",
    cublas_atomics_mode_required = "not_allowed",
    cublas_user_workspace_required = TRUE,
    cublas_workspace_bytes_required = 16 * 1024 * 1024,
    cublas_workspace_min_alignment_required = 256
  )
}

.fastkpc_full_cuda_phase3_static_environment_evidence <- function(
    device_id) {
  if (!exists("fastkpc_cuda_phase3_environment_identity",
             mode = "function")) {
    stop("authenticated Phase 3 static environment query is unavailable",
         call. = FALSE)
  }
  queried <- tryCatch(
    fastkpc_cuda_phase3_environment_identity(device_id),
    error = function(error) {
      stop("Phase 3 static environment query failed: ",
           conditionMessage(error), call. = FALSE)
    }
  )
  fields <- .fastkpc_full_cuda_phase3_static_environment_fields()
  if (!is.list(queried) || is.object(queried) ||
      !identical(names(queried), fields)) {
    stop("Phase 3 static environment query result is malformed",
         call. = FALSE)
  }
  policy <- .fastkpc_full_cuda_phase3_policy_contract()
  if (!identical(queried$schema_version, policy$schema_version) ||
      !identical(queried$runtime_abi_schema_version,
                 fastkpc_full_cuda_fixed_sp_runtime_abi()$schema_version) ||
      !identical(queried$configuration_schema_version,
                 policy$configuration_schema_version)) {
    stop("Phase 3 static environment configuration is not canonical",
         call. = FALSE)
  }
  if (!.fastkpc_full_cuda_phase3_bare_integer(device_id, 0L) ||
      !identical(queried$device_id, device_id)) {
    stop("Phase 3 static environment device identity mismatch",
         call. = FALSE)
  }
  for (field in c(
    "cuda_toolkit_version", "cuda_driver_version",
    "compute_capability_major", "compute_capability_minor", "sm_count"
  )) {
    if (!.fastkpc_full_cuda_phase3_bare_integer(queried[[field]], 0L)) {
      stop("Phase 3 static environment integer is malformed: ", field,
           call. = FALSE)
    }
  }
  for (field in c(
    "gpu_name", "gpu_uuid", "cusolver_deterministic_mode_required",
    "cublas_math_mode_required", "cublas_atomics_mode_required"
  )) {
    .fastkpc_full_cuda_phase3_scalar_text(queried[[field]], field)
  }
  if (!grepl("^GPU-[0-9a-f]{32}$", queried$gpu_uuid) ||
      !identical(queried$cusolver_deterministic_mode_required,
                 policy$cusolver_deterministic_mode_required) ||
      !identical(queried$cublas_math_mode_required,
                 policy$cublas_math_mode_required) ||
      !identical(queried$cublas_atomics_mode_required,
                 policy$cublas_atomics_mode_required) ||
      !.fastkpc_full_cuda_phase3_bare_scalar(
        queried$cublas_user_workspace_required, "logical"
      ) || !identical(queried$cublas_user_workspace_required,
                      policy$cublas_user_workspace_required) ||
      !.fastkpc_full_cuda_phase3_bare_number(
        queried$cublas_workspace_bytes_required, 0
      ) || !.fastkpc_full_cuda_phase3_bare_number(
        queried$cublas_workspace_min_alignment_required, 0
      ) || !identical(queried$cublas_workspace_bytes_required,
                      policy$cublas_workspace_bytes_required) ||
      !identical(queried$cublas_workspace_min_alignment_required,
                 policy$cublas_workspace_min_alignment_required)) {
    stop("Phase 3 static environment configuration is invalid",
         call. = FALSE)
  }
  abi <- fastkpc_full_cuda_fixed_sp_runtime_abi()
  runtime <- list(
    runtime_abi = abi$schema_version,
    runtime_abi_hash = abi$sha256,
    runtime_policy_schema_version =
      queried$configuration_schema_version,
    device_id = queried$device_id,
    cuda_toolkit_version = queried$cuda_toolkit_version,
    cuda_driver_version = queried$cuda_driver_version,
    gpu_name = queried$gpu_name,
    gpu_uuid = queried$gpu_uuid,
    compute_capability_major = queried$compute_capability_major,
    compute_capability_minor = queried$compute_capability_minor,
    compute_capability = paste0(
      queried$compute_capability_major, ".",
      queried$compute_capability_minor
    ),
    sm_count = queried$sm_count,
    cusolver_deterministic_mode_required =
      queried$cusolver_deterministic_mode_required,
    cublas_math_mode_required = queried$cublas_math_mode_required,
    cublas_atomics_mode_required = queried$cublas_atomics_mode_required,
    cublas_user_workspace_required = queried$cublas_user_workspace_required,
    cublas_workspace_bytes_required = queried$cublas_workspace_bytes_required,
    cublas_workspace_min_alignment_required =
      queried$cublas_workspace_min_alignment_required
  )
  runtime
}

.fastkpc_full_cuda_phase3_validate_runtime_evidence <- function(
    runtime, device_id) {
  if (!is.list(runtime) || is.object(runtime) || is.null(names(runtime)) ||
      anyDuplicated(names(runtime))) {
    stop("Phase 3 runtime/device evidence is malformed", call. = FALSE)
  }
  abi <- fastkpc_full_cuda_fixed_sp_runtime_abi()
  policy <- .fastkpc_full_cuda_phase3_policy_contract()
  required <- c(
    "runtime_abi", "runtime_abi_hash",
    "runtime_policy_schema_version", "device_id",
    "cuda_toolkit_version", "cuda_driver_version", "gpu_name", "gpu_uuid",
    "compute_capability_major", "compute_capability_minor",
    "compute_capability", "sm_count",
    "cusolver_deterministic_mode_required", "cublas_math_mode_required",
    "cublas_atomics_mode_required", "cublas_user_workspace_required",
    "cublas_workspace_bytes_required",
    "cublas_workspace_min_alignment_required"
  )
  missing <- setdiff(required, names(runtime))
  if (length(missing) > 0L) {
    stop("Phase 3 runtime/device evidence is missing: ",
         paste(missing, collapse = ","), call. = FALSE)
  }
  if (is.list(runtime$runtime_abi)) {
    supplied_abi <- runtime$runtime_abi
    supplied_hash <- supplied_abi$sha256
    supplied_schema <- supplied_abi$schema_version
  } else {
    supplied_hash <- runtime$runtime_abi_hash
    supplied_schema <- runtime$runtime_abi
  }
  if (!identical(supplied_schema, abi$schema_version) ||
      !identical(supplied_hash, abi$sha256) ||
      !identical(runtime$runtime_abi_hash, abi$sha256)) {
    stop("Phase 3 runtime ABI identity mismatch", call. = FALSE)
  }
  if (!.fastkpc_full_cuda_phase3_bare_integer(device_id, 0L) ||
      !.fastkpc_full_cuda_phase3_bare_integer(runtime$device_id, 0L) ||
      !identical(runtime$device_id, device_id)) {
    stop("Phase 3 device_id identity mismatch", call. = FALSE)
  }
  for (field in c("cuda_toolkit_version", "cuda_driver_version",
                  "compute_capability_major", "compute_capability_minor",
                  "sm_count")) {
    if (!.fastkpc_full_cuda_phase3_bare_integer(runtime[[field]], 0L)) {
      stop("Phase 3 runtime integer evidence is malformed: ", field,
           call. = FALSE)
    }
  }
  for (field in c("runtime_policy_schema_version", "gpu_name",
                  "gpu_uuid", "compute_capability",
                  "cusolver_deterministic_mode_required",
                  "cublas_math_mode_required",
                  "cublas_atomics_mode_required")) {
    .fastkpc_full_cuda_phase3_scalar_text(runtime[[field]], field)
  }
  expected_compute <- paste0(
    runtime$compute_capability_major, ".", runtime$compute_capability_minor
  )
  if (!identical(runtime$compute_capability, expected_compute)) {
    stop("Phase 3 compute capability identity mismatch", call. = FALSE)
  }
  for (field in c(
    "cublas_workspace_bytes_required",
    "cublas_workspace_min_alignment_required"
  )) {
    if (!.fastkpc_full_cuda_phase3_bare_number(runtime[[field]], 0)) {
      stop("Phase 3 runtime policy value is malformed: ", field,
           call. = FALSE)
    }
  }
  if (!identical(runtime$runtime_policy_schema_version,
                 policy$configuration_schema_version) ||
      !grepl("^GPU-[0-9a-f]{32}$", runtime$gpu_uuid) ||
      !identical(runtime$cusolver_deterministic_mode_required,
                 policy$cusolver_deterministic_mode_required) ||
      !identical(runtime$cublas_math_mode_required,
                 policy$cublas_math_mode_required) ||
      !identical(runtime$cublas_atomics_mode_required,
                 policy$cublas_atomics_mode_required) ||
      !.fastkpc_full_cuda_phase3_bare_scalar(
        runtime$cublas_user_workspace_required, "logical"
      ) || !identical(runtime$cublas_user_workspace_required,
                      policy$cublas_user_workspace_required) ||
      !identical(runtime$cublas_workspace_bytes_required,
                 policy$cublas_workspace_bytes_required) ||
      !identical(runtime$cublas_workspace_min_alignment_required,
                 policy$cublas_workspace_min_alignment_required)) {
    stop("Phase 3 runtime policy identity mismatch", call. = FALSE)
  }
  runtime
}

.fastkpc_full_cuda_phase3_merge_evidence <- function(left, right) {
  result <- left
  for (field in names(right)) {
    if (field %in% names(result) &&
        !identical(result[[field]], right[[field]])) {
      stop("Phase 3 runtime/device evidence disagrees: ", field,
           call. = FALSE)
    }
    result[[field]] <- right[[field]]
  }
  result
}

.fastkpc_full_cuda_phase3_validate_runtime_attestation_info <- function(
    info, device_id) {
  required <- c(
    "device_id", "gpu_name", "runtime_abi_schema_version",
    "configuration_schema_version", "gpu_uuid", "create_symbol_image_path",
    "create_symbol_device_major_hex", "create_symbol_device_minor_hex",
    "create_symbol_inode", "info_symbol_image_path",
    "info_symbol_device_major_hex", "info_symbol_device_minor_hex",
    "info_symbol_inode", "cuda_toolkit_version", "cuda_driver_version",
    "compute_capability_major", "compute_capability_minor", "sm_count",
    "cusolver_deterministic_mode", "cublas_math_mode",
    "cublas_atomics_mode", "cublas_user_workspace_installed",
    "cublas_workspace_bytes", "cublas_workspace_alignment", "freed"
  )
  missing <- setdiff(required, names(info))
  if (!is.list(info) || is.object(info) || is.null(names(info)) ||
      anyDuplicated(names(info)) || length(missing) > 0L) {
    stop("Phase 3 runtime attestation info is malformed", call. = FALSE)
  }
  for (field in c(
    "device_id", "cuda_toolkit_version", "cuda_driver_version",
    "compute_capability_major", "compute_capability_minor", "sm_count"
  )) {
    if (!.fastkpc_full_cuda_phase3_bare_integer(info[[field]], 0L)) {
      stop("Phase 3 runtime attestation integer is malformed: ", field,
           call. = FALSE)
    }
  }
  for (field in c("cublas_workspace_bytes", "cublas_workspace_alignment")) {
    if (!.fastkpc_full_cuda_phase3_bare_number(info[[field]], 0)) {
      stop("Phase 3 runtime attestation value is malformed: ", field,
           call. = FALSE)
    }
  }
  for (field in c(
    "gpu_name", "runtime_abi_schema_version",
    "configuration_schema_version", "gpu_uuid", "create_symbol_image_path",
    "create_symbol_device_major_hex", "create_symbol_device_minor_hex",
    "create_symbol_inode", "info_symbol_image_path",
    "info_symbol_device_major_hex", "info_symbol_device_minor_hex",
    "info_symbol_inode", "cusolver_deterministic_mode",
    "cublas_math_mode", "cublas_atomics_mode"
  )) {
    .fastkpc_full_cuda_phase3_scalar_text(info[[field]], field)
  }
  for (field in c("cublas_user_workspace_installed", "freed")) {
    if (!.fastkpc_full_cuda_phase3_bare_scalar(info[[field]], "logical")) {
      stop("Phase 3 runtime attestation flag is malformed: ", field,
           call. = FALSE)
    }
  }
  if (!identical(info$device_id, device_id) || isTRUE(info$freed)) {
    stop("Phase 3 runtime attestation device/lifetime mismatch",
         call. = FALSE)
  }
  if (!identical(
        info$runtime_abi_schema_version,
        fastkpc_full_cuda_fixed_sp_runtime_abi()$schema_version
      ) || !identical(
        info$configuration_schema_version,
        .fastkpc_full_cuda_phase3_policy_contract()$configuration_schema_version
      ) || !grepl("^GPU-[0-9a-f]{32}$", info$gpu_uuid) ||
      !startsWith(info$create_symbol_image_path, "/") ||
      !startsWith(info$info_symbol_image_path, "/") ||
      grepl("[\r\n]", info$create_symbol_image_path) ||
      grepl("[\r\n]", info$info_symbol_image_path) ||
      !grepl("^[0-9a-f]+$", info$create_symbol_device_major_hex) ||
      !grepl("^[0-9a-f]+$", info$create_symbol_device_minor_hex) ||
      !grepl("^[0-9]+$", info$create_symbol_inode) ||
      !grepl("^[0-9a-f]+$", info$info_symbol_device_major_hex) ||
      !grepl("^[0-9a-f]+$", info$info_symbol_device_minor_hex) ||
      !grepl("^[0-9]+$", info$info_symbol_inode)) {
    stop("Phase 3 runtime attestation context identity is malformed",
         call. = FALSE)
  }
  if (!identical(
        info[c(
          "create_symbol_image_path", "create_symbol_device_major_hex",
          "create_symbol_device_minor_hex", "create_symbol_inode"
        )],
        stats::setNames(
          info[c(
            "info_symbol_image_path", "info_symbol_device_major_hex",
            "info_symbol_device_minor_hex", "info_symbol_inode"
          )],
          c(
            "create_symbol_image_path", "create_symbol_device_major_hex",
            "create_symbol_device_minor_hex", "create_symbol_inode"
          )
        )
      )) {
    stop("Phase 3 runtime attestation native symbol identities disagree",
         call. = FALSE)
  }
  info
}

.fastkpc_full_cuda_phase3_validate_runtime_attestation <- function(
    runtime, expected_identity) {
  fastkpc_full_cuda_phase3_validate_input_identity(expected_identity)
  fastkpc_full_cuda_fixed_sp_verify_loaded_native_library(
    expected_identity$native_library_path,
    expected_identity$native_library_sha256
  )
  if (!exists("fixed_sp_cuda_runtime_info", mode = "function")) {
    stop("Phase 3 runtime attestation requires CUDA runtime info",
         call. = FALSE)
  }
  runtime_info <- tryCatch(
    fixed_sp_cuda_runtime_info(runtime),
    error = function(error) {
      stop("Phase 3 runtime attestation query failed: ",
           conditionMessage(error), call. = FALSE)
    }
  )
  runtime_info <- .fastkpc_full_cuda_phase3_validate_runtime_attestation_info(
    runtime_info, expected_identity$device_id
  )
  device_evidence <- .fastkpc_full_cuda_phase3_validate_runtime_evidence(
    fastkpc_full_cuda_phase3_discover_device_evidence(
      list(), expected_identity$device_id
    ),
    expected_identity$device_id
  )
  execution_evidence <- .fastkpc_full_cuda_phase3_validate_execution_evidence(
    fastkpc_full_cuda_phase3_discover_execution_evidence(
      list(), expected_identity$device_id
    )
  )
  for (field in c(
    "runtime_abi", "runtime_abi_hash", "runtime_policy_schema_version",
    "device_id", "cuda_toolkit_version", "cuda_driver_version",
    "gpu_name", "gpu_uuid", "compute_capability_major",
    "compute_capability_minor", "compute_capability", "sm_count",
    "cusolver_deterministic_mode_required", "cublas_math_mode_required",
    "cublas_atomics_mode_required", "cublas_user_workspace_required",
    "cublas_workspace_bytes_required",
    "cublas_workspace_min_alignment_required"
  )) {
    if (!.fastkpc_full_cuda_phase3_identity_value_equal(
          device_evidence[[field]], expected_identity[[field]]
        )) {
      stop("Phase 3 static device attestation mismatch: ", field,
           call. = FALSE)
    }
  }
  for (field in c(
    "source_commit", "phase3_source_commit", "R_version",
    "phase3_R_version", "mgcv_version", "phase3_mgcv_version",
    "provenance_schema_version", "provenance_mode",
    "source_closure_schema_version", "source_discovery_semantics",
    "source_closure_count", "source_closure_sha256",
    "execution_snapshot_sha256", "relevant_sources_dirty_or_untracked",
    "execution_sources_unchanged_after_run", "execution_provenance_state",
    "native_library_identity", "native_library_path",
    "native_library_device_major_hex", "native_library_device_minor_hex",
    "native_library_inode", "native_library_sha256",
    "native_build_inputs_sha256",
    "native_build_dependencies_schema_version",
    "native_build_attestation_schema_version",
    "native_build_attestation_sha256",
    "native_build_trace_semantics", "native_build_trace_invocation",
    "native_build_tracer_path", "native_build_dependency_count",
    "native_build_exclusion_count", "native_build_dependencies_sha256",
    "native_build_trace_sha256", "native_build_tracer_sha256"
  )) {
    if (!.fastkpc_full_cuda_phase3_identity_value_equal(
          execution_evidence[[field]], expected_identity[[field]]
        )) {
      stop("Phase 3 execution/native attestation mismatch: ", field,
           call. = FALSE)
    }
  }
  if (!identical(runtime_info$gpu_name, expected_identity$gpu_name) ||
      !identical(runtime_info$runtime_abi_schema_version,
                 expected_identity$runtime_abi) ||
      !identical(runtime_info$configuration_schema_version,
                 expected_identity$runtime_policy_schema_version) ||
      !identical(runtime_info$gpu_uuid, expected_identity$gpu_uuid) ||
      !identical(runtime_info$cuda_toolkit_version,
                 expected_identity$cuda_toolkit_version) ||
      !identical(runtime_info$cuda_driver_version,
                 expected_identity$cuda_driver_version) ||
      !identical(runtime_info$compute_capability_major,
                 expected_identity$compute_capability_major) ||
      !identical(runtime_info$compute_capability_minor,
                 expected_identity$compute_capability_minor) ||
      !identical(runtime_info$sm_count, expected_identity$sm_count) ||
      !identical(runtime_info$cusolver_deterministic_mode,
                 expected_identity$cusolver_deterministic_mode_required) ||
      !identical(runtime_info$cublas_math_mode,
                 expected_identity$cublas_math_mode_required) ||
      !identical(runtime_info$cublas_atomics_mode,
                 expected_identity$cublas_atomics_mode_required) ||
      !isTRUE(runtime_info$cublas_user_workspace_installed) ||
      !identical(runtime_info$cublas_workspace_bytes,
                 expected_identity$cublas_workspace_bytes_required) ||
      runtime_info$cublas_workspace_alignment <
        expected_identity$cublas_workspace_min_alignment_required) {
    stop("Phase 3 runtime attestation mismatch", call. = FALSE)
  }
  native_symbol_fields <- c(
    "path", "device_major_hex", "device_minor_hex", "inode"
  )
  create_symbol_identity <- list(
    path = runtime_info$create_symbol_image_path,
    device_major_hex = .fastkpc_cuda_normalize_hex_identity(
      runtime_info$create_symbol_device_major_hex
    ),
    device_minor_hex = .fastkpc_cuda_normalize_hex_identity(
      runtime_info$create_symbol_device_minor_hex
    ),
    inode = runtime_info$create_symbol_inode
  )
  info_symbol_identity <- list(
    path = runtime_info$info_symbol_image_path,
    device_major_hex = .fastkpc_cuda_normalize_hex_identity(
      runtime_info$info_symbol_device_major_hex
    ),
    device_minor_hex = .fastkpc_cuda_normalize_hex_identity(
      runtime_info$info_symbol_device_minor_hex
    ),
    inode = runtime_info$info_symbol_inode
  )
  expected_symbol_identity <- list(
    path = expected_identity$native_library_path,
    device_major_hex = .fastkpc_cuda_normalize_hex_identity(
      expected_identity$native_library_device_major_hex
    ),
    device_minor_hex = .fastkpc_cuda_normalize_hex_identity(
      expected_identity$native_library_device_minor_hex
    ),
    inode = expected_identity$native_library_inode
  )
  if (!identical(create_symbol_identity[native_symbol_fields],
                 expected_symbol_identity[native_symbol_fields]) ||
      !identical(info_symbol_identity[native_symbol_fields],
                 expected_symbol_identity[native_symbol_fields])) {
    stop("Phase 3 runtime native image attestation mismatch",
         call. = FALSE)
  }
  symbol_sha256 <- tryCatch(
    fastkpc_full_cuda_fixed_sp_sha256_file(
      runtime_info$create_symbol_image_path
    ),
    error = function(error) stop(
      "Phase 3 runtime native image hash check failed: ",
      conditionMessage(error), call. = FALSE
    )
  )
  if (!identical(symbol_sha256, expected_identity$native_library_sha256)) {
    stop("Phase 3 runtime native image SHA mismatch", call. = FALSE)
  }
  fastkpc_full_cuda_fixed_sp_verify_loaded_native_library(
    expected_identity$native_library_path,
    expected_identity$native_library_sha256
  )
  list(
    runtime_info = runtime_info,
    device_evidence = device_evidence,
    execution_evidence = execution_evidence,
    authenticated = TRUE
  )
}

.fastkpc_full_cuda_phase3_default_catalog_evidence <- function(catalog) {
  authority <- fastkpc_full_cuda_phase3_discover_catalog_authority(catalog)
  .fastkpc_full_cuda_phase3_validate_lineage(authority$lineage)
}

fastkpc_full_cuda_phase3_discover_catalog_evidence <- function(catalog) {
  .fastkpc_full_cuda_phase3_default_catalog_evidence(catalog)
}

fastkpc_full_cuda_phase3_validate_input_identity <- function(
    identity, expected = NULL) {
  fields <- .fastkpc_full_cuda_phase3_identity_fields()
  if (!is.list(identity) || is.object(identity) ||
      !identical(names(identity), c(fields, "sha256")) ||
      anyDuplicated(names(identity))) {
    stop("Phase 3 input identity schema is malformed", call. = FALSE)
  }
  if (!identical(identity$schema_version,
                 "full-cuda-ci-phase3-input-identity-v1") ||
      !isTRUE(identity$authenticated)) {
    stop("Phase 3 input identity is not authenticated", call. = FALSE)
  }
  for (field in c(
    "phase0_manifest_hash", "phase1_manifest_hash", "phase2_manifest_hash",
    "dataset_file_sha256", "dataset_matrix_sha256",
    "canonical_setup_corpus_hash", "canonical_target_corpus_hash",
    "runtime_abi_hash", "route_config_hash", "source_closure_sha256",
    "execution_snapshot_sha256", "native_library_sha256",
    "native_build_inputs_sha256", "native_build_dependencies_sha256",
    "native_build_trace_sha256", "native_build_tracer_sha256",
    "native_build_attestation_sha256"
  )) {
    if (!.fastkpc_full_cuda_phase3_sha256(identity[[field]])) {
      stop("Phase 3 input identity hash is malformed: ", field,
           call. = FALSE)
    }
  }
  for (field in c(
    "phase0_source_commit", "phase1_source_commit", "phase2_source_commit",
    "source_commit", "phase3_source_commit"
  )) {
    if (!.fastkpc_full_cuda_phase3_commit(identity[[field]])) {
      stop("Phase 3 input identity source commit is malformed: ", field,
           call. = FALSE)
    }
  }
  if (!identical(identity$source_commit, identity$phase3_source_commit)) {
    stop("Phase 3 input identity execution source aliases disagree",
         call. = FALSE)
  }
  for (field in c("runtime_abi", "runtime_policy_schema_version",
                  "R_version", "mgcv_version", "gpu_name",
                  "gpu_uuid", "compute_capability",
                  "cusolver_deterministic_mode_required",
                  "cublas_math_mode_required",
                  "cublas_atomics_mode_required",
                  "phase2_R_version", "phase2_mgcv_version",
                  "phase3_R_version", "phase3_mgcv_version",
                  "execution_provenance_state",
                  "provenance_schema_version", "provenance_mode",
                  "source_closure_schema_version",
                  "source_discovery_semantics",
                  "native_library_identity", "native_library_path",
                  "native_library_device_major_hex",
                  "native_library_device_minor_hex", "native_library_inode",
                  "native_build_dependencies_schema_version",
                  "native_build_attestation_schema_version",
                  "native_build_trace_semantics",
                  "native_build_trace_invocation",
                  "native_build_tracer_path")) {
    .fastkpc_full_cuda_phase3_scalar_text(identity[[field]], field)
  }
  if (!identical(identity$R_version, identity$phase3_R_version) ||
      !identical(identity$mgcv_version, identity$phase3_mgcv_version)) {
    stop("Phase 3 input identity execution version aliases disagree",
         call. = FALSE)
  }
  identity_state_consistent <- identity$execution_provenance_state %in% c(
    "pre-run-capture", "post-run-verified"
  ) && (
    identical(identity$execution_provenance_state, "pre-run-capture") &&
      identical(identity$execution_sources_unchanged_after_run, FALSE) ||
    identical(identity$execution_provenance_state, "post-run-verified") &&
      identical(identity$execution_sources_unchanged_after_run, TRUE)
  )
  if (!isTRUE(identity_state_consistent)) {
    stop("Phase 3 input identity provenance state is invalid", call. = FALSE)
  }
  if (!identical(
        identity$provenance_schema_version,
        "full-cuda-ci-execution-source-snapshot-v6"
      ) || !identical(
        identity$provenance_mode, "working-tree-execution-snapshot-v1"
      ) || !identical(
        identity$source_closure_schema_version,
        "full-cuda-ci-execution-source-closure-v1"
      ) || !identical(
        identity$source_discovery_semantics,
        "parsed-r-ast-load-time-literal-source-v1"
      ) || !.fastkpc_full_cuda_phase3_bare_integer(
        identity$source_closure_count, 1L
      )) {
    stop("Phase 3 input identity provenance configuration mismatch",
         call. = FALSE)
  }
  for (field in c(
    "relevant_sources_dirty_or_untracked",
    "execution_sources_unchanged_after_run",
    "cublas_user_workspace_required"
  )) {
    if (!.fastkpc_full_cuda_phase3_bare_scalar(identity[[field]], "logical")) {
      stop("Phase 3 input identity provenance flag is malformed: ", field,
           call. = FALSE)
    }
  }
  for (field in c("cuda_toolkit_version", "cuda_driver_version",
                  "compute_capability_major", "compute_capability_minor",
                  "sm_count", "device_id", "shard_count",
                  "native_build_dependency_count",
                  "native_build_exclusion_count")) {
    if (!.fastkpc_full_cuda_phase3_bare_integer(identity[[field]], 0L)) {
      stop("Phase 3 input identity integer is malformed: ", field,
           call. = FALSE)
    }
  }
  for (field in c(
    "cublas_workspace_bytes_required",
    "cublas_workspace_min_alignment_required"
  )) {
    if (!.fastkpc_full_cuda_phase3_bare_number(identity[[field]], 0)) {
      stop("Phase 3 input identity workspace value is malformed: ", field,
           call. = FALSE)
    }
  }
  policy <- .fastkpc_full_cuda_phase3_policy_contract()
  if (!identical(identity$runtime_abi,
                 fastkpc_full_cuda_fixed_sp_runtime_abi()$schema_version) ||
      !identical(identity$runtime_abi_hash,
                 fastkpc_full_cuda_fixed_sp_runtime_abi()$sha256) ||
      !identical(identity$runtime_policy_schema_version,
                 policy$configuration_schema_version) ||
      !identical(identity$cusolver_deterministic_mode_required,
                 policy$cusolver_deterministic_mode_required) ||
      !identical(identity$cublas_math_mode_required,
                 policy$cublas_math_mode_required) ||
      !identical(identity$cublas_atomics_mode_required,
                 policy$cublas_atomics_mode_required) ||
      !identical(identity$cublas_user_workspace_required,
                 policy$cublas_user_workspace_required) ||
      !identical(identity$cublas_workspace_bytes_required,
                 policy$cublas_workspace_bytes_required) ||
      !identical(identity$cublas_workspace_min_alignment_required,
                 policy$cublas_workspace_min_alignment_required) ||
      !identical(identity$compute_capability,
                 paste0(identity$compute_capability_major, ".",
                        identity$compute_capability_minor)) ||
      !grepl("^GPU-[0-9a-f]{32}$", identity$gpu_uuid) ||
      !identical(identity$native_library_identity,
                 "qualified-pinned-inode-sha-exact-registered-mapped-path-v3") ||
      !startsWith(identity$native_library_path, "/") ||
      grepl("[\r\n]", identity$native_library_path) ||
      !grepl("^[0-9a-f]+$", identity$native_library_device_major_hex) ||
      !grepl("^[0-9a-f]+$", identity$native_library_device_minor_hex) ||
      !grepl("^[0-9]+$", identity$native_library_inode) ||
      !identical(identity$native_build_dependencies_schema_version,
                 "full-cuda-ci-native-build-dependencies-v3") ||
      !identical(identity$native_build_attestation_schema_version,
                 "full-cuda-ci-native-build-trace-attestation-v1") ||
      !identical(identity$native_build_trace_semantics,
                 "linux-strace-successful-read-exec-evidence-v3") ||
      identity$native_build_dependency_count < 1L ||
      !identical(identity$shard_count,
                 fastkpc_full_cuda_phase3_route_config()$shard_count) ||
      !identical(identity$route_config_hash,
                 fastkpc_full_cuda_phase3_route_config()$sha256)) {
    stop("Phase 3 input identity execution configuration mismatch",
         call. = FALSE)
  }
  if (!.fastkpc_full_cuda_phase3_sha256(identity$sha256) ||
      !identical(identity$sha256,
                 .fastkpc_full_cuda_phase3_identity_hash(identity))) {
    stop("Phase 3 input identity canonical hash mismatch", call. = FALSE)
  }
  if (!is.null(expected)) {
    if (.fastkpc_full_cuda_phase3_bare_scalar(expected, "character")) {
      if (!identical(identity$sha256, expected)) {
        stop("Phase 3 input identity does not match expected identity",
             call. = FALSE)
      }
    } else {
      fastkpc_full_cuda_phase3_validate_input_identity(expected)
      stable_fields <- .fastkpc_full_cuda_phase3_stable_identity_fields()
      if (!identical(identity[stable_fields], expected[stable_fields])) {
        stop("Phase 3 input identity does not match expected identity",
             call. = FALSE)
      }
    }
  }
  invisible(TRUE)
}

fastkpc_full_cuda_phase3_input_identity <- function(catalog, device_id) {
  if (!is.list(catalog) || is.object(catalog) ||
      !.fastkpc_full_cuda_phase3_bare_integer(device_id, 0L)) {
    stop("Phase 3 catalog/device_id inputs are malformed", call. = FALSE)
  }
  lineage <- fastkpc_full_cuda_phase3_discover_catalog_evidence(catalog)
  lineage <- .fastkpc_full_cuda_phase3_validate_lineage(lineage)
  execution <- fastkpc_full_cuda_phase3_discover_execution_evidence(
    catalog, device_id
  )
  execution <- .fastkpc_full_cuda_phase3_validate_execution_evidence(
    execution
  )
  runtime <- fastkpc_full_cuda_phase3_discover_runtime_evidence(
    catalog, device_id
  )
  device <- fastkpc_full_cuda_phase3_discover_device_evidence(
    catalog, device_id
  )
  if (!is.list(runtime) || is.object(runtime) ||
      !is.list(device) || is.object(device)) {
    stop("Phase 3 runtime/device evidence returned malformed evidence",
         call. = FALSE)
  }
  runtime <- .fastkpc_full_cuda_phase3_merge_evidence(runtime, device)
  runtime <- .fastkpc_full_cuda_phase3_validate_runtime_evidence(
    runtime, device_id
  )
  route <- fastkpc_full_cuda_phase3_route_config()
  abi <- fastkpc_full_cuda_fixed_sp_runtime_abi()
  identity <- list(
    schema_version = "full-cuda-ci-phase3-input-identity-v1",
    phase0_manifest_hash = lineage$phase0_manifest_hash,
    phase1_manifest_hash = lineage$phase1_manifest_hash,
    phase2_manifest_hash = lineage$phase2_manifest_hash,
    dataset_file_sha256 = lineage$dataset_file_sha256,
    dataset_matrix_sha256 = lineage$dataset_matrix_sha256,
    canonical_setup_corpus_hash = lineage$canonical_setup_corpus_hash,
    canonical_target_corpus_hash = lineage$canonical_target_corpus_hash,
    phase0_source_commit = lineage$phase0_source_commit,
    phase1_source_commit = lineage$phase1_source_commit,
    phase2_source_commit = lineage$phase2_source_commit,
    phase2_R_version = lineage$phase2_R_version,
    phase2_mgcv_version = lineage$phase2_mgcv_version,
    runtime_abi = abi$schema_version,
    runtime_abi_hash = abi$sha256,
    runtime_policy_schema_version = runtime$runtime_policy_schema_version,
    route_config_hash = route$sha256,
    source_commit = execution$source_commit,
    phase3_source_commit = execution$phase3_source_commit,
    R_version = execution$R_version,
    phase3_R_version = execution$phase3_R_version,
    mgcv_version = execution$mgcv_version,
    phase3_mgcv_version = execution$phase3_mgcv_version,
    provenance_schema_version = execution$provenance_schema_version,
    provenance_mode = execution$provenance_mode,
    source_closure_schema_version = execution$source_closure_schema_version,
    source_discovery_semantics = execution$source_discovery_semantics,
    source_closure_count = execution$source_closure_count,
    source_closure_sha256 = execution$source_closure_sha256,
    execution_snapshot_sha256 = execution$execution_snapshot_sha256,
    relevant_sources_dirty_or_untracked =
      execution$relevant_sources_dirty_or_untracked,
    execution_sources_unchanged_after_run =
      execution$execution_sources_unchanged_after_run,
    execution_provenance_state = execution$execution_provenance_state,
    native_library_identity = execution$native_library_identity,
    native_library_path = execution$native_library_path,
    native_library_device_major_hex =
      execution$native_library_device_major_hex,
    native_library_device_minor_hex =
      execution$native_library_device_minor_hex,
    native_library_inode = execution$native_library_inode,
    native_library_sha256 = execution$native_library_sha256,
    native_build_inputs_sha256 = execution$native_build_inputs_sha256,
    native_build_dependencies_schema_version =
      execution$native_build_dependencies_schema_version,
    native_build_attestation_schema_version =
      execution$native_build_attestation_schema_version,
    native_build_attestation_sha256 =
      execution$native_build_attestation_sha256,
    native_build_trace_semantics = execution$native_build_trace_semantics,
    native_build_trace_invocation = execution$native_build_trace_invocation,
    native_build_tracer_path = execution$native_build_tracer_path,
    native_build_dependency_count = execution$native_build_dependency_count,
    native_build_exclusion_count = execution$native_build_exclusion_count,
    native_build_dependencies_sha256 =
      execution$native_build_dependencies_sha256,
    native_build_trace_sha256 = execution$native_build_trace_sha256,
    native_build_tracer_sha256 = execution$native_build_tracer_sha256,
    cuda_toolkit_version = runtime$cuda_toolkit_version,
    cuda_driver_version = runtime$cuda_driver_version,
    gpu_name = runtime$gpu_name,
    gpu_uuid = runtime$gpu_uuid,
    compute_capability_major = runtime$compute_capability_major,
    compute_capability_minor = runtime$compute_capability_minor,
    compute_capability = runtime$compute_capability,
    sm_count = runtime$sm_count,
    device_id = runtime$device_id,
    cusolver_deterministic_mode_required =
      runtime$cusolver_deterministic_mode_required,
    cublas_math_mode_required = runtime$cublas_math_mode_required,
    cublas_atomics_mode_required = runtime$cublas_atomics_mode_required,
    cublas_user_workspace_required = runtime$cublas_user_workspace_required,
    cublas_workspace_bytes_required = runtime$cublas_workspace_bytes_required,
    cublas_workspace_min_alignment_required =
      runtime$cublas_workspace_min_alignment_required,
    shard_count = route$shard_count,
    authenticated = TRUE
  )
  identity$sha256 <- .fastkpc_full_cuda_phase3_identity_hash(identity)
  fastkpc_full_cuda_phase3_validate_input_identity(identity)
  identity
}

.fastkpc_full_cuda_phase3_payload_keys <- function(kind) {
  paths <- fastkpc_full_cuda_phase3_artifact_paths(
    ".phase3-contract", kind = kind
  )
  setdiff(names(paths), c(
    "manifest_json", "summary_json", "shards_dir", "sessions_dir"
  ))
}

.fastkpc_full_cuda_phase3_read_json <- function(path, label) {
  if (!is.character(path) || length(path) != 1L || is.na(path) ||
      !file.exists(path) || dir.exists(path)) {
    stop(label, " is missing", call. = FALSE)
  }
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("jsonlite is required for Phase 3 artifact validation",
         call. = FALSE)
  }
  value <- tryCatch(
    jsonlite::read_json(path, simplifyVector = TRUE),
    error = function(error) {
      stop(label, " is not valid JSON: ", conditionMessage(error),
           call. = FALSE)
    }
  )
  if (!is.list(value) || is.object(value) || is.null(names(value)) ||
      anyDuplicated(names(value))) {
    stop(label, " JSON object is malformed", call. = FALSE)
  }
  value
}

.fastkpc_full_cuda_phase3_named_sha256 <- function(value, label) {
  if (is.null(value)) {
    stop(label, " is missing", call. = FALSE)
  }
  if (is.list(value)) {
    if (is.object(value) || is.null(names(value)) || anyDuplicated(names(value))) {
      stop(label, " must be a unique named SHA-256 map", call. = FALSE)
    }
    values <- vapply(value, function(item) {
      if (!.fastkpc_full_cuda_phase3_sha256(item)) {
        stop(label, " contains a malformed SHA-256", call. = FALSE)
      }
      item
    }, character(1L))
  } else if (typeof(value) == "character" && !is.object(value) &&
             !is.null(names(value)) && !anyDuplicated(names(value)) &&
             !anyNA(value) && all(grepl("^[0-9a-f]{64}$", value))) {
    values <- value
  } else {
    stop(label, " must be a unique named SHA-256 map", call. = FALSE)
  }
  values
}

.fastkpc_full_cuda_phase3_validate_payload_encoding <- function(
    path, label) {
  extension <- tolower(tools::file_ext(path))
  if (identical(extension, "rds")) {
    tryCatch(
      readRDS(path),
      error = function(error) {
        stop(label, " is not a readable RDS payload", call. = FALSE)
      }
    )
  } else if (identical(extension, "json")) {
    .fastkpc_full_cuda_phase3_read_json(path, label)
  } else if (identical(extension, "csv")) {
    parsed <- tryCatch(
      utils::read.csv(
        path, stringsAsFactors = FALSE, check.names = FALSE,
        blank.lines.skip = FALSE
      ),
      error = function(error) NULL
    )
    if (!is.data.frame(parsed)) {
      stop(label, " is not a readable CSV payload", call. = FALSE)
    }
  } else if (identical(extension, "txt")) {
    lines <- tryCatch(readLines(path, warn = FALSE, encoding = "bytes"),
                      error = function(error) character())
    if (length(lines) == 0L) {
      stop(label, " is not a readable text payload", call. = FALSE)
    }
  } else {
    stop(label, " has an unsupported payload encoding", call. = FALSE)
  }
  invisible(TRUE)
}

.fastkpc_full_cuda_phase3_identity_value_equal <- function(actual, expected) {
  if (.fastkpc_full_cuda_phase3_bare_scalar(expected, "character")) {
    return(.fastkpc_full_cuda_phase3_bare_scalar(actual, "character") &&
             identical(actual, expected))
  }
  if (.fastkpc_full_cuda_phase3_bare_scalar(expected, "logical")) {
    return(.fastkpc_full_cuda_phase3_bare_scalar(actual, "logical") &&
             identical(actual, expected))
  }
  if (.fastkpc_full_cuda_phase3_bare_integer(expected)) {
    return(.fastkpc_full_cuda_phase3_bare_number(actual) &&
             actual == expected && actual == floor(actual))
  }
  if (.fastkpc_full_cuda_phase3_bare_number(expected)) {
    return(.fastkpc_full_cuda_phase3_bare_number(actual) &&
             identical(as.numeric(actual), as.numeric(expected)))
  }
  identical(actual, expected)
}

.fastkpc_full_cuda_phase3_validate_manifest_identity <- function(
    manifest, identity) {
  fields <- .fastkpc_full_cuda_phase3_identity_fields()
  manifest_fields <- setdiff(fields, "schema_version")
  missing <- setdiff(manifest_fields, names(manifest))
  if (length(missing) > 0L) {
    stop("Phase 3 manifest identity is incomplete: ",
         paste(missing, collapse = ","), call. = FALSE)
  }
  for (field in manifest_fields) {
    if (!.fastkpc_full_cuda_phase3_identity_value_equal(
          manifest[[field]], identity[[field]]
        )) {
      stop("Phase 3 manifest identity mismatch: ", field, call. = FALSE)
    }
  }
  if (!identical(manifest$input_identity_schema_version,
                 identity$schema_version) ||
      !identical(manifest$input_identity_sha256, identity$sha256)) {
    stop("Phase 3 manifest input identity link is invalid", call. = FALSE)
  }
  if ("input_identity" %in% names(manifest)) {
    embedded <- manifest$input_identity
    if (!is.list(embedded) || is.object(embedded) ||
        !identical(embedded$sha256, identity$sha256)) {
      stop("Phase 3 embedded input identity is invalid", call. = FALSE)
    }
  }
  invisible(TRUE)
}

.fastkpc_full_cuda_phase3_validate_route_file <- function(
    path, expected_route_hash) {
  route <- .fastkpc_full_cuda_phase3_read_json(path, "route_config.json")
  expected <- fastkpc_full_cuda_phase3_route_config()
  if (!identical(names(route), names(expected)) ||
      any(!vapply(names(expected), function(field) {
        .fastkpc_full_cuda_phase3_identity_value_equal(
          route[[field]], expected[[field]]
        )
      }, logical(1L)))) {
    stop("Phase 3 route configuration is not canonical", call. = FALSE)
  }
  if (!identical(route$sha256, expected_route_hash) ||
      !identical(route$sha256,
                 .fastkpc_full_cuda_phase3_named_hash(
                   route[setdiff(names(route), "sha256")]
                 ))) {
    stop("Phase 3 route configuration hash mismatch", call. = FALSE)
  }
  invisible(route)
}

fastkpc_full_cuda_phase3_validate_artifact <- function(
    output_dir, kind = NULL, expected_identity = NULL, require_full = FALSE,
    catalog = NULL, device_id = NULL) {
  if (is.null(kind)) {
    stop("Phase 3 artifact kind is required", call. = FALSE)
  }
  paths <- fastkpc_full_cuda_phase3_artifact_paths(output_dir, kind = kind)
  if (!dir.exists(output_dir)) {
    stop("Phase 3 artifact directory is missing", call. = FALSE)
  }
  if (!.fastkpc_full_cuda_phase3_bare_scalar(require_full, "logical")) {
    stop("require_full must be a logical scalar", call. = FALSE)
  }
  if (xor(is.null(catalog), is.null(device_id))) {
    stop("catalog validation requires catalog and device_id together",
         call. = FALSE)
  }
  if (!is.null(catalog)) {
    discovered_identity <- fastkpc_full_cuda_phase3_input_identity(
      catalog, device_id
    )
    if (!is.null(expected_identity)) {
      fastkpc_full_cuda_phase3_validate_input_identity(expected_identity)
      if (!identical(
            expected_identity[.fastkpc_full_cuda_phase3_identity_fields()],
            discovered_identity[.fastkpc_full_cuda_phase3_identity_fields()]
          )) {
        stop("precomputed Phase 3 identity disagrees with catalog identity",
             call. = FALSE)
      }
    }
    expected_identity <- discovered_identity
  }
  if (is.null(expected_identity)) {
    stop("Phase 3 artifact validation requires authenticated expected identity",
         call. = FALSE)
  }
  fastkpc_full_cuda_phase3_validate_input_identity(expected_identity)

  final_keys <- setdiff(names(paths), c("shards_dir", "sessions_dir"))
  final_paths <- unlist(paths[final_keys], use.names = FALSE)
  expected_entries <- c(
    basename(final_paths), basename(paths$shards_dir),
    basename(paths$sessions_dir)
  )
  actual_entries <- list.files(
    output_dir, all.files = TRUE, no.. = TRUE, full.names = FALSE
  )
  if (!identical(
        sort(actual_entries, method = "radix"),
        sort(expected_entries, method = "radix")
      )) {
    stop("Phase 3 artifact top-level file surface is incomplete or forged",
         call. = FALSE)
  }
  if (!dir.exists(paths$shards_dir) || !dir.exists(paths$sessions_dir) ||
      any(!file.exists(final_paths)) || any(dir.exists(final_paths))) {
    stop("Phase 3 artifact path set is incomplete", call. = FALSE)
  }
  if (!file.exists(paths$manifest_json)) {
    stop("Phase 3 artifact manifest is missing", call. = FALSE)
  }
  if (!file.exists(paths$summary_json)) {
    stop("Phase 3 artifact summary completion marker is missing",
         call. = FALSE)
  }

  manifest <- .fastkpc_full_cuda_phase3_read_json(
    paths$manifest_json, "manifest.json"
  )
  summary <- .fastkpc_full_cuda_phase3_read_json(
    paths$summary_json, "summary.json"
  )
  expected_schema <- if (identical(kind, "oracle_sp")) {
    fastkpc_full_cuda_phase3_oracle_schema_version()
  } else {
    fastkpc_full_cuda_phase3_shadow_schema_version()
  }
  required_manifest <- c(
    "artifact_schema_version", "artifact_kind",
    "input_identity_schema_version", "input_identity_sha256",
    "payload_names", "publication_order", "payload_file_sha256",
    setdiff(.fastkpc_full_cuda_phase3_identity_fields(), "schema_version")
  )
  missing_manifest <- setdiff(required_manifest, names(manifest))
  if (length(missing_manifest) > 0L) {
    stop("Phase 3 manifest is incomplete: ",
         paste(missing_manifest, collapse = ","), call. = FALSE)
  }
  if (!identical(manifest$artifact_schema_version, expected_schema) ||
      !identical(manifest$artifact_kind, kind)) {
    stop("Phase 3 artifact schema/kind mismatch", call. = FALSE)
  }
  .fastkpc_full_cuda_phase3_validate_manifest_identity(
    manifest, expected_identity
  )

  payload_keys <- .fastkpc_full_cuda_phase3_payload_keys(kind)
  expected_payload_names <- unname(vapply(
    paths[payload_keys], basename, character(1L)
  ))
  payload_names <- manifest$payload_names
  if (typeof(payload_names) != "character" || is.object(payload_names) ||
      !is.null(attributes(payload_names)) || anyNA(payload_names) ||
      anyDuplicated(payload_names) ||
      !identical(payload_names, expected_payload_names)) {
    stop("Phase 3 manifest payload name set is invalid", call. = FALSE)
  }
  expected_publication_order <- c(
    expected_payload_names, "manifest.json", "summary.json"
  )
  if (typeof(manifest$publication_order) != "character" ||
      is.object(manifest$publication_order) ||
      !is.null(attributes(manifest$publication_order)) ||
      anyNA(manifest$publication_order) ||
      !identical(manifest$publication_order, expected_publication_order)) {
    stop("Phase 3 manifest publication order is invalid", call. = FALSE)
  }
  payload_hashes <- .fastkpc_full_cuda_phase3_named_sha256(
    manifest$payload_file_sha256, "manifest payload hashes"
  )
  if (!identical(names(payload_hashes), expected_payload_names)) {
    stop("Phase 3 manifest payload hash namespace is invalid", call. = FALSE)
  }
  actual_payload_hashes <- vapply(
    paths[payload_keys], fastkpc_full_cuda_census_file_hash, character(1L)
  )
  names(actual_payload_hashes) <- expected_payload_names
  payload_sizes <- as.numeric(file.info(
    unlist(paths[payload_keys], use.names = FALSE)
  )$size)
  if (anyNA(payload_sizes) || any(payload_sizes <= 0)) {
    stop("Phase 3 artifact contains an empty payload", call. = FALSE)
  }
  # Task 1 authenticates serialization formats; Task 4/9 own kind semantics.
  invisible(lapply(seq_along(payload_keys), function(index) {
    .fastkpc_full_cuda_phase3_validate_payload_encoding(
      unlist(paths[payload_keys], use.names = FALSE)[[index]],
      expected_payload_names[[index]]
    )
  }))
  if (!identical(actual_payload_hashes, payload_hashes)) {
    stop("Phase 3 artifact payload hash mismatch", call. = FALSE)
  }
  .fastkpc_full_cuda_phase3_validate_route_file(
    paths$route_config_json, expected_identity$route_config_hash
  )

  required_summary <- c(
    "artifact_schema_version", "artifact_kind", "manifest_sha256",
    "shard_count", "payload_count"
  )
  missing_summary <- setdiff(required_summary, names(summary))
  if (length(missing_summary) > 0L) {
    stop("Phase 3 summary is incomplete: ",
         paste(missing_summary, collapse = ","), call. = FALSE)
  }
  if (!identical(summary$artifact_schema_version, expected_schema) ||
      !identical(summary$artifact_kind, kind) ||
      !.fastkpc_full_cuda_phase3_sha256(summary$manifest_sha256) ||
      !identical(summary$manifest_sha256,
                 fastkpc_full_cuda_census_file_hash(paths$manifest_json)) ||
      !.fastkpc_full_cuda_phase3_bare_number(summary$shard_count, 0) ||
      as.integer(summary$shard_count) != expected_identity$shard_count ||
      !.fastkpc_full_cuda_phase3_bare_number(summary$payload_count, 0) ||
      as.integer(summary$payload_count) != length(expected_payload_names)) {
    stop("Phase 3 summary completion evidence is invalid", call. = FALSE)
  }
  if ("pass" %in% names(summary) &&
      !.fastkpc_full_cuda_phase3_bare_scalar(summary$pass, "logical")) {
    stop("Phase 3 summary pass metadata is malformed", call. = FALSE)
  }
  if (isTRUE(require_full)) {
    if (length(list.files(paths$shards_dir, all.files = TRUE,
                          no.. = TRUE)) == 0L ||
        length(list.files(paths$sessions_dir, all.files = TRUE,
                          no.. = TRUE)) == 0L) {
      stop("full Phase 3 artifact has no shard/session evidence", call. = FALSE)
    }
  }
  computed_contract_pass <- TRUE
  list(
    authenticated = TRUE,
    complete = TRUE,
    pass = computed_contract_pass,
    computed_contract_pass = computed_contract_pass,
    artifact_kind = kind,
    artifact_schema_version = expected_schema,
    manifest = manifest,
    summary = summary,
    payload_file_sha256 = actual_payload_hashes,
    input_identity = expected_identity
  )
}

fastkpc_validate_full_cuda_phase3_artifact <-
  fastkpc_full_cuda_phase3_validate_artifact

fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact <- function(
    output_dir, expected_identity = NULL, require_full = FALSE,
    catalog = NULL, device_id = NULL) {
  fastkpc_full_cuda_phase3_validate_artifact(
    output_dir = output_dir, kind = "oracle_sp",
    expected_identity = expected_identity, require_full = require_full,
    catalog = catalog, device_id = device_id
  )
}

fastkpc_validate_full_cuda_fixed_sp_shadow_artifact <- function(
    output_dir, expected_identity = NULL, require_full = FALSE,
    catalog = NULL, device_id = NULL) {
  fastkpc_full_cuda_phase3_validate_artifact(
    output_dir = output_dir, kind = "full_shadow",
    expected_identity = expected_identity, require_full = require_full,
    catalog = catalog, device_id = device_id
  )
}

.fastkpc_full_cuda_phase3_whole_scalar <- function(
    value, minimum, label) {
  clean <- typeof(value) %in% c("integer", "double") &&
    length(value) == 1L && !is.object(value) &&
    is.null(attributes(value)) && !is.na(value) && is.finite(value) &&
    value == floor(value) && value >= minimum &&
    value <= .Machine$integer.max
  if (!isTRUE(clean)) {
    stop(label, " must be one whole scalar >= ", minimum, call. = FALSE)
  }
  as.integer(value)
}

.fastkpc_full_cuda_phase3_scope <- function(scope) {
  clean <- typeof(scope) == "character" && length(scope) == 1L &&
    !is.object(scope) && is.null(attributes(scope)) && !is.na(scope) &&
    scope %in% c("iteration", "qualification", "full")
  if (!isTRUE(clean)) {
    stop("Phase 3 scope must be iteration, qualification, or full",
         call. = FALSE)
  }
  scope
}

fastkpc_full_cuda_phase3_resolve_shard_count <- function(
    scope, shard_count = NULL) {
  scope <- .fastkpc_full_cuda_phase3_scope(scope)
  environment_name <- "FASTKPC_FULL_CUDA_PHASE3_TEST_SHARD_COUNT"
  environment_value <- Sys.getenv(environment_name, unset = NA_character_)
  environment_set <- !is.na(environment_value)

  if (identical(scope, "full")) {
    if (!is.null(shard_count) || environment_set) {
      stop("full Phase 3 scope rejects every shard-count override",
           call. = FALSE)
    }
    return(64L)
  }
  if (!is.null(shard_count) && environment_set) {
    stop("non-full Phase 3 shard count has conflicting overrides",
         call. = FALSE)
  }
  if (!is.null(shard_count)) {
    return(.fastkpc_full_cuda_phase3_whole_scalar(
      shard_count, 1L, "shard_count"
    ))
  }
  if (!environment_set) return(64L)
  if (!grepl("^[1-9][0-9]*$", environment_value) ||
      nchar(environment_value, type = "bytes") > 10L) {
    stop(environment_name, " must be a canonical positive integer",
         call. = FALSE)
  }
  parsed <- suppressWarnings(as.numeric(environment_value))
  .fastkpc_full_cuda_phase3_whole_scalar(
    parsed, 1L, environment_name
  )
}

.fastkpc_full_cuda_phase3_key_vector <- function(
    value, label, allow_duplicates = FALSE, allow_empty = FALSE,
    allow_empty_list = FALSE) {
  if (isTRUE(allow_empty) && isTRUE(allow_empty_list) &&
      typeof(value) == "list" &&
      !is.object(value) && is.null(attributes(value)) &&
      length(value) == 0L) {
    return(character())
  }
  clean <- typeof(value) == "character" && !is.object(value) &&
    (is.null(attributes(value)) || identical(names(attributes(value)), "names")) &&
    (length(value) > 0L || isTRUE(allow_empty)) && !anyNA(value) &&
    all(grepl("^[0-9a-f]{64}$", value))
  if (!isTRUE(clean) || (!isTRUE(allow_duplicates) && anyDuplicated(value))) {
    stop(label, " must contain canonical lowercase SHA-256 keys",
         call. = FALSE)
  }
  unname(value)
}

fastkpc_full_cuda_phase3_assign_setup_shards <- function(
    setup_keys, shard_count) {
  keys <- .fastkpc_full_cuda_phase3_key_vector(
    setup_keys, "setup_keys", allow_duplicates = TRUE
  )
  shard_count <- .fastkpc_full_cuda_phase3_whole_scalar(
    shard_count, 1L, "shard_count"
  )
  keys <- sort(unique(keys), method = "radix")
  rank <- seq_along(keys)
  data.frame(
    prepared_s_key_sha256 = keys,
    sorted_rank = as.integer(rank),
    shard_id = as.integer((rank - 1L) %% shard_count),
    stringsAsFactors = FALSE,
    row.names = seq_along(keys)
  )
}

fastkpc_full_cuda_phase3_plan_shards <- function(
    setup_keys, target_rows, scope, shard_count = NULL) {
  scope <- .fastkpc_full_cuda_phase3_scope(scope)
  shard_count <- fastkpc_full_cuda_phase3_resolve_shard_count(
    scope, shard_count
  )
  assignments <- fastkpc_full_cuda_phase3_assign_setup_shards(
    setup_keys, shard_count
  )
  required <- c("prepared_s_key_sha256", "residual_key_sha256")
  clean_frame <- is.data.frame(target_rows) &&
    all(required %in% names(target_rows)) &&
    !anyDuplicated(names(target_rows))
  if (!isTRUE(clean_frame)) {
    stop("Phase 3 target rows are malformed", call. = FALSE)
  }
  target_setup_keys <- .fastkpc_full_cuda_phase3_key_vector(
    target_rows$prepared_s_key_sha256,
    "target_rows$prepared_s_key_sha256", allow_duplicates = TRUE,
    allow_empty = TRUE
  )
  target_keys <- .fastkpc_full_cuda_phase3_key_vector(
    target_rows$residual_key_sha256,
    "target_rows$residual_key_sha256", allow_empty = TRUE
  )
  setup_index <- match(
    target_setup_keys, assignments$prepared_s_key_sha256
  )
  if (anyNA(setup_index)) {
    stop("every Phase 3 target must inherit exactly one setup shard",
         call. = FALSE)
  }
  target_rows <- target_rows
  target_rows$prepared_s_key_sha256 <- target_setup_keys
  target_rows$residual_key_sha256 <- target_keys
  target_rows$shard_id <- as.integer(assignments$shard_id[setup_index])
  order_id <- order(
    assignments$sorted_rank[setup_index], target_keys, method = "radix"
  )
  target_rows <- target_rows[order_id, , drop = FALSE]
  rownames(target_rows) <- NULL
  list(
    assignments = assignments,
    target_rows = target_rows,
    shard_count = shard_count
  )
}

.fastkpc_full_cuda_phase3_kind_contract <- function(kind) {
  clean <- typeof(kind) == "character" && length(kind) == 1L &&
    !is.object(kind) && is.null(attributes(kind)) && !is.na(kind) &&
    kind %in% c("oracle_sp", "full_shadow")
  if (!isTRUE(clean)) {
    stop("unknown Phase 3 shard artifact kind", call. = FALSE)
  }
  list(
    kind = kind,
    artifact_schema_version = if (identical(kind, "oracle_sp")) {
      fastkpc_full_cuda_phase3_oracle_schema_version()
    } else {
      fastkpc_full_cuda_phase3_shadow_schema_version()
    }
  )
}

.fastkpc_full_cuda_phase3_test_identity_fields <- function() {
  c(
    "schema_version", "canonical_setup_corpus_hash",
    "canonical_target_corpus_hash", "route_config_hash", "source_commit",
    "cuda_toolkit_version", "cuda_driver_version", "gpu_name", "gpu_uuid",
    "compute_capability_major", "compute_capability_minor",
    "compute_capability", "sm_count", "device_id",
    "cusolver_deterministic_mode_required", "cublas_math_mode_required",
    "cublas_atomics_mode_required", "cublas_user_workspace_required",
    "cublas_workspace_bytes_required",
    "cublas_workspace_min_alignment_required"
  )
}

.fastkpc_full_cuda_phase3_gpu_environment_fields <- function() {
  c(
    "cuda_toolkit_version", "cuda_driver_version", "gpu_name", "gpu_uuid",
    "compute_capability_major", "compute_capability_minor",
    "compute_capability", "sm_count", "device_id",
    "cusolver_deterministic_mode_required", "cublas_math_mode_required",
    "cublas_atomics_mode_required", "cublas_user_workspace_required",
    "cublas_workspace_bytes_required",
    "cublas_workspace_min_alignment_required"
  )
}

.fastkpc_full_cuda_phase3_validate_route_for_shards <- function(
    route_config, scope) {
  clean <- is.list(route_config) && !is.object(route_config) &&
    !is.null(names(route_config)) && !anyDuplicated(names(route_config)) &&
    identical(
      attributes(route_config), list(names = names(route_config))
    ) &&
    length(route_config) > 1L &&
    identical(tail(names(route_config), 1L), "sha256") &&
    .fastkpc_full_cuda_phase3_sha256(route_config$sha256)
  if (!isTRUE(clean)) {
    stop("Phase 3 shard route configuration is malformed", call. = FALSE)
  }
  expected_hash <- .fastkpc_full_cuda_phase3_named_hash(
    route_config[setdiff(names(route_config), "sha256")]
  )
  if (!identical(route_config$sha256, expected_hash)) {
    stop("Phase 3 shard route configuration hash mismatch", call. = FALSE)
  }
  if (identical(scope, "full") &&
      !identical(route_config, fastkpc_full_cuda_phase3_route_config())) {
    stop("full Phase 3 scope requires the canonical route configuration",
         call. = FALSE)
  }
  invisible(TRUE)
}

.fastkpc_full_cuda_phase3_validate_execution_identity <- function(
    identity, route_config, scope, plan) {
  if (!is.list(identity) || is.object(identity) ||
      is.null(names(identity)) || anyDuplicated(names(identity)) ||
      !identical(attributes(identity), list(names = names(identity)))) {
    stop("Phase 3 shard input identity is malformed", call. = FALSE)
  }
  canonical <- identical(
    identity$schema_version, "full-cuda-ci-phase3-input-identity-v1"
  )
  if (canonical) {
    fastkpc_full_cuda_phase3_validate_input_identity(identity)
  } else {
    fields <- .fastkpc_full_cuda_phase3_test_identity_fields()
    if (!identical(names(identity), c(fields, "sha256")) ||
        !identical(
          identity$schema_version,
          "full-cuda-ci-phase3-test-input-identity-v1"
        )) {
      stop("Phase 3 shard test identity schema is malformed",
           call. = FALSE)
    }
    for (field in c(
      "canonical_setup_corpus_hash", "canonical_target_corpus_hash",
      "route_config_hash", "sha256"
    )) {
      if (!.fastkpc_full_cuda_phase3_sha256(identity[[field]])) {
        stop("Phase 3 shard test identity hash is malformed: ", field,
             call. = FALSE)
      }
    }
    if (!.fastkpc_full_cuda_phase3_commit(identity$source_commit)) {
      stop("Phase 3 shard test source commit is malformed", call. = FALSE)
    }
    for (field in c(
      "cuda_toolkit_version", "cuda_driver_version",
      "compute_capability_major", "compute_capability_minor",
      "sm_count", "device_id",
      "cublas_workspace_bytes_required",
      "cublas_workspace_min_alignment_required"
    )) {
      .fastkpc_full_cuda_phase3_whole_scalar(
        identity[[field]], 0L, paste("identity", field)
      )
    }
    for (field in c(
      "gpu_name", "gpu_uuid", "compute_capability",
      "cusolver_deterministic_mode_required", "cublas_math_mode_required",
      "cublas_atomics_mode_required"
    )) {
      .fastkpc_full_cuda_phase3_scalar_text(
        identity[[field]], paste("identity", field)
      )
    }
    if (!.fastkpc_full_cuda_phase3_bare_scalar(
          identity$cublas_user_workspace_required, "logical"
        ) || !grepl("^GPU-[0-9a-f]{32}$", identity$gpu_uuid) ||
        !identical(
          identity$compute_capability,
          paste0(identity$compute_capability_major, ".",
                 identity$compute_capability_minor)
        )) {
      stop("Phase 3 shard test GPU identity is malformed", call. = FALSE)
    }
    expected_hash <- .fastkpc_full_cuda_phase3_named_hash(
      identity[setdiff(names(identity), "sha256")]
    )
    if (!identical(identity$sha256, expected_hash)) {
      stop("Phase 3 shard test identity hash mismatch", call. = FALSE)
    }
  }
  if (identical(scope, "full") && !canonical) {
    stop("full Phase 3 scope requires an authenticated production identity",
         call. = FALSE)
  }
  if (!identical(identity$route_config_hash, route_config$sha256)) {
    stop("Phase 3 shard identity route hash mismatch", call. = FALSE)
  }
  if (identical(scope, "full")) {
    setup_hash <- fastkpc_full_cuda_census_key_set_hash(
      plan$assignments$prepared_s_key_sha256
    )
    target_hash <- fastkpc_full_cuda_census_key_set_hash(sort(
      plan$target_rows$residual_key_sha256, method = "radix"
    ))
    if (!identical(identity$canonical_setup_corpus_hash, setup_hash) ||
        !identical(identity$canonical_target_corpus_hash, target_hash)) {
      stop("full Phase 3 shard corpus identity mismatch", call. = FALSE)
    }
  }
  gpu_fields <- .fastkpc_full_cuda_phase3_gpu_environment_fields()
  gpu_environment <- identity[gpu_fields]
  list(
    input_identity_hash = identity$sha256,
    source_commit = identity$source_commit,
    gpu_environment = gpu_environment,
    gpu_environment_hash = .fastkpc_full_cuda_phase3_named_hash(
      gpu_environment
    )
  )
}

.fastkpc_full_cuda_phase3_session_fields <- function() {
  c(
    "schema_version", "session_id", "input_identity_hash",
    "route_config_hash", "requested_shard_ids", "completed_shard_ids",
    "runtime_context_create_count", "runtime_context_destroy_count",
    "prepared_handle_create_count", "prepared_handle_destroy_count",
    "residual_token_acquire_count", "residual_token_release_count",
    "output_slot_acquire_count", "output_slot_release_count",
    "target_level_stable_sync_count", "status"
  )
}

.fastkpc_full_cuda_phase3_integer_vector <- function(
    value, label, allow_empty = FALSE) {
  if (is.list(value) && length(value) == 0L && isTRUE(allow_empty)) {
    return(integer())
  }
  clean <- typeof(value) %in% c("integer", "double") && !is.object(value) &&
    is.null(attributes(value)) && !anyNA(value) && all(is.finite(value)) &&
    all(value == floor(value)) && all(value >= 0) &&
    all(value <= .Machine$integer.max) &&
    (isTRUE(allow_empty) || length(value) > 0L)
  if (!isTRUE(clean)) {
    stop(label, " must be a bare nonnegative integer vector", call. = FALSE)
  }
  as.integer(value)
}

.fastkpc_full_cuda_phase3_session_id_valid <- function(value) {
  .fastkpc_full_cuda_phase3_bare_scalar(value, "character") &&
    grepl("^[A-Za-z0-9_-]+$", value) && nchar(value, type = "bytes") <= 128L
}

fastkpc_full_cuda_phase3_session_id <- function() {
  timestamp <- format(
    Sys.time(), "%Y%m%dT%H%M%SZ", tz = "UTC", usetz = FALSE
  )
  seed <- paste(
    timestamp, Sys.getpid(), tempfile("phase3-session-seed-"),
    proc.time()[["elapsed"]], sep = "|"
  )
  suffix <- substr(fastkpc_full_cuda_census_hash_utf8(seed), 1L, 16L)
  paste0(timestamp, "_", Sys.getpid(), "_", suffix)
}

fastkpc_full_cuda_phase3_session_identity_hash <- function(session) {
  fields <- .fastkpc_full_cuda_phase3_session_fields()
  if (!.fastkpc_full_cuda_phase3_exact_named_list(session, fields)) {
    stop("Phase 3 session schema is malformed", call. = FALSE)
  }
  requested <- .fastkpc_full_cuda_phase3_integer_vector(
    session$requested_shard_ids, "requested_shard_ids"
  )
  if (anyDuplicated(requested)) {
    stop("requested_shard_ids must be unique", call. = FALSE)
  }
  if (!identical(session$schema_version,
                 "full-cuda-ci-phase3-session-v1") ||
      !.fastkpc_full_cuda_phase3_session_id_valid(session$session_id) ||
      !.fastkpc_full_cuda_phase3_sha256(session$input_identity_hash) ||
      !.fastkpc_full_cuda_phase3_sha256(session$route_config_hash)) {
    stop("Phase 3 session immutable identity is malformed", call. = FALSE)
  }
  .fastkpc_full_cuda_phase3_named_hash(list(
    schema_version = "full-cuda-ci-phase3-session-identity-v1",
    session = list(
      schema_version = session$schema_version,
      session_id = session$session_id,
      input_identity_hash = session$input_identity_hash,
      route_config_hash = session$route_config_hash,
      requested_shard_ids = requested
    )
  ))
}

.fastkpc_full_cuda_phase3_validate_session <- function(
    session, require_complete = FALSE, shard_id = NULL) {
  fields <- .fastkpc_full_cuda_phase3_session_fields()
  if (!.fastkpc_full_cuda_phase3_exact_named_list(session, fields)) {
    stop("Phase 3 session schema is malformed", call. = FALSE)
  }
  fastkpc_full_cuda_phase3_session_identity_hash(session)
  requested <- .fastkpc_full_cuda_phase3_integer_vector(
    session$requested_shard_ids, "requested_shard_ids"
  )
  completed <- .fastkpc_full_cuda_phase3_integer_vector(
    session$completed_shard_ids, "completed_shard_ids", allow_empty = TRUE
  )
  if (anyDuplicated(requested) || anyDuplicated(completed) ||
      any(!completed %in% requested)) {
    stop("Phase 3 session shard-id evidence is malformed", call. = FALSE)
  }
  counter_fields <- c(
    "runtime_context_create_count", "runtime_context_destroy_count",
    "prepared_handle_create_count", "prepared_handle_destroy_count",
    "residual_token_acquire_count", "residual_token_release_count",
    "output_slot_acquire_count", "output_slot_release_count",
    "target_level_stable_sync_count"
  )
  counters <- setNames(vapply(counter_fields, function(field) {
    .fastkpc_full_cuda_phase3_whole_scalar(
      session[[field]], 0L, paste("session", field)
    )
  }, integer(1L)), counter_fields)
  status_clean <- .fastkpc_full_cuda_phase3_bare_scalar(
    session$status, "character"
  ) && session$status %in% c("running", "complete")
  if (!isTRUE(status_clean)) {
    stop("Phase 3 session status is malformed", call. = FALSE)
  }
  if (identical(session$status, "running")) {
    if (counters[["runtime_context_create_count"]] > 1L ||
        counters[["runtime_context_destroy_count"]] >
          counters[["runtime_context_create_count"]]) {
      stop("running Phase 3 session context counters are invalid",
           call. = FALSE)
    }
  } else if (
    counters[["runtime_context_create_count"]] != 1L ||
      counters[["runtime_context_destroy_count"]] != 1L ||
      counters[["prepared_handle_create_count"]] !=
        counters[["prepared_handle_destroy_count"]] ||
      counters[["residual_token_acquire_count"]] !=
        counters[["residual_token_release_count"]] ||
      counters[["output_slot_acquire_count"]] !=
        counters[["output_slot_release_count"]] ||
      counters[["target_level_stable_sync_count"]] != 0L
  ) {
    stop("complete Phase 3 session resource counters are invalid",
         call. = FALSE)
  }
  if (isTRUE(require_complete) && !identical(session$status, "complete")) {
    stop("Phase 3 shard references an incomplete session", call. = FALSE)
  }
  if (!is.null(shard_id)) {
    shard_id <- .fastkpc_full_cuda_phase3_whole_scalar(
      shard_id, 0L, "shard_id"
    )
    if (!shard_id %in% completed) {
      stop("Phase 3 shard is absent from its completed session",
           call. = FALSE)
    }
  }
  invisible(list(
    requested_shard_ids = requested,
    completed_shard_ids = completed,
    counters = counters
  ))
}

.fastkpc_full_cuda_phase3_validate_session_plan_resources <- function(
    session, plan) {
  validated <- .fastkpc_full_cuda_phase3_validate_session(
    session, require_complete = TRUE
  )
  completed <- validated$completed_shard_ids
  if (length(completed) == 0L ||
      any(completed >= plan$shard_count)) {
    stop("complete Phase 3 session has no valid completed shards",
         call. = FALSE)
  }
  descriptors <- lapply(completed, function(shard_id) {
    .fastkpc_full_cuda_phase3_shard_descriptor(plan, shard_id)
  })
  expected_setup_count <- sum(vapply(
    descriptors, `[[`, integer(1L), "setup_count"
  ))
  counters <- validated$counters
  if (counters[["prepared_handle_create_count"]] !=
        expected_setup_count ||
      counters[["prepared_handle_destroy_count"]] !=
        expected_setup_count ||
      counters[["residual_token_acquire_count"]] !=
        expected_setup_count ||
      counters[["residual_token_release_count"]] !=
        expected_setup_count ||
      counters[["output_slot_acquire_count"]] !=
        expected_setup_count ||
      counters[["output_slot_release_count"]] !=
        expected_setup_count ||
      counters[["target_level_stable_sync_count"]] != 0L) {
    stop("complete Phase 3 session counters disagree with planned resources",
         call. = FALSE)
  }
  invisible(TRUE)
}

.fastkpc_full_cuda_phase3_session_path <- function(
    sessions_dir, session_id) {
  if (!.fastkpc_full_cuda_phase3_session_id_valid(session_id)) {
    stop("Phase 3 session id is malformed", call. = FALSE)
  }
  file.path(sessions_dir, paste0("session_", session_id, ".json"))
}

.fastkpc_full_cuda_phase3_atomic_write_session <- function(
    session, sessions_dir) {
  .fastkpc_full_cuda_phase3_validate_session(session)
  dir.create(sessions_dir, recursive = TRUE, showWarnings = FALSE)
  path <- .fastkpc_full_cuda_phase3_session_path(
    sessions_dir, session$session_id
  )
  temporary <- tempfile(
    ".phase3-session-", tmpdir = sessions_dir, fileext = ".tmp"
  )
  on.exit(unlink(temporary, force = TRUE), add = TRUE)
  fastkpc_full_cuda_write_json(session, temporary)
  written <- .fastkpc_full_cuda_phase3_read_json(
    temporary, "Phase 3 temporary session"
  )
  .fastkpc_full_cuda_phase3_validate_session(written)
  if (!file.rename(temporary, path)) {
    stop("failed to atomically rewrite Phase 3 session", call. = FALSE)
  }
  invisible(path)
}

.fastkpc_full_cuda_phase3_shard_payload_fields <- function() {
  c(
    "schema_version", "artifact_kind", "artifact_schema_version",
    "shard_id", "session_id", "session_identity_hash",
    "input_identity_hash", "route_config_hash", "expected_setup_keys",
    "expected_setup_count", "expected_setup_hash", "expected_target_keys",
    "expected_target_count", "expected_target_hash", "source_commit",
    "gpu_environment", "gpu_environment_hash", "payload",
    "payload_semantic_hashes", "payload_semantic_hash"
  )
}

.fastkpc_full_cuda_phase3_shard_summary_fields <- function() {
  c(
    "schema_version", "status", "artifact_kind",
    "artifact_schema_version", "shard_id", "session_id",
    "session_identity_hash", "input_identity_hash", "route_config_hash",
    "expected_setup_keys", "expected_setup_count", "expected_setup_hash",
    "expected_target_keys", "expected_target_count", "expected_target_hash",
    "source_commit", "gpu_environment", "gpu_environment_hash",
    "payload_semantic_hashes", "payload_semantic_hash", "rds_file_sha256"
  )
}

.fastkpc_full_cuda_phase3_exact_named_list <- function(value, fields) {
  typeof(value) == "list" && !is.object(value) &&
    identical(names(value), fields) &&
    identical(attributes(value), list(names = fields))
}

.fastkpc_full_cuda_phase3_payload_semantic_hashes <- function(payload) {
  clean <- is.list(payload) && !is.object(payload) &&
    !is.null(names(payload)) && length(payload) > 0L &&
    !anyNA(names(payload)) && !anyDuplicated(names(payload)) &&
    all(nzchar(names(payload))) &&
    identical(attributes(payload), list(names = names(payload)))
  if (!isTRUE(clean)) {
    stop("Phase 3 shard executor payload must be a nonempty named list",
         call. = FALSE)
  }
  values <- vapply(payload, function(value) {
    if (is.data.frame(value)) {
      return(fastkpc_full_cuda_census_frame_hash(value))
    }
    .fastkpc_full_cuda_phase3_named_hash(list(
      schema_version = "full-cuda-ci-phase3-payload-component-hash-v1",
      value = value
    ))
  }, character(1L))
  unname(values) |> setNames(names(payload))
}

.fastkpc_full_cuda_phase3_payload_semantic_hash <- function(hashes) {
  hashes <- .fastkpc_full_cuda_phase3_named_sha256(
    hashes, "Phase 3 payload semantic hashes"
  )
  .fastkpc_full_cuda_phase3_named_hash(list(
    schema_version = "full-cuda-ci-phase3-payload-semantic-hash-v1",
    payload_semantic_hashes = as.list(hashes)
  ))
}

.fastkpc_full_cuda_phase3_shard_descriptor <- function(plan, shard_id) {
  shard_id <- .fastkpc_full_cuda_phase3_whole_scalar(
    shard_id, 0L, "shard_id"
  )
  if (shard_id >= plan$shard_count) {
    stop("shard_id is outside the Phase 3 shard plan", call. = FALSE)
  }
  setup_keys <- plan$assignments$prepared_s_key_sha256[
    plan$assignments$shard_id == shard_id
  ]
  target_rows <- plan$target_rows[
    plan$target_rows$shard_id == shard_id, , drop = FALSE
  ]
  target_keys <- sort(
    as.character(target_rows$residual_key_sha256), method = "radix"
  )
  list(
    shard_id = shard_id,
    setup_keys = unname(setup_keys),
    setup_count = as.integer(length(setup_keys)),
    setup_hash = fastkpc_full_cuda_census_key_set_hash(setup_keys),
    target_rows = target_rows,
    target_keys = unname(target_keys),
    target_count = as.integer(length(target_keys)),
    target_hash = fastkpc_full_cuda_census_key_set_hash(target_keys)
  )
}

.fastkpc_full_cuda_phase3_shard_paths <- function(shards_dir, shard_id) {
  shard_id <- .fastkpc_full_cuda_phase3_whole_scalar(
    shard_id, 0L, "shard_id"
  )
  list(
    rds = file.path(shards_dir, paste0("shard_", shard_id, ".rds")),
    summary_json = file.path(
      shards_dir, paste0("shard_", shard_id, ".summary.json")
    )
  )
}

.fastkpc_full_cuda_phase3_build_shard_envelope <- function(
    contract, descriptor, session, identity_info, route_config, payload) {
  semantic_hashes <- .fastkpc_full_cuda_phase3_payload_semantic_hashes(
    payload
  )
  list(
    schema_version = "full-cuda-ci-phase3-shard-payload-v1",
    artifact_kind = contract$kind,
    artifact_schema_version = contract$artifact_schema_version,
    shard_id = descriptor$shard_id,
    session_id = session$session_id,
    session_identity_hash =
      fastkpc_full_cuda_phase3_session_identity_hash(session),
    input_identity_hash = identity_info$input_identity_hash,
    route_config_hash = route_config$sha256,
    expected_setup_keys = descriptor$setup_keys,
    expected_setup_count = descriptor$setup_count,
    expected_setup_hash = descriptor$setup_hash,
    expected_target_keys = descriptor$target_keys,
    expected_target_count = descriptor$target_count,
    expected_target_hash = descriptor$target_hash,
    source_commit = identity_info$source_commit,
    gpu_environment = identity_info$gpu_environment,
    gpu_environment_hash = identity_info$gpu_environment_hash,
    payload = payload,
    payload_semantic_hashes = as.list(semantic_hashes),
    payload_semantic_hash =
      .fastkpc_full_cuda_phase3_payload_semantic_hash(semantic_hashes)
  )
}

.fastkpc_full_cuda_phase3_build_shard_summary <- function(
    envelope, rds_file_sha256) {
  list(
    schema_version = "full-cuda-ci-phase3-shard-summary-v1",
    status = "complete",
    artifact_kind = envelope$artifact_kind,
    artifact_schema_version = envelope$artifact_schema_version,
    shard_id = envelope$shard_id,
    session_id = envelope$session_id,
    session_identity_hash = envelope$session_identity_hash,
    input_identity_hash = envelope$input_identity_hash,
    route_config_hash = envelope$route_config_hash,
    expected_setup_keys = envelope$expected_setup_keys,
    expected_setup_count = envelope$expected_setup_count,
    expected_setup_hash = envelope$expected_setup_hash,
    expected_target_keys = envelope$expected_target_keys,
    expected_target_count = envelope$expected_target_count,
    expected_target_hash = envelope$expected_target_hash,
    source_commit = envelope$source_commit,
    gpu_environment = envelope$gpu_environment,
    gpu_environment_hash = envelope$gpu_environment_hash,
    payload_semantic_hashes = envelope$payload_semantic_hashes,
    payload_semantic_hash = envelope$payload_semantic_hash,
    rds_file_sha256 = rds_file_sha256
  )
}

.fastkpc_full_cuda_phase3_gpu_environment_equal <- function(
    actual, expected) {
  fields <- .fastkpc_full_cuda_phase3_gpu_environment_fields()
  .fastkpc_full_cuda_phase3_exact_named_list(actual, fields) &&
    .fastkpc_full_cuda_phase3_exact_named_list(expected, fields) &&
    all(vapply(fields, function(field) {
      .fastkpc_full_cuda_phase3_identity_value_equal(
        actual[[field]], expected[[field]]
      )
    }, logical(1L)))
}

.fastkpc_full_cuda_phase3_validate_shard_pair <- function(
    envelope, summary, contract, descriptor, identity_info, route_config,
    rds_path = NULL) {
  if (!.fastkpc_full_cuda_phase3_exact_named_list(
        envelope, .fastkpc_full_cuda_phase3_shard_payload_fields()
      ) || !.fastkpc_full_cuda_phase3_exact_named_list(
        summary, .fastkpc_full_cuda_phase3_shard_summary_fields()
      )) {
    stop("Phase 3 shard payload/summary schema mismatch", call. = FALSE)
  }
  scalar_expected <- list(
    payload_schema = "full-cuda-ci-phase3-shard-payload-v1",
    summary_schema = "full-cuda-ci-phase3-shard-summary-v1",
    status = "complete",
    artifact_kind = contract$kind,
    artifact_schema_version = contract$artifact_schema_version,
    shard_id = descriptor$shard_id,
    input_identity_hash = identity_info$input_identity_hash,
    route_config_hash = route_config$sha256,
    expected_setup_count = descriptor$setup_count,
    expected_setup_hash = descriptor$setup_hash,
    expected_target_count = descriptor$target_count,
    expected_target_hash = descriptor$target_hash,
    source_commit = identity_info$source_commit,
    gpu_environment_hash = identity_info$gpu_environment_hash
  )
  scalar_clean <- identical(envelope$schema_version,
                            scalar_expected$payload_schema) &&
    identical(summary$schema_version, scalar_expected$summary_schema) &&
    identical(summary$status, scalar_expected$status) &&
    identical(envelope$artifact_kind, scalar_expected$artifact_kind) &&
    identical(summary$artifact_kind, scalar_expected$artifact_kind) &&
    identical(envelope$artifact_schema_version,
              scalar_expected$artifact_schema_version) &&
    identical(summary$artifact_schema_version,
              scalar_expected$artifact_schema_version) &&
    .fastkpc_full_cuda_phase3_identity_value_equal(
      envelope$shard_id, scalar_expected$shard_id
    ) && .fastkpc_full_cuda_phase3_identity_value_equal(
      summary$shard_id, scalar_expected$shard_id
    ) &&
    identical(envelope$input_identity_hash,
              scalar_expected$input_identity_hash) &&
    identical(summary$input_identity_hash,
              scalar_expected$input_identity_hash) &&
    identical(envelope$route_config_hash,
              scalar_expected$route_config_hash) &&
    identical(summary$route_config_hash,
              scalar_expected$route_config_hash) &&
    .fastkpc_full_cuda_phase3_identity_value_equal(
      envelope$expected_setup_count, scalar_expected$expected_setup_count
    ) && .fastkpc_full_cuda_phase3_identity_value_equal(
      summary$expected_setup_count, scalar_expected$expected_setup_count
    ) && identical(envelope$expected_setup_hash,
                   scalar_expected$expected_setup_hash) &&
    identical(summary$expected_setup_hash,
              scalar_expected$expected_setup_hash) &&
    .fastkpc_full_cuda_phase3_identity_value_equal(
      envelope$expected_target_count, scalar_expected$expected_target_count
    ) && .fastkpc_full_cuda_phase3_identity_value_equal(
      summary$expected_target_count, scalar_expected$expected_target_count
    ) && identical(envelope$expected_target_hash,
                   scalar_expected$expected_target_hash) &&
    identical(summary$expected_target_hash,
              scalar_expected$expected_target_hash) &&
    identical(envelope$source_commit, scalar_expected$source_commit) &&
    identical(summary$source_commit, scalar_expected$source_commit) &&
    identical(envelope$gpu_environment_hash,
              scalar_expected$gpu_environment_hash) &&
    identical(summary$gpu_environment_hash,
              scalar_expected$gpu_environment_hash)
  if (!isTRUE(scalar_clean)) {
    stop("Phase 3 shard immutable identity mismatch", call. = FALSE)
  }
  setup_keys <- .fastkpc_full_cuda_phase3_key_vector(
    envelope$expected_setup_keys, "shard expected setup keys",
    allow_empty = TRUE, allow_empty_list = TRUE
  )
  summary_setup_keys <- .fastkpc_full_cuda_phase3_key_vector(
    summary$expected_setup_keys, "summary expected setup keys",
    allow_empty = TRUE, allow_empty_list = TRUE
  )
  target_keys <- .fastkpc_full_cuda_phase3_key_vector(
    envelope$expected_target_keys, "shard expected target keys",
    allow_empty = TRUE, allow_empty_list = TRUE
  )
  summary_target_keys <- .fastkpc_full_cuda_phase3_key_vector(
    summary$expected_target_keys, "summary expected target keys",
    allow_empty = TRUE, allow_empty_list = TRUE
  )
  if (!identical(setup_keys, descriptor$setup_keys) ||
      !identical(summary_setup_keys, descriptor$setup_keys) ||
      !identical(target_keys, descriptor$target_keys) ||
      !identical(summary_target_keys, descriptor$target_keys)) {
    stop("Phase 3 shard expected key set mismatch", call. = FALSE)
  }
  session_fields <- c(
    "session_id", "session_identity_hash", "input_identity_hash",
    "route_config_hash"
  )
  if (!.fastkpc_full_cuda_phase3_session_id_valid(envelope$session_id) ||
      !.fastkpc_full_cuda_phase3_sha256(
        envelope$session_identity_hash
      ) || any(!vapply(session_fields, function(field) {
        identical(envelope[[field]], summary[[field]])
      }, logical(1L)))) {
    stop("Phase 3 shard session identity mismatch", call. = FALSE)
  }
  if (!.fastkpc_full_cuda_phase3_gpu_environment_equal(
        envelope$gpu_environment, identity_info$gpu_environment
      ) || !.fastkpc_full_cuda_phase3_gpu_environment_equal(
        summary$gpu_environment, identity_info$gpu_environment
      )) {
    stop("Phase 3 shard GPU environment mismatch", call. = FALSE)
  }
  actual_hashes <- .fastkpc_full_cuda_phase3_payload_semantic_hashes(
    envelope$payload
  )
  envelope_hashes <- .fastkpc_full_cuda_phase3_named_sha256(
    envelope$payload_semantic_hashes,
    "Phase 3 shard payload semantic hashes"
  )
  summary_hashes <- .fastkpc_full_cuda_phase3_named_sha256(
    summary$payload_semantic_hashes,
    "Phase 3 shard summary semantic hashes"
  )
  actual_hash <- .fastkpc_full_cuda_phase3_payload_semantic_hash(
    actual_hashes
  )
  if (!identical(envelope_hashes, actual_hashes) ||
      !identical(summary_hashes, actual_hashes) ||
      !identical(envelope$payload_semantic_hash, actual_hash) ||
      !identical(summary$payload_semantic_hash, actual_hash)) {
    stop("Phase 3 shard payload semantic hash mismatch", call. = FALSE)
  }
  if (identical(contract$kind, "oracle_sp") &&
      .fastkpc_full_cuda_phase3_is_oracle_payload(envelope$payload)) {
    .fastkpc_full_cuda_phase3_validate_oracle_payload(envelope$payload)
  }
  if (!.fastkpc_full_cuda_phase3_sha256(summary$rds_file_sha256)) {
    stop("Phase 3 shard RDS file hash is malformed", call. = FALSE)
  }
  if (!is.null(rds_path)) {
    if (!file.exists(rds_path) || dir.exists(rds_path) ||
        !identical(
          summary$rds_file_sha256,
          fastkpc_full_cuda_census_file_hash(rds_path)
        )) {
      stop("Phase 3 shard RDS byte hash mismatch", call. = FALSE)
    }
  }
  invisible(TRUE)
}

.fastkpc_full_cuda_phase3_atomic_write_shard <- function(
    envelope, contract, descriptor, identity_info, route_config, paths) {
  dir.create(dirname(paths$rds), recursive = TRUE, showWarnings = FALSE)
  if (file.exists(paths$rds) || file.exists(paths$summary_json)) {
    stop("Phase 3 shard final path exists before publication",
         call. = FALSE)
  }
  rds_temporary <- tempfile(
    ".phase3-shard-rds-", tmpdir = dirname(paths$rds), fileext = ".tmp"
  )
  summary_temporary <- tempfile(
    ".phase3-shard-summary-", tmpdir = dirname(paths$rds), fileext = ".tmp"
  )
  rds_published <- FALSE
  summary_published <- FALSE
  on.exit({
    unlink(c(rds_temporary, summary_temporary), force = TRUE)
    if (rds_published && !summary_published) {
      unlink(paths$rds, force = TRUE)
    }
  }, add = TRUE)

  saveRDS(envelope, rds_temporary, version = 2)
  summary <- .fastkpc_full_cuda_phase3_build_shard_summary(
    envelope,
    fastkpc_full_cuda_census_file_hash(rds_temporary)
  )
  .fastkpc_full_cuda_phase3_validate_shard_pair(
    envelope = envelope,
    summary = summary,
    contract = contract,
    descriptor = descriptor,
    identity_info = identity_info,
    route_config = route_config,
    rds_path = rds_temporary
  )
  fastkpc_full_cuda_write_json(summary, summary_temporary)
  disk_envelope <- tryCatch(
    readRDS(rds_temporary),
    error = function(error) {
      stop("Phase 3 temporary shard RDS is unreadable", call. = FALSE)
    }
  )
  disk_summary <- .fastkpc_full_cuda_phase3_read_json(
    summary_temporary, "Phase 3 temporary shard summary"
  )
  .fastkpc_full_cuda_phase3_validate_shard_pair(
    envelope = disk_envelope,
    summary = disk_summary,
    contract = contract,
    descriptor = descriptor,
    identity_info = identity_info,
    route_config = route_config,
    rds_path = rds_temporary
  )
  if (!file.rename(rds_temporary, paths$rds)) {
    stop("failed to publish Phase 3 shard RDS", call. = FALSE)
  }
  rds_published <- TRUE
  if (!file.rename(summary_temporary, paths$summary_json)) {
    stop("failed to publish Phase 3 shard summary", call. = FALSE)
  }
  summary_published <- TRUE
  rds_published <- FALSE
  invisible(list(envelope = envelope, summary = summary))
}

.fastkpc_full_cuda_phase3_validate_shard_session_binding <- function(
    envelope, summary, session) {
  session_hash <- fastkpc_full_cuda_phase3_session_identity_hash(session)
  ids_bound <- .fastkpc_full_cuda_phase3_session_id_valid(
    envelope$session_id
  ) && .fastkpc_full_cuda_phase3_session_id_valid(
    summary$session_id
  ) && .fastkpc_full_cuda_phase3_session_id_valid(
    session$session_id
  ) && identical(envelope$session_id, summary$session_id) &&
    identical(envelope$session_id, session$session_id)
  hashes_bound <- .fastkpc_full_cuda_phase3_sha256(
    envelope$session_identity_hash
  ) && .fastkpc_full_cuda_phase3_sha256(
    summary$session_identity_hash
  ) && identical(
    envelope$session_identity_hash, summary$session_identity_hash
  ) && identical(envelope$session_identity_hash, session_hash)
  if (!isTRUE(ids_bound) || !isTRUE(hashes_bound)) {
    stop("Phase 3 shard/session id and identity binding mismatch",
         call. = FALSE)
  }
  invisible(session_hash)
}

.fastkpc_full_cuda_phase3_read_reusable_shard <- function(
    paths, sessions_dir, contract, descriptor, plan, identity_info,
    route_config) {
  if (!file.exists(paths$rds) || !file.exists(paths$summary_json) ||
      dir.exists(paths$rds) || dir.exists(paths$summary_json)) {
    stop("Phase 3 shard pair is missing", call. = FALSE)
  }
  summary <- .fastkpc_full_cuda_phase3_read_json(
    paths$summary_json, "Phase 3 shard summary"
  )
  if (!.fastkpc_full_cuda_phase3_sha256(summary$rds_file_sha256) ||
      !identical(
        summary$rds_file_sha256,
        fastkpc_full_cuda_census_file_hash(paths$rds)
      )) {
    stop("Phase 3 shard RDS byte hash mismatch", call. = FALSE)
  }
  envelope <- tryCatch(
    readRDS(paths$rds),
    error = function(error) {
      stop("Phase 3 shard RDS is unreadable", call. = FALSE)
    }
  )
  .fastkpc_full_cuda_phase3_validate_shard_pair(
    envelope = envelope,
    summary = summary,
    contract = contract,
    descriptor = descriptor,
    identity_info = identity_info,
    route_config = route_config,
    rds_path = paths$rds
  )
  session_path <- .fastkpc_full_cuda_phase3_session_path(
    sessions_dir, summary$session_id
  )
  session <- .fastkpc_full_cuda_phase3_read_json(
    session_path, "Phase 3 shard session"
  )
  .fastkpc_full_cuda_phase3_validate_session(
    session, require_complete = TRUE, shard_id = descriptor$shard_id
  )
  .fastkpc_full_cuda_phase3_validate_session_plan_resources(
    session, plan
  )
  .fastkpc_full_cuda_phase3_validate_shard_session_binding(
    envelope, summary, session
  )
  if (!identical(
        session$input_identity_hash, identity_info$input_identity_hash
      ) || !identical(
        session$route_config_hash, route_config$sha256
      )) {
    stop("Phase 3 shard session authentication mismatch", call. = FALSE)
  }
  list(envelope = envelope, summary = summary, session = session)
}

.fastkpc_full_cuda_phase3_scan_shards <- function(
    shards_dir, shard_count, require_complete = FALSE) {
  expected_ids <- as.integer(seq.int(0L, shard_count - 1L))
  entries <- if (dir.exists(shards_dir)) {
    list.files(shards_dir, all.files = TRUE, no.. = TRUE)
  } else {
    character()
  }
  entries <- entries[!startsWith(entries, ".")]
  rds_entries <- entries[grepl("^shard_[0-9]+\\.rds$", entries)]
  summary_entries <- entries[grepl(
    "^shard_[0-9]+\\.summary\\.json$", entries
  )]
  unexpected_entries <- setdiff(
    entries, c(rds_entries, summary_entries)
  )
  if (length(unexpected_entries) > 0L) {
    stop("unexpected Phase 3 shard artifact entry: ",
         unexpected_entries[[1L]], call. = FALSE)
  }
  parse_ids <- function(values, suffix) {
    suppressWarnings(as.integer(sub(
      paste0("^shard_([0-9]+)", suffix, "$"), "\\1", values
    )))
  }
  rds_ids <- parse_ids(rds_entries, "\\.rds")
  summary_ids <- parse_ids(summary_entries, "\\.summary\\.json")
  if (anyNA(rds_ids) || anyNA(summary_ids) ||
      any(!rds_ids %in% expected_ids) ||
      any(!summary_ids %in% expected_ids)) {
    stop("unexpected Phase 3 shard id", call. = FALSE)
  }
  canonical_rds_entries <- sprintf("shard_%d.rds", rds_ids)
  canonical_summary_entries <- sprintf(
    "shard_%d.summary.json", summary_ids
  )
  if (!identical(rds_entries, canonical_rds_entries) ||
      !identical(summary_entries, canonical_summary_entries)) {
    stop("noncanonical Phase 3 shard filename", call. = FALSE)
  }
  if (anyDuplicated(rds_ids) || anyDuplicated(summary_ids)) {
    stop("duplicate Phase 3 shard id", call. = FALSE)
  }
  rds_paths <- setNames(
    file.path(shards_dir, rds_entries), as.character(rds_ids)
  )
  summary_paths <- setNames(
    file.path(shards_dir, summary_entries), as.character(summary_ids)
  )
  if (isTRUE(require_complete)) {
    if (!identical(sort(rds_ids), expected_ids) ||
        !identical(sort(summary_ids), expected_ids)) {
      stop("missing Phase 3 shard pair", call. = FALSE)
    }
  }
  list(
    expected_ids = expected_ids,
    rds_paths = rds_paths,
    summary_paths = summary_paths
  )
}

.fastkpc_full_cuda_phase3_oracle_fallback_rows <- function(
    target_parity, resource_metrics) {
  counts <- c(
    CPU = as.integer(sum(resource_metrics$cpu_fallback_count)),
    UNKNOWN = as.integer(sum(resource_metrics$unknown_fallback_count)),
    APPROXIMATE = as.integer(sum(target_parity$approximate_backend))
  )
  data.frame(
    fallback_type = names(counts), count = unname(counts),
    stringsAsFactors = FALSE, row.names = NULL
  )
}

.fastkpc_full_cuda_phase3_oracle_failure_rows <- function(target_parity) {
  required <- c(
    "prepared_s_key_sha256", "residual_key_sha256", "solver_status",
    "output_all_finite", "fitted_max_abs_diff", "fitted_relative_l2",
    "residual_max_abs_diff", "residual_relative_l2", "error_code",
    "error_message_sha256"
  )
  if (!is.data.frame(target_parity) ||
      !all(required %in% names(target_parity))) {
    stop("Phase 3 oracle target failure inputs are malformed",
         call. = FALSE)
  }
  failed <- !startsWith(target_parity$solver_status, "OK_") |
    !target_parity$output_all_finite |
    !is.finite(target_parity$fitted_max_abs_diff) |
    !is.finite(target_parity$fitted_relative_l2) |
    !is.finite(target_parity$residual_max_abs_diff) |
    !is.finite(target_parity$residual_relative_l2) |
    target_parity$fitted_max_abs_diff >= 1e-7 |
    target_parity$fitted_relative_l2 >= 1e-7 |
    target_parity$residual_max_abs_diff >= 1e-7 |
    target_parity$residual_relative_l2 >= 1e-7
  data.frame(
    prepared_s_key_sha256 = target_parity$prepared_s_key_sha256[failed],
    residual_key_sha256 = target_parity$residual_key_sha256[failed],
    solver_status = target_parity$solver_status[failed],
    error_code = target_parity$error_code[failed],
    error_message_sha256 = target_parity$error_message_sha256[failed],
    stringsAsFactors = FALSE, row.names = NULL
  )
}

.fastkpc_full_cuda_phase3_oracle_target_level_stable_sync <- function(
    setup_results, resource_metrics) {
  checkpoint_fields <- c(
    "qr_checkpoint_record_count", "qr_checkpoint_wait_count",
    "svd_checkpoint_record_count", "svd_checkpoint_wait_count"
  )
  clean <- nrow(setup_results) == nrow(resource_metrics) &&
    all(checkpoint_fields %in% names(resource_metrics)) &&
    all(vapply(checkpoint_fields, function(field) {
      value <- resource_metrics[[field]]
      typeof(value) == "integer" && !anyNA(value) && all(value >= 0L)
    }, logical(1L)))
  if (!isTRUE(clean)) {
    stop("Phase 3 oracle stable checkpoint rows are malformed", call. = FALSE)
  }
  qr_affected <- as.integer(setup_results$planned_qr_target_count > 0L)
  svd_affected <- as.integer(setup_results$executed_svd_target_count > 0L)
  per_setup <- pmax(
    resource_metrics$qr_checkpoint_wait_count - qr_affected, 0L
  ) + pmax(
    resource_metrics$svd_checkpoint_wait_count - svd_affected, 0L
  )
  count <- sum(as.double(per_setup))
  if (!is.finite(count) || count > .Machine$integer.max) {
    stop("Phase 3 oracle target-level stable sync count overflowed",
         call. = FALSE)
  }
  as.integer(count)
}

fastkpc_full_cuda_phase3_summarize_oracle_rows <- function(
    setup_results, target_parity, resource_metrics, stage_timing,
    fallbacks, failures) {
  required_helpers <- c(
    "fastkpc_full_cuda_fixed_sp_validate_oracle_frame",
    "fastkpc_full_cuda_census_frame_hash"
  )
  missing_helpers <- required_helpers[!vapply(
    required_helpers, exists, logical(1L), mode = "function", inherits = TRUE
  )]
  if (length(missing_helpers) > 0L) {
    stop("Phase 3 oracle summary helpers are unavailable", call. = FALSE)
  }
  setup_results <- fastkpc_full_cuda_fixed_sp_validate_oracle_frame(
    setup_results, "setup_results"
  )
  target_parity <- fastkpc_full_cuda_fixed_sp_validate_oracle_frame(
    target_parity, "target_parity"
  )
  resource_metrics <- fastkpc_full_cuda_fixed_sp_validate_oracle_frame(
    resource_metrics, "resource_metrics"
  )
  stage_timing <- fastkpc_full_cuda_fixed_sp_validate_oracle_frame(
    stage_timing, "stage_timing"
  )
  fallback_clean <- is.data.frame(fallbacks) && identical(
    names(fallbacks), c("fallback_type", "count")
  ) && typeof(fallbacks$fallback_type) == "character" &&
    typeof(fallbacks$count) == "integer" && !anyNA(fallbacks) &&
    all(fallbacks$fallback_type %in% c("CPU", "UNKNOWN", "APPROXIMATE")) &&
    all(fallbacks$count >= 0L)
  failure_clean <- is.data.frame(failures) && identical(
    names(failures), c(
      "prepared_s_key_sha256", "residual_key_sha256", "solver_status",
      "error_code", "error_message_sha256"
    )
  ) && all(vapply(failures, function(value) {
    typeof(value) == "character" && !is.object(value) &&
      is.null(attributes(value)) && !anyNA(value)
  }, logical(1L)))
  if (!isTRUE(fallback_clean) || !isTRUE(failure_clean)) {
    stop("Phase 3 oracle fallback/failure rows are malformed",
         call. = FALSE)
  }

  setup_count <- nrow(setup_results)
  target_count <- nrow(target_parity)
  setup_keys <- setup_results$prepared_s_key_sha256
  target_keys <- target_parity$residual_key_sha256
  expected_target_order <- if (target_count == 0L) integer() else order(
    match(target_parity$prepared_s_key_sha256, setup_keys), target_keys,
    method = "radix"
  )
  order_clean <- !anyDuplicated(setup_keys) && !anyDuplicated(target_keys) &&
    identical(setup_keys, sort(setup_keys, method = "radix")) &&
    identical(expected_target_order, seq_len(target_count)) &&
    identical(resource_metrics$prepared_s_key_sha256, setup_keys) &&
    all(target_parity$prepared_s_key_sha256 %in% setup_keys) &&
    all(stage_timing$prepared_s_key_sha256 %in% setup_keys)
  setup_target_counts <- if (setup_count == 0L) integer() else vapply(
    setup_keys, function(key) sum(
      target_parity$prepared_s_key_sha256 == key
    ), integer(1L), USE.NAMES = FALSE
  )
  if (!isTRUE(order_clean) ||
      !identical(setup_results$target_count, setup_target_counts) ||
      nrow(resource_metrics) != setup_count ||
      nrow(stage_timing) != 6L * setup_count) {
    stop("Phase 3 oracle setup/target/resource row identity mismatch",
         call. = FALSE)
  }

  planned_cholesky <- target_parity$planned_route == "CHOLESKY_BATCHED"
  planned_qr <- target_parity$planned_route == "AUGMENTED_QR"
  planned_svd <- target_parity$planned_route == "AUGMENTED_SVD"
  executed_cholesky <-
    target_parity$executed_route == "CHOLESKY_BATCHED"
  executed_qr <- target_parity$executed_route == "AUGMENTED_QR"
  executed_svd <- target_parity$executed_route == "AUGMENTED_SVD"
  cholesky_to_svd <- planned_cholesky & executed_svd
  qr_to_svd <- planned_qr & executed_svd
  stable_reroute <- target_parity$planned_route != target_parity$executed_route
  route_counts <- c(
    planned_cholesky_target_count = sum(planned_cholesky),
    planned_qr_target_count = sum(planned_qr),
    planned_svd_target_count = sum(planned_svd),
    executed_cholesky_target_count = sum(executed_cholesky),
    executed_qr_target_count = sum(executed_qr),
    executed_svd_target_count = sum(executed_svd),
    cholesky_to_svd_count = sum(cholesky_to_svd),
    qr_to_svd_count = sum(qr_to_svd),
    stable_reroute_count = sum(stable_reroute)
  )
  route_counts <- as.integer(route_counts)
  names(route_counts) <- c(
    "planned_cholesky_target_count", "planned_qr_target_count",
    "planned_svd_target_count", "executed_cholesky_target_count",
    "executed_qr_target_count", "executed_svd_target_count",
    "cholesky_to_svd_count", "qr_to_svd_count", "stable_reroute_count"
  )
  route_conserved <-
    route_counts[["executed_cholesky_target_count"]] ==
      route_counts[["planned_cholesky_target_count"]] -
        route_counts[["cholesky_to_svd_count"]] &&
    route_counts[["executed_qr_target_count"]] ==
      route_counts[["planned_qr_target_count"]] -
        route_counts[["qr_to_svd_count"]] &&
    route_counts[["executed_svd_target_count"]] ==
      route_counts[["planned_svd_target_count"]] +
        route_counts[["cholesky_to_svd_count"]] +
        route_counts[["qr_to_svd_count"]] &&
    route_counts[["stable_reroute_count"]] ==
      route_counts[["cholesky_to_svd_count"]] +
        route_counts[["qr_to_svd_count"]]
  reroute_rows_exact <- identical(
    target_parity$cholesky_to_svd_count, as.integer(cholesky_to_svd)
  ) && identical(
    target_parity$qr_to_svd_count, as.integer(qr_to_svd)
  ) && identical(
    target_parity$stable_reroute_count, as.integer(stable_reroute)
  )
  expected_reroute_reason <- rep("", target_count)
  expected_reroute_reason[cholesky_to_svd] <-
    "CHOLESKY_NON_POSITIVE_PIVOT"
  expected_reroute_reason[qr_to_svd] <- "QR_RANK_GUARD_REJECTED"
  if (!isTRUE(route_conserved) || !isTRUE(reroute_rows_exact) ||
      !identical(target_parity$reroute_reason, unname(expected_reroute_reason))) {
    stop("Phase 3 oracle route conservation failed", call. = FALSE)
  }

  setup_route_exact <- all(vapply(seq_along(setup_keys), function(index) {
    selected <- target_parity$prepared_s_key_sha256 == setup_keys[[index]]
    all(vapply(names(route_counts), function(field) {
      expected <- switch(
        field,
        planned_cholesky_target_count = sum(planned_cholesky[selected]),
        planned_qr_target_count = sum(planned_qr[selected]),
        planned_svd_target_count = sum(planned_svd[selected]),
        executed_cholesky_target_count = sum(executed_cholesky[selected]),
        executed_qr_target_count = sum(executed_qr[selected]),
        executed_svd_target_count = sum(executed_svd[selected]),
        cholesky_to_svd_count = sum(cholesky_to_svd[selected]),
        qr_to_svd_count = sum(qr_to_svd[selected]),
        stable_reroute_count = sum(stable_reroute[selected])
      )
      setup_results[[field]][[index]] == expected &&
        resource_metrics[[field]][[index]] == expected
    }, logical(1L)))
  }, logical(1L)))
  if (!isTRUE(setup_route_exact)) {
    stop("Phase 3 oracle setup route summaries are not row-derived",
         call. = FALSE)
  }

  recomputed_fallbacks <-
    .fastkpc_full_cuda_phase3_oracle_fallback_rows(
      target_parity, resource_metrics
    )
  supplied_fallback_counts <- vapply(
    recomputed_fallbacks$fallback_type, function(type) {
      as.integer(sum(fallbacks$count[fallbacks$fallback_type == type]))
    }, integer(1L)
  )
  if (!identical(unname(supplied_fallback_counts),
                 recomputed_fallbacks$count)) {
    stop("Phase 3 oracle fallback summary is not row-derived", call. = FALSE)
  }
  recomputed_failures <-
    .fastkpc_full_cuda_phase3_oracle_failure_rows(target_parity)
  if (!identical(failures, recomputed_failures)) {
    stop("Phase 3 oracle failure summary is not row-derived", call. = FALSE)
  }

  non_ok <- as.integer(sum(!startsWith(target_parity$solver_status, "OK_")))
  nonfinite <- as.integer(sum(!target_parity$output_all_finite))
  max_or_zero <- function(value) {
    if (length(value) == 0L) 0 else as.double(max(value))
  }
  resource_sum <- function(field) as.integer(sum(resource_metrics[[field]]))
  target_level_stable_sync_count <-
    .fastkpc_full_cuda_phase3_oracle_target_level_stable_sync(
      setup_results, resource_metrics
    )
  data.frame(
    setup_count = as.integer(setup_count), target_count = as.integer(target_count),
    non_ok_solver_status_count = non_ok,
    nonfinite_output_count = nonfinite,
    planned_cholesky_target_count =
      route_counts[["planned_cholesky_target_count"]],
    planned_qr_target_count = route_counts[["planned_qr_target_count"]],
    planned_svd_target_count = route_counts[["planned_svd_target_count"]],
    executed_cholesky_target_count =
      route_counts[["executed_cholesky_target_count"]],
    executed_qr_target_count = route_counts[["executed_qr_target_count"]],
    executed_svd_target_count = route_counts[["executed_svd_target_count"]],
    cholesky_to_svd_count = route_counts[["cholesky_to_svd_count"]],
    qr_to_svd_count = route_counts[["qr_to_svd_count"]],
    stable_reroute_count = route_counts[["stable_reroute_count"]],
    route_conservation_exact = isTRUE(route_conserved),
    cpu_fallback_count = recomputed_fallbacks$count[[1L]],
    unknown_fallback_count = recomputed_fallbacks$count[[2L]],
    approximate_backend_count = recomputed_fallbacks$count[[3L]],
    failure_count = as.integer(nrow(failures)),
    prepared_handle_create_count = resource_sum(
      "prepared_handle_create_count"
    ),
    prepared_handle_destroy_count = resource_sum(
      "prepared_handle_destroy_count"
    ),
    residual_token_acquire_count = resource_sum(
      "residual_token_acquire_count"
    ),
    residual_token_release_count = resource_sum(
      "residual_token_release_count"
    ),
    output_slot_acquire_count = resource_sum("output_slot_acquire_count"),
    output_slot_release_count = resource_sum("output_slot_release_count"),
    setup_h2d_upload_count = resource_sum("setup_h2d_upload_count"),
    per_target_allocation_count_after_warmup = resource_sum(
      "per_target_allocation_count_after_warmup"
    ),
    per_target_handle_create_count = resource_sum(
      "per_target_handle_create_count"
    ),
    implicit_residual_d2h_count = resource_sum(
      "implicit_residual_d2h_count"
    ),
    cuda_device_synchronize_count = resource_sum(
      "cuda_device_synchronize_count"
    ),
    target_level_stable_sync_count = target_level_stable_sync_count,
    qr_checkpoint_record_count = resource_sum("qr_checkpoint_record_count"),
    qr_checkpoint_wait_count = resource_sum("qr_checkpoint_wait_count"),
    svd_checkpoint_record_count = resource_sum("svd_checkpoint_record_count"),
    svd_checkpoint_wait_count = resource_sum("svd_checkpoint_wait_count"),
    output_slot_live_count = as.integer(sum(
      resource_metrics$output_slot_leased_after_release
    )),
    shadow_d2h_bytes = as.double(sum(resource_metrics$shadow_d2h_bytes)),
    max_coefficient_abs_diff = max_or_zero(
      target_parity$coefficient_max_abs_diff
    ),
    max_coefficient_relative_l2 = max_or_zero(
      target_parity$coefficient_relative_l2
    ),
    max_fitted_abs_diff = max_or_zero(target_parity$fitted_max_abs_diff),
    max_fitted_relative_l2 = max_or_zero(target_parity$fitted_relative_l2),
    max_residual_abs_diff = max_or_zero(target_parity$residual_max_abs_diff),
    max_residual_relative_l2 = max_or_zero(
      target_parity$residual_relative_l2
    ),
    max_rss_abs_diff = max_or_zero(target_parity$rss_max_abs_diff),
    max_rss_relative_l2 = max_or_zero(target_parity$rss_relative_l2),
    max_rhs_abs_diff = max_or_zero(target_parity$rhs_max_abs_diff),
    max_rhs_relative_l2 = max_or_zero(target_parity$rhs_relative_l2),
    setup_rows_sha256 = fastkpc_full_cuda_census_frame_hash(setup_results),
    target_rows_sha256 = fastkpc_full_cuda_census_frame_hash(target_parity),
    resource_rows_sha256 = fastkpc_full_cuda_census_frame_hash(
      resource_metrics
    ),
    stage_timing_rows_sha256 = fastkpc_full_cuda_census_frame_hash(
      stage_timing
    ),
    fallback_rows_sha256 = fastkpc_full_cuda_census_frame_hash(fallbacks),
    failure_rows_sha256 = fastkpc_full_cuda_census_frame_hash(failures),
    summary_recomputed = TRUE,
    stringsAsFactors = FALSE
  )
}

.fastkpc_full_cuda_phase3_is_oracle_payload <- function(payload) {
  .fastkpc_full_cuda_phase3_exact_named_list(
    payload,
    c(
      "setup_results", "target_parity", "resource_metrics", "stage_timing",
      "fallbacks", "failures", "summary"
    )
  )
}

.fastkpc_full_cuda_phase3_validate_oracle_payload <- function(payload) {
  if (!.fastkpc_full_cuda_phase3_is_oracle_payload(payload)) {
    stop("Phase 3 oracle shard payload schema is malformed", call. = FALSE)
  }
  recomputed <- fastkpc_full_cuda_phase3_summarize_oracle_rows(
    setup_results = payload$setup_results,
    target_parity = payload$target_parity,
    resource_metrics = payload$resource_metrics,
    stage_timing = payload$stage_timing,
    fallbacks = payload$fallbacks,
    failures = payload$failures
  )
  if (!identical(payload$summary, recomputed)) {
    stop("Phase 3 oracle shard summary is not row-derived", call. = FALSE)
  }
  if (recomputed$target_level_stable_sync_count != 0L) {
    stop("Phase 3 oracle target-level stable sync hard gate failed",
         call. = FALSE)
  }
  if (recomputed$cuda_device_synchronize_count != 0L) {
    stop("Phase 3 oracle cudaDeviceSynchronize hard gate failed",
         call. = FALSE)
  }
  resources_conserved <-
    recomputed$prepared_handle_create_count == recomputed$setup_count &&
    recomputed$prepared_handle_destroy_count == recomputed$setup_count &&
    recomputed$residual_token_acquire_count == recomputed$setup_count &&
    recomputed$residual_token_release_count == recomputed$setup_count &&
    recomputed$output_slot_acquire_count == recomputed$setup_count &&
    recomputed$output_slot_release_count == recomputed$setup_count &&
    recomputed$output_slot_live_count == 0L
  if (!isTRUE(resources_conserved)) {
    stop("Phase 3 oracle shard resource conservation failed", call. = FALSE)
  }
  invisible(recomputed)
}

fastkpc_full_cuda_phase3_run_oracle_shard <- function(
    context, shard_id, setup_keys, target_rows, catalog) {
  required_helpers <- c(
    "fastkpc_full_cuda_fixed_sp_execute_oracle_setup",
    "fastkpc_full_cuda_fixed_sp_oracle_empty_frame",
    "fixed_sp_cuda_runtime_info"
  )
  missing_helpers <- required_helpers[!vapply(
    required_helpers, exists, logical(1L), mode = "function", inherits = TRUE
  )]
  if (length(missing_helpers) > 0L) {
    stop("Phase 3 oracle shard runtime helpers are unavailable: ",
         paste(missing_helpers, collapse = ","), call. = FALSE)
  }
  shard_id <- .fastkpc_full_cuda_phase3_whole_scalar(
    shard_id, 0L, "shard_id"
  )
  setup_keys <- .fastkpc_full_cuda_phase3_key_vector(
    setup_keys, "setup_keys", allow_empty = TRUE, allow_empty_list = TRUE
  )
  if (!identical(setup_keys, sort(setup_keys, method = "radix"))) {
    stop("Phase 3 oracle shard setup order is not canonical", call. = FALSE)
  }
  required_target_fields <- c(
    "prepared_s_key_sha256", "residual_key_sha256", "planned_route"
  )
  if (!is.data.frame(target_rows) ||
      !all(required_target_fields %in% names(target_rows))) {
    stop("Phase 3 oracle shard target rows are malformed", call. = FALSE)
  }
  if (length(setup_keys) == 0L) {
    if (nrow(target_rows) != 0L) {
      stop("empty Phase 3 oracle shard has target rows", call. = FALSE)
    }
  } else {
    setup_rank <- match(target_rows$prepared_s_key_sha256, setup_keys)
    expected_order <- order(
      setup_rank, target_rows$residual_key_sha256, method = "radix"
    )
    if (nrow(target_rows) == 0L || anyNA(setup_rank) ||
        !identical(expected_order, seq_len(nrow(target_rows))) ||
        !identical(sort(unique(target_rows$prepared_s_key_sha256),
                        method = "radix"), setup_keys)) {
      stop("Phase 3 oracle shard target order/ownership is invalid",
           call. = FALSE)
    }
    if ("shard_id" %in% names(target_rows) &&
        any(as.integer(target_rows$shard_id) != shard_id)) {
      stop("Phase 3 oracle shard target assignment mismatch", call. = FALSE)
    }
  }
  fastkpc_full_cuda_phase3_discover_catalog_authority(catalog)
  runtime_info <- fixed_sp_cuda_runtime_info(context)
  fastkpc_full_cuda_fixed_sp_phase3c_validate_runtime_info(
    runtime_info, paste("Phase 3 oracle shard", shard_id, "runtime")
  )

  setup_outputs <- lapply(seq_along(setup_keys), function(index) {
    setup_key <- setup_keys[[index]]
    selected_targets <- target_rows[
      target_rows$prepared_s_key_sha256 == setup_key, , drop = FALSE
    ]
    rownames(selected_targets) <- NULL
    fastkpc_full_cuda_fixed_sp_execute_oracle_setup(
      context = context, catalog = catalog, setup_key = setup_key,
      target_rows = selected_targets, shard_id = shard_id,
      setup_ordinal = as.integer(index)
    )
  })
  bind_or_empty <- function(name) {
    if (length(setup_outputs) == 0L) {
      return(fastkpc_full_cuda_fixed_sp_oracle_empty_frame(name))
    }
    value <- do.call(rbind, lapply(setup_outputs, `[[`, name))
    rownames(value) <- NULL
    fastkpc_full_cuda_fixed_sp_validate_oracle_frame(value, name)
  }
  setup_results <- bind_or_empty("setup_results")
  target_parity <- bind_or_empty("target_parity")
  resource_metrics <- bind_or_empty("resource_metrics")
  stage_timing <- bind_or_empty("stage_timing")
  fallbacks <- .fastkpc_full_cuda_phase3_oracle_fallback_rows(
    target_parity, resource_metrics
  )
  failures <- .fastkpc_full_cuda_phase3_oracle_failure_rows(target_parity)
  summary <- fastkpc_full_cuda_phase3_summarize_oracle_rows(
    setup_results = setup_results, target_parity = target_parity,
    resource_metrics = resource_metrics, stage_timing = stage_timing,
    fallbacks = fallbacks, failures = failures
  )
  payload <- list(
    setup_results = setup_results,
    target_parity = target_parity,
    resource_metrics = resource_metrics,
    stage_timing = stage_timing,
    fallbacks = fallbacks,
    failures = failures,
    summary = summary
  )
  .fastkpc_full_cuda_phase3_validate_oracle_payload(payload)
  resource_count <- function(field) {
    as.integer(sum(resource_metrics[[field]]))
  }
  result <- list(
    payload = payload,
    resource_counts = list(
      prepared_handle_create_count = resource_count(
        "prepared_handle_create_count"
      ),
      prepared_handle_destroy_count = resource_count(
        "prepared_handle_destroy_count"
      ),
      residual_token_acquire_count = resource_count(
        "residual_token_acquire_count"
      ),
      residual_token_release_count = resource_count(
        "residual_token_release_count"
      ),
      output_slot_acquire_count = resource_count("output_slot_acquire_count"),
      output_slot_release_count = resource_count("output_slot_release_count")
    )
  )
  result
}

.fastkpc_full_cuda_phase3_resource_fields <- function() {
  c(
    "prepared_handle_create_count", "prepared_handle_destroy_count",
    "residual_token_acquire_count", "residual_token_release_count",
    "output_slot_acquire_count", "output_slot_release_count"
  )
}

.fastkpc_full_cuda_phase3_validate_executor_result <- function(value) {
  if (!.fastkpc_full_cuda_phase3_exact_named_list(
        value, c("payload", "resource_counts")
      ) || !.fastkpc_full_cuda_phase3_exact_named_list(
        value$resource_counts,
        .fastkpc_full_cuda_phase3_resource_fields()
      )) {
    stop("Phase 3 shard executor result schema is malformed",
         call. = FALSE)
  }
  .fastkpc_full_cuda_phase3_payload_semantic_hashes(value$payload)
  counts <- setNames(vapply(names(value$resource_counts), function(field) {
    .fastkpc_full_cuda_phase3_whole_scalar(
      value$resource_counts[[field]], 0L, paste("executor", field)
    )
  }, integer(1L)), names(value$resource_counts))
  if (counts[["prepared_handle_create_count"]] !=
        counts[["prepared_handle_destroy_count"]] ||
      counts[["residual_token_acquire_count"]] !=
        counts[["residual_token_release_count"]] ||
      counts[["output_slot_acquire_count"]] !=
        counts[["output_slot_release_count"]]) {
    stop("Phase 3 shard executor leaked a tracked resource",
         call. = FALSE)
  }
  list(payload = value$payload, resource_counts = counts)
}

fastkpc_full_cuda_phase3_run_shards <- function(
    output_dir, kind, setup_keys, target_rows, identity, route_config,
    executor, runtime_create, runtime_destroy, scope,
    shard_count = NULL, stop_after = NULL, progress_hook = NULL) {
  output_clean <- typeof(output_dir) == "character" &&
    length(output_dir) == 1L && !is.object(output_dir) &&
    is.null(attributes(output_dir)) && !is.na(output_dir) &&
    nzchar(output_dir) && !grepl("[\r\n]", output_dir)
  callback_clean <- is.function(executor) && is.function(runtime_create) &&
    is.function(runtime_destroy) &&
    (is.null(progress_hook) || is.function(progress_hook))
  if (!isTRUE(output_clean) || !isTRUE(callback_clean)) {
    stop("Phase 3 shard runner inputs are malformed", call. = FALSE)
  }
  scope <- .fastkpc_full_cuda_phase3_scope(scope)
  contract <- .fastkpc_full_cuda_phase3_kind_contract(kind)
  plan <- fastkpc_full_cuda_phase3_plan_shards(
    setup_keys = setup_keys,
    target_rows = target_rows,
    scope = scope,
    shard_count = shard_count
  )
  .fastkpc_full_cuda_phase3_validate_route_for_shards(
    route_config, scope
  )
  identity_info <- .fastkpc_full_cuda_phase3_validate_execution_identity(
    identity = identity,
    route_config = route_config,
    scope = scope,
    plan = plan
  )
  if (!is.null(stop_after)) {
    stop_after <- .fastkpc_full_cuda_phase3_whole_scalar(
      stop_after, 1L, "stop_after"
    )
  }

  shards_dir <- file.path(output_dir, "shards")
  sessions_dir <- file.path(output_dir, "sessions")
  dir.create(shards_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(sessions_dir, recursive = TRUE, showWarnings = FALSE)
  scan <- .fastkpc_full_cuda_phase3_scan_shards(
    shards_dir, plan$shard_count
  )
  reused <- integer()
  missing <- integer()
  for (shard_id in scan$expected_ids) {
    id <- as.character(shard_id)
    pair_exists <- c(
      rds = id %in% names(scan$rds_paths),
      summary = id %in% names(scan$summary_paths)
    )
    paths <- .fastkpc_full_cuda_phase3_shard_paths(
      shards_dir, shard_id
    )
    reusable <- FALSE
    if (all(pair_exists)) {
      descriptor <- .fastkpc_full_cuda_phase3_shard_descriptor(
        plan, shard_id
      )
      reusable <- !is.null(tryCatch(
        .fastkpc_full_cuda_phase3_read_reusable_shard(
          paths = paths,
          sessions_dir = sessions_dir,
          contract = contract,
          descriptor = descriptor,
          plan = plan,
          identity_info = identity_info,
          route_config = route_config
        ),
        error = function(error) NULL
      ))
    }
    if (reusable) {
      reused <- c(reused, shard_id)
    } else {
      unlink(c(paths$rds, paths$summary_json), force = TRUE)
      missing <- c(missing, shard_id)
    }
  }
  reused <- as.integer(reused)
  missing <- as.integer(missing)
  if (length(missing) == 0L) {
    return(list(
      status = "complete",
      requested_shard_ids = integer(),
      completed_shard_ids = integer(),
      reused_shard_ids = reused,
      written_shard_ids = integer(),
      missing_shard_ids = integer(),
      session_id = NULL,
      runtime_context_create_count = 0L,
      runtime_context_destroy_count = 0L
    ))
  }

  requested <- missing
  selected <- if (is.null(stop_after)) {
    requested
  } else {
    head(requested, stop_after)
  }
  session_id <- fastkpc_full_cuda_phase3_session_id()
  session_path <- .fastkpc_full_cuda_phase3_session_path(
    sessions_dir, session_id
  )
  while (file.exists(session_path)) {
    session_id <- fastkpc_full_cuda_phase3_session_id()
    session_path <- .fastkpc_full_cuda_phase3_session_path(
      sessions_dir, session_id
    )
  }
  session <- list(
    schema_version = "full-cuda-ci-phase3-session-v1",
    session_id = session_id,
    input_identity_hash = identity_info$input_identity_hash,
    route_config_hash = route_config$sha256,
    requested_shard_ids = requested,
    completed_shard_ids = integer(),
    runtime_context_create_count = 0L,
    runtime_context_destroy_count = 0L,
    prepared_handle_create_count = 0L,
    prepared_handle_destroy_count = 0L,
    residual_token_acquire_count = 0L,
    residual_token_release_count = 0L,
    output_slot_acquire_count = 0L,
    output_slot_release_count = 0L,
    target_level_stable_sync_count = 0L,
    status = "running"
  )
  .fastkpc_full_cuda_phase3_atomic_write_session(session, sessions_dir)

  context <- NULL
  context_open <- FALSE
  run_error <- NULL
  written <- integer()
  tryCatch({
    context <- runtime_create()
    if (is.null(context)) {
      stop("Phase 3 runtime_create returned NULL", call. = FALSE)
    }
    context_open <- TRUE
    session$runtime_context_create_count <- 1L
    .fastkpc_full_cuda_phase3_atomic_write_session(session, sessions_dir)

    for (shard_id in selected) {
      descriptor <- .fastkpc_full_cuda_phase3_shard_descriptor(
        plan, shard_id
      )
      result <- .fastkpc_full_cuda_phase3_validate_executor_result(
        executor(
          context = context,
          shard_id = shard_id,
          setup_keys = descriptor$setup_keys,
          target_rows = descriptor$target_rows
        )
      )
      oracle_accounting <- if (
        identical(contract$kind, "oracle_sp") &&
          .fastkpc_full_cuda_phase3_is_oracle_payload(result$payload)
      ) {
        .fastkpc_full_cuda_phase3_validate_oracle_payload(result$payload)
      } else {
        NULL
      }
      envelope <- .fastkpc_full_cuda_phase3_build_shard_envelope(
        contract = contract,
        descriptor = descriptor,
        session = session,
        identity_info = identity_info,
        route_config = route_config,
        payload = result$payload
      )
      paths <- .fastkpc_full_cuda_phase3_shard_paths(
        shards_dir, shard_id
      )
      .fastkpc_full_cuda_phase3_atomic_write_shard(
        envelope = envelope,
        contract = contract,
        descriptor = descriptor,
        identity_info = identity_info,
        route_config = route_config,
        paths = paths
      )
      counts <- result$resource_counts
      session$prepared_handle_create_count <- as.integer(
        session$prepared_handle_create_count +
          counts[["prepared_handle_create_count"]]
      )
      session$prepared_handle_destroy_count <- as.integer(
        session$prepared_handle_destroy_count +
          counts[["prepared_handle_destroy_count"]]
      )
      session$residual_token_acquire_count <- as.integer(
        session$residual_token_acquire_count +
          counts[["residual_token_acquire_count"]]
      )
      session$residual_token_release_count <- as.integer(
        session$residual_token_release_count +
          counts[["residual_token_release_count"]]
      )
      session$output_slot_acquire_count <- as.integer(
        session$output_slot_acquire_count +
          counts[["output_slot_acquire_count"]]
      )
      session$output_slot_release_count <- as.integer(
        session$output_slot_release_count +
          counts[["output_slot_release_count"]]
      )
      if (!is.null(oracle_accounting)) {
        session$target_level_stable_sync_count <- as.integer(
          session$target_level_stable_sync_count +
            oracle_accounting$target_level_stable_sync_count
        )
      }
      session$completed_shard_ids <- c(
        session$completed_shard_ids, as.integer(shard_id)
      )
      written <- c(written, as.integer(shard_id))
      .fastkpc_full_cuda_phase3_atomic_write_session(
        session, sessions_dir
      )
      if (!is.null(progress_hook)) {
        progress_hook(session = session, event = "shard_complete")
      }
    }
  }, error = function(error) {
    run_error <<- error
  })

  if (context_open) {
    destroy_error <- tryCatch({
      runtime_destroy(context)
      NULL
    }, error = function(error) error)
    if (is.null(destroy_error)) {
      session$runtime_context_destroy_count <- 1L
    } else if (is.null(run_error)) {
      run_error <- destroy_error
    }
    context_open <- FALSE
  }
  if (!is.null(run_error)) {
    session$status <- "running"
    .fastkpc_full_cuda_phase3_atomic_write_session(session, sessions_dir)
    stop(run_error)
  }

  session$status <- "complete"
  .fastkpc_full_cuda_phase3_validate_session_plan_resources(
    session, plan
  )
  .fastkpc_full_cuda_phase3_atomic_write_session(session, sessions_dir)
  if (!is.null(progress_hook)) {
    progress_hook(session = session, event = "session_complete")
  }
  remaining <- setdiff(requested, written)
  list(
    status = if (length(remaining) > 0L) "stopped" else "complete",
    requested_shard_ids = requested,
    completed_shard_ids = as.integer(written),
    reused_shard_ids = reused,
    written_shard_ids = as.integer(written),
    missing_shard_ids = as.integer(remaining),
    session_id = session$session_id,
    runtime_context_create_count =
      as.integer(session$runtime_context_create_count),
    runtime_context_destroy_count =
      as.integer(session$runtime_context_destroy_count)
  )
}

.fastkpc_full_cuda_phase3_merge_payloads <- function(
    payloads, plan) {
  if (length(payloads) != plan$shard_count ||
      any(!vapply(payloads, is.list, logical(1L)))) {
    stop("Phase 3 merge payload set is malformed", call. = FALSE)
  }
  payload_names <- names(payloads[[1L]])
  if (is.null(payload_names) || length(payload_names) == 0L ||
      anyDuplicated(payload_names) || any(!vapply(payloads, function(value) {
        identical(names(value), payload_names)
      }, logical(1L)))) {
    stop("Phase 3 merge payload schemas disagree", call. = FALSE)
  }
  setup_order <- plan$assignments$prepared_s_key_sha256
  target_order <- sort(
    plan$target_rows$residual_key_sha256, method = "radix"
  )
  merged <- lapply(payload_names, function(name) {
    values <- lapply(payloads, `[[`, name)
    if (all(vapply(values, is.data.frame, logical(1L)))) {
      fields <- names(values[[1L]])
      if (any(!vapply(values, function(value) {
            identical(names(value), fields)
          }, logical(1L)))) {
        stop("Phase 3 merge data-frame schemas disagree", call. = FALSE)
      }
      value <- do.call(rbind, values)
      rownames(value) <- NULL
      if ("prepared_s_key_sha256" %in% fields) {
        setup_rank <- match(value$prepared_s_key_sha256, setup_order)
        if (anyNA(setup_rank)) {
          stop("Phase 3 merged payload has an unknown setup key",
               call. = FALSE)
        }
        order_id <- if ("residual_key_sha256" %in% fields) {
          order(
            setup_rank, value$residual_key_sha256, method = "radix"
          )
        } else {
          order(setup_rank, method = "radix")
        }
        value <- value[order_id, , drop = FALSE]
        rownames(value) <- NULL
      } else if ("residual_key_sha256" %in% fields) {
        target_rank <- match(value$residual_key_sha256, target_order)
        if (anyNA(target_rank)) {
          stop("Phase 3 merged payload has an unknown target key",
               call. = FALSE)
        }
        value <- value[order(target_rank, method = "radix"), , drop = FALSE]
        rownames(value) <- NULL
      }
      return(value)
    }
    if (all(vapply(values, is.atomic, logical(1L))) &&
        length(unique(vapply(values, typeof, character(1L)))) == 1L) {
      return(do.call(c, unname(values)))
    }
    if (all(vapply(values, function(value) {
          is.list(value) && !is.object(value)
        }, logical(1L)))) {
      return(unlist(values, recursive = FALSE, use.names = FALSE))
    }
    stop("Phase 3 merge payload component is unsupported", call. = FALSE)
  })
  names(merged) <- payload_names
  merged
}

.fastkpc_full_cuda_phase3_atomic_write_merged_rds <- function(
    value, output_path) {
  clean <- typeof(output_path) == "character" &&
    length(output_path) == 1L && !is.object(output_path) &&
    is.null(attributes(output_path)) && !is.na(output_path) &&
    nzchar(output_path) && !grepl("[\r\n]", output_path)
  if (!isTRUE(clean)) {
    stop("Phase 3 merged RDS output path is malformed", call. = FALSE)
  }
  output_dir <- dirname(output_path)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  temporary <- tempfile(
    ".phase3-merged-", tmpdir = output_dir, fileext = ".tmp"
  )
  on.exit(unlink(temporary, force = TRUE), add = TRUE)
  saveRDS(value, temporary, version = 2, compress = FALSE)
  disk_value <- tryCatch(readRDS(temporary), error = function(error) NULL)
  if (!identical(disk_value, value)) {
    stop("Phase 3 merged RDS validation failed", call. = FALSE)
  }
  if (!file.rename(temporary, output_path)) {
    stop("failed to atomically publish Phase 3 merged RDS", call. = FALSE)
  }
  invisible(output_path)
}

fastkpc_full_cuda_phase3_merge_shards <- function(
    output_dir, kind, setup_keys, target_rows, identity, route_config,
    scope, shard_count = NULL, output_path = NULL) {
  scope <- .fastkpc_full_cuda_phase3_scope(scope)
  contract <- .fastkpc_full_cuda_phase3_kind_contract(kind)
  plan <- fastkpc_full_cuda_phase3_plan_shards(
    setup_keys = setup_keys,
    target_rows = target_rows,
    scope = scope,
    shard_count = shard_count
  )
  .fastkpc_full_cuda_phase3_validate_route_for_shards(
    route_config, scope
  )
  identity_info <- .fastkpc_full_cuda_phase3_validate_execution_identity(
    identity = identity,
    route_config = route_config,
    scope = scope,
    plan = plan
  )
  shards_dir <- file.path(output_dir, "shards")
  sessions_dir <- file.path(output_dir, "sessions")
  scan <- .fastkpc_full_cuda_phase3_scan_shards(
    shards_dir, plan$shard_count, require_complete = TRUE
  )
  loaded <- lapply(scan$expected_ids, function(shard_id) {
    descriptor <- .fastkpc_full_cuda_phase3_shard_descriptor(
      plan, shard_id
    )
    .fastkpc_full_cuda_phase3_read_reusable_shard(
      paths = .fastkpc_full_cuda_phase3_shard_paths(
        shards_dir, shard_id
      ),
      sessions_dir = sessions_dir,
      contract = contract,
      descriptor = descriptor,
      plan = plan,
      identity_info = identity_info,
      route_config = route_config
    )
  })
  payload <- .fastkpc_full_cuda_phase3_merge_payloads(
    lapply(loaded, function(value) value$envelope$payload), plan
  )
  semantic_hashes <- .fastkpc_full_cuda_phase3_payload_semantic_hashes(
    payload
  )
  setup_keys <- plan$assignments$prepared_s_key_sha256
  target_keys <- sort(
    plan$target_rows$residual_key_sha256, method = "radix"
  )
  merged <- list(
    schema_version = "full-cuda-ci-phase3-merged-shards-v1",
    artifact_kind = contract$kind,
    artifact_schema_version = contract$artifact_schema_version,
    input_identity_hash = identity_info$input_identity_hash,
    route_config_hash = route_config$sha256,
    shard_count = plan$shard_count,
    expected_setup_keys = setup_keys,
    expected_setup_count = as.integer(length(setup_keys)),
    expected_setup_hash =
      fastkpc_full_cuda_census_key_set_hash(setup_keys),
    expected_target_keys = target_keys,
    expected_target_count = as.integer(length(target_keys)),
    expected_target_hash =
      fastkpc_full_cuda_census_key_set_hash(target_keys),
    payload = payload,
    payload_semantic_hashes = as.list(semantic_hashes),
    payload_semantic_hash =
      .fastkpc_full_cuda_phase3_payload_semantic_hash(semantic_hashes)
  )
  if (!is.null(output_path)) {
    .fastkpc_full_cuda_phase3_atomic_write_merged_rds(
      merged, output_path
    )
  }
  merged
}
