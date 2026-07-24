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
        "full-cuda-ci-native-build-trace-attestation-v2"
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
    oracle_runner =
      "fastkpc/tools/run_full_cuda_ci_fixed_sp_oracle_sp.R",
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
  native_build_attestation_sha256 <-
    fastkpc_full_cuda_fixed_sp_native_build_attestation_hash(dependencies)
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
      "full-cuda-ci-native-build-trace-attestation-v2",
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
                 "full-cuda-ci-native-build-trace-attestation-v2") ||
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
  semantic_manifest_path <- file.path(output_dir, "manifest.json")
  if (identical(kind, "oracle_sp") &&
      file.exists(semantic_manifest_path) &&
      !dir.exists(semantic_manifest_path)) {
    semantic_manifest <- tryCatch(
      .fastkpc_full_cuda_phase3_read_json(
        semantic_manifest_path, "manifest.json"
      ),
      error = function(error) NULL
    )
    if (is.list(semantic_manifest) &&
        "oracle_semantics_version" %in% names(semantic_manifest)) {
      return(.fastkpc_full_cuda_phase3_validate_completed_oracle_artifact(
        output_dir = output_dir, expected_identity = expected_identity,
        require_full = require_full, catalog = catalog, device_id = device_id
      ))
    }
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

.fastkpc_full_cuda_phase3_oracle_artifact_lock_path <- function(output_dir) {
  clean <- typeof(output_dir) == "character" && length(output_dir) == 1L &&
    !is.object(output_dir) && is.null(attributes(output_dir)) &&
    !is.na(output_dir) && nzchar(output_dir) && !grepl("[\r\n]", output_dir)
  if (!isTRUE(clean)) {
    stop("Phase 3 oracle artifact output path is malformed", call. = FALSE)
  }
  paste0(normalizePath(output_dir, mustWork = FALSE),
         ".phase3-oracle.lock")
}

.fastkpc_full_cuda_phase3_require_filelock <- function(
    .namespace_checker = requireNamespace) {
  if (!is.function(.namespace_checker) || !isTRUE(tryCatch(
        .namespace_checker("filelock", quietly = TRUE),
        error = function(error) FALSE
      ))) {
    stop(
      "Phase 3 artifact and runner locks require the CRAN package filelock",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

.fastkpc_full_cuda_phase3_scalar_lock_text <- function(value) {
  typeof(value) == "character" && length(value) == 1L &&
    !is.object(value) && is.null(attributes(value)) &&
    !is.na(value) && nzchar(value) && !grepl("[\r\n]", value)
}

.fastkpc_full_cuda_phase3_lock_registry <- local({
  registry <- new.env(hash = TRUE, parent = emptyenv())
  function() registry
})

.fastkpc_full_cuda_phase3_normalize_lock_path <- function(lock_path) {
  if (!.fastkpc_full_cuda_phase3_scalar_lock_text(lock_path)) {
    stop("Phase 3 lock path is malformed", call. = FALSE)
  }
  parent <- dirname(lock_path)
  dir.create(parent, recursive = TRUE, showWarnings = FALSE)
  if (!dir.exists(parent)) {
    stop("failed to create Phase 3 lock parent directory", call. = FALSE)
  }
  file.path(
    normalizePath(parent, mustWork = TRUE), basename(lock_path)
  )
}

.fastkpc_full_cuda_phase3_active_lock_state <- function(state, owner_pid) {
  is.environment(state) &&
    identical(sort(ls(state, all.names = TRUE)),
              c("handle", "owner_pid", "released")) &&
    identical(state$owner_pid, owner_pid) &&
    identical(state$released, FALSE) &&
    inherits(state$handle, "filelock_lock")
}

.fastkpc_full_cuda_phase3_acquire_lock <- function(
    lock_path, purpose, .namespace_checker = requireNamespace,
    .lock_function = filelock::lock) {
  .fastkpc_full_cuda_phase3_require_filelock(.namespace_checker)
  if (!.fastkpc_full_cuda_phase3_scalar_lock_text(lock_path) ||
      !.fastkpc_full_cuda_phase3_scalar_lock_text(purpose) ||
      !is.function(.lock_function)) {
    stop("Phase 3 lock acquisition inputs are malformed", call. = FALSE)
  }
  lock_path <- .fastkpc_full_cuda_phase3_normalize_lock_path(lock_path)
  owner_pid <- as.integer(Sys.getpid())
  registry <- .fastkpc_full_cuda_phase3_lock_registry()
  if (exists(lock_path, envir = registry, inherits = FALSE)) {
    existing <- get(lock_path, envir = registry, inherits = FALSE)
    clean_entry <- is.list(existing) && !is.object(existing) &&
      identical(names(existing), c("owner_pid", "state")) &&
      typeof(existing$owner_pid) == "integer" &&
      length(existing$owner_pid) == 1L && !is.na(existing$owner_pid) &&
      is.environment(existing$state)
    if (!isTRUE(clean_entry)) {
      stop("Phase 3 process-local lock registry is malformed",
           call. = FALSE)
    }
    if (!identical(existing$owner_pid, owner_pid)) {
      rm(list = lock_path, envir = registry)
    } else if (.fastkpc_full_cuda_phase3_active_lock_state(
                 existing$state, owner_pid
               )) {
      stop("Phase 3 ", purpose,
           " lock is already held by this process", call. = FALSE)
    } else if (identical(existing$state$owner_pid, owner_pid) &&
               identical(existing$state$released, TRUE) &&
               is.null(existing$state$handle)) {
      rm(list = lock_path, envir = registry)
    } else {
      stop("Phase 3 process-local lock registry state is invalid",
           call. = FALSE)
    }
  }
  handle <- tryCatch(
    .lock_function(lock_path, exclusive = TRUE, timeout = 0),
    error = function(error) {
      stop("failed to acquire Phase 3 ", purpose, " lock: ",
           conditionMessage(error), call. = FALSE)
    }
  )
  if (is.null(handle)) {
    stop("Phase 3 ", purpose,
         " lock is already held by another process", call. = FALSE)
  }
  registered <- FALSE
  on.exit({
    if (!isTRUE(registered)) {
      try(filelock::unlock(handle), silent = TRUE)
    }
  }, add = TRUE)
  state <- new.env(parent = emptyenv())
  state$handle <- handle
  state$owner_pid <- owner_pid
  state$released <- FALSE
  assign(
    lock_path, list(owner_pid = owner_pid, state = state),
    envir = registry
  )
  registered <- TRUE
  list(lock_path = lock_path, purpose = purpose, state = state)
}

.fastkpc_full_cuda_phase3_lock_shape_valid <- function(lock, purpose) {
  is.list(lock) && !is.object(lock) &&
    identical(names(lock), c("lock_path", "purpose", "state")) &&
    .fastkpc_full_cuda_phase3_scalar_lock_text(lock$lock_path) &&
    identical(lock$purpose, purpose) &&
    .fastkpc_full_cuda_phase3_active_lock_state(
      lock$state, as.integer(Sys.getpid())
    )
}

.fastkpc_full_cuda_phase3_owns_lock <- function(lock, purpose) {
  if (!.fastkpc_full_cuda_phase3_lock_shape_valid(lock, purpose) ||
      !file.exists(lock$lock_path) || dir.exists(lock$lock_path)) {
    return(FALSE)
  }
  registry <- .fastkpc_full_cuda_phase3_lock_registry()
  if (!exists(lock$lock_path, envir = registry, inherits = FALSE)) {
    return(FALSE)
  }
  entry <- get(lock$lock_path, envir = registry, inherits = FALSE)
  is.list(entry) && !is.object(entry) &&
    identical(names(entry), c("owner_pid", "state")) &&
    identical(entry$owner_pid, as.integer(Sys.getpid())) &&
    identical(entry$state, lock$state)
}

.fastkpc_full_cuda_phase3_refresh_lock <- function(
    lock, purpose, .boundary = NULL) {
  if (!is.null(.boundary) &&
      !.fastkpc_full_cuda_phase3_scalar_lock_text(.boundary)) {
    stop("Phase 3 lock refresh boundary is malformed", call. = FALSE)
  }
  if (!.fastkpc_full_cuda_phase3_owns_lock(lock, purpose)) {
    stop("Phase 3 lock ownership is invalid", call. = FALSE)
  }
  invisible(TRUE)
}

.fastkpc_full_cuda_phase3_release_lock <- function(lock, purpose) {
  if (!is.list(lock) || !identical(
        names(lock), c("lock_path", "purpose", "state")
      ) || !identical(lock$purpose, purpose) || !is.environment(lock$state) ||
      !identical(lock$state$owner_pid, as.integer(Sys.getpid())) ||
      !identical(lock$state$released, FALSE) ||
      !inherits(lock$state$handle, "filelock_lock")) {
    return(invisible(NULL))
  }
  unlocked <- filelock::unlock(lock$state$handle)
  if (!isTRUE(unlocked)) {
    stop("failed to release Phase 3 ", purpose, " lock", call. = FALSE)
  }
  lock$state$handle <- NULL
  lock$state$released <- TRUE
  registry <- .fastkpc_full_cuda_phase3_lock_registry()
  if (exists(lock$lock_path, envir = registry, inherits = FALSE)) {
    entry <- get(lock$lock_path, envir = registry, inherits = FALSE)
    if (is.list(entry) && !is.object(entry) &&
        identical(names(entry), c("owner_pid", "state")) &&
        identical(entry$owner_pid, as.integer(Sys.getpid())) &&
        identical(entry$state, lock$state)) {
      rm(list = lock$lock_path, envir = registry)
    }
  }
  invisible(NULL)
}

.fastkpc_full_cuda_phase3_acquire_oracle_artifact_lock <- function(
    output_dir, .namespace_checker = requireNamespace) {
  .fastkpc_full_cuda_phase3_acquire_lock(
    lock_path =
      .fastkpc_full_cuda_phase3_oracle_artifact_lock_path(output_dir),
    purpose = "oracle_artifact", .namespace_checker = .namespace_checker
  )
}

.fastkpc_full_cuda_phase3_owns_oracle_artifact_lock <- function(lock) {
  .fastkpc_full_cuda_phase3_owns_lock(lock, "oracle_artifact")
}

.fastkpc_full_cuda_phase3_require_oracle_artifact_lock <- function(
    lock, output_dir) {
  expected_path <-
    .fastkpc_full_cuda_phase3_oracle_artifact_lock_path(output_dir)
  if (!.fastkpc_full_cuda_phase3_owns_oracle_artifact_lock(lock) ||
      !identical(lock$lock_path, expected_path)) {
    stop("Phase 3 oracle artifact lock ownership is invalid", call. = FALSE)
  }
  invisible(TRUE)
}

.fastkpc_full_cuda_phase3_refresh_oracle_artifact_lock <- function(
    lock, .boundary = NULL) {
  .fastkpc_full_cuda_phase3_refresh_lock(
    lock, "oracle_artifact", .boundary = .boundary
  )
}

.fastkpc_full_cuda_phase3_release_oracle_artifact_lock <- function(lock) {
  .fastkpc_full_cuda_phase3_release_lock(lock, "oracle_artifact")
}

.fastkpc_full_cuda_phase3_shard_runner_lock_path <- function(output_dir) {
  clean <- typeof(output_dir) == "character" && length(output_dir) == 1L &&
    !is.object(output_dir) && is.null(attributes(output_dir)) &&
    !is.na(output_dir) && nzchar(output_dir) && !grepl("[\r\n]", output_dir)
  if (!isTRUE(clean)) {
    stop("Phase 3 shard runner output path is malformed", call. = FALSE)
  }
  paste0(normalizePath(output_dir, mustWork = FALSE),
         ".phase3-runner.lock")
}

.fastkpc_full_cuda_phase3_acquire_shard_runner_lock <- function(
    output_dir, .namespace_checker = requireNamespace) {
  .fastkpc_full_cuda_phase3_acquire_lock(
    lock_path = .fastkpc_full_cuda_phase3_shard_runner_lock_path(output_dir),
    purpose = "shard_runner", .namespace_checker = .namespace_checker
  )
}

.fastkpc_full_cuda_phase3_refresh_shard_runner_lock <- function(
    lock, .boundary = NULL) {
  .fastkpc_full_cuda_phase3_refresh_lock(
    lock, "shard_runner", .boundary = .boundary
  )
}

.fastkpc_full_cuda_phase3_release_shard_runner_lock <- function(lock) {
  .fastkpc_full_cuda_phase3_release_lock(lock, "shard_runner")
}

fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact <- function(
    output_dir, expected_identity = NULL, require_full = FALSE,
    catalog = NULL, device_id = NULL, .artifact_lock = NULL) {
  .fastkpc_full_cuda_phase3_validate_completed_oracle_artifact(
    output_dir = output_dir,
    expected_identity = expected_identity, require_full = require_full,
    catalog = catalog, device_id = device_id,
    .artifact_lock = .artifact_lock
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

.fastkpc_full_cuda_phase3_oracle_descriptor_target_fields <- function() {
  c(
    "prepared_s_key_sha256", "residual_key_sha256", "shard_id",
    "canonical_setup_rank", "canonical_target_rank", "phase2_shard_id",
    "target", "null_dim", "condition", "coefficient_rank",
    "planned_route", "selected_sp_hash", "coefficient_hash",
    "fitted_hash", "residual_hash", "target_fit_fingerprint"
  )
}

.fastkpc_full_cuda_phase3_oracle_descriptor_target_rows <- function(
    catalog, target_rows) {
  required_helpers <- c(
    "fastkpc_full_cuda_phase3_discover_catalog_authority",
    "fastkpc_full_cuda_fixed_sp_route"
  )
  missing_helpers <- required_helpers[!vapply(
    required_helpers, exists, logical(1L), mode = "function", inherits = TRUE
  )]
  if (length(missing_helpers) > 0L) {
    stop("Phase 3 oracle descriptor helpers are unavailable", call. = FALSE)
  }
  required_target_fields <- c(
    "prepared_s_key_sha256", "residual_key_sha256", "target", "null_dim",
    "condition", "coefficient_rank", "planned_route", "selected_sp_hash",
    "coefficient_hash", "fitted_hash", "residual_hash",
    "target_fit_fingerprint"
  )
  catalog_clean <- is.list(catalog) && !is.object(catalog) &&
    is.data.frame(catalog$setup_index) &&
    is.list(catalog$inputs) && !is.object(catalog$inputs) &&
    is.data.frame(catalog$inputs$target_fit_metadata) &&
    is.data.frame(catalog$inputs$same_s_setup_metadata) &&
    is.list(catalog$catalog_contract) &&
    is.data.frame(target_rows) && !anyDuplicated(names(target_rows)) &&
    all(required_target_fields %in% names(target_rows))
  if (!isTRUE(catalog_clean)) {
    stop("Phase 3 oracle descriptor target rows are malformed", call. = FALSE)
  }
  fastkpc_full_cuda_phase3_discover_catalog_authority(catalog)

  setup_index <- catalog$setup_index
  canonical_targets <- catalog$inputs$target_fit_metadata
  canonical_setups <- catalog$inputs$same_s_setup_metadata
  setup_fields <- c("same_S_group_id", "prepared_s_key_sha256")
  canonical_target_fields <- c(
    "residual_key_sha256", "same_S_group_id", "target",
    "penalized_system_condition_at_selected_sp", "coefficient_rank",
    "selected_sp_hash", "coefficient_hash", "fitted_hash", "residual_hash",
    "target_fit_fingerprint"
  )
  canonical_setup_fields <- c(
    "same_S_group_id", "constraint_nullspace_dimension"
  )
  shard_count <- catalog$catalog_contract$shard_count
  canonical_clean <- all(setup_fields %in% names(setup_index)) &&
    all(canonical_target_fields %in% names(canonical_targets)) &&
    all(canonical_setup_fields %in% names(canonical_setups)) &&
    typeof(shard_count) == "integer" && length(shard_count) == 1L &&
    !is.na(shard_count) && shard_count > 0L &&
    !anyDuplicated(setup_index$same_S_group_id) &&
    !anyDuplicated(setup_index$prepared_s_key_sha256) &&
    !anyDuplicated(canonical_targets$residual_key_sha256) &&
    !anyDuplicated(canonical_setups$same_S_group_id)
  if (!isTRUE(canonical_clean)) {
    stop("Phase 3 oracle descriptor catalog indexes are malformed",
         call. = FALSE)
  }

  target_match <- match(
    as.character(target_rows$residual_key_sha256),
    as.character(canonical_targets$residual_key_sha256)
  )
  matched_targets <- canonical_targets[target_match, , drop = FALSE]
  setup_match <- match(
    as.character(matched_targets$same_S_group_id),
    as.character(setup_index$same_S_group_id)
  )
  phase1_setup_match <- match(
    as.character(matched_targets$same_S_group_id),
    as.character(canonical_setups$same_S_group_id)
  )
  canonical_target_keys <- sort(
    as.character(canonical_targets$residual_key_sha256), method = "radix"
  )
  canonical_setup_rank <- as.integer(setup_match)
  canonical_target_rank <- as.integer(match(
    as.character(matched_targets$residual_key_sha256), canonical_target_keys
  ))
  if (anyNA(target_match) || anyNA(canonical_setup_rank) ||
      anyNA(canonical_target_rank) || anyNA(phase1_setup_match)) {
    stop("Phase 3 oracle descriptor catalog mapping is incomplete",
         call. = FALSE)
  }

  null_dim <- as.integer(
    canonical_setups$constraint_nullspace_dimension[phase1_setup_match]
  )
  condition <- as.double(
    matched_targets$penalized_system_condition_at_selected_sp
  )
  coefficient_rank <- as.integer(matched_targets$coefficient_rank)
  planned_route <- fastkpc_full_cuda_fixed_sp_route(
    condition = condition,
    coefficient_rank = coefficient_rank,
    null_dim = null_dim,
    authenticated = rep(TRUE, nrow(target_rows))
  )
  projected <- data.frame(
    prepared_s_key_sha256 = as.character(
      setup_index$prepared_s_key_sha256[setup_match]
    ),
    residual_key_sha256 = as.character(
      matched_targets$residual_key_sha256
    ),
    canonical_setup_rank = canonical_setup_rank,
    canonical_target_rank = canonical_target_rank,
    phase2_shard_id = as.integer(
      (canonical_setup_rank - 1L) %% shard_count
    ),
    target = as.integer(matched_targets$target),
    null_dim = null_dim,
    condition = condition,
    coefficient_rank = coefficient_rank,
    planned_route = as.character(planned_route),
    selected_sp_hash = as.character(matched_targets$selected_sp_hash),
    coefficient_hash = as.character(matched_targets$coefficient_hash),
    fitted_hash = as.character(matched_targets$fitted_hash),
    residual_hash = as.character(matched_targets$residual_hash),
    target_fit_fingerprint = as.character(
      matched_targets$target_fit_fingerprint
    ),
    stringsAsFactors = FALSE
  )
  authority_exact <- all(vapply(required_target_fields, function(field) {
    identical(target_rows[[field]], projected[[field]])
  }, logical(1L)))
  setup_keys <- sort(
    unique(projected$prepared_s_key_sha256), method = "radix"
  )
  expected_order <- if (nrow(projected) == 0L) integer() else order(
    match(projected$prepared_s_key_sha256, setup_keys),
    projected$residual_key_sha256, method = "radix"
  )
  if (!isTRUE(authority_exact) ||
      anyDuplicated(projected$residual_key_sha256) ||
      !identical(expected_order, seq_len(nrow(projected)))) {
    stop("Phase 3 oracle descriptor does not match catalog authority",
         call. = FALSE)
  }
  fields <- setdiff(
    .fastkpc_full_cuda_phase3_oracle_descriptor_target_fields(), "shard_id"
  )
  projected <- projected[, fields, drop = FALSE]
  rownames(projected) <- NULL
  projected
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
  oracle_descriptor_fields <-
    .fastkpc_full_cuda_phase3_oracle_descriptor_target_fields()
  oracle_descriptor_input <- identical(
    names(target_rows), setdiff(oracle_descriptor_fields, "shard_id")
  ) || identical(names(target_rows), oracle_descriptor_fields)
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
  if (isTRUE(oracle_descriptor_input)) {
    target_rows <- target_rows[, oracle_descriptor_fields, drop = FALSE]
  }
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
    "route_config_hash", "executed_native_library_sha256",
    "requested_shard_ids", "completed_shard_ids",
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
      !.fastkpc_full_cuda_phase3_sha256(session$route_config_hash) ||
      !.fastkpc_full_cuda_phase3_sha256(
        session$executed_native_library_sha256
      )) {
    stop("Phase 3 session immutable identity is malformed", call. = FALSE)
  }
  .fastkpc_full_cuda_phase3_named_hash(list(
    schema_version = "full-cuda-ci-phase3-session-identity-v1",
    session = list(
      schema_version = session$schema_version,
      session_id = session$session_id,
      input_identity_hash = session$input_identity_hash,
      route_config_hash = session$route_config_hash,
      executed_native_library_sha256 =
        session$executed_native_library_sha256,
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
    "input_identity_hash", "route_config_hash",
    "executed_native_library_sha256", "expected_setup_keys",
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
    "executed_native_library_sha256",
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
    executed_native_library_sha256 =
      session$executed_native_library_sha256,
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
    executed_native_library_sha256 =
      envelope$executed_native_library_sha256,
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
    .fastkpc_full_cuda_phase3_sha256(
      envelope$executed_native_library_sha256
    ) && identical(
      envelope$executed_native_library_sha256,
      summary$executed_native_library_sha256
    ) &&
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
    "route_config_hash", "executed_native_library_sha256"
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
  if (identical(contract$kind, "oracle_sp")) {
    if (!.fastkpc_full_cuda_phase3_is_oracle_payload(envelope$payload)) {
      stop("Phase 3 oracle shard payload schema is required",
           call. = FALSE)
    }
    .fastkpc_full_cuda_phase3_validate_oracle_payload(
      envelope$payload,
      expected_setup_keys = descriptor$setup_keys,
      expected_target_rows = descriptor$target_rows
    )
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
  ) && identical(envelope$session_identity_hash, session_hash) &&
    identical(
      envelope$executed_native_library_sha256,
      session$executed_native_library_sha256
    )
  if (!isTRUE(ids_bound) || !isTRUE(hashes_bound)) {
    stop("Phase 3 shard/session id and identity binding mismatch",
         call. = FALSE)
  }
  invisible(session_hash)
}

.fastkpc_full_cuda_phase3_recomputable_shard_error <- function(message) {
  structure(
    list(message = message, call = NULL),
    class = c(
      "fastkpc_phase3_recomputable_shard", "error", "condition"
    )
  )
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
  validated_session <- .fastkpc_full_cuda_phase3_validate_session(
    session, require_complete = FALSE
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
  if (identical(session$status, "running")) {
    if (!descriptor$shard_id %in% validated_session$requested_shard_ids) {
      stop("running Phase 3 session did not request its bound shard",
           call. = FALSE)
    }
    stop(.fastkpc_full_cuda_phase3_recomputable_shard_error(
      "Phase 3 shard is bound to an incomplete running session"
    ))
  }
  .fastkpc_full_cuda_phase3_validate_session(
    session, require_complete = TRUE, shard_id = descriptor$shard_id
  )
  .fastkpc_full_cuda_phase3_validate_session_plan_resources(
    session, plan
  )
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
  ok_status <- c(
    "OK_CHOLESKY_BATCHED", "OK_CHOLESKY_SINGLE",
    "OK_AUGMENTED_QR", "OK_AUGMENTED_SVD"
  )
  failed <- !target_parity$solver_status %in% ok_status |
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

.fastkpc_full_cuda_phase3_validate_oracle_row_authority <- function(
    setup_results, target_parity, resource_metrics, stage_timing,
    target_groups = NULL) {
  route_levels <- c(
    "CHOLESKY_BATCHED", "AUGMENTED_QR", "AUGMENTED_SVD"
  )
  status_levels <- c(
    "OK_CHOLESKY_BATCHED", "OK_CHOLESKY_SINGLE",
    "OK_AUGMENTED_QR", "OK_AUGMENTED_SVD"
  )
  target_count <- nrow(target_parity)
  setup_count <- nrow(setup_results)
  if (target_count == 0L) {
    if (setup_count != 0L || nrow(resource_metrics) != 0L ||
        nrow(stage_timing) != 0L) {
      stop("Phase 3 oracle empty row authority is inconsistent",
           call. = FALSE)
    }
    return(invisible(TRUE))
  }
  if (is.null(target_groups)) {
    target_groups <- .fastkpc_full_cuda_phase3_oracle_target_groups(
      setup_results$prepared_s_key_sha256,
      target_parity$prepared_s_key_sha256
    )
  }
  mapped_setup <- target_groups$mapped_setup
  setup_target_count <- target_groups$target_count
  grouping_clean <- typeof(mapped_setup) == "integer" &&
    length(mapped_setup) == target_count && !anyNA(mapped_setup) &&
    all(mapped_setup > 0L & mapped_setup <= setup_count) &&
    typeof(setup_target_count) == "integer" &&
    length(setup_target_count) == setup_count &&
    sum(setup_target_count) == target_count
  if (!isTRUE(grouping_clean)) {
    stop("Phase 3 oracle row-authority grouping is invalid", call. = FALSE)
  }
  route_enum_exact <-
    all(target_parity$planned_route %in% route_levels) &&
    all(target_parity$authenticated_planned_route %in% route_levels) &&
    all(target_parity$executed_route %in% route_levels) &&
    all(target_parity$solver_status %in% status_levels) &&
    identical(
      target_parity$authenticated_planned_route,
      target_parity$planned_route
    )
  authoritative_route <- tryCatch(
    fastkpc_full_cuda_fixed_sp_route(
      condition = target_parity$condition,
      coefficient_rank = target_parity$phase1_coefficient_rank,
      null_dim = target_parity$null_dim,
      authenticated = rep(TRUE, target_count)
    ),
    error = function(error) NULL
  )
  authoritative_bucket <- tryCatch(vapply(
    seq_len(target_count), function(index) {
      fastkpc_full_cuda_census_condition_bucket(
        target_parity$condition[[index]],
        target_parity$phase1_coefficient_rank[[index]],
        target_parity$null_dim[[index]]
      )
    }, character(1L)
  ), error = function(error) NULL)
  if (!isTRUE(route_enum_exact) || is.null(authoritative_route) ||
      is.null(authoritative_bucket) || !identical(
        target_parity$planned_route, unname(authoritative_route)
      ) || !identical(
        target_parity$condition_bucket, unname(authoritative_bucket)
      )) {
    stop("Phase 3 oracle authenticated planned-route evidence is invalid",
         call. = FALSE)
  }

  planned_cholesky <- target_parity$planned_route == "CHOLESKY_BATCHED"
  executed_cholesky <- target_parity$executed_route == "CHOLESKY_BATCHED"
  executed_qr <- target_parity$executed_route == "AUGMENTED_QR"
  executed_svd <- target_parity$executed_route == "AUGMENTED_SVD"
  planned_cholesky_count <- tabulate(
    mapped_setup[planned_cholesky], nbins = setup_count
  )
  expected_true_batched <- planned_cholesky & executed_cholesky &
    planned_cholesky_count[mapped_setup] >= 2L
  expected_true_count_by_setup <- tabulate(
    mapped_setup[expected_true_batched], nbins = setup_count
  )
  expected_true_count <- expected_true_count_by_setup[mapped_setup]
  expected_kernel_by_setup <- setup_target_count >= 2L &
    expected_true_count_by_setup == setup_target_count
  expected_kernel <- expected_kernel_by_setup[mapped_setup]
  expected_subgroups <- as.integer(planned_cholesky_count >= 2L)
  expected_attempted <- as.integer(ifelse(
    planned_cholesky_count >= 2L, planned_cholesky_count, 0L
  ))
  batch_rows_exact <- identical(
    setup_results$true_batched_target_count,
    as.integer(expected_true_count_by_setup)
  ) && identical(
    resource_metrics$true_batched_subgroup_count, expected_subgroups
  ) && identical(
    resource_metrics$true_batched_attempted_target_count, expected_attempted
  ) && identical(
    resource_metrics$true_batched_target_count,
    as.integer(expected_true_count_by_setup)
  )
  expected_status <- ifelse(
    executed_cholesky,
    ifelse(expected_true_batched,
           "OK_CHOLESKY_BATCHED", "OK_CHOLESKY_SINGLE"),
    ifelse(executed_qr, "OK_AUGMENTED_QR", "OK_AUGMENTED_SVD")
  )
  if (!isTRUE(batch_rows_exact) || !identical(
        target_parity$target_true_batched, expected_true_batched
      ) || !identical(
        target_parity$true_batched_kernel, expected_kernel
      ) || !identical(
        target_parity$true_batched_target_count, expected_true_count
      ) || !identical(
        target_parity$solver_status, unname(expected_status)
      )) {
    stop("Phase 3 oracle route/status true-batch evidence is invalid",
         call. = FALSE)
  }

  qr_evidence_exact <-
    all(target_parity$qr_rank[executed_qr] ==
          target_parity$null_dim[executed_qr]) &&
    all(target_parity$geqrf_info[executed_qr] == 0L) &&
    all(target_parity$ormqr_info[executed_qr] == 0L) &&
    all(target_parity$qr_rank[!executed_qr] == -1L) &&
    all(target_parity$geqrf_info[!executed_qr] == -1L) &&
    all(target_parity$ormqr_info[!executed_qr] == -1L)
  svd_evidence_exact <-
    all(target_parity$effective_rank[executed_svd] >= 0L) &&
    all(target_parity$effective_rank[executed_svd] <=
          target_parity$null_dim[executed_svd]) &&
    all(target_parity$svd_info[executed_svd] == 0L) &&
    all(is.finite(target_parity$sigma_max[executed_svd])) &&
    all(target_parity$sigma_max[executed_svd] > 0) &&
    all(is.finite(
      target_parity$smallest_retained_sigma[executed_svd]
    )) &&
    all(target_parity$smallest_retained_sigma[executed_svd] > 0) &&
    all(target_parity$smallest_retained_sigma[executed_svd] <=
          target_parity$sigma_max[executed_svd]) &&
    all(target_parity$effective_rank[!executed_svd] == -1L) &&
    all(target_parity$svd_info[!executed_svd] == -1L) &&
    all(is.nan(target_parity$sigma_max[!executed_svd])) &&
    all(is.nan(target_parity$smallest_retained_sigma[!executed_svd]))
  aggregate_evidence_exact <-
    all(!is.na(target_parity$aggregate_penalty_root_rank[executed_svd])) &&
    all(target_parity$aggregate_penalty_root_rank[executed_svd] >= 0L) &&
    all(target_parity$aggregate_penalty_root_rank[executed_svd] <=
          target_parity$null_dim[executed_svd]) &&
    all(is.na(
      target_parity$aggregate_penalty_root_rank[!executed_svd]
    )) &&
    identical(
      target_parity$aggregate_factor_call_count,
      as.integer(executed_svd)
    ) && identical(
      target_parity$aggregate_b_build_count,
      2L * as.integer(executed_svd)
    ) &&
    all(is.finite(target_parity$aggregate_dstop[executed_svd])) &&
    all(target_parity$aggregate_dstop[executed_svd] >= 0) &&
    all(is.na(target_parity$aggregate_dstop[!executed_svd])) &&
    all(grepl(
      "^[0-9a-f]{64}$",
      target_parity$aggregate_penalty_root_pivot_sha256
    )) &&
    identical(
      resource_metrics$aggregate_penalty_factor_count,
      as.integer(tabulate(mapped_setup[executed_svd], nbins = setup_count))
    ) && identical(
      resource_metrics$aggregate_svd_b_build_count,
      2L * resource_metrics$aggregate_penalty_factor_count
    )
  if (!isTRUE(qr_evidence_exact) || !isTRUE(svd_evidence_exact) ||
      !isTRUE(aggregate_evidence_exact)) {
    stop("Phase 3 oracle route-specific numerical evidence is invalid",
         call. = FALSE)
  }

  checkpoint_pairs <- list(
    c("cholesky_factor_checkpoint_record_count",
      "cholesky_factor_checkpoint_wait_count"),
    c("cholesky_solve_checkpoint_record_count",
      "cholesky_solve_checkpoint_wait_count"),
    c("qr_checkpoint_record_count", "qr_checkpoint_wait_count"),
    c("svd_checkpoint_record_count", "svd_checkpoint_wait_count")
  )
  checkpoint_pairs_exact <- all(vapply(checkpoint_pairs, function(fields) {
    identical(resource_metrics[[fields[[1L]]]],
              resource_metrics[[fields[[2L]]]])
  }, logical(1L)))
  expected_checkpoint <- function(route, planned = FALSE) {
    routes <- if (planned) {
      target_parity$planned_route
    } else {
      target_parity$executed_route
    }
    as.integer(tabulate(
      mapped_setup[routes == route], nbins = setup_count
    ) > 0L)
  }
  checkpoint_exact <- checkpoint_pairs_exact && identical(
    resource_metrics$cholesky_factor_checkpoint_wait_count,
    expected_checkpoint("CHOLESKY_BATCHED", planned = TRUE)
  ) && identical(
    resource_metrics$cholesky_solve_checkpoint_wait_count,
    expected_checkpoint("CHOLESKY_BATCHED")
  ) && identical(
    resource_metrics$qr_checkpoint_wait_count,
    expected_checkpoint("AUGMENTED_QR", planned = TRUE)
  ) && identical(
    resource_metrics$svd_checkpoint_wait_count,
    expected_checkpoint("AUGMENTED_SVD")
  )
  if (!isTRUE(checkpoint_exact)) {
    stop("Phase 3 oracle checkpoint record/wait evidence is invalid",
         call. = FALSE)
  }

  hash_fields <- grep(
    "(sha256|fingerprint)$", names(target_parity), value = TRUE
  )
  error_fields <- grep(
    "_(max_abs_diff|relative_l2)$", names(target_parity), value = TRUE
  )
  finite_fields <- c(
    "coefficient_all_finite", "fitted_all_finite", "residual_all_finite",
    "rss_all_finite", "rhs_all_finite", "output_all_finite"
  )
  output_evidence_exact <-
    all(vapply(hash_fields, function(field) {
      all(grepl("^[0-9a-f]{64}$", target_parity[[field]]))
    }, logical(1L))) &&
    all(vapply(error_fields, function(field) {
      all(is.finite(target_parity[[field]])) &&
        all(target_parity[[field]] >= 0)
    }, logical(1L))) &&
    all(vapply(finite_fields, function(field) {
      identical(target_parity[[field]], rep(TRUE, target_count))
    }, logical(1L))) &&
    all(target_parity$fitted_max_abs_diff < 1e-7) &&
    all(target_parity$fitted_relative_l2 < 1e-7) &&
    all(target_parity$residual_max_abs_diff < 1e-7) &&
    all(target_parity$residual_relative_l2 < 1e-7) &&
    all(target_parity$rhs_max_abs_diff < 1e-12) &&
    all(target_parity$rhs_relative_l2 < 1e-12) &&
    identical(target_parity$rhs_authority,
              rep("cuda-x0-transpose-y", target_count)) &&
    identical(target_parity$full_cuda_data_plane,
              rep(TRUE, target_count)) &&
    identical(target_parity$numeric_reference,
              rep("mgcv-fixed-sp", target_count)) &&
    identical(target_parity$oracle_call_count,
              rep.int(1L, target_count)) &&
    identical(target_parity$coefficient_oracle_phase2_exact,
              rep(TRUE, target_count)) &&
    identical(target_parity$fitted_oracle_phase2_exact,
              rep(TRUE, target_count)) &&
    identical(target_parity$residual_oracle_phase2_exact,
              rep(TRUE, target_count)) &&
    identical(target_parity$cpu_fallback_count,
              integer(target_count)) &&
    identical(target_parity$unknown_fallback_count,
              integer(target_count)) &&
    identical(target_parity$approximate_backend,
              rep(FALSE, target_count)) &&
    identical(target_parity$fallback_type, rep("NONE", target_count)) &&
    identical(target_parity$error_code, rep("NONE", target_count))
  resource_zero_fields <- c(
    "aggregate_penalty_root_d2h_count",
    "aggregate_penalty_root_d2h_bytes",
    "cuda_device_allocation_count_during_solve",
    "cuda_host_allocation_count_during_solve",
    "stream_create_count_during_solve", "event_create_count_during_solve",
    "cublas_handle_create_count_during_solve",
    "cusolver_handle_create_count_during_solve",
    "per_target_allocation_count_after_warmup",
    "per_target_handle_create_count", "workspace_grow_count_after_warmup",
    "stable_workspace_grow_count_after_warmup",
    "cuda_device_synchronize_count", "implicit_residual_d2h_count",
    "implicit_residual_d2h_bytes", "nonfinite_output_count",
    "cpu_fallback_count",
    "unknown_fallback_count", "approximate_backend_count"
  )
  resource_evidence_exact <-
    all(vapply(resource_zero_fields, function(field) {
      all(resource_metrics[[field]] == 0)
    }, logical(1L))) &&
    identical(resource_metrics$rhs_authority,
              rep("cuda-x0-transpose-y", setup_count)) &&
    identical(resource_metrics$full_cuda_data_plane,
              rep(TRUE, setup_count)) &&
    identical(resource_metrics$prepared_handle_create_count,
              rep.int(1L, setup_count)) &&
    identical(resource_metrics$prepared_handle_destroy_count,
              rep.int(1L, setup_count)) &&
    identical(resource_metrics$residual_token_acquire_count,
              rep.int(1L, setup_count)) &&
    identical(resource_metrics$residual_token_release_count,
              rep.int(1L, setup_count)) &&
    identical(resource_metrics$output_slot_acquire_count,
              rep.int(1L, setup_count)) &&
    identical(resource_metrics$output_slot_release_count,
              rep.int(1L, setup_count)) &&
    identical(resource_metrics$output_slot_leased_after_release,
              rep(FALSE, setup_count)) &&
    identical(resource_metrics$shadow_materialize_call_count,
              rep.int(1L, setup_count)) &&
    identical(resource_metrics$shadow_materialize_target_count,
              resource_metrics$target_count) &&
    identical(resource_metrics$invalid_output_init_count,
              rep.int(1L, setup_count)) &&
    identical(resource_metrics$cusolver_deterministic_mode,
              rep("enabled", setup_count)) &&
    identical(resource_metrics$cublas_math_mode,
              rep("pedantic", setup_count)) &&
    identical(resource_metrics$cublas_atomics_mode,
              rep("not_allowed", setup_count)) &&
    identical(resource_metrics$cublas_user_workspace_installed,
              rep(TRUE, setup_count)) &&
    identical(
      resource_metrics$resource_allocation_count_after_solve -
        resource_metrics$resource_allocation_count_before_solve,
      resource_metrics$cuda_device_allocation_count_during_solve +
        resource_metrics$cuda_host_allocation_count_during_solve
    ) && identical(
      resource_metrics$resource_handle_create_count_after_solve -
        resource_metrics$resource_handle_create_count_before_solve,
      resource_metrics$stream_create_count_during_solve +
        resource_metrics$event_create_count_during_solve +
        resource_metrics$cublas_handle_create_count_during_solve +
        resource_metrics$cusolver_handle_create_count_during_solve
    ) &&
    all(resource_metrics$cublas_workspace_bytes > 0) &&
    all(resource_metrics$cublas_workspace_alignment >= 256)
  if (!isTRUE(output_evidence_exact) || !isTRUE(resource_evidence_exact)) {
    stop("Phase 3 oracle output/RHS/resource evidence is invalid",
         call. = FALSE)
  }

  expected_stage_names <- rep(c(
    "phase2_shard_load", "prepared_handle_create", "solve",
    "shadow_materialize", "cmagic_oracle", "release_and_free"
  ), setup_count)
  expected_stage_keys <- rep(
    setup_results$prepared_s_key_sha256, each = 6L
  )
  expected_stage_ordinals <- rep(
    setup_results$setup_ordinal, each = 6L
  )
  stage_exact <- identical(stage_timing$stage, expected_stage_names) &&
    identical(stage_timing$prepared_s_key_sha256, expected_stage_keys) &&
    identical(stage_timing$setup_ordinal,
              as.integer(expected_stage_ordinals)) &&
    all(is.finite(stage_timing$elapsed_ms)) &&
    all(stage_timing$elapsed_ms >= 0)
  if (!isTRUE(stage_exact)) {
    stop("Phase 3 oracle stage timing evidence is invalid", call. = FALSE)
  }
  invisible(TRUE)
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

.fastkpc_full_cuda_phase3_oracle_phase2_load_units <- function(
    setup_results) {
  required <- c("shard_id", "phase2_shard_id")
  clean <- is.data.frame(setup_results) &&
    all(required %in% names(setup_results)) &&
    all(vapply(required, function(field) {
      value <- setup_results[[field]]
      typeof(value) == "integer" && !is.object(value) &&
        is.null(attributes(value)) && !anyNA(value) && all(value >= 0L)
    }, logical(1L)))
  if (!isTRUE(clean)) {
    stop("Phase 3 oracle Phase 2 load-unit rows are malformed",
         call. = FALSE)
  }
  load_unit_rows <- setup_results[, required, drop = FALSE]
  expected_load_rows <- as.integer(!duplicated(load_unit_rows))
  list(
    expected_load_rows = expected_load_rows,
    unique_phase2_shard_count = as.integer(length(unique(
      setup_results$phase2_shard_id
    ))),
    phase2_shard_load_unit_count = as.integer(sum(expected_load_rows))
  )
}

.fastkpc_full_cuda_phase3_oracle_setup_ordinals_exact <- function(
    setup_results) {
  required <- c("shard_id", "setup_ordinal")
  clean <- is.data.frame(setup_results) &&
    all(required %in% names(setup_results)) &&
    all(vapply(required, function(field) {
      value <- setup_results[[field]]
      typeof(value) == "integer" && !is.object(value) &&
        is.null(attributes(value)) && !anyNA(value)
    }, logical(1L))) && all(setup_results$shard_id >= 0L) &&
    all(setup_results$setup_ordinal > 0L)
  if (!isTRUE(clean)) return(FALSE)
  all(vapply(unique(setup_results$shard_id), function(shard_id) {
    selected <- setup_results$shard_id == shard_id
    identical(
      setup_results$setup_ordinal[selected],
      as.integer(seq_len(sum(selected)))
    )
  }, logical(1L)))
}

.fastkpc_full_cuda_phase3_oracle_target_groups <- function(
    setup_keys, target_setup_keys) {
  mapped_setup <- match(target_setup_keys, setup_keys)
  if (anyNA(mapped_setup) ||
      (length(mapped_setup) > 1L && is.unsorted(mapped_setup))) {
    stop("Phase 3 oracle target/setup grouping is invalid", call. = FALSE)
  }
  setup_count <- length(setup_keys)
  target_count <- tabulate(mapped_setup, nbins = setup_count)
  first_index <- if (setup_count == 0L) {
    integer()
  } else {
    starts <- as.integer(cumsum(c(1L, head(target_count, -1L))))
    starts[target_count == 0L] <- NA_integer_
    starts
  }
  offsets <- if (setup_count == 0L) integer() else {
    rep.int(
      as.integer(cumsum(c(0L, head(target_count, -1L)))), target_count
    )
  }
  list(
    mapped_setup = as.integer(mapped_setup),
    target_count = as.integer(target_count),
    first_index = first_index,
    target_ordinal = as.integer(seq_along(mapped_setup) - offsets)
  )
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
  target_groups <- .fastkpc_full_cuda_phase3_oracle_target_groups(
    setup_keys, target_parity$prepared_s_key_sha256
  )
  mapped_setup <- target_groups$mapped_setup
  expected_target_order <- if (target_count == 0L) integer() else order(
    mapped_setup, target_keys,
    method = "radix"
  )
  order_clean <- !anyDuplicated(setup_keys) && !anyDuplicated(target_keys) &&
    identical(setup_keys, sort(setup_keys, method = "radix")) &&
    identical(expected_target_order, seq_len(target_count)) &&
    identical(resource_metrics$prepared_s_key_sha256, setup_keys) &&
    all(stage_timing$prepared_s_key_sha256 %in% setup_keys)
  if (!isTRUE(order_clean) ||
      !identical(setup_results$target_count, target_groups$target_count) ||
      nrow(resource_metrics) != setup_count ||
      nrow(stage_timing) != 6L * setup_count) {
    stop("Phase 3 oracle setup/target/resource row identity mismatch",
         call. = FALSE)
  }
  mapping_exact <- !anyNA(mapped_setup) &&
    identical(
      resource_metrics$shard_id, setup_results$shard_id
    ) && identical(
      resource_metrics$setup_ordinal, setup_results$setup_ordinal
    ) && identical(
      resource_metrics$canonical_setup_rank,
      setup_results$canonical_setup_rank
    ) && identical(
      target_parity$shard_id, setup_results$shard_id[mapped_setup]
    ) && identical(
      target_parity$setup_ordinal,
      setup_results$setup_ordinal[mapped_setup]
    ) && identical(
      target_parity$canonical_setup_rank,
      setup_results$canonical_setup_rank[mapped_setup]
    ) && identical(
      target_parity$null_dim, setup_results$null_dim[mapped_setup]
    ) && identical(
      stage_timing$shard_id,
      rep(setup_results$shard_id, each = 6L)
    )
  if (!isTRUE(mapping_exact)) {
    stop("Phase 3 oracle setup/target rank mapping is invalid",
         call. = FALSE)
  }
  phase2_load_units <-
    .fastkpc_full_cuda_phase3_oracle_phase2_load_units(setup_results)
  expected_phase2_load_rows <- phase2_load_units$expected_load_rows
  unique_phase2_shard_count <-
    phase2_load_units$unique_phase2_shard_count
  phase2_shard_load_unit_count <-
    phase2_load_units$phase2_shard_load_unit_count
  phase2_load_rows_exact <-
    identical(
      setup_results$phase2_shard_load_count, expected_phase2_load_rows
    ) && identical(
      setup_results$phase2_shard_authentication_count,
      expected_phase2_load_rows
    ) && identical(
      resource_metrics$phase2_shard_load_count, expected_phase2_load_rows
    ) && identical(
      resource_metrics$phase2_shard_authentication_count,
      expected_phase2_load_rows
    )
  if (!isTRUE(phase2_load_rows_exact)) {
    stop("Phase 3 oracle Phase 2 shard load rows are not authoritative",
         call. = FALSE)
  }
  if (.fastkpc_full_cuda_phase3_oracle_target_level_stable_sync(
        setup_results, resource_metrics
      ) != 0L) {
    stop("Phase 3 oracle target-level stable sync hard gate failed",
         call. = FALSE)
  }
  .fastkpc_full_cuda_phase3_validate_oracle_row_authority(
    setup_results, target_parity, resource_metrics, stage_timing,
    target_groups = target_groups
  )

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

  route_indicators <- cbind(
    planned_cholesky, planned_qr, planned_svd,
    executed_cholesky, executed_qr, executed_svd,
    cholesky_to_svd, qr_to_svd, stable_reroute
  )
  setup_route_counts <- matrix(
    0L, nrow = setup_count, ncol = length(route_counts),
    dimnames = list(NULL, names(route_counts))
  )
  if (target_count > 0L) {
    reduced <- rowsum(
      matrix(
        as.integer(route_indicators), nrow = target_count,
        dimnames = list(NULL, names(route_counts))
      ),
      mapped_setup, reorder = FALSE
    )
    setup_route_counts[as.integer(rownames(reduced)), ] <- reduced
  }
  setup_route_exact <- all(vapply(names(route_counts), function(field) {
    expected <- as.integer(setup_route_counts[, field])
    identical(setup_results[[field]], expected) &&
      identical(resource_metrics[[field]], expected)
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

  ok_status <- c(
    "OK_CHOLESKY_BATCHED", "OK_CHOLESKY_SINGLE",
    "OK_AUGMENTED_QR", "OK_AUGMENTED_SVD"
  )
  non_ok <- as.integer(sum(!target_parity$solver_status %in% ok_status))
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
    unique_phase2_shard_count = unique_phase2_shard_count,
    phase2_shard_load_unit_count = phase2_shard_load_unit_count,
    phase2_shard_load_count = resource_sum("phase2_shard_load_count"),
    phase2_shard_authentication_count = resource_sum(
      "phase2_shard_authentication_count"
    ),
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
  base <- c(
    "setup_results", "target_parity", "resource_metrics", "stage_timing",
    "fallbacks", "failures", "summary"
  )
  .fastkpc_full_cuda_phase3_exact_named_list(payload, base) ||
    .fastkpc_full_cuda_phase3_exact_named_list(
      payload, c(base, "qualification_dcov_parity")
    )
}

.fastkpc_full_cuda_phase3_validate_oracle_payload <- function(
    payload, expected_setup_keys = NULL, expected_target_rows = NULL) {
  if (!.fastkpc_full_cuda_phase3_is_oracle_payload(payload)) {
    stop("Phase 3 oracle shard payload schema is malformed", call. = FALSE)
  }
  if ("qualification_dcov_parity" %in% names(payload)) {
    records <- payload$qualification_dcov_parity
    if (!is.data.frame(records) ||
        !"logical_sequence_id" %in% names(records) ||
        anyDuplicated(records$logical_sequence_id) || !identical(
          records$logical_sequence_id,
          sort(records$logical_sequence_id, method = "radix")
        )) {
      stop("Phase 3 oracle qualification dCov shard rows are malformed",
           call. = FALSE)
    }
    full_dcov_names <- if (exists(
          "fastkpc_full_cuda_fixed_sp_qualification_dcov_schema",
          mode = "function", inherits = TRUE
        )) {
      fastkpc_full_cuda_fixed_sp_qualification_dcov_schema()$names
    } else {
      character()
    }
    if (nrow(records) > 0L && identical(names(records), full_dcov_names) &&
        exists(
          "fastkpc_full_cuda_fixed_sp_validate_qualification_dcov_frame",
          mode = "function", inherits = TRUE
        )) {
      fastkpc_full_cuda_fixed_sp_validate_qualification_dcov_frame(records)
    }
    .fastkpc_full_cuda_phase3_validate_qualification_dcov(
      records, require_full = FALSE
    )
  }
  expected_supplied <- !is.null(expected_setup_keys) ||
    !is.null(expected_target_rows)
  if (expected_supplied) {
    expected_setup_keys <- .fastkpc_full_cuda_phase3_key_vector(
      expected_setup_keys, "oracle expected setup keys",
      allow_empty = TRUE, allow_empty_list = TRUE
    )
    required_target_fields <-
      .fastkpc_full_cuda_phase3_oracle_descriptor_target_fields()
    target_rows_clean <- is.data.frame(expected_target_rows) &&
      all(required_target_fields %in% names(expected_target_rows)) &&
      !anyDuplicated(names(expected_target_rows)) &&
      typeof(expected_target_rows$prepared_s_key_sha256) == "character" &&
      typeof(expected_target_rows$residual_key_sha256) == "character" &&
      !anyNA(expected_target_rows$prepared_s_key_sha256) &&
      !anyNA(expected_target_rows$residual_key_sha256)
    if (!isTRUE(target_rows_clean)) {
      stop("Phase 3 oracle expected target corpus is malformed",
           call. = FALSE)
    }
    expected_target_groups <- .fastkpc_full_cuda_phase3_oracle_target_groups(
      expected_setup_keys, expected_target_rows$prepared_s_key_sha256
    )
    expected_target_setup_rank <- expected_target_groups$mapped_setup
    expected_target_order <- if (nrow(expected_target_rows) == 0L) {
      integer()
    } else {
      order(
        expected_target_setup_rank,
        expected_target_rows$residual_key_sha256,
        method = "radix"
      )
    }
    expected_field_map <- c(
      prepared_s_key_sha256 = "prepared_s_key_sha256",
      residual_key_sha256 = "residual_key_sha256",
      shard_id = "shard_id",
      canonical_setup_rank = "canonical_setup_rank",
      canonical_target_rank = "canonical_target_rank",
      target = "target",
      null_dim = "null_dim",
      condition = "condition",
      coefficient_rank = "phase1_coefficient_rank",
      planned_route = "planned_route",
      selected_sp_hash = "selected_sp_sha256",
      coefficient_hash = "coefficient_phase2_sha256",
      fitted_hash = "fitted_phase2_sha256",
      residual_hash = "residual_phase2_sha256",
      target_fit_fingerprint = "target_fit_fingerprint"
    )
    target_projection_exact <- all(vapply(
      names(expected_field_map), function(expected_field) {
        payload_field <- unname(expected_field_map[[expected_field]])
        identical(
          payload$target_parity[[payload_field]],
          expected_target_rows[[expected_field]]
        )
      }, logical(1L)
    ))
    expected_setup_rank <- as.integer(
      expected_target_rows$canonical_setup_rank[
        expected_target_groups$first_index
      ]
    )
    expected_phase2_shard_id <- as.integer(
      expected_target_rows$phase2_shard_id[
        expected_target_groups$first_index
      ]
    )
    authenticated_phase2_shard_id <- as.integer(
      (expected_setup_rank - 1L) %%
        fastkpc_full_cuda_fixed_sp_catalog_contract()$shard_count
    )
    expected_setup_projection_exact <-
      !anyNA(expected_setup_rank) && !anyNA(expected_phase2_shard_id) &&
      identical(
        expected_target_rows$canonical_setup_rank,
        expected_setup_rank[expected_target_setup_rank]
      ) && identical(
        expected_target_rows$phase2_shard_id,
        expected_phase2_shard_id[expected_target_setup_rank]
      ) &&
      identical(expected_phase2_shard_id, authenticated_phase2_shard_id) &&
      identical(
        payload$setup_results$canonical_setup_rank, expected_setup_rank
      ) && identical(
        payload$setup_results$phase2_shard_id, expected_phase2_shard_id
      )
    expected_corpus_clean <-
      !anyNA(expected_target_setup_rank) &&
      !anyDuplicated(expected_target_rows$residual_key_sha256) &&
      identical(expected_target_order, seq_len(nrow(expected_target_rows))) &&
      identical(
        payload$setup_results$prepared_s_key_sha256,
        expected_setup_keys
      ) && isTRUE(target_projection_exact) &&
      isTRUE(expected_setup_projection_exact)
    if (!isTRUE(expected_corpus_clean)) {
      stop("Phase 3 oracle payload does not match its expected corpus",
           call. = FALSE)
    }
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
  payload_target_groups <- .fastkpc_full_cuda_phase3_oracle_target_groups(
    payload$setup_results$prepared_s_key_sha256,
    payload$target_parity$prepared_s_key_sha256
  )
  grouped_target_keys <- split(
    payload$target_parity$residual_key_sha256,
    factor(
      payload_target_groups$mapped_setup,
      levels = seq_len(nrow(payload$setup_results))
    )
  )
  setup_target_hashes_exact <- identical(
    payload$setup_results$target_key_set_sha256,
    unname(vapply(
      grouped_target_keys, fastkpc_full_cuda_census_key_set_hash,
      character(1L)
    ))
  )
  expected_phase2_shard_id <- as.integer(
    (payload$setup_results$canonical_setup_rank - 1L) %%
      fastkpc_full_cuda_fixed_sp_catalog_contract()$shard_count
  )
  row_identity_exact <-
    all(grepl(
      "^[0-9a-f]{64}$",
      payload$setup_results$prepared_s_key_sha256
    )) && all(grepl(
      "^[0-9a-f]{64}$", payload$target_parity$residual_key_sha256
    )) &&
    .fastkpc_full_cuda_phase3_oracle_setup_ordinals_exact(
      payload$setup_results
    ) && !anyDuplicated(payload$setup_results$canonical_setup_rank) &&
    all(payload$setup_results$canonical_setup_rank > 0L) &&
    identical(
      payload$setup_results$canonical_setup_rank,
      sort(payload$setup_results$canonical_setup_rank, method = "radix")
    ) && identical(
      payload$setup_results$phase2_shard_id, expected_phase2_shard_id
    ) && identical(
      payload$target_parity$target_ordinal,
      payload_target_groups$target_ordinal
    ) && !anyDuplicated(payload$target_parity$canonical_target_rank) &&
    all(payload$target_parity$canonical_target_rank > 0L) &&
    isTRUE(setup_target_hashes_exact)
  if (!isTRUE(row_identity_exact)) {
    stop("Phase 3 oracle canonical row identity is invalid", call. = FALSE)
  }
  if (recomputed$target_level_stable_sync_count != 0L) {
    stop("Phase 3 oracle target-level stable sync hard gate failed",
         call. = FALSE)
  }
  if (recomputed$cuda_device_synchronize_count != 0L) {
    stop("Phase 3 oracle cudaDeviceSynchronize hard gate failed",
         call. = FALSE)
  }
  if (recomputed$phase2_shard_load_count !=
        recomputed$phase2_shard_load_unit_count ||
      recomputed$phase2_shard_authentication_count !=
        recomputed$phase2_shard_load_unit_count) {
    stop("Phase 3 oracle per-execution-shard Phase 2 load hard gate failed",
         call. = FALSE)
  }
  numeric_and_fallback_gates <-
    recomputed$non_ok_solver_status_count == 0L &&
    recomputed$nonfinite_output_count == 0L &&
    recomputed$cpu_fallback_count == 0L &&
    recomputed$unknown_fallback_count == 0L &&
    recomputed$approximate_backend_count == 0L &&
    recomputed$failure_count == 0L &&
    recomputed$max_fitted_abs_diff < 1e-7 &&
    recomputed$max_fitted_relative_l2 < 1e-7 &&
    recomputed$max_residual_abs_diff < 1e-7 &&
    recomputed$max_residual_relative_l2 < 1e-7 &&
    recomputed$max_rhs_abs_diff < 1e-12 &&
    recomputed$max_rhs_relative_l2 < 1e-12 &&
    recomputed$per_target_allocation_count_after_warmup == 0L &&
    recomputed$per_target_handle_create_count == 0L &&
    recomputed$implicit_residual_d2h_count == 0L &&
    recomputed$output_slot_live_count == 0L
  if (!isTRUE(numeric_and_fallback_gates)) {
    stop("Phase 3 oracle numeric/fallback hard gate failed", call. = FALSE)
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
    context, shard_id, setup_keys, target_rows, catalog,
    scope = "iteration") {
  scope <- .fastkpc_full_cuda_phase3_scope(scope)
  required_helpers <- c(
    "fastkpc_full_cuda_fixed_sp_execute_oracle_setup",
    "fastkpc_full_cuda_fixed_sp_load_oracle_phase2_shards",
    "fastkpc_full_cuda_fixed_sp_batches_from_loaded",
    "fastkpc_full_cuda_fixed_sp_oracle_setup_batch",
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
  qualification <- if (scope %in% c("qualification", "full")) {
    fastkpc_full_cuda_fixed_sp_load_qualification_logical_tests(
      census_dir = catalog$phase1_dir,
      prepared_dir = catalog$phase2_dir,
      data_path = catalog$data_path
    )
  } else {
    NULL
  }
  runtime_info <- fixed_sp_cuda_runtime_info(context)
  fastkpc_full_cuda_fixed_sp_phase3c_validate_runtime_info(
    runtime_info, paste("Phase 3 oracle shard", shard_id, "runtime")
  )

  setup_outputs <- vector("list", length(setup_keys))
  if (length(setup_keys) > 0L) {
    catalog_setup_rank <- match(
      setup_keys, catalog$setup_index$prepared_s_key_sha256
    )
    if (anyNA(catalog_setup_rank)) {
      stop("Phase 3 oracle setup is absent from the Phase 2 index",
           call. = FALSE)
    }
    setup_phase2_ids <- as.integer(
      (catalog_setup_rank - 1L) %% catalog$catalog_contract$shard_count
    )
    for (phase2_shard_id in sort(unique(setup_phase2_ids))) {
      setup_indices <- which(setup_phase2_ids == phase2_shard_id)
      group_setup_keys <- setup_keys[setup_indices]
      group_target_rows <- target_rows[
        target_rows$prepared_s_key_sha256 %in% group_setup_keys,
        , drop = FALSE
      ]
      rownames(group_target_rows) <- NULL
      preload_started <- proc.time()[["elapsed"]]
      phase2 <- fastkpc_full_cuda_fixed_sp_load_oracle_phase2_shards(
        catalog = catalog, setup_keys = group_setup_keys,
        target_rows = group_target_rows
      )
      if (!identical(phase2$phase2_shard_ids, phase2_shard_id) ||
          phase2$phase2_shard_load_count != 1L ||
          phase2$phase2_shard_authentication_count != 1L) {
        stop("Phase 3 oracle source-shard authentication is not singular",
             call. = FALSE)
      }
      selected_scope <- list(
        setup_rows = data.frame(
          prepared_s_key_sha256 = group_setup_keys,
          stringsAsFactors = FALSE
        ),
        target_rows = group_target_rows,
        shard_ids = phase2$phase2_shard_ids
      )
      batches <- fastkpc_full_cuda_fixed_sp_batches_from_loaded(
        catalog, selected_scope, phase2$loaded
      )
      selected_setups <- lapply(
        seq_along(group_setup_keys), function(local_index) {
          setup_key <- group_setup_keys[[local_index]]
          selected_targets <- group_target_rows[
            group_target_rows$prepared_s_key_sha256 == setup_key,
            , drop = FALSE
          ]
          rownames(selected_targets) <- NULL
          fastkpc_full_cuda_fixed_sp_oracle_setup_batch(
            catalog = catalog, setup_key = setup_key,
            target_rows = selected_targets, batch = batches[[setup_key]]
          )
        }
      )
      load_elapsed_ms <- as.double(
        (proc.time()[["elapsed"]] - preload_started) * 1000
      )
      for (local_index in seq_along(group_setup_keys)) {
        setup_index <- setup_indices[[local_index]]
        setup_key <- group_setup_keys[[local_index]]
        selected_targets <- group_target_rows[
          group_target_rows$prepared_s_key_sha256 == setup_key,
          , drop = FALSE
        ]
        rownames(selected_targets) <- NULL
        first_in_source_shard <- local_index == 1L
        setup_outputs[[setup_index]] <-
          fastkpc_full_cuda_fixed_sp_execute_oracle_setup(
            context = context, catalog = catalog, setup_key = setup_key,
            target_rows = selected_targets, shard_id = shard_id,
            setup_ordinal = as.integer(setup_index),
            selected = selected_setups[[local_index]],
            phase2_shard_load_count =
              as.integer(first_in_source_shard),
            phase2_shard_authentication_count =
              as.integer(first_in_source_shard),
            phase2_shard_load_elapsed_ms = if (first_in_source_shard) {
              load_elapsed_ms
            } else {
              0
            },
            shadow_callback = if (is.null(qualification)) NULL else {
              function(setup_key, target_keys, residuals) {
                .fastkpc_full_cuda_phase3_run_setup_qualification_dcov(
                  qualification$logical_tests, target_keys, residuals
                )
              }
            }
          )
      }
      rm(batches, phase2, selected_setups)
    }
    if (any(vapply(setup_outputs, is.null, logical(1L)))) {
      stop("Phase 3 oracle source-shard execution is incomplete",
           call. = FALSE)
    }
  }
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
  if (!is.null(qualification)) {
    qualification_rows <- lapply(
      setup_outputs, `[[`, "shadow_callback_result"
    )
    nonempty <- vapply(qualification_rows, nrow, integer(1L)) > 0L
    qualification_rows <- if (any(nonempty)) {
      do.call(rbind, qualification_rows[nonempty])
    } else {
      .fastkpc_full_cuda_phase3_empty_full_qualification_dcov()
    }
    qualification_rows <- qualification_rows[order(
      qualification_rows$logical_sequence_id, method = "radix"
    ), , drop = FALSE]
    rownames(qualification_rows) <- NULL
    if (anyDuplicated(qualification_rows$logical_sequence_id)) {
      stop("Phase 3 shard qualification dCov ownership is duplicated",
           call. = FALSE)
    }
    payload$qualification_dcov_parity <- qualification_rows
  }
  .fastkpc_full_cuda_phase3_validate_oracle_payload(
    payload, expected_setup_keys = setup_keys,
    expected_target_rows = target_rows
  )
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

.fastkpc_full_cuda_phase3_executed_native_library_sha256 <- function(
    identity, value = NULL) {
  production <- identical(
    identity$schema_version, "full-cuda-ci-phase3-input-identity-v1"
  )
  if (is.null(value)) {
    value <- if (isTRUE(production)) {
      identity$native_library_sha256
    } else {
      fastkpc_full_cuda_census_hash_utf8(
        "full-cuda-ci-phase3-test-executed-native-library-v1"
      )
    }
  }
  if (!.fastkpc_full_cuda_phase3_sha256(value)) {
    stop("executed native library SHA-256 is malformed", call. = FALSE)
  }
  if (isTRUE(production) &&
      !identical(value, identity$native_library_sha256)) {
    stop("executed native library disagrees with runtime attestation",
         call. = FALSE)
  }
  value
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
    shard_count = NULL, stop_after = NULL, progress_hook = NULL,
    executed_native_library_sha256 = NULL) {
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
  runner_lock <-
    .fastkpc_full_cuda_phase3_acquire_shard_runner_lock(output_dir)
  on.exit(
    .fastkpc_full_cuda_phase3_release_shard_runner_lock(runner_lock),
    add = TRUE
  )
  .fastkpc_full_cuda_phase3_refresh_shard_runner_lock(
    runner_lock, .boundary = "runner_start"
  )
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
  executed_native_library_sha256 <-
    .fastkpc_full_cuda_phase3_executed_native_library_sha256(
      identity, executed_native_library_sha256
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
  .fastkpc_full_cuda_phase3_refresh_shard_runner_lock(
    runner_lock, .boundary = "runner_scan_complete"
  )
  reused <- integer()
  reused_binary_sha256 <- character()
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
    recomputable <- FALSE
    if (xor(pair_exists[["rds"]], pair_exists[["summary"]])) {
      unlink(c(paths$rds, paths$summary_json), force = TRUE)
      missing <- c(missing, shard_id)
      next
    }
    if (all(pair_exists)) {
      descriptor <- .fastkpc_full_cuda_phase3_shard_descriptor(
        plan, shard_id
      )
      loaded <- tryCatch(
        .fastkpc_full_cuda_phase3_read_reusable_shard(
          paths = paths,
          sessions_dir = sessions_dir,
          contract = contract,
          descriptor = descriptor,
          plan = plan,
          identity_info = identity_info,
          route_config = route_config
        ),
        fastkpc_phase3_recomputable_shard = function(error) error
      )
      recomputable <- inherits(
        loaded, "fastkpc_phase3_recomputable_shard"
      )
      reusable <- !isTRUE(recomputable)
    }
    if (reusable) {
      reused <- c(reused, shard_id)
      reused_binary_sha256 <- c(
        reused_binary_sha256,
        loaded$envelope$executed_native_library_sha256
      )
    } else if (isTRUE(recomputable)) {
      unlink(c(paths$rds, paths$summary_json), force = TRUE)
      missing <- c(missing, shard_id)
    } else {
      missing <- c(missing, shard_id)
    }
  }
  reused <- as.integer(reused)
  missing <- as.integer(missing)
  reused_binary_sha256 <- unique(reused_binary_sha256)
  if (length(reused_binary_sha256) > 1L) {
    stop("Phase 3 reusable shards mix executed native libraries",
         call. = FALSE)
  }
  if (length(missing) == 0L) {
    .fastkpc_full_cuda_phase3_refresh_shard_runner_lock(
      runner_lock, .boundary = "runner_complete"
    )
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
  if (length(reused_binary_sha256) == 1L && !identical(
        reused_binary_sha256, executed_native_library_sha256
      )) {
    stop("Phase 3 partial resume would mix executed native libraries",
         call. = FALSE)
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
    executed_native_library_sha256 = executed_native_library_sha256,
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
  .fastkpc_full_cuda_phase3_refresh_shard_runner_lock(
    runner_lock, .boundary = "runner_before_runtime"
  )

  context <- NULL
  context_open <- FALSE
  context_destroyed <- FALSE
  run_error <- NULL
  written <- integer()
  destroy_context_once <- function() {
    if (is.null(context) || isTRUE(context_destroyed)) return(NULL)
    context_destroyed <<- TRUE
    context_open <<- FALSE
    destroy_error <- tryCatch({
      runtime_destroy(context)
      NULL
    }, error = function(error) error)
    if (is.null(destroy_error)) {
      session$runtime_context_destroy_count <<- 1L
    }
    destroy_error
  }
  on.exit({
    if (!is.null(context) && !isTRUE(context_destroyed)) {
      cleanup_error <- destroy_context_once()
      if (is.null(cleanup_error) &&
          session$runtime_context_create_count == 1L) {
        try(
          .fastkpc_full_cuda_phase3_atomic_write_session(
            session, sessions_dir
          ),
          silent = TRUE
        )
      }
    }
  }, add = TRUE, after = FALSE)
  tryCatch({
    context <- runtime_create()
    if (is.null(context)) {
      stop("Phase 3 runtime_create returned NULL", call. = FALSE)
    }
    context_open <- TRUE
    session$runtime_context_create_count <- 1L
    .fastkpc_full_cuda_phase3_atomic_write_session(session, sessions_dir)

    for (shard_id in selected) {
      .fastkpc_full_cuda_phase3_refresh_shard_runner_lock(
        runner_lock, .boundary = "runner_before_shard"
      )
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
      oracle_accounting <- if (identical(contract$kind, "oracle_sp")) {
        if (!.fastkpc_full_cuda_phase3_is_oracle_payload(result$payload)) {
          stop("Phase 3 oracle shard executor returned a non-oracle payload",
               call. = FALSE)
        }
        .fastkpc_full_cuda_phase3_validate_oracle_payload(
          result$payload,
          expected_setup_keys = descriptor$setup_keys,
          expected_target_rows = descriptor$target_rows
        )
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
      .fastkpc_full_cuda_phase3_refresh_shard_runner_lock(
        runner_lock, .boundary = "runner_shard_complete"
      )
      if (!is.null(progress_hook)) {
        progress_hook(session = session, event = "shard_complete")
      }
    }
  }, error = function(error) {
    run_error <<- error
  })

  if (context_open) {
    destroy_error <- destroy_context_once()
    if (!is.null(destroy_error) && is.null(run_error)) {
      run_error <- destroy_error
    }
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
  .fastkpc_full_cuda_phase3_refresh_shard_runner_lock(
    runner_lock, .boundary = "runner_complete"
  )
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
      } else if ("logical_sequence_id" %in% fields) {
        if (anyDuplicated(value$logical_sequence_id)) {
          stop("Phase 3 merged payload has duplicate logical test keys",
               call. = FALSE)
        }
        value <- value[order(
          value$logical_sequence_id, method = "radix"
        ), , drop = FALSE]
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
  executed_binary_sha256 <- unique(vapply(
    loaded, function(value) {
      value$envelope$executed_native_library_sha256
    }, character(1L)
  ))
  if (length(executed_binary_sha256) != 1L ||
      !.fastkpc_full_cuda_phase3_sha256(executed_binary_sha256)) {
    stop("Phase 3 shard set does not bind one executed native library",
         call. = FALSE)
  }
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
    executed_native_library_sha256 = executed_binary_sha256,
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

.fastkpc_full_cuda_phase3_oracle_semantics_version <- function() {
  "full-cuda-ci-fixed-sp-oracle-sp-semantics-v1"
}

.fastkpc_full_cuda_phase3_oracle_risk_fields <- function() {
  c(
    "high_condition", "rank_deficient", "nonfinite_metadata",
    "near_constant_target", "near_constant_conditioner", "mgcv_warning",
    "mgcv_nonconverged", "near_alpha"
  )
}

.fastkpc_full_cuda_phase3_validate_test_identity <- function(identity) {
  fields <- .fastkpc_full_cuda_phase3_test_identity_fields()
  if (!.fastkpc_full_cuda_phase3_exact_named_list(
        identity, c(fields, "sha256")
      ) || !identical(
        identity$schema_version,
        "full-cuda-ci-phase3-test-input-identity-v1"
      )) {
    stop("Phase 3 artifact test identity is malformed", call. = FALSE)
  }
  expected <- .fastkpc_full_cuda_phase3_named_hash(
    identity[setdiff(names(identity), "sha256")]
  )
  if (!identical(identity$sha256, expected)) {
    stop("Phase 3 artifact test identity hash mismatch", call. = FALSE)
  }
  invisible(identity)
}

.fastkpc_full_cuda_phase3_validate_artifact_identity <- function(
    identity, require_full = FALSE) {
  if (!is.list(identity) || is.object(identity) || is.null(names(identity)) ||
      anyDuplicated(names(identity))) {
    stop("Phase 3 artifact identity is malformed", call. = FALSE)
  }
  if (identical(
        identity$schema_version, "full-cuda-ci-phase3-input-identity-v1"
      )) {
    fastkpc_full_cuda_phase3_validate_input_identity(identity)
  } else {
    if (isTRUE(require_full)) {
      stop("full Phase 3 artifact requires production identity",
           call. = FALSE)
    }
    .fastkpc_full_cuda_phase3_validate_test_identity(identity)
  }
  invisible(identity)
}

.fastkpc_full_cuda_phase3_identity_json_exact <- function(
    actual, expected, fields = names(expected)) {
  is.list(actual) && !is.object(actual) &&
    identical(names(actual), names(expected)) &&
    typeof(fields) == "character" && !anyNA(fields) &&
    !anyDuplicated(fields) && all(fields %in% names(expected)) &&
    all(vapply(fields, function(field) {
      .fastkpc_full_cuda_phase3_identity_value_equal(
        actual[[field]], expected[[field]]
      )
    }, logical(1L)))
}

.fastkpc_full_cuda_phase3_validate_full_oracle_identity_lineage <- function(
    recorded, discovered) {
  fastkpc_full_cuda_phase3_validate_input_identity(recorded)
  fastkpc_full_cuda_phase3_validate_input_identity(discovered)
  fields <- c(.fastkpc_full_cuda_phase3_stable_identity_fields(), "sha256")
  if (!.fastkpc_full_cuda_phase3_identity_json_exact(
        recorded, discovered, fields = fields
      )) {
    stop("full Phase 3 oracle identity disagrees with canonical lineage",
         call. = FALSE)
  }
  invisible(discovered)
}

.fastkpc_full_cuda_phase3_canonical_oracle_catalog_inputs <- function() {
  list(
    phase0_dir = file.path(
      "fastkpc", "artifacts", "full_cuda_ci", "oracle_351x48_v1"
    ),
    phase1_dir = file.path(
      "fastkpc", "artifacts", "full_cuda_ci",
      "workload_census_351x48_v1"
    ),
    phase2_dir = file.path(
      "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1"
    ),
    data_path = file.path(
      "fastkpc", "artifacts", "kpc_tprs_real_zhu",
      "cancer_RD-causalDiscoveryInput.rds"
    )
  )
}

.fastkpc_full_cuda_phase3_open_canonical_oracle_catalog <- function() {
  inputs <- .fastkpc_full_cuda_phase3_canonical_oracle_catalog_inputs()
  fastkpc_full_cuda_open_fixed_sp_catalog(
    phase0_dir = inputs$phase0_dir,
    phase1_dir = inputs$phase1_dir,
    phase2_dir = inputs$phase2_dir,
    data_path = inputs$data_path,
    require_full = TRUE
  )
}

.fastkpc_full_cuda_phase3_full_oracle_device_id <- function(
    output_dir, expected_identity, device_id) {
  candidate <- device_id
  if (is.null(candidate) && is.list(expected_identity)) {
    candidate <- expected_identity$device_id
  }
  if (is.null(candidate)) {
    manifest <- .fastkpc_full_cuda_phase3_read_json(
      file.path(output_dir, "manifest.json"), "manifest.json"
    )
    if (!is.list(manifest$input_identity)) {
      stop("full Phase 3 oracle manifest has no device identity",
           call. = FALSE)
    }
    candidate <- manifest$input_identity$device_id
  }
  .fastkpc_full_cuda_phase3_whole_scalar(
    candidate, 0L, "full Phase 3 oracle device_id"
  )
}

.fastkpc_full_cuda_phase3_oracle_descriptor_from_parity <- function(
    target_parity) {
  fields <- .fastkpc_full_cuda_phase3_oracle_descriptor_target_fields()
  required <- c(
    "prepared_s_key_sha256", "residual_key_sha256", "shard_id",
    "canonical_setup_rank", "canonical_target_rank", "target", "null_dim",
    "condition", "phase1_coefficient_rank", "planned_route",
    "selected_sp_sha256", "coefficient_phase2_sha256",
    "fitted_phase2_sha256", "residual_phase2_sha256",
    "target_fit_fingerprint"
  )
  if (!is.data.frame(target_parity) ||
      !all(required %in% names(target_parity))) {
    stop("Phase 3 published target parity cannot form a descriptor",
         call. = FALSE)
  }
  value <- data.frame(
    prepared_s_key_sha256 = target_parity$prepared_s_key_sha256,
    residual_key_sha256 = target_parity$residual_key_sha256,
    shard_id = target_parity$shard_id,
    canonical_setup_rank = target_parity$canonical_setup_rank,
    canonical_target_rank = target_parity$canonical_target_rank,
    phase2_shard_id = as.integer(
      (target_parity$canonical_setup_rank - 1L) %%
        fastkpc_full_cuda_fixed_sp_catalog_contract()$shard_count
    ),
    target = target_parity$target,
    null_dim = target_parity$null_dim,
    condition = target_parity$condition,
    coefficient_rank = target_parity$phase1_coefficient_rank,
    planned_route = target_parity$planned_route,
    selected_sp_hash = target_parity$selected_sp_sha256,
    coefficient_hash = target_parity$coefficient_phase2_sha256,
    fitted_hash = target_parity$fitted_phase2_sha256,
    residual_hash = target_parity$residual_phase2_sha256,
    target_fit_fingerprint = target_parity$target_fit_fingerprint,
    stringsAsFactors = FALSE
  )
  value <- value[, fields, drop = FALSE]
  setup_keys <- sort(unique(value$prepared_s_key_sha256), method = "radix")
  value <- value[order(
    match(value$prepared_s_key_sha256, setup_keys),
    value$residual_key_sha256, method = "radix"
  ), , drop = FALSE]
  rownames(value) <- NULL
  value
}

.fastkpc_full_cuda_phase3_oracle_risk_cases <- function(
    target_parity, risk_rows = NULL, catalog = NULL) {
  if (is.null(risk_rows) && !is.null(catalog)) {
    risk_rows <- catalog$inputs$target_fit_metadata
  }
  risk_fields <- .fastkpc_full_cuda_phase3_oracle_risk_fields()
  if (is.null(risk_rows)) {
    risk_rows <- data.frame(
      residual_key_sha256 = target_parity$residual_key_sha256,
      stringsAsFactors = FALSE
    )
    for (field in risk_fields) risk_rows[[field]] <- FALSE
  }
  clean <- is.data.frame(risk_rows) &&
    all(c("residual_key_sha256", risk_fields) %in% names(risk_rows)) &&
    typeof(risk_rows$residual_key_sha256) == "character" &&
    !anyNA(risk_rows$residual_key_sha256) &&
    !anyDuplicated(risk_rows$residual_key_sha256) &&
    all(vapply(risk_fields, function(field) {
      typeof(risk_rows[[field]]) == "logical" &&
        !is.object(risk_rows[[field]]) && !anyNA(risk_rows[[field]])
    }, logical(1L)))
  if (!isTRUE(clean)) {
    stop("Phase 3 oracle risk selector rows are malformed", call. = FALSE)
  }
  matched <- match(target_parity$residual_key_sha256,
                   risk_rows$residual_key_sha256)
  if (anyNA(matched)) {
    stop("Phase 3 oracle risk selectors do not cover target parity",
         call. = FALSE)
  }
  selector_rows <- risk_rows[matched, c("residual_key_sha256", risk_fields),
                             drop = FALSE]
  selected <- Reduce(`|`, selector_rows[risk_fields])
  joined <- cbind(
    selector_rows[selected, , drop = FALSE],
    target_parity[selected, setdiff(
      names(target_parity), "residual_key_sha256"
    ), drop = FALSE]
  )
  joined <- joined[order(joined$residual_key_sha256, method = "radix"),
                   , drop = FALSE]
  rownames(joined) <- NULL
  joined
}

.fastkpc_full_cuda_phase3_validate_oracle_risk_cases <- function(
    value, target_parity, expected = NULL) {
  risk_fields <- .fastkpc_full_cuda_phase3_oracle_risk_fields()
  clean <- is.data.frame(value) &&
    all(c("residual_key_sha256", risk_fields) %in% names(value)) &&
    typeof(value$residual_key_sha256) == "character" &&
    !anyNA(value$residual_key_sha256) &&
    !anyDuplicated(value$residual_key_sha256) &&
    identical(
      value$residual_key_sha256,
      sort(value$residual_key_sha256, method = "radix")
    ) && all(value$residual_key_sha256 %in%
              target_parity$residual_key_sha256) &&
    all(vapply(risk_fields, function(field) {
      typeof(value[[field]]) == "logical" && !anyNA(value[[field]])
    }, logical(1L))) &&
    (nrow(value) == 0L || all(Reduce(`|`, value[risk_fields])))
  if (!isTRUE(clean)) {
    stop("Phase 3 oracle risk-case payload is malformed", call. = FALSE)
  }
  if (!is.null(expected) && !identical(value, expected)) {
    stop("Phase 3 oracle risk-case payload is not Phase 1 derived",
         call. = FALSE)
  }
  invisible(TRUE)
}

.fastkpc_full_cuda_phase3_empty_qualification_dcov <- function() {
  data.frame(
    logical_sequence_id = integer(), residual_key_x = character(),
    residual_key_y = character(), reference_p_value = double(),
    alpha = double(), p_value = double(), near_alpha = logical(),
    backend = character(), low_rank_backend = character(),
    backend_error = logical(), spectra_fallback = logical(),
    p_value_difference = double(),
    absolute_p_value_difference = double(),
    reference_independent = logical(), independent = logical(),
    decision_flip = logical(), stringsAsFactors = FALSE
  )
}

.fastkpc_full_cuda_phase3_empty_full_qualification_dcov <- function() {
  schema <- fastkpc_full_cuda_fixed_sp_qualification_dcov_schema()
  columns <- lapply(schema$names, function(field) {
    switch(
      schema$types[[field]],
      character = character(), integer = integer(), double = double(),
      logical = logical(), list = I(list()),
      stop("unsupported qualification dCov field type", call. = FALSE)
    )
  })
  names(columns) <- schema$names
  structure(columns, class = "data.frame", row.names = integer())
}

.fastkpc_full_cuda_phase3_run_setup_qualification_dcov <- function(
    logical_tests, target_keys, residuals) {
  if (!is.data.frame(logical_tests) || !is.matrix(residuals) ||
      typeof(residuals) != "double" || ncol(residuals) != length(target_keys) ||
      any(!is.finite(residuals)) || anyDuplicated(target_keys)) {
    stop("Phase 3 setup qualification dCov inputs are malformed",
         call. = FALSE)
  }
  selected <- logical_tests$residual_key_x %in% target_keys &
    logical_tests$residual_key_y %in% target_keys
  selected_tests <- logical_tests[selected, , drop = FALSE]
  if (nrow(selected_tests) == 0L) {
    return(.fastkpc_full_cuda_phase3_empty_full_qualification_dcov())
  }
  selected_tests <- selected_tests[order(
    selected_tests$logical_sequence_id, method = "radix"
  ), , drop = FALSE]
  rownames(selected_tests) <- NULL
  target_index <- setNames(seq_along(target_keys), target_keys)
  parity_rows <- fastkpc_full_cuda_fixed_sp_with_legacy_dcov_backend(
    function() {
      rows <- lapply(seq_len(nrow(selected_tests)), function(index) {
        logical_row <- selected_tests[index, , drop = FALSE]
        oracle <- fastkpc_cuda_legacy_dcov_gamma_cpp_oracle(
          as.double(residuals[, target_index[[
            logical_row$residual_key_x[[1L]]
          ]]]),
          as.double(residuals[, target_index[[
            logical_row$residual_key_y[[1L]]
          ]]]),
          numCol = 35L, index = 1
        )
        diagnostic <- fastkpc_full_cuda_fixed_sp_sanitize_dcov_diagnostic(
          oracle$diagnostics
        )
        p_value <- as.double(oracle$p.value)
        alpha <- as.double(logical_row$alpha[[1L]])
        reference <- as.double(logical_row$reference_p_value[[1L]])
        independent <- p_value >= alpha
        row <- data.frame(
          parity_scope = "qualification",
          logical_sequence_id = logical_row$logical_sequence_id[[1L]],
          residual_key_x = logical_row$residual_key_x[[1L]],
          residual_key_y = logical_row$residual_key_y[[1L]],
          index = 1L, numCol = 35L, alpha = alpha,
          reference_p_value = reference, p_value = p_value,
          p_value_drift = p_value - reference,
          absolute_p_value_drift = abs(p_value - reference),
          p_value_exact = identical(p_value, reference),
          reference_signed_alpha_distance = reference - alpha,
          signed_alpha_distance = p_value - alpha,
          reference_decision = logical_row$reference_decision[[1L]],
          reference_independent =
            logical_row$reference_independent[[1L]],
          decision = if (independent) "independent" else "dependent",
          decision_identical = identical(
            independent, logical_row$reference_independent[[1L]]
          ),
          spectra_no_fallback = diagnostic$lowrank_full_eig_count == 0L &&
            diagnostic$lowrank_spectra_failed_count == 0L &&
            diagnostic$lowrank_spectra_fallback_full_eig_count == 0L,
          stringsAsFactors = FALSE
        )
        row$diagnostics <- I(list(diagnostic))
        row
      })
      value <- do.call(rbind, rows)
      rownames(value) <- NULL
      value
    }
  )
  fastkpc_full_cuda_fixed_sp_build_qualification_dcov_records(
    selected_tests, parity_rows
  )
}

.fastkpc_full_cuda_phase3_validate_qualification_dcov <- function(
    value, require_full = FALSE, tolerance = 1e-10) {
  common <- names(.fastkpc_full_cuda_phase3_empty_qualification_dcov())
  clean <- is.data.frame(value) && all(common %in% names(value)) &&
    typeof(value$logical_sequence_id) == "integer" &&
    typeof(value$residual_key_x) == "character" &&
    typeof(value$residual_key_y) == "character" &&
    typeof(value$reference_p_value) == "double" &&
    typeof(value$alpha) == "double" && typeof(value$p_value) == "double" &&
    typeof(value$near_alpha) == "logical" &&
    typeof(value$backend) == "character" &&
    typeof(value$low_rank_backend) == "character" &&
    typeof(value$backend_error) == "logical" &&
    typeof(value$spectra_fallback) == "logical" &&
    typeof(value$p_value_difference) == "double" &&
    typeof(value$absolute_p_value_difference) == "double" &&
    typeof(value$reference_independent) == "logical" &&
    typeof(value$independent) == "logical" &&
    typeof(value$decision_flip) == "logical" &&
    !anyNA(value[common]) && !anyDuplicated(value$logical_sequence_id) &&
    identical(
      value$logical_sequence_id,
      sort(value$logical_sequence_id, method = "radix")
    ) && all(grepl("^[0-9a-f]{64}$", value$residual_key_x)) &&
    all(grepl("^[0-9a-f]{64}$", value$residual_key_y)) &&
    all(is.finite(value$reference_p_value)) &&
    all(is.finite(value$alpha)) && all(is.finite(value$p_value)) &&
    all(value$alpha > 0 & value$alpha < 1) &&
    identical(
      value$p_value_difference,
      value$p_value - value$reference_p_value
    ) && identical(
      value$absolute_p_value_difference,
      abs(value$p_value - value$reference_p_value)
    ) && identical(
      value$reference_independent,
      value$reference_p_value >= value$alpha
    ) && identical(value$independent, value$p_value >= value$alpha) &&
    identical(
      value$decision_flip,
      value$independent != value$reference_independent
    ) && all(value$backend == "cpp") &&
    all(value$low_rank_backend == "spectra") &&
    !any(value$backend_error) && !any(value$spectra_fallback)
  if (!isTRUE(clean)) {
    stop("Phase 3 qualification dCov evidence is malformed", call. = FALSE)
  }
  endpoints <- sort(unique(c(value$residual_key_x, value$residual_key_y)),
                    method = "radix")
  max_difference <- if (nrow(value) == 0L) 0 else {
    max(value$absolute_p_value_difference)
  }
  summary <- list(
    qualification_dcov_logical_test_count = as.integer(nrow(value)),
    qualification_dcov_near_alpha_count =
      as.integer(sum(value$near_alpha)),
    qualification_dcov_unique_residual_key_count =
      as.integer(length(endpoints)),
    qualification_dcov_max_absolute_p_value_difference =
      as.double(max_difference),
    qualification_dcov_decision_flip_count =
      as.integer(sum(value$decision_flip)),
    qualification_dcov_backend_error_count =
      as.integer(sum(value$backend_error)),
    qualification_dcov_spectra_fallback_count =
      as.integer(sum(value$spectra_fallback))
  )
  hard_gate <- is.finite(max_difference) && max_difference < tolerance &&
    summary$qualification_dcov_decision_flip_count == 0L &&
    summary$qualification_dcov_backend_error_count == 0L &&
    summary$qualification_dcov_spectra_fallback_count == 0L
  if (isTRUE(require_full)) {
    hard_gate <- hard_gate &&
      summary$qualification_dcov_logical_test_count == 3808L &&
      summary$qualification_dcov_near_alpha_count == 1478L &&
      summary$qualification_dcov_unique_residual_key_count == 6143L
  }
  if (!isTRUE(hard_gate)) {
    stop("Phase 3 qualification dCov hard gate failed", call. = FALSE)
  }
  summary
}

.fastkpc_full_cuda_phase3_validate_qualification_lineage <- function(
    value, logical_tests, target_keys) {
  fields <- c(
    "logical_sequence_id", "residual_key_x", "residual_key_y",
    "reference_p_value", "alpha", "near_alpha"
  )
  target_clean <- typeof(target_keys) == "character" &&
    !is.object(target_keys) && is.null(attributes(target_keys)) &&
    !anyNA(target_keys) && !anyDuplicated(target_keys) &&
    identical(target_keys, sort(target_keys, method = "radix")) &&
    all(grepl("^[0-9a-f]{64}$", target_keys))
  frame_clean <- is.data.frame(value) && is.data.frame(logical_tests) &&
    all(fields %in% names(value)) && all(fields %in% names(logical_tests)) &&
    nrow(value) == nrow(logical_tests) &&
    all(vapply(fields, function(field) {
      identical(value[[field]], logical_tests[[field]])
    }, logical(1L)))
  endpoints <- if (isTRUE(frame_clean)) {
    sort(unique(c(value$residual_key_x, value$residual_key_y)),
         method = "radix")
  } else {
    character()
  }
  if (!isTRUE(target_clean) || !isTRUE(frame_clean) ||
      !identical(endpoints, target_keys)) {
    stop("Phase 3 qualification dCov canonical lineage mismatch",
         call. = FALSE)
  }
  invisible(TRUE)
}

.fastkpc_full_cuda_phase3_csv_frame <- function(value) {
  if (!is.data.frame(value)) {
    stop("Phase 3 CSV payload must be a data frame", call. = FALSE)
  }
  result <- value
  for (field in names(result)[vapply(result, is.list, logical(1L))]) {
    result[[field]] <- vapply(result[[field]], function(element) {
      if (length(element) == 0L) "" else paste(element, collapse = ";")
    }, character(1L))
  }
  rownames(result) <- NULL
  result
}

.fastkpc_full_cuda_phase3_write_csv <- function(value, path) {
  old_digits <- getOption("digits")
  on.exit(options(digits = old_digits), add = TRUE)
  options(digits = 17L)
  utils::write.table(
    .fastkpc_full_cuda_phase3_csv_frame(value), path,
    sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE,
    qmethod = "double", na = "NA", fileEncoding = "UTF-8"
  )
  invisible(path)
}

.fastkpc_full_cuda_phase3_read_csv <- function(path, label) {
  value <- tryCatch(
    utils::read.csv(
      path, stringsAsFactors = FALSE, check.names = FALSE,
      na.strings = "NA", blank.lines.skip = FALSE
    ),
    error = function(error) {
      stop(label, " is not valid CSV: ", conditionMessage(error),
           call. = FALSE)
    }
  )
  if (!is.data.frame(value) || anyDuplicated(names(value))) {
    stop(label, " CSV frame is malformed", call. = FALSE)
  }
  value
}

.fastkpc_full_cuda_phase3_coerce_csv_like <- function(
    actual, expected, label) {
  if (!is.data.frame(actual) || !is.data.frame(expected) ||
      !identical(names(actual), names(expected)) ||
      nrow(actual) != nrow(expected)) {
    stop(label, " CSV/RDS schema mismatch", call. = FALSE)
  }
  result <- actual
  for (field in names(expected)) {
    template <- expected[[field]]
    if (is.list(template)) {
      next
    }
    result[[field]] <- switch(
      typeof(template),
      character = {
        converted <- as.character(result[[field]])
        converted[is.na(converted) & template == ""] <- ""
        converted
      },
      logical = {
        if (typeof(result[[field]]) != "logical" || anyNA(result[[field]])) {
          stop(label, " logical CSV column is malformed: ", field,
               call. = FALSE)
        }
        as.logical(result[[field]])
      },
      integer = {
        numeric <- suppressWarnings(as.double(result[[field]]))
        if (anyNA(numeric) != anyNA(template) ||
            any(is.finite(numeric) & numeric != floor(numeric)) ||
            any(abs(numeric[is.finite(numeric)]) > .Machine$integer.max)) {
          stop(label, " integer CSV column is malformed: ", field,
               call. = FALSE)
        }
        as.integer(numeric)
      },
      double = as.double(result[[field]]),
      stop(label, " has an unsupported CSV type: ", field, call. = FALSE)
    )
  }
  expected_csv <- .fastkpc_full_cuda_phase3_csv_frame(expected)
  for (field in names(expected)) {
    if (is.list(expected[[field]])) {
      result[[field]] <- as.character(result[[field]])
    } else if (typeof(expected[[field]]) == "double") {
      result[[field]][is.nan(result[[field]])] <- NA_real_
      expected_csv[[field]][is.nan(expected_csv[[field]])] <- NA_real_
    }
  }
  rownames(result) <- NULL
  mismatched <- names(expected_csv)[!vapply(
    names(expected_csv), function(field) {
      if (typeof(expected_csv[[field]]) == "double") {
        isTRUE(all.equal(
          result[[field]], expected_csv[[field]],
          tolerance = 1e-15, check.attributes = FALSE
        ))
      } else {
        identical(result[[field]], expected_csv[[field]])
      }
    }, logical(1L)
  )]
  if (length(mismatched) > 0L) {
    stop(label, " CSV values do not match RDS: ", mismatched[[1L]],
         call. = FALSE)
  }
  invisible(TRUE)
}

.fastkpc_full_cuda_phase3_oracle_runtime_lifecycle <- function(
    output_dir, shard_count) {
  shards_dir <- file.path(output_dir, "shards")
  sessions_dir <- file.path(output_dir, "sessions")
  scan <- .fastkpc_full_cuda_phase3_scan_shards(
    shards_dir, shard_count, require_complete = TRUE
  )
  session_ids <- unique(vapply(scan$expected_ids, function(shard_id) {
    summary <- .fastkpc_full_cuda_phase3_read_json(
      .fastkpc_full_cuda_phase3_shard_paths(
        shards_dir, shard_id
      )$summary_json,
      "Phase 3 shard summary"
    )
    summary$session_id
  }, character(1L)))
  session_ids <- sort(session_ids, method = "radix")
  rows <- lapply(session_ids, function(session_id) {
    session <- .fastkpc_full_cuda_phase3_read_json(
      .fastkpc_full_cuda_phase3_session_path(sessions_dir, session_id),
      "Phase 3 shard session"
    )
    .fastkpc_full_cuda_phase3_validate_session(
      session, require_complete = TRUE
    )
    data.frame(
      session_id = session$session_id,
      requested_shard_count = as.integer(length(
        .fastkpc_full_cuda_phase3_integer_vector(
          session$requested_shard_ids, "requested_shard_ids"
        )
      )),
      completed_shard_count = as.integer(length(
        .fastkpc_full_cuda_phase3_integer_vector(
          session$completed_shard_ids, "completed_shard_ids"
        )
      )),
      runtime_context_create_count =
        as.integer(session$runtime_context_create_count),
      runtime_context_destroy_count =
        as.integer(session$runtime_context_destroy_count),
      prepared_handle_create_count =
        as.integer(session$prepared_handle_create_count),
      prepared_handle_destroy_count =
        as.integer(session$prepared_handle_destroy_count),
      residual_token_acquire_count =
        as.integer(session$residual_token_acquire_count),
      residual_token_release_count =
        as.integer(session$residual_token_release_count),
      output_slot_acquire_count =
        as.integer(session$output_slot_acquire_count),
      output_slot_release_count =
        as.integer(session$output_slot_release_count),
      target_level_stable_sync_count =
        as.integer(session$target_level_stable_sync_count),
      status = session$status,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

.fastkpc_full_cuda_phase3_oracle_input_hashes <- function(identity) {
  selected <- names(identity)[vapply(identity, function(value) {
    .fastkpc_full_cuda_phase3_sha256(value)
  }, logical(1L))]
  selected <- sort(selected, method = "radix")
  data.frame(
    logical_path = selected,
    sha256 = vapply(selected, function(field) identity[[field]],
                   character(1L)),
    stringsAsFactors = FALSE
  )
}

.fastkpc_full_cuda_phase3_oracle_environment_lines <- function(identity) {
  fields <- sort(names(identity), method = "radix")
  unname(vapply(fields, function(field) {
    value <- identity[[field]]
    if (length(value) != 1L || is.object(value) || is.list(value) ||
        is.na(value)) {
      stop("Phase 3 environment identity field is malformed: ", field,
           call. = FALSE)
    }
    paste0(field, "=", as.character(value))
  }, character(1L)))
}

.fastkpc_full_cuda_phase3_oracle_summary_list <- function(
    row_summary, dcov_summary, risk_case_count, shard_count,
    payload_count, executed_native_library_sha256,
    manifest_sha256 = NULL, pass = TRUE) {
  if (!is.data.frame(row_summary) || nrow(row_summary) != 1L ||
      !is.list(dcov_summary)) {
    stop("Phase 3 oracle summary inputs are malformed", call. = FALSE)
  }
  c(
    list(
      artifact_schema_version =
        fastkpc_full_cuda_phase3_oracle_schema_version(),
      artifact_kind = "oracle_sp",
      oracle_semantics_version =
        .fastkpc_full_cuda_phase3_oracle_semantics_version(),
      executed_native_library_sha256 =
        executed_native_library_sha256,
      manifest_sha256 = manifest_sha256,
      shard_count = as.integer(shard_count),
      payload_count = as.integer(payload_count),
      risk_case_count = as.integer(risk_case_count),
      pass = as.logical(pass)
    ),
    as.list(row_summary[1L, , drop = FALSE]),
    dcov_summary
  )
}

.fastkpc_full_cuda_phase3_summary_claims_exact <- function(
    actual, expected) {
  required <- names(expected)
  if (!is.list(actual) || is.object(actual) ||
      !identical(names(actual), required)) {
    return(FALSE)
  }
  all(vapply(required, function(field) {
    expected_value <- expected[[field]]
    if (typeof(expected_value) == "double" && length(expected_value) == 1L &&
        is.finite(expected_value)) {
      isTRUE(all.equal(
        as.double(actual[[field]]), expected_value,
        tolerance = 1e-15, check.attributes = FALSE
      ))
    } else {
      .fastkpc_full_cuda_phase3_identity_value_equal(
        actual[[field]], expected_value
      )
    }
  }, logical(1L)))
}

fastkpc_full_cuda_phase3_publish_oracle_artifact <- function(
    output_dir, setup_keys, target_rows, identity, route_config, scope,
    shard_count = NULL, risk_rows = NULL, qualification_dcov = NULL,
    catalog = NULL, device_id = NULL, command_lines = NULL) {
  scope <- .fastkpc_full_cuda_phase3_scope(scope)
  require_full <- identical(scope, "full")
  .fastkpc_full_cuda_phase3_validate_artifact_identity(
    identity, require_full = require_full
  )
  plan <- fastkpc_full_cuda_phase3_plan_shards(
    setup_keys = setup_keys, target_rows = target_rows, scope = scope,
    shard_count = shard_count
  )
  .fastkpc_full_cuda_phase3_validate_route_for_shards(route_config, scope)
  .fastkpc_full_cuda_phase3_validate_execution_identity(
    identity = identity, route_config = route_config,
    scope = scope, plan = plan
  )
  paths <- fastkpc_full_cuda_phase3_artifact_paths(output_dir, "oracle_sp")
  artifact_lock <-
    .fastkpc_full_cuda_phase3_acquire_oracle_artifact_lock(output_dir)
  on.exit(
    .fastkpc_full_cuda_phase3_release_oracle_artifact_lock(artifact_lock),
    add = TRUE
  )
  completion <- c(
    manifest = file.exists(paths$manifest_json),
    summary = file.exists(paths$summary_json)
  )
  if (all(completion)) {
    validated <- fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
      output_dir = output_dir, expected_identity = identity,
      require_full = require_full, catalog = catalog, device_id = device_id,
      .artifact_lock = artifact_lock
    )
    if (!identical(validated$manifest$scope, scope)) {
      stop("completed Phase 3 oracle artifact scope differs from invocation",
           call. = FALSE)
    }
    return(list(
      status = "reused", validation = validated,
      summary = validated$summary
    ))
  }
  if (any(completion)) {
    unlink(c(paths$manifest_json, paths$summary_json), force = TRUE)
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  .fastkpc_full_cuda_phase3_refresh_oracle_artifact_lock(
    artifact_lock, .boundary = "publication_before_merge"
  )
  merged <- fastkpc_full_cuda_phase3_merge_shards(
    output_dir = output_dir, kind = "oracle_sp",
    setup_keys = setup_keys, target_rows = target_rows,
    identity = identity, route_config = route_config, scope = scope,
    shard_count = shard_count
  )
  .fastkpc_full_cuda_phase3_refresh_oracle_artifact_lock(
    artifact_lock, .boundary = "publication_after_merge"
  )
  executed_native_library_sha256 <-
    merged$executed_native_library_sha256
  if (!.fastkpc_full_cuda_phase3_sha256(
        executed_native_library_sha256
      )) {
    stop("merged Phase 3 executed native library SHA is malformed",
         call. = FALSE)
  }
  payload <- merged$payload
  setup_results <- payload$setup_results
  target_parity_internal <- payload$target_parity
  if (anyDuplicated(setup_results$prepared_s_key_sha256) ||
      anyDuplicated(target_parity_internal$residual_key_sha256)) {
    stop("Phase 3 oracle merge contains duplicate setup or target keys",
         call. = FALSE)
  }
  setup_order <- order(
    setup_results$prepared_s_key_sha256, method = "radix"
  )
  setup_results <- setup_results[setup_order, , drop = FALSE]
  resource_metrics <- payload$resource_metrics[setup_order, , drop = FALSE]
  stage_rank <- match(
    payload$stage_timing$prepared_s_key_sha256,
    setup_results$prepared_s_key_sha256
  )
  stage_timing <- payload$stage_timing[order(
    stage_rank, payload$stage_timing$setup_ordinal, method = "radix"
  ), , drop = FALSE]
  rownames(setup_results) <- rownames(resource_metrics) <-
    rownames(stage_timing) <- NULL
  internal_setup_rank <- match(
    target_parity_internal$prepared_s_key_sha256,
    setup_results$prepared_s_key_sha256
  )
  target_parity_internal <- target_parity_internal[order(
    internal_setup_rank, target_parity_internal$residual_key_sha256,
    method = "radix"
  ), , drop = FALSE]
  rownames(target_parity_internal) <- NULL
  fallbacks <- .fastkpc_full_cuda_phase3_oracle_fallback_rows(
    target_parity_internal, resource_metrics
  )
  failures <- .fastkpc_full_cuda_phase3_oracle_failure_rows(
    target_parity_internal
  )
  row_summary <- fastkpc_full_cuda_phase3_summarize_oracle_rows(
    setup_results, target_parity_internal, resource_metrics,
    stage_timing, fallbacks, failures
  )
  descriptor <- .fastkpc_full_cuda_phase3_oracle_descriptor_from_parity(
    target_parity_internal
  )
  checked_payload <- list(
    setup_results = setup_results,
    target_parity = target_parity_internal,
    resource_metrics = resource_metrics,
    stage_timing = stage_timing,
    fallbacks = fallbacks,
    failures = failures,
    summary = row_summary
  )
  .fastkpc_full_cuda_phase3_validate_oracle_payload(
    checked_payload,
    expected_setup_keys = setup_results$prepared_s_key_sha256,
    expected_target_rows = descriptor
  )

  target_parity <- target_parity_internal[order(
    target_parity_internal$residual_key_sha256, method = "radix"
  ), , drop = FALSE]
  rownames(target_parity) <- NULL
  risk_cases <- .fastkpc_full_cuda_phase3_oracle_risk_cases(
    target_parity, risk_rows = risk_rows, catalog = catalog
  )
  .fastkpc_full_cuda_phase3_validate_oracle_risk_cases(
    risk_cases, target_parity
  )
  if (is.null(qualification_dcov) &&
      "qualification_dcov_parity" %in% names(payload)) {
    qualification_dcov <- payload$qualification_dcov_parity
  }
  if (is.null(qualification_dcov)) {
    qualification_dcov <-
      .fastkpc_full_cuda_phase3_empty_qualification_dcov()
  }
  if (!is.data.frame(qualification_dcov) ||
      !"logical_sequence_id" %in% names(qualification_dcov)) {
    stop("Phase 3 qualification dCov merge input is malformed",
         call. = FALSE)
  }
  qualification_dcov <- qualification_dcov[order(
    qualification_dcov$logical_sequence_id, method = "radix"
  ), , drop = FALSE]
  rownames(qualification_dcov) <- NULL
  dcov_summary <- .fastkpc_full_cuda_phase3_validate_qualification_dcov(
    qualification_dcov, require_full = require_full,
    tolerance = route_config$qualification_dcov_p_tolerance
  )
  runtime_lifecycle <-
    .fastkpc_full_cuda_phase3_oracle_runtime_lifecycle(
      output_dir, plan$shard_count
    )
  input_hashes <- .fastkpc_full_cuda_phase3_oracle_input_hashes(identity)
  if (is.null(command_lines)) {
    command_lines <- paste(c("Rscript", commandArgs(trailingOnly = FALSE)),
                           collapse = " ")
  }
  if (typeof(command_lines) != "character" || length(command_lines) < 1L ||
      anyNA(command_lines) || any(!nzchar(command_lines)) ||
      any(grepl("[\r\n]", command_lines))) {
    stop("Phase 3 oracle command evidence is malformed", call. = FALSE)
  }
  environment_lines <-
    .fastkpc_full_cuda_phase3_oracle_environment_lines(identity)

  payload_keys <- .fastkpc_full_cuda_phase3_payload_keys("oracle_sp")
  values <- list(
    commands_txt = command_lines,
    environment_txt = environment_lines,
    input_hashes_csv = input_hashes,
    route_config_json = route_config,
    runtime_lifecycle_csv = runtime_lifecycle,
    resource_metrics_csv = resource_metrics,
    stage_timing_csv = stage_timing,
    fallbacks_csv = fallbacks,
    failures_csv = failures,
    setup_results_csv = setup_results,
    setup_results_rds = setup_results,
    target_parity_csv = target_parity,
    target_parity_rds = target_parity,
    risk_cases_csv = risk_cases,
    risk_cases_rds = risk_cases,
    qualification_dcov_csv = qualification_dcov,
    qualification_dcov_rds = qualification_dcov
  )
  if (!identical(names(values), payload_keys)) {
    stop("internal Phase 3 oracle payload namespace mismatch",
         call. = FALSE)
  }
  staging_dir <- tempfile(
    paste0(basename(normalizePath(output_dir, mustWork = FALSE)),
           ".phase3-oracle-publish-"),
    tmpdir = dirname(normalizePath(output_dir, mustWork = FALSE))
  )
  if (!dir.create(staging_dir, recursive = FALSE, showWarnings = FALSE)) {
    stop("failed to create Phase 3 oracle staging directory",
         call. = FALSE)
  }
  on.exit(
    unlink(staging_dir, recursive = TRUE, force = TRUE),
    add = TRUE, after = FALSE
  )
  staged_paths <- setNames(
    file.path(staging_dir, vapply(
      paths[payload_keys], basename, character(1L)
    )), payload_keys
  )
  for (key in payload_keys) {
    path <- staged_paths[[key]]
    if (endsWith(key, "_rds")) {
      saveRDS(values[[key]], path, version = 3L)
      if (!identical(readRDS(path), values[[key]])) {
        stop("Phase 3 staged RDS readback mismatch: ", key,
             call. = FALSE)
      }
    } else if (endsWith(key, "_csv")) {
      .fastkpc_full_cuda_phase3_write_csv(values[[key]], path)
      .fastkpc_full_cuda_phase3_coerce_csv_like(
        .fastkpc_full_cuda_phase3_read_csv(path, key),
        values[[key]], key
      )
    } else if (endsWith(key, "_json")) {
      fastkpc_full_cuda_write_json(values[[key]], path)
      .fastkpc_full_cuda_phase3_read_json(path, key)
    } else if (endsWith(key, "_txt")) {
      writeLines(values[[key]], path, useBytes = TRUE)
      if (!identical(readLines(path, warn = FALSE), values[[key]])) {
        stop("Phase 3 staged text readback mismatch: ", key,
             call. = FALSE)
      }
    } else {
      stop("unsupported Phase 3 oracle payload key: ", key,
           call. = FALSE)
    }
  }
  staged_hashes <- vapply(
    staged_paths, fastkpc_full_cuda_census_file_hash, character(1L)
  )
  payload_names <- vapply(paths[payload_keys], basename, character(1L))
  names(staged_hashes) <- payload_names
  expected_setup_hash <- fastkpc_full_cuda_census_key_set_hash(
    setup_results$prepared_s_key_sha256
  )
  manifest <- list(
    artifact_schema_version = fastkpc_full_cuda_phase3_oracle_schema_version(),
    artifact_kind = "oracle_sp",
    oracle_semantics_version =
      .fastkpc_full_cuda_phase3_oracle_semantics_version(),
    executed_native_library_sha256 =
      executed_native_library_sha256,
    scope = scope,
    shard_count = as.integer(plan$shard_count),
    input_identity_schema_version = identity$schema_version,
    input_identity_sha256 = identity$sha256,
    expected_setup_count = as.integer(nrow(setup_results)),
    expected_setup_hash = expected_setup_hash,
    expected_target_count = as.integer(nrow(target_parity)),
    expected_target_hash = fastkpc_full_cuda_census_key_set_hash(
      target_parity$residual_key_sha256
    ),
    payload_names = unname(payload_names),
    publication_order = c(
      unname(payload_names), "manifest.json", "summary.json"
    ),
    payload_file_sha256 = as.list(staged_hashes),
    input_identity = identity
  )
  if (!.fastkpc_full_cuda_phase3_sha256(expected_setup_hash)) {
    stop("internal Phase 3 setup hash is malformed", call. = FALSE)
  }

  for (key in payload_keys) {
    final_path <- paths[[key]]
    if (file.exists(final_path)) unlink(final_path, force = TRUE)
    if (!file.rename(staged_paths[[key]], final_path)) {
      stop("failed to publish Phase 3 oracle payload: ", key,
           call. = FALSE)
    }
  }
  .fastkpc_full_cuda_phase3_refresh_oracle_artifact_lock(
    artifact_lock, .boundary = "publication_payloads_published"
  )
  manifest_temporary <- file.path(staging_dir, "manifest.json")
  fastkpc_full_cuda_write_json(manifest, manifest_temporary)
  if (file.exists(paths$manifest_json)) unlink(paths$manifest_json, force = TRUE)
  if (!file.rename(manifest_temporary, paths$manifest_json)) {
    stop("failed to publish Phase 3 oracle manifest", call. = FALSE)
  }
  manifest_sha256 <- fastkpc_full_cuda_census_file_hash(
    paths$manifest_json
  )
  summary <- .fastkpc_full_cuda_phase3_oracle_summary_list(
    row_summary = row_summary, dcov_summary = dcov_summary,
    risk_case_count = nrow(risk_cases), shard_count = plan$shard_count,
    payload_count = length(payload_keys),
    executed_native_library_sha256 = executed_native_library_sha256,
    manifest_sha256 = manifest_sha256, pass = TRUE
  )
  summary_temporary <- file.path(staging_dir, "summary.json")
  fastkpc_full_cuda_write_json(summary, summary_temporary)
  if (file.exists(paths$summary_json)) unlink(paths$summary_json, force = TRUE)
  if (!file.rename(summary_temporary, paths$summary_json)) {
    stop("failed to publish Phase 3 oracle summary", call. = FALSE)
  }
  .fastkpc_full_cuda_phase3_refresh_oracle_artifact_lock(
    artifact_lock, .boundary = "publication_markers_published"
  )
  unlink(staging_dir, recursive = TRUE, force = TRUE)
  validated <- tryCatch(
    fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
      output_dir = output_dir, expected_identity = identity,
      require_full = require_full, catalog = catalog, device_id = device_id,
      .artifact_lock = artifact_lock
    ),
    error = function(error) {
      unlink(c(paths$manifest_json, paths$summary_json), force = TRUE)
      stop(error)
    }
  )
  list(status = "published", validation = validated, summary = summary)
}

.fastkpc_full_cuda_phase3_coerce_oracle_csv <- function(value, name) {
  schemas <- fastkpc_full_cuda_fixed_sp_oracle_row_schemas()
  if (!name %in% names(schemas) || !is.data.frame(value) ||
      !identical(names(value), names(schemas[[name]]))) {
    stop("Phase 3 oracle CSV schema mismatch: ", name, call. = FALSE)
  }
  schema <- schemas[[name]]
  result <- value
  for (field in names(schema)) {
    type <- unname(schema[[field]])
    result[[field]] <- switch(
      type,
      character = {
        converted <- as.character(result[[field]])
        converted[is.na(converted)] <- ""
        converted
      },
      logical = {
        if (typeof(result[[field]]) != "logical" || anyNA(result[[field]])) {
          stop("Phase 3 oracle logical CSV field is malformed: ", field,
               call. = FALSE)
        }
        as.logical(result[[field]])
      },
      integer = {
        numeric <- suppressWarnings(as.double(result[[field]]))
        finite <- is.finite(numeric)
        if (any(finite & numeric != floor(numeric)) ||
            any(abs(numeric[finite]) > .Machine$integer.max)) {
          stop("Phase 3 oracle integer CSV field is malformed: ", field,
               call. = FALSE)
        }
        as.integer(numeric)
      },
      double = as.double(result[[field]]),
      stop("unsupported Phase 3 oracle CSV field type", call. = FALSE)
    )
  }
  rownames(result) <- NULL
  fastkpc_full_cuda_fixed_sp_validate_oracle_frame(result, name)
}

.fastkpc_full_cuda_phase3_read_rds <- function(path, label) {
  if (!file.exists(path) || dir.exists(path)) {
    stop(label, " is missing", call. = FALSE)
  }
  tryCatch(
    readRDS(path),
    error = function(error) {
      stop(label, " is not readable RDS: ", conditionMessage(error),
           call. = FALSE)
    }
  )
}

.fastkpc_full_cuda_phase3_oracle_immutable_file_hashes <- function(
    paths, payload_keys) {
  selected <- c(
    unlist(paths[payload_keys], use.names = FALSE),
    paths$manifest_json, paths$summary_json
  )
  if (any(!file.exists(selected)) || any(dir.exists(selected))) {
    stop("Phase 3 oracle immutable file set is incomplete", call. = FALSE)
  }
  hashes <- vapply(
    selected, fastkpc_full_cuda_census_file_hash, character(1L)
  )
  names(hashes) <- basename(selected)
  hashes
}

.fastkpc_full_cuda_phase3_validate_completed_oracle_artifact <- function(
    output_dir, expected_identity = NULL, require_full = FALSE,
    catalog = NULL, device_id = NULL, .artifact_lock = NULL) {
  if (!.fastkpc_full_cuda_phase3_bare_scalar(require_full, "logical")) {
    stop("require_full must be a logical scalar", call. = FALSE)
  }
  acquired_lock <- is.null(.artifact_lock)
  if (isTRUE(acquired_lock)) {
    .artifact_lock <-
      .fastkpc_full_cuda_phase3_acquire_oracle_artifact_lock(output_dir)
    on.exit(
      .fastkpc_full_cuda_phase3_release_oracle_artifact_lock(.artifact_lock),
      add = TRUE
    )
  } else {
    .fastkpc_full_cuda_phase3_require_oracle_artifact_lock(
      .artifact_lock, output_dir
    )
  }
  .fastkpc_full_cuda_phase3_refresh_oracle_artifact_lock(
    .artifact_lock, .boundary = "validation_start"
  )
  paths <- fastkpc_full_cuda_phase3_artifact_paths(output_dir, "oracle_sp")
  if (!dir.exists(output_dir)) {
    stop("Phase 3 oracle artifact directory is missing", call. = FALSE)
  }
  manifest <- .fastkpc_full_cuda_phase3_read_json(
    paths$manifest_json, "manifest.json"
  )
  scope <- .fastkpc_full_cuda_phase3_scope(manifest$scope)
  if (isTRUE(require_full) && !identical(scope, "full")) {
    stop("full Phase 3 oracle artifact has the wrong scope", call. = FALSE)
  }
  semantic_full <- identical(scope, "full")
  production_schema <- "full-cuda-ci-phase3-input-identity-v1"
  manifest_production <- identical(
    manifest$input_identity_schema_version, production_schema
  )
  expected_production <- is.list(expected_identity) && identical(
    expected_identity$schema_version, production_schema
  )
  catalog_required <- semantic_full || manifest_production ||
    expected_production
  if (isTRUE(catalog_required)) {
    resolved_device_id <- .fastkpc_full_cuda_phase3_full_oracle_device_id(
      output_dir, expected_identity, device_id
    )
    canonical_catalog <-
      .fastkpc_full_cuda_phase3_open_canonical_oracle_catalog()
    discovered <- fastkpc_full_cuda_phase3_input_identity(
      canonical_catalog, resolved_device_id
    )
    if (!is.null(catalog)) {
      supplied <- fastkpc_full_cuda_phase3_input_identity(
        catalog, resolved_device_id
      )
      .fastkpc_full_cuda_phase3_validate_full_oracle_identity_lineage(
        supplied, discovered
      )
    }
    if (!is.null(expected_identity)) {
      if (isTRUE(expected_production)) {
        .fastkpc_full_cuda_phase3_validate_full_oracle_identity_lineage(
          expected_identity, discovered
        )
      } else {
        .fastkpc_full_cuda_phase3_validate_artifact_identity(
          expected_identity, require_full = semantic_full
        )
        stop("production Phase 3 oracle manifest requires catalog identity",
             call. = FALSE)
      }
    }
    expected_identity <- discovered
    catalog <- canonical_catalog
    device_id <- resolved_device_id
  } else {
    if (xor(is.null(catalog), is.null(device_id))) {
      stop("catalog validation requires catalog and device_id together",
           call. = FALSE)
    }
    if (!is.null(catalog)) {
      discovered <- fastkpc_full_cuda_phase3_input_identity(catalog, device_id)
      if (!is.null(expected_identity) &&
          !identical(expected_identity, discovered)) {
        stop("precomputed Phase 3 identity disagrees with catalog identity",
             call. = FALSE)
      }
      expected_identity <- discovered
    }
  }
  if (is.null(expected_identity)) {
    stop("completed Phase 3 oracle validation requires expected identity",
         call. = FALSE)
  }
  .fastkpc_full_cuda_phase3_validate_artifact_identity(
    expected_identity, require_full = semantic_full
  )
  final_keys <- setdiff(names(paths), c("shards_dir", "sessions_dir"))
  expected_entries <- c(
    basename(unlist(paths[final_keys], use.names = FALSE)),
    basename(paths$shards_dir), basename(paths$sessions_dir)
  )
  actual_entries <- list.files(
    output_dir, all.files = TRUE, no.. = TRUE, full.names = FALSE
  )
  if (!identical(
        sort(actual_entries, method = "radix"),
        sort(expected_entries, method = "radix")
      )) {
    stop("Phase 3 oracle top-level file surface is incomplete or forged",
         call. = FALSE)
  }
  final_paths <- unlist(paths[final_keys], use.names = FALSE)
  if (!dir.exists(paths$shards_dir) || !dir.exists(paths$sessions_dir) ||
      any(!file.exists(final_paths)) || any(dir.exists(final_paths))) {
    stop("Phase 3 oracle artifact path set is incomplete", call. = FALSE)
  }
  summary <- .fastkpc_full_cuda_phase3_read_json(
    paths$summary_json, "summary.json"
  )
  manifest_fields <- c(
    "artifact_schema_version", "artifact_kind",
    "oracle_semantics_version", "executed_native_library_sha256",
    "scope", "shard_count",
    "input_identity_schema_version", "input_identity_sha256",
    "expected_setup_count", "expected_setup_hash",
    "expected_target_count", "expected_target_hash", "payload_names",
    "publication_order", "payload_file_sha256", "input_identity"
  )
  production_identity <- identical(
    expected_identity$schema_version,
    "full-cuda-ci-phase3-input-identity-v1"
  )
  identity_comparison_fields <- if (isTRUE(production_identity)) {
    c(.fastkpc_full_cuda_phase3_stable_identity_fields(), "sha256")
  } else {
    names(expected_identity)
  }
  if (!identical(names(manifest), manifest_fields) ||
      !identical(
        manifest$artifact_schema_version,
        fastkpc_full_cuda_phase3_oracle_schema_version()
      ) || !identical(manifest$artifact_kind, "oracle_sp") ||
      !identical(
        manifest$oracle_semantics_version,
        .fastkpc_full_cuda_phase3_oracle_semantics_version()
      ) || !.fastkpc_full_cuda_phase3_sha256(
        manifest$executed_native_library_sha256
      ) || !identical(
        manifest$input_identity_schema_version,
        expected_identity$schema_version
      ) || !identical(
        manifest$input_identity_sha256, expected_identity$sha256
      ) || !.fastkpc_full_cuda_phase3_identity_json_exact(
        manifest$input_identity, expected_identity,
        fields = identity_comparison_fields
      )) {
    stop("Phase 3 oracle manifest identity/schema is invalid",
         call. = FALSE)
  }
  shard_count <- .fastkpc_full_cuda_phase3_whole_scalar(
    manifest$shard_count, 1L, "manifest shard_count"
  )
  if (identical(scope, "full") && shard_count != 64L) {
    stop("full Phase 3 oracle shard count is not canonical", call. = FALSE)
  }
  payload_keys <- .fastkpc_full_cuda_phase3_payload_keys("oracle_sp")
  payload_names <- unname(vapply(
    paths[payload_keys], basename, character(1L)
  ))
  if (typeof(manifest$payload_names) != "character" ||
      !identical(manifest$payload_names, payload_names) ||
      !identical(
        manifest$publication_order,
        c(payload_names, "manifest.json", "summary.json")
      )) {
    stop("Phase 3 oracle manifest payload order is invalid", call. = FALSE)
  }
  manifest_hashes <- .fastkpc_full_cuda_phase3_named_sha256(
    manifest$payload_file_sha256, "manifest payload hashes"
  )
  if (!identical(names(manifest_hashes), payload_names)) {
    stop("Phase 3 oracle manifest payload hash namespace is invalid",
         call. = FALSE)
  }
  immutable_file_hashes <-
    .fastkpc_full_cuda_phase3_oracle_immutable_file_hashes(
      paths, payload_keys
    )
  actual_hashes <- immutable_file_hashes[payload_names]
  sizes <- as.numeric(file.info(
    unlist(paths[payload_keys], use.names = FALSE)
  )$size)
  if (anyNA(sizes) || any(sizes <= 0) ||
      !identical(actual_hashes, manifest_hashes)) {
    stop("Phase 3 oracle payload byte hash validation failed",
         call. = FALSE)
  }
  .fastkpc_full_cuda_phase3_refresh_oracle_artifact_lock(
    .artifact_lock, .boundary = "validation_payload_hashes"
  )
  route_config <- .fastkpc_full_cuda_phase3_validate_route_file(
    paths$route_config_json, expected_identity$route_config_hash
  )

  setup_results <- .fastkpc_full_cuda_phase3_read_rds(
    paths$setup_results_rds, "setup_results.rds"
  )
  target_parity <- .fastkpc_full_cuda_phase3_read_rds(
    paths$target_parity_rds, "target_parity.rds"
  )
  risk_cases <- .fastkpc_full_cuda_phase3_read_rds(
    paths$risk_cases_rds, "risk_cases.rds"
  )
  qualification_dcov <- .fastkpc_full_cuda_phase3_read_rds(
    paths$qualification_dcov_rds, "qualification_dcov_parity.rds"
  )
  setup_results <- fastkpc_full_cuda_fixed_sp_validate_oracle_frame(
    setup_results, "setup_results"
  )
  target_parity <- fastkpc_full_cuda_fixed_sp_validate_oracle_frame(
    target_parity, "target_parity"
  )
  if (anyDuplicated(setup_results$prepared_s_key_sha256) ||
      anyDuplicated(target_parity$residual_key_sha256) ||
      !identical(
        setup_results$prepared_s_key_sha256,
        sort(setup_results$prepared_s_key_sha256, method = "radix")
      ) || !identical(
        target_parity$residual_key_sha256,
        sort(target_parity$residual_key_sha256, method = "radix")
      )) {
    stop("Phase 3 oracle published key order/uniqueness is invalid",
         call. = FALSE)
  }
  .fastkpc_full_cuda_phase3_coerce_csv_like(
    .fastkpc_full_cuda_phase3_read_csv(
      paths$setup_results_csv, "setup_results.csv"
    ), setup_results, "setup_results"
  )
  .fastkpc_full_cuda_phase3_coerce_csv_like(
    .fastkpc_full_cuda_phase3_read_csv(
      paths$target_parity_csv, "target_parity.csv"
    ), target_parity, "target_parity"
  )
  .fastkpc_full_cuda_phase3_coerce_csv_like(
    .fastkpc_full_cuda_phase3_read_csv(
      paths$risk_cases_csv, "risk_cases.csv"
    ), risk_cases, "risk_cases"
  )
  .fastkpc_full_cuda_phase3_coerce_csv_like(
    .fastkpc_full_cuda_phase3_read_csv(
      paths$qualification_dcov_csv, "qualification_dcov_parity.csv"
    ), qualification_dcov, "qualification_dcov_parity"
  )
  resource_metrics <- .fastkpc_full_cuda_phase3_coerce_oracle_csv(
    .fastkpc_full_cuda_phase3_read_csv(
      paths$resource_metrics_csv, "resource_metrics.csv"
    ), "resource_metrics"
  )
  stage_timing <- .fastkpc_full_cuda_phase3_coerce_oracle_csv(
    .fastkpc_full_cuda_phase3_read_csv(
      paths$stage_timing_csv, "stage_timing.csv"
    ), "stage_timing"
  )
  fallbacks <- .fastkpc_full_cuda_phase3_read_csv(
    paths$fallbacks_csv, "fallbacks.csv"
  )
  failures <- .fastkpc_full_cuda_phase3_read_csv(
    paths$failures_csv, "failures.csv"
  )
  if (!identical(names(fallbacks), c("fallback_type", "count")) ||
      any(!is.finite(fallbacks$count)) ||
      any(fallbacks$count != floor(fallbacks$count))) {
    stop("Phase 3 oracle fallback CSV is malformed", call. = FALSE)
  }
  fallbacks$fallback_type <- as.character(fallbacks$fallback_type)
  fallbacks$count <- as.integer(fallbacks$count)
  failure_fields <- c(
    "prepared_s_key_sha256", "residual_key_sha256", "solver_status",
    "error_code", "error_message_sha256"
  )
  if (!identical(names(failures), failure_fields)) {
    stop("Phase 3 oracle failure CSV is malformed", call. = FALSE)
  }
  for (field in failure_fields) failures[[field]] <- as.character(
    failures[[field]]
  )
  rownames(fallbacks) <- rownames(failures) <- NULL

  setup_rank <- match(
    target_parity$prepared_s_key_sha256,
    setup_results$prepared_s_key_sha256
  )
  target_internal <- target_parity[order(
    setup_rank, target_parity$residual_key_sha256, method = "radix"
  ), , drop = FALSE]
  rownames(target_internal) <- NULL
  descriptor <- .fastkpc_full_cuda_phase3_oracle_descriptor_from_parity(
    target_internal
  )
  plan <- fastkpc_full_cuda_phase3_plan_shards(
    setup_keys = setup_results$prepared_s_key_sha256,
    target_rows = descriptor, scope = scope,
    shard_count = if (identical(scope, "full")) NULL else shard_count
  )
  .fastkpc_full_cuda_phase3_validate_execution_identity(
    identity = expected_identity, route_config = route_config,
    scope = scope, plan = plan
  )
  if (!identical(
        manifest$expected_setup_count, as.integer(nrow(setup_results))
      ) || !identical(
        manifest$expected_setup_hash,
        fastkpc_full_cuda_census_key_set_hash(
          setup_results$prepared_s_key_sha256
        )
      ) || !identical(
        manifest$expected_target_count, as.integer(nrow(target_parity))
      ) || !identical(
        manifest$expected_target_hash,
        fastkpc_full_cuda_census_key_set_hash(
          target_parity$residual_key_sha256
        )
      )) {
    stop("Phase 3 oracle manifest corpus claims are invalid", call. = FALSE)
  }
  if (identical(
        expected_identity$schema_version,
        "full-cuda-ci-phase3-test-input-identity-v1"
      ) && (!identical(
        expected_identity$canonical_setup_corpus_hash,
        manifest$expected_setup_hash
      ) || !identical(
        expected_identity$canonical_target_corpus_hash,
        manifest$expected_target_hash
      ))) {
    stop("Phase 3 oracle test identity corpus mismatch", call. = FALSE)
  }

  authenticated <- fastkpc_full_cuda_phase3_merge_shards(
    output_dir = output_dir, kind = "oracle_sp",
    setup_keys = setup_results$prepared_s_key_sha256,
    target_rows = descriptor, identity = expected_identity,
    route_config = route_config, scope = scope,
    shard_count = if (identical(scope, "full")) NULL else shard_count
  )
  if (!identical(
        manifest$executed_native_library_sha256,
        authenticated$executed_native_library_sha256
      )) {
    stop("Phase 3 manifest executed native library differs from shards",
         call. = FALSE)
  }
  .fastkpc_full_cuda_phase3_refresh_oracle_artifact_lock(
    .artifact_lock, .boundary = "validation_shards_authenticated"
  )
  authenticated_setup <- authenticated$payload$setup_results
  authenticated_resource <- authenticated$payload$resource_metrics
  authenticated_stage <- authenticated$payload$stage_timing
  authenticated_target <- authenticated$payload$target_parity
  authenticated_target <- authenticated_target[order(
    authenticated_target$residual_key_sha256, method = "radix"
  ), , drop = FALSE]
  rownames(authenticated_setup) <- rownames(authenticated_resource) <-
    rownames(authenticated_stage) <- rownames(authenticated_target) <- NULL
  if (!identical(setup_results, authenticated_setup) ||
      !identical(target_parity, authenticated_target) ||
      !identical(resource_metrics, authenticated_resource) ||
      !identical(stage_timing, authenticated_stage)) {
    stop("Phase 3 top-level oracle rows differ from authenticated shards",
         call. = FALSE)
  }
  authenticated_dcov <- if (
      "qualification_dcov_parity" %in% names(authenticated$payload)
    ) {
    value <- authenticated$payload$qualification_dcov_parity
    value <- value[order(value$logical_sequence_id, method = "radix"),
                   , drop = FALSE]
    rownames(value) <- NULL
    value
  } else {
    NULL
  }
  if ((!is.null(authenticated_dcov) &&
       !identical(qualification_dcov, authenticated_dcov)) ||
      (isTRUE(production_identity) && nrow(qualification_dcov) > 0L &&
       is.null(authenticated_dcov)) ||
      (isTRUE(semantic_full) && is.null(authenticated_dcov))) {
    stop("Phase 3 qualification dCov rows differ from authenticated shards",
         call. = FALSE)
  }
  recomputed_fallbacks <- .fastkpc_full_cuda_phase3_oracle_fallback_rows(
    target_internal, resource_metrics
  )
  recomputed_failures <- .fastkpc_full_cuda_phase3_oracle_failure_rows(
    target_internal
  )
  if (!identical(fallbacks, recomputed_fallbacks) ||
      !identical(failures, recomputed_failures)) {
    stop("Phase 3 fallback/failure payload is not row-derived",
         call. = FALSE)
  }
  row_summary <- fastkpc_full_cuda_phase3_summarize_oracle_rows(
    setup_results, target_internal, resource_metrics, stage_timing,
    fallbacks, failures
  )
  .fastkpc_full_cuda_phase3_validate_oracle_payload(
    list(
      setup_results = setup_results, target_parity = target_internal,
      resource_metrics = resource_metrics, stage_timing = stage_timing,
      fallbacks = fallbacks, failures = failures, summary = row_summary
    ),
    expected_setup_keys = setup_results$prepared_s_key_sha256,
    expected_target_rows = descriptor
  )

  expected_risk <- if (is.null(catalog)) NULL else {
    .fastkpc_full_cuda_phase3_oracle_risk_cases(
      target_parity, catalog = catalog
    )
  }
  .fastkpc_full_cuda_phase3_validate_oracle_risk_cases(
    risk_cases, target_parity, expected = expected_risk
  )
  dcov_summary <- .fastkpc_full_cuda_phase3_validate_qualification_dcov(
    qualification_dcov, require_full = semantic_full,
    tolerance = route_config$qualification_dcov_p_tolerance
  )
  if (!is.null(catalog)) {
    if (scope %in% c("qualification", "full")) {
      canonical_dcov <-
        fastkpc_full_cuda_fixed_sp_load_qualification_logical_tests(
          census_dir = catalog$phase1_dir,
          prepared_dir = catalog$phase2_dir,
          data_path = catalog$data_path
        )
      .fastkpc_full_cuda_phase3_validate_qualification_lineage(
        qualification_dcov, canonical_dcov$logical_tests,
        canonical_dcov$endpoint_keys
      )
      if (!identical(
            names(qualification_dcov),
            fastkpc_full_cuda_fixed_sp_qualification_dcov_schema()$names
          )) {
        stop("Phase 3 production qualification dCov schema is incomplete",
             call. = FALSE)
      }
      canonical_dcov_summary <-
        fastkpc_full_cuda_fixed_sp_summarize_qualification_dcov_records(
          qualification_dcov, canonical_dcov$logical_tests,
          canonical_dcov$endpoint_keys
        )
      summary_fields <- c(
        "qualification_dcov_logical_test_count",
        "qualification_dcov_near_alpha_count",
        "qualification_dcov_unique_residual_key_count",
        "qualification_dcov_max_absolute_p_value_difference",
        "qualification_dcov_decision_flip_count",
        "qualification_dcov_backend_error_count",
        "qualification_dcov_spectra_fallback_count"
      )
      if (!all(vapply(summary_fields, function(field) {
            identical(dcov_summary[[field]], canonical_dcov_summary[[field]])
          }, logical(1L))) || canonical_dcov_summary[[
            "qualification_dcov_near_alpha_decision_flip_count"
          ]] != 0L) {
        stop("Phase 3 qualification dCov canonical summary mismatch",
             call. = FALSE)
      }
    }
    selected_scope <- fastkpc_full_cuda_fixed_sp_scope(catalog, scope)
    catalog_descriptor <-
      .fastkpc_full_cuda_phase3_oracle_descriptor_target_rows(
        catalog, selected_scope$target_rows
      )
    catalog_setup_keys <- sort(as.character(
      selected_scope$setup_rows$prepared_s_key_sha256
    ), method = "radix")
    if (!identical(
          catalog_setup_keys, setup_results$prepared_s_key_sha256
        ) || !identical(catalog_descriptor, descriptor)) {
      stop("Phase 3 oracle rows do not match reopened catalog lineage",
           call. = FALSE)
    }
  }

  lifecycle <- .fastkpc_full_cuda_phase3_oracle_runtime_lifecycle(
    output_dir, shard_count
  )
  .fastkpc_full_cuda_phase3_coerce_csv_like(
    .fastkpc_full_cuda_phase3_read_csv(
      paths$runtime_lifecycle_csv, "runtime_lifecycle.csv"
    ), lifecycle, "runtime_lifecycle"
  )
  input_hashes <- .fastkpc_full_cuda_phase3_oracle_input_hashes(
    manifest$input_identity
  )
  .fastkpc_full_cuda_phase3_coerce_csv_like(
    .fastkpc_full_cuda_phase3_read_csv(
      paths$input_hashes_csv, "input_hashes.csv"
    ), input_hashes, "input_hashes"
  )
  command_lines <- readLines(paths$commands_txt, warn = FALSE)
  environment_lines <- readLines(paths$environment_txt, warn = FALSE)
  if (length(command_lines) < 1L || any(!nzchar(command_lines)) ||
      !identical(
        environment_lines,
        .fastkpc_full_cuda_phase3_oracle_environment_lines(
          manifest$input_identity
        )
      )) {
    stop("Phase 3 command/environment evidence is invalid", call. = FALSE)
  }

  full_gate <- !isTRUE(semantic_full) || (
    row_summary$setup_count == 8634L &&
      row_summary$target_count == 110617L &&
      identical(
        as.integer(row_summary[1L, c(
          "planned_cholesky_target_count", "planned_qr_target_count",
          "planned_svd_target_count"
        )]), c(73158L, 4210L, 33249L)
      )
  )
  if (!isTRUE(full_gate)) {
    stop("full Phase 3 oracle canonical gate failed", call. = FALSE)
  }
  if (!.fastkpc_full_cuda_phase3_bare_scalar(summary$pass, "logical") ||
      !identical(summary$pass, TRUE) ||
      !.fastkpc_full_cuda_phase3_sha256(summary$manifest_sha256) ||
      !identical(
        summary$manifest_sha256,
        fastkpc_full_cuda_census_file_hash(paths$manifest_json)
      )) {
    stop("Phase 3 oracle summary completion marker is invalid",
         call. = FALSE)
  }
  expected_summary <- .fastkpc_full_cuda_phase3_oracle_summary_list(
    row_summary = row_summary, dcov_summary = dcov_summary,
    risk_case_count = nrow(risk_cases), shard_count = shard_count,
    payload_count = length(payload_keys),
    executed_native_library_sha256 =
      authenticated$executed_native_library_sha256,
    manifest_sha256 = summary$manifest_sha256, pass = TRUE
  )
  if (!.fastkpc_full_cuda_phase3_summary_claims_exact(
        summary, expected_summary
      )) {
    mismatched <- if (!identical(names(summary), names(expected_summary))) {
      "field_namespace"
    } else {
      names(expected_summary)[!vapply(names(expected_summary), function(field) {
        expected_value <- expected_summary[[field]]
        if (typeof(expected_value) == "double" &&
            length(expected_value) == 1L && is.finite(expected_value)) {
          isTRUE(all.equal(
            as.double(summary[[field]]), expected_value,
            tolerance = 1e-15, check.attributes = FALSE
          ))
        } else {
          .fastkpc_full_cuda_phase3_identity_value_equal(
            summary[[field]], expected_value
          )
        }
      }, logical(1L))][[1L]]
    }
    stop("Phase 3 oracle summary claims are not recomputed: ", mismatched,
         call. = FALSE)
  }
  .fastkpc_full_cuda_phase3_refresh_oracle_artifact_lock(
    .artifact_lock, .boundary = "validation_complete"
  )
  if (!identical(
        .fastkpc_full_cuda_phase3_oracle_immutable_file_hashes(
          paths, payload_keys
        ),
        immutable_file_hashes
      )) {
    stop("Phase 3 oracle immutable files changed during validation",
         call. = FALSE)
  }
  list(
    authenticated = TRUE, complete = TRUE, pass = TRUE,
    computed_contract_pass = TRUE, artifact_kind = "oracle_sp",
    artifact_schema_version =
      fastkpc_full_cuda_phase3_oracle_schema_version(),
    manifest = manifest, summary = summary,
    payload_file_sha256 = actual_hashes,
    input_identity = expected_identity, row_summary = row_summary,
    qualification_dcov_summary = dcov_summary
  )
}
