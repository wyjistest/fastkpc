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
      "max(augmented_rows,null_dim)*sigma_max*double_epsilon",
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
    "runtime_abi", "runtime_abi_hash", "route_config_hash",
    "source_commit", "phase3_source_commit", "R_version",
    "phase3_R_version", "mgcv_version", "phase3_mgcv_version",
    "provenance_schema_version", "provenance_mode",
    "source_closure_schema_version", "source_discovery_semantics",
    "source_closure_count", "source_closure_sha256",
    "execution_snapshot_sha256", "relevant_sources_dirty_or_untracked",
    "execution_sources_unchanged_after_run",
    "cuda_toolkit_version", "cuda_driver_version", "gpu_name", "gpu_uuid",
    "compute_capability_major", "compute_capability_minor",
    "compute_capability", "sm_count", "device_id",
    "cusolver_deterministic_mode", "cublas_math_mode", "cublas_atomics_mode",
    "cublas_user_workspace_installed", "cublas_workspace_bytes",
    "cublas_workspace_alignment", "cublas_workspace_identity", "shard_count",
    "authenticated"
  )
}

.fastkpc_full_cuda_phase3_identity_hash <- function(identity) {
  fields <- .fastkpc_full_cuda_phase3_identity_fields()
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
    "execution_sources_unchanged_after_run"
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
  for (field in c(
    "R_version", "phase3_R_version", "mgcv_version", "phase3_mgcv_version",
    "provenance_schema_version", "provenance_mode",
    "source_closure_schema_version", "source_discovery_semantics"
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
      )) {
    stop("Phase 3 execution provenance schema is unsupported", call. = FALSE)
  }
  if (!.fastkpc_full_cuda_phase3_bare_integer(value$source_closure_count, 1L) ||
      !.fastkpc_full_cuda_phase3_sha256(value$source_closure_sha256) ||
      !.fastkpc_full_cuda_phase3_sha256(value$execution_snapshot_sha256)) {
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
  if (!isTRUE(value$execution_sources_unchanged_after_run)) {
    stop("Phase 3 execution provenance is not post-run verified",
         call. = FALSE)
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
    workload_census = "fastkpc/R/full_cuda_ci_workload_census.R"
  )
  candidates[file.exists(candidates)]
}

.fastkpc_full_cuda_phase3_default_execution_evidence <- function(
    catalog, device_id) {
  supplied <- .fastkpc_full_cuda_phase3_catalog_value(
    catalog,
    c(
      "phase3_execution_evidence", "phase3_execution_provenance",
      "execution_evidence", "execution_provenance"
    )
  )
  if (!is.null(supplied)) {
    if (!is.list(supplied) || is.object(supplied)) {
      stop("Phase 3 supplied execution provenance is malformed",
           call. = FALSE)
    }
    if (!is.null(supplied$head_base_commit)) {
      if (!exists(
            "fastkpc_full_cuda_fixed_sp_verify_execution_provenance",
            mode = "function"
          )) {
        stop("Phase 3C provenance verifier is unavailable", call. = FALSE)
      }
      verifier_input <- supplied
      if ("authenticated" %in% names(verifier_input)) {
        verifier_input$authenticated <- NULL
      }
      supplied <- tryCatch(
        fastkpc_full_cuda_fixed_sp_verify_execution_provenance(verifier_input),
        error = function(error) {
          stop("Phase 3C execution provenance verification failed: ",
               conditionMessage(error), call. = FALSE)
        }
      )
      if (!is.list(supplied) ||
          !isTRUE(supplied$execution_sources_unchanged_after_run)) {
        stop("Phase 3C execution provenance is not post-run verified",
             call. = FALSE)
      }
      supplied$authenticated <- TRUE
      supplied$source_commit <- supplied$head_base_commit
      supplied$phase3_source_commit <- supplied$head_base_commit
      if (!requireNamespace("mgcv", quietly = TRUE)) {
        stop("Phase 3 execution requires installed mgcv", call. = FALSE)
      }
      supplied$R_version <- R.version.string
      supplied$phase3_R_version <- R.version.string
      supplied$mgcv_version <- as.character(utils::packageVersion("mgcv"))
      supplied$phase3_mgcv_version <- supplied$mgcv_version
    }
    if (is.null(supplied$authenticated)) {
      stop("Phase 3 supplied execution provenance is unmarked",
           call. = FALSE)
    }
    return(.fastkpc_full_cuda_phase3_validate_execution_evidence(supplied))
  }
  if (!exists(
        "fastkpc_full_cuda_fixed_sp_discover_execution_source_closure",
        mode = "function"
      ) || !exists(
        "fastkpc_full_cuda_fixed_sp_source_closure_hash",
        mode = "function"
      ) || !exists(
        "fastkpc_full_cuda_fixed_sp_sha256_file",
        mode = "function"
      ) || !exists(
        "fastkpc_full_cuda_fixed_sp_git_source_state",
        mode = "function"
      ) || !exists("fastkpc_full_cuda_source_commit", mode = "function")) {
    stop("authenticated Phase 3 source discovery helpers are unavailable",
         call. = FALSE)
  }
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
  source_states <- vapply(
    closure$source_file_paths,
    fastkpc_full_cuda_fixed_sp_git_source_state,
    character(1L)
  )
  source_commit <- fastkpc_full_cuda_source_commit()
  if (!.fastkpc_full_cuda_phase3_commit(source_commit)) {
    stop("authenticated current Phase 3 source commit is unavailable",
         call. = FALSE)
  }
  if (!requireNamespace("mgcv", quietly = TRUE)) {
    stop("Phase 3 execution requires installed mgcv", call. = FALSE)
  }
  R_version <- R.version.string
  mgcv_version <- as.character(utils::packageVersion("mgcv"))
  source_closure_sha256 <- fastkpc_full_cuda_fixed_sp_source_closure_hash(
    closure, source_hashes
  )
  relevant_dirty <- any(source_states != "clean")
  execution_snapshot_sha256 <- .fastkpc_full_cuda_phase3_named_hash(list(
    provenance_schema_version =
      "full-cuda-ci-execution-source-snapshot-v6",
    provenance_mode = "working-tree-execution-snapshot-v1",
    head_base_commit = source_commit,
    source_closure_sha256 = source_closure_sha256,
    R_version = R_version,
    mgcv_version = mgcv_version,
    relevant_sources_dirty_or_untracked = relevant_dirty
  ))
  .fastkpc_full_cuda_phase3_validate_execution_evidence(list(
    authenticated = TRUE,
    source_commit = source_commit,
    phase3_source_commit = source_commit,
    R_version = R_version,
    phase3_R_version = R_version,
    mgcv_version = mgcv_version,
    phase3_mgcv_version = mgcv_version,
    provenance_schema_version =
      "full-cuda-ci-execution-source-snapshot-v6",
    provenance_mode = "working-tree-execution-snapshot-v1",
    source_closure_schema_version =
      "full-cuda-ci-execution-source-closure-v1",
    source_discovery_semantics =
      "parsed-r-ast-load-time-literal-source-v1",
    source_closure_count = closure$source_closure_count,
    source_closure_sha256 = source_closure_sha256,
    execution_snapshot_sha256 = execution_snapshot_sha256,
    relevant_sources_dirty_or_untracked = relevant_dirty,
    execution_sources_unchanged_after_run = TRUE
  ))
}

fastkpc_full_cuda_phase3_discover_execution_evidence <- function(
    catalog, device_id) {
  .fastkpc_full_cuda_phase3_default_execution_evidence(catalog, device_id)
}

fastkpc_full_cuda_phase3_discover_runtime_evidence <- function(
    catalog, device_id) {
  supplied <- .fastkpc_full_cuda_phase3_catalog_value(
    catalog,
    c(
      "phase3_runtime_evidence", "phase3_runtime_info", "runtime_evidence",
      "runtime_info"
    )
  )
  if (is.null(supplied)) {
    handle <- .fastkpc_full_cuda_phase3_catalog_value(
      catalog, c("phase3_runtime", "runtime_handle")
    )
    if (!is.null(handle) && exists("fixed_sp_cuda_runtime_info",
                                   mode = "function")) {
      supplied <- fixed_sp_cuda_runtime_info(handle)
    }
  }
  if (is.null(supplied)) {
    stop("authenticated Phase 3 runtime evidence is unavailable",
         call. = FALSE)
  }
  if (!is.list(supplied) || is.object(supplied)) {
    stop("Phase 3 runtime evidence is malformed", call. = FALSE)
  }
  supplied
}

fastkpc_full_cuda_phase3_discover_device_evidence <- function(
    catalog, device_id) {
  supplied <- .fastkpc_full_cuda_phase3_catalog_value(
    catalog,
    c("phase3_device_evidence", "phase3_device_info", "device_evidence",
      "device_info")
  )
  if (!is.null(supplied)) {
    if (!is.list(supplied) || is.object(supplied)) {
      stop("Phase 3 device evidence is malformed", call. = FALSE)
    }
    return(supplied)
  }
  if (!exists("fastkpc_cuda_device_info", mode = "function")) {
    stop("authenticated Phase 3 device-query API is unavailable",
         call. = FALSE)
  }
  queried <- fastkpc_cuda_device_info()
  if (!is.list(queried) || is.object(queried)) {
    stop("Phase 3 device-query result is malformed", call. = FALSE)
  }
  if (!is.null(queried$name) && is.null(queried$gpu_name)) {
    queried$gpu_name <- queried$name
  }
  queried
}

.fastkpc_full_cuda_phase3_validate_runtime_evidence <- function(
    runtime, device_id) {
  if (!is.list(runtime) || is.object(runtime) || is.null(names(runtime)) ||
      anyDuplicated(names(runtime))) {
    stop("Phase 3 runtime/device evidence is malformed", call. = FALSE)
  }
  abi <- fastkpc_full_cuda_fixed_sp_runtime_abi()
  required <- c(
    "runtime_abi", "runtime_abi_hash", "device_id",
    "cuda_toolkit_version", "cuda_driver_version", "gpu_name", "gpu_uuid",
    "compute_capability_major", "compute_capability_minor",
    "compute_capability", "sm_count", "cusolver_deterministic_mode",
    "cublas_math_mode", "cublas_atomics_mode",
    "cublas_user_workspace_installed", "cublas_workspace_bytes",
    "cublas_workspace_alignment", "cublas_workspace_identity"
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
  for (field in c("gpu_name", "gpu_uuid", "compute_capability",
                  "cusolver_deterministic_mode", "cublas_math_mode",
                  "cublas_atomics_mode")) {
    .fastkpc_full_cuda_phase3_scalar_text(runtime[[field]], field)
  }
  expected_compute <- paste0(
    runtime$compute_capability_major, ".", runtime$compute_capability_minor
  )
  if (!identical(runtime$compute_capability, expected_compute)) {
    stop("Phase 3 compute capability identity mismatch", call. = FALSE)
  }
  if (!identical(runtime$cusolver_deterministic_mode, "enabled") ||
      !identical(runtime$cublas_math_mode, "pedantic") ||
      !identical(runtime$cublas_atomics_mode, "not_allowed") ||
      !.fastkpc_full_cuda_phase3_bare_scalar(
        runtime$cublas_user_workspace_installed, "logical"
      ) || !isTRUE(runtime$cublas_user_workspace_installed)) {
    stop("Phase 3 deterministic runtime configuration is invalid",
         call. = FALSE)
  }
  for (field in c("cublas_workspace_bytes", "cublas_workspace_alignment")) {
    if (!.fastkpc_full_cuda_phase3_bare_number(runtime[[field]], 0)) {
      stop("Phase 3 workspace evidence is malformed: ", field,
           call. = FALSE)
    }
  }
  if (runtime$cublas_workspace_alignment < 256 ||
      runtime$cublas_workspace_alignment !=
        floor(runtime$cublas_workspace_alignment) ||
      !.fastkpc_full_cuda_phase3_sha256(runtime$cublas_workspace_identity)) {
    stop("Phase 3 cuBLAS workspace identity is invalid", call. = FALSE)
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

.fastkpc_full_cuda_phase3_default_catalog_evidence <- function(catalog) {
  if (!is.list(catalog) || is.null(catalog$phase0) ||
      is.null(catalog$inputs) || is.null(catalog$phase2_manifest)) {
    stop("Phase 3 catalog is not an authenticated Phase 0/1/2 catalog",
         call. = FALSE)
  }
  phase0 <- catalog$phase0
  inputs <- catalog$inputs
  phase2 <- catalog$phase2_manifest
  if (!is.list(phase0) || !is.list(inputs) || !is.list(phase2) ||
      !is.list(phase0$manifest) || !is.list(inputs$manifest)) {
    stop("Phase 3 catalog manifests are malformed", call. = FALSE)
  }
  phase0_summary <- phase0$summary
  phase1_summary <- inputs$summary
  if (!isTRUE(phase0_summary$pass) || !isTRUE(phase1_summary$pass) ||
      !isTRUE(phase2$phase2_complete)) {
    stop("Phase 3 catalog completion gates are not authenticated",
         call. = FALSE)
  }
  phase0_hash <- .fastkpc_full_cuda_phase3_catalog_hash(
    catalog,
    c("phase0_manifest_hash", "phase0_manifest_sha256"),
    fastkpc_full_cuda_census_input_contract()$file_hashes[[
      "oracle/manifest.json"
    ]],
    "Phase 0 manifest hash"
  )
  phase1_hash <- .fastkpc_full_cuda_phase3_catalog_hash(
    catalog,
    c("phase1_manifest_hash", "phase1_manifest_sha256"),
    fastkpc_full_cuda_prepared_s_input_contract()$file_hashes[[
      "manifest.json"
    ]],
    "Phase 1 manifest hash"
  )
  phase2_hash <- catalog$phase2_file_hashes[["manifest.json"]]
  required <- c(
    phase0_hash, phase1_hash, phase2_hash,
    inputs$dataset_file_sha256, inputs$dataset_sha256,
    phase2$full_canonical_prepared_s_key_corpus_hash,
    phase2$full_canonical_target_key_corpus_hash,
    phase0$manifest$source_commit, inputs$manifest$source_commit,
    phase2$source_commit, phase2$R_version, phase2$mgcv_version
  )
  if (any(vapply(required, is.null, logical(1L)))) {
    stop("Phase 3 catalog does not expose complete authenticated lineage",
         call. = FALSE)
  }
  .fastkpc_full_cuda_phase3_validate_lineage(list(
    authenticated = TRUE,
    phase0_manifest_hash = unname(phase0_hash),
    phase1_manifest_hash = unname(phase1_hash),
    phase2_manifest_hash = as.character(phase2_hash),
    dataset_file_sha256 = as.character(inputs$dataset_file_sha256),
    dataset_matrix_sha256 = as.character(inputs$dataset_sha256),
    canonical_setup_corpus_hash = as.character(
      phase2$full_canonical_prepared_s_key_corpus_hash
    ),
    canonical_target_corpus_hash = as.character(
      phase2$full_canonical_target_key_corpus_hash
    ),
    phase0_source_commit = as.character(phase0$manifest$source_commit),
    phase1_source_commit = as.character(inputs$manifest$source_commit),
    phase2_source_commit = as.character(phase2$source_commit),
    phase2_R_version = as.character(phase2$R_version),
    phase2_mgcv_version = as.character(phase2$mgcv_version)
  ))
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
    "runtime_abi_hash", "route_config_hash", "cublas_workspace_identity",
    "source_closure_sha256", "execution_snapshot_sha256"
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
  for (field in c("runtime_abi", "R_version", "mgcv_version", "gpu_name",
                  "gpu_uuid", "compute_capability", "cusolver_deterministic_mode",
                  "cublas_math_mode", "cublas_atomics_mode",
                  "phase2_R_version", "phase2_mgcv_version",
                  "phase3_R_version", "phase3_mgcv_version",
                  "provenance_schema_version", "provenance_mode",
                  "source_closure_schema_version",
                  "source_discovery_semantics")) {
    .fastkpc_full_cuda_phase3_scalar_text(identity[[field]], field)
  }
  if (!identical(identity$R_version, identity$phase3_R_version) ||
      !identical(identity$mgcv_version, identity$phase3_mgcv_version)) {
    stop("Phase 3 input identity execution version aliases disagree",
         call. = FALSE)
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
    "execution_sources_unchanged_after_run"
  )) {
    if (!.fastkpc_full_cuda_phase3_bare_scalar(identity[[field]], "logical")) {
      stop("Phase 3 input identity provenance flag is malformed: ", field,
           call. = FALSE)
    }
  }
  for (field in c("cuda_toolkit_version", "cuda_driver_version",
                  "compute_capability_major", "compute_capability_minor",
                  "sm_count", "device_id", "shard_count")) {
    if (!.fastkpc_full_cuda_phase3_bare_integer(identity[[field]], 0L)) {
      stop("Phase 3 input identity integer is malformed: ", field,
           call. = FALSE)
    }
  }
  for (field in c("cublas_workspace_bytes", "cublas_workspace_alignment")) {
    if (!.fastkpc_full_cuda_phase3_bare_number(identity[[field]], 0)) {
      stop("Phase 3 input identity workspace value is malformed: ", field,
           call. = FALSE)
    }
  }
  if (!identical(identity$runtime_abi,
                 fastkpc_full_cuda_fixed_sp_runtime_abi()$schema_version) ||
      !identical(identity$runtime_abi_hash,
                 fastkpc_full_cuda_fixed_sp_runtime_abi()$sha256) ||
      !identical(identity$cusolver_deterministic_mode, "enabled") ||
      !identical(identity$cublas_math_mode, "pedantic") ||
      !identical(identity$cublas_atomics_mode, "not_allowed") ||
      !isTRUE(identity$cublas_user_workspace_installed) ||
      identity$cublas_workspace_alignment < 256 ||
      !identical(identity$compute_capability,
                 paste0(identity$compute_capability_major, ".",
                        identity$compute_capability_minor)) ||
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
      if (!identical(identity[fields], expected[fields])) {
        stop("Phase 3 input identity does not match expected identity",
             call. = FALSE)
      }
    }
  }
  invisible(TRUE)
}

fastkpc_full_cuda_phase3_input_identity <- function(
    catalog, device_id, catalog_evidence = NULL, runtime_evidence = NULL,
    device_evidence = NULL, execution_evidence = NULL, ...) {
  dots <- list(...)
  if (length(dots) > 0L) {
    if (is.null(names(dots)) || anyDuplicated(names(dots)) ||
        any(!names(dots) %in% c(
          "catalog_validator", "runtime_probe", "device_probe",
          "validated_catalog", "catalog_callback", "runtime_callback",
          "device_callback", "runtime_info", "device_info",
          "runtime_fixture", "device_fixture", "execution_probe",
          "execution_callback", "execution_info", "execution_fixture"
        ))) {
      stop("unknown Phase 3 input identity evidence argument", call. = FALSE)
    }
    if (is.null(catalog_evidence)) {
      catalog_evidence <- dots$catalog_validator
      if (is.null(catalog_evidence)) {
        catalog_evidence <- dots$validated_catalog
      }
      if (is.null(catalog_evidence)) {
        catalog_evidence <- dots$catalog_callback
      }
    }
    if (is.null(runtime_evidence)) runtime_evidence <- dots$runtime_probe
    if (is.null(runtime_evidence)) runtime_evidence <- dots$runtime_callback
    if (is.null(runtime_evidence)) runtime_evidence <- dots$runtime_info
    if (is.null(runtime_evidence)) runtime_evidence <- dots$runtime_fixture
    if (is.null(device_evidence)) device_evidence <- dots$device_probe
    if (is.null(device_evidence)) device_evidence <- dots$device_callback
    if (is.null(device_evidence)) device_evidence <- dots$device_info
    if (is.null(device_evidence)) device_evidence <- dots$device_fixture
    if (is.null(execution_evidence)) {
      execution_evidence <- dots$execution_probe
      if (is.null(execution_evidence)) {
        execution_evidence <- dots$execution_callback
      }
      if (is.null(execution_evidence)) {
        execution_evidence <- dots$execution_info
      }
      if (is.null(execution_evidence)) {
        execution_evidence <- dots$execution_fixture
      }
    }
  }
  resolve <- function(value, argument, input) {
    if (is.null(value)) {
      stop(argument, " is required for authenticated Phase 3 identity",
           call. = FALSE)
    }
    result <- if (is.function(value)) value(input) else value
    if (!is.list(result) || is.object(result)) {
      stop(argument, " returned malformed evidence", call. = FALSE)
    }
    result
  }
  lineage <- resolve(
    if (is.null(catalog_evidence)) {
      fastkpc_full_cuda_phase3_discover_catalog_evidence
    } else catalog_evidence,
    "catalog_evidence", catalog
  )
  lineage <- .fastkpc_full_cuda_phase3_validate_lineage(lineage)
  execution <- if (is.null(execution_evidence)) {
    fastkpc_full_cuda_phase3_discover_execution_evidence(catalog, device_id)
  } else if (is.function(execution_evidence)) {
    execution_evidence(catalog, device_id)
  } else {
    execution_evidence
  }
  execution <- .fastkpc_full_cuda_phase3_validate_execution_evidence(
    execution
  )
  runtime <- if (is.null(runtime_evidence)) {
    fastkpc_full_cuda_phase3_discover_runtime_evidence(catalog, device_id)
  } else if (is.function(runtime_evidence)) {
    runtime_evidence(device_id)
  } else {
    runtime_evidence
  }
  device <- if (is.null(device_evidence)) {
    fastkpc_full_cuda_phase3_discover_device_evidence(catalog, device_id)
  } else if (is.function(device_evidence)) {
    device_evidence(device_id)
  } else {
    device_evidence
  }
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
    cuda_toolkit_version = runtime$cuda_toolkit_version,
    cuda_driver_version = runtime$cuda_driver_version,
    gpu_name = runtime$gpu_name,
    gpu_uuid = runtime$gpu_uuid,
    compute_capability_major = runtime$compute_capability_major,
    compute_capability_minor = runtime$compute_capability_minor,
    compute_capability = runtime$compute_capability,
    sm_count = runtime$sm_count,
    device_id = runtime$device_id,
    cusolver_deterministic_mode = runtime$cusolver_deterministic_mode,
    cublas_math_mode = runtime$cublas_math_mode,
    cublas_atomics_mode = runtime$cublas_atomics_mode,
    cublas_user_workspace_installed = runtime$cublas_user_workspace_installed,
    cublas_workspace_bytes = runtime$cublas_workspace_bytes,
    cublas_workspace_alignment = runtime$cublas_workspace_alignment,
    cublas_workspace_identity = runtime$cublas_workspace_identity,
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
    ...) {
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
  dots <- list(...)
  if (length(dots) > 0L) {
    if (is.null(names(dots)) || anyDuplicated(names(dots)) ||
      any(!names(dots) %in% c("identity", "catalog", "device_id",
                                "catalog_evidence", "runtime_evidence",
                                "device_evidence", "execution_evidence"))) {
      stop("unknown Phase 3 artifact validation argument", call. = FALSE)
    }
    if (is.null(expected_identity) && !is.null(dots$identity)) {
      expected_identity <- dots$identity
    }
    if (is.null(expected_identity) && !is.null(dots$catalog)) {
      if (is.null(dots$device_id)) {
        stop("catalog validation requires device_id", call. = FALSE)
      }
      expected_identity <- fastkpc_full_cuda_phase3_input_identity(
        catalog = dots$catalog,
        device_id = dots$device_id,
        catalog_evidence = dots$catalog_evidence,
        runtime_evidence = dots$runtime_evidence,
        device_evidence = dots$device_evidence,
        execution_evidence = dots$execution_evidence
      )
    }
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
    output_dir, expected_identity = NULL, require_full = FALSE, ...) {
  fastkpc_full_cuda_phase3_validate_artifact(
    output_dir = output_dir, kind = "oracle_sp",
    expected_identity = expected_identity, require_full = require_full, ...
  )
}

fastkpc_validate_full_cuda_fixed_sp_shadow_artifact <- function(
    output_dir, expected_identity = NULL, require_full = FALSE, ...) {
  fastkpc_full_cuda_phase3_validate_artifact(
    output_dir = output_dir, kind = "full_shadow",
    expected_identity = expected_identity, require_full = require_full, ...
  )
}
