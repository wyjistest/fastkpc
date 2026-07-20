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
    "runtime_abi", "runtime_abi_hash", "route_config_hash",
    "source_commit", "R_version", "mgcv_version",
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
    source_commit = c("source_commit", "phase3_source_commit"),
    R_version = c("R_version", "r_version"),
    mgcv_version = c("mgcv_version", "MGCV_version")
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
    "source_commit", "R_version", "mgcv_version"
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
  for (field in setdiff(required, c("authenticated", "source_commit",
                                    "R_version", "mgcv_version"))) {
    if (!.fastkpc_full_cuda_phase3_sha256(value[[field]])) {
      stop("Phase 3 catalog lineage hash is malformed: ", field,
           call. = FALSE)
    }
  }
  if (!.fastkpc_full_cuda_phase3_commit(value$source_commit)) {
    stop("Phase 3 catalog lineage source commit is malformed", call. = FALSE)
  }
  .fastkpc_full_cuda_phase3_scalar_text(value$R_version, "R_version")
  .fastkpc_full_cuda_phase3_scalar_text(value$mgcv_version, "mgcv_version")
  value
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
  phase0_summary <- phase0$summary
  phase1_summary <- inputs$summary
  if (!isTRUE(phase0_summary$pass) || !isTRUE(phase1_summary$pass) ||
      !isTRUE(phase2$phase2_complete)) {
    stop("Phase 3 catalog completion gates are not authenticated",
         call. = FALSE)
  }
  phase0_hash <- fastkpc_full_cuda_census_input_contract()$file_hashes[[
    "oracle/manifest.json"
  ]]
  phase1_hash <- fastkpc_full_cuda_prepared_s_input_contract()$file_hashes[[
    "manifest.json"
  ]]
  phase2_hash <- catalog$phase2_file_hashes[["manifest.json"]]
  required <- c(phase0_hash, phase1_hash, phase2_hash,
                inputs$dataset_file_sha256, inputs$dataset_sha256,
                phase2$full_canonical_prepared_s_key_corpus_hash,
                phase2$full_canonical_target_key_corpus_hash,
                phase2$source_commit, phase2$R_version, phase2$mgcv_version)
  if (any(vapply(required, is.null, logical(1L)))) {
    stop("Phase 3 catalog does not expose complete authenticated lineage",
         call. = FALSE)
  }
  list(
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
    source_commit = as.character(phase2$source_commit),
    R_version = as.character(phase2$R_version),
    mgcv_version = as.character(phase2$mgcv_version)
  )
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
    "runtime_abi_hash", "route_config_hash", "cublas_workspace_identity"
  )) {
    if (!.fastkpc_full_cuda_phase3_sha256(identity[[field]])) {
      stop("Phase 3 input identity hash is malformed: ", field,
           call. = FALSE)
    }
  }
  if (!.fastkpc_full_cuda_phase3_commit(identity$source_commit)) {
    stop("Phase 3 input identity source commit is malformed", call. = FALSE)
  }
  for (field in c("runtime_abi", "R_version", "mgcv_version", "gpu_name",
                  "gpu_uuid", "compute_capability", "cusolver_deterministic_mode",
                  "cublas_math_mode", "cublas_atomics_mode")) {
    .fastkpc_full_cuda_phase3_scalar_text(identity[[field]], field)
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
    device_evidence = NULL, ...) {
  dots <- list(...)
  if (length(dots) > 0L) {
    if (is.null(names(dots)) || anyDuplicated(names(dots)) ||
        any(!names(dots) %in% c(
          "catalog_validator", "runtime_probe", "device_probe",
          "validated_catalog", "catalog_callback", "runtime_callback",
          "device_callback", "runtime_info", "device_info",
          "runtime_fixture", "device_fixture"
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
      .fastkpc_full_cuda_phase3_default_catalog_evidence
    } else catalog_evidence,
    "catalog_evidence", catalog
  )
  lineage <- .fastkpc_full_cuda_phase3_validate_lineage(lineage)
  runtime <- resolve(runtime_evidence, "runtime_evidence", device_id)
  device <- resolve(device_evidence, "device_evidence", device_id)
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
    runtime_abi = abi$schema_version,
    runtime_abi_hash = abi$sha256,
    route_config_hash = route$sha256,
    source_commit = lineage$source_commit,
    R_version = lineage$R_version,
    mgcv_version = lineage$mgcv_version,
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
                                "device_evidence"))) {
      stop("unknown Phase 3 artifact validation argument", call. = FALSE)
    }
    if (is.null(expected_identity) && !is.null(dots$identity)) {
      expected_identity <- dots$identity
    }
    if (is.null(expected_identity) && !is.null(dots$catalog)) {
      if (is.null(dots$device_id) || is.null(dots$runtime_evidence) ||
          is.null(dots$device_evidence)) {
        stop("catalog validation requires explicit runtime/device evidence",
             call. = FALSE)
      }
      expected_identity <- fastkpc_full_cuda_phase3_input_identity(
        catalog = dots$catalog,
        device_id = dots$device_id,
        catalog_evidence = dots$catalog_evidence,
        runtime_evidence = dots$runtime_evidence,
        device_evidence = dots$device_evidence
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
    "payload_names", "payload_file_sha256",
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
  if (!identical(actual_payload_hashes, payload_hashes)) {
    stop("Phase 3 artifact payload hash mismatch", call. = FALSE)
  }
  .fastkpc_full_cuda_phase3_validate_route_file(
    paths$route_config_json, expected_identity$route_config_hash
  )

  required_summary <- c(
    "artifact_schema_version", "artifact_kind", "manifest_sha256",
    "shard_count", "payload_count", "pass"
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
      as.integer(summary$payload_count) != length(expected_payload_names) ||
      !.fastkpc_full_cuda_phase3_bare_scalar(summary$pass, "logical") ||
      !isTRUE(summary$pass)) {
    stop("Phase 3 summary completion evidence is invalid", call. = FALSE)
  }
  if (isTRUE(require_full)) {
    if (length(list.files(paths$shards_dir, all.files = TRUE,
                          no.. = TRUE)) == 0L ||
        length(list.files(paths$sessions_dir, all.files = TRUE,
                          no.. = TRUE)) == 0L) {
      stop("full Phase 3 artifact has no shard/session evidence", call. = FALSE)
    }
  }
  list(
    authenticated = TRUE,
    complete = TRUE,
    pass = TRUE,
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
