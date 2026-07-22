source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/cuda_native.R")
fastkpc_cuda_dll_paths <- function(dlls = getLoadedDLLs()) {
  paths <- unname(vapply(
    dlls, function(dll) normalizePath(
      dll[["path"]], winslash = "/", mustWork = FALSE
    ), character(1L)
  ))
  paths[grepl("fastkpc_cuda", paths)]
}
cuda_dll_paths_before <- fastkpc_cuda_dll_paths()
if (length(cuda_dll_paths_before) != 0L) {
  stop(
    paste0(
      "contract fixture requires a fresh Rscript with zero fastkpc_cuda DLLs ",
      "before source: ", paste(cuda_dll_paths_before, collapse = ",")
    ),
    call. = FALSE
  )
}
if (file.exists("fastkpc/R/full_cuda_ci_phase3_artifacts.R")) {
  source("fastkpc/R/full_cuda_ci_phase3_artifacts.R")
}

assert_formals <- function(fun, expected, message) {
  assert_true(identical(names(formals(fun)), expected), message)
}

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) {
  if (!isTRUE(value)) fail(message)
}
assert_error <- function(expression, message) {
  error <- tryCatch({
    force(expression)
    NULL
  }, error = function(error) error)
  assert_true(inherits(error, "error"), message)
}

assert_formals(
  fastkpc_full_cuda_phase3_input_identity,
  c("catalog", "device_id"),
  "Phase 3 input identity exposes only catalog and device_id"
)
assert_formals(
  fastkpc_full_cuda_phase3_validate_artifact,
  c("output_dir", "kind", "expected_identity", "require_full", "catalog",
    "device_id"),
  "Phase 3 artifact validation exposes no evidence injection arguments"
)

oracle_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  tempfile("phase3-oracle-"), kind = "oracle_sp"
)
shadow_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  tempfile("phase3-shadow-"), kind = "full_shadow"
)

assert_true(identical(
  fastkpc_full_cuda_phase3_oracle_schema_version(),
  "full-cuda-ci-fixed-sp-oracle-sp-artifact-v1"
), "oracle artifact schema")
assert_true(identical(
  fastkpc_full_cuda_phase3_shadow_schema_version(),
  "full-cuda-ci-fixed-sp-shadow-artifact-v1"
), "shadow artifact schema")

assert_true(all(c(
  "manifest_json", "summary_json", "commands_txt", "environment_txt",
  "input_hashes_csv", "route_config_json", "runtime_lifecycle_csv",
  "resource_metrics_csv", "stage_timing_csv", "fallbacks_csv",
  "failures_csv", "shards_dir", "sessions_dir"
) %in% names(oracle_paths)), "common Phase 3 paths")
assert_true(all(c(
  "setup_results_csv", "setup_results_rds", "target_parity_csv",
  "target_parity_rds", "risk_cases_csv", "risk_cases_rds",
  "qualification_dcov_csv", "qualification_dcov_rds"
) %in% names(oracle_paths)), "oracle payload paths")
assert_true(all(c(
  "logical_ci_parity_csv", "logical_ci_parity_rds",
  "deletion_trace_csv", "sepset_agreement_csv", "n_edgetests_csv",
  "adjacency_rds", "first_divergence_json", "direct_ci_rds",
  "direct_ci_summary_json"
) %in% names(shadow_paths)), "shadow payload paths")
assert_true(identical(
  names(oracle_paths),
  c(
    "manifest_json", "summary_json", "commands_txt", "environment_txt",
    "input_hashes_csv", "route_config_json", "runtime_lifecycle_csv",
    "resource_metrics_csv", "stage_timing_csv", "fallbacks_csv",
    "failures_csv", "setup_results_csv", "setup_results_rds",
    "target_parity_csv", "target_parity_rds", "risk_cases_csv",
    "risk_cases_rds", "qualification_dcov_csv", "qualification_dcov_rds",
    "shards_dir", "sessions_dir"
  )
), "oracle path set is exact")
assert_true(identical(
  names(shadow_paths),
  c(
    "manifest_json", "summary_json", "commands_txt", "environment_txt",
    "input_hashes_csv", "route_config_json", "runtime_lifecycle_csv",
    "resource_metrics_csv", "stage_timing_csv", "fallbacks_csv",
    "failures_csv", "logical_ci_parity_csv", "logical_ci_parity_rds",
    "deletion_trace_csv", "sepset_agreement_csv", "n_edgetests_csv",
    "adjacency_rds", "first_divergence_json", "direct_ci_rds",
    "direct_ci_summary_json", "shards_dir", "sessions_dir"
  )
), "shadow path set is exact")

route <- fastkpc_full_cuda_phase3_route_config()
route_fields <- c(
  "schema_version", "condition_lt_1e8", "condition_1e8_to_lt_1e12",
  "condition_ge_1e12", "rank_deficient", "unauthenticated",
  "svd_rank_tolerance", "residual_tolerance", "fitted_tolerance",
  "qualification_dcov_p_tolerance", "dcov_backend", "reroute_policy",
  "cpu_fallback_allowed", "approximate_backend_allowed", "shard_count",
  "sha256"
)
assert_true(identical(names(route), route_fields),
            "route contract field order")
assert_true(identical(route$schema_version,
                      "full-cuda-ci-fixed-sp-route-v1"),
            "route contract schema")
assert_true(identical(route$condition_lt_1e8, "CHOLESKY_BATCHED") &&
              identical(route$condition_1e8_to_lt_1e12, "AUGMENTED_QR") &&
              identical(route$condition_ge_1e12, "AUGMENTED_SVD") &&
              identical(route$rank_deficient, "AUGMENTED_SVD") &&
              identical(route$unauthenticated, "AUGMENTED_SVD"),
            "route condition policy")
assert_true(identical(
  route$svd_rank_tolerance,
  "max(augmented_rows,null_dim)*sigma_max*double_epsilon"
) && identical(route$residual_tolerance, 1e-7) &&
              identical(route$fitted_tolerance, 1e-7) &&
              identical(route$qualification_dcov_p_tolerance, 1e-10),
            "route tolerances")
assert_true(identical(route$dcov_backend, "legacy-cpp-spectra") &&
              identical(
                route$reroute_policy,
                "declared-cuda-svd-reroute-with-conservation"
              ) && identical(route$cpu_fallback_allowed, FALSE) &&
              identical(route$approximate_backend_allowed, FALSE) &&
              identical(route$shard_count, 64L),
            "route execution policy")
assert_true(identical(
  route$sha256,
  fastkpc_full_cuda_census_named_metadata_hash(
    route[setdiff(names(route), "sha256")]
  )
), "route canonical SHA-256")

sha <- function(label) fastkpc_full_cuda_census_hash_utf8(label)
lineage_fixture <- list(
  authenticated = TRUE,
  phase0_manifest_hash = sha("phase0-manifest"),
  phase1_manifest_hash = sha("phase1-manifest"),
  phase2_manifest_hash = sha("phase2-manifest"),
  dataset_file_sha256 = sha("dataset-file"),
  dataset_matrix_sha256 = sha("dataset-matrix"),
  canonical_setup_corpus_hash = sha("canonical-setup-corpus"),
  canonical_target_corpus_hash = sha("canonical-target-corpus"),
  phase0_source_commit = strrep("0", 40L),
  phase1_source_commit = strrep("1", 40L),
  phase2_source_commit = "42ef3efa08327056ffe5c9aad7a8953ff6864c7e",
  phase2_R_version = "R version 4.4.1 (fixture)",
  phase2_mgcv_version = "1.9.1",
  source_commit = "42ef3efa08327056ffe5c9aad7a8953ff6864c7e",
  R_version = "R version 4.4.1 (fixture)",
  mgcv_version = "1.9.1"
)
runtime_abi <- fastkpc_full_cuda_fixed_sp_runtime_abi()
policy_fixture <- .fastkpc_full_cuda_phase3_policy_contract()
runtime_fixture <- list(
  runtime_abi = runtime_abi$schema_version,
  runtime_abi_hash = runtime_abi$sha256,
  device_id = 2L,
  cuda_toolkit_version = 12040L,
  cuda_driver_version = 55054L,
  gpu_name = "Fixture GPU",
  gpu_uuid = paste0("GPU-", strrep("b", 32L)),
  compute_capability_major = 8L,
  compute_capability_minor = 0L,
  compute_capability = "8.0",
  sm_count = 108L,
  cusolver_deterministic_mode_required =
    policy_fixture$cusolver_deterministic_mode_required,
  cublas_math_mode_required = policy_fixture$cublas_math_mode_required,
  cublas_atomics_mode_required = policy_fixture$cublas_atomics_mode_required,
  cublas_user_workspace_required =
    policy_fixture$cublas_user_workspace_required,
  cublas_workspace_bytes_required =
    policy_fixture$cublas_workspace_bytes_required,
  cublas_workspace_min_alignment_required =
    policy_fixture$cublas_workspace_min_alignment_required,
  runtime_policy_schema_version =
    policy_fixture$configuration_schema_version
)
static_environment_fixture <- list(
  schema_version = policy_fixture$schema_version,
  runtime_abi_schema_version = runtime_abi$schema_version,
  configuration_schema_version = policy_fixture$configuration_schema_version,
  device_id = 2L,
  cuda_toolkit_version = 12040L,
  cuda_driver_version = 55054L,
  gpu_name = "Fixture GPU",
  gpu_uuid = paste0("GPU-", strrep("a", 32L)),
  compute_capability_major = 8L,
  compute_capability_minor = 0L,
  sm_count = 108L,
  cusolver_deterministic_mode_required =
    policy_fixture$cusolver_deterministic_mode_required,
  cublas_math_mode_required = policy_fixture$cublas_math_mode_required,
  cublas_atomics_mode_required = policy_fixture$cublas_atomics_mode_required,
  cublas_user_workspace_required =
    policy_fixture$cublas_user_workspace_required,
  cublas_workspace_bytes_required =
    policy_fixture$cublas_workspace_bytes_required,
  cublas_workspace_min_alignment_required =
    policy_fixture$cublas_workspace_min_alignment_required
)
runtime_attestation_info_fixture <- list(
  device_id = 2L,
  gpu_name = "Fixture GPU",
  gpu_uuid = paste0("GPU-", strrep("b", 32L)),
  runtime_abi_schema_version = runtime_abi$schema_version,
  configuration_schema_version = policy_fixture$configuration_schema_version,
  create_symbol_image_path = "/tmp/fastkpc_cuda.so",
  create_symbol_device_major_hex = "8",
  create_symbol_device_minor_hex = "1",
  create_symbol_inode = "42",
  info_symbol_image_path = "/tmp/fastkpc_cuda.so",
  info_symbol_device_major_hex = "8",
  info_symbol_device_minor_hex = "1",
  info_symbol_inode = "42",
  cuda_toolkit_version = 12040L,
  cuda_driver_version = 55054L,
  compute_capability_major = 8L,
  compute_capability_minor = 0L,
  sm_count = 108L,
  cusolver_deterministic_mode = "enabled",
  cublas_math_mode = "pedantic",
  cublas_atomics_mode = "not_allowed",
  cublas_user_workspace_installed = TRUE,
  cublas_workspace_bytes = 16 * 1024 * 1024,
  cublas_workspace_alignment = 256,
  freed = FALSE
)
validated_runtime_attestation_info <-
  .fastkpc_full_cuda_phase3_validate_runtime_attestation_info(
    runtime_attestation_info_fixture, 2L
  )
assert_true(
  identical(
    validated_runtime_attestation_info[c(
      "gpu_uuid", "runtime_abi_schema_version",
      "configuration_schema_version", "create_symbol_image_path",
      "create_symbol_device_major_hex", "create_symbol_device_minor_hex",
      "create_symbol_inode", "info_symbol_image_path",
      "info_symbol_device_major_hex", "info_symbol_device_minor_hex",
      "info_symbol_inode"
    )],
    runtime_attestation_info_fixture[c(
      "gpu_uuid", "runtime_abi_schema_version",
      "configuration_schema_version", "create_symbol_image_path",
      "create_symbol_device_major_hex", "create_symbol_device_minor_hex",
      "create_symbol_inode", "info_symbol_image_path",
      "info_symbol_device_major_hex", "info_symbol_device_minor_hex",
      "info_symbol_inode"
    )]
  ),
  "runtime attestation validator preserves context-native identity fields"
)
runtime_attestation_missing_uuid <- runtime_attestation_info_fixture
runtime_attestation_missing_uuid$gpu_uuid <- NULL
assert_error(
  .fastkpc_full_cuda_phase3_validate_runtime_attestation_info(
    runtime_attestation_missing_uuid, 2L
  ),
  "runtime attestation info missing context GPU UUID must fail"
)
runtime_attestation_empty_symbol_path <- runtime_attestation_info_fixture
runtime_attestation_empty_symbol_path$create_symbol_image_path <- ""
assert_error(
  .fastkpc_full_cuda_phase3_validate_runtime_attestation_info(
    runtime_attestation_empty_symbol_path, 2L
  ),
  "runtime attestation info with empty native symbol path must fail"
)
catalog_fixture <- list(
  phase3_lineage = lineage_fixture
)

static_probe_calls <- 0L
old_static_probe <- if (exists(
  "fastkpc_cuda_phase3_environment_identity", envir = .GlobalEnv,
  inherits = FALSE
)) get("fastkpc_cuda_phase3_environment_identity", envir = .GlobalEnv) else NULL
old_runtime_discoverer <- get(
  "fastkpc_full_cuda_phase3_discover_runtime_evidence", envir = .GlobalEnv
)
old_runtime_create <- get("fixed_sp_cuda_runtime_create", envir = .GlobalEnv)
assign(
  "fastkpc_cuda_phase3_environment_identity",
  function(device_id) {
    static_probe_calls <<- static_probe_calls + 1L
    static_environment_fixture
  },
  envir = .GlobalEnv
)
assign(
  "fixed_sp_cuda_runtime_create",
  function(device_id) stop("runtime context creation is forbidden in probe"),
  envir = .GlobalEnv
)
assign(
  "fastkpc_full_cuda_phase3_discover_runtime_evidence",
  old_runtime_discoverer,
  envir = .GlobalEnv
)
static_runtime_evidence <- fastkpc_full_cuda_phase3_discover_runtime_evidence(
  catalog_fixture, 2L
)
assert_true(
  static_probe_calls == 1L &&
    identical(static_runtime_evidence$device_id, 2L),
  "default runtime discovery uses static environment query without context"
)
if (is.null(old_static_probe)) {
  rm("fastkpc_cuda_phase3_environment_identity", envir = .GlobalEnv)
} else {
  assign("fastkpc_cuda_phase3_environment_identity", old_static_probe,
         envir = .GlobalEnv)
}
assign("fixed_sp_cuda_runtime_create", old_runtime_create, envir = .GlobalEnv)

current_mgcv_version <- if (requireNamespace("mgcv", quietly = TRUE)) {
  as.character(utils::packageVersion("mgcv"))
} else {
  fail("contract test requires the installed mgcv version")
}
qualified_native_discoverer_original <- get(
  "fastkpc_full_cuda_phase3_discover_qualified_native_evidence",
  envir = .GlobalEnv
)
on.exit(
  assign(
    "fastkpc_full_cuda_phase3_discover_qualified_native_evidence",
    qualified_native_discoverer_original,
    envir = .GlobalEnv
  ),
  add = TRUE
)
qualified_native_fixture_path <- tempfile(
  "phase3-qualified-native-fixture-", fileext = .Platform$dynlib.ext
)
writeBin(charToRaw("qualified-native-fixture"), qualified_native_fixture_path)
on.exit(unlink(qualified_native_fixture_path, force = TRUE), add = TRUE)
qualified_native_fixture_identity <- .fastkpc_cuda_posix_file_identity(
  qualified_native_fixture_path
)
qualified_native_fixture <- list(
  schema_version = "full-cuda-ci-phase3-qualified-native-cache-v1",
  provenance = list(
    head_base_commit = fastkpc_full_cuda_source_commit(),
    provenance_schema_version = "full-cuda-ci-execution-source-snapshot-v6",
    provenance_mode = "working-tree-execution-snapshot-v1",
    source_closure_schema_version =
      "full-cuda-ci-execution-source-closure-v1",
    source_discovery_semantics =
      "parsed-r-ast-load-time-literal-source-v1",
    source_closure_count = 7L,
    source_closure_sha256 = sha("qualified-source-closure"),
    execution_snapshot_sha256 = sha("qualified-execution-snapshot"),
    relevant_sources_dirty_or_untracked = TRUE,
    native_library_identity =
      "qualified-pinned-inode-sha-exact-registered-mapped-path-v3",
    native_library_path = qualified_native_fixture_identity$path,
    native_library_device_major_hex =
      qualified_native_fixture_identity$device_major_hex,
    native_library_device_minor_hex =
      qualified_native_fixture_identity$device_minor_hex,
    native_library_inode = qualified_native_fixture_identity$inode,
    native_library_sha256 = sha("qualified-native-library"),
    native_build_inputs_sha256 = sha("qualified-build-inputs"),
    native_build_dependencies = list(
      schema_version = "full-cuda-ci-native-build-dependencies-v3",
      trace_semantics = "linux-strace-successful-read-exec-evidence-v3",
      trace_invocation = "fixture qualified trace invocation",
      tracer_path = "/usr/bin/strace",
      dependency_count = 17L,
      exclusion_count = 2L,
      aggregate_sha256 = sha("qualified-build-dependencies"),
      trace_sha256 = sha("qualified-build-trace"),
      tracer_sha256 = sha("qualified-build-tracer")
    )
  )
)
assign(
  "fastkpc_full_cuda_phase3_discover_qualified_native_evidence",
  function() qualified_native_fixture,
  envir = .GlobalEnv
)
default_execution_capture <- fastkpc_full_cuda_phase3_discover_execution_evidence(
  catalog_fixture, 2L
)
assert_true(
  identical(default_execution_capture$execution_sources_unchanged_after_run,
            FALSE) &&
    identical(default_execution_capture$execution_provenance_state,
              "pre-run-capture"),
  "pre-run source capture is not labeled post-run verified"
)
execution_fixture <- default_execution_capture
execution_fixture$execution_sources_unchanged_after_run <- TRUE
execution_fixture$execution_provenance_state <- "post-run-verified"
default_discoverer_calls <- new.env(parent = emptyenv())
default_discoverer_calls$catalog <- 0L
default_discoverer_calls$runtime <- 0L
default_discoverer_calls$device <- 0L
default_discoverer_calls$execution <- 0L
old_discoverers <- lapply(
  c(
    "fastkpc_full_cuda_phase3_discover_catalog_evidence",
    "fastkpc_full_cuda_phase3_discover_runtime_evidence",
    "fastkpc_full_cuda_phase3_discover_device_evidence",
    "fastkpc_full_cuda_phase3_discover_execution_evidence"
  ),
  function(name) {
    if (exists(name, envir = .GlobalEnv, inherits = FALSE)) {
      get(name, envir = .GlobalEnv, inherits = FALSE)
    } else {
      NULL
    }
  }
)
names(old_discoverers) <- c(
  "fastkpc_full_cuda_phase3_discover_catalog_evidence",
  "fastkpc_full_cuda_phase3_discover_runtime_evidence",
  "fastkpc_full_cuda_phase3_discover_device_evidence",
  "fastkpc_full_cuda_phase3_discover_execution_evidence"
)
restore_discoverers <- function() {
  for (name in names(old_discoverers)) {
    if (is.null(old_discoverers[[name]])) {
      if (exists(name, envir = .GlobalEnv, inherits = FALSE)) {
        rm(list = name, envir = .GlobalEnv)
      }
    } else {
      assign(name, old_discoverers[[name]], envir = .GlobalEnv)
    }
  }
}
on.exit(restore_discoverers(), add = TRUE)
assign(
  "fastkpc_full_cuda_phase3_discover_catalog_evidence",
  function(catalog) {
    default_discoverer_calls$catalog <-
      default_discoverer_calls$catalog + 1L
    catalog$phase3_lineage
  },
  envir = .GlobalEnv
)
assign(
  "fastkpc_full_cuda_phase3_discover_runtime_evidence",
  function(catalog, device_id) {
    default_discoverer_calls$runtime <-
      default_discoverer_calls$runtime + 1L
    runtime_fixture
  },
  envir = .GlobalEnv
)
assign(
  "fastkpc_full_cuda_phase3_discover_device_evidence",
  function(catalog, device_id) {
    default_discoverer_calls$device <-
      default_discoverer_calls$device + 1L
    runtime_fixture
  },
  envir = .GlobalEnv
)
assign(
  "fastkpc_full_cuda_phase3_discover_execution_evidence",
  function(catalog, device_id) {
    default_discoverer_calls$execution <-
      default_discoverer_calls$execution + 1L
    execution_fixture
  },
  envir = .GlobalEnv
)
default_identity <- fastkpc_full_cuda_phase3_input_identity(
  catalog_fixture, 2L
)
identity <- default_identity
local_session_identity <- default_identity
local_session_identity$native_library_path <-
  "/tmp/fastkpc-cross-process-pin/fastkpc_cuda.so"
local_session_identity$native_library_device_major_hex <- "f"
local_session_identity$native_library_device_minor_hex <- "e"
local_session_identity$native_library_inode <- "999999"
local_session_identity$execution_snapshot_sha256 <-
  sha("cross-process-local-execution-snapshot")
local_session_identity$sha256 <-
  .fastkpc_full_cuda_phase3_identity_hash(local_session_identity)
assert_true(
  identical(local_session_identity$sha256, default_identity$sha256),
  "stable input identity ignores session-local native path/inode evidence"
)
stable_child_script <- tempfile("phase3-stable-identity-child-", fileext = ".R")
writeLines(c(
  "source('fastkpc/R/full_cuda_ci_gate.R')",
  "source('fastkpc/R/full_cuda_ci_oracle_contract.R')",
  "source('fastkpc/R/full_cuda_ci_workload_census.R')",
  "source('fastkpc/R/full_cuda_ci_prepared_s_contract.R')",
  "source('fastkpc/R/full_cuda_ci_fixed_sp_runtime.R')",
  "source('fastkpc/R/cuda_native.R')",
  "source('fastkpc/R/full_cuda_ci_phase3_artifacts.R')",
  "sha <- function(label) fastkpc_full_cuda_census_hash_utf8(label)",
  "policy <- .fastkpc_full_cuda_phase3_policy_contract()",
  "abi <- fastkpc_full_cuda_fixed_sp_runtime_abi()",
  "lineage <- list(authenticated = TRUE, phase0_manifest_hash = sha('phase0-manifest'), phase1_manifest_hash = sha('phase1-manifest'), phase2_manifest_hash = sha('phase2-manifest'), dataset_file_sha256 = sha('dataset-file'), dataset_matrix_sha256 = sha('dataset-matrix'), canonical_setup_corpus_hash = sha('canonical-setup-corpus'), canonical_target_corpus_hash = sha('canonical-target-corpus'), phase0_source_commit = strrep('0', 40L), phase1_source_commit = strrep('1', 40L), phase2_source_commit = '42ef3efa08327056ffe5c9aad7a8953ff6864c7e', phase2_R_version = 'R version 4.4.1 (fixture)', phase2_mgcv_version = '1.9.1')",
  "runtime <- list(runtime_abi = abi$schema_version, runtime_abi_hash = abi$sha256, device_id = 2L, cuda_toolkit_version = 12040L, cuda_driver_version = 55054L, gpu_name = 'Fixture GPU', gpu_uuid = paste0('GPU-', strrep('b', 32L)), compute_capability_major = 8L, compute_capability_minor = 0L, compute_capability = '8.0', sm_count = 108L, cusolver_deterministic_mode_required = policy$cusolver_deterministic_mode_required, cublas_math_mode_required = policy$cublas_math_mode_required, cublas_atomics_mode_required = policy$cublas_atomics_mode_required, cublas_user_workspace_required = policy$cublas_user_workspace_required, cublas_workspace_bytes_required = policy$cublas_workspace_bytes_required, cublas_workspace_min_alignment_required = policy$cublas_workspace_min_alignment_required, runtime_policy_schema_version = policy$configuration_schema_version)",
  "native_path <- Sys.getenv('FASTKPC_STABLE_NATIVE_PATH')",
  "native_inode <- Sys.getenv('FASTKPC_STABLE_NATIVE_INODE')",
  "native_snapshot <- Sys.getenv('FASTKPC_STABLE_EXECUTION_SNAPSHOT')",
  "qualified <- list(schema_version = 'full-cuda-ci-phase3-qualified-native-cache-v1', provenance = list(head_base_commit = fastkpc_full_cuda_source_commit(), provenance_schema_version = 'full-cuda-ci-execution-source-snapshot-v6', provenance_mode = 'working-tree-execution-snapshot-v1', source_closure_schema_version = 'full-cuda-ci-execution-source-closure-v1', source_discovery_semantics = 'parsed-r-ast-load-time-literal-source-v1', source_closure_count = 7L, source_closure_sha256 = sha('stable-source-closure'), execution_snapshot_sha256 = native_snapshot, relevant_sources_dirty_or_untracked = FALSE, native_library_identity = 'qualified-pinned-inode-sha-exact-registered-mapped-path-v3', native_library_path = native_path, native_library_device_major_hex = Sys.getenv('FASTKPC_STABLE_NATIVE_DEVICE_MAJOR'), native_library_device_minor_hex = Sys.getenv('FASTKPC_STABLE_NATIVE_DEVICE_MINOR'), native_library_inode = native_inode, native_library_sha256 = sha('stable-native-library'), native_build_inputs_sha256 = sha('stable-build-inputs'), native_build_dependencies = list(schema_version = 'full-cuda-ci-native-build-dependencies-v3', trace_semantics = 'linux-strace-successful-read-exec-evidence-v3', trace_invocation = 'stable trace invocation', tracer_path = '/usr/bin/strace', dependency_count = 17L, exclusion_count = 2L, aggregate_sha256 = sha('stable-build-dependencies'), trace_sha256 = sha('stable-build-trace'), tracer_sha256 = sha('stable-build-tracer'))))",
  "assign('fastkpc_full_cuda_phase3_discover_qualified_native_evidence', function() qualified, envir = .GlobalEnv)",
  "execution <- .fastkpc_full_cuda_phase3_execution_projection(qualified$provenance)",
  "execution$execution_sources_unchanged_after_run <- TRUE",
  "execution$execution_provenance_state <- 'post-run-verified'",
  "assign('fastkpc_full_cuda_phase3_discover_catalog_evidence', function(catalog) lineage, envir = .GlobalEnv)",
  "assign('fastkpc_full_cuda_phase3_discover_runtime_evidence', function(catalog, device_id) runtime, envir = .GlobalEnv)",
  "assign('fastkpc_full_cuda_phase3_discover_device_evidence', function(catalog, device_id) runtime, envir = .GlobalEnv)",
  "assign('fastkpc_full_cuda_phase3_discover_execution_evidence', function(catalog, device_id) execution, envir = .GlobalEnv)",
  "identity <- fastkpc_full_cuda_phase3_input_identity(list(), 2L)",
  "cat(identity$sha256, '\\n', sep = '')"
), stable_child_script, useBytes = TRUE)
on.exit(unlink(stable_child_script, force = TRUE), add = TRUE)
stable_child_identity <- function(path, major, minor, inode, snapshot) {
  output <- system2(
    R.home("bin/Rscript"),
    c("--vanilla", stable_child_script),
    stdout = TRUE,
    stderr = TRUE,
    env = c(
      paste0("FASTKPC_STABLE_NATIVE_PATH=", path),
      paste0("FASTKPC_STABLE_NATIVE_DEVICE_MAJOR=", major),
      paste0("FASTKPC_STABLE_NATIVE_DEVICE_MINOR=", minor),
      paste0("FASTKPC_STABLE_NATIVE_INODE=", inode),
      paste0("FASTKPC_STABLE_EXECUTION_SNAPSHOT=", snapshot)
    )
  )
  status <- attr(output, "status", exact = TRUE)
  assert_true(
    is.null(status) && length(output) >= 1L,
    paste0("fresh-process stable identity failed: ",
           paste(output, collapse = "\n"))
  )
  output[[length(output)]]
}
stable_child_a <- stable_child_identity(
  "/tmp/fastkpc-stable-a/fastkpc_cuda.so", "8", "1", "1001",
  sha("stable-snapshot-a")
)
stable_child_b <- stable_child_identity(
  "/tmp/fastkpc-stable-b/fastkpc_cuda.so", "9", "2", "2002",
  sha("stable-snapshot-b")
)
assert_true(
  identical(stable_child_a, stable_child_b),
  "fresh Rscript identities are stable across distinct pin paths and inodes"
)
assert_true(is.list(identity) && identical(identity$device_id, 2L) &&
              identical(identity$shard_count, 64L) &&
              grepl("^[0-9a-f]{64}$", identity$sha256),
            "authenticated Phase 3 input identity")
assert_true(
  all(c(
    "runtime_policy_schema_version",
    "cusolver_deterministic_mode_required", "cublas_math_mode_required",
    "cublas_atomics_mode_required", "cublas_user_workspace_required",
    "cublas_workspace_bytes_required",
    "cublas_workspace_min_alignment_required",
    "native_library_identity", "native_library_path",
    "native_library_device_major_hex", "native_library_device_minor_hex",
    "native_library_inode", "native_library_sha256",
    "native_build_inputs_sha256",
    "native_build_dependencies_schema_version",
    "native_build_trace_semantics", "native_build_trace_invocation",
    "native_build_tracer_path", "native_build_dependency_count",
    "native_build_exclusion_count", "native_build_dependencies_sha256",
    "native_build_trace_sha256", "native_build_tracer_sha256"
  ) %in% names(identity)) &&
    !any(c(
      "cusolver_deterministic_mode", "cublas_math_mode",
      "cublas_atomics_mode", "cublas_user_workspace_installed",
      "cublas_workspace_bytes", "cublas_workspace_alignment",
      "cublas_workspace_identity"
    ) %in% names(identity)),
  "input identity separates stable policy from actual runtime attestation"
)
assert_true(
  default_discoverer_calls$catalog == 1L &&
    default_discoverer_calls$runtime == 1L &&
    default_discoverer_calls$device == 1L &&
    default_discoverer_calls$execution == 1L,
  "catalog/device_id call dispatches through authenticated default discoverers"
)
assert_true(
  identical(default_identity$phase2_source_commit, lineage_fixture$source_commit) &&
    identical(default_identity$phase2_R_version, lineage_fixture$R_version) &&
    identical(default_identity$phase2_mgcv_version, lineage_fixture$mgcv_version) &&
    identical(default_identity$source_commit, execution_fixture$source_commit) &&
    identical(default_identity$phase3_source_commit,
              execution_fixture$phase3_source_commit) &&
    identical(default_identity$R_version, execution_fixture$R_version) &&
    identical(default_identity$mgcv_version, execution_fixture$mgcv_version) &&
    identical(default_identity$phase3_R_version,
              execution_fixture$phase3_R_version) &&
    identical(default_identity$phase3_mgcv_version,
              execution_fixture$phase3_mgcv_version),
  "Phase 2 lineage and current Phase 3 execution identity remain separate"
)
assert_true(
  identical(
    default_identity$relevant_sources_dirty_or_untracked,
    execution_fixture$relevant_sources_dirty_or_untracked
  ) && identical(
    default_identity$execution_sources_unchanged_after_run,
    execution_fixture$execution_sources_unchanged_after_run
  ),
  "execution provenance dirtiness and post-run authentication are retained"
)
synthetic_non_head_execution <- execution_fixture
synthetic_non_head_execution$source_commit <- strrep("f", 40L)
synthetic_non_head_execution$phase3_source_commit <- strrep("f", 40L)
synthetic_execution_discoverer <- get(
  "fastkpc_full_cuda_phase3_discover_execution_evidence", envir = .GlobalEnv
)
assign(
  "fastkpc_full_cuda_phase3_discover_execution_evidence",
  function(catalog, device_id) synthetic_non_head_execution,
  envir = .GlobalEnv
)
assert_error(
  fastkpc_full_cuda_phase3_input_identity(catalog_fixture, 2L),
  "synthetic non-HEAD execution source cannot authenticate"
)
assign(
  "fastkpc_full_cuda_phase3_discover_execution_evidence",
  synthetic_execution_discoverer,
  envir = .GlobalEnv
)
test_catalog_discoverer <- get(
  "fastkpc_full_cuda_phase3_discover_catalog_evidence", envir = .GlobalEnv
)
repository_catalog <- list(
  phase0 = list(
    summary = list(pass = TRUE),
    manifest = list(source_commit = lineage_fixture$phase0_source_commit)
  ),
  inputs = list(
    summary = list(pass = TRUE),
    manifest = list(source_commit = lineage_fixture$phase1_source_commit),
    dataset_file_sha256 = lineage_fixture$dataset_file_sha256,
    dataset_sha256 = lineage_fixture$dataset_matrix_sha256
  ),
  phase2_manifest = list(
    phase2_complete = TRUE,
    full_canonical_prepared_s_key_corpus_hash =
      lineage_fixture$canonical_setup_corpus_hash,
    full_canonical_target_key_corpus_hash =
      lineage_fixture$canonical_target_corpus_hash,
    source_commit = lineage_fixture$phase2_source_commit,
    R_version = lineage_fixture$phase2_R_version,
    mgcv_version = lineage_fixture$phase2_mgcv_version
  ),
  phase2_file_hashes = c(
    manifest.json = lineage_fixture$phase2_manifest_hash
  ),
  phase0_manifest_hash = sha("repository-phase0-manifest"),
  phase1_manifest_hash = sha("repository-phase1-manifest")
)
assign(
  "fastkpc_full_cuda_phase3_discover_catalog_evidence",
  old_discoverers[["fastkpc_full_cuda_phase3_discover_catalog_evidence"]],
  envir = .GlobalEnv
)
assert_error(
  fastkpc_full_cuda_phase3_input_identity(repository_catalog, 2L),
  "default catalog discovery must reject forged-shaped catalogs without authority"
)
authority_tmp <- tempfile("phase3-authority-contract-")
dir.create(authority_tmp)
on.exit(unlink(authority_tmp, recursive = TRUE, force = TRUE), add = TRUE)
authority_phase0_dir <- file.path(authority_tmp, "phase0")
authority_phase1_dir <- file.path(authority_tmp, "phase1")
authority_phase2_dir <- file.path(authority_tmp, "phase2")
dir.create(authority_phase0_dir)
dir.create(authority_phase1_dir)
dir.create(authority_phase2_dir)
writeLines("phase0 manifest fixture",
           file.path(authority_phase0_dir, "manifest.json"),
           useBytes = TRUE)
writeLines("phase1 manifest fixture",
           file.path(authority_phase1_dir, "manifest.json"),
           useBytes = TRUE)
writeLines("phase2 manifest fixture",
           file.path(authority_phase2_dir, "manifest.json"),
           useBytes = TRUE)
authority_data_path <- file.path(authority_tmp, "data.rds")
saveRDS(matrix(1, nrow = 1L), authority_data_path, version = 3L)
authority_catalog <- repository_catalog
authority_catalog$phase0_dir <- authority_phase0_dir
authority_catalog$phase1_dir <- authority_phase1_dir
authority_catalog$phase2_dir <- authority_phase2_dir
authority_catalog$data_path <- authority_data_path
authority_catalog$phase0_manifest_hash <- lineage_fixture$phase0_manifest_hash
authority_catalog$phase1_manifest_hash <- lineage_fixture$phase1_manifest_hash
authority_record <- .fastkpc_full_cuda_phase3_catalog_authority_snapshot(
  authority_catalog
)
authority_catalog$phase3_catalog_authority_token <-
  .fastkpc_full_cuda_phase3_register_catalog_authority(authority_record)
authority_catalog$phase3_catalog_authority_sha256 <- authority_record$sha256
authority_builder <- get(
  ".fastkpc_full_cuda_phase3_build_catalog_authority", envir = .GlobalEnv
)
authority_builder_calls <- 0L
assign(
  ".fastkpc_full_cuda_phase3_build_catalog_authority",
  function(...) {
    authority_builder_calls <<- authority_builder_calls + 1L
    stop("full catalog builder must not run during authority lookup",
         call. = FALSE)
  },
  envir = .GlobalEnv
)
on.exit(
  assign(
    ".fastkpc_full_cuda_phase3_build_catalog_authority",
    authority_builder,
    envir = .GlobalEnv
  ),
  add = TRUE
)
authority_lookup <- fastkpc_full_cuda_phase3_discover_catalog_authority(
  authority_catalog
)
assert_true(
  authority_builder_calls == 0L &&
    identical(authority_lookup$authority_sha256, authority_record$sha256),
  "catalog authority lookup revalidates registered projection without full rebuild"
)
missing_file_identity_authority <- authority_record
missing_file_identity_authority$data_path <-
  file.path(authority_tmp, "missing-data.rds")
missing_file_identity_authority$sha256 <-
  .fastkpc_full_cuda_phase3_catalog_authority_hash(
    missing_file_identity_authority
  )
assert_error(
  .fastkpc_full_cuda_phase3_register_catalog_authority(
    missing_file_identity_authority
  ),
  "catalog authority registration requires immutable file identities"
)
authority_registry <- get(
  ".fastkpc_full_cuda_phase3_catalog_authority_registry", envir = .GlobalEnv
)
stale_authority <- get(
  authority_record$sha256, envir = authority_registry, inherits = FALSE
)
attr(stale_authority, "file_records") <- NULL
assign(authority_record$sha256, stale_authority, envir = authority_registry)
assert_error(
  fastkpc_full_cuda_phase3_discover_catalog_authority(authority_catalog),
  "catalog authority lookup rejects stale registry entries without file identity"
)
authority_catalog$phase3_catalog_authority_token <-
  .fastkpc_full_cuda_phase3_register_catalog_authority(authority_record)
authority_mutated <- authority_catalog
authority_mutated$phase2_manifest$source_commit <- strrep("c", 40L)
assert_error(
  fastkpc_full_cuda_phase3_discover_catalog_authority(authority_mutated),
  "catalog authority lookup rejects source projection mutation"
)
writeLines("phase1 manifest mutated",
           file.path(authority_phase1_dir, "manifest.json"),
           useBytes = TRUE)
assert_error(
  fastkpc_full_cuda_phase3_discover_catalog_authority(authority_catalog),
  "catalog authority lookup rejects mutated authority file identity"
)
writeLines("phase1 manifest fixture",
           file.path(authority_phase1_dir, "manifest.json"),
           useBytes = TRUE)
authority_registry <- get(
  ".fastkpc_full_cuda_phase3_catalog_authority_registry", envir = .GlobalEnv
)
authority_registry_limit <-
  .fastkpc_full_cuda_phase3_catalog_authority_registry_max_entries()
for (index in seq_len(authority_registry_limit + 3L)) {
  extra_record <- authority_record
  extra_record$phase2_source_commit <- sprintf("%040x", index)
  extra_record$sha256 <-
    .fastkpc_full_cuda_phase3_catalog_authority_hash(extra_record)
  .fastkpc_full_cuda_phase3_register_catalog_authority(extra_record)
}
assert_true(
  length(ls(authority_registry, all.names = TRUE)) <= authority_registry_limit,
  "catalog authority registry is bounded"
)
assign(
  "fastkpc_full_cuda_phase3_discover_catalog_evidence",
  test_catalog_discoverer,
  envir = .GlobalEnv
)
repository_execution_discoverer <- old_discoverers[[
  "fastkpc_full_cuda_phase3_discover_execution_evidence"
]]
test_execution_discoverer <- get(
  "fastkpc_full_cuda_phase3_discover_execution_evidence", envir = .GlobalEnv
)
assign(
  "fastkpc_full_cuda_phase3_discover_execution_evidence",
  repository_execution_discoverer,
  envir = .GlobalEnv
)
repository_execution_identity <- fastkpc_full_cuda_phase3_input_identity(
  catalog_fixture,
  2L
)
assert_true(
  identical(
    repository_execution_identity$source_commit,
    fastkpc_full_cuda_source_commit()
  ) && identical(
    repository_execution_identity$phase3_source_commit,
    repository_execution_identity$source_commit
  ) && identical(repository_execution_identity$R_version, R.version.string) &&
    identical(repository_execution_identity$mgcv_version,
              current_mgcv_version) &&
    identical(repository_execution_identity$phase2_source_commit,
              lineage_fixture$phase2_source_commit) &&
    !identical(repository_execution_identity$source_commit,
                repository_execution_identity$phase2_source_commit),
  "default execution discovery authenticates current implementation identity"
)
assign(
  "fastkpc_full_cuda_phase3_discover_execution_evidence",
  test_execution_discoverer,
  envir = .GlobalEnv
)
valid_runtime_discoverer <- get(
  "fastkpc_full_cuda_phase3_discover_runtime_evidence", envir = .GlobalEnv
)
valid_device_discoverer <- get(
  "fastkpc_full_cuda_phase3_discover_device_evidence", envir = .GlobalEnv
)
assign(
  "fastkpc_full_cuda_phase3_discover_runtime_evidence",
  function(catalog, device_id) {
    invalid <- runtime_fixture
    invalid$device_id <- device_id + 1L
    invalid
  },
  envir = .GlobalEnv
)
assert_error(
  fastkpc_full_cuda_phase3_input_identity(catalog_fixture, 2L),
  "mismatched discovered runtime evidence must fail closed"
)
assign(
  "fastkpc_full_cuda_phase3_discover_runtime_evidence",
  valid_runtime_discoverer,
  envir = .GlobalEnv
)
valid_execution_discoverer <- get(
  "fastkpc_full_cuda_phase3_discover_execution_evidence", envir = .GlobalEnv
)
assign(
  "fastkpc_full_cuda_phase3_discover_execution_evidence",
  function(catalog, device_id) {
    invalid <- execution_fixture
    invalid$authenticated <- FALSE
    invalid
  },
  envir = .GlobalEnv
)
assert_error(
  fastkpc_full_cuda_phase3_input_identity(catalog_fixture, 2L),
  "unauthenticated discovered execution evidence must fail closed"
)
assign(
  "fastkpc_full_cuda_phase3_discover_execution_evidence",
  function(catalog, device_id) {
    invalid <- execution_fixture
    invalid$execution_sources_unchanged_after_run <- FALSE
    invalid
  },
  envir = .GlobalEnv
)
assert_error(
  fastkpc_full_cuda_phase3_input_identity(catalog_fixture, 2L),
  "unverified execution source snapshot must fail closed"
)
assign(
  "fastkpc_full_cuda_phase3_discover_execution_evidence",
  valid_execution_discoverer,
  envir = .GlobalEnv
)
catalog_with_unmarked_provenance <- catalog_fixture
catalog_with_unmarked_provenance$phase3_execution_provenance <-
  execution_fixture
assign(
  "fastkpc_full_cuda_phase3_discover_execution_evidence",
  repository_execution_discoverer,
  envir = .GlobalEnv
)
assert_error(
  fastkpc_full_cuda_phase3_input_identity(
    catalog_with_unmarked_provenance, 2L
  ),
  "catalog-supplied execution provenance must fail closed"
)
assign(
  "fastkpc_full_cuda_phase3_discover_execution_evidence",
  valid_execution_discoverer,
  envir = .GlobalEnv
)
catalog_with_execution_evidence <- catalog_fixture
catalog_with_execution_evidence$phase3_execution_evidence <- execution_fixture
assign(
  "fastkpc_full_cuda_phase3_discover_execution_evidence",
  repository_execution_discoverer,
  envir = .GlobalEnv
)
assert_error(
  fastkpc_full_cuda_phase3_input_identity(
    catalog_with_execution_evidence, 2L
  ),
  "catalog-supplied execution evidence must fail closed"
)
assign(
  "fastkpc_full_cuda_phase3_discover_execution_evidence",
  valid_execution_discoverer,
  envir = .GlobalEnv
)

identity_rejected <- function(lineage = lineage_fixture,
                              runtime = runtime_fixture,
                              device = runtime_fixture,
                              execution = execution_fixture,
                              label = "invalid identity") {
  assign(
    "fastkpc_full_cuda_phase3_discover_catalog_evidence",
    function(catalog) lineage,
    envir = .GlobalEnv
  )
  assign(
    "fastkpc_full_cuda_phase3_discover_runtime_evidence",
    function(catalog, device_id) runtime,
    envir = .GlobalEnv
  )
  assign(
    "fastkpc_full_cuda_phase3_discover_device_evidence",
    function(catalog, device_id) device,
    envir = .GlobalEnv
  )
  assign(
    "fastkpc_full_cuda_phase3_discover_execution_evidence",
    function(catalog, device_id) execution,
    envir = .GlobalEnv
  )
  assert_error(
    fastkpc_full_cuda_phase3_input_identity(catalog_fixture, 2L),
    label
  )
  assign(
    "fastkpc_full_cuda_phase3_discover_catalog_evidence",
    test_catalog_discoverer,
    envir = .GlobalEnv
  )
  assign(
    "fastkpc_full_cuda_phase3_discover_runtime_evidence",
    valid_runtime_discoverer,
    envir = .GlobalEnv
  )
  assign(
    "fastkpc_full_cuda_phase3_discover_device_evidence",
    valid_device_discoverer,
    envir = .GlobalEnv
  )
  assign(
    "fastkpc_full_cuda_phase3_discover_execution_evidence",
    valid_execution_discoverer,
    envir = .GlobalEnv
  )
}
missing_lineage <- lineage_fixture
missing_lineage$phase2_manifest_hash <- NULL
identity_rejected(lineage = missing_lineage,
                  label = "missing lineage hash must fail")
duplicate_lineage <- lineage_fixture
names(duplicate_lineage)[[1L]] <- names(duplicate_lineage)[[2L]]
identity_rejected(lineage = duplicate_lineage,
                  label = "duplicate lineage fields must fail")
unvalidated_lineage <- lineage_fixture
unvalidated_lineage$authenticated <- FALSE
identity_rejected(lineage = unvalidated_lineage,
                  label = "unvalidated lineage must fail")
malformed_runtime <- runtime_fixture
malformed_runtime$gpu_uuid <- NULL
identity_rejected(runtime = malformed_runtime, device = malformed_runtime,
                  label = "missing GPU UUID must fail")
mismatched_runtime <- runtime_fixture
mismatched_runtime$device_id <- 3L
identity_rejected(runtime = mismatched_runtime, device = mismatched_runtime,
                  label = "mismatched device identity must fail")
wrong_native_library <- execution_fixture
wrong_native_library$native_library_sha256 <- sha("wrong-native-library")
identity_rejected(
  execution = wrong_native_library,
  label = "mutated native library evidence must fail"
)
wrong_build_inputs <- execution_fixture
wrong_build_inputs$native_build_inputs_sha256 <- sha("wrong-build-inputs")
identity_rejected(
  execution = wrong_build_inputs,
  label = "mutated native build input evidence must fail"
)
wrong_build_dependencies <- execution_fixture
wrong_build_dependencies$native_build_dependencies_sha256 <-
  sha("wrong-build-dependencies")
identity_rejected(
  execution = wrong_build_dependencies,
  label = "mutated native dependency evidence must fail"
)
wrong_build_trace <- execution_fixture
wrong_build_trace$native_build_trace_sha256 <- sha("wrong-build-trace")
identity_rejected(
  execution = wrong_build_trace,
  label = "mutated native build trace evidence must fail"
)
wrong_build_tracer <- execution_fixture
wrong_build_tracer$native_build_tracer_sha256 <- sha("wrong-build-tracer")
identity_rejected(
  execution = wrong_build_tracer,
  label = "mutated native tracer evidence must fail"
)
forged_identity <- identity
forged_identity$route_config_hash <- sha("forged-route")
assert_error(
  fastkpc_full_cuda_phase3_validate_input_identity(forged_identity),
  "caller-forged route identity must fail"
)

write_fixture_artifact <- function(kind, output_dir, identity,
                                   include_payload = TRUE) {
  paths <- fastkpc_full_cuda_phase3_artifact_paths(output_dir, kind)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(paths$shards_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(paths$sessions_dir, recursive = TRUE, showWarnings = FALSE)
  payload_keys <- setdiff(names(paths), c(
    "manifest_json", "summary_json", "shards_dir", "sessions_dir"
  ))
  if (isTRUE(include_payload)) {
    # Task 1 checks serialization and lineage; Task 4/9 own numeric/graph gates.
    for (key in payload_keys) {
      path <- paths[[key]]
      if (grepl("\\.rds$", path)) {
        value <- if (identical(key, "adjacency_rds")) {
          matrix(0, nrow = 1L, ncol = 1L,
                 dimnames = list("fixture", "fixture"))
        } else {
          list(
            schema_version = "full-cuda-ci-phase3-contract-fixture-v1",
            contract_fixture = TRUE,
            payload = key
          )
        }
        saveRDS(value, path, version = 3L)
      } else if (grepl("\\.json$", path)) {
        value <- switch(
          key,
          route_config_json = as.list(fastkpc_full_cuda_phase3_route_config()),
          first_divergence_json = list(
            schema_version =
              "full-cuda-ci-phase3-contract-fixture-v1",
            first_divergence_found = FALSE
          ),
          direct_ci_summary_json = list(
            schema_version =
              "full-cuda-ci-phase3-contract-fixture-v1",
            contract_fixture = TRUE,
            pass = TRUE
          ),
          list(
            schema_version = "full-cuda-ci-phase3-contract-fixture-v1",
            contract_fixture = TRUE,
            payload = key
          )
        )
        jsonlite::write_json(
          value, path, auto_unbox = TRUE, pretty = TRUE
        )
      } else if (grepl("\\.csv$", path)) {
        value <- switch(
          key,
          input_hashes_csv = data.frame(
            logical_path = "fixture",
            sha256 = sha("fixture-input"),
            stringsAsFactors = FALSE
          ),
          fallbacks_csv = data.frame(
            type = character(), key = character(), reason = character(),
            count = integer(), stringsAsFactors = FALSE
          ),
          failures_csv = data.frame(
            stage = character(), error_class = character(),
            error_message = character(), stringsAsFactors = FALSE
          ),
          data.frame(
            schema_version = "full-cuda-ci-phase3-contract-fixture-v1",
            contract_fixture = TRUE,
            payload = key,
            stringsAsFactors = FALSE
          )
        )
        utils::write.csv(value, path, row.names = FALSE)
      } else {
        writeLines(
          c("full-cuda-ci-phase3-contract-fixture-v1", key),
          path, useBytes = TRUE
        )
      }
    }
  }
  payload_hashes <- if (isTRUE(include_payload)) {
    vapply(paths[payload_keys], fastkpc_full_cuda_census_file_hash,
           character(1L))
  } else {
    setNames(character(), character())
  }
  identity_fields <- setdiff(names(identity), c("schema_version", "sha256"))
  manifest <- c(
    list(
      artifact_schema_version = if (kind == "oracle_sp") {
        fastkpc_full_cuda_phase3_oracle_schema_version()
      } else fastkpc_full_cuda_phase3_shadow_schema_version(),
      artifact_kind = kind,
      input_identity_schema_version = identity$schema_version,
      input_identity_sha256 = identity$sha256,
      payload_names = vapply(paths[payload_keys], basename, character(1L)),
      publication_order = c(
        unname(vapply(paths[payload_keys], basename, character(1L))),
        "manifest.json", "summary.json"
      ),
      payload_file_sha256 = if (isTRUE(include_payload)) {
        as.list(setNames(
          unname(payload_hashes),
          vapply(paths[payload_keys], basename, character(1L))
        ))
      } else list()
    ),
    as.list(identity[identity_fields])
  )
  jsonlite::write_json(
    manifest, paths$manifest_json, auto_unbox = TRUE, pretty = TRUE,
    null = "null", digits = NA
  )
  summary <- list(
    artifact_schema_version = manifest$artifact_schema_version,
    artifact_kind = kind,
    manifest_sha256 = fastkpc_full_cuda_census_file_hash(
      paths$manifest_json
    ),
    shard_count = identity$shard_count,
    payload_count = as.integer(length(payload_keys)),
    pass = TRUE
  )
  jsonlite::write_json(
    summary, paths$summary_json, auto_unbox = TRUE, pretty = TRUE,
    null = "null", digits = NA
  )
  list(paths = paths, manifest = manifest, summary = summary)
}

default_artifact_root <- tempfile("phase3-default-discovery-")
default_artifact <- write_fixture_artifact(
  "oracle_sp", default_artifact_root, default_identity
)
default_validated <- fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
  default_artifact_root, catalog = catalog_fixture, device_id = 2L
)
assert_true(
  isTRUE(default_validated$authenticated),
  "artifact validation uses default catalog/device discoverers"
)
explicit_validated <- fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
  default_artifact_root,
  catalog = catalog_fixture,
  device_id = 2L
)
assert_true(
  isTRUE(explicit_validated$authenticated),
  "artifact validation accepts explicit authenticated evidence seams"
)
assert_error(
  fastkpc_full_cuda_phase3_input_identity(
    catalog_fixture, 2L, runtime_evidence = runtime_fixture
  ),
  "input identity rejects caller-supplied evidence arguments"
)
forged_precomputed_identity <- identity
forged_precomputed_identity$device_id <- 3L
forged_precomputed_identity$sha256 <-
  .fastkpc_full_cuda_phase3_identity_hash(forged_precomputed_identity)
assert_error(
  fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
    default_artifact_root,
    expected_identity = forged_precomputed_identity,
    catalog = catalog_fixture,
    device_id = 2L
  ),
  "catalog validation independently recomputes a precomputed identity"
)
unlink(default_artifact_root, recursive = TRUE, force = TRUE)

artifact_root <- tempfile("phase3-artifact-contract-")
on.exit(unlink(artifact_root, recursive = TRUE, force = TRUE), add = TRUE)
oracle_fixture <- write_fixture_artifact(
  "oracle_sp", file.path(artifact_root, "oracle"), identity
)
shadow_fixture <- write_fixture_artifact(
  "full_shadow", file.path(artifact_root, "shadow"), identity
)
oracle_manifest_order <- jsonlite::read_json(
  oracle_fixture$paths$manifest_json, simplifyVector = TRUE
)
assert_true(
  identical(
    oracle_manifest_order$publication_order,
    c(
      oracle_manifest_order$payload_names,
      "manifest.json", "summary.json"
    )
  ),
  "manifest records payload-then-manifest-then-summary publication order"
)
oracle_validated <- fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
  file.path(artifact_root, "oracle"), expected_identity = identity
)
shadow_validated <- fastkpc_validate_full_cuda_fixed_sp_shadow_artifact(
  file.path(artifact_root, "shadow"), expected_identity = identity
)
assert_true(isTRUE(oracle_validated$authenticated) &&
              isTRUE(shadow_validated$authenticated),
            "authenticated oracle and shadow fixtures")

non_authoritative_summary <- jsonlite::read_json(
  oracle_fixture$paths$summary_json, simplifyVector = TRUE
)
non_authoritative_summary$pass <- FALSE
jsonlite::write_json(
  non_authoritative_summary, oracle_fixture$paths$summary_json,
  auto_unbox = TRUE, pretty = TRUE, null = "null", digits = NA
)
summary_revalidated <- fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
  file.path(artifact_root, "oracle"), expected_identity = identity
)
assert_true(
  isTRUE(summary_revalidated$pass) &&
    isTRUE(summary_revalidated$computed_contract_pass),
  "summary pass is ignored in favor of validator-owned contract result"
)
invisible(write_fixture_artifact(
  "oracle_sp", file.path(artifact_root, "oracle"), identity
))

common_payload_names <- vapply(
  oracle_paths[c(
    "commands_txt", "environment_txt", "input_hashes_csv",
    "route_config_json", "runtime_lifecycle_csv", "resource_metrics_csv",
    "stage_timing_csv", "fallbacks_csv", "failures_csv"
  )],
  basename,
  character(1L)
)
manifest_for_payload_closure <- jsonlite::read_json(
  oracle_fixture$paths$manifest_json, simplifyVector = TRUE
)
assert_true(
  all(common_payload_names %in% manifest_for_payload_closure$payload_names) &&
    identical(
      sort(unname(common_payload_names), method = "radix"),
      sort(unname(names(manifest_for_payload_closure$payload_file_sha256)[
        names(manifest_for_payload_closure$payload_file_sha256) %in%
          common_payload_names
      ]), method = "radix")
    ),
  "manifest hashes every common top-level payload"
)
common_payload_keys <- c(
  "commands_txt", "environment_txt", "input_hashes_csv",
  "route_config_json", "runtime_lifecycle_csv", "resource_metrics_csv",
  "stage_timing_csv", "fallbacks_csv", "failures_csv"
)
for (key in common_payload_keys) {
  writeLines(
    paste0("mutated-common-payload:", key),
    oracle_fixture$paths[[key]], useBytes = TRUE
  )
  assert_error(
    fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
      file.path(artifact_root, "oracle"), expected_identity = identity
    ),
    paste0("common payload mutation must invalidate ", key)
  )
  invisible(write_fixture_artifact(
    "oracle_sp", file.path(artifact_root, "oracle"), identity
  ))
}

forged_payload_manifest <- jsonlite::read_json(
  oracle_fixture$paths$manifest_json, simplifyVector = TRUE
)
forged_payload_summary <- jsonlite::read_json(
  oracle_fixture$paths$summary_json, simplifyVector = TRUE
)
writeLines("this is not an RDS payload", oracle_fixture$paths$setup_results_rds,
           useBytes = TRUE)
forged_payload_manifest$payload_file_sha256[["setup_results.rds"]] <-
  fastkpc_full_cuda_census_file_hash(oracle_fixture$paths$setup_results_rds)
jsonlite::write_json(
  forged_payload_manifest, oracle_fixture$paths$manifest_json,
  auto_unbox = TRUE, pretty = TRUE, null = "null", digits = NA
)
forged_payload_summary$manifest_sha256 <- fastkpc_full_cuda_census_file_hash(
  oracle_fixture$paths$manifest_json
)
jsonlite::write_json(
  forged_payload_summary, oracle_fixture$paths$summary_json,
  auto_unbox = TRUE, pretty = TRUE, null = "null", digits = NA
)
assert_error(
  fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
    file.path(artifact_root, "oracle"), expected_identity = identity
  ),
  "malformed payload must fail even when its manifest hash is forged"
)
invisible(write_fixture_artifact(
  "oracle_sp", file.path(artifact_root, "oracle"), identity
))

mutated_manifest_rejected <- function(field, value, label) {
  manifest <- jsonlite::read_json(
    oracle_fixture$paths$manifest_json, simplifyVector = TRUE
  )
  manifest[[field]] <- value
  jsonlite::write_json(
    manifest, oracle_fixture$paths$manifest_json, auto_unbox = TRUE,
    pretty = TRUE, null = "null", digits = NA
  )
  assert_error(
    fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
      file.path(artifact_root, "oracle"), expected_identity = identity
    ),
    label
  )
  invisible(write_fixture_artifact(
    "oracle_sp", file.path(artifact_root, "oracle"), identity
  ))
}

manifest_mutations <- list(
  list("phase0_manifest_hash", sha("wrong-phase0"),
       "Phase 0 manifest hash mutation must fail"),
  list("phase1_manifest_hash", sha("wrong-phase1"),
       "Phase 1 manifest hash mutation must fail"),
  list("phase2_manifest_hash", sha("wrong-phase2"),
       "Phase 2 manifest hash mutation must fail"),
  list("phase0_source_commit", strrep("a", 40L),
       "Phase 0 source lineage mutation must fail"),
  list("phase1_source_commit", strrep("b", 40L),
       "Phase 1 source lineage mutation must fail"),
  list("phase2_source_commit", strrep("c", 40L),
       "Phase 2 source lineage mutation must fail"),
  list("dataset_file_sha256", sha("wrong-dataset-file"),
       "dataset file hash mutation must fail"),
  list("dataset_matrix_sha256", sha("wrong-dataset-matrix"),
       "dataset matrix hash mutation must fail"),
  list("canonical_setup_corpus_hash", sha("wrong-setup-corpus"),
       "canonical setup corpus mutation must fail"),
  list("canonical_target_corpus_hash", sha("wrong-target-corpus"),
       "canonical target corpus mutation must fail"),
  list("route_config_hash", sha("wrong-route"),
       "route configuration hash mutation must fail"),
  list("runtime_abi", "wrong-runtime-abi",
       "runtime ABI mutation must fail"),
  list("runtime_abi_hash", sha("wrong-runtime-abi-hash"),
       "runtime ABI hash mutation must fail"),
  list("runtime_policy_schema_version", "wrong-policy-schema",
       "runtime policy schema mutation must fail"),
  list("source_commit", strrep("d", 40L),
       "current source commit mutation must fail"),
  list("phase3_source_commit", strrep("e", 40L),
       "current Phase 3 source commit mutation must fail"),
  list("phase2_R_version", "forged-Phase-2-R",
       "Phase 2 R version mutation must fail"),
  list("phase2_mgcv_version", "forged-Phase-2-mgcv",
       "Phase 2 mgcv version mutation must fail"),
  list("R_version", "forged-current-R",
       "current R version mutation must fail"),
  list("phase3_R_version", "forged-Phase-3-R",
       "current Phase 3 R version alias mutation must fail"),
  list("mgcv_version", "forged-current-mgcv",
       "current mgcv version mutation must fail"),
  list("phase3_mgcv_version", "forged-Phase-3-mgcv",
       "current Phase 3 mgcv version alias mutation must fail"),
  list("provenance_schema_version", "forged-provenance-schema",
       "provenance schema mutation must fail"),
  list("provenance_mode", "forged-provenance-mode",
       "provenance mode mutation must fail"),
  list("source_closure_schema_version", "forged-closure-schema",
       "source closure schema mutation must fail"),
  list("source_discovery_semantics", "forged-discovery-semantics",
       "source discovery semantics mutation must fail"),
  list("source_closure_count", 2L,
       "source closure count mutation must fail"),
  list("source_closure_sha256", sha("wrong-source-closure"),
       "source closure hash mutation must fail"),
  list("execution_snapshot_sha256", sha("wrong-execution-snapshot"),
       "execution snapshot hash mutation must fail"),
  list("execution_provenance_state", "pre-run-capture",
       "execution provenance state mutation must fail"),
  list("relevant_sources_dirty_or_untracked",
       !isTRUE(identity$relevant_sources_dirty_or_untracked),
       "source dirtiness mutation must fail"),
  list("execution_sources_unchanged_after_run", FALSE,
       "post-run source authentication mutation must fail"),
  list("native_library_identity",
       "qualified-pinned-inode-sha-exact-registered-mapped-path-v2",
       "native library identity mutation must fail"),
  list("native_library_sha256", sha("wrong-native-library"),
       "native library hash mutation must fail"),
  list("native_build_inputs_sha256", sha("wrong-build-inputs"),
       "native build input hash mutation must fail"),
  list("native_build_dependencies_schema_version",
       "full-cuda-ci-native-build-dependencies-v2",
       "native build dependency schema mutation must fail"),
  list("native_build_trace_semantics",
       "linux-strace-successful-read-exec-evidence-v2",
       "native build trace semantics mutation must fail"),
  list("native_build_trace_invocation", "forged trace invocation",
       "native build trace invocation mutation must fail"),
  list("native_build_tracer_path", "/tmp/forged-strace",
       "native build tracer path mutation must fail"),
  list("native_build_dependency_count", 99L,
       "native build dependency count mutation must fail"),
  list("native_build_exclusion_count", 99L,
       "native build exclusion count mutation must fail"),
  list("native_build_dependencies_sha256", sha("wrong-build-dependencies"),
       "native build dependency hash mutation must fail"),
  list("native_build_trace_sha256", sha("wrong-build-trace"),
       "native build trace hash mutation must fail"),
  list("native_build_tracer_sha256", sha("wrong-build-tracer"),
       "native build tracer hash mutation must fail"),
  list("cuda_toolkit_version", 99999L,
       "CUDA toolkit mutation must fail"),
  list("cuda_driver_version", 99999L,
       "CUDA driver mutation must fail"),
  list("gpu_uuid", paste0("GPU-", strrep("c", 32L)),
       "GPU UUID mutation must fail"),
  list("gpu_name", "Forged Fixture GPU",
       "GPU name mutation must fail"),
  list("compute_capability_major", 9L,
       "compute capability major mutation must fail"),
  list("compute_capability_minor", 9L,
       "compute capability minor mutation must fail"),
  list("compute_capability", "9.9",
       "compute capability mutation must fail"),
  list("sm_count", 1L, "SM count mutation must fail"),
  list("device_id", 7L, "device id mutation must fail"),
  list("cusolver_deterministic_mode_required", "disabled",
       "cuSOLVER deterministic policy mutation must fail"),
  list("cublas_math_mode_required", "default",
       "cuBLAS math policy mutation must fail"),
  list("cublas_atomics_mode_required", "allowed",
       "cuBLAS atomics policy mutation must fail"),
  list("cublas_user_workspace_required", FALSE,
       "cuBLAS workspace requirement mutation must fail"),
  list("cublas_workspace_bytes_required", 8192,
       "cuBLAS workspace size policy mutation must fail"),
  list("cublas_workspace_min_alignment_required", 512,
       "cuBLAS workspace alignment policy mutation must fail"),
  list("artifact_schema_version", "forged-artifact-schema",
       "artifact schema mutation must fail"),
  list("shard_count", 4L, "shard count mutation must fail")
)
for (mutation in manifest_mutations) {
  mutated_manifest_rejected(mutation[[1L]], mutation[[2L]], mutation[[3L]])
}

empty_completion <- file.path(artifact_root, "empty-completion")
invisible(write_fixture_artifact(
  "oracle_sp", empty_completion, identity, include_payload = FALSE
))
assert_error(
  fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
    empty_completion, expected_identity = identity
  ),
  "summary pass=true without payload must fail"
)

missing_manifest <- file.path(artifact_root, "missing-manifest")
invisible(write_fixture_artifact("oracle_sp", missing_manifest, identity))
missing_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  missing_manifest, "oracle_sp"
)
unlink(missing_paths$manifest_json)
assert_error(
  fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
    missing_manifest, expected_identity = identity
  ),
  "missing manifest must fail"
)

partial_completion <- file.path(artifact_root, "partial")
invisible(write_fixture_artifact("oracle_sp", partial_completion, identity))
partial_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  partial_completion, "oracle_sp"
)
unlink(partial_paths$target_parity_rds)
assert_error(
  fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
    partial_completion, expected_identity = identity
  ),
  "partial payload completion must fail"
)

duplicate_payload <- file.path(artifact_root, "duplicate-payload")
invisible(write_fixture_artifact("oracle_sp", duplicate_payload, identity))
duplicate_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  duplicate_payload, "oracle_sp"
)
duplicate_manifest <- jsonlite::read_json(
  duplicate_paths$manifest_json, simplifyVector = TRUE
)
duplicate_manifest$payload_names <- c(
  duplicate_manifest$payload_names,
  duplicate_manifest$payload_names[[1L]]
)
jsonlite::write_json(
  duplicate_manifest, duplicate_paths$manifest_json, auto_unbox = TRUE,
  pretty = TRUE, null = "null", digits = NA
)
assert_error(
  fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
    duplicate_payload, expected_identity = identity
  ),
  "duplicate payload names must fail"
)

dirs_only <- file.path(artifact_root, "dirs-only")
dir.create(dirs_only, recursive = TRUE)
dir.create(file.path(dirs_only, "shards"))
dir.create(file.path(dirs_only, "sessions"))
assert_error(
  fastkpc_full_cuda_phase3_validate_artifact(
    dirs_only, kind = "oracle_sp", expected_identity = identity
  ),
  "directories without top-level publication must fail"
)
assert_error(
  fastkpc_full_cuda_phase3_artifact_paths(tempfile("bad-kind-"), "bogus"),
  "unknown artifact kind must fail"
)
assert_error(
  fastkpc_full_cuda_phase3_artifact_paths(factor("bad-path"), "oracle_sp"),
  "malformed artifact path must fail"
)

cuda_dll_paths_after <- fastkpc_cuda_dll_paths()
assert_true(
  identical(cuda_dll_paths_after, character()),
  "contract fixture loads no fastkpc_cuda DLLs"
)

cat("PASS phase3 artifact contract route, identity, and publication validation\n")
