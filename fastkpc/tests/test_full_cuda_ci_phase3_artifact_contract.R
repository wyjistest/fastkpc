source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
if (file.exists("fastkpc/R/full_cuda_ci_phase3_artifacts.R")) {
  source("fastkpc/R/full_cuda_ci_phase3_artifacts.R")
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
  source_commit = strrep("1", 40L),
  R_version = "R version 4.4.1 (fixture)",
  mgcv_version = "1.9.1"
)
runtime_abi <- fastkpc_full_cuda_fixed_sp_runtime_abi()
runtime_fixture <- list(
  runtime_abi = runtime_abi$schema_version,
  runtime_abi_hash = runtime_abi$sha256,
  device_id = 2L,
  cuda_toolkit_version = 12040L,
  cuda_driver_version = 55054L,
  gpu_name = "Fixture GPU",
  gpu_uuid = "GPU-fixture-uuid-0001",
  compute_capability_major = 8L,
  compute_capability_minor = 0L,
  compute_capability = "8.0",
  sm_count = 108L,
  cusolver_deterministic_mode = "enabled",
  cublas_math_mode = "pedantic",
  cublas_atomics_mode = "not_allowed",
  cublas_user_workspace_installed = TRUE,
  cublas_workspace_bytes = 4096,
  cublas_workspace_alignment = 256,
  cublas_workspace_identity = sha("fixture-cublas-workspace-4096-256")
)
catalog_fixture <- list(
  phase3_lineage = lineage_fixture
)
identity <- fastkpc_full_cuda_phase3_input_identity(
  catalog = catalog_fixture,
  device_id = 2L,
  catalog_evidence = function(catalog) catalog$phase3_lineage,
  runtime_evidence = function(device_id) runtime_fixture,
  device_evidence = function(device_id) runtime_fixture
)
assert_true(is.list(identity) && identical(identity$device_id, 2L) &&
              identical(identity$shard_count, 64L) &&
              grepl("^[0-9a-f]{64}$", identity$sha256),
            "authenticated Phase 3 input identity")

identity_rejected <- function(lineage = lineage_fixture,
                              runtime = runtime_fixture,
                              device = runtime_fixture,
                              label = "invalid identity") {
  assert_error(
    fastkpc_full_cuda_phase3_input_identity(
      catalog = catalog_fixture,
      device_id = 2L,
      catalog_evidence = function(catalog) lineage,
      runtime_evidence = function(device_id) runtime,
      device_evidence = function(device_id) device
    ),
    label
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
    for (key in payload_keys) {
      path <- paths[[key]]
      if (grepl("\\.rds$", path)) {
        saveRDS(list(fixture = key), path, version = 3L)
      } else if (grepl("\\.json$", path)) {
        value <- if (key == "route_config_json") {
          as.list(fastkpc_full_cuda_phase3_route_config())
        } else {
          list(fixture = key)
        }
        jsonlite::write_json(
          value, path, auto_unbox = TRUE, pretty = TRUE
        )
      } else {
        writeLines(paste0("fixture-", key), path, useBytes = TRUE)
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

artifact_root <- tempfile("phase3-artifact-contract-")
on.exit(unlink(artifact_root, recursive = TRUE, force = TRUE), add = TRUE)
oracle_fixture <- write_fixture_artifact(
  "oracle_sp", file.path(artifact_root, "oracle"), identity
)
shadow_fixture <- write_fixture_artifact(
  "full_shadow", file.path(artifact_root, "shadow"), identity
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

mutated_manifest_rejected(
  "phase0_manifest_hash", sha("wrong-phase0"),
  "Phase 0 manifest hash mutation must fail"
)
mutated_manifest_rejected(
  "phase1_manifest_hash", sha("wrong-phase1"),
  "Phase 1 manifest hash mutation must fail"
)
mutated_manifest_rejected(
  "phase2_manifest_hash", sha("wrong-phase2"),
  "Phase 2 manifest hash mutation must fail"
)
mutated_manifest_rejected(
  "dataset_file_sha256", sha("wrong-dataset-file"),
  "dataset file hash mutation must fail"
)
mutated_manifest_rejected(
  "dataset_matrix_sha256", sha("wrong-dataset-matrix"),
  "dataset matrix hash mutation must fail"
)
mutated_manifest_rejected(
  "canonical_setup_corpus_hash", sha("wrong-setup-corpus"),
  "canonical setup corpus mutation must fail"
)
mutated_manifest_rejected(
  "canonical_target_corpus_hash", sha("wrong-target-corpus"),
  "canonical target corpus mutation must fail"
)
mutated_manifest_rejected(
  "route_config_hash", sha("wrong-route"),
  "route configuration hash mutation must fail"
)
mutated_manifest_rejected(
  "runtime_abi", "wrong-runtime-abi",
  "runtime ABI mutation must fail"
)
mutated_manifest_rejected(
  "source_commit", strrep("2", 40L),
  "source commit mutation must fail"
)
mutated_manifest_rejected(
  "cuda_toolkit_version", 99999L,
  "CUDA toolkit mutation must fail"
)
mutated_manifest_rejected(
  "cuda_driver_version", 99999L,
  "CUDA driver mutation must fail"
)
mutated_manifest_rejected(
  "gpu_uuid", "GPU-forged-uuid",
  "GPU UUID mutation must fail"
)
mutated_manifest_rejected(
  "compute_capability", "9.9",
  "compute capability mutation must fail"
)
mutated_manifest_rejected(
  "cusolver_deterministic_mode", "disabled",
  "cuSOLVER deterministic mode mutation must fail"
)
mutated_manifest_rejected(
  "cublas_math_mode", "default",
  "cuBLAS math mode mutation must fail"
)
mutated_manifest_rejected(
  "cublas_atomics_mode", "allowed",
  "cuBLAS atomics mode mutation must fail"
)
mutated_manifest_rejected(
  "cublas_workspace_identity", sha("wrong-workspace"),
  "cuBLAS workspace identity mutation must fail"
)
mutated_manifest_rejected(
  "artifact_schema_version", "forged-artifact-schema",
  "artifact schema mutation must fail"
)
mutated_manifest_rejected(
  "shard_count", 4L,
  "shard count mutation must fail"
)

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

cat("PASS phase3 artifact contract route and identity scaffold\n")

cat("PASS phase3 artifact contract RED scaffold\n")
