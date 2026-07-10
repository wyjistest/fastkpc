fail <- function(message) stop(message, call. = FALSE)

assert_true <- function(value, message) {
  if (!isTRUE(value)) fail(message)
}

assert_error <- function(expression, pattern, message) {
  error <- tryCatch({
    force(expression)
    NULL
  }, error = identity)
  assert_true(
    inherits(error, "error") &&
      grepl(pattern, conditionMessage(error), fixed = TRUE),
    message
  )
}

source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")

contract <- fastkpc_full_cuda_prepared_s_input_contract()
assert_true(
  identical(contract$schema_version, "full-cuda-ci-phase2-input-v1") &&
    identical(
      contract$phase1_source_commit,
      "1560068ba8d635e806612554e11bbed92c0b8843c"
    ) &&
    identical(
      contract$metadata_schema_version,
      "full-cuda-ci-metadata-v4"
    ),
  "Phase 2 must pin its schema and Phase 1 source identity"
)
assert_true(
  identical(
    unname(contract$file_hashes[["manifest.json"]]),
    "b0990cfc932a5fcabc09ad25e352e7babb67fc8a127f11c7d2b88887c4940574"
  ),
  "Phase 2 must pin the Phase 1 manifest bytes"
)

census_dir <-
  "fastkpc/artifacts/full_cuda_ci/workload_census_351x48_v1"
data_path <- paste0(
  "fastkpc/artifacts/kpc_tprs_real_zhu/",
  "cancer_RD-causalDiscoveryInput.rds"
)
inputs <- fastkpc_full_cuda_prepared_s_load_inputs(
  census_dir = census_dir,
  data_path = data_path,
  contract = contract
)

assert_true(
  nrow(inputs$logical_tests) == 240489L &&
    nrow(inputs$residual_requests) == 110617L &&
    nrow(inputs$same_s_setup_metadata) == 8634L &&
    nrow(inputs$target_fit_metadata) == 110617L &&
    nrow(inputs$setup_observations) == 110617L &&
    nrow(inputs$target_risks) == 110617L,
  "authenticated Phase 1 tables must retain canonical counts"
)
assert_true(
  isTRUE(inputs$setup_invariance$validated) &&
    identical(inputs$setup_invariance$group_count, 8634L) &&
    identical(
      inputs$setup_invariance$fields,
      setdiff(
        names(inputs$setup_observations),
        "representative_residual_key_sha256"
      )
    ),
  "all non-lineage setup-observation fields must be audited"
)

manifest_hash <- inputs$input_hashes$actual_sha256[
  inputs$input_hashes$logical_path == "phase1/manifest.json"
]
assert_true(
  identical(manifest_hash, unname(contract$file_hashes[["manifest.json"]])) &&
    identical(
      inputs$manifest$source_commit,
      contract$phase1_artifact_source_commit
    ) &&
    grepl("^[0-9a-f]{64}$", inputs$phase1_input_bundle_hash),
  "loaded inputs must retain exact source, manifest, and bundle hashes"
)

setup_row <- inputs$same_s_setup_metadata[1L, , drop = FALSE]
key <- fastkpc_full_cuda_prepared_s_key(
  setup_row = setup_row,
  dataset_sha256 = inputs$dataset_sha256,
  R_version = inputs$manifest$R_version,
  mgcv_version = inputs$manifest$mgcv_version
)
assert_true(
  startsWith(
    key$payload,
    "schema_version=full-cuda-ci-prepared-s-key-v1\n"
  ) &&
    endsWith(key$payload, "\n") &&
    grepl("^[0-9a-f]{64}$", key$sha256),
  "PreparedSKey must use canonical LF-terminated UTF-8 and SHA-256"
)

assert_error(
  fastkpc_full_cuda_prepared_s_validate_key_mapping(
    payload = c(key$payload, paste0(key$payload, "x")),
    hash = c(key$sha256, key$sha256)
  ),
  "PreparedSKey hash collision",
  "PreparedSKey collisions must fail closed"
)
assert_error(
  fastkpc_full_cuda_prepared_s_validate_key_mapping(
    payload = c(key$payload, key$payload),
    hash = c(strrep("a", 64L), strrep("b", 64L))
  ),
  "PreparedSKey serialization mismatch",
  "PreparedSKey serialization mismatches must fail closed"
)

tampered_data_path <- tempfile("fastkpc-prepared-s-data-", fileext = ".rds")
on.exit(unlink(tampered_data_path, force = TRUE), add = TRUE)
assert_true(file.copy(data_path, tampered_data_path),
            "temporary dataset copy must succeed")
connection <- file(tampered_data_path, open = "ab")
writeBin(as.raw(0L), connection)
close(connection)
assert_error(
  fastkpc_full_cuda_prepared_s_load_inputs(
    census_dir = census_dir,
    data_path = tampered_data_path,
    contract = contract
  ),
  "canonical dataset file hash mismatch",
  "tampered Phase 2 input bytes must fail before parsing"
)

cat("PASS full CUDA CI Prepared-S input and key contract\n")
