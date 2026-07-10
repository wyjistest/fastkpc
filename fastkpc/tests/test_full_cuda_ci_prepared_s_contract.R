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
      "1560068ba8d635e806612554e11bbed92c0b8843"
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
assert_true(
  identical(
    contract$shard_file_bundle_sha256,
    "fdb6ef491e74bd1a9ac8bc01682d83a1fb57839517eb0c0205536b24618323ee"
  ),
  "Phase 2 must pin the complete Phase 1 shard byte bundle"
)

mutated_schema <- contract
mutated_schema$schema_version <- "full-cuda-ci-phase2-input-v2"
assert_error(
  fastkpc_full_cuda_prepared_s_validate_contract(mutated_schema),
  "canonical Phase 2 input contract mismatch",
  "mutated Phase 2 schema must fail before input parsing"
)

mutated_source <- contract
mutated_source$phase1_source_commit <- paste0(
  "0", substring(mutated_source$phase1_source_commit, 2L)
)
assert_error(
  fastkpc_full_cuda_prepared_s_validate_contract(mutated_source),
  "canonical Phase 2 input contract mismatch",
  "mutated Phase 1 source commit must fail before input parsing"
)

mutated_file_hash <- contract
mutated_file_hash$file_hashes[["manifest.json"]] <- strrep("0", 64L)
assert_error(
  fastkpc_full_cuda_prepared_s_validate_contract(mutated_file_hash),
  "canonical Phase 2 input contract mismatch",
  "mutated Phase 1 file hash must fail before input parsing"
)

shard_fixture_dir <- tempfile("fastkpc-prepared-s-shards-")
dir.create(shard_fixture_dir)
on.exit(unlink(shard_fixture_dir, recursive = TRUE, force = TRUE), add = TRUE)
shard_fixture_names <- c(
  "shard_0.rds", "shard_0.summary.json",
  "shard_1.rds", "shard_1.summary.json"
)
for (name in shard_fixture_names) {
  connection <- file(file.path(shard_fixture_dir, name), open = "wb")
  writeBin(charToRaw(paste0("authenticated bytes for ", name)), connection)
  close(connection)
}
shard_fixture_paths <- file.path(shard_fixture_dir, shard_fixture_names)
shard_fixture_rows <- data.frame(
  logical_path = paste0("phase1/shards/", shard_fixture_names),
  actual_sha256 = vapply(
    shard_fixture_paths,
    fastkpc_full_cuda_census_file_hash,
    character(1L)
  ),
  stringsAsFactors = FALSE
)
shard_fixture_bundle <- fastkpc_full_cuda_prepared_s_input_bundle_hash(
  shard_fixture_rows
)
authenticated_shards <- fastkpc_full_cuda_prepared_s_authenticate_shards(
  shard_dir = shard_fixture_dir,
  shard_count = 2L,
  expected_bundle_sha256 = shard_fixture_bundle
)
assert_true(
  nrow(authenticated_shards) == 4L &&
    identical(
      authenticated_shards$logical_path,
      sort(shard_fixture_rows$logical_path, method = "radix")
    ) &&
    identical(
      fastkpc_full_cuda_prepared_s_input_bundle_hash(
        authenticated_shards[nrow(authenticated_shards):1L, , drop = FALSE]
      ),
      shard_fixture_bundle
    ),
  "shard byte authentication must use every sorted logical path and hash"
)

unexpected_shard_path <- file.path(shard_fixture_dir, "unexpected.txt")
connection <- file(unexpected_shard_path, open = "wb")
writeBin(charToRaw("unexpected"), connection)
close(connection)
assert_error(
  fastkpc_full_cuda_prepared_s_authenticate_shards(
    shard_dir = shard_fixture_dir,
    shard_count = 2L,
    expected_bundle_sha256 = shard_fixture_bundle
  ),
  "Phase 1 shard file set mismatch",
  "unexpected Phase 1 shard names must fail closed"
)
unlink(unexpected_shard_path, force = TRUE)

connection <- file(shard_fixture_paths[[1L]], open = "ab")
writeBin(as.raw(0L), connection)
close(connection)
assert_error(
  fastkpc_full_cuda_prepared_s_authenticate_shards(
    shard_dir = shard_fixture_dir,
    shard_count = 2L,
    expected_bundle_sha256 = shard_fixture_bundle
  ),
  "Phase 1 shard byte bundle hash mismatch",
  "changed Phase 1 shard bytes must fail before deserialization"
)

merge_fixture_dir <- tempfile("fastkpc-prepared-s-merge-")
dir.create(merge_fixture_dir)
on.exit(unlink(merge_fixture_dir, recursive = TRUE, force = TRUE), add = TRUE)
merge_requests <- data.frame(
  residual_key_sha256 = sprintf("%064d", c(4L, 1L, 3L, 2L)),
  same_S_group_id = sprintf("%064d", c(102L, 101L, 102L, 101L)),
  target = c(4L, 1L, 3L, 2L),
  S_key = c("2", "1", "2", "1"),
  S_size = 1L,
  formula_class = "full-smooth",
  stringsAsFactors = FALSE
)
merge_shard_count <- 2L
merge_assigned <- fastkpc_full_cuda_census_assign_shards(
  merge_requests, merge_shard_count
)
merge_risk_config <- fastkpc_full_cuda_census_risk_config()
merge_context <- list(
  canonical_key_corpus_hash = fastkpc_full_cuda_census_key_set_hash(
    merge_assigned$residual_key_sha256
  ),
  canonical_logical_census_hash = strrep("d", 64L),
  dataset_sha256 = strrep("a", 64L),
  oracle_input_bundle_sha256 = strrep("b", 64L),
  source_commit = strrep("c", 40L),
  R_version = "R fixture 4.4.1",
  mgcv_version = "1.9-1-fixture",
  BLAS_identity = "fixture-blas",
  LAPACK_identity = "fixture-lapack",
  BLAS_thread_count = 1L,
  formula_semantics_version = "kpcalg_regrXonS_v1",
  mgcv_semantics_version = "legacy-mgcv-gam-default-selection-v1",
  risk_threshold_config_hash =
    fastkpc_full_cuda_census_metadata_hash(merge_risk_config),
  metadata_schema_version = fastkpc_full_cuda_census_metadata_schema_version(),
  data = matrix(0, nrow = 2L, ncol = 2L),
  risk_config = merge_risk_config,
  logical_tests = data.frame(
    logical_sequence_id = integer(),
    absolute_log_distance_from_alpha = numeric(),
    stringsAsFactors = FALSE
  )
)
merge_fixture_fit <- function(data, request_row, risk_config) {
  group_id <- request_row$same_S_group_id[[1L]]
  key_sha256 <- request_row$residual_key_sha256[[1L]]
  setup_fingerprint <- fastkpc_full_cuda_census_hash_utf8(
    paste0("setup:", group_id)
  )
  setup <- data.frame(
    same_S_group_id = group_id,
    representative_residual_key_sha256 = key_sha256,
    model_matrix_hash = fastkpc_full_cuda_census_hash_utf8(
      paste0("model:", group_id)
    ),
    penalty_hashes = I(list(fastkpc_full_cuda_census_hash_utf8(
      paste0("penalty:", group_id)
    ))),
    constraint_hash = fastkpc_full_cuda_census_hash_utf8(
      paste0("constraint:", group_id)
    ),
    setup_fingerprint = setup_fingerprint,
    stringsAsFactors = FALSE
  )
  target <- data.frame(
    residual_key_sha256 = key_sha256,
    same_S_group_id = group_id,
    setup_fingerprint = setup_fingerprint,
    shard_id = as.integer(request_row$shard_id[[1L]]),
    target = as.integer(request_row$target[[1L]]),
    fit_status = "success",
    stringsAsFactors = FALSE
  )
  risk <- data.frame(
    case_type = "target_key",
    residual_key_sha256 = key_sha256,
    logical_sequence_id = NA_integer_,
    same_S_group_id = group_id,
    high_condition = FALSE,
    rank_deficient = FALSE,
    near_constant_target = FALSE,
    near_constant_conditioner = FALSE,
    multi_penalty = FALSE,
    near_alpha = FALSE,
    mgcv_warning = FALSE,
    mgcv_nonconverged = FALSE,
    nonfinite_metadata = FALSE,
    condition_bucket = "finite_lt_1e4",
    near_alpha_bucket = NA_character_,
    stringsAsFactors = FALSE
  )
  list(setup_observation = setup, target_fit = target, risk_cases = risk)
}
merge_runs <- lapply(0:(merge_shard_count - 1L), function(shard_id) {
  fastkpc_full_cuda_census_run_shard(
    assigned_requests = merge_assigned,
    shard_id = shard_id,
    context = merge_context,
    output_dir = merge_fixture_dir,
    fit_fun = merge_fixture_fit
  )
})
merge_setup_observations <- fastkpc_full_cuda_census_bind_rows(lapply(
  merge_runs, function(value) value$payload$setup_observations
))
merge_setup_observations <- merge_setup_observations[order(
  merge_setup_observations$representative_residual_key_sha256,
  method = "radix"
), , drop = FALSE]
rownames(merge_setup_observations) <- NULL
merge_target_fits <- fastkpc_full_cuda_census_bind_rows(lapply(
  merge_runs, function(value) value$payload$target_fits
))
merge_target_fits <- merge_target_fits[order(
  merge_target_fits$residual_key_sha256, method = "radix"
), , drop = FALSE]
rownames(merge_target_fits) <- NULL
merge_target_risks <- fastkpc_full_cuda_census_bind_rows(lapply(
  merge_runs, function(value) value$payload$target_risks
))
merge_target_risks <- merge_target_risks[order(
  merge_target_risks$residual_key_sha256, method = "radix"
), , drop = FALSE]
rownames(merge_target_risks) <- NULL
merge_same_s_setups <- fastkpc_full_cuda_census_compress_setups(
  merge_setup_observations
)
merge_risk_cases <- fastkpc_full_cuda_census_risk_cases(
  merge_target_risks, merge_context$logical_tests
)
merge_expected_hashes <- fastkpc_full_cuda_census_authenticated_metadata_hashes(
  list(
    setup_observation_metadata = merge_setup_observations,
    same_s_setup_metadata = merge_same_s_setups,
    target_fit_metadata = merge_target_fits,
    target_risk_metadata = merge_target_risks,
    risk_cases = merge_risk_cases
  )
)
rm(merge_runs)

streamed_merge <- fastkpc_full_cuda_prepared_s_merge_shards(
  requests = merge_requests,
  same_s_setup_metadata = merge_same_s_setups,
  target_fit_metadata = merge_target_fits,
  risk_cases = merge_risk_cases,
  shard_count = merge_shard_count,
  context = merge_context,
  shard_dir = merge_fixture_dir
)
assert_true(
  identical(
    names(streamed_merge),
    c(
      "setup_observation_metadata", "target_risk_metadata",
      "authenticated_metadata_hashes"
    )
  ) &&
    identical(
      fastkpc_full_cuda_prepared_s_named_character(
        streamed_merge$authenticated_metadata_hashes
      ),
      fastkpc_full_cuda_prepared_s_named_character(merge_expected_hashes)
    ),
  "Phase 2 shard merge must retain only authenticated setup and risk metadata"
)

wrong_manifest_paths <- fastkpc_full_cuda_census_shard_paths(
  merge_fixture_dir, 0L
)
wrong_manifest_payload <- readRDS(wrong_manifest_paths$rds)
wrong_manifest_summary <- jsonlite::read_json(
  wrong_manifest_paths$summary_json, simplifyVector = TRUE
)
wrong_manifest_payload$manifest$dataset_sha256 <- strrep("f", 64L)
wrong_manifest_summary$manifest <- wrong_manifest_payload$manifest
wrong_manifest_summary$manifest_hash <- fastkpc_full_cuda_census_metadata_hash(
  wrong_manifest_payload$manifest
)
wrong_manifest_hashes <- fastkpc_full_cuda_census_shard_metadata_hashes(
  wrong_manifest_payload
)
for (name in names(wrong_manifest_hashes)) {
  wrong_manifest_summary[[name]] <- wrong_manifest_hashes[[name]]
}
saveRDS(wrong_manifest_payload, wrong_manifest_paths$rds, version = 2)
fastkpc_full_cuda_write_json(
  wrong_manifest_summary, wrong_manifest_paths$summary_json
)
assert_error(
  fastkpc_full_cuda_prepared_s_merge_shards(
    requests = merge_requests,
    same_s_setup_metadata = merge_same_s_setups,
    target_fit_metadata = merge_target_fits,
    risk_cases = merge_risk_cases,
    shard_count = merge_shard_count,
    context = merge_context,
    shard_dir = merge_fixture_dir
  ),
  "shard manifest mismatch",
  "Phase 2 shard merge must validate each exact expected manifest"
)

setup_row <- data.frame(
  same_S_group_id = strrep("a", 64L),
  S_key = "1|2",
  S_size = 2L,
  formula_class = "full-smooth",
  formula_semantics_version = contract$formula_semantics_version,
  R_version = contract$R_version,
  mgcv_version = contract$mgcv_version,
  stringsAsFactors = FALSE
)
key <- fastkpc_full_cuda_prepared_s_key(
  setup_row = setup_row,
  dataset_sha256 = contract$dataset_matrix_sha256,
  R_version = contract$R_version,
  mgcv_version = contract$mgcv_version
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
  fastkpc_full_cuda_prepared_s_key(
    setup_row = setup_row,
    dataset_sha256 = contract$dataset_matrix_sha256,
    R_version = contract$R_version,
    mgcv_version = contract$mgcv_version,
    hash_fun = function(payload) strrep("A", 64L)
  ),
  "PreparedSKey hash function returned an invalid hash",
  "custom PreparedSKey hashes must be lowercase SHA-256"
)
assert_error(
  fastkpc_full_cuda_prepared_s_validate_key_mapping(
    payload = key$payload,
    hash = "not-a-sha256"
  ),
  "PreparedSKey hash is invalid",
  "PreparedSKey mappings must reject malformed hashes"
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

census_dir <- Sys.getenv(
  "FASTKPC_PHASE1_CENSUS_DIR",
  unset = "fastkpc/artifacts/full_cuda_ci/workload_census_351x48_v1"
)
data_path <- Sys.getenv(
  "FASTKPC_PHASE2_DATA_PATH",
  unset = paste0(
    "fastkpc/artifacts/kpc_tprs_real_zhu/",
    "cancer_RD-causalDiscoveryInput.rds"
  )
)
artifact_path_present <- c(dir.exists(census_dir), file.exists(data_path))

if (!any(artifact_path_present)) {
  cat("SKIP real-artifact integration: exact artifact paths unavailable\n")
} else {
  assert_true(
    all(artifact_path_present),
    "real-artifact integration paths must be jointly available"
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
  shard_ids <- 0:(contract$shard_count - 1L)
  expected_shard_logical_paths <- sort(c(
    paste0("phase1/shards/shard_", shard_ids, ".rds"),
    paste0("phase1/shards/shard_", shard_ids, ".summary.json")
  ), method = "radix")
  shard_input_hashes <- inputs$input_hashes[
    startsWith(inputs$input_hashes$logical_path, "phase1/shards/"),
    , drop = FALSE
  ]
  assert_true(
    identical(manifest_hash, unname(contract$file_hashes[["manifest.json"]])) &&
      identical(
        inputs$manifest$source_commit,
        contract$phase1_source_commit
      ) &&
      identical(
        shard_input_hashes$logical_path,
        expected_shard_logical_paths
      ) &&
      identical(
        fastkpc_full_cuda_prepared_s_input_bundle_hash(shard_input_hashes),
        contract$shard_file_bundle_sha256
      ) &&
      nrow(inputs$input_hashes) == 137L &&
      identical(
        inputs$phase1_input_bundle_hash,
        fastkpc_full_cuda_prepared_s_input_bundle_hash(inputs$input_hashes)
      ),
    "loaded inputs must retain exact source, manifest, and bundle hashes"
  )

  real_key <- fastkpc_full_cuda_prepared_s_key(
    setup_row = inputs$same_s_setup_metadata[1L, , drop = FALSE],
    dataset_sha256 = inputs$dataset_sha256,
    R_version = inputs$manifest$R_version,
    mgcv_version = inputs$manifest$mgcv_version
  )
  assert_true(
    grepl("^[0-9a-f]{64}$", real_key$sha256),
    "real-artifact PreparedSKey must retain canonical SHA-256"
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

  cat("PASS real-artifact integration\n")
}

cat("PASS full CUDA CI Prepared-S input and key contract\n")
