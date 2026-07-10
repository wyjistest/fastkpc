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
skip_real_artifacts <- identical(
  Sys.getenv("FASTKPC_PHASE2_SKIP_REAL", unset = "0"), "1"
)

if (skip_real_artifacts) {
  cat("SKIP real-artifact integration: FASTKPC_PHASE2_SKIP_REAL=1\n")
} else if (!any(artifact_path_present)) {
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

  penalty_counts <- c(1L, 3L, 4L, 5L, 6L, 7L)
  representatives <- do.call(rbind, lapply(penalty_counts, function(count) {
    rows <- inputs$same_s_setup_metadata[
      inputs$same_s_setup_metadata$penalty_count == count,
      , drop = FALSE
    ]
    rows[
      order(rows$same_S_group_id, method = "radix")[[1L]],
      , drop = FALSE
    ]
  }))
  rownames(representatives) <- NULL
  assert_true(
    identical(as.integer(representatives$penalty_count), penalty_counts),
    "real PreparedSSetup representatives must cover canonical penalty counts"
  )

  prepared <- lapply(seq_len(nrow(representatives)), function(index) {
    fastkpc_full_cuda_build_prepared_s_setup(
      inputs = inputs,
      setup_row = representatives[index, , drop = FALSE]
    )
  })
  for (index in seq_along(prepared)) {
    fastkpc_full_cuda_validate_prepared_s_setup(
      setup = prepared[[index]],
      setup_row = representatives[index, , drop = FALSE],
      dataset_sha256 = inputs$dataset_sha256
    )
  }
  assert_true(
    identical(
      vapply(prepared, function(value) {
        length(value$penalty_blocks)
      }, integer(1L)),
      penalty_counts
    ),
    "PreparedSSetup must retain every canonical penalty block"
  )
  assert_true(
    all(vapply(seq_along(prepared), function(index) {
      identical(
        prepared[[index]]$phase1_model_matrix_hash,
        as.character(representatives$model_matrix_hash[[index]])
      ) &&
        !identical(
          fastkpc_full_cuda_census_metadata_hash(prepared[[index]]$X),
          as.character(representatives$model_matrix_hash[[index]])
        )
    }, logical(1L))),
    paste(
      "PreparedSSetup must retain the Phase 1 fitted-lpmatrix hash as",
      "lineage without requiring provider-X raw-hash equality"
    )
  )

  leaked <- prepared[[1L]]
  leaked$nested <- list(y = inputs$data[, 1L])
  assert_error(
    fastkpc_full_cuda_validate_prepared_s_setup(
      leaked,
      representatives[1L, , drop = FALSE],
      inputs$dataset_sha256
    ),
    "response-bearing field",
    "PreparedSSetup must reject nested response leakage"
  )

  leaked_environment <- prepared[[1L]]
  leaked_environment$formula_environment <- new.env(parent = emptyenv())
  assert_error(
    fastkpc_full_cuda_validate_prepared_s_setup(
      leaked_environment,
      representatives[1L, , drop = FALSE],
      inputs$dataset_sha256
    ),
    "executable object",
    "PreparedSSetup must reject retained environments"
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

fixture_index <- seq_len(80L)
fixture_data <- cbind(
  sin(fixture_index / 5) + fixture_index / 100,
  cos(fixture_index / 7) + fixture_index / 200,
  ((fixture_index * 7L) %% 23L) / 23 + sin(fixture_index / 11),
  ((fixture_index * 11L) %% 29L) / 29 + cos(fixture_index / 13)
)
storage.mode(fixture_data) <- "double"
fixture_S <- 2:4
fixture_dataset_sha256 <- fastkpc_full_cuda_data_hash(fixture_data)
fixture_formula_class <- fastkpc_regrxons_formula_class(fixture_S)
fixture_group_payload <- fastkpc_full_cuda_census_same_s_payload(
  S = fixture_S,
  formula_class = fixture_formula_class,
  data_hash = fixture_dataset_sha256,
  n = nrow(fixture_data),
  p = ncol(fixture_data)
)
fixture_group_id <- fastkpc_full_cuda_census_hash_utf8(
  fixture_group_payload
)
fixture_request <- data.frame(
  residual_key_sha256 = strrep("b", 64L),
  target = 1L,
  S_key = paste(fixture_S, collapse = "|"),
  S_size = length(fixture_S),
  formula_class = fixture_formula_class,
  same_S_group_id = fixture_group_id,
  shard_id = 0L,
  stringsAsFactors = FALSE
)
fixture_frame <- data.frame(
  fixture_data[, c(1L, fixture_S), drop = FALSE],
  check.names = FALSE
)
names(fixture_frame) <- paste0("x", seq_len(ncol(fixture_frame)))
fixture_formula <- fastkpc_full_cuda_census_formula_function(
  length(fixture_S)
)(1L, 2L:(1L + length(fixture_S)))
fixture_fit <- mgcv::gam(
  formula = fixture_formula,
  data = fixture_frame,
  method = "GCV.Cp"
)
fixture_components <- fastkpc_full_cuda_census_setup_components(
  fit = fixture_fit,
  request_row = fixture_request,
  data = fixture_data
)
fixture_setup_row <- fixture_components$row
fixture_inputs <- list(
  data = fixture_data,
  dataset_sha256 = fixture_dataset_sha256,
  manifest = list(
    R_version = R.version.string,
    mgcv_version = as.character(utils::packageVersion("mgcv"))
  ),
  same_s_setup_metadata = fixture_setup_row
)

fixture_provider <- fastkpc_full_cuda_prepared_s_provider_descriptor()
assert_true(
  identical(
    names(fixture_provider$contrasts), c("unordered", "ordered")
  ) &&
    identical(
      unname(fixture_provider$contrasts),
      c("contr.treatment", "contr.poly")
    ) &&
    identical(fixture_provider$runtime_na_action, "na.omit") &&
    identical(fixture_provider$contract_na_action, "na.fail") &&
    all(vapply(
      fixture_provider[c(
        "regrXonS_body_sha256", "full_smooth_formula_body_sha256",
        "additive_smooth_formula_body_sha256"
      )],
      function(value) grepl("^[0-9a-f]{64}$", value),
      logical(1L)
    )),
  "provider fingerprint inputs must retain exact helper, contrast, and NA lineage"
)

fixture_layout <- fastkpc_full_cuda_prepared_s_layout(
  fixture_data, fixture_S
)
assert_true(
    identical(names(fixture_layout$data), paste0("x", 1:4)) &&
    identical(fixture_layout$data[[1L]], rep(0, nrow(fixture_data))) &&
    identical(
      unname(as.matrix(fixture_layout$data[, -1L, drop = FALSE])),
      unname(fixture_data[, fixture_S, drop = FALSE])
    ) &&
    identical(deparse(fixture_layout$formula), deparse(fixture_formula)),
  "PreparedSSetup layout must use the exact zero-response legacy formula"
)

fixture_setup <- fastkpc_full_cuda_build_prepared_s_setup(
  inputs = fixture_inputs,
  setup_row = fixture_setup_row
)
fastkpc_full_cuda_validate_prepared_s_setup(
  fixture_setup,
  fixture_setup_row,
  fixture_dataset_sha256
)
assert_true(
  identical(fixture_setup$schema_version,
            "full-cuda-ci-prepared-s-setup-v1") &&
    length(fixture_setup$penalty_blocks) == 3L &&
    identical(fixture_setup$constraint_mode, "identity") &&
    identical(dim(fixture_setup$constraint),
              c(0L, ncol(fixture_setup$X))) &&
    is.null(fixture_setup$constraint_nullspace) &&
    identical(fixture_setup$nullspace_gram_policy, "alias-gram") &&
    is.null(fixture_setup$nullspace_gram_matrix) &&
    is.null(fixture_setup$H) &&
    is.null(fixture_setup$weights) &&
    identical(fixture_setup$weights_policy, "none-or-unit") &&
    is.null(fixture_setup$offset) &&
    identical(fixture_setup$offset_policy, "none-or-zero") &&
    is.null(fixture_setup$sp_mapping) &&
    is.numeric(fixture_setup$sp_mapping_offset) &&
    length(fixture_setup$sp_mapping_offset) == 3L &&
    all(fixture_setup$sp_mapping_offset == 0) &&
    is.null(fixture_setup$min_sp) &&
    !any(c("G", "y", "target", "sp", "lsp0", "selected_sp") %in%
         names(fixture_setup)),
  "PreparedSSetup must use compact response-independent neutral encodings"
)

assert_setup_error <- function(mutator, pattern, message) {
  candidate <- fixture_setup
  candidate <- mutator(candidate)
  assert_error(
    fastkpc_full_cuda_validate_prepared_s_setup(
      candidate,
      fixture_setup_row,
      fixture_dataset_sha256
    ),
    pattern,
    message
  )
}

refresh_self_fingerprints <- function(value) {
  value$semantic_fingerprint <-
    fastkpc_full_cuda_prepared_s_semantic_fingerprint(value)
  value$representation_fingerprint <-
    fastkpc_full_cuda_prepared_s_representation_fingerprint(value)
  value
}

approved_fingerprint_fields <- fixture_setup[c(
  "provider_fingerprint", "semantic_fingerprint",
  "representation_fingerprint", "phase1_setup_fingerprint"
)]
assert_true(
  length(fastkpc_full_cuda_prepared_s_find_response_fields(
    approved_fingerprint_fields
  )) == 0L,
  "approved setup and provider fingerprints must remain whitelisted"
)

assert_setup_error(
  function(value) {
    attr(value$smooth_terms[[1L]], "UZ") <- numeric()
    refresh_self_fingerprints(value)
  },
  "response-bearing field",
  "UZ hidden in smooth attributes must fail the safety scanner"
)
assert_setup_error(
  function(value) {
    names(value$smooth_terms[[1L]]) <- "Xu"
    refresh_self_fingerprints(value)
  },
  "response-bearing field",
  "Xu hidden in smooth vector element names must fail the safety scanner"
)
assert_setup_error(
  function(value) {
    value$smooth_terms[[1L]][[1L]] <- "UZ"
    refresh_self_fingerprints(value)
  },
  "response-bearing field",
  "UZ hidden in smooth values must fail the safety scanner"
)
assert_setup_error(
  function(value) {
    value$smooth_terms[[1L]][[1L]] <- "Xu"
    refresh_self_fingerprints(value)
  },
  "response-bearing field",
  "Xu hidden in smooth values must fail the safety scanner"
)
assert_setup_error(
  function(value) {
    value$smooth_terms[[1L]][[1L]] <- "target_fingerprint"
    refresh_self_fingerprints(value)
  },
  "response-bearing field",
  "generic target fingerprints in smooth values must fail the safety scanner"
)
assert_setup_error(
  function(value) {
    value$smooth_terms[[1L]][[1L]] <- "target_state_fingerprint"
    refresh_self_fingerprints(value)
  },
  "response-bearing field",
  "target-state fingerprints in smooth values must fail the safety scanner"
)

assert_setup_error(
  function(value) {
    value$nested <- list(y = fixture_data[, 1L])
    value
  },
  "response-bearing field",
  "nested response fields must fail closed"
)
assert_setup_error(
  function(value) {
    attr(value$cmX, "target") <- 1L
    value
  },
  "response-bearing field",
  "response-bearing attribute names must fail closed"
)
assert_setup_error(
  function(value) {
    attr(value$cmX, "metadata") <- list(alias = "target")
    value
  },
  "response-bearing field",
  "response-bearing attribute values must fail closed"
)
assert_setup_error(
  function(value) {
    value$selected_sp <- 1
    value
  },
  "response-bearing field",
  "selected smoothing parameters belong to TargetState"
)

executable_injections <- list(
  environment = new.env(parent = emptyenv()),
  formula = fixture_formula,
  call = quote(sum(1, 2)),
  function_object = function() NULL
)
for (label in names(executable_injections)) {
  object <- executable_injections[[label]]
  assert_setup_error(
    function(value) {
      value$coefficient_labels <- object
      value
    },
    "executable object",
    paste("PreparedSSetup must reject", label)
  )
}
assert_setup_error(
  function(value) {
    attr(value$cmX, "payload") <- new.env(parent = emptyenv())
    value
  },
  "executable object",
  "executable objects hidden in attributes must fail closed"
)
assert_setup_error(
  function(value) {
    value$coefficient_labels <- fixture_fit$smooth[[1L]]
    value
  },
  "executable object",
  "raw mgcv smooth objects must fail closed"
)
assert_setup_error(
  function(value) {
    value$coefficient_labels <- fixture_fit
    value
  },
  "executable object",
  "fitted gam objects must fail closed"
)

assert_setup_error(
  function(value) {
    value$X[[1L]] <- Inf
    value
  },
  "X must be finite",
  "nonfinite provider matrices must fail closed"
)
assert_setup_error(
  function(value) {
    value$smooth_shift[[1L]] <- fixture_data[, 1L]
    refresh_self_fingerprints(value)
  },
  "smooth metadata shape mismatch",
  paste(
    "response-sized payloads in smooth descriptors must fail after",
    "semantic and representation fingerprints are refreshed"
  )
)
assert_setup_error(
  function(value) {
    value$sp_mapping <- do.call(
      rbind,
      rep(list(fixture_data[, 1L]), length(value$penalty_blocks))
    )
    refresh_self_fingerprints(value)
  },
  "smoothing mapping mismatch",
  paste(
    "finite response-bearing smoothing mappings must fail after",
    "semantic and representation fingerprints are refreshed"
  )
)
assert_setup_error(
  function(value) {
    value$sp_mapping_offset[[1L]] <- fixture_data[[1L]]
    refresh_self_fingerprints(value)
  },
  "smoothing mapping mismatch",
  "nonzero smoothing mapping offsets must fail canonical v1 validation"
)
assert_setup_error(
  function(value) {
    value$min_sp <- fixture_data[seq_along(value$penalty_blocks), 1L]
    refresh_self_fingerprints(value)
  },
  "smoothing mapping mismatch",
  "non-NULL minimum smoothing parameters must fail canonical v1 validation"
)
assert_setup_error(
  function(value) {
    value$weights <- rep(1, nrow(value$X))
    value$weights_policy <- "none"
    value$weighted_X_policy <- "sqrt-weights-row-scaled"
    value$gram_matrix <- fastkpc_full_cuda_prepared_s_matrix(
      crossprod(value$X * sqrt(value$weights))
    )
    refresh_self_fingerprints(value)
  },
  "weights policy mismatch",
  paste(
    "explicit all-one weights must be rejected after dependent",
    "algebra and fingerprints are refreshed"
  )
)
assert_setup_error(
  function(value) {
    value$offset <- rep(0, nrow(value$X))
    value$offset_policy <- "none"
    refresh_self_fingerprints(value)
  },
  "offset policy mismatch",
  paste(
    "explicit all-zero offset must be rejected after dependent",
    "fingerprints are refreshed"
  )
)
assert_setup_error(
  function(value) {
    value$same_S_group_id <- strrep("0", 64L)
    value
  },
  "lineage mismatch",
  "same-S lineage tampering must fail closed"
)
assert_setup_error(
  function(value) {
    value$phase1_setup_fingerprint <- "not-a-sha256"
    value
  },
  "lineage fingerprint",
  "malformed Phase 1 fingerprints must fail closed"
)
assert_setup_error(
  function(value) {
    value$prepared_s_key_sha256 <- "not-a-sha256"
    value
  },
  "PreparedSKey",
  "malformed PreparedSKey fingerprints must fail closed"
)

assert_setup_error(
  function(value) {
    value$penalty_order <- rev(value$penalty_order)
    value
  },
  "penalty order mismatch",
  "penalty order tampering must fail closed"
)
assert_setup_error(
  function(value) {
    value$penalty_offsets[[1L]] <- value$penalty_offsets[[1L]] + 1L
    value
  },
  "penalty offset mismatch",
  "penalty offset tampering must fail closed"
)
assert_setup_error(
  function(value) {
    value$penalty_blocks[[1L]][[1L]] <-
      value$penalty_blocks[[1L]][[1L]] + 1
    value
  },
  "penalty hash mismatch",
  "penalty block tampering must fail Phase 1 hash validation"
)

assert_setup_error(
  function(value) {
    value$provider_fingerprint <- strrep("0", 64L)
    value
  },
  "provider fingerprint mismatch",
  "provider lineage tampering must fail closed"
)
assert_setup_error(
  function(value) {
    value$representation_fingerprint <- strrep("0", 64L)
    value
  },
  "representation fingerprint mismatch",
  "representation fingerprint tampering must fail closed"
)
assert_setup_error(
  function(value) {
    value$semantic_fingerprint <- strrep("0", 64L)
    value
  },
  "semantic fingerprint mismatch",
  "semantic fingerprint tampering must fail closed"
)

cat("PASS canonical self-contained PreparedSSetup contract\n")

cat("PASS full CUDA CI Prepared-S input and key contract\n")
