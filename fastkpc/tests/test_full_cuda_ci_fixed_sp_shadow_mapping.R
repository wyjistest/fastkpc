fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_error <- function(expression, pattern, message) {
  error <- tryCatch({
    force(expression)
    NULL
  }, error = identity)
  assert_true(
    inherits(error, "error") && grepl(pattern, conditionMessage(error)),
    message
  )
}

source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/full_cuda_ci_phase3_artifacts.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_shadow.R")

catalog <- fastkpc_full_cuda_open_fixed_sp_catalog(
  phase0_dir = "fastkpc/artifacts/full_cuda_ci/oracle_351x48_v1",
  phase1_dir =
    "fastkpc/artifacts/full_cuda_ci/workload_census_351x48_v1",
  phase2_dir =
    "fastkpc/artifacts/full_cuda_ci/prepared_s_contract_v1",
  data_path = paste0(
    "fastkpc/artifacts/kpc_tprs_real_zhu/",
    "cancer_RD-causalDiscoveryInput.rds"
  )
)

plan <- fastkpc_full_cuda_shadow_plan(catalog)
assert_true(nrow(plan$direct_tests) == 2213L, "direct level-0 count")
assert_true(
  nrow(plan$conditional_tests) == 238276L,
  "conditional logical count"
)
assert_true(
  all(is.na(plan$direct_tests$residual_key_x)) &&
    all(is.na(plan$direct_tests$residual_key_y)),
  "direct tests have no residual keys"
)
assert_true(
  all(nzchar(plan$conditional_tests$residual_key_x)) &&
    all(nzchar(plan$conditional_tests$residual_key_y)),
  "conditional tests have two residual keys"
)
assert_true(
  all(plan$conditional_tests$prepared_s_key_x ==
        plan$conditional_tests$prepared_s_key_y),
  "conditional endpoints share PreparedSKey"
)
assert_true(
  length(unique(plan$conditional_tests$prepared_s_key_x)) == 8634L,
  "all canonical setups consumed"
)
assert_true(
  identical(
    sort(c(
      plan$direct_tests$logical_sequence_id,
      plan$conditional_tests$logical_sequence_id
    )),
    seq_len(240489L)
  ),
  "logical sequence coverage"
)
assert_true(
  identical(
    plan$conditional_tests$shard_id,
    as.integer((match(
      plan$conditional_tests$prepared_s_key_x,
      sort(catalog$setup_index$prepared_s_key_sha256, method = "radix")
    ) - 1L) %% 64L)
  ) && identical(sort(unique(plan$conditional_tests$shard_id)), 0:63),
  "conditional rows use deterministic canonical 64-shard assignment"
)
assert_true(
  identical(
    plan$phase2_setup_index_csv_sha256,
    unname(fastkpc_full_cuda_fixed_sp_catalog_contract()[[
      "phase2_file_sha256"
    ]][["prepared_s_setup_index.csv"]])
  ) &&
    grepl("^[0-9a-f]{64}$", plan$setup_association_sha256),
  "shadow plan binds authenticated Phase 2 setup associations"
)
assert_true(
  identical(
    plan$phase2_target_state_index_rds_sha256,
    unname(fastkpc_full_cuda_fixed_sp_catalog_contract()[[
      "target_state_index_rds_sha256"
    ]])
  ) && grepl("^[0-9a-f]{64}$", plan$target_association_sha256),
  "shadow plan binds authenticated Phase 2 target associations"
)

association_swap_catalog <- catalog
association_swap_keys <-
  association_swap_catalog$setup_index$prepared_s_key_sha256[1:2]
association_swap_shards <- as.integer((match(
  association_swap_keys,
  sort(catalog$setup_index$prepared_s_key_sha256, method = "radix")
) - 1L) %% 64L)
assert_true(
  length(unique(association_swap_shards)) == 2L,
  "association-swap probe must move setup ownership across shards"
)
association_swap_catalog$setup_index$same_S_group_id[1:2] <-
  rev(association_swap_catalog$setup_index$same_S_group_id[1:2])
assert_error(
  fastkpc_full_cuda_shadow_plan(association_swap_catalog),
  "authenticated Phase 2 setup association mismatch",
  paste(
    "same-S group to PreparedSKey association swap must fail closed even",
    "when the canonical key set is unchanged"
  )
)

target_association_swap_catalog <- catalog
target_setup_shards <- as.integer((match(
  target_association_swap_catalog$setup_index$prepared_s_key_sha256,
  sort(catalog$setup_index$prepared_s_key_sha256, method = "radix")
) - 1L) %% 64L)
target_setup_a <- 1L
target_setup_b <- which(target_setup_shards != target_setup_shards[[1L]])[[1L]]
target_swap_groups <-
  target_association_swap_catalog$setup_index$same_S_group_id[
    c(target_setup_a, target_setup_b)
  ]
target_setup_metadata <-
  target_association_swap_catalog$inputs$same_s_setup_metadata
target_swap_setup_match <- match(
  target_swap_groups, target_setup_metadata$same_S_group_id
)
target_swap_fingerprints <-
  target_setup_metadata$setup_fingerprint[target_swap_setup_match]
target_metadata <-
  target_association_swap_catalog$inputs$target_fit_metadata
target_rows_a <- which(target_metadata$same_S_group_id == target_swap_groups[[1L]])
target_rows_b <- which(target_metadata$same_S_group_id == target_swap_groups[[2L]])
target_swap_rows <- c(target_rows_a, target_rows_b)
target_swap_identity <- target_metadata[
  target_swap_rows, c("residual_key_sha256", "target"), drop = FALSE
]
assert_true(
  !anyNA(target_swap_setup_match) &&
    length(target_rows_a) > 0L && length(target_rows_b) > 0L &&
    target_setup_shards[[target_setup_a]] !=
      target_setup_shards[[target_setup_b]],
  "target-association swap probe must cover populated different-shard setups"
)
target_metadata$same_S_group_id[target_rows_a] <- target_swap_groups[[2L]]
target_metadata$setup_fingerprint[target_rows_a] <-
  target_swap_fingerprints[[2L]]
target_metadata$same_S_group_id[target_rows_b] <- target_swap_groups[[1L]]
target_metadata$setup_fingerprint[target_rows_b] <-
  target_swap_fingerprints[[1L]]
target_association_swap_catalog$inputs$target_fit_metadata <- target_metadata
assert_true(
  identical(
    target_association_swap_catalog$inputs$target_fit_metadata[
      target_swap_rows, c("residual_key_sha256", "target"), drop = FALSE
    ],
    target_swap_identity
  ),
  "target-association swap probe must preserve residual keys and targets"
)
assert_error(
  fastkpc_full_cuda_shadow_plan(target_association_swap_catalog),
  "authenticated Phase 2 target association mismatch",
  paste(
    "self-consistent residual-key target association swaps must fail closed",
    "before conditional shard assignment"
  )
)

full_scope <- fastkpc_full_cuda_fixed_sp_scope(catalog, "full")
logical_tests <- catalog$inputs$logical_tests
authenticated_phase2_evidence <-
  fastkpc_full_cuda_shadow_authenticated_phase2_records(catalog)
map_units <- function(
    logical = logical_tests, setup = full_scope$setup_rows,
    target = full_scope$target_rows, setup_index = catalog$setup_index,
    logical_contract = fastkpc_full_cuda_shadow_logical_contract(logical)) {
  fastkpc_full_cuda_shadow_map_execution_units(
    logical_tests = logical,
    setup_rows = setup,
    target_rows = target,
    setup_index = setup_index,
    expected_logical_contract = logical_contract,
    authenticated_setup_evidence = authenticated_phase2_evidence$setup,
    authenticated_target_evidence = authenticated_phase2_evidence$target,
    shard_count = 64L
  )
}

first_conditional <- which(logical_tests$level > 0L)[[1L]]
endpoint_key <- logical_tests$residual_key_x[[first_conditional]]
endpoint_row <- match(endpoint_key, full_scope$target_rows$residual_key_sha256)

missing_endpoint <- full_scope$target_rows
missing_endpoint$residual_key_sha256[[endpoint_row]] <- strrep("f", 64L)
assert_error(
  map_units(target = missing_endpoint),
  "authenticated Phase 2 target association mismatch",
  "missing conditional endpoint key must fail closed"
)
rm(missing_endpoint)

duplicate_endpoint <- full_scope$target_rows
duplicate_endpoint$residual_key_sha256[[2L]] <-
  duplicate_endpoint$residual_key_sha256[[1L]]
assert_error(
  map_units(target = duplicate_endpoint),
  "duplicate residual key",
  "duplicate target endpoint key must fail closed"
)
rm(duplicate_endpoint)

fingerprint_conflict <- full_scope$target_rows
fingerprint_conflict$setup_fingerprint[[endpoint_row]] <- strrep("e", 64L)
assert_error(
  map_units(target = fingerprint_conflict),
  "authenticated Phase 2 target association mismatch",
  "target/setup fingerprint conflict must fail closed"
)
rm(fingerprint_conflict)

cross_setup <- logical_tests
row_y <- cross_setup$y[[first_conditional]]
endpoint_x_setup <- full_scope$target_rows$prepared_s_key_sha256[[endpoint_row]]
cross_target <- which(
  full_scope$target_rows$target == row_y &
    full_scope$target_rows$prepared_s_key_sha256 != endpoint_x_setup
)[[1L]]
cross_setup$residual_key_y[[first_conditional]] <-
  full_scope$target_rows$residual_key_sha256[[cross_target]]
assert_error(
  map_units(logical = cross_setup),
  "different PreparedSKeys",
  "cross-setup conditional endpoints must fail closed"
)
rm(cross_setup)

direct_with_key <- logical_tests
direct_with_key$residual_key_x[[1L]] <- endpoint_key
assert_error(
  map_units(logical = direct_with_key),
  "direct logical rows must not carry residual keys",
  "direct rows carrying residual keys must fail closed"
)
rm(direct_with_key)

conditional_without_key <- logical_tests
conditional_without_key$residual_key_y[[first_conditional]] <- NA_character_
assert_error(
  map_units(logical = conditional_without_key),
  "conditional logical rows require two residual keys",
  "conditional rows missing residual keys must fail closed"
)
rm(conditional_without_key)

duplicate_logical_id <- logical_tests
duplicate_logical_id$logical_sequence_id[[2L]] <-
  duplicate_logical_id$logical_sequence_id[[1L]]
assert_error(
  map_units(
    logical = duplicate_logical_id,
    logical_contract =
      fastkpc_full_cuda_shadow_logical_contract(duplicate_logical_id)
  ),
  "duplicate logical_sequence_id",
  "duplicate logical sequence IDs must fail closed"
)

direct_output_dir <- tempfile("full-cuda-ci-direct-")
direct_artifact <- fastkpc_full_cuda_shadow_write_direct_ci(
  catalog = catalog,
  output_dir = direct_output_dir,
  plan = plan
)
validated_direct <- fastkpc_full_cuda_phase3_validate_direct_ci_payload(
  output_dir = direct_output_dir,
  catalog = catalog
)
direct_schema <- c(
  "logical_sequence_id", "source_sequence_id", "source_task_index",
  "level", "x", "y", "S_key", "residual_key_x", "residual_key_y",
  "reference_p_value", "candidate_p_value",
  "absolute_p_value_difference", "alpha", "reference_decision",
  "candidate_decision", "decision_flip", "backend",
  "low_rank_backend", "backend_error", "spectra_fallback"
)
assert_true(
  identical(names(direct_artifact$rows), direct_schema) &&
    identical(names(validated_direct$payload$rows), direct_schema),
  "direct-CI row schema must be exact"
)
assert_true(
  nrow(direct_artifact$rows) == 2213L &&
    identical(direct_artifact$rows$logical_sequence_id, seq_len(2213L)) &&
    all(direct_artifact$rows$level == 0L) &&
    all(direct_artifact$rows$S_key == "") &&
    all(is.na(direct_artifact$rows$residual_key_x)) &&
    all(is.na(direct_artifact$rows$residual_key_y)),
  "direct-CI canonical row coverage"
)
assert_true(
  all(direct_artifact$rows$backend == "legacy-cpp") &&
    all(direct_artifact$rows$low_rank_backend == "spectra") &&
    !any(direct_artifact$rows$backend_error) &&
    !any(direct_artifact$rows$spectra_fallback) &&
    !any(direct_artifact$rows$decision_flip),
  "direct-CI route and correctness gates"
)
assert_true(
  identical(
    direct_artifact$rows$candidate_decision,
    ifelse(
      direct_artifact$rows$candidate_p_value > direct_artifact$rows$alpha,
      "independent", "dependent"
    )
  ),
  "direct-CI candidate decision uses strict p-value greater than alpha"
)
assert_true(
  identical(
    sort(list.files(direct_output_dir), method = "radix"),
    c("direct_ci.rds", "direct_ci.summary.json")
  ),
  "direct-CI publication is one atomic payload pair"
)
assert_true(
  isTRUE(validated_direct$authenticated) &&
    identical(validated_direct$summary$row_count, 2213L) &&
    identical(validated_direct$summary$backend_error_count, 0L) &&
    identical(validated_direct$summary$spectra_fallback_count, 0L) &&
    identical(validated_direct$summary$decision_flip_count, 0L) &&
    isTRUE(validated_direct$summary$pass),
  "direct-CI summary correctness gates"
)
catalog_lineage <- fastkpc_full_cuda_phase3_discover_catalog_evidence(
  catalog
)
assert_true(
  identical(
    validated_direct$payload$lineage$phase0_manifest_hash,
    catalog_lineage$phase0_manifest_hash
  ) && identical(
    validated_direct$payload$lineage$phase1_manifest_hash,
    catalog_lineage$phase1_manifest_hash
  ) && identical(
    validated_direct$payload$lineage$dataset_matrix_sha256,
    catalog_lineage$dataset_matrix_sha256
  ) && identical(
    validated_direct$payload$lineage$route_config_hash,
    fastkpc_full_cuda_phase3_route_config()$sha256
  ) && identical(
    validated_direct$payload$lineage$execution_device, "cpu"
  ) && identical(
    validated_direct$payload$lineage$residual_backend, "none"
  ),
  "direct-CI artifact authenticates Phase 0/1/data/route lineage"
)

assert_true(
  identical(
    fastkpc_full_cuda_data_hash(catalog$inputs$data),
    catalog_lineage$dataset_matrix_sha256
  ),
  "canonical in-memory data must match authenticated matrix hash"
)
mutated_data_catalog <- catalog
mutated_data_catalog$inputs$data[[1L]] <-
  mutated_data_catalog$inputs$data[[1L]] + 1e-8
assert_true(
  !identical(
    fastkpc_full_cuda_data_hash(mutated_data_catalog$inputs$data),
    catalog_lineage$dataset_matrix_sha256
  ),
  "data-mutation probe must change the actual matrix hash"
)
mutated_data_output <- tempfile("full-cuda-ci-direct-mutated-data-")
assert_error(
  fastkpc_full_cuda_shadow_write_direct_ci(
    mutated_data_catalog, mutated_data_output
  ),
  "direct-CI canonical data matrix hash mismatch",
  "direct-CI writer must reject mutated in-memory canonical data"
)
mutated_data_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  mutated_data_output, kind = "full_shadow"
)
assert_true(
  !file.exists(mutated_data_paths$direct_ci_rds) &&
    !file.exists(mutated_data_paths$direct_ci_summary_json),
  "mutated input data must fail before direct-CI publication"
)
assert_error(
  fastkpc_full_cuda_phase3_validate_direct_ci_payload(
    direct_output_dir, mutated_data_catalog
  ),
  "direct-CI canonical data matrix hash mismatch",
  "direct-CI validator must recompute the actual input matrix hash"
)

invalid_publish_rows <- direct_artifact$rows
invalid_x <- invalid_publish_rows$x[[1L]]
invalid_publish_rows$x[[1L]] <- invalid_publish_rows$y[[1L]]
invalid_publish_rows$y[[1L]] <- invalid_x
invalid_publish_dir <- tempfile("full-cuda-ci-direct-invalid-")
assert_error(
  fastkpc_full_cuda_phase3_publish_direct_ci_payload(
    invalid_publish_rows, catalog, invalid_publish_dir
  ),
  "Phase 1 direct row lineage mismatch",
  "direct-CI publisher must reject noncanonical logical rows"
)
invalid_publish_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  invalid_publish_dir, kind = "full_shadow"
)
assert_true(
  !file.exists(invalid_publish_paths$direct_ci_rds) &&
    !file.exists(invalid_publish_paths$direct_ci_summary_json),
  "failed direct-CI publication must leave no payload or completion marker"
)

direct_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  direct_output_dir, kind = "full_shadow"
)
summary_backup <- tempfile("direct-ci-summary-", fileext = ".json")
assert_true(
  file.copy(
    direct_paths$direct_ci_summary_json, summary_backup,
    overwrite = TRUE
  ),
  "direct-CI summary backup"
)
corrupt_hash_summary <- .fastkpc_full_cuda_phase3_read_json(
  direct_paths$direct_ci_summary_json, "direct_ci.summary.json"
)
corrupt_hash_summary$direct_ci_rds_sha256 <- strrep("0", 64L)
.fastkpc_full_cuda_phase3_write_json_exact(
  corrupt_hash_summary, direct_paths$direct_ci_summary_json
)
assert_error(
  fastkpc_full_cuda_phase3_validate_direct_ci_payload(
    direct_output_dir, catalog
  ),
  "direct_ci.rds SHA-256 mismatch",
  "corrupt direct payload hash must fail closed"
)
assert_true(
  file.copy(
    summary_backup, direct_paths$direct_ci_summary_json,
    overwrite = TRUE
  ),
  "direct-CI summary restore"
)

payload_backup <- tempfile("direct-ci-payload-", fileext = ".rds")
assert_true(
  file.copy(direct_paths$direct_ci_rds, payload_backup, overwrite = TRUE),
  "direct-CI payload backup"
)
wrong_data_payload <- readRDS(direct_paths$direct_ci_rds)
wrong_data_payload$lineage$dataset_matrix_sha256 <- strrep("d", 64L)
wrong_data_payload$lineage_sha256 <-
  fastkpc_full_cuda_census_named_metadata_hash(wrong_data_payload$lineage)
.fastkpc_full_cuda_phase3_atomic_write_merged_rds(
  wrong_data_payload, direct_paths$direct_ci_rds
)
wrong_data_summary <- .fastkpc_full_cuda_phase3_read_json(
  direct_paths$direct_ci_summary_json, "direct_ci.summary.json"
)
wrong_data_summary$direct_ci_rds_sha256 <-
  fastkpc_full_cuda_census_file_hash(direct_paths$direct_ci_rds)
wrong_data_summary$lineage <- wrong_data_payload$lineage
wrong_data_summary$lineage_sha256 <- wrong_data_payload$lineage_sha256
.fastkpc_full_cuda_phase3_write_json_exact(
  wrong_data_summary, direct_paths$direct_ci_summary_json
)
assert_error(
  fastkpc_full_cuda_phase3_validate_direct_ci_payload(
    direct_output_dir, catalog
  ),
  "dataset matrix hash mismatch",
  "wrong direct-CI data hash must fail closed"
)
assert_true(
  file.copy(payload_backup, direct_paths$direct_ci_rds, overwrite = TRUE),
  "direct-CI payload restore after data-hash attack"
)
assert_true(
  file.copy(
    summary_backup, direct_paths$direct_ci_summary_json,
    overwrite = TRUE
  ),
  "direct-CI summary restore after data-hash attack"
)

fallback_payload <- readRDS(direct_paths$direct_ci_rds)
fallback_payload$rows$spectra_fallback[[1L]] <- TRUE
fallback_payload$rows_sha256 <- fastkpc_full_cuda_census_frame_hash(
  fallback_payload$rows
)
.fastkpc_full_cuda_phase3_atomic_write_merged_rds(
  fallback_payload, direct_paths$direct_ci_rds
)
fallback_summary <- .fastkpc_full_cuda_phase3_read_json(
  direct_paths$direct_ci_summary_json, "direct_ci.summary.json"
)
fallback_summary$direct_ci_rds_sha256 <-
  fastkpc_full_cuda_census_file_hash(direct_paths$direct_ci_rds)
.fastkpc_full_cuda_phase3_write_json_exact(
  fallback_summary, direct_paths$direct_ci_summary_json
)
assert_error(
  fastkpc_full_cuda_phase3_validate_direct_ci_payload(
    direct_output_dir, catalog
  ),
  "Spectra fallback is not allowed",
  "direct-CI backend fallback row must fail closed"
)
assert_true(
  file.copy(payload_backup, direct_paths$direct_ci_rds, overwrite = TRUE),
  "direct-CI payload restore after fallback attack"
)
assert_true(
  file.copy(
    summary_backup, direct_paths$direct_ci_summary_json,
    overwrite = TRUE
  ),
  "direct-CI summary restore after fallback attack"
)

wrong_row_lineage <- readRDS(direct_paths$direct_ci_rds)
original_x <- wrong_row_lineage$rows$x[[1L]]
wrong_row_lineage$rows$x[[1L]] <- wrong_row_lineage$rows$y[[1L]]
wrong_row_lineage$rows$y[[1L]] <- original_x
wrong_row_lineage$rows_sha256 <- fastkpc_full_cuda_census_frame_hash(
  wrong_row_lineage$rows
)
.fastkpc_full_cuda_phase3_atomic_write_merged_rds(
  wrong_row_lineage, direct_paths$direct_ci_rds
)
wrong_row_summary <- .fastkpc_full_cuda_phase3_read_json(
  direct_paths$direct_ci_summary_json, "direct_ci.summary.json"
)
wrong_row_summary$direct_ci_rds_sha256 <-
  fastkpc_full_cuda_census_file_hash(direct_paths$direct_ci_rds)
wrong_row_summary$rows_sha256 <- wrong_row_lineage$rows_sha256
.fastkpc_full_cuda_phase3_write_json_exact(
  wrong_row_summary, direct_paths$direct_ci_summary_json
)
assert_error(
  fastkpc_full_cuda_phase3_validate_direct_ci_payload(
    direct_output_dir, catalog
  ),
  "Phase 1 direct row lineage mismatch",
  "self-consistently re-signed direct logical row drift must fail closed"
)
assert_true(
  file.copy(payload_backup, direct_paths$direct_ci_rds, overwrite = TRUE) &&
    file.copy(
      summary_backup, direct_paths$direct_ci_summary_json,
      overwrite = TRUE
    ),
  "direct-CI artifact restore after logical-lineage attack"
)
invisible(fastkpc_full_cuda_phase3_validate_direct_ci_payload(
  direct_output_dir, catalog
))

cat("full CUDA CI fixed-sp shadow mapping: PASS\n")
