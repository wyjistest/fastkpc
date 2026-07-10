source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/mgcv_compat_contract.R")
source("fastkpc/R/mgcv_extract_oracle.R")
source("fastkpc/R/native.R")

fastkpc_full_cuda_prepared_s_input_contract <- function() {
  list(
    schema_version = "full-cuda-ci-phase2-input-v1",
    phase1_artifact_schema_version =
      "full-cuda-ci-workload-census-artifact-v1",
    phase1_source_commit =
      "1560068ba8d635e806612554e11bbed92c0b8843c",
    phase1_artifact_source_commit =
      "1560068ba8d635e806612554e11bbed92c0b8843",
    metadata_schema_version = "full-cuda-ci-metadata-v4",
    R_version = "R version 4.4.1 (2024-06-14)",
    mgcv_version = "1.9.1",
    formula_semantics_version = "kpcalg_regrXonS_v1",
    mgcv_semantics_version =
      "legacy-mgcv-gam-default-selection-v1",
    shard_count = 64L,
    dataset_file_sha256 =
      "e03cbfafed3336f3da7725878e92af24556b5ebdc6227b94ea84469e62e94036",
    dataset_matrix_sha256 =
      "971f1e0784817c6febfc84154f19540be02a92f6e4acad784dc7f92b979f1df7",
    canonical_logical_census_hash =
      "c9b48074dd59a439fceb9d5e64806adda5620cc4abe32095371abc447ef98634",
    canonical_key_corpus_hash =
      "b843630969f116da63f7fad095c54de2ff471540159ff97ca56c3871d6b2e1fa",
    file_hashes = c(
      "manifest.json" =
        "b0990cfc932a5fcabc09ad25e352e7babb67fc8a127f11c7d2b88887c4940574",
      "summary.json" =
        "4f71d1bbbbdd2436e3576b728363120bb9b911897b9dee3ecf6f8a5d3379eb24",
      "logical_ci_tests.rds" =
        "17781421df868ae7822c022ac58ad6322292ad51053d5da686a3d6f79b40d7c8",
      "residual_requests.rds" =
        "d7a995f12f6bc118a39009b0b685cb5c28068d418c5a569d552cf26e8748ec8b",
      "same_s_setup_metadata.rds" =
        "8b35a463b17a64512d653da949f5ac74f7cc21223f346a304ac52fdfe8434a3f",
      "target_fit_metadata.rds" =
        "af09b5dc4c6a34d7ec126e1fe7f3f1f9c3d7fcb6316ada759a293abe76d8323c",
      "risk_cases.rds" =
        "1e0951e9856bea3c9a1b7ba83ec03b79a678e7aa60464d7f6808397ab8d9a7bc"
    ),
    named_metadata_hashes = c(
      "setup_observation_metadata" =
        "5282820451b2658c636132e579859c8c2c8e6497a926b8b6d9c393e0043e667a",
      "same_s_setup_metadata" =
        "07830db88c62aa7658d44373e86d897b254e453773b4c0070460dc20fce91113",
      "target_fit_metadata" =
        "361672b87cd056a689f578a5eb7660a55d056395ac270d3e28dbbe24738bab40",
      "target_risk_metadata" =
        "95eba27f5ea7904761ae4afbc203c58a84566fc3cc308d1b9c90acca69cc96f2",
      "risk_cases" =
        "4a2748ba469e039143c482fd4cf0367324886cc526552e9529117cab7c596d91"
    )
  )
}

fastkpc_full_cuda_prepared_s_require <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
  invisible(TRUE)
}

fastkpc_full_cuda_prepared_s_named_character <- function(value) {
  if (is.null(value)) return(character())
  result <- unlist(value, recursive = TRUE, use.names = TRUE)
  setNames(as.character(result), names(result))
}

fastkpc_full_cuda_prepared_s_named_identical <- function(actual, expected) {
  actual <- fastkpc_full_cuda_prepared_s_named_character(actual)
  expected <- fastkpc_full_cuda_prepared_s_named_character(expected)
  actual_names <- sort(names(actual), method = "radix")
  expected_names <- sort(names(expected), method = "radix")
  identical(actual_names, expected_names) &&
    identical(unname(actual[actual_names]), unname(expected[expected_names]))
}

fastkpc_full_cuda_prepared_s_validate_contract <- function(contract) {
  required <- c(
    "schema_version", "phase1_artifact_schema_version",
    "phase1_source_commit", "phase1_artifact_source_commit",
    "metadata_schema_version", "R_version",
    "mgcv_version", "formula_semantics_version",
    "mgcv_semantics_version", "shard_count", "dataset_file_sha256",
    "dataset_matrix_sha256", "canonical_logical_census_hash",
    "canonical_key_corpus_hash", "file_hashes", "named_metadata_hashes"
  )
  missing <- setdiff(required, names(contract))
  if (!is.list(contract) || length(missing) > 0L) {
    stop("Phase 2 input contract is incomplete", call. = FALSE)
  }
  expected_files <- c(
    "manifest.json", "summary.json", "logical_ci_tests.rds",
    "residual_requests.rds", "same_s_setup_metadata.rds",
    "target_fit_metadata.rds", "risk_cases.rds"
  )
  expected_metadata <- c(
    "setup_observation_metadata", "same_s_setup_metadata",
    "target_fit_metadata", "target_risk_metadata", "risk_cases"
  )
  hashes <- c(
    unname(contract$file_hashes),
    unname(contract$named_metadata_hashes),
    contract$dataset_file_sha256,
    contract$dataset_matrix_sha256,
    contract$canonical_logical_census_hash,
    contract$canonical_key_corpus_hash
  )
  clean <- identical(names(contract$file_hashes), expected_files) &&
    identical(names(contract$named_metadata_hashes), expected_metadata) &&
    all(grepl("^[0-9a-f]{64}$", hashes)) &&
    grepl("^[0-9a-f]{41}$", contract$phase1_source_commit) &&
    grepl("^[0-9a-f]{40}$", contract$phase1_artifact_source_commit) &&
    startsWith(
      contract$phase1_source_commit,
      contract$phase1_artifact_source_commit
    ) &&
    identical(as.integer(contract$shard_count), 64L)
  fastkpc_full_cuda_prepared_s_require(
    clean, "Phase 2 input contract identity is invalid"
  )
}

fastkpc_full_cuda_prepared_s_authenticate_files <- function(
    census_dir, data_path, contract) {
  fastkpc_full_cuda_prepared_s_validate_contract(contract)
  phase1_paths <- setNames(
    file.path(census_dir, names(contract$file_hashes)),
    names(contract$file_hashes)
  )
  missing <- names(phase1_paths)[!file.exists(phase1_paths)]
  if (length(missing) > 0L) {
    stop(
      "Phase 1 input is missing: ", paste(missing, collapse = ","),
      call. = FALSE
    )
  }
  if (!file.exists(data_path)) {
    stop("canonical dataset is missing", call. = FALSE)
  }
  fallback_path <- file.path(census_dir, "fallbacks.csv")
  if (!file.exists(fallback_path)) {
    stop("Phase 1 fallback evidence is missing", call. = FALSE)
  }

  phase1_actual <- vapply(
    phase1_paths, fastkpc_full_cuda_census_file_hash, character(1L)
  )
  phase1_expected <- unname(contract$file_hashes[names(phase1_paths)])
  phase1_match <- phase1_actual == phase1_expected
  if (!all(phase1_match)) {
    stop(
      "Phase 1 input file hash mismatch: ",
      paste(names(phase1_paths)[!phase1_match], collapse = ","),
      call. = FALSE
    )
  }

  dataset_actual <- fastkpc_full_cuda_census_file_hash(data_path)
  if (!identical(dataset_actual, contract$dataset_file_sha256)) {
    stop("canonical dataset file hash mismatch", call. = FALSE)
  }

  inherited <- fastkpc_full_cuda_census_input_contract()
  fallback_expected <- unname(
    inherited$file_hashes[["oracle/fallbacks.csv"]]
  )
  fallback_actual <- fastkpc_full_cuda_census_file_hash(fallback_path)
  if (!identical(fallback_actual, fallback_expected)) {
    stop("Phase 1 fallback evidence hash mismatch", call. = FALSE)
  }

  logical_path <- c(
    paste0("phase1/", names(phase1_paths)),
    "dataset/cancer_RD-causalDiscoveryInput.rds",
    "phase1/fallbacks.csv"
  )
  paths <- c(unname(phase1_paths), data_path, fallback_path)
  expected <- c(phase1_expected, contract$dataset_file_sha256,
                fallback_expected)
  actual <- c(unname(phase1_actual), dataset_actual, fallback_actual)
  order_id <- order(logical_path, method = "radix")
  data.frame(
    logical_path = logical_path[order_id],
    path = paths[order_id],
    expected_sha256 = expected[order_id],
    actual_sha256 = actual[order_id],
    identical = actual[order_id] == expected[order_id],
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_prepared_s_input_bundle_hash <- function(input_hashes) {
  input_hashes <- as.data.frame(input_hashes, stringsAsFactors = FALSE)
  required <- c("logical_path", "actual_sha256")
  if (length(setdiff(required, names(input_hashes))) > 0L ||
      nrow(input_hashes) == 0L || anyNA(input_hashes[required]) ||
      anyDuplicated(input_hashes$logical_path) ||
      !all(grepl("^[0-9a-f]{64}$", input_hashes$actual_sha256))) {
    stop("Phase 1 input hash bundle is incomplete", call. = FALSE)
  }
  order_id <- order(input_hashes$logical_path, method = "radix")
  rows <- input_hashes[order_id, , drop = FALSE]
  payload <- paste0(
    paste0(rows$logical_path, "\t", rows$actual_sha256, collapse = "\n"),
    "\n"
  )
  fastkpc_full_cuda_census_hash_utf8(payload)
}

fastkpc_full_cuda_prepared_s_validate_manifest <- function(
    manifest, contract) {
  inherited <- fastkpc_full_cuda_census_input_contract()
  risk_config <- fastkpc_full_cuda_census_risk_config()
  risk_hash <- fastkpc_full_cuda_census_metadata_hash(risk_config)
  declared_metadata <- manifest$authenticated_metadata_hashes
  expected_hash_schemas <- fastkpc_full_cuda_census_hash_schema_versions()
  clean <- is.list(manifest) &&
    identical(as.character(manifest$schema_version),
              contract$phase1_artifact_schema_version) &&
    identical(as.character(manifest$mode), "metadata") &&
    identical(as.character(manifest$run_scope), "full") &&
    isTRUE(manifest$phase1_complete) &&
    identical(as.character(manifest$source_commit),
              contract$phase1_artifact_source_commit) &&
    identical(as.character(manifest$metadata_schema_version),
              contract$metadata_schema_version) &&
    identical(as.character(manifest$R_version), contract$R_version) &&
    identical(as.character(manifest$mgcv_version), contract$mgcv_version) &&
    identical(as.character(manifest$formula_semantics_version),
              contract$formula_semantics_version) &&
    identical(as.character(manifest$mgcv_semantics_version),
              contract$mgcv_semantics_version) &&
    identical(as.character(manifest$dataset_file_sha256),
              contract$dataset_file_sha256) &&
    identical(as.character(manifest$dataset_matrix_sha256),
              contract$dataset_matrix_sha256) &&
    identical(as.character(manifest$canonical_logical_census_hash),
              contract$canonical_logical_census_hash) &&
    identical(as.character(manifest$canonical_key_corpus_hash),
              contract$canonical_key_corpus_hash) &&
    identical(as.character(manifest$selected_key_corpus_hash),
              contract$canonical_key_corpus_hash) &&
    identical(as.character(manifest$phase0_source_commit),
              inherited$phase0_source_commit) &&
    identical(as.character(manifest$oracle_input_bundle_sha256),
              inherited$oracle_input_bundle_sha256) &&
    identical(as.character(manifest$risk_threshold_config_hash),
              risk_hash) &&
    identical(as.integer(manifest$shard_count),
              as.integer(contract$shard_count)) &&
    identical(as.integer(manifest$selected_key_count), 110617L) &&
    identical(as.integer(manifest$canonical_key_count), 110617L) &&
    isTRUE(manifest$oracle_inherited_graph_gate) &&
    identical(as.character(manifest$new_candidate_graph_gate),
              "NOT_APPLICABLE") &&
    isTRUE(manifest$parity_pass) &&
    fastkpc_full_cuda_prepared_s_named_identical(
      manifest$hash_schema_versions, expected_hash_schemas
    ) &&
    fastkpc_full_cuda_prepared_s_named_identical(
      declared_metadata, contract$named_metadata_hashes
    )
  fastkpc_full_cuda_prepared_s_require(
    clean, "Phase 1 manifest identity mismatch"
  )
  fastkpc_full_cuda_require_namespace("mgcv")
  runtime_clean <- identical(R.version.string, contract$R_version) &&
    identical(as.character(utils::packageVersion("mgcv")),
              contract$mgcv_version)
  fastkpc_full_cuda_prepared_s_require(
    runtime_clean, "Phase 2 R/mgcv runtime identity mismatch"
  )
  risk_config
}

fastkpc_full_cuda_prepared_s_summary_zero <- function(summary, field) {
  value <- suppressWarnings(as.numeric(summary[[field]]))
  length(value) == 1L && !is.na(value) && value == 0
}

fastkpc_full_cuda_prepared_s_validate_summary <- function(summary) {
  zero_fields <- c(
    "mgcv_fit_error_count", "fit_error_count",
    "same_s_invariant_violation_count",
    "target_near_constant_mismatch_count", "risk_semantic_mismatch_count",
    "unclassified_warning_count", "misclassified_warning_count",
    "unclassified_nonfinite_count", "misclassified_nonfinite_count"
  )
  true_fields <- c(
    "phase1_complete", "required_field_coverage_complete",
    "legacy_layout_parity_pass", "exact_target_near_constant_semantics",
    "exact_selected_key_set", "exact_target_request_lineage",
    "exact_selected_same_S_group_count", "exact_target_risk_key_set",
    "exact_target_risk_lineage", "exact_risk_semantics",
    "exact_setup_observation_key_set", "exact_target_setup_lineage",
    "exact_authenticated_metadata", "exact_warning_classification",
    "exact_nonfinite_classification", "parity_pass",
    "oracle_inherited_graph_gate", "pass"
  )
  counts_clean <-
    identical(as.integer(summary$logical_test_count), 240489L) &&
    identical(as.integer(summary$conditional_logical_test_count), 238276L) &&
    identical(as.integer(summary$conditional_residual_request_count),
              476552L) &&
    identical(
      as.integer(summary$canonical_global_unique_conditional_target_s_count),
      110617L
    ) &&
    identical(as.integer(summary$unique_conditional_S_count), 8634L) &&
    identical(as.integer(summary$same_s_setup_metadata_rows), 8634L) &&
    identical(as.integer(summary$target_fit_metadata_rows), 110617L) &&
    identical(as.integer(summary$setup_observation_metadata_rows), 110617L) &&
    identical(as.integer(summary$target_risk_metadata_rows), 110617L) &&
    identical(as.integer(summary$selected_key_count), 110617L) &&
    identical(as.integer(summary$canonical_key_count), 110617L) &&
    identical(as.integer(summary$same_S_setup_count), 8634L) &&
    identical(as.integer(summary$shard_count), 64L)
  clean <- is.list(summary) &&
    identical(as.character(summary$mode), "metadata") &&
    identical(as.character(summary$run_scope), "full") &&
    identical(as.character(summary$new_candidate_graph_gate),
              "NOT_APPLICABLE") &&
    identical(as.character(summary$canonical_logical_census_hash),
              fastkpc_full_cuda_prepared_s_input_contract()$
                canonical_logical_census_hash) &&
    identical(as.character(summary$canonical_key_corpus_hash),
              fastkpc_full_cuda_prepared_s_input_contract()$
                canonical_key_corpus_hash) &&
    isTRUE(all(vapply(true_fields, function(field) {
      isTRUE(summary[[field]])
    }, logical(1L)))) &&
    isTRUE(all(vapply(zero_fields, function(field) {
      fastkpc_full_cuda_prepared_s_summary_zero(summary, field)
    }, logical(1L)))) &&
    isTRUE(all.equal(as.numeric(summary$required_field_coverage), 1)) &&
    counts_clean
  fastkpc_full_cuda_prepared_s_require(
    clean, "Phase 1 summary evidence mismatch"
  )
  invisible(TRUE)
}

fastkpc_full_cuda_prepared_s_validate_fallbacks <- function(fallbacks) {
  fallbacks <- as.data.frame(fallbacks, stringsAsFactors = FALSE)
  required <- c("key", "count")
  required_keys <- c("unknown_fallback_count", "approximate_backend_count")
  missing <- setdiff(required, names(fallbacks))
  counts <- suppressWarnings(as.numeric(fallbacks$count))
  clean <- length(missing) == 0L && nrow(fallbacks) == 2L &&
    identical(sort(as.character(fallbacks$key), method = "radix"),
              sort(required_keys, method = "radix")) &&
    length(counts) == 2L && all(is.finite(counts)) && all(counts == 0)
  fastkpc_full_cuda_prepared_s_require(
    clean, "Phase 1 fallback/approximation evidence is invalid"
  )
  invisible(TRUE)
}

fastkpc_full_cuda_prepared_s_validate_data <- function(data, contract) {
  canonical <- fastkpc_full_cuda_canonical_contract()
  clean <- identical(dim(data), c(canonical$n, canonical$p)) &&
    identical(colnames(data), canonical$column_order) &&
    all(is.finite(data)) &&
    identical(fastkpc_full_cuda_data_hash(data),
              contract$dataset_matrix_sha256)
  fastkpc_full_cuda_prepared_s_require(
    clean, "canonical dataset matrix identity mismatch"
  )
  invisible(TRUE)
}

fastkpc_full_cuda_prepared_s_validate_tables <- function(
    logical_tests, residual_requests, same_s_setup_metadata,
    target_fit_metadata, risk_cases, contract) {
  tables <- list(
    logical_tests = logical_tests,
    residual_requests = residual_requests,
    same_s_setup_metadata = same_s_setup_metadata,
    target_fit_metadata = target_fit_metadata,
    risk_cases = risk_cases
  )
  if (!all(vapply(tables, is.data.frame, logical(1L)))) {
    stop("Phase 1 canonical table type mismatch", call. = FALSE)
  }
  counts_clean <- nrow(logical_tests) == 240489L &&
    nrow(residual_requests) == 110617L &&
    nrow(same_s_setup_metadata) == 8634L &&
    nrow(target_fit_metadata) == 110617L
  fastkpc_full_cuda_prepared_s_require(
    counts_clean, "Phase 1 canonical table count mismatch"
  )

  logical_hash <- fastkpc_full_cuda_census_logical_hash(logical_tests)
  logical_clean <-
    identical(logical_hash, contract$canonical_logical_census_hash) &&
    identical(attr(logical_tests, "dataset_sha256", exact = TRUE),
              contract$dataset_matrix_sha256) &&
    identical(as.integer(attr(logical_tests, "data_n", exact = TRUE)),
              351L) &&
    identical(as.integer(attr(logical_tests, "data_p", exact = TRUE)),
              48L) &&
    identical(as.integer(logical_tests$logical_sequence_id),
              seq_len(nrow(logical_tests))) &&
    sum(logical_tests$S_size > 0L) == 238276L
  fastkpc_full_cuda_prepared_s_require(
    logical_clean, "Phase 1 logical census identity mismatch"
  )

  request_fields <- c(
    "residual_key_payload", "residual_key_sha256", "target", "S_key",
    "S_size", "formula_class", "same_S_group_id", "same_S_group_size",
    "request_multiplicity"
  )
  if (length(setdiff(request_fields, names(residual_requests))) > 0L) {
    stop("Phase 1 residual request table is incomplete", call. = FALSE)
  }
  request_keys <- as.character(residual_requests$residual_key_sha256)
  key_hash <- fastkpc_full_cuda_census_key_set_hash(
    sort(request_keys, method = "radix")
  )
  request_clean <- !anyNA(request_keys) && !anyDuplicated(request_keys) &&
    identical(request_keys, sort(request_keys, method = "radix")) &&
    all(grepl("^[0-9a-f]{64}$", request_keys)) &&
    all(endsWith(residual_requests$residual_key_payload, "\n")) &&
    identical(key_hash, contract$canonical_key_corpus_hash) &&
    identical(
      attr(residual_requests, "canonical_key_corpus_hash", exact = TRUE),
      contract$canonical_key_corpus_hash
    ) &&
    identical(
      as.integer(attr(
        residual_requests, "conditional_residual_request_count", exact = TRUE
      )),
      476552L
    ) &&
    identical(sum(as.integer(residual_requests$request_multiplicity)),
              476552L)
  fastkpc_full_cuda_prepared_s_require(
    request_clean, "Phase 1 residual request identity mismatch"
  )
  fastkpc_full_cuda_census_validate_key_mapping(
    residual_requests$residual_key_payload,
    residual_requests$residual_key_sha256
  )

  conditional <- logical_tests$S_size > 0L
  logical_request_keys <- c(
    as.character(logical_tests$residual_key_x[conditional]),
    as.character(logical_tests$residual_key_y[conditional])
  )
  logical_counts <- table(logical_request_keys)
  logical_index <- match(names(logical_counts), request_keys)
  multiplicity_clean <- length(logical_request_keys) == 476552L &&
    length(logical_counts) == 110617L && !anyNA(logical_index) &&
    identical(
      as.integer(logical_counts),
      as.integer(residual_requests$request_multiplicity[logical_index])
    )
  fastkpc_full_cuda_prepared_s_require(
    multiplicity_clean, "Phase 1 logical-to-residual lineage mismatch"
  )

  setup_fields <- c(
    "same_S_group_id", "S_key", "S_size", "formula_class",
    "formula_semantics_version", "penalty_count", "setup_fingerprint",
    "mgcv_version", "R_version"
  )
  if (length(setdiff(setup_fields, names(same_s_setup_metadata))) > 0L) {
    stop("Phase 1 same-S setup table is incomplete", call. = FALSE)
  }
  group_ids <- as.character(same_s_setup_metadata$same_S_group_id)
  S <- lapply(
    as.character(same_s_setup_metadata$S_key),
    fastkpc_full_cuda_census_parse_s
  )
  expected_formula <- vapply(
    S, fastkpc_regrxons_formula_class, character(1L)
  )
  same_s_payload <- mapply(
    fastkpc_full_cuda_census_same_s_payload,
    S = S,
    formula_class = as.character(same_s_setup_metadata$formula_class),
    MoreArgs = list(
      data_hash = contract$dataset_matrix_sha256, n = 351L, p = 48L
    ),
    USE.NAMES = FALSE
  )
  expected_group_ids <- unname(vapply(
    same_s_payload, fastkpc_full_cuda_census_hash_utf8, character(1L)
  ))
  setup_clean <- !anyNA(group_ids) && !anyDuplicated(group_ids) &&
    identical(group_ids, sort(group_ids, method = "radix")) &&
    identical(as.integer(same_s_setup_metadata$S_size),
              as.integer(lengths(S))) &&
    identical(as.character(same_s_setup_metadata$formula_class),
              expected_formula) &&
    identical(group_ids, expected_group_ids) &&
    all(same_s_setup_metadata$formula_semantics_version ==
          contract$formula_semantics_version) &&
    all(same_s_setup_metadata$R_version == contract$R_version) &&
    all(same_s_setup_metadata$mgcv_version == contract$mgcv_version)
  fastkpc_full_cuda_prepared_s_require(
    setup_clean, "Phase 1 same-S setup identity mismatch"
  )

  setup_identity <- paste(
    same_s_setup_metadata$S_key,
    same_s_setup_metadata$formula_class,
    sep = "\t"
  )
  request_identity <- paste(
    residual_requests$S_key, residual_requests$formula_class, sep = "\t"
  )
  setup_index <- match(request_identity, setup_identity)
  group_sizes <- table(residual_requests$same_S_group_id)
  request_lineage_clean <- !anyNA(setup_index) &&
    identical(
      as.character(residual_requests$same_S_group_id),
      group_ids[setup_index]
    ) &&
    identical(
      as.integer(residual_requests$same_S_group_size),
      as.integer(group_sizes[residual_requests$same_S_group_id])
    )
  fastkpc_full_cuda_prepared_s_require(
    request_lineage_clean, "Phase 1 residual-to-setup lineage mismatch"
  )

  target_fields <- c(
    "residual_key_sha256", "same_S_group_id", "shard_id", "target",
    "fit_status", "fit_error", "selected_sp", "selected_sp_names"
  )
  if (length(setdiff(target_fields, names(target_fit_metadata))) > 0L) {
    stop("Phase 1 target fit table is incomplete", call. = FALSE)
  }
  target_keys <- as.character(target_fit_metadata$residual_key_sha256)
  target_index <- match(target_keys, request_keys)
  assigned <- fastkpc_full_cuda_census_assign_shards(
    residual_requests, contract$shard_count
  )
  assigned_index <- match(target_keys, assigned$residual_key_sha256)
  target_clean <- !anyNA(target_keys) && !anyDuplicated(target_keys) &&
    identical(target_keys, request_keys) && !anyNA(target_index) &&
    !anyNA(assigned_index) &&
    identical(
      as.character(target_fit_metadata$same_S_group_id),
      as.character(residual_requests$same_S_group_id[target_index])
    ) &&
    identical(
      as.integer(target_fit_metadata$target),
      as.integer(residual_requests$target[target_index])
    ) &&
    identical(
      as.integer(target_fit_metadata$shard_id),
      as.integer(assigned$shard_id[assigned_index])
    ) &&
    all(target_fit_metadata$fit_status == "success") &&
    all(target_fit_metadata$fit_error == "NONE")
  fastkpc_full_cuda_prepared_s_require(
    target_clean, "Phase 1 target fit lineage mismatch"
  )

  target_setup_index <- match(
    target_fit_metadata$same_S_group_id,
    same_s_setup_metadata$same_S_group_id
  )
  expected_sp_count <- as.integer(
    same_s_setup_metadata$penalty_count[target_setup_index]
  )
  sp_count <- lengths(target_fit_metadata$selected_sp)
  sp_name_count <- lengths(target_fit_metadata$selected_sp_names)
  sp_finite <- vapply(target_fit_metadata$selected_sp, function(value) {
    all(is.finite(as.numeric(value)))
  }, logical(1L))
  sp_clean <- !anyNA(target_setup_index) &&
    identical(as.integer(sp_count), expected_sp_count) &&
    identical(as.integer(sp_name_count), expected_sp_count) &&
    all(sp_finite)
  if (!isTRUE(sp_clean)) {
    mismatch <- which(
      is.na(target_setup_index) | sp_count != expected_sp_count |
        sp_name_count != expected_sp_count | !sp_finite
    )[[1L]]
    stop(
      "Phase 1 selected-sp length mismatch: ", target_keys[[mismatch]],
      call. = FALSE
    )
  }

  invisible(list(logical_hash = logical_hash, key_hash = key_hash))
}

fastkpc_full_cuda_prepared_s_frame_hashes <- function(tables) {
  vapply(
    tables, fastkpc_full_cuda_census_frame_hash, character(1L)
  )
}

fastkpc_full_cuda_prepared_s_require_frame_hashes <- function(
    actual, expected, label) {
  expected <- expected[names(actual)]
  matches <- !is.na(expected) & actual == unname(expected)
  if (!all(matches)) {
    stop(
      label, ": ", paste(names(actual)[!matches], collapse = ","),
      call. = FALSE
    )
  }
  invisible(TRUE)
}

fastkpc_full_cuda_prepared_s_shard_context <- function(
    manifest, logical_tests, risk_config) {
  fields <- fastkpc_full_cuda_census_manifest_context_fields()
  manifest_context <- manifest
  manifest_context$dataset_sha256 <- manifest$dataset_matrix_sha256
  missing <- setdiff(fields, names(manifest_context))
  if (length(missing) > 0L) {
    stop(
      "Phase 1 manifest is missing shard context: ",
      paste(missing, collapse = ","),
      call. = FALSE
    )
  }
  context <- setNames(lapply(fields, function(field) {
    manifest_context[[field]]
  }), fields)
  context$BLAS_thread_count <- as.integer(context$BLAS_thread_count)
  context$risk_config <- risk_config
  context$logical_tests <- logical_tests
  context
}

fastkpc_full_cuda_prepared_s_normalize_atomic <- function(value) {
  if (is.factor(value)) value <- as.character(value)
  attributes(value) <- NULL
  if (is.integer(value) || is.double(value)) {
    value <- as.numeric(value)
  } else if (is.character(value)) {
    value <- enc2utf8(as.character(value))
  } else if (is.logical(value)) {
    value <- as.logical(value)
  } else if (is.complex(value)) {
    value <- as.complex(value)
  }
  unname(value)
}

fastkpc_full_cuda_prepared_s_validate_setup_invariance <- function(
    observations) {
  observations <- as.data.frame(observations, stringsAsFactors = FALSE)
  lineage_field <- "representative_residual_key_sha256"
  if (!all(c("same_S_group_id", lineage_field) %in% names(observations)) ||
      nrow(observations) == 0L) {
    stop("Phase 1 setup observations are incomplete", call. = FALSE)
  }
  fields <- setdiff(names(observations), lineage_field)
  list_fields <- fields[vapply(
    observations[fields], is.list, logical(1L)
  )]
  atomic_fields <- setdiff(fields, list_fields)
  groups <- split(seq_len(nrow(observations)), observations$same_S_group_id)

  for (group_id in names(groups)) {
    index <- groups[[group_id]]
    for (field in atomic_fields) {
      values <- fastkpc_full_cuda_prepared_s_normalize_atomic(
        observations[[field]][index]
      )
      expected <- rep(values[[1L]], length(values))
      if (!identical(values, expected)) {
        stop(
          "Phase 1 same-S setup invariant mismatch: group ", group_id,
          ", field ", field,
          call. = FALSE
        )
      }
    }

    if (length(list_fields) > 0L) {
      actual_lists <- setNames(lapply(list_fields, function(field) {
        lapply(index, function(row) observations[[field]][[row]])
      }), list_fields)
      expected_lists <- setNames(lapply(list_fields, function(field) {
        rep(list(observations[[field]][[index[[1L]]]]), length(index))
      }), list_fields)
      actual_hash <- fastkpc_full_cuda_census_named_metadata_hash(
        actual_lists
      )
      expected_hash <- fastkpc_full_cuda_census_named_metadata_hash(
        expected_lists
      )
      if (!identical(actual_hash, expected_hash)) {
        mismatch <- vapply(list_fields, function(field) {
          !identical(
            fastkpc_full_cuda_census_named_metadata_hash(
              actual_lists[[field]]
            ),
            fastkpc_full_cuda_census_named_metadata_hash(
              expected_lists[[field]]
            )
          )
        }, logical(1L))
        field <- list_fields[which(mismatch)[[1L]]]
        stop(
          "Phase 1 same-S setup invariant mismatch: group ", group_id,
          ", field ", field,
          call. = FALSE
        )
      }
    }
  }

  list(
    validated = TRUE,
    group_count = as.integer(length(groups)),
    field_count = as.integer(length(fields)),
    fields = fields,
    atomic_fields = atomic_fields,
    list_fields = list_fields
  )
}

fastkpc_full_cuda_prepared_s_load_inputs <- function(
    census_dir, data_path,
    contract = fastkpc_full_cuda_prepared_s_input_contract()) {
  input_hashes <- fastkpc_full_cuda_prepared_s_authenticate_files(
    census_dir, data_path, contract
  )
  phase1_input_bundle_hash <-
    fastkpc_full_cuda_prepared_s_input_bundle_hash(input_hashes)

  fastkpc_full_cuda_require_namespace("jsonlite")
  manifest <- jsonlite::read_json(
    file.path(census_dir, "manifest.json"), simplifyVector = TRUE
  )
  summary <- jsonlite::read_json(
    file.path(census_dir, "summary.json"), simplifyVector = TRUE
  )
  logical_tests <- readRDS(file.path(census_dir, "logical_ci_tests.rds"))
  residual_requests <- readRDS(
    file.path(census_dir, "residual_requests.rds")
  )
  same_s_setup_metadata <- readRDS(
    file.path(census_dir, "same_s_setup_metadata.rds")
  )
  target_fit_metadata <- readRDS(
    file.path(census_dir, "target_fit_metadata.rds")
  )
  risk_cases <- readRDS(file.path(census_dir, "risk_cases.rds"))
  fallbacks <- utils::read.csv(
    file.path(census_dir, "fallbacks.csv"), stringsAsFactors = FALSE
  )
  data <- as.matrix(readRDS(data_path))
  storage.mode(data) <- "double"

  risk_config <- fastkpc_full_cuda_prepared_s_validate_manifest(
    manifest, contract
  )
  fastkpc_full_cuda_prepared_s_validate_summary(summary)
  fastkpc_full_cuda_prepared_s_validate_fallbacks(fallbacks)
  fastkpc_full_cuda_prepared_s_validate_data(data, contract)
  fastkpc_full_cuda_prepared_s_validate_tables(
    logical_tests, residual_requests, same_s_setup_metadata,
    target_fit_metadata, risk_cases, contract
  )

  loaded_hashes <- fastkpc_full_cuda_prepared_s_frame_hashes(list(
    same_s_setup_metadata = same_s_setup_metadata,
    target_fit_metadata = target_fit_metadata,
    risk_cases = risk_cases
  ))
  fastkpc_full_cuda_prepared_s_require_frame_hashes(
    loaded_hashes, contract$named_metadata_hashes,
    "Phase 1 named metadata hash mismatch"
  )

  context <- fastkpc_full_cuda_prepared_s_shard_context(
    manifest, logical_tests, risk_config
  )
  merged <- fastkpc_full_cuda_census_merge_shards(
    requests = residual_requests,
    shard_count = contract$shard_count,
    context = context,
    shard_dir = file.path(census_dir, "shards")
  )
  merged_hashes <- fastkpc_full_cuda_prepared_s_named_character(
    merged$authenticated_metadata_hashes
  )
  fastkpc_full_cuda_prepared_s_require_frame_hashes(
    merged_hashes, contract$named_metadata_hashes,
    "Phase 1 merged metadata hash mismatch"
  )

  setup_observations <- merged$setup_observation_metadata
  target_risks <- merged$target_risk_metadata
  setup_invariance <-
    fastkpc_full_cuda_prepared_s_validate_setup_invariance(
      setup_observations
    )
  fastkpc_full_cuda_prepared_s_require(
    identical(setup_invariance$group_count, 8634L),
    "Phase 1 setup invariance group count mismatch"
  )

  list(
    logical_tests = logical_tests,
    residual_requests = residual_requests,
    same_s_setup_metadata = same_s_setup_metadata,
    target_fit_metadata = target_fit_metadata,
    risk_cases = risk_cases,
    setup_observations = setup_observations,
    target_risks = target_risks,
    data = data,
    manifest = manifest,
    summary = summary,
    fallbacks = fallbacks,
    dataset_file_sha256 = contract$dataset_file_sha256,
    dataset_sha256 = contract$dataset_matrix_sha256,
    input_hashes = input_hashes,
    phase1_input_bundle_hash = phase1_input_bundle_hash,
    setup_invariance = setup_invariance
  )
}

fastkpc_full_cuda_prepared_s_key <- function(
    setup_row, dataset_sha256, R_version, mgcv_version,
    hash_fun = fastkpc_full_cuda_census_hash_utf8) {
  setup_row <- as.data.frame(setup_row, stringsAsFactors = FALSE)
  if (nrow(setup_row) != 1L) {
    stop("PreparedSKey requires one same-S row", call. = FALSE)
  }
  required <- c("same_S_group_id", "S_key", "formula_class")
  missing <- setdiff(required, names(setup_row))
  if (length(missing) > 0L) {
    stop(
      "PreparedSKey setup row is missing: ", paste(missing, collapse = ","),
      call. = FALSE
    )
  }
  scalar <- function(value) {
    length(value) == 1L && !is.na(value) && nzchar(as.character(value))
  }
  values <- c(
    dataset_sha256, setup_row$same_S_group_id[[1L]],
    setup_row$S_key[[1L]], setup_row$formula_class[[1L]],
    R_version, mgcv_version
  )
  if (!all(vapply(as.list(values), scalar, logical(1L))) ||
      any(grepl("[\r\n]", as.character(values)))) {
    stop("PreparedSKey identity is incomplete", call. = FALSE)
  }
  if (!grepl("^[0-9a-f]{64}$", dataset_sha256) ||
      !grepl("^[0-9a-f]{64}$", setup_row$same_S_group_id[[1L]])) {
    stop("PreparedSKey hash identity is invalid", call. = FALSE)
  }
  S <- fastkpc_full_cuda_census_parse_s(setup_row$S_key[[1L]])
  formula_class <- fastkpc_regrxons_formula_class(S)
  if (!identical(as.character(setup_row$formula_class[[1L]]),
                 formula_class) ||
      ("S_size" %in% names(setup_row) &&
       !identical(as.integer(setup_row$S_size[[1L]]),
                  as.integer(length(S)))) ||
      ("formula_semantics_version" %in% names(setup_row) &&
       !identical(
         as.character(setup_row$formula_semantics_version[[1L]]),
         "kpcalg_regrXonS_v1"
       )) ||
      ("R_version" %in% names(setup_row) &&
       !identical(as.character(setup_row$R_version[[1L]]),
                  as.character(R_version))) ||
      ("mgcv_version" %in% names(setup_row) &&
       !identical(as.character(setup_row$mgcv_version[[1L]]),
                  as.character(mgcv_version)))) {
    stop("PreparedSKey same-S row identity mismatch", call. = FALSE)
  }
  fields <- c(
    "schema_version=full-cuda-ci-prepared-s-key-v1",
    paste0("dataset_sha256=", dataset_sha256),
    paste0("same_S_group_id=", setup_row$same_S_group_id[[1L]]),
    paste0("sorted_S=", paste(S, collapse = ",")),
    paste0("formula_class=", setup_row$formula_class[[1L]]),
    "formula_semantics_version=kpcalg_regrXonS_v1",
    "mgcv_semantics_version=mgcv-gam-gcv-cp-v1",
    "family=gaussian",
    "link=identity",
    "method=GCV.Cp",
    "optimizer=mgcv-default",
    "bs=tp",
    "k=mgcv-default",
    "select=false",
    "scale=mgcv-default",
    "na_action=na.fail",
    paste0("R_version=", R_version),
    paste0("mgcv_version=", mgcv_version)
  )
  payload <- enc2utf8(paste0(paste(fields, collapse = "\n"), "\n"))
  if (!is.function(hash_fun)) {
    stop("PreparedSKey hash function is invalid", call. = FALSE)
  }
  sha256 <- hash_fun(payload)
  if (length(sha256) != 1L || is.na(sha256) || !nzchar(sha256)) {
    stop("PreparedSKey hash function returned an invalid hash", call. = FALSE)
  }
  fastkpc_full_cuda_prepared_s_validate_key_mapping(payload, sha256)
  list(payload = payload, sha256 = as.character(sha256))
}

fastkpc_full_cuda_prepared_s_validate_key_mapping <- function(payload, hash) {
  if (length(payload) != length(hash) || anyNA(payload) || anyNA(hash) ||
      any(!nzchar(as.character(payload))) || any(!nzchar(as.character(hash)))) {
    stop("PreparedSKey mapping is incomplete", call. = FALSE)
  }
  mapping <- unique(data.frame(
    payload = as.character(payload),
    hash = as.character(hash),
    stringsAsFactors = FALSE
  ))
  if (anyDuplicated(mapping$hash)) {
    stop("PreparedSKey hash collision", call. = FALSE)
  }
  if (anyDuplicated(mapping$payload)) {
    stop("PreparedSKey serialization mismatch", call. = FALSE)
  }
  invisible(TRUE)
}
