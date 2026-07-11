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
      "1560068ba8d635e806612554e11bbed92c0b8843",
    metadata_schema_version = "full-cuda-ci-metadata-v4",
    R_version = "R version 4.4.1 (2024-06-14)",
    mgcv_version = "1.9.1",
    formula_semantics_version = "kpcalg_regrXonS_v1",
    mgcv_semantics_version =
      "legacy-mgcv-gam-default-selection-v1",
    shard_count = 64L,
    shard_file_bundle_sha256 =
      "fdb6ef491e74bd1a9ac8bc01682d83a1fb57839517eb0c0205536b24618323ee",
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
  canonical <- fastkpc_full_cuda_prepared_s_input_contract()
  if (!is.list(contract) || !identical(contract, canonical)) {
    stop("canonical Phase 2 input contract mismatch", call. = FALSE)
  }
  invisible(TRUE)
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

  shard_hashes <- fastkpc_full_cuda_prepared_s_authenticate_shards(
    shard_dir = file.path(census_dir, "shards"),
    shard_count = contract$shard_count,
    expected_bundle_sha256 = contract$shard_file_bundle_sha256
  )

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
  core_hashes <- data.frame(
    logical_path = logical_path[order_id],
    path = paths[order_id],
    expected_sha256 = expected[order_id],
    actual_sha256 = actual[order_id],
    identical = actual[order_id] == expected[order_id],
    stringsAsFactors = FALSE
  )
  hashes <- rbind(core_hashes, shard_hashes)
  hashes[order(hashes$logical_path, method = "radix"), , drop = FALSE]
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

fastkpc_full_cuda_prepared_s_authenticate_shards <- function(
    shard_dir, shard_count, expected_bundle_sha256) {
  shard_count <- fastkpc_full_cuda_census_validate_shard_count(shard_count)
  if (length(expected_bundle_sha256) != 1L ||
      is.na(expected_bundle_sha256) ||
      !grepl("^[0-9a-f]{64}$", expected_bundle_sha256)) {
    stop("Phase 1 shard byte bundle identity is invalid", call. = FALSE)
  }
  if (!dir.exists(shard_dir)) {
    stop("Phase 1 shard directory is missing", call. = FALSE)
  }

  shard_ids <- 0:(shard_count - 1L)
  expected_names <- c(
    paste0("shard_", shard_ids, ".rds"),
    paste0("shard_", shard_ids, ".summary.json")
  )
  actual_names <- list.files(
    shard_dir, all.files = TRUE, no.. = TRUE
  )
  if (!identical(
        sort(actual_names, method = "radix"),
        sort(expected_names, method = "radix")
      )) {
    stop("Phase 1 shard file set mismatch", call. = FALSE)
  }

  logical_path <- paste0("phase1/shards/", expected_names)
  paths <- file.path(shard_dir, expected_names)
  actual_sha256 <- unname(vapply(
    paths, fastkpc_full_cuda_census_file_hash, character(1L)
  ))
  order_id <- order(logical_path, method = "radix")
  hashes <- data.frame(
    logical_path = logical_path[order_id],
    path = paths[order_id],
    expected_sha256 = actual_sha256[order_id],
    actual_sha256 = actual_sha256[order_id],
    identical = TRUE,
    stringsAsFactors = FALSE
  )
  actual_bundle_sha256 <-
    fastkpc_full_cuda_prepared_s_input_bundle_hash(hashes)
  if (!identical(actual_bundle_sha256, expected_bundle_sha256)) {
    stop("Phase 1 shard byte bundle hash mismatch", call. = FALSE)
  }
  hashes
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
              contract$phase1_source_commit) &&
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

fastkpc_full_cuda_prepared_s_merge_shards <- function(
    requests, same_s_setup_metadata, target_fit_metadata, risk_cases,
    shard_count, context, shard_dir) {
  requests <- as.data.frame(requests, stringsAsFactors = FALSE)
  same_s_setup_metadata <- as.data.frame(
    same_s_setup_metadata, stringsAsFactors = FALSE
  )
  target_fit_metadata <- as.data.frame(
    target_fit_metadata, stringsAsFactors = FALSE
  )
  risk_cases <- as.data.frame(risk_cases, stringsAsFactors = FALSE)
  shard_count <- fastkpc_full_cuda_census_validate_shard_count(shard_count)
  assigned <- fastkpc_full_cuda_census_assign_shards(requests, shard_count)
  expected_ids <- 0:(shard_count - 1L)
  expected_keys <- as.character(assigned$residual_key_sha256)

  target_fields <- c(
    "residual_key_sha256", "same_S_group_id", "setup_fingerprint",
    "shard_id", "target", "fit_status"
  )
  target_keys <- as.character(target_fit_metadata$residual_key_sha256)
  if (length(setdiff(target_fields, names(target_fit_metadata))) > 0L ||
      anyNA(target_keys) || anyDuplicated(target_keys) ||
      !identical(target_keys, expected_keys)) {
    stop("merged shard target key set mismatch", call. = FALSE)
  }
  request_index <- match(target_keys, expected_keys)
  if (anyNA(request_index) ||
      !identical(
        as.character(target_fit_metadata$same_S_group_id),
        as.character(assigned$same_S_group_id[request_index])
      ) ||
      !identical(
        as.integer(target_fit_metadata$target),
        as.integer(assigned$target[request_index])
      ) ||
      !identical(
        as.integer(target_fit_metadata$shard_id),
        as.integer(assigned$shard_id[request_index])
      )) {
    stop("merged target/request lineage mismatch", call. = FALSE)
  }

  setup_chunks <- vector("list", shard_count)
  risk_chunks <- vector("list", shard_count)
  seen_targets <- logical(nrow(assigned))
  for (index in seq_along(expected_ids)) {
    shard_id <- expected_ids[[index]]
    paths <- fastkpc_full_cuda_census_shard_paths(shard_dir, shard_id)
    expected_manifest <- fastkpc_full_cuda_census_shard_manifest(
      assigned, shard_id, context
    )
    completed <- fastkpc_full_cuda_census_read_shard(
      paths, expected_manifest
    )
    payload <- completed$payload
    rm(completed)

    shard_targets <- as.data.frame(
      payload$target_fits, stringsAsFactors = FALSE
    )
    shard_keys <- as.character(shard_targets$residual_key_sha256)
    assigned_index <- match(shard_keys, expected_keys)
    if (anyNA(assigned_index) || any(seen_targets[assigned_index]) ||
        !identical(
          as.character(shard_targets$same_S_group_id),
          as.character(assigned$same_S_group_id[assigned_index])
        ) ||
        !identical(
          as.integer(shard_targets$target),
          as.integer(assigned$target[assigned_index])
        ) ||
        !identical(
          as.integer(shard_targets$shard_id),
          as.integer(assigned$shard_id[assigned_index])
        )) {
      stop("merged target/request lineage mismatch", call. = FALSE)
    }

    canonical_index <- match(shard_keys, target_keys)
    shard_order <- order(shard_keys, method = "radix")
    shard_targets <- shard_targets[shard_order, , drop = FALSE]
    canonical_targets <- target_fit_metadata[
      canonical_index[shard_order], , drop = FALSE
    ]
    rownames(shard_targets) <- NULL
    rownames(canonical_targets) <- NULL
    if (anyNA(canonical_index) ||
        !identical(
          fastkpc_full_cuda_census_frame_hash(shard_targets),
          fastkpc_full_cuda_census_frame_hash(canonical_targets)
        )) {
      stop("Phase 1 shard target metadata mismatch", call. = FALSE)
    }

    seen_targets[assigned_index] <- TRUE
    setup_chunks[[index]] <- as.data.frame(
      payload$setup_observations, stringsAsFactors = FALSE
    )
    risk_chunks[[index]] <- as.data.frame(
      payload$target_risks, stringsAsFactors = FALSE
    )
    rm(payload, shard_targets, canonical_targets)
    if (index %% 8L == 0L || index == length(expected_ids)) {
      invisible(gc(verbose = FALSE))
    }
  }
  if (!all(seen_targets)) {
    stop("merged shard target key set mismatch", call. = FALSE)
  }

  setup_observations <- fastkpc_full_cuda_census_bind_rows(setup_chunks)
  rm(setup_chunks)
  invisible(gc(verbose = FALSE))
  target_risks <- fastkpc_full_cuda_census_bind_rows(risk_chunks)
  rm(risk_chunks)
  invisible(gc(verbose = FALSE))

  successful <- as.character(target_fit_metadata$fit_status) == "success"
  successful_keys <- target_keys[successful]
  if (length(successful_keys) == 0L) {
    if (nrow(setup_observations) != 0L) {
      stop("merged setup key set mismatch", call. = FALSE)
    }
  } else {
    setup_fields <- c(
      "representative_residual_key_sha256", "same_S_group_id",
      "setup_fingerprint"
    )
    setup_keys <- as.character(
      setup_observations$representative_residual_key_sha256
    )
    if (length(setdiff(setup_fields, names(setup_observations))) > 0L ||
        anyNA(setup_keys) || anyDuplicated(setup_keys) ||
        !identical(
          sort(setup_keys, method = "radix"),
          sort(successful_keys, method = "radix")
        )) {
      stop("merged setup key set mismatch", call. = FALSE)
    }
    setup_order <- order(setup_keys, method = "radix")
    setup_observations <- setup_observations[
      setup_order, , drop = FALSE
    ]
    rownames(setup_observations) <- NULL
    setup_keys <- as.character(
      setup_observations$representative_residual_key_sha256
    )
    setup_index <- match(successful_keys, setup_keys)
    if (anyNA(setup_index) ||
        !identical(
          as.character(target_fit_metadata$same_S_group_id[successful]),
          as.character(setup_observations$same_S_group_id[setup_index])
        ) ||
        !identical(
          as.character(target_fit_metadata$setup_fingerprint[successful]),
          as.character(setup_observations$setup_fingerprint[setup_index])
        )) {
      stop("merged target/setup lineage mismatch", call. = FALSE)
    }
  }

  computed_same_s <- if (nrow(setup_observations) == 0L) {
    fastkpc_full_cuda_census_empty_frame(
      fastkpc_full_cuda_census_setup_metadata_fields()
    )
  } else {
    fastkpc_full_cuda_census_compress_setups(setup_observations)
  }
  if (!identical(
        fastkpc_full_cuda_census_frame_hash(computed_same_s),
        fastkpc_full_cuda_census_frame_hash(same_s_setup_metadata)
      )) {
    stop("Phase 1 merged same-S metadata mismatch", call. = FALSE)
  }

  risk_fields <- c(
    "case_type", "residual_key_sha256", "same_S_group_id"
  )
  risk_keys <- as.character(target_risks$residual_key_sha256)
  if (length(setdiff(risk_fields, names(target_risks))) > 0L ||
      anyNA(risk_keys) || anyDuplicated(risk_keys) ||
      !identical(sort(risk_keys, method = "radix"), expected_keys)) {
    stop("merged target risk key set mismatch", call. = FALSE)
  }
  risk_order <- order(risk_keys, method = "radix")
  target_risks <- target_risks[risk_order, , drop = FALSE]
  rownames(target_risks) <- NULL
  risk_keys <- as.character(target_risks$residual_key_sha256)
  risk_index <- match(target_keys, risk_keys)
  if (anyNA(risk_index) ||
      any(target_risks$case_type[risk_index] != "target_key") ||
      !identical(
        as.character(target_fit_metadata$same_S_group_id),
        as.character(target_risks$same_S_group_id[risk_index])
      )) {
    stop("merged target/risk lineage mismatch", call. = FALSE)
  }

  computed_risk_cases <- fastkpc_full_cuda_census_risk_cases(
    target_risks = target_risks,
    logical_tests = context$logical_tests
  )
  if (!identical(
        fastkpc_full_cuda_census_frame_hash(computed_risk_cases),
        fastkpc_full_cuda_census_frame_hash(risk_cases)
      )) {
    stop("Phase 1 merged risk metadata mismatch", call. = FALSE)
  }

  authenticated_metadata_hashes <-
    fastkpc_full_cuda_census_authenticated_metadata_hashes(list(
      setup_observation_metadata = setup_observations,
      same_s_setup_metadata = computed_same_s,
      target_fit_metadata = target_fit_metadata,
      target_risk_metadata = target_risks,
      risk_cases = computed_risk_cases
    ))
  list(
    setup_observation_metadata = setup_observations,
    target_risk_metadata = target_risks,
    authenticated_metadata_hashes = authenticated_metadata_hashes
  )
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
  merged <- fastkpc_full_cuda_prepared_s_merge_shards(
    requests = residual_requests,
    same_s_setup_metadata = same_s_setup_metadata,
    target_fit_metadata = target_fit_metadata,
    risk_cases = risk_cases,
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
  if (length(sha256) != 1L || is.na(sha256) ||
      !grepl("^[0-9a-f]{64}$", as.character(sha256))) {
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
  if (!all(grepl("^[0-9a-f]{64}$", as.character(hash)))) {
    stop("PreparedSKey hash is invalid", call. = FALSE)
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

fastkpc_full_cuda_prepared_s_setup_schema_version <- function() {
  "full-cuda-ci-prepared-s-setup-v1"
}

fastkpc_full_cuda_prepared_s_setup_field_names <- function() {
  c(
    "schema_version", "dataset_sha256", "prepared_s_key_payload",
    "prepared_s_key_sha256", "same_S_group_id",
    "phase1_setup_fingerprint", "phase1_model_matrix_hash", "sorted_S",
    "formula_class", "formula_semantics_version",
    "mgcv_semantics_version", "R_version", "mgcv_version", "family",
    "link", "method", "optimizer", "provider_fingerprint",
    "provider_X_hash", "X", "coefficient_labels", "intercept", "assign",
    "cmX", "penalty_blocks", "penalty_offsets", "penalty_ranks",
    "penalty_order", "penalty_sp_indices", "penalty_sp_labels",
    "sp_mapping", "sp_mapping_offset", "min_sp", "constraint",
    "constraint_mode", "constraint_nullspace", "constraint_rank",
    "constraint_nullspace_dimension", "H", "weights", "weights_policy",
    "offset", "offset_policy", "mgcv_penalty_rank_metadata",
    "smooth_classes", "smooth_labels", "smooth_terms", "smooth_by",
    "basis_dimensions", "smooth_p_order",
    "smooth_null_space_dimensions", "smooth_ranks",
    "smooth_side_constraints", "smooth_reparameterized",
    "smooth_parameter_ranges", "smooth_sp_ranges", "smooth_S_scale",
    "smooth_shift", "model_matrix_rank", "model_matrix_condition",
    "conditioning_rank", "conditioning_condition", "penalty_nullity",
    "scoring_n", "scoring_n_true", "scoring_min_edf",
    "scoring_pearson_extra", "scoring_deviance_extra",
    "weighted_X_policy", "gram_matrix", "nullspace_gram_policy",
    "nullspace_gram_matrix", "semantic_fingerprint",
    "representation_fingerprint"
  )
}

fastkpc_full_cuda_prepared_s_semantic_field_names <- function() {
  c(
    "sorted_S", "formula_class", "formula_semantics_version",
    "mgcv_semantics_version", "R_version", "mgcv_version", "family",
    "link", "method", "optimizer", "provider_fingerprint", "X",
    "coefficient_labels", "intercept", "assign", "cmX",
    "penalty_blocks", "penalty_offsets", "penalty_ranks",
    "penalty_order", "penalty_sp_indices", "penalty_sp_labels",
    "sp_mapping", "sp_mapping_offset", "min_sp", "constraint",
    "constraint_mode", "constraint_nullspace", "constraint_rank",
    "constraint_nullspace_dimension", "H", "weights", "weights_policy",
    "offset", "offset_policy", "mgcv_penalty_rank_metadata",
    "smooth_classes", "smooth_labels", "smooth_terms", "smooth_by",
    "basis_dimensions", "smooth_p_order",
    "smooth_null_space_dimensions", "smooth_ranks",
    "smooth_side_constraints", "smooth_reparameterized",
    "smooth_parameter_ranges", "smooth_sp_ranges", "smooth_S_scale",
    "smooth_shift", "model_matrix_rank", "model_matrix_condition",
    "conditioning_rank", "conditioning_condition", "penalty_nullity",
    "scoring_n", "scoring_n_true", "scoring_min_edf",
    "scoring_pearson_extra", "scoring_deviance_extra",
    "weighted_X_policy", "gram_matrix", "nullspace_gram_policy",
    "nullspace_gram_matrix"
  )
}

fastkpc_full_cuda_prepared_s_is_sha256 <- function(value) {
  length(value) == 1L && !is.na(value) &&
    grepl("^[0-9a-f]{64}$", as.character(value))
}

fastkpc_full_cuda_prepared_s_plain_hash <- function(value) {
  fastkpc_full_cuda_census_hash_raw(
    serialize(value, NULL, version = 2)
  )
}

fastkpc_full_cuda_prepared_s_field_hash <- function(value, fields) {
  if (!is.list(value) || length(setdiff(fields, names(value))) > 0L) {
    stop("PreparedSSetup fingerprint input is incomplete", call. = FALSE)
  }
  hashes <- setNames(vapply(fields, function(field) {
    fastkpc_full_cuda_prepared_s_plain_hash(value[[field]])
  }, character(1L)), fields)
  fastkpc_full_cuda_prepared_s_plain_hash(hashes)
}

fastkpc_full_cuda_prepared_s_semantic_fingerprint <- function(setup) {
  fastkpc_full_cuda_prepared_s_field_hash(
    setup, fastkpc_full_cuda_prepared_s_semantic_field_names()
  )
}

fastkpc_full_cuda_prepared_s_representation_fingerprint <- function(setup) {
  fields <- setdiff(
    fastkpc_full_cuda_prepared_s_setup_field_names(),
    "representation_fingerprint"
  )
  fastkpc_full_cuda_prepared_s_field_hash(setup, fields)
}

fastkpc_full_cuda_prepared_s_function_body_hash <- function(fun) {
  if (!is.function(fun)) {
    stop("PreparedSSetup provider helper is not a function", call. = FALSE)
  }
  body_text <- deparse(
    body(fun), width.cutoff = 500L,
    control = c("keepInteger", "keepNA")
  )
  fastkpc_full_cuda_census_hash_utf8(
    paste0(paste(body_text, collapse = "\n"), "\n")
  )
}

fastkpc_full_cuda_prepared_s_provider_descriptor <- function(
    R_version = R.version.string,
    mgcv_version = as.character(utils::packageVersion("mgcv"))) {
  fastkpc_full_cuda_census_require_parity_namespaces()
  if (!identical(as.character(R_version), R.version.string) ||
      !identical(
        as.character(mgcv_version),
        as.character(utils::packageVersion("mgcv"))
      )) {
    stop("PreparedSSetup provider runtime lineage mismatch", call. = FALSE)
  }
  body_hash <- function(name) {
    fastkpc_full_cuda_prepared_s_function_body_hash(
      getFromNamespace(name, "kpcalg")
    )
  }
  gam_formals <- formals(mgcv::gam)
  default_text <- function(field) {
    paste(deparse(gam_formals[[field]], width.cutoff = 500L), collapse = "\n")
  }
  na_action <- getOption("na.action")
  na_action <- if (is.function(na_action)) {
    paste0(
      "function:",
      fastkpc_full_cuda_prepared_s_function_body_hash(na_action)
    )
  } else {
    paste(as.character(na_action), collapse = ",")
  }
  contrasts <- getOption("contrasts")
  if (is.null(contrasts)) contrasts <- character()
  contrast_names <- names(contrasts)
  contrasts <- as.character(contrasts)
  names(contrasts) <- contrast_names
  list(
    schema_version = "full-cuda-ci-prepared-s-provider-v1",
    extractor_schema = fastkpc_full_cuda_prepared_s_setup_schema_version(),
    R_version = as.character(R_version),
    mgcv_version = as.character(mgcv_version),
    kpcalg_version = as.character(utils::packageVersion("kpcalg")),
    regrXonS_body_sha256 = body_hash("regrXonS"),
    full_smooth_formula_body_sha256 = body_hash("frml.full.smooth"),
    additive_smooth_formula_body_sha256 =
      body_hash("frml.additive.smooth"),
    contrasts = contrasts,
    runtime_na_action = na_action,
    contract_na_action = "na.fail",
    finite_input_policy = "finite-complete-data-v1",
    family = "gaussian",
    link = "identity",
    family_default = default_text("family"),
    method = "GCV.Cp",
    method_default = default_text("method"),
    optimizer = "mgcv-default",
    optimizer_default = default_text("optimizer"),
    select_default = default_text("select"),
    scale_default = default_text("scale"),
    fit = FALSE,
    fixed_sp = FALSE,
    formula_semantics_version = "kpcalg_regrXonS_v1",
    mgcv_semantics_version = "mgcv-gam-gcv-cp-v1"
  )
}

fastkpc_full_cuda_prepared_s_provider_fingerprint <- function(
    R_version = R.version.string,
    mgcv_version = as.character(utils::packageVersion("mgcv"))) {
  fastkpc_full_cuda_prepared_s_plain_hash(
    fastkpc_full_cuda_prepared_s_provider_descriptor(
      R_version = R_version, mgcv_version = mgcv_version
    )
  )
}

fastkpc_full_cuda_prepared_s_response_field_names <- function() {
  c(
    "G", "y", "target", "sp", "lsp0", "Xty", "Xty_null",
    "UZ", "Xu", "target_fingerprint", "target_fit_fingerprint",
    "target_state_fingerprint", "residual_hash", "fitted_hash",
    "selected_sp", "selected_sp_names", "selected_sp_hash"
  )
}

fastkpc_full_cuda_prepared_s_normalize_field_name <- function(value) {
  tolower(gsub("[^[:alnum:]]", "", as.character(value)))
}

fastkpc_full_cuda_prepared_s_find_response_fields <- function(
    value, path = "setup") {
  forbidden <- fastkpc_full_cuda_prepared_s_normalize_field_name(
    fastkpc_full_cuda_prepared_s_response_field_names()
  )
  hits <- character()
  if (is.character(value) && length(value) > 0L) {
    normalized_values <-
      fastkpc_full_cuda_prepared_s_normalize_field_name(value)
    bad_value <- nzchar(normalized_values) &
      normalized_values %in% forbidden
    if (any(bad_value)) {
      hits <- c(hits, paste0(path, "=<response-value>"))
    }
  }
  value_names <- names(value)
  if (!is.null(value_names)) {
    normalized <- fastkpc_full_cuda_prepared_s_normalize_field_name(
      value_names
    )
    bad <- nzchar(normalized) & normalized %in% forbidden
    if (any(bad)) {
      hits <- c(hits, paste0(path, "$", value_names[bad]))
    }
  }
  if (is.list(value)) {
    labels <- value_names
    if (is.null(labels)) labels <- as.character(seq_along(value))
    labels[!nzchar(labels)] <- as.character(which(!nzchar(labels)))
    for (index in seq_along(value)) {
      hits <- c(
        hits,
        fastkpc_full_cuda_prepared_s_find_response_fields(
          value[[index]], paste0(path, "$", labels[[index]])
        )
      )
    }
  }
  attrs <- attributes(value)
  if (!is.null(attrs)) {
    attr_names <- names(attrs)
    normalized <- fastkpc_full_cuda_prepared_s_normalize_field_name(
      attr_names
    )
    bad <- nzchar(normalized) & normalized %in% forbidden
    if (any(bad)) {
      hits <- c(hits, paste0(path, "@", attr_names[bad]))
    }
    for (index in seq_along(attrs)) {
      hits <- c(
        hits,
        fastkpc_full_cuda_prepared_s_find_response_fields(
          attrs[[index]], paste0(path, "@", attr_names[[index]])
        )
      )
    }
  }
  unique(hits)
}

fastkpc_full_cuda_prepared_s_is_executable_object <- function(value) {
  is.environment(value) || is.function(value) ||
    inherits(value, "formula") || is.call(value) ||
    typeof(value) %in% c(
      "externalptr", "weakref", "bytecode", "language", "symbol",
      "expression"
    ) ||
    isS4(value) || inherits(value, "gam") ||
    any(grepl("smooth", tolower(class(value)), fixed = TRUE))
}

fastkpc_full_cuda_prepared_s_find_executable_objects <- function(
    value, path = "setup") {
  if (fastkpc_full_cuda_prepared_s_is_executable_object(value)) {
    return(path)
  }
  hits <- character()
  if (is.list(value)) {
    labels <- names(value)
    if (is.null(labels)) labels <- as.character(seq_along(value))
    labels[!nzchar(labels)] <- as.character(which(!nzchar(labels)))
    for (index in seq_along(value)) {
      hits <- c(
        hits,
        fastkpc_full_cuda_prepared_s_find_executable_objects(
          value[[index]], paste0(path, "$", labels[[index]])
        )
      )
    }
  }
  attrs <- attributes(value)
  if (!is.null(attrs)) {
    attr_names <- names(attrs)
    for (index in seq_along(attrs)) {
      hits <- c(
        hits,
        fastkpc_full_cuda_prepared_s_find_executable_objects(
          attrs[[index]], paste0(path, "@", attr_names[[index]])
        )
      )
    }
  }
  unique(hits)
}

fastkpc_full_cuda_prepared_s_validate_plain_data <- function(
    value, path = "setup") {
  if (is.null(value)) return(invisible(TRUE))
  if (is.data.frame(value)) {
    fields <- names(value)
    if (is.null(fields) || any(!nzchar(fields)) || anyDuplicated(fields)) {
      stop("PreparedSSetup plain data requires named data frames: ", path,
           call. = FALSE)
    }
    allowed_attrs <- c("names", "row.names", "class")
    if (length(setdiff(names(attributes(value)), allowed_attrs)) > 0L) {
      stop("PreparedSSetup plain data has forbidden attributes: ", path,
           call. = FALSE)
    }
    for (field in fields) {
      fastkpc_full_cuda_prepared_s_validate_plain_data(
        value[[field]], paste0(path, "$", field)
      )
    }
    return(invisible(TRUE))
  }
  if (is.list(value)) {
    fields <- names(value)
    if (is.null(fields) || length(fields) != length(value) ||
        any(!nzchar(fields)) || anyDuplicated(fields)) {
      stop("PreparedSSetup plain data requires named lists: ", path,
           call. = FALSE)
    }
    if (!identical(names(attributes(value)), "names")) {
      stop("PreparedSSetup plain list has forbidden attributes: ", path,
           call. = FALSE)
    }
    for (field in fields) {
      fastkpc_full_cuda_prepared_s_validate_plain_data(
        value[[field]], paste0(path, "$", field)
      )
    }
    return(invisible(TRUE))
  }
  allowed_type <- typeof(value) %in% c(
    "logical", "integer", "double", "character"
  )
  if (!is.atomic(value) || !allowed_type) {
    stop("PreparedSSetup contains non-plain data: ", path, call. = FALSE)
  }
  attrs <- attributes(value)
  if (is.matrix(value)) {
    if (length(dim(value)) != 2L ||
        length(setdiff(names(attrs), c("dim", "dimnames"))) > 0L) {
      stop("PreparedSSetup matrix has forbidden attributes: ", path,
           call. = FALSE)
    }
  } else if (!is.null(attrs) &&
             length(setdiff(names(attrs), "names")) > 0L) {
    stop("PreparedSSetup vector has forbidden attributes: ", path,
         call. = FALSE)
  }
  invisible(TRUE)
}

fastkpc_full_cuda_prepared_s_matrix <- function(value) {
  value <- as.matrix(value)
  storage.mode(value) <- "double"
  dimnames(value) <- NULL
  value
}

fastkpc_full_cuda_prepared_s_named_list <- function(value, prefix) {
  names(value) <- paste0(prefix, seq_along(value))
  value
}

fastkpc_full_cuda_prepared_s_setup_row <- function(setup_row) {
  setup_row <- as.data.frame(setup_row, stringsAsFactors = FALSE)
  required <- c(
    "same_S_group_id", "S_key", "S_size", "formula_class",
    "formula_semantics_version", "model_matrix_nrow", "model_matrix_ncol",
    "model_matrix_hash", "model_matrix_rank", "model_matrix_condition",
    "penalty_count", "penalty_block_dimensions", "penalty_ranks",
    "penalty_offsets", "penalty_hashes", "penalty_nullity",
    "constraint_dimensions", "constraint_rank",
    "constraint_nullspace_dimension", "constraint_hash", "H_dimensions",
    "H_hash", "weights_policy", "offset_policy", "smooth_classes",
    "basis_dimensions", "conditioning_rank", "conditioning_condition",
    "setup_fingerprint", "mgcv_version", "R_version"
  )
  missing <- setdiff(required, names(setup_row))
  if (nrow(setup_row) != 1L || length(missing) > 0L) {
    stop(
      "PreparedSSetup requires one complete Phase 1 setup row",
      if (length(missing) > 0L) paste0(": ", paste(missing, collapse = ",")),
      call. = FALSE
    )
  }
  setup_row
}

fastkpc_full_cuda_prepared_s_row_value <- function(setup_row, field) {
  setup_row[[field]][[1L]]
}

fastkpc_full_cuda_prepared_s_layout <- function(data, S) {
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  S <- as.integer(S)
  if (nrow(data) == 0L || ncol(data) == 0L || any(!is.finite(data)) ||
      length(S) == 0L || anyNA(S) || any(S < 1L | S > ncol(data)) ||
      !identical(S, sort(unique(S), method = "radix"))) {
    stop("PreparedSSetup layout indexes or data are invalid", call. = FALSE)
  }
  frame <- data.frame(
    cbind(rep(0, nrow(data)), data[, S, drop = FALSE]),
    check.names = FALSE
  )
  names(frame) <- paste0("x", seq_len(ncol(frame)))
  formula_fun <- fastkpc_full_cuda_census_formula_function(length(S))
  list(
    data = frame,
    formula = formula_fun(1L, 2L:(1L + length(S)))
  )
}

fastkpc_full_cuda_prepared_s_smooth_state <- function(smooths) {
  if (is.null(smooths)) smooths <- list()
  smooth_ids <- paste0("smooth_", seq_along(smooths))
  named_list <- function(field, transform) {
    values <- lapply(smooths, function(smooth) transform(smooth[[field]]))
    names(values) <- smooth_ids
    values
  }
  ranges <- function(first, last) {
    if (length(smooths) == 0L) {
      return(matrix(integer(), nrow = 0L, ncol = 2L))
    }
    result <- t(vapply(smooths, function(smooth) {
      as.integer(c(smooth[[first]], smooth[[last]]))
    }, integer(2L)))
    dimnames(result) <- NULL
    result
  }
  list(
    classes = unname(vapply(smooths, function(smooth) {
      as.character(class(smooth)[[1L]])
    }, character(1L))),
    labels = unname(vapply(smooths, function(smooth) {
      as.character(smooth$label)
    }, character(1L))),
    terms = named_list("term", function(value) unname(as.character(value))),
    by = unname(vapply(smooths, function(smooth) {
      paste(as.character(smooth$by), collapse = ",")
    }, character(1L))),
    basis_dimensions = unname(vapply(smooths, function(smooth) {
      as.integer(smooth$bs.dim)
    }, integer(1L))),
    p_order = named_list("p.order", function(value) unname(as.numeric(value))),
    null_space_dimensions = unname(vapply(smooths, function(smooth) {
      as.integer(smooth$null.space.dim)
    }, integer(1L))),
    ranks = named_list("rank", function(value) unname(as.integer(value))),
    side_constraints = unname(vapply(smooths, function(smooth) {
      isTRUE(smooth$side.constrain)
    }, logical(1L))),
    reparameterized = unname(vapply(smooths, function(smooth) {
      isTRUE(smooth$repara)
    }, logical(1L))),
    parameter_ranges = ranges("first.para", "last.para"),
    sp_ranges = ranges("first.sp", "last.sp"),
    S_scale = named_list("S.scale", function(value) unname(as.numeric(value))),
    shift = named_list("shift", function(value) unname(as.numeric(value)))
  )
}

fastkpc_full_cuda_prepared_s_expected_smooth_state <- function(
    sorted_S, formula_class) {
  S_size <- length(as.integer(sorted_S))
  if (S_size == 0L ||
      !formula_class %in% c("full-smooth", "additive-smooth")) {
    stop("PreparedSSetup smooth formula identity is invalid", call. = FALSE)
  }
  predictor_terms <- paste0("x", 2L:(1L + S_size))
  terms <- if (identical(formula_class, "full-smooth")) {
    list(unname(predictor_terms))
  } else {
    lapply(predictor_terms, function(term) unname(term))
  }
  names(terms) <- paste0("smooth_", seq_along(terms))
  labels <- if (identical(formula_class, "full-smooth")) {
    paste0("s(", paste(predictor_terms, collapse = ","), ")")
  } else {
    paste0("s(", predictor_terms, ")")
  }
  list(
    count = as.integer(length(terms)),
    classes = rep("tprs.smooth", length(terms)),
    labels = unname(labels),
    terms = terms,
    by = rep("NA", length(terms))
  )
}

fastkpc_full_cuda_prepared_s_policy_state <- function(
    value, neutral, neutral_policy, nonneutral_label) {
  if (is.null(value) || length(value) == 0L) {
    return(list(value = NULL, policy = neutral_policy))
  }
  value <- unname(as.numeric(value))
  if (any(!is.finite(value))) {
    stop("PreparedSSetup policy state must be finite", call. = FALSE)
  }
  if (all(value == neutral)) {
    return(list(value = NULL, policy = neutral_policy))
  }
  list(
    value = value,
    policy = fastkpc_full_cuda_prepared_s_nonneutral_policy(
      value, nonneutral_label
    )
  )
}

fastkpc_full_cuda_prepared_s_nonneutral_policy <- function(value, label) {
  paste0(
    label, ":",
    fastkpc_full_cuda_census_metadata_hash(unname(as.numeric(value)))
  )
}

fastkpc_full_cuda_build_prepared_s_setup <- function(inputs, setup_row) {
  setup_row <- fastkpc_full_cuda_prepared_s_setup_row(setup_row)
  if (!is.list(inputs) ||
      length(setdiff(c("data", "dataset_sha256", "manifest"),
                     names(inputs))) > 0L) {
    stop("PreparedSSetup inputs are incomplete", call. = FALSE)
  }
  data <- as.matrix(inputs$data)
  storage.mode(data) <- "double"
  dataset_sha256 <- as.character(inputs$dataset_sha256)
  if (!fastkpc_full_cuda_prepared_s_is_sha256(dataset_sha256) ||
      any(!is.finite(data)) ||
      !identical(fastkpc_full_cuda_data_hash(data), dataset_sha256)) {
    stop("PreparedSSetup dataset identity mismatch", call. = FALSE)
  }
  R_version <- as.character(inputs$manifest$R_version)
  mgcv_version <- as.character(inputs$manifest$mgcv_version)
  if (!identical(R_version, R.version.string) ||
      !identical(mgcv_version,
                 as.character(utils::packageVersion("mgcv")))) {
    stop("PreparedSSetup runtime lineage mismatch", call. = FALSE)
  }

  if (!is.null(inputs$same_s_setup_metadata)) {
    canonical_rows <- as.data.frame(
      inputs$same_s_setup_metadata, stringsAsFactors = FALSE
    )
    group_id <- as.character(
      fastkpc_full_cuda_prepared_s_row_value(
        setup_row, "same_S_group_id"
      )
    )
    index <- which(as.character(canonical_rows$same_S_group_id) == group_id)
    if (length(index) != 1L ||
        !identical(
          fastkpc_full_cuda_census_frame_hash(
            canonical_rows[index, names(setup_row), drop = FALSE]
          ),
          fastkpc_full_cuda_census_frame_hash(setup_row)
        )) {
      stop("PreparedSSetup setup-row lineage mismatch", call. = FALSE)
    }
  }

  S <- fastkpc_full_cuda_census_parse_s(
    fastkpc_full_cuda_prepared_s_row_value(setup_row, "S_key")
  )
  formula_class <- fastkpc_regrxons_formula_class(S)
  expected_group_payload <- fastkpc_full_cuda_census_same_s_payload(
    S = S, formula_class = formula_class, data_hash = dataset_sha256,
    n = nrow(data), p = ncol(data)
  )
  expected_group_id <- fastkpc_full_cuda_census_hash_utf8(
    expected_group_payload
  )
  row_clean <-
    identical(
      as.character(fastkpc_full_cuda_prepared_s_row_value(
        setup_row, "same_S_group_id"
      )),
      expected_group_id
    ) &&
    identical(
      as.integer(fastkpc_full_cuda_prepared_s_row_value(setup_row, "S_size")),
      as.integer(length(S))
    ) &&
    identical(
      as.character(fastkpc_full_cuda_prepared_s_row_value(
        setup_row, "formula_class"
      )),
      formula_class
    ) &&
    identical(
      as.character(fastkpc_full_cuda_prepared_s_row_value(
        setup_row, "formula_semantics_version"
      )),
      "kpcalg_regrXonS_v1"
    ) &&
    identical(
      as.character(fastkpc_full_cuda_prepared_s_row_value(
        setup_row, "R_version"
      )),
      R_version
    ) &&
    identical(
      as.character(fastkpc_full_cuda_prepared_s_row_value(
        setup_row, "mgcv_version"
      )),
      mgcv_version
    )
  if (!isTRUE(row_clean)) {
    stop("PreparedSSetup setup-row identity mismatch", call. = FALSE)
  }

  layout <- fastkpc_full_cuda_prepared_s_layout(data, S)
  raw <- mgcv::gam(
    formula = layout$formula,
    data = layout$data,
    method = "GCV.Cp",
    fit = FALSE
  )
  family <- raw$family
  if (is.null(family) || !identical(as.character(family$family), "gaussian") ||
      !identical(as.character(family$link), "identity")) {
    stop("PreparedSSetup provider family mismatch", call. = FALSE)
  }

  coefficient_labels <- colnames(raw$X)
  X <- fastkpc_full_cuda_prepared_s_matrix(raw$X)
  coefficient_count <- ncol(X)
  if (is.null(coefficient_labels)) {
    coefficient_labels <- rep("", coefficient_count)
  }
  coefficient_labels <- unname(as.character(coefficient_labels))
  if (length(coefficient_labels) != coefficient_count) {
    stop("PreparedSSetup coefficient labels are incomplete", call. = FALSE)
  }

  penalty_blocks <- lapply(raw$S, fastkpc_full_cuda_prepared_s_matrix)
  penalty_blocks <- fastkpc_full_cuda_prepared_s_named_list(
    penalty_blocks, "penalty_"
  )
  penalty_count <- length(penalty_blocks)
  penalty_offsets <- unname(as.integer(raw$off))
  penalty_ranks <- unname(vapply(penalty_blocks, function(block) {
    fastkpc_full_cuda_census_svd_diagnostics(
      block, expected_rank = ncol(block)
    )$rank
  }, integer(1L)))
  mgcv_penalty_rank_metadata <- unname(as.integer(raw$rank))
  smooth_state <- fastkpc_full_cuda_prepared_s_smooth_state(raw$smooth)
  penalty_sp_indices <- if (nrow(smooth_state$sp_ranges) == 0L) {
    integer()
  } else {
    unname(as.integer(unlist(lapply(seq_len(nrow(
      smooth_state$sp_ranges
    )), function(index) {
      seq.int(
        smooth_state$sp_ranges[index, 1L],
        smooth_state$sp_ranges[index, 2L]
      )
    }), use.names = FALSE)))
  }
  if (!identical(penalty_sp_indices, seq_len(penalty_count)) ||
      length(penalty_offsets) != penalty_count ||
      length(mgcv_penalty_rank_metadata) != penalty_count) {
    stop("PreparedSSetup provider penalty order mismatch", call. = FALSE)
  }
  penalty_order <- seq_len(penalty_count)
  sp_mapping_offset <- if (is.null(raw$lsp0)) {
    numeric(penalty_count)
  } else {
    unname(as.numeric(raw$lsp0))
  }
  penalty_sp_labels <- names(raw$lsp0)
  if (is.null(penalty_sp_labels) ||
      length(penalty_sp_labels) != penalty_count ||
      any(!nzchar(penalty_sp_labels))) {
    penalty_sp_labels <- character(penalty_count)
    for (index in seq_len(nrow(smooth_state$sp_ranges))) {
      sp_index <- seq.int(
        smooth_state$sp_ranges[index, 1L],
        smooth_state$sp_ranges[index, 2L]
      )
      label <- smooth_state$labels[[index]]
      penalty_sp_labels[sp_index] <- if (length(sp_index) == 1L) {
        label
      } else {
        paste0(label, ".", seq_along(sp_index))
      }
    }
  }
  penalty_sp_labels <- unname(as.character(penalty_sp_labels))
  sp_mapping <- if (is.null(raw$L) || length(raw$L) == 0L) {
    NULL
  } else {
    fastkpc_full_cuda_prepared_s_matrix(raw$L)
  }
  min_sp <- if (is.null(raw$min.sp) || length(raw$min.sp) == 0L) {
    NULL
  } else {
    unname(as.numeric(raw$min.sp))
  }
  canonical_mapping <- is.null(sp_mapping) &&
    is.numeric(sp_mapping_offset) &&
    length(sp_mapping_offset) == penalty_count &&
    all(is.finite(sp_mapping_offset)) &&
    all(sp_mapping_offset == 0) && is.null(min_sp)
  if (!isTRUE(canonical_mapping)) {
    stop(
      "PreparedSSetup provider smoothing mapping mismatch",
      call. = FALSE
    )
  }

  constraint <- if (is.null(raw$C) || length(raw$C) == 0L) {
    matrix(numeric(), nrow = 0L, ncol = coefficient_count)
  } else {
    fastkpc_full_cuda_prepared_s_matrix(raw$C)
  }
  dimnames(constraint) <- NULL
  constraint_mode <- if (nrow(constraint) == 0L) "identity" else "explicit"
  constraint_nullspace <- if (identical(constraint_mode, "identity")) {
    NULL
  } else {
    fastkpc_full_cuda_prepared_s_matrix(
      fastkpc_constraint_nullspace(constraint, coefficient_count)
    )
  }
  constraint_rank <- fastkpc_full_cuda_census_svd_diagnostics(
    constraint, expected_rank = min(dim(constraint))
  )$rank
  constraint_nullspace_dimension <- if (is.null(constraint_nullspace)) {
    coefficient_count
  } else {
    ncol(constraint_nullspace)
  }

  H <- if (is.null(raw$H) || length(raw$H) == 0L) {
    NULL
  } else {
    fastkpc_full_cuda_prepared_s_matrix(raw$H)
  }
  weights_state <- fastkpc_full_cuda_prepared_s_policy_state(
    raw$w, neutral = 1, neutral_policy = "none-or-unit",
    nonneutral_label = "nonunit"
  )
  offset_state <- fastkpc_full_cuda_prepared_s_policy_state(
    raw$offset, neutral = 0, neutral_policy = "none-or-zero",
    nonneutral_label = "nonzero"
  )

  X_weighted <- if (is.null(weights_state$value)) {
    X
  } else {
    X * sqrt(weights_state$value)
  }
  weighted_X_policy <- if (is.null(weights_state$value)) {
    "alias-X-unit-weights"
  } else {
    "sqrt-weights-row-scaled"
  }
  gram_matrix <- fastkpc_full_cuda_prepared_s_matrix(crossprod(X_weighted))
  nullspace_gram_policy <- if (identical(constraint_mode, "identity")) {
    "alias-gram"
  } else {
    "explicit-nullspace-gram"
  }
  nullspace_gram_matrix <- if (is.null(constraint_nullspace)) {
    NULL
  } else {
    fastkpc_full_cuda_prepared_s_matrix(
      crossprod(X_weighted %*% constraint_nullspace)
    )
  }

  P_unit <- if (penalty_count == 0L) {
    matrix(0, coefficient_count, coefficient_count)
  } else {
    fastkpc_assemble_penalty(
      p = coefficient_count, S = penalty_blocks,
      off = penalty_offsets, sp = rep(1, penalty_count), H = NULL
    )
  }
  projected_penalty <- if (is.null(constraint_nullspace)) {
    P_unit
  } else {
    crossprod(
      constraint_nullspace, P_unit %*% constraint_nullspace
    )
  }
  projected_penalty <- fastkpc_full_cuda_prepared_s_matrix(
    projected_penalty
  )
  projected_rank <- fastkpc_full_cuda_census_svd_diagnostics(
    projected_penalty, expected_rank = ncol(projected_penalty)
  )$rank
  penalty_nullity <- as.integer(
    constraint_nullspace_dimension - projected_rank
  )

  model_diagnostics <- fastkpc_full_cuda_census_svd_diagnostics(
    X, expected_rank = ncol(X)
  )
  conditioning_data <- data[, S, drop = FALSE]
  conditioning_diagnostics <- fastkpc_full_cuda_census_svd_diagnostics(
    conditioning_data, expected_rank = ncol(conditioning_data)
  )
  key <- fastkpc_full_cuda_prepared_s_key(
    setup_row = setup_row,
    dataset_sha256 = dataset_sha256,
    R_version = R_version,
    mgcv_version = mgcv_version
  )
  provider_fingerprint <-
    fastkpc_full_cuda_prepared_s_provider_fingerprint(
      R_version = R_version, mgcv_version = mgcv_version
    )

  setup <- list(
    schema_version = fastkpc_full_cuda_prepared_s_setup_schema_version(),
    dataset_sha256 = dataset_sha256,
    prepared_s_key_payload = key$payload,
    prepared_s_key_sha256 = key$sha256,
    same_S_group_id = as.character(
      fastkpc_full_cuda_prepared_s_row_value(
        setup_row, "same_S_group_id"
      )
    ),
    phase1_setup_fingerprint = as.character(
      fastkpc_full_cuda_prepared_s_row_value(
        setup_row, "setup_fingerprint"
      )
    ),
    phase1_model_matrix_hash = as.character(
      fastkpc_full_cuda_prepared_s_row_value(
        setup_row, "model_matrix_hash"
      )
    ),
    sorted_S = unname(as.integer(S)),
    formula_class = formula_class,
    formula_semantics_version = "kpcalg_regrXonS_v1",
    mgcv_semantics_version = "mgcv-gam-gcv-cp-v1",
    R_version = R_version,
    mgcv_version = mgcv_version,
    family = "gaussian",
    link = "identity",
    method = "GCV.Cp",
    optimizer = "mgcv-default",
    provider_fingerprint = provider_fingerprint,
    provider_X_hash = fastkpc_full_cuda_prepared_s_plain_hash(X),
    X = X,
    coefficient_labels = coefficient_labels,
    intercept = isTRUE(raw$intercept),
    assign = unname(as.integer(raw$assign)),
    cmX = unname(as.numeric(raw$cmX)),
    penalty_blocks = penalty_blocks,
    penalty_offsets = penalty_offsets,
    penalty_ranks = penalty_ranks,
    penalty_order = unname(as.integer(penalty_order)),
    penalty_sp_indices = penalty_sp_indices,
    penalty_sp_labels = penalty_sp_labels,
    sp_mapping = sp_mapping,
    sp_mapping_offset = sp_mapping_offset,
    min_sp = min_sp,
    constraint = constraint,
    constraint_mode = constraint_mode,
    constraint_nullspace = constraint_nullspace,
    constraint_rank = as.integer(constraint_rank),
    constraint_nullspace_dimension =
      as.integer(constraint_nullspace_dimension),
    H = H,
    weights = weights_state$value,
    weights_policy = weights_state$policy,
    offset = offset_state$value,
    offset_policy = offset_state$policy,
    mgcv_penalty_rank_metadata = mgcv_penalty_rank_metadata,
    smooth_classes = smooth_state$classes,
    smooth_labels = smooth_state$labels,
    smooth_terms = smooth_state$terms,
    smooth_by = smooth_state$by,
    basis_dimensions = smooth_state$basis_dimensions,
    smooth_p_order = smooth_state$p_order,
    smooth_null_space_dimensions = smooth_state$null_space_dimensions,
    smooth_ranks = smooth_state$ranks,
    smooth_side_constraints = smooth_state$side_constraints,
    smooth_reparameterized = smooth_state$reparameterized,
    smooth_parameter_ranges = smooth_state$parameter_ranges,
    smooth_sp_ranges = smooth_state$sp_ranges,
    smooth_S_scale = smooth_state$S_scale,
    smooth_shift = smooth_state$shift,
    model_matrix_rank = as.integer(model_diagnostics$rank),
    model_matrix_condition = as.numeric(model_diagnostics$condition),
    conditioning_rank = as.integer(conditioning_diagnostics$rank),
    conditioning_condition = as.numeric(conditioning_diagnostics$condition),
    penalty_nullity = penalty_nullity,
    scoring_n = as.integer(raw$n),
    scoring_n_true = as.integer(raw$n.true),
    scoring_min_edf = as.numeric(raw$min.edf),
    scoring_pearson_extra = as.numeric(raw$pearson.extra),
    scoring_deviance_extra = as.numeric(raw$dev.extra),
    weighted_X_policy = weighted_X_policy,
    gram_matrix = gram_matrix,
    nullspace_gram_policy = nullspace_gram_policy,
    nullspace_gram_matrix = nullspace_gram_matrix
  )
  setup$semantic_fingerprint <-
    fastkpc_full_cuda_prepared_s_semantic_fingerprint(setup)
  setup$representation_fingerprint <-
    fastkpc_full_cuda_prepared_s_representation_fingerprint(setup)
  if (!identical(
        names(setup), fastkpc_full_cuda_prepared_s_setup_field_names()
      )) {
    stop("PreparedSSetup canonical field order mismatch", call. = FALSE)
  }
  setup
}

fastkpc_full_cuda_prepared_s_condition_compatible <- function(
    actual, expected, tolerance = sqrt(.Machine$double.eps)) {
  actual <- as.numeric(actual)
  expected <- as.numeric(expected)
  if (length(actual) != 1L || length(expected) != 1L) return(FALSE)
  if (is.na(actual) || is.na(expected)) return(is.na(actual) && is.na(expected))
  if (is.infinite(actual) || is.infinite(expected)) {
    return(identical(actual, expected))
  }
  if (!is.finite(actual) || !is.finite(expected)) return(FALSE)
  abs(actual - expected) <=
    as.numeric(tolerance) * max(1, abs(actual), abs(expected))
}

fastkpc_full_cuda_prepared_s_require_matrix <- function(
    value, dimensions, label, finite = TRUE) {
  clean <- is.matrix(value) && is.numeric(value) &&
    identical(dim(value), as.integer(dimensions)) &&
    (!isTRUE(finite) || all(is.finite(value)))
  if (!isTRUE(clean)) {
    stop("PreparedSSetup ", label, " mismatch", call. = FALSE)
  }
  invisible(TRUE)
}

fastkpc_full_cuda_prepared_s_require_named_list <- function(
    value, count, prefix, label) {
  expected_names <- paste0(prefix, seq_len(count))
  if (!is.list(value) || length(value) != count ||
      !identical(names(value), expected_names)) {
    stop("PreparedSSetup ", label, " mismatch", call. = FALSE)
  }
  invisible(TRUE)
}

fastkpc_full_cuda_validate_prepared_s_setup <- function(
    setup, setup_row, dataset_sha256,
    model_condition_tolerance = sqrt(.Machine$double.eps), ...) {
  executable_hits <-
    fastkpc_full_cuda_prepared_s_find_executable_objects(setup)
  if (length(executable_hits) > 0L) {
    stop(
      "PreparedSSetup executable object: ", executable_hits[[1L]],
      call. = FALSE
    )
  }
  response_hits <- fastkpc_full_cuda_prepared_s_find_response_fields(setup)
  if (length(response_hits) > 0L) {
    stop(
      "PreparedSSetup response-bearing field: ", response_hits[[1L]],
      call. = FALSE
    )
  }
  fastkpc_full_cuda_prepared_s_validate_plain_data(setup)
  expected_fields <- fastkpc_full_cuda_prepared_s_setup_field_names()
  if (!is.list(setup) || !identical(names(setup), expected_fields)) {
    stop("PreparedSSetup schema field mismatch", call. = FALSE)
  }
  if (!identical(
        setup$schema_version,
        fastkpc_full_cuda_prepared_s_setup_schema_version()
      )) {
    stop("PreparedSSetup schema mismatch", call. = FALSE)
  }
  dataset_sha256 <- as.character(dataset_sha256)
  if (!fastkpc_full_cuda_prepared_s_is_sha256(dataset_sha256) ||
      !identical(setup$dataset_sha256, dataset_sha256)) {
    stop("PreparedSSetup dataset lineage mismatch", call. = FALSE)
  }
  if (!fastkpc_full_cuda_prepared_s_is_sha256(
        setup$phase1_setup_fingerprint
      ) ||
      !fastkpc_full_cuda_prepared_s_is_sha256(
        setup$phase1_model_matrix_hash
      )) {
    stop("PreparedSSetup lineage fingerprint is malformed", call. = FALSE)
  }
  if (!fastkpc_full_cuda_prepared_s_is_sha256(
        setup$prepared_s_key_sha256
      )) {
    stop("PreparedSKey fingerprint is malformed", call. = FALSE)
  }
  fingerprint_fields <- c(
    "provider_fingerprint", "provider_X_hash", "semantic_fingerprint",
    "representation_fingerprint"
  )
  malformed_fingerprint <- fingerprint_fields[!vapply(
    setup[fingerprint_fields],
    fastkpc_full_cuda_prepared_s_is_sha256,
    logical(1L)
  )]
  if (length(malformed_fingerprint) > 0L) {
    stop(
      "PreparedSSetup fingerprint is malformed: ",
      malformed_fingerprint[[1L]], call. = FALSE
    )
  }

  setup_row <- fastkpc_full_cuda_prepared_s_setup_row(setup_row)
  row_group_id <- as.character(
    fastkpc_full_cuda_prepared_s_row_value(
      setup_row, "same_S_group_id"
    )
  )
  row_setup_fingerprint <- as.character(
    fastkpc_full_cuda_prepared_s_row_value(
      setup_row, "setup_fingerprint"
    )
  )
  row_model_hash <- as.character(
    fastkpc_full_cuda_prepared_s_row_value(
      setup_row, "model_matrix_hash"
    )
  )
  if (!fastkpc_full_cuda_prepared_s_is_sha256(row_group_id) ||
      !fastkpc_full_cuda_prepared_s_is_sha256(row_setup_fingerprint) ||
      !fastkpc_full_cuda_prepared_s_is_sha256(row_model_hash) ||
      !identical(setup$same_S_group_id, row_group_id) ||
      !identical(
        setup$phase1_setup_fingerprint, row_setup_fingerprint
      ) ||
      !identical(setup$phase1_model_matrix_hash, row_model_hash)) {
    stop("PreparedSSetup lineage mismatch", call. = FALSE)
  }

  S <- fastkpc_full_cuda_census_parse_s(
    fastkpc_full_cuda_prepared_s_row_value(setup_row, "S_key")
  )
  formula_class <- fastkpc_regrxons_formula_class(S)
  identity_clean <-
    identical(setup$sorted_S, unname(as.integer(S))) &&
    identical(setup$formula_class, formula_class) &&
    identical(
      setup$formula_class,
      as.character(fastkpc_full_cuda_prepared_s_row_value(
        setup_row, "formula_class"
      ))
    ) &&
    identical(setup$formula_semantics_version, "kpcalg_regrXonS_v1") &&
    identical(
      setup$formula_semantics_version,
      as.character(fastkpc_full_cuda_prepared_s_row_value(
        setup_row, "formula_semantics_version"
      ))
    ) &&
    identical(setup$mgcv_semantics_version, "mgcv-gam-gcv-cp-v1") &&
    identical(setup$family, "gaussian") &&
    identical(setup$link, "identity") &&
    identical(setup$method, "GCV.Cp") &&
    identical(setup$optimizer, "mgcv-default") &&
    identical(
      setup$R_version,
      as.character(fastkpc_full_cuda_prepared_s_row_value(
        setup_row, "R_version"
      ))
    ) &&
    identical(
      setup$mgcv_version,
      as.character(fastkpc_full_cuda_prepared_s_row_value(
        setup_row, "mgcv_version"
      ))
    ) &&
    identical(setup$R_version, R.version.string) &&
    identical(
      setup$mgcv_version,
      as.character(utils::packageVersion("mgcv"))
    )
  if (!isTRUE(identity_clean)) {
    stop("PreparedSSetup provider/formula lineage mismatch", call. = FALSE)
  }
  expected_key <- fastkpc_full_cuda_prepared_s_key(
    setup_row = setup_row,
    dataset_sha256 = dataset_sha256,
    R_version = setup$R_version,
    mgcv_version = setup$mgcv_version
  )
  if (!identical(setup$prepared_s_key_payload, expected_key$payload) ||
      !identical(setup$prepared_s_key_sha256, expected_key$sha256)) {
    stop("PreparedSKey exact mapping mismatch", call. = FALSE)
  }
  expected_provider <- fastkpc_full_cuda_prepared_s_provider_fingerprint(
    R_version = setup$R_version, mgcv_version = setup$mgcv_version
  )
  if (!identical(setup$provider_fingerprint, expected_provider)) {
    stop("PreparedSSetup provider fingerprint mismatch", call. = FALSE)
  }

  if (!is.matrix(setup$X) || !is.numeric(setup$X) ||
      any(!is.finite(setup$X))) {
    stop("PreparedSSetup X must be finite", call. = FALSE)
  }
  X <- fastkpc_full_cuda_prepared_s_matrix(setup$X)
  n <- nrow(X)
  p <- ncol(X)
  if (!identical(
        setup$provider_X_hash,
        fastkpc_full_cuda_prepared_s_plain_hash(X)
      )) {
    stop("PreparedSSetup provider X hash mismatch", call. = FALSE)
  }
  phase1_dimensions <- c(
    as.integer(fastkpc_full_cuda_prepared_s_row_value(
      setup_row, "model_matrix_nrow"
    )),
    as.integer(fastkpc_full_cuda_prepared_s_row_value(
      setup_row, "model_matrix_ncol"
    ))
  )
  if (!identical(dim(X), phase1_dimensions)) {
    stop("PreparedSSetup Phase 1 model dimensions mismatch", call. = FALSE)
  }
  model_diagnostics <- fastkpc_full_cuda_census_svd_diagnostics(
    X, expected_rank = p
  )
  phase1_model_rank <- as.integer(
    fastkpc_full_cuda_prepared_s_row_value(setup_row, "model_matrix_rank")
  )
  phase1_model_condition <- as.numeric(
    fastkpc_full_cuda_prepared_s_row_value(
      setup_row, "model_matrix_condition"
    )
  )
  if (!identical(setup$model_matrix_rank, model_diagnostics$rank) ||
      !identical(setup$model_matrix_rank, phase1_model_rank) ||
      !identical(
        as.numeric(setup$model_matrix_condition),
        as.numeric(model_diagnostics$condition)
      ) ||
      !fastkpc_full_cuda_prepared_s_condition_compatible(
        setup$model_matrix_condition, phase1_model_condition,
        tolerance = model_condition_tolerance
      )) {
    stop("PreparedSSetup model-space diagnostics mismatch", call. = FALSE)
  }
  coefficient_clean <- is.character(setup$coefficient_labels) &&
    length(setup$coefficient_labels) == p &&
    length(setup$intercept) == 1L && is.logical(setup$intercept) &&
    is.integer(setup$assign) && all(is.finite(setup$assign)) &&
    is.numeric(setup$cmX) && length(setup$cmX) == p &&
    all(is.finite(setup$cmX))
  if (!isTRUE(coefficient_clean)) {
    stop("PreparedSSetup coefficient metadata mismatch", call. = FALSE)
  }

  penalty_count <- as.integer(
    fastkpc_full_cuda_prepared_s_row_value(setup_row, "penalty_count")
  )
  fastkpc_full_cuda_prepared_s_require_named_list(
    setup$penalty_blocks, penalty_count, "penalty_", "penalty blocks"
  )
  expected_penalty_order <- seq_len(penalty_count)
  if (!identical(setup$penalty_order, expected_penalty_order) ||
      !identical(setup$penalty_sp_indices, expected_penalty_order)) {
    stop("PreparedSSetup penalty order mismatch", call. = FALSE)
  }
  phase1_offsets <- unname(as.integer(
    fastkpc_full_cuda_prepared_s_row_value(setup_row, "penalty_offsets")
  ))
  if (!identical(setup$penalty_offsets, phase1_offsets) ||
      length(setup$penalty_offsets) != penalty_count) {
    stop("PreparedSSetup penalty offset mismatch", call. = FALSE)
  }
  phase1_dimensions <- as.character(
    fastkpc_full_cuda_prepared_s_row_value(
      setup_row, "penalty_block_dimensions"
    )
  )
  actual_dimensions <- vapply(
    setup$penalty_blocks,
    function(block) paste(dim(block), collapse = "x"),
    character(1L)
  )
  block_clean <- vapply(setup$penalty_blocks, function(block) {
    is.matrix(block) && is.numeric(block) && nrow(block) == ncol(block) &&
      all(is.finite(block))
  }, logical(1L))
  if (!all(block_clean) ||
      !identical(unname(actual_dimensions), unname(phase1_dimensions))) {
    stop("PreparedSSetup penalty dimension mismatch", call. = FALSE)
  }
  for (index in seq_len(penalty_count)) {
    block_size <- nrow(setup$penalty_blocks[[index]])
    coefficient_index <- seq.int(
      setup$penalty_offsets[[index]], length.out = block_size
    )
    if (length(coefficient_index) != block_size ||
        min(coefficient_index) < 1L || max(coefficient_index) > p) {
      stop("PreparedSSetup penalty offset mismatch", call. = FALSE)
    }
  }
  phase1_penalty_hashes <- as.character(
    fastkpc_full_cuda_prepared_s_row_value(setup_row, "penalty_hashes")
  )
  actual_penalty_hashes <- unname(vapply(
    setup$penalty_blocks,
    fastkpc_full_cuda_census_metadata_hash,
    character(1L)
  ))
  if (!identical(actual_penalty_hashes, unname(phase1_penalty_hashes))) {
    stop("PreparedSSetup penalty hash mismatch", call. = FALSE)
  }
  actual_penalty_ranks <- unname(vapply(
    setup$penalty_blocks,
    function(block) fastkpc_full_cuda_census_svd_diagnostics(
      block, expected_rank = ncol(block)
    )$rank,
    integer(1L)
  ))
  phase1_penalty_ranks <- unname(as.integer(
    fastkpc_full_cuda_prepared_s_row_value(setup_row, "penalty_ranks")
  ))
  if (!identical(setup$penalty_ranks, actual_penalty_ranks) ||
      !identical(setup$penalty_ranks, phase1_penalty_ranks) ||
      !identical(
        setup$mgcv_penalty_rank_metadata, phase1_penalty_ranks
      )) {
    stop("PreparedSSetup penalty rank mismatch", call. = FALSE)
  }
  if (!is.character(setup$penalty_sp_labels) ||
      length(setup$penalty_sp_labels) != penalty_count ||
      anyNA(setup$penalty_sp_labels) ||
      any(!nzchar(setup$penalty_sp_labels))) {
    stop("PreparedSSetup penalty label mismatch", call. = FALSE)
  }
  mapping_clean <-
    is.null(setup$sp_mapping) &&
    is.numeric(setup$sp_mapping_offset) &&
    length(setup$sp_mapping_offset) == penalty_count &&
    all(is.finite(setup$sp_mapping_offset)) &&
    all(setup$sp_mapping_offset == 0) && is.null(setup$min_sp)
  if (!isTRUE(mapping_clean)) {
    stop("PreparedSSetup smoothing mapping mismatch", call. = FALSE)
  }

  fastkpc_full_cuda_prepared_s_require_matrix(
    setup$constraint, c(nrow(setup$constraint), p), "constraint"
  )
  expected_constraint_mode <- if (nrow(setup$constraint) == 0L) {
    "identity"
  } else {
    "explicit"
  }
  if (!identical(setup$constraint_mode, expected_constraint_mode)) {
    stop("PreparedSSetup constraint mode mismatch", call. = FALSE)
  }
  constraint_diagnostics <- fastkpc_full_cuda_census_svd_diagnostics(
    setup$constraint, expected_rank = min(dim(setup$constraint))
  )
  if (identical(expected_constraint_mode, "explicit") &&
      !identical(
        constraint_diagnostics$rank,
        as.integer(nrow(setup$constraint))
      )) {
    stop(
      "PreparedSSetup constraint rows must be independent",
      call. = FALSE
    )
  }
  phase1_constraint_dimensions <- unname(as.integer(
    fastkpc_full_cuda_prepared_s_row_value(
      setup_row, "constraint_dimensions"
    )
  ))
  phase1_constraint_rank <- as.integer(
    fastkpc_full_cuda_prepared_s_row_value(setup_row, "constraint_rank")
  )
  phase1_nullspace_dimension <- as.integer(
    fastkpc_full_cuda_prepared_s_row_value(
      setup_row, "constraint_nullspace_dimension"
    )
  )
  phase1_constraint_hash <- as.character(
    fastkpc_full_cuda_prepared_s_row_value(setup_row, "constraint_hash")
  )
  if (!identical(dim(setup$constraint), phase1_constraint_dimensions) ||
      !identical(setup$constraint_rank, constraint_diagnostics$rank) ||
      !identical(setup$constraint_rank, phase1_constraint_rank) ||
      !identical(
        fastkpc_full_cuda_census_metadata_hash(setup$constraint),
        phase1_constraint_hash
      )) {
    stop("PreparedSSetup constraint hash or rank mismatch", call. = FALSE)
  }
  if (identical(expected_constraint_mode, "explicit") &&
      !identical(
        as.integer(qr(t(setup$constraint))$rank),
        as.integer(nrow(setup$constraint))
      )) {
    stop(
      "PreparedSSetup constraint rows must be independent",
      call. = FALSE
    )
  }
  if (identical(setup$constraint_mode, "identity")) {
    nullspace_clean <- is.null(setup$constraint_nullspace) &&
      identical(setup$constraint_nullspace_dimension, p) &&
      identical(setup$constraint_nullspace_dimension,
                phase1_nullspace_dimension)
  } else {
    expected_nullspace <- fastkpc_full_cuda_prepared_s_matrix(
      fastkpc_constraint_nullspace(setup$constraint, p)
    )
    nullspace_clean <- is.matrix(setup$constraint_nullspace) &&
      all(is.finite(setup$constraint_nullspace)) &&
      identical(
        fastkpc_full_cuda_prepared_s_plain_hash(
          setup$constraint_nullspace
        ),
        fastkpc_full_cuda_prepared_s_plain_hash(expected_nullspace)
      ) &&
      identical(setup$constraint_nullspace_dimension,
                ncol(expected_nullspace)) &&
      identical(setup$constraint_nullspace_dimension,
                phase1_nullspace_dimension)
  }
  if (!isTRUE(nullspace_clean)) {
    stop("PreparedSSetup constraint nullspace mismatch", call. = FALSE)
  }

  phase1_H_dimensions <- unname(as.integer(
    fastkpc_full_cuda_prepared_s_row_value(setup_row, "H_dimensions")
  ))
  phase1_H_hash <- as.character(
    fastkpc_full_cuda_prepared_s_row_value(setup_row, "H_hash")
  )
  if (identical(phase1_H_hash, "NONE")) {
    H_clean <- is.null(setup$H) && length(phase1_H_dimensions) == 0L
  } else {
    H_clean <- is.matrix(setup$H) && is.numeric(setup$H) &&
      identical(dim(setup$H), c(p, p)) &&
      identical(dim(setup$H), phase1_H_dimensions) &&
      all(is.finite(setup$H)) &&
      identical(
        fastkpc_full_cuda_census_metadata_hash(setup$H), phase1_H_hash
      )
  }
  if (!isTRUE(H_clean)) {
    stop("PreparedSSetup H policy mismatch", call. = FALSE)
  }

  phase1_weights_policy <- as.character(
    fastkpc_full_cuda_prepared_s_row_value(setup_row, "weights_policy")
  )
  if (is.null(setup$weights)) {
    weights_clean <- identical(setup$weights_policy, "none-or-unit") &&
      identical(phase1_weights_policy, "none")
  } else {
    weights_clean <- is.numeric(setup$weights) &&
      length(setup$weights) == n && all(is.finite(setup$weights)) &&
      all(setup$weights >= 0) && !all(setup$weights == 1)
    if (isTRUE(weights_clean)) {
      expected_weights_policy <-
        fastkpc_full_cuda_prepared_s_nonneutral_policy(
          setup$weights, "nonunit"
        )
      weights_clean <-
        identical(setup$weights_policy, expected_weights_policy) &&
        identical(phase1_weights_policy, expected_weights_policy)
    }
  }
  if (!isTRUE(weights_clean)) {
    stop("PreparedSSetup weights policy mismatch", call. = FALSE)
  }
  phase1_offset_policy <- as.character(
    fastkpc_full_cuda_prepared_s_row_value(setup_row, "offset_policy")
  )
  if (is.null(setup$offset)) {
    offset_clean <- identical(setup$offset_policy, "none-or-zero") &&
      identical(phase1_offset_policy, "none")
  } else {
    offset_clean <- is.numeric(setup$offset) &&
      length(setup$offset) == n && all(is.finite(setup$offset)) &&
      !all(setup$offset == 0)
    if (isTRUE(offset_clean)) {
      expected_offset_policy <-
        fastkpc_full_cuda_prepared_s_nonneutral_policy(
          setup$offset, "nonzero"
        )
      offset_clean <-
        identical(setup$offset_policy, expected_offset_policy) &&
        identical(phase1_offset_policy, expected_offset_policy)
    }
  }
  if (!isTRUE(offset_clean)) {
    stop("PreparedSSetup offset policy mismatch", call. = FALSE)
  }

  expected_smooth <- fastkpc_full_cuda_prepared_s_expected_smooth_state(
    setup$sorted_S, setup$formula_class
  )
  smooth_count <- length(setup$smooth_classes)
  smooth_atomic_clean <-
    identical(as.integer(smooth_count), expected_smooth$count) &&
    is.character(setup$smooth_classes) &&
    is.character(setup$smooth_labels) &&
    is.character(setup$smooth_by) &&
    is.integer(setup$basis_dimensions) &&
    is.integer(setup$smooth_null_space_dimensions) &&
    is.logical(setup$smooth_side_constraints) &&
    is.logical(setup$smooth_reparameterized) &&
    all(vapply(
      list(
        setup$smooth_labels, setup$smooth_by, setup$basis_dimensions,
        setup$smooth_null_space_dimensions,
        setup$smooth_side_constraints, setup$smooth_reparameterized
      ),
      length, integer(1L)
    ) == smooth_count) &&
    !anyNA(setup$smooth_classes) && !anyNA(setup$smooth_labels) &&
    !anyNA(setup$smooth_by) &&
    all(is.finite(setup$basis_dimensions)) &&
    all(is.finite(setup$smooth_null_space_dimensions)) &&
    identical(setup$smooth_classes, expected_smooth$classes) &&
    identical(setup$smooth_labels, expected_smooth$labels) &&
    identical(setup$smooth_by, expected_smooth$by)
  if (!isTRUE(smooth_atomic_clean)) {
    stop("PreparedSSetup smooth metadata mismatch", call. = FALSE)
  }
  for (field in c(
    "smooth_terms", "smooth_p_order", "smooth_ranks",
    "smooth_S_scale", "smooth_shift"
  )) {
    fastkpc_full_cuda_prepared_s_require_named_list(
      setup[[field]], smooth_count, "smooth_", field
    )
  }
  smooth_lists_finite <- all(vapply(
    c(
      setup$smooth_p_order, setup$smooth_ranks,
      setup$smooth_S_scale, setup$smooth_shift
    ),
    function(value) is.numeric(value) && all(is.finite(value)),
    logical(1L)
  )) && all(vapply(
    setup$smooth_terms,
    function(value) is.character(value) && !anyNA(value),
    logical(1L)
  ))
  ranges_clean <- is.matrix(setup$smooth_parameter_ranges) &&
    is.integer(setup$smooth_parameter_ranges) &&
    identical(dim(setup$smooth_parameter_ranges), c(smooth_count, 2L)) &&
    is.matrix(setup$smooth_sp_ranges) &&
    is.integer(setup$smooth_sp_ranges) &&
    identical(dim(setup$smooth_sp_ranges), c(smooth_count, 2L)) &&
    all(setup$smooth_parameter_ranges[, 1L] >= 1L) &&
    all(setup$smooth_parameter_ranges[, 2L] <= p) &&
    all(setup$smooth_parameter_ranges[, 1L] <=
          setup$smooth_parameter_ranges[, 2L]) &&
    all(setup$smooth_sp_ranges[, 1L] >= 1L) &&
    all(setup$smooth_sp_ranges[, 1L] <= setup$smooth_sp_ranges[, 2L])
  if (!isTRUE(smooth_lists_finite) || !isTRUE(ranges_clean)) {
    stop("PreparedSSetup smooth ranges mismatch", call. = FALSE)
  }
  parameter_widths <- as.integer(
    setup$smooth_parameter_ranges[, 2L] -
      setup$smooth_parameter_ranges[, 1L] + 1L
  )
  smooth_sp_widths <- as.integer(
    setup$smooth_sp_ranges[, 2L] -
      setup$smooth_sp_ranges[, 1L] + 1L
  )
  descriptor_shape_clean <-
    identical(setup$smooth_terms, expected_smooth$terms) &&
    identical(unname(lengths(setup$smooth_shift)),
              unname(lengths(expected_smooth$terms))) &&
    identical(unname(lengths(setup$smooth_p_order)),
              rep(1L, smooth_count)) &&
    identical(unname(lengths(setup$smooth_ranks)), smooth_sp_widths) &&
    identical(unname(lengths(setup$smooth_S_scale)), smooth_sp_widths)
  parameter_ranges_contiguous <-
    smooth_count > 0L &&
    setup$smooth_parameter_ranges[1L, 1L] == 2L &&
    setup$smooth_parameter_ranges[smooth_count, 2L] == p &&
    all(parameter_widths > 0L) &&
    identical(parameter_widths, setup$basis_dimensions - 1L) &&
    (smooth_count == 1L || all(
      setup$smooth_parameter_ranges[-1L, 1L] ==
        setup$smooth_parameter_ranges[-smooth_count, 2L] + 1L
    ))
  sp_ranges_contiguous <-
    smooth_count > 0L && penalty_count > 0L &&
    setup$smooth_sp_ranges[1L, 1L] == 1L &&
    setup$smooth_sp_ranges[smooth_count, 2L] == penalty_count &&
    all(smooth_sp_widths > 0L) &&
    (smooth_count == 1L || all(
      setup$smooth_sp_ranges[-1L, 1L] ==
        setup$smooth_sp_ranges[-smooth_count, 2L] + 1L
    ))
  penalty_span_clean <- all(vapply(seq_len(smooth_count), function(index) {
    sp_index <- seq.int(
      setup$smooth_sp_ranges[index, 1L],
      setup$smooth_sp_ranges[index, 2L]
    )
    all(vapply(
      setup$penalty_blocks[sp_index], nrow, integer(1L)
    ) == parameter_widths[[index]]) &&
      identical(
        unname(as.integer(setup$smooth_ranks[[index]])),
        unname(as.integer(setup$penalty_ranks[sp_index]))
      )
  }, logical(1L)))
  if (!isTRUE(descriptor_shape_clean) ||
      !isTRUE(parameter_ranges_contiguous) ||
      !isTRUE(sp_ranges_contiguous) || !isTRUE(penalty_span_clean)) {
    stop("PreparedSSetup smooth metadata shape mismatch", call. = FALSE)
  }
  smooth_sp_order <- unname(as.integer(unlist(lapply(
    seq_len(smooth_count),
    function(index) seq.int(
      setup$smooth_sp_ranges[index, 1L],
      setup$smooth_sp_ranges[index, 2L]
    )
  ), use.names = FALSE)))
  if (!identical(smooth_sp_order, expected_penalty_order)) {
    stop("PreparedSSetup penalty order mismatch", call. = FALSE)
  }
  expected_offsets_from_smooths <- integer(penalty_count)
  for (index in seq_len(smooth_count)) {
    sp_index <- seq.int(
      setup$smooth_sp_ranges[index, 1L],
      setup$smooth_sp_ranges[index, 2L]
    )
    expected_offsets_from_smooths[sp_index] <-
      setup$smooth_parameter_ranges[index, 1L]
  }
  if (!identical(setup$penalty_offsets, expected_offsets_from_smooths)) {
    stop("PreparedSSetup penalty offset mismatch", call. = FALSE)
  }
  phase1_smooth_classes <- as.character(
    fastkpc_full_cuda_prepared_s_row_value(setup_row, "smooth_classes")
  )
  phase1_basis_dimensions <- unname(as.integer(
    fastkpc_full_cuda_prepared_s_row_value(setup_row, "basis_dimensions")
  ))
  if (!identical(setup$smooth_classes, unname(phase1_smooth_classes)) ||
      !identical(setup$basis_dimensions, phase1_basis_dimensions)) {
    stop("PreparedSSetup Phase 1 smooth lineage mismatch", call. = FALSE)
  }

  P_unit <- if (penalty_count == 0L) {
    matrix(0, p, p)
  } else {
    fastkpc_assemble_penalty(
      p = p, S = setup$penalty_blocks, off = setup$penalty_offsets,
      sp = rep(1, penalty_count), H = NULL
    )
  }
  projected_penalty <- if (is.null(setup$constraint_nullspace)) {
    P_unit
  } else {
    crossprod(
      setup$constraint_nullspace,
      P_unit %*% setup$constraint_nullspace
    )
  }
  projected_penalty <- fastkpc_full_cuda_prepared_s_matrix(
    projected_penalty
  )
  projected_rank <- fastkpc_full_cuda_census_svd_diagnostics(
    projected_penalty, expected_rank = ncol(projected_penalty)
  )$rank
  expected_penalty_nullity <- as.integer(
    setup$constraint_nullspace_dimension - projected_rank
  )
  phase1_penalty_nullity <- as.integer(
    fastkpc_full_cuda_prepared_s_row_value(setup_row, "penalty_nullity")
  )
  if (!identical(setup$penalty_nullity, expected_penalty_nullity) ||
      !identical(setup$penalty_nullity, phase1_penalty_nullity)) {
    stop("PreparedSSetup penalty nullity mismatch", call. = FALSE)
  }

  phase1_conditioning_rank <- as.integer(
    fastkpc_full_cuda_prepared_s_row_value(setup_row, "conditioning_rank")
  )
  phase1_conditioning_condition <- as.numeric(
    fastkpc_full_cuda_prepared_s_row_value(
      setup_row, "conditioning_condition"
    )
  )
  if (!identical(setup$conditioning_rank, phase1_conditioning_rank) ||
      !fastkpc_full_cuda_prepared_s_condition_compatible(
        setup$conditioning_condition, phase1_conditioning_condition,
        tolerance = model_condition_tolerance
      )) {
    stop("PreparedSSetup conditioning diagnostics mismatch", call. = FALSE)
  }

  X_weighted <- if (is.null(setup$weights)) {
    if (!identical(setup$weighted_X_policy, "alias-X-unit-weights")) {
      stop("PreparedSSetup weighted X policy mismatch", call. = FALSE)
    }
    X
  } else {
    if (!identical(
          setup$weighted_X_policy, "sqrt-weights-row-scaled"
        )) {
      stop("PreparedSSetup weighted X policy mismatch", call. = FALSE)
    }
    X * sqrt(setup$weights)
  }
  expected_gram <- fastkpc_full_cuda_prepared_s_matrix(
    crossprod(X_weighted)
  )
  fastkpc_full_cuda_prepared_s_require_matrix(
    setup$gram_matrix, c(p, p), "gram matrix"
  )
  if (!identical(
        fastkpc_full_cuda_prepared_s_plain_hash(setup$gram_matrix),
        fastkpc_full_cuda_prepared_s_plain_hash(expected_gram)
      )) {
    stop("PreparedSSetup gram matrix mismatch", call. = FALSE)
  }
  if (identical(setup$constraint_mode, "identity")) {
    null_gram_clean <-
      identical(setup$nullspace_gram_policy, "alias-gram") &&
      is.null(setup$nullspace_gram_matrix)
  } else {
    expected_null_gram <- fastkpc_full_cuda_prepared_s_matrix(
      crossprod(X_weighted %*% setup$constraint_nullspace)
    )
    null_gram_clean <-
      identical(
        setup$nullspace_gram_policy, "explicit-nullspace-gram"
      ) &&
      is.matrix(setup$nullspace_gram_matrix) &&
      all(is.finite(setup$nullspace_gram_matrix)) &&
      identical(
        fastkpc_full_cuda_prepared_s_plain_hash(
          setup$nullspace_gram_matrix
        ),
        fastkpc_full_cuda_prepared_s_plain_hash(expected_null_gram)
      )
  }
  if (!isTRUE(null_gram_clean)) {
    stop("PreparedSSetup nullspace gram mismatch", call. = FALSE)
  }

  scoring_clean <- identical(setup$scoring_n, as.integer(n)) &&
    length(setup$scoring_n_true) == 1L &&
    is.integer(setup$scoring_n_true) &&
    length(setup$scoring_min_edf) == 1L &&
    is.finite(setup$scoring_min_edf) &&
    length(setup$scoring_pearson_extra) == 1L &&
    is.finite(setup$scoring_pearson_extra) &&
    length(setup$scoring_deviance_extra) == 1L &&
    is.finite(setup$scoring_deviance_extra)
  if (!isTRUE(scoring_clean)) {
    stop("PreparedSSetup scoring constants mismatch", call. = FALSE)
  }

  expected_semantic <-
    fastkpc_full_cuda_prepared_s_semantic_fingerprint(setup)
  if (!identical(setup$semantic_fingerprint, expected_semantic)) {
    stop("PreparedSSetup semantic fingerprint mismatch", call. = FALSE)
  }
  expected_representation <-
    fastkpc_full_cuda_prepared_s_representation_fingerprint(setup)
  if (!identical(
        setup$representation_fingerprint, expected_representation
      )) {
    stop("PreparedSSetup representation fingerprint mismatch", call. = FALSE)
  }
  invisible(TRUE)
}

fastkpc_full_cuda_target_state_schema_version <- function() {
  "full-cuda-ci-target-state-v1"
}

fastkpc_full_cuda_target_state_field_names <- function() {
  c(
    "schema_version", "residual_key_payload", "residual_key_sha256",
    "prepared_s_key_sha256", "same_S_group_id",
    "phase1_setup_fingerprint", "target", "y_source", "y_hash",
    "projected_rhs", "nullspace_projected_rhs", "selected_sp",
    "selected_sp_names", "selected_sp_hash", "GCV_Cp_score", "EDF",
    "convergence_fields", "warning_classes", "warning_messages",
    "coefficient_rank", "coefficient_hash", "fitted_hash",
    "residual_hash", "target_fit_fingerprint",
    "target_state_fingerprint"
  )
}

fastkpc_full_cuda_target_state_fingerprint_field_names <- function(
    schema_version = fastkpc_full_cuda_target_state_schema_version()) {
  if (!identical(
        schema_version, fastkpc_full_cuda_target_state_schema_version()
      )) {
    stop("TargetState fingerprint schema is unsupported", call. = FALSE)
  }
  setdiff(
    fastkpc_full_cuda_target_state_field_names(),
    "target_state_fingerprint"
  )
}

fastkpc_full_cuda_target_state_phase1_field_names <- function() {
  c(
    "selected_sp", "selected_sp_names", "selected_sp_hash",
    "GCV_Cp_score", "EDF", "convergence_fields", "warning_classes",
    "warning_messages", "coefficient_rank", "coefficient_hash",
    "fitted_hash", "residual_hash", "target_fit_fingerprint"
  )
}

fastkpc_full_cuda_target_state_require <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
  invisible(TRUE)
}

fastkpc_full_cuda_target_state_data_identity_valid <- function(
    data, dataset_sha256) {
  is.matrix(data) && is.double(data) && length(dim(data)) == 2L &&
    nrow(data) > 0L && ncol(data) > 0L && all(is.finite(data)) &&
    fastkpc_full_cuda_prepared_s_is_sha256(dataset_sha256) &&
    identical(fastkpc_full_cuda_data_hash(data), dataset_sha256)
}

fastkpc_full_cuda_target_state_row_value <- function(state_row, field) {
  if (!is.data.frame(state_row) || nrow(state_row) != 1L ||
      !field %in% names(state_row)) {
    stop("TargetState fingerprint row is malformed", call. = FALSE)
  }
  state_row[[field]][[1L]]
}

fastkpc_full_cuda_target_state_fingerprint <- function(state_row) {
  if (!is.data.frame(state_row) || nrow(state_row) != 1L ||
      !identical(
        names(state_row), fastkpc_full_cuda_target_state_field_names()
      )) {
    stop("TargetState fingerprint row is malformed", call. = FALSE)
  }
  schema_version <- fastkpc_full_cuda_target_state_row_value(
    state_row, "schema_version"
  )
  fields <- fastkpc_full_cuda_target_state_fingerprint_field_names(
    schema_version
  )
  values <- setNames(lapply(fields, function(field) {
    fastkpc_full_cuda_target_state_row_value(state_row, field)
  }), fields)
  fastkpc_full_cuda_prepared_s_field_hash(values, fields)
}

fastkpc_full_cuda_target_state_context <- function(
    inputs, prepared_setup) {
  required_inputs <- c(
    "data", "dataset_sha256", "same_s_setup_metadata",
    "residual_requests", "target_fit_metadata"
  )
  fastkpc_full_cuda_target_state_require(
    is.list(inputs) &&
      length(setdiff(required_inputs, names(inputs))) == 0L,
    "TargetState inputs are incomplete"
  )
  data <- inputs$data
  dataset_sha256 <- inputs$dataset_sha256
  fastkpc_full_cuda_target_state_require(
    fastkpc_full_cuda_target_state_data_identity_valid(
      data, dataset_sha256
    ),
    "TargetState dataset identity mismatch"
  )
  fastkpc_full_cuda_target_state_require(
    is.list(prepared_setup) &&
      fastkpc_full_cuda_prepared_s_is_sha256(
        prepared_setup$same_S_group_id
      ),
    "TargetState PreparedSSetup lineage is invalid"
  )
  group_id <- as.character(prepared_setup$same_S_group_id)

  setup_rows <- as.data.frame(
    inputs$same_s_setup_metadata, stringsAsFactors = FALSE
  )
  fastkpc_full_cuda_target_state_require(
    "same_S_group_id" %in% names(setup_rows) &&
      !anyNA(setup_rows$same_S_group_id),
    "TargetState canonical same-S metadata is incomplete"
  )
  setup_index <- which(as.character(setup_rows$same_S_group_id) == group_id)
  fastkpc_full_cuda_target_state_require(
    length(setup_index) == 1L,
    "TargetState canonical same-S row is not unique"
  )
  setup_row <- setup_rows[setup_index, , drop = FALSE]
  fastkpc_full_cuda_validate_prepared_s_setup(
    setup = prepared_setup,
    setup_row = setup_row,
    dataset_sha256 = dataset_sha256
  )
  fastkpc_full_cuda_target_state_require(
    nrow(prepared_setup$X) == nrow(data),
    "TargetState PreparedSSetup data dimensions mismatch"
  )

  requests <- as.data.frame(
    inputs$residual_requests, stringsAsFactors = FALSE
  )
  request_fields <- c(
    "residual_key_payload", "residual_key_sha256", "target", "S_key",
    "S_size", "formula_class", "same_S_group_id"
  )
  fastkpc_full_cuda_target_state_require(
    length(setdiff(request_fields, names(requests))) == 0L &&
      !anyNA(requests[c(
        "residual_key_payload", "residual_key_sha256", "target", "S_key",
        "S_size", "formula_class", "same_S_group_id"
      )]),
    "TargetState canonical residual requests are incomplete"
  )
  requests$residual_key_sha256 <- as.character(
    requests$residual_key_sha256
  )
  fastkpc_full_cuda_target_state_require(
    all(vapply(
      requests$residual_key_sha256,
      fastkpc_full_cuda_prepared_s_is_sha256,
      logical(1L)
    )),
    "TargetState canonical residual request key is invalid"
  )

  target_rows <- as.data.frame(
    inputs$target_fit_metadata, stringsAsFactors = FALSE
  )
  target_fields <- c(
    "residual_key_sha256", "same_S_group_id", "setup_fingerprint",
    "target", "fit_status", "fit_error",
    fastkpc_full_cuda_target_state_phase1_field_names()
  )
  fastkpc_full_cuda_target_state_require(
    length(setdiff(target_fields, names(target_rows))) == 0L &&
      !anyNA(target_rows[c(
        "residual_key_sha256", "same_S_group_id", "setup_fingerprint",
        "target", "fit_status", "fit_error"
      )]),
    "TargetState canonical target metadata is incomplete"
  )
  target_rows$residual_key_sha256 <- as.character(
    target_rows$residual_key_sha256
  )
  fastkpc_full_cuda_target_state_require(
    all(vapply(
      target_rows$residual_key_sha256,
      fastkpc_full_cuda_prepared_s_is_sha256,
      logical(1L)
    )),
    "TargetState canonical target metadata key is invalid"
  )

  request_rows <- requests[
    as.character(requests$same_S_group_id) == group_id, , drop = FALSE
  ]
  target_rows <- target_rows[
    as.character(target_rows$same_S_group_id) == group_id, , drop = FALSE
  ]
  fastkpc_full_cuda_target_state_require(
    nrow(request_rows) > 0L && nrow(target_rows) > 0L,
    "TargetState canonical same-S group has no targets"
  )
  request_keys <- as.character(request_rows$residual_key_sha256)
  target_keys <- as.character(target_rows$residual_key_sha256)
  request_rows <- request_rows[
    order(request_keys, method = "radix"),
    , drop = FALSE
  ]
  target_rows <- target_rows[
    order(target_keys, method = "radix"),
    , drop = FALSE
  ]
  rownames(request_rows) <- NULL
  rownames(target_rows) <- NULL

  request_keys <- as.character(request_rows$residual_key_sha256)
  target_keys <- as.character(target_rows$residual_key_sha256)
  fastkpc_full_cuda_target_state_require(
    !anyDuplicated(request_keys) && !anyDuplicated(target_keys) &&
      identical(request_keys, target_keys),
    "TargetState canonical residual key set mismatch"
  )
  fastkpc_full_cuda_target_state_require(
    is.integer(request_rows$target) && is.integer(target_rows$target) &&
      identical(
        as.integer(request_rows$target), as.integer(target_rows$target)
      ) &&
      all(target_rows$target >= 1L & target_rows$target <= ncol(data)),
    "TargetState canonical target index mismatch"
  )
  fastkpc_full_cuda_target_state_require(
    all(target_rows$fit_status == "success") &&
      all(target_rows$fit_error == "NONE"),
    "TargetState requires successful Phase 1 target fits"
  )
  fastkpc_full_cuda_target_state_require(
    identical(
      as.character(target_rows$same_S_group_id),
      rep(group_id, nrow(target_rows))
    ) &&
      identical(
        as.character(target_rows$setup_fingerprint),
        rep(
          as.character(prepared_setup$phase1_setup_fingerprint),
          nrow(target_rows)
        )
      ),
    "TargetState Phase 1 target lineage mismatch"
  )
  setup_S_key <- as.character(
    fastkpc_full_cuda_prepared_s_row_value(setup_row, "S_key")
  )
  setup_S_size <- as.integer(
    fastkpc_full_cuda_prepared_s_row_value(setup_row, "S_size")
  )
  setup_formula <- as.character(
    fastkpc_full_cuda_prepared_s_row_value(setup_row, "formula_class")
  )
  fastkpc_full_cuda_target_state_require(
    identical(
      as.character(request_rows$same_S_group_id),
      rep(group_id, nrow(request_rows))
    ) &&
      identical(
        as.character(request_rows$S_key),
        rep(setup_S_key, nrow(request_rows))
      ) &&
      identical(
        as.integer(request_rows$S_size),
        rep(setup_S_size, nrow(request_rows))
      ) &&
      identical(
        as.character(request_rows$formula_class),
        rep(setup_formula, nrow(request_rows))
      ),
    "TargetState residual request lineage mismatch"
  )
  representative_key <- as.character(
    fastkpc_full_cuda_prepared_s_row_value(
      setup_row, "representative_residual_key_sha256"
    )
  )
  fastkpc_full_cuda_target_state_require(
    representative_key %in% request_keys,
    "TargetState setup representative lineage mismatch"
  )

  S <- fastkpc_full_cuda_census_parse_s(setup_S_key)
  expected_payloads <- vapply(seq_len(nrow(request_rows)), function(index) {
    fastkpc_full_cuda_census_residual_payload(
      target = request_rows$target[[index]],
      S = S,
      formula_class = setup_formula,
      data_hash = dataset_sha256,
      n = nrow(data),
      p = ncol(data)
    )
  }, character(1L))
  expected_hashes <- unname(vapply(
    expected_payloads, fastkpc_full_cuda_census_hash_utf8, character(1L)
  ))
  fastkpc_full_cuda_target_state_require(
    identical(
      as.character(request_rows$residual_key_payload), expected_payloads
    ) && identical(request_keys, expected_hashes),
    "TargetState residual key serialization mismatch"
  )

  list(
    data = data,
    dataset_sha256 = as.character(dataset_sha256),
    setup = prepared_setup,
    setup_row = setup_row,
    request_rows = request_rows,
    target_rows = target_rows
  )
}

fastkpc_full_cuda_target_state_projected_rhs <- function(context) {
  targets <- as.integer(context$target_rows$target)
  Y <- context$data[, targets, drop = FALSE]
  projected <- if (is.null(context$setup$weights)) {
    crossprod(context$setup$X, Y)
  } else {
    crossprod(
      context$setup$X,
      Y * as.numeric(context$setup$weights)
    )
  }
  null_projected <- if (identical(
    context$setup$constraint_mode, "identity"
  )) {
    projected
  } else {
    crossprod(context$setup$constraint_nullspace, projected)
  }
  list(Y = Y, projected = projected, null_projected = null_projected)
}

fastkpc_full_cuda_build_target_states <- function(
    inputs, prepared_setup) {
  context <- fastkpc_full_cuda_target_state_context(inputs, prepared_setup)
  algebra <- fastkpc_full_cuda_target_state_projected_rhs(context)
  target_rows <- context$target_rows
  request_rows <- context$request_rows
  row_count <- nrow(target_rows)
  list_column <- function(field) {
    lapply(seq_len(row_count), function(index) {
      target_rows[[field]][[index]]
    })
  }
  states <- data.frame(
    schema_version = rep(
      fastkpc_full_cuda_target_state_schema_version(), row_count
    ),
    residual_key_payload = as.character(
      request_rows$residual_key_payload
    ),
    residual_key_sha256 = as.character(
      request_rows$residual_key_sha256
    ),
    prepared_s_key_sha256 = rep(
      as.character(prepared_setup$prepared_s_key_sha256), row_count
    ),
    same_S_group_id = rep(
      as.character(prepared_setup$same_S_group_id), row_count
    ),
    phase1_setup_fingerprint = rep(
      as.character(prepared_setup$phase1_setup_fingerprint), row_count
    ),
    target = as.integer(target_rows$target),
    y_source = I(lapply(seq_len(row_count), function(index) {
      list(
        dataset_sha256 = context$dataset_sha256,
        target_column = as.integer(target_rows$target[[index]])
      )
    })),
    y_hash = vapply(seq_len(row_count), function(index) {
      fastkpc_full_cuda_census_metadata_hash(
        as.numeric(algebra$Y[, index])
      )
    }, character(1L)),
    projected_rhs = I(lapply(seq_len(row_count), function(index) {
      as.numeric(algebra$projected[, index])
    })),
    nullspace_projected_rhs = I(lapply(
      seq_len(row_count), function(index) {
        as.numeric(algebra$null_projected[, index])
      }
    )),
    selected_sp = I(list_column("selected_sp")),
    selected_sp_names = I(list_column("selected_sp_names")),
    selected_sp_hash = as.character(target_rows$selected_sp_hash),
    GCV_Cp_score = as.numeric(target_rows$GCV_Cp_score),
    EDF = as.numeric(target_rows$EDF),
    convergence_fields = I(list_column("convergence_fields")),
    warning_classes = I(list_column("warning_classes")),
    warning_messages = I(list_column("warning_messages")),
    coefficient_rank = as.integer(target_rows$coefficient_rank),
    coefficient_hash = as.character(target_rows$coefficient_hash),
    fitted_hash = as.character(target_rows$fitted_hash),
    residual_hash = as.character(target_rows$residual_hash),
    target_fit_fingerprint = as.character(
      target_rows$target_fit_fingerprint
    ),
    target_state_fingerprint = rep(NA_character_, row_count),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  states$target_state_fingerprint <- vapply(
    seq_len(row_count), function(index) {
      fastkpc_full_cuda_target_state_fingerprint(
        states[index, , drop = FALSE]
      )
    }, character(1L)
  )
  fastkpc_full_cuda_validate_target_states(
    states = states,
    inputs = inputs,
    prepared_setup = prepared_setup
  )
  states
}

fastkpc_full_cuda_validate_target_states <- function(
    states, inputs, prepared_setup, ...) {
  dots <- list(...)
  fastkpc_full_cuda_target_state_require(
    length(dots) == 0L,
    "TargetState validation options are unsupported"
  )
  context <- fastkpc_full_cuda_target_state_context(inputs, prepared_setup)
  expected_fields <- fastkpc_full_cuda_target_state_field_names()
  fastkpc_full_cuda_target_state_require(
    is.data.frame(states) && identical(names(states), expected_fields),
    "TargetState schema fields mismatch"
  )
  row_count <- nrow(context$target_rows)
  fastkpc_full_cuda_target_state_require(
    nrow(states) == row_count,
    "TargetState residual key order mismatch"
  )
  list_fields <- c(
    "y_source", "projected_rhs", "nullspace_projected_rhs",
    "selected_sp", "selected_sp_names", "convergence_fields",
    "warning_classes", "warning_messages"
  )
  fastkpc_full_cuda_target_state_require(
    all(vapply(states[list_fields], is.list, logical(1L))),
    "TargetState list-column schema mismatch"
  )
  fastkpc_full_cuda_target_state_require(
    is.character(states$schema_version) &&
      identical(
        states$schema_version,
        rep(fastkpc_full_cuda_target_state_schema_version(), row_count)
      ),
    "TargetState schema version mismatch"
  )
  fastkpc_full_cuda_target_state_require(
    is.character(states$residual_key_sha256) &&
      identical(
        states$residual_key_sha256,
        as.character(context$request_rows$residual_key_sha256)
      ),
    "TargetState residual key order mismatch"
  )
  fastkpc_full_cuda_target_state_require(
    is.character(states$prepared_s_key_sha256) &&
      identical(
        states$prepared_s_key_sha256,
        rep(as.character(prepared_setup$prepared_s_key_sha256), row_count)
      ),
    "TargetState PreparedSKey mismatch"
  )
  fastkpc_full_cuda_target_state_require(
    is.character(states$same_S_group_id) &&
      identical(
        states$same_S_group_id,
        rep(as.character(prepared_setup$same_S_group_id), row_count)
      ),
    "TargetState group lineage mismatch"
  )
  fastkpc_full_cuda_target_state_require(
    is.character(states$phase1_setup_fingerprint) &&
      identical(
        states$phase1_setup_fingerprint,
        rep(
          as.character(prepared_setup$phase1_setup_fingerprint),
          row_count
        )
      ),
    "TargetState setup lineage mismatch"
  )
  fastkpc_full_cuda_target_state_require(
    is.character(states$residual_key_payload) &&
      identical(
        states$residual_key_payload,
        as.character(context$request_rows$residual_key_payload)
      ) &&
      all(endsWith(states$residual_key_payload, "\n")) &&
      identical(
        unname(vapply(
          states$residual_key_payload,
          fastkpc_full_cuda_census_hash_utf8,
          character(1L)
        )),
        states$residual_key_sha256
      ),
    "TargetState residual key serialization mismatch"
  )
  fastkpc_full_cuda_target_state_require(
    is.integer(states$target) &&
      identical(states$target, as.integer(context$target_rows$target)) &&
      all(states$target >= 1L & states$target <= ncol(context$data)),
    "TargetState target index mismatch"
  )

  fingerprint_fields <-
    fastkpc_full_cuda_target_state_fingerprint_field_names()
  nonfinite <- vapply(seq_len(row_count), function(index) {
    any(vapply(fingerprint_fields, function(field) {
      fastkpc_full_cuda_census_value_has_nonfinite(
        fastkpc_full_cuda_target_state_row_value(
          states[index, , drop = FALSE], field
        )
      )
    }, logical(1L)))
  }, logical(1L))
  fastkpc_full_cuda_target_state_require(
    !any(nonfinite),
    "TargetState payload must be finite"
  )

  algebra <- fastkpc_full_cuda_target_state_projected_rhs(context)
  phase1_rows <- context$target_rows
  hash_fields <- c(
    "y_hash", "selected_sp_hash", "coefficient_hash", "fitted_hash",
    "residual_hash", "target_fit_fingerprint",
    "target_state_fingerprint"
  )
  fastkpc_full_cuda_target_state_require(
    all(vapply(states[hash_fields], function(value) {
      is.character(value) && !anyNA(value) &&
        all(grepl("^[0-9a-f]{64}$", value))
    }, logical(1L))),
    "TargetState hash field is invalid"
  )
  fastkpc_full_cuda_target_state_require(
    is.numeric(states$GCV_Cp_score) &&
      is.numeric(states$EDF) &&
      is.integer(states$coefficient_rank) &&
      all(is.finite(states$GCV_Cp_score)) &&
      all(is.finite(states$EDF)) &&
      all(is.finite(states$coefficient_rank)),
    "TargetState scalar metadata is invalid"
  )

  for (index in seq_len(row_count)) {
    target <- states$target[[index]]
    expected_source <- list(
      dataset_sha256 = context$dataset_sha256,
      target_column = target
    )
    source <- states$y_source[[index]]
    fastkpc_full_cuda_target_state_require(
      is.list(source) && is.null(attr(source, "class", exact = TRUE)) &&
        identical(source, expected_source),
      "TargetState y_source mismatch"
    )
    expected_y_hash <- fastkpc_full_cuda_census_metadata_hash(
      as.numeric(algebra$Y[, index])
    )
    fastkpc_full_cuda_target_state_require(
      identical(states$y_hash[[index]], expected_y_hash),
      "TargetState y hash mismatch"
    )

    selected_sp <- states$selected_sp[[index]]
    selected_sp_names <- states$selected_sp_names[[index]]
    fastkpc_full_cuda_target_state_require(
      is.numeric(selected_sp) &&
        length(selected_sp) == length(prepared_setup$penalty_blocks) &&
        all(is.finite(selected_sp)) && all(selected_sp > 0) &&
        is.character(selected_sp_names) &&
        length(selected_sp_names) == length(selected_sp),
      "TargetState selected sp is invalid"
    )
    fastkpc_full_cuda_target_state_require(
      identical(
        states$selected_sp_hash[[index]],
        fastkpc_full_cuda_census_metadata_hash(selected_sp)
      ),
      "TargetState selected sp hash mismatch"
    )
    fastkpc_full_cuda_target_state_require(
      identical(selected_sp, phase1_rows$selected_sp[[index]]) &&
        identical(
          selected_sp_names, phase1_rows$selected_sp_names[[index]]
        ),
      "TargetState selected sp mismatch"
    )

    projected_rhs <- states$projected_rhs[[index]]
    null_projected_rhs <- states$nullspace_projected_rhs[[index]]
    fastkpc_full_cuda_target_state_require(
      is.numeric(projected_rhs) &&
        length(projected_rhs) == ncol(prepared_setup$X) &&
        all(is.finite(projected_rhs)) &&
        identical(
          projected_rhs, as.numeric(algebra$projected[, index])
        ),
      "TargetState projected RHS mismatch"
    )
    fastkpc_full_cuda_target_state_require(
      is.numeric(null_projected_rhs) &&
        length(null_projected_rhs) ==
          prepared_setup$constraint_nullspace_dimension &&
        all(is.finite(null_projected_rhs)) &&
        identical(
          null_projected_rhs,
          as.numeric(algebra$null_projected[, index])
        ),
      "TargetState nullspace projected RHS mismatch"
    )

    phase1_fields <- setdiff(
      fastkpc_full_cuda_target_state_phase1_field_names(),
      c("selected_sp", "selected_sp_names", "selected_sp_hash")
    )
    for (field in phase1_fields) {
      actual <- fastkpc_full_cuda_target_state_row_value(
        states[index, , drop = FALSE], field
      )
      expected <- phase1_rows[[field]][[index]]
      fastkpc_full_cuda_target_state_require(
        identical(actual, expected),
        paste0("TargetState Phase 1 metadata mismatch: ", field)
      )
    }
    fastkpc_full_cuda_target_state_require(
      identical(
        states$selected_sp_hash[[index]],
        phase1_rows$selected_sp_hash[[index]]
      ),
      "TargetState selected sp hash mismatch"
    )
    fastkpc_full_cuda_target_state_require(
      identical(
        states$target_state_fingerprint[[index]],
        fastkpc_full_cuda_target_state_fingerprint(
          states[index, , drop = FALSE]
        )
      ),
      "TargetState fingerprint mismatch"
    )
  }
  invisible(TRUE)
}

fastkpc_full_cuda_materialize_target_state <- function(
    state_row, data, dataset_sha256) {
  fastkpc_full_cuda_target_state_require(
    is.data.frame(state_row) && nrow(state_row) == 1L,
    "TargetState requires exactly one row"
  )
  fastkpc_full_cuda_target_state_require(
    identical(
      names(state_row), fastkpc_full_cuda_target_state_field_names()
    ) &&
      identical(
        state_row$schema_version[[1L]],
        fastkpc_full_cuda_target_state_schema_version()
      ),
    "TargetState schema is malformed"
  )
  fastkpc_full_cuda_target_state_require(
    fastkpc_full_cuda_target_state_data_identity_valid(
      data, dataset_sha256
    ),
    "TargetState dataset identity mismatch"
  )
  source <- state_row$y_source[[1L]]
  fastkpc_full_cuda_target_state_require(
    is.list(source) && is.null(attr(source, "class", exact = TRUE)) &&
      identical(names(source), c("dataset_sha256", "target_column")) &&
      fastkpc_full_cuda_prepared_s_is_sha256(source$dataset_sha256) &&
      identical(source$dataset_sha256, as.character(dataset_sha256)),
    "TargetState dataset lineage mismatch"
  )
  target <- state_row$target[[1L]]
  fastkpc_full_cuda_target_state_require(
    is.integer(target) && length(target) == 1L && !is.na(target) &&
      target >= 1L && target <= ncol(data),
    "TargetState target index is invalid"
  )
  fastkpc_full_cuda_target_state_require(
    identical(source$target_column, target),
    "TargetState y_source identity mismatch"
  )
  y <- as.numeric(data[, target])
  y_hash <- state_row$y_hash[[1L]]
  fastkpc_full_cuda_target_state_require(
    fastkpc_full_cuda_prepared_s_is_sha256(y_hash) &&
      identical(
        y_hash, fastkpc_full_cuda_census_metadata_hash(y)
      ),
    "TargetState y hash mismatch"
  )
  fastkpc_full_cuda_target_state_require(
    fastkpc_full_cuda_prepared_s_is_sha256(
      state_row$target_state_fingerprint[[1L]]
    ) &&
      identical(
        state_row$target_state_fingerprint[[1L]],
        fastkpc_full_cuda_target_state_fingerprint(state_row)
      ),
    "TargetState fingerprint mismatch"
  )
  list(row = state_row, y = y)
}

fastkpc_full_cuda_prepared_s_adapter_setup_row <- function(setup) {
  expected_fields <- fastkpc_full_cuda_prepared_s_setup_field_names()
  if (!is.list(setup) || !identical(names(setup), expected_fields)) {
    stop("PreparedSSetup schema field mismatch", call. = FALSE)
  }
  penalty_dimensions <- unname(vapply(
    setup$penalty_blocks,
    function(block) paste(dim(block), collapse = "x"),
    character(1L)
  ))
  penalty_hashes <- unname(vapply(
    setup$penalty_blocks,
    fastkpc_full_cuda_census_metadata_hash,
    character(1L)
  ))
  constraint_dimensions <- unname(as.integer(dim(setup$constraint)))
  constraint_hash <- fastkpc_full_cuda_census_metadata_hash(
    setup$constraint
  )
  H_dimensions <- if (is.null(setup$H)) {
    integer()
  } else {
    unname(as.integer(dim(setup$H)))
  }
  H_hash <- if (is.null(setup$H)) {
    "NONE"
  } else {
    fastkpc_full_cuda_census_metadata_hash(setup$H)
  }
  row <- data.frame(
    same_S_group_id = as.character(setup$same_S_group_id),
    S_key = paste(as.integer(setup$sorted_S), collapse = "|"),
    S_size = as.integer(length(setup$sorted_S)),
    formula_class = as.character(setup$formula_class),
    formula_semantics_version = as.character(
      setup$formula_semantics_version
    ),
    model_matrix_nrow = as.integer(nrow(setup$X)),
    model_matrix_ncol = as.integer(ncol(setup$X)),
    model_matrix_hash = as.character(setup$phase1_model_matrix_hash),
    model_matrix_rank = as.integer(setup$model_matrix_rank),
    model_matrix_condition = as.numeric(setup$model_matrix_condition),
    penalty_count = as.integer(length(setup$penalty_blocks)),
    penalty_nullity = as.integer(setup$penalty_nullity),
    constraint_rank = as.integer(setup$constraint_rank),
    constraint_nullspace_dimension = as.integer(
      setup$constraint_nullspace_dimension
    ),
    constraint_hash = as.character(constraint_hash),
    H_hash = as.character(H_hash),
    weights_policy = if (is.null(setup$weights)) {
      "none"
    } else {
      as.character(setup$weights_policy)
    },
    offset_policy = if (is.null(setup$offset)) {
      "none"
    } else {
      as.character(setup$offset_policy)
    },
    conditioning_rank = as.integer(setup$conditioning_rank),
    conditioning_condition = as.numeric(setup$conditioning_condition),
    setup_fingerprint = as.character(setup$phase1_setup_fingerprint),
    mgcv_version = as.character(setup$mgcv_version),
    R_version = as.character(setup$R_version),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  row$penalty_block_dimensions <- I(list(penalty_dimensions))
  row$penalty_ranks <- I(list(unname(as.integer(setup$penalty_ranks))))
  row$penalty_offsets <- I(list(unname(as.integer(setup$penalty_offsets))))
  row$penalty_hashes <- I(list(penalty_hashes))
  row$constraint_dimensions <- I(list(constraint_dimensions))
  row$H_dimensions <- I(list(H_dimensions))
  row$smooth_classes <- I(list(unname(as.character(
    setup$smooth_classes
  ))))
  row$basis_dimensions <- I(list(unname(as.integer(
    setup$basis_dimensions
  ))))
  row
}

fastkpc_full_cuda_validate_prepared_s_for_adapter <- function(setup) {
  setup_row <- fastkpc_full_cuda_prepared_s_adapter_setup_row(setup)
  fastkpc_full_cuda_validate_prepared_s_setup(
    setup = setup,
    setup_row = setup_row,
    dataset_sha256 = setup$dataset_sha256
  )
  invisible(TRUE)
}

fastkpc_full_cuda_prepared_s_payload_integer <- function(payload, field) {
  if (!is.character(payload) || length(payload) != 1L || is.na(payload) ||
      !endsWith(payload, "\n")) {
    stop("TargetState residual key serialization mismatch", call. = FALSE)
  }
  lines <- strsplit(
    substring(payload, 1L, nchar(payload) - 1L),
    "\n", fixed = TRUE
  )[[1L]]
  prefix <- paste0(field, "=")
  matches <- lines[startsWith(lines, prefix)]
  if (length(matches) != 1L) {
    stop("TargetState residual key serialization mismatch", call. = FALSE)
  }
  text <- substring(matches, nchar(prefix) + 1L)
  value <- suppressWarnings(as.integer(text))
  if (length(value) != 1L || is.na(value) || value < 1L ||
      !identical(text, as.character(value))) {
    stop("TargetState residual key serialization mismatch", call. = FALSE)
  }
  value
}

fastkpc_full_cuda_validate_target_state_row_for_prepared <- function(
    prepared_setup, state_row) {
  expected_fields <- fastkpc_full_cuda_target_state_field_names()
  fastkpc_full_cuda_target_state_require(
    is.data.frame(state_row) && nrow(state_row) == 1L &&
      identical(names(state_row), expected_fields),
    "TargetState schema fields mismatch"
  )
  fastkpc_full_cuda_target_state_require(
    identical(
      state_row$schema_version[[1L]],
      fastkpc_full_cuda_target_state_schema_version()
    ),
    "TargetState schema version mismatch"
  )
  list_fields <- c(
    "y_source", "projected_rhs", "nullspace_projected_rhs",
    "selected_sp", "selected_sp_names", "convergence_fields",
    "warning_classes", "warning_messages"
  )
  fastkpc_full_cuda_target_state_require(
    all(vapply(state_row[list_fields], is.list, logical(1L))),
    "TargetState list-column schema mismatch"
  )
  hash_fields <- c(
    "residual_key_sha256", "prepared_s_key_sha256",
    "same_S_group_id", "phase1_setup_fingerprint", "y_hash",
    "selected_sp_hash", "coefficient_hash", "fitted_hash",
    "residual_hash", "target_fit_fingerprint",
    "target_state_fingerprint"
  )
  fastkpc_full_cuda_target_state_require(
    all(vapply(state_row[hash_fields], function(value) {
      is.character(value) && length(value) == 1L &&
        fastkpc_full_cuda_prepared_s_is_sha256(value[[1L]])
    }, logical(1L))),
    "TargetState hash field is invalid"
  )
  fastkpc_full_cuda_target_state_require(
    identical(
      state_row$target_state_fingerprint[[1L]],
      fastkpc_full_cuda_target_state_fingerprint(state_row)
    ),
    "TargetState fingerprint mismatch"
  )
  fastkpc_full_cuda_target_state_require(
    identical(
      state_row$prepared_s_key_sha256[[1L]],
      prepared_setup$prepared_s_key_sha256
    ),
    "TargetState PreparedSKey mismatch"
  )
  fastkpc_full_cuda_target_state_require(
    identical(
      state_row$same_S_group_id[[1L]],
      prepared_setup$same_S_group_id
    ),
    "TargetState group lineage mismatch"
  )
  fastkpc_full_cuda_target_state_require(
    identical(
      state_row$phase1_setup_fingerprint[[1L]],
      prepared_setup$phase1_setup_fingerprint
    ),
    "TargetState setup lineage mismatch"
  )

  target <- state_row$target[[1L]]
  fastkpc_full_cuda_target_state_require(
    is.integer(target) && length(target) == 1L && !is.na(target) &&
      target >= 1L,
    "TargetState target index is invalid"
  )
  source <- state_row$y_source[[1L]]
  fastkpc_full_cuda_target_state_require(
    is.list(source) && is.null(attr(source, "class", exact = TRUE)) &&
      identical(names(source), c("dataset_sha256", "target_column")) &&
      identical(source$dataset_sha256, prepared_setup$dataset_sha256) &&
      identical(source$target_column, target),
    "TargetState y_source mismatch"
  )

  payload <- state_row$residual_key_payload[[1L]]
  dataset_p <- fastkpc_full_cuda_prepared_s_payload_integer(payload, "p")
  fastkpc_full_cuda_target_state_require(
    dataset_p >= target &&
      dataset_p >= max(as.integer(prepared_setup$sorted_S)),
    "TargetState residual key serialization mismatch"
  )
  expected_payload <- fastkpc_full_cuda_census_residual_payload(
    target = target,
    S = prepared_setup$sorted_S,
    formula_class = prepared_setup$formula_class,
    data_hash = prepared_setup$dataset_sha256,
    n = nrow(prepared_setup$X),
    p = dataset_p
  )
  expected_key <- fastkpc_full_cuda_census_hash_utf8(expected_payload)
  fastkpc_full_cuda_target_state_require(
    identical(payload, expected_payload) &&
      identical(state_row$residual_key_sha256[[1L]], expected_key),
    "TargetState residual key serialization mismatch"
  )
  expected_group_id <- fastkpc_full_cuda_census_hash_utf8(
    fastkpc_full_cuda_census_same_s_payload(
      S = prepared_setup$sorted_S,
      formula_class = prepared_setup$formula_class,
      data_hash = prepared_setup$dataset_sha256,
      n = nrow(prepared_setup$X),
      p = dataset_p
    )
  )
  fastkpc_full_cuda_target_state_require(
    identical(expected_group_id, prepared_setup$same_S_group_id),
    "TargetState group lineage mismatch"
  )

  selected_sp <- state_row$selected_sp[[1L]]
  selected_sp_names <- state_row$selected_sp_names[[1L]]
  penalty_count <- length(prepared_setup$penalty_blocks)
  fastkpc_full_cuda_target_state_require(
    is.numeric(selected_sp) && length(selected_sp) == penalty_count &&
      all(is.finite(selected_sp)) && all(selected_sp > 0) &&
      is.character(selected_sp_names) &&
      length(selected_sp_names) == penalty_count &&
      identical(
        as.character(selected_sp_names),
        as.character(prepared_setup$penalty_sp_labels)
      ),
    "TargetState selected sp is invalid"
  )
  fastkpc_full_cuda_target_state_require(
    identical(
      state_row$selected_sp_hash[[1L]],
      fastkpc_full_cuda_census_metadata_hash(selected_sp)
    ),
    "TargetState selected sp hash mismatch"
  )
  projected_rhs <- state_row$projected_rhs[[1L]]
  nullspace_projected_rhs <- state_row$nullspace_projected_rhs[[1L]]
  fastkpc_full_cuda_target_state_require(
    is.numeric(projected_rhs) &&
      length(projected_rhs) == ncol(prepared_setup$X) &&
      all(is.finite(projected_rhs)),
    "TargetState projected RHS mismatch"
  )
  fastkpc_full_cuda_target_state_require(
    is.numeric(nullspace_projected_rhs) &&
      length(nullspace_projected_rhs) ==
        prepared_setup$constraint_nullspace_dimension &&
      all(is.finite(nullspace_projected_rhs)),
    "TargetState nullspace projected RHS mismatch"
  )
  fastkpc_full_cuda_target_state_require(
    is.numeric(state_row$GCV_Cp_score) &&
      is.finite(state_row$GCV_Cp_score[[1L]]) &&
      is.numeric(state_row$EDF) && is.finite(state_row$EDF[[1L]]) &&
      is.integer(state_row$coefficient_rank) &&
      is.finite(state_row$coefficient_rank[[1L]]),
    "TargetState scalar metadata is invalid"
  )
  list(
    row = state_row,
    target = target,
    source = source,
    dataset_p = dataset_p,
    sp = as.numeric(selected_sp),
    projected_rhs = projected_rhs,
    nullspace_projected_rhs = nullspace_projected_rhs
  )
}

fastkpc_full_cuda_validate_materialized_target_for_prepared <- function(
    prepared_setup, target_state) {
  fastkpc_full_cuda_target_state_require(
    is.list(target_state) &&
      is.null(attr(target_state, "class", exact = TRUE)) &&
      identical(names(target_state), c("row", "y")),
    "materialized TargetState is malformed"
  )
  context <- fastkpc_full_cuda_validate_target_state_row_for_prepared(
    prepared_setup, target_state$row
  )
  y <- target_state$y
  fastkpc_full_cuda_target_state_require(
    is.numeric(y) && is.null(attributes(y)),
    "TargetState y must be numeric"
  )
  fastkpc_full_cuda_target_state_require(
    length(y) == nrow(prepared_setup$X),
    "TargetState y length mismatch"
  )
  fastkpc_full_cuda_target_state_require(
    all(is.finite(y)),
    "TargetState y must be finite"
  )
  fastkpc_full_cuda_target_state_require(
    identical(
      target_state$row$y_hash[[1L]],
      fastkpc_full_cuda_census_metadata_hash(y)
    ),
    "TargetState y hash mismatch"
  )
  expected_projected <- if (is.null(prepared_setup$weights)) {
    as.numeric(crossprod(prepared_setup$X, y))
  } else {
    as.numeric(crossprod(
      prepared_setup$X, y * as.numeric(prepared_setup$weights)
    ))
  }
  expected_null_projected <- if (identical(
    prepared_setup$constraint_mode, "identity"
  )) {
    expected_projected
  } else {
    as.numeric(crossprod(
      prepared_setup$constraint_nullspace, expected_projected
    ))
  }
  fastkpc_full_cuda_target_state_require(
    identical(context$projected_rhs, expected_projected),
    "TargetState projected RHS mismatch"
  )
  fastkpc_full_cuda_target_state_require(
    identical(context$nullspace_projected_rhs, expected_null_projected),
    "TargetState nullspace projected RHS mismatch"
  )
  context$y <- y
  context
}

fastkpc_mgcv_magic_fixed_sp_from_prepared <- function(
    prepared_setup, target_state) {
  fastkpc_full_cuda_validate_prepared_s_for_adapter(prepared_setup)
  target <- fastkpc_full_cuda_validate_materialized_target_for_prepared(
    prepared_setup, target_state
  )
  penalty_count <- length(prepared_setup$penalty_blocks)
  minimal <- list(
    G = list(
      L = matrix(numeric(), nrow = penalty_count, ncol = 0L),
      lsp0 = log(target$sp)
    ),
    X = prepared_setup$X,
    y = target$y,
    S = prepared_setup$penalty_blocks,
    off = prepared_setup$penalty_offsets,
    rank = prepared_setup$mgcv_penalty_rank_metadata,
    H = prepared_setup$H,
    C = prepared_setup$constraint,
    w = prepared_setup$weights,
    sp = target$sp,
    setup_fingerprint = list(
      schema_version = "prepared-s-fixed-sp-kernel-v1",
      prepared_s_key_sha256 = prepared_setup$prepared_s_key_sha256,
      semantic_fingerprint = prepared_setup$semantic_fingerprint,
      representation_fingerprint =
        prepared_setup$representation_fingerprint
    )
  )
  beta <- fastkpc_mgcv_magic_kernel_fixed_sp_coefficients(
    minimal, sp = target$sp
  )
  if (!is.numeric(beta) || length(beta) != ncol(prepared_setup$X) ||
      any(!is.finite(beta))) {
    stop("Prepared fixed-sp coefficients are invalid", call. = FALSE)
  }
  beta <- as.numeric(beta)
  fitted <- as.numeric(prepared_setup$X %*% beta)
  residuals <- as.numeric(target$y - fitted)
  if (any(!is.finite(fitted)) || any(!is.finite(residuals))) {
    stop("Prepared fixed-sp fitted values are invalid", call. = FALSE)
  }
  list(
    backend_family = "mgcvExtractCPU",
    mode = "prepared-s-fixed-sp-mgcv-reference",
    solve_source = "mgcv-C-magic-from-prepared-s",
    authoritative = TRUE,
    coefficients = beta,
    fitted = fitted,
    residuals = residuals,
    sp = target$sp,
    prepared_s_key_sha256 = prepared_setup$prepared_s_key_sha256,
    residual_key_sha256 = target$row$residual_key_sha256[[1L]]
  )
}

fastkpc_full_cuda_prepared_s_semantic_angle_tolerance <- function(X) {
  if (!is.matrix(X) || !is.numeric(X) || any(dim(X) == 0L) ||
      any(!is.finite(X))) {
    stop("semantic angle tolerance requires a finite nonempty matrix",
         call. = FALSE)
  }
  64 * .Machine$double.eps * max(nrow(X), ncol(X))
}

fastkpc_full_cuda_prepared_s_column_space <- function(X) {
  X <- fastkpc_full_cuda_prepared_s_matrix(X)
  diagnostics <- fastkpc_full_cuda_census_svd_diagnostics(
    X, expected_rank = ncol(X)
  )
  if (is.na(diagnostics$rank)) {
    stop("semantic model matrix rank is unavailable", call. = FALSE)
  }
  if (diagnostics$rank == 0L) {
    return(list(
      rank = 0L,
      basis = matrix(numeric(), nrow = nrow(X), ncol = 0L)
    ))
  }
  decomposition <- La.svd(
    X, nu = min(dim(X)), nv = 0L
  )
  list(
    rank = diagnostics$rank,
    basis = decomposition$u[, seq_len(diagnostics$rank), drop = FALSE]
  )
}

fastkpc_full_cuda_prepared_s_max_principal_angle <- function(left, right) {
  left <- fastkpc_full_cuda_prepared_s_matrix(left)
  right <- fastkpc_full_cuda_prepared_s_matrix(right)
  if (nrow(left) != nrow(right)) return(Inf)
  if (identical(
        fastkpc_full_cuda_census_metadata_hash(left),
        fastkpc_full_cuda_census_metadata_hash(right)
      )) {
    return(0)
  }
  left_space <- fastkpc_full_cuda_prepared_s_column_space(left)
  right_space <- fastkpc_full_cuda_prepared_s_column_space(right)
  if (left_space$rank == 0L && right_space$rank == 0L) return(0)
  if (left_space$rank == 0L || right_space$rank == 0L) return(pi / 2)
  cosines <- La.svd(
    crossprod(left_space$basis, right_space$basis),
    nu = 0L, nv = 0L
  )$d
  cosines <- pmin(1, pmax(0, as.numeric(cosines)))
  angles <- acos(cosines)
  if (left_space$rank != right_space$rank) angles <- c(angles, pi / 2)
  max(angles)
}

fastkpc_full_cuda_prepared_s_constraint_projector <- function(C, p) {
  Z <- fastkpc_constraint_nullspace(C = C, p = p)
  fastkpc_full_cuda_prepared_s_matrix(Z %*% t(Z))
}

fastkpc_full_cuda_compare_prepared_s_semantics <- function(
    prepared_setup, reference_setup, solved_result, state_row) {
  fastkpc_full_cuda_validate_prepared_s_for_adapter(prepared_setup)
  state <- fastkpc_full_cuda_validate_target_state_row_for_prepared(
    prepared_setup, state_row
  )
  required_reference <- c("X", "S", "off", "C", "rank", "H", "w", "sp")
  if (!is.list(reference_setup) ||
      length(setdiff(required_reference, names(reference_setup))) > 0L ||
      !is.matrix(reference_setup$X) ||
      !is.numeric(reference_setup$X) ||
      any(!is.finite(reference_setup$X)) ||
      nrow(reference_setup$X) != nrow(prepared_setup$X)) {
    stop("reference setup is malformed", call. = FALSE)
  }
  expected_result_fields <- c(
    "backend_family", "mode", "solve_source", "authoritative",
    "coefficients", "fitted", "residuals", "sp",
    "prepared_s_key_sha256", "residual_key_sha256"
  )
  if (!is.list(solved_result) ||
      !identical(names(solved_result), expected_result_fields) ||
      !identical(
        solved_result$prepared_s_key_sha256,
        prepared_setup$prepared_s_key_sha256
      ) ||
      !identical(
        solved_result$residual_key_sha256,
        state$row$residual_key_sha256[[1L]]
      ) ||
      !identical(as.numeric(solved_result$sp), state$sp) ||
      !is.numeric(solved_result$coefficients) ||
      length(solved_result$coefficients) != ncol(prepared_setup$X) ||
      !is.numeric(solved_result$fitted) ||
      length(solved_result$fitted) != nrow(prepared_setup$X) ||
      !is.numeric(solved_result$residuals) ||
      length(solved_result$residuals) != nrow(prepared_setup$X) ||
      any(!is.finite(c(
        solved_result$coefficients, solved_result$fitted,
        solved_result$residuals, solved_result$sp
      )))) {
    stop("solved Prepared-S result is malformed", call. = FALSE)
  }

  prepared_X <- fastkpc_full_cuda_prepared_s_matrix(prepared_setup$X)
  reference_X <- fastkpc_full_cuda_prepared_s_matrix(reference_setup$X)
  prepared_model <- fastkpc_full_cuda_prepared_s_column_space(prepared_X)
  reference_model <- fastkpc_full_cuda_prepared_s_column_space(reference_X)
  model_matrix_hash_equal <- identical(
    fastkpc_full_cuda_census_metadata_hash(prepared_X),
    fastkpc_full_cuda_census_metadata_hash(reference_X)
  )
  model_matrix_rank_equal <- identical(
    prepared_model$rank, reference_model$rank
  )
  semantic_angle_tolerance <-
    fastkpc_full_cuda_prepared_s_semantic_angle_tolerance(prepared_X)
  max_column_space_principal_angle <-
    fastkpc_full_cuda_prepared_s_max_principal_angle(
      prepared_X, reference_X
    )

  p <- ncol(prepared_X)
  prepared_constraint_rank <- fastkpc_full_cuda_census_svd_diagnostics(
    prepared_setup$constraint,
    expected_rank = min(dim(prepared_setup$constraint))
  )$rank
  reference_C <- if (is.null(reference_setup$C) ||
                     length(reference_setup$C) == 0L) {
    matrix(numeric(), nrow = 0L, ncol = p)
  } else {
    fastkpc_full_cuda_prepared_s_matrix(reference_setup$C)
  }
  reference_constraint_rank <- fastkpc_full_cuda_census_svd_diagnostics(
    reference_C, expected_rank = min(dim(reference_C))
  )$rank
  constraint_rank_equal <- identical(
    prepared_constraint_rank, reference_constraint_rank
  )
  prepared_projector <-
    fastkpc_full_cuda_prepared_s_constraint_projector(
      prepared_setup$constraint, p
    )
  reference_projector <-
    fastkpc_full_cuda_prepared_s_constraint_projector(reference_C, p)
  constraint_projector_max_abs_diff <-
    fastkpc_full_cuda_census_max_abs_diff(
      prepared_projector, reference_projector
    )
  constraint_action_equal <- is.finite(
    constraint_projector_max_abs_diff
  ) && constraint_projector_max_abs_diff <= semantic_angle_tolerance

  reference_sp <- fastkpc_validate_fixed_positive_sp(
    reference_setup$sp,
    expected_length = length(reference_setup$S)
  )
  prepared_penalty_hashes <- unname(vapply(
    prepared_setup$penalty_blocks,
    fastkpc_full_cuda_census_metadata_hash,
    character(1L)
  ))
  reference_penalty_hashes <- unname(vapply(
    reference_setup$S,
    fastkpc_full_cuda_census_metadata_hash,
    character(1L)
  ))
  prepared_penalty_action <- fastkpc_assemble_penalty(
    p = p,
    S = prepared_setup$penalty_blocks,
    off = prepared_setup$penalty_offsets,
    sp = state$sp,
    H = prepared_setup$H
  )
  reference_penalty_action <- fastkpc_assemble_penalty(
    p = ncol(reference_X),
    S = reference_setup$S,
    off = reference_setup$off,
    sp = reference_sp,
    H = reference_setup$H
  )
  penalty_order_equal <-
    identical(length(prepared_setup$penalty_blocks),
              length(reference_setup$S)) &&
    identical(
      as.integer(prepared_setup$penalty_offsets),
      as.integer(reference_setup$off)
    ) &&
    identical(
      as.integer(prepared_setup$mgcv_penalty_rank_metadata),
      as.integer(reference_setup$rank)
    ) &&
    identical(prepared_penalty_hashes, reference_penalty_hashes) &&
    identical(state$sp, as.numeric(reference_sp)) &&
    identical(
      fastkpc_full_cuda_census_metadata_hash(prepared_penalty_action),
      fastkpc_full_cuda_census_metadata_hash(reference_penalty_action)
    )

  coefficient_hash_equal <- identical(
    fastkpc_full_cuda_census_metadata_hash(solved_result$coefficients),
    state$row$coefficient_hash[[1L]]
  )
  fitted_hash_equal <- identical(
    fastkpc_full_cuda_census_metadata_hash(solved_result$fitted),
    state$row$fitted_hash[[1L]]
  )
  residual_hash_equal <- identical(
    fastkpc_full_cuda_census_metadata_hash(solved_result$residuals),
    state$row$residual_hash[[1L]]
  )
  exact_behavior_gate <- coefficient_hash_equal &&
    fitted_hash_equal && residual_hash_equal

  list(
    model_matrix_hash_equal = model_matrix_hash_equal,
    model_matrix_rank_equal = model_matrix_rank_equal,
    max_column_space_principal_angle =
      max_column_space_principal_angle,
    semantic_angle_tolerance = semantic_angle_tolerance,
    constraint_rank_equal = constraint_rank_equal,
    constraint_projector_max_abs_diff =
      constraint_projector_max_abs_diff,
    constraint_action_equal = constraint_action_equal,
    penalty_order_equal = penalty_order_equal,
    coefficient_hash_equal = coefficient_hash_equal,
    fitted_hash_equal = fitted_hash_equal,
    residual_hash_equal = residual_hash_equal,
    exact_behavior_gate = exact_behavior_gate
  )
}

fastkpc_full_cuda_prepared_s_selection_character <- function(value) {
  enc2utf8(as.character(value))
}

fastkpc_full_cuda_prepared_s_selection_numeric <- function(value) {
  if (is.factor(value)) value <- as.character(value)
  suppressWarnings(as.numeric(value))
}

fastkpc_full_cuda_prepared_s_selection_integer <- function(value) {
  if (is.factor(value)) value <- as.character(value)
  suppressWarnings(as.integer(value))
}

fastkpc_full_cuda_prepared_s_selection_logical <- function(value) {
  if (is.factor(value)) value <- as.character(value)
  as.logical(value)
}

fastkpc_full_cuda_prepared_s_selection_normalize_fields <- function(
    value, character_fields = character(), integer_fields = character(),
    numeric_fields = character(), logical_fields = character()) {
  value <- as.data.frame(value, stringsAsFactors = FALSE)
  for (field in intersect(character_fields, names(value))) {
    value[[field]] <-
      fastkpc_full_cuda_prepared_s_selection_character(value[[field]])
  }
  for (field in intersect(integer_fields, names(value))) {
    value[[field]] <-
      fastkpc_full_cuda_prepared_s_selection_integer(value[[field]])
  }
  for (field in intersect(numeric_fields, names(value))) {
    value[[field]] <-
      fastkpc_full_cuda_prepared_s_selection_numeric(value[[field]])
  }
  for (field in intersect(logical_fields, names(value))) {
    value[[field]] <-
      fastkpc_full_cuda_prepared_s_selection_logical(value[[field]])
  }
  value
}

fastkpc_full_cuda_prepared_s_selection_setup_rows <- function(inputs) {
  if (!is.list(inputs) || !is.data.frame(inputs$same_s_setup_metadata)) {
    stop("Prepared-S selection setup metadata is unavailable",
         call. = FALSE)
  }
  rows <- fastkpc_full_cuda_prepared_s_selection_normalize_fields(
    inputs$same_s_setup_metadata,
    character_fields = c(
      "same_S_group_id", "S_key", "formula_class",
      "representative_residual_key_sha256", "setup_fingerprint"
    ),
    integer_fields = c(
      "S_size", "model_matrix_ncol", "model_matrix_rank",
      "penalty_count", "conditioning_rank",
      "near_constant_conditioning_count"
    ),
    numeric_fields = c(
      "model_matrix_condition", "conditioning_condition"
    )
  )
  required <- c(
    "same_S_group_id", "S_size", "representative_residual_key_sha256",
    "model_matrix_ncol", "model_matrix_rank", "model_matrix_condition",
    "penalty_count", "conditioning_rank", "conditioning_condition",
    "near_constant_conditioning_count"
  )
  if (length(setdiff(required, names(rows))) > 0L) {
    stop("Prepared-S selection setup metadata is incomplete",
         call. = FALSE)
  }
  rows <- rows[
    order(rows$same_S_group_id, method = "radix"), , drop = FALSE
  ]
  rownames(rows) <- NULL
  if (nrow(rows) == 0L || anyNA(rows$same_S_group_id) ||
      anyDuplicated(rows$same_S_group_id)) {
    stop("Prepared-S selection setup keys are invalid", call. = FALSE)
  }
  rows
}

fastkpc_full_cuda_prepared_s_selection_target_rows <- function(inputs) {
  if (!is.list(inputs) || !is.data.frame(inputs$target_fit_metadata)) {
    stop("Prepared-S selection target metadata is unavailable",
         call. = FALSE)
  }
  rows <- fastkpc_full_cuda_prepared_s_selection_normalize_fields(
    inputs$target_fit_metadata,
    character_fields = c(
      "residual_key_sha256", "same_S_group_id", "setup_fingerprint",
      "fit_status", "fit_error", "selected_sp_hash",
      "coefficient_hash", "fitted_hash", "residual_hash",
      "target_fit_fingerprint"
    ),
    integer_fields = c("target", "coefficient_rank"),
    numeric_fields = c(
      "fit_time_ms", "GCV_Cp_score", "EDF",
      "penalized_system_condition_at_selected_sp", "target_sd"
    ),
    logical_fields = c(
      "target_near_constant", "coefficient_all_finite",
      "fitted_all_finite", "residual_all_finite"
    )
  )
  rows <- rows[
    order(rows$residual_key_sha256, method = "radix"), , drop = FALSE
  ]
  rownames(rows) <- NULL
  if (nrow(rows) == 0L || anyNA(rows$residual_key_sha256) ||
      anyDuplicated(rows$residual_key_sha256)) {
    stop("Prepared-S selection target keys are invalid", call. = FALSE)
  }
  rows
}

fastkpc_full_cuda_prepared_s_selection_request_rows <- function(inputs) {
  if (!is.list(inputs) || !is.data.frame(inputs$residual_requests)) {
    stop("Prepared-S selection residual requests are unavailable",
         call. = FALSE)
  }
  rows <- fastkpc_full_cuda_prepared_s_selection_normalize_fields(
    inputs$residual_requests,
    character_fields = c(
      "residual_key_payload", "residual_key_sha256", "S_key",
      "formula_class", "same_S_group_id"
    ),
    integer_fields = c(
      "target", "S_size", "same_S_group_size", "request_multiplicity",
      "first_logical_sequence_id", "last_logical_sequence_id"
    )
  )
  rows <- rows[
    order(rows$residual_key_sha256, method = "radix"), , drop = FALSE
  ]
  rownames(rows) <- NULL
  if (nrow(rows) == 0L || anyNA(rows$residual_key_sha256) ||
      anyDuplicated(rows$residual_key_sha256)) {
    stop("Prepared-S selection residual request keys are invalid",
         call. = FALSE)
  }
  rows
}

fastkpc_full_cuda_prepared_s_selection_logical_rows <- function(inputs) {
  if (!is.list(inputs) || !is.data.frame(inputs$logical_tests)) {
    stop("Prepared-S selection logical tests are unavailable",
         call. = FALSE)
  }
  rows <- fastkpc_full_cuda_prepared_s_selection_normalize_fields(
    inputs$logical_tests,
    character_fields = c(
      "S_key", "formula_class", "reference_decision",
      "residual_key_x", "residual_key_y"
    ),
    integer_fields = c(
      "logical_sequence_id", "source_sequence_id", "source_task_index",
      "level", "x", "y", "S_size"
    ),
    numeric_fields = c(
      "reference_p_value", "alpha", "signed_distance_from_alpha",
      "absolute_distance_from_alpha", "signed_log_ratio_from_alpha",
      "absolute_log_distance_from_alpha"
    ),
    logical_fields = c(
      "reference_independent", "deletes_edge"
    )
  )
  rows <- rows[
    order(rows$logical_sequence_id), , drop = FALSE
  ]
  rownames(rows) <- NULL
  if (nrow(rows) == 0L || anyNA(rows$logical_sequence_id) ||
      anyDuplicated(rows$logical_sequence_id)) {
    stop("Prepared-S selection logical identities are invalid",
         call. = FALSE)
  }
  rows
}

fastkpc_full_cuda_prepared_s_convergence_field_value <- function(
    fields, name) {
  if (is.null(fields) || !is.list(fields)) return(NULL)
  field <- fields[[name]]
  if (is.null(field) || !is.list(field)) return(NULL)
  field$value
}

fastkpc_full_cuda_prepared_s_convergence_boolean_tag <- function(value) {
  if (is.null(value)) return("missing")
  if (length(value) != 1L || is.na(value)) return("invalid")
  if (isTRUE(value)) "TRUE" else "FALSE"
}

fastkpc_full_cuda_prepared_s_convergence_signature <- function(fields) {
  converged <- fastkpc_full_cuda_prepared_s_convergence_field_value(
    fields, "converged"
  )
  mgcv_conv <- fastkpc_full_cuda_prepared_s_convergence_field_value(
    fields, "mgcv.conv"
  )
  outer_info <- fastkpc_full_cuda_prepared_s_convergence_field_value(
    fields, "outer.info"
  )
  rank_full <- if (is.null(mgcv_conv$rank) ||
                   is.null(mgcv_conv$full.rank)) {
    "missing"
  } else if (identical(
    as.integer(mgcv_conv$rank), as.integer(mgcv_conv$full.rank)
  )) {
    "TRUE"
  } else {
    "FALSE"
  }
  outer_convergence <- if (is.null(outer_info$conv)) {
    "missing"
  } else if (fastkpc_full_cuda_census_nonconverged_values(
    outer_info = list(conv = outer_info$conv)
  )) {
    "failed"
  } else {
    "converged"
  }
  paste0(
    "converged=",
    fastkpc_full_cuda_prepared_s_convergence_boolean_tag(converged),
    ";fully_converged=",
    fastkpc_full_cuda_prepared_s_convergence_boolean_tag(
      mgcv_conv$fully.converged
    ),
    ";hessian_positive_definite=",
    fastkpc_full_cuda_prepared_s_convergence_boolean_tag(
      mgcv_conv$hess.pos.def
    ),
    ";rank_full=", rank_full,
    ";outer_convergence=", outer_convergence
  )
}

fastkpc_full_cuda_prepared_s_optimizer_iterations <- function(fields) {
  mgcv_conv <- fastkpc_full_cuda_prepared_s_convergence_field_value(
    fields, "mgcv.conv"
  )
  outer_info <- fastkpc_full_cuda_prepared_s_convergence_field_value(
    fields, "outer.info"
  )
  candidates <- list(mgcv_conv$iter, outer_info$iter)
  for (candidate in candidates) {
    value <- suppressWarnings(as.numeric(candidate))
    if (length(value) == 1L && is.finite(value) && value >= 0 &&
        value == floor(value)) {
      return(as.integer(value))
    }
  }
  NA_integer_
}

fastkpc_full_cuda_prepared_s_normalize_risk_rows <- function(value) {
  fastkpc_full_cuda_prepared_s_selection_normalize_fields(
    value,
    character_fields = c(
      "case_type", "residual_key_sha256", "same_S_group_id",
      "condition_bucket", "near_alpha_bucket"
    ),
    integer_fields = "logical_sequence_id",
    logical_fields = c(
      "high_condition", "rank_deficient", "near_constant_target",
      "near_constant_conditioner", "multi_penalty", "near_alpha",
      "mgcv_warning", "mgcv_nonconverged", "nonfinite_metadata"
    )
  )
}

fastkpc_full_cuda_reconstruct_prepared_s_target_risk_table <- function(
    inputs) {
  required_inputs <- c(
    "target_fit_metadata", "same_s_setup_metadata", "residual_requests",
    "target_risks", "risk_cases", "logical_tests"
  )
  if (!is.list(inputs) ||
      length(setdiff(required_inputs, names(inputs))) > 0L) {
    stop("Prepared-S target-risk inputs are incomplete", call. = FALSE)
  }
  target <- fastkpc_full_cuda_prepared_s_selection_target_rows(inputs)
  setup <- fastkpc_full_cuda_prepared_s_selection_setup_rows(inputs)
  requests <- fastkpc_full_cuda_prepared_s_selection_request_rows(inputs)
  logical_tests <-
    fastkpc_full_cuda_prepared_s_selection_logical_rows(inputs)
  if (!identical(
        target$residual_key_sha256, requests$residual_key_sha256
      )) {
    stop("Prepared-S target/request key set mismatch", call. = FALSE)
  }
  if (any(target$fit_status != "success") ||
      any(target$fit_error != "NONE")) {
    stop("Prepared-S target-risk reconstruction requires successful fits",
         call. = FALSE)
  }

  risk_config <- fastkpc_full_cuda_census_risk_config()
  expected <- fastkpc_full_cuda_census_expected_target_risks(
    target = target, setup = setup, risk_config = risk_config
  )
  expected <- fastkpc_full_cuda_prepared_s_normalize_risk_rows(expected)
  rownames(expected) <- NULL

  actual <- fastkpc_full_cuda_prepared_s_normalize_risk_rows(
    inputs$target_risks
  )
  actual <- actual[
    order(actual$residual_key_sha256, method = "radix"), , drop = FALSE
  ]
  rownames(actual) <- NULL
  if (!identical(names(actual), names(expected)) ||
      !identical(actual, expected)) {
    stop("Prepared-S reconstructed target-risk semantics mismatch",
         call. = FALSE)
  }

  expected_cases <- fastkpc_full_cuda_census_risk_cases(
    target_risks = expected, logical_tests = logical_tests
  )
  expected_cases <-
    fastkpc_full_cuda_prepared_s_normalize_risk_rows(expected_cases)
  actual_cases <- fastkpc_full_cuda_prepared_s_normalize_risk_rows(
    inputs$risk_cases
  )
  actual_target_cases <- actual_cases[
    actual_cases$case_type == "target_key", , drop = FALSE
  ]
  actual_target_cases <- actual_target_cases[
    order(actual_target_cases$residual_key_sha256, method = "radix"),
    , drop = FALSE
  ]
  actual_logical_cases <- actual_cases[
    actual_cases$case_type == "logical_test", , drop = FALSE
  ]
  actual_logical_cases <- actual_logical_cases[
    order(actual_logical_cases$logical_sequence_id), , drop = FALSE
  ]
  actual_cases <- rbind(actual_target_cases, actual_logical_cases)
  rownames(actual_cases) <- NULL
  rownames(expected_cases) <- NULL
  if (!identical(names(actual_cases), names(expected_cases)) ||
      !identical(actual_cases, expected_cases)) {
    stop("Prepared-S filtered risk-case semantics mismatch",
         call. = FALSE)
  }

  setup_index <- match(target$same_S_group_id, setup$same_S_group_id)
  request_index <- match(
    target$residual_key_sha256, requests$residual_key_sha256
  )
  if (anyNA(setup_index) || anyNA(request_index)) {
    stop("Prepared-S target-risk metadata joins are incomplete",
         call. = FALSE)
  }
  group_ids <- setup$same_S_group_id
  request_groups <- requests$same_S_group_id
  target_counts <- table(request_groups)
  logical_counts <- tapply(
    requests$request_multiplicity, request_groups, sum
  )
  setup_target_count <- as.integer(target_counts[group_ids])
  setup_logical_count <- as.integer(logical_counts[group_ids])
  if (anyNA(setup_target_count) || anyNA(setup_logical_count)) {
    stop("Prepared-S setup multiplicities are incomplete", call. = FALSE)
  }
  expected_group_size <- as.integer(
    target_counts[requests$same_S_group_id]
  )
  if (!identical(requests$same_S_group_size, expected_group_size)) {
    stop("Prepared-S setup target multiplicity mismatch", call. = FALSE)
  }

  convergence_signature <- vapply(
    target$convergence_fields,
    fastkpc_full_cuda_prepared_s_convergence_signature,
    character(1L)
  )
  optimizer_iterations <- vapply(
    target$convergence_fields,
    fastkpc_full_cuda_prepared_s_optimizer_iterations,
    integer(1L)
  )
  if (anyNA(convergence_signature) || any(!nzchar(convergence_signature)) ||
      anyNA(optimizer_iterations)) {
    stop("Prepared-S convergence selection metadata is incomplete",
         call. = FALSE)
  }

  result <- expected
  result$target <- target$target
  result$S_size <- setup$S_size[setup_index]
  result$penalty_count <- setup$penalty_count[setup_index]
  result$condition <-
    target$penalized_system_condition_at_selected_sp
  result$request_multiplicity <-
    requests$request_multiplicity[request_index]
  result$same_S_group_size <-
    requests$same_S_group_size[request_index]
  result$setup_target_count <- setup_target_count[setup_index]
  result$setup_logical_request_count <-
    setup_logical_count[setup_index]
  result$representative_residual_key_sha256 <-
    setup$representative_residual_key_sha256[setup_index]
  result$convergence_signature <- convergence_signature
  result$optimizer_iterations <- optimizer_iterations
  result$selected_sp <- target$selected_sp
  result$selected_sp_names <- target$selected_sp_names
  result$selected_sp_hash <- target$selected_sp_hash
  result$fit_time_ms <- target$fit_time_ms
  result$coefficient_all_finite <- target$coefficient_all_finite
  result$fitted_all_finite <- target$fitted_all_finite
  result$residual_all_finite <- target$residual_all_finite
  result$coefficient_hash <- target$coefficient_hash
  result$fitted_hash <- target$fitted_hash
  result$residual_hash <- target$residual_hash
  result$target_fit_fingerprint <- target$target_fit_fingerprint
  result$setup_fingerprint <- target$setup_fingerprint

  rank_deficient <- result$rank_deficient %in% TRUE
  rank_clean <- !any(rank_deficient) || (
    all(is.infinite(result$condition[rank_deficient])) &&
      all(result$condition_bucket[rank_deficient] ==
            "rank_deficient_inf") &&
      all(result$nonfinite_metadata[rank_deficient]) &&
      all(result$coefficient_all_finite[rank_deficient]) &&
      all(result$fitted_all_finite[rank_deficient]) &&
      all(result$residual_all_finite[rank_deficient])
  )
  if (!rank_clean) {
    stop("Prepared-S rank-deficient target metadata is invalid",
         call. = FALSE)
  }
  rownames(result) <- NULL
  attr(result, "schema_version") <-
    "full-cuda-ci-prepared-s-target-risk-selection-v1"
  result
}

fastkpc_full_cuda_prepared_s_lower_median_index <- function(length) {
  length <- as.integer(length)
  if (length(length) != 1L || is.na(length) || length < 1L) {
    stop("Prepared-S lower median requires a positive length",
         call. = FALSE)
  }
  (length + 1L) %/% 2L
}

fastkpc_full_cuda_prepared_s_selection_reason_store <- function() {
  new.env(hash = TRUE, parent = emptyenv())
}

fastkpc_full_cuda_prepared_s_add_selection_reason <- function(
    store, ids, reasons) {
  ids <- as.character(ids)
  reasons <- as.character(reasons)
  if (length(ids) == 1L && length(reasons) > 1L) {
    ids <- rep(ids, length(reasons))
  } else if (length(reasons) == 1L && length(ids) > 1L) {
    reasons <- rep(reasons, length(ids))
  }
  if (!is.environment(store) || length(ids) != length(reasons) ||
      anyNA(ids) || anyNA(reasons) || any(!nzchar(ids)) ||
      any(!nzchar(reasons))) {
    stop("Prepared-S selection reason is invalid", call. = FALSE)
  }
  for (index in seq_along(ids)) {
    id <- ids[[index]]
    current <- if (exists(id, envir = store, inherits = FALSE)) {
      get(id, envir = store, inherits = FALSE)
    } else {
      character()
    }
    assign(
      id,
      sort(unique(c(current, reasons[[index]])), method = "radix"),
      envir = store
    )
  }
  invisible(store)
}

fastkpc_full_cuda_prepared_s_selection_ids <- function(store) {
  if (!is.environment(store)) {
    stop("Prepared-S selection reason store is invalid", call. = FALSE)
  }
  sort(ls(store, all.names = TRUE), method = "radix")
}

fastkpc_full_cuda_prepared_s_selection_reasons <- function(store, ids) {
  ids <- as.character(ids)
  lapply(ids, function(id) {
    if (!exists(id, envir = store, inherits = FALSE)) {
      stop("Prepared-S selected row has no reason", call. = FALSE)
    }
    value <- get(id, envir = store, inherits = FALSE)
    sort(unique(as.character(value)), method = "radix")
  })
}

fastkpc_full_cuda_prepared_s_selection_reason_rows <- function(
    entity_type, ids, reasons) {
  chunks <- lapply(seq_along(ids), function(index) {
    data.frame(
      entity_type = entity_type,
      entity_id = as.character(ids[[index]]),
      reason = as.character(reasons[[index]]),
      stringsAsFactors = FALSE
    )
  })
  rows <- do.call(rbind, chunks)
  rows <- rows[order(
    rows$entity_type, rows$entity_id, rows$reason, method = "radix"
  ), , drop = FALSE]
  rownames(rows) <- NULL
  rows
}

fastkpc_full_cuda_prepared_s_selection_hash <- function(
    schema_version, setup_ids, target_keys, logical_ids, reason_rows,
    seed_target_keys = NULL) {
  fastkpc_full_cuda_census_metadata_hash(list(
    schema_version = as.character(schema_version),
    setup_ids = sort(unique(as.character(setup_ids)), method = "radix"),
    target_keys = sort(
      unique(as.character(target_keys)), method = "radix"
    ),
    logical_ids = sort(unique(as.integer(logical_ids))),
    seed_target_keys = if (is.null(seed_target_keys)) {
      character()
    } else {
      sort(unique(as.character(seed_target_keys)), method = "radix")
    },
    reason_rows_hash = fastkpc_full_cuda_census_frame_hash(reason_rows)
  ))
}

fastkpc_full_cuda_prepared_s_selection_groups <- function(
    labels, indices = seq_along(labels)) {
  labels <- as.character(labels)
  present <- !is.na(labels) & nzchar(labels)
  labels <- labels[present]
  indices <- indices[present]
  unique_labels <- sort(unique(labels), method = "radix")
  setNames(lapply(unique_labels, function(label) {
    indices[labels == label]
  }), unique_labels)
}

fastkpc_full_cuda_prepared_s_group_selection_metadata <- function(
    targets, setup_rows) {
  first_target <- match(
    setup_rows$same_S_group_id, targets$same_S_group_id
  )
  if (anyNA(first_target)) {
    stop("Prepared-S setup selection metadata is incomplete",
         call. = FALSE)
  }
  result <- data.frame(
    same_S_group_id = setup_rows$same_S_group_id,
    setup_target_count = targets$setup_target_count[first_target],
    setup_logical_request_count =
      targets$setup_logical_request_count[first_target],
    penalty_count = setup_rows$penalty_count,
    S_size = setup_rows$S_size,
    stringsAsFactors = FALSE
  )
  result
}

fastkpc_full_cuda_prepared_s_closest_consumer_rows <- function(
    logical_tests) {
  conditional_index <- which(logical_tests$S_size > 0L)
  rows <- rbind(
    data.frame(
      logical_row = conditional_index,
      logical_sequence_id =
        logical_tests$logical_sequence_id[conditional_index],
      residual_key_sha256 =
        logical_tests$residual_key_x[conditional_index],
      absolute_log_distance_from_alpha =
        logical_tests$absolute_log_distance_from_alpha[conditional_index],
      endpoint = "x",
      stringsAsFactors = FALSE
    ),
    data.frame(
      logical_row = conditional_index,
      logical_sequence_id =
        logical_tests$logical_sequence_id[conditional_index],
      residual_key_sha256 =
        logical_tests$residual_key_y[conditional_index],
      absolute_log_distance_from_alpha =
        logical_tests$absolute_log_distance_from_alpha[conditional_index],
      endpoint = "y",
      stringsAsFactors = FALSE
    )
  )
  rows <- rows[order(
    rows$residual_key_sha256,
    rows$absolute_log_distance_from_alpha,
    rows$logical_sequence_id,
    rows$endpoint,
    method = "radix",
    na.last = TRUE
  ), , drop = FALSE]
  rownames(rows) <- NULL
  rows
}

fastkpc_full_cuda_prepared_s_attach_group_reasons <- function(
    group_store, target_store, logical_store, targets, logical_tests,
    selected_group_ids) {
  selected_target_ids <-
    fastkpc_full_cuda_prepared_s_selection_ids(target_store)
  target_index <- match(
    selected_target_ids, targets$residual_key_sha256
  )
  if (anyNA(target_index)) {
    stop("Prepared-S selected target lineage is incomplete",
         call. = FALSE)
  }
  for (index in seq_along(selected_target_ids)) {
    key <- selected_target_ids[[index]]
    group_id <- targets$same_S_group_id[[target_index[[index]]]]
    reasons <- fastkpc_full_cuda_prepared_s_selection_reasons(
      target_store, key
    )[[1L]]
    fastkpc_full_cuda_prepared_s_add_selection_reason(
      group_store, group_id, paste0("target:", reasons)
    )
  }
  selected_logical_ids <- as.integer(
    fastkpc_full_cuda_prepared_s_selection_ids(logical_store)
  )
  logical_index <- match(
    selected_logical_ids, logical_tests$logical_sequence_id
  )
  if (anyNA(logical_index)) {
    stop("Prepared-S selected logical lineage is incomplete",
         call. = FALSE)
  }
  endpoint_index <- match(
    logical_tests$residual_key_x[logical_index],
    targets$residual_key_sha256
  )
  if (anyNA(endpoint_index)) {
    stop("Prepared-S selected logical endpoint is incomplete",
         call. = FALSE)
  }
  for (index in seq_along(selected_logical_ids)) {
    logical_id <- as.character(selected_logical_ids[[index]])
    group_id <- targets$same_S_group_id[[endpoint_index[[index]]]]
    reasons <- fastkpc_full_cuda_prepared_s_selection_reasons(
      logical_store, logical_id
    )[[1L]]
    fastkpc_full_cuda_prepared_s_add_selection_reason(
      group_store, group_id, paste0("logical:", reasons)
    )
  }
  missing <- setdiff(
    selected_group_ids,
    fastkpc_full_cuda_prepared_s_selection_ids(group_store)
  )
  if (length(missing) > 0L) {
    fastkpc_full_cuda_prepared_s_add_selection_reason(
      group_store, missing, "selected_setup"
    )
  }
  invisible(group_store)
}

fastkpc_full_cuda_select_prepared_s_iteration_subset <- function(inputs) {
  schema_version <- "full-cuda-ci-prepared-s-iteration-subset-v1"
  targets <-
    fastkpc_full_cuda_reconstruct_prepared_s_target_risk_table(inputs)
  setup_rows <- fastkpc_full_cuda_prepared_s_selection_setup_rows(inputs)
  logical_tests <-
    fastkpc_full_cuda_prepared_s_selection_logical_rows(inputs)
  target_keys <- targets$residual_key_sha256
  conditional <- logical_tests$S_size > 0L
  endpoint_x <- match(logical_tests$residual_key_x, target_keys)
  endpoint_y <- match(logical_tests$residual_key_y, target_keys)
  if (anyNA(endpoint_x[conditional]) || anyNA(endpoint_y[conditional])) {
    stop("Prepared-S iteration logical endpoints are incomplete",
         call. = FALSE)
  }

  logical_reasons <-
    fastkpc_full_cuda_prepared_s_selection_reason_store()
  target_reasons <-
    fastkpc_full_cuda_prepared_s_selection_reason_store()
  group_reasons <-
    fastkpc_full_cuda_prepared_s_selection_reason_store()

  tight <- conditional &
    is.finite(logical_tests$absolute_log_distance_from_alpha) &
    logical_tests$absolute_log_distance_from_alpha <= 1e-3
  fastkpc_full_cuda_prepared_s_add_selection_reason(
    logical_reasons,
    logical_tests$logical_sequence_id[tight],
    "conditional_log_distance_le_1e-3"
  )

  near <- conditional &
    is.finite(logical_tests$absolute_log_distance_from_alpha) &
    logical_tests$absolute_log_distance_from_alpha <= log(2)
  decision_tag <- logical_tests$reference_decision
  near_labels <- paste(
    logical_tests$S_size[near], decision_tag[near], sep = "|"
  )
  near_groups <- fastkpc_full_cuda_prepared_s_selection_groups(
    near_labels, which(near)
  )
  for (label in names(near_groups)) {
    candidates <- near_groups[[label]]
    selected <- candidates[order(
      logical_tests$absolute_log_distance_from_alpha[candidates],
      logical_tests$logical_sequence_id[candidates]
    )[[1L]]]
    fastkpc_full_cuda_prepared_s_add_selection_reason(
      logical_reasons,
      logical_tests$logical_sequence_id[[selected]],
      paste0("closest_near_alpha:", label)
    )
  }

  finite_ordinary <- with(
    targets,
    coefficient_all_finite & fitted_all_finite & residual_all_finite &
      is.finite(condition) & condition < 1e8 &
      !high_condition & !rank_deficient & !near_constant_target &
      !near_constant_conditioner & !mgcv_warning &
      !mgcv_nonconverged & !nonfinite_metadata
  )
  ordinary <- conditional & !is.na(endpoint_x) & !is.na(endpoint_y) &
    finite_ordinary[endpoint_x] & finite_ordinary[endpoint_y] &
    is.finite(logical_tests$absolute_log_distance_from_alpha) &
    logical_tests$absolute_log_distance_from_alpha > log(2)
  key_x <- logical_tests$residual_key_x
  key_y <- logical_tests$residual_key_y
  x_first <- key_x <= key_y
  pair_key <- paste0(
    ifelse(x_first, key_x, key_y), "|", ifelse(x_first, key_y, key_x)
  )
  for (S_size in seq_len(7L)) {
    candidates <- which(ordinary & logical_tests$S_size == S_size)
    if (length(candidates) == 0L) {
      stop("Prepared-S iteration ordinary stratum is empty",
           call. = FALSE)
    }
    candidates <- candidates[order(
      pair_key[candidates],
      logical_tests$logical_sequence_id[candidates],
      method = "radix"
    )]
    selected <- candidates[[
      fastkpc_full_cuda_prepared_s_lower_median_index(length(candidates))
    ]]
    fastkpc_full_cuda_prepared_s_add_selection_reason(
      logical_reasons,
      logical_tests$logical_sequence_id[[selected]],
      paste0("ordinary_lower_median:S_size=", S_size)
    )
  }

  risk_buckets <- c(
    "finite_1e8_to_lt_1e12", "finite_ge_1e12",
    "rank_deficient_inf"
  )
  risk_index <- which(targets$condition_bucket %in% risk_buckets)
  risk_labels <- paste(
    targets$penalty_count[risk_index],
    targets$condition_bucket[risk_index],
    sep = "|"
  )
  risk_groups <- fastkpc_full_cuda_prepared_s_selection_groups(
    risk_labels, risk_index
  )
  selected_risk_keys <- character()
  for (label in names(risk_groups)) {
    candidates <- risk_groups[[label]]
    if (targets$condition_bucket[[candidates[[1L]]]] ==
        "rank_deficient_inf") {
      selected <- candidates[order(
        targets$residual_key_sha256[candidates], method = "radix"
      )[[1L]]]
    } else {
      selected <- candidates[order(
        -targets$condition[candidates],
        targets$residual_key_sha256[candidates],
        method = "radix"
      )[[1L]]]
    }
    key <- targets$residual_key_sha256[[selected]]
    selected_risk_keys <- c(selected_risk_keys, key)
    fastkpc_full_cuda_prepared_s_add_selection_reason(
      target_reasons, key, paste0("condition_risk:", label)
    )
  }

  convergence_index <- which(targets$mgcv_nonconverged %in% TRUE)
  convergence_labels <- paste(
    targets$convergence_signature[convergence_index],
    targets$S_size[convergence_index],
    targets$condition_bucket[convergence_index],
    sep = "|"
  )
  convergence_groups <- fastkpc_full_cuda_prepared_s_selection_groups(
    convergence_labels, convergence_index
  )
  selected_convergence_keys <- character()
  for (label in names(convergence_groups)) {
    candidates <- convergence_groups[[label]]
    selected <- candidates[order(
      -targets$optimizer_iterations[candidates],
      targets$residual_key_sha256[candidates],
      method = "radix"
    )[[1L]]]
    key <- targets$residual_key_sha256[[selected]]
    selected_convergence_keys <- c(selected_convergence_keys, key)
    fastkpc_full_cuda_prepared_s_add_selection_reason(
      target_reasons, key, paste0("convergence_risk:", label)
    )
  }

  group_metadata <- fastkpc_full_cuda_prepared_s_group_selection_metadata(
    targets, setup_rows
  )
  anchor_groups <- character()
  select_anchor <- function(metric, anchor, reason) {
    distance <- abs(metric - anchor)
    candidates <- which(distance == min(distance))
    selected <- candidates[[1L]]
    group_id <- group_metadata$same_S_group_id[[selected]]
    keys <- targets$residual_key_sha256[
      targets$same_S_group_id == group_id
    ]
    fastkpc_full_cuda_prepared_s_add_selection_reason(
      target_reasons, keys, reason
    )
    fastkpc_full_cuda_prepared_s_add_selection_reason(
      group_reasons, group_id, reason
    )
    anchor_groups <<- c(anchor_groups, group_id)
    invisible(group_id)
  }
  for (anchor in c(2L, 9L, 47L)) {
    select_anchor(
      group_metadata$setup_target_count,
      anchor,
      paste0("setup_target_fanout_anchor=", anchor)
    )
  }
  for (anchor in c(2L, 16L, 3092L)) {
    select_anchor(
      group_metadata$setup_logical_request_count,
      anchor,
      paste0("setup_logical_request_load_anchor=", anchor)
    )
  }
  anchor_groups <- sort(unique(anchor_groups), method = "radix")

  consumers <-
    fastkpc_full_cuda_prepared_s_closest_consumer_rows(logical_tests)
  first_consumer <- consumers[
    !duplicated(consumers$residual_key_sha256), , drop = FALSE
  ]
  attach_target_consumer <- function(keys, reason_prefix) {
    consumer_index <- match(
      keys, first_consumer$residual_key_sha256
    )
    if (anyNA(consumer_index)) {
      stop("Prepared-S iteration target has no conditional consumer",
           call. = FALSE)
    }
    fastkpc_full_cuda_prepared_s_add_selection_reason(
      logical_reasons,
      first_consumer$logical_sequence_id[consumer_index],
      paste0(reason_prefix, keys)
    )
  }
  attach_target_consumer(selected_risk_keys, "condition_risk_consumer:")
  attach_target_consumer(
    selected_convergence_keys, "convergence_risk_consumer:"
  )
  for (group_id in anchor_groups) {
    keys <- targets$residual_key_sha256[
      targets$same_S_group_id == group_id
    ]
    candidates <- which(consumers$residual_key_sha256 %in% keys)
    candidates <- candidates[order(
      consumers$absolute_log_distance_from_alpha[candidates],
      consumers$logical_sequence_id[candidates],
      consumers$residual_key_sha256[candidates],
      method = "radix",
      na.last = TRUE
    )]
    if (length(candidates) == 0L) {
      stop("Prepared-S iteration group has no conditional consumer",
           call. = FALSE)
    }
    fastkpc_full_cuda_prepared_s_add_selection_reason(
      logical_reasons,
      consumers$logical_sequence_id[[candidates[[1L]]]],
      paste0("multiplicity_group_consumer:", group_id)
    )
  }

  selected_logical_ids <- sort(as.integer(
    fastkpc_full_cuda_prepared_s_selection_ids(logical_reasons)
  ))
  selected_logical_index <- match(
    selected_logical_ids, logical_tests$logical_sequence_id
  )
  if (anyNA(selected_logical_index)) {
    stop("Prepared-S iteration logical selection is invalid",
         call. = FALSE)
  }
  for (index in selected_logical_index) {
    logical_id <- logical_tests$logical_sequence_id[[index]]
    endpoints <- c(
      logical_tests$residual_key_x[[index]],
      logical_tests$residual_key_y[[index]]
    )
    fastkpc_full_cuda_prepared_s_add_selection_reason(
      target_reasons, endpoints,
      paste0("logical_endpoint:", logical_id)
    )
  }

  selected_target_keys <-
    fastkpc_full_cuda_prepared_s_selection_ids(target_reasons)
  selected_target_index <- match(selected_target_keys, target_keys)
  selected_group_ids <- sort(unique(
    targets$same_S_group_id[selected_target_index]
  ), method = "radix")
  for (group_id in selected_group_ids) {
    group_keys <- sort(
      targets$residual_key_sha256[
        targets$same_S_group_id == group_id
      ],
      method = "radix"
    )
    representative <- unique(
      targets$representative_residual_key_sha256[
        targets$same_S_group_id == group_id
      ]
    )
    if (length(representative) != 1L ||
        !representative %in% group_keys) {
      stop("Prepared-S iteration setup representative is invalid",
           call. = FALSE)
    }
    fastkpc_full_cuda_prepared_s_add_selection_reason(
      target_reasons, representative, "setup_representative"
    )
    fastkpc_full_cuda_prepared_s_add_selection_reason(
      target_reasons,
      group_keys[[
        fastkpc_full_cuda_prepared_s_lower_median_index(
          length(group_keys)
        )
      ]],
      "setup_lower_median_target"
    )
    fastkpc_full_cuda_prepared_s_add_selection_reason(
      target_reasons, group_keys[[length(group_keys)]],
      "setup_maximum_target"
    )
  }

  selected_target_keys <-
    fastkpc_full_cuda_prepared_s_selection_ids(target_reasons)
  selected_target_index <- match(selected_target_keys, target_keys)
  selected_group_ids <- sort(unique(
    targets$same_S_group_id[selected_target_index]
  ), method = "radix")
  if (length(selected_logical_ids) != 44L ||
      length(selected_target_keys) != 270L ||
      length(selected_group_ids) != 44L) {
    stop(
      "Prepared-S iteration subset count mismatch: setups=",
      length(selected_group_ids), ", targets=", length(selected_target_keys),
      ", logical_tests=", length(selected_logical_ids),
      call. = FALSE
    )
  }

  fastkpc_full_cuda_prepared_s_attach_group_reasons(
    group_store = group_reasons,
    target_store = target_reasons,
    logical_store = logical_reasons,
    targets = targets,
    logical_tests = logical_tests,
    selected_group_ids = selected_group_ids
  )

  selected_targets <- targets[selected_target_index, , drop = FALSE]
  selected_targets$selection_reasons <- I(
    fastkpc_full_cuda_prepared_s_selection_reasons(
      target_reasons, selected_target_keys
    )
  )
  selected_setups <- setup_rows[
    match(selected_group_ids, setup_rows$same_S_group_id), , drop = FALSE
  ]
  selected_setups$setup_target_count <- group_metadata$setup_target_count[
    match(selected_group_ids, group_metadata$same_S_group_id)
  ]
  selected_setups$setup_logical_request_count <-
    group_metadata$setup_logical_request_count[
      match(selected_group_ids, group_metadata$same_S_group_id)
    ]
  selected_setups$selection_reasons <- I(
    fastkpc_full_cuda_prepared_s_selection_reasons(
      group_reasons, selected_group_ids
    )
  )
  selected_logical <- logical_tests[
    selected_logical_index, , drop = FALSE
  ]
  selected_logical$near_alpha <-
    is.finite(selected_logical$absolute_log_distance_from_alpha) &
    selected_logical$absolute_log_distance_from_alpha <= log(2)
  selected_logical$canonical_residual_pair_key <- pair_key[
    selected_logical_index
  ]
  selected_logical$selection_reasons <- I(
    fastkpc_full_cuda_prepared_s_selection_reasons(
      logical_reasons, as.character(selected_logical_ids)
    )
  )
  rownames(selected_targets) <- NULL
  rownames(selected_setups) <- NULL
  rownames(selected_logical) <- NULL

  reason_rows <- rbind(
    fastkpc_full_cuda_prepared_s_selection_reason_rows(
      "setup_group", selected_group_ids,
      selected_setups$selection_reasons
    ),
    fastkpc_full_cuda_prepared_s_selection_reason_rows(
      "target_key", selected_target_keys,
      selected_targets$selection_reasons
    ),
    fastkpc_full_cuda_prepared_s_selection_reason_rows(
      "logical_test", selected_logical_ids,
      selected_logical$selection_reasons
    )
  )
  reason_rows <- reason_rows[order(
    reason_rows$entity_type, reason_rows$entity_id, reason_rows$reason,
    method = "radix"
  ), , drop = FALSE]
  rownames(reason_rows) <- NULL
  iteration_subset_hash <-
    fastkpc_full_cuda_prepared_s_selection_hash(
      schema_version = schema_version,
      setup_ids = selected_group_ids,
      target_keys = selected_target_keys,
      logical_ids = selected_logical_ids,
      reason_rows = reason_rows
    )
  list(
    schema_version = schema_version,
    setup_groups = selected_setups,
    target_keys = selected_targets,
    logical_tests = selected_logical,
    reason_rows = reason_rows,
    setup_group_ids_hash = fastkpc_full_cuda_census_metadata_hash(
      selected_group_ids
    ),
    target_keys_hash = fastkpc_full_cuda_census_metadata_hash(
      selected_target_keys
    ),
    logical_test_ids_hash = fastkpc_full_cuda_census_metadata_hash(
      selected_logical_ids
    ),
    reason_rows_hash = fastkpc_full_cuda_census_frame_hash(reason_rows),
    iteration_subset_hash = iteration_subset_hash
  )
}

fastkpc_full_cuda_prepared_s_qualification_coverage <- function(
    targets, selected_targets, logical_tests, selected_logical,
    selected_group_ids, multiplicity_witnesses) {
  rows <- list()
  append_row <- function(
      coverage_type, coverage_value, canonical_count, selected_count,
      coverage_claimed) {
    rows[[length(rows) + 1L]] <<- data.frame(
      coverage_type = as.character(coverage_type),
      coverage_value = as.character(coverage_value),
      canonical_count = as.integer(canonical_count),
      selected_count = as.integer(selected_count),
      coverage_claimed = as.logical(coverage_claimed),
      stringsAsFactors = FALSE
    )
  }

  risk_fields <- c(
    "high_condition", "rank_deficient", "near_constant_target",
    "near_constant_conditioner", "multi_penalty", "near_alpha",
    "mgcv_warning", "mgcv_nonconverged", "nonfinite_metadata"
  )
  for (field in risk_fields) {
    canonical_count <- sum(targets[[field]] %in% TRUE)
    selected_count <- sum(selected_targets[[field]] %in% TRUE)
    append_row(
      "risk_class", field, canonical_count, selected_count,
      canonical_count > 0L && selected_count > 0L
    )
  }

  condition_buckets <-
    fastkpc_full_cuda_census_risk_config()$condition_buckets
  for (bucket in condition_buckets) {
    canonical_count <- sum(targets$condition_bucket == bucket)
    selected_count <- sum(selected_targets$condition_bucket == bucket)
    append_row(
      "condition_bucket", bucket, canonical_count, selected_count,
      canonical_count > 0L && selected_count > 0L
    )
  }

  penalty_counts <- sort(unique(targets$penalty_count))
  for (penalty_count in penalty_counts) {
    canonical_count <- sum(targets$penalty_count == penalty_count)
    selected_count <- sum(
      selected_targets$penalty_count == penalty_count
    )
    append_row(
      "penalty_count", as.character(penalty_count),
      canonical_count, selected_count,
      canonical_count > 0L && selected_count > 0L
    )
  }

  conditional <- logical_tests$S_size > 0L
  for (S_size in sort(unique(logical_tests$S_size[conditional]))) {
    canonical_count <- sum(conditional & logical_tests$S_size == S_size)
    selected_count <- sum(selected_logical$S_size == S_size)
    append_row(
      "S_size", as.character(S_size), canonical_count, selected_count,
      canonical_count > 0L && selected_count > 0L
    )
  }
  decisions <- sort(unique(
    logical_tests$reference_decision[conditional]
  ), method = "radix")
  for (decision in decisions) {
    canonical_count <- sum(
      conditional & logical_tests$reference_decision == decision
    )
    selected_count <- sum(
      selected_logical$reference_decision == decision
    )
    append_row(
      "reference_decision", decision, canonical_count, selected_count,
      canonical_count > 0L && selected_count > 0L
    )
  }

  if (nrow(multiplicity_witnesses) > 0L) {
    for (index in seq_len(nrow(multiplicity_witnesses))) {
      witness <- multiplicity_witnesses[index, , drop = FALSE]
      group_id <- witness$same_S_group_id[[1L]]
      selected_count <- as.integer(group_id %in% selected_group_ids)
      append_row(
        "setup_target_count_quantile",
        paste0(
          "penalty_count=", witness$penalty_count[[1L]],
          ";probability=", witness$probability[[1L]],
          ";quantile=", witness$target_count_quantile[[1L]],
          ";selected_count=", witness$selected_target_count[[1L]],
          ";same_S_group_id=", group_id
        ),
        1L, selected_count, selected_count == 1L
      )
    }
  }
  result <- do.call(rbind, rows)
  result <- result[order(
    result$coverage_type, result$coverage_value, method = "radix"
  ), , drop = FALSE]
  rownames(result) <- NULL
  result
}

fastkpc_full_cuda_select_prepared_s_qualification_subset <- function(
    inputs) {
  schema_version <- "full-cuda-ci-prepared-s-qualification-subset-v1"
  targets <-
    fastkpc_full_cuda_reconstruct_prepared_s_target_risk_table(inputs)
  setup_rows <- fastkpc_full_cuda_prepared_s_selection_setup_rows(inputs)
  logical_tests <-
    fastkpc_full_cuda_prepared_s_selection_logical_rows(inputs)
  target_reasons <-
    fastkpc_full_cuda_prepared_s_selection_reason_store()
  logical_reasons <-
    fastkpc_full_cuda_prepared_s_selection_reason_store()
  group_reasons <-
    fastkpc_full_cuda_prepared_s_selection_reason_store()

  rare_fields <- c(
    "rank_deficient", "nonfinite_metadata", "mgcv_nonconverged"
  )
  for (field in rare_fields) {
    keys <- targets$residual_key_sha256[targets[[field]] %in% TRUE]
    fastkpc_full_cuda_prepared_s_add_selection_reason(
      target_reasons, keys, paste0("rare:", field)
    )
  }
  single_penalty_high <- targets$high_condition %in% TRUE &
    targets$penalty_count == 1L
  fastkpc_full_cuda_prepared_s_add_selection_reason(
    target_reasons,
    targets$residual_key_sha256[single_penalty_high],
    "rare:single_penalty_high_condition"
  )

  condition_labels <- paste(
    targets$penalty_count, targets$condition_bucket, sep = "|"
  )
  condition_groups <- fastkpc_full_cuda_prepared_s_selection_groups(
    condition_labels
  )
  for (label in names(condition_groups)) {
    candidates <- condition_groups[[label]]
    candidates <- candidates[order(
      targets$condition[candidates],
      targets$residual_key_sha256[candidates],
      method = "radix",
      na.last = TRUE
    )]
    keys <- targets$residual_key_sha256[candidates]
    selected <- c(
      keys[[1L]],
      keys[[fastkpc_full_cuda_prepared_s_lower_median_index(
        length(keys)
      )]],
      keys[[length(keys)]],
      sort(keys, method = "radix")[[1L]]
    )
    reasons <- paste0(
      c(
        "condition_min:", "condition_lower_median:",
        "condition_max:", "condition_lexical_min:"
      ),
      label
    )
    fastkpc_full_cuda_prepared_s_add_selection_reason(
      target_reasons, selected, reasons
    )
  }

  penalty_counts <- sort(unique(targets$penalty_count))
  for (penalty_count in penalty_counts) {
    candidates <- which(targets$penalty_count == penalty_count)
    if (any(vapply(targets$selected_sp[candidates], function(value) {
      !is.numeric(value) || length(value) != penalty_count ||
        any(!is.finite(value)) || any(value <= 0)
    }, logical(1L)))) {
      stop("Prepared-S qualification selected-sp metadata is invalid",
           call. = FALSE)
    }
    for (component in seq_len(penalty_count)) {
      values <- vapply(candidates, function(index) {
        as.numeric(targets$selected_sp[[index]][[component]])
      }, numeric(1L))
      ordered <- candidates[order(
        values,
        targets$residual_key_sha256[candidates],
        method = "radix"
      )]
      keys <- targets$residual_key_sha256[ordered]
      selected <- c(
        keys[[1L]],
        keys[[fastkpc_full_cuda_prepared_s_lower_median_index(
          length(keys)
        )]],
        keys[[length(keys)]]
      )
      reasons <- paste0(
        c("selected_sp_min:", "selected_sp_lower_median:",
          "selected_sp_max:"),
        "penalty_count=", penalty_count,
        ";component=", component
      )
      fastkpc_full_cuda_prepared_s_add_selection_reason(
        target_reasons, selected, reasons
      )
    }
  }

  group_metadata <- fastkpc_full_cuda_prepared_s_group_selection_metadata(
    targets, setup_rows
  )
  witness_chunks <- list()
  probabilities <- c(0, 0.25, 0.5, 0.75, 1)
  for (penalty_count in penalty_counts) {
    setup_index <- which(
      group_metadata$penalty_count == penalty_count
    )
    counts <- group_metadata$setup_target_count[setup_index]
    for (probability in probabilities) {
      target_count_quantile <- as.numeric(stats::quantile(
        counts, probs = probability, type = 1, names = FALSE
      ))
      distance <- abs(counts - target_count_quantile)
      candidates <- setup_index[distance == min(distance)]
      selected_setup <- candidates[[1L]]
      group_id <- group_metadata$same_S_group_id[[selected_setup]]
      group_keys <- sort(
        targets$residual_key_sha256[
          targets$same_S_group_id == group_id
        ],
        method = "radix"
      )
      selected <- c(
        group_keys[[1L]],
        group_keys[[fastkpc_full_cuda_prepared_s_lower_median_index(
          length(group_keys)
        )]],
        group_keys[[length(group_keys)]]
      )
      reason_suffix <- paste0(
        "penalty_count=", penalty_count,
        ";probability=", format(probability, trim = TRUE),
        ";same_S_group_id=", group_id
      )
      fastkpc_full_cuda_prepared_s_add_selection_reason(
        target_reasons,
        selected,
        paste0(
          c(
            "setup_quantile_lexical_min:",
            "setup_quantile_lower_median:",
            "setup_quantile_max:"
          ),
          reason_suffix
        )
      )
      fastkpc_full_cuda_prepared_s_add_selection_reason(
        group_reasons, group_id,
        paste0("setup_target_count_quantile:", reason_suffix)
      )
      witness_chunks[[length(witness_chunks) + 1L]] <- data.frame(
        penalty_count = as.integer(penalty_count),
        probability = as.numeric(probability),
        target_count_quantile = as.integer(target_count_quantile),
        same_S_group_id = group_id,
        selected_target_count = as.integer(
          group_metadata$setup_target_count[[selected_setup]]
        ),
        stringsAsFactors = FALSE
      )
    }
  }
  multiplicity_witnesses <- do.call(rbind, witness_chunks)
  multiplicity_witnesses <- multiplicity_witnesses[order(
    multiplicity_witnesses$penalty_count,
    multiplicity_witnesses$probability,
    multiplicity_witnesses$same_S_group_id,
    method = "radix"
  ), , drop = FALSE]
  rownames(multiplicity_witnesses) <- NULL

  seed_target_keys <-
    fastkpc_full_cuda_prepared_s_selection_ids(target_reasons)
  if (length(seed_target_keys) != 2356L) {
    stop(
      "Prepared-S qualification seed count mismatch: ",
      length(seed_target_keys),
      call. = FALSE
    )
  }
  seed_target_index <- match(
    seed_target_keys, targets$residual_key_sha256
  )
  seed_reason_lists <-
    fastkpc_full_cuda_prepared_s_selection_reasons(
      target_reasons, seed_target_keys
    )

  conditional <- logical_tests$S_size > 0L
  near_alpha <- conditional &
    is.finite(logical_tests$absolute_log_distance_from_alpha) &
    logical_tests$absolute_log_distance_from_alpha <= log(2)
  fastkpc_full_cuda_prepared_s_add_selection_reason(
    logical_reasons,
    logical_tests$logical_sequence_id[near_alpha],
    "conditional_near_alpha"
  )
  if (sum(near_alpha) != 1478L) {
    stop("Prepared-S qualification near-alpha count mismatch",
         call. = FALSE)
  }

  consumers <-
    fastkpc_full_cuda_prepared_s_closest_consumer_rows(logical_tests)
  canonical_consumers <- consumers[order(
    consumers$residual_key_sha256,
    consumers$logical_sequence_id,
    consumers$endpoint,
    method = "radix"
  ), , drop = FALSE]
  canonical_consumers <- canonical_consumers[
    !duplicated(canonical_consumers$residual_key_sha256),
    , drop = FALSE
  ]
  consumer_index <- match(
    seed_target_keys, canonical_consumers$residual_key_sha256
  )
  if (anyNA(consumer_index)) {
    stop("Prepared-S qualification seed has no conditional consumer",
         call. = FALSE)
  }
  fastkpc_full_cuda_prepared_s_add_selection_reason(
    logical_reasons,
    canonical_consumers$logical_sequence_id[consumer_index],
    paste0("seed_consumer:", seed_target_keys)
  )

  decision_labels <- paste(
    logical_tests$S_size[conditional],
    logical_tests$reference_decision[conditional],
    sep = "|"
  )
  decision_groups <- fastkpc_full_cuda_prepared_s_selection_groups(
    decision_labels, which(conditional)
  )
  for (label in names(decision_groups)) {
    candidates <- decision_groups[[label]]
    selected <- candidates[order(
      logical_tests$logical_sequence_id[candidates]
    )[[1L]]]
    fastkpc_full_cuda_prepared_s_add_selection_reason(
      logical_reasons,
      logical_tests$logical_sequence_id[[selected]],
      paste0("first_decision_witness:", label)
    )
  }

  selected_logical_ids <- sort(as.integer(
    fastkpc_full_cuda_prepared_s_selection_ids(logical_reasons)
  ))
  selected_logical_index <- match(
    selected_logical_ids, logical_tests$logical_sequence_id
  )
  if (anyNA(selected_logical_index)) {
    stop("Prepared-S qualification logical selection is invalid",
         call. = FALSE)
  }
  for (index in selected_logical_index) {
    logical_id <- logical_tests$logical_sequence_id[[index]]
    endpoints <- c(
      logical_tests$residual_key_x[[index]],
      logical_tests$residual_key_y[[index]]
    )
    fastkpc_full_cuda_prepared_s_add_selection_reason(
      target_reasons, endpoints,
      paste0("logical_endpoint:", logical_id)
    )
  }

  selected_target_keys <-
    fastkpc_full_cuda_prepared_s_selection_ids(target_reasons)
  selected_target_index <- match(
    selected_target_keys, targets$residual_key_sha256
  )
  selected_group_ids <- sort(unique(
    targets$same_S_group_id[selected_target_index]
  ), method = "radix")
  penalty_distribution <- table(factor(
    targets$penalty_count[selected_target_index],
    levels = c(1L, 3L, 4L, 5L, 6L, 7L)
  ))
  expected_penalty_distribution <-
    c(3327L, 872L, 837L, 730L, 312L, 65L)
  if (length(selected_logical_ids) != 3808L ||
      length(selected_target_keys) != 6143L ||
      length(selected_group_ids) != 2061L ||
      !identical(
        unname(as.integer(penalty_distribution)),
        expected_penalty_distribution
      )) {
    stop(
      "Prepared-S qualification subset count mismatch: setups=",
      length(selected_group_ids), ", targets=", length(selected_target_keys),
      ", logical_tests=", length(selected_logical_ids),
      call. = FALSE
    )
  }

  rare <- with(
    targets,
    rank_deficient | nonfinite_metadata | mgcv_nonconverged |
      (high_condition & penalty_count == 1L)
  )
  if (!all(targets$residual_key_sha256[rare] %in% seed_target_keys) ||
      !all(
        logical_tests$logical_sequence_id[near_alpha] %in%
          selected_logical_ids
      )) {
    stop("Prepared-S qualification rare coverage is incomplete",
         call. = FALSE)
  }
  absent_fields <- c(
    "near_constant_target", "near_constant_conditioner", "mgcv_warning"
  )
  if (any(vapply(absent_fields, function(field) {
    any(targets[[field]] %in% TRUE)
  }, logical(1L)))) {
    stop("Prepared-S canonical absent risk class is unexpectedly present",
         call. = FALSE)
  }

  fastkpc_full_cuda_prepared_s_attach_group_reasons(
    group_store = group_reasons,
    target_store = target_reasons,
    logical_store = logical_reasons,
    targets = targets,
    logical_tests = logical_tests,
    selected_group_ids = selected_group_ids
  )
  selected_targets <- targets[selected_target_index, , drop = FALSE]
  selected_targets$selection_reasons <- I(
    fastkpc_full_cuda_prepared_s_selection_reasons(
      target_reasons, selected_target_keys
    )
  )
  seed_targets <- targets[seed_target_index, , drop = FALSE]
  seed_targets$selection_reasons <- I(seed_reason_lists)
  selected_setups <- setup_rows[
    match(selected_group_ids, setup_rows$same_S_group_id), , drop = FALSE
  ]
  selected_setups$setup_target_count <- group_metadata$setup_target_count[
    match(selected_group_ids, group_metadata$same_S_group_id)
  ]
  selected_setups$setup_logical_request_count <-
    group_metadata$setup_logical_request_count[
      match(selected_group_ids, group_metadata$same_S_group_id)
    ]
  selected_setups$selection_reasons <- I(
    fastkpc_full_cuda_prepared_s_selection_reasons(
      group_reasons, selected_group_ids
    )
  )
  selected_logical <- logical_tests[
    selected_logical_index, , drop = FALSE
  ]
  selected_logical$near_alpha <-
    is.finite(selected_logical$absolute_log_distance_from_alpha) &
    selected_logical$absolute_log_distance_from_alpha <= log(2)
  selected_logical$selection_reasons <- I(
    fastkpc_full_cuda_prepared_s_selection_reasons(
      logical_reasons, as.character(selected_logical_ids)
    )
  )
  rownames(seed_targets) <- NULL
  rownames(selected_targets) <- NULL
  rownames(selected_setups) <- NULL
  rownames(selected_logical) <- NULL

  coverage <- fastkpc_full_cuda_prepared_s_qualification_coverage(
    targets = targets,
    selected_targets = selected_targets,
    logical_tests = logical_tests,
    selected_logical = selected_logical,
    selected_group_ids = selected_group_ids,
    multiplicity_witnesses = multiplicity_witnesses
  )
  required_coverage <- c(
    "risk_class", "condition_bucket", "penalty_count", "S_size",
    "reference_decision", "setup_target_count_quantile"
  )
  if (!all(required_coverage %in% coverage$coverage_type)) {
    stop("Prepared-S qualification coverage rows are incomplete",
         call. = FALSE)
  }

  reason_rows <- rbind(
    fastkpc_full_cuda_prepared_s_selection_reason_rows(
      "setup_group", selected_group_ids,
      selected_setups$selection_reasons
    ),
    fastkpc_full_cuda_prepared_s_selection_reason_rows(
      "target_key", selected_target_keys,
      selected_targets$selection_reasons
    ),
    fastkpc_full_cuda_prepared_s_selection_reason_rows(
      "logical_test", selected_logical_ids,
      selected_logical$selection_reasons
    )
  )
  reason_rows <- reason_rows[order(
    reason_rows$entity_type, reason_rows$entity_id, reason_rows$reason,
    method = "radix"
  ), , drop = FALSE]
  rownames(reason_rows) <- NULL
  qualification_subset_hash <-
    fastkpc_full_cuda_prepared_s_selection_hash(
      schema_version = schema_version,
      setup_ids = selected_group_ids,
      target_keys = selected_target_keys,
      logical_ids = selected_logical_ids,
      reason_rows = reason_rows,
      seed_target_keys = seed_target_keys
    )
  list(
    schema_version = schema_version,
    seed_target_keys = seed_targets,
    setup_groups = selected_setups,
    target_keys = selected_targets,
    logical_tests = selected_logical,
    coverage = coverage,
    multiplicity_witnesses = multiplicity_witnesses,
    reason_rows = reason_rows,
    seed_target_keys_hash = fastkpc_full_cuda_census_metadata_hash(
      seed_target_keys
    ),
    setup_group_ids_hash = fastkpc_full_cuda_census_metadata_hash(
      selected_group_ids
    ),
    target_keys_hash = fastkpc_full_cuda_census_metadata_hash(
      selected_target_keys
    ),
    logical_test_ids_hash = fastkpc_full_cuda_census_metadata_hash(
      selected_logical_ids
    ),
    coverage_hash = fastkpc_full_cuda_census_frame_hash(coverage),
    reason_rows_hash = fastkpc_full_cuda_census_frame_hash(reason_rows),
    qualification_subset_hash = qualification_subset_hash
  )
}

fastkpc_full_cuda_run_prepared_s_target_parity <- function(
    inputs, prepared_by_group, target_keys) {
  if (!is.data.frame(target_keys) ||
      !"residual_key_sha256" %in% names(target_keys)) {
    stop("Prepared-S target parity target keys are incomplete",
         call. = FALSE)
  }
  selected_keys <- fastkpc_full_cuda_prepared_s_selection_character(
    target_keys$residual_key_sha256
  )
  canonical_order <- sort(selected_keys, method = "radix")
  if (length(selected_keys) != 270L || anyNA(selected_keys) ||
      anyDuplicated(selected_keys) ||
      !identical(selected_keys, canonical_order)) {
    stop("Prepared-S target key set/order mismatch", call. = FALSE)
  }
  expected <- fastkpc_full_cuda_select_prepared_s_iteration_subset(inputs)
  expected_keys <- as.character(
    expected$target_keys$residual_key_sha256
  )
  if (!identical(selected_keys, expected_keys)) {
    stop("Prepared-S target key set/order mismatch", call. = FALSE)
  }
  expected_groups <- as.character(
    expected$setup_groups$same_S_group_id
  )
  if (!is.list(prepared_by_group) || is.null(names(prepared_by_group)) ||
      anyNA(names(prepared_by_group)) || anyDuplicated(names(prepared_by_group)) ||
      !identical(names(prepared_by_group), expected_groups)) {
    stop("Prepared-S target parity setup group set/order mismatch",
         call. = FALSE)
  }

  target_metadata <-
    fastkpc_full_cuda_prepared_s_selection_target_rows(inputs)
  target_index <- match(selected_keys, target_metadata$residual_key_sha256)
  if (anyNA(target_index)) {
    stop("Prepared-S target parity lineage is incomplete",
         call. = FALSE)
  }
  target_state_cache <- new.env(hash = TRUE, parent = emptyenv())
  residuals <- new.env(hash = TRUE, parent = emptyenv())
  target_state_build_count <- 0L
  rows <- vector("list", length(selected_keys))

  for (index in seq_along(selected_keys)) {
    key <- selected_keys[[index]]
    metadata_row <- target_metadata[target_index[[index]], , drop = FALSE]
    group_id <- metadata_row$same_S_group_id[[1L]]
    if (!exists(group_id, envir = target_state_cache, inherits = FALSE)) {
      prepared <- prepared_by_group[[group_id]]
      if (is.null(prepared)) {
        stop("Prepared-S target parity setup is missing", call. = FALSE)
      }
      states <- fastkpc_full_cuda_build_target_states(inputs, prepared)
      assign(group_id, states, envir = target_state_cache)
      target_state_build_count <- target_state_build_count + 1L
    }
    states <- get(group_id, envir = target_state_cache, inherits = FALSE)
    state_index <- match(key, states$residual_key_sha256)
    if (is.na(state_index)) {
      stop("Prepared-S target parity TargetState is missing",
           call. = FALSE)
    }
    state <- states[state_index, , drop = FALSE]
    prepared <- prepared_by_group[[group_id]]
    materialized <- fastkpc_full_cuda_materialize_target_state(
      state_row = state,
      data = inputs$data,
      dataset_sha256 = inputs$dataset_sha256
    )
    start <- proc.time()[["elapsed"]]
    solved <- fastkpc_mgcv_magic_fixed_sp_from_prepared(
      prepared_setup = prepared,
      target_state = materialized
    )
    solve_time_ms <- 1000 * (proc.time()[["elapsed"]] - start)
    coefficient_hash <- fastkpc_full_cuda_census_metadata_hash(
      solved$coefficients
    )
    fitted_hash <- fastkpc_full_cuda_census_metadata_hash(
      solved$fitted
    )
    residual_hash <- fastkpc_full_cuda_census_metadata_hash(
      solved$residuals
    )
    coefficient_hash_exact <- identical(
      coefficient_hash, state$coefficient_hash[[1L]]
    )
    fitted_hash_exact <- identical(
      fitted_hash, state$fitted_hash[[1L]]
    )
    residual_hash_exact <- identical(
      residual_hash, state$residual_hash[[1L]]
    )
    assign(key, solved$residuals, envir = residuals)
    rows[[index]] <- data.frame(
      residual_key_sha256 = key,
      same_S_group_id = group_id,
      target = as.integer(state$target[[1L]]),
      prepared_s_key_sha256 = solved$prepared_s_key_sha256,
      target_state_fingerprint =
        state$target_state_fingerprint[[1L]],
      coefficient_hash = coefficient_hash,
      expected_coefficient_hash = state$coefficient_hash[[1L]],
      coefficient_hash_exact = coefficient_hash_exact,
      fitted_hash = fitted_hash,
      expected_fitted_hash = state$fitted_hash[[1L]],
      fitted_hash_exact = fitted_hash_exact,
      residual_hash = residual_hash,
      expected_residual_hash = state$residual_hash[[1L]],
      residual_hash_exact = residual_hash_exact,
      residual_length = as.integer(length(solved$residuals)),
      solve_time_ms = as.numeric(solve_time_ms),
      stringsAsFactors = FALSE
    )
  }
  parity_rows <- do.call(rbind, rows)
  rownames(parity_rows) <- NULL
  exact <- parity_rows$coefficient_hash_exact &
    parity_rows$fitted_hash_exact & parity_rows$residual_hash_exact
  if (!all(exact)) {
    stop(
      "Prepared-S target parity hash mismatch: ",
      paste(parity_rows$residual_key_sha256[!exact], collapse = ","),
      call. = FALSE
    )
  }
  cache_group_count <- as.integer(length(ls(
    target_state_cache, all.names = TRUE
  )))
  if (!identical(target_state_build_count, 44L) ||
      !identical(cache_group_count, 44L) ||
      length(ls(residuals, all.names = TRUE)) != 270L) {
    stop("Prepared-S target parity cache count mismatch", call. = FALSE)
  }
  list(
    schema_version = "full-cuda-ci-prepared-s-target-parity-v1",
    rows = parity_rows,
    residuals = residuals,
    target_state_build_count = as.integer(target_state_build_count),
    target_state_cache_group_count = cache_group_count,
    exact_target_count = as.integer(sum(exact)),
    target_keys_hash = fastkpc_full_cuda_census_metadata_hash(
      selected_keys
    ),
    parity_rows_hash = fastkpc_full_cuda_census_frame_hash(parity_rows)
  )
}

fastkpc_full_cuda_prepared_s_restore_environment <- function(
    name, value) {
  if (is.na(value)) {
    Sys.unsetenv(name)
  } else {
    do.call(Sys.setenv, setNames(list(value), name))
  }
  invisible(TRUE)
}

fastkpc_full_cuda_prepared_s_dcov_oracle_identity <- function(
    inputs, logical_tests, oracle_manifest = NULL) {
  alpha_values <- sort(unique(as.numeric(logical_tests$alpha)))
  if (length(alpha_values) != 1L || !is.finite(alpha_values) ||
      !identical(alpha_values, 0.1)) {
    stop("Prepared-S dCov oracle alpha identity mismatch",
         call. = FALSE)
  }
  expected <- list(
    index = 1L,
    numCol = 35L,
    dataset_sha256 = as.character(inputs$dataset_sha256),
    alpha = alpha_values[[1L]],
    source = "pinned-phase0-contract"
  )
  if (is.null(oracle_manifest)) return(expected)
  if (is.character(oracle_manifest) && length(oracle_manifest) == 1L) {
    if (!file.exists(oracle_manifest)) {
      stop("Prepared-S dCov oracle manifest is unavailable",
           call. = FALSE)
    }
    fastkpc_full_cuda_require_namespace("jsonlite")
    oracle_manifest <- jsonlite::read_json(
      oracle_manifest, simplifyVector = TRUE
    )
  }
  if (!is.list(oracle_manifest)) {
    stop("Prepared-S dCov oracle manifest is invalid", call. = FALSE)
  }
  manifest_hash <- if (!is.null(oracle_manifest$data_hash)) {
    as.character(oracle_manifest$data_hash)
  } else {
    as.character(oracle_manifest$dataset_sha256)
  }
  clean <- identical(as.integer(oracle_manifest$index), 1L) &&
    identical(as.integer(oracle_manifest$numCol), 35L) &&
    identical(as.numeric(oracle_manifest$alpha), alpha_values[[1L]]) &&
    identical(manifest_hash, expected$dataset_sha256)
  if (!isTRUE(clean)) {
    stop("Prepared-S dCov oracle manifest identity mismatch",
         call. = FALSE)
  }
  expected$source <- "authenticated-oracle-manifest"
  expected
}

fastkpc_full_cuda_prepared_s_dcov_diagnostic_integer <- function(
    diagnostics, field) {
  if (!field %in% names(diagnostics)) {
    stop(
      "Prepared-S dCov Spectra diagnostics mismatch: missing ", field,
      call. = FALSE
    )
  }
  value <- diagnostics[[field]]
  if (!is.numeric(value) || length(value) != 1L || !is.finite(value) ||
      value != floor(value)) {
    stop(
      "Prepared-S dCov Spectra diagnostics mismatch: invalid ", field,
      call. = FALSE
    )
  }
  as.integer(value)
}

fastkpc_full_cuda_prepared_s_validate_dcov_spectra_diagnostics <- function(
    diagnostics, numCol) {
  numCol <- as.integer(numCol)
  if (!is.list(diagnostics) || is.null(names(diagnostics)) ||
      length(numCol) != 1L || is.na(numCol) || numCol < 1L) {
    stop("Prepared-S dCov Spectra diagnostics mismatch: malformed",
         call. = FALSE)
  }
  mode <- diagnostics$lowrank_mode
  if (!is.character(mode) || length(mode) != 1L || is.na(mode) ||
      !identical(mode, "spectra")) {
    stop(
      "Prepared-S dCov Spectra diagnostics mismatch: lowrank_mode",
      call. = FALSE
    )
  }
  values <- c(
    lowrank_full_eig_count =
      fastkpc_full_cuda_prepared_s_dcov_diagnostic_integer(
        diagnostics, "lowrank_full_eig_count"
      ),
    lowrank_spectra_count =
      fastkpc_full_cuda_prepared_s_dcov_diagnostic_integer(
        diagnostics, "lowrank_spectra_count"
      ),
    lowrank_spectra_converged_count =
      fastkpc_full_cuda_prepared_s_dcov_diagnostic_integer(
        diagnostics, "lowrank_spectra_converged_count"
      ),
    lowrank_spectra_failed_count =
      fastkpc_full_cuda_prepared_s_dcov_diagnostic_integer(
        diagnostics, "lowrank_spectra_failed_count"
      ),
    lowrank_spectra_fallback_full_eig_count =
      fastkpc_full_cuda_prepared_s_dcov_diagnostic_integer(
        diagnostics, "lowrank_spectra_fallback_full_eig_count"
      ),
    lowrank_spectra_nconv =
      fastkpc_full_cuda_prepared_s_dcov_diagnostic_integer(
        diagnostics, "lowrank_spectra_nconv"
      )
  )
  expected <- c(
    lowrank_full_eig_count = 0L,
    lowrank_spectra_count = 2L,
    lowrank_spectra_converged_count = 2L,
    lowrank_spectra_failed_count = 0L,
    lowrank_spectra_fallback_full_eig_count = 0L
  )
  mismatch <- names(expected)[
    values[names(expected)] != expected
  ]
  if (length(mismatch) > 0L) {
    stop(
      "Prepared-S dCov Spectra diagnostics mismatch: ",
      paste(mismatch, collapse = ","),
      call. = FALSE
    )
  }
  if (values[["lowrank_spectra_nconv"]] < 2L * numCol) {
    stop(
      paste0(
        "Prepared-S dCov Spectra diagnostics mismatch: ",
        "lowrank_spectra_nconv"
      ),
      call. = FALSE
    )
  }
  invisible(TRUE)
}

fastkpc_full_cuda_run_prepared_s_dcov_parity <- function(
    inputs, logical_tests, residuals, oracle_manifest = NULL,
    oracle_fun = fastkpc_legacy_dcov_gamma_cpp_oracle, ...) {
  environment_name <- "FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK"
  prior_environment <- Sys.getenv(
    environment_name, unset = NA_character_
  )
  on.exit(
    fastkpc_full_cuda_prepared_s_restore_environment(
      environment_name, prior_environment
    ),
    add = TRUE
  )
  Sys.setenv(FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = "spectra")

  dots <- list(...)
  if (length(dots) > 0L) {
    stop("Prepared-S dCov parity options are unsupported",
         call. = FALSE)
  }
  if (!is.environment(residuals) || !is.data.frame(logical_tests) ||
      !"logical_sequence_id" %in% names(logical_tests)) {
    stop("Prepared-S dCov parity inputs are incomplete", call. = FALSE)
  }
  if (!is.function(oracle_fun)) {
    stop("Prepared-S dCov parity oracle function is invalid",
         call. = FALSE)
  }
  logical_tests <- fastkpc_full_cuda_prepared_s_selection_normalize_fields(
    logical_tests,
    character_fields = c(
      "reference_decision", "residual_key_x", "residual_key_y"
    ),
    integer_fields = c("logical_sequence_id", "S_size"),
    numeric_fields = c(
      "reference_p_value", "alpha",
      "absolute_log_distance_from_alpha"
    ),
    logical_fields = "reference_independent"
  )
  selected_ids <- logical_tests$logical_sequence_id
  if (length(selected_ids) != 44L || anyNA(selected_ids) ||
      anyDuplicated(selected_ids) ||
      !identical(selected_ids, sort(selected_ids))) {
    stop("Prepared-S dCov logical test set/order mismatch",
         call. = FALSE)
  }
  expected <- fastkpc_full_cuda_select_prepared_s_iteration_subset(inputs)
  expected_logical <- expected$logical_tests
  expected_fields <- c(
    "logical_sequence_id", "residual_key_x", "residual_key_y",
    "reference_p_value", "alpha", "reference_decision",
    "reference_independent"
  )
  if (!identical(
        logical_tests[expected_fields],
        expected_logical[expected_fields]
      )) {
    stop("Prepared-S dCov logical test set/order mismatch",
         call. = FALSE)
  }
  oracle_identity <- fastkpc_full_cuda_prepared_s_dcov_oracle_identity(
    inputs = inputs,
    logical_tests = logical_tests,
    oracle_manifest = oracle_manifest
  )
  if (!is.matrix(inputs$data) || nrow(inputs$data) < 1L ||
      !identical(
        fastkpc_full_cuda_data_hash(inputs$data),
        oracle_identity$dataset_sha256
      )) {
    stop("Prepared-S dCov dataset identity mismatch", call. = FALSE)
  }

  endpoint_keys <- unique(c(
    logical_tests$residual_key_x, logical_tests$residual_key_y
  ))
  missing <- endpoint_keys[!vapply(endpoint_keys, function(key) {
    exists(key, envir = residuals, inherits = FALSE)
  }, logical(1L))]
  if (length(missing) > 0L) {
    stop(
      "Prepared-S dCov parity residual is missing: ",
      paste(sort(missing, method = "radix"), collapse = ","),
      call. = FALSE
    )
  }
  for (key in endpoint_keys) {
    value <- get(key, envir = residuals, inherits = FALSE)
    if (!is.numeric(value) || length(value) != nrow(inputs$data) ||
        any(!is.finite(value))) {
      stop("Prepared-S dCov parity residual is invalid: ", key,
           call. = FALSE)
    }
  }

  rows <- vector("list", nrow(logical_tests))
  for (row_index in seq_len(nrow(logical_tests))) {
    logical_row <- logical_tests[row_index, , drop = FALSE]
    key_x <- logical_row$residual_key_x[[1L]]
    key_y <- logical_row$residual_key_y[[1L]]
    residual_x <- get(key_x, envir = residuals, inherits = FALSE)
    residual_y <- get(key_y, envir = residuals, inherits = FALSE)
    start <- proc.time()[["elapsed"]]
    oracle <- oracle_fun(
      residual_x,
      residual_y,
      numCol = oracle_identity$numCol,
      index = oracle_identity$index
    )
    elapsed_ms <- 1000 * (proc.time()[["elapsed"]] - start)
    if (!is.list(oracle) || !is.list(oracle$diagnostics) ||
        length(oracle$diagnostics) == 0L) {
      stop("Prepared-S dCov oracle result is invalid", call. = FALSE)
    }
    fastkpc_full_cuda_prepared_s_validate_dcov_spectra_diagnostics(
      diagnostics = oracle$diagnostics,
      numCol = oracle_identity$numCol
    )
    if (!is.numeric(oracle$p.value) || length(oracle$p.value) != 1L ||
        !is.finite(oracle$p.value)) {
      stop("Prepared-S dCov oracle result is invalid", call. = FALSE)
    }
    p_value <- as.numeric(oracle$p.value)
    reference_p_value <- as.numeric(
      logical_row$reference_p_value[[1L]]
    )
    alpha <- as.numeric(logical_row$alpha[[1L]])
    reference_independent <- reference_p_value >= alpha
    reference_decision <- if (reference_independent) {
      "independent"
    } else {
      "dependent"
    }
    if (!identical(
          logical_row$reference_independent[[1L]],
          reference_independent
        ) || !identical(
          logical_row$reference_decision[[1L]], reference_decision
        )) {
      stop("Prepared-S dCov reference decision semantics mismatch",
           call. = FALSE)
    }
    independent <- p_value >= alpha
    p_value_drift <- p_value - reference_p_value
    result_row <- data.frame(
      logical_sequence_id = logical_row$logical_sequence_id[[1L]],
      residual_key_x = key_x,
      residual_key_y = key_y,
      index = as.integer(oracle_identity$index),
      numCol = as.integer(oracle_identity$numCol),
      alpha = alpha,
      reference_p_value = reference_p_value,
      p_value = p_value,
      p_value_drift = p_value_drift,
      absolute_p_value_drift = abs(p_value_drift),
      p_value_exact = identical(p_value, reference_p_value),
      reference_signed_alpha_distance = reference_p_value - alpha,
      signed_alpha_distance = p_value - alpha,
      reference_decision = reference_decision,
      decision = if (independent) "independent" else "dependent",
      decision_identical = identical(independent, reference_independent),
      elapsed_ms = as.numeric(elapsed_ms),
      stringsAsFactors = FALSE
    )
    result_row$diagnostics <- I(list(oracle$diagnostics))
    rows[[row_index]] <- result_row
  }
  parity_rows <- do.call(rbind, rows)
  rownames(parity_rows) <- NULL
  max_drift <- max(parity_rows$absolute_p_value_drift)
  exact_count <- as.integer(sum(parity_rows$p_value_exact))
  flip_count <- as.integer(sum(!parity_rows$decision_identical))
  if (!is.finite(max_drift) || max_drift > 1e-12) {
    stop(
      "Prepared-S dCov p-value drift exceeds 1e-12: ",
      format(max_drift, scientific = TRUE),
      call. = FALSE
    )
  }
  if (flip_count != 0L) {
    stop("Prepared-S dCov decision flip detected", call. = FALSE)
  }
  list(
    schema_version = "full-cuda-ci-prepared-s-dcov-parity-v1",
    oracle_identity = oracle_identity,
    rows = parity_rows,
    max_absolute_p_value_drift = max_drift,
    exact_p_value_count = exact_count,
    decision_flip_count = flip_count,
    logical_test_ids_hash = fastkpc_full_cuda_census_metadata_hash(
      selected_ids
    ),
    parity_rows_hash = fastkpc_full_cuda_census_frame_hash(parity_rows)
  )
}
