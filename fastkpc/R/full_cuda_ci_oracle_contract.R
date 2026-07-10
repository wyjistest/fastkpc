source("fastkpc/R/full_cuda_ci_gate.R")

fastkpc_full_cuda_census_hash_raw <- function(value) {
  fastkpc_full_cuda_require_namespace("digest")
  digest::digest(value, algo = "sha256", serialize = FALSE)
}

fastkpc_full_cuda_census_hash_utf8 <- function(value) {
  fastkpc_full_cuda_census_hash_raw(charToRaw(enc2utf8(value)))
}

fastkpc_full_cuda_census_file_hash <- function(path) {
  fastkpc_full_cuda_require_namespace("digest")
  digest::digest(file = path, algo = "sha256", serialize = FALSE)
}

fastkpc_full_cuda_census_input_contract <- function() {
  list(
    schema_version = "full-cuda-ci-phase1-input-v2",
    phase0_source_commit =
      "93ae8430aa24ef4458f6ae62451982fb04bab804",
    dataset_matrix_sha256 =
      "971f1e0784817c6febfc84154f19540be02a92f6e4acad784dc7f92b979f1df7",
    oracle_input_bundle_sha256 =
      "7700bc78240984c36f8ae5ca281362a0afb8d7dedd34a5711ce4ab76a2ebee0e",
    canonical_key_corpus_hash =
      "b843630969f116da63f7fad095c54de2ff471540159ff97ca56c3871d6b2e1fa",
    canonical_logical_census_hash =
      "c9b48074dd59a439fceb9d5e64806adda5620cc4abe32095371abc447ef98634",
    file_hashes = c(
      "dataset/cancer_RD-causalDiscoveryInput.rds" =
        "e03cbfafed3336f3da7725878e92af24556b5ebdc6227b94ea84469e62e94036",
      "oracle/adjacency.rds" =
        "6701a033e821f8433842ae825f67715b6c2349e3c36515b45296f125ff7e1d4e",
      "oracle/deletion_trace.csv" =
        "00eeb3fe843e9f868b133cef9f573430ea9d49fe7b0dd53ba3be51e2e9e94486",
      "oracle/fallbacks.csv" =
        "dc45430a89ad1c4fb85cf9f4c63b4babe89df8cf2b505759e7389f7b5462beca",
      "oracle/first_divergence.json" =
        "b29373f20a56d99ce76ef4c21f2d508a13ed22ed009a30c902a1e6d23e46fef6",
      "oracle/graph_agreement.csv" =
        "f1eabdbc578607ee0b124a6dcd3fd8250077493472605583889eb532cd3b85d5",
      "oracle/logical_ci_trace.rds" =
        "b777c5dc1b9acad08c133ccd668eae5e9444a89d481c7f8adf9b2c2a0dd6cda5",
      "oracle/manifest.json" =
        "f907559586c4b766f483bdc01b4074d93ce2c8b80972c3199c4493848b2b8750",
      "oracle/n_edgetests.csv" =
        "6c0e1ccb14c9721e7056aa91e851065877965ab33869867dd130c7ca3d503058",
      "oracle/pmax.rds" =
        "2fafe1f5084dcb86114adfb86d06855350d36872e59795b6ee604bf4c6e19df5",
      "oracle/sepset_agreement.csv" =
        "9e57978d03fa0e62526b884e0566d2671d2c74417f117fd420b0fb4afb85a256",
      "oracle/sepsets.rds" =
        "69853449f95e1486ef237a2b1bd7c3a99d94cac4c0f202d7c509c890a49e1ca6",
      "oracle/summary.json" =
        "eec6724d9fd69671399783b565c2dd8bbdbc3a4e553ba742ac781d483354ade7"
    )
  )
}

fastkpc_full_cuda_census_bundle_payload <- function(file_hashes) {
  file_hashes <- file_hashes[order(names(file_hashes), method = "radix")]
  paste0(
    paste0(names(file_hashes), "\t", unname(file_hashes), collapse = "\n"),
    "\n"
  )
}

fastkpc_full_cuda_census_input_paths <- function(oracle_dir, data_path) {
  c(
    "dataset/cancer_RD-causalDiscoveryInput.rds" = data_path,
    "oracle/adjacency.rds" = file.path(oracle_dir, "adjacency.rds"),
    "oracle/deletion_trace.csv" = file.path(oracle_dir,
                                             "deletion_trace.csv"),
    "oracle/fallbacks.csv" = file.path(oracle_dir, "fallbacks.csv"),
    "oracle/first_divergence.json" = file.path(oracle_dir,
                                                "first_divergence.json"),
    "oracle/graph_agreement.csv" = file.path(oracle_dir,
                                              "graph_agreement.csv"),
    "oracle/logical_ci_trace.rds" = file.path(oracle_dir,
                                               "logical_ci_trace.rds"),
    "oracle/manifest.json" = file.path(oracle_dir, "manifest.json"),
    "oracle/n_edgetests.csv" = file.path(oracle_dir, "n_edgetests.csv"),
    "oracle/pmax.rds" = file.path(oracle_dir, "pmax.rds"),
    "oracle/sepset_agreement.csv" = file.path(oracle_dir,
                                               "sepset_agreement.csv"),
    "oracle/sepsets.rds" = file.path(oracle_dir, "sepsets.rds"),
    "oracle/summary.json" = file.path(oracle_dir, "summary.json")
  )
}

fastkpc_full_cuda_census_validate_input_hashes <- function(
    oracle_dir, data_path,
    contract = fastkpc_full_cuda_census_input_contract()) {
  paths <- fastkpc_full_cuda_census_input_paths(oracle_dir, data_path)
  if (!identical(sort(names(paths), method = "radix"),
                 sort(names(contract$file_hashes), method = "radix"))) {
    stop("Phase 1 input contract path set mismatch", call. = FALSE)
  }
  missing <- names(paths)[!file.exists(paths)]
  if (length(missing) > 0L) {
    stop("Phase 1 input is missing: ", paste(missing, collapse = ","),
         call. = FALSE)
  }
  actual <- vapply(paths, fastkpc_full_cuda_census_file_hash, character(1L))
  expected <- unname(contract$file_hashes[names(paths)])
  identical_hash <- actual == expected
  table <- data.frame(
    logical_path = names(paths),
    path = unname(paths),
    expected_sha256 = expected,
    actual_sha256 = unname(actual),
    identical = identical_hash,
    stringsAsFactors = FALSE
  )
  if (!all(identical_hash)) {
    stop("Phase 1 input hash mismatch: ",
         paste(table$logical_path[!identical_hash], collapse = ","),
         call. = FALSE)
  }
  bundle_hash <- fastkpc_full_cuda_census_hash_utf8(
    fastkpc_full_cuda_census_bundle_payload(actual)
  )
  if (!identical(bundle_hash, contract$oracle_input_bundle_sha256)) {
    stop("Phase 1 oracle input bundle hash mismatch", call. = FALSE)
  }
  attr(table, "oracle_input_bundle_sha256") <- bundle_hash
  table
}

fastkpc_full_cuda_census_validate_semantic_inputs <- function(
    data, oracle, oracle_dir, contract) {
  fastkpc_full_cuda_require_namespace("digest")
  canonical <- fastkpc_full_cuda_canonical_contract()
  manifest <- oracle$manifest
  manifest_clean <-
    identical(contract$dataset_matrix_sha256, canonical$data_hash) &&
    identical(as.character(manifest$schema_version),
              "full-cuda-ci-oracle-v1") &&
    identical(as.character(manifest$source_commit),
              contract$phase0_source_commit) &&
    identical(as.character(manifest$source_result_hash),
              canonical$source_result_hash) &&
    identical(as.character(manifest$data_hash), canonical$data_hash) &&
    identical(as.integer(manifest$data_dimensions$n), canonical$n) &&
    identical(as.integer(manifest$data_dimensions$p), canonical$p) &&
    identical(as.character(manifest$column_order), canonical$column_order) &&
    identical(as.numeric(manifest$alpha), canonical$alpha) &&
    identical(as.integer(manifest$max_conditioning_size),
              canonical$max_conditioning_size) &&
    identical(as.integer(manifest$index), canonical$index) &&
    identical(as.integer(manifest$numCol), canonical$numCol) &&
    isTRUE(manifest$logical_ci_trace_available) &&
    identical(as.integer(manifest$logical_ci_trace_count), 240489L) &&
    identical(as.integer(manifest$deletion_trace_count), 1018L) &&
    nrow(oracle$logical_trace) == 240489L &&
    nrow(oracle$deletion_trace) == 1018L
  if (!isTRUE(manifest_clean)) {
    stop("Phase 1 oracle manifest semantic contract mismatch",
         call. = FALSE)
  }
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  if (!identical(dim(data), c(canonical$n, canonical$p)) ||
      !identical(colnames(data), canonical$column_order) ||
      !identical(fastkpc_full_cuda_data_hash(data),
                 contract$dataset_matrix_sha256)) {
    stop("Phase 1 canonical data semantic contract mismatch",
         call. = FALSE)
  }
  labels <- canonical$column_order
  adjacency <- fastkpc_full_cuda_align_matrix(
    oracle$reference$adjacency, labels
  )
  if (is.null(adjacency) ||
      !identical(fastkpc_full_cuda_edge_count(adjacency),
                 canonical$edge_count) ||
      !identical(digest::digest(adjacency, algo = "sha256",
                               serialize = TRUE),
                 canonical$adjacency_hash)) {
    stop("Phase 1 inherited adjacency semantic contract mismatch",
         call. = FALSE)
  }
  normalized_sepsets <- fastkpc_full_cuda_normalize_sepsets(
    oracle$reference, labels
  )
  if (!identical(digest::digest(normalized_sepsets, algo = "sha256",
                               serialize = TRUE),
                 canonical$sepset_hash)) {
    stop("Phase 1 inherited sepset semantic contract mismatch",
         call. = FALSE)
  }
  deletion_trace <- utils::read.csv(
    file.path(oracle_dir, "deletion_trace.csv"),
    stringsAsFactors = FALSE
  )
  deletion_trace$p_value <- as.numeric(deletion_trace$p_value)
  if (!identical(digest::digest(deletion_trace, algo = "sha256",
                               serialize = TRUE),
                 canonical$deletion_trace_hash)) {
    stop("Phase 1 inherited deletion semantic contract mismatch",
         call. = FALSE)
  }
  if (!identical(as.integer(oracle$reference$n.edgetests),
                 as.integer(canonical$n_edgetests))) {
    stop("Phase 1 inherited n.edgetests semantic contract mismatch",
         call. = FALSE)
  }
  invisible(TRUE)
}

fastkpc_full_cuda_census_validate_inherited_evidence <- function(
    oracle, oracle_dir,
    contract = fastkpc_full_cuda_census_input_contract()) {
  fastkpc_full_cuda_require_namespace("jsonlite")
  canonical <- fastkpc_full_cuda_canonical_contract()
  summary <- oracle$summary
  fallback_counts <- c(
    unknown = as.integer(summary$unknown_fallback_count),
    approximate = as.integer(summary$approximate_backend_count),
    backend = as.integer(summary$backend_fallback_error_count)
  )
  if (length(fallback_counts) != 3L || anyNA(fallback_counts) ||
      any(fallback_counts != 0L)) {
    stop("Phase 1 inherited fallback counters are missing or nonzero",
         call. = FALSE)
  }
  edge_counts <- c(
    reference = as.integer(summary$edge_count_reference),
    candidate = as.integer(summary$edge_count_candidate)
  )
  if (length(edge_counts) != 2L || anyNA(edge_counts) ||
      any(edge_counts != canonical$edge_count)) {
    stop("Phase 1 inherited edge count evidence is invalid", call. = FALSE)
  }
  summary_clean <-
    identical(as.character(summary$run_status), "ok") &&
    !isTRUE(summary$timeout) &&
    identical(as.character(summary$source_commit),
              contract$phase0_source_commit) &&
    identical(as.character(summary$candidate_route), "oracle-self") &&
    identical(as.integer(summary$SHD), 0L) &&
    isTRUE(summary$adjacency_identical) &&
    isTRUE(summary$sepsets_identical) &&
    isTRUE(summary$n_edgetests_identical) &&
    isTRUE(summary$deletions_identical) &&
    isTRUE(summary$logical_ci_trace_identical) &&
    isTRUE(summary$deleting_test_identical) &&
    isTRUE(summary$pass)
  if (!isTRUE(summary_clean)) {
    stop("Phase 1 inherited summary evidence is invalid", call. = FALSE)
  }

  graph <- utils::read.csv(file.path(oracle_dir, "graph_agreement.csv"),
                           stringsAsFactors = FALSE)
  graph_clean <- nrow(graph) == 1L &&
    identical(as.integer(graph$edge_count_reference), canonical$edge_count) &&
    identical(as.integer(graph$edge_count_candidate), canonical$edge_count) &&
    identical(as.integer(graph$SHD), 0L) &&
    isTRUE(graph$adjacency_identical)
  if (!isTRUE(graph_clean)) {
    stop("Phase 1 inherited graph agreement evidence is invalid",
         call. = FALSE)
  }

  sepsets <- utils::read.csv(
    file.path(oracle_dir, "sepset_agreement.csv"),
    stringsAsFactors = FALSE
  )
  if (nrow(sepsets) == 0L || !all(sepsets$identical %in% TRUE) ||
      any(sepsets$direction_conflict %in% TRUE)) {
    stop("Phase 1 inherited sepset agreement evidence is invalid",
         call. = FALSE)
  }

  tests <- utils::read.csv(file.path(oracle_dir, "n_edgetests.csv"),
                           stringsAsFactors = FALSE)
  tests_clean <- identical(as.integer(tests$level),
                            seq_along(canonical$n_edgetests) - 1L) &&
    identical(as.integer(tests$n_edgetests),
              as.integer(canonical$n_edgetests)) &&
    identical(as.integer(oracle$reference$n.edgetests),
              as.integer(canonical$n_edgetests))
  if (!isTRUE(tests_clean)) {
    stop("Phase 1 inherited n.edgetests evidence is invalid",
         call. = FALSE)
  }

  fallbacks <- utils::read.csv(file.path(oracle_dir, "fallbacks.csv"),
                               stringsAsFactors = FALSE)
  required_fallbacks <- c("unknown_fallback_count",
                          "approximate_backend_count")
  fallback_file_clean <- nrow(fallbacks) >= 2L &&
    all(required_fallbacks %in% fallbacks$key) &&
    all(is.finite(as.numeric(fallbacks$count))) &&
    all(as.numeric(fallbacks$count) == 0)
  if (!isTRUE(fallback_file_clean)) {
    stop("Phase 1 inherited fallback file evidence is invalid",
         call. = FALSE)
  }

  first <- jsonlite::read_json(
    file.path(oracle_dir, "first_divergence.json"),
    simplifyVector = TRUE
  )
  if (!identical(first$first_divergence_found, FALSE)) {
    stop("Phase 1 inherited first-divergence evidence is invalid",
         call. = FALSE)
  }
  TRUE
}

fastkpc_full_cuda_census_load_inputs <- function(
    oracle_dir, data_path,
    contract = fastkpc_full_cuda_census_input_contract()) {
  hashes <- fastkpc_full_cuda_census_validate_input_hashes(
    oracle_dir, data_path, contract
  )
  fastkpc_full_cuda_require_namespace("jsonlite")
  oracle <- fastkpc_load_full_cuda_ci_oracle(oracle_dir)
  data <- readRDS(data_path)
  fastkpc_full_cuda_census_validate_semantic_inputs(
    data, oracle, oracle_dir, contract
  )
  fastkpc_full_cuda_census_validate_inherited_evidence(
    oracle, oracle_dir, contract
  )
  list(
    data = as.matrix(data),
    oracle = oracle,
    oracle_input_hashes = hashes,
    oracle_input_bundle_sha256 =
      attr(hashes, "oracle_input_bundle_sha256", exact = TRUE),
    oracle_inherited_graph_gate = TRUE,
    new_candidate_graph_gate = "NOT_APPLICABLE"
  )
}
