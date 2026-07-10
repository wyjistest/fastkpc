source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/mgcv_compat_contract.R")

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
    schema_version = "full-cuda-ci-phase1-input-v1",
    phase0_source_commit =
      "93ae8430aa24ef4458f6ae62451982fb04bab804",
    dataset_matrix_sha256 =
      "971f1e0784817c6febfc84154f19540be02a92f6e4acad784dc7f92b979f1df7",
    oracle_input_bundle_sha256 =
      "7700bc78240984c36f8ae5ca281362a0afb8d7dedd34a5711ce4ab76a2ebee0e",
    canonical_key_corpus_hash =
      "b843630969f116da63f7fad095c54de2ff471540159ff97ca56c3871d6b2e1fa",
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
  canonical <- fastkpc_full_cuda_canonical_contract()
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
  manifest <- oracle$manifest
  summary <- oracle$summary
  graph <- utils::read.csv(file.path(oracle_dir, "graph_agreement.csv"),
                           stringsAsFactors = FALSE)
  sepsets <- utils::read.csv(
    file.path(oracle_dir, "sepset_agreement.csv"),
    stringsAsFactors = FALSE
  )
  tests <- utils::read.csv(file.path(oracle_dir, "n_edgetests.csv"),
                           stringsAsFactors = FALSE)
  fallbacks <- utils::read.csv(file.path(oracle_dir, "fallbacks.csv"),
                               stringsAsFactors = FALSE)
  first <- jsonlite::read_json(
    file.path(oracle_dir, "first_divergence.json"),
    simplifyVector = TRUE
  )
  inherited_clean <-
    identical(as.character(manifest$source_commit),
              contract$phase0_source_commit) &&
    isTRUE(summary$pass) && identical(as.integer(summary$SHD), 0L) &&
    isTRUE(summary$adjacency_identical) &&
    isTRUE(summary$sepsets_identical) &&
    isTRUE(summary$n_edgetests_identical) &&
    isTRUE(summary$deletions_identical) &&
    isTRUE(summary$logical_ci_trace_identical) &&
    nrow(graph) == 1L && identical(as.integer(graph$SHD), 0L) &&
    all(graph$adjacency_identical %in% TRUE) &&
    nrow(sepsets) > 0L && all(sepsets$identical %in% TRUE) &&
    !any(sepsets$direction_conflict %in% TRUE) &&
    identical(as.integer(tests$n_edgetests),
              as.integer(oracle$reference$n.edgetests)) &&
    nrow(fallbacks) > 0L && all(as.numeric(fallbacks$count) == 0) &&
    !isTRUE(first$first_divergence_found)
  if (!isTRUE(inherited_clean)) {
    stop("Phase 1 inherited oracle graph gate failed", call. = FALSE)
  }
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

fastkpc_full_cuda_census_parse_s <- function(S_key) {
  if (length(S_key) != 1L || is.na(S_key)) {
    stop("conditioning set key must be one non-missing string",
         call. = FALSE)
  }
  if (!nzchar(S_key)) return(integer())
  values <- suppressWarnings(as.integer(
    strsplit(S_key, "|", fixed = TRUE)[[1L]]
  ))
  if (anyNA(values) ||
      !identical(values, sort(unique(values), method = "radix"))) {
    stop("conditioning set key is not canonical sorted unique",
         call. = FALSE)
  }
  values
}

fastkpc_full_cuda_census_residual_payload <- function(
    target, S, formula_class, data_hash, n, p) {
  fields <- c(
    "schema_version=full-cuda-ci-residual-key-v1",
    paste0("dataset_sha256=", data_hash),
    paste0("n=", as.integer(n)),
    paste0("p=", as.integer(p)),
    paste0("target_index=", as.integer(target)),
    paste0("sorted_S=", paste(as.integer(S), collapse = ",")),
    paste0("formula_class=", formula_class),
    "family=gaussian",
    "link=identity",
    "method=GCV.Cp",
    "optimizer=mgcv-default",
    "gamma=1",
    "select=false",
    "scale=mgcv-default",
    "weights=none",
    "offset=none",
    "formula_semantics_version=kpcalg_regrXonS_v1",
    "mgcv_semantics_version=legacy-mgcv-gam-default-selection-v1"
  )
  paste0(paste(fields, collapse = "\n"), "\n")
}

fastkpc_full_cuda_census_same_s_payload <- function(
    S, formula_class, data_hash, n, p) {
  fields <- c(
    "schema_version=full-cuda-ci-same-s-key-v1",
    paste0("dataset_sha256=", data_hash),
    paste0("n=", as.integer(n)),
    paste0("p=", as.integer(p)),
    paste0("sorted_S=", paste(as.integer(S), collapse = ",")),
    paste0("formula_class=", formula_class),
    "family=gaussian",
    "link=identity",
    "method=GCV.Cp",
    "optimizer=mgcv-default",
    "gamma=1",
    "select=false",
    "scale=mgcv-default",
    "weights=none",
    "offset=none",
    "formula_semantics_version=kpcalg_regrXonS_v1",
    "mgcv_semantics_version=legacy-mgcv-gam-default-selection-v1"
  )
  paste0(paste(fields, collapse = "\n"), "\n")
}

fastkpc_full_cuda_census_request_identity <- function(
    target, S_key, formula_class) {
  paste(as.integer(target), as.character(S_key),
        as.character(formula_class), sep = "\t")
}

fastkpc_full_cuda_census_build_key_map <- function(
    target, S_key, formula_class, data_hash, n, p,
    hash_fun = fastkpc_full_cuda_census_hash_utf8) {
  identity <- fastkpc_full_cuda_census_request_identity(
    target, S_key, formula_class
  )
  keep <- !duplicated(identity)
  key_target <- as.integer(target[keep])
  key_S_key <- as.character(S_key[keep])
  key_formula <- as.character(formula_class[keep])
  key_S <- lapply(key_S_key, fastkpc_full_cuda_census_parse_s)
  payload <- mapply(
    fastkpc_full_cuda_census_residual_payload,
    target = key_target,
    S = key_S,
    formula_class = key_formula,
    MoreArgs = list(data_hash = data_hash, n = n, p = p),
    USE.NAMES = FALSE
  )
  hash <- vapply(payload, hash_fun, character(1L))

  same_identity <- paste(key_S_key, key_formula, sep = "\t")
  same_keep <- !duplicated(same_identity)
  same_S_key <- key_S_key[same_keep]
  same_formula <- key_formula[same_keep]
  same_S <- key_S[same_keep]
  same_payload <- mapply(
    fastkpc_full_cuda_census_same_s_payload,
    S = same_S,
    formula_class = same_formula,
    MoreArgs = list(data_hash = data_hash, n = n, p = p),
    USE.NAMES = FALSE
  )
  same_hash <- vapply(same_payload, hash_fun, character(1L))
  fastkpc_full_cuda_census_validate_key_mapping(payload, hash)
  fastkpc_full_cuda_census_validate_key_mapping(same_payload, same_hash)
  same_index <- match(same_identity, same_identity[same_keep])

  list(
    map = data.frame(
      request_identity = identity[keep],
      residual_key_payload = payload,
      residual_key_sha256 = hash,
      target = key_target,
      S_key = key_S_key,
      S_size = as.integer(lengths(key_S)),
      formula_class = key_formula,
      same_S_group_id = same_hash[same_index],
      stringsAsFactors = FALSE
    ),
    raw_hash = hash[match(identity, identity[keep])]
  )
}

fastkpc_full_cuda_census_residual_hash <- function(
    target, S, formula_class, data_hash, n, p) {
  fastkpc_full_cuda_census_hash_utf8(
    fastkpc_full_cuda_census_residual_payload(
      target = target,
      S = S,
      formula_class = formula_class,
      data_hash = data_hash,
      n = n,
      p = p
    )
  )
}

fastkpc_full_cuda_census_reference_decision <- function(p, alpha, deleted) {
  if (length(p) != 1L || !is.finite(p)) {
    stop("canonical reference p-value is non-finite", call. = FALSE)
  }
  independent <- p >= alpha
  if (!identical(independent, isTRUE(deleted))) {
    stop("reference alpha decision disagrees with deletes_edge",
         call. = FALSE)
  }
  if (independent) "independent" else "dependent"
}

fastkpc_full_cuda_census_deletion_key <- function(level, x, y, S_key) {
  paste(as.integer(level), pmin(as.integer(x), as.integer(y)),
        pmax(as.integer(x), as.integer(y)), as.character(S_key), sep = "|")
}

fastkpc_full_cuda_census_validate_deletions <- function(trace, deletions) {
  selected <- trace[trace$deletes_edge %in% TRUE, , drop = FALSE]
  selected_key <- fastkpc_full_cuda_census_deletion_key(
    selected$level, selected$x, selected$y, selected$S_key
  )
  deletion_key <- fastkpc_full_cuda_census_deletion_key(
    deletions$level, deletions$edge_x, deletions$edge_y, deletions$S_key
  )
  if (anyDuplicated(selected_key) || anyDuplicated(deletion_key) ||
      !identical(sort(selected_key, method = "radix"),
                 sort(deletion_key, method = "radix"))) {
    stop("logical deleting rows do not match deletion trace one to one",
         call. = FALSE)
  }
  invisible(TRUE)
}

fastkpc_full_cuda_census_logical_tests <- function(
    trace, deletions, alpha, data_hash, n, p) {
  trace <- as.data.frame(trace, stringsAsFactors = FALSE)
  deletions <- as.data.frame(deletions, stringsAsFactors = FALSE)
  required_trace <- c(
    "logical_sequence_id", "source_sequence_id", "source_task_index",
    "level", "x", "y", "S_key", "p_value", "deletes_edge"
  )
  missing_trace <- setdiff(required_trace, names(trace))
  if (length(missing_trace) > 0L) {
    stop("logical trace missing fields: ",
         paste(missing_trace, collapse = ","), call. = FALSE)
  }
  required_deletions <- c("level", "edge_x", "edge_y", "S_key")
  missing_deletions <- setdiff(required_deletions, names(deletions))
  if (length(missing_deletions) > 0L) {
    stop("deletion trace missing fields: ",
         paste(missing_deletions, collapse = ","), call. = FALSE)
  }
  alpha <- as.numeric(alpha)
  if (length(alpha) != 1L || !is.finite(alpha) || alpha <= 0) {
    stop("alpha must be one finite positive value", call. = FALSE)
  }

  S <- lapply(trace$S_key, fastkpc_full_cuda_census_parse_s)
  invisible(lapply(deletions$S_key, fastkpc_full_cuda_census_parse_s))
  S_size <- lengths(S)
  formula_class <- vapply(
    S, fastkpc_regrxons_formula_class, character(1L)
  )
  decisions <- mapply(
    fastkpc_full_cuda_census_reference_decision,
    p = as.numeric(trace$p_value),
    deleted = as.logical(trace$deletes_edge),
    MoreArgs = list(alpha = alpha),
    USE.NAMES = FALSE
  )
  fastkpc_full_cuda_census_validate_deletions(trace, deletions)

  conditional <- which(S_size > 0L)
  key_info <- fastkpc_full_cuda_census_build_key_map(
    target = c(trace$x[conditional], trace$y[conditional]),
    S_key = rep(trace$S_key[conditional], 2L),
    formula_class = rep(formula_class[conditional], 2L),
    data_hash = data_hash,
    n = n,
    p = p
  )
  conditional_count <- length(conditional)
  residual_key_x <- rep(NA_character_, nrow(trace))
  residual_key_y <- rep(NA_character_, nrow(trace))
  residual_key_x[conditional] <- key_info$raw_hash[
    seq_len(conditional_count)
  ]
  residual_key_y[conditional] <- key_info$raw_hash[
    conditional_count + seq_len(conditional_count)
  ]
  reference_p <- as.numeric(trace$p_value)
  signed_distance <- reference_p - alpha
  signed_log_ratio <- log(pmax(reference_p, .Machine$double.xmin) / alpha)

  out <- data.frame(
    logical_sequence_id = as.integer(trace$logical_sequence_id),
    source_sequence_id = as.integer(trace$source_sequence_id),
    source_task_index = as.integer(trace$source_task_index),
    level = as.integer(trace$level),
    x = as.integer(trace$x),
    y = as.integer(trace$y),
    S_key = as.character(trace$S_key),
    S_size = as.integer(S_size),
    formula_class = formula_class,
    reference_p_value = reference_p,
    alpha = rep(alpha, nrow(trace)),
    reference_decision = decisions,
    reference_independent = as.logical(trace$deletes_edge),
    deletes_edge = as.logical(trace$deletes_edge),
    selected_sepset = as.logical(trace$deletes_edge),
    signed_distance_from_alpha = signed_distance,
    absolute_distance_from_alpha = abs(signed_distance),
    signed_log_ratio_from_alpha = signed_log_ratio,
    absolute_log_distance_from_alpha = abs(signed_log_ratio),
    residual_key_x = residual_key_x,
    residual_key_y = residual_key_y,
    stringsAsFactors = FALSE
  )
  attr(out, "dataset_sha256") <- as.character(data_hash)
  attr(out, "data_n") <- as.integer(n)
  attr(out, "data_p") <- as.integer(p)
  attr(out, "p_floor") <- .Machine$double.xmin
  attr(out, "residual_key_map") <- key_info$map
  out
}

fastkpc_full_cuda_census_validate_key_mapping <- function(payload, hash) {
  mapping <- unique(data.frame(
    payload = as.character(payload),
    hash = as.character(hash),
    stringsAsFactors = FALSE
  ))
  if (anyDuplicated(mapping$hash)) {
    stop("residual key SHA-256 collision: one hash maps to multiple payloads",
         call. = FALSE)
  }
  if (anyDuplicated(mapping$payload)) {
    stop("residual key serialization error: one payload maps to multiple hashes",
         call. = FALSE)
  }
  invisible(TRUE)
}

fastkpc_full_cuda_census_residual_requests <- function(
    logical_tests, hash_fun = fastkpc_full_cuda_census_hash_utf8) {
  use_cached_map <- missing(hash_fun)
  logical_tests <- as.data.frame(logical_tests, stringsAsFactors = FALSE)
  required <- c(
    "logical_sequence_id", "level", "x", "y", "S_key", "S_size",
    "formula_class"
  )
  missing <- setdiff(required, names(logical_tests))
  if (length(missing) > 0L) {
    stop("logical census missing fields: ", paste(missing, collapse = ","),
         call. = FALSE)
  }
  data_hash <- attr(logical_tests, "dataset_sha256", exact = TRUE)
  n <- attr(logical_tests, "data_n", exact = TRUE)
  p <- attr(logical_tests, "data_p", exact = TRUE)
  if (is.null(data_hash) || is.null(n) || is.null(p)) {
    stop("logical census is missing canonical data identity attributes",
         call. = FALSE)
  }
  conditional <- logical_tests[logical_tests$S_size > 0L, , drop = FALSE]
  if (nrow(conditional) == 0L) {
    stop("logical census has no conditional residual requests", call. = FALSE)
  }

  raw <- rbind(
    data.frame(
      logical_sequence_id = conditional$logical_sequence_id,
      level = conditional$level,
      target = conditional$x,
      S_key = conditional$S_key,
      S_size = conditional$S_size,
      formula_class = conditional$formula_class,
      stringsAsFactors = FALSE
    ),
    data.frame(
      logical_sequence_id = conditional$logical_sequence_id,
      level = conditional$level,
      target = conditional$y,
      S_key = conditional$S_key,
      S_size = conditional$S_size,
      formula_class = conditional$formula_class,
      stringsAsFactors = FALSE
    )
  )
  raw_identity <- fastkpc_full_cuda_census_request_identity(
    raw$target, raw$S_key, raw$formula_class
  )
  key_map <- if (use_cached_map) {
    attr(logical_tests, "residual_key_map", exact = TRUE)
  } else {
    NULL
  }
  if (is.null(key_map)) {
    key_info <- fastkpc_full_cuda_census_build_key_map(
      target = raw$target,
      S_key = raw$S_key,
      formula_class = raw$formula_class,
      data_hash = data_hash,
      n = n,
      p = p,
      hash_fun = hash_fun
    )
    key_map <- key_info$map
  }
  map_index <- match(raw_identity, key_map$request_identity)
  if (anyNA(map_index)) {
    stop("logical requests do not map to the canonical residual key corpus",
         call. = FALSE)
  }
  raw$residual_key_sha256 <- key_map$residual_key_sha256[map_index]

  order_id <- order(raw$residual_key_sha256, raw$logical_sequence_id,
                    method = "radix")
  raw <- raw[order_id, , drop = FALSE]
  starts <- c(1L, which(raw$residual_key_sha256[-1L] !=
                          raw$residual_key_sha256[-nrow(raw)]) + 1L)
  ends <- c(starts[-1L] - 1L, nrow(raw))
  first <- raw[starts, , drop = FALSE]
  first_key <- key_map[
    match(first$residual_key_sha256, key_map$residual_key_sha256),
    , drop = FALSE
  ]
  result <- data.frame(
    residual_key_payload = first_key$residual_key_payload,
    residual_key_sha256 = first$residual_key_sha256,
    target = as.integer(first_key$target),
    S_key = first_key$S_key,
    S_size = as.integer(first_key$S_size),
    formula_class = first_key$formula_class,
    same_S_group_id = first_key$same_S_group_id,
    same_S_group_size = integer(length(starts)),
    request_multiplicity = as.integer(ends - starts + 1L),
    first_logical_sequence_id = vapply(
      seq_along(starts),
      function(i) min(raw$logical_sequence_id[starts[[i]]:ends[[i]]]),
      integer(1L)
    ),
    last_logical_sequence_id = vapply(
      seq_along(starts),
      function(i) max(raw$logical_sequence_id[starts[[i]]:ends[[i]]]),
      integer(1L)
    ),
    first_level = vapply(
      seq_along(starts),
      function(i) min(raw$level[starts[[i]]:ends[[i]]]),
      integer(1L)
    ),
    last_level = vapply(
      seq_along(starts),
      function(i) max(raw$level[starts[[i]]:ends[[i]]]),
      integer(1L)
    ),
    stringsAsFactors = FALSE
  )
  group_sizes <- table(result$same_S_group_id)
  result$same_S_group_size <- as.integer(
    group_sizes[result$same_S_group_id]
  )
  result <- result[
    order(result$residual_key_sha256, method = "radix"), , drop = FALSE
  ]
  rownames(result) <- NULL
  corpus_payload <- paste0(
    paste(result$residual_key_sha256, collapse = "\n"), "\n"
  )
  attr(result, "canonical_key_corpus_hash") <- hash_fun(corpus_payload)
  attr(result, "conditional_residual_request_count") <- nrow(raw)
  result
}

fastkpc_full_cuda_census_structural <- function(inputs) {
  required <- c(
    "data", "oracle", "oracle_input_hashes",
    "oracle_input_bundle_sha256", "oracle_inherited_graph_gate",
    "new_candidate_graph_gate"
  )
  missing <- setdiff(required, names(inputs))
  if (length(missing) > 0L) {
    stop("Phase 1 inputs missing fields: ", paste(missing, collapse = ","),
         call. = FALSE)
  }
  manifest <- inputs$oracle$manifest
  logical_tests <- fastkpc_full_cuda_census_logical_tests(
    trace = inputs$oracle$logical_trace,
    deletions = inputs$oracle$deletion_trace,
    alpha = as.numeric(manifest$alpha),
    data_hash = fastkpc_full_cuda_census_input_contract()$dataset_matrix_sha256,
    n = nrow(inputs$data),
    p = ncol(inputs$data)
  )
  residual_requests <- fastkpc_full_cuda_census_residual_requests(
    logical_tests
  )
  list(
    logical_tests = logical_tests,
    residual_requests = residual_requests,
    canonical_key_corpus_hash = attr(
      residual_requests, "canonical_key_corpus_hash", exact = TRUE
    ),
    conditional_residual_request_count = attr(
      residual_requests, "conditional_residual_request_count", exact = TRUE
    ),
    unique_conditional_S_count = length(unique(
      residual_requests$same_S_group_id
    )),
    oracle_input_hashes = inputs$oracle_input_hashes,
    oracle_input_bundle_sha256 = inputs$oracle_input_bundle_sha256,
    oracle_inherited_graph_gate = inputs$oracle_inherited_graph_gate,
    new_candidate_graph_gate = inputs$new_candidate_graph_gate,
    p_floor = attr(logical_tests, "p_floor", exact = TRUE),
    historical_route_metric = list(
      name = "s_affinity_executed_mgcv_fit_count",
      value = 273284L,
      metric_role = "historical_route_metric",
      hard_gate = FALSE,
      provenance_status = "no_hash_protected_route_trace"
    )
  )
}

fastkpc_full_cuda_census_validate_structural <- function(
    structural, canonical = FALSE) {
  required <- c(
    "logical_tests", "residual_requests", "canonical_key_corpus_hash",
    "conditional_residual_request_count", "unique_conditional_S_count",
    "oracle_inherited_graph_gate", "new_candidate_graph_gate",
    "historical_route_metric"
  )
  missing <- setdiff(required, names(structural))
  if (length(missing) > 0L) {
    stop("structural census missing fields: ", paste(missing, collapse = ","),
         call. = FALSE)
  }
  logical_tests <- structural$logical_tests
  requests <- structural$residual_requests
  if (!isTRUE(structural$oracle_inherited_graph_gate) ||
      !identical(structural$new_candidate_graph_gate, "NOT_APPLICABLE")) {
    stop("structural census inherited graph scope is invalid", call. = FALSE)
  }
  if (anyDuplicated(requests$residual_key_payload) ||
      anyDuplicated(requests$residual_key_sha256)) {
    stop("structural census contains duplicate residual keys", call. = FALSE)
  }
  if (!identical(sum(requests$request_multiplicity),
                 as.integer(structural$conditional_residual_request_count))) {
    stop("structural census request multiplicity is inconsistent",
         call. = FALSE)
  }
  if (isTRUE(structural$historical_route_metric$hard_gate) ||
      !identical(structural$historical_route_metric$metric_role,
                 "historical_route_metric")) {
    stop("historical S-affinity route metric is incorrectly gating",
         call. = FALSE)
  }
  if (!isTRUE(canonical)) return(TRUE)

  contract <- fastkpc_full_cuda_census_input_contract()
  gates <- c(
    nrow(logical_tests) == 240489L,
    sum(logical_tests$S_size > 0L) == 238276L,
    structural$conditional_residual_request_count == 476552L,
    nrow(requests) == 110617L,
    structural$unique_conditional_S_count == 8634L,
    identical(structural$canonical_key_corpus_hash,
              contract$canonical_key_corpus_hash)
  )
  if (!all(gates)) {
    stop("canonical structural census hard gate failed", call. = FALSE)
  }
  TRUE
}
