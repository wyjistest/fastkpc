.fastkpc_full_cuda_phase35_contract_order <- c(
  "architecture_contract_v1",
  "numerical_contract_v1",
  "artifact_identity_contract_v1",
  "reference_machine_v1",
  "performance_budget_v1",
  "development_qualification_corpus_v1",
  "metamorphic_contract_v1",
  "promotion_holdout_manifest_v1"
)

fastkpc_full_cuda_phase35_contract_names <- function() {
  .fastkpc_full_cuda_phase35_contract_order
}

.fastkpc_full_cuda_phase35_contract_root <- function() {
  candidates <- c(
    file.path("fastkpc", "inst", "contracts", "full_cuda_ci"),
    file.path("inst", "contracts", "full_cuda_ci")
  )
  present <- candidates[dir.exists(candidates)]
  if (length(present) == 0L) {
    stop("cannot locate tracked full CUDA CI contracts", call. = FALSE)
  }
  normalizePath(present[[1L]], winslash = "/", mustWork = TRUE)
}

.fastkpc_full_cuda_phase35_scalar_character <- function(value) {
  typeof(value) == "character" && length(value) == 1L &&
    !is.object(value) && is.null(attributes(value)) && !is.na(value)
}

.fastkpc_full_cuda_phase35_scalar_logical <- function(value) {
  typeof(value) == "logical" && length(value) == 1L &&
    !is.object(value) && is.null(attributes(value)) && !is.na(value)
}

.fastkpc_full_cuda_phase35_scalar_integer <- function(value,
                                                       minimum = NULL) {
  clean <- typeof(value) == "integer" && length(value) == 1L &&
    !is.object(value) && is.null(attributes(value)) && !is.na(value)
  if (!isTRUE(clean)) return(FALSE)
  is.null(minimum) || value >= minimum
}

.fastkpc_full_cuda_phase35_sha256 <- function(value) {
  .fastkpc_full_cuda_phase35_scalar_character(value) &&
    grepl("^[0-9a-f]{64}$", value)
}

.fastkpc_full_cuda_phase35_plain_object <- function(value) {
  is.list(value) && !is.object(value) && !is.null(names(value)) &&
    !anyNA(names(value)) && !anyDuplicated(names(value)) &&
    all(nzchar(names(value)))
}

.fastkpc_full_cuda_phase35_ascii <- function(value, label) {
  if (!.fastkpc_full_cuda_phase35_scalar_character(value) ||
      is.na(iconv(value, from = "UTF-8", to = "ASCII", sub = NA))) {
    stop(label, " must be one ASCII string", call. = FALSE)
  }
  value
}

.fastkpc_full_cuda_phase35_validate_json_tree <- function(value,
                                                           label = "JSON") {
  if (is.null(value)) return(invisible(TRUE))
  if (is.list(value) && !is.object(value)) {
    object <- !is.null(names(value))
    if (object) {
      keys <- names(value)
      if (anyNA(keys) || any(!nzchar(keys)) || anyDuplicated(keys)) {
        duplicate <- unique(keys[duplicated(keys)])
        if (length(duplicate) > 0L) {
          stop(label, " has duplicate object key: ", duplicate[[1L]],
               call. = FALSE)
        }
        stop(label, " has malformed object keys", call. = FALSE)
      }
      for (key in keys) {
        .fastkpc_full_cuda_phase35_ascii(
          key, paste0(label, " object key")
        )
      }
    }
    for (index in seq_along(value)) {
      child <- if (object) names(value)[[index]] else as.character(index)
      .fastkpc_full_cuda_phase35_validate_json_tree(
        value[[index]], paste0(label, "$", child)
      )
    }
    return(invisible(TRUE))
  }
  if (typeof(value) == "character" && length(value) == 1L &&
      !is.object(value) && is.null(attributes(value)) && !is.na(value)) {
    .fastkpc_full_cuda_phase35_ascii(value, label)
    return(invisible(TRUE))
  }
  if (typeof(value) == "logical" && length(value) == 1L &&
      !is.object(value) && is.null(attributes(value)) && !is.na(value)) {
    return(invisible(TRUE))
  }
  if (typeof(value) %in% c("integer", "double") && length(value) == 1L &&
      !is.object(value) && is.null(attributes(value)) && !is.na(value) &&
      is.finite(value) && value == floor(value) && abs(value) <= 2^53) {
    return(invisible(TRUE))
  }
  stop(
    label,
    " must use null, booleans, strings, arrays, objects, or signed integer JSON numbers; finite decimals must be strings",
    call. = FALSE
  )
}

fastkpc_full_cuda_phase35_parse_json <- function(value, label = "JSON") {
  if (!.fastkpc_full_cuda_phase35_scalar_character(value)) {
    stop(label, " must be one UTF-8 JSON string", call. = FALSE)
  }
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("jsonlite is required for Phase 3.5 contracts", call. = FALSE)
  }
  parsed <- tryCatch(
    jsonlite::fromJSON(value, simplifyVector = FALSE),
    error = function(error) {
      stop(label, " is not valid JSON: ", conditionMessage(error),
           call. = FALSE)
    }
  )
  .fastkpc_full_cuda_phase35_validate_json_tree(parsed, label)
  parsed
}

.fastkpc_full_cuda_phase35_escape_json_string <- function(value) {
  value <- .fastkpc_full_cuda_phase35_ascii(value, "canonical JSON string")
  bytes <- as.integer(charToRaw(enc2utf8(value)))
  encoded <- vapply(bytes, function(byte) {
    if (byte == 34L) return("\\\"")
    if (byte == 92L) return("\\\\")
    if (byte == 8L) return("\\b")
    if (byte == 12L) return("\\f")
    if (byte == 10L) return("\\n")
    if (byte == 13L) return("\\r")
    if (byte == 9L) return("\\t")
    if (byte < 32L) return(sprintf("\\u%04x", byte))
    rawToChar(as.raw(byte))
  }, character(1L), USE.NAMES = FALSE)
  paste0('"', paste0(encoded, collapse = ""), '"')
}

fastkpc_full_cuda_phase35_canonical_json <- function(value) {
  .fastkpc_full_cuda_phase35_validate_json_tree(value, "canonical JSON")
  encode <- function(node) {
    if (is.null(node)) return("null")
    if (is.list(node)) {
      if (is.null(names(node))) {
        return(paste0(
          "[", paste(vapply(node, encode, character(1L)), collapse = ","),
          "]"
        ))
      }
      keys <- sort(names(node), method = "radix")
      fields <- vapply(keys, function(key) {
        paste0(.fastkpc_full_cuda_phase35_escape_json_string(key), ":",
               encode(node[[key]]))
      }, character(1L), USE.NAMES = FALSE)
      return(paste0("{", paste(fields, collapse = ","), "}"))
    }
    if (typeof(node) == "character") {
      return(.fastkpc_full_cuda_phase35_escape_json_string(node))
    }
    if (typeof(node) == "logical") {
      return(if (node) "true" else "false")
    }
    if (typeof(node) == "integer") return(as.character(node))
    if (typeof(node) == "double") return(sprintf("%.0f", node))
    stop("unsupported canonical JSON value", call. = FALSE)
  }
  encode(value)
}

fastkpc_full_cuda_phase35_sha256_utf8 <- function(value) {
  if (!.fastkpc_full_cuda_phase35_scalar_character(value)) {
    stop("SHA-256 input must be one UTF-8 string", call. = FALSE)
  }
  if (!requireNamespace("digest", quietly = TRUE)) {
    stop("digest is required for Phase 3.5 contracts", call. = FALSE)
  }
  unname(digest::digest(enc2utf8(value), algo = "sha256", serialize = FALSE))
}

.fastkpc_full_cuda_phase35_read_utf8 <- function(path, label) {
  if (!.fastkpc_full_cuda_phase35_scalar_character(path) ||
      !file.exists(path) || dir.exists(path) || nzchar(Sys.readlink(path))) {
    stop(label, " path is missing, non-regular, or a symbolic link",
         call. = FALSE)
  }
  size <- file.info(path)$size
  if (is.na(size) || size <= 0 || size > 4L * 1024L * 1024L) {
    stop(label, " file size is invalid", call. = FALSE)
  }
  connection <- file(path, open = "rb")
  on.exit(close(connection), add = TRUE)
  raw <- readBin(connection, what = "raw", n = size)
  if (length(raw) != size || any(raw == as.raw(0L))) {
    stop(label, " file bytes are invalid", call. = FALSE)
  }
  text <- tryCatch(
    rawToChar(raw),
    error = function(error) stop(label, " is not UTF-8 text", call. = FALSE)
  )
  if (is.na(iconv(text, from = "UTF-8", to = "UTF-8", sub = NA))) {
    stop(label, " is not valid UTF-8", call. = FALSE)
  }
  text
}

.fastkpc_full_cuda_phase35_validate_envelope <- function(document,
                                                          expected_name) {
  required <- c(
    "campaign", "contract_name", "contract_schema_version",
    "phase_introduced", "semantic_version", "payload"
  )
  if (!.fastkpc_full_cuda_phase35_plain_object(document) ||
      !setequal(names(document), required)) {
    stop("tracked contract envelope is malformed", call. = FALSE)
  }
  if (!identical(document$campaign, "full-cuda-legacy-compatible-ci") ||
      !identical(document$contract_schema_version,
                 "full-cuda-ci-tracked-contract-v1") ||
      !identical(document$phase_introduced, "3.5")) {
    stop("tracked contract envelope authority is invalid", call. = FALSE)
  }
  if (!.fastkpc_full_cuda_phase35_scalar_character(expected_name) ||
      !identical(document$contract_name, expected_name)) {
    stop("tracked contract name mismatch", call. = FALSE)
  }
  version <- document$semantic_version
  if (!.fastkpc_full_cuda_phase35_plain_object(version) ||
      !setequal(names(version), c("major", "minor", "patch")) ||
      !.fastkpc_full_cuda_phase35_scalar_integer(version$major, 1L) ||
      !.fastkpc_full_cuda_phase35_scalar_integer(version$minor, 0L) ||
      !.fastkpc_full_cuda_phase35_scalar_integer(version$patch, 0L)) {
    stop("tracked contract semantic version is malformed", call. = FALSE)
  }
  if (!.fastkpc_full_cuda_phase35_plain_object(document$payload)) {
    stop("tracked contract payload must be one JSON object", call. = FALSE)
  }
  invisible(TRUE)
}

.fastkpc_full_cuda_phase35_require_fields <- function(value, fields, label) {
  if (!.fastkpc_full_cuda_phase35_plain_object(value) ||
      !all(fields %in% names(value))) {
    stop(label, " is missing required fields", call. = FALSE)
  }
  invisible(TRUE)
}

.fastkpc_full_cuda_phase35_decimal <- function(value, label,
                                                nonnegative = TRUE) {
  if (!.fastkpc_full_cuda_phase35_scalar_character(value) ||
      !grepl("^-?(0|[1-9][0-9]*)(\\.[0-9]+)?(e[+-]?[0-9]+)?$", value) ||
      !is.finite(suppressWarnings(as.numeric(value))) ||
      (nonnegative && as.numeric(value) < 0)) {
    stop(label, " must be one canonical finite decimal string",
         call. = FALSE)
  }
  as.numeric(value)
}

.fastkpc_full_cuda_phase35_validate_architecture <- function(payload) {
  .fastkpc_full_cuda_phase35_require_fields(
    payload,
    c("abi", "required_capability_query_fields", "capabilities",
      "ownership", "asynchrony", "error_model", "selected_sp_semantics",
      "large_payload_policy", "compact_result_fields", "semantic_objects",
      "not_frozen"),
    "architecture contract"
  )
  expected <- c(
    "PreparedSGpuHandle", "TargetOptimizerStateHandle",
    "DeviceResidualHandle", "DcovComponentHandle", "LogicalCiBatchHandle",
    "CompactCiResult"
  )
  objects <- payload$semantic_objects
  if (!is.list(objects) || !is.null(names(objects)) ||
      !identical(vapply(objects, `[[`, character(1L), "name"), expected) ||
      !identical(payload$abi$major, 1L) ||
      !identical(payload$large_payload_policy$residual_d2h, "forbidden") ||
      !identical(payload$large_payload_policy$component_d2h, "forbidden")) {
    stop("architecture contract semantic ABI is invalid", call. = FALSE)
  }
  invisible(TRUE)
}

.fastkpc_full_cuda_phase35_validate_numerical <- function(payload) {
  .fastkpc_full_cuda_phase35_require_fields(
    payload,
    c("decimal_encoding", "precision", "comparison_metrics",
      "denominator_floors", "gam_formulas", "smoothing_parameter",
      "condition_buckets", "rank_policy", "dcov_formulas", "tolerances",
      "decision_semantics", "nonfinite_policy", "boundary_policy",
      "tolerance_change_policy"),
    "numerical contract"
  )
  decimal_values <- c(
    unlist(payload$denominator_floors, use.names = FALSE),
    unlist(lapply(payload$tolerances, function(gate) {
      gate[vapply(gate, .fastkpc_full_cuda_phase35_scalar_character,
                  logical(1L))]
    }), use.names = FALSE)
  )
  for (index in seq_along(decimal_values)) {
    value <- decimal_values[[index]]
    if (grepl("^-?[0-9]", value)) {
      .fastkpc_full_cuda_phase35_decimal(
        value, "numerical contract decimal"
      )
    }
  }
  solvers <- vapply(payload$condition_buckets, `[[`, character(1L), "solver")
  if (!identical(
        solvers,
        c("CHOLESKY_BATCHED", "AUGMENTED_QR", "AUGMENTED_SVD",
          "AUGMENTED_SVD")
      ) ||
      !identical(payload$decision_semantics$independent_when,
                 "p_value >= alpha") ||
      !identical(payload$decision_semantics$allowed_flip_count, 0L) ||
      !identical(payload$nonfinite_policy, "fail-closed-before-replay")) {
    stop("numerical contract decision or solver policy is invalid",
         call. = FALSE)
  }
  invisible(TRUE)
}

.fastkpc_full_cuda_phase35_validate_identity_contract <- function(payload) {
  .fastkpc_full_cuda_phase35_require_fields(
    payload,
    c("canonicalization", "contract_snapshot", "producer_semantic_identity",
      "validator_attestation_identity", "volatile_execution_receipt",
      "artifact_semantic_identity", "publication"),
    "artifact identity contract"
  )
  canonical <- payload$canonicalization
  if (!identical(canonical$schema, "full-cuda-ci-canonical-json-v1") ||
      !identical(canonical$duplicate_object_keys, "forbidden") ||
      !identical(canonical$json_numbers,
                 "signed-safe-53-bit-integers-only") ||
      !isTRUE(payload$producer_semantic_identity$immutable_after_publication) ||
      !isTRUE(payload$validator_attestation_identity$append_only) ||
      !isTRUE(payload$volatile_execution_receipt$audit_only)) {
    stop("artifact identity contract namespace policy is invalid",
         call. = FALSE)
  }
  invisible(TRUE)
}

.fastkpc_full_cuda_phase35_validate_reference_machine <- function(payload) {
  .fastkpc_full_cuda_phase35_require_fields(
    payload, c("host", "cpu", "gpu", "software", "timing"),
    "reference machine contract"
  )
  if (!identical(payload$gpu$model, "NVIDIA GeForce RTX 4090") ||
      !identical(payload$gpu$device_id, 0L) ||
      !identical(payload$gpu$compute_capability, "8.9") ||
      payload$gpu$total_memory_bytes <= 0 ||
      !identical(payload$timing$build_included, FALSE) ||
      !identical(payload$timing$measured_warm_repetitions, 5L)) {
    stop("reference machine contract timing or device policy is invalid",
         call. = FALSE)
  }
  invisible(TRUE)
}

.fastkpc_full_cuda_phase35_validate_budget <- function(payload) {
  .fastkpc_full_cuda_phase35_require_fields(
    payload,
    c("reference_machine_contract", "component_budgets", "feasibility",
      "promotion", "measurement"),
    "performance budget contract"
  )
  expected <- c(
    "input_and_h2d", "native_setup", "gcv_selection",
    "fixed_sp_solve_and_residual", "dcov_component",
    "dcov_pair_and_gamma", "control_replay_and_packaging", "contingency"
  )
  budgets <- payload$component_budgets
  if (!identical(names(budgets), expected)) {
    stop("performance component budget order is invalid", call. = FALSE)
  }
  values <- vapply(budgets, function(value) {
    if (!.fastkpc_full_cuda_phase35_scalar_integer(
          value$warm_upper_bound_ms, 0L
        )) {
      stop("performance component budget is malformed", call. = FALSE)
    }
    value$warm_upper_bound_ms
  }, integer(1L))
  if (!identical(sum(values), payload$feasibility$total_upper_bound_ms) ||
      !identical(payload$feasibility$total_upper_bound_ms, 120000L) ||
      !identical(
        sum(values[c("dcov_component", "dcov_pair_and_gamma")]),
        payload$feasibility$dcov_total_upper_bound_ms
      ) ||
      !identical(payload$promotion$warm_repetitions, 5L)) {
    stop("performance budget does not conserve its declared envelope",
         call. = FALSE)
  }
  .fastkpc_full_cuda_phase35_decimal(
    payload$promotion$correct_baseline_ratio_max,
    "correct baseline ratio"
  )
  invisible(TRUE)
}

.fastkpc_full_cuda_phase35_validate_development_corpus <- function(payload) {
  .fastkpc_full_cuda_phase35_require_fields(
    payload,
    c("purpose", "source_identities", "canonical_counts",
      "required_risk_classes", "selection", "allowed_use",
      "promotion_holdout_claim"),
    "development corpus contract"
  )
  if (!all(vapply(
        payload$source_identities,
        .fastkpc_full_cuda_phase35_sha256, logical(1L)
      )) ||
      !identical(payload$canonical_counts$target_count, 6143L) ||
      !identical(payload$canonical_counts$dcov_pair_count, 3808L) ||
      !identical(payload$canonical_counts$near_alpha_pair_count, 1478L) ||
      !identical(payload$promotion_holdout_claim, FALSE)) {
    stop("development corpus identities or counts are invalid",
         call. = FALSE)
  }
  invisible(TRUE)
}

.fastkpc_full_cuda_phase35_validate_metamorphic <- function(payload) {
  .fastkpc_full_cuda_phase35_require_fields(
    payload,
    c("purpose", "base_corpus", "global_invariants", "transformations",
      "minimum_coverage", "failure_policy"),
    "metamorphic contract"
  )
  expected <- c(
    "conditioning-column-permutation", "basis-sign-flip",
    "equivalent-orthogonal-rotation", "batch-split-merge",
    "stream-count-variation", "cache-capacity-variation",
    "standalone-versus-batched"
  )
  if (!identical(
        vapply(payload$transformations, `[[`, character(1L), "name"),
        expected
      ) || !identical(payload$minimum_coverage$stream_counts,
                      list(1L, 2L, 4L))) {
    stop("metamorphic transformation coverage is invalid", call. = FALSE)
  }
  invisible(TRUE)
}

.fastkpc_full_cuda_phase35_validate_holdout <- function(payload) {
  .fastkpc_full_cuda_phase35_require_fields(
    payload,
    c("holdout_id", "state", "ordinary_development_access",
      "payload_present_in_repository", "custody", "commitment",
      "release_protocol", "implementation_change_after_open_policy",
      "result_policy"),
    "promotion holdout manifest"
  )
  commitment_fields <- list(
    custody_authority = payload$custody$authority,
    holdout_id = payload$holdout_id,
    payload_present_in_repository = payload$payload_present_in_repository,
    release_phase = payload$custody$release_phase,
    state = payload$state
  )
  commitment_sha256 <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(commitment_fields)
  )
  if (!identical(payload$state, "SEALED_NOT_RELEASED") ||
      !identical(payload$ordinary_development_access, "forbidden") ||
      !identical(payload$payload_present_in_repository, FALSE) ||
      !identical(
        payload$commitment$identity_formula,
        paste0(
          "sha256(canonical(custody_authority,holdout_id,",
          "payload_present_in_repository,release_phase,state))"
        )
      ) ||
      !.fastkpc_full_cuda_phase35_sha256(
        payload$commitment$manifest_identity_sha256
      ) ||
      !identical(payload$commitment$manifest_identity_sha256,
                 commitment_sha256) ||
      !identical(payload$custody$release_phase, "10")) {
    stop("promotion holdout release policy is invalid", call. = FALSE)
  }
  invisible(TRUE)
}

.fastkpc_full_cuda_phase35_validate_known_payload <- function(name, payload) {
  validator <- switch(
    name,
    architecture_contract_v1 =
      .fastkpc_full_cuda_phase35_validate_architecture,
    numerical_contract_v1 = .fastkpc_full_cuda_phase35_validate_numerical,
    artifact_identity_contract_v1 =
      .fastkpc_full_cuda_phase35_validate_identity_contract,
    reference_machine_v1 =
      .fastkpc_full_cuda_phase35_validate_reference_machine,
    performance_budget_v1 = .fastkpc_full_cuda_phase35_validate_budget,
    development_qualification_corpus_v1 =
      .fastkpc_full_cuda_phase35_validate_development_corpus,
    metamorphic_contract_v1 =
      .fastkpc_full_cuda_phase35_validate_metamorphic,
    promotion_holdout_manifest_v1 =
      .fastkpc_full_cuda_phase35_validate_holdout,
    NULL
  )
  if (is.null(validator)) {
    stop("unknown tracked Phase 3.5 contract: ", name, call. = FALSE)
  }
  validator(payload)
}

fastkpc_full_cuda_phase35_contract_identity_from_path <- function(
    path, expected_contract_name,
    validate_known_contract = expected_contract_name %in%
      fastkpc_full_cuda_phase35_contract_names()) {
  text <- .fastkpc_full_cuda_phase35_read_utf8(
    path, paste0(expected_contract_name, " contract")
  )
  document <- fastkpc_full_cuda_phase35_parse_json(
    text, paste0(expected_contract_name, " contract")
  )
  .fastkpc_full_cuda_phase35_validate_envelope(
    document, expected_contract_name
  )
  if (isTRUE(validate_known_contract)) {
    .fastkpc_full_cuda_phase35_validate_known_payload(
      expected_contract_name, document$payload
    )
  }
  canonical <- fastkpc_full_cuda_phase35_canonical_json(document)
  c(
    document,
    list(
      path = normalizePath(path, winslash = "/", mustWork = TRUE),
      canonical_json = canonical,
      sha256 = fastkpc_full_cuda_phase35_sha256_utf8(canonical),
      document = document
    )
  )
}

fastkpc_full_cuda_phase35_load_contract <- function(
    name, root = .fastkpc_full_cuda_phase35_contract_root()) {
  if (!.fastkpc_full_cuda_phase35_scalar_character(name) ||
      !name %in% fastkpc_full_cuda_phase35_contract_names()) {
    stop("unknown tracked Phase 3.5 contract", call. = FALSE)
  }
  fastkpc_full_cuda_phase35_contract_identity_from_path(
    file.path(root, paste0(name, ".json")), name,
    validate_known_contract = TRUE
  )
}

.fastkpc_full_cuda_phase35_file_sha256 <- function(path) {
  if (!requireNamespace("digest", quietly = TRUE)) {
    stop("digest is required for Phase 3.5 contracts", call. = FALSE)
  }
  if (!file.exists(path) || dir.exists(path) || nzchar(Sys.readlink(path))) {
    stop("authenticated source artifact is missing or unsafe: ", path,
         call. = FALSE)
  }
  unname(digest::digest(file = path, algo = "sha256", serialize = FALSE))
}

.fastkpc_full_cuda_phase35_validate_source_identities <- function(contracts) {
  source <- contracts$development_qualification_corpus_v1$payload$
    source_identities
  paths <- c(
    phase0_manifest_sha256 =
      "fastkpc/artifacts/full_cuda_ci/oracle_351x48_v1/manifest.json",
    phase1_manifest_sha256 =
      "fastkpc/artifacts/full_cuda_ci/workload_census_351x48_v1/manifest.json",
    phase2_manifest_sha256 =
      "fastkpc/artifacts/full_cuda_ci/prepared_s_contract_v1/manifest.json",
    phase3_oracle_manifest_sha256 =
      "fastkpc/artifacts/full_cuda_ci/fixed_sp_cuda_oracle_sp_v1/manifest.json",
    phase3_shadow_manifest_sha256 =
      "fastkpc/artifacts/full_cuda_ci/fixed_sp_cuda_full_shadow_v1/manifest.json",
    oracle_logical_trace_file_sha256 =
      "fastkpc/artifacts/full_cuda_ci/oracle_351x48_v1/logical_ci_trace.rds"
  )
  actual <- vapply(paths, .fastkpc_full_cuda_phase35_file_sha256,
                   character(1L))
  if (!identical(unname(actual), unname(unlist(source[names(paths)])))) {
    stop("development corpus source artifact identity mismatch",
         call. = FALSE)
  }
  invisible(TRUE)
}

fastkpc_full_cuda_phase35_load_contract_set <- function(
    root = .fastkpc_full_cuda_phase35_contract_root(),
    verify_source_artifacts = TRUE) {
  contracts <- lapply(
    fastkpc_full_cuda_phase35_contract_names(),
    fastkpc_full_cuda_phase35_load_contract,
    root = root
  )
  names(contracts) <- fastkpc_full_cuda_phase35_contract_names()
  if (!identical(
        contracts$performance_budget_v1$payload$reference_machine_contract,
        contracts$reference_machine_v1$contract_name
      ) ||
      !identical(
        contracts$metamorphic_contract_v1$payload$base_corpus,
        contracts$development_qualification_corpus_v1$contract_name
      )) {
    stop("tracked Phase 3.5 cross-contract reference mismatch",
         call. = FALSE)
  }
  if (isTRUE(verify_source_artifacts)) {
    .fastkpc_full_cuda_phase35_validate_source_identities(contracts)
  }
  contracts
}

fastkpc_full_cuda_phase35_semantic_version <- function(contract) {
  version <- contract$semantic_version
  paste(version$major, version$minor, version$patch, sep = ".")
}

fastkpc_full_cuda_phase35_contract_snapshots <- function(contracts =
    fastkpc_full_cuda_phase35_load_contract_set()) {
  expected <- fastkpc_full_cuda_phase35_contract_names()
  if (!is.list(contracts) || !identical(names(contracts), expected)) {
    stop("tracked contract set is malformed", call. = FALSE)
  }
  snapshots <- lapply(contracts, function(contract) {
    snapshot_canonical <- fastkpc_full_cuda_phase35_canonical_json(
      contract$document
    )
    if (!.fastkpc_full_cuda_phase35_sha256(contract$sha256) ||
        !identical(snapshot_canonical, contract$canonical_json) ||
        !identical(
          fastkpc_full_cuda_phase35_sha256_utf8(contract$canonical_json),
          contract$sha256
        )) {
      stop("tracked contract hash is invalid", call. = FALSE)
    }
    list(
      contract_name = contract$contract_name,
      semantic_version =
        fastkpc_full_cuda_phase35_semantic_version(contract),
      sha256 = contract$sha256,
      snapshot = contract$document
    )
  })
  names(snapshots) <- expected
  snapshots
}

.fastkpc_full_cuda_phase35_validate_contract_snapshots <- function(
    snapshots, tracked_contracts = NULL) {
  expected <- fastkpc_full_cuda_phase35_contract_names()
  if (!is.list(snapshots) || !identical(names(snapshots), expected)) {
    stop("contract snapshots are malformed", call. = FALSE)
  }
  if (is.null(tracked_contracts)) {
    tracked_contracts <- fastkpc_full_cuda_phase35_load_contract_set(
      verify_source_artifacts = FALSE
    )
  }
  if (!is.list(tracked_contracts) ||
      !identical(names(tracked_contracts), expected)) {
    stop("tracked contract authority is malformed", call. = FALSE)
  }
  for (name in expected) {
    snapshot <- snapshots[[name]]
    if (!.fastkpc_full_cuda_phase35_plain_object(snapshot) ||
        !identical(
          names(snapshot),
          c("contract_name", "semantic_version", "sha256", "snapshot")
        ) ||
        !identical(snapshot$contract_name, name) ||
        !identical(snapshot$snapshot$contract_name, name) ||
        !.fastkpc_full_cuda_phase35_sha256(snapshot$sha256)) {
      stop("contract snapshot is malformed: ", name, call. = FALSE)
    }
    canonical <- fastkpc_full_cuda_phase35_canonical_json(snapshot$snapshot)
    actual_hash <- fastkpc_full_cuda_phase35_sha256_utf8(canonical)
    tracked <- tracked_contracts[[name]]
    if (!identical(snapshot$semantic_version,
                   fastkpc_full_cuda_phase35_semantic_version(tracked)) ||
        !identical(snapshot$sha256, actual_hash) ||
        !identical(snapshot$sha256, tracked$sha256) ||
        !identical(canonical, tracked$canonical_json)) {
      stop("contract snapshot does not match tracked authority: ", name,
           call. = FALSE)
    }
  }
  invisible(TRUE)
}

.fastkpc_full_cuda_phase35_require_hash <- function(value, label) {
  if (!.fastkpc_full_cuda_phase35_sha256(value)) {
    stop(label, " must be one lowercase SHA-256", call. = FALSE)
  }
  value
}

.fastkpc_full_cuda_phase35_hash_fields <- function(value, hash_field) {
  if (!.fastkpc_full_cuda_phase35_plain_object(value) ||
      hash_field %in% names(value)) {
    stop("identity fields are malformed", call. = FALSE)
  }
  hash <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(value)
  )
  c(value, setNames(list(hash), hash_field))
}

fastkpc_full_cuda_phase35_producer_identity <- function(
    producer_source_closure_sha256,
    native_binary_sha256,
    route_semantic_version,
    dataset_or_corpus_sha256,
    oracle_sha256,
    backend_configuration_sha256,
    build_recipe_sha256,
    contracts = fastkpc_full_cuda_phase35_load_contract_set()) {
  hashes <- list(
    producer_source_closure_sha256 = producer_source_closure_sha256,
    native_binary_sha256 = native_binary_sha256,
    dataset_or_corpus_sha256 = dataset_or_corpus_sha256,
    oracle_sha256 = oracle_sha256,
    backend_configuration_sha256 = backend_configuration_sha256,
    build_recipe_sha256 = build_recipe_sha256
  )
  for (name in names(hashes)) {
    .fastkpc_full_cuda_phase35_require_hash(hashes[[name]], name)
  }
  if (!.fastkpc_full_cuda_phase35_scalar_character(route_semantic_version) ||
      !nzchar(route_semantic_version)) {
    stop("route semantic version is malformed", call. = FALSE)
  }
  fields <- c(
    list(
      schema_version = "full-cuda-ci-producer-semantic-identity-v1",
      producer_source_closure_sha256 = producer_source_closure_sha256,
      native_binary_sha256 = native_binary_sha256,
      route_semantic_version = route_semantic_version,
      contract_snapshots =
        fastkpc_full_cuda_phase35_contract_snapshots(contracts),
      dataset_or_corpus_sha256 = dataset_or_corpus_sha256,
      oracle_sha256 = oracle_sha256,
      backend_configuration_sha256 = backend_configuration_sha256,
      build_recipe_sha256 = build_recipe_sha256
    )
  )
  .fastkpc_full_cuda_phase35_hash_fields(fields, "identity_sha256")
}

.fastkpc_full_cuda_phase35_validate_producer <- function(producer) {
  fields <- c(
    "schema_version", "producer_source_closure_sha256",
    "native_binary_sha256", "route_semantic_version", "contract_snapshots",
    "dataset_or_corpus_sha256", "oracle_sha256",
    "backend_configuration_sha256", "build_recipe_sha256",
    "identity_sha256"
  )
  if (!.fastkpc_full_cuda_phase35_plain_object(producer) ||
      !identical(names(producer), fields) ||
      !identical(producer$schema_version,
                 "full-cuda-ci-producer-semantic-identity-v1")) {
    stop("producer semantic identity is malformed", call. = FALSE)
  }
  for (field in c(
    "producer_source_closure_sha256", "native_binary_sha256",
    "dataset_or_corpus_sha256", "oracle_sha256",
    "backend_configuration_sha256", "build_recipe_sha256",
    "identity_sha256"
  )) {
    .fastkpc_full_cuda_phase35_require_hash(producer[[field]], field)
  }
  .fastkpc_full_cuda_phase35_validate_contract_snapshots(
    producer$contract_snapshots
  )
  unhashed <- producer
  actual <- unhashed$identity_sha256
  unhashed$identity_sha256 <- NULL
  expected <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(unhashed)
  )
  if (!identical(actual, expected)) {
    stop("producer identity hash mismatch", call. = FALSE)
  }
  invisible(TRUE)
}

fastkpc_full_cuda_phase35_validator_attestation <- function(
    producer,
    validator_source_closure_sha256,
    validator_semantic_version,
    validator_contracts = fastkpc_full_cuda_phase35_load_contract_set(),
    validation_timestamp_utc,
    environment_sha256,
    validation_result = c("PASS", "FAIL")) {
  .fastkpc_full_cuda_phase35_validate_producer(producer)
  validation_result <- match.arg(validation_result)
  .fastkpc_full_cuda_phase35_require_hash(
    validator_source_closure_sha256, "validator source closure"
  )
  .fastkpc_full_cuda_phase35_require_hash(
    environment_sha256, "validator environment"
  )
  if (!.fastkpc_full_cuda_phase35_scalar_character(
        validator_semantic_version
      ) || !nzchar(validator_semantic_version)) {
    stop("validator semantic version is malformed", call. = FALSE)
  }
  if (!.fastkpc_full_cuda_phase35_scalar_character(
        validation_timestamp_utc
      ) ||
      !grepl("^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$",
             validation_timestamp_utc)) {
    stop("validation timestamp must be canonical UTC", call. = FALSE)
  }
  fields <- list(
    schema_version = "full-cuda-ci-validator-attestation-v1",
    attested_producer_sha256 = producer$identity_sha256,
    validator_source_closure_sha256 = validator_source_closure_sha256,
    validator_semantic_version = validator_semantic_version,
    validator_contract_snapshots =
      fastkpc_full_cuda_phase35_contract_snapshots(validator_contracts),
    validation_timestamp_utc = validation_timestamp_utc,
    environment_sha256 = environment_sha256,
    validation_result = validation_result
  )
  .fastkpc_full_cuda_phase35_hash_fields(fields, "attestation_sha256")
}

.fastkpc_full_cuda_phase35_validate_attestation <- function(attestation) {
  fields <- c(
    "schema_version", "attested_producer_sha256",
    "validator_source_closure_sha256", "validator_semantic_version",
    "validator_contract_snapshots", "validation_timestamp_utc",
    "environment_sha256", "validation_result", "attestation_sha256"
  )
  if (!.fastkpc_full_cuda_phase35_plain_object(attestation) ||
      !identical(names(attestation), fields) ||
      !identical(attestation$schema_version,
                 "full-cuda-ci-validator-attestation-v1")) {
    stop("validator attestation is malformed", call. = FALSE)
  }
  for (field in c(
    "attested_producer_sha256", "validator_source_closure_sha256",
    "environment_sha256", "attestation_sha256"
  )) {
    .fastkpc_full_cuda_phase35_require_hash(attestation[[field]], field)
  }
  .fastkpc_full_cuda_phase35_validate_contract_snapshots(
    attestation$validator_contract_snapshots
  )
  unhashed <- attestation
  actual <- unhashed$attestation_sha256
  unhashed$attestation_sha256 <- NULL
  expected <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(unhashed)
  )
  if (!identical(actual, expected)) {
    stop("validator attestation hash mismatch", call. = FALSE)
  }
  invisible(TRUE)
}

fastkpc_full_cuda_phase35_execution_receipt <- function(
    producer, pid, session_id, cuda_context_id, artifact_path,
    artifact_inode, staging_path, recorded_at_utc) {
  .fastkpc_full_cuda_phase35_validate_producer(producer)
  if (!.fastkpc_full_cuda_phase35_scalar_integer(pid, 1L)) {
    stop("receipt PID is malformed", call. = FALSE)
  }
  strings <- list(
    session_id = session_id,
    cuda_context_id = cuda_context_id,
    artifact_path = artifact_path,
    artifact_inode = artifact_inode,
    staging_path = staging_path,
    recorded_at_utc = recorded_at_utc
  )
  for (name in names(strings)) {
    if (!.fastkpc_full_cuda_phase35_scalar_character(strings[[name]]) ||
        !nzchar(strings[[name]])) {
      stop("receipt field is malformed: ", name, call. = FALSE)
    }
  }
  fields <- list(
    schema_version = "full-cuda-ci-execution-receipt-v1",
    producer_sha256 = producer$identity_sha256,
    pid = pid,
    session_id = session_id,
    cuda_context_id = cuda_context_id,
    artifact_path = artifact_path,
    artifact_inode = artifact_inode,
    staging_path = staging_path,
    recorded_at_utc = recorded_at_utc
  )
  .fastkpc_full_cuda_phase35_hash_fields(fields, "receipt_sha256")
}

.fastkpc_full_cuda_phase35_validate_receipt <- function(receipt) {
  fields <- c(
    "schema_version", "producer_sha256", "pid", "session_id",
    "cuda_context_id", "artifact_path", "artifact_inode", "staging_path",
    "recorded_at_utc", "receipt_sha256"
  )
  if (!.fastkpc_full_cuda_phase35_plain_object(receipt) ||
      !identical(names(receipt), fields) ||
      !identical(receipt$schema_version,
                 "full-cuda-ci-execution-receipt-v1") ||
      !.fastkpc_full_cuda_phase35_scalar_integer(receipt$pid, 1L)) {
    stop("execution receipt is malformed", call. = FALSE)
  }
  .fastkpc_full_cuda_phase35_require_hash(
    receipt$producer_sha256, "receipt producer identity"
  )
  .fastkpc_full_cuda_phase35_require_hash(
    receipt$receipt_sha256, "receipt identity"
  )
  unhashed <- receipt
  actual <- unhashed$receipt_sha256
  unhashed$receipt_sha256 <- NULL
  expected <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(unhashed)
  )
  if (!identical(actual, expected)) {
    stop("execution receipt hash mismatch", call. = FALSE)
  }
  invisible(TRUE)
}

.fastkpc_full_cuda_phase35_artifact_semantic_hash <- function(
    producer_sha256, payload_manifest_sha256) {
  fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(list(
      payload_manifest_sha256 = payload_manifest_sha256,
      producer_identity_sha256 = producer_sha256
    ))
  )
}

fastkpc_full_cuda_phase35_identity_envelope <- function(
    producer, payload_manifest_sha256) {
  .fastkpc_full_cuda_phase35_validate_producer(producer)
  .fastkpc_full_cuda_phase35_require_hash(
    payload_manifest_sha256, "payload manifest identity"
  )
  list(
    schema_version = "full-cuda-ci-artifact-semantic-envelope-v1",
    producer = producer,
    payload_manifest_sha256 = payload_manifest_sha256,
    artifact_semantic_sha256 =
      .fastkpc_full_cuda_phase35_artifact_semantic_hash(
        producer$identity_sha256, payload_manifest_sha256
      ),
    attestations = list(),
    execution_receipts = list()
  )
}

fastkpc_full_cuda_phase35_validate_identity_envelope <- function(value) {
  fields <- c(
    "schema_version", "producer", "payload_manifest_sha256",
    "artifact_semantic_sha256", "attestations", "execution_receipts"
  )
  if (!.fastkpc_full_cuda_phase35_plain_object(value) ||
      !identical(names(value), fields) ||
      !identical(value$schema_version,
                 "full-cuda-ci-artifact-semantic-envelope-v1") ||
      !is.list(value$attestations) || !is.null(names(value$attestations)) ||
      !is.list(value$execution_receipts) ||
      !is.null(names(value$execution_receipts))) {
    stop("artifact identity envelope is malformed", call. = FALSE)
  }
  .fastkpc_full_cuda_phase35_validate_producer(value$producer)
  .fastkpc_full_cuda_phase35_require_hash(
    value$payload_manifest_sha256, "payload manifest identity"
  )
  expected <- .fastkpc_full_cuda_phase35_artifact_semantic_hash(
    value$producer$identity_sha256, value$payload_manifest_sha256
  )
  if (!identical(value$artifact_semantic_sha256, expected)) {
    stop("artifact semantic identity hash mismatch", call. = FALSE)
  }
  for (attestation in value$attestations) {
    .fastkpc_full_cuda_phase35_validate_attestation(attestation)
    if (!identical(
          attestation$attested_producer_sha256,
          value$producer$identity_sha256
        )) {
      stop("attested producer identity mismatch", call. = FALSE)
    }
  }
  for (receipt in value$execution_receipts) {
    .fastkpc_full_cuda_phase35_validate_receipt(receipt)
    if (!identical(receipt$producer_sha256,
                   value$producer$identity_sha256)) {
      stop("receipt producer identity mismatch", call. = FALSE)
    }
  }
  TRUE
}

fastkpc_full_cuda_phase35_append_attestation <- function(value,
                                                         attestation) {
  fastkpc_full_cuda_phase35_validate_identity_envelope(value)
  .fastkpc_full_cuda_phase35_validate_attestation(attestation)
  if (!identical(
        attestation$attested_producer_sha256,
        value$producer$identity_sha256
      )) {
    stop("attested producer identity mismatch", call. = FALSE)
  }
  existing <- vapply(
    value$attestations, `[[`, character(1L), "attestation_sha256"
  )
  if (attestation$attestation_sha256 %in% existing) {
    stop("duplicate validator attestation", call. = FALSE)
  }
  value$attestations[[length(value$attestations) + 1L]] <- attestation
  value
}

fastkpc_full_cuda_phase35_append_receipt <- function(value, receipt) {
  fastkpc_full_cuda_phase35_validate_identity_envelope(value)
  .fastkpc_full_cuda_phase35_validate_receipt(receipt)
  if (!identical(receipt$producer_sha256,
                 value$producer$identity_sha256)) {
    stop("receipt producer identity mismatch", call. = FALSE)
  }
  existing <- vapply(
    value$execution_receipts, `[[`, character(1L), "receipt_sha256"
  )
  if (receipt$receipt_sha256 %in% existing) {
    stop("duplicate execution receipt", call. = FALSE)
  }
  value$execution_receipts[[length(value$execution_receipts) + 1L]] <- receipt
  value
}

fastkpc_full_cuda_phase35_native_contract_identity <- function(
    path, expected_contract_name) {
  text <- .fastkpc_full_cuda_phase35_read_utf8(
    path, paste0(expected_contract_name, " native contract")
  )
  .Call(
    "C_full_cuda_ci_contract_identity", text, expected_contract_name,
    PACKAGE = "fastkpc_cuda"
  )
}

fastkpc_full_cuda_phase35_native_sha256_utf8 <- function(value) {
  if (!.fastkpc_full_cuda_phase35_scalar_character(value)) {
    stop("native SHA-256 input must be one string", call. = FALSE)
  }
  .Call("C_full_cuda_ci_sha256_utf8", value, PACKAGE = "fastkpc_cuda")
}

.fastkpc_full_cuda_phase35_validate_semantic_abi_info <- function(
    info, contracts) {
  required <- c(
    "schema_version", "abi_major", "abi_minor", "capabilities",
    "capability_status", "backend_semantic_version",
    "producer_contract_hash", "device_residency_flags", "semantic_objects",
    "compact_result_fields"
  )
  architecture <- contracts$architecture_contract_v1
  vocabulary <- unlist(
    architecture$payload$capabilities, use.names = FALSE
  )
  object_names <- vapply(
    architecture$payload$semantic_objects, `[[`, character(1L), "name"
  )
  compact_fields <- unlist(
    architecture$payload$compact_result_fields, use.names = FALSE
  )
  expected_residency <- c(
    "prepared_s_device_resident", "target_optimizer_device_resident",
    "residual_device_resident", "dcov_component_device_resident",
    "logical_ci_batch_device_resident", "compact_result_host_visible",
    "production_residual_d2h_forbidden",
    "production_component_d2h_forbidden"
  )
  clean_status <- is.character(info$capability_status) &&
    !anyNA(info$capability_status) &&
    identical(names(info$capability_status), vocabulary) &&
    all(info$capability_status %in%
        c("available", "interface_only", "unavailable"))
  clean_capabilities <- is.character(info$capabilities) &&
    !anyNA(info$capabilities) && !anyDuplicated(info$capabilities) &&
    identical(
      info$capabilities,
      vocabulary[info$capability_status == "available"]
    )
  clean_residency <- is.logical(info$device_residency_flags) &&
    !anyNA(info$device_residency_flags) &&
    identical(names(info$device_residency_flags), expected_residency)
  if (!is.list(info) || !identical(names(info), required) ||
      !identical(info$schema_version,
                 "full-cuda-ci-semantic-abi-info-v1") ||
      !.fastkpc_full_cuda_phase35_scalar_integer(info$abi_major, 1L) ||
      !.fastkpc_full_cuda_phase35_scalar_integer(info$abi_minor, 0L) ||
      !isTRUE(clean_status) || !isTRUE(clean_capabilities) ||
      !identical(info$backend_semantic_version,
                 architecture$payload$abi$backend_semantic_version) ||
      !identical(info$producer_contract_hash, architecture$sha256) ||
      !isTRUE(clean_residency) ||
      !identical(info$semantic_objects, object_names) ||
      !identical(info$compact_result_fields, compact_fields)) {
    stop("native semantic ABI information is malformed or unauthenticated",
         call. = FALSE)
  }
  invisible(TRUE)
}

fastkpc_full_cuda_phase35_semantic_abi_info <- function(
    contracts = fastkpc_full_cuda_phase35_load_contract_set()) {
  info <- .Call(
    "C_full_cuda_ci_semantic_abi_info", PACKAGE = "fastkpc_cuda"
  )
  .fastkpc_full_cuda_phase35_validate_semantic_abi_info(info, contracts)
  info
}

.fastkpc_full_cuda_phase35_capability_vector <- function(value, label) {
  if (is.null(value)) return(character())
  if (typeof(value) != "character" || is.object(value) || anyNA(value) ||
      any(!nzchar(value)) || anyDuplicated(value)) {
    stop(label, " must be unique non-missing capability strings",
         call. = FALSE)
  }
  unname(value)
}

fastkpc_full_cuda_phase35_negotiate_semantic_abi <- function(
    info,
    required_major = 1L,
    required_minor = 0L,
    required_capabilities = character(),
    optional_capabilities = character(),
    contracts = fastkpc_full_cuda_phase35_load_contract_set()) {
  .fastkpc_full_cuda_phase35_validate_semantic_abi_info(info, contracts)
  if (!.fastkpc_full_cuda_phase35_scalar_integer(required_major, 0L) ||
      !.fastkpc_full_cuda_phase35_scalar_integer(required_minor, 0L)) {
    stop("required semantic ABI version is malformed", call. = FALSE)
  }
  required_capabilities <- .fastkpc_full_cuda_phase35_capability_vector(
    required_capabilities, "required capabilities"
  )
  optional_capabilities <- .fastkpc_full_cuda_phase35_capability_vector(
    optional_capabilities, "optional capabilities"
  )
  if (length(intersect(required_capabilities, optional_capabilities)) > 0L) {
    stop("required and optional capabilities must be disjoint",
         call. = FALSE)
  }
  if (!identical(info$abi_major, required_major)) {
    stop("semantic ABI major mismatch", call. = FALSE)
  }
  if (info$abi_minor < required_minor) {
    stop("semantic ABI minor capability mismatch", call. = FALSE)
  }
  vocabulary <- names(info$capability_status)
  unknown_required <- setdiff(required_capabilities, vocabulary)
  if (length(unknown_required) > 0L) {
    stop("required semantic capability is unknown: ", unknown_required[[1L]],
         call. = FALSE)
  }
  unavailable_required <- required_capabilities[
    info$capability_status[required_capabilities] != "available"
  ]
  if (length(unavailable_required) > 0L) {
    stop(
      "required semantic capability is unavailable: ",
      unavailable_required[[1L]], call. = FALSE
    )
  }
  unavailable_optional <- optional_capabilities[
    !optional_capabilities %in% info$capabilities
  ]
  list(
    compatible = TRUE,
    provider_abi_major = info$abi_major,
    provider_abi_minor = info$abi_minor,
    available_required_capabilities = required_capabilities,
    unavailable_optional_capabilities = unavailable_optional,
    producer_contract_hash = info$producer_contract_hash
  )
}
