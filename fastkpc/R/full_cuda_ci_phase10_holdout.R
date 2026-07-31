fastkpc_full_cuda_phase10_holdout_require <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
  invisible(TRUE)
}

fastkpc_full_cuda_phase10_holdout_artifact_dir <- function(
    root = file.path("fastkpc", "artifacts", "full_cuda_ci")) {
  file.path(root, "sealed_promotion_holdout_v1")
}

fastkpc_full_cuda_phase10_holdout_staging_dir <- function(
    root = file.path("fastkpc", "artifacts", "full_cuda_ci")) {
  file.path(root, "phase10_holdout_release_staging_v1")
}

fastkpc_full_cuda_phase10_holdout_required_files <- function() {
  c(
    "manifest.json", "summary.json", "summary.md", "commands.txt",
    "environment.txt", "graph_agreement.csv", "sepset_agreement.csv",
    "n_edgetests.csv", "first_divergence.json", "fallbacks.csv",
    "stage_timing.csv", "raw_runs.csv", "case_results.csv",
    "near_alpha_results.csv", "rank_condition_results.csv", "cache.csv",
    "coverage.csv", "release_attestation.json", "open_event.json",
    "source_closure.csv", "source_evidence.rds", "producer_identity.json",
    "backend_configuration.json", "build_recipe.json",
    "validator_attestations.json", "execution_receipts.json"
  )
}

fastkpc_full_cuda_phase10_holdout_source_paths <- function() {
  paths <- sort(unique(c(
    fastkpc_full_cuda_phase10_campaign_source_paths(),
    "fastkpc/R/full_cuda_ci_phase10_holdout.R",
    "fastkpc/tests/test_full_cuda_ci_phase10_completion_audit.R",
    "fastkpc/tests/test_full_cuda_ci_phase10_holdout_artifact.R",
    "fastkpc/tests/test_full_cuda_ci_phase10_holdout_helpers.R",
    "fastkpc/tools/run_full_cuda_ci_phase10_holdout.R"
  )), method = "radix")
  fastkpc_full_cuda_phase10_holdout_require(
    all(file.exists(paths) & !dir.exists(paths)) && !anyDuplicated(paths),
    "Phase 10 holdout source closure is incomplete"
  )
  paths
}

fastkpc_full_cuda_phase10_holdout_source_closure <- function() {
  paths <- fastkpc_full_cuda_phase10_holdout_source_paths()
  hashes <- setNames(as.list(vapply(
    paths, fastkpc_full_cuda_census_file_hash, character(1L)
  )), paths)
  list(
    table = data.frame(
      path = names(hashes),
      sha256 = unlist(hashes, use.names = FALSE),
      stringsAsFactors = FALSE
    ),
    hashes = hashes,
    sha256 = fastkpc_full_cuda_phase35_sha256_utf8(
      fastkpc_full_cuda_phase35_canonical_json(hashes)
    )
  )
}

fastkpc_full_cuda_phase10_holdout_attestation_fields <- function() {
  c(
    "schema_version", "custody_authority", "holdout_id",
    "contract_manifest_identity_sha256", "payload_file",
    "payload_sha256", "release_token_sha256", "released_at_utc",
    "attestation_identity_sha256"
  )
}

fastkpc_full_cuda_phase10_holdout_contract <- function(
    contracts = fastkpc_full_cuda_phase35_load_contract_set()) {
  contracts$promotion_holdout_manifest_v1$payload
}

fastkpc_full_cuda_phase10_holdout_validate_attestation <- function(
    attestation, token = NULL,
    contracts = fastkpc_full_cuda_phase35_load_contract_set()) {
  contract <- fastkpc_full_cuda_phase10_holdout_contract(contracts)
  clean <- is.list(attestation) && !is.object(attestation) &&
    identical(names(attestation),
              fastkpc_full_cuda_phase10_holdout_attestation_fields()) &&
    identical(
      attestation$schema_version,
      "full-cuda-ci-promotion-holdout-release-attestation-v1"
    ) && identical(attestation$custody_authority,
                  contract$custody$authority) &&
    identical(attestation$holdout_id, contract$holdout_id) &&
    identical(
      attestation$contract_manifest_identity_sha256,
      contract$commitment$manifest_identity_sha256
    ) && is.character(attestation$payload_file) &&
    length(attestation$payload_file) == 1L &&
    identical(basename(attestation$payload_file),
              attestation$payload_file) &&
    !attestation$payload_file %in% c("", ".", "..") &&
    all(vapply(c(
      "payload_sha256", "release_token_sha256",
      "attestation_identity_sha256"
    ), function(field) {
      is.character(attestation[[field]]) &&
        length(attestation[[field]]) == 1L &&
        grepl("^[0-9a-f]{64}$", attestation[[field]])
    }, logical(1L))) && is.character(attestation$released_at_utc) &&
    length(attestation$released_at_utc) == 1L && grepl(
      "^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$",
      attestation$released_at_utc
    )
  fastkpc_full_cuda_phase10_holdout_require(
    clean, "Phase 10 holdout custodian attestation is malformed"
  )
  core <- attestation
  claimed <- core$attestation_identity_sha256
  core$attestation_identity_sha256 <- NULL
  expected <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(core)
  )
  fastkpc_full_cuda_phase10_holdout_require(
    identical(claimed, expected),
    "Phase 10 holdout custodian attestation identity mismatch"
  )
  if (!is.null(token)) {
    fastkpc_full_cuda_phase10_holdout_require(
      is.character(token) && length(token) == 1L &&
        !is.na(token) && nchar(token, type = "bytes") >= 32L &&
        identical(
          fastkpc_full_cuda_phase35_sha256_utf8(token),
          attestation$release_token_sha256
        ),
      "Phase 10 holdout release token is invalid"
    )
  }
  invisible(attestation)
}

fastkpc_full_cuda_phase10_holdout_open_event <- function(
    campaign, attestation, token_sha256, recorded_at_utc) {
  value <- list(
    schema_version = "full-cuda-ci-promotion-holdout-open-event-v1",
    holdout_id = attestation$holdout_id,
    custody_authority = attestation$custody_authority,
    campaign_producer_identity_sha256 = campaign$producer$identity_sha256,
    campaign_freeze_identity_sha256 =
      campaign$evidence$freeze$freeze_identity_sha256,
    release_attestation_identity_sha256 =
      attestation$attestation_identity_sha256,
    payload_sha256 = attestation$payload_sha256,
    release_token_sha256 = token_sha256,
    recorded_at_utc = recorded_at_utc,
    state = "OPEN_RECORDED_PAYLOAD_UNREAD"
  )
  value$open_event_identity_sha256 <-
    fastkpc_full_cuda_phase35_sha256_utf8(
      fastkpc_full_cuda_phase35_canonical_json(value)
    )
  value
}

fastkpc_full_cuda_phase10_holdout_validate_open_event <- function(
    event, campaign = NULL, attestation = NULL) {
  fields <- c(
    "schema_version", "holdout_id", "custody_authority",
    "campaign_producer_identity_sha256", "campaign_freeze_identity_sha256",
    "release_attestation_identity_sha256", "payload_sha256",
    "release_token_sha256", "recorded_at_utc", "state",
    "open_event_identity_sha256"
  )
  hash_fields <- c(
    "campaign_producer_identity_sha256", "campaign_freeze_identity_sha256",
    "release_attestation_identity_sha256", "payload_sha256",
    "release_token_sha256", "open_event_identity_sha256"
  )
  clean <- is.list(event) && !is.object(event) &&
    identical(names(event), fields) && identical(
      event$schema_version,
      "full-cuda-ci-promotion-holdout-open-event-v1"
    ) && identical(event$holdout_id,
                  "full-cuda-ci-promotion-holdout-v1") &&
    identical(event$custody_authority, "external-release-envelope") &&
    identical(event$state, "OPEN_RECORDED_PAYLOAD_UNREAD") &&
    all(vapply(hash_fields, function(field) {
      is.character(event[[field]]) && length(event[[field]]) == 1L &&
        grepl("^[0-9a-f]{64}$", event[[field]])
    }, logical(1L))) && grepl(
      "^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$",
      event$recorded_at_utc
    )
  fastkpc_full_cuda_phase10_holdout_require(
    clean, "Phase 10 holdout open event is malformed"
  )
  core <- event
  claimed <- core$open_event_identity_sha256
  core$open_event_identity_sha256 <- NULL
  fastkpc_full_cuda_phase10_holdout_require(
    identical(
      claimed,
      fastkpc_full_cuda_phase35_sha256_utf8(
        fastkpc_full_cuda_phase35_canonical_json(core)
      )
    ),
    "Phase 10 holdout open event identity mismatch"
  )
  if (!is.null(campaign)) {
    fastkpc_full_cuda_phase10_holdout_require(
      identical(event$campaign_producer_identity_sha256,
                campaign$producer$identity_sha256) &&
        identical(event$campaign_freeze_identity_sha256,
                  campaign$evidence$freeze$freeze_identity_sha256),
      "Phase 10 holdout open event campaign identity mismatch"
    )
  }
  if (!is.null(attestation)) {
    fastkpc_full_cuda_phase10_holdout_require(
      identical(event$release_attestation_identity_sha256,
                attestation$attestation_identity_sha256) &&
        identical(event$payload_sha256, attestation$payload_sha256) &&
        identical(event$release_token_sha256,
                  attestation$release_token_sha256),
      "Phase 10 holdout open event release identity mismatch"
    )
  }
  invisible(event)
}

fastkpc_full_cuda_phase10_holdout_write_json_atomic <- function(value, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- tempfile(".phase10-holdout-json-", tmpdir = dirname(path))
  on.exit(unlink(temporary, force = TRUE), add = TRUE)
  fastkpc_full_cuda_write_json(value, temporary)
  fastkpc_full_cuda_phase10_holdout_require(
    file.rename(temporary, path),
    "Phase 10 holdout audit record could not be committed atomically"
  )
  invisible(path)
}

fastkpc_full_cuda_phase10_holdout_parse_s_size <- function(value) {
  value <- as.character(value)
  ifelse(!nzchar(value), 0L, lengths(strsplit(value, "|", fixed = TRUE)))
}

fastkpc_full_cuda_phase10_holdout_validate_oracle <- function(oracle, p) {
  clean <- is.list(oracle) &&
    identical(names(oracle), c(
      "reference", "deletion_trace", "logical_trace"
    )) && fastkpc_full_cuda_is_skeleton(oracle$reference) &&
    is.data.frame(oracle$deletion_trace) &&
    is.data.frame(oracle$logical_trace) &&
    identical(dim(oracle$reference$adjacency), c(p, p)) &&
    identical(dim(oracle$reference$pMax), c(p, p)) &&
    length(oracle$reference$sepsets) == p &&
    sum(as.integer(oracle$reference$n.edgetests)) ==
      nrow(oracle$logical_trace) &&
    all(is.finite(oracle$logical_trace$p_value))
  fastkpc_full_cuda_phase10_holdout_require(
    clean, "Phase 10 holdout oracle is malformed"
  )
  fastkpc_full_cuda_validate_logical_trace(
    oracle$logical_trace, oracle$reference$n.edgetests,
    role = "promotion holdout oracle"
  )
  fastkpc_full_cuda_validate_logical_trace_plan(
    oracle$logical_trace, p = p,
    level_count = length(oracle$reference$n.edgetests),
    expected_adjacency = oracle$reference$adjacency,
    role = "promotion holdout oracle"
  )
  invisible(oracle)
}

fastkpc_full_cuda_phase10_holdout_case_diagnostics <- function(case) {
  data <- case$data
  standard_deviation <- apply(data, 2L, stats::sd)
  correlation <- suppressWarnings(stats::cor(data))
  diag(correlation) <- 0
  max_abs_correlation <- suppressWarnings(max(abs(correlation), na.rm = TRUE))
  if (!is.finite(max_abs_correlation)) max_abs_correlation <- 0
  centered <- sweep(data, 2L, colMeans(data), "-")
  singular <- svd(centered, nu = 0L, nv = 0L)$d
  tolerance <- max(dim(centered)) * max(singular, 0) *
    .Machine$double.eps
  rank <- sum(singular > tolerance)
  condition <- if (length(singular) == 0L || min(singular) <= tolerance) {
    Inf
  } else max(singular) / min(singular)
  logical <- case$oracle$logical_trace
  s_size <- fastkpc_full_cuda_phase10_holdout_parse_s_size(logical$S_key)
  near <- is.finite(logical$p_value) & logical$p_value > 0 &
    abs(log(logical$p_value / case$alpha)) <= log(2)
  list(
    case_id = case$case_id,
    n = nrow(data), p = ncol(data),
    matrix_rank = rank, matrix_condition = condition,
    max_abs_correlation = max_abs_correlation,
    near_constant_column_count = sum(
      !is.finite(standard_deviation) |
        standard_deviation <= sqrt(.Machine$double.eps)
    ),
    has_s_size_1 = any(s_size == 1L),
    has_s_size_2 = any(s_size == 2L),
    has_s_size_gt_2 = any(s_size > 2L),
    near_alpha_count = sum(near),
    logical_test_count = nrow(logical)
  )
}

fastkpc_full_cuda_phase10_validate_holdout_payload <- function(payload) {
  fastkpc_full_cuda_phase10_holdout_require(
    is.list(payload) && identical(names(payload), c(
      "schema_version", "holdout_id", "corpus_id", "cases"
    )) && identical(
      payload$schema_version,
      "full-cuda-ci-promotion-holdout-payload-v1"
    ) && identical(payload$holdout_id,
                  "full-cuda-ci-promotion-holdout-v1") &&
      is.character(payload$corpus_id) && length(payload$corpus_id) == 1L &&
      nzchar(payload$corpus_id) && is.list(payload$cases) &&
      length(payload$cases) > 0L,
    "Phase 10 holdout payload is malformed"
  )
  case_ids <- character(length(payload$cases))
  diagnostics <- vector("list", length(payload$cases))
  for (index in seq_along(payload$cases)) {
    case <- payload$cases[[index]]
    clean <- is.list(case) && identical(names(case), c(
      "case_id", "data", "alpha", "max_conditioning_size", "index",
      "num_col", "oracle"
    )) && is.character(case$case_id) && length(case$case_id) == 1L &&
      nzchar(case$case_id) && is.matrix(case$data) &&
      is.numeric(case$data) && nrow(case$data) >= 8L &&
      ncol(case$data) >= 4L && !is.null(colnames(case$data)) &&
      !anyDuplicated(colnames(case$data)) && all(is.finite(case$data)) &&
      identical(as.numeric(case$alpha), 0.1) &&
      length(case$max_conditioning_size) == 1L &&
      as.integer(case$max_conditioning_size) >= 3L &&
      as.integer(case$max_conditioning_size) <= ncol(case$data) - 2L &&
      identical(as.integer(case$index), 1L) &&
      identical(as.integer(case$num_col), 35L)
    fastkpc_full_cuda_phase10_holdout_require(
      clean, paste0("Phase 10 holdout case is malformed: ", index)
    )
    fastkpc_full_cuda_phase10_holdout_validate_oracle(
      case$oracle, ncol(case$data)
    )
    case_ids[[index]] <- case$case_id
    diagnostics[[index]] <-
      fastkpc_full_cuda_phase10_holdout_case_diagnostics(case)
  }
  fastkpc_full_cuda_phase10_holdout_require(
    !anyDuplicated(case_ids), "Phase 10 holdout case ids are duplicated"
  )
  coverage <- list(
    different_n = any(vapply(diagnostics, `[[`, integer(1L), "n") != 351L),
    different_p = any(vapply(diagnostics, `[[`, integer(1L), "p") != 48L),
    s_size_1 = any(vapply(diagnostics, `[[`, logical(1L), "has_s_size_1")),
    s_size_2 = any(vapply(diagnostics, `[[`, logical(1L), "has_s_size_2")),
    s_size_gt_2 = any(vapply(
      diagnostics, `[[`, logical(1L), "has_s_size_gt_2"
    )),
    collinearity = any(vapply(diagnostics, function(value) {
      value$matrix_rank < value$p || value$matrix_condition >= 1e8 ||
        value$max_abs_correlation >= 0.999
    }, logical(1L))),
    near_constants = any(vapply(
      diagnostics, `[[`, integer(1L), "near_constant_column_count"
    ) > 0L),
    multiple_penalty_counts = any(vapply(
      diagnostics, `[[`, logical(1L), "has_s_size_gt_2"
    )),
    near_alpha_decisions = any(vapply(
      diagnostics, `[[`, integer(1L), "near_alpha_count"
    ) > 0L)
  )
  fastkpc_full_cuda_phase10_holdout_require(
    all(unlist(coverage, use.names = FALSE)),
    "Phase 10 holdout payload does not cover the frozen promotion envelope"
  )
  list(case_ids = case_ids, diagnostics = diagnostics,
       coverage = coverage, pass = TRUE)
}

fastkpc_full_cuda_phase10_holdout_capture_case <- function(case) {
  fastkpc_full_cuda_phase10_campaign_cache_control("configure", 262144L)
  fastkpc_full_cuda_phase10_campaign_cache_control(
    "configure_target", 131072L
  )
  fastkpc_full_cuda_phase10_campaign_cache_control("reset")
  resource_before <- fastkpc_full_cuda_phase10_campaign_resource_snapshot()
  started <- Sys.time()
  timing <- system.time(result <- fastkpc_compatible_cuda_skeleton(
    data = case$data,
    alpha = case$alpha,
    labels = colnames(case$data),
    options = list(
      route = "full_cuda", compatible_cuda_strict = TRUE,
      max_conditioning_size = as.integer(case$max_conditioning_size),
      index = as.integer(case$index), numCol = as.integer(case$num_col),
      trace_level = "logical"
    )
  ))
  ended <- Sys.time()
  resource_after <- fastkpc_full_cuda_phase10_campaign_resource_snapshot()
  comparison <- fastkpc_full_cuda_compare_candidate_skeleton(
    case$oracle, result
  )
  summary <- result$summary
  zero_fields <- fastkpc_full_cuda_phase10_campaign_authority_zero_fields()
  zero_values <- vapply(zero_fields, function(field) {
    value <- summary[[field]]
    if (is.null(value)) NA_real_ else as.numeric(value)
  }, numeric(1L))
  logical <- case$oracle$logical_trace
  tasks <- result$tasks
  decision_flip <- (as.numeric(tasks$p_used) >= case$alpha) !=
    (as.numeric(logical$p_value) >= case$alpha)
  clean <- isTRUE(comparison$summary$pass) &&
    nrow(tasks) == nrow(logical) &&
    identical(as.integer(result$n.edgetests),
              as.integer(case$oracle$reference$n.edgetests)) &&
    identical(as.integer(tasks$canonical_test_order_id),
              as.integer(logical$logical_sequence_id)) &&
    all(is.finite(tasks$p_used)) && !any(decision_flip) &&
    all(!is.na(zero_values) & zero_values == 0) &&
    identical(summary$run_status, "ok") &&
    identical(summary$entrypoint,
              "compatible-cuda-full-skeleton-native-v1") &&
    identical(summary$compatible_cuda_route, "compatible.cuda") &&
    isTRUE(summary$compatible_cuda_strict) &&
    isTRUE(summary$authority_gate_pass) &&
    as.integer(summary$native_call_count) == 1L &&
    as.integer(summary$logical_tests_consumed) == nrow(logical) &&
    as.integer(summary$speculative_tests_ignored) == 0L &&
    identical(summary$scheduler, "cache-aware-frontier-4x-v1") &&
    as.integer(summary$result_cache_capacity) == 262144L &&
    as.integer(summary$target_cache_capacity) == 131072L &&
    as.integer(summary$native_setup_cache_capacity) == 64L &&
    as.integer(summary$component_cache_capacity) == 47L &&
    all(fastkpc_full_cuda_phase10_campaign_active_resources(
      resource_before
    ) == 0) && all(fastkpc_full_cuda_phase10_campaign_active_resources(
      resource_after
    ) == 0) && is.finite(timing[["elapsed"]]) && timing[["elapsed"]] > 0
  fastkpc_full_cuda_phase10_holdout_require(
    clean,
    paste0("Phase 10 holdout case failed: ", case$case_id)
  )
  list(
    schema_version = "full-cuda-ci-phase10-holdout-case-result-v1",
    case_id = case$case_id,
    started_utc = format(started, tz = "UTC", usetz = TRUE),
    ended_utc = format(ended, tz = "UTC", usetz = TRUE),
    elapsed_sec = as.numeric(timing[["elapsed"]]),
    timing = timing,
    result = result,
    comparison = comparison,
    decision_flip_count = sum(decision_flip),
    authority_zero_values = zero_values,
    cache_after = full_cuda_ci_one_call_cache_control_native("info"),
    resource_before = resource_before,
    resource_after = resource_after,
    pass = TRUE
  )
}

fastkpc_full_cuda_phase10_holdout_backend_configuration <- function(
    campaign) {
  value <- list(
    schema_version = "full-cuda-ci-phase10-holdout-configuration-v1",
    campaign_producer_identity_sha256 = campaign$producer$identity_sha256,
    campaign_freeze_identity_sha256 =
      campaign$evidence$freeze$freeze_identity_sha256,
    candidate_route = "compatible.cuda/full_cuda-explicit",
    native_entrypoint = "compatible-cuda-full-skeleton-native-v1",
    scheduler = "cache-aware-frontier-4x-v1",
    compact_result_cache_capacity = 262144L,
    target_state_cache_capacity = 131072L,
    prepared_setup_cache_capacity = 64L,
    component_capacity = 47L,
    strict_fail_closed = TRUE,
    run_once = TRUE,
    holdout_id = "full-cuda-ci-promotion-holdout-v1",
    precision = "float64", fmad = FALSE, fast_math = FALSE
  )
  list(
    value = value,
    sha256 = fastkpc_full_cuda_phase35_sha256_utf8(
      fastkpc_full_cuda_phase35_canonical_json(value)
    )
  )
}

fastkpc_full_cuda_phase10_holdout_build_recipe <- function(campaign) {
  value <- list(
    schema_version = "full-cuda-ci-phase10-holdout-build-recipe-v1",
    campaign_build_recipe_sha256 =
      campaign$producer$build_recipe_sha256,
    native_binary_sha256 = campaign$producer$native_binary_sha256,
    runner_source = "fastkpc/tools/run_full_cuda_ci_phase10_holdout.R",
    fmad = FALSE, fast_math = FALSE
  )
  list(
    value = value,
    sha256 = fastkpc_full_cuda_phase35_sha256_utf8(
      fastkpc_full_cuda_phase35_canonical_json(value)
    )
  )
}

fastkpc_full_cuda_phase10_holdout_producer <- function(
    evidence, source_closure, campaign,
    contracts = fastkpc_full_cuda_phase35_load_contract_set()) {
  backend <- fastkpc_full_cuda_phase10_holdout_backend_configuration(campaign)
  build <- fastkpc_full_cuda_phase10_holdout_build_recipe(campaign)
  producer <- fastkpc_full_cuda_phase35_producer_identity(
    producer_source_closure_sha256 = source_closure$sha256,
    native_binary_sha256 = campaign$producer$native_binary_sha256,
    route_semantic_version =
      "full-cuda-ci-phase10-sealed-promotion-holdout-v1",
    dataset_or_corpus_sha256 = evidence$payload_sha256,
    oracle_sha256 = evidence$attestation$attestation_identity_sha256,
    backend_configuration_sha256 = backend$sha256,
    build_recipe_sha256 = build$sha256,
    contracts = contracts
  )
  list(producer = producer, backend = backend, build = build)
}

fastkpc_full_cuda_phase10_holdout_case_rows <- function(evidence) {
  do.call(rbind, lapply(seq_along(evidence$results), function(index) {
    run <- evidence$results[[index]]
    case <- evidence$payload_metadata[[index]]
    logical <- run$comparison$candidate_logical
    tasks <- run$result$tasks
    data.frame(
      case_id = run$case_id,
      logical_sequence_id = as.integer(logical$logical_sequence_id),
      level = as.integer(logical$level),
      x = as.integer(logical$x), y = as.integer(logical$y),
      S_key = as.character(logical$S_key), alpha = case$alpha,
      reference_p_value = as.numeric(case$reference_p_value),
      candidate_p_value = as.numeric(tasks$p_used),
      reference_independent = as.logical(
        case$reference_p_value >= case$alpha
      ),
      candidate_independent = as.logical(tasks$p_used >= case$alpha),
      decision_flip = as.logical(
        (tasks$p_used >= case$alpha) !=
          (case$reference_p_value >= case$alpha)
      ),
      deletes_edge = as.logical(logical$deletes_edge),
      absolute_log_distance_from_alpha = abs(log(
        pmax(case$reference_p_value, .Machine$double.xmin) / case$alpha
      )),
      stringsAsFactors = FALSE
    )
  }))
}

fastkpc_full_cuda_phase10_holdout_summary <- function(
    evidence, producer_bundle, source_evidence_sha256, campaign, contracts) {
  results <- evidence$results
  comparisons <- lapply(results, `[[`, "comparison")
  list(
    schema_version = "full-cuda-ci-phase10-holdout-summary-v1",
    run_status = "ok", timeout = FALSE,
    holdout_id = evidence$attestation$holdout_id,
    holdout_state = "OPENED_VALIDATED",
    release_event_persisted_before_payload_read = TRUE,
    run_once_gate = TRUE,
    case_count = length(results),
    logical_test_count = sum(vapply(results, function(run) {
      nrow(run$result$tasks)
    }, integer(1L))),
    maximum_SHD = max(vapply(comparisons, function(value) {
      as.integer(value$summary$SHD)
    }, integer(1L))),
    adjacency_identical = all(vapply(comparisons, function(value) {
      isTRUE(value$summary$adjacency_identical)
    }, logical(1L))),
    sepsets_identical = all(vapply(comparisons, function(value) {
      isTRUE(value$summary$sepsets_identical)
    }, logical(1L))),
    n_edgetests_identical = all(vapply(comparisons, function(value) {
      isTRUE(value$summary$n_edgetests_identical)
    }, logical(1L))),
    deletions_identical = all(vapply(comparisons, function(value) {
      isTRUE(value$summary$deletions_identical)
    }, logical(1L))),
    logical_ci_trace_identical = all(vapply(comparisons, function(value) {
      isTRUE(value$summary$logical_ci_trace_identical)
    }, logical(1L))),
    decision_flip_count = sum(vapply(
      results, `[[`, integer(1L), "decision_flip_count"
    )),
    unknown_fallback_count = 0L,
    approximate_backend_count = 0L,
    cpu_numerical_fallback_count = 0L,
    nonfinite_result_count = 0L,
    coverage = evidence$coverage,
    coverage_gate = all(unlist(evidence$coverage, use.names = FALSE)),
    canonical_campaign_producer_identity_sha256 =
      campaign$producer$identity_sha256,
    canonical_campaign_freeze_identity_sha256 =
      campaign$evidence$freeze$freeze_identity_sha256,
    canonical_campaign_gate = isTRUE(campaign$summary$pass),
    payload_sha256 = evidence$payload_sha256,
    release_attestation_identity_sha256 =
      evidence$attestation$attestation_identity_sha256,
    open_event_identity_sha256 =
      evidence$open_event$open_event_identity_sha256,
    architecture_contract_sha256 = contracts$architecture_contract_v1$sha256,
    numerical_contract_sha256 = contracts$numerical_contract_v1$sha256,
    artifact_identity_contract_sha256 =
      contracts$artifact_identity_contract_v1$sha256,
    promotion_holdout_contract_sha256 =
      contracts$promotion_holdout_manifest_v1$sha256,
    source_evidence_sha256 = source_evidence_sha256,
    producer_identity_sha256 =
      producer_bundle$producer$identity_sha256,
    source_closure_sha256 =
      producer_bundle$producer$producer_source_closure_sha256,
    native_binary_sha256 =
      producer_bundle$producer$native_binary_sha256,
    elapsed_sec = sum(vapply(results, `[[`, numeric(1L), "elapsed_sec")),
    phase10_promotion_claim = TRUE,
    recommended_compatible_cuda = TRUE,
    possible_default_requires_explicit_approval = TRUE,
    pass = TRUE
  )
}

fastkpc_full_cuda_phase10_holdout_validate_summary <- function(summary) {
  required_coverage <- c(
    "different_n", "different_p", "s_size_1", "s_size_2",
    "s_size_gt_2", "collinearity", "near_constants",
    "multiple_penalty_counts", "near_alpha_decisions"
  )
  hash_fields <- c(
    "canonical_campaign_producer_identity_sha256",
    "canonical_campaign_freeze_identity_sha256", "payload_sha256",
    "release_attestation_identity_sha256", "open_event_identity_sha256",
    "architecture_contract_sha256", "numerical_contract_sha256",
    "artifact_identity_contract_sha256",
    "promotion_holdout_contract_sha256", "source_evidence_sha256",
    "producer_identity_sha256", "source_closure_sha256",
    "native_binary_sha256"
  )
  clean <- is.list(summary) && identical(
    summary$schema_version, "full-cuda-ci-phase10-holdout-summary-v1"
  ) && identical(summary$run_status, "ok") && !isTRUE(summary$timeout) &&
    identical(summary$holdout_id,
              "full-cuda-ci-promotion-holdout-v1") &&
    identical(summary$holdout_state, "OPENED_VALIDATED") &&
    isTRUE(summary$release_event_persisted_before_payload_read) &&
    isTRUE(summary$run_once_gate) && summary$case_count >= 1L &&
    summary$logical_test_count >= 1L && summary$maximum_SHD == 0L &&
    isTRUE(summary$adjacency_identical) &&
    isTRUE(summary$sepsets_identical) &&
    isTRUE(summary$n_edgetests_identical) &&
    isTRUE(summary$deletions_identical) &&
    isTRUE(summary$logical_ci_trace_identical) &&
    summary$decision_flip_count == 0L &&
    summary$unknown_fallback_count == 0L &&
    summary$approximate_backend_count == 0L &&
    summary$cpu_numerical_fallback_count == 0L &&
    summary$nonfinite_result_count == 0L &&
    is.list(summary$coverage) &&
    identical(names(summary$coverage), required_coverage) &&
    all(unlist(summary$coverage, use.names = FALSE)) &&
    isTRUE(summary$coverage_gate) &&
    isTRUE(summary$canonical_campaign_gate) &&
    isTRUE(summary$phase10_promotion_claim) &&
    isTRUE(summary$recommended_compatible_cuda) &&
    isTRUE(summary$possible_default_requires_explicit_approval) &&
    is.finite(summary$elapsed_sec) && summary$elapsed_sec > 0 &&
    isTRUE(summary$pass) && all(vapply(hash_fields, function(field) {
      is.character(summary[[field]]) && length(summary[[field]]) == 1L &&
        grepl("^[0-9a-f]{64}$", summary[[field]])
    }, logical(1L)))
  fastkpc_full_cuda_phase10_holdout_require(
    clean, "Phase 10 holdout summary gate failed"
  )
  invisible(TRUE)
}

fastkpc_full_cuda_phase10_holdout_write_table <- function(
    value, directory, name) {
  utils::write.csv(
    value, file.path(directory, name), row.names = FALSE, na = ""
  )
}

fastkpc_full_cuda_phase10_publish_holdout <- function(
    evidence, campaign,
    output_dir = fastkpc_full_cuda_phase10_holdout_artifact_dir()) {
  contracts <- fastkpc_full_cuda_phase35_load_contract_set()
  source_closure <- fastkpc_full_cuda_phase10_holdout_source_closure()
  producer_bundle <- fastkpc_full_cuda_phase10_holdout_producer(
    evidence, source_closure, campaign, contracts
  )
  parent <- dirname(output_dir)
  dir.create(parent, recursive = TRUE, showWarnings = FALSE)
  stage <- tempfile(".phase10-holdout-stage-", tmpdir = parent)
  dir.create(stage, recursive = TRUE)
  active <- TRUE
  on.exit({
    if (active && dir.exists(stage)) {
      unlink(stage, recursive = TRUE, force = TRUE)
    }
  }, add = TRUE)
  saveRDS(evidence, file.path(stage, "source_evidence.rds"), compress = "xz")
  evidence_sha256 <- fastkpc_full_cuda_census_file_hash(
    file.path(stage, "source_evidence.rds")
  )
  summary <- fastkpc_full_cuda_phase10_holdout_summary(
    evidence, producer_bundle, evidence_sha256, campaign, contracts
  )
  fastkpc_full_cuda_phase10_holdout_validate_summary(summary)
  cases <- fastkpc_full_cuda_phase10_holdout_case_rows(evidence)
  near <- cases[
    is.finite(cases$absolute_log_distance_from_alpha) &
      cases$absolute_log_distance_from_alpha <= log(2),
    , drop = FALSE
  ]
  graph <- do.call(rbind, lapply(evidence$results, function(run) {
    value <- run$comparison$graph_agreement
    value$case_id <- run$case_id
    value[, c("case_id", setdiff(names(value), "case_id"))]
  }))
  sepsets <- do.call(rbind, lapply(evidence$results, function(run) {
    value <- run$comparison$sepset_agreement
    value$case_id <- run$case_id
    value[, c("case_id", setdiff(names(value), "case_id"))]
  }))
  n_edgetests <- do.call(rbind, lapply(evidence$results, function(run) {
    value <- run$comparison$n_edgetests
    value$case_id <- run$case_id
    value[, c("case_id", setdiff(names(value), "case_id"))]
  }))
  timing <- do.call(rbind, lapply(evidence$results, function(run) {
    data.frame(
      case_id = run$case_id, elapsed_sec = run$elapsed_sec,
      logical_tests = nrow(run$result$tasks),
      physical_tests = run$result$summary$physical_tests_evaluated,
      pass = run$pass, stringsAsFactors = FALSE
    )
  }))
  raw_runs <- timing
  raw_runs$SHD <- vapply(evidence$results, function(run) {
    as.integer(run$comparison$summary$SHD)
  }, integer(1L))
  raw_runs$decision_flip_count <- vapply(
    evidence$results, `[[`, integer(1L), "decision_flip_count"
  )
  cache <- do.call(rbind, lapply(evidence$results, function(run) {
    source <- run$result$summary
    data.frame(
      case_id = run$case_id,
      cache = c("compact-result", "target-state", "native-prepared-setup"),
      capacity = c(
        source$result_cache_capacity, source$target_cache_capacity,
        source$native_setup_cache_capacity
      ),
      requests = c(
        source$result_cache_request_count,
        source$target_cache_request_count,
        source$native_setup_cache_request_count
      ),
      hits = c(
        source$result_cache_hit_count, source$target_cache_hit_count,
        source$native_setup_cache_hit_count
      ),
      misses = c(
        source$result_cache_miss_count, source$target_cache_miss_count,
        source$native_setup_cache_miss_count
      ),
      evictions = c(
        source$result_cache_eviction_count,
        source$target_cache_eviction_count,
        source$native_setup_cache_eviction_count
      ),
      bounded = TRUE, stringsAsFactors = FALSE
    )
  }))
  diagnostic_rows <- do.call(rbind, lapply(
    evidence$diagnostics, function(value) data.frame(
      case_id = value$case_id, n = value$n, p = value$p,
      matrix_rank = value$matrix_rank,
      matrix_condition = value$matrix_condition,
      max_abs_correlation = value$max_abs_correlation,
      near_constant_column_count = value$near_constant_column_count,
      has_s_size_1 = value$has_s_size_1,
      has_s_size_2 = value$has_s_size_2,
      has_s_size_gt_2 = value$has_s_size_gt_2,
      near_alpha_count = value$near_alpha_count,
      logical_test_count = value$logical_test_count,
      stringsAsFactors = FALSE
    )
  ))
  coverage <- data.frame(
    coverage_class = names(evidence$coverage),
    covered = as.logical(unlist(evidence$coverage, use.names = FALSE)),
    stringsAsFactors = FALSE
  )
  fastkpc_full_cuda_phase10_holdout_write_table(
    graph, stage, "graph_agreement.csv"
  )
  fastkpc_full_cuda_phase10_holdout_write_table(
    sepsets, stage, "sepset_agreement.csv"
  )
  fastkpc_full_cuda_phase10_holdout_write_table(
    n_edgetests, stage, "n_edgetests.csv"
  )
  fastkpc_full_cuda_write_json(
    list(first_divergence_found = FALSE, cases = list()),
    file.path(stage, "first_divergence.json")
  )
  fastkpc_full_cuda_phase10_holdout_write_table(data.frame(
    fallback_class = c("unknown", "approximate", "cpu-numerical"),
    count = 0L, accepted_for_phase10 = FALSE,
    stringsAsFactors = FALSE
  ), stage, "fallbacks.csv")
  fastkpc_full_cuda_phase10_holdout_write_table(
    timing, stage, "stage_timing.csv"
  )
  fastkpc_full_cuda_phase10_holdout_write_table(
    raw_runs, stage, "raw_runs.csv"
  )
  fastkpc_full_cuda_phase10_holdout_write_table(
    cases, stage, "case_results.csv"
  )
  fastkpc_full_cuda_phase10_holdout_write_table(
    near, stage, "near_alpha_results.csv"
  )
  fastkpc_full_cuda_phase10_holdout_write_table(
    diagnostic_rows, stage, "rank_condition_results.csv"
  )
  fastkpc_full_cuda_phase10_holdout_write_table(
    cache, stage, "cache.csv"
  )
  fastkpc_full_cuda_phase10_holdout_write_table(
    coverage, stage, "coverage.csv"
  )
  fastkpc_full_cuda_phase10_holdout_write_table(
    source_closure$table, stage, "source_closure.csv"
  )
  fastkpc_full_cuda_write_json(
    evidence$attestation, file.path(stage, "release_attestation.json")
  )
  fastkpc_full_cuda_write_json(
    evidence$open_event, file.path(stage, "open_event.json")
  )
  fastkpc_full_cuda_write_json(summary, file.path(stage, "summary.json"))
  writeLines(c(
    "# Full CUDA CI Phase 10 sealed promotion holdout",
    "",
    paste0("- holdout state: ", summary$holdout_state),
    paste0("- cases: ", summary$case_count),
    paste0("- logical tests: ", summary$logical_test_count),
    paste0("- maximum SHD: ", summary$maximum_SHD),
    paste0("- decision flips: ", summary$decision_flip_count),
    paste0("- coverage gate: ", summary$coverage_gate),
    paste0("- promotion claim: ", summary$phase10_promotion_claim)
  ), file.path(stage, "summary.md"), useBytes = TRUE)
  fastkpc_full_cuda_write_json(
    producer_bundle$producer, file.path(stage, "producer_identity.json")
  )
  fastkpc_full_cuda_write_json(
    producer_bundle$backend$value,
    file.path(stage, "backend_configuration.json")
  )
  fastkpc_full_cuda_write_json(
    producer_bundle$build$value, file.path(stage, "build_recipe.json")
  )
  writeLines(c(
    "Rscript fastkpc/tools/run_full_cuda_ci_phase10_holdout.R release",
    "Rscript fastkpc/tools/run_full_cuda_ci_phase10_holdout.R validate",
    paste0("payload_sha256=", evidence$payload_sha256),
    paste0("open_event_identity_sha256=",
           evidence$open_event$open_event_identity_sha256),
    paste0("campaign_producer_identity_sha256=",
           campaign$producer$identity_sha256)
  ), file.path(stage, "commands.txt"), useBytes = TRUE)
  writeLines(c(
    fastkpc_full_cuda_environment_lines(),
    paste0("native_binary_sha256=", campaign$producer$native_binary_sha256),
    paste0("campaign_freeze_identity_sha256=",
           campaign$evidence$freeze$freeze_identity_sha256),
    "release_token_persisted=false",
    "CUDA_VISIBLE_DEVICES=0", "OPENBLAS_NUM_THREADS=1",
    "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
    "BLIS_NUM_THREADS=1", "VECLIB_MAXIMUM_THREADS=1"
  ), file.path(stage, "environment.txt"), useBytes = TRUE)

  excluded <- c(
    "manifest.json", "validator_attestations.json",
    "execution_receipts.json", "environment.txt", "commands.txt"
  )
  semantic_files <- sort(setdiff(
    list.files(stage, all.files = FALSE, no.. = TRUE), excluded
  ), method = "radix")
  payload_hashes <- setNames(lapply(
    file.path(stage, semantic_files), fastkpc_full_cuda_census_file_hash
  ), semantic_files)
  payload_manifest_sha256 <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(payload_hashes)
  )
  envelope <- fastkpc_full_cuda_phase35_identity_envelope(
    producer_bundle$producer, payload_manifest_sha256
  )
  environment_sha256 <- fastkpc_full_cuda_census_file_hash(
    file.path(stage, "environment.txt")
  )
  timestamp <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  attestation <- fastkpc_full_cuda_phase35_validator_attestation(
    producer = producer_bundle$producer,
    validator_source_closure_sha256 = source_closure$sha256,
    validator_semantic_version = "full-cuda-ci-phase10-holdout-validator-v1",
    validator_contracts = contracts,
    validation_timestamp_utc = timestamp,
    environment_sha256 = environment_sha256,
    validation_result = "PASS"
  )
  receipt <- fastkpc_full_cuda_phase35_execution_receipt(
    producer = producer_bundle$producer,
    pid = as.integer(Sys.getpid()),
    session_id = paste0("R-", Sys.getpid(), "-", format(Sys.time(), "%s")),
    cuda_context_id = "cuda-device-0-phase10-holdout",
    artifact_path = file.path(
      normalizePath(parent, winslash = "/", mustWork = TRUE),
      basename(output_dir)
    ),
    artifact_inode = "pre-publication",
    staging_path = normalizePath(stage, winslash = "/", mustWork = TRUE),
    recorded_at_utc = timestamp
  )
  fastkpc_full_cuda_write_json(
    list(attestations = list(attestation)),
    file.path(stage, "validator_attestations.json")
  )
  fastkpc_full_cuda_write_json(
    list(execution_receipts = list(receipt)),
    file.path(stage, "execution_receipts.json")
  )
  manifest <- list(
    schema_version = "full-cuda-ci-phase10-holdout-manifest-v1",
    artifact_kind = "sealed_promotion_holdout",
    claim_scope = "phase10-sealed-holdout-promotion",
    producer_semantic_envelope = envelope,
    payload_manifest_sha256 = payload_manifest_sha256,
    payload_file_sha256 = payload_hashes,
    semantic_file_count = length(payload_hashes),
    validator_attestations_file = "validator_attestations.json",
    volatile_receipt_file = "execution_receipts.json",
    environment_file = "environment.txt",
    environment_file_sha256 = environment_sha256
  )
  fastkpc_full_cuda_write_json(manifest, file.path(stage, "manifest.json"))
  fastkpc_full_cuda_phase10_validate_holdout_artifact(
    stage, verify_current_sources = TRUE
  )
  fastkpc_full_cuda_phase10_holdout_require(
    !dir.exists(output_dir),
    "Phase 10 holdout artifact already exists; replay is forbidden"
  )
  fastkpc_full_cuda_phase10_holdout_require(
    file.rename(stage, output_dir),
    "Phase 10 holdout artifact publication failed"
  )
  active <- FALSE
  fastkpc_full_cuda_phase10_validate_holdout_artifact(
    output_dir, verify_current_sources = TRUE
  )
}

fastkpc_full_cuda_phase10_validate_holdout_result <- function(
    run, oracle, alpha) {
  fastkpc_full_cuda_phase10_holdout_require(
    is.list(run) && identical(
      run$schema_version,
      "full-cuda-ci-phase10-holdout-case-result-v1"
    ) && isTRUE(run$pass) && is.finite(run$elapsed_sec) &&
      run$elapsed_sec > 0 && fastkpc_full_cuda_is_skeleton(run$result),
    "Phase 10 holdout case evidence is malformed"
  )
  rebuilt <- fastkpc_full_cuda_compare_candidate_skeleton(
    oracle, run$result
  )
  tasks <- run$result$tasks
  logical <- oracle$logical_trace
  zero_fields <- fastkpc_full_cuda_phase10_campaign_authority_zero_fields()
  zero_values <- vapply(zero_fields, function(field) {
    value <- run$result$summary[[field]]
    if (is.null(value)) NA_real_ else as.numeric(value)
  }, numeric(1L))
  flips <- (tasks$p_used >= alpha) != (logical$p_value >= alpha)
  comparison_gate <- isTRUE(rebuilt$summary$pass) &&
    identical(rebuilt$summary, run$comparison$summary) &&
    identical(rebuilt$graph_agreement, run$comparison$graph_agreement) &&
    identical(rebuilt$sepset_agreement, run$comparison$sepset_agreement) &&
    identical(rebuilt$n_edgetests, run$comparison$n_edgetests) &&
    identical(rebuilt$candidate_deletions,
              run$comparison$candidate_deletions) &&
    identical(rebuilt$candidate_logical,
              run$comparison$candidate_logical)
  authority_gate <- all(!is.na(zero_values) & zero_values == 0) &&
    identical(run$result$summary$run_status, "ok") &&
    identical(run$result$summary$entrypoint,
              "compatible-cuda-full-skeleton-native-v1") &&
    identical(run$result$summary$compatible_cuda_route, "compatible.cuda") &&
    isTRUE(run$result$summary$compatible_cuda_strict) &&
    isTRUE(run$result$summary$authority_gate_pass) &&
    as.integer(run$result$summary$native_call_count) == 1L &&
    as.integer(run$result$summary$logical_tests_consumed) == nrow(logical) &&
    as.integer(run$result$summary$speculative_tests_ignored) == 0L &&
    identical(run$result$summary$scheduler,
              "cache-aware-frontier-4x-v1") &&
    all(is.finite(tasks$p_used)) && !any(flips) &&
    as.integer(run$decision_flip_count) == 0L &&
    identical(as.numeric(run$authority_zero_values),
              as.numeric(zero_values)) &&
    all(fastkpc_full_cuda_phase10_campaign_active_resources(
      run$resource_before
    ) == 0) && all(fastkpc_full_cuda_phase10_campaign_active_resources(
      run$resource_after
    ) == 0)
  fastkpc_full_cuda_phase10_holdout_require(
    comparison_gate && authority_gate,
    paste0("Phase 10 holdout result validation failed: ", run$case_id)
  )
  invisible(run)
}

fastkpc_full_cuda_phase10_release_holdout <- function(
    release_dir = Sys.getenv(
      "FASTKPC_PROMOTION_HOLDOUT_RELEASE_DIR", unset = ""
    ),
    token = Sys.getenv(
      "FASTKPC_PROMOTION_HOLDOUT_RELEASE_TOKEN", unset = ""
    ),
    campaign_dir = fastkpc_full_cuda_phase10_campaign_artifact_dir(),
    staging_dir = fastkpc_full_cuda_phase10_holdout_staging_dir(),
    output_dir = fastkpc_full_cuda_phase10_holdout_artifact_dir()) {
  fastkpc_full_cuda_phase10_holdout_require(
    nzchar(release_dir) && dir.exists(release_dir) &&
      nzchar(token) && !dir.exists(output_dir) &&
      !dir.exists(staging_dir),
    paste(
      "Phase 10 holdout release requires a new custodian envelope, token,",
      "and unused staging/output paths"
    )
  )
  campaign <- fastkpc_full_cuda_phase10_validate_campaign_artifact(
    campaign_dir, verify_current_sources = TRUE
  )
  fastkpc_full_cuda_phase10_holdout_require(
    isTRUE(campaign$summary$pass) &&
      isTRUE(campaign$summary$phase10_canonical_campaign_claim) &&
      !isTRUE(campaign$summary$phase10_promotion_claim) &&
      identical(campaign$evidence$freeze$holdout_state,
                "SEALED_NOT_RELEASED") &&
      !isTRUE(campaign$evidence$freeze$holdout_opened),
    "Phase 10 holdout cannot open before the canonical campaign is frozen"
  )
  machine_before <- fastkpc_full_cuda_phase10_campaign_machine_snapshot(
    require_idle_gpu = TRUE
  )
  attestation_path <- file.path(release_dir, "custodian_attestation.json")
  fastkpc_full_cuda_phase10_holdout_require(
    file.exists(attestation_path) && !dir.exists(attestation_path),
    "Phase 10 holdout custodian attestation is missing"
  )
  attestation <- jsonlite::read_json(
    attestation_path, simplifyVector = FALSE
  )
  fastkpc_full_cuda_phase10_holdout_validate_attestation(
    attestation, token = token
  )
  payload_path <- file.path(release_dir, attestation$payload_file)
  fastkpc_full_cuda_phase10_holdout_require(
    file.exists(payload_path) && !dir.exists(payload_path),
    "Phase 10 holdout custodian payload is missing"
  )
  dir.create(staging_dir, recursive = TRUE)
  open_event <- fastkpc_full_cuda_phase10_holdout_open_event(
    campaign, attestation,
    fastkpc_full_cuda_phase35_sha256_utf8(token),
    format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  )
  open_event_path <- file.path(staging_dir, "open_event.json")
  fastkpc_full_cuda_phase10_holdout_write_json_atomic(
    open_event, open_event_path
  )
  persisted_event <- jsonlite::read_json(
    open_event_path, simplifyVector = FALSE
  )
  fastkpc_full_cuda_phase10_holdout_validate_open_event(
    persisted_event, campaign, attestation
  )

  payload_sha256 <- fastkpc_full_cuda_census_file_hash(payload_path)
  fastkpc_full_cuda_phase10_holdout_require(
    identical(payload_sha256, attestation$payload_sha256),
    "Phase 10 holdout payload hash does not match custodian attestation"
  )
  Sys.unsetenv("FASTKPC_PROMOTION_HOLDOUT_RELEASE_TOKEN")
  token <- NULL
  payload <- readRDS(payload_path)
  validation <- fastkpc_full_cuda_phase10_validate_holdout_payload(payload)
  results <- vector("list", length(payload$cases))
  payload_metadata <- vector("list", length(payload$cases))
  oracles <- vector("list", length(payload$cases))
  for (index in seq_along(payload$cases)) {
    case <- payload$cases[[index]]
    case_started <- list(
      schema_version = "full-cuda-ci-phase10-holdout-case-start-v1",
      case_id = case$case_id,
      sequence = as.integer(index),
      recorded_at_utc = format(
        Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"
      ),
      state = "CASE_STARTED_NO_REPLAY"
    )
    fastkpc_full_cuda_phase10_holdout_write_json_atomic(
      case_started,
      file.path(staging_dir, sprintf("case-%03d-start.json", index))
    )
    results[[index]] <- fastkpc_full_cuda_phase10_holdout_capture_case(case)
    saveRDS(
      results[[index]],
      file.path(staging_dir, sprintf("case-%03d-result.rds", index)),
      compress = "xz"
    )
    payload_metadata[[index]] <- list(
      case_id = case$case_id, alpha = case$alpha,
      n = nrow(case$data), p = ncol(case$data),
      reference_p_value = as.numeric(case$oracle$logical_trace$p_value)
    )
    oracles[[index]] <- case$oracle
  }
  names(results) <- validation$case_ids
  names(payload_metadata) <- validation$case_ids
  names(oracles) <- validation$case_ids
  machine_after <- fastkpc_full_cuda_phase10_campaign_machine_snapshot(
    require_idle_gpu = FALSE
  )
  fastkpc_full_cuda_phase10_holdout_require(
    identical(
      fastkpc_full_cuda_phase10_campaign_machine_identity(machine_before),
      campaign$evidence$freeze$machine_identity
    ) && identical(
      fastkpc_full_cuda_phase10_campaign_machine_identity(machine_after),
      campaign$evidence$freeze$machine_identity
    ),
    "Phase 10 holdout reference machine drifted during release"
  )
  evidence <- list(
    schema_version = "full-cuda-ci-phase10-holdout-evidence-v1",
    attestation = attestation,
    open_event = persisted_event,
    payload_sha256 = payload_sha256,
    corpus_id = payload$corpus_id,
    payload_metadata = payload_metadata,
    oracles = oracles,
    results = results,
    diagnostics = validation$diagnostics,
    coverage = validation$coverage,
    campaign_producer_identity_sha256 = campaign$producer$identity_sha256,
    campaign_freeze_identity_sha256 =
      campaign$evidence$freeze$freeze_identity_sha256,
    machine_before = machine_before,
    machine_after = machine_after,
    pass = TRUE
  )
  artifact <- fastkpc_full_cuda_phase10_publish_holdout(
    evidence, campaign, output_dir = output_dir
  )
  unlink(staging_dir, recursive = TRUE, force = TRUE)
  artifact
}

fastkpc_full_cuda_phase10_validate_holdout_artifact <- function(
    artifact_dir = fastkpc_full_cuda_phase10_holdout_artifact_dir(),
    verify_current_sources = FALSE) {
  artifact_dir <- normalizePath(
    artifact_dir, winslash = "/", mustWork = TRUE
  )
  required <- sort(
    fastkpc_full_cuda_phase10_holdout_required_files(), method = "radix"
  )
  actual <- sort(
    list.files(artifact_dir, all.files = FALSE, no.. = TRUE),
    method = "radix"
  )
  fastkpc_full_cuda_phase10_holdout_require(
    identical(actual, required),
    "Phase 10 holdout artifact standard file set is incomplete"
  )
  manifest <- jsonlite::read_json(
    file.path(artifact_dir, "manifest.json"), simplifyVector = FALSE
  )
  manifest_fields <- c(
    "schema_version", "artifact_kind", "claim_scope",
    "producer_semantic_envelope", "payload_manifest_sha256",
    "payload_file_sha256", "semantic_file_count",
    "validator_attestations_file", "volatile_receipt_file",
    "environment_file", "environment_file_sha256"
  )
  fastkpc_full_cuda_phase10_holdout_require(
    is.list(manifest) && identical(names(manifest), manifest_fields) &&
      identical(
        manifest$schema_version,
        "full-cuda-ci-phase10-holdout-manifest-v1"
      ) && identical(manifest$artifact_kind,
                    "sealed_promotion_holdout") &&
      identical(manifest$claim_scope,
                "phase10-sealed-holdout-promotion") &&
      identical(manifest$validator_attestations_file,
                "validator_attestations.json") &&
      identical(manifest$volatile_receipt_file,
                "execution_receipts.json") &&
      identical(manifest$environment_file, "environment.txt"),
    "Phase 10 holdout artifact manifest schema mismatch"
  )
  excluded <- c(
    "manifest.json", "validator_attestations.json",
    "execution_receipts.json", "environment.txt", "commands.txt"
  )
  semantic <- sort(setdiff(required, excluded), method = "radix")
  payload_hashes <- manifest$payload_file_sha256
  fastkpc_full_cuda_phase10_holdout_require(
    is.list(payload_hashes) &&
      identical(sort(names(payload_hashes), method = "radix"), semantic),
    "Phase 10 holdout payload manifest is malformed"
  )
  actual_hashes <- setNames(lapply(names(payload_hashes), function(name) {
    fastkpc_full_cuda_census_file_hash(file.path(artifact_dir, name))
  }), names(payload_hashes))
  payload_manifest_sha256 <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(actual_hashes)
  )
  fastkpc_full_cuda_phase10_holdout_require(
    all(vapply(names(payload_hashes), function(name) {
      identical(payload_hashes[[name]], actual_hashes[[name]])
    }, logical(1L))) &&
      identical(payload_manifest_sha256,
                manifest$payload_manifest_sha256) &&
      as.integer(manifest$semantic_file_count) == length(actual_hashes),
    "Phase 10 holdout artifact payload identity mismatch"
  )
  fastkpc_full_cuda_phase35_validate_identity_envelope(
    manifest$producer_semantic_envelope
  )
  producer <- jsonlite::read_json(
    file.path(artifact_dir, "producer_identity.json"), simplifyVector = FALSE
  )
  .fastkpc_full_cuda_phase35_validate_producer(producer)
  fastkpc_full_cuda_phase10_holdout_require(
    identical(
      fastkpc_full_cuda_phase35_canonical_json(producer),
      fastkpc_full_cuda_phase35_canonical_json(
        manifest$producer_semantic_envelope$producer
      )
    ) && identical(
      manifest$producer_semantic_envelope$payload_manifest_sha256,
      payload_manifest_sha256
    ),
    "Phase 10 holdout producer envelope mismatch"
  )
  source_closure <- utils::read.csv(
    file.path(artifact_dir, "source_closure.csv"),
    stringsAsFactors = FALSE, check.names = FALSE
  )
  closure_hashes <- setNames(
    as.list(as.character(source_closure$sha256)),
    as.character(source_closure$path)
  )
  closure_sha256 <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(closure_hashes)
  )
  fastkpc_full_cuda_phase10_holdout_require(
    identical(names(source_closure), c("path", "sha256")) &&
      !anyDuplicated(source_closure$path) &&
      identical(closure_sha256, producer$producer_source_closure_sha256),
    "Phase 10 holdout source closure identity mismatch"
  )
  backend <- jsonlite::read_json(
    file.path(artifact_dir, "backend_configuration.json"),
    simplifyVector = FALSE
  )
  build <- jsonlite::read_json(
    file.path(artifact_dir, "build_recipe.json"), simplifyVector = FALSE
  )
  fastkpc_full_cuda_phase10_holdout_require(
    identical(
      fastkpc_full_cuda_phase35_sha256_utf8(
        fastkpc_full_cuda_phase35_canonical_json(backend)
      ), producer$backend_configuration_sha256
    ) && identical(
      fastkpc_full_cuda_phase35_sha256_utf8(
        fastkpc_full_cuda_phase35_canonical_json(build)
      ), producer$build_recipe_sha256
    ),
    "Phase 10 holdout backend or build identity mismatch"
  )
  environment_sha256 <- fastkpc_full_cuda_census_file_hash(
    file.path(artifact_dir, "environment.txt")
  )
  fastkpc_full_cuda_phase10_holdout_require(
    identical(environment_sha256, manifest$environment_file_sha256),
    "Phase 10 holdout environment evidence hash mismatch"
  )
  attestations <- jsonlite::read_json(
    file.path(artifact_dir, "validator_attestations.json"),
    simplifyVector = FALSE
  )$attestations
  receipts <- jsonlite::read_json(
    file.path(artifact_dir, "execution_receipts.json"),
    simplifyVector = FALSE
  )$execution_receipts
  fastkpc_full_cuda_phase10_holdout_require(
    is.list(attestations) && length(attestations) > 0L &&
      is.list(receipts) && length(receipts) > 0L,
    "Phase 10 holdout attestation or receipt is missing"
  )
  for (attestation in attestations) {
    .fastkpc_full_cuda_phase35_validate_attestation(attestation)
    fastkpc_full_cuda_phase10_holdout_require(
      identical(attestation$attested_producer_sha256,
                producer$identity_sha256) &&
        identical(attestation$validator_source_closure_sha256,
                  closure_sha256) &&
        identical(attestation$environment_sha256, environment_sha256) &&
        identical(attestation$validation_result, "PASS"),
      "Phase 10 holdout validator attestation mismatch"
    )
  }
  for (receipt in receipts) {
    .fastkpc_full_cuda_phase35_validate_receipt(receipt)
    fastkpc_full_cuda_phase10_holdout_require(
      identical(receipt$producer_sha256, producer$identity_sha256),
      "Phase 10 holdout execution receipt mismatch"
    )
  }
  summary <- jsonlite::read_json(
    file.path(artifact_dir, "summary.json"), simplifyVector = TRUE
  )
  fastkpc_full_cuda_phase10_holdout_validate_summary(summary)
  evidence_path <- file.path(artifact_dir, "source_evidence.rds")
  evidence <- readRDS(evidence_path)
  fastkpc_full_cuda_phase10_holdout_require(
    identical(summary$source_evidence_sha256,
              fastkpc_full_cuda_census_file_hash(evidence_path)) &&
      identical(summary$producer_identity_sha256, producer$identity_sha256) &&
      identical(summary$source_closure_sha256,
                producer$producer_source_closure_sha256) &&
      identical(summary$native_binary_sha256,
                producer$native_binary_sha256) &&
      is.list(evidence) && identical(
        evidence$schema_version,
        "full-cuda-ci-phase10-holdout-evidence-v1"
      ) && isTRUE(evidence$pass),
    "Phase 10 holdout summary or evidence linkage failed"
  )
  release_attestation <- jsonlite::read_json(
    file.path(artifact_dir, "release_attestation.json"),
    simplifyVector = FALSE
  )
  open_event <- jsonlite::read_json(
    file.path(artifact_dir, "open_event.json"), simplifyVector = FALSE
  )
  fastkpc_full_cuda_phase10_holdout_validate_attestation(
    release_attestation
  )
  fastkpc_full_cuda_phase10_holdout_validate_open_event(
    open_event, attestation = release_attestation
  )
  fastkpc_full_cuda_phase10_holdout_require(
    identical(release_attestation, evidence$attestation) &&
      identical(open_event, evidence$open_event) &&
      identical(evidence$payload_sha256,
                release_attestation$payload_sha256) &&
      length(evidence$results) == length(evidence$oracles) &&
      length(evidence$results) == length(evidence$payload_metadata) &&
      length(evidence$results) >= 1L &&
      all(unlist(evidence$coverage, use.names = FALSE)),
    "Phase 10 holdout release evidence is inconsistent"
  )
  for (index in seq_along(evidence$results)) {
    fastkpc_full_cuda_phase10_validate_holdout_result(
      evidence$results[[index]], evidence$oracles[[index]],
      evidence$payload_metadata[[index]]$alpha
    )
  }
  campaign <- fastkpc_full_cuda_phase10_validate_campaign_artifact(
    fastkpc_full_cuda_phase10_campaign_artifact_dir(),
    verify_current_sources = isTRUE(verify_current_sources)
  )
  fastkpc_full_cuda_phase10_holdout_require(
    identical(campaign$producer$identity_sha256,
              evidence$campaign_producer_identity_sha256) &&
      identical(campaign$evidence$freeze$freeze_identity_sha256,
                evidence$campaign_freeze_identity_sha256) &&
      identical(campaign$producer$identity_sha256,
                summary$canonical_campaign_producer_identity_sha256),
    "Phase 10 holdout canonical campaign linkage failed"
  )
  contracts <- fastkpc_full_cuda_phase35_load_contract_set()
  producer_bundle <- fastkpc_full_cuda_phase10_holdout_producer(
    evidence, list(sha256 = closure_sha256), campaign, contracts
  )
  expected_summary <- fastkpc_full_cuda_phase10_holdout_summary(
    evidence, producer_bundle,
    fastkpc_full_cuda_census_file_hash(evidence_path),
    campaign, contracts
  )
  fastkpc_full_cuda_phase10_holdout_require(
    identical(
      fastkpc_full_cuda_phase35_canonical_json(
        producer_bundle$producer
      ),
      fastkpc_full_cuda_phase35_canonical_json(producer)
    ) && fastkpc_full_cuda_phase10_campaign_json_equivalent(
      expected_summary, summary
    ),
    "Phase 10 holdout producer or summary could not be rebuilt"
  )
  read_table <- function(name) utils::read.csv(
    file.path(artifact_dir, name), stringsAsFactors = FALSE,
    check.names = FALSE
  )
  graph <- read_table("graph_agreement.csv")
  sepsets <- read_table("sepset_agreement.csv")
  n_edgetests <- read_table("n_edgetests.csv")
  fallbacks <- read_table("fallbacks.csv")
  raw <- read_table("raw_runs.csv")
  cases <- read_table("case_results.csv")
  near <- read_table("near_alpha_results.csv")
  cache <- read_table("cache.csv")
  coverage <- read_table("coverage.csv")
  payload_gate <- nrow(graph) == length(evidence$results) &&
    all(graph$SHD == 0L) && all(as.logical(graph$adjacency_identical)) &&
    nrow(sepsets) > 0L && all(as.logical(sepsets$identical)) &&
    nrow(n_edgetests) > 0L && all(as.logical(n_edgetests$identical)) &&
    nrow(fallbacks) == 3L && sum(fallbacks$count) == 0L &&
    nrow(raw) == length(evidence$results) && all(as.logical(raw$pass)) &&
    all(raw$SHD == 0L) && all(raw$decision_flip_count == 0L) &&
    nrow(cases) == summary$logical_test_count &&
    !any(as.logical(cases$decision_flip)) &&
    nrow(near) > 0L && !any(as.logical(near$decision_flip)) &&
    nrow(cache) == 3L * length(evidence$results) &&
    all(as.logical(cache$bounded)) &&
    all(cache$requests == cache$hits + cache$misses) &&
    nrow(coverage) == length(evidence$coverage) &&
    all(as.logical(coverage$covered))
  fastkpc_full_cuda_phase10_holdout_require(
    payload_gate, "Phase 10 holdout standard payload is malformed"
  )
  if (isTRUE(verify_current_sources)) {
    current <- fastkpc_full_cuda_phase10_holdout_source_closure()
    fastkpc_full_cuda_phase10_holdout_require(
      identical(current$sha256, closure_sha256) &&
        identical(current$hashes, closure_hashes) &&
        identical(fastkpc_full_cuda_phase7_native_identity()$sha256,
                  producer$native_binary_sha256),
      "Phase 10 holdout current source or native binary drifted"
    )
  }
  list(
    manifest = manifest, summary = summary, producer = producer,
    evidence = evidence, source_closure = source_closure
  )
}
