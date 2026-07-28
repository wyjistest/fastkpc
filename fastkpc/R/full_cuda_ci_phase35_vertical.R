.fastkpc_full_cuda_phase35_vertical_sha256 <- function(value) {
  if (!exists("fastkpc_full_cuda_phase35_sha256_utf8", mode = "function",
              inherits = TRUE)) {
    stop(
      "source full_cuda_ci_phase35_contracts.R before building a vertical request",
      call. = FALSE
    )
  }
  fastkpc_full_cuda_phase35_sha256_utf8(value)
}

.fastkpc_full_cuda_phase35_vertical_is_sha256 <- function(value) {
  is.character(value) && length(value) == 1L && !is.na(value) &&
    grepl("^[0-9a-f]{64}$", value)
}

fastkpc_full_cuda_phase35_vertical_request <- function(
    expected_prepared_s_key_sha256,
    target_keys,
    logical_sequence_id,
    left_target_ordinal = 1L,
    right_target_ordinal = 2L,
    alpha = 0.1,
    exercise_eviction = TRUE) {
  if (!.fastkpc_full_cuda_phase35_vertical_is_sha256(
        expected_prepared_s_key_sha256)) {
    stop("expected PreparedSKey must be a lowercase SHA-256", call. = FALSE)
  }
  if (!is.character(target_keys) || length(target_keys) < 2L ||
      anyNA(target_keys) ||
      any(!vapply(target_keys,
                  .fastkpc_full_cuda_phase35_vertical_is_sha256,
                  logical(1L))) || anyDuplicated(target_keys)) {
    stop("target_keys must be distinct lowercase SHA-256 strings",
         call. = FALSE)
  }
  if (!is.double(logical_sequence_id) || length(logical_sequence_id) != 1L ||
      is.na(logical_sequence_id) || !is.finite(logical_sequence_id) ||
      logical_sequence_id < 1 || logical_sequence_id > 2^53 - 1 ||
      logical_sequence_id != floor(logical_sequence_id) ||
      !is.null(attributes(logical_sequence_id))) {
    stop("logical_sequence_id must be a bare positive safe-53-bit double integer",
         call. = FALSE)
  }
  ordinals <- c(left_target_ordinal, right_target_ordinal)
  if (!is.integer(ordinals) || length(ordinals) != 2L || anyNA(ordinals) ||
      any(ordinals < 1L) || any(ordinals > length(target_keys)) ||
      ordinals[[1L]] == ordinals[[2L]]) {
    stop("vertical target ordinals are invalid", call. = FALSE)
  }
  if (!is.double(alpha) || length(alpha) != 1L || is.na(alpha) ||
      !is.finite(alpha) || !identical(alpha, 0.1) ||
      !is.null(attributes(alpha))) {
    stop("vertical prototype requires bare canonical alpha 0.1",
         call. = FALSE)
  }
  if (!is.logical(exercise_eviction) || length(exercise_eviction) != 1L ||
      is.na(exercise_eviction) || !isTRUE(exercise_eviction) ||
      !is.null(attributes(exercise_eviction))) {
    stop("vertical prototype requires deterministic eviction replay",
         call. = FALSE)
  }

  identity_payload <- paste(
    "schema_version=full-cuda-ci-phase35-vertical-request-v1",
    paste0("expected_prepared_s_key_sha256=",
           expected_prepared_s_key_sha256),
    paste0("logical_sequence_id=", sprintf("%.0f", logical_sequence_id)),
    paste0("left_target_key=", target_keys[[left_target_ordinal]]),
    paste0("right_target_key=", target_keys[[right_target_ordinal]]),
    "alpha=0.1",
    "exercise_eviction=true",
    sep = "\n"
  )
  list(
    schema_version = "full-cuda-ci-phase35-vertical-request-v1",
    expected_prepared_s_key_sha256 = expected_prepared_s_key_sha256,
    request_identity_sha256 =
      .fastkpc_full_cuda_phase35_vertical_sha256(identity_payload),
    logical_sequence_id = logical_sequence_id,
    left_target_ordinal = as.integer(left_target_ordinal),
    right_target_ordinal = as.integer(right_target_ordinal),
    alpha = alpha,
    exercise_eviction = exercise_eviction
  )
}

fastkpc_full_cuda_phase35_vertical_resource_snapshot <- function() {
  load_fastkpc_cuda_native()
  .Call(
    "C_full_cuda_ci_phase35_vertical_resource_snapshot",
    PACKAGE = "fastkpc_cuda"
  )
}

fastkpc_full_cuda_phase35_vertical_ci <- function(
    prepared_s, Y, SP, planned_route, target_keys, request) {
  load_fastkpc_cuda_native()
  Y <- matrix(as.double(Y), nrow = nrow(Y), ncol = ncol(Y))
  SP <- matrix(as.double(SP), nrow = nrow(SP), ncol = ncol(SP))
  .Call(
    "C_full_cuda_ci_phase35_vertical",
    prepared_s,
    Y,
    SP,
    as.character(planned_route),
    as.character(target_keys),
    request,
    PACKAGE = "fastkpc_cuda"
  )
}

fastkpc_full_cuda_phase35_canonical_vertical_fixture <- function() {
  required <- c(
    "fastkpc_full_cuda_prepared_s_shard_authentication",
    "fastkpc_full_cuda_fixed_sp_read_json",
    "fastkpc_full_cuda_census_file_hash",
    "fastkpc_full_cuda_fixed_sp_native_dto"
  )
  missing <- required[!vapply(required, exists, logical(1L),
                             mode = "function", inherits = TRUE)]
  if (length(missing) > 0L) {
    stop("canonical vertical fixture dependency is missing: ", missing[[1L]],
         call. = FALSE)
  }

  shard_path <- file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1",
    "shards", "shard_0.rds"
  )
  summary_path <- sub("[.]rds$", ".summary.json", shard_path)
  shard <- readRDS(shard_path)
  shard_summary <- fastkpc_full_cuda_fixed_sp_read_json(summary_path)
  authentication <- fastkpc_full_cuda_prepared_s_shard_authentication(shard)
  if (!identical(authentication$payload_hash, shard_summary$payload_hash) ||
      !identical(authentication$manifest_hash,
                 shard_summary$manifest_hash) ||
      !identical(fastkpc_full_cuda_census_file_hash(shard_path),
                 shard_summary$rds_file_sha256)) {
    stop("canonical vertical Prepared-S shard authentication failed",
         call. = FALSE)
  }

  prepared_key <-
    "000bf94226b34186828cfa30c400753eb19ca2ff99409573df21ac06da2a72be"
  logical_sequence_id <- 139040L
  logical_tests <- readRDS(file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "workload_census_351x48_v1",
    "logical_ci_tests.rds"
  ))
  logical_row <- logical_tests[
    logical_tests$logical_sequence_id == logical_sequence_id,
    , drop = FALSE
  ]
  setup <- shard$prepared_s_setups[[prepared_key]]
  states <- shard$target_states[
    shard$target_states$prepared_s_key_sha256 == prepared_key &
      shard$target_states$target %in% c(logical_row$x, logical_row$y),
    , drop = FALSE
  ]
  states <- states[match(c(logical_row$x, logical_row$y), states$target),
                   , drop = FALSE]
  if (nrow(logical_row) != 1L || logical_row$x != 17L ||
      logical_row$y != 30L || !identical(logical_row$S_key, "11|35") ||
      nrow(states) != 2L ||
      !identical(states$residual_key_sha256,
                 c(logical_row$residual_key_x,
                   logical_row$residual_key_y)) ||
      !identical(setup$sorted_S, c(11L, 35L))) {
    stop("canonical vertical logical request authentication failed",
         call. = FALSE)
  }

  phase3_setups <- readRDS(file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "fixed_sp_cuda_oracle_sp_v1",
    "setup_results.rds"
  ))
  phase3_setup <- phase3_setups[
    phase3_setups$prepared_s_key_sha256 == prepared_key,
    , drop = FALSE
  ]
  if (nrow(phase3_setup) != 1L || phase3_setup$target_count != 41L ||
      phase3_setup$planned_cholesky_target_count != 41L ||
      phase3_setup$executed_cholesky_target_count != 41L ||
      phase3_setup$stable_reroute_count != 0L) {
    stop("canonical vertical Phase 3 route authentication failed",
         call. = FALSE)
  }

  data_path <- file.path(
    "fastkpc", "artifacts", "kpc_tprs_real_zhu",
    "cancer_RD-causalDiscoveryInput.rds"
  )
  data <- readRDS(data_path)
  dto <- fastkpc_full_cuda_fixed_sp_native_dto(setup)
  Y <- data[, states$target, drop = FALSE]
  SP <- do.call(cbind, lapply(states$selected_sp, as.numeric))
  request <- fastkpc_full_cuda_phase35_vertical_request(
    expected_prepared_s_key_sha256 = prepared_key,
    target_keys = states$residual_key_sha256,
    logical_sequence_id = as.double(logical_sequence_id),
    left_target_ordinal = 1L,
    right_target_ordinal = 2L,
    alpha = 0.1,
    exercise_eviction = TRUE
  )
  list(
    shard_path = shard_path,
    shard_summary_path = summary_path,
    shard_authentication = authentication,
    prepared_key = prepared_key,
    logical_row = logical_row,
    setup = setup,
    states = states,
    dto = dto,
    data_path = data_path,
    Y = Y,
    SP = SP,
    planned_route = rep("CHOLESKY_BATCHED", 2L),
    request = request
  )
}

fastkpc_full_cuda_phase35_run_canonical_vertical <- function(
    rebuild = FALSE, include_cpu_oracle = TRUE) {
  load_fastkpc_cuda_native(rebuild = rebuild)
  fixture <- fastkpc_full_cuda_phase35_canonical_vertical_fixture()
  runtime <- fixed_sp_cuda_runtime_create(0L)
  handle <- NULL
  on.exit({
    if (!is.null(handle)) {
      try(fixed_sp_cuda_prepared_free(handle), silent = TRUE)
    }
    if (!is.null(runtime)) {
      try(fixed_sp_cuda_runtime_free(runtime), silent = TRUE)
    }
  }, add = TRUE)
  dto <- fixture$dto
  fixed_sp_cuda_runtime_reserve(
    runtime, dto$n, dto$null_dim, 2L, dto$penalty_count,
    dto$n + dto$null_dim
  )
  handle <- fixed_sp_cuda_prepared_create(runtime, dto)
  runtime_info <- fixed_sp_cuda_runtime_info(runtime)
  resources_before <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
  result <- fastkpc_full_cuda_phase35_vertical_ci(
    handle, fixture$Y, fixture$SP, fixture$planned_route,
    fixture$states$residual_key_sha256, fixture$request
  )
  resources_after <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
  prepared_after <- fixed_sp_cuda_prepared_info(handle)

  oracle <- NULL
  if (isTRUE(include_cpu_oracle)) {
    if (!exists("dcov_gamma_exact", mode = "function", inherits = TRUE)) {
      stop("source dcov_exact.R before requesting the CPU exact oracle",
           call. = FALSE)
    }
    residuals <- lapply(seq_len(2L), function(index) {
      fastkpc_mgcv_magic_fixed_sp_from_prepared(
        prepared_setup = fixture$setup,
        target_state = list(
          row = fixture$states[index, , drop = FALSE],
          y = as.numeric(fixture$Y[, index])
        )
      )$residuals
    })
    oracle <- dcov_gamma_exact(residuals[[1L]], residuals[[2L]])
  }

  fixed_sp_cuda_prepared_free(handle)
  handle <- NULL
  fixed_sp_cuda_runtime_free(runtime)
  runtime <- NULL
  list(
    fixture = fixture,
    result = result,
    exact_cpu_oracle = oracle,
    runtime_info = runtime_info,
    resources_before = resources_before,
    resources_after = resources_after,
    prepared_after = prepared_after
  )
}

fastkpc_full_cuda_phase35_validate_vertical_artifact <- function(output_dir) {
  required_functions <- c(
    "fastkpc_full_cuda_phase35_validate_identity_envelope",
    "fastkpc_full_cuda_phase35_canonical_json",
    "fastkpc_full_cuda_phase35_sha256_utf8",
    "fastkpc_full_cuda_census_file_hash"
  )
  missing <- required_functions[!vapply(
    required_functions, exists, logical(1L), mode = "function",
    inherits = TRUE
  )]
  if (length(missing) > 0L) {
    stop("vertical artifact validator dependency is missing: ", missing[[1L]],
         call. = FALSE)
  }
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("jsonlite is required to validate the vertical artifact",
         call. = FALSE)
  }
  expected_files <- c(
    "cache.csv", "case_results.csv", "commands.txt", "environment.txt",
    "execution_receipts.json", "fallbacks.csv", "first_divergence.json",
    "graph_agreement.csv", "manifest.json", "n_edgetests.csv",
    "producer_identity.json", "raw_runs.csv", "sepset_agreement.csv",
    "source_closure.csv", "stage_timing.csv", "summary.json",
    "summary.md", "validator_attestations.json"
  )
  if (!dir.exists(output_dir) ||
      !identical(sort(list.files(output_dir), method = "radix"),
                 sort(expected_files, method = "radix"))) {
    stop("vertical artifact file set is incomplete", call. = FALSE)
  }
  read_json <- function(name) {
    jsonlite::read_json(file.path(output_dir, name), simplifyVector = FALSE)
  }
  manifest <- read_json("manifest.json")
  producer <- read_json("producer_identity.json")
  attestations <- read_json("validator_attestations.json")$attestations
  receipts <- read_json("execution_receipts.json")$execution_receipts
  summary <- read_json("summary.json")
  if (!is.list(manifest) ||
      !identical(names(manifest), c(
        "schema_version", "claim_scope", "producer_semantic_envelope",
        "payload_manifest_sha256", "payload_file_sha256",
        "semantic_file_count", "validator_attestations_file",
        "volatile_receipt_file"
      )) ||
      !identical(manifest$schema_version,
                 "full-cuda-ci-phase35d-vertical-manifest-v1") ||
      !identical(manifest$claim_scope, "phase3.5D-structural-only")) {
    stop("vertical artifact manifest schema mismatch", call. = FALSE)
  }
  .fastkpc_full_cuda_phase35_validate_producer(producer)
  if (!fastkpc_full_cuda_phase35_validate_identity_envelope(
        manifest$producer_semantic_envelope) ||
      !identical(manifest$producer_semantic_envelope$producer, producer) ||
      !identical(manifest$producer_semantic_envelope$payload_manifest_sha256,
                 manifest$payload_manifest_sha256) ||
      length(manifest$producer_semantic_envelope$attestations) != 0L ||
      length(manifest$producer_semantic_envelope$execution_receipts) != 0L) {
    stop("vertical producer semantic envelope mismatch", call. = FALSE)
  }
  if (!is.list(attestations) || length(attestations) < 1L ||
      !is.list(receipts) || length(receipts) < 1L) {
    stop("vertical attestation or receipt namespace is empty", call. = FALSE)
  }
  for (attestation in attestations) {
    .fastkpc_full_cuda_phase35_validate_attestation(attestation)
    if (!identical(attestation$attested_producer_sha256,
                   producer$identity_sha256) ||
        !identical(attestation$validation_result, "PASS")) {
      stop("vertical validator attestation linkage mismatch",
           call. = FALSE)
    }
  }
  for (receipt in receipts) {
    .fastkpc_full_cuda_phase35_validate_receipt(receipt)
    if (!identical(receipt$producer_sha256, producer$identity_sha256)) {
      stop("vertical execution receipt linkage mismatch", call. = FALSE)
    }
  }

  payload_hashes <- manifest$payload_file_sha256
  if (!is.list(payload_hashes) || is.null(names(payload_hashes)) ||
      anyNA(names(payload_hashes)) || anyDuplicated(names(payload_hashes)) ||
      length(payload_hashes) != as.integer(manifest$semantic_file_count)) {
    stop("vertical payload file hash manifest is malformed", call. = FALSE)
  }
  for (name in names(payload_hashes)) {
    path <- file.path(output_dir, name)
    if (!file.exists(path) ||
        !identical(payload_hashes[[name]],
                   fastkpc_full_cuda_census_file_hash(path))) {
      stop("vertical payload file hash mismatch: ", name, call. = FALSE)
    }
  }

  source_closure <- read.csv(
    file.path(output_dir, "source_closure.csv"),
    stringsAsFactors = FALSE, check.names = FALSE
  )
  if (!identical(names(source_closure), c("path", "sha256")) ||
      nrow(source_closure) < 1L || anyNA(source_closure) ||
      anyDuplicated(source_closure$path) ||
      any(!grepl("^[0-9a-f]{64}$", source_closure$sha256))) {
    stop("vertical source closure schema mismatch", call. = FALSE)
  }
  closure_hashes <- setNames(
    as.list(source_closure$sha256), source_closure$path
  )
  closure_sha256 <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(closure_hashes)
  )
  if (!identical(closure_sha256,
                 producer$producer_source_closure_sha256) ||
      !identical(summary$source_closure_sha256, closure_sha256) ||
      !identical(summary$producer_identity_sha256,
                 producer$identity_sha256) ||
      !identical(summary$native_binary_sha256,
                 producer$native_binary_sha256)) {
    stop("vertical producer closure identity mismatch", call. = FALSE)
  }

  case_results <- read.csv(
    file.path(output_dir, "case_results.csv"),
    stringsAsFactors = FALSE, check.names = FALSE
  )
  cache <- read.csv(file.path(output_dir, "cache.csv"),
                    stringsAsFactors = FALSE, check.names = FALSE)
  raw_runs <- read.csv(file.path(output_dir, "raw_runs.csv"),
                       stringsAsFactors = FALSE, check.names = FALSE)
  fallbacks <- read.csv(file.path(output_dir, "fallbacks.csv"),
                        stringsAsFactors = FALSE, check.names = FALSE)
  timings <- read.csv(file.path(output_dir, "stage_timing.csv"),
                      stringsAsFactors = FALSE, check.names = FALSE)
  graph <- read.csv(file.path(output_dir, "graph_agreement.csv"),
                    stringsAsFactors = FALSE, check.names = FALSE)
  if (nrow(case_results) != 1L ||
      case_results$logical_sequence_id[[1L]] != 139040L ||
      !isTRUE(case_results$numerical_pass[[1L]]) ||
      case_results$statistic_absolute_error[[1L]] > 1e-9 ||
      case_results$mean_absolute_error[[1L]] > 1e-10 ||
      case_results$variance_absolute_error[[1L]] > 1e-10 ||
      case_results$p_value_absolute_error[[1L]] > 1e-10 ||
      case_results$candidate_independent[[1L]] !=
        case_results$exact_oracle_independent[[1L]] ||
      case_results$candidate_independent[[1L]] !=
        case_results$legacy_reference_independent[[1L]]) {
    stop("vertical numerical case gate failed", call. = FALSE)
  }
  if (nrow(cache) != 1L || cache$component_capacity[[1L]] != 2L ||
      !isTRUE(cache$result_bit_identical[[1L]]) ||
      !isTRUE(cache$bounded_allocation[[1L]]) ||
      !isTRUE(cache$leak_free_teardown[[1L]]) ||
      !isTRUE(cache$caller_device_restored[[1L]]) ||
      cache$peak_live_device_bytes[[1L]] <= 0) {
    stop("vertical cache and memory gate failed", call. = FALSE)
  }
  if (nrow(raw_runs) != 2L ||
      length(unique(raw_runs$logical_sequence_id)) != 1L ||
      length(unique(raw_runs$p_value)) != 1L ||
      any(raw_runs$status != "OK") ||
      any(raw_runs$dcov_status != "OK_EXACT_CUDA_GAMMA")) {
    stop("vertical deterministic replay artifact mismatch", call. = FALSE)
  }
  if (nrow(fallbacks) != 6L || anyNA(fallbacks$count) ||
      any(fallbacks$count != 0L) || nrow(timings) != 9L ||
      anyNA(timings$elapsed_ms) || any(!is.finite(timings$elapsed_ms)) ||
      any(timings$elapsed_ms < 0) || nrow(graph) != 1L ||
      isTRUE(graph$full_graph_claim[[1L]])) {
    stop("vertical authority, timing, or claim-scope gate failed",
         call. = FALSE)
  }
  if (!identical(summary$schema_version,
                 "full-cuda-ci-phase35d-vertical-summary-v1") ||
      !identical(summary$claim_scope, "phase3.5D-structural-only") ||
      !identical(summary$run_status, "COMPLETE") ||
      !isTRUE(summary$structural_pass) || !isTRUE(summary$pass) ||
      isTRUE(summary$full_graph_claim) ||
      isTRUE(summary$promotion_authority) ||
      summary$residual_d2h_bytes != 0 ||
      summary$component_d2h_bytes != 0 ||
      summary$cpu_numerical_dcov_count != 0L ||
      summary$cpu_gamma_p_value_count != 0L ||
      summary$unknown_fallback_count != 0L ||
      summary$approximate_backend_count != 0L) {
    stop("vertical artifact summary hard gate failed", call. = FALSE)
  }
  invisible(list(
    manifest = manifest,
    producer = producer,
    summary = summary,
    case_results = case_results,
    cache = cache,
    raw_runs = raw_runs
  ))
}
