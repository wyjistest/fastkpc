fail <- function(message) stop(message, call. = FALSE)

assert_true <- function(value, message) {
  if (!isTRUE(value)) fail(message)
}

assert_identical <- function(actual, expected, message) {
  if (!identical(actual, expected)) fail(message)
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

assert_sha256 <- function(value, message) {
  assert_true(
    is.character(value) && length(value) == 1L && !is.na(value) &&
      grepl("^[0-9a-f]{64}$", value),
    message
  )
}

assert_file_bytes_identical <- function(left, right, message) {
  left_info <- file.info(left)
  right_info <- file.info(right)
  assert_true(
    file.exists(left) && file.exists(right) &&
      identical(as.numeric(left_info$size), as.numeric(right_info$size)),
    message
  )
  left_connection <- file(left, open = "rb")
  right_connection <- file(right, open = "rb")
  on.exit(close(left_connection), add = TRUE)
  on.exit(close(right_connection), add = TRUE)
  repeat {
    left_bytes <- readBin(left_connection, what = "raw", n = 1024L^2L)
    right_bytes <- readBin(right_connection, what = "raw", n = 1024L^2L)
    assert_true(identical(left_bytes, right_bytes), message)
    if (length(left_bytes) == 0L) break
  }
  invisible(TRUE)
}

runner_status <- function(output) {
  status <- attr(output, "status", exact = TRUE)
  if (is.null(status)) 0L else as.integer(status)
}

run_prepared_s_contract_runner <- function(
    runner_path, census_dir, data_path, output_dir,
    max_groups = "64", parity_scope = "iteration", workers = "1",
    shard_count = "2", resume = "1") {
  environment <- c(
    paste0("FASTKPC_FULL_CUDA_PREPARED_S_CENSUS_DIR=", census_dir),
    paste0("FASTKPC_FULL_CUDA_PREPARED_S_DATA_PATH=", data_path),
    paste0("FASTKPC_FULL_CUDA_PREPARED_S_OUTPUT_DIR=", output_dir),
    paste0("FASTKPC_FULL_CUDA_PREPARED_S_MAX_GROUPS=", max_groups),
    paste0("FASTKPC_FULL_CUDA_PREPARED_S_PARITY_SCOPE=", parity_scope),
    paste0("FASTKPC_FULL_CUDA_PREPARED_S_WORKERS=", workers),
    paste0("FASTKPC_FULL_CUDA_PREPARED_S_SHARD_COUNT=", shard_count),
    paste0("FASTKPC_FULL_CUDA_PREPARED_S_RESUME=", resume)
  )
  output <- suppressWarnings(system2(
    command = file.path(R.home("bin"), "Rscript"),
    args = runner_path,
    stdout = TRUE,
    stderr = TRUE,
    env = environment
  ))
  list(status = runner_status(output), output = output)
}

assert_sorted_reasons <- function(table, message) {
  assert_true(
    is.data.frame(table) &&
      "selection_reasons" %in% names(table) &&
      is.list(table$selection_reasons) &&
      all(vapply(table$selection_reasons, function(value) {
        is.character(value) && length(value) > 0L &&
          identical(value, sort(unique(value), method = "radix"))
      }, logical(1L))),
    message
  )
}

rows_with_reason <- function(table, reason) {
  which(vapply(table$selection_reasons, function(value) {
    reason %in% value
  }, logical(1L)))
}

scramble_selection_inputs <- function(inputs) {
  result <- inputs
  table_names <- c(
    "logical_tests", "residual_requests", "same_s_setup_metadata",
    "target_fit_metadata", "target_risks", "risk_cases"
  )
  set.seed(20260711L)
  for (table_name in table_names) {
    value <- result[[table_name]]
    value <- value[sample.int(nrow(value)), , drop = FALSE]
    rownames(value) <- NULL
    character_fields <- names(value)[vapply(
      value, is.character, logical(1L)
    )]
    factor_fields <- intersect(
      character_fields,
      c(
        "case_type", "residual_key_sha256", "same_S_group_id",
        "condition_bucket", "near_alpha_bucket", "formula_class",
        "residual_key_x", "residual_key_y"
      )
    )
    for (field in factor_fields) {
      levels <- rev(sort(unique(value[[field]]), method = "radix"))
      value[[field]] <- factor(value[[field]], levels = levels)
    }
    result[[table_name]] <- value
  }
  result
}

source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")

prepared_s_runner_path <-
  "fastkpc/tools/run_full_cuda_ci_prepared_s_contract.R"
assert_true(
  file.exists(prepared_s_runner_path),
  "scaled Prepared-S runner does not exist"
)

run_spectra_fallback_injection <- function() {
  selector_name <-
    "fastkpc_full_cuda_select_prepared_s_iteration_subset"
  original_selector <- get(
    selector_name, envir = .GlobalEnv, inherits = FALSE
  )
  environment_name <- "FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK"
  original_environment <- Sys.getenv(
    environment_name, unset = NA_character_
  )
  on.exit({
    assign(
      selector_name, original_selector,
      envir = .GlobalEnv
    )
    if (is.na(original_environment)) {
      Sys.unsetenv(environment_name)
    } else {
      Sys.setenv(
        FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = original_environment
      )
    }
  }, add = TRUE)

  sample_count <- 40L
  data <- matrix(
    seq_len(sample_count * 2L) / (sample_count * 2L),
    nrow = sample_count,
    ncol = 2L
  )
  storage.mode(data) <- "double"
  keys <- sprintf("%064x", seq_len(88L))
  logical_tests <- data.frame(
    logical_sequence_id = seq_len(44L),
    S_size = rep(1L, 44L),
    reference_p_value = rep(0.2, 44L),
    alpha = rep(0.1, 44L),
    reference_decision = rep("independent", 44L),
    reference_independent = rep(TRUE, 44L),
    residual_key_x = keys[seq_len(44L)],
    residual_key_y = keys[44L + seq_len(44L)],
    stringsAsFactors = FALSE
  )
  assign(
    selector_name,
    function(inputs) list(logical_tests = logical_tests),
    envir = .GlobalEnv
  )
  residuals <- new.env(hash = TRUE, parent = emptyenv())
  for (index in seq_along(keys)) {
    assign(
      keys[[index]],
      as.numeric(seq_len(sample_count) + index),
      envir = residuals
    )
  }
  valid_diagnostics <- list(
    lowrank_mode = "spectra",
    lowrank_full_eig_count = 0L,
    lowrank_spectra_count = 2L,
    lowrank_spectra_converged_count = 2L,
    lowrank_spectra_failed_count = 0L,
    lowrank_spectra_fallback_full_eig_count = 0L,
    lowrank_spectra_nconv = 2L * 35L
  )
  fastkpc_full_cuda_prepared_s_validate_dcov_spectra_diagnostics(
    valid_diagnostics, numCol = 35L
  )
  valid_oracle_call_count <- 0L
  valid_oracle <- function(x, y, numCol, index) {
    valid_oracle_call_count <<- valid_oracle_call_count + 1L
    list(
      p.value = 0.2,
      diagnostics = valid_diagnostics
    )
  }
  fallback_diagnostics <- valid_diagnostics
  fallback_diagnostics$lowrank_spectra_fallback_full_eig_count <- 1L
  fallback_oracle <- function(x, y, numCol, index) {
    list(
      p.value = 0.2,
      diagnostics = fallback_diagnostics
    )
  }
  Sys.setenv(
    FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK =
      "prepared-s-focused-fallback-sentinel"
  )
  assert_error(
    fastkpc_full_cuda_run_prepared_s_dcov_parity(
      inputs = list(
        data = data,
        dataset_sha256 = fastkpc_full_cuda_data_hash(data)
      ),
      logical_tests = logical_tests,
      residuals = residuals,
      oracle_fun = valid_oracle
    ),
    "Prepared-S dCov parity options are unsupported",
    paste0(
      "public dCov parity must reject a caller-supplied valid oracle"
    )
  )
  assert_identical(
    valid_oracle_call_count,
    0L,
    "public dCov parity must never invoke a caller-supplied oracle"
  )
  assert_identical(
    Sys.getenv(environment_name, unset = NA_character_),
    "prepared-s-focused-fallback-sentinel",
    "public oracle rejection must preserve the prior Spectra environment"
  )
  assert_error(
    .fastkpc_full_cuda_run_prepared_s_dcov_parity_core(
      inputs = list(
        data = data,
        dataset_sha256 = fastkpc_full_cuda_data_hash(data)
      ),
      logical_tests = logical_tests,
      residuals = residuals,
      oracle_fun = fallback_oracle
    ),
    "Spectra diagnostics mismatch",
    paste0(
      "matching p-value parity must reject a reported full-eigen fallback"
    )
  )
  assert_identical(
    Sys.getenv(environment_name, unset = NA_character_),
    "prepared-s-focused-fallback-sentinel",
    "fallback rejection must restore the prior Spectra environment"
  )
}

run_scaled_selection_count_regression <- function() {
  function_name <- "fastkpc_full_cuda_prepared_s_setup_index"
  original <- get(function_name, envir = .GlobalEnv, inherits = FALSE)
  on.exit(assign(
    function_name, original, envir = .GlobalEnv
  ), add = TRUE)
  keys <- sprintf("%064x", seq_len(80L))
  groups <- sprintf("%064x", 1000L + seq_len(80L))
  synthetic_index <- data.frame(
    setup_row_index = seq_len(80L),
    same_S_group_id = groups,
    prepared_s_key_payload = paste0("payload-", seq_len(80L), "\n"),
    prepared_s_key_sha256 = keys,
    stringsAsFactors = FALSE
  )
  assign(
    function_name,
    function(inputs) synthetic_index,
    envir = .GlobalEnv
  )
  iteration <- list(
    setup_groups = data.frame(
      same_S_group_id = groups[seq_len(44L)],
      stringsAsFactors = FALSE
    )
  )
  assert_error(
    fastkpc_full_cuda_prepared_s_select_setup_corpus(
      inputs = list(),
      max_groups = 1L,
      parity_scope = "none",
      iteration = iteration
    ),
    "smaller than the iteration setup group count",
    "positive max_groups=1 must reject even when parity scope is none"
  )
  assert_error(
    fastkpc_full_cuda_prepared_s_select_setup_corpus(
      inputs = list(),
      max_groups = 43L,
      parity_scope = "none",
      iteration = iteration
    ),
    "smaller than the iteration setup group count",
    "positive max_groups=43 must reject even when parity scope is none"
  )
  selected_44 <- fastkpc_full_cuda_prepared_s_select_setup_corpus(
    inputs = list(),
    max_groups = 44L,
    parity_scope = "none",
    iteration = iteration
  )
  assert_identical(
    selected_44$prepared_s_key_sha256,
    keys[seq_len(44L)],
    "max_groups=44 must select the exact iteration closure"
  )
  selected <- fastkpc_full_cuda_prepared_s_select_setup_corpus(
    inputs = list(),
    max_groups = 64L,
    parity_scope = "none",
    iteration = iteration
  )
  assert_true(
    nrow(selected) == 64L &&
      all(groups[seq_len(44L)] %in% selected$same_S_group_id) &&
      identical(
        selected$prepared_s_key_sha256,
        keys[seq_len(64L)]
      ),
    "positive max_groups must select 44 iteration groups plus radix fill"
  )

  assign(
    function_name,
    function(inputs) synthetic_index[0L, , drop = FALSE],
    envir = .GlobalEnv
  )
  assert_error(
    fastkpc_full_cuda_prepared_s_select_setup_corpus(
      inputs = list(),
      max_groups = 0L,
      parity_scope = "none",
      iteration = iteration
    ),
    "empty",
    "an empty canonical PreparedSKey corpus must fail closed"
  )
}

synthetic_selection_evidence <- function(prefix) {
  setup_group <- paste0(prefix, "-setup")
  target_key <- paste0(prefix, "-target")
  logical_id <- 1L
  setup_groups <- data.frame(
    same_S_group_id = setup_group,
    stringsAsFactors = FALSE
  )
  setup_groups$selection_reasons <- I(list("setup-reason"))
  target_keys <- data.frame(
    residual_key_sha256 = target_key,
    same_S_group_id = setup_group,
    target = 1L,
    stringsAsFactors = FALSE
  )
  target_keys$selection_reasons <- I(list("target-reason"))
  logical_tests <- data.frame(
    logical_sequence_id = logical_id,
    residual_key_x = target_key,
    residual_key_y = paste0(prefix, "-target-y"),
    alpha = 0.1,
    reference_p_value = 0.2,
    reference_decision = "independent",
    reference_independent = TRUE,
    stringsAsFactors = FALSE
  )
  logical_tests$selection_reasons <- I(list("logical-reason"))
  list(
    setup_groups = setup_groups,
    target_keys = target_keys,
    logical_tests = logical_tests,
    reason_rows = data.frame(
      entity_type = "setup_group",
      entity_id = setup_group,
      reason = "setup-reason",
      stringsAsFactors = FALSE
    ),
    coverage = data.frame(
      coverage_type = "risk_class",
      coverage_value = "ordinary",
      canonical_count = 1L,
      selected_count = 1L,
      coverage_claimed = TRUE,
      stringsAsFactors = FALSE
    ),
    setup_group_ids_hash = paste0(prefix, "-setup-hash"),
    target_keys_hash = paste0(prefix, "-target-hash"),
    logical_test_ids_hash = paste0(prefix, "-logical-hash")
  )
}

run_canonical_artifact_evidence_regressions <- function() {
  for (selection_name in c("iteration", "qualification")) {
    canonical <- synthetic_selection_evidence(selection_name)
    changed_setup <- canonical
    changed_setup$setup_groups$same_S_group_id[[1L]] <-
      paste0(selection_name, "-other-setup")
    assert_error(
      .fastkpc_full_cuda_prepared_s_require_canonical_selection(
        changed_setup, canonical, selection_name
      ),
      "canonical selection mismatch",
      paste(selection_name, "setup substitution must fail the gate")
    )
    changed_target <- canonical
    changed_target$target_keys$residual_key_sha256[[1L]] <-
      paste0(selection_name, "-other-target")
    assert_error(
      .fastkpc_full_cuda_prepared_s_require_canonical_selection(
        changed_target, canonical, selection_name
      ),
      "canonical selection mismatch",
      paste(selection_name, "target substitution must fail the gate")
    )
    changed_logical <- canonical
    changed_logical$logical_tests$residual_key_x[[1L]] <-
      paste0(selection_name, "-other-endpoint")
    assert_error(
      .fastkpc_full_cuda_prepared_s_require_canonical_selection(
        changed_logical, canonical, selection_name
      ),
      "canonical selection mismatch",
      paste(selection_name, "logical endpoint substitution must fail")
    )
  }

  group_id <- sprintf("%064x", 101L)
  target_key <- sprintf("%064x", 102L)
  prepared_key <- sprintf("%064x", 103L)
  target_fingerprint <- sprintf("%064x", 104L)
  coefficient_hash <- sprintf("%064x", 105L)
  fitted_hash <- sprintf("%064x", 106L)
  residual_hash <- sprintf("%064x", 107L)
  canonical_targets <- data.frame(
    residual_key_sha256 = target_key,
    same_S_group_id = group_id,
    target = 1L,
    stringsAsFactors = FALSE
  )
  authenticated_states <- data.frame(
    residual_key_sha256 = target_key,
    same_S_group_id = group_id,
    target = 1L,
    prepared_s_key_sha256 = prepared_key,
    target_state_fingerprint = target_fingerprint,
    coefficient_hash = coefficient_hash,
    fitted_hash = fitted_hash,
    residual_hash = residual_hash,
    stringsAsFactors = FALSE
  )
  target_rows <- data.frame(
    parity_scope = "iteration",
    residual_key_sha256 = target_key,
    same_S_group_id = group_id,
    target = 1L,
    prepared_s_key_sha256 = prepared_key,
    target_state_fingerprint = target_fingerprint,
    coefficient_hash = coefficient_hash,
    expected_coefficient_hash = coefficient_hash,
    coefficient_hash_exact = TRUE,
    fitted_hash = fitted_hash,
    expected_fitted_hash = fitted_hash,
    fitted_hash_exact = TRUE,
    residual_hash = residual_hash,
    expected_residual_hash = residual_hash,
    residual_hash_exact = TRUE,
    residual_length = 10L,
    backend_family = "mgcvExtractCPU",
    mode = "prepared-s-fixed-sp-mgcv-reference",
    solve_source = "mgcv-C-magic-from-prepared-s",
    authoritative = TRUE,
    stringsAsFactors = FALSE
  )
  wrong_hash_rows <- target_rows
  wrong_hash_rows$coefficient_hash[[1L]] <- sprintf("%064x", 108L)
  assert_error(
    .fastkpc_full_cuda_prepared_s_validate_target_parity_evidence(
      canonical_targets,
      authenticated_states,
      wrong_hash_rows,
      scope = "iteration"
    ),
    "target parity hash mismatch",
    "target parity must recompute exactness instead of trusting booleans"
  )
  nonauthoritative_rows <- target_rows
  nonauthoritative_rows$authoritative[[1L]] <- FALSE
  assert_error(
    .fastkpc_full_cuda_prepared_s_validate_target_parity_evidence(
      canonical_targets,
      authenticated_states,
      nonauthoritative_rows,
      scope = "iteration"
    ),
    "target parity backend mismatch",
    "a non-authoritative target backend must fail the artifact gate"
  )

  other_target <- sprintf("%064x", 109L)
  canonical_logical <- data.frame(
    logical_sequence_id = 1L,
    residual_key_x = target_key,
    residual_key_y = other_target,
    alpha = 0.1,
    reference_p_value = 0.2,
    reference_decision = "independent",
    reference_independent = TRUE,
    stringsAsFactors = FALSE
  )
  diagnostics <- list(
    lowrank_mode = "spectra",
    lowrank_full_eig_count = 0L,
    lowrank_spectra_count = 2L,
    lowrank_spectra_converged_count = 2L,
    lowrank_spectra_failed_count = 0L,
    lowrank_spectra_fallback_full_eig_count = 0L,
    lowrank_spectra_nconv = 70L
  )
  dcov_rows <- data.frame(
    parity_scope = "iteration",
    logical_sequence_id = 1L,
    residual_key_x = target_key,
    residual_key_y = other_target,
    index = 1L,
    numCol = 35L,
    alpha = 0.1,
    reference_p_value = 0.2,
    p_value = 0.2,
    p_value_drift = 0,
    absolute_p_value_drift = 0,
    p_value_exact = TRUE,
    reference_signed_alpha_distance = 0.1,
    signed_alpha_distance = 0.1,
    reference_decision = "independent",
    reference_independent = TRUE,
    decision = "independent",
    decision_identical = TRUE,
    spectra_no_fallback = TRUE,
    stringsAsFactors = FALSE
  )
  dcov_rows$diagnostics <- I(list(diagnostics))
  wrong_dcov_rows <- dcov_rows
  wrong_dcov_rows$reference_p_value[[1L]] <- 0.05
  wrong_dcov_rows$p_value[[1L]] <- 0.05
  wrong_dcov_rows$reference_decision[[1L]] <- "dependent"
  wrong_dcov_rows$decision[[1L]] <- "dependent"
  assert_error(
    .fastkpc_full_cuda_prepared_s_validate_dcov_parity_evidence(
      canonical_logical, wrong_dcov_rows, scope = "iteration"
    ),
    "dCov canonical evidence mismatch",
    "dCov parity must bind p-values and decisions to canonical rows"
  )

  semantic_fingerprint <- sprintf("%064x", 110L)
  representation_fingerprint <- sprintf("%064x", 111L)
  canonical_groups <- data.frame(
    same_S_group_id = group_id,
    stringsAsFactors = FALSE
  )
  setup_records <- data.frame(
    same_S_group_id = group_id,
    prepared_s_key_sha256 = prepared_key,
    phase1_setup_fingerprint = sprintf("%064x", 112L),
    semantic_fingerprint = semantic_fingerprint,
    representation_fingerprint = representation_fingerprint,
    validator_pass = TRUE,
    stringsAsFactors = FALSE
  )
  setup_rows <- data.frame(
    parity_scope = "iteration",
    same_S_group_id = group_id,
    prepared_s_key_sha256 = prepared_key,
    phase1_setup_fingerprint = setup_records$phase1_setup_fingerprint,
    semantic_fingerprint = semantic_fingerprint,
    representation_fingerprint = representation_fingerprint,
    validator_pass = TRUE,
    semantic_fingerprint_exact = TRUE,
    representation_fingerprint_exact = TRUE,
    independent_reference_comparison = FALSE,
    independent_reference_scope = "focused_task4_only",
    semantic_evidence_provenance =
      "phase2_validator_and_authenticated_fingerprints",
    fixed_sp_behavior_evidence =
      "selected_scope_target_retarget_exact",
    fixed_sp_behavior_exact = TRUE,
    stringsAsFactors = FALSE
  )
  wrong_setup_rows <- setup_rows
  wrong_setup_rows$prepared_s_key_sha256[[1L]] <-
    sprintf("%064x", 113L)
  assert_error(
    .fastkpc_full_cuda_prepared_s_validate_setup_semantic_evidence(
      canonical_groups,
      setup_records,
      target_rows,
      wrong_setup_rows,
      scope = "iteration"
    ),
    "setup semantic canonical evidence mismatch",
    "setup semantic parity must bind keys to authenticated setups"
  )

  phase2_arguments <- list(
    structural_pass = TRUE,
    parity_pass = TRUE,
    full_counts = TRUE,
    parity_scope = "qualification",
    shard_count = 64L,
    unsupported_canonical_evaluated = TRUE,
    unsupported_canonical_setup_count = 0L
  )
  assert_true(
    do.call(
      .fastkpc_full_cuda_prepared_s_phase2_complete,
      phase2_arguments
    ),
    "authenticated full qualification evidence must complete Phase 2"
  )
  for (change in list(
      list(full_counts = FALSE),
      list(parity_scope = "iteration"),
      list(shard_count = 2L),
      list(unsupported_canonical_evaluated = FALSE),
      list(unsupported_canonical_setup_count = 1L))) {
    changed <- utils::modifyList(phase2_arguments, change)
    assert_true(
      !do.call(
        .fastkpc_full_cuda_prepared_s_phase2_complete,
        changed
      ),
      "phase2 completion must require full counts, qualification parity, canonical unsupported evidence, and 64 shards"
    )
  }

  input_fallbacks <- data.frame(
    type = c("unknown", "approximate"),
    key = c("unknown_fallback_count", "approximate_backend_count"),
    reason = rep("oracle summary counter", 2L),
    count = c(0L, 0L),
    stringsAsFactors = FALSE
  )
  canonical_iteration <- list(
    setup_groups = canonical_groups,
    target_keys = canonical_targets,
    logical_tests = canonical_logical,
    reason_rows = data.frame(
      entity_type = "setup_group",
      entity_id = group_id,
      reason = "focused",
      stringsAsFactors = FALSE
    ),
    setup_group_ids_hash = sprintf("%064x", 116L),
    target_keys_hash = sprintf("%064x", 117L),
    logical_test_ids_hash = sprintf("%064x", 118L),
    iteration_subset_hash = sprintf("%064x", 119L)
  )
  canonical_qualification <- canonical_iteration
  canonical_qualification$seed_target_keys <- canonical_targets
  canonical_qualification$coverage <- data.frame(
    coverage_type = "risk_class",
    coverage_value = "focused",
    canonical_count = 1L,
    selected_count = 1L,
    coverage_claimed = TRUE,
    stringsAsFactors = FALSE
  )
  canonical_qualification$qualification_subset_hash <-
    sprintf("%064x", 120L)
  selector_names <- c(
    "fastkpc_full_cuda_select_prepared_s_iteration_subset",
    "fastkpc_full_cuda_select_prepared_s_qualification_subset"
  )
  original_selectors <- lapply(selector_names, function(name) {
    get(name, envir = .GlobalEnv, inherits = FALSE)
  })
  names(original_selectors) <- selector_names
  on.exit({
    for (name in selector_names) {
      assign(name, original_selectors[[name]], envir = .GlobalEnv)
    }
  }, add = TRUE)
  assign(
    selector_names[[1L]],
    function(inputs) canonical_iteration,
    envir = .GlobalEnv
  )
  assign(
    selector_names[[2L]],
    function(inputs) canonical_qualification,
    envir = .GlobalEnv
  )
  authenticated <-
    .fastkpc_full_cuda_prepared_s_authenticate_artifact_evidence(
      inputs = list(fallbacks = input_fallbacks),
      iteration = canonical_iteration,
      qualification = canonical_qualification,
      authenticated_setup_records = setup_records,
      authenticated_target_states = authenticated_states,
      setup_semantic_parity = setup_rows,
      target_parity_rows = target_rows,
      dcov_parity_rows = dcov_rows,
      parity_scope = "iteration"
    )
  assert_true(
    identical(
      authenticated$target_parity_rows$coefficient_hash,
      coefficient_hash
    ) && authenticated$fallback_evidence$approximate_backend_count == 0L,
    "artifact evidence authenticator must return normalized evidence"
  )
  tampered_iteration <- canonical_iteration
  tampered_iteration$target_keys$residual_key_sha256[[1L]] <-
    sprintf("%064x", 121L)
  assert_error(
    .fastkpc_full_cuda_prepared_s_authenticate_artifact_evidence(
      inputs = list(fallbacks = input_fallbacks),
      iteration = tampered_iteration,
      qualification = canonical_qualification,
      authenticated_setup_records = setup_records,
      authenticated_target_states = authenticated_states,
      setup_semantic_parity = setup_rows,
      target_parity_rows = target_rows,
      dcov_parity_rows = dcov_rows,
      parity_scope = "iteration"
    ),
    "canonical selection mismatch",
    "artifact evidence authenticator must reject provided tampering"
  )
  tampered_fallbacks <- input_fallbacks
  tampered_fallbacks$count[[1L]] <- 1L
  assert_error(
    .fastkpc_full_cuda_prepared_s_derive_fallback_evidence(
      tampered_fallbacks, target_rows, dcov_rows
    ),
    "authenticated fallback evidence mismatch",
    "tampered authenticated fallback counters must fail the gate"
  )
  scaled_unsupported <-
    .fastkpc_full_cuda_prepared_s_unsupported_evidence(
      selected_group_count = 64L,
      validated_selected_setup_count = 64L,
      canonical_group_count = 8634L
    )
  assert_true(
    scaled_unsupported$unsupported_selected_setup_count == 0L &&
      is.na(scaled_unsupported$unsupported_canonical_setup_count) &&
      !scaled_unsupported$unsupported_canonical_evaluated &&
      identical(scaled_unsupported$unsupported_scope, "selected"),
    "scaled unsupported evidence must be selected-scope only"
  )

  lineage <- .fastkpc_full_cuda_prepared_s_selected_corpus_lineage(
    full_canonical_target_key_corpus_hash = sprintf("%064x", 114L),
    full_canonical_prepared_s_key_corpus_hash = sprintf("%064x", 115L),
    selected_target_keys = target_key,
    selected_prepared_s_keys = prepared_key
  )
  assert_true(
    all(c(
      "full_canonical_target_key_corpus_hash",
      "full_canonical_prepared_s_key_corpus_hash",
      "selected_target_key_corpus_hash",
      "selected_prepared_s_key_corpus_hash"
    ) %in% names(lineage)) &&
      !identical(
        lineage$full_canonical_target_key_corpus_hash,
        lineage$selected_target_key_corpus_hash
    ),
    "selected shard lineage must distinguish full and selected corpora"
  )
  manifest_lineage <- c(
    list(
      canonical_target_key_corpus_hash =
        lineage$full_canonical_target_key_corpus_hash
    ),
    lineage
  )
  assert_true(
    .fastkpc_full_cuda_prepared_s_manifest_lineage_exact(
      manifest_lineage, lineage
    ),
    "artifact lineage must bind full and selected corpus hashes"
  )
  for (field in names(manifest_lineage)) {
    tampered_manifest <- manifest_lineage
    tampered_manifest[[field]] <- sprintf("%064x", 900L)
    assert_true(
      !.fastkpc_full_cuda_prepared_s_manifest_lineage_exact(
        tampered_manifest, lineage
      ),
      paste("artifact lineage must reject tampered", field)
    )
  }
}

run_empty_environment_regressions <- function() {
  ambient_summary_path <- file.path(getwd(), "summary.json")
  ambient_summary_existed <- file.exists(ambient_summary_path)
  ambient_summary_hash <- if (ambient_summary_existed) {
    fastkpc_full_cuda_census_file_hash(ambient_summary_path)
  } else {
    NA_character_
  }
  variables <- c(
    "FASTKPC_FULL_CUDA_PREPARED_S_CENSUS_DIR",
    "FASTKPC_FULL_CUDA_PREPARED_S_DATA_PATH",
    "FASTKPC_FULL_CUDA_PREPARED_S_OUTPUT_DIR",
    "FASTKPC_FULL_CUDA_PREPARED_S_MAX_GROUPS",
    "FASTKPC_FULL_CUDA_PREPARED_S_PARITY_SCOPE",
    "FASTKPC_FULL_CUDA_PREPARED_S_WORKERS",
    "FASTKPC_FULL_CUDA_PREPARED_S_SHARD_COUNT",
    "FASTKPC_FULL_CUDA_PREPARED_S_RESUME"
  )
  for (variable in variables) {
    output_dir <- tempfile("prepared-s-empty-env-")
    values <- c(
      FASTKPC_FULL_CUDA_PREPARED_S_CENSUS_DIR =
        file.path(output_dir, "missing-census"),
      FASTKPC_FULL_CUDA_PREPARED_S_DATA_PATH =
        file.path(output_dir, "missing-data.rds"),
      FASTKPC_FULL_CUDA_PREPARED_S_OUTPUT_DIR = output_dir,
      FASTKPC_FULL_CUDA_PREPARED_S_MAX_GROUPS = "44",
      FASTKPC_FULL_CUDA_PREPARED_S_PARITY_SCOPE = "none",
      FASTKPC_FULL_CUDA_PREPARED_S_WORKERS = "1",
      FASTKPC_FULL_CUDA_PREPARED_S_SHARD_COUNT = "1",
      FASTKPC_FULL_CUDA_PREPARED_S_RESUME = "1"
    )
    values[[variable]] <- ""
    output <- suppressWarnings(system2(
      command = file.path(R.home("bin"), "Rscript"),
      args = prepared_s_runner_path,
      stdout = TRUE,
      stderr = TRUE,
      env = paste0(names(values), "=", unname(values))
    ))
    assert_true(
      runner_status(output) != 0L &&
        any(grepl(variable, output, fixed = TRUE)),
      paste(variable, "explicit empty value must fail its strict parser")
    )
    if (identical(
          variable, "FASTKPC_FULL_CUDA_PREPARED_S_OUTPUT_DIR"
        )) {
      assert_true(
        identical(file.exists(ambient_summary_path), ambient_summary_existed) &&
          (!ambient_summary_existed || identical(
            fastkpc_full_cuda_census_file_hash(ambient_summary_path),
            ambient_summary_hash
          )),
        "empty output path must not publish a fallback summary in the worktree"
      )
      unlink(output_dir, recursive = TRUE, force = TRUE)
      next
    }
    failure_path <- file.path(output_dir, "summary.json")
    assert_true(
      file.exists(failure_path),
      paste(variable, "empty-value failure summary must exist")
    )
    failure <- jsonlite::read_json(
      failure_path, simplifyVector = TRUE
    )
    assert_true(
      identical(as.character(failure$stage), "parse_environment") &&
        grepl(
          variable,
          as.character(failure$error_message),
          fixed = TRUE
        ),
      paste(variable, "must be rejected by its strict parser")
    )
    unlink(output_dir, recursive = TRUE, force = TRUE)
  }
}

run_resume_zero_preflight_regression <- function() {
  output_dir <- tempfile("prepared-s-resume-zero-")
  dir.create(output_dir, recursive = TRUE)
  paths <- fastkpc_full_cuda_prepared_s_artifact_paths(output_dir)
  dir.create(paths$shards_dir, recursive = TRUE)
  fixtures <- c(
    manifest = paths$manifest_json,
    csv = paths$iteration_coverage_csv,
    rds = paths$prepared_s_setup_index_rds,
    unknown = file.path(output_dir, "unexpected-entry.txt")
  )
  fixture_bytes <- list(
    charToRaw("existing-manifest"),
    charToRaw("existing-csv"),
    charToRaw("existing-rds"),
    charToRaw("existing-unknown-entry")
  )
  for (index in seq_along(fixtures)) {
    connection <- file(fixtures[[index]], open = "wb")
    writeBin(fixture_bytes[[index]], connection)
    close(connection)
  }
  values <- c(
    FASTKPC_FULL_CUDA_PREPARED_S_CENSUS_DIR =
      file.path(output_dir, "missing-census"),
    FASTKPC_FULL_CUDA_PREPARED_S_DATA_PATH =
      file.path(output_dir, "missing-data.rds"),
    FASTKPC_FULL_CUDA_PREPARED_S_OUTPUT_DIR = output_dir,
    FASTKPC_FULL_CUDA_PREPARED_S_MAX_GROUPS = "44",
    FASTKPC_FULL_CUDA_PREPARED_S_PARITY_SCOPE = "none",
    FASTKPC_FULL_CUDA_PREPARED_S_WORKERS = "1",
    FASTKPC_FULL_CUDA_PREPARED_S_SHARD_COUNT = "1",
    FASTKPC_FULL_CUDA_PREPARED_S_RESUME = "0"
  )
  output <- suppressWarnings(system2(
    command = file.path(R.home("bin"), "Rscript"),
    args = prepared_s_runner_path,
    stdout = TRUE,
    stderr = TRUE,
    env = paste0(names(values), "=", unname(values))
  ))
  assert_true(
    runner_status(output) != 0L &&
      any(grepl("resume is disabled", output, fixed = TRUE)),
    "resume=0 must fail before input loading when any artifact exists"
  )
  for (index in seq_along(fixtures)) {
    connection <- file(fixtures[[index]], open = "rb")
    actual <- readBin(connection, what = "raw", n = 1000L)
    close(connection)
    assert_identical(
      actual,
      fixture_bytes[[index]],
      paste(names(fixtures)[[index]], "bytes must remain unchanged")
    )
  }
  failed_paths <- list.files(
    file.path(output_dir, "failed_runs"),
    pattern = "^failed_[0-9]+\\.summary\\.json$",
    full.names = TRUE
  )
  assert_true(
    length(failed_paths) == 1L && !file.exists(paths$summary_json),
    "resume preflight failure must publish only an independent summary"
  )
  unlink(output_dir, recursive = TRUE, force = TRUE)
}

reason_store <- fastkpc_full_cuda_prepared_s_selection_reason_store()
fastkpc_full_cuda_prepared_s_add_selection_reason(
  reason_store, "group-a", c("reason-z", "reason-a", "reason-z")
)
fastkpc_full_cuda_prepared_s_add_selection_reason(
  reason_store,
  c("group-a", "group-b", "group-b"),
  c("reason-b", "reason-z", "reason-a")
)
assert_identical(
  fastkpc_full_cuda_prepared_s_selection_reasons(
    reason_store, c("group-a", "group-b")
  ),
  list(
    c("reason-a", "reason-b", "reason-z"),
    c("reason-a", "reason-z")
  ),
  "selection reasons must support one-to-many and pairwise aggregation"
)
run_spectra_fallback_injection()
run_scaled_selection_count_regression()
run_canonical_artifact_evidence_regressions()
run_empty_environment_regressions()
run_resume_zero_preflight_regression()

if (identical(
      Sys.getenv("FASTKPC_PREPARED_S_TEST_SCOPE", unset = "full"),
      "focused"
    )) {
  cat("PASS focused Prepared-S reason and Spectra fail-closed tests\n")
  quit(save = "no", status = 0L)
}

required_packages <- c("mgcv", "jsonlite", "Rcpp")
missing_packages <- required_packages[!vapply(
  required_packages, requireNamespace, logical(1L), quietly = TRUE
)]
if (length(missing_packages) > 0L) {
  cat(
    "SKIP Prepared-S real subset: required packages are unavailable: ",
    paste(missing_packages, collapse = ","), "\n",
    sep = ""
  )
  quit(save = "no", status = 0L)
}

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
if (!dir.exists(census_dir) || !file.exists(data_path)) {
  cat(
    "SKIP Prepared-S real subset: authenticated artifacts are unavailable\n"
  )
  quit(save = "no", status = 0L)
}

inputs <- fastkpc_full_cuda_prepared_s_load_inputs(
  census_dir = census_dir,
  data_path = data_path
)

iteration <- fastkpc_full_cuda_select_prepared_s_iteration_subset(inputs)
assert_true(nrow(iteration$setup_groups) == 44L, "iteration setup count")
assert_true(nrow(iteration$target_keys) == 270L, "iteration target count")
assert_true(nrow(iteration$logical_tests) == 44L, "iteration dCov count")

selected_setup_index <-
  fastkpc_full_cuda_prepared_s_select_setup_corpus(
    inputs = inputs,
    max_groups = 64L,
    parity_scope = "iteration",
    iteration = iteration
  )
selected_inputs <- .fastkpc_full_cuda_prepared_s_selected_inputs(
  inputs, selected_setup_index
)
assert_identical(
  as.character(selected_inputs$manifest$canonical_key_corpus_hash),
  as.character(inputs$manifest$canonical_key_corpus_hash),
  "selected inputs must not overwrite full canonical target lineage"
)
selected_plan <-
  .fastkpc_full_cuda_prepared_s_selected_shard_plan(
    inputs = inputs,
    selected_setup_index = selected_setup_index,
    shard_count = 2L
  )
full_setup_index <- fastkpc_full_cuda_prepared_s_setup_index(inputs)
expected_full_prepared_hash <- fastkpc_full_cuda_census_key_set_hash(sort(
  as.character(full_setup_index$prepared_s_key_sha256), method = "radix"
))
expected_selected_prepared_hash <-
  fastkpc_full_cuda_census_key_set_hash(sort(
    as.character(selected_setup_index$prepared_s_key_sha256),
    method = "radix"
  ))
expected_selected_targets <-
  fastkpc_full_cuda_prepared_s_target_rows_for_setups(
    inputs, selected_setup_index
  )
expected_selected_target_hash <- fastkpc_full_cuda_census_key_set_hash(sort(
  as.character(expected_selected_targets$residual_key_sha256),
  method = "radix"
))
assert_true(
  identical(
    names(selected_plan$context),
    fastkpc_full_cuda_prepared_s_output_shard_context_fields()
  ) && identical(
    selected_plan$full_canonical_target_key_corpus_hash,
    as.character(inputs$manifest$canonical_key_corpus_hash)
  ) && identical(
    selected_plan$full_canonical_prepared_s_key_corpus_hash,
    expected_full_prepared_hash
  ) && identical(
    selected_plan$selected_prepared_s_key_corpus_hash,
    expected_selected_prepared_hash
  ) && identical(
    selected_plan$selected_target_key_corpus_hash,
    expected_selected_target_hash
  ) && identical(
    selected_plan$context$canonical_target_key_corpus_hash,
    as.character(inputs$manifest$canonical_key_corpus_hash)
  ) && identical(
    selected_plan$context$prepared_s_key_corpus_hash,
    expected_selected_prepared_hash
  ),
  "selected shard plan must bind full and selected corpus lineage"
)
assert_true(
  identical(
    selected_plan$inputs$manifest$canonical_key_corpus_hash,
    as.character(inputs$manifest$canonical_key_corpus_hash)
  ) && identical(
    selected_plan$inputs$manifest$full_canonical_target_key_corpus_hash,
    as.character(inputs$manifest$canonical_key_corpus_hash)
  ) && identical(
    selected_plan$inputs$manifest$full_canonical_prepared_s_key_corpus_hash,
    expected_full_prepared_hash
  ) && identical(
    selected_plan$inputs$manifest$selected_prepared_s_key_corpus_hash,
    expected_selected_prepared_hash
  ) && identical(
    selected_plan$inputs$manifest$selected_target_key_corpus_hash,
    expected_selected_target_hash
  ),
  "private selected manifest must carry separate full and selected hashes"
)
assert_identical(
  as.character(iteration$setup_groups$same_S_group_id),
  sort(as.character(iteration$setup_groups$same_S_group_id),
       method = "radix"),
  "iteration setup order"
)
assert_identical(
  as.character(iteration$target_keys$residual_key_sha256),
  sort(as.character(iteration$target_keys$residual_key_sha256),
       method = "radix"),
  "iteration target order"
)
assert_identical(
  as.integer(iteration$logical_tests$logical_sequence_id),
  sort(as.integer(iteration$logical_tests$logical_sequence_id)),
  "iteration logical-test order"
)
assert_sorted_reasons(
  iteration$setup_groups, "iteration setup reasons must be canonical"
)
assert_sorted_reasons(
  iteration$target_keys, "iteration target reasons must be canonical"
)
assert_sorted_reasons(
  iteration$logical_tests, "iteration logical reasons must be canonical"
)
assert_sha256(
  iteration$iteration_subset_hash,
  "iteration subset hash must be versioned and deterministic"
)
assert_identical(
  iteration$iteration_subset_hash,
  "d69956655e2ee0186aceb4bf2d545b831051c389aed139769d9e5902f433fb96",
  "iteration subset hash must retain the frozen Task 5 identity"
)

risk_table <-
  fastkpc_full_cuda_reconstruct_prepared_s_target_risk_table(inputs)
assert_true(
  nrow(risk_table) == 110617L &&
    !anyDuplicated(as.character(risk_table$residual_key_sha256)) &&
    identical(
      as.character(risk_table$residual_key_sha256),
      sort(as.character(risk_table$residual_key_sha256), method = "radix")
    ),
  "reconstructed target-risk table must preserve canonical key order"
)
assert_true(
  all(c(
    "penalty_count", "condition", "condition_bucket", "S_size",
    "request_multiplicity", "setup_target_count",
    "setup_logical_request_count", "convergence_signature",
    "optimizer_iterations"
  ) %in% names(risk_table)),
  "reconstructed target-risk table must expose selection metadata"
)
rank_deficient <- risk_table$rank_deficient %in% TRUE
assert_true(
  any(rank_deficient) &&
    all(is.infinite(risk_table$condition[rank_deficient])) &&
    all(risk_table$condition_bucket[rank_deficient] ==
          "rank_deficient_inf") &&
    all(risk_table$nonfinite_metadata[rank_deficient]) &&
    all(risk_table$coefficient_all_finite[rank_deficient]) &&
    all(risk_table$fitted_all_finite[rank_deficient]) &&
    all(risk_table$residual_all_finite[rank_deficient]),
  "rank-deficient targets must retain Inf metadata and finite outputs"
)

all_logical <- as.data.frame(
  inputs$logical_tests, stringsAsFactors = FALSE
)
tight <- all_logical$S_size > 0L &
  is.finite(all_logical$absolute_log_distance_from_alpha) &
  all_logical$absolute_log_distance_from_alpha <= 1e-3
tight_ids <- as.integer(all_logical$logical_sequence_id[tight])
assert_true(
  length(tight_ids) == 6L &&
    all(tight_ids %in% iteration$logical_tests$logical_sequence_id),
  "iteration must include all six canonical tight-alpha tests"
)

risk_key_index_x <- match(
  as.character(all_logical$residual_key_x),
  as.character(risk_table$residual_key_sha256)
)
risk_key_index_y <- match(
  as.character(all_logical$residual_key_y),
  as.character(risk_table$residual_key_sha256)
)
numerical_or_convergence_risk <- with(
  risk_table,
  high_condition | rank_deficient | near_constant_target |
    near_constant_conditioner | mgcv_warning | mgcv_nonconverged |
    nonfinite_metadata
)
finite_ordinary_target <- with(
  risk_table,
  coefficient_all_finite & fitted_all_finite & residual_all_finite &
    is.finite(condition) & condition < 1e8 &
    !numerical_or_convergence_risk
)
ordinary_eligible <- all_logical$S_size > 0L &
  !is.na(risk_key_index_x) & !is.na(risk_key_index_y) &
  finite_ordinary_target[risk_key_index_x] &
  finite_ordinary_target[risk_key_index_y] &
  is.finite(all_logical$absolute_log_distance_from_alpha) &
  all_logical$absolute_log_distance_from_alpha > log(2)
key_x <- as.character(all_logical$residual_key_x)
key_y <- as.character(all_logical$residual_key_y)
x_first <- key_x <= key_y
canonical_pair_key <- paste0(
  ifelse(x_first, key_x, key_y), "|",
  ifelse(x_first, key_y, key_x)
)
assert_true(
  all(risk_table$multi_penalty[risk_table$S_size %in% 3:7]),
  paste0(
    "canonical S_size 3..7 targets must retain unavoidable structural ",
    "multi-penalty classification"
  )
)
for (S_size in seq_len(7L)) {
  reason <- paste0("ordinary_lower_median:S_size=", S_size)
  selected_index <- rows_with_reason(iteration$logical_tests, reason)
  assert_true(
    length(selected_index) == 1L,
    paste0("iteration must select one ordinary witness for S_size=", S_size)
  )
  selected <- iteration$logical_tests[selected_index, , drop = FALSE]
  endpoints <- match(
    c(selected$residual_key_x[[1L]], selected$residual_key_y[[1L]]),
    risk_table$residual_key_sha256
  )
  assert_true(
    !anyNA(endpoints) && all(finite_ordinary_target[endpoints]) &&
      selected$absolute_log_distance_from_alpha[[1L]] > log(2),
    paste0(
      "ordinary witness endpoints must satisfy frozen numerical and ",
      "convergence risk rules for S_size=", S_size
    )
  )
  if (S_size >= 3L) {
    assert_true(
      all(risk_table$multi_penalty[endpoints]),
      paste0(
        "ordinary S_size=", S_size,
        " witness must allow structural multi_penalty"
      )
    )
  }
  candidates <- which(
    ordinary_eligible & all_logical$S_size == S_size
  )
  candidates <- candidates[order(
    canonical_pair_key[candidates],
    all_logical$logical_sequence_id[candidates],
    method = "radix"
  )]
  expected <- candidates[[(length(candidates) + 1L) %/% 2L]]
  assert_identical(
    selected$logical_sequence_id[[1L]],
    as.integer(all_logical$logical_sequence_id[[expected]]),
    paste0("ordinary lower-median identity mismatch for S_size=", S_size)
  )
}

nonconverged <- risk_table$mgcv_nonconverged %in% TRUE
convergence_strata <- paste(
  risk_table$convergence_signature[nonconverged],
  risk_table$S_size[nonconverged],
  risk_table$condition_bucket[nonconverged],
  sep = "|"
)
observed_convergence_strata <- sort(
  unique(convergence_strata), method = "radix"
)
assert_true(
  length(observed_convergence_strata) == 3L,
  "canonical frozen mgcv_nonconverged target strata must equal three"
)
for (stratum in observed_convergence_strata) {
  reason <- paste0("convergence_risk:", stratum)
  selected_index <- rows_with_reason(iteration$target_keys, reason)
  assert_true(
    length(selected_index) == 1L,
    paste0("iteration must select one frozen convergence stratum: ", stratum)
  )
  candidates <- which(
    nonconverged & paste(
      risk_table$convergence_signature,
      risk_table$S_size,
      risk_table$condition_bucket,
      sep = "|"
    ) == stratum
  )
  expected <- candidates[order(
    -risk_table$optimizer_iterations[candidates],
    risk_table$residual_key_sha256[candidates],
    method = "radix"
  )[[1L]]]
  assert_identical(
    iteration$target_keys$residual_key_sha256[[selected_index]],
    risk_table$residual_key_sha256[[expected]],
    paste0("frozen convergence stratum ordering mismatch: ", stratum)
  )
}
convergence_reason_count <- sum(vapply(
  iteration$target_keys$selection_reasons,
  function(value) sum(startsWith(value, "convergence_risk:")),
  integer(1L)
))
assert_identical(
  as.integer(convergence_reason_count), 3L,
  "iteration must record exactly three frozen convergence reasons"
)

setup_group_ids <- sort(unique(
  as.character(risk_table$same_S_group_id)
), method = "radix")
first_target_by_group <- match(
  setup_group_ids, risk_table$same_S_group_id
)
setup_metrics <- data.frame(
  same_S_group_id = setup_group_ids,
  setup_target_count =
    risk_table$setup_target_count[first_target_by_group],
  setup_logical_request_count =
    risk_table$setup_logical_request_count[first_target_by_group],
  stringsAsFactors = FALSE
)
anchor_specs <- list(
  list("setup_target_fanout_anchor", "setup_target_count", 2L),
  list("setup_target_fanout_anchor", "setup_target_count", 9L),
  list("setup_target_fanout_anchor", "setup_target_count", 47L),
  list(
    "setup_logical_request_load_anchor",
    "setup_logical_request_count", 2L
  ),
  list(
    "setup_logical_request_load_anchor",
    "setup_logical_request_count", 16L
  ),
  list(
    "setup_logical_request_load_anchor",
    "setup_logical_request_count", 3092L
  )
)
for (spec in anchor_specs) {
  reason <- paste0(spec[[1L]], "=", spec[[3L]])
  selected_index <- rows_with_reason(iteration$setup_groups, reason)
  assert_true(
    length(selected_index) == 1L,
    paste0("iteration must record one anchor reason: ", reason)
  )
  metric <- setup_metrics[[spec[[2L]]]]
  distance <- abs(metric - spec[[3L]])
  candidates <- setup_metrics$same_S_group_id[
    distance == min(distance)
  ]
  expected_group <- sort(candidates, method = "radix")[[1L]]
  selected_group <-
    iteration$setup_groups$same_S_group_id[[selected_index]]
  assert_identical(
    selected_group, expected_group,
    paste0("anchor nearest/tie semantics mismatch: ", reason)
  )
  canonical_group_keys <- risk_table$residual_key_sha256[
    risk_table$same_S_group_id == selected_group
  ]
  assert_true(
    all(canonical_group_keys %in%
          iteration$target_keys$residual_key_sha256),
    paste0("anchor must include every target in group: ", reason)
  )
}

for (index in seq_len(nrow(iteration$logical_tests))) {
  logical_row <- iteration$logical_tests[index, , drop = FALSE]
  logical_id <- logical_row$logical_sequence_id[[1L]]
  endpoints <- c(
    logical_row$residual_key_x[[1L]],
    logical_row$residual_key_y[[1L]]
  )
  endpoint_index <- match(
    endpoints, iteration$target_keys$residual_key_sha256
  )
  assert_true(
    !anyNA(endpoint_index) && all(vapply(
      iteration$target_keys$selection_reasons[endpoint_index],
      function(value) paste0("logical_endpoint:", logical_id) %in% value,
      logical(1L)
    )),
    paste0("logical endpoint closure mismatch: ", logical_id)
  )
}

canonical_setups <- as.data.frame(
  inputs$same_s_setup_metadata, stringsAsFactors = FALSE
)
for (group_id in iteration$setup_groups$same_S_group_id) {
  group_keys <- sort(
    risk_table$residual_key_sha256[
      risk_table$same_S_group_id == group_id
    ],
    method = "radix"
  )
  setup_index <- match(group_id, canonical_setups$same_S_group_id)
  representative <-
    canonical_setups$representative_residual_key_sha256[[setup_index]]
  lower_median <- group_keys[[(length(group_keys) + 1L) %/% 2L]]
  maximum <- group_keys[[length(group_keys)]]
  closure <- c(representative, lower_median, maximum)
  closure_reasons <- c(
    "setup_representative", "setup_lower_median_target",
    "setup_maximum_target"
  )
  closure_index <- match(
    closure, iteration$target_keys$residual_key_sha256
  )
  assert_true(
    !anyNA(closure_index) && all(vapply(seq_along(closure), function(i) {
      closure_reasons[[i]] %in%
        iteration$target_keys$selection_reasons[[closure_index[[i]]]]
    }, logical(1L))),
    paste0("setup three-target closure mismatch: ", group_id)
  )
}

qualification <-
  fastkpc_full_cuda_select_prepared_s_qualification_subset(inputs)
assert_true(
  nrow(qualification$seed_target_keys) == 2356L,
  "qualification seed count"
)
assert_true(
  nrow(qualification$target_keys) == 6143L,
  "qualification expanded target count"
)
assert_true(
  nrow(qualification$logical_tests) == 3808L,
  "qualification logical-test count"
)
assert_true(
  nrow(qualification$setup_groups) == 2061L,
  "qualification same-S group count"
)
assert_true(
  sum(qualification$logical_tests$near_alpha) == 1478L,
  "all conditional near-alpha tests must be selected"
)

penalty_distribution <- table(factor(
  qualification$target_keys$penalty_count,
  levels = c(1L, 3L, 4L, 5L, 6L, 7L)
))
assert_identical(
  unname(as.integer(penalty_distribution)),
  c(3327L, 872L, 837L, 730L, 312L, 65L),
  "qualification penalty distribution"
)

rare <- with(
  risk_table,
  rank_deficient | nonfinite_metadata | mgcv_nonconverged |
    (high_condition & penalty_count == 1L)
)
rare_keys <- as.character(risk_table$residual_key_sha256[rare])
seed_keys <- as.character(
  qualification$seed_target_keys$residual_key_sha256
)
assert_true(
  all(rare_keys %in% seed_keys),
  "qualification seed must include every canonical rare-risk key"
)

conditional_near_alpha <- with(
  inputs$logical_tests,
  S_size > 0L & is.finite(absolute_log_distance_from_alpha) &
    absolute_log_distance_from_alpha <= log(2)
)
near_alpha_ids <- as.integer(
  inputs$logical_tests$logical_sequence_id[conditional_near_alpha]
)
assert_true(
  length(near_alpha_ids) == 1478L &&
    all(near_alpha_ids %in%
          qualification$logical_tests$logical_sequence_id),
  "qualification must include every conditional near-alpha test"
)

assert_sorted_reasons(
  qualification$seed_target_keys,
  "qualification seed reasons must be canonical"
)
assert_sorted_reasons(
  qualification$setup_groups,
  "qualification setup reasons must be canonical"
)
assert_sorted_reasons(
  qualification$target_keys,
  "qualification target reasons must be canonical"
)
assert_sorted_reasons(
  qualification$logical_tests,
  "qualification logical reasons must be canonical"
)
assert_sha256(
  qualification$qualification_subset_hash,
  "qualification subset hash must be versioned and deterministic"
)
assert_identical(
  qualification$qualification_subset_hash,
  "0adea2bac7b31615421f180b6caa5aeef5567bafa0e45a319b358136bf429c61",
  "qualification subset hash must retain the canonical Task 5 identity"
)
qualification_comma_decimal <- local({
  prior_options <- options(OutDec = ",")
  on.exit(options(prior_options), add = TRUE)
  fastkpc_full_cuda_select_prepared_s_qualification_subset(inputs)
})
assert_identical(
  qualification_comma_decimal$qualification_subset_hash,
  qualification$qualification_subset_hash,
  "qualification hash must be independent of the decimal output option"
)
qualification_hash_fields <- sort(
  grep("_hash$", names(qualification), value = TRUE),
  method = "radix"
)
assert_identical(
  qualification_comma_decimal[qualification_hash_fields],
  qualification[qualification_hash_fields],
  "all qualification hashes must be independent of output options"
)
assert_identical(
  qualification_comma_decimal$reason_rows,
  qualification$reason_rows,
  "qualification reason rows must be byte-identical across decimal options"
)
assert_identical(
  qualification_comma_decimal$coverage,
  qualification$coverage,
  "qualification coverage rows must be byte-identical across decimal options"
)

coverage <- qualification$coverage
assert_true(
  is.data.frame(coverage) &&
    all(c(
      "coverage_type", "coverage_value", "canonical_count",
      "selected_count", "coverage_claimed"
    ) %in% names(coverage)) &&
    all(c(
      "risk_class", "condition_bucket", "penalty_count", "S_size",
      "reference_decision", "setup_target_count_quantile"
    ) %in% coverage$coverage_type),
  "qualification coverage must report every required dimension"
)
absent_risks <- c(
  "near_constant_target", "near_constant_conditioner", "mgcv_warning"
)
absent_rows <- coverage$coverage_type == "risk_class" &
  coverage$coverage_value %in% absent_risks
assert_true(
  sum(absent_rows) == length(absent_risks) &&
    all(coverage$canonical_count[absent_rows] == 0L) &&
    all(coverage$selected_count[absent_rows] == 0L) &&
    !any(coverage$coverage_claimed[absent_rows]),
  "canonically absent risk classes must remain zero and unclaimed"
)

scrambled <- scramble_selection_inputs(inputs)
iteration_scrambled <-
  fastkpc_full_cuda_select_prepared_s_iteration_subset(scrambled)
qualification_scrambled <-
  fastkpc_full_cuda_select_prepared_s_qualification_subset(scrambled)
assert_identical(
  iteration_scrambled$iteration_subset_hash,
  iteration$iteration_subset_hash,
  "iteration selection must ignore input row and factor order"
)
assert_identical(
  qualification_scrambled$qualification_subset_hash,
  qualification$qualification_subset_hash,
  "qualification selection must ignore input row and factor order"
)
assert_identical(
  iteration_scrambled$target_keys$residual_key_sha256,
  iteration$target_keys$residual_key_sha256,
  "iteration selected keys must be stable under scrambling"
)
assert_identical(
  qualification_scrambled$target_keys$residual_key_sha256,
  qualification$target_keys$residual_key_sha256,
  "qualification selected keys must be stable under scrambling"
)

if (identical(
      Sys.getenv("FASTKPC_PREPARED_S_TEST_SCOPE", unset = "full"),
      "selection"
    )) {
  cat("PASS exact Prepared-S iteration and qualification selection tests\n")
  quit(save = "no", status = 0L)
}

prepared_by_group <- setNames(lapply(
  seq_len(nrow(iteration$setup_groups)),
  function(index) {
    group_id <- iteration$setup_groups$same_S_group_id[[index]]
    setup_index <- match(
      group_id, inputs$same_s_setup_metadata$same_S_group_id
    )
    assert_true(
      !is.na(setup_index),
      "selected group must resolve to canonical same-S metadata"
    )
    fastkpc_full_cuda_build_prepared_s_setup(
      inputs,
      inputs$same_s_setup_metadata[setup_index, , drop = FALSE]
    )
  }
), as.character(iteration$setup_groups$same_S_group_id))
assert_true(
  length(prepared_by_group) == 44L &&
    identical(
      names(prepared_by_group),
      as.character(iteration$setup_groups$same_S_group_id)
    ),
  "iteration must build every selected PreparedSSetup exactly once"
)

assert_error(
  fastkpc_full_cuda_run_prepared_s_target_parity(
    inputs = inputs,
    prepared_by_group = prepared_by_group,
    target_keys = iteration$target_keys[
      rev(seq_len(nrow(iteration$target_keys))), , drop = FALSE
    ]
  ),
  "target key set/order mismatch",
  "target parity must reject a noncanonical selected-key order"
)
target_parity <- fastkpc_full_cuda_run_prepared_s_target_parity(
  inputs = inputs,
  prepared_by_group = prepared_by_group,
  target_keys = iteration$target_keys
)
assert_true(
  is.list(target_parity) && is.data.frame(target_parity$rows) &&
    nrow(target_parity$rows) == 270L &&
    identical(
      target_parity$rows$residual_key_sha256,
      iteration$target_keys$residual_key_sha256
    ),
  "target parity must preserve the exact iteration target order"
)
assert_true(
  all(target_parity$rows$coefficient_hash_exact) &&
    all(target_parity$rows$fitted_hash_exact) &&
    all(target_parity$rows$residual_hash_exact),
  "all iteration targets must reproduce exact Phase 1 hashes"
)
assert_true(
  is.environment(target_parity$residuals) &&
    identical(target_parity$target_state_build_count, 44L) &&
    identical(target_parity$target_state_cache_group_count, 44L),
  "target parity must cache one TargetState table per same-S group"
)

dcov_env_name <- "FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK"
original_dcov_env <- Sys.getenv(dcov_env_name, unset = NA_character_)
Sys.setenv(FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK =
             "prepared-s-restoration-sentinel")
assert_error(
  fastkpc_full_cuda_run_prepared_s_dcov_parity(
    inputs = inputs,
    logical_tests = iteration$logical_tests[
      rev(seq_len(nrow(iteration$logical_tests))), , drop = FALSE
    ],
    residuals = target_parity$residuals
  ),
  "logical test set/order mismatch",
  "dCov parity must reject a noncanonical logical-test order"
)
assert_identical(
  Sys.getenv(dcov_env_name, unset = NA_character_),
  "prepared-s-restoration-sentinel",
  "dCov order validation must restore the prior environment"
)
dcov_parity <- fastkpc_full_cuda_run_prepared_s_dcov_parity(
  inputs = inputs,
  logical_tests = iteration$logical_tests,
  residuals = target_parity$residuals
)
assert_identical(
  Sys.getenv(dcov_env_name, unset = NA_character_),
  "prepared-s-restoration-sentinel",
  "dCov parity must restore the prior Spectra route environment"
)
assert_true(
  is.data.frame(dcov_parity$rows) && nrow(dcov_parity$rows) == 44L &&
    identical(
      dcov_parity$rows$logical_sequence_id,
      iteration$logical_tests$logical_sequence_id
    ),
  "dCov parity must preserve the exact iteration logical-test order"
)
assert_true(
  max(dcov_parity$rows$absolute_p_value_drift) == 0 &&
    all(dcov_parity$rows$p_value_exact) &&
    sum(dcov_parity$rows$p_value_exact) == 44L &&
    all(dcov_parity$rows$decision_identical) &&
    identical(dcov_parity$decision_flip_count, 0L),
  "all 44 dCov results must be exact with zero drift or decision flips"
)
assert_true(
  is.list(dcov_parity$rows$diagnostics) &&
    all(vapply(
      dcov_parity$rows$diagnostics,
      function(value) is.list(value) && length(value) > 0L,
      logical(1L)
    )),
  "dCov parity must retain native diagnostics without fallbacks"
)
assert_true(
  all(vapply(dcov_parity$rows$diagnostics, function(diagnostics) {
    identical(diagnostics$lowrank_mode, "spectra") &&
      identical(as.integer(diagnostics$lowrank_full_eig_count), 0L) &&
      identical(as.integer(diagnostics$lowrank_spectra_count), 2L) &&
      identical(
        as.integer(diagnostics$lowrank_spectra_converged_count), 2L
      ) &&
      identical(
        as.integer(diagnostics$lowrank_spectra_failed_count), 0L
      ) &&
      identical(
        as.integer(
          diagnostics$lowrank_spectra_fallback_full_eig_count
        ),
        0L
      ) &&
      as.integer(diagnostics$lowrank_spectra_nconv) >= 2L * 35L
  }, logical(1L))),
  "all dCov rows must prove the exact no-fallback Spectra route"
)

missing_residuals <- list2env(
  as.list(target_parity$residuals, all.names = TRUE),
  envir = new.env(hash = TRUE, parent = emptyenv())
)
missing_key <- iteration$logical_tests$residual_key_x[[1L]]
rm(list = missing_key, envir = missing_residuals)
Sys.setenv(FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK =
             "prepared-s-missing-sentinel")
assert_error(
  fastkpc_full_cuda_run_prepared_s_dcov_parity(
    inputs = inputs,
    logical_tests = iteration$logical_tests,
    residuals = missing_residuals
  ),
  "dCov parity residual is missing",
  "dCov parity must fail closed when an endpoint residual is missing"
)
assert_identical(
  Sys.getenv(dcov_env_name, unset = NA_character_),
  "prepared-s-missing-sentinel",
  "dCov parity must restore the environment after endpoint failure"
)
if (is.na(original_dcov_env)) {
  Sys.unsetenv(dcov_env_name)
} else {
  Sys.setenv(FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = original_dcov_env)
}

runner_output_dir <- tempfile("prepared-s-scaled-runner-")
first_runner <- run_prepared_s_contract_runner(
  runner_path = prepared_s_runner_path,
  census_dir = census_dir,
  data_path = data_path,
  output_dir = runner_output_dir
)
assert_true(
  first_runner$status == 0L,
  paste0(
    "first scaled Prepared-S runner failed: ",
    paste(first_runner$output, collapse = "\n")
  )
)

artifact_paths <- fastkpc_full_cuda_prepared_s_artifact_paths(
  runner_output_dir
)
expected_artifact_path_names <- c(
  "manifest_json", "summary_json", "summary_md", "commands_txt",
  "environment_txt", "input_hashes_csv",
  "prepared_s_setup_index_rds", "prepared_s_setup_index_csv",
  "target_state_index_rds", "target_state_index_csv",
  "iteration_setup_groups_rds", "iteration_setup_groups_csv",
  "iteration_target_keys_rds", "iteration_target_keys_csv",
  "iteration_logical_tests_rds", "iteration_logical_tests_csv",
  "iteration_coverage_csv", "qualification_setup_groups_rds",
  "qualification_setup_groups_csv", "qualification_target_keys_rds",
  "qualification_target_keys_csv", "qualification_logical_tests_rds",
  "qualification_logical_tests_csv", "qualification_coverage_csv",
  "setup_semantic_parity_csv", "target_retarget_parity_csv",
  "dcov_parity_csv", "unsupported_envelope_csv", "fallbacks_csv",
  "stage_timing_csv", "shards_dir"
)
assert_identical(
  names(artifact_paths),
  expected_artifact_path_names,
  "Prepared-S standard artifact path schema"
)
assert_true(
  all(file.exists(unlist(artifact_paths, use.names = FALSE))),
  "every standard Prepared-S artifact path must exist"
)
expected_shard_files <- sort(c(
  paste0("shard_", 0:1, ".rds"),
  paste0("shard_", 0:1, ".summary.json")
), method = "radix")
assert_identical(
  sort(list.files(artifact_paths$shards_dir), method = "radix"),
  expected_shard_files,
  "scaled Prepared-S runner must publish exactly two complete shards"
)

first_summary <- jsonlite::read_json(
  artifact_paths$summary_json, simplifyVector = TRUE
)
first_manifest <- jsonlite::read_json(
  artifact_paths$manifest_json, simplifyVector = TRUE
)
required_summary_fields <- c(
  "pass", "phase2_complete", "run_scope", "parity_scope",
  "phase2_input_authenticated", "selected_group_count",
  "target_state_count", "prepared_s_key_corpus_exact",
  "target_key_corpus_exact", "setup_lineage_exact",
  "target_lineage_exact", "response_leakage_count",
  "prepared_setup_fingerprint_collision_count",
  "target_state_fingerprint_collision_count",
  "unsupported_selected_setup_count",
  "unsupported_canonical_setup_count",
  "unsupported_canonical_evaluated", "unsupported_scope",
  "unknown_fallback_count", "approximate_backend_count",
  "iteration_setup_group_count",
  "iteration_target_key_count", "iteration_logical_test_count",
  "seed_target_key_count", "qualification_target_key_count",
  "qualification_logical_test_count",
  "qualification_same_S_group_count",
  "conditional_near_alpha_test_count",
  "fixed_sp_coefficient_hash_exact_count",
  "fixed_sp_fitted_hash_exact_count",
  "fixed_sp_residual_hash_exact_count",
  "fixed_sp_coefficient_hash_exact", "fixed_sp_fitted_hash_exact",
  "fixed_sp_residual_hash_exact",
  "legacy_dcov_max_abs_p_value_diff",
  "legacy_dcov_p_value_exact_count",
  "legacy_dcov_decision_flip_count",
  "written_shard_count", "reused_shard_count",
  "executed_group_count", "reused_group_count",
  "artifact_payload_size_bytes", "stage_timing_total_seconds",
  "elapsed_seconds", "max_rss_kb", "oracle_inherited_graph_gate",
  "new_candidate_graph_gate"
)
assert_true(
  all(required_summary_fields %in% names(first_summary)),
  "Prepared-S summary must expose every required gate and metric"
)
assert_true(
  isTRUE(first_summary$pass) && !isTRUE(first_summary$phase2_complete) &&
    identical(as.character(first_summary$run_scope), "scaled_iteration") &&
    identical(as.character(first_summary$parity_scope), "iteration") &&
    isTRUE(first_summary$phase2_input_authenticated) &&
    as.integer(first_summary$selected_group_count) == 64L &&
    isTRUE(first_summary$prepared_s_key_corpus_exact) &&
    isTRUE(first_summary$target_key_corpus_exact) &&
    isTRUE(first_summary$setup_lineage_exact) &&
    isTRUE(first_summary$target_lineage_exact),
  "first scaled Prepared-S summary scope and lineage gates"
)

canonical_setup_index <-
  fastkpc_full_cuda_prepared_s_setup_index(inputs)
iteration_group_ids <- as.character(
  iteration$setup_groups$same_S_group_id
)
iteration_setup_index <- match(
  iteration_group_ids, canonical_setup_index$same_S_group_id
)
assert_true(
  !anyNA(iteration_setup_index),
  "iteration groups must resolve to the canonical PreparedSKey index"
)
iteration_prepared_keys <- as.character(
  canonical_setup_index$prepared_s_key_sha256[iteration_setup_index]
)
remaining_setup_index <- canonical_setup_index[
  !canonical_setup_index$same_S_group_id %in% iteration_group_ids,
  , drop = FALSE
]
remaining_setup_index <- remaining_setup_index[order(
  remaining_setup_index$prepared_s_key_sha256, method = "radix"
), , drop = FALSE]
expected_selected_keys <- sort(c(
  iteration_prepared_keys,
  head(as.character(remaining_setup_index$prepared_s_key_sha256), 20L)
), method = "radix")
expected_selected_index <- canonical_setup_index[match(
  expected_selected_keys,
  canonical_setup_index$prepared_s_key_sha256
), , drop = FALSE]
rownames(expected_selected_index) <- NULL
expected_selected_targets <-
  fastkpc_full_cuda_prepared_s_target_rows_for_setups(
    inputs, expected_selected_index
  )

prepared_s_setup_index <- readRDS(
  artifact_paths$prepared_s_setup_index_rds
)
target_state_index <- readRDS(artifact_paths$target_state_index_rds)
assert_true(
  is.list(prepared_s_setup_index) &&
    identical(names(prepared_s_setup_index), expected_selected_keys) &&
    all(iteration_prepared_keys %in% names(prepared_s_setup_index)),
  paste(
    "scaled setup corpus must contain all 44 iteration groups and the",
    "next 20 PreparedSKeys in radix order"
  )
)
assert_true(
  is.data.frame(target_state_index) &&
    nrow(target_state_index) == nrow(expected_selected_targets) &&
    identical(
      as.character(target_state_index$residual_key_sha256),
      as.character(expected_selected_targets$residual_key_sha256)
    ) &&
    identical(
      as.character(target_state_index$prepared_s_key_sha256),
      as.character(expected_selected_targets$prepared_s_key_sha256)
    ) &&
    as.integer(first_summary$target_state_count) ==
      nrow(expected_selected_targets),
  "scaled TargetState index must retain exact selected target lineage"
)
assert_identical(
  as.character(first_manifest$canonical_logical_census_hash),
  as.character(inputs$manifest$canonical_logical_census_hash),
  "artifact manifest must retain full canonical logical lineage"
)
assert_identical(
  as.character(first_manifest$canonical_target_key_corpus_hash),
  as.character(inputs$manifest$canonical_key_corpus_hash),
  "artifact manifest must retain full canonical target lineage"
)
assert_identical(
  as.character(first_manifest$full_canonical_target_key_corpus_hash),
  as.character(inputs$manifest$canonical_key_corpus_hash),
  "artifact manifest must bind the authenticated full target corpus"
)
assert_identical(
  as.character(first_manifest$full_canonical_prepared_s_key_corpus_hash),
  expected_full_prepared_hash,
  "artifact manifest must bind the authenticated full PreparedSKey corpus"
)
assert_identical(
  as.character(first_manifest$selected_prepared_s_key_corpus_hash),
  fastkpc_full_cuda_census_key_set_hash(expected_selected_keys),
  "artifact manifest must authenticate the scaled PreparedSKey corpus"
)
assert_identical(
  as.character(first_manifest$selected_target_key_corpus_hash),
  fastkpc_full_cuda_census_key_set_hash(sort(
    as.character(expected_selected_targets$residual_key_sha256),
    method = "radix"
  )),
  "artifact manifest must authenticate the scaled target corpus"
)

assert_identical(
  readRDS(artifact_paths$iteration_setup_groups_rds),
  iteration$setup_groups,
  "artifact iteration setup selection must be exact"
)
assert_identical(
  readRDS(artifact_paths$iteration_target_keys_rds),
  iteration$target_keys,
  "artifact iteration target selection must be exact"
)
assert_identical(
  readRDS(artifact_paths$iteration_logical_tests_rds),
  iteration$logical_tests,
  "artifact iteration logical selection must be exact"
)
assert_identical(
  readRDS(artifact_paths$qualification_setup_groups_rds),
  qualification$setup_groups,
  "artifact qualification setup selection must be exact"
)
assert_identical(
  readRDS(artifact_paths$qualification_target_keys_rds),
  qualification$target_keys,
  "artifact qualification target selection must be exact"
)
assert_identical(
  readRDS(artifact_paths$qualification_logical_tests_rds),
  qualification$logical_tests,
  "artifact qualification logical selection must be exact"
)

setup_semantic_parity <- utils::read.csv(
  artifact_paths$setup_semantic_parity_csv,
  stringsAsFactors = FALSE,
  check.names = FALSE
)
target_retarget_parity <- utils::read.csv(
  artifact_paths$target_retarget_parity_csv,
  stringsAsFactors = FALSE,
  check.names = FALSE
)
dcov_artifact_parity <- utils::read.csv(
  artifact_paths$dcov_parity_csv,
  stringsAsFactors = FALSE,
  check.names = FALSE
)
assert_true(
  nrow(setup_semantic_parity) == 44L &&
    all(setup_semantic_parity$validator_pass) &&
    all(setup_semantic_parity$semantic_fingerprint_exact) &&
    !any(setup_semantic_parity$independent_reference_comparison) &&
    all(
      setup_semantic_parity$independent_reference_scope ==
        "focused_task4_only"
    ) &&
    all(
      setup_semantic_parity$semantic_evidence_provenance ==
        "phase2_validator_and_authenticated_fingerprints"
    ),
  paste(
    "setup semantic evidence must be complete and honest about the",
    "focused-only independent comparator"
  )
)
assert_true(
  nrow(target_retarget_parity) == 270L &&
    all(target_retarget_parity$coefficient_hash_exact) &&
    all(target_retarget_parity$fitted_hash_exact) &&
    all(target_retarget_parity$residual_hash_exact),
  "iteration target retarget parity must be complete and exact"
)
assert_true(
  nrow(dcov_artifact_parity) == 44L &&
    max(dcov_artifact_parity$absolute_p_value_drift) == 0 &&
    all(dcov_artifact_parity$p_value_exact) &&
    all(dcov_artifact_parity$decision_identical) &&
    all(dcov_artifact_parity$spectra_no_fallback),
  "iteration dCov parity must contain 44 exact fallback-free rows"
)
assert_true(
  as.integer(first_summary$iteration_setup_group_count) == 44L &&
    as.integer(first_summary$iteration_target_key_count) == 270L &&
    as.integer(first_summary$iteration_logical_test_count) == 44L &&
    as.integer(first_summary$seed_target_key_count) == 2356L &&
    as.integer(first_summary$qualification_target_key_count) == 6143L &&
    as.integer(first_summary$qualification_logical_test_count) == 3808L &&
    as.integer(first_summary$qualification_same_S_group_count) == 2061L &&
    as.integer(first_summary$conditional_near_alpha_test_count) == 1478L &&
    as.integer(first_summary$fixed_sp_coefficient_hash_exact_count) == 270L &&
    as.integer(first_summary$fixed_sp_fitted_hash_exact_count) == 270L &&
    as.integer(first_summary$fixed_sp_residual_hash_exact_count) == 270L &&
    isTRUE(first_summary$fixed_sp_coefficient_hash_exact) &&
    isTRUE(first_summary$fixed_sp_fitted_hash_exact) &&
    isTRUE(first_summary$fixed_sp_residual_hash_exact) &&
    as.numeric(first_summary$legacy_dcov_max_abs_p_value_diff) == 0 &&
    as.integer(first_summary$legacy_dcov_p_value_exact_count) == 44L &&
    as.integer(first_summary$legacy_dcov_decision_flip_count) == 0L,
  "scaled artifact parity counts and fixed-sp/dCov gates"
)
assert_true(
  isTRUE(first_summary$oracle_inherited_graph_gate) &&
    identical(
      as.character(first_summary$new_candidate_graph_gate),
      "NOT_APPLICABLE"
    ) &&
    as.integer(first_summary$response_leakage_count) == 0L &&
    as.integer(
      first_summary$prepared_setup_fingerprint_collision_count
    ) == 0L &&
    as.integer(
      first_summary$target_state_fingerprint_collision_count
    ) == 0L &&
    as.integer(first_summary$unsupported_selected_setup_count) == 0L &&
    is.null(first_summary$unsupported_canonical_setup_count) &&
    !isTRUE(first_summary$unsupported_canonical_evaluated) &&
    identical(as.character(first_summary$unsupported_scope), "selected") &&
    as.integer(first_summary$unknown_fallback_count) == 0L &&
    as.integer(first_summary$approximate_backend_count) == 0L &&
    as.integer(first_summary$written_shard_count) == 2L &&
    as.integer(first_summary$reused_shard_count) == 0L &&
    as.integer(first_summary$executed_group_count) == 64L &&
    as.integer(first_summary$reused_group_count) == 0L &&
    as.numeric(first_summary$artifact_payload_size_bytes) > 0 &&
    as.numeric(first_summary$stage_timing_total_seconds) > 0 &&
    is.finite(as.numeric(first_summary$elapsed_seconds)) &&
    as.numeric(first_summary$elapsed_seconds) > 0 &&
    is.finite(as.numeric(first_summary$max_rss_kb)) &&
    as.numeric(first_summary$max_rss_kb) >= 0,
  "scaled artifact structural, shard, size, and inherited graph gates"
)

target_csv_header <- names(utils::read.csv(
  artifact_paths$target_state_index_csv,
  stringsAsFactors = FALSE,
  check.names = FALSE,
  nrows = 1L
))
assert_true(
  !any(c("projected_rhs", "nullspace_projected_rhs") %in%
         target_csv_header) &&
    all(c(
      "projected_rhs_length", "projected_rhs_sha256",
      "nullspace_projected_rhs_length",
      "nullspace_projected_rhs_sha256"
    ) %in% target_csv_header),
  "TargetState CSV must replace numeric RHS vectors with lengths and hashes"
)
input_hashes_artifact <- utils::read.csv(
  artifact_paths$input_hashes_csv,
  stringsAsFactors = FALSE,
  check.names = FALSE
)
assert_identical(
  as.character(input_hashes_artifact$logical_path),
  as.character(inputs$input_hashes$logical_path),
  "artifact input hashes must retain canonical logical path order"
)
assert_identical(
  as.character(input_hashes_artifact$actual_sha256),
  as.character(inputs$input_hashes$actual_sha256),
  "artifact input hashes must retain authenticated bytes"
)
assert_true(
  nrow(utils::read.csv(
    artifact_paths$unsupported_envelope_csv,
    stringsAsFactors = FALSE
  )) == 0L &&
    nrow(utils::read.csv(
      artifact_paths$fallbacks_csv,
      stringsAsFactors = FALSE
    )) == 0L,
  "scaled success must publish zero unsupported/fallback rows"
)
expected_commands <- c(
  paste0("FASTKPC_FULL_CUDA_PREPARED_S_CENSUS_DIR=", census_dir),
  paste0("FASTKPC_FULL_CUDA_PREPARED_S_DATA_PATH=", data_path),
  paste0(
    "FASTKPC_FULL_CUDA_PREPARED_S_OUTPUT_DIR=", runner_output_dir
  ),
  "FASTKPC_FULL_CUDA_PREPARED_S_MAX_GROUPS=64",
  "FASTKPC_FULL_CUDA_PREPARED_S_PARITY_SCOPE=iteration",
  "FASTKPC_FULL_CUDA_PREPARED_S_WORKERS=1",
  "FASTKPC_FULL_CUDA_PREPARED_S_SHARD_COUNT=2",
  "FASTKPC_FULL_CUDA_PREPARED_S_RESUME=1",
  "Rscript fastkpc/tools/run_full_cuda_ci_prepared_s_contract.R"
)
assert_identical(
  readLines(artifact_paths$commands_txt, warn = FALSE),
  expected_commands,
  "Prepared-S artifact commands must exactly reproduce orchestration"
)
environment_lines <- readLines(
  artifact_paths$environment_txt, warn = FALSE
)
assert_true(
  all(c(
    paste0("R_version=", inputs$manifest$R_version),
    paste0("mgcv_version=", inputs$manifest$mgcv_version),
    "requested_workers=1", "actual_workers=1", "shard_count=2",
    "resume=1", "parity_scope=iteration", "max_groups=64"
  ) %in% environment_lines),
  "Prepared-S environment evidence must retain exact runtime controls"
)

backup_dir <- tempfile("prepared-s-scaled-backup-")
dir.create(backup_dir, recursive = TRUE)
deterministic_files <- c(
  prepared_s_setup_index = artifact_paths$prepared_s_setup_index_rds,
  target_state_index = artifact_paths$target_state_index_rds,
  shard_0 = file.path(artifact_paths$shards_dir, "shard_0.rds"),
  shard_1 = file.path(artifact_paths$shards_dir, "shard_1.rds")
)
backup_files <- file.path(
  backup_dir, paste0(names(deterministic_files), ".rds")
)
assert_true(
  all(file.copy(
    unname(deterministic_files), backup_files, overwrite = TRUE
  )),
  "test must preserve first-run deterministic RDS bytes"
)
first_deterministic_hashes <- vapply(
  deterministic_files,
  fastkpc_full_cuda_census_file_hash,
  character(1L)
)

second_runner <- run_prepared_s_contract_runner(
  runner_path = prepared_s_runner_path,
  census_dir = census_dir,
  data_path = data_path,
  output_dir = runner_output_dir
)
assert_true(
  second_runner$status == 0L,
  paste0(
    "second scaled Prepared-S runner failed: ",
    paste(second_runner$output, collapse = "\n")
  )
)
second_summary <- jsonlite::read_json(
  artifact_paths$summary_json, simplifyVector = TRUE
)
assert_true(
  isTRUE(second_summary$pass) &&
    !isTRUE(second_summary$phase2_complete) &&
    identical(as.character(second_summary$run_scope),
              "scaled_iteration") &&
    as.integer(second_summary$executed_group_count) == 0L &&
    as.integer(second_summary$reused_group_count) == 64L &&
    as.integer(second_summary$written_shard_count) == 0L &&
    as.integer(second_summary$reused_shard_count) == 2L,
  "second scaled run must be a complete two-shard resume"
)
second_deterministic_hashes <- vapply(
  deterministic_files,
  fastkpc_full_cuda_census_file_hash,
  character(1L)
)
assert_identical(
  second_deterministic_hashes,
  first_deterministic_hashes,
  "resume must retain merged and shard RDS hashes"
)
for (index in seq_along(deterministic_files)) {
  assert_file_bytes_identical(
    deterministic_files[[index]], backup_files[[index]],
    paste0(
      "resume must retain byte-identical ",
      names(deterministic_files)[[index]]
    )
  )
}

completed_summary_hash <- fastkpc_full_cuda_census_file_hash(
  artifact_paths$summary_json
)
completed_manifest_hash <- fastkpc_full_cuda_census_file_hash(
  artifact_paths$manifest_json
)
failed_runner <- run_prepared_s_contract_runner(
  runner_path = prepared_s_runner_path,
  census_dir = census_dir,
  data_path = data_path,
  output_dir = runner_output_dir,
  max_groups = "64.0"
)
assert_true(
  failed_runner$status != 0L,
  "runner must reject a non-bare integer environment value"
)
assert_identical(
  fastkpc_full_cuda_census_file_hash(artifact_paths$summary_json),
  completed_summary_hash,
  "failed rerun must preserve the prior completed summary"
)
assert_identical(
  fastkpc_full_cuda_census_file_hash(artifact_paths$manifest_json),
  completed_manifest_hash,
  "failed rerun must preserve the prior completed manifest"
)
failed_summary_paths <- list.files(
  file.path(runner_output_dir, "failed_runs"),
  pattern = "^failed_[0-9]+(_[0-9]+)?\\.summary\\.json$",
  full.names = TRUE
)
assert_true(
  length(failed_summary_paths) == 1L,
  "failed rerun beside a completed artifact must publish one failed-run summary"
)
failed_summary <- jsonlite::read_json(
  failed_summary_paths[[1L]], simplifyVector = TRUE
)
assert_true(
  !isTRUE(failed_summary$pass) &&
    !isTRUE(failed_summary$phase2_complete) &&
    identical(as.character(failed_summary$stage), "parse_environment") &&
    nzchar(as.character(failed_summary$error_class)) &&
    grepl(
      "FASTKPC_FULL_CUDA_PREPARED_S_MAX_GROUPS",
      as.character(failed_summary$error_message), fixed = TRUE
    ) &&
    is.finite(as.numeric(failed_summary$elapsed_seconds)),
  "failed-run summary must retain stage, class, message, and elapsed time"
)

unlink(backup_dir, recursive = TRUE, force = TRUE)
unlink(runner_output_dir, recursive = TRUE, force = TRUE)

cat(
  "METRICS iteration_hash=", iteration$iteration_subset_hash,
  " setup_groups=", nrow(iteration$setup_groups),
  " target_keys=", nrow(iteration$target_keys),
  " logical_tests=", nrow(iteration$logical_tests),
  " qualification_seed=", nrow(qualification$seed_target_keys),
  " qualification_targets=", nrow(qualification$target_keys),
  " qualification_logical=", nrow(qualification$logical_tests),
  " qualification_groups=", nrow(qualification$setup_groups),
  " qualification_hash=", qualification$qualification_subset_hash,
  " target_exact=", sum(
    target_parity$rows$coefficient_hash_exact &
      target_parity$rows$fitted_hash_exact &
      target_parity$rows$residual_hash_exact
  ),
  " dcov_max_drift=", format(
    max(dcov_parity$rows$absolute_p_value_drift), scientific = TRUE
  ),
  " dcov_exact=", sum(dcov_parity$rows$p_value_exact),
  " dcov_flips=", dcov_parity$decision_flip_count,
  " runner_first_elapsed=", first_summary$elapsed_seconds,
  " runner_second_elapsed=", second_summary$elapsed_seconds,
  " runner_first_max_rss_kb=", first_summary$max_rss_kb,
  "\n",
  sep = ""
)
cat("PASS Prepared-S iteration target and dCov parity\n")
