fastkpc_full_cuda_phase8_require <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
  invisible(TRUE)
}

fastkpc_full_cuda_phase8_artifact_root <- function() {
  file.path("fastkpc", "artifacts", "full_cuda_ci")
}

fastkpc_full_cuda_phase8_phase7_artifact <- function() {
  file.path(
    fastkpc_full_cuda_phase8_artifact_root(), "native_setup_backend_v1"
  )
}

fastkpc_full_cuda_phase8_source_paths <- function() {
  c(
    "fastkpc/R/full_cuda_ci_phase8_dcov.R",
    "fastkpc/R/full_cuda_ci_phase35_bakeoff.R",
    "fastkpc/R/full_cuda_ci_phase35_contracts.R",
    "fastkpc/R/full_cuda_ci_native_setup.R",
    "fastkpc/R/full_cuda_ci_fixed_sp_runtime.R",
    "fastkpc/R/cuda_native.R",
    "fastkpc/src/cuda/full_cuda_ci_vertical.cu",
    "fastkpc/src/cuda/full_cuda_ci_vertical.hpp",
    "fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu",
    "fastkpc/src/cuda/mgcv_fixed_sp_runtime.hpp",
    "fastkpc/src/full_cuda_ci_native_setup.cpp",
    "fastkpc/src/full_cuda_ci_native_setup.hpp",
    "fastkpc/src/r_api_cuda.cpp",
    "fastkpc/tools/build_cuda_native.sh",
    "fastkpc/tools/run_full_cuda_ci_phase8_dcov.R"
  )
}

fastkpc_full_cuda_phase8_source_closure <- function() {
  paths <- fastkpc_full_cuda_phase8_source_paths()
  fastkpc_full_cuda_phase8_require(
    all(file.exists(paths)) && !anyDuplicated(paths),
    "Phase 8 source closure is incomplete"
  )
  hashes <- setNames(as.list(vapply(
    paths, fastkpc_full_cuda_census_file_hash, character(1L)
  )), paths)
  list(
    paths = paths,
    hashes = hashes,
    sha256 = fastkpc_full_cuda_phase35_sha256_utf8(
      fastkpc_full_cuda_phase35_canonical_json(hashes)
    )
  )
}

fastkpc_full_cuda_phase8_execution_identity <- function(inputs) {
  closure <- fastkpc_full_cuda_phase8_source_closure()
  native <- fastkpc_full_cuda_phase7_native_identity()
  value <- list(
    schema_version = "full-cuda-ci-phase8-execution-identity-v1",
    source_commit = fastkpc_full_cuda_command_output(
      "git", c("rev-parse", "HEAD")
    ),
    source_closure_sha256 = closure$sha256,
    native_binary_sha256 = native$sha256,
    phase7_evidence_sha256 = inputs$phase7_evidence_sha256,
    dataset_sha256 = inputs$catalog$inputs$dataset_sha256,
    route_semantic_version =
      "guarded-exact-screen-legacy-full-eig-cuda-v1",
    guard_lower_inclusive = 0.05,
    guard_upper_inclusive = 0.15,
    alpha = 0.1,
    index = 1L,
    num_col = 35L
  )
  list(
    value = value,
    identity_sha256 = fastkpc_full_cuda_census_named_metadata_hash(value),
    source_closure = closure,
    native = native
  )
}

fastkpc_full_cuda_phase8_selected_sp <- function(phase7_evidence) {
  fastkpc_full_cuda_phase8_require(
    is.list(phase7_evidence) &&
      identical(
        phase7_evidence$schema_version,
        "full-cuda-ci-native-setup-full-evidence-v1"
      ) && isTRUE(phase7_evidence$summary$pass),
    "Phase 8 requires accepted Phase 7 evidence"
  )
  single <- phase7_evidence$phase4$targets
  multi <- phase7_evidence$phase6$targets
  fastkpc_full_cuda_phase8_require(
    is.data.frame(single) && nrow(single) == 44941L &&
      is.data.frame(multi) && nrow(multi) == 65676L &&
      !anyDuplicated(c(single$residual_key_sha256,
                       multi$residual_key_sha256)) &&
      all(is.finite(single$candidate_sp)) && all(single$candidate_sp > 0),
    "Phase 8 Phase 7 selected-sp corpus is malformed"
  )

  parse_log_sp <- function(value, penalty_count) {
    pieces <- strsplit(value, ",", fixed = TRUE)[[1L]]
    parsed <- suppressWarnings(as.numeric(trimws(pieces)))
    fastkpc_full_cuda_phase8_require(
      length(parsed) == penalty_count && all(is.finite(parsed)),
      "Phase 8 multi-penalty selected log-sp is malformed"
    )
    selected <- exp(parsed)
    fastkpc_full_cuda_phase8_require(
      all(is.finite(selected)) && all(selected > 0),
      "Phase 8 multi-penalty selected sp is non-finite"
    )
    unname(selected)
  }

  single_values <- lapply(single$candidate_sp, function(value) {
    unname(as.numeric(value))
  })
  multi_values <- Map(
    parse_log_sp, as.character(multi$selected_log_sp),
    as.integer(multi$penalty_count)
  )
  values <- c(single_values, multi_values)
  names(values) <- c(
    as.character(single$residual_key_sha256),
    as.character(multi$residual_key_sha256)
  )
  fastkpc_full_cuda_phase8_require(
    length(values) == 110617L && !anyNA(names(values)) &&
      !anyDuplicated(names(values)),
    "Phase 8 selected-sp key map is incomplete"
  )
  values
}

fastkpc_full_cuda_phase8_load_inputs <- function(
    validate_phase7_artifact = TRUE,
    validate_phase7_evidence = validate_phase7_artifact) {
  corpus <- fastkpc_full_cuda_phase35_load_bakeoff_corpus(
    include_canonical_logical = TRUE
  )
  catalog <- fastkpc_full_cuda_open_fixed_sp_catalog(
    file.path(
      "fastkpc", "artifacts", "full_cuda_ci", "oracle_351x48_v1"
    ),
    file.path(
      "fastkpc", "artifacts", "full_cuda_ci",
      "workload_census_351x48_v1"
    ),
    file.path(
      "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1"
    ),
    file.path(
      "fastkpc", "artifacts", "kpc_tprs_real_zhu",
      "cancer_RD-causalDiscoveryInput.rds"
    ),
    require_full = TRUE
  )
  phase7_dir <- fastkpc_full_cuda_phase8_phase7_artifact()
  phase7_path <- file.path(phase7_dir, "source_evidence.rds")
  fastkpc_full_cuda_phase8_require(
    file.exists(phase7_path) && !dir.exists(phase7_path),
    "Phase 8 accepted Phase 7 source evidence is missing"
  )
  phase7_validation <- if (isTRUE(validate_phase7_artifact)) {
    fastkpc_full_cuda_phase7_validate_artifact(
      phase7_dir, expected_kind = "backend", catalog = catalog,
      verify_current_sources = FALSE
    )
  } else {
    NULL
  }
  phase7 <- if (is.null(phase7_validation)) {
    readRDS(phase7_path)
  } else {
    phase7_validation$evidence
  }
  if (isTRUE(validate_phase7_evidence)) {
    fastkpc_full_cuda_phase7_validate_full_evidence(
      phase7, catalog, verify_current_identity = FALSE
    )
  } else {
    fastkpc_full_cuda_phase8_require(
      is.list(phase7) &&
        identical(
          phase7$schema_version,
          "full-cuda-ci-native-setup-full-evidence-v1"
        ) && isTRUE(phase7$summary$pass) &&
        phase7$summary$setup_count == 8634L &&
        phase7$summary$target_count == 110617L &&
        phase7$summary$logical_test_count == 240489L &&
        phase7$summary$SHD == 0L &&
        phase7$summary$legacy_mgcv_setup_count == 0L &&
        phase7$summary$legacy_mgcv_fit_count == 0L &&
        phase7$summary$r_callback_count == 0L,
      "Phase 8 lightweight Phase 7 evidence gate failed"
    )
  }
  setup_scope <- fastkpc_full_cuda_phase7_setup_scope(catalog)$setup_rows
  setup_index <- match(
    names(corpus$prepared_setups), setup_scope$prepared_s_key_sha256
  )
  fastkpc_full_cuda_phase8_require(
    identical(corpus$data, catalog$inputs$data) &&
      nrow(setup_scope) == 8634L && !anyNA(setup_index) &&
      !anyDuplicated(setup_index),
    "Phase 8 Phase 7/canonical corpus linkage failed"
  )
  list(
    schema_version = "full-cuda-ci-phase8-inputs-v1",
    corpus = corpus,
    catalog = catalog,
    phase7 = phase7,
    phase7_artifact_dir = phase7_dir,
    phase7_evidence_sha256 =
      fastkpc_full_cuda_census_file_hash(phase7_path),
    selected_sp = fastkpc_full_cuda_phase8_selected_sp(phase7),
    setup_rows = setup_scope,
    setup_row_index = setNames(
      seq_len(nrow(setup_scope)),
      as.character(setup_scope$prepared_s_key_sha256)
    )
  )
}

fastkpc_full_cuda_phase8_direct_context <- function(inputs, logical_rows) {
  data <- inputs$corpus$data
  dataset_sha256 <- inputs$catalog$inputs$dataset_sha256
  prepared_key <- fastkpc_full_cuda_phase35_sha256_utf8(paste0(
    "full-cuda-ci-phase8-direct-shift-adapter-v1\n", dataset_sha256
  ))
  target_keys <- vapply(seq_len(ncol(data)), function(index) {
    fastkpc_full_cuda_phase35_sha256_utf8(paste0(
      "full-cuda-ci-phase8-direct-target-v1\n", dataset_sha256,
      "\ncolumn=", index
    ))
  }, character(1L))
  names(target_keys) <- NULL
  direct <- as.data.frame(logical_rows, stringsAsFactors = FALSE)
  direct$residual_key_x <- target_keys[direct$x]
  direct$residual_key_y <- target_keys[direct$y]
  direct$near_alpha <-
    abs(direct$signed_log_ratio_from_alpha) <= log(2)
  contract <- fastkpc_full_cuda_fixed_sp_contract()
  identity <- function(label) {
    fastkpc_full_cuda_phase35_sha256_utf8(paste0(
      "full-cuda-ci-phase8-direct-", label, "-v1\n", dataset_sha256
    ))
  }
  X <- matrix(1, nrow(data), 1L)
  storage.mode(X) <- "double"
  gram <- matrix(as.numeric(nrow(data)), 1L, 1L)
  penalty <- matrix(1, 1L, 1L)
  penalty_blocks <- list(penalty_1 = penalty)
  dto <- list(
    schema_version = contract$native_dto_schema_version,
    dataset_sha256 = dataset_sha256,
    prepared_s_key_sha256 = prepared_key,
    same_S_group_id = identity("group"),
    phase1_setup_fingerprint = identity("phase1-setup"),
    provider_fingerprint = identity("provider"),
    semantic_fingerprint = identity("semantic"),
    representation_fingerprint = identity("representation"),
    prepared_s_setup_schema_version = "full-cuda-ci-prepared-s-setup-v1",
    native_dto_schema_version = contract$native_dto_schema_version,
    data_p = as.integer(ncol(data)),
    n = as.integer(nrow(data)),
    coefficient_dim = 1L,
    null_dim = 1L,
    penalty_count = 1L,
    X = X,
    constraint_mode = "identity",
    constraint_nullspace = NULL,
    gram_matrix = gram,
    nullspace_gram_matrix = NULL,
    penalty_blocks = penalty_blocks,
    penalty_offsets_zero_based = 0L,
    penalty_ranks = 1L,
    penalty_sp_indices_zero_based = 0L,
    penalty_sp_labels = "direct-shift-invariant",
    H = NULL,
    weights_policy = "none-or-unit",
    offset_policy = "none-or-zero"
  )
  fastkpc_full_cuda_phase8_require(
    identical(names(dto), fastkpc_full_cuda_fixed_sp_native_dto_fields()),
    "Phase 8 direct adapter DTO fields drifted"
  )
  list(
    prepared_key = prepared_key,
    target_keys = target_keys,
    logical_rows = direct,
    dto = dto,
    Y = data,
    SP = matrix(0, 1L, ncol(data)),
    planned_route = rep.int("CHOLESKY_BATCHED", ncol(data)),
    context_kind = "direct-shift-invariant"
  )
}

fastkpc_full_cuda_phase8_conditional_context <- function(
    inputs, prepared_key, logical_rows) {
  corpus <- inputs$corpus
  target_keys <- sort(unique(c(
    logical_rows$residual_key_x, logical_rows$residual_key_y
  )), method = "radix")
  state_index <- match(
    target_keys, corpus$target_states$residual_key_sha256
  )
  parity_index <- match(
    target_keys, corpus$target_parity$residual_key_sha256
  )
  setup_row_index <- unname(inputs$setup_row_index[[prepared_key]])
  fastkpc_full_cuda_phase8_require(
    !anyNA(state_index) && !anyNA(parity_index) &&
      length(setup_row_index) == 1L && !is.na(setup_row_index),
    "Phase 8 conditional group lineage is incomplete"
  )
  states <- corpus$target_states[state_index, , drop = FALSE]
  parity <- corpus$target_parity[parity_index, , drop = FALSE]
  setup_row <- inputs$setup_rows[setup_row_index, , drop = FALSE]
  setup <- fastkpc_full_cuda_phase7_runtime_setup(
    catalog = inputs$catalog, setup_row = setup_row,
    target_states = states, setup_key = prepared_key
  )
  selected <- inputs$selected_sp[target_keys]
  fastkpc_full_cuda_phase8_require(
    !anyNA(names(selected)) && all(lengths(selected) ==
      length(setup$penalty_blocks)) &&
      identical(states$residual_key_sha256, target_keys) &&
      identical(parity$residual_key_sha256, target_keys) &&
      identical(parity$planned_route, parity$executed_route) &&
      all(parity$cpu_fallback_count == 0L) &&
      all(parity$unknown_fallback_count == 0L) &&
      all(!parity$approximate_backend),
    "Phase 8 conditional residual authority is malformed"
  )
  SP <- do.call(cbind, selected)
  storage.mode(SP) <- "double"
  list(
    prepared_key = prepared_key,
    target_keys = target_keys,
    logical_rows = logical_rows,
    dto = fastkpc_full_cuda_phase7_fixed_sp_dto(setup),
    Y = corpus$data[, states$target, drop = FALSE],
    SP = SP,
    planned_route = as.character(parity$planned_route),
    context_kind = "phase7-native-selected-sp"
  )
}

fastkpc_full_cuda_phase8_group_diagnostics <- function(
    context, exact, refinement, create_ms, free_ms) {
  exact_d <- exact$diagnostics
  refined <- !is.null(refinement)
  refine_d <- if (refined) refinement$diagnostics else NULL
  value <- function(field, default = 0) {
    if (is.null(refine_d)) default else refine_d[[field]]
  }
  data.frame(
    prepared_s_key_sha256 = context$prepared_key,
    context_kind = context$context_kind,
    level = as.integer(unique(context$logical_rows$level)),
    pair_count = nrow(exact$records),
    screen_component_count = exact_d$referenced_component_count,
    refined_pair_count = if (refined) nrow(refinement$records) else 0L,
    refined_component_count =
      as.integer(value("referenced_component_count", 0L)),
    handle_create_ms = create_ms,
    screen_residual_solve_ms = exact_d$residual_solve_host_ms,
    screen_component_cuda_ms = exact_d$component_build_cuda_ms,
    screen_pair_gamma_cuda_ms = exact_d$pair_evaluation_cuda_ms,
    screen_host_boundary_ms = exact_d$dcov_host_boundary_ms,
    refinement_residual_solve_ms = value("residual_solve_host_ms"),
    refinement_component_cuda_ms = value("component_build_cuda_ms"),
    refinement_pair_gamma_cuda_ms = value("pair_evaluation_cuda_ms"),
    refinement_host_boundary_ms = value("dcov_host_boundary_ms"),
    handle_free_ms = free_ms,
    residual_d2h_bytes = exact_d$residual_d2h_bytes +
      value("residual_d2h_bytes"),
    component_d2h_bytes = exact_d$component_d2h_bytes +
      value("component_d2h_bytes"),
    compact_result_d2h_bytes = exact_d$compact_result_d2h_bytes +
      value("compact_result_d2h_bytes"),
    matrix_h2d_bytes = as.numeric(
      nrow(context$Y) * ncol(context$Y) * 8 * (1L + as.integer(refined)) +
        length(context$SP) * 8 * (1L + as.integer(refined))
    ),
    host_synchronization_count =
      as.integer(exact_d$explicit_host_wait_count) +
        as.integer(value("explicit_host_wait_count", 0L)),
    component_cache_request_count =
      as.integer(exact_d$component_cache_lookup_count) +
        if (refined) 2L * nrow(refinement$records) else 0L,
    component_cache_hit_count =
      as.integer(exact_d$component_cache_hit_count) +
        if (refined) {
          2L * nrow(refinement$records) -
            as.integer(refine_d$referenced_component_count)
        } else 0L,
    component_cache_miss_count =
      as.integer(exact_d$component_cache_miss_count) +
        as.integer(value("referenced_component_count", 0L)),
    component_cache_eviction_count = 0L,
    cpu_dcov_component_count = exact_d$cpu_dcov_component_count +
      value("cpu_dcov_component_count"),
    cpu_dcov_eigen_or_lowrank_count =
      as.integer(value("cpu_dcov_eigen_count", 0L)),
    cpu_dcov_pair_statistic_count =
      exact_d$cpu_dcov_pair_statistic_count +
        value("cpu_dcov_pair_statistic_count"),
    cpu_gamma_p_value_count = exact_d$cpu_gamma_p_value_count +
      value("cpu_gamma_p_value_count"),
    cpu_spectra_count = 0L,
    cuda_exact_component_count = exact_d$component_build_count,
    cuda_full_eig_component_count =
      as.integer(value("cuda_full_eig_count", 0L)),
    cuda_pair_count = exact_d$pair_evaluation_count +
      as.integer(value("cuda_pair_count", 0L)),
    cuda_gamma_count = exact_d$pair_evaluation_count +
      as.integer(value("cuda_gamma_count", 0L)),
    solver_failure_count = as.integer(value("solver_failure_count", 0L)),
    exact_bounded_allocation = isTRUE(exact_d$bounded_allocation),
    refinement_bounded_allocation =
      !refined || isTRUE(refine_d$bounded_allocation),
    leak_free = isTRUE(exact_d$leak_free_teardown) &&
      (!refined || isTRUE(refine_d$leak_free_teardown)),
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_phase8_run_context <- function(
    runtime, context, guard_lower = 0.05, guard_upper = 0.15) {
  fastkpc_full_cuda_phase8_require(
    is.list(context) && nrow(context$logical_rows) > 0L &&
      is.double(guard_lower) && length(guard_lower) == 1L &&
      is.double(guard_upper) && length(guard_upper) == 1L &&
      guard_lower < 0.1 && guard_upper > 0.1,
    "Phase 8 group execution inputs are malformed"
  )
  handle <- NULL
  create_start <- proc.time()[["elapsed"]]
  handle <- fixed_sp_cuda_prepared_create(runtime, context$dto)
  create_ms <- 1000 * (proc.time()[["elapsed"]] - create_start)
  tryCatch({
    exact_request <-
      fastkpc_full_cuda_phase35_exact_batch_request_from_logical(
        context$prepared_key, context$target_keys, context$logical_rows
      )
    exact <- fastkpc_full_cuda_phase35_exact_batch_ci(
      handle, context$Y, context$SP, context$planned_route,
      context$target_keys, exact_request
    )
    fastkpc_full_cuda_phase35_validate_exact_batch_result(
      exact, exact_request, context$target_keys
    )
    screen_p <- as.numeric(exact$records$p_value)
    refined <- screen_p >= guard_lower & screen_p <= guard_upper
    refinement <- NULL
    if (any(refined)) {
      refinement_rows <- context$logical_rows[refined, , drop = FALSE]
      refinement_request <-
        fastkpc_full_cuda_phase35_legacy_eig_batch_request_from_logical(
          context$prepared_key, context$target_keys, refinement_rows
        )
      refinement <- fastkpc_full_cuda_phase35_legacy_eig_batch_ci(
        handle, context$Y, context$SP, context$planned_route,
        context$target_keys, refinement_request
      )
      fastkpc_full_cuda_phase35_validate_legacy_eig_batch_result(
        refinement, refinement_request, context$target_keys
      )
    }
    output_slot_released <- identical(
      fixed_sp_cuda_prepared_info(handle)$output_slot_state, "free"
    )
    free_start <- proc.time()[["elapsed"]]
    fixed_sp_cuda_prepared_free(handle)
    handle <- NULL
    free_ms <- 1000 * (proc.time()[["elapsed"]] - free_start)

    final_p <- screen_p
    final_numerical <- exact$numerical
    if (!is.null(refinement)) {
      final_p[refined] <- refinement$records$p_value
      final_numerical[refined, ] <- refinement$numerical
    }
    logical <- context$logical_rows
    reference_independent <- as.logical(logical$reference_independent)
    screen_independent <- screen_p > logical$alpha
    final_independent <- final_p > logical$alpha
    screen_flip <- screen_independent != reference_independent
    final_flip <- final_independent != reference_independent
    pairs <- data.frame(
      logical_sequence_id = as.integer(logical$logical_sequence_id),
      level = as.integer(logical$level),
      prepared_s_key_sha256 = context$prepared_key,
      x = as.integer(logical$x),
      y = as.integer(logical$y),
      S_key = as.character(logical$S_key),
      residual_key_x = as.character(logical$residual_key_x),
      residual_key_y = as.character(logical$residual_key_y),
      alpha = as.numeric(logical$alpha),
      reference_p_value = as.numeric(logical$reference_p_value),
      screen_p_value = screen_p,
      final_p_value = final_p,
      refined = refined,
      final_backend = ifelse(
        refined, "guarded-legacy-full-eig-cuda",
        "exact-cuda-certified-outside-guard"
      ),
      reference_independent = reference_independent,
      screen_independent = screen_independent,
      final_independent = final_independent,
      screen_decision_flip = screen_flip,
      final_decision_flip = final_flip,
      near_alpha = as.logical(logical$near_alpha),
      deletes_edge = as.logical(logical$deletes_edge),
      final_p_value_absolute_error =
        abs(final_p - as.numeric(logical$reference_p_value)),
      statistic = final_numerical$statistic,
      mean = final_numerical$mean,
      variance = final_numerical$variance,
      gamma_shape = final_numerical$gamma_shape,
      gamma_scale = final_numerical$gamma_scale,
      gamma_iterations = final_numerical$gamma_iterations,
      stringsAsFactors = FALSE
    )
    fastkpc_full_cuda_phase8_require(
      output_slot_released && all(!screen_flip | refined) &&
        !any(final_flip) && all(is.finite(final_p)) &&
        all(is.finite(as.matrix(final_numerical))),
      paste0(
        "Phase 8 guarded group gate failed: ", context$prepared_key,
        "; screen_flips=", sum(screen_flip),
        "; final_flips=", sum(final_flip)
      )
    )
    list(
      pairs = pairs,
      diagnostics = fastkpc_full_cuda_phase8_group_diagnostics(
        context, exact, refinement, create_ms, free_ms
      ),
      exact = exact,
      refinement = refinement
    )
  }, error = function(error) {
    if (!is.null(handle)) {
      try(fixed_sp_cuda_prepared_free(handle), silent = TRUE)
    }
    stop(error)
  })
}

fastkpc_full_cuda_phase8_run_logical_rows <- function(
    inputs, logical_rows, runtime = NULL, progress = interactive(),
    guard_lower = 0.05, guard_upper = 0.15) {
  logical_rows <- as.data.frame(logical_rows, stringsAsFactors = FALSE)
  logical_rows <- logical_rows[order(
    logical_rows$logical_sequence_id, method = "radix"
  ), , drop = FALSE]
  fastkpc_full_cuda_phase8_require(
    nrow(logical_rows) > 0L && !anyDuplicated(logical_rows$logical_sequence_id) &&
      all(logical_rows$alpha == 0.1),
    "Phase 8 logical execution slice is malformed"
  )
  owns_runtime <- is.null(runtime)
  if (owns_runtime) {
    load_fastkpc_cuda_native()
    runtime <- fixed_sp_cuda_runtime_create(0L)
    on.exit({
      if (!is.null(runtime)) {
        try(fixed_sp_cuda_runtime_free(runtime), silent = TRUE)
      }
    }, add = TRUE)
    fixed_sp_cuda_runtime_reserve(
      runtime, 351L, 64L, 48L, 7L, 415L
    )
  }

  pair_chunks <- list()
  diagnostic_chunks <- list()
  ordinal <- 0L
  direct <- logical_rows$level == 0L
  if (any(direct)) {
    ordinal <- ordinal + 1L
    context <- fastkpc_full_cuda_phase8_direct_context(
      inputs, logical_rows[direct, , drop = FALSE]
    )
    result <- fastkpc_full_cuda_phase8_run_context(
      runtime, context, guard_lower, guard_upper
    )
    pair_chunks[[ordinal]] <- result$pairs
    diagnostic_chunks[[ordinal]] <- result$diagnostics
  }

  conditional <- logical_rows[!direct, , drop = FALSE]
  if (nrow(conditional) > 0L) {
    groups <- .fastkpc_full_cuda_phase35_group_logical_rows(
      conditional, inputs$corpus$target_states
    )
    group_keys <- names(groups)
    for (group_ordinal in seq_along(group_keys)) {
      prepared_key <- group_keys[[group_ordinal]]
      rows <- conditional[groups[[prepared_key]], , drop = FALSE]
      context <- fastkpc_full_cuda_phase8_conditional_context(
        inputs, prepared_key, rows
      )
      result <- fastkpc_full_cuda_phase8_run_context(
        runtime, context, guard_lower, guard_upper
      )
      ordinal <- ordinal + 1L
      pair_chunks[[ordinal]] <- result$pairs
      diagnostic_chunks[[ordinal]] <- result$diagnostics
      if (isTRUE(progress) &&
          (group_ordinal == 1L || group_ordinal %% 250L == 0L ||
           group_ordinal == length(group_keys))) {
        cat(
          "Phase 8 groups ", group_ordinal, "/", length(group_keys),
          "; pairs=", sum(vapply(pair_chunks, nrow, integer(1L))), "\n",
          sep = ""
        )
      }
    }
  }
  pairs <- do.call(rbind, pair_chunks)
  diagnostics <- do.call(rbind, diagnostic_chunks)
  rownames(pairs) <- NULL
  rownames(diagnostics) <- NULL
  pairs <- pairs[order(pairs$logical_sequence_id, method = "radix"),
                 , drop = FALSE]
  if (owns_runtime) {
    runtime_info <- fixed_sp_cuda_runtime_info(runtime)
    fixed_sp_cuda_runtime_free(runtime)
    runtime <- NULL
  } else {
    runtime_info <- fixed_sp_cuda_runtime_info(runtime)
  }
  fastkpc_full_cuda_phase8_require(
    identical(pairs$logical_sequence_id,
              as.integer(logical_rows$logical_sequence_id)) &&
      !any(pairs$final_decision_flip) &&
      all(!pairs$screen_decision_flip | pairs$refined) &&
      all(diagnostics$residual_d2h_bytes == 0) &&
      all(diagnostics$component_d2h_bytes == 0) &&
      all(diagnostics$cpu_dcov_component_count == 0L) &&
      all(diagnostics$cpu_dcov_eigen_or_lowrank_count == 0L) &&
      all(diagnostics$cpu_dcov_pair_statistic_count == 0L) &&
      all(diagnostics$cpu_gamma_p_value_count == 0L) &&
      all(diagnostics$cpu_spectra_count == 0L) &&
      all(diagnostics$solver_failure_count == 0L) &&
      all(diagnostics$leak_free),
    "Phase 8 logical execution authority gate failed"
  )
  list(
    schema_version = "full-cuda-ci-phase8-logical-execution-v1",
    pairs = pairs,
    diagnostics = diagnostics,
    runtime_info = runtime_info,
    guard = list(
      lower_inclusive = guard_lower,
      upper_inclusive = guard_upper,
      alpha = 0.1,
      policy = "refine-exact-screen-p-in-closed-interval"
    )
  )
}

fastkpc_full_cuda_phase8_graph <- function(inputs, pairs) {
  authority <- inputs$corpus$canonical_logical
  pair_index <- match(
    authority$logical_sequence_id, pairs$logical_sequence_id
  )
  fastkpc_full_cuda_phase8_require(
    nrow(authority) == 240489L && nrow(pairs) == 240489L &&
      !anyNA(pair_index),
    "Phase 8 full graph logical coverage is incomplete"
  )
  replay <- fastkpc_full_cuda_replay_logical_ci(
    logical_tests = authority,
    candidate_p_value = pairs$final_p_value[pair_index],
    labels = colnames(inputs$corpus$data),
    expected_logical_contract =
      fastkpc_full_cuda_shadow_logical_contract(authority)
  )
  phase0_authority <- .fastkpc_full_cuda_phase3_shadow_phase0_authority(
    inputs$catalog, inputs$catalog$phase0_dir
  )
  phase0 <- .fastkpc_full_cuda_phase3_load_shadow_phase0_oracle(
    inputs$catalog, phase0_authority
  )
  comparison <- fastkpc_full_cuda_compare_candidate_skeleton(
    phase0, replay$skeleton
  )
  graph <- comparison$summary
  pass <- isTRUE(graph$pass) && graph$edge_count_reference == 110L &&
    graph$edge_count_candidate == 110L && graph$SHD == 0L &&
    isTRUE(graph$adjacency_identical) && isTRUE(graph$sepsets_identical) &&
    isTRUE(graph$n_edgetests_identical) &&
    isTRUE(graph$deletions_identical)
  fastkpc_full_cuda_phase8_require(pass, "Phase 8 full graph gate failed")
  list(
    schema_version = "full-cuda-ci-phase8-graph-v1",
    summary = graph,
    replay = replay,
    comparison = comparison,
    pass = pass
  )
}

fastkpc_full_cuda_phase8_run_full <- function(
    inputs = NULL, progress = interactive()) {
  if (is.null(inputs)) {
    inputs <- fastkpc_full_cuda_phase8_load_inputs(
      validate_phase7_artifact = TRUE
    )
  }
  logical <- inputs$corpus$canonical_logical
  logical$near_alpha <-
    abs(logical$signed_log_ratio_from_alpha) <= log(2)
  started <- proc.time()[["elapsed"]]
  execution <- fastkpc_full_cuda_phase8_run_logical_rows(
    inputs, logical, progress = progress
  )
  elapsed <- proc.time()[["elapsed"]] - started
  graph <- fastkpc_full_cuda_phase8_graph(inputs, execution$pairs)
  execution_identity <- fastkpc_full_cuda_phase8_execution_identity(inputs)
  pairs <- execution$pairs
  diagnostics <- execution$diagnostics
  cuda_dcov_host_boundary_ms <-
    sum(diagnostics$screen_host_boundary_ms) +
      sum(diagnostics$refinement_host_boundary_ms)
  legacy_cpu_dcov_ms <-
    sum(inputs$phase7$phase4$timings$dcov_ms) +
      sum(inputs$phase7$phase6$timings$dcov_ms)
  summary <- list(
    schema_version = "full-cuda-ci-phase8-full-summary-v1",
    run_status = "ok",
    logical_test_count = nrow(pairs),
    direct_logical_test_count = sum(pairs$level == 0L),
    conditional_logical_test_count = sum(pairs$level > 0L),
    unique_residual_component_count = sum(
      diagnostics$screen_component_count
    ),
    exact_screen_component_count = sum(
      diagnostics$cuda_exact_component_count
    ),
    guarded_pair_count = sum(pairs$refined),
    refined_component_count = sum(
      diagnostics$cuda_full_eig_component_count
    ),
    screen_decision_flip_count = sum(pairs$screen_decision_flip),
    final_decision_flip_count = sum(pairs$final_decision_flip),
    near_alpha_count = sum(pairs$near_alpha),
    near_alpha_final_decision_flip_count = sum(
      pairs$near_alpha & pairs$final_decision_flip
    ),
    maximum_refined_p_value_absolute_error = max(
      pairs$final_p_value_absolute_error[pairs$refined]
    ),
    screen_component_cuda_ms = sum(
      diagnostics$screen_component_cuda_ms
    ),
    screen_pair_gamma_cuda_ms = sum(
      diagnostics$screen_pair_gamma_cuda_ms
    ),
    refinement_component_cuda_ms = sum(
      diagnostics$refinement_component_cuda_ms
    ),
    refinement_pair_gamma_cuda_ms = sum(
      diagnostics$refinement_pair_gamma_cuda_ms
    ),
    cuda_dcov_host_boundary_ms = cuda_dcov_host_boundary_ms,
    legacy_cpu_spectra_dcov_ms = legacy_cpu_dcov_ms,
    dcov_performance_ratio = cuda_dcov_host_boundary_ms / legacy_cpu_dcov_ms,
    dcov_budget_ms = 47000,
    dcov_budget_pass = cuda_dcov_host_boundary_ms <= 47000,
    dcov_same_machine_speed_pass =
      cuda_dcov_host_boundary_ms < legacy_cpu_dcov_ms,
    matrix_h2d_bytes = sum(diagnostics$matrix_h2d_bytes),
    residual_d2h_bytes = sum(diagnostics$residual_d2h_bytes),
    component_d2h_bytes = sum(diagnostics$component_d2h_bytes),
    host_synchronization_count = sum(
      diagnostics$host_synchronization_count
    ),
    component_cache_request_count = sum(
      diagnostics$component_cache_request_count
    ),
    component_cache_hit_count = sum(
      diagnostics$component_cache_hit_count
    ),
    component_cache_miss_count = sum(
      diagnostics$component_cache_miss_count
    ),
    component_cache_eviction_count = sum(
      diagnostics$component_cache_eviction_count
    ),
    cpu_dcov_component_count = sum(
      diagnostics$cpu_dcov_component_count
    ),
    cpu_dcov_eigen_or_lowrank_count = sum(
      diagnostics$cpu_dcov_eigen_or_lowrank_count
    ),
    cpu_dcov_pair_statistic_count = sum(
      diagnostics$cpu_dcov_pair_statistic_count
    ),
    cpu_gamma_p_value_count = sum(
      diagnostics$cpu_gamma_p_value_count
    ),
    cpu_spectra_count = sum(diagnostics$cpu_spectra_count),
    cuda_dcov_pair_count = sum(diagnostics$cuda_pair_count),
    cuda_gamma_p_value_count = sum(diagnostics$cuda_gamma_count),
    unknown_fallback_count = 0L,
    approximate_backend_count = 0L,
    edge_count_reference = as.integer(
      graph$summary$edge_count_reference
    ),
    edge_count_candidate = as.integer(
      graph$summary$edge_count_candidate
    ),
    SHD = as.integer(graph$summary$SHD),
    adjacency_identical = isTRUE(graph$summary$adjacency_identical),
    sepsets_identical = isTRUE(graph$summary$sepsets_identical),
    n_edgetests_identical = isTRUE(
      graph$summary$n_edgetests_identical
    ),
    deletions_identical = isTRUE(graph$summary$deletions_identical),
    elapsed_sec = as.numeric(elapsed),
    pass = TRUE
  )
  fastkpc_full_cuda_phase8_require(
    summary$logical_test_count == 240489L &&
      summary$direct_logical_test_count == 2213L &&
      summary$conditional_logical_test_count == 238276L &&
      summary$unique_residual_component_count == 110665L &&
      summary$exact_screen_component_count == 110665L &&
      summary$guarded_pair_count > 0L &&
      summary$screen_decision_flip_count > 0L &&
      summary$final_decision_flip_count == 0L &&
      summary$near_alpha_count == 1529L &&
      summary$near_alpha_final_decision_flip_count == 0L &&
      is.finite(summary$cuda_dcov_host_boundary_ms) &&
      summary$cuda_dcov_host_boundary_ms > 0 &&
      summary$legacy_cpu_spectra_dcov_ms >
        summary$cuda_dcov_host_boundary_ms &&
      summary$dcov_performance_ratio < 1 &&
      summary$dcov_budget_ms == 47000 &&
      summary$dcov_budget_pass &&
      summary$dcov_same_machine_speed_pass &&
      summary$matrix_h2d_bytes > 0 &&
      summary$residual_d2h_bytes == 0 &&
      summary$component_d2h_bytes == 0 &&
      summary$host_synchronization_count > 0L &&
      summary$component_cache_request_count ==
        summary$component_cache_hit_count +
          summary$component_cache_miss_count &&
      summary$component_cache_eviction_count == 0L &&
      summary$cpu_dcov_component_count == 0L &&
      summary$cpu_dcov_eigen_or_lowrank_count == 0L &&
      summary$cpu_dcov_pair_statistic_count == 0L &&
      summary$cpu_gamma_p_value_count == 0L &&
      summary$cpu_spectra_count == 0L &&
      summary$edge_count_reference == 110L &&
      summary$edge_count_candidate == 110L && summary$SHD == 0L &&
      summary$adjacency_identical && summary$sepsets_identical &&
      summary$n_edgetests_identical && summary$deletions_identical,
    "Phase 8 full evidence gate failed"
  )
  list(
    schema_version = "full-cuda-ci-phase8-full-evidence-v1",
    summary = summary,
    guard = execution$guard,
    pairs = pairs,
    diagnostics = diagnostics,
    runtime_info = execution$runtime_info,
    graph = graph,
    execution_identity = execution_identity$value,
    execution_identity_sha256 = execution_identity$identity_sha256,
    source_closure = execution_identity$source_closure,
    phase7_evidence_sha256 = inputs$phase7_evidence_sha256,
    phase7_summary = inputs$phase7$summary
  )
}

fastkpc_full_cuda_phase8_validate_full <- function(
    evidence, inputs = NULL, verify_current_identity = TRUE) {
  fastkpc_full_cuda_phase8_require(
    is.list(evidence) && identical(
      evidence$schema_version, "full-cuda-ci-phase8-full-evidence-v1"
    ) && is.list(evidence$summary) &&
      identical(
        evidence$summary$schema_version,
        "full-cuda-ci-phase8-full-summary-v1"
      ) && identical(evidence$summary$run_status, "ok") &&
      isTRUE(evidence$summary$pass) &&
      is.data.frame(evidence$pairs) && nrow(evidence$pairs) == 240489L &&
      is.data.frame(evidence$diagnostics) &&
      nrow(evidence$diagnostics) == 8635L &&
      identical(evidence$pairs$logical_sequence_id, 1:240489) &&
      sum(evidence$pairs$level == 0L) == 2213L &&
      sum(evidence$pairs$level > 0L) == 238276L &&
      sum(evidence$pairs$near_alpha) == 1529L &&
      all(!evidence$pairs$screen_decision_flip | evidence$pairs$refined) &&
      !any(evidence$pairs$final_decision_flip) &&
      all(is.finite(evidence$pairs$screen_p_value)) &&
      all(is.finite(evidence$pairs$final_p_value)) &&
      all(evidence$diagnostics$residual_d2h_bytes == 0) &&
      all(evidence$diagnostics$component_d2h_bytes == 0) &&
      all(evidence$diagnostics$cpu_dcov_component_count == 0L) &&
      all(evidence$diagnostics$cpu_dcov_eigen_or_lowrank_count == 0L) &&
      all(evidence$diagnostics$cpu_dcov_pair_statistic_count == 0L) &&
      all(evidence$diagnostics$cpu_gamma_p_value_count == 0L) &&
      all(evidence$diagnostics$cpu_spectra_count == 0L) &&
      all(evidence$diagnostics$solver_failure_count == 0L) &&
      all(evidence$diagnostics$leak_free) &&
      evidence$summary$logical_test_count == 240489L &&
      evidence$summary$unique_residual_component_count == 110665L &&
      evidence$summary$exact_screen_component_count == 110665L &&
      evidence$summary$screen_decision_flip_count > 0L &&
      evidence$summary$final_decision_flip_count == 0L &&
      evidence$summary$cuda_dcov_host_boundary_ms > 0 &&
      evidence$summary$cuda_dcov_host_boundary_ms <= 47000 &&
      evidence$summary$legacy_cpu_spectra_dcov_ms >
        evidence$summary$cuda_dcov_host_boundary_ms &&
      evidence$summary$dcov_performance_ratio < 1 &&
      isTRUE(evidence$summary$dcov_budget_pass) &&
      isTRUE(evidence$summary$dcov_same_machine_speed_pass) &&
      evidence$summary$matrix_h2d_bytes > 0 &&
      evidence$summary$residual_d2h_bytes == 0 &&
      evidence$summary$component_d2h_bytes == 0 &&
      evidence$summary$host_synchronization_count > 0L &&
      evidence$summary$component_cache_request_count ==
        evidence$summary$component_cache_hit_count +
          evidence$summary$component_cache_miss_count &&
      evidence$summary$component_cache_eviction_count == 0L &&
      evidence$summary$cpu_dcov_component_count == 0L &&
      evidence$summary$cpu_dcov_eigen_or_lowrank_count == 0L &&
      evidence$summary$cpu_dcov_pair_statistic_count == 0L &&
      evidence$summary$cpu_gamma_p_value_count == 0L &&
      evidence$summary$cpu_spectra_count == 0L &&
      evidence$summary$unknown_fallback_count == 0L &&
      evidence$summary$approximate_backend_count == 0L &&
      evidence$summary$edge_count_reference == 110L &&
      evidence$summary$edge_count_candidate == 110L &&
      evidence$summary$SHD == 0L &&
      isTRUE(evidence$summary$adjacency_identical) &&
      isTRUE(evidence$summary$sepsets_identical) &&
      isTRUE(evidence$summary$n_edgetests_identical) &&
      isTRUE(evidence$summary$deletions_identical) &&
      isTRUE(evidence$graph$pass) &&
      is.character(evidence$execution_identity_sha256) &&
      grepl("^[0-9a-f]{64}$", evidence$execution_identity_sha256) &&
      identical(
        evidence$execution_identity_sha256,
        fastkpc_full_cuda_census_named_metadata_hash(
          evidence$execution_identity
        )
      ) && identical(
        evidence$source_closure$sha256,
        evidence$execution_identity$source_closure_sha256
      ) && identical(
        evidence$phase7_evidence_sha256,
        evidence$execution_identity$phase7_evidence_sha256
      ),
    "Phase 8 full evidence structural gate failed"
  )
  expected_n_edgetests <- c(
    2213L, 52659L, 125293L, 40694L, 13293L, 5422L, 835L, 80L
  )
  fastkpc_full_cuda_phase8_require(
    identical(
      as.integer(evidence$graph$replay$skeleton$n.edgetests),
      expected_n_edgetests
    ),
    "Phase 8 full evidence logical n.edgetests drifted"
  )
  if (!is.null(inputs)) {
    fastkpc_full_cuda_phase8_require(
      identical(
        evidence$phase7_evidence_sha256,
        inputs$phase7_evidence_sha256
      ) && identical(
        evidence$phase7_summary$SHD,
        inputs$phase7$summary$SHD
      ),
      "Phase 8 Phase 7 evidence linkage failed"
    )
  }
  if (isTRUE(verify_current_identity)) {
    current <- if (is.null(inputs)) {
      fastkpc_full_cuda_phase8_source_closure()
    } else {
      fastkpc_full_cuda_phase8_execution_identity(inputs)$source_closure
    }
    fastkpc_full_cuda_phase8_require(
      identical(current$sha256, evidence$source_closure$sha256) &&
        identical(
          current$hashes[evidence$source_closure$paths],
          evidence$source_closure$hashes
        ) && identical(
          fastkpc_full_cuda_phase7_native_identity()$sha256,
          evidence$execution_identity$native_binary_sha256
        ),
      "Phase 8 full evidence current source/binary identity mismatch"
    )
  }
  invisible(TRUE)
}
