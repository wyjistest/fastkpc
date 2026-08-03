fastkpc_full_cuda_phase10_optimizer_residual_require <-
function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
  invisible(TRUE)
}

fastkpc_full_cuda_phase10_optimizer_residual_default_output <- function() {
  file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "phase10_profile_v2",
    "optimizer-residual-reuse-qualification-v1.rds"
  )
}

fastkpc_full_cuda_phase10_optimizer_residual_bits_equal <- function(
    left, right) {
  left <- as.double(left)
  right <- as.double(right)
  if (length(left) != length(right)) return(FALSE)
  identical(
    writeBin(left, raw(), size = 8L, endian = .Platform$endian),
    writeBin(right, raw(), size = 8L, endian = .Platform$endian)
  )
}

fastkpc_full_cuda_phase10_optimizer_residual_routes <- function(
    optimization, coefficient_dim) {
  contract <- fastkpc_full_cuda_fixed_sp_contract()
  ifelse(
    optimization$numerical_rank < coefficient_dim |
      !is.finite(optimization$condition) |
      optimization$condition >= contract$svd_condition_min,
    "AUGMENTED_SVD",
    ifelse(
      optimization$condition >= contract$cholesky_condition_max,
      "AUGMENTED_QR", "CHOLESKY_BATCHED"
    )
  )
}

fastkpc_full_cuda_phase10_optimizer_residual_target_keys <- function(
    prepared_s_key, targets) {
  vapply(targets, function(target) {
    fastkpc_full_cuda_phase35_sha256_utf8(paste0(
      "schema=full-cuda-ci-phase10-optimizer-residual-diagnostic-key-v1\n",
      "prepared=", prepared_s_key, "\n",
      "target=", as.integer(target), "\n"
    ))
  }, character(1L), USE.NAMES = FALSE)
}

fastkpc_full_cuda_phase10_optimizer_residual_pair_parity <- function(
    fixed, optimizer, alpha = 0.1) {
  count <- length(fixed$records$p_value)
  fastkpc_full_cuda_phase10_optimizer_residual_require(
    count > 0L && length(optimizer$records$p_value) == count,
    "optimizer residual dCov result count changed"
  )
  bitwise <- vapply(seq_len(count), function(index) {
    fastkpc_full_cuda_phase10_optimizer_residual_bits_equal(
      fixed$records$p_value[[index]], optimizer$records$p_value[[index]]
    )
  }, logical(1L))
  data.frame(
    pair_index = seq_len(count),
    fixed_p_value = as.numeric(fixed$records$p_value),
    optimizer_p_value = as.numeric(optimizer$records$p_value),
    bitwise_equal = bitwise,
    absolute_difference = abs(
      as.numeric(fixed$records$p_value) -
        as.numeric(optimizer$records$p_value)
    ),
    decision_flip =
      (as.numeric(fixed$records$p_value) >= alpha) !=
        (as.numeric(optimizer$records$p_value) >= alpha),
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_phase10_qualify_optimizer_residual_reuse <- function(
    setup, states, data, device_id = 0L) {
  required <- c(
    "fastkpc_full_cuda_phase6_prepare",
    "fastkpc_full_cuda_phase6_prepared_create",
    "fastkpc_full_cuda_phase6_optimize_prepared",
    "fastkpc_full_cuda_fixed_sp_native_dto",
    "fixed_sp_cuda_runtime_create", "fixed_sp_cuda_runtime_reserve",
    "fixed_sp_cuda_prepared_create", "fixed_sp_cuda_solve_batch",
    "full_cuda_ci_phase10_optimizer_residual_parity_native",
    "fastkpc_full_cuda_phase35_exact_batch_request",
    "fastkpc_full_cuda_phase35_exact_batch_ci",
    "fastkpc_full_cuda_phase35_exact_batch_from_optimizer_residual",
    "fastkpc_full_cuda_phase35_legacy_eig_batch_request",
    "fastkpc_full_cuda_phase35_legacy_eig_batch_ci",
    "fastkpc_full_cuda_phase35_legacy_eig_batch_from_optimizer_residual"
  )
  missing <- required[!vapply(
    required, exists, logical(1L), mode = "function", inherits = TRUE
  )]
  fastkpc_full_cuda_phase10_optimizer_residual_require(
    length(missing) == 0L,
    paste("optimizer residual qualification API is unavailable:",
          paste(missing, collapse = ", "))
  )
  fastkpc_full_cuda_phase10_optimizer_residual_require(
    is.list(setup) && is.data.frame(states) && nrow(states) >= 2L &&
      length(setup$penalty_blocks) > 1L && is.matrix(data) &&
      all(states$prepared_s_key_sha256 == setup$prepared_s_key_sha256) &&
      all(states$target >= 1L & states$target <= ncol(data)) &&
      length(device_id) == 1L && is.finite(device_id) && device_id >= 0L,
    "optimizer residual qualification fixture is malformed"
  )
  states <- states[seq_len(2L), , drop = FALSE]
  contexts <- lapply(seq_len(nrow(states)), function(index) {
    value <- fastkpc_full_cuda_materialize_target_state(
      states[index, , drop = FALSE], data, setup$dataset_sha256
    )
    fastkpc_full_cuda_validate_materialized_target_for_prepared(setup, value)
  })
  Y <- do.call(cbind, lapply(contexts, `[[`, "y"))
  target_keys <- fastkpc_full_cuda_phase10_optimizer_residual_target_keys(
    setup$prepared_s_key_sha256, states$target
  )

  multi_handle <- NULL
  optimizer_token <- NULL
  fixed_runtime <- NULL
  fixed_handle <- NULL
  fixed_token <- NULL
  on.exit({
    if (!is.null(fixed_token)) {
      try(fixed_sp_cuda_residual_free(fixed_token), silent = TRUE)
    }
    if (!is.null(optimizer_token)) {
      try(full_cuda_ci_multi_penalty_gcv_residual_free_native(
        optimizer_token
      ), silent = TRUE)
    }
    if (!is.null(multi_handle)) {
      try(full_cuda_ci_multi_penalty_gcv_prepared_free_native(
        multi_handle
      ), silent = TRUE)
    }
    if (!is.null(fixed_handle)) {
      try(fixed_sp_cuda_prepared_free(fixed_handle), silent = TRUE)
    }
    if (!is.null(fixed_runtime)) {
      try(fixed_sp_cuda_runtime_free(fixed_runtime), silent = TRUE)
    }
  }, add = TRUE)

  prepared <- fastkpc_full_cuda_phase6_prepare(setup)
  multi_handle <- fastkpc_full_cuda_phase6_prepared_create(
    prepared, target_capacity = ncol(Y), device_id = as.integer(device_id)
  )
  optimizer <- fastkpc_full_cuda_phase6_optimize_prepared(
    multi_handle, Y, target_keys
  )
  optimizer_token <- optimizer$residual
  selected_sp <- matrix(
    exp(optimizer$optimization$selected_log_sp),
    nrow = length(setup$penalty_blocks), ncol = ncol(Y)
  )
  planned_route <- fastkpc_full_cuda_phase10_optimizer_residual_routes(
    optimizer$optimization, ncol(setup$X)
  )

  dto <- fastkpc_full_cuda_fixed_sp_native_dto(setup)
  fixed_runtime <- fixed_sp_cuda_runtime_create(as.integer(device_id))
  fixed_sp_cuda_runtime_reserve(
    fixed_runtime, dto$n, dto$null_dim, ncol(Y), dto$penalty_count,
    dto$n + sum(dto$penalty_ranks)
  )
  fixed_handle <- fixed_sp_cuda_prepared_create(fixed_runtime, dto)
  fixed_token <- fixed_sp_cuda_solve_batch(
    fixed_handle, Y, selected_sp, planned_route, target_keys,
    outputs = "residuals"
  )
  residual_parity <- full_cuda_ci_phase10_optimizer_residual_parity_native(
    optimizer_token, fixed_token
  )
  fixed_sp_cuda_residual_free(fixed_token)
  fixed_token <- NULL

  exact_request <- fastkpc_full_cuda_phase35_exact_batch_request(
    expected_prepared_s_key_sha256 = setup$prepared_s_key_sha256,
    target_keys = target_keys,
    logical_sequence_ids = 1,
    left_target_ordinals = 1L,
    right_target_ordinals = 2L,
    alpha = 0.1,
    component_capacity = 2L
  )
  exact_fixed <- fastkpc_full_cuda_phase35_exact_batch_ci(
    fixed_handle, Y, selected_sp, planned_route, target_keys, exact_request
  )
  exact_optimizer <-
    fastkpc_full_cuda_phase35_exact_batch_from_optimizer_residual(
      fixed_handle, optimizer_token, exact_request
    )
  exact_parity <- fastkpc_full_cuda_phase10_optimizer_residual_pair_parity(
    exact_fixed, exact_optimizer
  )

  legacy_request <- fastkpc_full_cuda_phase35_legacy_eig_batch_request(
    expected_prepared_s_key_sha256 = setup$prepared_s_key_sha256,
    target_keys = target_keys,
    logical_sequence_ids = 1,
    left_target_ordinals = 1L,
    right_target_ordinals = 2L,
    alpha = 0.1,
    component_capacity = 2L,
    num_col = 35L
  )
  legacy_fixed <- fastkpc_full_cuda_phase35_legacy_eig_batch_ci(
    fixed_handle, Y, selected_sp, planned_route, target_keys, legacy_request
  )
  legacy_optimizer <-
    fastkpc_full_cuda_phase35_legacy_eig_batch_from_optimizer_residual(
      fixed_handle, optimizer_token, legacy_request
    )
  legacy_parity <- fastkpc_full_cuda_phase10_optimizer_residual_pair_parity(
    legacy_fixed, legacy_optimizer
  )

  prepared_before_release <-
    full_cuda_ci_multi_penalty_gcv_prepared_info_native(multi_handle)
  full_cuda_ci_multi_penalty_gcv_residual_release_native(optimizer_token)
  released_info <-
    full_cuda_ci_multi_penalty_gcv_residual_info_native(optimizer_token)
  stale_error <- tryCatch({
    fastkpc_full_cuda_phase35_exact_batch_from_optimizer_residual(
      fixed_handle, optimizer_token, exact_request
    )
    NULL
  }, error = identity)
  stale_rejected <- inherits(stale_error, "error") && grepl(
    "stale or released", conditionMessage(stale_error), fixed = TRUE
  )
  full_cuda_ci_multi_penalty_gcv_residual_free_native(optimizer_token)
  optimizer_token <- NULL
  full_cuda_ci_multi_penalty_gcv_prepared_free_native(multi_handle)
  multi_handle <- NULL

  numerical_gate <-
    residual_parity$bitwise_equal_target_count == ncol(Y) &&
      residual_parity$mismatch_value_count == 0 &&
      all(exact_parity$bitwise_equal) &&
      all(legacy_parity$bitwise_equal)
  authority_gate <-
    residual_parity$residual_d2h_bytes == 0 &&
      exact_optimizer$diagnostics$residual_d2h_bytes == 0 &&
      legacy_optimizer$diagnostics$residual_d2h_bytes == 0 &&
      prepared_before_release$residual_shadow_d2h_count == 0L &&
      isTRUE(stale_rejected) && isTRUE(released_info$released)
  decision <- if (numerical_gate && authority_gate) {
    "GO_DETACHED_ARENA_QUALIFICATION"
  } else if (!numerical_gate) {
    "STOP_OPTIMIZER_RESIDUAL_NUMERICAL_PARITY"
  } else {
    "STOP_OPTIMIZER_RESIDUAL_AUTHORITY_OR_LIFETIME"
  }

  list(
    schema_version =
      "full-cuda-ci-phase10-optimizer-residual-qualification-v1",
    decision = decision,
    fixture = list(
      prepared_s_key_sha256 = setup$prepared_s_key_sha256,
      target_indices = as.integer(states$target),
      target_keys = target_keys,
      n = as.integer(nrow(Y)),
      coefficient_dim = as.integer(ncol(setup$X)),
      penalty_count = as.integer(length(setup$penalty_blocks)),
      target_count = as.integer(ncol(Y)),
      planned_route = as.character(planned_route)
    ),
    residual_parity = residual_parity,
    exact_p_value_parity = exact_parity,
    legacy_eig_p_value_parity = legacy_parity,
    consumer = list(
      exact_residual_solve_bypassed =
        exact_optimizer$diagnostics$residual_solve_bypassed,
      legacy_residual_solve_bypassed =
        legacy_optimizer$diagnostics$residual_solve_bypassed,
      exact_residual_d2h_bytes =
        exact_optimizer$diagnostics$residual_d2h_bytes,
      legacy_residual_d2h_bytes =
        legacy_optimizer$diagnostics$residual_d2h_bytes,
      exact_producer_semantic_identity =
        exact_optimizer$diagnostics$residual_producer_semantic_identity,
      legacy_producer_semantic_identity =
        legacy_optimizer$diagnostics$residual_producer_semantic_identity
    ),
    lifetime = list(
      residual_slot_leased_before_release =
        prepared_before_release$residual_slot_leased,
      residual_shadow_d2h_count =
        prepared_before_release$residual_shadow_d2h_count,
      released = released_info$released,
      stale_token_rejected = stale_rejected,
      stale_error = if (inherits(stale_error, "error"))
        conditionMessage(stale_error) else ""
    ),
    detached_arena = list(
      attempted = FALSE,
      reason = if (!numerical_gate)
        "not-attempted-after-numerical-parity-stop" else
        "not-implemented-in-qualification-v1",
      d2d_copy_bytes = 0,
      d2d_copy_ms = NA_real_,
      peak_live_bytes = NA_real_
    ),
    gates = list(
      numerical_gate = numerical_gate,
      authority_gate = authority_gate,
      exact_p_value_mismatch_count = sum(!exact_parity$bitwise_equal),
      legacy_eig_p_value_mismatch_count =
        sum(!legacy_parity$bitwise_equal),
      final_decision_flip_count = sum(exact_parity$decision_flip) +
        sum(legacy_parity$decision_flip)
    )
  )
}

fastkpc_full_cuda_phase10_validate_optimizer_residual_qualification <-
function(value) {
  valid <- is.list(value) && identical(
    value$schema_version,
    "full-cuda-ci-phase10-optimizer-residual-qualification-v1"
  ) && value$decision %in% c(
    "GO_DETACHED_ARENA_QUALIFICATION",
    "STOP_OPTIMIZER_RESIDUAL_NUMERICAL_PARITY",
    "STOP_OPTIMIZER_RESIDUAL_AUTHORITY_OR_LIFETIME"
  ) && is.list(value$residual_parity) &&
    value$residual_parity$value_count ==
      value$fixture$n * value$fixture$target_count &&
    value$residual_parity$bitwise_equal_target_count +
      value$residual_parity$mismatch_target_count ==
        value$fixture$target_count &&
    value$residual_parity$bitwise_equal_value_count +
      value$residual_parity$mismatch_value_count ==
        value$residual_parity$value_count &&
    value$residual_parity$residual_d2h_bytes == 0 &&
    isTRUE(value$residual_parity$compact_diagnostics_only_d2h) &&
    isTRUE(value$lifetime$stale_token_rejected) &&
    value$lifetime$residual_shadow_d2h_count == 0L &&
    identical(
      value$decision == "GO_DETACHED_ARENA_QUALIFICATION",
      isTRUE(value$gates$numerical_gate) && isTRUE(value$gates$authority_gate)
    )
  fastkpc_full_cuda_phase10_optimizer_residual_require(
    valid, "optimizer residual qualification is malformed"
  )
  invisible(TRUE)
}
