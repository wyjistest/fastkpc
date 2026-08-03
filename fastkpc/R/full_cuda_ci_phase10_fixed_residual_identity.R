fastkpc_full_cuda_phase10_fixed_identity_require <- function(
    condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
  invisible(TRUE)
}

fastkpc_full_cuda_phase10_fixed_identity_default_output <- function() {
  file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "phase10_profile_v2",
    "fixed-residual-cross-batch-identity-v1.rds"
  )
}

fastkpc_full_cuda_phase10_fixed_identity_bits_equal <- function(left, right) {
  left <- as.double(left)
  right <- as.double(right)
  if (length(left) != length(right)) return(FALSE)
  identical(
    writeBin(left, raw(), size = 8L, endian = .Platform$endian),
    writeBin(right, raw(), size = 8L, endian = .Platform$endian)
  )
}

fastkpc_full_cuda_phase10_fixed_identity_fixture <- function(
    setup, states, data, planned_route) {
  fastkpc_full_cuda_phase10_fixed_identity_require(
    is.list(setup) && is.data.frame(states) && nrow(states) >= 2L &&
      is.matrix(data) && nrow(data) == nrow(setup$X) &&
      all(states$prepared_s_key_sha256 == setup$prepared_s_key_sha256) &&
      length(planned_route) == nrow(states) &&
      all(planned_route %in%
            fastkpc_full_cuda_fixed_sp_contract()$route_levels),
    "cross-batch fixed residual fixture is malformed"
  )
  contexts <- lapply(seq_len(nrow(states)), function(index) {
    value <- fastkpc_full_cuda_materialize_target_state(
      states[index, , drop = FALSE], data, setup$dataset_sha256
    )
    fastkpc_full_cuda_validate_materialized_target_for_prepared(setup, value)
  })
  Y <- do.call(cbind, lapply(contexts, `[[`, "y"))
  SP <- do.call(cbind, unclass(states$selected_sp))
  target_keys <- as.character(states$residual_key_sha256)
  fastkpc_full_cuda_phase10_fixed_identity_require(
    is.matrix(Y) && is.matrix(SP) &&
      identical(dim(SP), c(length(setup$penalty_blocks), ncol(Y))) &&
      length(target_keys) == ncol(Y) && !anyDuplicated(target_keys),
    "cross-batch fixed residual fixture payload is malformed"
  )
  list(
    setup = setup,
    Y = unname(Y),
    SP = unname(SP),
    planned_route = as.character(planned_route),
    target_keys = target_keys,
    targets = as.integer(states$target)
  )
}

fastkpc_full_cuda_phase10_fixed_identity_variant <- function(
    name, left, right, subset_sensitive) {
  list(
    name = as.character(name),
    left = as.integer(left),
    right = as.integer(right),
    subset_sensitive = isTRUE(subset_sensitive)
  )
}

fastkpc_full_cuda_phase10_fixed_identity_default_fixtures <- function() {
  data <- readRDS(file.path(
    "fastkpc", "artifacts", "kpc_tprs_real_zhu",
    "cancer_RD-causalDiscoveryInput.rds"
  ))
  fixture_spec <- list(
    route_mixed_multi = list(
      shard = "shard_63.rds",
      prepared_s_key =
        "507f4d6258e2f8dd3e185088be1a533ba6c602a919a0469de4c636ff6ea2abb1",
      targets = c(19L, 6L, 25L, 22L),
      planned_route = c(
        "CHOLESKY_BATCHED", "AUGMENTED_SVD", "AUGMENTED_QR",
        "CHOLESKY_BATCHED"
      ),
      variants = list(
        fastkpc_full_cuda_phase10_fixed_identity_variant(
          "repeat_original_cohort", 1:4, 1:4, FALSE
        ),
        fastkpc_full_cuda_phase10_fixed_identity_variant(
          "permuted_original_cohort", 1:4, c(4, 2, 1, 3), FALSE
        ),
        fastkpc_full_cuda_phase10_fixed_identity_variant(
          "miss_only_three_route_subset", 1:4, 1:3, TRUE
        ),
        fastkpc_full_cuda_phase10_fixed_identity_variant(
          "miss_only_cholesky_singleton", 1:4, 1L, TRUE
        ),
        fastkpc_full_cuda_phase10_fixed_identity_variant(
          "miss_only_svd_singleton", 1:4, 2L, TRUE
        ),
        fastkpc_full_cuda_phase10_fixed_identity_variant(
          "miss_only_qr_singleton", 1:4, 3L, TRUE
        )
      )
    ),
    single_penalty_cholesky = list(
      shard = "shard_0.rds",
      prepared_s_key =
        "07ebec707653cb97d9ad52d983ffdad00904756981ab369db0368bb783532cbf",
      targets = c(47L, 26L, 6L, 3L),
      planned_route = rep("CHOLESKY_BATCHED", 4L),
      variants = list(
        fastkpc_full_cuda_phase10_fixed_identity_variant(
          "repeat_original_cohort", 1:4, 1:4, FALSE
        ),
        fastkpc_full_cuda_phase10_fixed_identity_variant(
          "permuted_original_cohort", 1:4, c(3, 1, 4, 2), FALSE
        ),
        fastkpc_full_cuda_phase10_fixed_identity_variant(
          "miss_only_two_target_subset", 1:4, 1:2, TRUE
        ),
        fastkpc_full_cuda_phase10_fixed_identity_variant(
          "miss_only_cholesky_singleton", 1:4, 1L, TRUE
        )
      )
    )
  )
  fixtures <- lapply(fixture_spec, function(spec) {
    shard <- readRDS(file.path(
      "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1",
      "shards", spec$shard
    ))
    setup <- shard$prepared_s_setups[[spec$prepared_s_key]]
    states <- shard$target_states[
      shard$target_states$prepared_s_key_sha256 == spec$prepared_s_key,
      , drop = FALSE
    ]
    states <- states[match(spec$targets, states$target), , drop = FALSE]
    fastkpc_full_cuda_phase10_fixed_identity_require(
      !is.null(setup) && nrow(states) == length(spec$targets) &&
        !anyNA(states$target) &&
        identical(as.integer(states$target), spec$targets),
      "cross-batch fixed residual default fixture is unavailable"
    )
    list(
      fixture = fastkpc_full_cuda_phase10_fixed_identity_fixture(
        setup, states, data, spec$planned_route
      ),
      variants = spec$variants
    )
  })
  fixtures
}

fastkpc_full_cuda_phase10_fixed_identity_slice <- function(fixture, indices) {
  list(
    Y = fixture$Y[, indices, drop = FALSE],
    SP = fixture$SP[, indices, drop = FALSE],
    planned_route = fixture$planned_route[indices],
    target_keys = fixture$target_keys[indices]
  )
}

fastkpc_full_cuda_phase10_fixed_identity_pair_result <- function(
    prepared, left, right) {
  common <- left$target_keys[left$target_keys %in% right$target_keys]
  if (length(common) < 2L) {
    return(list(
      pair_count = 0L,
      exact_p_value_mismatch_count = 0L,
      exact_numerical_mismatch_count = 0L,
      guard_membership_mismatch_count = 0L,
      legacy_p_value_mismatch_count = 0L,
      final_p_value_mismatch_count = 0L,
      final_decision_flip_count = 0L
    ))
  }
  pairs <- utils::combn(common, 2L)
  pair_count <- ncol(pairs)
  left_ordinals <- match(pairs[1L, ], left$target_keys)
  left_right_ordinals <- match(pairs[2L, ], left$target_keys)
  right_ordinals <- match(pairs[1L, ], right$target_keys)
  right_right_ordinals <- match(pairs[2L, ], right$target_keys)
  logical_ids <- as.double(seq_len(pair_count))

  exact_left_request <- fastkpc_full_cuda_phase35_exact_batch_request(
    expected_prepared_s_key_sha256 =
      prepared$prepared_s_key_sha256,
    target_keys = left$target_keys,
    logical_sequence_ids = logical_ids,
    left_target_ordinals = left_ordinals,
    right_target_ordinals = left_right_ordinals,
    alpha = 0.1,
    component_capacity = length(left$target_keys)
  )
  exact_right_request <- fastkpc_full_cuda_phase35_exact_batch_request(
    expected_prepared_s_key_sha256 =
      prepared$prepared_s_key_sha256,
    target_keys = right$target_keys,
    logical_sequence_ids = logical_ids,
    left_target_ordinals = right_ordinals,
    right_target_ordinals = right_right_ordinals,
    alpha = 0.1,
    component_capacity = length(right$target_keys)
  )
  exact_left <- fastkpc_full_cuda_phase35_exact_batch_ci(
    prepared$left_handle, left$Y, left$SP, left$planned_route,
    left$target_keys, exact_left_request
  )
  exact_right <- fastkpc_full_cuda_phase35_exact_batch_ci(
    prepared$right_handle, right$Y, right$SP, right$planned_route,
    right$target_keys, exact_right_request
  )
  exact_p_equal <- vapply(seq_len(pair_count), function(index) {
    fastkpc_full_cuda_phase10_fixed_identity_bits_equal(
      exact_left$records$p_value[[index]],
      exact_right$records$p_value[[index]]
    )
  }, logical(1L))
  numerical_fields <- c(
    "statistic", "mean", "variance", "gamma_shape", "gamma_scale"
  )
  exact_numerical_equal <- vapply(seq_len(pair_count), function(index) {
    all(vapply(numerical_fields, function(field) {
      fastkpc_full_cuda_phase10_fixed_identity_bits_equal(
        exact_left$numerical[[field]][[index]],
        exact_right$numerical[[field]][[index]]
      )
    }, logical(1L))) &&
      identical(
        exact_left$numerical$gamma_iterations[[index]],
        exact_right$numerical$gamma_iterations[[index]]
      )
  }, logical(1L))
  left_guard <- exact_left$records$p_value >= 0.05 &
    exact_left$records$p_value <= 0.15
  right_guard <- exact_right$records$p_value >= 0.05 &
    exact_right$records$p_value <= 0.15

  legacy_left_request <- fastkpc_full_cuda_phase35_legacy_eig_batch_request(
    expected_prepared_s_key_sha256 =
      prepared$prepared_s_key_sha256,
    target_keys = left$target_keys,
    logical_sequence_ids = logical_ids,
    left_target_ordinals = left_ordinals,
    right_target_ordinals = left_right_ordinals,
    alpha = 0.1,
    component_capacity = length(left$target_keys),
    num_col = 35L
  )
  legacy_right_request <- fastkpc_full_cuda_phase35_legacy_eig_batch_request(
    expected_prepared_s_key_sha256 =
      prepared$prepared_s_key_sha256,
    target_keys = right$target_keys,
    logical_sequence_ids = logical_ids,
    left_target_ordinals = right_ordinals,
    right_target_ordinals = right_right_ordinals,
    alpha = 0.1,
    component_capacity = length(right$target_keys),
    num_col = 35L
  )
  legacy_left <- fastkpc_full_cuda_phase35_legacy_eig_batch_ci(
    prepared$left_handle, left$Y, left$SP, left$planned_route,
    left$target_keys, legacy_left_request
  )
  legacy_right <- fastkpc_full_cuda_phase35_legacy_eig_batch_ci(
    prepared$right_handle, right$Y, right$SP, right$planned_route,
    right$target_keys, legacy_right_request
  )
  legacy_equal <- vapply(seq_len(pair_count), function(index) {
    fastkpc_full_cuda_phase10_fixed_identity_bits_equal(
      legacy_left$records$p_value[[index]],
      legacy_right$records$p_value[[index]]
    )
  }, logical(1L))
  final_left <- ifelse(
    left_guard, legacy_left$records$p_value, exact_left$records$p_value
  )
  final_right <- ifelse(
    right_guard, legacy_right$records$p_value, exact_right$records$p_value
  )
  final_equal <- vapply(seq_len(pair_count), function(index) {
    fastkpc_full_cuda_phase10_fixed_identity_bits_equal(
      final_left[[index]], final_right[[index]]
    )
  }, logical(1L))
  list(
    pair_count = as.integer(pair_count),
    exact_p_value_mismatch_count = as.integer(sum(!exact_p_equal)),
    exact_numerical_mismatch_count =
      as.integer(sum(!exact_numerical_equal)),
    guard_membership_mismatch_count =
      as.integer(sum(left_guard != right_guard)),
    legacy_p_value_mismatch_count = as.integer(sum(!legacy_equal)),
    final_p_value_mismatch_count = as.integer(sum(!final_equal)),
    final_decision_flip_count = as.integer(sum(
      (final_left >= 0.1) != (final_right >= 0.1)
    ))
  )
}

fastkpc_full_cuda_phase10_qualify_fixed_identity_fixture <- function(
    fixture, variants, device_id = 0L) {
  max_targets <- max(vapply(variants, function(value) {
    max(length(value$left), length(value$right))
  }, integer(1L)))
  dto <- fastkpc_full_cuda_fixed_sp_native_dto(fixture$setup)
  runtime <- fixed_sp_cuda_runtime_create(as.integer(device_id))
  left_handle <- NULL
  right_handle <- NULL
  on.exit({
    if (!is.null(left_handle)) {
      try(fixed_sp_cuda_prepared_free(left_handle), silent = TRUE)
    }
    if (!is.null(right_handle)) {
      try(fixed_sp_cuda_prepared_free(right_handle), silent = TRUE)
    }
    if (!is.null(runtime)) {
      try(fixed_sp_cuda_runtime_free(runtime), silent = TRUE)
    }
  }, add = TRUE)
  fixed_sp_cuda_runtime_reserve(
    runtime, dto$n, dto$null_dim, max_targets, dto$penalty_count,
    dto$n + sum(dto$penalty_ranks)
  )
  left_handle <- fixed_sp_cuda_prepared_create(runtime, dto)
  right_handle <- fixed_sp_cuda_prepared_create(runtime, dto)
  prepared <- list(
    prepared_s_key_sha256 = fixture$setup$prepared_s_key_sha256,
    left_handle = left_handle,
    right_handle = right_handle
  )

  rows <- lapply(variants, function(variant) {
    left <- fastkpc_full_cuda_phase10_fixed_identity_slice(
      fixture, variant$left
    )
    right <- fastkpc_full_cuda_phase10_fixed_identity_slice(
      fixture, variant$right
    )
    left_token <- fixed_sp_cuda_solve_batch(
      left_handle, left$Y, left$SP, left$planned_route, left$target_keys,
      outputs = "residuals"
    )
    right_token <- NULL
    on.exit({
      if (!is.null(left_token)) {
        try(fixed_sp_cuda_residual_free(left_token), silent = TRUE)
      }
      if (!is.null(right_token)) {
        try(fixed_sp_cuda_residual_free(right_token), silent = TRUE)
      }
    }, add = TRUE)
    right_token <- fixed_sp_cuda_solve_batch(
      right_handle, right$Y, right$SP, right$planned_route,
      right$target_keys, outputs = "residuals"
    )
    identity <- full_cuda_ci_phase10_fixed_residual_identity_native(
      left_token, right_token
    )
    fixed_sp_cuda_residual_free(left_token)
    left_token <- NULL
    fixed_sp_cuda_residual_free(right_token)
    right_token <- NULL
    pair <- fastkpc_full_cuda_phase10_fixed_identity_pair_result(
      prepared, left, right
    )
    data.frame(
      variant = variant$name,
      subset_sensitive = variant$subset_sensitive,
      left_target_count = identity$left_target_count,
      right_target_count = identity$right_target_count,
      matched_target_count = identity$matched_target_count,
      residual_mismatch_value_count =
        identity$residual_mismatch_value_count,
      residual_mismatch_target_count =
        identity$residual_mismatch_target_count,
      residual_max_abs_difference = identity$residual_max_abs_difference,
      residual_relative_l2_difference =
        identity$residual_relative_l2_difference,
      centered_component_mismatch_value_count =
        identity$centered_component_mismatch_value_count,
      row_sum_mismatch_value_count =
        identity$row_sum_mismatch_value_count,
      total_mismatch_value_count = identity$total_mismatch_value_count,
      self_moment_mismatch_value_count =
        identity$self_moment_mismatch_value_count,
      component_mismatch_target_count =
        identity$component_mismatch_target_count,
      component_max_abs_difference = identity$component_max_abs_difference,
      component_relative_l2_difference =
        identity$component_relative_l2_difference,
      planned_route_mismatch_count =
        identity$planned_route_mismatch_count,
      executed_route_mismatch_count =
        identity$executed_route_mismatch_count,
      solver_status_mismatch_count = identity$solver_status_mismatch_count,
      pair_count = pair$pair_count,
      exact_p_value_mismatch_count = pair$exact_p_value_mismatch_count,
      exact_numerical_mismatch_count =
        pair$exact_numerical_mismatch_count,
      guard_membership_mismatch_count =
        pair$guard_membership_mismatch_count,
      legacy_p_value_mismatch_count =
        pair$legacy_p_value_mismatch_count,
      final_p_value_mismatch_count = pair$final_p_value_mismatch_count,
      final_decision_flip_count = pair$final_decision_flip_count,
      compact_d2h_bytes = identity$compact_d2h_bytes,
      residual_d2h_bytes = identity$residual_d2h_bytes,
      structural_gate =
        isTRUE(identity$target_identity_authenticated) &&
        isTRUE(identity$device_identity_authenticated) &&
        isTRUE(identity$residual_payload_device_resident) &&
        isTRUE(identity$component_payload_device_resident) &&
        isTRUE(identity$compact_diagnostics_only_d2h) &&
        isTRUE(identity$bounded_allocation) &&
        isTRUE(identity$leak_free_teardown) &&
        isTRUE(identity$caller_device_restored),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

fastkpc_full_cuda_phase10_qualify_fixed_residual_identity <- function(
    fixtures, device_id = 0L) {
  required <- c(
    "fastkpc_full_cuda_fixed_sp_native_dto",
    "fixed_sp_cuda_runtime_create", "fixed_sp_cuda_runtime_reserve",
    "fixed_sp_cuda_prepared_create", "fixed_sp_cuda_solve_batch",
    "full_cuda_ci_phase10_fixed_residual_identity_native",
    "fastkpc_full_cuda_phase35_exact_batch_request",
    "fastkpc_full_cuda_phase35_exact_batch_ci",
    "fastkpc_full_cuda_phase35_legacy_eig_batch_request",
    "fastkpc_full_cuda_phase35_legacy_eig_batch_ci"
  )
  missing <- required[!vapply(
    required, exists, logical(1L), mode = "function", inherits = TRUE
  )]
  fastkpc_full_cuda_phase10_fixed_identity_require(
    length(missing) == 0L && is.list(fixtures) && length(fixtures) > 0L,
    paste("cross-batch fixed residual API is unavailable:",
          paste(missing, collapse = ", "))
  )
  resource_before <- fixed_sp_cuda_live_owner_snapshot()
  rows <- do.call(rbind, lapply(names(fixtures), function(name) {
    entry <- fixtures[[name]]
    result <- fastkpc_full_cuda_phase10_qualify_fixed_identity_fixture(
      entry$fixture, entry$variants, device_id = device_id
    )
    result$fixture <- name
    result[, c("fixture", setdiff(names(result), "fixture")), drop = FALSE]
  }))
  resource_after <- fixed_sp_cuda_live_owner_snapshot()
  numeric_mismatch <- rows$residual_mismatch_value_count > 0 |
    rows$centered_component_mismatch_value_count > 0 |
    rows$row_sum_mismatch_value_count > 0 |
    rows$total_mismatch_value_count > 0 |
    rows$self_moment_mismatch_value_count > 0 |
    rows$exact_p_value_mismatch_count > 0 |
    rows$exact_numerical_mismatch_count > 0 |
    rows$guard_membership_mismatch_count > 0 |
    rows$legacy_p_value_mismatch_count > 0 |
    rows$final_p_value_mismatch_count > 0
  metadata_mismatch <- rows$planned_route_mismatch_count > 0 |
    rows$executed_route_mismatch_count > 0 |
    rows$solver_status_mismatch_count > 0
  row_gate <- !numeric_mismatch & !metadata_mismatch & rows$structural_gate
  cohort_gate <- all(row_gate[!rows$subset_sensitive])
  target_granular_gate <- cohort_gate && all(row_gate)
  decision <- if (target_granular_gate) {
    "GO_TARGET_GRANULAR_FIXED_RESIDUAL_CACHE"
  } else if (cohort_gate) {
    "CONDITIONAL_ALL_HIT_BATCH_ONLY"
  } else {
    "STOP_CROSS_BATCH_FIXED_RESIDUAL_IDENTITY"
  }
  list(
    schema_version =
      "full-cuda-ci-phase10-fixed-residual-identity-qualification-v1",
    decision = decision,
    rows = rows,
    gates = list(
      cohort_gate = cohort_gate,
      target_granular_gate = target_granular_gate,
      residual_mismatch_value_count =
        sum(rows$residual_mismatch_value_count),
      component_mismatch_value_count = sum(
        rows$centered_component_mismatch_value_count +
          rows$row_sum_mismatch_value_count +
          rows$total_mismatch_value_count +
          rows$self_moment_mismatch_value_count
      ),
      exact_p_value_mismatch_count =
        sum(rows$exact_p_value_mismatch_count),
      final_p_value_mismatch_count =
        sum(rows$final_p_value_mismatch_count),
      final_decision_flip_count = sum(rows$final_decision_flip_count),
      metadata_mismatch_count = sum(
        rows$planned_route_mismatch_count +
          rows$executed_route_mismatch_count +
          rows$solver_status_mismatch_count
      ),
      structural_gate = all(rows$structural_gate),
      resource_gate = identical(resource_before, resource_after)
    )
  )
}

fastkpc_full_cuda_phase10_validate_fixed_residual_identity <- function(value) {
  decision_levels <- c(
    "GO_TARGET_GRANULAR_FIXED_RESIDUAL_CACHE",
    "CONDITIONAL_ALL_HIT_BATCH_ONLY",
    "STOP_CROSS_BATCH_FIXED_RESIDUAL_IDENTITY"
  )
  rows <- value$rows
  expected_columns <- c(
    "fixture", "variant", "subset_sensitive", "left_target_count",
    "right_target_count", "matched_target_count",
    "residual_mismatch_value_count", "residual_mismatch_target_count",
    "residual_max_abs_difference", "residual_relative_l2_difference",
    "centered_component_mismatch_value_count",
    "row_sum_mismatch_value_count", "total_mismatch_value_count",
    "self_moment_mismatch_value_count", "component_mismatch_target_count",
    "component_max_abs_difference", "component_relative_l2_difference",
    "planned_route_mismatch_count", "executed_route_mismatch_count",
    "solver_status_mismatch_count", "pair_count",
    "exact_p_value_mismatch_count", "exact_numerical_mismatch_count",
    "guard_membership_mismatch_count", "legacy_p_value_mismatch_count",
    "final_p_value_mismatch_count", "final_decision_flip_count",
    "compact_d2h_bytes", "residual_d2h_bytes", "structural_gate"
  )
  numeric_columns <- setdiff(
    expected_columns,
    c("fixture", "variant", "subset_sensitive", "structural_gate")
  )
  integer_columns <- setdiff(numeric_columns, c(
    "residual_max_abs_difference", "residual_relative_l2_difference",
    "component_max_abs_difference", "component_relative_l2_difference"
  ))
  valid_rows <- is.data.frame(rows) &&
    identical(names(rows), expected_columns) && nrow(rows) > 0L &&
    all(vapply(rows[numeric_columns], function(column) {
      is.numeric(column) && all(is.finite(column)) && all(column >= 0)
    }, logical(1L))) &&
    all(vapply(rows[integer_columns], function(column) {
      all(column == floor(column))
    }, logical(1L))) &&
    is.logical(rows$subset_sensitive) && !anyNA(rows$subset_sensitive) &&
    is.logical(rows$structural_gate) && !anyNA(rows$structural_gate) &&
    all(nzchar(rows$fixture)) && all(nzchar(rows$variant)) &&
    !anyDuplicated(paste(rows$fixture, rows$variant, sep = "|")) &&
    all(rows$matched_target_count >= 1L) &&
    all(rows$matched_target_count <= rows$left_target_count) &&
    all(rows$matched_target_count <= rows$right_target_count) &&
    all(rows$residual_mismatch_target_count <= rows$matched_target_count) &&
    all(rows$component_mismatch_target_count <= rows$matched_target_count) &&
    all(rows$exact_p_value_mismatch_count <= rows$pair_count) &&
    all(rows$exact_numerical_mismatch_count <= rows$pair_count) &&
    all(rows$guard_membership_mismatch_count <= rows$pair_count) &&
    all(rows$legacy_p_value_mismatch_count <= rows$pair_count) &&
    all(rows$final_p_value_mismatch_count <= rows$pair_count) &&
    all(rows$final_decision_flip_count <= rows$pair_count) &&
    all(rows$residual_d2h_bytes == 0) &&
    all(rows$compact_d2h_bytes > 0) && all(rows$structural_gate) &&
    any(rows$subset_sensitive) && any(!rows$subset_sensitive)
  numeric_mismatch <- if (valid_rows) {
    rows$residual_mismatch_value_count > 0 |
      rows$centered_component_mismatch_value_count > 0 |
      rows$row_sum_mismatch_value_count > 0 |
      rows$total_mismatch_value_count > 0 |
      rows$self_moment_mismatch_value_count > 0 |
      rows$exact_p_value_mismatch_count > 0 |
      rows$exact_numerical_mismatch_count > 0 |
      rows$guard_membership_mismatch_count > 0 |
      rows$legacy_p_value_mismatch_count > 0 |
      rows$final_p_value_mismatch_count > 0
  } else logical()
  metadata_mismatch <- if (valid_rows) {
    rows$planned_route_mismatch_count > 0 |
      rows$executed_route_mismatch_count > 0 |
      rows$solver_status_mismatch_count > 0
  } else logical()
  row_gate <- if (valid_rows) {
    !numeric_mismatch & !metadata_mismatch & rows$structural_gate
  } else logical()
  cohort_gate <- valid_rows && all(row_gate[!rows$subset_sensitive])
  target_granular_gate <- cohort_gate && all(row_gate)
  expected_decision <- if (target_granular_gate) {
    "GO_TARGET_GRANULAR_FIXED_RESIDUAL_CACHE"
  } else if (cohort_gate) {
    "CONDITIONAL_ALL_HIT_BATCH_ONLY"
  } else {
    "STOP_CROSS_BATCH_FIXED_RESIDUAL_IDENTITY"
  }
  valid <- is.list(value) && identical(
    value$schema_version,
    "full-cuda-ci-phase10-fixed-residual-identity-qualification-v1"
  ) && value$decision %in% decision_levels && valid_rows &&
    identical(value$decision, expected_decision) && is.list(value$gates) &&
    identical(value$gates$cohort_gate, cohort_gate) &&
    identical(value$gates$target_granular_gate, target_granular_gate) &&
    value$gates$residual_mismatch_value_count ==
      sum(rows$residual_mismatch_value_count) &&
    value$gates$component_mismatch_value_count == sum(
      rows$centered_component_mismatch_value_count +
        rows$row_sum_mismatch_value_count +
        rows$total_mismatch_value_count +
        rows$self_moment_mismatch_value_count
    ) &&
    value$gates$exact_p_value_mismatch_count ==
      sum(rows$exact_p_value_mismatch_count) &&
    value$gates$final_p_value_mismatch_count ==
      sum(rows$final_p_value_mismatch_count) &&
    value$gates$final_decision_flip_count ==
      sum(rows$final_decision_flip_count) &&
    value$gates$metadata_mismatch_count == sum(
      rows$planned_route_mismatch_count +
        rows$executed_route_mismatch_count +
        rows$solver_status_mismatch_count
    ) && isTRUE(value$gates$structural_gate) &&
    isTRUE(value$gates$resource_gate)
  fastkpc_full_cuda_phase10_fixed_identity_require(
    valid, "cross-batch fixed residual qualification is malformed"
  )
  invisible(TRUE)
}
