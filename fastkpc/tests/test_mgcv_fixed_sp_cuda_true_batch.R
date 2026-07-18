source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
relative_l2 <- function(candidate, reference) {
  sqrt(sum((candidate - reference)^2)) /
    max(sqrt(sum(reference^2)), 1e-300)
}
assert_aggregate_svd_diagnostics <- function(
  info, q, expected_executed_svd_count, message
) {
  required <- c(
    "aggregate_penalty_root_rank", "aggregate_penalty_root_pivot",
    "aggregate_factor_call_count", "aggregate_b_build_count",
    "aggregate_dstop", "aggregate_penalty_factor_count",
    "aggregate_svd_b_build_count", "aggregate_penalty_root_d2h_count"
  )
  missing <- setdiff(required, names(info))
  assert_true(
    length(missing) == 0L,
    paste0(message, ": missing ", paste(missing, collapse = ","))
  )
  target_count <- length(info$executed_route)
  assert_true(
    target_count > 0L &&
      is.integer(expected_executed_svd_count) &&
      length(expected_executed_svd_count) == 1L &&
      !is.na(expected_executed_svd_count) &&
      expected_executed_svd_count >= 0L &&
      expected_executed_svd_count <= target_count,
    paste(message, "has a valid positive target count and expected SVD count")
  )
  executed_svd <- !is.na(info$executed_route) &
    info$executed_route == "AUGMENTED_SVD"
  assert_true(
    identical(as.integer(sum(executed_svd)), expected_executed_svd_count),
    paste(message, "executed-SVD mask has the exact expected count")
  )
  expected_factor <- as.integer(executed_svd)
  expected_build <- 2L * expected_factor
  pivot_shapes <- vapply(seq_len(target_count), function(index) {
    pivot <- info$aggregate_penalty_root_pivot[[index]]
    if (!executed_svd[[index]]) {
      return(is.integer(pivot) && identical(pivot, integer()))
    }
    is.integer(pivot) && length(pivot) == q &&
      identical(sort(pivot), seq_len(q))
  }, logical(1L))
  assert_true(
    is.integer(info$aggregate_penalty_root_rank) &&
      length(info$aggregate_penalty_root_rank) == target_count &&
      is.list(info$aggregate_penalty_root_pivot) &&
      length(info$aggregate_penalty_root_pivot) == target_count &&
      is.integer(info$aggregate_factor_call_count) &&
      identical(info$aggregate_factor_call_count, expected_factor) &&
      is.integer(info$aggregate_b_build_count) &&
      identical(info$aggregate_b_build_count, expected_build) &&
      is.double(info$aggregate_dstop) &&
      length(info$aggregate_dstop) == target_count &&
      is.integer(info$effective_rank) &&
      length(info$effective_rank) == target_count &&
      all(info$aggregate_penalty_root_rank[executed_svd] >= 0L) &&
      all(info$aggregate_penalty_root_rank[executed_svd] <= q) &&
      all(is.na(info$aggregate_penalty_root_rank[!executed_svd])) &&
      all(info$effective_rank[executed_svd] >= 0L) &&
      all(info$effective_rank[executed_svd] <= q) &&
      all(is.finite(info$aggregate_dstop[executed_svd])) &&
      all(info$aggregate_dstop[executed_svd] >= 0) &&
      all(is.na(info$aggregate_dstop[!executed_svd])) &&
      identical(lengths(info$aggregate_penalty_root_pivot),
                q * expected_factor) &&
      all(pivot_shapes) &&
      identical(info$aggregate_penalty_factor_count,
                as.integer(sum(expected_factor))) &&
      identical(info$aggregate_svd_b_build_count,
                as.integer(sum(expected_build))) &&
      identical(info$aggregate_penalty_root_d2h_count, 0L),
    message
  )
}

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP Phase 3B fixed-sp true-batch diagnostics\n")
  quit(save = "no", status = 0)
}

build_fastkpc_cuda_native(rebuild = TRUE)
assert_true(fastkpc_cuda_available(), "CUDA must be available")

catalog <- fastkpc_full_cuda_open_fixed_sp_catalog(
  file.path("fastkpc", "artifacts", "full_cuda_ci", "oracle_351x48_v1"),
  file.path("fastkpc", "artifacts", "full_cuda_ci",
            "workload_census_351x48_v1"),
  file.path("fastkpc", "artifacts", "full_cuda_ci",
            "prepared_s_contract_v1"),
  file.path("fastkpc", "artifacts", "kpc_tprs_real_zhu",
            "cancer_RD-causalDiscoveryInput.rds")
)
iteration <- fastkpc_full_cuda_fixed_sp_scope(catalog, "iteration")

target_rows_by_setup <- split(
  iteration$target_rows,
  as.character(iteration$target_rows$prepared_s_key_sha256)
)
eligible_keys <- iteration$setup_rows$prepared_s_key_sha256[vapply(
  iteration$setup_rows$prepared_s_key_sha256,
  function(setup_key) {
    setup_row <- iteration$setup_rows[
      iteration$setup_rows$prepared_s_key_sha256 == setup_key,
      , drop = FALSE
    ]
    target_rows <- target_rows_by_setup[[setup_key]]
    nrow(setup_row) == 1L && setup_row$penalty_count[[1L]] > 1L &&
      sum(target_rows$planned_route == "CHOLESKY_BATCHED") >= 3L
  },
  logical(1L)
)]
eligible_keys <- sort(eligible_keys, method = "radix")
assert_true(
  length(eligible_keys) >= 1L,
  "iteration scope contains a multi-penalty setup with three safe targets"
)

setup_key <- eligible_keys[[1L]]
selected_scope <- iteration
selected_scope$setup_rows <- iteration$setup_rows[
  iteration$setup_rows$prepared_s_key_sha256 == setup_key,
  , drop = FALSE
]
selected_scope$target_rows <- iteration$target_rows[
  iteration$target_rows$prepared_s_key_sha256 == setup_key,
  , drop = FALSE
]
setup_rank <- match(setup_key, catalog$setup_index$prepared_s_key_sha256)
assert_true(!is.na(setup_rank), "selected setup belongs to the catalog index")
selected_scope$shard_ids <- as.integer(
  (setup_rank - 1L) %% catalog$catalog_contract$shard_count
)
batch <- fastkpc_full_cuda_fixed_sp_batches(
  catalog, selected_scope
)[[setup_key]]
safe_indices <- which(batch$planned_route == "CHOLESKY_BATCHED")
assert_true(
  length(safe_indices) >= 3L && length(batch$setup$penalty_blocks) > 1L,
  "selected payload preserves three safe targets and multiple penalties"
)

subset_target <- function(source, index) {
  list(
    setup = source$setup,
    target_rows = source$target_rows[index, , drop = FALSE],
    Y = source$Y[, index, drop = FALSE],
    SP = source$SP[, index, drop = FALSE],
    oracle_nullspace_rhs =
      source$oracle_nullspace_rhs[, index, drop = FALSE],
    planned_route = source$planned_route[index],
    condition = source$condition[index],
    prepared_s_key_sha256 = source$prepared_s_key_sha256
  )
}

safe_batch <- subset_target(batch, safe_indices)
safe_count <- length(safe_indices)
dto <- fastkpc_full_cuda_fixed_sp_native_dto(safe_batch$setup)
native_batch <- fastkpc_full_cuda_fixed_sp_native_batch(
  safe_batch, dto
)
assert_true(
  identical(native_batch$target_count, safe_count) &&
    all(native_batch$planned_route == "CHOLESKY_BATCHED") &&
    nrow(native_batch$SP) > 1L,
  "Task 2 submits every safe target from the selected setup"
)

oracle_results <- lapply(
  seq_len(safe_count),
  function(index) {
    fastkpc_mgcv_magic_fixed_sp_from_prepared(
      prepared_setup = safe_batch$setup,
      target_state = list(
        row = safe_batch$target_rows[index, , drop = FALSE],
        y = as.numeric(safe_batch$Y[, index])
      )
    )
  }
)
oracle_coefficients <- vapply(
  oracle_results, `[[`, numeric(dto$coefficient_dim), "coefficients"
)
oracle_fitted <- vapply(oracle_results, `[[`, numeric(dto$n), "fitted")
oracle_residuals <- vapply(
  oracle_results, `[[`, numeric(dto$n), "residuals"
)
oracle_rss <- vapply(
  oracle_results,
  function(result) sum(result$residuals^2),
  numeric(1L)
)

runtime <- fixed_sp_cuda_runtime_create(0L)
handle <- NULL
token <- NULL
partial_factor_fail_token <- NULL
all_factor_fail_token <- NULL
on.exit({
  if (!is.null(token)) {
    try(fixed_sp_cuda_residual_release(token), silent = TRUE)
    try(fixed_sp_cuda_residual_free(token), silent = TRUE)
  }
  if (!is.null(partial_factor_fail_token)) {
    try(
      fixed_sp_cuda_residual_release(partial_factor_fail_token),
      silent = TRUE
    )
    try(
      fixed_sp_cuda_residual_free(partial_factor_fail_token),
      silent = TRUE
    )
  }
  if (!is.null(all_factor_fail_token)) {
    try(
      fixed_sp_cuda_residual_release(all_factor_fail_token),
      silent = TRUE
    )
    try(fixed_sp_cuda_residual_free(all_factor_fail_token), silent = TRUE)
  }
  if (!is.null(handle)) {
    try(fixed_sp_cuda_prepared_free(handle), silent = TRUE)
  }
  if (!is.null(runtime)) {
    try(fixed_sp_cuda_runtime_free(runtime), silent = TRUE)
  }
}, add = TRUE)

fixed_sp_cuda_runtime_reserve(
  runtime, dto$n, dto$null_dim, safe_count, dto$penalty_count,
  as.integer(dto$n + sum(dto$penalty_ranks))
)
handle <- fixed_sp_cuda_prepared_create(runtime, dto)
runtime_info_before_solve <- fixed_sp_cuda_runtime_info(runtime)
token <- fixed_sp_cuda_solve_batch(
  handle, native_batch$Y, native_batch$SP, native_batch$planned_route,
  native_batch$target_keys,
  outputs = c("coefficients", "fitted", "residuals", "rss", "rhs")
)
info <- fixed_sp_cuda_residual_info(token)
runtime_info_after_solve <- fixed_sp_cuda_runtime_info(runtime)

required_fields <- c(
  "native_batch_call",
  "batch_call_count",
  "true_batched_kernel",
  "true_batched_subgroup_count",
  "true_batched_attempted_target_count",
  "true_batched_target_count",
  "cholesky_single_target_count",
  "potrf_batched_call_count",
  "potrs_batched_call_count",
  "target_batch_h2d_call_count",
  "target_h2d_copy_count",
  "target_h2d_bytes",
  "canonical_output_order_exact",
  "planned_route",
  "executed_route",
  "reroute_reason",
  "solver_status"
)
missing_fields <- setdiff(required_fields, names(info))
assert_true(
  length(missing_fields) == 0L,
  paste0(
    "Phase 3B residual info fields missing: ",
    paste(missing_fields, collapse = ", ")
  )
)

assert_true(
  isTRUE(info$native_batch_call),
  "one native batch call"
)
assert_true(
  info$batch_call_count == 1L,
  "one public solve call"
)

assert_true(
  info$true_batched_subgroup_count == 1L,
  "one true-batched subgroup"
)
assert_true(
  info$true_batched_attempted_target_count == safe_count,
  "all targets attempted in batched Cholesky"
)
assert_true(
  info$true_batched_target_count == safe_count,
  "all targets completed in batched Cholesky"
)
assert_true(
  info$cholesky_single_target_count == 0L,
  "no repeated single Cholesky"
)
assert_true(
  info$potrf_batched_call_count == 1L &&
    info$potrs_batched_call_count == 1L,
  "one batched factor and solve call"
)
assert_true(
  isTRUE(info$true_batched_kernel),
  "all-safe multi-target batch is truly batched"
)
assert_true(
  runtime_info_after_solve$cholesky_factor_checkpoint_record_count -
      runtime_info_before_solve$cholesky_factor_checkpoint_record_count == 1L &&
    runtime_info_after_solve$cholesky_factor_checkpoint_wait_count -
      runtime_info_before_solve$cholesky_factor_checkpoint_wait_count == 1L &&
    runtime_info_after_solve$cholesky_solve_checkpoint_record_count -
      runtime_info_before_solve$cholesky_solve_checkpoint_record_count == 1L &&
    runtime_info_after_solve$cholesky_solve_checkpoint_wait_count -
      runtime_info_before_solve$cholesky_solve_checkpoint_wait_count == 1L &&
    runtime_info_after_solve$cuda_device_synchronize_count ==
      runtime_info_before_solve$cuda_device_synchronize_count,
  "normal batch records and waits on exactly two events"
)
assert_true(
  runtime_info_after_solve$workspace_grow_count ==
      runtime_info_before_solve$workspace_grow_count &&
    runtime_info_after_solve$workspace_bytes ==
      runtime_info_before_solve$workspace_bytes &&
    all(c(
      info$cuda_device_allocation_count_during_solve,
      info$cuda_host_allocation_count_during_solve,
      info$stream_create_count_during_solve,
      info$event_create_count_during_solve,
      info$cublas_handle_create_count_during_solve,
      info$cusolver_handle_create_count_during_solve,
      info$per_target_allocation_count_after_warmup,
      info$per_target_handle_create_count,
      info$cpu_fallback_count,
      info$unknown_fallback_count
    ) == 0L),
  "true batch performs no solve-time growth, allocation, or fallback"
)
assert_true(
  info$target_batch_h2d_call_count == 1L,
  "one target batch upload phase"
)
assert_true(
  info$target_h2d_copy_count == 2L,
  "one Y and one SP copy"
)
assert_true(
  info$target_h2d_bytes == 8 * (length(native_batch$Y) +
    length(native_batch$SP)),
  "target H2D byte accounting"
)
assert_true(
  info$rhs_device_build_count == 1L &&
    identical(info$rhs_authority, "cuda-x0-transpose-y") &&
    isTRUE(info$full_cuda_data_plane),
  "one CUDA RHS build"
)
assert_true(
  isTRUE(info$canonical_output_order_exact) &&
    identical(info$target_keys, native_batch$target_keys),
  "canonical output mapping preserves target-key order"
)
assert_true(
  identical(info$planned_route, native_batch$planned_route) &&
    identical(info$executed_route,
              rep("CHOLESKY_BATCHED", safe_count)) &&
    identical(info$reroute_reason, rep("", safe_count)) &&
    identical(info$solver_status,
              rep("OK_CHOLESKY_BATCHED", safe_count)),
  "all safe targets preserve canonical route and status order"
)

shadow <- fixed_sp_cuda_materialize_shadow(
  token,
  outputs = c("coefficients", "fitted", "residuals", "rss", "rhs")
)
output_max_abs_errors <- c(
  coefficients = max(abs(shadow$coefficients - oracle_coefficients)),
  fitted = max(abs(shadow$fitted - oracle_fitted)),
  residuals = max(abs(shadow$residuals - oracle_residuals)),
  rss = max(abs(shadow$rss - oracle_rss)),
  rhs = max(abs(shadow$cuda_nullspace_rhs -
                  safe_batch$oracle_nullspace_rhs))
)
numerical_max_abs_error <- max(output_max_abs_errors)
fitted_column_max_abs_errors <- vapply(
  seq_len(safe_count),
  function(index) max(abs(
    shadow$fitted[, index] - oracle_fitted[, index]
  )),
  numeric(1L)
)
fitted_column_relative_l2_errors <- vapply(
  seq_len(safe_count),
  function(index) relative_l2(
    shadow$fitted[, index], oracle_fitted[, index]
  ),
  numeric(1L)
)
residual_column_max_abs_errors <- vapply(
  seq_len(safe_count),
  function(index) max(abs(
    shadow$residuals[, index] - oracle_residuals[, index]
  )),
  numeric(1L)
)
residual_column_relative_l2_errors <- vapply(
  seq_len(safe_count),
  function(index) relative_l2(
    shadow$residuals[, index], oracle_residuals[, index]
  ),
  numeric(1L)
)
numerical_max_relative_l2_error <- max(c(
  fitted_column_relative_l2_errors,
  residual_column_relative_l2_errors
))
assert_true(
  identical(dim(shadow$coefficients),
            c(dto$coefficient_dim, safe_count)) &&
    identical(dim(shadow$fitted), c(dto$n, safe_count)) &&
    identical(dim(shadow$residuals), c(dto$n, safe_count)) &&
    identical(length(shadow$rss), safe_count) &&
    identical(dim(shadow$cuda_nullspace_rhs),
              c(dto$null_dim, safe_count)) &&
    all(output_max_abs_errors[c(
      "coefficients", "fitted", "residuals", "rss"
    )] < 1e-7) && output_max_abs_errors[["rhs"]] < 1e-12 &&
    all(fitted_column_max_abs_errors < 1e-7) &&
    all(fitted_column_relative_l2_errors < 1e-7) &&
    all(residual_column_max_abs_errors < 1e-7) &&
    all(residual_column_relative_l2_errors < 1e-7),
  paste0(
    "all canonical safe output columns match the fixed-sp oracle; max=",
    format(numerical_max_abs_error, digits = 17L),
    "; relative_l2=",
    format(numerical_max_relative_l2_error, digits = 17L)
  )
)

first_hashes <- c(
  coefficients = fastkpc_full_cuda_census_metadata_hash(
    shadow$coefficients
  ),
  fitted = fastkpc_full_cuda_census_metadata_hash(shadow$fitted),
  residuals = fastkpc_full_cuda_census_metadata_hash(shadow$residuals),
  rss = fastkpc_full_cuda_census_metadata_hash(shadow$rss)
)
fixed_sp_cuda_residual_release(token)
fixed_sp_cuda_residual_free(token)
token <- NULL

token <- fixed_sp_cuda_solve_batch(
  handle, native_batch$Y, native_batch$SP, native_batch$planned_route,
  native_batch$target_keys,
  outputs = c("coefficients", "fitted", "residuals", "rss", "rhs")
)
shadow_repeat <- fixed_sp_cuda_materialize_shadow(
  token,
  outputs = c("coefficients", "fitted", "residuals", "rss")
)
repeat_hashes <- c(
  coefficients = fastkpc_full_cuda_census_metadata_hash(
    shadow_repeat$coefficients
  ),
  fitted = fastkpc_full_cuda_census_metadata_hash(shadow_repeat$fitted),
  residuals = fastkpc_full_cuda_census_metadata_hash(
    shadow_repeat$residuals
  ),
  rss = fastkpc_full_cuda_census_metadata_hash(shadow_repeat$rss)
)
assert_true(
  identical(first_hashes, repeat_hashes),
  "same-environment batched output hashes"
)
fixed_sp_cuda_residual_release(token)
fixed_sp_cuda_residual_free(token)
token <- NULL

batch_finalize_fields <- c(
  "coefficient_batch_finalize_call_count",
  "fitted_batch_finalize_call_count",
  "residual_rss_batch_finalize_call_count",
  "per_target_output_finalize_call_count",
  "batch_output_finalized_target_count"
)
missing_batch_finalize_fields <- setdiff(batch_finalize_fields, names(info))
assert_true(
  length(missing_batch_finalize_fields) == 0L,
  paste0(
    "Phase 3B batch-finalization fields missing: ",
    paste(missing_batch_finalize_fields, collapse = ", ")
  )
)
assert_true(
  identical(info$coefficient_batch_finalize_call_count, 1L) &&
    identical(info$fitted_batch_finalize_call_count, 1L) &&
    identical(info$residual_rss_batch_finalize_call_count, 1L) &&
    identical(info$per_target_output_finalize_call_count, 0L) &&
    identical(info$batch_output_finalized_target_count, safe_count),
  "all-output solve uses one canonical batch finalizer per output family"
)

assert_output_finalize_counts <- function(
    outputs, coefficient, fitted, residual_rss, finalized, label) {
  mask_token <- fixed_sp_cuda_solve_batch(
    handle, native_batch$Y, native_batch$SP, native_batch$planned_route,
    native_batch$target_keys, outputs = outputs
  )
  released <- FALSE
  freed <- FALSE
  on.exit({
    if (!released) {
      try(fixed_sp_cuda_residual_release(mask_token), silent = TRUE)
    }
    if (!freed) {
      try(fixed_sp_cuda_residual_free(mask_token), silent = TRUE)
    }
  }, add = TRUE)
  mask_info <- fixed_sp_cuda_residual_info(mask_token)
  assert_true(
    identical(mask_info$coefficient_batch_finalize_call_count,
              as.integer(coefficient)) &&
      identical(mask_info$fitted_batch_finalize_call_count,
                as.integer(fitted)) &&
      identical(mask_info$residual_rss_batch_finalize_call_count,
                as.integer(residual_rss)) &&
      identical(mask_info$per_target_output_finalize_call_count, 0L) &&
      identical(mask_info$batch_output_finalized_target_count,
                as.integer(finalized)),
    paste0(label, " output-mask batch-finalization accounting")
  )
  fixed_sp_cuda_residual_release(mask_token)
  released <- TRUE
  fixed_sp_cuda_residual_free(mask_token)
  freed <- TRUE
  invisible(mask_info)
}

assert_output_finalize_counts(
  "coefficients", 1L, 0L, 0L, safe_count, "coefficient-only"
)
assert_output_finalize_counts(
  "fitted", 0L, 1L, 0L, safe_count, "fitted-only"
)
assert_output_finalize_counts(
  "residuals", 0L, 1L, 1L, safe_count, "residual-only"
)
assert_output_finalize_counts(
  "rss", 0L, 1L, 1L, safe_count, "RSS-only"
)
assert_output_finalize_counts(
  "rhs", 0L, 0L, 0L, safe_count, "RHS-only"
)

clear_forced_info <- function() {
  try(
    invisible(.Call(
      "C_fixed_sp_cuda_test_force_next_potrf_info",
      integer(), PACKAGE = "fastkpc_cuda"
    )),
    silent = TRUE
  )
  try(
    invisible(.Call(
      "C_fixed_sp_cuda_test_force_next_potrs_info",
      0L, PACKAGE = "fastkpc_cuda"
    )),
    silent = TRUE
  )
}
on.exit(clear_forced_info(), add = TRUE)

middle_ordinal <- as.integer((safe_count + 1L) %/% 2L)
partial_potrf_info <- integer(safe_count)
partial_potrf_info[[middle_ordinal]] <- 1L
invisible(.Call(
  "C_fixed_sp_cuda_test_force_next_potrf_info",
  partial_potrf_info, PACKAGE = "fastkpc_cuda"
))
partial_runtime_before <- fixed_sp_cuda_runtime_info(runtime)
partial_factor_fail_token <- fixed_sp_cuda_solve_batch(
  handle, native_batch$Y, native_batch$SP, native_batch$planned_route,
  native_batch$target_keys,
  outputs = c("coefficients", "fitted", "residuals", "rss", "rhs")
)
partial_factor_fail_info <- fixed_sp_cuda_residual_info(
  partial_factor_fail_token
)
partial_runtime_after <- fixed_sp_cuda_runtime_info(runtime)
expected_partial_status <- rep("OK_CHOLESKY_BATCHED", safe_count)
expected_partial_status[[middle_ordinal]] <-
  "OK_AUGMENTED_SVD"
expected_partial_reason <- rep("", safe_count)
expected_partial_reason[[middle_ordinal]] <-
  "CHOLESKY_NON_POSITIVE_PIVOT"
expected_partial_route <- rep("CHOLESKY_BATCHED", safe_count)
expected_partial_route[[middle_ordinal]] <- "AUGMENTED_SVD"
assert_true(
  partial_factor_fail_info$potrf_batched_call_count == 1L &&
    partial_factor_fail_info$potrs_batched_call_count == 1L &&
    partial_factor_fail_info$true_batched_attempted_target_count ==
      safe_count &&
    partial_factor_fail_info$true_batched_target_count == safe_count - 1L &&
    partial_factor_fail_info$cholesky_to_svd_count == 1L &&
    partial_factor_fail_info$stable_reroute_count == 1L &&
    partial_factor_fail_info$executed_cholesky_target_count ==
      safe_count - 1L &&
    partial_factor_fail_info$executed_svd_target_count == 1L &&
    partial_factor_fail_info$coefficient_batch_finalize_call_count == 1L &&
    partial_factor_fail_info$fitted_batch_finalize_call_count == 1L &&
    partial_factor_fail_info$residual_rss_batch_finalize_call_count == 1L &&
    partial_factor_fail_info$per_target_output_finalize_call_count == 1L &&
    partial_factor_fail_info$batch_output_finalized_target_count ==
      safe_count &&
    !isTRUE(partial_factor_fail_info$true_batched_kernel) &&
    isTRUE(partial_factor_fail_info$canonical_output_order_exact) &&
    identical(
      partial_factor_fail_info$solver_status,
      expected_partial_status
    ) &&
    identical(
      partial_factor_fail_info$reroute_reason,
      expected_partial_reason
    ) &&
    identical(partial_factor_fail_info$executed_route,
              expected_partial_route),
  "one middle factor failure preserves canonical batch status accounting"
)
assert_aggregate_svd_diagnostics(
  partial_factor_fail_info, dto$null_dim, 1L,
  "partial Cholesky-to-SVD reroute aggregate lifecycle"
)
logical_augmented_rows <- as.integer(dto$n + sum(dto$penalty_ranks))
assert_true(
  partial_runtime_before$augmented_workspace_bytes ==
      8 * max(logical_augmented_rows, dto$n + dto$null_dim) * dto$null_dim &&
    identical(partial_runtime_after$workspace_grow_count,
              partial_runtime_before$workspace_grow_count) &&
    identical(partial_runtime_after$workspace_bytes,
              partial_runtime_before$workspace_bytes) &&
    identical(partial_runtime_after$stable_workspace_grow_count,
              partial_runtime_before$stable_workspace_grow_count) &&
    partial_runtime_after$svd_checkpoint_record_count -
      partial_runtime_before$svd_checkpoint_record_count == 1L &&
    partial_runtime_after$svd_checkpoint_wait_count -
      partial_runtime_before$svd_checkpoint_wait_count == 1L &&
    identical(partial_runtime_after$cuda_device_synchronize_count,
              partial_runtime_before$cuda_device_synchronize_count),
  "partial reroute fits n + q behind the logical QR reserve without growth"
)
partial_shadow <- fixed_sp_cuda_materialize_shadow(
  partial_factor_fail_token,
  outputs = c("coefficients", "fitted", "residuals", "rss", "rhs")
)
partial_successful <- seq_len(safe_count)
partial_output_max_abs_errors <- c(
  coefficients = max(abs(
    partial_shadow$coefficients[, partial_successful, drop = FALSE] -
      oracle_coefficients[, partial_successful, drop = FALSE]
  )),
  fitted = max(abs(
    partial_shadow$fitted[, partial_successful, drop = FALSE] -
      oracle_fitted[, partial_successful, drop = FALSE]
  )),
  residuals = max(abs(
    partial_shadow$residuals[, partial_successful, drop = FALSE] -
      oracle_residuals[, partial_successful, drop = FALSE]
  )),
  rss = max(abs(
    partial_shadow$rss[partial_successful] -
      oracle_rss[partial_successful]
  )),
  rhs = max(abs(
    partial_shadow$cuda_nullspace_rhs[
      , partial_successful, drop = FALSE
    ] - safe_batch$oracle_nullspace_rhs[
      , partial_successful, drop = FALSE
    ]
  ))
)
partial_numerical_max_abs_error <- max(partial_output_max_abs_errors)
assert_true(
  all(partial_output_max_abs_errors[c(
    "coefficients", "fitted", "residuals", "rss"
  )] < 1e-7) && partial_output_max_abs_errors[["rhs"]] < 1e-12,
  paste0(
    "compacted successful columns match canonical oracles; max=",
    format(partial_numerical_max_abs_error, digits = 17L)
  )
)
assert_true(
  all(is.finite(partial_shadow$coefficients[, middle_ordinal])) &&
    all(is.finite(partial_shadow$fitted[, middle_ordinal])) &&
    all(is.finite(partial_shadow$residuals[, middle_ordinal])) &&
    is.finite(partial_shadow$rss[[middle_ordinal]]) &&
    all(is.finite(partial_shadow$cuda_nullspace_rhs[, middle_ordinal])),
  "rerouted canonical column exposes the successful SVD output"
)
fixed_sp_cuda_residual_release(partial_factor_fail_token)
fixed_sp_cuda_residual_free(partial_factor_fail_token)
partial_factor_fail_token <- NULL
invisible(.Call(
  "C_fixed_sp_cuda_test_force_next_potrf_info",
  integer(), PACKAGE = "fastkpc_cuda"
))

negative_potrf_info <- integer(safe_count)
negative_potrf_info[[middle_ordinal]] <- -1L
invisible(.Call(
  "C_fixed_sp_cuda_test_force_next_potrf_info",
  negative_potrf_info, PACKAGE = "fastkpc_cuda"
))
negative_potrf_error <- tryCatch(
  fixed_sp_cuda_solve_batch(
    handle, native_batch$Y, native_batch$SP,
    native_batch$planned_route, native_batch$target_keys
  ),
  error = identity
)
invisible(.Call(
  "C_fixed_sp_cuda_test_force_next_potrf_info",
  integer(), PACKAGE = "fastkpc_cuda"
))
if (!inherits(negative_potrf_error, "error")) {
  try(fixed_sp_cuda_residual_release(negative_potrf_error), silent = TRUE)
  try(fixed_sp_cuda_residual_free(negative_potrf_error), silent = TRUE)
  fail("negative potrf info returned a partial token")
}
assert_true(
  grepl(
    "Phase 3B batched potrf info contains a negative value",
    conditionMessage(negative_potrf_error), fixed = TRUE
  ) &&
    identical(
      fixed_sp_cuda_prepared_info(handle)$output_slot_state,
      "free"
    ),
  "negative potrf info is a whole-call error that restores the output slot"
)
token <- fixed_sp_cuda_solve_batch(
  handle, native_batch$Y, native_batch$SP, native_batch$planned_route,
  native_batch$target_keys
)
negative_recovery_info <- fixed_sp_cuda_residual_info(token)
assert_true(
  negative_recovery_info$true_batched_target_count == safe_count &&
    negative_recovery_info$cholesky_to_svd_count == 0L &&
    negative_recovery_info$stable_reroute_count == 0L &&
    negative_recovery_info$executed_cholesky_target_count == safe_count &&
    isTRUE(negative_recovery_info$true_batched_kernel) &&
    all(negative_recovery_info$reroute_reason == "") &&
    all(negative_recovery_info$solver_status == "OK_CHOLESKY_BATCHED"),
  "negative potrf failure leaves no partial reroute and slot is reusable"
)
fixed_sp_cuda_residual_release(token)
fixed_sp_cuda_residual_free(token)
token <- NULL

invisible(.Call(
  "C_fixed_sp_cuda_test_force_next_potrf_info",
  rep.int(1L, safe_count), PACKAGE = "fastkpc_cuda"
))
all_factor_runtime_before <- fixed_sp_cuda_runtime_info(runtime)
all_factor_fail_token <- fixed_sp_cuda_solve_batch(
  handle, native_batch$Y, native_batch$SP, native_batch$planned_route,
  native_batch$target_keys
)
all_factor_fail_info <- fixed_sp_cuda_residual_info(all_factor_fail_token)
all_factor_runtime_after <- fixed_sp_cuda_runtime_info(runtime)
assert_true(
  all_factor_fail_info$potrf_batched_call_count == 1L &&
    all_factor_fail_info$potrs_batched_call_count == 0L &&
    all_factor_fail_info$true_batched_attempted_target_count == safe_count &&
    all_factor_fail_info$true_batched_target_count == 0L &&
    all_factor_fail_info$cholesky_single_target_count == 0L &&
    all_factor_fail_info$cholesky_to_svd_count == safe_count &&
    all_factor_fail_info$stable_reroute_count == safe_count &&
    all_factor_fail_info$executed_cholesky_target_count == 0L &&
    all_factor_fail_info$executed_svd_target_count == safe_count &&
    all_factor_fail_info$coefficient_batch_finalize_call_count == 0L &&
    all_factor_fail_info$fitted_batch_finalize_call_count == 0L &&
    all_factor_fail_info$residual_rss_batch_finalize_call_count == 0L &&
    all_factor_fail_info$per_target_output_finalize_call_count == safe_count &&
    all_factor_fail_info$batch_output_finalized_target_count == safe_count &&
    !isTRUE(all_factor_fail_info$true_batched_kernel) &&
    all(all_factor_fail_info$reroute_reason ==
          "CHOLESKY_NON_POSITIVE_PIVOT") &&
    all(all_factor_fail_info$executed_route == "AUGMENTED_SVD") &&
    all(all_factor_fail_info$solver_status ==
          "OK_AUGMENTED_SVD"),
  "all factor failures reroute and skip zero-sized batched solve"
)
assert_aggregate_svd_diagnostics(
  all_factor_fail_info, dto$null_dim, as.integer(safe_count),
  "all-target Cholesky-to-SVD reroute aggregate lifecycle"
)
assert_true(
  identical(all_factor_runtime_after$workspace_grow_count,
            all_factor_runtime_before$workspace_grow_count) &&
    identical(all_factor_runtime_after$workspace_bytes,
              all_factor_runtime_before$workspace_bytes) &&
    identical(all_factor_runtime_after$stable_workspace_grow_count,
              all_factor_runtime_before$stable_workspace_grow_count) &&
    all_factor_runtime_after$svd_checkpoint_record_count -
      all_factor_runtime_before$svd_checkpoint_record_count == 1L &&
    all_factor_runtime_after$svd_checkpoint_wait_count -
      all_factor_runtime_before$svd_checkpoint_wait_count == 1L &&
    identical(all_factor_runtime_after$cuda_device_synchronize_count,
              all_factor_runtime_before$cuda_device_synchronize_count),
  "all-target reroute uses one SVD checkpoint and no solve-time growth"
)
fixed_sp_cuda_residual_release(all_factor_fail_token)
fixed_sp_cuda_residual_free(all_factor_fail_token)
all_factor_fail_token <- NULL
invisible(.Call(
  "C_fixed_sp_cuda_test_force_next_potrf_info",
  integer(), PACKAGE = "fastkpc_cuda"
))

invisible(.Call(
  "C_fixed_sp_cuda_test_force_next_potrs_info",
  -1L, PACKAGE = "fastkpc_cuda"
))
potrs_error <- tryCatch(
  fixed_sp_cuda_solve_batch(
    handle, native_batch$Y, native_batch$SP, native_batch$planned_route,
    native_batch$target_keys
  ),
  error = identity
)
invisible(.Call(
  "C_fixed_sp_cuda_test_force_next_potrs_info",
  0L, PACKAGE = "fastkpc_cuda"
))
if (!inherits(potrs_error, "error")) {
  try(fixed_sp_cuda_residual_release(potrs_error), silent = TRUE)
  try(fixed_sp_cuda_residual_free(potrs_error), silent = TRUE)
  fail("potrs scalar info returned a partial token")
}
assert_true(
  grepl(
      "Phase 3B batched potrs info",
      conditionMessage(potrs_error), fixed = TRUE
    ),
  "potrs scalar info is a whole-batch failure"
)

token <- fixed_sp_cuda_solve_batch(
  handle, native_batch$Y, native_batch$SP, native_batch$planned_route,
  native_batch$target_keys
)
recovery_info <- fixed_sp_cuda_residual_info(token)
assert_true(
  recovery_info$true_batched_target_count == safe_count &&
    recovery_info$cholesky_to_svd_count == 0L &&
    recovery_info$stable_reroute_count == 0L &&
    all(recovery_info$solver_status == "OK_CHOLESKY_BATCHED"),
  "potrs batch failure leaves the output slot reusable"
)
fixed_sp_cuda_residual_release(token)
fixed_sp_cuda_residual_free(token)
token <- NULL

fixed_sp_cuda_prepared_free(handle)
handle <- NULL
fixed_sp_cuda_runtime_free(runtime)
runtime <- NULL

multi_target_nonfinite_info <- local({
  sha <- function(character) strrep(character, 64L)
  overflow_dto <- list(
    schema_version = "full-cuda-ci-prepared-s-native-dto-v1",
    dataset_sha256 = sha("1"),
    prepared_s_key_sha256 = sha("2"),
    same_S_group_id = sha("3"),
    phase1_setup_fingerprint = sha("4"),
    provider_fingerprint = sha("5"),
    semantic_fingerprint = sha("6"),
    representation_fingerprint = sha("7"),
    prepared_s_setup_schema_version = "full-cuda-ci-prepared-s-setup-v1",
    native_dto_schema_version = "full-cuda-ci-prepared-s-native-dto-v1",
    data_p = 48L,
    n = 2L,
    coefficient_dim = 1L,
    null_dim = 1L,
    penalty_count = 1L,
    X = matrix(.Machine$double.xmax, nrow = 2L, ncol = 1L),
    constraint_mode = "identity",
    constraint_nullspace = NULL,
    gram_matrix = matrix(1, nrow = 1L, ncol = 1L),
    nullspace_gram_matrix = NULL,
    penalty_blocks = list(penalty_1 = matrix(0, 1L, 1L)),
    penalty_offsets_zero_based = 0L,
    penalty_ranks = 0L,
    penalty_sp_indices_zero_based = 0L,
    penalty_sp_labels = "sp1",
    H = NULL,
    weights_policy = "none-or-unit",
    offset_policy = "none-or-zero"
  )
  overflow_runtime <- fixed_sp_cuda_runtime_create(0L)
  overflow_handle <- NULL
  overflow_token <- NULL
  on.exit({
    if (!is.null(overflow_token)) {
      try(fixed_sp_cuda_residual_release(overflow_token), silent = TRUE)
      try(fixed_sp_cuda_residual_free(overflow_token), silent = TRUE)
    }
    if (!is.null(overflow_handle)) {
      try(fixed_sp_cuda_prepared_free(overflow_handle), silent = TRUE)
    }
    try(fixed_sp_cuda_runtime_free(overflow_runtime), silent = TRUE)
  }, add = TRUE)
  fixed_sp_cuda_runtime_reserve(
    overflow_runtime, 2L, 1L, 2L, 1L, 3L
  )
  overflow_handle <- fixed_sp_cuda_prepared_create(
    overflow_runtime, overflow_dto
  )
  overflow_token <- fixed_sp_cuda_solve_batch(
    overflow_handle,
    matrix(2, nrow = 2L, ncol = 2L),
    matrix(0, nrow = 1L, ncol = 2L),
    rep("CHOLESKY_BATCHED", 2L),
    c(sha("8"), sha("9")),
    outputs = c("coefficients", "fitted", "residuals", "rss", "rhs")
  )
  overflow_info <- fixed_sp_cuda_residual_info(overflow_token)
  fixed_sp_cuda_residual_release(overflow_token)
  fixed_sp_cuda_residual_free(overflow_token)
  overflow_token <- NULL
  fixed_sp_cuda_prepared_free(overflow_handle)
  overflow_handle <- NULL
  fixed_sp_cuda_runtime_free(overflow_runtime)
  overflow_runtime <- NULL
  overflow_info
})
assert_true(
  identical(
    multi_target_nonfinite_info$solver_status,
    rep("ERR_NONFINITE_OUTPUT", 2L)
  ) &&
    multi_target_nonfinite_info$true_batched_target_count == 0L &&
    multi_target_nonfinite_info$batch_output_finalized_target_count == 0L &&
    !isTRUE(multi_target_nonfinite_info$true_batched_kernel),
  "true-batched count and flag follow final per-target output statuses"
)

cat(
  "PASS Phase 3B fixed-sp batched output finalization; max abs error: ",
  format(numerical_max_abs_error, digits = 17L),
  "; max relative L2: ",
  format(numerical_max_relative_l2_error, digits = 17L), "\n",
  "exact repeat hashes: ", paste(first_hashes, collapse = ", "), "\n",
  "batch finalize counts: ",
  info$coefficient_batch_finalize_call_count, "/",
  info$fitted_batch_finalize_call_count, "/",
  info$residual_rss_batch_finalize_call_count,
  "; per-target=", info$per_target_output_finalize_call_count,
  "; targets=", info$batch_output_finalized_target_count, "\n",
  "forced potrf reroutes: ",
  all_factor_fail_info$cholesky_to_svd_count, "/", safe_count,
  "; partial compaction max abs error: ",
  format(partial_numerical_max_abs_error, digits = 17L),
  "; forced potrs error: ", conditionMessage(potrs_error), "\n",
  sep = ""
)
