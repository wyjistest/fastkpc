source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
is_integer_scalar <- function(value) {
  is.integer(value) && length(value) == 1L && !is.na(value)
}
assert_integer_scalar <- function(value, expected, message) {
  assert_true(
    is_integer_scalar(value) && identical(value, as.integer(expected)),
    message
  )
}
assert_double_scalar <- function(value, expected, message) {
  assert_true(
    is.double(value) && length(value) == 1L && is.finite(value) &&
      identical(value, as.double(expected)),
    message
  )
}
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
  invisible(executed_svd)
}

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP Phase 3B fixed-sp mixed-batch semantics\n")
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
    if (nrow(setup_row) != 1L || is.null(target_rows)) return(FALSE)
    safe <- which(target_rows$planned_route == "CHOLESKY_BATCHED")
    stable <- which(target_rows$planned_route != "CHOLESKY_BATCHED")
    length(safe) >= 2L && length(stable) >= 1L &&
      !identical(safe, seq_len(length(safe)))
  },
  logical(1L)
)]
eligible_keys <- sort(eligible_keys, method = "radix")
assert_true(
  length(eligible_keys) >= 1L,
  paste0(
    "iteration scope contains a mixed setup with two safe targets and ",
    "a non-prefix canonical partition"
  )
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
dto <- fastkpc_full_cuda_fixed_sp_native_dto(batch$setup)
native_batch <- fastkpc_full_cuda_fixed_sp_native_batch(batch, dto)

safe <- which(native_batch$planned_route == "CHOLESKY_BATCHED")
stable <- which(native_batch$planned_route != "CHOLESKY_BATCHED")
target_count <- native_batch$target_count
assert_true(
  length(safe) >= 2L && length(stable) >= 1L &&
    !identical(safe, seq_len(length(safe))) &&
    identical(native_batch$target_keys, batch$target_rows$residual_key_sha256),
  "selected native batch preserves a nontrivial mixed canonical order"
)

oracle_results <- lapply(
  safe,
  function(index) {
    fastkpc_mgcv_magic_fixed_sp_from_prepared(
      prepared_setup = batch$setup,
      target_state = list(
        row = batch$target_rows[index, , drop = FALSE],
        y = as.numeric(batch$Y[, index])
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
oracle_rhs <- batch$oracle_nullspace_rhs[, safe, drop = FALSE]

runtime <- fixed_sp_cuda_runtime_create(0L)
handle <- NULL
token <- NULL
on.exit({
  if (!is.null(token)) {
    try(fixed_sp_cuda_residual_release(token), silent = TRUE)
    try(fixed_sp_cuda_residual_free(token), silent = TRUE)
  }
  if (!is.null(handle)) {
    try(fixed_sp_cuda_prepared_free(handle), silent = TRUE)
  }
  if (!is.null(runtime)) {
    try(fixed_sp_cuda_runtime_free(runtime), silent = TRUE)
  }
}, add = TRUE)

fixed_sp_cuda_runtime_reserve(
  runtime, dto$n, dto$null_dim, target_count, dto$penalty_count,
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
prepared_info_after_solve <- fixed_sp_cuda_prepared_info(handle)

required_resource_info_fields <- c(
  "invalid_output_init_count",
  "output_slot_release_count",
  "resource_snapshot_captured",
  "resource_instrumentation_version",
  "resource_allocation_count_before_solve",
  "resource_allocation_count_after_solve",
  "resource_handle_create_count_before_solve",
  "resource_handle_create_count_after_solve",
  "cuda_device_allocation_count_during_solve",
  "cuda_host_allocation_count_during_solve",
  "stream_create_count_during_solve",
  "event_create_count_during_solve",
  "cublas_handle_create_count_during_solve",
  "cusolver_handle_create_count_during_solve",
  "per_target_allocation_count_after_warmup",
  "per_target_handle_create_count",
  "implicit_residual_d2h_count",
  "cpu_fallback_count",
  "unknown_fallback_count",
  "shadow_materialize_call_count",
  "shadow_materialize_target_count",
  "shadow_d2h_bytes"
)
required_runtime_resource_fields <- c(
  "workspace_grow_count",
  "workspace_bytes",
  "cuda_device_synchronize_count",
  "cholesky_factor_checkpoint_record_count",
  "cholesky_factor_checkpoint_wait_count",
  "cholesky_solve_checkpoint_record_count",
  "cholesky_solve_checkpoint_wait_count",
  "svd_checkpoint_record_count", "svd_checkpoint_wait_count",
  "augmented_workspace_bytes"
)
missing_resource_info_fields <- setdiff(
  required_resource_info_fields, names(info)
)
missing_runtime_resource_fields <- setdiff(
  required_runtime_resource_fields,
  intersect(names(runtime_info_before_solve), names(runtime_info_after_solve))
)
assert_true(
  length(missing_resource_info_fields) == 0L &&
    length(missing_runtime_resource_fields) == 0L,
  paste0(
    "mixed resource diagnostics are fail-closed; residual missing=",
    paste(missing_resource_info_fields, collapse = ","),
    "; runtime missing=",
    paste(missing_runtime_resource_fields, collapse = ",")
  )
)
assert_true(
  "output_slot_leased" %in% names(prepared_info_after_solve) &&
    identical(prepared_info_after_solve$output_slot_leased, TRUE),
  "mixed solve retains the prepared output-slot lease"
)

expected_status <- rep("ERR_STABLE_PATH_NOT_IMPLEMENTED", target_count)
expected_status[safe] <- "OK_CHOLESKY_BATCHED"
expected_status[native_batch$planned_route == "AUGMENTED_QR"] <-
  "OK_AUGMENTED_QR"
expected_status[native_batch$planned_route == "AUGMENTED_SVD"] <-
  "OK_AUGMENTED_SVD"
expected_executed_route <- native_batch$planned_route
planned_counts <- c(
  CHOLESKY_BATCHED = sum(native_batch$planned_route == "CHOLESKY_BATCHED"),
  AUGMENTED_QR = sum(native_batch$planned_route == "AUGMENTED_QR"),
  AUGMENTED_SVD = sum(native_batch$planned_route == "AUGMENTED_SVD")
)
executed_counts <- c(
  CHOLESKY_BATCHED = info$executed_cholesky_target_count,
  AUGMENTED_QR = info$executed_qr_target_count,
  AUGMENTED_SVD = info$executed_svd_target_count
)

assert_true(
  isTRUE(info$native_batch_call) &&
    identical(info$batch_call_count, 1L) &&
    identical(info$output_slot_acquire_count, 1L) &&
    identical(info$true_batched_subgroup_count, 1L) &&
    identical(info$true_batched_attempted_target_count,
              as.integer(length(safe))) &&
    identical(info$true_batched_target_count,
              as.integer(length(safe))) &&
    identical(info$potrf_batched_call_count, 1L) &&
    identical(info$potrs_batched_call_count, 1L) &&
    identical(info$cholesky_single_target_count, 0L) &&
    identical(info$true_batched_kernel, FALSE),
  "mixed batch executes one true-batched safe subgroup truthfully"
)
assert_true(
  identical(info$solver_status, expected_status) &&
    identical(info$executed_route, expected_executed_route) &&
    identical(info$planned_route, native_batch$planned_route) &&
    identical(info$target_keys, native_batch$target_keys) &&
    identical(info$reroute_reason, rep("", target_count)) &&
    isTRUE(info$canonical_output_order_exact),
  "mixed route metadata and statuses preserve public canonical order"
)
assert_true(
  identical(info$planned_cholesky_target_count,
            as.integer(planned_counts[["CHOLESKY_BATCHED"]])) &&
    identical(info$planned_qr_target_count,
              as.integer(planned_counts[["AUGMENTED_QR"]])) &&
    identical(info$planned_svd_target_count,
              as.integer(planned_counts[["AUGMENTED_SVD"]])) &&
    identical(executed_counts,
              c(CHOLESKY_BATCHED = as.integer(length(safe)),
                AUGMENTED_QR =
                  as.integer(planned_counts[["AUGMENTED_QR"]]),
                AUGMENTED_SVD =
                  as.integer(planned_counts[["AUGMENTED_SVD"]]))) &&
    sum(planned_counts) == target_count &&
    sum(executed_counts) == target_count &&
    !anyNA(info$executed_route) &&
    identical(info$stable_reroute_count, 0L) &&
    identical(info$cholesky_to_svd_count, 0L) &&
    identical(info$qr_to_svd_count, 0L),
  "mixed planned and executed route counts conserve targets"
)
executed_svd <- assert_aggregate_svd_diagnostics(
  info, dto$null_dim,
  as.integer(executed_counts[["AUGMENTED_SVD"]]),
  "mixed declared/rerouted SVD aggregate lifecycle"
)
assert_true(
  identical(info$coefficient_batch_finalize_call_count, 1L) &&
    identical(info$fitted_batch_finalize_call_count, 1L) &&
    identical(info$residual_rss_batch_finalize_call_count, 1L) &&
    identical(info$per_target_output_finalize_call_count,
              as.integer(length(stable))) &&
    identical(info$batch_output_finalized_target_count,
              as.integer(target_count)),
  "mixed batch finalizes batched and stable targets canonically"
)
expected_target_h2d_bytes <- 8 * as.double(
  length(native_batch$Y) + length(native_batch$SP)
)
assert_true(
  identical(info$target_batch_h2d_call_count, 1L) &&
    identical(info$target_h2d_copy_count, 2L) &&
    identical(info$rhs_device_build_count, 1L) &&
    identical(info$rhs_authority, "cuda-x0-transpose-y") &&
    isTRUE(info$full_cuda_data_plane),
  "mixed batch uses one Y/SP upload pair and one CUDA RHS build"
)
assert_double_scalar(
  info$target_h2d_bytes, expected_target_h2d_bytes,
  "mixed batch uploads the exact double byte count for Y and SP"
)
assert_true(
  is_integer_scalar(runtime_info_before_solve$workspace_grow_count) &&
    runtime_info_before_solve$workspace_grow_count >= 0L,
  "pre-solve workspace growth count is one non-negative integer scalar"
)
assert_integer_scalar(
  runtime_info_after_solve$workspace_grow_count,
  runtime_info_before_solve$workspace_grow_count,
  "mixed solve does not grow the workspace"
)
assert_true(
  is.double(runtime_info_before_solve$workspace_bytes) &&
    length(runtime_info_before_solve$workspace_bytes) == 1L &&
    is.finite(runtime_info_before_solve$workspace_bytes) &&
    runtime_info_before_solve$workspace_bytes >= 0,
  "pre-solve workspace bytes is one non-negative finite double scalar"
)
assert_double_scalar(
  runtime_info_after_solve$workspace_bytes,
  runtime_info_before_solve$workspace_bytes,
  "mixed solve preserves workspace bytes"
)
logical_augmented_rows <- as.integer(dto$n + sum(dto$penalty_ranks))
assert_double_scalar(
  runtime_info_before_solve$augmented_workspace_bytes,
  8 * max(logical_augmented_rows, dto$n + dto$null_dim) * dto$null_dim,
  "mixed logical QR reserve includes internal n + q SVD capacity"
)
assert_true(
  runtime_info_after_solve$svd_checkpoint_record_count -
      runtime_info_before_solve$svd_checkpoint_record_count ==
        as.integer(any(executed_svd)) &&
    runtime_info_after_solve$svd_checkpoint_wait_count -
      runtime_info_before_solve$svd_checkpoint_wait_count ==
        as.integer(any(executed_svd)),
  "mixed batch uses one compact SVD checkpoint iff it executes SVD targets"
)
assert_true(
  is_integer_scalar(
    runtime_info_before_solve$cuda_device_synchronize_count
  ) && runtime_info_before_solve$cuda_device_synchronize_count >= 0L,
  "pre-solve device synchronize count is one non-negative integer scalar"
)
assert_integer_scalar(
  runtime_info_after_solve$cuda_device_synchronize_count,
  runtime_info_before_solve$cuda_device_synchronize_count,
  "mixed solve does not call cudaDeviceSynchronize"
)
assert_true(
  is.logical(info$resource_snapshot_captured) &&
    length(info$resource_snapshot_captured) == 1L &&
    !is.na(info$resource_snapshot_captured) &&
    isTRUE(info$resource_snapshot_captured),
  "resource snapshot flag is one frozen TRUE logical scalar"
)
assert_integer_scalar(
  info$resource_instrumentation_version, 1L,
  "resource instrumentation version is integer one"
)
assert_true(
  is_integer_scalar(info$resource_allocation_count_before_solve) &&
    info$resource_allocation_count_before_solve >= 0L,
  "pre-solve allocation count is one non-negative integer scalar"
)
assert_integer_scalar(
  info$resource_allocation_count_after_solve,
  info$resource_allocation_count_before_solve,
  "solve preserves the integer allocation count"
)
assert_true(
  is_integer_scalar(info$resource_handle_create_count_before_solve) &&
    info$resource_handle_create_count_before_solve >= 0L,
  "pre-solve handle count is one non-negative integer scalar"
)
assert_integer_scalar(
  info$resource_handle_create_count_after_solve,
  info$resource_handle_create_count_before_solve,
  "solve preserves the integer handle count"
)
assert_integer_scalar(
  info$cuda_device_allocation_count_during_solve, 0L,
  "solve-time CUDA device allocation count is integer zero"
)
assert_integer_scalar(
  info$cuda_host_allocation_count_during_solve, 0L,
  "solve-time CUDA host allocation count is integer zero"
)
assert_integer_scalar(
  info$stream_create_count_during_solve, 0L,
  "solve-time stream creation count is integer zero"
)
assert_integer_scalar(
  info$event_create_count_during_solve, 0L,
  "solve-time event creation count is integer zero"
)
assert_integer_scalar(
  info$cublas_handle_create_count_during_solve, 0L,
  "solve-time cuBLAS handle creation count is integer zero"
)
assert_integer_scalar(
  info$cusolver_handle_create_count_during_solve, 0L,
  "solve-time cuSOLVER handle creation count is integer zero"
)
assert_integer_scalar(
  info$per_target_allocation_count_after_warmup, 0L,
  "per-target allocation count after warmup is integer zero"
)
assert_integer_scalar(
  info$per_target_handle_create_count, 0L,
  "per-target handle creation count is integer zero"
)
assert_integer_scalar(
  info$cpu_fallback_count, 0L,
  "CPU fallback count is integer zero"
)
assert_integer_scalar(
  info$unknown_fallback_count, 0L,
  "unknown fallback count is integer zero"
)
assert_integer_scalar(
  info$invalid_output_init_count, 1L,
  "invalid-output initialization count is integer one"
)
assert_integer_scalar(
  info$implicit_residual_d2h_count, 0L,
  "implicit residual D2H count is integer zero"
)
assert_integer_scalar(
  info$shadow_materialize_call_count, 0L,
  "pre-shadow materialize call count is integer zero"
)
assert_integer_scalar(
  info$shadow_materialize_target_count, 0L,
  "pre-shadow materialize target count is integer zero"
)
assert_double_scalar(
  info$shadow_d2h_bytes, 0,
  "pre-shadow D2H byte count is double zero"
)
assert_integer_scalar(
  runtime_info_before_solve$cholesky_factor_checkpoint_record_count, 0L,
  "new runtime factor checkpoint record count starts at integer zero"
)
assert_integer_scalar(
  runtime_info_after_solve$cholesky_factor_checkpoint_record_count,
  1L,
  "factor checkpoint record count reaches integer one after solve"
)
assert_integer_scalar(
  runtime_info_before_solve$cholesky_factor_checkpoint_wait_count, 0L,
  "new runtime factor checkpoint wait count starts at integer zero"
)
assert_integer_scalar(
  runtime_info_after_solve$cholesky_factor_checkpoint_wait_count,
  1L,
  "factor checkpoint wait count reaches integer one after solve"
)
assert_integer_scalar(
  runtime_info_before_solve$cholesky_solve_checkpoint_record_count, 0L,
  "new runtime solve checkpoint record count starts at integer zero"
)
assert_integer_scalar(
  runtime_info_after_solve$cholesky_solve_checkpoint_record_count,
  1L,
  "solve checkpoint record count reaches integer one after solve"
)
assert_integer_scalar(
  runtime_info_before_solve$cholesky_solve_checkpoint_wait_count, 0L,
  "new runtime solve checkpoint wait count starts at integer zero"
)
assert_integer_scalar(
  runtime_info_after_solve$cholesky_solve_checkpoint_wait_count,
  1L,
  "solve checkpoint wait count reaches integer one after solve"
)

raw_coefficient_shadow <- .Call(
  "C_fixed_sp_cuda_test_coefficient_shadow",
  token,
  PACKAGE = "fastkpc_cuda"
)
assert_true(
  identical(dim(raw_coefficient_shadow),
            c(dto$coefficient_dim, target_count)) &&
    all(is.finite(raw_coefficient_shadow)),
  "raw coefficient columns are finite for every completed route"
)

shadow <- fixed_sp_cuda_materialize_shadow(
  token,
  outputs = c("coefficients", "fitted", "residuals", "rss", "rhs")
)
assert_true(
  identical(dim(shadow$coefficients), c(dto$coefficient_dim, target_count)) &&
    identical(dim(shadow$fitted), c(dto$n, target_count)) &&
    identical(dim(shadow$residuals), c(dto$n, target_count)) &&
    identical(length(shadow$rss), target_count) &&
    identical(dim(shadow$cuda_nullspace_rhs),
              c(dto$null_dim, target_count)),
  "mixed shadow preserves full public batch dimensions"
)
info_after_shadow <- fixed_sp_cuda_residual_info(token)
prepared_info_after_shadow <- fixed_sp_cuda_prepared_info(handle)
missing_after_shadow_fields <- setdiff(
  required_resource_info_fields, names(info_after_shadow)
)
assert_true(
  length(missing_after_shadow_fields) == 0L,
  paste0(
    "post-materialize resource diagnostics are fail-closed; missing=",
    paste(missing_after_shadow_fields, collapse = ",")
  )
)
expected_shadow_d2h_bytes <- 8 * as.double(target_count) *
  (dto$coefficient_dim + dto$n + dto$n + 1L + dto$null_dim)
assert_true(
  "output_slot_leased" %in% names(prepared_info_after_shadow) &&
    identical(prepared_info_after_shadow$output_slot_leased, TRUE),
  "explicit shadow materialization retains the output-slot lease"
)
assert_integer_scalar(
  info_after_shadow$shadow_materialize_call_count, 1L,
  "post-shadow materialize call count is integer one"
)
assert_integer_scalar(
  info_after_shadow$shadow_materialize_target_count,
  as.integer(target_count),
  "post-shadow materialize target count covers the public batch"
)
assert_double_scalar(
  info_after_shadow$shadow_d2h_bytes, expected_shadow_d2h_bytes,
  "post-shadow D2H bytes match successful full-output columns"
)
assert_integer_scalar(
  info_after_shadow$implicit_residual_d2h_count, 0L,
  "post-shadow implicit residual D2H count remains integer zero"
)
assert_integer_scalar(
  info_after_shadow$output_slot_release_count, 0L,
  "shadow materialization does not release the output slot"
)

coefficient_column_max_abs_errors <- vapply(
  seq_along(safe),
  function(index) max(abs(
    shadow$coefficients[, safe[[index]]] - oracle_coefficients[, index]
  )),
  numeric(1L)
)
fitted_column_max_abs_errors <- vapply(
  seq_along(safe),
  function(index) max(abs(
    shadow$fitted[, safe[[index]]] - oracle_fitted[, index]
  )),
  numeric(1L)
)
residual_column_max_abs_errors <- vapply(
  seq_along(safe),
  function(index) max(abs(
    shadow$residuals[, safe[[index]]] - oracle_residuals[, index]
  )),
  numeric(1L)
)
rss_column_max_abs_errors <- vapply(
  seq_along(safe),
  function(index) abs(shadow$rss[[safe[[index]]]] - oracle_rss[[index]]),
  numeric(1L)
)
rhs_column_max_abs_errors <- vapply(
  seq_along(safe),
  function(index) max(abs(
    shadow$cuda_nullspace_rhs[, safe[[index]]] - oracle_rhs[, index]
  )),
  numeric(1L)
)
output_max_abs_errors <- c(
  coefficients = max(coefficient_column_max_abs_errors),
  fitted = max(fitted_column_max_abs_errors),
  residuals = max(residual_column_max_abs_errors),
  rss = max(rss_column_max_abs_errors),
  rhs = max(rhs_column_max_abs_errors)
)
coefficient_relative_l2_errors <- vapply(
  seq_along(safe),
  function(index) relative_l2(
    shadow$coefficients[, safe[[index]]], oracle_coefficients[, index]
  ),
  numeric(1L)
)
fitted_relative_l2_errors <- vapply(
  seq_along(safe),
  function(index) relative_l2(
    shadow$fitted[, safe[[index]]], oracle_fitted[, index]
  ),
  numeric(1L)
)
residual_relative_l2_errors <- vapply(
  seq_along(safe),
  function(index) relative_l2(
    shadow$residuals[, safe[[index]]], oracle_residuals[, index]
  ),
  numeric(1L)
)
rss_relative_l2_errors <- vapply(
  seq_along(safe),
  function(index) relative_l2(
    shadow$rss[[safe[[index]]]], oracle_rss[[index]]
  ),
  numeric(1L)
)
rhs_relative_l2_errors <- vapply(
  seq_along(safe),
  function(index) relative_l2(
    shadow$cuda_nullspace_rhs[, safe[[index]]], oracle_rhs[, index]
  ),
  numeric(1L)
)
output_max_relative_l2_errors <- c(
  coefficients = max(coefficient_relative_l2_errors),
  fitted = max(fitted_relative_l2_errors),
  residuals = max(residual_relative_l2_errors),
  rss = max(rss_relative_l2_errors),
  rhs = max(rhs_relative_l2_errors)
)
numerical_max_abs_error <- max(output_max_abs_errors)
numerical_max_relative_l2_error <- max(output_max_relative_l2_errors)
assert_true(
  all(coefficient_column_max_abs_errors < 1e-7) &&
    all(fitted_column_max_abs_errors < 1e-7) &&
    all(residual_column_max_abs_errors < 1e-7) &&
    all(rss_column_max_abs_errors < 1e-7) &&
    all(rhs_column_max_abs_errors < 1e-12) &&
    all(coefficient_relative_l2_errors < 1e-7) &&
    all(fitted_relative_l2_errors < 1e-7) &&
    all(residual_relative_l2_errors < 1e-7) &&
    all(rss_relative_l2_errors < 1e-7) &&
    all(rhs_relative_l2_errors < 1e-12),
  paste0(
    "every safe canonical output matches the Phase 2 oracle; max=",
    format(numerical_max_abs_error, digits = 17L),
    "; relative_l2=",
    format(numerical_max_relative_l2_error, digits = 17L)
  )
)

stable_outputs_are_finite <- all(vapply(
  stable,
  function(index) {
    all(is.finite(shadow$coefficients[, index])) &&
      all(is.finite(shadow$fitted[, index])) &&
      all(is.finite(shadow$residuals[, index])) &&
      is.finite(shadow$rss[[index]]) &&
      all(is.finite(shadow$cuda_nullspace_rhs[, index]))
  },
  logical(1L)
))
shadow_successful_target_mask <- vapply(
  seq_len(target_count),
  function(index) {
    all(is.finite(shadow$coefficients[, index])) &&
      all(is.finite(shadow$fitted[, index])) &&
      all(is.finite(shadow$residuals[, index])) &&
      is.finite(shadow$rss[[index]]) &&
      all(is.finite(shadow$cuda_nullspace_rhs[, index]))
  },
  logical(1L)
)
assert_true(
  stable_outputs_are_finite && all(shadow_successful_target_mask),
  "shadow success mask covers every canonical route"
)

fixed_sp_cuda_residual_release(token)
released_info <- fixed_sp_cuda_residual_info(token)
prepared_info_after_release <- fixed_sp_cuda_prepared_info(handle)
assert_true(
  "output_slot_leased" %in% names(prepared_info_after_release) &&
    identical(prepared_info_after_release$output_slot_leased, FALSE),
  "mixed batch release returns its output-slot lease"
)
assert_integer_scalar(
  released_info$output_slot_release_count, 1L,
  "mixed batch output-slot release count is integer one"
)
fixed_sp_cuda_residual_free(token)
token <- NULL
fixed_sp_cuda_prepared_free(handle)
handle <- NULL
fixed_sp_cuda_runtime_free(runtime)
runtime <- NULL

cat("PASS Phase 3B fixed-sp truthful mixed batch\n")
cat("setup key:", setup_key, "\n")
cat("target count:", target_count, "\n")
cat("safe count:", length(safe), "\n")
cat("stable count:", length(stable), "\n")
cat("safe ordinals:", paste(safe, collapse = ","), "\n")
cat("stable ordinals:", paste(stable, collapse = ","), "\n")
cat("planned routes:", paste(native_batch$planned_route, collapse = ","),
    "\n")
cat("max abs error:", format(numerical_max_abs_error, digits = 17L),
    "\n")
cat("max abs by output:",
    paste(
      names(output_max_abs_errors),
      format(output_max_abs_errors, digits = 17L),
      sep = "=", collapse = ","
    ), "\n")
cat("max relative L2 error:",
    format(numerical_max_relative_l2_error, digits = 17L), "\n")
cat("max relative L2 by output:",
    paste(
      names(output_max_relative_l2_errors),
      format(output_max_relative_l2_errors, digits = 17L),
      sep = "=", collapse = ","
    ), "\n")
cat("true-batched subgroup/attempted/successful:",
    info$true_batched_subgroup_count,
    info$true_batched_attempted_target_count,
    info$true_batched_target_count, "\n")
cat("batch finalize coefficient/fitted/residual-RSS/per-target:",
    info$coefficient_batch_finalize_call_count,
    info$fitted_batch_finalize_call_count,
    info$residual_rss_batch_finalize_call_count,
    info$per_target_output_finalize_call_count, "\n")
cat("shadow calls/targets/bytes:",
    info_after_shadow$shadow_materialize_call_count,
    info_after_shadow$shadow_materialize_target_count,
    format(info_after_shadow$shadow_d2h_bytes, scientific = FALSE), "\n")
