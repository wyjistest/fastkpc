source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_integer_scalar <- function(value, expected, message) {
  assert_true(
    is.integer(value) && length(value) == 1L && !is.na(value) &&
      identical(value, as.integer(expected)),
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
  pivots_are_exact <- vapply(seq_len(target_count), function(index) {
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
      all(pivots_are_exact) &&
      identical(info$aggregate_penalty_factor_count,
                as.integer(sum(expected_factor))) &&
      identical(info$aggregate_svd_b_build_count,
                as.integer(sum(expected_build))) &&
      identical(info$aggregate_penalty_root_d2h_count, 0L),
    message
  )
}
expected_complete_batch <- function(native, null_dim) {
  routes <- native$planned_route
  target_count <- native$target_count
  cholesky <- routes == "CHOLESKY_BATCHED"
  qr <- routes == "AUGMENTED_QR"
  svd <- routes == "AUGMENTED_SVD"
  safe_count <- as.integer(sum(cholesky))
  qr_count <- as.integer(sum(qr))
  svd_count <- as.integer(sum(svd))
  true_batched <- cholesky & safe_count >= 2L

  solver_status <- rep("ERR_STABLE_PATH_NOT_IMPLEMENTED", target_count)
  solver_status[qr] <- "OK_AUGMENTED_QR"
  solver_status[svd] <- "OK_AUGMENTED_SVD"
  solver_status[cholesky] <- if (safe_count >= 2L) {
    "OK_CHOLESKY_BATCHED"
  } else {
    "OK_CHOLESKY_SINGLE"
  }
  executed_route <- rep(NA_character_, target_count)
  executed_route[cholesky] <- "CHOLESKY_BATCHED"
  executed_route[qr] <- "AUGMENTED_QR"
  executed_route[svd] <- "AUGMENTED_SVD"
  qr_rank <- rep(-1L, target_count)
  geqrf_info <- rep(-1L, target_count)
  ormqr_info <- rep(-1L, target_count)
  qr_rank[qr] <- null_dim
  geqrf_info[qr] <- 0L
  ormqr_info[qr] <- 0L

  list(
    qr_indices = which(qr),
    non_qr_indices = which(!qr),
    solver_status = solver_status,
    executed_route = executed_route,
    reroute_reason = rep("", target_count),
    target_true_batched = true_batched,
    qr_rank = qr_rank,
    geqrf_info = geqrf_info,
    ormqr_info = ormqr_info,
    planned_cholesky_target_count = safe_count,
    planned_qr_target_count = qr_count,
    planned_svd_target_count = svd_count,
    executed_cholesky_target_count = safe_count,
    executed_qr_target_count = qr_count,
    executed_svd_target_count = svd_count,
    batch_output_finalized_target_count = target_count,
    true_batched_subgroup_count = as.integer(safe_count >= 2L),
    true_batched_attempted_target_count = if (safe_count >= 2L) {
      safe_count
    } else {
      0L
    },
    true_batched_target_count = if (safe_count >= 2L) safe_count else 0L,
    cholesky_single_target_count = as.integer(safe_count == 1L),
    true_batched_kernel = FALSE
  )
}

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP Phase 3C fixed-sp augmented QR\n")
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
iteration_batches <- fastkpc_full_cuda_fixed_sp_batches(catalog, iteration)
qr_batch_mask <- vapply(iteration_batches, function(batch) {
  any(batch$planned_route == "AUGMENTED_QR")
}, logical(1L))
qr_batches <- iteration_batches[qr_batch_mask]
qr_setup_keys <- names(qr_batches)
qr_batch_count <- as.integer(length(qr_batches))

assert_true(
  identical(qr_batch_count, 15L) &&
    identical(qr_setup_keys, sort(qr_setup_keys, method = "radix")) &&
    !anyDuplicated(qr_setup_keys),
  "authenticated iteration scope has exactly 15 canonical QR-bearing batches"
)
assert_true(
  all(vapply(qr_batches, function(batch) {
    expected_rows <- iteration$target_rows[
      iteration$target_rows$prepared_s_key_sha256 ==
        batch$prepared_s_key_sha256,
      , drop = FALSE
    ]
    identical(batch$target_rows$residual_key_sha256,
              expected_rows$residual_key_sha256) &&
      identical(batch$planned_route, expected_rows$planned_route)
  }, logical(1L))),
  "every selected QR setup retains its complete canonical target batch"
)

dtos <- lapply(qr_batches, function(batch) {
  fastkpc_full_cuda_fixed_sp_native_dto(batch$setup)
})
native_batches <- lapply(seq_along(qr_batches), function(index) {
  fastkpc_full_cuda_fixed_sp_native_batch(
    qr_batches[[index]], dtos[[index]]
  )
})
names(dtos) <- names(native_batches) <- qr_setup_keys

planned_qr_total <- as.integer(sum(vapply(native_batches, function(batch) {
  sum(batch$planned_route == "AUGMENTED_QR")
}, integer(1L))))
assert_integer_scalar(
  planned_qr_total, 31L,
  "authenticated iteration scope has exactly 31 planned QR targets"
)

max_n <- max(vapply(dtos, `[[`, integer(1L), "n"))
max_q <- max(vapply(dtos, `[[`, integer(1L), "null_dim"))
max_targets <- max(vapply(
  native_batches, `[[`, integer(1L), "target_count"
))
max_penalties <- max(vapply(dtos, `[[`, integer(1L), "penalty_count"))
max_augmented_rows <- max(vapply(dtos, function(dto) {
  as.integer(dto$n + sum(dto$penalty_ranks))
}, integer(1L)))

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
  runtime, max_n, max_q, max_targets, max_penalties, max_augmented_rows
)
runtime_before_iteration <- fixed_sp_cuda_runtime_info(runtime)

required_target_fields <- c(
  "qr_rank", "geqrf_info", "ormqr_info", "target_true_batched"
)
required_runtime_fields <- c(
  "workspace_grow_count", "workspace_bytes",
  "cuda_device_synchronize_count", "qr_checkpoint_record_count",
  "qr_checkpoint_wait_count"
)
required_resource_fields <- c(
  "resource_snapshot_captured",
  "resource_allocation_count_before_solve",
  "resource_allocation_count_after_solve",
  "resource_handle_create_count_before_solve",
  "resource_handle_create_count_after_solve",
  "cuda_device_allocation_count_during_solve",
  "cuda_host_allocation_count_during_solve",
  "stream_create_count_during_solve", "event_create_count_during_solve",
  "cublas_handle_create_count_during_solve",
  "cusolver_handle_create_count_during_solve",
  "per_target_allocation_count_after_warmup",
  "per_target_handle_create_count", "cpu_fallback_count",
  "unknown_fallback_count"
)

executed_qr_total <- 0L
qr_to_svd_total <- 0L
max_residual_abs <- 0
max_residual_relative_l2 <- 0
max_fitted_abs <- 0
max_fitted_relative_l2 <- 0
regular_batch_infos <- vector("list", length(qr_batches))
regular_batch_q <- integer(length(qr_batches))
regular_batch_expected_svd_count <- integer(length(qr_batches))

for (batch_index in seq_along(qr_batches)) {
  setup_key <- qr_setup_keys[[batch_index]]
  batch <- qr_batches[[setup_key]]
  dto <- dtos[[setup_key]]
  native <- native_batches[[setup_key]]
  expected <- expected_complete_batch(native, dto$null_dim)
  qr_indices <- expected$qr_indices

  runtime_before <- fixed_sp_cuda_runtime_info(runtime)
  handle <- fixed_sp_cuda_prepared_create(runtime, dto)
  token <- fixed_sp_cuda_solve_batch(
    handle, native$Y, native$SP, native$planned_route,
    native$target_keys, outputs = c("fitted", "residuals")
  )
  info <- fixed_sp_cuda_residual_info(token)
  runtime_after <- fixed_sp_cuda_runtime_info(runtime)
  regular_batch_infos[[batch_index]] <- info
  regular_batch_q[[batch_index]] <- dto$null_dim
  regular_batch_expected_svd_count[[batch_index]] <-
    expected$executed_svd_target_count

  missing_target_fields <- setdiff(required_target_fields, names(info))
  missing_runtime_fields <- setdiff(
    required_runtime_fields,
    intersect(names(runtime_before), names(runtime_after))
  )
  missing_resource_fields <- setdiff(required_resource_fields, names(info))
  assert_true(
    length(missing_target_fields) == 0L &&
      length(missing_runtime_fields) == 0L &&
      length(missing_resource_fields) == 0L,
    paste0(
      "QR diagnostics are fail-closed; target=",
      paste(missing_target_fields, collapse = ","),
      "; runtime=", paste(missing_runtime_fields, collapse = ","),
      "; resource=", paste(missing_resource_fields, collapse = ",")
    )
  )
  assert_true(
    identical(info$target_keys, native$target_keys) &&
      identical(info$planned_route, native$planned_route) &&
      identical(info$solver_status, expected$solver_status) &&
      identical(info$executed_route, expected$executed_route) &&
      identical(info$reroute_reason, expected$reroute_reason) &&
      identical(info$target_true_batched,
                expected$target_true_batched) &&
      identical(info$qr_rank, expected$qr_rank) &&
      identical(info$geqrf_info, expected$geqrf_info) &&
      identical(info$ormqr_info, expected$ormqr_info) &&
      identical(info$planned_cholesky_target_count,
                expected$planned_cholesky_target_count) &&
      identical(info$planned_qr_target_count,
                expected$planned_qr_target_count) &&
      identical(info$planned_svd_target_count,
                expected$planned_svd_target_count) &&
      identical(info$executed_cholesky_target_count,
                expected$executed_cholesky_target_count) &&
      identical(info$executed_qr_target_count,
                expected$executed_qr_target_count) &&
      identical(info$executed_svd_target_count,
                expected$executed_svd_target_count) &&
      identical(info$batch_output_finalized_target_count,
                expected$batch_output_finalized_target_count) &&
      identical(info$true_batched_subgroup_count,
                expected$true_batched_subgroup_count) &&
      identical(info$true_batched_attempted_target_count,
                expected$true_batched_attempted_target_count) &&
      identical(info$true_batched_target_count,
                expected$true_batched_target_count) &&
      identical(info$cholesky_single_target_count,
                expected$cholesky_single_target_count) &&
      identical(info$true_batched_kernel,
                expected$true_batched_kernel) &&
      isTRUE(info$canonical_output_order_exact) &&
      identical(info$cholesky_to_svd_count, 0L) &&
      identical(info$qr_to_svd_count, 0L) &&
      identical(info$stable_reroute_count, 0L),
    paste("complete mixed-batch route/status conservation is exact for setup",
          setup_key)
  )

  assert_true(
    runtime_after$qr_checkpoint_record_count -
        runtime_before$qr_checkpoint_record_count == 1L &&
      runtime_after$qr_checkpoint_wait_count -
        runtime_before$qr_checkpoint_wait_count == 1L &&
      identical(runtime_after$cuda_device_synchronize_count,
                runtime_before$cuda_device_synchronize_count) &&
      identical(runtime_after$workspace_grow_count,
                runtime_before$workspace_grow_count) &&
      identical(runtime_after$workspace_bytes,
                runtime_before$workspace_bytes),
    paste("QR uses one event checkpoint and no synchronization/growth for setup",
          setup_key)
  )
  assert_true(
    isTRUE(info$resource_snapshot_captured) &&
      identical(info$resource_allocation_count_after_solve,
                info$resource_allocation_count_before_solve) &&
      identical(info$resource_handle_create_count_after_solve,
                info$resource_handle_create_count_before_solve) &&
      all(unlist(info[c(
        "cuda_device_allocation_count_during_solve",
        "cuda_host_allocation_count_during_solve",
        "stream_create_count_during_solve", "event_create_count_during_solve",
        "cublas_handle_create_count_during_solve",
        "cusolver_handle_create_count_during_solve",
        "per_target_allocation_count_after_warmup",
        "per_target_handle_create_count", "cpu_fallback_count",
        "unknown_fallback_count"
      )], use.names = FALSE) == 0L),
    paste("QR solve creates no resources or fallbacks for setup", setup_key)
  )

  shadow <- fixed_sp_cuda_materialize_shadow(
    token, outputs = c("fitted", "residuals")
  )
  for (target_index in qr_indices) {
    oracle <- fastkpc_mgcv_magic_fixed_sp_from_prepared(
      prepared_setup = batch$setup,
      target_state = list(
        row = batch$target_rows[target_index, , drop = FALSE],
        y = as.numeric(batch$Y[, target_index])
      )
    )
    residual_abs <- max(abs(
      shadow$residuals[, target_index] - oracle$residuals
    ))
    residual_relative <- relative_l2(
      shadow$residuals[, target_index], oracle$residuals
    )
    fitted_abs <- max(abs(shadow$fitted[, target_index] - oracle$fitted))
    fitted_relative <- relative_l2(
      shadow$fitted[, target_index], oracle$fitted
    )
    assert_true(
      is.finite(residual_abs) && residual_abs < 1e-7 &&
        is.finite(residual_relative) && residual_relative < 1e-7 &&
        is.finite(fitted_abs) && fitted_abs < 1e-7 &&
        is.finite(fitted_relative) && fitted_relative < 1e-7,
      paste0(
        "QR oracle parity failed for target ", native$target_keys[[target_index]],
        "; residual_abs=", format(residual_abs, digits = 17L),
        "; residual_relative_l2=", format(residual_relative, digits = 17L),
        "; fitted_abs=", format(fitted_abs, digits = 17L),
        "; fitted_relative_l2=", format(fitted_relative, digits = 17L)
      )
    )
    max_residual_abs <- max(max_residual_abs, residual_abs)
    max_residual_relative_l2 <- max(
      max_residual_relative_l2, residual_relative
    )
    max_fitted_abs <- max(max_fitted_abs, fitted_abs)
    max_fitted_relative_l2 <- max(max_fitted_relative_l2, fitted_relative)
  }

  executed_qr_total <- executed_qr_total + info$executed_qr_target_count
  qr_to_svd_total <- qr_to_svd_total + info$qr_to_svd_count
  fixed_sp_cuda_residual_release(token)
  fixed_sp_cuda_residual_free(token)
  token <- NULL
  fixed_sp_cuda_prepared_free(handle)
  handle <- NULL
}

runtime_after_iteration <- fixed_sp_cuda_runtime_info(runtime)
qr_checkpoint_record_delta <-
  runtime_after_iteration$qr_checkpoint_record_count -
    runtime_before_iteration$qr_checkpoint_record_count
qr_checkpoint_wait_delta <-
  runtime_after_iteration$qr_checkpoint_wait_count -
    runtime_before_iteration$qr_checkpoint_wait_count
assert_true(
  identical(executed_qr_total, 31L) && identical(qr_to_svd_total, 0L) &&
    identical(qr_batch_count, 15L) &&
    identical(qr_checkpoint_record_delta, 15L) &&
    identical(qr_checkpoint_wait_delta, 15L) &&
    identical(runtime_after_iteration$cuda_device_synchronize_count,
              runtime_before_iteration$cuda_device_synchronize_count),
  "aggregate planned/executed QR and event-checkpoint counts are exact"
)

synthetic_setup_index <- which(vapply(dtos, function(dto) {
  dto$null_dim >= 2L
}, logical(1L)))[[1L]]
synthetic_key <- qr_setup_keys[[synthetic_setup_index]]
synthetic_dto <- dtos[[synthetic_key]]
synthetic_batch <- qr_batches[[synthetic_key]]
synthetic_qr_index <- which(
  native_batches[[synthetic_key]]$planned_route == "AUGMENTED_QR"
)[[1L]]
synthetic_dto$X[,] <- 0
synthetic_dto$gram_matrix <- crossprod(synthetic_dto$X)
if (identical(synthetic_dto$constraint_mode, "explicit")) {
  synthetic_dto$nullspace_gram_matrix <- crossprod(
    synthetic_dto$X %*% synthetic_dto$constraint_nullspace
  )
}
synthetic_Y <- matrix(
  synthetic_batch$Y[, synthetic_qr_index], ncol = 1L
)
synthetic_SP <- matrix(
  0, nrow = synthetic_dto$penalty_count, ncol = 1L
)
runtime_before_rank_guard <- fixed_sp_cuda_runtime_info(runtime)
handle <- fixed_sp_cuda_prepared_create(runtime, synthetic_dto)
token <- fixed_sp_cuda_solve_batch(
  handle, synthetic_Y, synthetic_SP, "AUGMENTED_QR",
  native_batches[[synthetic_key]]$target_keys[[synthetic_qr_index]],
  outputs = c("fitted", "residuals")
)
rank_guard_info <- fixed_sp_cuda_residual_info(token)
runtime_after_rank_guard <- fixed_sp_cuda_runtime_info(runtime)
assert_true(
  identical(rank_guard_info$planned_route, "AUGMENTED_QR") &&
    identical(rank_guard_info$executed_route, "AUGMENTED_SVD") &&
    identical(rank_guard_info$reroute_reason,
              "QR_RANK_GUARD_REJECTED") &&
    identical(rank_guard_info$solver_status,
              "OK_AUGMENTED_SVD") &&
    rank_guard_info$qr_rank[[1L]] < synthetic_dto$null_dim &&
    identical(rank_guard_info$geqrf_info, 0L) &&
    identical(rank_guard_info$ormqr_info, 0L) &&
    identical(rank_guard_info$effective_rank, 0L) &&
    identical(rank_guard_info$svd_info, 0L) &&
    identical(rank_guard_info$sigma_max, 0) &&
    identical(rank_guard_info$smallest_retained_sigma, 0) &&
    identical(rank_guard_info$target_true_batched, FALSE) &&
    identical(rank_guard_info$executed_qr_target_count, 0L) &&
    identical(rank_guard_info$executed_svd_target_count, 1L) &&
    identical(rank_guard_info$stable_reroute_count, 1L) &&
    identical(rank_guard_info$qr_to_svd_count, 1L),
  "rank-deficient augmented QR reroutes successfully through SVD"
)
assert_aggregate_svd_diagnostics(
  rank_guard_info, synthetic_dto$null_dim, 1L,
  "real QR_RANK_GUARD_REJECTED aggregate lifecycle"
)
for (batch_index in seq_along(regular_batch_infos)) {
  assert_aggregate_svd_diagnostics(
    regular_batch_infos[[batch_index]], regular_batch_q[[batch_index]],
    regular_batch_expected_svd_count[[batch_index]],
    paste("accepted QR and other canonical target aggregate lifecycle in batch",
          batch_index)
  )
}
assert_true(
  runtime_after_rank_guard$qr_checkpoint_record_count -
      runtime_before_rank_guard$qr_checkpoint_record_count == 1L &&
    runtime_after_rank_guard$qr_checkpoint_wait_count -
      runtime_before_rank_guard$qr_checkpoint_wait_count == 1L &&
    runtime_after_rank_guard$svd_checkpoint_record_count -
      runtime_before_rank_guard$svd_checkpoint_record_count == 1L &&
    runtime_after_rank_guard$svd_checkpoint_wait_count -
      runtime_before_rank_guard$svd_checkpoint_wait_count == 1L &&
    identical(runtime_after_rank_guard$cuda_device_synchronize_count,
              runtime_before_rank_guard$cuda_device_synchronize_count) &&
    identical(runtime_after_rank_guard$workspace_grow_count,
              runtime_before_rank_guard$workspace_grow_count) &&
    identical(runtime_after_rank_guard$workspace_bytes,
              runtime_before_rank_guard$workspace_bytes) &&
    runtime_before_rank_guard$augmented_workspace_bytes >=
      8 * (synthetic_dto$n + synthetic_dto$null_dim) *
        synthetic_dto$null_dim,
  paste0(
    "rank guard uses one QR/SVD checkpoint pair, logical reserve capacity, ",
    "and no solve-time growth"
  )
)
rank_guard_shadow <- fixed_sp_cuda_materialize_shadow(
  token, outputs = c("fitted", "residuals")
)
assert_true(
  max(abs(rank_guard_shadow$fitted)) < 1e-12 &&
    max(abs(rank_guard_shadow$residuals - synthetic_Y)) < 1e-12,
  "rank-rejected provisional QR output is replaced by the SVD solution"
)

fixed_sp_cuda_residual_release(token)
fixed_sp_cuda_residual_free(token)
token <- NULL
fixed_sp_cuda_prepared_free(handle)
handle <- NULL
fixed_sp_cuda_runtime_free(runtime)
runtime <- NULL

cat("PASS Phase 3C fixed-sp augmented QR\n")
cat("QR batches:", qr_batch_count, "\n")
cat("planned/executed/qr-to-svd:", planned_qr_total,
    executed_qr_total, qr_to_svd_total, "\n")
cat("QR checkpoint record/wait:",
    runtime_after_iteration$qr_checkpoint_record_count -
      runtime_before_iteration$qr_checkpoint_record_count,
    runtime_after_iteration$qr_checkpoint_wait_count -
      runtime_before_iteration$qr_checkpoint_wait_count, "\n")
cat("max residual abs/relative-L2:",
    format(max_residual_abs, digits = 17L),
    format(max_residual_relative_l2, digits = 17L), "\n")
cat("max fitted abs/relative-L2:",
    format(max_fitted_abs, digits = 17L),
    format(max_fitted_relative_l2, digits = 17L), "\n")
cat("rank guard rank/q:", rank_guard_info$qr_rank[[1L]],
    synthetic_dto$null_dim, "\n")
