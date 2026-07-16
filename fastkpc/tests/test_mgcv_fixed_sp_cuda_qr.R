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
  qr_batch_count > 0L && qr_batch_count < 31L &&
    identical(qr_setup_keys, sort(qr_setup_keys, method = "radix")) &&
    !anyDuplicated(qr_setup_keys),
  "QR-bearing setup batches are non-empty, canonical, and fewer than targets"
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

for (batch_index in seq_along(qr_batches)) {
  setup_key <- qr_setup_keys[[batch_index]]
  batch <- qr_batches[[setup_key]]
  dto <- dtos[[setup_key]]
  native <- native_batches[[setup_key]]
  qr_indices <- which(native$planned_route == "AUGMENTED_QR")
  non_qr_indices <- which(native$planned_route != "AUGMENTED_QR")

  runtime_before <- fixed_sp_cuda_runtime_info(runtime)
  handle <- fixed_sp_cuda_prepared_create(runtime, dto)
  token <- fixed_sp_cuda_solve_batch(
    handle, native$Y, native$SP, native$planned_route,
    native$target_keys, outputs = c("fitted", "residuals")
  )
  info <- fixed_sp_cuda_residual_info(token)
  runtime_after <- fixed_sp_cuda_runtime_info(runtime)

  observed_qr_status <- info$solver_status[qr_indices]
  assert_true(
    identical(observed_qr_status, rep("OK_AUGMENTED_QR", length(qr_indices))),
    paste0(
      "QR targets must complete through augmented QR; setup=", setup_key,
      "; observed=", paste(observed_qr_status, collapse = ",")
    )
  )

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
    identical(info$planned_route[qr_indices],
              rep("AUGMENTED_QR", length(qr_indices))) &&
      identical(info$executed_route[qr_indices],
                rep("AUGMENTED_QR", length(qr_indices))) &&
      identical(info$reroute_reason[qr_indices],
                rep("", length(qr_indices))) &&
      identical(info$target_true_batched[qr_indices],
                rep(FALSE, length(qr_indices))) &&
      identical(info$qr_rank[qr_indices],
                rep(dto$null_dim, length(qr_indices))) &&
      identical(info$geqrf_info[qr_indices], integer(length(qr_indices))) &&
      identical(info$ormqr_info[qr_indices], integer(length(qr_indices))),
    paste("accepted QR metadata is exact for setup", setup_key)
  )
  if (length(non_qr_indices) > 0L) {
    assert_true(
      identical(info$qr_rank[non_qr_indices],
                rep(-1L, length(non_qr_indices))) &&
        identical(info$geqrf_info[non_qr_indices],
                  rep(-1L, length(non_qr_indices))) &&
        identical(info$ormqr_info[non_qr_indices],
                  rep(-1L, length(non_qr_indices))) &&
        all(!info$target_true_batched[non_qr_indices] |
              info$solver_status[non_qr_indices] ==
                "OK_CHOLESKY_BATCHED"),
      paste("non-QR sentinel diagnostics are exact for setup", setup_key)
    )
  }
  assert_true(
    identical(info$planned_qr_target_count, as.integer(length(qr_indices))) &&
      identical(info$executed_qr_target_count,
                as.integer(length(qr_indices))) &&
      identical(info$qr_to_svd_count, 0L) &&
      identical(info$stable_reroute_count, 0L),
    paste("QR route accounting is exact for setup", setup_key)
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
assert_true(
  identical(executed_qr_total, 31L) && identical(qr_to_svd_total, 0L) &&
    runtime_after_iteration$qr_checkpoint_record_count -
        runtime_before_iteration$qr_checkpoint_record_count ==
      qr_batch_count &&
    runtime_after_iteration$qr_checkpoint_wait_count -
        runtime_before_iteration$qr_checkpoint_wait_count ==
      qr_batch_count &&
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
              "ERR_STABLE_PATH_NOT_IMPLEMENTED") &&
    rank_guard_info$qr_rank[[1L]] < synthetic_dto$null_dim &&
    identical(rank_guard_info$geqrf_info, 0L) &&
    identical(rank_guard_info$ormqr_info, 0L) &&
    identical(rank_guard_info$target_true_batched, FALSE) &&
    identical(rank_guard_info$executed_qr_target_count, 0L) &&
    identical(rank_guard_info$executed_svd_target_count, 1L) &&
    identical(rank_guard_info$stable_reroute_count, 1L) &&
    identical(rank_guard_info$qr_to_svd_count, 1L),
  "rank-deficient augmented QR reroutes fail-closed to the unimplemented SVD"
)
assert_true(
  runtime_after_rank_guard$qr_checkpoint_record_count -
      runtime_before_rank_guard$qr_checkpoint_record_count == 1L &&
    runtime_after_rank_guard$qr_checkpoint_wait_count -
      runtime_before_rank_guard$qr_checkpoint_wait_count == 1L &&
    identical(runtime_after_rank_guard$cuda_device_synchronize_count,
              runtime_before_rank_guard$cuda_device_synchronize_count),
  "rank guard uses one QR event checkpoint without cudaDeviceSynchronize"
)
rank_guard_shadow <- fixed_sp_cuda_materialize_shadow(
  token, outputs = c("fitted", "residuals")
)
assert_true(
  all(is.na(rank_guard_shadow$fitted) & !is.nan(rank_guard_shadow$fitted)) &&
    all(is.na(rank_guard_shadow$residuals) &
          !is.nan(rank_guard_shadow$residuals)),
  "rank-rejected provisional QR output is not publicly exposed"
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
