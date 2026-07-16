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
expected_X_null <- function(dto) {
  if (identical(dto$constraint_mode, "identity")) {
    dto$X
  } else {
    dto$X %*% dto$constraint_nullspace
  }
}
cpu_augmented_B <- function(dto, roots, sp) {
  blocks <- list(expected_X_null(dto))
  for (penalty_index in seq_len(dto$penalty_count)) {
    rank <- dto$penalty_ranks[[penalty_index]]
    if (rank > 0L) {
      blocks[[length(blocks) + 1L]] <-
        sqrt(sp[[penalty_index]]) *
        roots$penalty_roots[[penalty_index]]
    }
  }
  if (roots$H_root_rank > 0L) {
    blocks[[length(blocks) + 1L]] <- roots$H_root
  }
  do.call(rbind, blocks)
}
cpu_augmented_reference <- function(dto, roots, sp, y) {
  B <- cpu_augmented_B(dto, roots, sp)
  decomposition <- svd(B)
  tolerance <- max(nrow(B), ncol(B)) * max(decomposition$d) *
    .Machine$double.eps
  retained <- decomposition$d > tolerance
  c_augmented <- c(y, rep(0, nrow(B) - dto$n))
  scaled <- as.vector(crossprod(decomposition$u, c_augmented))
  scaled[retained] <- scaled[retained] / decomposition$d[retained]
  scaled[!retained] <- 0
  theta <- as.vector(decomposition$v %*% scaled)
  fitted <- as.vector(expected_X_null(dto) %*% theta)
  list(
    expected_augmented_rank = as.integer(sum(retained)),
    fitted = fitted,
    residuals = y - fitted
  )
}

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP Phase 3C fixed-sp augmented SVD\n")
  quit(save = "no", status = 0L)
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
svd_batch_mask <- vapply(iteration_batches, function(batch) {
  any(batch$planned_route == "AUGMENTED_SVD")
}, logical(1L))
svd_batches <- iteration_batches[svd_batch_mask]
svd_setup_keys <- names(svd_batches)
svd_batch_count <- as.integer(length(svd_batches))

assert_true(
  length(svd_batches) > 0L &&
    identical(svd_setup_keys, sort(svd_setup_keys, method = "radix")) &&
    !anyDuplicated(svd_setup_keys),
  "SVD-bearing setup batches are non-empty and canonical"
)
assert_true(
  all(vapply(svd_batches, function(batch) {
    expected_rows <- iteration$target_rows[
      iteration$target_rows$prepared_s_key_sha256 ==
        batch$prepared_s_key_sha256,
      , drop = FALSE
    ]
    identical(batch$target_rows$residual_key_sha256,
              expected_rows$residual_key_sha256) &&
      identical(batch$planned_route, expected_rows$planned_route)
  }, logical(1L))),
  "every selected SVD setup retains its complete canonical target batch"
)

dtos <- lapply(svd_batches, function(batch) {
  fastkpc_full_cuda_fixed_sp_native_dto(batch$setup)
})
native_batches <- lapply(seq_along(svd_batches), function(index) {
  fastkpc_full_cuda_fixed_sp_native_batch(
    svd_batches[[index]], dtos[[index]]
  )
})
names(dtos) <- names(native_batches) <- svd_setup_keys

planned_svd_target_count <- as.integer(sum(vapply(
  native_batches, function(batch) {
    sum(batch$planned_route == "AUGMENTED_SVD")
  }, integer(1L)
)))
svd_conditions <- unlist(lapply(svd_batches, function(batch) {
  batch$condition[batch$planned_route == "AUGMENTED_SVD"]
}), use.names = FALSE)
svd_finite_high_count <- as.integer(sum(is.finite(svd_conditions)))
svd_nonfinite_count <- as.integer(sum(!is.finite(svd_conditions)))
svd_condition_min <-
  fastkpc_full_cuda_fixed_sp_contract()$svd_condition_min

assert_integer_scalar(
  planned_svd_target_count, 67L,
  "authenticated iteration scope has exactly 67 planned SVD targets"
)
assert_integer_scalar(
  svd_finite_high_count, 59L,
  "authenticated iteration SVD scope has exactly 59 finite-high targets"
)
assert_integer_scalar(
  svd_nonfinite_count, 8L,
  "authenticated iteration SVD scope has exactly 8 nonfinite targets"
)
assert_true(
  all(svd_conditions[is.finite(svd_conditions)] >= svd_condition_min),
  "finite planned SVD conditions meet the authenticated high-condition gate"
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
  "effective_rank", "sigma_max", "smallest_retained_sigma", "svd_info",
  "target_true_batched"
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

executed_svd_target_count <- 0L
max_residual_abs <- 0
max_residual_relative_l2 <- 0
max_fitted_abs <- 0
max_fitted_relative_l2 <- 0
rank_diagnostic_rows <- vector("list", planned_svd_target_count)
rank_diagnostic_index <- 0L

for (batch_index in seq_along(svd_batches)) {
  setup_key <- svd_setup_keys[[batch_index]]
  batch <- svd_batches[[setup_key]]
  dto <- dtos[[setup_key]]
  native <- native_batches[[setup_key]]
  svd_indices <- which(native$planned_route == "AUGMENTED_SVD")
  authenticated_rows <- iteration$target_rows[match(
    native$target_keys, iteration$target_rows$residual_key_sha256
  ), , drop = FALSE]

  assert_true(
    identical(as.numeric(batch$condition),
              as.numeric(authenticated_rows$condition)) &&
      identical(batch$target_rows$coefficient_rank,
                authenticated_rows$coefficient_rank),
    paste("authenticated Phase 1 condition and mgcv rank are retained for setup",
          setup_key)
  )

  runtime_before <- fixed_sp_cuda_runtime_info(runtime)
  handle <- fixed_sp_cuda_prepared_create(runtime, dto)
  roots <- fixed_sp_cuda_prepared_materialize_roots_for_test(handle)
  cpu_references <- lapply(svd_indices, function(target_index) {
    cpu_augmented_reference(
      dto, roots, native$SP[, target_index], native$Y[, target_index]
    )
  })
  expected_ranks <- vapply(
    cpu_references, `[[`, integer(1L), "expected_augmented_rank"
  )

  token <- fixed_sp_cuda_solve_batch(
    handle, native$Y, native$SP, native$planned_route,
    native$target_keys, outputs = c("fitted", "residuals")
  )
  info <- fixed_sp_cuda_residual_info(token)
  runtime_after <- fixed_sp_cuda_runtime_info(runtime)

  observed_status <- info$solver_status[svd_indices]
  assert_true(
    identical(observed_status,
              rep("OK_AUGMENTED_SVD", length(svd_indices))),
    paste0(
      "SVD targets must complete through augmented SVD; setup=", setup_key,
      "; observed=", paste(observed_status, collapse = ",")
    )
  )

  missing_target_fields <- setdiff(required_target_fields, names(info))
  missing_resource_fields <- setdiff(required_resource_fields, names(info))
  assert_true(
    length(missing_target_fields) == 0L &&
      length(missing_resource_fields) == 0L,
    paste0(
      "SVD diagnostics are fail-closed; target=",
      paste(missing_target_fields, collapse = ","),
      "; resource=", paste(missing_resource_fields, collapse = ",")
    )
  )
  assert_true(
    identical(info$planned_route[svd_indices],
              rep("AUGMENTED_SVD", length(svd_indices))) &&
      identical(info$executed_route[svd_indices],
                rep("AUGMENTED_SVD", length(svd_indices))) &&
      identical(info$reroute_reason[svd_indices],
                rep("", length(svd_indices))) &&
      identical(info$target_true_batched[svd_indices],
                rep(FALSE, length(svd_indices))) &&
      identical(info$svd_info[svd_indices],
                integer(length(svd_indices))) &&
      identical(info$effective_rank[svd_indices], expected_ranks) &&
      all(is.finite(info$sigma_max[svd_indices])) &&
      all(info$sigma_max[svd_indices] > 0) &&
      all(is.finite(info$smallest_retained_sigma[svd_indices])) &&
      all(info$smallest_retained_sigma[svd_indices] > 0) &&
      all(info$smallest_retained_sigma[svd_indices] <=
            info$sigma_max[svd_indices]) &&
      identical(info$planned_svd_target_count,
                as.integer(length(svd_indices))) &&
      identical(info$executed_svd_target_count,
                as.integer(length(svd_indices))) &&
      identical(info$stable_reroute_count, 0L) &&
      identical(info$cholesky_to_svd_count, 0L) &&
      identical(info$qr_to_svd_count, 0L) &&
      identical(info$batch_output_finalized_target_count,
                native$target_count) &&
      isTRUE(info$canonical_output_order_exact),
    paste("SVD route, rank, and compact diagnostics are exact for setup",
          setup_key)
  )
  assert_true(
    identical(runtime_after$cuda_device_synchronize_count,
              runtime_before$cuda_device_synchronize_count) &&
      runtime_after$svd_checkpoint_record_count -
        runtime_before$svd_checkpoint_record_count == 1L &&
      runtime_after$svd_checkpoint_wait_count -
        runtime_before$svd_checkpoint_wait_count == 1L &&
      identical(runtime_after$workspace_grow_count,
                runtime_before$workspace_grow_count) &&
      identical(runtime_after$workspace_bytes,
                runtime_before$workspace_bytes),
    paste("SVD uses no device synchronization or workspace growth for setup",
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
    paste("SVD solve creates no resources or fallbacks for setup", setup_key)
  )

  shadow <- fixed_sp_cuda_materialize_shadow(
    token, outputs = c("fitted", "residuals")
  )
  for (svd_ordinal in seq_along(svd_indices)) {
    target_index <- svd_indices[[svd_ordinal]]
    reference <- cpu_references[[svd_ordinal]]
    residual_abs <- max(abs(
      shadow$residuals[, target_index] - reference$residuals
    ))
    residual_relative <- relative_l2(
      shadow$residuals[, target_index], reference$residuals
    )
    fitted_abs <- max(abs(
      shadow$fitted[, target_index] - reference$fitted
    ))
    fitted_relative <- relative_l2(
      shadow$fitted[, target_index], reference$fitted
    )
    assert_true(
      is.finite(residual_abs) && residual_abs < 1e-7 &&
        is.finite(residual_relative) && residual_relative < 1e-7 &&
        is.finite(fitted_abs) && fitted_abs < 1e-7 &&
        is.finite(fitted_relative) && fitted_relative < 1e-7,
      paste0(
        "SVD augmented CPU parity failed for target ",
        native$target_keys[[target_index]],
        "; residual_abs=", format(residual_abs, digits = 17L),
        "; residual_relative_l2=", format(residual_relative, digits = 17L),
        "; fitted_abs=", format(fitted_abs, digits = 17L),
        "; fitted_relative_l2=", format(fitted_relative, digits = 17L)
      )
    )

    rank_diagnostic_index <- rank_diagnostic_index + 1L
    coefficient_rank <- batch$target_rows$coefficient_rank[[target_index]]
    rank_diagnostic_rows[[rank_diagnostic_index]] <- data.frame(
      setup_key = setup_key,
      target_key = native$target_keys[[target_index]],
      phase1_condition = batch$condition[[target_index]],
      mgcv_coefficient_rank = coefficient_rank,
      expected_augmented_rank = expected_ranks[[svd_ordinal]],
      effective_rank = info$effective_rank[[target_index]],
      effective_minus_coefficient_rank =
        info$effective_rank[[target_index]] - coefficient_rank,
      stringsAsFactors = FALSE
    )
    max_residual_abs <- max(max_residual_abs, residual_abs)
    max_residual_relative_l2 <- max(
      max_residual_relative_l2, residual_relative
    )
    max_fitted_abs <- max(max_fitted_abs, fitted_abs)
    max_fitted_relative_l2 <- max(max_fitted_relative_l2, fitted_relative)
  }

  executed_svd_target_count <- executed_svd_target_count +
    info$executed_svd_target_count
  fixed_sp_cuda_residual_release(token)
  fixed_sp_cuda_residual_free(token)
  token <- NULL
  fixed_sp_cuda_prepared_free(handle)
  handle <- NULL
}

rank_diagnostics <- do.call(rbind, rank_diagnostic_rows)
runtime_after_iteration <- fixed_sp_cuda_runtime_info(runtime)
assert_integer_scalar(
  executed_svd_target_count, 67L,
  "authenticated iteration scope executes exactly 67 SVD targets"
)
assert_true(
  nrow(rank_diagnostics) == 67L &&
    !anyDuplicated(rank_diagnostics$target_key) &&
    identical(rank_diagnostics$expected_augmented_rank,
              rank_diagnostics$effective_rank) &&
    runtime_after_iteration$svd_checkpoint_record_count -
        runtime_before_iteration$svd_checkpoint_record_count ==
      svd_batch_count &&
    runtime_after_iteration$svd_checkpoint_wait_count -
        runtime_before_iteration$svd_checkpoint_wait_count ==
      svd_batch_count &&
    identical(runtime_after_iteration$cuda_device_synchronize_count,
              runtime_before_iteration$cuda_device_synchronize_count),
  "all 67 SVD rank diagnostics and synchronization evidence are complete"
)

fixed_sp_cuda_runtime_free(runtime)
runtime <- NULL

cat("PASS Phase 3C fixed-sp augmented SVD\n")
cat("planned/executed SVD:", planned_svd_target_count,
    executed_svd_target_count, "\n")
cat("finite-high/nonfinite:", svd_finite_high_count,
    svd_nonfinite_count, "\n")
cat("max residual abs/relative-L2:",
    format(max_residual_abs, digits = 17L),
    format(max_residual_relative_l2, digits = 17L), "\n")
cat("max fitted abs/relative-L2:",
    format(max_fitted_abs, digits = 17L),
    format(max_fitted_relative_l2, digits = 17L), "\n")
cat("effective-minus-coefficient-rank range:",
    paste(range(rank_diagnostics$effective_minus_coefficient_rank),
          collapse = " "), "\n")
