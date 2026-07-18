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
assert_error <- function(expression, pattern, message) {
  error <- tryCatch(force(expression), error = identity)
  assert_true(
    inherits(error, "error") &&
      grepl(pattern, conditionMessage(error), fixed = TRUE),
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
expected_Z <- function(dto) {
  if (identical(dto$constraint_mode, "identity")) {
    diag(dto$coefficient_dim)
  } else {
    dto$constraint_nullspace
  }
}
projected_penalty <- function(dto, penalty_index) {
  block <- dto$penalty_blocks[[penalty_index]]
  offset <- dto$penalty_offsets_zero_based[[penalty_index]] + 1L
  indices <- offset:(offset + nrow(block) - 1L)
  full <- matrix(0, dto$coefficient_dim, dto$coefficient_dim)
  full[indices, indices] <- block
  Z <- expected_Z(dto)
  crossprod(Z, full %*% Z)
}
aggregate_penalty <- function(dto, sp) {
  Z <- expected_Z(dto)
  aggregate <- if (is.null(dto$H)) {
    matrix(0, dto$null_dim, dto$null_dim)
  } else {
    crossprod(Z, dto$H %*% Z)
  }
  for (penalty_index in seq_len(dto$penalty_count)) {
    aggregate <- aggregate +
      sp[[penalty_index]] * projected_penalty(dto, penalty_index)
  }
  aggregate
}
cpu_aggregate_factor <- function(P) {
  q <- ncol(P)
  factor <- suppressWarnings(chol(P, pivot = TRUE, tol = -1))
  rank <- as.integer(attr(factor, "rank"))
  pivot <- as.integer(attr(factor, "pivot"))
  root <- matrix(0, nrow = rank, ncol = q)
  if (rank > 0L) {
    root[, pivot] <- factor[seq_len(rank), , drop = FALSE]
  }
  list(
    rank = rank,
    pivot = pivot,
    root = root
  )
}
cpu_aggregate_effective_rank <- function(dto, aggregate_factor) {
  q <- dto$null_dim
  B <- rbind(
    expected_X_null(dto),
    aggregate_factor$root,
    matrix(0, nrow = q - aggregate_factor$rank, ncol = q)
  )
  singular_values <- svd(B, nu = 0L, nv = 0L)$d
  threshold <- max(singular_values) * sqrt(.Machine$double.eps)
  as.integer(sum(singular_values > 0.0 & singular_values >= threshold))
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
  expected_factor_calls <- as.integer(executed_svd)
  expected_b_builds <- 2L * expected_factor_calls
  expected_pivot_lengths <- q * expected_factor_calls
  pivot_permutations <- vapply(seq_len(target_count), function(index) {
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
      length(info$aggregate_factor_call_count) == target_count &&
      is.integer(info$aggregate_b_build_count) &&
      length(info$aggregate_b_build_count) == target_count &&
      is.double(info$aggregate_dstop) &&
      length(info$aggregate_dstop) == target_count &&
      is.integer(info$effective_rank) &&
      length(info$effective_rank) == target_count &&
      identical(info$aggregate_factor_call_count, expected_factor_calls) &&
      identical(info$aggregate_b_build_count, expected_b_builds) &&
      identical(lengths(info$aggregate_penalty_root_pivot),
                expected_pivot_lengths) &&
      all(pivot_permutations) &&
      all(info$aggregate_penalty_root_rank[executed_svd] >= 0L) &&
      all(info$aggregate_penalty_root_rank[executed_svd] <= q) &&
      all(is.na(info$aggregate_penalty_root_rank[!executed_svd])) &&
      all(info$effective_rank[executed_svd] >= 0L) &&
      all(info$effective_rank[executed_svd] <= q) &&
      all(is.finite(info$aggregate_dstop[executed_svd])) &&
      all(info$aggregate_dstop[executed_svd] >= 0) &&
      all(is.na(info$aggregate_dstop[!executed_svd])) &&
      is.integer(info$aggregate_penalty_factor_count) &&
      identical(info$aggregate_penalty_factor_count,
                as.integer(sum(expected_factor_calls))) &&
      is.integer(info$aggregate_svd_b_build_count) &&
      identical(info$aggregate_svd_b_build_count,
                as.integer(sum(expected_b_builds))) &&
      is.integer(info$aggregate_penalty_root_d2h_count) &&
      identical(info$aggregate_penalty_root_d2h_count, 0L),
    message
  )
  invisible(executed_svd)
}

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP Phase 3C fixed-sp augmented SVD\n")
  quit(save = "no", status = 0L)
}

subnormal_sigma <- .Machine$double.xmin * .Machine$double.eps
subnormal_tolerance <- subnormal_sigma * sqrt(.Machine$double.eps)
subnormal_spectrum <- c(subnormal_sigma, 0.0)
subnormal_retained <- subnormal_spectrum > 0.0 &
  subnormal_spectrum >= subnormal_tolerance
assert_true(
  subnormal_sigma > 0.0 && identical(subnormal_tolerance, 0.0) &&
    identical(subnormal_retained, c(TRUE, FALSE)),
  "positive subnormal rank boundary retains only the positive singular value"
)

stable_source <- paste(
  readLines("fastkpc/src/cuda/mgcv_fixed_sp_stable.cu", warn = FALSE),
  collapse = "\n"
)
compact_stable_source <- gsub("[[:space:]]+", " ", stable_source)
rank_kernel_start <- regexpr(
  "__global__ void fixed_sp_svd_rank_scale_kernel",
  compact_stable_source, fixed = TRUE
)[[1L]]
rank_kernel_end <- regexpr(
  "__global__ void initialize_fixed_sp_aggregate_diagnostics_kernel",
  compact_stable_source, fixed = TRUE
)[[1L]]
factor_kernel_start <- regexpr(
  "__global__ void factor_fixed_sp_aggregate_penalty_kernel",
  compact_stable_source, fixed = TRUE
)[[1L]]
factor_kernel_end <- regexpr(
  "__global__ void emit_fixed_sp_aggregate_root_kernel",
  compact_stable_source, fixed = TRUE
)[[1L]]
assert_true(
  rank_kernel_start > 0L && rank_kernel_end > rank_kernel_start &&
    factor_kernel_start > rank_kernel_end &&
    factor_kernel_end > factor_kernel_start,
  "stable SVD source-contract kernels are present and ordered"
)
rank_kernel_source <- substring(
  compact_stable_source, rank_kernel_start, rank_kernel_end - 1L
)
factor_kernel_source <- substring(
  compact_stable_source, factor_kernel_start, factor_kernel_end - 1L
)
fixed_occurrence_count <- function(text, pattern) {
  matches <- gregexpr(pattern, text, fixed = TRUE)[[1L]]
  if (length(matches) == 1L && matches[[1L]] == -1L) 0L else length(matches)
}
factor_increment_count <- fixed_occurrence_count(
  factor_kernel_source, "*aggregate_factor_call_count += 1;"
)
factor_assignment_count <- fixed_occurrence_count(
  factor_kernel_source, "*aggregate_factor_call_count = 1;"
)
source_contract_failures <- character()
if (!grepl(
  "if (sigma > 0.0 && sigma >= rank_tolerance)",
  rank_kernel_source, fixed = TRUE
)) {
  source_contract_failures <- c(
    source_contract_failures, "positive-only rank predicate is missing"
  )
}
if (factor_increment_count != 2L) {
  source_contract_failures <- c(
    source_contract_failures,
    paste0("aggregate factor increment count is ", factor_increment_count,
           ", expected 2")
  )
}
if (factor_assignment_count != 0L) {
  source_contract_failures <- c(
    source_contract_failures,
    paste0("aggregate factor assignment count is ", factor_assignment_count,
           ", expected 0")
  )
}
assert_true(
  length(source_contract_failures) == 0L,
  paste("stable SVD source contract failed:",
        paste(source_contract_failures, collapse = "; "))
)

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
unaffected_batches <- iteration_batches[!svd_batch_mask]
unaffected_setup_keys <- names(unaffected_batches)
svd_batches <- iteration_batches[svd_batch_mask]
svd_setup_keys <- names(svd_batches)
svd_batch_count <- as.integer(length(svd_batches))

assert_true(
  length(unaffected_batches) > 0L &&
    identical(unaffected_setup_keys,
              sort(unaffected_setup_keys, method = "radix")) &&
    !anyDuplicated(unaffected_setup_keys) &&
    all(vapply(unaffected_batches, function(batch) {
      !any(batch$planned_route == "AUGMENTED_SVD")
    }, logical(1L))),
  "unaffected iteration batches are non-empty, canonical, and exclude SVD"
)
unaffected_setup_key <- unaffected_setup_keys[[1L]]
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
unaffected_batch <- unaffected_batches[[unaffected_setup_key]]
unaffected_dto <- fastkpc_full_cuda_fixed_sp_native_dto(
  unaffected_batch$setup
)
unaffected_native <- fastkpc_full_cuda_fixed_sp_native_batch(
  unaffected_batch, unaffected_dto
)
unaffected_expected_rows <- iteration$target_rows[
  iteration$target_rows$prepared_s_key_sha256 ==
    unaffected_batch$prepared_s_key_sha256,
  , drop = FALSE
]
assert_true(
  identical(unaffected_batch$target_rows$residual_key_sha256,
            unaffected_expected_rows$residual_key_sha256) &&
    identical(unaffected_native$target_keys,
              unaffected_expected_rows$residual_key_sha256) &&
    identical(unaffected_native$planned_route,
              unaffected_expected_rows$planned_route) &&
    !any(unaffected_native$planned_route == "AUGMENTED_SVD"),
  "first unaffected batch is a complete real canonical iteration batch"
)

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

reserve_dtos <- c(unname(dtos), list(unaffected_dto))
reserve_native_batches <- c(unname(native_batches), list(unaffected_native))
max_n <- max(vapply(reserve_dtos, `[[`, integer(1L), "n"))
max_q <- max(vapply(reserve_dtos, `[[`, integer(1L), "null_dim"))
max_targets <- max(vapply(
  reserve_native_batches, `[[`, integer(1L), "target_count"
))
max_penalties <- max(vapply(
  reserve_dtos, `[[`, integer(1L), "penalty_count"
))
max_augmented_rows <- max(vapply(reserve_dtos, function(dto) {
  as.integer(dto$n + sum(dto$penalty_ranks))
}, integer(1L)))
assert_integer_scalar(
  max_augmented_rows, 407L,
  "canonical caller retains the authenticated logical QR row capacity"
)

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
  "target_true_batched", "aggregate_penalty_root_rank",
  "aggregate_penalty_root_pivot", "aggregate_factor_call_count",
  "aggregate_b_build_count", "aggregate_dstop",
  "aggregate_penalty_factor_count", "aggregate_svd_b_build_count",
  "aggregate_penalty_root_d2h_count"
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
oracle_call_count <- 0L
observed_aggregate_factor_count <- 0L
observed_aggregate_b_build_count <- 0L
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
  oracle_results <- lapply(svd_indices, function(target_index) {
    oracle_call_count <<- oracle_call_count + 1L
    fastkpc_mgcv_magic_fixed_sp_from_prepared(
      prepared_setup = batch$setup,
      target_state = list(
        row = batch$target_rows[target_index, , drop = FALSE],
        y = as.numeric(batch$Y[, target_index])
      )
    )
  })
  assert_true(
    all(vapply(oracle_results, function(reference) {
      isTRUE(reference$authoritative) &&
        identical(reference$solve_source, "mgcv-C-magic-from-prepared-s")
    }, logical(1L))),
    paste("all SVD numeric references are authoritative C_magic for setup",
          setup_key)
  )
  aggregate_references <- lapply(svd_indices, function(target_index) {
    penalty <- aggregate_penalty(dto, native$SP[, target_index])
    factor <- cpu_aggregate_factor(penalty)
    list(
      penalty = penalty,
      factor = factor,
      expected_effective_rank = cpu_aggregate_effective_rank(dto, factor)
    )
  })
  expected_ranks <- vapply(aggregate_references, function(reference) {
    reference$expected_effective_rank
  }, integer(1L))

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
  executed_svd <- assert_aggregate_svd_diagnostics(
    info, dto$null_dim, as.integer(length(svd_indices)),
    paste("aggregate diagnostic shapes and counts are exact for setup",
          setup_key)
  )
  observed_aggregate_factor_count <- observed_aggregate_factor_count +
    sum(info$aggregate_factor_call_count)
  observed_aggregate_b_build_count <- observed_aggregate_b_build_count +
    sum(info$aggregate_b_build_count)
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
    reference <- oracle_results[[svd_ordinal]]
    aggregate_reference <- aggregate_references[[svd_ordinal]]
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
        "SVD C_magic parity failed for target ",
        native$target_keys[[target_index]],
        "; residual_abs=", format(residual_abs, digits = 17L),
        "; residual_relative_l2=", format(residual_relative, digits = 17L),
        "; fitted_abs=", format(fitted_abs, digits = 17L),
        "; fitted_relative_l2=", format(fitted_relative, digits = 17L)
      )
    )
    assert_true(
      identical(
        info$aggregate_penalty_root_rank[[target_index]],
        aggregate_reference$factor$rank
      ) && identical(
        info$aggregate_penalty_root_pivot[[target_index]],
        aggregate_reference$factor$pivot
      ),
      paste("aggregate LAPACK rank/pivot parity for target",
            native$target_keys[[target_index]])
    )

    rank_diagnostic_index <- rank_diagnostic_index + 1L
    coefficient_rank <- batch$target_rows$coefficient_rank[[target_index]]
    rank_diagnostic_rows[[rank_diagnostic_index]] <- data.frame(
      setup_key = setup_key,
      target_key = native$target_keys[[target_index]],
      phase1_condition = batch$condition[[target_index]],
      numeric_reference = "mgcv-fixed-sp",
      phase1_coefficient_rank = coefficient_rank,
      aggregate_penalty_root_rank = aggregate_reference$factor$rank,
      expected_effective_rank = expected_ranks[[svd_ordinal]],
      effective_rank = info$effective_rank[[target_index]],
      aggregate_minus_phase1_rank =
        aggregate_reference$factor$rank - coefficient_rank,
      effective_minus_coefficient_rank =
        info$effective_rank[[target_index]] - coefficient_rank,
      residual_max_abs_diff = residual_abs,
      residual_relative_l2_diff = residual_relative,
      fitted_max_abs_diff = fitted_abs,
      fitted_relative_l2_diff = fitted_relative,
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

handle <- fixed_sp_cuda_prepared_create(runtime, unaffected_dto)
unaffected_prepared_before <- fixed_sp_cuda_prepared_info(handle)
unaffected_runtime_before <- fixed_sp_cuda_runtime_info(runtime)
token <- fixed_sp_cuda_solve_batch(
  handle, unaffected_native$Y, unaffected_native$SP,
  unaffected_native$planned_route, unaffected_native$target_keys,
  outputs = c("fitted", "residuals")
)
unaffected_info <- fixed_sp_cuda_residual_info(token)
unaffected_runtime_after <- fixed_sp_cuda_runtime_info(runtime)
unaffected_prepared_leased <- fixed_sp_cuda_prepared_info(handle)
unaffected_executed_svd <- assert_aggregate_svd_diagnostics(
  unaffected_info, unaffected_dto$null_dim, 0L,
  "unaffected real batch has zero aggregate SVD lifecycle entries and totals"
)
assert_true(
  !any(unaffected_native$planned_route == "AUGMENTED_SVD") &&
    !any(unaffected_executed_svd) &&
    identical(unaffected_info$planned_svd_target_count, 0L) &&
    identical(unaffected_info$executed_svd_target_count, 0L) &&
    all(!is.na(unaffected_info$executed_route)) &&
    !any(unaffected_info$executed_route == "AUGMENTED_SVD") &&
    all(startsWith(unaffected_info$solver_status, "OK_")) &&
    isTRUE(unaffected_info$canonical_output_order_exact),
  "unaffected real batch plans and executes no SVD targets"
)
assert_true(
  identical(unaffected_runtime_after$svd_checkpoint_record_count,
            unaffected_runtime_before$svd_checkpoint_record_count) &&
    identical(unaffected_runtime_after$svd_checkpoint_wait_count,
              unaffected_runtime_before$svd_checkpoint_wait_count),
  "unaffected real batch records and waits on no SVD checkpoint"
)
assert_true(
  identical(unaffected_prepared_before$output_slot_leased, FALSE) &&
    identical(unaffected_prepared_before$output_slot_state, "free") &&
    identical(unaffected_prepared_leased$output_slot_leased, TRUE) &&
    identical(unaffected_prepared_leased$output_slot_state, "leased") &&
    identical(unaffected_info$output_slot_acquire_count, 1L) &&
    identical(unaffected_info$output_slot_release_count, 0L),
  "unaffected real batch acquires exactly one prepared output-slot lease"
)
fixed_sp_cuda_residual_release(token)
unaffected_released_info <- fixed_sp_cuda_residual_info(token)
unaffected_prepared_released <- fixed_sp_cuda_prepared_info(handle)
assert_true(
  identical(unaffected_released_info$output_slot_release_count, 1L) &&
    identical(unaffected_prepared_released$output_slot_leased, FALSE) &&
    identical(unaffected_prepared_released$output_slot_state, "free") &&
    identical(unaffected_prepared_released$output_slot_poison_reason, ""),
  "unaffected real batch releases its token lease exactly once"
)
fixed_sp_cuda_residual_free(token)
assert_error(
  fixed_sp_cuda_residual_info(token), "has been freed",
  "unaffected real batch token is unusable after explicit free"
)
token <- NULL
fixed_sp_cuda_prepared_free(handle)
assert_error(
  fixed_sp_cuda_prepared_info(handle), "has been freed",
  "unaffected real batch prepared handle is unusable after cleanup"
)
handle <- NULL

rank_diagnostics <- do.call(rbind, rank_diagnostic_rows)
runtime_after_iteration <- fixed_sp_cuda_runtime_info(runtime)
assert_integer_scalar(
  executed_svd_target_count, 67L,
  "authenticated iteration scope executes exactly 67 SVD targets"
)
assert_true(
  nrow(rank_diagnostics) == 67L &&
    !anyDuplicated(rank_diagnostics$target_key) &&
    identical(rank_diagnostics$expected_effective_rank,
              rank_diagnostics$effective_rank) &&
    identical(oracle_call_count, 67L) &&
    all(rank_diagnostics$numeric_reference == "mgcv-fixed-sp") &&
    all(rank_diagnostics$residual_max_abs_diff < 1e-7) &&
    all(rank_diagnostics$residual_relative_l2_diff < 1e-7) &&
    all(rank_diagnostics$fitted_max_abs_diff < 1e-7) &&
    all(rank_diagnostics$fitted_relative_l2_diff < 1e-7) &&
    identical(observed_aggregate_factor_count, 67L) &&
    identical(observed_aggregate_b_build_count, 134L) &&
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

make_synthetic_aggregate_dto <- function(base, diagonal, root_rank) {
  q <- length(diagonal)
  dto <- base
  dto$n <- as.integer(q)
  dto$coefficient_dim <- as.integer(q)
  dto$null_dim <- as.integer(q)
  dto$X <- diag(as.double(rep(1, q)))
  dto$constraint_mode <- "identity"
  dto["constraint_nullspace"] <- list(NULL)
  dto$gram_matrix <- crossprod(dto$X)
  dto["nullspace_gram_matrix"] <- list(NULL)
  dto$penalty_count <- 1L
  dto$penalty_blocks <- list(
    penalty_1 = diag(as.double(diagonal), nrow = q, ncol = q)
  )
  dto$penalty_offsets_zero_based <- 0L
  dto$penalty_ranks <- as.integer(root_rank)
  dto$penalty_sp_indices_zero_based <- 0L
  dto$penalty_sp_labels <- "synthetic-sp"
  dto["H"] <- list(NULL)
  dto
}
run_synthetic_aggregate_case <- function(dto, target_key, message) {
  synthetic_handle <- fixed_sp_cuda_prepared_create(runtime, dto)
  synthetic_token <- NULL
  on.exit({
    if (!is.null(synthetic_token)) {
      try(fixed_sp_cuda_residual_release(synthetic_token), silent = TRUE)
      try(fixed_sp_cuda_residual_free(synthetic_token), silent = TRUE)
    }
    try(fixed_sp_cuda_prepared_free(synthetic_handle), silent = TRUE)
  }, add = TRUE)
  runtime_before <- fixed_sp_cuda_runtime_info(runtime)
  synthetic_token <- fixed_sp_cuda_solve_batch(
    synthetic_handle,
    matrix(as.double(seq_len(dto$n)), nrow = dto$n, ncol = 1L),
    matrix(1, nrow = 1L, ncol = 1L),
    "AUGMENTED_SVD", target_key, outputs = "residuals"
  )
  info <- fixed_sp_cuda_residual_info(synthetic_token)
  runtime_after <- fixed_sp_cuda_runtime_info(runtime)
  assert_aggregate_svd_diagnostics(info, dto$null_dim, 1L, message)
  assert_true(
    identical(info$planned_route, "AUGMENTED_SVD") &&
      identical(info$executed_route, "AUGMENTED_SVD") &&
      identical(info$reroute_reason, "") &&
      identical(info$solver_status, "OK_AUGMENTED_SVD") &&
      identical(runtime_after$workspace_grow_count,
                runtime_before$workspace_grow_count) &&
      identical(runtime_after$workspace_bytes,
                runtime_before$workspace_bytes) &&
      identical(runtime_after$cuda_device_synchronize_count,
                runtime_before$cuda_device_synchronize_count) &&
      runtime_after$svd_checkpoint_record_count -
        runtime_before$svd_checkpoint_record_count == 1L &&
      runtime_after$svd_checkpoint_wait_count -
        runtime_before$svd_checkpoint_wait_count == 1L,
    paste(message, "uses the public SVD path without growth or synchronization")
  )
  fixed_sp_cuda_residual_release(synthetic_token)
  fixed_sp_cuda_residual_free(synthetic_token)
  synthetic_token <- NULL
  fixed_sp_cuda_prepared_free(synthetic_handle)
  synthetic_handle <- NULL
  info
}

synthetic_base <- dtos[[1L]]
equal_diagonal <- c(1, 1, 0.5)
equal_dto <- make_synthetic_aggregate_dto(
  synthetic_base, equal_diagonal, root_rank = 3L
)
equal_cpu <- cpu_aggregate_factor(diag(equal_diagonal))
assert_true(
  identical(equal_cpu$rank, 3L) &&
    identical(equal_cpu$pivot, c(1L, 2L, 3L)),
  "test-only LAPACK keeps the first remaining canonical equal-diagonal pivot"
)
equal_info <- run_synthetic_aggregate_case(
  equal_dto, strrep("b", 64L), "equal-diagonal aggregate factor"
)
assert_true(
  identical(equal_info$aggregate_penalty_root_rank, 3L) &&
    identical(equal_info$aggregate_penalty_root_pivot[[1L]],
              c(1L, 2L, 3L)),
  "public aggregate factor keeps the first remaining canonical pivot on ties"
)

halved_q <- 3L
halved_next_diagonal <- 0.75 * halved_q * .Machine$double.eps
assert_true(
  halved_next_diagonal >
      halved_q * (.Machine$double.eps / 2) &&
    halved_next_diagonal < halved_q * .Machine$double.eps,
  "synthetic next pivot lies strictly inside the halved-epsilon interval"
)
halved_diagonal <- c(1, halved_next_diagonal, 0)
halved_dto <- make_synthetic_aggregate_dto(
  synthetic_base, halved_diagonal, root_rank = 1L
)
halved_cpu <- cpu_aggregate_factor(diag(halved_diagonal))
assert_true(
  identical(halved_cpu$rank, 2L) &&
    identical(halved_cpu$pivot, c(1L, 2L, 3L)),
  "test-only LAPACK retains the pivot above q times half epsilon"
)
halved_info <- run_synthetic_aggregate_case(
  halved_dto, strrep("c", 64L), "halved-epsilon aggregate factor"
)
assert_true(
  identical(halved_info$aggregate_penalty_root_rank, 2L) &&
    identical(halved_info$aggregate_penalty_root_pivot[[1L]],
              c(1L, 2L, 3L)) &&
    identical(
      halved_info$aggregate_dstop,
      as.double(halved_q * (.Machine$double.eps / 2))
    ),
  "public aggregate factor uses the DPSTF2 halved-epsilon dstop"
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
