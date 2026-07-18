source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_error <- function(expression, pattern, message) {
  error <- tryCatch(force(expression), error = identity)
  assert_true(
    inherits(error, "error") &&
      grepl(pattern, conditionMessage(error), fixed = TRUE),
    paste0(message, if (inherits(error, "error")) {
      paste0(": ", conditionMessage(error))
    } else {
      ": no error"
    })
  )
  invisible(error)
}
capture_prepared_create_error <- function(runtime, dto) {
  handle <- NULL
  tryCatch({
    handle <- fixed_sp_cuda_prepared_create(runtime, dto)
    fixed_sp_cuda_prepared_free(handle)
    NULL
  }, error = function(error) {
    if (!is.null(handle)) {
      try(fixed_sp_cuda_prepared_free(handle), silent = TRUE)
    }
    error
  })
}
error_matches <- function(error, pattern) {
  inherits(error, "error") &&
    grepl(pattern, conditionMessage(error), fixed = TRUE)
}
error_description <- function(error) {
  if (inherits(error, "error")) conditionMessage(error) else "no error"
}
assert_reconstructs <- function(root, expected, tolerance, message) {
  actual <- crossprod(root)
  error <- max(abs(actual - expected))
  assert_true(
    identical(dim(actual), dim(expected)) &&
      is.finite(error) && error < tolerance,
    paste0(message, ": max error ", format(error, digits = 17L))
  )
}
projected_penalty <- function(dto, penalty_index) {
  block <- dto$penalty_blocks[[penalty_index]]
  offset <- dto$penalty_offsets_zero_based[[penalty_index]] + 1L
  rows <- offset:(offset + nrow(block) - 1L)
  full <- matrix(0, dto$coefficient_dim, dto$coefficient_dim)
  full[rows, rows] <- block
  Z <- if (identical(dto$constraint_mode, "identity")) {
    diag(dto$coefficient_dim)
  } else {
    dto$constraint_nullspace
  }
  crossprod(Z, full %*% Z)
}
cpu_pivoted_cholesky_root <- function(P) {
  factor <- suppressWarnings(chol(P, pivot = TRUE, tol = -1))
  rank <- as.integer(attr(factor, "rank"))
  pivot <- as.integer(attr(factor, "pivot"))
  root <- matrix(0, nrow = rank, ncol = ncol(P))
  if (rank > 0L) {
    root[, pivot] <- factor[seq_len(rank), , drop = FALSE]
  }
  list(rank = rank, pivot = pivot, root = root)
}
cpu_augmented_svd_reference <- function(X_null, root, y) {
  q <- ncol(X_null)
  padded_root <- rbind(
    root,
    matrix(0, nrow = q - nrow(root), ncol = q)
  )
  B <- rbind(X_null, padded_root)
  decomposition <- svd(B, nu = q, nv = q)
  sigma_max <- max(decomposition$d)
  solve_rank_threshold <- sigma_max * sqrt(.Machine$double.eps)
  retained <- decomposition$d >= solve_rank_threshold
  augmented_y <- c(y, rep(0, q))
  scaled <- as.vector(crossprod(decomposition$u, augmented_y))
  scaled[retained] <- scaled[retained] / decomposition$d[retained]
  scaled[!retained] <- 0
  coefficients <- as.vector(decomposition$v %*% scaled)
  fitted <- as.vector(X_null %*% coefficients)
  list(
    fitted = fitted,
    residuals = y - fitted,
    effective_rank = as.integer(sum(retained)),
    sigma_max = sigma_max,
    solve_rank_threshold = solve_rank_threshold
  )
}
cpu_fixed_sp_reference <- function(
    X_null, projected_penalties, projected_H, sp, y) {
  system <- crossprod(X_null) + projected_H
  for (penalty_index in seq_along(projected_penalties)) {
    system <- system +
      sp[[penalty_index]] * projected_penalties[[penalty_index]]
  }
  coefficients <- as.vector(solve(system, crossprod(X_null, y)))
  fitted <- as.vector(X_null %*% coefficients)
  list(
    coefficients = coefficients,
    fitted = fitted,
    residuals = y - fitted
  )
}
max_abs_difference <- function(actual, expected) {
  max(abs(actual - expected))
}

stable_source <- paste(
  readLines("fastkpc/src/cuda/mgcv_fixed_sp_stable.cu", warn = FALSE),
  collapse = "\n"
)
root_kernel_start <- regexpr(
  "__global__ void build_fixed_sp_root_kernel", stable_source, fixed = TRUE
)
root_kernel_tail <- substring(stable_source, root_kernel_start)
root_kernel_end <- regexpr("\n}\n\n}  // namespace", root_kernel_tail,
                           fixed = TRUE)
root_kernel_source <- substring(
  root_kernel_tail, 1L, root_kernel_end + nchar("\n}") - 1L
)
required_size_indices <- c(
  "const std::size_t q_size = static_cast<std::size_t>(q);",
  "const std::size_t eigenvector_count = q_size * q_size;",
  "const std::size_t eigenvector_column_offset = q_size * eigen_index;",
  "const std::size_t root_leading_dimension_size =",
  "static_cast<std::size_t>(root_leading_dimension);",
  "const std::size_t root_row_offset_size =",
  "static_cast<std::size_t>(root_row_offset);",
  "const std::size_t root_index ="
)
assert_true(
  root_kernel_start > 0L && root_kernel_end > 0L &&
    all(vapply(
      required_size_indices, grepl, logical(1L),
      x = root_kernel_source, fixed = TRUE
    )) &&
    !grepl("q * q", root_kernel_source, fixed = TRUE) &&
    !grepl("q * eigen_index", root_kernel_source, fixed = TRUE) &&
    !grepl("root_leading_dimension * column", root_kernel_source,
           fixed = TRUE),
  "root kernel flattened matrix indices use std::size_t"
)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP Phase 3C fixed-sp penalty roots\n")
  quit(save = "no", status = 0L)
}

build_fastkpc_cuda_native(rebuild = TRUE)
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
batches <- fastkpc_full_cuda_fixed_sp_batches(catalog, iteration)
assert_true(length(batches) == 44L, "iteration setup batch count")

runtime <- fixed_sp_cuda_runtime_create(0L)
on.exit(try(fixed_sp_cuda_runtime_free(runtime), silent = TRUE), add = TRUE)
capacities <- fastkpc_full_cuda_fixed_sp_contract()$canonical_capacities
fixed_sp_cuda_runtime_reserve(
  runtime, capacities$n, capacities$null_dim, capacities$target_count,
  capacities$penalty_count, capacities$augmented_rows
)

total_root_matrix_count <- 0L
total_root_row_count <- 0L
total_rank_mismatch_count <- 0L
inspect_canonical_roots <- function(runtime, batch) {
  dto <- fastkpc_full_cuda_fixed_sp_native_dto(batch$setup)
  handle <- NULL
  on.exit({
    if (!is.null(handle)) {
      try(fixed_sp_cuda_prepared_free(handle), silent = TRUE)
    }
  }, add = TRUE)
  handle <- fixed_sp_cuda_prepared_create(runtime, dto)
  created_info <- fixed_sp_cuda_prepared_info(handle)
  assert_true(
    created_info$penalty_root_build_count == 1L &&
      created_info$setup_shadow_d2h_count == 0L,
    "smooth roots are built during setup before observation"
  )
  roots <- fixed_sp_cuda_prepared_materialize_roots_for_test(handle)
  info <- fixed_sp_cuda_prepared_info(handle)

  assert_true(
    length(roots$penalty_roots) == dto$penalty_count,
    "one root per smooth penalty"
  )
  for (penalty_index in seq_len(dto$penalty_count)) {
    root <- roots$penalty_roots[[penalty_index]]
    assert_reconstructs(
      root, projected_penalty(dto, penalty_index), 1e-10,
      "canonical smooth root reconstruction"
    )
    assert_true(
      nrow(root) == dto$penalty_ranks[[penalty_index]],
      "canonical smooth root row count matches authenticated rank"
    )
  }
  assert_true(is.null(roots$H_root), "canonical setup has no H root")
  assert_true(
    info$penalty_root_build_count == 1L &&
      info$penalty_root_rank_mismatch_count == 0L &&
      info$H_root_matrix_count == 0L && info$H_root_rank == 0L,
    "canonical roots are built once with separate smooth/H diagnostics"
  )
  assert_true(
    info$setup_shadow_d2h_count == 1L && info$setup_shadow_d2h_bytes > 0,
    "root materialization is counted as setup-shadow D2H"
  )

  list(
    matrix_count = info$penalty_root_matrix_count,
    row_count = info$penalty_root_row_count,
    rank_mismatch_count = info$penalty_root_rank_mismatch_count
  )
}
for (batch in batches) {
  counts <- inspect_canonical_roots(runtime, batch)
  total_root_matrix_count <- total_root_matrix_count + counts$matrix_count
  total_root_row_count <- total_root_row_count + counts$row_count
  total_rank_mismatch_count <- total_rank_mismatch_count +
    counts$rank_mismatch_count
}

assert_true(total_root_matrix_count == 159L,
            "iteration penalty root matrix count")
assert_true(total_root_row_count == 1424L,
            "iteration penalty root row count")
assert_true(total_rank_mismatch_count == 0L,
            "iteration penalty root ranks")

synthetic <- fastkpc_full_cuda_fixed_sp_native_dto(batches[[1L]]$setup)
synthetic$n <- 5L
synthetic$coefficient_dim <- 4L
synthetic$null_dim <- 3L
synthetic_q <- synthetic$null_dim
projected_H <- diag(c(
  1,
  0.60 * synthetic_q * .Machine$double.eps,
  0.90 * synthetic_q * .Machine$double.eps
))
weak_design_scales <- sqrt(diag(projected_H)[2:3])
synthetic$X <- matrix(0, synthetic$n, synthetic$coefficient_dim)
synthetic$X[1L, 2L] <- weak_design_scales[[1L]]
synthetic$X[2L, 3L] <- weak_design_scales[[2L]]
synthetic$X[, 4L] <- c(3, 0, -2, 1, 2)
synthetic$constraint_mode <- "explicit"
synthetic$constraint_nullspace <- rbind(diag(3), rep(0, 3))
synthetic$gram_matrix <- crossprod(synthetic$X)
synthetic$nullspace_gram_matrix <- crossprod(
  synthetic$X %*% synthetic$constraint_nullspace
)
smooth_direction <- synthetic$constraint_nullspace[, 1L]
synthetic$penalty_count <- 1L
synthetic$penalty_blocks <- list(
  penalty_1 = 4 * tcrossprod(smooth_direction)
)
synthetic$penalty_offsets_zero_based <- 0L
synthetic$penalty_ranks <- 1L
synthetic$penalty_sp_indices_zero_based <- 0L
synthetic$penalty_sp_labels <- "synthetic-sp"
synthetic$H <- synthetic$constraint_nullspace %*%
  projected_H %*% t(synthetic$constraint_nullspace)
synthetic_y <- c(1, -2, 0, 0, 0)

synthetic_handle <- fixed_sp_cuda_prepared_create(runtime, synthetic)
synthetic_token <- NULL
on.exit({
  if (!is.null(synthetic_token)) {
    try(fixed_sp_cuda_residual_release(synthetic_token), silent = TRUE)
    try(fixed_sp_cuda_residual_free(synthetic_token), silent = TRUE)
  }
  try(fixed_sp_cuda_prepared_free(synthetic_handle), silent = TRUE)
}, add = TRUE)
synthetic_roots <- fixed_sp_cuda_prepared_materialize_roots_for_test(
  synthetic_handle
)
synthetic_info <- fixed_sp_cuda_prepared_info(synthetic_handle)

assert_true(
  length(synthetic_roots$penalty_roots) == 1L &&
    nrow(synthetic_roots$penalty_roots[[1L]]) == 1L,
  "synthetic explicit constraint preserves the declared smooth rank"
)
assert_reconstructs(
  synthetic_roots$penalty_roots[[1L]], projected_penalty(synthetic, 1L),
  1e-10, "synthetic smooth root reconstruction"
)
expected_H <- crossprod(
  synthetic$constraint_nullspace,
  synthetic$H %*% synthetic$constraint_nullspace
)
assert_true(
  is.matrix(synthetic_roots$H_root) &&
    nrow(synthetic_roots$H_root) == synthetic_roots$H_root_rank &&
    synthetic_roots$H_root_rank == 1L &&
    ncol(synthetic_roots$H_root) == synthetic$null_dim,
  "q-epsilon H root drops both small projected-H directions"
)
expected_truncated_H <- diag(c(1, 0, 0))
# The retained unit eigenpair is exact; 64 eps permits only solver roundoff.
H_root_reconstruction_tolerance <- 64 * .Machine$double.eps
assert_reconstructs(
  synthetic_roots$H_root, expected_truncated_H,
  H_root_reconstruction_tolerance,
  "compact H root reconstructs the expected rank-one truncation"
)
truncated_projected_H <- crossprod(synthetic_roots$H_root)
assert_true(
  max_abs_difference(truncated_projected_H, projected_H) >
    0.5 * projected_H[3L, 3L],
  "the compact H root remains numerically distinct from exact projected_H"
)
exact_H_factor <- cpu_pivoted_cholesky_root(projected_H)
truncated_H_factor <- cpu_pivoted_cholesky_root(truncated_projected_H)
assert_true(
  identical(exact_H_factor$rank, 3L) &&
    identical(exact_H_factor$pivot, c(1L, 3L, 2L)) &&
    identical(truncated_H_factor$rank, 1L) &&
    identical(truncated_H_factor$pivot, c(1L, 2L, 3L)),
  "LAPACK distinguishes exact projected_H from its truncated H root"
)
synthetic_X_null <- synthetic$X %*% synthetic$constraint_nullspace
exact_H_reference <- cpu_augmented_svd_reference(
  synthetic_X_null, exact_H_factor$root, synthetic_y
)
truncated_H_reference <- cpu_augmented_svd_reference(
  synthetic_X_null, synthetic_roots$H_root, synthetic_y
)
output_sensitivity_margin <- 0.5
cpu_fitted_sensitivity <- max_abs_difference(
  exact_H_reference$fitted, truncated_H_reference$fitted
)
cpu_residual_sensitivity <- max_abs_difference(
  exact_H_reference$residuals, truncated_H_reference$residuals
)
assert_true(
  identical(exact_H_reference$effective_rank, 3L) &&
    identical(truncated_H_reference$effective_rank, 3L) &&
    abs(exact_H_reference$sigma_max - 1) <
      H_root_reconstruction_tolerance &&
    abs(truncated_H_reference$sigma_max - 1) <
      H_root_reconstruction_tolerance &&
    cpu_fitted_sensitivity > output_sensitivity_margin &&
    cpu_residual_sensitivity > output_sensitivity_margin,
  paste0(
    "exact and truncated CPU augmented-SVD references are output-sensitive; ",
    "fitted margin=", format(cpu_fitted_sensitivity, digits = 17L),
    "; residual margin=", format(cpu_residual_sensitivity, digits = 17L)
  )
)
assert_true(
  is.integer(synthetic_info$projected_H_test_shadow_d2h_count) &&
    identical(synthetic_info$projected_H_test_shadow_d2h_count, 0L) &&
    is.double(synthetic_info$projected_H_test_shadow_d2h_bytes) &&
    identical(synthetic_info$projected_H_test_shadow_d2h_bytes, 0),
  "projected-H observer diagnostics start at exact typed zeros"
)
synthetic_static_shadow <- .Call(
  "C_fixed_sp_cuda_test_prepared_static_shadow", synthetic_handle,
  PACKAGE = "fastkpc_cuda"
)
synthetic_info_after_projected_shadow <- fixed_sp_cuda_prepared_info(
  synthetic_handle
)
assert_true(
  identical(names(synthetic_static_shadow),
            c("X_null", "gram", "projected_penalties", "projected_H")) &&
    identical(synthetic_static_shadow$projected_H, projected_H) &&
    identical(expected_H, projected_H),
  "test-only static shadow exactly exposes projected_H"
)
assert_true(
  synthetic_info_after_projected_shadow$projected_H_test_shadow_d2h_count -
      synthetic_info$projected_H_test_shadow_d2h_count == 1L &&
    synthetic_info_after_projected_shadow$projected_H_test_shadow_d2h_bytes -
      synthetic_info$projected_H_test_shadow_d2h_bytes ==
        8 * synthetic_q * synthetic_q,
  "projected-H observer has one separate exact q-square D2H delta"
)
assert_true(
  synthetic_info$penalty_root_build_count == 1L &&
    synthetic_info$penalty_root_rank_mismatch_count == 0L &&
    synthetic_info$penalty_root_matrix_count == 1L &&
    synthetic_info$penalty_root_row_count == 1L &&
    synthetic_info$H_root_matrix_count == 1L &&
    synthetic_info$H_root_rank == 1L,
  "synthetic root diagnostics separate smooth and H matrices"
)

prepared_before_H_solve <- fixed_sp_cuda_prepared_info(synthetic_handle)
runtime_before_H_solve <- fixed_sp_cuda_runtime_info(runtime)
synthetic_token <- fixed_sp_cuda_solve_batch(
  synthetic_handle,
  matrix(synthetic_y, nrow = synthetic$n, ncol = 1L),
  matrix(0, nrow = 1L, ncol = 1L),
  "AUGMENTED_SVD", strrep("a", 64L),
  outputs = c("fitted", "residuals")
)
H_solve_info <- fixed_sp_cuda_residual_info(synthetic_token)
runtime_after_H_solve <- fixed_sp_cuda_runtime_info(runtime)
prepared_after_H_solve <- fixed_sp_cuda_prepared_info(synthetic_handle)
required_aggregate_fields <- c(
  "aggregate_penalty_root_rank", "aggregate_penalty_root_pivot",
  "aggregate_factor_call_count", "aggregate_b_build_count",
  "aggregate_dstop", "aggregate_penalty_factor_count",
  "aggregate_svd_b_build_count", "aggregate_penalty_root_d2h_count"
)
missing_aggregate_fields <- setdiff(required_aggregate_fields,
                                    names(H_solve_info))
assert_true(
  length(missing_aggregate_fields) == 0L,
  paste0("non-null-H aggregate diagnostics are fail-closed: ",
         paste(missing_aggregate_fields, collapse = ","))
)
assert_true(
  identical(H_solve_info$planned_route, "AUGMENTED_SVD") &&
    identical(H_solve_info$executed_route, "AUGMENTED_SVD") &&
    identical(H_solve_info$reroute_reason, "") &&
    identical(H_solve_info$solver_status, "OK_AUGMENTED_SVD") &&
    is.integer(H_solve_info$aggregate_penalty_root_rank) &&
    identical(H_solve_info$aggregate_penalty_root_rank, 3L) &&
    is.list(H_solve_info$aggregate_penalty_root_pivot) &&
    length(H_solve_info$aggregate_penalty_root_pivot) == 1L &&
    identical(H_solve_info$aggregate_penalty_root_pivot[[1L]],
              c(1L, 3L, 2L)) &&
    is.integer(H_solve_info$aggregate_factor_call_count) &&
    identical(H_solve_info$aggregate_factor_call_count, 1L) &&
    is.integer(H_solve_info$aggregate_b_build_count) &&
    identical(H_solve_info$aggregate_b_build_count, 2L) &&
    is.double(H_solve_info$aggregate_dstop) &&
    identical(
      H_solve_info$aggregate_dstop,
      as.double(synthetic_q * (.Machine$double.eps / 2))
    ) &&
    identical(H_solve_info$aggregate_penalty_factor_count, 1L) &&
    identical(H_solve_info$aggregate_svd_b_build_count, 2L) &&
    identical(H_solve_info$aggregate_penalty_root_d2h_count, 0L),
  "public SVD factors exact projected_H with rank 3 and pivot 1,3,2"
)
synthetic_shadow <- fixed_sp_cuda_materialize_shadow(
  synthetic_token, outputs = c("fitted", "residuals")
)
cuda_fitted <- as.vector(synthetic_shadow$fitted[, 1L])
cuda_residuals <- as.vector(synthetic_shadow$residuals[, 1L])
exact_reference_tolerance <- 1e-7
cuda_exact_fitted_error <- max_abs_difference(
  cuda_fitted, exact_H_reference$fitted
)
cuda_exact_residual_error <- max_abs_difference(
  cuda_residuals, exact_H_reference$residuals
)
cuda_truncated_fitted_gap <- max_abs_difference(
  cuda_fitted, truncated_H_reference$fitted
)
cuda_truncated_residual_gap <- max_abs_difference(
  cuda_residuals, truncated_H_reference$residuals
)
assert_true(
  identical(dim(synthetic_shadow$fitted), c(synthetic$n, 1L)) &&
    identical(dim(synthetic_shadow$residuals), c(synthetic$n, 1L)) &&
    is.finite(cuda_exact_fitted_error) &&
    cuda_exact_fitted_error < exact_reference_tolerance &&
    is.finite(cuda_exact_residual_error) &&
    cuda_exact_residual_error < exact_reference_tolerance &&
    is.finite(cuda_truncated_fitted_gap) &&
    cuda_truncated_fitted_gap > output_sensitivity_margin &&
    is.finite(cuda_truncated_residual_gap) &&
    cuda_truncated_residual_gap > output_sensitivity_margin,
  paste0(
    "public SVD outputs consume exact projected_H; exact fitted error=",
    format(cuda_exact_fitted_error, digits = 17L),
    "; exact residual error=",
    format(cuda_exact_residual_error, digits = 17L),
    "; truncated fitted gap=",
    format(cuda_truncated_fitted_gap, digits = 17L),
    "; truncated residual gap=",
    format(cuda_truncated_residual_gap, digits = 17L)
  )
)
assert_true(
  identical(runtime_after_H_solve$workspace_grow_count,
            runtime_before_H_solve$workspace_grow_count) &&
    identical(runtime_after_H_solve$workspace_bytes,
              runtime_before_H_solve$workspace_bytes) &&
    identical(runtime_after_H_solve$cuda_device_synchronize_count,
              runtime_before_H_solve$cuda_device_synchronize_count) &&
    runtime_after_H_solve$svd_checkpoint_record_count -
      runtime_before_H_solve$svd_checkpoint_record_count == 1L &&
    runtime_after_H_solve$svd_checkpoint_wait_count -
      runtime_before_H_solve$svd_checkpoint_wait_count == 1L,
  "non-null-H aggregate SVD uses one checkpoint and no runtime growth"
)
assert_true(
  identical(prepared_after_H_solve$setup_shadow_d2h_count,
            prepared_before_H_solve$setup_shadow_d2h_count) &&
    identical(prepared_after_H_solve$setup_shadow_d2h_bytes,
              prepared_before_H_solve$setup_shadow_d2h_bytes) &&
    identical(
      prepared_after_H_solve$projected_H_test_shadow_d2h_count,
      prepared_before_H_solve$projected_H_test_shadow_d2h_count
    ) && identical(
      prepared_after_H_solve$projected_H_test_shadow_d2h_bytes,
      prepared_before_H_solve$projected_H_test_shadow_d2h_bytes
    ) &&
    identical(prepared_after_H_solve$augmented_test_shadow_d2h_count,
              prepared_before_H_solve$augmented_test_shadow_d2h_count),
  "production aggregate matrix and root paths perform no D2H observation"
)
fixed_sp_cuda_residual_release(synthetic_token)
fixed_sp_cuda_residual_free(synthetic_token)
synthetic_token <- NULL

cholesky <- synthetic
cholesky$n <- 3L
cholesky$coefficient_dim <- 3L
cholesky$null_dim <- 2L
cholesky$X <- rbind(c(1, 0, 0), c(0, 2, 0), c(1, 1, 1))
cholesky$constraint_mode <- "explicit"
cholesky$constraint_nullspace <- rbind(diag(2), c(0, 0))
cholesky$gram_matrix <- crossprod(cholesky$X)
cholesky_X_null <- cholesky$X %*% cholesky$constraint_nullspace
cholesky$nullspace_gram_matrix <- crossprod(cholesky_X_null)
cholesky$penalty_count <- 1L
cholesky$penalty_blocks <- list(penalty_1 = diag(c(2, 1)))
cholesky$penalty_offsets_zero_based <- 0L
cholesky$penalty_ranks <- 2L
cholesky$penalty_sp_indices_zero_based <- 0L
cholesky$penalty_sp_labels <- "cholesky-sp"
cholesky$H <- cholesky$constraint_nullspace %*%
  diag(c(4, 2)) %*% t(cholesky$constraint_nullspace)
cholesky_y <- c(2, -1, 3)
cholesky_sp <- 0.5
cholesky_projected_penalties <- list(projected_penalty(cholesky, 1L))
cholesky_projected_H <- crossprod(
  cholesky$constraint_nullspace,
  cholesky$H %*% cholesky$constraint_nullspace
)
cholesky_exact_H_reference <- cpu_fixed_sp_reference(
  cholesky_X_null, cholesky_projected_penalties, cholesky_projected_H,
  cholesky_sp, cholesky_y
)
cholesky_no_H_reference <- cpu_fixed_sp_reference(
  cholesky_X_null, cholesky_projected_penalties,
  matrix(0, cholesky$null_dim, cholesky$null_dim),
  cholesky_sp, cholesky_y
)
cholesky_H_sensitivity <- max_abs_difference(
  cholesky_exact_H_reference$fitted, cholesky_no_H_reference$fitted
)
assert_true(
  cholesky_H_sensitivity > output_sensitivity_margin,
  paste0(
    "CPU fixed-sp reference is sensitive to exact projected_H; fitted margin=",
    format(cholesky_H_sensitivity, digits = 17L)
  )
)
cholesky_handle <- NULL
cholesky_token <- NULL
on.exit({
  if (!is.null(cholesky_token)) {
    try(fixed_sp_cuda_residual_release(cholesky_token), silent = TRUE)
    try(fixed_sp_cuda_residual_free(cholesky_token), silent = TRUE)
  }
  if (!is.null(cholesky_handle)) {
    try(fixed_sp_cuda_prepared_free(cholesky_handle), silent = TRUE)
  }
}, add = TRUE)
cholesky_handle <- fixed_sp_cuda_prepared_create(runtime, cholesky)
cholesky_token <- fixed_sp_cuda_solve_batch(
  cholesky_handle,
  matrix(cholesky_y, nrow = cholesky$n, ncol = 1L),
  matrix(cholesky_sp, nrow = 1L, ncol = 1L),
  "CHOLESKY_BATCHED", strrep("c", 64L),
  outputs = c("fitted", "residuals")
)
cholesky_H_info <- fixed_sp_cuda_residual_info(cholesky_token)
cholesky_H_shadow <- fixed_sp_cuda_materialize_shadow(
  cholesky_token, outputs = c("fitted", "residuals")
)
cholesky_cuda_fitted <- as.vector(cholesky_H_shadow$fitted[, 1L])
cholesky_cuda_residuals <- as.vector(cholesky_H_shadow$residuals[, 1L])
cholesky_exact_fitted_error <- max_abs_difference(
  cholesky_cuda_fitted, cholesky_exact_H_reference$fitted
)
cholesky_exact_residual_error <- max_abs_difference(
  cholesky_cuda_residuals, cholesky_exact_H_reference$residuals
)
cholesky_no_H_fitted_gap <- max_abs_difference(
  cholesky_cuda_fitted, cholesky_no_H_reference$fitted
)
assert_true(
  identical(cholesky_H_info$planned_route, "CHOLESKY_BATCHED") &&
    identical(cholesky_H_info$executed_route, "CHOLESKY_BATCHED") &&
    identical(cholesky_H_info$reroute_reason, "") &&
    identical(cholesky_H_info$solver_status, "OK_CHOLESKY_SINGLE") &&
    cholesky_exact_fitted_error < exact_reference_tolerance &&
    cholesky_exact_residual_error < exact_reference_tolerance &&
    cholesky_no_H_fitted_gap > output_sensitivity_margin,
  paste0(
    "planned Cholesky consumes exact projected_H; exact fitted error=",
    format(cholesky_exact_fitted_error, digits = 17L),
    "; exact residual error=",
    format(cholesky_exact_residual_error, digits = 17L),
    "; no-H fitted gap=",
    format(cholesky_no_H_fitted_gap, digits = 17L),
    "; status=", cholesky_H_info$solver_status
  )
)
fixed_sp_cuda_residual_release(cholesky_token)
fixed_sp_cuda_residual_free(cholesky_token)
cholesky_token <- NULL
fixed_sp_cuda_prepared_free(cholesky_handle)
cholesky_handle <- NULL

asymmetric_smooth <- synthetic
asymmetric_smooth["H"] <- list(NULL)
asymmetric_smooth$penalty_blocks <- list(penalty_1 = diag(c(4, 2, 1)))
asymmetric_smooth$penalty_blocks[[1L]][1L, 2L] <- 0.125
asymmetric_smooth$penalty_ranks <- 2L
asymmetric_smooth_error <- capture_prepared_create_error(
  runtime, asymmetric_smooth
)

asymmetric_H <- synthetic
asymmetric_H$H <- synthetic$constraint_nullspace %*%
  diag(c(9, 4, 1)) %*% t(synthetic$constraint_nullspace)
asymmetric_H$H[1L, 2L] <- asymmetric_H$H[1L, 2L] + 1e-4
asymmetric_H_error <- capture_prepared_create_error(runtime, asymmetric_H)

behavior_errors <- character()
if (!error_matches(
  asymmetric_smooth_error,
  "prepared smooth penalty block must be symmetric"
)) {
  behavior_errors <- c(
    behavior_errors,
    paste0("asymmetric smooth penalty: ",
           error_description(asymmetric_smooth_error))
  )
}
if (!error_matches(asymmetric_H_error, "prepared H must be symmetric")) {
  behavior_errors <- c(
    behavior_errors,
    paste0("asymmetric H: ", error_description(asymmetric_H_error))
  )
}
assert_true(
  length(behavior_errors) == 0L,
  paste0("prepared safety contract failures: ",
         paste(behavior_errors, collapse = "; "))
)

post_solve_info <- fixed_sp_cuda_prepared_info(synthetic_handle)
assert_true(
  post_solve_info$penalty_root_build_count == 1L &&
    post_solve_info$setup_shadow_d2h_count == 1L &&
    post_solve_info$projected_H_test_shadow_d2h_count == 1L &&
    post_solve_info$projected_H_test_shadow_d2h_bytes ==
      8 * synthetic_q * synthetic_q &&
    !isTRUE(post_solve_info$output_slot_leased),
  "materialization and H solve preserve roots, observers, and output lease"
)

fixed_sp_cuda_prepared_free(synthetic_handle)
synthetic_handle <- NULL

rank_mismatch <- synthetic
rank_mismatch["H"] <- list(NULL)
rank_mismatch$penalty_ranks <- 2L
assert_error(
  fixed_sp_cuda_prepared_create(runtime, rank_mismatch),
  "rank mismatch", "authenticated smooth-root rank mismatch fails setup"
)

undersized_rank <- synthetic
undersized_rank["H"] <- list(NULL)
undersized_rank$penalty_ranks <- 0L
assert_error(
  fixed_sp_cuda_prepared_create(runtime, undersized_rank),
  "rank mismatch",
  "derived smooth rank above declared root capacity fails setup"
)

non_psd_H <- synthetic
non_psd_H$H <- synthetic$constraint_nullspace %*%
  diag(c(-1, 1, 1)) %*% t(synthetic$constraint_nullspace)
assert_error(
  fixed_sp_cuda_prepared_create(runtime, non_psd_H),
  "non-PSD H", "non-PSD H fails setup"
)

fixed_sp_cuda_runtime_free(runtime)

cat("PASS Phase 3C fixed-sp penalty roots; iteration matrices=159 rows=1424\n")
