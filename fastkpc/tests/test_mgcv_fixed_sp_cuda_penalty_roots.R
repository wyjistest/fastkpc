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
for (batch in batches) {
  dto <- fastkpc_full_cuda_fixed_sp_native_dto(batch$setup)
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

  total_root_matrix_count <- total_root_matrix_count +
    info$penalty_root_matrix_count
  total_root_row_count <- total_root_row_count + info$penalty_root_row_count
  total_rank_mismatch_count <- total_rank_mismatch_count +
    info$penalty_root_rank_mismatch_count
  fixed_sp_cuda_prepared_free(handle)
}

assert_true(total_root_matrix_count == 159L,
            "iteration penalty root matrix count")
assert_true(total_root_row_count == 1424L,
            "iteration penalty root row count")
assert_true(total_rank_mismatch_count == 0L,
            "iteration penalty root ranks")

synthetic <- fastkpc_full_cuda_fixed_sp_native_dto(batches[[1L]]$setup)
synthetic$n <- 5L
synthetic$coefficient_dim <- 3L
synthetic$null_dim <- 2L
synthetic$X <- matrix(c(
  1, 2, 3, 4, 5,
  2, -1, 0, 1, 3,
  -1, 1, 2, 0, 4
), nrow = synthetic$n, ncol = synthetic$coefficient_dim)
synthetic$constraint_mode <- "explicit"
synthetic$constraint_nullspace <- cbind(
  c(1, 1, 0) / sqrt(2),
  c(1, -1, 2) / sqrt(6)
)
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
  diag(c(9, 4)) %*% t(synthetic$constraint_nullspace)

synthetic_handle <- fixed_sp_cuda_prepared_create(runtime, synthetic)
on.exit(try(fixed_sp_cuda_prepared_free(synthetic_handle), silent = TRUE),
        add = TRUE)
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
    synthetic_roots$H_root_rank == 2L,
  "synthetic H root has the derived PSD rank"
)
assert_reconstructs(
  synthetic_roots$H_root, expected_H, 1e-10,
  "synthetic H root reconstruction"
)
assert_true(
  synthetic_info$penalty_root_build_count == 1L &&
    synthetic_info$penalty_root_rank_mismatch_count == 0L &&
    synthetic_info$penalty_root_matrix_count == 1L &&
    synthetic_info$penalty_root_row_count == 1L &&
    synthetic_info$H_root_matrix_count == 1L &&
    synthetic_info$H_root_rank == 2L,
  "synthetic root diagnostics separate smooth and H matrices"
)

fixed_sp_cuda_prepared_free(synthetic_handle)

rank_mismatch <- synthetic
rank_mismatch$penalty_ranks <- 2L
assert_error(
  fixed_sp_cuda_prepared_create(runtime, rank_mismatch),
  "rank mismatch", "authenticated smooth-root rank mismatch fails setup"
)

non_psd_H <- synthetic
non_psd_H$H <- synthetic$constraint_nullspace %*%
  diag(c(-1, 1)) %*% t(synthetic$constraint_nullspace)
assert_error(
  fixed_sp_cuda_prepared_create(runtime, non_psd_H),
  "non-PSD H", "non-PSD H fails setup"
)

fixed_sp_cuda_runtime_free(runtime)

cat("PASS Phase 3C fixed-sp penalty roots; iteration matrices=159 rows=1424\n")
