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
assert_close <- function(actual, expected, tolerance, message) {
  assert_true(identical(dim(actual), dim(expected)),
              paste0(message, ": dimensions"))
  error <- max(abs(actual - expected))
  assert_true(
    is.finite(error) && error <= tolerance,
    paste0(message, ": max error ", format(error, digits = 17L))
  )
}
assert_vector_close <- function(actual, expected, tolerance, message) {
  assert_true(
    is.double(actual) && is.double(expected) &&
      identical(length(actual), length(expected)),
    paste0(message, ": vector contract")
  )
  error <- max(abs(actual - expected))
  assert_true(
    is.finite(error) && error <= tolerance,
    paste0(message, ": max error ", format(error, digits = 17L))
  )
}

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP Phase 3C fixed-sp augmented system\n")
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

find_route_representative <- function(route) {
  for (batch in batches) {
    dto <- fastkpc_full_cuda_fixed_sp_native_dto(batch$setup)
    native <- fastkpc_full_cuda_fixed_sp_native_batch(batch, dto)
    target_index <- which(native$planned_route == route)
    if (length(target_index) > 0L && dto$penalty_ranks[[1L]] > 0L) {
      target_index <- target_index[[1L]]
      return(list(
        dto = dto,
        native = native,
        target_index = as.integer(target_index),
        target_key = native$target_keys[[target_index]],
        route = route
      ))
    }
  }
  fail(paste("missing authenticated iteration representative for", route))
}

representatives <- list(
  find_route_representative("AUGMENTED_QR"),
  find_route_representative("AUGMENTED_SVD")
)

capacities <- fastkpc_full_cuda_fixed_sp_contract()$canonical_capacities
runtime <- fixed_sp_cuda_runtime_create(0L)
on.exit(try(fixed_sp_cuda_runtime_free(runtime), silent = TRUE), add = TRUE)
fixed_sp_cuda_runtime_reserve(
  runtime, capacities$n, capacities$null_dim, capacities$target_count,
  capacities$penalty_count, capacities$augmented_rows
)

expected_X_null <- function(dto) {
  if (identical(dto$constraint_mode, "identity")) {
    dto$X
  } else {
    dto$X %*% dto$constraint_nullspace
  }
}

expected_augmented_crossprod <- function(dto, roots, SP) {
  expected <- matrix(0, dto$null_dim, dto$null_dim)
  for (penalty_index in seq_len(dto$penalty_count)) {
    expected <- expected +
      SP[[penalty_index]] *
      crossprod(roots$penalty_roots[[penalty_index]])
  }
  if (!is.null(roots$H_root)) {
    expected <- expected + crossprod(roots$H_root)
  }
  expected
}

assert_augmented <- function(augmented, dto, roots, Y, SP, target_index,
                             label) {
  smooth_rows <- sum(dto$penalty_ranks)
  root_rows <- smooth_rows + roots$H_root_rank
  rows <- dto$n + root_rows
  assert_true(
    identical(dim(augmented$B), as.integer(c(rows, dto$null_dim))),
    paste(label, "B dimensions")
  )
  assert_true(
    is.double(augmented$c) && length(augmented$c) == rows,
    paste(label, "c dimensions")
  )
  assert_true(
    identical(augmented$leading_dimension, as.integer(rows)) &&
      identical(augmented$rows, as.integer(rows)) &&
      identical(augmented$cols, dto$null_dim) &&
      identical(augmented$target_index, as.integer(target_index)),
    paste(label, "augmented view metadata")
  )
  assert_close(
    augmented$B[seq_len(dto$n), , drop = FALSE], expected_X_null(dto),
    1e-12, paste(label, "top design rows")
  )
  assert_vector_close(
    augmented$c[seq_len(dto$n)], Y, 1e-12,
    paste(label, "response rows")
  )
  assert_true(
    root_rows > 0L &&
      all(augmented$c[dto$n + seq_len(root_rows)] == 0),
    paste(label, "augmented response tail is exactly zero")
  )

  row_offset <- dto$n
  for (penalty_index in seq_len(dto$penalty_count)) {
    rank <- dto$penalty_ranks[[penalty_index]]
    if (rank > 0L) {
      block_rows <- row_offset + seq_len(rank)
      assert_close(
        augmented$B[block_rows, , drop = FALSE],
        sqrt(SP[[penalty_index]]) *
          roots$penalty_roots[[penalty_index]],
        1e-12, paste(label, "smooth root block", penalty_index)
      )
    }
    row_offset <- row_offset + rank
  }
  if (roots$H_root_rank > 0L) {
    H_rows <- row_offset + seq_len(roots$H_root_rank)
    assert_close(
      augmented$B[H_rows, , drop = FALSE], roots$H_root,
      1e-12, paste(label, "H root block")
    )
  }

  penalty_rows <- dto$n + seq_len(root_rows)
  actual_crossprod <- crossprod(
    augmented$B[penalty_rows, , drop = FALSE]
  )
  expected_crossprod <- expected_augmented_crossprod(dto, roots, SP)
  reconstruction_absolute_error <-
    max(abs(actual_crossprod - expected_crossprod))
  reconstruction_scale <- max(1, max(abs(expected_crossprod)))
  reconstruction_error <-
    reconstruction_absolute_error / reconstruction_scale
  assert_true(
    is.finite(reconstruction_error) && reconstruction_error < 1e-8,
    paste0(label, " augmented penalty reconstruction: relative error ",
           format(reconstruction_error, digits = 17L),
           ", absolute error ",
           format(reconstruction_absolute_error, digits = 17L))
  )
  invisible(augmented)
}

unchanged_runtime_fields <- c(
  "cuda_device_allocation_count", "cuda_host_allocation_count",
  "stream_create_count", "event_create_count", "workspace_reserve_count",
  "workspace_grow_count", "stable_workspace_grow_count", "workspace_bytes",
  "augmented_workspace_bytes"
)

exercise_case <- function(dto, Y, SP, target_index, label,
                          test_zero_sp = FALSE,
                          test_invalid_sp = FALSE) {
  assert_true(
    identical(dto$weights_policy, "none-or-unit") &&
      identical(dto$offset_policy, "none-or-zero"),
    paste(label, "authenticated neutral weights and offsets")
  )
  handle <- fixed_sp_cuda_prepared_create(runtime, dto)
  on.exit(try(fixed_sp_cuda_prepared_free(handle), silent = TRUE), add = TRUE)
  roots <- fixed_sp_cuda_prepared_materialize_roots_for_test(handle)
  root_rows <- sum(dto$penalty_ranks) + roots$H_root_rank
  expected_shadow_bytes <-
    (root_rows + dto$n) * (dto$null_dim + 1) * 8
  before_handle <- fixed_sp_cuda_prepared_info(handle)
  before_runtime <- fixed_sp_cuda_runtime_info(runtime)
  assert_true(
    !isTRUE(before_handle$output_slot_leased) &&
      identical(before_handle$output_slot_state, "free"),
    paste(label, "output slot starts free")
  )

  augmented <- fixed_sp_cuda_build_augmented_for_test(
    handle, Y, SP, target_index = target_index
  )
  assert_augmented(augmented, dto, roots, Y, SP, target_index, label)
  repeated <- fixed_sp_cuda_build_augmented_for_test(
    handle, Y, SP, target_index = target_index
  )
  assert_augmented(repeated, dto, roots, Y, SP, target_index,
                   paste(label, "repeated"))
  assert_true(
    identical(augmented$B, repeated$B) &&
      identical(augmented$c, repeated$c),
    paste(label, "repeated device build is deterministic")
  )
  successful_build_count <- 2L

  if (test_zero_sp) {
    zero_SP <- SP
    zero_SP[[1L]] <- 0
    zero_augmented <- fixed_sp_cuda_build_augmented_for_test(
      handle, Y, zero_SP, target_index = target_index
    )
    assert_augmented(
      zero_augmented, dto, roots, Y, zero_SP, target_index,
      paste(label, "zero SP")
    )
    first_root_rows <- dto$n + seq_len(dto$penalty_ranks[[1L]])
    assert_true(
      all(zero_augmented$B[first_root_rows, , drop = FALSE] == 0),
      paste(label, "SP[1]=0 emits an exactly zero first root block")
    )
    successful_build_count <- successful_build_count + 1L
  }

  if (test_invalid_sp) {
    diagnostics_before_rejection <- fixed_sp_cuda_prepared_info(handle)
    negative_SP <- SP
    negative_SP[[1L]] <- -1
    assert_error(
      fixed_sp_cuda_build_augmented_for_test(
        handle, Y, negative_SP, target_index = target_index
      ),
      "finite non-negative", paste(label, "negative SP rejection")
    )
    nonfinite_SP <- SP
    nonfinite_SP[[1L]] <- Inf
    assert_error(
      fixed_sp_cuda_build_augmented_for_test(
        handle, Y, nonfinite_SP, target_index = target_index
      ),
      "finite non-negative", paste(label, "nonfinite SP rejection")
    )
    assert_error(
      fixed_sp_cuda_build_augmented_for_test(
        handle, Y, SP, target_index = 0L
      ),
      "target_index", paste(label, "target index rejection")
    )
    diagnostics_after_rejection <- fixed_sp_cuda_prepared_info(handle)
    assert_true(
      identical(
        diagnostics_after_rejection$augmented_test_shadow_d2h_count,
        diagnostics_before_rejection$augmented_test_shadow_d2h_count
      ) && identical(
        diagnostics_after_rejection$augmented_test_shadow_d2h_bytes,
        diagnostics_before_rejection$augmented_test_shadow_d2h_bytes
      ),
      paste(label, "rejected SP does not materialize a test shadow")
    )
  }

  after_handle <- fixed_sp_cuda_prepared_info(handle)
  after_runtime <- fixed_sp_cuda_runtime_info(runtime)
  assert_true(
    !isTRUE(after_handle$output_slot_leased) &&
      identical(after_handle$output_slot_state, "free") &&
      after_handle$penalty_root_build_count == 1L &&
      identical(after_handle$setup_shadow_d2h_count,
                before_handle$setup_shadow_d2h_count),
    paste(label, "test builder does not solve, lease output, or rebuild roots")
  )
  assert_true(
    identical(
      after_handle$augmented_test_shadow_d2h_count -
        before_handle$augmented_test_shadow_d2h_count,
      as.integer(successful_build_count)
    ) && identical(
      after_handle$augmented_test_shadow_d2h_bytes -
        before_handle$augmented_test_shadow_d2h_bytes,
      as.numeric(successful_build_count * expected_shadow_bytes)
    ),
    paste(label, "augmented test-shadow D2H diagnostics")
  )
  assert_true(
    all(vapply(unchanged_runtime_fields, function(field) {
      identical(before_runtime[[field]], after_runtime[[field]])
    }, logical(1L))),
    paste(label, "repeated builds do not allocate or grow workspace")
  )

  fixed_sp_cuda_prepared_free(handle)
  invisible(NULL)
}

for (representative_index in seq_along(representatives)) {
  representative <- representatives[[representative_index]]
  target_index <- representative$target_index
  exercise_case(
    representative$dto,
    as.numeric(representative$native$Y[, target_index]),
    as.numeric(representative$native$SP[, target_index]),
    target_index,
    paste(representative$route, representative$target_key),
    test_zero_sp = representative_index == 1L,
    test_invalid_sp = representative_index == 1L
  )
}

synthetic <- representatives[[1L]]$dto
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
  diag(c(9, 0)) %*% t(synthetic$constraint_nullspace)

exercise_case(
  synthetic, c(1, -2, 0.5, 4, -3), 2.25, 1L,
  "synthetic explicit constraint with non-null H", test_zero_sp = TRUE
)

fixed_sp_cuda_runtime_free(runtime)

cat(paste0(
  "PASS Phase 3C fixed-sp augmented systems; QR=",
  representatives[[1L]]$target_key, " SVD=",
  representatives[[2L]]$target_key, "\n"
))
