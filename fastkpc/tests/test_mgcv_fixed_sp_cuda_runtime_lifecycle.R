source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_error <- function(expression, pattern, message) {
  error <- tryCatch(force(expression), error = identity)
  assert_true(
    inherits(error, "error") &&
      grepl(pattern, conditionMessage(error), fixed = TRUE),
    message
  )
}

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP Phase 3 CUDA runtime lifecycle\n")
  quit(save = "no", status = 0)
}

build_fastkpc_cuda_native(rebuild = TRUE)
assert_true(fastkpc_cuda_available(), "CUDA must be available")

runtime <- fixed_sp_cuda_runtime_create(device_id = 0L)
on.exit(try(fixed_sp_cuda_runtime_free(runtime), silent = TRUE), add = TRUE)
before <- fixed_sp_cuda_runtime_info(runtime)
assert_true(before$stream_create_count == 1L, "one stream")
assert_true(before$cublas_handle_create_count == 1L, "one cuBLAS handle")
assert_true(before$cusolver_handle_create_count == 1L, "one cuSOLVER handle")
assert_true(identical(before$cusolver_deterministic_mode, "enabled"),
            "cuSOLVER deterministic mode")
assert_true(identical(before$cublas_math_mode, "pedantic"),
            "cuBLAS pedantic math")
assert_true(identical(before$cublas_atomics_mode, "not_allowed"),
            "cuBLAS atomics disabled")

fixed_sp_cuda_runtime_reserve(
  runtime, n = 351L, null_dim = 64L, target_count = 47L,
  penalty_count = 7L, augmented_rows = 407L
)
after <- fixed_sp_cuda_runtime_info(runtime)
assert_true(after$workspace_reserve_count == 1L, "one reserve")
assert_true(after$workspace_grow_count == 1L, "one workspace growth")
assert_true(after$workspace_bytes > 0, "workspace allocated")
assert_true(isTRUE(after$cublas_user_workspace_installed),
            "user cuBLAS workspace installed")
assert_true(after$cublas_workspace_bytes >= 16L * 1024L * 1024L,
            "cuBLAS workspace size")
assert_true(after$cublas_workspace_alignment >= 256L,
            "cuBLAS workspace alignment")
assert_true(after$compute_capability_major == 8L &&
              after$compute_capability_minor == 9L &&
              after$sm_count > 0L,
            "declared GPU identity")

fixed_sp_cuda_runtime_reserve(
  runtime, n = 351L, null_dim = 64L, target_count = 47L,
  penalty_count = 7L, augmented_rows = 407L
)
equal <- fixed_sp_cuda_runtime_info(runtime)
assert_true(equal$workspace_reserve_count == 2L, "equal reserve counted")
assert_true(equal$workspace_grow_count == 1L, "equal reserve does not grow")
assert_true(equal$workspace_bytes == after$workspace_bytes,
            "equal reserve keeps workspace")

fixed_sp_cuda_runtime_reserve(
  runtime, n = 128L, null_dim = 32L, target_count = 8L,
  penalty_count = 3L, augmented_rows = 160L
)
smaller <- fixed_sp_cuda_runtime_info(runtime)
assert_true(smaller$workspace_reserve_count == 3L, "smaller reserve counted")
assert_true(smaller$workspace_grow_count == 1L,
            "smaller reserve does not grow")
assert_true(smaller$workspace_bytes == after$workspace_bytes,
            "smaller reserve keeps workspace")

fixed_sp_cuda_runtime_reserve(
  runtime, n = 400L, null_dim = 32L, target_count = 47L,
  penalty_count = 7L, augmented_rows = 407L
)
cross_growth <- fixed_sp_cuda_runtime_info(runtime)
assert_true(cross_growth$workspace_reserve_count == 4L,
            "cross-dimension reserve counted")
assert_true(cross_growth$workspace_grow_count == 2L,
            "merged cross-dimension capacity grows")
assert_true(cross_growth$workspace_bytes > after$workspace_bytes,
            "merged cross-dimension workspace is larger")

fixed_sp_cuda_runtime_reserve(
  runtime, n = 400L, null_dim = 64L, target_count = 47L,
  penalty_count = 7L, augmented_rows = 407L
)
cross_merged <- fixed_sp_cuda_runtime_info(runtime)
assert_true(cross_merged$workspace_reserve_count == 5L,
            "merged maxima reserve counted")
assert_true(cross_merged$workspace_grow_count == 2L,
            "merged maxima reserve does not grow again")
assert_true(cross_merged$workspace_bytes == cross_growth$workspace_bytes,
            "merged maxima reserve reuses workspace")

assert_error(
  fixed_sp_cuda_runtime_create(c(0L, 1L)),
  "scalar integer", "device id must be scalar"
)
assert_error(
  fixed_sp_cuda_runtime_reserve(
    runtime, n = 0L, null_dim = 1L, target_count = 1L,
    penalty_count = 1L, augmented_rows = 1L
  ),
  "positive", "reserve capacities must be positive"
)
assert_error(
  fixed_sp_cuda_runtime_reserve(
    runtime, n = c(1L, 2L), null_dim = 1L, target_count = 1L,
    penalty_count = 1L, augmented_rows = 1L
  ),
  "scalar integer", "reserve inputs must be scalar"
)
assert_error(
  fixed_sp_cuda_runtime_reserve(
    runtime, n = 1L, null_dim = .Machine$integer.max,
    target_count = 1L, penalty_count = 1L, augmented_rows = 1L
  ),
  "size overflow", "reserve size overflow must fail before allocation"
)

fixed_sp_cuda_runtime_free(runtime)
fixed_sp_cuda_runtime_free(runtime)
assert_error(
  fixed_sp_cuda_runtime_info(runtime),
  "freed", "freed runtime must reject use"
)

cat("PASS Phase 3 CUDA runtime lifecycle\n")
