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

resource_snapshot <- function() {
  .Call("C_fixed_sp_cuda_test_resource_snapshot", PACKAGE = "fastkpc_cuda")
}
resource_delta <- function(before, after, field) {
  as.numeric(after[[field]] - before[[field]])
}
ledger_before <- resource_snapshot()

runtime <- fixed_sp_cuda_runtime_create(device_id = 0L)
on.exit(try(fixed_sp_cuda_runtime_free(runtime), silent = TRUE), add = TRUE)
before <- fixed_sp_cuda_runtime_info(runtime)
assert_true(is.character(before$gpu_name) && length(before$gpu_name) == 1L &&
              nzchar(before$gpu_name), "GPU name")
assert_true(before$cuda_device_allocation_count == 0L,
            "runtime creation performs no device allocation")
assert_true(before$cuda_host_allocation_count == 0L,
            "runtime creation performs no pinned-host allocation")
assert_true(before$stream_create_count == 1L, "one stream")
assert_true(before$event_create_count == 2L,
            "two runtime checkpoint events")
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
assert_true(after$cuda_device_allocation_count >
              before$cuda_device_allocation_count,
            "workspace reserve records device allocations")
assert_true(after$cuda_host_allocation_count >
              before$cuda_host_allocation_count,
            "workspace reserve records pinned-host allocations")
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

ledger_after <- resource_snapshot()
resources <- list(
  cuda_device = c("allocate", "free"),
  cuda_host = c("allocate", "free"),
  stream = c("create", "destroy"),
  event = c("create", "destroy"),
  cublas_handle = c("create", "destroy"),
  cusolver_handle = c("create", "destroy")
)
for (resource in names(resources)) {
  verbs <- resources[[resource]]
  acquire_attempt <- paste(resource, verbs[[1L]], "attempt_count", sep = "_")
  acquire_success <- paste(resource, verbs[[1L]], "success_count", sep = "_")
  acquire_failure <- paste(resource, verbs[[1L]], "failure_count", sep = "_")
  teardown_attempt <- paste(resource, verbs[[2L]], "attempt_count", sep = "_")
  teardown_success <- paste(resource, verbs[[2L]], "success_count", sep = "_")
  teardown_failure <- paste(resource, verbs[[2L]], "failure_count", sep = "_")
  active <- paste(resource, "active_count", sep = "_")
  acquired_delta <- resource_delta(
    ledger_before, ledger_after, acquire_success
  )
  assert_true(
    acquired_delta > 0 &&
      resource_delta(ledger_before, ledger_after, acquire_attempt) ==
        acquired_delta &&
      resource_delta(ledger_before, ledger_after, acquire_failure) == 0 &&
      resource_delta(ledger_before, ledger_after, teardown_attempt) ==
        acquired_delta &&
      resource_delta(ledger_before, ledger_after, teardown_success) ==
        acquired_delta &&
      resource_delta(ledger_before, ledger_after, teardown_failure) == 0 &&
      resource_delta(ledger_before, ledger_after, active) == 0,
    paste(resource, "creation and explicit teardown are balanced")
  )
}
assert_true(
  resource_delta(ledger_before, ledger_after, "cleanup_error_count") == 0,
  "scoped explicit cleanup records no teardown errors"
)

exercise_injected_acquire_failure <- function(resource) {
  .Call(
    "C_fixed_sp_cuda_test_inject_next_resource_acquire_failure",
    resource, PACKAGE = "fastkpc_cuda"
  )
  failed_runtime <- NULL
  on.exit({
    if (!is.null(failed_runtime)) {
      try(fixed_sp_cuda_runtime_free(failed_runtime), silent = TRUE)
    }
  }, add = TRUE)
  failed_runtime <- fixed_sp_cuda_runtime_create(device_id = 0L)
  if (resource %in% c("cuda_device", "cuda_host")) {
    fixed_sp_cuda_runtime_reserve(
      failed_runtime, n = 16L, null_dim = 8L, target_count = 1L,
      penalty_count = 1L, augmented_rows = 16L
    )
  }
  fail(paste(resource, "injected acquisition unexpectedly succeeded"))
}

for (resource in names(resources)) {
  failure_before <- resource_snapshot()
  assert_error(
    exercise_injected_acquire_failure(resource),
    "injected tracked fixed-sp resource acquire failure",
    paste(resource, "acquisition failure is surfaced")
  )
  failure_after <- resource_snapshot()
  verbs <- resources[[resource]]
  acquire_attempt <- paste(resource, verbs[[1L]], "attempt_count", sep = "_")
  acquire_success <- paste(resource, verbs[[1L]], "success_count", sep = "_")
  acquire_failure <- paste(resource, verbs[[1L]], "failure_count", sep = "_")
  assert_true(
    resource_delta(failure_before, failure_after, acquire_attempt) == 1 &&
      resource_delta(failure_before, failure_after, acquire_success) == 0 &&
      resource_delta(failure_before, failure_after, acquire_failure) == 1 &&
      all(vapply(names(resources), function(candidate) {
        active <- paste(candidate, "active_count", sep = "_")
        identical(failure_after[[active]], failure_before[[active]])
      }, logical(1L))) &&
      resource_delta(failure_before, failure_after,
                     "cleanup_error_count") == 0,
    paste(resource, "failed acquisition is counted and fully unwound")
  )
}

for (resource in names(resources)) {
  teardown_before <- resource_snapshot()
  invisible(.Call(
    "C_fixed_sp_cuda_test_exercise_resource_teardown_failure",
    resource, PACKAGE = "fastkpc_cuda"
  ))
  teardown_after <- resource_snapshot()
  verbs <- resources[[resource]]
  acquire_attempt <- paste(resource, verbs[[1L]], "attempt_count", sep = "_")
  acquire_success <- paste(resource, verbs[[1L]], "success_count", sep = "_")
  acquire_failure <- paste(resource, verbs[[1L]], "failure_count", sep = "_")
  teardown_attempt <- paste(resource, verbs[[2L]], "attempt_count", sep = "_")
  teardown_success <- paste(resource, verbs[[2L]], "success_count", sep = "_")
  teardown_failure <- paste(resource, verbs[[2L]], "failure_count", sep = "_")
  active <- paste(resource, "active_count", sep = "_")
  assert_true(
    resource_delta(teardown_before, teardown_after, acquire_attempt) == 1 &&
      resource_delta(teardown_before, teardown_after, acquire_success) == 1 &&
      resource_delta(teardown_before, teardown_after, acquire_failure) == 0 &&
      resource_delta(teardown_before, teardown_after, teardown_attempt) == 2 &&
      resource_delta(teardown_before, teardown_after, teardown_success) == 1 &&
      resource_delta(teardown_before, teardown_after, teardown_failure) == 1 &&
      resource_delta(teardown_before, teardown_after, active) == 0 &&
      resource_delta(teardown_before, teardown_after,
                     "cleanup_error_count") == 1,
    paste(resource, "failed noexcept teardown retains and retries ownership")
  )
}

growth_runtime <- fixed_sp_cuda_runtime_create(device_id = 0L)
on.exit(try(fixed_sp_cuda_runtime_free(growth_runtime), silent = TRUE),
        add = TRUE)
fixed_sp_cuda_runtime_reserve(
  growth_runtime, n = 16L, null_dim = 8L, target_count = 1L,
  penalty_count = 1L, augmented_rows = 16L
)
growth_before <- fixed_sp_cuda_runtime_info(growth_runtime)
growth_ledger_before <- resource_snapshot()
.Call(
  "C_fixed_sp_cuda_test_inject_next_device_free_failure",
  PACKAGE = "fastkpc_cuda"
)
assert_error(
  fixed_sp_cuda_runtime_reserve(
    growth_runtime, n = 32L, null_dim = 8L, target_count = 1L,
    penalty_count = 1L, augmented_rows = 32L
  ),
  "injected tracked CUDA device free failure",
  "growth fails closed when replacing an arena cannot free its old owner"
)
growth_after <- fixed_sp_cuda_runtime_info(growth_runtime)
growth_ledger_after <- resource_snapshot()
assert_true(
  growth_after$workspace_reserve_count ==
      growth_before$workspace_reserve_count &&
    growth_after$workspace_grow_count == growth_before$workspace_grow_count &&
    growth_after$workspace_bytes == growth_before$workspace_bytes &&
    resource_delta(growth_ledger_before, growth_ledger_after,
                   "cuda_device_allocate_success_count") == 1 &&
    resource_delta(growth_ledger_before, growth_ledger_after,
                   "cuda_device_free_attempt_count") == 2 &&
    resource_delta(growth_ledger_before, growth_ledger_after,
                   "cuda_device_free_success_count") == 1 &&
    resource_delta(growth_ledger_before, growth_ledger_after,
                   "cuda_device_free_failure_count") == 1 &&
    resource_delta(growth_ledger_before, growth_ledger_after,
                   "cuda_device_active_count") == 0 &&
    resource_delta(growth_ledger_before, growth_ledger_after,
                   "cleanup_error_count") == 1,
  "failed growth retains the old arena and cleans the uncommitted allocation"
)
fixed_sp_cuda_runtime_reserve(
  growth_runtime, n = 16L, null_dim = 8L, target_count = 1L,
  penalty_count = 1L, augmented_rows = 16L
)
growth_active_before_free <- resource_snapshot()
fixed_sp_cuda_runtime_free(growth_runtime)
growth_active_after_free <- resource_snapshot()
for (resource in names(resources)) {
  active <- paste(resource, "active_count", sep = "_")
  assert_true(
    growth_active_after_free[[active]] <= growth_active_before_free[[active]],
    paste(resource, "cleanup does not increase active ownership")
  )
}
assert_true(
  all(vapply(names(resources), function(resource) {
    active <- paste(resource, "active_count", sep = "_")
    identical(growth_active_after_free[[active]], ledger_after[[active]])
  }, logical(1L))),
  "explicit cleanup releases every resource retained after failed growth"
)

cat("PASS Phase 3 CUDA runtime lifecycle\n")
