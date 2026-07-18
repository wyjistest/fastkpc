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
assert_named_scalar <- function(object, name, type, message) {
  object_names <- names(object)
  assert_true(
    is.list(object) && !is.null(object_names) && !anyNA(object_names) &&
      identical(sum(object_names == name), 1L),
    paste(message, "has exactly one named field")
  )
  value <- object[[name]]
  assert_true(
    !is.null(value) && identical(typeof(value), type) &&
      length(value) == 1L && !is.na(value),
    paste(message, "has the required scalar type and length")
  )
  value
}

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP Phase 3 CUDA runtime lifecycle\n")
  quit(save = "no", status = 0)
}

runtime_source <- paste(
  readLines("fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu", warn = FALSE),
  collapse = "\n"
)
compact_runtime_source <- gsub("[[:space:]]+", " ", runtime_source)
assert_true(
  grepl(
    "constexpr std::size_t kStableBaseIntArraysPerTarget = 6U;",
    compact_runtime_source, fixed = TRUE
  ),
  "stable checkpoint base diagnostics retain exactly six integer arrays"
)
assert_true(
  grepl(
    paste(
      "allocation_bytes( stable_compact_count, sizeof(int),",
      "\"fixed-sp SVD compact integer diagnostics\"),",
      "cudaMemcpyDeviceToHost, context->stream"
    ),
    compact_runtime_source, fixed = TRUE
  ),
  "SVD checkpoint copies the full stable compact integer diagnostics"
)
assert_true(
  grepl(
    paste(
      "allocation_bytes( checked_multiply( reserved_targets,",
      "kStableBaseIntArraysPerTarget,",
      "\"fixed-sp QR compact diagnostics\"), sizeof(int),",
      "\"fixed-sp QR compact diagnostics\"),",
      "cudaMemcpyDeviceToHost, context->stream"
    ),
    compact_runtime_source, fixed = TRUE
  ),
  "QR checkpoint copies exactly the six base integer diagnostic arrays"
)

build_fastkpc_cuda_native(rebuild = TRUE)
assert_true(fastkpc_cuda_available(), "CUDA must be available")

resource_snapshot <- function() {
  .Call("C_fixed_sp_cuda_test_resource_snapshot", PACKAGE = "fastkpc_cuda")
}
resource_delta <- function(before, after, field) {
  as.numeric(after[[field]] - before[[field]])
}
inject_teardown_failure <- function(resource) {
  .Call(
    "C_fixed_sp_cuda_test_inject_next_resource_teardown_failure",
    resource, PACKAGE = "fastkpc_cuda"
  )
}
inject_post_call_teardown_failure <- function(resource) {
  .Call(
    "C_fixed_sp_cuda_test_inject_next_resource_post_call_teardown_failure",
    resource, PACKAGE = "fastkpc_cuda"
  )
}
assert_true(
  grepl("std::shared_ptr<FixedSpResourceLedger> resource_ledger",
        runtime_source, fixed = TRUE) &&
    !grepl("FixedSpResourceCounters* resource_counters",
           runtime_source, fixed = TRUE),
  "teardown-capable objects retain an owning shared resource ledger"
)
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
stable_field_types <- c(
  gesvdj_info_create_count = "integer",
  gesvdj_info_destroy_count = "integer",
  eigen_workspace_bytes = "double",
  qr_workspace_bytes = "double",
  svd_workspace_bytes = "double",
  augmented_workspace_bytes = "double",
  aggregate_factor_workspace_bytes = "double",
  stable_workspace_grow_count = "integer"
)
stable_info <- lapply(names(stable_field_types), function(field) {
  assert_named_scalar(
    after, field, stable_field_types[[field]],
    paste("stable runtime diagnostic", field)
  )
})
names(stable_info) <- names(stable_field_types)
assert_true(
  !all(vapply(stable_info, is.null, logical(1L))),
  "stable runtime diagnostics cannot degrade to all NULL"
)
assert_true(stable_info$gesvdj_info_create_count == 1L,
            "one persistent gesvdj info object")
assert_true(stable_info$gesvdj_info_destroy_count == 0L,
            "reserve does not destroy the persistent gesvdj info object")
assert_true(stable_info$eigen_workspace_bytes > 0,
            "eigensolver workspace is reserved")
assert_true(stable_info$qr_workspace_bytes > 0,
            "QR workspace is reserved")
assert_true(stable_info$svd_workspace_bytes > 0,
            "SVD workspace is reserved")
assert_true(stable_info$augmented_workspace_bytes == 8 * 415 * 64,
            "internal stable matrix reserves max(407, 351 + 64) rows")
assert_true(stable_info$aggregate_factor_workspace_bytes ==
              8 * (64 * 64 + 2 * 64),
            "aggregate factor and DPSTF2 workspaces are reserved")
assert_true(stable_info$stable_workspace_grow_count == 0L,
            "canonical reserve performs no solve-time stable growth")
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
assert_true(
  equal$cuda_device_allocation_count == after$cuda_device_allocation_count &&
    equal$cuda_host_allocation_count == after$cuda_host_allocation_count,
  "equal reserve does not allocate or rerun reserve-time probes"
)
assert_true(
  equal$eigen_workspace_bytes == after$eigen_workspace_bytes &&
    equal$qr_workspace_bytes == after$qr_workspace_bytes &&
    equal$svd_workspace_bytes == after$svd_workspace_bytes &&
    equal$augmented_workspace_bytes == after$augmented_workspace_bytes &&
    equal$aggregate_factor_workspace_bytes ==
      after$aggregate_factor_workspace_bytes &&
    equal$stable_workspace_grow_count == 0L,
  "equal reserve keeps stable workspace diagnostics without growth"
)

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
assert_true(
  cross_growth$augmented_workspace_bytes == 8 * 464 * 64,
  "merged n=400 and q=64 capacities reserve 464 internal stable rows"
)

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
assert_true(
  cross_merged$augmented_workspace_bytes == 8 * 464 * 64 &&
    cross_merged$cuda_device_allocation_count ==
      cross_growth$cuda_device_allocation_count &&
    cross_merged$cuda_host_allocation_count ==
      cross_growth$cuda_host_allocation_count &&
    cross_merged$eigen_workspace_bytes ==
      cross_growth$eigen_workspace_bytes &&
    cross_merged$qr_workspace_bytes == cross_growth$qr_workspace_bytes &&
    cross_merged$svd_workspace_bytes == cross_growth$svd_workspace_bytes &&
    cross_merged$aggregate_factor_workspace_bytes ==
      cross_growth$aggregate_factor_workspace_bytes,
  "exact merged reserve reuses allocations and stable workspace queries"
)

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
  cusolver_handle = c("create", "destroy"),
  gesvdj_info = c("create", "destroy")
)
assert_resource_invariants <- function(snapshot, message,
                                       require_quiescent = FALSE) {
  assert_true(
    all(vapply(names(resources), function(resource) {
      verbs <- resources[[resource]]
      acquired <- snapshot[[paste(
        resource, verbs[[1L]], "success_count", sep = "_"
      )]]
      teardown_success <- snapshot[[paste(
        resource, verbs[[2L]], "success_count", sep = "_"
      )]]
      active <- snapshot[[paste(resource, "active_count", sep = "_")]]
      indeterminate <- snapshot[[paste(
        resource, "ownership_indeterminate_count", sep = "_"
      )]]
      identical(
        as.numeric(acquired),
        as.numeric(teardown_success + active + indeterminate)
      ) && (!require_quiescent ||
            (identical(as.numeric(active), 0) &&
             identical(as.numeric(indeterminate), 0)))
    }, logical(1L))),
    message
  )
}
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
  resource_delta(ledger_before, ledger_after,
                 "gesvdj_info_create_success_count") == 1 &&
    resource_delta(ledger_before, ledger_after,
                   "gesvdj_info_destroy_attempt_count") == 1 &&
    resource_delta(ledger_before, ledger_after,
                   "gesvdj_info_destroy_success_count") == 1 &&
    resource_delta(ledger_before, ledger_after,
                   "gesvdj_info_destroy_failure_count") == 0 &&
    resource_delta(ledger_before, ledger_after,
                   "gesvdj_info_active_count") == 0,
  "persistent gesvdj info object is destroyed exactly once at teardown"
)
assert_true(
  resource_delta(ledger_before, ledger_after, "cleanup_error_count") == 0,
  "scoped explicit cleanup records no teardown errors"
)

for (narrow_q in c(63L, 64L)) {
  narrow_runtime <- fixed_sp_cuda_runtime_create(device_id = 0L)
  on.exit(try(fixed_sp_cuda_runtime_free(narrow_runtime), silent = TRUE),
          add = TRUE)
  fixed_sp_cuda_runtime_reserve(
    narrow_runtime, n = 1L, null_dim = narrow_q, target_count = 1L,
    penalty_count = 1L, augmented_rows = 1L
  )
  narrow <- fixed_sp_cuda_runtime_info(narrow_runtime)
  assert_true(
    narrow$augmented_workspace_bytes == 8 * (1L + narrow_q) * narrow_q,
    paste(
      "narrow reserve includes n + q internal stable rows for q", narrow_q
    )
  )
  assert_true(
    narrow$eigen_workspace_bytes > 0 && narrow$qr_workspace_bytes > 0 &&
      narrow$svd_workspace_bytes > 0 &&
      narrow$aggregate_factor_workspace_bytes ==
        8 * (narrow_q * narrow_q + 2L * narrow_q) &&
      narrow$workspace_reserve_count == 1L &&
      narrow$workspace_grow_count == 1L &&
      narrow$stable_workspace_grow_count == 0L,
    paste("narrow reserve creates persistent stable workspace for q", narrow_q)
  )

  fixed_sp_cuda_runtime_reserve(
    narrow_runtime, n = 1L, null_dim = narrow_q, target_count = 1L,
    penalty_count = 1L, augmented_rows = 1L
  )
  narrow_equal <- fixed_sp_cuda_runtime_info(narrow_runtime)
  assert_true(
    narrow_equal$workspace_reserve_count == 2L &&
      narrow_equal$workspace_grow_count == narrow$workspace_grow_count &&
      narrow_equal$stable_workspace_grow_count == 0L &&
      narrow_equal$cuda_device_allocation_count ==
        narrow$cuda_device_allocation_count &&
      narrow_equal$cuda_host_allocation_count ==
        narrow$cuda_host_allocation_count &&
      narrow_equal$workspace_bytes == narrow$workspace_bytes &&
      narrow_equal$eigen_workspace_bytes == narrow$eigen_workspace_bytes &&
      narrow_equal$qr_workspace_bytes == narrow$qr_workspace_bytes &&
      narrow_equal$svd_workspace_bytes == narrow$svd_workspace_bytes &&
      narrow_equal$aggregate_factor_workspace_bytes ==
        narrow$aggregate_factor_workspace_bytes &&
      narrow_equal$augmented_workspace_bytes ==
        8 * (1L + narrow_q) * narrow_q,
    paste(
      "equal narrow reserve does not allocate, query, or grow for q", narrow_q
    )
  )
  fixed_sp_cuda_runtime_free(narrow_runtime)
}
stable_source <- paste(
  readLines("fastkpc/src/cuda/mgcv_fixed_sp_stable.cu", warn = FALSE),
  collapse = "\n"
)
assert_true(
  grepl(
    "const int probe_rows = std::max(workspace->max_rows, workspace->max_q);",
    stable_source, fixed = TRUE
  ) &&
    grepl(
      "return data_double_count(std::max(max_rows, max_q), max_q);",
      stable_source, fixed = TRUE
    ) &&
    grepl("solver, probe_rows, workspace->max_q,", stable_source,
          fixed = TRUE) &&
    grepl("probe_rows, 1, workspace->max_q,", stable_source,
          fixed = TRUE) &&
    grepl("workspace->c, probe_rows, &workspace->ormqr_lwork", stable_source,
          fixed = TRUE),
  "stable workspace queries use a safely allocated tall probe geometry"
)

runtime_retry_before <- resource_snapshot()
runtime_retry <- fixed_sp_cuda_runtime_create(device_id = 0L)
on.exit(try(fixed_sp_cuda_runtime_free(runtime_retry), silent = TRUE),
        add = TRUE)
runtime_retry_acquired <- resource_snapshot()
inject_teardown_failure("stream")
assert_error(
  fixed_sp_cuda_runtime_free(runtime_retry),
  "retryable teardown",
  "runtime owner retains an injected not-attempted stream teardown"
)
assert_error(
  fixed_sp_cuda_runtime_info(runtime_retry),
  "TeardownOnly",
  "runtime info rejects a runtime awaiting teardown retry"
)
assert_error(
  fixed_sp_cuda_runtime_reserve(
    runtime_retry, n = 16L, null_dim = 8L, target_count = 1L,
    penalty_count = 1L, augmented_rows = 16L
  ),
  "TeardownOnly",
  "runtime reserve rejects a runtime awaiting teardown retry"
)
runtime_retry_partial <- resource_snapshot()
assert_true(
  resource_delta(runtime_retry_acquired, runtime_retry_partial,
                 "stream_destroy_attempt_count") == 1 &&
    resource_delta(runtime_retry_acquired, runtime_retry_partial,
                   "stream_destroy_success_count") == 0 &&
    resource_delta(runtime_retry_acquired, runtime_retry_partial,
                   "stream_destroy_failure_count") == 1 &&
    resource_delta(runtime_retry_acquired, runtime_retry_partial,
                   "stream_active_count") == 0 &&
    resource_delta(runtime_retry_acquired, runtime_retry_partial,
                   "stream_ownership_indeterminate_count") == 0,
  "first runtime close retains only retryable stream ownership"
)
fixed_sp_cuda_runtime_free(runtime_retry)
fixed_sp_cuda_runtime_free(runtime_retry)
runtime_retry_after <- resource_snapshot()
assert_true(
  resource_delta(runtime_retry_acquired, runtime_retry_after,
                 "stream_destroy_attempt_count") == 2 &&
    resource_delta(runtime_retry_acquired, runtime_retry_after,
                   "stream_destroy_success_count") == 1 &&
    resource_delta(runtime_retry_acquired, runtime_retry_after,
                   "stream_destroy_failure_count") == 1 &&
    resource_delta(runtime_retry_acquired, runtime_retry_after,
                   "stream_active_count") == -1 &&
    resource_delta(runtime_retry_acquired, runtime_retry_after,
                   "stream_ownership_indeterminate_count") == 0 &&
    resource_delta(runtime_retry_before, runtime_retry_after,
                   "cleanup_error_count") == 1,
  "second runtime close completes and balances retryable stream ownership"
)
assert_resource_invariants(
  runtime_retry_after,
  "runtime retry preserves every resource ownership invariant",
  require_quiescent = TRUE
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
  indeterminate <- paste(
    resource, "ownership_indeterminate_count", sep = "_"
  )
  assert_true(
    resource_delta(teardown_before, teardown_after, acquire_attempt) == 1 &&
      resource_delta(teardown_before, teardown_after, acquire_success) == 1 &&
      resource_delta(teardown_before, teardown_after, acquire_failure) == 0 &&
      resource_delta(teardown_before, teardown_after, teardown_attempt) == 2 &&
      resource_delta(teardown_before, teardown_after, teardown_success) == 1 &&
      resource_delta(teardown_before, teardown_after, teardown_failure) == 1 &&
      resource_delta(teardown_before, teardown_after, active) == 0 &&
      resource_delta(teardown_before, teardown_after, indeterminate) == 0 &&
      resource_delta(teardown_before, teardown_after,
                     "cleanup_error_count") == 1,
    paste(resource,
          "pre-call teardown failure retains and retries ownership")
  )
}

for (resource in names(resources)) {
  teardown_before <- resource_snapshot()
  inject_post_call_teardown_failure(resource)
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
  indeterminate <- paste(
    resource, "ownership_indeterminate_count", sep = "_"
  )
  assert_true(
    resource_delta(teardown_before, teardown_after, acquire_attempt) == 1 &&
      resource_delta(teardown_before, teardown_after, acquire_success) == 1 &&
      resource_delta(teardown_before, teardown_after, acquire_failure) == 0 &&
      resource_delta(teardown_before, teardown_after, teardown_attempt) == 1 &&
      resource_delta(teardown_before, teardown_after, teardown_success) == 0 &&
      resource_delta(teardown_before, teardown_after, teardown_failure) == 1 &&
      resource_delta(teardown_before, teardown_after, active) == 0 &&
      resource_delta(teardown_before, teardown_after, indeterminate) == 1 &&
      resource_delta(teardown_before, teardown_after,
                     "cleanup_error_count") == 1,
    paste(resource,
          "post-call teardown failure consumes ownership without retry")
  )
  assert_resource_invariants(
    teardown_after,
    paste(resource, "post-call failure preserves ownership invariant")
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
                   "cuda_device_allocate_success_count") == 2 &&
    resource_delta(growth_ledger_before, growth_ledger_after,
                   "cuda_device_free_attempt_count") == 3 &&
    resource_delta(growth_ledger_before, growth_ledger_after,
                   "cuda_device_free_success_count") == 2 &&
    resource_delta(growth_ledger_before, growth_ledger_after,
                   "cuda_device_free_failure_count") == 1 &&
    resource_delta(growth_ledger_before, growth_ledger_after,
                   "cuda_device_active_count") == 0 &&
    resource_delta(growth_ledger_before, growth_ledger_after,
                   "cuda_device_ownership_indeterminate_count") == 0 &&
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

terminal_runtime_before <- resource_snapshot()
terminal_runtime <- fixed_sp_cuda_runtime_create(device_id = 0L)
terminal_runtime_acquired <- resource_snapshot()
inject_post_call_teardown_failure("stream")
assert_error(
  fixed_sp_cuda_runtime_free(terminal_runtime),
  "ownership is indeterminate",
  "runtime post-call stream error is surfaced after consuming ownership"
)
assert_error(
  fixed_sp_cuda_runtime_info(terminal_runtime),
  "freed",
  "runtime post-call stream error clears the external holder"
)
fixed_sp_cuda_runtime_free(terminal_runtime)
terminal_runtime_after <- resource_snapshot()
assert_true(
  resource_delta(terminal_runtime_acquired, terminal_runtime_after,
                 "stream_destroy_attempt_count") == 1 &&
    resource_delta(terminal_runtime_acquired, terminal_runtime_after,
                   "stream_destroy_success_count") == 0 &&
    resource_delta(terminal_runtime_acquired, terminal_runtime_after,
                   "stream_destroy_failure_count") == 1 &&
    resource_delta(terminal_runtime_acquired, terminal_runtime_after,
                   "stream_active_count") == -1 &&
    resource_delta(terminal_runtime_acquired, terminal_runtime_after,
                   "stream_ownership_indeterminate_count") == 1 &&
    resource_delta(terminal_runtime_before, terminal_runtime_after,
                   "cleanup_error_count") == 1,
  "runtime post-call stream failure is terminal and never retried"
)
assert_resource_invariants(
  terminal_runtime_after,
  "runtime terminal close preserves every ownership invariant"
)

replacement_runtime <- fixed_sp_cuda_runtime_create(device_id = 0L)
on.exit(try(fixed_sp_cuda_runtime_free(replacement_runtime), silent = TRUE),
        add = TRUE)
fixed_sp_cuda_runtime_reserve(
  replacement_runtime, n = 16L, null_dim = 8L, target_count = 1L,
  penalty_count = 1L, augmented_rows = 16L
)
replacement_before <- resource_snapshot()
inject_post_call_teardown_failure("cuda_device")
assert_error(
  fixed_sp_cuda_runtime_reserve(
    replacement_runtime, n = 32L, null_dim = 8L, target_count = 1L,
    penalty_count = 1L, augmented_rows = 32L
  ),
  "free old fixed-sp double arena",
  "post-call replacement failure is surfaced"
)
assert_error(
  fixed_sp_cuda_runtime_info(replacement_runtime),
  "TeardownOnly",
  "post-call replacement failure poisons runtime info"
)
assert_error(
  fixed_sp_cuda_runtime_reserve(
    replacement_runtime, n = 16L, null_dim = 8L, target_count = 1L,
    penalty_count = 1L, augmented_rows = 16L
  ),
  "TeardownOnly",
  "post-call replacement failure poisons later reserve"
)
replacement_partial <- resource_snapshot()
assert_true(
  resource_delta(replacement_before, replacement_partial,
                 "cuda_device_allocate_success_count") == 2 &&
    resource_delta(replacement_before, replacement_partial,
                   "cuda_device_free_attempt_count") == 3 &&
    resource_delta(replacement_before, replacement_partial,
                   "cuda_device_free_success_count") == 2 &&
    resource_delta(replacement_before, replacement_partial,
                   "cuda_device_free_failure_count") == 1 &&
    resource_delta(replacement_before, replacement_partial,
                   "cuda_device_active_count") == -1 &&
    resource_delta(replacement_before, replacement_partial,
                   "cuda_device_ownership_indeterminate_count") == 1,
  "reserve clears the consumed old arena and rolls back its new allocation"
)
fixed_sp_cuda_runtime_free(replacement_runtime)
replacement_after <- resource_snapshot()
assert_true(
  resource_delta(replacement_before, replacement_after,
                 "cuda_device_free_attempt_count") == 6 &&
    resource_delta(replacement_before, replacement_after,
                   "cuda_device_free_success_count") == 5 &&
    resource_delta(replacement_before, replacement_after,
                   "cuda_device_free_failure_count") == 1 &&
    resource_delta(replacement_before, replacement_after,
                   "cuda_device_active_count") == -4 &&
    resource_delta(replacement_before, replacement_after,
                   "cuda_device_ownership_indeterminate_count") == 1,
  "runtime close never retries the arena consumed by replacement failure"
)
assert_resource_invariants(
  replacement_after,
  "reserve terminal failure preserves every ownership invariant"
)

finalizer_retry_before <- resource_snapshot()
finalizer_retry_runtime <- fixed_sp_cuda_runtime_create(device_id = 0L)
finalizer_retry_acquired <- resource_snapshot()
inject_teardown_failure("stream")
rm(finalizer_retry_runtime)
invisible(gc())
finalizer_retry_after <- resource_snapshot()
assert_true(
  resource_delta(finalizer_retry_acquired, finalizer_retry_after,
                 "stream_destroy_attempt_count") == 2 &&
    resource_delta(finalizer_retry_acquired, finalizer_retry_after,
                   "stream_destroy_success_count") == 1 &&
    resource_delta(finalizer_retry_acquired, finalizer_retry_after,
                   "stream_destroy_failure_count") == 1 &&
    resource_delta(finalizer_retry_acquired, finalizer_retry_after,
                   "stream_active_count") == -1 &&
    resource_delta(finalizer_retry_acquired, finalizer_retry_after,
                   "stream_ownership_indeterminate_count") == 0 &&
    resource_delta(finalizer_retry_before, finalizer_retry_after,
                   "cleanup_error_count") == 1,
  "runtime finalizer retries a one-shot not-attempted stream teardown"
)

finalizer_terminal_before <- resource_snapshot()
finalizer_terminal_runtime <- fixed_sp_cuda_runtime_create(device_id = 0L)
finalizer_terminal_acquired <- resource_snapshot()
inject_post_call_teardown_failure("stream")
rm(finalizer_terminal_runtime)
invisible(gc())
finalizer_terminal_after <- resource_snapshot()
assert_true(
  resource_delta(finalizer_terminal_acquired, finalizer_terminal_after,
                 "stream_destroy_attempt_count") == 1 &&
    resource_delta(finalizer_terminal_acquired, finalizer_terminal_after,
                   "stream_destroy_success_count") == 0 &&
    resource_delta(finalizer_terminal_acquired, finalizer_terminal_after,
                   "stream_destroy_failure_count") == 1 &&
    resource_delta(finalizer_terminal_acquired, finalizer_terminal_after,
                   "stream_active_count") == -1 &&
    resource_delta(finalizer_terminal_acquired, finalizer_terminal_after,
                   "stream_ownership_indeterminate_count") == 1 &&
    resource_delta(finalizer_terminal_before, finalizer_terminal_after,
                   "cleanup_error_count") == 1,
  "runtime finalizer never retries a post-call stream teardown failure"
)
assert_resource_invariants(
  finalizer_terminal_after,
  "finalizer teardown modes preserve every ownership invariant"
)

cat("PASS Phase 3 CUDA runtime lifecycle\n")
