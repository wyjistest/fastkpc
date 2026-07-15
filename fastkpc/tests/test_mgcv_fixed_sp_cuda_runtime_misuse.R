source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
expect_error_contains <- function(expr, text, message = text) {
  error <- tryCatch(force(expr), error = identity)
  assert_true(
    inherits(error, "error") &&
      grepl(text, conditionMessage(error), fixed = TRUE),
    paste0(message, if (inherits(error, "error")) {
      paste0(": ", conditionMessage(error))
    } else {
      ": no error"
    })
  )
  invisible(error)
}

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP Phase 3A fixed-sp runtime misuse\n")
  quit(save = "no", status = 0)
}

build_fastkpc_cuda_native(rebuild = TRUE)
assert_true(fastkpc_cuda_available(), "CUDA must be available")

test_native_names <- c(
  "fixed_sp_cuda_test_device_count",
  "fixed_sp_cuda_test_set_device",
  "fixed_sp_cuda_test_register_blocked_consumer",
  "fixed_sp_cuda_test_complete_consumer"
)
assert_true(
  !any(vapply(test_native_names, exists, logical(1L), mode = "function",
              inherits = TRUE)),
  "fixed-sp misuse hooks are not exposed as production R wrappers"
)

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

subset_target <- function(batch, index) {
  list(
    setup = batch$setup,
    target_rows = batch$target_rows[index, , drop = FALSE],
    Y = batch$Y[, index, drop = FALSE],
    SP = batch$SP[, index, drop = FALSE],
    oracle_nullspace_rhs =
      batch$oracle_nullspace_rhs[, index, drop = FALSE],
    planned_route = batch$planned_route[index],
    condition = batch$condition[index],
    prepared_s_key_sha256 = batch$prepared_s_key_sha256
  )
}

safe_keys <- sort(names(batches)[vapply(batches, function(batch) {
  any(batch$planned_route == "CHOLESKY_BATCHED")
}, logical(1L))], method = "radix")
assert_true(length(safe_keys) >= 1L,
            "iteration scope contains a safe fixed-sp target")
safe_source <- batches[[safe_keys[[1L]]]]
safe_index <- which(safe_source$planned_route == "CHOLESKY_BATCHED")[[1L]]
safe_target <- subset_target(safe_source, safe_index)
safe_dto <- fastkpc_full_cuda_fixed_sp_native_dto(safe_target$setup)
native_batch <- fastkpc_full_cuda_fixed_sp_native_batch(
  safe_target, safe_dto
)
assert_true(
  nrow(safe_target$target_rows) == 1L &&
    identical(native_batch$target_count, 1L) &&
    identical(native_batch$planned_route, "CHOLESKY_BATCHED"),
  "misuse test uses one authenticated iteration-scope safe target"
)

runtime <- fixed_sp_cuda_runtime_create(0L)
on.exit(try(fixed_sp_cuda_runtime_free(runtime), silent = TRUE), add = TRUE)
fixed_sp_cuda_runtime_reserve(runtime, 351L, 64L, 47L, 7L, 407L)
handle <- fixed_sp_cuda_prepared_create(runtime, safe_dto)
on.exit(try(fixed_sp_cuda_prepared_free(handle), silent = TRUE), add = TRUE)

freed_runtime <- fixed_sp_cuda_runtime_create(0L)
fixed_sp_cuda_runtime_free(freed_runtime)
expect_error_contains(
  fixed_sp_cuda_runtime_info(freed_runtime), "freed",
  "freed runtime rejects info"
)

freed_handle <- fixed_sp_cuda_prepared_create(runtime, safe_dto)
fixed_sp_cuda_prepared_free(freed_handle)
expect_error_contains(
  fixed_sp_cuda_prepared_info(freed_handle), "freed",
  "freed prepared handle rejects info"
)

first_token <- fixed_sp_cuda_solve_batch(
  handle, native_batch$Y, native_batch$SP,
  native_batch$planned_route, native_batch$target_keys
)
on.exit(try(fixed_sp_cuda_residual_free(first_token), silent = TRUE),
        add = TRUE)
expect_error_contains(
  fixed_sp_cuda_solve_batch(
    handle, native_batch$Y, native_batch$SP,
    native_batch$planned_route, native_batch$target_keys
  ),
  "ERR_OUTPUT_SLOT_BUSY", "an active output lease blocks the next solve"
)
fixed_sp_cuda_residual_release(first_token)
second_token <- fixed_sp_cuda_solve_batch(
  handle, native_batch$Y, native_batch$SP,
  native_batch$planned_route, native_batch$target_keys
)
on.exit(try(fixed_sp_cuda_residual_free(second_token), silent = TRUE),
        add = TRUE)
expect_error_contains(
  fixed_sp_cuda_materialize_shadow(first_token), "STALE_TOKEN",
  "a released token cannot access a reused slot generation"
)
fixed_sp_cuda_residual_free(first_token)
fixed_sp_cuda_residual_release(second_token)
fixed_sp_cuda_residual_free(second_token)

freed_token <- fixed_sp_cuda_solve_batch(
  handle, native_batch$Y, native_batch$SP,
  native_batch$planned_route, native_batch$target_keys
)
fixed_sp_cuda_residual_release(freed_token)
fixed_sp_cuda_residual_free(freed_token)
expect_error_contains(
  fixed_sp_cuda_residual_info(freed_token), "freed",
  "freed residual token rejects info"
)

consumer_token <- fixed_sp_cuda_solve_batch(
  handle, native_batch$Y, native_batch$SP,
  native_batch$planned_route, native_batch$target_keys
)
on.exit(try(fixed_sp_cuda_residual_free(consumer_token), silent = TRUE),
        add = TRUE)
.Call(
  "C_fixed_sp_cuda_test_register_blocked_consumer", consumer_token,
  PACKAGE = "fastkpc_cuda"
)
expect_error_contains(
  fixed_sp_cuda_residual_release(consumer_token), "ERR_OUTPUT_SLOT_BUSY",
  "an incomplete consumer prevents lease release"
)
expect_error_contains(
  fixed_sp_cuda_solve_batch(
    handle, native_batch$Y, native_batch$SP,
    native_batch$planned_route, native_batch$target_keys
  ),
  "ERR_OUTPUT_SLOT_BUSY",
  "an incomplete consumer preserves the busy lease for the next solve"
)
.Call(
  "C_fixed_sp_cuda_test_complete_consumer", consumer_token,
  PACKAGE = "fastkpc_cuda"
)
fixed_sp_cuda_residual_release(consumer_token)
fixed_sp_cuda_residual_free(consumer_token)

post_consumer_token <- fixed_sp_cuda_solve_batch(
  handle, native_batch$Y, native_batch$SP,
  native_batch$planned_route, native_batch$target_keys
)
fixed_sp_cuda_residual_release(post_consumer_token)
fixed_sp_cuda_residual_free(post_consumer_token)

wrong_tag_handle <- legacy_dcov_spectra_matvec_cuda_handle(diag(2))
on.exit(
  try(legacy_dcov_spectra_matvec_cuda_handle_free(wrong_tag_handle),
      silent = TRUE),
  add = TRUE
)
expect_error_contains(
  fixed_sp_cuda_runtime_info(wrong_tag_handle$ptr),
  "wrong fixed-sp external pointer tag",
  "runtime entry rejects a wrong external pointer tag"
)
expect_error_contains(
  fixed_sp_cuda_prepared_info(wrong_tag_handle$ptr),
  "wrong fixed-sp external pointer tag",
  "prepared entry rejects a wrong external pointer tag"
)
expect_error_contains(
  fixed_sp_cuda_residual_info(wrong_tag_handle$ptr),
  "wrong fixed-sp external pointer tag",
  "residual entry rejects a wrong external pointer tag"
)

device_count <- .Call(
  "C_fixed_sp_cuda_test_device_count", PACKAGE = "fastkpc_cuda"
)
assert_true(
  is.integer(device_count) && length(device_count) == 1L &&
    !is.na(device_count) && device_count >= 1L,
  "device-count misuse hook returns the available CUDA device count"
)
if (device_count >= 2L) {
  .Call("C_fixed_sp_cuda_test_set_device", 1L, PACKAGE = "fastkpc_cuda")
  on.exit(
    try(.Call("C_fixed_sp_cuda_test_set_device", 0L,
              PACKAGE = "fastkpc_cuda"), silent = TRUE),
    add = TRUE
  )
  expect_error_contains(
    fixed_sp_cuda_runtime_info(runtime), "wrong device",
    "runtime info rejects a changed current device"
  )
  expect_error_contains(
    fixed_sp_cuda_runtime_reserve(runtime, 351L, 64L, 47L, 7L, 407L),
    "wrong device", "runtime reserve rejects a changed current device"
  )
  expect_error_contains(
    fixed_sp_cuda_prepared_info(handle), "wrong device",
    "prepared info rejects a changed current device"
  )
  expect_error_contains(
    fixed_sp_cuda_prepared_create(runtime, safe_dto), "wrong device",
    "prepared create rejects a changed current device"
  )
  expect_error_contains(
    fixed_sp_cuda_solve_batch(
      handle, native_batch$Y, native_batch$SP,
      native_batch$planned_route, native_batch$target_keys
    ),
    "wrong device", "solve rejects a changed current device"
  )
  expect_error_contains(
    fixed_sp_cuda_prepared_free(handle), "wrong device",
    "prepared free rejects a changed current device"
  )
  expect_error_contains(
    fixed_sp_cuda_runtime_free(runtime), "wrong device",
    "runtime free rejects a changed current device"
  )

  .Call("C_fixed_sp_cuda_test_set_device", 0L, PACKAGE = "fastkpc_cuda")
  device_token <- fixed_sp_cuda_solve_batch(
    handle, native_batch$Y, native_batch$SP,
    native_batch$planned_route, native_batch$target_keys
  )
  on.exit(try(fixed_sp_cuda_residual_free(device_token), silent = TRUE),
          add = TRUE)
  .Call("C_fixed_sp_cuda_test_set_device", 1L, PACKAGE = "fastkpc_cuda")
  expect_error_contains(
    fixed_sp_cuda_residual_info(device_token), "wrong device",
    "residual info rejects a changed current device"
  )
  expect_error_contains(
    fixed_sp_cuda_materialize_shadow(device_token), "wrong device",
    "shadow materialization rejects a changed current device"
  )
  expect_error_contains(
    fixed_sp_cuda_residual_release(device_token), "wrong device",
    "residual release rejects a changed current device"
  )
  expect_error_contains(
    fixed_sp_cuda_residual_free(device_token), "wrong device",
    "residual free rejects a changed current device"
  )
  .Call("C_fixed_sp_cuda_test_set_device", 0L, PACKAGE = "fastkpc_cuda")
  fixed_sp_cuda_residual_release(device_token)
  fixed_sp_cuda_residual_free(device_token)
}

if (.Platform$OS.type == "unix") {
  resource_snapshot <- function() {
    .Call("C_fixed_sp_cuda_test_resource_snapshot",
          PACKAGE = "fastkpc_cuda")
  }
  teardown_attempts <- function(snapshot) {
    fields <- grep(
      "_(free|destroy)_attempt_count$", names(snapshot), value = TRUE
    )
    unlist(snapshot[fields], use.names = TRUE)
  }
  capture_error <- function(expr) {
    error <- tryCatch(force(expr), error = identity)
    if (inherits(error, "error")) conditionMessage(error) else "NO_ERROR"
  }

  fork_runtime <- fixed_sp_cuda_runtime_create(0L)
  on.exit(try(fixed_sp_cuda_runtime_free(fork_runtime), silent = TRUE),
          add = TRUE)
  fixed_sp_cuda_runtime_reserve(fork_runtime, 351L, 64L, 47L, 7L, 407L)
  fork_handle <- fixed_sp_cuda_prepared_create(fork_runtime, safe_dto)
  on.exit(try(fixed_sp_cuda_prepared_free(fork_handle), silent = TRUE),
          add = TRUE)
  fork_token <- fixed_sp_cuda_solve_batch(
    fork_handle, native_batch$Y, native_batch$SP,
    native_batch$planned_route, native_batch$target_keys
  )
  on.exit(try(fixed_sp_cuda_residual_free(fork_token), silent = TRUE),
          add = TRUE)

  child <- parallel::mcparallel({
    before <- resource_snapshot()
    messages <- c(
      runtime_info = capture_error(fixed_sp_cuda_runtime_info(fork_runtime)),
      prepared_info = capture_error(fixed_sp_cuda_prepared_info(fork_handle)),
      residual_info = capture_error(fixed_sp_cuda_residual_info(fork_token)),
      residual_free = capture_error(fixed_sp_cuda_residual_free(fork_token)),
      prepared_free = capture_error(fixed_sp_cuda_prepared_free(fork_handle)),
      runtime_free = capture_error(fixed_sp_cuda_runtime_free(fork_runtime))
    )
    rm(list = c("fork_token", "fork_handle", "fork_runtime"),
       envir = .GlobalEnv)
    invisible(gc())
    after <- resource_snapshot()
    list(
      messages = messages,
      teardown_before = teardown_attempts(before),
      teardown_after = teardown_attempts(after),
      cleanup_error_before = before$cleanup_error_count,
      cleanup_error_after = after$cleanup_error_count
    )
  }, mc.set.seed = FALSE)
  child_result <- parallel::mccollect(child)[[1L]]
  assert_true(
    is.list(child_result) &&
      all(grepl("creator PID", child_result$messages, fixed = TRUE)),
    "every forked fixed-sp entry rejects the inherited creator PID"
  )
  assert_true(
    identical(child_result$teardown_after,
              child_result$teardown_before) &&
      identical(child_result$cleanup_error_after,
                child_result$cleanup_error_before),
    "fork-child finalizers clear host wrappers without CUDA teardown"
  )

  assert_true(
    fixed_sp_cuda_runtime_info(fork_runtime)$device_id == 0L &&
      identical(fixed_sp_cuda_prepared_info(fork_handle)$n, safe_dto$n) &&
      fixed_sp_cuda_residual_info(fork_token)$target_count == 1L,
    "parent runtime, prepared handle, and token remain valid after child GC"
  )
  fixed_sp_cuda_residual_release(fork_token)
  fixed_sp_cuda_residual_free(fork_token)
  fixed_sp_cuda_prepared_free(fork_handle)
  fixed_sp_cuda_runtime_free(fork_runtime)
}

fixed_sp_cuda_prepared_free(handle)
fixed_sp_cuda_runtime_free(runtime)

cat("PASS Phase 3A fixed-sp runtime misuse\n")
