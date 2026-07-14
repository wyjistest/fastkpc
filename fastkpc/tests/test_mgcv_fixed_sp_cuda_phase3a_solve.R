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

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP Phase 3A fixed-sp solve\n")
  quit(save = "no", status = 0)
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
routes_by_setup <- split(
  as.character(iteration$target_rows$planned_route),
  as.character(iteration$target_rows$prepared_s_key_sha256)
)
mixed_keys <- names(routes_by_setup)[vapply(routes_by_setup, function(routes) {
  any(routes == "CHOLESKY_BATCHED") &&
    any(routes != "CHOLESKY_BATCHED")
}, logical(1L))]
assert_true(length(mixed_keys) >= 1L,
            "iteration scope contains an authenticated mixed-route setup")
mixed_key <- sort(mixed_keys, method = "radix")[[1L]]
mixed_scope <- iteration
mixed_scope$setup_rows <- iteration$setup_rows[
  iteration$setup_rows$prepared_s_key_sha256 == mixed_key,
  , drop = FALSE
]
mixed_scope$target_rows <- iteration$target_rows[
  iteration$target_rows$prepared_s_key_sha256 == mixed_key,
  , drop = FALSE
]
mixed_rank <- match(mixed_key, catalog$setup_index$prepared_s_key_sha256)
mixed_scope$shard_ids <- as.integer(
  (mixed_rank - 1L) %% catalog$catalog_contract$shard_count
)
batches <- fastkpc_full_cuda_fixed_sp_batches(catalog, mixed_scope)

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

mixed_batch <- batches[[mixed_key]]
safe_batch <- subset_target(
  mixed_batch, which(mixed_batch$planned_route == "CHOLESKY_BATCHED")[[1L]]
)
stable_batch <- subset_target(
  mixed_batch, which(mixed_batch$planned_route != "CHOLESKY_BATCHED")[[1L]]
)
assert_true(
  nrow(safe_batch$target_rows) == 1L &&
    ncol(safe_batch$Y) == 1L && ncol(safe_batch$SP) == 1L &&
    ncol(safe_batch$oracle_nullspace_rhs) == 1L &&
    identical(safe_batch$planned_route, "CHOLESKY_BATCHED"),
  "focused batch preserves exactly one authenticated safe target"
)
assert_true(
  nrow(stable_batch$target_rows) == 1L &&
    ncol(stable_batch$Y) == 1L && ncol(stable_batch$SP) == 1L &&
    ncol(stable_batch$oracle_nullspace_rhs) == 1L &&
    stable_batch$planned_route != "CHOLESKY_BATCHED",
  "focused batch preserves exactly one stable target"
)
stable_dto <- fastkpc_full_cuda_fixed_sp_native_dto(stable_batch$setup)
safe_native_batch <- fastkpc_full_cuda_fixed_sp_native_batch(
  safe_batch, stable_dto
)
stable_native_batch <- fastkpc_full_cuda_fixed_sp_native_batch(
  stable_batch, stable_dto
)
assert_true(
  identical(names(safe_native_batch), names(stable_native_batch)) &&
    identical(names(stable_native_batch),
              c("Y", "SP", "planned_route", "target_keys", "target_count")) &&
    safe_native_batch$target_count == 1L &&
    stable_native_batch$target_count == 1L,
  "production native batch excludes the CPU oracle RHS"
)
safe_oracle <- fastkpc_mgcv_magic_fixed_sp_from_prepared(
  prepared_setup = safe_batch$setup,
  target_state = list(
    row = safe_batch$target_rows,
    y = as.numeric(safe_batch$Y[, 1L])
  )
)
safe_oracle$rss <- sum(safe_oracle$residuals^2)

max_abs_error <- function(actual, expected) {
  max(abs(as.numeric(actual) - as.numeric(expected)))
}
relative_l2_error <- function(actual, expected) {
  difference <- as.numeric(actual) - as.numeric(expected)
  sqrt(sum(difference^2)) / max(sqrt(sum(as.numeric(expected)^2)),
                                .Machine$double.xmin)
}

explicit_dto <- stable_dto
explicit_q <- stable_dto$coefficient_dim - 1L
householder_v <- seq_len(stable_dto$coefficient_dim)
householder_v <- householder_v / sqrt(sum(householder_v^2))
householder <- diag(stable_dto$coefficient_dim) -
  2 * tcrossprod(householder_v)
explicit_Z <- householder[, seq_len(explicit_q), drop = FALSE]
explicit_X_null <- stable_dto$X %*% explicit_Z
assert_true(
  explicit_q > 0L && all(explicit_Z != 0) &&
    max(abs(crossprod(explicit_Z) - diag(explicit_q))) <= 1e-12,
  "explicit-Z coefficient fixture is dense and orthonormal"
)
explicit_dto$constraint_mode <- "explicit"
explicit_dto$constraint_nullspace <- explicit_Z
explicit_dto$null_dim <- explicit_q
explicit_dto$nullspace_gram_matrix <- crossprod(explicit_X_null)

runtime <- fixed_sp_cuda_runtime_create(0L)
on.exit(try(fixed_sp_cuda_runtime_free(runtime), silent = TRUE), add = TRUE)
fixed_sp_cuda_runtime_reserve(runtime, 351L, 64L, 47L, 7L, 407L)
stable_handle <- fixed_sp_cuda_prepared_create(runtime, stable_dto)
on.exit(try(fixed_sp_cuda_prepared_free(stable_handle), silent = TRUE),
        add = TRUE)

explicit_runtime <- fixed_sp_cuda_runtime_create(0L)
on.exit(try(fixed_sp_cuda_runtime_free(explicit_runtime), silent = TRUE),
        add = TRUE)
fixed_sp_cuda_runtime_reserve(explicit_runtime, 351L, 64L, 1L, 7L, 407L)
explicit_handle <- fixed_sp_cuda_prepared_create(explicit_runtime, explicit_dto)
on.exit(try(fixed_sp_cuda_prepared_free(explicit_handle), silent = TRUE),
        add = TRUE)

explicit_token <- fixed_sp_cuda_solve_batch(
  explicit_handle, stable_native_batch$Y, stable_native_batch$SP,
  stable_native_batch$planned_route, stable_native_batch$target_keys,
  outputs = c("coefficients")
)
on.exit(try(fixed_sp_cuda_residual_free(explicit_token), silent = TRUE),
        add = TRUE)
explicit_info <- fixed_sp_cuda_residual_info(explicit_token)
explicit_prepared_info <- fixed_sp_cuda_prepared_info(explicit_handle)
assert_true(
  stable_dto$coefficient_dim > explicit_q &&
    identical(explicit_info$coefficient_dim, stable_dto$coefficient_dim) &&
    identical(explicit_prepared_info$coefficient_output_capacity,
              as.double(stable_dto$coefficient_dim)),
  "coefficient diagnostics retain the full-space p allocation bound"
)
explicit_coefficients <- .Call(
  "C_fixed_sp_cuda_test_coefficient_shadow", explicit_token,
  PACKAGE = "fastkpc_cuda"
)
assert_true(
  identical(dim(explicit_coefficients),
            c(stable_dto$coefficient_dim, 1L)) &&
    length(explicit_coefficients) == stable_dto$coefficient_dim &&
    identical(as.vector(explicit_coefficients),
              rep(NaN, stable_dto$coefficient_dim)),
  "coefficient shadow reads p by one initialized NaN values"
)
assert_true(
  identical(explicit_info$solver_status,
            "ERR_STABLE_PATH_NOT_IMPLEMENTED") &&
    explicit_info$invalid_output_init_count == 1L,
  "explicit-Z coefficient request fails closed after one initialization"
)
fixed_sp_cuda_residual_release(explicit_token)
fixed_sp_cuda_residual_free(explicit_token)
fixed_sp_cuda_prepared_free(explicit_handle)
fixed_sp_cuda_runtime_free(explicit_runtime)

forged_Y <- stable_native_batch$Y
forged_dimensions <- attr(forged_Y, "dim")
attr(forged_dimensions, "forged") <- TRUE
attr(forged_Y, "dim") <- forged_dimensions
assert_error(
  .Call(
    "C_fixed_sp_cuda_solve_batch", stable_handle, forged_Y,
    stable_native_batch$SP, as.character(stable_native_batch$planned_route),
    as.character(stable_native_batch$target_keys), "residuals",
    PACKAGE = "fastkpc_cuda"
  ),
  "Y must be a bare double matrix",
  "direct native solve rejects an attributed Y dim vector"
)

task7_outputs <- c("coefficients", "fitted", "residuals", "rss", "rhs")
task7_result_names <- c(
  "coefficients", "fitted", "residuals", "rss", "cuda_nullspace_rhs"
)
materializer_available <- exists(
  "fixed_sp_cuda_materialize_shadow", mode = "function", inherits = TRUE
)
safe_token <- tryCatch(
  fixed_sp_cuda_solve_batch(
    stable_handle, safe_native_batch$Y, safe_native_batch$SP,
    safe_native_batch$planned_route, safe_native_batch$target_keys,
    outputs = task7_outputs
  ),
  error = identity
)
assert_true(
  materializer_available && !inherits(safe_token, "error"),
  paste0(
    "Task 7 safe solve and materializer must be available; materializer=",
    materializer_available, "; safe_solve=",
    if (inherits(safe_token, "error")) conditionMessage(safe_token) else "OK"
  )
)
on.exit(try(fixed_sp_cuda_residual_free(safe_token), silent = TRUE),
        add = TRUE)
safe_info <- fixed_sp_cuda_residual_info(safe_token)
safe_runtime_info <- fixed_sp_cuda_runtime_info(runtime)
safe_prepared_info <- fixed_sp_cuda_prepared_info(stable_handle)
assert_true(
  identical(safe_info$planned_route, "CHOLESKY_BATCHED") &&
    identical(safe_info$executed_route, "CHOLESKY_BATCHED") &&
    identical(safe_info$reroute_reason, "") &&
    identical(safe_info$solver_status, "OK_CHOLESKY_SINGLE") &&
    !isTRUE(safe_info$true_batched_kernel) &&
    safe_info$true_batched_target_count == 0L &&
    safe_info$planned_cholesky_target_count == 1L &&
    safe_info$executed_cholesky_target_count == 1L,
  "safe target reports a single-target Cholesky execution without batching"
)
assert_true(
  safe_info$rhs_device_build_count == 1L &&
    identical(safe_info$rhs_authority, "cuda-x0-transpose-y") &&
    isTRUE(safe_info$full_cuda_data_plane) &&
    safe_info$implicit_residual_d2h_count == 0L &&
    safe_info$shadow_materialize_call_count == 0L &&
    safe_info$shadow_materialize_target_count == 0L &&
    safe_info$shadow_d2h_bytes == 0,
  "safe solve builds its RHS on CUDA without an implicit D2H"
)
assert_true(
  safe_runtime_info$cholesky_factor_checkpoint_record_count == 1L &&
    safe_runtime_info$cholesky_factor_checkpoint_wait_count == 1L &&
    safe_runtime_info$cholesky_solve_checkpoint_record_count == 1L &&
    safe_runtime_info$cholesky_solve_checkpoint_wait_count == 1L &&
    safe_runtime_info$cuda_device_synchronize_count == 0L,
  "first safe solve uses separate factor and solve checkpoints exactly once"
)

safe_shadow <- fixed_sp_cuda_materialize_shadow(
  safe_token, outputs = task7_outputs
)
assert_true(
  identical(names(safe_shadow), task7_result_names) &&
    identical(dim(safe_shadow$coefficients),
              c(stable_dto$coefficient_dim, 1L)) &&
    identical(dim(safe_shadow$fitted), c(stable_dto$n, 1L)) &&
    identical(dim(safe_shadow$residuals), c(stable_dto$n, 1L)) &&
    identical(length(safe_shadow$rss), 1L) &&
    identical(dim(safe_shadow$cuda_nullspace_rhs),
              c(stable_dto$null_dim, 1L)) &&
    all(vapply(safe_shadow, function(value) all(is.finite(value)), logical(1L))),
  "safe shadow materializes finite requested outputs with canonical shapes"
)
coefficient_max_abs <- max_abs_error(
  safe_shadow$coefficients, safe_oracle$coefficients
)
coefficient_relative_l2 <- relative_l2_error(
  safe_shadow$coefficients, safe_oracle$coefficients
)
fitted_max_abs <- max_abs_error(safe_shadow$fitted, safe_oracle$fitted)
fitted_relative_l2 <- relative_l2_error(
  safe_shadow$fitted, safe_oracle$fitted
)
residual_max_abs <- max_abs_error(
  safe_shadow$residuals, safe_oracle$residuals
)
residual_relative_l2 <- relative_l2_error(
  safe_shadow$residuals, safe_oracle$residuals
)
rss_max_abs <- max_abs_error(safe_shadow$rss, safe_oracle$rss)
rss_relative_l2 <- relative_l2_error(safe_shadow$rss, safe_oracle$rss)
rhs_max_abs <- max_abs_error(
  safe_shadow$cuda_nullspace_rhs, safe_batch$oracle_nullspace_rhs
)
assert_true(
  coefficient_max_abs < 1e-7 && coefficient_relative_l2 < 1e-7 &&
    fitted_max_abs < 1e-7 && fitted_relative_l2 < 1e-7 &&
    residual_max_abs < 1e-7 && residual_relative_l2 < 1e-7 &&
    rss_max_abs < 1e-7 && rss_relative_l2 < 1e-7,
  "safe shadow matches the Phase 2 prepared fixed-sp oracle"
)
assert_true(rhs_max_abs < 1e-12,
            "CUDA X_null transpose Y matches the authenticated RHS oracle")

safe_info_after_shadow <- fixed_sp_cuda_residual_info(safe_token)
expected_safe_d2h_bytes <- 8 * sum(vapply(
  safe_shadow[task7_result_names], length, integer(1L)
))
assert_true(
  safe_info_after_shadow$shadow_materialize_call_count == 1L &&
    safe_info_after_shadow$shadow_materialize_target_count == 1L &&
    identical(safe_info_after_shadow$shadow_d2h_bytes,
              as.double(expected_safe_d2h_bytes)) &&
    safe_info_after_shadow$implicit_residual_d2h_count == 0L &&
    identical(safe_info_after_shadow$solver_status,
              safe_info$solver_status) &&
    identical(safe_info_after_shadow$executed_route,
              safe_info$executed_route) &&
    isTRUE(fixed_sp_cuda_prepared_info(stable_handle)$output_slot_leased),
  "explicit safe shadow counts only requested D2H and preserves ownership"
)

safe_result_hash <- fastkpc_full_cuda_census_metadata_hash(list(
  outputs = safe_shadow,
  planned_route = safe_info$planned_route,
  executed_route = safe_info$executed_route,
  reroute_reason = safe_info$reroute_reason,
  solver_status = safe_info$solver_status
))
fixed_sp_cuda_residual_release(safe_token)
fixed_sp_cuda_residual_free(safe_token)
assert_true(!isTRUE(fixed_sp_cuda_prepared_info(
  stable_handle
)$output_slot_leased), "safe token is released before stable reuse")

stable_after_safe_token <- fixed_sp_cuda_solve_batch(
  stable_handle, stable_native_batch$Y, stable_native_batch$SP,
  stable_native_batch$planned_route, stable_native_batch$target_keys,
  outputs = task7_outputs
)
on.exit(try(fixed_sp_cuda_residual_free(stable_after_safe_token), silent = TRUE),
        add = TRUE)
stable_after_safe_shadow <- fixed_sp_cuda_materialize_shadow(
  stable_after_safe_token, outputs = task7_outputs
)
stable_after_safe_info <- fixed_sp_cuda_residual_info(stable_after_safe_token)
assert_true(
  identical(names(stable_after_safe_shadow), task7_result_names) &&
    identical(dim(stable_after_safe_shadow$coefficients),
              c(stable_dto$coefficient_dim, 1L)) &&
    identical(dim(stable_after_safe_shadow$fitted), c(stable_dto$n, 1L)) &&
    identical(dim(stable_after_safe_shadow$residuals), c(stable_dto$n, 1L)) &&
    identical(length(stable_after_safe_shadow$rss), 1L) &&
    identical(dim(stable_after_safe_shadow$cuda_nullspace_rhs),
              c(stable_dto$null_dim, 1L)) &&
    all(vapply(stable_after_safe_shadow, function(value) {
      all(is.na(value))
    }, logical(1L))),
  "same-handle stable shadow returns explicit NA without prior-output leakage"
)
assert_true(
  identical(stable_after_safe_info$solver_status,
            "ERR_STABLE_PATH_NOT_IMPLEMENTED") &&
    stable_after_safe_info$shadow_materialize_call_count == 1L &&
    stable_after_safe_info$shadow_materialize_target_count == 1L &&
    stable_after_safe_info$shadow_d2h_bytes == 0 &&
    stable_after_safe_info$implicit_residual_d2h_count == 0L,
  "stable shadow records the call without reading a failed device column"
)
fixed_sp_cuda_residual_release(stable_after_safe_token)
fixed_sp_cuda_residual_free(stable_after_safe_token)

repeat_token <- fixed_sp_cuda_solve_batch(
  stable_handle, safe_native_batch$Y, safe_native_batch$SP,
  safe_native_batch$planned_route, safe_native_batch$target_keys,
  outputs = task7_outputs
)
on.exit(try(fixed_sp_cuda_residual_free(repeat_token), silent = TRUE),
        add = TRUE)
repeat_shadow <- fixed_sp_cuda_materialize_shadow(
  repeat_token, outputs = task7_outputs
)
repeat_info <- fixed_sp_cuda_residual_info(repeat_token)
repeat_result_hash <- fastkpc_full_cuda_census_metadata_hash(list(
  outputs = repeat_shadow,
  planned_route = repeat_info$planned_route,
  executed_route = repeat_info$executed_route,
  reroute_reason = repeat_info$reroute_reason,
  solver_status = repeat_info$solver_status
))
assert_true(identical(repeat_result_hash, safe_result_hash),
            "repeated safe solve has an exact same-environment result hash")
repeat_runtime_info <- fixed_sp_cuda_runtime_info(runtime)
repeat_prepared_info <- fixed_sp_cuda_prepared_info(stable_handle)
runtime_reuse_fields <- c(
  "runtime_context_create_count", "stream_create_count",
  "cublas_handle_create_count", "cusolver_handle_create_count",
  "workspace_grow_count", "workspace_bytes", "cublas_workspace_bytes"
)
prepared_reuse_fields <- c(
  "setup_h2d_upload_count", "setup_h2d_bytes",
  "coefficient_output_capacity", "generation"
)
assert_true(
  identical(safe_runtime_info[runtime_reuse_fields],
            repeat_runtime_info[runtime_reuse_fields]) &&
    identical(safe_prepared_info[prepared_reuse_fields],
              repeat_prepared_info[prepared_reuse_fields]) &&
    safe_info$per_target_allocation_count_after_warmup == 0L &&
    repeat_info$per_target_allocation_count_after_warmup == 0L &&
    safe_info$per_target_handle_create_count == 0L &&
    repeat_info$per_target_handle_create_count == 0L,
  "repeat solve reuses workspace, setup uploads, streams, handles, and slots"
)
fixed_sp_cuda_residual_release(repeat_token)
fixed_sp_cuda_residual_free(repeat_token)

solve_stable <- function(Y = stable_native_batch$Y,
                         SP = stable_native_batch$SP,
                         route = stable_native_batch$planned_route) {
  fixed_sp_cuda_solve_batch(
    stable_handle, Y, SP, route, stable_native_batch$target_keys,
    outputs = c("residuals")
  )
}

stable_token <- solve_stable()
on.exit(try(fixed_sp_cuda_residual_free(stable_token), silent = TRUE),
        add = TRUE)
stable_info <- fixed_sp_cuda_residual_info(stable_token)
assert_true(
  identical(stable_info$planned_route,
            stable_native_batch$planned_route) &&
    length(stable_info$executed_route) == 1L &&
    is.na(stable_info$executed_route[[1L]]) &&
    identical(stable_info$reroute_reason, "") &&
    identical(stable_info$solver_status,
              "ERR_STABLE_PATH_NOT_IMPLEMENTED"),
  "Phase 3A stable target reports explicit route and status fields"
)
assert_true(
  !any(c("route", "status") %in% names(stable_info)),
  "residual diagnostics expose no ambiguous route or status fields"
)
assert_true(stable_info$invalid_output_init_count == 1L,
            "stable-only batch initializes outputs invalid once")
assert_true(stable_info$cpu_fallback_count == 0L,
            "stable-only batch has no CPU fallback")
assert_true(
  identical(stable_info$rhs_authority, "cuda-x0-transpose-y") &&
    isTRUE(stable_info$full_cuda_data_plane),
  "production solve retains CUDA RHS authority"
)
assert_true(isTRUE(fixed_sp_cuda_prepared_info(
  stable_handle
)$output_slot_leased), "returned token owns the output-slot lease")

assert_error(
  solve_stable(), "ERR_OUTPUT_SLOT_BUSY",
  "an unreleased token must make the output slot busy"
)
busy_info <- fixed_sp_cuda_residual_info(stable_token)
assert_true(busy_info$output_slot_busy_count == 1L,
            "busy solve attempt is counted on the active token")
fixed_sp_cuda_residual_release(stable_token)
assert_true(!isTRUE(fixed_sp_cuda_prepared_info(
  stable_handle
)$output_slot_leased), "release returns the output slot")
released_info <- fixed_sp_cuda_residual_info(stable_token)
assert_true(released_info$output_slot_release_count == 1L,
            "token records one lease release")

zero_sp <- stable_native_batch$SP
zero_sp[[1L]] <- 0
zero_token <- solve_stable(SP = zero_sp)
zero_info <- fixed_sp_cuda_residual_info(zero_token)
assert_true(
  identical(zero_info$solver_status, "ERR_STABLE_PATH_NOT_IMPLEMENTED"),
  "SP equal to zero is accepted"
)
fixed_sp_cuda_residual_release(zero_token)
fixed_sp_cuda_residual_free(zero_token)

assert_error(
  solve_stable(SP = replace(stable_native_batch$SP, 1L, -1)),
  "SP must be finite and non-negative", "negative SP must fail closed"
)
assert_error(
  solve_stable(SP = replace(stable_native_batch$SP, 1L, Inf)),
  "SP must be finite and non-negative", "nonfinite SP must fail closed"
)
assert_error(
  solve_stable(Y = replace(stable_native_batch$Y, 1L, NA_real_)),
  "Y must be finite", "nonfinite Y must fail closed"
)

stale_token <- solve_stable()
stale_generation <- fixed_sp_cuda_residual_info(
  stale_token
)$slot_generation
fixed_sp_cuda_residual_release(stale_token)
replacement_token <- solve_stable()
replacement_generation <- fixed_sp_cuda_residual_info(
  replacement_token
)$slot_generation
assert_true(
  replacement_generation == stale_generation + 1 &&
    identical(fixed_sp_cuda_residual_info(stale_token)$slot_generation,
              stale_generation),
  "released stale token retains compact metadata after slot reuse"
)
fixed_sp_cuda_residual_free(stale_token)
assert_error(
  solve_stable(), "ERR_OUTPUT_SLOT_BUSY",
  "freeing a stale token cannot release the current lease"
)
fixed_sp_cuda_residual_release(replacement_token)
fixed_sp_cuda_residual_free(replacement_token)
assert_error(
  fixed_sp_cuda_residual_info(replacement_token), "freed",
  "freed residual token rejects use"
)

fixed_sp_cuda_residual_free(stable_token)
fixed_sp_cuda_prepared_free(stable_handle)
fixed_sp_cuda_runtime_free(runtime)

poison_runtime <- fixed_sp_cuda_runtime_create(0L)
on.exit(try(fixed_sp_cuda_runtime_free(poison_runtime), silent = TRUE),
        add = TRUE)
fixed_sp_cuda_runtime_reserve(poison_runtime, 351L, 64L, 1L, 7L, 407L)
poison_handle <- fixed_sp_cuda_prepared_create(poison_runtime, stable_dto)
on.exit(try(fixed_sp_cuda_prepared_free(poison_handle), silent = TRUE),
        add = TRUE)
solve_poison_fixture <- function() {
  fixed_sp_cuda_solve_batch(
    poison_handle, stable_native_batch$Y, stable_native_batch$SP,
    stable_native_batch$planned_route, stable_native_batch$target_keys,
    outputs = c("residuals")
  )
}
poison_token <- solve_poison_fixture()
assert_error(
  .Call(
    "C_fixed_sp_cuda_test_inject_consumer_registration_failure",
    poison_token, PACKAGE = "fastkpc_cuda"
  ),
  "INJECTED_CONSUMER_REGISTRATION_FAILURE",
  "consumer registration test hook injects after the poison-safe transition"
)
poison_info <- fixed_sp_cuda_prepared_info(poison_handle)
assert_true(
  identical(poison_info$output_slot_state, "poisoned") &&
    !isTRUE(poison_info$output_slot_leased) &&
    grepl("INJECTED_CONSUMER_REGISTRATION_FAILURE",
          poison_info$output_slot_poison_reason, fixed = TRUE),
  "prepared diagnostics report the poisoned output slot and first reason"
)
assert_error(
  fixed_sp_cuda_residual_release(poison_token),
  "ERR_OUTPUT_SLOT_POISONED",
  "release fails closed for a poisoned output slot"
)
assert_error(
  solve_poison_fixture(), "ERR_OUTPUT_SLOT_POISONED",
  "the next solve fails closed instead of reporting ordinary contention"
)
poison_info_after_errors <- fixed_sp_cuda_prepared_info(poison_handle)
assert_true(
  identical(poison_info_after_errors$output_slot_state,
            poison_info$output_slot_state) &&
    identical(poison_info_after_errors$output_slot_poison_reason,
              poison_info$output_slot_poison_reason),
  "failed release and solve preserve the first poison reason"
)
fixed_sp_cuda_residual_free(poison_token)
fixed_sp_cuda_residual_free(poison_token)
assert_error(
  fixed_sp_cuda_residual_info(poison_token), "freed",
  "explicit residual free clears the poisoned token"
)
poison_info_after_free <- fixed_sp_cuda_prepared_info(poison_handle)
assert_true(
  identical(poison_info_after_free$output_slot_state,
            poison_info$output_slot_state) &&
    identical(poison_info_after_free$output_slot_poison_reason,
              poison_info$output_slot_poison_reason),
  "explicit residual free preserves the poisoned slot and first reason"
)
fixed_sp_cuda_prepared_free(poison_handle)
fixed_sp_cuda_runtime_free(poison_runtime)

cat(sprintf(
  paste0(
    "METRICS coefficient_max_abs=%.17g coefficient_relative_l2=%.17g ",
    "fitted_max_abs=%.17g fitted_relative_l2=%.17g ",
    "residual_max_abs=%.17g residual_relative_l2=%.17g ",
    "rss_max_abs=%.17g rss_relative_l2=%.17g rhs_max_abs=%.17g\n"
  ),
  coefficient_max_abs, coefficient_relative_l2,
  fitted_max_abs, fitted_relative_l2,
  residual_max_abs, residual_relative_l2,
  rss_max_abs, rss_relative_l2, rhs_max_abs
))
cat("PASS Phase 3A safe and stable fixed-sp solve\n")
