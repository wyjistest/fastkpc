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
stable_target <- iteration$target_rows[
  iteration$target_rows$planned_route != "CHOLESKY_BATCHED",
  , drop = FALSE
][1L, , drop = FALSE]
stable_key <- stable_target$prepared_s_key_sha256[[1L]]
stable_scope <- iteration
stable_scope$setup_rows <- iteration$setup_rows[
  iteration$setup_rows$prepared_s_key_sha256 == stable_key,
  , drop = FALSE
]
stable_scope$target_rows <- stable_target
stable_rank <- match(stable_key, catalog$setup_index$prepared_s_key_sha256)
stable_scope$shard_ids <- as.integer(
  (stable_rank - 1L) %% catalog$catalog_contract$shard_count
)
batches <- fastkpc_full_cuda_fixed_sp_batches(catalog, stable_scope)

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

stable_batch <- subset_target(batches[[1L]], 1L)
assert_true(
  nrow(stable_batch$target_rows) == 1L &&
    ncol(stable_batch$Y) == 1L && ncol(stable_batch$SP) == 1L &&
    ncol(stable_batch$oracle_nullspace_rhs) == 1L &&
    stable_batch$planned_route != "CHOLESKY_BATCHED",
  "focused batch preserves exactly one stable target"
)
stable_dto <- fastkpc_full_cuda_fixed_sp_native_dto(stable_batch$setup)
stable_native_batch <- fastkpc_full_cuda_fixed_sp_native_batch(
  stable_batch, stable_dto
)
assert_true(
  identical(names(stable_native_batch),
            c("Y", "SP", "planned_route", "target_keys", "target_count")) &&
    stable_native_batch$target_count == 1L,
  "production native batch excludes the CPU oracle RHS"
)

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
explicit_handle <- fixed_sp_cuda_prepared_create(runtime, explicit_dto)
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
assert_true(
  stable_dto$coefficient_dim > explicit_q &&
    identical(explicit_info$coefficient_dim, stable_dto$coefficient_dim),
  "coefficient diagnostics retain full-space p when p exceeds q"
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

assert_error(
  solve_stable(route = "CHOLESKY_BATCHED"),
  "Phase 3A Cholesky solve is not implemented",
  "Task 6 Cholesky input must throw clearly"
)
assert_true(!isTRUE(fixed_sp_cuda_prepared_info(
  stable_handle
)$output_slot_leased), "Cholesky exception releases its acquired lease")

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

cat("PASS Phase 3A stable fixed-sp solve\n")
