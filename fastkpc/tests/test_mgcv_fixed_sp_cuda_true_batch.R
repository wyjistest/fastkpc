source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP Phase 3B fixed-sp true-batch diagnostics\n")
  quit(save = "no", status = 0)
}

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

target_rows_by_setup <- split(
  iteration$target_rows,
  as.character(iteration$target_rows$prepared_s_key_sha256)
)
eligible_keys <- iteration$setup_rows$prepared_s_key_sha256[vapply(
  iteration$setup_rows$prepared_s_key_sha256,
  function(setup_key) {
    setup_row <- iteration$setup_rows[
      iteration$setup_rows$prepared_s_key_sha256 == setup_key,
      , drop = FALSE
    ]
    target_rows <- target_rows_by_setup[[setup_key]]
    nrow(setup_row) == 1L && setup_row$penalty_count[[1L]] > 1L &&
      sum(target_rows$planned_route == "CHOLESKY_BATCHED") >= 3L
  },
  logical(1L)
)]
eligible_keys <- sort(eligible_keys, method = "radix")
assert_true(
  length(eligible_keys) >= 1L,
  "iteration scope contains a multi-penalty setup with three safe targets"
)

setup_key <- eligible_keys[[1L]]
selected_scope <- iteration
selected_scope$setup_rows <- iteration$setup_rows[
  iteration$setup_rows$prepared_s_key_sha256 == setup_key,
  , drop = FALSE
]
selected_scope$target_rows <- iteration$target_rows[
  iteration$target_rows$prepared_s_key_sha256 == setup_key,
  , drop = FALSE
]
setup_rank <- match(setup_key, catalog$setup_index$prepared_s_key_sha256)
assert_true(!is.na(setup_rank), "selected setup belongs to the catalog index")
selected_scope$shard_ids <- as.integer(
  (setup_rank - 1L) %% catalog$catalog_contract$shard_count
)
batch <- fastkpc_full_cuda_fixed_sp_batches(
  catalog, selected_scope
)[[setup_key]]
safe_indices <- which(batch$planned_route == "CHOLESKY_BATCHED")
assert_true(
  length(safe_indices) >= 3L && length(batch$setup$penalty_blocks) > 1L,
  "selected payload preserves three safe targets and multiple penalties"
)

subset_target <- function(source, index) {
  list(
    setup = source$setup,
    target_rows = source$target_rows[index, , drop = FALSE],
    Y = source$Y[, index, drop = FALSE],
    SP = source$SP[, index, drop = FALSE],
    oracle_nullspace_rhs =
      source$oracle_nullspace_rhs[, index, drop = FALSE],
    planned_route = source$planned_route[index],
    condition = source$condition[index],
    prepared_s_key_sha256 = source$prepared_s_key_sha256
  )
}

safe_batch <- subset_target(batch, safe_indices)
safe_count <- length(safe_indices)
dto <- fastkpc_full_cuda_fixed_sp_native_dto(safe_batch$setup)
native_batch <- fastkpc_full_cuda_fixed_sp_native_batch(
  safe_batch, dto
)
assert_true(
  identical(native_batch$target_count, safe_count) &&
    all(native_batch$planned_route == "CHOLESKY_BATCHED") &&
    nrow(native_batch$SP) > 1L,
  "Task 2 submits every safe target from the selected setup"
)

oracle_results <- lapply(
  seq_len(safe_count),
  function(index) {
    fastkpc_mgcv_magic_fixed_sp_from_prepared(
      prepared_setup = safe_batch$setup,
      target_state = list(
        row = safe_batch$target_rows[index, , drop = FALSE],
        y = as.numeric(safe_batch$Y[, index])
      )
    )
  }
)
oracle_coefficients <- vapply(
  oracle_results, `[[`, numeric(dto$coefficient_dim), "coefficients"
)
oracle_fitted <- vapply(oracle_results, `[[`, numeric(dto$n), "fitted")
oracle_residuals <- vapply(
  oracle_results, `[[`, numeric(dto$n), "residuals"
)
oracle_rss <- vapply(
  oracle_results,
  function(result) sum(result$residuals^2),
  numeric(1L)
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
  runtime, dto$n, dto$null_dim, safe_count, dto$penalty_count,
  as.integer(dto$n + sum(dto$penalty_ranks))
)
handle <- fixed_sp_cuda_prepared_create(runtime, dto)
token <- fixed_sp_cuda_solve_batch(
  handle, native_batch$Y, native_batch$SP, native_batch$planned_route,
  native_batch$target_keys,
  outputs = c("coefficients", "fitted", "residuals", "rss", "rhs")
)
info <- fixed_sp_cuda_residual_info(token)

required_fields <- c(
  "native_batch_call",
  "batch_call_count",
  "true_batched_kernel",
  "true_batched_subgroup_count",
  "true_batched_attempted_target_count",
  "true_batched_target_count",
  "cholesky_single_target_count",
  "potrf_batched_call_count",
  "potrs_batched_call_count",
  "target_batch_h2d_call_count",
  "target_h2d_copy_count",
  "target_h2d_bytes",
  "canonical_output_order_exact",
  "planned_route",
  "executed_route",
  "reroute_reason",
  "solver_status"
)
missing_fields <- setdiff(required_fields, names(info))
assert_true(
  length(missing_fields) == 0L,
  paste0(
    "Phase 3B residual info fields missing: ",
    paste(missing_fields, collapse = ", ")
  )
)

assert_true(
  isTRUE(info$native_batch_call),
  "one native batch call"
)
assert_true(
  info$batch_call_count == 1L,
  "one public solve call"
)

# Task 2 fuses upload and construction but still sends every safe target
# through the existing single-target factor and solve calls.
assert_true(
  info$true_batched_attempted_target_count == 0L &&
    info$true_batched_target_count == 0L &&
    info$cholesky_single_target_count == safe_count &&
    info$potrf_batched_call_count == 0L &&
    info$potrs_batched_call_count == 0L,
  "Task 2 retains repeated single Cholesky after fused construction"
)

# True-batch flags remain false until Task 3 replaces these single-target
# solver calls with potrfBatched and potrsBatched.
assert_true(
  !isTRUE(info$true_batched_kernel) &&
    info$true_batched_subgroup_count == 0L,
  "Task 2 does not yet execute a true batched kernel"
)
assert_true(
  info$target_batch_h2d_call_count == 1L,
  "one target batch upload phase"
)
assert_true(
  info$target_h2d_copy_count == 2L,
  "one Y and one SP copy"
)
assert_true(
  info$target_h2d_bytes == 8 * (length(native_batch$Y) +
    length(native_batch$SP)),
  "target H2D byte accounting"
)
assert_true(
  info$rhs_device_build_count == 1L &&
    identical(info$rhs_authority, "cuda-x0-transpose-y") &&
    isTRUE(info$full_cuda_data_plane),
  "one CUDA RHS build"
)
assert_true(
  isTRUE(info$canonical_output_order_exact) &&
    identical(info$target_keys, native_batch$target_keys),
  "canonical output mapping preserves target-key order"
)
assert_true(
  identical(info$planned_route, native_batch$planned_route) &&
    identical(info$executed_route,
              rep("CHOLESKY_BATCHED", safe_count)) &&
    identical(info$reroute_reason, rep("", safe_count)) &&
    identical(info$solver_status,
              rep("OK_CHOLESKY_SINGLE", safe_count)),
  "all safe targets preserve canonical route and status order"
)

shadow <- fixed_sp_cuda_materialize_shadow(
  token,
  outputs = c("coefficients", "fitted", "residuals", "rss", "rhs")
)
output_max_abs_errors <- c(
  coefficients = max(abs(shadow$coefficients - oracle_coefficients)),
  fitted = max(abs(shadow$fitted - oracle_fitted)),
  residuals = max(abs(shadow$residuals - oracle_residuals)),
  rss = max(abs(shadow$rss - oracle_rss)),
  rhs = max(abs(shadow$cuda_nullspace_rhs -
                  safe_batch$oracle_nullspace_rhs))
)
numerical_max_abs_error <- max(output_max_abs_errors)
assert_true(
  identical(dim(shadow$coefficients),
            c(dto$coefficient_dim, safe_count)) &&
    identical(dim(shadow$fitted), c(dto$n, safe_count)) &&
    identical(dim(shadow$residuals), c(dto$n, safe_count)) &&
    identical(length(shadow$rss), safe_count) &&
    identical(dim(shadow$cuda_nullspace_rhs),
              c(dto$null_dim, safe_count)) &&
    all(output_max_abs_errors[c(
      "coefficients", "fitted", "residuals", "rss"
    )] < 1e-7) && output_max_abs_errors[["rhs"]] < 1e-12,
  paste0(
    "all canonical safe output columns match the fixed-sp oracle; max=",
    format(numerical_max_abs_error, digits = 17L)
  )
)

fixed_sp_cuda_residual_release(token)
fixed_sp_cuda_residual_free(token)
token <- NULL
fixed_sp_cuda_prepared_free(handle)
handle <- NULL
fixed_sp_cuda_runtime_free(runtime)
runtime <- NULL

cat(
  "PASS Phase 3B fixed-sp fused target upload; max abs error: ",
  format(numerical_max_abs_error, digits = 17L), "\n",
  sep = ""
)
