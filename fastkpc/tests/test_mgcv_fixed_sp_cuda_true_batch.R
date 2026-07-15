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

first_safe_target <- subset_target(batch, safe_indices[[1L]])
dto <- fastkpc_full_cuda_fixed_sp_native_dto(first_safe_target$setup)
native_batch <- fastkpc_full_cuda_fixed_sp_native_batch(
  first_safe_target, dto
)
assert_true(
  identical(native_batch$target_count, 1L) &&
    identical(native_batch$planned_route, "CHOLESKY_BATCHED") &&
    nrow(native_batch$SP) > 1L,
  "Task 1 submits only the first safe target from the selected setup"
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
  runtime, dto$n, dto$null_dim, 1L, dto$penalty_count,
  as.integer(dto$n + sum(dto$penalty_ranks))
)
handle <- fixed_sp_cuda_prepared_create(runtime, dto)
token <- fixed_sp_cuda_solve_batch(
  handle, native_batch$Y, native_batch$SP, native_batch$planned_route,
  native_batch$target_keys, outputs = "residuals"
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
  isTRUE(info$native_batch_call) && info$batch_call_count == 0L,
  "Phase 3A remains one native submission without a Phase 3B batch call"
)

# Attempted targets are those passed to potrfBatched. Successful targets also
# require potrfBatched, potrsBatched, and OK_CHOLESKY_BATCHED.
assert_true(
  info$true_batched_attempted_target_count == 0L &&
    info$true_batched_target_count == 0L &&
    info$cholesky_single_target_count == 0L &&
    info$potrf_batched_call_count == 0L &&
    info$potrs_batched_call_count == 0L,
  "Phase 3A reports no Phase 3B Cholesky batch activity"
)

# The whole-batch flag requires at least two public targets and every target to
# be OK_CHOLESKY_BATCHED. One subgroup requires two safe targets to enter the
# batched factor and solve.
assert_true(
  !isTRUE(info$true_batched_kernel) &&
    info$true_batched_subgroup_count == 0L,
  "single-target Phase 3A execution is not a true batched kernel"
)
assert_true(
  info$target_batch_h2d_call_count == 0L &&
    info$target_h2d_copy_count == 0L && info$target_h2d_bytes == 0 &&
    !isTRUE(info$canonical_output_order_exact),
  "Phase 3A reports no fused target upload or batched output ordering"
)
assert_true(
  identical(info$planned_route, "CHOLESKY_BATCHED") &&
    identical(info$executed_route, "CHOLESKY_BATCHED") &&
    identical(info$reroute_reason, "") &&
    identical(info$solver_status, "OK_CHOLESKY_SINGLE"),
  "Phase 3A route and status diagnostics remain unchanged"
)

fixed_sp_cuda_residual_release(token)
fixed_sp_cuda_residual_free(token)
token <- NULL
fixed_sp_cuda_prepared_free(handle)
handle <- NULL
fixed_sp_cuda_runtime_free(runtime)
runtime <- NULL

cat("PASS Phase 3B fixed-sp true-batch diagnostics\n")
