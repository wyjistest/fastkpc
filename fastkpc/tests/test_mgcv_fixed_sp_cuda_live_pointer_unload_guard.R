source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_phase3_artifacts.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_error <- function(expression, pattern, message) {
  error <- tryCatch({
    force(expression)
    NULL
  }, error = identity)
  assert_true(
    inherits(error, "error") &&
      grepl(pattern, conditionMessage(error), fixed = TRUE),
    paste0(message, if (inherits(error, "error")) {
      paste0(": ", conditionMessage(error))
    } else {
      ": no error"
    })
  )
}
assert_owner_counts <- function(runtime, prepared, residual, message) {
  owners <- fixed_sp_cuda_live_owner_snapshot()
  assert_true(
    identical(owners$runtime, as.integer(runtime)) &&
      identical(owners$prepared, as.integer(prepared)) &&
      identical(owners$residual, as.integer(residual)) &&
      identical(owners$total, as.integer(runtime + prepared + residual)),
    paste0(
      message, "; observed runtime=", owners$runtime,
      " prepared=", owners$prepared,
      " residual=", owners$residual,
      " total=", owners$total
    )
  )
}

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP fixed-SP CUDA live pointer unload guard\n")
  quit(save = "no", status = 0L)
}

fastkpc_full_cuda_phase3_discover_qualified_native_evidence()
if (!isTRUE(.Call("C_fastkpc_cuda_available", PACKAGE = "fastkpc_cuda"))) {
  cat("SKIP fixed-SP CUDA live pointer unload guard: CUDA unavailable\n")
  quit(save = "no", status = 0L)
}

assert_owner_counts(0L, 0L, 0L, "owner ledger starts empty")

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
assert_owner_counts(1L, 0L, 0L, "runtime extptr increments owner ledger")
assert_error(
  .fastkpc_cuda_assert_no_live_fixed_sp_owners(),
  "live fixed-SP CUDA external pointers",
  "live runtime blocks qualified unload/rebuild"
)

qualified_cache <- get(
  ".fastkpc_full_cuda_phase3_qualified_native_cache", envir = .GlobalEnv
)
cached_qualified <- qualified_cache$value
registered_path_before <- normalizePath(
  getLoadedDLLs()[["fastkpc_cuda"]][["path"]],
  winslash = "/", mustWork = TRUE
)
rm("value", envir = qualified_cache)
assert_error(
  fastkpc_full_cuda_phase3_discover_qualified_native_evidence(),
  "live fixed-SP CUDA external pointers",
  "repeated qualified discovery fails closed while runtime is live"
)
assert_true(
  !exists("value", envir = qualified_cache, inherits = FALSE) &&
    identical(
      normalizePath(
        getLoadedDLLs()[["fastkpc_cuda"]][["path"]],
        winslash = "/", mustWork = TRUE
      ),
      registered_path_before
    ),
  "failed live discovery does not alter the registered native image"
)
qualified_cache$value <- cached_qualified

catalog <- fastkpc_full_cuda_open_fixed_sp_catalog(
  file.path("fastkpc", "artifacts", "full_cuda_ci", "oracle_351x48_v1"),
  file.path("fastkpc", "artifacts", "full_cuda_ci",
            "workload_census_351x48_v1"),
  file.path("fastkpc", "artifacts", "full_cuda_ci",
            "prepared_s_contract_v1"),
  file.path("fastkpc", "artifacts", "kpc_tprs_real_zhu",
            "cancer_RD-causalDiscoveryInput.rds"),
  require_full = TRUE
)
iteration <- fastkpc_full_cuda_fixed_sp_scope(catalog, "iteration")
setup_index <- which(iteration$setup_rows$penalty_count > 1L)[[1L]]
selected_key <- iteration$setup_rows$prepared_s_key_sha256[[setup_index]]
scope <- iteration
scope$setup_rows <- iteration$setup_rows[setup_index, , drop = FALSE]
scope$target_rows <- iteration$target_rows[
  iteration$target_rows$prepared_s_key_sha256 == selected_key,
  , drop = FALSE
]
selected_rank <- match(selected_key, catalog$setup_index$prepared_s_key_sha256)
scope$shard_ids <- as.integer(
  (selected_rank - 1L) %% catalog$catalog_contract$shard_count
)
batch <- fastkpc_full_cuda_fixed_sp_batches(catalog, scope)[[1L]]
native_batch <- fastkpc_full_cuda_fixed_sp_native_batch(
  list(
    setup = batch$setup,
    target_rows = batch$target_rows[1L, , drop = FALSE],
    Y = batch$Y[, 1L, drop = FALSE],
    SP = batch$SP[, 1L, drop = FALSE],
    oracle_nullspace_rhs = batch$oracle_nullspace_rhs[, 1L, drop = FALSE],
    planned_route = batch$planned_route[1L],
    condition = batch$condition[1L],
    prepared_s_key_sha256 = batch$prepared_s_key_sha256
  ),
  fastkpc_full_cuda_fixed_sp_native_dto(batch$setup)
)
dto <- fastkpc_full_cuda_fixed_sp_native_dto(batch$setup)
fixed_sp_cuda_runtime_reserve(
  runtime, dto$n, dto$null_dim, 1L, dto$penalty_count,
  as.integer(dto$n + sum(dto$penalty_ranks))
)
handle <- fixed_sp_cuda_prepared_create(runtime, dto)
assert_owner_counts(1L, 1L, 0L, "prepared extptr increments owner ledger")
assert_error(
  .fastkpc_cuda_assert_no_live_fixed_sp_owners(),
  "live fixed-SP CUDA external pointers",
  "live prepared handle blocks qualified unload/rebuild"
)
token <- fixed_sp_cuda_solve_batch(
  handle, native_batch$Y, native_batch$SP, native_batch$planned_route,
  native_batch$target_keys,
  outputs = c("residuals")
)
assert_owner_counts(1L, 1L, 1L, "residual token increments owner ledger")
fixed_sp_cuda_residual_release(token)
assert_owner_counts(
  1L, 1L, 1L,
  "released-but-live residual token still blocks qualified unload/rebuild"
)
fixed_sp_cuda_residual_free(token)
token <- NULL
assert_owner_counts(1L, 1L, 0L, "residual free decrements owner ledger")
fixed_sp_cuda_prepared_free(handle)
handle <- NULL
assert_owner_counts(1L, 0L, 0L, "prepared free decrements owner ledger")
fixed_sp_cuda_runtime_free(runtime)
runtime <- NULL
assert_owner_counts(0L, 0L, 0L, "runtime free decrements owner ledger")
.fastkpc_cuda_assert_no_live_fixed_sp_owners()

local({
  gc_runtime <- fixed_sp_cuda_runtime_create(0L)
  fixed_sp_cuda_runtime_reserve(
    gc_runtime, dto$n, dto$null_dim, 1L, dto$penalty_count,
    as.integer(dto$n + sum(dto$penalty_ranks))
  )
  gc_handle <- fixed_sp_cuda_prepared_create(gc_runtime, dto)
  gc_token <- fixed_sp_cuda_solve_batch(
    gc_handle, native_batch$Y, native_batch$SP, native_batch$planned_route,
    native_batch$target_keys,
    outputs = c("residuals")
  )
  invisible(NULL)
})
for (index in seq_len(4L)) gc()
assert_owner_counts(0L, 0L, 0L, "GC finalizers decrement owner ledger")

fastkpc_full_cuda_phase3_discover_qualified_native_evidence()

cat("PASS fixed-SP CUDA live pointer unload guard\n")
