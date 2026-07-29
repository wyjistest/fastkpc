source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cpp.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cuda.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS", unset = "0"), "1")) {
  cat("SKIP Phase 6 persistent CUDA batch: FASTKPC_RUN_CUDA_TESTS != 1\n")
  quit(save = "no", status = 0)
}
if (!isTRUE(tryCatch(fastkpc_cuda_available(), error = function(error) FALSE))) {
  cat("SKIP Phase 6 persistent CUDA batch: CUDA unavailable\n")
  quit(save = "no", status = 0)
}

data <- readRDS(file.path(
  "fastkpc", "artifacts", "kpc_tprs_real_zhu",
  "cancer_RD-causalDiscoveryInput.rds"
))
shard <- readRDS(file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1",
  "shards", "shard_1.rds"
))
setup_key <-
  "05b18f09b303d481e84ec14c6a3d11da92db2fcc0f9f213aa8319015f7369ae4"
setup <- shard$prepared_s_setups[[setup_key]]
state_indices <- head(which(
  shard$target_states$prepared_s_key_sha256 == setup_key
), 2L)
assert_true(
  !is.null(setup) && length(state_indices) == 2L && ncol(setup$X) == 64L &&
    length(setup$penalty_blocks) == 7L,
  "Phase 6 persistent CUDA fixture is malformed"
)
states <- shard$target_states[state_indices, , drop = FALSE]
contexts <- lapply(seq_len(nrow(states)), function(index) {
  target <- fastkpc_full_cuda_materialize_target_state(
    states[index, , drop = FALSE], data, setup$dataset_sha256
  )
  fastkpc_full_cuda_validate_materialized_target_for_prepared(setup, target)
})
Y <- do.call(cbind, lapply(contexts, `[[`, "y"))
target_keys <- states$residual_key_sha256
prepared <- fastkpc_full_cuda_phase6_prepare(setup)
handle <- fastkpc_full_cuda_phase6_prepared_create(
  prepared, target_capacity = ncol(Y)
)
on.exit(
  try(full_cuda_ci_multi_penalty_gcv_prepared_free_native(handle),
      silent = TRUE),
  add = TRUE
)
before <- full_cuda_ci_multi_penalty_gcv_prepared_info_native(handle)
assert_true(
  before$n == nrow(Y) && before$coefficient_dim == 64L &&
    before$penalty_count == 7L && before$target_capacity == ncol(Y) &&
    before$setup_upload_count == 1L && before$setup_h2d_bytes > 0 &&
    before$device_allocation_count == 1L &&
    before$workspace_grow_count == 0L && before$solve_count == 0L &&
    before$residual_shadow_d2h_count == 0L &&
    !isTRUE(before$residual_slot_leased),
  "Phase 6 persistent CUDA setup diagnostics drifted"
)

batch <- fastkpc_full_cuda_phase6_optimize_prepared(
  handle, Y, target_keys
)
token <- batch$residual
on.exit(
  try(full_cuda_ci_multi_penalty_gcv_residual_free_native(token),
      silent = TRUE),
  add = TRUE
)
candidate <- batch$optimization
info <- full_cuda_ci_multi_penalty_gcv_residual_info_native(token)
after <- full_cuda_ci_multi_penalty_gcv_prepared_info_native(handle)
assert_true(
  all(candidate$optimizer_status == 0L) &&
    identical(
      candidate$diagnostics$execution_strategy,
      "persistent-one-setup-one-block-per-target-independent-optimizer"
    ) && candidate$diagnostics$device_allocation_count == 0L &&
    candidate$diagnostics$h2d_copy_count == 1L &&
    candidate$diagnostics$d2h_copy_count == 1L &&
    info$n == nrow(Y) && info$target_count == ncol(Y) &&
    identical(info$target_keys, target_keys) &&
    all(info$optimizer_status == 0L) && isTRUE(info$device_resident) &&
    !isTRUE(info$released) && isTRUE(after$residual_slot_leased) &&
    after$solve_count == 1L && after$cublas_gemm_count == 1L &&
    after$residual_kernel_count == 1L &&
    after$residual_shadow_d2h_count == 0L &&
    after$workspace_grow_count == 0L,
  "Phase 6 persistent CUDA solve authority diagnostics drifted"
)

busy_error <- tryCatch(
  {
    fastkpc_full_cuda_phase6_optimize_prepared(handle, Y, target_keys)
    NULL
  },
  error = identity
)
assert_true(
  inherits(busy_error, "error") && grepl(
    "ERR_MULTI_PENALTY_OUTPUT_SLOT_BUSY",
    conditionMessage(busy_error), fixed = TRUE
  ),
  "Phase 6 persistent CUDA output slot must fail closed while leased"
)

shadow <- full_cuda_ci_multi_penalty_gcv_residual_shadow_native(token)
reference <- lapply(seq_len(ncol(Y)), function(target) {
  fastkpc_full_cuda_phase5_optimize_cpp(setup, Y[, target])
})
reference_residuals <- do.call(cbind, lapply(reference, `[[`, "residuals"))
reference_fitted <- do.call(cbind, lapply(reference, `[[`, "fitted"))
reference_coefficients <- do.call(
  cbind, lapply(reference, `[[`, "coefficients")
)
assert_true(
  max(abs(shadow$residuals - reference_residuals)) <= 1e-7 &&
    max(abs(shadow$fitted - reference_fitted)) <= 1e-7 &&
    max(abs(shadow$coefficients - reference_coefficients)) <= 1e-7 &&
    max(abs(shadow$residuals - (Y - shadow$fitted))) <= 1e-14,
  "Phase 6 persistent CUDA GEMM residual shadow drifted"
)
after_shadow <- full_cuda_ci_multi_penalty_gcv_prepared_info_native(handle)
assert_true(
  after_shadow$residual_shadow_d2h_count == 3L &&
    after_shadow$residual_shadow_d2h_bytes > 0,
  "Phase 6 explicit residual shadow accounting drifted"
)

full_cuda_ci_multi_penalty_gcv_residual_release_native(token)
released <- full_cuda_ci_multi_penalty_gcv_residual_info_native(token)
after_release <- full_cuda_ci_multi_penalty_gcv_prepared_info_native(handle)
assert_true(
  isTRUE(released$released) && !isTRUE(released$device_resident) &&
    !isTRUE(after_release$residual_slot_leased),
  "Phase 6 persistent CUDA residual release drifted"
)
full_cuda_ci_multi_penalty_gcv_residual_free_native(token)

second <- fastkpc_full_cuda_phase6_optimize_prepared(handle, Y, target_keys)
second_token <- second$residual
on.exit(
  try(full_cuda_ci_multi_penalty_gcv_residual_free_native(second_token),
      silent = TRUE),
  add = TRUE
)
second_info <- full_cuda_ci_multi_penalty_gcv_prepared_info_native(handle)
assert_true(
  identical(second$optimization$selected_log_sp,
            candidate$selected_log_sp) &&
    identical(second$optimization$optimizer_iterations,
              candidate$optimizer_iterations) &&
    second_info$solve_count == 2L && second_info$cublas_gemm_count == 2L &&
    second_info$residual_kernel_count == 2L &&
    second_info$workspace_grow_count == 0L &&
    second_info$device_allocation_count == before$device_allocation_count,
  "Phase 6 persistent CUDA workspace reuse or determinism drifted"
)
full_cuda_ci_multi_penalty_gcv_residual_free_native(second_token)
full_cuda_ci_multi_penalty_gcv_prepared_free_native(handle)

cat("PASS Phase 6 persistent CUDA optimizer, GEMM, and residual token\n")
