source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cpp.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

setup_key <-
  "ee2c83c20b7a5ccdc34db211bf4b5e09749765bbbef9f818b4336115ba515579"
residual_keys <- c(
  "1607648a62eb6b2e6b5ed1d85c22eb1f67a9515eae7cea517631150efab0fc69",
  "9bf4590c478ef3cc0d8fe0c8a1db6feae6124507cfcc5f7a05ef9c9f2d0c4dc6"
)
shard <- readRDS(file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1",
  "shards", "shard_3.rds"
))
setup <- shard$prepared_s_setups[[setup_key]]
state_indices <- match(
  residual_keys, shard$target_states$residual_key_sha256
)
assert_true(
  !is.null(setup) && !anyNA(state_indices),
  "Phase 5 p-value refinement witness is missing"
)
states <- shard$target_states[state_indices, , drop = FALSE]
data <- readRDS(file.path(
  "fastkpc", "artifacts", "kpc_tprs_real_zhu",
  "cancer_RD-causalDiscoveryInput.rds"
))

candidate_residuals <- oracle_residuals <- matrix(
  NA_real_, nrow(data), length(residual_keys)
)
residual_errors <- numeric(length(residual_keys))
for (index in seq_along(residual_keys)) {
  target <- fastkpc_full_cuda_materialize_target_state(
    states[index, , drop = FALSE], data, setup$dataset_sha256
  )
  context <- fastkpc_full_cuda_validate_materialized_target_for_prepared(
    setup, target
  )
  candidate <- fastkpc_full_cuda_phase5_optimize_cpp(setup, context$y)
  oracle <- fastkpc_mgcv_magic_fixed_sp_from_prepared(setup, target)
  assert_true(
    identical(
      candidate$selected_fit_refinement_path,
      "pivoted-qr-augmented-lapack-dgesdd-svd"
    ),
    "Phase 5 selected fit must declare LAPACK SVD refinement"
  )
  candidate_residuals[, index] <- candidate$residuals
  oracle_residuals[, index] <- oracle$residuals
  residual_errors[[index]] <- max(abs(
    candidate$residuals - oracle$residuals
  ))
}

old_mode <- Sys.getenv(
  "FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK", unset = NA_character_
)
on.exit({
  if (is.na(old_mode)) {
    Sys.unsetenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK")
  } else {
    Sys.setenv(FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = old_mode)
  }
}, add = TRUE)
Sys.setenv(FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = "spectra")

candidate <- legacy_dcov_gamma_cpp_component_cache_batch_native(
  candidate_residuals, 1L, 2L, numCol = 35L, index = 1
)
oracle <- legacy_dcov_gamma_cpp_component_cache_batch_native(
  oracle_residuals, 1L, 2L, numCol = 35L, index = 1
)
p_value_error <- abs(candidate$p.value - oracle$p.value)
assert_true(
  max(residual_errors) <= 1e-9 && p_value_error <= 1e-10 &&
    identical(candidate$p.value > 0.1, oracle$p.value > 0.1),
  "Phase 5 selected fit refinement must satisfy the p-value contract"
)

print(data.frame(
  residual_key_sha256 = residual_keys,
  residual_max_absolute = residual_errors,
  stringsAsFactors = FALSE
), row.names = FALSE, digits = 17)
cat(
  "p_value_error=", format(p_value_error, digits = 17), "\n",
  "PASS Phase 5 selected fit p-value refinement\n", sep = ""
)
