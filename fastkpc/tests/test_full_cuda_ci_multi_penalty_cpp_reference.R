source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cpp.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
relative_l2 <- function(candidate, reference) {
  sqrt(sum((candidate - reference)^2)) /
    max(sqrt(sum(reference^2)), 1e-300)
}

data <- readRDS(file.path(
  "fastkpc", "artifacts", "kpc_tprs_real_zhu",
  "cancer_RD-causalDiscoveryInput.rds"
))
setup_key <-
  "001245052f571033286b2dc7526c24dbe5ec5c221660c094a8b9f052376b91da"
shard <- readRDS(file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1",
  "shards", "shard_1.rds"
))
setup <- shard$prepared_s_setups[[setup_key]]
state_index <- which(
  shard$target_states$prepared_s_key_sha256 == setup_key
)[[1L]]
state <- shard$target_states[state_index, , drop = FALSE]
target <- fastkpc_full_cuda_materialize_target_state(
  state, data, setup$dataset_sha256
)
context <- fastkpc_full_cuda_validate_materialized_target_for_prepared(
  setup, target
)
oracle <- fastkpc_mgcv_magic_fixed_sp_from_prepared(setup, target)

assert_true(
  identical(setup$formula_class, "additive-smooth") &&
    length(setup$sorted_S) == 3L &&
    length(setup$penalty_blocks) == 3L,
  "Phase 5 reference case must exercise three additive penalties"
)

evaluate <- function(log_sp) {
  fastkpc_full_cuda_phase5_evaluate_cpp(setup, context$y, log_sp)
}
log_sp <- log(context$sp)
candidate <- evaluate(log_sp)
assert_true(
  identical(candidate$schema_version,
            "full-cuda-ci-multi-penalty-cpp-evaluation-v1") &&
    identical(candidate$rank_path, "augmented-jacobi-svd") &&
    isTRUE(candidate$constraint_aware) &&
    !isTRUE(candidate$normal_equations_used) &&
    candidate$penalty_count == 3L,
  "Phase 5 C++ evaluation must declare the stable multi-penalty path"
)
assert_true(
  abs(candidate$score - state$GCV_Cp_score[[1L]]) <= 1e-8 &&
    abs(candidate$edf - state$EDF[[1L]]) <= 1e-8,
  "Phase 5 fixed-sp score and EDF must match the pinned oracle"
)
assert_true(
  relative_l2(candidate$residuals, oracle$residuals) <= 1e-8 &&
    max(abs(candidate$residuals - oracle$residuals)) <= 1e-7,
  "Phase 5 fixed-sp residual must satisfy numerical_contract_v1"
)

epsilon <- 2e-5
gradient_fd <- numeric(length(log_sp))
hessian_fd <- matrix(0, length(log_sp), length(log_sp))
for (component in seq_along(log_sp)) {
  upper <- lower <- log_sp
  upper[[component]] <- upper[[component]] + epsilon
  lower[[component]] <- lower[[component]] - epsilon
  upper_value <- evaluate(upper)
  lower_value <- evaluate(lower)
  gradient_fd[[component]] <-
    (upper_value$score - lower_value$score) / (2 * epsilon)
  hessian_fd[, component] <-
    (upper_value$gradient - lower_value$gradient) / (2 * epsilon)
}
assert_true(
  max(abs(candidate$gradient - gradient_fd)) <= 2e-7,
  "Phase 5 analytic objective gradient must match finite differences"
)
assert_true(
  max(abs(candidate$hessian - hessian_fd)) <= 2e-6 &&
    max(abs(candidate$hessian - t(candidate$hessian))) <= 1e-12,
  "Phase 5 analytic objective Hessian must match finite differences"
)

cat("PASS Phase 5 multi-penalty C++ stable objective reference\n")
