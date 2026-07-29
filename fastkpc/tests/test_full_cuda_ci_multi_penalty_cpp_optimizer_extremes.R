source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
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
cases <- data.frame(
  label = c("maximum-iterations", "maximum-score-calls"),
  shard_id = c(10L, 59L),
  prepared_s_key_sha256 = c(
    "31ff054083fb4a0cdfb0be76305e057a6ab086c134f68a0ba7fbae8661ab2e14",
    "7844c90be2027d3b78f4750e99ca43cb6ee8621e0aca651d2422e3884eb81168"
  ),
  residual_key_sha256 = c(
    "8e56ba8371592d8daed48dfa00cac511893d66746ca43f2941954dfa1cf8e81e",
    "d5333f96a4468977ddc61a6526d5a7f255885a9f514f4e990188770e474088d3"
  ),
  expected_iterations = c(187L, 149L),
  expected_score_calls = c(696L, 575L),
  expected_hessian_positive = c(TRUE, FALSE),
  stringsAsFactors = FALSE
)

results <- lapply(seq_len(nrow(cases)), function(index) {
  case <- cases[index, , drop = FALSE]
  shard <- readRDS(file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1",
    "shards", paste0("shard_", case$shard_id, ".rds")
  ))
  setup <- shard$prepared_s_setups[[case$prepared_s_key_sha256]]
  state_index <- match(
    case$residual_key_sha256, shard$target_states$residual_key_sha256
  )
  assert_true(
    !is.null(setup) && !is.na(state_index),
    paste0("Phase 5 extreme optimizer witness is missing: ", case$label)
  )
  state <- shard$target_states[state_index, , drop = FALSE]
  target <- fastkpc_full_cuda_materialize_target_state(
    state, data, setup$dataset_sha256
  )
  context <- fastkpc_full_cuda_validate_materialized_target_for_prepared(
    setup, target
  )
  oracle <- fastkpc_mgcv_magic_fixed_sp_from_prepared(setup, target)
  convergence <- state$convergence_fields[[1L]]$mgcv.conv$value
  elapsed <- system.time({
    candidate <- fastkpc_full_cuda_phase5_optimize_cpp(setup, context$y)
  })[["elapsed"]]

  assert_true(
    candidate$optimizer_iterations == case$expected_iterations &&
      candidate$score_calls == case$expected_score_calls &&
      convergence$iter == case$expected_iterations &&
      convergence$score.calls == case$expected_score_calls &&
      identical(
        candidate$hessian_positive_definite,
        case$expected_hessian_positive[[1L]]
      ) &&
      identical(
        candidate$hessian_positive_definite, convergence$hess.pos.def
      ) &&
      isTRUE(candidate$fully_converged),
    paste0("Phase 5 extreme convergence drift: ", case$label)
  )
  assert_true(
    max(abs(candidate$selected_log_sp - log(context$sp))) <= 1e-6 &&
      abs(candidate$score - state$GCV_Cp_score[[1L]]) <= 1e-8 &&
      abs(candidate$edf - state$EDF[[1L]]) <= 1e-8 &&
      relative_l2(candidate$residuals, oracle$residuals) <= 1e-8 &&
      max(abs(candidate$residuals - oracle$residuals)) <= 1e-7,
    paste0("Phase 5 extreme selected fit drift: ", case$label)
  )
  assert_true(
    candidate$objective_calls > candidate$score_calls &&
      candidate$boundary_probe_count > 0L,
    paste0("Phase 5 extreme objective accounting drift: ", case$label)
  )
  data.frame(
    label = case$label,
    penalty_count = candidate$penalty_count,
    optimizer_iterations = candidate$optimizer_iterations,
    score_calls = candidate$score_calls,
    objective_calls = candidate$objective_calls,
    step_halving_count = candidate$step_halving_count,
    selected_log_sp_max_error =
      max(abs(candidate$selected_log_sp - log(context$sp))),
    residual_relative_l2 =
      relative_l2(candidate$residuals, oracle$residuals),
    elapsed_sec = elapsed,
    stringsAsFactors = FALSE
  )
})
results <- do.call(rbind, results)

case <- cases[1L, , drop = FALSE]
shard <- readRDS(file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1",
  "shards", paste0("shard_", case$shard_id, ".rds")
))
setup <- shard$prepared_s_setups[[case$prepared_s_key_sha256]]
state <- shard$target_states[
  match(case$residual_key_sha256, shard$target_states$residual_key_sha256),
  , drop = FALSE
]
target <- fastkpc_full_cuda_materialize_target_state(
  state, data, setup$dataset_sha256
)
context <- fastkpc_full_cuda_validate_materialized_target_for_prepared(
  setup, target
)
iteration_error <- tryCatch(
  {
    fastkpc_full_cuda_phase5_optimize_cpp(
      setup, context$y, max_iterations = 4L
    )
    NULL
  },
  error = identity
)
assert_true(
  inherits(iteration_error, "error") &&
    grepl("exceeded its iteration limit", conditionMessage(iteration_error),
          fixed = TRUE),
  "Phase 5 iteration-limit exhaustion must fail closed"
)

print(results, row.names = FALSE, digits = 17)
cat("PASS Phase 5 multi-penalty C++ optimizer extremes\n")
