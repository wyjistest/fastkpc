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

cases <- data.frame(
  label = c("partition-4-edf", "partition-7-trajectory", "partition-12-sp"),
  shard_id = c(36L, 23L, 28L),
  prepared_s_key_sha256 = c(
    "18b367b21d9ab1f391df7f790af330b1e0770e0db7a83c032cfbeb73645b5b9a",
    "2aa8d10950718cdf445b9ebfc9c38698d38aa88ffa4e9e0c878e283b720d7353",
    "c8522edfd10de23093ae68eed6e2196fffec6daa944ec04c687c3c8e2ba7840e"
  ),
  residual_key_sha256 = c(
    "4e8c62a69abbd86d18f2131e33603c5251b3ae5517b41dd4f8208b0c55267523",
    "6daa4be9b8905becae36c2861da8744637c1907ffe8422d6d772e9f77b2bfee1",
    "4ad8305edcf7995e18b75e5a1c7c39ba86aa6f597f11128e1cec5810475bfd55"
  ),
  stringsAsFactors = FALSE
)

data <- readRDS(file.path(
  "fastkpc", "artifacts", "kpc_tprs_real_zhu",
  "cancer_RD-causalDiscoveryInput.rds"
))
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
    paste0("Phase 5 formal witness is missing: ", case$label)
  )
  state <- shard$target_states[state_index, , drop = FALSE]
  target <- fastkpc_full_cuda_materialize_target_state(
    state, data, setup$dataset_sha256
  )
  context <- fastkpc_full_cuda_validate_materialized_target_for_prepared(
    setup, target
  )
  oracle <- fastkpc_mgcv_magic_fixed_sp_from_prepared(setup, target)
  candidate <- fastkpc_full_cuda_phase5_optimize_cpp(setup, context$y)
  fixed_candidate <- fastkpc_full_cuda_phase5_evaluate_cpp(
    setup, context$y, log(context$sp)
  )
  convergence <- state$convergence_fields[[1L]]$mgcv.conv$value
  oracle_initial_log_sp <- log(get(
    "initial.sp", envir = asNamespace("mgcv")
  )(setup$X, setup$penalty_blocks, setup$penalty_offsets))
  sp_error <- max(abs(candidate$selected_log_sp - log(context$sp)))
  score_error <- abs(candidate$score - state$GCV_Cp_score[[1L]])
  edf_error <- abs(candidate$edf - state$EDF[[1L]])
  residual_max_absolute <- max(abs(candidate$residuals - oracle$residuals))
  residual_relative_l2 <- relative_l2(candidate$residuals, oracle$residuals)

  data.frame(
    label = case$label,
    optimizer_iterations = candidate$optimizer_iterations,
    score_calls = candidate$score_calls,
    selected_log_sp_max_error = sp_error,
    score_absolute_error = score_error,
    edf_absolute_error = edf_error,
    residual_max_absolute = residual_max_absolute,
    residual_relative_l2 = residual_relative_l2,
    initial_log_sp_identical = identical(
      unname(candidate$initial_log_sp), unname(oracle_initial_log_sp)
    ),
    fixed_score_absolute_error = abs(
      fixed_candidate$score - state$GCV_Cp_score[[1L]]
    ),
    fixed_edf_absolute_error = abs(
      fixed_candidate$edf - state$EDF[[1L]]
    ),
    trajectory_match =
      candidate$optimizer_iterations == convergence$iter &&
        candidate$score_calls == convergence$score.calls,
    numerical_match =
      sp_error <= 1e-7 && score_error <= 1e-8 && edf_error <= 1e-8 &&
        residual_max_absolute <= 1e-7 && residual_relative_l2 <= 1e-8,
    stringsAsFactors = FALSE
  )
})

results <- do.call(rbind, results)
print(results, row.names = FALSE, digits = 17)
assert_true(
  all(results$initial_log_sp_identical),
  "Phase 5 formal witness initial log-sp vectors drifted"
)
assert_true(
  all(results$trajectory_match),
  "Phase 5 formal witness optimizer trajectories drifted"
)
assert_true(
  all(results$numerical_match),
  "Phase 5 formal witness numerical results drifted"
)
cat("PASS Phase 5 multi-penalty C++ formal failure witnesses\n")
