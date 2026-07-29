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
qualification <- readRDS(file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1",
  "qualification_target_keys.rds"
))
cases <- data.frame(
  label = c("high-condition", "rank-deficient"),
  shard_id = c(6L, 56L),
  prepared_s_key_sha256 = c(
    "0409825a8e63ab50aaabc5b415f81c4f4e565acb86fdbe41d7601be525ecbaba",
    "e24dde52f78eab81021dc092fb072111f0d27733d6056b5490162039f8e87094"
  ),
  residual_key_sha256 = c(
    "0000996f68455bcb85082cd0d10586afcd9c90f38daf60dcb9f5ef620d291deb",
    "00354ddff81cd49307434189f6bba0fc009f59b6eba9e118bd34b468583cea69"
  ),
  expected_penalty_count = c(3L, 5L),
  expect_conditioning_rank_risk = c(FALSE, TRUE),
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
    paste0("Phase 5 stability witness is missing: ", case$label)
  )
  state <- shard$target_states[state_index, , drop = FALSE]
  risk <- qualification[match(
    case$residual_key_sha256, qualification$residual_key_sha256
  ), , drop = FALSE]
  target <- fastkpc_full_cuda_materialize_target_state(
    state, data, setup$dataset_sha256
  )
  context <- fastkpc_full_cuda_validate_materialized_target_for_prepared(
    setup, target
  )
  oracle <- fastkpc_mgcv_magic_fixed_sp_from_prepared(setup, target)
  candidate <- fastkpc_full_cuda_phase5_evaluate_cpp(
    setup, context$y, log(context$sp)
  )
  residual_relative_l2 <- relative_l2(
    candidate$residuals, oracle$residuals
  )
  residual_max_absolute <- max(abs(
    candidate$residuals - oracle$residuals
  ))
  assert_true(
    candidate$penalty_count == case$expected_penalty_count &&
      identical(candidate$rank_path, "augmented-jacobi-svd") &&
      !isTRUE(candidate$normal_equations_used) &&
      is.finite(candidate$condition),
    paste0("Phase 5 stable route mismatch: ", case$label)
  )
  assert_true(
    abs(candidate$score - state$GCV_Cp_score[[1L]]) <= 1e-8 &&
      abs(candidate$edf - state$EDF[[1L]]) <= 1e-8 &&
      residual_relative_l2 <= 1e-8 &&
      residual_max_absolute <= 1e-7,
    paste0("Phase 5 stability oracle mismatch: ", case$label)
  )
  assert_true(
    !case$expect_conditioning_rank_risk ||
      (nrow(risk) == 1L && isTRUE(risk$rank_deficient[[1L]]) &&
         is.infinite(risk$condition[[1L]])),
    "Phase 5 rank-risk witness must retain its qualification metadata"
  )
  data.frame(
    label = case$label,
    penalty_count = candidate$penalty_count,
    numerical_rank = candidate$numerical_rank,
    free_dim = candidate$free_dim,
    condition = candidate$condition,
    score_absolute_error =
      abs(candidate$score - state$GCV_Cp_score[[1L]]),
    edf_absolute_error = abs(candidate$edf - state$EDF[[1L]]),
    residual_relative_l2 = residual_relative_l2,
    residual_max_absolute = residual_max_absolute,
    stringsAsFactors = FALSE
  )
})
results <- do.call(rbind, results)
print(results, row.names = FALSE, digits = 17)
cat("PASS Phase 5 multi-penalty C++ stability witnesses\n")
