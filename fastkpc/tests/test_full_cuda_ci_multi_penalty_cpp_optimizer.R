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
  label = c(
    "reference", "five-penalty-stability", "indefinite-hessian",
    "rank-deficient"
  ),
  shard_id = c(1L, 56L, 0L, 37L),
  prepared_s_key_sha256 = c(
    "001245052f571033286b2dc7526c24dbe5ec5c221660c094a8b9f052376b91da",
    "e24dde52f78eab81021dc092fb072111f0d27733d6056b5490162039f8e87094",
    "334c0e1a40c98ae302ced2a9573755e48fe19be1d9e7753862a00b620ea71746",
    "2959705cc686141ff4e758d64525e70a211bc845a21ee3085c97d0b5d8abedab"
  ),
  residual_key_sha256 = c(
    NA_character_,
    "00354ddff81cd49307434189f6bba0fc009f59b6eba9e118bd34b468583cea69",
    "7795b8ddd608d04621c3474df34751044f21077eab6089130662ec93a338fdff",
    "1db147c7e70b187f54304a14982184b7bc5b221d71051abfbfc21c3312399432"
  ),
  stringsAsFactors = FALSE
)

results <- lapply(seq_len(nrow(cases)), function(index) {
  case <- cases[index, , drop = FALSE]
  shard <- readRDS(file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1",
    "shards", paste0("shard_", case$shard_id, ".rds")
  ))
  setup <- shard$prepared_s_setups[[case$prepared_s_key_sha256]]
  state_index <- if (is.na(case$residual_key_sha256)) {
    which(
      shard$target_states$prepared_s_key_sha256 ==
        case$prepared_s_key_sha256
    )[[1L]]
  } else {
    match(
      case$residual_key_sha256,
      shard$target_states$residual_key_sha256
    )
  }
  assert_true(
    !is.null(setup) && !is.na(state_index),
    paste0("Phase 5 optimizer witness is missing: ", case$label)
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
  candidate <- fastkpc_full_cuda_phase5_optimize_cpp(
    setup, context$y, keep_transcript = TRUE
  )

  assert_true(
    identical(
      candidate$schema_version,
      "full-cuda-ci-multi-penalty-cpp-optimization-v1"
    ) &&
      identical(candidate$rank_path, "augmented-jacobi-svd") &&
      identical(
        candidate$selected_fit_refinement_path,
        "augmented-lapack-dgesdd-svd"
      ) &&
      isTRUE(candidate$response_independent_initialization) &&
      !isTRUE(candidate$normal_equations_used) &&
      identical(candidate$fallback_reason, "NONE"),
    paste0("Phase 5 optimizer contract mismatch: ", case$label)
  )
  initial_sp <- get("initial.sp", envir = asNamespace("mgcv"))(
    setup$X, setup$penalty_blocks, setup$penalty_offsets
  )
  assert_true(
    max(abs(candidate$initial_log_sp - log(initial_sp))) <= 1e-12,
    paste0("Phase 5 response-independent initialization drift: ", case$label)
  )
  assert_true(
    max(abs(candidate$selected_log_sp - log(context$sp))) <= 1e-7 &&
      abs(candidate$score - state$GCV_Cp_score[[1L]]) <= 1e-8 &&
      abs(candidate$edf - state$EDF[[1L]]) <= 1e-8 &&
      relative_l2(candidate$residuals, oracle$residuals) <= 1e-8 &&
      max(abs(candidate$residuals - oracle$residuals)) <= 1e-7,
    paste0("Phase 5 selected fit drift: ", case$label)
  )
  assert_true(
    candidate$optimizer_iterations == convergence$iter &&
      candidate$score_calls == convergence$score.calls &&
      identical(
        candidate$fully_converged, convergence$fully.converged
      ) &&
      identical(
        candidate$hessian_positive_definite, convergence$hess.pos.def
      ) &&
      candidate$numerical_rank == convergence$rank &&
      candidate$free_dim == convergence$full.rank &&
      abs(candidate$rms_gradient - convergence$rms.grad) <= 1e-7,
    paste0("Phase 5 convergence diagnostic drift: ", case$label)
  )
  assert_true(
    is.list(candidate$transcript) && length(candidate$transcript) > 0L &&
      all(c("iteration_state", "step_trial", "boundary_probe") %in%
            vapply(candidate$transcript, `[[`, character(1L), "stage")) &&
      length(candidate$boundary_status) == candidate$penalty_count,
    paste0("Phase 5 optimizer transcript is incomplete: ", case$label)
  )
  if (identical(case$label, "indefinite-hessian")) {
    assert_true(
      any(grepl(
        "steepest_descent",
        vapply(candidate$transcript, `[[`, character(1L), "step_source"),
        fixed = TRUE
      )),
      "Phase 5 indefinite-Hessian witness must exercise steepest descent"
    )
  }

  data.frame(
    label = case$label,
    penalty_count = candidate$penalty_count,
    optimizer_iterations = candidate$optimizer_iterations,
    score_calls = candidate$score_calls,
    step_halving_count = candidate$step_halving_count,
    boundary_probe_count = candidate$boundary_probe_count,
    selected_log_sp_max_error =
      max(abs(candidate$selected_log_sp - log(context$sp))),
    score_absolute_error =
      abs(candidate$score - state$GCV_Cp_score[[1L]]),
    residual_relative_l2 =
      relative_l2(candidate$residuals, oracle$residuals),
    stringsAsFactors = FALSE
  )
})
results <- do.call(rbind, results)
print(results, row.names = FALSE, digits = 17)
cat("PASS Phase 5 multi-penalty C++ optimizer parity\n")
