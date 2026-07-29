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
  cat("SKIP Phase 6 multi-penalty CUDA: FASTKPC_RUN_CUDA_TESTS != 1\n")
  quit(save = "no", status = 0)
}
if (!isTRUE(tryCatch(fastkpc_cuda_available(), error = function(error) FALSE))) {
  cat("SKIP Phase 6 multi-penalty CUDA: CUDA unavailable\n")
  quit(save = "no", status = 0)
}

cases <- data.frame(
  label = c("three-penalty", "four-penalty", "seven-penalty"),
  shard_id = c(36L, 28L, 1L),
  prepared_s_key_sha256 = c(
    "18b367b21d9ab1f391df7f790af330b1e0770e0db7a83c032cfbeb73645b5b9a",
    "c8522edfd10de23093ae68eed6e2196fffec6daa944ec04c687c3c8e2ba7840e",
    "05b18f09b303d481e84ec14c6a3d11da92db2fcc0f9f213aa8319015f7369ae4"
  ),
  expected_p = c(28L, 37L, 64L),
  expected_penalty_count = c(3L, 4L, 7L),
  stringsAsFactors = FALSE
)
data <- readRDS(file.path(
  "fastkpc", "artifacts", "kpc_tprs_real_zhu",
  "cancer_RD-causalDiscoveryInput.rds"
))

rows <- lapply(seq_len(nrow(cases)), function(case_index) {
  case <- cases[case_index, , drop = FALSE]
  shard <- readRDS(file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1",
    "shards", paste0("shard_", case$shard_id, ".rds")
  ))
  setup <- shard$prepared_s_setups[[case$prepared_s_key_sha256]]
  state_indices <- head(which(
    shard$target_states$prepared_s_key_sha256 ==
      case$prepared_s_key_sha256
  ), 3L)
  assert_true(
    !is.null(setup) && length(state_indices) >= 2L &&
      ncol(setup$X) == case$expected_p &&
      length(setup$penalty_blocks) == case$expected_penalty_count,
    paste0("Phase 6 reference fixture is malformed: ", case$label)
  )
  contexts <- lapply(state_indices, function(index) {
    state <- shard$target_states[index, , drop = FALSE]
    target <- fastkpc_full_cuda_materialize_target_state(
      state, data, setup$dataset_sha256
    )
    fastkpc_full_cuda_validate_materialized_target_for_prepared(
      setup, target
    )
  })
  Y <- do.call(cbind, lapply(contexts, `[[`, "y"))
  log_sp <- do.call(cbind, lapply(contexts, function(value) log(value$sp)))
  prepared <- fastkpc_full_cuda_phase6_prepare(setup)
  candidate <- fastkpc_full_cuda_phase6_evaluate_cuda(
    prepared, Y, log_sp
  )
  reference <- lapply(seq_len(ncol(Y)), function(target_index) {
    fastkpc_full_cuda_phase5_evaluate_cpp(
      setup, Y[, target_index], log_sp[, target_index]
    )
  })

  assert_true(
    identical(
      candidate$schema_version,
      "full-cuda-ci-multi-penalty-gcv-cuda-evaluation-v1"
    ) && identical(
      candidate$rank_path,
      "cuda-dpstf2-guarded-qr-lapack-3.12-dgesdd"
    ) && candidate$n == nrow(Y) &&
      candidate$coefficient_dim == case$expected_p &&
      candidate$penalty_count == case$expected_penalty_count &&
      candidate$target_count == ncol(Y),
    paste0("Phase 6 CUDA result schema drifted: ", case$label)
  )
  diagnostics <- candidate$diagnostics
  assert_true(
    identical(
      diagnostics$execution_strategy,
      "one-setup-one-block-per-target-true-batch"
    ) && diagnostics$prepared_setup_upload_count == 1L &&
      diagnostics$target_batch_upload_count == 1L &&
      diagnostics$cuda_qt_y_kernel_launch_count == 1L &&
      diagnostics$cuda_objective_kernel_launch_count == 1L &&
      diagnostics$cuda_objective_target_count == ncol(Y) &&
      diagnostics$cuda_guarded_qr_evaluation_count +
        diagnostics$cuda_stable_svd_evaluation_count == ncol(Y) &&
      diagnostics$cpu_objective_count == 0L &&
      diagnostics$cpu_multi_penalty_solve_count == 0L &&
      diagnostics$fallback_count == 0L &&
      diagnostics$cuda_error_count == 0L &&
      diagnostics$svd_nonconverged_count == 0L &&
      isTRUE(diagnostics$target_specific_log_sp) &&
      isTRUE(diagnostics$true_batched_kernel) &&
      !isTRUE(diagnostics$normal_equations_used),
    paste0("Phase 6 CUDA authority diagnostics drifted: ", case$label)
  )

  reference_score <- vapply(reference, `[[`, numeric(1L), "score")
  reference_edf <- vapply(reference, `[[`, numeric(1L), "edf")
  reference_gradient <- do.call(cbind, lapply(reference, `[[`, "gradient"))
  reference_coefficients <- do.call(
    cbind, lapply(reference, `[[`, "coefficients")
  )
  reference_hessian <- array(0, dim(candidate$hessian))
  for (target_index in seq_along(reference)) {
    reference_hessian[, , target_index] <- reference[[target_index]]$hessian
  }
  data.frame(
    label = case$label,
    p = case$expected_p,
    penalty_count = case$expected_penalty_count,
    target_count = ncol(Y),
    score_error = max(abs(candidate$score - reference_score)),
    edf_error = max(abs(candidate$edf - reference_edf)),
    gradient_error = max(abs(candidate$gradient - reference_gradient)),
    hessian_error = max(abs(candidate$hessian - reference_hessian)),
    coefficient_error = max(abs(
      candidate$coefficients - reference_coefficients
    )),
    stringsAsFactors = FALSE
  )
})

rows <- do.call(rbind, rows)
print(rows, row.names = FALSE, digits = 17)
assert_true(
  max(rows$score_error) <= 1e-8 && max(rows$edf_error) <= 1e-8 &&
    max(rows$gradient_error) <= 1e-7 &&
    max(rows$hessian_error) <= 2e-6 &&
    max(rows$coefficient_error) <= 1e-8,
  "Phase 6 batched CUDA objective derivatives drifted from Phase 5"
)
cat("PASS Phase 6 multi-penalty CUDA fixed-objective true batch\n")
