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
  cat("SKIP Phase 6 CUDA optimizer: FASTKPC_RUN_CUDA_TESTS != 1\n")
  quit(save = "no", status = 0)
}
if (!isTRUE(tryCatch(fastkpc_cuda_available(), error = function(error) FALSE))) {
  cat("SKIP Phase 6 CUDA optimizer: CUDA unavailable\n")
  quit(save = "no", status = 0)
}

data <- readRDS(file.path(
  "fastkpc", "artifacts", "kpc_tprs_real_zhu",
  "cancer_RD-causalDiscoveryInput.rds"
))
cases <- data.frame(
  label = c(
    "reference", "five-penalty-stability", "indefinite-hessian",
    "rank-deficient", "maximum-iterations", "maximum-score-calls",
    "near-flat-boundary", "near-convergence-score-rounding",
    "formal-partition-log-sp-tolerance",
    "formal-partition-edf-tolerance",
    "formal-partition-dense-boundary-edf-tolerance",
    "formal-partition-single-boundary-primary",
    "formal-partition-high-condition-log-sp-tolerance",
    "formal-partition-boundary-trajectory",
    "formal-partition-legitimate-boundary-a",
    "formal-partition-legitimate-boundary-b",
    "formal-partition-legitimate-boundary-c",
    "formal-partition-legitimate-boundary-d",
    "formal-p8-direct-newton-flat-boundary",
    "formal-p11-ambiguous-tie-replay",
    "formal-p12-ambiguous-descent-halving",
    "formal-p13-delta-identity-boundary",
    "formal-p13-direct-newton-halving",
    "formal-p14-replay-score-guard",
    "formal-p15-rejected-boundary-replay",
    "formal-p15-low-condition-delta-identity",
    "formal-p15-identity-consistent-descent",
    "formal-p7-moderate-condition-descent"
  ),
  shard_id = c(
    1L, 56L, 0L, 37L, 10L, 59L, 10L, 48L, 1L, 36L, 55L, 23L, 7L,
    33L, 1L, 1L, 17L, 37L, 24L, 43L, 28L, 45L, 45L, 46L, 31L,
    47L, 31L, 55L
  ),
  prepared_s_key_sha256 = c(
    "001245052f571033286b2dc7526c24dbe5ec5c221660c094a8b9f052376b91da",
    "e24dde52f78eab81021dc092fb072111f0d27733d6056b5490162039f8e87094",
    "334c0e1a40c98ae302ced2a9573755e48fe19be1d9e7753862a00b620ea71746",
    "2959705cc686141ff4e758d64525e70a211bc845a21ee3085c97d0b5d8abedab",
    "31ff054083fb4a0cdfb0be76305e057a6ab086c134f68a0ba7fbae8661ab2e14",
    "7844c90be2027d3b78f4750e99ca43cb6ee8621e0aca651d2422e3884eb81168",
    "0088c39e91afee87a1a347802db2fb51395e025689e3f13da5a0fe71b71c6bdf",
    "43470d9bf1e9680097ac2b5bf142836e76e88de2660f680c64f4b62bd8bcd3cd",
    "4b5e8d92c4e86cdddeb003da79188d933cf991b521151134a1f85ab1947d4cb1",
    "18b367b21d9ab1f391df7f790af330b1e0770e0db7a83c032cfbeb73645b5b9a",
    "1932f64beca9d12f8c8ec21bc8c3f51d910c2d1c0dceb8a93ad3ec414498b612",
    "f953f56967b02ddc1d93b0c03d6721295f70d11641f420e60583f1e0cb2fe371",
    "1d5a8de575fae279f3fb99e27ba6d9154537d9d20f42fa42f2543a6489b493c7",
    "d19398722be813273a4d9d038b0100d6e425487fadc52091802d1990c0b728f5",
    "2683a26ce86357d2d4706b2ab91e9e6117a4b96cbeff61cc0fab787ed720b088",
    "922330870499c78da8ff876abf363766c571a53536a85b58ed4c829b74d502e1",
    "a7b667e962ff5dd521f4573adfca50b04b443054179da19e9e96523c16706766",
    "ef263c8336fa2dde2c90e8db0a4f5ba9e3086c710e43a3c1536af28e24571eca",
    "01079f3f0d837959a7136b15b32e8b3421f8bffab53e3b7700eacee8700795a9",
    "8f5031d0f6d46d95da3270c798549c4e466469b5da74f312449e1eca7df6e96c",
    "c8522edfd10de23093ae68eed6e2196fffec6daa944ec04c687c3c8e2ba7840e",
    "8757315071ef93776d72fa2cde8a9bc8d514ec18ddf31ec0d091d4f48c6f451a",
    "c8c47b11af8006e5c845c00af69822d72f56e0c718effe23b5b28c33ef5417df",
    "bdcff60a7bb7db1555116105d64fa491f476abcb2fc68b148036e48d87297a00",
    "e931fdb62d19fe6739f096548cf013ddacd60a2d4f4d5d582ba609102625a605",
    "a68f011613627f330caeb8756780b2a8b340375ccea2f03b6fd328b79d4399c5",
    "5f58ebdfee2aed7b65bf9b4e722ef3dcfa097ae9d5d6b44c554d680cd2c711fd",
    "1ece0b46e42a51ecb6b2158e7bd284860bf1b061707239e842426910f59644c5"
  ),
  residual_key_sha256 = c(
    NA_character_,
    "00354ddff81cd49307434189f6bba0fc009f59b6eba9e118bd34b468583cea69",
    "7795b8ddd608d04621c3474df34751044f21077eab6089130662ec93a338fdff",
    "1db147c7e70b187f54304a14982184b7bc5b221d71051abfbfc21c3312399432",
    "8e56ba8371592d8daed48dfa00cac511893d66746ca43f2941954dfa1cf8e81e",
    "d5333f96a4468977ddc61a6526d5a7f255885a9f514f4e990188770e474088d3",
    "8b770202d16c751864c6fa19b51355ace98a070645b916949d825ffcf1327ced",
    "473654d1eb6bc2d4f707c970109139d5417cefa303941838a959663719b29888",
    "ad4d0942fca390d0997052899c1f99c4071414d21493dae5c10126348a012e14",
    "4e8c62a69abbd86d18f2131e33603c5251b3ae5517b41dd4f8208b0c55267523",
    "85d4c492bb6f9752910bdcc580d2959ffc610966fcc317a1254277a26216dbd6",
    "1d78237a7c18c04003d25cd11455512e2d7cc0ba3429b3e7e5c5dd743d145ce3",
    "abcb9d71b53933c115666f031c1a20fdacfe48940fa3d4237849bdfbd5e52cc0",
    "d190d6c386123aa4eb52016bb1d9149027dd1cdd844b3c7219a9c57d163783af",
    "406ff4c472c7ef03668f8d416362a0c7041415579019fb30d88a7493056f1580",
    "bec8345d9c183a753ff5be4e81983b2e655c7ba83efa2b2bb1cb236cbc927c57",
    "b8bfb8f557d7623169f4d76e9a1691541e5a5c017731bef22fe6c3f9f486ee13",
    "51f6eee9ba1ead5075b60d20cab243c4fe2d8e6cebbed977c8ebe61f2d9717e4",
    "442c69affc453227bc3d15e30f75241378c63498fa8f833475b3ea29391ac10b",
    "81d53d84fd0fa13d854d8e1873759af0ccd7770a7ad7478a26b3193f78fc8239",
    "4ad8305edcf7995e18b75e5a1c7c39ba86aa6f597f11128e1cec5810475bfd55",
    "69f3518997be4b03172893055a7031956391439dd8c7e24564e4e86cd53fb2d0",
    "8d4a2fb574d88f3bc0784d3ae393d9572c388a4cf1685479c1886d7237a1f84b",
    "00cbbe34a514a98b974718870623778bad887df96737164e8e275d88b954e52c",
    "3fc46c612c28ff37e64d4adc7433c473510749e0d09812d88e501b2bf032d568",
    "baf88fb40a265e123d3c7007ba00f58df88f06b1d90b871567ad99be0eca1564",
    "e88a3fac1b8da94d36f574fe8d33e45e7bce765dd909861a09b711de7edb349d",
    "3bd59a45c4138f0e5d404628bfb39629af66917a5dcfb01e205d6cdf44ec2f41"
  ),
  stringsAsFactors = FALSE
)

rows <- lapply(seq_len(nrow(cases)), function(case_index) {
  case <- cases[case_index, , drop = FALSE]
  shard <- readRDS(file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1",
    "shards", paste0("shard_", case$shard_id, ".rds")
  ))
  setup <- shard$prepared_s_setups[[case$prepared_s_key_sha256]]
  witness_index <- if (is.na(case$residual_key_sha256)) {
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
    !is.null(setup) && !is.na(witness_index),
    paste0("Phase 6 optimizer witness is missing: ", case$label)
  )
  state_indices <- unique(c(
    witness_index,
    which(
      shard$target_states$prepared_s_key_sha256 ==
        case$prepared_s_key_sha256
    )
  ))
  state_indices <- head(state_indices, 2L)
  if (length(state_indices) == 1L) state_indices <- rep(state_indices, 2L)
  contexts <- lapply(state_indices, function(index) {
    state <- shard$target_states[index, , drop = FALSE]
    target <- fastkpc_full_cuda_materialize_target_state(
      state, data, setup$dataset_sha256
    )
    fastkpc_full_cuda_validate_materialized_target_for_prepared(setup, target)
  })
  Y <- do.call(cbind, lapply(contexts, `[[`, "y"))
  prepared <- fastkpc_full_cuda_phase6_prepare(setup)
  candidate <- fastkpc_full_cuda_phase6_optimize_cuda(prepared, Y)
  reference <- lapply(seq_len(ncol(Y)), function(target) {
    fastkpc_full_cuda_phase5_optimize_cpp(setup, Y[, target])
  })
  reference_log_sp <- do.call(cbind, lapply(
    reference, `[[`, "selected_log_sp"
  ))
  reference_score <- vapply(reference, `[[`, numeric(1L), "score")
  reference_edf <- vapply(reference, `[[`, numeric(1L), "edf")
  reference_iterations <- vapply(
    reference, `[[`, integer(1L), "optimizer_iterations"
  )
  reference_score_calls <- vapply(
    reference, `[[`, integer(1L), "score_calls"
  )
  reference_objective_calls <- vapply(
    reference, `[[`, integer(1L), "objective_calls"
  )
  reference_step_halving <- vapply(
    reference, `[[`, integer(1L), "step_halving_count"
  )
  reference_boundary_probes <- vapply(
    reference, `[[`, integer(1L), "boundary_probe_count"
  )
  reference_boundary_accepted <- vapply(
    reference, `[[`, integer(1L), "boundary_accepted_count"
  )
  reference_boundary_status <- do.call(cbind, lapply(
    reference, `[[`, "boundary_status"
  ))
  reference_hessian_positive <- vapply(
    reference, `[[`, logical(1L), "hessian_positive_definite"
  )
  reference_fully_converged <- vapply(
    reference, `[[`, logical(1L), "fully_converged"
  )
  reference_residuals <- do.call(cbind, lapply(reference, `[[`, "residuals"))
  candidate_residuals <- Y - prepared$X %*% candidate$coefficients
  diagnostics <- candidate$diagnostics
  replay_discarded_evaluations <-
    diagnostics$cuda_stability_replay_discarded_complete_evaluation_count +
      diagnostics$cuda_stability_replay_discarded_score_only_evaluation_count
  replay_discarded_factorizations <-
    diagnostics$cuda_stability_replay_discarded_guarded_qr_evaluation_count +
      diagnostics$cuda_stability_replay_discarded_stable_svd_evaluation_count
  confirmation_evaluations <- diagnostics[[
    "cuda_terminal_boundary_confirmation_complete_evaluation_count"
  ]]
  confirmation_factorizations <- diagnostics[[
    "cuda_terminal_boundary_confirmation_stable_svd_evaluation_count"
  ]]
  physical_evaluations <-
    diagnostics$cuda_complete_evaluation_count +
      diagnostics$cuda_score_only_evaluation_count +
      replay_discarded_evaluations + confirmation_evaluations
  physical_factorizations <-
    diagnostics$cuda_guarded_qr_evaluation_count +
      diagnostics$cuda_stable_svd_evaluation_count +
      replay_discarded_factorizations + confirmation_factorizations
  replay_reason_counts <- c(
    diagnostics$cuda_stability_replay_long_trajectory_reason_count,
    diagnostics$cuda_stability_replay_dense_boundary_reason_count,
    diagnostics$cuda_stability_replay_high_condition_reason_count,
    diagnostics$cuda_stability_replay_ambiguous_step_reason_count,
    diagnostics$cuda_stability_replay_rejected_boundary_reason_count,
    diagnostics$cuda_stability_replay_direct_newton_reason_count
  )
  assert_true(
    identical(
      candidate$schema_version,
      "full-cuda-ci-multi-penalty-gcv-cuda-optimization-v1"
    ) && identical(
      candidate$rank_path,
      "cuda-dpstf2-guarded-qr-lapack-3.12-dgesdd"
    ) && all(candidate$optimizer_status == 0L) &&
      identical(
        diagnostics$execution_strategy,
        "one-setup-one-block-per-target-independent-optimizer"
      ) && diagnostics$cuda_optimizer_kernel_launch_count == 1L &&
      diagnostics$cuda_optimizer_target_count == ncol(Y) &&
      diagnostics$cuda_selected_fit_count == ncol(Y) &&
      diagnostics$cuda_optimizer_objective_count ==
        sum(candidate$objective_calls) &&
      diagnostics$cuda_complete_evaluation_count +
        diagnostics$cuda_score_only_evaluation_count +
        diagnostics$cuda_selected_evaluation_reuse_count ==
          diagnostics$cuda_optimizer_objective_count &&
      diagnostics$cuda_score_only_evaluation_count ==
        sum(candidate$boundary_probe_count) &&
      diagnostics$cuda_selected_evaluation_reuse_count ==
        sum(candidate$boundary_accepted_count == 0L) &&
      diagnostics$cuda_guarded_qr_evaluation_count +
        diagnostics$cuda_stable_svd_evaluation_count ==
          diagnostics$cuda_complete_evaluation_count +
            diagnostics$cuda_score_only_evaluation_count &&
      diagnostics$cuda_stability_replay_kernel_launch_count == 1L &&
      diagnostics$cuda_stability_merge_kernel_launch_count == 1L &&
      diagnostics$cuda_stability_replay_target_count >=
        diagnostics$cuda_stability_replay_selected_count &&
      diagnostics$cuda_stability_replay_screened_count >= 0L &&
      diagnostics$cuda_stability_replay_screened_count <=
        diagnostics$cuda_stability_replay_target_count &&
      diagnostics$cuda_stability_replay_screened_count +
        diagnostics$cuda_stability_replay_selected_count <=
          diagnostics$cuda_stability_replay_target_count &&
      diagnostics$cuda_stability_replay_error_count == 0L &&
      all(replay_reason_counts >= 0L) &&
      all(replay_reason_counts <=
            diagnostics$cuda_stability_replay_target_count) &&
      sum(replay_reason_counts) >=
        diagnostics$cuda_stability_replay_target_count &&
      diagnostics$cuda_stability_replay_dense_score_guard_rejected_count >=
        0L &&
      diagnostics$cuda_stability_replay_dense_score_guard_rejected_count <=
        diagnostics$cuda_stability_replay_dense_boundary_reason_count &&
      diagnostics$cuda_stability_replay_extrapolation_target_count >= 0L &&
      diagnostics$cuda_stability_replay_extrapolation_target_count <=
        diagnostics$cuda_stability_replay_selected_count &&
      is.finite(diagnostics$cuda_stability_replay_max_extrapolation) &&
      diagnostics$cuda_stability_replay_max_extrapolation >= 0 &&
      diagnostics$cuda_stability_replay_max_extrapolation <= 2e-5 &&
      replay_discarded_evaluations == replay_discarded_factorizations &&
      diagnostics$cuda_terminal_boundary_confirmation_accepted_count +
        diagnostics$cuda_terminal_boundary_confirmation_rejected_count ==
          diagnostics$cuda_terminal_boundary_confirmation_count &&
      diagnostics[[
        "cuda_terminal_boundary_confirmation_strong_delta_accepted_count"
      ]] + diagnostics[[
        "cuda_terminal_boundary_confirmation_identity_tie_accepted_count"
      ]] + diagnostics[[
        "cuda_terminal_boundary_confirmation_delta_identity_accepted_count"
      ]] == diagnostics$cuda_terminal_boundary_confirmation_accepted_count &&
      confirmation_evaluations ==
        2L * diagnostics$cuda_terminal_boundary_confirmation_count &&
      confirmation_factorizations == confirmation_evaluations &&
      is.finite(diagnostics[[
        "cuda_terminal_boundary_confirmation_max_delta_disagreement"
      ]]) && diagnostics[[
        "cuda_terminal_boundary_confirmation_max_delta_disagreement"
      ]] >= 0 &&
      is.finite(diagnostics[[
        "cuda_terminal_boundary_confirmation_max_delta_ratio"
      ]]) && diagnostics[[
        "cuda_terminal_boundary_confirmation_max_delta_ratio"
      ]] >= 0 &&
      physical_evaluations == physical_factorizations &&
      diagnostics$cuda_guarded_qr_evaluation_count > 0L &&
      diagnostics$cuda_penalty_factor_augmentation_cycles > 0 &&
      diagnostics$cuda_qr_svd_cycles > 0 &&
      diagnostics$cuda_qr_bidiagonal_reduction_cycles > 0 &&
      (diagnostics$cuda_stable_svd_evaluation_count == 0L || (
        diagnostics$cuda_bidiagonal_svd_cycles > 0 &&
          diagnostics$cuda_svd_vector_postback_cycles > 0 &&
          diagnostics$cuda_left_vector_product_cycles > 0
      )) &&
      diagnostics$cuda_score_construction_cycles > 0 &&
      diagnostics$cuda_derivative_hessian_cycles > 0 &&
      diagnostics$cpu_objective_count == 0L &&
      diagnostics$cpu_optimizer_count == 0L &&
      diagnostics$cpu_multi_penalty_solve_count == 0L &&
      diagnostics$fallback_count == 0L &&
      diagnostics$cuda_error_count == 0L &&
      isTRUE(diagnostics$independent_target_states) &&
      isTRUE(diagnostics$true_batched_kernel) &&
      !isTRUE(diagnostics$normal_equations_used),
    paste0("Phase 6 CUDA optimizer failed: ", case$label)
  )
  assert_true(
    identical(candidate$optimizer_iterations, reference_iterations) &&
      identical(candidate$score_calls, reference_score_calls) &&
      identical(candidate$objective_calls, reference_objective_calls) &&
      identical(candidate$step_halving_count, reference_step_halving) &&
      identical(candidate$boundary_probe_count, reference_boundary_probes) &&
      identical(
        candidate$boundary_accepted_count, reference_boundary_accepted
      ) && identical(
        candidate$boundary_status, reference_boundary_status
      ) &&
      identical(
        candidate$hessian_positive_definite,
        reference_hessian_positive
      ) && identical(
        candidate$fully_converged, reference_fully_converged
      ),
    paste0("Phase 6 CUDA optimizer trajectory drifted: ", case$label)
  )
  assert_true(
    max(abs(candidate$selected_log_sp - reference_log_sp)) <= 1e-6 &&
      max(abs(candidate$score - reference_score)) <= 1e-8 &&
      max(abs(candidate$edf - reference_edf)) <= 1e-8 &&
      max(abs(candidate_residuals - reference_residuals)) <= 1e-7,
    paste0("Phase 6 CUDA optimizer selected fit drifted: ", case$label)
  )
  decomposition_cycles <- sum(c(
    diagnostics$cuda_qr_bidiagonal_reduction_cycles,
    diagnostics$cuda_bidiagonal_svd_cycles,
    diagnostics$cuda_svd_vector_postback_cycles,
    diagnostics$cuda_left_vector_product_cycles
  ))
  data.frame(
    label = case$label,
    coefficient_dim = ncol(setup$X),
    penalty_count = length(setup$penalty_blocks),
    target_count = ncol(Y),
    max_log_sp_error = max(abs(
      candidate$selected_log_sp - reference_log_sp
    )),
    max_score_error = max(abs(candidate$score - reference_score)),
    max_edf_error = max(abs(candidate$edf - reference_edf)),
    max_iteration_delta = max(abs(
      candidate$optimizer_iterations - reference_iterations
    )),
    max_score_call_delta = max(abs(
      candidate$score_calls - reference_score_calls
    )),
    witness_optimizer_iterations = candidate$optimizer_iterations[[1L]],
    witness_step_halving_count = candidate$step_halving_count[[1L]],
    witness_boundary_accepted_count =
      candidate$boundary_accepted_count[[1L]],
    qr_reduction_share =
      diagnostics$cuda_qr_bidiagonal_reduction_cycles /
        decomposition_cycles,
    bidiagonal_svd_share =
      diagnostics$cuda_bidiagonal_svd_cycles / decomposition_cycles,
    vector_postback_share =
      diagnostics$cuda_svd_vector_postback_cycles / decomposition_cycles,
    left_product_share =
      diagnostics$cuda_left_vector_product_cycles / decomposition_cycles,
    guarded_qr_evaluation_count =
      diagnostics$cuda_guarded_qr_evaluation_count,
    stable_svd_evaluation_count =
      diagnostics$cuda_stable_svd_evaluation_count,
    stability_replay_target_count =
      diagnostics$cuda_stability_replay_target_count,
    stability_replay_screened_count =
      diagnostics$cuda_stability_replay_screened_count,
    stability_replay_selected_count =
      diagnostics$cuda_stability_replay_selected_count,
    stability_replay_error_count =
      diagnostics$cuda_stability_replay_error_count,
    stability_replay_long_trajectory_reason_count =
      diagnostics$cuda_stability_replay_long_trajectory_reason_count,
    stability_replay_dense_boundary_reason_count =
      diagnostics$cuda_stability_replay_dense_boundary_reason_count,
    stability_replay_high_condition_reason_count =
      diagnostics$cuda_stability_replay_high_condition_reason_count,
    stability_replay_ambiguous_step_reason_count =
      diagnostics$cuda_stability_replay_ambiguous_step_reason_count,
    stability_replay_rejected_boundary_reason_count =
      diagnostics$cuda_stability_replay_rejected_boundary_reason_count,
    stability_replay_direct_newton_reason_count =
      diagnostics$cuda_stability_replay_direct_newton_reason_count,
    stability_replay_dense_score_guard_rejected_count =
      diagnostics$cuda_stability_replay_dense_score_guard_rejected_count,
    stability_replay_extrapolation_target_count =
      diagnostics$cuda_stability_replay_extrapolation_target_count,
    stability_replay_max_log_sp_spread =
      diagnostics$cuda_stability_replay_max_log_sp_spread,
    stability_replay_max_extrapolation =
      diagnostics$cuda_stability_replay_max_extrapolation,
    stability_replay_discarded_evaluation_count =
      diagnostics$cuda_stability_replay_discarded_complete_evaluation_count +
        diagnostics[[
          "cuda_stability_replay_discarded_score_only_evaluation_count"
        ]],
    terminal_boundary_confirmation_count =
      diagnostics$cuda_terminal_boundary_confirmation_count,
    terminal_boundary_confirmation_accepted_count =
      diagnostics$cuda_terminal_boundary_confirmation_accepted_count,
    terminal_boundary_confirmation_rejected_count =
      diagnostics$cuda_terminal_boundary_confirmation_rejected_count,
    terminal_boundary_confirmation_strong_delta_accepted_count =
      diagnostics[[
        "cuda_terminal_boundary_confirmation_strong_delta_accepted_count"
      ]],
    terminal_boundary_confirmation_identity_tie_accepted_count =
      diagnostics[[
        "cuda_terminal_boundary_confirmation_identity_tie_accepted_count"
      ]],
    terminal_boundary_confirmation_delta_identity_accepted_count =
      diagnostics[[
        "cuda_terminal_boundary_confirmation_delta_identity_accepted_count"
      ]],
    terminal_boundary_confirmation_complete_evaluation_count =
      diagnostics[[
        "cuda_terminal_boundary_confirmation_complete_evaluation_count"
      ]],
    terminal_boundary_confirmation_stable_svd_evaluation_count =
      diagnostics[[
        "cuda_terminal_boundary_confirmation_stable_svd_evaluation_count"
      ]],
    terminal_boundary_confirmation_cycles =
      diagnostics$cuda_terminal_boundary_confirmation_cycles,
    terminal_boundary_confirmation_max_identity_disagreement =
      diagnostics[[
        "cuda_terminal_boundary_confirmation_max_identity_disagreement"
      ]],
    terminal_boundary_confirmation_max_identity_ratio =
      diagnostics$cuda_terminal_boundary_confirmation_max_identity_ratio,
    terminal_boundary_confirmation_max_delta_disagreement =
      diagnostics[[
        "cuda_terminal_boundary_confirmation_max_delta_disagreement"
      ]],
    terminal_boundary_confirmation_max_delta_ratio =
      diagnostics$cuda_terminal_boundary_confirmation_max_delta_ratio,
    max_residual_error = max(abs(
      candidate_residuals - reference_residuals
    )),
    stringsAsFactors = FALSE
  )
})

rows <- do.call(rbind, rows)
assert_true(
  sum(rows$guarded_qr_evaluation_count) > 0L &&
    sum(rows$stable_svd_evaluation_count) > 0L,
  "Phase 6 guarded QR/SVD optimizer route coverage is incomplete"
)
stability_row <- rows[
  rows$label == "formal-partition-log-sp-tolerance", , drop = FALSE
]
edf_stability_row <- rows[
  rows$label == "formal-partition-edf-tolerance", , drop = FALSE
]
dense_boundary_edf_row <- rows[
  rows$label == "formal-partition-dense-boundary-edf-tolerance",
  , drop = FALSE
]
single_boundary_primary_row <- rows[
  rows$label == "formal-partition-single-boundary-primary",
  , drop = FALSE
]
high_condition_stability_row <- rows[
  rows$label == "formal-partition-high-condition-log-sp-tolerance",
  , drop = FALSE
]
boundary_row <- rows[
  rows$label == "formal-partition-boundary-trajectory", , drop = FALSE
]
legitimate_boundary_rows <- rows[
  startsWith(rows$label, "formal-partition-legitimate-boundary-"),
  , drop = FALSE
]
identity_tie_boundary_row <- rows[
  rows$label == "formal-partition-legitimate-boundary-a", , drop = FALSE
]
strong_delta_boundary_row <- rows[
  rows$label == "formal-partition-legitimate-boundary-d", , drop = FALSE
]
direct_flat_boundary_row <- rows[
  rows$label == "formal-p8-direct-newton-flat-boundary", , drop = FALSE
]
ambiguous_tie_row <- rows[
  rows$label == "formal-p11-ambiguous-tie-replay", , drop = FALSE
]
ambiguous_descent_row <- rows[
  rows$label == "formal-p12-ambiguous-descent-halving", , drop = FALSE
]
delta_identity_boundary_row <- rows[
  rows$label == "formal-p13-delta-identity-boundary", , drop = FALSE
]
direct_halving_row <- rows[
  rows$label == "formal-p13-direct-newton-halving", , drop = FALSE
]
replay_score_guard_row <- rows[
  rows$label == "formal-p14-replay-score-guard", , drop = FALSE
]
rejected_boundary_replay_row <- rows[
  rows$label == "formal-p15-rejected-boundary-replay", , drop = FALSE
]
low_condition_delta_identity_row <- rows[
  rows$label == "formal-p15-low-condition-delta-identity", , drop = FALSE
]
identity_consistent_descent_row <- rows[
  rows$label == "formal-p15-identity-consistent-descent", , drop = FALSE
]
moderate_condition_descent_row <- rows[
  rows$label == "formal-p7-moderate-condition-descent", , drop = FALSE
]
assert_true(
  nrow(stability_row) == 1L &&
    stability_row$stability_replay_target_count >= 1L &&
    stability_row$stability_replay_selected_count >= 1L &&
    stability_row$stability_replay_long_trajectory_reason_count >= 1L &&
    stability_row$stability_replay_error_count == 0L &&
    stability_row$stability_replay_max_log_sp_spread > 1e-7 &&
    stability_row$stability_replay_discarded_evaluation_count > 0L,
  "Phase 6 long-trajectory stability replay coverage is incomplete"
)
assert_true(
  nrow(edf_stability_row) == 1L &&
    edf_stability_row$stability_replay_target_count >= 1L &&
    edf_stability_row$stability_replay_selected_count >= 1L &&
    edf_stability_row$stability_replay_error_count == 0L &&
    edf_stability_row$stability_replay_max_log_sp_spread > 5e-8 &&
    edf_stability_row$max_edf_error <= 1e-8,
  "Phase 6 boundary-risk stability replay coverage is incomplete"
)
assert_true(
  nrow(dense_boundary_edf_row) == 1L &&
    dense_boundary_edf_row$stability_replay_target_count >= 1L &&
    dense_boundary_edf_row$stability_replay_selected_count >= 1L &&
    dense_boundary_edf_row$stability_replay_dense_boundary_reason_count >= 1L &&
    dense_boundary_edf_row$stability_replay_error_count == 0L &&
    dense_boundary_edf_row$stability_replay_extrapolation_target_count >= 1L &&
    dense_boundary_edf_row$stability_replay_max_extrapolation > 0 &&
    dense_boundary_edf_row$stability_replay_max_extrapolation <= 4e-6 &&
    dense_boundary_edf_row$stability_replay_max_log_sp_spread > 3e-8 &&
    dense_boundary_edf_row$stability_replay_max_log_sp_spread <= 5e-8 &&
    dense_boundary_edf_row$max_log_sp_error <= 1e-6 &&
    dense_boundary_edf_row$max_edf_error <= 1e-8,
  "Phase 6 dense-boundary EDF replay coverage is incomplete"
)
assert_true(
  nrow(single_boundary_primary_row) == 1L &&
    single_boundary_primary_row$witness_optimizer_iterations >= 25L &&
    single_boundary_primary_row$witness_optimizer_iterations <= 100L &&
    single_boundary_primary_row$witness_step_halving_count * 4L >=
      single_boundary_primary_row$witness_optimizer_iterations * 9L &&
    single_boundary_primary_row$witness_boundary_accepted_count == 1L &&
    single_boundary_primary_row$stability_replay_target_count == 0L &&
    single_boundary_primary_row$stability_replay_selected_count == 0L &&
    single_boundary_primary_row$stability_replay_extrapolation_target_count ==
      0L &&
    single_boundary_primary_row$stability_replay_max_log_sp_spread == 0 &&
    single_boundary_primary_row$stability_replay_max_extrapolation == 0 &&
    single_boundary_primary_row$max_log_sp_error <= 1e-6 &&
    single_boundary_primary_row$max_edf_error <= 1e-8,
  "Phase 6 single-boundary primary-path coverage is incomplete"
)
assert_true(
  nrow(high_condition_stability_row) == 1L &&
    high_condition_stability_row$stability_replay_target_count >= 1L &&
    high_condition_stability_row$stability_replay_selected_count >= 1L &&
    high_condition_stability_row[[
      "stability_replay_high_condition_reason_count"
    ]] >= 1L &&
    high_condition_stability_row$stability_replay_error_count == 0L &&
    high_condition_stability_row$stability_replay_extrapolation_target_count >=
      1L &&
    high_condition_stability_row$stability_replay_max_extrapolation > 0 &&
    high_condition_stability_row$stability_replay_max_extrapolation <= 4e-6 &&
    high_condition_stability_row$stability_replay_max_log_sp_spread > 5e-8 &&
    high_condition_stability_row$max_log_sp_error <= 1e-6,
  "Phase 6 high-condition boundary replay coverage is incomplete"
)
assert_true(
  nrow(boundary_row) == 1L &&
    boundary_row$terminal_boundary_confirmation_count >= 1L &&
    boundary_row$terminal_boundary_confirmation_rejected_count >= 1L &&
    boundary_row[[
      "terminal_boundary_confirmation_strong_delta_accepted_count"
    ]] == 0L &&
    boundary_row[[
      "terminal_boundary_confirmation_identity_tie_accepted_count"
    ]] == 0L &&
    boundary_row$terminal_boundary_confirmation_complete_evaluation_count ==
      2L * boundary_row$terminal_boundary_confirmation_count &&
    boundary_row$terminal_boundary_confirmation_stable_svd_evaluation_count ==
      boundary_row$terminal_boundary_confirmation_complete_evaluation_count &&
    boundary_row$terminal_boundary_confirmation_cycles > 0 &&
    boundary_row$terminal_boundary_confirmation_max_identity_disagreement > 0 &&
    boundary_row$terminal_boundary_confirmation_max_identity_ratio > 1 &&
    boundary_row$terminal_boundary_confirmation_max_delta_disagreement > 0 &&
    boundary_row$terminal_boundary_confirmation_max_delta_ratio > 1,
  "Phase 6 rejected terminal boundary confirmation coverage is incomplete"
)
assert_true(
  nrow(legitimate_boundary_rows) == 4L &&
    all(legitimate_boundary_rows$terminal_boundary_confirmation_count >= 1L) &&
    all(
      legitimate_boundary_rows$terminal_boundary_confirmation_accepted_count >=
        1L
    ) &&
    all(
      legitimate_boundary_rows$terminal_boundary_confirmation_rejected_count ==
        0L
    ) &&
    all(
      legitimate_boundary_rows[[
        "terminal_boundary_confirmation_strong_delta_accepted_count"
      ]] + legitimate_boundary_rows[[
        "terminal_boundary_confirmation_identity_tie_accepted_count"
      ]] + legitimate_boundary_rows[[
        "terminal_boundary_confirmation_delta_identity_accepted_count"
      ]] == legitimate_boundary_rows[[
        "terminal_boundary_confirmation_accepted_count"
      ]]
    ) &&
    all(
      legitimate_boundary_rows[[
        "terminal_boundary_confirmation_complete_evaluation_count"
      ]] == 2L * legitimate_boundary_rows$terminal_boundary_confirmation_count
    ) &&
    all(
      legitimate_boundary_rows[[
        "terminal_boundary_confirmation_stable_svd_evaluation_count"
      ]] == legitimate_boundary_rows[[
        "terminal_boundary_confirmation_complete_evaluation_count"
      ]]
    ) &&
    all(legitimate_boundary_rows$terminal_boundary_confirmation_cycles > 0) &&
    all(
      legitimate_boundary_rows[[
        "terminal_boundary_confirmation_max_identity_disagreement"
      ]] > 0
    ) &&
    all(
      legitimate_boundary_rows[[
        "terminal_boundary_confirmation_max_delta_disagreement"
      ]] > 0
    ) &&
    nrow(identity_tie_boundary_row) == 1L &&
    identity_tie_boundary_row[[
      "terminal_boundary_confirmation_identity_tie_accepted_count"
    ]] >= 1L &&
    identity_tie_boundary_row$terminal_boundary_confirmation_max_identity_ratio <=
      1 &&
    nrow(strong_delta_boundary_row) == 1L &&
    strong_delta_boundary_row[[
      "terminal_boundary_confirmation_strong_delta_accepted_count"
    ]] >= 1L &&
    strong_delta_boundary_row$terminal_boundary_confirmation_max_identity_ratio >
      1 &&
    strong_delta_boundary_row$terminal_boundary_confirmation_max_delta_ratio >
      1 &&
    all(legitimate_boundary_rows$stability_replay_error_count == 0L),
  "Phase 6 accepted terminal boundary confirmation coverage is incomplete"
)
assert_true(
  nrow(direct_flat_boundary_row) == 1L &&
    direct_flat_boundary_row$stability_replay_target_count >= 1L &&
    direct_flat_boundary_row$stability_replay_selected_count >= 1L &&
    direct_flat_boundary_row$stability_replay_direct_newton_reason_count >= 1L &&
    direct_flat_boundary_row$stability_replay_extrapolation_target_count ==
      0L &&
    direct_flat_boundary_row$stability_replay_max_log_sp_spread > 1e-5 &&
    direct_flat_boundary_row$max_log_sp_error <= 1e-6 &&
    direct_flat_boundary_row$max_edf_error <= 1e-8,
  "Phase 6 direct-Newton flat-boundary replay coverage is incomplete"
)
assert_true(
  nrow(ambiguous_tie_row) == 1L &&
    ambiguous_tie_row$stability_replay_target_count >= 1L &&
    ambiguous_tie_row$stability_replay_selected_count >= 1L &&
    ambiguous_tie_row$stability_replay_ambiguous_step_reason_count >= 1L &&
    ambiguous_tie_row$max_log_sp_error <= 1e-6 &&
    nrow(ambiguous_descent_row) == 1L &&
    ambiguous_descent_row$stability_replay_target_count >= 1L &&
    ambiguous_descent_row$stability_replay_selected_count >= 1L &&
    ambiguous_descent_row$stability_replay_ambiguous_step_reason_count >= 1L &&
    ambiguous_descent_row$max_log_sp_error <= 1e-6,
  "Phase 6 ambiguous first-step replay coverage is incomplete"
)
assert_true(
  nrow(delta_identity_boundary_row) == 1L &&
    delta_identity_boundary_row$terminal_boundary_confirmation_count >= 1L &&
    delta_identity_boundary_row[[
      "terminal_boundary_confirmation_accepted_count"
    ]] >= 1L &&
    delta_identity_boundary_row[[
      "terminal_boundary_confirmation_rejected_count"
    ]] == 0L &&
    delta_identity_boundary_row[[
      "terminal_boundary_confirmation_delta_identity_accepted_count"
    ]] >= 1L &&
    delta_identity_boundary_row$max_log_sp_error <= 1e-6 &&
    nrow(low_condition_delta_identity_row) == 1L &&
    low_condition_delta_identity_row[[
      "terminal_boundary_confirmation_accepted_count"
    ]] >= 1L &&
    low_condition_delta_identity_row[[
      "terminal_boundary_confirmation_delta_identity_accepted_count"
    ]] >= 1L &&
    low_condition_delta_identity_row$max_log_sp_error <= 1e-6,
  "Phase 6 delta-identity terminal confirmation coverage is incomplete"
)
assert_true(
  nrow(direct_halving_row) == 1L &&
    direct_halving_row$stability_replay_target_count >= 1L &&
    direct_halving_row$stability_replay_selected_count >= 1L &&
    direct_halving_row$stability_replay_direct_newton_reason_count >= 1L &&
    direct_halving_row$stability_replay_extrapolation_target_count >= 1L &&
    direct_halving_row$stability_replay_max_extrapolation > 4e-6 &&
    direct_halving_row$stability_replay_max_extrapolation <= 2e-5 &&
    direct_halving_row$max_log_sp_error <= 1e-6 &&
    direct_halving_row$max_edf_error <= 1e-8,
  "Phase 6 direct-Newton halving refinement coverage is incomplete"
)
assert_true(
  nrow(replay_score_guard_row) == 1L &&
    replay_score_guard_row$stability_replay_target_count >= 1L &&
    replay_score_guard_row$stability_replay_screened_count >= 1L &&
    replay_score_guard_row$stability_replay_selected_count == 0L &&
    replay_score_guard_row$stability_replay_dense_boundary_reason_count >= 1L &&
    replay_score_guard_row[[
      "stability_replay_dense_score_guard_rejected_count"
    ]] == 0L &&
    replay_score_guard_row$stability_replay_extrapolation_target_count == 0L &&
    replay_score_guard_row$max_log_sp_error <= 1e-6 &&
    replay_score_guard_row$max_edf_error <= 1e-8,
  "Phase 6 replay risk-screen coverage is incomplete"
)
assert_true(
  nrow(rejected_boundary_replay_row) == 1L &&
    rejected_boundary_replay_row[[
      "terminal_boundary_confirmation_rejected_count"
    ]] >= 1L &&
    rejected_boundary_replay_row$stability_replay_target_count >= 1L &&
    rejected_boundary_replay_row[[
      "stability_replay_rejected_boundary_reason_count"
    ]] >= 1L &&
    rejected_boundary_replay_row$stability_replay_selected_count >= 1L &&
    rejected_boundary_replay_row$max_log_sp_error <= 1e-6,
  "Phase 6 rejected-boundary stability replay coverage is incomplete"
)
assert_true(
  nrow(identity_consistent_descent_row) == 1L &&
    identity_consistent_descent_row$stability_replay_target_count == 0L &&
    identity_consistent_descent_row$stability_replay_selected_count == 0L &&
    identity_consistent_descent_row$max_log_sp_error <= 1e-6 &&
    identity_consistent_descent_row$max_edf_error <= 1e-8,
  "Phase 6 identity-consistent ambiguous descent coverage is incomplete"
)
assert_true(
  nrow(moderate_condition_descent_row) == 1L &&
    moderate_condition_descent_row$stability_replay_target_count == 0L &&
    moderate_condition_descent_row$stability_replay_selected_count == 0L &&
    moderate_condition_descent_row$max_log_sp_error <= 1e-6 &&
    moderate_condition_descent_row$max_edf_error <= 1e-8,
  "Phase 6 moderate-condition ambiguous descent coverage is incomplete"
)
print(rows, row.names = FALSE, digits = 17)
cat("PASS Phase 6 CUDA independent-target optimizer smoke\n")
