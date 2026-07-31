source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_shadow.R")
source("fastkpc/R/full_cuda_ci_phase3_artifacts.R")
source("fastkpc/R/full_cuda_ci_phase35_contracts.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_phase35_bakeoff.R")
source("fastkpc/R/full_cuda_ci_single_penalty_gcv.R")
source("fastkpc/R/full_cuda_ci_phase4_artifacts.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cpp.R")
source("fastkpc/R/full_cuda_ci_phase5_artifacts.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cuda.R")
source("fastkpc/R/full_cuda_ci_phase6_artifacts.R")
source("fastkpc/R/full_cuda_ci_native_setup.R")
source("fastkpc/R/full_cuda_ci_phase7_artifacts.R")
source("fastkpc/R/full_cuda_ci_phase7_publication.R")
source("fastkpc/R/full_cuda_ci_phase8_dcov.R")

inputs <- fastkpc_full_cuda_phase8_load_inputs(
  validate_phase7_artifact = FALSE,
  validate_phase7_evidence = FALSE
)
logical <- inputs$corpus$canonical_logical
witness_ids <- c(
  6359L, 6465L, 11978L,
  183015L, 229972L, 233433L
)
selected <- logical$level == 0L |
  logical$logical_sequence_id %in% witness_ids
fixture <- logical[selected, , drop = FALSE]
fixture$near_alpha <-
  abs(fixture$signed_log_ratio_from_alpha) <= log(2)

result <- fastkpc_full_cuda_phase8_run_logical_rows(
  inputs, fixture, progress = FALSE
)
pairs <- result$pairs
diagnostics <- result$diagnostics

fastkpc_full_cuda_phase8_require(
  nrow(pairs) == 2213L + length(witness_ids) &&
    sum(pairs$level == 0L) == 2213L &&
    all(witness_ids %in% pairs$logical_sequence_id) &&
    sum(pairs$screen_decision_flip) > 0L &&
    all(!pairs$screen_decision_flip | pairs$refined) &&
    !any(pairs$final_decision_flip) &&
    all(is.finite(pairs$final_p_value)) &&
    all(diagnostics$residual_d2h_bytes == 0) &&
    all(diagnostics$component_d2h_bytes == 0) &&
    all(diagnostics$cpu_dcov_component_count == 0L) &&
    all(diagnostics$cpu_dcov_eigen_or_lowrank_count == 0L) &&
    all(diagnostics$cpu_dcov_pair_statistic_count == 0L) &&
    all(diagnostics$cpu_gamma_p_value_count == 0L) &&
    all(diagnostics$cpu_spectra_count == 0L) &&
    all(diagnostics$solver_failure_count == 0L) &&
    all(diagnostics$leak_free),
  paste0(
    "Phase 8 focused guarded CUDA dCov gate failed; pairs=", nrow(pairs),
    "; refined=", sum(pairs$refined),
    "; screen_flips=", sum(pairs$screen_decision_flip),
    "; final_flips=", sum(pairs$final_decision_flip),
    "; residual_d2h=", sum(diagnostics$residual_d2h_bytes),
    "; component_d2h=", sum(diagnostics$component_d2h_bytes)
  )
)

cat(
  "PASS Phase 8 guarded CUDA dCov focused gate; pairs=", nrow(pairs),
  " refined=", sum(pairs$refined),
  " screen_flips=", sum(pairs$screen_decision_flip), "\n", sep = ""
)
