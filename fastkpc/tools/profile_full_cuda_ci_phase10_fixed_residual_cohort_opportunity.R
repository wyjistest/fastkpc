source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_shadow.R")
source("fastkpc/R/full_cuda_ci_phase3_artifacts.R")
source("fastkpc/R/full_cuda_ci_phase35_contracts.R")
source("fastkpc/R/full_cuda_ci_phase10_performance_v2.R")
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
source("fastkpc/R/full_cuda_ci_phase8_publication.R")
source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase9_artifact.R")
source("fastkpc/R/full_cuda_ci_phase10_hardening.R")
source("fastkpc/R/full_cuda_ci_phase10_campaign.R")
source("fastkpc/R/full_cuda_ci_phase10_compute_campaign.R")
source(
  "fastkpc/R/full_cuda_ci_phase10_fixed_residual_cohort_opportunity.R"
)

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) < 1L || length(arguments) > 2L) {
  stop(
    paste(
      "usage: profile_full_cuda_ci_phase10_fixed_residual_cohort_opportunity.R",
      "COMPUTE_WARM_EVIDENCE_RDS [OUTPUT_RDS]"
    ),
    call. = FALSE
  )
}
evidence_path <- arguments[[1L]]
output_path <- if (length(arguments) == 2L) arguments[[2L]] else
  fastkpc_full_cuda_phase10_fixed_cohort_default_output()

evidence <- readRDS(evidence_path)
fastkpc_full_cuda_phase10_validate_compute_warm_evidence(evidence)
opportunity <-
  fastkpc_full_cuda_phase10_fixed_residual_cohort_opportunity(
    evidence$result$summary
  )
fastkpc_full_cuda_phase10_write_fixed_residual_cohort_opportunity(
  opportunity, output_path
)

cat(
  "PASS Phase 10 fixed residual cohort opportunity; decision=",
  opportunity$decision,
  " qualified_upper_bound_ms=",
  opportunity$qualified_opportunity$combined_upper_bound_ms,
  " output=", output_path, "\n", sep = ""
)
print(opportunity$categories, row.names = FALSE)
