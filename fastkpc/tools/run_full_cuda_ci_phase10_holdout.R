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
source("fastkpc/R/full_cuda_ci_phase8_publication.R")
source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase9_artifact.R")
source("fastkpc/R/full_cuda_ci_phase10_hardening.R")
source("fastkpc/R/full_cuda_ci_phase10_campaign.R")
source("fastkpc/R/full_cuda_ci_phase10_holdout.R")

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) != 1L) {
  stop(
    "usage: run_full_cuda_ci_phase10_holdout.R release|validate",
    call. = FALSE
  )
}
action <- match.arg(arguments[[1L]], c("release", "validate"))
if (identical(action, "release")) {
  artifact <- fastkpc_full_cuda_phase10_release_holdout()
  cat(
    "PASS Phase 10 sealed holdout release: ",
    fastkpc_full_cuda_phase10_holdout_artifact_dir(),
    " producer=", artifact$producer$identity_sha256,
    " cases=", artifact$summary$case_count,
    " logical_tests=", artifact$summary$logical_test_count,
    "\n", sep = ""
  )
} else {
  artifact <- fastkpc_full_cuda_phase10_validate_holdout_artifact(
    verify_current_sources = TRUE
  )
  cat(
    "PASS Phase 10 sealed holdout validation: ",
    fastkpc_full_cuda_phase10_holdout_artifact_dir(),
    " producer=", artifact$producer$identity_sha256, "\n", sep = ""
  )
}
