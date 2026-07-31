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

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) < 1L || length(arguments) > 3L) {
  stop(
    paste(
      "usage: run_full_cuda_ci_phase10_campaign.R",
      "prepare|publish|validate [STAGING_DIR] [OUTPUT_DIR]"
    ),
    call. = FALSE
  )
}
action <- match.arg(arguments[[1L]], c("prepare", "publish", "validate"))
staging_dir <- if (length(arguments) >= 2L) arguments[[2L]] else
  Sys.getenv(
    "FASTKPC_PHASE10_CAMPAIGN_STAGING_DIR",
    unset = fastkpc_full_cuda_phase10_campaign_staging_dir()
  )
output_dir <- if (length(arguments) >= 3L) arguments[[3L]] else
  Sys.getenv(
    "FASTKPC_PHASE10_CAMPAIGN_ARTIFACT_DIR",
    unset = fastkpc_full_cuda_phase10_campaign_artifact_dir()
  )

if (identical(action, "prepare")) {
  freeze <- fastkpc_full_cuda_phase10_campaign_prepare_staging(staging_dir)
  cat(
    "PASS Phase 10 campaign freeze: ", freeze$freeze_identity_sha256,
    " staging=", staging_dir, "\n", sep = ""
  )
} else if (identical(action, "publish")) {
  artifact <- fastkpc_full_cuda_phase10_publish_campaign(
    staging_dir = staging_dir, output_dir = output_dir
  )
  cat(
    "PASS Phase 10 canonical campaign artifact: ", output_dir,
    " producer=", artifact$producer$identity_sha256,
    " warm_median_sec=", artifact$summary$warm_median_sec,
    " baseline_ratio=", artifact$summary$candidate_to_baseline_ratio,
    "\n", sep = ""
  )
} else {
  artifact <- fastkpc_full_cuda_phase10_validate_campaign_artifact(
    output_dir, verify_current_sources = TRUE
  )
  cat(
    "PASS Phase 10 canonical campaign validation: ", output_dir,
    " producer=", artifact$producer$identity_sha256, "\n", sep = ""
  )
}
