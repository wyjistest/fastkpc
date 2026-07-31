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

evidence_path <- Sys.getenv("FASTKPC_PHASE9_EVIDENCE_RDS", unset = "")
output_dir <- Sys.getenv(
  "FASTKPC_PHASE9_ARTIFACT_DIR",
  unset = fastkpc_full_cuda_phase9_artifact_dir()
)
data_path <- Sys.getenv(
  "FASTKPC_FULL_CUDA_CI_DATA_PATH",
  unset = file.path(
    "fastkpc", "artifacts", "kpc_tprs_real_zhu",
    "cancer_RD-causalDiscoveryInput.rds"
  )
)

raw <- if (nzchar(evidence_path)) {
  if (!file.exists(evidence_path)) {
    stop("Phase 9 evidence input does not exist: ", evidence_path,
         call. = FALSE)
  }
  readRDS(evidence_path)
} else {
  fastkpc_full_cuda_phase9_capture(data_path)
}

artifact <- fastkpc_full_cuda_phase9_publish(
  raw = raw,
  output_dir = output_dir,
  data_path = data_path
)
cat(
  "PASS Phase 9 one-call artifact: ", output_dir,
  " producer=", artifact$producer$identity_sha256,
  " elapsed_sec=", artifact$summary$elapsed_sec,
  " phase10_performance_gate=", artifact$summary$phase10_performance_gate,
  "\n", sep = ""
)
