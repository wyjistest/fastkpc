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

evidence_path <- Sys.getenv(
  "FASTKPC_PHASE8_EVIDENCE_RDS",
  unset = "/tmp/fastkpc-phase8-dcov-full-evidence-v1.rds"
)
output_root <- Sys.getenv(
  "FASTKPC_PHASE8_ARTIFACT_ROOT",
  unset = fastkpc_full_cuda_phase8_artifact_root()
)
artifacts <- fastkpc_full_cuda_phase8_publish_artifacts(
  evidence_path = evidence_path, output_root = output_root
)
for (kind in names(artifacts)) {
  cat(
    "PASS Phase 8 artifact ", kind, ": ",
    file.path(
      output_root, fastkpc_full_cuda_phase8_artifact_directories()[[kind]]
    ),
    " producer=", artifacts[[kind]]$producer$identity_sha256,
    "\n", sep = ""
  )
}
