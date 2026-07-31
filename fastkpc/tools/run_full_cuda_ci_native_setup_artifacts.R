source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_shadow.R")
source("fastkpc/R/full_cuda_ci_phase3_artifacts.R")
source("fastkpc/R/full_cuda_ci_phase35_contracts.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_single_penalty_gcv.R")
source("fastkpc/R/full_cuda_ci_phase4_artifacts.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cpp.R")
source("fastkpc/R/full_cuda_ci_phase5_artifacts.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cuda.R")
source("fastkpc/R/full_cuda_ci_phase6_artifacts.R")
source("fastkpc/R/full_cuda_ci_native_setup.R")
source("fastkpc/R/full_cuda_ci_phase7_artifacts.R")
source("fastkpc/R/full_cuda_ci_phase7_publication.R")

evidence_path <- Sys.getenv(
  "FASTKPC_PHASE7_MERGED_EVIDENCE",
  unset = "/tmp/fastkpc-phase7-native-setup-full-evidence-v1.rds"
)
output_root <- Sys.getenv(
  "FASTKPC_PHASE7_ARTIFACT_ROOT",
  unset = file.path("fastkpc", "artifacts", "full_cuda_ci")
)
if (!file.exists(evidence_path) || dir.exists(evidence_path)) {
  stop("FASTKPC_PHASE7_MERGED_EVIDENCE is missing", call. = FALSE)
}
catalog <- fastkpc_full_cuda_open_fixed_sp_catalog(
  file.path("fastkpc", "artifacts", "full_cuda_ci", "oracle_351x48_v1"),
  file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "workload_census_351x48_v1"
  ),
  file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1"
  ),
  file.path(
    "fastkpc", "artifacts", "kpc_tprs_real_zhu",
    "cancer_RD-causalDiscoveryInput.rds"
  ),
  require_full = TRUE
)
artifacts <- fastkpc_full_cuda_phase7_publish_artifacts(
  catalog = catalog, evidence_path = evidence_path,
  output_root = output_root
)
for (kind in names(artifacts)) {
  cat(
    "PASS Phase 7 artifact ", kind, ": ",
    file.path(
      output_root,
      fastkpc_full_cuda_phase7_artifact_directory_names()[[kind]]
    ),
    " producer=", artifacts[[kind]]$producer$identity_sha256,
    "\n", sep = ""
  )
}
