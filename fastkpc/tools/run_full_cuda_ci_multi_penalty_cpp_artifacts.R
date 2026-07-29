source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_shadow.R")
source("fastkpc/R/full_cuda_ci_phase3_artifacts.R")
source("fastkpc/R/full_cuda_ci_phase35_contracts.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cpp.R")
source("fastkpc/R/full_cuda_ci_phase5_artifacts.R")
source("fastkpc/R/full_cuda_ci_phase5_publication.R")

evidence_path <- Sys.getenv("FASTKPC_PHASE5_MERGED_EVIDENCE", unset = "")
output_root <- Sys.getenv(
  "FASTKPC_PHASE5_ARTIFACT_ROOT",
  unset = file.path("fastkpc", "artifacts", "full_cuda_ci")
)
if (!nzchar(evidence_path) || !file.exists(evidence_path)) {
  stop("FASTKPC_PHASE5_MERGED_EVIDENCE must name the merged RDS",
       call. = FALSE)
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
artifacts <- fastkpc_full_cuda_phase5_publish_artifacts(
  catalog, evidence_path, output_root = output_root
)
for (kind in names(artifacts)) {
  cat(
    "PASS Phase 5 artifact", kind, ":",
    artifacts[[kind]]$artifact_dir, "producer=",
    artifacts[[kind]]$producer$identity_sha256, "\n"
  )
}
