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
source("fastkpc/R/full_cuda_ci_phase4_backend.R")
source("fastkpc/R/full_cuda_ci_phase4_publication.R")

if (!identical(Sys.getenv("CUDA_VISIBLE_DEVICES", unset = ""), "0")) {
  stop("Phase 4 artifact publication requires physical GPU 0", call. = FALSE)
}
if (!isTRUE(fastkpc_cuda_available())) {
  stop("Phase 4 artifact publication requires CUDA", call. = FALSE)
}

oracle_path <- Sys.getenv(
  "FASTKPC_FULL_CUDA_PHASE4_ORACLE_RDS",
  unset = "/tmp/fastkpc-phase4-oracle-current-v1.rds"
)
shadow_path <- Sys.getenv(
  "FASTKPC_FULL_CUDA_PHASE4_MERGED_RDS",
  unset = "/tmp/fastkpc-phase4-full-shadow-merged-precision-floor-v1.rds"
)
backend_path <- Sys.getenv(
  "FASTKPC_FULL_CUDA_PHASE4_BACKEND_RDS",
  unset = "/tmp/fastkpc-phase4-backend-current-v1.rds"
)
gpu_samples_path <- Sys.getenv(
  "FASTKPC_FULL_CUDA_PHASE4_GPU_SAMPLES",
  unset = "/tmp/fastkpc-phase4-backend-gpu0-current-v1.csv"
)
output_root <- Sys.getenv(
  "FASTKPC_FULL_CUDA_PHASE4_ARTIFACT_ROOT",
  unset = file.path("fastkpc", "artifacts", "full_cuda_ci")
)

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
artifacts <- fastkpc_full_cuda_phase4_publish_artifacts(
  catalog = catalog,
  oracle_path = oracle_path,
  shadow_path = shadow_path,
  backend_path = backend_path,
  gpu_samples_path = gpu_samples_path,
  output_root = output_root
)
for (kind in names(artifacts)) {
  cat(
    "PASS Phase 4 artifact", kind, ":",
    artifacts[[kind]]$artifact_dir, "producer=",
    artifacts[[kind]]$producer$identity_sha256, "\n"
  )
}
