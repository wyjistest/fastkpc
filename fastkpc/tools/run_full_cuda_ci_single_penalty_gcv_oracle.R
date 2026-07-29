source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/full_cuda_ci_phase35_contracts.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_single_penalty_gcv.R")
source("fastkpc/R/full_cuda_ci_phase4_artifacts.R")

if (!isTRUE(fastkpc_cuda_available())) {
  stop("Phase 4 oracle scan requires CUDA", call. = FALSE)
}

output <- Sys.getenv(
  "FASTKPC_FULL_CUDA_PHASE4_ORACLE_RDS",
  unset = "/tmp/fastkpc-phase4-oracle-scan-v1.rds"
)
dense_grid_size <- as.integer(Sys.getenv(
  "FASTKPC_FULL_CUDA_PHASE4_DENSE_GRID_SIZE", unset = "161"
))
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
evidence <- fastkpc_full_cuda_phase4_scan_optimizer(
  catalog,
  dense_grid_size = dense_grid_size,
  preserve_transcripts = TRUE,
  progress = TRUE
)
saveRDS(evidence, output, version = 3L)
print(evidence$summary)
if (!isTRUE(evidence$summary$objective_curve_gate) ||
    !isTRUE(evidence$summary$optimizer_objective_gate) ||
    !isTRUE(evidence$summary$optimizer_coverage_gate) ||
    !isTRUE(evidence$summary$backend_gate) ||
    !isTRUE(evidence$summary$transcript_gate)) {
  stop("Phase 4 oracle scan gate failed", call. = FALSE)
}
cat("PASS Phase 4 single-penalty CUDA GCV oracle scan:", output, "\n")
