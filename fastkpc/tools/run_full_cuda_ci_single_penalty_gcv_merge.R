source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_shadow.R")
source("fastkpc/R/full_cuda_ci_phase3_artifacts.R")
source("fastkpc/R/full_cuda_ci_phase35_contracts.R")
source("fastkpc/R/full_cuda_ci_phase4_artifacts.R")

partition_dir <- Sys.getenv(
  "FASTKPC_FULL_CUDA_PHASE4_PARTITION_DIR",
  unset = "/tmp/fastkpc-phase4-full-shadow-partitions-v1"
)
partition_count <- as.integer(Sys.getenv(
  "FASTKPC_FULL_CUDA_PHASE4_PARTITION_COUNT", unset = "16"
))
output <- Sys.getenv(
  "FASTKPC_FULL_CUDA_PHASE4_MERGED_RDS",
  unset = "/tmp/fastkpc-phase4-full-shadow-merged-v1.rds"
)
if (is.na(partition_count) || partition_count < 1L) {
  stop("Phase 4 partition count must be positive", call. = FALSE)
}
partition_paths <- file.path(
  partition_dir,
  sprintf("partition-%03d-of-%03d.rds", 0:(partition_count - 1L),
          partition_count)
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
evidence <- fastkpc_full_cuda_phase4_merge_full_shadow(
  catalog, partition_paths
)
saveRDS(evidence, output, version = 3L)
print(evidence$summary)
print(evidence$mixed_graph$summary)
if (!isTRUE(evidence$summary$same_sp_fixed_solver_gate) ||
    !isTRUE(evidence$summary$oracle_residual_gate) ||
    !isTRUE(evidence$summary$downstream_decision_gate) ||
    !isTRUE(evidence$summary$backend_gate) ||
    !isTRUE(evidence$mixed_graph$summary$pass)) {
  stop("Phase 4 merged full-shadow gate failed", call. = FALSE)
}
cat("PASS Phase 4 merged full shadow:", output, "\n")
