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
source("fastkpc/R/full_cuda_ci_native_setup.R")

if (!isTRUE(fastkpc_cuda_available())) {
  stop("Phase 4 full shadow requires CUDA", call. = FALSE)
}
output <- Sys.getenv(
  "FASTKPC_FULL_CUDA_PHASE4_SHADOW_RDS",
  unset = "/tmp/fastkpc-phase4-full-shadow-v1.rds"
)
max_setups_text <- Sys.getenv(
  "FASTKPC_FULL_CUDA_PHASE4_MAX_SETUPS", unset = ""
)
max_setups <- if (nzchar(max_setups_text)) {
  as.integer(max_setups_text)
} else {
  NULL
}
run_dcov <- !identical(
  Sys.getenv("FASTKPC_FULL_CUDA_PHASE4_RUN_DCOV", unset = "1"), "0"
)
partition_count_text <- Sys.getenv(
  "FASTKPC_FULL_CUDA_PHASE4_PARTITION_COUNT", unset = ""
)
partition_id_text <- Sys.getenv(
  "FASTKPC_FULL_CUDA_PHASE4_PARTITION_ID", unset = ""
)
partition_count <- if (nzchar(partition_count_text)) {
  as.integer(partition_count_text)
} else {
  NULL
}
partition_id <- if (nzchar(partition_id_text)) {
  as.integer(partition_id_text)
} else {
  NULL
}
native_setup <- identical(
  Sys.getenv("FASTKPC_PHASE7_NATIVE_SETUP", unset = "0"), "1"
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
evidence <- fastkpc_full_cuda_phase4_scan_full_shadow(
  catalog,
  max_setups = max_setups,
  run_dcov = run_dcov,
  partition_id = partition_id,
  partition_count = partition_count,
  progress = TRUE,
  setup_builder = if (native_setup) {
    fastkpc_full_cuda_phase7_setup_builder
  } else {
    NULL
  }
)
if (native_setup) {
  evidence$phase7_execution_identity <-
    fastkpc_full_cuda_phase7_execution_identity(
      catalog, "native-setup-backend"
    )
}
saveRDS(evidence, output, version = 3L)
print(evidence$summary)
if (!isTRUE(evidence$summary$same_sp_fixed_solver_gate) ||
    !isTRUE(evidence$summary$oracle_residual_gate) ||
    !isTRUE(evidence$summary$downstream_decision_gate) ||
    !isTRUE(evidence$summary$optimizer_coverage_gate) ||
    !isTRUE(evidence$summary$backend_gate)) {
  stop("Phase 4 full shadow gate failed", call. = FALSE)
}
cat("PASS Phase 4 single-penalty CUDA GCV full shadow:", output, "\n")
