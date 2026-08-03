source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cpp.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cuda.R")
source("fastkpc/R/full_cuda_ci_phase35_contracts.R")
source("fastkpc/R/full_cuda_ci_phase35_bakeoff.R")
source("fastkpc/R/full_cuda_ci_phase10_optimizer_residual_reuse.R")

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) > 1L) {
  stop(
    "usage: profile_full_cuda_ci_phase10_optimizer_residual_reuse.R [OUTPUT_RDS]",
    call. = FALSE
  )
}
output_path <- if (length(arguments) == 1L) arguments[[1L]] else
  fastkpc_full_cuda_phase10_optimizer_residual_default_output()

data <- readRDS(file.path(
  "fastkpc", "artifacts", "kpc_tprs_real_zhu",
  "cancer_RD-causalDiscoveryInput.rds"
))
shard <- readRDS(file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1",
  "shards", "shard_1.rds"
))
setup_key <-
  "05b18f09b303d481e84ec14c6a3d11da92db2fcc0f9f213aa8319015f7369ae4"
setup <- shard$prepared_s_setups[[setup_key]]
states <- shard$target_states[head(which(
  shard$target_states$prepared_s_key_sha256 == setup_key
), 2L), , drop = FALSE]

qualification <- fastkpc_full_cuda_phase10_qualify_optimizer_residual_reuse(
  setup, states, data, device_id = 0L
)
fastkpc_full_cuda_phase10_validate_optimizer_residual_qualification(
  qualification
)
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
saveRDS(qualification, output_path, compress = "xz")

cat(
  "PASS Phase 10 optimizer residual qualification; decision=",
  qualification$decision, " output=", output_path, "\n", sep = ""
)
print(as.data.frame(qualification$residual_parity), row.names = FALSE)
print(qualification$exact_p_value_parity, row.names = FALSE)
print(qualification$legacy_eig_p_value_parity, row.names = FALSE)
