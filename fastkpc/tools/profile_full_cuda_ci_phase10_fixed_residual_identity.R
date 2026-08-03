source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_phase35_contracts.R")
source("fastkpc/R/full_cuda_ci_phase35_bakeoff.R")
source("fastkpc/R/full_cuda_ci_phase10_fixed_residual_identity.R")

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) > 1L) {
  stop(
    paste(
      "usage: profile_full_cuda_ci_phase10_fixed_residual_identity.R",
      "[OUTPUT_RDS]"
    ),
    call. = FALSE
  )
}
output_path <- if (length(arguments) == 1L) arguments[[1L]] else
  fastkpc_full_cuda_phase10_fixed_identity_default_output()

fixtures <- fastkpc_full_cuda_phase10_fixed_identity_default_fixtures()
qualification <- fastkpc_full_cuda_phase10_qualify_fixed_residual_identity(
  fixtures, device_id = 0L
)
fastkpc_full_cuda_phase10_validate_fixed_residual_identity(qualification)
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
saveRDS(qualification, output_path, compress = "xz")

cat(
  "PASS Phase 10 fixed residual cross-batch qualification; decision=",
  qualification$decision, " output=", output_path, "\n", sep = ""
)
print(qualification$rows, row.names = FALSE)
print(as.data.frame(qualification$gates), row.names = FALSE)
