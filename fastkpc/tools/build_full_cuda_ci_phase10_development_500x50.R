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
source("fastkpc/R/full_cuda_ci_phase10_hardening.R")
source("fastkpc/R/full_cuda_ci_phase10_campaign.R")
source("fastkpc/R/full_cuda_ci_phase10_development_500x50.R")

path <- fastkpc_full_cuda_phase10_development_500x50_path()
if (file.exists(path)) {
  stop("Phase 10 public 500x50 artifact already exists", call. = FALSE)
}
artifact <- fastkpc_full_cuda_phase10_build_development_500x50()
fastkpc_full_cuda_phase10_validate_development_500x50(artifact)
dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
temporary <- tempfile(".development-500x50-", tmpdir = dirname(path))
on.exit(unlink(temporary, force = TRUE), add = TRUE)
saveRDS(artifact, temporary, compress = "xz")
fastkpc_full_cuda_phase10_development_require(
  file.rename(temporary, path),
  "Phase 10 public 500x50 artifact publication failed"
)
invisible(fastkpc_full_cuda_phase10_load_development_500x50(path))
cat(
  "PASS Phase 10 public 500x50 CPU oracle; elapsed_sec=",
  artifact$baseline_elapsed_sec,
  " logical_tests=", nrow(artifact$oracle$logical_trace),
  "\n", sep = ""
)
