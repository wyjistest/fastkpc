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
source("fastkpc/R/full_cuda_ci_phase9_artifact.R")

artifact_dir <- Sys.getenv(
  "FASTKPC_PHASE9_ARTIFACT_DIR",
  unset = fastkpc_full_cuda_phase9_artifact_dir()
)
validated <- fastkpc_full_cuda_phase9_validate_artifact(
  artifact_dir, verify_current_sources = TRUE
)
stopifnot(
  isTRUE(validated$summary$pass),
  isTRUE(validated$summary$phase9_correctness_gate),
  validated$summary$SHD == 0L,
  validated$summary$logical_tests_consumed == 240498L,
  identical(validated$summary$max_conditioning_size_requested, "Inf"),
  validated$summary$max_conditioning_size_resolved == 46L,
  validated$summary$natural_stop_level == 8L,
  validated$summary$unknown_fallback_count == 0L,
  validated$summary$approximate_backend_count == 0L
)
cat(
  "PASS Phase 9 one-call artifact validation; producer=",
  validated$producer$identity_sha256, "\n", sep = ""
)
