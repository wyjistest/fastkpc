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

output <- Sys.getenv(
  "FASTKPC_PHASE8_EVIDENCE_RDS",
  unset = "/tmp/fastkpc-phase8-dcov-full-evidence-v1.rds"
)
inputs <- fastkpc_full_cuda_phase8_load_inputs(
  validate_phase7_artifact = TRUE,
  validate_phase7_evidence = TRUE
)
evidence <- fastkpc_full_cuda_phase8_run_full(
  inputs = inputs, progress = TRUE
)
fastkpc_full_cuda_phase8_validate_full(
  evidence, inputs = inputs, verify_current_identity = TRUE
)
saveRDS(evidence, output, version = 3L)
cat(
  "PASS Phase 8 full CUDA dCov evidence; pairs=",
  evidence$summary$logical_test_count,
  " guarded=", evidence$summary$guarded_pair_count,
  " screen_flips=", evidence$summary$screen_decision_flip_count,
  " final_flips=", evidence$summary$final_decision_flip_count,
  " SHD=", evidence$summary$SHD,
  " elapsed_sec=", format(evidence$summary$elapsed_sec, digits = 8L),
  " output=", output, "\n", sep = ""
)
