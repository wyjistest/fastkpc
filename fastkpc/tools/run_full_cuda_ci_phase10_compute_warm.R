source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_shadow.R")
source("fastkpc/R/full_cuda_ci_phase3_artifacts.R")
source("fastkpc/R/full_cuda_ci_phase35_contracts.R")
source("fastkpc/R/full_cuda_ci_phase10_performance_v2.R")
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
source("fastkpc/R/full_cuda_ci_phase10_compute_campaign.R")

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) != 1L) {
  stop(
    "usage: run_full_cuda_ci_phase10_compute_warm.R REPETITION",
    call. = FALSE
  )
}
repetition <- suppressWarnings(as.integer(arguments[[1L]]))
base_environment <- c(
  CUDA_VISIBLE_DEVICES = "0", OPENBLAS_NUM_THREADS = "1",
  OMP_NUM_THREADS = "1", MKL_NUM_THREADS = "1", BLIS_NUM_THREADS = "1",
  VECLIB_MAXIMUM_THREADS = "1"
)
fastkpc_full_cuda_phase10_compute_require(
  identical(Sys.getenv(names(base_environment), unset = ""),
            base_environment),
  "Phase 10 compute-warm thread or CUDA environment drifted"
)
path <- fastkpc_full_cuda_phase10_compute_run_path(repetition)
if (file.exists(path)) {
  evidence <- readRDS(path)
  fastkpc_full_cuda_phase10_validate_compute_warm_evidence(evidence)
  cat(
    "PASS Phase 10 compute-warm resume: ", evidence$run_key,
    " elapsed_sec=", evidence$elapsed_sec, "\n", sep = ""
  )
  quit(save = "no", status = 0L)
}

data <- readRDS(fastkpc_full_cuda_phase10_campaign_paths()[["data"]])
evidence <- fastkpc_full_cuda_phase10_capture_compute_warm(
  data, repetition = repetition, max_conditioning_size = 7L,
  formal_canonical = TRUE, capture_machine = TRUE
)
fastkpc_full_cuda_phase10_write_compute_warm_evidence(evidence, path)
cat(
  "PASS Phase 10 fresh-data compute-warm: ", evidence$run_key,
  " elapsed_sec=", evidence$elapsed_sec,
  " physical_tests=", evidence$result$summary$physical_tests_evaluated,
  "\n", sep = ""
)
