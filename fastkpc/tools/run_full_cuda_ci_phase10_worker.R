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

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) != 3L) {
  stop(
    "usage: run_full_cuda_ci_phase10_worker.R MODE REPETITION STAGING_DIR",
    call. = FALSE
  )
}
mode <- arguments[[1L]]
repetition <- suppressWarnings(as.integer(arguments[[2L]]))
staging_dir <- arguments[[3L]]
fastkpc_full_cuda_phase10_campaign_run_key(mode, repetition)

base_environment <- c(
  CUDA_VISIBLE_DEVICES = "0",
  OPENBLAS_NUM_THREADS = "1",
  OMP_NUM_THREADS = "1",
  MKL_NUM_THREADS = "1",
  BLIS_NUM_THREADS = "1",
  VECLIB_MAXIMUM_THREADS = "1"
)
fastkpc_full_cuda_phase10_campaign_require(
  identical(Sys.getenv(names(base_environment), unset = ""),
            base_environment),
  "Phase 10 campaign worker thread or CUDA environment drifted"
)
baseline_environment <- c(
  FASTKPC_LEGACY_DCOV_GAMMA_BACKEND = "cpp",
  FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = "spectra",
  FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE = "1",
  FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY = "s",
  FASTKPC_NATIVE_LEGACY_MGCV_PROVIDER_CORES = "20",
  FASTKPC_NATIVE_LEGACY_DCOV_BATCH = "round",
  FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH_THREADS = "20"
)
if (identical(mode, "correct_baseline")) {
  do.call(Sys.setenv, as.list(baseline_environment))
} else {
  Sys.unsetenv(names(baseline_environment))
}

freeze_path <- file.path(staging_dir, "freeze.rds")
fastkpc_full_cuda_phase10_campaign_require(
  file.exists(freeze_path) && !dir.exists(freeze_path),
  "Phase 10 campaign worker freeze is missing"
)
freeze <- readRDS(freeze_path)
output_path <- fastkpc_full_cuda_phase10_campaign_run_path(
  staging_dir, mode, repetition
)
if (file.exists(output_path)) {
  existing <- readRDS(output_path)
  fastkpc_full_cuda_phase10_validate_campaign_run(
    existing, freeze, verify_current = TRUE
  )
  cat(
    "PASS Phase 10 campaign resume: ", existing$run_key,
    " elapsed_sec=", existing$elapsed_sec, "\n", sep = ""
  )
  quit(save = "no", status = 0L)
}

evidence <- fastkpc_full_cuda_phase10_capture_campaign_run(
  mode = mode, repetition = repetition, freeze = freeze
)
fastkpc_full_cuda_phase10_write_campaign_run(evidence, output_path)
persisted <- readRDS(output_path)
fastkpc_full_cuda_phase10_validate_campaign_run(
  persisted, freeze, verify_current = TRUE
)
cat(
  "PASS Phase 10 campaign run: ", persisted$run_key,
  " elapsed_sec=", persisted$elapsed_sec,
  if (is.null(persisted$warmup)) "" else paste0(
    " warmup_elapsed_sec=", persisted$warmup$elapsed_sec
  ),
  "\n", sep = ""
)
