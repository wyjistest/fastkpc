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
if (length(arguments) < 1L || length(arguments) > 3L) {
  stop(
    paste(
      "usage: profile_full_cuda_ci_phase10_fresh_data.R",
      "MAX_CONDITIONING_SIZE [OUTPUT_RDS] [formal|development]"
    ),
    call. = FALSE
  )
}
max_conditioning_size <- suppressWarnings(as.integer(arguments[[1L]]))
fastkpc_full_cuda_phase10_compute_require(
  length(max_conditioning_size) == 1L &&
    !is.na(max_conditioning_size) &&
    max_conditioning_size >= 0L && max_conditioning_size <= 7L,
  "Phase 10 profile max conditioning size is outside [0, 7]"
)
path <- if (length(arguments) >= 2L) arguments[[2L]] else file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "phase10_profile_v2",
  sprintf("fresh-data-level-%02d.rds", max_conditioning_size)
)
profile_mode <- if (length(arguments) == 3L) {
  match.arg(arguments[[3L]], c("formal", "development"))
} else if (max_conditioning_size == 7L) {
  "formal"
} else {
  "development"
}
fastkpc_full_cuda_phase10_compute_require(
  profile_mode != "formal" || max_conditioning_size == 7L,
  "Phase 10 formal profile requires max conditioning size 7"
)
base_environment <- c(
  CUDA_VISIBLE_DEVICES = "0", OPENBLAS_NUM_THREADS = "1",
  OMP_NUM_THREADS = "1", MKL_NUM_THREADS = "1", BLIS_NUM_THREADS = "1",
  VECLIB_MAXIMUM_THREADS = "1"
)
fastkpc_full_cuda_phase10_compute_require(
  identical(Sys.getenv(names(base_environment), unset = ""),
            base_environment),
  "Phase 10 profile thread or CUDA environment drifted"
)

if (file.exists(path)) {
  evidence <- readRDS(path)
  fastkpc_full_cuda_phase10_validate_compute_warm_evidence(evidence)
} else {
  data <- readRDS(fastkpc_full_cuda_phase10_campaign_paths()[["data"]])
  evidence <- fastkpc_full_cuda_phase10_capture_compute_warm(
    data, repetition = 1L,
    max_conditioning_size = max_conditioning_size,
    formal_canonical = profile_mode == "formal",
    capture_machine = TRUE
  )
  fastkpc_full_cuda_phase10_write_compute_warm_evidence(evidence, path)
}
fastkpc_full_cuda_phase10_compute_require(
  identical(isTRUE(evidence$formal_canonical), profile_mode == "formal"),
  "Phase 10 profile mode disagrees with the stored evidence"
)
profile <- fastkpc_full_cuda_phase10_compute_profile(evidence$result$summary)
cat(
  "PASS Phase 10 fresh-data profile; max_S=", max_conditioning_size,
  " mode=", profile_mode,
  " elapsed_sec=", evidence$elapsed_sec, "\n", sep = ""
)
print(profile$stage_timing, row.names = FALSE)
print(profile$physical_work, row.names = FALSE)
