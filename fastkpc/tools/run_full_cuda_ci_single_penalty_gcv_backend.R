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
source("fastkpc/R/full_cuda_ci_phase4_backend.R")

if (!identical(Sys.getenv("CUDA_VISIBLE_DEVICES", unset = ""), "0")) {
  stop("Phase 4 backend benchmark requires physical GPU 0", call. = FALSE)
}
if (!isTRUE(fastkpc_cuda_available())) {
  stop("Phase 4 backend benchmark requires CUDA", call. = FALSE)
}
device <- fastkpc_cuda_device_info()
if (device$device_id != 0L) {
  stop("Phase 4 backend benchmark logical device must be zero", call. = FALSE)
}

output <- Sys.getenv(
  "FASTKPC_FULL_CUDA_PHASE4_BACKEND_RDS",
  unset = "/tmp/fastkpc-phase4-backend-current-v1.rds"
)
repetitions <- as.integer(Sys.getenv(
  "FASTKPC_FULL_CUDA_PHASE4_BACKEND_REPETITIONS", unset = "5"
))
request_cache <- Sys.getenv(
  "FASTKPC_FULL_CUDA_PHASE4_BACKEND_REQUEST_CACHE",
  unset = "/tmp/fastkpc-phase4-backend-requests-current-v1.rds"
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
requests <- if (file.exists(request_cache) && !dir.exists(request_cache)) {
  value <- readRDS(request_cache)
  fastkpc_full_cuda_phase4_validate_backend_requests(value)
  cat("Phase 4 backend request cache validated:", request_cache, "\n")
  value
} else {
  value <- fastkpc_full_cuda_phase4_backend_build_requests(
    catalog, progress = TRUE
  )
  fastkpc_full_cuda_phase4_validate_backend_requests(value)
  saveRDS(value, request_cache, version = 3L)
  cat("Phase 4 backend request cache written:", request_cache, "\n")
  value
}
evidence <- fastkpc_full_cuda_phase4_run_backend_benchmark(
  catalog, repetitions = repetitions, progress = TRUE,
  request_corpus = requests
)
fastkpc_full_cuda_phase4_validate_backend_evidence(evidence)
saveRDS(evidence, output, version = 3L)
print(evidence$summary)
cat("PASS Phase 4 single-penalty CUDA GCV backend:", output, "\n")
