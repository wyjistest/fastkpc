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

required_environment <- c(
  CUDA_VISIBLE_DEVICES = "0", OPENBLAS_NUM_THREADS = "1",
  OMP_NUM_THREADS = "1", MKL_NUM_THREADS = "1", BLIS_NUM_THREADS = "1",
  VECLIB_MAXIMUM_THREADS = "1"
)
fastkpc_full_cuda_phase10_campaign_require(
  identical(
    Sys.getenv(names(required_environment), unset = ""),
    required_environment
  ),
  "default-Inf cold/warm qualification requires device 0 and single-thread BLAS"
)

output_path <- Sys.getenv(
  "FASTKPC_DEFAULT_INF_COLD_WARM_RDS",
  unset = "/tmp/fastkpc-default-inf-cold-warm-v2.rds"
)
fastkpc_full_cuda_phase10_campaign_require(
  !file.exists(output_path),
  paste0("default-Inf cold/warm evidence already exists: ", output_path)
)

data <- readRDS(fastkpc_full_cuda_phase10_campaign_paths()[["data"]])
fastkpc_full_cuda_phase10_campaign_require(
  is.matrix(data) && identical(dim(data), c(351L, 48L)) &&
    !is.null(colnames(data)) && all(is.finite(data)),
  "default-Inf cold/warm canonical data is malformed"
)

fastkpc_full_cuda_phase10_campaign_cache_control("configure", 262144L)
fastkpc_full_cuda_phase10_campaign_cache_control(
  "configure_target", 131072L
)
fastkpc_full_cuda_phase10_campaign_cache_control("reset")

cold <- fastkpc_full_cuda_phase10_campaign_timed_call(
  fastkpc_full_cuda_phase10_campaign_candidate_call(data)
)
cold_validation <- fastkpc_full_cuda_phase10_validate_candidate_result(
  cold$value, boundary = "cold"
)
cache_after_cold <- full_cuda_ci_one_call_cache_state_native(data)

warm <- fastkpc_full_cuda_phase10_campaign_timed_call(
  fastkpc_full_cuda_phase10_campaign_candidate_call(data)
)
warm_validation <- fastkpc_full_cuda_phase10_validate_candidate_result(
  warm$value, boundary = "warm"
)
fastkpc_full_cuda_phase10_campaign_require(
  fastkpc_full_cuda_phase10_same_candidate_result(cold$value, warm$value),
  "default-Inf warm replay differs from the complete cold call"
)
cache_after_warm <- full_cuda_ci_one_call_cache_state_native(data)

evidence <- list(
  schema_version = "full-cuda-ci-default-inf-cold-warm-evidence-v2",
  source_closure_sha256 =
    fastkpc_full_cuda_phase10_campaign_source_closure()$sha256,
  native_binary_sha256 = fastkpc_full_cuda_phase7_native_identity()$sha256,
  requested_max_conditioning_size = "Inf",
  resolved_max_conditioning_size = 46L,
  natural_stop_level = 8L,
  cold = list(
    elapsed_sec = cold$elapsed_sec,
    validation = cold_validation,
    summary = cold$value$summary,
    n.edgetests = cold$value$n.edgetests,
    tasks = cold$value$tasks
  ),
  warm = list(
    elapsed_sec = warm$elapsed_sec,
    validation = warm_validation,
    summary = warm$value$summary,
    n.edgetests = warm$value$n.edgetests,
    tasks = warm$value$tasks
  ),
  cache_after_cold = cache_after_cold,
  cache_after_warm = cache_after_warm,
  pass = TRUE
)

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
temporary <- tempfile(".default-inf-cold-warm-", tmpdir = dirname(output_path))
on.exit(unlink(temporary, force = TRUE), add = TRUE)
saveRDS(evidence, temporary, compress = "xz")
fastkpc_full_cuda_phase10_campaign_require(
  file.rename(temporary, output_path),
  "default-Inf cold/warm evidence publication failed"
)

cat(
  "PASS default-Inf cold/warm qualification; cold_sec=", cold$elapsed_sec,
  "; warm_sec=", warm$elapsed_sec,
  "; cold_physical_tests=", cold$value$summary$physical_tests_evaluated,
  "; warm_result_hits=", warm$value$summary$result_cache_hit_count,
  "; target_warm_entries=", warm$value$summary$target_cache_warm_start_entries,
  "\n", sep = ""
)
