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
source("fastkpc/R/full_cuda_ci_phase10_development_500x50.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

artifact <- fastkpc_full_cuda_phase10_load_development_500x50()
data <- artifact$data
assert_true(
  identical(dim(data), c(500L, 50L)) && all(is.finite(data)) &&
    identical(storage.mode(data), "double") &&
    length(unique(colnames(data))) == 50L,
  "Phase 10 public development matrix is not deterministic 500x50 binary64"
)

if (!isTRUE(tryCatch(
  fastkpc_cuda_available(), error = function(error) FALSE
))) {
  cat("SKIP Phase 10 public 500x50 CUDA route: CUDA unavailable\n")
  quit(save = "no", status = 0L)
}

invisible(full_cuda_ci_one_call_cache_control_native("configure", 262144L))
invisible(full_cuda_ci_one_call_cache_control_native(
  "configure_target", 131072L
))
invisible(full_cuda_ci_one_call_cache_control_native("reset"))
before <- full_cuda_ci_one_call_cache_state_native(data)
assert_true(
  before$result_cache_entries == 0L &&
    before$result_cache_dataset_entries == 0L &&
    before$target_cache_entries == 0L &&
    before$target_cache_dataset_entries == 0L,
  "Phase 10 public 500x50 test did not start from empty semantic caches"
)

result <- fastkpc_full_cuda_phase10_compute_candidate_call(
  data, max_conditioning_size = 7L
)
comparison <- fastkpc_full_cuda_compare_candidate_skeleton(
  artifact$oracle, result
)
summary <- result$summary
zero_fields <- fastkpc_full_cuda_phase10_campaign_authority_zero_fields()
assert_true(
  isTRUE(comparison$summary$pass) && comparison$summary$SHD == 0L &&
    isTRUE(comparison$summary$adjacency_identical) &&
    isTRUE(comparison$summary$sepsets_identical) &&
    isTRUE(comparison$summary$n_edgetests_identical) &&
    isTRUE(comparison$summary$deletions_identical) &&
    isTRUE(comparison$summary$logical_ci_trace_identical) &&
    summary$n == 500L && summary$p == 50L &&
    summary$native_call_count == 1L &&
    summary$logical_tests_consumed == sum(result$n.edgetests) &&
    summary$physical_tests_evaluated > 0L &&
    summary$physical_residual_fits > 0L &&
    summary$result_cache_dataset_warm_start_entries == 0L &&
    summary$target_cache_dataset_warm_start_entries == 0L &&
    summary$result_cache_preexisting_hit_count == 0L &&
    summary$target_cache_preexisting_hit_count == 0L &&
    all(vapply(zero_fields, function(field) {
      identical(as.numeric(summary[[field]]), 0)
    }, logical(1L))),
  "Phase 10 public 500x50 full-CUDA production-shape gate failed"
)

production_source <- paste(readLines(
  "fastkpc/src/full_cuda_ci_one_call.cpp", warn = FALSE
), collapse = "\n")
assert_true(
  !grepl("(^|[^0-9])(351|48)([^0-9]|$)", production_source,
         perl = TRUE),
  "Phase 10 production one-call source contains a canonical 351/48 constant"
)

orientation <- fast_orient_wanpdag_cpp(
  result, data, residual_backend = "linear", alpha = 0.1,
  ci_diagnostics = TRUE
)
assert_true(
  is.list(orientation) && is.integer(orientation$pdag) &&
    identical(dim(orientation$pdag), c(50L, 50L)),
  "Phase 10 public 500x50 skeleton cannot continue into R orientation"
)

invisible(full_cuda_ci_one_call_cache_control_native("reset"))
cat(
  "PASS Phase 10 public 500x50 production-shape fixture; tests=",
  summary$logical_tests_consumed, " edges=",
  sum(result$adjacency[upper.tri(result$adjacency)]), "\n", sep = ""
)
