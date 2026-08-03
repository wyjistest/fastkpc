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

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_error <- function(expression, pattern, message) {
  error <- tryCatch(force(expression), error = identity)
  assert_true(
    inherits(error, "error") &&
      grepl(pattern, conditionMessage(error), fixed = TRUE),
    paste0(message, if (inherits(error, "error")) {
      paste0(": ", conditionMessage(error))
    } else ": no error")
  )
}

if (!isTRUE(tryCatch(
  fastkpc_cuda_available(), error = function(error) FALSE
))) {
  cat("SKIP Phase 10 compute-warm helpers: CUDA unavailable\n")
  quit(save = "no", status = 0L)
}

z <- seq(-1.8, 1.8, length.out = 72L)
measured <- cbind(
  measured_a = sin(1.1 * z) + 0.02 * cos(7 * z),
  measured_b = cos(0.9 * z) + 0.02 * sin(5 * z),
  measured_c = z + 0.02 * cos(9 * z),
  measured_d = z^2 + 0.02 * sin(3 * z)
)
storage.mode(measured) <- "double"

evidence <- fastkpc_full_cuda_phase10_capture_compute_warm(
  measured, repetition = 1L, max_conditioning_size = 1L,
  formal_canonical = FALSE, capture_machine = FALSE
)
fastkpc_full_cuda_phase10_validate_compute_warm_evidence(evidence)
profile <- fastkpc_full_cuda_phase10_compute_profile(evidence$result$summary)
assert_true(
    identical(profile$schema_version, "full-cuda-ci-compute-profile-v5") &&
    is.data.frame(profile$stage_timing) &&
    nrow(profile$stage_timing) == 36L &&
    is.data.frame(profile$physical_work) &&
    nrow(profile$physical_work) == 106L &&
    is.data.frame(profile$prefill_batches) &&
    nrow(profile$prefill_batches) ==
      evidence$result$summary$prefill_window_count &&
    all(is.finite(profile$stage_timing$elapsed_ms)) &&
    all(is.finite(profile$physical_work$value)),
  "compute-warm profile must expose complete finite stage/work counters"
)
tampered_summary <- evidence$result$summary
tampered_summary$prefill_batches$target_optimization_count[[1L]] <-
  tampered_summary$prefill_batches$target_optimization_count[[1L]] + 1L
assert_error(
  fastkpc_full_cuda_phase10_compute_profile(tampered_summary),
  "Phase 10 fresh-data compute profile is malformed",
  "compute profile must reject a malformed prefill window receipt"
)
tampered_summary <- evidence$result$summary
tampered_summary$singleton_padding_target_count <-
  tampered_summary$singleton_padding_target_count + 1L
assert_error(
  fastkpc_full_cuda_phase10_compute_profile(tampered_summary),
  "Phase 10 fresh-data compute profile is malformed",
  "compute profile must reject unbalanced singleton padding"
)
assert_true(
  !isTRUE(evidence$result$summary[[
    "cuda_multi_penalty_decomposition_trace_enabled"
  ]]) &&
    evidence$result$summary[[
      "cuda_multi_penalty_decomposition_request_count"
    ]] == 0L,
  "formal-compatible compute-warm helpers must leave trace disabled"
)
assert_error(
  fastkpc_full_cuda_phase10_decomposition_reuse_profile(
    evidence$result$summary
  ),
  "Phase 10 exact decomposition reuse profile is malformed",
  "decomposition reuse profile must reject a trace-free run"
)

tampered <- evidence
tampered$cache_precondition$cache_epoch_after_reset <-
  tampered$cache_precondition$cache_epoch_before_reset
assert_error(
  fastkpc_full_cuda_phase10_validate_compute_warm_evidence(tampered),
  "compute-warm cache precondition failed",
  "compute-warm must reject a reset that did not advance its epoch"
)
tampered <- evidence
tampered$cache_precondition$result_cache_entries_after_reset <- 1L
assert_error(
  fastkpc_full_cuda_phase10_validate_compute_warm_evidence(tampered),
  "compute-warm cache precondition failed",
  "compute-warm must reject a partial result-cache reset"
)
tampered <- evidence
tampered$result$summary$result_cache_preexisting_hit_count <- 1L
assert_error(
  fastkpc_full_cuda_phase10_validate_compute_warm_evidence(tampered),
  "was replayed or skipped CUDA work",
  "compute-warm must reject preexisting result-cache hits"
)
tampered <- evidence
tampered$result$summary$physical_tests_evaluated <- 0L
assert_error(
  fastkpc_full_cuda_phase10_validate_compute_warm_evidence(tampered),
  "generic CUDA prewarm result is invalid",
  "compute-warm must reject a result-cache replay"
)
tampered <- evidence
tampered$cache_precondition$prewarm_dataset_key <-
  tampered$cache_precondition$dataset_key
assert_error(
  fastkpc_full_cuda_phase10_validate_compute_warm_evidence(tampered),
  "compute-warm cache precondition failed",
  "compute-warm must reject same-DatasetKey prewarming"
)

invisible(full_cuda_ci_one_call_cache_control_native("configure", 262144L))
invisible(full_cuda_ci_one_call_cache_control_native(
  "configure_target", 131072L
))
invisible(full_cuda_ci_one_call_cache_control_native("reset"))

cat("PASS Phase 10 fresh-data compute-warm fail-closed helpers\n")
