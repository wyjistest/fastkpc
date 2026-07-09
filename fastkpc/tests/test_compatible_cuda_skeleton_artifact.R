if (!file.exists("fastkpc/R/compatible_cuda_skeleton_artifact.R")) {
  stop("compatible CUDA skeleton artifact runner is missing",
       call. = FALSE)
}
source("fastkpc/R/compatible_cuda_skeleton_artifact.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

missing <- c("mgcv", "RSpectra")[
  !vapply(c("mgcv", "RSpectra"), requireNamespace, logical(1),
          quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP compatible CUDA skeleton artifact: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

old_mgcv_env <- Sys.getenv(c(
  "FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND",
  "FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_NATIVE_S_SIZE_LIMIT",
  "FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_CONDITION_THRESHOLD"
), unset = NA_character_)
on.exit({
  for (name in names(old_mgcv_env)) {
    if (is.na(old_mgcv_env[[name]])) {
      Sys.unsetenv(name)
    } else {
      do.call(Sys.setenv, stats::setNames(list(old_mgcv_env[[name]]), name))
    }
  }
}, add = TRUE)
Sys.setenv(FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND = "caller-sentinel")
Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_NATIVE_S_SIZE_LIMIT")
Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_CONDITION_THRESHOLD")

if (identical(.Platform$OS.type, "unix")) {
  timeout_start <- proc.time()[["elapsed"]]
  timeout_result <- tryCatch(
    fastkpc_compatible_cuda_run_with_timeout(
      function() {
        system("sleep 3")
        "completed"
      },
      candidate_timeout_sec = 1
    ),
    fastkpc_compatible_cuda_timeout = function(e) e
  )
  timeout_elapsed <- proc.time()[["elapsed"]] - timeout_start
  assert_true(inherits(timeout_result, "fastkpc_compatible_cuda_timeout"),
              "compatible CUDA timeout helper should raise timeout condition")
  assert_true(timeout_elapsed < 2.5,
              "compatible CUDA timeout helper should interrupt long system calls")
}

set.seed(91573)
n <- 64L
z1 <- stats::runif(n, -2, 2)
z2 <- stats::rnorm(n)
data <- cbind(
  x1 = sin(z1) + stats::rnorm(n, sd = 0.08),
  x2 = cos(z1) + stats::rnorm(n, sd = 0.08),
  x3 = z1 + stats::rnorm(n, sd = 0.04),
  x4 = z2 + 0.2 * z1,
  x5 = stats::rnorm(n),
  x6 = z1 * z2 + stats::rnorm(n, sd = 0.1)
)

invalid_lowrank <- tryCatch(
  {
    fastkpc_run_compatible_cuda_skeleton_artifact(
      data = data,
      output_dir = tempfile("compatible-cuda-skeleton-invalid-lowrank-"),
      artifact_name = "compatible_cuda_skeleton_invalid_lowrank_test",
      alpha = 0.08,
      max_conditioning_size = 1L,
      low_rank = "definitely_not_lowrank"
    )
    NULL
  },
  error = function(e) e
)
assert_true(inherits(invalid_lowrank, "error"),
            "artifact should reject unsupported native dCov lowrank modes")
assert_true(grepl("low_rank", conditionMessage(invalid_lowrank), fixed = TRUE),
            "unsupported lowrank error should name low_rank")

timeout_ref_path <- tempfile("compatible-cuda-skeleton-timeout-reference-",
                             fileext = ".rds")
timeout_reference <- list(
  adjacency = matrix(0L, ncol(data), ncol(data)),
  n.edgetests = integer(),
  sepsets = vector("list", ncol(data)),
  pMax = matrix(0, ncol(data), ncol(data)),
  summary = list()
)

lowrank_progress_path <- tempfile("compatible-cuda-lowrank-progress-",
                                  fileext = ".csv")
utils::write.csv(
  data.frame(
    event = c("component_cache_batch_complete",
              "component_cache_batch_complete",
              "component_cache_ignored"),
    batch_size = c(4L, 6L, 99L),
    component_cache_scope = c("level", "level", "level"),
    component_cache_level_max_entries = c(128L, 128L, 128L),
    component_lookup_count = c(8L, 12L, 99L),
    component_hit_count = c(3L, 5L, 99L),
    component_miss_count = c(5L, 7L, 99L),
    component_entry_count = c(5L, 7L, 99L),
    component_cross_batch_hit_count = c(2L, 4L, 99L),
    component_eviction_count = c(0L, 2L, 99L),
    component_level_entry_count_max = c(5L, 7L, 99L),
    component_count = c(5L, 7L, 99L),
    component_total_ms = c(13.6, 28.1, 99),
    component_distance_ms = c(1.25, 2.75, 99),
    component_lowrank_ms = c(10, 20, 99),
    component_moment_ms = c(0.5, 1.5, 99),
    component_unaccounted_ms = c(0.1, 0.2, 99),
    component_eig_ms = c(7, 14, 99),
    combine_ms = c(0.25, 0.75, 99),
    elapsed_ms = c(100, 220, 999),
    stringsAsFactors = FALSE
  ),
  lowrank_progress_path,
  row.names = FALSE
)
lowrank_progress_summary <-
  fastkpc_compatible_cuda_lowrank_progress_summary(lowrank_progress_path)
assert_true(lowrank_progress_summary$component_batch_substrate_count == 2L,
            "lowrank progress summary should count completed component batches")
assert_true(lowrank_progress_summary$component_batch_substrate_pair_count == 10L,
            "lowrank progress summary should sum completed batch sizes")
assert_true(lowrank_progress_summary$component_cache_lookup_count == 20L,
            "lowrank progress summary should sum cache lookups")
assert_true(lowrank_progress_summary$component_cache_hit_count == 8L,
            "lowrank progress summary should sum cache hits")
assert_true(lowrank_progress_summary$component_cache_cross_batch_hit_count == 6L,
            "lowrank progress summary should sum cross-batch cache hits")
assert_true(lowrank_progress_summary$component_cache_eviction_count == 2L,
            "lowrank progress summary should sum evictions")
assert_true(lowrank_progress_summary$component_cache_level_entry_count_max == 7L,
            "lowrank progress summary should keep max level entry count")
assert_true(abs(lowrank_progress_summary$component_distance_ms - 4) < 1e-12,
            "lowrank progress summary should sum distance stage ms")
assert_true(abs(lowrank_progress_summary$component_lowrank_ms - 30) < 1e-12,
            "lowrank progress summary should sum lowrank stage ms")
assert_true(abs(lowrank_progress_summary$component_moment_ms - 2) < 1e-12,
            "lowrank progress summary should sum moment stage ms")
assert_true(abs(lowrank_progress_summary$component_unaccounted_ms - 0.3) < 1e-12,
            "lowrank progress summary should sum unaccounted stage ms")
assert_true(abs(lowrank_progress_summary$component_eig_ms - 21) < 1e-12,
            "lowrank progress summary should sum eig stage ms")
assert_true(abs(lowrank_progress_summary$combine_ms - 1) < 1e-12,
            "lowrank progress summary should sum combine stage ms")

timeout_row_with_lowrank_progress <- fastkpc_compatible_cuda_timeout_summary_row(
  artifact_name = "compatible_cuda_skeleton_timeout_progress_test",
  data = data,
  columns = NULL,
  alpha = 0.08,
  max_conditioning_size = 2L,
  index = 0L,
  numCol = ncol(data),
  dcov_batch = "round",
  low_rank = "cuda_spectra",
  mgcv_residual_backend = "legacy",
  mgcv_residual_backend_native_s_size_limit = NULL,
  mgcv_residual_backend_condition_threshold = NULL,
  reference = timeout_reference,
  reference_source = "rds",
  reference_result_path = timeout_ref_path,
  reference_slot = "root",
  expected_edge_count = 0L,
  expected_n_edgetests = integer(),
  timeout_sec = 0,
  elapsed_sec = 0,
  reference_elapsed_sec = 0,
  lowrank_progress_summary = lowrank_progress_summary
)
assert_true(
  timeout_row_with_lowrank_progress$
    legacy_dcov_native_cuda_lowrank_component_batch_substrate_count[[1L]] == 2L,
  "timeout summary should include component progress batch count"
)
assert_true(
  timeout_row_with_lowrank_progress$
    legacy_dcov_native_cuda_lowrank_component_batch_substrate_pair_count[[1L]] == 10L,
  "timeout summary should include component progress pair count"
)
assert_true(
  abs(timeout_row_with_lowrank_progress$
        legacy_dcov_native_cuda_lowrank_backend_component_lowrank_ms[[1L]] -
        30) < 1e-12,
  "timeout summary should include component progress lowrank timing"
)
assert_true(
  abs(timeout_row_with_lowrank_progress$
        legacy_dcov_native_cuda_lowrank_backend_component_eig_ms[[1L]] -
        21) < 1e-12,
  "timeout summary should include component progress eig timing"
)
assert_true(
  abs(timeout_row_with_lowrank_progress$
        legacy_dcov_native_cuda_lowrank_backend_combine_ms[[1L]] -
        1) < 1e-12,
  "timeout summary should include component progress combine timing"
)
saveRDS(timeout_reference, timeout_ref_path)
timeout_dir <- tempfile("compatible-cuda-skeleton-timeout-")
timeout_artifact <- fastkpc_run_compatible_cuda_skeleton_artifact(
  data = data,
  output_dir = timeout_dir,
  artifact_name = "compatible_cuda_skeleton_timeout_test",
  alpha = 0.08,
  max_conditioning_size = 2L,
  dcov_batch = "level",
  reference_result_path = timeout_ref_path,
  candidate_timeout_sec = 0
)
timeout_summary <- utils::read.csv(timeout_artifact$paths$summary_csv,
                                   stringsAsFactors = FALSE)
timeout_progress <- utils::read.csv(timeout_artifact$paths$progress_csv,
                                    stringsAsFactors = FALSE)
timeout_required <- c("run_status", "timeout", "timeout_sec")
timeout_missing <- setdiff(timeout_required, names(timeout_summary))
assert_true(length(timeout_missing) == 0L,
            paste("compatible CUDA timeout summary missing",
                  timeout_missing[[1L]]))
assert_true(identical(timeout_summary$run_status[[1L]], "timeout"),
            "timeout artifact should record timeout run status")
assert_true(isTRUE(timeout_summary$timeout[[1L]]),
            "timeout artifact should mark timeout TRUE")
assert_true(timeout_summary$timeout_sec[[1L]] == 0,
            "timeout artifact should record configured timeout")
assert_true(is.na(timeout_summary$shd[[1L]]),
            "timeout artifact should leave correctness fields unknown")
assert_true(any(timeout_progress$route == "facade" &
                  timeout_progress$event == "timeout" &
                  timeout_progress$status == "timeout"),
            "timeout artifact progress should record facade timeout")
assert_true(any(timeout_progress$route == "reference" &
                  timeout_progress$event == "start" &
                  timeout_progress$status == "rds"),
            "timeout artifact progress should record RDS reference start")
assert_true(is.null(timeout_artifact$facade),
            "timeout artifact should not return a completed facade result")

out_dir <- tempfile("compatible-cuda-skeleton-artifact-")
artifact <- fastkpc_run_compatible_cuda_skeleton_artifact(
  data = data,
  output_dir = out_dir,
  artifact_name = "compatible_cuda_skeleton_artifact_test",
  alpha = 0.08,
  max_conditioning_size = 2L,
  dcov_batch = "level"
)
summary <- artifact$summary[1L, , drop = FALSE]

assert_true(file.exists(artifact$paths$summary_csv),
            "artifact runner should write summary.csv")
assert_true(file.exists(artifact$paths$result_rds),
            "artifact runner should write result.rds")
assert_true(file.exists(artifact$paths$summary_md),
            "artifact runner should write summary.md")
assert_true(file.exists(artifact$paths$native_dcov_stage_csv),
            "artifact runner should write native dCov stage timing CSV")
assert_true(!is.null(artifact$paths$native_progress_csv),
            "artifact runner should expose native_progress.csv path")
assert_true(file.exists(artifact$paths$native_progress_csv),
            "artifact runner should write native_progress.csv")
native_dcov_stage <- utils::read.csv(artifact$paths$native_dcov_stage_csv,
                                     stringsAsFactors = FALSE)
native_dcov_stage_required <- c(
  "artifact", "route", "batch_mode", "direct_input", "stage",
  "elapsed_ms", "share_of_scalar_total", "share_of_batch_call"
)
native_dcov_stage_missing <- setdiff(native_dcov_stage_required,
                                     names(native_dcov_stage))
assert_true(length(native_dcov_stage_missing) == 0L,
            paste("native dCov stage timing missing",
                  native_dcov_stage_missing[[1L]]))
required_native_dcov_stages <- c(
  "materialize", "call_wall", "input", "distance", "lowrank",
  "lowrank_eig", "lowrank_select", "lowrank_center",
  "lowrank_unaccounted", "statistic", "moment", "pgamma", "accounted",
  "scalar_total", "wrapper_overhead", "batch_overhead"
)
missing_native_dcov_stages <- setdiff(required_native_dcov_stages,
                                      native_dcov_stage$stage)
assert_true(length(missing_native_dcov_stages) == 0L,
            paste("native dCov stage timing missing stage",
                  missing_native_dcov_stages[[1L]]))
lowrank_stage <- native_dcov_stage[native_dcov_stage$stage == "lowrank",
                                   , drop = FALSE]
assert_true(lowrank_stage$elapsed_ms[[1L]] > 0,
            "native dCov stage timing should report lowrank elapsed ms")
assert_true(is.finite(lowrank_stage$share_of_scalar_total[[1L]]) &&
              lowrank_stage$share_of_scalar_total[[1L]] >= 0,
            "native dCov stage timing should report lowrank worker-share")
native_progress <- utils::read.csv(artifact$paths$native_progress_csv,
                                   stringsAsFactors = FALSE)
native_progress_required <- c("event", "level", "task_count",
                              "residual_request_count", "tests_replayed",
                              "elapsed_ms")
native_progress_missing <- setdiff(native_progress_required,
                                   names(native_progress))
assert_true(length(native_progress_missing) == 0L,
            paste("native progress missing",
                  native_progress_missing[[1L]]))
assert_true(any(native_progress$event == "level_start"),
            "native progress should record level_start rows")
assert_true(any(native_progress$event == "level_complete"),
            "native progress should record level_complete rows")
assert_true(max(native_progress$level, na.rm = TRUE) >= 0L,
            "native progress should record numeric skeleton levels")

required <- c(
  "artifact", "route", "n", "p", "alpha", "max_conditioning_size",
  "reference_source", "reference_result_path", "run_status", "timeout",
  "timeout_sec",
  "edge_count", "reference_edge_count", "shd", "adjacency_identical",
  "sepsets_identical", "n_edgetests_identical", "n_edgetests_exact",
  "pmax_max_abs_diff", "residual_provider_request_count",
  "residual_provider_call_ms", "residual_provider_matrix_copy_ms",
  "residual_provider_total_ms", "residual_provider_parallel_enabled",
  "residual_provider_parallel_cores",
  "residual_provider_parallel_level_count",
  "residual_provider_parallel_request_count",
  "legacy_dcov_native_count", "legacy_dcov_native_batch_enabled",
  "legacy_dcov_native_batch_mode", "legacy_dcov_native_batch_count",
  "legacy_dcov_native_batch_pair_count",
  "legacy_dcov_native_batch_parallel_enabled",
  "legacy_dcov_native_batch_parallel_threads",
  "legacy_dcov_native_batch_direct_input_enabled",
  "legacy_dcov_native_batch_column_materialize_count",
  "legacy_dcov_native_batch_materialize_ms",
  "legacy_dcov_native_batch_call_ms",
  "legacy_dcov_native_batch_input_ms",
  "legacy_dcov_native_batch_distance_ms",
  "legacy_dcov_native_batch_lowrank_ms",
  "legacy_dcov_native_batch_lowrank_eig_ms",
  "legacy_dcov_native_batch_lowrank_select_ms",
  "legacy_dcov_native_batch_lowrank_center_ms",
  "legacy_dcov_native_batch_lowrank_unaccounted_ms",
  "legacy_dcov_native_lowrank_mode",
  "legacy_dcov_native_lowrank_full_eig_count",
  "legacy_dcov_native_lowrank_spectra_count",
  "legacy_dcov_native_lowrank_spectra_converged_count",
  "legacy_dcov_native_lowrank_spectra_failed_count",
  "legacy_dcov_native_lowrank_spectra_fallback_full_eig_count",
  "legacy_dcov_native_lowrank_spectra_iterations",
  "legacy_dcov_native_lowrank_spectra_nconv",
  "legacy_dcov_native_lowrank_spectra_ncv",
  "legacy_dcov_native_lowrank_spectra_tol",
  "legacy_dcov_native_lowrank_spectra_matvec_count",
  "legacy_dcov_native_lowrank_spectra_matvec_ms",
  "legacy_dcov_native_cuda_lowrank_backend_enabled",
  "legacy_dcov_native_cuda_lowrank_backend_count",
  "legacy_dcov_native_cuda_lowrank_backend_ms",
  "legacy_dcov_native_cuda_lowrank_backend_error_count",
  "legacy_dcov_native_cuda_lowrank_backend_fallback_count",
  "legacy_dcov_native_cuda_lowrank_backend_converged_count",
  "legacy_dcov_native_cuda_lowrank_component_cache_enabled",
  "legacy_dcov_native_cuda_lowrank_component_cache_scope",
  "legacy_dcov_native_cuda_lowrank_component_cache_level_max_entries",
  "legacy_dcov_native_cuda_lowrank_component_cache_lookup_count",
  "legacy_dcov_native_cuda_lowrank_component_cache_hit_count",
  "legacy_dcov_native_cuda_lowrank_component_cache_miss_count",
  "legacy_dcov_native_cuda_lowrank_component_cache_entry_count",
  "legacy_dcov_native_cuda_lowrank_component_cache_cross_batch_hit_count",
  "legacy_dcov_native_cuda_lowrank_component_cache_eviction_count",
  "legacy_dcov_native_cuda_lowrank_component_cache_level_entry_count_max",
  "legacy_dcov_native_cuda_lowrank_backend_spectra_matvec_count",
  "legacy_dcov_native_cuda_lowrank_backend_spectra_matvec_ms",
  "legacy_dcov_native_cuda_lowrank_backend_kernel_launch_count",
  "legacy_dcov_native_cuda_lowrank_backend_device_matrix_reuse_count",
  "legacy_dcov_native_cuda_lowrank_backend_device_workspace_reuse_count",
  "legacy_dcov_native_cuda_lowrank_backend_workspace_realloc_count",
  "legacy_dcov_native_cuda_lowrank_backend_matrix_bytes",
  "legacy_dcov_native_cuda_lowrank_backend_workspace_bytes",
  "legacy_dcov_native_cuda_lowrank_backend_matrix_h2d_ms",
  "legacy_dcov_native_cuda_lowrank_backend_matrix_h2d_ms_during_compute",
  "legacy_dcov_native_cuda_lowrank_backend_matrix_h2d_ms_during_compute_max",
  "legacy_dcov_native_cuda_lowrank_backend_workspace_alloc_ms",
  "legacy_dcov_native_cuda_lowrank_backend_h2d_ms",
  "legacy_dcov_native_cuda_lowrank_backend_kernel_ms",
  "legacy_dcov_native_cuda_lowrank_backend_d2h_ms",
  "legacy_dcov_native_batch_statistic_ms",
  "legacy_dcov_native_batch_moment_ms",
  "legacy_dcov_native_batch_pgamma_ms",
  "legacy_dcov_native_batch_accounted_ms",
  "legacy_dcov_native_batch_scalar_total_ms",
  "legacy_dcov_native_batch_wrapper_overhead_ms",
  "legacy_dcov_native_batch_overhead_ms",
  "compatible_cuda_facade", "compatible_cuda_route",
  "compatible_cuda_residual_authority", "compatible_cuda_ci_authority",
  "mgcv_residual_backend", "residual_provider_response_backend",
  "residual_provider_mgcv_backend",
  "residual_provider_mgcv_cpp_backend_enabled",
  "residual_provider_mgcv_cpp_backend_count",
  "residual_provider_mgcv_cpp_backend_native_count",
  "residual_provider_mgcv_cpp_backend_fallback_count",
  "residual_provider_mgcv_cpp_backend_error_count",
  "elapsed_sec", "reference_elapsed_sec"
)
missing_fields <- setdiff(required, names(summary))
assert_true(length(missing_fields) == 0L,
            paste("compatible CUDA skeleton artifact summary missing",
                  missing_fields[[1L]]))

assert_true(identical(summary$route[[1L]], "facade"),
            "artifact should run the compatible CUDA facade route")
assert_true(isTRUE(summary$compatible_cuda_facade[[1L]]),
            "artifact should record compatible CUDA facade metadata")
assert_true(identical(summary$compatible_cuda_route[[1L]],
                      "legacy-mgcv-provider-native-legacy-dcov"),
            "artifact should record the current facade route")
assert_true(identical(summary$compatible_cuda_residual_authority[[1L]],
                      "legacy-mgcv-regrXonS-provider"),
            "artifact should record residual authority")
assert_true(identical(summary$mgcv_residual_backend[[1L]], "env"),
            "artifact should record default mgcv residual backend mode")
assert_true(identical(artifact$facade$summary$compatible_cuda_mgcv_residual_backend,
                      "env"),
            "artifact facade should record default mgcv residual backend option")
assert_true(identical(summary$residual_provider_mgcv_backend[[1L]], "r"),
            "artifact should record default provider mgcv backend")
assert_true(!isTRUE(summary$residual_provider_mgcv_cpp_backend_enabled[[1L]]),
            "artifact default route should not enable provider C++ residual backend")
assert_true(identical(summary$compatible_cuda_ci_authority[[1L]],
                      "native-legacy-dcov.gamma"),
            "artifact should record CI authority")
assert_true(isTRUE(summary$legacy_dcov_native_batch_enabled[[1L]]),
            "artifact should pass dcov_batch='level' to the facade")
assert_true(identical(summary$legacy_dcov_native_batch_mode[[1L]], "level"),
            "artifact should record the requested native dCov batch mode")
assert_true(identical(as.integer(summary$legacy_dcov_native_batch_pair_count[[1L]]),
                      as.integer(summary$legacy_dcov_native_count[[1L]])),
            "artifact should record batch pair count for all native dCov computations")
assert_true(!isTRUE(summary$legacy_dcov_native_batch_parallel_enabled[[1L]]),
            "artifact default route should not enable threaded native dCov batch")
assert_true(identical(as.integer(summary$legacy_dcov_native_batch_parallel_threads[[1L]]),
                      1L),
            "artifact default route should record one native dCov batch thread")
assert_true(isTRUE(summary$adjacency_identical[[1L]]),
            "artifact facade adjacency should match explicit provider reference")
assert_true(isTRUE(summary$sepsets_identical[[1L]]),
            "artifact facade sepsets should match explicit provider reference")
assert_true(isTRUE(summary$n_edgetests_identical[[1L]]),
            "artifact facade n.edgetests should match explicit provider reference")
assert_true(isTRUE(summary$n_edgetests_exact[[1L]]),
            "artifact n.edgetests exact should be true when reference matches")
assert_true(summary$shd[[1L]] == 0L,
            "artifact facade SHD should be zero against explicit provider reference")
assert_true(summary$pmax_max_abs_diff[[1L]] < 1e-12,
            "artifact facade pMax should match explicit provider reference")
assert_true(summary$residual_provider_request_count[[1L]] > 0L,
            "artifact should exercise residual-provider requests")
assert_true(summary$residual_provider_call_ms[[1L]] >= 0,
            "artifact should report residual provider call timing")
assert_true(summary$residual_provider_matrix_copy_ms[[1L]] >= 0,
            "artifact should report residual provider matrix copy timing")
assert_true(summary$residual_provider_total_ms[[1L]] >=
              summary$residual_provider_call_ms[[1L]],
            "artifact residual provider total should include call timing")
assert_true(!isTRUE(summary$residual_provider_parallel_enabled[[1L]]),
            "artifact default route should not enable parallel residual provider")
assert_true(identical(as.integer(summary$residual_provider_parallel_cores[[1L]]),
                      0L),
            "artifact default route should record zero parallel provider cores")
assert_true(summary$legacy_dcov_native_count[[1L]] > 0L,
            "artifact should exercise native legacy dCov tasks")
assert_true(isTRUE(summary$legacy_dcov_native_batch_direct_input_enabled[[1L]]),
            "artifact should report direct native dCov batch inputs")
assert_true(identical(as.integer(summary$legacy_dcov_native_batch_column_materialize_count[[1L]]),
                      0L),
            "artifact should report zero native dCov host matrix materialization")
assert_true(summary$legacy_dcov_native_batch_materialize_ms[[1L]] >= 0,
            "artifact should report native dCov batch materialization timing")
assert_true(summary$legacy_dcov_native_batch_call_ms[[1L]] >= 0,
            "artifact should report native dCov batch call timing")
assert_true(summary$legacy_dcov_native_batch_lowrank_ms[[1L]] > 0,
            "artifact should report native dCov batch lowrank stage timing")
assert_true(summary$legacy_dcov_native_batch_lowrank_eig_ms[[1L]] > 0,
            "artifact should report native dCov batch lowrank eig timing")
assert_true(identical(summary$legacy_dcov_native_lowrank_mode[[1L]],
                      "spectra"),
            "artifact should report native dCov lowrank mode")
assert_true(summary$legacy_dcov_native_lowrank_spectra_count[[1L]] > 0,
            "artifact should report native dCov Spectra solve count")
assert_true(identical(
  as.integer(summary$legacy_dcov_native_lowrank_spectra_count[[1L]]),
  as.integer(summary$legacy_dcov_native_lowrank_spectra_converged_count[[1L]])
), "artifact should report converged Spectra solves")
assert_true(identical(
  as.integer(summary$legacy_dcov_native_lowrank_spectra_failed_count[[1L]]),
  0L
), "artifact should report zero Spectra failures")
assert_true(identical(
  as.integer(summary$legacy_dcov_native_lowrank_spectra_fallback_full_eig_count[[1L]]),
  0L
), "artifact should report zero Spectra full-eig fallbacks")
assert_true(summary$legacy_dcov_native_lowrank_spectra_iterations[[1L]] > 0,
            "artifact should report native dCov Spectra iteration work")
assert_true(summary$legacy_dcov_native_lowrank_spectra_nconv[[1L]] > 0,
            "artifact should report native dCov Spectra convergence work")
assert_true(summary$legacy_dcov_native_lowrank_spectra_ncv[[1L]] > 0,
            "artifact should report native dCov Spectra ncv")
assert_true(summary$legacy_dcov_native_lowrank_spectra_tol[[1L]] > 0,
            "artifact should report native dCov Spectra tolerance")
assert_true(summary$legacy_dcov_native_lowrank_spectra_matvec_count[[1L]] >= 0,
            "artifact should report native dCov Spectra matvec count")
assert_true(summary$legacy_dcov_native_lowrank_spectra_matvec_ms[[1L]] >= 0,
            "artifact should report native dCov Spectra matvec timing")
artifact_lowrank_parts <- summary$legacy_dcov_native_batch_lowrank_eig_ms[[1L]] +
  summary$legacy_dcov_native_batch_lowrank_select_ms[[1L]] +
  summary$legacy_dcov_native_batch_lowrank_center_ms[[1L]] +
  summary$legacy_dcov_native_batch_lowrank_unaccounted_ms[[1L]]
assert_true(abs(artifact_lowrank_parts -
                  summary$legacy_dcov_native_batch_lowrank_ms[[1L]]) <
              max(1e-6, summary$legacy_dcov_native_batch_lowrank_ms[[1L]] * 1e-6),
            "artifact native dCov batch lowrank substages should reconcile")
assert_true(summary$legacy_dcov_native_batch_scalar_total_ms[[1L]] > 0,
            "artifact should report native dCov batch aggregate scalar timing")

reuse_dir <- tempfile("compatible-cuda-skeleton-artifact-reuse-")
reuse <- fastkpc_run_compatible_cuda_skeleton_artifact(
  data = data,
  output_dir = reuse_dir,
  artifact_name = "compatible_cuda_skeleton_artifact_reuse_test",
  alpha = 0.08,
  max_conditioning_size = 2L,
  dcov_batch = "level",
  reference_result_path = artifact$paths$result_rds
)
reuse_summary <- reuse$summary[1L, , drop = FALSE]
assert_true(identical(reuse_summary$reference_source[[1L]], "rds"),
            "artifact should record loaded reference source")
assert_true(identical(reuse_summary$reference_result_path[[1L]],
                      artifact$paths$result_rds),
            "artifact should record loaded reference path")
assert_true(reuse_summary$reference_elapsed_sec[[1L]] == 0,
            "loaded reference should not report recompute elapsed time")
assert_true(isTRUE(reuse_summary$adjacency_identical[[1L]]),
            "reference reuse artifact should preserve adjacency agreement")
assert_true(reuse_summary$shd[[1L]] == 0L,
            "reference reuse artifact should preserve zero SHD")

cpp_dir <- tempfile("compatible-cuda-skeleton-artifact-cpp-")
cpp_artifact <- fastkpc_run_compatible_cuda_skeleton_artifact(
  data = data,
  output_dir = cpp_dir,
  artifact_name = "compatible_cuda_skeleton_artifact_cpp_guarded_test",
  alpha = 0.08,
  max_conditioning_size = 1L,
  dcov_batch = "level",
  mgcv_residual_backend = "cpp_guarded",
  mgcv_residual_backend_native_s_size_limit = 1,
  mgcv_residual_backend_condition_threshold = 1e300
)
cpp_summary <- cpp_artifact$summary[1L, , drop = FALSE]
assert_true(identical(Sys.getenv("FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND",
                                 unset = ""),
                      "caller-sentinel"),
            "artifact should restore caller mgcv residual backend env")
assert_true(identical(cpp_summary$mgcv_residual_backend[[1L]], "cpp_guarded"),
            "artifact should record scoped cpp_guarded mgcv residual backend")
assert_true(identical(cpp_artifact$facade$summary$compatible_cuda_mgcv_residual_backend,
                      "cpp_guarded"),
            "artifact facade should record scoped cpp_guarded residual backend option")
assert_true(identical(cpp_summary$residual_provider_response_backend[[1L]],
                      "legacy-mgcv-cpp-guarded-level-batch"),
            "artifact should record structured cpp guarded provider backend")
assert_true(identical(cpp_summary$residual_provider_mgcv_backend[[1L]],
                      "cpp_guarded"),
            "artifact should record provider selected cpp_guarded backend")
assert_true(isTRUE(cpp_summary$residual_provider_mgcv_cpp_backend_enabled[[1L]]),
            "artifact should report provider C++ residual backend enabled")
assert_true(identical(as.integer(cpp_summary$residual_provider_mgcv_cpp_backend_count[[1L]]),
                      as.integer(cpp_summary$residual_provider_request_count[[1L]])),
            "artifact should report provider C++ residual backend coverage")
assert_true(cpp_summary$residual_provider_mgcv_cpp_backend_native_count[[1L]] > 0L,
            "artifact should report native fixed-sp residual replay")
assert_true(identical(as.integer(cpp_summary$residual_provider_mgcv_cpp_backend_fallback_count[[1L]]),
                      0L),
            "artifact smoke route should not need provider C++ residual fallback")
assert_true(identical(as.integer(cpp_summary$residual_provider_mgcv_cpp_backend_error_count[[1L]]),
                      0L),
            "artifact smoke route should not report provider C++ residual errors")
assert_true(isTRUE(cpp_summary$adjacency_identical[[1L]]) &&
              cpp_summary$shd[[1L]] == 0L,
            "artifact cpp_guarded route should preserve zero SHD")

full_reference_path <- file.path(
  "fastkpc", "artifacts", "legacy_mgcv_residual_cache_s_affinity_v1",
  "compatible_legacy_cpp_dcov_mgcv_cache_s_affinity_result.rds"
)
if (file.exists(full_reference_path)) {
  full_reference <- fastkpc_compatible_cuda_extract_reference(
    readRDS(full_reference_path)
  )
  assert_true(sum(full_reference$adjacency) / 2L == 110L,
              "full reference artifact should expose 110 skeleton edges")
  assert_true(identical(
    as.integer(full_reference$n.edgetests),
    c(2213L, 52659L, 125293L, 40694L, 13293L, 5422L, 835L, 80L)
  ), "full reference artifact should expose canonical n.edgetests")
}

summary_md <- paste(readLines(artifact$paths$summary_md, warn = FALSE),
                    collapse = "\n")
assert_true(grepl("compatible_cuda_skeleton_artifact_test", summary_md,
                  fixed = TRUE),
            "artifact summary markdown should name the artifact")
assert_true(grepl("SHD: 0", summary_md, fixed = TRUE),
            "artifact summary markdown should report SHD")

cat("PASS compatible CUDA skeleton artifact\n")
