if (!file.exists("fastkpc/R/compatible_cuda_skeleton_artifact.R") ||
    !file.exists("fastkpc/R/cuda_native.R")) {
  stop("compatible CUDA skeleton artifact or CUDA native helpers are missing",
       call. = FALSE)
}
if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP compatible CUDA skeleton native cuda_spectra lowrank: FASTKPC_RUN_CUDA_TESTS != 1\n")
  quit(save = "no", status = 0)
}

source("fastkpc/R/compatible_cuda_skeleton_artifact.R")
source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
`%||%` <- function(x, y) if (is.null(x)) y else x

missing <- c("mgcv", "RSpectra", "Rcpp")[
  !vapply(c("mgcv", "RSpectra", "Rcpp"), requireNamespace, logical(1),
          quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP compatible CUDA skeleton native cuda_spectra lowrank: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}
build_fastkpc_cuda_native(rebuild = TRUE)
if (!isTRUE(fastkpc_cuda_available())) {
  cat("SKIP compatible CUDA skeleton native cuda_spectra lowrank: CUDA unavailable\n")
  quit(save = "no", status = 0)
}

old_cuda_lowrank_threads <- Sys.getenv("FASTKPC_NATIVE_CUDA_LOWRANK_BATCH_THREADS",
                                       unset = NA_character_)
old_cuda_lowrank_component_cache <- Sys.getenv(
  "FASTKPC_NATIVE_CUDA_LOWRANK_COMPONENT_CACHE",
  unset = NA_character_
)
old_cuda_lowrank_component_cache_scope <- Sys.getenv(
  "FASTKPC_NATIVE_CUDA_LOWRANK_COMPONENT_CACHE_SCOPE",
  unset = NA_character_
)
old_cuda_lowrank_component_cache_max_entries <- Sys.getenv(
  "FASTKPC_NATIVE_CUDA_LOWRANK_COMPONENT_CACHE_MAX_ENTRIES",
  unset = NA_character_
)
Sys.setenv(FASTKPC_NATIVE_CUDA_LOWRANK_BATCH_THREADS = "2")
Sys.setenv(FASTKPC_NATIVE_CUDA_LOWRANK_COMPONENT_CACHE = "1")
Sys.setenv(FASTKPC_NATIVE_CUDA_LOWRANK_COMPONENT_CACHE_SCOPE = "level")
Sys.setenv(FASTKPC_NATIVE_CUDA_LOWRANK_COMPONENT_CACHE_MAX_ENTRIES = "128")
on.exit({
  if (is.na(old_cuda_lowrank_threads)) {
    Sys.unsetenv("FASTKPC_NATIVE_CUDA_LOWRANK_BATCH_THREADS")
  } else {
    Sys.setenv(FASTKPC_NATIVE_CUDA_LOWRANK_BATCH_THREADS =
                 old_cuda_lowrank_threads)
  }
  if (is.na(old_cuda_lowrank_component_cache)) {
    Sys.unsetenv("FASTKPC_NATIVE_CUDA_LOWRANK_COMPONENT_CACHE")
  } else {
    Sys.setenv(FASTKPC_NATIVE_CUDA_LOWRANK_COMPONENT_CACHE =
                 old_cuda_lowrank_component_cache)
  }
  if (is.na(old_cuda_lowrank_component_cache_scope)) {
    Sys.unsetenv("FASTKPC_NATIVE_CUDA_LOWRANK_COMPONENT_CACHE_SCOPE")
  } else {
    Sys.setenv(FASTKPC_NATIVE_CUDA_LOWRANK_COMPONENT_CACHE_SCOPE =
                 old_cuda_lowrank_component_cache_scope)
  }
  if (is.na(old_cuda_lowrank_component_cache_max_entries)) {
    Sys.unsetenv("FASTKPC_NATIVE_CUDA_LOWRANK_COMPONENT_CACHE_MAX_ENTRIES")
  } else {
    Sys.setenv(FASTKPC_NATIVE_CUDA_LOWRANK_COMPONENT_CACHE_MAX_ENTRIES =
                 old_cuda_lowrank_component_cache_max_entries)
  }
}, add = TRUE)

set.seed(261901)
n <- 54L
z1 <- stats::runif(n, -2, 2)
z2 <- stats::rnorm(n)
data <- cbind(
  x1 = sin(z1) + stats::rnorm(n, sd = 0.08),
  x2 = cos(z1) + stats::rnorm(n, sd = 0.08),
  x3 = z1 + stats::rnorm(n, sd = 0.04),
  x4 = z2 + 0.2 * z1,
  x5 = z1 * z2 + stats::rnorm(n, sd = 0.1)
)

baseline <- fastkpc_run_compatible_cuda_skeleton_artifact(
  data = data,
  output_dir = tempfile("compatible-cuda-native-spectra-reference-"),
  artifact_name = "compatible_cuda_native_spectra_reference_test",
  alpha = 0.08,
  max_conditioning_size = 1L,
  dcov_batch = "level",
  low_rank = "spectra"
)

candidate <- fastkpc_run_compatible_cuda_skeleton_artifact(
  data = data,
  output_dir = tempfile("compatible-cuda-native-cuda-spectra-"),
  artifact_name = "compatible_cuda_native_cuda_spectra_test",
  alpha = 0.08,
  max_conditioning_size = 1L,
  dcov_batch = "round",
  low_rank = "cuda_spectra",
  reference_result_path = baseline$paths$result_rds
)
summary <- candidate$summary[1L, , drop = FALSE]

required <- c(
  "legacy_dcov_native_lowrank_mode",
  "legacy_dcov_native_cuda_lowrank_backend_enabled",
  "legacy_dcov_native_cuda_lowrank_backend_count",
  "legacy_dcov_native_cuda_lowrank_backend_error_count",
  "legacy_dcov_native_cuda_lowrank_backend_fallback_count",
  "legacy_dcov_native_cuda_lowrank_backend_converged_count",
  "legacy_dcov_native_cuda_lowrank_backend_spectra_matvec_count",
  "legacy_dcov_native_cuda_lowrank_backend_matrix_h2d_ms_during_compute_max",
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
  "legacy_dcov_native_cuda_lowrank_component_batch_substrate_count",
  "legacy_dcov_native_cuda_lowrank_component_batch_substrate_pair_count",
  "legacy_dcov_native_cuda_lowrank_backend_component_distance_ms",
  "legacy_dcov_native_cuda_lowrank_backend_component_lowrank_ms",
  "legacy_dcov_native_cuda_lowrank_backend_component_moment_ms",
  "legacy_dcov_native_cuda_lowrank_backend_component_unaccounted_ms",
  "legacy_dcov_native_cuda_lowrank_backend_component_eig_ms",
  "legacy_dcov_native_cuda_lowrank_backend_combine_ms",
  "legacy_dcov_native_batch_parallel_enabled",
  "legacy_dcov_native_batch_parallel_threads"
)
missing_fields <- setdiff(required, names(summary))
assert_true(length(missing_fields) == 0L,
            paste("native cuda_spectra summary missing",
                  missing_fields[[1L]]))
assert_true(identical(summary$run_status[[1L]], "ok"),
            "native cuda_spectra artifact should complete")
assert_true(identical(summary$low_rank[[1L]], "cuda_spectra"),
            "artifact should record requested cuda_spectra lowrank")
assert_true(isTRUE(summary$adjacency_identical[[1L]]),
            "native cuda_spectra adjacency should match CPU Spectra reference")
assert_true(summary$shd[[1L]] == 0L,
            "native cuda_spectra should have SHD 0 versus CPU Spectra reference")
assert_true(isTRUE(summary$n_edgetests_identical[[1L]]),
            "native cuda_spectra n.edgetests should match CPU Spectra reference")
assert_true(identical(summary$legacy_dcov_native_lowrank_mode[[1L]],
                      "cuda_spectra"),
            "native dCov summary should report cuda_spectra lowrank mode")
assert_true(isTRUE(summary$legacy_dcov_native_cuda_lowrank_backend_enabled[[1L]]),
            "native CUDA lowrank backend should report enabled")
assert_true(identical(
  as.integer(summary$legacy_dcov_native_cuda_lowrank_backend_count[[1L]]),
  as.integer(summary$legacy_dcov_native_count[[1L]])
), "native CUDA lowrank backend should cover every native dCov task")
assert_true(identical(
  as.integer(summary$legacy_dcov_native_cuda_lowrank_backend_error_count[[1L]]),
  0L
), "native CUDA lowrank backend should report zero errors")
assert_true(identical(
  as.integer(summary$legacy_dcov_native_cuda_lowrank_backend_fallback_count[[1L]]),
  0L
), "native CUDA lowrank backend should report zero fallbacks")
assert_true(identical(
  as.integer(summary$legacy_dcov_native_cuda_lowrank_backend_converged_count[[1L]]),
  as.integer(summary$legacy_dcov_native_cuda_lowrank_backend_count[[1L]])
), "native CUDA lowrank backend should converge every dCov pair")
assert_true(
  summary$legacy_dcov_native_cuda_lowrank_backend_spectra_matvec_count[[1L]] > 0,
  "native CUDA lowrank backend should report Spectra matvecs"
)
assert_true(identical(
  as.numeric(summary$legacy_dcov_native_cuda_lowrank_backend_matrix_h2d_ms_during_compute_max[[1L]]),
  0
), "native CUDA lowrank backend should not re-upload matrices during compute")
assert_true(isTRUE(summary$legacy_dcov_native_cuda_lowrank_component_cache_enabled[[1L]]),
            "native CUDA lowrank component cache should report enabled")
assert_true(identical(
  summary$legacy_dcov_native_cuda_lowrank_component_cache_scope[[1L]],
  "level"
), "native CUDA lowrank component cache should report level scope")
assert_true(identical(
  as.integer(summary$legacy_dcov_native_cuda_lowrank_component_cache_level_max_entries[[1L]]),
  128L
), "native CUDA lowrank component cache should report configured max entries")
assert_true(
  as.integer(summary$legacy_dcov_native_cuda_lowrank_component_cache_lookup_count[[1L]]) ==
    2L * as.integer(summary$legacy_dcov_native_cuda_lowrank_backend_count[[1L]]),
  "native CUDA lowrank component cache should look up both components per pair"
)
assert_true(
  as.integer(summary$legacy_dcov_native_cuda_lowrank_component_cache_hit_count[[1L]]) > 0L,
  "native CUDA lowrank component cache should hit repeated residual components"
)
assert_true(
  as.integer(summary$legacy_dcov_native_cuda_lowrank_component_cache_miss_count[[1L]]) <
    2L * as.integer(summary$legacy_dcov_native_cuda_lowrank_backend_count[[1L]]),
  "native CUDA lowrank component cache should avoid repeated component solves"
)
assert_true(
  as.integer(summary$legacy_dcov_native_cuda_lowrank_component_cache_entry_count[[1L]]) ==
    as.integer(summary$legacy_dcov_native_cuda_lowrank_component_cache_miss_count[[1L]]),
  "native CUDA lowrank component cache should count one entry per component miss"
)
assert_true(
  as.integer(summary$legacy_dcov_native_cuda_lowrank_component_cache_cross_batch_hit_count[[1L]]) > 0L,
  "native CUDA lowrank level component cache should hit components from prior batches"
)
assert_true(
  as.integer(summary$legacy_dcov_native_cuda_lowrank_component_cache_level_entry_count_max[[1L]]) <=
    as.integer(summary$legacy_dcov_native_cuda_lowrank_component_cache_level_max_entries[[1L]]),
  "native CUDA lowrank level component cache should respect configured max entries"
)
assert_true(identical(
  as.integer(summary$legacy_dcov_native_cuda_lowrank_component_batch_substrate_count[[1L]]),
  as.integer(summary$legacy_dcov_native_batch_count[[1L]])
), "native CUDA lowrank should route each dCov batch through the component batch substrate")
assert_true(identical(
  as.integer(summary$legacy_dcov_native_cuda_lowrank_component_batch_substrate_pair_count[[1L]]),
  as.integer(summary$legacy_dcov_native_cuda_lowrank_backend_count[[1L]])
), "native CUDA lowrank component batch substrate should cover every dCov pair")
component_stage_ms <- c(
  distance = as.numeric(summary$legacy_dcov_native_cuda_lowrank_backend_component_distance_ms[[1L]]),
  lowrank = as.numeric(summary$legacy_dcov_native_cuda_lowrank_backend_component_lowrank_ms[[1L]]),
  moment = as.numeric(summary$legacy_dcov_native_cuda_lowrank_backend_component_moment_ms[[1L]]),
  unaccounted = as.numeric(summary$legacy_dcov_native_cuda_lowrank_backend_component_unaccounted_ms[[1L]])
)
component_substage_ms <- c(
  eig = as.numeric(summary$legacy_dcov_native_cuda_lowrank_backend_component_eig_ms[[1L]]),
  combine = as.numeric(summary$legacy_dcov_native_cuda_lowrank_backend_combine_ms[[1L]])
)
assert_true(all(is.finite(component_stage_ms)),
            "native CUDA lowrank component stage timings should be finite")
assert_true(all(is.finite(component_substage_ms)),
            "native CUDA lowrank component substage timings should be finite")
assert_true(all(component_stage_ms >= 0),
            "native CUDA lowrank component stage timings should be non-negative")
assert_true(all(component_substage_ms >= 0),
            "native CUDA lowrank component substage timings should be non-negative")
assert_true(component_stage_ms[["lowrank"]] > 0,
            "native CUDA lowrank component lowrank timing should be positive")
assert_true(component_substage_ms[["eig"]] > 0,
            "native CUDA lowrank component eig timing should be positive")
assert_true(component_substage_ms[["combine"]] > 0,
            "native CUDA lowrank component combine timing should be positive")
assert_true(sum(component_stage_ms) <=
              as.numeric(summary$legacy_dcov_native_cuda_lowrank_backend_ms[[1L]]) * 1.05,
            "native CUDA lowrank component stage timings should fit within backend timing")
assert_true(component_substage_ms[["eig"]] <=
              component_stage_ms[["lowrank"]] * 1.05,
            "native CUDA lowrank eig substage timing should fit within lowrank timing")
assert_true(as.integer(summary$legacy_dcov_native_batch_count[[1L]]) > 1L,
            "native CUDA lowrank round mode should exercise multiple dCov batches")
assert_true(isTRUE(summary$legacy_dcov_native_batch_parallel_enabled[[1L]]),
            "native CUDA lowrank batch should enable host pair parallelism")
assert_true(as.integer(summary$legacy_dcov_native_batch_parallel_threads[[1L]]) >= 2L,
            "native CUDA lowrank batch should report requested host threads")

progress <- utils::read.csv(candidate$paths$native_progress_csv,
                            check.names = FALSE)
batch_start <- progress[
  progress$event == "dcov_cuda_lowrank_batch_start",
  , drop = FALSE
]
batch_complete <- progress[
  progress$event == "dcov_cuda_lowrank_batch_complete",
  , drop = FALSE
]
pair_progress <- progress[
  progress$event == "dcov_cuda_lowrank_pair_progress",
  , drop = FALSE
]
assert_true(nrow(batch_start) > 0L,
            "native CUDA lowrank progress should mark batch start")
assert_true(nrow(batch_complete) > 0L,
            "native CUDA lowrank progress should mark batch completion")
assert_true(nrow(pair_progress) > 0L,
            "native CUDA lowrank progress should mark pair progress")
assert_true(max(pair_progress$tests_replayed, na.rm = TRUE) ==
              as.integer(summary$legacy_dcov_native_cuda_lowrank_backend_count[[1L]]),
            "native CUDA lowrank pair progress should reach all dCov pairs")

assert_true(file.exists(candidate$paths$native_lowrank_cache_progress_csv),
            "native CUDA lowrank cache progress CSV should be written")
cache_progress <- utils::read.csv(
  candidate$paths$native_lowrank_cache_progress_csv,
  check.names = FALSE
)
cache_required <- c(
  "event",
  "level",
  "batch_size",
  "component_lookup_count",
  "component_hit_count",
  "component_miss_count",
  "component_cross_batch_hit_count",
  "component_eviction_count",
  "component_level_entry_count_max",
  "component_distance_ms",
  "component_lowrank_ms",
  "component_moment_ms",
  "component_unaccounted_ms",
  "spectra_matvec_count",
  "spectra_matvec_ms",
  "kernel_launch_count",
  "device_matrix_reuse_count",
  "device_workspace_reuse_count",
  "workspace_realloc_count",
  "matrix_bytes",
  "workspace_bytes",
  "matrix_h2d_ms",
  "matrix_h2d_ms_during_compute",
  "matrix_h2d_ms_during_compute_max",
  "workspace_alloc_ms",
  "h2d_ms",
  "kernel_ms",
  "d2h_ms"
)
cache_missing_fields <- setdiff(cache_required, names(cache_progress))
assert_true(length(cache_missing_fields) == 0L,
            paste("native CUDA lowrank cache progress missing",
                  cache_missing_fields[[1L]]))
cache_batches <- cache_progress[
  cache_progress$event == "component_cache_batch_complete",
  , drop = FALSE
]
assert_true(nrow(cache_batches) > 0L,
            "native CUDA lowrank cache progress should include batch rows")
assert_true(
  max(cache_batches$component_cross_batch_hit_count, na.rm = TRUE) > 0L,
  "native CUDA lowrank cache progress should expose cross-batch hits"
)
assert_true(
  max(cache_batches$component_level_entry_count_max, na.rm = TRUE) <=
    as.integer(summary$legacy_dcov_native_cuda_lowrank_component_cache_level_max_entries[[1L]]),
  "native CUDA lowrank cache progress should expose bounded level entry count"
)
cache_component_stage_ms <- c(
  distance = sum(cache_batches$component_distance_ms),
  lowrank = sum(cache_batches$component_lowrank_ms),
  moment = sum(cache_batches$component_moment_ms),
  unaccounted = sum(cache_batches$component_unaccounted_ms)
)
assert_true(all(is.finite(cache_component_stage_ms)),
            "native CUDA lowrank cache progress component stage timings should be finite")
assert_true(all(cache_component_stage_ms >= 0),
            "native CUDA lowrank cache progress component stage timings should be non-negative")
assert_true(cache_component_stage_ms[["lowrank"]] > 0,
            "native CUDA lowrank cache progress should expose positive lowrank timing")
assert_true(sum(cache_component_stage_ms) <=
              sum(cache_batches$component_total_ms) * 1.05,
            "native CUDA lowrank cache progress stage timings should fit within component total")
assert_true(sum(cache_batches$spectra_matvec_count) ==
              as.integer(summary$legacy_dcov_native_cuda_lowrank_backend_spectra_matvec_count[[1L]]),
            "native CUDA lowrank cache progress should account Spectra matvec count")
assert_true(abs(sum(cache_batches$spectra_matvec_ms) -
                  as.numeric(summary$legacy_dcov_native_cuda_lowrank_backend_spectra_matvec_ms[[1L]])) <
              1e-8,
            "native CUDA lowrank cache progress should account Spectra matvec timing")
assert_true(sum(cache_batches$kernel_launch_count) ==
              as.integer(summary$legacy_dcov_native_cuda_lowrank_backend_kernel_launch_count[[1L]]),
            "native CUDA lowrank cache progress should account CUDA kernel launches")
assert_true(abs(sum(cache_batches$kernel_ms) -
                  as.numeric(summary$legacy_dcov_native_cuda_lowrank_backend_kernel_ms[[1L]])) <
              1e-8,
            "native CUDA lowrank cache progress should account CUDA kernel timing")

cat("PASS compatible CUDA skeleton native cuda_spectra lowrank\n")
