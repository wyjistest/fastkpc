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
  dcov_batch = "level",
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
  "legacy_dcov_native_cuda_lowrank_backend_matrix_h2d_ms_during_compute_max"
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

cat("PASS compatible CUDA skeleton native cuda_spectra lowrank\n")
