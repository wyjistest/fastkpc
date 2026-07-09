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

required <- c(
  "artifact", "route", "n", "p", "alpha", "max_conditioning_size",
  "reference_source", "reference_result_path",
  "edge_count", "reference_edge_count", "shd", "adjacency_identical",
  "sepsets_identical", "n_edgetests_identical", "n_edgetests_exact",
  "pmax_max_abs_diff", "residual_provider_request_count",
  "legacy_dcov_native_count", "legacy_dcov_native_batch_enabled",
  "compatible_cuda_facade", "compatible_cuda_route",
  "compatible_cuda_residual_authority", "compatible_cuda_ci_authority",
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
assert_true(identical(summary$compatible_cuda_ci_authority[[1L]],
                      "native-legacy-dcov.gamma"),
            "artifact should record CI authority")
assert_true(isTRUE(summary$legacy_dcov_native_batch_enabled[[1L]]),
            "artifact should pass dcov_batch='level' to the facade")
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
assert_true(summary$legacy_dcov_native_count[[1L]] > 0L,
            "artifact should exercise native legacy dCov tasks")

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
