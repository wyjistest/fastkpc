runner <- "fastkpc/R/legacy_dcov_cpp_batch_min_size_artifact.R"
if (!file.exists(runner)) {
  stop("legacy dCov C++ batch min-size artifact runner is missing",
       call. = FALSE)
}
source(runner)

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

missing <- c("graph", "mgcv", "pcalg", "Rcpp", "RcppArmadillo",
             "RcppEigen", "RSpectra")[
  !vapply(c("graph", "mgcv", "pcalg", "Rcpp", "RcppArmadillo",
            "RcppEigen", "RSpectra"),
          requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP legacy dCov C++ batch min-size artifact: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

set.seed(18127)
n <- 56L
z <- stats::runif(n, -2, 2)
z2 <- stats::rnorm(n)
data <- cbind(
  x1 = sin(z) + stats::rnorm(n, sd = 0.12),
  x2 = cos(z) + stats::rnorm(n, sd = 0.12),
  x3 = z,
  x4 = z2 + 0.2 * z,
  x5 = stats::rnorm(n)
)

timeout_ref_path <- tempfile("legacy-dcov-cpp-batch-timeout-reference-",
                             fileext = ".rds")
timeout_reference <- list(
  skeleton = list(
    adjacency = matrix(0L, ncol(data), ncol(data)),
    n.edgetests = integer(),
    scheduler_diagnostics = list(summary = list())
  )
)
saveRDS(timeout_reference, timeout_ref_path)
timeout_dir <- tempfile("legacy-dcov-cpp-batch-min-size-timeout-")
timeout_artifact <- fastkpc_run_legacy_dcov_cpp_batch_min_size_artifact(
  data = data,
  output_dir = timeout_dir,
  artifact_name = "legacy_dcov_cpp_batch_min_size_timeout_test",
  alpha = 0.08,
  max_conditioning_size = 3L,
  batch_mode = "round",
  min_sizes = c(1L),
  num_cores = 2L,
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
            paste("timeout summary missing", timeout_missing[[1L]]))
timeout_candidate <- timeout_summary[
  timeout_summary$route == "candidate", , drop = FALSE
]
assert_true(nrow(timeout_candidate) == 1L,
            "timeout artifact should contain one candidate row")
assert_true(identical(timeout_candidate$run_status[[1L]], "timeout"),
            "timeout candidate should record timeout run status")
assert_true(isTRUE(timeout_candidate$timeout[[1L]]),
            "timeout candidate should mark timeout TRUE")
assert_true(timeout_candidate$timeout_sec[[1L]] == 0,
            "timeout candidate should record configured timeout")
assert_true(is.na(timeout_candidate$shd[[1L]]),
            "timeout candidate should leave correctness fields unknown")
assert_true(any(timeout_progress$route == "candidate" &
                  timeout_progress$event == "timeout" &
                  timeout_progress$status == "timeout"),
            "artifact progress should record candidate timeout")
assert_true(length(timeout_artifact$candidates) == 0L,
            "timeout artifact should not store a completed candidate result")

out_dir <- tempfile("legacy-dcov-cpp-batch-min-size-artifact-")
artifact <- fastkpc_run_legacy_dcov_cpp_batch_min_size_artifact(
  data = data,
  output_dir = out_dir,
  artifact_name = "legacy_dcov_cpp_batch_min_size_artifact_test",
  alpha = 0.08,
  max_conditioning_size = 3L,
  batch_mode = "round",
  min_sizes = c(1L, 999999L),
  num_cores = 2L
)

assert_true(file.exists(artifact$paths$summary_csv),
            "artifact runner should write summary.csv")
assert_true(file.exists(artifact$paths$progress_csv),
            "artifact runner should write progress.csv")
assert_true(file.exists(artifact$paths$runtime_by_level_csv),
            "artifact runner should write runtime_by_level.csv")
assert_true(file.exists(artifact$paths$result_rds),
            "artifact runner should write result.rds")
assert_true(file.exists(artifact$paths$summary_md),
            "artifact runner should write summary.md")

summary <- utils::read.csv(artifact$paths$summary_csv,
                           stringsAsFactors = FALSE)
progress <- utils::read.csv(artifact$paths$progress_csv,
                            stringsAsFactors = FALSE)
required_progress <- c(
  "artifact", "route", "batch_mode", "batch_min_size", "event",
  "elapsed_sec", "status"
)
missing_progress <- setdiff(required_progress, names(progress))
assert_true(length(missing_progress) == 0L,
            paste("artifact progress missing", missing_progress[[1L]]))
assert_true(any(progress$route == "reference" &
                  progress$event == "start"),
            "artifact progress should record reference start")
assert_true(any(progress$route == "reference" &
                  progress$event == "complete"),
            "artifact progress should record reference completion")
assert_true(sum(progress$route == "candidate" &
                  progress$event == "start") == 2L,
            "artifact progress should record candidate starts")
assert_true(sum(progress$route == "candidate" &
                  progress$event == "complete") == 2L,
            "artifact progress should record candidate completions")
required <- c(
  "artifact", "route", "batch_mode", "batch_min_size", "n", "p",
  "reference_source", "reference_result_path",
  "edge_count", "reference_edge_count", "shd", "adjacency_identical",
  "n_edgetests_identical", "n_edgetests_exact", "elapsed_sec",
  "legacy_dcov_cpp_backend_count",
  "legacy_dcov_cpp_batch_backend_count",
  "legacy_dcov_cpp_batch_backend_pair_count",
  "legacy_dcov_cpp_batch_candidate_pair_count",
  "legacy_dcov_cpp_batch_pair_coverage_ratio",
  "legacy_dcov_cpp_batch_skipped_pair_count",
  "legacy_dcov_cpp_batch_skipped_pair_ratio"
)
missing_fields <- setdiff(required, names(summary))
assert_true(length(missing_fields) == 0L,
            paste("artifact summary missing", missing_fields[[1L]]))
assert_true(nrow(summary) == 3L,
            "artifact should contain one reference row and two candidate rows")
assert_true(all(summary$n == n),
            "artifact summary should report input row count")
assert_true(all(summary$p == ncol(data)),
            "artifact summary should report input column count")
assert_true(identical(summary$route[[1L]], "reference"),
            "first artifact row should be the unbatched reference route")
candidate <- summary[summary$route == "candidate", , drop = FALSE]
assert_true(nrow(candidate) == 2L,
            "artifact should contain two candidate rows")
assert_true(all(candidate$shd == 0L),
            "candidate rows should preserve zero SHD on smoke data")
assert_true(all(candidate$adjacency_identical),
            "candidate rows should preserve reference adjacency")
assert_true(all(candidate$n_edgetests_identical),
            "candidate rows should preserve reference n.edgetests")

min_one <- candidate[candidate$batch_min_size == 1L, , drop = FALSE]
assert_true(nrow(min_one) == 1L,
            "artifact should record min-size one candidate")
assert_true(min_one$legacy_dcov_cpp_batch_candidate_pair_count[[1L]] > 0L,
            "min-size one candidate should exercise batch candidates")
assert_true(min_one$legacy_dcov_cpp_batch_pair_coverage_ratio[[1L]] == 1,
            "min-size one candidate should batch all candidate pairs")
assert_true(min_one$legacy_dcov_cpp_batch_skipped_pair_ratio[[1L]] == 0,
            "min-size one candidate should skip no candidate pairs")

min_huge <- candidate[candidate$batch_min_size == 999999L, , drop = FALSE]
assert_true(nrow(min_huge) == 1L,
            "artifact should record huge min-size candidate")
assert_true(min_huge$legacy_dcov_cpp_batch_backend_count[[1L]] == 0L,
            "huge min-size candidate should skip the batch backend")
assert_true(min_huge$legacy_dcov_cpp_batch_pair_coverage_ratio[[1L]] == 0,
            "huge min-size candidate should report zero batch coverage")
assert_true(min_huge$legacy_dcov_cpp_batch_skipped_pair_ratio[[1L]] == 1,
            "huge min-size candidate should report full skipped coverage")

by_level <- utils::read.csv(artifact$paths$runtime_by_level_csv,
                            stringsAsFactors = FALSE)
required_by_level <- c(
  "route", "batch_min_size", "level", "dcov_cpp_batch_candidate_pair_count",
  "dcov_cpp_batch_pair_coverage_ratio",
  "dcov_cpp_batch_skipped_pair_ratio"
)
missing_by_level <- setdiff(required_by_level, names(by_level))
assert_true(length(missing_by_level) == 0L,
            paste("artifact by-level output missing", missing_by_level[[1L]]))
assert_true(any(by_level$route == "candidate"),
            "artifact by-level output should include candidate rows")

summary_md <- paste(readLines(artifact$paths$summary_md, warn = FALSE),
                    collapse = "\n")
assert_true(grepl("legacy_dcov_cpp_batch_min_size_artifact_test",
                  summary_md, fixed = TRUE),
            "artifact summary markdown should name the artifact")
assert_true(grepl("batch coverage", summary_md, fixed = TRUE),
            "artifact summary markdown should mention batch coverage")

reuse_dir <- tempfile("legacy-dcov-cpp-batch-min-size-artifact-reuse-")
reuse <- fastkpc_run_legacy_dcov_cpp_batch_min_size_artifact(
  data = data,
  output_dir = reuse_dir,
  artifact_name = "legacy_dcov_cpp_batch_min_size_artifact_reuse_test",
  alpha = 0.08,
  max_conditioning_size = 3L,
  batch_mode = "round",
  min_sizes = c(999999L),
  num_cores = 2L,
  reference_result_path = artifact$paths$result_rds
)
reuse_summary <- utils::read.csv(reuse$paths$summary_csv,
                                 stringsAsFactors = FALSE)
assert_true(identical(reuse_summary$reference_source[[1L]], "rds"),
            "reuse artifact should record loaded reference source")
assert_true(identical(reuse_summary$reference_result_path[[1L]],
                      artifact$paths$result_rds),
            "reuse artifact should record loaded reference path")
assert_true(reuse_summary$elapsed_sec[[1L]] == 0,
            "reuse artifact should not recompute reference elapsed time")
assert_true(reuse_summary$shd[[2L]] == 0L &&
              isTRUE(reuse_summary$adjacency_identical[[2L]]),
            "reuse artifact candidate should compare against loaded reference")

cat("PASS legacy dCov C++ batch min-size artifact\n")
