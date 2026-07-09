source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

missing <- c("graph", "mgcv", "pcalg", "RSpectra")[
  !vapply(c("graph", "mgcv", "pcalg", "RSpectra"), requireNamespace,
          logical(1), quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP precision compatible legacy parallel runtime breakdown: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

set.seed(8804)
n <- 54L
z1 <- stats::runif(n, -2, 2)
z2 <- stats::runif(n, -2, 2)
data <- cbind(
  x1 = sin(z1) + stats::rnorm(n, sd = 0.1),
  x2 = cos(z1) + stats::rnorm(n, sd = 0.1),
  x3 = z1,
  x4 = z2,
  x5 = stats::rnorm(n)
)

tracked_env <- c(
  "FASTKPC_LEGACY_PROGRESS_CSV",
  "FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE",
  "FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY",
  "FASTKPC_LEGACY_PARALLEL_CORES"
)
old_progress_env <- Sys.getenv(tracked_env, unset = NA_character_)
on.exit({
  for (name in names(old_progress_env)) {
    if (is.na(old_progress_env[[name]])) {
      Sys.unsetenv(name)
    } else {
      do.call(Sys.setenv, stats::setNames(list(old_progress_env[[name]]),
                                          name))
    }
  }
}, add = TRUE)
progress_csv <- tempfile("legacy-compatible-progress-", fileext = ".csv")
Sys.setenv(
  FASTKPC_LEGACY_PROGRESS_CSV = progress_csv,
  FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE = "1",
  FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY = "s",
  FASTKPC_LEGACY_PARALLEL_CORES = "2"
)

result <- fast_kpc(
  data,
  alpha = 0.05,
  max_conditioning_size = 3,
  engine = "cuda",
  precision = "compatible",
  graph_stage = "skeleton",
  ci_method = "dcc.gamma",
  precision_trace_level = "summary",
  benchmark = TRUE
)

summary <- result$skeleton$scheduler_diagnostics$summary
required <- c(
  "legacy_scheduler_elapsed_ms",
  "legacy_ci_total_ms",
  "legacy_residual_total_ms",
  "legacy_dcov_gamma_ms",
  "legacy_dcov_distance_ms",
  "legacy_dcov_lowrank_ms",
  "legacy_dcov_statistic_ms",
  "legacy_dcov_moment_ms",
  "legacy_dcov_pgamma_ms",
  "legacy_direct_ci_count",
  "legacy_conditional_ci_count",
  "legacy_mgcv_fit_count",
  "legacy_dcov_gamma_count",
  "legacy_fake_level0_test_count",
  "legacy_parallel_worker_count"
)
missing_fields <- setdiff(required, names(summary))
assert_true(length(missing_fields) == 0L,
            paste("legacy runtime summary missing", missing_fields[[1L]]))
assert_true(summary$legacy_scheduler_elapsed_ms > 0,
            "legacy scheduler elapsed time should be recorded")
assert_true(summary$legacy_ci_total_ms > 0,
            "legacy CI total time should be recorded")
assert_true(summary$legacy_dcov_gamma_ms > 0,
            "legacy dCov gamma time should be recorded")
assert_true(summary$legacy_dcov_distance_ms > 0,
            "legacy dCov distance time should be recorded")
assert_true(summary$legacy_dcov_lowrank_ms > 0,
            "legacy dCov lowrank time should be recorded")
assert_true(summary$legacy_dcov_gamma_count > 0L,
            "legacy dCov gamma call count should be recorded")
assert_true(summary$legacy_dcov_gamma_count <= sum(result$skeleton$n.edgetests),
            "legacy dCov gamma calls should not exceed recorded edge tests")
assert_true(summary$legacy_fake_level0_test_count >= 0L,
            "legacy fake level-0 test count should be recorded")

by_level <- result$skeleton$scheduler_diagnostics$legacy_runtime_by_level
assert_true(is.data.frame(by_level),
            "legacy runtime by level should be a data frame")
assert_true(all(c("level", "recorded_tests", "ci_calls",
                  "residual_ms", "dcov_gamma_ms", "dcov_distance_ms",
                  "dcov_lowrank_ms", "dcov_moment_ms") %in%
                  names(by_level)),
            "legacy runtime by level should include component columns")
assert_true(nrow(by_level) == length(result$skeleton$n.edgetests),
            "legacy runtime by level should align with n.edgetests levels")

assert_true(file.exists(progress_csv),
            "legacy progress CSV should be written when env path is set")
progress <- utils::read.csv(progress_csv, stringsAsFactors = FALSE)
required_progress <- c("event", "level", "edge_count", "task_count",
                       "worker_count", "dcov_backend", "dcov_lowrank",
                       "elapsed_sec", "n_edgetests", "remaining_edges",
                       "chunk_id", "chunk_size", "edge_index",
                       "completed_edges", "ci_calls", "residual_ms",
                       "dcov_ms")
missing_progress <- setdiff(required_progress, names(progress))
assert_true(length(missing_progress) == 0L,
            paste("legacy progress missing", missing_progress[[1L]]))
assert_true(any(progress$event == "level_start"),
            "legacy progress should record level_start")
assert_true(any(progress$event == "level_complete"),
            "legacy progress should record level_complete")
assert_true(max(progress$level, na.rm = TRUE) >= 0L,
            "legacy progress should record numeric levels")
assert_true(any(progress$event == "level_complete" &
                  progress$n_edgetests > 0),
            "legacy progress should record completed level test counts")
if (any(progress$level > 0, na.rm = TRUE)) {
  assert_true(any(progress$event == "chunk_start"),
              "legacy progress should record chunk_start on conditional levels")
  assert_true(any(progress$event == "chunk_complete"),
              "legacy progress should record chunk_complete on conditional levels")
  assert_true(any(progress$event == "edge_complete"),
              "legacy progress should record edge_complete on conditional levels")
  chunk_complete <- progress[progress$event == "chunk_complete", ,
                             drop = FALSE]
  assert_true(any(chunk_complete$chunk_size > 0),
              "legacy progress should record chunk sizes")
  assert_true(any(chunk_complete$completed_edges > 0),
              "legacy progress should record completed chunk edges")
  assert_true(any(chunk_complete$ci_calls > 0),
              "legacy progress should record chunk CI calls")
}

cat("PASS precision compatible legacy parallel runtime breakdown\n")
