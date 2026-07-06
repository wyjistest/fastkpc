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

cat("PASS precision compatible legacy parallel runtime breakdown\n")
