source("fastkpc/R/legacy_runner.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

missing <- c("mgcv", "RSpectra")[
  !vapply(c("mgcv", "RSpectra"), requireNamespace, logical(1),
          quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP legacy dCov gamma timed oracle: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

set.seed(48291)
n <- 72L
z1 <- stats::runif(n, -2, 2)
z2 <- stats::runif(n, -2, 2)
data <- cbind(
  x = sin(z1) + 0.25 * z2 + stats::rnorm(n, sd = 0.12),
  y = cos(z1) - 0.10 * z2 + stats::rnorm(n, sd = 0.12),
  z1 = z1,
  z2 = z2
)
env <- fastkpc_legacy_env()
numCol <- floor(nrow(data) / 10)

legacy <- env$dcov.gamma(data[, 1L], data[, 2L], index = 1, numCol = numCol)
timed <- fastkpc_legacy_dcov_gamma_timed(
  data[, 1L], data[, 2L], index = 1, numCol = numCol, env = env
)
diag <- timed$diagnostics

assert_true(abs(timed$result$p.value - legacy$p.value) < 1e-12,
            "timed legacy dCov should match legacy p-value")
assert_true(max(abs(timed$result$estimates - legacy$estimates)) < 1e-10,
            "timed legacy dCov should match legacy estimates")
assert_true(all(c("distance_ms", "lowrank_ms", "statistic_ms",
                  "moment_ms", "pgamma_ms", "total_ms") %in% names(diag)),
            "timed legacy dCov should expose component timings")
assert_true(diag$total_ms > 0,
            "timed legacy dCov should record total elapsed time")
assert_true(diag$n == nrow(data) && diag$numCol == numCol,
            "timed legacy dCov should record n and numCol")

oracle <- fastkpc_legacy_dcov_gamma_oracle_case(
  data, x = 1L, y = 2L, S = c(3L, 4L),
  alpha = 0.1, index = 1, numCol = numCol, env = env,
  case_id = "conditional-fixture"
)
expected_p <- fastkpc_legacy_kernel_ci(
  data, x = 1L, y = 2L, S = c(3L, 4L),
  ic.method = "dcc.gamma", index = 1, numCol = numCol, env = env
)
expected_residuals <- env$regrXonS(data[, c(1L, 2L)],
                                   data[, c(3L, 4L)])

assert_true(abs(oracle$meta$p.value - expected_p) < 1e-12,
            "oracle case should match legacy kernelCItest p-value")
assert_true(nrow(oracle$residuals) == nrow(data),
            "oracle case should retain one residual row per sample")
assert_true(max(abs(oracle$residuals$rx - expected_residuals[, 1L])) < 1e-12,
            "oracle case should retain legacy x residuals")
assert_true(max(abs(oracle$residuals$ry - expected_residuals[, 2L])) < 1e-12,
            "oracle case should retain legacy y residuals")
assert_true(identical(oracle$meta$S_key, "3|4"),
            "oracle case should record S key")
assert_true(oracle$meta$delete_edge == (oracle$meta$p.value >= 0.1),
            "oracle case should record alpha decision")
assert_true(oracle$meta$dcov_total_ms > 0,
            "oracle case should record dCov timing")

cat("PASS legacy dCov gamma timed oracle\n")
