source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/legacy_runner.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!requireNamespace("RSpectra", quietly = TRUE)) {
  cat("SKIP precision compatible direct legacy dCov: missing RSpectra\n")
  quit(save = "no", status = 0)
}

set.seed(7341)
data <- cbind(
  x = stats::rnorm(90),
  y = stats::rnorm(90)
)
env <- fastkpc_legacy_env()
route <- list(
  precision = "compatible",
  primary_backend = "direct-ci",
  setup_fingerprint = "direct-ci:S:"
)

actual <- fastkpc_execute_ci_direct(
  data = data, x = 1L, y = 2L, S = integer(),
  ci_method = "dcc.gamma", index = 1, legacy_index = TRUE,
  hsic_params = list(), permutation_params = list(),
  route = route, role = "primary"
)
expected <- fastkpc_legacy_kernel_ci(
  data, x = 1L, y = 2L, S = integer(),
  ic.method = "dcc.gamma", index = 1,
  numCol = floor(nrow(data) / 10), env = env
)

assert_true(abs(actual$p.value - expected) < 1e-12,
            "compatible direct dcc.gamma should use legacy dcov.gamma")
assert_true(actual$ci_backend_executed == "legacy-dcov.gamma",
            "compatible direct dcc.gamma should report legacy dCov backend")

cat("PASS precision compatible direct legacy dCov\n")
