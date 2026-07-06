source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/legacy_runner.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

missing <- c("mgcv", "RSpectra")[
  !vapply(c("mgcv", "RSpectra"), requireNamespace, logical(1),
          quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP precision legacy mgcv unbounded executor: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

set.seed(7601)
n <- 72L
z1 <- stats::runif(n, -2, 2)
z2 <- stats::runif(n, -2, 2)
z3 <- stats::runif(n, -2, 2)
data <- cbind(
  x = sin(z1) + 0.25 * z2 + stats::rnorm(n, sd = 0.15),
  y = cos(z1) - 0.20 * z3 + stats::rnorm(n, sd = 0.15),
  z1 = z1,
  z2 = z2,
  z3 = z3
)

env <- fastkpc_legacy_env()
route <- list(primary_backend = "legacy-mgcv", setup_fingerprint = "S:3|4|5")
executors <- fastkpc_default_precision_executors()

actual <- executors[["legacy-mgcv"]](
  data = data, x = 1L, y = 2L, S = c(3L, 4L, 5L),
  ci_method = "dcc.gamma", index = 1, legacy_index = TRUE,
  hsic_params = list(), permutation_params = list(),
  route = route, role = "primary"
)
expected <- fastkpc_legacy_kernel_ci(
  data, x = 1L, y = 2L, S = c(3L, 4L, 5L),
  ic.method = "dcc.gamma", index = 1,
  numCol = floor(nrow(data) / 10), env = env
)

assert_true(is.finite(actual$p.value),
            "legacy-mgcv executor should return finite p-value for |S| > 2")
assert_true(abs(actual$p.value - expected) < 1e-12,
            "legacy-mgcv executor should match legacy kernelCItest for |S| > 2")
assert_true(actual$residual_backend_executed == "legacy-mgcv",
            "legacy-mgcv executor should report legacy residual backend")
assert_true(actual$ci_backend_executed == "legacy-dcov.gamma",
            "legacy-mgcv executor should report legacy dCov gamma backend")

cat("PASS precision legacy mgcv unbounded executor\n")
