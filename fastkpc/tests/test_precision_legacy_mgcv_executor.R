source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP precision legacy mgcv executor: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

set.seed(997)
n <- 44
s <- seq(-1, 1, length.out = n)
data <- cbind(
  x = sin(2 * s) + rnorm(n, sd = 0.05),
  y = cos(2 * s) + rnorm(n, sd = 0.05),
  z = s + rnorm(n, sd = 0.01)
)
route <- list(primary_backend = "legacy-mgcv", setup_fingerprint = "S:3")
executors <- fastkpc_default_precision_executors()

extract <- executors$mgcvExtractGPUGCV(
  data = data, x = 1L, y = 2L, S = 3L, ci_method = "dcc.gamma",
  index = 1, legacy_index = TRUE, hsic_params = list(),
  permutation_params = list(), route = route, role = "primary"
)
legacy <- executors[["legacy-mgcv"]](
  data = data, x = 1L, y = 2L, S = 3L, ci_method = "dcc.gamma",
  index = 1, legacy_index = TRUE, hsic_params = list(),
  permutation_params = list(), route = route, role = "primary"
)

assert_true(extract$residual_backend_executed == "mgcvExtractCPU",
            "mgcvExtractGPUGCV CPU fallback should report mgcvExtractCPU")
assert_true(legacy$residual_backend_executed == "legacy-mgcv",
            "legacy-mgcv executor should report distinct legacy backend")
assert_true(grepl("legacy-mgcv", legacy$p_source_used, fixed = TRUE),
            "legacy p_source_used should name legacy-mgcv")
assert_true(is.finite(legacy$p.value),
            "legacy mgcv executor should return finite p-value")

cat("PASS precision legacy mgcv executor\n")
