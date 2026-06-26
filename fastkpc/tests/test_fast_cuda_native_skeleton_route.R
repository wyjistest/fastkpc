source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

set.seed(8127)
data <- matrix(stats::rnorm(72 * 5), 72, 5)
params <- list(knots = 8, lambda_count = 17, ridge = 1e-8)

native_fast <- fast_kpc(
  data,
  alpha = 0.2,
  max_conditioning_size = 1,
  engine = "cuda",
  precision = "fast",
  graph_stage = "skeleton",
  fastspline_params = params,
  seed = 8127
)

assert_true(identical(native_fast$skeleton$scheduler, "layer"),
            "default fast CUDA skeleton should use native layer scheduler")
assert_true(identical(native_fast$config$precision_execution_status, "executed"),
            "default fast CUDA skeleton should not enter precision overlay")
assert_true(identical(native_fast$config$backend_executed, "fastSplineCUDA"),
            "default fast CUDA skeleton should execute fastSplineCUDA")
assert_true(identical(native_fast$config$ci_backend, "cuda-dcov"),
            "default fast CUDA skeleton should use CUDA dCov")
assert_true(!is.data.frame(native_fast$skeleton$precision_trace),
            "native fast CUDA skeleton should not materialize precision trace")
assert_true(!is.data.frame(native_fast$diagnostics$precision_trace),
            "native fast CUDA diagnostics should not materialize precision trace")

precision_overlay <- fast_kpc(
  data,
  alpha = 0.2,
  max_conditioning_size = 1,
  engine = "cuda",
  precision = "fast",
  graph_stage = "skeleton",
  fastspline_params = params,
  precision_executors = fastkpc_default_precision_executors(),
  precision_trace_level = "summary",
  seed = 8127
)

assert_true(identical(precision_overlay$skeleton$scheduler, "layer-precision"),
            "explicit precision executors should use precision layer overlay")
assert_true(identical(precision_overlay$config$precision_execution_status,
                      "batched-primary-data-plane"),
            "explicit precision overlay should report batched primary data plane")
assert_true(identical(precision_overlay$skeleton$adjacency,
                      native_fast$skeleton$adjacency),
            "precision overlay should preserve native fast CUDA adjacency")
assert_true(identical(precision_overlay$skeleton$n.edgetests,
                      native_fast$skeleton$n.edgetests),
            "precision overlay should preserve native fast CUDA n.edgetests")
assert_true(max(abs(precision_overlay$skeleton$pMax -
                      native_fast$skeleton$pMax)) < 1e-8,
            "precision overlay should preserve native fast CUDA pMax")

cat("PASS fast CUDA native skeleton route\n")
