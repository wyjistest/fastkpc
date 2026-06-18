source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

set.seed(101)
data <- matrix(rnorm(80 * 5), 80, 5)

legacy_fast <- fast_kpc(
  data, alpha = 0.2, max_conditioning_size = 1,
  engine = "cpu", residual_backend = "fastSpline",
  graph_stage = "skeleton", seed = 101
)
precision_fast <- fast_kpc(
  data, alpha = 0.2, max_conditioning_size = 1,
  engine = "cpu", precision = "fast",
  graph_stage = "skeleton", seed = 101
)

assert_true(identical(legacy_fast$skeleton$adjacency,
                      precision_fast$skeleton$adjacency),
            "precision fast must preserve existing fastSpline execution")
assert_true(precision_fast$config$precision == "fast",
            "config should record precision")
assert_true(precision_fast$config$precision_route$primary_backend %in%
              c("fastSplineCUDA", "fastSplineCPU"),
            "fast route should record fastSpline primary")
assert_true(precision_fast$config$precision_route$compatibility_claim == "approximate",
            "fast route should be approximate")

cat("PASS precision fast mode e2e\n")
