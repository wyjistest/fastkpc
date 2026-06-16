source("fastkpc/R/fast_kpc.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

set.seed(101)
data <- cbind(
  x1 = seq(-pi, pi, length.out = 90),
  x2 = sin(seq(-pi, pi, length.out = 90)),
  x3 = cos(seq(-pi, pi, length.out = 90)),
  x4 = rnorm(90)
)

result <- fast_kpc(
  data,
  alpha = 0.2,
  max_conditioning_size = 1L,
  engine = "cpu",
  residual_backend = "fastSpline",
  graph_stage = "wanpdag",
  seed = 101
)

assert_true(inherits(result, "fastkpc_result"), "result should have fastkpc_result class")
assert_true(result$config$engine_used == "cpu", "engine_used should be cpu")
assert_true(result$config$residual_backend == "fastSpline", "residual backend should be fastSpline")
assert_true(is.list(result$skeleton), "skeleton should be present")
assert_true(is.list(result$orientation), "orientation should be present")
assert_true(is.integer(result$orientation$pdag), "orientation pdag should be integer")
assert_true(identical(dim(result$orientation$pdag), c(ncol(data), ncol(data))),
            "pdag dimension should match variable count")
assert_true(is.data.frame(result$timings), "timings should be a data.frame")
assert_true(all(c("stage", "elapsed_sec") %in% names(result$timings)),
            "timings should have stage and elapsed_sec")
assert_true(is.list(result$metrics), "metrics should be present")
assert_true(is.list(result$diagnostics), "diagnostics should be present")

skeleton_only <- fast_kpc(
  data,
  alpha = 0.2,
  max_conditioning_size = 1L,
  engine = "cpu",
  residual_backend = "linear",
  graph_stage = "skeleton"
)
assert_true(is.null(skeleton_only$orientation), "skeleton graph_stage should not orient")
assert_true(skeleton_only$config$graph_stage == "skeleton", "graph_stage should be recorded")

cat("test_fast_kpc_public_api.R: PASS\n")
