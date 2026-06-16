source("fastkpc/R/scheduler_validation.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

bench <- benchmark_layer_scheduler(
  seed = 407,
  n = 70,
  p = 5,
  alpha = 0.2,
  max_conditioning_size = 1,
  residual_backend = "fastSpline",
  residual_device = "cuda",
  schedulers = c("legacy", "layer"),
  batch_sizes = c(0L),
  residual_batch_sizes = c(1L, 0L),
  fastspline_params = list(knots = 7, lambda_count = 13, ridge = 1e-8)
)

assert_true(is.data.frame(bench$runs), "benchmark runs should be a data.frame")
assert_true(is.data.frame(bench$summary), "benchmark summary should be a data.frame")
assert_true(all(c("legacy", "layer") %in% bench$runs$scheduler),
            "benchmark should include legacy and layer")
assert_true(isTRUE(bench$graph_equal), "benchmark graph outputs should match")
assert_true(all(bench$runs$tasks_planned >= bench$runs$tests_replayed),
            "planned tasks should cover replayed tests")
assert_true(all(bench$runs$dcov_batches > 0),
            "benchmark should record dCov batches")

cat("test_layer_scheduler_benchmark.R: PASS\n")
