source("fastkpc/R/cuda_residual_validation.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

bench <- benchmark_cuda_fastspline_residual_batch(
  seed = 511,
  n = 120,
  repeats = 2,
  fastspline_params = list(knots = 7, lambda_count = 13, ridge = 1e-8)
)

assert_true(is.data.frame(bench$timings), "timings should be a data frame")
assert_true(is.data.frame(bench$summary), "summary should be a data frame")
assert_true(all(c("single_loop", "true_batch") %in% bench$timings$mode),
            "benchmark should include single_loop and true_batch modes")
assert_true(any(bench$timings$mode == "true_batch" &
                  bench$timings$status == "ok"),
            "true_batch benchmark run should succeed")
assert_true(is.list(bench$batch_diagnostics),
            "benchmark should return batch diagnostics")
assert_true(as.integer(bench$batch_diagnostics$true_batched_groups) > 0L,
            "benchmark should record true-batched groups")

cat("test_true_batched_fastspline_benchmark.R: PASS\n")
