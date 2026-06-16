source("fastkpc/R/wanpdag_validation.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

comparison <- compare_wanpdag_orientation_devices(
  seed = 151,
  n = 96,
  alpha = 0.18,
  max_conditioning_size = 1L,
  orientation_batch_size = 0L,
  fastspline_params = list(knots = 7, lambda_count = 13, ridge = 1e-8)
)

assert_true(isTRUE(comparison$metrics$pdag_identical),
            "orientation device comparison should preserve pdag")
assert_true(isTRUE(comparison$metrics$orientation_counts_identical),
            "orientation device comparison should preserve counts")
assert_true(comparison$metrics$orientation_dcov_batches > 0L,
            "orientation device comparison should record CUDA dCov batches")

bench <- benchmark_wanpdag_orientation_devices(
  seed = 152,
  n = 96,
  alpha = 0.18,
  max_conditioning_size = 1L,
  repeats = 1L,
  fastspline_params = list(knots = 7, lambda_count = 13, ridge = 1e-8)
)

assert_true(is.data.frame(bench$timings),
            "orientation benchmark should return timing rows")
assert_true(nrow(bench$timings) == 2L,
            "orientation benchmark with one repeat should return CPU and CUDA rows")
assert_true(all(is.finite(bench$timings$elapsed_sec)),
            "orientation benchmark elapsed seconds should be finite")
assert_true(any(bench$timings$orientation_cuda_residual_fits > 0L),
            "orientation benchmark should record CUDA residual fits")

cat("test_orientation_device_benchmark.R: PASS\n")
