source("fastkpc/R/wanpdag_validation.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

build_fastkpc_cuda_native(rebuild = TRUE)

comparison <- compare_wanpdag_orientation_devices(
  seed = 149,
  n = 96,
  alpha = 0.18,
  max_conditioning_size = 1L,
  orientation_batch_size = 0L,
  fastspline_params = list(knots = 7, lambda_count = 13, ridge = 1e-8)
)

assert_true(isTRUE(comparison$pdag_identical),
            "CUDA orientation pdag should match CPU orientation")
assert_true(isTRUE(comparison$counts_identical),
            "CUDA orientation counts should match CPU orientation")
assert_true(isTRUE(comparison$batch_size_one_pdag_identical),
            "orientation_batch_size=1 should match auto")
assert_true(any(comparison$diagnostics$orientation_residual_device == "cuda"),
            "comparison diagnostics should include cuda orientation row")
assert_true(any(comparison$diagnostics$regrvonps_cuda_calls > 0L),
            "comparison diagnostics should record CUDA regrVonPS calls")
assert_true(any(comparison$diagnostics$orientation_dcov_pairs > 0L),
            "comparison diagnostics should record dCov pairs")

bench <- benchmark_wanpdag_orientation_devices(
  seed = 150,
  n = 96,
  alpha = 0.18,
  max_conditioning_size = 1L,
  repeats = 1L,
  fastspline_params = list(knots = 7, lambda_count = 13, ridge = 1e-8)
)

assert_true(is.data.frame(bench$timings),
            "benchmark should return timings data frame")
assert_true(all(c("cpu", "cuda") %in% bench$timings$orientation_residual_device),
            "benchmark should include CPU and CUDA orientation rows")
assert_true(any(bench$timings$orientation_cuda_residual_fits > 0L),
            "benchmark should record CUDA residual fit counters")

cat("test_wanpdag_orientation_device_validation.R: PASS\n")
