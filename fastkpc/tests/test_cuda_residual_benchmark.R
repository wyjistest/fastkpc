source("fastkpc/R/cuda_residual_validation.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

validation <- validate_cuda_fastspline_residuals(seed = 108, n = 96)
assert_true(is.data.frame(validation$cases), "validation cases should be data.frame")
assert_true(all(validation$cases$status == "ok"), "all validation cases should be ok")
assert_true(all(validation$cases$max_abs_residual_diff < 1e-7),
            "residual diffs should be tiny")
assert_true(all(validation$cases$max_abs_fitted_diff < 1e-7),
            "fitted diffs should be tiny")

bench <- benchmark_cuda_fastspline_residuals(seed = 109, n = 160, repeats = 2)
required <- c("case", "device", "repeat", "elapsed_sec", "residual_backend",
              "residual_device", "status")
assert_true(all(required %in% names(bench$timings)),
            "benchmark timings should have required columns")
assert_true(all(bench$timings$status == "ok"), "benchmark timings should be ok")
assert_true(any(bench$timings$residual_device == "cuda"),
            "benchmark should include cuda residual device")
assert_true(any(bench$timings$residual_device == "cpu"),
            "benchmark should include cpu residual device")

cat("test_cuda_residual_benchmark.R: PASS\n")
