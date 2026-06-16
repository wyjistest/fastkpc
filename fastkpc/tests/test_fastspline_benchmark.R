source("fastkpc/R/fastspline_validation.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

result <- benchmark_fastspline_backends(seed = 71, n = 180,
                                        alpha = 0.2,
                                        max_conditioning_size = 2)

assert_true(is.data.frame(result$timings), "timings should be a data.frame")
assert_true(is.data.frame(result$cache), "cache should be a data.frame")

required <- paste(c("linear", "fastSpline", "fastSpline"),
                  c("cpu", "cpu", "cuda"), sep = "/")
observed <- paste(result$timings$backend, result$timings$engine, sep = "/")
assert_true(all(required %in% observed),
            "timings should include linear/cpu, fastSpline/cpu, and fastSpline/cuda")
assert_true(all(is.finite(result$timings$elapsed_sec)),
            "elapsed timings should be finite")
assert_true(all(result$timings$elapsed_sec > 0),
            "elapsed timings should be positive")

assert_true(any(result$cache$hits > 0), "cached skeleton runs should report cache hits")
assert_true(identical(result$fastspline_cpu_vs_cuda$adjacency_identical, TRUE),
            "fastSpline CPU-vs-CUDA adjacency should be identical")
assert_true(result$fastspline_cpu_vs_cuda$max_abs_pmax_diff < 1e-8,
            "fastSpline CPU-vs-CUDA pMax diff should be small")

assert_true(is.list(result$graph$linear_vs_fastspline_cpu),
            "benchmark should include linear vs fastSpline diff")
assert_true(is.list(result$graph$fastspline_cpu_vs_cuda),
            "benchmark should include fastSpline CPU vs CUDA diff")

cat("test_fastspline_benchmark.R: PASS\n")
