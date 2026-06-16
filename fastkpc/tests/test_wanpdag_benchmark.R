source("fastkpc/R/wanpdag_validation.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

bench <- benchmark_wanpdag_pipelines(
  seed = 91,
  n = 140,
  alpha = 0.2,
  max_conditioning_size = 1L
)

required_rows <- data.frame(
  engine = c("cpu", "cpu", "cuda", "cuda"),
  residual_backend = c("fastSpline", "fastSpline", "fastSpline", "fastSpline"),
  stage = c("skeleton", "orientation", "skeleton", "orientation")
)

for (i in seq_len(nrow(required_rows))) {
  row <- required_rows[i, ]
  found <- with(bench$timings,
                engine == row$engine &
                  residual_backend == row$residual_backend &
                  stage == row$stage)
  assert_true(any(found), paste("missing timing row", paste(row, collapse = "/")))
}

assert_true(all(is.finite(bench$timings$elapsed_sec)) &&
              all(bench$timings$elapsed_sec > 0),
            "benchmark elapsed times should be finite and positive")
assert_true(any(bench$cache$residual_backend == "fastSpline" &
                  bench$cache$stage %in% c("orientation", "orientation_probe") &
                  bench$cache$hits > 0),
            "benchmark should record fastSpline orientation cache hits")
assert_true(isTRUE(bench$diff$cpu_vs_cuda$pdag_identical),
            "benchmark CPU-vs-CUDA pdag should be identical")
assert_true(bench$diff$cpu_vs_cuda$max_skeleton_pmax_diff < 1e-8,
            "benchmark CPU-vs-CUDA skeleton pMax diff should be tiny")
assert_true(is.list(bench$diff$linear_vs_fastspline),
            "benchmark should include linear-vs-fastSpline diff")

cat("test_wanpdag_benchmark.R: PASS\n")
