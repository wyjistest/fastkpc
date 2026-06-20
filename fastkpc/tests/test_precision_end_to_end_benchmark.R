source("fastkpc/R/precision_end_to_end_benchmark.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

output_dir <- tempfile("precision-e2e-benchmark-")
scenario <- list(
  scenario_id = "smoke-chain",
  generator = function(n, seed) {
    set.seed(seed)
    z1 <- stats::rnorm(n)
    z2 <- sin(z1) + stats::rnorm(n, sd = 0.15)
    z3 <- cos(z2) + stats::rnorm(n, sd = 0.15)
    z4 <- stats::rnorm(n)
    cbind(z1, z2, z3, z4)
  },
  n = 48L,
  seed = 902L
)

benchmark <- fastkpc_run_precision_end_to_end_benchmark(
  output_dir = output_dir,
  scenarios = list(scenario),
  modes = c("legacy_mgcv", "fast_cuda", "compatible_cuda", "hybrid_cuda"),
  repeats = 1L,
  alpha = 0.15,
  max_conditioning_size = 1L,
  run_native_cuda = FALSE
)

required <- c("runs", "stage_timing", "cache", "graph_agreement",
              "summary", "paths")
missing <- setdiff(required, names(benchmark))
assert_true(length(missing) == 0L,
            paste("benchmark missing fields:", paste(missing, collapse = ", ")))

assert_true(is.data.frame(benchmark$runs), "runs should be a data.frame")
assert_true(nrow(benchmark$runs) == 4L,
            "one row should be recorded for each requested mode")
assert_true(all(benchmark$runs$mode %in%
                  c("legacy_mgcv", "fast_cuda", "compatible_cuda",
                    "hybrid_cuda")),
            "runs should keep requested mode names")
assert_true(all(benchmark$runs$status %in% c("ok", "skipped", "error")),
            "runs should use explicit status values")
assert_true(any(benchmark$runs$status == "ok"),
            "at least one benchmark mode should run in the smoke test")
assert_true(any(benchmark$runs$status == "skipped" &
                  benchmark$runs$engine == "cuda"),
            "CUDA rows should be explicit skips when native CUDA is disabled")

assert_true(is.data.frame(benchmark$stage_timing),
            "stage_timing should be a data.frame")
assert_true(is.data.frame(benchmark$cache), "cache should be a data.frame")
assert_true(is.data.frame(benchmark$graph_agreement),
            "graph_agreement should be a data.frame")
assert_true(nrow(benchmark$graph_agreement) >= 1L,
            "graph agreement should include at least one baseline comparison")
assert_true(is.list(benchmark$summary), "summary should exist")
assert_true(nzchar(benchmark$summary$recommended_next_optimization),
            "summary should recommend the next optimization")

for (path in unlist(benchmark$paths, use.names = FALSE)) {
  assert_true(file.exists(path), paste("missing artifact:", path))
}

runs_csv <- utils::read.csv(benchmark$paths$runs, stringsAsFactors = FALSE)
assert_true(nrow(runs_csv) == nrow(benchmark$runs),
            "runs.csv should mirror returned run rows")

cat("test_precision_end_to_end_benchmark.R: PASS\n")
