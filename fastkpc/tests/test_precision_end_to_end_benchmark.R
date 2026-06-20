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
  modes = c("legacy_mgcv", "primary_only_cuda", "hybrid_cuda"),
  repeats = 1L,
  alpha = 0.15,
  max_conditioning_size = 1L,
  run_native_cuda = FALSE,
  warmup = TRUE,
  randomize_mode_order = TRUE
)

required <- c("runs", "stage_timing", "cache", "graph_agreement",
              "mode_summary", "comparison_summary", "bottleneck_decision",
              "summary", "paths")
missing <- setdiff(required, names(benchmark))
assert_true(length(missing) == 0L,
            paste("benchmark missing fields:", paste(missing, collapse = ", ")))

assert_true(is.data.frame(benchmark$runs), "runs should be a data.frame")
assert_true(nrow(benchmark$runs) == 3L,
            "one row should be recorded for each requested mode")
assert_true(all(benchmark$runs$mode %in%
                  c("legacy_mgcv", "primary_only_cuda", "hybrid_cuda")),
            "runs should keep requested mode names")
assert_true("execution_order" %in% names(benchmark$runs),
            "runs should record randomized execution order")
assert_true("warmup_enabled" %in% names(benchmark$runs),
            "runs should record whether warm-up was enabled")
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
assert_true(is.data.frame(benchmark$mode_summary),
            "mode_summary should be a data.frame")
assert_true(is.data.frame(benchmark$comparison_summary),
            "comparison_summary should be a data.frame")
assert_true(is.data.frame(benchmark$bottleneck_decision),
            "bottleneck_decision should be a data.frame")
assert_true(nrow(benchmark$graph_agreement) >= 1L,
            "graph agreement should include at least one baseline comparison")
assert_true(all(c("median_wall_time_sec", "p90_wall_time_sec",
                  "geomean_wall_time_sec",
                  "median_stage_share_ci") %in%
                  names(benchmark$mode_summary)),
            "mode_summary should include robust timing and stage-share columns")
assert_true(any(benchmark$comparison_summary$comparison ==
                  "hybrid_cuda_vs_primary_only_cuda"),
            "comparison_summary should include hybrid overhead against primary-only")
assert_true(all(c("analysis_scope", "dominant_phase",
                  "recommended_next_optimization") %in%
                  names(benchmark$bottleneck_decision)),
            "bottleneck decision should name the scope, dominant phase, and action")
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
