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
              "mode_summary", "comparison_summary", "tail_latency",
              "slow_run_attribution", "graph_value",
              "bottleneck_decision", "summary", "paths")
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
assert_true(is.data.frame(benchmark$tail_latency),
            "tail_latency should be a data.frame")
assert_true(is.data.frame(benchmark$slow_run_attribution),
            "slow_run_attribution should be a data.frame")
assert_true(is.data.frame(benchmark$graph_value),
            "graph_value should be a data.frame")
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
assert_true(all(c("scenario_id", "comparison", "pair_count",
                  "median_overhead_ms", "p90_wall_time_ratio",
                  "p95_wall_time_ratio", "bootstrap_ratio_ci_low",
                  "bootstrap_ratio_ci_high") %in%
                  names(benchmark$tail_latency)),
            "tail latency should include paired per-scenario ratio and absolute overhead")
assert_true(any(benchmark$tail_latency$scenario_id == "pooled"),
            "tail latency should include pooled summary")
assert_true(all(c("scenario_id", "repeat", "wall_time_ratio",
                  "overhead_ms", "verified_tests", "verifier_S_groups",
                  "unique_targets", "max_targets_per_group",
                  "verified_but_unreplayed", "tail_attribution") %in%
                  names(benchmark$slow_run_attribution)),
            "slow run attribution should include verifier workload and reason fields")
assert_true(all(c("scenario_id", "repeat", "primary_vs_legacy_flips",
                  "hybrid_vs_legacy_flips", "corrected_flips",
                  "introduced_flips", "primary_legacy_skeleton_shd",
                  "hybrid_legacy_skeleton_shd") %in%
                  names(benchmark$graph_value)),
            "graph value should summarize primary/hybrid differences against legacy")
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

assert_true(file.exists(benchmark$paths$tail_latency),
            "tail latency artifact should be written")
assert_true(file.exists(benchmark$paths$slow_run_attribution),
            "slow run attribution artifact should be written")
assert_true(file.exists(benchmark$paths$graph_value),
            "graph value artifact should be written")

runs_csv <- utils::read.csv(benchmark$paths$runs, stringsAsFactors = FALSE)
assert_true(nrow(runs_csv) == nrow(benchmark$runs),
            "runs.csv should mirror returned run rows")

old_modes <- Sys.getenv("FASTKPC_PRECISION_E2E_MODES", unset = NA_character_)
on.exit({
  if (is.na(old_modes)) {
    Sys.unsetenv("FASTKPC_PRECISION_E2E_MODES")
  } else {
    Sys.setenv(FASTKPC_PRECISION_E2E_MODES = old_modes)
  }
}, add = TRUE)
Sys.setenv(FASTKPC_PRECISION_E2E_MODES = "primary_only_cuda, hybrid_cuda")
assert_true(identical(
  fastkpc_precision_e2e_env_modes(c("legacy_mgcv", "fast_cuda")),
  c("primary_only_cuda", "hybrid_cuda")
), "env mode filter should parse comma-separated precision benchmark modes")

summary_only_entry <- list(
  run_id = "summary-only",
  mode = "hybrid_cuda",
  status = "ok",
  error_message = "",
  wall_time_sec = 0.1,
  result = list(
    config = list(engine_used = "cuda", precision_requested = "hybrid"),
    skeleton = list(
      scheduler_diagnostics = list(summary = list(
        tests_replayed = 40L,
        precision_verifier_tests = 5L
      )),
      residual_cache = list()
    ),
    timings = data.frame(stage = "skeleton", elapsed_sec = 0.1),
    diagnostics = list(precision_trace = NULL)
  )
)
summary_only_row <- fastkpc_precision_e2e_run_row(
  summary_only_entry,
  scenario_id = "summary-only",
  repeat_id = 1L,
  n = 10L,
  p = 4L,
  alpha = 0.1,
  max_conditioning_size = 1L,
  execution_order = 1L,
  warmup_enabled = FALSE
)
assert_true(abs(summary_only_row$verifier_rate - 0.125) < 1e-12,
            "summary trace mode should derive verifier rate from scheduler summary")

legacy_adj <- matrix(c(
  FALSE, TRUE, FALSE,
  TRUE, FALSE, TRUE,
  FALSE, TRUE, FALSE
), nrow = 3L, byrow = TRUE)
primary_adj <- matrix(c(
  FALSE, FALSE, FALSE,
  FALSE, FALSE, TRUE,
  FALSE, TRUE, FALSE
), nrow = 3L, byrow = TRUE)
graph_value <- fastkpc_precision_e2e_graph_value(list(list(
  list(
    run_id = "stress-legacy_mgcv-r1",
    mode = "legacy_mgcv",
    status = "ok",
    wall_time_sec = 0.5,
    result = list(skeleton = list(adjacency = legacy_adj,
                                  pMax = matrix(0, 3L, 3L)))
  ),
  list(
    run_id = "stress-primary_only_cuda-r1",
    mode = "primary_only_cuda",
    status = "ok",
    wall_time_sec = 0.1,
    result = list(skeleton = list(adjacency = primary_adj,
                                  pMax = matrix(0, 3L, 3L)))
  ),
  list(
    run_id = "stress-hybrid_cuda-r1",
    mode = "hybrid_cuda",
    status = "ok",
    wall_time_sec = 1.0,
    result = list(skeleton = list(adjacency = legacy_adj,
                                  pMax = matrix(0, 3L, 3L)))
  )
)))
assert_true(graph_value$primary_vs_legacy_flips[[1L]] == 1L,
            "graph value should count primary flips against legacy")
assert_true(graph_value$corrected_flips[[1L]] == 1L,
            "graph value should count hybrid corrections against legacy")
assert_true(abs(graph_value$hybrid_runtime_vs_legacy[[1L]] - 2) < 1e-12,
            "graph value runtime ratio should be hybrid wall time over legacy wall time")

cat("test_precision_end_to_end_benchmark.R: PASS\n")
