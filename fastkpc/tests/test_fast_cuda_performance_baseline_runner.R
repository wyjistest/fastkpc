source("fastkpc/R/fast_cuda_performance_baseline.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

scenario <- list(
  scenario_id = "smoke-n40-p4-m1",
  source = "synthetic",
  n = 40L,
  p = 4L,
  max_conditioning_size = 1L,
  seed = 911L,
  data = fastkpc_fast_cuda_baseline_synthetic_data(40L, 4L, 911L)
)

artifact <- fastkpc_run_fast_cuda_performance_baseline(
  output_dir = tempfile("fast-cuda-baseline-"),
  scenarios = list(scenario),
  modes = "fast_cpu",
  repeats = 1L,
  warmup = FALSE,
  run_native_cuda = FALSE
)

runs <- artifact$runs
summary <- artifact$summary[1L, , drop = FALSE]
required_runs <- c(
  "run_id", "scenario_id", "source", "mode", "repeat_id", "status",
  "n", "p", "max_conditioning_size", "wall_ms", "n_edgetests",
  "final_edges", "route_scheduler", "route_data_plane",
  "precision_overlay_used", "cuda_used", "cpu_fallback_count"
)
missing_runs <- setdiff(required_runs, names(runs))
assert_true(length(missing_runs) == 0L,
            paste("missing run fields:", paste(missing_runs, collapse = ",")))
assert_true(nrow(runs) == 1L, "smoke baseline should have one run")
assert_true(identical(runs$status[[1L]], "ok"),
            "smoke baseline fast_cpu should pass")
assert_true(identical(runs$route_data_plane[[1L]], "native-cpu-skeleton"),
            "fast_cpu baseline should use native CPU skeleton")
assert_true(!isTRUE(runs$precision_overlay_used[[1L]]),
            "fast_cpu baseline should not use precision overlay")
assert_true(is.finite(runs$wall_ms[[1L]]) && runs$wall_ms[[1L]] >= 0,
            "smoke baseline should record wall time")

required_summary <- c(
  "run_count", "ok_count", "skipped_count", "error_count",
  "fast_cuda_route_violations", "fast_cuda_median_speedup_vs_fast_cpu"
)
missing_summary <- setdiff(required_summary, names(summary))
assert_true(length(missing_summary) == 0L,
            paste("missing summary fields:",
                  paste(missing_summary, collapse = ",")))
assert_true(summary$run_count[[1L]] == 1L, "summary should count one run")
assert_true(summary$ok_count[[1L]] == 1L, "summary should count one ok run")
assert_true(summary$error_count[[1L]] == 0L, "summary should report no errors")
assert_true(file.exists(artifact$paths$runs_csv),
            "baseline runs CSV should be written")
assert_true(file.exists(artifact$paths$mode_summary_csv),
            "baseline mode summary CSV should be written")
assert_true(file.exists(artifact$paths$stage_timing_csv),
            "baseline stage timing CSV should be written")
assert_true(file.exists(artifact$paths$graph_agreement_csv),
            "baseline graph agreement CSV should be written")
assert_true(file.exists(artifact$paths$route_summary_csv),
            "baseline route summary CSV should be written")
assert_true(file.exists(artifact$paths$speedup_summary_csv),
            "baseline speedup summary CSV should be written")
assert_true(file.exists(artifact$paths$summary_json),
            "baseline summary JSON should be written")
assert_true(file.exists(artifact$paths$summary_md),
            "baseline summary Markdown should be written")

cat("PASS fast CUDA performance baseline runner\n")
