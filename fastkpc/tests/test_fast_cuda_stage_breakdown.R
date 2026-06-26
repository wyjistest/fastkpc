source("fastkpc/R/fast_cuda_stage_breakdown.R")
source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP fast CUDA stage breakdown: FASTKPC_RUN_CUDA_TESTS != 1\n")
  quit(save = "no", status = 0)
}

build_fastkpc_cuda_native(rebuild = FALSE)
if (!fastkpc_cuda_available()) {
  cat("SKIP fast CUDA stage breakdown: CUDA unavailable\n")
  quit(save = "no", status = 0)
}

scenario <- list(
  scenario_id = "smoke-n48-p5-m1",
  n = 48L,
  p = 5L,
  max_conditioning_size = 1L,
  seed = 9201L,
  data = fastkpc_stage_breakdown_synthetic_data(48L, 5L, 9201L)
)

artifact <- fastkpc_run_fast_cuda_stage_breakdown(
  output_dir = tempfile("fast-cuda-stage-breakdown-"),
  scenarios = list(scenario),
  repeats = 1L
)

breakdown <- artifact$breakdown
runs <- artifact$runs
summary <- artifact$summary

required_breakdown <- c(
  "scenario_id", "repeat_id", "n", "p", "max_conditioning_size",
  "stage", "stage_group", "elapsed_ms", "share_of_skeleton"
)
missing_breakdown <- setdiff(required_breakdown, names(breakdown))
assert_true(length(missing_breakdown) == 0L,
            paste("missing breakdown fields:",
                  paste(missing_breakdown, collapse = ",")))
required_stages <- c(
  "skeleton_total", "fastspline_residual_prefetch", "ci_eval_total",
  "native_replay", "dcov_rowsum_distance", "dcov_fused_center_reduce",
  "dcov_measured_total", "residual_host_pack", "residual_factor_solve",
  "residual_d2h", "residual_true_batch_total"
)
missing_stages <- setdiff(required_stages, unique(breakdown$stage))
assert_true(length(missing_stages) == 0L,
            paste("missing stages:", paste(missing_stages, collapse = ",")))
assert_true(all(is.finite(breakdown$elapsed_ms) | is.na(breakdown$elapsed_ms)),
            "stage elapsed_ms values should be finite or NA")
assert_true(all(breakdown$elapsed_ms[is.finite(breakdown$elapsed_ms)] >= 0),
            "stage elapsed_ms values should be nonnegative")

assert_true(nrow(runs) == 1L, "smoke run should produce one run row")
assert_true(identical(runs$scheduler[[1L]], "layer"),
            "stage breakdown should use native layer scheduler")
assert_true(!isTRUE(runs$precision_overlay_used[[1L]]),
            "stage breakdown should not use precision overlay")
assert_true(runs$cpu_fallback_count[[1L]] == 0L,
            "stage breakdown should not use CPU fallback")
assert_true(runs$dcov_batches[[1L]] > 0L,
            "stage breakdown should record dCov batches")
assert_true(runs$dcov_chunks[[1L]] > 0L,
            "stage breakdown should record dCov chunks")
assert_true(runs$unique_residual_requests[[1L]] > 0L,
            "stage breakdown should record unique residual requests")
assert_true(runs$cuda_residual_true_batched_fits[[1L]] >= 0L,
            "stage breakdown should record true batched residual fits")

required_summary <- c(
  "stage", "stage_group", "run_count", "finite_count", "median_ms", "p90_ms"
)
missing_summary <- setdiff(required_summary, names(summary))
assert_true(length(missing_summary) == 0L,
            paste("missing summary fields:",
                  paste(missing_summary, collapse = ",")))
assert_true(file.exists(artifact$paths$breakdown_csv),
            "stage breakdown CSV should be written")
assert_true(file.exists(artifact$paths$runs_csv),
            "stage breakdown runs CSV should be written")
assert_true(file.exists(artifact$paths$summary_csv),
            "stage breakdown summary CSV should be written")
assert_true(file.exists(artifact$paths$summary_md),
            "stage breakdown Markdown summary should be written")

cat("PASS fast CUDA stage breakdown\n")
