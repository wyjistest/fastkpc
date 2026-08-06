source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!isTRUE(tryCatch(
  fastkpc_cuda_available(),
  error = function(error) FALSE
))) {
  cat("SKIP Phase 10 setup/optimizer pipeline: CUDA unavailable\n")
  quit(save = "no", status = 0L)
}

pipeline_environment <- "FASTKPC_PHASE10_SETUP_OPTIMIZER_PIPELINE"
old_pipeline_environment <- Sys.getenv(
  pipeline_environment, unset = NA_character_
)
on.exit({
  if (is.na(old_pipeline_environment)) {
    Sys.unsetenv(pipeline_environment)
  } else {
    do.call(
      Sys.setenv,
      setNames(list(old_pipeline_environment), pipeline_environment)
    )
  }
}, add = TRUE)

resource_snapshot <- function() {
  .Call("C_fixed_sp_cuda_test_resource_snapshot", PACKAGE = "fastkpc_cuda")
}

normalize_sepsets <- function(value) {
  unname(lapply(unname(value), function(row) {
    unname(lapply(unname(row), function(entry) sort(as.integer(entry))))
  }))
}

same_result <- function(left, right) {
  identical(left$adjacency, right$adjacency) &&
    identical(normalize_sepsets(left$sepsets),
              normalize_sepsets(right$sepsets)) &&
    identical(left$pMax, right$pMax) &&
    identical(as.integer(left$n.edgetests), as.integer(right$n.edgetests)) &&
    identical(left$levels[, setdiff(names(left$levels), "elapsed_ms")],
              right$levels[, setdiff(names(right$levels), "elapsed_ms")]) &&
    identical(left$tasks, right$tasks)
}

run_candidate <- function(data, pipeline) {
  full_cuda_ci_one_call_cache_control_native("reset")
  do.call(
    Sys.setenv,
    setNames(list(if (pipeline) "1" else "0"), pipeline_environment)
  )
  precision_run_skeleton_full_cuda_native(
    data = data,
    alpha = 0.1,
    max_conditioning_size = 3L,
    index = 1,
    numCol = 35L,
    trace_level = "logical",
    compatible_cuda_strict = TRUE
  )
}

set.seed(10093)
n <- 90L
p <- 10L
common <- stats::rnorm(n)
data <- sapply(seq_len(p), function(column) {
  sqrt(0.72) * common + sqrt(0.28) * stats::rnorm(n)
})
colnames(data) <- paste0("v", seq_len(p))

before <- resource_snapshot()
baseline <- run_candidate(data, FALSE)
candidate <- run_candidate(data, TRUE)
after <- resource_snapshot()

assert_true(
  same_result(baseline, candidate),
  "setup/optimizer pipeline changed strict one-call output bits"
)
assert_true(
  !isTRUE(baseline$summary$setup_optimizer_pipeline_enabled) &&
    baseline$summary$setup_optimizer_pipeline_window_count == 0L &&
    baseline$summary$setup_optimizer_pipeline_peak_pending_count == 0L &&
    baseline$summary$setup_optimizer_pipeline_producer_delay_us == 0L &&
    baseline$summary$setup_optimizer_pipeline_producer_delay_count == 0L &&
    baseline$summary$setup_optimizer_pipeline_producer_delay_ms == 0 &&
    baseline$summary$setup_optimizer_pipeline_prepare_ms == 0 &&
    baseline$summary$setup_optimizer_pipeline_device_prepare_ms == 0 &&
    baseline$summary$setup_optimizer_pipeline_wait_ms == 0 &&
    baseline$summary$setup_optimizer_pipeline_overlap_ms == 0 &&
    baseline$summary$setup_optimizer_pipeline_level_wall_ms == 0,
  "disabled setup/optimizer pipeline performed pipeline work"
)

summary <- candidate$summary
assert_true(
  isTRUE(summary$setup_optimizer_pipeline_enabled) &&
    summary$setup_optimizer_pipeline_window_count >= 2L &&
    summary$setup_optimizer_pipeline_window_count ==
      summary$cuda_single_penalty_optimizer_call_count +
        summary$cuda_multi_penalty_optimizer_call_count &&
    summary$setup_optimizer_pipeline_peak_pending_count == 1L &&
    summary$setup_optimizer_pipeline_producer_delay_us == 0L &&
    summary$setup_optimizer_pipeline_producer_delay_count == 0L &&
    summary$setup_optimizer_pipeline_producer_delay_ms == 0 &&
    summary$setup_optimizer_pipeline_prepare_ms > 0 &&
    summary$setup_optimizer_pipeline_device_prepare_ms > 0 &&
    summary$setup_optimizer_pipeline_wait_ms >= 0 &&
    summary$setup_optimizer_pipeline_overlap_ms >= 0 &&
    summary$setup_optimizer_pipeline_level_wall_ms > 0,
  "setup/optimizer pipeline receipt is malformed"
)
assert_true(
  summary$cuda_multi_penalty_prepared_build_count ==
      summary$cuda_multi_penalty_prepared_release_count &&
    summary$cuda_multi_penalty_optimizer_setup_count > 0 &&
    summary$cuda_multi_penalty_optimizer_iteration_sum ==
      baseline$summary$cuda_multi_penalty_optimizer_iteration_sum &&
    summary$cuda_multi_penalty_objective_call_sum ==
      baseline$summary$cuda_multi_penalty_objective_call_sum &&
    summary$cuda_multi_penalty_step_halving_sum ==
      baseline$summary$cuda_multi_penalty_step_halving_sum,
  "setup/optimizer pipeline changed optimizer work or lifecycle"
)

active_fields <- grep("_active_count$", names(before), value = TRUE)
assert_true(
  length(active_fields) > 0L &&
    identical(as.numeric(unlist(before[active_fields], use.names = FALSE)),
              as.numeric(unlist(after[active_fields], use.names = FALSE))),
  "setup/optimizer pipeline left a live CUDA resource after teardown"
)

cat(
  "PASS Phase 10 setup/optimizer pipeline; windows=",
  summary$setup_optimizer_pipeline_window_count,
  " overlap_ms=", summary$setup_optimizer_pipeline_overlap_ms,
  " wait_ms=", summary$setup_optimizer_pipeline_wait_ms,
  "\n",
  sep = ""
)
