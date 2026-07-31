source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!isTRUE(tryCatch(
  fastkpc_cuda_available(),
  error = function(error) FALSE
))) {
  cat("SKIP Phase 10 one-call compact-result cache: CUDA unavailable\n")
  quit(save = "no", status = 0L)
}

assert_true(
  exists("full_cuda_ci_one_call_cache_control_native", mode = "function"),
  "Phase 10 one-call cache control wrapper is missing"
)

normalize_sepsets <- function(value) {
  unname(lapply(unname(value), function(row) {
    unname(lapply(unname(row), function(entry) sort(as.integer(entry))))
  }))
}

same_result <- function(left, right) {
  identical(left$adjacency, right$adjacency) &&
    identical(normalize_sepsets(left$sepsets),
              normalize_sepsets(right$sepsets)) &&
    identical(as.integer(left$n.edgetests),
              as.integer(right$n.edgetests)) &&
    identical(left$tasks[, c(
      "level", "task_index", "edge_x", "edge_y", "x", "y", "S_key",
      "native_edge_deleted"
    )], right$tasks[, c(
      "level", "task_index", "edge_x", "edge_y", "x", "y", "S_key",
      "native_edge_deleted"
    )]) &&
    identical(left$tasks$p_used, right$tasks$p_used)
}

set.seed(6010)
n <- 72L
z <- stats::runif(n, -2, 2)
data <- cbind(
  x1 = sin(z) + stats::rnorm(n, sd = 0.08),
  x2 = cos(z) + stats::rnorm(n, sd = 0.08),
  x3 = z + stats::rnorm(n, sd = 0.08),
  x4 = z^2 + stats::rnorm(n, sd = 0.08)
)

run_candidate <- function() {
  precision_run_skeleton_full_cuda_native(
    data = data,
    alpha = 0.1,
    max_conditioning_size = 1L,
    index = 1,
    numCol = 35L,
    trace_level = "logical",
    compatible_cuda_strict = TRUE
  )
}

full_cuda_ci_one_call_cache_control_native("configure", 4096L)
full_cuda_ci_one_call_cache_control_native("reset")
cold <- run_candidate()
warm <- run_candidate()
conditional_tasks <- cold$tasks[nzchar(cold$tasks$S_key), , drop = FALSE]
unique_target_keys <- unique(c(
  paste(conditional_tasks$S_key, conditional_tasks$x, sep = ":"),
  paste(conditional_tasks$S_key, conditional_tasks$y, sep = ":")
))

assert_true(
  cold$summary$result_cache_request_count ==
      cold$summary$logical_tests_consumed &&
    cold$summary$result_cache_hit_count == 0L &&
    cold$summary$result_cache_miss_count ==
      cold$summary$logical_tests_consumed &&
    cold$summary$result_cache_insert_count ==
      cold$summary$logical_tests_consumed &&
    cold$summary$physical_tests_evaluated > 0L,
  "Phase 10 cold compact-result cache accounting is invalid"
)
assert_true(
  identical(cold$summary$scheduler, "cuda-level-target-prefill-host-v5") &&
    cold$summary$frontier_batch_count == sum(cold$levels$rounds) &&
    cold$summary$native_setup_cache_miss_count ==
      length(unique(conditional_tasks$S_key)) &&
    cold$summary$native_setup_count ==
      length(unique(conditional_tasks$S_key)) &&
    cold$summary$native_setup_cache_eviction_count == 0L &&
    cold$summary$native_setup_cache_request_count ==
      cold$summary$native_setup_cache_hit_count +
        cold$summary$native_setup_cache_miss_count,
  "Phase 10 cache-aware frontier scheduler is not active or bounded"
)
assert_true(
  cold$summary$target_cache_capacity == 131072L &&
    cold$summary$target_cache_request_count ==
      cold$summary$target_cache_hit_count +
        cold$summary$target_cache_miss_count &&
    cold$summary$target_cache_hit_count > 0L &&
    cold$summary$target_cache_miss_count == length(unique_target_keys) &&
    cold$summary$target_cache_insert_count == length(unique_target_keys) &&
    cold$summary$cuda_single_penalty_target_count ==
      length(unique_target_keys),
  "Phase 10 target optimizer cache did not reduce cold work to unique keys"
)
assert_true(
  warm$summary$result_cache_request_count ==
      warm$summary$logical_tests_consumed &&
    warm$summary$result_cache_hit_count ==
      warm$summary$logical_tests_consumed &&
    warm$summary$result_cache_miss_count == 0L &&
    warm$summary$result_cache_insert_count == 0L &&
    warm$summary$physical_tests_evaluated == 0L &&
    warm$summary$physical_residual_fits == 0L &&
    warm$summary$native_setup_count == 0L &&
    warm$summary$result_cache_warm_start_entries >=
      warm$summary$logical_tests_consumed,
  "Phase 10 warm compact-result cache did not eliminate physical work"
)
assert_true(
  same_result(cold, warm),
  "Phase 10 warm compact-result replay changed canonical semantics"
)

full_cuda_ci_one_call_cache_control_native("configure", 4L)
full_cuda_ci_one_call_cache_control_native("reset")
small_first <- run_candidate()
small_second <- run_candidate()
assert_true(
  same_result(cold, small_first) && same_result(cold, small_second) &&
    small_first$summary$result_cache_capacity == 4L &&
    small_first$summary$result_cache_eviction_count > 0L &&
    small_second$summary$result_cache_miss_count > 0L &&
    small_second$summary$physical_tests_evaluated > 0L,
  "Phase 10 compact-result eviction changed results or skipped reconstruction"
)

info <- full_cuda_ci_one_call_cache_control_native("info")
assert_true(
  info$capacity == 4L && info$entries <= info$capacity &&
    info$total_requests >= info$total_hits + info$total_misses &&
    info$total_evictions > 0L,
  "Phase 10 compact-result cache control info is inconsistent"
)

full_cuda_ci_one_call_cache_control_native("configure", 262144L)
full_cuda_ci_one_call_cache_control_native("reset")

cat(
  "PASS Phase 10 one-call compact-result cache; cold_physical=",
  cold$summary$physical_tests_evaluated,
  " warm_hits=", warm$summary$result_cache_hit_count,
  "\n", sep = ""
)
