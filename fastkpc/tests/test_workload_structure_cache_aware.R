source("fastkpc/R/workload_structure_stats.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

requests <- data.frame(
  setup_fingerprint = c("s1", "s1", "s1", "s2", "s2"),
  canonical_test_order_id = c(1L, 1L, 2L, 3L, 3L),
  target_id = c("x", "y", "x", "a", "b"),
  cache_hit = c(FALSE, FALSE, TRUE, FALSE, TRUE),
  device_solve_called = c(TRUE, TRUE, FALSE, TRUE, FALSE),
  S_size = c(1L, 1L, 1L, 2L, 2L),
  conditioning_level = c(1L, 1L, 1L, 2L, 2L),
  stringsAsFactors = FALSE
)

stats <- fastkpc_cache_aware_workload_stats(
  residual_requests = requests,
  dataset_id = "unit",
  n = 50L,
  p = 4L,
  alpha = 0.05,
  max_conditioning_level = 2L
)

required <- c("ci_tests_per_setup", "raw_residual_requests_per_setup",
              "unique_targets_per_setup", "uncached_targets_per_setup",
              "device_solve_calls_per_setup", "cache_hit_rate",
              "setup_fingerprint")
assert_true(all(required %in% names(stats)), "cache-aware fields missing")
s1 <- stats[stats$setup_fingerprint == "s1", , drop = FALSE]
assert_true(s1$ci_tests_per_setup == 2L, "s1 has two CI tests")
assert_true(s1$raw_residual_requests_per_setup == 3L, "s1 has three requests")
assert_true(s1$unique_targets_per_setup == 2L, "s1 has two unique targets")
assert_true(s1$uncached_targets_per_setup == 2L, "s1 has two uncached requests")
assert_true(s1$device_solve_calls_per_setup == 2L, "s1 has two device solves")

cat("PASS workload structure cache aware\n")
