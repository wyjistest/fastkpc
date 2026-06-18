fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

source("fastkpc/R/workload_structure_stats.R")

test_plan <- data.frame(
  canonical_test_order_id = seq_len(8),
  x = c(1, 1, 2, 2, 3, 3, 4, 4),
  y = c(2, 3, 3, 4, 4, 5, 5, 6),
  S_key = c("1", "1", "1,2", "1,2", "1,2,3", "1,2,3", "2", "2"),
  S_size = c(1L, 1L, 2L, 2L, 3L, 3L, 1L, 1L),
  conditioning_level = c(1L, 1L, 2L, 2L, 3L, 3L, 1L, 1L),
  near_alpha = c(TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, FALSE, TRUE),
  verifier_called = c(TRUE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, TRUE),
  mgcvExtractGPU_supported = c(TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, TRUE, TRUE),
  stringsAsFactors = FALSE
)

stats <- fastkpc_workload_structure_stats(
  test_plan = test_plan,
  dataset_id = "synthetic-unit",
  n = 100L,
  p = 6L,
  alpha = 0.05,
  max_conditioning_level = 3L
)

required <- c(
  "dataset_id", "n", "p", "alpha", "max_conditioning_level",
  "conditioning_level", "num_ci_tests", "num_unique_S",
  "num_same_setup_groups", "targets_per_setup_p50",
  "targets_per_setup_p95", "targets_per_setup_max",
  "num_tests_by_S_size", "runtime_by_S_size",
  "near_alpha_tests_by_S_size", "verifier_calls_by_S_size",
  "mgcvExtractGPU_supported_tests", "mgcvExtractGPU_unsupported_tests"
)
missing <- setdiff(required, names(stats))
assert_true(length(missing) == 0L,
            paste("missing workload stats fields:", paste(missing, collapse = ", ")))
assert_true(sum(stats$num_ci_tests) == 8L,
            "stats should preserve total CI test count")
assert_true(sum(stats$mgcvExtractGPU_unsupported_tests) == 2L,
            "stats should count unsupported GPU tests")

out <- fastkpc_write_workload_structure_stats(stats, output_dir = tempdir())
assert_true(file.exists(out$csv_path), "workload stats CSV should exist")
assert_true(file.exists(out$report_path), "workload stats report should exist")
txt <- paste(readLines(out$report_path, warn = FALSE), collapse = "\n")
assert_true(grepl("same-setup multiplicity", txt, fixed = TRUE),
            "report should discuss same-setup multiplicity")
assert_true(grepl("|S| > 2", txt, fixed = TRUE),
            "report should discuss high-order S")

cat("PASS workload structure stats\n")
