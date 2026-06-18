fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

source("fastkpc/R/true_batched_kernel_decision.R")

timing <- data.frame(
  backend = "mgcvExtractGPUFixedSP",
  mode = "same-setup-native-batch",
  true_batched_kernel = FALSE,
  targets_per_setup = 16L,
  mgcv_setup_cpu_ms = 5,
  gcv_score_ms = 5,
  linear_solve_ms = 80,
  ci_test_ms = 5,
  total_ms = 100
)

workload <- data.frame(
  dataset_id = "synthetic-unit",
  targets_per_setup_p95 = 16,
  mgcvExtractGPU_supported_tests = 80L,
  mgcvExtractGPU_unsupported_tests = 5L,
  near_alpha_tests_by_S_size = 20L
)

decision <- fastkpc_true_batched_kernel_decision(
  timing = timing,
  workload = workload
)
assert_true(decision$decision %in% c("proceed", "defer"),
            "decision should be proceed or defer")
assert_true(decision$decision == "proceed",
            "linear solve dominated high-multiplicity case should proceed")
assert_true(grepl("linear_solve_ms", decision$rationale, fixed = TRUE),
            "rationale should cite linear solve")

defer_timing <- timing
defer_timing$mgcv_setup_cpu_ms <- 80
defer_timing$linear_solve_ms <- 5
defer_timing$total_ms <- 100
defer_decision <- fastkpc_true_batched_kernel_decision(
  timing = defer_timing,
  workload = workload
)
assert_true(defer_decision$decision == "defer",
            "mgcv setup dominated case should defer")
assert_true(grepl("mgcv setup", defer_decision$rationale, fixed = TRUE),
            "defer rationale should cite mgcv setup")

out <- fastkpc_write_true_batched_kernel_decision(decision, output_dir = tempdir())
assert_true(file.exists(out$csv_path), "decision CSV should exist")
assert_true(file.exists(out$report_path), "decision report should exist")
txt <- paste(readLines(out$report_path, warn = FALSE), collapse = "\n")
assert_true(grepl("true batched mgcvExtractGPU kernel", txt, fixed = TRUE),
            "report should name kernel decision")

cat("PASS true batched kernel decision\n")
