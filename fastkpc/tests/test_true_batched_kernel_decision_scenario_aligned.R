source("fastkpc/R/true_batched_kernel_decision.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

timing <- data.frame(
  scenario_id = c("a", "b"),
  dataset_id = c("d1", "d2"),
  backend = "mgcvExtractGPUFixedSP",
  conditioning_level = c(1L, 1L),
  linear_solve_ms = c(90, 5),
  mgcv_setup_cpu_ms = c(5, 80),
  ci_test_ms = c(5, 5),
  total_ms = c(100, 100)
)
workload <- data.frame(
  scenario_id = c("a", "b"),
  dataset_id = c("d1", "d2"),
  backend = "mgcvExtractGPUFixedSP",
  conditioning_level = c(1L, 1L),
  uncached_targets_per_setup_p95 = c(10, 20),
  supported_wall_time_fraction = c(0.9, 0.9),
  evidence_runs = c(2L, 2L)
)

decision <- fastkpc_true_batched_kernel_decision_scenario_aligned(
  timing = timing,
  workload = workload,
  min_evidence_runs = 2L
)
assert_true(decision$decision %in% c("proceed", "defer", "insufficient-evidence"),
            "decision should be enumerated")
assert_true(decision$decision == "defer",
            "mixed evidence should defer because setup-dominated scenario has equal weight")
assert_true("scenario_id" %in% names(decision$evidence),
            "decision should retain scenario evidence")

cat("PASS true batched kernel scenario-aligned decision\n")
