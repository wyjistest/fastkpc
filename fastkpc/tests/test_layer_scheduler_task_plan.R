source("fastkpc/R/cuda_native.R")
source("fastkpc/R/legacy_runner.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

check_scheduler_diagnostics <- function(result, expected_scheduler) {
  diag <- result$scheduler_diagnostics
  assert_true(result$scheduler == expected_scheduler,
              paste("scheduler should be", expected_scheduler))
  assert_true(result$scheduler_requested %in% c(expected_scheduler, "auto"),
              "requested scheduler should be recorded")
  assert_true(is.list(diag), "scheduler diagnostics should be a list")
  assert_true(is.list(diag$summary), "scheduler summary should be present")
  assert_true(is.data.frame(diag$levels), "scheduler levels should be a data.frame")
  assert_true(is.data.frame(diag$batches), "scheduler batches should be a data.frame")
  assert_true(is.data.frame(diag$residuals), "scheduler residuals should be a data.frame")
  assert_true(diag$summary$tasks_planned >= sum(result$n.edgetests),
              "planned tasks should cover replayed tests")
  assert_true(diag$summary$tasks_ignored_after_delete ==
                diag$summary$tasks_evaluated - diag$summary$tests_replayed,
              "ignored count should match evaluated minus replayed")
  assert_true(sum(diag$levels$tasks_evaluated) == diag$summary$tasks_evaluated,
              "level evaluated counts should sum to summary")
  assert_true(sum(diag$levels$tests_replayed) == diag$summary$tests_replayed,
              "level replay counts should sum to summary")
  assert_true(diag$summary$tests_replayed == sum(result$n.edgetests),
              "summary replayed tests should match n.edgetests")
}

build_fastkpc_cuda_native(rebuild = TRUE)
scenario <- fastkpc_fixed_scenario()

legacy <- fast_skeleton_cuda_backend(
  scenario$data,
  alpha = scenario$alpha,
  max_conditioning_size = scenario$max_conditioning_size,
  scheduler = "legacy",
  scheduler_diagnostics = TRUE
)
check_scheduler_diagnostics(legacy, "legacy")

layer <- fast_skeleton_cuda_backend(
  scenario$data,
  alpha = scenario$alpha,
  max_conditioning_size = scenario$max_conditioning_size,
  scheduler = "layer",
  scheduler_diagnostics = TRUE
)
check_scheduler_diagnostics(layer, "layer")
assert_true(identical(layer$adjacency, legacy$adjacency),
            "layer scheduler adjacency should match legacy")
assert_true(max(abs(layer$pMax - legacy$pMax)) < 1e-8,
            "layer scheduler pMax should match legacy")
assert_true(identical(layer$n.edgetests, legacy$n.edgetests),
            "layer scheduler n.edgetests should match legacy")

cat("test_layer_scheduler_task_plan.R: PASS\n")
