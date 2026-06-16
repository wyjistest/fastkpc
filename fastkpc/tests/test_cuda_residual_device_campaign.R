source("fastkpc/R/validation_campaign.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

campaign <- run_fastkpc_validation_campaign(
  seeds = c(105),
  n_values = c(70),
  scenarios = c("chain", "additive"),
  engines = c("cuda"),
  residual_backends = c("fastSpline"),
  residual_devices = c("cpu", "cuda"),
  alpha = 0.2,
  max_conditioning_size = 1L,
  legacy = FALSE,
  benchmark = TRUE
)

assert_true("residual_device" %in% names(campaign$runs),
            "runs should include residual_device")
assert_true(all(c("cpu", "cuda") %in% campaign$runs$residual_device),
            "runs should include cpu and cuda residual devices")
assert_true("residual_device_diffs" %in% names(campaign),
            "campaign should include residual_device_diffs")
assert_true(is.data.frame(campaign$residual_device_diffs),
            "residual_device_diffs should be data.frame")
assert_true(nrow(campaign$residual_device_diffs) == 2L,
            "one residual-device diff row per scenario")
assert_true(all(campaign$residual_device_diffs$pdag_identical),
            "CPU/CUDA residual-device pdag should match")
assert_true(all(campaign$residual_device_diffs$skeleton_adjacency_identical),
            "CPU/CUDA residual-device skeleton should match")
assert_true(all(campaign$residual_device_diffs$max_abs_pmax_diff < 1e-7),
            "CPU/CUDA residual-device pMax diff should be tiny")
assert_true(campaign$summary$total_runs == nrow(campaign$runs),
            "summary total_runs should match")

cat("test_cuda_residual_device_campaign.R: PASS\n")
