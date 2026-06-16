source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/validation_campaign.R")
source("fastkpc/R/report_writer.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

scenario <- generate_fastkpc_scenario("additive", seed = 71, n = 90)
single <- fast_kpc(
  scenario$data,
  alpha = 0.2,
  max_conditioning_size = 1L,
  engine = "auto",
  residual_backend = "fastSpline",
  validate = TRUE,
  benchmark = TRUE,
  legacy = TRUE,
  seed = 71
)

assert_true(inherits(single, "fastkpc_result"), "single run should return fastkpc_result")
assert_true(single$config$engine_used %in% c("cpu", "cuda"), "engine_used should be concrete")
assert_true(is.list(single$validation), "validation section should exist")
assert_true(is.list(single$benchmark), "benchmark section should exist")

campaign <- run_fastkpc_validation_campaign(
  seeds = c(71),
  n_values = c(70),
  scenarios = c("chain", "fork", "collider"),
  engines = c("cpu", "cuda"),
  residual_backends = c("fastSpline"),
  alpha = 0.2,
  max_conditioning_size = 1L,
  legacy = TRUE,
  benchmark = TRUE
)

assert_true(all(campaign$runs$status == "ok"), "framework smoke campaign runs should be ok")
assert_true(any(campaign$cpu_cuda$pdag_identical), "some CPU/CUDA pdag rows should match")
output_dir <- tempfile("fastkpc-framework-report-")
artifacts <- write_fastkpc_validation_report(campaign, output_dir)
assert_true(file.exists(artifacts$summary_md), "framework report summary should exist")

cat("test_full_framework_smoke.R: PASS\n")
