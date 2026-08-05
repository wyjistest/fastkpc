source("fastkpc/R/validation_campaign.R")
source("fastkpc/R/report_writer.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

out_dir <- tempfile("fastkpc-dcc-perm-report-")
campaign <- run_fastkpc_validation_campaign(
  seeds = 331L,
  n_values = 36L,
  scenarios = "chain",
  engines = "cuda",
  residual_backends = "linear",
  residual_devices = "cpu",
  orientation_residual_devices = "cpu",
  schedulers = "legacy",
  ci_methods = "dcc.perm",
  permutation_params = list(
    replicates = 9L,
    seed = 331L,
    include_observed = TRUE
  ),
  alpha = 0.2,
  max_conditioning_size = 1L,
  legacy = FALSE,
  benchmark = FALSE
)

diagnostics <- campaign$ci_method_diagnostics
required <- c(
  "cuda_dcov_permutation_requested",
  "cuda_dcov_permutation_used",
  "ci_dcc_perm_tests",
  "ci_dcc_permutation_replicates",
  "ci_dcc_perm_cuda_tests",
  "ci_dcc_cuda_fallback_tests",
  "regrvonps_dcc_perm_tests",
  "regrvonps_dcc_permutation_replicates",
  "regrvonps_dcc_perm_cuda_tests"
)
assert_true(nrow(diagnostics) == 1L && all(required %in% names(diagnostics)),
            "dcc.perm campaign diagnostics schema is incomplete")
assert_true(isTRUE(diagnostics$cuda_dcov_permutation_requested[[1L]]) &&
              isTRUE(diagnostics$cuda_dcov_permutation_used[[1L]]) &&
              diagnostics$ci_dcc_perm_tests[[1L]] > 0L &&
              diagnostics$ci_dcc_perm_cuda_tests[[1L]] > 0L,
            "dcc.perm campaign did not record CUDA execution")

artifacts <- write_fastkpc_validation_report(campaign, out_dir)
written <- utils::read.csv(artifacts$ci_method_diagnostics_csv,
                           check.names = FALSE)
assert_true(all(required %in% names(written)),
            "ci_method_diagnostics.csv dropped dcc.perm fields")

input <- file.path(out_dir, "input.csv")
output <- file.path(out_dir, "result.rds")
utils::write.csv(
  generate_fastkpc_scenario("chain", 332L, 36L)$data,
  input,
  row.names = FALSE
)
cli <- system2(
  "Rscript",
  c(
    "fastkpc/tools/run_fast_kpc.R",
    "--input", input,
    "--output", output,
    "--engine", "cuda",
    "--graph-stage", "skeleton",
    "--residual-backend", "linear",
    "--residual-device", "cpu",
    "--scheduler", "legacy",
    "--ci-method", "dcc.perm",
    "--permutation-replicates", "9",
    "--permutation-seed", "332",
    "--max-conditioning-size", "1"
  ),
  stdout = TRUE,
  stderr = TRUE
)
assert_true(file.exists(output), "dcc.perm CLI did not write result RDS")
result <- readRDS(output)
assert_true(result$config$ci_backend == "cuda-dcov" &&
              isTRUE(result$config$cuda_dcov_permutation_used),
            "dcc.perm CLI result did not use cuda-dcov")
assert_true(any(grepl("cuda_dcov_permutation_used=TRUE", cli, fixed = TRUE)) &&
              any(grepl("ci_dcc_perm_cuda_tests=", cli, fixed = TRUE)) &&
              any(grepl("ci_dcc_permutation_replicates=", cli, fixed = TRUE)),
            "dcc.perm CLI stdout omitted execution receipts")

cat("test_dcc_permutation_campaign_report_cli.R: PASS\n")
