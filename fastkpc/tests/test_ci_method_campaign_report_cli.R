source("fastkpc/R/validation_campaign.R")
source("fastkpc/R/report_writer.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

campaign <- run_fastkpc_validation_campaign(
  seeds = 217L,
  n_values = 36L,
  scenarios = "chain",
  engines = "cpu",
  residual_backends = "linear",
  residual_devices = "auto",
  orientation_residual_devices = "auto",
  schedulers = "auto",
  ci_methods = "hsic.gamma",
  hsic_params = list(sig = 1),
  alpha = 0.2,
  max_conditioning_size = 1,
  legacy = FALSE,
  benchmark = FALSE
)

assert_true("ci_method" %in% names(campaign$runs),
            "campaign runs should include ci_method")
assert_true(identical(as.character(campaign$runs$ci_method), "hsic.gamma"),
            "campaign should record requested HSIC gamma method")
assert_true(nrow(campaign$ci_method_diagnostics) == 1L,
            "campaign should include CI method diagnostics")
assert_true(campaign$ci_method_diagnostics$ci_hsic_gamma_tests[[1L]] > 0L,
            "CI diagnostics should record HSIC gamma tests")

out_dir <- tempfile("fastkpc-ci-method-report-")
artifacts <- write_fastkpc_validation_report(campaign, out_dir)
assert_true(file.exists(artifacts$ci_method_diagnostics_csv),
            "report should write ci_method_diagnostics.csv")
assert_true(file.exists(artifacts$ci_method_diffs_csv),
            "report should write ci_method_diffs.csv")
summary_text <- paste(readLines(artifacts$summary_md, warn = FALSE),
                      collapse = "\n")
assert_true(grepl("CI Method Diagnostics", summary_text, fixed = TRUE),
            "summary should include CI Method Diagnostics section")

input <- tempfile(fileext = ".csv")
utils::write.csv(generate_fastkpc_scenario("chain", 218L, 36L)$data,
                 input, row.names = FALSE)
output <- tempfile(fileext = ".rds")
cli <- system2(
  "Rscript",
  c("fastkpc/tools/run_fast_kpc.R",
    "--input", input,
    "--output", output,
    "--engine", "cpu",
    "--graph-stage", "skeleton",
    "--residual-backend", "linear",
    "--ci-method", "hsic.gamma",
    "--hsic-sig", "1",
    "--max-conditioning-size", "1"),
  stdout = TRUE,
  stderr = TRUE
)
assert_true(file.exists(output), "run_fast_kpc CLI should write output RDS")
assert_true(any(grepl("ci_method=hsic.gamma", cli, fixed = TRUE)),
            "run_fast_kpc CLI should print ci_method")

cat("test_ci_method_campaign_report_cli.R: PASS\n")
