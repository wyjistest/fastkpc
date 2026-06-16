source("fastkpc/R/validation_campaign.R")
source("fastkpc/R/report_writer.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

out_dir <- file.path(tempdir(), "fastkpc_hsic_cuda_campaign_report_cli")
unlink(out_dir, recursive = TRUE, force = TRUE)

campaign <- run_fastkpc_validation_campaign(
  seeds = 31L,
  n_values = 48L,
  scenarios = "chain",
  engines = "cuda",
  residual_backends = "linear",
  residual_devices = "cuda",
  orientation_residual_devices = "cpu",
  schedulers = "legacy",
  ci_methods = "hsic.gamma",
  hsic_params = list(sig = 1),
  permutation_params = list(replicates = 20L, seed = 123L,
                            include_observed = TRUE),
  ci_diagnostics = TRUE,
  alpha = 0.2,
  max_conditioning_size = 1L,
  legacy = FALSE,
  benchmark = TRUE,
  output_dir = out_dir
)

assert_true(nrow(campaign$ci_method_diagnostics) > 0L,
            "campaign should record CI method diagnostics")
assert_true(all(campaign$ci_method_diagnostics$ci_backend == "cuda-hsic"),
            "campaign CI diagnostics should record cuda-hsic backend")
assert_true("ci_backend_requested" %in% names(campaign$ci_method_diagnostics),
            "campaign CI diagnostics should record requested backend")
assert_true("cuda_hsic_used" %in% names(campaign$ci_method_diagnostics),
            "campaign CI diagnostics should record CUDA HSIC usage")
assert_true("hsic_cuda_backend_diagnostics" %in% names(campaign),
            "campaign should include HSIC CUDA backend diagnostics artifact")
assert_true(nrow(campaign$hsic_cuda_backend_diagnostics) > 0L,
            "HSIC CUDA backend diagnostics should have rows")
assert_true(all(campaign$hsic_cuda_backend_diagnostics$ci_backend == "cuda-hsic"),
            "HSIC CUDA backend diagnostics should record cuda-hsic rows")
assert_true("hsic_cuda_cpu_fallbacks" %in% names(campaign),
            "campaign should include HSIC CUDA CPU fallback artifact")
assert_true("hsic_cuda_perf" %in% names(campaign),
            "campaign should include HSIC CUDA performance artifact")

artifacts <- write_fastkpc_validation_report(campaign, out_dir)
assert_true(file.exists(file.path(out_dir, "hsic_cuda_backend_diagnostics.csv")),
            "report should write hsic_cuda_backend_diagnostics.csv")
assert_true(file.exists(file.path(out_dir, "hsic_cuda_cpu_fallbacks.csv")),
            "report should write hsic_cuda_cpu_fallbacks.csv")
assert_true(file.exists(file.path(out_dir, "hsic_cuda_perf.csv")),
            "report should write hsic_cuda_perf.csv")
summary <- paste(readLines(artifacts$summary_md, warn = FALSE), collapse = "\n")
assert_true(grepl("## HSIC CUDA Backend", summary, fixed = TRUE),
            "summary markdown should include HSIC CUDA Backend section")

input_csv <- file.path(out_dir, "input.csv")
utils::write.csv(data.frame(
  a = seq(-2, 2, length.out = 48),
  b = sin(seq(-2, 2, length.out = 48)),
  c = cos(seq(-2, 2, length.out = 48)),
  d = rnorm(48)
), input_csv, row.names = FALSE)
single_rds <- file.path(out_dir, "single.rds")
single_out <- system2(
  "Rscript",
  c("fastkpc/tools/run_fast_kpc.R",
    "--input", input_csv,
    "--output", single_rds,
    "--engine", "cuda",
    "--graph-stage", "skeleton",
    "--residual-backend", "linear",
    "--residual-device", "cuda",
    "--scheduler", "legacy",
    "--ci-method", "hsic.gamma",
    "--ci-backend", "cuda"),
  stdout = TRUE,
  stderr = TRUE
)
assert_true(file.exists(single_rds),
            "run_fast_kpc.R should write output RDS")
assert_true(any(grepl("cuda_hsic_used=TRUE", single_out, fixed = TRUE)),
            "run_fast_kpc.R should print cuda_hsic_used=TRUE")
assert_true(any(grepl("ci_hsic_cuda_batches=", single_out, fixed = TRUE)),
            "run_fast_kpc.R should print CUDA HSIC batch count")

cli_dir <- file.path(out_dir, "cli_campaign")
cli_out <- system2(
  "Rscript",
  c("fastkpc/tools/run_validation_campaign.R",
    "--output-dir", cli_dir,
    "--engines", "cuda",
    "--residual-backends", "linear",
    "--residual-devices", "cuda",
    "--orientation-residual-devices", "cpu",
    "--schedulers", "legacy",
    "--ci-methods", "hsic.gamma",
    "--ci-backend", "cuda",
    "--seeds", "32",
    "--n-values", "48",
    "--scenarios", "chain",
    "--max-conditioning-size", "1",
    "--legacy", "FALSE"),
  stdout = TRUE,
  stderr = TRUE
)
assert_true(any(grepl("wrote report:", cli_out, fixed = TRUE)),
            "run_validation_campaign.R should report output directory")
assert_true(file.exists(file.path(cli_dir, "hsic_cuda_backend_diagnostics.csv")),
            "run_validation_campaign.R should write HSIC CUDA backend diagnostics")

cat("test_hsic_cuda_campaign_report_cli.R: PASS\n")
