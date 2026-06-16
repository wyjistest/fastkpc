#!/usr/bin/env Rscript
source("fastkpc/R/validation_campaign.R")
source("fastkpc/R/report_writer.R")

parse_args <- function(args) {
  out <- list()
  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (!startsWith(key, "--")) stop("Invalid argument: ", key, call. = FALSE)
    name <- sub("^--", "", key)
    if (i == length(args)) stop("Missing value for ", key, call. = FALSE)
    out[[name]] <- args[[i + 1L]]
    i <- i + 2L
  }
  out
}

parse_csv <- function(value, default) {
  value <- value %||% default
  parts <- strsplit(value, ",", fixed = TRUE)[[1L]]
  parts[nzchar(parts)]
}

parse_bool <- function(value, default = TRUE) {
  value <- value %||% if (default) "TRUE" else "FALSE"
  toupper(value) %in% c("TRUE", "T", "1", "YES")
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  output_dir <- args[["output-dir"]]
  if (is.null(output_dir) || !nzchar(output_dir)) {
    stop("--output-dir is required", call. = FALSE)
  }
  hsic_params <- list(sig = as.numeric(args[["hsic-sig"]] %||% "1"))
  if (!is.null(args[["hsic-cuda-max-n"]])) {
    hsic_params$cuda_max_n <- as.integer(args[["hsic-cuda-max-n"]])
  }
  if (!is.null(args[["hsic-cuda-max-batch-pairs"]])) {
    hsic_params$cuda_max_batch_pairs <-
      as.integer(args[["hsic-cuda-max-batch-pairs"]])
  }
  if (!is.null(args[["hsic-cuda-memory-fallback"]])) {
    hsic_params$cuda_memory_fallback <-
      parse_bool(args[["hsic-cuda-memory-fallback"]], TRUE)
  }
  campaign <- run_fastkpc_validation_campaign(
    seeds = as.integer(parse_csv(args[["seeds"]], "11")),
    n_values = as.integer(parse_csv(args[["n-values"]], "80")),
    scenarios = parse_csv(args[["scenarios"]], "chain"),
    engines = parse_csv(args[["engines"]], "cpu"),
    residual_backends = parse_csv(args[["residual-backends"]], "fastSpline"),
    ci_methods = parse_csv(args[["ci-methods"]], "dcc.gamma"),
    hsic_params = hsic_params,
    permutation_params = list(
      replicates = as.integer(args[["permutation-replicates"]] %||% "100"),
      seed = if (is.null(args[["permutation-seed"]])) NULL else
        as.integer(args[["permutation-seed"]]),
      include_observed =
        parse_bool(args[["permutation-include-observed"]], TRUE)
    ),
    ci_diagnostics = parse_bool(args[["ci-diagnostics"]], TRUE),
    residual_devices = parse_csv(args[["residual-devices"]], "auto"),
    orientation_residual_devices =
      parse_csv(args[["orientation-residual-devices"]], "auto"),
    schedulers = parse_csv(args[["schedulers"]], "auto"),
    residual_batch_size = as.integer(args[["residual-batch-size"]] %||% "0"),
    orientation_batch_size =
      as.integer(args[["orientation-batch-size"]] %||% "0"),
    scheduler_diagnostics = parse_bool(args[["scheduler-diagnostics"]], TRUE),
    orientation_diagnostics =
      parse_bool(args[["orientation-diagnostics"]], TRUE),
    alpha = as.numeric(args[["alpha"]] %||% "0.2"),
    max_conditioning_size = as.integer(args[["max-conditioning-size"]] %||% "2"),
    legacy = parse_bool(args[["legacy"]], TRUE),
    benchmark = TRUE,
    output_dir = output_dir
  )
  write_fastkpc_validation_report(campaign, output_dir)
  cat("wrote report: ", output_dir, "\n", sep = "")
}

tryCatch(main(), error = function(e) {
  message(conditionMessage(e))
  quit(status = 1L)
})
