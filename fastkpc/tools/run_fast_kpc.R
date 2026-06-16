#!/usr/bin/env Rscript
source("fastkpc/R/fast_kpc.R")

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

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  input <- args[["input"]]
  output <- args[["output"]]
  if (is.null(input) || !nzchar(input)) stop("--input is required", call. = FALSE)
  if (is.null(output) || !nzchar(output)) stop("--output is required", call. = FALSE)

  data <- utils::read.csv(input, check.names = FALSE)
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
      toupper(args[["hsic-cuda-memory-fallback"]]) %in%
        c("TRUE", "T", "1", "YES")
  }
  result <- fast_kpc(
    data,
    engine = args[["engine"]] %||% "auto",
    residual_backend = args[["residual-backend"]] %||% "fastSpline",
    ci_method = args[["ci-method"]] %||% "dcc.gamma",
    hsic_params = hsic_params,
    permutation_params = list(
      replicates = as.integer(args[["permutation-replicates"]] %||% "100"),
      seed = if (is.null(args[["permutation-seed"]])) NULL else
        as.integer(args[["permutation-seed"]]),
      include_observed =
        toupper(args[["permutation-include-observed"]] %||% "TRUE") %in%
          c("TRUE", "T", "1", "YES")
    ),
    ci_diagnostics =
      toupper(args[["ci-diagnostics"]] %||% "TRUE") %in%
        c("TRUE", "T", "1", "YES"),
    residual_device = args[["residual-device"]] %||% "auto",
    orientation_residual_device =
      args[["orientation-residual-device"]] %||% "auto",
    scheduler = args[["scheduler"]] %||% "auto",
    alpha = as.numeric(args[["alpha"]] %||% "0.2"),
    max_conditioning_size = as.integer(args[["max-conditioning-size"]] %||% "2"),
    graph_stage = args[["graph-stage"]] %||% "wanpdag",
    batch_size = as.integer(args[["batch-size"]] %||% "0"),
    residual_batch_size = as.integer(args[["residual-batch-size"]] %||% "0"),
    orientation_batch_size =
      as.integer(args[["orientation-batch-size"]] %||% "0"),
    scheduler_diagnostics =
      toupper(args[["scheduler-diagnostics"]] %||% "TRUE") %in%
        c("TRUE", "T", "1", "YES"),
    orientation_diagnostics =
      toupper(args[["orientation-diagnostics"]] %||% "TRUE") %in%
        c("TRUE", "T", "1", "YES")
  )
  saveRDS(result, output)
  cat("wrote: ", output, "\n", sep = "")
  scheduler_summary <- result$skeleton$scheduler_diagnostics$summary %||% list()
  if (!is.null(scheduler_summary$cuda_residual_true_batched_groups)) {
    cat("cuda_residual_true_batched_groups=",
        scheduler_summary$cuda_residual_true_batched_groups, "\n", sep = "")
    cat("cuda_residual_true_batched_fits=",
        scheduler_summary$cuda_residual_true_batched_fits, "\n", sep = "")
    cat("cuda_residual_single_fit_calls=",
        scheduler_summary$cuda_residual_single_fit_calls, "\n", sep = "")
    cat("cuda_residual_cpu_fallback_fits=",
        scheduler_summary$cuda_residual_cpu_fallback_fits, "\n", sep = "")
  }
  cat("ci_method=", result$config$ci_method, "\n", sep = "")
  cat("ci_backend=", result$config$ci_backend, "\n", sep = "")
  cat("cuda_hsic_used=", isTRUE(result$config$cuda_hsic_used), "\n", sep = "")
  ci_diag <- result$skeleton$ci_diagnostics %||% list()
  cat("ci_hsic_gamma_tests=", ci_diag$ci_hsic_gamma_tests %||% 0L, "\n", sep = "")
  cat("ci_hsic_perm_tests=", ci_diag$ci_hsic_perm_tests %||% 0L, "\n", sep = "")
  cat("ci_hsic_cuda_batches=", ci_diag$ci_hsic_cuda_batches %||% 0L, "\n", sep = "")
  cat("ci_hsic_cuda_pairs=", ci_diag$ci_hsic_cuda_pairs %||% 0L, "\n", sep = "")
  orientation_diag <- result$orientation$diagnostics %||% list()
  if (!is.null(orientation_diag$orientation_residual_device)) {
    cat("orientation_residual_device=",
        orientation_diag$orientation_residual_device, "\n", sep = "")
    cat("orientation_dcov_batches=",
        orientation_diag$orientation_dcov_batches %||% 0L, "\n", sep = "")
    cat("orientation_dcov_pairs=",
        orientation_diag$orientation_dcov_pairs %||% 0L, "\n", sep = "")
    cat("orientation_cuda_residual_fits=",
        orientation_diag$orientation_cuda_residual_fits %||% 0L, "\n", sep = "")
    cat("orientation_cpu_fallback_fits=",
        orientation_diag$orientation_cpu_fallback_fits %||% 0L, "\n", sep = "")
  }
}

tryCatch(main(), error = function(e) {
  message(conditionMessage(e))
  quit(status = 1L)
})
