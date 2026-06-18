fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

source("fastkpc/R/precision_ladder_timing.R")

rows <- rbind(
  fastkpc_precision_ladder_timing_row(
    backend = "mgcvExtractGPUFixedSP",
    mode = "fixed-sp",
    solve_source = "cuda-fixed-sp",
    native_gpu_solve_used = TRUE,
    true_batched_kernel = FALSE,
    targets_per_setup = 8L,
    linear_solve_ms = 90,
    total_ms = 120
  ),
  fastkpc_precision_ladder_timing_row(
    backend = "legacy-mgcv",
    mode = "reference",
    solve_source = "mgcv",
    mgcv_setup_cpu_ms = 80,
    total_ms = 100
  )
)

out <- fastkpc_write_precision_ladder_timing_report(rows, output_dir = tempdir())
assert_true(file.exists(out$csv_path), "timing CSV should exist")
assert_true(file.exists(out$report_path), "timing report should exist")

written <- utils::read.csv(out$csv_path, stringsAsFactors = FALSE)
assert_true("bottleneck" %in% names(written), "timing CSV should include bottleneck")
assert_true(any(written$bottleneck == "linear_solve_dominated"),
            "timing CSV should classify linear solve bottleneck")

txt <- paste(readLines(out$report_path, warn = FALSE), collapse = "\n")
assert_true(grepl("true batched solve kernel", txt, fixed = TRUE),
            "report should mention true batched solve kernel")
assert_true(grepl("same-setup batch", txt, fixed = TRUE),
            "report should mention same-setup batch")
assert_true(grepl("linear_solve_dominated", txt, fixed = TRUE),
            "report should include bottleneck labels")

cat("PASS precision ladder timing report\n")
