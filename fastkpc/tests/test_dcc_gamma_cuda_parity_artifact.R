source("fastkpc/R/dcc_gamma_cuda_parity.R")
source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP dcc.gamma CUDA parity artifact: FASTKPC_RUN_CUDA_TESTS != 1\n")
  quit(save = "no", status = 0)
}

build_fastkpc_cuda_native(rebuild = FALSE)
if (!fastkpc_cuda_available()) {
  cat("SKIP dcc.gamma CUDA parity artifact: CUDA unavailable\n")
  quit(save = "no", status = 0)
}

out_dir <- tempfile("dcc-gamma-cuda-parity-")
artifact <- fastkpc_run_dcc_gamma_cuda_parity(output_dir = out_dir)
rows <- artifact$rows
summary <- artifact$summary[1L, , drop = FALSE]

required <- c(
  "scenario_id", "column_id", "n", "batch", "index", "legacy_index",
  "alpha", "expected_error", "status", "error_cpu", "error_cuda",
  "cpu_p", "cuda_p", "p_abs_diff", "log_p_diff",
  "cpu_statistic", "cuda_nV2", "stat_abs_diff",
  "cpu_mean", "cuda_mean", "mean_abs_diff",
  "cpu_variance", "cuda_variance", "variance_abs_diff",
  "cpu_delete", "cuda_delete", "decision_flip"
)
missing <- setdiff(required, names(rows))
assert_true(length(missing) == 0L,
            paste("missing parity fields:", paste(missing, collapse = ",")))
assert_true(file.exists(artifact$paths$csv),
            "dcc_gamma_cpu_cuda_parity.csv should be written")
assert_true(file.exists(artifact$paths$summary_csv),
            "summary CSV should be written")
assert_true(file.exists(artifact$paths$summary_json),
            "summary JSON should be written")
assert_true(file.exists(artifact$paths$summary_md),
            "summary Markdown should be written")

written <- utils::read.csv(artifact$paths$csv, stringsAsFactors = FALSE)
assert_true(nrow(written) == nrow(rows),
            "written parity CSV should contain all rows")
assert_true(all(c("ordinary_batch", "ties_batch", "small_n_valid",
                  "semantic_index_1_5", "legacy_index_ignored",
                  "near_alpha", "near_constant", "invalid_n",
                  "nonfinite") %in% unique(rows$scenario_id)),
            "artifact should include required dcc.gamma scenarios")

valid <- rows[rows$status == "ok", , drop = FALSE]
assert_true(nrow(valid) > 0L, "artifact should include valid parity rows")
assert_true(all(is.finite(valid$cpu_p) & valid$cpu_p >= 0 & valid$cpu_p <= 1),
            "CPU p-values should be finite probabilities")
assert_true(all(is.finite(valid$cuda_p) & valid$cuda_p >= 0 & valid$cuda_p <= 1),
            "CUDA p-values should be finite probabilities")
assert_true(max(valid$p_abs_diff, na.rm = TRUE) < 1e-10,
            "CPU/CUDA p-value drift should be below tolerance")
assert_true(max(valid$stat_abs_diff, na.rm = TRUE) < 1e-10,
            "CPU/CUDA statistic drift should be below tolerance")
assert_true(max(valid$mean_abs_diff, na.rm = TRUE) < 1e-10,
            "CPU/CUDA mean drift should be below tolerance")
assert_true(max(valid$variance_abs_diff, na.rm = TRUE) < 1e-10,
            "CPU/CUDA variance drift should be below tolerance")
assert_true(!any(valid$decision_flip %in% TRUE),
            "CPU/CUDA dcc.gamma decisions should not flip")

error_rows <- rows[rows$expected_error, , drop = FALSE]
assert_true(nrow(error_rows) > 0L, "artifact should include error scenarios")
assert_true(all(error_rows$status == "error_parity"),
            "error scenarios should fail consistently on CPU and CUDA")
assert_true(summary$error_mismatch_rows[[1L]] == 0L,
            "summary should report zero error mismatches")
assert_true(summary$decision_flips[[1L]] == 0L,
            "summary should report zero decision flips")

summary_md <- paste(readLines(artifact$paths$summary_md, warn = FALSE),
                    collapse = "\n")
assert_true(grepl("dcc.gamma CUDA Parity Summary", summary_md, fixed = TRUE),
            "summary Markdown should name the parity gate")

cat("PASS dcc.gamma CUDA parity artifact\n")
