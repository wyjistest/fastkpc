source("fastkpc/R/fast_cuda_conditional_ci_parity.R")
source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP fast CUDA conditional CI parity: FASTKPC_RUN_CUDA_TESTS != 1\n")
  quit(save = "no", status = 0)
}

build_fastkpc_cuda_native(rebuild = FALSE)
if (!fastkpc_cuda_available()) {
  cat("SKIP fast CUDA conditional CI parity: CUDA unavailable\n")
  quit(save = "no", status = 0)
}

out_dir <- tempfile("fast-cuda-conditional-ci-")
artifact <- fastkpc_run_fast_cuda_conditional_ci_parity(output_dir = out_dir)
rows <- artifact$rows
summary <- artifact$summary[1L, , drop = FALSE]

required <- c(
  "scenario_id", "n", "p", "x", "y", "S_key", "conditioning_size",
  "residual_rel_l2_x", "residual_rel_l2_y", "fitted_rel_l2_x",
  "fitted_rel_l2_y", "cpu_p", "cuda_p", "p_abs_diff",
  "log_p_diff", "cpu_statistic", "cuda_nV2", "stat_abs_diff",
  "cpu_mean", "cuda_mean", "mean_abs_diff", "cpu_variance",
  "cuda_variance", "variance_abs_diff", "decision_flip",
  "fallback_used_x", "fallback_used_y"
)
missing <- setdiff(required, names(rows))
assert_true(length(missing) == 0L,
            paste("missing conditional CI fields:",
                  paste(missing, collapse = ",")))
assert_true(all(c("one_s", "two_s", "translation", "scale", "ties") %in%
                  unique(rows$scenario_id)),
            "conditional CI artifact should cover required scenarios")
assert_true(all(rows$conditioning_size > 0L),
            "conditional CI parity should test non-empty conditioning sets")
assert_true(max(c(rows$residual_rel_l2_x, rows$residual_rel_l2_y),
                na.rm = TRUE) < 1e-7,
            "CUDA fastSpline residuals should match CPU residuals")
assert_true(max(c(rows$fitted_rel_l2_x, rows$fitted_rel_l2_y),
                na.rm = TRUE) < 1e-7,
            "CUDA fastSpline fitted values should match CPU fitted values")
assert_true(max(rows$p_abs_diff, na.rm = TRUE) < 1e-9,
            "conditional CI CPU/GPU p-value drift should be small")
assert_true(max(rows$stat_abs_diff, na.rm = TRUE) < 1e-8,
            "conditional CI CPU/GPU statistic drift should be small")
assert_true(!any(rows$decision_flip %in% TRUE),
            "conditional CI CPU/GPU decisions should not flip")
assert_true(summary$fallback_count[[1L]] == 0L,
            "conditional CI parity should not use CUDA residual fallback")
assert_true(file.exists(artifact$paths$csv),
            "conditional CI parity CSV should be written")
assert_true(file.exists(artifact$paths$summary_csv),
            "conditional CI summary CSV should be written")
assert_true(file.exists(artifact$paths$summary_md),
            "conditional CI summary Markdown should be written")

cat("PASS fast CUDA conditional CI parity artifact\n")
