source("fastkpc/R/native_cuda_precision_parity.R")
source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP native CUDA precision parity artifact: FASTKPC_RUN_CUDA_TESTS != 1\n")
  quit(save = "no", status = 0)
}

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP native CUDA precision parity artifact: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

build_fastkpc_cuda_native(rebuild = FALSE)
if (!fastkpc_cuda_available()) {
  cat("SKIP native CUDA precision parity artifact: CUDA unavailable\n")
  quit(save = "no", status = 0)
}

set.seed(2104)
n <- 64
z <- stats::runif(n, -2, 2)
x <- sin(z) + stats::rnorm(n, sd = 0.07)
y <- cos(z) + stats::rnorm(n, sd = 0.07)
w <- z + stats::rnorm(n, sd = 0.07)
data <- cbind(x, y, w)
out_dir <- tempfile("native-cuda-parity-")

artifact <- fastkpc_run_native_cuda_precision_parity(
  data = data,
  x = 1L,
  y = 2L,
  S = 3L,
  alpha = 0.05,
  output_dir = out_dir
)
row <- artifact$parity[1L, , drop = FALSE]

required <- c(
  "cpu_selected_sp_x", "cpu_selected_sp_y",
  "gpu_selected_sp_x", "gpu_selected_sp_y",
  "selected_sp_x", "selected_sp_y",
  "selected_grid_index_x", "selected_grid_index_y",
  "gcv_score_x", "gcv_score_y", "edf_x", "edf_y",
  "coefficient_rel_l2_x", "coefficient_rel_l2_y",
  "fitted_rel_l2_x", "fitted_rel_l2_y",
  "residual_rel_l2_x", "residual_rel_l2_y",
  "ci_statistic_cpu", "ci_statistic_gpu",
  "p_value_cpu", "p_value_gpu",
  "decision_cpu", "decision_gpu",
  "adjacency_identical", "first_sepset_identical",
  "pmax_max_abs_diff"
)
missing <- setdiff(required, names(row))
assert_true(length(missing) == 0L,
            paste("missing parity fields:", paste(missing, collapse = ",")))
assert_true(file.exists(artifact$path),
            "native_cuda_precision_parity.csv should be written")
written <- utils::read.csv(artifact$path)
assert_true(nrow(written) == 1L,
            "parity CSV should contain one scenario row")
assert_true(isTRUE(artifact$gpu_pair$fit$native_gpu_solve_used_x) &&
              isTRUE(artifact$gpu_pair$fit$native_gpu_solve_used_y),
            "parity artifact should use native GPU solves")
assert_true(isTRUE(artifact$gpu_pair$fit$same_setup_pair_batch_used),
            "parity artifact should use same-setup pair batch")
assert_true(all(is.finite(as.numeric(row[, c(
  "cpu_selected_sp_x", "cpu_selected_sp_y",
  "gpu_selected_sp_x", "gpu_selected_sp_y",
  "selected_sp_x", "selected_sp_y", "gcv_score_x", "gcv_score_y",
  "coefficient_rel_l2_x", "coefficient_rel_l2_y",
  "fitted_rel_l2_x", "fitted_rel_l2_y",
  "residual_rel_l2_x", "residual_rel_l2_y",
  "p_value_cpu", "p_value_gpu", "pmax_max_abs_diff"
)]))),
            "numeric parity metrics should be finite")
assert_true(identical(row$decision_cpu, row$decision_gpu),
            "CPU/GPU parity row should preserve edge decision")
assert_true(isTRUE(row$adjacency_identical),
            "CPU/GPU compatible skeleton adjacency should match")

cat("PASS native CUDA precision parity artifact\n")
