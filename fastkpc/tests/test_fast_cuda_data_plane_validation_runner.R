source("fastkpc/R/fast_cuda_data_plane_validation.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP fast CUDA data-plane validation: FASTKPC_RUN_CUDA_TESTS != 1\n")
  quit(save = "no", status = 0)
}

build_fastkpc_native(rebuild = TRUE)
build_fastkpc_cuda_native(rebuild = FALSE)
if (!fastkpc_cuda_available()) {
  cat("SKIP fast CUDA data-plane validation: CUDA unavailable\n")
  quit(save = "no", status = 0)
}

out_dir <- tempfile("fast-cuda-data-plane-")
artifact <- fastkpc_run_fast_cuda_data_plane_validation(
  output_dir = out_dir,
  include_benchmark = FALSE
)
summary <- artifact$summary[1L, , drop = FALSE]

required <- c(
  "dcc_gamma_max_p_abs_diff", "dcc_gamma_decision_flips",
  "skeleton_ptable_exact", "route_violations",
  "conditional_ci_max_p_abs_diff", "conditional_ci_decision_flips",
  "conditional_ci_fallback_count", "e2e_graph_exact",
  "benchmark_included"
)
missing <- setdiff(required, names(summary))
assert_true(length(missing) == 0L,
            paste("missing validation summary fields:",
                  paste(missing, collapse = ",")))
assert_true(summary$dcc_gamma_max_p_abs_diff[[1L]] < 1e-10,
            "data-plane gate should pass dcc.gamma p-value parity")
assert_true(summary$dcc_gamma_decision_flips[[1L]] == 0L,
            "data-plane gate should have no dcc.gamma decision flips")
assert_true(isTRUE(summary$skeleton_ptable_exact[[1L]]),
            "data-plane gate should pass p-table replay parity")
assert_true(summary$route_violations[[1L]] == 0L,
            "data-plane gate should have no fast route violations")
assert_true(summary$conditional_ci_max_p_abs_diff[[1L]] < 1e-9,
            "data-plane gate should pass conditional CI parity")
assert_true(summary$conditional_ci_decision_flips[[1L]] == 0L,
            "data-plane gate should have no conditional CI decision flips")
assert_true(summary$conditional_ci_fallback_count[[1L]] == 0L,
            "data-plane gate should not use conditional CI fallback")
assert_true(isTRUE(summary$e2e_graph_exact[[1L]]),
            "data-plane gate should pass small e2e graph agreement")
assert_true(!isTRUE(summary$benchmark_included[[1L]]),
            "test runner should skip benchmark by default")
assert_true(file.exists(artifact$paths$summary_csv),
            "data-plane summary CSV should be written")
assert_true(file.exists(artifact$paths$summary_json),
            "data-plane summary JSON should be written")
assert_true(file.exists(artifact$paths$summary_md),
            "data-plane summary Markdown should be written")

cat("PASS fast CUDA data-plane validation runner\n")
