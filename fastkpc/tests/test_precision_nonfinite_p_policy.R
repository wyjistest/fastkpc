source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

make_spy <- function(p_value, backend_name) {
  force(p_value)
  force(backend_name)
  function(data, x, y, S, ci_method, index, legacy_index, hsic_params,
           permutation_params, route, role) {
    list(
      p.value = p_value,
      residual_backend_executed = backend_name,
      ci_backend_executed = "spy-ci",
      setup_fingerprint = paste0("spy:", paste(S, collapse = "|")),
      p_source_used = paste0(role, ":", backend_name, "+spy-ci"),
      timings = list(ci_test_ms = 0)
    )
  }
}

set.seed(996)
data <- matrix(rnorm(45 * 3), 45, 3)
executors <- list(
  `direct-ci` = make_spy(NA_real_, "direct-ci-spy"),
  fastSplineCPU = make_spy(NA_real_, "fastSplineCPU-spy"),
  mgcvExtractGPUGCV = make_spy(NA_real_, "mgcvExtractCPU-spy")
)
caps <- list(
  R_version = "4.5.0",
  mgcv_version = "1.9-4",
  cuda_available = TRUE,
  mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
  spectral_gcv_version = "single-penalty-spectral-gcv-v1",
  setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
)

result <- fast_kpc(
  data,
  alpha = 0.05,
  max_conditioning_size = 1,
  engine = "cpu",
  precision = "compatible",
  graph_stage = "skeleton",
  precision_executors = executors,
  runtime_capabilities = caps
)

trace <- result$diagnostics$precision_trace
required <- c("p_raw", "p_was_nonfinite", "nonfinite_action")
missing <- setdiff(required, names(trace))
assert_true(length(missing) == 0L,
            paste("trace missing nonfinite policy fields:",
                  paste(missing, collapse = ", ")))
assert_true(all(is.na(trace$p_raw)),
            "p_raw should preserve NA returned by executor")
assert_true(all(trace$p_was_nonfinite),
            "p_was_nonfinite should mark all NA p-values")
assert_true(all(trace$nonfinite_action == "na-delete-use-1"),
            "NA p-values should document NAdelete action")
assert_true(all(trace$p_used == 1),
            "NAdelete policy should use p=1 for edge deletion")

cat("PASS precision nonfinite p policy\n")
