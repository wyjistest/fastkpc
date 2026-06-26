source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_equal <- function(actual, expected, message) {
  if (!identical(actual, expected)) {
    fail(paste0(message, "\nactual: ", paste(actual, collapse = ","),
                "\nexpected: ", paste(expected, collapse = ",")))
  }
}

make_table_executor <- function(calls_env) {
  function(data, x, y, S, ci_method, index, legacy_index, hsic_params,
           permutation_params, route, role) {
    S_key <- if (length(S) == 0L) "" else paste(S, collapse = "|")
    key <- paste(x, y, S_key, sep = ":")
    calls_env$keys <- c(calls_env$keys, key)
    p <- switch(key,
      "1:2:" = 0.01,
      "2:1:" = 0.01,
      "1:3:" = 0.01,
      "3:1:" = 0.01,
      "2:3:" = 0.01,
      "3:2:" = 0.01,
      "1:2:3" = 0.06,
      "1:3:2" = 0.06,
      "2:3:1" = 0.06,
      0.01
    )
    list(
      p.value = p,
      residual_backend_executed = "table-backend",
      ci_backend_executed = "table-ci",
      setup_fingerprint = paste0("table:", S_key),
      p_source_used = "primary:table-backend+table-ci",
      timings = list(ci_test_ms = 0)
    )
  }
}

set.seed(998)
data <- matrix(rnorm(40 * 3), 40, 3)
calls <- new.env(parent = emptyenv())
calls$keys <- character()
executors <- list(
  `direct-ci` = make_table_executor(calls),
  mgcvExtractCPUGCVBridge = make_table_executor(calls)
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
  runtime_capabilities = caps,
  precision_trace_level = "full"
)

expected_keys <- c(
  "1:2:", "2:1:", "1:3:", "3:1:", "2:3:", "3:2:",
  "1:2:3", "1:3:2", "2:3:1"
)
assert_equal(calls$keys, expected_keys,
             "precision scheduler should preserve canonical test order")

trace <- result$diagnostics$precision_trace
trace_keys <- paste(trace$x, trace$y, trace$S_key, sep = ":")
assert_equal(trace_keys, expected_keys,
             "trace order should match canonical executor order")
assert_true(!isTRUE(result$skeleton$adjacency[1, 2]),
            "edge 1-2 should be deleted by first canonical separating set")
assert_true(identical(as.integer(result$skeleton$sepsets[[1]][[2]]), 3L),
            "edge 1-2 sepset should be S=3")
assert_true(result$skeleton$pMax[1, 2] == 0.06,
            "pMax should record max p-value for edge 1-2")

cat("PASS precision canonical order table\n")
