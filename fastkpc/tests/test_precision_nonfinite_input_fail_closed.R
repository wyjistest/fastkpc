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

expect_precision_finite_error <- function(expr, label) {
  error_message <- NULL
  tryCatch(
    force(expr),
    error = function(e) error_message <<- conditionMessage(e)
  )
  assert_true(!is.null(error_message),
              paste(label, "should fail closed on non-finite input"))
  assert_true(grepl("precision data plane", error_message, fixed = TRUE),
              paste(label, "error should name precision data plane"))
  assert_true(grepl("finite input", error_message, fixed = TRUE),
              paste(label, "error should require finite input"))
}

caps_cpu <- list(
  R_version = "4.5.0",
  mgcv_version = "1.9-4",
  cuda_available = FALSE,
  mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
  spectral_gcv_version = "single-penalty-spectral-gcv-v1",
  setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
)

set.seed(1507)
base <- matrix(rnorm(48 * 3), 48, 3)

for (case in c("NA", "NaN", "Inf")) {
  bad <- base
  bad[2, 1] <- switch(case, "NA" = NA_real_, "NaN" = NaN, "Inf" = Inf)
  expect_precision_finite_error(
    fast_kpc(
      bad,
      alpha = 0.05,
      max_conditioning_size = 1,
      engine = "cpu",
      precision = "compatible",
      graph_stage = "skeleton",
      runtime_capabilities = caps_cpu
    ),
    paste("compatible", case)
  )
}

hybrid_bad <- base
hybrid_bad[3, 2] <- NaN
expect_precision_finite_error(
  fast_kpc(
    hybrid_bad,
    alpha = 0.05,
    max_conditioning_size = 1,
    engine = "cpu",
    precision = "hybrid",
    tau = Inf,
    graph_stage = "skeleton",
    runtime_capabilities = caps_cpu,
    precision_executors = list(
      `direct-ci` = make_spy(0.001, "direct-ci-spy"),
      fastSplineCPU = make_spy(0.049, "fastSplineCPU-spy"),
      mgcvExtractCPUGCVBridge = make_spy(0.051, "mgcvExtractCPU-spy"),
      `legacy-mgcv` = make_spy(0.9, "legacy-mgcv-spy")
    )
  ),
  "hybrid NaN"
)

cat("PASS precision non-finite input fail closed\n")
