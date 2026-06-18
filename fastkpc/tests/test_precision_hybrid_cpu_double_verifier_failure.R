source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

make_spy <- function(p_value, backend_name, calls_env, fail_nonempty = FALSE) {
  force(p_value)
  force(backend_name)
  force(calls_env)
  force(fail_nonempty)
  function(data, x, y, S, ci_method, index, legacy_index, hsic_params,
           permutation_params, route, role) {
    calls_env$count <- calls_env$count + 1L
    if (isTRUE(fail_nonempty) && length(S) > 0L) {
      stop(paste("forced failure", backend_name), call. = FALSE)
    }
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

set.seed(1001)
data <- matrix(rnorm(48 * 3), 48, 3)
direct_calls <- new.env(parent = emptyenv())
direct_calls$count <- 0L
primary_calls <- new.env(parent = emptyenv())
primary_calls$count <- 0L
verifier_calls <- new.env(parent = emptyenv())
verifier_calls$count <- 0L
legacy_calls <- new.env(parent = emptyenv())
legacy_calls$count <- 0L

caps <- list(
  R_version = "4.5.0",
  mgcv_version = "1.9-4",
  cuda_available = TRUE,
  mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
  spectral_gcv_version = "single-penalty-spectral-gcv-v1",
  setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
)

err <- tryCatch({
  fast_kpc(
    data,
    alpha = 0.05,
    max_conditioning_size = 1,
    engine = "cpu",
    precision = "hybrid",
    tau = log(2),
    graph_stage = "skeleton",
    precision_executors = list(
      `direct-ci` = make_spy(0.049, "direct-ci-spy", direct_calls),
      fastSplineCPU = make_spy(0.049, "fastSplineCPU-spy", primary_calls),
      mgcvExtractGPUGCV = make_spy(0.9, "mgcvExtractCPU-spy",
                                   verifier_calls, fail_nonempty = TRUE),
      `legacy-mgcv` = make_spy(0.9, "legacy-mgcv-spy",
                               legacy_calls, fail_nonempty = TRUE)
    ),
    runtime_capabilities = caps
  )
  NULL
}, error = function(e) e)

assert_true(inherits(err, "error"),
            "double verifier failure should raise an explicit error")
assert_true(grepl("fallback legacy-mgcv failed", conditionMessage(err),
                  fixed = TRUE),
            "error should name failed legacy fallback")
assert_true(primary_calls$count > 0L,
            "primary should run before verifier failure")
assert_true(verifier_calls$count > 0L,
            "mgcvExtract verifier should be attempted")
assert_true(legacy_calls$count > 0L,
            "legacy verifier fallback should be attempted")

cat("PASS precision hybrid CPU double verifier failure\n")
