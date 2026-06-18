source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

make_spy <- function(p_value, backend_name, calls_env) {
  force(p_value)
  force(backend_name)
  force(calls_env)
  function(data, x, y, S, ci_method, index, legacy_index, hsic_params,
           permutation_params, route, role) {
    calls_env$count <- calls_env$count + 1L
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

set.seed(992)
data <- matrix(rnorm(70 * 3), 70, 3)
direct_calls <- new.env(parent = emptyenv())
direct_calls$count <- 0L
primary_calls <- new.env(parent = emptyenv())
primary_calls$count <- 0L
verifier_calls <- new.env(parent = emptyenv())
verifier_calls$count <- 0L

caps <- list(
  R_version = "4.5.0",
  mgcv_version = "1.9-4",
  cuda_available = TRUE,
  mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
  spectral_gcv_version = "single-penalty-spectral-gcv-v1",
  setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
)

hybrid <- fast_kpc(
  data,
  alpha = 0.05,
  max_conditioning_size = 1,
  engine = "cpu",
  precision = "hybrid",
  tau = log(2),
  graph_stage = "skeleton",
  precision_executors = list(
    `direct-ci` = make_spy(0.001, "direct-ci-spy", direct_calls),
    fastSplineCPU = make_spy(0.001, "fastSplineCPU-spy", primary_calls),
    mgcvExtractCPUGCVBridge = make_spy(0.9, "mgcvExtractCPU-spy",
                                 verifier_calls)
  ),
  runtime_capabilities = caps
)

assert_true(hybrid$config$precision == "hybrid",
            "test should exercise hybrid precision")
assert_true(hybrid$config$backend_executed == "fastSplineCPU-spy",
            "hybrid data plane should execute fastSplineCPU primary")
assert_true(is.na(hybrid$config$verifier_executed),
            "far-primary hybrid run must not claim verifier execution")
assert_true(hybrid$config$precision_execution_status ==
              "data-plane-executed",
            "hybrid CPU slice should disclose data-plane execution")
assert_true(primary_calls$count > 0L,
            "hybrid should execute primary residual tests")
assert_true(verifier_calls$count == 0L,
            "far-primary p-values must not call verifier")

trace <- hybrid$diagnostics$precision_trace
assert_true(all(!is.na(trace$p_used)),
            "data-plane trace should record real p_used values")
assert_true(all(!trace$near_alpha_triggered),
            "primary p=0.001 should not trigger near-alpha verification")
assert_true(all(is.na(trace$verifier_p)),
            "non-triggered tests must not synthesize verifier_p")
assert_true(all(is.na(trace$verifier_executed)),
            "non-triggered tests must not claim verifier execution")
assert_true(all(grepl("primary:", trace$p_source_used)),
            "non-triggered hybrid trace should use primary p-value sources")

cat("PASS precision hybrid no fake verifier\n")
