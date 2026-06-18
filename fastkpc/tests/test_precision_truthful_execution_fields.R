source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

set.seed(991)
data <- matrix(rnorm(60 * 5), 60, 5)
caps <- list(
  R_version = "unsupported-R",
  mgcv_version = "unsupported-mgcv",
  cuda_available = FALSE,
  mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
  spectral_gcv_version = "single-penalty-spectral-gcv-v1",
  setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
)

compatible <- fast_kpc(
  data,
  alpha = 0.2,
  max_conditioning_size = 1,
  engine = "cpu",
  precision = "compatible",
  graph_stage = "skeleton",
  runtime_capabilities = caps
)

assert_true("backend_planned" %in% names(compatible$config),
            "config should distinguish backend_planned")
assert_true("backend_executed" %in% names(compatible$config),
            "config should distinguish backend_executed")
assert_true("verifier_planned" %in% names(compatible$config),
            "config should distinguish verifier_planned")
assert_true("verifier_executed" %in% names(compatible$config),
            "config should distinguish verifier_executed")
assert_true(compatible$config$backend_planned == "legacy-mgcv",
            "compatible unsupported route may plan legacy fallback")
assert_true(compatible$config$backend_executed == "fastSplineCPU",
            "current compatible data plane still executes fastSplineCPU")
assert_true(compatible$config$backend_used == compatible$config$backend_executed,
            "backend_used must describe executed backend, not planned backend")
assert_true(compatible$config$precision_execution_status ==
              "control-plane-only",
            "compatible must disclose that precision backend execution is not wired")

trace <- compatible$diagnostics$precision_trace
assert_true(all(trace$backend_planned == "legacy-mgcv"),
            "trace should record planned fallback backend")
assert_true(all(trace$backend_executed == "fastSplineCPU"),
            "trace should record actual executor backend")
assert_true(all(is.na(trace$p_source_used) |
                  trace$p_source_used == "not-recorded"),
            "trace must not invent p-value source before real CI trace exists")

cat("PASS precision truthful execution fields\n")
