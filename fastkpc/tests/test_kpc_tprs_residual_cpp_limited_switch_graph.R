source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_identical <- function(left, right, message) {
  if (!identical(left, right)) fail(message)
}

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP kpcTprsResidualCPP limited switch graph: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

set.seed(62321)
n <- 72L
s <- stats::runif(n, -2, 2)
data <- cbind(
  x = sin(s) + stats::rnorm(n, sd = 0.04),
  y = cos(s) + stats::rnorm(n, sd = 0.04),
  z = s + stats::rnorm(n, sd = 0.02),
  w = 0.35 * sin(s) - 0.2 * cos(s) + stats::rnorm(n, sd = 0.06)
)

caps_mgcv <- list(
  R_version = "4.5.0",
  mgcv_version = "1.9-4",
  cuda_available = FALSE,
  mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
  spectral_gcv_version = "single-penalty-spectral-gcv-v1",
  setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
)
caps_kpc <- modifyList(caps_mgcv, list(
  kpcTprsResidualCPP_supported = TRUE,
  kpcTprsResidualCPP_backend_version = "kpcTprsResidualCPP-v1"
))

reference <- fast_kpc(
  data,
  alpha = 0.05,
  max_conditioning_size = 1,
  engine = "cpu",
  precision = "compatible",
  graph_stage = "skeleton",
  runtime_capabilities = caps_mgcv
)
candidate <- fast_kpc(
  data,
  alpha = 0.05,
  max_conditioning_size = 1,
  engine = "cpu",
  precision = "compatible",
  graph_stage = "skeleton",
  runtime_capabilities = caps_kpc
)

assert_true(candidate$config$backend_planned == "kpcTprsResidualCPP",
            "limited switch should plan kpcTprsResidualCPP")
assert_true(candidate$config$backend_executed == "kpcTprsResidualCPP",
            "limited switch should execute kpcTprsResidualCPP")
assert_identical(candidate$skeleton$adjacency, reference$skeleton$adjacency,
                 "limited switch adjacency should match mgcv reference")
assert_identical(candidate$skeleton$n.edgetests, reference$skeleton$n.edgetests,
                 "limited switch n.edgetests should match mgcv reference")
assert_true(max(abs(candidate$skeleton$pMax - reference$skeleton$pMax),
                na.rm = TRUE) < 1e-4,
            "limited switch pMax drift should be small")

trace <- candidate$diagnostics$precision_trace
conditional <- trace[nzchar(trace$S_key), , drop = FALSE]
assert_true(nrow(conditional) > 0L,
            "limited switch graph test should include conditional rows")
assert_true(all(conditional$backend_requested == "kpcTprsResidualCPP"),
            "conditional rows should request kpcTprsResidualCPP")
assert_true(all(conditional$backend_executed == "kpcTprsResidualCPP"),
            "conditional rows should execute kpcTprsResidualCPP")

cat("PASS kpcTprsResidualCPP limited switch graph\n")
