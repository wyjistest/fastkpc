source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

set.seed(992)
data <- matrix(rnorm(70 * 6), 70, 6)
hybrid <- fast_kpc(
  data,
  alpha = 0.2,
  max_conditioning_size = 1,
  engine = "cpu",
  precision = "hybrid",
  tau = log(2),
  graph_stage = "skeleton"
)

assert_true(hybrid$config$precision == "hybrid",
            "test should exercise hybrid precision")
assert_true(hybrid$config$backend_executed == "fastSplineCPU",
            "current hybrid data plane executes fastSplineCPU primary")
assert_true(is.na(hybrid$config$verifier_executed),
            "hybrid must not claim a verifier executed before data plane wiring")
assert_true(hybrid$config$precision_execution_status ==
              "control-plane-only",
            "hybrid must disclose control-plane-only status")

trace <- hybrid$diagnostics$precision_trace
assert_true(all(is.na(trace$verifier_p)),
            "trace must not synthesize verifier_p")
assert_true(all(is.na(trace$p_used)),
            "trace must not synthesize p_used")
assert_true(all(is.na(trace$p_source_used) |
                  trace$p_source_used == "not-recorded"),
            "trace must not claim verifier p_source_used")
assert_true(!any(trace$p_source_used %in% c("mgcvExtractGPUGCV",
                                            "mgcvExtractGPU",
                                            "legacy-mgcv"),
                 na.rm = TRUE),
            "hybrid trace must not claim verifier/legacy p-value source")

cat("PASS precision hybrid no fake verifier\n")
