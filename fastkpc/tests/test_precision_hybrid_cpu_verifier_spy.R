source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

make_spy <- function(p_value, backend_name, calls_env, p_value_nonempty = p_value,
                     fail_nonempty = FALSE) {
  force(p_value)
  force(backend_name)
  force(calls_env)
  force(p_value_nonempty)
  force(fail_nonempty)
  function(data, x, y, S, ci_method, index, legacy_index, hsic_params,
           permutation_params, route, role) {
    calls_env$count <- calls_env$count + 1L
    calls_env$rows[[length(calls_env$rows) + 1L]] <- list(
      x = x, y = y, S = S, role = role, backend = backend_name
    )
    if (isTRUE(fail_nonempty) && length(S) > 0L) {
      stop("forced verifier failure", call. = FALSE)
    }
    list(
      p.value = if (length(S) == 0L) p_value else p_value_nonempty,
      residual_backend_executed = backend_name,
      ci_backend_executed = "spy-ci",
      setup_fingerprint = paste0("spy:", paste(S, collapse = "|")),
      p_source_used = paste0(role, ":", backend_name, "+spy-ci"),
      timings = list(ci_test_ms = 0)
    )
  }
}

set.seed(999)
data <- matrix(rnorm(50 * 3), 50, 3)
direct_calls <- new.env(parent = emptyenv())
direct_calls$count <- 0L
direct_calls$rows <- list()
primary_calls <- new.env(parent = emptyenv())
primary_calls$count <- 0L
primary_calls$rows <- list()
verifier_calls <- new.env(parent = emptyenv())
verifier_calls$count <- 0L
verifier_calls$rows <- list()

executors <- list(
  `direct-ci` = make_spy(0.049, "direct-ci-spy", direct_calls),
  fastSplineCPU = make_spy(0.049, "fastSplineCPU-spy", primary_calls),
  mgcvExtractCPUGCVBridge = make_spy(0.049, "mgcvExtractCPU-spy",
                               verifier_calls, p_value_nonempty = 0.051)
)
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
  precision_executors = executors,
  runtime_capabilities = caps
)

assert_true(primary_calls$count > 0L,
            "hybrid should execute primary fastSpline spy")
assert_true(verifier_calls$count > 0L,
            "hybrid near-alpha tests should execute verifier spy")
assert_true(hybrid$config$precision_execution_status == "data-plane-executed",
            "hybrid CPU slice should disclose data-plane execution")
assert_true(hybrid$config$backend_executed == "fastSplineCPU-spy",
            "hybrid backend_executed should record primary residual backend")
assert_true(hybrid$config$verifier_executed == "mgcvExtractCPU-spy",
            "hybrid verifier_executed should record verifier backend")
assert_true(!isTRUE(hybrid$skeleton$adjacency[1, 2]),
            "verifier p=0.051 at alpha=0.05 should delete edge 1-2")
assert_true(identical(as.integer(hybrid$skeleton$sepsets[[1]][[2]]), 3L),
            "hybrid sepset should come from verifier-used canonical test")

trace <- hybrid$diagnostics$precision_trace
verified <- trace[trace$verifier_executed == "mgcvExtractCPU-spy", ,
                  drop = FALSE]
assert_true(nrow(verified) > 0L, "trace should contain verifier rows")
assert_true(any(verified$primary_p_raw == 0.049 &
                  verified$verifier_p_raw == 0.051 &
                  verified$p_used == 0.051),
            "trace should preserve primary and verifier p-values")
assert_true(any(verified$decision_before_verify == FALSE &
                  verified$decision_after_verify == TRUE),
            "verifier should change the decision")
assert_true(any(verified$p_source_used ==
                  "verifier:mgcvExtractCPU-spy+spy-ci"),
            "p_source_used should name verifier source")

cat("PASS precision hybrid CPU verifier spy\n")
