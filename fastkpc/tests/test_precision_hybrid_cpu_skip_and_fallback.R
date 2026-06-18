source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

make_spy <- function(p_value, backend_name, calls_env, fail = FALSE) {
  force(p_value)
  force(backend_name)
  force(calls_env)
  force(fail)
  function(data, x, y, S, ci_method, index, legacy_index, hsic_params,
           permutation_params, route, role) {
    calls_env$count <- calls_env$count + 1L
    calls_env$rows[[length(calls_env$rows) + 1L]] <- list(
      x = x, y = y, S = S, role = role, backend = backend_name
    )
    if (isTRUE(fail)) stop("forced failure", call. = FALSE)
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

caps <- list(
  R_version = "4.5.0",
  mgcv_version = "1.9-4",
  cuda_available = TRUE,
  mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
  spectral_gcv_version = "single-penalty-spectral-gcv-v1",
  setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
)

set.seed(1000)
data <- matrix(rnorm(48 * 3), 48, 3)

primary_calls <- new.env(parent = emptyenv())
primary_calls$count <- 0L
primary_calls$rows <- list()
verifier_calls <- new.env(parent = emptyenv())
verifier_calls$count <- 0L
verifier_calls$rows <- list()
legacy_calls <- new.env(parent = emptyenv())
legacy_calls$count <- 0L
legacy_calls$rows <- list()

far <- fast_kpc(
  data,
  alpha = 0.05,
  max_conditioning_size = 1,
  engine = "cpu",
  precision = "hybrid",
  tau = log(2),
  graph_stage = "skeleton",
  precision_executors = list(
    `direct-ci` = make_spy(0.001, "direct-ci-spy", primary_calls),
    fastSplineCPU = make_spy(0.001, "fastSplineCPU-spy", primary_calls),
    mgcvExtractGPUGCV = make_spy(0.9, "mgcvExtractCPU-spy", verifier_calls),
    `legacy-mgcv` = make_spy(0.9, "legacy-mgcv-spy", legacy_calls)
  ),
  runtime_capabilities = caps
)

assert_true(primary_calls$count > 0L,
            "far-primary scenario should run primary")
assert_true(verifier_calls$count == 0L,
            "primary p=0.001 should not trigger verifier")
assert_true(legacy_calls$count == 0L,
            "primary p=0.001 should not trigger legacy fallback")
assert_true(all(!far$diagnostics$precision_trace$near_alpha_triggered),
            "trace should record no near-alpha triggers")

primary_calls$count <- 0L
primary_calls$rows <- list()
verifier_calls$count <- 0L
verifier_calls$rows <- list()
legacy_calls$count <- 0L
legacy_calls$rows <- list()

fallback <- fast_kpc(
  data,
  alpha = 0.05,
  max_conditioning_size = 1,
  engine = "cpu",
  precision = "hybrid",
  tau = log(2),
  graph_stage = "skeleton",
  precision_executors = list(
    `direct-ci` = make_spy(0.049, "direct-ci-spy", primary_calls),
    fastSplineCPU = make_spy(0.049, "fastSplineCPU-spy", primary_calls),
    mgcvExtractGPUGCV = make_spy(0.9, "mgcvExtractCPU-spy",
                                 verifier_calls, fail = TRUE),
    `legacy-mgcv` = make_spy(0.051, "legacy-mgcv-spy", legacy_calls)
  ),
  runtime_capabilities = caps
)

assert_true(verifier_calls$count > 0L,
            "near-alpha scenario should attempt mgcvExtract verifier")
assert_true(legacy_calls$count > 0L,
            "failed verifier should fall back to legacy-mgcv")
trace <- fallback$diagnostics$precision_trace
assert_true(any(trace$fallback_triggered),
            "trace should record fallback trigger")
assert_true(any(trace$verifier_executed == "legacy-mgcv-spy"),
            "trace should record legacy verifier execution")
assert_true(any(trace$p_source_used == "verifier:legacy-mgcv-spy+spy-ci"),
            "p_source_used should name legacy verifier fallback")

cat("PASS precision hybrid CPU skip and fallback\n")
