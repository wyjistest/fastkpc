source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

make_spy <- function(p_empty, p_nonempty, backend_name) {
  force(p_empty)
  force(p_nonempty)
  force(backend_name)
  function(data, x, y, S, ci_method, index, legacy_index, hsic_params,
           permutation_params, route, role) {
    list(
      p.value = if (length(S) == 0L) p_empty else p_nonempty,
      residual_backend_executed = backend_name,
      ci_backend_executed = "spy-ci",
      setup_fingerprint = paste0("spy:", paste(S, collapse = "|")),
      p_source_used = paste0(role, ":", backend_name, "+spy-ci"),
      timings = list(ci_test_ms = 0)
    )
  }
}

caps_cpu_only <- list(
  R_version = "4.5.0",
  mgcv_version = "1.9-4",
  cuda_available = FALSE,
  cuda_device_capability = NA_character_,
  mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
  spectral_gcv_version = "single-penalty-spectral-gcv-v1",
  setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
)

route <- fastkpc_resolve_backend_request(
  precision = "compatible",
  alpha = 0.05,
  tau = log(2),
  S = 1L,
  formula_class = "full-smooth",
  penalty_count = 1L,
  family = "gaussian",
  link = "identity",
  setup_fingerprint = "S:1",
  runtime_capabilities = caps_cpu_only,
  execution_engine = "cpu"
)
assert_true(route$primary_backend == "mgcvExtractCPUGCVBridge",
            "CPU compatible route should plan explicit CPU mgcvExtract backend")
assert_true(route$compatibility_status == "supported",
            "CPU mgcvExtract route should not require CUDA availability")
assert_true(!("cuda_available" %in% route$unsupported_checks),
            "CPU compatibility envelope must not fail on cuda_available")

route_hybrid <- fastkpc_resolve_backend_request(
  precision = "hybrid",
  alpha = 0.05,
  tau = log(2),
  S = 1L,
  formula_class = "full-smooth",
  penalty_count = 1L,
  family = "gaussian",
  link = "identity",
  setup_fingerprint = "S:1",
  runtime_capabilities = caps_cpu_only,
  execution_engine = "cpu"
)
assert_true(route_hybrid$verifier_backend == "mgcvExtractCPUGCVBridge",
            "CPU hybrid verifier should plan explicit CPU mgcvExtract backend")

set.seed(1101)
data <- matrix(rnorm(70 * 4), 70, 4)
result <- fast_kpc(
  data,
  alpha = 0.01,
  max_conditioning_size = 1,
  engine = "cpu",
  precision = "compatible",
  graph_stage = "skeleton",
  runtime_capabilities = caps_cpu_only,
  precision_trace_level = "full",
  precision_executors = list(
    `direct-ci` = make_spy(0.001, 0.001, "direct-ci-spy"),
    mgcvExtractCPUGCVBridge = make_spy(0.001, 0.9,
                                       "mgcvExtractCPU-spy"),
    `legacy-mgcv` = make_spy(0.9, 0.9, "legacy-mgcv-spy")
  )
)
trace <- result$diagnostics$precision_trace
non_direct <- trace[nzchar(trace$S_key), , drop = FALSE]
assert_true(any(non_direct$backend_planned == "mgcvExtractCPUGCVBridge"),
            "compatible CPU data plane should plan CPU mgcvExtract backend")
assert_true(any(non_direct$backend_executed == "mgcvExtractCPU-spy"),
            "compatible CPU data plane should execute mgcvExtractCPU receipt")

cat("PASS precision CPU backend identity\n")
