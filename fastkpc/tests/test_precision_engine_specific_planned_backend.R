source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

caps_cpu <- list(
  R_version = "4.5.0",
  mgcv_version = "1.9-4",
  cuda_available = FALSE,
  mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
  spectral_gcv_version = "single-penalty-spectral-gcv-v1",
  setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
)
caps_cuda <- modifyList(caps_cpu, list(cuda_available = TRUE))

cpu_route <- fastkpc_resolve_backend_request(
  precision = "hybrid", alpha = 0.05, tau = log(2), S = 1L,
  formula_class = "full-smooth", penalty_count = 1L,
  family = "gaussian", link = "identity",
  setup_fingerprint = "setup-cpu",
  runtime_capabilities = caps_cpu,
  execution_engine = "cpu"
)
assert_true(cpu_route$primary_backend == "fastSplineCPU",
            "CPU hybrid route should plan fastSplineCPU primary")

cuda_route <- fastkpc_resolve_backend_request(
  precision = "hybrid", alpha = 0.05, tau = log(2), S = 1L,
  formula_class = "full-smooth", penalty_count = 1L,
  family = "gaussian", link = "identity",
  setup_fingerprint = "setup-cuda",
  runtime_capabilities = caps_cuda,
  execution_engine = "cuda"
)
assert_true(cuda_route$primary_backend == "fastSplineCUDA",
            "CUDA hybrid route should plan fastSplineCUDA primary")

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

set.seed(1291)
result <- fast_kpc(
  matrix(rnorm(48 * 3), 48, 3),
  alpha = 0.05,
  max_conditioning_size = 1,
  engine = "cpu",
  precision = "hybrid",
  tau = Inf,
  graph_stage = "skeleton",
  precision_executors = list(
    `direct-ci` = make_spy(0.001, 0.001, "direct-ci-spy"),
    fastSplineCPU = make_spy(0.001, 0.049, "fastSplineCPU-spy"),
    mgcvExtractCPUGCVBridge = make_spy(0.001, 0.051, "mgcvExtractCPU-spy"),
    `legacy-mgcv` = make_spy(0.9, 0.9, "legacy-mgcv-spy")
  ),
  runtime_capabilities = caps_cpu
)

trace <- result$diagnostics$precision_trace
non_direct <- trace[trace$S_key != "", , drop = FALSE]
assert_true(nrow(non_direct) > 0L, "test should exercise non-empty S rows")
assert_true(all(non_direct$backend_planned == "fastSplineCPU"),
            "CPU hybrid trace should plan fastSplineCPU for primary rows")

cat("PASS precision engine-specific planned backend\n")
