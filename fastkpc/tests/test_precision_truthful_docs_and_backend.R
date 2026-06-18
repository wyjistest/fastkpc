source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

readme <- paste(readLines("README.md", warn = FALSE), collapse = "\n")
pkg_readme <- paste(readLines("fastkpc/README.md", warn = FALSE), collapse = "\n")
combined <- paste(readme, pkg_readme, sep = "\n")

assert_true(!grepl("CPU skeleton vertical slice for \\|S\\| <= 1", combined),
            "README files must not claim precision data-plane is only CPU |S|<=1")
assert_true(!grepl("CUDA, and hybrid verifier execution remain future work",
                   combined, fixed = TRUE),
            "README files must not claim CUDA/hybrid verifier execution is future work")
assert_true(grepl("CPU and CUDA skeleton data-plane", combined, fixed = TRUE),
            "README files should describe current CPU/CUDA skeleton data-plane scope")
assert_true(grepl("|S| <= 2", combined, fixed = TRUE),
            "README files should describe current |S|<=2 precision scope")

make_spy <- function(p_value, backend_name, calls_env,
                     p_value_nonempty = p_value) {
  force(p_value)
  force(backend_name)
  force(calls_env)
  force(p_value_nonempty)
  function(data, x, y, S, ci_method, index, legacy_index, hsic_params,
           permutation_params, route, role) {
    calls_env$count <- calls_env$count + 1L
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

caps <- list(
  R_version = "4.5.0",
  mgcv_version = "1.9-4",
  cuda_available = TRUE,
  cuda_device_capability = "8.9",
  mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
  spectral_gcv_version = "single-penalty-spectral-gcv-v1",
  setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
)

set.seed(1401)
calls <- new.env(parent = emptyenv())
calls$count <- 0L
result <- fast_kpc(
  matrix(rnorm(48 * 3), 48, 3),
  alpha = 0.05,
  max_conditioning_size = 1,
  engine = "cuda",
  precision = "compatible",
  graph_stage = "skeleton",
  precision_executors = list(
    `direct-ci` = make_spy(0.001, "direct-ci-spy",
                           new.env(parent = emptyenv())),
    mgcvExtractGPUGCV = make_spy(0.001, "mgcvExtractGPU-spy", calls,
                                 p_value_nonempty = 0.051),
    mgcvExtractCPUGCVBridge = make_spy(0.9, "mgcvExtractCPU-spy",
                                       new.env(parent = emptyenv())),
    `legacy-mgcv` = make_spy(0.9, "legacy-mgcv-spy",
                             new.env(parent = emptyenv()))
  ),
  runtime_capabilities = caps
)

assert_true(result$skeleton$backend == "cuda",
            "CUDA precision skeleton should report backend = 'cuda'")
assert_true(result$config$engine_used == "cuda",
            "test should exercise CUDA engine")

cat("PASS precision truthful docs and backend\n")
