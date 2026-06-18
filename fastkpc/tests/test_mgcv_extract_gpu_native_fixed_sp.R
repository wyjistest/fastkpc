source("fastkpc/R/mgcv_extract_oracle.R")
source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP mgcvExtractGPU native fixed-sp solve: FASTKPC_RUN_CUDA_TESTS != 1\n")
  quit(save = "no", status = 0)
}

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP mgcvExtractGPU native fixed-sp solve: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

build_fastkpc_cuda_native(rebuild = TRUE)
if (!fastkpc_cuda_available()) {
  cat("SKIP mgcvExtractGPU native fixed-sp solve: CUDA unavailable\n")
  quit(save = "no", status = 0)
}

set.seed(245)
n <- 64
s1 <- stats::runif(n, -2, 2)
s2 <- stats::runif(n, -2, 2)
y <- sin(s1) + cos(s2) + stats::rnorm(n, sd = 0.05)
data <- data.frame(y = y, s1 = s1, s2 = s2)
formula <- y ~ s(s1, s2, k = 12, bs = "tp")
sp <- 0.55

setup <- fastkpc_mgcv_extract_setup(
  formula = formula,
  data = data,
  sp = sp,
  target = 1L,
  S = c(2L, 3L),
  k = 12L,
  bs = "tp"
)
handle <- fastkpc_mgcv_extract_gpu_setup_handle(setup)
cpu <- fastkpc_mgcv_extract_gpu_solve_handle_fixed_sp(handle)
gpu <- fastkpc_mgcv_extract_gpu_solve_handle_fixed_sp_cuda(handle)
api_gpu <- fastkpc_mgcv_extract_gpu_fixed_sp(
  formula = formula,
  data = data,
  sp = sp,
  target = 1L,
  S = c(2L, 3L),
  k = 12L,
  bs = "tp",
  device = "cuda",
  allow_cpu_fallback = FALSE,
  solve_strategy = "handle"
)

assert_true(identical(gpu$backend_family, "mgcvExtractGPU"),
            "native solve backend should identify mgcvExtractGPU")
assert_true(identical(gpu$mode, "fixed-sp-native-gpu-solve"),
            "native solve mode should identify CUDA solve")
assert_true(identical(gpu$solve_source, "mgcvExtractGPU-native-fixed-sp"),
            "native solve source should be explicit")
assert_true(identical(gpu$used_device, "cuda"),
            "native solve should report cuda")
assert_true(isTRUE(gpu$native_gpu_solve_used),
            "native solve should report native GPU usage")
assert_true(length(gpu$coefficients) == length(cpu$coefficients),
            "native solve coefficient length")
assert_true(max(abs(gpu$theta - cpu$theta)) < 1e-7,
            "native theta should match CPU handle solve")
assert_true(max(abs(gpu$coefficients - cpu$coefficients)) < 1e-7,
            "native coefficients should match CPU handle solve")
assert_true(max(abs(gpu$fitted - cpu$fitted)) < 1e-7,
            "native fitted values should match CPU handle solve")
assert_true(max(abs(gpu$residuals - cpu$residuals)) < 1e-7,
            "native residuals should match CPU handle solve")
assert_true(abs(gpu$rss - cpu$rss) < 1e-7,
            "native RSS should match CPU handle solve")
assert_true(identical(api_gpu$used_device, "cuda"),
            "top-level API should use CUDA handle solve")
assert_true(isTRUE(api_gpu$native_gpu_solve_used),
            "top-level API should report native GPU solve usage")
assert_true(!isTRUE(api_gpu$fallback_used),
            "top-level API should not report fallback when CUDA solve succeeds")
assert_true(identical(api_gpu$solve_source, "mgcvExtractGPU-native-fixed-sp"),
            "top-level API should preserve native solve source")
assert_true(max(abs(api_gpu$residuals - cpu$residuals)) < 1e-7,
            "top-level CUDA residuals should match CPU handle solve")

cat("PASS mgcvExtractGPU native fixed-sp solve\n")
