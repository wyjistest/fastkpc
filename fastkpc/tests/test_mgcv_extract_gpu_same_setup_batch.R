source("fastkpc/R/mgcv_extract_oracle.R")
source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP mgcvExtractGPU same-setup batch: FASTKPC_RUN_CUDA_TESTS != 1\n")
  quit(save = "no", status = 0)
}

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP mgcvExtractGPU same-setup batch: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

build_fastkpc_cuda_native(rebuild = TRUE)
if (!fastkpc_cuda_available()) {
  cat("SKIP mgcvExtractGPU same-setup batch: CUDA unavailable\n")
  quit(save = "no", status = 0)
}

set.seed(248)
n <- 62
s1 <- stats::runif(n, -2, 2)
Y <- cbind(
  y1 = sin(s1) + stats::rnorm(n, sd = 0.04),
  y2 = cos(s1) + stats::rnorm(n, sd = 0.04)
)
sp <- c(0.3, 1.1)
S_data <- data.frame(s1 = s1)

batch <- fastkpc_mgcv_extract_gpu_same_setup_batch_fixed_sp_cuda(
  Y = Y,
  S_data = S_data,
  S = 1L,
  sp = sp,
  k = 9L,
  bs = "tp",
  target_ids = c(10L, 11L)
)

single_setups <- lapply(seq_len(ncol(Y)), function(j) {
  fastkpc_mgcv_extract_setup(
    formula = y ~ s(s1, k = 9, bs = "tp"),
    data = data.frame(y = Y[, j], S_data),
    sp = sp[j],
    target = j,
    S = 1L,
    k = 9L,
    bs = "tp"
  )
})
single <- fastkpc_mgcv_extract_gpu_solve_handle_batch_fixed_sp_cuda(
  setups = single_setups,
  target_ids = c(10L, 11L)
)

assert_true(identical(batch$backend_family, "mgcvExtractGPU"),
            "same-setup batch backend")
assert_true(identical(batch$mode, "fixed-sp-same-setup-native-gpu-batch-bridge"),
            "same-setup batch mode")
assert_true(identical(batch$solve_source,
                      "mgcvExtractGPU-native-same-setup-fixed-sp-batch"),
            "same-setup batch solve source")
assert_true(identical(batch$used_device, "cuda"),
            "same-setup batch should use CUDA")
assert_true(isTRUE(batch$native_gpu_solve_used),
            "same-setup batch should report native GPU usage")
assert_true(all(batch$target_ids == c(10L, 11L)),
            "same-setup batch preserves target ids")
assert_true(all(abs(batch$sp - sp) < 1e-12),
            "same-setup batch preserves per-target sp")
assert_true(all(dim(batch$residuals) == c(n, 2L)),
            "same-setup residual dimensions")
assert_true(max(abs(batch$residuals - single$residuals)) < 1e-8,
            "same-setup residuals should match independently extracted native batch")
assert_true(max(abs(batch$fitted - single$fitted)) < 1e-8,
            "same-setup fitted values should match independently extracted native batch")
assert_true(length(unique(batch$setup_fingerprints)) == 1L,
            "same-setup batch should reuse one setup fingerprint")
assert_true(identical(batch$diagnostics$setup_reused, TRUE),
            "diagnostics should record setup reuse")
assert_true(identical(batch$diagnostics$native_batch_call, TRUE),
            "diagnostics should record one native batch call")
assert_true(identical(batch$diagnostics$targets, 2L),
            "diagnostics should record target count")

cat("PASS mgcvExtractGPU same-setup batch\n")
