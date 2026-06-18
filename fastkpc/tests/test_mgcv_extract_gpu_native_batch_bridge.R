source("fastkpc/R/mgcv_extract_oracle.R")
source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP mgcvExtractGPU native batch bridge: FASTKPC_RUN_CUDA_TESTS != 1\n")
  quit(save = "no", status = 0)
}

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP mgcvExtractGPU native batch bridge: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

build_fastkpc_cuda_native(rebuild = TRUE)
if (!fastkpc_cuda_available()) {
  cat("SKIP mgcvExtractGPU native batch bridge: CUDA unavailable\n")
  quit(save = "no", status = 0)
}

set.seed(247)
n <- 58
s1 <- stats::runif(n, -2, 2)
Y <- cbind(
  y1 = sin(s1) + stats::rnorm(n, sd = 0.05),
  y2 = cos(s1) + stats::rnorm(n, sd = 0.05),
  y3 = sin(2 * s1) + stats::rnorm(n, sd = 0.05)
)
sp <- c(0.2, 0.8, 1.4)
setups <- lapply(seq_len(ncol(Y)), function(j) {
  fastkpc_mgcv_extract_setup(
    formula = y ~ s(s1, k = 9, bs = "tp"),
    data = data.frame(y = Y[, j], s1 = s1),
    sp = sp[j],
    target = j,
    S = 2L,
    k = 9L,
    bs = "tp"
  )
})

batch <- fastkpc_mgcv_extract_gpu_solve_handle_batch_fixed_sp_cuda(
  setups = setups,
  target_ids = seq_len(ncol(Y))
)
single <- lapply(setups, function(setup) {
  handle <- fastkpc_mgcv_extract_gpu_setup_handle(
    setup = setup,
    device_resident = TRUE
  )
  fastkpc_mgcv_extract_gpu_solve_handle_fixed_sp_cuda(handle)
})

assert_true(identical(batch$backend_family, "mgcvExtractGPU"),
            "batch backend should identify mgcvExtractGPU")
assert_true(identical(batch$mode, "fixed-sp-native-gpu-batch-bridge"),
            "batch mode should identify native CUDA batch bridge")
assert_true(identical(batch$solve_source, "mgcvExtractGPU-native-fixed-sp-batch-bridge"),
            "batch solve source should be explicit")
assert_true(identical(batch$used_device, "cuda"),
            "batch bridge should report CUDA device")
assert_true(isTRUE(batch$native_gpu_solve_used),
            "batch bridge should report native GPU usage")
assert_true(all(batch$target_ids == seq_len(ncol(Y))),
            "batch should preserve target ids")
assert_true(all(abs(batch$sp - sp) < 1e-12),
            "batch should preserve per-target fixed sp")
assert_true(all(dim(batch$residuals) == c(n, ncol(Y))),
            "batch residual matrix dimensions")
assert_true(all(dim(batch$fitted) == c(n, ncol(Y))),
            "batch fitted matrix dimensions")
for (j in seq_len(ncol(Y))) {
  assert_true(max(abs(batch$residuals[, j] - single[[j]]$residuals)) < 1e-8,
              paste("batch residual column", j, "matches single native solve"))
  assert_true(max(abs(batch$fitted[, j] - single[[j]]$fitted)) < 1e-8,
              paste("batch fitted column", j, "matches single native solve"))
}
assert_true(identical(batch$diagnostics$targets, as.integer(ncol(Y))),
            "batch diagnostics should record target count")
assert_true(identical(batch$diagnostics$device_resident, TRUE),
            "batch diagnostics should record device residency")
assert_true(identical(batch$diagnostics$batch_stage,
                      "native-fixed-sp-repeated-handle-bridge"),
            "diagnostics should avoid claiming true batched kernel")
assert_true(length(unique(batch$setup_fingerprints)) == 1L,
            "same-S setup fingerprint should be reusable across targets")

cat("PASS mgcvExtractGPU native batch bridge\n")
