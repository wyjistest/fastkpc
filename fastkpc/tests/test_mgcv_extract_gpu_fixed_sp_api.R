source("fastkpc/R/mgcv_extract_oracle.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP mgcv extract GPU fixed-sp API: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

set.seed(241)
n <- 48
s1 <- stats::runif(n, -2, 2)
y <- sin(s1) + stats::rnorm(n, sd = 0.05)
data <- data.frame(y = y, s1 = s1)
formula <- y ~ s(s1, k = 8, bs = "tp")
sp <- 0.4

cpu <- fastkpc_mgcv_extract_fixed_sp_solve(
  formula = formula,
  data = data,
  sp = sp,
  target = 1L,
  S = 2L,
  k = 8L,
  bs = "tp"
)
gpu <- fastkpc_mgcv_extract_gpu_fixed_sp(
  formula = formula,
  data = data,
  sp = sp,
  target = 1L,
  S = 2L,
  k = 8L,
  bs = "tp",
  device = "cuda",
  allow_cpu_fallback = TRUE
)
handle_gpu <- fastkpc_mgcv_extract_gpu_fixed_sp(
  formula = formula,
  data = data,
  sp = sp,
  target = 1L,
  S = 2L,
  k = 8L,
  bs = "tp",
  device = "cuda",
  allow_cpu_fallback = TRUE,
  solve_strategy = "handle"
)

assert_true(identical(gpu$backend_family, "mgcvExtractGPU"),
            "backend family should identify mgcvExtractGPU")
assert_true(identical(gpu$mode, "fixed-sp-gpu-bridge"),
            "mode should identify fixed-sp GPU bridge")
assert_true(identical(gpu$requested_device, "cuda"),
            "requested device should be recorded")
assert_true(identical(gpu$used_device, "cpu"),
            "CPU-only test context should fall back when CUDA wrapper is unavailable")
assert_true(isTRUE(gpu$fallback_used), "fallback_used should be TRUE")
assert_true(grepl("unavailable", gpu$fallback_reason, fixed = TRUE),
            "fallback reason should explain native GPU solve is unavailable")
assert_true(identical(gpu$solve_source, "fastkpc-fixed-sp"),
            "fallback solve source should remain Gate B fixed-sp")
assert_true(identical(gpu$sp_source, "fixed-input"),
            "fixed-sp bridge should record fixed-input sp source")
assert_true(identical(gpu$gcv_source, "none"),
            "fixed-sp bridge should not claim GCV")
assert_true(!isTRUE(gpu$is_self_contained_gcv),
            "fixed-sp bridge should not claim self-contained GCV")
assert_true(max(abs(gpu$residuals - cpu$residuals)) < 1e-10,
            "CPU fallback residuals should match Gate B CPU")
assert_true(max(abs(gpu$fitted - cpu$fitted)) < 1e-10,
            "CPU fallback fitted values should match Gate B CPU")
assert_true(identical(gpu$setup_fingerprint$fingerprint,
                      cpu$setup_fingerprint$fingerprint),
            "fallback should preserve setup fingerprint")
assert_true(identical(handle_gpu$solve_source, "fastkpc-handle-fixed-sp"),
            "handle strategy should use handle fixed-sp solve source")
assert_true(identical(handle_gpu$used_device, "cpu"),
            "handle strategy should fall back when CUDA wrapper is unavailable")
assert_true(isTRUE(handle_gpu$fallback_used),
            "handle strategy should still report fallback for cuda request")
assert_true(max(abs(handle_gpu$residuals - cpu$residuals)) < 1e-8,
            "handle strategy residuals should match Gate B CPU")

no_fallback_error <- tryCatch({
  fastkpc_mgcv_extract_gpu_fixed_sp(
    formula = formula,
    data = data,
    sp = sp,
    target = 1L,
    S = 2L,
    k = 8L,
    bs = "tp",
    device = "cuda",
    allow_cpu_fallback = FALSE
  )
  ""
}, error = function(e) conditionMessage(e))
assert_true(grepl("mgcvExtractGPU native fixed-sp solve is not implemented",
                  no_fallback_error, fixed = TRUE) ||
              grepl("mgcvExtractGPU native fixed-sp solve is unavailable",
                    no_fallback_error, fixed = TRUE),
            "disabling fallback should fail explicitly")

cap <- fastkpc_mgcv_extract_gpu_capabilities()
assert_true(identical(cap$backend, "mgcvExtractGPU"),
            "GPU capability backend should be mgcvExtractGPU")
assert_true(isTRUE(cap$supported$fixed_sp_api),
            "GPU capability should expose fixed-sp API")
assert_true(isFALSE(cap$supported$native_gpu_fixed_sp_solve),
            "native GPU fixed-sp solve should remain unsupported in v1")
assert_true(isTRUE(cap$supported$cpu_gate_b_fallback),
            "CPU Gate B fallback should be supported")

cat("PASS mgcv extract GPU fixed-sp API\n")
