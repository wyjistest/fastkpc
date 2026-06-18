source("fastkpc/R/mgcv_extract_oracle.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP mgcvExtractGPU spectral GCV: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

set.seed(249)
n <- 56
s1 <- stats::runif(n, -2, 2)
y <- sin(s1) + stats::rnorm(n, sd = 0.07)
data <- data.frame(y = y, s1 = s1)
legacy <- mgcv::gam(y ~ s(s1, k = 9, bs = "tp"), data = data, method = "GCV.Cp")
sp_grid <- exp(seq(log(as.numeric(legacy$sp) / 4),
                   log(as.numeric(legacy$sp) * 4),
                   length.out = 11L))

direct <- fastkpc_mgcv_extract_gpu_gcv(
  formula = y ~ s(s1, k = 9, bs = "tp"),
  data = data,
  setup_sp = 1,
  sp_grid = sp_grid,
  target = 1L,
  S = 2L,
  k = 9L,
  bs = "tp",
  device = "cpu",
  gcv_strategy = "direct"
)
spectral <- fastkpc_mgcv_extract_gpu_gcv(
  formula = y ~ s(s1, k = 9, bs = "tp"),
  data = data,
  setup_sp = 1,
  sp_grid = sp_grid,
  target = 1L,
  S = 2L,
  k = 9L,
  bs = "tp",
  device = "cpu",
  gcv_strategy = "spectral"
)

assert_true(identical(spectral$diagnostics$gcv_stage,
                      "single-penalty-spectral-grid-search"),
            "spectral strategy should report Demmler-Reinsch-style stage")
assert_true(identical(spectral$diagnostics$gcv_strategy, "spectral"),
            "diagnostics should record spectral strategy")
assert_true(isTRUE(spectral$diagnostics$spectral_reparameterization),
            "diagnostics should record spectral reparameterization")
assert_true(identical(spectral$selected_grid_index, direct$selected_grid_index),
            "spectral and direct grid should select the same sp")
assert_true(max(abs(spectral$grid$gcv - direct$grid$gcv)) < 1e-7,
            "spectral GCV should match direct grid scores")
assert_true(max(abs(spectral$residuals - direct$residuals)) < 1e-8,
            "spectral selected residuals should match direct solve")
assert_true(abs(spectral$edf - direct$edf) < 1e-8,
            "spectral EDF should match direct EDF")

cat("PASS mgcvExtractGPU spectral GCV\n")
