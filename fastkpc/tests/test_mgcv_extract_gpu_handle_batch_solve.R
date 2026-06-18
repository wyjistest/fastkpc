source("fastkpc/R/mgcv_extract_oracle.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP mgcv extract GPU handle batch solve: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

set.seed(244)
n <- 54
s1 <- stats::runif(n, -2, 2)
Y <- cbind(
  y1 = sin(s1) + stats::rnorm(n, sd = 0.04),
  y2 = cos(s1) + stats::rnorm(n, sd = 0.04)
)
S_data <- data.frame(s1 = s1)
sp <- c(0.25, 0.9)

setup1 <- fastkpc_mgcv_extract_setup(
  formula = y ~ s(s1, k = 8, bs = "tp"),
  data = data.frame(y = Y[, 1], S_data),
  sp = sp[1],
  target = 1L,
  S = 2L,
  k = 8L,
  bs = "tp"
)
setup2 <- fastkpc_mgcv_extract_setup(
  formula = y ~ s(s1, k = 8, bs = "tp"),
  data = data.frame(y = Y[, 2], S_data),
  sp = sp[2],
  target = 2L,
  S = 2L,
  k = 8L,
  bs = "tp"
)

batch <- fastkpc_mgcv_extract_gpu_solve_handle_batch_fixed_sp(
  setups = list(setup1, setup2),
  target_ids = c(1L, 2L)
)
one <- fastkpc_mgcv_extract_gpu_solve_handle_fixed_sp(
  fastkpc_mgcv_extract_gpu_setup_handle(setup1)
)
two <- fastkpc_mgcv_extract_gpu_solve_handle_fixed_sp(
  fastkpc_mgcv_extract_gpu_setup_handle(setup2)
)

assert_true(identical(batch$backend_family, "mgcvExtractGPU"),
            "batch backend should identify mgcvExtractGPU")
assert_true(identical(batch$mode, "fixed-sp-handle-batch-solve"),
            "batch mode should identify handle batch solve")
assert_true(identical(batch$solve_source, "fastkpc-handle-fixed-sp-batch"),
            "batch solve source should be explicit")
assert_true(identical(batch$used_device, "cpu"),
            "R-level batch solve should report cpu")
assert_true(!isTRUE(batch$native_gpu_solve_used),
            "R-level batch solve should not claim native GPU")
assert_true(all(batch$target_ids == c(1L, 2L)),
            "batch should preserve target ids")
assert_true(all(abs(batch$sp - sp) < 1e-12),
            "batch should preserve per-target fixed sp")
assert_true(all(dim(batch$residuals) == c(n, 2L)),
            "batch residual matrix dimensions")
assert_true(max(abs(batch$residuals[, 1] - one$residuals)) < 1e-8,
            "first batch residual should match single handle solve")
assert_true(max(abs(batch$residuals[, 2] - two$residuals)) < 1e-8,
            "second batch residual should match single handle solve")
assert_true(identical(batch$diagnostics$targets, 2L),
            "batch diagnostics should record target count")
assert_true(length(unique(batch$setup_fingerprints)) == 1L,
            "same-S setup fingerprints should be reusable across targets")

cat("PASS mgcv extract GPU handle batch solve\n")
