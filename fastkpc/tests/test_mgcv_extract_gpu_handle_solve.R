source("fastkpc/R/mgcv_extract_oracle.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP mgcv extract GPU handle solve: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

set.seed(243)
n <- 58
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
solved <- fastkpc_mgcv_extract_gpu_solve_handle_fixed_sp(handle)
gate_b <- fastkpc_mgcv_solve_setup_fixed_sp(setup)

assert_true(identical(solved$backend_family, "mgcvExtractGPU"),
            "handle solve backend should identify mgcvExtractGPU")
assert_true(identical(solved$mode, "fixed-sp-handle-solve"),
            "handle solve mode should be fixed-sp-handle-solve")
assert_true(identical(solved$solve_source, "fastkpc-handle-fixed-sp"),
            "handle solve source should be explicit")
assert_true(identical(solved$used_device, "cpu"),
            "R-level handle solve should report cpu")
assert_true(!isTRUE(solved$native_gpu_solve_used),
            "R-level handle solve should not claim native GPU")
assert_true(identical(solved$setup_fingerprint$fingerprint,
                      setup$setup_fingerprint$fingerprint),
            "handle solve should preserve setup fingerprint")
assert_true(max(abs(solved$fitted - gate_b$fitted)) < 1e-8,
            "handle fitted values should match Gate B CPU")
assert_true(max(abs(solved$residuals - gate_b$residuals)) < 1e-8,
            "handle residuals should match Gate B CPU")
assert_true(length(solved$coefficients) == ncol(setup$X),
            "handle solve should return full coefficient vector")
assert_true(is.finite(solved$rss) && solved$rss >= 0,
            "handle solve should report finite RSS")

cat("PASS mgcv extract GPU handle solve\n")
