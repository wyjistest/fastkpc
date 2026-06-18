source("fastkpc/R/mgcv_extract_oracle.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP mgcvExtractGPU GCV diagnostics truthful: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

set.seed(1402)
n <- 42
s1 <- stats::runif(n, -2, 2)
y <- sin(s1) + stats::rnorm(n, sd = 0.06)
data <- data.frame(y = y, s1 = s1)
sp_grid <- exp(seq(log(0.05), log(20), length.out = 7L))

fit <- fastkpc_mgcv_extract_gpu_gcv(
  formula = y ~ s(s1, k = 8, bs = "tp"),
  data = data,
  setup_sp = 1,
  sp_grid = sp_grid,
  target = 1L,
  S = 2L,
  k = 8L,
  bs = "tp",
  device = "cpu",
  allow_cpu_fallback = TRUE,
  gcv_strategy = "spectral"
)

assert_true(identical(fit$gcv_source, "fastkpc-r-cpu-spectral"),
            "spectral GCV scoring should be identified as R/CPU spectral")
assert_true(identical(fit$gcv_score_backend_executed, "r-cpu-spectral"),
            "GCV score backend should be separate from final solve backend")
assert_true(identical(fit$selected_solve_backend_executed, "cpu"),
            "selected solve backend should name CPU for explicit CPU solve")
assert_true(identical(fit$diagnostics$gcv_score_backend_executed,
                      "r-cpu-spectral"),
            "diagnostics should carry truthful GCV score backend")
assert_true(identical(fit$diagnostics$selected_solve_backend_executed,
                      "cpu"),
            "diagnostics should carry selected solve backend")

cat("PASS mgcvExtractGPU GCV diagnostics truthful\n")
