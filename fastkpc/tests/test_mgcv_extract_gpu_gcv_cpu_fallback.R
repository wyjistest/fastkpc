source("fastkpc/R/mgcv_extract_oracle.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP mgcvExtractGPU GCV CPU fallback: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

set.seed(247)
n <- 50
s1 <- stats::runif(n, -2, 2)
y <- sin(s1) + stats::rnorm(n, sd = 0.08)
data <- data.frame(y = y, s1 = s1)
legacy <- mgcv::gam(y ~ s(s1, k = 8, bs = "tp"), data = data, method = "GCV.Cp")
sp_grid <- exp(seq(log(as.numeric(legacy$sp) / 4),
                   log(as.numeric(legacy$sp) * 4),
                   length.out = 9L))

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
  allow_cpu_fallback = TRUE
)

assert_true(identical(fit$backend_family, "mgcvExtractGPU"),
            "backend family should remain mgcvExtractGPU bridge")
assert_true(identical(fit$mode, "single-penalty-gpu-gcv"),
            "mode should identify single-penalty GCV")
assert_true(identical(fit$used_device, "cpu"),
            "explicit cpu device should use CPU handle solve")
assert_true(!isTRUE(fit$native_gpu_solve_used),
            "CPU fallback should not claim native GPU solve")
assert_true(identical(fit$sp_source, "fastkpc-cpu-grid-solve"),
            "CPU fallback sp source should be explicit")
assert_true(identical(fit$sp_selection_backend_executed, "cpu-grid-solve"),
            "CPU fallback sp selection backend should be explicit")
assert_true(identical(fit$gcv_source, "fastkpc-cpu-grid-solve"),
            "direct grid GCV source should identify CPU solve scoring")
assert_true(identical(fit$gcv_score_backend_executed, "cpu-grid-solve"),
            "GCV score backend should be explicit")
assert_true(identical(fit$selected_solve_backend_executed, "cpu"),
            "selected solve backend should be explicit")
assert_true(isTRUE(fit$is_self_contained_gcv),
            "grid search is still self-contained GCV")
assert_true(fit$sp %in% sp_grid, "selected sp should come from grid")
assert_true(is.data.frame(fit$grid) && nrow(fit$grid) == length(sp_grid),
            "grid diagnostics should be present")
assert_true(which.min(fit$grid$gcv) == fit$selected_grid_index,
            "selected grid index should minimize GCV")
assert_true(all(is.finite(fit$residuals)) && length(fit$residuals) == n,
            "finite residual output")

cat("PASS mgcvExtractGPU GCV CPU fallback\n")
