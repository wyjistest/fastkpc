source("fastkpc/R/native.R")
source("fastkpc/R/cuda_native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

max_abs_diff <- function(a, b) max(abs(as.numeric(a) - as.numeric(b)))
`%||%` <- function(x, y) if (is.null(x)) y else x
residual_values <- function(fit) fit$residuals %||% fit$residual

build_fastkpc_native(rebuild = TRUE)
build_fastkpc_cuda_native(rebuild = TRUE)

set.seed(101)
n <- 96
z1 <- runif(n, -2, 2)
z2 <- runif(n, -2, 2)
z3 <- runif(n, -2, 2)
y <- sin(z1) + cos(z2) + 0.25 * z3 + rnorm(n, sd = 0.08)

cases <- list(
  empty = matrix(numeric(0), nrow = n, ncol = 0),
  one = cbind(z1 = z1),
  two = cbind(z1 = z1, z2 = z2),
  three = cbind(z1 = z1, z2 = z2, z3 = z3)
)

params <- list(knots = 8, lambda_count = 17, ridge = 1e-8)

for (name in names(cases)) {
  S <- cases[[name]]
  cpu <- fastspline_residual(y, S, fastspline_params = params)
  cuda <- fastspline_residual_cuda(y, S, fastspline_params = params,
                                   fallback = FALSE)

  assert_true(is.list(cuda), paste(name, "CUDA result should be a list"))
  assert_true(cuda$backend == "cuda", paste(name, "backend should be cuda"))
  assert_true(cuda$residual_backend == "fastSpline",
              paste(name, "residual backend should be fastSpline"))
  assert_true(cuda$residual_device == "cuda",
              paste(name, "residual_device should be cuda"))
  assert_true(identical(cuda$fallback_used, FALSE),
              paste(name, "fallback should not be used"))
  assert_true(length(cuda$residuals) == n, paste(name, "residual length"))
  assert_true(length(cuda$fitted) == n, paste(name, "fitted length"))
  assert_true(all(is.finite(cuda$residuals)), paste(name, "finite residuals"))
  assert_true(all(is.finite(cuda$fitted)), paste(name, "finite fitted"))
  assert_true(is.finite(cuda$selected_lambda) && cuda$selected_lambda > 0,
              paste(name, "selected lambda should be positive"))
  assert_true(is.finite(cuda$gcv), paste(name, "gcv should be finite"))
  assert_true(is.finite(cuda$rss) && cuda$rss >= 0,
              paste(name, "rss should be finite"))
  assert_true(is.finite(cuda$edf) && cuda$edf > 0,
              paste(name, "edf should be positive"))
  assert_true(cuda$design_cols == cpu$design_cols,
              paste(name, "design_cols should match CPU"))

  assert_true(max_abs_diff(cuda$residuals, residual_values(cpu)) < 1e-7,
              paste(name, "residuals should match CPU"))
  assert_true(max_abs_diff(cuda$fitted, cpu$fitted) < 1e-7,
              paste(name, "fitted values should match CPU"))
  rel_rss <- abs(cuda$rss - cpu$rss) / max(1, abs(cpu$rss))
  assert_true(rel_rss < 1e-8, paste(name, "rss should match CPU"))
}

cat("test_cuda_fastspline_residual_kernel.R: PASS\n")
