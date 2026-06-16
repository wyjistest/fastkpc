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

set.seed(102)
n <- 90
z1 <- runif(n, -2, 2)
z2 <- runif(n, -2, 2)
z3 <- runif(n, -2, 2)
data <- cbind(
  x1 = sin(z1) + rnorm(n, sd = 0.08),
  x2 = cos(z1) + rnorm(n, sd = 0.08),
  x3 = sin(z2) + cos(z3) + rnorm(n, sd = 0.08),
  x4 = z1 * z2 + rnorm(n, sd = 0.08),
  x5 = rnorm(n)
)

targets <- c(1L, 2L, 3L, 4L)
conditioning_sets <- list(integer(0), 1L, c(1L, 2L), c(1L, 2L, 3L))
params <- list(knots = 8, lambda_count = 17, ridge = 1e-8)

batch <- fastspline_residual_batch_cuda(
  data,
  targets = targets,
  conditioning_sets = conditioning_sets,
  fastspline_params = params,
  fallback = FALSE
)

assert_true(is.matrix(batch$residuals), "batch residuals should be a matrix")
assert_true(all(dim(batch$residuals) == c(n, length(targets))),
            "batch residual matrix dimension should match")
assert_true(is.matrix(batch$fitted), "batch fitted should be a matrix")
assert_true(all(dim(batch$fitted) == c(n, length(targets))),
            "batch fitted matrix dimension should match")
assert_true(length(batch$selected_lambda) == length(targets),
            "lambda length should match batch")
assert_true(length(batch$gcv) == length(targets), "gcv length should match batch")
assert_true(length(batch$rss) == length(targets), "rss length should match batch")
assert_true(length(batch$edf) == length(targets), "edf length should match batch")
assert_true(all(batch$residual_device == "cuda"),
            "all batch residual devices should be cuda")
assert_true(!any(batch$fallback_used), "fallback should not be used")

for (k in seq_along(targets)) {
  target <- targets[[k]]
  S_idx <- conditioning_sets[[k]]
  y <- data[, target]
  S <- if (length(S_idx) == 0L) {
    matrix(numeric(0), nrow = n, ncol = 0)
  } else {
    data[, S_idx, drop = FALSE]
  }
  cpu <- fastspline_residual(y, S, fastspline_params = params)
  assert_true(max_abs_diff(batch$residuals[, k], residual_values(cpu)) < 1e-7,
              paste("batch residual", k, "should match CPU"))
  assert_true(max_abs_diff(batch$fitted[, k], cpu$fitted) < 1e-7,
              paste("batch fitted", k, "should match CPU"))
  rel_rss <- abs(batch$rss[[k]] - cpu$rss) / max(1, abs(cpu$rss))
  assert_true(rel_rss < 1e-8, paste("batch rss", k, "should match CPU"))
}

cat("test_cuda_fastspline_residual_batch.R: PASS\n")
