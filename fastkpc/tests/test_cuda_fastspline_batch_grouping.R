source("fastkpc/R/native.R")
source("fastkpc/R/cuda_native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

`%||%` <- function(x, y) if (is.null(x)) y else x
residual_values <- function(fit) fit$residuals %||% fit$residual
max_abs_diff <- function(a, b) max(abs(as.numeric(a) - as.numeric(b)))

build_fastkpc_native(rebuild = TRUE)
build_fastkpc_cuda_native(rebuild = TRUE)

set.seed(502)
n <- 110
z1 <- runif(n, -2, 2)
z2 <- runif(n, -2, 2)
z3 <- runif(n, -2, 2)
data <- cbind(
  x1 = sin(z1) + rnorm(n, sd = 0.08),
  x2 = cos(z1) + rnorm(n, sd = 0.08),
  x3 = z1 * z2 + rnorm(n, sd = 0.08),
  x4 = sin(z2) + cos(z3) + rnorm(n, sd = 0.08),
  x5 = z3 + rnorm(n, sd = 0.08),
  x6 = rnorm(n)
)

targets <- c(1L, 2L, 3L, 4L, 5L)
conditioning_sets <- list(integer(0), 6L, 6L, c(1L, 2L), c(1L, 2L, 3L))
params <- list(knots = 8, lambda_count = 17, ridge = 1e-8)

batch <- fastspline_residual_batch_cuda(
  data,
  targets = targets,
  conditioning_sets = conditioning_sets,
  fastspline_params = params,
  fallback = FALSE
)

diag <- batch$batch_diagnostics
assert_true(is.list(diag), "batch diagnostics should be present")
assert_true(as.integer(diag$groups) >= 3L,
            "mixed design dimensions should produce at least three groups")
assert_true(is.data.frame(diag$group_table), "group_table should be a data frame")
assert_true(sum(as.integer(diag$group_table$fit_count)) == length(targets),
            "group fit counts should sum to requested fits")
assert_true(all(dim(batch$residuals) == c(n, length(targets))),
            "residual output should preserve input order and dimensions")
assert_true(all(dim(batch$fitted) == c(n, length(targets))),
            "fitted output should preserve input order and dimensions")

for (k in seq_along(targets)) {
  S_idx <- conditioning_sets[[k]]
  S <- if (length(S_idx) == 0L) {
    matrix(numeric(0), nrow = n, ncol = 0)
  } else {
    data[, S_idx, drop = FALSE]
  }
  cpu <- fastspline_residual(data[, targets[[k]]], S, fastspline_params = params)
  assert_true(max_abs_diff(batch$residuals[, k], residual_values(cpu)) < 1e-7,
              paste("residual", k, "should match CPU in original order"))
  assert_true(max_abs_diff(batch$fitted[, k], cpu$fitted) < 1e-7,
              paste("fitted", k, "should match CPU in original order"))
  fit_diag <- batch$diagnostics[[k]]
  assert_true(!is.null(fit_diag$batch_group_id),
              paste("fit", k, "should report batch_group_id"))
  assert_true(!is.null(fit_diag$batch_position),
              paste("fit", k, "should report batch_position"))
}

cat("test_cuda_fastspline_batch_grouping.R: PASS\n")
