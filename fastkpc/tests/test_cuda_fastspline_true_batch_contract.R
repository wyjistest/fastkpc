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

set.seed(501)
n <- 120
z1 <- runif(n, -2, 2)
z2 <- runif(n, -2, 2)
data <- cbind(
  x1 = sin(z1) + rnorm(n, sd = 0.08),
  x2 = cos(z1) + rnorm(n, sd = 0.08),
  x3 = z1 * z2 + rnorm(n, sd = 0.08),
  x4 = sin(z2) + rnorm(n, sd = 0.08),
  x5 = rnorm(n)
)

targets <- c(1L, 2L, 3L, 4L)
conditioning_sets <- list(5L, 5L, 5L, 5L)
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
assert_true(identical(as.integer(diag$requested_fits), length(targets)),
            "requested_fits should match input batch length")
assert_true(as.integer(diag$true_batched_groups) >= 1L,
            "compatible batch should use at least one true batched group")
assert_true(identical(as.integer(diag$true_batched_fits), length(targets)),
            "all compatible fits should be true batched")
assert_true(identical(as.integer(diag$single_fit_calls), 0L),
            "compatible batch must not call the single-fit CUDA path")
assert_true(identical(as.integer(diag$cpu_fallback_fits), 0L),
            "fallback should not be used")
assert_true(is.data.frame(diag$group_table), "group_table should be a data frame")
assert_true(all(diag$group_table$true_batched),
            "every compatible group should be marked true_batched")
assert_true(all(diag$group_table$cholesky_backend == "cusolver-batched"),
            "true batch groups should use cuSOLVER batched Cholesky")
assert_true("unique_designs" %in% names(diag$group_table),
            "group_table should expose exact-design reuse diagnostics")
assert_true(identical(as.integer(diag$group_table$unique_designs[[1]]), 1L),
            "same conditioning set should share one design inside the batch")
assert_true(identical(as.integer(diag$group_table$max_fits_per_design[[1]]),
                      length(targets)),
            "same conditioning set should report all targets on one design")
assert_true(identical(as.integer(diag$per_request_design_x_values), 0L),
            "true batch should not pack per-request duplicate design X")
assert_true(as.integer(diag$duplicate_design_x_values_avoided) > 0L,
            "true batch should report avoided duplicate design X values")
assert_true(as.integer(diag$algebraic_rss_count) > 0L,
            "true batch should score candidate lambdas with algebraic RSS")
assert_true(identical(as.integer(diag$candidate_residual_materialize_count), 0L),
            "candidate lambda scoring should not materialize residual vectors")
assert_true(as.integer(diag$winning_residual_materialize_count) > 0L,
            "true batch should materialize residuals for winning lambdas")

for (k in seq_along(targets)) {
  cpu <- fastspline_residual(data[, targets[[k]]],
                             data[, conditioning_sets[[k]], drop = FALSE],
                             fastspline_params = params)
  assert_true(max_abs_diff(batch$residuals[, k], residual_values(cpu)) < 1e-7,
              paste("residual", k, "should match CPU"))
  assert_true(max_abs_diff(batch$fitted[, k], cpu$fitted) < 1e-7,
              paste("fitted", k, "should match CPU"))
  rel_rss <- abs(batch$rss[[k]] - cpu$rss) / max(1, abs(cpu$rss))
  assert_true(rel_rss < 1e-8, paste("rss", k, "should match CPU"))
  assert_true(isTRUE(batch$diagnostics[[k]]$true_batched),
              paste("fit", k, "should be marked true_batched"))
}

cat("test_cuda_fastspline_true_batch_contract.R: PASS\n")
