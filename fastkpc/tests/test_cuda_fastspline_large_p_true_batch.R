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

# Regression for the large-p (p > kSmallPRhsMaxDesignCols = 64) true-batched
# candidate RHS path. knots = 80 yields design_cols = 81, which routes GCV
# candidate solves through cusolverDnDpotrsBatched over group_size *
# lambda_count systems (not the small-p fused Cholesky kernel), while staying
# under kMaxTrueBatchedDesignCols = 128. This is the path introduced by
# "perf: batch fastspline candidate rhs solves across lambda grid" and is not
# covered by the small-p contract test.
set.seed(8081)
n <- 160
z <- runif(n, -2.5, 2.5)
data <- cbind(
  x1 = sin(z) + rnorm(n, sd = 0.08),
  x2 = z^2 / 3 + rnorm(n, sd = 0.08),
  x3 = tanh(z) + rnorm(n, sd = 0.08),
  z = z
)

targets <- c(1L, 2L, 3L)
conditioning_sets <- list(4L, 4L, 4L)
params <- list(knots = 80, lambda_count = 7, ridge = 1e-8)

batch <- fastspline_residual_batch_cuda(
  data,
  targets = targets,
  conditioning_sets = conditioning_sets,
  fastspline_params = params,
  fallback = FALSE
)

diag <- batch$batch_diagnostics
assert_true(is.data.frame(diag$group_table), "group_table should be a data frame")
assert_true(all(diag$group_table$design_cols > 64L),
            "knots=80 should route through the large-p (>64) candidate path")
assert_true(all(diag$group_table$design_cols <= 128L),
            "large-p design should still be true-batchable (<=128)")
assert_true(all(diag$group_table$true_batched),
            "large-p group should use the true batched solve")
assert_true(all(diag$group_table$cholesky_backend == "cusolver-batched"),
            "large-p candidate RHS should use cuSOLVER batched solves")
assert_true(identical(as.integer(diag$single_fit_calls), 0L),
            "large-p batch must not fall back to the single-fit path")
assert_true(identical(as.integer(diag$cpu_fallback_fits), 0L),
            "large-p batch must not fall back to CPU")
assert_true(as.integer(diag$algebraic_rss_count) > 0L,
            "large-p batch should score candidate lambdas with algebraic RSS")

for (k in seq_along(targets)) {
  cpu <- fastspline_residual(data[, targets[[k]]],
                             data[, conditioning_sets[[k]], drop = FALSE],
                             fastspline_params = params)
  assert_true(max_abs_diff(batch$residuals[, k], residual_values(cpu)) < 1e-7,
              paste("large-p residual", k, "should match CPU"))
  assert_true(max_abs_diff(batch$fitted[, k], cpu$fitted) < 1e-7,
              paste("large-p fitted", k, "should match CPU"))
  rel_rss <- abs(batch$rss[[k]] - cpu$rss) / max(1, abs(cpu$rss))
  assert_true(rel_rss < 1e-8, paste("large-p rss", k, "should match CPU"))
  assert_true(isTRUE(batch$diagnostics[[k]]$true_batched),
              paste("large-p fit", k, "should be marked true_batched"))
}

old_edf_trace_mode <- Sys.getenv("FASTKPC_FASTSPLINE_EDF_TRACE_MODE",
                                 unset = NA_character_)
Sys.setenv(FASTKPC_FASTSPLINE_EDF_TRACE_MODE = "cholesky_cuda")
on.exit({
  if (is.na(old_edf_trace_mode)) {
    Sys.unsetenv("FASTKPC_FASTSPLINE_EDF_TRACE_MODE")
  } else {
    Sys.setenv(FASTKPC_FASTSPLINE_EDF_TRACE_MODE = old_edf_trace_mode)
  }
}, add = TRUE)

cholesky_batch <- fastspline_residual_batch_cuda(
  data,
  targets = targets,
  conditioning_sets = conditioning_sets,
  fastspline_params = params,
  fallback = FALSE
)
cholesky_diag <- cholesky_batch$batch_diagnostics
assert_true(as.integer(cholesky_diag$edf_trace_cuda_candidate_count) > 0L,
            "cholesky CUDA EDF mode should score candidates on device")
assert_true(as.integer(cholesky_diag$edf_trace_full_inverse_skipped_count) > 0L,
            "cholesky CUDA EDF mode should skip candidate full inverses")
assert_true(as.numeric(cholesky_diag$candidate_inverse_values_avoided) > 0,
            "cholesky CUDA EDF mode should count avoided inverse values")
assert_true(identical(as.integer(cholesky_diag$inverse_solve_count), 0L),
            "cholesky CUDA EDF mode should avoid candidate inverse solves")
assert_true(as.numeric(cholesky_diag$factor_inverse_solve_sec) == 0,
            "cholesky CUDA EDF mode should not time candidate inverse solves")
assert_true(identical(
  as.integer(cholesky_diag$edf_trace_mode_full_inverse_count), 0L
), "cholesky CUDA EDF mode should not count full-inverse EDF candidates")
assert_true(identical(as.integer(cholesky_diag$edf_trace_cuda_fallback_count),
                      0L),
            "cholesky CUDA EDF mode should not fall back in this scenario")

for (k in seq_along(targets)) {
  cpu <- fastspline_residual(data[, targets[[k]]],
                             data[, conditioning_sets[[k]], drop = FALSE],
                             fastspline_params = params)
  assert_true(max_abs_diff(cholesky_batch$residuals[, k],
                           residual_values(cpu)) < 1e-7,
              paste("cholesky CUDA residual", k, "should match CPU"))
  assert_true(max_abs_diff(cholesky_batch$fitted[, k], cpu$fitted) < 1e-7,
              paste("cholesky CUDA fitted", k, "should match CPU"))
  rel_rss <- abs(cholesky_batch$rss[[k]] - cpu$rss) / max(1, abs(cpu$rss))
  assert_true(rel_rss < 1e-8,
              paste("cholesky CUDA rss", k, "should match CPU"))
}

cat("test_cuda_fastspline_large_p_true_batch.R: PASS\n")
