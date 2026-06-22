source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP mgcvExtractGPU spectral batch score: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

set.seed(2601)
n <- 84
s1 <- stats::runif(n, -2.5, 2.5)
s2 <- stats::runif(n, -1.5, 1.5)
data <- cbind(
  y1 = sin(s1) + 0.2 * s2 + stats::rnorm(n, sd = 0.04),
  y2 = cos(s1) - 0.1 * s2 + stats::rnorm(n, sd = 0.04),
  y3 = s1 * s2 + stats::rnorm(n, sd = 0.04),
  s1 = s1,
  s2 = s2
)
sp_grid <- exp(seq(log(1e-4), log(1e4), length.out = 17L))
context <- fastkpc_precision_create_execution_context(
  data = data,
  residual_cache = TRUE,
  runtime_capabilities = fastkpc_precision_runtime_capabilities(),
  execution_engine = "cuda"
)
fastkpc_precision_init_cache_stats(context)
prepared_setup <- fastkpc_prepare_gpu_setup_state(
  data = data,
  S = c(4L, 5L),
  template_target = 1L,
  sp_grid = sp_grid,
  context = context
)
prepared_spectral <- fastkpc_prepare_gpu_spectral_state(
  prepared_setup,
  context = context
)
targets <- c(1L, 2L, 3L)

scalar <- lapply(targets, function(target) {
  fastkpc_score_gpu_target_from_prepared(
    data = data,
    target = target,
    prepared_setup = prepared_setup,
    prepared_spectral = prepared_spectral,
    sp_grid = sp_grid
  )
})
batched <- fastkpc_score_gpu_targets_from_prepared(
  data = data,
  targets = targets,
  prepared_setup = prepared_setup,
  prepared_spectral = prepared_spectral,
  sp_grid = sp_grid
)

assert_true(length(batched) == length(targets),
            "batched scorer should return one entry per target")
for (j in seq_along(targets)) {
  assert_true(identical(batched[[j]]$target, scalar[[j]]$target),
              "batched scorer should preserve target id")
  assert_true(identical(batched[[j]]$selected_grid_index,
                        scalar[[j]]$selected_grid_index),
              "batched scorer should preserve selected grid index")
  assert_true(abs(batched[[j]]$sp - scalar[[j]]$sp) < 1e-14,
              "batched scorer should preserve selected sp")
  assert_true(abs(batched[[j]]$score - scalar[[j]]$score) < 1e-10,
              "batched scorer should preserve selected GCV score")
  assert_true(abs(batched[[j]]$edf - scalar[[j]]$edf) < 1e-10,
              "batched scorer should preserve selected EDF")
  assert_true(max(abs(batched[[j]]$grid$rss - scalar[[j]]$grid$rss)) < 1e-8,
              "batched scorer should preserve RSS grid")
  assert_true(max(abs(batched[[j]]$grid$edf - scalar[[j]]$grid$edf)) < 1e-10,
              "batched scorer should preserve EDF grid")
  assert_true(max(abs(batched[[j]]$grid$gcv - scalar[[j]]$grid$gcv)) < 1e-8,
              "batched scorer should preserve GCV grid")
}

batch_fit <- fastkpc_mgcv_extract_gpu_gcv_for_targets(
  data = data,
  targets = targets,
  S = c(4L, 5L),
  sp_grid = sp_grid,
  context = context,
  prepared_setup = prepared_setup,
  prepared_spectral = prepared_spectral,
  solve_backend = "cpu"
)
assert_true(isTRUE(batch_fit$fit$same_setup_target_batch_used),
            "target wrapper should still report same-setup target batch")
assert_true(identical(batch_fit$fit$gcv_score_batch_used, TRUE),
            "target wrapper should report batched GCV scoring")
assert_true(all(batch_fit$selected_grid_index ==
                  vapply(scalar, `[[`, integer(1L), "selected_grid_index")),
            "target wrapper should use batched scorer selected grid indices")
assert_true(max(abs(batch_fit$sp -
                      vapply(scalar, `[[`, numeric(1L), "sp"))) < 1e-14,
            "target wrapper should use batched scorer selected sp")

cat("PASS mgcvExtractGPU spectral batch score\n")
