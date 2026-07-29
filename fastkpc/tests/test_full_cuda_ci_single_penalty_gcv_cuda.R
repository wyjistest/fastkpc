source("fastkpc/R/mgcv_extract_oracle.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_single_penalty_gcv.R")
source("fastkpc/R/full_cuda_ci_phase4_backend.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS", unset = "0"), "1")) {
  cat("SKIP Phase 4 single-penalty CUDA GCV: FASTKPC_RUN_CUDA_TESTS != 1\n")
  quit(save = "no", status = 0)
}
if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP Phase 4 single-penalty CUDA GCV: mgcv unavailable\n")
  quit(save = "no", status = 0)
}
if (!isTRUE(tryCatch(fastkpc_cuda_available(), error = function(error) FALSE))) {
  cat("SKIP Phase 4 single-penalty CUDA GCV: CUDA unavailable\n")
  quit(save = "no", status = 0)
}

set.seed(40562)
n <- 112L
x1 <- stats::runif(n, -2.5, 2.5)
x2 <- stats::runif(n, -1.5, 1.5)
data <- data.frame(
  y1 = sin(x1) + 0.25 * x2 + stats::rnorm(n, sd = 0.08),
  y2 = cos(1.5 * x1) - 0.1 * x2 + stats::rnorm(n, sd = 0.08),
  y3 = x1 * x2 + stats::rnorm(n, sd = 0.08),
  x1 = x1,
  x2 = x2
)
setup <- fastkpc_mgcv_extract_setup(
  formula = y1 ~ s(x1, x2),
  data = data,
  sp = 1,
  method = "GCV.Cp",
  target = 1L,
  S = c(4L, 5L)
)
prepared <- list(
  X = setup$X,
  gram_matrix = crossprod(setup$X),
  penalty_blocks = setup$S,
  penalty_offsets = setup$off,
  penalty_ranks = setup$rank
)
Y <- as.matrix(data[c("y1", "y2", "y3")])
sp_grid <- sort(unique(c(
  exp(c(-40, 40)),
  exp(seq(-25, 25, length.out = 81L)),
  vapply(seq_len(ncol(Y)), function(index) {
    local_data <- data.frame(y = Y[, index], x1 = x1, x2 = x2)
    as.numeric(mgcv::gam(
      y ~ s(x1, x2), data = local_data, method = "GCV.Cp"
    )$sp)
  }, numeric(1L))
)))

reference <- fastkpc_full_cuda_phase4_magic1_batch(
  prepared, Y, target_ids = 1:3, keep_transcript = TRUE
)
backend_request <- fastkpc_full_cuda_phase4_cuda_native_input(
  prepared, Y, target_ids = 1:3
)$native
backend_cpu <- fastkpc_full_cuda_phase4_backend_cpu_request(backend_request)
backend_cpu$target <- backend_request$target_ids
assert_true(
  identical(
    backend_cpu[names(reference$target_results)],
    reference$target_results
  ),
  "Phase 4 backend CPU baseline must preserve the r-cpu-spectral route"
)

penalty_matrix <- tcrossprod(reference$spectral$magic_penalty_root)
root_log_sp <- sort(unique(c(
  -25, -5, log(reference$spectral$initial_sp),
  reference$target_results$log_sp, 5, 25
)))
cuda_roots <- full_cuda_ci_single_penalty_mroot_cuda(
  penalty_matrix,
  penalty_rank = reference$spectral$penalty_rank,
  log_sp = root_log_sp
)
reference_roots <- lapply(root_log_sp, function(value) {
  factor <- suppressWarnings(chol(
    exp(value) * penalty_matrix, pivot = TRUE, tol = -1
  ))
  pivot <- attr(factor, "pivot")
  rank <- attr(factor, "rank")
  if (rank < ncol(factor)) {
    trailing <- (rank + 1L):ncol(factor)
    factor[trailing, trailing] <- 0
  }
  list(
    rank = rank,
    pivot = pivot,
    root = t(factor[
      seq_len(reference$spectral$penalty_rank), order(pivot), drop = FALSE
    ])
  )
})
assert_true(
  identical(cuda_roots$rank, vapply(reference_roots, `[[`, integer(1L), "rank")),
  "Phase 4 CUDA dynamic mroot ranks must match DPSTF2"
)
for (candidate in seq_along(root_log_sp)) {
  expected <- reference_roots[[candidate]]
  assert_true(
    identical(cuda_roots$pivot[, candidate], expected$pivot),
    "Phase 4 CUDA dynamic mroot pivots must match DPSTF2"
  )
  relative_root_error <- max(abs(
    cuda_roots$root[, , candidate] - expected$root
  )) / max(1, max(abs(expected$root)))
  assert_true(
    relative_root_error <= 1e-12,
    "Phase 4 CUDA dynamic mroot must match the pinned LAPACK root"
  )
}

cuda <- fastkpc_full_cuda_phase4_cuda_batch(
  prepared, Y, target_ids = 1:3, sp_grid = sp_grid,
  materialize_grid = TRUE, keep_transcript = TRUE
)

assert_true(
  identical(cuda$diagnostics$sp_selection_backend_executed, "cuda") &&
    identical(cuda$diagnostics$gcv_score_backend_executed, "cuda") &&
    identical(
      cuda$diagnostics$optimizer_backend_executed,
      "cuda-spectral-risk-gated-exact-replay"
    ) && identical(
      cuda$diagnostics$exact_replay_backend_executed,
      "cuda-dpstf2-lapack-3.12-dgesdd"
    ),
  "Phase 4 must report CUDA scoring and selection"
)
assert_true(
  cuda$diagnostics$legacy_mgcv_target_calls == 0L &&
    cuda$diagnostics$cpu_score_count == 0L &&
    cuda$diagnostics$cpu_optimizer_count == 0L &&
    cuda$diagnostics$fallback_count == 0L,
  "Phase 4 CUDA primitive must not use CPU scoring or fallback"
)
assert_true(
  isTRUE(cuda$diagnostics$target_rhs_projected_on_cuda) &&
    isTRUE(cuda$diagnostics$target_selection_on_cuda) &&
    cuda$diagnostics$projection_gemm_count == 2L &&
    cuda$diagnostics$mgcv_qt_y_kernel_launch_count == 1L &&
    cuda$diagnostics$optimizer_kernel_launch_count > 1L &&
    cuda$diagnostics$exact_endpoint_kernel_launch_count > 0L &&
    cuda$diagnostics$exact_endpoint_comparison_count ==
      cuda$diagnostics$exact_endpoint_svd_call_count &&
    cuda$diagnostics$spectral_optimizer_target_count == ncol(Y) &&
    cuda$diagnostics$exact_replay_target_count ==
      cuda$diagnostics$exact_derivative_refresh_count &&
    cuda$diagnostics$spectral_only_target_count +
      cuda$diagnostics$exact_replay_target_count == ncol(Y) &&
    isTRUE(cuda$diagnostics$optimizer_target_coverage_complete) &&
    all(c(
      cuda$diagnostics$exact_replay_endpoint_risk_count,
      cuda$diagnostics$exact_replay_convergence_risk_count,
      cuda$diagnostics$exact_replay_boundary_risk_count,
      cuda$diagnostics$exact_replay_numerical_risk_count
    ) <= cuda$diagnostics$exact_replay_target_count) &&
    cuda$diagnostics$augmented_eigensolver_call_count == 0L &&
    cuda$diagnostics$augmented_objective_kernel_launch_count == 0L,
  paste0(
    "Phase 4 target selection must cover every target through the CUDA ",
    "spectral/exact-replay optimizer without the rejected normal-matrix ",
    "eigensolver"
  )
)
assert_true(
  cuda$diagnostics$exact_mroot_kernel_launch_count == 1L &&
    cuda$diagnostics$exact_svd_call_count == length(sp_grid) &&
    cuda$diagnostics$exact_objective_kernel_launch_count == 1L,
  "Phase 4 materialized objective grid must use dynamic mroot and CUDA SVD"
)
assert_true(
  isTRUE(cuda$diagnostics$cublas_pedantic_math) &&
    isTRUE(cuda$diagnostics$cublas_atomics_not_allowed) &&
    isTRUE(cuda$diagnostics$cublas_user_workspace_installed) &&
    cuda$diagnostics$device_allocation_count ==
      cuda$diagnostics$stream_ordered_allocation_count &&
    cuda$diagnostics$synchronous_allocation_count == 0L,
  "Phase 4 CUDA primitive must preserve the numerical environment contract"
)

spectral <- reference$spectral
cpu_grid <- fastkpc_full_cuda_phase4_exact_reference_grid(
  spectral, Y, log(sp_grid)
)
for (target in seq_len(ncol(Y))) {
  assert_true(
    max(abs(cuda$grid$rss[target, ] - cpu_grid$rss[target, ])) < 1e-8,
    "Phase 4 CUDA RSS grid must match the augmented-SVD reference"
  )
  assert_true(
    max(abs(cuda$grid$edf[target, ] - cpu_grid$edf[target, ])) < 1e-8,
    "Phase 4 CUDA EDF grid must match the augmented-SVD reference"
  )
  assert_true(
    max(abs(cuda$grid$score[target, ] - cpu_grid$score[target, ])) < 1e-8,
    "Phase 4 CUDA score grid must match the augmented-SVD reference"
  )
}

fixed_grid_indices <- unique(as.integer(round(seq(
  1, length(sp_grid), length.out = 5L
))))
exact_errors <- c(rss = 0, edf = 0, score = 0)
for (target in seq_len(ncol(Y))) {
  local_data <- data.frame(y = Y[, target], x1 = x1, x2 = x2)
  for (candidate in fixed_grid_indices) {
    fixed <- mgcv::gam(
      y ~ s(x1, x2), data = local_data, method = "GCV.Cp",
      sp = sp_grid[[candidate]]
    )
    exact_errors[["rss"]] <- max(
      exact_errors[["rss"]],
      abs(cuda$grid$rss[target, candidate] - sum(stats::residuals(fixed)^2))
    )
    exact_errors[["edf"]] <- max(
      exact_errors[["edf"]],
      abs(cuda$grid$edf[target, candidate] - sum(fixed$edf))
    )
    exact_errors[["score"]] <- max(
      exact_errors[["score"]],
      abs(cuda$grid$score[target, candidate] - as.numeric(fixed$gcv.ubre))
    )
  }
}
assert_true(
  all(exact_errors <= 1e-8),
  "Phase 4 exact CUDA grid must satisfy the pinned mgcv objective contract"
)

assert_true(
  max(abs(cuda$targets$log_sp -
          reference$target_results$log_sp)) < 5e-6,
  "Phase 4 CUDA selected log-sp must match the magic1 reference"
)
assert_true(
  max(abs(cuda$targets$score -
          reference$target_results$score)) < 1e-8,
  "Phase 4 CUDA selected score must match the magic1 reference"
)
assert_true(
  identical(cuda$targets$iteration_count,
            reference$target_results$iteration_count),
  "Phase 4 CUDA optimizer iterations must match the magic1 reference"
)
assert_true(
  all(vapply(cuda$transcripts, nrow, integer(1L)) > 0L),
  "Phase 4 CUDA transcript must be present for requested targets"
)
assert_true(
  all(cuda$transcript_overflow == 0L),
  "Phase 4 CUDA transcript must not overflow"
)

repeat_cuda <- fastkpc_full_cuda_phase4_cuda_batch(
  prepared, Y, target_ids = 1:3
)
assert_true(
  identical(serialize(cuda$targets, NULL),
            serialize(repeat_cuda$targets, NULL)),
  "Phase 4 CUDA compact optimizer results must be deterministic"
)

multi_batches <- list(
  first = list(setup = prepared, Y = Y, target_ids = 1:3),
  second = list(
    setup = prepared, Y = Y[, c(3L, 1L), drop = FALSE],
    target_ids = c(3L, 1L)
  ),
  third = list(
    setup = prepared, Y = Y[, 2:3, drop = FALSE],
    target_ids = 2:3
  )
)
direct_compact <- lapply(multi_batches, function(batch) {
  fastkpc_full_cuda_phase4_cuda_batch(
    batch$setup, batch$Y, target_ids = batch$target_ids
  )
})
fused_compact <- fastkpc_full_cuda_phase4_cuda_batches(
  multi_batches, concurrency = 1L
)
assert_true(
  identical(
    fused_compact$diagnostics$execution_strategy,
    "cuda-cross-setup-fused-exact-replay"
  ) && isTRUE(fused_compact$diagnostics$fused_exact_replay_executed) &&
    fused_compact$diagnostics$fused_exact_replay_target_count ==
      sum(vapply(fused_compact$setups, function(setup) {
        setup$diagnostics$exact_replay_target_count
      }, integer(1L))) &&
    all(vapply(fused_compact$setups, function(setup) {
      isTRUE(setup$diagnostics$optimizer_target_coverage_complete)
    }, logical(1L))),
  "Phase 4 compact multi-setup execution must prove fused CUDA coverage"
)
for (index in seq_along(multi_batches)) {
  assert_true(
    identical(
      serialize(fused_compact$setups[[index]]$targets, NULL),
      serialize(direct_compact[[index]]$targets, NULL)
    ),
    paste0(
      "Phase 4 fused compact targets must be bit-identical to direct setup ",
      index
    )
  )
}
multi_grid <- sp_grid[fixed_grid_indices]
serial_multi <- fastkpc_full_cuda_phase4_cuda_batches(
  multi_batches,
  concurrency = 1L,
  sp_grids = multi_grid,
  materialize_grid = TRUE,
  keep_transcript = TRUE
)
for (repetition in seq_len(5L)) {
  concurrent_multi <- fastkpc_full_cuda_phase4_cuda_batches(
    multi_batches,
    concurrency = 3L,
    sp_grids = multi_grid,
    materialize_grid = TRUE,
    keep_transcript = TRUE
  )
  diagnostics <- concurrent_multi$diagnostics
  assert_true(
    identical(names(concurrent_multi$setups), names(multi_batches)) &&
      diagnostics$setup_count == 3L &&
      diagnostics$requested_concurrency == 3L &&
      diagnostics$worker_count == 3L &&
      diagnostics$worker_device_bind_count == 3L &&
      diagnostics$max_host_calls_in_flight == 3L &&
      is.finite(diagnostics$wall_host_ms) && diagnostics$wall_host_ms > 0 &&
      is.finite(diagnostics$host_overlap_factor) &&
      diagnostics$host_overlap_factor > 0,
    "Phase 4 multi-setup diagnostics must prove three bound CUDA workers"
  )
  for (index in seq_along(multi_batches)) {
    expected <- serial_multi$setups[[index]]
    observed <- concurrent_multi$setups[[index]]
    assert_true(
      identical(serialize(observed$targets, NULL),
                serialize(expected$targets, NULL)) &&
        identical(serialize(observed$grid, NULL),
                  serialize(expected$grid, NULL)) &&
        identical(serialize(observed$transcripts, NULL),
                  serialize(expected$transcripts, NULL)) &&
        identical(observed$transcript_overflow,
                  expected$transcript_overflow),
      paste0(
        "Phase 4 concurrent setup output must be bit-identical on repetition ",
        repetition, ", setup ", index
      )
    )
    assert_true(
      observed$diagnostics$legacy_mgcv_target_calls == 0L &&
        observed$diagnostics$cpu_score_count == 0L &&
        observed$diagnostics$cpu_optimizer_count == 0L &&
        observed$diagnostics$fallback_count == 0L,
      "Phase 4 concurrent setup must not use CPU scoring or fallback"
    )
  }
}

cat("PASS Phase 4 single-penalty CUDA GCV\n")
