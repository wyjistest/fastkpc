source("fastkpc/R/mgcv_extract_oracle.R")
source("fastkpc/R/full_cuda_ci_single_penalty_gcv.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP Phase 4 single-penalty GCV reference: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

set.seed(40561)
n <- 96L
x1 <- stats::runif(n, -2.5, 2.5)
x2 <- stats::runif(n, -1.5, 1.5)
data <- data.frame(
  y1 = sin(x1) + 0.25 * x2 + stats::rnorm(n, sd = 0.08),
  y2 = cos(1.5 * x1) - 0.1 * x2 + stats::rnorm(n, sd = 0.08),
  x1 = x1,
  x2 = x2
)
formula <- y1 ~ s(x1, x2)
setup <- fastkpc_mgcv_extract_setup(
  formula = formula,
  data = data,
  sp = 1,
  method = "GCV.Cp",
  target = 1L,
  S = c(3L, 4L)
)
prepared <- list(
  X = setup$X,
  gram_matrix = crossprod(setup$X),
  penalty_blocks = setup$S,
  penalty_offsets = setup$off,
  penalty_ranks = setup$rank
)
spectral <- fastkpc_full_cuda_phase4_spectral_prepare(prepared)

expected_initial_sp <- mgcv:::initial.sp(
  setup$X, setup$S, setup$off
)
assert_true(
  abs(log(spectral$initial_sp) - log(expected_initial_sp)) < 1e-13,
  "Phase 4 initial.sp replica must match mgcv"
)
assert_true(
  sum(spectral$eigenvalues == 0) ==
    ncol(setup$X) - as.integer(setup$rank),
  "Phase 4 spectral state must preserve the authenticated penalty nullity"
)

projection <- fastkpc_full_cuda_phase4_target_projection(
  prepared, spectral, data$y1
)
legacy <- mgcv::gam(formula, data = data, method = "GCV.Cp")
at_oracle <- fastkpc_full_cuda_phase4_objective(
  log_sp = log(as.numeric(legacy$sp)),
  eigenvalues = spectral$eigenvalues,
  squared_projection = projection$squared_projection,
  y_squared_norm = projection$y_squared_norm,
  n = n
)
assert_true(
  abs(at_oracle$score - as.numeric(legacy$gcv.ubre)) < 1e-10,
  "Phase 4 score must match mgcv at oracle sp"
)
assert_true(
  abs(at_oracle$edf - sum(legacy$edf)) < 1e-10,
  "Phase 4 EDF must match mgcv at oracle sp"
)

epsilon <- 1e-5
lower <- fastkpc_full_cuda_phase4_objective(
  log(as.numeric(legacy$sp)) - epsilon,
  spectral$eigenvalues,
  projection$squared_projection,
  projection$y_squared_norm,
  n
)
upper <- fastkpc_full_cuda_phase4_objective(
  log(as.numeric(legacy$sp)) + epsilon,
  spectral$eigenvalues,
  projection$squared_projection,
  projection$y_squared_norm,
  n
)
finite_difference_gradient <- (upper$score - lower$score) / (2 * epsilon)
assert_true(
  abs(at_oracle$gradient - finite_difference_gradient) < 1e-8,
  "Phase 4 analytic score gradient must match finite differences"
)

optimized <- fastkpc_full_cuda_phase4_magic1_optimize(
  spectral = spectral,
  squared_projection = projection$squared_projection,
  y_squared_norm = projection$y_squared_norm,
  keep_transcript = TRUE
)
assert_true(
  abs(log(optimized$sp) - log(as.numeric(legacy$sp))) < 1e-6,
  "Phase 4 magic1 replica must match mgcv selected log-sp"
)
assert_true(
  abs(optimized$score - as.numeric(legacy$gcv.ubre)) < 1e-8,
  "Phase 4 magic1 score must match mgcv"
)
assert_true(
  optimized$iteration_count == legacy$mgcv.conv$iter,
  "Phase 4 magic1 iteration count must match mgcv"
)
assert_true(
  optimized$score_call_count == legacy$mgcv.conv$score.calls,
  "Phase 4 magic1 score-call count must match mgcv"
)
assert_true(
  identical(optimized$fully_converged,
            isTRUE(legacy$mgcv.conv$fully.converged)),
  "Phase 4 magic1 convergence state must match mgcv"
)
assert_true(
  nrow(optimized$transcript) > optimized$iteration_count,
  "Phase 4 transcript must preserve trial and iteration states"
)
assert_true(
  all(c("objective", "gradient", "hessian", "proposed_step",
        "accepted", "step_source") %in% names(optimized$transcript)),
  "Phase 4 transcript must preserve optimizer evidence"
)

batch <- fastkpc_full_cuda_phase4_magic1_batch(
  prepared_setup = prepared,
  Y = as.matrix(data[c("y1", "y2")]),
  target_ids = c(1L, 2L)
)
assert_true(
  nrow(batch$target_results) == 2L &&
    all(is.finite(batch$target_results$sp)),
  "Phase 4 reference batch must select one finite sp per target"
)

cat("PASS Phase 4 single-penalty GCV reference\n")
