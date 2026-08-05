source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cuda.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS", unset = "0"), "1")) {
  cat("SKIP default-Inf extended CUDA capacity qualification\n")
  quit(save = "no", status = 0L)
}
if (!isTRUE(tryCatch(fastkpc_cuda_available(), error = function(error) FALSE))) {
  cat("SKIP default-Inf level-9 CUDA capacity qualification: CUDA unavailable\n")
  quit(save = "no", status = 0L)
}

data <- as.matrix(readRDS(file.path(
  "fastkpc", "artifacts", "kpc_tprs_real_zhu",
  "cancer_RD-causalDiscoveryInput.rds"
)))
storage.mode(data) <- "double"
control <- list(
  convergence_tolerance = 1e-7,
  max_step_halving = 25L,
  max_iterations = 400L,
  max_newton_step = 5,
  boundary_probe_step = 2,
  max_boundary_probes = 5L,
  rank_tolerance = sqrt(.Machine$double.eps),
  keep_transcript = TRUE
)

small_setup <- full_cuda_ci_native_setup_native(data[, seq_len(3L), drop = FALSE])
small_geometry <- full_cuda_ci_native_geometry_prepare_native(
  small_setup$X, small_setup$penalty_blocks,
  small_setup$penalty_offsets, small_setup$penalty_ranks
)
small_Y <- data[, 4:5, drop = FALSE]
small_candidate <- full_cuda_ci_multi_penalty_capacity_qualify_native(
  small_setup, small_geometry, small_Y
)
small_reference <- full_cuda_ci_multi_penalty_gcv_optimize_cuda_native(
  small_setup$X, small_Y, small_geometry$magic_qr_packed,
  small_geometry$magic_tau, small_geometry$magic_r,
  small_geometry$magic_pivot, small_geometry$penalty_roots,
  small_geometry$penalty_matrices, small_setup$penalty_ranks,
  small_geometry$initial_log_sp, control = control
)
small_fields <- c(
  "selected_log_sp", "condition", "numerical_rank", "optimizer_status",
  "optimizer_iterations", "score_calls", "objective_calls",
  "step_halving_count", "boundary_probe_count", "boundary_accepted_count",
  "boundary_status"
)
assert_true(
  identical(small_candidate$capacity_bucket, "q64-p7") &&
    identical(small_candidate$coefficient_dim, 28L) &&
    all(vapply(small_fields, function(field) {
      identical(small_candidate[[field]], small_reference[[field]])
    }, logical(1L))),
  "q64-p7 capacity backend changed the accepted optimizer output"
)

S <- seq_len(9L)
targets <- 10:11
setup <- full_cuda_ci_native_setup_native(data[, S, drop = FALSE])
geometry <- full_cuda_ci_native_geometry_prepare_native(
  setup$X, setup$penalty_blocks, setup$penalty_offsets,
  setup$penalty_ranks
)
assert_true(
  identical(dim(setup$X), c(351L, 82L)) &&
    length(setup$penalty_blocks) == 9L &&
    identical(as.integer(setup$penalty_ranks), rep.int(8L, 9L)),
  "level-9 native setup shape changed"
)

candidate <- full_cuda_ci_multi_penalty_capacity_qualify_native(
  setup, geometry, data[, targets, drop = FALSE]
)
assert_true(
  identical(candidate$schema_version,
            "full-cuda-ci-multi-penalty-capacity-qualification-v1") &&
    identical(candidate$capacity_bucket, "q192-p21") &&
    identical(candidate$max_concurrent_setups, 8L) &&
    identical(candidate$coefficient_dim, 82L) &&
    identical(candidate$penalty_count, 9L) &&
    identical(candidate$target_count, 2L) &&
    all(candidate$optimizer_status == 0L) &&
    identical(candidate$diagnostics$cpu_objective_count, 0L) &&
    identical(candidate$diagnostics$cpu_optimizer_count, 0L) &&
    identical(candidate$diagnostics$cpu_multi_penalty_solve_count, 0L) &&
    identical(candidate$diagnostics$fallback_count, 0L) &&
    identical(candidate$diagnostics$cuda_error_count, 0L) &&
    isTRUE(candidate$diagnostics$true_batched_kernel) &&
    isTRUE(candidate$diagnostics$independent_target_states),
  "level-9 capacity backend authority gate failed"
)

Y <- data[, targets, drop = FALSE]
references <- lapply(seq_len(ncol(Y)), function(target_index) {
  full_cuda_ci_multi_penalty_gcv_optimize_cpp_native(
    setup$X, Y[, target_index], setup$penalty_blocks,
    setup$penalty_offsets, setup$penalty_ranks, control = control
  )
})
reference_integer <- function(name) {
  as.integer(vapply(references, `[[`, integer(1L), name))
}
assert_true(
  identical(as.integer(candidate$optimizer_iterations),
            reference_integer("optimizer_iterations")) &&
    identical(as.integer(candidate$score_calls),
              reference_integer("score_calls")) &&
    identical(as.integer(candidate$objective_calls),
              reference_integer("objective_calls")) &&
    identical(as.integer(candidate$step_halving_count),
              reference_integer("step_halving_count")) &&
    identical(as.integer(candidate$boundary_probe_count),
              reference_integer("boundary_probe_count")) &&
    identical(as.integer(candidate$boundary_accepted_count),
              reference_integer("boundary_accepted_count")) &&
    identical(candidate$boundary_status,
              do.call(cbind, lapply(references, `[[`, "boundary_status"))),
  "level-9 capacity optimizer trajectory changed"
)
reference_log_sp <- do.call(cbind, lapply(
  references, `[[`, "selected_log_sp"
))
max_log_sp_error <- max(abs(candidate$selected_log_sp - reference_log_sp))
assert_true(
  is.finite(max_log_sp_error) && max_log_sp_error <= 1e-6,
  "level-9 capacity selected log-sp exceeded the qualified tolerance"
)

assert_capacity_authority <- function(candidate, bucket, concurrency, q,
                                      penalty_count, label) {
  assert_true(
    identical(candidate$capacity_bucket, bucket) &&
      identical(candidate$max_concurrent_setups, concurrency) &&
      identical(candidate$coefficient_dim, q) &&
      identical(candidate$penalty_count, penalty_count) &&
      identical(candidate$target_count, 2L) &&
      all(candidate$optimizer_status == 0L) &&
      all(is.finite(candidate$selected_log_sp)) &&
      identical(candidate$diagnostics$cpu_objective_count, 0L) &&
      identical(candidate$diagnostics$cpu_optimizer_count, 0L) &&
      identical(candidate$diagnostics$cpu_multi_penalty_solve_count, 0L) &&
      identical(candidate$diagnostics$fallback_count, 0L) &&
      identical(candidate$diagnostics$cuda_error_count, 0L) &&
      isTRUE(candidate$diagnostics$true_batched_kernel) &&
      isTRUE(candidate$diagnostics$independent_target_states),
    paste0(label, " capacity backend authority gate failed")
  )
}

setup384 <- full_cuda_ci_native_setup_native(data[, seq_len(22L), drop = FALSE])
geometry384 <- full_cuda_ci_native_geometry_prepare_native(
  setup384$X, setup384$penalty_blocks, setup384$penalty_offsets,
  setup384$penalty_ranks
)
elapsed384 <- system.time(candidate384 <-
  full_cuda_ci_multi_penalty_capacity_qualify_native(
    setup384, geometry384, data[, 23:24, drop = FALSE]
  ))[["elapsed"]]
assert_true(
  identical(dim(setup384$X), c(351L, 199L)) &&
    length(setup384$penalty_blocks) == 22L,
  "q384-p42 native setup shape changed"
)
assert_capacity_authority(
  candidate384, "q384-p42", 2L, 199L, 22L, "q384-p42"
)

set.seed(620559L)
maximum_data <- matrix(stats::rnorm(600L * 64L), nrow = 600L, ncol = 64L)
storage.mode(maximum_data) <- "double"
maximum_setup <- full_cuda_ci_native_setup_native(
  maximum_data[, seq_len(62L), drop = FALSE]
)
maximum_geometry <- full_cuda_ci_native_geometry_prepare_native(
  maximum_setup$X, maximum_setup$penalty_blocks,
  maximum_setup$penalty_offsets, maximum_setup$penalty_ranks
)
elapsed559 <- system.time(maximum_candidate <-
  full_cuda_ci_multi_penalty_capacity_qualify_native(
    maximum_setup, maximum_geometry,
    maximum_data[, 63:64, drop = FALSE]
  ))[["elapsed"]]
assert_true(
  identical(dim(maximum_setup$X), c(600L, 559L)) &&
    length(maximum_setup$penalty_blocks) == 62L,
  "q559-p62 native setup shape changed"
)
assert_capacity_authority(
  maximum_candidate, "q559-p62", 1L, 559L, 62L, "q559-p62"
)

cat(
  "PASS default-Inf extended CUDA capacity qualification; ",
  "q192_max_log_sp_error=", format(max_log_sp_error, digits = 7L),
  "; q384_sec=", elapsed384, "; q559_sec=", elapsed559, "\n", sep = ""
)
