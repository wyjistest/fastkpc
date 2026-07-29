fastkpc_full_cuda_phase5_require <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
}

fastkpc_full_cuda_phase5_evaluate_cpp <- function(
    prepared_setup, y, log_sp) {
  fastkpc_full_cuda_phase5_require(
    is.list(prepared_setup) &&
      identical(prepared_setup$formula_class, "additive-smooth") &&
      length(prepared_setup$penalty_blocks) > 1L,
    "Phase 5 requires an additive multi-penalty PreparedSSetup"
  )
  y <- as.numeric(y)
  log_sp <- as.numeric(log_sp)
  X <- as.matrix(prepared_setup$X)
  storage.mode(X) <- "double"
  fastkpc_full_cuda_phase5_require(
    length(y) == nrow(X) && all(is.finite(y)) &&
      length(log_sp) == length(prepared_setup$penalty_blocks) &&
      all(is.finite(log_sp)),
    "Phase 5 response or log-sp vector is malformed"
  )
  if (!is.null(prepared_setup$weights)) {
    fastkpc_full_cuda_phase5_require(
      length(prepared_setup$weights) == nrow(X) &&
        all(as.numeric(prepared_setup$weights) == 1),
      "Phase 5 currently supports only absent or unit weights"
    )
  }
  full_cuda_ci_multi_penalty_gcv_evaluate_cpp_native(
    X = X,
    y = y,
    penalty_blocks = prepared_setup$penalty_blocks,
    penalty_offsets = prepared_setup$penalty_offsets,
    penalty_ranks = prepared_setup$mgcv_penalty_rank_metadata,
    log_sp = log_sp,
    H = prepared_setup$H,
    constraint = prepared_setup$constraint
  )
}

fastkpc_full_cuda_phase5_optimize_cpp <- function(
    prepared_setup, y, keep_transcript = FALSE,
    convergence_tolerance = 1e-7, max_step_halving = 25L,
    max_iterations = 400L, max_newton_step = 5,
    boundary_probe_step = 2, max_boundary_probes = 5L,
    rank_tolerance = sqrt(.Machine$double.eps)) {
  fastkpc_full_cuda_phase5_require(
    is.list(prepared_setup) &&
      identical(prepared_setup$formula_class, "additive-smooth") &&
      length(prepared_setup$penalty_blocks) > 1L,
    "Phase 5 requires an additive multi-penalty PreparedSSetup"
  )
  mapping <- prepared_setup$sp_mapping
  mapping_is_identity <- is.null(mapping) ||
    (is.matrix(mapping) && nrow(mapping) == ncol(mapping) &&
       identical(unname(mapping), unname(diag(nrow(mapping)))))
  fastkpc_full_cuda_phase5_require(
    mapping_is_identity &&
      (length(prepared_setup$sp_mapping_offset) == 0L ||
         all(as.numeric(prepared_setup$sp_mapping_offset) == 0)) &&
      (length(prepared_setup$min_sp) == 0L ||
         all(as.numeric(prepared_setup$min_sp) == 0)),
    "Phase 5 optimizer requires the canonical identity smoothing mapping"
  )
  y <- as.numeric(y)
  X <- as.matrix(prepared_setup$X)
  storage.mode(X) <- "double"
  fastkpc_full_cuda_phase5_require(
    length(y) == nrow(X) && all(is.finite(y)),
    "Phase 5 optimizer response is malformed"
  )
  if (!is.null(prepared_setup$weights)) {
    fastkpc_full_cuda_phase5_require(
      length(prepared_setup$weights) == nrow(X) &&
        all(as.numeric(prepared_setup$weights) == 1),
      "Phase 5 currently supports only absent or unit weights"
    )
  }
  control <- list(
    convergence_tolerance = as.numeric(convergence_tolerance),
    max_step_halving = as.integer(max_step_halving),
    max_iterations = as.integer(max_iterations),
    max_newton_step = as.numeric(max_newton_step),
    boundary_probe_step = as.numeric(boundary_probe_step),
    max_boundary_probes = as.integer(max_boundary_probes),
    rank_tolerance = as.numeric(rank_tolerance),
    keep_transcript = isTRUE(keep_transcript)
  )
  full_cuda_ci_multi_penalty_gcv_optimize_cpp_native(
    X = X,
    y = y,
    penalty_blocks = prepared_setup$penalty_blocks,
    penalty_offsets = prepared_setup$penalty_offsets,
    penalty_ranks = prepared_setup$mgcv_penalty_rank_metadata,
    H = prepared_setup$H,
    constraint = prepared_setup$constraint,
    control = control
  )
}
