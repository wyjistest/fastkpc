fastkpc_full_cuda_phase6_require <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
}

fastkpc_full_cuda_phase6_prepare <- function(prepared_setup) {
  fastkpc_full_cuda_phase6_require(
    is.list(prepared_setup) &&
      identical(prepared_setup$formula_class, "additive-smooth") &&
      length(prepared_setup$penalty_blocks) > 1L,
    "Phase 6 requires an additive multi-penalty PreparedSSetup"
  )
  X <- as.matrix(prepared_setup$X)
  storage.mode(X) <- "double"
  p <- ncol(X)
  penalty_count <- length(prepared_setup$penalty_blocks)
  ranks <- as.integer(prepared_setup$mgcv_penalty_rank_metadata)
  offsets <- as.integer(prepared_setup$penalty_offsets)
  constraint <- prepared_setup$constraint
  constraint_is_identity <- is.null(constraint) ||
    (is.matrix(constraint) && nrow(constraint) == 0L && ncol(constraint) == p)
  fastkpc_full_cuda_phase6_require(
    is.numeric(X) && nrow(X) > p && p > 0L && p <= 64L &&
      all(is.finite(X)) && penalty_count <= 7L &&
      length(ranks) == penalty_count && all(ranks > 0L) &&
      length(offsets) == penalty_count && constraint_is_identity &&
      is.null(prepared_setup$H),
    "Phase 6 setup is outside the canonical CUDA envelope"
  )
  if (!is.null(prepared_setup$weights)) {
    fastkpc_full_cuda_phase6_require(
      length(prepared_setup$weights) == nrow(X) &&
        all(as.numeric(prepared_setup$weights) == 1),
      "Phase 6 supports only absent or unit weights"
    )
  }

  magic_qr <- qr(X, LAPACK = TRUE)
  magic_r <- qr.R(magic_qr, complete = FALSE)
  magic_qr_packed <- as.matrix(magic_qr$qr)
  magic_tau <- as.numeric(magic_qr$qraux)
  magic_pivot <- as.integer(magic_qr$pivot)
  mroot <- get("mroot", envir = asNamespace("mgcv"))
  penalty_roots <- lapply(seq_len(penalty_count), function(index) {
    block <- as.matrix(prepared_setup$penalty_blocks[[index]])
    storage.mode(block) <- "double"
    root <- mroot(block, rank = ranks[[index]], method = "chol")
    full_root <- matrix(0, p, ranks[[index]])
    rows <- offsets[[index]] + seq_len(nrow(root)) - 1L
    full_root[rows, ] <- root
    result <- full_root[magic_pivot, , drop = FALSE]
    storage.mode(result) <- "double"
    result
  })
  penalty_matrices <- lapply(penalty_roots, function(root) {
    result <- tcrossprod(root)
    storage.mode(result) <- "double"
    result
  })
  initial_sp <- get("initial.sp", envir = asNamespace("mgcv"))(
    X, prepared_setup$penalty_blocks, offsets
  )
  initial_log_sp <- log(as.numeric(initial_sp))
  storage.mode(magic_r) <- "double"
  storage.mode(magic_qr_packed) <- "double"
  fastkpc_full_cuda_phase6_require(
    identical(dim(magic_r), c(p, p)) &&
      identical(dim(magic_qr_packed), c(nrow(X), p)) &&
      length(magic_tau) == p && length(magic_pivot) == p &&
      identical(sort(magic_pivot), seq_len(p)) &&
      sum(ranks) + p <= 128L &&
      length(initial_log_sp) == penalty_count &&
      all(is.finite(initial_log_sp)) &&
      all(vapply(penalty_roots, function(root) {
        is.matrix(root) && nrow(root) == p && all(is.finite(root))
      }, logical(1L))) &&
      all(vapply(penalty_matrices, function(penalty) {
        identical(dim(penalty), c(p, p)) && all(is.finite(penalty))
      }, logical(1L))),
    "Phase 6 CUDA geometry preparation failed"
  )
  list(
    schema_version = "full-cuda-ci-multi-penalty-cuda-setup-v1",
    X = X,
    magic_qr_packed = magic_qr_packed,
    magic_tau = magic_tau,
    magic_r = magic_r,
    magic_pivot = magic_pivot,
    penalty_roots = penalty_roots,
    penalty_matrices = penalty_matrices,
    penalty_ranks = ranks,
    initial_log_sp = initial_log_sp
  )
}

fastkpc_full_cuda_phase6_evaluate_cuda <- function(
    prepared, Y, log_sp,
    rank_tolerance = sqrt(.Machine$double.eps)) {
  fastkpc_full_cuda_phase6_require(
    is.list(prepared) && identical(
      prepared$schema_version,
      "full-cuda-ci-multi-penalty-cuda-setup-v1"
    ),
    "Phase 6 CUDA prepared setup is invalid"
  )
  Y <- as.matrix(Y)
  log_sp <- as.matrix(log_sp)
  storage.mode(Y) <- "double"
  storage.mode(log_sp) <- "double"
  fastkpc_full_cuda_phase6_require(
    nrow(Y) == nrow(prepared$X) && ncol(Y) > 1L && all(is.finite(Y)) &&
      identical(
        dim(log_sp), c(length(prepared$penalty_roots), ncol(Y))
      ) && all(is.finite(log_sp)),
    "Phase 6 CUDA target batch or log-sp matrix is malformed"
  )
  full_cuda_ci_multi_penalty_gcv_evaluate_cuda_native(
    X = prepared$X,
    Y = Y,
    magic_qr_packed = prepared$magic_qr_packed,
    magic_tau = prepared$magic_tau,
    magic_r = prepared$magic_r,
    magic_pivot = prepared$magic_pivot,
    penalty_roots = prepared$penalty_roots,
    penalty_matrices = prepared$penalty_matrices,
    penalty_ranks = prepared$penalty_ranks,
    log_sp = log_sp,
    rank_tolerance = rank_tolerance
  )
}

fastkpc_full_cuda_phase6_optimize_cuda <- function(
    prepared, Y, convergence_tolerance = 1e-7,
    max_step_halving = 25L, max_iterations = 400L,
    max_newton_step = 5, boundary_probe_step = 2,
    max_boundary_probes = 5L,
    rank_tolerance = sqrt(.Machine$double.eps)) {
  fastkpc_full_cuda_phase6_require(
    is.list(prepared) && identical(
      prepared$schema_version,
      "full-cuda-ci-multi-penalty-cuda-setup-v1"
    ),
    "Phase 6 CUDA prepared setup is invalid"
  )
  Y <- as.matrix(Y)
  storage.mode(Y) <- "double"
  fastkpc_full_cuda_phase6_require(
    nrow(Y) == nrow(prepared$X) && ncol(Y) > 1L && all(is.finite(Y)),
    "Phase 6 CUDA optimizer target batch is malformed"
  )
  control <- list(
    convergence_tolerance = as.numeric(convergence_tolerance),
    max_step_halving = as.integer(max_step_halving),
    max_iterations = as.integer(max_iterations),
    max_newton_step = as.numeric(max_newton_step),
    boundary_probe_step = as.numeric(boundary_probe_step),
    max_boundary_probes = as.integer(max_boundary_probes),
    rank_tolerance = as.numeric(rank_tolerance)
  )
  full_cuda_ci_multi_penalty_gcv_optimize_cuda_native(
    X = prepared$X,
    Y = Y,
    magic_qr_packed = prepared$magic_qr_packed,
    magic_tau = prepared$magic_tau,
    magic_r = prepared$magic_r,
    magic_pivot = prepared$magic_pivot,
    penalty_roots = prepared$penalty_roots,
    penalty_matrices = prepared$penalty_matrices,
    penalty_ranks = prepared$penalty_ranks,
    initial_log_sp = prepared$initial_log_sp,
    control = control
  )
}

fastkpc_full_cuda_phase6_prepared_create <- function(
    prepared, target_capacity, device_id = 0L) {
  fastkpc_full_cuda_phase6_require(
    is.list(prepared) && identical(
      prepared$schema_version,
      "full-cuda-ci-multi-penalty-cuda-setup-v1"
    ) && length(target_capacity) == 1L && is.finite(target_capacity) &&
      target_capacity >= 2L,
    "Phase 6 persistent CUDA setup request is malformed"
  )
  full_cuda_ci_multi_penalty_gcv_prepared_create_native(
    prepared, as.integer(target_capacity), as.integer(device_id)
  )
}

fastkpc_full_cuda_phase6_optimize_prepared <- function(
    handle, Y, target_keys,
    convergence_tolerance = 1e-7, max_step_halving = 25L,
    max_iterations = 400L, max_newton_step = 5,
    boundary_probe_step = 2, max_boundary_probes = 5L,
    rank_tolerance = sqrt(.Machine$double.eps)) {
  Y <- as.matrix(Y)
  storage.mode(Y) <- "double"
  fastkpc_full_cuda_phase6_require(
    ncol(Y) > 1L && length(target_keys) == ncol(Y) &&
      all(is.finite(Y)) && all(!is.na(target_keys)) &&
      all(nzchar(as.character(target_keys))),
    "Phase 6 persistent CUDA target batch is malformed"
  )
  control <- list(
    convergence_tolerance = as.numeric(convergence_tolerance),
    max_step_halving = as.integer(max_step_halving),
    max_iterations = as.integer(max_iterations),
    max_newton_step = as.numeric(max_newton_step),
    boundary_probe_step = as.numeric(boundary_probe_step),
    max_boundary_probes = as.integer(max_boundary_probes),
    rank_tolerance = as.numeric(rank_tolerance)
  )
  full_cuda_ci_multi_penalty_gcv_optimize_batch_native(
    handle, Y, as.character(target_keys), control
  )
}

fastkpc_full_cuda_phase6_optimize_prepared_multi <- function(
    handles, target_batches, target_keys, concurrency = 32L,
    convergence_tolerance = 1e-7, max_step_halving = 25L,
    max_iterations = 400L, max_newton_step = 5,
    boundary_probe_step = 2, max_boundary_probes = 5L,
    rank_tolerance = sqrt(.Machine$double.eps)) {
  concurrency <- as.integer(concurrency)
  fastkpc_full_cuda_phase6_require(
    is.list(handles) && length(handles) > 0L &&
      is.list(target_batches) && is.list(target_keys) &&
      length(target_batches) == length(handles) &&
      length(target_keys) == length(handles) &&
      length(concurrency) == 1L && !is.na(concurrency) &&
      concurrency >= 1L && concurrency <= 32L,
    "Phase 6 bounded multi-setup CUDA request is malformed"
  )
  requests <- lapply(seq_along(handles), function(index) {
    Y <- as.matrix(target_batches[[index]])
    storage.mode(Y) <- "double"
    keys <- as.character(target_keys[[index]])
    fastkpc_full_cuda_phase6_require(
      ncol(Y) > 1L && length(keys) == ncol(Y) && all(is.finite(Y)) &&
        all(!is.na(keys)) && all(nzchar(keys)),
      "Phase 6 bounded multi-setup target batch is malformed"
    )
    list(handle = handles[[index]], Y = Y, target_keys = keys)
  })
  control <- list(
    convergence_tolerance = as.numeric(convergence_tolerance),
    max_step_halving = as.integer(max_step_halving),
    max_iterations = as.integer(max_iterations),
    max_newton_step = as.numeric(max_newton_step),
    boundary_probe_step = as.numeric(boundary_probe_step),
    max_boundary_probes = as.integer(max_boundary_probes),
    rank_tolerance = as.numeric(rank_tolerance)
  )
  full_cuda_ci_multi_penalty_gcv_optimize_multi_native(
    requests, concurrency = concurrency, control = control
  )
}
