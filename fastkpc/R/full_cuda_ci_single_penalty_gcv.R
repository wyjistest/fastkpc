fastkpc_full_cuda_phase4_require <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
}

fastkpc_full_cuda_phase4_full_penalty <- function(prepared_setup) {
  blocks <- prepared_setup$penalty_blocks
  offsets <- as.integer(prepared_setup$penalty_offsets)
  p <- ncol(prepared_setup$X)
  fastkpc_full_cuda_phase4_require(
    is.list(blocks) && length(blocks) == 1L && length(offsets) == 1L,
    "Phase 4 supports exactly one penalty block"
  )
  block <- as.matrix(blocks[[1L]])
  offset <- offsets[[1L]]
  fastkpc_full_cuda_phase4_require(
    is.numeric(block) && nrow(block) == ncol(block) &&
      offset >= 1L && offset + nrow(block) - 1L <= p,
    "Phase 4 penalty block geometry is invalid"
  )
  penalty <- matrix(0, p, p)
  indices <- offset:(offset + nrow(block) - 1L)
  penalty[indices, indices] <- block
  penalty
}

fastkpc_full_cuda_phase4_penalty_rank <- function(prepared_setup) {
  rank <- prepared_setup$penalty_ranks
  if (is.null(rank)) rank <- prepared_setup$mgcv_penalty_rank_metadata
  rank <- as.integer(rank)
  fastkpc_full_cuda_phase4_require(
    length(rank) == 1L && !is.na(rank) && rank > 0L &&
      rank <= ncol(prepared_setup$X),
    "Phase 4 requires one authenticated positive penalty rank"
  )
  rank
}

fastkpc_full_cuda_phase4_initial_sp <- function(prepared_setup) {
  X <- as.matrix(prepared_setup$X)
  block <- as.matrix(prepared_setup$penalty_blocks[[1L]])
  offset <- as.integer(prepared_setup$penalty_offsets[[1L]])
  indices <- offset:(offset + nrow(block) - 1L)
  leading_x <- colSums(X * X)
  maximum_penalty <- max(abs(block))
  threshold <- .Machine$double.eps^0.8 * maximum_penalty
  active <- rowMeans(abs(block)) > threshold &
    colMeans(abs(block)) > threshold &
    diag(abs(block)) > threshold
  fastkpc_full_cuda_phase4_require(
    any(active), "Phase 4 penalty has no active initial-sp coordinates"
  )
  value <- mean(leading_x[indices][active]) / mean(diag(block)[active])
  leading_penalty <- numeric(length(leading_x))
  leading_penalty[indices] <- value * diag(block)
  used <- leading_penalty > 0 & leading_x > 0
  fastkpc_full_cuda_phase4_require(
    any(used) && is.finite(value) && value > 0,
    "Phase 4 initial smoothing parameter is invalid"
  )
  while (mean(leading_x[used] /
              (leading_x[used] + leading_penalty[used])) > 0.4) {
    value <- value * 10
    leading_penalty <- leading_penalty * 10
  }
  while (mean(leading_x[used] /
              (leading_x[used] + leading_penalty[used])) < 0.4) {
    value <- value / 10
    leading_penalty <- leading_penalty / 10
  }
  as.numeric(value)
}

fastkpc_full_cuda_phase4_spectral_prepare <- function(prepared_setup) {
  X <- as.matrix(prepared_setup$X)
  p <- ncol(X)
  fastkpc_full_cuda_phase4_require(
    is.numeric(X) && nrow(X) > p && p > 0L && all(is.finite(X)),
    "Phase 4 model matrix must be finite with n > p > 0"
  )
  penalty <- fastkpc_full_cuda_phase4_full_penalty(prepared_setup)
  penalty_rank <- fastkpc_full_cuda_phase4_penalty_rank(prepared_setup)
  gram <- prepared_setup$gram_matrix
  if (is.null(gram)) gram <- crossprod(X)
  gram <- as.matrix(gram)
  fastkpc_full_cuda_phase4_require(
    identical(dim(gram), c(p, p)) && all(is.finite(gram)),
    "Phase 4 Gram matrix is invalid"
  )
  gram_cholesky <- tryCatch(
    chol(gram, pivot = FALSE),
    error = function(error) NULL
  )
  fastkpc_full_cuda_phase4_require(
    !is.null(gram_cholesky),
    "Phase 4 single-penalty spectral setup requires full-rank X"
  )
  inverse_cholesky <- backsolve(gram_cholesky, diag(p))
  whitened_penalty <- crossprod(
    inverse_cholesky,
    penalty %*% inverse_cholesky
  )
  whitened_penalty <- (whitened_penalty + t(whitened_penalty)) / 2
  decomposition <- eigen(whitened_penalty, symmetric = TRUE)
  eigenvalues <- pmax(as.numeric(decomposition$values), 0)
  penalty_nullity <- p - penalty_rank
  if (penalty_nullity > 0L) {
    null_indices <- (p - penalty_nullity + 1L):p
    eigenvalues[null_indices] <- 0
  }
  fastkpc_full_cuda_phase4_require(
    sum(eigenvalues > 0) == penalty_rank,
    "Phase 4 rank-aware spectral decomposition disagrees with penalty rank"
  )
  magic_qr <- qr(X, LAPACK = TRUE)
  magic_q <- qr.Q(magic_qr, complete = FALSE)
  magic_r <- qr.R(magic_qr, complete = FALSE)
  magic_qr_packed <- as.matrix(magic_qr$qr)
  magic_tau <- as.numeric(magic_qr$qraux)
  magic_pivot <- as.integer(magic_qr$pivot)
  mroot <- get("mroot", envir = asNamespace("mgcv"))
  block_root <- mroot(
    as.matrix(prepared_setup$penalty_blocks[[1L]]),
    rank = penalty_rank,
    method = "chol"
  )
  full_root <- matrix(0, p, penalty_rank)
  root_indices <- prepared_setup$penalty_offsets[[1L]] +
    seq_len(nrow(block_root)) - 1L
  full_root[root_indices, ] <- block_root
  magic_penalty_root <- full_root[magic_pivot, , drop = FALSE]
  magic_penalty_matrix <- tcrossprod(magic_penalty_root)
  storage.mode(magic_q) <- "double"
  storage.mode(magic_r) <- "double"
  storage.mode(magic_qr_packed) <- "double"
  storage.mode(magic_penalty_root) <- "double"
  storage.mode(magic_penalty_matrix) <- "double"
  fastkpc_full_cuda_phase4_require(
      identical(dim(magic_q), c(nrow(X), p)) &&
      identical(dim(magic_r), c(p, p)) &&
      identical(dim(magic_qr_packed), c(nrow(X), p)) &&
      length(magic_tau) == p &&
      identical(dim(magic_penalty_root), c(p, penalty_rank)) &&
      identical(dim(magic_penalty_matrix), c(p, p)) &&
      all(is.finite(magic_q)) && all(is.finite(magic_r)) &&
      all(is.finite(magic_qr_packed)) && all(is.finite(magic_tau)) &&
      all(is.finite(magic_penalty_root)) &&
      all(is.finite(magic_penalty_matrix)),
    "Phase 4 mgcv-compatible QR/root preparation is invalid"
  )
  list(
    schema_version = "full-cuda-ci-single-penalty-spectral-v1",
    n = nrow(X),
    p = p,
    penalty_rank = penalty_rank,
    penalty_nullity = penalty_nullity,
    eigenvalues = eigenvalues,
    eigenvectors = decomposition$vectors,
    inverse_cholesky = inverse_cholesky,
    rhs_transform = t(decomposition$vectors) %*% t(inverse_cholesky),
    magic_q = magic_q,
    magic_r = magic_r,
    magic_qr_packed = magic_qr_packed,
    magic_tau = magic_tau,
    magic_penalty_root = magic_penalty_root,
    magic_penalty_matrix = magic_penalty_matrix,
    magic_pivot = magic_pivot,
    initial_sp = fastkpc_full_cuda_phase4_initial_sp(prepared_setup)
  )
}

fastkpc_full_cuda_phase4_hat_values <- function(log_sp, eigenvalues) {
  eigenvalues <- as.numeric(eigenvalues)
  positive <- eigenvalues > 0
  values <- rep(1, length(eigenvalues))
  values[positive] <- stats::plogis(
    -(as.numeric(log_sp) + log(eigenvalues[positive]))
  )
  values
}

fastkpc_full_cuda_phase4_objective <- function(
    log_sp, eigenvalues, squared_projection, y_squared_norm, n,
    denominator_floor = 1e-8) {
  h <- fastkpc_full_cuda_phase4_hat_values(log_sp, eigenvalues)
  squared_projection <- as.numeric(squared_projection)
  fastkpc_full_cuda_phase4_require(
    length(h) == length(squared_projection) &&
      all(is.finite(squared_projection)) && all(squared_projection >= 0),
    "Phase 4 spectral target projection is invalid"
  )
  rss <- as.numeric(y_squared_norm) -
    sum((2 * h - h * h) * squared_projection)
  if (is.finite(rss) && rss < 0) rss <- 0
  edf <- sum(h)
  effective_residual_df <- as.numeric(n) - edf
  valid <- is.finite(rss) && is.finite(edf) &&
    effective_residual_df > denominator_floor
  if (!valid) {
    return(list(
      log_sp = as.numeric(log_sp),
      sp = exp(as.numeric(log_sp)),
      rss = rss,
      edf = edf,
      score = Inf,
      gradient = NA_real_,
      hessian = NA_real_,
      valid = FALSE
    ))
  }
  score <- as.numeric(n) * rss / (effective_residual_df^2)
  one_minus_h <- 1 - h
  h_weight <- h * one_minus_h
  rss_gradient <- 2 * sum(
    h * one_minus_h^2 * squared_projection
  )
  rss_hessian <- 2 * sum(
    h * one_minus_h^2 * (3 * h - 1) * squared_projection
  )
  df_gradient <- sum(h_weight)
  df_hessian <- -sum(h_weight * (1 - 2 * h))
  log_score_gradient <- rss_gradient / rss -
    2 * df_gradient / effective_residual_df
  log_score_hessian <- rss_hessian / rss -
    (rss_gradient / rss)^2 -
    2 * (df_hessian / effective_residual_df -
         (df_gradient / effective_residual_df)^2)
  list(
    log_sp = as.numeric(log_sp),
    sp = exp(as.numeric(log_sp)),
    rss = rss,
    edf = edf,
    score = score,
    gradient = score * log_score_gradient,
    hessian = score *
      (log_score_hessian + log_score_gradient^2),
    valid = is.finite(score)
  )
}

fastkpc_full_cuda_phase4_target_projection <- function(
    prepared_setup, spectral, y) {
  X <- as.matrix(prepared_setup$X)
  y <- as.numeric(y)
  fastkpc_full_cuda_phase4_require(
    length(y) == nrow(X) && all(is.finite(y)),
    "Phase 4 target must be finite and match setup rows"
  )
  projected_rhs <- as.numeric(crossprod(X, y))
  spectral_projection <- as.numeric(
    spectral$rhs_transform %*% projected_rhs
  )
  list(
    projected_rhs = projected_rhs,
    spectral_projection = spectral_projection,
    squared_projection = spectral_projection^2,
    y_squared_norm = sum(y * y)
  )
}

fastkpc_full_cuda_phase4_exact_reference_grid <- function(
    spectral, Y, log_sp_grid) {
  Y <- as.matrix(Y)
  storage.mode(Y) <- "double"
  log_sp_grid <- as.numeric(log_sp_grid)
  q <- ncol(spectral$magic_r)
  penalty_rank <- as.integer(spectral$penalty_rank)
  fastkpc_full_cuda_phase4_require(
    nrow(Y) == spectral$n && ncol(Y) > 0L && all(is.finite(Y)) &&
      length(log_sp_grid) > 0L && all(is.finite(log_sp_grid)) &&
      identical(dim(spectral$magic_r), c(q, q)) &&
      identical(dim(spectral$magic_q), c(spectral$n, q)) &&
      identical(dim(spectral$magic_penalty_matrix), c(q, q)) &&
      penalty_rank > 0L && penalty_rank <= q,
    "Phase 4 exact reference-grid inputs are malformed"
  )

  y0 <- crossprod(spectral$magic_q, Y)
  y_squared_norm <- colSums(Y * Y)
  target_count <- ncol(Y)
  candidate_count <- length(log_sp_grid)
  rss <- matrix(NA_real_, target_count, candidate_count)
  edf <- matrix(NA_real_, target_count, candidate_count)
  score <- matrix(Inf, target_count, candidate_count)
  rank <- integer(candidate_count)

  for (candidate in seq_len(candidate_count)) {
    sp <- exp(log_sp_grid[[candidate]])
    factor <- suppressWarnings(chol(
      sp * spectral$magic_penalty_matrix, pivot = TRUE, tol = -1
    ))
    pivot <- attr(factor, "pivot")
    root_rank <- as.integer(attr(factor, "rank"))
    fastkpc_full_cuda_phase4_require(
      root_rank == penalty_rank && length(pivot) == q,
      "Phase 4 exact reference-grid penalty root rank is invalid"
    )
    root <- t(factor[
      seq_len(penalty_rank), order(pivot), drop = FALSE
    ])
    augmented <- rbind(spectral$magic_r, t(root))
    decomposition <- La.svd(augmented, nu = q, nv = 0L)
    threshold <- decomposition$d[[1L]] * sqrt(.Machine$double.eps)
    numerical_rank <- q
    while (numerical_rank > 0L &&
           decomposition$d[[numerical_rank]] < threshold) {
      numerical_rank <- numerical_rank - 1L
    }
    fastkpc_full_cuda_phase4_require(
      numerical_rank > 0L,
      "Phase 4 exact reference-grid augmented rank is zero"
    )
    rank[[candidate]] <- numerical_rank
    u1 <- decomposition$u[
      seq_len(q), seq_len(numerical_rank), drop = FALSE
    ]
    projected <- crossprod(u1, y0)
    y_ay <- colSums(projected * projected)
    fitted_coordinates <- u1 %*% projected
    y_aay <- colSums(fitted_coordinates * fitted_coordinates)
    candidate_rss <- y_squared_norm - 2 * y_ay + y_aay
    candidate_rss[is.finite(candidate_rss) & candidate_rss < 0] <- 0
    candidate_edf <- sum(u1 * u1)
    residual_df <- spectral$n - candidate_edf
    valid <- all(is.finite(candidate_rss)) && is.finite(candidate_edf) &&
      residual_df > 1e-8
    rss[, candidate] <- candidate_rss
    edf[, candidate] <- candidate_edf
    if (valid) {
      score[, candidate] <-
        spectral$n * candidate_rss / (residual_df * residual_df)
    }
  }
  list(
    rss = rss,
    edf = edf,
    score = score,
    augmented_rank = rank,
    y0 = y0
  )
}

fastkpc_full_cuda_phase4_magic1_optimize <- function(
    spectral,
    squared_projection,
    y_squared_norm,
    convergence_tolerance = 1e-7,
    max_step_halving = 15L,
    max_iterations = 400L,
    max_newton_step = 5,
    boundary_probe_step = 2,
    max_boundary_probes = 5L,
    keep_transcript = FALSE) {
  convergence_tolerance <- as.numeric(convergence_tolerance)
  max_step_halving <- as.integer(max_step_halving)
  max_iterations <- as.integer(max_iterations)
  objective <- function(log_sp) {
    fastkpc_full_cuda_phase4_objective(
      log_sp = log_sp,
      eigenvalues = spectral$eigenvalues,
      squared_projection = squared_projection,
      y_squared_norm = y_squared_norm,
      n = spectral$n
    )
  }
  log_sp <- log(as.numeric(spectral$initial_sp))
  current <- objective(log_sp)
  fastkpc_full_cuda_phase4_require(
    isTRUE(current$valid), "Phase 4 initial objective is invalid"
  )
  minimum_score <- current$score
  score_reduction <- 1e10
  use_steepest_descent <- FALSE
  iteration <- 0L
  score_calls <- 1L
  actual_objective_calls <- 1L
  step_failed <- FALSE
  prior_gradient <- NA_real_
  newton_step <- 0
  steepest_step <- 0
  reported_rms_gradient <- NA_real_
  flat_objective <- FALSE
  transcript <- list()
  transcript_index <- 0L
  append_transcript <- function(row) {
    if (!isTRUE(keep_transcript)) return(invisible(NULL))
    transcript_index <<- transcript_index + 1L
    transcript[[transcript_index]] <<- row
    invisible(NULL)
  }

  repeat {
    iteration <- iteration + 1L
    fastkpc_full_cuda_phase4_require(
      iteration <= max_iterations,
      "Phase 4 magic1 optimizer exceeded its iteration limit"
    )
    try_step <- iteration > 1L
    step_halving_count <- 0L
    proposed_step <- if (use_steepest_descent) {
      steepest_step
    } else {
      newton_step
    }
    accepted_step <- 0
    step_source <- if (use_steepest_descent) {
      "steepest_descent"
    } else {
      "newton"
    }
    iteration_flat <- FALSE
    while (try_step) {
      step_halving_count <- step_halving_count + 1L
      if (step_halving_count == 4L && !use_steepest_descent) {
        use_steepest_descent <- TRUE
        proposed_step <- steepest_step
        step_source <- "steepest_descent_after_newton_rejection"
      }
      trial_log_sp <- log_sp + proposed_step
      trial <- objective(trial_log_sp)
      score_calls <- score_calls + 1L
      actual_objective_calls <- actual_objective_calls + 1L
      accepted <- isTRUE(trial$valid) && trial$score < minimum_score
      append_transcript(data.frame(
        stage = "refinement",
        iteration = iteration,
        evaluation = step_halving_count,
        current_log_sp = log_sp,
        proposed_step = proposed_step,
        trial_log_sp = trial_log_sp,
        objective = trial$score,
        gradient = trial$gradient,
        hessian = trial$hessian,
        accepted = accepted,
        step_source = step_source,
        stringsAsFactors = FALSE
      ))
      if (accepted) {
        try_step <- FALSE
        accepted_step <- proposed_step
        score_reduction <- minimum_score - trial$score
        minimum_score <- trial$score
        log_sp <- trial_log_sp
        current <- trial
      } else {
        predicted_change <- current$gradient * proposed_step +
          0.5 * current$hessian * proposed_step^2
        resolution <- 8 * .Machine$double.eps *
          (1 + abs(minimum_score))
        unresolved_newton_step <- identical(step_source, "newton") &&
          is.finite(predicted_change) && predicted_change <= 0 &&
          abs(predicted_change) <= resolution
        if (unresolved_newton_step) {
          try_step <- FALSE
          iteration_flat <- TRUE
          flat_objective <- TRUE
        } else {
          proposed_step <- proposed_step / 2
        }
      }
      if (step_halving_count == max_step_halving - 1L && try_step) {
        proposed_step <- 0
      }
      if (step_halving_count == max_step_halving) try_step <- FALSE
    }

    converged <- FALSE
    if (iteration > 3L) {
      converged <- TRUE
      if (score_reduction >
          convergence_tolerance * (1 + minimum_score)) {
        converged <- FALSE
      }
      gradient_norm <- abs(prior_gradient)
      if (gradient_norm >
          convergence_tolerance^(1 / 3) * (1 + abs(minimum_score))) {
        converged <- FALSE
      }
      if (step_halving_count == max_step_halving) converged <- TRUE
      if (iteration_flat) converged <- TRUE
      if (converged) {
        reported_rms_gradient <- gradient_norm
        if (step_halving_count == max_step_halving) step_failed <- TRUE
      }
    }

    current <- objective(log_sp)
    prior_gradient <- current$gradient
    use_steepest_descent <- current$hessian < 0
    if (!use_steepest_descent) {
      newton_step <- -current$gradient / current$hessian
      if (abs(newton_step) > max_newton_step) {
        newton_step <- sign(newton_step) * max_newton_step
      }
    }
    steepest_step <- if (current$gradient == 0) {
      0
    } else {
      -current$gradient / abs(current$gradient)
    }
    append_transcript(data.frame(
      stage = "iteration_state",
      iteration = iteration,
      evaluation = 0L,
      current_log_sp = log_sp,
      proposed_step = if (use_steepest_descent) {
        steepest_step
      } else {
        newton_step
      },
      trial_log_sp = log_sp,
      objective = current$score,
      gradient = current$gradient,
      hessian = current$hessian,
      accepted = TRUE,
      step_source = if (use_steepest_descent) {
        "steepest_descent"
      } else {
        "newton"
      },
      stringsAsFactors = FALSE
    ))
    if (converged) break
  }

  hessian_positive_definite <- !use_steepest_descent
  pre_boundary_log_sp <- log_sp
  probe_direction <- if (current$gradient < 0) 1 else -1
  boundary_probe_count <- 0L
  boundary_accepted_count <- 0L
  for (probe in seq_len(max_boundary_probes)) {
    boundary_probe_count <- boundary_probe_count + 1L
    trial_log_sp <- log_sp + probe_direction * boundary_probe_step
    trial <- objective(trial_log_sp)
    actual_objective_calls <- actual_objective_calls + 1L
    accepted <- isTRUE(trial$valid) && trial$score < minimum_score
    append_transcript(data.frame(
      stage = "boundary_probe",
      iteration = iteration,
      evaluation = probe,
      current_log_sp = log_sp,
      proposed_step = probe_direction * boundary_probe_step,
      trial_log_sp = trial_log_sp,
      objective = trial$score,
      gradient = trial$gradient,
      hessian = trial$hessian,
      accepted = accepted,
      step_source = "mgcv_infinity_probe",
      stringsAsFactors = FALSE
    ))
    if (!accepted) break
    log_sp <- trial_log_sp
    current <- trial
    minimum_score <- trial$score
    boundary_accepted_count <- boundary_accepted_count + 1L
  }
  current <- objective(log_sp)
  actual_objective_calls <- actual_objective_calls + 1L
  termination_reason <- if (step_failed) {
    "step_halving_exhausted"
  } else if (flat_objective) {
    "flat_objective"
  } else {
    "score_and_gradient"
  }
  transcript_frame <- if (length(transcript) == 0L) {
    data.frame()
  } else {
    do.call(rbind, transcript)
  }
  list(
    schema_version = "full-cuda-ci-single-penalty-magic1-v1",
    sp = exp(log_sp),
    log_sp = log_sp,
    score = current$score,
    edf = current$edf,
    rss = current$rss,
    gradient = current$gradient,
    hessian = current$hessian,
    iteration_count = iteration,
    score_call_count = score_calls,
    actual_objective_call_count = actual_objective_calls,
    fully_converged = !step_failed && !flat_objective,
    hessian_positive_definite = hessian_positive_definite,
    reported_rms_gradient = reported_rms_gradient,
    termination_reason = termination_reason,
    boundary_status = if (boundary_accepted_count > 0L) {
      "mgcv_infinity_probe_accepted"
    } else {
      "finite_refinement"
    },
    boundary_probe_count = boundary_probe_count,
    boundary_accepted_count = boundary_accepted_count,
    pre_boundary_log_sp = pre_boundary_log_sp,
    transcript = transcript_frame
  )
}

fastkpc_full_cuda_phase4_magic1_batch <- function(
    prepared_setup, Y, target_ids = seq_len(ncol(as.matrix(Y))),
    keep_transcript = FALSE) {
  Y <- as.matrix(Y)
  storage.mode(Y) <- "double"
  target_ids <- as.integer(target_ids)
  fastkpc_full_cuda_phase4_require(
    nrow(Y) == nrow(prepared_setup$X) && ncol(Y) > 0L &&
      length(target_ids) == ncol(Y) && all(is.finite(Y)),
    "Phase 4 target batch geometry is invalid"
  )
  spectral <- fastkpc_full_cuda_phase4_spectral_prepare(prepared_setup)
  projected_rhs <- crossprod(as.matrix(prepared_setup$X), Y)
  spectral_projection <- spectral$rhs_transform %*% projected_rhs
  y_squared_norm <- colSums(Y * Y)
  optimized <- lapply(seq_len(ncol(Y)), function(index) {
    fastkpc_full_cuda_phase4_magic1_optimize(
      spectral = spectral,
      squared_projection = spectral_projection[, index]^2,
      y_squared_norm = y_squared_norm[[index]],
      keep_transcript = keep_transcript
    )
  })
  data <- data.frame(
    target = target_ids,
    sp = vapply(optimized, `[[`, numeric(1L), "sp"),
    log_sp = vapply(optimized, `[[`, numeric(1L), "log_sp"),
    score = vapply(optimized, `[[`, numeric(1L), "score"),
    edf = vapply(optimized, `[[`, numeric(1L), "edf"),
    rss = vapply(optimized, `[[`, numeric(1L), "rss"),
    iteration_count = vapply(
      optimized, `[[`, integer(1L), "iteration_count"
    ),
    score_call_count = vapply(
      optimized, `[[`, integer(1L), "score_call_count"
    ),
    fully_converged = vapply(
      optimized, `[[`, logical(1L), "fully_converged"
    ),
    hessian_positive_definite = vapply(
      optimized, `[[`, logical(1L), "hessian_positive_definite"
    ),
    termination_reason = vapply(
      optimized, `[[`, character(1L), "termination_reason"
    ),
    boundary_status = vapply(
      optimized, `[[`, character(1L), "boundary_status"
    ),
    stringsAsFactors = FALSE
  )
  list(
    schema_version = "full-cuda-ci-single-penalty-magic1-batch-v1",
    spectral = spectral,
    projected_rhs = projected_rhs,
    spectral_projection = spectral_projection,
    target_results = data,
    optimizers = optimized
  )
}

fastkpc_full_cuda_phase4_cuda_native_input <- function(
    prepared_setup, Y, target_ids = seq_len(ncol(as.matrix(Y))),
    sp_grid = numeric(), materialize_grid = length(sp_grid) > 0L,
    keep_transcript = FALSE) {
  X <- as.matrix(prepared_setup$X)
  Y <- as.matrix(Y)
  sp_grid <- as.numeric(sp_grid)
  storage.mode(X) <- "double"
  storage.mode(Y) <- "double"
  target_ids <- as.integer(target_ids)
  fastkpc_full_cuda_phase4_require(
    nrow(Y) == nrow(X) && ncol(Y) > 0L &&
      length(target_ids) == ncol(Y) && all(is.finite(Y)) &&
      all(is.finite(sp_grid)) && all(sp_grid > 0) &&
      (!isTRUE(materialize_grid) || length(sp_grid) > 0L),
    "Phase 4 CUDA target batch geometry is invalid"
  )
  spectral <- fastkpc_full_cuda_phase4_spectral_prepare(prepared_setup)
  list(
    native = list(
      X = X,
      Y = Y,
      rhs_transform = spectral$rhs_transform,
      eigenvalues = as.numeric(spectral$eigenvalues),
      magic_qr_packed = spectral$magic_qr_packed,
      magic_tau = as.numeric(spectral$magic_tau),
      magic_r = spectral$magic_r,
      magic_penalty_root = spectral$magic_penalty_root,
      magic_penalty_matrix = spectral$magic_penalty_matrix,
      target_ids = target_ids,
      penalty_rank = as.integer(spectral$penalty_rank),
      initial_sp = as.numeric(spectral$initial_sp),
      sp_grid = sp_grid,
      materialize_grid = isTRUE(materialize_grid),
      keep_transcript = isTRUE(keep_transcript)
    ),
    spectral = spectral,
    target_ids = target_ids
  )
}

fastkpc_full_cuda_phase4_cuda_batch <- function(
    prepared_setup, Y, target_ids = seq_len(ncol(as.matrix(Y))),
    sp_grid = numeric(), materialize_grid = length(sp_grid) > 0L,
    keep_transcript = FALSE) {
  fastkpc_full_cuda_phase4_require(
    exists("full_cuda_ci_single_penalty_gcv_cuda", mode = "function"),
    paste0(
      "Phase 4 CUDA wrapper is unavailable; source ",
      "fastkpc/R/cuda_native.R"
    )
  )
  request <- fastkpc_full_cuda_phase4_cuda_native_input(
    prepared_setup = prepared_setup,
    Y = Y,
    target_ids = target_ids,
    sp_grid = sp_grid,
    materialize_grid = materialize_grid,
    keep_transcript = keep_transcript
  )
  result <- do.call(
    full_cuda_ci_single_penalty_gcv_cuda,
    request$native
  )
  result$targets$target <- request$target_ids
  result$targets$sp_selection_backend_executed <- "cuda"
  result$targets$gcv_score_backend_executed <- "cuda"
  result$targets$legacy_mgcv_target_calls <- 0L
  result$spectral <- request$spectral
  result
}

fastkpc_full_cuda_phase4_cuda_batches <- function(
    batches, concurrency = 1L, sp_grids = NULL,
    materialize_grid = FALSE, keep_transcript = FALSE) {
  fastkpc_full_cuda_phase4_require(
    exists("full_cuda_ci_single_penalty_gcv_multi_cuda", mode = "function"),
    "Phase 4 multi-setup CUDA wrapper is unavailable"
  )
  fastkpc_full_cuda_phase4_require(
    is.list(batches) && length(batches) > 0L,
    "Phase 4 CUDA batches must be a non-empty list"
  )
  setup_count <- length(batches)
  if (is.null(sp_grids)) {
    sp_grids <- rep(list(numeric()), setup_count)
  } else if (is.numeric(sp_grids)) {
    sp_grids <- rep(list(as.numeric(sp_grids)), setup_count)
  }
  fastkpc_full_cuda_phase4_require(
    is.list(sp_grids) && length(sp_grids) == setup_count,
    "Phase 4 CUDA sp_grids must match the setup count"
  )
  recycle_flag <- function(value, label) {
    value <- as.logical(value)
    if (length(value) == 1L) value <- rep(value, setup_count)
    fastkpc_full_cuda_phase4_require(
      length(value) == setup_count && !anyNA(value),
      paste0("Phase 4 ", label, " flags must match the setup count")
    )
    value
  }
  materialize_grid <- recycle_flag(materialize_grid, "materialize_grid")
  keep_transcript <- recycle_flag(keep_transcript, "keep_transcript")
  requests <- lapply(seq_along(batches), function(index) {
    batch <- batches[[index]]
    fastkpc_full_cuda_phase4_require(
      is.list(batch) && !is.null(batch$setup) && !is.null(batch$Y),
      "Phase 4 CUDA batch is missing setup or Y"
    )
    target_ids <- batch$target_ids
    if (is.null(target_ids)) target_ids <- seq_len(ncol(as.matrix(batch$Y)))
    fastkpc_full_cuda_phase4_cuda_native_input(
      prepared_setup = batch$setup,
      Y = batch$Y,
      target_ids = target_ids,
      sp_grid = sp_grids[[index]],
      materialize_grid = materialize_grid[[index]],
      keep_transcript = keep_transcript[[index]]
    )
  })
  native_inputs <- lapply(requests, `[[`, "native")
  names(native_inputs) <- names(batches)
  result <- full_cuda_ci_single_penalty_gcv_multi_cuda(
    native_inputs, concurrency = concurrency
  )
  result$setups <- lapply(seq_along(result$setups), function(index) {
    setup_result <- result$setups[[index]]
    setup_result$targets$target <- requests[[index]]$target_ids
    setup_result$targets$sp_selection_backend_executed <- "cuda"
    setup_result$targets$gcv_score_backend_executed <- "cuda"
    setup_result$targets$legacy_mgcv_target_calls <- 0L
    setup_result$spectral <- requests[[index]]$spectral
    setup_result
  })
  names(result$setups) <- names(batches)
  result
}

fastkpc_full_cuda_phase4_select_and_solve_cuda <- function(
    prepared_handle, prepared_setup, Y, planned_route, target_keys,
    target_ids = seq_len(ncol(as.matrix(Y))),
    outputs = c("residuals")) {
  fastkpc_full_cuda_phase4_require(
    exists(
      "full_cuda_ci_single_penalty_gcv_fixed_sp_cuda", mode = "function"
    ),
    "Phase 4 integrated CUDA selection/solve wrapper is unavailable"
  )
  Y <- as.matrix(Y)
  storage.mode(Y) <- "double"
  target_ids <- as.integer(target_ids)
  planned_route <- as.character(planned_route)
  target_keys <- as.character(target_keys)
  target_count <- ncol(Y)
  fastkpc_full_cuda_phase4_require(
    nrow(Y) == nrow(prepared_setup$X) && target_count > 0L &&
      length(target_ids) == target_count &&
      length(planned_route) == target_count &&
      length(target_keys) == target_count && all(is.finite(Y)),
    "Phase 4 integrated CUDA target batch is malformed"
  )
  spectral <- fastkpc_full_cuda_phase4_spectral_prepare(prepared_setup)
  result <- full_cuda_ci_single_penalty_gcv_fixed_sp_cuda(
    handle = prepared_handle,
    X = prepared_setup$X,
    Y = Y,
    rhs_transform = spectral$rhs_transform,
    eigenvalues = spectral$eigenvalues,
    magic_qr_packed = spectral$magic_qr_packed,
    magic_tau = spectral$magic_tau,
    magic_r = spectral$magic_r,
    magic_penalty_root = spectral$magic_penalty_root,
    magic_penalty_matrix = spectral$magic_penalty_matrix,
    target_ids = target_ids,
    penalty_rank = spectral$penalty_rank,
    initial_sp = spectral$initial_sp,
    planned_route = planned_route,
    target_keys = target_keys,
    outputs = outputs
  )
  result$targets$target <- target_ids
  result
}
