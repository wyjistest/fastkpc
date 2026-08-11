source("fastkpc/R/legacy_runner.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/native.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cuda.R")

args <- commandArgs(trailingOnly = TRUE)
source_path <- if (length(args) >= 1L) args[[1L]] else {
  "/tmp/fastkpc_causal_large_parity_20260810/residual_route_bakeoff.rds"
}
replay_path <- if (length(args) >= 2L) args[[2L]] else {
  "/tmp/fastkpc_causal_large_parity_20260810/source10_target2_objective_replay.rds"
}
output_path <- if (length(args) >= 3L) args[[3L]] else {
  "/tmp/fastkpc_causal_large_parity_20260810/source10_target2_derivatives.rds"
}
source_index <- if (length(args) >= 4L) as.integer(args[[4L]]) else 10L
target_index <- if (length(args) >= 5L) as.integer(args[[5L]]) else 2L

if (!file.exists(replay_path)) {
  stop("selected-SP derivative replay input is missing", call. = FALSE)
}
if (file.exists(output_path)) stop("output already exists", call. = FALSE)
if (is.na(source_index) || source_index <= 0L ||
    is.na(target_index) || target_index <= 0L) {
  stop("selected-SP derivative diagnostic index is invalid", call. = FALSE)
}

data_paths <- c(
  `20230331145508` =
    "causalDiscovery/20230331145508/combined_data_shuffle-causalDiscoveryInput.RData",
  `20240102191119` =
    "causalDiscovery/20240102191119/GSE99254-482col-causalDiscoveryInput.RData",
  `20241215091815` =
    "causalDiscovery/20241215091815/logged-lung-exhausted-allData-causalDiscoveryInput.RData"
)

prepared_setup_from_native <- function(native_setup) {
  penalty_count <- length(native_setup$penalty_blocks)
  list(
    formula_class = "additive-smooth",
    X = unname(native_setup$X),
    gram_matrix = unname(native_setup$gram_matrix),
    penalty_blocks = setNames(
      lapply(native_setup$penalty_blocks, unname),
      paste0("penalty_", seq_len(penalty_count))
    ),
    penalty_offsets = as.integer(native_setup$penalty_offsets),
    penalty_ranks = as.integer(native_setup$penalty_ranks),
    mgcv_penalty_rank_metadata = as.integer(native_setup$penalty_ranks),
    constraint = matrix(
      numeric(), nrow = 0L, ncol = ncol(native_setup$X)
    ),
    H = NULL,
    weights = NULL
  )
}

aggregate_root <- function(penalty_matrices, log_sp) {
  q <- nrow(penalty_matrices[[1L]])
  aggregate <- matrix(0, q, q)
  for (index in seq_along(penalty_matrices)) {
    aggregate[] <- aggregate[] +
      exp(log_sp[[index]]) * penalty_matrices[[index]][]
  }
  factor <- suppressWarnings(chol(aggregate, pivot = TRUE, tol = -1))
  rank <- as.integer(attr(factor, "rank"))
  pivot <- as.integer(attr(factor, "pivot"))
  root <- matrix(0, rank, q)
  root[, pivot] <- factor[seq_len(rank), , drop = FALSE]
  list(aggregate = aggregate, factor = factor, root = root,
       rank = rank, pivot = pivot)
}

align_gpu_svd <- function(cpu, gpu) {
  q <- length(cpu$d)
  u <- gpu$left_vectors
  vt <- gpu$right_vectors_transposed
  for (component in seq_len(q)) {
    correlation <- 0
    for (row in seq_len(nrow(u))) {
      correlation <- correlation + cpu$u[row, component] * u[row, component]
    }
    if (is.finite(correlation) && correlation < 0) {
      u[, component] <- -u[, component]
      vt[component, ] <- -vt[component, ]
    }
  }
  list(u = u, vt = vt, d = as.double(gpu$singular_values))
}

dot_serial <- function(left, right) {
  value <- 0
  for (index in seq_along(left)) value <- value + left[[index]] * right[[index]]
  value
}

apply_qt_serial <- function(qr_packed, tau, value) {
  output <- as.double(value)
  for (reflector in seq_along(tau)) {
    dot <- output[[reflector]]
    if (reflector < length(output)) {
      for (row in seq.int(reflector + 1L, length(output))) {
        dot <- dot + qr_packed[row, reflector] * output[[row]]
      }
    }
    scale <- -tau[[reflector]] * dot
    output[[reflector]] <- output[[reflector]] + scale
    if (reflector < length(output)) {
      for (row in seq.int(reflector + 1L, length(output))) {
        output[[row]] <-
          output[[row]] + qr_packed[row, reflector] * scale
      }
    }
  }
  output[seq_along(tau)]
}

multiply_serial <- function(left, right) {
  stopifnot(ncol(left) == nrow(right))
  output <- matrix(0, nrow(left), ncol(right))
  for (column in seq_len(ncol(right))) {
    for (row in seq_len(nrow(left))) {
      value <- 0
      for (inner in seq_len(ncol(left))) {
        value <- value + left[row, inner] * right[inner, column]
      }
      output[row, column] <- value
    }
  }
  output
}

xtx_serial <- function(input) multiply_serial(t(input), input)
xxt_serial <- function(input) multiply_serial(input, t(input))

jacobi_step <- function(hessian, gradient, max_newton_step = 5) {
  dimension <- nrow(hessian)
  matrix <- hessian
  vectors <- diag(dimension)
  converged <- FALSE
  for (sweep in seq_len(128L)) {
    maximum_off_diagonal <- 0
    maximum_diagonal <- 0
    for (column in seq_len(dimension)) {
      maximum_diagonal <- max(
        maximum_diagonal, abs(matrix[column, column])
      )
      if (column > 1L) {
        for (row in seq_len(column - 1L)) {
          maximum_off_diagonal <- max(
            maximum_off_diagonal, abs(matrix[row, column])
          )
        }
      }
    }
    if (maximum_off_diagonal <=
        8 * 1.1102230246251565e-16 * max(1, maximum_diagonal)) {
      converged <- TRUE
      break
    }
    if (dimension > 1L) {
      for (p in seq_len(dimension - 1L)) {
        for (q_index in seq.int(p + 1L, dimension)) {
          apq <- matrix[p, q_index]
          if (apq == 0) next
          app <- matrix[p, p]
          aqq <- matrix[q_index, q_index]
          tau <- (aqq - app) / (2 * apq)
          root <- sqrt(1 + tau * tau)
          magnitude <- 1 / (abs(tau) + root)
          t <- if (tau < 0) -magnitude else magnitude
          cosine <- 1 / sqrt(1 + t * t)
          sine <- t * cosine
          for (index in seq_len(dimension)) {
            if (index == p || index == q_index) next
            aip <- matrix[index, p]
            aiq <- matrix[index, q_index]
            new_aip <- cosine * aip - sine * aiq
            new_aiq <- sine * aip + cosine * aiq
            matrix[index, p] <- matrix[p, index] <- new_aip
            matrix[index, q_index] <- matrix[q_index, index] <- new_aiq
          }
          matrix[p, p] <- app - t * apq
          matrix[q_index, q_index] <- aqq + t * apq
          matrix[p, q_index] <- matrix[q_index, p] <- 0
          for (row in seq_len(dimension)) {
            vip <- vectors[row, p]
            viq <- vectors[row, q_index]
            vectors[row, p] <- cosine * vip - sine * viq
            vectors[row, q_index] <- sine * vip + cosine * viq
          }
        }
      }
    }
  }
  values <- diag(matrix)
  order <- order(values)
  values <- values[order]
  vectors <- vectors[, order, drop = FALSE]
  use_steepest <- any(values < 0)
  if (use_steepest) {
    maximum_gradient <- max(abs(gradient))
    step <- if (maximum_gradient == 0) {
      numeric(dimension)
    } else {
      -gradient / maximum_gradient
    }
  } else {
    projected <- numeric(dimension)
    for (column in seq_len(dimension)) {
      projected[[column]] <-
        dot_serial(vectors[, column], gradient) / values[[column]]
    }
    step <- numeric(dimension)
    for (row in seq_len(dimension)) {
      step[[row]] <- -dot_serial(vectors[row, ], projected)
    }
    maximum_component <- max(abs(step))
    if (maximum_component > max_newton_step) {
      step <- step * (max_newton_step / maximum_component)
    }
  }
  list(
    converged = converged, values = values, vectors = vectors,
    use_steepest = use_steepest, step = step
  )
}

evaluate_derivatives <- function(
    decomposition, y0, squared_norm, penalty_roots, log_sp, n,
    serial_products) {
  q <- length(decomposition$d)
  rank <- sum(decomposition$d >=
    decomposition$d[[1L]] * sqrt(.Machine$double.eps))
  U1 <- decomposition$u[seq_len(q), seq_len(rank), drop = FALSE]
  V <- t(decomposition$vt)[, seq_len(rank), drop = FALSE]
  d <- decomposition$d[seq_len(rank)]
  y1 <- numeric(rank)
  for (column in seq_len(rank)) y1[[column]] <- dot_serial(U1[, column], y0)
  y_ay <- dot_serial(y1, y1)
  fitted_coordinates <- numeric(q)
  for (row in seq_len(q)) {
    fitted_coordinates[[row]] <- dot_serial(U1[row, ], y1)
  }
  y_aay <- dot_serial(fitted_coordinates, fitted_coordinates)
  rss <- squared_norm - 2 * y_ay + y_aay
  if (rss < 0) rss <- 0
  edf <- 0
  for (element in seq_along(U1)) edf <- edf + U1[[element]] * U1[[element]]
  delta <- n - edf
  score <- n * rss / (delta * delta)

  multiply <- if (serial_products) multiply_serial else function(a, b) a %*% b
  symmetric_xtx <- if (serial_products) xtx_serial else crossprod
  symmetric_xxt <- if (serial_products) xxt_serial else tcrossprod
  U1U1 <- symmetric_xtx(U1)
  penalty_count <- length(penalty_roots)
  M <- K <- My <- Ky <- yK <- vector("list", penalty_count)
  for (penalty in seq_len(penalty_count)) {
    transformed <- multiply(t(V), penalty_roots[[penalty]])
    for (row in seq_len(rank)) transformed[row, ] <- transformed[row, ] / d[[row]]
    intermediate <- multiply(t(transformed), U1U1)
    K[[penalty]] <- multiply(transformed, intermediate)
    M[[penalty]] <- symmetric_xxt(transformed)
    My[[penalty]] <- Ky[[penalty]] <- yK[[penalty]] <- numeric(rank)
    for (output in seq_len(rank)) {
      My[[penalty]][[output]] <- dot_serial(y1, M[[penalty]][, output])
      yK[[penalty]][[output]] <- dot_serial(y1, K[[penalty]][, output])
      Ky[[penalty]][[output]] <- dot_serial(y1, K[[penalty]][output, ])
    }
  }

  lambda <- exp(log_sp)
  dnorm <- ddelta <- numeric(penalty_count)
  d2norm <- d2delta <- matrix(0, penalty_count, penalty_count)
  for (i in seq_len(penalty_count)) {
    trace <- 0
    for (component in seq_len(rank)) trace <- trace + K[[i]][component, component]
    ddelta[[i]] <- lambda[[i]] * trace
    for (j in seq_len(i)) {
      product_sum <- 0
      for (element in seq_along(M[[j]])) {
        product_sum <- product_sum + M[[j]][[element]] * K[[i]][[element]]
      }
      value <- -2 * exp(log_sp[[i]] + log_sp[[j]]) * product_sum
      d2delta[i, j] <- d2delta[j, i] <- value
    }
    d2delta[i, i] <- d2delta[i, i] + ddelta[[i]]
    norm_derivative <- 0
    for (component in seq_len(rank)) {
      norm_derivative <- norm_derivative + y1[[component]] *
        (My[[i]][[component]] - Ky[[i]][[component]])
    }
    dnorm[[i]] <- 2 * lambda[[i]] * norm_derivative
    for (j in seq_len(i)) {
      norm_value <- 0
      for (component in seq_len(rank)) {
        norm_value <- norm_value +
          My[[i]][[component]] * Ky[[j]][[component]] +
          My[[j]][[component]] * Ky[[i]][[component]] -
          2 * My[[i]][[component]] * My[[j]][[component]] +
          yK[[i]][[component]] * My[[j]][[component]]
      }
      value <- norm_value * 2 * exp(log_sp[[i]] + log_sp[[j]])
      d2norm[i, j] <- d2norm[j, i] <- value
    }
    d2norm[i, i] <- d2norm[i, i] + dnorm[[i]]
  }
  score_scale <- n / (delta * delta)
  delta_scale <- 2 * score_scale * rss / delta
  cross_scale <- -2 * score_scale / delta
  curvature_scale <- 3 * delta_scale / delta
  gradient <- numeric(penalty_count)
  hessian <- matrix(0, penalty_count, penalty_count)
  for (i in seq_len(penalty_count)) {
    gradient[[i]] <- score_scale * dnorm[[i]] - delta_scale * ddelta[[i]]
    for (j in seq_len(i)) {
      value <-
        cross_scale * (ddelta[[j]] * dnorm[[i]] + ddelta[[i]] * dnorm[[j]]) +
        score_scale * d2norm[i, j] +
        curvature_scale * ddelta[[i]] * ddelta[[j]] -
        delta_scale * d2delta[i, j]
      hessian[i, j] <- hessian[j, i] <- value
    }
  }
  list(
    rss = rss, edf = edf, score = score, U1 = U1, V = V, d = d,
    y1 = y1, fitted_coordinates = fitted_coordinates,
    U1U1 = U1U1, M = M, K = K,
    My = My, Ky = Ky, yK = yK, dnorm = dnorm, ddelta = ddelta,
    d2norm = d2norm, d2delta = d2delta, gradient = gradient,
    hessian = hessian
  )
}

replay <- readRDS(replay_path)
if (!is.null(replay$formal_context)) {
  setup <- replay$formal_context$setup
  Y <- replay$formal_context$Y
  target_index <- as.integer(replay$formal_context$target_index)
  prepared <- fastkpc_full_cuda_phase6_prepare(setup)
} else {
  if (!file.exists(source_path)) {
    stop("selected-SP derivative source input is missing", call. = FALSE)
  }
  source_receipt <- readRDS(source_path)
  if (source_index > length(source_receipt$details)) {
    stop("source index exceeds receipt", call. = FALSE)
  }
  detail <- source_receipt$details[[source_index]]
  if (target_index > length(detail$targets)) {
    stop("target index exceeds source target count", call. = FALSE)
  }
  dataset <- as.character(detail$request$dataset)
  payload <- new.env(parent = emptyenv())
  object_names <- load(data_paths[[dataset]], envir = payload)
  if (length(object_names) != 1L) {
    stop("data payload is malformed", call. = FALSE)
  }
  data <- as.matrix(payload[[object_names[[1L]]]])
  storage.mode(data) <- "double"
  Y <- unname(data[, detail$targets, drop = FALSE])
  native_setup <- full_cuda_ci_native_setup_native(
    data[, detail$conditioning_set, drop = FALSE]
  )
  prepared <- fastkpc_full_cuda_phase6_prepare(
    prepared_setup_from_native(native_setup)
  )
}
q <- ncol(prepared$X)
qr_reference <- qr(prepared$X, LAPACK = TRUE)
y0 <- qr.qty(qr_reference, Y[, target_index])[seq_len(q)]
serial_y0 <- apply_qt_serial(
  prepared$magic_qr_packed, prepared$magic_tau, Y[, target_index]
)
squared_norm <- dot_serial(Y[, target_index], Y[, target_index])
load_fastkpc_cuda_native()

state_entries <- which(vapply(
  replay$cpp_transcript,
  function(entry) identical(entry$stage, "iteration_state"),
  logical(1L)
))
state_log_sp <- do.call(cbind, lapply(
  replay$cpp_transcript[state_entries],
  function(entry) as.double(entry$current_log_sp)
))
state_Y <- matrix(
  rep(Y[, target_index], length(state_entries)),
  nrow = nrow(Y), ncol = length(state_entries)
)
forced_stable <- fastkpc_full_cuda_phase10_grouped_evaluate_prototype(
  prepared, state_Y, state_log_sp, force_stable_svd = TRUE,
  grouped_warps_per_block = 2L, timing_repetitions = 1L
)
if (!isTRUE(forced_stable$prototype_diagnostics$exact_parity)) {
  stop("forced stable grouped evaluator lost baseline parity", call. = FALSE)
}

rows <- list()
details <- list()
for (entry_index in seq_along(replay$cpp_transcript)) {
  entry <- replay$cpp_transcript[[entry_index]]
  if (!identical(entry$stage, "iteration_state")) next
  log_sp <- as.double(entry$current_log_sp)
  root <- aggregate_root(prepared$penalty_matrices, log_sp)
  augmented <- rbind(prepared$magic_r, root$root)
  gpu_augmented <-
    full_cuda_ci_multi_penalty_gcv_augmented_transcript_for_test(
      prepared$magic_r, prepared$penalty_matrices, log_sp
    )
  cpu_svd <- La.svd(augmented, nu = q, nv = q)
  gpu_raw <- full_cuda_ci_multi_penalty_gcv_svd_transcript_for_test(
    augmented, force_stable_svd = TRUE
  )
  gpu_svd <- align_gpu_svd(cpu_svd, gpu_raw)
  cpu_decomposition <- list(u = cpu_svd$u, vt = cpu_svd$vt, d = cpu_svd$d)
  gpu_decomposition <- list(u = gpu_svd$u, vt = gpu_svd$vt, d = gpu_svd$d)
  cpu_blas <- evaluate_derivatives(
    cpu_decomposition, y0, squared_norm, prepared$penalty_roots,
    log_sp, nrow(Y), FALSE
  )
  cpu_serial <- evaluate_derivatives(
    cpu_decomposition, y0, squared_norm, prepared$penalty_roots,
    log_sp, nrow(Y), TRUE
  )
  gpu_blas <- evaluate_derivatives(
    gpu_decomposition, y0, squared_norm, prepared$penalty_roots,
    log_sp, nrow(Y), FALSE
  )
  gpu_serial <- evaluate_derivatives(
    gpu_decomposition, y0, squared_norm, prepared$penalty_roots,
    log_sp, nrow(Y), TRUE
  )
  objective_transcript <-
    full_cuda_ci_multi_penalty_gcv_objective_transcript_for_test(
      prepared, y0, squared_norm, log_sp, force_stable_svd = TRUE
    )
  expected_metric <- simplify2array(gpu_serial$M)
  expected_influence <- simplify2array(gpu_serial$K)
  expected_metric_y <- do.call(cbind, gpu_serial$My)
  expected_influence_y <- do.call(cbind, gpu_serial$Ky)
  expected_y_influence <- do.call(cbind, gpu_serial$yK)
  replay_gradient <- replay$gpu$gradient[, entry_index]
  replay_hessian <- replay$gpu$hessian[, , entry_index]
  state_position <- match(entry_index, state_entries)
  stable_gradient <- forced_stable$evaluation$gradient[, state_position]
  stable_hessian <- forced_stable$evaluation$hessian[, , state_position]
  jacobi <- jacobi_step(entry$hessian, entry$gradient)
  row <- data.frame(
    entry = entry_index,
    iteration = as.integer(entry$iteration),
    q = q,
    augmented_rows = nrow(augmented),
    aggregate_rank = root$rank,
    aggregate_error = max(abs(
      root$aggregate - gpu_augmented$aggregate_penalty
    )),
    augmented_error = max(abs(augmented - gpu_augmented$augmented)),
    singular_error = max(abs(cpu_svd$d - gpu_svd$d)),
    u_error = max(abs(cpu_svd$u - gpu_svd$u)),
    vt_error = max(abs(cpu_svd$vt - gpu_svd$vt)),
    cpp_blas_gradient_error = max(abs(cpu_blas$gradient - entry$gradient)),
    cpp_blas_hessian_error = max(abs(cpu_blas$hessian - entry$hessian)),
    cpu_serial_gradient_error = max(abs(cpu_serial$gradient - entry$gradient)),
    cpu_serial_hessian_error = max(abs(cpu_serial$hessian - entry$hessian)),
    gpu_blas_gradient_error = max(abs(gpu_blas$gradient - entry$gradient)),
    gpu_blas_hessian_error = max(abs(gpu_blas$hessian - entry$hessian)),
    gpu_serial_gradient_error = max(abs(gpu_serial$gradient - entry$gradient)),
    gpu_serial_hessian_error = max(abs(gpu_serial$hessian - entry$hessian)),
    replay_gradient_error = max(abs(replay_gradient - entry$gradient)),
    replay_hessian_error = max(abs(replay_hessian - entry$hessian)),
    stable_rss_error = abs(
      forced_stable$evaluation$rss[[state_position]] -
        replay$gpu$rss[[entry_index]]
    ),
    stable_edf_error = abs(
      forced_stable$evaluation$edf[[state_position]] -
        replay$gpu$edf[[entry_index]]
    ),
    stable_score_error = abs(
      forced_stable$evaluation$score[[state_position]] - entry$objective
    ),
    stable_gradient_error = max(abs(stable_gradient - entry$gradient)),
    stable_hessian_error = max(abs(stable_hessian - entry$hessian)),
    objective_singular_error = max(abs(
      objective_transcript$singular_values - gpu_svd$d
    )),
    objective_u_error = max(abs(
      objective_transcript$left_vectors - gpu_svd$u
    )),
    objective_vt_error = max(abs(
      objective_transcript$right_vectors_transposed - gpu_svd$vt
    )),
    objective_y1_error = max(abs(
      objective_transcript$projected_response - gpu_serial$y1
    )),
    objective_scaled_y1_error = max(abs(
      objective_transcript$scaled_projection -
        gpu_serial$y1 / gpu_serial$d
    )),
    objective_rss_error = abs(
      objective_transcript$rss - gpu_serial$rss
    ),
    objective_edf_error = abs(
      objective_transcript$edf - gpu_serial$edf
    ),
    objective_score_error = abs(
      objective_transcript$score - gpu_serial$score
    ),
    objective_u1u1_error = max(abs(
      objective_transcript$u1_u1 - gpu_serial$U1U1
    )),
    objective_metric_error = max(abs(
      objective_transcript$metric - expected_metric
    )),
    objective_influence_error = max(abs(
      objective_transcript$influence - expected_influence
    )),
    objective_metric_y_error = max(abs(
      objective_transcript$metric_y - expected_metric_y
    )),
    objective_influence_y_error = max(abs(
      objective_transcript$influence_y - expected_influence_y
    )),
    objective_y_influence_error = max(abs(
      objective_transcript$y_influence - expected_y_influence
    )),
    objective_dnorm_error = max(abs(
      objective_transcript$derivative_norm - gpu_serial$dnorm
    )),
    objective_ddelta_error = max(abs(
      objective_transcript$derivative_delta - gpu_serial$ddelta
    )),
    objective_d2norm_error = max(abs(
      objective_transcript$second_derivative_norm - gpu_serial$d2norm
    )),
    objective_d2delta_error = max(abs(
      objective_transcript$second_derivative_delta - gpu_serial$d2delta
    )),
    objective_gradient_error = max(abs(
      objective_transcript$gradient - gpu_serial$gradient
    )),
    objective_hessian_error = max(abs(
      objective_transcript$hessian - gpu_serial$hessian
    )),
    jacobi_eigenvalue_error = max(abs(
      jacobi$values - entry$hessian_eigenvalues
    )),
    jacobi_step_error = max(abs(jacobi$step - entry$proposed_step)),
    jacobi_route_matches = identical(
      jacobi$use_steepest,
      identical(as.character(entry$step_source), "steepest_descent")
    ),
    serial_vs_replay_gradient_error = max(abs(
      gpu_serial$gradient - replay_gradient
    )),
    serial_vs_replay_hessian_error = max(abs(
      gpu_serial$hessian - replay_hessian
    )),
    stringsAsFactors = FALSE
  )
  rows[[length(rows) + 1L]] <- row
  details[[as.character(entry_index)]] <- list(
    entry = entry, root = root, gpu_augmented = gpu_augmented,
    cpu_svd = cpu_svd, gpu_svd = gpu_svd, cpu_blas = cpu_blas,
    cpu_serial = cpu_serial, gpu_blas = gpu_blas,
    gpu_serial = gpu_serial, replay_gradient = replay_gradient,
    replay_hessian = replay_hessian, stable_gradient = stable_gradient,
    stable_hessian = stable_hessian,
    objective_transcript = objective_transcript, jacobi = jacobi
  )
  cat(sprintf(
    "iter=%d aug=% .3e d=% .3e U=% .3e Vt=% .3e post=(y1 % .3e,rss % .3e,u1u1 % .3e,M % .3e,K % .3e,My % .3e,d1 % .3e,d2 % .3e,g % .3e,h % .3e) cpp-blas=(% .3e,% .3e) replay=(% .3e,% .3e) stable=(% .3e,% .3e) jacobi=(eig % .3e,step % .3e)\n",
    row$iteration, row$augmented_error, row$singular_error,
    row$u_error, row$vt_error, row$objective_y1_error,
    row$objective_rss_error, row$objective_u1u1_error,
    row$objective_metric_error, row$objective_influence_error,
    row$objective_metric_y_error, row$objective_dnorm_error,
    row$objective_d2norm_error, row$objective_gradient_error,
    row$objective_hessian_error, row$cpp_blas_gradient_error,
    row$cpp_blas_hessian_error, row$replay_gradient_error,
    row$replay_hessian_error, row$stable_gradient_error,
    row$stable_hessian_error, row$jacobi_eigenvalue_error,
    row$jacobi_step_error
  ))
}

rows <- do.call(rbind, rows)
receipt <- list(
  schema_version = "mgcv-gpu-selected-sp-derivative-attribution-v1",
  native_sha256 = unname(digest::digest(
    file = "fastkpc/build/fastkpc_cuda.so", algo = "sha256",
    serialize = FALSE
  )),
  source_path = if (file.exists(source_path)) normalizePath(source_path) else NA,
  replay_path = normalizePath(replay_path),
  source_index = source_index,
  target_index = target_index,
  y0_error = max(abs(serial_y0 - y0)),
  y0_lapack = y0,
  y0_serial = serial_y0,
  squared_norm = squared_norm,
  forced_stable_evaluation = forced_stable$evaluation,
  forced_stable_prototype_diagnostics =
    forced_stable$prototype_diagnostics,
  rows = rows,
  details = details
)
saveRDS(receipt, output_path, compress = "xz", version = 3L)
print(rows, digits = 17, row.names = FALSE)
