source("fastkpc/R/legacy_runner.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/native.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cpp.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cuda.R")

args <- commandArgs(trailingOnly = TRUE)
input_path <- if (length(args) >= 1L) args[[1L]] else {
  "/tmp/fastkpc_causal_large_parity_20260810/residual_route_bakeoff.rds"
}
output_path <- if (length(args) >= 2L) args[[2L]] else {
  "/tmp/fastkpc_causal_large_parity_20260810/svd_transcript_diagnosis.rds"
}
requested_indices <- if (length(args) >= 3L) {
  as.integer(strsplit(args[[3L]], ",", fixed = TRUE)[[1L]])
} else {
  # q=55, q=64, q=73, and the rank-deficient q=37 witness.
  c(1L, 4L, 7L, 11L)
}

if (!file.exists(input_path)) stop("source receipt is missing", call. = FALSE)
if (file.exists(output_path)) stop("output already exists", call. = FALSE)
if (anyNA(requested_indices) || any(requested_indices <= 0L)) {
  stop("witness indices must be positive integers", call. = FALSE)
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
    formula_class = if (penalty_count == 1L) {
      "full-smooth"
    } else {
      "additive-smooth"
    },
    X = unname(native_setup$X),
    gram_matrix = unname(native_setup$gram_matrix),
    penalty_blocks = setNames(
      lapply(native_setup$penalty_blocks, unname),
      paste0("penalty_", seq_len(penalty_count))
    ),
    penalty_offsets = as.integer(native_setup$penalty_offsets),
    penalty_ranks = as.integer(native_setup$penalty_ranks),
    mgcv_penalty_rank_metadata = as.integer(native_setup$penalty_ranks),
    constraint = matrix(numeric(), nrow = 0L, ncol = ncol(native_setup$X)),
    H = NULL,
    weights = NULL
  )
}

cuda_geometry_from_setup <- function(setup) {
  geometry <- full_cuda_ci_native_geometry_prepare_native(
    setup$X, setup$penalty_blocks, setup$penalty_offsets,
    setup$mgcv_penalty_rank_metadata
  )
  list(
    X = setup$X,
    magic_qr_packed = geometry$magic_qr_packed,
    magic_tau = geometry$magic_tau,
    magic_r = geometry$magic_r,
    magic_pivot = geometry$magic_pivot,
    penalty_matrices = geometry$penalty_matrices
  )
}

aggregate_root <- function(penalty_matrices, sp) {
  q <- nrow(penalty_matrices[[1L]])
  aggregate <- matrix(0, q, q)
  for (index in seq_along(penalty_matrices)) {
    aggregate[] <- aggregate[] +
      as.double(sp[[index]]) * penalty_matrices[[index]][]
  }
  factor <- suppressWarnings(chol(aggregate, pivot = TRUE, tol = -1))
  rank <- as.integer(attr(factor, "rank"))
  pivot <- as.integer(attr(factor, "pivot"))
  root <- matrix(0, rank, q)
  root[, pivot] <- factor[seq_len(rank), , drop = FALSE]
  list(root = root, rank = rank, pivot = pivot)
}

align_gpu_svd <- function(cpu, gpu) {
  q <- length(cpu$d)
  aligned_u <- gpu$left_vectors
  aligned_vt <- gpu$right_vectors_transposed
  signs <- rep(1, q)
  correlations <- numeric(q)
  for (component in seq_len(q)) {
    correlation <- sum(cpu$u[, component] * aligned_u[, component])
    correlations[[component]] <- correlation
    if (is.finite(correlation) && correlation < 0) {
      signs[[component]] <- -1
      aligned_u[, component] <- -aligned_u[, component]
      aligned_vt[component, ] <- -aligned_vt[component, ]
    }
  }
  list(
    u = aligned_u,
    vt = aligned_vt,
    d = as.double(gpu$singular_values),
    signs = signs,
    correlations = correlations
  )
}

svd_residual <- function(
    X, y, y0, pivot, decomposition, q,
    augmented = NULL, refinement_iterations = 0L) {
  threshold <- decomposition$d[[1L]] * sqrt(.Machine$double.eps)
  retained <- decomposition$d > 0 & decomposition$d >= threshold
  augmented_response <- c(y0, rep(0, nrow(decomposition$u) - q))
  projection <- as.vector(crossprod(
    decomposition$u[, retained, drop = FALSE], augmented_response
  ))
  projection <- projection / decomposition$d[retained]
  theta <- as.vector(
    t(decomposition$vt[retained, , drop = FALSE]) %*% projection
  )
  if (refinement_iterations > 0L) {
    if (is.null(augmented)) {
      stop("SVD refinement requires the augmented matrix", call. = FALSE)
    }
    for (iteration in seq_len(refinement_iterations)) {
      correction_response <- as.double(
        augmented_response - augmented %*% theta
      )
      correction_projection <- as.vector(crossprod(
        decomposition$u[, retained, drop = FALSE], correction_response
      ))
      correction_projection <-
        correction_projection / decomposition$d[retained]
      theta <- theta + as.vector(
        t(decomposition$vt[retained, , drop = FALSE]) %*%
          correction_projection
      )
    }
  }
  coefficients <- numeric(q)
  coefficients[pivot] <- theta
  list(
    residual = as.double(y - X %*% coefficients),
    coefficients = coefficients,
    retained = retained,
    rank = sum(retained),
    threshold = threshold
  )
}

p_value <- function(residuals) {
  fastkpc_cuda_legacy_dcov_gamma_cpp_oracle(
    residuals[, 1L], residuals[, 2L], numCol = 35L, index = 1
  )$p.value
}

source_receipt <- readRDS(input_path)
if (any(requested_indices > length(source_receipt$details))) {
  stop("witness index exceeds source receipt", call. = FALSE)
}
load_fastkpc_cuda_native()
if (!fastkpc_cuda_available()) stop("CUDA is unavailable", call. = FALSE)
transcript_bridge <- Sys.getenv("FASTKPC_SVD_TRANSCRIPT_BRIDGE", unset = "")
transcript_max_q <- as.integer(Sys.getenv(
  "FASTKPC_SVD_TRANSCRIPT_MAX_Q", unset = "80"
))
if (is.na(transcript_max_q) || transcript_max_q <= 0L) {
  stop("SVD transcript maximum q is invalid", call. = FALSE)
}
if (nzchar(transcript_bridge)) dyn.load(transcript_bridge)
run_gpu_svd_transcript <- function(augmented) {
  if (nzchar(transcript_bridge)) {
    .Call("C_fastkpc_svd_transcript_minimal", augmented, TRUE)
  } else {
    full_cuda_ci_multi_penalty_gcv_svd_transcript_for_test(
      augmented, force_stable_svd = TRUE
    )
  }
}

loaded_data <- new.env(parent = emptyenv())
case_rows <- list()
target_rows <- list()
variant_rows <- list()
details <- list()

for (output_index in seq_along(requested_indices)) {
  source_index <- requested_indices[[output_index]]
  source_detail <- source_receipt$details[[source_index]]
  request <- source_detail$request
  dataset <- as.character(request$dataset)
  if (!exists(dataset, loaded_data, inherits = FALSE)) {
    payload <- new.env(parent = emptyenv())
    object_names <- load(data_paths[[dataset]], envir = payload)
    if (length(object_names) != 1L) {
      stop("data payload must contain one object", call. = FALSE)
    }
    data <- as.matrix(payload[[object_names[[1L]]]])
    storage.mode(data) <- "double"
    assign(dataset, data, loaded_data)
  }
  data <- get(dataset, loaded_data, inherits = FALSE)
  targets <- as.integer(source_detail$targets)
  Y <- unname(data[, targets, drop = FALSE])
  native_setup <- full_cuda_ci_native_setup_native(
    data[, source_detail$conditioning_set, drop = FALSE]
  )
  setup <- prepared_setup_from_native(native_setup)
  prepared <- cuda_geometry_from_setup(setup)
  q <- ncol(prepared$X)
  if (q > transcript_max_q) {
    stop("SVD transcript witness exceeds the CUDA transcript q limit",
         call. = FALSE)
  }

  qr_reference <- qr(prepared$X, LAPACK = TRUE)
  qr_geometry_exact <-
    identical(unname(qr_reference$qr), unname(prepared$magic_qr_packed)) &&
    identical(unname(qr_reference$qraux), unname(prepared$magic_tau)) &&
    identical(as.integer(qr_reference$pivot),
              as.integer(prepared$magic_pivot)) &&
    identical(unname(qr.R(qr_reference)), unname(prepared$magic_r))
  if (!qr_geometry_exact) {
    stop("native and LAPACK QR geometry differ", call. = FALSE)
  }

  SP <- unname(source_detail$gpu_sp)
  target_decompositions <- vector("list", ncol(Y))
  target_residuals <- vector("list", ncol(Y))
  cpu_reference_residuals <- matrix(NA_real_, nrow(Y), ncol(Y))

  for (target_index in seq_len(ncol(Y))) {
    root <- aggregate_root(prepared$penalty_matrices, SP[, target_index])
    augmented <- rbind(prepared$magic_r, root$root)
    cpu_svd <- La.svd(augmented, nu = q, nv = q)
    gpu_raw <- run_gpu_svd_transcript(augmented)
    if (gpu_raw$solver_info != 0L || gpu_raw$decomposition_route != 2L) {
      stop("GPU stable SVD transcript failed", call. = FALSE)
    }
    gpu <- align_gpu_svd(cpu_svd, gpu_raw)
    y0 <- qr.qty(qr_reference, Y[, target_index])[seq_len(q)]

    decompositions <- list(
      cpu_all = list(u = cpu_svd$u, vt = cpu_svd$vt, d = cpu_svd$d),
      gpu_all = list(u = gpu$u, vt = gpu$vt, d = gpu$d),
      gpu_u_cpu_v_cpu_d =
        list(u = gpu$u, vt = cpu_svd$vt, d = cpu_svd$d),
      cpu_u_gpu_v_cpu_d =
        list(u = cpu_svd$u, vt = gpu$vt, d = cpu_svd$d),
      cpu_uv_gpu_d =
        list(u = cpu_svd$u, vt = cpu_svd$vt, d = gpu$d),
      gpu_u_cpu_v_gpu_d =
        list(u = gpu$u, vt = cpu_svd$vt, d = gpu$d),
      cpu_u_gpu_v_gpu_d =
        list(u = cpu_svd$u, vt = gpu$vt, d = gpu$d),
      gpu_uv_cpu_d =
        list(u = gpu$u, vt = gpu$vt, d = cpu_svd$d)
    )
    residuals <- lapply(decompositions, function(decomposition) {
      svd_residual(
        prepared$X, Y[, target_index], y0, prepared$magic_pivot,
        decomposition, q, augmented = augmented
      )
    })
    for (refinement_iterations in c(1L, 2L, 4L)) {
      label <- paste0("gpu_all_refine_", refinement_iterations)
      residuals[[label]] <- svd_residual(
        prepared$X, Y[, target_index], y0, prepared$magic_pivot,
        decompositions$gpu_all, q, augmented = augmented,
        refinement_iterations = refinement_iterations
      )
    }
    oracle_residual <-
      as.double(source_detail$residuals$native_gpu_sp[, target_index])
    cpu_evaluator_exact <- TRUE
    if (length(setup$penalty_blocks) > 1L) {
      cpu_evaluation <- fastkpc_full_cuda_phase5_evaluate_cpp(
        setup, Y[, target_index], log(SP[, target_index])
      )
      cpu_evaluator_exact <- identical(
        as.double(residuals$cpu_all$residual),
        as.double(cpu_evaluation$residuals)
      )
    }
    cpu_reference_exact <- cpu_evaluator_exact && identical(
      as.double(residuals$cpu_all$residual),
      oracle_residual
    )
    if (!cpu_reference_exact) {
      stop("CPU transcript does not reproduce the fixed-SP oracle",
           call. = FALSE)
    }
    cpu_reference_residuals[, target_index] <- residuals$cpu_all$residual

    cpu_reconstruction <- cpu_svd$u %*% (cpu_svd$d * cpu_svd$vt)
    gpu_reconstruction <- gpu$u %*% (gpu$d * gpu$vt)
    retained <- residuals$cpu_all$retained
    target_rows[[length(target_rows) + 1L]] <- data.frame(
      source_index = source_index,
      dataset = dataset,
      position = as.integer(request$position),
      target = targets[[target_index]],
      q = q,
      augmented_rows = nrow(augmented),
      aggregate_rank = root$rank,
      effective_rank_cpu = residuals$cpu_all$rank,
      effective_rank_gpu = residuals$gpu_all$rank,
      max_singular_absolute_error = max(abs(gpu$d - cpu_svd$d)),
      max_singular_relative_error = max(
        abs(gpu$d - cpu_svd$d) / pmax(abs(cpu_svd$d), .Machine$double.xmin)
      ),
      max_retained_u_error = max(abs(
        gpu$u[, retained, drop = FALSE] -
          cpu_svd$u[, retained, drop = FALSE]
      )),
      max_retained_vt_error = max(abs(
        gpu$vt[retained, , drop = FALSE] -
          cpu_svd$vt[retained, , drop = FALSE]
      )),
      cpu_reconstruction_error = max(abs(cpu_reconstruction - augmented)),
      gpu_reconstruction_error = max(abs(gpu_reconstruction - augmented)),
      cpu_orthogonality_error = max(abs(
        crossprod(cpu_svd$u) - diag(q)
      )),
      gpu_orthogonality_error = max(abs(
        crossprod(gpu$u) - diag(q)
      )),
      cpu_reference_exact = cpu_reference_exact,
      stringsAsFactors = FALSE
    )
    target_decompositions[[target_index]] <- list(
      augmented = augmented,
      root = root,
      cpu = cpu_svd,
      gpu_raw = gpu_raw,
      gpu_aligned = gpu,
      y0 = y0
    )
    target_residuals[[target_index]] <- residuals
  }

  variant_names <- names(target_residuals[[1L]])
  fixed_reference_p <- p_value(cpu_reference_residuals)
  for (variant in variant_names) {
    residual_matrix <- do.call(cbind, lapply(
      target_residuals, function(value) value[[variant]]$residual
    ))
    candidate_p <- p_value(residual_matrix)
    variant_rows[[length(variant_rows) + 1L]] <- data.frame(
      source_index = source_index,
      dataset = dataset,
      position = as.integer(request$position),
      q = q,
      variant = variant,
      p_value = candidate_p,
      fixed_reference_p = fixed_reference_p,
      p_value_error = candidate_p - fixed_reference_p,
      max_residual_error = max(abs(
        residual_matrix - cpu_reference_residuals
      )),
      contract_passed = abs(candidate_p - fixed_reference_p) <= 1e-10,
      stringsAsFactors = FALSE
    )
  }
  gpu_variant <- variant_rows[[length(variant_rows) -
    length(variant_names) + match("gpu_all", variant_names)]]
  case_rows[[length(case_rows) + 1L]] <- data.frame(
    source_index = source_index,
    dataset = dataset,
    position = as.integer(request$position),
    level = as.integer(request$level),
    x = targets[[1L]],
    y = targets[[2L]],
    S_key = as.character(request$S_key),
    q = q,
    fixed_reference_p = fixed_reference_p,
    custom_gpu_p = gpu_variant$p_value,
    custom_gpu_p_error = gpu_variant$p_value_error,
    custom_gpu_contract_passed = gpu_variant$contract_passed,
    current_fixed_solver_p_shift =
      source_receipt$rows$fixed_solver_p_shift[[source_index]],
    qr_geometry_exact = qr_geometry_exact,
    stringsAsFactors = FALSE
  )
  details[[output_index]] <- list(
    source_index = source_index,
    request = request,
    targets = targets,
    SP = SP,
    decompositions = target_decompositions,
    residuals = target_residuals
  )
  cat(sprintf(
    "%d/%d source=%d q=%d fixed=% .17g custom-gpu=% .17g error=% .3e pass=%s\n",
    output_index, length(requested_indices), source_index, q,
    fixed_reference_p, gpu_variant$p_value, gpu_variant$p_value_error,
    gpu_variant$contract_passed
  ))
}

case_rows <- do.call(rbind, case_rows)
target_rows <- do.call(rbind, target_rows)
variant_rows <- do.call(rbind, variant_rows)
receipt <- list(
  schema_version = "mgcv-gpu-svd-transcript-diagnosis-v1",
  native_sha256 = unname(digest::digest(
    file = "fastkpc/build/fastkpc_cuda.so", algo = "sha256",
    serialize = FALSE
  )),
  source_receipt = normalizePath(input_path),
  requested_indices = requested_indices,
  cases = case_rows,
  targets = target_rows,
  variants = variant_rows,
  details = details,
  summary = list(
    case_count = nrow(case_rows),
    target_count = nrow(target_rows),
    cpu_reference_exact = all(target_rows$cpu_reference_exact),
    custom_gpu_contract_pass_count =
      sum(case_rows$custom_gpu_contract_passed),
    max_custom_gpu_p_error = max(abs(case_rows$custom_gpu_p_error)),
    max_gpu_reconstruction_error = max(target_rows$gpu_reconstruction_error),
    max_gpu_orthogonality_error = max(target_rows$gpu_orthogonality_error)
  )
)
saveRDS(receipt, output_path, compress = "xz", version = 3L)
print(case_rows, digits = 17, row.names = FALSE)
print(target_rows, digits = 17, row.names = FALSE)
print(variant_rows, digits = 17, row.names = FALSE)
print(receipt$summary, digits = 17)
