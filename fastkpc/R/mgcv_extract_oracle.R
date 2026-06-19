source("fastkpc/R/mgcv_compat_contract.R")

fastkpc_require_mgcv <- function() {
  if (!requireNamespace("mgcv", quietly = TRUE)) {
    stop("mgcv is required for mgcvExtractCPU", call. = FALSE)
  }
  invisible(TRUE)
}

fastkpc_mgcv_hash_numeric <- function(x) {
  fastkpc_hash_object(round(as.numeric(x), digits = 14))
}

fastkpc_mgcv_penalty_hashes <- function(fit) {
  hashes <- character()
  smooths <- fit$smooth
  for (i in seq_along(smooths)) {
    penalties <- smooths[[i]]$S
    for (j in seq_along(penalties)) {
      hashes <- c(hashes, fastkpc_hash_object(penalties[[j]]))
    }
  }
  hashes
}

fastkpc_mgcv_selected_sp <- function(fit, fallback = NULL) {
  if (!is.null(fit$sp) && length(fit$sp) > 0L) return(fit$sp)
  if (!is.null(fit$full.sp) && length(fit$full.sp) > 0L) return(fit$full.sp)
  fallback
}

fastkpc_stop_unsupported_setup <- function(message) {
  stop(paste0("Unsupported mgcv fixed-sp setup: ", message), call. = FALSE)
}

fastkpc_mgcv_extract_capabilities <- function() {
  mgcv_version <- tryCatch(
    as.character(utils::packageVersion("mgcv")),
    error = function(e) NA_character_
  )
  list(
    backend = "mgcvExtract",
    role = "version-pinned oracle",
    supported = list(
      family = "gaussian_identity",
      residual_output_only = TRUE,
      full_smooth_when_S_leq_2 = TRUE,
      additive_smooth_when_S_gt_2 = TRUE,
      fixed_sp_self_solve = TRUE,
      gcv_bridge = TRUE,
      all_fixed_L_lsp0_semantics = TRUE,
      canonical_hybrid_verifier = TRUE
    ),
    unsupported = list(
      non_gaussian = TRUE,
      summary_gam = TRUE,
      vcov = TRUE,
      se = TRUE,
      prediction_intervals = TRUE,
      gamm = TRUE,
      by_smooths = TRUE,
      factor_smooths = TRUE,
      tensor_replacement_for_s_s1_s2 = TRUE,
      self_contained_gcv = TRUE,
      cuda_mgcv_subset = TRUE,
      full_mgcv_clone = TRUE,
      bam_gpu = TRUE
    ),
    version_pins = list(
      R_version = R.version.string,
      mgcv_version = mgcv_version
    ),
    baseline = list(
      name = "mgcv Gate B fixed-sp self-solve + hybrid canonical replay",
      tag = "mgcv-gate-b-v1",
      commit = "5da2313"
    )
  )
}

fastkpc_mgcv_extract_gpu_capabilities <- function() {
  mgcv_version <- tryCatch(
    as.character(utils::packageVersion("mgcv")),
    error = function(e) NA_character_
  )
  native_gpu_available <- tryCatch({
    exists("fastkpc_cuda_available", mode = "function") &&
      isTRUE(fastkpc_cuda_available())
  }, error = function(e) FALSE)
  list(
    backend = "mgcvExtractGPU",
    role = "mgcv setup anchored GPU compatibility bridge",
    supported = list(
      family = "gaussian_identity",
      residual_output_only = TRUE,
      fixed_sp_api = TRUE,
      cpu_gate_b_fallback = TRUE,
      native_gpu_fixed_sp_solve = native_gpu_available,
      gcv_bridge_api = FALSE,
      single_penalty_gpu_gcv = native_gpu_available,
      self_contained_gcv = native_gpu_available
    ),
    unsupported = list(
      full_mgcv_clone = TRUE,
      bam_gpu = TRUE,
      non_gaussian = TRUE,
      native_gpu_fixed_sp_solve = !native_gpu_available,
      native_gpu_gcv = !native_gpu_available,
      multi_penalty_gpu_gcv = TRUE,
      tprs_approximation = TRUE
    ),
    version_pins = list(
      R_version = R.version.string,
      mgcv_version = mgcv_version,
      backend_version = "mgcvExtractGPU-fixed-sp-api-v1",
      cpu_fallback_baseline = "mgcv-gate-b-v1",
      cpu_fallback_commit = "5da2313",
      native_cuda_available = native_gpu_available
    )
  )
}

fastkpc_validate_fixed_positive_sp <- function(sp, expected_length = NULL) {
  if (is.null(sp) || length(sp) == 0L) {
    fastkpc_stop_unsupported_setup("sp must be supplied for fixed-sp self-solve")
  }
  sp <- as.numeric(sp)
  if (any(!is.finite(sp)) || any(sp <= 0)) {
    fastkpc_stop_unsupported_setup("sp must contain fixed positive finite values")
  }
  if (!is.null(expected_length) && length(sp) != expected_length) {
    fastkpc_stop_unsupported_setup(
      paste0("length(sp) must equal length(G$S); got ", length(sp),
             " and expected ", expected_length)
    )
  }
  sp
}

fastkpc_setup_weights_policy <- function(G) {
  w <- G$w
  if (is.null(w)) return(list(policy = "none-or-unit", w = NULL))
  w <- as.numeric(w)
  if (length(w) == 0L || all(abs(w - 1) < 1e-12)) {
    return(list(policy = "none-or-unit", w = NULL))
  }
  fastkpc_stop_unsupported_setup("non-unit weights are not supported in Gate B v1")
}

fastkpc_setup_offset_policy <- function(G) {
  offset <- G$offset
  if (is.null(offset)) return(list(policy = "none-or-zero", offset = NULL))
  offset <- as.numeric(offset)
  if (length(offset) == 0L || all(abs(offset) < 1e-12)) {
    return(list(policy = "none-or-zero", offset = NULL))
  }
  fastkpc_stop_unsupported_setup("non-zero offsets are not supported in Gate B v1")
}

fastkpc_check_setup_family <- function(G) {
  fam <- G$family
  if (is.null(fam) ||
      !identical(fam$family, "gaussian") ||
      !identical(fam$link, "identity")) {
    fastkpc_stop_unsupported_setup("only gaussian identity family is supported")
  }
  "gaussian_identity"
}

fastkpc_check_setup_L <- function(G) {
  L <- G$L
  if (is.null(L) || length(L) == 0L) return(invisible(TRUE))
  L <- as.matrix(L)
  if (nrow(L) == ncol(L) && all(abs(L - diag(nrow(L))) < 1e-12)) {
    return(invisible(TRUE))
  }
  fastkpc_stop_unsupported_setup("non-identity smoothing parameter mapping G$L")
}

fastkpc_mgcv_extract_setup <- function(formula, data, sp,
                                       method = "GCV.Cp",
                                       target = 1L,
                                       S = integer(),
                                       k = NA_integer_,
                                       bs = "tp") {
  fastkpc_require_mgcv()
  data <- as.data.frame(data)
  sp <- fastkpc_validate_fixed_positive_sp(sp)

  G <- mgcv::gam(
    formula = formula,
    data = data,
    family = stats::gaussian(),
    sp = sp,
    method = method,
    fit = FALSE
  )

  if (is.null(G$X) || is.null(G$y) || is.null(G$S) || is.null(G$off)) {
    fastkpc_stop_unsupported_setup("G must contain X, y, S, and off")
  }
  sp <- fastkpc_validate_fixed_positive_sp(sp, expected_length = length(G$S))
  family <- fastkpc_check_setup_family(G)
  weights_info <- fastkpc_setup_weights_policy(G)
  offset_info <- fastkpc_setup_offset_policy(G)
  fastkpc_check_setup_L(G)
  if (!is.null(G$paraPen) && length(G$paraPen) > 0L) {
    fastkpc_stop_unsupported_setup("paraPen is not supported in Gate B v1")
  }

  X <- as.matrix(G$X)
  y <- as.numeric(G$y)
  sem <- fastkpc_regrxons_semantics(S = S, target = target,
                                    n = length(y), p = ncol(data))
  setup_fp <- fastkpc_setup_fingerprint(
    sem,
    mgcv_version = as.character(utils::packageVersion("mgcv")),
    backend_family = "mgcvExtractCPU",
    backend_version = "setup-fixed-sp-v1",
    k = k,
    bs = bs,
    method = method,
    model_matrix_hash = fastkpc_hash_object(round(as.numeric(X), digits = 14)),
    penalty_hashes = vapply(G$S, fastkpc_hash_object, character(1)),
    constraint_hash = fastkpc_hash_object(G$C),
    rank_metadata = paste0("rank=", paste(G$rank, collapse = "|")),
    weights_policy = weights_info$policy
  )

  list(
    G = G,
    X = X,
    y = y,
    S = G$S,
    off = as.integer(G$off),
    C = G$C,
    rank = G$rank,
    H = G$H,
    w = weights_info$w,
    offset = offset_info$offset,
    sp = sp,
    formula = formula,
    method = method,
    family = family,
    weights_policy = weights_info$policy,
    offset_policy = offset_info$policy,
    mgcv_version = as.character(utils::packageVersion("mgcv")),
    setup_fingerprint = setup_fp
  )
}

fastkpc_mgcv_gam_fixed_sp_reference <- function(formula, data, sp,
                                                method = "GCV.Cp",
                                                target = 1L,
                                                S = integer(),
                                                k = NA_integer_,
                                                bs = "tp") {
  fastkpc_require_mgcv()
  data <- as.data.frame(data)
  sp <- fastkpc_validate_fixed_positive_sp(sp)

  fit <- mgcv::gam(
    formula = formula,
    data = data,
    family = stats::gaussian(),
    sp = sp,
    method = method,
    fit = TRUE
  )

  residuals <- as.numeric(stats::residuals(fit))
  fitted <- as.numeric(stats::fitted(fit))
  coefficients <- as.numeric(stats::coef(fit))
  selected_sp <- fastkpc_mgcv_selected_sp(fit, fallback = sp)
  response <- if (!is.null(fit$y)) fit$y else model.response(stats::model.frame(fit))
  lpmatrix_hash <- tryCatch(
    fastkpc_hash_object(round(as.numeric(stats::predict(fit, type = "lpmatrix")),
                              digits = 14)),
    error = function(e) ""
  )
  sem <- fastkpc_regrxons_semantics(S = S, target = target,
                                    n = length(residuals), p = ncol(data))
  setup <- fastkpc_setup_fingerprint(
    sem,
    mgcv_version = as.character(utils::packageVersion("mgcv")),
    backend_family = "mgcvExtractCPU",
    backend_version = "mgcv-gam-fixed-sp-reference-v1",
    k = k,
    bs = bs,
    method = method,
    model_matrix_hash = lpmatrix_hash,
    penalty_hashes = fastkpc_mgcv_penalty_hashes(fit),
    constraint_hash = fastkpc_hash_object(lapply(fit$smooth, `[[`, "C")),
    rank_metadata = paste0("rank=", fit$rank)
  )
  target_fp <- fastkpc_target_fingerprint(
    target = target,
    y_hash = fastkpc_mgcv_hash_numeric(response),
    sp_input = sp,
    sp_output = selected_sp,
    selected_sp = selected_sp,
    score = if (!is.null(fit$gcv.ubre)) as.numeric(fit$gcv.ubre) else NA_real_,
    edf = if (!is.null(fit$edf)) sum(fit$edf) else NA_real_,
    rank_if_target_specific = fit$rank,
    residual_hash = fastkpc_mgcv_hash_numeric(residuals),
    fitted_hash = fastkpc_mgcv_hash_numeric(fitted)
  )

  list(
    backend_family = "mgcvExtractCPU",
    mode = "mgcv-gam-fixed-sp-reference",
    solve_source = "mgcv",
    sp_source = "fixed-input",
    gcv_source = "none",
    is_self_contained_gcv = FALSE,
    formula = formula,
    method = method,
    coefficients = coefficients,
    sp = selected_sp,
    residuals = residuals,
    fitted = fitted,
    score = if (!is.null(fit$gcv.ubre)) as.numeric(fit$gcv.ubre) else NA_real_,
    edf = if (!is.null(fit$edf)) sum(fit$edf) else NA_real_,
    rank = fit$rank,
    fit = fit,
    setup_fingerprint = setup,
    target_fingerprint = target_fp,
    mgcv_version = as.character(utils::packageVersion("mgcv"))
  )
}

fastkpc_assemble_penalty <- function(p, S, off, sp, H = NULL) {
  p <- as.integer(p)
  if (length(p) != 1L || is.na(p) || p <= 0L) {
    stop("p must be a positive scalar coefficient dimension", call. = FALSE)
  }
  if (length(S) != length(off) || length(S) != length(sp)) {
    stop("length(S), length(off), and length(sp) must match", call. = FALSE)
  }
  sp <- fastkpc_validate_fixed_positive_sp(sp, expected_length = length(S))

  P <- matrix(0, p, p)
  if (!is.null(H)) {
    H <- as.matrix(H)
    if (!identical(dim(H), c(p, p))) {
      stop("H must have dimension p x p", call. = FALSE)
    }
    P <- P + H
  }

  for (j in seq_along(S)) {
    Sj <- as.matrix(S[[j]])
    if (nrow(Sj) != ncol(Sj)) {
      stop("Each penalty matrix S[[j]] must be square", call. = FALSE)
    }
    kj <- nrow(Sj)
    idx <- seq.int(as.integer(off[j]), length.out = kj)
    if (min(idx) < 1L || max(idx) > p) {
      stop("Penalty block indexed by off is outside coefficient dimension",
           call. = FALSE)
    }
    P[idx, idx] <- P[idx, idx, drop = FALSE] + sp[j] * Sj
  }

  P
}

fastkpc_constraint_nullspace <- function(C, p, tol = sqrt(.Machine$double.eps)) {
  if (is.null(C) || length(C) == 0L) return(diag(p))
  C <- as.matrix(C)
  if (nrow(C) == 0L) return(diag(p))
  if (ncol(C) != p) {
    stop("Constraint matrix C must have ncol(C) equal to coefficient dimension",
         call. = FALSE)
  }
  qrCt <- qr(t(C), tol = tol)
  Q <- qr.Q(qrCt, complete = TRUE)
  rC <- qrCt$rank
  if (rC >= p) {
    stop("Constraint matrix leaves no free coefficient space", call. = FALSE)
  }
  Q[, seq.int(rC + 1L, p), drop = FALSE]
}

fastkpc_solve_gaussian_penalized_fixed_sp <- function(
    X, y, S, off, sp, C = NULL, H = NULL, w = NULL,
    tol = sqrt(.Machine$double.eps)) {
  X <- as.matrix(X)
  y <- as.numeric(y)
  if (nrow(X) != length(y)) {
    stop("nrow(X) must equal length(y)", call. = FALSE)
  }
  p <- ncol(X)
  P <- fastkpc_assemble_penalty(p = p, S = S, off = off, sp = sp, H = H)

  if (is.null(w)) {
    Xw <- X
    yw <- y
  } else {
    w <- as.numeric(w)
    if (length(w) != length(y)) {
      stop("length(w) must equal length(y)", call. = FALSE)
    }
    if (any(!is.finite(w)) || any(w < 0)) {
      stop("weights must be finite and nonnegative", call. = FALSE)
    }
    sw <- sqrt(w)
    Xw <- X * sw
    yw <- y * sw
  }

  Z <- fastkpc_constraint_nullspace(C = C, p = p, tol = tol)
  XZ <- Xw %*% Z
  A <- crossprod(XZ) + crossprod(Z, P %*% Z)
  b <- crossprod(XZ, yw)

  theta <- as.numeric(qr.solve(A, b, tol = tol))
  as.numeric(Z %*% theta)
}

fastkpc_mgcv_magic_kernel_fixed_sp_coefficients <- function(
    setup,
    sp = setup$sp,
    control = list(tol = 1e-06,
                   step.half = 25L,
                   rank.tol = sqrt(.Machine$double.eps))) {
  fastkpc_require_mgcv()
  G <- setup$G
  sp <- fastkpc_validate_fixed_positive_sp(sp, expected_length = length(setup$S))

  if (is.null(G$L) || is.null(G$lsp0)) {
    stop("G$L and G$lsp0 are required for mgcv fixed-sp kernel solve",
         call. = FALSE)
  }
  L <- as.matrix(G$L)
  if (nrow(L) != length(setup$S) || ncol(L) != 0L) {
    stop("Only all-fixed mgcv smoothing parameter mappings are supported",
         call. = FALSE)
  }
  lsp0 <- as.numeric(G$lsp0)
  if (length(lsp0) != length(sp) || any(abs(lsp0 - log(sp)) > 1e-7)) {
    stop("G$lsp0 must encode the supplied fixed smoothing parameters",
         call. = FALSE)
  }

  if (is.null(control$tol)) control$tol <- 1e-06
  if (is.null(control$step.half)) control$step.half <- 25L
  if (is.null(control$rank.tol)) control$rank.tol <- sqrt(.Machine$double.eps)

  X <- setup$X
  y <- setup$y
  S <- setup$S
  off <- setup$off
  rank <- setup$rank
  H <- setup$H
  C <- setup$C
  w <- setup$w
  n.p <- length(S)
  n.b <- ncol(X)

  initial_sp <- get("initial.sp", envir = asNamespace("mgcv"))
  mroot <- get("mroot", envir = asNamespace("mgcv"))
  C_magic <- get("C_magic", envir = asNamespace("mgcv"))
  def.sp <- if (n.p > 0L) initial_sp(X, S, off) else numeric()

  if (n.p > 0L) {
    for (i in seq_len(n.p)) {
      B <- mroot(S[[i]], rank = rank[i], method = "chol")
      R <- matrix(0, n.b, ncol(B))
      idx <- seq.int(off[i], length.out = nrow(B))
      R[idx, ] <- B
      S[[i]] <- R
    }
  }

  n.con <- 0L
  ns.qr <- NULL
  if (!is.null(C) && length(C) > 0L && nrow(as.matrix(C)) > 0L) {
    C <- as.matrix(C)
    n.con <- nrow(C)
    ns.qr <- qr(t(C))
    X <- t(qr.qty(ns.qr, t(X)))[, (n.con + 1L):n.b, drop = FALSE]
    if (n.p > 0L) {
      for (i in seq_len(n.p)) {
        S[[i]] <- qr.qty(ns.qr, S[[i]])[(n.con + 1L):n.b, , drop = FALSE]
        if (ncol(S[[i]]) > nrow(S[[i]])) {
          S[[i]] <- t(qr.R(qr(t(S[[i]]))))
        }
      }
    }
    if (!is.null(H)) {
      H <- qr.qty(ns.qr, H)[(n.con + 1L):n.b, , drop = FALSE]
      H <- t(qr.qty(ns.qr, t(H))[(n.con + 1L):n.b, , drop = FALSE])
    }
  }

  if (!is.null(w)) {
    if (is.matrix(w)) {
      y <- w %*% y
      X <- w %*% X
    } else {
      y <- y * as.vector(w)
      X <- as.vector(w) * X
    }
  }

  Si <- array(0, 0)
  cS <- 0
  if (n.p > 0L) {
    for (i in seq_len(n.p)) {
      Si <- c(Si, S[[i]])
      cS[i] <- ncol(S[[i]])
    }
  }

  rdef <- ncol(X) - nrow(X)
  if (rdef > 0L) {
    X <- rbind(X, matrix(0, rdef, ncol(X)))
    y <- c(y, rep(0, rdef))
  }

  q <- ncol(X)
  icontrol <- as.integer(TRUE)
  icontrol[2] <- length(y)
  icontrol[3] <- q
  icontrol[4] <- as.integer(!is.null(H))
  icontrol[5] <- n.p
  icontrol[6] <- as.integer(control$step.half)
  icontrol[7] <- ncol(L)

  um <- .C(C_magic,
           as.double(y),
           X = as.double(X),
           sp = as.double(numeric(0)),
           as.double(def.sp),
           as.double(Si),
           as.double(H),
           as.double(L),
           lsp0 = as.double(lsp0),
           score = as.double(1),
           scale = as.double(1),
           info = as.integer(icontrol),
           as.integer(cS),
           as.double(control$rank.tol),
           rms.grad = as.double(control$tol),
           b = as.double(array(0, q)),
           rV = double(q * q),
           as.double(0),
           as.integer(length(setup$y)),
           as.integer(1))

  beta <- as.numeric(um$b)
  if (!is.null(ns.qr)) {
    beta <- qr.qy(ns.qr, c(rep(0, n.con), beta))
  }
  beta
}

fastkpc_relative_l2_diff <- function(a, b) {
  a <- as.numeric(a)
  b <- as.numeric(b)
  denom <- sqrt(sum(b^2))
  if (denom == 0) return(sqrt(sum((a - b)^2)))
  sqrt(sum((a - b)^2)) / denom
}

fastkpc_mgcv_solve_setup_fixed_sp <- function(setup,
                                              sp = setup$sp,
                                              tol = sqrt(.Machine$double.eps)) {
  sp <- fastkpc_validate_fixed_positive_sp(sp, expected_length = length(setup$S))
  beta <- fastkpc_mgcv_magic_kernel_fixed_sp_coefficients(setup = setup, sp = sp)
  fitted <- as.numeric(setup$X %*% beta)
  residuals <- as.numeric(setup$y - fitted)
  list(
    backend_family = "mgcvExtractCPU",
    mode = "fixed-sp-setup-self-solve",
    solve_source = "fastkpc-fixed-sp",
    sp_source = "fixed-input",
    gcv_source = "none",
    is_self_contained_gcv = FALSE,
    coefficients = beta,
    fitted = fitted,
    residuals = residuals,
    sp = sp,
    setup_fingerprint = setup$setup_fingerprint,
    setup_diagnostics = list(
      n = nrow(setup$X),
      p = ncol(setup$X),
      penalty_count = length(setup$S),
      off = setup$off,
      has_C = !is.null(setup$C) && length(setup$C) > 0L,
      has_H = !is.null(setup$H),
      weights_policy = setup$weights_policy,
      offset_policy = setup$offset_policy,
      solver_kernel = "mgcv-C-magic-fixed-sp"
    ),
    penalty_diagnostics = list(
      sp = sp,
      penalty_dims = lapply(setup$S, dim)
    ),
    constraint_diagnostics = list(
      C_dim = if (is.null(setup$C)) c(0L, ncol(setup$X)) else dim(as.matrix(setup$C))
    ),
    rank_diagnostics = list(rank = setup$rank)
  )
}

fastkpc_mgcv_extract_fixed_sp_solve <- function(formula, data, sp,
                                                method = "GCV.Cp",
                                                target = 1L,
                                                S = integer(),
                                                k = NA_integer_,
                                                bs = "tp",
                                                tol = sqrt(.Machine$double.eps)) {
  ref <- fastkpc_mgcv_gam_fixed_sp_reference(
    formula = formula,
    data = data,
    sp = sp,
    method = method,
    target = target,
    S = S,
    k = k,
    bs = bs
  )
  setup <- fastkpc_mgcv_extract_setup(
    formula = formula,
    data = data,
    sp = ref$sp,
    method = method,
    target = target,
    S = S,
    k = k,
    bs = bs
  )

  solution <- fastkpc_mgcv_solve_setup_fixed_sp(setup = setup, sp = setup$sp, tol = tol)
  beta <- solution$coefficients
  fitted <- solution$fitted
  residuals <- solution$residuals
  target_fp <- fastkpc_target_fingerprint(
    target = target,
    y_hash = fastkpc_mgcv_hash_numeric(setup$y),
    sp_input = sp,
    sp_output = setup$sp,
    selected_sp = setup$sp,
    score = NA_real_,
    edf = NA_real_,
    rank_if_target_specific = setup$rank,
    residual_hash = fastkpc_mgcv_hash_numeric(residuals),
    fitted_hash = fastkpc_mgcv_hash_numeric(fitted)
  )

  list(
    backend_family = "mgcvExtractCPU",
    mode = "fixed-sp-self-solve",
    reference_mode = ref$mode,
    solve_source = "fastkpc-fixed-sp",
    sp_source = "fixed-input",
    gcv_source = "none",
    is_self_contained_gcv = FALSE,
    formula = formula,
    method = method,
    coefficients = beta,
    fitted = fitted,
    residuals = residuals,
    sp = setup$sp,
    score = NA_real_,
    edf = NA_real_,
    rank = setup$rank,
    reference_coefficients = ref$coefficients,
    reference_fitted = ref$fitted,
    reference_residuals = ref$residuals,
    max_abs_fitted_diff = max(abs(fitted - ref$fitted)),
    relative_l2_fitted_diff = fastkpc_relative_l2_diff(fitted, ref$fitted),
    max_abs_residual_diff = max(abs(residuals - ref$residuals)),
    relative_l2_residual_diff = fastkpc_relative_l2_diff(residuals, ref$residuals),
    setup_diagnostics = solution$setup_diagnostics,
    penalty_diagnostics = solution$penalty_diagnostics,
    constraint_diagnostics = solution$constraint_diagnostics,
    rank_diagnostics = solution$rank_diagnostics,
    setup = setup,
    reference = ref,
    setup_fingerprint = setup$setup_fingerprint,
    target_fingerprint = target_fp,
    mgcv_version = setup$mgcv_version
  )
}

fastkpc_mgcv_extract_fixed_sp <- function(formula, data, sp,
                                          method = "GCV.Cp",
                                          target = 1L,
                                          S = integer(),
                                          k = NA_integer_,
                                          bs = "tp") {
  out <- fastkpc_mgcv_extract_fixed_sp_solve(
    formula = formula,
    data = data,
    sp = sp,
    method = method,
    target = target,
    S = S,
    k = k,
    bs = bs
  )
  out$compatibility_alias_for <- "fastkpc_mgcv_extract_fixed_sp_solve"
  out
}

fastkpc_mgcv_extract_gpu_fixed_sp <- function(formula, data, sp,
                                              method = "GCV.Cp",
                                              target = 1L,
                                              S = integer(),
                                              k = NA_integer_,
                                              bs = "tp",
                                              device = c("cuda", "auto", "cpu"),
                                              allow_cpu_fallback = TRUE,
                                              solve_strategy = c("gate_b", "handle"),
                                              tol = sqrt(.Machine$double.eps)) {
  device <- match.arg(device)
  solve_strategy <- match.arg(solve_strategy)
  run_gate_b_solve <- function() {
    fastkpc_mgcv_extract_fixed_sp_solve(
      formula = formula,
      data = data,
      sp = sp,
      method = method,
      target = target,
      S = S,
      k = k,
      bs = bs,
      tol = tol
    )
  }
  run_handle_solve <- function(use_native_cuda) {
    ref <- fastkpc_mgcv_gam_fixed_sp_reference(
      formula = formula,
      data = data,
      sp = sp,
      method = method,
      target = target,
      S = S,
      k = k,
      bs = bs
    )
    setup <- fastkpc_mgcv_extract_setup(
      formula = formula,
      data = data,
      sp = ref$sp,
      method = method,
      target = target,
      S = S,
      k = k,
      bs = bs
    )
    handle <- fastkpc_mgcv_extract_gpu_setup_handle(
      setup = setup,
      sp = setup$sp,
      device_resident = isTRUE(use_native_cuda),
      tol = tol
    )
    solved <- if (isTRUE(use_native_cuda)) {
      fastkpc_mgcv_extract_gpu_solve_handle_fixed_sp_cuda(handle = handle)
    } else {
      fastkpc_mgcv_extract_gpu_solve_handle_fixed_sp(
        handle = handle,
        tol = tol
      )
    }
    target_fp <- fastkpc_target_fingerprint(
      target = target,
      y_hash = fastkpc_mgcv_hash_numeric(setup$y),
      sp_input = sp,
      sp_output = setup$sp,
      selected_sp = setup$sp,
      score = NA_real_,
      edf = NA_real_,
      rank_if_target_specific = setup$rank,
      residual_hash = fastkpc_mgcv_hash_numeric(solved$residuals),
      fitted_hash = fastkpc_mgcv_hash_numeric(solved$fitted)
    )
    solved$formula <- formula
    solved$method <- method
    solved$reference_mode <- ref$mode
    solved$reference_coefficients <- ref$coefficients
    solved$reference_fitted <- ref$fitted
    solved$reference_residuals <- ref$residuals
    solved$max_abs_fitted_diff <- max(abs(solved$fitted - ref$fitted))
    solved$relative_l2_fitted_diff <- fastkpc_relative_l2_diff(solved$fitted, ref$fitted)
    solved$max_abs_residual_diff <- max(abs(solved$residuals - ref$residuals))
    solved$relative_l2_residual_diff <- fastkpc_relative_l2_diff(solved$residuals, ref$residuals)
    solved$setup <- setup
    solved$reference <- ref
    solved$target_fingerprint <- target_fp
    solved$mgcv_version <- setup$mgcv_version
    solved
  }
  native_cuda_available <- function() {
    exists("fastkpc_mgcv_extract_gpu_solve_handle_fixed_sp_cuda", mode = "function") &&
      exists("fastkpc_cuda_available", mode = "function") &&
      isTRUE(tryCatch(fastkpc_cuda_available(), error = function(e) FALSE))
  }
  run_fallback_solve <- function() {
    if (identical(solve_strategy, "gate_b")) {
      return(run_gate_b_solve())
    }
    run_handle_solve(use_native_cuda = FALSE)
  }

  if (identical(device, "cpu")) {
    solved <- run_fallback_solve()
    requested_device <- "cpu"
    used_device <- "cpu"
    fallback_used <- FALSE
    fallback_reason <- ""
  } else if (identical(solve_strategy, "handle") && native_cuda_available()) {
    solved <- run_handle_solve(use_native_cuda = TRUE)
    requested_device <- device
    used_device <- "cuda"
    fallback_used <- FALSE
    fallback_reason <- ""
  } else if (isTRUE(allow_cpu_fallback)) {
    solved <- run_fallback_solve()
    requested_device <- device
    used_device <- "cpu"
    fallback_used <- TRUE
    fallback_reason <- paste(
      "mgcvExtractGPU native fixed-sp solve is unavailable;",
      "using Gate B CPU fixed-sp self-solve"
    )
  } else {
    stop(
      "mgcvExtractGPU native fixed-sp solve is unavailable and CPU fallback is disabled",
      call. = FALSE
    )
  }

  solved$backend_family <- "mgcvExtractGPU"
  solved$mode <- "fixed-sp-gpu-bridge"
  solved$requested_device <- requested_device
  solved$used_device <- used_device
  solved$fallback_used <- fallback_used
  solved$fallback_reason <- fallback_reason
  solved$gpu_bridge_version <- "mgcvExtractGPU-fixed-sp-api-v1"
  solved$native_gpu_solve_available <- native_cuda_available()
  solved$solve_strategy <- solve_strategy
  solved$capabilities <- fastkpc_mgcv_extract_gpu_capabilities()
  solved
}

fastkpc_mgcv_extract_gpu_setup_handle <- function(
    setup,
    sp = setup$sp,
    device_resident = FALSE,
    tol = sqrt(.Machine$double.eps)) {
  sp <- fastkpc_validate_fixed_positive_sp(sp, expected_length = length(setup$S))
  X <- as.matrix(setup$X)
  y <- as.numeric(setup$y)
  p <- ncol(X)
  P <- fastkpc_assemble_penalty(
    p = p,
    S = setup$S,
    off = setup$off,
    sp = sp,
    H = setup$H
  )
  Z <- fastkpc_constraint_nullspace(C = setup$C, p = p, tol = tol)
  X_null <- X %*% Z
  penalty_null <- crossprod(Z, P %*% Z)
  XtX_null <- crossprod(X_null)
  Xty_null <- crossprod(X_null, y)
  constraint_rank <- p - ncol(Z)

  list(
    backend_family = "mgcvExtractGPU",
    mode = "fixed-sp-setup-handle",
    handle_version = "mgcvExtractGPU-setup-handle-v1",
    device_resident = isTRUE(device_resident),
    native_gpu_solve_available = FALSE,
    setup_fingerprint = setup$setup_fingerprint,
    mgcv_version = setup$mgcv_version,
    sp = sp,
    X = X,
    y = y,
    penalty = P,
    Z = Z,
    X_null = X_null,
    penalty_null = penalty_null,
    XtX_null = XtX_null,
    Xty_null = Xty_null,
    diagnostics = list(
      n = nrow(X),
      coefficient_dim = p,
      null_dim = ncol(Z),
      constraint_rank = as.integer(constraint_rank),
      penalty_count = length(setup$S),
      penalty_dims = lapply(setup$S, dim),
      off = setup$off,
      has_C = !is.null(setup$C) && length(setup$C) > 0L,
      has_H = !is.null(setup$H),
      weights_policy = setup$weights_policy,
      offset_policy = setup$offset_policy,
      setup_stage = "host-prepared",
      device_resident = isTRUE(device_resident)
    )
  )
}

fastkpc_mgcv_extract_gpu_solve_handle_fixed_sp <- function(
    handle,
    tol = sqrt(.Machine$double.eps)) {
  A <- handle$XtX_null + handle$penalty_null
  b <- handle$Xty_null
  theta <- as.numeric(qr.solve(A, b, tol = tol))
  beta <- as.numeric(handle$Z %*% theta)
  fitted <- as.numeric(handle$X %*% beta)
  residuals <- as.numeric(handle$y - fitted)

  list(
    backend_family = "mgcvExtractGPU",
    mode = "fixed-sp-handle-solve",
    solve_source = "fastkpc-handle-fixed-sp",
    sp_source = "fixed-input",
    gcv_source = "none",
    is_self_contained_gcv = FALSE,
    used_device = "cpu",
    native_gpu_solve_used = FALSE,
    coefficients = beta,
    theta = theta,
    fitted = fitted,
    residuals = residuals,
    sp = handle$sp,
    rss = sum(residuals^2),
    setup_fingerprint = handle$setup_fingerprint,
    handle_version = handle$handle_version,
    diagnostics = c(
      handle$diagnostics,
      list(
        solve_stage = "host-handle-linear-solve",
        linear_system_dim = ncol(handle$X_null)
      )
    )
  )
}

fastkpc_mgcv_extract_gpu_solve_handle_fixed_sp_cuda <- function(handle) {
  if (!exists("mgcv_extract_gpu_solve_handle_fixed_sp_cuda", mode = "function")) {
    stop(
      "mgcvExtractGPU native fixed-sp solve wrapper is unavailable; source fastkpc/R/cuda_native.R",
      call. = FALSE
    )
  }
  native <- mgcv_extract_gpu_solve_handle_fixed_sp_cuda(handle)
  list(
    backend_family = "mgcvExtractGPU",
    mode = "fixed-sp-native-gpu-solve",
    solve_source = "mgcvExtractGPU-native-fixed-sp",
    sp_source = "fixed-input",
    gcv_source = "none",
    is_self_contained_gcv = FALSE,
    used_device = "cuda",
    native_gpu_solve_used = TRUE,
    coefficients = as.numeric(native$coefficients),
    theta = as.numeric(native$theta),
    fitted = as.numeric(native$fitted),
    residuals = as.numeric(native$residuals),
    sp = handle$sp,
    rss = as.numeric(native$rss),
    setup_fingerprint = handle$setup_fingerprint,
    handle_version = handle$handle_version,
    diagnostics = c(
      handle$diagnostics,
      native$diagnostics,
      list(
        device_resident = TRUE,
        native_gpu_solve_used = TRUE
      )
    )
  )
}

fastkpc_mgcv_extract_gpu_edf_for_handle <- function(handle, sp) {
  if (length(handle$sp) != 1L || length(sp) != 1L) {
    stop("EDF helper currently supports exactly one smoothing parameter",
         call. = FALSE)
  }
  base_sp <- as.numeric(handle$sp)
  sp <- as.numeric(sp)
  if (!is.finite(base_sp) || base_sp <= 0 || !is.finite(sp) || sp <= 0) {
    stop("sp values must be positive and finite", call. = FALSE)
  }
  penalty_at_sp <- handle$penalty_null * (sp / base_sp)
  A <- handle$XtX_null + penalty_at_sp
  A_inv <- tryCatch(solve(A), error = function(e) NULL)
  if (is.null(A_inv)) return(NA_real_)
  as.numeric(sum(diag(handle$XtX_null %*% A_inv)))
}

fastkpc_mgcv_extract_gpu_spectral_prepare <- function(
    handle,
    tol = sqrt(.Machine$double.eps)) {
  if (length(handle$sp) != 1L) {
    stop("spectral GCV currently supports exactly one smoothing parameter",
         call. = FALSE)
  }
  base_sp <- as.numeric(handle$sp)
  if (!is.finite(base_sp) || base_sp <= 0) {
    stop("handle sp must be positive and finite", call. = FALSE)
  }
  XtX <- as.matrix(handle$XtX_null)
  P_base <- as.matrix(handle$penalty_null) / base_sp
  chol_xtx <- tryCatch(chol(XtX, pivot = FALSE), error = function(e) NULL)
  if (is.null(chol_xtx)) {
    stop("spectral GCV requires positive definite XtX in the constraint null space",
         call. = FALSE)
  }
  inv_chol <- backsolve(chol_xtx, diag(ncol(XtX)))
  symmetric_penalty <- crossprod(inv_chol, P_base %*% inv_chol)
  symmetric_penalty <- (symmetric_penalty + t(symmetric_penalty)) / 2
  eigen_penalty <- eigen(symmetric_penalty, symmetric = TRUE)
  d <- pmax(as.numeric(eigen_penalty$values), 0)
  list(
    base_sp = base_sp,
    inv_chol = inv_chol,
    eigenvectors = eigen_penalty$vectors,
    eigenvalues = d,
    spectral_rank = length(d)
  )
}

fastkpc_mgcv_extract_gpu_spectral_score_grid <- function(
    spectral,
    y,
    Xty_null,
    sp_grid,
    tol = sqrt(.Machine$double.eps)) {
  y <- as.numeric(y)
  Xty_null <- as.numeric(Xty_null)
  d <- as.numeric(spectral$eigenvalues)
  n <- length(y)
  z <- as.numeric(t(spectral$eigenvectors) %*%
                    as.numeric(crossprod(spectral$inv_chol, Xty_null)))
  y_sq <- sum(y^2)

  rows <- lapply(sp_grid, function(sp_value) {
    h <- 1 / (1 + as.numeric(sp_value) * d)
    rss <- y_sq - 2 * sum(h * z^2) + sum((h^2) * z^2)
    rss <- max(0, rss)
    edf <- sum(h)
    denom <- n - edf
    gcv <- if (is.finite(edf) && denom > tol) n * rss / (denom * denom) else Inf
    data.frame(
      sp = as.numeric(sp_value),
      rss = as.numeric(rss),
      edf = as.numeric(edf),
      gcv = as.numeric(gcv),
      valid = is.finite(gcv),
      stringsAsFactors = FALSE
    )
  })
  grid <- do.call(rbind, rows)
  list(
    grid = grid,
    eigenvalues = d,
    spectral_rank = length(d)
  )
}

fastkpc_mgcv_extract_gpu_spectral_gcv_grid <- function(handle, sp_grid,
                                                       tol = sqrt(.Machine$double.eps)) {
  spectral <- fastkpc_mgcv_extract_gpu_spectral_prepare(handle, tol = tol)
  fastkpc_mgcv_extract_gpu_spectral_score_grid(
    spectral = spectral,
    y = handle$y,
    Xty_null = handle$Xty_null,
    sp_grid = sp_grid,
    tol = tol
  )
}

fastkpc_mgcv_extract_gpu_gcv <- function(
    formula,
    data,
    setup_sp = 1,
    sp_grid,
    method = "GCV.Cp",
    target = 1L,
    S = integer(),
    k = NA_integer_,
    bs = "tp",
    device = c("cuda", "auto", "cpu"),
    allow_cpu_fallback = TRUE,
    gcv_strategy = c("direct", "spectral"),
    tol = sqrt(.Machine$double.eps)) {
  device <- match.arg(device)
  gcv_strategy <- match.arg(gcv_strategy)
  sp_grid <- fastkpc_validate_fixed_positive_sp(sp_grid)
  if (length(sp_grid) == 0L) {
    stop("sp_grid must contain at least one candidate", call. = FALSE)
  }
  setup <- fastkpc_mgcv_extract_setup(
    formula = formula,
    data = data,
    sp = setup_sp,
    method = method,
    target = target,
    S = S,
    k = k,
    bs = bs
  )
  if (length(setup$S) != 1L) {
    stop("mgcvExtractGPU GCV currently supports single-penalty setups only",
         call. = FALSE)
  }
  if (!identical(method, "GCV.Cp")) {
    stop("mgcvExtractGPU GCV v1 supports method = 'GCV.Cp' only",
         call. = FALSE)
  }
  native_cuda_available <- exists(
    "fastkpc_mgcv_extract_gpu_solve_handle_fixed_sp_cuda",
    mode = "function"
  ) && exists("fastkpc_cuda_available", mode = "function") &&
    isTRUE(tryCatch(fastkpc_cuda_available(), error = function(e) FALSE))

  if (!identical(device, "cpu") && !native_cuda_available && !isTRUE(allow_cpu_fallback)) {
    stop("mgcvExtractGPU GCV native CUDA solve is unavailable and CPU fallback is disabled",
         call. = FALSE)
  }

  solve_one <- function(sp_value) {
    handle <- fastkpc_mgcv_extract_gpu_setup_handle(
      setup = setup,
      sp = sp_value,
      device_resident = !identical(device, "cpu") && native_cuda_available,
      tol = tol
    )
    solved <- if (!identical(device, "cpu") && native_cuda_available) {
      fastkpc_mgcv_extract_gpu_solve_handle_fixed_sp_cuda(handle)
    } else {
      fastkpc_mgcv_extract_gpu_solve_handle_fixed_sp(handle, tol = tol)
    }
    list(handle = handle, solved = solved)
  }

  if (identical(gcv_strategy, "spectral")) {
    base_handle <- fastkpc_mgcv_extract_gpu_setup_handle(
      setup = setup,
      sp = setup_sp,
      device_resident = !identical(device, "cpu") && native_cuda_available,
      tol = tol
    )
    spectral <- fastkpc_mgcv_extract_gpu_spectral_gcv_grid(
      handle = base_handle,
      sp_grid = sp_grid,
      tol = tol
    )
    grid <- spectral$grid
    rss <- grid$rss
    edf <- grid$edf
    gcv <- grid$gcv
    evaluations <- vector("list", length(sp_grid))
  } else {
    evaluations <- lapply(sp_grid, function(sp_value) {
      solved <- solve_one(sp_value)
      edf <- fastkpc_mgcv_extract_gpu_edf_for_handle(
        solved$handle,
        sp = sp_value
      )
      denom <- length(solved$solved$residuals) - edf
      gcv <- if (is.finite(edf) && denom > 1e-8) {
        length(solved$solved$residuals) * solved$solved$rss / (denom * denom)
      } else {
        Inf
      }
      c(solved, list(edf = edf, gcv = gcv))
    })
    rss <- vapply(evaluations, function(x) x$solved$rss, numeric(1))
    edf <- vapply(evaluations, function(x) x$edf, numeric(1))
    gcv <- vapply(evaluations, function(x) x$gcv, numeric(1))
    spectral <- NULL
  }
  valid <- is.finite(rss) & is.finite(edf) & is.finite(gcv)
  if (!any(valid)) {
    stop("No valid GCV candidate in sp_grid", call. = FALSE)
  }
  selected_grid_index <- which(gcv == min(gcv[valid]))[1L]
  selected_sp <- sp_grid[selected_grid_index]
  if (identical(gcv_strategy, "spectral")) {
    selected_eval <- solve_one(selected_sp)
    selected <- selected_eval$solved
    selected_handle <- selected_eval$handle
  } else {
    selected <- evaluations[[selected_grid_index]]$solved
    selected_handle <- evaluations[[selected_grid_index]]$handle
  }
  target_fp <- fastkpc_target_fingerprint(
    target = target,
    y_hash = fastkpc_mgcv_hash_numeric(setup$y),
    sp_input = setup_sp,
    sp_output = selected_sp,
    selected_sp = selected_sp,
    score = gcv[selected_grid_index],
    edf = edf[selected_grid_index],
    rank_if_target_specific = setup$rank,
    residual_hash = fastkpc_mgcv_hash_numeric(selected$residuals),
    fitted_hash = fastkpc_mgcv_hash_numeric(selected$fitted)
  )

  used_cuda <- identical(selected$used_device, "cuda")
  gcv_score_backend <- if (identical(gcv_strategy, "spectral")) {
    "r-cpu-spectral"
  } else if (used_cuda) {
    "cuda-grid-solve"
  } else {
    "cpu-grid-solve"
  }
  sp_selection_backend <- gcv_score_backend
  selected_solve_backend <- selected$used_device
  list(
    backend_family = "mgcvExtractGPU",
    mode = "single-penalty-gpu-gcv",
    solve_source = "mgcvExtractGPU",
    sp_source = paste0("fastkpc-", sp_selection_backend),
    sp_selection_backend_executed = sp_selection_backend,
    gcv_source = paste0("fastkpc-", gcv_score_backend),
    gcv_score_backend_executed = gcv_score_backend,
    selected_solve_backend_executed = selected_solve_backend,
    is_self_contained_gcv = TRUE,
    used_device = selected$used_device,
    native_gpu_solve_used = isTRUE(selected$native_gpu_solve_used),
    fallback_used = !used_cuda && !identical(device, "cpu"),
    fallback_reason = if (!used_cuda && !identical(device, "cpu")) {
      "mgcvExtractGPU GCV native CUDA solve is unavailable; using CPU handle solve"
    } else {
      ""
    },
    formula = formula,
    method = method,
    coefficients = selected$coefficients,
    theta = selected$theta,
    fitted = selected$fitted,
    residuals = selected$residuals,
    sp = selected_sp,
    score = gcv[selected_grid_index],
    edf = edf[selected_grid_index],
    rss = selected$rss,
    selected_grid_index = selected_grid_index,
    grid = data.frame(
      sp = as.numeric(sp_grid),
      rss = as.numeric(rss),
      edf = as.numeric(edf),
      gcv = as.numeric(gcv),
      valid = as.logical(valid)
    ),
    setup = setup,
    setup_fingerprint = setup$setup_fingerprint,
    target_fingerprint = target_fp,
    handle_version = selected_handle$handle_version,
    diagnostics = c(
      selected$diagnostics,
      list(
        gcv_stage = if (identical(gcv_strategy, "spectral")) {
          "single-penalty-spectral-grid-search"
        } else {
          "single-penalty-grid-search"
        },
        gcv_strategy = gcv_strategy,
        spectral_reparameterization = identical(gcv_strategy, "spectral"),
        spectral_rank = if (identical(gcv_strategy, "spectral")) {
          spectral$spectral_rank
        } else {
          NA_integer_
        },
        grid_size = length(sp_grid),
        selected_grid_index = selected_grid_index,
        setup_sp = as.numeric(setup_sp),
        penalty_count = length(setup$S),
        sp_selection_backend_executed = sp_selection_backend,
        gcv_score_backend_executed = gcv_score_backend,
        selected_solve_backend_executed = selected_solve_backend,
        is_self_contained_gcv = TRUE
      )
    ),
    mgcv_version = setup$mgcv_version,
    capabilities = fastkpc_mgcv_extract_gpu_capabilities()
  )
}

fastkpc_mgcv_extract_gpu_solve_handle_batch_fixed_sp <- function(
    setups,
    target_ids = seq_along(setups),
    tol = sqrt(.Machine$double.eps)) {
  if (!is.list(setups) || length(setups) == 0L) {
    stop("setups must be a non-empty list of mgcv extracted setups", call. = FALSE)
  }
  if (length(target_ids) != length(setups)) {
    stop("length(target_ids) must equal length(setups)", call. = FALSE)
  }

  handles <- lapply(setups, fastkpc_mgcv_extract_gpu_setup_handle, tol = tol)
  solved <- lapply(handles, fastkpc_mgcv_extract_gpu_solve_handle_fixed_sp, tol = tol)
  n <- length(solved[[1]]$residuals)
  q <- length(solved)
  if (any(vapply(solved, function(x) length(x$residuals) != n, logical(1)))) {
    stop("all setup solves must have the same row count", call. = FALSE)
  }

  residuals <- do.call(cbind, lapply(solved, `[[`, "residuals"))
  fitted <- do.call(cbind, lapply(solved, `[[`, "fitted"))
  coefficients <- lapply(solved, `[[`, "coefficients")
  setup_fingerprints <- vapply(
    solved,
    function(x) x$setup_fingerprint$fingerprint,
    character(1)
  )
  sp <- vapply(solved, function(x) as.numeric(x$sp)[1L], numeric(1))
  colnames(residuals) <- paste0("target", as.integer(target_ids))
  colnames(fitted) <- colnames(residuals)

  list(
    backend_family = "mgcvExtractGPU",
    mode = "fixed-sp-handle-batch-solve",
    solve_source = "fastkpc-handle-fixed-sp-batch",
    sp_source = "fixed-input-per-target",
    gcv_source = "none",
    is_self_contained_gcv = FALSE,
    used_device = "cpu",
    native_gpu_solve_used = FALSE,
    residuals = residuals,
    fitted = fitted,
    coefficients = coefficients,
    sp = sp,
    target_ids = as.integer(target_ids),
    setup_fingerprints = setup_fingerprints,
    handles = handles,
    solved = solved,
    diagnostics = list(
      targets = as.integer(q),
      n = as.integer(n),
      same_setup_fingerprint_count = length(unique(setup_fingerprints)),
      device_resident = FALSE,
      solve_stage = "host-handle-batch-linear-solve"
    )
  )
}

fastkpc_mgcv_extract_gpu_solve_handle_batch_fixed_sp_cuda <- function(
    setups,
    target_ids = seq_along(setups),
    tol = sqrt(.Machine$double.eps)) {
  if (!exists("fastkpc_mgcv_extract_gpu_solve_handle_fixed_sp_cuda",
              mode = "function")) {
    stop(
      "mgcvExtractGPU native fixed-sp solve wrapper is unavailable; source fastkpc/R/cuda_native.R",
      call. = FALSE
    )
  }
  if (!exists("fastkpc_cuda_available", mode = "function") ||
      !isTRUE(tryCatch(fastkpc_cuda_available(), error = function(e) FALSE))) {
    stop("mgcvExtractGPU native CUDA solve is unavailable", call. = FALSE)
  }
  if (!is.list(setups) || length(setups) == 0L) {
    stop("setups must be a non-empty list of mgcv extracted setups", call. = FALSE)
  }
  if (length(target_ids) != length(setups)) {
    stop("length(target_ids) must equal length(setups)", call. = FALSE)
  }

  handles <- lapply(
    setups,
    fastkpc_mgcv_extract_gpu_setup_handle,
    device_resident = TRUE,
    tol = tol
  )
  solved <- lapply(handles, fastkpc_mgcv_extract_gpu_solve_handle_fixed_sp_cuda)
  n <- length(solved[[1]]$residuals)
  q <- length(solved)
  if (any(vapply(solved, function(x) length(x$residuals) != n, logical(1)))) {
    stop("all setup solves must have the same row count", call. = FALSE)
  }

  residuals <- do.call(cbind, lapply(solved, `[[`, "residuals"))
  fitted <- do.call(cbind, lapply(solved, `[[`, "fitted"))
  coefficients <- lapply(solved, `[[`, "coefficients")
  theta <- lapply(solved, `[[`, "theta")
  setup_fingerprints <- vapply(
    solved,
    function(x) x$setup_fingerprint$fingerprint,
    character(1)
  )
  sp <- vapply(solved, function(x) as.numeric(x$sp)[1L], numeric(1))
  rss <- vapply(solved, function(x) as.numeric(x$rss), numeric(1))
  colnames(residuals) <- paste0("target", as.integer(target_ids))
  colnames(fitted) <- colnames(residuals)

  list(
    backend_family = "mgcvExtractGPU",
    mode = "fixed-sp-native-gpu-batch-bridge",
    solve_source = "mgcvExtractGPU-native-fixed-sp-batch-bridge",
    sp_source = "fixed-input-per-target",
    gcv_source = "none",
    is_self_contained_gcv = FALSE,
    used_device = "cuda",
    native_gpu_solve_used = TRUE,
    residuals = residuals,
    fitted = fitted,
    coefficients = coefficients,
    theta = theta,
    sp = sp,
    rss = rss,
    target_ids = as.integer(target_ids),
    setup_fingerprints = setup_fingerprints,
    handles = handles,
    solved = solved,
    diagnostics = list(
      targets = as.integer(q),
      n = as.integer(n),
      same_setup_fingerprint_count = length(unique(setup_fingerprints)),
      device_resident = TRUE,
      solve_stage = "native-fixed-sp-repeated-handle-solve",
      batch_stage = "native-fixed-sp-repeated-handle-bridge",
      true_batched_kernel = FALSE
    )
  )
}

fastkpc_mgcv_regrxons_rhs <- function(S_data, S, k = NA_integer_, bs = "tp",
                                      formula_class = NULL) {
  if (is.null(formula_class)) formula_class <- fastkpc_regrxons_formula_class(S)
  terms <- colnames(S_data)
  smooth_arg <- function(vars) {
    args <- vars
    if (!is.na(k)) args <- c(args, paste0("k = ", as.integer(k)))
    if (!is.null(bs) && nzchar(bs)) args <- c(args, paste0("bs = \"", bs, "\""))
    paste0("s(", paste(args, collapse = ", "), ")")
  }
  if (identical(formula_class, "additive-smooth")) {
    paste(vapply(terms, smooth_arg, character(1)), collapse = " + ")
  } else {
    smooth_arg(terms)
  }
}

fastkpc_mgcv_extract_retarget_setup <- function(setup, y, sp, target) {
  y <- as.numeric(y)
  if (length(y) != nrow(setup$X)) {
    stop("target y length must match setup row count", call. = FALSE)
  }
  sp <- fastkpc_validate_fixed_positive_sp(sp, expected_length = length(setup$S))
  out <- setup
  out$y <- y
  out$sp <- sp
  out$target <- as.integer(target)
  out$setup_fingerprint <- setup$setup_fingerprint
  out
}

fastkpc_mgcv_extract_gpu_same_setup_batch_fixed_sp_cuda <- function(
    Y,
    S_data,
    S,
    sp,
    k = NA_integer_,
    bs = "tp",
    method = "GCV.Cp",
    target_ids = seq_len(ncol(as.matrix(Y))),
    tol = sqrt(.Machine$double.eps)) {
  Y <- as.matrix(Y)
  S_data <- as.data.frame(S_data)
  if (nrow(Y) != nrow(S_data)) {
    stop("Y and S_data must have the same row count", call. = FALSE)
  }
  q <- ncol(Y)
  if (length(target_ids) != q) {
    stop("length(target_ids) must equal ncol(Y)", call. = FALSE)
  }
  if (length(sp) == 1L && q > 1L) {
    sp <- rep(as.numeric(sp), q)
  }
  if (length(sp) != q) {
    stop("sp must have length 1 or one value per target", call. = FALSE)
  }
  sp <- fastkpc_validate_fixed_positive_sp(sp, expected_length = q)

  s_names <- names(S_data)
  if (is.null(s_names) || any(!nzchar(s_names))) {
    s_names <- paste0("s", seq_len(ncol(S_data)))
    names(S_data) <- s_names
  }
  rhs <- fastkpc_mgcv_regrxons_rhs(
    S_data = S_data,
    S = S,
    k = k,
    bs = bs
  )
  formula <- stats::as.formula(paste("y ~", rhs))
  template_data <- data.frame(y = Y[, 1L], S_data, check.names = FALSE)
  template_setup <- fastkpc_mgcv_extract_setup(
    formula = formula,
    data = template_data,
    sp = sp[1L],
    method = method,
    target = target_ids[1L],
    S = S,
    k = k,
    bs = bs
  )
  setups <- lapply(seq_len(q), function(j) {
    fastkpc_mgcv_extract_retarget_setup(
      setup = template_setup,
      y = Y[, j],
      sp = sp[j],
      target = target_ids[j]
    )
  })
  handles <- lapply(
    setups,
    fastkpc_mgcv_extract_gpu_setup_handle,
    device_resident = TRUE,
    tol = tol
  )
  native <- mgcv_extract_gpu_solve_same_setup_batch_fixed_sp_cuda(handles)
  residuals <- as.matrix(native$residuals)
  fitted <- as.matrix(native$fitted)
  colnames(residuals) <- paste0("target", as.integer(target_ids))
  colnames(fitted) <- colnames(residuals)
  setup_fingerprints <- vapply(
    setups,
    function(x) x$setup_fingerprint$fingerprint,
    character(1)
  )
  coefficients <- lapply(seq_len(q), function(j) native$coefficients[, j])
  theta <- lapply(seq_len(q), function(j) native$theta[, j])

  batch <- list(
    backend_family = "mgcvExtractGPU",
    mode = "fixed-sp-same-setup-native-gpu-batch-bridge",
    solve_source = "mgcvExtractGPU-native-same-setup-fixed-sp-batch",
    sp_source = "fixed-input-per-target",
    gcv_source = "none",
    is_self_contained_gcv = FALSE,
    used_device = "cuda",
    native_gpu_solve_used = TRUE,
    residuals = residuals,
    fitted = fitted,
    coefficients = coefficients,
    theta = theta,
    sp = sp,
    rss = as.numeric(native$rss),
    target_ids = as.integer(target_ids),
    setup_fingerprints = setup_fingerprints,
    handles = handles,
    solved = native,
    diagnostics = c(
      native$batch_diagnostics,
      list(
        same_setup_fingerprint_count = length(unique(setup_fingerprints))
      )
    )
  )
  batch$template_setup <- template_setup
  batch$diagnostics$setup_reused <- TRUE
  batch$diagnostics$template_setup_fingerprint <-
    template_setup$setup_fingerprint$fingerprint
  batch
}

fastkpc_mgcv_extract_gcv_bridge <- function(formula, data,
                                            method = "GCV.Cp",
                                            target = 1L,
                                            S = integer(),
                                            k = NA_integer_,
                                            bs = "tp") {
  fastkpc_require_mgcv()
  data <- as.data.frame(data)
  legacy <- mgcv::gam(formula = formula, data = data,
                      family = stats::gaussian(), method = method)
  fixed <- fastkpc_mgcv_extract_fixed_sp_solve(
    formula = formula,
    data = data,
    sp = legacy$sp,
    method = method,
    target = target,
    S = S,
    k = k,
    bs = bs
  )
  fixed$mode <- "gcv-bridge"
  fixed$sp_source <- "mgcv"
  fixed$gcv_source <- "mgcv"
  fixed$solve_source <- "fastkpc-fixed-sp"
  fixed$is_self_contained_gcv <- FALSE
  fixed$legacy_score <- if (!is.null(legacy$gcv.ubre)) as.numeric(legacy$gcv.ubre) else NA_real_
  fixed$legacy_edf <- if (!is.null(legacy$edf)) sum(legacy$edf) else NA_real_
  fixed$legacy_rank <- legacy$rank
  fixed$legacy_sp <- legacy$sp
  fixed$score <- fixed$legacy_score
  fixed$edf <- fixed$legacy_edf
  fixed$rank <- fixed$legacy_rank
  fixed
}

fastkpc_mgcv_extract_batch <- function(Y, S_data, S,
                                       target_ids = seq_len(ncol(Y)),
                                       formula_class = NULL,
                                       method = "GCV.Cp") {
  fastkpc_require_mgcv()
  Y <- as.matrix(Y)
  S_data <- as.data.frame(S_data)
  if (is.null(colnames(Y))) colnames(Y) <- paste0("target", seq_len(ncol(Y)))
  if (is.null(colnames(S_data))) colnames(S_data) <- paste0("s", seq_len(ncol(S_data)))
  if (is.null(formula_class)) formula_class <- fastkpc_regrxons_formula_class(S)

  n <- nrow(Y)
  q <- ncol(Y)
  residuals <- matrix(NA_real_, n, q)
  fitted <- matrix(NA_real_, n, q)
  coefficients <- vector("list", q)
  sp <- vector("list", q)
  score <- rep(NA_real_, q)
  edf <- rep(NA_real_, q)
  ranks <- rep(NA_integer_, q)
  target_fps <- vector("list", q)

  rhs <- fastkpc_mgcv_regrxons_rhs(
    S_data = S_data,
    S = S,
    formula_class = formula_class
  )

  sem <- fastkpc_regrxons_semantics(S = S, target = target_ids[1],
                                    n = n, p = q + ncol(S_data))
  setup_fp <- fastkpc_setup_fingerprint(
    sem,
    mgcv_version = as.character(utils::packageVersion("mgcv")),
    backend_family = "mgcvExtractCPU",
    backend_version = "batch-gcv-bridge-v1",
    method = method,
    model_matrix_hash = fastkpc_hash_object(list(S_data = S_data, rhs = rhs))
  )

  for (j in seq_len(q)) {
    local_data <- cbind(data.frame(.target = Y[, j]), S_data)
    form <- stats::as.formula(paste(".target ~", rhs))
    fit <- fastkpc_mgcv_extract_gcv_bridge(
      formula = form,
      data = local_data,
      method = method,
      target = target_ids[j],
      S = S
    )
    residuals[, j] <- fit$residuals
    fitted[, j] <- fit$fitted
    coefficients[[j]] <- fit$coefficients
    sp[[j]] <- fit$sp
    score[j] <- fit$score
    edf[j] <- fit$edf
    ranks[j] <- fit$rank
    target_fps[[j]] <- fit$target_fingerprint
  }

  colnames(residuals) <- colnames(Y)
  colnames(fitted) <- colnames(Y)
  list(
    backend_family = "mgcvExtractCPU",
    mode = "batch-gcv-bridge",
    solve_source = "fastkpc-fixed-sp",
    sp_source = "mgcv",
    gcv_source = "mgcv",
    is_self_contained_gcv = FALSE,
    residuals = residuals,
    fitted = fitted,
    coefficients = coefficients,
    sp = sp,
    score = score,
    edf = edf,
    rank = ranks,
    setup_fingerprint = setup_fp,
    target_fingerprints = target_fps,
    formula_class = formula_class,
    method = method
  )
}
