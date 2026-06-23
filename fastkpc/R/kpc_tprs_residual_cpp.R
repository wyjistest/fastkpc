source("fastkpc/R/mgcv_compat_contract.R")
source("fastkpc/R/native.R")
source("fastkpc/R/dcov_exact.R")

`%||%` <- function(x, y) if (is.null(x)) y else x

fastkpc_kpc_tprs_stop <- function(message) {
  stop(paste0("kpcTprsResidualCPP unsupported input: ", message),
       call. = FALSE)
}

fastkpc_kpc_tprs_validate_input <- function(y, S) {
  y <- as.numeric(y)
  if (length(y) == 0L || any(!is.finite(y))) {
    fastkpc_kpc_tprs_stop("y must be finite numeric")
  }
  S <- as.matrix(S)
  storage.mode(S) <- "double"
  if (nrow(S) != length(y)) {
    fastkpc_kpc_tprs_stop("y and S must have the same number of rows")
  }
  if (ncol(S) < 1L || ncol(S) > 2L) {
    fastkpc_kpc_tprs_stop("|S| = 1 or 2 is required")
  }
  if (any(!is.finite(S))) {
    fastkpc_kpc_tprs_stop("S must be finite numeric")
  }
  list(y = y, S = S)
}

fastkpc_kpc_tprs_setup_fingerprint <- function(S, setup) {
  fastkpc_hash_object(list(
    backend = "kpcTprsResidualCPP",
    schema = "setup-shadow-v1",
    n = nrow(S),
    d = ncol(S),
    input_hash = fastkpc_hash_object(round(as.numeric(S), digits = 14)),
    basis_rank = setup$basis_rank,
    null_space_rank = setup$null_space_rank,
    effective_rank = setup$effective_rank,
    basis_dim = dim(setup$X),
    penalty_hash = fastkpc_hash_object(round(as.numeric(setup$penalty),
                                             digits = 14)),
    constraint_hash = fastkpc_hash_object(round(as.numeric(setup$constraint),
                                                digits = 14))
  ))
}

fastkpc_kpc_tprs_residual_cpp_candidate <- function(
    y, S, k = NA_integer_, tol = sqrt(.Machine$double.eps)) {
  input <- fastkpc_kpc_tprs_validate_input(y, S)
  setup <- kpc_tprs_residual_cpp_setup(
    input$S,
    as.integer(if (is.na(k)) 0L else k),
    as.numeric(tol)
  )
  fingerprint <- fastkpc_kpc_tprs_setup_fingerprint(input$S, setup)

  list(
    backend_family = "kpcTprsResidualCPP",
    mode = "shadow-candidate-setup-only",
    authoritative = FALSE,
    family = "gaussian_identity",
    smooth_class = "tp",
    smooth_geometry = "joint-isotropic",
    residual_output_only = TRUE,
    conditioning_size = as.integer(ncol(input$S)),
    X = setup$X,
    penalty = setup$penalty,
    constraint = setup$constraint,
    coefficients = numeric(),
    fitted = numeric(),
    residuals = numeric(),
    edf = NA_real_,
    selected_sp = NA_real_,
    score = NA_real_,
    basis_rank = as.integer(setup$basis_rank),
    null_space_rank = as.integer(setup$null_space_rank),
    effective_rank = as.integer(setup$effective_rank),
    setup_fingerprint = fingerprint,
    diagnostics = list(
      schema_version = "kpcTprsResidualCPP-shadow-v1",
      setup_only = TRUE,
      native_backend = setup$backend_family,
      radial_basis = setup$radial_basis,
      polynomial_basis = setup$polynomial_basis,
      k = as.integer(setup$k),
      tol = as.numeric(tol),
      does_not_drive_graph_decisions = TRUE
    )
  )
}

fastkpc_kpc_tprs_residual_cpp_shadow <- function(y, S, oracle, ...) {
  if (!is.function(oracle)) {
    stop("oracle must be a function", call. = FALSE)
  }
  oracle_result <- oracle(y, S, ...)
  candidate_result <- tryCatch(
    fastkpc_kpc_tprs_residual_cpp_candidate(y = y, S = S),
    error = function(e) {
      list(
        backend_family = "kpcTprsResidualCPP",
        mode = "shadow-candidate-failed-closed",
        authoritative = FALSE,
        residuals = numeric(),
        fitted = numeric(),
        coefficients = numeric(),
        edf = NA_real_,
        selected_sp = NA_real_,
        score = NA_real_,
        basis_rank = NA_integer_,
        null_space_rank = NA_integer_,
        setup_fingerprint = NA_character_,
        diagnostics = list(
          schema_version = "kpcTprsResidualCPP-shadow-v1",
          error = conditionMessage(e),
          does_not_drive_graph_decisions = TRUE
        )
      )
    }
  )

  list(
    oracle = oracle_result,
    candidate = candidate_result,
    oracle_authoritative = TRUE,
    p_used_source = "oracle",
    decision_source = "oracle",
    graph_decision_changed = FALSE,
    diagnostics = list(
      schema_version = "kpcTprsResidualCPP-shadow-v1",
      candidate_mode = candidate_result$mode,
      candidate_available = !identical(candidate_result$mode,
                                       "shadow-candidate-failed-closed")
    )
  )
}

fastkpc_kpc_tprs_formula <- function(S) {
  S <- as.matrix(S)
  if (ncol(S) == 1L) {
    stats::as.formula("y ~ s(s1, bs = 'tp', m = 2)")
  } else if (ncol(S) == 2L) {
    stats::as.formula("y ~ s(s1, s2, bs = 'tp', m = 2)")
  } else {
    fastkpc_kpc_tprs_stop("|S| = 1 or 2 is required")
  }
}

fastkpc_kpc_tprs_data_frame <- function(y, S) {
  S <- as.matrix(S)
  colnames(S) <- paste0("s", seq_len(ncol(S)))
  data.frame(y = as.numeric(y), S, check.names = FALSE)
}

fastkpc_kpc_tprs_mgcv_oracle_setup <- function(y, S, sp = 1,
                                               method = "GCV.Cp") {
  if (!requireNamespace("mgcv", quietly = TRUE)) {
    stop("mgcv is required for kpcTprsResidualCPP oracle setup",
         call. = FALSE)
  }
  input <- fastkpc_kpc_tprs_validate_input(y, S)
  data <- fastkpc_kpc_tprs_data_frame(input$y, input$S)
  formula <- fastkpc_kpc_tprs_formula(input$S)
  smooth_spec <- eval(formula[[3L]], envir = data,
                      enclos = asNamespace("mgcv"))
  smooth <- mgcv::smoothCon(
    smooth_spec,
    data = data,
    knots = NULL,
    absorb.cons = FALSE,
    scale.penalty = TRUE
  )[[1L]]
  fit <- mgcv::gam(
    formula = formula,
    data = data,
    family = stats::gaussian(),
    sp = sp,
    method = method,
    fit = TRUE
  )
  setup_fp <- fastkpc_hash_object(list(
    backend = "mgcv",
    schema = "kpcTprsResidualCPP-oracle-v1",
    mgcv_version = as.character(utils::packageVersion("mgcv")),
    n = nrow(input$S),
    d = ncol(input$S),
    sp = sp,
    X_hash = fastkpc_hash_object(round(as.numeric(stats::predict(fit, type = "lpmatrix")),
                                       digits = 14)),
    S_hash = fastkpc_hash_object(round(as.numeric(smooth$S[[1L]]), digits = 14)),
    C_hash = fastkpc_hash_object(round(as.numeric(smooth$C), digits = 14))
  ))
  list(
    backend_family = "mgcv",
    mode = "oracle-fixed-sp-setup",
    formula = formula,
    smooth = smooth,
    fit = fit,
    X = as.matrix(stats::predict(fit, type = "lpmatrix")),
    penalty = as.matrix(smooth$S[[1L]]),
    constraint = as.matrix(smooth$C),
    fitted = as.numeric(stats::fitted(fit)),
    residuals = as.numeric(stats::residuals(fit)),
    edf = if (!is.null(fit$edf)) sum(fit$edf) else NA_real_,
    selected_sp = if (!is.null(fit$sp) && length(fit$sp) > 0L) {
      fit$sp
    } else if (!is.null(fit$full.sp) && length(fit$full.sp) > 0L) {
      fit$full.sp
    } else {
      sp
    },
    basis_rank = ncol(smooth$X),
    null_space_rank = as.integer(smooth$null.space.dim),
    effective_rank = ncol(stats::predict(fit, type = "lpmatrix")),
    setup_fingerprint = setup_fp,
    raw = list(
      shift = smooth$shift,
      unique_locations = smooth$Xu,
      UZ = smooth$UZ,
      constraint = as.matrix(smooth$C),
      penalty = as.matrix(smooth$S[[1L]]),
      S_scale = smooth$S.scale
    ),
    diagnostics = list(
      mgcv_version = as.character(utils::packageVersion("mgcv")),
      rank = smooth$rank,
      bs_dim = smooth$bs.dim,
      df = smooth$df
    )
  )
}

fastkpc_kpc_tprs_projector <- function(X, tol = sqrt(.Machine$double.eps)) {
  X <- as.matrix(X)
  qrX <- qr(X, tol = tol)
  if (qrX$rank == 0L) return(matrix(0, nrow(X), nrow(X)))
  Q <- qr.Q(qrX)[, seq_len(qrX$rank), drop = FALSE]
  tcrossprod(Q)
}

fastkpc_kpc_tprs_projector_distance <- function(X_candidate, X_oracle,
                                                tol = sqrt(.Machine$double.eps)) {
  Pc <- fastkpc_kpc_tprs_projector(X_candidate, tol = tol)
  Po <- fastkpc_kpc_tprs_projector(X_oracle, tol = tol)
  denom <- sqrt(sum(Po^2))
  if (denom == 0) return(sqrt(sum((Pc - Po)^2)))
  sqrt(sum((Pc - Po)^2)) / denom
}

fastkpc_kpc_tprs_penalty_spectrum <- function(P, tol = sqrt(.Machine$double.eps)) {
  values <- eigen((as.matrix(P) + t(as.matrix(P))) / 2,
                  symmetric = TRUE, only.values = TRUE)$values
  values <- sort(pmax(as.numeric(values), 0), decreasing = TRUE)
  values <- values[values > tol * max(1, max(values))]
  if (length(values) == 0L) return(numeric())
  values / max(values)
}

fastkpc_kpc_tprs_spectrum_distance <- function(P_candidate, P_oracle,
                                               tol = sqrt(.Machine$double.eps)) {
  sc <- fastkpc_kpc_tprs_penalty_spectrum(P_candidate, tol = tol)
  so <- fastkpc_kpc_tprs_penalty_spectrum(P_oracle, tol = tol)
  m <- max(length(sc), length(so))
  if (m == 0L) return(0)
  length(sc) <- m
  length(so) <- m
  sc[is.na(sc)] <- 0
  so[is.na(so)] <- 0
  denom <- sqrt(sum(so^2))
  if (denom == 0) return(sqrt(sum((sc - so)^2)))
  sqrt(sum((sc - so)^2)) / denom
}

fastkpc_kpc_tprs_constraint_rank <- function(C, tol = sqrt(.Machine$double.eps)) {
  C <- as.matrix(C)
  if (length(C) == 0L || nrow(C) == 0L) return(0L)
  qr(C, tol = tol)$rank
}

fastkpc_kpc_tprs_transform_scenarios <- function(S) {
  S <- as.matrix(S)
  n <- nrow(S)
  out <- list(
    identity = S,
    translation = sweep(S, 2, seq_len(ncol(S)) * 1.75, "+"),
    scale = S * 2.5,
    duplicate_rows = rbind(S, S[seq_len(min(5L, n)), , drop = FALSE]),
    row_permutation = S[rev(seq_len(n)), , drop = FALSE]
  )
  if (ncol(S) == 2L) {
    theta <- pi / 7
    rotation <- matrix(c(cos(theta), -sin(theta), sin(theta), cos(theta)), 2, 2)
    out$rotation <- S %*% rotation
  }
  out
}

fastkpc_kpc_tprs_geometry_parity <- function(
    y, S, tol = sqrt(.Machine$double.eps)) {
  input <- fastkpc_kpc_tprs_validate_input(y, S)
  candidate <- kpc_tprs_residual_cpp_setup(input$S, tol = tol)
  oracle <- fastkpc_kpc_tprs_mgcv_oracle_setup(input$y, input$S, sp = 1)
  projector_distance <- fastkpc_kpc_tprs_projector_distance(
    candidate$X, oracle$smooth$X, tol = tol
  )
  penalty_spectrum_distance <- fastkpc_kpc_tprs_spectrum_distance(
    candidate$penalty, oracle$penalty, tol = tol
  )

  transforms <- fastkpc_kpc_tprs_transform_scenarios(input$S)
  base_setup <- kpc_tprs_residual_cpp_setup(input$S, tol = tol)
  transform_rows <- lapply(names(transforms), function(name) {
    transformed_setup <- kpc_tprs_residual_cpp_setup(transforms[[name]], tol = tol)
    rows <- min(nrow(base_setup$X), nrow(transformed_setup$X))
    data.frame(
      transform = name,
      candidate_projector_distance = fastkpc_kpc_tprs_projector_distance(
        transformed_setup$X[seq_len(rows), , drop = FALSE],
        base_setup$X[seq_len(rows), , drop = FALSE],
        tol = tol
      ),
      candidate_penalty_spectrum_distance =
        fastkpc_kpc_tprs_spectrum_distance(transformed_setup$penalty,
                                           base_setup$penalty,
                                           tol = tol),
      stringsAsFactors = FALSE
    )
  })

  list(
    backend_family = "kpcTprsResidualCPP",
    mode = "geometry-parity-shadow",
    authoritative = FALSE,
    conditioning_size = as.integer(ncol(input$S)),
    null_space_rank_candidate = as.integer(candidate$null_space_rank),
    null_space_rank_oracle = as.integer(oracle$null_space_rank),
    null_space_rank_match =
      identical(as.integer(candidate$null_space_rank),
                as.integer(oracle$null_space_rank)),
    effective_rank_candidate = as.integer(candidate$effective_rank),
    effective_rank_oracle = as.integer(oracle$basis_rank - 1L),
    effective_rank_match =
      identical(as.integer(candidate$effective_rank),
                as.integer(oracle$basis_rank - 1L)),
    constraint_rank_candidate =
      fastkpc_kpc_tprs_constraint_rank(candidate$constraint, tol = tol),
    constraint_rank_oracle =
      fastkpc_kpc_tprs_constraint_rank(oracle$constraint, tol = tol),
    projector_distance = projector_distance,
    penalty_spectrum_distance = penalty_spectrum_distance,
    transform_metrics = do.call(rbind, transform_rows),
    oracle_setup_fingerprint = oracle$setup_fingerprint,
    candidate_setup_fingerprint =
      fastkpc_kpc_tprs_setup_fingerprint(input$S, candidate),
    diagnostics = list(
      schema_version = "kpcTprsResidualCPP-geometry-parity-v1",
      does_not_drive_graph_decisions = TRUE
    )
  )
}

fastkpc_kpc_tprs_fixed_sp_solve_candidate <- function(
    y, S, sp, k = NA_integer_, tol = sqrt(.Machine$double.eps)) {
  input <- fastkpc_kpc_tprs_validate_input(y, S)
  setup <- kpc_tprs_residual_cpp_setup(input$S, k = k, tol = tol)
  fastkpc_kpc_tprs_solve_candidate_setup(input$y, setup, sp = sp, tol = tol)
}

fastkpc_kpc_tprs_solve_candidate_setup <- function(
    y, setup, sp, penalty = setup$absorbed$penalty,
    include_intercept = FALSE,
    tol = sqrt(.Machine$double.eps)) {
  X_smooth <- as.matrix(setup$absorbed$X)
  P_smooth <- as.matrix(penalty)
  if (isTRUE(include_intercept)) {
    X <- cbind(`(Intercept)` = 1, X_smooth)
    P <- matrix(0, nrow = ncol(X), ncol = ncol(X))
    P[-1L, -1L] <- P_smooth
  } else {
    X <- X_smooth
    P <- P_smooth
  }
  A <- crossprod(X) + as.numeric(sp) * P
  b <- crossprod(X, as.numeric(y))
  theta <- as.numeric(qr.solve(A, b, tol = tol))
  fitted <- as.numeric(X %*% theta)
  residuals <- as.numeric(y - fitted)
  A_inv <- tryCatch(
    solve(A),
    error = function(e) qr.solve(A, diag(ncol(A)), tol = tol)
  )
  edf <- sum(diag(X %*% A_inv %*% t(X)))
  list(
    backend_family = "kpcTprsResidualCPP",
    mode = "fixed-sp-candidate-solve",
    authoritative = FALSE,
    coefficients = theta,
    fitted = fitted,
    residuals = residuals,
    edf = as.numeric(edf),
    selected_sp = as.numeric(sp),
    score = NA_real_,
    basis_rank = as.integer(setup$basis_rank %||% ncol(X_smooth)),
    null_space_rank = as.integer(setup$null_space_rank %||% NA_integer_),
    setup = setup,
    setup_fingerprint = setup$schema_version %||% "kpcTprsResidualCPP",
    diagnostics = list(
      solver = "absorbed-penalized-least-squares",
      include_intercept = isTRUE(include_intercept),
      does_not_drive_graph_decisions = TRUE
    )
  )
}

fastkpc_kpc_tprs_mapped_sp_scale <- function(
    setup, oracle_absorbed,
    tol = sqrt(.Machine$double.eps)) {
  generalized_spectrum <- fastkpc_kpc_tprs_generalized_spectrum_compare(
    setup$absorbed$X,
    setup$absorbed$penalty,
    oracle_absorbed$X,
    oracle_absorbed$S[[1L]],
    tol = tol
  )
  offset <- as.numeric(generalized_spectrum$log_spectrum_scale_offset)
  scale <- if (is.finite(offset)) exp(-offset) else NA_real_
  list(
    source_sp_semantics = "mgcv-smoothCon-scaled-penalty",
    target_lambda_semantics = "kpc-canonical-penalty",
    scale = as.numeric(scale),
    log_spectrum_scale_offset = offset,
    log_spectrum_shape_rmse =
      as.numeric(generalized_spectrum$log_spectrum_shape_rmse),
    generalized_penalty_rank =
      as.integer(generalized_spectrum$generalized_penalty_rank)
  )
}

fastkpc_kpc_tprs_map_mgcv_sp_to_canonical <- function(
    sp, setup, oracle_absorbed = NULL,
    tol = sqrt(.Machine$double.eps)) {
  if (is.null(oracle_absorbed)) {
    stop("oracle_absorbed is required for mapped-sp parity", call. = FALSE)
  }
  mapping <- fastkpc_kpc_tprs_mapped_sp_scale(
    setup = setup,
    oracle_absorbed = oracle_absorbed,
    tol = tol
  )
  if (!is.finite(mapping$scale) || mapping$scale <= 0) {
    stop("could not derive a finite positive mapped-sp scale", call. = FALSE)
  }
  mapping$source_sp <- as.numeric(sp)
  mapping$canonical_lambda <- as.numeric(sp) * mapping$scale
  mapping
}

fastkpc_kpc_tprs_mapped_sp_solve_candidate <- function(
    y, S, sp, k = NA_integer_, oracle_absorbed = NULL,
    tol = sqrt(.Machine$double.eps)) {
  input <- fastkpc_kpc_tprs_validate_input(y, S)
  setup <- kpc_tprs_residual_cpp_setup(input$S, k = k, tol = tol)
  if (is.null(oracle_absorbed)) {
    oracle_absorbed <- fastkpc_kpc_tprs_absorbed_oracle_setup(input$y, input$S)
  }
  mapped <- fastkpc_kpc_tprs_map_mgcv_sp_to_canonical(
    sp = sp,
    setup = setup,
    oracle_absorbed = oracle_absorbed,
    tol = tol
  )
  solved <- fastkpc_kpc_tprs_solve_candidate_setup(
    input$y,
    setup,
    sp = mapped$canonical_lambda,
    include_intercept = TRUE,
    tol = tol
  )
  solved$mode <- "mapped-sp-candidate-solve"
  solved$source_sp <- as.numeric(sp)
  solved$selected_sp <- as.numeric(mapped$canonical_lambda)
  solved$mapped_sp <- mapped
  solved$diagnostics$solver <- "intercept-plus-absorbed-smooth-pls"
  solved$diagnostics$sp_semantics <- mapped$target_lambda_semantics
  solved
}

fastkpc_kpc_tprs_oracle_absorbed_basis <- function(oracle) {
  as.matrix(oracle$X[, -1L, drop = FALSE])
}

fastkpc_kpc_tprs_candidate_uz_design <- function(setup) {
  raw <- setup$raw %||% list()
  UZ <- raw$UZ
  unique_locations <- raw$unique_locations
  if (is.null(UZ) || is.null(unique_locations)) return(NULL)
  UZ <- as.matrix(UZ)
  unique_locations <- as.matrix(unique_locations)
  if (ncol(unique_locations) != 1L) return(NULL)
  n_unique <- nrow(unique_locations)
  if (nrow(UZ) < n_unique + 2L) return(NULL)
  b <- cbind(
    fastkpc_kpc_tprs_eta_1d(unique_locations[, 1L],
                            unique_locations[, 1L]),
    1,
    unique_locations[, 1L]
  )
  b %*% UZ
}

fastkpc_kpc_tprs_eta_1d <- function(x, knots) {
  outer(as.numeric(x), as.numeric(knots),
        function(a, b) abs(a - b)^3 / 12)
}

fastkpc_kpc_tprs_oracle_uz_design <- function(oracle) {
  raw <- oracle$raw %||% list()
  UZ <- raw$UZ
  unique_locations <- raw$unique_locations
  if (is.null(UZ) || is.null(unique_locations)) return(NULL)
  UZ <- as.matrix(UZ)
  unique_locations <- as.matrix(unique_locations)
  if (ncol(unique_locations) != 1L) return(NULL)
  b <- cbind(
    fastkpc_kpc_tprs_eta_1d(unique_locations[, 1L],
                            unique_locations[, 1L]),
    1,
    unique_locations[, 1L]
  )
  b %*% UZ
}

fastkpc_kpc_tprs_diagnostic_or_na <- function(x, name) {
  if (is.null(x) || is.null(x[[name]])) NA_real_ else x[[name]]
}

fastkpc_kpc_tprs_absorbed_oracle_setup <- function(y, S) {
  if (!requireNamespace("mgcv", quietly = TRUE)) {
    stop("mgcv is required for kpcTprsResidualCPP oracle setup",
         call. = FALSE)
  }
  input <- fastkpc_kpc_tprs_validate_input(y, S)
  data <- fastkpc_kpc_tprs_data_frame(input$y, input$S)
  formula <- fastkpc_kpc_tprs_formula(input$S)
  smooth_spec <- eval(formula[[3L]], envir = data,
                      enclos = asNamespace("mgcv"))
  mgcv::smoothCon(
    smooth_spec,
    data = data,
    knots = NULL,
    absorb.cons = TRUE,
    scale.penalty = TRUE
  )[[1L]]
}

fastkpc_kpc_tprs_mgcv_gcv_oracle_fit <- function(y, S) {
  if (!requireNamespace("mgcv", quietly = TRUE)) {
    stop("mgcv is required for kpcTprsResidualCPP oracle setup",
         call. = FALSE)
  }
  input <- fastkpc_kpc_tprs_validate_input(y, S)
  data <- fastkpc_kpc_tprs_data_frame(input$y, input$S)
  formula <- fastkpc_kpc_tprs_formula(input$S)
  fit <- mgcv::gam(
    formula = formula,
    data = data,
    family = stats::gaussian(),
    method = "GCV.Cp",
    fit = TRUE
  )
  list(
    residuals = as.numeric(stats::residuals(fit)),
    fitted = as.numeric(stats::fitted(fit)),
    selected_sp = if (!is.null(fit$sp) && length(fit$sp) > 0L) {
      as.numeric(fit$sp)
    } else if (!is.null(fit$full.sp) && length(fit$full.sp) > 0L) {
      as.numeric(fit$full.sp)
    } else {
      NA_real_
    },
    score = as.numeric(fit$gcv.ubre %||% NA_real_),
    edf = if (!is.null(fit$edf)) sum(fit$edf) else NA_real_,
    fit = fit
  )
}

fastkpc_kpc_tprs_rel_fro <- function(candidate, oracle) {
  candidate <- as.matrix(candidate)
  oracle <- as.matrix(oracle)
  denom <- sqrt(sum(oracle^2))
  if (denom == 0) return(sqrt(sum((candidate - oracle)^2)))
  sqrt(sum((candidate - oracle)^2)) / denom
}

fastkpc_kpc_tprs_candidate_uz_penalty_identity <- function(setup) {
  E <- as.matrix(setup$raw$radial_kernel_block)
  UZ <- as.matrix(setup$raw$UZ)
  S <- as.matrix(setup$penalty)
  n_unique <- nrow(E)
  UZ_delta <- UZ[seq_len(n_unique), , drop = FALSE]
  fastkpc_kpc_tprs_rel_fro(t(UZ_delta) %*% E %*% UZ_delta, S)
}

fastkpc_kpc_tprs_oracle_uz_penalty_identity <- function(oracle_raw) {
  Xu <- as.matrix(oracle_raw$unique_locations)
  UZ <- as.matrix(oracle_raw$UZ)
  S <- as.matrix(oracle_raw$penalty)
  S_scale <- as.numeric(oracle_raw$S_scale %||% 1)
  E <- fastkpc_kpc_tprs_eta_1d(Xu[, 1L], Xu[, 1L])
  UZ_delta <- UZ[seq_len(nrow(Xu)), , drop = FALSE]
  fastkpc_kpc_tprs_rel_fro(t(UZ_delta) %*% E %*% UZ_delta, S * S_scale)
}

fastkpc_kpc_tprs_candidate_stage_identities <- function(setup) {
  raw <- setup$raw
  S_eigen <- as.matrix(raw$S_eigen)
  Z_tps <- as.matrix(raw$tps_null_space)
  S_tps <- as.matrix(raw$S_tps_constrained)
  S_pre <- as.matrix(raw$S_pre_rms)
  W <- as.matrix(raw$W_rms)
  S_rms <- as.matrix(raw$S_rms)
  Q <- as.matrix(setup$absorbed$Z)
  S_abs <- as.matrix(setup$absorbed$penalty)
  list(
    uz_penalty_identity_rel_error =
      fastkpc_kpc_tprs_candidate_uz_penalty_identity(setup),
    pre_rms_congruence_rel_error =
      fastkpc_kpc_tprs_rel_fro(
        S_pre[seq_len(nrow(S_tps)), seq_len(ncol(S_tps)), drop = FALSE],
        S_tps
      ),
    post_rms_congruence_rel_error =
      fastkpc_kpc_tprs_rel_fro(t(W) %*% S_pre %*% W, S_rms),
    ident_absorb_congruence_rel_error =
      fastkpc_kpc_tprs_rel_fro(t(Q) %*% S_rms %*% Q, S_abs),
    tps_from_eigen_rel_error =
      fastkpc_kpc_tprs_rel_fro(t(Z_tps) %*% S_eigen %*% Z_tps, S_tps)
  )
}

fastkpc_kpc_tprs_whitened_penalty_eigenvalues <- function(
    X, S, tol = sqrt(.Machine$double.eps)) {
  X <- as.matrix(X)
  S <- as.matrix(S)
  G <- crossprod(X)
  eigG <- eigen((G + t(G)) / 2, symmetric = TRUE)
  keep <- eigG$values > tol * max(1, max(eigG$values))
  if (!any(keep)) return(numeric())
  Q <- eigG$vectors[, keep, drop = FALSE]
  values <- eigG$values[keep]
  G_inv_half <- Q %*% diag(1 / sqrt(values), nrow = length(values)) %*% t(Q)
  K <- G_inv_half %*% S %*% G_inv_half
  eigK <- eigen((K + t(K)) / 2, symmetric = TRUE, only.values = TRUE)$values
  eigK <- sort(as.numeric(eigK[eigK > tol * max(1, max(abs(eigK)))]),
               decreasing = TRUE)
  eigK
}

fastkpc_kpc_tprs_generalized_spectrum_compare <- function(
    X_candidate, S_candidate, X_oracle, S_oracle,
    tol = sqrt(.Machine$double.eps)) {
  candidate <- fastkpc_kpc_tprs_whitened_penalty_eigenvalues(
    X_candidate, S_candidate, tol = tol)
  oracle <- fastkpc_kpc_tprs_whitened_penalty_eigenvalues(
    X_oracle, S_oracle, tol = tol)
  m <- min(length(candidate), length(oracle))
  if (m == 0L) {
    offset <- NA_real_
    rmse <- NA_real_
  } else {
    delta <- log(candidate[seq_len(m)]) - log(oracle[seq_len(m)])
    offset <- stats::median(delta)
    rmse <- sqrt(mean((delta - offset)^2))
  }
  list(
    generalized_penalty_rank = as.integer(length(candidate)),
    generalized_eigenvalues_positive = list(
      candidate = candidate,
      oracle = oracle
    ),
    log_spectrum_scale_offset = as.numeric(offset),
    log_spectrum_shape_rmse = as.numeric(rmse)
  )
}

fastkpc_kpc_tprs_smoother_matrix <- function(X, S, lambda,
                                             tol = sqrt(.Machine$double.eps)) {
  X <- as.matrix(X)
  S <- as.matrix(S)
  A <- crossprod(X) + as.numeric(lambda) * S
  A_inv <- tryCatch(
    solve(A),
    error = function(e) qr.solve(A, diag(ncol(A)), tol = tol)
  )
  X %*% A_inv %*% t(X)
}

fastkpc_kpc_tprs_smoother_rel_distance <- function(
    X_candidate, S_candidate, X_oracle, S_oracle,
    lambda_oracle, scale, tol = sqrt(.Machine$double.eps)) {
  H_candidate <- fastkpc_kpc_tprs_smoother_matrix(
    X_candidate, S_candidate, lambda_oracle * scale, tol = tol)
  H_oracle <- fastkpc_kpc_tprs_smoother_matrix(
    X_oracle, S_oracle, lambda_oracle, tol = tol)
  fastkpc_kpc_tprs_rel_fro(H_candidate, H_oracle)
}

fastkpc_kpc_tprs_smoother_comparison <- function(
    X_candidate, S_candidate, X_oracle, S_oracle, lambda_values,
    global_scale = 1,
    tol = sqrt(.Machine$double.eps)) {
  local_grid <- exp(seq(log(1e-6), log(1e6), length.out = 49L))
  rows <- lapply(as.numeric(lambda_values), function(lambda) {
    distances <- vapply(local_grid, function(scale) {
      fastkpc_kpc_tprs_smoother_rel_distance(
        X_candidate, S_candidate, X_oracle, S_oracle,
        lambda_oracle = lambda, scale = scale, tol = tol)
    }, numeric(1L))
    best <- which.min(distances)
    data.frame(
      lambda = lambda,
      best_scale = local_grid[[best]],
      smoother_rel_frobenius = distances[[best]],
      global_scale = global_scale,
      global_scale_smoother_rel_frobenius =
        fastkpc_kpc_tprs_smoother_rel_distance(
          X_candidate, S_candidate, X_oracle, S_oracle,
          lambda_oracle = lambda, scale = global_scale, tol = tol),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

fastkpc_kpc_tprs_classify_penalty_geometry <- function(
    candidate_stage_identities, oracle_stage_identities,
    generalized_spectrum, smoother_comparison,
    identity_tol = 1e-8,
    spectrum_shape_tol = 1e-4,
    smoother_tol = 1e-4) {
  if (candidate_stage_identities$uz_penalty_identity_rel_error > identity_tol ||
      candidate_stage_identities$post_rms_congruence_rel_error > identity_tol ||
      candidate_stage_identities$ident_absorb_congruence_rel_error > identity_tol) {
    return("penalty-assembly")
  }
  if (oracle_stage_identities$uz_penalty_identity_rel_error > identity_tol) {
    return("stage-mismatch")
  }
  if (is.finite(generalized_spectrum$log_spectrum_shape_rmse) &&
      generalized_spectrum$log_spectrum_shape_rmse > spectrum_shape_tol) {
    return("penalty-shape")
  }
  global_dist <- smoother_comparison$global_scale_smoother_rel_frobenius
  if (all(is.finite(global_dist)) && max(global_dist) <= smoother_tol) {
    return("penalty-scale")
  }
  if (all(is.finite(smoother_comparison$smoother_rel_frobenius)) &&
      max(smoother_comparison$smoother_rel_frobenius) <= smoother_tol) {
    return("metric-false-positive")
  }
  "unclassified"
}

fastkpc_kpc_tprs_penalty_geometry_isolation <- function(
    y, S, lambda_values = c(1e-6, 1e-3, 1, 1e3, 1e6),
    tol = sqrt(.Machine$double.eps)) {
  input <- fastkpc_kpc_tprs_validate_input(y, S)
  if (ncol(input$S) != 1L) {
    fastkpc_kpc_tprs_stop("Phase 1e penalty geometry isolation is 1D only")
  }
  candidate_setup <- kpc_tprs_residual_cpp_setup(input$S, tol = tol)
  oracle_raw <- fastkpc_kpc_tprs_mgcv_oracle_setup(input$y, input$S, sp = 1)
  oracle_absorbed <- fastkpc_kpc_tprs_absorbed_oracle_setup(input$y, input$S)

  candidate_stage_identities <-
    fastkpc_kpc_tprs_candidate_stage_identities(candidate_setup)
  oracle_stage_identities <- list(
    uz_penalty_identity_rel_error =
      fastkpc_kpc_tprs_oracle_uz_penalty_identity(oracle_raw$raw)
  )
  generalized_spectrum <- fastkpc_kpc_tprs_generalized_spectrum_compare(
    candidate_setup$absorbed$X,
    candidate_setup$absorbed$penalty,
    oracle_absorbed$X,
    oracle_absorbed$S[[1L]],
    tol = tol
  )
  global_scale <- if (is.finite(generalized_spectrum$log_spectrum_scale_offset)) {
    exp(-generalized_spectrum$log_spectrum_scale_offset)
  } else {
    1
  }
  smoother_comparison <- fastkpc_kpc_tprs_smoother_comparison(
    candidate_setup$absorbed$X,
    candidate_setup$absorbed$penalty,
    oracle_absorbed$X,
    oracle_absorbed$S[[1L]],
    lambda_values = lambda_values,
    global_scale = global_scale,
    tol = tol
  )
  classification <- fastkpc_kpc_tprs_classify_penalty_geometry(
    candidate_stage_identities = candidate_stage_identities,
    oracle_stage_identities = oracle_stage_identities,
    generalized_spectrum = generalized_spectrum,
    smoother_comparison = smoother_comparison
  )

  list(
    backend_family = "kpcTprsResidualCPP",
    mode = "penalty-geometry-isolation",
    authoritative = FALSE,
    conditioning_size = as.integer(ncol(input$S)),
    signed_eigenvalues = as.numeric(candidate_setup$raw$selected_eigenvalues),
    candidate_stage_identities = candidate_stage_identities,
    oracle_stage_identities = oracle_stage_identities,
    generalized_spectrum = generalized_spectrum,
    smoother_comparison = smoother_comparison,
    classification = classification,
    diagnostics = list(
      schema_version = "kpcTprsResidualCPP-penalty-geometry-v1",
      candidate_basis_rank = candidate_setup$basis_rank,
      oracle_absorbed_rank = ncol(oracle_absorbed$X),
      oracle_s_scale = oracle_raw$raw$S_scale,
      does_not_drive_graph_decisions = TRUE
    )
  )
}

fastkpc_kpc_tprs_align_rows <- function(candidate, oracle) {
  n <- min(nrow(candidate), nrow(oracle))
  list(
    candidate = candidate[seq_len(n), , drop = FALSE],
    oracle = oracle[seq_len(n), , drop = FALSE]
  )
}

fastkpc_kpc_tprs_basis_change <- function(X_from, X_to,
                                          tol = sqrt(.Machine$double.eps)) {
  qr.solve(as.matrix(X_from), as.matrix(X_to), tol = tol)
}

fastkpc_kpc_tprs_penalty_shape_scale <- function(S_candidate_mapped,
                                                 S_oracle) {
  Sc <- as.matrix(S_candidate_mapped)
  So <- as.matrix(S_oracle)
  n <- min(nrow(Sc), nrow(So))
  m <- min(ncol(Sc), ncol(So))
  Sc <- Sc[seq_len(n), seq_len(m), drop = FALSE]
  So <- So[seq_len(n), seq_len(m), drop = FALSE]
  denom <- sum(Sc * Sc)
  scale <- if (denom == 0) NA_real_ else sum(So * Sc) / denom
  shape <- if (!is.finite(scale)) {
    Inf
  } else {
    sqrt(sum((So - scale * Sc)^2)) / max(sqrt(sum(So^2)), .Machine$double.eps)
  }
  list(shape_distance = as.numeric(shape), scale_ratio = as.numeric(scale))
}

fastkpc_kpc_tprs_compare_fit <- function(candidate, oracle) {
  data.frame(
    candidate_sp = as.numeric(candidate$selected_sp),
    oracle_sp = as.numeric(oracle$selected_sp[1L]),
    residual_rel_l2 = fastkpc_kpc_tprs_rel_l2(candidate$residuals,
                                              oracle$residuals),
    fitted_rel_l2 = fastkpc_kpc_tprs_rel_l2(candidate$fitted, oracle$fitted),
    edf_candidate = as.numeric(candidate$edf),
    edf_oracle = as.numeric(oracle$edf),
    edf_abs_diff = abs(as.numeric(candidate$edf) - as.numeric(oracle$edf)),
    stringsAsFactors = FALSE
  )
}

fastkpc_kpc_tprs_edf_matched_candidate <- function(
    y, setup, oracle_edf, grid,
    penalty = setup$absorbed$penalty,
    tol = sqrt(.Machine$double.eps)) {
  fits <- lapply(as.numeric(grid), function(sp) {
    tryCatch(
      fastkpc_kpc_tprs_solve_candidate_setup(y, setup, sp = sp,
                                             penalty = penalty, tol = tol),
      error = function(e) NULL
    )
  })
  fits <- Filter(Negate(is.null), fits)
  if (length(fits) == 0L) {
    stop("all EDF-matching candidate solves failed", call. = FALSE)
  }
  edf <- vapply(fits, `[[`, numeric(1L), "edf")
  best <- which.min(abs(edf - oracle_edf))
  fits[[best]]
}

fastkpc_kpc_tprs_classify_drift <- function(raw_projector_distance,
                                            absorbed_projector_distance,
                                            penalty_shape_distance,
                                            same_raw_sp,
                                            scale_corrected,
                                            edf_matched,
                                            projector_tol = 1e-4,
                                            shape_tol = 1e-4,
                                            residual_tol = 1e-4) {
  if (raw_projector_distance > projector_tol) return("function-space")
  if (absorbed_projector_distance > projector_tol) return("constraint-intercept")
  if (penalty_shape_distance > shape_tol) return("penalty-shape")
  if (edf_matched$residual_rel_l2 <= residual_tol &&
      same_raw_sp$residual_rel_l2 > residual_tol) {
    return("penalty-scale")
  }
  if (scale_corrected$residual_rel_l2 < same_raw_sp$residual_rel_l2) {
    return("penalty-scale")
  }
  "unclassified"
}

fastkpc_kpc_tprs_fixed_sp_drift_isolation <- function(
    y, S, sp = 1,
    edf_search_grid = exp(seq(log(1e-6), log(1e6), length.out = 61L)),
    tol = sqrt(.Machine$double.eps)) {
  input <- fastkpc_kpc_tprs_validate_input(y, S)
  candidate_setup <- kpc_tprs_residual_cpp_setup(input$S, tol = tol)
  oracle <- fastkpc_kpc_tprs_mgcv_oracle_setup(input$y, input$S, sp = sp)

  candidate_uz <- fastkpc_kpc_tprs_candidate_uz_design(candidate_setup)
  oracle_uz <- fastkpc_kpc_tprs_oracle_uz_design(oracle)
  uz_projector_distance <- if (is.null(candidate_uz) || is.null(oracle_uz)) {
    NA_real_
  } else {
    fastkpc_kpc_tprs_projector_distance(candidate_uz, oracle_uz, tol = tol)
  }
  raw_projector_distance <- fastkpc_kpc_tprs_projector_distance(
    candidate_setup$X, oracle$smooth$X, tol = tol
  )
  candidate_absorbed_X <- as.matrix(candidate_setup$absorbed$X)
  oracle_absorbed_X <- fastkpc_kpc_tprs_oracle_absorbed_basis(oracle)
  absorbed_projector_distance <- fastkpc_kpc_tprs_projector_distance(
    candidate_absorbed_X, oracle_absorbed_X, tol = tol
  )

  aligned <- fastkpc_kpc_tprs_align_rows(candidate_absorbed_X, oracle_absorbed_X)
  A <- fastkpc_kpc_tprs_basis_change(aligned$candidate, aligned$oracle, tol = tol)
  S_candidate_mapped <- t(A) %*% as.matrix(candidate_setup$absorbed$penalty) %*% A
  shape_scale <- fastkpc_kpc_tprs_penalty_shape_scale(
    S_candidate_mapped, oracle$penalty
  )

  candidate_raw <- fastkpc_kpc_tprs_solve_candidate_setup(
    input$y, candidate_setup, sp = sp, tol = tol
  )
  oracle_raw <- fastkpc_kpc_tprs_mgcv_oracle_setup(input$y, input$S, sp = sp)
  same_raw_sp <- fastkpc_kpc_tprs_compare_fit(candidate_raw, oracle_raw)

  corrected_sp <- sp * shape_scale$scale_ratio
  if (!is.finite(corrected_sp) || corrected_sp <= 0) corrected_sp <- sp
  candidate_scale <- fastkpc_kpc_tprs_solve_candidate_setup(
    input$y, candidate_setup, sp = corrected_sp, tol = tol
  )
  scale_corrected <- fastkpc_kpc_tprs_compare_fit(candidate_scale, oracle_raw)

  candidate_edf <- fastkpc_kpc_tprs_edf_matched_candidate(
    input$y, candidate_setup, oracle_edf = oracle_raw$edf,
    grid = edf_search_grid, tol = tol
  )
  edf_matched <- fastkpc_kpc_tprs_compare_fit(candidate_edf, oracle_raw)

  mapped_classification <- fastkpc_kpc_tprs_classify_drift(
    raw_projector_distance = raw_projector_distance,
    absorbed_projector_distance = absorbed_projector_distance,
    penalty_shape_distance = shape_scale$shape_distance,
    same_raw_sp = same_raw_sp,
    scale_corrected = scale_corrected,
    edf_matched = edf_matched
  )
  penalty_geometry <- NULL
  classification <- mapped_classification
  if (ncol(input$S) == 1L) {
    penalty_geometry <- tryCatch(
      fastkpc_kpc_tprs_penalty_geometry_isolation(
        y = input$y,
        S = input$S,
        lambda_values = c(1e-6, 1e-3, 1, 1e3, 1e6),
        tol = tol
      ),
      error = function(e) {
        list(
          classification = "unclassified",
          diagnostics = list(failure_reason = conditionMessage(e))
        )
      }
    )
    if (penalty_geometry$classification %in%
        c("penalty-scale", "metric-false-positive")) {
      classification <- penalty_geometry$classification
    }
  }

  list(
    backend_family = "kpcTprsResidualCPP",
    mode = "fixed-sp-drift-isolation",
    authoritative = FALSE,
    conditioning_size = as.integer(ncol(input$S)),
    uz_projector_distance = uz_projector_distance,
    raw_projector_distance = raw_projector_distance,
    absorbed_projector_distance = absorbed_projector_distance,
    penalty_shape_distance = shape_scale$shape_distance,
    penalty_scale_ratio = shape_scale$scale_ratio,
    same_raw_sp = same_raw_sp,
    scale_corrected = scale_corrected,
    edf_matched = edf_matched,
    mapped_penalty_classification = mapped_classification,
    penalty_geometry = penalty_geometry,
    classification = classification,
    diagnostics = list(
      schema_version = "kpcTprsResidualCPP-drift-isolation-v1",
      oracle_setup_fingerprint = oracle$setup_fingerprint,
      candidate_basis_rank = candidate_setup$basis_rank,
      oracle_basis_rank = oracle$basis_rank,
      selected_eigenvalues =
        fastkpc_kpc_tprs_diagnostic_or_na(candidate_setup$diagnostics,
                                          "selected_eigenvalues"),
      truncation_eigengap =
        fastkpc_kpc_tprs_diagnostic_or_na(candidate_setup$diagnostics,
                                          "truncation_eigengap"),
      rank_T =
        fastkpc_kpc_tprs_diagnostic_or_na(candidate_setup$diagnostics,
                                          "rank_T"),
      rank_TU =
        fastkpc_kpc_tprs_diagnostic_or_na(candidate_setup$diagnostics,
                                          "rank_TU"),
      Z_orthogonality_error =
        fastkpc_kpc_tprs_diagnostic_or_na(candidate_setup$diagnostics,
                                          "Z_orthogonality_error"),
      TPS_constraint_error =
        fastkpc_kpc_tprs_diagnostic_or_na(candidate_setup$diagnostics,
                                          "TPS_constraint_error"),
      pre_rms_column_norms =
        fastkpc_kpc_tprs_diagnostic_or_na(candidate_setup$diagnostics,
                                          "pre_rms_column_norms"),
      post_rms_column_norms =
        fastkpc_kpc_tprs_diagnostic_or_na(candidate_setup$diagnostics,
                                          "post_rms_column_norms"),
      does_not_drive_graph_decisions = TRUE
    )
  )
}

fastkpc_kpc_tprs_1d_first_drift_scenarios <- function(seed = 1L, n = 100L) {
  set.seed(seed)
  x <- as.numeric(scale(seq(-2, 2, length.out = n)))
  y <- sin(x) + stats::rnorm(n, sd = 0.03)
  list(
    one_d_standard = list(y = y, S = matrix(x, ncol = 1)),
    one_d_translation = list(y = y, S = matrix(x + 3.5, ncol = 1)),
    one_d_rescale = list(y = y, S = matrix(2.25 * x, ncol = 1)),
    one_d_duplicates = list(
      y = c(y, y[seq_len(8L)]),
      S = matrix(c(x, x[seq_len(8L)]), ncol = 1)
    )
  )
}

fastkpc_run_kpc_tprs_drift_isolation_campaign <- function(
    scenarios, sp = 1,
    edf_search_grid = exp(seq(log(1e-6), log(1e6), length.out = 61L))) {
  rows <- list()
  details <- list()
  for (name in names(scenarios)) {
    scenario <- scenarios[[name]]
    result <- tryCatch(
      fastkpc_kpc_tprs_fixed_sp_drift_isolation(
        y = scenario$y,
        S = scenario$S,
        sp = sp,
        edf_search_grid = edf_search_grid
      ),
      error = function(e) {
        list(
          backend_family = "kpcTprsResidualCPP",
          mode = "fixed-sp-drift-isolation-failed-closed",
          authoritative = FALSE,
          conditioning_size = as.integer(ncol(as.matrix(scenario$S))),
          uz_projector_distance = NA_real_,
          raw_projector_distance = NA_real_,
          absorbed_projector_distance = NA_real_,
          penalty_shape_distance = NA_real_,
          penalty_scale_ratio = NA_real_,
          same_raw_sp = data.frame(residual_rel_l2 = NA_real_),
          scale_corrected = data.frame(residual_rel_l2 = NA_real_),
          edf_matched = data.frame(residual_rel_l2 = NA_real_),
          classification = "unclassified",
          diagnostics = list(
            schema_version = "kpcTprsResidualCPP-drift-isolation-v1",
            failed_closed = TRUE,
            failure_reason = conditionMessage(e),
            does_not_drive_graph_decisions = TRUE
          )
        )
      }
    )
    details[[name]] <- result
    rows[[length(rows) + 1L]] <- data.frame(
      scenario = name,
      conditioning_size = result$conditioning_size,
      uz_projector_distance = result$uz_projector_distance,
      raw_projector_distance = result$raw_projector_distance,
      absorbed_projector_distance = result$absorbed_projector_distance,
      penalty_shape_distance = result$penalty_shape_distance,
      penalty_scale_ratio = result$penalty_scale_ratio,
      same_raw_sp_residual_rel_l2 = result$same_raw_sp$residual_rel_l2,
      scale_corrected_residual_rel_l2 =
        result$scale_corrected$residual_rel_l2,
      edf_matched_residual_rel_l2 = result$edf_matched$residual_rel_l2,
      classification = result$classification,
      failure_reason = result$diagnostics$failure_reason %||% "",
      stringsAsFactors = FALSE
    )
  }
  list(
    summary = if (length(rows) == 0L) data.frame() else do.call(rbind, rows),
    details = details
  )
}

fastkpc_kpc_tprs_rel_l2 <- function(a, b) {
  a <- as.numeric(a)
  b <- as.numeric(b)
  denom <- sqrt(sum(b^2))
  if (denom == 0) return(sqrt(sum((a - b)^2)))
  sqrt(sum((a - b)^2)) / denom
}

fastkpc_kpc_tprs_fixed_sp_parity <- function(
    y, S, sp_values = c(1e-6, 1e-3, 1, 1e3, 1e6),
    fitted_rel_l2_tol = 1e-5,
    residual_rel_l2_tol = 1e-5,
    edf_abs_tol = 1e-4,
    tol = sqrt(.Machine$double.eps)) {
  input <- fastkpc_kpc_tprs_validate_input(y, S)
  candidate_setup <- kpc_tprs_residual_cpp_setup(input$S, tol = tol)
  oracle_setup <- fastkpc_kpc_tprs_mgcv_oracle_setup(
    y = input$y, S = input$S, sp = sp_values[[1L]]
  )
  projector_distance <- fastkpc_kpc_tprs_projector_distance(
    candidate_setup$X, oracle_setup$smooth$X, tol = tol
  )
  penalty_spectrum_distance <- fastkpc_kpc_tprs_spectrum_distance(
    candidate_setup$penalty, oracle_setup$penalty, tol = tol
  )

  rows <- lapply(as.numeric(sp_values), function(sp) {
    oracle <- fastkpc_kpc_tprs_mgcv_oracle_setup(input$y, input$S, sp = sp)
    candidate <- fastkpc_kpc_tprs_fixed_sp_solve_candidate(
      input$y, input$S, sp = sp, tol = tol
    )
    data.frame(
      sp = sp,
      fitted_rel_l2 = fastkpc_kpc_tprs_rel_l2(candidate$fitted, oracle$fitted),
      residual_rel_l2 = fastkpc_kpc_tprs_rel_l2(candidate$residuals,
                                                oracle$residuals),
      edf_candidate = candidate$edf,
      edf_oracle = oracle$edf,
      edf_abs_diff = abs(candidate$edf - oracle$edf),
      stringsAsFactors = FALSE
    )
  })
  fixed_sp <- do.call(rbind, rows)
  isolation <- tryCatch(
    fastkpc_kpc_tprs_fixed_sp_drift_isolation(
      y = input$y,
      S = input$S,
      sp = sp_values[[1L]],
      tol = tol
    ),
    error = function(e) {
      list(
        raw_projector_distance = projector_distance,
        absorbed_projector_distance = NA_real_,
        penalty_shape_distance = NA_real_,
        penalty_scale_ratio = NA_real_,
        scale_corrected = data.frame(residual_rel_l2 = NA_real_,
                                     edf_abs_diff = NA_real_),
        edf_matched = data.frame(residual_rel_l2 = NA_real_,
                                 edf_abs_diff = NA_real_),
        classification = "unclassified",
        diagnostics = list(failure_reason = conditionMessage(e))
      )
    }
  )
  gate_b1 <- is.finite(projector_distance) &&
    is.finite(penalty_spectrum_distance)
  gate_b2 <- all(is.finite(fixed_sp$fitted_rel_l2)) &&
    all(is.finite(fixed_sp$residual_rel_l2)) &&
    all(fixed_sp$fitted_rel_l2 <= fitted_rel_l2_tol) &&
    all(fixed_sp$residual_rel_l2 <= residual_rel_l2_tol) &&
    all(fixed_sp$edf_abs_diff <= edf_abs_tol)

  list(
    backend_family = "kpcTprsResidualCPP",
    mode = "fixed-sp-parity-shadow",
    authoritative = FALSE,
    conditioning_size = as.integer(ncol(input$S)),
    oracle_setup_fingerprint = oracle_setup$setup_fingerprint,
    candidate_setup_fingerprint =
      fastkpc_kpc_tprs_setup_fingerprint(input$S, candidate_setup),
    projector_distance = projector_distance,
    penalty_spectrum_distance = penalty_spectrum_distance,
    uz_projector_distance = isolation$uz_projector_distance,
    raw_projector_distance = isolation$raw_projector_distance,
    absorbed_projector_distance = isolation$absorbed_projector_distance,
    penalty_shape_distance = isolation$penalty_shape_distance,
    penalty_scale_ratio = isolation$penalty_scale_ratio,
    scale_corrected = isolation$scale_corrected,
    edf_matched = isolation$edf_matched,
    drift_classification = isolation$classification,
    fixed_sp = fixed_sp,
    fixed_sp_residual_rel_l2 = max(fixed_sp$residual_rel_l2, na.rm = TRUE),
    fixed_sp_fitted_rel_l2 = max(fixed_sp$fitted_rel_l2, na.rm = TRUE),
    edf_abs_diff = max(fixed_sp$edf_abs_diff, na.rm = TRUE),
    gate_b1_passed = isTRUE(gate_b1),
    gate_b2_passed = isTRUE(gate_b2),
    passed = isTRUE(gate_b1 && gate_b2),
    diagnostics = list(
      schema_version = "kpcTprsResidualCPP-fixed-sp-parity-v1",
      candidate_basis_rank = candidate_setup$basis_rank,
      oracle_basis_rank = oracle_setup$basis_rank,
      candidate_null_space_rank = candidate_setup$null_space_rank,
      oracle_null_space_rank = oracle_setup$null_space_rank,
      drift_isolation_failure_reason =
        isolation$diagnostics$failure_reason %||% "",
      does_not_drive_graph_decisions = TRUE
    )
  )
}

fastkpc_run_kpc_tprs_fixed_sp_parity_campaign <- function(
    scenarios, sp_values = c(1e-6, 1e-3, 1, 1e3, 1e6),
    fitted_rel_l2_tol = 1e-5,
    residual_rel_l2_tol = 1e-5,
    edf_abs_tol = 1e-4) {
  rows <- list()
  details <- list()
  for (name in names(scenarios)) {
    scenario <- scenarios[[name]]
    result <- fastkpc_kpc_tprs_fixed_sp_parity(
      y = scenario$y,
      S = scenario$S,
      sp_values = sp_values,
      fitted_rel_l2_tol = fitted_rel_l2_tol,
      residual_rel_l2_tol = residual_rel_l2_tol,
      edf_abs_tol = edf_abs_tol
    )
    details[[name]] <- result
    rows[[length(rows) + 1L]] <- data.frame(
      scenario = name,
      conditioning_size = result$conditioning_size,
      projector_distance = result$projector_distance,
      penalty_spectrum_distance = result$penalty_spectrum_distance,
      max_residual_rel_l2 = result$fixed_sp_residual_rel_l2,
      max_fitted_rel_l2 = result$fixed_sp_fitted_rel_l2,
      max_edf_abs_diff = result$edf_abs_diff,
      gate_b1_passed = result$gate_b1_passed,
      gate_b2_passed = result$gate_b2_passed,
      passed = result$passed,
      stringsAsFactors = FALSE
    )
  }
  list(
    summary = if (length(rows) == 0L) data.frame() else do.call(rbind, rows),
    details = details
  )
}

fastkpc_kpc_tprs_mapped_sp_residual_parity <- function(
    y, S, sp_values = c(1e-6, 1e-3, 1, 1e3, 1e6),
    fitted_rel_l2_tol = 1e-6,
    residual_rel_l2_tol = 1e-6,
    edf_abs_tol = 1e-6,
    tol = sqrt(.Machine$double.eps)) {
  input <- fastkpc_kpc_tprs_validate_input(y, S)
  candidate_setup <- kpc_tprs_residual_cpp_setup(input$S, tol = tol)
  oracle_absorbed <- fastkpc_kpc_tprs_absorbed_oracle_setup(input$y, input$S)
  mapped_sp <- fastkpc_kpc_tprs_map_mgcv_sp_to_canonical(
    sp = sp_values,
    setup = candidate_setup,
    oracle_absorbed = oracle_absorbed,
    tol = tol
  )
  rows <- lapply(seq_along(as.numeric(sp_values)), function(i) {
    sp <- as.numeric(sp_values[[i]])
    oracle <- fastkpc_kpc_tprs_mgcv_oracle_setup(input$y, input$S, sp = sp)
    candidate <- fastkpc_kpc_tprs_solve_candidate_setup(
      input$y,
      candidate_setup,
      sp = mapped_sp$canonical_lambda[[i]],
      include_intercept = TRUE,
      tol = tol
    )
    data.frame(
      source_sp = sp,
      canonical_lambda = mapped_sp$canonical_lambda[[i]],
      fitted_rel_l2 = fastkpc_kpc_tprs_rel_l2(candidate$fitted, oracle$fitted),
      residual_rel_l2 = fastkpc_kpc_tprs_rel_l2(candidate$residuals,
                                                oracle$residuals),
      edf_candidate = candidate$edf,
      edf_oracle = oracle$edf,
      edf_abs_diff = abs(candidate$edf - oracle$edf),
      stringsAsFactors = FALSE
    )
  })
  fixed_sp <- do.call(rbind, rows)
  gate_b_r <- all(is.finite(fixed_sp$fitted_rel_l2)) &&
    all(is.finite(fixed_sp$residual_rel_l2)) &&
    all(is.finite(fixed_sp$edf_abs_diff)) &&
    all(fixed_sp$fitted_rel_l2 <= fitted_rel_l2_tol) &&
    all(fixed_sp$residual_rel_l2 <= residual_rel_l2_tol) &&
    all(fixed_sp$edf_abs_diff <= edf_abs_tol)

  list(
    backend_family = "kpcTprsResidualCPP",
    mode = "mapped-sp-residual-parity-shadow",
    authoritative = FALSE,
    conditioning_size = as.integer(ncol(input$S)),
    mapped_sp = mapped_sp,
    fixed_sp = fixed_sp,
    fixed_sp_residual_rel_l2 = max(fixed_sp$residual_rel_l2, na.rm = TRUE),
    fixed_sp_fitted_rel_l2 = max(fixed_sp$fitted_rel_l2, na.rm = TRUE),
    edf_abs_diff = max(fixed_sp$edf_abs_diff, na.rm = TRUE),
    gate_b_r_passed = isTRUE(gate_b_r),
    passed = isTRUE(gate_b_r),
    diagnostics = list(
      schema_version = "kpcTprsResidualCPP-mapped-sp-parity-v1",
      candidate_setup_fingerprint =
        fastkpc_kpc_tprs_setup_fingerprint(input$S, candidate_setup),
      source_sp_semantics = mapped_sp$source_sp_semantics,
      target_lambda_semantics = mapped_sp$target_lambda_semantics,
      include_intercept = TRUE,
      exact_mgcv_s_scale_required = FALSE,
      does_not_drive_graph_decisions = TRUE
    )
  )
}

fastkpc_run_kpc_tprs_mapped_sp_residual_parity_campaign <- function(
    scenarios, sp_values = c(1e-6, 1e-3, 1, 1e3, 1e6),
    fitted_rel_l2_tol = 1e-6,
    residual_rel_l2_tol = 1e-6,
    edf_abs_tol = 1e-6) {
  rows <- list()
  details <- list()
  for (name in names(scenarios)) {
    scenario <- scenarios[[name]]
    result <- tryCatch(
      fastkpc_kpc_tprs_mapped_sp_residual_parity(
        y = scenario$y,
        S = scenario$S,
        sp_values = sp_values,
        fitted_rel_l2_tol = fitted_rel_l2_tol,
        residual_rel_l2_tol = residual_rel_l2_tol,
        edf_abs_tol = edf_abs_tol
      ),
      error = function(e) {
        list(
          backend_family = "kpcTprsResidualCPP",
          mode = "mapped-sp-residual-parity-failed-closed",
          authoritative = FALSE,
          conditioning_size = as.integer(ncol(as.matrix(scenario$S))),
          mapped_sp = list(),
          fixed_sp = data.frame(),
          fixed_sp_residual_rel_l2 = NA_real_,
          fixed_sp_fitted_rel_l2 = NA_real_,
          edf_abs_diff = NA_real_,
          gate_b_r_passed = FALSE,
          passed = FALSE,
          diagnostics = list(
            schema_version = "kpcTprsResidualCPP-mapped-sp-parity-v1",
            failed_closed = TRUE,
            failure_reason = conditionMessage(e),
            does_not_drive_graph_decisions = TRUE
          )
        )
      }
    )
    details[[name]] <- result
    rows[[length(rows) + 1L]] <- data.frame(
      scenario = name,
      conditioning_size = result$conditioning_size,
      max_residual_rel_l2 = result$fixed_sp_residual_rel_l2,
      max_fitted_rel_l2 = result$fixed_sp_fitted_rel_l2,
      max_edf_abs_diff = result$edf_abs_diff,
      gate_b_r_passed = result$gate_b_r_passed,
      passed = result$passed,
      failure_reason = result$diagnostics$failure_reason %||% "",
      stringsAsFactors = FALSE
    )
  }
  list(
    summary = if (length(rows) == 0L) data.frame() else do.call(rbind, rows),
    details = details
  )
}

fastkpc_kpc_tprs_mapped_sp_ci_parity <- function(
    x, y, S, sp = 1, index = 1, legacy_index = TRUE,
    p_abs_tol = 1e-8,
    tol = sqrt(.Machine$double.eps)) {
  input_x <- fastkpc_kpc_tprs_validate_input(x, S)
  input_y <- fastkpc_kpc_tprs_validate_input(y, S)
  candidate_x <- fastkpc_kpc_tprs_mapped_sp_solve_candidate(
    y = input_x$y, S = input_x$S, sp = sp, tol = tol)
  candidate_y <- fastkpc_kpc_tprs_mapped_sp_solve_candidate(
    y = input_y$y, S = input_y$S, sp = sp, tol = tol)
  oracle_x <- fastkpc_kpc_tprs_mgcv_oracle_setup(input_x$y, input_x$S, sp = sp)
  oracle_y <- fastkpc_kpc_tprs_mgcv_oracle_setup(input_y$y, input_y$S, sp = sp)

  candidate_p <- dcov_gamma_exact(candidate_x$residuals,
                                  candidate_y$residuals,
                                  index = index,
                                  legacy_index = legacy_index)$p.value
  oracle_p <- dcov_gamma_exact(oracle_x$residuals,
                               oracle_y$residuals,
                               index = index,
                               legacy_index = legacy_index)$p.value
  p_abs_diff <- abs(as.numeric(candidate_p) - as.numeric(oracle_p))
  list(
    backend_family = "kpcTprsResidualCPP",
    mode = "mapped-sp-ci-parity-shadow",
    authoritative = FALSE,
    candidate_p = as.numeric(candidate_p),
    oracle_p = as.numeric(oracle_p),
    p_abs_diff = p_abs_diff,
    passed = is.finite(p_abs_diff) && p_abs_diff <= p_abs_tol,
    diagnostics = list(
      schema_version = "kpcTprsResidualCPP-mapped-sp-ci-parity-v1",
      candidate_x_mapped_sp = candidate_x$mapped_sp,
      candidate_y_mapped_sp = candidate_y$mapped_sp,
      does_not_drive_graph_decisions = TRUE
    )
  )
}

fastkpc_kpc_tprs_gcv_score_for_setup <- function(
    y, setup, lambda, tol = sqrt(.Machine$double.eps)) {
  fit <- fastkpc_kpc_tprs_solve_candidate_setup(
    y = y,
    setup = setup,
    sp = lambda,
    include_intercept = TRUE,
    tol = tol
  )
  n <- length(y)
  rss <- sum(fit$residuals^2)
  denom <- n - fit$edf
  gcv <- if (is.finite(fit$edf) && denom > tol) {
    n * rss / (denom * denom)
  } else {
    Inf
  }
  list(
    lambda = as.numeric(lambda),
    rss = as.numeric(rss),
    edf = as.numeric(fit$edf),
    gcv = as.numeric(gcv),
    valid = is.finite(gcv),
    fit = fit
  )
}

fastkpc_kpc_tprs_gcv_candidate <- function(
    y, S,
    lambda_grid = exp(seq(log(1e-4), log(1e4), length.out = 17L)),
    refine = TRUE,
    tol = sqrt(.Machine$double.eps)) {
  input <- fastkpc_kpc_tprs_validate_input(y, S)
  lambda_grid <- as.numeric(lambda_grid)
  if (length(lambda_grid) == 0L ||
      any(!is.finite(lambda_grid)) || any(lambda_grid <= 0)) {
    fastkpc_kpc_tprs_stop("lambda_grid must be positive finite")
  }
  lambda_grid <- sort(unique(lambda_grid))
  setup <- kpc_tprs_residual_cpp_setup(input$S, tol = tol)
  grid_eval <- lapply(lambda_grid, function(lambda) {
    fastkpc_kpc_tprs_gcv_score_for_setup(
      y = input$y, setup = setup, lambda = lambda, tol = tol)
  })
  grid <- do.call(rbind, lapply(grid_eval, function(x) {
    data.frame(
      lambda = x$lambda,
      rss = x$rss,
      edf = x$edf,
      gcv = x$gcv,
      valid = x$valid,
      stringsAsFactors = FALSE
    )
  }))
  if (!any(grid$valid)) {
    stop("No valid GCV candidate in lambda_grid", call. = FALSE)
  }
  selected_grid_index <- which(grid$gcv == min(grid$gcv[grid$valid]))[1L]
  selected_lambda <- grid$lambda[[selected_grid_index]]
  selected_score <- grid$gcv[[selected_grid_index]]
  brent_refined <- FALSE
  brent_selected <- FALSE

  if (isTRUE(refine) && length(lambda_grid) >= 3L) {
    idx <- selected_grid_index
    lower_idx <- max(1L, idx - 1L)
    upper_idx <- min(length(lambda_grid), idx + 1L)
    if (lower_idx == idx && upper_idx < length(lambda_grid)) {
      upper_idx <- idx + 1L
    }
    if (upper_idx == idx && lower_idx > 1L) {
      lower_idx <- idx - 1L
    }
    lower <- log(lambda_grid[[lower_idx]])
    upper <- log(lambda_grid[[upper_idx]])
    if (is.finite(lower) && is.finite(upper) && lower < upper) {
      brent_refined <- TRUE
      objective <- function(log_lambda) {
        fastkpc_kpc_tprs_gcv_score_for_setup(
          y = input$y,
          setup = setup,
          lambda = exp(log_lambda),
          tol = tol
        )$gcv
      }
      opt <- stats::optimize(objective, interval = c(lower, upper))
      refined_lambda <- exp(opt$minimum)
      refined_score <- as.numeric(opt$objective)
      improvement_tol <- max(1e-12, abs(selected_score) * 1e-8)
      if (is.finite(refined_score) &&
          refined_score < selected_score - improvement_tol) {
        selected_lambda <- refined_lambda
        selected_score <- refined_score
        brent_selected <- TRUE
      }
    }
  }

  selected <- fastkpc_kpc_tprs_solve_candidate_setup(
    y = input$y,
    setup = setup,
    sp = selected_lambda,
    include_intercept = TRUE,
    tol = tol
  )
  selected$mode <- "continuous-gcv-candidate-shadow"
  selected$selected_sp <- as.numeric(selected_lambda)
  selected$score <- as.numeric(selected_score)
  selected$grid <- grid
  selected$selected_grid_index <- as.integer(selected_grid_index)
  selected$backend_family <- "kpcTprsResidualCPP"
  selected$authoritative <- FALSE
  selected$diagnostics <- c(
    selected$diagnostics,
    list(
      schema_version = "kpcTprsResidualCPP-continuous-gcv-v1",
      gcv_source = "kpc-canonical-continuous-gcv",
      sp_semantics = "kpc-canonical-penalty",
      brent_refined = isTRUE(brent_refined),
      brent_selected = isTRUE(brent_selected),
      bracket_lower_lambda = lambda_grid[[max(1L, selected_grid_index - 1L)]],
      bracket_upper_lambda =
        lambda_grid[[min(length(lambda_grid), selected_grid_index + 1L)]],
      does_not_drive_graph_decisions = TRUE
    )
  )
  selected
}

fastkpc_kpc_tprs_shadow_sepsets <- function(p) {
  replicate(p, replicate(p, integer(), simplify = FALSE), simplify = FALSE)
}

fastkpc_kpc_tprs_shadow_combinations <- function(values, choose) {
  values <- as.integer(values)
  if (choose == 0L) return(list(integer()))
  if (length(values) < choose) return(list())
  lapply(utils::combn(values, choose, simplify = FALSE), as.integer)
}

fastkpc_kpc_tprs_shadow_neighbors <- function(adjacency, vertex, excluded) {
  as.integer(which(adjacency[, vertex] & seq_len(nrow(adjacency)) != excluded))
}

fastkpc_kpc_tprs_named_or_na <- function(x, name) {
  if (is.null(x) || is.null(names(x)) || !(name %in% names(x))) return(NA_real_)
  as.numeric(x[[name]])
}

fastkpc_kpc_tprs_shadow_ci <- function(data, x, y, S,
                                       backend = c("oracle", "candidate"),
                                       index = 1,
                                       legacy_index = TRUE) {
  backend <- match.arg(backend)
  if (length(S) == 0L) {
    p <- dcov_gamma_exact(data[, x], data[, y],
                          index = index,
                          legacy_index = legacy_index)$p.value
    return(list(
      p.value = as.numeric(p),
      mode = "unconditional-dcov",
      selected_sp = NA_real_,
      score = NA_real_,
      edf = NA_real_
    ))
  }
  S_matrix <- data[, S, drop = FALSE]
  if (identical(backend, "oracle")) {
    px <- fastkpc_kpc_tprs_mgcv_gcv_oracle_fit(data[, x], S_matrix)
    py <- fastkpc_kpc_tprs_mgcv_gcv_oracle_fit(data[, y], S_matrix)
    rx <- px$residuals
    ry <- py$residuals
    mode <- "mgcv-oracle-gcv"
  } else {
    px <- fastkpc_kpc_tprs_gcv_candidate(data[, x], S_matrix)
    py <- fastkpc_kpc_tprs_gcv_candidate(data[, y], S_matrix)
    rx <- px$residuals
    ry <- py$residuals
    mode <- "continuous-gcv-candidate-shadow"
  }
  p <- dcov_gamma_exact(rx, ry, index = index,
                        legacy_index = legacy_index)$p.value
  list(
    p.value = as.numeric(p),
    mode = mode,
    selected_sp = c(x = as.numeric(px$selected_sp), y = as.numeric(py$selected_sp)),
    score = c(x = as.numeric(px$score), y = as.numeric(py$score)),
    edf = c(x = as.numeric(px$edf), y = as.numeric(py$edf))
  )
}

fastkpc_kpc_tprs_shadow_run_skeleton <- function(
    data, alpha, max_conditioning_size,
    backend = c("oracle", "candidate"),
    index = 1, legacy_index = TRUE) {
  backend <- match.arg(backend)
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  p <- ncol(data)
  adjacency <- matrix(TRUE, p, p)
  diag(adjacency) <- FALSE
  pmax <- matrix(-Inf, p, p)
  diag(pmax) <- 1
  sepsets <- fastkpc_kpc_tprs_shadow_sepsets(p)
  n_edge_tests <- integer(max_conditioning_size + 1L)
  trace_rows <- list()
  test_id <- 0L

  for (ord in seq.int(0L, as.integer(max_conditioning_size))) {
    snapshot <- adjacency
    delete_edges <- matrix(FALSE, p, p)
    for (x in seq_len(p - 1L)) {
      for (y in seq.int(x + 1L, p)) {
        if (!snapshot[x, y]) next
        edge_done <- FALSE
        for (side in c("x", "y")) {
          if (edge_done) break
          source_vertex <- if (identical(side, "x")) x else y
          target_vertex <- if (identical(side, "x")) y else x
          neighbors <- fastkpc_kpc_tprs_shadow_neighbors(
            snapshot, source_vertex, target_vertex)
          for (S in fastkpc_kpc_tprs_shadow_combinations(neighbors, ord)) {
            test_id <- test_id + 1L
            ci <- fastkpc_kpc_tprs_shadow_ci(
              data = data,
              x = source_vertex,
              y = target_vertex,
              S = S,
              backend = backend,
              index = index,
              legacy_index = legacy_index
            )
            pval <- as.numeric(ci$p.value)
            if (pval > pmax[x, y]) {
              pmax[x, y] <- pval
              pmax[y, x] <- pval
            }
            deleted <- is.finite(pval) && pval >= as.numeric(alpha)
            if (deleted) {
              delete_edges[x, y] <- TRUE
              delete_edges[y, x] <- TRUE
              sepsets[[x]][[y]] <- as.integer(S)
              sepsets[[y]][[x]] <- as.integer(S)
              edge_done <- TRUE
            }
            trace_rows[[length(trace_rows) + 1L]] <- data.frame(
              canonical_test_order_id = test_id,
              conditioning_level = ord,
              edge_x = x,
              edge_y = y,
              x = source_vertex,
              y = target_vertex,
              conditioning_target_side = side,
              S_key = if (length(S) == 0L) "" else paste(S, collapse = "|"),
              p.value = pval,
              delete_edge = deleted,
              mode = ci$mode,
              selected_sp_x = fastkpc_kpc_tprs_named_or_na(ci$selected_sp, "x"),
              selected_sp_y = fastkpc_kpc_tprs_named_or_na(ci$selected_sp, "y"),
              score_x = fastkpc_kpc_tprs_named_or_na(ci$score, "x"),
              score_y = fastkpc_kpc_tprs_named_or_na(ci$score, "y"),
              edf_x = fastkpc_kpc_tprs_named_or_na(ci$edf, "x"),
              edf_y = fastkpc_kpc_tprs_named_or_na(ci$edf, "y"),
              stringsAsFactors = FALSE
            )
            n_edge_tests[[ord + 1L]] <- n_edge_tests[[ord + 1L]] + 1L
            if (edge_done) break
          }
        }
      }
    }
    adjacency[delete_edges] <- FALSE
  }
  list(
    adjacency = adjacency,
    sepsets = sepsets,
    pMax = pmax,
    n.edgetests = as.integer(n_edge_tests),
    trace = if (length(trace_rows) == 0L) data.frame() else do.call(rbind, trace_rows)
  )
}

fastkpc_kpc_tprs_shadow_skeleton_shd <- function(left, right) {
  left <- as.matrix(left)
  right <- as.matrix(right)
  sum(left[upper.tri(left)] != right[upper.tri(right)])
}

fastkpc_kpc_tprs_shadow_replay_trace <- function(p, trace) {
  adjacency <- matrix(TRUE, p, p)
  diag(adjacency) <- FALSE
  pmax <- matrix(-Inf, p, p)
  diag(pmax) <- 1
  sepsets <- fastkpc_kpc_tprs_shadow_sepsets(p)
  max_level <- if (nrow(trace) == 0L) 0L else max(trace$conditioning_level)
  n_edge_tests <- integer(max_level + 1L)
  if (nrow(trace) > 0L) {
    for (i in seq_len(nrow(trace))) {
      row <- trace[i, , drop = FALSE]
      x <- as.integer(row$edge_x)
      y <- as.integer(row$edge_y)
      pval <- as.numeric(row$p.value)
      if (pval > pmax[x, y]) {
        pmax[x, y] <- pval
        pmax[y, x] <- pval
      }
      level <- as.integer(row$conditioning_level)
      n_edge_tests[[level + 1L]] <- n_edge_tests[[level + 1L]] + 1L
      if (isTRUE(row$delete_edge) && isTRUE(adjacency[x, y])) {
        adjacency[x, y] <- FALSE
        adjacency[y, x] <- FALSE
        S <- if (!nzchar(row$S_key)) {
          integer()
        } else {
          as.integer(strsplit(row$S_key, "|", fixed = TRUE)[[1L]])
        }
        sepsets[[x]][[y]] <- S
        sepsets[[y]][[x]] <- S
      }
    }
  }
  list(
    adjacency = adjacency,
    sepsets = sepsets,
    pMax = pmax,
    n.edgetests = as.integer(n_edge_tests),
    trace = trace
  )
}

fastkpc_kpc_tprs_shadow_sepset_mismatch_rate <- function(left, right) {
  p <- length(left)
  mismatches <- 0L
  total <- 0L
  for (i in seq_len(p - 1L)) {
    for (j in seq.int(i + 1L, p)) {
      total <- total + 1L
      li <- sort(as.integer(left[[i]][[j]]))
      ri <- sort(as.integer(right[[i]][[j]]))
      if (!identical(li, ri)) mismatches <- mismatches + 1L
    }
  }
  mismatches / max(1L, total)
}

fastkpc_kpc_tprs_shadow_campaign <- function(
    data, alpha = 0.05, max_conditioning_size = 1L,
    index = 1, legacy_index = TRUE) {
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  oracle <- fastkpc_kpc_tprs_shadow_run_skeleton(
    data = data,
    alpha = alpha,
    max_conditioning_size = max_conditioning_size,
    backend = "oracle",
    index = index,
    legacy_index = legacy_index
  )
  candidate_trace_rows <- lapply(seq_len(nrow(oracle$trace)), function(i) {
    row <- oracle$trace[i, , drop = FALSE]
    S <- if (!nzchar(row$S_key)) {
      integer()
    } else {
      as.integer(strsplit(row$S_key, "|", fixed = TRUE)[[1L]])
    }
    ci <- fastkpc_kpc_tprs_shadow_ci(
      data = data,
      x = as.integer(row$x),
      y = as.integer(row$y),
      S = S,
      backend = "candidate",
      index = index,
      legacy_index = legacy_index
    )
    data.frame(
      canonical_test_order_id = row$canonical_test_order_id,
      conditioning_level = row$conditioning_level,
      edge_x = row$edge_x,
      edge_y = row$edge_y,
      x = row$x,
      y = row$y,
      conditioning_target_side = row$conditioning_target_side,
      S_key = row$S_key,
      p.value = ci$p.value,
      delete_edge = is.finite(ci$p.value) && ci$p.value >= as.numeric(alpha),
      mode = ci$mode,
      selected_sp_x = fastkpc_kpc_tprs_named_or_na(ci$selected_sp, "x"),
      selected_sp_y = fastkpc_kpc_tprs_named_or_na(ci$selected_sp, "y"),
      score_x = fastkpc_kpc_tprs_named_or_na(ci$score, "x"),
      score_y = fastkpc_kpc_tprs_named_or_na(ci$score, "y"),
      edf_x = fastkpc_kpc_tprs_named_or_na(ci$edf, "x"),
      edf_y = fastkpc_kpc_tprs_named_or_na(ci$edf, "y"),
      stringsAsFactors = FALSE
    )
  })
  candidate_trace <- if (length(candidate_trace_rows) == 0L) {
    data.frame()
  } else {
    do.call(rbind, candidate_trace_rows)
  }
  candidate <- fastkpc_kpc_tprs_shadow_replay_trace(
    p = ncol(data),
    trace = candidate_trace
  )
  n <- min(nrow(oracle$trace), nrow(candidate$trace))
  trace <- data.frame()
  if (n > 0L) {
    ot <- oracle$trace[seq_len(n), , drop = FALSE]
    ct <- candidate$trace[seq_len(n), , drop = FALSE]
    trace <- data.frame(
      canonical_test_order_id = ot$canonical_test_order_id,
      conditioning_level = ot$conditioning_level,
      x = ot$x,
      y = ot$y,
      edge_x = ot$edge_x,
      edge_y = ot$edge_y,
      S_key = ot$S_key,
      oracle_p = ot$p.value,
      candidate_p = ct$p.value,
      oracle_delete = ot$delete_edge,
      candidate_delete = ct$delete_edge,
      decision_flip = ot$delete_edge != ct$delete_edge,
      candidate_mode = ct$mode,
      candidate_selected_sp = paste(ct$selected_sp_x, ct$selected_sp_y, sep = "|"),
      candidate_score = paste(ct$score_x, ct$score_y, sep = "|"),
      candidate_edf = paste(ct$edf_x, ct$edf_y, sep = "|"),
      stringsAsFactors = FALSE
    )
  }
  pmax_diff <- abs(candidate$pMax - oracle$pMax)
  pmax_diff <- pmax_diff[upper.tri(pmax_diff)]
  log_p_diff <- if (nrow(trace) > 0L) {
    abs(log(pmax(trace$candidate_p, .Machine$double.xmin)) -
          log(pmax(trace$oracle_p, .Machine$double.xmin)))
  } else {
    numeric()
  }
  list(
    backend_family = "kpcTprsResidualCPP",
    mode = "shadow-graph-campaign",
    authoritative = FALSE,
    oracle_authoritative = TRUE,
    p_used_source = "oracle",
    decision_source = "oracle",
    oracle = oracle[c("adjacency", "sepsets", "pMax", "n.edgetests")],
    candidate = candidate[c("adjacency", "sepsets", "pMax", "n.edgetests")],
    trace = trace,
    agreement = list(
      adjacency_identical = identical(oracle$adjacency, candidate$adjacency),
      skeleton_shd = as.integer(fastkpc_kpc_tprs_shadow_skeleton_shd(
        oracle$adjacency, candidate$adjacency)),
      sepset_mismatch_rate = fastkpc_kpc_tprs_shadow_sepset_mismatch_rate(
        oracle$sepsets, candidate$sepsets),
      pmax_max_abs_diff =
        if (length(pmax_diff) == 0L) 0 else max(pmax_diff, na.rm = TRUE),
      n_edgetests_identical =
        identical(oracle$n.edgetests, candidate$n.edgetests),
      decision_flip_count =
        if (nrow(trace) == 0L) 0L else sum(trace$decision_flip, na.rm = TRUE),
      max_log_p_abs_diff =
        if (length(log_p_diff) == 0L) 0 else max(log_p_diff, na.rm = TRUE)
    ),
    diagnostics = list(
      schema_version = "kpcTprsResidualCPP-shadow-campaign-v1",
      does_not_drive_graph_decisions = TRUE
    )
  )
}

