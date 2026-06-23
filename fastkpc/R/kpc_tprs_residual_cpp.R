source("fastkpc/R/mgcv_compat_contract.R")
source("fastkpc/R/native.R")

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
  X <- as.matrix(setup$absorbed$X)
  P <- as.matrix(setup$absorbed$penalty)
  A <- crossprod(X) + as.numeric(sp) * P
  b <- crossprod(X, input$y)
  theta <- as.numeric(qr.solve(A, b, tol = tol))
  fitted <- as.numeric(X %*% theta)
  residuals <- as.numeric(input$y - fitted)
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
    setup = setup,
    setup_fingerprint = fastkpc_kpc_tprs_setup_fingerprint(input$S, setup),
    diagnostics = list(
      solver = "absorbed-penalized-least-squares",
      does_not_drive_graph_decisions = TRUE
    )
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
