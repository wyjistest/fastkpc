if (!exists("fastkpc_run_mgcv_residual_setup_shadow", mode = "function")) {
  source("fastkpc/R/mgcv_residual_setup_shadow.R")
}

fastkpc_drift_iso_read_shadow <- function(shadow_dir) {
  cases_path <- file.path(shadow_dir, "cases.csv")
  if (!file.exists(cases_path)) {
    stop("shadow cases.csv not found: ", cases_path, call. = FALSE)
  }
  utils::read.csv(cases_path, stringsAsFactors = FALSE)
}

fastkpc_drift_iso_read_oracle <- function(oracle_dir) {
  cases_path <- file.path(oracle_dir, "cases.csv")
  residuals_path <- file.path(oracle_dir, "residuals.rds")
  if (!file.exists(cases_path)) {
    stop("oracle cases.csv not found: ", cases_path, call. = FALSE)
  }
  if (!file.exists(residuals_path)) {
    stop("oracle residuals.rds not found: ", residuals_path, call. = FALSE)
  }
  list(
    cases = utils::read.csv(cases_path, stringsAsFactors = FALSE),
    residuals = readRDS(residuals_path)
  )
}

fastkpc_drift_iso_max_abs <- function(a, b) {
  max(abs(as.numeric(a) - as.numeric(b)))
}

fastkpc_drift_iso_rel_l2 <- function(a, b) {
  fastkpc_mgcv_setup_shadow_rel_l2(a, b)
}

fastkpc_drift_iso_normal_matrix_condition <- function(
    setup, sp = setup$sp, tol = sqrt(.Machine$double.eps)) {
  X <- as.matrix(setup$X)
  y <- as.numeric(setup$y)
  P <- fastkpc_assemble_penalty(
    p = ncol(X), S = setup$S, off = setup$off, sp = sp, H = setup$H
  )
  if (is.null(setup$w)) {
    Xw <- X
  } else {
    Xw <- X * sqrt(as.numeric(setup$w))
  }
  Z <- fastkpc_constraint_nullspace(C = setup$C, p = ncol(X), tol = tol)
  XZ <- Xw %*% Z
  A <- crossprod(XZ) + crossprod(Z, P %*% Z)
  suppressWarnings(kappa(A, exact = TRUE))
}

fastkpc_drift_iso_solve_target <- function(data, case, entry, suffix,
                                           target_col, env, tol) {
  fit_data_info <- fastkpc_mgcv_setup_shadow_fit_data(data, case)
  formula <- fastkpc_mgcv_oracle_formula(
    target_col, length(fit_data_info$S), env
  )
  setup <- fastkpc_mgcv_extract_setup(
    formula = formula,
    data = fit_data_info$data,
    sp = fastkpc_mgcv_setup_shadow_sp(entry, suffix),
    method = fastkpc_mgcv_setup_shadow_method(entry, suffix),
    target = target_col,
    S = fit_data_info$S_fit
  )
  cmagic <- fastkpc_mgcv_solve_setup_fixed_sp(setup, tol = tol)
  cpp <- fastkpc_mgcv_solve_setup_fixed_sp_cpp(setup, tol = tol)
  r_beta <- fastkpc_solve_gaussian_penalized_fixed_sp(
    X = setup$X,
    y = setup$y,
    S = setup$S,
    off = setup$off,
    sp = setup$sp,
    C = setup$C,
    H = setup$H,
    w = setup$w,
    tol = tol
  )
  r_fitted <- as.numeric(setup$X %*% r_beta)
  r_residuals <- as.numeric(setup$y - r_fitted)
  oracle_residuals <- as.numeric(entry[[paste0("residual_", suffix)]])
  oracle_fitted <- as.numeric(entry[[paste0("fitted_", suffix)]])

  list(
    setup = setup,
    oracle = list(fitted = oracle_fitted, residuals = oracle_residuals),
    cmagic = cmagic,
    r_normal = list(
      backend_family = "mgcvExtractR",
      mode = "fixed-sp-r-normal-equation-solve",
      solve_source = "fastkpc-r-normal-equations",
      coefficients = r_beta,
      fitted = r_fitted,
      residuals = r_residuals
    ),
    cpp = cpp,
    condition = fastkpc_drift_iso_normal_matrix_condition(
      setup, sp = setup$sp, tol = tol
    )
  )
}

fastkpc_drift_iso_target_row <- function(case, suffix, target_col, solved,
                                         residual_tol) {
  data.frame(
    case_id = case$case_id,
    target_role = suffix,
    target_col = target_col,
    source_level = fastkpc_mgcv_setup_shadow_case_value(
      case, "source_level", NA_integer_
    ),
    S_size = case$S_size,
    S_key = case$S_key,
    normal_matrix_condition = solved$condition,
    cmagic_vs_oracle_max_abs_diff =
      fastkpc_drift_iso_max_abs(
        solved$cmagic$residuals, solved$oracle$residuals
      ),
    cmagic_vs_oracle_rel_l2 =
      fastkpc_drift_iso_rel_l2(
        solved$cmagic$residuals, solved$oracle$residuals
      ),
    r_normal_vs_oracle_max_abs_diff =
      fastkpc_drift_iso_max_abs(
        solved$r_normal$residuals, solved$oracle$residuals
      ),
    r_normal_vs_oracle_rel_l2 =
      fastkpc_drift_iso_rel_l2(
        solved$r_normal$residuals, solved$oracle$residuals
      ),
    cpp_vs_oracle_max_abs_diff =
      fastkpc_drift_iso_max_abs(
        solved$cpp$residuals, solved$oracle$residuals
      ),
    cpp_vs_oracle_rel_l2 =
      fastkpc_drift_iso_rel_l2(
        solved$cpp$residuals, solved$oracle$residuals
      ),
    cpp_vs_r_normal_max_abs_diff =
      fastkpc_drift_iso_max_abs(
        solved$cpp$residuals, solved$r_normal$residuals
      ),
    cpp_vs_cmagic_max_abs_diff =
      fastkpc_drift_iso_max_abs(
        solved$cpp$residuals, solved$cmagic$residuals
      ),
    cmagic_matches_oracle =
      fastkpc_drift_iso_max_abs(
        solved$cmagic$residuals, solved$oracle$residuals
      ) <= residual_tol,
    cpp_matches_r_normal =
      fastkpc_drift_iso_max_abs(
        solved$cpp$residuals, solved$r_normal$residuals
      ) <= residual_tol,
    cpp_matches_cmagic =
      fastkpc_drift_iso_max_abs(
        solved$cpp$residuals, solved$cmagic$residuals
      ) <= residual_tol,
    stringsAsFactors = FALSE
  )
}

fastkpc_drift_iso_p_value <- function(rx, ry, index, numCol, env) {
  as.numeric(fastkpc_legacy_dcov_gamma_timed(
    rx, ry, index = index, numCol = numCol, env = env
  )$result$p.value)
}

fastkpc_drift_iso_case_row <- function(case, entry, solved_x, solved_y,
                                       alpha, index, numCol, env,
                                       residual_tol) {
  p_oracle <- as.numeric(entry$dcov_p_value)
  p_cmagic <- fastkpc_drift_iso_p_value(
    solved_x$cmagic$residuals, solved_y$cmagic$residuals,
    index, numCol, env
  )
  p_r_normal <- fastkpc_drift_iso_p_value(
    solved_x$r_normal$residuals, solved_y$r_normal$residuals,
    index, numCol, env
  )
  p_cpp <- fastkpc_drift_iso_p_value(
    solved_x$cpp$residuals, solved_y$cpp$residuals,
    index, numCol, env
  )
  cpp_vs_r_normal <- max(
    fastkpc_drift_iso_max_abs(
      solved_x$cpp$residuals, solved_x$r_normal$residuals
    ),
    fastkpc_drift_iso_max_abs(
      solved_y$cpp$residuals, solved_y$r_normal$residuals
    )
  )
  cmagic_vs_oracle <- max(
    fastkpc_drift_iso_max_abs(
      solved_x$cmagic$residuals, solved_x$oracle$residuals
    ),
    fastkpc_drift_iso_max_abs(
      solved_y$cmagic$residuals, solved_y$oracle$residuals
    )
  )
  cpp_vs_cmagic <- max(
    fastkpc_drift_iso_max_abs(
      solved_x$cpp$residuals, solved_x$cmagic$residuals
    ),
    fastkpc_drift_iso_max_abs(
      solved_y$cpp$residuals, solved_y$cmagic$residuals
    )
  )
  drift_layer <- if (cpp_vs_r_normal <= residual_tol &&
                     cmagic_vs_oracle <= residual_tol &&
                     cpp_vs_cmagic > residual_tol) {
    "normal_equation_vs_mgcv_magic"
  } else if (cpp_vs_r_normal > residual_tol) {
    "native_cpp_vs_r_normal"
  } else if (cmagic_vs_oracle > residual_tol) {
    "fixed_sp_setup_vs_full_mgcv"
  } else {
    "within_tolerance"
  }
  decision_oracle <- isTRUE(p_oracle > alpha)
  decision_cpp <- isTRUE(p_cpp > alpha)
  data.frame(
    case_id = case$case_id,
    role = case$role,
    source_level = fastkpc_mgcv_setup_shadow_case_value(
      case, "source_level", NA_integer_
    ),
    source_pmax = fastkpc_mgcv_setup_shadow_case_value(
      case, "source_pmax", NA_real_
    ),
    S_size = case$S_size,
    S_key = case$S_key,
    target_x = case$target_x,
    target_y = case$target_y,
    p_oracle = p_oracle,
    p_cmagic = p_cmagic,
    p_r_normal = p_r_normal,
    p_cpp = p_cpp,
    p_cpp_abs_diff = abs(p_cpp - p_oracle),
    p_r_normal_abs_diff = abs(p_r_normal - p_oracle),
    p_cmagic_abs_diff = abs(p_cmagic - p_oracle),
    decision_oracle = decision_oracle,
    decision_cpp = decision_cpp,
    decision_match = identical(decision_oracle, decision_cpp),
    decision_flip = !identical(decision_oracle, decision_cpp),
    cpp_vs_r_normal_max_abs_diff = cpp_vs_r_normal,
    cmagic_vs_oracle_max_abs_diff = cmagic_vs_oracle,
    cpp_vs_cmagic_max_abs_diff = cpp_vs_cmagic,
    drift_layer = drift_layer,
    recommended_action = if (identical(drift_layer,
                                       "normal_equation_vs_mgcv_magic")) {
      "fallback-or-implement-mgcv-equivalent-kernel"
    } else {
      "continue-shadow"
    },
    stringsAsFactors = FALSE
  )
}

fastkpc_run_mgcv_residual_cpp_numeric_drift_isolation <- function(
    data = NULL,
    shadow_dir = file.path("fastkpc", "artifacts",
                           "mgcv_residual_cpp_numeric_shadow_expanded_v1"),
    oracle_dir = file.path(shadow_dir, "oracle"),
    output_dir = file.path("fastkpc", "artifacts",
                           "mgcv_residual_cpp_numeric_drift_isolation_v1"),
    alpha = 0.1,
    index = 1,
    numCol = NULL,
    residual_tol = 1e-5,
    include_all = FALSE,
    env = fastkpc_legacy_env(),
    tol = sqrt(.Machine$double.eps)) {
  if (is.null(data)) data <- fastkpc_mgcv_oracle_default_data()
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  shadow_cases <- fastkpc_drift_iso_read_shadow(shadow_dir)
  oracle <- fastkpc_drift_iso_read_oracle(oracle_dir)
  residuals <- oracle$residuals
  if (length(residuals) == 0L) {
    stop("oracle residuals.rds contains no residual entries", call. = FALSE)
  }
  if (is.null(numCol)) numCol <- floor(length(residuals[[1L]]$residual_x) / 10)
  residual_by_id <- setNames(residuals, vapply(residuals, `[[`,
                                              character(1), "case_id"))
  selected <- shadow_cases
  if (!isTRUE(include_all)) {
    selected <- selected[
      !selected$residual_pair_match |
        !selected$dcov_p_match |
        !selected$decision_match |
        !selected$setup_supported,
      , drop = FALSE
    ]
  }
  case_rows <- list()
  target_rows <- list()
  for (i in seq_len(nrow(selected))) {
    case <- selected[i, , drop = FALSE]
    case_id <- as.character(case$case_id[[1L]])
    if (!case_id %in% names(residual_by_id)) {
      stop("oracle residual missing for case: ", case_id, call. = FALSE)
    }
    entry <- residual_by_id[[case_id]]
    solved_x <- fastkpc_drift_iso_solve_target(
      data, case, entry, suffix = "x", target_col = 1L, env = env, tol = tol
    )
    solved_y <- fastkpc_drift_iso_solve_target(
      data, case, entry, suffix = "y", target_col = 2L, env = env, tol = tol
    )
    target_rows[[length(target_rows) + 1L]] <-
      fastkpc_drift_iso_target_row(
        case, "x", 1L, solved_x, residual_tol = residual_tol
      )
    target_rows[[length(target_rows) + 1L]] <-
      fastkpc_drift_iso_target_row(
        case, "y", 2L, solved_y, residual_tol = residual_tol
      )
    case_rows[[length(case_rows) + 1L]] <- fastkpc_drift_iso_case_row(
      case, entry, solved_x, solved_y, alpha = alpha, index = index,
      numCol = numCol, env = env, residual_tol = residual_tol
    )
  }

  cases <- if (length(case_rows) > 0L) {
    do.call(rbind, case_rows)
  } else {
    data.frame()
  }
  targets <- if (length(target_rows) > 0L) {
    do.call(rbind, target_rows)
  } else {
    data.frame()
  }
  summary <- data.frame(
    artifact = "mgcv_residual_cpp_numeric_drift_isolation_v1",
    case_count = nrow(cases),
    target_count = nrow(targets),
    cpp_matches_r_normal_target_count =
      if (nrow(targets)) sum(targets$cpp_matches_r_normal) else 0L,
    cmagic_matches_oracle_target_count =
      if (nrow(targets)) sum(targets$cmagic_matches_oracle) else 0L,
    cpp_matches_cmagic_target_count =
      if (nrow(targets)) sum(targets$cpp_matches_cmagic) else 0L,
    decision_match_count =
      if (nrow(cases)) sum(cases$decision_match) else 0L,
    decision_flip_count =
      if (nrow(cases)) sum(cases$decision_flip) else 0L,
    normal_equation_vs_mgcv_magic_count =
      if (nrow(cases)) {
        sum(cases$drift_layer == "normal_equation_vs_mgcv_magic")
      } else {
        0L
      },
    max_cpp_vs_r_normal_abs_diff =
      if (nrow(targets)) max(targets$cpp_vs_r_normal_max_abs_diff) else NA_real_,
    max_cpp_vs_cmagic_abs_diff =
      if (nrow(targets)) max(targets$cpp_vs_cmagic_max_abs_diff) else NA_real_,
    max_cmagic_vs_oracle_abs_diff =
      if (nrow(targets)) max(targets$cmagic_vs_oracle_max_abs_diff) else NA_real_,
    max_normal_matrix_condition =
      if (nrow(targets)) max(targets$normal_matrix_condition) else NA_real_,
    include_all = isTRUE(include_all),
    residual_tol = residual_tol,
    stringsAsFactors = FALSE
  )

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  paths <- list(
    summary_csv = file.path(output_dir, "summary.csv"),
    cases_csv = file.path(output_dir, "cases.csv"),
    targets_csv = file.path(output_dir, "targets.csv"),
    summary_md = file.path(output_dir, "summary.md")
  )
  utils::write.csv(summary, paths$summary_csv, row.names = FALSE)
  utils::write.csv(cases, paths$cases_csv, row.names = FALSE)
  utils::write.csv(targets, paths$targets_csv, row.names = FALSE)
  writeLines(c(
    "# mgcv Residual C++ Numeric Drift Isolation v1",
    "",
    paste0("- cases: ", summary$case_count[[1L]]),
    paste0("- targets: ", summary$target_count[[1L]]),
    paste0("- C++ matches R normal targets: ",
           summary$cpp_matches_r_normal_target_count[[1L]]),
    paste0("- C_magic matches oracle targets: ",
           summary$cmagic_matches_oracle_target_count[[1L]]),
    paste0("- decision flips: ", summary$decision_flip_count[[1L]]),
    paste0("- normal-equation vs mgcv-magic cases: ",
           summary$normal_equation_vs_mgcv_magic_count[[1L]]),
    paste0("- max C++ vs R normal abs diff: ",
           signif(summary$max_cpp_vs_r_normal_abs_diff[[1L]], 8L)),
    paste0("- max C++ vs C_magic abs diff: ",
           signif(summary$max_cpp_vs_cmagic_abs_diff[[1L]], 8L)),
    paste0("- max C_magic vs oracle abs diff: ",
           signif(summary$max_cmagic_vs_oracle_abs_diff[[1L]], 8L))
  ), paths$summary_md)

  list(summary = summary, cases = cases, targets = targets, paths = paths,
       output_dir = output_dir)
}
