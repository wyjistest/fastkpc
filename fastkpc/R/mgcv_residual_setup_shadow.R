if (!exists("fastkpc_run_mgcv_residual_oracle_trace", mode = "function")) {
  source("fastkpc/R/mgcv_residual_oracle_trace.R")
}
if (!exists("fastkpc_mgcv_extract_setup", mode = "function")) {
  source("fastkpc/R/mgcv_extract_oracle.R")
}

fastkpc_mgcv_setup_shadow_read_oracle <- function(oracle_dir) {
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

fastkpc_mgcv_setup_shadow_rel_l2 <- function(candidate, oracle) {
  candidate <- as.numeric(candidate)
  oracle <- as.numeric(oracle)
  denom <- sqrt(sum(oracle * oracle))
  if (!is.finite(denom) || denom == 0) {
    return(sqrt(sum((candidate - oracle)^2)))
  }
  sqrt(sum((candidate - oracle)^2)) / denom
}

fastkpc_mgcv_setup_shadow_case_value <- function(case, name, default = NA) {
  if (name %in% names(case)) case[[name]][[1L]] else default
}

fastkpc_mgcv_setup_shadow_fit_data <- function(data, case) {
  x <- as.integer(case$target_x[[1L]])
  y <- as.integer(case$target_y[[1L]])
  S <- fastkpc_mgcv_oracle_parse_s(case$S_key[[1L]])
  if (x < 1L || y < 1L || x > ncol(data) || y > ncol(data)) {
    stop("case target is outside data columns: ", case$case_id[[1L]],
         call. = FALSE)
  }
  if (length(S) == 0L) {
    stop("setup shadow cases require non-empty S: ",
         case$case_id[[1L]], call. = FALSE)
  }
  if (any(S < 1L | S > ncol(data))) {
    stop("case conditioning set is outside data columns: ",
         case$case_id[[1L]], call. = FALSE)
  }
  fit_data <- data.frame(cbind(
    data[, c(x, y), drop = FALSE],
    data[, S, drop = FALSE]
  ))
  colnames(fit_data) <- paste0("x", seq_len(ncol(fit_data)))
  list(data = fit_data, S = S, S_fit = seq.int(3L, 2L + length(S)))
}

fastkpc_mgcv_setup_shadow_method <- function(entry, suffix) {
  meta <- entry[[paste0("fit_metadata_", suffix)]]
  method <- if (!is.null(meta$method)) as.character(meta$method[[1L]]) else ""
  if (nzchar(method)) method else "GCV.Cp"
}

fastkpc_mgcv_setup_shadow_sp <- function(entry, suffix) {
  meta <- entry[[paste0("fit_metadata_", suffix)]]
  sp <- if (!is.null(meta$sp)) as.numeric(meta$sp) else numeric()
  if (length(sp) == 0L || any(!is.finite(sp)) || any(sp <= 0)) {
    stop("oracle entry does not contain fixed positive sp for ", suffix,
         call. = FALSE)
  }
  sp
}

fastkpc_mgcv_setup_shadow_solve_one <- function(entry, fit_data_info,
                                                target_col, suffix, env,
                                                solver,
                                                condition_threshold,
                                                native_s_size_limit) {
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
  fallback_used <- FALSE
  fallback_reason <- ""
  normal_matrix_condition <- NA_real_
  solution <- if (identical(solver, "cpp")) {
    fastkpc_mgcv_solve_setup_fixed_sp_cpp(setup)
  } else if (identical(solver, "cpp_guarded")) {
    s_size <- length(fit_data_info$S)
    normal_matrix_condition <- fastkpc_mgcv_fixed_sp_normal_matrix_condition(
      setup, sp = setup$sp
    )
    if (is.finite(native_s_size_limit) &&
        s_size > native_s_size_limit) {
      fallback_used <- TRUE
      fallback_reason <- "outside_native_s_size_envelope"
      fastkpc_mgcv_solve_setup_fixed_sp(setup)
    } else if (is.finite(normal_matrix_condition) &&
        normal_matrix_condition > condition_threshold) {
      fallback_used <- TRUE
      fallback_reason <- "high_normal_matrix_condition"
      fastkpc_mgcv_solve_setup_fixed_sp(setup)
    } else {
      fastkpc_mgcv_solve_setup_fixed_sp_cpp(setup)
    }
  } else {
    fastkpc_mgcv_solve_setup_fixed_sp(setup)
  }
  list(
    setup = setup,
    solution = solution,
    normal_matrix_condition = normal_matrix_condition,
    fallback_used = fallback_used,
    fallback_reason = fallback_reason
  )
}

fastkpc_mgcv_setup_shadow_unsupported_row <- function(case, entry,
                                                      message) {
  data.frame(
    case_id = case$case_id,
    role = case$role,
    source = fastkpc_mgcv_setup_shadow_case_value(case, "source", ""),
    source_level =
      fastkpc_mgcv_setup_shadow_case_value(case, "source_level", NA_integer_),
    source_pmax =
      fastkpc_mgcv_setup_shadow_case_value(case, "source_pmax", NA_real_),
    source_distance_to_alpha =
      fastkpc_mgcv_setup_shadow_case_value(
        case, "source_distance_to_alpha", NA_real_
      ),
    target_x = case$target_x,
    target_y = case$target_y,
    S_key = case$S_key,
    S_size = case$S_size,
    formula_route = case$formula_route,
    solver = "",
    setup_supported = FALSE,
    authoritative = FALSE,
    backend_family_x = "",
    backend_family_y = "",
    mode_x = "",
    mode_y = "",
    solve_source_x = "",
    solve_source_y = "",
    fallback_used_x = FALSE,
    fallback_used_y = FALSE,
    fallback_reason_x = "",
    fallback_reason_y = "",
    normal_matrix_condition_x = NA_real_,
    normal_matrix_condition_y = NA_real_,
    residual_x_match = FALSE,
    residual_y_match = FALSE,
    residual_pair_match = FALSE,
    residual_x_max_abs_diff = NA_real_,
    residual_y_max_abs_diff = NA_real_,
    residual_x_rel_l2 = NA_real_,
    residual_y_rel_l2 = NA_real_,
    dcov_p_oracle = as.numeric(entry$dcov_p_value),
    dcov_p_setup = NA_real_,
    dcov_p_abs_diff = NA_real_,
    dcov_p_match = FALSE,
    decision_oracle = isTRUE(entry$decision_at_alpha),
    decision_setup = NA,
    decision_match = FALSE,
    decision_flip = TRUE,
    setup_fingerprint_x = "",
    setup_fingerprint_y = "",
    setup_status = "unsupported",
    error = as.character(message),
    stringsAsFactors = FALSE
  )
}

fastkpc_mgcv_setup_shadow_case <- function(data, case, entry, alpha, index,
                                           numCol, env, residual_tol, p_tol,
                                           solver, condition_threshold,
                                           native_s_size_limit) {
  solved <- tryCatch({
    fit_data_info <- fastkpc_mgcv_setup_shadow_fit_data(data, case)
    list(
      x = fastkpc_mgcv_setup_shadow_solve_one(
        entry, fit_data_info, target_col = 1L, suffix = "x", env = env,
        solver = solver, condition_threshold = condition_threshold,
        native_s_size_limit = native_s_size_limit
      ),
      y = fastkpc_mgcv_setup_shadow_solve_one(
        entry, fit_data_info, target_col = 2L, suffix = "y", env = env,
        solver = solver, condition_threshold = condition_threshold,
        native_s_size_limit = native_s_size_limit
      )
    )
  }, error = function(e) e)
  if (inherits(solved, "error")) {
    return(fastkpc_mgcv_setup_shadow_unsupported_row(
      case = case, entry = entry, message = conditionMessage(solved)
    ))
  }

  residual_x <- as.numeric(solved$x$solution$residuals)
  residual_y <- as.numeric(solved$y$solution$residuals)
  residual_x_abs <- max(abs(residual_x - entry$residual_x))
  residual_y_abs <- max(abs(residual_y - entry$residual_y))
  dcov <- fastkpc_legacy_dcov_gamma_timed(
    residual_x, residual_y, index = index, numCol = numCol, env = env
  )
  p_setup <- as.numeric(dcov$result$p.value)
  p_oracle <- as.numeric(entry$dcov_p_value)
  decision_oracle <- isTRUE(entry$decision_at_alpha)
  decision_setup <- isTRUE(p_setup > alpha)
  residual_x_match <- residual_x_abs <= residual_tol
  residual_y_match <- residual_y_abs <= residual_tol
  p_match <- abs(p_setup - p_oracle) <= p_tol
  decision_match <- identical(decision_oracle, decision_setup)

  data.frame(
    case_id = case$case_id,
    role = case$role,
    source = fastkpc_mgcv_setup_shadow_case_value(case, "source", ""),
    source_level =
      fastkpc_mgcv_setup_shadow_case_value(case, "source_level", NA_integer_),
    source_pmax =
      fastkpc_mgcv_setup_shadow_case_value(case, "source_pmax", NA_real_),
    source_distance_to_alpha =
      fastkpc_mgcv_setup_shadow_case_value(
        case, "source_distance_to_alpha", NA_real_
      ),
    target_x = case$target_x,
    target_y = case$target_y,
    S_key = case$S_key,
    S_size = case$S_size,
    formula_route = case$formula_route,
    solver = solver,
    setup_supported = TRUE,
    authoritative = FALSE,
    backend_family_x = solved$x$solution$backend_family,
    backend_family_y = solved$y$solution$backend_family,
    mode_x = solved$x$solution$mode,
    mode_y = solved$y$solution$mode,
    solve_source_x = solved$x$solution$solve_source,
    solve_source_y = solved$y$solution$solve_source,
    fallback_used_x = isTRUE(solved$x$fallback_used),
    fallback_used_y = isTRUE(solved$y$fallback_used),
    fallback_reason_x = solved$x$fallback_reason,
    fallback_reason_y = solved$y$fallback_reason,
    normal_matrix_condition_x = solved$x$normal_matrix_condition,
    normal_matrix_condition_y = solved$y$normal_matrix_condition,
    residual_x_match = residual_x_match,
    residual_y_match = residual_y_match,
    residual_pair_match = isTRUE(residual_x_match && residual_y_match),
    residual_x_max_abs_diff = residual_x_abs,
    residual_y_max_abs_diff = residual_y_abs,
    residual_x_rel_l2 =
      fastkpc_mgcv_setup_shadow_rel_l2(residual_x, entry$residual_x),
    residual_y_rel_l2 =
      fastkpc_mgcv_setup_shadow_rel_l2(residual_y, entry$residual_y),
    dcov_p_oracle = p_oracle,
    dcov_p_setup = p_setup,
    dcov_p_abs_diff = abs(p_setup - p_oracle),
    dcov_p_match = p_match,
    decision_oracle = decision_oracle,
    decision_setup = decision_setup,
    decision_match = decision_match,
    decision_flip = !decision_match,
    setup_fingerprint_x =
      solved$x$setup$setup_fingerprint$fingerprint %||% "",
    setup_fingerprint_y =
      solved$y$setup$setup_fingerprint$fingerprint %||% "",
    setup_status = if (isTRUE(residual_x_match && residual_y_match &&
                              decision_match)) {
      "pass"
    } else {
      "mismatch"
    },
    error = "",
    stringsAsFactors = FALSE
  )
}

fastkpc_run_mgcv_residual_setup_shadow <- function(
    data = NULL,
    oracle_dir = file.path("fastkpc", "artifacts",
                           "mgcv_residual_oracle_v1"),
    output_dir = file.path("fastkpc", "artifacts",
                           "mgcv_residual_setup_shadow_v1"),
    alpha = 0.1,
    index = 1,
    numCol = NULL,
    residual_tol = 1e-5,
    p_tol = 1e-5,
    solver = c("mgcv_magic", "cpp", "cpp_guarded"),
    condition_threshold = 1e12,
    native_s_size_limit = Inf,
    artifact_name = "mgcv_residual_setup_shadow_v1",
    env = fastkpc_legacy_env()) {
  solver <- match.arg(solver)
  condition_threshold <- as.numeric(condition_threshold)
  if (length(condition_threshold) != 1L ||
      !is.finite(condition_threshold) ||
      condition_threshold < 0) {
    stop("condition_threshold must be a finite nonnegative scalar",
         call. = FALSE)
  }
  native_s_size_limit <- as.numeric(native_s_size_limit)
  if (length(native_s_size_limit) != 1L ||
      is.na(native_s_size_limit) ||
      native_s_size_limit < 0) {
    stop("native_s_size_limit must be a nonnegative scalar or Inf",
         call. = FALSE)
  }
  if (!requireNamespace("mgcv", quietly = TRUE)) {
    stop("mgcv is required for mgcv residual setup shadow", call. = FALSE)
  }
  fastkpc_require_legacy_packages("RSpectra")
  if (is.null(data)) data <- fastkpc_mgcv_oracle_default_data()
  data <- as.matrix(data)
  storage.mode(data) <- "double"

  oracle <- fastkpc_mgcv_setup_shadow_read_oracle(oracle_dir)
  cases <- oracle$cases
  residuals <- oracle$residuals
  if (length(residuals) == 0L) {
    stop("oracle residuals.rds contains no residual entries", call. = FALSE)
  }
  if (is.null(numCol)) numCol <- floor(length(residuals[[1L]]$residual_x) / 10)
  residual_by_id <- setNames(residuals, vapply(residuals, `[[`,
                                              character(1), "case_id"))

  rows <- list()
  for (i in seq_len(nrow(cases))) {
    case_id <- as.character(cases$case_id[[i]])
    if (!case_id %in% names(residual_by_id)) {
      stop("oracle residual missing for case: ", case_id, call. = FALSE)
    }
    rows[[length(rows) + 1L]] <- fastkpc_mgcv_setup_shadow_case(
      data = data,
      case = cases[i, , drop = FALSE],
      entry = residual_by_id[[case_id]],
      alpha = alpha,
      index = index,
      numCol = numCol,
      env = env,
      residual_tol = residual_tol,
      p_tol = p_tol,
      solver = solver,
      condition_threshold = condition_threshold,
      native_s_size_limit = native_s_size_limit
    )
  }

  case_rows <- do.call(rbind, rows)
  normal_conditions <- c(
    case_rows$normal_matrix_condition_x,
    case_rows$normal_matrix_condition_y
  )
  finite_normal_conditions <- normal_conditions[is.finite(normal_conditions)]
  fallback_count <- sum(case_rows$fallback_used_x, na.rm = TRUE) +
    sum(case_rows$fallback_used_y, na.rm = TRUE)
  high_condition_fallback_count <-
    sum(case_rows$fallback_reason_x == "high_normal_matrix_condition",
        na.rm = TRUE) +
    sum(case_rows$fallback_reason_y == "high_normal_matrix_condition",
        na.rm = TRUE)
  outside_envelope_fallback_count <-
    sum(case_rows$fallback_reason_x == "outside_native_s_size_envelope",
        na.rm = TRUE) +
    sum(case_rows$fallback_reason_y == "outside_native_s_size_envelope",
        na.rm = TRUE)
  supported_target_count <- sum(case_rows$setup_supported, na.rm = TRUE) * 2L
  cpp_guarded_count <- if (identical(solver, "cpp_guarded")) {
    supported_target_count - fallback_count
  } else {
    0L
  }
  summary <- data.frame(
    artifact = artifact_name,
    case_count = nrow(case_rows),
    setup_supported_count = sum(case_rows$setup_supported),
    setup_unsupported_count = sum(!case_rows$setup_supported),
    residual_pair_match_count = sum(case_rows$residual_pair_match),
    residual_pair_mismatch_count = sum(!case_rows$residual_pair_match),
    dcov_p_match_count = sum(case_rows$dcov_p_match),
    dcov_p_mismatch_count = sum(!case_rows$dcov_p_match),
    decision_match_count = sum(case_rows$decision_match),
    decision_mismatch_count = sum(!case_rows$decision_match),
    decision_flip_count = sum(case_rows$decision_flip),
    max_residual_x_abs_diff =
      max(case_rows$residual_x_max_abs_diff, na.rm = TRUE),
    max_residual_y_abs_diff =
      max(case_rows$residual_y_max_abs_diff, na.rm = TRUE),
    max_dcov_p_abs_diff = max(case_rows$dcov_p_abs_diff, na.rm = TRUE),
    residual_tol = residual_tol,
    p_tol = p_tol,
    solver = solver,
    condition_threshold = if (identical(solver, "cpp_guarded")) {
      condition_threshold
    } else {
      NA_real_
    },
    native_s_size_limit = if (identical(solver, "cpp_guarded")) {
      native_s_size_limit
    } else {
      NA_real_
    },
    fallback_count = fallback_count,
    high_condition_fallback_count = high_condition_fallback_count,
    outside_envelope_fallback_count = outside_envelope_fallback_count,
    cpp_guarded_count = cpp_guarded_count,
    max_normal_matrix_condition =
      if (length(finite_normal_conditions)) {
        max(finite_normal_conditions)
      } else {
        NA_real_
      },
    authoritative = FALSE,
    pass = all(case_rows$setup_supported) &&
      all(case_rows$residual_pair_match) &&
      all(case_rows$decision_match),
    stringsAsFactors = FALSE
  )
  for (name in c("max_residual_x_abs_diff", "max_residual_y_abs_diff",
                 "max_dcov_p_abs_diff",
                 "max_normal_matrix_condition")) {
    if (!is.finite(summary[[name]][[1L]])) summary[[name]][[1L]] <- NA_real_
  }

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  paths <- list(
    summary_csv = file.path(output_dir, "summary.csv"),
    cases_csv = file.path(output_dir, "cases.csv"),
    summary_md = file.path(output_dir, "summary.md")
  )
  utils::write.csv(summary, paths$summary_csv, row.names = FALSE)
  utils::write.csv(case_rows, paths$cases_csv, row.names = FALSE)
  writeLines(c(
    "# mgcv Residual Setup Shadow v1",
    "",
    paste0("- cases: ", summary$case_count[[1L]]),
    paste0("- solver: ", summary$solver[[1L]]),
    paste0("- setup supported: ", summary$setup_supported_count[[1L]]),
    paste0("- residual matches: ",
           summary$residual_pair_match_count[[1L]]),
    paste0("- strict dCov p matches: ",
           summary$dcov_p_match_count[[1L]]),
    paste0("- decision flips: ", summary$decision_flip_count[[1L]]),
    paste0("- guarded fallback count: ",
           summary$fallback_count[[1L]]),
    paste0("- guarded high-condition fallback count: ",
           summary$high_condition_fallback_count[[1L]]),
    paste0("- guarded outside-envelope fallback count: ",
           summary$outside_envelope_fallback_count[[1L]]),
    paste0("- guarded native C++ target count: ",
           summary$cpp_guarded_count[[1L]]),
    paste0("- max normal matrix condition: ",
           signif(summary$max_normal_matrix_condition[[1L]], 8L)),
    paste0("- max residual x abs diff: ",
           signif(summary$max_residual_x_abs_diff[[1L]], 8L)),
    paste0("- max residual y abs diff: ",
           signif(summary$max_residual_y_abs_diff[[1L]], 8L)),
    paste0("- max dCov p abs diff: ",
           signif(summary$max_dcov_p_abs_diff[[1L]], 8L)),
    paste0("- authoritative: ", summary$authoritative[[1L]]),
    paste0("- pass: ", summary$pass[[1L]])
  ), paths$summary_md)

  list(summary = summary, cases = case_rows, paths = paths,
       output_dir = output_dir)
}
