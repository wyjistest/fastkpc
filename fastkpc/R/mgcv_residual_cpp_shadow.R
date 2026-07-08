if (!exists("fastkpc_run_mgcv_residual_oracle_trace", mode = "function")) {
  source("fastkpc/R/mgcv_residual_oracle_trace.R")
}
if (!exists("fastkpc_mgcv_residual_replay_from_setup_cpp", mode = "function")) {
  source("fastkpc/R/native.R")
}

fastkpc_mgcv_cpp_shadow_read_oracle <- function(oracle_dir) {
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

fastkpc_mgcv_cpp_shadow_supported <- function(entry) {
  !is.null(entry$model_matrix_x) &&
    !is.null(entry$model_matrix_y) &&
    !is.null(entry$coefficients_x) &&
    !is.null(entry$coefficients_y) &&
    !is.null(entry$fitted_x) &&
    !is.null(entry$fitted_y)
}

fastkpc_mgcv_cpp_shadow_response <- function(entry, suffix) {
  name <- paste0("response_", suffix)
  if (!is.null(entry[[name]])) return(entry[[name]])
  entry[[paste0("fitted_", suffix)]] + entry[[paste0("residual_", suffix)]]
}

fastkpc_mgcv_cpp_shadow_rel_l2 <- function(candidate, oracle) {
  candidate <- as.numeric(candidate)
  oracle <- as.numeric(oracle)
  denom <- sqrt(sum(oracle * oracle))
  if (!is.finite(denom) || denom == 0) {
    return(sqrt(sum((candidate - oracle)^2)))
  }
  sqrt(sum((candidate - oracle)^2)) / denom
}

fastkpc_mgcv_cpp_shadow_case <- function(case, entry, alpha, index, numCol,
                                         env, residual_tol, p_tol) {
  supported <- fastkpc_mgcv_cpp_shadow_supported(entry)
  if (!isTRUE(supported)) {
    return(data.frame(
      case_id = case$case_id,
      role = case$role,
      target_x = case$target_x,
      target_y = case$target_y,
      S_key = case$S_key,
      S_size = case$S_size,
      formula_route = case$formula_route,
      cpp_supported = FALSE,
      authoritative = FALSE,
      backend_family = "mgcvCapturedCppReplay",
      backend_version = "captured-setup-matvec-v1",
      residual_x_match = FALSE,
      residual_y_match = FALSE,
      residual_pair_match = FALSE,
      residual_x_max_abs_diff = NA_real_,
      residual_y_max_abs_diff = NA_real_,
      residual_x_rel_l2 = NA_real_,
      residual_y_rel_l2 = NA_real_,
      dcov_p_oracle = entry$dcov_p_value,
      dcov_p_cpp = NA_real_,
      dcov_p_abs_diff = NA_real_,
      dcov_p_match = FALSE,
      decision_oracle = isTRUE(entry$decision_at_alpha),
      decision_cpp = NA,
      decision_match = FALSE,
      decision_flip = TRUE,
      replay_status = "unsupported",
      stringsAsFactors = FALSE
    ))
  }

  x_replay <- fastkpc_mgcv_residual_replay_from_setup_cpp(
    entry$model_matrix_x, entry$coefficients_x,
    fastkpc_mgcv_cpp_shadow_response(entry, "x")
  )
  y_replay <- fastkpc_mgcv_residual_replay_from_setup_cpp(
    entry$model_matrix_y, entry$coefficients_y,
    fastkpc_mgcv_cpp_shadow_response(entry, "y")
  )

  residual_x <- as.numeric(x_replay$residuals)
  residual_y <- as.numeric(y_replay$residuals)
  residual_x_abs <- max(abs(residual_x - entry$residual_x))
  residual_y_abs <- max(abs(residual_y - entry$residual_y))
  dcov <- fastkpc_legacy_dcov_gamma_timed(
    residual_x, residual_y, index = index, numCol = numCol, env = env
  )
  p_cpp <- as.numeric(dcov$result$p.value)
  p_oracle <- as.numeric(entry$dcov_p_value)
  decision_oracle <- isTRUE(entry$decision_at_alpha)
  decision_cpp <- isTRUE(p_cpp > alpha)
  residual_x_match <- residual_x_abs <= residual_tol
  residual_y_match <- residual_y_abs <= residual_tol
  p_match <- abs(p_cpp - p_oracle) <= p_tol
  decision_match <- identical(decision_oracle, decision_cpp)

  data.frame(
    case_id = case$case_id,
    role = case$role,
    target_x = case$target_x,
    target_y = case$target_y,
    S_key = case$S_key,
    S_size = case$S_size,
    formula_route = case$formula_route,
    cpp_supported = TRUE,
    authoritative = FALSE,
    backend_family = x_replay$backend_family,
    backend_version = x_replay$backend_version,
    residual_x_match = residual_x_match,
    residual_y_match = residual_y_match,
    residual_pair_match = isTRUE(residual_x_match && residual_y_match),
    residual_x_max_abs_diff = residual_x_abs,
    residual_y_max_abs_diff = residual_y_abs,
    residual_x_rel_l2 =
      fastkpc_mgcv_cpp_shadow_rel_l2(residual_x, entry$residual_x),
    residual_y_rel_l2 =
      fastkpc_mgcv_cpp_shadow_rel_l2(residual_y, entry$residual_y),
    dcov_p_oracle = p_oracle,
    dcov_p_cpp = p_cpp,
    dcov_p_abs_diff = abs(p_cpp - p_oracle),
    dcov_p_match = p_match,
    decision_oracle = decision_oracle,
    decision_cpp = decision_cpp,
    decision_match = decision_match,
    decision_flip = !decision_match,
    replay_status = if (isTRUE(residual_x_match && residual_y_match &&
                               decision_match)) {
      "pass"
    } else {
      "mismatch"
    },
    stringsAsFactors = FALSE
  )
}

fastkpc_run_mgcv_residual_cpp_shadow <- function(
    oracle_dir = file.path("fastkpc", "artifacts",
                           "mgcv_residual_oracle_v1"),
    output_dir = file.path("fastkpc", "artifacts",
                           "mgcv_residual_cpp_shadow_v1"),
    alpha = 0.1,
    index = 1,
    numCol = NULL,
    residual_tol = 1e-8,
    p_tol = 1e-9,
    env = fastkpc_legacy_env()) {
  oracle <- fastkpc_mgcv_cpp_shadow_read_oracle(oracle_dir)
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
    rows[[length(rows) + 1L]] <- fastkpc_mgcv_cpp_shadow_case(
      case = cases[i, , drop = FALSE],
      entry = residual_by_id[[case_id]],
      alpha = alpha,
      index = index,
      numCol = numCol,
      env = env,
      residual_tol = residual_tol,
      p_tol = p_tol
    )
  }

  case_rows <- do.call(rbind, rows)
  summary <- data.frame(
    artifact = "mgcv_residual_cpp_shadow_v1",
    case_count = nrow(case_rows),
    cpp_supported_count = sum(case_rows$cpp_supported),
    residual_pair_match_count = sum(case_rows$residual_pair_match),
    dcov_p_match_count = sum(case_rows$dcov_p_match),
    decision_match_count = sum(case_rows$decision_match),
    decision_flip_count = sum(case_rows$decision_flip),
    max_residual_x_abs_diff =
      max(case_rows$residual_x_max_abs_diff, na.rm = TRUE),
    max_residual_y_abs_diff =
      max(case_rows$residual_y_max_abs_diff, na.rm = TRUE),
    max_dcov_p_abs_diff = max(case_rows$dcov_p_abs_diff, na.rm = TRUE),
    residual_tol = residual_tol,
    p_tol = p_tol,
    authoritative = FALSE,
    pass = all(case_rows$cpp_supported) &&
      all(case_rows$residual_pair_match) &&
      all(case_rows$decision_match),
    stringsAsFactors = FALSE
  )

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  paths <- list(
    summary_csv = file.path(output_dir, "summary.csv"),
    cases_csv = file.path(output_dir, "cases.csv"),
    summary_md = file.path(output_dir, "summary.md")
  )
  utils::write.csv(summary, paths$summary_csv, row.names = FALSE)
  utils::write.csv(case_rows, paths$cases_csv, row.names = FALSE)
  writeLines(c(
    "# mgcv Residual C++ Shadow v1",
    "",
    paste0("- cases: ", summary$case_count[[1L]]),
    paste0("- C++ supported: ", summary$cpp_supported_count[[1L]]),
    paste0("- residual matches: ",
           summary$residual_pair_match_count[[1L]]),
    paste0("- strict dCov p matches: ",
           summary$dcov_p_match_count[[1L]]),
    paste0("- decision flips: ", summary$decision_flip_count[[1L]]),
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
