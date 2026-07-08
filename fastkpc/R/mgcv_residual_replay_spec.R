if (!exists("fastkpc_run_mgcv_residual_oracle_trace", mode = "function")) {
  source("fastkpc/R/mgcv_residual_oracle_trace.R")
}

fastkpc_mgcv_replay_rel_l2 <- function(candidate, oracle) {
  candidate <- as.numeric(candidate)
  oracle <- as.numeric(oracle)
  denom <- sqrt(sum(oracle * oracle))
  if (!is.finite(denom) || denom == 0) {
    return(sqrt(sum((candidate - oracle)^2)))
  }
  sqrt(sum((candidate - oracle)^2)) / denom
}

fastkpc_mgcv_replay_read_oracle <- function(oracle_dir) {
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

fastkpc_mgcv_replay_case <- function(data, oracle_case, oracle_residual,
                                     alpha, index, numCol, env,
                                     residual_tol, p_tol) {
  replay_case <- data.frame(
    case_id = oracle_case$case_id[[1L]],
    x = oracle_case$target_x[[1L]],
    y = oracle_case$target_y[[1L]],
    S = gsub("\\|", ",", oracle_case$S_key[[1L]]),
    role = oracle_case$role[[1L]],
    source = if ("source" %in% names(oracle_case)) {
      oracle_case$source[[1L]]
    } else {
      "oracle_trace"
    },
    source_level = if ("source_level" %in% names(oracle_case)) {
      oracle_case$source_level[[1L]]
    } else {
      NA_integer_
    },
    source_pmax = if ("source_pmax" %in% names(oracle_case)) {
      oracle_case$source_pmax[[1L]]
    } else {
      NA_real_
    },
    source_distance_to_alpha =
      if ("source_distance_to_alpha" %in% names(oracle_case)) {
        oracle_case$source_distance_to_alpha[[1L]]
      } else {
        NA_real_
      },
    source_case_rank = if ("source_case_rank" %in% names(oracle_case)) {
      oracle_case$source_case_rank[[1L]]
    } else {
      NA_integer_
    },
    stringsAsFactors = FALSE
  )

  traced <- fastkpc_mgcv_oracle_case(
    data = data, case = replay_case, alpha = alpha, index = index,
    numCol = numCol, env = env
  )
  row <- traced$row
  residual <- traced$residual

  residual_x_max_abs <- max(abs(residual$residual_x -
                                  oracle_residual$residual_x))
  residual_y_max_abs <- max(abs(residual$residual_y -
                                  oracle_residual$residual_y))
  residual_x_rel_l2 <- fastkpc_mgcv_replay_rel_l2(
    residual$residual_x, oracle_residual$residual_x
  )
  residual_y_rel_l2 <- fastkpc_mgcv_replay_rel_l2(
    residual$residual_y, oracle_residual$residual_y
  )
  p_oracle <- as.numeric(oracle_residual$dcov_p_value)
  p_replay <- as.numeric(residual$dcov_p_value)
  decision_oracle <- isTRUE(oracle_residual$decision_at_alpha)
  decision_replay <- isTRUE(residual$decision_at_alpha)
  residual_x_match <- residual_x_max_abs <= residual_tol
  residual_y_match <- residual_y_max_abs <= residual_tol
  p_match <- abs(p_replay - p_oracle) <= p_tol

  data.frame(
    case_id = row$case_id,
    role = row$role,
    target_x = row$target_x,
    target_y = row$target_y,
    S_key = row$S_key,
    S_size = row$S_size,
    formula_route = row$formula_route,
    residual_x_match = residual_x_match,
    residual_y_match = residual_y_match,
    residual_pair_match = isTRUE(residual_x_match && residual_y_match),
    residual_x_max_abs_diff = residual_x_max_abs,
    residual_y_max_abs_diff = residual_y_max_abs,
    residual_x_rel_l2 = residual_x_rel_l2,
    residual_y_rel_l2 = residual_y_rel_l2,
    residual_x_hash_oracle = oracle_case$residual_x_hash,
    residual_x_hash_replay = row$residual_x_hash,
    residual_y_hash_oracle = oracle_case$residual_y_hash,
    residual_y_hash_replay = row$residual_y_hash,
    dcov_p_oracle = p_oracle,
    dcov_p_replay = p_replay,
    dcov_p_abs_diff = abs(p_replay - p_oracle),
    dcov_p_match = p_match,
    decision_oracle = decision_oracle,
    decision_replay = decision_replay,
    decision_match = identical(decision_oracle, decision_replay),
    source = row$source,
    source_level = row$source_level,
    source_pmax = row$source_pmax,
    source_distance_to_alpha = row$source_distance_to_alpha,
    replay_status = if (isTRUE(residual_x_match && residual_y_match &&
                               p_match &&
                               identical(decision_oracle, decision_replay))) {
      "pass"
    } else {
      "mismatch"
    },
    stringsAsFactors = FALSE
  )
}

fastkpc_run_mgcv_residual_replay_spec <- function(
    data = NULL,
    oracle_dir = file.path("fastkpc", "artifacts",
                           "mgcv_residual_oracle_v1"),
    output_dir = file.path("fastkpc", "artifacts",
                           "mgcv_residual_replay_spec_v1"),
    alpha = 0.1,
    index = 1,
    numCol = NULL,
    residual_tol = 1e-10,
    p_tol = 1e-12,
    env = fastkpc_legacy_env()) {
  if (!requireNamespace("mgcv", quietly = TRUE)) {
    stop("mgcv is required for mgcv residual replay spec", call. = FALSE)
  }
  fastkpc_require_legacy_packages("RSpectra")
  if (is.null(data)) data <- fastkpc_mgcv_oracle_default_data()
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  if (is.null(numCol)) numCol <- floor(nrow(data) / 10)

  oracle <- fastkpc_mgcv_replay_read_oracle(oracle_dir)
  cases <- oracle$cases
  residuals <- oracle$residuals
  residual_by_id <- setNames(residuals, vapply(residuals, `[[`,
                                              character(1), "case_id"))

  rows <- list()
  for (i in seq_len(nrow(cases))) {
    case_id <- as.character(cases$case_id[[i]])
    if (!case_id %in% names(residual_by_id)) {
      stop("oracle residual missing for case: ", case_id, call. = FALSE)
    }
    rows[[length(rows) + 1L]] <- fastkpc_mgcv_replay_case(
      data = data,
      oracle_case = cases[i, , drop = FALSE],
      oracle_residual = residual_by_id[[case_id]],
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
    artifact = "mgcv_residual_replay_spec_v1",
    case_count = nrow(case_rows),
    residual_pair_match_count = sum(case_rows$residual_pair_match),
    dcov_p_match_count = sum(case_rows$dcov_p_match),
    decision_match_count = sum(case_rows$decision_match),
    decision_flip_count = sum(!case_rows$decision_match),
    max_residual_x_abs_diff = max(case_rows$residual_x_max_abs_diff),
    max_residual_y_abs_diff = max(case_rows$residual_y_max_abs_diff),
    max_dcov_p_abs_diff = max(case_rows$dcov_p_abs_diff),
    residual_tol = residual_tol,
    p_tol = p_tol,
    pass = all(case_rows$residual_pair_match) &&
      all(case_rows$dcov_p_match) &&
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
    "# mgcv Residual Replay Spec v1",
    "",
    paste0("- cases: ", summary$case_count[[1L]]),
    paste0("- residual matches: ",
           summary$residual_pair_match_count[[1L]]),
    paste0("- dCov p matches: ", summary$dcov_p_match_count[[1L]]),
    paste0("- decision flips: ", summary$decision_flip_count[[1L]]),
    paste0("- max residual x abs diff: ",
           signif(summary$max_residual_x_abs_diff[[1L]], 8L)),
    paste0("- max residual y abs diff: ",
           signif(summary$max_residual_y_abs_diff[[1L]], 8L)),
    paste0("- max dCov p abs diff: ",
           signif(summary$max_dcov_p_abs_diff[[1L]], 8L)),
    paste0("- pass: ", summary$pass[[1L]])
  ), paths$summary_md)

  list(summary = summary, cases = case_rows, paths = paths,
       output_dir = output_dir)
}
