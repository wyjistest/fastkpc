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

fastkpc_mgcv_replay_value <- function(row, name, default = NA) {
  if (name %in% names(row)) row[[name]][[1L]] else default
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
    dcov_alpha_oracle = as.numeric(fastkpc_mgcv_replay_value(
      oracle_case, "dcov_alpha", alpha
    )),
    dcov_alpha_replay = row$dcov_alpha,
    dcov_log_alpha_distance_oracle = as.numeric(fastkpc_mgcv_replay_value(
      oracle_case, "dcov_log_alpha_distance", abs(log(p_oracle) - log(alpha))
    )),
    dcov_log_alpha_distance_replay = row$dcov_log_alpha_distance,
    decision_oracle = decision_oracle,
    decision_replay = decision_replay,
    decision_match = identical(decision_oracle, decision_replay),
    conditioning_rank_oracle = as.integer(fastkpc_mgcv_replay_value(
      oracle_case, "conditioning_rank", NA_integer_
    )),
    conditioning_rank_replay = row$conditioning_rank,
    conditioning_rank_deficient_oracle =
      as.logical(fastkpc_mgcv_replay_value(
        oracle_case, "conditioning_rank_deficient", NA
      )),
    conditioning_rank_deficient_replay =
      row$conditioning_rank_deficient,
    conditioning_condition_kappa_oracle =
      as.numeric(fastkpc_mgcv_replay_value(
        oracle_case, "conditioning_condition_kappa", NA_real_
      )),
    conditioning_condition_kappa_replay =
      row$conditioning_condition_kappa,
    near_constant_column_count_oracle =
      as.integer(fastkpc_mgcv_replay_value(
        oracle_case, "near_constant_column_count", NA_integer_
      )),
    near_constant_column_count_replay =
      row$near_constant_column_count,
    target_near_constant_count_oracle =
      as.integer(fastkpc_mgcv_replay_value(
        oracle_case, "target_near_constant_count", NA_integer_
      )),
    target_near_constant_count_replay =
      row$target_near_constant_count,
    lpmatrix_ncol_x_oracle = as.integer(fastkpc_mgcv_replay_value(
      oracle_case, "lpmatrix_ncol_x", NA_integer_
    )),
    lpmatrix_ncol_x_replay = row$lpmatrix_ncol_x,
    lpmatrix_ncol_y_oracle = as.integer(fastkpc_mgcv_replay_value(
      oracle_case, "lpmatrix_ncol_y", NA_integer_
    )),
    lpmatrix_ncol_y_replay = row$lpmatrix_ncol_y,
    lpmatrix_rank_x_oracle = as.integer(fastkpc_mgcv_replay_value(
      oracle_case, "lpmatrix_rank_x", NA_integer_
    )),
    lpmatrix_rank_x_replay = row$lpmatrix_rank_x,
    lpmatrix_rank_y_oracle = as.integer(fastkpc_mgcv_replay_value(
      oracle_case, "lpmatrix_rank_y", NA_integer_
    )),
    lpmatrix_rank_y_replay = row$lpmatrix_rank_y,
    lpmatrix_rank_deficient_x_oracle =
      as.logical(fastkpc_mgcv_replay_value(
        oracle_case, "lpmatrix_rank_deficient_x", NA
      )),
    lpmatrix_rank_deficient_x_replay =
      row$lpmatrix_rank_deficient_x,
    lpmatrix_rank_deficient_y_oracle =
      as.logical(fastkpc_mgcv_replay_value(
        oracle_case, "lpmatrix_rank_deficient_y", NA
      )),
    lpmatrix_rank_deficient_y_replay =
      row$lpmatrix_rank_deficient_y,
    smooth_count_x_oracle = as.integer(fastkpc_mgcv_replay_value(
      oracle_case, "smooth_count_x", NA_integer_
    )),
    smooth_count_x_replay = row$smooth_count_x,
    smooth_count_y_oracle = as.integer(fastkpc_mgcv_replay_value(
      oracle_case, "smooth_count_y", NA_integer_
    )),
    smooth_count_y_replay = row$smooth_count_y,
    smooth_labels_x_oracle = as.character(fastkpc_mgcv_replay_value(
      oracle_case, "smooth_labels_x", ""
    )),
    smooth_labels_x_replay = row$smooth_labels_x,
    smooth_labels_y_oracle = as.character(fastkpc_mgcv_replay_value(
      oracle_case, "smooth_labels_y", ""
    )),
    smooth_labels_y_replay = row$smooth_labels_y,
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
  finite_log_alpha_oracle <- case_rows$dcov_log_alpha_distance_oracle[
    is.finite(case_rows$dcov_log_alpha_distance_oracle)
  ]
  finite_log_alpha_replay <- case_rows$dcov_log_alpha_distance_replay[
    is.finite(case_rows$dcov_log_alpha_distance_replay)
  ]
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
    rank_deficient_case_count =
      sum(case_rows$conditioning_rank_deficient_oracle %in% TRUE |
            case_rows$conditioning_rank_deficient_replay %in% TRUE),
    lpmatrix_rank_deficient_case_count =
      sum(case_rows$lpmatrix_rank_deficient_x_oracle %in% TRUE |
            case_rows$lpmatrix_rank_deficient_y_oracle %in% TRUE |
            case_rows$lpmatrix_rank_deficient_x_replay %in% TRUE |
            case_rows$lpmatrix_rank_deficient_y_replay %in% TRUE),
    near_constant_case_count = sum(
      case_rows$near_constant_column_count_oracle > 0L |
        case_rows$near_constant_column_count_replay > 0L |
        case_rows$target_near_constant_count_oracle > 0L |
        case_rows$target_near_constant_count_replay > 0L,
      na.rm = TRUE
    ),
    min_dcov_log_alpha_distance_oracle =
      if (length(finite_log_alpha_oracle) > 0L) {
        min(finite_log_alpha_oracle)
      } else {
        NA_real_
      },
    min_dcov_log_alpha_distance_replay =
      if (length(finite_log_alpha_replay) > 0L) {
        min(finite_log_alpha_replay)
      } else {
        NA_real_
      },
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
    paste0("- rank-deficient cases: ",
           summary$rank_deficient_case_count[[1L]]),
    paste0("- near-constant cases: ",
           summary$near_constant_case_count[[1L]]),
    paste0("- min oracle log-alpha distance: ",
           signif(summary$min_dcov_log_alpha_distance_oracle[[1L]], 8L)),
    paste0("- pass: ", summary$pass[[1L]])
  ), paths$summary_md)

  list(summary = summary, cases = case_rows, paths = paths,
       output_dir = output_dir)
}
