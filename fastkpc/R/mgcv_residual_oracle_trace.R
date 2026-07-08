if (!exists("fastkpc_hash_object", mode = "function")) {
  source("fastkpc/R/mgcv_compat_contract.R")
}
if (!exists("fastkpc_legacy_env", mode = "function")) {
  source("fastkpc/R/legacy_runner.R")
}

fastkpc_mgcv_oracle_parse_s <- function(value) {
  if (is.null(value) || length(value) == 0L || is.na(value[[1L]])) {
    return(integer())
  }
  if (is.numeric(value) || is.integer(value)) return(as.integer(value))
  value <- trimws(as.character(value[[1L]]))
  if (!nzchar(value)) return(integer())
  as.integer(strsplit(value, "[,| ]+", perl = TRUE)[[1L]])
}

fastkpc_mgcv_oracle_s_key <- function(S) {
  S <- as.integer(S)
  if (length(S) == 0L) return("")
  paste(S, collapse = "|")
}

fastkpc_mgcv_oracle_formula_route <- function(S) {
  if (length(as.integer(S)) <= 2L) "full_smooth" else "additive_smooth"
}

fastkpc_mgcv_oracle_source_value <- function(case, name, default = NA) {
  if (name %in% names(case)) case[[name]][[1L]] else default
}

fastkpc_mgcv_oracle_formula <- function(target_index, S_size, env) {
  pred <- seq.int(3L, 2L + as.integer(S_size))
  if (S_size <= 2L) {
    env$frml.full.smooth(target_index, pred)
  } else {
    env$frml.additive.smooth(target_index, pred)
  }
}

fastkpc_mgcv_oracle_default_data <- function() {
  path <- file.path(
    "fastkpc", "artifacts", "kpc_tprs_real_zhu",
    "cancer_RD-causalDiscoveryInput.rds"
  )
  if (!file.exists(path)) {
    stop("default mgcv oracle data not found: ", path, call. = FALSE)
  }
  readRDS(path)
}

fastkpc_mgcv_oracle_default_result_path <- function() {
  candidates <- c(
    file.path(
      "fastkpc", "artifacts", "legacy_mgcv_residual_cache_s_affinity_v1",
      "compatible_legacy_cpp_dcov_mgcv_cache_s_affinity_result.rds"
    ),
    file.path(
      "fastkpc", "artifacts",
      "fast_cuda_real_n351_p48_alpha01_compatible_legacy_parallel_v1",
      "compatible_legacy_parallel_result.rds"
    )
  )
  existing <- candidates[file.exists(candidates)]
  if (length(existing) == 0L) "" else existing[[1L]]
}

fastkpc_mgcv_oracle_log_s <- function(entry) {
  s_xy <- entry$S_xy
  if (!is.null(s_xy) && length(s_xy) > 0L) return(as.integer(s_xy))
  s_yx <- entry$S_yx
  if (!is.null(s_yx) && length(s_yx) > 0L) return(as.integer(s_yx))
  integer()
}

fastkpc_mgcv_oracle_cases_from_skeleton_result <- function(
    result, alpha = 0.1, near_alpha_count = 6L) {
  cases <- fastkpc_mgcv_oracle_all_cases_from_skeleton_result(
    result, alpha = alpha
  )

  pick_one <- function(predicate, role) {
    subset <- cases[predicate(cases), , drop = FALSE]
    if (nrow(subset) == 0L) return(NULL)
    subset <- subset[order(subset$source_distance_to_alpha,
                           subset$x, subset$y), , drop = FALSE]
    subset <- subset[1L, , drop = FALSE]
    subset$role <- role
    subset
  }

  selected <- list(
    pick_one(
      function(x) x$S_size == 1L,
      "full skeleton |S|=1 near-alpha deletion"
    ),
    pick_one(
      function(x) x$S_size == 2L,
      "full skeleton hot level-2 near-alpha deletion"
    ),
    pick_one(
      function(x) x$S_size > 2L,
      "full skeleton late sparse near-alpha deletion"
    ),
    pick_one(
      function(x) x$S_size == max(x$S_size, na.rm = TRUE),
      "full skeleton largest-S sparse deletion"
    )
  )
  selected <- Filter(Negate(is.null), selected)
  selected <- if (length(selected) > 0L) do.call(rbind, selected) else
    cases[integer(), , drop = FALSE]

  near <- cases[order(cases$source_distance_to_alpha,
                      -cases$S_size, cases$x, cases$y), , drop = FALSE]
  near <- near[seq_len(min(as.integer(near_alpha_count), nrow(near))),
               , drop = FALSE]
  near$role <- paste0("full skeleton near-alpha deletion rank ",
                      seq_len(nrow(near)))

  selected <- rbind(selected, near)
  selected <- selected[!duplicated(paste(selected$x, selected$y, selected$S,
                                        sep = "|")), , drop = FALSE]
  fastkpc_mgcv_oracle_finalize_cases(selected, prefix = "full351")
}

fastkpc_mgcv_oracle_all_cases_from_skeleton_result <- function(
    result, alpha = 0.1) {
  skeleton <- result$skeleton
  if (is.null(skeleton) || is.null(skeleton$per.level.log)) {
    stop("skeleton result must contain skeleton$per.level.log",
         call. = FALSE)
  }
  pMax <- skeleton$pMax
  rows <- list()
  for (level_index in seq_along(skeleton$per.level.log)) {
    level_logs <- skeleton$per.level.log[[level_index]]
    if (length(level_logs) == 0L) next
    source_level <- as.integer(level_index - 1L)
    for (entry in level_logs) {
      S <- fastkpc_mgcv_oracle_log_s(entry)
      if (length(S) == 0L) next
      x <- as.integer(entry$x[[1L]])
      y <- as.integer(entry$y[[1L]])
      p_value <- if (is.matrix(pMax) && x <= nrow(pMax) && y <= ncol(pMax)) {
        as.numeric(pMax[x, y])
      } else {
        NA_real_
      }
      rows[[length(rows) + 1L]] <- data.frame(
        x = x,
        y = y,
        S = paste(S, collapse = ","),
        S_size = length(S),
        source = "full_351x48_skeleton_deletion",
        source_level = source_level,
        source_pmax = p_value,
        source_distance_to_alpha = abs(p_value - alpha),
        stringsAsFactors = FALSE
      )
    }
  }
  if (length(rows) == 0L) {
    stop("skeleton result did not contain conditional deletion cases",
         call. = FALSE)
  }
  cases <- do.call(rbind, rows)
  cases <- cases[order(cases$source_distance_to_alpha, cases$S_size,
                       cases$x, cases$y), , drop = FALSE]
  cases <- cases[!duplicated(paste(cases$x, cases$y, cases$S, sep = "|")),
                 , drop = FALSE]
  rownames(cases) <- NULL
  cases
}

fastkpc_mgcv_oracle_finalize_cases <- function(cases, prefix) {
  cases$source_case_rank <- seq_len(nrow(cases))
  cases$case_id <- sprintf(
    "%s_%02d_s%d_x%d_y%d",
    prefix, cases$source_case_rank, cases$S_size, cases$x, cases$y
  )
  cases <- cases[, c(
    "case_id", "x", "y", "S", "role", "source", "source_level",
    "source_pmax", "source_distance_to_alpha", "source_case_rank"
  ), drop = FALSE]
  rownames(cases) <- NULL
  cases
}

fastkpc_mgcv_oracle_expanded_cases_from_skeleton_result <- function(
    result, alpha = 0.1, near_alpha_count = 12L, per_s_size_count = 6L,
    per_level_count = 4L, max_cases = 48L) {
  cases <- fastkpc_mgcv_oracle_all_cases_from_skeleton_result(
    result, alpha = alpha
  )

  selected <- list()
  add_rows <- function(rows, role_prefix) {
    if (is.null(rows) || nrow(rows) == 0L) return(invisible(NULL))
    rows <- rows[order(rows$source_distance_to_alpha, -rows$S_size,
                       rows$x, rows$y), , drop = FALSE]
    rows$role <- paste0(role_prefix, " rank ", seq_len(nrow(rows)))
    selected[[length(selected) + 1L]] <<- rows
    invisible(NULL)
  }

  near <- cases[order(cases$source_distance_to_alpha,
                      -cases$S_size, cases$x, cases$y), , drop = FALSE]
  add_rows(
    near[seq_len(min(as.integer(near_alpha_count), nrow(near))),
         , drop = FALSE],
    "expanded near-alpha deletion"
  )

  for (s_size in sort(unique(cases$S_size))) {
    rows <- cases[cases$S_size == s_size, , drop = FALSE]
    rows <- rows[order(rows$source_distance_to_alpha, rows$x, rows$y),
                 , drop = FALSE]
    add_rows(
      rows[seq_len(min(as.integer(per_s_size_count), nrow(rows))),
           , drop = FALSE],
      paste0("expanded |S|=", s_size, " deletion")
    )
  }

  for (level in sort(unique(cases$source_level))) {
    rows <- cases[cases$source_level == level, , drop = FALSE]
    rows <- rows[order(rows$source_distance_to_alpha, -rows$S_size,
                       rows$x, rows$y), , drop = FALSE]
    add_rows(
      rows[seq_len(min(as.integer(per_level_count), nrow(rows))),
           , drop = FALSE],
      paste0("expanded level ", level, " deletion")
    )
  }

  max_s <- max(cases$S_size, na.rm = TRUE)
  add_rows(
    cases[cases$S_size == max_s, , drop = FALSE],
    paste0("expanded largest |S|=", max_s, " deletion")
  )

  selected <- do.call(rbind, selected)
  selected <- selected[!duplicated(paste(selected$x, selected$y, selected$S,
                                        sep = "|")), , drop = FALSE]
  if (is.finite(max_cases) && nrow(selected) > as.integer(max_cases)) {
    selected <- selected[seq_len(as.integer(max_cases)), , drop = FALSE]
  }
  fastkpc_mgcv_oracle_finalize_cases(selected, prefix = "expanded351")
}

fastkpc_mgcv_oracle_default_cases <- function(
    data, source_result_path = fastkpc_mgcv_oracle_default_result_path(),
    alpha = 0.1) {
  if (nzchar(source_result_path) && file.exists(source_result_path)) {
    return(fastkpc_mgcv_oracle_cases_from_skeleton_result(
      readRDS(source_result_path), alpha = alpha
    ))
  }

  p <- ncol(data)
  if (p < 7L) {
    stop("default mgcv oracle cases require at least 7 columns",
         call. = FALSE)
  }
  data.frame(
    case_id = c(
      "s1_full_smooth_v1",
      "hot_level_2_full_smooth_v1",
      "late_sparse_additive_v1",
      "additional_additive_v1"
    ),
    x = c(1L, 1L, 2L, min(4L, p - 3L)),
    y = c(2L, min(6L, p), min(7L, p), min(5L, p - 2L)),
    S = c("3", "3,5", "3,4,5", "1,2,3,6"),
    role = c(
      "|S|=1",
      "hot level-2",
      "late sparse |S|>2",
      "representative additive |S|>2"
    ),
    source = "handpicked_seed",
    source_level = c(1L, 2L, 3L, 4L),
    source_pmax = NA_real_,
    source_distance_to_alpha = NA_real_,
    source_case_rank = seq_len(4L),
    stringsAsFactors = FALSE
  )
}

fastkpc_mgcv_oracle_fit_metadata <- function(fit) {
  family <- fit$family
  smooth_labels <- if (!is.null(fit$smooth) && length(fit$smooth) > 0L) {
    vapply(fit$smooth, function(smooth) {
      if (!is.null(smooth$label)) as.character(smooth$label) else ""
    }, character(1))
  } else {
    character()
  }
  list(
    edf = if (!is.null(fit$edf)) as.numeric(fit$edf) else numeric(),
    rank = if (!is.null(fit$rank)) as.integer(fit$rank) else NA_integer_,
    sp = if (!is.null(fit$sp)) as.numeric(fit$sp) else numeric(),
    method = if (!is.null(fit$method)) as.character(fit$method) else "",
    optimizer = if (!is.null(fit$optimizer)) {
      paste(as.character(fit$optimizer), collapse = "|")
    } else {
      ""
    },
    family = if (!is.null(family$family)) as.character(family$family) else "",
    link = if (!is.null(family$link)) as.character(family$link) else "",
    converged = if (!is.null(fit$converged)) isTRUE(fit$converged) else NA,
    smooth_count = length(smooth_labels),
    smooth_labels = paste(smooth_labels, collapse = "|"),
    nobs = suppressWarnings(as.integer(stats::nobs(fit))),
    coef_count = length(stats::coef(fit))
  )
}

fastkpc_mgcv_oracle_numeric_sd <- function(x) {
  value <- suppressWarnings(stats::sd(as.numeric(x)))
  if (is.finite(value)) value else NA_real_
}

fastkpc_mgcv_oracle_matrix_rank <- function(mat) {
  mat <- as.matrix(mat)
  if (ncol(mat) == 0L || nrow(mat) == 0L) return(0L)
  as.integer(qr(mat)$rank)
}

fastkpc_mgcv_oracle_matrix_kappa <- function(mat) {
  mat <- as.matrix(mat)
  if (ncol(mat) == 0L || nrow(mat) == 0L) return(NA_real_)
  rank <- fastkpc_mgcv_oracle_matrix_rank(mat)
  if (rank < ncol(mat)) return(Inf)
  value <- tryCatch(
    suppressWarnings(kappa(mat, exact = FALSE)),
    error = function(e) NA_real_
  )
  as.numeric(value)
}

fastkpc_mgcv_oracle_data_diagnostics <- function(fit_data, S_size) {
  S_size <- as.integer(S_size)
  conditioning <- if (S_size > 0L) {
    as.matrix(fit_data[, seq.int(3L, 2L + S_size), drop = FALSE])
  } else {
    matrix(numeric(), nrow = nrow(fit_data), ncol = 0L)
  }
  conditioning_centered <- if (ncol(conditioning) > 0L) {
    scale(conditioning, center = TRUE, scale = FALSE)
  } else {
    conditioning
  }
  conditioning_sd <- if (ncol(conditioning) > 0L) {
    apply(conditioning, 2L, fastkpc_mgcv_oracle_numeric_sd)
  } else {
    numeric()
  }
  near_threshold <- sqrt(.Machine$double.eps)
  target_sd <- c(
    fastkpc_mgcv_oracle_numeric_sd(fit_data$x1),
    fastkpc_mgcv_oracle_numeric_sd(fit_data$x2)
  )
  conditioning_rank <- fastkpc_mgcv_oracle_matrix_rank(conditioning_centered)
  list(
    conditioning_rank = conditioning_rank,
    conditioning_rank_deficient = conditioning_rank < S_size,
    conditioning_condition_kappa =
      fastkpc_mgcv_oracle_matrix_kappa(conditioning_centered),
    near_constant_column_count =
      sum(!is.finite(conditioning_sd) | conditioning_sd <= near_threshold),
    min_conditioning_sd = if (length(conditioning_sd) > 0L) {
      suppressWarnings(min(conditioning_sd, na.rm = TRUE))
    } else {
      NA_real_
    },
    target_x_sd = target_sd[[1L]],
    target_y_sd = target_sd[[2L]],
    target_near_constant_count =
      sum(!is.finite(target_sd) | target_sd <= near_threshold)
  )
}

fastkpc_mgcv_oracle_case <- function(data, case, alpha, index, numCol, env) {
  x <- as.integer(case$x[[1L]])
  y <- as.integer(case$y[[1L]])
  S <- fastkpc_mgcv_oracle_parse_s(case$S[[1L]])
  if (x < 1L || y < 1L || x > ncol(data) || y > ncol(data)) {
    stop("case target is outside data columns: ", case$case_id[[1L]],
         call. = FALSE)
  }
  if (length(S) == 0L) {
    stop("mgcv residual oracle cases require non-empty S: ",
         case$case_id[[1L]], call. = FALSE)
  }
  if (any(S < 1L | S > ncol(data))) {
    stop("case conditioning set is outside data columns: ",
         case$case_id[[1L]], call. = FALSE)
  }

  X <- data[, c(x, y), drop = FALSE]
  S_data <- data[, S, drop = FALSE]
  fit_data <- data.frame(cbind(X, S_data))
  colnames(fit_data) <- paste0("x", seq_len(ncol(fit_data)))
  data_diag <- fastkpc_mgcv_oracle_data_diagnostics(
    fit_data, S_size = length(S)
  )

  formula_x <- fastkpc_mgcv_oracle_formula(1L, length(S), env)
  formula_y <- fastkpc_mgcv_oracle_formula(2L, length(S), env)
  route <- fastkpc_mgcv_oracle_formula_route(S)

  fit_start <- proc.time()[["elapsed"]]
  fit_x <- mgcv::gam(formula_x, data = fit_data)
  fit_y <- mgcv::gam(formula_y, data = fit_data)
  runtime_ms <- (proc.time()[["elapsed"]] - fit_start) * 1000

  residual_x <- as.numeric(fit_x$residuals)
  residual_y <- as.numeric(fit_y$residuals)
  fitted_x <- as.numeric(stats::fitted(fit_x))
  fitted_y <- as.numeric(stats::fitted(fit_y))
  response_x <- as.numeric(fit_data$x1)
  response_y <- as.numeric(fit_data$x2)
  model_matrix_x <- as.matrix(stats::predict(fit_x, type = "lpmatrix"))
  model_matrix_y <- as.matrix(stats::predict(fit_y, type = "lpmatrix"))
  coefficients_x <- as.numeric(stats::coef(fit_x))
  coefficients_y <- as.numeric(stats::coef(fit_y))

  dcov <- fastkpc_legacy_dcov_gamma_timed(
    x = residual_x, y = residual_y, index = index, numCol = numCol, env = env
  )
  p_value <- as.numeric(dcov$result$p.value)
  log_alpha_distance <- abs(log(p_value) - log(alpha))

  meta_x <- fastkpc_mgcv_oracle_fit_metadata(fit_x)
  meta_y <- fastkpc_mgcv_oracle_fit_metadata(fit_y)
  lpmatrix_rank_x <- fastkpc_mgcv_oracle_matrix_rank(model_matrix_x)
  lpmatrix_rank_y <- fastkpc_mgcv_oracle_matrix_rank(model_matrix_y)
  case_id <- as.character(case$case_id[[1L]])
  role <- if ("role" %in% names(case)) as.character(case$role[[1L]]) else ""
  source <- as.character(fastkpc_mgcv_oracle_source_value(
    case, "source", "manual"
  ))
  source_level <- as.integer(fastkpc_mgcv_oracle_source_value(
    case, "source_level", length(S)
  ))
  source_pmax <- as.numeric(fastkpc_mgcv_oracle_source_value(
    case, "source_pmax", NA_real_
  ))
  source_distance_to_alpha <- as.numeric(fastkpc_mgcv_oracle_source_value(
    case, "source_distance_to_alpha", abs(p_value - alpha)
  ))
  source_case_rank <- as.integer(fastkpc_mgcv_oracle_source_value(
    case, "source_case_rank", NA_integer_
  ))

  row <- data.frame(
    case_id = case_id,
    role = role,
    target_x = x,
    target_y = y,
    S_key = fastkpc_mgcv_oracle_s_key(S),
    S_size = length(S),
    formula_route = route,
    formula_x = paste(deparse(formula_x), collapse = ""),
    formula_y = paste(deparse(formula_y), collapse = ""),
    regrxons_parameters = paste0(
      "family=gaussian_identity;method=mgcv_default;S_order=",
      fastkpc_mgcv_oracle_s_key(S)
    ),
    runtime_ms = runtime_ms,
    dcov_p_value = p_value,
    dcov_alpha = alpha,
    decision_at_alpha = isTRUE(p_value > alpha),
    distance_to_alpha = abs(p_value - alpha),
    dcov_log_alpha_distance = log_alpha_distance,
    conditioning_rank = data_diag$conditioning_rank,
    conditioning_rank_deficient = data_diag$conditioning_rank_deficient,
    conditioning_condition_kappa = data_diag$conditioning_condition_kappa,
    near_constant_column_count = data_diag$near_constant_column_count,
    min_conditioning_sd = data_diag$min_conditioning_sd,
    target_x_sd = data_diag$target_x_sd,
    target_y_sd = data_diag$target_y_sd,
    target_near_constant_count = data_diag$target_near_constant_count,
    source = source,
    source_level = source_level,
    source_pmax = source_pmax,
    source_distance_to_alpha = source_distance_to_alpha,
    source_case_rank = source_case_rank,
    residual_x_hash =
      fastkpc_hash_object(round(residual_x, digits = 14L)),
    residual_y_hash =
      fastkpc_hash_object(round(residual_y, digits = 14L)),
    fitted_x_hash = fastkpc_hash_object(round(fitted_x, digits = 14L)),
    fitted_y_hash = fastkpc_hash_object(round(fitted_y, digits = 14L)),
    model_matrix_x_hash =
      fastkpc_hash_object(round(as.numeric(model_matrix_x), digits = 14L)),
    model_matrix_y_hash =
      fastkpc_hash_object(round(as.numeric(model_matrix_y), digits = 14L)),
    coefficients_x_hash =
      fastkpc_hash_object(round(coefficients_x, digits = 14L)),
    coefficients_y_hash =
      fastkpc_hash_object(round(coefficients_y, digits = 14L)),
    residual_length = length(residual_x),
    edf_x_sum = sum(meta_x$edf),
    edf_y_sum = sum(meta_y$edf),
    rank_x = meta_x$rank,
    rank_y = meta_y$rank,
    sp_x = paste(signif(meta_x$sp, 16L), collapse = "|"),
    sp_y = paste(signif(meta_y$sp, 16L), collapse = "|"),
    mgcv_method_x = meta_x$method,
    mgcv_method_y = meta_y$method,
    mgcv_optimizer_x = meta_x$optimizer,
    mgcv_optimizer_y = meta_y$optimizer,
    mgcv_family_x = meta_x$family,
    mgcv_family_y = meta_y$family,
    mgcv_link_x = meta_x$link,
    mgcv_link_y = meta_y$link,
    mgcv_converged_x = meta_x$converged,
    mgcv_converged_y = meta_y$converged,
    smooth_count_x = meta_x$smooth_count,
    smooth_count_y = meta_y$smooth_count,
    smooth_labels_x = meta_x$smooth_labels,
    smooth_labels_y = meta_y$smooth_labels,
    mgcv_nobs_x = meta_x$nobs,
    mgcv_nobs_y = meta_y$nobs,
    mgcv_coef_count_x = meta_x$coef_count,
    mgcv_coef_count_y = meta_y$coef_count,
    lpmatrix_ncol_x = ncol(model_matrix_x),
    lpmatrix_ncol_y = ncol(model_matrix_y),
    lpmatrix_rank_x = lpmatrix_rank_x,
    lpmatrix_rank_y = lpmatrix_rank_y,
    lpmatrix_rank_deficient_x = lpmatrix_rank_x < ncol(model_matrix_x),
    lpmatrix_rank_deficient_y = lpmatrix_rank_y < ncol(model_matrix_y),
    lpmatrix_condition_kappa_x =
      fastkpc_mgcv_oracle_matrix_kappa(model_matrix_x),
    lpmatrix_condition_kappa_y =
      fastkpc_mgcv_oracle_matrix_kappa(model_matrix_y),
    mgcv_version = as.character(utils::packageVersion("mgcv")),
    R_version = R.version.string,
    error = "",
    stringsAsFactors = FALSE
  )

  residual_entry <- list(
    case_id = case_id,
    target_x = x,
    target_y = y,
    S = S,
    formula_x = formula_x,
    formula_y = formula_y,
    response_x = response_x,
    response_y = response_y,
    residual_x = residual_x,
    residual_y = residual_y,
    fitted_x = fitted_x,
    fitted_y = fitted_y,
    model_matrix_x = model_matrix_x,
    model_matrix_y = model_matrix_y,
    coefficients_x = coefficients_x,
    coefficients_y = coefficients_y,
    fit_metadata_x = meta_x,
    fit_metadata_y = meta_y,
    data_diagnostics = data_diag,
    lpmatrix_rank_x = lpmatrix_rank_x,
    lpmatrix_rank_y = lpmatrix_rank_y,
    lpmatrix_condition_kappa_x =
      fastkpc_mgcv_oracle_matrix_kappa(model_matrix_x),
    lpmatrix_condition_kappa_y =
      fastkpc_mgcv_oracle_matrix_kappa(model_matrix_y),
    dcov_p_value = p_value,
    dcov_alpha = alpha,
    dcov_log_alpha_distance = log_alpha_distance,
    decision_at_alpha = isTRUE(p_value > alpha)
  )
  list(row = row, residual = residual_entry)
}

fastkpc_run_mgcv_residual_oracle_trace <- function(
    data = NULL,
    cases = NULL,
    alpha = 0.1,
    output_dir = file.path("fastkpc", "artifacts", "mgcv_residual_oracle_v1"),
    index = 1,
    numCol = NULL,
    source_result_path = fastkpc_mgcv_oracle_default_result_path(),
    env = fastkpc_legacy_env()) {
  if (!requireNamespace("mgcv", quietly = TRUE)) {
    stop("mgcv is required for mgcv residual oracle trace", call. = FALSE)
  }
  fastkpc_require_legacy_packages("RSpectra")

  if (is.null(data)) data <- fastkpc_mgcv_oracle_default_data()
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  if (is.null(numCol)) numCol <- floor(nrow(data) / 10)
  if (is.null(cases)) {
    cases <- fastkpc_mgcv_oracle_default_cases(
      data, source_result_path = source_result_path, alpha = alpha
    )
  }
  cases <- as.data.frame(cases, stringsAsFactors = FALSE)
  if (!all(c("case_id", "x", "y", "S") %in% names(cases))) {
    stop("cases must contain case_id, x, y, and S columns", call. = FALSE)
  }

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  rows <- list()
  residuals <- list()
  error_count <- 0L
  for (i in seq_len(nrow(cases))) {
    traced <- tryCatch(
      fastkpc_mgcv_oracle_case(
        data = data, case = cases[i, , drop = FALSE], alpha = alpha,
        index = index, numCol = numCol, env = env
      ),
      error = function(e) {
        error_count <<- error_count + 1L
        data.frame(
          case_id = as.character(cases$case_id[[i]]),
          role = if ("role" %in% names(cases)) {
            as.character(cases$role[[i]])
          } else {
            ""
          },
          target_x = as.integer(cases$x[[i]]),
          target_y = as.integer(cases$y[[i]]),
          S_key = as.character(cases$S[[i]]),
          S_size = length(fastkpc_mgcv_oracle_parse_s(cases$S[[i]])),
          formula_route = "",
          formula_x = "",
          formula_y = "",
          regrxons_parameters = "",
          runtime_ms = NA_real_,
          dcov_p_value = NA_real_,
          dcov_alpha = alpha,
          decision_at_alpha = NA,
          distance_to_alpha = NA_real_,
          dcov_log_alpha_distance = NA_real_,
          conditioning_rank = NA_integer_,
          conditioning_rank_deficient = NA,
          conditioning_condition_kappa = NA_real_,
          near_constant_column_count = NA_integer_,
          min_conditioning_sd = NA_real_,
          target_x_sd = NA_real_,
          target_y_sd = NA_real_,
          target_near_constant_count = NA_integer_,
          source = if ("source" %in% names(cases)) {
            as.character(cases$source[[i]])
          } else {
            "manual"
          },
          source_level = if ("source_level" %in% names(cases)) {
            as.integer(cases$source_level[[i]])
          } else {
            NA_integer_
          },
          source_pmax = if ("source_pmax" %in% names(cases)) {
            as.numeric(cases$source_pmax[[i]])
          } else {
            NA_real_
          },
          source_distance_to_alpha =
            if ("source_distance_to_alpha" %in% names(cases)) {
              as.numeric(cases$source_distance_to_alpha[[i]])
            } else {
              NA_real_
            },
          source_case_rank = if ("source_case_rank" %in% names(cases)) {
            as.integer(cases$source_case_rank[[i]])
          } else {
            NA_integer_
          },
          residual_x_hash = "",
          residual_y_hash = "",
          fitted_x_hash = "",
          fitted_y_hash = "",
          model_matrix_x_hash = "",
          model_matrix_y_hash = "",
          coefficients_x_hash = "",
          coefficients_y_hash = "",
          residual_length = NA_integer_,
          edf_x_sum = NA_real_,
          edf_y_sum = NA_real_,
          rank_x = NA_integer_,
          rank_y = NA_integer_,
          sp_x = "",
          sp_y = "",
          mgcv_method_x = "",
          mgcv_method_y = "",
          mgcv_optimizer_x = "",
          mgcv_optimizer_y = "",
          mgcv_family_x = "",
          mgcv_family_y = "",
          mgcv_link_x = "",
          mgcv_link_y = "",
          mgcv_converged_x = NA,
          mgcv_converged_y = NA,
          smooth_count_x = NA_integer_,
          smooth_count_y = NA_integer_,
          smooth_labels_x = "",
          smooth_labels_y = "",
          mgcv_nobs_x = NA_integer_,
          mgcv_nobs_y = NA_integer_,
          mgcv_coef_count_x = NA_integer_,
          mgcv_coef_count_y = NA_integer_,
          lpmatrix_ncol_x = NA_integer_,
          lpmatrix_ncol_y = NA_integer_,
          lpmatrix_rank_x = NA_integer_,
          lpmatrix_rank_y = NA_integer_,
          lpmatrix_rank_deficient_x = NA,
          lpmatrix_rank_deficient_y = NA,
          lpmatrix_condition_kappa_x = NA_real_,
          lpmatrix_condition_kappa_y = NA_real_,
          mgcv_version = as.character(utils::packageVersion("mgcv")),
          R_version = R.version.string,
          error = conditionMessage(e),
          stringsAsFactors = FALSE
        )
      }
    )
    if (is.data.frame(traced)) {
      rows[[length(rows) + 1L]] <- traced
    } else {
      rows[[length(rows) + 1L]] <- traced$row
      residuals[[length(residuals) + 1L]] <- traced$residual
    }
  }

  case_rows <- do.call(rbind, rows)
  finite_condition_kappa <- case_rows$conditioning_condition_kappa[
    is.finite(case_rows$conditioning_condition_kappa)
  ]
  finite_log_alpha_distance <- case_rows$dcov_log_alpha_distance[
    is.finite(case_rows$dcov_log_alpha_distance)
  ]
  summary <- data.frame(
    artifact = "mgcv_residual_oracle_v1",
    case_count = nrow(case_rows),
    success_count = sum(!nzchar(case_rows$error)),
    error_count = error_count,
    n = nrow(data),
    p = ncol(data),
    alpha = alpha,
    numCol = numCol,
    s_size_1_count = sum(case_rows$S_size == 1L),
    s_size_2_count = sum(case_rows$S_size == 2L),
    s_size_gt2_count = sum(case_rows$S_size > 2L),
    full_smooth_count = sum(case_rows$formula_route == "full_smooth"),
    additive_smooth_count = sum(case_rows$formula_route == "additive_smooth"),
    full_skeleton_source_count =
      sum(case_rows$source == "full_351x48_skeleton_deletion"),
    near_alpha_source_count =
      sum(grepl("near-alpha", case_rows$role, fixed = TRUE)),
    rank_deficient_case_count =
      sum(case_rows$conditioning_rank_deficient %in% TRUE),
    lpmatrix_rank_deficient_case_count =
      sum(case_rows$lpmatrix_rank_deficient_x %in% TRUE |
            case_rows$lpmatrix_rank_deficient_y %in% TRUE),
    near_constant_case_count = sum(
      case_rows$near_constant_column_count > 0L |
        case_rows$target_near_constant_count > 0L,
      na.rm = TRUE
    ),
    max_conditioning_condition_kappa =
      if (length(finite_condition_kappa) > 0L) {
        max(finite_condition_kappa)
      } else if (any(is.infinite(case_rows$conditioning_condition_kappa))) {
        Inf
      } else {
        NA_real_
      },
    min_dcov_log_alpha_distance =
      if (length(finite_log_alpha_distance) > 0L) {
        min(finite_log_alpha_distance)
      } else {
        NA_real_
      },
    max_runtime_ms = suppressWarnings(max(case_rows$runtime_ms, na.rm = TRUE)),
    mgcv_version = as.character(utils::packageVersion("mgcv")),
    R_version = R.version.string,
    stringsAsFactors = FALSE
  )
  if (!is.finite(summary$max_runtime_ms[[1L]])) {
    summary$max_runtime_ms[[1L]] <- NA_real_
  }

  paths <- list(
    summary_csv = file.path(output_dir, "summary.csv"),
    cases_csv = file.path(output_dir, "cases.csv"),
    residuals_rds = file.path(output_dir, "residuals.rds"),
    summary_md = file.path(output_dir, "summary.md")
  )
  utils::write.csv(summary, paths$summary_csv, row.names = FALSE)
  utils::write.csv(case_rows, paths$cases_csv, row.names = FALSE)
  saveRDS(residuals, paths$residuals_rds)
  writeLines(c(
    "# mgcv Residual Oracle v1",
    "",
    paste0("- cases: ", summary$case_count[[1L]]),
    paste0("- successes: ", summary$success_count[[1L]]),
    paste0("- errors: ", summary$error_count[[1L]]),
    paste0("- n / p: ", summary$n[[1L]], " / ", summary$p[[1L]]),
    paste0("- alpha: ", summary$alpha[[1L]]),
    paste0("- |S|=1 cases: ", summary$s_size_1_count[[1L]]),
    paste0("- |S|=2 cases: ", summary$s_size_2_count[[1L]]),
    paste0("- |S|>2 cases: ", summary$s_size_gt2_count[[1L]]),
    paste0("- full smooth cases: ", summary$full_smooth_count[[1L]]),
    paste0("- additive smooth cases: ",
           summary$additive_smooth_count[[1L]]),
    paste0("- full skeleton source cases: ",
           summary$full_skeleton_source_count[[1L]]),
    paste0("- near-alpha source cases: ",
           summary$near_alpha_source_count[[1L]]),
    paste0("- rank-deficient conditioning cases: ",
           summary$rank_deficient_case_count[[1L]]),
    paste0("- near-constant data cases: ",
           summary$near_constant_case_count[[1L]]),
    paste0("- min log-alpha distance: ",
           signif(summary$min_dcov_log_alpha_distance[[1L]], 8L)),
    paste0("- mgcv version: ", summary$mgcv_version[[1L]])
  ), paths$summary_md)

  list(summary = summary, cases = case_rows, residuals = residuals,
       paths = paths, output_dir = output_dir)
}
