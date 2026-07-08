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
  selected$source_case_rank <- seq_len(nrow(selected))
  selected$case_id <- sprintf(
    "full351_%02d_s%d_x%d_y%d",
    selected$source_case_rank, selected$S_size, selected$x, selected$y
  )
  selected <- selected[, c(
    "case_id", "x", "y", "S", "role", "source", "source_level",
    "source_pmax", "source_distance_to_alpha", "source_case_rank"
  ), drop = FALSE]
  rownames(selected) <- NULL
  selected
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
  list(
    edf = if (!is.null(fit$edf)) as.numeric(fit$edf) else numeric(),
    rank = if (!is.null(fit$rank)) as.integer(fit$rank) else NA_integer_,
    sp = if (!is.null(fit$sp)) as.numeric(fit$sp) else numeric(),
    method = if (!is.null(fit$method)) as.character(fit$method) else "",
    optimizer = if (!is.null(fit$optimizer)) {
      paste(as.character(fit$optimizer), collapse = "|")
    } else {
      ""
    }
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

  meta_x <- fastkpc_mgcv_oracle_fit_metadata(fit_x)
  meta_y <- fastkpc_mgcv_oracle_fit_metadata(fit_y)
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
    decision_at_alpha = isTRUE(p_value > alpha),
    distance_to_alpha = abs(p_value - alpha),
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
    dcov_p_value = p_value,
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
          decision_at_alpha = NA,
          distance_to_alpha = NA_real_,
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
    paste0("- mgcv version: ", summary$mgcv_version[[1L]])
  ), paths$summary_md)

  list(summary = summary, cases = case_rows, residuals = residuals,
       paths = paths, output_dir = output_dir)
}
