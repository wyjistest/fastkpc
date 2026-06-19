source("fastkpc/R/fast_kpc.R")

fastkpc_relative_l2_or_na <- function(a, b) {
  if (exists("fastkpc_relative_l2_diff", mode = "function")) {
    return(fastkpc_relative_l2_diff(a, b))
  }
  a <- as.numeric(a)
  b <- as.numeric(b)
  denom <- sqrt(sum(b^2))
  if (denom == 0) return(sqrt(sum((a - b)^2)))
  sqrt(sum((a - b)^2)) / denom
}

fastkpc_precision_first_sepset_key <- function(sepsets) {
  for (i in seq_along(sepsets)) {
    row <- sepsets[[i]]
    for (j in seq_along(row)) {
      value <- as.integer(row[[j]])
      if (length(value) > 0L) {
        return(paste(i, j, paste(value, collapse = "|"), sep = ":"))
      }
    }
  }
  ""
}

fastkpc_precision_log_p_drift <- function(a, b) {
  a <- as.numeric(a)[1L]
  b <- as.numeric(b)[1L]
  if (!is.finite(a) || !is.finite(b) || a <= 0 || b <= 0) return(NA_real_)
  abs(log(a) - log(b))
}

fastkpc_native_cuda_fixed_sp_pair_parity <- function(data, x, y, S,
                                                     fixed_sp) {
  S_data <- as.data.frame(data[, S, drop = FALSE])
  colnames(S_data) <- paste0("s", seq_along(S))
  rhs <- fastkpc_mgcv_regrxons_rhs(
    S_data = S_data,
    S = S,
    formula_class = fastkpc_regrxons_formula_class(S)
  )
  form <- stats::as.formula(paste(".target ~", rhs))
  make_setup <- function(target, sp) {
    fastkpc_mgcv_extract_setup(
      formula = form,
      data = data.frame(.target = as.numeric(data[, target]), S_data),
      sp = sp,
      method = "GCV.Cp",
      target = target,
      S = S,
      bs = "tp"
    )
  }
  setups <- list(
    make_setup(x, fixed_sp[[1L]]),
    make_setup(y, fixed_sp[[2L]])
  )
  cpu <- fastkpc_mgcv_extract_gpu_solve_handle_batch_fixed_sp(
    setups,
    target_ids = c(x, y)
  )
  cuda <- fastkpc_mgcv_extract_gpu_solve_handle_batch_fixed_sp_cuda(
    setups,
    target_ids = c(x, y)
  )
  data.frame(
    artifact_type = "native_cuda_fixed_sp_parity",
    x = as.integer(x),
    y = as.integer(y),
    S_key = fastkpc_precision_S_key(S),
    fixed_sp_x = as.numeric(fixed_sp[[1L]]),
    fixed_sp_y = as.numeric(fixed_sp[[2L]]),
    coefficient_rel_l2_x =
      fastkpc_relative_l2_or_na(cuda$coefficients[[1L]],
                                cpu$coefficients[[1L]]),
    coefficient_rel_l2_y =
      fastkpc_relative_l2_or_na(cuda$coefficients[[2L]],
                                cpu$coefficients[[2L]]),
    fitted_rel_l2_x =
      fastkpc_relative_l2_or_na(cuda$fitted[, 1L], cpu$fitted[, 1L]),
    fitted_rel_l2_y =
      fastkpc_relative_l2_or_na(cuda$fitted[, 2L], cpu$fitted[, 2L]),
    residual_rel_l2_x =
      fastkpc_relative_l2_or_na(cuda$residuals[, 1L], cpu$residuals[, 1L]),
    residual_rel_l2_y =
      fastkpc_relative_l2_or_na(cuda$residuals[, 2L], cpu$residuals[, 2L]),
    setup_fingerprint_x = as.character(cpu$setup_fingerprints[[1L]]),
    setup_fingerprint_y = as.character(cpu$setup_fingerprints[[2L]]),
    stringsAsFactors = FALSE
  )
}

fastkpc_json_number <- function(value) {
  value <- as.numeric(value)[1L]
  if (!is.finite(value)) return("null")
  format(signif(value, 8), scientific = FALSE, trim = TRUE)
}

fastkpc_json_bool <- function(value) {
  if (isTRUE(value)) return("true")
  if (identical(value, FALSE)) return("false")
  "null"
}

fastkpc_max_finite_or_na <- function(values) {
  values <- as.numeric(values)
  values <- values[is.finite(values)]
  if (length(values) == 0L) return(NA_real_)
  max(values)
}

fastkpc_write_precision_parity_summary <- function(paths, rows, fixed_sp,
                                                   spectral, compat) {
  json_path <- paths$summary_json
  md_path <- paths$summary_md
  fixed_sp_residual_max <- fastkpc_max_finite_or_na(c(
    fixed_sp$residual_rel_l2_x,
    fixed_sp$residual_rel_l2_y
  ))
  json_lines <- c(
    "{",
    paste0('  "legacy_artifact": "', basename(paths$legacy), '",'),
    paste0('  "native_cuda_fixed_sp_parity": "', basename(paths$native_cuda_fixed_sp_parity), '",'),
    paste0('  "spectral_cpu_vs_cuda_solve_parity": "', basename(paths$spectral_cpu_vs_cuda_solve_parity), '",'),
    paste0('  "mgcv_vs_spectral_gcv_compatibility": "', basename(paths$mgcv_vs_spectral_gcv_compatibility), '",'),
    paste0('  "fixed_sp_residual_rel_l2_max": ',
           fastkpc_json_number(fixed_sp_residual_max), ","),
    paste0('  "compatibility_log_p_drift": ',
           fastkpc_json_number(compat$log_p_drift[[1L]]), ","),
    paste0('  "compatibility_decision_flip": ',
           fastkpc_json_bool(compat$decision_flip[[1L]])),
    "}"
  )
  writeLines(json_lines, json_path)
  md <- c(
    "# Native CUDA Precision Parity Summary",
    "",
    "## Fixed-sp CUDA Gate",
    "",
    paste0("- CSV: `", basename(paths$native_cuda_fixed_sp_parity), "`"),
    paste0("- Max residual rel-L2: ",
           fastkpc_json_number(fixed_sp_residual_max)),
    "",
    "## Spectral CPU/CUDA Gate",
    "",
    paste0("- CSV: `", basename(paths$spectral_cpu_vs_cuda_solve_parity), "`"),
    paste0("- Selected grid x/y: ", spectral$selected_grid_index_x[[1L]],
           " / ", spectral$selected_grid_index_y[[1L]]),
    "",
    "## Legacy compatibility Gate",
    "",
    paste0("- CSV: `", basename(paths$mgcv_vs_spectral_gcv_compatibility), "`"),
    paste0("- log-p drift: ",
           fastkpc_json_number(compat$log_p_drift[[1L]])),
    paste0("- decision flip: ", compat$decision_flip[[1L]])
  )
  writeLines(md, md_path)
}

fastkpc_run_native_cuda_precision_parity <- function(
    data,
    x = 1L,
    y = 2L,
    S = 3L,
    alpha = 0.05,
    sp_grid = exp(seq(log(1e-4), log(1e4), length.out = 17L)),
    ci_method = "dcc.gamma",
    index = 1,
    legacy_index = TRUE,
    hsic_params = list(),
    permutation_params = list(),
    output_dir = file.path("fastkpc", "artifacts", "native_cuda_precision")) {
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  x <- as.integer(x)
  y <- as.integer(y)
  S <- as.integer(S)
  if (length(S) == 0L) {
    stop("native CUDA precision parity requires non-empty S", call. = FALSE)
  }
  if (!exists("fastkpc_cuda_available", mode = "function") ||
      !isTRUE(tryCatch(fastkpc_cuda_available(), error = function(e) FALSE))) {
    stop("CUDA unavailable for native precision parity", call. = FALSE)
  }

  S_data <- as.data.frame(data[, S, drop = FALSE])
  colnames(S_data) <- paste0("s", seq_along(S))
  Y <- cbind(x = data[, x], y = data[, y])
  cpu_batch <- fastkpc_mgcv_extract_batch(
    Y = Y,
    S_data = S_data,
    S = S,
    target_ids = c(x, y),
    formula_class = "full-smooth"
  )
  gpu_pair <- fastkpc_mgcv_extract_gpu_gcv_for_pair(
    data = data,
    x = x,
    y = y,
    S = S,
    sp_grid = sp_grid
  )

  cpu_ci <- fastkpc_precision_ci_from_residuals(
    cpu_batch$residuals[, 1L], cpu_batch$residuals[, 2L],
    ci_method = ci_method, index = index, legacy_index = legacy_index,
    hsic_params = hsic_params, permutation_params = permutation_params
  )
  gpu_ci <- fastkpc_precision_ci_from_residuals(
    gpu_pair$residuals[, 1L], gpu_pair$residuals[, 2L],
    ci_method = ci_method, index = index, legacy_index = legacy_index,
    hsic_params = hsic_params, permutation_params = permutation_params
  )

  cpu_graph <- fast_kpc(
    data,
    alpha = alpha,
    max_conditioning_size = min(2L, max(1L, length(S))),
    engine = "cpu",
    precision = "compatible",
    graph_stage = "skeleton",
    allow_canary_mgcv_extract = TRUE,
    ci_method = ci_method,
    index = index,
    legacy_index = legacy_index,
    hsic_params = hsic_params,
    permutation_params = permutation_params
  )
  gpu_graph <- fast_kpc(
    data,
    alpha = alpha,
    max_conditioning_size = min(2L, max(1L, length(S))),
    engine = "cuda",
    precision = "compatible",
    graph_stage = "skeleton",
    allow_canary_mgcv_extract = TRUE,
    ci_method = ci_method,
    index = index,
    legacy_index = legacy_index,
    hsic_params = hsic_params,
    permutation_params = permutation_params
  )

  decision_cpu <- fastkpc_resolve_ci_decision(cpu_ci$p.value, alpha = alpha)
  decision_gpu <- fastkpc_resolve_ci_decision(gpu_ci$p.value, alpha = alpha)
  rows <- data.frame(
    scenario_id = "native-cuda-precision-parity",
    x = x,
    y = y,
    S_key = fastkpc_precision_S_key(S),
    cpu_selected_sp_x = as.numeric(cpu_batch$sp[[1L]][1L]),
    cpu_selected_sp_y = as.numeric(cpu_batch$sp[[2L]][1L]),
    gpu_selected_sp_x = as.numeric(gpu_pair$sp[1L]),
    gpu_selected_sp_y = as.numeric(gpu_pair$sp[2L]),
    selected_sp_x = as.numeric(gpu_pair$sp[1L]),
    selected_sp_y = as.numeric(gpu_pair$sp[2L]),
    selected_grid_index_x = as.integer(gpu_pair$selected_grid_index[1L]),
    selected_grid_index_y = as.integer(gpu_pair$selected_grid_index[2L]),
    gcv_score_x = as.numeric(gpu_pair$score[1L]),
    gcv_score_y = as.numeric(gpu_pair$score[2L]),
    edf_x = as.numeric(gpu_pair$edf[1L]),
    edf_y = as.numeric(gpu_pair$edf[2L]),
    coefficient_rel_l2_x = fastkpc_relative_l2_or_na(
      gpu_pair$coefficients[[1L]],
      cpu_batch$coefficients[[1L]]
    ),
    coefficient_rel_l2_y = fastkpc_relative_l2_or_na(
      gpu_pair$coefficients[[2L]],
      cpu_batch$coefficients[[2L]]
    ),
    fitted_rel_l2_x = fastkpc_relative_l2_or_na(gpu_pair$fitted[, 1L],
                                                cpu_batch$fitted[, 1L]),
    fitted_rel_l2_y = fastkpc_relative_l2_or_na(gpu_pair$fitted[, 2L],
                                                cpu_batch$fitted[, 2L]),
    residual_rel_l2_x = fastkpc_relative_l2_or_na(gpu_pair$residuals[, 1L],
                                                  cpu_batch$residuals[, 1L]),
    residual_rel_l2_y = fastkpc_relative_l2_or_na(gpu_pair$residuals[, 2L],
                                                  cpu_batch$residuals[, 2L]),
    ci_statistic_cpu = as.numeric(cpu_ci$statistic %||% cpu_ci$estimate %||% NA_real_),
    ci_statistic_gpu = as.numeric(gpu_ci$statistic %||% gpu_ci$estimate %||% NA_real_),
    p_value_cpu = as.numeric(cpu_ci$p.value),
    p_value_gpu = as.numeric(gpu_ci$p.value),
    decision_cpu = isTRUE(decision_cpu$delete_edge),
    decision_gpu = isTRUE(decision_gpu$delete_edge),
    adjacency_identical = identical(cpu_graph$skeleton$adjacency,
                                    gpu_graph$skeleton$adjacency),
    first_sepset_cpu = fastkpc_precision_first_sepset_key(cpu_graph$skeleton$sepsets),
    first_sepset_gpu = fastkpc_precision_first_sepset_key(gpu_graph$skeleton$sepsets),
    pmax_max_abs_diff = max(abs(cpu_graph$skeleton$pMax -
                                  gpu_graph$skeleton$pMax), na.rm = TRUE),
    stringsAsFactors = FALSE
  )
  rows$first_sepset_identical <- identical(rows$first_sepset_cpu,
                                           rows$first_sepset_gpu)
  rows$edge_decision_identical <- identical(rows$decision_cpu,
                                            rows$decision_gpu)

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  csv_path <- file.path(output_dir, "native_cuda_precision_parity.csv")
  utils::write.csv(rows, csv_path, row.names = FALSE)

  fixed_sp_rows <- fastkpc_native_cuda_fixed_sp_pair_parity(
    data = data,
    x = x,
    y = y,
    S = S,
    fixed_sp = as.numeric(gpu_pair$sp)
  )
  spectral_rows <- data.frame(
    artifact_type = "spectral_cpu_vs_cuda_solve_parity",
    x = x,
    y = y,
    S_key = fastkpc_precision_S_key(S),
    selected_sp_x = as.numeric(gpu_pair$sp[1L]),
    selected_sp_y = as.numeric(gpu_pair$sp[2L]),
    selected_grid_index_x = as.integer(gpu_pair$selected_grid_index[1L]),
    selected_grid_index_y = as.integer(gpu_pair$selected_grid_index[2L]),
    grid_boundary_hit_x = gpu_pair$selected_grid_index[1L] %in%
      c(1L, nrow(gpu_pair$grid$x)),
    grid_boundary_hit_y = gpu_pair$selected_grid_index[2L] %in%
      c(1L, nrow(gpu_pair$grid$y)),
    gcv_score_x = as.numeric(gpu_pair$score[1L]),
    gcv_score_y = as.numeric(gpu_pair$score[2L]),
    residual_rel_l2_x = fixed_sp_rows$residual_rel_l2_x,
    residual_rel_l2_y = fixed_sp_rows$residual_rel_l2_y,
    fitted_rel_l2_x = fixed_sp_rows$fitted_rel_l2_x,
    fitted_rel_l2_y = fixed_sp_rows$fitted_rel_l2_y,
    stringsAsFactors = FALSE
  )
  compat_rows <- data.frame(
    artifact_type = "mgcv_vs_spectral_gcv_compatibility",
    x = x,
    y = y,
    S_key = fastkpc_precision_S_key(S),
    cpu_selected_sp_x = rows$cpu_selected_sp_x,
    cpu_selected_sp_y = rows$cpu_selected_sp_y,
    gpu_selected_sp_x = rows$gpu_selected_sp_x,
    gpu_selected_sp_y = rows$gpu_selected_sp_y,
    selected_grid_index_x = rows$selected_grid_index_x,
    selected_grid_index_y = rows$selected_grid_index_y,
    grid_boundary_hit_x = spectral_rows$grid_boundary_hit_x,
    grid_boundary_hit_y = spectral_rows$grid_boundary_hit_y,
    residual_rel_l2_x = rows$residual_rel_l2_x,
    residual_rel_l2_y = rows$residual_rel_l2_y,
    fitted_rel_l2_x = rows$fitted_rel_l2_x,
    fitted_rel_l2_y = rows$fitted_rel_l2_y,
    log_p_drift = fastkpc_precision_log_p_drift(rows$p_value_cpu,
                                                rows$p_value_gpu),
    decision_flip = !identical(rows$decision_cpu, rows$decision_gpu),
    adjacency_identical = rows$adjacency_identical,
    first_sepset_identical = rows$first_sepset_identical,
    pmax_max_abs_diff = rows$pmax_max_abs_diff,
    stringsAsFactors = FALSE
  )
  paths <- list(
    legacy = csv_path,
    native_cuda_fixed_sp_parity =
      file.path(output_dir, "native_cuda_fixed_sp_parity.csv"),
    spectral_cpu_vs_cuda_solve_parity =
      file.path(output_dir, "spectral_cpu_vs_cuda_solve_parity.csv"),
    mgcv_vs_spectral_gcv_compatibility =
      file.path(output_dir, "mgcv_vs_spectral_gcv_compatibility.csv"),
    summary_json = file.path(output_dir, "native_cuda_goal_summary.json"),
    summary_md = file.path(output_dir, "native_cuda_goal_summary.md")
  )
  utils::write.csv(fixed_sp_rows, paths$native_cuda_fixed_sp_parity,
                   row.names = FALSE)
  utils::write.csv(spectral_rows, paths$spectral_cpu_vs_cuda_solve_parity,
                   row.names = FALSE)
  utils::write.csv(compat_rows, paths$mgcv_vs_spectral_gcv_compatibility,
                   row.names = FALSE)
  fastkpc_write_precision_parity_summary(paths, rows, fixed_sp_rows,
                                         spectral_rows, compat_rows)
  list(
    parity = rows,
    path = csv_path,
    paths = paths,
    fixed_sp_parity = fixed_sp_rows,
    spectral_parity = spectral_rows,
    compatibility = compat_rows,
    cpu_graph = cpu_graph,
    gpu_graph = gpu_graph,
    cpu_batch = cpu_batch,
    gpu_pair = gpu_pair
  )
}
