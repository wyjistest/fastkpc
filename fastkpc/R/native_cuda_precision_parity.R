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
  list(
    parity = rows,
    path = csv_path,
    cpu_graph = cpu_graph,
    gpu_graph = gpu_graph,
    cpu_batch = cpu_batch,
    gpu_pair = gpu_pair
  )
}
