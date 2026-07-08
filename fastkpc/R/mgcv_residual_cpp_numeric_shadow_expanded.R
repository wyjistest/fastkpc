if (!exists("fastkpc_mgcv_oracle_expanded_cases_from_skeleton_result",
            mode = "function")) {
  source("fastkpc/R/mgcv_residual_oracle_trace.R")
}
if (!exists("fastkpc_run_mgcv_residual_setup_shadow", mode = "function")) {
  source("fastkpc/R/mgcv_residual_setup_shadow.R")
}

fastkpc_run_mgcv_residual_cpp_numeric_shadow_expanded <- function(
    data = NULL,
    source_result = NULL,
    source_result_path = fastkpc_mgcv_oracle_default_result_path(),
    output_dir = file.path("fastkpc", "artifacts",
                           "mgcv_residual_cpp_numeric_shadow_expanded_v1"),
    alpha = 0.1,
    near_alpha_count = 12L,
    per_s_size_count = 6L,
    per_level_count = 4L,
    max_cases = 48L,
    index = 1,
    numCol = NULL,
    residual_tol = 1e-5,
    p_tol = 1e-5,
    solver = c("cpp", "cpp_guarded"),
    condition_threshold = 1e12,
    env = fastkpc_legacy_env()) {
  solver <- match.arg(solver)
  if (is.null(data)) data <- fastkpc_mgcv_oracle_default_data()
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  if (is.null(source_result)) {
    if (!nzchar(source_result_path) || !file.exists(source_result_path)) {
      stop("expanded C++ numeric shadow requires a source skeleton result",
           call. = FALSE)
    }
    source_result <- readRDS(source_result_path)
  }

  cases <- fastkpc_mgcv_oracle_expanded_cases_from_skeleton_result(
    source_result,
    alpha = alpha,
    near_alpha_count = near_alpha_count,
    per_s_size_count = per_s_size_count,
    per_level_count = per_level_count,
    max_cases = max_cases
  )

  oracle_dir <- file.path(output_dir, "oracle")
  oracle <- fastkpc_run_mgcv_residual_oracle_trace(
    data = data,
    cases = cases,
    alpha = alpha,
    output_dir = oracle_dir,
    index = index,
    numCol = numCol,
    source_result_path = "",
    env = env
  )
  artifact <- fastkpc_run_mgcv_residual_setup_shadow(
    data = data,
    oracle_dir = oracle_dir,
    output_dir = output_dir,
    alpha = alpha,
    index = index,
    numCol = numCol,
    residual_tol = residual_tol,
    p_tol = p_tol,
    solver = solver,
    condition_threshold = condition_threshold,
    artifact_name = if (identical(solver, "cpp_guarded")) {
      "mgcv_residual_cpp_numeric_shadow_guarded_expanded_v1"
    } else {
      "mgcv_residual_cpp_numeric_shadow_expanded_v1"
    },
    env = env
  )
  artifact$oracle <- oracle
  artifact$selected_cases <- cases
  artifact
}
