source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")

fail <- function(message) stop(message, call. = FALSE)

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) != 1L) {
  fail("usage: qualify_exact_primary_fixed_reuse_real_inputs.R OUTPUT")
}
output_path <- arguments[[1L]]
if (file.exists(output_path) || dir.exists(output_path)) {
  fail("exact-primary/fixed reuse qualification output already exists")
}
if (!requireNamespace("digest", quietly = TRUE)) fail("digest is required")
if (!isTRUE(tryCatch(fastkpc_cuda_available(), error = function(error) FALSE))) {
  fail("CUDA is unavailable")
}

environment_names <- c(
  "FASTKPC_DCC_GAMMA_EXACT_PRIMARY_FIXED_REUSE",
  "FASTKPC_DIAG_EXACT_PRIMARY_COEFFICIENT_PARITY",
  "FASTKPC_DIAG_EXACT_PRIMARY_FIXED_REUSE_KEYS",
  "FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES",
  "FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS",
  "FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES"
)
old_environment <- Sys.getenv(environment_names, unset = NA_character_)
on.exit({
  try(full_cuda_ci_one_call_cache_control_native("reset"), silent = TRUE)
  for (index in seq_along(environment_names)) {
    name <- environment_names[[index]]
    value <- old_environment[[index]]
    if (is.na(value)) {
      Sys.unsetenv(name)
    } else {
      do.call(Sys.setenv, stats::setNames(list(value), name))
    }
  }
}, add = TRUE)

datasets <- data.frame(
  shape = c("179x8", "243x9", "300x10"),
  path = c(
    file.path(
      "causalDiscovery", "20221022181603",
      "CD8.c11.Tex.PDCD1-causalDiscoveryInput.RData"
    ),
    file.path(
      "causalDiscovery", "20220418113957",
      "logged-liver-exhausted-allData-causalDiscoveryInput.RData"
    ),
    file.path(
      "causalDiscovery", "20240315113222",
      "CD8.c11.Tex.PDCD1-causalDiscoveryInput.RData"
    )
  ),
  stringsAsFactors = FALSE
)

live_fields <- c(
  "live_device_allocations", "live_device_bytes", "live_streams",
  "live_events"
)
authority_fields <- c(
  "r_callback_count", "legacy_mgcv_fit_count", "legacy_mgcv_setup_count",
  "cpu_residual_solve_count", "cpu_dcov_component_count",
  "cpu_dcov_eigen_or_lowrank_count", "cpu_dcov_pair_stat_count",
  "cpu_gamma_pvalue_count", "cpu_spectra_count", "residual_d2h_bytes",
  "component_d2h_bytes", "unknown_fallback_count",
  "approximate_backend_count"
)
task_fields <- c(
  "level", "task_index", "x", "y", "S_key",
  "canonical_test_order_id", "native_edge_deleted", "native_edge_ignored"
)

normalize_sepsets <- function(value) {
  unname(lapply(unname(value), function(row) {
    unname(lapply(unname(row), function(entry) sort(as.integer(entry))))
  }))
}

run_route <- function(data, reuse) {
  invisible(full_cuda_ci_one_call_cache_control_native("reset"))
  Sys.setenv(
    FASTKPC_DCC_GAMMA_EXACT_PRIMARY_FIXED_REUSE = if (reuse) "1" else "0",
    FASTKPC_DIAG_EXACT_PRIMARY_COEFFICIENT_PARITY = "0",
    FASTKPC_DIAG_EXACT_PRIMARY_FIXED_REUSE_KEYS = "0",
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES = "0",
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS = "0",
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES = "0"
  )
  resources_before <-
    fastkpc_full_cuda_phase35_vertical_resource_snapshot()
  started <- proc.time()[["elapsed"]]
  result <- fastkpc_compatible_cuda_skeleton(
    data = data,
    alpha = 0.1,
    labels = colnames(data),
    options = list(
      route = "full_cuda",
      compatible_cuda_strict = TRUE,
      max_conditioning_size = Inf,
      index = 1,
      numCol = 35L,
      trace_level = "logical",
      ci_method = "dcc.gamma"
    )
  )
  elapsed_sec <- unname(proc.time()[["elapsed"]] - started)
  resources_after <-
    fastkpc_full_cuda_phase35_vertical_resource_snapshot()
  list(
    result = result,
    elapsed_sec = elapsed_sec,
    resource_closed = identical(
      resources_after[live_fields], resources_before[live_fields]
    )
  )
}

rows <- vector("list", nrow(datasets))
details <- vector("list", nrow(datasets))
for (index in seq_len(nrow(datasets))) {
  path <- datasets$path[[index]]
  if (!file.exists(path)) fail(paste0("missing real-input fixture: ", path))
  input_environment <- new.env(parent = emptyenv())
  loaded_names <- load(path, envir = input_environment)
  if (!identical(loaded_names, "causalData")) {
    fail(paste0("unexpected real-input payload: ", path))
  }
  data <- input_environment$causalData
  storage.mode(data) <- "double"
  if (!is.matrix(data) || any(!is.finite(data)) ||
      paste(dim(data), collapse = "x") != datasets$shape[[index]]) {
    fail(paste0("invalid real-input fixture: ", path))
  }

  baseline <- run_route(data, reuse = FALSE)
  candidate <- run_route(data, reuse = TRUE)
  baseline_result <- baseline$result
  candidate_result <- candidate$result
  baseline_tasks <- baseline_result$tasks[
    order(baseline_result$tasks$canonical_test_order_id), , drop = FALSE
  ]
  candidate_tasks <- candidate_result$tasks[
    order(candidate_result$tasks$canonical_test_order_id), , drop = FALSE
  ]
  task_identity_exact <- nrow(candidate_tasks) == nrow(baseline_tasks) &&
    all(vapply(task_fields, function(field) {
      identical(candidate_tasks[[field]], baseline_tasks[[field]])
    }, logical(1L)))
  p_value_bitwise_exact <- identical(
    candidate_tasks$p_used, baseline_tasks$p_used
  )
  p_value_max_abs_error <- if (
    length(candidate_tasks$p_used) == length(baseline_tasks$p_used)
  ) {
    max(abs(
      as.double(candidate_tasks$p_used) - as.double(baseline_tasks$p_used)
    ))
  } else {
    Inf
  }
  pmax_bitwise_exact <- identical(candidate_result$pMax, baseline_result$pMax)
  pmax_finite <- is.finite(candidate_result$pMax) &
    is.finite(baseline_result$pMax)
  pmax_max_abs_error <- if (identical(
    dim(candidate_result$pMax), dim(baseline_result$pMax)
  ) && identical(
    is.na(candidate_result$pMax), is.na(baseline_result$pMax)
  ) && identical(
    is.infinite(candidate_result$pMax), is.infinite(baseline_result$pMax)
  )) {
    if (any(pmax_finite)) max(abs(
      candidate_result$pMax[pmax_finite] -
        baseline_result$pMax[pmax_finite]
    )) else 0
  } else {
    Inf
  }
  graph_exact <- identical(
    unname(candidate_result$adjacency), unname(baseline_result$adjacency)
  ) && identical(
    normalize_sepsets(candidate_result$sepsets),
    normalize_sepsets(baseline_result$sepsets)
  ) && identical(
    as.integer(candidate_result$n.edgetests),
    as.integer(baseline_result$n.edgetests)
  )

  summary <- candidate_result$summary
  baseline_summary <- baseline_result$summary
  authority_clean <- all(vapply(authority_fields, function(field) {
    !is.null(summary[[field]]) && identical(as.numeric(summary[[field]]), 0)
  }, logical(1L)))
  accounting_pass <-
    isTRUE(summary$exact_primary_fixed_reuse_requested) &&
    summary$cuda_multi_penalty_selected_evaluation_reuse_count == 0 &&
    summary$cuda_multi_penalty_selected_evaluation_refresh_count > 0 &&
    summary$exact_primary_fixed_reuse_hit_target_count > 0 &&
    summary$exact_primary_fixed_reuse_fallback_target_count == 0 &&
    summary$exact_primary_fixed_reuse_allocation_count ==
      summary$exact_primary_fixed_reuse_free_count &&
    summary$exact_primary_fixed_reuse_live_device_bytes_after == 0 &&
    baseline_summary$cuda_exact_fixed_evaluator_target_count -
      summary$cuda_exact_fixed_evaluator_target_count ==
        summary$exact_primary_fixed_reuse_hit_target_count
  hard_gate_pass <- task_identity_exact && p_value_bitwise_exact &&
    p_value_max_abs_error == 0 && pmax_bitwise_exact &&
    pmax_max_abs_error == 0 && graph_exact &&
    baseline$resource_closed && candidate$resource_closed &&
    authority_clean && accounting_pass

  rows[[index]] <- data.frame(
    shape = datasets$shape[[index]],
    baseline_sec = baseline$elapsed_sec,
    candidate_sec = candidate$elapsed_sec,
    speedup = baseline$elapsed_sec / candidate$elapsed_sec,
    task_count = nrow(candidate_tasks),
    task_identity_exact = task_identity_exact,
    p_value_bitwise_exact = p_value_bitwise_exact,
    p_value_max_abs_error = p_value_max_abs_error,
    pmax_bitwise_exact = pmax_bitwise_exact,
    pmax_max_abs_error = pmax_max_abs_error,
    graph_exact = graph_exact,
    baseline_resource_closed = baseline$resource_closed,
    candidate_resource_closed = candidate$resource_closed,
    authority_clean = authority_clean,
    selected_refresh_target_count =
      summary$cuda_multi_penalty_selected_evaluation_refresh_count,
    reuse_hit_target_count =
      summary$exact_primary_fixed_reuse_hit_target_count,
    reuse_fallback_target_count =
      summary$exact_primary_fixed_reuse_fallback_target_count,
    baseline_exact_fixed_target_count =
      baseline_summary$cuda_exact_fixed_evaluator_target_count,
    candidate_exact_fixed_target_count =
      summary$cuda_exact_fixed_evaluator_target_count,
    peak_device_bytes = summary$exact_primary_fixed_reuse_peak_device_bytes,
    allocation_count = summary$exact_primary_fixed_reuse_allocation_count,
    free_count = summary$exact_primary_fixed_reuse_free_count,
    live_device_bytes_after =
      summary$exact_primary_fixed_reuse_live_device_bytes_after,
    accounting_pass = accounting_pass,
    hard_gate_pass = hard_gate_pass,
    stringsAsFactors = FALSE
  )
  details[[index]] <- list(
    path = normalizePath(path), baseline = baseline, candidate = candidate
  )
}

rows <- do.call(rbind, rows)
receipt <- list(
  schema_version = "exact-primary-fixed-reuse-real-input-qualification-v2",
  created_at = format(Sys.time(), tz = "UTC", usetz = TRUE),
  git_head = system2("git", c("rev-parse", "HEAD"), stdout = TRUE),
  native_sha256 = unname(digest::digest(
    file = "fastkpc/build/fastkpc_cuda.so", algo = "sha256",
    serialize = FALSE
  )),
  rows = rows,
  summary = list(
    fixture_count = nrow(rows),
    hard_gate_pass_count = sum(rows$hard_gate_pass),
    all_hard_gates_pass = all(rows$hard_gate_pass),
    maximum_p_value_error = max(rows$p_value_max_abs_error),
    maximum_pmax_error = max(rows$pmax_max_abs_error),
    maximum_peak_device_bytes = max(rows$peak_device_bytes)
  ),
  details = details
)
if (!isTRUE(receipt$summary$all_hard_gates_pass)) {
  fail("exact-primary/fixed reuse real-input qualification failed")
}
output_directory <- dirname(output_path)
if (!dir.exists(output_directory)) {
  dir.create(output_directory, recursive = TRUE, showWarnings = FALSE)
}
saveRDS(receipt, output_path, version = 2)
print(rows)
cat("PASS exact-primary/fixed reuse real-input qualification\n")
