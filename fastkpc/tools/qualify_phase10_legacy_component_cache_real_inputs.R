source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")

fail <- function(message) stop(message, call. = FALSE)
arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) > 1L) {
  fail("usage: qualify_phase10_legacy_component_cache_real_inputs.R [OUTPUT]")
}
output_path <- if (length(arguments)) arguments[[1L]] else file.path(
  "fastkpc", "artifacts", "full_cuda_ci",
  "phase10_legacy_component_cache_v1", "real_input_qualification.rds"
)
if (file.exists(output_path) || dir.exists(output_path)) {
  fail("legacy component cache qualification output already exists")
}
if (!requireNamespace("digest", quietly = TRUE)) fail("digest is required")
if (!isTRUE(tryCatch(fastkpc_cuda_available(), error = function(error) FALSE))) {
  fail("CUDA is unavailable")
}

cache_environment <- "FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES"
bypass_environment <-
  "FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS"
lane_environment <- "FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES"
old_cache_budget <- Sys.getenv(cache_environment, unset = NA_character_)
old_bypass <- Sys.getenv(bypass_environment, unset = NA_character_)
old_lanes <- Sys.getenv(lane_environment, unset = NA_character_)
on.exit({
  try(full_cuda_ci_one_call_cache_control_native("reset"), silent = TRUE)
  if (is.na(old_cache_budget)) {
    Sys.unsetenv(cache_environment)
  } else {
    do.call(Sys.setenv, setNames(list(old_cache_budget), cache_environment))
  }
  if (is.na(old_bypass)) {
    Sys.unsetenv(bypass_environment)
  } else {
    do.call(Sys.setenv, setNames(list(old_bypass), bypass_environment))
  }
  if (is.na(old_lanes)) {
    Sys.unsetenv(lane_environment)
  } else {
    do.call(Sys.setenv, setNames(list(old_lanes), lane_environment))
  }
}, add = TRUE)

datasets <- data.frame(
  label = c("179x8", "243x9", "300x10"),
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
cache_budget_bytes <- 128 * 1024^2
qualification_bypass_raw <- Sys.getenv(
  "FASTKPC_QUALIFY_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS", unset = "0"
)
if (!qualification_bypass_raw %in% c("0", "1")) {
  fail("cache residual-bypass qualification flag must be 0 or 1")
}
qualification_residual_bypass <- identical(qualification_bypass_raw, "1")
qualification_lane_raw <- Sys.getenv(
  "FASTKPC_QUALIFY_LEGACY_COMPONENT_EIG_LANES", unset = "0"
)
if (!qualification_lane_raw %in% c("0", "1", "2", "4", "8")) {
  fail("cache eig-lane qualification count must be 0, 1, 2, 4, or 8")
}
qualification_eig_lanes <- as.integer(qualification_lane_raw)
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
run_candidate <- function(data, cache_bytes, residual_bypass = FALSE,
                          eig_lanes = 0L) {
  full_cuda_ci_one_call_cache_control_native("reset")
  Sys.setenv(FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES =
               as.character(cache_bytes))
  Sys.setenv(FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS =
               if (isTRUE(residual_bypass)) "1" else "0")
  Sys.setenv(FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES =
               as.character(eig_lanes))
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

details <- vector("list", nrow(datasets))
rows <- vector("list", nrow(datasets))
for (index in seq_len(nrow(datasets))) {
  path <- datasets$path[[index]]
  input_environment <- new.env(parent = emptyenv())
  loaded_names <- load(path, envir = input_environment)
  if (!identical(loaded_names, "causalData")) {
    fail(paste0("unexpected causalDiscovery payload in ", path))
  }
  data <- input_environment$causalData
  storage.mode(data) <- "double"
  if (!is.matrix(data) || any(!is.finite(data)) ||
      paste(dim(data), collapse = "x") != datasets$label[[index]]) {
    fail(paste0("invalid real-input fixture ", datasets$label[[index]]))
  }

  uncached <- run_candidate(data, 0)
  cached <- run_candidate(
    data, cache_budget_bytes,
    residual_bypass = qualification_residual_bypass,
    eig_lanes = qualification_eig_lanes
  )
  uncached_result <- uncached$result
  cached_result <- cached$result
  uncached_tasks <- uncached_result$tasks[
    order(uncached_result$tasks$canonical_test_order_id), , drop = FALSE
  ]
  cached_tasks <- cached_result$tasks[
    order(cached_result$tasks$canonical_test_order_id), , drop = FALSE
  ]
  task_identity_exact <- all(vapply(task_fields, function(field) {
    identical(cached_tasks[[field]], uncached_tasks[[field]])
  }, logical(1L)))
  p_value_bitwise_exact <- identical(
    cached_tasks$p_used, uncached_tasks$p_used
  )
  pmax_bitwise_exact <- identical(cached_result$pMax, uncached_result$pMax)
  p_value_max_abs_error <- max(abs(
    as.numeric(cached_tasks$p_used) - as.numeric(uncached_tasks$p_used)
  ))
  pmax_same_na <- identical(
    is.na(cached_result$pMax), is.na(uncached_result$pMax)
  )
  pmax_finite <- is.finite(cached_result$pMax) &
    is.finite(uncached_result$pMax)
  pmax_max_abs_error <- if (any(pmax_finite)) max(abs(
    as.numeric(cached_result$pMax[pmax_finite]) -
      as.numeric(uncached_result$pMax[pmax_finite])
  )) else 0
  numerical_contract_pass <- is.finite(p_value_max_abs_error) &&
    p_value_max_abs_error <= 1e-10 && pmax_same_na &&
    is.finite(pmax_max_abs_error) && pmax_max_abs_error <= 1e-10
  graph_exact <- identical(
    unname(cached_result$adjacency), unname(uncached_result$adjacency)
  ) && identical(
    normalize_sepsets(cached_result$sepsets),
    normalize_sepsets(uncached_result$sepsets)
  ) && identical(
    as.integer(cached_result$n.edgetests),
    as.integer(uncached_result$n.edgetests)
  )
  summary <- cached_result$summary
  uncached_summary <- uncached_result$summary
  authority_zero <- vapply(authority_fields, function(field) {
    !is.null(summary[[field]]) && identical(as.numeric(summary[[field]]), 0)
  }, logical(1L))
  cache_accounted <-
    summary$legacy_component_cache_requested_budget_bytes ==
      cache_budget_bytes &&
    summary$legacy_component_cache_device_bytes <= cache_budget_bytes &&
    summary$legacy_component_cache_request_count ==
      summary$legacy_component_cache_hit_count +
        summary$legacy_component_cache_miss_count &&
    summary$legacy_component_cache_miss_count ==
      summary$legacy_component_cache_insert_count +
        summary$legacy_component_cache_bypass_count &&
    summary$legacy_component_cache_bypass_count == 0 &&
    summary$legacy_component_cache_route_mismatch_count == 0 &&
    summary$legacy_component_cache_status_mismatch_count == 0 &&
    identical(
      isTRUE(summary$legacy_component_cache_residual_bypass_requested),
      qualification_residual_bypass
    ) &&
    summary$legacy_component_cache_residual_bypass_batch_count ==
      (if (qualification_residual_bypass) {
        summary$legacy_component_cache_all_hit_batch_count
      } else 0) &&
    summary$legacy_component_cache_residual_bypass_target_count >=
      summary$legacy_component_cache_residual_bypass_batch_count &&
    summary$legacy_component_eig_lane_count_requested ==
      qualification_eig_lanes &&
    ((qualification_eig_lanes == 0L &&
      summary$legacy_component_eig_lane_count == 0 &&
      summary$legacy_component_eig_active_lane_peak == 0 &&
      summary$legacy_component_eig_multilane_batch_count == 0 &&
      summary$legacy_component_execution_context_reuse_count == 0 &&
      summary$legacy_component_execution_context_device_bytes == 0) ||
     (qualification_eig_lanes > 0L &&
      summary$legacy_component_eig_lane_count ==
        min(qualification_eig_lanes, ncol(data)) &&
      summary$legacy_component_eig_active_lane_peak >= 1 &&
      summary$legacy_component_eig_active_lane_peak <=
        summary$legacy_component_eig_lane_count &&
      summary$legacy_component_execution_context_reuse_count ==
        summary$legacy_component_cache_batch_count &&
      summary$legacy_component_execution_context_device_bytes > 0)) &&
    summary$legacy_component_cache_entries ==
      summary$legacy_component_cache_insert_count &&
    summary$legacy_component_cache_hit_count > 0 &&
    summary$legacy_component_cache_request_count ==
      uncached_summary$cuda_dcov_component_count &&
    summary$cuda_dcov_component_count ==
      summary$legacy_component_cache_miss_count &&
    summary$cuda_dcov_component_count <
      uncached_summary$cuda_dcov_component_count
  route_gate <- identical(
    summary$ci_backend, "legacy-compatible-full-eig-cuda-v1"
  ) && identical(summary$strict_gamma_residual_route,
                 "exact-fixed-stable-svd") &&
    summary$cuda_legacy_compatible_pair_count ==
      summary$physical_tests_evaluated &&
    summary$legacy_compatible_pair_count == summary$physical_tests_evaluated &&
    summary$guarded_pair_count == 0 &&
    summary$cuda_exact_screen_pair_count == 0 &&
    summary$cuda_guard_refinement_pair_count == 0
  pass <- task_identity_exact && numerical_contract_pass &&
    graph_exact && all(authority_zero) &&
    cache_accounted && route_gate && uncached$resource_closed &&
    cached$resource_closed && isTRUE(summary$authority_gate_pass)

  rows[[index]] <- data.frame(
    label = datasets$label[[index]], n = nrow(data), p = ncol(data),
    ci_tests = nrow(cached_tasks),
    uncached_elapsed_sec = uncached$elapsed_sec,
    cached_elapsed_sec = cached$elapsed_sec,
    speedup = uncached$elapsed_sec / cached$elapsed_sec,
    component_requests = summary$legacy_component_cache_request_count,
    cache_hits = summary$legacy_component_cache_hit_count,
    cache_misses = summary$legacy_component_cache_miss_count,
    uncached_full_eig_count = uncached_summary$cuda_dcov_component_count,
    cached_full_eig_count = summary$cuda_dcov_component_count,
    uncached_full_eig_ms = uncached_summary$cuda_dcov_full_eig_ms,
    cached_full_eig_ms = summary$cuda_dcov_full_eig_ms,
    residual_bypass = qualification_residual_bypass,
    eig_lanes = qualification_eig_lanes,
    eig_active_lane_peak = summary$legacy_component_eig_active_lane_peak,
    eig_multilane_batches =
      summary$legacy_component_eig_multilane_batch_count,
    residual_bypass_batches =
      summary$legacy_component_cache_residual_bypass_batch_count,
    uncached_physical_residual_fits =
      uncached_summary$physical_residual_fits,
    cached_physical_residual_fits = summary$physical_residual_fits,
    p_value_bitwise_exact = p_value_bitwise_exact,
    p_value_max_abs_error = p_value_max_abs_error,
    pmax_bitwise_exact = pmax_bitwise_exact,
    pmax_max_abs_error = pmax_max_abs_error,
    pass = pass,
    stringsAsFactors = FALSE
  )
  details[[index]] <- list(
    label = datasets$label[[index]],
    input_path = normalizePath(path, winslash = "/", mustWork = TRUE),
    input_sha256 = unname(digest::digest(
      file = path, algo = "sha256", serialize = FALSE
    )),
    task_identity_exact = task_identity_exact,
    p_value_bitwise_exact = p_value_bitwise_exact,
    p_value_max_abs_error = p_value_max_abs_error,
    pmax_bitwise_exact = pmax_bitwise_exact,
    pmax_max_abs_error = pmax_max_abs_error,
    numerical_contract_pass = numerical_contract_pass,
    graph_exact = graph_exact,
    authority_zero = authority_zero,
    cache_accounted = cache_accounted,
    route_gate = route_gate,
    uncached_resource_closed = uncached$resource_closed,
    cached_resource_closed = cached$resource_closed,
    pass = pass
  )
  cat(sprintf(
    paste0(
      "%s %s cache qualification; residual_bypass=%s; eig_lanes=%d; ",
      "ci=%d; hits=%d/%d; ",
      "max_diff=%.17g; %.3fs -> %.3fs\n"
    ),
    if (pass) "PASS" else "FAIL", datasets$label[[index]],
    qualification_residual_bypass,
    qualification_eig_lanes,
    nrow(cached_tasks), as.integer(summary$legacy_component_cache_hit_count),
    as.integer(summary$legacy_component_cache_request_count),
    p_value_max_abs_error,
    uncached$elapsed_sec, cached$elapsed_sec
  ))
  if (!pass) {
    gates <- c(
      task_identity_exact = task_identity_exact,
      numerical_contract_pass = numerical_contract_pass,
      p_value_bitwise_exact = p_value_bitwise_exact,
      pmax_bitwise_exact = pmax_bitwise_exact,
      graph_exact = graph_exact,
      authority_zero = all(authority_zero),
      cache_accounted = cache_accounted,
      route_gate = route_gate,
      uncached_resource_closed = uncached$resource_closed,
      cached_resource_closed = cached$resource_closed,
      authority_gate_pass = isTRUE(summary$authority_gate_pass)
    )
    print(gates)
    fail(paste0("cache qualification failed for ", path))
  }
}

native_path <- normalizePath(
  "fastkpc/build/fastkpc_cuda.so", winslash = "/", mustWork = TRUE
)
receipt <- list(
  schema_version =
    "full-cuda-ci-phase10-legacy-component-cache-real-input-v1",
  generated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  cache_budget_bytes = cache_budget_bytes,
  residual_bypass = qualification_residual_bypass,
  eig_lanes = qualification_eig_lanes,
  native_path = native_path,
  native_sha256 = unname(digest::digest(
    file = native_path, algo = "sha256", serialize = FALSE
  )),
  cases = do.call(rbind, rows),
  details = details,
  pass = all(vapply(details, function(value) isTRUE(value$pass), logical(1L)))
)
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
temporary_path <- paste0(output_path, ".tmp")
on.exit(unlink(temporary_path, force = TRUE), add = TRUE)
saveRDS(receipt, temporary_path, compress = "xz", version = 3L)
if (!file.rename(temporary_path, output_path)) fail("failed to publish receipt")
cat("PASS real-input legacy component cache qualification; output=",
    output_path, "\n", sep = "")
