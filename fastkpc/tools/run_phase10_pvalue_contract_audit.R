source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")

fail <- function(message) stop(message, call. = FALSE)
arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) > 1L) {
  fail("usage: run_phase10_pvalue_contract_audit.R [OUTPUT]")
}

output_path <- if (length(arguments)) arguments[[1L]] else file.path(
  "fastkpc", "artifacts", "full_cuda_ci",
  "phase10_pvalue_contract_repair_v1", "canonical_audit.rds"
)
if (file.exists(output_path) || dir.exists(output_path)) {
  fail("Phase 10 p-value audit output already exists")
}
if (!requireNamespace("digest", quietly = TRUE)) fail("digest is required")
if (!isTRUE(tryCatch(fastkpc_cuda_available(), error = function(error) FALSE))) {
  fail("CUDA is unavailable")
}

oracle_dir <- file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "oracle_351x48_default_inf_v2"
)
data_path <- file.path(
  "fastkpc", "artifacts", "kpc_tprs_real_zhu",
  "cancer_RD-causalDiscoveryInput.rds"
)
data <- as.matrix(readRDS(data_path))
storage.mode(data) <- "double"
logical <- readRDS(file.path(oracle_dir, "logical_ci_trace.rds"))
reference_pmax <- readRDS(file.path(oracle_dir, "pmax.rds"))
reference_adjacency <- readRDS(file.path(oracle_dir, "adjacency.rds"))
canonical_deletion_sepsets <- function(tasks, deletion_field) {
  selected <- tasks[as.logical(tasks[[deletion_field]]),
                    c("x", "y", "S_key"), drop = FALSE]
  keys <- paste(pmin(selected$x, selected$y),
                pmax(selected$x, selected$y), sep = "|")
  values <- split(as.character(selected$S_key), keys)
  sort(vapply(values, function(value) {
    unique_value <- unique(value)
    if (length(unique_value) != 1L) {
      fail("one deleted edge has multiple separating sets")
    }
    unique_value[[1L]]
  }, character(1L)))
}

numeric_comparison <- function(candidate, reference, tolerance) {
  candidate <- as.numeric(candidate)
  reference <- as.numeric(reference)
  same_shape <- length(candidate) == length(reference)
  same_na <- same_shape && identical(is.na(candidate), is.na(reference))
  same_infinite <- same_shape && identical(
    is.infinite(candidate), is.infinite(reference)
  ) && identical(
    candidate[is.infinite(candidate)], reference[is.infinite(reference)]
  )
  comparable <- if (same_shape) {
    is.finite(candidate) & is.finite(reference)
  } else {
    logical()
  }
  errors <- if (same_shape) {
    abs(candidate[comparable] - reference[comparable])
  } else {
    Inf
  }
  list(
    same_shape = same_shape,
    same_na = same_na,
    same_infinite = same_infinite,
    comparable_count = sum(comparable),
    mismatch_count = if (same_na && same_infinite) {
      sum(errors > tolerance)
    } else {
      max(length(candidate), length(reference))
    },
    maximum_absolute_error = if (length(errors)) max(errors) else 0,
    pass = same_na && same_infinite && all(errors <= tolerance)
  )
}

invisible(full_cuda_ci_one_call_cache_control_native("configure", 262144L))
invisible(full_cuda_ci_one_call_cache_control_native(
  "configure_target", 131072L
))
invisible(full_cuda_ci_one_call_cache_control_native("reset"))

resource_fields <- c(
  "live_device_allocations", "live_device_bytes", "live_streams",
  "live_events"
)
resources_before <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
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
resources_after <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
vertical_resource_closed <- identical(
  resources_after[resource_fields], resources_before[resource_fields]
)

tasks <- result$tasks
task_identity_exact <- nrow(tasks) == nrow(logical) && identical(
  as.integer(tasks$canonical_test_order_id),
  as.integer(logical$logical_sequence_id)
) && identical(as.integer(tasks$level), as.integer(logical$level)) &&
  identical(as.integer(tasks$task_index),
            as.integer(logical$source_task_index)) &&
  identical(as.integer(tasks$x), as.integer(logical$x)) &&
  identical(as.integer(tasks$y), as.integer(logical$y)) &&
  identical(as.character(tasks$S_key), as.character(logical$S_key)) &&
  identical(as.logical(tasks$native_edge_deleted),
            as.logical(logical$deletes_edge))

ci_comparison <- if (task_identity_exact) {
  numeric_comparison(tasks$p_used, logical$p_value, 1e-10)
} else {
  list(
    same_shape = FALSE, same_na = FALSE, same_infinite = FALSE,
    comparable_count = 0L,
    mismatch_count = max(nrow(tasks), nrow(logical)),
    maximum_absolute_error = Inf, pass = FALSE
  )
}
pmax_comparison <- numeric_comparison(result$pMax, reference_pmax, 1e-10)

summary <- result$summary
authority_zero_fields <- c(
  "r_callback_count", "legacy_mgcv_fit_count", "legacy_mgcv_setup_count",
  "cpu_residual_solve_count", "cpu_dcov_component_count",
  "cpu_dcov_eigen_or_lowrank_count", "cpu_dcov_pair_stat_count",
  "cpu_gamma_pvalue_count", "cpu_spectra_count", "residual_d2h_bytes",
  "component_d2h_bytes", "unknown_fallback_count",
  "approximate_backend_count"
)
authority_zero <- vapply(authority_zero_fields, function(field) {
  !is.null(summary[[field]]) && identical(as.numeric(summary[[field]]), 0)
}, logical(1L))
gpu_route_gate <- identical(
  summary$ci_backend, "legacy-compatible-full-eig-cuda-v1"
) && identical(
  summary$strict_gamma_residual_route, "exact-fixed-stable-svd"
) && as.integer(summary$cuda_legacy_compatible_pair_count) ==
  as.integer(summary$physical_tests_evaluated) &&
  as.integer(summary$legacy_compatible_pair_count) ==
    as.integer(summary$physical_tests_evaluated) &&
  as.integer(summary$guarded_pair_count) == 0L &&
  as.integer(summary$cuda_exact_screen_pair_count) == 0L &&
  as.integer(summary$cuda_guard_refinement_pair_count) == 0L
exact_primary_optimizer_gate <- isTRUE(
  summary$dcc_gamma_exact_primary_optimizer_requested
) && as.numeric(summary$cuda_multi_penalty_target_count) > 0 &&
  as.numeric(summary$cuda_multi_penalty_exact_primary_target_count) ==
    as.numeric(summary$cuda_multi_penalty_target_count) &&
  as.numeric(summary$cuda_multi_penalty_stability_replay_target_count) == 0 &&
  as.numeric(summary$cuda_multi_penalty_stability_replay_selected_count) == 0 &&
  as.numeric(summary$cuda_multi_penalty_terminal_confirmation_count) == 0
cache_enabled <- summary$legacy_component_cache_requested_budget_bytes > 0
cache_fields <- list(
  requested_budget_bytes = summary$legacy_component_cache_requested_budget_bytes,
  capacity_entries = summary$legacy_component_cache_capacity_entries,
  device_bytes = summary$legacy_component_cache_device_bytes,
  entries = summary$legacy_component_cache_entries,
  request_count = summary$legacy_component_cache_request_count,
  hit_count = summary$legacy_component_cache_hit_count,
  miss_count = summary$legacy_component_cache_miss_count,
  insert_count = summary$legacy_component_cache_insert_count,
  bypass_count = summary$legacy_component_cache_bypass_count,
  route_mismatch_count =
    summary$legacy_component_cache_route_mismatch_count,
  status_mismatch_count =
    summary$legacy_component_cache_status_mismatch_count,
  store_d2d_bytes = summary$legacy_component_cache_store_d2d_bytes,
  create_host_ms = summary$legacy_component_cache_create_host_ms,
  residual_bypass_requested =
    summary$legacy_component_cache_residual_bypass_requested,
  residual_bypass_batch_count =
    summary$legacy_component_cache_residual_bypass_batch_count,
  residual_bypass_target_count =
    summary$legacy_component_cache_residual_bypass_target_count,
  batch_count = summary$legacy_component_cache_batch_count,
  all_hit_batch_count = summary$legacy_component_cache_all_hit_batch_count,
  mixed_batch_count = summary$legacy_component_cache_mixed_batch_count,
  all_miss_batch_count = summary$legacy_component_cache_all_miss_batch_count,
  eig_lane_count_requested =
    summary$legacy_component_eig_lane_count_requested,
  eig_lane_count = summary$legacy_component_eig_lane_count,
  eig_active_lane_peak = summary$legacy_component_eig_active_lane_peak,
  eig_multilane_batch_count =
    summary$legacy_component_eig_multilane_batch_count,
  execution_context_reuse_count =
    summary$legacy_component_execution_context_reuse_count,
  execution_context_device_bytes =
    summary$legacy_component_execution_context_device_bytes
)
cache_accounted <- if (cache_enabled) {
  cache_fields[["requested_budget_bytes"]] >=
      cache_fields[["device_bytes"]] &&
    cache_fields[["capacity_entries"]] >= cache_fields[["entries"]] &&
    cache_fields[["request_count"]] ==
      cache_fields[["hit_count"]] + cache_fields[["miss_count"]] &&
    cache_fields[["miss_count"]] ==
      cache_fields[["insert_count"]] + cache_fields[["bypass_count"]] &&
    cache_fields[["entries"]] == cache_fields[["insert_count"]] &&
    cache_fields[["route_mismatch_count"]] == 0 &&
    cache_fields[["status_mismatch_count"]] == 0 &&
    cache_fields[["batch_count"]] ==
      cache_fields[["all_hit_batch_count"]] +
        cache_fields[["mixed_batch_count"]] +
        cache_fields[["all_miss_batch_count"]] &&
    cache_fields[["residual_bypass_batch_count"]] ==
      (if (isTRUE(cache_fields[["residual_bypass_requested"]])) {
        cache_fields[["all_hit_batch_count"]]
      } else 0) &&
    cache_fields[["residual_bypass_target_count"]] >=
      cache_fields[["residual_bypass_batch_count"]] &&
    ((cache_fields[["eig_lane_count"]] == 0 &&
      cache_fields[["eig_lane_count_requested"]] == 0 &&
      cache_fields[["eig_active_lane_peak"]] == 0 &&
      cache_fields[["eig_multilane_batch_count"]] == 0 &&
      cache_fields[["execution_context_reuse_count"]] == 0 &&
      cache_fields[["execution_context_device_bytes"]] == 0) ||
     (cache_fields[["eig_lane_count"]] > 0 &&
      cache_fields[["eig_lane_count_requested"]] >=
        cache_fields[["eig_lane_count"]] &&
      cache_fields[["eig_active_lane_peak"]] >= 1 &&
      cache_fields[["eig_active_lane_peak"]] <=
        cache_fields[["eig_lane_count"]] &&
      cache_fields[["eig_multilane_batch_count"]] <=
        cache_fields[["mixed_batch_count"]] +
          cache_fields[["all_miss_batch_count"]] &&
      cache_fields[["execution_context_reuse_count"]] ==
        cache_fields[["batch_count"]] &&
      cache_fields[["execution_context_device_bytes"]] > 0)) &&
    cache_fields[["store_d2d_bytes"]] > 0 &&
    cache_fields[["create_host_ms"]] >= 0
} else {
  all(unlist(cache_fields, use.names = FALSE) == 0)
}

audit <- list(
  schema_version = "full-cuda-ci-phase10-pvalue-contract-audit-v1",
  tolerance = 1e-10,
  elapsed_sec = elapsed_sec,
  task_identity_exact = task_identity_exact,
  ci = ci_comparison,
  pMax = pmax_comparison,
  adjacency_exact = identical(
    unname(result$adjacency), unname(reference_adjacency)
  ),
  semantic_sepsets_exact = identical(
    canonical_deletion_sepsets(tasks, "native_edge_deleted"),
    canonical_deletion_sepsets(logical, "deletes_edge")
  ),
  n_edgetests_exact = identical(
    as.integer(result$n.edgetests),
    c(2213L, 52659L, 125293L, 40694L, 13293L, 5422L, 835L, 80L, 9L)
  ),
  authority_zero = authority_zero,
  gpu_route_gate = gpu_route_gate,
  exact_primary_optimizer_gate = exact_primary_optimizer_gate,
  authority_gate_pass = isTRUE(summary$authority_gate_pass),
  legacy_component_cache = c(
    list(enabled = cache_enabled, accounted = cache_accounted),
    as.list(cache_fields)
  ),
  vertical_resource_closed = vertical_resource_closed
)
audit$pass <- isTRUE(audit$task_identity_exact) && isTRUE(audit$ci$pass) &&
  isTRUE(audit$pMax$pass) && isTRUE(audit$adjacency_exact) &&
  isTRUE(audit$semantic_sepsets_exact) &&
  isTRUE(audit$n_edgetests_exact) &&
  all(audit$authority_zero) && isTRUE(audit$gpu_route_gate) &&
  isTRUE(audit$exact_primary_optimizer_gate) &&
  isTRUE(audit$authority_gate_pass) &&
  isTRUE(audit$legacy_component_cache$accounted) &&
  isTRUE(audit$vertical_resource_closed)

native_path <- normalizePath(
  "fastkpc/build/fastkpc_cuda.so", winslash = "/", mustWork = TRUE
)
receipt <- list(
  schema_version = "full-cuda-ci-phase10-pvalue-contract-receipt-v1",
  generated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  input_path = normalizePath(data_path, winslash = "/", mustWork = TRUE),
  input_sha256 = unname(digest::digest(
    file = data_path, algo = "sha256", serialize = FALSE
  )),
  native_path = native_path,
  native_sha256 = unname(digest::digest(
    file = native_path, algo = "sha256", serialize = FALSE
  )),
  audit = audit,
  result = result
)
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
temporary_path <- paste0(output_path, ".tmp")
on.exit(unlink(temporary_path, force = TRUE), add = TRUE)
saveRDS(receipt, temporary_path, compress = "xz", version = 3L)
if (!file.rename(temporary_path, output_path)) fail("failed to publish audit")

cat(sprintf(
  paste0(
    "%s Phase 10 p-value audit; elapsed=%.3fs; ci=%d; ",
    "max_ci_diff=%.17g; max_pmax_diff=%.17g; output=%s\n"
  ),
  if (audit$pass) "PASS" else "FAIL", elapsed_sec, nrow(tasks),
  audit$ci$maximum_absolute_error, audit$pMax$maximum_absolute_error,
  output_path
))
if (!audit$pass) fail("Phase 10 p-value contract audit failed")
