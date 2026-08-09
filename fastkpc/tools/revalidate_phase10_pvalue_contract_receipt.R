fail <- function(message) stop(message, call. = FALSE)
arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) != 2L) {
  fail("usage: revalidate_phase10_pvalue_contract_receipt.R INPUT OUTPUT")
}
input_path <- arguments[[1L]]
output_path <- arguments[[2L]]
if (!file.exists(input_path) || file.exists(output_path) ||
    dir.exists(output_path)) {
  fail("Phase 10 receipt revalidation paths are invalid")
}
if (!requireNamespace("digest", quietly = TRUE)) fail("digest is required")

receipt <- readRDS(input_path)
if (!identical(
  receipt$schema_version,
  "full-cuda-ci-phase10-pvalue-contract-receipt-v1"
)) {
  fail("Phase 10 source receipt schema is invalid")
}
logical <- readRDS(file.path(
  "fastkpc", "artifacts", "full_cuda_ci",
  "oracle_351x48_default_inf_v2", "logical_ci_trace.rds"
))
tasks <- receipt$result$tasks

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

semantic_sepsets_exact <- identical(
  canonical_deletion_sepsets(tasks, "native_edge_deleted"),
  canonical_deletion_sepsets(logical, "deletes_edge")
)
source_audit <- receipt$audit
summary <- receipt$result$summary
zero_if_null <- function(value) if (is.null(value)) 0 else value
false_if_null <- function(value) if (is.null(value)) FALSE else value
cache_summary_present <-
  !is.null(summary$legacy_component_cache_requested_budget_bytes)
cache_enabled <- cache_summary_present &&
  summary$legacy_component_cache_requested_budget_bytes > 0
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
    zero_if_null(summary$legacy_component_cache_route_mismatch_count),
  status_mismatch_count =
    zero_if_null(summary$legacy_component_cache_status_mismatch_count),
  store_d2d_bytes = summary$legacy_component_cache_store_d2d_bytes,
  create_host_ms = summary$legacy_component_cache_create_host_ms,
  residual_bypass_requested =
    false_if_null(summary$legacy_component_cache_residual_bypass_requested),
  residual_bypass_batch_count =
    zero_if_null(summary$legacy_component_cache_residual_bypass_batch_count),
  residual_bypass_target_count =
    zero_if_null(summary$legacy_component_cache_residual_bypass_target_count),
  batch_count = summary$legacy_component_cache_batch_count,
  all_hit_batch_count = summary$legacy_component_cache_all_hit_batch_count,
  mixed_batch_count = summary$legacy_component_cache_mixed_batch_count,
  all_miss_batch_count = summary$legacy_component_cache_all_miss_batch_count,
  eig_lane_count_requested =
    zero_if_null(summary$legacy_component_eig_lane_count_requested),
  eig_lane_count = zero_if_null(summary$legacy_component_eig_lane_count),
  eig_active_lane_peak =
    zero_if_null(summary$legacy_component_eig_active_lane_peak),
  eig_multilane_batch_count =
    zero_if_null(summary$legacy_component_eig_multilane_batch_count),
  execution_context_reuse_count = zero_if_null(
    summary$legacy_component_execution_context_reuse_count
  ),
  execution_context_device_bytes = zero_if_null(
    summary$legacy_component_execution_context_device_bytes
  ),
  logical_residual_requests = summary$logical_residual_requests,
  physical_residual_fits = summary$physical_residual_fits
)
cache_accounted <- if (!cache_summary_present) {
  is.null(source_audit$legacy_component_cache) ||
    isTRUE(source_audit$legacy_component_cache$accounted)
} else if (cache_enabled) {
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
    cache_fields[["physical_residual_fits"]] ==
      cache_fields[["logical_residual_requests"]] -
        cache_fields[["residual_bypass_target_count"]] &&
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
  cache_specific_names <- setdiff(
    names(cache_fields),
    c("logical_residual_requests", "physical_residual_fits")
  )
  all(unlist(cache_fields[cache_specific_names], use.names = FALSE) == 0) &&
    cache_fields[["physical_residual_fits"]] ==
      cache_fields[["logical_residual_requests"]]
}
vertical_resource_closed <- is.null(source_audit$vertical_resource_closed) ||
  isTRUE(source_audit$vertical_resource_closed)
pass <- isTRUE(source_audit$task_identity_exact) &&
  isTRUE(source_audit$ci$pass) && isTRUE(source_audit$pMax$pass) &&
  isTRUE(source_audit$adjacency_exact) && semantic_sepsets_exact &&
  isTRUE(source_audit$n_edgetests_exact) &&
  all(source_audit$authority_zero) && isTRUE(source_audit$gpu_route_gate) &&
  isTRUE(source_audit$authority_gate_pass) && cache_accounted &&
  vertical_resource_closed
attestation <- list(
  schema_version =
    "full-cuda-ci-phase10-pvalue-contract-revalidation-v1",
  generated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  source_receipt_path = normalizePath(
    input_path, winslash = "/", mustWork = TRUE
  ),
  source_receipt_sha256 = unname(digest::digest(
    file = input_path, algo = "sha256", serialize = FALSE
  )),
  source_native_sha256 = receipt$native_sha256,
  semantic_sepsets_exact = semantic_sepsets_exact,
  numerical = list(ci = source_audit$ci, pMax = source_audit$pMax),
  task_identity_exact = source_audit$task_identity_exact,
  adjacency_exact = source_audit$adjacency_exact,
  n_edgetests_exact = source_audit$n_edgetests_exact,
  authority_zero = source_audit$authority_zero,
  gpu_route_gate = source_audit$gpu_route_gate,
  authority_gate_pass = source_audit$authority_gate_pass,
  legacy_component_cache_source_audit = source_audit$legacy_component_cache,
  legacy_component_cache = c(cache_fields, list(
    enabled = cache_enabled,
    accounted = cache_accounted
  )),
  vertical_resource_closed = source_audit$vertical_resource_closed,
  pass = pass
)
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
temporary_path <- paste0(output_path, ".tmp")
on.exit(unlink(temporary_path, force = TRUE), add = TRUE)
saveRDS(attestation, temporary_path, compress = "xz", version = 3L)
if (!file.rename(temporary_path, output_path)) {
  fail("failed to publish Phase 10 receipt revalidation")
}
cat(sprintf(
  "%s Phase 10 receipt revalidation; semantic_sepsets=%s; output=%s\n",
  if (pass) "PASS" else "FAIL", semantic_sepsets_exact, output_path
))
if (!pass) fail("Phase 10 receipt revalidation failed")
