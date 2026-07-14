source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP Phase 3A fixed-sp iteration gate\n")
  quit(save = "no", status = 0)
}

build_fastkpc_cuda_native(rebuild = FALSE)

result <- fastkpc_run_full_cuda_fixed_sp_phase3a_iteration(
  phase2_dir = file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "oracle_351x48_v1"
  ),
  census_dir = file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "workload_census_351x48_v1"
  ),
  prepared_dir = file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1"
  ),
  data_path = file.path(
    "fastkpc", "artifacts", "kpc_tprs_real_zhu",
    "cancer_RD-causalDiscoveryInput.rds"
  ),
  device_id = 0L
)

assert_true(
  identical(names(result), c(
    "catalog_records", "runtime_records", "prepared_records",
    "target_records", "timing", "summary"
  )),
  "iteration runner returns records, timing evidence, and a summary"
)

catalog_records <- result$catalog_records
runtime_records <- result$runtime_records
prepared_records <- result$prepared_records
target_records <- result$target_records
timing <- result$timing
summary <- result$summary

assert_true(
  is.data.frame(catalog_records) && nrow(catalog_records) == 1L &&
    identical(catalog_records$scope, "iteration") &&
    isTRUE(catalog_records$authenticated[[1L]]),
  "catalog is opened and authenticated exactly once for iteration scope"
)
assert_true(
  is.data.frame(prepared_records) && nrow(prepared_records) == 44L &&
    identical(
      prepared_records$prepared_s_key_sha256,
      sort(prepared_records$prepared_s_key_sha256, method = "radix")
    ) && !anyDuplicated(prepared_records$prepared_s_key_sha256),
  "all prepared handles follow canonical PreparedSKey radix order"
)
assert_true(
  is.data.frame(target_records) && nrow(target_records) == 270L &&
    all(target_records$target_count == 1L) &&
    all(target_records$native_batch_call),
  "every authenticated target is submitted as one native target"
)

safe <- target_records$planned_route == "CHOLESKY_BATCHED"
stable <- !safe
ok_status <- startsWith(target_records$solver_status, "OK_")
assert_true(
  sum(safe) == 172L && sum(stable) == 98L &&
    identical(target_records$planned_route,
              target_records$authenticated_planned_route) &&
    all(target_records$planned_route[stable] %in%
        c("AUGMENTED_QR", "AUGMENTED_SVD")) &&
    all(target_records$solver_status[safe] == "OK_CHOLESKY_SINGLE") &&
    all(target_records$solver_status[stable] ==
        "ERR_STABLE_PATH_NOT_IMPLEMENTED") &&
    !any(ok_status[stable]),
  "iteration status gate is exactly 172 Cholesky OK and 98 stable errors"
)
assert_true(
  all(target_records$shadow_materialize_call_count[safe] == 1L) &&
    all(target_records$shadow_materialize_call_count[stable] == 0L) &&
    all(target_records$lease_released_before_reuse) &&
    all(!target_records$output_slot_leased_after_release),
  "safe targets are explicitly materialized and every token is released"
)

error_fields <- c(
  "residual_max_abs_diff", "residual_relative_l2_diff",
  "fitted_max_abs_diff", "fitted_relative_l2_diff"
)
assert_true(
  all(vapply(target_records[safe, error_fields, drop = FALSE], function(x) {
    all(is.finite(x)) && max(x) < 1e-7
  }, logical(1L))) &&
    all(vapply(target_records[stable, error_fields, drop = FALSE], function(x) {
      all(is.na(x) & !is.nan(x))
    }, logical(1L))),
  "persistent safe outputs match the Phase 2 oracle below 1e-7"
)

solve_resource_fields <- c(
  "cuda_device_allocation_count_during_solve",
  "cuda_host_allocation_count_during_solve",
  "stream_create_count_during_solve",
  "event_create_count_during_solve",
  "cublas_handle_create_count_during_solve",
  "cusolver_handle_create_count_during_solve",
  "per_target_allocation_count_after_warmup",
  "per_target_handle_create_count"
)
assert_true(
  all(target_records$resource_snapshot_captured) &&
    all(target_records$resource_instrumentation_version == 1L) &&
    all(vapply(target_records[solve_resource_fields], function(x) {
    all(x == 0L)
  }, logical(1L))) &&
    all(target_records$resource_allocation_count_before_solve ==
        target_records$resource_allocation_count_after_solve) &&
    all(target_records$resource_handle_create_count_before_solve ==
        target_records$resource_handle_create_count_after_solve),
  "authoritative resource snapshots prove zero solve-time resource creation"
)
assert_true(
  all(prepared_records$setup_h2d_upload_count == 1L) &&
    sum(prepared_records$setup_h2d_upload_count) == 44L &&
    all(prepared_records$cuda_device_allocation_count_during_setup > 0L) &&
    all(prepared_records$cuda_host_allocation_count_during_setup > 0L) &&
    all(prepared_records$event_create_count_during_setup > 0L),
  "resource counters observe nonzero prepared-setup allocations"
)
created <- runtime_records[runtime_records$stage == "runtime-created", ,
                           drop = FALSE]
reserved <- runtime_records[runtime_records$stage == "workspace-reserved", ,
                            drop = FALSE]
assert_true(
  nrow(created) == 1L && nrow(reserved) == 1L &&
    created$stream_create_count == 1L &&
    created$cublas_handle_create_count == 1L &&
    created$cusolver_handle_create_count == 1L &&
    created$event_create_count > 0L &&
    reserved$cuda_device_allocation_count >
      created$cuda_device_allocation_count &&
    reserved$cuda_host_allocation_count >
      created$cuda_host_allocation_count,
  "resource counters observe lifecycle and workspace resource creation"
)

assert_true(
  identical(timing$warmup_count, c(persistent = 1L, prototype = 1L)) &&
    length(timing$persistent_raw_ms) == 3L &&
    length(timing$prototype_raw_ms) == 3L &&
    all(is.finite(timing$persistent_raw_ms)) &&
    all(is.finite(timing$prototype_raw_ms)) &&
    all(timing$persistent_raw_ms > 0) &&
    all(timing$prototype_raw_ms > 0) &&
    identical(timing$persistent_median_ms,
              unname(stats::median(timing$persistent_raw_ms))) &&
    identical(timing$prototype_median_ms,
              unname(stats::median(timing$prototype_raw_ms))) &&
    timing$persistent_median_ms < timing$prototype_median_ms &&
    timing$speedup > 1,
  "three raw repetitions satisfy the frozen median performance rule"
)
canonical_safe_keys <- target_records$residual_key_sha256[safe]
assert_true(
  timing$benchmark_target_count == 172L &&
    identical(timing$ordered_target_keys, canonical_safe_keys) &&
    identical(
      timing$target_key_corpus_hash,
      fastkpc_full_cuda_census_key_set_hash(canonical_safe_keys)
    ) &&
    identical(timing$persistent_workload_identity, list(
      benchmark_target_count = 172L,
      ordered_target_keys = canonical_safe_keys,
      target_key_corpus_hash = timing$target_key_corpus_hash
    )) &&
    identical(
      timing$prototype_workload_identity[c(
        "benchmark_target_count", "ordered_target_keys", "target_key_corpus_hash"
      )],
      timing$persistent_workload_identity
    ) &&
    length(timing$prototype_workload_identity$ordered_payload_hashes) == 172L &&
    all(grepl(
      "^[0-9a-f]{64}$",
      timing$prototype_workload_identity$ordered_payload_hashes
    )) &&
    identical(
      timing$prototype_workload_identity$payload_corpus_hash,
      fastkpc_full_cuda_census_key_set_hash(
        timing$prototype_workload_identity$ordered_payload_hashes
      )
    ),
  "both timing paths bind the exact ordered corpus and prototype payloads"
)
assert_true(
  is.list(timing$gpu_identity) &&
    identical(timing$gpu_identity$device_id, 0L) &&
    is.character(timing$gpu_identity$name) &&
    length(timing$gpu_identity$name) == 1L &&
    nzchar(timing$gpu_identity$name) &&
    timing$gpu_identity$compute_capability_major >= 1L &&
    timing$gpu_identity$sm_count >= 1L,
  "timing evidence records the CUDA GPU identity"
)
assert_true(
  all(timing$persistent_resource_records$allocation_count == 0L) &&
    all(timing$persistent_resource_records$handle_create_count == 0L),
  "warm and measured persistent corpus repetitions create no resources"
)

recomputed <- list(
  catalog_open_count = as.integer(nrow(catalog_records)),
  setup_count = as.integer(nrow(prepared_records)),
  target_count = as.integer(nrow(target_records)),
  cholesky_ok_count = as.integer(sum(
    target_records$solver_status == "OK_CHOLESKY_SINGLE"
  )),
  stable_not_implemented_count = as.integer(sum(
    target_records$solver_status == "ERR_STABLE_PATH_NOT_IMPLEMENTED"
  )),
  setup_h2d_upload_count = as.integer(sum(
    prepared_records$setup_h2d_upload_count
  )),
  invalid_output_init_count = as.integer(sum(
    target_records$invalid_output_init_count
  )),
  cpu_fallback_count = as.integer(sum(target_records$cpu_fallback_count)),
  unknown_fallback_count = as.integer(sum(
    target_records$unknown_fallback_count
  )),
  approximate_fallback_count = as.integer(sum(
    target_records$approximate_fallback_count
  ))
)
assert_true(
  identical(summary[names(recomputed)], recomputed),
  "summary counts are recomputed from returned records"
)

assert_true(
  summary$catalog_open_count == 1L &&
    summary$setup_count == 44L && summary$target_count == 270L &&
    summary$cholesky_ok_count == 172L &&
    summary$stable_not_implemented_count == 98L &&
    summary$residual_max_abs_diff_max < 1e-7 &&
    summary$residual_relative_l2_diff_max < 1e-7 &&
    summary$fitted_max_abs_diff_max < 1e-7 &&
    summary$fitted_relative_l2_diff_max < 1e-7 &&
    summary$setup_h2d_upload_count == 44L &&
    summary$runtime_context_create_count == 1L &&
    isTRUE(summary$deterministic_runtime_config_exact) &&
    identical(summary$rhs_authority, "cuda-x0-transpose-y") &&
    isTRUE(summary$full_cuda_data_plane) &&
    summary$post_warmup_workspace_grow_count == 0L &&
    summary$cuda_device_synchronize_count == 0L &&
    summary$per_target_allocation_count_after_warmup == 0L &&
    summary$per_target_handle_create_count == 0L &&
    summary$cpu_fallback_count == 0L &&
    summary$unknown_fallback_count == 0L &&
    summary$approximate_fallback_count == 0L &&
    isTRUE(summary$all_output_slot_leases_released) &&
    isTRUE(summary$invalid_output_init_matches_batch_calls) &&
    isTRUE(summary$no_non_cholesky_target_ok) &&
    isTRUE(summary$persistent_faster_than_repeated_prototype) &&
    summary$persistent_speedup > 1,
  "Phase 3A iteration summary satisfies the complete frozen gate"
)

cat(sprintf(
  paste0(
    "PASS Phase 3A fixed-sp iteration gate; persistent raw ms=%s ",
    "median=%.3f; prototype raw ms=%s median=%.3f; speedup=%.3f; gpu=%s\n"
  ),
  paste(format(timing$persistent_raw_ms, digits = 8), collapse = ","),
  timing$persistent_median_ms,
  paste(format(timing$prototype_raw_ms, digits = 8), collapse = ","),
  timing$prototype_median_ms,
  timing$speedup,
  timing$gpu_identity$name
))
