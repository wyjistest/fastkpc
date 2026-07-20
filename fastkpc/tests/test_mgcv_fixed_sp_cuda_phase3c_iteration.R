source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_identical <- function(actual, expected, message) {
  if (!identical(actual, expected)) {
    fail(paste0(
      message, "; actual=", paste(actual, collapse = ","),
      "; expected=", paste(expected, collapse = ",")
    ))
  }
}
assert_sha_vector <- function(value, size, message) {
  assert_true(
    is.character(value) && length(value) == size && !anyNA(value) &&
      all(grepl("^[0-9a-f]{64}$", value)),
    message
  )
}
assert_error_field <- function(value, size, message) {
  assert_true(
    is.double(value) && length(value) == size &&
      all(is.finite(value)) && all(value >= 0) && all(value < 1e-7),
    message
  )
}
assert_error_matching <- function(expression, pattern, message) {
  error <- tryCatch({
    force(expression)
    NULL
  }, error = identity)
  assert_true(
    inherits(error, "error") && grepl(
      pattern, conditionMessage(error), fixed = TRUE
    ),
    message
  )
}

repeat_fixture <- list(
  target_records = data.frame(
    residual_key_sha256 = c(strrep("a", 64L), strrep("b", 64L)),
    planned_route = c("CHOLESKY_BATCHED", "AUGMENTED_SVD"),
    executed_route = c("CHOLESKY_BATCHED", "AUGMENTED_SVD"),
    reroute_reason = c("", ""),
    solver_status = c("OK_CHOLESKY_BATCHED", "OK_AUGMENTED_SVD"),
    fitted_numeric_hash = c(strrep("c", 64L), strrep("d", 64L)),
    residual_numeric_hash = c(strrep("e", 64L), strrep("f", 64L)),
    stringsAsFactors = FALSE
  ),
  summary = list()
)
repeat_fixture$summary <- list(
  route_status_hash = fastkpc_full_cuda_census_metadata_hash(list(
    repeat_fixture$target_records$residual_key_sha256,
    repeat_fixture$target_records$planned_route,
    repeat_fixture$target_records$executed_route,
    repeat_fixture$target_records$reroute_reason,
    repeat_fixture$target_records$solver_status
  )),
  numeric_hash = fastkpc_full_cuda_census_metadata_hash(list(
    repeat_fixture$target_records$residual_key_sha256,
    repeat_fixture$target_records$fitted_numeric_hash,
    repeat_fixture$target_records$residual_numeric_hash
  ))
)
assert_identical(
  fastkpc_full_cuda_fixed_sp_validate_repeat_exact(
    repeat_fixture, repeat_fixture
  ),
  TRUE,
  "repeat validator accepts exact route/status and numeric evidence"
)
repeat_route_drift <- repeat_fixture
repeat_route_drift$target_records$executed_route[[2L]] <- "AUGMENTED_QR"
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_validate_repeat_exact(
    repeat_fixture, repeat_route_drift
  ),
  "route/status evidence changed",
  "repeat validator rejects target-level route drift"
)
repeat_numeric_drift <- repeat_fixture
repeat_numeric_drift$target_records$residual_numeric_hash[[1L]] <-
  strrep("0", 64L)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_validate_repeat_exact(
    repeat_fixture, repeat_numeric_drift
  ),
  "numeric evidence changed",
  "repeat validator rejects target-level numeric drift"
)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP Phase 3C fixed-sp three-route iteration gate\n")
  quit(save = "no", status = 0L)
}

runner_name <- "fastkpc_run_full_cuda_fixed_sp_phase3c_iteration"
runner <- get0(runner_name, mode = "function", inherits = TRUE)
if (is.null(runner)) {
  fail(paste(runner_name, "is missing"))
}

runner_calls <- all.names(body(runner), functions = TRUE, unique = FALSE)
assert_identical(
  as.integer(sum(runner_calls == "fixed_sp_cuda_runtime_create")), 1L,
  "Phase 3C iteration runner must create exactly one persistent runtime"
)
assert_identical(
  as.integer(sum(
    runner_calls == "fastkpc_mgcv_magic_fixed_sp_from_prepared"
  )), 1L,
  "Phase 3C iteration runner must have exactly one C_magic call site"
)
runner_source <- paste(
  deparse(body(runner), width.cutoff = 500L), collapse = "\n"
)
stale_numeric_authority <- any(vapply(c(
  "cpu-augmented-svd", "cpu_augmented_fitted",
  "cpu_augmented_residuals"
), grepl, logical(1L), x = runner_source, fixed = TRUE))
assert_true(
  !stale_numeric_authority,
  paste0(
    "Phase 3C runner must use mgcv-fixed-sp as the sole numeric authority ",
    "and must not replace fitted/residual references by route"
  )
)
assert_true(
  !grepl(
    "fixed_sp_cuda_prepared_materialize_roots_for_test",
    runner_source, fixed = TRUE
  ),
  paste0(
    "Phase 3C aggregate SVD shadow must use the authenticated DTO without ",
    "materializing prepared individual roots"
  )
)
assert_true(
  grepl("max_augmented_rows", runner_source, fixed = TRUE) &&
    grepl("407L", runner_source, fixed = TRUE) &&
    !grepl("dto$n + dto$null_dim", runner_source, fixed = TRUE),
  "Phase 3C runner must reserve the exact logical 407 augmented rows"
)

build_fastkpc_cuda_native(rebuild = TRUE)
assert_true(fastkpc_cuda_available(), "CUDA must be available")

phase2_dir <- file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "oracle_351x48_v1"
)
census_dir <- file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "workload_census_351x48_v1"
)
prepared_dir <- file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1"
)
data_path <- file.path(
  "fastkpc", "artifacts", "kpc_tprs_real_zhu",
  "cancer_RD-causalDiscoveryInput.rds"
)

iteration_setup_rows <- readRDS(file.path(
  prepared_dir, "iteration_setup_groups.rds"
))
iteration_target_rows <- readRDS(file.path(
  prepared_dir, "iteration_target_keys.rds"
))
setup_index <- utils::read.csv(
  file.path(prepared_dir, "prepared_s_setup_index.csv"),
  stringsAsFactors = FALSE, check.names = FALSE
)
setup_key_by_group <- setNames(
  setup_index$prepared_s_key_sha256, setup_index$same_S_group_id
)
expected_setup_keys <- sort(
  unname(setup_key_by_group[iteration_setup_rows$same_S_group_id]),
  method = "radix"
)
target_setup_keys <- unname(
  setup_key_by_group[iteration_target_rows$same_S_group_id]
)
target_order <- order(
  target_setup_keys, iteration_target_rows$residual_key_sha256,
  method = "radix"
)
expected_target_keys <- as.character(
  iteration_target_rows$residual_key_sha256[target_order]
)
assert_true(
  length(expected_setup_keys) == 44L && !anyNA(expected_setup_keys) &&
    length(expected_target_keys) == 270L && !anyNA(expected_target_keys),
  "frozen iteration rows map completely to canonical prepared keys"
)

result <- runner(
  phase2_dir = phase2_dir,
  census_dir = census_dir,
  prepared_dir = prepared_dir,
  data_path = data_path,
  device_id = 0L
)
repeat_result <- runner(
  phase2_dir = phase2_dir,
  census_dir = census_dir,
  prepared_dir = prepared_dir,
  data_path = data_path,
  device_id = 0L
)
assert_identical(
  fastkpc_full_cuda_fixed_sp_validate_repeat_exact(result, repeat_result),
  TRUE,
  "independent Phase 3C iteration executions are exact"
)

assert_identical(
  names(result),
  c(
    "catalog_records", "runtime_records", "setup_records",
    "batch_records", "target_records", "summary"
  ),
  "iteration runner returns the complete evidence surface"
)
catalog_records <- result$catalog_records
runtime_records <- result$runtime_records
setup_records <- result$setup_records
batch_records <- result$batch_records
target_records <- result$target_records
summary <- result$summary

assert_true(
  is.data.frame(catalog_records) && nrow(catalog_records) == 1L &&
    is.data.frame(runtime_records) && nrow(runtime_records) == 3L &&
    is.data.frame(setup_records) && nrow(setup_records) == 44L &&
    is.data.frame(batch_records) && nrow(batch_records) == 44L &&
    is.data.frame(target_records) && nrow(target_records) == 270L &&
    is.list(summary),
  "iteration evidence row counts are exact"
)
assert_identical(
  setup_records$prepared_s_key_sha256, expected_setup_keys,
  "setup rows preserve canonical PreparedSKey order"
)
assert_identical(
  batch_records$prepared_s_key_sha256, expected_setup_keys,
  "batch rows preserve canonical PreparedSKey order"
)
assert_identical(
  target_records$residual_key_sha256, expected_target_keys,
  "target rows preserve canonical residual-key order"
)
assert_true(
  identical(expected_setup_keys, sort(expected_setup_keys, method = "radix")) &&
    !anyDuplicated(expected_setup_keys) &&
    !anyDuplicated(expected_target_keys),
  "authenticated iteration keys are canonical and unique"
)
assert_true(
  identical(
    catalog_records$ordered_setup_key_digest[[1L]],
    fastkpc_full_cuda_census_key_set_hash(expected_setup_keys)
  ) && identical(
    catalog_records$ordered_target_key_digest[[1L]],
    fastkpc_full_cuda_census_key_set_hash(expected_target_keys)
  ),
  "authenticated catalog digests bind the independent canonical key order"
)

required_target_fields <- c(
  "null_dim", "planned_route", "authenticated_planned_route", "executed_route",
  "reroute_reason", "solver_status", "target_true_batched",
  "qr_rank", "geqrf_info", "ormqr_info", "effective_rank",
  "sigma_max", "smallest_retained_sigma", "svd_info",
  "aggregate_penalty_root_rank", "aggregate_penalty_root_pivot",
  "aggregate_factor_call_count", "aggregate_b_build_count",
  "aggregate_dstop", "cpu_aggregate_penalty_root_rank",
  "cpu_aggregate_penalty_root_pivot", "cpu_aggregate_effective_rank",
  "cpu_aggregate_effective_rank_threshold", "cpu_aggregate_sigma_max",
  "aggregate_penalty_root_rank_exact",
  "aggregate_penalty_root_pivot_exact", "aggregate_effective_rank_exact",
  "phase1_coefficient_rank",
  "numeric_reference", "outputs_all_finite", "residual_max_abs_diff",
  "residual_relative_l2_diff", "fitted_max_abs_diff",
  "fitted_relative_l2_diff", "fitted_numeric_hash",
  "residual_numeric_hash", "oracle_call_count", "oracle_fitted_hash",
  "oracle_residual_hash", "authenticated_fitted_hash",
  "authenticated_residual_hash", "oracle_fitted_hash_exact",
  "oracle_residual_hash_exact", "approximate_backend"
)
required_batch_fields <- c(
  "planned_cholesky_target_count", "planned_qr_target_count",
  "planned_svd_target_count", "executed_cholesky_target_count",
  "executed_qr_target_count", "executed_svd_target_count",
  "cholesky_to_svd_count", "qr_to_svd_count",
  "stable_reroute_count", "true_batched_target_count",
  "cholesky_single_target_count", "true_batched_kernel",
  "qr_checkpoint_record_count", "qr_checkpoint_wait_count",
  "svd_checkpoint_record_count", "svd_checkpoint_wait_count",
  "workspace_grow_count_after_warmup",
  "per_target_allocation_count_after_warmup",
  "per_target_handle_create_count", "cuda_device_synchronize_count",
  "implicit_residual_d2h_count", "invalid_output_init_count",
  "cpu_fallback_count", "unknown_fallback_count",
  "coefficient_batch_finalize_call_count",
  "fitted_batch_finalize_call_count",
  "residual_rss_batch_finalize_call_count",
  "per_target_output_finalize_call_count",
  "batch_output_finalized_target_count",
  "aggregate_penalty_factor_count", "aggregate_svd_b_build_count",
  "aggregate_penalty_root_d2h_count",
  "aggregate_penalty_root_d2h_bytes",
  "pre_shadow_materialize_call_count",
  "pre_shadow_materialize_target_count", "pre_shadow_d2h_bytes",
  "post_shadow_materialize_call_count",
  "post_shadow_materialize_target_count", "post_shadow_d2h_bytes",
  "solve_elapsed_ms", "shadow_materialize_elapsed_ms",
  "output_slot_release_count", "output_slot_leased_after_release"
)
required_setup_fields <- c(
  "penalty_root_matrix_count", "penalty_root_row_count",
  "H_root_matrix_count", "penalty_root_rank_mismatch_count",
  "setup_h2d_upload_count", "prepared_handle_create_count",
  "prepared_handle_free_count", "rank_reference_materialize_call_count",
  "rank_reference_materialize_elapsed_ms",
  "setup_shadow_d2h_count", "setup_shadow_d2h_bytes",
  "output_slot_state_after_release", "output_slot_leased_after_release"
)
required_runtime_fields <- c(
  "stage", "generation", "runtime_context_create_count",
  "workspace_grow_count", "cuda_device_synchronize_count",
  "stable_workspace_grow_count", "augmented_workspace_bytes",
  "aggregate_factor_workspace_bytes",
  "qr_checkpoint_record_count", "qr_checkpoint_wait_count",
  "svd_checkpoint_record_count", "svd_checkpoint_wait_count"
)
assert_true(
  length(setdiff(required_target_fields, names(target_records))) == 0L &&
    length(setdiff(required_batch_fields, names(batch_records))) == 0L &&
    length(setdiff(required_setup_fields, names(setup_records))) == 0L &&
    length(setdiff(required_runtime_fields, names(runtime_records))) == 0L,
  "route, rank, resource, timing, and shadow diagnostics are complete"
)

target_count <- nrow(target_records)
for (field in c(
  "residual_max_abs_diff", "residual_relative_l2_diff",
  "fitted_max_abs_diff", "fitted_relative_l2_diff"
)) {
  assert_error_field(
    target_records[[field]], target_count,
    paste("every target satisfies the frozen error ceiling for", field)
  )
}
for (field in c(
  "fitted_numeric_hash", "residual_numeric_hash", "oracle_fitted_hash",
  "oracle_residual_hash", "authenticated_fitted_hash",
  "authenticated_residual_hash"
)) {
  assert_sha_vector(
    target_records[[field]], target_count,
    paste("every target records a valid", field)
  )
}
assert_true(
  all(target_records$outputs_all_finite) &&
    identical(
      target_records$numeric_reference,
      rep("mgcv-fixed-sp", target_count)
    ) &&
    is.integer(target_records$oracle_call_count) &&
    identical(target_records$oracle_call_count, rep(1L, target_count)) &&
    identical(as.integer(sum(target_records$oracle_call_count)), 270L) &&
    all(target_records$oracle_fitted_hash_exact) &&
    all(target_records$oracle_residual_hash_exact) &&
    identical(
      target_records$oracle_fitted_hash,
      target_records$authenticated_fitted_hash
    ) &&
    identical(
      target_records$oracle_residual_hash,
      target_records$authenticated_residual_hash
    ) &&
    !any(target_records$approximate_backend),
  paste0(
    "all outputs use one authoritative C_magic call and exact ",
    "authenticated oracle hashes"
  )
)
assert_identical(
  target_records$planned_route,
  target_records$authenticated_planned_route,
  "planned routes exactly repeat authenticated routes"
)

target_indices <- lapply(expected_setup_keys, function(setup_key) {
  which(target_records$prepared_s_key_sha256 == setup_key)
})
names(target_indices) <- expected_setup_keys
assert_true(
  all(vapply(seq_along(target_indices), function(index) {
    indices <- target_indices[[index]]
    identical(
      target_records$target_ordinal[indices], seq_along(indices)
    ) && identical(
      batch_records$target_count[[index]], as.integer(length(indices))
    )
  }, logical(1L))),
  "every setup owns one complete canonical public batch"
)

route_counts <- function(rows) {
  planned_cholesky <- sum(rows$planned_route == "CHOLESKY_BATCHED")
  planned_qr <- sum(rows$planned_route == "AUGMENTED_QR")
  planned_svd <- sum(rows$planned_route == "AUGMENTED_SVD")
  executed_cholesky <- sum(rows$executed_route == "CHOLESKY_BATCHED")
  executed_qr <- sum(rows$executed_route == "AUGMENTED_QR")
  executed_svd <- sum(rows$executed_route == "AUGMENTED_SVD")
  cholesky_to_svd <- sum(
    rows$planned_route == "CHOLESKY_BATCHED" &
      rows$executed_route == "AUGMENTED_SVD"
  )
  qr_to_svd <- sum(
    rows$planned_route == "AUGMENTED_QR" &
      rows$executed_route == "AUGMENTED_SVD"
  )
  stable_reroute <- sum(rows$planned_route != rows$executed_route)
  list(
    planned_cholesky = as.integer(planned_cholesky),
    planned_qr = as.integer(planned_qr),
    planned_svd = as.integer(planned_svd),
    executed_cholesky = as.integer(executed_cholesky),
    executed_qr = as.integer(executed_qr),
    executed_svd = as.integer(executed_svd),
    cholesky_to_svd = as.integer(cholesky_to_svd),
    qr_to_svd = as.integer(qr_to_svd),
    stable_reroute = as.integer(stable_reroute),
    conserved =
      planned_cholesky == executed_cholesky + cholesky_to_svd &&
      planned_qr == executed_qr + qr_to_svd &&
      executed_svd == planned_svd + cholesky_to_svd + qr_to_svd &&
      stable_reroute == cholesky_to_svd + qr_to_svd
  )
}

per_batch_route <- lapply(target_indices, function(indices) {
  route_counts(target_records[indices, , drop = FALSE])
})
assert_true(
  all(vapply(per_batch_route, `[[`, logical(1L), "conserved")),
  "route conservation equations hold independently for every public batch"
)
route_totals <- route_counts(target_records)
assert_true(
  isTRUE(route_totals$conserved),
  "route conservation equations hold independently for the iteration"
)

batch_route_fields <- c(
  planned_cholesky = "planned_cholesky_target_count",
  planned_qr = "planned_qr_target_count",
  planned_svd = "planned_svd_target_count",
  executed_cholesky = "executed_cholesky_target_count",
  executed_qr = "executed_qr_target_count",
  executed_svd = "executed_svd_target_count",
  cholesky_to_svd = "cholesky_to_svd_count",
  qr_to_svd = "qr_to_svd_count",
  stable_reroute = "stable_reroute_count"
)
assert_true(
  all(vapply(seq_along(per_batch_route), function(index) {
    all(vapply(names(batch_route_fields), function(metric) {
      identical(
        per_batch_route[[index]][[metric]],
        batch_records[[batch_route_fields[[metric]]]][[index]]
      )
    }, logical(1L)))
  }, logical(1L))),
  "batch route counters are recomputed from target evidence"
)
expected_batch_finalize_calls <- as.integer(
  batch_records$executed_cholesky_target_count > 0L
)
assert_true(
  all(batch_records$coefficient_batch_finalize_call_count == 0L) &&
    identical(
      batch_records$fitted_batch_finalize_call_count,
      expected_batch_finalize_calls
    ) &&
    identical(
      batch_records$residual_rss_batch_finalize_call_count,
      expected_batch_finalize_calls
    ) &&
    identical(
      batch_records$per_target_output_finalize_call_count,
      batch_records$executed_qr_target_count +
        batch_records$executed_svd_target_count
    ) &&
    identical(
      batch_records$batch_output_finalized_target_count,
      batch_records$target_count
    ),
  "batch and per-target output finalizers follow executed-route ownership"
)

expected_status <- ifelse(
  target_records$executed_route == "CHOLESKY_BATCHED",
  ifelse(target_records$target_true_batched,
         "OK_CHOLESKY_BATCHED", "OK_CHOLESKY_SINGLE"),
  ifelse(target_records$executed_route == "AUGMENTED_QR",
         "OK_AUGMENTED_QR", "OK_AUGMENTED_SVD")
)
assert_true(
  !anyNA(target_records$executed_route) &&
    all(startsWith(target_records$solver_status, "OK_")) &&
    identical(target_records$solver_status, unname(expected_status)) &&
    all(target_records$reroute_reason == "") &&
    all(target_records$planned_route == target_records$executed_route),
  "all 270 targets complete through their declared route without rerouting"
)

svd <- target_records$executed_route == "AUGMENTED_SVD"
non_svd <- !svd
expected_factor_calls <- as.integer(svd)
expected_b_builds <- 2L * expected_factor_calls
expected_pivot_lengths <- target_records$null_dim * expected_factor_calls
native_pivot_shape_exact <- vapply(seq_len(target_count), function(index) {
  pivot <- target_records$aggregate_penalty_root_pivot[[index]]
  if (!svd[[index]]) {
    return(is.integer(pivot) && identical(pivot, integer()))
  }
  is.integer(pivot) &&
    identical(length(pivot), target_records$null_dim[[index]]) &&
    identical(sort(pivot), seq_len(target_records$null_dim[[index]]))
}, logical(1L))
cpu_pivot_shape_exact <- vapply(seq_len(target_count), function(index) {
  pivot <- target_records$cpu_aggregate_penalty_root_pivot[[index]]
  if (!svd[[index]]) {
    return(is.integer(pivot) && identical(pivot, integer()))
  }
  is.integer(pivot) &&
    identical(length(pivot), target_records$null_dim[[index]]) &&
    identical(sort(pivot), seq_len(target_records$null_dim[[index]]))
}, logical(1L))
assert_true(
  identical(as.integer(sum(svd)), 67L) &&
    identical(as.integer(sum(non_svd)), 203L) &&
    is.integer(target_records$null_dim) &&
    length(target_records$null_dim) == target_count &&
    !anyNA(target_records$null_dim) &&
    all(target_records$null_dim > 0L) &&
    is.integer(target_records$aggregate_penalty_root_rank) &&
    is.list(target_records$aggregate_penalty_root_pivot) &&
    is.integer(target_records$aggregate_factor_call_count) &&
    is.integer(target_records$aggregate_b_build_count) &&
    is.double(target_records$aggregate_dstop) &&
    is.integer(target_records$cpu_aggregate_penalty_root_rank) &&
    is.list(target_records$cpu_aggregate_penalty_root_pivot) &&
    is.integer(target_records$cpu_aggregate_effective_rank) &&
    is.double(target_records$cpu_aggregate_effective_rank_threshold) &&
    is.double(target_records$cpu_aggregate_sigma_max) &&
    identical(
      target_records$aggregate_factor_call_count,
      expected_factor_calls
    ) &&
    identical(target_records$aggregate_b_build_count, expected_b_builds) &&
    identical(
      lengths(target_records$aggregate_penalty_root_pivot),
      expected_pivot_lengths
    ) &&
    identical(
      lengths(target_records$cpu_aggregate_penalty_root_pivot),
      expected_pivot_lengths
    ) &&
    all(native_pivot_shape_exact) && all(cpu_pivot_shape_exact) &&
    all(!is.na(target_records$aggregate_penalty_root_rank[svd])) &&
    all(target_records$aggregate_penalty_root_rank[svd] >= 0L) &&
    all(target_records$aggregate_penalty_root_rank[svd] <=
          target_records$null_dim[svd]) &&
    all(is.na(target_records$aggregate_penalty_root_rank[non_svd])) &&
    all(is.finite(target_records$aggregate_dstop[svd])) &&
    all(target_records$aggregate_dstop[svd] >= 0) &&
    all(is.na(target_records$aggregate_dstop[non_svd])) &&
    all(!is.na(target_records$cpu_aggregate_penalty_root_rank[svd])) &&
    all(target_records$cpu_aggregate_penalty_root_rank[svd] >= 0L) &&
    all(target_records$cpu_aggregate_penalty_root_rank[svd] <=
          target_records$null_dim[svd]) &&
    all(is.na(target_records$cpu_aggregate_penalty_root_rank[non_svd])) &&
    all(!is.na(target_records$cpu_aggregate_effective_rank[svd])) &&
    all(target_records$cpu_aggregate_effective_rank[svd] >= 0L) &&
    all(target_records$cpu_aggregate_effective_rank[svd] <=
          target_records$null_dim[svd]) &&
    all(is.na(target_records$cpu_aggregate_effective_rank[non_svd])) &&
    all(is.finite(
      target_records$cpu_aggregate_effective_rank_threshold[svd]
    )) &&
    all(target_records$cpu_aggregate_effective_rank_threshold[svd] > 0) &&
    all(is.na(
      target_records$cpu_aggregate_effective_rank_threshold[non_svd]
    )) &&
    all(is.finite(target_records$cpu_aggregate_sigma_max[svd])) &&
    all(target_records$cpu_aggregate_sigma_max[svd] > 0) &&
    all(is.na(target_records$cpu_aggregate_sigma_max[non_svd])) &&
    all(target_records$aggregate_penalty_root_rank_exact[svd]) &&
    all(target_records$aggregate_penalty_root_pivot_exact[svd]) &&
    all(target_records$aggregate_effective_rank_exact[svd]) &&
    all(is.na(target_records$aggregate_penalty_root_rank_exact[non_svd])) &&
    all(is.na(target_records$aggregate_penalty_root_pivot_exact[non_svd])) &&
    all(is.na(target_records$aggregate_effective_rank_exact[non_svd])) &&
    identical(
      target_records$aggregate_penalty_root_rank[svd],
      target_records$cpu_aggregate_penalty_root_rank[svd]
    ) &&
    all(vapply(which(svd), function(index) {
      identical(
        target_records$aggregate_penalty_root_pivot[[index]],
        target_records$cpu_aggregate_penalty_root_pivot[[index]]
      )
    }, logical(1L))) &&
    identical(
      target_records$effective_rank[svd],
      target_records$cpu_aggregate_effective_rank[svd]
    ) &&
    all(target_records$svd_info[svd] == 0L) &&
    all(is.finite(target_records$sigma_max[svd])) &&
    all(target_records$sigma_max[svd] > 0) &&
    all(is.finite(target_records$smallest_retained_sigma[svd])) &&
    all(target_records$smallest_retained_sigma[svd] > 0) &&
    is.integer(target_records$phase1_coefficient_rank) &&
    !anyNA(target_records$phase1_coefficient_rank) &&
    all(target_records$numeric_reference == "mgcv-fixed-sp"),
  paste0(
    "all SVD targets have exact aggregate LAPACK rank/pivot and padded-B ",
    "effective-rank parity while non-SVD lifecycle entries stay empty"
  )
)

batch_aggregate_factor_counts <- unname(vapply(
  target_indices, function(indices) {
    as.integer(sum(target_records$aggregate_factor_call_count[indices]))
  }, integer(1L)
))
batch_aggregate_b_build_counts <- unname(vapply(
  target_indices, function(indices) {
    as.integer(sum(target_records$aggregate_b_build_count[indices]))
  }, integer(1L)
))
assert_true(
  identical(
    batch_records$aggregate_penalty_factor_count,
    batch_aggregate_factor_counts
  ) &&
    identical(
      batch_records$aggregate_svd_b_build_count,
      batch_aggregate_b_build_counts
    ) &&
    all(batch_records$aggregate_penalty_root_d2h_count == 0L) &&
    all(batch_records$aggregate_penalty_root_d2h_bytes == 0) &&
    all(setup_records$rank_reference_materialize_call_count == 0L) &&
    all(setup_records$rank_reference_materialize_elapsed_ms == 0) &&
    all(setup_records$setup_shadow_d2h_count == 0L) &&
    all(setup_records$setup_shadow_d2h_bytes == 0),
  paste0(
    "native aggregate totals equal target-vector sums and aggregate/rank ",
    "shadows perform no D2H materialization"
  )
)

qr_affected <- batch_records$planned_qr_target_count > 0L
svd_affected <- batch_records$executed_svd_target_count > 0L
assert_true(
  all(batch_records$qr_checkpoint_record_count >= 0L) &&
    all(batch_records$qr_checkpoint_wait_count >= 0L) &&
    all(batch_records$svd_checkpoint_record_count >= 0L) &&
    all(batch_records$svd_checkpoint_wait_count >= 0L) &&
    all(batch_records$qr_checkpoint_record_count <= as.integer(qr_affected)) &&
    all(batch_records$qr_checkpoint_wait_count <= as.integer(qr_affected)) &&
    all(batch_records$svd_checkpoint_record_count <= as.integer(svd_affected)) &&
    all(batch_records$svd_checkpoint_wait_count <= as.integer(svd_affected)) &&
    sum(batch_records$qr_checkpoint_wait_count) <= sum(qr_affected) &&
    sum(batch_records$svd_checkpoint_wait_count) <= sum(svd_affected),
  "QR/SVD checkpoints occur at most once per affected public batch"
)

reserved_runtime <- match("workspace-reserved", runtime_records$stage)
final_runtime <- match("final", runtime_records$stage)
assert_true(
  !is.na(reserved_runtime) && !is.na(final_runtime) &&
    length(unique(runtime_records$generation)) == 1L &&
    all(runtime_records$runtime_context_create_count == 1L) &&
    identical(
      runtime_records$augmented_workspace_bytes[[reserved_runtime]],
      as.double(8 * 415L * 64L)
    ) &&
    identical(
      runtime_records$augmented_workspace_bytes[[final_runtime]],
      as.double(8 * 415L * 64L)
    ) &&
    identical(
      runtime_records$aggregate_factor_workspace_bytes[[reserved_runtime]],
      as.double(8 * (64L * 64L + 2L * 64L))
    ) &&
    identical(
      runtime_records$aggregate_factor_workspace_bytes[[final_runtime]],
      as.double(8 * (64L * 64L + 2L * 64L))
    ),
  "all 44 batches share one deterministic runtime generation"
)
runtime_workspace_delta <- as.integer(
  runtime_records$workspace_grow_count[[final_runtime]] -
    runtime_records$workspace_grow_count[[reserved_runtime]]
)
runtime_synchronize_delta <- as.integer(
  runtime_records$cuda_device_synchronize_count[[final_runtime]] -
    runtime_records$cuda_device_synchronize_count[[reserved_runtime]]
)
runtime_stable_workspace_delta <- as.integer(
  runtime_records$stable_workspace_grow_count[[final_runtime]] -
    runtime_records$stable_workspace_grow_count[[reserved_runtime]]
)
target_level_stable_sync_count <- as.integer(sum(
  pmax(
    batch_records$qr_checkpoint_wait_count - as.integer(qr_affected), 0L
  ) + pmax(
    batch_records$svd_checkpoint_wait_count - as.integer(svd_affected), 0L
  )
))

whole_batch_true_batched <- as.integer(sum(vapply(
  target_indices, function(indices) {
    length(indices) >= 2L && all(target_records$target_true_batched[indices])
  }, logical(1L)
)))
recomputed <- list(
  setup_count = as.integer(nrow(setup_records)),
  target_count = as.integer(nrow(target_records)),
  penalty_root_matrix_count = as.integer(sum(
    setup_records$penalty_root_matrix_count
  )),
  penalty_root_row_count = as.integer(sum(
    setup_records$penalty_root_row_count
  )),
  H_root_matrix_count = as.integer(sum(setup_records$H_root_matrix_count)),
  planned_cholesky_count = route_totals$planned_cholesky,
  planned_qr_count = route_totals$planned_qr,
  planned_svd_count = route_totals$planned_svd,
  executed_cholesky_count = route_totals$executed_cholesky,
  executed_qr_count = route_totals$executed_qr,
  executed_svd_count = route_totals$executed_svd,
  cholesky_to_svd_count = route_totals$cholesky_to_svd,
  qr_to_svd_count = route_totals$qr_to_svd,
  true_batched_target_count = as.integer(sum(
    target_records$target_true_batched
  )),
  cholesky_single_target_count = as.integer(sum(
    target_records$solver_status == "OK_CHOLESKY_SINGLE"
  )),
  whole_batch_true_batched_count = whole_batch_true_batched,
  stable_not_implemented_count = as.integer(sum(
    target_records$solver_status == "ERR_STABLE_PATH_NOT_IMPLEMENTED"
  )),
  stable_reroute_count = route_totals$stable_reroute,
  non_ok_status_count = as.integer(sum(
    !startsWith(target_records$solver_status, "OK_")
  )),
  root_rank_mismatch_count = as.integer(sum(
    setup_records$penalty_root_rank_mismatch_count
  )),
  aggregate_penalty_factor_count = as.integer(sum(
    target_records$aggregate_factor_call_count
  )),
  aggregate_svd_b_build_count = as.integer(sum(
    target_records$aggregate_b_build_count
  )),
  aggregate_penalty_root_rank_mismatch_count = as.integer(sum(
    !target_records$aggregate_penalty_root_rank_exact[svd]
  )),
  aggregate_penalty_root_pivot_mismatch_count = as.integer(sum(
    !target_records$aggregate_penalty_root_pivot_exact[svd]
  )),
  aggregate_penalty_root_d2h_count = as.integer(sum(
    batch_records$aggregate_penalty_root_d2h_count
  )),
  aggregate_penalty_root_d2h_bytes = as.double(sum(
    batch_records$aggregate_penalty_root_d2h_bytes
  )),
  workspace_grow_count_after_warmup = runtime_workspace_delta,
  stable_workspace_grow_count_after_warmup =
    runtime_stable_workspace_delta,
  per_target_allocation_count_after_warmup = as.integer(sum(
    batch_records$per_target_allocation_count_after_warmup
  )),
  per_target_handle_create_count = as.integer(sum(
    batch_records$per_target_handle_create_count
  )),
  cuda_device_synchronize_count = runtime_synchronize_delta,
  target_level_stable_sync_count = target_level_stable_sync_count,
  implicit_residual_d2h_count = as.integer(sum(
    batch_records$implicit_residual_d2h_count
  )),
  all_output_slot_leases_released = all(
    setup_records$output_slot_state_after_release == "free" &
      !setup_records$output_slot_leased_after_release &
      batch_records$output_slot_release_count == 1L &
      !batch_records$output_slot_leased_after_release
  ),
  invalid_output_init_count = as.integer(sum(
    batch_records$invalid_output_init_count
  )),
  cpu_fallback_count = as.integer(sum(batch_records$cpu_fallback_count)),
  unknown_fallback_count = as.integer(sum(
    batch_records$unknown_fallback_count
  )),
  approximate_backend_count = as.integer(sum(
    target_records$approximate_backend
  ))
)
expected <- list(
  setup_count = 44L,
  target_count = 270L,
  penalty_root_matrix_count = 159L,
  penalty_root_row_count = 1424L,
  H_root_matrix_count = 0L,
  planned_cholesky_count = 172L,
  planned_qr_count = 31L,
  planned_svd_count = 67L,
  executed_cholesky_count = 172L,
  executed_qr_count = 31L,
  executed_svd_count = 67L,
  cholesky_to_svd_count = 0L,
  qr_to_svd_count = 0L,
  true_batched_target_count = 160L,
  cholesky_single_target_count = 12L,
  whole_batch_true_batched_count = 5L,
  stable_not_implemented_count = 0L,
  stable_reroute_count = 0L,
  non_ok_status_count = 0L,
  root_rank_mismatch_count = 0L,
  aggregate_penalty_factor_count = 67L,
  aggregate_svd_b_build_count = 134L,
  aggregate_penalty_root_rank_mismatch_count = 0L,
  aggregate_penalty_root_pivot_mismatch_count = 0L,
  aggregate_penalty_root_d2h_count = 0L,
  aggregate_penalty_root_d2h_bytes = 0,
  workspace_grow_count_after_warmup = 0L,
  stable_workspace_grow_count_after_warmup = 0L,
  per_target_allocation_count_after_warmup = 0L,
  per_target_handle_create_count = 0L,
  cuda_device_synchronize_count = 0L,
  target_level_stable_sync_count = 0L,
  implicit_residual_d2h_count = 0L,
  all_output_slot_leases_released = TRUE,
  invalid_output_init_count = 44L,
  cpu_fallback_count = 0L,
  unknown_fallback_count = 0L,
  approximate_backend_count = 0L
)
assert_identical(
  recomputed, expected,
  "independently recomputed Phase 3C iteration totals are exact"
)
assert_identical(
  summary[names(expected)], expected,
  "runner summary is derived from the exact evidence rows"
)

assert_true(
  all(batch_records$pre_shadow_materialize_call_count == 0L) &&
    all(batch_records$pre_shadow_materialize_target_count == 0L) &&
    all(batch_records$pre_shadow_d2h_bytes == 0) &&
    all(batch_records$post_shadow_materialize_call_count == 1L) &&
    identical(
      batch_records$post_shadow_materialize_target_count,
      batch_records$target_count
    ) &&
    all(batch_records$post_shadow_d2h_bytes > 0) &&
    all(is.finite(batch_records$solve_elapsed_ms)) &&
    all(batch_records$solve_elapsed_ms >= 0) &&
    all(is.finite(batch_records$shadow_materialize_elapsed_ms)) &&
    all(batch_records$shadow_materialize_elapsed_ms >= 0),
  "explicit shadow transfer metrics and timing remain separate from solves"
)

route_status_hash <- fastkpc_full_cuda_census_metadata_hash(list(
  target_records$residual_key_sha256,
  target_records$planned_route,
  target_records$executed_route,
  target_records$reroute_reason,
  target_records$solver_status
))
numeric_hash <- fastkpc_full_cuda_census_metadata_hash(list(
  target_records$residual_key_sha256,
  target_records$fitted_numeric_hash,
  target_records$residual_numeric_hash
))
assert_identical(
  summary$route_status_hash, route_status_hash,
  "route/status repeat digest is exact"
)
assert_identical(
  summary$numeric_hash, numeric_hash,
  "GPU numeric repeat digest is exact"
)

max_errors <- vapply(c(
  "residual_max_abs_diff", "residual_relative_l2_diff",
  "fitted_max_abs_diff", "fitted_relative_l2_diff"
), function(field) max(target_records[[field]]), double(1L))

cat("PASS Phase 3C fixed-sp three-route iteration gate\n")
cat(
  "counts setups/targets/roots/root_rows/H_roots=",
  recomputed$setup_count, "/", recomputed$target_count, "/",
  recomputed$penalty_root_matrix_count, "/",
  recomputed$penalty_root_row_count, "/",
  recomputed$H_root_matrix_count, "\n", sep = ""
)
cat(
  "planned/executed cholesky/qr/svd=",
  recomputed$planned_cholesky_count, "/",
  recomputed$planned_qr_count, "/", recomputed$planned_svd_count, " ",
  recomputed$executed_cholesky_count, "/",
  recomputed$executed_qr_count, "/", recomputed$executed_svd_count,
  "\n", sep = ""
)
cat(
  "batching true/single/whole=", recomputed$true_batched_target_count,
  "/", recomputed$cholesky_single_target_count, "/",
  recomputed$whole_batch_true_batched_count, "\n", sep = ""
)
cat(
  "aggregate factors/builds/rank_mismatch/pivot_mismatch/root_d2h=",
  recomputed$aggregate_penalty_factor_count, "/",
  recomputed$aggregate_svd_b_build_count, "/",
  recomputed$aggregate_penalty_root_rank_mismatch_count, "/",
  recomputed$aggregate_penalty_root_pivot_mismatch_count, "/",
  recomputed$aggregate_penalty_root_d2h_count, "\n", sep = ""
)
cat(
  "workspace augmented/aggregate_factor/stable_growth=",
  runtime_records$augmented_workspace_bytes[[final_runtime]], "/",
  runtime_records$aggregate_factor_workspace_bytes[[final_runtime]], "/",
  recomputed$stable_workspace_grow_count_after_warmup, "\n", sep = ""
)
cat(
  "oracle calls/numeric_reference=",
  sum(target_records$oracle_call_count), "/mgcv-fixed-sp\n", sep = ""
)
cat(
  "qr/svd checkpoint waits=", sum(batch_records$qr_checkpoint_wait_count),
  "/", sum(batch_records$svd_checkpoint_wait_count), "\n", sep = ""
)
cat(
  "max errors residual_abs/residual_rel/fitted_abs/fitted_rel=",
  paste(format(max_errors, digits = 17L), collapse = "/"), "\n", sep = ""
)
cat("route_status_hash=", route_status_hash, "\n", sep = "")
cat("numeric_hash=", numeric_hash, "\n", sep = "")
