source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/full_cuda_ci_phase3_artifacts.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) {
  if (!isTRUE(value)) fail(message)
}
assert_identical <- function(actual, expected, message) {
  if (!identical(actual, expected)) {
    fail(paste0(
      message, "; actual=", paste(actual, collapse = ","),
      "; expected=", paste(expected, collapse = ",")
    ))
  }
}
assert_error <- function(expression, message) {
  error <- tryCatch({
    force(expression)
    NULL
  }, error = function(error) error)
  assert_true(inherits(error, "error"), message)
  invisible(error)
}

sha <- function(label) fastkpc_full_cuda_census_hash_utf8(label)
key_hash <- function(keys) {
  fastkpc_full_cuda_census_key_set_hash(sort(keys, method = "radix"))
}
refresh_hash <- function(value) {
  value$sha256 <- NULL
  value$sha256 <- fastkpc_full_cuda_census_named_metadata_hash(value)
  value
}
recorded_identity <- as.list(setNames(
  rep("recorded", length(.fastkpc_full_cuda_phase3_identity_fields()) + 1L),
  c(.fastkpc_full_cuda_phase3_identity_fields(), "sha256")
))
current_identity <- recorded_identity
current_identity$native_library_inode <- "current-inode"
assert_true(
  .fastkpc_full_cuda_phase3_identity_json_exact(
    recorded_identity, current_identity,
    fields = c(.fastkpc_full_cuda_phase3_stable_identity_fields(), "sha256")
  ),
  "completed identity comparison permits volatile native-session changes"
)
current_identity$gpu_uuid <- "different-gpu"
assert_true(
  !.fastkpc_full_cuda_phase3_identity_json_exact(
    recorded_identity, current_identity,
    fields = c(.fastkpc_full_cuda_phase3_stable_identity_fields(), "sha256")
  ),
  "completed identity comparison still rejects stable GPU changes"
)
assert_true(
  "shadow_callback" %in% names(formals(
    fastkpc_full_cuda_fixed_sp_execute_oracle_setup
  )),
  "oracle setup execution exposes a synchronous explicit-shadow callback"
)
execute_body <- as.list(body(
  fastkpc_full_cuda_fixed_sp_execute_oracle_setup
))
execute_return_expression <- execute_body[[length(execute_body)]]
execute_return_environment <- list2env(list(
  setup_results = "setup", target_parity = "target",
  resource_metrics = "resource", stage_timing = "timing",
  shadow_callback = NULL, shadow_callback_result = NULL
), parent = baseenv())
default_execute_result <- eval(
  execute_return_expression, envir = execute_return_environment
)
assert_identical(
  names(default_execute_result),
  c("setup_results", "target_parity", "resource_metrics", "stage_timing"),
  "oracle setup default return preserves the four-component contract"
)
execute_return_environment$shadow_callback <- identity
execute_return_environment$shadow_callback_result <- "callback-result"
callback_execute_result <- eval(
  execute_return_expression, envir = execute_return_environment
)
assert_identical(
  names(callback_execute_result),
  c(
    "setup_results", "target_parity", "resource_metrics", "stage_timing",
    "shadow_callback_result"
  ),
  "oracle setup appends callback evidence only when a callback is supplied"
)
frame <- function(name, row_count) {
  schema <- fastkpc_full_cuda_fixed_sp_oracle_row_schemas()[[name]]
  columns <- lapply(unname(schema), function(type) switch(
    type,
    character = rep("", row_count),
    integer = rep.int(0L, row_count),
    double = rep(0, row_count),
    logical = rep.int(FALSE, row_count)
  ))
  names(columns) <- names(schema)
  as.data.frame(columns, stringsAsFactors = FALSE, optional = TRUE)
}

setup_keys <- sort(vapply(
  sprintf("oracle-artifact-setup-%02d", seq_len(44L)),
  sha, character(1L)
), method = "radix")
setup_target_count <- c(rep.int(7L, 6L), rep.int(6L, 38L))
target_rows <- do.call(rbind, lapply(seq_along(setup_keys), function(index) {
  keys <- sort(vapply(
    sprintf("oracle-artifact-target-%02d-%02d", index,
            seq_len(setup_target_count[[index]])),
    sha, character(1L)
  ), method = "radix")
  data.frame(
    prepared_s_key_sha256 = rep(setup_keys[[index]], length(keys)),
    residual_key_sha256 = keys,
    stringsAsFactors = FALSE
  )
}))
rownames(target_rows) <- NULL
target_rows$canonical_setup_rank <- as.integer(match(
  target_rows$prepared_s_key_sha256, setup_keys
))
target_rows$canonical_target_rank <- as.integer(match(
  target_rows$residual_key_sha256,
  sort(target_rows$residual_key_sha256, method = "radix")
))
target_rows$phase2_shard_id <- as.integer(
  (target_rows$canonical_setup_rank - 1L) %%
    fastkpc_full_cuda_fixed_sp_catalog_contract()$shard_count
)
target_rows$target <- as.integer(
  ave(seq_len(nrow(target_rows)), target_rows$prepared_s_key_sha256,
      FUN = seq_along)
)
target_rows$null_dim <- rep.int(4L, nrow(target_rows))
routes <- c(
  rep("CHOLESKY_BATCHED", 172L), rep("AUGMENTED_QR", 31L),
  rep("AUGMENTED_SVD", 67L)
)
# Spread every route across the setup fixture while retaining exact totals.
target_local_rank <- as.integer(ave(
  seq_len(nrow(target_rows)), target_rows$prepared_s_key_sha256,
  FUN = seq_along
))
route_rank <- order(
  target_local_rank, target_rows$canonical_setup_rank, method = "radix"
)
target_rows$planned_route <- character(nrow(target_rows))
target_rows$planned_route[route_rank] <- routes
target_rows$condition <- c(
  CHOLESKY_BATCHED = 1e4, AUGMENTED_QR = 1e10, AUGMENTED_SVD = 1e13
)[target_rows$planned_route]
target_rows$condition <- as.double(target_rows$condition)
target_rows$coefficient_rank <- ifelse(
  target_rows$planned_route == "AUGMENTED_SVD", 3L, 4L
)
target_rows$coefficient_rank <- as.integer(target_rows$coefficient_rank)
target_rows$selected_sp_hash <- vapply(
  target_rows$residual_key_sha256,
  function(key) sha(paste0("selected-sp-", key)), character(1L)
)
for (field in c(
  "coefficient_hash", "fitted_hash", "residual_hash",
  "target_fit_fingerprint"
)) {
  target_rows[[field]] <- vapply(
    target_rows$residual_key_sha256,
    function(key) sha(paste0(field, "-", key)), character(1L)
  )
}
target_rows <- target_rows[, setdiff(
  .fastkpc_full_cuda_phase3_oracle_descriptor_target_fields(), "shard_id"
), drop = FALSE]

route_config <- fastkpc_full_cuda_phase3_route_config()
identity <- refresh_hash(list(
  schema_version = "full-cuda-ci-phase3-test-input-identity-v1",
  canonical_setup_corpus_hash = key_hash(setup_keys),
  canonical_target_corpus_hash = key_hash(target_rows$residual_key_sha256),
  route_config_hash = route_config$sha256,
  source_commit = strrep("1", 40L),
  cuda_toolkit_version = 12040L,
  cuda_driver_version = 55054L,
  gpu_name = "Synthetic GPU",
  gpu_uuid = paste0("GPU-", strrep("a", 32L)),
  compute_capability_major = 8L,
  compute_capability_minor = 0L,
  compute_capability = "8.0",
  sm_count = 108L,
  device_id = 0L,
  cusolver_deterministic_mode_required = "enabled",
  cublas_math_mode_required = "pedantic",
  cublas_atomics_mode_required = "not_allowed",
  cublas_user_workspace_required = TRUE,
  cublas_workspace_bytes_required = 16777216,
  cublas_workspace_min_alignment_required = 256
))

canonical_opener_name <-
  ".fastkpc_full_cuda_phase3_open_canonical_oracle_catalog"
original_canonical_opener <- get0(
  canonical_opener_name, envir = .GlobalEnv, inherits = FALSE
)
canonical_opener_calls <- 0L
assign(
  canonical_opener_name,
  function() {
    canonical_opener_calls <<- canonical_opener_calls + 1L
    stop("canonical full lineage reopened", call. = FALSE)
  },
  envir = .GlobalEnv
)
assert_error(
  fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
    tempfile("missing-full-oracle-"), expected_identity = identity,
    require_full = TRUE
  ),
  "full validation without caller inputs attempts canonical lineage reopening"
)
assert_error(
  fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
    tempfile("synthetic-full-oracle-"), expected_identity = identity,
    require_full = TRUE,
    catalog = list(
      risk_summary = list(high_condition_count = 33249L),
      qualification_summary = list(
        logical_test_count = 3808L, near_alpha_count = 1478L,
        unique_residual_key_count = 6143L
      )
    ),
    device_id = 0L
  ),
  "synthetic aggregate-only full inputs cannot replace canonical lineage"
)
if (is.null(original_canonical_opener)) {
  rm(list = canonical_opener_name, envir = .GlobalEnv)
} else {
  assign(
    canonical_opener_name, original_canonical_opener, envir = .GlobalEnv
  )
}
assert_identical(
  canonical_opener_calls, 2L,
  "every full validation independently reopens canonical catalog lineage"
)

make_oracle_payload <- function(shard_id, shard_setup_keys, shard_targets) {
  setup_count <- length(shard_setup_keys)
  target_count <- nrow(shard_targets)
  setup_rank <- match(shard_targets$prepared_s_key_sha256, shard_setup_keys)
  target_parity <- frame("target_parity", target_count)
  target_parity$prepared_s_key_sha256 <-
    as.character(shard_targets$prepared_s_key_sha256)
  target_parity$shard_id <- rep.int(as.integer(shard_id), target_count)
  target_parity$setup_ordinal <- as.integer(setup_rank)
  target_parity$canonical_setup_rank <-
    as.integer(shard_targets$canonical_setup_rank)
  target_parity$target_ordinal <- as.integer(ave(
    seq_len(target_count), shard_targets$prepared_s_key_sha256,
    FUN = seq_along
  ))
  target_parity$canonical_target_rank <-
    as.integer(shard_targets$canonical_target_rank)
  target_parity$residual_key_sha256 <-
    as.character(shard_targets$residual_key_sha256)
  target_parity$target <- as.integer(shard_targets$target)
  target_parity$null_dim <- as.integer(shard_targets$null_dim)
  target_parity$condition <- as.double(shard_targets$condition)
  target_parity$condition_bucket <- vapply(seq_len(target_count), function(i) {
    fastkpc_full_cuda_census_condition_bucket(
      target_parity$condition[[i]],
      shard_targets$coefficient_rank[[i]], target_parity$null_dim[[i]]
    )
  }, character(1L))
  target_parity$phase1_coefficient_rank <-
    as.integer(shard_targets$coefficient_rank)
  target_parity$planned_route <- as.character(shard_targets$planned_route)
  target_parity$authenticated_planned_route <-
    as.character(shard_targets$planned_route)
  target_parity$executed_route <- as.character(shard_targets$planned_route)
  target_parity$reroute_reason <- rep("", target_count)
  cholesky <- target_parity$planned_route == "CHOLESKY_BATCHED"
  qr <- target_parity$planned_route == "AUGMENTED_QR"
  svd <- target_parity$planned_route == "AUGMENTED_SVD"
  cholesky_count <- vapply(shard_setup_keys, function(key) {
    sum(cholesky[target_parity$prepared_s_key_sha256 == key])
  }, integer(1L))
  true_batched <- cholesky & cholesky_count[setup_rank] >= 2L
  target_parity$solver_status <- ifelse(
    cholesky, ifelse(true_batched, "OK_CHOLESKY_BATCHED",
                    "OK_CHOLESKY_SINGLE"),
    ifelse(qr, "OK_AUGMENTED_QR", "OK_AUGMENTED_SVD")
  )
  target_parity$target_true_batched <- true_batched
  target_parity$true_batched_kernel <- vapply(seq_len(target_count), function(i) {
    selected <- setup_rank == setup_rank[[i]]
    sum(selected) >= 2L && all(true_batched[selected])
  }, logical(1L))
  target_parity$true_batched_target_count <-
    as.integer(cholesky_count[setup_rank])
  target_parity$qr_rank <- ifelse(qr, 4L, -1L)
  target_parity$qr_rank <- as.integer(target_parity$qr_rank)
  target_parity$geqrf_info <- as.integer(ifelse(qr, 0L, -1L))
  target_parity$ormqr_info <- as.integer(ifelse(qr, 0L, -1L))
  target_parity$effective_rank <- as.integer(ifelse(svd, 3L, -1L))
  target_parity$sigma_max <- ifelse(svd, 10, NaN)
  target_parity$smallest_retained_sigma <- ifelse(svd, 1, NaN)
  target_parity$svd_info <- as.integer(ifelse(svd, 0L, -1L))
  target_parity$aggregate_penalty_root_rank <-
    as.integer(ifelse(svd, 3L, NA_integer_))
  target_parity$aggregate_penalty_root_pivot_sha256 <- vapply(
    seq_len(target_count),
    function(i) sha(paste0("pivot-", svd[[i]], "-", i)), character(1L)
  )
  target_parity$aggregate_factor_call_count <- as.integer(svd)
  target_parity$aggregate_b_build_count <- 2L * as.integer(svd)
  target_parity$aggregate_dstop <- ifelse(svd, 1e-12, NA_real_)
  target_parity$numeric_reference <- rep("mgcv-fixed-sp", target_count)
  for (field in c(
    "coefficient_all_finite", "fitted_all_finite", "residual_all_finite",
    "rss_all_finite", "rhs_all_finite", "output_all_finite",
    "coefficient_oracle_phase2_exact", "fitted_oracle_phase2_exact",
    "residual_oracle_phase2_exact", "full_cuda_data_plane"
  )) target_parity[[field]] <- rep(TRUE, target_count)
  for (field in grep(
    "_(max_abs_diff|relative_l2)$", names(target_parity), value = TRUE
  )) target_parity[[field]] <- rep(1e-14, target_count)
  for (field in grep(
    "(sha256|fingerprint)$", names(target_parity), value = TRUE
  )) {
    if (!field %in% c(
      "prepared_s_key_sha256", "residual_key_sha256",
      "aggregate_penalty_root_pivot_sha256"
    )) {
      target_parity[[field]] <- vapply(
        shard_targets$residual_key_sha256,
        function(key) sha(paste0(field, "-", key)), character(1L)
      )
    }
  }
  target_parity$selected_sp_sha256 <- shard_targets$selected_sp_hash
  target_parity$coefficient_phase2_sha256 <-
    shard_targets$coefficient_hash
  target_parity$fitted_phase2_sha256 <- shard_targets$fitted_hash
  target_parity$residual_phase2_sha256 <- shard_targets$residual_hash
  target_parity$target_fit_fingerprint <-
    shard_targets$target_fit_fingerprint
  target_parity$oracle_call_count <- rep.int(1L, target_count)
  target_parity$rhs_authority <- rep("cuda-x0-transpose-y", target_count)
  target_parity$approximate_backend <- rep(FALSE, target_count)
  target_parity$fallback_type <- rep("NONE", target_count)
  target_parity$error_code <- rep("NONE", target_count)
  target_parity$error_message_sha256 <- rep(sha(""), target_count)

  route_count <- function(key, route) sum(
    target_parity$prepared_s_key_sha256 == key &
      target_parity$planned_route == route
  )
  setup_results <- frame("setup_results", setup_count)
  setup_results$prepared_s_key_sha256 <- shard_setup_keys
  setup_results$shard_id <- rep.int(as.integer(shard_id), setup_count)
  setup_results$setup_ordinal <- seq_len(setup_count)
  setup_results$canonical_setup_rank <-
    as.integer(match(shard_setup_keys, setup_keys))
  setup_results$phase2_shard_id <- as.integer(
    (setup_results$canonical_setup_rank - 1L) %%
      fastkpc_full_cuda_fixed_sp_catalog_contract()$shard_count
  )
  setup_results$phase2_shard_load_count <- rep.int(1L, setup_count)
  setup_results$phase2_shard_authentication_count <- rep.int(1L, setup_count)
  setup_results$n <- rep.int(351L, setup_count)
  setup_results$coefficient_dim <- rep.int(5L, setup_count)
  setup_results$null_dim <- rep.int(4L, setup_count)
  setup_results$penalty_count <- rep.int(2L, setup_count)
  setup_results$target_count <- as.integer(vapply(
    shard_setup_keys,
    function(key) sum(target_parity$prepared_s_key_sha256 == key),
    integer(1L)
  ))
  setup_results$target_key_set_sha256 <- vapply(
    shard_setup_keys, function(key) key_hash(
      target_parity$residual_key_sha256[
        target_parity$prepared_s_key_sha256 == key
      ]
    ), character(1L)
  )
  setup_results$prepared_handle_create_count <- rep.int(1L, setup_count)
  setup_results$prepared_handle_destroy_count <- rep.int(1L, setup_count)
  setup_results$setup_h2d_upload_count <- rep.int(1L, setup_count)
  setup_results$setup_h2d_bytes <- rep(1024, setup_count)
  setup_results$penalty_root_build_count <- rep.int(2L, setup_count)
  setup_results$penalty_root_matrix_count <- rep.int(2L, setup_count)
  setup_results$penalty_root_row_count <- rep.int(8L, setup_count)
  for (prefix in c("planned", "executed")) {
    setup_results[[paste0(prefix, "_cholesky_target_count")]] <-
      as.integer(vapply(shard_setup_keys, route_count, integer(1L),
                        route = "CHOLESKY_BATCHED"))
    setup_results[[paste0(prefix, "_qr_target_count")]] <-
      as.integer(vapply(shard_setup_keys, route_count, integer(1L),
                        route = "AUGMENTED_QR"))
    setup_results[[paste0(prefix, "_svd_target_count")]] <-
      as.integer(vapply(shard_setup_keys, route_count, integer(1L),
                        route = "AUGMENTED_SVD"))
  }
  setup_results$true_batched_target_count <- cholesky_count
  setup_results$setup_load_elapsed_ms <- rep(1, setup_count)
  setup_results$total_elapsed_ms <- rep(2, setup_count)

  resource_metrics <- frame("resource_metrics", setup_count)
  for (field in intersect(names(resource_metrics), names(setup_results))) {
    resource_metrics[[field]] <- setup_results[[field]]
  }
  for (field in c(
    "prepared_handle_create_count", "prepared_handle_destroy_count",
    "residual_token_acquire_count", "residual_token_release_count",
    "output_slot_acquire_count", "output_slot_release_count",
    "setup_h2d_upload_count", "target_batch_h2d_call_count",
    "rhs_device_build_count", "coefficient_batch_finalize_call_count",
    "fitted_batch_finalize_call_count",
    "residual_rss_batch_finalize_call_count", "shadow_materialize_call_count"
  )) resource_metrics[[field]] <- rep.int(1L, setup_count)
  resource_metrics$setup_h2d_bytes <- rep(1024, setup_count)
  resource_metrics$target_h2d_copy_count <- rep.int(2L, setup_count)
  resource_metrics$target_h2d_bytes <- rep(2048, setup_count)
  resource_metrics$rhs_authority <-
    rep("cuda-x0-transpose-y", setup_count)
  resource_metrics$full_cuda_data_plane <- rep(TRUE, setup_count)
  resource_metrics$batch_output_finalized_target_count <-
    setup_results$target_count
  resource_metrics$true_batched_subgroup_count <-
    as.integer(cholesky_count >= 2L)
  resource_metrics$true_batched_attempted_target_count <-
    as.integer(ifelse(cholesky_count >= 2L, cholesky_count, 0L))
  resource_metrics$true_batched_target_count <- cholesky_count
  resource_metrics$aggregate_penalty_factor_count <-
    setup_results$executed_svd_target_count
  resource_metrics$aggregate_svd_b_build_count <-
    2L * resource_metrics$aggregate_penalty_factor_count
  resource_metrics$cholesky_factor_checkpoint_record_count <-
    as.integer(cholesky_count > 0L)
  resource_metrics$cholesky_factor_checkpoint_wait_count <-
    resource_metrics$cholesky_factor_checkpoint_record_count
  resource_metrics$cholesky_solve_checkpoint_record_count <-
    as.integer(cholesky_count > 0L)
  resource_metrics$cholesky_solve_checkpoint_wait_count <-
    resource_metrics$cholesky_solve_checkpoint_record_count
  resource_metrics$qr_checkpoint_record_count <-
    as.integer(setup_results$planned_qr_target_count > 0L)
  resource_metrics$qr_checkpoint_wait_count <-
    resource_metrics$qr_checkpoint_record_count
  resource_metrics$svd_checkpoint_record_count <-
    as.integer(setup_results$executed_svd_target_count > 0L)
  resource_metrics$svd_checkpoint_wait_count <-
    resource_metrics$svd_checkpoint_record_count
  resource_metrics$shadow_materialize_target_count <-
    setup_results$target_count
  resource_metrics$shadow_d2h_bytes <- 4096 * setup_results$target_count
  resource_metrics$invalid_output_init_count <- rep.int(1L, setup_count)
  resource_metrics$cusolver_deterministic_mode <- rep("enabled", setup_count)
  resource_metrics$cublas_math_mode <- rep("pedantic", setup_count)
  resource_metrics$cublas_atomics_mode <- rep("not_allowed", setup_count)
  resource_metrics$cublas_user_workspace_installed <- rep(TRUE, setup_count)
  resource_metrics$cublas_workspace_bytes <- rep(16777216, setup_count)
  resource_metrics$cublas_workspace_alignment <- rep(256, setup_count)

  stage_timing <- frame("stage_timing", 6L * setup_count)
  stage_timing$prepared_s_key_sha256 <- rep(shard_setup_keys, each = 6L)
  stage_timing$shard_id <- rep.int(as.integer(shard_id), 6L * setup_count)
  stage_timing$setup_ordinal <- rep(seq_len(setup_count), each = 6L)
  stage_timing$stage <- rep(c(
    "phase2_shard_load", "prepared_handle_create", "solve",
    "shadow_materialize", "cmagic_oracle", "release_and_free"
  ), setup_count)
  stage_timing$elapsed_ms <- rep(1, 6L * setup_count)
  fallbacks <- .fastkpc_full_cuda_phase3_oracle_fallback_rows(
    target_parity, resource_metrics
  )
  failures <- .fastkpc_full_cuda_phase3_oracle_failure_rows(target_parity)
  summary <- fastkpc_full_cuda_phase3_summarize_oracle_rows(
    setup_results, target_parity, resource_metrics, stage_timing,
    fallbacks, failures
  )
  payload <- list(
    setup_results = setup_results, target_parity = target_parity,
    resource_metrics = resource_metrics, stage_timing = stage_timing,
    fallbacks = fallbacks, failures = failures, summary = summary
  )
  payload$qualification_dcov_parity <- qualification_dcov[
    (qualification_dcov$logical_sequence_id - 1L) %% 4L == shard_id,
    , drop = FALSE
  ]
  rownames(payload$qualification_dcov_parity) <- NULL
  payload
}

executor <- function(context, shard_id, setup_keys, target_rows) {
  context$executor_count <- context$executor_count + 1L
  payload <- make_oracle_payload(shard_id, setup_keys, target_rows)
  count <- as.integer(length(setup_keys))
  list(
    payload = payload,
    resource_counts = list(
      prepared_handle_create_count = count,
      prepared_handle_destroy_count = count,
      residual_token_acquire_count = count,
      residual_token_release_count = count,
      output_slot_acquire_count = count,
      output_slot_release_count = count
    )
  )
}

risk_rows <- data.frame(
  residual_key_sha256 = target_rows$residual_key_sha256,
  high_condition = target_rows$planned_route == "AUGMENTED_SVD",
  rank_deficient = target_rows$coefficient_rank < target_rows$null_dim,
  nonfinite_metadata = seq_len(nrow(target_rows)) == 1L,
  near_constant_target = seq_len(nrow(target_rows)) %% 17L == 0L,
  near_constant_conditioner = seq_len(nrow(target_rows)) %% 19L == 0L,
  mgcv_warning = seq_len(nrow(target_rows)) %% 23L == 0L,
  mgcv_nonconverged = seq_len(nrow(target_rows)) %% 29L == 0L,
  near_alpha = seq_len(nrow(target_rows)) %% 31L == 0L,
  stringsAsFactors = FALSE
)
risk_rows$high_condition[[1L]] <- TRUE
risk_rows$rank_deficient[[1L]] <- TRUE

qualification_dcov <- data.frame(
  logical_sequence_id = 1:4,
  residual_key_x = target_rows$residual_key_sha256[1:4],
  residual_key_y = target_rows$residual_key_sha256[5:8],
  reference_p_value = c(0.2, 0.05, 0.8, 0.01),
  alpha = rep(0.05, 4L),
  p_value = c(0.2, 0.05 + 1e-12, 0.8, 0.01),
  near_alpha = c(FALSE, TRUE, FALSE, TRUE),
  backend = rep("cpp", 4L),
  low_rank_backend = rep("spectra", 4L),
  backend_error = rep(FALSE, 4L),
  spectra_fallback = rep(FALSE, 4L),
  stringsAsFactors = FALSE
)
qualification_dcov$p_value_difference <-
  qualification_dcov$p_value - qualification_dcov$reference_p_value
qualification_dcov$absolute_p_value_difference <-
  abs(qualification_dcov$p_value_difference)
qualification_dcov$reference_independent <-
  qualification_dcov$reference_p_value >= qualification_dcov$alpha
qualification_dcov$independent <-
  qualification_dcov$p_value >= qualification_dcov$alpha
qualification_dcov$decision_flip <-
  qualification_dcov$independent != qualification_dcov$reference_independent
qualification_logical_fixture <- qualification_dcov[, c(
  "logical_sequence_id", "residual_key_x", "residual_key_y",
  "reference_p_value", "alpha", "near_alpha"
), drop = FALSE]
qualification_endpoint_fixture <- sort(unique(c(
  qualification_logical_fixture$residual_key_x,
  qualification_logical_fixture$residual_key_y
)), method = "radix")
invisible(.fastkpc_full_cuda_phase3_validate_qualification_lineage(
  qualification_dcov, qualification_logical_fixture,
  qualification_endpoint_fixture
))
forged_logical_fixture <- qualification_logical_fixture
forged_logical_fixture$reference_p_value[[1L]] <- 0.3
assert_error(
  .fastkpc_full_cuda_phase3_validate_qualification_lineage(
    qualification_dcov, forged_logical_fixture,
    qualification_endpoint_fixture
  ),
  "qualification dCov rows reject forged canonical reference metadata"
)

output_dir <- tempfile("phase3-oracle-artifact-")
on.exit(unlink(output_dir, recursive = TRUE, force = TRUE), add = TRUE)
lifecycle <- new.env(parent = emptyenv())
lifecycle$create_count <- 0L
lifecycle$destroy_count <- 0L
runtime_create <- function() {
  lifecycle$create_count <- lifecycle$create_count + 1L
  context <- new.env(parent = emptyenv())
  context$executor_count <- 0L
  context
}
runtime_destroy <- function(context) {
  lifecycle$destroy_count <- lifecycle$destroy_count + 1L
  invisible(NULL)
}
run <- fastkpc_full_cuda_phase3_run_shards(
  output_dir = output_dir, kind = "oracle_sp",
  setup_keys = setup_keys, target_rows = target_rows,
  identity = identity, route_config = route_config,
  executor = executor, runtime_create = runtime_create,
  runtime_destroy = runtime_destroy, scope = "iteration", shard_count = 4L
)
assert_identical(run$written_shard_ids, 0:3,
                 "fixture writes four authenticated shards")

publish <- get0(
  "fastkpc_full_cuda_phase3_publish_oracle_artifact",
  mode = "function", inherits = TRUE
)
assert_true(!is.null(publish),
            "authenticated oracle artifact publisher is available")
published <- publish(
  output_dir = output_dir, setup_keys = setup_keys,
  target_rows = target_rows, identity = identity,
  route_config = route_config, scope = "iteration", shard_count = 4L,
  risk_rows = risk_rows, qualification_dcov = qualification_dcov,
  command_lines = "Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_oracle_artifact.R"
)
assert_identical(published$status, "published",
                 "first artifact publication reports published")

paths <- fastkpc_full_cuda_phase3_artifact_paths(output_dir, "oracle_sp")
payload_keys <- .fastkpc_full_cuda_phase3_payload_keys("oracle_sp")
assert_true(all(file.exists(unlist(paths[payload_keys], use.names = FALSE))),
            "every oracle artifact contract payload is published")
validated <- fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
  output_dir, expected_identity = identity, require_full = FALSE
)
assert_true(isTRUE(validated$authenticated) && isTRUE(validated$pass),
            "completed artifact validates independently")

merged <- fastkpc_full_cuda_phase3_merge_shards(
  output_dir = output_dir, kind = "oracle_sp",
  setup_keys = setup_keys, target_rows = target_rows,
  identity = identity, route_config = route_config,
  scope = "iteration", shard_count = 4L
)
recomputed <- fastkpc_full_cuda_phase3_summarize_oracle_rows(
  merged$payload$setup_results, merged$payload$target_parity,
  merged$payload$resource_metrics, merged$payload$stage_timing,
  merged$payload$fallbacks, merged$payload$failures
)
assert_identical(recomputed$setup_count, 44L, "recomputed setup count")
assert_identical(recomputed$target_count, 270L, "recomputed target count")
assert_identical(recomputed$non_ok_solver_status_count, 0L,
                 "recomputed non-OK count")
assert_identical(
  as.integer(recomputed[1L, c(
    "planned_cholesky_target_count", "planned_qr_target_count",
    "planned_svd_target_count"
  )]), c(172L, 31L, 67L), "recomputed planned routes"
)
assert_identical(
  as.integer(recomputed[1L, c(
    "executed_cholesky_target_count", "executed_qr_target_count",
    "executed_svd_target_count"
  )]), c(172L, 31L, 67L), "recomputed executed routes"
)
assert_identical(recomputed$cholesky_to_svd_count, 0L,
                 "recomputed Cholesky reroutes")
assert_identical(recomputed$qr_to_svd_count, 0L,
                 "recomputed QR reroutes")
assert_identical(
  as.integer(recomputed[1L, c(
    "cpu_fallback_count", "unknown_fallback_count",
    "approximate_backend_count"
  )]), c(0L, 0L, 0L), "recomputed fallback counts"
)
assert_identical(recomputed$summary_recomputed, TRUE,
                 "summary is marked recomputed")

risk_cases <- readRDS(paths$risk_cases_rds)
overlap <- risk_cases[
  risk_cases$residual_key_sha256 == risk_rows$residual_key_sha256[[1L]],
  , drop = FALSE
]
assert_true(nrow(overlap) == 1L && overlap$high_condition &&
              overlap$rank_deficient && overlap$nonfinite_metadata,
            "overlapping risk selectors survive the join")
qualification_rows <- readRDS(paths$qualification_dcov_rds)
assert_identical(qualification_rows$logical_sequence_id, 1:4,
                 "qualification rows use numeric logical order")

mtime_snapshot <- file.info(unlist(paths, use.names = FALSE))$mtime
byte_snapshot <- vapply(
  unlist(paths[setdiff(names(paths), c("shards_dir", "sessions_dir"))],
         use.names = FALSE),
  fastkpc_full_cuda_census_file_hash, character(1L)
)
resume <- publish(
  output_dir = output_dir, setup_keys = setup_keys,
  target_rows = target_rows, identity = identity,
  route_config = route_config, scope = "iteration", shard_count = 4L,
  risk_rows = risk_rows, qualification_dcov = qualification_dcov,
  command_lines = "Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_oracle_artifact.R"
)
assert_identical(resume$status, "reused",
                 "exact pure resume returns the completed artifact")
assert_identical(
  vapply(
    unlist(paths[setdiff(names(paths), c("shards_dir", "sessions_dir"))],
           use.names = FALSE),
    fastkpc_full_cuda_census_file_hash, character(1L)
  ), byte_snapshot, "pure resume leaves bytes unchanged"
)
assert_identical(file.info(unlist(paths, use.names = FALSE))$mtime,
                 mtime_snapshot, "pure resume leaves mtimes unchanged")
assert_identical(lifecycle$create_count, 1L,
                 "publication resume creates no CUDA context")

unlink(paths$summary_json, force = TRUE)
partial <- publish(
  output_dir = output_dir, setup_keys = setup_keys,
  target_rows = target_rows, identity = identity,
  route_config = route_config, scope = "iteration", shard_count = 4L,
  risk_rows = risk_rows, qualification_dcov = qualification_dcov,
  command_lines = "Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_oracle_artifact.R"
)
assert_identical(partial$status, "published",
                 "partial completion marker is republished gracefully")
assert_true(file.exists(paths$manifest_json) && file.exists(paths$summary_json),
            "partial resume restores both completion markers")

shard_session_paths <- c(
  list.files(paths$shards_dir, full.names = TRUE),
  list.files(paths$sessions_dir, full.names = TRUE)
)
shard_session_hashes <- vapply(
  shard_session_paths, fastkpc_full_cuda_census_file_hash, character(1L)
)
corrupt_manifest <- jsonlite::read_json(
  paths$manifest_json, simplifyVector = FALSE
)
corrupt_manifest$oracle_semantics_version <- "legacy-forged-semantics"
fastkpc_full_cuda_write_json(corrupt_manifest, paths$manifest_json)
corrupt_summary <- jsonlite::read_json(
  paths$summary_json, simplifyVector = FALSE
)
corrupt_summary$manifest_sha256 <-
  fastkpc_full_cuda_census_file_hash(paths$manifest_json)
fastkpc_full_cuda_write_json(corrupt_summary, paths$summary_json)
assert_error(
  fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
    output_dir, expected_identity = identity, require_full = FALSE
  ),
  "public validator rejects a corrupt pair of completion markers"
)
recovered <- publish(
  output_dir = output_dir, setup_keys = setup_keys,
  target_rows = target_rows, identity = identity,
  route_config = route_config, scope = "iteration", shard_count = 4L,
  risk_rows = risk_rows, qualification_dcov = qualification_dcov,
  command_lines = "Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_oracle_artifact.R"
)
assert_identical(recovered$status, "published",
                 "corrupt completion markers are republished")
assert_identical(
  vapply(shard_session_paths, fastkpc_full_cuda_census_file_hash, character(1L)),
  shard_session_hashes,
  "corrupt-marker recovery retains authenticated shard/session pairs"
)

unlink(paths$summary_json, force = TRUE)
original_oracle_validator <-
  fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact
assign(
  "fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact",
  function(...) stop("forced final artifact validation failure", call. = FALSE),
  envir = .GlobalEnv
)
forced_final_error <- tryCatch({
  publish(
    output_dir = output_dir, setup_keys = setup_keys,
    target_rows = target_rows, identity = identity,
    route_config = route_config, scope = "iteration", shard_count = 4L,
    risk_rows = risk_rows, qualification_dcov = qualification_dcov,
    command_lines = "Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_oracle_artifact.R"
  )
  NULL
}, error = function(error) error)
assign(
  "fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact",
  original_oracle_validator, envir = .GlobalEnv
)
assert_true(inherits(forced_final_error, "error"),
            "forced final artifact validation fails publication")
assert_true(!file.exists(paths$manifest_json) &&
              !file.exists(paths$summary_json),
            "failed final validation removes both completion markers")
recovered_after_final_failure <- publish(
  output_dir = output_dir, setup_keys = setup_keys,
  target_rows = target_rows, identity = identity,
  route_config = route_config, scope = "iteration", shard_count = 4L,
  risk_rows = risk_rows, qualification_dcov = qualification_dcov,
  command_lines = "Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_oracle_artifact.R"
)
assert_identical(recovered_after_final_failure$status, "published",
                 "publication resumes after final-validation cleanup")

pass_false_dir <- tempfile("phase3-oracle-pass-false-")
on.exit(unlink(pass_false_dir, recursive = TRUE, force = TRUE), add = TRUE)
dir.create(pass_false_dir, recursive = TRUE, showWarnings = FALSE)
assert_true(file.copy(output_dir, pass_false_dir, recursive = TRUE),
            "pass=false fixture copy succeeds")
pass_false_root <- file.path(pass_false_dir, basename(output_dir))
pass_false_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  pass_false_root, "oracle_sp"
)
pass_false_summary <- jsonlite::read_json(
  pass_false_paths$summary_json, simplifyVector = FALSE
)
pass_false_summary$pass <- FALSE
fastkpc_full_cuda_write_json(
  pass_false_summary, pass_false_paths$summary_json
)
assert_error(
  fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
    pass_false_root, expected_identity = identity, require_full = FALSE
  ),
  "persisted pass=false cannot replace validator-derived hard-gate success"
)

hostile_dir <- tempfile("phase3-oracle-hostile-")
on.exit(unlink(hostile_dir, recursive = TRUE, force = TRUE), add = TRUE)
dir.create(hostile_dir, recursive = TRUE, showWarnings = FALSE)
assert_true(file.copy(output_dir, hostile_dir, recursive = TRUE),
            "hostile fixture copy succeeds")
hostile_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  file.path(hostile_dir, basename(output_dir)), "oracle_sp"
)
hostile_target <- readRDS(hostile_paths$target_parity_rds)
hostile_target$fitted_max_abs_diff[[1L]] <- 1
saveRDS(hostile_target, hostile_paths$target_parity_rds, version = 3L)
hostile_manifest <- jsonlite::read_json(
  hostile_paths$manifest_json, simplifyVector = FALSE
)
hostile_manifest$payload_file_sha256[["target_parity.rds"]] <-
  fastkpc_full_cuda_census_file_hash(hostile_paths$target_parity_rds)
fastkpc_full_cuda_write_json(hostile_manifest, hostile_paths$manifest_json)
hostile_summary <- jsonlite::read_json(
  hostile_paths$summary_json, simplifyVector = FALSE
)
hostile_summary$pass <- TRUE
hostile_summary$manifest_sha256 <-
  fastkpc_full_cuda_census_file_hash(hostile_paths$manifest_json)
fastkpc_full_cuda_write_json(hostile_summary, hostile_paths$summary_json)
assert_error(
  fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
    dirname(hostile_paths$manifest_json), expected_identity = identity,
    require_full = FALSE
  ),
  "forged pass and payload hashes cannot hide semantic numeric corruption"
)

cat(
  "PASS Phase 3 fixed-sp oracle artifact:", recomputed$setup_count,
  "setups /", recomputed$target_count, "targets / four shards\n"
)
