source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_shadow.R")
source("fastkpc/R/full_cuda_ci_phase3_artifacts.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

assert_true(
  exists("fastkpc_full_cuda_phase3_publish_shadow_artifact",
         mode = "function"),
  "authenticated fixed-sp shadow artifact publisher should exist"
)
assert_true(
  exists("fastkpc_validate_full_cuda_fixed_sp_shadow_artifact",
         mode = "function"),
  "independent fixed-sp shadow artifact validator should exist"
)

assert_true(
  exists("fastkpc_full_cuda_shadow_merge_logical_rows", mode = "function"),
  "fixed-sp shadow direct/conditional normalizer should exist"
)

phase1_rows <- readRDS(file.path(
  "fastkpc", "artifacts", "full_cuda_ci",
  "workload_census_351x48_v1", "logical_ci_tests.rds"
))
direct_source <- phase1_rows[phase1_rows$logical_sequence_id == 1L,
                             , drop = FALSE]
conditional_source <- phase1_rows[
  phase1_rows$logical_sequence_id == 2214L, , drop = FALSE
]
direct <- data.frame(
  logical_sequence_id = as.integer(direct_source$logical_sequence_id),
  source_sequence_id = as.integer(direct_source$source_sequence_id),
  source_task_index = as.integer(direct_source$source_task_index),
  level = as.integer(direct_source$level),
  x = as.integer(direct_source$x), y = as.integer(direct_source$y),
  S_key = as.character(direct_source$S_key),
  residual_key_x = NA_character_, residual_key_y = NA_character_,
  reference_p_value = as.double(direct_source$reference_p_value),
  candidate_p_value = as.double(direct_source$reference_p_value),
  absolute_p_value_difference = 0,
  alpha = as.double(direct_source$alpha),
  reference_decision = as.character(direct_source$reference_decision),
  candidate_decision = as.character(direct_source$reference_decision),
  decision_flip = FALSE, backend = "legacy-cpp",
  low_rank_backend = "spectra", backend_error = FALSE,
  spectra_fallback = FALSE, stringsAsFactors = FALSE
)
conditional <- as.data.frame(lapply(
  fastkpc_full_cuda_shadow_conditional_row_schema(),
  function(type) switch(type, integer = 0L, double = 0,
                        logical = FALSE, character = "")
), stringsAsFactors = FALSE, optional = TRUE)
for (field in intersect(names(conditional_source), names(conditional))) {
  conditional[[field]] <- conditional_source[[field]]
}
conditional$logical_sequence_id <- 2214L
conditional$candidate_p_value <- conditional$reference_p_value
conditional$absolute_p_value_difference <- 0
conditional$candidate_decision <- conditional$reference_decision
conditional$decision_flip <- FALSE
conditional$near_alpha <- is.finite(
  conditional_source$absolute_log_distance_from_alpha
) && conditional_source$absolute_log_distance_from_alpha <= log(2)
conditional$near_alpha_bucket <- fastkpc_full_cuda_census_near_alpha_bucket(
  abs(log(pmax(conditional$reference_p_value, .Machine$double.xmin) /
            conditional$alpha))
)
conditional$prepared_s_key_sha256 <- strrep("a", 64L)
conditional$shard_id <- 0L
conditional$backend <- "cpp"
conditional$backend_version <- fastkpc_full_cuda_shadow_dcov_backend_version()
conditional$low_rank_backend <- "spectra"
for (endpoint in c("x", "y")) {
  conditional[[paste0("planned_route_", endpoint)]] <- "AUGMENTED_SVD"
  conditional[[paste0("executed_route_", endpoint)]] <- "AUGMENTED_SVD"
  conditional[[paste0("reroute_reason_", endpoint)]] <- ""
  conditional[[paste0("solver_status_", endpoint)]] <- "OK_AUGMENTED_SVD"
}
merged_rows <- fastkpc_full_cuda_shadow_merge_logical_rows(
  direct_rows = direct, conditional_rows = conditional,
  expected_logical_sequence_id = c(1L, 2214L)
)
assert_true(
  identical(merged_rows$logical_sequence_id, c(1L, 2214L)) &&
    identical(merged_rows$scope_sequence_ordinal, 1:2) &&
    identical(merged_rows$source_type, c("direct", "conditional")) &&
    !any(merged_rows$decision_flip),
  "merged logical rows preserve selected numeric order and derive decisions"
)

assert_true(
  exists("fastkpc_full_cuda_shadow_reconstruct_target_routes",
         mode = "function"),
  "fixed-sp shadow target route reconstructor should exist"
)
expected_target_keys <- sort(c(
  conditional$residual_key_x, conditional$residual_key_y
), method = "radix")
route_evidence <- fastkpc_full_cuda_shadow_reconstruct_target_routes(
  merged_rows, expected_target_keys = expected_target_keys,
  expected_target_corpus_hash =
    fastkpc_full_cuda_census_key_set_hash(expected_target_keys)
)
assert_true(
  nrow(route_evidence$target_routes) == 2L &&
    route_evidence$summary$planned_svd_target_count == 2L &&
    route_evidence$summary$executed_svd_target_count == 2L &&
    route_evidence$summary$stable_reroute_count == 0L,
  "target route evidence is reconstructed and conserved"
)
conflicting_rows <- rbind(merged_rows, merged_rows[2L, , drop = FALSE])
conflicting_rows$logical_sequence_id[[3L]] <- 2215L
conflicting_rows$planned_route_x[[3L]] <- "AUGMENTED_QR"
conflict <- tryCatch({
  fastkpc_full_cuda_shadow_reconstruct_target_routes(conflicting_rows)
  NULL
}, error = function(error) error)
assert_true(
  inherits(conflict, "error") && grepl(
    "conflicting repeated target route evidence",
    conditionMessage(conflict), fixed = TRUE
  ),
  "conflicting repeated target route observations fail closed"
)

phase0_dir <- file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "oracle_351x48_v1"
)
phase1_dir <- file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "workload_census_351x48_v1"
)
phase2_dir <- file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1"
)
data_path <- file.path(
  "fastkpc", "artifacts", "kpc_tprs_real_zhu",
  "cancer_RD-causalDiscoveryInput.rds"
)
catalog <- fastkpc_full_cuda_open_fixed_sp_catalog(
  phase0_dir, phase1_dir, phase2_dir, data_path
)
plan <- fastkpc_full_cuda_shadow_plan(catalog)
execution_snapshot <-
  fastkpc_full_cuda_phase3_create_shadow_execution_snapshot(
    catalog = catalog, plan = plan, scope = "iteration"
  )
on.exit(
  fastkpc_full_cuda_phase3_release_shadow_execution_snapshot(
    execution_snapshot
  ), add = TRUE
)
selected <- .fastkpc_full_cuda_phase3_resolve_shadow_execution_snapshot(
  execution_snapshot, expected_scope = "iteration"
)
logical_rows <- selected$logical_rows
target_rows <- selected$target_rows
setup_keys <- selected$setup_keys
setup_authority <- selected$setup_authority
fixture_assignments <- fastkpc_full_cuda_phase3_assign_setup_shards(
  setup_keys, 64L
)
setup_shard_match <- match(
  setup_keys, fixture_assignments$prepared_s_key_sha256
)
setup_authority$shard_id <-
  fixture_assignments$shard_id[setup_shard_match]
target_rows$shard_id <- fixture_assignments$shard_id[match(
  target_rows$prepared_s_key_sha256,
  fixture_assignments$prepared_s_key_sha256
)]
logical_rows$shard_id <- fixture_assignments$shard_id[match(
  logical_rows$prepared_s_key_x,
  fixture_assignments$prepared_s_key_sha256
)]
endpoint_keys <- sort(unique(c(
  logical_rows$residual_key_x, logical_rows$residual_key_y
)), method = "radix")
assert_true(
  nrow(logical_rows) == 44L && nrow(target_rows) == 270L &&
    length(setup_keys) == 44L &&
    identical(setup_authority$prepared_s_key_sha256, setup_keys),
  "fixture uses authenticated iteration logical/setup/target authority"
)

route_status <- function(route, target_index) {
  if (identical(route, "CHOLESKY_BATCHED")) {
    setup_key <- target_rows$prepared_s_key_sha256[[target_index]]
    batched <- sum(
      target_rows$prepared_s_key_sha256 == setup_key &
        target_rows$planned_route == "CHOLESKY_BATCHED"
    ) >= 2L
    return(if (batched) "OK_CHOLESKY_BATCHED" else "OK_CHOLESKY_SINGLE")
  }
  switch(route, AUGMENTED_QR = "OK_AUGMENTED_QR",
         AUGMENTED_SVD = "OK_AUGMENTED_SVD")
}
conditional_fixture <- as.data.frame(lapply(
  fastkpc_full_cuda_shadow_conditional_row_schema(),
  function(type) switch(type,
    integer = rep.int(0L, nrow(logical_rows)),
    double = rep.int(0, nrow(logical_rows)),
    logical = rep.int(FALSE, nrow(logical_rows)),
    character = rep.int("", nrow(logical_rows))
  )
), stringsAsFactors = FALSE)
for (field in intersect(names(logical_rows), names(conditional_fixture))) {
  conditional_fixture[[field]] <- logical_rows[[field]]
}
conditional_fixture$prepared_s_key_sha256 <-
  logical_rows$prepared_s_key_x
conditional_fixture$shard_id <- logical_rows$shard_id
conditional_fixture$candidate_p_value <- logical_rows$reference_p_value
conditional_fixture$absolute_p_value_difference <- rep.int(
  0, nrow(logical_rows)
)
conditional_fixture$candidate_decision <- logical_rows$reference_decision
conditional_fixture$decision_flip <- rep.int(FALSE, nrow(logical_rows))
distance <- abs(log(pmax(
  logical_rows$reference_p_value, .Machine$double.xmin
) / logical_rows$alpha))
conditional_fixture$near_alpha <- distance <= log(2)
conditional_fixture$near_alpha_bucket <- vapply(
  distance, fastkpc_full_cuda_census_near_alpha_bucket, character(1L)
)
conditional_fixture$backend <- rep.int("cpp", nrow(logical_rows))
conditional_fixture$backend_version <-
  rep.int(fastkpc_full_cuda_shadow_dcov_backend_version(),
          nrow(logical_rows))
conditional_fixture$low_rank_backend <- rep.int(
  "spectra", nrow(logical_rows)
)
for (endpoint in c("x", "y")) {
  key <- logical_rows[[paste0("residual_key_", endpoint)]]
  route_index <- match(key, target_rows$residual_key_sha256)
  route <- target_rows$planned_route[route_index]
  conditional_fixture[[paste0("planned_route_", endpoint)]] <- route
  conditional_fixture[[paste0("executed_route_", endpoint)]] <- route
  conditional_fixture[[paste0("reroute_reason_", endpoint)]] <-
    rep.int("", nrow(logical_rows))
  conditional_fixture[[paste0("solver_status_", endpoint)]] <-
    vapply(seq_along(route), function(index) {
      route_status(route[[index]], route_index[[index]])
    }, character(1L))
}
invisible(fastkpc_full_cuda_shadow_validate_conditional_rows(
  conditional_fixture, expected_logical_tests = logical_rows
))

oracle_frame <- function(name, row_count) {
  schema <- fastkpc_full_cuda_fixed_sp_oracle_row_schemas()[[name]]
  columns <- lapply(unname(schema), function(type) switch(
    type,
    character = rep.int("", row_count),
    integer = rep.int(0L, row_count),
    double = rep.int(0, row_count),
    logical = rep.int(FALSE, row_count)
  ))
  names(columns) <- names(schema)
  as.data.frame(columns, stringsAsFactors = FALSE, optional = TRUE)
}
target_setup <- match(
  target_rows$prepared_s_key_sha256, setup_keys
)
target_count <- as.integer(tabulate(
  target_setup, nbins = length(setup_keys)
))
setup_shard <- setup_authority$shard_id
setup_ordinal <- as.integer(ave(
  seq_along(setup_keys), setup_shard, FUN = seq_along
))
route_count <- function(route) as.integer(tabulate(
  target_setup[target_rows$planned_route == route],
  nbins = length(setup_keys)
))
planned_cholesky <- route_count("CHOLESKY_BATCHED")
planned_qr <- route_count("AUGMENTED_QR")
planned_svd <- route_count("AUGMENTED_SVD")
resource_fixture <- oracle_frame("resource_metrics", length(setup_keys))
resource_fixture$prepared_s_key_sha256 <- setup_keys
resource_fixture$shard_id <- setup_shard
resource_fixture$setup_ordinal <- setup_ordinal
resource_fixture$canonical_setup_rank <-
  setup_authority$canonical_setup_rank
resource_fixture$target_count <- target_count
resource_fixture$phase2_shard_load_count <- as.integer(!duplicated(setup_shard))
resource_fixture$phase2_shard_authentication_count <-
  resource_fixture$phase2_shard_load_count
resource_fixture$planned_cholesky_target_count <- planned_cholesky
resource_fixture$planned_qr_target_count <- planned_qr
resource_fixture$planned_svd_target_count <- planned_svd
resource_fixture$executed_cholesky_target_count <- planned_cholesky
resource_fixture$executed_qr_target_count <- planned_qr
resource_fixture$executed_svd_target_count <- planned_svd
resource_fixture$true_batched_subgroup_count <-
  as.integer(planned_cholesky >= 2L)
resource_fixture$true_batched_attempted_target_count <- as.integer(ifelse(
  planned_cholesky >= 2L, planned_cholesky, 0L
))
resource_fixture$true_batched_target_count <- as.integer(ifelse(
  planned_cholesky >= 2L, planned_cholesky, 0L
))
resource_fixture$aggregate_penalty_factor_count <- planned_svd
resource_fixture$aggregate_svd_b_build_count <- 2L * planned_svd
resource_fixture$cholesky_factor_checkpoint_record_count <-
  as.integer(planned_cholesky > 0L)
resource_fixture$cholesky_factor_checkpoint_wait_count <-
  resource_fixture$cholesky_factor_checkpoint_record_count
resource_fixture$cholesky_solve_checkpoint_record_count <-
  as.integer(planned_cholesky > 0L)
resource_fixture$cholesky_solve_checkpoint_wait_count <-
  resource_fixture$cholesky_solve_checkpoint_record_count
resource_fixture$qr_checkpoint_record_count <- as.integer(planned_qr > 0L)
resource_fixture$qr_checkpoint_wait_count <-
  resource_fixture$qr_checkpoint_record_count
resource_fixture$svd_checkpoint_record_count <- as.integer(planned_svd > 0L)
resource_fixture$svd_checkpoint_wait_count <-
  resource_fixture$svd_checkpoint_record_count
resource_fixture$coefficient_batch_finalize_call_count <-
  as.integer(planned_cholesky > 0L)
resource_fixture$fitted_batch_finalize_call_count <-
  as.integer(planned_cholesky > 0L)
resource_fixture$residual_rss_batch_finalize_call_count <-
  as.integer(planned_cholesky > 0L)
resource_fixture$per_target_output_finalize_call_count <-
  planned_qr + planned_svd
resource_fixture$batch_output_finalized_target_count <- target_count
for (field in c(
  "prepared_handle_create_count", "prepared_handle_destroy_count",
  "residual_token_acquire_count", "residual_token_release_count",
  "output_slot_acquire_count", "output_slot_release_count",
  "setup_h2d_upload_count", "target_batch_h2d_call_count",
  "rhs_device_build_count", "shadow_materialize_call_count",
  "invalid_output_init_count"
)) resource_fixture[[field]] <- rep.int(1L, length(setup_keys))
explicit_constraint <- setup_authority$constraint_mode == "explicit"
resource_fixture$setup_h2d_bytes <- 8 * (
  as.double(setup_authority$n) * as.double(setup_authority$coefficient_dim) +
    as.double(explicit_constraint) * (
      as.double(setup_authority$coefficient_dim) *
        as.double(setup_authority$null_dim) +
      as.double(setup_authority$n) * as.double(setup_authority$null_dim)
    ) + as.double(setup_authority$null_dim)^2 +
    as.double(setup_authority$penalty_count) *
      as.double(setup_authority$null_dim)^2 +
    as.double(setup_authority$has_H) *
      as.double(setup_authority$null_dim)^2
)
resource_fixture$target_h2d_copy_count <- rep.int(2L, length(setup_keys))
resource_fixture$target_h2d_bytes <-
  8 * as.double(target_count) * as.double(
    setup_authority$n + setup_authority$penalty_count
  )
resource_fixture$rhs_authority <- rep.int(
  "cuda-x0-transpose-y", length(setup_keys)
)
resource_fixture$full_cuda_data_plane <- rep.int(TRUE, length(setup_keys))
resource_fixture$shadow_materialize_target_count <- target_count
resource_fixture$shadow_d2h_bytes <-
  8 * as.double(target_count) * as.double(
    setup_authority$coefficient_dim + 2L * setup_authority$n + 1L +
      setup_authority$null_dim
  )
resource_fixture$cusolver_deterministic_mode <- rep.int(
  "enabled", length(setup_keys)
)
resource_fixture$cublas_math_mode <- rep.int(
  "pedantic", length(setup_keys)
)
resource_fixture$cublas_atomics_mode <- rep.int(
  "not_allowed", length(setup_keys)
)
resource_fixture$cublas_user_workspace_installed <- rep.int(
  TRUE, length(setup_keys)
)
resource_fixture$cublas_workspace_bytes <- rep.int(
  16777216, length(setup_keys)
)
resource_fixture$cublas_workspace_alignment <- rep.int(
  256, length(setup_keys)
)
stage_fixture <- oracle_frame("stage_timing", 6L * length(setup_keys))
stage_fixture$prepared_s_key_sha256 <- rep(setup_keys, each = 6L)
stage_fixture$shard_id <- rep(setup_shard, each = 6L)
stage_fixture$setup_ordinal <- rep(setup_ordinal, each = 6L)
stage_fixture$stage <- rep(c(
  "phase2_shard_load", "prepared_handle_create", "solve",
  "shadow_materialize", "cmagic_oracle", "release_and_free"
), length(setup_keys))
stage_fixture$elapsed_ms <- rep.int(0, nrow(stage_fixture))
payload_fixture <- list(
  logical_ci_parity = conditional_fixture,
  resource_metrics = resource_fixture,
  stage_timing = stage_fixture
)
invisible(fastkpc_full_cuda_phase3_validate_shadow_payload(
  payload_fixture, expected_setup_keys = setup_keys,
  expected_target_rows = target_rows,
  expected_logical_tests = logical_rows,
  require_logical_authority = TRUE,
  expected_setup_rows = setup_authority
))
assert_true(
  all(c(
    "planned_cholesky_target_count", "planned_qr_target_count",
    "planned_svd_target_count", "executed_cholesky_target_count",
    "executed_qr_target_count", "executed_svd_target_count",
    "cholesky_to_svd_count", "qr_to_svd_count", "stable_reroute_count"
  ) %in% names(fastkpc_full_cuda_shadow_runtime_counters(
    resource_fixture
  ))),
  "runtime aggregate exposes every independently checked route counter"
)

sha <- fastkpc_full_cuda_census_hash_utf8
route_config <- fastkpc_full_cuda_phase3_route_config()
identity <- list(
  schema_version = "full-cuda-ci-phase3-test-input-identity-v1",
  canonical_setup_corpus_hash =
    fastkpc_full_cuda_census_key_set_hash(setup_keys),
  canonical_target_corpus_hash =
    fastkpc_full_cuda_census_key_set_hash(
      sort(target_rows$residual_key_sha256, method = "radix")
    ),
  route_config_hash = route_config$sha256,
  source_commit = strrep("1", 40L), cuda_toolkit_version = 12040L,
  cuda_driver_version = 55054L, gpu_name = "Synthetic GPU",
  gpu_uuid = paste0("GPU-", strrep("a", 32L)),
  compute_capability_major = 8L, compute_capability_minor = 0L,
  compute_capability = "8.0", sm_count = 108L, device_id = 0L,
  cusolver_deterministic_mode_required = "enabled",
  cublas_math_mode_required = "pedantic",
  cublas_atomics_mode_required = "not_allowed",
  cublas_user_workspace_required = TRUE,
  cublas_workspace_bytes_required = 16777216,
  cublas_workspace_min_alignment_required = 256
)
identity$sha256 <- .fastkpc_full_cuda_phase3_named_hash(identity)

output_dir <- tempfile("phase3-shadow-artifact-")
oracle_sp_dir <- tempfile("phase3-shadow-oracle-sp-")
on.exit(unlink(c(output_dir, oracle_sp_dir), recursive = TRUE, force = TRUE),
        add = TRUE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

direct_authority <- phase1_rows[phase1_rows$level == 0L, , drop = FALSE]
direct_fixture <- data.frame(
  logical_sequence_id = as.integer(direct_authority$logical_sequence_id),
  source_sequence_id = as.integer(direct_authority$source_sequence_id),
  source_task_index = as.integer(direct_authority$source_task_index),
  level = as.integer(direct_authority$level),
  x = as.integer(direct_authority$x), y = as.integer(direct_authority$y),
  S_key = as.character(direct_authority$S_key),
  residual_key_x = as.character(direct_authority$residual_key_x),
  residual_key_y = as.character(direct_authority$residual_key_y),
  reference_p_value = as.double(direct_authority$reference_p_value),
  candidate_p_value = as.double(direct_authority$reference_p_value),
  absolute_p_value_difference = rep.int(0, nrow(direct_authority)),
  alpha = as.double(direct_authority$alpha),
  reference_decision = as.character(direct_authority$reference_decision),
  candidate_decision = as.character(direct_authority$reference_decision),
  decision_flip = rep.int(FALSE, nrow(direct_authority)),
  backend = rep.int("legacy-cpp", nrow(direct_authority)),
  low_rank_backend = rep.int("spectra", nrow(direct_authority)),
  backend_error = rep.int(FALSE, nrow(direct_authority)),
  spectra_fallback = rep.int(FALSE, nrow(direct_authority)),
  stringsAsFactors = FALSE
)
direct_payload <- .fastkpc_full_cuda_phase3_direct_ci_payload(
  direct_fixture, catalog
)
artifact_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  output_dir, "full_shadow"
)
saveRDS(direct_payload, artifact_paths$direct_ci_rds,
        version = 2, compress = FALSE)
direct_summary <- .fastkpc_full_cuda_phase3_direct_ci_summary(
  direct_payload,
  fastkpc_full_cuda_census_file_hash(artifact_paths$direct_ci_rds)
)
.fastkpc_full_cuda_phase3_write_json_exact(
  direct_summary, artifact_paths$direct_ci_summary_json
)
invisible(fastkpc_full_cuda_phase3_validate_direct_ci_payload(
  output_dir, catalog
))

context_count <- 0L
run <- fastkpc_full_cuda_phase3_run_shards(
  output_dir = output_dir, kind = "full_shadow",
  setup_keys = setup_keys, target_rows = target_rows,
  identity = identity, route_config = route_config,
  executor = function(context, shard_id, setup_keys, target_rows) {
    rows <- conditional_fixture[
      conditional_fixture$shard_id == shard_id, , drop = FALSE
    ]
    resources <- resource_fixture[
      resource_fixture$shard_id == shard_id, , drop = FALSE
    ]
    stages <- stage_fixture[
      stage_fixture$shard_id == shard_id, , drop = FALSE
    ]
    rownames(rows) <- rownames(resources) <- rownames(stages) <- NULL
    count <- as.integer(length(setup_keys))
    list(
      payload = list(
        logical_ci_parity = rows,
        resource_metrics = resources,
        stage_timing = stages
      ),
      resource_counts = list(
        prepared_handle_create_count = count,
        prepared_handle_destroy_count = count,
        residual_token_acquire_count = count,
        residual_token_release_count = count,
        output_slot_acquire_count = count,
        output_slot_release_count = count
      )
    )
  },
  runtime_create = function() {
    context_count <<- context_count + 1L
    new.env(parent = emptyenv())
  },
  runtime_destroy = function(context) invisible(NULL),
  scope = "iteration", shard_count = 64L
)
assert_true(identical(run$status, "complete") && context_count == 1L,
            "fixture shard source is complete")

make_oracle_payload <- function(shard_id, shard_setup_keys, shard_targets) {
  setup_count <- length(shard_setup_keys)
  target_count_value <- nrow(shard_targets)
  if (setup_count == 0L) {
    setup_results <- oracle_frame("setup_results", 0L)
    target_parity <- oracle_frame("target_parity", 0L)
    resource_metrics <- oracle_frame("resource_metrics", 0L)
    stage_timing <- oracle_frame("stage_timing", 0L)
    fallbacks <- .fastkpc_full_cuda_phase3_oracle_fallback_rows(
      target_parity, resource_metrics
    )
    failures <- .fastkpc_full_cuda_phase3_oracle_failure_rows(target_parity)
    summary <- fastkpc_full_cuda_phase3_summarize_oracle_rows(
      setup_results, target_parity, resource_metrics, stage_timing,
      fallbacks, failures
    )
    return(list(
      setup_results = setup_results, target_parity = target_parity,
      resource_metrics = resource_metrics, stage_timing = stage_timing,
      fallbacks = fallbacks, failures = failures, summary = summary
    ))
  }
  setup_rank <- match(
    shard_targets$prepared_s_key_sha256, shard_setup_keys
  )
  target_parity <- oracle_frame("target_parity", target_count_value)
  target_parity$prepared_s_key_sha256 <-
    shard_targets$prepared_s_key_sha256
  target_parity$shard_id <- rep.int(as.integer(shard_id), target_count_value)
  target_parity$setup_ordinal <- as.integer(setup_rank)
  target_parity$canonical_setup_rank <-
    shard_targets$canonical_setup_rank
  target_parity$target_ordinal <- as.integer(ave(
    seq_len(target_count_value), shard_targets$prepared_s_key_sha256,
    FUN = seq_along
  ))
  target_parity$canonical_target_rank <-
    shard_targets$canonical_target_rank
  target_parity$residual_key_sha256 <-
    shard_targets$residual_key_sha256
  target_parity$target <- shard_targets$target
  target_parity$null_dim <- shard_targets$null_dim
  target_parity$condition <- shard_targets$condition
  target_parity$condition_bucket <- vapply(
    seq_len(target_count_value), function(index) {
      fastkpc_full_cuda_census_condition_bucket(
        target_parity$condition[[index]],
        shard_targets$coefficient_rank[[index]],
        target_parity$null_dim[[index]]
      )
    }, character(1L)
  )
  target_parity$phase1_coefficient_rank <-
    shard_targets$coefficient_rank
  target_parity$planned_route <- shard_targets$planned_route
  target_parity$authenticated_planned_route <- shard_targets$planned_route
  target_parity$executed_route <- shard_targets$planned_route
  target_parity$reroute_reason <- rep.int("", target_count_value)
  cholesky <- target_parity$planned_route == "CHOLESKY_BATCHED"
  qr <- target_parity$planned_route == "AUGMENTED_QR"
  svd <- target_parity$planned_route == "AUGMENTED_SVD"
  cholesky_count <- vapply(shard_setup_keys, function(key) {
    sum(cholesky[target_parity$prepared_s_key_sha256 == key])
  }, integer(1L))
  true_batched <- cholesky & cholesky_count[setup_rank] >= 2L
  target_parity$solver_status <- ifelse(
    cholesky,
    ifelse(true_batched, "OK_CHOLESKY_BATCHED", "OK_CHOLESKY_SINGLE"),
    ifelse(qr, "OK_AUGMENTED_QR", "OK_AUGMENTED_SVD")
  )
  target_parity$target_true_batched <- true_batched
  target_parity$true_batched_kernel <- vapply(
    seq_len(target_count_value), function(index) {
      selected_target <- setup_rank == setup_rank[[index]]
      sum(selected_target) >= 2L && all(true_batched[selected_target])
    }, logical(1L)
  )
  target_parity$true_batched_target_count <-
    as.integer(ifelse(
      cholesky_count[setup_rank] >= 2L,
      cholesky_count[setup_rank], 0L
    ))
  target_parity$qr_rank <- as.integer(ifelse(
    qr, target_parity$null_dim, -1L
  ))
  target_parity$geqrf_info <- as.integer(ifelse(qr, 0L, -1L))
  target_parity$ormqr_info <- as.integer(ifelse(qr, 0L, -1L))
  target_parity$effective_rank <- as.integer(ifelse(
    svd, pmax(1L, pmin(
      shard_targets$coefficient_rank, target_parity$null_dim
    )), -1L
  ))
  target_parity$sigma_max <- ifelse(svd, 10, NaN)
  target_parity$smallest_retained_sigma <- ifelse(svd, 1, NaN)
  target_parity$svd_info <- as.integer(ifelse(svd, 0L, -1L))
  target_parity$aggregate_penalty_root_rank <- as.integer(ifelse(
    svd, pmax(1L, pmin(
      shard_targets$coefficient_rank, target_parity$null_dim
    )), NA_integer_
  ))
  target_parity$aggregate_penalty_root_pivot_sha256 <- vapply(
    shard_targets$residual_key_sha256,
    function(key) sha(paste0("pivot-", key)), character(1L)
  )
  target_parity$aggregate_factor_call_count <- as.integer(svd)
  target_parity$aggregate_b_build_count <- 2L * as.integer(svd)
  target_parity$aggregate_dstop <- ifelse(svd, 1e-12, NA_real_)
  target_parity$numeric_reference <- rep.int(
    "mgcv-fixed-sp", target_count_value
  )
  for (field in c(
    "coefficient_all_finite", "fitted_all_finite",
    "residual_all_finite", "rss_all_finite", "rhs_all_finite",
    "output_all_finite", "coefficient_oracle_phase2_exact",
    "fitted_oracle_phase2_exact", "residual_oracle_phase2_exact",
    "full_cuda_data_plane"
  )) target_parity[[field]] <- rep.int(TRUE, target_count_value)
  for (field in grep(
    "_(max_abs_diff|relative_l2)$", names(target_parity), value = TRUE
  )) target_parity[[field]] <- rep.int(0, target_count_value)
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
  target_parity$coefficient_phase2_sha256 <- shard_targets$coefficient_hash
  target_parity$fitted_phase2_sha256 <- shard_targets$fitted_hash
  target_parity$residual_phase2_sha256 <- shard_targets$residual_hash
  target_parity$target_fit_fingerprint <-
    shard_targets$target_fit_fingerprint
  target_parity$oracle_call_count <- rep.int(1L, target_count_value)
  target_parity$rhs_authority <- rep.int(
    "cuda-x0-transpose-y", target_count_value
  )
  target_parity$approximate_backend <- rep.int(FALSE, target_count_value)
  target_parity$fallback_type <- rep.int("NONE", target_count_value)
  target_parity$error_code <- rep.int("NONE", target_count_value)
  target_parity$error_message_sha256 <- rep.int(
    sha(""), target_count_value
  )

  authority_match <- match(
    shard_setup_keys, setup_authority$prepared_s_key_sha256
  )
  shard_setup_authority <- setup_authority[authority_match, , drop = FALSE]
  setup_results <- oracle_frame("setup_results", setup_count)
  setup_results$prepared_s_key_sha256 <- shard_setup_keys
  setup_results$shard_id <- rep.int(as.integer(shard_id), setup_count)
  setup_results$setup_ordinal <- seq_len(setup_count)
  setup_results$canonical_setup_rank <-
    shard_setup_authority$canonical_setup_rank
  setup_results$phase2_shard_id <- as.integer(vapply(
    shard_setup_keys, function(key) unique(
      shard_targets$phase2_shard_id[
        shard_targets$prepared_s_key_sha256 == key
      ]
    ), integer(1L)
  ))
  oracle_phase2_unit <- as.integer(!duplicated(data.frame(
    shard_id = setup_results$shard_id,
    phase2_shard_id = setup_results$phase2_shard_id
  )))
  setup_results$phase2_shard_load_count <- oracle_phase2_unit
  setup_results$phase2_shard_authentication_count <- oracle_phase2_unit
  setup_results$n <- shard_setup_authority$n
  setup_results$coefficient_dim <- shard_setup_authority$coefficient_dim
  setup_results$null_dim <- shard_setup_authority$null_dim
  setup_results$penalty_count <- shard_setup_authority$penalty_count
  setup_results$target_count <- as.integer(vapply(
    shard_setup_keys, function(key) {
      sum(target_parity$prepared_s_key_sha256 == key)
    }, integer(1L)
  ))
  setup_results$target_key_set_sha256 <- vapply(
    shard_setup_keys, function(key) {
      fastkpc_full_cuda_census_key_set_hash(
        target_parity$residual_key_sha256[
          target_parity$prepared_s_key_sha256 == key
        ]
      )
    }, character(1L)
  )
  setup_results$prepared_handle_create_count <- rep.int(1L, setup_count)
  setup_results$prepared_handle_destroy_count <- rep.int(1L, setup_count)
  setup_results$setup_h2d_upload_count <- rep.int(1L, setup_count)
  setup_results$setup_h2d_bytes <-
    resource_fixture$setup_h2d_bytes[match(shard_setup_keys, setup_keys)]
  setup_results$penalty_root_build_count <-
    shard_setup_authority$penalty_count
  setup_results$penalty_root_matrix_count <-
    shard_setup_authority$penalty_count
  setup_results$penalty_root_row_count <-
    shard_setup_authority$penalty_count * shard_setup_authority$null_dim
  route_count_for_setup <- function(key, route) sum(
    target_parity$prepared_s_key_sha256 == key &
      target_parity$planned_route == route
  )
  for (prefix in c("planned", "executed")) {
    for (route_name in c("cholesky", "qr", "svd")) {
      route <- c(
        cholesky = "CHOLESKY_BATCHED", qr = "AUGMENTED_QR",
        svd = "AUGMENTED_SVD"
      )[[route_name]]
      setup_results[[paste0(prefix, "_", route_name, "_target_count")]] <-
        as.integer(vapply(
          shard_setup_keys, route_count_for_setup, integer(1L), route = route
        ))
    }
  }
  setup_results$true_batched_target_count <- as.integer(ifelse(
    cholesky_count >= 2L, cholesky_count, 0L
  ))
  setup_results$setup_load_elapsed_ms <- rep.int(0, setup_count)
  setup_results$total_elapsed_ms <- rep.int(0, setup_count)

  resource_metrics <- resource_fixture[
    match(shard_setup_keys, resource_fixture$prepared_s_key_sha256),
    , drop = FALSE
  ]
  resource_metrics$shard_id <- rep.int(as.integer(shard_id), setup_count)
  resource_metrics$setup_ordinal <- seq_len(setup_count)
  resource_metrics$phase2_shard_load_count <- oracle_phase2_unit
  resource_metrics$phase2_shard_authentication_count <- oracle_phase2_unit
  rownames(resource_metrics) <- NULL
  stage_timing <- do.call(rbind, lapply(seq_along(shard_setup_keys),
    function(index) {
      rows <- stage_fixture[
        stage_fixture$prepared_s_key_sha256 == shard_setup_keys[[index]],
        , drop = FALSE
      ]
      rows$shard_id <- rep.int(as.integer(shard_id), nrow(rows))
      rows$setup_ordinal <- rep.int(as.integer(index), nrow(rows))
      rows
    }
  ))
  rownames(stage_timing) <- NULL
  fallbacks <- .fastkpc_full_cuda_phase3_oracle_fallback_rows(
    target_parity, resource_metrics
  )
  failures <- .fastkpc_full_cuda_phase3_oracle_failure_rows(target_parity)
  summary <- fastkpc_full_cuda_phase3_summarize_oracle_rows(
    setup_results, target_parity, resource_metrics, stage_timing,
    fallbacks, failures
  )
  list(
    setup_results = setup_results, target_parity = target_parity,
    resource_metrics = resource_metrics, stage_timing = stage_timing,
    fallbacks = fallbacks, failures = failures, summary = summary
  )
}

oracle_context_count <- 0L
oracle_target_rows <- target_rows
oracle_target_rows$shard_id <- rep.int(0L, nrow(oracle_target_rows))
oracle_run <- fastkpc_full_cuda_phase3_run_shards(
  output_dir = oracle_sp_dir, kind = "oracle_sp",
  setup_keys = setup_keys, target_rows = oracle_target_rows,
  identity = identity, route_config = route_config,
  executor = function(context, shard_id, setup_keys, target_rows) {
    payload <- make_oracle_payload(
      shard_id, setup_keys, target_rows
    )
    tryCatch(
      .fastkpc_full_cuda_phase3_validate_oracle_payload(
        payload, expected_setup_keys = setup_keys,
        expected_target_rows = target_rows
      ),
      error = function(error) {
        field_map <- c(
          prepared_s_key_sha256 = "prepared_s_key_sha256",
          residual_key_sha256 = "residual_key_sha256",
          shard_id = "shard_id",
          canonical_setup_rank = "canonical_setup_rank",
          canonical_target_rank = "canonical_target_rank",
          target = "target", null_dim = "null_dim",
          condition = "condition",
          coefficient_rank = "phase1_coefficient_rank",
          planned_route = "planned_route",
          selected_sp_hash = "selected_sp_sha256",
          coefficient_hash = "coefficient_phase2_sha256",
          fitted_hash = "fitted_phase2_sha256",
          residual_hash = "residual_phase2_sha256",
          target_fit_fingerprint = "target_fit_fingerprint"
        )
        mismatched <- names(field_map)[!vapply(
          names(field_map), function(field) identical(
            payload$target_parity[[field_map[[field]]]],
            target_rows[[field]]
          ), logical(1L)
        )]
        stop(
          "oracle fixture shard ", shard_id, ": ",
          conditionMessage(error), "; target fields=",
          paste(mismatched, collapse = ","), call. = FALSE
        )
      }
    )
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
  },
  runtime_create = function() {
    oracle_context_count <<- oracle_context_count + 1L
    new.env(parent = emptyenv())
  },
  runtime_destroy = function(context) invisible(NULL),
  scope = "iteration", shard_count = 1L
)
assert_true(
  identical(oracle_run$status, "complete") && oracle_context_count == 1L,
  "oracle fixture writes authenticated Phase 3 shard/session evidence"
)
oracle_merged <- fastkpc_full_cuda_phase3_merge_shards(
  output_dir = oracle_sp_dir, kind = "oracle_sp",
  setup_keys = setup_keys, target_rows = oracle_target_rows,
  identity = identity, route_config = route_config,
  scope = "iteration", shard_count = 1L
)
assert_true(
  identical(
    fastkpc_full_cuda_census_key_set_hash(
      sort(
        oracle_merged$payload$target_parity$residual_key_sha256,
        method = "radix"
      )
    ), identity$canonical_target_corpus_hash
  ),
  "merged oracle shard corpus matches the test identity"
)
risk_rows <- data.frame(
  residual_key_sha256 = target_rows$residual_key_sha256,
  high_condition = rep.int(FALSE, nrow(target_rows)),
  rank_deficient = rep.int(FALSE, nrow(target_rows)),
  nonfinite_metadata = rep.int(FALSE, nrow(target_rows)),
  near_constant_target = rep.int(FALSE, nrow(target_rows)),
  near_constant_conditioner = rep.int(FALSE, nrow(target_rows)),
  mgcv_warning = rep.int(FALSE, nrow(target_rows)),
  mgcv_nonconverged = rep.int(FALSE, nrow(target_rows)),
  near_alpha = rep.int(FALSE, nrow(target_rows)),
  stringsAsFactors = FALSE
)
oracle_publication <- tryCatch(
  fastkpc_full_cuda_phase3_publish_oracle_artifact(
    output_dir = oracle_sp_dir, setup_keys = setup_keys,
    target_rows = oracle_target_rows, identity = identity,
    route_config = route_config, scope = "iteration", shard_count = 1L,
    risk_rows = risk_rows,
    command_lines =
      "Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_shadow_artifact.R"
  ),
  error = function(error) {
    manifest_path <- file.path(oracle_sp_dir, "manifest.json")
    manifest <- if (file.exists(manifest_path)) {
      jsonlite::read_json(manifest_path, simplifyVector = TRUE)
    } else list(expected_target_hash = "missing")
    target_path <- file.path(oracle_sp_dir, "target_parity.rds")
    published_target_hash <- if (file.exists(target_path)) {
      fastkpc_full_cuda_census_key_set_hash(
        readRDS(target_path)$residual_key_sha256
      )
    } else "missing"
    stop(
      conditionMessage(error), "; identity target=",
      identity$canonical_target_corpus_hash, "; manifest target=",
      manifest$expected_target_hash, "; payload target=",
      published_target_hash, call. = FALSE
    )
  }
)
assert_true(
  identical(oracle_publication$status, "published") &&
    isTRUE(oracle_publication$validation$authenticated),
  "oracle fixture is completed through the Phase 3 publisher"
)
oracle_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  oracle_sp_dir, "oracle_sp"
)

publish_args <- list(
  output_dir = output_dir, catalog = catalog, setup_keys = setup_keys,
  target_rows = target_rows, identity = identity,
  route_config = route_config, scope = "iteration",
  phase0_dir = phase0_dir, oracle_sp_dir = oracle_sp_dir,
  shard_count = 64L, direct_logical_sequence_id = 1L,
  command_lines =
    "Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_shadow_artifact.R"
)
published <- do.call(
  fastkpc_full_cuda_phase3_publish_shadow_artifact, publish_args
)
assert_true(
  identical(published$status, "published") &&
    isTRUE(published$validation$authenticated) &&
    !isTRUE(published$summary$full_scope) &&
    identical(published$summary$first_divergence, "NOT_APPLICABLE"),
  "authenticated selected-scope shadow artifact publishes and validates"
)
subprocess_sources <- c(
  "fastkpc/R/full_cuda_ci_gate.R",
  "fastkpc/R/full_cuda_ci_oracle_contract.R",
  "fastkpc/R/full_cuda_ci_workload_census.R",
  "fastkpc/R/full_cuda_ci_prepared_s_contract.R",
  "fastkpc/R/full_cuda_ci_fixed_sp_runtime.R",
  "fastkpc/R/full_cuda_ci_fixed_sp_shadow.R",
  "fastkpc/R/full_cuda_ci_phase3_artifacts.R"
)
subprocess_expression <- paste0(
  paste(sprintf(
    "source(%s)", encodeString(subprocess_sources, quote = "\"")
  ), collapse = ";"),
  ";value <- fastkpc_validate_full_cuda_fixed_sp_shadow_artifact(",
  encodeString(
    normalizePath(output_dir, winslash = "/", mustWork = TRUE),
    quote = "\""
  ),
  ", require_full=FALSE);stopifnot(isTRUE(value$authenticated),",
  "!is.null(value$recomputed_graph));cat('TASK9_RECOMPUTED\\n')"
)
subprocess_output <- system2(
  "Rscript", c("--vanilla", "-e", shQuote(subprocess_expression)),
  stdout = TRUE, stderr = TRUE
)
assert_true(
  is.null(attr(subprocess_output, "status")) &&
    any(subprocess_output == "TASK9_RECOMPUTED"),
  paste(
    "clean-process documented public validator call performs Task 9",
    "source/oracle/graph recomputation"
  )
)
full_error <- tryCatch({
  fastkpc_validate_full_cuda_fixed_sp_shadow_artifact(
    output_dir, require_full = TRUE
  )
  NULL
}, error = function(error) error)
assert_true(
  inherits(full_error, "error") && grepl(
    "non-full shadow artifact", conditionMessage(full_error), fixed = TRUE
  ),
  "non-full artifact cannot satisfy require_full=TRUE"
)

shadow_shard_paths <- sort(list.files(
  artifact_paths$shards_dir,
  pattern = "^shard_[0-9]+\\.(rds|summary\\.json)$",
  full.names = TRUE
), method = "radix")
shadow_session_paths <- sort(list.files(
  artifact_paths$sessions_dir, full.names = TRUE
), method = "radix")
resume_byte_paths <- c(
  artifact_paths$direct_ci_rds, artifact_paths$direct_ci_summary_json,
  shadow_shard_paths, shadow_session_paths,
  unlist(artifact_paths[.fastkpc_full_cuda_phase3_shadow_publication_keys()],
         use.names = FALSE)
)
resume_byte_hashes <- vapply(
  resume_byte_paths, fastkpc_full_cuda_census_file_hash, character(1L)
)
semantic_snapshot <- function() {
  shard_rds <- shadow_shard_paths[grepl("\\.rds$", shadow_shard_paths)]
  list(
    direct_rows = fastkpc_full_cuda_census_frame_hash(
      readRDS(artifact_paths$direct_ci_rds)$rows
    ),
    direct_summary = .fastkpc_full_cuda_phase3_named_hash(
      jsonlite::read_json(
        artifact_paths$direct_ci_summary_json, simplifyVector = FALSE
      )
    ),
    shard_payload = unname(vapply(shard_rds, function(path) {
      readRDS(path)$payload_semantic_hash
    }, character(1L))),
    logical_rows = fastkpc_full_cuda_census_frame_hash(
      readRDS(artifact_paths$logical_ci_parity_rds)
    ),
    adjacency = digest::digest(
      readRDS(artifact_paths$adjacency_rds),
      algo = "sha256", serialize = TRUE
    ),
    deletion = digest::digest(
      read.csv(artifact_paths$deletion_trace_csv,
               stringsAsFactors = FALSE, check.names = FALSE),
      algo = "sha256", serialize = TRUE
    ),
    sepset = digest::digest(
      read.csv(artifact_paths$sepset_agreement_csv,
               stringsAsFactors = FALSE, check.names = FALSE),
      algo = "sha256", serialize = TRUE
    ),
    n_edgetests = digest::digest(
      read.csv(artifact_paths$n_edgetests_csv,
               stringsAsFactors = FALSE, check.names = FALSE),
      algo = "sha256", serialize = TRUE
    ),
    first_divergence = .fastkpc_full_cuda_phase3_named_hash(
      jsonlite::read_json(
        artifact_paths$first_divergence_json, simplifyVector = FALSE
      )
    ),
    manifest = .fastkpc_full_cuda_phase3_named_hash(
      jsonlite::read_json(
        artifact_paths$manifest_json, simplifyVector = FALSE
      )
    ),
    summary = .fastkpc_full_cuda_phase3_named_hash(
      jsonlite::read_json(
        artifact_paths$summary_json, simplifyVector = FALSE
      )
    )
  )
}
resume_semantic_hashes <- semantic_snapshot()
resume_context_create <- 0L
resume_context_destroy <- 0L
pure_resume <- fastkpc_full_cuda_phase3_run_shards(
  output_dir = output_dir, kind = "full_shadow",
  setup_keys = setup_keys, target_rows = target_rows,
  identity = identity, route_config = route_config,
  executor = function(...) stop("pure resume executed a shard", call. = FALSE),
  runtime_create = function() {
    resume_context_create <<- resume_context_create + 1L
    stop("pure resume created a runtime", call. = FALSE)
  },
  runtime_destroy = function(context) {
    resume_context_destroy <<- resume_context_destroy + 1L
    invisible(NULL)
  },
  scope = "iteration", shard_count = 64L
)
reused_publication <- do.call(
  fastkpc_full_cuda_phase3_publish_shadow_artifact, publish_args
)
assert_true(
  identical(pure_resume$reused_shard_ids, 0:63) &&
    length(pure_resume$written_shard_ids) == 0L &&
    is.null(pure_resume$session_id) &&
    pure_resume$runtime_context_create_count == 0L &&
    pure_resume$runtime_context_destroy_count == 0L &&
    resume_context_create == 0L && resume_context_destroy == 0L &&
    identical(reused_publication$status, "reused") && identical(
      vapply(resume_byte_paths, fastkpc_full_cuda_census_file_hash,
             character(1L)), resume_byte_hashes
    ) && identical(semantic_snapshot(), resume_semantic_hashes),
  paste(
    "pure resume reuses direct, all 64 shards, sessions, and publication",
    "with zero CUDA contexts/writes and identical byte/semantic hashes"
  )
)

incomplete_dir <- tempfile("phase3-shadow-incomplete-session-")
dir.create(incomplete_dir, recursive = TRUE, showWarnings = FALSE)
on.exit(unlink(incomplete_dir, recursive = TRUE, force = TRUE), add = TRUE)
copied <- file.copy(
  list.files(output_dir, all.files = TRUE, no.. = TRUE, full.names = TRUE),
  incomplete_dir, recursive = TRUE, copy.mode = TRUE, copy.date = TRUE
)
assert_true(all(copied), "incomplete-session fixture clones publication")
incomplete_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  incomplete_dir, "full_shadow"
)
incomplete_session_path <- list.files(
  incomplete_paths$sessions_dir, pattern = "^session_.*\\.json$",
  full.names = TRUE
)[[1L]]
incomplete_session <- jsonlite::read_json(
  incomplete_session_path, simplifyVector = FALSE
)
incomplete_session$status <- "running"
.fastkpc_full_cuda_phase3_write_json_exact(
  incomplete_session, incomplete_session_path
)
incomplete_direct_hashes <- vapply(
  c(incomplete_paths$direct_ci_rds,
    incomplete_paths$direct_ci_summary_json),
  fastkpc_full_cuda_census_file_hash, character(1L)
)
incomplete_context_create <- 0L
incomplete_context_destroy <- 0L
incomplete_resume <- fastkpc_full_cuda_phase3_run_shards(
  output_dir = incomplete_dir, kind = "full_shadow",
  setup_keys = setup_keys, target_rows = target_rows,
  identity = identity, route_config = route_config,
  executor = function(context, shard_id, setup_keys, target_rows) {
    rows <- conditional_fixture[
      conditional_fixture$shard_id == shard_id, , drop = FALSE
    ]
    resources <- resource_fixture[
      resource_fixture$shard_id == shard_id, , drop = FALSE
    ]
    stages <- stage_fixture[
      stage_fixture$shard_id == shard_id, , drop = FALSE
    ]
    rownames(rows) <- rownames(resources) <- rownames(stages) <- NULL
    count <- as.integer(length(setup_keys))
    list(
      payload = list(
        logical_ci_parity = rows,
        resource_metrics = resources,
        stage_timing = stages
      ),
      resource_counts = list(
        prepared_handle_create_count = count,
        prepared_handle_destroy_count = count,
        residual_token_acquire_count = count,
        residual_token_release_count = count,
        output_slot_acquire_count = count,
        output_slot_release_count = count
      )
    )
  },
  runtime_create = function() {
    incomplete_context_create <<- incomplete_context_create + 1L
    new.env(parent = emptyenv())
  },
  runtime_destroy = function(context) {
    incomplete_context_destroy <<- incomplete_context_destroy + 1L
    invisible(NULL)
  },
  scope = "iteration", shard_count = 64L
)
assert_true(
  identical(incomplete_resume$written_shard_ids, 0:63) &&
    length(incomplete_resume$reused_shard_ids) == 0L &&
    incomplete_resume$runtime_context_create_count == 1L &&
    incomplete_resume$runtime_context_destroy_count == 1L &&
    incomplete_context_create == 1L && incomplete_context_destroy == 1L &&
    identical(
      vapply(
        c(incomplete_paths$direct_ci_rds,
          incomplete_paths$direct_ci_summary_json),
        fastkpc_full_cuda_census_file_hash, character(1L)
      ), incomplete_direct_hashes
    ),
  paste(
    "all 64 shards from an incomplete session are recomputed once while",
    "the authenticated direct pair is reused byte-for-byte"
  )
)

validation_args <- c(
  publish_args[setdiff(names(publish_args), "command_lines")],
  list(require_full = FALSE)
)
snapshot_bytes <- function(paths) {
  setNames(lapply(paths, function(path) {
    readBin(path, what = "raw", n = file.info(path)$size)
  }), paths)
}
restore_bytes <- function(snapshot) {
  for (path in names(snapshot)) {
    connection <- file(path, open = "wb")
    on.exit(close(connection), add = TRUE)
    writeBin(snapshot[[path]], connection)
    close(connection)
    on.exit(NULL, add = FALSE)
  }
}
expect_rejected <- function(paths, mutate, label) {
  snapshot <- snapshot_bytes(paths)
  on.exit(restore_bytes(snapshot), add = TRUE)
  mutate()
  error <- tryCatch({
    do.call(
      fastkpc_validate_full_cuda_fixed_sp_shadow_artifact,
      validation_args
    )
    NULL
  }, error = function(error) error)
  assert_true(inherits(error, "error"), paste(label, "must fail closed"))
  restore_bytes(snapshot)
  on.exit(NULL, add = FALSE)
  invisible(error)
}

expect_rejected(
  c(
    artifact_paths$logical_ci_parity_rds,
    artifact_paths$manifest_json, artifact_paths$summary_json
  ),
  function() {
    rows <- readRDS(artifact_paths$logical_ci_parity_rds)
    rows$candidate_p_value[[1L]] <- if (
      rows$reference_decision[[1L]] == "dependent"
    ) rows$alpha[[1L]] * 2 else rows$alpha[[1L]] / 2
    rows$candidate_decision[[1L]] <- ifelse(
      rows$candidate_p_value[[1L]] > rows$alpha[[1L]],
      "independent", "dependent"
    )
    rows$decision_flip[[1L]] <- TRUE
    rows$absolute_p_value_difference[[1L]] <- abs(
      rows$candidate_p_value[[1L]] - rows$reference_p_value[[1L]]
    )
    saveRDS(rows, artifact_paths$logical_ci_parity_rds,
            version = 2, compress = FALSE)
    manifest <- jsonlite::read_json(
      artifact_paths$manifest_json, simplifyVector = FALSE
    )
    manifest$merged_logical_rows_sha256 <-
      fastkpc_full_cuda_census_frame_hash(rows)
    manifest$file_sha256$logical_ci_parity_rds <-
      fastkpc_full_cuda_census_file_hash(
        artifact_paths$logical_ci_parity_rds
      )
    .fastkpc_full_cuda_phase3_write_json_exact(
      manifest, artifact_paths$manifest_json
    )
    summary <- jsonlite::read_json(
      artifact_paths$summary_json, simplifyVector = FALSE
    )
    summary$pass <- TRUE
    summary$decision_flip_count <- 0L
    summary$manifest_sha256 <- fastkpc_full_cuda_census_file_hash(
      artifact_paths$manifest_json
    )
    .fastkpc_full_cuda_phase3_write_json_exact(
      summary, artifact_paths$summary_json
    )
  },
  "candidate p-value with forged hashes and pass claims"
)

sequence_mutations <- list(
  missing_logical_sequence_id = function(rows) {
    rows$logical_sequence_id <- NULL
    rows
  },
  duplicate_logical_sequence_id = function(rows) {
    rows$logical_sequence_id[[2L]] <- rows$logical_sequence_id[[1L]]
    rows
  },
  reordered_logical_sequence_id = function(rows) rows[2:1, , drop = FALSE],
  noncontiguous_logical_sequence_id = function(rows) {
    rows$logical_sequence_id[[2L]] <- rows$logical_sequence_id[[2L]] + 1L
    rows
  },
  missing_scope_sequence_ordinal = function(rows) {
    rows$scope_sequence_ordinal <- NULL
    rows
  },
  duplicate_scope_sequence_ordinal = function(rows) {
    rows$scope_sequence_ordinal[[2L]] <- rows$scope_sequence_ordinal[[1L]]
    rows
  },
  reordered_scope_sequence_ordinal = function(rows) {
    rows$scope_sequence_ordinal[1:2] <- rows$scope_sequence_ordinal[2:1]
    rows
  },
  noncontiguous_scope_sequence_ordinal = function(rows) {
    rows$scope_sequence_ordinal[[2L]] <- 3L
    rows
  },
  wrong_sparse_global_logical_sequence_id = function(rows) {
    rows$logical_sequence_id[[2L]] <- 2215L
    rows
  }
)
for (case_name in names(sequence_mutations)) {
  mutate <- sequence_mutations[[case_name]]
  expect_rejected(
    artifact_paths$logical_ci_parity_rds,
    function() {
      rows <- mutate(readRDS(artifact_paths$logical_ci_parity_rds))
      rownames(rows) <- NULL
      saveRDS(rows, artifact_paths$logical_ci_parity_rds,
              version = 2, compress = FALSE)
    }, case_name
  )
}

nonempty_shard_rds <- shadow_shard_paths[grepl(
  "\\.rds$", shadow_shard_paths
)]
nonempty_shard_rds <- nonempty_shard_rds[vapply(
  nonempty_shard_rds, function(path) {
    nrow(readRDS(path)$payload$resource_metrics) > 0L
  }, logical(1L)
)]
assert_true(
  length(nonempty_shard_rds) == 44L &&
    length(shadow_shard_paths[grepl("\\.rds$", shadow_shard_paths)]) == 64L,
  "fixture contains 64 complete shard pairs and 44 executed setup shards"
)
mutated_shard_rds <- nonempty_shard_rds[[1L]]
mutated_shard_summary <- sub(
  "\\.rds$", ".summary.json", mutated_shard_rds
)
forge_shadow_shard_payload <- function(mutate_payload) {
  envelope <- readRDS(mutated_shard_rds)
  envelope$payload <- mutate_payload(envelope$payload)
  hashes <- .fastkpc_full_cuda_phase3_payload_semantic_hashes(
    envelope$payload
  )
  envelope$payload_semantic_hashes <- as.list(hashes)
  envelope$payload_semantic_hash <-
    .fastkpc_full_cuda_phase3_payload_semantic_hash(hashes)
  saveRDS(envelope, mutated_shard_rds, version = 2, compress = FALSE)
  summary <- jsonlite::read_json(
    mutated_shard_summary, simplifyVector = FALSE
  )
  summary$payload_semantic_hashes <- as.list(hashes)
  summary$payload_semantic_hash <- envelope$payload_semantic_hash
  summary$rds_file_sha256 <- fastkpc_full_cuda_census_file_hash(
    mutated_shard_rds
  )
  .fastkpc_full_cuda_phase3_write_json_exact(
    summary, mutated_shard_summary
  )
}
source_payload_mutations <- list(
  missing_resource_column = function(payload) {
    payload$resource_metrics$cpu_fallback_count <- NULL
    payload
  },
  missing_zero_counter = function(payload) {
    payload$resource_metrics$unknown_fallback_count <- NULL
    payload
  },
  junk_stage = function(payload) {
    payload$stage_timing$stage[[1L]] <- "junk_stage"
    payload
  },
  fabricated_fallback = function(payload) {
    payload$resource_metrics$cpu_fallback_count[[1L]] <- 1L
    payload
  }
)
for (case_name in names(source_payload_mutations)) {
  mutate_payload <- source_payload_mutations[[case_name]]
  pair_snapshot <- snapshot_bytes(c(
    mutated_shard_rds, mutated_shard_summary
  ))
  forge_shadow_shard_payload(mutate_payload)
  error <- tryCatch({
    forged_merged <- fastkpc_full_cuda_phase3_merge_shards(
      output_dir = output_dir, kind = "full_shadow",
      setup_keys = setup_keys, target_rows = target_rows,
      identity = identity, route_config = route_config,
      scope = "iteration", shard_count = 64L
    )
    fastkpc_full_cuda_phase3_validate_shadow_payload(
      forged_merged$payload,
      expected_setup_keys = setup_keys,
      expected_target_rows = target_rows,
      expected_logical_tests = logical_rows,
      require_logical_authority = TRUE,
      expected_setup_rows = setup_authority
    )
    NULL
  }, error = function(error) error)
  restore_bytes(pair_snapshot)
  assert_true(
    inherits(error, "error"),
    paste(case_name, "authenticated merged payload must fail closed")
  )
}

expect_rejected(
  c(
    artifact_paths$resource_metrics_csv,
    artifact_paths$manifest_json, artifact_paths$summary_json
  ),
  function() {
    resources <- read.csv(
      artifact_paths$resource_metrics_csv,
      stringsAsFactors = FALSE, check.names = FALSE
    )
    resources <- resources[rev(seq_len(nrow(resources))), , drop = FALSE]
    write.csv(resources, artifact_paths$resource_metrics_csv,
              row.names = FALSE, na = "NA", quote = TRUE)
    manifest <- jsonlite::read_json(
      artifact_paths$manifest_json, simplifyVector = FALSE
    )
    manifest$file_sha256$resource_metrics_csv <-
      fastkpc_full_cuda_census_file_hash(
        artifact_paths$resource_metrics_csv
      )
    .fastkpc_full_cuda_phase3_write_json_exact(
      manifest, artifact_paths$manifest_json
    )
    summary <- jsonlite::read_json(
      artifact_paths$summary_json, simplifyVector = FALSE
    )
    summary$manifest_sha256 <- fastkpc_full_cuda_census_file_hash(
      artifact_paths$manifest_json
    )
    .fastkpc_full_cuda_phase3_write_json_exact(
      summary, artifact_paths$summary_json
    )
  }, "reordered published resource evidence"
)

expect_rejected(artifact_paths$adjacency_rds, function() {
  adjacency <- readRDS(artifact_paths$adjacency_rds)
  adjacency[1L, 2L] <- !adjacency[1L, 2L]
  adjacency[2L, 1L] <- adjacency[1L, 2L]
  saveRDS(adjacency, artifact_paths$adjacency_rds,
          version = 2, compress = FALSE)
}, "forged adjacency")

for (case in list(
  list(path = artifact_paths$deletion_trace_csv, label = "forged deletion trace"),
  list(path = artifact_paths$sepset_agreement_csv,
       label = "forged sepset agreement"),
  list(path = artifact_paths$n_edgetests_csv,
       label = "forged n.edgetests")
)) {
  expect_rejected(case$path, function() {
    writeLines("forged", case$path, useBytes = TRUE)
  }, case$label)
}

expect_rejected(artifact_paths$first_divergence_json, function() {
  first <- jsonlite::read_json(
    artifact_paths$first_divergence_json, simplifyVector = FALSE
  )
  first$first_divergence_found <- TRUE
  first$type <- "adjacency"
  first$message <- "forged"
  .fastkpc_full_cuda_phase3_write_json_exact(
    first, artifact_paths$first_divergence_json
  )
}, "forged first divergence")

expect_rejected(artifact_paths$summary_json, function() {
  summary <- jsonlite::read_json(
    artifact_paths$summary_json, simplifyVector = FALSE
  )
  summary$pass <- TRUE
  summary$logical_test_count <- 240489L
  summary$candidate_graph_gate <- TRUE
  .fastkpc_full_cuda_phase3_write_json_exact(
    summary, artifact_paths$summary_json
  )
}, "forged pass and summary counters")

expect_rejected(
  c(oracle_paths$manifest_json, oracle_paths$summary_json),
  function() {
    manifest <- jsonlite::read_json(
      oracle_paths$manifest_json, simplifyVector = FALSE
    )
    manifest$fixture_nonce <- "wrong-oracle-sp"
    .fastkpc_full_cuda_phase3_write_json_exact(
      manifest, oracle_paths$manifest_json
    )
    summary <- jsonlite::read_json(
      oracle_paths$summary_json, simplifyVector = FALSE
    )
    summary$manifest_sha256 <- fastkpc_full_cuda_census_file_hash(
      oracle_paths$manifest_json
    )
    .fastkpc_full_cuda_phase3_write_json_exact(
      summary, oracle_paths$summary_json
    )
  }, "wrong oracle-sp manifest hash"
)

expect_oracle_rejected <- function(paths, mutate, label) {
  snapshot <- snapshot_bytes(paths)
  on.exit(restore_bytes(snapshot), add = TRUE)
  mutate()
  error <- tryCatch({
    fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
      oracle_sp_dir, expected_identity = identity, require_full = FALSE
    )
    NULL
  }, error = function(error) error)
  assert_true(inherits(error, "error"), paste(label, "must fail closed"))
  restore_bytes(snapshot)
  on.exit(NULL, add = FALSE)
  invisible(error)
}

expect_oracle_rejected(oracle_paths$summary_json, function() {
  summary <- jsonlite::read_json(
    oracle_paths$summary_json, simplifyVector = FALSE
  )
  summary$pass <- TRUE
  summary$target_count <- 0L
  summary$manifest_sha256 <- fastkpc_full_cuda_census_file_hash(
    oracle_paths$manifest_json
  )
  .fastkpc_full_cuda_phase3_write_json_exact(
    summary, oracle_paths$summary_json
  )
}, "forged oracle summary pass/hash/counters")

expect_oracle_rejected(
  c(
    oracle_paths$target_parity_rds, oracle_paths$manifest_json,
    oracle_paths$summary_json
  ),
  function() {
    targets <- readRDS(oracle_paths$target_parity_rds)
    targets$planned_route[[1L]] <- if (
      targets$planned_route[[1L]] == "AUGMENTED_SVD"
    ) "AUGMENTED_QR" else "AUGMENTED_SVD"
    saveRDS(targets, oracle_paths$target_parity_rds,
            version = 2, compress = FALSE)
    manifest <- jsonlite::read_json(
      oracle_paths$manifest_json, simplifyVector = FALSE
    )
    manifest$payload_file_sha256[["target_parity.rds"]] <-
      fastkpc_full_cuda_census_file_hash(oracle_paths$target_parity_rds)
    .fastkpc_full_cuda_phase3_write_json_exact(
      manifest, oracle_paths$manifest_json
    )
    oracle_summary <- jsonlite::read_json(
      oracle_paths$summary_json, simplifyVector = FALSE
    )
    oracle_summary$manifest_sha256 <- fastkpc_full_cuda_census_file_hash(
      oracle_paths$manifest_json
    )
    .fastkpc_full_cuda_phase3_write_json_exact(
      oracle_summary, oracle_paths$summary_json
    )
  }, "forged oracle target route payload with refreshed hashes"
)

for (linkage_case in c("native", "route")) {
  force(linkage_case)
  expect_oracle_rejected(
    c(
      oracle_paths$manifest_json, oracle_paths$summary_json
    ),
    function() {
      manifest <- jsonlite::read_json(
        oracle_paths$manifest_json, simplifyVector = FALSE
      )
      if (identical(linkage_case, "native")) {
        manifest$executed_native_library_sha256 <- strrep("0", 64L)
      } else {
        manifest$input_identity$route_config_hash <- strrep("0", 64L)
      }
      .fastkpc_full_cuda_phase3_write_json_exact(
        manifest, oracle_paths$manifest_json
      )
      summary <- jsonlite::read_json(
        oracle_paths$summary_json, simplifyVector = FALSE
      )
      summary$manifest_sha256 <- fastkpc_full_cuda_census_file_hash(
        oracle_paths$manifest_json
      )
      .fastkpc_full_cuda_phase3_write_json_exact(
        summary, oracle_paths$summary_json
      )
    }, paste("forged oracle", linkage_case, "linkage")
  )
}

expect_rejected(
  c(artifact_paths$manifest_json, artifact_paths$summary_json),
  function() {
    manifest <- jsonlite::read_json(
      artifact_paths$manifest_json, simplifyVector = FALSE
    )
    manifest$source_inputs$catalog_authority$phase1_dir <-
      paste0(manifest$source_inputs$catalog_authority$phase1_dir, "-moved")
    .fastkpc_full_cuda_phase3_write_json_exact(
      manifest, artifact_paths$manifest_json
    )
    summary <- jsonlite::read_json(
      artifact_paths$summary_json, simplifyVector = FALSE
    )
    summary$manifest_sha256 <- fastkpc_full_cuda_census_file_hash(
      artifact_paths$manifest_json
    )
    .fastkpc_full_cuda_phase3_write_json_exact(
      summary, artifact_paths$summary_json
    )
  }, "moved authenticated catalog input path"
)

expect_rejected(artifact_paths$failures_csv, function() {
  unlink(artifact_paths$failures_csv, force = TRUE)
}, "missing required output file")

cat("full CUDA CI fixed-sp shadow artifact: PASS\n")
