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
logical_row <- phase1_rows[
  phase1_rows$logical_sequence_id == 2214L, , drop = FALSE
]
endpoint_keys <- sort(c(
  logical_row$residual_key_x, logical_row$residual_key_y
), method = "radix")
target_metadata <- catalog$inputs$target_fit_metadata
setup_metadata <- catalog$inputs$same_s_setup_metadata
endpoint_metadata <- target_metadata[match(
  endpoint_keys, target_metadata$residual_key_sha256
), , drop = FALSE]
setup_match <- match(
  endpoint_metadata$same_S_group_id,
  catalog$setup_index$same_S_group_id
)
null_match <- match(
  endpoint_metadata$same_S_group_id, setup_metadata$same_S_group_id
)
endpoint_source <- data.frame(
  prepared_s_key_sha256 =
    catalog$setup_index$prepared_s_key_sha256[setup_match],
  residual_key_sha256 = endpoint_metadata$residual_key_sha256,
  target = as.integer(endpoint_metadata$target),
  null_dim = as.integer(
    setup_metadata$constraint_nullspace_dimension[null_match]
  ),
  condition = as.double(
    endpoint_metadata$penalized_system_condition_at_selected_sp
  ),
  coefficient_rank = as.integer(endpoint_metadata$coefficient_rank),
  planned_route = character(nrow(endpoint_metadata)),
  selected_sp_hash = endpoint_metadata$selected_sp_hash,
  coefficient_hash = endpoint_metadata$coefficient_hash,
  fitted_hash = endpoint_metadata$fitted_hash,
  residual_hash = endpoint_metadata$residual_hash,
  target_fit_fingerprint = endpoint_metadata$target_fit_fingerprint,
  stringsAsFactors = FALSE
)
endpoint_source$planned_route <- vapply(
  seq_len(nrow(endpoint_source)), function(index) {
    fastkpc_full_cuda_fixed_sp_route(
      endpoint_source$condition[[index]],
      endpoint_source$coefficient_rank[[index]],
      endpoint_source$null_dim[[index]], TRUE
    )
  }, character(1L)
)
target_rows <- .fastkpc_full_cuda_phase3_oracle_descriptor_target_rows(
  catalog, endpoint_source
)
target_rows$shard_id <- rep.int(0L, nrow(target_rows))
target_rows <- target_rows[
  , .fastkpc_full_cuda_phase3_oracle_descriptor_target_fields(),
  drop = FALSE
]
setup_keys <- unique(target_rows$prepared_s_key_sha256)
assert_true(length(setup_keys) == 1L, "fixture endpoints share one setup")
logical_row$prepared_s_key_x <- setup_keys
logical_row$prepared_s_key_y <- setup_keys
logical_row$shard_id <- 0L

route_status <- function(route) switch(
  route,
  CHOLESKY_BATCHED = "OK_CHOLESKY_BATCHED",
  AUGMENTED_QR = "OK_AUGMENTED_QR",
  AUGMENTED_SVD = "OK_AUGMENTED_SVD"
)
conditional_fixture <- as.data.frame(lapply(
  fastkpc_full_cuda_shadow_conditional_row_schema(),
  function(type) switch(type, integer = 0L, double = 0,
                        logical = FALSE, character = "")
), stringsAsFactors = FALSE)
for (field in intersect(names(logical_row), names(conditional_fixture))) {
  conditional_fixture[[field]] <- logical_row[[field]]
}
conditional_fixture$prepared_s_key_sha256 <- setup_keys
conditional_fixture$shard_id <- 0L
conditional_fixture$candidate_p_value <- logical_row$reference_p_value
conditional_fixture$absolute_p_value_difference <- 0
conditional_fixture$candidate_decision <- logical_row$reference_decision
conditional_fixture$decision_flip <- FALSE
distance <- abs(log(pmax(
  logical_row$reference_p_value, .Machine$double.xmin
) / logical_row$alpha))
conditional_fixture$near_alpha <- distance <= log(2)
conditional_fixture$near_alpha_bucket <-
  fastkpc_full_cuda_census_near_alpha_bucket(distance)
conditional_fixture$backend <- "cpp"
conditional_fixture$backend_version <-
  fastkpc_full_cuda_shadow_dcov_backend_version()
conditional_fixture$low_rank_backend <- "spectra"
for (endpoint in c("x", "y")) {
  key <- logical_row[[paste0("residual_key_", endpoint)]]
  route_index <- match(key, target_rows$residual_key_sha256)
  route <- target_rows$planned_route[[route_index]]
  conditional_fixture[[paste0("planned_route_", endpoint)]] <- route
  conditional_fixture[[paste0("executed_route_", endpoint)]] <- route
  conditional_fixture[[paste0("reroute_reason_", endpoint)]] <- ""
  conditional_fixture[[paste0("solver_status_", endpoint)]] <-
    route_status(route)
}
invisible(fastkpc_full_cuda_shadow_validate_conditional_rows(
  conditional_fixture
))

route_count <- function(route) as.integer(sum(
  target_rows$planned_route == route
))
resource_fixture <- data.frame(
  planned_cholesky_target_count = route_count("CHOLESKY_BATCHED"),
  planned_qr_target_count = route_count("AUGMENTED_QR"),
  planned_svd_target_count = route_count("AUGMENTED_SVD"),
  executed_cholesky_target_count = route_count("CHOLESKY_BATCHED"),
  executed_qr_target_count = route_count("AUGMENTED_QR"),
  executed_svd_target_count = route_count("AUGMENTED_SVD"),
  cholesky_to_svd_count = 0L, qr_to_svd_count = 0L,
  stable_reroute_count = 0L, cpu_fallback_count = 0L,
  unknown_fallback_count = 0L, approximate_backend_count = 0L,
  implicit_residual_d2h_count = 0L,
  cuda_device_synchronize_count = 0L, nonfinite_output_count = 0L
)
stage_fixture <- data.frame(
  stage = "fixture", elapsed_ms = 0, stringsAsFactors = FALSE
)
payload_fixture <- list(
  logical_ci_parity = conditional_fixture,
  resource_metrics = resource_fixture,
  stage_timing = stage_fixture
)

sha <- fastkpc_full_cuda_census_hash_utf8
route_config <- fastkpc_full_cuda_phase3_route_config()
identity <- list(
  schema_version = "full-cuda-ci-phase3-test-input-identity-v1",
  canonical_setup_corpus_hash =
    fastkpc_full_cuda_census_key_set_hash(setup_keys),
  canonical_target_corpus_hash =
    fastkpc_full_cuda_census_key_set_hash(endpoint_keys),
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

full_resume_dir <- tempfile("phase3-shadow-full-resume-")
on.exit(unlink(full_resume_dir, recursive = TRUE, force = TRUE), add = TRUE)
full_resume_setups <- sort(vapply(
  sprintf("shadow-full-resume-setup-%02d", 0:63), sha, character(1L)
), method = "radix")
full_resume_targets <- data.frame(
  prepared_s_key_sha256 = full_resume_setups,
  residual_key_sha256 = vapply(
    full_resume_setups, function(key) sha(paste0("target-", key)),
    character(1L)
  ), stringsAsFactors = FALSE
)
full_resume_identity <- identity
full_resume_identity$canonical_setup_corpus_hash <-
  fastkpc_full_cuda_census_key_set_hash(full_resume_setups)
full_resume_identity$canonical_target_corpus_hash <-
  fastkpc_full_cuda_census_key_set_hash(
    full_resume_targets$residual_key_sha256
  )
full_resume_identity$sha256 <- NULL
full_resume_identity$sha256 <- .fastkpc_full_cuda_phase3_named_hash(
  full_resume_identity
)
full_resume_executor <- function(context, shard_id, setup_keys, target_rows) {
  count <- as.integer(length(setup_keys))
  list(
    payload = list(
      logical_ci_parity = data.frame(
        logical_sequence_id = as.integer(shard_id + 1L),
        stringsAsFactors = FALSE
      ),
      resource_metrics = data.frame(
        shard_id = as.integer(shard_id), stringsAsFactors = FALSE
      ),
      stage_timing = data.frame(
        shard_id = as.integer(shard_id), stringsAsFactors = FALSE
      )
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
}
run_full_resume <- function(stop_after = NULL, forbid_runtime = FALSE) {
  creates <- 0L
  result <- fastkpc_full_cuda_phase3_run_shards(
    output_dir = full_resume_dir, kind = "full_shadow",
    setup_keys = full_resume_setups, target_rows = full_resume_targets,
    identity = full_resume_identity, route_config = route_config,
    executor = full_resume_executor,
    runtime_create = function() {
      creates <<- creates + 1L
      if (isTRUE(forbid_runtime)) {
        stop("64-shard pure resume created a runtime", call. = FALSE)
      }
      new.env(parent = emptyenv())
    },
    runtime_destroy = function(context) invisible(NULL),
    scope = "qualification", shard_count = 64L, stop_after = stop_after
  )
  list(result = result, creates = creates)
}
full_stopped <- run_full_resume(stop_after = 16L)
full_resumed <- run_full_resume()
full_pure_resume <- run_full_resume(forbid_runtime = TRUE)
assert_true(
  identical(full_stopped$result$status, "stopped") &&
    identical(full_stopped$result$written_shard_ids, 0:15) &&
    identical(full_resumed$result$reused_shard_ids, 0:15) &&
    identical(full_resumed$result$written_shard_ids, 16:63) &&
    identical(full_pure_resume$result$reused_shard_ids, 0:63) &&
    length(full_pure_resume$result$written_shard_ids) == 0L &&
    full_pure_resume$creates == 0L,
  "64-shard run stops gracefully, resumes, and purely reuses every shard"
)

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
  executor = function(...) list(
    payload = payload_fixture,
    resource_counts = list(
      prepared_handle_create_count = 1L,
      prepared_handle_destroy_count = 1L,
      residual_token_acquire_count = 1L,
      residual_token_release_count = 1L,
      output_slot_acquire_count = 1L,
      output_slot_release_count = 1L
    )
  ),
  runtime_create = function() {
    context_count <<- context_count + 1L
    new.env(parent = emptyenv())
  },
  runtime_destroy = function(context) invisible(NULL),
  scope = "iteration", shard_count = 1L
)
assert_true(identical(run$status, "complete") && context_count == 1L,
            "fixture shard source is complete")

oracle_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  oracle_sp_dir, "oracle_sp"
)
dir.create(oracle_sp_dir, recursive = TRUE, showWarnings = FALSE)
oracle_target_routes <- data.frame(
  residual_key_sha256 = target_rows$residual_key_sha256,
  planned_route = target_rows$planned_route,
  executed_route = target_rows$planned_route,
  reroute_reason = rep.int("", nrow(target_rows)),
  solver_status = vapply(
    target_rows$planned_route, route_status, character(1L)
  ), stringsAsFactors = FALSE
)
saveRDS(oracle_target_routes, oracle_paths$target_parity_rds,
        version = 2, compress = FALSE)
oracle_manifest <- list(
  artifact_schema_version =
    fastkpc_full_cuda_phase3_oracle_schema_version(),
  artifact_kind = "oracle_sp", status = "complete", scope = "iteration"
)
.fastkpc_full_cuda_phase3_write_json_exact(
  oracle_manifest, oracle_paths$manifest_json
)
oracle_summary <- list(
  artifact_schema_version =
    fastkpc_full_cuda_phase3_oracle_schema_version(),
  artifact_kind = "oracle_sp", pass = TRUE,
  manifest_sha256 =
    fastkpc_full_cuda_census_file_hash(oracle_paths$manifest_json)
)
.fastkpc_full_cuda_phase3_write_json_exact(
  oracle_summary, oracle_paths$summary_json
)

publish_args <- list(
  output_dir = output_dir, catalog = catalog, setup_keys = setup_keys,
  target_rows = target_rows, identity = identity,
  route_config = route_config, scope = "iteration",
  phase0_dir = phase0_dir, oracle_sp_dir = oracle_sp_dir,
  shard_count = 1L, direct_logical_sequence_id = 1L,
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
full_error <- tryCatch({
  do.call(
    fastkpc_validate_full_cuda_fixed_sp_shadow_artifact,
    c(publish_args[setdiff(names(publish_args), "command_lines")],
      list(require_full = TRUE))
  )
  NULL
}, error = function(error) error)
assert_true(
  inherits(full_error, "error") && grepl(
    "non-full shadow artifact", conditionMessage(full_error), fixed = TRUE
  ),
  "non-full artifact cannot satisfy require_full=TRUE"
)

pure_resume <- fastkpc_full_cuda_phase3_run_shards(
  output_dir = output_dir, kind = "full_shadow",
  setup_keys = setup_keys, target_rows = target_rows,
  identity = identity, route_config = route_config,
  executor = function(...) stop("pure resume executed a shard", call. = FALSE),
  runtime_create = function() {
    context_count <<- context_count + 1L
    stop("pure resume created a runtime", call. = FALSE)
  },
  runtime_destroy = function(context) invisible(NULL),
  scope = "iteration", shard_count = 1L
)
reused_publication <- do.call(
  fastkpc_full_cuda_phase3_publish_shadow_artifact, publish_args
)
assert_true(
  identical(pure_resume$reused_shard_ids, 0L) && context_count == 1L &&
    identical(reused_publication$status, "reused"),
  "pure resume reuses shard/direct/publication evidence without a runtime"
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

expect_rejected(artifact_paths$failures_csv, function() {
  unlink(artifact_paths$failures_csv, force = TRUE)
}, "missing required output file")

cat("full CUDA CI fixed-sp shadow artifact: PASS\n")
