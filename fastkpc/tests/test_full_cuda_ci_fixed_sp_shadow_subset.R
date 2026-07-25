source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_shadow.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_phase3_artifacts.R")

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
assert_error <- function(expression, message, pattern = NULL) {
  error <- tryCatch({
    force(expression)
    NULL
  }, error = function(error) error)
  assert_true(
    inherits(error, "error") &&
      (is.null(pattern) || grepl(pattern, conditionMessage(error),
                                fixed = TRUE)),
    message
  )
  invisible(error)
}

executor_name <- "fastkpc_full_cuda_phase3_run_shadow_shard"
executor <- get0(executor_name, mode = "function", inherits = TRUE)
if (is.null(executor)) fail(paste(executor_name, "is missing"))

required_helpers <- c(
  "fastkpc_full_cuda_shadow_compute_setup_rows",
  "fastkpc_full_cuda_shadow_attach_target_routes",
  "fastkpc_full_cuda_shadow_validate_conditional_rows",
  "fastkpc_full_cuda_shadow_runtime_counters",
  "fastkpc_full_cuda_shadow_scope",
  "fastkpc_full_cuda_phase3_validate_shadow_payload"
)
missing_helpers <- required_helpers[!vapply(
  required_helpers, exists, logical(1L), mode = "function", inherits = TRUE
)]
assert_true(
  length(missing_helpers) == 0L,
  paste("conditional shadow helpers are missing:",
        paste(missing_helpers, collapse = ","))
)

counter_fixture <- data.frame(
  implicit_residual_d2h_count = c(0L, 0L),
  shadow_materialize_call_count = c(1L, 1L),
  shadow_materialize_target_count = c(2L, 3L)
)
counter_summary <- fastkpc_full_cuda_shadow_runtime_counters(counter_fixture)
assert_identical(
  counter_summary,
  list(
    implicit_residual_d2h_count = 0L,
    shadow_materialize_call_count = 2L,
    shadow_materialize_target_count = 5L
  ),
  "shadow runtime counters are exact sums of token-local resource rows"
)

execution_roots <- .fastkpc_full_cuda_phase3_execution_roots()
shadow_runner_id <- "fastkpc/tools/run_full_cuda_ci_fixed_sp_shadow.R"
assert_identical(
  unname(execution_roots[["shadow_runner"]]), shadow_runner_id,
  "executable shadow runner is an authenticated source root"
)
execution_closure <-
  fastkpc_full_cuda_fixed_sp_discover_execution_source_closure(
    execution_roots, project_root = "."
  )
assert_true(
  shadow_runner_id %in% execution_closure$source_ids &&
    shadow_runner_id %in% unname(execution_closure$direct_source_ids),
  "executable shadow runner enters authenticated source identity"
)

sha <- function(label) fastkpc_full_cuda_census_hash_utf8(label)
setup_a <- sha("shadow setup a")
setup_b <- sha("shadow setup b")
key_x <- sha("shadow target x")
key_y <- sha("shadow target y")

descriptor_fixture <- list(
  shard_count = 2L,
  assignments = data.frame(
    prepared_s_key_sha256 = c(setup_a, setup_b),
    shard_id = c(0L, 1L),
    stringsAsFactors = FALSE
  ),
  target_rows = data.frame(
    prepared_s_key_sha256 = c(setup_a, setup_b, setup_a, setup_b),
    residual_key_sha256 = vapply(
      paste("descriptor target", seq_len(4L)), sha, character(1L)
    ),
    shard_id = c(0L, 1L, 0L, 1L),
    stringsAsFactors = FALSE
  )
)
descriptor_rows <- .fastkpc_full_cuda_phase3_shard_descriptor(
  descriptor_fixture, 0L
)$target_rows
assert_identical(
  rownames(descriptor_rows), as.character(seq_len(nrow(descriptor_rows))),
  "shadow shard descriptors reset row names after subsetting"
)

interleaved_ids <- as.integer(c(
  seq.int(2L, 64L, 2L), seq.int(1L, 63L, 2L)
))
interleaved_setup_keys <- unname(vapply(
  paste("interleaved setup", seq_len(64L)), sha, character(1L)
))
interleaved_payloads <- lapply(seq_len(64L), function(index) {
  list(logical_ci_parity = data.frame(
    prepared_s_key_sha256 = interleaved_setup_keys[[index]],
    logical_sequence_id = interleaved_ids[[index]],
    stringsAsFactors = FALSE
  ))
})
interleaved_plan <- list(
  shard_count = 64L,
  assignments = data.frame(
    prepared_s_key_sha256 = interleaved_setup_keys,
    shard_id = 0:63,
    stringsAsFactors = FALSE
  ),
  target_rows = data.frame(
    residual_key_sha256 = unname(vapply(
      paste("interleaved target", seq_len(64L)), sha, character(1L)
    )),
    stringsAsFactors = FALSE
  )
)
interleaved_rows <- .fastkpc_full_cuda_phase3_merge_payloads(
  interleaved_payloads, interleaved_plan
)$logical_ci_parity
assert_identical(
  interleaved_rows$logical_sequence_id, seq_len(64L),
  "64-shard merge restores canonical logical order across setup shards"
)

logical_fixture <- data.frame(
  logical_sequence_id = 10L,
  source_sequence_id = 20L,
  source_task_index = 1L,
  level = 1L,
  x = 1L,
  y = 2L,
  S_key = "3",
  residual_key_x = key_x,
  residual_key_y = key_y,
  prepared_s_key_x = setup_a,
  prepared_s_key_y = setup_a,
  shard_id = 7L,
  reference_p_value = 0.05,
  alpha = 0.05,
  reference_decision = "dependent",
  absolute_log_distance_from_alpha = 0,
  near_alpha = TRUE,
  stringsAsFactors = FALSE
)
target_keys_fixture <- sort(c(key_x, key_y), method = "radix")
residuals_fixture <- matrix(
  as.double(seq_len(16L)), nrow = 8L, ncol = 2L,
  dimnames = list(NULL, target_keys_fixture)
)
route_fixture <- data.frame(
  prepared_s_key_sha256 = rep(setup_a, 2L),
  residual_key_sha256 = target_keys_fixture,
  planned_route = c("CHOLESKY_BATCHED", "AUGMENTED_QR"),
  executed_route = c("CHOLESKY_BATCHED", "AUGMENTED_QR"),
  reroute_reason = c("", ""),
  solver_status = c("OK_CHOLESKY_BATCHED", "OK_AUGMENTED_QR"),
  stringsAsFactors = FALSE
)

dcov_result <- function(p_value = 0.05, pair_count = 1L) {
  list(
    p.value = rep(as.double(p_value), pair_count),
    diagnostics = list(
      n = 8L,
      batch_count = as.integer(pair_count),
      numCol = 35L,
      index = 1,
      lowrank_mode = "spectra",
      lowrank_full_eig_count = 0L,
      lowrank_spectra_count = as.integer(2L * pair_count),
      lowrank_spectra_converged_count = as.integer(2L * pair_count),
      lowrank_spectra_failed_count = 0L,
      lowrank_spectra_fallback_full_eig_count = 0L
    )
  )
}
identity_backend_scope <- function(callback) callback()
equal_boundary_base <- fastkpc_full_cuda_shadow_compute_setup_rows(
  logical_tests = logical_fixture,
  setup_key = setup_a,
  shard_id = 7L,
  target_keys = target_keys_fixture,
  residuals = residuals_fixture,
  .dcov_batch_fun = function(...) dcov_result(),
  .backend_scope = identity_backend_scope
)
assert_true(
  fastkpc_full_cuda_fixed_sp_callback_result_is_compact(
    equal_boundary_base
  ),
  "setup-local dCov callback rows satisfy the residual leak guard"
)
equal_boundary_rows <- fastkpc_full_cuda_shadow_attach_target_routes(
  equal_boundary_base, route_fixture, logical_fixture
)
assert_true(
  identical(equal_boundary_rows$candidate_p_value, 0.05) &&
    identical(equal_boundary_rows$alpha, 0.05) &&
    identical(equal_boundary_rows$candidate_decision, "dependent") &&
    !equal_boundary_rows$decision_flip,
  "conditional equality boundary uses strict candidate_p_value > alpha"
)
assert_true(
  isTRUE(fastkpc_full_cuda_shadow_validate_conditional_rows(
    equal_boundary_rows,
    expected_logical_tests = logical_fixture,
    expected_setup_key = setup_a,
    expected_shard_id = 7L,
    expected_target_rows = route_fixture
  )),
  "strict equality-boundary row validates"
)

zero_reference_log_distance <- abs(log(
  .Machine$double.xmin / logical_fixture$alpha
))
zero_reference_logical <- logical_fixture
zero_reference_logical$reference_p_value <- 0
zero_reference_logical$absolute_log_distance_from_alpha <-
  zero_reference_log_distance
zero_reference_logical$near_alpha <- FALSE
zero_reference_rows <- equal_boundary_rows
zero_reference_rows$reference_p_value <- 0
zero_reference_rows$candidate_p_value <- 0
zero_reference_rows$absolute_p_value_difference <- 0
zero_reference_rows$near_alpha <- FALSE
zero_reference_rows$near_alpha_bucket <-
  fastkpc_full_cuda_census_near_alpha_bucket(
    zero_reference_log_distance
  )
assert_true(
  isTRUE(fastkpc_full_cuda_shadow_validate_conditional_rows(
    zero_reference_rows,
    expected_logical_tests = zero_reference_logical,
    expected_setup_key = setup_a,
    expected_shard_id = 7L,
    expected_target_rows = route_fixture
  )),
  "zero reference p-value preserves the authenticated p-floor bucket"
)

backend_called <- FALSE
missing_endpoint <- logical_fixture
missing_endpoint$residual_key_y <- sha("missing endpoint")
assert_error(
  fastkpc_full_cuda_shadow_compute_setup_rows(
    missing_endpoint, setup_a, 7L, target_keys_fixture,
    residuals_fixture,
    .dcov_batch_fun = function(...) {
      backend_called <<- TRUE
      dcov_result()
    },
    .backend_scope = identity_backend_scope
  ),
  "missing endpoint key is a setup failure",
  "endpoint residual key"
)
assert_true(!backend_called, "missing endpoint fails before dCov")

assert_error(
  fastkpc_full_cuda_shadow_compute_setup_rows(
    logical_fixture, setup_a, 7L,
    c(key_x, key_x), residuals_fixture,
    .dcov_batch_fun = function(...) dcov_result(),
    .backend_scope = identity_backend_scope
  ),
  "duplicate residual columns/keys fail closed", "duplicate"
)

cross_setup <- logical_fixture
cross_setup$prepared_s_key_y <- setup_b
assert_error(
  fastkpc_full_cuda_shadow_compute_setup_rows(
    cross_setup, setup_a, 7L, target_keys_fixture,
    residuals_fixture,
    .dcov_batch_fun = function(...) dcov_result(),
    .backend_scope = identity_backend_scope
  ),
  "cross-setup logical ownership fails closed", "setup ownership"
)

noncanonical <- rbind(logical_fixture, logical_fixture)
noncanonical$logical_sequence_id <- c(11L, 10L)
noncanonical$source_sequence_id <- c(21L, 20L)
noncanonical$source_task_index <- c(2L, 1L)
assert_error(
  fastkpc_full_cuda_shadow_compute_setup_rows(
    noncanonical, setup_a, 7L, target_keys_fixture,
    residuals_fixture,
    .dcov_batch_fun = function(...) dcov_result(pair_count = 2L),
    .backend_scope = identity_backend_scope
  ),
  "noncanonical logical order fails closed", "canonical"
)

assert_error(
  fastkpc_full_cuda_shadow_compute_setup_rows(
    logical_fixture, setup_a, 7L, target_keys_fixture,
    residuals_fixture,
    .dcov_batch_fun = function(...) list(p.value = numeric()),
    .backend_scope = identity_backend_scope
  ),
  "malformed dCov output fails closed", "dCov output"
)
bad_diagnostics <- dcov_result()
bad_diagnostics$diagnostics$lowrank_spectra_failed_count <- 1L
assert_error(
  fastkpc_full_cuda_shadow_compute_setup_rows(
    logical_fixture, setup_a, 7L, target_keys_fixture,
    residuals_fixture,
    .dcov_batch_fun = function(...) bad_diagnostics,
    .backend_scope = identity_backend_scope
  ),
  "malformed/fallback dCov diagnostics fail closed", "diagnostics"
)
assert_error(
  fastkpc_full_cuda_shadow_compute_setup_rows(
    logical_fixture, setup_a, 7L, target_keys_fixture,
    residuals_fixture,
    .dcov_batch_fun = function(...) stop("injected dCov failure"),
    .backend_scope = identity_backend_scope
  ),
  "dCov backend errors are shard failures", "backend error"
)

duplicate_routes <- rbind(route_fixture, route_fixture[1L, , drop = FALSE])
assert_error(
  fastkpc_full_cuda_shadow_attach_target_routes(
    equal_boundary_base, duplicate_routes, logical_fixture
  ),
  "duplicate target route keys fail closed", "duplicate"
)
cross_route <- route_fixture
cross_route$prepared_s_key_sha256[[2L]] <- setup_b
assert_error(
  fastkpc_full_cuda_shadow_attach_target_routes(
    equal_boundary_base, cross_route, logical_fixture
  ),
  "cross-setup target route ownership fails closed", "ownership"
)

leaked_executor_result <- list(
  payload = list(logical_ci_parity = equal_boundary_rows),
  resource_counts = list(
    prepared_handle_create_count = 1L,
    prepared_handle_destroy_count = 0L,
    residual_token_acquire_count = 1L,
    residual_token_release_count = 1L,
    output_slot_acquire_count = 1L,
    output_slot_release_count = 1L
  )
)
assert_error(
  .fastkpc_full_cuda_phase3_validate_executor_result(
    leaked_executor_result
  ),
  "executor resource leaks fail closed", "leaked"
)

authority_logical <- rbind(logical_fixture, logical_fixture)
authority_logical$logical_sequence_id <- c(10L, 11L)
authority_logical$source_sequence_id <- c(20L, 21L)
authority_logical$source_task_index <- c(1L, 2L)
authority_logical$shard_id <- rep.int(0L, 2L)
rownames(authority_logical) <- NULL
authority_base <- fastkpc_full_cuda_shadow_compute_setup_rows(
  logical_tests = authority_logical,
  setup_key = setup_a,
  shard_id = 0L,
  target_keys = target_keys_fixture,
  residuals = residuals_fixture,
  .dcov_batch_fun = function(...) dcov_result(pair_count = 2L),
  .backend_scope = identity_backend_scope
)
authority_rows <- fastkpc_full_cuda_shadow_attach_target_routes(
  authority_base, route_fixture, authority_logical
)

oracle_frame_fixture <- function(name, row_count = 1L) {
  schema <- fastkpc_full_cuda_fixed_sp_oracle_row_schemas()[[name]]
  columns <- lapply(unname(schema), function(type) switch(
    type,
    character = rep.int("", row_count),
    integer = rep.int(0L, row_count),
    double = rep.int(0, row_count),
    logical = rep.int(FALSE, row_count),
    fail(paste("unsupported oracle fixture type", type))
  ))
  names(columns) <- names(schema)
  as.data.frame(columns, stringsAsFactors = FALSE, optional = TRUE)
}
authority_resources <- oracle_frame_fixture("resource_metrics")
authority_resources$prepared_s_key_sha256 <- setup_a
authority_resources$shard_id <- 0L
authority_resources$setup_ordinal <- 1L
authority_resources$target_count <- as.integer(length(target_keys_fixture))
for (field in c(
  "prepared_handle_create_count", "prepared_handle_destroy_count",
  "residual_token_acquire_count", "residual_token_release_count",
  "output_slot_acquire_count", "output_slot_release_count",
  "shadow_materialize_call_count", "invalid_output_init_count"
)) {
  authority_resources[[field]] <- 1L
}
authority_resources$shadow_materialize_target_count <-
  as.integer(length(target_keys_fixture))
authority_stage_timing <- data.frame(
  prepared_s_key_sha256 = rep.int(setup_a, 6L),
  shard_id = rep.int(0L, 6L),
  setup_ordinal = rep.int(1L, 6L),
  stage = c(
    "phase2_shard_load", "prepared_handle_create", "solve",
    "shadow_materialize", "cmagic_oracle", "release_and_free"
  ),
  elapsed_ms = rep.int(0, 6L),
  stringsAsFactors = FALSE
)
authority_payload <- list(
  logical_ci_parity = authority_rows,
  resource_metrics = authority_resources,
  stage_timing = authority_stage_timing
)

authority_target_rows <- data.frame(
  prepared_s_key_sha256 = rep.int(setup_a, 2L),
  residual_key_sha256 = target_keys_fixture,
  shard_id = rep.int(0L, 2L),
  canonical_setup_rank = rep.int(1L, 2L),
  canonical_target_rank = seq_len(2L),
  phase2_shard_id = rep.int(0L, 2L),
  target = as.integer(match(target_keys_fixture, c(key_x, key_y))),
  null_dim = rep.int(2L, 2L),
  condition = rep.int(1, 2L),
  coefficient_rank = rep.int(2L, 2L),
  planned_route = route_fixture$planned_route,
  selected_sp_hash = unname(vapply(
    paste("authority selected sp", seq_len(2L)), sha, character(1L)
  )),
  coefficient_hash = unname(vapply(
    paste("authority coefficient", seq_len(2L)), sha, character(1L)
  )),
  fitted_hash = unname(vapply(
    paste("authority fitted", seq_len(2L)), sha, character(1L)
  )),
  residual_hash = unname(vapply(
    paste("authority residual", seq_len(2L)), sha, character(1L)
  )),
  target_fit_fingerprint = unname(vapply(
    paste("authority target fit", seq_len(2L)), sha, character(1L)
  )),
  stringsAsFactors = FALSE
)
authority_plan <- fastkpc_full_cuda_phase3_plan_shards(
  setup_keys = setup_a,
  target_rows = authority_target_rows,
  scope = "iteration",
  shard_count = 1L,
  conditional_tests = authority_logical[2:1, , drop = FALSE]
)
authority_descriptor <- .fastkpc_full_cuda_phase3_shard_descriptor(
  authority_plan, 0L
)
assert_true(
  identical(authority_descriptor$logical_rows, authority_logical) &&
    identical(
      authority_descriptor$logical_hash,
      .fastkpc_full_cuda_phase3_shadow_logical_authority_hash(
        authority_logical
      )
    ),
  "shadow descriptor canonicalizes and version-hashes plan conditional rows"
)

authority_route <- fastkpc_full_cuda_phase3_route_config()
authority_gpu <- list(
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
)
authority_identity_info <- list(
  input_identity_hash = sha("authority input identity"),
  source_commit = strrep("1", 40L),
  gpu_environment = authority_gpu,
  gpu_environment_hash =
    .fastkpc_full_cuda_phase3_named_hash(authority_gpu),
  production_identity = TRUE
)
authority_session <- list(
  schema_version = "full-cuda-ci-phase3-session-v1",
  session_id = "authoritative_shadow_fixture",
  input_identity_hash = authority_identity_info$input_identity_hash,
  route_config_hash = authority_route$sha256,
  executed_native_library_sha256 = sha("authority native library"),
  requested_shard_ids = 0L,
  completed_shard_ids = integer(),
  runtime_context_create_count = 1L,
  runtime_context_destroy_count = 0L,
  prepared_handle_create_count = 0L,
  prepared_handle_destroy_count = 0L,
  residual_token_acquire_count = 0L,
  residual_token_release_count = 0L,
  output_slot_acquire_count = 0L,
  output_slot_release_count = 0L,
  target_level_stable_sync_count = 0L,
  status = "running"
)
authority_contract <- .fastkpc_full_cuda_phase3_kind_contract("full_shadow")
authority_envelope <- .fastkpc_full_cuda_phase3_build_shard_envelope(
  contract = authority_contract,
  descriptor = authority_descriptor,
  session = authority_session,
  identity_info = authority_identity_info,
  route_config = authority_route,
  payload = authority_payload
)
unauthenticated_descriptor <- authority_descriptor
unauthenticated_descriptor$shadow_payload_semantics <-
  "nonproduction-generic-payload-v1"
unauthenticated_descriptor$logical_rows <- NULL
unauthenticated_descriptor$logical_ids <- integer()
unauthenticated_descriptor$logical_count <- 0L
unauthenticated_descriptor$logical_hash <-
  .fastkpc_full_cuda_phase3_shadow_generic_logical_hash()
assert_error(
  .fastkpc_full_cuda_phase3_build_shard_envelope(
    contract = authority_contract,
    descriptor = unauthenticated_descriptor,
    session = authority_session,
    identity_info = authority_identity_info,
    route_config = authority_route,
    payload = authority_payload
  ),
  "production full-shadow descriptors require logical authority",
  "authority is unavailable"
)

self_consistent_pair <- function(envelope, mutate_payload) {
  forged <- envelope
  forged$payload <- mutate_payload(forged$payload)
  hashes <- .fastkpc_full_cuda_phase3_payload_semantic_hashes(
    forged$payload
  )
  forged$payload_semantic_hashes <- as.list(hashes)
  forged$payload_semantic_hash <-
    .fastkpc_full_cuda_phase3_payload_semantic_hash(hashes)
  list(
    envelope = forged,
    summary = .fastkpc_full_cuda_phase3_build_shard_summary(
      forged, sha("self-consistent hostile RDS bytes")
    )
  )
}
validate_authority_pair <- function(pair) {
  .fastkpc_full_cuda_phase3_validate_shard_pair(
    envelope = pair$envelope,
    summary = pair$summary,
    contract = authority_contract,
    descriptor = authority_descriptor,
    identity_info = authority_identity_info,
    route_config = authority_route
  )
}
baseline_pair <- list(
  envelope = authority_envelope,
  summary = .fastkpc_full_cuda_phase3_build_shard_summary(
    authority_envelope, sha("baseline authoritative RDS bytes")
  )
)
assert_true(
  isTRUE(validate_authority_pair(baseline_pair)),
  "authoritative full-shadow pair fixture validates before hostile mutation"
)

dropped_pair <- self_consistent_pair(authority_envelope, function(payload) {
  payload$logical_ci_parity <-
    payload$logical_ci_parity[-1L, , drop = FALSE]
  rownames(payload$logical_ci_parity) <- NULL
  payload
})
assert_error(
  validate_authority_pair(dropped_pair),
  "self-consistent rehash cannot authenticate a dropped logical row",
  "logical"
)

balanced_resource_pair <- self_consistent_pair(
  authority_envelope,
  function(payload) {
    fields <- c(
      "prepared_handle_create_count", "prepared_handle_destroy_count",
      "residual_token_acquire_count", "residual_token_release_count",
      "output_slot_acquire_count", "output_slot_release_count"
    )
    for (field in fields) payload$resource_metrics[[field]] <- 2L
    payload
  }
)
assert_error(
  validate_authority_pair(balanced_resource_pair),
  "self-consistent rehash cannot authenticate balanced forged resources",
  "resource"
)

forged_resource_ordinal_pair <- self_consistent_pair(
  authority_envelope,
  function(payload) {
    payload$resource_metrics$setup_ordinal <- 2L
    payload
  }
)
assert_error(
  validate_authority_pair(forged_resource_ordinal_pair),
  "self-consistent rehash cannot forge resource setup authority",
  "resource"
)

forged_stage_ordinal_pair <- self_consistent_pair(
  authority_envelope,
  function(payload) {
    payload$stage_timing$setup_ordinal <- rep.int(2L, 6L)
    payload
  }
)
assert_error(
  validate_authority_pair(forged_stage_ordinal_pair),
  "self-consistent rehash cannot forge stage setup authority",
  "stage"
)

missing_residual_column_pair <- self_consistent_pair(
  authority_envelope,
  function(payload) {
    payload$logical_ci_parity$residual_key_y <- NULL
    payload
  }
)
assert_error(
  validate_authority_pair(missing_residual_column_pair),
  "self-consistent rehash cannot hide a missing residual column",
  "schema"
)

missing_residual_key_pair <- self_consistent_pair(
  authority_envelope,
  function(payload) {
    payload$logical_ci_parity$residual_key_y[[1L]] <-
      sha("missing authoritative residual key")
    payload
  }
)
assert_error(
  validate_authority_pair(missing_residual_key_pair),
  "self-consistent rehash cannot hide a missing residual key",
  "logical"
)

mislabelled_residual_pair <- self_consistent_pair(
  authority_envelope,
  function(payload) {
    rows <- payload$logical_ci_parity
    for (field in c(
      "residual_key", "planned_route", "executed_route", "reroute_reason",
      "solver_status"
    )) {
      x_field <- paste0(field, "_x")
      y_field <- paste0(field, "_y")
      temporary <- rows[[x_field]]
      rows[[x_field]] <- rows[[y_field]]
      rows[[y_field]] <- temporary
    }
    temporary <- rows$x
    rows$x <- rows$y
    rows$y <- temporary
    payload$logical_ci_parity <- rows
    payload
  }
)
assert_error(
  validate_authority_pair(mislabelled_residual_pair),
  "self-consistent rehash cannot relabel residual endpoint columns",
  "lineage"
)

relabelled_order_pair <- self_consistent_pair(
  authority_envelope,
  function(payload) {
    rows <- payload$logical_ci_parity[2:1, , drop = FALSE]
    rows$logical_sequence_id <- authority_descriptor$logical_ids
    rownames(rows) <- NULL
    payload$logical_ci_parity <- rows
    payload
  }
)
assert_error(
  validate_authority_pair(relabelled_order_pair),
  "self-consistent rehash cannot reorder and relabel logical rows",
  "lineage"
)

unlabelled_backend_called <- FALSE
unlabelled_residuals <- residuals_fixture
dimnames(unlabelled_residuals) <- NULL
assert_error(
  fastkpc_full_cuda_shadow_compute_setup_rows(
    logical_fixture, setup_a, 7L, target_keys_fixture,
    unlabelled_residuals,
    .dcov_batch_fun = function(...) {
      unlabelled_backend_called <<- TRUE
      dcov_result()
    },
    .backend_scope = identity_backend_scope
  ),
  "missing residual column keys fail closed", "residual matrix/columns"
)
assert_true(
  !unlabelled_backend_called,
  "missing residual column keys fail before dCov"
)
mislabelled_residuals <- residuals_fixture
colnames(mislabelled_residuals) <- rev(target_keys_fixture)
assert_error(
  fastkpc_full_cuda_shadow_compute_setup_rows(
    logical_fixture, setup_a, 7L, target_keys_fixture,
    mislabelled_residuals,
    .dcov_batch_fun = function(...) dcov_result(),
    .backend_scope = identity_backend_scope
  ),
  "reordered or mislabelled residual columns fail closed",
  "residual matrix/columns"
)

expected_dcov_backend_version <- "legacy-dcov-gamma-cpp-v1"
assert_true(
  identical(
    fastkpc_full_cuda_shadow_dcov_backend_version(),
    expected_dcov_backend_version
  ) && all(c("backend", "backend_version", "low_rank_backend") %in%
            names(equal_boundary_rows)) &&
    identical(equal_boundary_rows$backend, "cpp") &&
    identical(
      equal_boundary_rows$backend_version, expected_dcov_backend_version
    ) && identical(equal_boundary_rows$low_rank_backend, "spectra"),
  "conditional rows freeze one pinned C++ Spectra dCov backend identity"
)
wrong_backend_version <- equal_boundary_rows
wrong_backend_version$backend_version <- "legacy-dcov-gamma-cpp-v0"
assert_error(
  fastkpc_full_cuda_shadow_validate_conditional_rows(
    wrong_backend_version,
    expected_logical_tests = logical_fixture,
    expected_setup_key = setup_a,
    expected_shard_id = 7L,
    expected_target_rows = route_fixture
  ),
  "conditional validator rejects an unpinned dCov backend version",
  "canonical row contract"
)
direct_backend_fields <- names(
  fastkpc_full_cuda_phase3_direct_ci_row_schema()
)
conditional_backend_fields <- names(
  fastkpc_full_cuda_shadow_conditional_row_schema()
)
assert_true(
  all(c("backend", "backend_version", "low_rank_backend") %in%
        direct_backend_fields) &&
    all(c("backend", "backend_version", "low_rank_backend") %in%
          conditional_backend_fields),
  "direct and conditional logical rows share the pinned dCov version field"
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
qualification <- fastkpc_full_cuda_shadow_scope(
  catalog = catalog, plan = plan, scope = "qualification"
)
assert_true(
  length(qualification$setup_keys) == 2061L &&
    nrow(qualification$target_rows) == 6143L &&
    nrow(qualification$logical_tests) == 3808L,
  "authenticated qualification shadow scope"
)
qualification_shards <- fastkpc_full_cuda_phase3_plan_shards(
  setup_keys = qualification$setup_keys,
  target_rows = qualification$target_rows,
  scope = "qualification",
  conditional_tests = qualification$logical_tests,
  canonical_setup_shards = TRUE
)
qualification_descriptor <- .fastkpc_full_cuda_phase3_shard_descriptor(
  qualification_shards, 0L
)
descriptor_fields <-
  .fastkpc_full_cuda_phase3_oracle_descriptor_target_fields()
qualification_projected <-
  .fastkpc_full_cuda_phase3_oracle_descriptor_target_rows(
    catalog,
    qualification_descriptor$target_rows[
      , setdiff(descriptor_fields, "shard_id"), drop = FALSE
    ]
  )
qualification_projected$shard_id <- rep.int(
  qualification_descriptor$shard_id, nrow(qualification_projected)
)
qualification_projected <- qualification_projected[
  , descriptor_fields, drop = FALSE
]
assert_true(
  all(vapply(descriptor_fields, function(field) {
    identical(
      qualification_descriptor$target_rows[[field]],
      qualification_projected[[field]]
    )
  }, logical(1L))),
  "qualification shadow descriptor fields match catalog authority"
)
assert_identical(
  rownames(qualification_descriptor$target_rows),
  as.character(seq_len(nrow(qualification_descriptor$target_rows))),
  "qualification shadow descriptor row names are canonical"
)
assert_true(
  identical(qualification_descriptor$target_rows, qualification_projected),
  "qualification shadow descriptor frame matches catalog authority"
)
qualification_expected_logical <- qualification$logical_tests[
  qualification$logical_tests$shard_id == qualification_descriptor$shard_id,
  , drop = FALSE
]
rownames(qualification_expected_logical) <- NULL
assert_true(
  identical(
    qualification_descriptor$shadow_payload_semantics,
    "authoritative-logical-corpus-v1"
  ) && identical(
    qualification_descriptor$logical_rows, qualification_expected_logical
  ) && identical(
    qualification_descriptor$logical_ids,
    qualification_expected_logical$logical_sequence_id
  ) && identical(
    qualification_descriptor$logical_count,
    as.integer(nrow(qualification_expected_logical))
  ) && identical(
    qualification_descriptor$logical_hash,
    .fastkpc_full_cuda_phase3_shadow_logical_authority_hash(
      qualification_expected_logical
    )
  ),
  "qualification shadow descriptor freezes exact per-shard logical authority"
)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP Phase 3 fixed-sp conditional shadow subset CUDA execution\n")
  quit(save = "no", status = 0L)
}

qualified_native <-
  fastkpc_full_cuda_phase3_discover_qualified_native_evidence()
assert_true(
  shadow_runner_id %in%
    names(qualified_native$provenance$source_file_paths) &&
    shadow_runner_id %in%
      unname(qualified_native$provenance$direct_source_ids),
  "qualified native provenance authenticates the shadow runner"
)
build_fastkpc_cuda_native(rebuild = FALSE)
assert_true(fastkpc_cuda_available(), "CUDA must be available")

iteration <- fastkpc_full_cuda_shadow_scope(
  catalog = catalog, plan = plan, scope = "iteration"
)
assert_true(
  length(iteration$setup_keys) == 44L &&
    nrow(iteration$target_rows) == 270L &&
    nrow(iteration$logical_tests) == 44L,
  "authenticated iteration shadow scope"
)

device_id <- suppressWarnings(as.integer(Sys.getenv(
  "FASTKPC_FULL_CUDA_PHASE3_DEVICE", unset = "0"
)))
assert_true(
  length(device_id) == 1L && !is.na(device_id) && device_id >= 0L,
  "test device id"
)
capacity <- fastkpc_full_cuda_fixed_sp_contract()$canonical_capacities
create_runtime <- function() {
  runtime <- fixed_sp_cuda_runtime_create(device_id)
  keep <- FALSE
  on.exit({
    if (!keep) try(fixed_sp_cuda_runtime_free(runtime), silent = TRUE)
  }, add = TRUE)
  fixed_sp_cuda_runtime_reserve(
    runtime, capacity$n, capacity$null_dim, capacity$target_count,
    capacity$penalty_count, capacity$augmented_rows
  )
  keep <- TRUE
  runtime
}

shard_ids <- sort(unique(iteration$setup_assignments$shard_id))
run_iteration_shard <- function(runtime, shard_id) {
  setup_keys <- iteration$setup_assignments$prepared_s_key_sha256[
    iteration$setup_assignments$shard_id == shard_id
  ]
  target_rows <- iteration$target_rows[
    iteration$target_rows$shard_id == shard_id, , drop = FALSE
  ]
  logical_tests <- iteration$logical_tests[
    iteration$logical_tests$shard_id == shard_id, , drop = FALSE
  ]
  rownames(target_rows) <- rownames(logical_tests) <- NULL
  executor(
    context = runtime, shard_id = as.integer(shard_id),
    setup_keys = setup_keys, target_rows = target_rows,
    catalog = catalog, plan = plan, logical_tests = logical_tests
  )
}

# A callback-side backend failure must release the token, slot, and handle.
failure_runtime <- create_runtime()
failure_runtime_open <- TRUE
on.exit({
  if (isTRUE(failure_runtime_open)) {
    try(fixed_sp_cuda_runtime_free(failure_runtime), silent = TRUE)
  }
}, add = TRUE)
original_backend <- fastkpc_legacy_dcov_gamma_cpp_oracle_batch
assign(
  "fastkpc_legacy_dcov_gamma_cpp_oracle_batch",
  function(...) stop("injected setup-local dCov failure", call. = FALSE),
  envir = .GlobalEnv
)
failure_shard <- shard_ids[[1L]]
assert_error(
  run_iteration_shard(failure_runtime, failure_shard),
  "setup-local dCov failure propagates", "backend error"
)
assign(
  "fastkpc_legacy_dcov_gamma_cpp_oracle_batch", original_backend,
  envir = .GlobalEnv
)
recovered <- run_iteration_shard(failure_runtime, failure_shard)
assert_true(
  nrow(recovered$payload$logical_ci_parity) > 0L &&
    identical(recovered$payload$resource_metrics$implicit_residual_d2h_count,
              0L) &&
    identical(recovered$payload$resource_metrics$shadow_materialize_call_count,
              1L) &&
    identical(
      recovered$payload$resource_metrics$shadow_materialize_target_count,
      recovered$payload$resource_metrics$target_count
    ) &&
    all(recovered$payload$resource_metrics[
      c(
        "prepared_handle_destroy_count", "residual_token_release_count",
        "output_slot_release_count"
      )
    ] == 1L) &&
    all(!recovered$payload$resource_metrics$output_slot_leased_after_release),
  "backend failure releases resources before the next shadow solve"
)
fixed_sp_cuda_runtime_free(failure_runtime)
failure_runtime_open <- FALSE

runtime_context <- create_runtime()
runtime_open <- TRUE
on.exit({
  if (isTRUE(runtime_open)) {
    try(fixed_sp_cuda_runtime_free(runtime_context), silent = TRUE)
  }
}, add = TRUE)
outputs <- lapply(shard_ids, function(shard_id) {
  run_iteration_shard(runtime_context, shard_id)
})
fixed_sp_cuda_runtime_free(runtime_context)
runtime_open <- FALSE

rows <- do.call(rbind, lapply(outputs, function(value) {
  value$payload$logical_ci_parity
}))
rows <- rows[order(rows$logical_sequence_id, method = "radix"), , drop = FALSE]
rownames(rows) <- NULL
resources <- do.call(rbind, lapply(outputs, function(value) {
  value$payload$resource_metrics
}))
rownames(resources) <- NULL
runtime <- fastkpc_full_cuda_shadow_runtime_counters(resources)

assert_true(nrow(rows) == 44L, "iteration shadow logical rows")
assert_true(
  all(order(rows$logical_sequence_id) == seq_len(nrow(rows))),
  "iteration shadow canonical order"
)
assert_true(sum(rows$decision_flip) == 0L,
            "iteration shadow decision flips")
assert_true(sum(rows$backend_error) == 0L,
            "iteration shadow backend errors")
assert_true(sum(rows$spectra_fallback) == 0L,
            "iteration shadow Spectra fallbacks")
assert_true(runtime$implicit_residual_d2h_count == 0L,
            "iteration no implicit residual D2H")
assert_true(runtime$shadow_materialize_target_count == 270L,
            "iteration explicit shadow materialization")
assert_true(
  runtime$shadow_materialize_call_count == 44L &&
    nrow(resources) == 44L && sum(resources$target_count) == 270L &&
    all(resources$prepared_handle_create_count == 1L) &&
    all(resources$prepared_handle_destroy_count == 1L) &&
    all(resources$residual_token_acquire_count == 1L) &&
    all(resources$residual_token_release_count == 1L) &&
    all(resources$output_slot_acquire_count == 1L) &&
    all(resources$output_slot_release_count == 1L) &&
    all(!resources$output_slot_leased_after_release),
  "iteration exact setup resource and lease gates"
)
assert_true(
  all(resources$implicit_residual_d2h_count == 0L) &&
    all(resources$shadow_materialize_call_count == 1L) &&
    identical(
      as.integer(resources$shadow_materialize_target_count),
      as.integer(resources$target_count)
    ) &&
    all(resources$cpu_fallback_count == 0L) &&
    all(resources$unknown_fallback_count == 0L) &&
    all(resources$approximate_backend_count == 0L) &&
    all(resources$invalid_output_init_count == 1L) &&
    all(resources$nonfinite_output_count == 0L) &&
    all(resources$per_target_allocation_count_after_warmup == 0L) &&
    all(resources$per_target_handle_create_count == 0L),
  "iteration persistent runtime and no-fallback gates"
)
assert_true(
    identical(rows$logical_sequence_id,
            iteration$logical_tests$logical_sequence_id) &&
    all(rows$prepared_s_key_sha256 %in% iteration$setup_keys) &&
    all(rows$backend == "cpp") &&
    all(rows$backend_version == expected_dcov_backend_version) &&
    all(rows$low_rank_backend == "spectra") &&
    max(rows$absolute_p_value_difference) <
      fastkpc_full_cuda_phase3_route_config()$
        qualification_dcov_p_tolerance,
  "iteration conditional row lineage and dCov parity"
)
assert_true(
  !"deletes_edge" %in% names(rows) &&
    all(c(
      "planned_route_x", "executed_route_x", "reroute_reason_x",
      "solver_status_x", "planned_route_y", "executed_route_y",
      "reroute_reason_y", "solver_status_y"
    ) %in% names(rows)),
  "conditional rows defer replay and freeze both endpoint routes"
)

cat(
  "PASS Phase 3 fixed-sp conditional shadow subset:",
  nrow(rows), "logical rows /",
  runtime$shadow_materialize_target_count, "explicit targets\n"
)
