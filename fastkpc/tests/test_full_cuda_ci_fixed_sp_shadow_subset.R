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

destroy_events <- character()
destroy_error <- assert_error(
  .fastkpc_full_cuda_phase3_destroy_shadow_runtime(
    runtime = structure(list(), class = "synthetic-runtime"),
    info_fun = function(runtime) {
      destroy_events <<- c(destroy_events, "info")
      list(state = "open")
    },
    validate_fun = function(info) {
      destroy_events <<- c(destroy_events, "validate")
      stop("injected runtime info validation failure", call. = FALSE)
    },
    free_fun = function(runtime) {
      destroy_events <<- c(destroy_events, "free")
      stop("injected runtime free failure", call. = FALSE)
    }
  ),
  "runtime teardown preserves validation and free failures",
  "runtime info validation failure"
)
assert_true(
  identical(destroy_events, c("info", "validate", "free")) &&
    grepl("runtime free failure", conditionMessage(destroy_error), fixed = TRUE),
  "runtime teardown always frees and aggregates both failures"
)
destroy_events <- character()
destroy_info <- .fastkpc_full_cuda_phase3_destroy_shadow_runtime(
  runtime = structure(list(), class = "synthetic-runtime"),
  info_fun = function(runtime) {
    destroy_events <<- c(destroy_events, "info")
    list(state = "open")
  },
  validate_fun = function(info) {
    destroy_events <<- c(destroy_events, "validate")
    invisible(TRUE)
  },
  free_fun = function(runtime) {
    destroy_events <<- c(destroy_events, "free")
    invisible(NULL)
  }
)
assert_true(
  identical(destroy_info, list(state = "open")) &&
    identical(destroy_events, c("info", "validate", "free")),
  "runtime teardown reports success only after confirmed free"
)

lifecycle_probe <- function(label, body_error = FALSE) {
  mint_count <- 0L
  release_count <- 0L
  body_count <- 0L
  expression <- function() {
    .fastkpc_full_cuda_phase3_with_shadow_execution_snapshot(
      mint_fun = function() {
        mint_count <<- mint_count + 1L
        structure(list(label = label), class = "snapshot-token")
      },
      release_fun = function(token) {
        release_count <<- release_count + 1L
        invisible(TRUE)
      },
      body_fun = function(token) {
        body_count <<- body_count + 1L
        if (isTRUE(body_error)) stop(label, call. = FALSE)
        label
      }
    )
  }
  if (isTRUE(body_error)) {
    assert_error(expression(), paste(label, "propagates"), label)
    result <- NULL
  } else {
    result <- expression()
  }
  list(
    result = result, mint_count = mint_count,
    release_count = release_count, body_count = body_count
  )
}
for (scenario in list(
  list(label = "success", body_error = FALSE),
  list(label = "pure-resume", body_error = FALSE),
  list(label = "executor-error", body_error = TRUE),
  list(label = "destroy-error", body_error = TRUE)
)) {
  probe <- lifecycle_probe(scenario$label, scenario$body_error)
  assert_true(
    probe$mint_count == 1L && probe$body_count == 1L &&
      probe$release_count == 1L,
    paste("snapshot release is singular for", scenario$label)
  )
}

assert_true(
  is.null(getOption("fastkpc.phase3.shadow_snapshot_registry.v1")) &&
    is.null(getOption(
      "fastkpc.phase3.shadow_snapshot_registry.initialized.v1"
    )),
  "shadow snapshot registry is not exposed through global options"
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
authority_resources$canonical_setup_rank <- 1L
authority_resources$target_count <- as.integer(length(target_keys_fixture))
authority_resources$phase2_shard_load_count <- 1L
authority_resources$phase2_shard_authentication_count <- 1L
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
authority_resources$setup_h2d_upload_count <- 1L
authority_resources$setup_h2d_bytes <- 5648
authority_resources$target_batch_h2d_call_count <- 1L
authority_resources$target_h2d_copy_count <- 2L
authority_resources$target_h2d_bytes <- 5616
authority_resources$rhs_device_build_count <- 1L
authority_resources$rhs_authority <- "cuda-x0-transpose-y"
authority_resources$full_cuda_data_plane <- TRUE
authority_resources$coefficient_batch_finalize_call_count <- 1L
authority_resources$fitted_batch_finalize_call_count <- 1L
authority_resources$residual_rss_batch_finalize_call_count <- 1L
authority_resources$per_target_output_finalize_call_count <- 1L
authority_resources$batch_output_finalized_target_count <- 2L
authority_resources$planned_cholesky_target_count <- 1L
authority_resources$planned_qr_target_count <- 1L
authority_resources$executed_cholesky_target_count <- 1L
authority_resources$executed_qr_target_count <- 1L
for (field in c(
  "cholesky_factor_checkpoint_record_count",
  "cholesky_factor_checkpoint_wait_count",
  "cholesky_solve_checkpoint_record_count",
  "cholesky_solve_checkpoint_wait_count",
  "qr_checkpoint_record_count", "qr_checkpoint_wait_count"
)) {
  authority_resources[[field]] <- 1L
}
authority_resources$shadow_d2h_bytes <- 11312
authority_resources$cusolver_deterministic_mode <- "enabled"
authority_resources$cublas_math_mode <- "pedantic"
authority_resources$cublas_atomics_mode <- "not_allowed"
authority_resources$cublas_user_workspace_installed <- TRUE
authority_resources$cublas_workspace_bytes <- 16777216
authority_resources$cublas_workspace_alignment <- 256
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
attacker_selected_plan <- fastkpc_full_cuda_phase3_plan_shards(
  setup_keys = setup_a,
  target_rows = authority_target_rows,
  scope = "iteration",
  shard_count = 1L,
  conditional_tests = authority_logical[2:1, , drop = FALSE]
)
attacker_selected_descriptor <- .fastkpc_full_cuda_phase3_shard_descriptor(
  attacker_selected_plan, 0L
)
assert_true(
  identical(
    attacker_selected_descriptor$shadow_payload_semantics,
    "nonproduction-generic-payload-v1"
  ) && is.null(attacker_selected_descriptor$logical_rows) &&
    identical(attacker_selected_descriptor$logical_ids, integer()),
  "caller conditional rows cannot promote a shard plan to production authority"
)
authority_schema <-
  .fastkpc_full_cuda_phase3_shadow_logical_authority_schema(
    authority_logical[
      .fastkpc_full_cuda_phase3_shadow_logical_authority_fields()
    ]
  )
authority_plan <- .fastkpc_full_cuda_phase3_bind_shadow_authority(
  attacker_selected_plan, authority_logical, authority_schema
)
authority_descriptor <- .fastkpc_full_cuda_phase3_shard_descriptor(
  authority_plan, 0L
)
assert_true(
  identical(
    authority_descriptor$logical_rows,
    authority_logical[
      .fastkpc_full_cuda_phase3_shadow_logical_authority_fields()
    ]
  ) &&
    identical(
      authority_descriptor$logical_hash,
      .fastkpc_full_cuda_phase3_shadow_logical_authority_hash(
        authority_logical[
          .fastkpc_full_cuda_phase3_shadow_logical_authority_fields()
        ]
      )
    ),
  "shadow descriptor canonicalizes and version-hashes plan conditional rows"
)
type_drift_logical <- authority_logical
type_drift_logical$source_sequence_id <-
  as.double(type_drift_logical$source_sequence_id)
type_drift_logical <- type_drift_logical[
  .fastkpc_full_cuda_phase3_shadow_logical_authority_fields()
]
assert_error(
  .fastkpc_full_cuda_phase3_validate_shadow_logical_authority(
    type_drift_logical, authority_schema
  ),
  "logical authority rejects integer-to-double type drift",
  "schema"
)
assert_true(
  !identical(
    authority_schema$sha256,
    .fastkpc_full_cuda_phase3_shadow_logical_authority_schema(
      type_drift_logical
    )$sha256
  ),
  "logical authority hash binds exact column types"
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
aggregate_identity <- c(
  list(
    schema_version = "full-cuda-ci-phase3-test-input-identity-v1",
    canonical_setup_corpus_hash =
      fastkpc_full_cuda_census_key_set_hash(setup_a),
    canonical_target_corpus_hash =
      fastkpc_full_cuda_census_key_set_hash(sort(
        authority_target_rows$residual_key_sha256, method = "radix"
      )),
    route_config_hash = authority_route$sha256,
    source_commit = strrep("1", 40L)
  ),
  authority_gpu
)
aggregate_identity$sha256 <- .fastkpc_full_cuda_phase3_named_hash(
  aggregate_identity
)
aggregate_dir <- tempfile("phase3-run-destroy-aggregate-")
on.exit(unlink(aggregate_dir, recursive = TRUE, force = TRUE), add = TRUE)
aggregate_destroy_count <- 0L
aggregate_error <- tryCatch({
  fastkpc_full_cuda_phase3_run_shards(
    output_dir = aggregate_dir,
    kind = "full_shadow",
    setup_keys = setup_a,
    target_rows = authority_target_rows,
    identity = aggregate_identity,
    route_config = authority_route,
    executor = function(...) {
      stop("injected executor failure", call. = FALSE)
    },
    runtime_create = function() new.env(parent = emptyenv()),
    runtime_destroy = function(runtime) {
      aggregate_destroy_count <<- aggregate_destroy_count + 1L
      .fastkpc_full_cuda_phase3_destroy_shadow_runtime(
        runtime,
        info_fun = function(runtime) list(state = "open"),
        validate_fun = function(info) {
          stop("injected nested validation failure", call. = FALSE)
        },
        free_fun = function(runtime) {
          stop("injected nested free failure", call. = FALSE)
        }
      )
    },
    scope = "iteration",
    shard_count = 1L
  )
  NULL
}, error = function(error) error)
aggregate_sessions <- list.files(
  file.path(aggregate_dir, "sessions"), full.names = TRUE
)
aggregate_session <- .fastkpc_full_cuda_phase3_read_json(
  aggregate_sessions[[1L]], "aggregate failure session"
)
assert_true(
  inherits(aggregate_error, "fastkpc_phase3_run_destroy_error") &&
    identical(names(aggregate_error), c(
      "message", "call", "run_error", "destroy_error"
    )) && inherits(aggregate_error$run_error, "error") &&
    inherits(
      aggregate_error$destroy_error,
      "fastkpc_shadow_runtime_destroy_error"
    ) && grepl(
      "injected executor failure",
      conditionMessage(aggregate_error$run_error), fixed = TRUE
    ) && grepl(
      "injected nested validation failure",
      conditionMessage(aggregate_error$destroy_error), fixed = TRUE
    ) && grepl(
      "injected nested free failure",
      conditionMessage(aggregate_error$destroy_error), fixed = TRUE
    ) && aggregate_destroy_count == 1L &&
    !identical(aggregate_session$status, "complete") &&
    as.integer(aggregate_session$runtime_context_destroy_count) == 0L,
  "runner aggregates executor and nested destroy failures without completion"
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

na_setup_rank_pair <- self_consistent_pair(
  authority_envelope,
  function(payload) {
    payload$resource_metrics$canonical_setup_rank <- NA_integer_
    payload
  }
)
assert_error(
  validate_authority_pair(na_setup_rank_pair),
  "self-consistent rehash cannot hide a missing canonical setup rank",
  "resource"
)

forged_phase2_auth_pair <- self_consistent_pair(
  authority_envelope,
  function(payload) {
    payload$resource_metrics$phase2_shard_authentication_count <- 999L
    payload
  }
)
assert_error(
  validate_authority_pair(forged_phase2_auth_pair),
  "self-consistent rehash cannot forge Phase 2 authentication counters",
  "resource"
)

forged_mode_pair <- self_consistent_pair(
  authority_envelope,
  function(payload) {
    payload$resource_metrics$cublas_math_mode <- "fast"
    payload
  }
)
assert_error(
  validate_authority_pair(forged_mode_pair),
  "self-consistent rehash cannot forge deterministic data-plane modes",
  "resource"
)

forged_shadow_bytes_pair <- self_consistent_pair(
  authority_envelope,
  function(payload) {
    payload$resource_metrics$shadow_d2h_bytes <- 11320
    payload
  }
)
assert_error(
  validate_authority_pair(forged_shadow_bytes_pair),
  "self-consistent rehash cannot forge shadow transfer bytes",
  "resource"
)

forged_setup_bytes_pair <- self_consistent_pair(
  authority_envelope,
  function(payload) {
    payload$resource_metrics$setup_h2d_bytes <-
      payload$resource_metrics$setup_h2d_bytes + 8
    payload
  }
)
assert_error(
  validate_authority_pair(forged_setup_bytes_pair),
  "self-consistent rehash cannot forge setup upload bytes",
  "resource"
)

overflow_bytes_pair <- self_consistent_pair(
  authority_envelope,
  function(payload) {
    payload$resource_metrics$setup_h2d_bytes <- .Machine$double.xmax
    payload
  }
)
assert_error(
  validate_authority_pair(overflow_bytes_pair),
  "self-consistent rehash cannot hide transfer byte overflow",
  "resource"
)

negative_counter_pair <- self_consistent_pair(
  authority_envelope,
  function(payload) {
    payload$resource_metrics$resource_allocation_count_before_solve <- -1L
    payload$resource_metrics$resource_allocation_count_after_solve <- -1L
    payload
  }
)
assert_error(
  validate_authority_pair(negative_counter_pair),
  "self-consistent rehash cannot hide balanced negative counters",
  "resource"
)

resource_frame_attribute_pair <- self_consistent_pair(
  authority_envelope,
  function(payload) {
    attr(payload$resource_metrics, "hostile") <- TRUE
    payload
  }
)
assert_error(
  validate_authority_pair(resource_frame_attribute_pair),
  "resource frame attributes fail the exact frame contract",
  "resource"
)

resource_row_names_pair <- self_consistent_pair(
  authority_envelope,
  function(payload) {
    rownames(payload$resource_metrics) <- "hostile"
    payload
  }
)
assert_error(
  validate_authority_pair(resource_row_names_pair),
  "resource row names fail the automatic row-name contract",
  "resource"
)

resource_column_attribute_pair <- self_consistent_pair(
  authority_envelope,
  function(payload) {
    attr(payload$resource_metrics$target_count, "units") <- "targets"
    payload
  }
)
assert_error(
  validate_authority_pair(resource_column_attribute_pair),
  "resource column attributes fail the bare-column contract",
  "resource"
)

nonfinite_stage_pair <- self_consistent_pair(
  authority_envelope,
  function(payload) {
    payload$stage_timing$elapsed_ms[[1L]] <- Inf
    payload
  }
)
assert_error(
  validate_authority_pair(nonfinite_stage_pair),
  "nonfinite stage timing fails before authority arithmetic",
  "stage"
)

na_phase2_auth_pair <- self_consistent_pair(
  authority_envelope,
  function(payload) {
    payload$resource_metrics$phase2_shard_authentication_count <- NA_integer_
    payload
  }
)
assert_error(
  validate_authority_pair(na_phase2_auth_pair),
  "missing Phase 2 authentication count fails closed",
  "resource"
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
  all(c("backend", "low_rank_backend") %in% direct_backend_fields) &&
    !"backend_version" %in% direct_backend_fields &&
    all(c("backend", "backend_version", "low_rank_backend") %in%
          conditional_backend_fields),
  "only conditional logical rows carry the pinned dCov version field"
)

scaled_setup_keys <- vapply(
  paste("scaled setup", seq_len(256L)), sha, character(1L)
)
scaled_target_setup_keys <- rep(scaled_setup_keys, each = 16L)
scaled_grouping <- .fastkpc_full_cuda_phase3_index_setup_targets(
  scaled_setup_keys, scaled_target_setup_keys,
  "scaled shadow grouping probe"
)
assert_true(
  identical(scaled_grouping$index, rep(seq_len(256L), each = 16L)) &&
    identical(scaled_grouping$first, as.integer(seq(1L, 4096L, by = 16L))) &&
    identical(scaled_grouping$count, rep.int(16L, 256L)) &&
    identical(scaled_grouping$work_count, 4352L),
  "snapshot grouping work scales with targets plus setups"
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
snapshot_probe <- local({
  original_validate <- .fastkpc_full_cuda_shadow_validate_supplied_plan
  validation_count <- 0L
  assign(
    ".fastkpc_full_cuda_shadow_validate_supplied_plan",
    function(catalog, plan) {
      validation_count <<- validation_count + 1L
      original_validate(catalog, plan)
    },
    envir = .GlobalEnv
  )
  on.exit(assign(
    ".fastkpc_full_cuda_shadow_validate_supplied_plan",
    original_validate, envir = .GlobalEnv
  ), add = TRUE)
  snapshot <- fastkpc_full_cuda_phase3_create_shadow_execution_snapshot(
    catalog, plan, "qualification"
  )
  authority <-
    .fastkpc_full_cuda_phase3_resolve_shadow_execution_snapshot(snapshot)
  populated <- which(vapply(
    authority$shards, function(shard) shard$setup_count > 0L, logical(1L)
  ))
  for (index in head(populated, 2L)) {
    .fastkpc_full_cuda_phase3_validate_shadow_snapshot_shard(
      snapshot, authority$shards[[index]]
    )
  }
  list(
    snapshot = snapshot,
    authority = authority,
    validation_count = validation_count,
    populated = populated
  )
})
assert_true(
  snapshot_probe$validation_count == 1L &&
    identical(snapshot_probe$authority$scope, "qualification") &&
    identical(
      snapshot_probe$authority$setup_keys, qualification$setup_keys
    ) && identical(
      snapshot_probe$authority$target_rows, qualification$target_rows
    ) && length(
      snapshot_probe$authority$phase3_plan$precomputed_descriptors
    ) == 64L,
  "execution snapshot performs heavy plan/file validation exactly once"
)
setup_type_drift <- snapshot_probe$authority$setup_authority
setup_type_drift$n <- as.double(setup_type_drift$n)
assert_error(
  .fastkpc_full_cuda_phase3_validate_shadow_setup_authority(
    setup_type_drift,
    snapshot_probe$authority$setup_authority_schema,
    snapshot_probe$authority$phase3_plan$assignments
  ),
  "snapshot setup authority rejects integer-to-double dimensions",
  "setup authority"
)
assignment_mutation <- snapshot_probe$authority$phase3_plan$assignments
assignment_mutation$sorted_rank[[1L]] <-
  assignment_mutation$sorted_rank[[1L]] + 1L
assert_error(
  .fastkpc_full_cuda_phase3_validate_shadow_assignments(
    assignment_mutation,
    snapshot_probe$authority$assignments_schema,
    snapshot_probe$authority$target_rows,
    64L
  ),
  "snapshot assignment authority rejects sorted-rank mutation",
  "assignment authority"
)
target_type_drift <- snapshot_probe$authority$target_rows
target_type_drift$canonical_target_rank <-
  as.double(target_type_drift$canonical_target_rank)
assert_error(
  .fastkpc_full_cuda_phase3_validate_shadow_target_authority(
    target_type_drift,
    snapshot_probe$authority$target_schema,
    snapshot_probe$authority$phase3_plan$assignments,
    64L
  ),
  "snapshot target authority rejects equal-value integer-to-double drift",
  "target authority"
)
attributed_target <- snapshot_probe$authority$target_rows
attr(attributed_target, "hostile") <- TRUE
assert_error(
  .fastkpc_full_cuda_phase3_validate_shadow_target_authority(
    attributed_target,
    snapshot_probe$authority$target_schema,
    snapshot_probe$authority$phase3_plan$assignments,
    64L
  ),
  "snapshot target authority rejects frame attributes",
  "target authority"
)
renamed_assignments <- snapshot_probe$authority$phase3_plan$assignments
rownames(renamed_assignments) <- paste0("hostile-", seq_len(
  nrow(renamed_assignments)
))
assert_error(
  .fastkpc_full_cuda_phase3_validate_shadow_assignments(
    renamed_assignments,
    snapshot_probe$authority$assignments_schema,
    snapshot_probe$authority$target_rows,
    64L
  ),
  "snapshot assignment authority rejects hostile row names",
  "assignment authority"
)
overflow_setup <- snapshot_probe$authority$setup_authority
overflow_setup$n[[1L]] <- .Machine$integer.max
assert_error(
  .fastkpc_full_cuda_phase3_validate_shadow_setup_authority(
    overflow_setup,
    snapshot_probe$authority$setup_authority_schema,
    snapshot_probe$authority$phase3_plan$assignments
  ),
  "snapshot setup authority rejects count/byte overflow dimensions",
  "setup authority"
)
assert_true(
  !identical(
    .fastkpc_full_cuda_phase3_shadow_setup_authority_schema(
      setup_type_drift
    )$sha256,
    snapshot_probe$authority$setup_authority_schema$sha256
  ) && !identical(
    .fastkpc_full_cuda_phase3_shadow_target_authority_schema(
      target_type_drift
    )$sha256,
    snapshot_probe$authority$target_schema$sha256
  ),
  "snapshot exact frame identities bind column typeof independently"
)
catalog_authority <- fastkpc_full_cuda_phase3_discover_catalog_authority(catalog)
snapshot_identity_binding <- c(
  list(schema_version = "full-cuda-ci-phase3-input-identity-v1"),
  catalog_authority$lineage[setdiff(
    names(catalog_authority$lineage), "authenticated"
  )],
  list(route_config_hash = plan$route_config_sha256)
)
assert_true(
  identical(
    .fastkpc_full_cuda_phase3_resolve_shadow_execution_snapshot(
      snapshot_probe$snapshot,
      expected_scope = "qualification",
      expected_identity = snapshot_identity_binding,
      expected_setup_keys = qualification$setup_keys,
      expected_target_rows = qualification$target_rows,
      expected_plan_identity_sha256 = plan$plan_identity_sha256
    )$snapshot_identity_sha256,
    snapshot_probe$authority$snapshot_identity_sha256
  ),
  "snapshot resolve independently binds production identity and runner corpus"
)
assert_error(
  .fastkpc_full_cuda_phase3_resolve_shadow_execution_snapshot(
    snapshot_probe$snapshot, expected_scope = "iteration"
  ),
  "snapshot rejects wrong-scope replay", "scope"
)
cross_catalog_identity <- snapshot_identity_binding
cross_catalog_identity$phase1_manifest_hash <- sha("cross-catalog replay")
assert_error(
  .fastkpc_full_cuda_phase3_resolve_shadow_execution_snapshot(
    snapshot_probe$snapshot,
    expected_scope = "qualification",
    expected_identity = cross_catalog_identity,
    expected_setup_keys = qualification$setup_keys,
    expected_target_rows = qualification$target_rows,
    expected_plan_identity_sha256 = plan$plan_identity_sha256
  ),
  "snapshot rejects cross-catalog identity replay", "identity"
)
assert_error(
  .fastkpc_full_cuda_phase3_resolve_shadow_execution_snapshot(
    snapshot_probe$snapshot,
    expected_scope = "qualification",
    expected_identity = snapshot_identity_binding,
    expected_setup_keys = qualification$setup_keys,
    expected_target_rows = qualification$target_rows,
    expected_plan_identity_sha256 = sha("cross-plan replay")
  ),
  "snapshot rejects cross-plan replay", "plan"
)
assert_error(
  .fastkpc_full_cuda_phase3_resolve_shadow_execution_snapshot(
    snapshot_probe$snapshot,
    expected_scope = "qualification",
    expected_identity = snapshot_identity_binding,
    expected_setup_keys = rev(qualification$setup_keys),
    expected_target_rows = qualification$target_rows,
    expected_plan_identity_sha256 = plan$plan_identity_sha256
  ),
  "snapshot rejects caller corpus replay", "corpus"
)
local({
  old <- options()[c(
    "fastkpc.phase3.shadow_snapshot_registry.v1",
    "fastkpc.phase3.shadow_snapshot_registry.initialized.v1"
  )]
  on.exit(options(old), add = TRUE)
  options(
    fastkpc.phase3.shadow_snapshot_registry.v1 =
      new.env(parent = emptyenv()),
    fastkpc.phase3.shadow_snapshot_registry.initialized.v1 =
      "attacker-controlled"
  )
  assert_true(
    identical(
      .fastkpc_full_cuda_phase3_resolve_shadow_execution_snapshot(
        snapshot_probe$snapshot, expected_scope = "qualification"
      )$snapshot_identity_sha256,
      snapshot_probe$authority$snapshot_identity_sha256
    ),
    "global option injection cannot replace the private snapshot registry"
  )
})
mutated_authority_copy <-
  .fastkpc_full_cuda_phase3_resolve_shadow_execution_snapshot(
    snapshot_probe$snapshot
  )
mutated_authority_copy$scope <- "full"
assert_true(
  identical(
    .fastkpc_full_cuda_phase3_resolve_shadow_execution_snapshot(
      snapshot_probe$snapshot
    )$scope,
    "qualification"
  ),
  "mutating a resolved authority copy cannot mutate the private entry"
)
wrong_pid_snapshot <- new.env(parent = emptyenv())
for (field in ls(snapshot_probe$snapshot, all.names = TRUE)) {
  assign(field, snapshot_probe$snapshot[[field]], envir = wrong_pid_snapshot)
}
wrong_pid_snapshot$creator_pid <- as.integer(Sys.getpid() + 1L)
lockEnvironment(wrong_pid_snapshot, bindings = TRUE)
assert_error(
  .fastkpc_full_cuda_phase3_resolve_shadow_execution_snapshot(
    wrong_pid_snapshot
  ),
  "snapshot rejects a locked wrong-PID token", "token"
)
tampered_snapshot_shard <-
  snapshot_probe$authority$shards[[snapshot_probe$populated[[1L]]]]
tampered_snapshot_shard$logical_rows$source_sequence_id[[1L]] <-
  tampered_snapshot_shard$logical_rows$source_sequence_id[[1L]] + 1L
assert_error(
  .fastkpc_full_cuda_phase3_validate_shadow_snapshot_shard(
    snapshot_probe$snapshot, tampered_snapshot_shard
  ),
  "tampered execution snapshot shard authority fails closed",
  "snapshot shard"
)
fake_snapshot <- new.env(parent = emptyenv())
fake_snapshot_values <- list(
  schema_version = "full-cuda-ci-phase3-shadow-snapshot-token-v2",
  capability_id = sha("attacker capability id"),
  creator_pid = as.integer(Sys.getpid()),
  snapshot_identity_sha256 = sha("attacker snapshot identity")
)
for (field in names(fake_snapshot_values)) {
  assign(field, fake_snapshot_values[[field]], envir = fake_snapshot)
}
lockEnvironment(fake_snapshot, bindings = TRUE)
local({
  fake_registry <- new.env(parent = emptyenv())
  assign(fake_snapshot$capability_id, list(
    schema_version =
      "full-cuda-ci-phase3-shadow-snapshot-private-entry-v1",
    sequence_id = 1L,
    token = fake_snapshot,
    authority = list(
      snapshot_identity_sha256 = fake_snapshot$snapshot_identity_sha256
    )
  ), envir = fake_registry)
  old <- options()["fastkpc.phase3.shadow_snapshot_registry.v1"]
  on.exit(options(old), add = TRUE)
  options(fastkpc.phase3.shadow_snapshot_registry.v1 = fake_registry)
  assert_error(
    .fastkpc_full_cuda_phase3_resolve_shadow_execution_snapshot(fake_snapshot),
    "self-consistent fake option entry cannot mint a snapshot capability",
    "released"
  )
})
assert_error(
  .fastkpc_full_cuda_phase3_resolve_execution_plan(
    kind = "full_shadow",
    identity = list(
      schema_version = "full-cuda-ci-phase3-input-identity-v1"
    ),
    setup_keys = setup_a,
    target_rows = authority_target_rows,
    scope = "iteration",
    canonical_setup_shards = FALSE,
    conditional_tests = authority_logical[1:2, , drop = FALSE],
    execution_snapshot = fake_snapshot
  ),
  "production identity rejects attacker-selected logical rows and snapshot",
  "snapshot"
)
iteration_snapshot <-
  fastkpc_full_cuda_phase3_create_shadow_execution_snapshot(
    catalog, plan, "iteration"
  )
iteration_authority <-
  .fastkpc_full_cuda_phase3_resolve_shadow_execution_snapshot(
    iteration_snapshot, expected_scope = "iteration"
  )
iteration_empty <- vapply(
  iteration_authority$shards,
  function(shard) shard$setup_count == 0L,
  logical(1L)
)
assert_true(
  length(iteration_authority$shards) == 64L && any(iteration_empty) &&
    all(vapply(iteration_authority$shards, function(shard) {
      isTRUE(.fastkpc_full_cuda_phase3_validate_shadow_snapshot_shard(
        iteration_snapshot, shard
      ))
    }, logical(1L))),
  "iteration snapshot precomputes and authenticates all empty shards"
)
fastkpc_full_cuda_phase3_release_shadow_execution_snapshot(iteration_snapshot)
assert_error(
  .fastkpc_full_cuda_phase3_resolve_shadow_execution_snapshot(
    iteration_snapshot
  ),
  "released iteration snapshot fails closed", "released"
)
fastkpc_full_cuda_phase3_release_shadow_execution_snapshot(
  snapshot_probe$snapshot
)
assert_error(
  .fastkpc_full_cuda_phase3_resolve_shadow_execution_snapshot(
    snapshot_probe$snapshot
  ),
  "released qualification snapshot fails closed", "released"
)
qualification_shards <- fastkpc_full_cuda_phase3_plan_shards(
  setup_keys = qualification$setup_keys,
  target_rows = qualification$target_rows,
  scope = "qualification",
  conditional_tests = qualification$logical_tests,
  canonical_setup_shards = TRUE
)
qualification_schema <-
  .fastkpc_full_cuda_phase3_shadow_logical_authority_schema(
    qualification$logical_tests[
      .fastkpc_full_cuda_phase3_shadow_logical_authority_fields()
    ]
  )
qualification_shards <- .fastkpc_full_cuda_phase3_bind_shadow_authority(
  qualification_shards, qualification$logical_tests, qualification_schema
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
  .fastkpc_full_cuda_phase3_shadow_logical_authority_fields(), drop = FALSE
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

iteration_snapshot <-
  fastkpc_full_cuda_phase3_create_shadow_execution_snapshot(
    catalog = catalog, plan = plan, scope = "iteration"
  )
on.exit(
  fastkpc_full_cuda_phase3_release_shadow_execution_snapshot(
    iteration_snapshot
  ),
  add = TRUE
)
iteration <- .fastkpc_full_cuda_phase3_resolve_shadow_execution_snapshot(
  iteration_snapshot, expected_scope = "iteration"
)
iteration$setup_assignments <- iteration$phase3_plan$assignments
iteration$logical_tests <- iteration$logical_rows
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
iteration_identity <- fastkpc_full_cuda_phase3_input_identity(
  catalog, device_id
)
iteration <- .fastkpc_full_cuda_phase3_resolve_shadow_execution_snapshot(
  iteration_snapshot,
  expected_scope = "iteration",
  expected_identity = iteration_identity,
  expected_setup_keys = iteration$setup_keys,
  expected_target_rows = iteration$target_rows,
  expected_plan_identity_sha256 = plan$plan_identity_sha256
)
iteration$setup_assignments <- iteration$phase3_plan$assignments
iteration$logical_tests <- iteration$logical_rows
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
  descriptor <- iteration$shards[[as.character(shard_id)]]
  executor(
    context = runtime, shard_id = as.integer(shard_id),
    setup_keys = descriptor$setup_keys, target_rows = descriptor$target_rows,
    catalog = catalog, execution_snapshot = iteration_snapshot
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
