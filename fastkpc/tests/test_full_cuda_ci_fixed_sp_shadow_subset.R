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
    all(rows$backend == "legacy-cpp") &&
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
