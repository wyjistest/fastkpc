fail <- function(message) stop(message, call. = FALSE)

assert_true <- function(value, message) {
  if (!isTRUE(value)) fail(message)
}

assert_error <- function(expression, pattern, message) {
  error <- tryCatch({
    force(expression)
    NULL
  }, error = identity)
  assert_true(
    inherits(error, "error") &&
      grepl(pattern, conditionMessage(error), fixed = TRUE),
    paste0(message, if (inherits(error, "error")) {
      paste0(": ", conditionMessage(error))
    } else {
      ": no error"
    })
  )
}

source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")

hex64 <- function(value) sprintf("%064x", as.integer(value))

fixture_index <- seq_len(80L)
fixture_data <- cbind(
  sin(fixture_index / 5) + fixture_index / 100,
  cos(fixture_index / 7) + fixture_index / 200,
  ((fixture_index * 7L) %% 23L) / 23 + sin(fixture_index / 11),
  ((fixture_index * 11L) %% 29L) / 29 + cos(fixture_index / 13),
  sin(fixture_index / 3) + ((fixture_index * 5L) %% 31L) / 31,
  cos(fixture_index / 4) + ((fixture_index * 13L) %% 37L) / 37,
  sin(fixture_index / 8) + ((fixture_index * 17L) %% 41L) / 41,
  cos(fixture_index / 9) + ((fixture_index * 19L) %% 43L) / 43
)
storage.mode(fixture_data) <- "double"
fixture_dataset_sha256 <- fastkpc_full_cuda_data_hash(fixture_data)

fixture_S <- list(
  2L, 3L, 4L, c(2L, 3L), c(3L, 4L), c(2L, 3L, 4L)
)
fixture_targets <- c(1L, 5L, 6L, 7L, 8L, 1L)
fixture_S_key <- vapply(
  fixture_S, function(value) paste(value, collapse = "|"), character(1L)
)
fixture_formula_class <- vapply(
  fixture_S, fastkpc_regrxons_formula_class, character(1L)
)
fixture_requests <- fastkpc_full_cuda_census_build_key_map(
  target = fixture_targets,
  S_key = fixture_S_key,
  formula_class = fixture_formula_class,
  data_hash = fixture_dataset_sha256,
  n = nrow(fixture_data),
  p = ncol(fixture_data)
)$map
fixture_requests$shard_id <- 0L

fixture_runs <- lapply(seq_len(nrow(fixture_requests)), function(index) {
  fastkpc_full_cuda_census_fit_key(
    data = fixture_data,
    request_row = fixture_requests[index, , drop = FALSE]
  )
})
fixture_setup_observations <- fastkpc_full_cuda_census_bind_rows(lapply(
  fixture_runs, `[[`, "setup_observation"
))
fixture_setup_rows <- fastkpc_full_cuda_census_compress_setups(
  fixture_setup_observations
)
fixture_target_rows <- fastkpc_full_cuda_census_bind_rows(lapply(
  fixture_runs, `[[`, "target_fit"
))

fixture_key_by_row <- vapply(seq_len(nrow(fixture_setup_rows)), function(i) {
  fastkpc_full_cuda_prepared_s_key(
    setup_row = fixture_setup_rows[i, , drop = FALSE],
    dataset_sha256 = fixture_dataset_sha256,
    R_version = R.version.string,
    mgcv_version = as.character(utils::packageVersion("mgcv"))
  )$sha256
}, character(1L))
fixture_scramble <- order(
  fixture_key_by_row, decreasing = TRUE, method = "radix"
)
fixture_setup_rows <- fixture_setup_rows[
  fixture_scramble, , drop = FALSE
]
rownames(fixture_setup_rows) <- NULL
fixture_requests <- fixture_requests[
  rev(seq_len(nrow(fixture_requests))), , drop = FALSE
]
rownames(fixture_requests) <- NULL
fixture_target_rows <- fixture_target_rows[
  rev(seq_len(nrow(fixture_target_rows))), , drop = FALSE
]
rownames(fixture_target_rows) <- NULL

fixture_inputs <- list(
  data = fixture_data,
  dataset_sha256 = fixture_dataset_sha256,
  dataset_file_sha256 = hex64(1001L),
  phase1_input_bundle_hash = hex64(1002L),
  manifest = list(
    canonical_logical_census_hash = hex64(1003L),
    canonical_key_corpus_hash = fastkpc_full_cuda_census_key_set_hash(
      sort(
        as.character(fixture_requests$residual_key_sha256),
        method = "radix"
      )
    ),
    R_version = R.version.string,
    mgcv_version = as.character(utils::packageVersion("mgcv"))
  ),
  same_s_setup_metadata = fixture_setup_rows,
  residual_requests = fixture_requests,
  target_fit_metadata = fixture_target_rows
)

fixture_prepared <- lapply(seq_len(nrow(fixture_setup_rows)), function(i) {
  fastkpc_full_cuda_build_prepared_s_setup(
    inputs = fixture_inputs,
    setup_row = fixture_setup_rows[i, , drop = FALSE]
  )
})
fixture_prepared_keys <- vapply(
  fixture_prepared, `[[`, character(1L), "prepared_s_key_sha256"
)
names(fixture_prepared) <- fixture_prepared_keys
fixture_states <- lapply(fixture_prepared, function(setup) {
  fastkpc_full_cuda_build_target_states(
    inputs = fixture_inputs,
    prepared_setup = setup
  )
})
names(fixture_states) <- fixture_prepared_keys

setup_index <- fastkpc_full_cuda_prepared_s_setup_index(fixture_inputs)
assert_true(
  nrow(setup_index) == 6L &&
    identical(
      setup_index$prepared_s_key_sha256,
      sort(setup_index$prepared_s_key_sha256, decreasing = TRUE,
           method = "radix")
    ) &&
    all(grepl("^[0-9a-f]{64}$", setup_index$prepared_s_key_sha256)),
  "fixture must contain six scrambled lowercase hexadecimal PreparedSKeys"
)

assigned <- fastkpc_full_cuda_prepared_s_assign_shards(
  setup_index, shard_count = 3L
)
sorted_setup_keys <- sort(
  setup_index$prepared_s_key_sha256, method = "radix"
)
assert_true(
  identical(assigned$prepared_s_key_sha256, sorted_setup_keys) &&
    identical(assigned$sorted_rank, seq_len(6L)) &&
    identical(assigned$shard_id, rep(0:2, 2L)),
  "Prepared-S shards must use radix sorted-rank modulo assignment"
)

assert_error(
  fastkpc_full_cuda_prepared_s_assign_shards(
    setup_index[0L, , drop = FALSE], 3L
  ),
  "requires at least one PreparedSKey",
  "Prepared-S shard assignment must reject an empty setup set"
)
na_index <- setup_index
na_index$prepared_s_key_sha256[[1L]] <- NA_character_
assert_error(
  fastkpc_full_cuda_prepared_s_assign_shards(na_index, 3L),
  "PreparedSKey is invalid",
  "Prepared-S shard assignment must reject NA keys"
)
uppercase_index <- setup_index
uppercase_index$prepared_s_key_sha256[[1L]] <-
  toupper(uppercase_index$prepared_s_key_sha256[[1L]])
assert_error(
  fastkpc_full_cuda_prepared_s_assign_shards(uppercase_index, 3L),
  "PreparedSKey is invalid",
  "Prepared-S shard assignment must reject non-lowercase hexadecimal keys"
)
duplicate_index <- setup_index
duplicate_index$prepared_s_key_sha256[[2L]] <-
  duplicate_index$prepared_s_key_sha256[[1L]]
assert_error(
  fastkpc_full_cuda_prepared_s_assign_shards(duplicate_index, 3L),
  "duplicate PreparedSKey",
  "Prepared-S shard assignment must reject duplicate keys"
)
assert_error(
  fastkpc_full_cuda_prepared_s_assign_shards(setup_index, 0L),
  "shard_count must be one positive integer",
  "Prepared-S shard assignment must reject a nonpositive shard count"
)

sparse_assigned <- fastkpc_full_cuda_prepared_s_assign_shards(
  setup_index, shard_count = 7L
)
context <- fastkpc_full_cuda_prepared_s_output_shard_context(fixture_inputs)
sparse_manifest <- fastkpc_full_cuda_prepared_s_shard_manifest(
  sparse_assigned, shard_id = 6L, inputs = fixture_inputs,
  context = context
)
assert_true(
  sparse_manifest$shard_count == 7L &&
    sparse_manifest$shard_id == 6L &&
    sparse_manifest$expected_setup_count_for_shard == 0L &&
    sparse_manifest$expected_target_count_for_shard == 0L,
  "Prepared-S manifests must retain explicit empty trailing shards"
)

required_manifest_fields <- c(
  "phase1_input_bundle_hash", "dataset_file_sha256",
  "dataset_matrix_sha256", "canonical_logical_census_hash",
  "canonical_target_key_corpus_hash", "prepared_s_key_corpus_hash",
  "source_commit", "R_version", "mgcv_version", "BLAS_identity",
  "LAPACK_identity", "BLAS_thread_count",
  "prepared_s_setup_schema_version", "target_state_schema_version",
  "semantic_tolerance_config_hash",
  "qualification_selection_config_hash",
  "expected_setup_count_for_shard", "expected_setup_hash_for_shard",
  "expected_target_count_for_shard", "expected_target_hash_for_shard",
  "shard_count", "shard_id"
)
for (shard_id in 0:2) {
  manifest <- fastkpc_full_cuda_prepared_s_shard_manifest(
    assigned, shard_id, fixture_inputs, context
  )
  shard_groups <- assigned$same_S_group_id[
    assigned$shard_id == shard_id
  ]
  expected_targets <- fixture_inputs$residual_requests[
    fixture_inputs$residual_requests$same_S_group_id %in% shard_groups,
    , drop = FALSE
  ]
  assert_true(
    all(required_manifest_fields %in% names(manifest)) &&
      manifest$expected_setup_count_for_shard == 2L &&
      manifest$expected_target_count_for_shard == nrow(expected_targets) &&
      grepl("^[0-9a-f]{64}$", manifest$expected_setup_hash_for_shard) &&
      grepl("^[0-9a-f]{64}$", manifest$expected_target_hash_for_shard),
    "Prepared-S manifest must freeze complete shard identity and counts"
  )
}
assert_error(
  fastkpc_full_cuda_prepared_s_shard_manifest(
    assigned, 3L, fixture_inputs, context
  ),
  "shard_id is outside assigned shard range",
  "Prepared-S shard_id must remain zero based and in range"
)

named_states <- fixture_states[[1L]]
renamed_states <- named_states
names(renamed_states$convergence_fields[[1L]])[[1L]] <- "renamed"
assert_true(
  !identical(
    fastkpc_full_cuda_census_frame_hash(named_states),
    fastkpc_full_cuda_census_frame_hash(renamed_states)
  ),
  "TargetState frame authentication must preserve semantic nested names"
)

builder_calls <- new.env(parent = emptyenv())
builder_calls$setup <- 0L
builder_calls$target <- 0L
fixture_setup_builder <- function(inputs, setup_row) {
  builder_calls$setup <- builder_calls$setup + 1L
  group_id <- as.character(setup_row$same_S_group_id[[1L]])
  index <- which(vapply(fixture_prepared, function(value) {
    identical(value$same_S_group_id, group_id)
  }, logical(1L)))
  fixture_prepared[[index]]
}
fixture_target_builder <- function(inputs, prepared_setup) {
  builder_calls$target <- builder_calls$target + 1L
  fixture_states[[prepared_setup$prepared_s_key_sha256]]
}

forbidden_public_formals <- c(
  "setup_builder", "target_builder", "builder", "validator",
  "hash_fun", "interruption_hook"
)
public_formals <- names(formals(fastkpc_full_cuda_run_prepared_s_shard))
public_body <- paste(
  deparse(body(fastkpc_full_cuda_run_prepared_s_shard)), collapse = "\n"
)
assert_true(
  length(intersect(public_formals, forbidden_public_formals)) == 0L &&
    grepl("fastkpc_full_cuda_build_prepared_s_setup", public_body,
          fixed = TRUE) &&
    grepl("fastkpc_full_cuda_build_target_states", public_body,
          fixed = TRUE),
  "public Prepared-S runner must bind real builders without authority injection"
)

assert_context_error <- function(mutator, pattern, message) {
  wrong <- mutator(context)
  builder_calls$setup <- 0L
  builder_calls$target <- 0L
  assert_error(
    .fastkpc_full_cuda_run_prepared_s_shard_core(
      inputs = fixture_inputs,
      assigned_setups = assigned,
      shard_id = 0L,
      context = wrong,
      output_dir = tempfile("prepared-s-wrong-context-"),
      setup_builder = fixture_setup_builder,
      target_builder = fixture_target_builder
    ),
    pattern,
    message
  )
  assert_true(
    builder_calls$setup == 0L && builder_calls$target == 0L,
    paste(message, "must fail before any builder call")
  )
}

assert_context_error(
  function(value) {
    value$phase1_input_bundle_hash <- hex64(2001L)
    value
  },
  "Phase 1 input bundle hash mismatch",
  "Prepared-S shard must reject a wrong Phase 1 input bundle hash"
)
assert_context_error(
  function(value) {
    value$prepared_s_key_corpus_hash <- hex64(2002L)
    value
  },
  "PreparedSKey corpus hash mismatch",
  "Prepared-S shard must reject a wrong PreparedSKey corpus hash"
)
assert_context_error(
  function(value) {
    value$schema_config$prepared_s_setup_schema_version <- "wrong-schema"
    value
  },
  "schema config mismatch",
  "Prepared-S shard must reject a wrong schema config"
)
assert_context_error(
  function(value) {
    value$semantic_tolerance_config$dcov_absolute_p_tolerance <- 1e-9
    value
  },
  "semantic tolerance config mismatch",
  "Prepared-S shard must reject a wrong semantic tolerance config"
)
assert_context_error(
  function(value) {
    value$qualification_selection_config$iteration_setup_count <- 45L
    value
  },
  "qualification selection config mismatch",
  "Prepared-S shard must reject a wrong qualification selection config"
)

response_builder <- function(inputs, setup_row) {
  setup <- fixture_setup_builder(inputs, setup_row)
  setup$nested <- list(y = inputs$data[, 1L])
  setup
}
response_dir <- tempfile("prepared-s-response-leak-")
assert_error(
  .fastkpc_full_cuda_run_prepared_s_shard_core(
    inputs = fixture_inputs,
    assigned_setups = assigned,
    shard_id = 0L,
    context = context,
    output_dir = response_dir,
    setup_builder = response_builder,
    target_builder = fixture_target_builder
  ),
  "PreparedSSetup response-bearing field",
  "recursive shard validation must reject response-bearing y in setup"
)
response_paths <- fastkpc_full_cuda_prepared_s_shard_paths(
  response_dir, 0L
)
assert_true(
  !file.exists(response_paths$rds) &&
    !file.exists(response_paths$summary_json),
  "response leakage failure must not publish a reusable final pair"
)

interrupt_dir <- tempfile("prepared-s-interrupt-")
interrupt_hook <- function(stage) {
  if (identical(stage, "after_temp_rds")) {
    stop("injected interruption after temporary RDS", call. = FALSE)
  }
}
assert_error(
  .fastkpc_full_cuda_run_prepared_s_shard_core(
    inputs = fixture_inputs,
    assigned_setups = assigned,
    shard_id = 1L,
    context = context,
    output_dir = interrupt_dir,
    setup_builder = fixture_setup_builder,
    target_builder = fixture_target_builder,
    interruption_hook = interrupt_hook
  ),
  "injected interruption after temporary RDS",
  "interruption after temp RDS must propagate"
)
interrupt_paths <- fastkpc_full_cuda_prepared_s_shard_paths(
  interrupt_dir, 1L
)
assert_true(
  !file.exists(interrupt_paths$rds) &&
    !file.exists(interrupt_paths$summary_json) &&
    length(list.files(interrupt_dir, pattern = "\\.tmp")) == 0L,
  "interruption before publish must leave no final pair or temp files"
)
interrupted_retry <- .fastkpc_full_cuda_run_prepared_s_shard_core(
  inputs = fixture_inputs,
  assigned_setups = assigned,
  shard_id = 1L,
  context = context,
  output_dir = interrupt_dir,
  setup_builder = fixture_setup_builder,
  target_builder = fixture_target_builder
)
assert_true(
  interrupted_retry$status == "written" &&
    file.exists(interrupt_paths$rds) &&
    file.exists(interrupt_paths$summary_json),
  "rerun after pre-publish interruption must complete deterministically"
)

output_dir <- tempfile("prepared-s-restart-")
builder_calls$setup <- 0L
builder_calls$target <- 0L
first <- .fastkpc_full_cuda_run_prepared_s_shard_core(
  inputs = fixture_inputs,
  assigned_setups = assigned,
  shard_id = 0L,
  context = context,
  output_dir = output_dir,
  setup_builder = fixture_setup_builder,
  target_builder = fixture_target_builder
)
assert_true(
  first$status == "written" &&
    builder_calls$setup == 2L && builder_calls$target == 2L &&
    file.exists(first$paths$rds) &&
    file.exists(first$paths$summary_json),
  "first valid Prepared-S shard run must build and publish one final pair"
)
first_summary <- jsonlite::read_json(
  first$paths$summary_json, simplifyVector = TRUE
)
required_authentication <- c(
  "manifest_hash", "setup_key_set_hash", "target_key_set_hash",
  "prepared_s_setup_hashes", "prepared_s_setup_hashes_hash",
  "target_state_frame_hash", "payload_hash", "rds_file_sha256",
  "completion_hash"
)
assert_true(
  all(required_authentication %in% names(first_summary)) &&
    identical(
      as.character(first_summary$rds_file_sha256),
      fastkpc_full_cuda_census_file_hash(first$paths$rds)
    ),
  "completion JSON must authenticate payload semantics and final RDS bytes"
)

first_rds_hash <- fastkpc_full_cuda_census_file_hash(first$paths$rds)
first_json_hash <- fastkpc_full_cuda_census_file_hash(
  first$paths$summary_json
)
builder_calls$setup <- 0L
builder_calls$target <- 0L
reused <- .fastkpc_full_cuda_run_prepared_s_shard_core(
  inputs = fixture_inputs,
  assigned_setups = assigned,
  shard_id = 0L,
  context = context,
  output_dir = output_dir,
  setup_builder = fixture_setup_builder,
  target_builder = fixture_target_builder
)
assert_true(
  reused$status == "reused" &&
    builder_calls$setup == 0L && builder_calls$target == 0L &&
    identical(first_rds_hash,
              fastkpc_full_cuda_census_file_hash(first$paths$rds)) &&
    identical(first_json_hash,
              fastkpc_full_cuda_census_file_hash(first$paths$summary_json)),
  "valid resume must reuse with zero builder calls and byte-identical files"
)

summary_backup <- tempfile("prepared-s-summary-backup-")
assert_true(
  file.copy(first$paths$summary_json, summary_backup, overwrite = TRUE),
  "test fixture must back up completion JSON"
)
corrupt_summary <- jsonlite::read_json(
  first$paths$summary_json, simplifyVector = TRUE
)
corrupt_summary$payload_hash <- hex64(3001L)
fastkpc_full_cuda_write_json(corrupt_summary, first$paths$summary_json)
assert_error(
  .fastkpc_full_cuda_run_prepared_s_shard_core(
    inputs = fixture_inputs,
    assigned_setups = assigned,
    shard_id = 0L,
    context = context,
    output_dir = output_dir,
    setup_builder = fixture_setup_builder,
    target_builder = fixture_target_builder
  ),
  "Prepared-S shard payload hash mismatch",
  "resume must reject a corrupt payload hash"
)
assert_true(
  file.exists(first$paths$rds) && file.exists(first$paths$summary_json),
  "corrupt final pair must fail closed without silent deletion"
)
assert_true(
  file.copy(summary_backup, first$paths$summary_json, overwrite = TRUE),
  "test fixture must restore completion JSON"
)

missing_json_backup <- tempfile("prepared-s-missing-json-")
assert_true(
  file.rename(first$paths$summary_json, missing_json_backup),
  "test fixture must hide completion JSON"
)
assert_error(
  .fastkpc_full_cuda_run_prepared_s_shard_core(
    inputs = fixture_inputs,
    assigned_setups = assigned,
    shard_id = 0L,
    context = context,
    output_dir = output_dir,
    setup_builder = fixture_setup_builder,
    target_builder = fixture_target_builder
  ),
  "half-published Prepared-S shard",
  "missing completion JSON must fail closed"
)
assert_true(
  file.exists(first$paths$rds) && !file.exists(first$paths$summary_json),
  "half-published RDS must not be silently overwritten"
)
assert_true(
  file.rename(missing_json_backup, first$paths$summary_json),
  "test fixture must restore completion JSON"
)

stale_inputs <- fixture_inputs
stale_inputs$phase1_input_bundle_hash <- hex64(4001L)
stale_context <- fastkpc_full_cuda_prepared_s_output_shard_context(
  stale_inputs
)
assert_error(
  .fastkpc_full_cuda_run_prepared_s_shard_core(
    inputs = stale_inputs,
    assigned_setups = assigned,
    shard_id = 0L,
    context = stale_context,
    output_dir = output_dir,
    setup_builder = fixture_setup_builder,
    target_builder = fixture_target_builder
  ),
  "Prepared-S shard manifest mismatch",
  "resume must reject a stale but internally valid manifest context"
)

builder_calls$setup <- 0L
builder_calls$target <- 0L
empty_run <- .fastkpc_full_cuda_run_prepared_s_shard_core(
  inputs = fixture_inputs,
  assigned_setups = sparse_assigned,
  shard_id = 6L,
  context = context,
  output_dir = tempfile("prepared-s-empty-shard-"),
  setup_builder = fixture_setup_builder,
  target_builder = fixture_target_builder
)
assert_true(
  empty_run$status == "written" &&
    builder_calls$setup == 0L && builder_calls$target == 0L &&
    length(empty_run$payload$ordered_setup_keys) == 0L &&
    nrow(empty_run$payload$target_states) == 0L,
  "empty trailing shard must publish without invoking builders"
)

merge_dir <- tempfile("prepared-s-merge-")
for (shard_id in 0:2) {
  .fastkpc_full_cuda_run_prepared_s_shard_core(
    inputs = fixture_inputs,
    assigned_setups = assigned,
    shard_id = shard_id,
    context = context,
    output_dir = merge_dir,
    setup_builder = fixture_setup_builder,
    target_builder = fixture_target_builder
  )
}
merged <- fastkpc_full_cuda_merge_prepared_s_shards(
  inputs = fixture_inputs,
  shard_count = 3L,
  shard_dir = merge_dir
)
assert_true(
  identical(names(merged$prepared_s_setups), sorted_setup_keys) &&
    identical(
      as.character(merged$target_states$prepared_s_key_sha256),
      sorted_setup_keys
    ) &&
    nrow(merged$target_states) == 6L,
  "Prepared-S merge must restore exact canonical setup and target order"
)

missing_paths <- fastkpc_full_cuda_prepared_s_shard_paths(merge_dir, 2L)
missing_backup <- tempfile("prepared-s-merge-missing-")
assert_true(
  file.rename(missing_paths$summary_json, missing_backup),
  "test fixture must hide one merge completion JSON"
)
assert_error(
  fastkpc_full_cuda_merge_prepared_s_shards(
    fixture_inputs, 3L, merge_dir
  ),
  "missing Prepared-S shard completion",
  "Prepared-S merge must reject a missing shard"
)
assert_true(
  file.rename(missing_backup, missing_paths$summary_json),
  "test fixture must restore missing merge completion JSON"
)

shard_zero <- fastkpc_full_cuda_prepared_s_shard_paths(merge_dir, 0L)
shard_one <- fastkpc_full_cuda_prepared_s_shard_paths(merge_dir, 1L)
one_rds_backup <- tempfile("prepared-s-merge-one-rds-")
one_json_backup <- tempfile("prepared-s-merge-one-json-")
assert_true(
  file.copy(shard_one$rds, one_rds_backup, overwrite = TRUE) &&
    file.copy(shard_one$summary_json, one_json_backup, overwrite = TRUE) &&
    file.copy(shard_zero$rds, shard_one$rds, overwrite = TRUE) &&
    file.copy(shard_zero$summary_json, shard_one$summary_json,
              overwrite = TRUE),
  "test fixture must install a duplicate declared Prepared-S shard"
)
assert_error(
  fastkpc_full_cuda_merge_prepared_s_shards(
    fixture_inputs, 3L, merge_dir
  ),
  "duplicate Prepared-S shard id",
  "Prepared-S merge must reject duplicate declared shards"
)
assert_true(
  file.copy(one_rds_backup, shard_one$rds, overwrite = TRUE) &&
    file.copy(one_json_backup, shard_one$summary_json, overwrite = TRUE),
  "test fixture must restore duplicate merge shard"
)

cat("PASS full CUDA CI Prepared-S restart qualification\n")
