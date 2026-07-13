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
  2L, 3L, 4L, c(2L, 3L), c(3L, 4L), c(2L, 3L, 4L), 2L
)
fixture_targets <- c(1L, 5L, 6L, 7L, 8L, 1L, 5L)
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

multi_target_setup_index <- which(vapply(
  fixture_states, nrow, integer(1L)
) == 2L)
assert_true(
  length(fixture_prepared) == 6L &&
    length(multi_target_setup_index) == 1L &&
    sum(vapply(fixture_states, nrow, integer(1L))) == 7L,
  "fixture must contain six setup groups and one two-target group"
)
indexed_fixture_input <- .fastkpc_full_cuda_target_state_input_index(
  fixture_inputs
)
indexed_fixture_context <-
  .fastkpc_full_cuda_target_state_context_from_index(
    indexed_fixture_input,
    fixture_prepared[[multi_target_setup_index]]
  )
indexed_fixture_states <-
  .fastkpc_full_cuda_build_target_states_from_context(
    context = indexed_fixture_context,
    prepared_setup = fixture_prepared[[multi_target_setup_index]]
  )
.fastkpc_full_cuda_validate_target_states_with_context(
  states = indexed_fixture_states,
  context = indexed_fixture_context,
  prepared_setup = fixture_prepared[[multi_target_setup_index]]
)
assert_true(
  identical(
    indexed_fixture_states,
    fixture_states[[multi_target_setup_index]]
  ) &&
    identical(
      fastkpc_full_cuda_census_frame_hash(indexed_fixture_states),
      fastkpc_full_cuda_census_frame_hash(
        fixture_states[[multi_target_setup_index]]
      )
    ),
  "indexed and public TargetState paths must be frame/hash exact"
)

pairlist_setup_fixture <- as.pairlist(fixture_prepared[[1L]])
pairlist_setup_row_index <- which(
  as.character(fixture_setup_rows$same_S_group_id) ==
    fixture_prepared[[1L]]$same_S_group_id
)
assert_true(
  length(pairlist_setup_row_index) == 1L,
  "pairlist PreparedSSetup fixture must retain one setup-row lineage"
)
assert_error(
  fastkpc_full_cuda_validate_prepared_s_setup(
    setup = pairlist_setup_fixture,
    setup_row = fixture_setup_rows[
      pairlist_setup_row_index, , drop = FALSE
    ],
    dataset_sha256 = fixture_dataset_sha256
  ),
  "PreparedSSetup contains non-plain data",
  "PreparedSSetup validator must reject a pairlist root object"
)

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
fixture_target_builder <- function(
    inputs = NULL, context = NULL, prepared_setup) {
  builder_calls$target <- builder_calls$target + 1L
  fixture_states[[prepared_setup$prepared_s_key_sha256]]
}

index_builder_calls <- new.env(parent = emptyenv())
index_builder_calls$run <- 0L
counting_run_index_builder <- function(inputs) {
  index_builder_calls$run <- index_builder_calls$run + 1L
  .fastkpc_full_cuda_target_state_input_index(inputs)
}
indexed_run_dir <- tempfile("prepared-s-indexed-run-")
indexed_run <- .fastkpc_full_cuda_run_prepared_s_shard_core(
  inputs = fixture_inputs,
  assigned_setups = assigned,
  shard_id = 0L,
  context = context,
  output_dir = indexed_run_dir,
  setup_builder = fixture_setup_builder,
  target_builder = fixture_target_builder,
  target_state_index_builder = counting_run_index_builder
)
assert_true(
  indexed_run$status == "written" && index_builder_calls$run == 1L,
  "one multi-setup shard run must build the TargetState index once"
)

forbidden_public_formals <- c(
  "setup_builder", "target_builder", "builder", "validator",
  "hash_fun", "interruption_hook", "target_state_index",
  "target_state_index_builder", "target_state_context"
)
public_formals <- names(formals(fastkpc_full_cuda_run_prepared_s_shard))
assert_true(
  length(intersect(public_formals, forbidden_public_formals)) == 0L,
  "public Prepared-S runner must expose no builder or hash authority"
)
public_runner_dir <- tempfile("prepared-s-public-runner-")
public_written <- fastkpc_full_cuda_run_prepared_s_shard(
  inputs = fixture_inputs,
  shard_count = 3L,
  shard_id = 2L,
  output_dir = public_runner_dir
)
public_written_summary <- jsonlite::read_json(
  public_written$paths$summary_json, simplifyVector = TRUE
)
fastkpc_full_cuda_validate_prepared_s_shard(
  payload = readRDS(public_written$paths$rds),
  summary = public_written_summary,
  inputs = fixture_inputs,
  shard_count = 3L,
  shard_id = 2L,
  rds_path = public_written$paths$rds
)
public_reused <- fastkpc_full_cuda_run_prepared_s_shard(
  inputs = fixture_inputs,
  shard_count = 3L,
  shard_id = 2L,
  output_dir = public_runner_dir
)
assert_true(
  public_written$status == "written" && public_reused$status == "reused",
  paste(
    "public production runner must write and reuse a shard validated",
    "against the caller's trusted layout"
  )
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
    !file.exists(response_paths$summary_json) &&
    !file.exists(response_paths$lock_dir),
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
    !file.exists(interrupt_paths$lock_dir) &&
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

lock_race_dir <- tempfile("prepared-s-lock-race-")
lock_race_path <- file.path(lock_race_dir, "shard_0.lock")
second_writer_calls <- new.env(parent = emptyenv())
second_writer_calls$setup <- 0L
second_writer_calls$target <- 0L
second_writer_setup_builder <- function(inputs, setup_row) {
  second_writer_calls$setup <- second_writer_calls$setup + 1L
  fixture_setup_builder(inputs, setup_row)
}
second_writer_target_builder <- function(
    inputs = NULL, context = NULL, prepared_setup) {
  second_writer_calls$target <- second_writer_calls$target + 1L
  fixture_target_builder(
    inputs = inputs, context = context, prepared_setup = prepared_setup
  )
}
lock_race_hook <- function(stage) {
  if (!identical(stage, "before_publish")) return(invisible(NULL))
  assert_true(
    dir.exists(lock_race_path),
    "first Prepared-S writer must hold its shard lock before publish"
  )
  assert_error(
    .fastkpc_full_cuda_run_prepared_s_shard_core(
      inputs = fixture_inputs,
      assigned_setups = assigned,
      shard_id = 0L,
      context = context,
      output_dir = lock_race_dir,
      setup_builder = second_writer_setup_builder,
      target_builder = second_writer_target_builder
    ),
    "Prepared-S shard lock exists",
    "second Prepared-S writer must fail closed while the lock is held"
  )
  assert_true(
    second_writer_calls$setup == 0L &&
      second_writer_calls$target == 0L,
    "second writer lock failure must occur before any builder call"
  )
  invisible(NULL)
}
lock_race_written <- .fastkpc_full_cuda_run_prepared_s_shard_core(
  inputs = fixture_inputs,
  assigned_setups = assigned,
  shard_id = 0L,
  context = context,
  output_dir = lock_race_dir,
  setup_builder = fixture_setup_builder,
  target_builder = fixture_target_builder,
  interruption_hook = lock_race_hook
)
lock_race_loaded <- fastkpc_full_cuda_prepared_s_read_shard(
  shard_dir = lock_race_dir,
  inputs = fixture_inputs,
  shard_count = 3L,
  shard_id = 0L
)
assert_true(
  lock_race_written$status == "written" &&
    !dir.exists(lock_race_path) &&
    identical(
      lock_race_loaded$payload$manifest,
      lock_race_written$payload$manifest
    ),
  "first writer must release its lock and publish one valid final pair"
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

local({
  source_commit_a <- first$payload$manifest$source_commit
  source_commit_b <- paste(rep("b", 40L), collapse = "")
  batch_dir <- tempfile("prepared-s-batch-reader-")
  batch_first <- fastkpc_full_cuda_run_prepared_s_shard(
    inputs = fixture_inputs,
    shard_count = 3L,
    shard_id = 0L,
    output_dir = batch_dir
  )
  batch_second <- fastkpc_full_cuda_run_prepared_s_shard(
    inputs = fixture_inputs,
    shard_count = 3L,
    shard_id = 1L,
    output_dir = batch_dir
  )
  assert_true(
    batch_first$status == "written" && batch_second$status == "written",
    "batch reader fixture must write two authenticated Prepared-S shards"
  )
  original_runtime_identity <- fastkpc_full_cuda_census_runtime_identity
  on.exit(
    assign(
      "fastkpc_full_cuda_census_runtime_identity",
      original_runtime_identity,
      envir = .GlobalEnv
    ),
    add = TRUE
  )
  assign(
    "fastkpc_full_cuda_census_runtime_identity",
    function() {
      identity <- original_runtime_identity()
      identity$source_commit <- source_commit_b
      identity
    },
    envir = .GlobalEnv
  )

  preparation_calls <- new.env(parent = emptyenv())
  preparation_calls$assignment <- 0L
  preparation_calls$output_context <- 0L
  preparation_calls$target_state_index <- 0L
  original_assign_shards <- fastkpc_full_cuda_prepared_s_assign_shards
  original_output_context <- fastkpc_full_cuda_prepared_s_output_shard_context
  original_target_state_index <- .fastkpc_full_cuda_target_state_input_index
  on.exit(
    assign(
      "fastkpc_full_cuda_prepared_s_assign_shards",
      original_assign_shards,
      envir = .GlobalEnv
    ),
    add = TRUE
  )
  on.exit(
    assign(
      "fastkpc_full_cuda_prepared_s_output_shard_context",
      original_output_context,
      envir = .GlobalEnv
    ),
    add = TRUE
  )
  on.exit(
    assign(
      ".fastkpc_full_cuda_target_state_input_index",
      original_target_state_index,
      envir = .GlobalEnv
    ),
    add = TRUE
  )
  assign(
    "fastkpc_full_cuda_prepared_s_assign_shards",
    function(setup_index, shard_count) {
      preparation_calls$assignment <- preparation_calls$assignment + 1L
      original_assign_shards(setup_index, shard_count)
    },
    envir = .GlobalEnv
  )
  assign(
    "fastkpc_full_cuda_prepared_s_output_shard_context",
    function(inputs) {
      preparation_calls$output_context <-
        preparation_calls$output_context + 1L
      original_output_context(inputs)
    },
    envir = .GlobalEnv
  )
  assign(
    ".fastkpc_full_cuda_target_state_input_index",
    function(inputs) {
      preparation_calls$target_state_index <-
        preparation_calls$target_state_index + 1L
      original_target_state_index(inputs)
    },
    envir = .GlobalEnv
  )

  traversed_setup_keys <- c(
    as.character(batch_second$payload$ordered_setup_keys[[1L]]),
    as.character(batch_first$payload$ordered_setup_keys[[1L]])
  )
  traversed_target_keys <- c(
    as.character(batch_second$payload$target_states$residual_key_sha256[
      match(
        traversed_setup_keys[[1L]],
        batch_second$payload$target_states$prepared_s_key_sha256
      )
    ]),
    as.character(batch_first$payload$target_states$residual_key_sha256[
      match(
        traversed_setup_keys[[2L]],
        batch_first$payload$target_states$prepared_s_key_sha256
      )
    ])
  )
  selected_setup_keys <- rev(traversed_setup_keys)
  selected_target_keys <- rev(traversed_target_keys)
  expected_selected_setups <- c(
    batch_first$payload$prepared_s_setups[selected_setup_keys[[1L]]],
    batch_second$payload$prepared_s_setups[selected_setup_keys[[2L]]]
  )
  expected_selected_states <- fastkpc_full_cuda_prepared_s_bind_target_states(
    list(
      batch_first$payload$target_states[
        match(
          selected_target_keys[[1L]],
          batch_first$payload$target_states$residual_key_sha256
        ), , drop = FALSE
      ],
      batch_second$payload$target_states[
        match(
          selected_target_keys[[2L]],
          batch_second$payload$target_states$residual_key_sha256
        ), , drop = FALSE
      ]
    )
  )
  selected_replayed <- fastkpc_full_cuda_prepared_s_read_selected_shards(
    shard_dir = batch_dir,
    inputs = fixture_inputs,
    shard_count = 3L,
    shard_ids = c(1L, 0L),
    setup_keys = selected_setup_keys,
    target_keys = selected_target_keys,
    expected_source_commit = source_commit_a
  )
  assert_true(
    identical(names(selected_replayed), c(
      "prepared_s_setups", "target_states", "shard_ids"
    )) &&
      identical(selected_replayed$shard_ids, c(1L, 0L)) &&
      identical(names(selected_replayed$prepared_s_setups),
                selected_setup_keys) &&
      identical(selected_replayed$prepared_s_setups,
                expected_selected_setups) &&
      identical(
        as.character(selected_replayed$target_states$residual_key_sha256),
        selected_target_keys
      ) &&
      identical(selected_replayed$target_states, expected_selected_states) &&
      all(as.character(
        selected_replayed$target_states$prepared_s_key_sha256
      ) %in% selected_setup_keys) &&
      preparation_calls$assignment == 1L &&
      preparation_calls$output_context == 1L &&
      preparation_calls$target_state_index == 1L,
    paste(
      "selected shard reader must retain only requested exact payloads in",
      "caller order after one shared preparation"
    )
  )

  invalid_selected_keys <- list(
    missing = paste(rep("f", 64L), collapse = ""),
    duplicate = rep(selected_setup_keys[[1L]], 2L),
    empty = character(),
    attributed = structure(selected_setup_keys[[1L]], note = "unallowed"),
    malformed = "not-a-sha"
  )
  for (label in names(invalid_selected_keys)) {
    assert_error(
      fastkpc_full_cuda_prepared_s_read_selected_shards(
        shard_dir = batch_dir,
        inputs = fixture_inputs,
        shard_count = 3L,
        shard_ids = c(1L, 0L),
        setup_keys = invalid_selected_keys[[label]],
        target_keys = selected_target_keys,
        expected_source_commit = source_commit_a
      ),
      "setup_keys",
      paste("selected shard reader setup_keys must reject", label, "input")
    )
    assert_error(
      fastkpc_full_cuda_prepared_s_read_selected_shards(
        shard_dir = batch_dir,
        inputs = fixture_inputs,
        shard_count = 3L,
        shard_ids = c(1L, 0L),
        setup_keys = selected_setup_keys,
        target_keys = invalid_selected_keys[[label]],
        expected_source_commit = source_commit_a
      ),
      "target_keys",
      paste("selected shard reader target_keys must reject", label, "input")
    )
  }

  selected_corruption_paths <- fastkpc_full_cuda_prepared_s_shard_paths(
    batch_dir, 1L
  )
  selected_corruption_backup <- tempfile("prepared-s-selected-rds-")
  selected_summary_backup <- tempfile("prepared-s-selected-summary-")
  assert_true(
    file.copy(
      selected_corruption_paths$rds, selected_corruption_backup,
      overwrite = TRUE
    ),
    "selected shard reader test must back up its shard RDS"
  )
  assert_true(
    file.copy(
      selected_corruption_paths$summary_json, selected_summary_backup,
      overwrite = TRUE
    ),
    "selected shard reader test must back up its completion JSON"
  )
  selected_corruption <- readRDS(selected_corruption_paths$rds)
  unselected_setup_indices <- which(
    !(as.character(selected_corruption$ordered_setup_keys) %in%
        selected_setup_keys)
  )
  assert_true(
    length(unselected_setup_indices) > 0L,
    "selected shard reader corruption fixture must contain an unrequested setup"
  )
  unselected_setup_index <- unselected_setup_indices[[1L]]
  selected_corruption$prepared_s_setups[[unselected_setup_index]]$
    prepared_s_key_sha256 <- paste(rep("0", 64L), collapse = "")
  saveRDS(selected_corruption, selected_corruption_paths$rds, version = 2)
  selected_summary <- jsonlite::read_json(
    selected_summary_backup, simplifyVector = TRUE
  )
  selected_authentication <-
    fastkpc_full_cuda_prepared_s_shard_authentication(selected_corruption)
  for (field in setdiff(
    names(selected_authentication), "prepared_s_setup_hashes"
  )) {
    selected_summary[[field]] <- selected_authentication[[field]]
  }
  selected_summary$prepared_s_setup_hashes <-
    as.list(selected_authentication$prepared_s_setup_hashes)
  selected_summary$rds_file_sha256 <- fastkpc_full_cuda_census_file_hash(
    selected_corruption_paths$rds
  )
  selected_summary$completion_hash <-
    fastkpc_full_cuda_prepared_s_completion_hash(selected_summary)
  fastkpc_full_cuda_write_json(
    selected_summary, selected_corruption_paths$summary_json
  )
  assert_error(
    fastkpc_full_cuda_prepared_s_read_selected_shards(
      shard_dir = batch_dir,
      inputs = fixture_inputs,
      shard_count = 3L,
      shard_ids = c(1L, 0L),
      setup_keys = selected_setup_keys,
      target_keys = selected_target_keys,
      expected_source_commit = source_commit_a
    ),
    "Prepared-S shard setup lineage mismatch",
    paste(
      "selected shard reader must fully validate reauthenticated unselected",
      "payload content before returning filtered results"
    )
  )
  assert_true(
    file.copy(
      selected_corruption_backup, selected_corruption_paths$rds,
      overwrite = TRUE
    ),
    "selected shard reader test must restore its shard RDS"
  )
  assert_true(
    file.copy(
      selected_summary_backup, selected_corruption_paths$summary_json,
      overwrite = TRUE
    ),
    "selected shard reader test must restore its completion JSON"
  )

  preparation_calls$assignment <- 0L
  preparation_calls$output_context <- 0L
  preparation_calls$target_state_index <- 0L
  batch_replayed <- fastkpc_full_cuda_prepared_s_read_shards(
    shard_dir = batch_dir,
    inputs = fixture_inputs,
    shard_count = 3L,
    shard_ids = c(1L, 0L),
    expected_source_commit = source_commit_a
  )
  assert_true(
    identical(names(batch_replayed), c("1", "0")) &&
      identical(batch_replayed[[1L]]$payload, batch_second$payload) &&
      identical(batch_replayed[[2L]]$payload, batch_first$payload) &&
      preparation_calls$assignment == 1L &&
      preparation_calls$output_context == 1L &&
      preparation_calls$target_state_index == 1L,
    paste(
      "batch reader must preserve requested authenticated payload order",
      "and prepare shared assignment, output context, and target index once"
    )
  )

  assert_error(
    fastkpc_full_cuda_prepared_s_read_shards(
      shard_dir = batch_dir,
      inputs = fixture_inputs,
      shard_count = 3L,
      shard_ids = c(1L, 0L)
    ),
    "Prepared-S shard manifest mismatch",
    "default batch reader must reject a historical source commit"
  )
  assert_error(
    fastkpc_full_cuda_prepared_s_read_shards(
      shard_dir = batch_dir,
      inputs = fixture_inputs,
      shard_count = 3L,
      shard_ids = c(1L, 0L),
      expected_source_commit = source_commit_b
    ),
    "Prepared-S shard manifest mismatch",
    "batch reader must reject an unauthenticated requested source commit"
  )
  assign(
    "fastkpc_full_cuda_census_runtime_identity",
    function() {
      identity <- original_runtime_identity()
      identity$source_commit <- source_commit_b
      identity$BLAS_identity <- "non-source-runtime-identity-mismatch"
      identity
    },
    envir = .GlobalEnv
  )
  assert_error(
    fastkpc_full_cuda_prepared_s_read_shards(
      shard_dir = batch_dir,
      inputs = fixture_inputs,
      shard_count = 3L,
      shard_ids = c(1L, 0L),
      expected_source_commit = source_commit_a
    ),
    "Prepared-S shard manifest mismatch",
    "batch reader must reject a non-source runtime identity mismatch"
  )
  assign(
    "fastkpc_full_cuda_census_runtime_identity",
    function() {
      identity <- original_runtime_identity()
      identity$source_commit <- source_commit_b
      identity
    },
    envir = .GlobalEnv
  )
  invalid_batch_shard_ids <- list(
    duplicate = c(0L, 0L),
    empty = integer(),
    out_of_range = c(0L, 3L),
    attributed = structure(0:1, note = "unallowed"),
    non_integer = c(0, 1)
  )
  for (label in names(invalid_batch_shard_ids)) {
    assert_error(
      fastkpc_full_cuda_prepared_s_read_shards(
        shard_dir = batch_dir,
        inputs = fixture_inputs,
        shard_count = 3L,
        shard_ids = invalid_batch_shard_ids[[label]],
        expected_source_commit = source_commit_a
      ),
      "shard_ids",
      paste("batch reader shard_ids must reject", label, "input")
    )
  }

  assert_error(
    fastkpc_full_cuda_prepared_s_read_shard(
      shard_dir = output_dir,
      inputs = fixture_inputs,
      shard_count = 3L,
      shard_id = 0L
    ),
    "Prepared-S shard manifest mismatch",
    "default public reader must reject a historical source commit"
  )
  replayed <- fastkpc_full_cuda_prepared_s_read_shard(
    shard_dir = output_dir,
    inputs = fixture_inputs,
    shard_count = 3L,
    shard_id = 0L,
    expected_source_commit = source_commit_a
  )
  assert_true(
    identical(replayed$payload, first$payload),
    "public reader must authenticate a requested historical source commit"
  )
  assert_error(
    fastkpc_full_cuda_prepared_s_read_shard(
      shard_dir = output_dir,
      inputs = fixture_inputs,
      shard_count = 3L,
      shard_id = 0L,
      expected_source_commit = source_commit_b
    ),
    "Prepared-S shard manifest mismatch",
    "public reader must reject an unauthenticated requested source commit"
  )
  invalid_expected_source_commits <- list(
    uppercase = toupper(source_commit_a),
    attributed = structure(source_commit_a, note = "unallowed"),
    malformed = "not-a-sha"
  )
  for (label in names(invalid_expected_source_commits)) {
    assert_error(
      fastkpc_full_cuda_prepared_s_read_shard(
        shard_dir = output_dir,
        inputs = fixture_inputs,
        shard_count = 3L,
        shard_id = 0L,
        expected_source_commit = invalid_expected_source_commits[[label]]
      ),
      "expected_source_commit must be a bare lowercase 40-hex SHA-1 scalar",
      paste("expected_source_commit must reject", label, "input")
    )
  }
})

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

layout_dir <- tempfile("prepared-s-layout-mismatch-")
assigned_two <- fastkpc_full_cuda_prepared_s_assign_shards(
  setup_index, shard_count = 2L
)
layout_written <- .fastkpc_full_cuda_run_prepared_s_shard_core(
  inputs = fixture_inputs,
  assigned_setups = assigned_two,
  shard_id = 0L,
  context = context,
  output_dir = layout_dir,
  setup_builder = fixture_setup_builder,
  target_builder = fixture_target_builder
)
assert_true(
  layout_written$status == "written",
  "test fixture must publish the initial two-shard layout"
)
builder_calls$setup <- 0L
builder_calls$target <- 0L
assert_error(
  .fastkpc_full_cuda_run_prepared_s_shard_core(
    inputs = fixture_inputs,
    assigned_setups = assigned,
    shard_id = 0L,
    context = context,
    output_dir = layout_dir,
    setup_builder = fixture_setup_builder,
    target_builder = fixture_target_builder
  ),
  "Prepared-S shard manifest mismatch",
  "resume must reject a final pair from a different shard_count layout"
)
assert_true(
  builder_calls$setup == 0L && builder_calls$target == 0L,
  "layout-mismatched resume must fail before any builder call"
)
layout_summary <- jsonlite::read_json(
  layout_written$paths$summary_json, simplifyVector = TRUE
)
assert_error(
  fastkpc_full_cuda_validate_prepared_s_shard(
    payload = readRDS(layout_written$paths$rds),
    summary = layout_summary,
    inputs = fixture_inputs,
    shard_count = 3L,
    shard_id = 0L,
    rds_path = layout_written$paths$rds
  ),
  "Prepared-S shard manifest mismatch",
  paste(
    "public validator must bind a canonical two-shard payload to the",
    "caller's trusted three-shard layout"
  )
)

misplaced_dir <- tempfile("prepared-s-misplaced-shard-")
dir.create(misplaced_dir, recursive = TRUE)
misplaced_paths <- fastkpc_full_cuda_prepared_s_shard_paths(
  misplaced_dir, 1L
)
assert_true(
  file.copy(first$paths$rds, misplaced_paths$rds) &&
    file.copy(first$paths$summary_json, misplaced_paths$summary_json),
  "test fixture must place shard zero bytes at shard one final paths"
)
builder_calls$setup <- 0L
builder_calls$target <- 0L
assert_error(
  .fastkpc_full_cuda_run_prepared_s_shard_core(
    inputs = fixture_inputs,
    assigned_setups = assigned,
    shard_id = 1L,
    context = context,
    output_dir = misplaced_dir,
    setup_builder = fixture_setup_builder,
    target_builder = fixture_target_builder
  ),
  "Prepared-S shard manifest mismatch",
  "resume must reject a shard pair published under the wrong shard filename"
)
assert_true(
  builder_calls$setup == 0L && builder_calls$target == 0L,
  "misplaced shard resume must fail before any builder call"
)
assert_error(
  fastkpc_full_cuda_prepared_s_read_shard(
    shard_dir = misplaced_dir,
    inputs = fixture_inputs,
    shard_count = 3L,
    shard_id = 1L
  ),
  "Prepared-S shard manifest mismatch",
  "public reader must bind trusted shard identity to its canonical path"
)

pristine_rds_backup <- tempfile("prepared-s-pristine-rds-")
pristine_json_backup <- tempfile("prepared-s-pristine-json-")
assert_true(
  file.copy(first$paths$rds, pristine_rds_backup, overwrite = TRUE) &&
    file.copy(first$paths$summary_json, pristine_json_backup,
              overwrite = TRUE),
  "test fixture must preserve a pristine authenticated shard pair"
)
restore_pristine_pair <- function() {
  assert_true(
    file.copy(pristine_rds_backup, first$paths$rds, overwrite = TRUE) &&
      file.copy(pristine_json_backup, first$paths$summary_json,
                overwrite = TRUE),
    "test fixture must restore the pristine authenticated shard pair"
  )
}

prelocked_dir <- tempfile("prepared-s-prelocked-")
dir.create(prelocked_dir, recursive = TRUE)
prelocked_paths <- fastkpc_full_cuda_prepared_s_shard_paths(
  prelocked_dir, 0L
)
assert_true(
  file.copy(first$paths$rds, prelocked_paths$rds) &&
    file.copy(first$paths$summary_json, prelocked_paths$summary_json),
  "test fixture must install a valid final pair beside a stale lock"
)
prelocked_rds_hash <- fastkpc_full_cuda_census_file_hash(
  prelocked_paths$rds
)
prelocked_json_hash <- fastkpc_full_cuda_census_file_hash(
  prelocked_paths$summary_json
)
prelocked_lock <- file.path(prelocked_dir, "shard_0.lock")
assert_true(
  dir.create(prelocked_lock),
  "test fixture must create a pre-existing shard lock"
)
builder_calls$setup <- 0L
builder_calls$target <- 0L
assert_error(
  .fastkpc_full_cuda_run_prepared_s_shard_core(
    inputs = fixture_inputs,
    assigned_setups = assigned,
    shard_id = 0L,
    context = context,
    output_dir = prelocked_dir,
    setup_builder = fixture_setup_builder,
    target_builder = fixture_target_builder
  ),
  "Prepared-S shard lock exists",
  "pre-existing shard locks must fail closed before resume validation"
)
assert_true(
  builder_calls$setup == 0L && builder_calls$target == 0L &&
    identical(
      prelocked_rds_hash,
      fastkpc_full_cuda_census_file_hash(prelocked_paths$rds)
    ) &&
    identical(
      prelocked_json_hash,
      fastkpc_full_cuda_census_file_hash(prelocked_paths$summary_json)
    ),
  "lock failure must not build, delete, or replace an existing final pair"
)
unlink(prelocked_lock, recursive = TRUE)

malformed_payload_authentication <- function(payload) {
  setup_keys <- as.character(payload$ordered_setup_keys)
  setups <- payload$prepared_s_setups
  setup_hashes <- if (length(setup_keys) == 0L) {
    character()
  } else {
    setNames(vapply(
      setups,
      fastkpc_full_cuda_prepared_s_normalized_setup_hash,
      character(1L)
    ), setup_keys)
  }
  target_keys <- as.character(
    payload$target_states$residual_key_sha256
  )
  manifest_hash <- fastkpc_full_cuda_census_named_metadata_hash(
    payload$manifest
  )
  setup_key_set_hash <- fastkpc_full_cuda_census_key_set_hash(
    setup_keys
  )
  target_key_set_hash <- fastkpc_full_cuda_census_key_set_hash(
    sort(target_keys, method = "radix")
  )
  prepared_s_setup_hashes_hash <-
    fastkpc_full_cuda_census_named_metadata_hash(as.list(setup_hashes))
  target_state_frame_hash <- fastkpc_full_cuda_census_frame_hash(
    payload$target_states
  )
  payload_hash <- fastkpc_full_cuda_census_named_metadata_hash(list(
    schema_version = "full-cuda-ci-prepared-s-shard-payload-hash-v1",
    manifest_hash = manifest_hash,
    setup_key_set_hash = setup_key_set_hash,
    target_key_set_hash = target_key_set_hash,
    prepared_s_setup_hashes = as.list(setup_hashes),
    prepared_s_setup_hashes_hash = prepared_s_setup_hashes_hash,
    target_state_frame_hash = target_state_frame_hash
  ))
  list(
    manifest_hash = manifest_hash,
    setup_key_set_hash = setup_key_set_hash,
    target_key_set_hash = target_key_set_hash,
    prepared_s_setup_hashes = setup_hashes,
    prepared_s_setup_hashes_hash = prepared_s_setup_hashes_hash,
    target_state_frame_hash = target_state_frame_hash,
    payload_hash = payload_hash
  )
}
install_payload_attack <- function(payload) {
  saveRDS(payload, first$paths$rds, version = 2)
  summary <- jsonlite::read_json(
    pristine_json_backup, simplifyVector = TRUE
  )
  authentication <- malformed_payload_authentication(payload)
  for (field in setdiff(names(authentication), "prepared_s_setup_hashes")) {
    summary[[field]] <- authentication[[field]]
  }
  summary$prepared_s_setup_hashes <-
    as.list(authentication$prepared_s_setup_hashes)
  summary$rds_file_sha256 <-
    fastkpc_full_cuda_census_file_hash(first$paths$rds)
  summary$completion_hash <-
    fastkpc_full_cuda_prepared_s_completion_hash(summary)
  fastkpc_full_cuda_write_json(summary, first$paths$summary_json)
}
assert_resume_rejects <- function(pattern, message) {
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
    pattern,
    message
  )
}

payload_pairlist_attack <- as.pairlist(readRDS(pristine_rds_backup))
install_payload_attack(payload_pairlist_attack)
assert_resume_rejects(
  "Prepared-S shard payload schema mismatch",
  "payload root must be an exact plain list rather than a pairlist"
)
restore_pristine_pair()

setup_pairlist_attack <- readRDS(pristine_rds_backup)
setup_pairlist_attack$prepared_s_setups <- as.pairlist(
  setup_pairlist_attack$prepared_s_setups
)
install_payload_attack(setup_pairlist_attack)
assert_resume_rejects(
  "Prepared-S shard setup object order mismatch",
  "prepared_s_setups must be an exact named plain list"
)
restore_pristine_pair()

setup_root_pairlist_attack <- readRDS(pristine_rds_backup)
setup_root_pairlist_attack$prepared_s_setups[[1L]] <- as.pairlist(
  setup_root_pairlist_attack$prepared_s_setups[[1L]]
)
install_payload_attack(setup_root_pairlist_attack)
assert_resume_rejects(
  "PreparedSSetup contains non-plain data",
  paste(
    "resume must reject a pairlist PreparedSSetup root after all",
    "payload and file authentication is recomputed"
  )
)
pairlist_setup_summary <- jsonlite::read_json(
  first$paths$summary_json, simplifyVector = TRUE
)
assert_error(
  fastkpc_full_cuda_validate_prepared_s_shard(
    payload = readRDS(first$paths$rds),
    summary = pairlist_setup_summary,
    inputs = fixture_inputs,
    shard_count = 3L,
    shard_id = 0L,
    rds_path = first$paths$rds
  ),
  "PreparedSSetup contains non-plain data",
  "public shard validator must reject a pairlist PreparedSSetup root"
)
restore_pristine_pair()

payload_response_attack <- readRDS(pristine_rds_backup)
attr(payload_response_attack, "nested") <- list(
  y = fixture_inputs$data[, 1L]
)
install_payload_attack(payload_response_attack)
assert_resume_rejects(
  "Prepared-S shard artifact response-bearing field",
  paste(
    "artifact-scope validation must reject response-bearing payload",
    "attributes after all hashes are recomputed"
  )
)
restore_pristine_pair()

payload_attribute_attack <- readRDS(pristine_rds_backup)
attr(payload_attribute_attack, "note") <- "unallowed"
install_payload_attack(payload_attribute_attack)
assert_resume_rejects(
  "Prepared-S shard payload schema mismatch",
  "payload attributes outside the exact allowlist must fail closed"
)
restore_pristine_pair()

target_state_attribute_attack <- readRDS(pristine_rds_backup)
attr(target_state_attribute_attack$target_states, "note") <- "unallowed"
install_payload_attack(target_state_attribute_attack)
assert_resume_rejects(
  "Prepared-S shard TargetState schema mismatch",
  paste(
    "TargetState must reject extra frame attributes without applying",
    "the PreparedSSetup forbidden-name policy to canonical fields"
  )
)
restore_pristine_pair()

install_summary_attack <- function(summary) {
  fastkpc_full_cuda_write_json(summary, first$paths$summary_json)
}
malformed_summary_completion_hash <- function(summary) {
  canonical <- list(
    status = as.character(summary$status),
    manifest = summary$manifest,
    manifest_hash = as.character(summary$manifest_hash),
    setup_key_count = as.integer(summary$setup_key_count),
    setup_key_set_hash = as.character(summary$setup_key_set_hash),
    target_key_count = as.integer(summary$target_key_count),
    target_key_set_hash = as.character(summary$target_key_set_hash),
    prepared_s_setup_hashes = as.list(
      fastkpc_full_cuda_prepared_s_summary_setup_hashes(
        summary$prepared_s_setup_hashes
      )
    ),
    prepared_s_setup_hashes_hash =
      as.character(summary$prepared_s_setup_hashes_hash),
    target_state_frame_hash =
      as.character(summary$target_state_frame_hash),
    payload_hash = as.character(summary$payload_hash),
    build_elapsed_nanoseconds = round(
      as.numeric(summary$build_elapsed_seconds) * 1e9
    ),
    rds_file_sha256 = as.character(summary$rds_file_sha256)
  )
  fastkpc_full_cuda_census_named_metadata_hash(list(
    schema_version = "full-cuda-ci-prepared-s-shard-completion-hash-v1",
    summary = canonical
  ))
}
clean_summary <- jsonlite::read_json(
  pristine_json_backup, simplifyVector = TRUE
)

summary_character_count <- clean_summary
summary_character_count$setup_key_count <- as.character(
  summary_character_count$setup_key_count
)
summary_character_count$completion_hash <-
  malformed_summary_completion_hash(summary_character_count)
install_summary_attack(summary_character_count)
assert_resume_rejects(
  "Prepared-S shard completion field schema mismatch",
  paste(
    "completion summary must reject a character count even when the",
    "completion hash is recomputed"
  )
)
restore_pristine_pair()
assert_error(
  fastkpc_full_cuda_prepared_s_completion_hash(
    summary_character_count
  ),
  "Prepared-S shard completion field schema mismatch",
  "completion hash helper must validate raw field types before coercion"
)

summary_response_attack <- clean_summary
summary_response_attack$nested <- list(y = fixture_inputs$data[, 1L])
install_summary_attack(summary_response_attack)
assert_resume_rejects(
  "Prepared-S shard artifact response-bearing field",
  "completion JSON response leakage must fail with an unchanged hash"
)
restore_pristine_pair()

summary_response_rehashed <- clean_summary
summary_response_rehashed$nested <- list(y = fixture_inputs$data[, 1L])
summary_response_rehashed$completion_hash <- tryCatch(
  fastkpc_full_cuda_prepared_s_completion_hash(summary_response_rehashed),
  error = function(error) clean_summary$completion_hash
)
install_summary_attack(summary_response_rehashed)
assert_resume_rejects(
  "Prepared-S shard artifact response-bearing field",
  "completion JSON response leakage must fail after hash recomputation"
)
restore_pristine_pair()

summary_extra_attack <- clean_summary
summary_extra_attack$note <- "unallowed"
install_summary_attack(summary_extra_attack)
assert_resume_rejects(
  "Prepared-S shard summary schema mismatch",
  "completion JSON fields outside the exact allowlist must fail closed"
)
restore_pristine_pair()

summary_timing_attack <- clean_summary
summary_timing_attack$build_elapsed_seconds <-
  summary_timing_attack$build_elapsed_seconds + 1
install_summary_attack(summary_timing_attack)
assert_resume_rejects(
  "Prepared-S shard completion hash mismatch",
  "completion hash must authenticate build timing"
)
restore_pristine_pair()

summary_order_attack <- clean_summary[rev(seq_along(clean_summary))]
install_summary_attack(summary_order_attack)
assert_resume_rejects(
  "Prepared-S shard summary schema mismatch",
  "completion JSON field order must be canonical after JSON round-trip"
)
restore_pristine_pair()

summary_attribute_attack <- clean_summary
attr(summary_attribute_attack, "note") <- "unallowed"
assert_error(
  fastkpc_full_cuda_validate_prepared_s_shard(
    payload = readRDS(pristine_rds_backup),
    summary = summary_attribute_attack,
    inputs = fixture_inputs,
    shard_count = 3L,
    shard_id = 0L,
    rds_path = pristine_rds_backup
  ),
  "Prepared-S shard summary schema mismatch",
  "in-memory completion summary attributes must fail closed"
)

summary_hash_attribute_attack <- clean_summary
summary_hash_attribute_attack$payload_hash <- structure(
  summary_hash_attribute_attack$payload_hash,
  note = "unallowed"
)
summary_hash_attribute_attack$completion_hash <-
  malformed_summary_completion_hash(summary_hash_attribute_attack)
assert_error(
  fastkpc_full_cuda_validate_prepared_s_shard(
    payload = readRDS(pristine_rds_backup),
    summary = summary_hash_attribute_attack,
    inputs = fixture_inputs,
    shard_count = 3L,
    shard_id = 0L,
    rds_path = pristine_rds_backup
  ),
  "Prepared-S shard completion field schema mismatch",
  "completion summary hash scalars with attributes must fail closed"
)

validator_formals <- names(formals(
  fastkpc_full_cuda_validate_prepared_s_shard
))
assert_true(
  identical(
    validator_formals,
    c(
      "payload", "summary", "inputs", "shard_count", "shard_id",
      "rds_path"
    )
  ) &&
    length(intersect(
      validator_formals,
      c("expected_manifest", forbidden_public_formals)
    )) == 0L,
  paste(
    "public shard validator must require trusted layout scalars without",
    "expected-manifest or hash authority injection"
  )
)
reader_formals <- names(formals(
  fastkpc_full_cuda_prepared_s_read_shard
))
batch_reader_formals <- names(formals(
  fastkpc_full_cuda_prepared_s_read_shards
))
selected_reader_formals <- names(formals(
  fastkpc_full_cuda_prepared_s_read_selected_shards
))
writer_formals <- names(formals(
  fastkpc_full_cuda_run_prepared_s_shard
))
context_validator_formals <- names(formals(
  fastkpc_full_cuda_prepared_s_validate_output_shard_context
))
manifest_constructor_formals <- names(formals(
  fastkpc_full_cuda_prepared_s_shard_manifest
))
assert_true(
  identical(
    reader_formals,
    c(
      "shard_dir", "inputs", "shard_count", "shard_id",
      "expected_source_commit"
    )
  ) &&
    length(intersect(
      reader_formals,
      c("paths", "expected_manifest", forbidden_public_formals)
    )) == 0L,
  "public shard reader must accept only canonical directory and layout"
)
assert_true(
  identical(
    batch_reader_formals,
    c(
      "shard_dir", "inputs", "shard_count", "shard_ids",
      "expected_source_commit"
    )
  ) &&
    length(intersect(
      batch_reader_formals,
      c("paths", "expected_manifest", forbidden_public_formals)
    )) == 0L &&
    !"expected_source_commit" %in% writer_formals &&
    length(intersect(
      writer_formals,
      c("context", "assigned_setups", "expected_manifest",
        "target_state_index", "target_state_context", "setup_keys",
        "target_keys", "shard_ids")
    )) == 0L,
  paste(
    "only public readers may authorize historical source commits or",
    "accept generic prepared-shard context overrides"
  )
)
assert_true(
  identical(
    selected_reader_formals,
    c(
      "shard_dir", "inputs", "shard_count", "shard_ids", "setup_keys",
      "target_keys", "expected_source_commit"
    )
  ) &&
    length(intersect(
      selected_reader_formals,
      c("paths", "expected_manifest", forbidden_public_formals)
    )) == 0L,
  paste(
    "selected shard reader must accept canonical layout and selection keys",
    "without caller-supplied authentication authority"
  )
)
assert_true(
  identical(
    context_validator_formals,
    c("inputs", "context", "assigned_setups")
  ) &&
    identical(
      manifest_constructor_formals,
      c("assigned_setups", "shard_id", "inputs", "context")
    ) &&
    !"expected_source_commit" %in% c(
      context_validator_formals, manifest_constructor_formals
    ) &&
    length(intersect(
      c(context_validator_formals, manifest_constructor_formals),
      c("setup_keys", "target_keys", "shard_ids", "expected_manifest")
    )) == 0L,
  paste(
    "historical replay and selected payload authority must remain on public",
    "readers rather than generic context or manifest helpers"
  )
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
  file.exists(first$paths$rds) && !file.exists(first$paths$summary_json) &&
    !file.exists(first$paths$lock_dir),
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

stale_output_dir <- tempfile("prepared-s-stale-public-validator-")
stale_written <- .fastkpc_full_cuda_run_prepared_s_shard_core(
  inputs = stale_inputs,
  assigned_setups = assigned,
  shard_id = 0L,
  context = stale_context,
  output_dir = stale_output_dir,
  setup_builder = fixture_setup_builder,
  target_builder = fixture_target_builder
)
stale_summary <- jsonlite::read_json(
  stale_written$paths$summary_json, simplifyVector = TRUE
)
assert_error(
  fastkpc_full_cuda_validate_prepared_s_shard(
    payload = readRDS(stale_written$paths$rds),
    summary = stale_summary,
    inputs = fixture_inputs,
    shard_count = 3L,
    shard_id = 0L,
    rds_path = stale_written$paths$rds
  ),
  "Prepared-S shard manifest mismatch",
  paste(
    "public validation must reject an internally valid shard whose manifest",
    "was derived from stale inputs"
  )
)

invalid_shard_counts <- list(
  factor = factor("3"),
  logical = TRUE,
  character = "3",
  list = list(3L),
  fractional = 1.5,
  vector = c(1L, 2L),
  missing = NA_integer_,
  infinite = Inf,
  matrix = matrix(3, 1L, 1L),
  named = structure(3, names = "n")
)
for (label in names(invalid_shard_counts)) {
  assert_error(
    fastkpc_full_cuda_prepared_s_assign_shards(
      setup_index, invalid_shard_counts[[label]]
    ),
    "shard_count must be one positive integer",
    paste("shard_count must reject", label, "input before coercion")
  )
}
assigned_double_count <- fastkpc_full_cuda_prepared_s_assign_shards(
  setup_index, 3.0
)
assert_true(
  identical(attr(assigned_double_count, "shard_count"), 3L),
  "whole scalar double shard_count must remain valid"
)

invalid_shard_ids <- list(
  factor = factor("3"),
  logical = TRUE,
  character = "1",
  list = list(1L),
  fractional = 1.5,
  vector = c(1L, 2L),
  missing = NA_integer_,
  infinite = Inf,
  matrix = matrix(1, 1L, 1L),
  named = structure(1, names = "n")
)
for (label in names(invalid_shard_ids)) {
  assert_error(
    fastkpc_full_cuda_prepared_s_shard_manifest(
      assigned, invalid_shard_ids[[label]], fixture_inputs, context
    ),
    "shard_id",
    paste("shard_id must reject", label, "input before coercion")
  )
}
double_id_manifest <- fastkpc_full_cuda_prepared_s_shard_manifest(
  assigned, 1.0, fixture_inputs, context
)
assert_true(
  identical(double_id_manifest$shard_id, 1L),
  "whole scalar double shard_id must remain valid and zero based"
)
character_assignment <- assigned
character_assignment$shard_id <- as.character(
  character_assignment$shard_id
)
assert_error(
  fastkpc_full_cuda_prepared_s_shard_manifest(
    character_assignment, 0L, fixture_inputs, context
  ),
  "Prepared-S shard assignment lineage mismatch",
  "assigned shard_id columns must not be accepted through coercion"
)

builder_calls$setup <- 0L
builder_calls$target <- 0L
empty_dir <- tempfile("prepared-s-empty-shard-")
empty_run <- .fastkpc_full_cuda_run_prepared_s_shard_core(
  inputs = fixture_inputs,
  assigned_setups = sparse_assigned,
  shard_id = 6L,
  context = context,
  output_dir = empty_dir,
  setup_builder = fixture_setup_builder,
  target_builder = fixture_target_builder
)
empty_reused <- .fastkpc_full_cuda_run_prepared_s_shard_core(
  inputs = fixture_inputs,
  assigned_setups = sparse_assigned,
  shard_id = 6L,
  context = context,
  output_dir = empty_dir,
  setup_builder = fixture_setup_builder,
  target_builder = fixture_target_builder
)
assert_true(
  empty_run$status == "written" && empty_reused$status == "reused" &&
    builder_calls$setup == 0L && builder_calls$target == 0L &&
    length(empty_run$payload$ordered_setup_keys) == 0L &&
    nrow(empty_run$payload$target_states) == 0L,
  "empty trailing shard must publish without invoking builders"
)

merge_dir <- tempfile("prepared-s-merge-")
for (shard_id in 0:6) {
  .fastkpc_full_cuda_run_prepared_s_shard_core(
    inputs = fixture_inputs,
    assigned_setups = sparse_assigned,
    shard_id = shard_id,
    context = context,
    output_dir = merge_dir,
    setup_builder = fixture_setup_builder,
    target_builder = fixture_target_builder
  )
}
merge_lock <- file.path(merge_dir, "shard_1.lock")
assert_true(
  dir.create(merge_lock),
  "test fixture must create an active merge shard lock"
)
assert_error(
  fastkpc_full_cuda_merge_prepared_s_shards(
    inputs = fixture_inputs,
    shard_count = 7L,
    shard_dir = merge_dir
  ),
  "Prepared-S shard lock exists",
  "Prepared-S merge must fail closed while any shard lock exists"
)
unlink(merge_lock, recursive = TRUE)
merged <- fastkpc_full_cuda_merge_prepared_s_shards(
  inputs = fixture_inputs,
  shard_count = 7L,
  shard_dir = merge_dir
)
expected_merged_targets <-
  fastkpc_full_cuda_prepared_s_target_rows_for_setups(
    fixture_inputs, sparse_assigned
  )
merged_target_keys_by_setup <- split(
  as.character(merged$target_states$residual_key_sha256),
  as.character(merged$target_states$prepared_s_key_sha256)
)
assert_true(
  identical(names(merged$prepared_s_setups), sorted_setup_keys) &&
    identical(
      as.character(merged$target_states$prepared_s_key_sha256),
      as.character(expected_merged_targets$prepared_s_key_sha256)
    ) &&
    identical(
      as.character(merged$target_states$residual_key_sha256),
      as.character(expected_merged_targets$residual_key_sha256)
    ) &&
    nrow(merged$target_states) == 7L &&
    sum(vapply(merged_target_keys_by_setup, length, integer(1L)) == 2L) ==
      1L &&
    all(vapply(merged_target_keys_by_setup, function(keys) {
      identical(keys, sort(keys, method = "radix"))
    }, logical(1L))),
  "Prepared-S merge must restore exact canonical setup and target order"
)

index_builder_calls$merge <- 0L
counting_merge_index_builder <- function(inputs) {
  index_builder_calls$merge <- index_builder_calls$merge + 1L
  .fastkpc_full_cuda_target_state_input_index(inputs)
}
indexed_merged <- .fastkpc_full_cuda_merge_prepared_s_shards_core(
  inputs = fixture_inputs,
  shard_count = 7L,
  shard_dir = merge_dir,
  target_state_index_builder = counting_merge_index_builder
)
assert_true(
  index_builder_calls$merge == 1L &&
    identical(
      fastkpc_full_cuda_census_frame_hash(indexed_merged$target_states),
      fastkpc_full_cuda_census_frame_hash(merged$target_states)
    ),
  "Prepared-S merge must build one index and preserve the target frame"
)

missing_paths <- fastkpc_full_cuda_prepared_s_shard_paths(merge_dir, 2L)
missing_backup <- tempfile("prepared-s-merge-missing-")
assert_true(
  file.rename(missing_paths$summary_json, missing_backup),
  "test fixture must hide one merge completion JSON"
)
assert_error(
  fastkpc_full_cuda_merge_prepared_s_shards(
    fixture_inputs, 7L, merge_dir
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
    fixture_inputs, 7L, merge_dir
  ),
  "Prepared-S shard manifest mismatch",
  "Prepared-S merge must bind each canonical path to its trusted shard id"
)
assert_true(
  file.copy(one_rds_backup, shard_one$rds, overwrite = TRUE) &&
    file.copy(one_json_backup, shard_one$summary_json, overwrite = TRUE),
  "test fixture must restore duplicate merge shard"
)

cat("PASS full CUDA CI Prepared-S restart qualification\n")
