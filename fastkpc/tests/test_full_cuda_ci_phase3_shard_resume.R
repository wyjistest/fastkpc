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
assert_error <- function(expression, message, pattern = NULL) {
  error <- tryCatch({
    force(expression)
    NULL
  }, error = function(error) error)
  clean <- inherits(error, "error") && (
    is.null(pattern) || grepl(pattern, conditionMessage(error), fixed = TRUE)
  )
  assert_true(clean, message)
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
read_json <- function(path) {
  jsonlite::read_json(path, simplifyVector = TRUE)
}
write_json <- function(value, path) {
  fastkpc_full_cuda_write_json(value, path)
}
copy_tree <- function(from, to) {
  dir.create(to, recursive = TRUE, showWarnings = FALSE)
  entries <- list.files(
    from, all.files = TRUE, no.. = TRUE, full.names = TRUE
  )
  for (entry in entries) {
    destination <- file.path(to, basename(entry))
    if (dir.exists(entry)) {
      copy_tree(entry, destination)
    } else if (!file.copy(entry, destination, overwrite = FALSE)) {
      fail(paste("failed to copy fixture path", entry))
    }
  }
  invisible(to)
}

original_test_shard_count <- Sys.getenv(
  "FASTKPC_FULL_CUDA_PHASE3_TEST_SHARD_COUNT", unset = NA_character_
)
on.exit({
  if (is.na(original_test_shard_count)) {
    Sys.unsetenv("FASTKPC_FULL_CUDA_PHASE3_TEST_SHARD_COUNT")
  } else {
    Sys.setenv(
      FASTKPC_FULL_CUDA_PHASE3_TEST_SHARD_COUNT =
        original_test_shard_count
    )
  }
}, add = TRUE)
Sys.unsetenv("FASTKPC_FULL_CUDA_PHASE3_TEST_SHARD_COUNT")

setup_keys <- vapply(
  sprintf("synthetic-setup-%02d", 1:8), sha, character(1L)
)
setup_input <- c(setup_keys[c(8L, 2L, 5L, 1L, 7L, 3L, 6L, 4L)],
                 setup_keys[[2L]])
assigned <- fastkpc_full_cuda_phase3_assign_setup_shards(
  setup_input, 4L
)
expected_setup_keys <- sort(unique(setup_input), method = "radix")
assert_identical(
  names(assigned),
  c("prepared_s_key_sha256", "sorted_rank", "shard_id"),
  "Phase 3 setup assignment schema is exact"
)
assert_identical(
  assigned$prepared_s_key_sha256, expected_setup_keys,
  "Phase 3 setup assignment uses unique radix PreparedSKey order"
)
assert_identical(
  assigned$sorted_rank, seq_along(expected_setup_keys),
  "Phase 3 setup assignment records canonical one-based rank"
)
assert_identical(
  assigned$shard_id,
  as.integer((seq_along(expected_setup_keys) - 1L) %% 4L),
  "Phase 3 setup assignment uses zero-based rank modulo shard count"
)
assert_error(
  fastkpc_full_cuda_phase3_assign_setup_shards(character(), 4L),
  "empty setup assignment must fail closed"
)
assert_error(
  fastkpc_full_cuda_phase3_assign_setup_shards(toupper(setup_keys), 4L),
  "noncanonical PreparedSKey assignment must fail closed"
)
assert_error(
  fastkpc_full_cuda_phase3_assign_setup_shards(setup_keys, 0L),
  "nonpositive shard count must fail closed"
)

assert_identical(
  fastkpc_full_cuda_phase3_resolve_shard_count("full"), 64L,
  "full Phase 3 scope hardcodes 64 shards"
)
assert_error(
  fastkpc_full_cuda_phase3_resolve_shard_count("full", 64L),
  "full Phase 3 scope rejects even an equal explicit override",
  "full"
)
Sys.setenv(FASTKPC_FULL_CUDA_PHASE3_TEST_SHARD_COUNT = "4")
assert_error(
  fastkpc_full_cuda_phase3_resolve_shard_count("full"),
  "full Phase 3 scope rejects the test shard environment override",
  "full"
)
assert_identical(
  fastkpc_full_cuda_phase3_resolve_shard_count("qualification"), 4L,
  "qualification scope may use the test shard environment override"
)
Sys.setenv(FASTKPC_FULL_CUDA_PHASE3_TEST_SHARD_COUNT = "")
assert_error(
  fastkpc_full_cuda_phase3_resolve_shard_count("full"),
  "full Phase 3 scope rejects an explicitly empty test shard variable",
  "full"
)
Sys.unsetenv("FASTKPC_FULL_CUDA_PHASE3_TEST_SHARD_COUNT")
assert_identical(
  fastkpc_full_cuda_phase3_resolve_shard_count("iteration", 4L), 4L,
  "iteration scope may use an explicit shard count"
)

target_rows <- do.call(rbind, lapply(seq_along(setup_keys), function(index) {
  data.frame(
    prepared_s_key_sha256 = rep(setup_keys[[index]], 2L),
    residual_key_sha256 = vapply(
      sprintf("synthetic-target-%02d-%02d", index, 1:2),
      sha, character(1L)
    ),
    target_ordinal = as.integer(1:2),
    stringsAsFactors = FALSE
  )
}))
target_rows <- target_rows[rev(seq_len(nrow(target_rows))), , drop = FALSE]
rownames(target_rows) <- NULL

route_fixture <- function(mode = "route-a", shard_count = 4L) {
  refresh_hash(list(
    schema_version = "full-cuda-ci-phase3-test-route-v1",
    mode = mode,
    shard_count = as.integer(shard_count)
  ))
}
identity_fixture <- function(
    route, targets = target_rows, setups = setup_keys,
    gpu_name = "Synthetic GPU",
    gpu_uuid = paste0("GPU-", strrep("a", 32L))) {
  refresh_hash(list(
    schema_version = "full-cuda-ci-phase3-test-input-identity-v1",
    canonical_setup_corpus_hash = key_hash(setups),
    canonical_target_corpus_hash = key_hash(targets$residual_key_sha256),
    route_config_hash = route$sha256,
    source_commit = strrep("1", 40L),
    cuda_toolkit_version = 12040L,
    cuda_driver_version = 55054L,
    gpu_name = gpu_name,
    gpu_uuid = gpu_uuid,
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
}

route <- route_fixture()
identity <- identity_fixture(route)
synthetic_plan <- fastkpc_full_cuda_phase3_plan_shards(
  setup_keys = setup_input,
  target_rows = target_rows,
  scope = "qualification",
  shard_count = 4L
)
assert_identical(
  synthetic_plan$assignments, assigned,
  "synthetic shard plan uses the deterministic setup assignment"
)
assert_true(
  nrow(synthetic_plan$target_rows) == 16L &&
    !anyNA(synthetic_plan$target_rows$shard_id) &&
    all(synthetic_plan$target_rows$shard_id %in% 0:3),
  "every synthetic target inherits exactly one setup shard"
)
assert_identical(
  key_hash(synthetic_plan$target_rows$residual_key_sha256),
  identity$canonical_target_corpus_hash,
  "synthetic shard plan preserves the target corpus hash"
)
bad_targets <- target_rows
bad_targets$prepared_s_key_sha256[[1L]] <- sha("unknown-setup")
assert_error(
  fastkpc_full_cuda_phase3_plan_shards(
    setup_keys, bad_targets, scope = "qualification", shard_count = 4L
  ),
  "targets without one setup assignment must fail closed"
)

session_fields <- c(
  "schema_version", "session_id", "input_identity_hash",
  "route_config_hash", "requested_shard_ids", "completed_shard_ids",
  "runtime_context_create_count", "runtime_context_destroy_count",
  "prepared_handle_create_count", "prepared_handle_destroy_count",
  "residual_token_acquire_count", "residual_token_release_count",
  "output_slot_acquire_count", "output_slot_release_count",
  "target_level_stable_sync_count", "status"
)
resource_fields <- c(
  "prepared_handle_create_count", "prepared_handle_destroy_count",
  "residual_token_acquire_count", "residual_token_release_count",
  "output_slot_acquire_count", "output_slot_release_count"
)

fake_executor <- function(context, shard_id, setup_keys, target_rows) {
  context$executor_calls <- context$executor_calls + 1L
  setup_batch_count <- as.integer(length(setup_keys))
  setup_result <- data.frame(
    prepared_s_key_sha256 = setup_keys,
    shard_id = rep.int(as.integer(shard_id), length(setup_keys)),
    stringsAsFactors = FALSE
  )
  target_result <- target_rows[, c(
    "prepared_s_key_sha256", "residual_key_sha256", "target_ordinal"
  ), drop = FALSE]
  target_result$shard_id <- rep.int(
    as.integer(shard_id), nrow(target_result)
  )
  counts <- list(
    prepared_handle_create_count = setup_batch_count,
    prepared_handle_destroy_count = setup_batch_count,
    residual_token_acquire_count = setup_batch_count,
    residual_token_release_count = setup_batch_count,
    output_slot_acquire_count = setup_batch_count,
    output_slot_release_count = setup_batch_count
  )
  assert_identical(
    names(counts), resource_fields,
    "fake executor resource schema matches the Phase 3 runner contract"
  )
  list(
    payload = list(
      setup_results = setup_result,
      target_results = target_result
    ),
    resource_counts = counts
  )
}

run_fixture <- function(
    output_dir, identity_value = identity, route_value = route,
    targets = target_rows, setup_values = setup_keys,
    shard_count_value = 4L, stop_after = NULL,
    kind_value = "full_shadow") {
  lifecycle <- new.env(parent = emptyenv())
  lifecycle$create_count <- 0L
  lifecycle$destroy_count <- 0L
  lifecycle$progress <- list()
  runtime_create <- function() {
    lifecycle$create_count <- lifecycle$create_count + 1L
    context <- new.env(parent = emptyenv())
    context$executor_calls <- 0L
    context
  }
  runtime_destroy <- function(context) {
    assert_true(is.environment(context), "runtime destroy receives context")
    lifecycle$destroy_count <- lifecycle$destroy_count + 1L
    invisible(NULL)
  }
  progress_hook <- function(session, event) {
    lifecycle$progress[[length(lifecycle$progress) + 1L]] <- list(
      session = session, event = event
    )
    invisible(NULL)
  }
  result <- fastkpc_full_cuda_phase3_run_shards(
    output_dir = output_dir,
    kind = kind_value,
    setup_keys = setup_values,
    target_rows = targets,
    identity = identity_value,
    route_config = route_value,
    executor = fake_executor,
    runtime_create = runtime_create,
    runtime_destroy = runtime_destroy,
    scope = "qualification",
    shard_count = shard_count_value,
    stop_after = stop_after,
    progress_hook = progress_hook
  )
  list(result = result, lifecycle = lifecycle)
}

generic_oracle_dir <- tempfile("phase3-generic-oracle-rejected-")
on.exit(unlink(generic_oracle_dir, recursive = TRUE, force = TRUE), add = TRUE)
assert_error(
  run_fixture(generic_oracle_dir, stop_after = 1L,
              kind_value = "oracle_sp"),
  "oracle shard publication rejects a generic non-oracle payload",
  "oracle"
)

shared_source_setup_index <- data.frame(
  prepared_s_key_sha256 = sort(setup_keys[1:3], method = "radix"),
  stringsAsFactors = FALSE
)
shared_source_setup_keys <-
  shared_source_setup_index$prepared_s_key_sha256[c(1L, 3L)]
shared_source_targets <- target_rows[
  target_rows$prepared_s_key_sha256 %in% shared_source_setup_keys,
  , drop = FALSE
]
shared_source_targets <- shared_source_targets[order(
  match(
    shared_source_targets$prepared_s_key_sha256,
    shared_source_setup_keys
  ),
  shared_source_targets$residual_key_sha256,
  method = "radix"
), , drop = FALSE]
rownames(shared_source_targets) <- NULL
loader_evidence <- new.env(parent = emptyenv())
loader_evidence$call_count <- 0L
loader_evidence$shard_ids <- list()
fake_phase2_loader <- function(
    shard_dir, inputs, shard_count, shard_ids, setup_keys, target_keys,
    expected_source_commit) {
  loader_evidence$call_count <- loader_evidence$call_count + 1L
  loader_evidence$shard_ids[[loader_evidence$call_count]] <- shard_ids
  list(
    prepared_s_setups = stats::setNames(
      lapply(setup_keys, function(key) list(prepared_s_key_sha256 = key)),
      setup_keys
    ),
    target_states = data.frame(
      prepared_s_key_sha256 = shared_source_targets$prepared_s_key_sha256,
      residual_key_sha256 = shared_source_targets$residual_key_sha256,
      stringsAsFactors = FALSE
    ),
    shard_ids = as.integer(shard_ids)
  )
}
shared_source_catalog <- list(
  setup_index = shared_source_setup_index,
  catalog_contract = list(shard_count = 2L),
  phase2_dir = "/synthetic/phase2",
  inputs = list(synthetic = TRUE),
  phase2_manifest = list(source_commit = strrep("2", 40L))
)
shared_source_load <-
  fastkpc_full_cuda_fixed_sp_load_oracle_phase2_shards(
    catalog = shared_source_catalog,
    setup_keys = shared_source_setup_keys,
    target_rows = shared_source_targets,
    shard_loader = fake_phase2_loader
  )
assert_true(
  loader_evidence$call_count == 1L &&
    identical(loader_evidence$shard_ids[[1L]], 0L) &&
    identical(shared_source_load$phase2_shard_ids, 0L) &&
    shared_source_load$phase2_shard_load_count == 1L &&
    shared_source_load$phase2_shard_authentication_count == 1L,
  "setups sharing one Phase 2 shard perform one read/authentication"
)

interrupt_dir <- tempfile("phase3-interrupt-cleanup-")
on.exit(unlink(interrupt_dir, recursive = TRUE, force = TRUE), add = TRUE)
interrupt_lifecycle <- new.env(parent = emptyenv())
interrupt_lifecycle$create_count <- 0L
interrupt_lifecycle$destroy_count <- 0L
interrupt_runtime_create <- function() {
  interrupt_lifecycle$create_count <- interrupt_lifecycle$create_count + 1L
  new.env(parent = emptyenv())
}
interrupt_runtime_destroy <- function(context) {
  interrupt_lifecycle$destroy_count <- interrupt_lifecycle$destroy_count + 1L
  invisible(NULL)
}
synthetic_interrupt <- structure(
  list(message = "synthetic Phase 3 interrupt", call = NULL),
  class = c("interrupt", "condition")
)
interrupt_executor <- function(...) stop(synthetic_interrupt)
interrupt_condition <- tryCatch(
  fastkpc_full_cuda_phase3_run_shards(
    output_dir = interrupt_dir,
    kind = "full_shadow",
    setup_keys = single_setup <- unname(setup_keys[[1L]]),
    target_rows = target_rows[
      target_rows$prepared_s_key_sha256 == single_setup,
      , drop = FALSE
    ],
    identity = identity_fixture(
      route_fixture("interrupt", shard_count = 1L),
      targets = target_rows[
        target_rows$prepared_s_key_sha256 == single_setup,
        , drop = FALSE
      ],
      setups = single_setup
    ),
    route_config = route_fixture("interrupt", shard_count = 1L),
    executor = interrupt_executor,
    runtime_create = interrupt_runtime_create,
    runtime_destroy = interrupt_runtime_destroy,
    scope = "qualification",
    shard_count = 1L
  ),
  interrupt = function(condition) condition
)
assert_true(
  inherits(interrupt_condition, "interrupt") &&
    interrupt_lifecycle$create_count == 1L &&
    interrupt_lifecycle$destroy_count == 1L,
  "runtime context is destroyed exactly once on interrupt"
)

single_setup <- unname(setup_keys[[1L]])
single_target <- target_rows[
  target_rows$prepared_s_key_sha256 == single_setup,
  , drop = FALSE
][1L, , drop = FALSE]
sparse_route <- route_fixture("sparse-route", shard_count = 2L)
sparse_identity <- identity_fixture(
  sparse_route, targets = single_target, setups = single_setup
)
sparse_dir <- tempfile("phase3-sparse-shards-")
on.exit(unlink(sparse_dir, recursive = TRUE, force = TRUE), add = TRUE)
sparse_run <- run_fixture(
  sparse_dir,
  identity_value = sparse_identity,
  route_value = sparse_route,
  targets = single_target,
  setup_values = single_setup,
  shard_count_value = 2L
)
assert_identical(
  sparse_run$result$written_shard_ids, as.integer(0:1),
  "more shards than setups publishes the empty shard"
)
empty_shard_payload <- readRDS(file.path(
  sparse_dir, "shards", "shard_1.rds"
))
empty_shard_summary <- read_json(file.path(
  sparse_dir, "shards", "shard_1.summary.json"
))
assert_true(
  length(empty_shard_payload$expected_setup_keys) == 0L &&
    length(empty_shard_payload$expected_target_keys) == 0L &&
    as.integer(empty_shard_summary$expected_setup_count) == 0L &&
    as.integer(empty_shard_summary$expected_target_count) == 0L &&
    identical(
      empty_shard_summary$expected_setup_hash, key_hash(character())
    ) && identical(
      empty_shard_summary$expected_target_hash, key_hash(character())
    ),
  "empty shard pair binds exact zero counts and canonical empty hashes"
)
sparse_resume <- run_fixture(
  sparse_dir,
  identity_value = sparse_identity,
  route_value = sparse_route,
  targets = single_target,
  setup_values = single_setup,
  shard_count_value = 2L
)
assert_true(
  identical(sparse_resume$result$reused_shard_ids, as.integer(0:1)) &&
    identical(sparse_resume$result$written_shard_ids, integer()) &&
    sparse_resume$lifecycle$create_count == 0L,
  "pure resume reuses an authenticated empty shard without publication"
)

empty_targets <- single_target[0L, , drop = FALSE]
rownames(empty_targets) <- NULL
empty_target_route <- route_fixture("empty-target-route", shard_count = 2L)
empty_target_identity <- identity_fixture(
  empty_target_route, targets = empty_targets, setups = single_setup
)
empty_target_dir <- tempfile("phase3-empty-targets-")
on.exit(unlink(empty_target_dir, recursive = TRUE, force = TRUE), add = TRUE)
empty_target_run <- run_fixture(
  empty_target_dir,
  identity_value = empty_target_identity,
  route_value = empty_target_route,
  targets = empty_targets,
  setup_values = single_setup,
  shard_count_value = 2L
)
assert_identical(
  empty_target_run$result$written_shard_ids, as.integer(0:1),
  "an empty target corpus publishes valid zero-target shard descriptors"
)
for (shard_id in 0:1) {
  summary <- read_json(file.path(
    empty_target_dir, "shards",
    paste0("shard_", shard_id, ".summary.json")
  ))
  assert_true(
    as.integer(summary$expected_target_count) == 0L &&
      identical(summary$expected_target_hash, key_hash(character())),
    "every empty-target shard binds the canonical empty target hash"
  )
}

artifact_dir <- tempfile("phase3-shard-resume-")
on.exit(unlink(artifact_dir, recursive = TRUE, force = TRUE), add = TRUE)
first <- run_fixture(artifact_dir, stop_after = 2L)
assert_identical(
  first$result$status, "stopped",
  "first synthetic execution stops gracefully after its shard limit"
)
assert_identical(
  first$result$requested_shard_ids, as.integer(0:3),
  "first execution requests every missing shard in one session"
)
assert_identical(
  first$result$written_shard_ids, as.integer(0:1),
  "first execution writes exactly two deterministic shards"
)
assert_identical(
  first$result$reused_shard_ids, integer(),
  "first execution reuses no shards"
)
assert_true(
  first$lifecycle$create_count == 1L &&
    first$lifecycle$destroy_count == 1L,
  "graceful stop creates and destroys exactly one context"
)
assert_identical(
  vapply(first$lifecycle$progress, `[[`, character(1L), "event"),
  c("shard_complete", "shard_complete", "session_complete"),
  "running session is rewritten after each shard before final closure"
)
assert_identical(
  lapply(first$lifecycle$progress[1:2], function(value) {
    value$session$completed_shard_ids
  }),
  list(0L, as.integer(0:1)),
  "session progress snapshots record every completed shard"
)

session_files <- list.files(
  file.path(artifact_dir, "sessions"),
  pattern = "^session_[A-Za-z0-9_-]+\\.json$", full.names = TRUE
)
assert_true(length(session_files) == 1L, "first invocation writes one session")
first_session <- read_json(session_files[[1L]])
first_completed_shards <- as.integer(first_session$completed_shard_ids)
first_setup_count <- as.integer(sum(
  synthetic_plan$assignments$shard_id %in% first_completed_shards
))
first_target_count <- as.integer(sum(
  synthetic_plan$target_rows$shard_id %in% first_completed_shards
))
assert_identical(
  names(first_session), session_fields,
  "Phase 3 session JSON schema is exact"
)
assert_true(
  identical(first_session$schema_version,
            "full-cuda-ci-phase3-session-v1") &&
    identical(first_session$status, "complete") &&
    identical(as.integer(first_session$requested_shard_ids), 0:3) &&
    identical(as.integer(first_session$completed_shard_ids), 0:1) &&
    as.integer(first_session$runtime_context_create_count) == 1L &&
    as.integer(first_session$runtime_context_destroy_count) == 1L &&
    first_setup_count != first_target_count &&
    as.integer(first_session$prepared_handle_create_count) ==
      first_setup_count &&
    as.integer(first_session$prepared_handle_destroy_count) ==
      first_setup_count &&
    as.integer(first_session$residual_token_acquire_count) ==
      first_setup_count &&
    as.integer(first_session$residual_token_release_count) ==
      first_setup_count &&
    as.integer(first_session$output_slot_acquire_count) ==
      first_setup_count &&
    as.integer(first_session$output_slot_release_count) ==
      first_setup_count &&
    as.integer(first_session$target_level_stable_sync_count) == 0L,
  "gracefully stopped session conserves one resource per setup batch"
)
first_session_hash <- fastkpc_full_cuda_phase3_session_identity_hash(
  first_session
)
attributed_session <- first_session
attr(attributed_session, "forged") <- TRUE
assert_error(
  fastkpc_full_cuda_phase3_session_identity_hash(attributed_session),
  "session identity rejects non-schema attributes"
)
mutable_session <- first_session
mutable_session$completed_shard_ids <- integer()
mutable_session$runtime_context_create_count <- 99L
mutable_session$runtime_context_destroy_count <- 99L
mutable_session$prepared_handle_create_count <- 99L
mutable_session$prepared_handle_destroy_count <- 99L
mutable_session$residual_token_acquire_count <- 99L
mutable_session$residual_token_release_count <- 99L
mutable_session$output_slot_acquire_count <- 99L
mutable_session$output_slot_release_count <- 99L
mutable_session$target_level_stable_sync_count <- 99L
mutable_session$status <- "running"
assert_identical(
  fastkpc_full_cuda_phase3_session_identity_hash(mutable_session),
  first_session_hash,
  "session identity excludes mutable completion and counter fields"
)
immutable_session <- first_session
immutable_session$requested_shard_ids <- rev(
  as.integer(first_session$requested_shard_ids)
)
assert_true(
  !identical(
    fastkpc_full_cuda_phase3_session_identity_hash(immutable_session),
    first_session_hash
  ),
  "session identity binds immutable requested shard order"
)

for (shard_id in 0:1) {
  summary <- read_json(file.path(
    artifact_dir, "shards", paste0("shard_", shard_id, ".summary.json")
  ))
  assert_identical(
    summary$session_identity_hash, first_session_hash,
    "shard summary binds its immutable execution-session identity"
  )
  assert_true(
    all(c(
      "compute_capability_major", "compute_capability_minor",
      "compute_capability"
    ) %in% names(summary$gpu_environment)),
    "shard summary explicitly binds the complete GPU capability identity"
  )
}

second <- run_fixture(artifact_dir)
assert_identical(second$result$status, "complete", "resume completes corpus")
assert_identical(
  second$result$reused_shard_ids, as.integer(0:1),
  "resume reuses the two cleanly closed shards"
)
assert_identical(
  second$result$written_shard_ids, as.integer(2:3),
  "resume writes only the two missing shards"
)
assert_true(
  second$lifecycle$create_count == 1L &&
    second$lifecycle$destroy_count == 1L,
  "resume uses one execution context for all missing shards"
)
session_files_after_resume <- list.files(
  file.path(artifact_dir, "sessions"), pattern = "^session_.*\\.json$",
  full.names = TRUE
)
assert_true(
  length(session_files_after_resume) == 2L,
  "resume adds exactly one execution session"
)

pure <- run_fixture(artifact_dir)
assert_identical(pure$result$status, "complete", "pure resume is complete")
assert_identical(
  pure$result$reused_shard_ids, as.integer(0:3),
  "pure resume reuses every shard"
)
assert_identical(
  pure$result$written_shard_ids, integer(),
  "pure resume writes no shards"
)
assert_true(
  pure$lifecycle$create_count == 0L &&
    pure$lifecycle$destroy_count == 0L,
  "pure resume creates and destroys no CUDA context"
)
assert_true(
  length(list.files(
    file.path(artifact_dir, "sessions"), pattern = "^session_.*\\.json$"
  )) == 2L,
  "pure resume creates no execution session"
)

clone_fixture <- function(label) {
  destination <- tempfile(paste0("phase3-hostile-", label, "-"))
  copy_tree(artifact_dir, destination)
  destination
}

route_dir <- clone_fixture("route")
route_changed <- route_fixture("route-b")
route_identity <- identity_fixture(route_changed)
route_run <- run_fixture(
  route_dir, identity_value = route_identity, route_value = route_changed
)
assert_true(
  length(route_run$result$reused_shard_ids) == 0L &&
    length(route_run$result$written_shard_ids) == 4L &&
    route_run$lifecycle$create_count == 1L,
  "wrong route identity rejects every prior shard"
)

gpu_dir <- clone_fixture("gpu")
gpu_identity <- identity_fixture(
  route, gpu_name = "Different Synthetic GPU",
  gpu_uuid = paste0("GPU-", strrep("b", 32L))
)
gpu_run <- run_fixture(gpu_dir, identity_value = gpu_identity)
assert_true(
  length(gpu_run$result$reused_shard_ids) == 0L &&
    length(gpu_run$result$written_shard_ids) == 4L,
  "wrong GPU identity rejects every prior shard"
)

corpus_dir <- clone_fixture("corpus")
changed_targets <- target_rows
changed_targets$residual_key_sha256[[1L]] <- sha("replacement-target")
corpus_identity <- identity_fixture(route, targets = changed_targets)
corpus_run <- run_fixture(
  corpus_dir, identity_value = corpus_identity, targets = changed_targets
)
assert_true(
  length(corpus_run$result$reused_shard_ids) == 0L &&
    length(corpus_run$result$written_shard_ids) == 4L,
  "wrong target corpus identity rejects every prior shard"
)

corrupt_rds_dir <- clone_fixture("corrupt-rds")
corrupt_rds_path <- file.path(corrupt_rds_dir, "shards", "shard_0.rds")
corrupt_summary_path <- file.path(
  corrupt_rds_dir, "shards", "shard_0.summary.json"
)
corrupt_payload <- readRDS(corrupt_rds_path)
corrupt_payload$payload$target_results$target_ordinal[[1L]] <- 99L
saveRDS(corrupt_payload, corrupt_rds_path, version = 2)
corrupt_summary <- read_json(corrupt_summary_path)
corrupt_summary$rds_file_sha256 <- fastkpc_full_cuda_census_file_hash(
  corrupt_rds_path
)
write_json(corrupt_summary, corrupt_summary_path)
corrupt_rds_run <- run_fixture(corrupt_rds_dir)
assert_true(
  identical(corrupt_rds_run$result$reused_shard_ids, as.integer(1:3)) &&
    identical(corrupt_rds_run$result$written_shard_ids, 0L),
  "payload semantic mismatch is recomputed rather than reused"
)

corrupt_summary_dir <- clone_fixture("corrupt-summary")
corrupt_summary_path <- file.path(
  corrupt_summary_dir, "shards", "shard_0.summary.json"
)
corrupt_summary <- read_json(corrupt_summary_path)
corrupt_summary$payload_semantic_hash <- strrep("0", 64L)
write_json(corrupt_summary, corrupt_summary_path)
corrupt_summary_run <- run_fixture(corrupt_summary_dir)
assert_true(
  identical(corrupt_summary_run$result$reused_shard_ids,
            as.integer(1:3)) &&
    identical(corrupt_summary_run$result$written_shard_ids, 0L),
  "corrupt shard summary is recomputed rather than reused"
)

summary_only_dir <- clone_fixture("summary-only")
unlink(file.path(summary_only_dir, "shards", "shard_0.rds"))
summary_only_run <- run_fixture(summary_only_dir)
assert_true(
  identical(summary_only_run$result$reused_shard_ids,
            as.integer(1:3)) &&
    identical(summary_only_run$result$written_shard_ids, 0L),
  "summary-only shard is not reusable"
)

rds_only_dir <- clone_fixture("rds-only")
unlink(file.path(rds_only_dir, "shards", "shard_0.summary.json"))
rds_only_run <- run_fixture(rds_only_dir)
assert_true(
  identical(rds_only_run$result$reused_shard_ids, as.integer(1:3)) &&
    identical(rds_only_run$result$written_shard_ids, 0L),
  "RDS-only shard is not reusable"
)

running_session_dir <- clone_fixture("running-session")
running_summary <- read_json(file.path(
  running_session_dir, "shards", "shard_0.summary.json"
))
running_session_path <- file.path(
  running_session_dir, "sessions",
  paste0("session_", running_summary$session_id, ".json")
)
running_session <- read_json(running_session_path)
running_session$status <- "running"
write_json(running_session, running_session_path)
running_session_run <- run_fixture(running_session_dir)
assert_true(
  identical(running_session_run$result$reused_shard_ids,
            as.integer(2:3)) &&
    identical(running_session_run$result$written_shard_ids,
              as.integer(0:1)),
  "shards from a running session are recomputed"
)

wrong_counters_dir <- clone_fixture("wrong-counters")
wrong_counters_summary <- read_json(file.path(
  wrong_counters_dir, "shards", "shard_0.summary.json"
))
wrong_counters_path <- file.path(
  wrong_counters_dir, "sessions",
  paste0("session_", wrong_counters_summary$session_id, ".json")
)
wrong_counters <- read_json(wrong_counters_path)
wrong_completed_shards <- as.integer(wrong_counters$completed_shard_ids)
wrong_setup_count <- as.integer(sum(
  synthetic_plan$assignments$shard_id %in% wrong_completed_shards
))
wrong_target_count <- as.integer(sum(
  synthetic_plan$target_rows$shard_id %in% wrong_completed_shards
))
assert_true(
  wrong_setup_count != wrong_target_count,
  "hostile fixture distinguishes setup batches from logical targets"
)
wrong_counters$residual_token_acquire_count <- wrong_target_count
wrong_counters$residual_token_release_count <- wrong_target_count
wrong_counters$output_slot_acquire_count <- wrong_target_count
wrong_counters$output_slot_release_count <- wrong_target_count
write_json(wrong_counters, wrong_counters_path)
wrong_counters_run <- run_fixture(wrong_counters_dir)
assert_true(
  identical(wrong_counters_run$result$reused_shard_ids,
            as.integer(2:3)) &&
    identical(wrong_counters_run$result$written_shard_ids,
              as.integer(0:1)),
  "superseded target-count token and lease accounting rejects reuse"
)

forged_session_id_dir <- clone_fixture("forged-session-id")
forged_reference_summary <- read_json(file.path(
  forged_session_id_dir, "shards", "shard_0.summary.json"
))
declared_session_id <- forged_reference_summary$session_id
forged_session_path <- file.path(
  forged_session_id_dir, "sessions",
  paste0("session_", declared_session_id, ".json")
)
forged_session <- read_json(forged_session_path)
forged_shard_ids <- as.integer(forged_session$completed_shard_ids)
forged_session$session_id <- paste0("forged_", declared_session_id)
forged_session_hash <- fastkpc_full_cuda_phase3_session_identity_hash(
  forged_session
)
write_json(forged_session, forged_session_path)
for (shard_id in forged_shard_ids) {
  shard_rds_path <- file.path(
    forged_session_id_dir, "shards", paste0("shard_", shard_id, ".rds")
  )
  shard_summary_path <- file.path(
    forged_session_id_dir, "shards",
    paste0("shard_", shard_id, ".summary.json")
  )
  shard_payload <- readRDS(shard_rds_path)
  shard_summary <- read_json(shard_summary_path)
  assert_true(
    identical(shard_payload$session_id, declared_session_id) &&
      identical(shard_summary$session_id, declared_session_id),
    "hostile fixture preserves shard-declared session ids"
  )
  shard_payload$session_identity_hash <- forged_session_hash
  saveRDS(shard_payload, shard_rds_path, version = 2)
  shard_summary$session_identity_hash <- forged_session_hash
  shard_summary$rds_file_sha256 <- fastkpc_full_cuda_census_file_hash(
    shard_rds_path
  )
  write_json(shard_summary, shard_summary_path)
}
assert_error(
  fastkpc_full_cuda_phase3_merge_shards(
    output_dir = forged_session_id_dir,
    kind = "full_shadow",
    setup_keys = setup_keys,
    target_rows = target_rows,
    identity = identity,
    route_config = route,
    scope = "qualification",
    shard_count = 4L
  ),
  "merge rejects a session whose internal id differs from shard declarations",
  "session"
)
forged_session_id_run <- run_fixture(forged_session_id_dir)
assert_true(
  identical(
    forged_session_id_run$result$written_shard_ids,
    sort(forged_shard_ids)
  ) && identical(
    forged_session_id_run$result$reused_shard_ids,
    setdiff(as.integer(0:3), forged_shard_ids)
  ),
  "forged loaded-session id rejects every shard from that session"
)

noncanonical_name_dir <- clone_fixture("noncanonical-name")
assert_true(file.rename(
  file.path(noncanonical_name_dir, "shards", "shard_0.rds"),
  file.path(noncanonical_name_dir, "shards", "shard_00.rds")
), "rename shard RDS to a noncanonical id spelling")
assert_true(file.rename(
  file.path(noncanonical_name_dir, "shards", "shard_0.summary.json"),
  file.path(noncanonical_name_dir, "shards", "shard_00.summary.json")
), "rename shard summary to a noncanonical id spelling")
assert_error(
  run_fixture(noncanonical_name_dir),
  "noncanonical shard pair filename must fail during scan",
  "noncanonical"
)
assert_true(
  !file.exists(file.path(
    noncanonical_name_dir, "shards", "shard_0.rds"
  )) && !file.exists(file.path(
    noncanonical_name_dir, "shards", "shard_0.summary.json"
  )),
  "noncanonical shard rejection publishes no canonical duplicate pair"
)

unexpected_dir <- clone_fixture("unexpected")
assert_true(file.copy(
  file.path(unexpected_dir, "shards", "shard_0.rds"),
  file.path(unexpected_dir, "shards", "shard_4.rds")
), "copy unexpected shard RDS fixture")
assert_true(file.copy(
  file.path(unexpected_dir, "shards", "shard_0.summary.json"),
  file.path(unexpected_dir, "shards", "shard_4.summary.json")
), "copy unexpected shard summary fixture")
assert_error(
  run_fixture(unexpected_dir),
  "unexpected shard id must fail before reuse", "unexpected"
)

duplicate_dir <- clone_fixture("duplicate")
assert_true(file.copy(
  file.path(duplicate_dir, "shards", "shard_0.rds"),
  file.path(duplicate_dir, "shards", "shard_00.rds")
), "copy duplicate shard RDS fixture")
assert_true(file.copy(
  file.path(duplicate_dir, "shards", "shard_0.summary.json"),
  file.path(duplicate_dir, "shards", "shard_00.summary.json")
), "copy duplicate shard summary fixture")
assert_error(
  run_fixture(duplicate_dir),
  "duplicate spelling of a shard id must fail before reuse", "noncanonical"
)

merged_path_1 <- tempfile("phase3-merged-1-", fileext = ".rds")
merged_path_2 <- tempfile("phase3-merged-2-", fileext = ".rds")
merged_1 <- fastkpc_full_cuda_phase3_merge_shards(
  output_dir = artifact_dir,
  kind = "full_shadow",
  setup_keys = setup_keys,
  target_rows = target_rows,
  identity = identity,
  route_config = route,
  scope = "qualification",
  shard_count = 4L,
  output_path = merged_path_1
)
merged_2 <- fastkpc_full_cuda_phase3_merge_shards(
  output_dir = artifact_dir,
  kind = "full_shadow",
  setup_keys = setup_keys,
  target_rows = target_rows,
  identity = identity,
  route_config = route,
  scope = "qualification",
  shard_count = 4L,
  output_path = merged_path_2
)
assert_identical(merged_1, merged_2, "repeated merge objects are identical")
assert_identical(
  fastkpc_full_cuda_census_file_hash(merged_path_1),
  fastkpc_full_cuda_census_file_hash(merged_path_2),
  "repeated merge writes byte-identical RDS payloads"
)
assert_identical(
  merged_1$payload$setup_results$prepared_s_key_sha256,
  expected_setup_keys,
  "merged setup payload is in canonical PreparedSKey order"
)
expected_target_order <- order(
  match(target_rows$prepared_s_key_sha256, expected_setup_keys),
  target_rows$residual_key_sha256, method = "radix"
)
assert_identical(
  merged_1$payload$target_results$residual_key_sha256,
  target_rows$residual_key_sha256[expected_target_order],
  "merged target payload is in canonical setup/target order"
)

missing_merge_dir <- clone_fixture("missing-merge")
unlink(file.path(
  missing_merge_dir, "shards", "shard_0.summary.json"
))
assert_error(
  fastkpc_full_cuda_phase3_merge_shards(
    missing_merge_dir, "full_shadow", setup_keys, target_rows, identity,
    route, "qualification", 4L
  ),
  "merge rejects a missing shard pair", "missing"
)

duplicate_merge_dir <- clone_fixture("duplicate-merge")
assert_true(file.copy(
  file.path(duplicate_merge_dir, "shards", "shard_0.rds"),
  file.path(duplicate_merge_dir, "shards", "shard_00.rds")
), "copy duplicate merge RDS fixture")
assert_true(file.copy(
  file.path(duplicate_merge_dir, "shards", "shard_0.summary.json"),
  file.path(duplicate_merge_dir, "shards", "shard_00.summary.json")
), "copy duplicate merge summary fixture")
assert_error(
  fastkpc_full_cuda_phase3_merge_shards(
    duplicate_merge_dir, "full_shadow", setup_keys, target_rows, identity,
    route, "qualification", 4L
  ),
  "merge rejects a duplicate spelling of a shard id", "noncanonical"
)

# Authenticate the real Phase 0/1/2 lineage and prove the complete key join.
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
  phase0_dir = phase0_dir,
  phase1_dir = phase1_dir,
  phase2_dir = phase2_dir,
  data_path = data_path,
  require_full = TRUE
)
phase2_paths <- fastkpc_full_cuda_prepared_s_artifact_paths(phase2_dir)
assert_identical(
  fastkpc_full_cuda_census_file_hash(phase2_paths$target_state_index_rds),
  catalog$catalog_contract$target_state_index_rds_sha256,
  "real TargetState index RDS authenticates to the frozen Phase 2 hash"
)
target_states <- readRDS(phase2_paths$target_state_index_rds)
real_setup_keys <- as.character(
  catalog$setup_index$prepared_s_key_sha256
)
assert_true(
  length(real_setup_keys) == 8634L && !anyDuplicated(real_setup_keys) &&
    nrow(target_states) == 110617L,
  "real authenticated Phase 2 indexes expose all setup and target rows"
)
real_plan <- fastkpc_full_cuda_phase3_plan_shards(
  setup_keys = real_setup_keys,
  target_rows = target_states[, c(
    "prepared_s_key_sha256", "residual_key_sha256"
  ), drop = FALSE],
  scope = "full"
)
assert_identical(
  real_plan$assignments$prepared_s_key_sha256,
  sort(unique(real_setup_keys), method = "radix"),
  "real setup assignment preserves one radix row per PreparedSKey"
)
assert_identical(
  real_plan$assignments$shard_id,
  as.integer((seq_along(real_setup_keys) - 1L) %% 64L),
  "real setup assignment uses canonical rank modulo 64"
)
assert_true(
  nrow(real_plan$target_rows) == 110617L &&
    !anyNA(real_plan$target_rows$shard_id) &&
    all(real_plan$target_rows$shard_id %in% 0:63),
  "every real TargetState inherits exactly one setup shard"
)
assert_identical(
  key_hash(real_plan$target_rows$residual_key_sha256),
  as.character(
    catalog$phase2_manifest$full_canonical_target_key_corpus_hash
  ),
  "merged real target-key hash equals the Phase 2 canonical target hash"
)

cat("PASS Phase 3 deterministic shard/session/resume contract\n")
