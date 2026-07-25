fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_error <- function(expression, pattern, message) {
  error <- tryCatch({
    force(expression)
    NULL
  }, error = identity)
  assert_true(
    inherits(error, "error") && grepl(pattern, conditionMessage(error)),
    message
  )
}

hardening_failures <- character()
record_hardening_error <- function(expression, pattern, message) {
  error <- tryCatch({
    force(expression)
    NULL
  }, error = identity)
  if (!inherits(error, "error") ||
      !grepl(pattern, conditionMessage(error))) {
    actual <- if (inherits(error, "error")) {
      strsplit(conditionMessage(error), "\n", fixed = TRUE)[[1L]][[1L]]
    } else {
      "no error"
    }
    if (nchar(actual, type = "chars") > 160L) {
      actual <- paste0(substr(actual, 1L, 157L), "...")
    }
    hardening_failures <<- c(
      hardening_failures,
      paste0(message, " (got: ", actual, ")")
    )
  }
  invisible(error)
}
record_hardening_true <- function(value, message) {
  if (!isTRUE(value)) {
    hardening_failures <<- c(hardening_failures, message)
  }
  invisible(value)
}

source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/full_cuda_ci_phase3_artifacts.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_shadow.R")

phase2_identity_dir <- tempfile("shadow-phase2-file-identity-")
dir.create(phase2_identity_dir)
phase2_identity_paths <- file.path(
  phase2_identity_dir,
  c("prepared_s_setup_index.csv", "target_state_index.rds")
)
write_fixture_bytes <- function(path, value) {
  connection <- file(path, open = "wb")
  on.exit(close(connection), add = TRUE)
  writeBin(charToRaw(value), connection)
}
write_fixture_bytes(phase2_identity_paths[[1L]], "setup-a")
write_fixture_bytes(phase2_identity_paths[[2L]], "target-a")
phase2_identity_hashes <- unname(vapply(
  phase2_identity_paths,
  fastkpc_full_cuda_fixed_sp_sha256_file,
  character(1L)
))
phase2_identity_records <- .fastkpc_full_cuda_shadow_phase2_file_records(
  list(phase2_dir = phase2_identity_dir),
  phase2_identity_hashes[[1L]],
  phase2_identity_hashes[[2L]]
)
assert_true(
  identical(phase2_identity_records$sha256, phase2_identity_hashes),
  "Phase 2 file records must store hashes computed from current bytes"
)
target_identity_mtime <- file.info(phase2_identity_paths[[2L]])$mtime
write_fixture_bytes(phase2_identity_paths[[2L]], "target-b")
Sys.setFileTime(phase2_identity_paths[[2L]], target_identity_mtime)
assert_error(
  .fastkpc_full_cuda_shadow_phase2_file_records(
    list(phase2_dir = phase2_identity_dir),
    phase2_identity_hashes[[1L]],
    phase2_identity_hashes[[2L]]
  ),
  "Phase 2 current file hash mismatch",
  paste(
    "changed Phase 2 bytes must be rejected even when expected hashes and",
    "stable file metadata claims are reused"
  )
)
unlink(phase2_identity_dir, recursive = TRUE, force = TRUE)

if (!identical(.Platform$OS.type, "unix")) {
  message("SKIP: Phase 2 non-symlink identity probe is unavailable")
} else {
  assert_phase2_symlink_rejected <- function() {
    symlink_dir <- tempfile("shadow-phase2-symlink-")
    dir.create(symlink_dir)
    on.exit(unlink(symlink_dir, recursive = TRUE, force = TRUE), add = TRUE)
    setup_path <- file.path(symlink_dir, "prepared_s_setup_index.csv")
    target_backing_path <- file.path(symlink_dir, "target-backing.rds")
    target_path <- file.path(symlink_dir, "target_state_index.rds")
    write_fixture_bytes(setup_path, "setup-symlink-probe")
    write_fixture_bytes(target_backing_path, "target-symlink-probe")
    linked <- file.symlink(target_backing_path, target_path)
    if (!isTRUE(linked)) {
      message("SKIP: Phase 2 non-symlink identity probe is unavailable")
      return(invisible(FALSE))
    }
    expected_hashes <- c(
      fastkpc_full_cuda_fixed_sp_sha256_file(setup_path),
      fastkpc_full_cuda_fixed_sp_sha256_file(target_backing_path)
    )
    assert_error(
      .fastkpc_full_cuda_shadow_phase2_file_records(
        list(phase2_dir = symlink_dir),
        expected_hashes[[1L]], expected_hashes[[2L]]
      ),
      "regular non-symlink",
      "Phase 2 file records must reject symlinked payload paths"
    )
    invisible(TRUE)
  }
  assert_phase2_symlink_rejected()
  rm(assert_phase2_symlink_rejected)
}

assert_phase2_transition_rejected <- function() {
  transition_dir <- tempfile("shadow-phase2-transition-")
  dir.create(transition_dir)
  on.exit(unlink(transition_dir, recursive = TRUE, force = TRUE), add = TRUE)
  transition_paths <- file.path(
    transition_dir,
    c("prepared_s_setup_index.csv", "target_state_index.rds")
  )
  write_fixture_bytes(transition_paths[[1L]], "setup-transition-stable")
  write_fixture_bytes(transition_paths[[2L]], "target-transition-before")
  expected_hashes <- unname(vapply(
    transition_paths,
    fastkpc_full_cuda_fixed_sp_sha256_file,
    character(1L)
  ))
  transition_calls <- 0L
  transition_hook <- function(paths, stage) {
    transition_calls <<- transition_calls + 1L
    assert_true(
      identical(stage, "between_hash_passes"),
      "Phase 2 transition hook must run between hash passes"
    )
    write_fixture_bytes(paths[[2L]], "target-transition-after")
    invisible(NULL)
  }
  assert_error(
    .fastkpc_full_cuda_shadow_phase2_file_records(
      list(phase2_dir = transition_dir),
      expected_hashes[[1L]], expected_hashes[[2L]],
      .transition_hook = transition_hook
    ),
    "changed while snapshotting",
    "Phase 2 file transitions between hash passes must fail closed"
  )
  assert_true(
    identical(transition_calls, 1L),
    "Phase 2 transition probe must run exactly once"
  )
  invisible(TRUE)
}
assert_phase2_transition_rejected()
rm(assert_phase2_transition_rejected)

catalog <- fastkpc_full_cuda_open_fixed_sp_catalog(
  phase0_dir = "fastkpc/artifacts/full_cuda_ci/oracle_351x48_v1",
  phase1_dir =
    "fastkpc/artifacts/full_cuda_ci/workload_census_351x48_v1",
  phase2_dir =
    "fastkpc/artifacts/full_cuda_ci/prepared_s_contract_v1",
  data_path = paste0(
    "fastkpc/artifacts/kpc_tprs_real_zhu/",
    "cancer_RD-causalDiscoveryInput.rds"
  )
)

plan <- fastkpc_full_cuda_shadow_plan(catalog)
assert_true(nrow(plan$direct_tests) == 2213L, "direct level-0 count")
assert_true(
  nrow(plan$conditional_tests) == 238276L,
  "conditional logical count"
)
assert_true(
  all(is.na(plan$direct_tests$residual_key_x)) &&
    all(is.na(plan$direct_tests$residual_key_y)),
  "direct tests have no residual keys"
)
assert_true(
  all(nzchar(plan$conditional_tests$residual_key_x)) &&
    all(nzchar(plan$conditional_tests$residual_key_y)),
  "conditional tests have two residual keys"
)
assert_true(
  all(plan$conditional_tests$prepared_s_key_x ==
        plan$conditional_tests$prepared_s_key_y),
  "conditional endpoints share PreparedSKey"
)
assert_true(
  length(unique(plan$conditional_tests$prepared_s_key_x)) == 8634L,
  "all canonical setups consumed"
)
assert_true(
  identical(
    sort(c(
      plan$direct_tests$logical_sequence_id,
      plan$conditional_tests$logical_sequence_id
    )),
    seq_len(240489L)
  ),
  "logical sequence coverage"
)
assert_true(
  identical(
    plan$conditional_tests$shard_id,
    as.integer((match(
      plan$conditional_tests$prepared_s_key_x,
      sort(catalog$setup_index$prepared_s_key_sha256, method = "radix")
    ) - 1L) %% 64L)
  ) && identical(sort(unique(plan$conditional_tests$shard_id)), 0:63),
  "conditional rows use deterministic canonical 64-shard assignment"
)
assert_true(
  identical(
    plan$phase2_setup_index_csv_sha256,
    unname(fastkpc_full_cuda_fixed_sp_catalog_contract()[[
      "phase2_file_sha256"
    ]][["prepared_s_setup_index.csv"]])
  ) &&
    grepl("^[0-9a-f]{64}$", plan$setup_association_sha256),
  "shadow plan binds authenticated Phase 2 setup associations"
)
assert_true(
  identical(
    plan$phase2_target_state_index_rds_sha256,
    unname(fastkpc_full_cuda_fixed_sp_catalog_contract()[[
      "target_state_index_rds_sha256"
    ]])
  ) && grepl("^[0-9a-f]{64}$", plan$target_association_sha256),
  "shadow plan binds authenticated Phase 2 target associations"
)

fork_available <- identical(.Platform$OS.type, "unix") &&
  exists("mcparallel", envir = asNamespace("parallel"), inherits = FALSE) &&
  exists("mccollect", envir = asNamespace("parallel"), inherits = FALSE)
if (!isTRUE(fork_available)) {
  message("SKIP: forked shadow-plan token authentication is unavailable")
} else {
  validate_in_fork <- function() {
    child <- parallel::mcparallel({
      error <- tryCatch(
        .fastkpc_full_cuda_shadow_validate_supplied_plan(catalog, plan),
        error = identity
      )
      list(
        rejected = inherits(error, "error"),
        message = if (inherits(error, "error")) {
          conditionMessage(error)
        } else {
          NA_character_
        }
      )
    }, mc.set.seed = FALSE, silent = TRUE)
    collected <- FALSE
    on.exit({
      if (!isTRUE(collected)) {
        try(parallel::mckill(child), silent = TRUE)
        try(parallel::mccollect(child, wait = TRUE), silent = TRUE)
      }
    }, add = TRUE)
    result <- parallel::mccollect(child, wait = TRUE)
    collected <- TRUE
    result[[1L]]
  }
  fork_result <- validate_in_fork()
  assert_true(
    is.list(fork_result) && isTRUE(fork_result$rejected) && identical(
      fork_result$message,
      "direct-CI execution plan does not match authenticated catalog"
    ),
    "forked child must reject a plan token inherited from its parent"
  )
  assert_true(
    identical(
      .fastkpc_full_cuda_shadow_validate_supplied_plan(catalog, plan),
      plan
    ),
    "fork rejection must leave the parent plan token valid"
  )
  rm(fork_result, validate_in_fork)
}

shadow_registry_before_resource <- get(
  ".fastkpc_full_cuda_shadow_plan_registry", envir = .GlobalEnv,
  inherits = FALSE
)
source("fastkpc/R/full_cuda_ci_fixed_sp_shadow.R")
shadow_registry_after_resource <- get(
  ".fastkpc_full_cuda_shadow_plan_registry", envir = .GlobalEnv,
  inherits = FALSE
)
assert_true(
  identical(shadow_registry_after_resource, shadow_registry_before_resource),
  "re-sourcing shadow planner code must preserve the plan registry singleton"
)
assert_true(
  identical(
    .fastkpc_full_cuda_shadow_validate_supplied_plan(catalog, plan),
    plan
  ),
  "re-sourcing shadow planner code must preserve held plan tokens"
)
rm(shadow_registry_before_resource, shadow_registry_after_resource)

assert_registry_retains_held_plan <- function() {
  registry <- .fastkpc_full_cuda_shadow_current_plan_registry()
  original_keys <- ls(registry, all.names = TRUE)
  original_entries <- mget(
    original_keys, envir = registry, inherits = FALSE
  )
  original_locks <- setNames(vapply(
    original_keys, bindingIsLocked, logical(1L), env = registry
  ), original_keys)
  on.exit({
    current_keys <- ls(registry, all.names = TRUE)
    for (key in current_keys) {
      if (bindingIsLocked(key, registry)) unlockBinding(key, registry)
    }
    if (length(current_keys) > 0L) {
      rm(list = current_keys, envir = registry)
    }
    for (key in original_keys) {
      assign(key, original_entries[[key]], envir = registry)
      if (isTRUE(original_locks[[key]])) lockBinding(key, registry)
    }
  }, add = TRUE)

  held_registry_id <- plan$authentication_token$registry_id
  held_entry <- get(held_registry_id, envir = registry, inherits = FALSE)
  metadata <- plan[.fastkpc_full_cuda_shadow_plan_metadata_fields()]
  extra_tokens <- lapply(seq_len(33L), function(index) {
    extra_identity <- fastkpc_full_cuda_census_named_metadata_hash(list(
      schema_version = "full-cuda-ci-shadow-plan-retention-probe-v1",
      registration_index = as.integer(index),
      held_plan_identity_sha256 = plan$plan_identity_sha256
    ))
    .fastkpc_full_cuda_shadow_register_plan_identity(
      metadata, extra_identity, held_entry$phase2_file_records
    )
  })
  assert_true(
    length(extra_tokens) == 33L,
    "registry retention probe must register more than 32 compact identities"
  )
  assert_true(
    identical(
      .fastkpc_full_cuda_shadow_validate_supplied_plan(catalog, plan),
      plan
    ),
    "unrelated plan registrations must not invalidate a held plan token"
  )
  invisible(TRUE)
}
assert_registry_retains_held_plan()
rm(assert_registry_retains_held_plan)

assert_corrupt_registry_rejected <- function() {
  option_name <- .fastkpc_full_cuda_shadow_plan_registry_option_name()
  marker_name <-
    .fastkpc_full_cuda_shadow_plan_registry_marker_option_name()
  original_registry <- getOption(option_name, NULL)
  original_marker <- getOption(marker_name, NULL)
  on.exit(options(structure(
    list(original_registry, original_marker),
    names = c(option_name, marker_name)
  )), add = TRUE)

  corrupt_registry <- new.env(hash = TRUE, parent = emptyenv())
  attr(corrupt_registry, "schema_version") <-
    .fastkpc_full_cuda_shadow_plan_registry_schema_version()
  options(structure(list(corrupt_registry), names = option_name))
  assert_error(
    .fastkpc_full_cuda_shadow_current_plan_registry(),
    "shadow plan registry singleton is corrupt",
    "registry accessor must reject a replaced option-held singleton"
  )
  assert_error(
    source("fastkpc/R/full_cuda_ci_fixed_sp_shadow.R"),
    "shadow plan registry singleton is corrupt",
    "re-source must reject a replaced option-held registry singleton"
  )
  invisible(TRUE)
}
assert_corrupt_registry_rejected()
rm(assert_corrupt_registry_rejected)

association_swap_catalog <- catalog
association_swap_keys <-
  association_swap_catalog$setup_index$prepared_s_key_sha256[1:2]
association_swap_shards <- as.integer((match(
  association_swap_keys,
  sort(catalog$setup_index$prepared_s_key_sha256, method = "radix")
) - 1L) %% 64L)
assert_true(
  length(unique(association_swap_shards)) == 2L,
  "association-swap probe must move setup ownership across shards"
)
association_swap_catalog$setup_index$same_S_group_id[1:2] <-
  rev(association_swap_catalog$setup_index$same_S_group_id[1:2])
assert_error(
  fastkpc_full_cuda_shadow_plan(association_swap_catalog),
  "authenticated Phase 2 setup association mismatch",
  paste(
    "same-S group to PreparedSKey association swap must fail closed even",
    "when the canonical key set is unchanged"
  )
)

target_association_swap_catalog <- catalog
target_setup_shards <- as.integer((match(
  target_association_swap_catalog$setup_index$prepared_s_key_sha256,
  sort(catalog$setup_index$prepared_s_key_sha256, method = "radix")
) - 1L) %% 64L)
target_setup_a <- 1L
target_setup_b <- which(target_setup_shards != target_setup_shards[[1L]])[[1L]]
target_swap_groups <-
  target_association_swap_catalog$setup_index$same_S_group_id[
    c(target_setup_a, target_setup_b)
  ]
target_setup_metadata <-
  target_association_swap_catalog$inputs$same_s_setup_metadata
target_swap_setup_match <- match(
  target_swap_groups, target_setup_metadata$same_S_group_id
)
target_swap_fingerprints <-
  target_setup_metadata$setup_fingerprint[target_swap_setup_match]
target_metadata <-
  target_association_swap_catalog$inputs$target_fit_metadata
target_rows_a <- which(target_metadata$same_S_group_id == target_swap_groups[[1L]])
target_rows_b <- which(target_metadata$same_S_group_id == target_swap_groups[[2L]])
target_swap_rows <- c(target_rows_a, target_rows_b)
target_swap_identity <- target_metadata[
  target_swap_rows, c("residual_key_sha256", "target"), drop = FALSE
]
assert_true(
  !anyNA(target_swap_setup_match) &&
    length(target_rows_a) > 0L && length(target_rows_b) > 0L &&
    target_setup_shards[[target_setup_a]] !=
      target_setup_shards[[target_setup_b]],
  "target-association swap probe must cover populated different-shard setups"
)
target_metadata$same_S_group_id[target_rows_a] <- target_swap_groups[[2L]]
target_metadata$setup_fingerprint[target_rows_a] <-
  target_swap_fingerprints[[2L]]
target_metadata$same_S_group_id[target_rows_b] <- target_swap_groups[[1L]]
target_metadata$setup_fingerprint[target_rows_b] <-
  target_swap_fingerprints[[1L]]
target_association_swap_catalog$inputs$target_fit_metadata <- target_metadata
assert_true(
  identical(
    target_association_swap_catalog$inputs$target_fit_metadata[
      target_swap_rows, c("residual_key_sha256", "target"), drop = FALSE
    ],
    target_swap_identity
  ),
  "target-association swap probe must preserve residual keys and targets"
)
assert_error(
  fastkpc_full_cuda_shadow_plan(target_association_swap_catalog),
  "authenticated Phase 2 target association mismatch",
  paste(
    "self-consistent residual-key target association swaps must fail closed",
    "before conditional shard assignment"
  )
)

full_scope <- fastkpc_full_cuda_fixed_sp_scope(catalog, "full")
logical_tests <- catalog$inputs$logical_tests
authenticated_phase2_evidence <-
  fastkpc_full_cuda_shadow_authenticated_phase2_records(catalog)
map_units <- function(
    logical = logical_tests, setup = full_scope$setup_rows,
    target = full_scope$target_rows, setup_index = catalog$setup_index,
    logical_contract = fastkpc_full_cuda_shadow_logical_contract(logical)) {
  fastkpc_full_cuda_shadow_map_execution_units(
    logical_tests = logical,
    setup_rows = setup,
    target_rows = target,
    setup_index = setup_index,
    expected_logical_contract = logical_contract,
    authenticated_setup_evidence = authenticated_phase2_evidence$setup,
    authenticated_target_evidence = authenticated_phase2_evidence$target,
    shard_count = 64L
  )
}

first_conditional <- which(logical_tests$level > 0L)[[1L]]
endpoint_key <- logical_tests$residual_key_x[[first_conditional]]
endpoint_row <- match(endpoint_key, full_scope$target_rows$residual_key_sha256)

missing_endpoint <- full_scope$target_rows
missing_endpoint$residual_key_sha256[[endpoint_row]] <- strrep("f", 64L)
assert_error(
  map_units(target = missing_endpoint),
  "authenticated Phase 2 target association mismatch",
  "missing conditional endpoint key must fail closed"
)
rm(missing_endpoint)

duplicate_endpoint <- full_scope$target_rows
duplicate_endpoint$residual_key_sha256[[2L]] <-
  duplicate_endpoint$residual_key_sha256[[1L]]
assert_error(
  map_units(target = duplicate_endpoint),
  "duplicate residual key",
  "duplicate target endpoint key must fail closed"
)
rm(duplicate_endpoint)

fingerprint_conflict <- full_scope$target_rows
fingerprint_conflict$setup_fingerprint[[endpoint_row]] <- strrep("e", 64L)
assert_error(
  map_units(target = fingerprint_conflict),
  "authenticated Phase 2 target association mismatch",
  "target/setup fingerprint conflict must fail closed"
)
rm(fingerprint_conflict)

cross_setup <- logical_tests
row_y <- cross_setup$y[[first_conditional]]
endpoint_x_setup <- full_scope$target_rows$prepared_s_key_sha256[[endpoint_row]]
cross_target <- which(
  full_scope$target_rows$target == row_y &
    full_scope$target_rows$prepared_s_key_sha256 != endpoint_x_setup
)[[1L]]
cross_setup$residual_key_y[[first_conditional]] <-
  full_scope$target_rows$residual_key_sha256[[cross_target]]
assert_error(
  map_units(logical = cross_setup),
  "different PreparedSKeys",
  "cross-setup conditional endpoints must fail closed"
)
rm(cross_setup)

direct_with_key <- logical_tests
direct_with_key$residual_key_x[[1L]] <- endpoint_key
assert_error(
  map_units(logical = direct_with_key),
  "direct logical rows must not carry residual keys",
  "direct rows carrying residual keys must fail closed"
)
rm(direct_with_key)

conditional_without_key <- logical_tests
conditional_without_key$residual_key_y[[first_conditional]] <- NA_character_
assert_error(
  map_units(logical = conditional_without_key),
  "conditional logical rows require two residual keys",
  "conditional rows missing residual keys must fail closed"
)
rm(conditional_without_key)

duplicate_logical_id <- logical_tests
duplicate_logical_id$logical_sequence_id[[2L]] <-
  duplicate_logical_id$logical_sequence_id[[1L]]
assert_error(
  map_units(
    logical = duplicate_logical_id,
    logical_contract =
      fastkpc_full_cuda_shadow_logical_contract(duplicate_logical_id)
  ),
  "duplicate logical_sequence_id",
  "duplicate logical sequence IDs must fail closed"
)

assert_current_catalog_association_rejected <- function(
    candidate_catalog, message, candidate_plan = plan) {
  instrumented <- c(
    canonical_plan = "fastkpc_full_cuda_shadow_plan",
    authenticated_phase2 =
      "fastkpc_full_cuda_shadow_authenticated_phase2_records",
    phase2_capture = "fastkpc_full_cuda_fixed_sp_capture_phase2_files"
  )
  originals <- mget(instrumented, envir = .GlobalEnv, inherits = FALSE)
  original_backend <- fastkpc_legacy_dcov_gamma_cpp_oracle_batch
  forbidden_calls <- setNames(integer(length(instrumented)), names(instrumented))
  backend_calls <- 0L
  on.exit({
    for (index in seq_along(instrumented)) {
      assign(
        instrumented[[index]], originals[[index]], envir = .GlobalEnv
      )
    }
    assign(
      "fastkpc_legacy_dcov_gamma_cpp_oracle_batch", original_backend,
      envir = .GlobalEnv
    )
  }, add = TRUE)
  for (field in names(instrumented)) {
    local({
      call_field <- field
      call_name <- instrumented[[field]]
      assign(call_name, function(...) {
        forbidden_calls[[call_field]] <<-
          forbidden_calls[[call_field]] + 1L
        stop(
          "hostile catalog validation called forbidden ", call_field,
          call. = FALSE
        )
      }, envir = .GlobalEnv)
    })
  }
  assign(
    "fastkpc_legacy_dcov_gamma_cpp_oracle_batch",
    function(...) {
      backend_calls <<- backend_calls + 1L
      stop("hostile catalog association reached direct-CI backend",
           call. = FALSE)
    },
    envir = .GlobalEnv
  )
  error <- tryCatch(
    .fastkpc_full_cuda_shadow_direct_ci_rows(
      candidate_catalog, candidate_plan
    ),
    error = identity
  )
  assert_true(
    inherits(error, "error") && identical(
      conditionMessage(error),
      "direct-CI execution plan does not match authenticated catalog"
    ),
    paste0(message, " must fail at supplied-plan authentication")
  )
  assert_true(
    identical(backend_calls, 0L),
    paste0(message, " must fail before the direct-CI backend")
  )
  assert_true(
    identical(forbidden_calls, setNames(
      integer(length(instrumented)), names(instrumented)
    )),
    paste0(message, " must not rebuild or capture Phase 2 evidence")
  )
  invisible(error)
}

assert_current_catalog_association_rejected(
  association_swap_catalog,
  "mutated current setup-group association"
)
assert_current_catalog_association_rejected(
  target_association_swap_catalog,
  "mutated current target group/fingerprint association"
)

direct_type_plan <- plan
direct_type_plan$direct_tests$shard_id <-
  as.double(direct_type_plan$direct_tests$shard_id)
assert_true(
  identical(
    fastkpc_full_cuda_census_frame_hash(direct_type_plan$direct_tests),
    plan$direct_tests_sha256
  ),
  "direct type-mutation probe must preserve the portable frame hash"
)
assert_current_catalog_association_rejected(
  catalog, "mutated direct frame column type",
  candidate_plan = direct_type_plan
)
rm(direct_type_plan)

conditional_type_plan <- plan
conditional_type_plan$conditional_tests$shard_id <-
  as.double(conditional_type_plan$conditional_tests$shard_id)
assert_true(
  identical(
    fastkpc_full_cuda_census_frame_hash(conditional_type_plan$conditional_tests),
    plan$conditional_tests_sha256
  ),
  "conditional type-mutation probe must preserve the portable frame hash"
)
assert_current_catalog_association_rejected(
  catalog, "mutated conditional frame column type",
  candidate_plan = conditional_type_plan
)
rm(conditional_type_plan)

direct_output_dir <- tempfile("full-cuda-ci-direct-")
supplied_plan_calls <- new.env(parent = emptyenv())
supplied_plan_calls$canonical_plan <- 0L
supplied_plan_calls$authenticated_phase2 <- 0L
supplied_plan_calls$phase2_capture <- 0L
run_supplied_plan_direct_ci <- function() {
  instrumented <- c(
    canonical_plan = "fastkpc_full_cuda_shadow_plan",
    authenticated_phase2 =
      "fastkpc_full_cuda_shadow_authenticated_phase2_records",
    phase2_capture = "fastkpc_full_cuda_fixed_sp_capture_phase2_files"
  )
  originals <- mget(instrumented, envir = .GlobalEnv, inherits = FALSE)
  on.exit({
    for (index in seq_along(instrumented)) {
      assign(
        instrumented[[index]], originals[[index]], envir = .GlobalEnv
      )
    }
  }, add = TRUE)
  for (field in names(instrumented)) {
    local({
      call_field <- field
      call_name <- instrumented[[field]]
      assign(call_name, function(...) {
        supplied_plan_calls[[call_field]] <-
          supplied_plan_calls[[call_field]] + 1L
        stop(
          "supplied-plan direct CI called forbidden ", call_field,
          call. = FALSE
        )
      }, envir = .GlobalEnv)
    })
  }
  fastkpc_full_cuda_shadow_write_direct_ci(
    catalog = catalog,
    output_dir = direct_output_dir,
    plan = plan
  )
}
direct_artifact <- run_supplied_plan_direct_ci()
assert_true(
  identical(
    unlist(as.list(supplied_plan_calls), use.names = TRUE)[
      c("canonical_plan", "authenticated_phase2", "phase2_capture")
    ],
    c(canonical_plan = 0L, authenticated_phase2 = 0L, phase2_capture = 0L)
  ),
  "supplied-plan direct CI must not rebuild or recapture Phase 2 evidence"
)

assert_supplied_plan_rejected <- function(
    candidate_plan, message, candidate_catalog = catalog) {
  original_backend <- fastkpc_legacy_dcov_gamma_cpp_oracle_batch
  backend_calls <- 0L
  assign(
    "fastkpc_legacy_dcov_gamma_cpp_oracle_batch",
    function(...) {
      backend_calls <<- backend_calls + 1L
      stop("hostile supplied plan reached direct-CI backend", call. = FALSE)
    },
    envir = .GlobalEnv
  )
  on.exit(assign(
    "fastkpc_legacy_dcov_gamma_cpp_oracle_batch", original_backend,
    envir = .GlobalEnv
  ), add = TRUE)
  error <- tryCatch(
    .fastkpc_full_cuda_shadow_direct_ci_rows(
      candidate_catalog, candidate_plan
    ),
    error = identity
  )
  record_hardening_true(
    inherits(error, "error") && identical(
      conditionMessage(error),
      "direct-CI execution plan does not match authenticated catalog"
    ),
    paste0(message, " must fail closed at plan authentication")
  )
  record_hardening_true(
    identical(backend_calls, 0L),
    paste0(message, " must fail before the direct-CI backend")
  )
  invisible(error)
}

resign_supplied_plan <- function(
    candidate_plan, rehash_direct = FALSE,
    rehash_conditional = FALSE) {
  if (isTRUE(rehash_direct)) {
    candidate_plan$direct_tests_sha256 <-
      fastkpc_full_cuda_census_frame_hash(candidate_plan$direct_tests)
  }
  if (isTRUE(rehash_conditional)) {
    candidate_plan$conditional_tests_sha256 <-
      fastkpc_full_cuda_census_frame_hash(candidate_plan$conditional_tests)
  }
  candidate_plan$plan_identity_sha256 <-
    .fastkpc_full_cuda_shadow_plan_metadata_hash(candidate_plan)
  forged_token <- new.env(parent = emptyenv())
  for (field in ls(candidate_plan$authentication_token, all.names = TRUE)) {
    value <- candidate_plan$authentication_token[[field]]
    if (identical(field, "plan_identity_sha256")) {
      value <- candidate_plan$plan_identity_sha256
    }
    assign(field, value, envir = forged_token)
  }
  lockEnvironment(forged_token, bindings = TRUE)
  candidate_plan$authentication_token <- forged_token
  candidate_plan
}

hostile_direct_plan <- plan
hostile_direct_plan$direct_tests$source_task_index[[1L]] <-
  hostile_direct_plan$direct_tests$source_task_index[[1L]] + 1L
assert_supplied_plan_rejected(
  hostile_direct_plan, "mutated direct plan row"
)
resigned_direct_plan <- resign_supplied_plan(
  hostile_direct_plan, rehash_direct = TRUE
)
assert_supplied_plan_rejected(
  resigned_direct_plan, "self-consistently re-signed direct plan row"
)
rm(hostile_direct_plan, resigned_direct_plan)

hostile_conditional_plan <- plan
hostile_conditional_plan$conditional_tests$shard_id[[1L]] <- as.integer(
  (hostile_conditional_plan$conditional_tests$shard_id[[1L]] + 1L) %% 64L
)
assert_supplied_plan_rejected(
  hostile_conditional_plan, "mutated conditional plan row"
)
resigned_conditional_plan <- resign_supplied_plan(
  hostile_conditional_plan, rehash_conditional = TRUE
)
assert_supplied_plan_rejected(
  resigned_conditional_plan,
  "self-consistently re-signed conditional plan row"
)
rm(hostile_conditional_plan, resigned_conditional_plan)

hostile_setup_association_plan <- plan
hostile_setup_association_plan$setup_association_sha256 <- strrep("0", 64L)
assert_supplied_plan_rejected(
  hostile_setup_association_plan, "mutated setup association metadata"
)
resigned_association_plan <- resign_supplied_plan(
  hostile_setup_association_plan
)
assert_supplied_plan_rejected(
  resigned_association_plan,
  "self-consistently re-signed setup association metadata"
)
rm(hostile_setup_association_plan, resigned_association_plan)

hostile_target_association_plan <- plan
hostile_target_association_plan$target_association_sha256 <- strrep("1", 64L)
assert_supplied_plan_rejected(
  hostile_target_association_plan, "mutated target association metadata"
)
rm(hostile_target_association_plan)

hostile_setup_file_plan <- plan
hostile_setup_file_plan$phase2_setup_index_csv_sha256 <- strrep("7", 64L)
assert_supplied_plan_rejected(
  hostile_setup_file_plan, "mutated Phase 2 setup file identity"
)
rm(hostile_setup_file_plan)

hostile_target_file_plan <- plan
hostile_target_file_plan$phase2_target_state_index_rds_sha256 <-
  strrep("8", 64L)
assert_supplied_plan_rejected(
  hostile_target_file_plan, "mutated Phase 2 target file identity"
)
rm(hostile_target_file_plan)

hostile_route_plan <- plan
hostile_route_plan$route_config_sha256 <- strrep("2", 64L)
assert_supplied_plan_rejected(
  hostile_route_plan, "mutated route metadata"
)
rm(hostile_route_plan)

hostile_catalog_binding_plan <- plan
hostile_catalog_binding_plan$catalog_authority_sha256 <- strrep("3", 64L)
assert_supplied_plan_rejected(
  hostile_catalog_binding_plan, "mutated plan catalog binding"
)
rm(hostile_catalog_binding_plan)

hostile_logical_catalog <- catalog
hostile_logical_catalog$inputs$logical_tests$x[[1L]] <-
  hostile_logical_catalog$inputs$logical_tests$y[[1L]]
assert_supplied_plan_rejected(
  plan, "mutated canonical direct catalog row",
  candidate_catalog = hostile_logical_catalog
)
rm(hostile_logical_catalog)

missing_token_plan <- plan
missing_token_plan$authentication_token <- NULL
assert_supplied_plan_rejected(missing_token_plan, "missing plan token")
rm(missing_token_plan)

forged_token <- new.env(parent = emptyenv())
if (is.environment(plan$authentication_token)) {
  for (field in ls(plan$authentication_token, all.names = TRUE)) {
    assign(field, plan$authentication_token[[field]], envir = forged_token)
  }
} else {
  forged_token$schema_version <- "full-cuda-ci-shadow-plan-token-v2"
  forged_token$registry_id <- strrep("4", 64L)
  forged_token$creator_pid <- as.integer(Sys.getpid())
  forged_token$catalog_authority_sha256 <- strrep("5", 64L)
}
lockEnvironment(forged_token, bindings = TRUE)
forged_token_plan <- plan
forged_token_plan$authentication_token <- forged_token
assert_supplied_plan_rejected(forged_token_plan, "forged plan token")
rm(forged_token_plan, forged_token)

shadow_registry_available <-
  exists(
    ".fastkpc_full_cuda_shadow_plan_registry", envir = .GlobalEnv,
    inherits = FALSE
  ) && is.environment(plan$authentication_token) &&
  exists("registry_id", envir = plan$authentication_token, inherits = FALSE)
record_hardening_true(
  shadow_registry_available,
  "authenticated plan must expose a process-local registry token"
)
if (isTRUE(shadow_registry_available)) {
  assert_registry_entry_pid_rejected <- function() {
    shadow_registry <- .fastkpc_full_cuda_shadow_current_plan_registry()
    registry_id <- plan$authentication_token$registry_id
    original_entry <- get(
      registry_id, envir = shadow_registry, inherits = FALSE
    )
    original_locked <- bindingIsLocked(registry_id, shadow_registry)
    on.exit({
      if (exists(registry_id, envir = shadow_registry, inherits = FALSE)) {
        if (bindingIsLocked(registry_id, shadow_registry)) {
          unlockBinding(registry_id, shadow_registry)
        }
        rm(list = registry_id, envir = shadow_registry)
      }
      assign(registry_id, original_entry, envir = shadow_registry)
      if (isTRUE(original_locked)) lockBinding(registry_id, shadow_registry)
    }, add = TRUE)

    hostile_entry <- original_entry
    hostile_entry$creator_pid <- as.double(hostile_entry$creator_pid)
    if (isTRUE(original_locked)) unlockBinding(registry_id, shadow_registry)
    assign(registry_id, hostile_entry, envir = shadow_registry)
    if (isTRUE(original_locked)) lockBinding(registry_id, shadow_registry)
    assert_supplied_plan_rejected(
      plan, "non-integer registry creator PID"
    )
    invisible(TRUE)
  }
  assert_registry_entry_pid_rejected()
  rm(assert_registry_entry_pid_rejected)

  assert_stale_plan_token_rejected <- function() {
    shadow_registry <- .fastkpc_full_cuda_shadow_current_plan_registry()
    stale_registry_id <- plan$authentication_token$registry_id
    stale_registry_entry <- get(
      stale_registry_id, envir = shadow_registry, inherits = FALSE
    )
    stale_binding_locked <- bindingIsLocked(
      stale_registry_id, shadow_registry
    )
    removed <- FALSE
    on.exit({
      if (isTRUE(removed)) {
        if (exists(
              stale_registry_id, envir = shadow_registry, inherits = FALSE
            )) {
          if (bindingIsLocked(stale_registry_id, shadow_registry)) {
            unlockBinding(stale_registry_id, shadow_registry)
          }
          rm(list = stale_registry_id, envir = shadow_registry)
        }
        assign(
          stale_registry_id, stale_registry_entry, envir = shadow_registry
        )
        if (isTRUE(stale_binding_locked)) {
          lockBinding(stale_registry_id, shadow_registry)
        }
      }
    }, add = TRUE)
    record_hardening_true(
      !any(c("direct_tests", "conditional_tests", "setup", "target") %in%
             names(stale_registry_entry)) &&
        identical(
          plan$authentication_token$creator_pid, as.integer(Sys.getpid())
        ) && identical(
          stale_registry_entry$creator_pid, as.integer(Sys.getpid())
        ) &&
        is.data.frame(stale_registry_entry$phase2_file_records) &&
        nrow(stale_registry_entry$phase2_file_records) == 2L &&
        as.numeric(object.size(stale_registry_entry)) < 1000000,
      paste(
        "shadow plan registry entry must retain only compact authenticated",
        "identity"
      )
    )
    if (isTRUE(stale_binding_locked)) {
      unlockBinding(stale_registry_id, shadow_registry)
    }
    rm(list = stale_registry_id, envir = shadow_registry)
    removed <- TRUE
    assert_supplied_plan_rejected(plan, "stale plan token")
    invisible(TRUE)
  }
  assert_stale_plan_token_rejected()
  rm(assert_stale_plan_token_rejected)
}

assert_cross_catalog_plan_rejected <- function() {
  catalog_authority <-
    fastkpc_full_cuda_phase3_discover_catalog_authority(catalog)
  registry <- get(
    ".fastkpc_full_cuda_phase3_catalog_authority_registry",
    envir = .GlobalEnv, inherits = FALSE
  )
  original_keys <- ls(registry, all.names = TRUE)
  original_entries <- mget(
    original_keys, envir = registry, inherits = FALSE
  )
  original_locks <- setNames(vapply(
    original_keys, bindingIsLocked, logical(1L), env = registry
  ), original_keys)
  on.exit({
    current_keys <- ls(registry, all.names = TRUE)
    for (key in current_keys) {
      if (bindingIsLocked(key, registry)) unlockBinding(key, registry)
    }
    if (length(current_keys) > 0L) {
      rm(list = current_keys, envir = registry)
    }
    for (key in original_keys) {
      assign(key, original_entries[[key]], envir = registry)
      if (isTRUE(original_locks[[key]])) lockBinding(key, registry)
    }
  }, add = TRUE)

  cross_authority <- get(
    catalog_authority$authority_sha256,
    envir = registry, inherits = FALSE
  )
  cross_authority$phase2_source_commit <- strrep("6", 40L)
  cross_authority$sha256 <-
    .fastkpc_full_cuda_phase3_catalog_authority_hash(cross_authority)
  cross_catalog <- catalog
  cross_catalog$phase2_manifest$source_commit <-
    cross_authority$phase2_source_commit
  cross_catalog$phase3_catalog_authority_token <-
    .fastkpc_full_cuda_phase3_register_catalog_authority(cross_authority)
  cross_catalog$phase3_catalog_authority_sha256 <- cross_authority$sha256
  cross_catalog_authority <-
    fastkpc_full_cuda_phase3_discover_catalog_authority(cross_catalog)
  assert_true(
    !identical(
      cross_catalog_authority$authority_sha256,
      catalog_authority$authority_sha256
    ),
    "cross-catalog probe must have a distinct authenticated authority"
  )
  assert_supplied_plan_rejected(
    plan, "cross-catalog supplied plan", candidate_catalog = cross_catalog
  )
  invisible(TRUE)
}
assert_cross_catalog_plan_rejected()
rm(assert_cross_catalog_plan_rejected)

validated_direct <- fastkpc_full_cuda_phase3_validate_direct_ci_payload(
  output_dir = direct_output_dir,
  catalog = catalog
)
direct_schema <- c(
  "logical_sequence_id", "source_sequence_id", "source_task_index",
  "level", "x", "y", "S_key", "residual_key_x", "residual_key_y",
  "reference_p_value", "candidate_p_value",
  "absolute_p_value_difference", "alpha", "reference_decision",
  "candidate_decision", "decision_flip", "backend",
  "low_rank_backend", "backend_error", "spectra_fallback"
)
assert_true(
  identical(names(direct_artifact$rows), direct_schema) &&
    identical(names(validated_direct$payload$rows), direct_schema),
  "direct-CI row schema must be exact"
)
assert_true(
  nrow(direct_artifact$rows) == 2213L &&
    identical(direct_artifact$rows$logical_sequence_id, seq_len(2213L)) &&
    all(direct_artifact$rows$level == 0L) &&
    all(direct_artifact$rows$S_key == "") &&
    all(is.na(direct_artifact$rows$residual_key_x)) &&
    all(is.na(direct_artifact$rows$residual_key_y)),
  "direct-CI canonical row coverage"
)
assert_true(
  all(direct_artifact$rows$backend == "legacy-cpp") &&
    all(direct_artifact$rows$low_rank_backend == "spectra") &&
    !any(direct_artifact$rows$backend_error) &&
    !any(direct_artifact$rows$spectra_fallback) &&
    !any(direct_artifact$rows$decision_flip),
  "direct-CI route and correctness gates"
)
assert_true(
  identical(
    direct_artifact$rows$candidate_decision,
    ifelse(
      direct_artifact$rows$candidate_p_value > direct_artifact$rows$alpha,
      "independent", "dependent"
    )
  ),
  "direct-CI candidate decision uses strict p-value greater than alpha"
)
assert_true(
  identical(
    sort(list.files(direct_output_dir), method = "radix"),
    c("direct_ci.rds", "direct_ci.summary.json")
  ),
  "direct-CI publication is one atomic payload pair"
)
assert_true(
  isTRUE(validated_direct$authenticated) &&
    identical(validated_direct$summary$row_count, 2213L) &&
    identical(validated_direct$summary$backend_error_count, 0L) &&
    identical(validated_direct$summary$spectra_fallback_count, 0L) &&
    identical(validated_direct$summary$decision_flip_count, 0L) &&
    isTRUE(validated_direct$summary$pass),
  "direct-CI summary correctness gates"
)
catalog_lineage <- fastkpc_full_cuda_phase3_discover_catalog_evidence(
  catalog
)
assert_true(
  identical(
    validated_direct$payload$lineage$phase0_manifest_hash,
    catalog_lineage$phase0_manifest_hash
  ) && identical(
    validated_direct$payload$lineage$phase1_manifest_hash,
    catalog_lineage$phase1_manifest_hash
  ) && identical(
    validated_direct$payload$lineage$dataset_matrix_sha256,
    catalog_lineage$dataset_matrix_sha256
  ) && identical(
    validated_direct$payload$lineage$route_config_hash,
    fastkpc_full_cuda_phase3_route_config()$sha256
  ) && identical(
    validated_direct$payload$lineage$execution_device, "cpu"
  ) && identical(
    validated_direct$payload$lineage$residual_backend, "none"
  ),
  "direct-CI artifact authenticates Phase 0/1/data/route lineage"
)

assert_true(
  identical(
    fastkpc_full_cuda_data_hash(catalog$inputs$data),
    catalog_lineage$dataset_matrix_sha256
  ),
  "canonical in-memory data must match authenticated matrix hash"
)
mutated_data_catalog <- catalog
mutated_data_catalog$inputs$data[[1L]] <-
  mutated_data_catalog$inputs$data[[1L]] + 1e-8
assert_true(
  !identical(
    fastkpc_full_cuda_data_hash(mutated_data_catalog$inputs$data),
    catalog_lineage$dataset_matrix_sha256
  ),
  "data-mutation probe must change the actual matrix hash"
)
mutated_data_output <- tempfile("full-cuda-ci-direct-mutated-data-")
assert_error(
  fastkpc_full_cuda_shadow_write_direct_ci(
    mutated_data_catalog, mutated_data_output
  ),
  "direct-CI canonical data matrix hash mismatch",
  "direct-CI writer must reject mutated in-memory canonical data"
)
mutated_data_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  mutated_data_output, kind = "full_shadow"
)
assert_true(
  !file.exists(mutated_data_paths$direct_ci_rds) &&
    !file.exists(mutated_data_paths$direct_ci_summary_json),
  "mutated input data must fail before direct-CI publication"
)
assert_error(
  fastkpc_full_cuda_phase3_validate_direct_ci_payload(
    direct_output_dir, mutated_data_catalog
  ),
  "direct-CI canonical data matrix hash mismatch",
  "direct-CI validator must recompute the actual input matrix hash"
)

invalid_publish_rows <- direct_artifact$rows
invalid_x <- invalid_publish_rows$x[[1L]]
invalid_publish_rows$x[[1L]] <- invalid_publish_rows$y[[1L]]
invalid_publish_rows$y[[1L]] <- invalid_x
invalid_publish_dir <- tempfile("full-cuda-ci-direct-invalid-")
assert_error(
  fastkpc_full_cuda_phase3_publish_direct_ci_payload(
    invalid_publish_rows, catalog, invalid_publish_dir
  ),
  "Phase 1 direct row lineage mismatch",
  "direct-CI publisher must reject noncanonical logical rows"
)
invalid_publish_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  invalid_publish_dir, kind = "full_shadow"
)
assert_true(
  !file.exists(invalid_publish_paths$direct_ci_rds) &&
    !file.exists(invalid_publish_paths$direct_ci_summary_json),
  "failed direct-CI publication must leave no payload or completion marker"
)

publication_failure_dir <- tempfile("full-cuda-ci-direct-publish-failure-")
publication_failure_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  publication_failure_dir, kind = "full_shadow"
)
publication_failure_hook_count <- 0L
record_hardening_error(
  fastkpc_full_cuda_phase3_publish_direct_ci_payload(
    direct_artifact$rows, catalog, publication_failure_dir,
    .transition_hook = function(stage, paths, artifact_lock) {
      if (identical(stage, "after_final_rds_publication")) {
        publication_failure_hook_count <<-
          publication_failure_hook_count + 1L
        if (!.fastkpc_full_cuda_phase3_owns_lock(
              artifact_lock, "direct_artifact"
            )) {
          stop("direct artifact lock was not held at publication boundary",
               call. = FALSE)
        }
        stop("injected failure after final direct_ci.rds publication",
             call. = FALSE)
      }
    }
  ),
  "^injected failure after final direct_ci.rds publication$",
  "final-RDS publication failure boundary was not reached"
)
record_hardening_true(
  identical(publication_failure_hook_count, 1L),
  "final-RDS publication hook did not run exactly once"
)
record_hardening_true(
  !file.exists(publication_failure_paths$direct_ci_rds) &&
    !file.exists(publication_failure_paths$direct_ci_summary_json),
  paste(
    "failure immediately after final RDS publication must remove both",
    "direct-CI final files"
  )
)

direct_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  direct_output_dir, kind = "full_shadow"
)

copy_direct_pair <- function(source_paths, output_dir) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  destination_paths <- fastkpc_full_cuda_phase3_artifact_paths(
    output_dir, kind = "full_shadow"
  )
  copied <- file.copy(
    c(source_paths$direct_ci_rds, source_paths$direct_ci_summary_json),
    c(
      destination_paths$direct_ci_rds,
      destination_paths$direct_ci_summary_json
    ),
    overwrite = TRUE
  )
  assert_true(all(copied), "direct-CI pair fixture copy")
  destination_paths
}
direct_pair_hashes <- function(paths) {
  pair <- c(paths$direct_ci_rds, paths$direct_ci_summary_json)
  setNames(
    vapply(pair, fastkpc_full_cuda_census_file_hash, character(1L)),
    c("direct_ci_rds", "direct_ci_summary_json")
  )
}
direct_pair_matches <- function(paths, expected_hashes) {
  pair <- c(paths$direct_ci_rds, paths$direct_ci_summary_json)
  all(file.exists(pair)) && !any(dir.exists(pair)) &&
    !any(nzchar(Sys.readlink(pair))) &&
    identical(direct_pair_hashes(paths), expected_hashes)
}
retarget_output_alias <- function(alias, target) {
  unlink(alias, recursive = FALSE, force = TRUE)
  if (!isTRUE(file.symlink(target, alias))) {
    stop("failed to retarget direct-CI output alias", call. = FALSE)
  }
  invisible(alias)
}

validation_alias_original <- tempfile("direct-ci-validation-original-")
validation_alias_retarget <- tempfile("direct-ci-validation-retarget-")
validation_alias <- tempfile("direct-ci-validation-alias-")
validation_alias_original_paths <- copy_direct_pair(
  direct_paths, validation_alias_original
)
validation_alias_retarget_paths <- copy_direct_pair(
  direct_paths, validation_alias_retarget
)
validation_alias_retarget_hashes <- direct_pair_hashes(
  validation_alias_retarget_paths
)
assert_true(
  isTRUE(file.symlink(validation_alias_original, validation_alias)),
  "direct-CI validation output alias fixture"
)
validation_alias_hook_count <- 0L
validation_alias_bound <- FALSE
validation_alias_result <- tryCatch(
  fastkpc_full_cuda_phase3_validate_direct_ci_payload(
    validation_alias, catalog,
    .transition_hook = function(stage, paths, artifact_lock) {
      if (identical(stage, "after_capture")) {
        validation_alias_hook_count <<- validation_alias_hook_count + 1L
        validation_alias_bound <<- identical(
          dirname(paths$direct_ci_rds),
          normalizePath(validation_alias_original, mustWork = TRUE)
        )
        retarget_output_alias(
          validation_alias,
          normalizePath(validation_alias_retarget, mustWork = TRUE)
        )
      }
    }
  ),
  error = identity
)
record_hardening_true(
  !inherits(validation_alias_result, "error") &&
    isTRUE(validation_alias_result$authenticated),
  "validation through a retargeted output alias must remain authenticated"
)
record_hardening_true(
  identical(validation_alias_hook_count, 1L) &&
    isTRUE(validation_alias_bound),
  "validation paths must bind once to the resolved output directory"
)
record_hardening_true(
  direct_pair_matches(
    validation_alias_retarget_paths, validation_alias_retarget_hashes
  ),
  "validation alias retarget must not mutate the retargeted directory"
)
unlink(validation_alias, recursive = FALSE, force = TRUE)
unlink(
  c(validation_alias_original, validation_alias_retarget),
  recursive = TRUE, force = TRUE
)

publication_alias_original <- tempfile("direct-ci-publication-original-")
publication_alias_retarget <- tempfile("direct-ci-publication-retarget-")
publication_alias <- tempfile("direct-ci-publication-alias-")
dir.create(publication_alias_original, recursive = TRUE)
publication_alias_original_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  publication_alias_original, kind = "full_shadow"
)
publication_alias_retarget_paths <- copy_direct_pair(
  direct_paths, publication_alias_retarget
)
publication_alias_retarget_hashes <- direct_pair_hashes(
  publication_alias_retarget_paths
)
assert_true(
  isTRUE(file.symlink(publication_alias_original, publication_alias)),
  "direct-CI publication output alias fixture"
)
publication_alias_bound <- FALSE
record_hardening_error(
  fastkpc_full_cuda_phase3_publish_direct_ci_payload(
    direct_artifact$rows, catalog, publication_alias,
    .transition_hook = function(stage, paths, artifact_lock) {
      if (identical(stage, "after_final_rds_publication")) {
        publication_alias_bound <<- identical(
          dirname(paths$direct_ci_rds),
          normalizePath(publication_alias_original, mustWork = TRUE)
        )
        retarget_output_alias(
          publication_alias,
          normalizePath(publication_alias_retarget, mustWork = TRUE)
        )
        stop("injected publication alias retarget failure", call. = FALSE)
      }
    }
  ),
  "^injected publication alias retarget failure$",
  "publication alias retarget boundary was not reached"
)
record_hardening_true(
  isTRUE(publication_alias_bound),
  "publication paths must bind once to the resolved output directory"
)
record_hardening_true(
  !file.exists(publication_alias_original_paths$direct_ci_rds) &&
    !file.exists(publication_alias_original_paths$direct_ci_summary_json),
  "failed publication through an alias must leave its empty target empty"
)
record_hardening_true(
  direct_pair_matches(
    publication_alias_retarget_paths, publication_alias_retarget_hashes
  ),
  "publication alias retarget must not mutate the retargeted directory"
)
unlink(publication_alias, recursive = FALSE, force = TRUE)
unlink(
  c(publication_alias_original, publication_alias_retarget),
  recursive = TRUE, force = TRUE
)

rollback_output <- tempfile("direct-ci-publication-rollback-")
rollback_paths <- copy_direct_pair(direct_paths, rollback_output)
rollback_hashes <- direct_pair_hashes(rollback_paths)
rollback_rows <- direct_artifact$rows
rollback_row <- 1L
rollback_candidate <- if (
  identical(rollback_rows$candidate_decision[[rollback_row]], "dependent")
) {
  if (rollback_rows$candidate_p_value[[rollback_row]] != 0) {
    0
  } else {
    rollback_rows$alpha[[rollback_row]]
  }
} else if (rollback_rows$candidate_p_value[[rollback_row]] != 1) {
  1
} else {
  (1 + rollback_rows$alpha[[rollback_row]]) / 2
}
rollback_rows$candidate_p_value[[rollback_row]] <- rollback_candidate
rollback_rows$absolute_p_value_difference[[rollback_row]] <- abs(
  rollback_candidate - rollback_rows$reference_p_value[[rollback_row]]
)
rollback_rows$candidate_decision[[rollback_row]] <-
  rollback_rows$reference_decision[[rollback_row]]
rollback_rows$decision_flip[[rollback_row]] <- FALSE
assert_true(
  !identical(
    fastkpc_full_cuda_census_frame_hash(rollback_rows),
    direct_artifact$rows_sha256
  ),
  "direct-CI rollback probe must stage a distinct valid payload"
)
record_hardening_error(
  fastkpc_full_cuda_phase3_publish_direct_ci_payload(
    rollback_rows, catalog, rollback_output,
    .transition_hook = function(stage, paths, artifact_lock) {
      if (identical(stage, "after_final_rds_publication")) {
        stop("injected update failure after final direct_ci.rds",
             call. = FALSE)
      }
    }
  ),
  "^injected update failure after final direct_ci.rds$",
  "pre-existing pair rollback boundary was not reached"
)
record_hardening_true(
  direct_pair_matches(rollback_paths, rollback_hashes),
  "failed direct-CI update must restore the complete prior pair"
)
rollback_validation <- tryCatch(
  fastkpc_full_cuda_phase3_validate_direct_ci_payload(
    rollback_output, catalog
  ),
  error = identity
)
record_hardening_true(
  !inherits(rollback_validation, "error") &&
    isTRUE(rollback_validation$authenticated),
  "restored direct-CI pair must remain authenticated"
)
unlink(rollback_output, recursive = TRUE, force = TRUE)

interrupt_rollback_output <- tempfile("direct-ci-interrupt-rollback-")
interrupt_rollback_paths <- copy_direct_pair(
  direct_paths, interrupt_rollback_output
)
interrupt_rollback_hashes <- direct_pair_hashes(interrupt_rollback_paths)
interrupt_cleanup_calls <- 0L
interrupt_staging_paths <- character()
synthetic_cleanup_interrupt <- structure(
  list(message = "synthetic direct-CI rollback cleanup interrupt", call = NULL),
  class = c("interrupt", "condition")
)
interrupt_rollback_condition <- tryCatch(
  fastkpc_full_cuda_phase3_publish_direct_ci_payload(
    rollback_rows, catalog, interrupt_rollback_output,
    .transition_hook = function(stage, paths, artifact_lock) {
      if (identical(stage, "after_final_rds_publication")) {
        stop("trigger interrupt-safe direct-CI rollback", call. = FALSE)
      }
    },
    .cleanup_function = function(paths, recursive = FALSE, force = FALSE) {
      interrupt_cleanup_calls <<- interrupt_cleanup_calls + 1L
      if (any(grepl(
            "phase3-direct-ci-stage", basename(paths), fixed = TRUE
          ))) {
        interrupt_staging_paths <<- c(interrupt_staging_paths, paths)
      }
      if (identical(interrupt_cleanup_calls, 3L)) {
        stop(synthetic_cleanup_interrupt)
      }
      unlink(paths, recursive = recursive, force = force)
    }
  ),
  interrupt = function(condition) condition,
  error = function(condition) condition
)
record_hardening_true(
  inherits(interrupt_rollback_condition, "interrupt") &&
    identical(
      conditionMessage(interrupt_rollback_condition),
      "synthetic direct-CI rollback cleanup interrupt"
    ),
  "rollback cleanup interrupt must propagate at the publisher boundary"
)
record_hardening_true(
  interrupt_cleanup_calls >= 6L,
  "rollback cleanup interrupt must not skip later cleanup operations"
)
record_hardening_true(
  direct_pair_matches(interrupt_rollback_paths, interrupt_rollback_hashes),
  "rollback cleanup interrupt must restore the hash-exact prior pair"
)
interrupt_lock_path <-
  .fastkpc_full_cuda_phase3_direct_artifact_lock_path(
    interrupt_rollback_output
  )
interrupt_registry <- .fastkpc_full_cuda_phase3_lock_registry()
interrupt_registry_clear <- !exists(
  interrupt_lock_path, envir = interrupt_registry, inherits = FALSE
)
record_hardening_true(
  interrupt_registry_clear,
  "rollback cleanup interrupt must clear direct lock registry ownership"
)
interrupt_output_resolved <- normalizePath(
  interrupt_rollback_output, mustWork = TRUE
)
interrupt_stage_prefix <- paste0(
  ".", basename(interrupt_output_resolved), ".phase3-direct-ci-stage-"
)
interrupt_stage_entries <- list.files(
  dirname(interrupt_output_resolved), all.files = TRUE, no.. = TRUE,
  full.names = TRUE
)
interrupt_stage_entries <- interrupt_stage_entries[startsWith(
  basename(interrupt_stage_entries), interrupt_stage_prefix
)]
record_hardening_true(
  length(interrupt_staging_paths) > 0L &&
    !any(vapply(
      interrupt_staging_paths,
      .fastkpc_full_cuda_phase3_direct_ci_path_present,
      logical(1L)
    )) && length(interrupt_stage_entries) == 0L,
  "rollback cleanup interrupt must remove staging directories and locks"
)

# Restore test-process ownership after the expected RED implementation leak.
if (!interrupt_registry_clear && exists(
      interrupt_lock_path, envir = interrupt_registry, inherits = FALSE
    )) {
  interrupt_entry <- get(
    interrupt_lock_path, envir = interrupt_registry, inherits = FALSE
  )
  .fastkpc_full_cuda_phase3_release_direct_artifact_lock(list(
    lock_path = interrupt_lock_path,
    purpose = "direct_artifact",
    state = interrupt_entry$state
  ))
}
unlink(interrupt_stage_entries, recursive = TRUE, force = TRUE)
unlink(interrupt_rollback_output, recursive = TRUE, force = TRUE)

cleanup_failure_output <- tempfile("direct-ci-cleanup-failure-")
cleanup_failure_stage <- character()
record_hardening_error(
  fastkpc_full_cuda_phase3_publish_direct_ci_payload(
    direct_artifact$rows, catalog, cleanup_failure_output,
    .cleanup_function = function(paths, recursive = FALSE, force = FALSE) {
      if (length(paths) == 1L && dir.exists(paths) &&
          grepl("phase3-direct-ci-stage", basename(paths), fixed = TRUE)) {
        cleanup_failure_stage <<- paths
        return(1L)
      }
      unlink(paths, recursive = recursive, force = force)
    }
  ),
  "^direct-CI staging cleanup failed$",
  "staging cleanup failure must fail loudly"
)
record_hardening_true(
  length(cleanup_failure_stage) == 1L &&
    dir.exists(cleanup_failure_stage),
  "injected staging cleanup failure must expose its failed postcondition"
)
cleanup_failure_validation <- tryCatch(
  fastkpc_full_cuda_phase3_validate_direct_ci_payload(
    cleanup_failure_output, catalog
  ),
  error = identity
)
record_hardening_true(
  !inherits(cleanup_failure_validation, "error") &&
    isTRUE(cleanup_failure_validation$authenticated),
  "cleanup reporting must not corrupt a successfully published pair"
)
if (length(cleanup_failure_stage) == 1L) {
  cleanup_failure_lock <-
    .fastkpc_full_cuda_phase3_direct_artifact_lock_path(cleanup_failure_stage)
  unlink(cleanup_failure_stage, recursive = TRUE, force = TRUE)
  unlink(cleanup_failure_lock, force = TRUE)
}

release_failure_output <- tempfile("direct-ci-release-failure-")
release_failure_staging <- character()
record_hardening_error(
  fastkpc_full_cuda_phase3_publish_direct_ci_payload(
    direct_artifact$rows, catalog, release_failure_output,
    .transition_hook = function(stage, paths, artifact_lock) {
      if (identical(stage, "after_final_rds_publication")) {
        stop("injected pre-release publication failure", call. = FALSE)
      }
    },
    .cleanup_function = function(paths, recursive = FALSE, force = FALSE) {
      if (any(grepl(
            "phase3-direct-ci-stage", basename(paths), fixed = TRUE
          ))) {
        release_failure_staging <<- c(release_failure_staging, paths)
      }
      unlink(paths, recursive = recursive, force = force)
    },
    .release_function = function(lock) {
      .fastkpc_full_cuda_phase3_release_direct_artifact_lock(lock)
      stop("injected direct artifact lock release failure", call. = FALSE)
    }
  ),
  "^injected direct artifact lock release failure$",
  "direct artifact lock release failure boundary was not reached"
)
record_hardening_true(
  length(release_failure_staging) > 0L &&
    !any(file.exists(release_failure_staging)) &&
    !any(dir.exists(release_failure_staging)),
  "staging cleanup must finish before direct lock release"
)
record_hardening_true(
  !file.exists(file.path(release_failure_output, "direct_ci.rds")) &&
    !file.exists(file.path(
      release_failure_output, "direct_ci.summary.json"
    )),
  "final rollback must finish before direct lock release"
)

summary_backup <- tempfile("direct-ci-summary-", fileext = ".json")
assert_true(
  file.copy(
    direct_paths$direct_ci_summary_json, summary_backup,
    overwrite = TRUE
  ),
  "direct-CI summary backup"
)
corrupt_hash_summary <- .fastkpc_full_cuda_phase3_read_json(
  direct_paths$direct_ci_summary_json, "direct_ci.summary.json"
)
corrupt_hash_summary$direct_ci_rds_sha256 <- strrep("0", 64L)
.fastkpc_full_cuda_phase3_write_json_exact(
  corrupt_hash_summary, direct_paths$direct_ci_summary_json
)
assert_error(
  fastkpc_full_cuda_phase3_validate_direct_ci_payload(
    direct_output_dir, catalog
  ),
  "direct_ci.rds SHA-256 mismatch",
  "corrupt direct payload hash must fail closed"
)
assert_true(
  file.copy(
    summary_backup, direct_paths$direct_ci_summary_json,
    overwrite = TRUE
  ),
  "direct-CI summary restore"
)

payload_backup <- tempfile("direct-ci-payload-", fileext = ".rds")
assert_true(
  file.copy(direct_paths$direct_ci_rds, payload_backup, overwrite = TRUE),
  "direct-CI payload backup"
)

hostile_replacement <- tempfile("direct-ci-hostile-replacement-", fileext = ".rds")
saveRDS(
  list(hostile_replacement = TRUE), hostile_replacement,
  version = 2, compress = FALSE
)
validation_transition_count <- 0L
record_hardening_error(
  fastkpc_full_cuda_phase3_validate_direct_ci_payload(
    direct_output_dir, catalog,
    .transition_hook = function(stage, paths, artifact_lock) {
      if (identical(stage, "after_capture")) {
        validation_transition_count <<- validation_transition_count + 1L
        if (!.fastkpc_full_cuda_phase3_owns_lock(
              artifact_lock, "direct_artifact"
            )) {
          stop("direct artifact lock was not held during validation capture",
               call. = FALSE)
        }
        if (!file.copy(
              hostile_replacement, paths$direct_ci_rds, overwrite = TRUE
            )) {
          stop("failed to install hostile direct-CI replacement",
               call. = FALSE)
        }
      }
    }
  ),
  "^direct-CI artifact pair changed during validation$",
  "post-capture replacement was not rejected"
)
record_hardening_true(
  identical(validation_transition_count, 1L),
  "post-capture hook did not run exactly once"
)
assert_true(
  file.copy(payload_backup, direct_paths$direct_ci_rds, overwrite = TRUE),
  "direct-CI payload restore after post-capture replacement"
)
unlink(hostile_replacement, force = TRUE)

valid_domain_payload <- readRDS(payload_backup)
p_value_attacks <- data.frame(
  field = rep(c("reference_p_value", "candidate_p_value"), each = 2L),
  value = rep(c(-1, 2), times = 2L),
  stringsAsFactors = FALSE
)
for (attack_index in seq_len(nrow(p_value_attacks))) {
  attack_field <- p_value_attacks$field[[attack_index]]
  attack_value <- as.double(p_value_attacks$value[[attack_index]])
  hostile_payload <- valid_domain_payload
  rows <- hostile_payload$rows
  resulting_decision <- ifelse(
    attack_value > rows$alpha, "independent", "dependent"
  )
  other_decision <- if (identical(attack_field, "reference_p_value")) {
    rows$candidate_decision
  } else {
    rows$reference_decision
  }
  attack_row <- which(resulting_decision == other_decision)[[1L]]
  rows[[attack_field]][[attack_row]] <- attack_value
  rows$absolute_p_value_difference[[attack_row]] <- abs(
    rows$candidate_p_value[[attack_row]] -
      rows$reference_p_value[[attack_row]]
  )
  rows$reference_decision[[attack_row]] <- if (
    rows$reference_p_value[[attack_row]] > rows$alpha[[attack_row]]
  ) "independent" else "dependent"
  rows$candidate_decision[[attack_row]] <- if (
    rows$candidate_p_value[[attack_row]] > rows$alpha[[attack_row]]
  ) "independent" else "dependent"
  rows$decision_flip[[attack_row]] <-
    rows$candidate_decision[[attack_row]] !=
      rows$reference_decision[[attack_row]]
  assert_true(
    !rows$decision_flip[[attack_row]] && identical(
      rows$absolute_p_value_difference[[attack_row]],
      abs(
        rows$candidate_p_value[[attack_row]] -
          rows$reference_p_value[[attack_row]]
      )
    ),
    paste("hostile", attack_field, attack_value, "row must be self-consistent")
  )
  hostile_payload$rows <- rows
  hostile_payload$rows_sha256 <- fastkpc_full_cuda_census_frame_hash(rows)
  .fastkpc_full_cuda_phase3_atomic_write_merged_rds(
    hostile_payload, direct_paths$direct_ci_rds
  )
  hostile_payload_sha256 <- fastkpc_full_cuda_census_file_hash(
    direct_paths$direct_ci_rds
  )
  hostile_summary <- .fastkpc_full_cuda_phase3_direct_ci_summary(
    hostile_payload, hostile_payload_sha256
  )
  .fastkpc_full_cuda_phase3_write_json_exact(
    hostile_summary, direct_paths$direct_ci_summary_json
  )
  record_hardening_error(
    fastkpc_full_cuda_phase3_validate_direct_ci_payload(
      direct_output_dir, catalog
    ),
    "^direct-CI p-values must be finite and in \\[0,1\\]$",
    paste0(attack_field, "=", attack_value, " escaped p-value validation")
  )
}
assert_true(
  file.copy(payload_backup, direct_paths$direct_ci_rds, overwrite = TRUE) &&
    file.copy(
      summary_backup, direct_paths$direct_ci_summary_json,
      overwrite = TRUE
    ),
  "direct-CI artifact restore after p-value domain attacks"
)

wrong_data_payload <- readRDS(direct_paths$direct_ci_rds)
wrong_data_payload$lineage$dataset_matrix_sha256 <- strrep("d", 64L)
wrong_data_payload$lineage_sha256 <-
  fastkpc_full_cuda_census_named_metadata_hash(wrong_data_payload$lineage)
.fastkpc_full_cuda_phase3_atomic_write_merged_rds(
  wrong_data_payload, direct_paths$direct_ci_rds
)
wrong_data_summary <- .fastkpc_full_cuda_phase3_read_json(
  direct_paths$direct_ci_summary_json, "direct_ci.summary.json"
)
wrong_data_summary$direct_ci_rds_sha256 <-
  fastkpc_full_cuda_census_file_hash(direct_paths$direct_ci_rds)
wrong_data_summary$lineage <- wrong_data_payload$lineage
wrong_data_summary$lineage_sha256 <- wrong_data_payload$lineage_sha256
.fastkpc_full_cuda_phase3_write_json_exact(
  wrong_data_summary, direct_paths$direct_ci_summary_json
)
assert_error(
  fastkpc_full_cuda_phase3_validate_direct_ci_payload(
    direct_output_dir, catalog
  ),
  "dataset matrix hash mismatch",
  "wrong direct-CI data hash must fail closed"
)
assert_true(
  file.copy(payload_backup, direct_paths$direct_ci_rds, overwrite = TRUE),
  "direct-CI payload restore after data-hash attack"
)
assert_true(
  file.copy(
    summary_backup, direct_paths$direct_ci_summary_json,
    overwrite = TRUE
  ),
  "direct-CI summary restore after data-hash attack"
)

fallback_payload <- readRDS(direct_paths$direct_ci_rds)
fallback_payload$rows$spectra_fallback[[1L]] <- TRUE
fallback_payload$rows_sha256 <- fastkpc_full_cuda_census_frame_hash(
  fallback_payload$rows
)
.fastkpc_full_cuda_phase3_atomic_write_merged_rds(
  fallback_payload, direct_paths$direct_ci_rds
)
fallback_summary <- .fastkpc_full_cuda_phase3_read_json(
  direct_paths$direct_ci_summary_json, "direct_ci.summary.json"
)
fallback_summary$direct_ci_rds_sha256 <-
  fastkpc_full_cuda_census_file_hash(direct_paths$direct_ci_rds)
.fastkpc_full_cuda_phase3_write_json_exact(
  fallback_summary, direct_paths$direct_ci_summary_json
)
assert_error(
  fastkpc_full_cuda_phase3_validate_direct_ci_payload(
    direct_output_dir, catalog
  ),
  "Spectra fallback is not allowed",
  "direct-CI backend fallback row must fail closed"
)
assert_true(
  file.copy(payload_backup, direct_paths$direct_ci_rds, overwrite = TRUE),
  "direct-CI payload restore after fallback attack"
)
assert_true(
  file.copy(
    summary_backup, direct_paths$direct_ci_summary_json,
    overwrite = TRUE
  ),
  "direct-CI summary restore after fallback attack"
)

wrong_row_lineage <- readRDS(direct_paths$direct_ci_rds)
original_x <- wrong_row_lineage$rows$x[[1L]]
wrong_row_lineage$rows$x[[1L]] <- wrong_row_lineage$rows$y[[1L]]
wrong_row_lineage$rows$y[[1L]] <- original_x
wrong_row_lineage$rows_sha256 <- fastkpc_full_cuda_census_frame_hash(
  wrong_row_lineage$rows
)
.fastkpc_full_cuda_phase3_atomic_write_merged_rds(
  wrong_row_lineage, direct_paths$direct_ci_rds
)
wrong_row_summary <- .fastkpc_full_cuda_phase3_read_json(
  direct_paths$direct_ci_summary_json, "direct_ci.summary.json"
)
wrong_row_summary$direct_ci_rds_sha256 <-
  fastkpc_full_cuda_census_file_hash(direct_paths$direct_ci_rds)
wrong_row_summary$rows_sha256 <- wrong_row_lineage$rows_sha256
.fastkpc_full_cuda_phase3_write_json_exact(
  wrong_row_summary, direct_paths$direct_ci_summary_json
)
assert_error(
  fastkpc_full_cuda_phase3_validate_direct_ci_payload(
    direct_output_dir, catalog
  ),
  "Phase 1 direct row lineage mismatch",
  "self-consistently re-signed direct logical row drift must fail closed"
)
assert_true(
  file.copy(payload_backup, direct_paths$direct_ci_rds, overwrite = TRUE) &&
    file.copy(
      summary_backup, direct_paths$direct_ci_summary_json,
      overwrite = TRUE
    ),
  "direct-CI artifact restore after logical-lineage attack"
)
invisible(fastkpc_full_cuda_phase3_validate_direct_ci_payload(
  direct_output_dir, catalog
))

assert_true(
  length(hardening_failures) == 0L,
  paste(
    c("direct-CI hardening expectations failed:", hardening_failures),
    collapse = "\n- "
  )
)

cat("full CUDA CI fixed-sp shadow mapping: PASS\n")
