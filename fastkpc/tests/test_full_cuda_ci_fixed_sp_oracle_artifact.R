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

ieee_double_bytes <- function(value) {
  writeBin(as.double(value), raw(), size = 8L, endian = "little")
}
ieee_double_equal <- function(actual, expected) {
  length(actual) == length(expected) && identical(
    ieee_double_bytes(actual), ieee_double_bytes(expected)
  )
}

exact_double_values <- c(
  simple = 0.2,
  canonical_integer = 10,
  canonical_large_scientific = 1e5,
  canonical_small_scientific = 1e-4,
  pi = pi,
  minimum_normal = .Machine$double.xmin,
  minimum_subnormal = 5e-324,
  adjacent_to_one = 1 + .Machine$double.eps,
  tiny = 5e-30,
  json_parser_boundary = -6720.151716784982,
  positive_zero = 0.0,
  negative_zero = -0.0,
  missing = NA_real_,
  not_a_number = NaN
)
exact_csv_frame <- data.frame(
  label = names(exact_double_values),
  value = unname(exact_double_values),
  stringsAsFactors = FALSE
)
exact_csv_path <- tempfile("phase3-exact-double-", fileext = ".csv")
on.exit(unlink(exact_csv_path, force = TRUE), add = TRUE)
.fastkpc_full_cuda_phase3_write_csv(exact_csv_frame, exact_csv_path)
exact_csv_roundtrip <- .fastkpc_full_cuda_phase3_read_csv(
  exact_csv_path, "exact-double.csv"
)
.fastkpc_full_cuda_phase3_coerce_csv_like(
  exact_csv_roundtrip, exact_csv_frame, "exact-double"
)
exact_csv_lines <- readLines(exact_csv_path, warn = FALSE)
positive_zero_csv_line <- match(
  "positive_zero", names(exact_double_values)
) + 1L
negative_zero_csv_line <- match(
  "negative_zero", names(exact_double_values)
) + 1L
assert_true(
  identical(exact_csv_lines[[2L]], '"simple",0.2') &&
    identical(exact_csv_lines[[3L]], '"canonical_integer",10') &&
    identical(
      exact_csv_lines[[4L]],
      '"canonical_large_scientific",1e+05'
    ) && identical(
      exact_csv_lines[[5L]],
      '"canonical_small_scientific",1e-04'
    ) &&
    identical(
      tail(exact_csv_lines, 2L),
      c('"missing",NA', '"not_a_number",NA')
    ) && identical(
      exact_csv_lines[[positive_zero_csv_line]],
      '"positive_zero",0'
    ) && identical(
      exact_csv_lines[[negative_zero_csv_line]],
      '"negative_zero",-0.0'
    ) && !identical(
      ieee_double_bytes(0.0), ieee_double_bytes(-0.0)
    ) && ieee_double_equal(
      exact_csv_roundtrip$value[c(
        positive_zero_csv_line, negative_zero_csv_line
      ) - 1L],
      c(0.0, -0.0)
    ),
  "exact CSV keeps canonical numeric encoding and the NA/NaN policy"
)

exact_json_values <- as.list(
  exact_double_values[seq_len(length(exact_double_values) - 2L)]
)
exact_json_path <- tempfile("phase3-exact-double-", fileext = ".json")
on.exit(unlink(exact_json_path, force = TRUE), add = TRUE)
.fastkpc_full_cuda_phase3_write_json_exact(
  exact_json_values, exact_json_path
)
exact_json_roundtrip <- jsonlite::read_json(
  exact_json_path, simplifyVector = FALSE
)
assert_true(
  all(vapply(names(exact_json_values), function(field) {
    ieee_double_equal(
      as.double(exact_json_roundtrip[[field]]),
      exact_json_values[[field]]
    )
  }, logical(1L))) && ieee_double_equal(
    jsonlite::fromJSON("-0.0"), -0.0
  ) && any(grepl(
    '"simple": 0.2', readLines(exact_json_path, warn = FALSE), fixed = TRUE
  )) && any(grepl(
    '"negative_zero": -0.0',
    readLines(exact_json_path, warn = FALSE), fixed = TRUE
  )),
  "exact JSON preserves finite doubles without changing canonical spellings"
)

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

runner_source <- readLines(
  "fastkpc/tools/run_full_cuda_ci_fixed_sp_oracle_sp.R", warn = FALSE
)
runner_runtime_line <- which(grepl(
  "^runtime_create <- function", runner_source
))[[1L]]
runner_preflight <- runner_source[seq_len(runner_runtime_line - 1L)]
assert_true(
  any(grepl(
    "fastkpc_full_cuda_phase3_publish_oracle_artifact", runner_preflight,
    fixed = TRUE
  )) && !any(grepl("unlink(", runner_preflight, fixed = TRUE)),
  "runner delegates completion recovery to the locked publisher before CUDA"
)

large_group_count <- 12000L
large_setup_keys <- sprintf("%064x", seq_len(large_group_count))
large_target_setup_keys <- rep(large_setup_keys, each = 3L)
large_groups <- .fastkpc_full_cuda_phase3_oracle_target_groups(
  large_setup_keys, large_target_setup_keys
)
assert_identical(
  large_groups$mapped_setup,
  rep(seq_len(large_group_count), each = 3L),
  "large oracle target grouping maps each row once"
)
assert_identical(
  large_groups$target_count, rep.int(3L, large_group_count),
  "large oracle target grouping tabulates setup counts"
)
assert_identical(
  large_groups$first_index,
  as.integer(seq.int(1L, 3L * large_group_count, by = 3L)),
  "large oracle target grouping records setup starts"
)
assert_identical(
  large_groups$target_ordinal, rep.int(1:3, large_group_count),
  "large oracle target grouping derives target ordinals"
)
authority_validator_source <- paste(deparse(body(
  .fastkpc_full_cuda_phase3_validate_oracle_row_authority
)), collapse = "\n")
assert_true(
  !grepl("for (setup_index", authority_validator_source, fixed = TRUE) &&
    !grepl(
      "target_parity$prepared_s_key_sha256 == key",
      authority_validator_source, fixed = TRUE
    ),
  "oracle row authority uses mapped linear group reductions"
)

assert_true(.Platform$OS.type == "unix",
            "Phase 3 advisory-lock process tests require Unix fork")
lock_inode <- function(path) {
  output <- system2(
    "stat", c("-c", "%i", path), stdout = TRUE, stderr = TRUE
  )
  assert_true(is.null(attr(output, "status")) && length(output) == 1L,
              "lock inode is readable")
  output[[1L]]
}
lock_evidence_hash <- function(path) {
  fastkpc_full_cuda_census_file_hash(path)
}
lock_children <- new.env(parent = emptyenv())
on.exit({
  for (name in ls(lock_children, all.names = TRUE)) {
    child <- lock_children[[name]]
    try(tools::pskill(child$job$pid, 9L), silent = TRUE)
    try(suppressWarnings(parallel::mccollect(child$job)), silent = TRUE)
    unlink(c(child$ready_path, child$release_path), force = TRUE)
  }
}, add = TRUE)
start_lock_child <- function(lock_case, output_dir) {
  ready_path <- tempfile("phase3-lock-child-ready-")
  release_path <- tempfile("phase3-lock-child-release-")
  job <- parallel::mcparallel({
    lock <- lock_case$acquire(output_dir)
    writeLines(as.character(Sys.getpid()), ready_path, useBytes = TRUE)
    while (!file.exists(release_path)) Sys.sleep(0.01)
    lock_case$release(lock)
    TRUE
  }, silent = TRUE)
  child <- list(
    job = job, ready_path = ready_path, release_path = release_path
  )
  lock_children[[as.character(job$pid)]] <- child
  deadline <- proc.time()[["elapsed"]] + 10
  while (!file.exists(ready_path) &&
         proc.time()[["elapsed"]] < deadline) {
    completed <- parallel::mccollect(job, wait = FALSE)
    if (!is.null(completed)) {
      fail("lock child exited before signaling readiness")
    }
    Sys.sleep(0.01)
  }
  assert_true(file.exists(ready_path), "lock child signals readiness")
  child$pid <- as.integer(readLines(ready_path, warn = FALSE)[[1L]])
  assert_true(
    child$pid == job$pid && child$pid != Sys.getpid(),
    "lock owner is an independent child process"
  )
  child
}
finish_lock_child <- function(child, crash = FALSE) {
  if (isTRUE(crash)) {
    tools::pskill(child$job$pid, 9L)
  } else {
    writeLines("release", child$release_path, useBytes = TRUE)
  }
  result <- suppressWarnings(parallel::mccollect(child$job))
  rm(list = as.character(child$job$pid), envir = lock_children)
  unlink(c(child$ready_path, child$release_path), force = TRUE)
  if (!isTRUE(crash)) {
    assert_true(identical(unname(result), list(TRUE)),
                "lock child releases normally")
  }
  invisible(result)
}
inspect_inherited_lock_in_child <- function(
    lock_case, output_dir, inherited_lock) {
  parent_pid <- as.integer(Sys.getpid())
  job <- parallel::mcparallel({
    inherited_owns <- lock_case$owns(inherited_lock)
    acquired_lock <- NULL
    acquisition_error <- tryCatch({
      acquired_lock <- lock_case$acquire(output_dir)
      NULL
    }, error = function(error) conditionMessage(error))
    acquired <- !is.null(acquired_lock)
    if (isTRUE(acquired)) lock_case$release(acquired_lock)
    registry <- .fastkpc_full_cuda_phase3_lock_registry()
    inherited_entry_preserved <- exists(
      inherited_lock$lock_path, envir = registry, inherits = FALSE
    )
    if (isTRUE(inherited_entry_preserved)) {
      entry <- get(
        inherited_lock$lock_path, envir = registry, inherits = FALSE
      )
      inherited_entry_preserved <-
        identical(entry$owner_pid, parent_pid) &&
        identical(entry$state, inherited_lock$state)
    }
    lock_case$release(inherited_lock)
    list(
      pid = as.integer(Sys.getpid()),
      inherited_owns = inherited_owns,
      acquired = acquired,
      inherited_entry_preserved = inherited_entry_preserved,
      inherited_released = inherited_lock$state$released,
      error = acquisition_error
    )
  }, silent = TRUE)
  result <- suppressWarnings(parallel::mccollect(job))
  assert_true(
    length(result) == 1L && is.list(result[[1L]]) &&
      result[[1L]]$pid != parent_pid,
    "lock attempt executes in an independent child"
  )
  result[[1L]]
}
attempt_os_lock_in_rscript <- function(lock_path) {
  script <- tempfile("phase3-filelock-child-", fileext = ".R")
  on.exit(unlink(script, force = TRUE), add = TRUE)
  writeLines(c(
    "args <- commandArgs(trailingOnly = TRUE)",
    "handle <- filelock::lock(args[[1L]], exclusive = TRUE, timeout = 0)",
    "if (is.null(handle)) {",
    "  cat('blocked')",
    "} else {",
    "  filelock::unlock(handle)",
    "  cat('acquired')",
    "}"
  ), script, useBytes = TRUE)
  result <- system2(
    file.path(R.home("bin"), "Rscript"), c(script, lock_path),
    stdout = TRUE, stderr = TRUE
  )
  assert_true(is.null(attr(result, "status")) && length(result) == 1L,
              "independent Rscript lock attempt completes")
  result[[1L]]
}
lock_cases <- list(
  artifact = list(
    purpose = "oracle_artifact",
    path = .fastkpc_full_cuda_phase3_oracle_artifact_lock_path,
    acquire = .fastkpc_full_cuda_phase3_acquire_oracle_artifact_lock,
    owns = .fastkpc_full_cuda_phase3_owns_oracle_artifact_lock,
    release = .fastkpc_full_cuda_phase3_release_oracle_artifact_lock
  ),
  runner = list(
    purpose = "shard_runner",
    path = .fastkpc_full_cuda_phase3_shard_runner_lock_path,
    acquire = .fastkpc_full_cuda_phase3_acquire_shard_runner_lock,
    owns = function(lock) {
      .fastkpc_full_cuda_phase3_owns_lock(lock, "shard_runner")
    },
    release = .fastkpc_full_cuda_phase3_release_shard_runner_lock
  )
)
registry_option_name <- "fastkpc.phase3.lock_registry.v1"
registry_schema_version <- "full-cuda-ci-phase3-lock-registry-v1"
resource_output <- tempfile("phase3-runner-registry-resource-")
resource_lock <-
  .fastkpc_full_cuda_phase3_acquire_shard_runner_lock(resource_output)
on.exit(
  try(
    .fastkpc_full_cuda_phase3_release_shard_runner_lock(resource_lock),
    silent = TRUE
  ),
  add = TRUE
)
registry_before_resource <- .fastkpc_full_cuda_phase3_lock_registry()
source("fastkpc/R/full_cuda_ci_phase3_artifacts.R", local = .GlobalEnv)
registry_after_resource <- .fastkpc_full_cuda_phase3_lock_registry()
resource_lock_function_calls <- 0L
resource_duplicate_error <- tryCatch({
  .fastkpc_full_cuda_phase3_acquire_lock(
    lock_path = resource_lock$lock_path, purpose = "shard_runner",
    .lock_function = function(path, exclusive, timeout) {
      resource_lock_function_calls <<- resource_lock_function_calls + 1L
      filelock::lock(path, exclusive = exclusive, timeout = timeout)
    }
  )
  NULL
}, error = function(error) error)
assert_true(
  identical(registry_after_resource, registry_before_resource) &&
    identical(getOption(registry_option_name), registry_before_resource) &&
    identical(
      attr(registry_before_resource, "schema_version", exact = TRUE),
      registry_schema_version
    ) && .fastkpc_full_cuda_phase3_owns_lock(
      resource_lock, "shard_runner"
    ) && inherits(resource_duplicate_error, "error") && grepl(
      "already held by this process",
      conditionMessage(resource_duplicate_error), fixed = TRUE
    ) && resource_lock_function_calls == 0L &&
    identical(
      attempt_os_lock_in_rscript(resource_lock$lock_path), "blocked"
    ),
  "runner lock registry survives module re-source"
)
.fastkpc_full_cuda_phase3_release_shard_runner_lock(resource_lock)
resource_reacquired <-
  .fastkpc_full_cuda_phase3_acquire_shard_runner_lock(resource_output)
.fastkpc_full_cuda_phase3_release_shard_runner_lock(resource_reacquired)

corruption_lock <-
  .fastkpc_full_cuda_phase3_acquire_shard_runner_lock(resource_output)
saved_registry_option <- getOption(registry_option_name)
impostor_registry <- new.env(hash = TRUE, parent = emptyenv())
attr(impostor_registry, "schema_version") <- registry_schema_version
options(structure(list(impostor_registry), names = registry_option_name))
corruption_error <- tryCatch({
  .fastkpc_full_cuda_phase3_lock_registry()
  NULL
}, error = function(error) error)
options(structure(list(saved_registry_option), names = registry_option_name))
assert_true(
  inherits(corruption_error, "error") &&
    grepl("registry singleton", conditionMessage(corruption_error),
          fixed = TRUE) &&
    identical(
      .fastkpc_full_cuda_phase3_lock_registry(), saved_registry_option
    ) && .fastkpc_full_cuda_phase3_owns_lock(
      corruption_lock, "shard_runner"
    ),
  "registry option replacement fails closed without losing held state"
)
.fastkpc_full_cuda_phase3_release_shard_runner_lock(corruption_lock)

synthetic_lock_interrupt <- structure(
  list(message = "synthetic Phase 3 lock transition interrupt", call = NULL),
  class = c("interrupt", "condition")
)
for (stage in c("after_os_acquire", "after_registry_assign")) {
  interrupted_output <- tempfile(paste0("phase3-lock-", stage, "-"))
  interrupted_path <-
    .fastkpc_full_cuda_phase3_oracle_artifact_lock_path(interrupted_output)
  interrupted_condition <- tryCatch(
    .fastkpc_full_cuda_phase3_acquire_lock(
      lock_path = interrupted_path, purpose = "oracle_artifact",
      .transition_hook = function(actual_stage, lock) {
        if (identical(actual_stage, stage)) stop(synthetic_lock_interrupt)
      }
    ),
    interrupt = function(condition) condition,
    error = function(condition) condition
  )
  normalized_interrupted_path <-
    .fastkpc_full_cuda_phase3_normalize_lock_path(interrupted_path)
  assert_true(
    inherits(interrupted_condition, "interrupt") &&
      !exists(
        normalized_interrupted_path,
        envir = .fastkpc_full_cuda_phase3_lock_registry(),
        inherits = FALSE
      ) && identical(
        attempt_os_lock_in_rscript(normalized_interrupted_path), "acquired"
      ),
    paste(stage, "interrupt cleans incomplete lock acquisition")
  )
}

release_interrupt_output <- tempfile("phase3-lock-release-interrupt-")
release_interrupt_lock <-
  .fastkpc_full_cuda_phase3_acquire_oracle_artifact_lock(
    release_interrupt_output
  )
release_interrupt_condition <- tryCatch(
  .fastkpc_full_cuda_phase3_release_lock(
    release_interrupt_lock, "oracle_artifact",
    .transition_hook = function(stage, lock) {
      if (identical(stage, "after_os_unlock")) {
        stop(synthetic_lock_interrupt)
      }
    }
  ),
  interrupt = function(condition) condition,
  error = function(condition) condition
)
assert_true(
  inherits(release_interrupt_condition, "interrupt") &&
    identical(release_interrupt_lock$state$released, TRUE) &&
    is.null(release_interrupt_lock$state$handle) &&
    !exists(
      release_interrupt_lock$lock_path,
      envir = .fastkpc_full_cuda_phase3_lock_registry(),
      inherits = FALSE
    ) && identical(
      attempt_os_lock_in_rscript(release_interrupt_lock$lock_path),
      "acquired"
    ),
  "interrupt after OS unlock cannot leave active local ownership"
)
.fastkpc_full_cuda_phase3_release_oracle_artifact_lock(
  release_interrupt_lock
)

for (label in names(lock_cases)) {
  lock_case <- lock_cases[[label]]
  registry_output <- tempfile(paste0("phase3-", label, "-registry-"))
  outer <- lock_case$acquire(registry_output)
  on.exit(try(lock_case$release(outer), silent = TRUE), add = TRUE)
  lock_path <- outer$lock_path
  lock_function_call_count <- 0L
  counting_lock <- function(path, exclusive, timeout) {
    lock_function_call_count <<- lock_function_call_count + 1L
    filelock::lock(path, exclusive = exclusive, timeout = timeout)
  }
  duplicate_error <- tryCatch({
    .fastkpc_full_cuda_phase3_acquire_lock(
      lock_path = lock_path, purpose = lock_case$purpose,
      .lock_function = counting_lock
    )
    NULL
  }, error = function(error) error)
  assert_true(
    inherits(duplicate_error, "error") && grepl(
      "already held by this process", conditionMessage(duplicate_error),
      fixed = TRUE
    ) && lock_function_call_count == 0L,
    paste(label, "same-process duplicate is rejected before OS acquisition")
  )
  alias_path <- file.path(dirname(lock_path), ".", basename(lock_path))
  alias_error <- tryCatch({
    .fastkpc_full_cuda_phase3_acquire_lock(
      lock_path = alias_path, purpose = lock_case$purpose,
      .lock_function = counting_lock
    )
    NULL
  }, error = function(error) error)
  assert_true(
    inherits(alias_error, "error") && grepl(
      "already held by this process", conditionMessage(alias_error),
      fixed = TRUE
    ) && lock_function_call_count == 0L,
    paste(label, "normalized lock-path aliases collide before OS acquisition")
  )

  inherited_child <- inspect_inherited_lock_in_child(
    lock_case, registry_output, outer
  )
  assert_true(
    identical(inherited_child$inherited_owns, FALSE) &&
      identical(inherited_child$acquired, FALSE) &&
      isTRUE(inherited_child$inherited_entry_preserved) &&
      identical(inherited_child$inherited_released, FALSE) &&
      grepl(
        "active inherited parent lock", inherited_child$error,
        fixed = TRUE
      ) &&
      identical(attempt_os_lock_in_rscript(lock_path), "blocked") &&
      isTRUE(lock_case$owns(outer)),
    paste(
      label,
      "fork rejects and preserves an active inherited parent lock"
    )
  )

  lock_registry <- .fastkpc_full_cuda_phase3_lock_registry()
  registry_entry <- get(lock_path, envir = lock_registry, inherits = FALSE)
  assert_true(
    identical(registry_entry$owner_pid, as.integer(Sys.getpid())) &&
      identical(registry_entry$state, outer$state),
    paste(label, "registry binds the normalized path to the outer state")
  )
  lock_case$release(outer)
  lock_case$release(outer)
  assert_true(
    !exists(lock_path, envir = lock_registry, inherits = FALSE) &&
      !isTRUE(lock_case$owns(outer)),
    paste(label, "exact-state release clears the registry idempotently")
  )
  assert_true(
    identical(attempt_os_lock_in_rscript(lock_path), "acquired"),
    paste(label, "independent child acquires immediately after outer release")
  )

  remapped <- lock_case$acquire(registry_output)
  replacement_state <- new.env(parent = emptyenv())
  assign(
    remapped$lock_path,
    list(owner_pid = as.integer(Sys.getpid()), state = replacement_state),
    envir = lock_registry
  )
  lock_case$release(remapped)
  preserved_entry <- get(
    remapped$lock_path, envir = lock_registry, inherits = FALSE
  )
  assert_true(
    identical(preserved_entry$state, replacement_state) &&
      identical(remapped$state$released, TRUE),
    paste(label, "release never removes a registry entry for another state")
  )
  rm(list = remapped$lock_path, envir = lock_registry)
  final_lock <- lock_case$acquire(registry_output)
  lock_case$release(final_lock)
}
for (purpose in names(lock_cases)) {
  lock_case <- lock_cases[[purpose]]
  lock_output <- tempfile(paste0("phase3-", purpose, "-os-lock-"))
  lock_path <- lock_case$path(lock_output)
  child <- start_lock_child(lock_case, lock_output)
  inode_before <- lock_inode(lock_path)
  evidence_before <- lock_evidence_hash(lock_path)
  assert_error(
    lock_case$acquire(lock_output),
    paste("live child excludes parent", purpose, "lock")
  )
  assert_true(
    identical(lock_inode(lock_path), inode_before) &&
      identical(lock_evidence_hash(lock_path), evidence_before),
    paste("failed parent acquisition preserves", purpose, "lock evidence")
  )
  finish_lock_child(child, crash = TRUE)
  recovered <- lock_case$acquire(lock_output)
  on.exit(try(lock_case$release(recovered), silent = TRUE), add = TRUE)
  invisible(gc())
  assert_true(
    file.exists(lock_path) && !dir.exists(lock_path) &&
      identical(lock_inode(lock_path), inode_before) &&
      isTRUE(lock_case$owns(recovered)),
    paste(
      purpose,
      "lock is immediately recovered after SIGKILL without inode replacement"
    )
  )
  lock_case$release(recovered)
  lock_case$release(recovered)
  assert_true(
    file.exists(lock_path) && !dir.exists(lock_path) &&
      identical(lock_inode(lock_path), inode_before) &&
      !isTRUE(lock_case$owns(recovered)),
    paste(purpose, "lock release is idempotent and keeps its inode")
  )
}

unavailable_namespace <- function(package, quietly = FALSE) FALSE
for (purpose in names(lock_cases)) {
  lock_case <- lock_cases[[purpose]]
  dependency_output <- tempfile(
    paste0("phase3-", purpose, "-missing-filelock-")
  )
  error <- tryCatch({
    lock_case$acquire(
      dependency_output, .namespace_checker = unavailable_namespace
    )
    NULL
  }, error = function(error) error)
  assert_true(
    inherits(error, "error") &&
      grepl("filelock", conditionMessage(error), fixed = TRUE) &&
      !dir.exists(dependency_output) &&
      !file.exists(lock_case$path(dependency_output)),
    paste("missing filelock fails before", purpose, "output mutation")
  )
}

exactness_schema <- fastkpc_full_cuda_fixed_sp_qualification_dcov_schema()
exactness_diagnostic <- list(
  n = 351L, numCol = 35L, index = 1, lowrank_mode = "spectra",
  lowrank_full_eig_count = 0L, lowrank_spectra_count = 2L,
  lowrank_spectra_converged_count = 2L,
  lowrank_spectra_failed_count = 0L,
  lowrank_spectra_fallback_full_eig_count = 0L,
  lowrank_spectra_iterations = 4L, lowrank_spectra_nconv = 70L,
  lowrank_spectra_ncv = 71L, lowrank_spectra_tol = 1e-10,
  lowrank_spectra_matvec_count = 0L
)
exactness_columns <- lapply(exactness_schema$names, function(field) {
  switch(
    exactness_schema$types[[field]],
    character = "fixture", integer = 1L, double = 0,
    logical = FALSE, list = I(list(character())),
    fail("unsupported qualification exactness fixture type")
  )
})
names(exactness_columns) <- exactness_schema$names
exactness_columns$parity_scope <- "qualification"
exactness_columns$residual_key_x <- sha("qualification-exactness-x")
exactness_columns$residual_key_y <- sha("qualification-exactness-y")
exactness_columns$reference_p_value <- 0.5
exactness_columns$alpha <- 0.05
exactness_columns$reference_decision <- "independent"
exactness_columns$reference_independent <- TRUE
exactness_columns$index <- 1L
exactness_columns$numCol <- 35L
exactness_columns$backend <- "cpp"
exactness_columns$low_rank_backend <- "spectra"
exactness_columns$p_value <- exactness_columns$reference_p_value
exactness_columns$p_value_difference <- 0
exactness_columns$absolute_p_value_difference <- 0
exactness_columns$p_value_exact <- TRUE
exactness_columns$signed_alpha_distance <-
  exactness_columns$p_value - exactness_columns$alpha
exactness_columns$decision <- "independent"
exactness_columns$independent <- TRUE
exactness_columns$decision_flip <- FALSE
exactness_columns$backend_error <- FALSE
exactness_columns$spectra_fallback <- FALSE
exactness_columns$selection_reasons <- I(list(character()))
exactness_columns$diagnostics <- I(list(exactness_diagnostic))
exactness_records <- structure(
  exactness_columns, class = "data.frame", row.names = 1L
)
exactness_logical <- exactness_records[exactness_schema$logical_names]
exactness_target_keys <- sort(c(
  exactness_records$residual_key_x, exactness_records$residual_key_y
), method = "radix")
exactness_summary <-
  fastkpc_full_cuda_fixed_sp_summarize_qualification_dcov_records(
    exactness_records, exactness_logical, exactness_target_keys
  )
assert_identical(
  exactness_summary$qualification_dcov_logical_test_count, 1L,
  "valid production qualification exactness row summarizes"
)
forged_exactness_records <- exactness_records
forged_exactness_records$p_value_exact[[1L]] <- FALSE
assert_error(
  fastkpc_full_cuda_fixed_sp_summarize_qualification_dcov_records(
    forged_exactness_records, exactness_logical, exactness_target_keys
  ),
  "qualification summary rejects a forged p_value_exact flag"
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
  target_parity$rhs_relative_l2 <- rep(0.0, target_count)
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
binary_a <- sha("phase3-oracle-executed-native-binary-a")
binary_b <- sha("phase3-oracle-executed-native-binary-b")
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
  runtime_destroy = runtime_destroy, scope = "iteration", shard_count = 4L,
  executed_native_library_sha256 = binary_a
)
assert_identical(run$written_shard_ids, 0:3,
                 "fixture writes four authenticated shards")
binary_pure_resume <- fastkpc_full_cuda_phase3_run_shards(
  output_dir = output_dir, kind = "oracle_sp",
  setup_keys = setup_keys, target_rows = target_rows,
  identity = identity, route_config = route_config,
  executor = executor, runtime_create = runtime_create,
  runtime_destroy = runtime_destroy, scope = "iteration", shard_count = 4L,
  executed_native_library_sha256 = binary_b
)
assert_identical(
  binary_pure_resume$reused_shard_ids, 0:3,
  "complete binary-A shard set is reusable under current binary B"
)
assert_identical(
  lifecycle$create_count, 1L,
  "complete cross-build pure resume creates no runtime context"
)

binary_mix_dir <- tempfile("phase3-oracle-binary-mix-")
on.exit(unlink(binary_mix_dir, recursive = TRUE, force = TRUE), add = TRUE)
binary_mix_lifecycle <- new.env(parent = emptyenv())
binary_mix_lifecycle$create_count <- 0L
binary_mix_create <- function() {
  binary_mix_lifecycle$create_count <- binary_mix_lifecycle$create_count + 1L
  new.env(parent = emptyenv())
}
binary_mix_destroy <- function(context) invisible(NULL)
binary_partial <- fastkpc_full_cuda_phase3_run_shards(
  output_dir = binary_mix_dir, kind = "oracle_sp",
  setup_keys = setup_keys, target_rows = target_rows,
  identity = identity, route_config = route_config,
  executor = executor, runtime_create = binary_mix_create,
  runtime_destroy = binary_mix_destroy, scope = "iteration", shard_count = 4L,
  stop_after = 1L, executed_native_library_sha256 = binary_a
)
assert_identical(binary_partial$written_shard_ids, 0L,
                 "binary-A partial run writes one shard")
binary_partial_paths <- c(
  list.files(file.path(binary_mix_dir, "shards"), full.names = TRUE),
  list.files(file.path(binary_mix_dir, "sessions"), full.names = TRUE)
)
binary_partial_hashes <- vapply(
  binary_partial_paths, fastkpc_full_cuda_census_file_hash, character(1L)
)
assert_error(
  fastkpc_full_cuda_phase3_run_shards(
    output_dir = binary_mix_dir, kind = "oracle_sp",
    setup_keys = setup_keys, target_rows = target_rows,
    identity = identity, route_config = route_config,
    executor = executor, runtime_create = binary_mix_create,
    runtime_destroy = binary_mix_destroy, scope = "iteration", shard_count = 4L,
    executed_native_library_sha256 = binary_b
  ),
  "partial binary-A shards reject current binary B before runtime creation"
)
assert_identical(
  binary_mix_lifecycle$create_count, 1L,
  "binary mismatch rejection occurs before runtime creation"
)
assert_identical(
  vapply(
    binary_partial_paths, fastkpc_full_cuda_census_file_hash, character(1L)
  ),
  binary_partial_hashes,
  "binary mismatch preserves existing shard and session evidence"
)

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
interrupting_artifact_acquire <-
  .fastkpc_full_cuda_phase3_acquire_oracle_artifact_lock
artifact_handoff_calls <- list(
  validator = function() {
    fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
      output_dir, expected_identity = identity, require_full = FALSE
    )
  },
  publisher = function() {
    publish(
      output_dir = output_dir, setup_keys = setup_keys,
      target_rows = target_rows, identity = identity,
      route_config = route_config, scope = "iteration", shard_count = 4L,
      risk_rows = risk_rows, qualification_dcov = qualification_dcov,
      command_lines = paste0(
        "Rscript ",
        "fastkpc/tests/test_full_cuda_ci_fixed_sp_oracle_artifact.R"
      )
    )
  }
)
for (caller in names(artifact_handoff_calls)) {
  assign(
    ".fastkpc_full_cuda_phase3_acquire_oracle_artifact_lock",
    function(...) {
      lock <- interrupting_artifact_acquire(...)
      tools::pskill(Sys.getpid(), 2L)
      for (index in seq_len(100000L)) sqrt(index)
      lock
    },
    envir = .GlobalEnv
  )
  handoff_condition <- tryCatch(
    artifact_handoff_calls[[caller]](),
    interrupt = function(condition) condition,
    error = function(condition) condition
  )
  assign(
    ".fastkpc_full_cuda_phase3_acquire_oracle_artifact_lock",
    interrupting_artifact_acquire,
    envir = .GlobalEnv
  )
  handoff_lock_path <-
    .fastkpc_full_cuda_phase3_oracle_artifact_lock_path(output_dir)
  assert_true(
    inherits(handoff_condition, "interrupt") &&
      !exists(
        handoff_lock_path,
        envir = .fastkpc_full_cuda_phase3_lock_registry(),
        inherits = FALSE
      ) && identical(
        attempt_os_lock_in_rscript(handoff_lock_path), "acquired"
      ),
    paste(caller, "registers artifact cleanup before interrupt delivery")
  )
}
immutable_hash_reader <- get0(
  ".fastkpc_full_cuda_phase3_oracle_immutable_file_hashes",
  mode = "function", inherits = TRUE
)
assert_true(!is.null(immutable_hash_reader),
            "oracle validator exposes immutable file rehashing")
immutable_hash_read_count <- 0L
lock_refresh <- .fastkpc_full_cuda_phase3_refresh_oracle_artifact_lock
validation_lock_boundaries <- character()
assign(
  ".fastkpc_full_cuda_phase3_oracle_immutable_file_hashes",
  function(...) {
    immutable_hash_read_count <<- immutable_hash_read_count + 1L
    immutable_hash_reader(...)
  },
  envir = .GlobalEnv
)
assign(
  ".fastkpc_full_cuda_phase3_refresh_oracle_artifact_lock",
  function(lock, ...) {
    arguments <- list(...)
    boundary <- arguments$.boundary
    arguments$.boundary <- NULL
    if (!is.null(boundary)) {
      validation_lock_boundaries <<-
        c(validation_lock_boundaries, boundary)
    }
    do.call(lock_refresh, c(list(lock = lock), arguments))
  },
  envir = .GlobalEnv
)
rehash_validation <- fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
  output_dir, expected_identity = identity, require_full = FALSE
)
assign(
  ".fastkpc_full_cuda_phase3_oracle_immutable_file_hashes",
  immutable_hash_reader, envir = .GlobalEnv
)
assign(
  ".fastkpc_full_cuda_phase3_refresh_oracle_artifact_lock",
  lock_refresh, envir = .GlobalEnv
)
assert_identical(
  immutable_hash_read_count, 2L,
  "semantic validator rehashes immutable files before locked return"
)
assert_true(
  all(c(
    "validation_start", "validation_payload_hashes",
    "validation_shards_authenticated", "validation_complete"
  ) %in% validation_lock_boundaries),
  paste0(
    "semantic validator asserts artifact-lock ownership at long boundaries; ",
    "actual=", paste(validation_lock_boundaries, collapse = ",")
  )
)
assert_identical(
  validated$manifest$executed_native_library_sha256, binary_a,
  "published manifest records the shard-executed binary SHA"
)
assert_identical(
  validated$summary$executed_native_library_sha256, binary_a,
  "published summary records the shard-executed binary SHA"
)
completed_hashes_before_lock <- vapply(
  unlist(paths[setdiff(names(paths), c("shards_dir", "sessions_dir"))],
         use.names = FALSE),
  fastkpc_full_cuda_census_file_hash, character(1L)
)
artifact_lock_path <-
  .fastkpc_full_cuda_phase3_oracle_artifact_lock_path(output_dir)
artifact_lock_inode <- lock_inode(artifact_lock_path)
artifact_lock_hash <- lock_evidence_hash(artifact_lock_path)
held_validation_child <- start_lock_child(lock_cases$artifact, output_dir)
assert_error(
  fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
    output_dir, expected_identity = identity, require_full = FALSE
  ),
  "semantic validation serializes with a live publisher lock"
)
assert_identical(
  vapply(
    unlist(paths[setdiff(names(paths), c("shards_dir", "sessions_dir"))],
           use.names = FALSE),
    fastkpc_full_cuda_census_file_hash, character(1L)
  ),
  completed_hashes_before_lock,
  "blocked semantic validation preserves every marker and payload byte"
)
assert_true(
  identical(lock_inode(artifact_lock_path), artifact_lock_inode) &&
    identical(lock_evidence_hash(artifact_lock_path), artifact_lock_hash),
  "blocked semantic validation preserves artifact lock inode and evidence"
)
finish_lock_child(held_validation_child)
assert_true(
  file.exists(artifact_lock_path) && !dir.exists(artifact_lock_path) &&
    identical(lock_inode(artifact_lock_path), artifact_lock_inode),
  "artifact validation unlock keeps the persistent lock file and inode"
)
generic_semantic_validated <- fastkpc_full_cuda_phase3_validate_artifact(
  output_dir, kind = "oracle_sp", expected_identity = identity,
  require_full = FALSE
)
assert_true(
  isTRUE(generic_semantic_validated$authenticated) &&
    identical(
      generic_semantic_validated$manifest$oracle_semantics_version,
      .fastkpc_full_cuda_phase3_oracle_semantics_version()
    ),
  "generic oracle validator delegates semantic artifacts to strict validation"
)
assert_error(
  fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
    output_dir, expected_identity = identity, require_full = TRUE
  ),
  "caller require_full=TRUE rejects a non-full semantic artifact"
)

scope_downgrade_dir <- tempfile("phase3-oracle-full-downgrade-")
on.exit(unlink(scope_downgrade_dir, recursive = TRUE, force = TRUE), add = TRUE)
dir.create(scope_downgrade_dir, recursive = TRUE, showWarnings = FALSE)
assert_true(file.copy(output_dir, scope_downgrade_dir, recursive = TRUE),
            "full-scope downgrade fixture copy succeeds")
scope_downgrade_root <- file.path(scope_downgrade_dir, basename(output_dir))
scope_downgrade_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  scope_downgrade_root, "oracle_sp"
)
scope_downgrade_manifest <- jsonlite::read_json(
  scope_downgrade_paths$manifest_json, simplifyVector = FALSE
)
scope_downgrade_manifest$scope <- "full"
fastkpc_full_cuda_write_json(
  scope_downgrade_manifest, scope_downgrade_paths$manifest_json
)
scope_downgrade_summary <- jsonlite::read_json(
  scope_downgrade_paths$summary_json, simplifyVector = FALSE
)
scope_downgrade_summary$manifest_sha256 <-
  fastkpc_full_cuda_census_file_hash(scope_downgrade_paths$manifest_json)
fastkpc_full_cuda_write_json(
  scope_downgrade_summary, scope_downgrade_paths$summary_json
)
canonical_opener_calls <- 0L
original_canonical_opener <-
  .fastkpc_full_cuda_phase3_open_canonical_oracle_catalog
assign(
  ".fastkpc_full_cuda_phase3_open_canonical_oracle_catalog",
  function() {
    canonical_opener_calls <<- canonical_opener_calls + 1L
    stop("manifest full scope forced canonical reopen", call. = FALSE)
  },
  envir = .GlobalEnv
)
assert_error(
  fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
    scope_downgrade_root, expected_identity = identity,
    require_full = FALSE
  ),
  "manifest full scope cannot be downgraded by caller require_full=FALSE"
)
assign(
  ".fastkpc_full_cuda_phase3_open_canonical_oracle_catalog",
  original_canonical_opener, envir = .GlobalEnv
)
assert_identical(
  canonical_opener_calls, 1L,
  "manifest full scope independently forces canonical catalog reopening"
)

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
foreign_staging_dir <- paste0(output_dir, ".phase3-oracle-publish-foreign")
dir.create(foreign_staging_dir, recursive = FALSE, showWarnings = FALSE)
on.exit(unlink(foreign_staging_dir, recursive = TRUE, force = TRUE), add = TRUE)
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
assert_true(
  dir.exists(foreign_staging_dir),
  "publisher never removes another publisher's staging directory"
)

unlink(paths$summary_json, force = TRUE)
publication_lock_boundaries <- character()
assign(
  ".fastkpc_full_cuda_phase3_refresh_oracle_artifact_lock",
  function(lock, ...) {
    arguments <- list(...)
    boundary <- arguments$.boundary
    arguments$.boundary <- NULL
    if (!is.null(boundary)) {
      publication_lock_boundaries <<-
        c(publication_lock_boundaries, boundary)
    }
    do.call(lock_refresh, c(list(lock = lock), arguments))
  },
  envir = .GlobalEnv
)
partial <- publish(
  output_dir = output_dir, setup_keys = setup_keys,
  target_rows = target_rows, identity = identity,
  route_config = route_config, scope = "iteration", shard_count = 4L,
  risk_rows = risk_rows, qualification_dcov = qualification_dcov,
  command_lines = "Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_oracle_artifact.R"
)
assign(
  ".fastkpc_full_cuda_phase3_refresh_oracle_artifact_lock",
  lock_refresh, envir = .GlobalEnv
)
assert_identical(partial$status, "published",
                 "partial completion marker is republished gracefully")
assert_true(file.exists(paths$manifest_json) && file.exists(paths$summary_json),
            "partial resume restores both completion markers")
assert_true(
  all(c(
    "publication_before_merge", "publication_after_merge",
    "publication_payloads_published", "publication_markers_published"
  ) %in% publication_lock_boundaries),
  paste0(
    "publisher asserts artifact-lock ownership at merge/publication ",
    "boundaries; ",
    "actual=", paste(publication_lock_boundaries, collapse = ",")
  )
)

clone_completed_artifact <- function(label) {
  parent <- tempfile(paste0("phase3-oracle-", label, "-"))
  dir.create(parent, recursive = TRUE, showWarnings = FALSE)
  assert_true(
    file.copy(output_dir, parent, recursive = TRUE),
    paste(label, "completed artifact fixture copy succeeds")
  )
  file.path(parent, basename(output_dir))
}
snapshot_artifact_files <- function(root) {
  files <- sort(list.files(
    root, all.files = TRUE, no.. = TRUE, recursive = TRUE,
    full.names = TRUE, include.dirs = FALSE
  ), method = "radix")
  hashes <- vapply(
    files, fastkpc_full_cuda_census_file_hash, character(1L)
  )
  names(hashes) <- substring(files, nchar(root) + 2L)
  hashes
}

original_recursive_snapshot <- get0(
  ".fastkpc_full_cuda_phase3_oracle_immutable_file_hashes",
  envir = .GlobalEnv, inherits = FALSE
)
assert_true(
  is.function(original_recursive_snapshot),
  "oracle race tests can intercept immutable evidence snapshots"
)

recursive_snapshot <- original_recursive_snapshot(
  paths, .fastkpc_full_cuda_phase3_payload_keys("oracle_sp")
)
recursive_entries <- sort(list.files(
  output_dir, all.files = TRUE, no.. = TRUE, recursive = TRUE,
  full.names = TRUE, include.dirs = TRUE
), method = "radix")
recursive_relative_paths <- substring(
  recursive_entries, nchar(output_dir) + 2L
)
recursive_regular_files <- vapply(
  recursive_entries, function(path) file_test("-f", path), logical(1L)
)
assert_true(
  is.data.frame(recursive_snapshot) &&
    identical(
      names(recursive_snapshot),
      c("relative_path", "entry_type", "byte_size", "sha256")
    ) &&
    identical(recursive_snapshot$relative_path, recursive_relative_paths) &&
    identical(
      recursive_snapshot$entry_type,
      unname(ifelse(
        recursive_regular_files, "regular_file", "directory"
      ))
    ) &&
    identical(
      recursive_snapshot$byte_size[recursive_regular_files],
      as.numeric(file.info(recursive_entries[recursive_regular_files])$size)
    ) &&
    all(is.na(recursive_snapshot$byte_size[!recursive_regular_files])) &&
    identical(
      recursive_snapshot$sha256[recursive_regular_files],
      unname(vapply(
        recursive_entries[recursive_regular_files],
        fastkpc_full_cuda_census_file_hash, character(1L)
      ))
    ) && all(is.na(recursive_snapshot$sha256[!recursive_regular_files])),
  "oracle snapshot records every recursive path, type, byte size, and hash"
)

symlink_evidence_root <- clone_completed_artifact(
  "directory-symlink-evidence"
)
on.exit(
  unlink(dirname(symlink_evidence_root), recursive = TRUE, force = TRUE),
  add = TRUE
)
symlink_evidence_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  symlink_evidence_root, "oracle_sp"
)
external_evidence_root <- tempfile("phase3-external-evidence-")
dir.create(
  file.path(external_evidence_root, "nested"),
  recursive = TRUE, showWarnings = FALSE
)
on.exit(
  unlink(external_evidence_root, recursive = TRUE, force = TRUE),
  add = TRUE
)
external_sentinel_name <- "must-not-be-traversed.txt"
writeLines(
  "external evidence sentinel",
  file.path(external_evidence_root, "nested", external_sentinel_name),
  useBytes = TRUE
)
directory_symlink_path <- file.path(
  symlink_evidence_paths$sessions_dir, "external-session-tree"
)
directory_symlink_created <- isTRUE(suppressWarnings(file.symlink(
  external_evidence_root, directory_symlink_path
)))
if (!directory_symlink_created) {
  assert_true(
    !file.exists(directory_symlink_path) &&
      !nzchar(Sys.readlink(directory_symlink_path)),
    "directory-symlink regression skips only when link creation is unsupported"
  )
} else {
  evidence_list_calls <- list()
  external_sentinel_observed <- FALSE
  assign(
    "list.files",
    function(path = ".", pattern = NULL, all.files = FALSE,
             full.names = FALSE, recursive = FALSE,
             ignore.case = FALSE, include.dirs = FALSE, no.. = FALSE) {
      result <- base::list.files(
        path = path, pattern = pattern, all.files = all.files,
        full.names = full.names, recursive = recursive,
        ignore.case = ignore.case, include.dirs = include.dirs,
        no.. = no..
      )
      evidence_list_calls[[length(evidence_list_calls) + 1L]] <<- list(
        path = normalizePath(path, mustWork = FALSE),
        recursive = recursive
      )
      if (any(grepl(external_sentinel_name, result, fixed = TRUE))) {
        external_sentinel_observed <<- TRUE
      }
      result
    },
    envir = .GlobalEnv
  )
  directory_symlink_error <- tryCatch(
    original_recursive_snapshot(
      symlink_evidence_paths,
      .fastkpc_full_cuda_phase3_payload_keys("oracle_sp")
    ),
    error = function(error) error,
    finally = rm("list.files", envir = .GlobalEnv)
  )
  external_evidence_path <- normalizePath(
    external_evidence_root, mustWork = TRUE
  )
  assert_true(
    inherits(directory_symlink_error, "error") &&
      grepl(
        "evidence path surface is invalid",
        conditionMessage(directory_symlink_error), fixed = TRUE
      ) && length(evidence_list_calls) > 0L &&
      all(!vapply(
        evidence_list_calls, function(call) call$recursive, logical(1L)
      )) && !external_sentinel_observed &&
      !external_evidence_path %in% vapply(
        evidence_list_calls, function(call) call$path, character(1L)
      ),
    paste0(
      "oracle evidence walk rejects directory links without following ",
      "their external contents; error=",
      if (inherits(directory_symlink_error, "error")) {
        conditionMessage(directory_symlink_error)
      } else "accepted",
      "; recursive_calls=",
      sum(vapply(
        evidence_list_calls, function(call) call$recursive, logical(1L)
      )),
      "; external_sentinel_observed=", external_sentinel_observed
    )
  )
}

manifest_race_root <- clone_completed_artifact("manifest-snapshot-race")
on.exit(
  unlink(dirname(manifest_race_root), recursive = TRUE, force = TRUE),
  add = TRUE
)
manifest_race_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  manifest_race_root, "oracle_sp"
)
replacement_manifest <- jsonlite::read_json(
  manifest_race_paths$manifest_json, simplifyVector = FALSE
)
replacement_manifest_path <- tempfile(
  "phase3-hostile-manifest-", fileext = ".json"
)
on.exit(unlink(replacement_manifest_path, force = TRUE), add = TRUE)
writeLines(
  c("", readLines(manifest_race_paths$manifest_json, warn = FALSE)),
  replacement_manifest_path, useBytes = TRUE
)
assert_true(
  identical(
    jsonlite::read_json(
      replacement_manifest_path, simplifyVector = FALSE
    ),
    replacement_manifest
  ) && !identical(
    fastkpc_full_cuda_census_file_hash(replacement_manifest_path),
    fastkpc_full_cuda_census_file_hash(manifest_race_paths$manifest_json)
  ),
  "hostile manifest replacement changes bytes without changing semantics"
)
manifest_race_summary <- jsonlite::read_json(
  manifest_race_paths$summary_json, simplifyVector = FALSE
)
replacement_manifest_sha256 <-
  fastkpc_full_cuda_census_file_hash(replacement_manifest_path)
replacement_summary_path <- tempfile(
  "phase3-hostile-summary-", fileext = ".json"
)
on.exit(unlink(replacement_summary_path, force = TRUE), add = TRUE)
replacement_summary_lines <- readLines(
  manifest_race_paths$summary_json, warn = FALSE
)
assert_true(
  sum(grepl(
    manifest_race_summary$manifest_sha256,
    replacement_summary_lines, fixed = TRUE
  )) == 1L,
  "summary contains one manifest hash token"
)
writeLines(
  sub(
    manifest_race_summary$manifest_sha256, replacement_manifest_sha256,
    replacement_summary_lines, fixed = TRUE
  ),
  replacement_summary_path, useBytes = TRUE
)
manifest_snapshot_calls <- 0L
assign(
  ".fastkpc_full_cuda_phase3_oracle_immutable_file_hashes",
  function(paths, payload_keys) {
    manifest_snapshot_calls <<- manifest_snapshot_calls + 1L
    snapshot <- original_recursive_snapshot(paths, payload_keys)
    if (manifest_snapshot_calls == 1L) {
      assert_true(
        file.copy(
          replacement_manifest_path, paths$manifest_json,
          overwrite = TRUE
        ),
        "hostile manifest replacement succeeds at the snapshot boundary"
      )
      assert_true(
        file.copy(
          replacement_summary_path, paths$summary_json,
          overwrite = TRUE
        ),
        "hostile summary replacement succeeds at the snapshot boundary"
      )
    }
    snapshot
  },
  envir = .GlobalEnv
)
manifest_race_error <- tryCatch({
  fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
    manifest_race_root, expected_identity = identity, require_full = FALSE
  )
  NULL
}, error = function(error) error)
assign(
  ".fastkpc_full_cuda_phase3_oracle_immutable_file_hashes",
  original_recursive_snapshot, envir = .GlobalEnv
)

evidence_race_root <- clone_completed_artifact("recursive-evidence-race")
on.exit(
  unlink(dirname(evidence_race_root), recursive = TRUE, force = TRUE),
  add = TRUE
)
evidence_race_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  evidence_race_root, "oracle_sp"
)
shard_race_path <- sort(list.files(
  evidence_race_paths$shards_dir, all.files = TRUE, no.. = TRUE,
  recursive = TRUE, full.names = TRUE
), method = "radix")[[1L]]
session_race_path <- sort(list.files(
  evidence_race_paths$sessions_dir, all.files = TRUE, no.. = TRUE,
  recursive = TRUE, full.names = TRUE
), method = "radix")[[1L]]
append_hostile_byte <- function(path) {
  connection <- file(path, open = "ab")
  on.exit(close(connection), add = TRUE)
  writeBin(as.raw(0L), connection)
  invisible(TRUE)
}
evidence_snapshot_calls <- 0L
assign(
  ".fastkpc_full_cuda_phase3_oracle_immutable_file_hashes",
  function(paths, payload_keys) {
    evidence_snapshot_calls <<- evidence_snapshot_calls + 1L
    if (evidence_snapshot_calls == 2L) {
      append_hostile_byte(shard_race_path)
      append_hostile_byte(session_race_path)
    }
    original_recursive_snapshot(paths, payload_keys)
  },
  envir = .GlobalEnv
)
evidence_race_error <- tryCatch({
  fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
    evidence_race_root, expected_identity = identity, require_full = FALSE
  )
  NULL
}, error = function(error) error)
assign(
  ".fastkpc_full_cuda_phase3_oracle_immutable_file_hashes",
  original_recursive_snapshot, envir = .GlobalEnv
)

snapshot_acquisition_root <- clone_completed_artifact(
  "snapshot-acquisition-race"
)
on.exit(
  unlink(dirname(snapshot_acquisition_root), recursive = TRUE, force = TRUE),
  add = TRUE
)
snapshot_acquisition_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  snapshot_acquisition_root, "oracle_sp"
)
snapshot_acquisition_addition <- file.path(
  snapshot_acquisition_paths$sessions_dir, "hostile-added-session.json"
)
original_file_hash <- fastkpc_full_cuda_census_file_hash
snapshot_hash_calls <- 0L
assign(
  "fastkpc_full_cuda_census_file_hash",
  function(path) {
    snapshot_hash_calls <<- snapshot_hash_calls + 1L
    if (snapshot_hash_calls == 1L) {
      writeLines("{}", snapshot_acquisition_addition, useBytes = TRUE)
    }
    original_file_hash(path)
  },
  envir = .GlobalEnv
)
snapshot_acquisition_error <- tryCatch(
  original_recursive_snapshot(
    snapshot_acquisition_paths,
    .fastkpc_full_cuda_phase3_payload_keys("oracle_sp")
  ),
  error = function(error) error,
  finally = assign(
    "fastkpc_full_cuda_census_file_hash", original_file_hash,
    envir = .GlobalEnv
  )
)
assert_true(
  inherits(manifest_race_error, "error") &&
    inherits(evidence_race_error, "error") &&
    inherits(snapshot_acquisition_error, "error") &&
    grepl(
      "immutable files changed during validation",
      conditionMessage(manifest_race_error), fixed = TRUE
    ) && grepl(
      "immutable files changed during validation",
      conditionMessage(evidence_race_error), fixed = TRUE
    ) && grepl(
      "evidence changed while snapshotting",
      conditionMessage(snapshot_acquisition_error), fixed = TRUE
    ),
  paste0(
    "oracle validation rejects manifest parse/snapshot, recursive ",
    "evidence, and acquisition races; manifest=",
    if (inherits(manifest_race_error, "error")) {
      conditionMessage(manifest_race_error)
    } else "accepted",
    "; recursive=",
    if (inherits(evidence_race_error, "error")) {
      conditionMessage(evidence_race_error)
    } else "accepted",
    "; acquisition_accepted=", is.null(snapshot_acquisition_error)
  )
)

refresh_oracle_payload_markers <- function(root, payload_path) {
  paths <- fastkpc_full_cuda_phase3_artifact_paths(root, "oracle_sp")
  manifest <- jsonlite::read_json(
    paths$manifest_json, simplifyVector = FALSE
  )
  manifest$payload_file_sha256[[basename(payload_path)]] <-
    fastkpc_full_cuda_census_file_hash(payload_path)
  .fastkpc_full_cuda_phase3_write_json_exact(
    manifest, paths$manifest_json
  )
  summary <- jsonlite::read_json(
    paths$summary_json, simplifyVector = FALSE
  )
  summary$manifest_sha256 <-
    fastkpc_full_cuda_census_file_hash(paths$manifest_json)
  .fastkpc_full_cuda_phase3_write_json_exact(summary, paths$summary_json)
  invisible(TRUE)
}

csv_ulp_root <- clone_completed_artifact("csv-ulp-corruption")
on.exit(
  unlink(dirname(csv_ulp_root), recursive = TRUE, force = TRUE),
  add = TRUE
)
csv_ulp_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  csv_ulp_root, "oracle_sp"
)
csv_ulp_rows <- .fastkpc_full_cuda_phase3_read_csv(
  csv_ulp_paths$qualification_dcov_csv,
  "qualification_dcov_parity.csv"
)
original_csv_p_value <- csv_ulp_rows$p_value[[1L]]
csv_ulp_rows$p_value[[1L]] <- original_csv_p_value + 1e-16
assert_true(
  is.finite(csv_ulp_rows$p_value[[1L]]) &&
    !identical(csv_ulp_rows$p_value[[1L]], original_csv_p_value) &&
    abs(csv_ulp_rows$p_value[[1L]] - original_csv_p_value) < 1e-15,
  "hostile CSV p-value mutation is finite, nonzero, and below tolerance"
)
.fastkpc_full_cuda_phase3_write_csv(
  csv_ulp_rows, csv_ulp_paths$qualification_dcov_csv
)
refresh_oracle_payload_markers(
  csv_ulp_root, csv_ulp_paths$qualification_dcov_csv
)
csv_ulp_error <- tryCatch({
  fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
    csv_ulp_root, expected_identity = identity, require_full = FALSE
  )
  NULL
}, error = function(error) error)

summary_ulp_root <- clone_completed_artifact("summary-ulp-corruption")
on.exit(
  unlink(dirname(summary_ulp_root), recursive = TRUE, force = TRUE),
  add = TRUE
)
summary_ulp_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  summary_ulp_root, "oracle_sp"
)
summary_ulp <- jsonlite::read_json(
  summary_ulp_paths$summary_json, simplifyVector = FALSE
)
original_summary_max_residual <- as.double(
  summary_ulp$max_residual_abs_diff
)
summary_ulp$max_residual_abs_diff <- original_summary_max_residual + 5e-30
assert_true(
  is.finite(summary_ulp$max_residual_abs_diff) &&
    !identical(
      summary_ulp$max_residual_abs_diff, original_summary_max_residual
    ) && abs(
      summary_ulp$max_residual_abs_diff - original_summary_max_residual
    ) < 1e-15,
  "hostile summary mutation is finite, nonzero, and below tolerance"
)
.fastkpc_full_cuda_phase3_write_json_exact(
  summary_ulp, summary_ulp_paths$summary_json
)
summary_ulp_error <- tryCatch({
  fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
    summary_ulp_root, expected_identity = identity, require_full = FALSE
  )
  NULL
}, error = function(error) error)
assert_true(
  inherits(csv_ulp_error, "error") &&
    inherits(summary_ulp_error, "error") &&
    grepl(
      "CSV values do not match RDS: p_value",
      conditionMessage(csv_ulp_error), fixed = TRUE
    ) && grepl(
      "summary claims are not recomputed: max_residual_abs_diff",
      conditionMessage(summary_ulp_error), fixed = TRUE
    ),
  paste0(
    "oracle validation rejects refreshed-hash sub-tolerance numeric ",
    "corruption; csv=",
    if (inherits(csv_ulp_error, "error")) {
      conditionMessage(csv_ulp_error)
    } else "accepted",
    "; summary=",
    if (inherits(summary_ulp_error, "error")) {
      conditionMessage(summary_ulp_error)
    } else "accepted"
  )
)

csv_signed_zero_root <- clone_completed_artifact(
  "csv-signed-zero-corruption"
)
on.exit(
  unlink(dirname(csv_signed_zero_root), recursive = TRUE, force = TRUE),
  add = TRUE
)
csv_signed_zero_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  csv_signed_zero_root, "oracle_sp"
)
csv_signed_zero_rows <- .fastkpc_full_cuda_phase3_read_csv(
  csv_signed_zero_paths$qualification_dcov_csv,
  "qualification_dcov_parity.csv"
)
positive_zero_rows <- which(vapply(
  csv_signed_zero_rows$p_value_difference,
  function(value) ieee_double_equal(value, 0.0), logical(1L)
))
assert_true(
  length(positive_zero_rows) > 0L,
  "qualification dCov fixture contains a positive-zero p-value difference"
)
csv_signed_zero_row <- positive_zero_rows[[1L]]
csv_signed_zero_rows$p_value_difference[[csv_signed_zero_row]] <- -0.0
.fastkpc_full_cuda_phase3_write_csv(
  csv_signed_zero_rows, csv_signed_zero_paths$qualification_dcov_csv
)
csv_signed_zero_roundtrip <- .fastkpc_full_cuda_phase3_read_csv(
  csv_signed_zero_paths$qualification_dcov_csv,
  "qualification_dcov_parity.csv"
)
assert_true(
  ieee_double_equal(
    csv_signed_zero_roundtrip$p_value_difference[[csv_signed_zero_row]],
    -0.0
  ),
  "hostile CSV p-value difference retains negative zero"
)
refresh_oracle_payload_markers(
  csv_signed_zero_root, csv_signed_zero_paths$qualification_dcov_csv
)
csv_signed_zero_error <- tryCatch({
  fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
    csv_signed_zero_root, expected_identity = identity, require_full = FALSE
  )
  NULL
}, error = function(error) error)

summary_signed_zero_root <- clone_completed_artifact(
  "summary-signed-zero-corruption"
)
on.exit(
  unlink(dirname(summary_signed_zero_root), recursive = TRUE, force = TRUE),
  add = TRUE
)
summary_signed_zero_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  summary_signed_zero_root, "oracle_sp"
)
summary_signed_zero <- jsonlite::read_json(
  summary_signed_zero_paths$summary_json, simplifyVector = FALSE
)
assert_true(
  ieee_double_equal(summary_signed_zero$max_rhs_relative_l2, 0.0),
  "oracle summary fixture contains a positive-zero RHS claim"
)
summary_signed_zero$max_rhs_relative_l2 <- -0.0
.fastkpc_full_cuda_phase3_write_json_exact(
  summary_signed_zero, summary_signed_zero_paths$summary_json
)
summary_signed_zero_roundtrip <- jsonlite::read_json(
  summary_signed_zero_paths$summary_json, simplifyVector = FALSE
)
assert_true(
  ieee_double_equal(
    summary_signed_zero_roundtrip$max_rhs_relative_l2, -0.0
  ),
  "hostile summary RHS claim retains negative zero"
)
summary_signed_zero_error <- tryCatch({
  fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
    summary_signed_zero_root,
    expected_identity = identity, require_full = FALSE
  )
  NULL
}, error = function(error) error)
assert_true(
  inherits(csv_signed_zero_error, "error") &&
    inherits(summary_signed_zero_error, "error") &&
    grepl(
      "CSV values do not match RDS: p_value_difference",
      conditionMessage(csv_signed_zero_error), fixed = TRUE
    ) && grepl(
      "summary claims are not recomputed: max_rhs_relative_l2",
      conditionMessage(summary_signed_zero_error), fixed = TRUE
    ) && .fastkpc_full_cuda_phase3_identity_value_equal(0L, 0.0) &&
    .fastkpc_full_cuda_phase3_identity_value_equal(10L, 10.0),
  paste0(
    "oracle validation rejects signed-zero corruption while retaining ",
    "integer JSON compatibility; csv=",
    if (inherits(csv_signed_zero_error, "error")) {
      conditionMessage(csv_signed_zero_error)
    } else "accepted",
    "; summary=",
    if (inherits(summary_signed_zero_error, "error")) {
      conditionMessage(summary_signed_zero_error)
    } else "accepted"
  )
)

incompatible_identity <- identity
incompatible_identity$gpu_uuid <- paste0("GPU-", strrep("f", 32L))
incompatible_identity <- refresh_hash(incompatible_identity)
incompatible_invocations <- list(
  scope = list(scope = "qualification"),
  identity = list(identity = incompatible_identity),
  catalog = list(catalog = list(malformed = TRUE), device_id = 0L),
  device = list(device_id = 0L)
)
incompatible_results <- lapply(
  names(incompatible_invocations),
  function(label) {
    root <- clone_completed_artifact(label)
    on.exit(unlink(dirname(root), recursive = TRUE, force = TRUE), add = TRUE)
    before <- snapshot_artifact_files(root)
    arguments <- list(
      output_dir = root, setup_keys = setup_keys,
      target_rows = target_rows, identity = identity,
      route_config = route_config, scope = "iteration", shard_count = 4L,
      risk_rows = risk_rows, qualification_dcov = qualification_dcov,
      command_lines = paste("incompatible", label)
    )
    arguments[names(incompatible_invocations[[label]])] <-
      incompatible_invocations[[label]]
    error <- tryCatch({
      do.call(publish, arguments)
      NULL
    }, error = function(error) error)
    after <- snapshot_artifact_files(root)
    c(
      rejected = inherits(error, "error"),
      unchanged = identical(after, before)
    )
  }
)
names(incompatible_results) <- names(incompatible_invocations)
incompatible_matrix <- do.call(rbind, incompatible_results)
assert_true(
  all(incompatible_matrix[, "rejected"]) &&
    all(incompatible_matrix[, "unchanged"]),
  paste0(
    "incompatible completed-artifact invocation is rejected without byte ",
    "mutation; rejected=",
    paste(incompatible_matrix[, "rejected"], collapse = ","),
    "; unchanged=",
    paste(incompatible_matrix[, "unchanged"], collapse = ",")
  )
)

corrupt_marker_root <- clone_completed_artifact("corrupt-markers")
on.exit(
  unlink(dirname(corrupt_marker_root), recursive = TRUE, force = TRUE),
  add = TRUE
)
corrupt_marker_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  corrupt_marker_root, "oracle_sp"
)
corrupt_manifest <- jsonlite::read_json(
  corrupt_marker_paths$manifest_json, simplifyVector = FALSE
)
corrupt_manifest$oracle_semantics_version <- "legacy-forged-semantics"
fastkpc_full_cuda_write_json(
  corrupt_manifest, corrupt_marker_paths$manifest_json
)
corrupt_summary <- jsonlite::read_json(
  corrupt_marker_paths$summary_json, simplifyVector = FALSE
)
corrupt_summary$manifest_sha256 <-
  fastkpc_full_cuda_census_file_hash(corrupt_marker_paths$manifest_json)
fastkpc_full_cuda_write_json(
  corrupt_summary, corrupt_marker_paths$summary_json
)
corrupt_marker_hashes <- snapshot_artifact_files(corrupt_marker_root)
assert_error(
  fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
    corrupt_marker_root, expected_identity = identity, require_full = FALSE
  ),
  "public validator rejects a corrupt pair of completion markers"
)
assert_error(
  publish(
    output_dir = corrupt_marker_root, setup_keys = setup_keys,
    target_rows = target_rows, identity = identity,
    route_config = route_config, scope = "iteration", shard_count = 4L,
    risk_rows = risk_rows, qualification_dcov = qualification_dcov,
    command_lines = "corrupt complete markers"
  ),
  "publisher fails closed on corrupt complete markers"
)
assert_identical(
  snapshot_artifact_files(corrupt_marker_root), corrupt_marker_hashes,
  "corrupt complete marker failure preserves every existing byte"
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

binary_forgery_dir <- tempfile("phase3-oracle-binary-forgery-")
on.exit(unlink(binary_forgery_dir, recursive = TRUE, force = TRUE), add = TRUE)
dir.create(binary_forgery_dir, recursive = TRUE, showWarnings = FALSE)
assert_true(file.copy(output_dir, binary_forgery_dir, recursive = TRUE),
            "binary-forgery fixture copy succeeds")
binary_forgery_root <- file.path(binary_forgery_dir, basename(output_dir))
binary_forgery_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  binary_forgery_root, "oracle_sp"
)
binary_forgery_manifest <- jsonlite::read_json(
  binary_forgery_paths$manifest_json, simplifyVector = FALSE
)
binary_forgery_manifest$executed_native_library_sha256 <- binary_b
fastkpc_full_cuda_write_json(
  binary_forgery_manifest, binary_forgery_paths$manifest_json
)
binary_forgery_summary <- jsonlite::read_json(
  binary_forgery_paths$summary_json, simplifyVector = FALSE
)
binary_forgery_summary$executed_native_library_sha256 <- binary_b
binary_forgery_summary$manifest_sha256 <-
  fastkpc_full_cuda_census_file_hash(binary_forgery_paths$manifest_json)
fastkpc_full_cuda_write_json(
  binary_forgery_summary, binary_forgery_paths$summary_json
)
assert_error(
  fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
    binary_forgery_root, expected_identity = identity, require_full = FALSE
  ),
  "manifest and summary cannot forge the shard-executed binary SHA"
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
