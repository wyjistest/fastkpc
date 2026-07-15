source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
expect_error_contains <- function(expr, text) {
  error <- tryCatch(force(expr), error = identity)
  assert_true(
    inherits(error, "error") &&
      grepl(text, conditionMessage(error), fixed = TRUE),
    paste0("expected error containing '", text, "'")
  )
  invisible(error)
}

column_schema <- function(typeof, na = "none", nonnegative = FALSE,
                          finite = FALSE) {
  list(
    typeof = typeof,
    class = if (identical(typeof, "double")) "numeric" else typeof,
    na = na,
    nonnegative = nonnegative,
    finite = finite
  )
}

validate_record_schema <- function(records, schema, expected_rows, context) {
  if (!is.data.frame(records)) {
    fail(paste(context, "is not a data frame"))
  }
  expected_rows_clean <- typeof(expected_rows) == "integer" &&
    length(expected_rows) == 1L && !is.object(expected_rows) &&
    is.null(attributes(expected_rows)) && !is.na(expected_rows) &&
    expected_rows > 0L
  if (!isTRUE(expected_rows_clean)) {
    fail(paste(context, "expected row count is malformed"))
  }
  if (!identical(nrow(records), expected_rows)) {
    fail(paste(context, "row count is not exact"))
  }
  schema_names <- names(schema)
  schema_clean <- is.list(schema) && is.character(schema_names) &&
    length(schema_names) > 0L && !anyNA(schema_names) &&
    !anyDuplicated(schema_names) && all(nzchar(schema_names))
  if (!isTRUE(schema_clean)) {
    fail(paste(context, "schema definition is malformed"))
  }
  if (!identical(names(records), schema_names)) {
    fail(paste(context, "column names are not exact"))
  }

  for (field in schema_names) {
    spec <- schema[[field]]
    spec_clean <- is.list(spec) && identical(
      names(spec),
      c("typeof", "class", "na", "nonnegative", "finite")
    ) && typeof(spec$typeof) == "character" &&
      length(spec$typeof) == 1L && !is.na(spec$typeof) &&
      typeof(spec$class) == "character" &&
      length(spec$class) == 1L && !is.na(spec$class) &&
      typeof(spec$na) == "character" && length(spec$na) == 1L &&
      !is.na(spec$na) && spec$na %in% c("none", "allow") &&
      typeof(spec$nonnegative) == "logical" &&
      length(spec$nonnegative) == 1L && !is.na(spec$nonnegative) &&
      typeof(spec$finite) == "logical" &&
      length(spec$finite) == 1L && !is.na(spec$finite)
    if (!isTRUE(spec_clean)) {
      fail(paste(context, field, "schema is malformed"))
    }
    value <- records[[field]]
    type_exact <- identical(typeof(value), spec$typeof) &&
      identical(class(value), spec$class) && !is.object(value) &&
      is.null(attributes(value))
    if (!isTRUE(type_exact)) {
      fail(paste(context, field, "type/class is not exact"))
    }
    if (!identical(length(value), expected_rows)) {
      fail(paste(context, field, "length is not exact"))
    }
    if (identical(spec$na, "none") && anyNA(value)) {
      fail(paste(context, field, "contains NA"))
    }
    observed <- value[!is.na(value)]
    if (isTRUE(spec$finite) && length(observed) > 0L &&
        any(!is.finite(observed))) {
      fail(paste(context, field, "contains a non-finite value"))
    }
    if (isTRUE(spec$nonnegative) && length(observed) > 0L &&
        any(observed < 0)) {
      fail(paste(context, field, "contains a negative value"))
    }
  }
  invisible(records)
}

schema_contract_error <- tryCatch(
  validate_record_schema(
    data.frame(counter = integer()),
    schema = list(counter = list(
      typeof = "integer", class = "integer", na = "none",
      nonnegative = TRUE, finite = FALSE
    )),
    expected_rows = 1L,
    context = "schema contract fixture"
  ),
  error = identity
)
assert_true(
  inherits(schema_contract_error, "error") &&
    grepl("schema contract fixture row count is not exact",
          conditionMessage(schema_contract_error), fixed = TRUE),
  "record schema validator must reject zero-row records before field access"
)
wrong_type_error <- tryCatch(
  validate_record_schema(
    data.frame(counter = 1),
    schema = list(counter = column_schema("integer", nonnegative = TRUE)),
    expected_rows = 1L,
    context = "schema integer fixture"
  ),
  error = identity
)
assert_true(
  inherits(wrong_type_error, "error") &&
    grepl("schema integer fixture counter type/class is not exact",
          conditionMessage(wrong_type_error), fixed = TRUE),
  "record schema validator must reject double counters"
)
missing_column_error <- tryCatch(
  validate_record_schema(
    data.frame(other = 1L),
    schema = list(counter = column_schema("integer", nonnegative = TRUE)),
    expected_rows = 1L,
    context = "schema missing-column fixture"
  ),
  error = identity
)
assert_true(
  inherits(missing_column_error, "error") &&
    grepl("schema missing-column fixture column names are not exact",
          conditionMessage(missing_column_error), fixed = TRUE),
  "record schema validator must reject missing columns"
)

underflow_errors <- fastkpc_full_cuda_fixed_sp_phase3b_numeric_errors(
  c(candidate = 2e-200), c(reference = 1e-200), "underflow fixture"
)
assert_true(
  isTRUE(all.equal(
    unname(underflow_errors[["relative_l2"]]), 1,
    tolerance = 1e-14
  )),
  "scaled relative-L2 must not underflow for 1e-200 values"
)
named_errors <- fastkpc_full_cuda_fixed_sp_phase3b_numeric_errors(
  c(candidate_name = 2), c(reference_name = 1), "named vector fixture"
)
assert_true(
  identical(unname(named_errors), c(1, 1)),
  "numeric evidence ignores names only after strict vector validation"
)
large_errors <- fastkpc_full_cuda_fixed_sp_phase3b_numeric_errors(
  c(1e308), c(-1e308), "large-value fixture"
)
assert_true(
  all(is.finite(large_errors)) && large_errors[["max_abs"]] > 1e307 &&
    isTRUE(all.equal(
      unname(large_errors[["relative_l2"]]), 2,
      tolerance = 1e-14
    )),
  "scaled numeric evidence remains finite for large opposite values"
)

numeric_bad_inputs <- list(
  different_matrix_shape = list(
    matrix(as.double(1:4), 2L, 2L),
    matrix(as.double(1:4), 1L, 4L)
  ),
  character = list("1", 1),
  factor = list(factor("1"), 1),
  complex = list(1 + 0i, 1),
  empty = list(double(), double()),
  na = list(c(NA_real_), 1),
  nan = list(c(NaN), 1),
  inf = list(c(Inf), 1)
)
for (label in names(numeric_bad_inputs)) {
  values <- numeric_bad_inputs[[label]]
  expect_error_contains(
    fastkpc_full_cuda_fixed_sp_phase3b_numeric_errors(
      values[[1L]], values[[2L]], paste(label, "numeric fixture")
    ),
    "requires finite non-empty bare double vectors with equal shape"
  )
}

strict_named_list <- list(alpha = 1L)
fastkpc_full_cuda_fixed_sp_phase3b_require_fields(
  strict_named_list, "alpha", "strict list fixture"
)
expect_error_contains(
  fastkpc_full_cuda_fixed_sp_phase3b_require_fields(
    data.frame(alpha = 1L), "alpha", "data-frame fixture"
  ),
  "must be a bare named list"
)
expect_error_contains(
  fastkpc_full_cuda_fixed_sp_phase3b_require_fields(
    structure(strict_named_list, class = "review-list"),
    "alpha", "classed-list fixture"
  ),
  "must be a bare named list"
)
expect_error_contains(
  fastkpc_full_cuda_fixed_sp_phase3b_require_fields(
    setNames(list(1L), ""), "alpha", "empty-name fixture"
  ),
  "must be a bare named list"
)
expect_error_contains(
  fastkpc_full_cuda_fixed_sp_phase3b_require_fields(
    strict_named_list, structure("alpha", class = "review-fields"),
    "classed-fields fixture"
  ),
  "fields must be bare non-empty unique character names"
)
expect_error_contains(
  fastkpc_full_cuda_fixed_sp_phase3b_require_fields(
    strict_named_list, "alpha", c("bad", "context")
  ),
  "context must be one bare non-empty character scalar"
)
expect_error_contains(
  fastkpc_full_cuda_fixed_sp_phase3b_character_vector(
    "value", 1, "bad size fixture"
  ),
  "size must be one non-negative integer"
)
expect_error_contains(
  fastkpc_full_cuda_fixed_sp_phase3b_character_vector(
    "value", 1L, c("bad", "name")
  ),
  "name must be one bare non-empty character scalar"
)
expect_error_contains(
  fastkpc_full_cuda_fixed_sp_phase3b_character_vector(
    "value", 1L, "bad allow_na fixture", allow_na = 0L
  ),
  "allow_na must be one non-NA logical scalar"
)

valid_paths <- fastkpc_full_cuda_fixed_sp_phase3b_validate_paths(
  "phase2", "census", "prepared", "data"
)
assert_true(
  identical(valid_paths, list(
    phase2_dir = "phase2", census_dir = "census",
    prepared_dir = "prepared", data_path = "data"
  )),
  "strict path validation preserves four scalar paths"
)
for (bad_path in list(
    character(), c("one", "two"),
    structure("classed", class = "review-path"), NA_character_, ""
  )) {
  expect_error_contains(
    fastkpc_full_cuda_fixed_sp_phase3b_validate_paths(
      bad_path, "census", "prepared", "data"
    ),
    "phase2_dir must be one bare non-empty character scalar"
  )
}

ast_is_call_to <- function(node, function_name) {
  is.call(node) && length(node) >= 1L && is.symbol(node[[1L]]) &&
    identical(as.character(node[[1L]]), function_name)
}

ast_count_calls <- function(node, function_name) {
  count <- as.integer(ast_is_call_to(node, function_name))
  if (is.call(node) || is.pairlist(node) || is.expression(node)) {
    children <- as.list(node)
    if (length(children) > 0L) {
      count <- count + sum(vapply(
        children, ast_count_calls, integer(1L),
        function_name = function_name
      ))
    }
  }
  as.integer(count)
}

ast_collect_calls <- function(node, function_name) {
  matches <- if (ast_is_call_to(node, function_name)) list(node) else list()
  if (is.call(node) || is.pairlist(node) || is.expression(node)) {
    children <- as.list(node)
    if (length(children) > 0L) {
      child_matches <- lapply(
        children, ast_collect_calls, function_name = function_name
      )
      matches <- c(matches, unlist(child_matches, recursive = FALSE))
    }
  }
  matches
}

ast_count_calls_outside <- function(node, function_name, protector_name,
                                    protected = FALSE) {
  protected_here <- protected || ast_is_call_to(node, protector_name)
  count <- as.integer(
    ast_is_call_to(node, function_name) && !protected_here
  )
  if (is.call(node) || is.pairlist(node) || is.expression(node)) {
    children <- as.list(node)
    if (length(children) > 0L) {
      count <- count + sum(vapply(
        children,
        ast_count_calls_outside,
        integer(1L),
        function_name = function_name,
        protector_name = protector_name,
        protected = protected_here
      ))
    }
  }
  as.integer(count)
}

ast_is_runtime_null_assignment <- function(node) {
  is.call(node) && length(node) == 3L &&
    is.symbol(node[[1L]]) &&
    identical(as.character(node[[1L]]), "<-") &&
    is.symbol(node[[2L]]) &&
    identical(as.character(node[[2L]]), "runtime") &&
    is.null(node[[3L]])
}

ast_count_runtime_null_assignments <- function(node) {
  count <- as.integer(ast_is_runtime_null_assignment(node))
  if (is.call(node) || is.pairlist(node) || is.expression(node)) {
    children <- as.list(node)
    if (length(children) > 0L) {
      count <- count + sum(vapply(
        children, ast_count_runtime_null_assignments, integer(1L)
      ))
    }
  }
  as.integer(count)
}

runtime_create_name <- "fixed_sp_cuda_runtime_create"
cleanup_scope_name <-
  "fastkpc_full_cuda_fixed_sp_phase3b_run_cleanup_scope"
phase3a_runner_body <- body(
  fastkpc_run_full_cuda_fixed_sp_phase3a_iteration
)
phase3b_runner_body <- body(
  fastkpc_run_full_cuda_fixed_sp_phase3b_iteration
)
assert_true(
  identical(ast_count_calls(phase3a_runner_body, runtime_create_name), 1L),
  "Phase 3A iteration runner must retain exactly one runtime create call"
)
assert_true(
  identical(ast_count_calls(phase3b_runner_body, runtime_create_name), 1L),
  "Phase 3B iteration runner must contain exactly one runtime create call"
)
phase3b_cleanup_scopes <- ast_collect_calls(
  phase3b_runner_body, cleanup_scope_name
)
phase3b_iteration_scopes <- Filter(function(scope_call) {
  arguments <- as.list(scope_call)
  identical(arguments$context, "Phase 3B iteration cleanup")
}, phase3b_cleanup_scopes)
assert_true(
  length(phase3b_iteration_scopes) == 1L &&
    identical(
      ast_count_calls(
        as.list(phase3b_iteration_scopes[[1L]])$body,
        runtime_create_name
      ),
      1L
    ) &&
    identical(
      ast_count_calls_outside(
        phase3b_runner_body, runtime_create_name, cleanup_scope_name
      ),
      0L
    ) &&
    identical(ast_count_runtime_null_assignments(phase3b_runner_body), 1L),
  paste(
    "Phase 3B runtime create must be unique inside the iteration cleanup",
    "scope body, with one NULL ownership initializer outside"
  )
)

runtime_identity_fixture <- list(
  device_id = 0L,
  creator_pid = 123,
  generation = 7,
  gpu_name = "fixture-gpu",
  compute_capability_major = 8L,
  compute_capability_minor = 0L,
  sm_count = 10L,
  cuda_toolkit_version = 12000L,
  cuda_driver_version = 12000L,
  cusolver_deterministic_mode = "enabled",
  cublas_math_mode = "pedantic",
  cublas_atomics_mode = "not_allowed"
)
fastkpc_full_cuda_fixed_sp_phase3b_validate_runtime_identity(
  runtime_identity_fixture, runtime_identity_fixture,
  runtime_identity_fixture, 0L
)
mutated_runtime_identity <- runtime_identity_fixture
mutated_runtime_identity$generation <- 8
expect_error_contains(
  fastkpc_full_cuda_fixed_sp_phase3b_validate_runtime_identity(
    runtime_identity_fixture, mutated_runtime_identity,
    runtime_identity_fixture, 0L
  ),
  "runtime immutable identity changed"
)

prepared_free_fixture <- list(
  prepared_s_key_sha256 = sprintf("%064x", 1L),
  setup_h2d_upload_count = 1L,
  setup_h2d_bytes = 128,
  coefficient_output_capacity = 16,
  generation = 3,
  output_slot_leased = FALSE,
  output_slot_state = "free",
  output_slot_poison_reason = ""
)
prepared_leased_fixture <- prepared_free_fixture
prepared_leased_fixture$output_slot_leased <- TRUE
prepared_leased_fixture$output_slot_state <- "leased"
fastkpc_full_cuda_fixed_sp_phase3b_validate_prepared_snapshots(
  prepared_free_fixture, prepared_leased_fixture, prepared_free_fixture,
  prepared_free_fixture$prepared_s_key_sha256, 8
)
bad_prepared_snapshot <- prepared_leased_fixture
bad_prepared_snapshot$output_slot_state <- "free"
expect_error_contains(
  fastkpc_full_cuda_fixed_sp_phase3b_validate_prepared_snapshots(
    prepared_free_fixture, bad_prepared_snapshot, prepared_free_fixture,
    prepared_free_fixture$prepared_s_key_sha256, 8
  ),
  "prepared output-slot lifecycle is invalid"
)

cleanup_retry_evidence <- local({
  state <- new.env(parent = emptyenv())
  state$released <- FALSE
  state$token_freed <- FALSE
  state$handle_freed <- FALSE
  state$runtime_freed <- FALSE
  state$release_attempts <- 0L
  state$order <- character()
  operations <- list(
    token_release = list(
      needed = function() !state$released,
      run = function() {
        state$order <- c(state$order, "token_release")
        state$release_attempts <- state$release_attempts + 1L
        if (state$release_attempts == 1L) stop("retry release")
        state$released <- TRUE
      }
    ),
    token_free = list(
      needed = function() !state$token_freed,
      run = function() {
        state$order <- c(state$order, "token_free")
        state$token_freed <- TRUE
      }
    ),
    handle_free = list(
      needed = function() !state$handle_freed,
      run = function() {
        state$order <- c(state$order, "handle_free")
        state$handle_freed <- TRUE
      }
    ),
    runtime_free = list(
      needed = function() !state$runtime_freed,
      run = function() {
        state$order <- c(state$order, "runtime_free")
        state$runtime_freed <- TRUE
      }
    )
  )
  fastkpc_full_cuda_fixed_sp_phase3b_cleanup_operations(
    operations, context = "cleanup retry fixture"
  )
  first_order <- state$order
  fastkpc_full_cuda_fixed_sp_phase3b_cleanup_operations(
    operations, context = "cleanup idempotence fixture"
  )
  list(
    attempts = state$release_attempts,
    first_order = first_order,
    final_order = state$order
  )
})
assert_true(
  identical(cleanup_retry_evidence$attempts, 2L) &&
    identical(
      cleanup_retry_evidence$first_order,
      c("token_release", "token_release", "token_free",
        "handle_free", "runtime_free")
    ) &&
    identical(cleanup_retry_evidence$final_order,
              cleanup_retry_evidence$first_order),
  "cleanup retries once, preserves order, and skips released ownership"
)

cleanup_persistent_evidence <- local({
  order <- character()
  token_freed <- FALSE
  handle_freed <- FALSE
  runtime_freed <- FALSE
  operations <- list(
    token_release = list(
      needed = function() TRUE,
      run = function() {
        order <<- c(order, "token_release")
        stop("persistent release")
      }
    ),
    token_free = list(
      needed = function() !token_freed,
      run = function() {
        order <<- c(order, "token_free")
        token_freed <<- TRUE
      }
    ),
    handle_free = list(
      needed = function() !handle_freed,
      run = function() {
        order <<- c(order, "handle_free")
        handle_freed <<- TRUE
      }
    ),
    runtime_free = list(
      needed = function() !runtime_freed,
      run = function() {
        order <<- c(order, "runtime_free")
        runtime_freed <<- TRUE
      }
    )
  )
  error <- tryCatch(
    fastkpc_full_cuda_fixed_sp_phase3b_cleanup_operations(
      operations, body_error = simpleError("body failure"),
      context = "cleanup persistent fixture"
    ),
    error = identity
  )
  list(error = error, order = order)
})
assert_true(
  inherits(cleanup_persistent_evidence$error, "error") &&
    grepl("body failure", conditionMessage(
      cleanup_persistent_evidence$error
    ), fixed = TRUE) &&
    grepl("persistent release", conditionMessage(
      cleanup_persistent_evidence$error
    ), fixed = TRUE) &&
    identical(
      cleanup_persistent_evidence$order,
      c("token_release", "token_release", "token_free", "handle_free",
        "runtime_free")
    ),
  "persistent cleanup failure is surfaced after ordered best-effort cleanup"
)

synthetic_interrupt_evidence <- local({
  state <- new.env(parent = emptyenv())
  state$token_released <- FALSE
  state$token_freed <- FALSE
  state$handle_freed <- FALSE
  state$runtime_freed <- FALSE
  state$order <- character()
  inner_operations <- list(
    token_release = list(
      needed = function() !state$token_released,
      run = function() {
        state$order <- c(state$order, "token_release")
        state$token_released <- TRUE
      }
    ),
    token_free = list(
      needed = function() !state$token_freed,
      run = function() {
        state$order <- c(state$order, "token_free")
        state$token_freed <- TRUE
      }
    ),
    handle_free = list(
      needed = function() !state$handle_freed,
      run = function() {
        state$order <- c(state$order, "handle_free")
        state$handle_freed <- TRUE
      }
    ),
    runtime_free = list(
      needed = function() FALSE,
      run = function() invisible(NULL)
    )
  )
  outer_operations <- list(
    token_release = list(
      needed = function() FALSE,
      run = function() invisible(NULL)
    ),
    token_free = list(
      needed = function() FALSE,
      run = function() invisible(NULL)
    ),
    handle_free = list(
      needed = function() FALSE,
      run = function() invisible(NULL)
    ),
    runtime_free = list(
      needed = function() !state$runtime_freed,
      run = function() {
        state$order <- c(state$order, "runtime_free")
        state$runtime_freed <- TRUE
      }
    )
  )
  interrupt <- structure(
    list(message = "synthetic Phase 3B interrupt", call = NULL),
    class = c("interrupt", "condition")
  )
  condition_handler <- function(condition) condition
  propagated <- tryCatch(
    fastkpc_full_cuda_fixed_sp_phase3b_run_cleanup_scope(
      body = {
        fastkpc_full_cuda_fixed_sp_phase3b_run_cleanup_scope(
          body = stop(interrupt),
          operations = inner_operations,
          context = "synthetic inner interrupt fixture"
        )
      },
      operations = outer_operations,
      context = "synthetic outer interrupt fixture"
    ),
    error = condition_handler,
    interrupt = condition_handler
  )
  list(
    propagated = propagated,
    order = state$order,
    released = c(
      state$token_released, state$token_freed,
      state$handle_freed, state$runtime_freed
    )
  )
})
assert_true(
  inherits(synthetic_interrupt_evidence$propagated, "interrupt") &&
    identical(
      conditionMessage(synthetic_interrupt_evidence$propagated),
      "synthetic Phase 3B interrupt"
    ) &&
    identical(
      synthetic_interrupt_evidence$order,
      c("token_release", "token_free", "handle_free", "runtime_free")
    ) &&
    identical(synthetic_interrupt_evidence$released, rep(TRUE, 4L)),
  paste(
    "nested cleanup scopes preserve interrupt semantics, clean inner then",
    "outer ownership, and do not repeat successful on.exit operations"
  )
)

interrupt_cleanup_failure_evidence <- local({
  state <- new.env(parent = emptyenv())
  state$release_attempts <- 0L
  state$token_freed <- FALSE
  state$handle_freed <- FALSE
  state$runtime_freed <- FALSE
  operations <- list(
    token_release = list(
      needed = function() TRUE,
      run = function() {
        state$release_attempts <- state$release_attempts + 1L
        stop("synthetic persistent cleanup failure")
      }
    ),
    token_free = list(
      needed = function() !state$token_freed,
      run = function() state$token_freed <- TRUE
    ),
    handle_free = list(
      needed = function() !state$handle_freed,
      run = function() state$handle_freed <- TRUE
    ),
    runtime_free = list(
      needed = function() !state$runtime_freed,
      run = function() state$runtime_freed <- TRUE
    )
  )
  interrupt <- structure(
    list(message = "synthetic interrupt with cleanup failure", call = NULL),
    class = c("interrupt", "condition")
  )
  condition_handler <- function(condition) condition
  propagated <- tryCatch(
    fastkpc_full_cuda_fixed_sp_phase3b_run_cleanup_scope(
      body = stop(interrupt),
      operations = operations,
      context = "synthetic interrupt cleanup-failure fixture"
    ),
    error = condition_handler,
    interrupt = condition_handler
  )
  list(
    propagated = propagated,
    release_attempts = state$release_attempts,
    later_cleanup_exact = identical(
      c(state$token_freed, state$handle_freed, state$runtime_freed),
      rep(TRUE, 3L)
    )
  )
})
assert_true(
  inherits(interrupt_cleanup_failure_evidence$propagated, "interrupt") &&
    inherits(interrupt_cleanup_failure_evidence$propagated, "error") &&
    inherits(interrupt_cleanup_failure_evidence$propagated$parent,
             "interrupt") &&
    grepl(
      "synthetic interrupt with cleanup failure",
      conditionMessage(interrupt_cleanup_failure_evidence$propagated),
      fixed = TRUE
    ) &&
    grepl(
      "synthetic persistent cleanup failure",
      conditionMessage(interrupt_cleanup_failure_evidence$propagated),
      fixed = TRUE
    ) &&
    identical(interrupt_cleanup_failure_evidence$release_attempts, 4L) &&
    interrupt_cleanup_failure_evidence$later_cleanup_exact,
  "interrupt plus cleanup failure remains interrupt-classed and fully visible"
)

ordinary_error_cleanup_evidence <- local({
  state <- new.env(parent = emptyenv())
  state$released <- FALSE
  state$token_freed <- FALSE
  state$handle_freed <- FALSE
  state$order <- character()
  operations <- list(
    token_release = list(
      needed = function() !state$released,
      run = function() {
        state$order <- c(state$order, "token_release")
        state$released <- TRUE
      }
    ),
    token_free = list(
      needed = function() !state$token_freed,
      run = function() {
        state$order <- c(state$order, "token_free")
        state$token_freed <- TRUE
      }
    ),
    handle_free = list(
      needed = function() !state$handle_freed,
      run = function() {
        state$order <- c(state$order, "handle_free")
        state$handle_freed <- TRUE
      }
    ),
    runtime_free = list(
      needed = function() FALSE,
      run = function() invisible(NULL)
    )
  )
  condition_handler <- function(condition) condition
  propagated <- tryCatch(
    fastkpc_full_cuda_fixed_sp_phase3b_run_cleanup_scope(
      body = stop("ordinary cleanup-scope body error", call. = FALSE),
      operations = operations,
      context = "ordinary cleanup-scope error fixture"
    ),
    error = condition_handler,
    interrupt = condition_handler
  )
  list(propagated = propagated, order = state$order)
})
assert_true(
  inherits(ordinary_error_cleanup_evidence$propagated, "error") &&
    !inherits(ordinary_error_cleanup_evidence$propagated, "interrupt") &&
    grepl(
      "ordinary cleanup-scope body error",
      conditionMessage(ordinary_error_cleanup_evidence$propagated),
      fixed = TRUE
    ) &&
    identical(
      ordinary_error_cleanup_evidence$order,
      c("token_release", "token_free", "handle_free")
    ),
  "ordinary errors retain their class and ordered cleanup behavior"
)

fixture_setup_keys <- sprintf("%064x", 1:2)
fixture_target_keys <- sprintf("%064x", 3:4)
fixture_iteration_hash <- sprintf("%064x", 5L)
fixture_safe_fitted_hash <- fastkpc_full_cuda_census_metadata_hash(c(1, 2))
fixture_safe_residual_hash <- fastkpc_full_cuda_census_metadata_hash(c(3, 4))
fixture_stable_fitted_hash <- fastkpc_full_cuda_census_metadata_hash(c(5, 6))
fixture_stable_residual_hash <- fastkpc_full_cuda_census_metadata_hash(c(7, 8))
fixture_catalog_records <- data.frame(
  iteration_subset_hash = fixture_iteration_hash,
  ordered_setup_key_digest = fastkpc_full_cuda_census_key_set_hash(
    fixture_setup_keys
  ),
  ordered_target_key_digest = fastkpc_full_cuda_census_key_set_hash(
    fixture_target_keys
  ),
  stringsAsFactors = FALSE
)
fixture_batch_records <- data.frame(
  prepared_s_key_sha256 = fixture_setup_keys,
  target_count = c(1L, 1L),
  coefficient_batch_finalize_call_count = c(0L, 0L),
  fitted_batch_finalize_call_count = c(1L, 0L),
  residual_rss_batch_finalize_call_count = c(1L, 0L),
  per_target_output_finalize_call_count = c(0L, 0L),
  batch_output_finalized_target_count = c(1L, 0L),
  route_status_conservation_exact = c(TRUE, TRUE),
  stringsAsFactors = FALSE
)
fixture_target_records <- data.frame(
  prepared_s_key_sha256 = fixture_setup_keys,
  residual_key_sha256 = fixture_target_keys,
  target = c(1L, 2L),
  authenticated_planned_route = c("CHOLESKY_BATCHED", "AUGMENTED_QR"),
  executed_route = c("CHOLESKY_BATCHED", NA_character_),
  solver_status = c(
    "OK_CHOLESKY_SINGLE", "ERR_STABLE_PATH_NOT_IMPLEMENTED"
  ),
  residual_max_abs_diff = c(1e-12, NA_real_),
  residual_relative_l2_diff = c(1e-12, NA_real_),
  fitted_max_abs_diff = c(1e-12, NA_real_),
  fitted_relative_l2_diff = c(1e-12, NA_real_),
  oracle_call_count = c(1L, 0L),
  oracle_fitted_hash = c(fixture_safe_fitted_hash, NA_character_),
  authenticated_fitted_hash = c(
    fixture_safe_fitted_hash, fixture_stable_fitted_hash
  ),
  oracle_residual_hash = c(fixture_safe_residual_hash, NA_character_),
  authenticated_residual_hash = c(
    fixture_safe_residual_hash, fixture_stable_residual_hash
  ),
  oracle_fitted_hash_exact = c(TRUE, NA),
  oracle_residual_hash_exact = c(TRUE, NA),
  stringsAsFactors = FALSE
)
fixture_evidence <- fastkpc_full_cuda_fixed_sp_phase3b_validate_result_evidence(
  fixture_catalog_records, fixture_batch_records, fixture_target_records,
  fixture_iteration_hash
)
assert_true(
  identical(fixture_evidence$oracle_call_count, 1L) &&
    identical(fixture_evidence$batch_output_finalized_target_count, 1L),
  "valid CPU result evidence is accepted"
)

mutated_safe_error <- fixture_target_records
mutated_safe_error$residual_max_abs_diff[[1L]] <- 1e-7
expect_error_contains(
  fastkpc_full_cuda_fixed_sp_phase3b_validate_result_evidence(
    fixture_catalog_records, fixture_batch_records, mutated_safe_error,
    fixture_iteration_hash
  ),
  "target numeric evidence is invalid"
)
mutated_safe_na <- fixture_target_records
mutated_safe_na$fitted_relative_l2_diff[[1L]] <- NA_real_
expect_error_contains(
  fastkpc_full_cuda_fixed_sp_phase3b_validate_result_evidence(
    fixture_catalog_records, fixture_batch_records, mutated_safe_na,
    fixture_iteration_hash
  ),
  "target numeric evidence is invalid"
)
mutated_oracle_count <- fixture_target_records
mutated_oracle_count$oracle_call_count[[1L]] <- 0L
expect_error_contains(
  fastkpc_full_cuda_fixed_sp_phase3b_validate_result_evidence(
    fixture_catalog_records, fixture_batch_records, mutated_oracle_count,
    fixture_iteration_hash
  ),
  "target oracle evidence is invalid"
)
mutated_oracle_hash <- fixture_target_records
mutated_oracle_hash$oracle_fitted_hash[[1L]] <- fixture_stable_fitted_hash
expect_error_contains(
  fastkpc_full_cuda_fixed_sp_phase3b_validate_result_evidence(
    fixture_catalog_records, fixture_batch_records, mutated_oracle_hash,
    fixture_iteration_hash
  ),
  "target oracle evidence is invalid"
)
mutated_key <- fixture_target_records
mutated_key$residual_key_sha256[[1L]] <- paste0(
  "A", substring(fixture_target_keys[[1L]], 2L)
)
expect_error_contains(
  fastkpc_full_cuda_fixed_sp_phase3b_validate_result_evidence(
    fixture_catalog_records, fixture_batch_records, mutated_key,
    fixture_iteration_hash
  ),
  "target key evidence is invalid"
)
mutated_digest <- fixture_catalog_records
mutated_digest$ordered_target_key_digest[[1L]] <- fixture_iteration_hash
expect_error_contains(
  fastkpc_full_cuda_fixed_sp_phase3b_validate_result_evidence(
    mutated_digest, fixture_batch_records, fixture_target_records,
    fixture_iteration_hash
  ),
  "catalog digest evidence is invalid"
)
finalize_fields <- c(
  "coefficient_batch_finalize_call_count",
  "fitted_batch_finalize_call_count",
  "residual_rss_batch_finalize_call_count",
  "per_target_output_finalize_call_count",
  "batch_output_finalized_target_count"
)
for (field in finalize_fields) {
  mutated_finalize <- fixture_batch_records
  mutated_finalize[[field]][[1L]] <- mutated_finalize[[field]][[1L]] + 1L
  expect_error_contains(
    fastkpc_full_cuda_fixed_sp_phase3b_validate_result_evidence(
      fixture_catalog_records, mutated_finalize, fixture_target_records,
      fixture_iteration_hash
    ),
    "batch finalize evidence is invalid"
  )
}

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP Phase 3B fixed-sp iteration batch gate\n")
  quit(save = "no", status = 0)
}

build_fastkpc_cuda_native(rebuild = FALSE)
assert_true(fastkpc_cuda_available(), "CUDA must be available")

result <- fastkpc_run_full_cuda_fixed_sp_phase3b_iteration(
  phase2_dir = file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "oracle_351x48_v1"
  ),
  census_dir = file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "workload_census_351x48_v1"
  ),
  prepared_dir = file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1"
  ),
  data_path = file.path(
    "fastkpc", "artifacts", "kpc_tprs_real_zhu",
    "cancer_RD-causalDiscoveryInput.rds"
  ),
  device_id = 0L
)

assert_true(
  identical(names(result), c(
    "catalog_records", "runtime_records", "prepared_records",
    "batch_records", "target_records", "summary"
  )),
  "iteration runner returns source, lifecycle, batch, target, and summary rows"
)

catalog_records <- result$catalog_records
runtime_records <- result$runtime_records
prepared_records <- result$prepared_records
batch_records <- result$batch_records
target_records <- result$target_records
summary <- result$summary

expected <- list(
  iteration_subset_hash =
    "d69956655e2ee0186aceb4bf2d545b831051c389aed139769d9e5902f433fb96",
  ordered_setup_key_digest =
    "0987396d35f1277a435474be52309dc05c84f585431fdee8e583e00494e445b4",
  ordered_target_key_digest =
    "5ebd9800378fe8016d6853726a963d1dc814d817385c081aa7948b2c08e6934f",
  setup_count = 44L,
  target_count = 270L,
  batch_call_count = 44L,
  all_safe_batch_count = 5L,
  mixed_batch_count = 33L,
  all_stable_batch_count = 6L,
  cholesky_ok_count = 172L,
  true_batched_subgroup_count = 26L,
  true_batched_target_count = 160L,
  cholesky_single_target_count = 12L,
  whole_batch_true_batched_count = 5L,
  stable_not_implemented_count = 98L,
  oracle_call_count = 172L,
  oracle_fitted_hash_exact_count = 172L,
  oracle_residual_hash_exact_count = 172L,
  setup_h2d_upload_count = 44L,
  target_batch_h2d_call_count = 44L,
  target_h2d_copy_count = 88L,
  rhs_device_build_count = 44L,
  full_cuda_data_plane = TRUE,
  invalid_output_init_count = 44L,
  coefficient_batch_finalize_call_count = 0L,
  fitted_batch_finalize_call_count = 38L,
  residual_rss_batch_finalize_call_count = 38L,
  per_target_output_finalize_call_count = 0L,
  batch_output_finalized_target_count = 172L,
  workspace_grow_count_after_warmup = 0L,
  per_target_allocation_count_after_warmup = 0L,
  per_target_handle_create_count = 0L,
  cuda_device_synchronize_count = 0L,
  implicit_residual_d2h_count = 0L,
  all_output_slot_leases_released = TRUE,
  cpu_fallback_count = 0L,
  unknown_fallback_count = 0L
)

integer_field <- function() column_schema("integer", nonnegative = TRUE)
logical_field <- function() column_schema("logical")
character_field <- function(na = "none") column_schema("character", na = na)
double_field <- function(na = "none") {
  column_schema("double", na = na, nonnegative = TRUE, finite = TRUE)
}

catalog_schema <- list(
  scope = character_field(),
  authenticated = logical_field(),
  catalog_open_count = integer_field(),
  setup_count = integer_field(),
  target_count = integer_field(),
  iteration_subset_hash = character_field(),
  ordered_setup_key_digest = character_field(),
  ordered_target_key_digest = character_field()
)
runtime_schema <- list(
  stage = character_field(),
  device_id = integer_field(),
  gpu_name = character_field(),
  runtime_context_create_count = integer_field(),
  cuda_device_allocation_count = integer_field(),
  cuda_host_allocation_count = integer_field(),
  stream_create_count = integer_field(),
  event_create_count = integer_field(),
  cublas_handle_create_count = integer_field(),
  cusolver_handle_create_count = integer_field(),
  workspace_grow_count = integer_field(),
  cuda_device_synchronize_count = integer_field(),
  compute_capability_major = integer_field(),
  compute_capability_minor = integer_field(),
  sm_count = integer_field(),
  cuda_toolkit_version = integer_field(),
  cuda_driver_version = integer_field(),
  cusolver_deterministic_mode = character_field(),
  cublas_math_mode = character_field(),
  cublas_atomics_mode = character_field(),
  cublas_user_workspace_installed = logical_field(),
  cublas_workspace_alignment = double_field(),
  workspace_reserve_count = integer_field(),
  freed = logical_field(),
  creator_pid = double_field(),
  generation = double_field()
)
prepared_schema <- list(
  prepared_s_key_sha256 = character_field(),
  batch_ordinal = integer_field(),
  prepared_handle_create_count = integer_field(),
  prepared_handle_free_count = integer_field(),
  setup_h2d_upload_count = integer_field(),
  setup_h2d_bytes = double_field(),
  coefficient_output_capacity = double_field(),
  prepared_generation = double_field(),
  output_slot_state_before_solve = character_field(),
  output_slot_state_after_solve = character_field(),
  output_slot_state_after_release = character_field(),
  output_slot_poison_reason_empty = logical_field(),
  output_slot_leased_after_release = logical_field()
)
batch_schema <- list(
  prepared_s_key_sha256 = character_field(),
  batch_ordinal = integer_field(),
  target_count = integer_field(),
  batch_call_count = integer_field(),
  native_batch_call = logical_field(),
  true_batched_kernel = logical_field(),
  true_batched_subgroup_count = integer_field(),
  true_batched_attempted_target_count = integer_field(),
  true_batched_target_count = integer_field(),
  cholesky_single_target_count = integer_field(),
  potrf_batched_call_count = integer_field(),
  potrs_batched_call_count = integer_field(),
  planned_cholesky_target_count = integer_field(),
  planned_qr_target_count = integer_field(),
  planned_svd_target_count = integer_field(),
  executed_cholesky_target_count = integer_field(),
  executed_qr_target_count = integer_field(),
  executed_svd_target_count = integer_field(),
  stable_reroute_count = integer_field(),
  cholesky_to_svd_count = integer_field(),
  qr_to_svd_count = integer_field(),
  setup_h2d_upload_count = integer_field(),
  target_batch_h2d_call_count = integer_field(),
  target_h2d_copy_count = integer_field(),
  target_h2d_bytes = double_field(),
  rhs_device_build_count = integer_field(),
  full_cuda_data_plane = logical_field(),
  invalid_output_init_count = integer_field(),
  coefficient_batch_finalize_call_count = integer_field(),
  fitted_batch_finalize_call_count = integer_field(),
  residual_rss_batch_finalize_call_count = integer_field(),
  per_target_output_finalize_call_count = integer_field(),
  batch_output_finalized_target_count = integer_field(),
  canonical_output_order_exact = logical_field(),
  target_keys_exact = logical_field(),
  route_status_conservation_exact = logical_field(),
  resource_snapshot_captured = logical_field(),
  resource_instrumentation_version = integer_field(),
  resource_allocation_count_before_solve = integer_field(),
  resource_allocation_count_after_solve = integer_field(),
  resource_handle_create_count_before_solve = integer_field(),
  resource_handle_create_count_after_solve = integer_field(),
  cuda_device_allocation_count_during_solve = integer_field(),
  cuda_host_allocation_count_during_solve = integer_field(),
  stream_create_count_during_solve = integer_field(),
  event_create_count_during_solve = integer_field(),
  cublas_handle_create_count_during_solve = integer_field(),
  cusolver_handle_create_count_during_solve = integer_field(),
  per_target_allocation_count_after_warmup = integer_field(),
  per_target_handle_create_count = integer_field(),
  workspace_grow_count_after_warmup = integer_field(),
  cuda_device_synchronize_count = integer_field(),
  implicit_residual_d2h_count = integer_field(),
  cpu_fallback_count = integer_field(),
  unknown_fallback_count = integer_field(),
  pre_shadow_materialize_call_count = integer_field(),
  pre_shadow_materialize_target_count = integer_field(),
  pre_shadow_d2h_bytes = double_field(),
  explicit_oracle_shadow_observation = logical_field(),
  shadow_observation_purpose = character_field(),
  post_shadow_materialize_call_count = integer_field(),
  post_shadow_materialize_target_count = integer_field(),
  post_shadow_d2h_bytes = double_field(),
  output_slot_release_count = integer_field(),
  output_slot_leased_after_release = logical_field()
)
target_schema <- list(
  prepared_s_key_sha256 = character_field(),
  batch_ordinal = integer_field(),
  target_ordinal = integer_field(),
  residual_key_sha256 = character_field(),
  target = integer_field(),
  planned_route = character_field(),
  authenticated_planned_route = character_field(),
  executed_route = character_field(na = "allow"),
  reroute_reason = character_field(),
  solver_status = character_field(),
  residual_max_abs_diff = double_field(na = "allow"),
  residual_relative_l2_diff = double_field(na = "allow"),
  fitted_max_abs_diff = double_field(na = "allow"),
  fitted_relative_l2_diff = double_field(na = "allow"),
  explicit_oracle_shadow_observation = logical_field(),
  oracle_call_count = integer_field(),
  oracle_fitted_hash = character_field(na = "allow"),
  authenticated_fitted_hash = character_field(),
  oracle_residual_hash = character_field(na = "allow"),
  authenticated_residual_hash = character_field(),
  oracle_fitted_hash_exact = column_schema("logical", na = "allow"),
  oracle_residual_hash_exact = column_schema("logical", na = "allow")
)

validate_record_schema(
  catalog_records, catalog_schema, 1L, "catalog records"
)
validate_record_schema(
  runtime_records, runtime_schema, 3L, "runtime records"
)
validate_record_schema(
  prepared_records, prepared_schema, 44L, "prepared records"
)
validate_record_schema(
  batch_records, batch_schema, 44L, "batch records"
)
validate_record_schema(
  target_records, target_schema, 270L, "target records"
)
production_result_evidence <-
  fastkpc_full_cuda_fixed_sp_phase3b_validate_result_evidence(
    catalog_records, batch_records, target_records,
    expected$iteration_subset_hash
  )

is_bare_sha_vector <- function(value) {
  typeof(value) == "character" && !is.object(value) &&
    is.null(attributes(value)) && !anyNA(value) &&
    all(grepl("^[0-9a-f]{64}$", value))
}
assert_true(
  is_bare_sha_vector(batch_records$prepared_s_key_sha256) &&
    is_bare_sha_vector(target_records$prepared_s_key_sha256) &&
    is_bare_sha_vector(target_records$residual_key_sha256) &&
    all(target_records$target >= 1L & target_records$target <= 48L),
  "setup/target identities are bare lowercase SHA-256 keys with target 1..48"
)
independent_setup_key_digest <- fastkpc_full_cuda_census_key_set_hash(
  batch_records$prepared_s_key_sha256
)
independent_target_key_digest <- fastkpc_full_cuda_census_key_set_hash(
  target_records$residual_key_sha256
)
assert_true(
  identical(catalog_records$iteration_subset_hash,
            expected$iteration_subset_hash) &&
    identical(catalog_records$ordered_setup_key_digest,
              independent_setup_key_digest) &&
    identical(catalog_records$ordered_target_key_digest,
              independent_target_key_digest),
  "catalog identity binds the authenticated iteration and ordered key digests"
)

assert_true(
  is.data.frame(catalog_records) && nrow(catalog_records) == 1L &&
    identical(catalog_records$scope, "iteration") &&
    identical(catalog_records$authenticated, TRUE) &&
    identical(catalog_records$catalog_open_count, 1L) &&
    identical(catalog_records$setup_count,
              as.integer(nrow(prepared_records))) &&
    identical(catalog_records$target_count,
              as.integer(nrow(target_records))),
  "catalog is opened and authenticated exactly once"
)
assert_true(
  identical(runtime_records$stage,
            c("runtime-created", "workspace-reserved", "final")) &&
    identical(runtime_records$workspace_reserve_count, c(0L, 1L, 1L)) &&
    all(runtime_records$runtime_context_create_count == 1L) &&
    all(!runtime_records$freed),
  "one runtime is created, reserved once, and observed at final teardown"
)
runtime_immutable_fields <- c(
  "device_id", "gpu_name", "compute_capability_major",
  "compute_capability_minor", "sm_count", "cuda_toolkit_version",
  "cuda_driver_version", "cusolver_deterministic_mode", "cublas_math_mode",
  "cublas_atomics_mode", "creator_pid", "generation"
)
assert_true(
  identical(runtime_records$device_id, rep(0L, 3L)) &&
    all(vapply(runtime_immutable_fields, function(field) {
      identical(
        runtime_records[[field]],
        rep(runtime_records[[field]][[1L]], 3L)
      )
    }, logical(1L))),
  "runtime requested device, creator, generation, GPU identity, and config persist"
)
assert_true(
  identical(prepared_records$batch_ordinal, seq_len(44L)) &&
    identical(
      prepared_records$prepared_s_key_sha256,
      sort(prepared_records$prepared_s_key_sha256, method = "radix")
    ) &&
    !anyDuplicated(prepared_records$prepared_s_key_sha256) &&
    all(prepared_records$prepared_handle_create_count == 1L) &&
    all(prepared_records$prepared_handle_free_count == 1L) &&
    all(prepared_records$coefficient_output_capacity > 0) &&
    all(prepared_records$output_slot_state_before_solve == "free") &&
    all(prepared_records$output_slot_state_after_solve == "leased") &&
    all(prepared_records$output_slot_state_after_release == "free") &&
    all(prepared_records$output_slot_poison_reason_empty),
  "one prepared handle is created and freed per canonical setup"
)
assert_true(
  identical(batch_records$batch_ordinal, seq_len(44L)) &&
    identical(batch_records$prepared_s_key_sha256,
              prepared_records$prepared_s_key_sha256) &&
    identical(
      batch_records$prepared_s_key_sha256,
      sort(batch_records$prepared_s_key_sha256, method = "radix")
    ) &&
    !anyDuplicated(batch_records$prepared_s_key_sha256),
  "one metric row is returned for each canonical setup batch"
)
assert_true(
  !anyDuplicated(target_records$residual_key_sha256),
  "one status and numeric row is returned for each canonical target"
)

reserved_runtime_index <- match("workspace-reserved", runtime_records$stage)
final_runtime_index <- match("final", runtime_records$stage)
assert_true(
  identical(reserved_runtime_index, 2L) &&
    identical(final_runtime_index, 3L),
  "runtime stages have one exact canonical row each"
)
runtime_workspace_grow_delta <- as.integer(
  runtime_records$workspace_grow_count[[final_runtime_index]] -
    runtime_records$workspace_grow_count[[reserved_runtime_index]]
)
runtime_synchronize_delta <- as.integer(
  runtime_records$cuda_device_synchronize_count[[final_runtime_index]] -
    runtime_records$cuda_device_synchronize_count[[reserved_runtime_index]]
)
batch_workspace_grow_delta <- as.integer(sum(
  batch_records$workspace_grow_count_after_warmup
))
batch_synchronize_delta <- as.integer(sum(
  batch_records$cuda_device_synchronize_count
))
assert_true(
  runtime_workspace_grow_delta >= 0L && runtime_synchronize_delta >= 0L &&
    identical(runtime_workspace_grow_delta, batch_workspace_grow_delta) &&
    identical(runtime_synchronize_delta, batch_synchronize_delta),
  "runtime deltas exactly match the sum of non-overlapping batch deltas"
)
assert_true(
  identical(prepared_records$setup_h2d_upload_count,
            batch_records$setup_h2d_upload_count),
  "prepared and batch records agree on every setup H2D upload count"
)

setup_keys <- batch_records$prepared_s_key_sha256
target_indices_by_setup <- lapply(setup_keys, function(setup_key) {
  which(target_records$prepared_s_key_sha256 == setup_key)
})
names(target_indices_by_setup) <- setup_keys
target_count_by_setup <- unname(vapply(
  target_indices_by_setup, length, integer(1L)
))
assert_true(
  all(batch_records$target_count > 0L) &&
    sum(batch_records$target_count) == nrow(target_records),
  "every canonical batch owns a non-empty target block"
)
expected_setup_keys <- rep(
  batch_records$prepared_s_key_sha256, batch_records$target_count
)
expected_batch_ordinals <- rep(
  batch_records$batch_ordinal, batch_records$target_count
)
expected_target_ordinals <- unlist(
  lapply(batch_records$target_count, seq_len), use.names = FALSE
)
assert_true(
  identical(target_records$prepared_s_key_sha256, expected_setup_keys),
  "target setup-key blocks are globally canonical and contiguous"
)
assert_true(
  identical(target_records$batch_ordinal, expected_batch_ordinals),
  "target batch ordinals are globally canonical and contiguous"
)
assert_true(
  identical(target_records$target_ordinal, expected_target_ordinals),
  "target ordinals restart canonically inside globally ordered blocks"
)
safe_count_by_setup <- unname(vapply(target_indices_by_setup, function(indices) {
  as.integer(sum(
    target_records$authenticated_planned_route[indices] == "CHOLESKY_BATCHED"
  ))
}, integer(1L)))
all_safe <- safe_count_by_setup == target_count_by_setup
all_stable <- safe_count_by_setup == 0L
mixed <- !all_safe & !all_stable

canonical_target_order_exact <- all(vapply(
  seq_along(setup_keys), function(index) {
    indices <- target_indices_by_setup[[index]]
    keys <- target_records$residual_key_sha256[indices]
    identical(target_records$batch_ordinal[indices],
              rep(as.integer(index), length(indices))) &&
      identical(target_records$target_ordinal[indices],
                seq_along(indices)) &&
      identical(keys, sort(keys, method = "radix"))
  }, logical(1L)
))
assert_true(
  canonical_target_order_exact &&
    identical(batch_records$target_count, target_count_by_setup) &&
    all(batch_records$canonical_output_order_exact) &&
    all(batch_records$target_keys_exact),
  "target rows and native output metadata preserve canonical batch order"
)

route_count <- function(route, executed = FALSE) {
  unname(vapply(target_indices_by_setup, function(indices) {
    values <- if (executed) {
      target_records$executed_route[indices]
    } else {
      target_records$authenticated_planned_route[indices]
    }
    as.integer(sum(!is.na(values) & values == route))
  }, integer(1L)))
}
planned_cholesky <- route_count("CHOLESKY_BATCHED")
planned_qr <- route_count("AUGMENTED_QR")
planned_svd <- route_count("AUGMENTED_SVD")
executed_cholesky <- route_count("CHOLESKY_BATCHED", executed = TRUE)
executed_qr <- route_count("AUGMENTED_QR", executed = TRUE)
executed_svd <- route_count("AUGMENTED_SVD", executed = TRUE)
batched_status_count <- unname(vapply(target_indices_by_setup, function(indices) {
  as.integer(sum(
    target_records$solver_status[indices] == "OK_CHOLESKY_BATCHED"
  ))
}, integer(1L)))
single_status_count <- unname(vapply(target_indices_by_setup, function(indices) {
  as.integer(sum(
    target_records$solver_status[indices] == "OK_CHOLESKY_SINGLE"
  ))
}, integer(1L)))

safe <- target_records$authenticated_planned_route == "CHOLESKY_BATCHED"
stable <- !safe
assert_true(
  length(which(safe)) > 0L && length(which(stable)) > 0L,
  "safe and stable target partitions are both non-empty"
)
route_status_conservation_exact <-
  identical(target_records$planned_route,
            target_records$authenticated_planned_route) &&
  all(target_records$authenticated_planned_route[stable] %in%
      c("AUGMENTED_QR", "AUGMENTED_SVD")) &&
  !anyNA(target_records$executed_route[safe]) &&
  all(target_records$executed_route[safe] == "CHOLESKY_BATCHED") &&
  all(is.na(target_records$executed_route[stable])) &&
  all(target_records$solver_status[safe] %in%
      c("OK_CHOLESKY_BATCHED", "OK_CHOLESKY_SINGLE")) &&
  all(target_records$solver_status[stable] ==
      "ERR_STABLE_PATH_NOT_IMPLEMENTED") &&
  !any(startsWith(target_records$solver_status[stable], "OK_")) &&
  all(target_records$reroute_reason == "") &&
  identical(batch_records$planned_cholesky_target_count,
            planned_cholesky) &&
  identical(batch_records$planned_qr_target_count, planned_qr) &&
  identical(batch_records$planned_svd_target_count, planned_svd) &&
  identical(batch_records$executed_cholesky_target_count,
            executed_cholesky) &&
  identical(batch_records$executed_qr_target_count, executed_qr) &&
  identical(batch_records$executed_svd_target_count, executed_svd) &&
  identical(batch_records$true_batched_target_count,
            batched_status_count) &&
  identical(batch_records$cholesky_single_target_count,
            single_status_count) &&
  all(batch_records$true_batched_subgroup_count ==
      as.integer(safe_count_by_setup >= 2L)) &&
  all(batch_records$true_batched_attempted_target_count ==
      batch_records$true_batched_target_count) &&
  all(batch_records$potrf_batched_call_count ==
      batch_records$true_batched_subgroup_count) &&
  all(batch_records$potrs_batched_call_count ==
      batch_records$true_batched_subgroup_count) &&
  all(batch_records$true_batched_kernel ==
      (all_safe & safe_count_by_setup >= 2L)) &&
  all(batch_records$stable_reroute_count == 0L) &&
  all(batch_records$cholesky_to_svd_count == 0L) &&
  all(batch_records$qr_to_svd_count == 0L) &&
  all(batch_records$route_status_conservation_exact)
assert_true(
  route_status_conservation_exact,
  "planned routes, executed routes, statuses, and batch diagnostics conserve targets"
)

error_fields <- c(
  "residual_max_abs_diff", "residual_relative_l2_diff",
  "fitted_max_abs_diff", "fitted_relative_l2_diff"
)
assert_true(
  all(error_fields %in% names(target_records)) &&
    all(vapply(target_records[safe, error_fields, drop = FALSE], function(x) {
      is.double(x) && all(is.finite(x)) && all(x >= 0) && all(x < 1e-7)
    }, logical(1L))) &&
    all(vapply(target_records[stable, error_fields, drop = FALSE], function(x) {
      is.double(x) && all(is.na(x) & !is.nan(x))
    }, logical(1L))),
  "all 172 safe targets match Phase 2 residual/fitted oracles below 1e-7"
)
assert_true(
  all(target_records$oracle_call_count[safe] == 1L) &&
    all(target_records$oracle_call_count[stable] == 0L) &&
    is_bare_sha_vector(target_records$authenticated_fitted_hash) &&
    is_bare_sha_vector(target_records$authenticated_residual_hash) &&
    is_bare_sha_vector(target_records$oracle_fitted_hash[safe]) &&
    is_bare_sha_vector(target_records$oracle_residual_hash[safe]) &&
    all(is.na(target_records$oracle_fitted_hash[stable])) &&
    all(is.na(target_records$oracle_residual_hash[stable])) &&
    all(target_records$oracle_fitted_hash_exact[safe] %in% TRUE) &&
    all(target_records$oracle_residual_hash_exact[safe] %in% TRUE) &&
    all(is.na(target_records$oracle_fitted_hash_exact[stable])) &&
    all(is.na(target_records$oracle_residual_hash_exact[stable])) &&
    identical(target_records$oracle_fitted_hash[safe],
              target_records$authenticated_fitted_hash[safe]) &&
    identical(target_records$oracle_residual_hash[safe],
              target_records$authenticated_residual_hash[safe]),
  "each safe target has one authenticated oracle/hash observation; stable has none"
)

assert_true(
  all(batch_records$coefficient_batch_finalize_call_count == 0L) &&
    identical(
      batch_records$fitted_batch_finalize_call_count,
      as.integer(!all_stable)
    ) &&
    identical(
      batch_records$residual_rss_batch_finalize_call_count,
      as.integer(!all_stable)
    ) &&
    all(batch_records$per_target_output_finalize_call_count == 0L) &&
    identical(batch_records$batch_output_finalized_target_count,
              safe_count_by_setup),
  "fitted/residual batch finalizers truthfully cover each safe-containing batch"
)

assert_true(
  all(batch_records$batch_call_count == 1L) &&
    all(batch_records$native_batch_call) &&
    all(batch_records$resource_snapshot_captured) &&
    all(batch_records$resource_instrumentation_version == 1L) &&
    all(batch_records$resource_allocation_count_after_solve -
        batch_records$resource_allocation_count_before_solve ==
        batch_records$cuda_device_allocation_count_during_solve +
        batch_records$cuda_host_allocation_count_during_solve) &&
    all(batch_records$resource_handle_create_count_after_solve -
        batch_records$resource_handle_create_count_before_solve ==
        batch_records$stream_create_count_during_solve +
        batch_records$event_create_count_during_solve +
        batch_records$cublas_handle_create_count_during_solve +
        batch_records$cusolver_handle_create_count_during_solve) &&
    all(batch_records$pre_shadow_materialize_call_count == 0L) &&
    all(batch_records$pre_shadow_materialize_target_count == 0L) &&
    all(batch_records$pre_shadow_d2h_bytes == 0) &&
    all(batch_records$explicit_oracle_shadow_observation == !all_stable) &&
    all(batch_records$post_shadow_materialize_call_count ==
        as.integer(!all_stable)) &&
    all(batch_records$post_shadow_materialize_target_count ==
        ifelse(all_stable, 0L, target_count_by_setup)) &&
    all(batch_records$post_shadow_d2h_bytes[!all_stable] > 0) &&
    all(batch_records$post_shadow_d2h_bytes[all_stable] == 0) &&
    all(batch_records$shadow_observation_purpose ==
        ifelse(all_stable, "none", "explicit-oracle-comparison")) &&
    all(batch_records$output_slot_release_count == 1L) &&
    all(!batch_records$output_slot_leased_after_release),
  "resource deltas fail closed and shadow D2H is explicitly oracle-only"
)

independent_exact <- list(
  iteration_subset_hash = catalog_records$iteration_subset_hash[[1L]],
  ordered_setup_key_digest = independent_setup_key_digest,
  ordered_target_key_digest = independent_target_key_digest,
  setup_count = as.integer(nrow(prepared_records)),
  target_count = as.integer(nrow(target_records)),
  batch_call_count = as.integer(sum(batch_records$batch_call_count)),
  all_safe_batch_count = as.integer(sum(all_safe)),
  mixed_batch_count = as.integer(sum(mixed)),
  all_stable_batch_count = as.integer(sum(all_stable)),
  cholesky_ok_count = as.integer(sum(
    target_records$solver_status %in%
      c("OK_CHOLESKY_BATCHED", "OK_CHOLESKY_SINGLE")
  )),
  true_batched_subgroup_count = as.integer(sum(
    batch_records$true_batched_subgroup_count
  )),
  true_batched_target_count = as.integer(sum(batched_status_count)),
  cholesky_single_target_count = as.integer(sum(single_status_count)),
  whole_batch_true_batched_count = as.integer(sum(
    batch_records$true_batched_kernel
  )),
  stable_not_implemented_count = as.integer(sum(
    target_records$solver_status == "ERR_STABLE_PATH_NOT_IMPLEMENTED"
  )),
  oracle_call_count = as.integer(sum(target_records$oracle_call_count)),
  oracle_fitted_hash_exact_count = as.integer(sum(
    target_records$oracle_fitted_hash_exact[safe]
  )),
  oracle_residual_hash_exact_count = as.integer(sum(
    target_records$oracle_residual_hash_exact[safe]
  )),
  setup_h2d_upload_count = as.integer(sum(
    prepared_records$setup_h2d_upload_count
  )),
  target_batch_h2d_call_count = as.integer(sum(
    batch_records$target_batch_h2d_call_count
  )),
  target_h2d_copy_count = as.integer(sum(
    batch_records$target_h2d_copy_count
  )),
  rhs_device_build_count = as.integer(sum(
    batch_records$rhs_device_build_count
  )),
  full_cuda_data_plane = all(batch_records$full_cuda_data_plane),
  invalid_output_init_count = as.integer(sum(
    batch_records$invalid_output_init_count
  )),
  coefficient_batch_finalize_call_count = as.integer(sum(
    batch_records$coefficient_batch_finalize_call_count
  )),
  fitted_batch_finalize_call_count = as.integer(sum(
    batch_records$fitted_batch_finalize_call_count
  )),
  residual_rss_batch_finalize_call_count = as.integer(sum(
    batch_records$residual_rss_batch_finalize_call_count
  )),
  per_target_output_finalize_call_count = as.integer(sum(
    batch_records$per_target_output_finalize_call_count
  )),
  batch_output_finalized_target_count = as.integer(sum(
    batch_records$batch_output_finalized_target_count
  )),
  workspace_grow_count_after_warmup = runtime_workspace_grow_delta,
  per_target_allocation_count_after_warmup = as.integer(sum(
    batch_records$per_target_allocation_count_after_warmup
  )),
  per_target_handle_create_count = as.integer(sum(
    batch_records$per_target_handle_create_count
  )),
  cuda_device_synchronize_count = runtime_synchronize_delta,
  implicit_residual_d2h_count = as.integer(sum(
    batch_records$implicit_residual_d2h_count
  )),
  all_output_slot_leases_released =
    all(!prepared_records$output_slot_leased_after_release) &&
      all(batch_records$output_slot_release_count == 1L &
          !batch_records$output_slot_leased_after_release),
  cpu_fallback_count = as.integer(sum(batch_records$cpu_fallback_count)),
  unknown_fallback_count = as.integer(sum(
    batch_records$unknown_fallback_count
  ))
)

assert_true(
  identical(independent_exact, expected),
  paste0(
    "independent source records must reproduce every frozen count; observed=",
    paste(names(independent_exact), unlist(independent_exact),
          sep = "=", collapse = ",")
  )
)
assert_true(
  identical(summary, independent_exact),
  "implementation summary agrees with the independent multi-record recomputation"
)

cat(paste0(
  "PASS Phase 3B fixed-sp iteration batch gate; setups=",
  independent_exact$setup_count, "; targets=", independent_exact$target_count,
  "; batches=", independent_exact$batch_call_count,
  "; all-safe/mixed/all-stable=",
  independent_exact$all_safe_batch_count, "/",
  independent_exact$mixed_batch_count, "/",
  independent_exact$all_stable_batch_count,
  "; batched-subgroups/targets/single=",
  independent_exact$true_batched_subgroup_count, "/",
  independent_exact$true_batched_target_count, "/",
  independent_exact$cholesky_single_target_count,
  "; stable-errors=", independent_exact$stable_not_implemented_count, "\n"
))
