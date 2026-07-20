source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/native.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_identical <- function(actual, expected, message) {
  if (!identical(actual, expected)) fail(message)
}
assert_error_matching <- function(expression, pattern, message) {
  error <- tryCatch({
    force(expression)
    NULL
  }, error = identity)
  assert_true(
    inherits(error, "error") && grepl(
      pattern, conditionMessage(error), fixed = TRUE
    ),
    message
  )
}

assert_true(
  exists(
    "fastkpc_cuda_legacy_dcov_gamma_cpp_oracle",
    mode = "function", inherits = TRUE
  ),
  "Task 8 registered dCov authority wrapper is available"
)
strict_dcov_fixture <- as.double(seq_len(6L))
assert_error_matching(
  fastkpc_cuda_legacy_dcov_gamma_cpp_oracle(
    as.integer(strict_dcov_fixture), strict_dcov_fixture,
    numCol = 1L, index = 1
  ),
  "registered legacy dCov gamma inputs are malformed",
  "registered dCov authority rejects non-double residual vectors"
)
assert_error_matching(
  fastkpc_cuda_legacy_dcov_gamma_cpp_oracle(
    strict_dcov_fixture, strict_dcov_fixture,
    numCol = 1, index = 1
  ),
  "registered legacy dCov gamma inputs are malformed",
  "registered dCov authority rejects non-integer numCol"
)
task8_dcov_runtime_text <- paste(deparse(
  body(fastkpc_full_cuda_fixed_sp_run_qualification_dcov_parity),
  width.cutoff = 500L
), collapse = "\n")
assert_true(
  grepl(
    "oracle_fun = fastkpc_cuda_legacy_dcov_gamma_cpp_oracle",
    task8_dcov_runtime_text, fixed = TRUE
  ) && !grepl(
    "oracle_fun = fastkpc_legacy_dcov_gamma_cpp_oracle",
    task8_dcov_runtime_text, fixed = TRUE
  ),
  "Task 8 routes dCov authority through the registered CUDA library"
)

fixture_keys <- c(
  paste(rep("a", 64L), collapse = ""),
  paste(rep("b", 64L), collapse = ""),
  paste(rep("c", 64L), collapse = "")
)
fixture_owners <- c(
  paste(rep("1", 64L), collapse = ""),
  paste(rep("1", 64L), collapse = ""),
  paste(rep("2", 64L), collapse = "")
)
fixture_registry <- fastkpc_full_cuda_fixed_sp_residual_registry_create(
  expected_keys = fixture_keys,
  expected_owners = fixture_owners,
  n = 3L
)
fastkpc_full_cuda_fixed_sp_residual_registry_capture(
  fixture_registry,
  owner = fixture_owners[[1L]],
  target_keys = fixture_keys[1:2],
  residuals = matrix(as.double(seq_len(6L)), nrow = 3L)
)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_residual_registry_capture(
    fixture_registry,
    owner = fixture_owners[[1L]],
    target_keys = fixture_keys[[2L]],
    residuals = matrix(as.double(1:3), nrow = 3L)
  ),
  "duplicate residual key",
  "qualification residual registry rejects duplicate vectors"
)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_residual_registry_capture(
    fixture_registry,
    owner = fixture_owners[[3L]],
    target_keys = paste(rep("d", 64L), collapse = ""),
    residuals = matrix(as.double(1:3), nrow = 3L)
  ),
  "unexpected residual key",
  "qualification residual registry rejects extra vectors"
)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_residual_registry_validate(fixture_registry),
  "missing residual vectors",
  "qualification residual registry fails closed when incomplete"
)
fastkpc_full_cuda_fixed_sp_residual_registry_capture(
  fixture_registry,
  owner = fixture_owners[[3L]],
  target_keys = fixture_keys[[3L]],
  residuals = matrix(as.double(7:9), nrow = 3L)
)
assert_identical(
  fastkpc_full_cuda_fixed_sp_residual_registry_validate(fixture_registry),
  fixture_keys,
  "qualification residual registry authenticates exact insertion order"
)

mutation_registry <- fastkpc_full_cuda_fixed_sp_residual_registry_create(
  expected_keys = fixture_keys[[1L]],
  expected_owners = fixture_owners[[1L]],
  n = 3L
)
fastkpc_full_cuda_fixed_sp_residual_registry_capture(
  mutation_registry,
  owner = fixture_owners[[1L]],
  target_keys = fixture_keys[[1L]],
  residuals = matrix(as.double(1:3), nrow = 3L)
)
mutation_key <- fixture_keys[[1L]]
if (bindingIsLocked(mutation_key, mutation_registry)) {
  unlockBinding(mutation_key, mutation_registry)
}
assign(
  mutation_key,
  get(mutation_key, envir = mutation_registry, inherits = FALSE) + 1,
  envir = mutation_registry
)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_residual_registry_validate(
    mutation_registry
  ),
  "residual hash mismatch",
  paste(
    "qualification residual registry rejects a different finite vector",
    "after binding replacement"
  )
)
fastkpc_full_cuda_fixed_sp_residual_registry_clear(mutation_registry)

assert_true(
  all(vapply(fixture_keys, bindingIsLocked, logical(1L),
             env = fixture_registry)),
  "qualification residual registry locks every captured binding"
)
fixture_target_records <- data.frame(
  residual_key_sha256 = fixture_keys,
  residual_numeric_hash = vapply(fixture_keys, function(key) {
    fastkpc_full_cuda_census_metadata_hash(
      get(key, envir = fixture_registry, inherits = FALSE)
    )
  }, character(1L)),
  stringsAsFactors = FALSE
)
assert_identical(
  fastkpc_full_cuda_fixed_sp_residual_registry_validate(
    fixture_registry, target_records = fixture_target_records
  ),
  fixture_keys,
  "qualification residual registry matches Task 7 numeric hashes"
)
forged_target_records <- fixture_target_records
forged_target_records$residual_numeric_hash[[2L]] <- paste0(
  "f", substring(forged_target_records$residual_numeric_hash[[2L]], 2L)
)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_residual_registry_validate(
    fixture_registry, target_records = forged_target_records
  ),
  "Task 7 residual numeric hash mismatch",
  "qualification residual registry rejects forged Task 7 hash authority"
)

collect_named_calls <- function(node, name) {
  result <- list()
  if (is.call(node)) {
    if (is.symbol(node[[1L]]) && identical(as.character(node[[1L]]), name)) {
      result <- list(node)
    }
    for (index in seq_along(node)) {
      result <- c(result, collect_named_calls(node[[index]], name))
    }
  } else if (is.pairlist(node) || is.expression(node)) {
    for (index in seq_along(node)) {
      result <- c(result, collect_named_calls(node[[index]], name))
    }
  }
  result
}
dcov_runtime_body <- body(
  fastkpc_full_cuda_fixed_sp_run_qualification_dcov_parity
)
dcov_validation_calls <- collect_named_calls(
  dcov_runtime_body,
  "fastkpc_full_cuda_fixed_sp_residual_registry_validate"
)
dcov_backend_calls <- collect_named_calls(
  dcov_runtime_body,
  "fastkpc_full_cuda_fixed_sp_with_legacy_dcov_backend"
)
assert_identical(
  length(dcov_validation_calls), 2L,
  "qualification dCov validates authenticated residual hashes twice"
)
assert_identical(
  length(dcov_backend_calls), 1L,
  "qualification dCov has one pinned backend callback"
)
dcov_callback_body <- dcov_backend_calls[[1L]][[2L]][[3L]]
dcov_callback_expressions <- as.list(dcov_callback_body)[-1L]
dcov_callback_heads <- vapply(dcov_callback_expressions, function(node) {
  if (is.call(node) && is.symbol(node[[1L]])) {
    as.character(node[[1L]])
  } else {
    ""
  }
}, character(1L))
dcov_core_index <- which(dcov_callback_heads == "qualification_core")
dcov_immediate_validation_index <- which(
  dcov_callback_heads ==
    "fastkpc_full_cuda_fixed_sp_residual_registry_validate"
)
assert_true(
  identical(length(dcov_core_index), 1L) &&
    identical(dcov_immediate_validation_index, dcov_core_index - 1L) &&
    grepl(
      "target_records = target_records",
      paste(deparse(
        dcov_callback_expressions[[dcov_immediate_validation_index]],
        width.cutoff = 500L
      ), collapse = "\n"),
      fixed = TRUE
    ),
  "qualification dCov validates Task 7 hashes immediately before consumption"
)
qualification_runner_text <- paste(readLines(
  "fastkpc/tools/run_full_cuda_ci_fixed_sp_qualification.R",
  warn = FALSE
), collapse = "\n")
assert_true(
  grepl(
    "target_records = qualification$target_records",
    qualification_runner_text, fixed = TRUE
  ),
  "qualification runner supplies Task 7 records to dCov authentication"
)

invalid_registry <- fastkpc_full_cuda_fixed_sp_residual_registry_create(
  expected_keys = fixture_keys[1:2],
  expected_owners = fixture_owners[1:2],
  n = 3L
)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_residual_registry_capture(
    invalid_registry,
    owner = fixture_owners[[1L]],
    target_keys = rev(fixture_keys[1:2]),
    residuals = matrix(as.double(seq_len(6L)), nrow = 3L)
  ),
  "canonical residual order",
  "qualification residual registry rejects out-of-order vectors"
)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_residual_registry_capture(
    invalid_registry,
    owner = fixture_owners[[3L]],
    target_keys = fixture_keys[[1L]],
    residuals = matrix(as.double(1:3), nrow = 3L)
  ),
  "residual owner",
  "qualification residual registry rejects conflicting ownership"
)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_residual_registry_capture(
    invalid_registry,
    owner = fixture_owners[[1L]],
    target_keys = fixture_keys[[1L]],
    residuals = matrix(c(1, 2, Inf), nrow = 3L)
  ),
  "nonfinite",
  "qualification residual registry rejects nonfinite vectors"
)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_residual_registry_capture(
    invalid_registry,
    owner = fixture_owners[[1L]],
    target_keys = fixture_keys[[1L]],
    residuals = matrix(as.double(1:2), nrow = 2L)
  ),
  "residual length",
  "qualification residual registry rejects wrong-length vectors"
)

prior_backend <- Sys.getenv(
  "FASTKPC_LEGACY_DCOV_GAMMA_BACKEND", unset = NA_character_
)
prior_low_rank <- Sys.getenv(
  "FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK", unset = NA_character_
)
on.exit({
  fastkpc_full_cuda_prepared_s_restore_environment(
    "FASTKPC_LEGACY_DCOV_GAMMA_BACKEND", prior_backend
  )
  fastkpc_full_cuda_prepared_s_restore_environment(
    "FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK", prior_low_rank
  )
}, add = TRUE)
Sys.setenv(
  FASTKPC_LEGACY_DCOV_GAMMA_BACKEND = "fixture-prior-backend",
  FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = "fixture-prior-low-rank"
)
observed_environment <-
  fastkpc_full_cuda_fixed_sp_with_legacy_dcov_backend(function() {
    c(
      Sys.getenv("FASTKPC_LEGACY_DCOV_GAMMA_BACKEND"),
      Sys.getenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK")
    )
  })
assert_identical(
  observed_environment, c("cpp", "spectra"),
  "qualification dCov helper installs the pinned backend environment"
)
assert_identical(
  c(
    Sys.getenv("FASTKPC_LEGACY_DCOV_GAMMA_BACKEND"),
    Sys.getenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK")
  ),
  c("fixture-prior-backend", "fixture-prior-low-rank"),
  "qualification dCov helper restores backend environment after success"
)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_with_legacy_dcov_backend(function() {
    stop("injected dCov fixture error", call. = FALSE)
  }),
  "injected dCov fixture error",
  "qualification dCov helper propagates backend errors"
)
assert_identical(
  c(
    Sys.getenv("FASTKPC_LEGACY_DCOV_GAMMA_BACKEND"),
    Sys.getenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK")
  ),
  c("fixture-prior-backend", "fixture-prior-low-rank"),
  "qualification dCov helper restores backend environment after failure"
)
qualification_dcov_core <-
  fastkpc_full_cuda_fixed_sp_qualification_dcov_core()
prepared_s_dcov_core <-
  get(".fastkpc_full_cuda_run_prepared_s_dcov_parity_core", mode = "function")
assert_true(
  is.function(qualification_dcov_core) &&
    identical(formals(qualification_dcov_core),
              formals(prepared_s_dcov_core)) &&
    identical(environment(qualification_dcov_core),
              environment(prepared_s_dcov_core)),
  "qualification dCov adapter reuses the qualified core interface/environment"
)
qualification_core_text <- paste(deparse(
  body(qualification_dcov_core), width.cutoff = 500L
), collapse = "\n")
prepared_core_text <- paste(deparse(
  body(prepared_s_dcov_core), width.cutoff = 500L
), collapse = "\n")
assert_true(
  grepl("max_drift > 1e-10", qualification_core_text, fixed = TRUE) &&
    grepl("exceeds 1e-10", qualification_core_text, fixed = TRUE) &&
    identical(
      gsub("1e-10", "1e-12", qualification_core_text, fixed = TRUE),
      prepared_core_text
  ),
  "qualification adapter changes only the qualified core tolerance literal"
)
collect_assignment_calls <- function(node) {
  result <- list()
  if (is.call(node)) {
    if (is.symbol(node[[1L]]) && identical(as.character(node[[1L]]), "<-")) {
      result <- list(node)
    }
    for (index in seq_along(node)) {
      result <- c(result, collect_assignment_calls(node[[index]]))
    }
  } else if (is.pairlist(node) || is.expression(node)) {
    for (index in seq_along(node)) {
      result <- c(result, collect_assignment_calls(node[[index]]))
    }
  }
  result
}
qualification_assignment_calls <- collect_assignment_calls(
  body(qualification_dcov_core)
)
assert_true(
  length(qualification_assignment_calls) > 0L && all(vapply(
    qualification_assignment_calls, length, integer(1L)
  ) == 3L),
  "qualification adapter preserves every assignment call arity"
)
rownames_assignment <- Filter(function(node) {
  lhs <- node[[2L]]
  is.call(lhs) && is.symbol(lhs[[1L]]) &&
    identical(as.character(lhs[[1L]]), "rownames")
}, qualification_assignment_calls)
assert_identical(
  length(rownames_assignment), 1L,
  "qualification adapter preserves the rownames assignment expression"
)
assignment_fixture <- new.env(parent = baseenv())
assignment_fixture$parity_rows <- data.frame(value = 1:2)
assignment_error <- tryCatch({
  eval(rownames_assignment[[1L]], envir = assignment_fixture)
  NULL
}, error = identity)
assert_true(
  is.null(assignment_error),
  "qualification adapter rownames assignment executes without corruption"
)
fixture_registry_state <-
  fastkpc_full_cuda_fixed_sp_residual_registry_state(fixture_registry)
fastkpc_full_cuda_fixed_sp_residual_registry_clear(fixture_registry)
assert_identical(
  ls(fixture_registry, all.names = TRUE), character(),
  "qualification residual registry is cleared after consumption"
)
assert_true(
  isTRUE(fixture_registry_state$cleared) &&
    identical(fixture_registry_state$expected_keys, character()) &&
    identical(fixture_registry_state$expected_owners, character()) &&
    identical(fixture_registry_state$captured_hashes, character()) &&
    identical(fixture_registry_state$n, 0L) &&
    identical(fixture_registry_state$next_index, 0L),
  "qualification residual registry clears all hash and corpus state"
)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_residual_registry_validate(fixture_registry),
  "residual registry was cleared",
  "qualification residual registry validation is unusable after clear"
)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_residual_registry_capture(
    fixture_registry,
    owner = fixture_owners[[1L]],
    target_keys = fixture_keys[[1L]],
    residuals = matrix(as.double(1:3), nrow = 3L)
  ),
  "residual registry was cleared",
  "qualification residual registry capture is unusable after clear"
)

fixture_logical_all <- readRDS(file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1",
  "qualification_logical_tests.rds"
))
fixture_logical_indices <- c(1L, which(fixture_logical_all$near_alpha)[[1L]])
fixture_logical <- fixture_logical_all[
  fixture_logical_indices, , drop = FALSE
]
fixture_diagnostic <- list(
  n = 351L,
  numCol = 35L,
  index = 1,
  lowrank_mode = "spectra",
  input_ms = 1,
  distance_ms = 2,
  lowrank_ms = 3,
  lowrank_eig_ms = 2,
  lowrank_select_ms = 0.5,
  lowrank_center_ms = 0.4,
  lowrank_unaccounted_ms = 0.1,
  lowrank_full_eig_count = 0L,
  lowrank_spectra_count = 2L,
  lowrank_spectra_converged_count = 2L,
  lowrank_spectra_failed_count = 0L,
  lowrank_spectra_fallback_full_eig_count = 0L,
  lowrank_spectra_iterations = 4L,
  lowrank_spectra_nconv = 70L,
  lowrank_spectra_ncv = 71L,
  lowrank_spectra_tol = 1e-10,
  lowrank_spectra_matvec_count = 0L,
  lowrank_spectra_matvec_ms = 0,
  statistic_ms = 1,
  moment_ms = 1,
  pgamma_ms = 1,
  accounted_ms = 9,
  unaccounted_ms = 0,
  total_ms = 9
)
fixture_p_value <- as.double(fixture_logical$reference_p_value)
fixture_parity <- data.frame(
  parity_scope = rep("qualification", 2L),
  logical_sequence_id = as.integer(fixture_logical$logical_sequence_id),
  residual_key_x = as.character(fixture_logical$residual_key_x),
  residual_key_y = as.character(fixture_logical$residual_key_y),
  index = rep(1L, 2L),
  numCol = rep(35L, 2L),
  alpha = as.double(fixture_logical$alpha),
  reference_p_value = as.double(fixture_logical$reference_p_value),
  p_value = fixture_p_value,
  p_value_drift = rep(0, 2L),
  absolute_p_value_drift = rep(0, 2L),
  p_value_exact = rep(TRUE, 2L),
  reference_signed_alpha_distance =
    as.double(fixture_logical$reference_p_value - fixture_logical$alpha),
  signed_alpha_distance =
    as.double(fixture_p_value - fixture_logical$alpha),
  reference_decision = as.character(fixture_logical$reference_decision),
  reference_independent = as.logical(
    fixture_logical$reference_independent
  ),
  decision = as.character(fixture_logical$reference_decision),
  decision_identical = rep(TRUE, 2L),
  spectra_no_fallback = rep(TRUE, 2L),
  stringsAsFactors = FALSE
)
fixture_parity$diagnostics <- I(list(
  fixture_diagnostic, fixture_diagnostic
))
fixture_dcov <- fastkpc_full_cuda_fixed_sp_build_qualification_dcov_records(
  fixture_logical, fixture_parity
)
fixture_dcov_summary <-
  fastkpc_full_cuda_fixed_sp_summarize_qualification_dcov_records(
    fixture_dcov,
    fixture_logical,
    sort(unique(c(
      fixture_logical$residual_key_x,
      fixture_logical$residual_key_y
    )), method = "radix")
  )
assert_identical(
  fixture_dcov$logical_sequence_id,
  as.integer(fixture_logical$logical_sequence_id),
  "qualification dCov rows preserve authenticated logical order"
)
assert_identical(
  fixture_dcov$near_alpha, as.logical(fixture_logical$near_alpha),
  "qualification dCov rows preserve authenticated near-alpha labels"
)
assert_true(
  all(!fixture_dcov$decision_flip) &&
    all(!fixture_dcov$backend_error) &&
    all(!fixture_dcov$spectra_fallback),
  "qualification dCov rows derive clean decision/backend flags"
)
assert_identical(
  fixture_dcov_summary$qualification_dcov_logical_test_count, 2L,
  "qualification dCov summary derives row count"
)
assert_identical(
  fixture_dcov_summary$qualification_dcov_near_alpha_count, 1L,
  "qualification dCov summary derives near-alpha count"
)
assert_true(
  grepl(
    "^[0-9a-f]{64}$",
    fixture_dcov_summary$qualification_dcov_rows_hash
  ),
  "qualification dCov summary derives deterministic rows hash"
)
assert_identical(
  formals(fastkpc_run_full_cuda_fixed_sp_phase3c_iteration)[[
    "return_residual_registry"
  ]],
  FALSE,
  "Phase 3C iteration keeps residual-registry capture opt-in"
)
assert_identical(
  formals(fastkpc_run_full_cuda_fixed_sp_phase3c_qualification)[[
    "return_residual_registry"
  ]],
  FALSE,
  "Phase 3C qualification keeps its Task 7 default result schema"
)
assert_identical(
  fastkpc_full_cuda_fixed_sp_qualification_payload_names(),
  c(
    "target_parity.rds", "target_parity.csv",
    "batch_metrics.rds", "batch_metrics.csv",
    "setup_metrics.rds", "setup_metrics.csv",
    "qualification_dcov_parity.rds",
    "qualification_dcov_parity.csv",
    "runtime_metrics.csv", "stage_timing.csv", "fallbacks.csv",
    "failures.csv", "native_build_dependencies.csv", "commands.txt",
    "environment.txt"
  ),
  "qualification v4 payload surface includes provenance and dCov evidence"
)
assert_identical(
  fastkpc_full_cuda_fixed_sp_qualification_summary_schema()$dcov_names,
  c(
    "qualification_dcov_logical_test_count",
    "qualification_dcov_near_alpha_count",
    "qualification_dcov_unique_residual_key_count",
    "qualification_dcov_max_absolute_p_value_difference",
    "qualification_dcov_decision_flip_count",
    "qualification_dcov_near_alpha_decision_flip_count",
    "qualification_dcov_backend_error_count",
    "qualification_dcov_spectra_fallback_count",
    "qualification_dcov_logical_ids_hash",
    "qualification_dcov_residual_key_hash",
    "qualification_dcov_rows_hash"
  ),
  "qualification summary freezes writer-derived dCov field order"
)
json_roundtrip_path <- tempfile("fastkpc_phase3c_dcov_json_")
on.exit(unlink(json_roundtrip_path, force = TRUE), add = TRUE)
json_roundtrip_value <- 6.9555694537370982e-11
fastkpc_full_cuda_fixed_sp_write_qualification_json(
  list(maximum = json_roundtrip_value), json_roundtrip_path
)
json_roundtrip <- jsonlite::fromJSON(
  json_roundtrip_path, simplifyVector = FALSE
)
assert_identical(
  json_roundtrip$maximum, json_roundtrip_value,
  "qualification JSON preserves exact Task 8 maximum-drift doubles"
)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP Phase 3C qualification dCov gate\n")
  quit(save = "no", status = 0L)
}

authority_fixture <- readRDS(file.path(
  "fastkpc", "tests", "fixtures", "legacy_dcov_gamma_oracle_v1.rds"
))
authority_cases <- authority_fixture[c(1L, 2L, 6L)]
assert_true(
  requireNamespace("Rcpp", quietly = TRUE),
  "Rcpp is required for registered dCov authority provenance"
)
loaded_dll_paths <- function() {
  sort(vapply(getLoadedDLLs(), function(dll) {
    normalizePath(dll[["path"]], winslash = "/", mustWork = FALSE)
  }, character(1L)), method = "radix")
}
authority_dlls_before <- loaded_dll_paths()
authority_build_calls <- 0L
authority_source_cpp_calls <- 0L
authority_original_build <- build_fastkpc_native
authority_source_cpp_traced <- FALSE
restore_authority_guards <- function() {
  if (isTRUE(authority_source_cpp_traced)) {
    suppressMessages(invisible(untrace(
      "sourceCpp", where = asNamespace("Rcpp")
    )))
    authority_source_cpp_traced <<- FALSE
  }
  assign(
    "build_fastkpc_native", authority_original_build,
    envir = .GlobalEnv
  )
  invisible(NULL)
}
on.exit(restore_authority_guards(), add = TRUE)
assign("build_fastkpc_native", function(...) {
  authority_build_calls <<- authority_build_calls + 1L
  stop("Task 8 invoked build_fastkpc_native", call. = FALSE)
}, envir = .GlobalEnv)
suppressMessages(invisible(trace(
  "sourceCpp",
  tracer = quote({
    assign(
      "authority_source_cpp_calls",
      get("authority_source_cpp_calls", envir = .GlobalEnv) + 1L,
      envir = .GlobalEnv
    )
    stop("Task 8 invoked Rcpp::sourceCpp", call. = FALSE)
  }),
  where = asNamespace("Rcpp"), print = FALSE
)))
authority_source_cpp_traced <- TRUE
authority_native <- fastkpc_full_cuda_fixed_sp_with_legacy_dcov_backend(
  function() lapply(authority_cases, function(case) {
    meta <- case$meta[1L, , drop = FALSE]
    fastkpc_cuda_legacy_dcov_gamma_cpp_oracle(
      as.numeric(case$residuals$rx),
      as.numeric(case$residuals$ry),
      numCol = as.integer(meta$numCol),
      index = as.integer(meta$index)
    )
  })
)
authority_dlls_after_native <- loaded_dll_paths()
restore_authority_guards()
assert_identical(
  c(authority_build_calls, authority_source_cpp_calls), c(0L, 0L),
  "Task 8 native authority invokes neither build_fastkpc_native nor sourceCpp"
)
authority_new_native_dlls <- setdiff(
  authority_dlls_after_native, authority_dlls_before
)
assert_true(
  length(authority_new_native_dlls) <= 1L &&
    all(basename(authority_new_native_dlls) == "fastkpc_cuda.so") &&
    !exists(
      "legacy_dcov_gamma_cpp_oracle_export",
      mode = "function", inherits = TRUE
    ),
  "Task 8 native authority loads no sourceCpp dCov DLL"
)
authority_source_cpp <- fastkpc_full_cuda_fixed_sp_with_legacy_dcov_backend(
  function() lapply(authority_cases, function(case) {
    meta <- case$meta[1L, , drop = FALSE]
    fastkpc_legacy_dcov_gamma_cpp_oracle(
      as.numeric(case$residuals$rx),
      as.numeric(case$residuals$ry),
      numCol = as.integer(meta$numCol),
      index = as.double(meta$index)
    )
  })
)
authority_numeric_fields <- c(
  "p.value", "nV2", "mean", "variance", "statistic", "estimate"
)
for (index in seq_along(authority_cases)) {
  native <- authority_native[[index]]
  source_cpp <- authority_source_cpp[[index]]
  native_diagnostic <-
    fastkpc_full_cuda_fixed_sp_sanitize_dcov_diagnostic(
      native$diagnostics
    )
  source_cpp_diagnostic <-
    fastkpc_full_cuda_fixed_sp_sanitize_dcov_diagnostic(
      source_cpp$diagnostics
    )
  assert_true(
    identical(names(native), names(source_cpp)) &&
      identical(
        native[authority_numeric_fields],
        source_cpp[authority_numeric_fields]
      ) &&
      identical(native$estimates, source_cpp$estimates) &&
      identical(native_diagnostic, source_cpp_diagnostic) &&
      identical(native_diagnostic$lowrank_mode, "spectra") &&
      identical(native_diagnostic$lowrank_spectra_count, 2L) &&
      identical(native_diagnostic$lowrank_spectra_failed_count, 0L) &&
      identical(
        native_diagnostic$lowrank_spectra_fallback_full_eig_count, 0L
      ),
    paste(
      "registered dCov authority matches sourceCpp Spectra case",
      authority_cases[[index]]$meta$case_id[[1L]]
    )
  )
}

artifact_dir <- file.path(
  "fastkpc", "artifacts", "full_cuda_ci",
  "fixed_sp_cuda_qualification_v1"
)
dcov_rds_path <- file.path(
  artifact_dir, "qualification_dcov_parity.rds"
)
dcov_csv_path <- file.path(
  artifact_dir, "qualification_dcov_parity.csv"
)

assert_true(
  file.exists(dcov_rds_path) && file.exists(dcov_csv_path),
  "qualification artifact lacks dCov rows/files"
)

assert_true(requireNamespace("jsonlite", quietly = TRUE), "jsonlite required")
assert_true(requireNamespace("digest", quietly = TRUE), "digest required")

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
expected_payload_names <- c(
  "target_parity.rds", "target_parity.csv",
  "batch_metrics.rds", "batch_metrics.csv",
  "setup_metrics.rds", "setup_metrics.csv",
  "qualification_dcov_parity.rds", "qualification_dcov_parity.csv",
  "runtime_metrics.csv", "stage_timing.csv", "fallbacks.csv",
  "failures.csv", "commands.txt", "environment.txt"
)
expected_files <- c(expected_payload_names, "manifest.json", "summary.json")
assert_identical(
  sort(list.files(artifact_dir, all.files = FALSE), method = "radix"),
  sort(expected_files, method = "radix"),
  "qualification dCov artifact has exact v2 payload surface"
)

manifest <- jsonlite::fromJSON(
  file.path(artifact_dir, "manifest.json"), simplifyVector = FALSE
)
summary <- jsonlite::fromJSON(
  file.path(artifact_dir, "summary.json"), simplifyVector = FALSE
)
assert_true(
  identical(manifest$schema_version,
            "full-cuda-ci-fixed-sp-qualification-v4") &&
    identical(summary$artifact_schema_version,
              "full-cuda-ci-fixed-sp-qualification-v4") &&
    identical(manifest$scope, "qualification") &&
    identical(summary$scope, "qualification") &&
    !"pass" %in% names(manifest) && !"pass" %in% names(summary),
  "qualification dCov artifact schema/scope is exact and claim-free"
)
assert_identical(
  unname(unlist(manifest$publication_order, use.names = FALSE)),
  c(expected_payload_names, "manifest.json", "summary.json"),
  "qualification dCov publication order is exact"
)
payload_hashes <- unlist(manifest$payload_file_sha256, use.names = TRUE)
assert_identical(
  names(payload_hashes), expected_payload_names,
  "qualification dCov manifest hashes every payload in order"
)
actual_payload_hashes <- vapply(expected_payload_names, function(name) {
  digest::digest(
    file = file.path(artifact_dir, name), algo = "sha256",
    serialize = FALSE
  )
}, character(1L))
assert_identical(
  unname(payload_hashes), unname(actual_payload_hashes),
  "qualification dCov payload hashes match published bytes"
)

catalog <- fastkpc_full_cuda_open_fixed_sp_catalog(
  phase0_dir = phase0_dir,
  phase1_dir = phase1_dir,
  phase2_dir = phase2_dir,
  data_path = data_path
)
qualification_scope <- fastkpc_full_cuda_fixed_sp_scope(
  catalog, "qualification"
)
logical_path <- file.path(phase2_dir, "qualification_logical_tests.rds")
logical_sha256 <- digest::digest(
  file = logical_path, algo = "sha256", serialize = FALSE
)
assert_identical(
  logical_sha256,
  "da7bfb2e13606f00523c8fcbaca87848bf89c784a319c8e98be7b0856653aff0",
  "qualification logical RDS matches frozen source-controlled SHA"
)
phase2_manifest <- jsonlite::fromJSON(
  file.path(phase2_dir, "manifest.json"), simplifyVector = FALSE
)
assert_identical(
  phase2_manifest$semantic_file_sha256[[
    "qualification_logical_tests_rds"
  ]],
  logical_sha256,
  "Phase 2 manifest authenticates qualification logical RDS"
)
assert_identical(
  manifest$qualification_logical_tests_sha256, logical_sha256,
  "qualification manifest binds authenticated logical RDS"
)
logical_tests <- readRDS(logical_path)
expected_logical <- fastkpc_full_cuda_prepared_s_selection_for_scope(
  catalog$inputs, "qualification"
)$logical_tests
assert_true(
  is.data.frame(logical_tests) && is.data.frame(expected_logical) &&
    identical(names(logical_tests), names(expected_logical)) &&
    nrow(logical_tests) == 3808L &&
    all(vapply(names(logical_tests), function(field) {
      identical(logical_tests[[field]], expected_logical[[field]])
    }, logical(1L))) &&
    identical(logical_tests$logical_sequence_id,
              sort(logical_tests$logical_sequence_id, method = "radix")) &&
    !anyDuplicated(logical_tests$logical_sequence_id) &&
    identical(sum(logical_tests$near_alpha), 1478L),
  "qualification logical rows independently reopen in canonical order"
)

expected_dcov_names <- c(
  "parity_scope", "logical_sequence_id", "source_sequence_id",
  "source_task_index", "level", "x", "y", "S_key", "S_size",
  "formula_class", "reference_p_value", "alpha", "reference_decision",
  "reference_independent", "deletes_edge", "selected_sepset",
  "signed_distance_from_alpha", "absolute_distance_from_alpha",
  "signed_log_ratio_from_alpha", "absolute_log_distance_from_alpha",
  "residual_key_x", "residual_key_y", "near_alpha",
  "selection_reasons", "index", "numCol", "backend",
  "low_rank_backend", "p_value", "p_value_difference",
  "absolute_p_value_difference", "p_value_exact",
  "signed_alpha_distance", "decision", "independent", "decision_flip",
  "backend_error", "spectra_fallback", "diagnostics"
)
integer_fields <- c(
  "logical_sequence_id", "source_sequence_id", "source_task_index",
  "level", "x", "y", "S_size", "index", "numCol"
)
double_fields <- c(
  "reference_p_value", "alpha", "signed_distance_from_alpha",
  "absolute_distance_from_alpha", "signed_log_ratio_from_alpha",
  "absolute_log_distance_from_alpha", "p_value",
  "p_value_difference", "absolute_p_value_difference",
  "signed_alpha_distance"
)
logical_fields <- c(
  "reference_independent", "deletes_edge", "selected_sepset",
  "near_alpha", "p_value_exact", "independent", "decision_flip",
  "backend_error", "spectra_fallback"
)
list_fields <- c("selection_reasons", "diagnostics")
dcov_types <- setNames(rep.int("character", length(expected_dcov_names)),
                       expected_dcov_names)
dcov_types[integer_fields] <- "integer"
dcov_types[double_fields] <- "double"
dcov_types[logical_fields] <- "logical"
dcov_types[list_fields] <- "list"
dcov <- readRDS(dcov_rds_path)
assert_true(
  is.data.frame(dcov) && nrow(dcov) == 3808L &&
    identical(names(dcov), expected_dcov_names) &&
    all(vapply(expected_dcov_names, function(field) {
      value <- dcov[[field]]
      if (field %in% list_fields) {
        typeof(value) == "list" && length(value) == 3808L &&
          is.object(value) &&
          identical(attributes(value), list(class = "AsIs"))
      } else {
        typeof(value) == dcov_types[[field]] && length(value) == 3808L &&
          !is.object(value) && is.null(attributes(value)) && !anyNA(value)
      }
    }, logical(1L))),
  "qualification dCov RDS exact names/types are frozen"
)
assert_true(
  all(vapply(dcov$selection_reasons, function(value) {
    typeof(value) == "character" && !is.object(value) &&
      is.null(attributes(value)) && !anyNA(value)
  }, logical(1L))),
  "qualification dCov selection-reason list projection is exact"
)

expected_diagnostic_names <- c(
  "n", "numCol", "index", "lowrank_mode",
  "lowrank_full_eig_count", "lowrank_spectra_count",
  "lowrank_spectra_converged_count", "lowrank_spectra_failed_count",
  "lowrank_spectra_fallback_full_eig_count",
  "lowrank_spectra_iterations", "lowrank_spectra_nconv",
  "lowrank_spectra_ncv", "lowrank_spectra_tol",
  "lowrank_spectra_matvec_count"
)
diagnostic_types <- setNames(
  rep.int("integer", length(expected_diagnostic_names)),
  expected_diagnostic_names
)
diagnostic_types[c("index", "lowrank_spectra_tol")] <- "double"
diagnostic_types[["lowrank_mode"]] <- "character"
diagnostics_exact <- vapply(dcov$diagnostics, function(diagnostic) {
  is.list(diagnostic) && !is.object(diagnostic) &&
    identical(names(diagnostic), expected_diagnostic_names) &&
    all(vapply(expected_diagnostic_names, function(field) {
      value <- diagnostic[[field]]
      typeof(value) == diagnostic_types[[field]] && length(value) == 1L &&
        !is.object(value) && is.null(attributes(value)) && !anyNA(value) &&
        if (typeof(value) %in% c("integer", "double")) {
          is.finite(value)
        } else {
          nzchar(value)
        }
    }, logical(1L))) && identical(diagnostic$n, 351L) &&
    identical(diagnostic$numCol, 35L) && identical(diagnostic$index, 1) &&
    identical(diagnostic$lowrank_mode, "spectra")
}, logical(1L))
assert_true(all(diagnostics_exact), "dCov diagnostic provenance is exact")

for (field in names(logical_tests)) {
  assert_identical(
    dcov[[field]], logical_tests[[field]],
    paste("dCov rows preserve authenticated logical field", field)
  )
}
endpoint_keys <- sort(unique(c(
  logical_tests$residual_key_x, logical_tests$residual_key_y
)), method = "radix")
target_keys <- sort(
  as.character(qualification_scope$target_rows$residual_key_sha256),
  method = "radix"
)
assert_true(
  length(endpoint_keys) == 6143L && identical(endpoint_keys, target_keys),
  "qualification dCov endpoints equal authenticated residual corpus"
)

p_value_difference <- dcov$p_value - logical_tests$reference_p_value
absolute_difference <- abs(p_value_difference)
independent <- dcov$p_value >= logical_tests$alpha
decision <- ifelse(independent, "independent", "dependent")
decision_flip <- independent != logical_tests$reference_independent
p_value_exact <- vapply(seq_len(nrow(dcov)), function(index) {
  identical(
    dcov$p_value[[index]], logical_tests$reference_p_value[[index]]
  )
}, logical(1L))
spectra_fallback <- vapply(dcov$diagnostics, function(diagnostic) {
  diagnostic$lowrank_full_eig_count != 0L ||
    diagnostic$lowrank_spectra_failed_count != 0L ||
    diagnostic$lowrank_spectra_fallback_full_eig_count != 0L
}, logical(1L))
assert_true(
  identical(dcov$parity_scope, rep("qualification", 3808L)) &&
    identical(dcov$index, rep(1L, 3808L)) &&
    identical(dcov$numCol, rep(35L, 3808L)) &&
    identical(dcov$backend, rep("cpp", 3808L)) &&
    identical(dcov$low_rank_backend, rep("spectra", 3808L)) &&
    identical(dcov$p_value_difference, p_value_difference) &&
    identical(dcov$absolute_p_value_difference, absolute_difference) &&
    identical(dcov$p_value_exact, p_value_exact) &&
    identical(dcov$signed_alpha_distance,
              dcov$p_value - logical_tests$alpha) &&
    identical(dcov$decision, decision) &&
    identical(dcov$independent, independent) &&
    identical(dcov$decision_flip, decision_flip) &&
    identical(dcov$backend_error, rep(FALSE, 3808L)) &&
    identical(dcov$spectra_fallback, spectra_fallback),
  "qualification dCov candidate fields are independently derived"
)

max_difference <- max(absolute_difference)
decision_flip_count <- as.integer(sum(decision_flip))
near_alpha_flip_count <- as.integer(sum(
  logical_tests$near_alpha & decision_flip
))
backend_error_count <- as.integer(sum(dcov$backend_error))
spectra_fallback_count <- as.integer(sum(spectra_fallback))
assert_true(
  identical(nrow(dcov), 3808L) &&
    identical(sum(dcov$near_alpha), 1478L) &&
    is.finite(max_difference) && max_difference < 1e-10 &&
    identical(decision_flip_count, 0L) &&
    identical(near_alpha_flip_count, 0L) &&
    identical(backend_error_count, 0L) &&
    identical(spectra_fallback_count, 0L),
  "qualification dCov hard numerical/decision gate passes"
)

logical_ids_hash <- fastkpc_full_cuda_census_metadata_hash(
  dcov$logical_sequence_id
)
residual_key_hash <- fastkpc_full_cuda_census_key_set_hash(endpoint_keys)
rows_hash <- fastkpc_full_cuda_census_frame_hash(dcov)
recomputed_summary <- list(
  qualification_dcov_logical_test_count = 3808L,
  qualification_dcov_near_alpha_count = 1478L,
  qualification_dcov_unique_residual_key_count = 6143L,
  qualification_dcov_max_absolute_p_value_difference =
    as.double(max_difference),
  qualification_dcov_decision_flip_count = decision_flip_count,
  qualification_dcov_near_alpha_decision_flip_count =
    near_alpha_flip_count,
  qualification_dcov_backend_error_count = backend_error_count,
  qualification_dcov_spectra_fallback_count = spectra_fallback_count,
  qualification_dcov_logical_ids_hash = logical_ids_hash,
  qualification_dcov_residual_key_hash = residual_key_hash,
  qualification_dcov_rows_hash = rows_hash
)
assert_true(
  all(vapply(names(recomputed_summary), function(field) {
    identical(summary[[field]], recomputed_summary[[field]])
  }, logical(1L))) &&
    identical(manifest$qualification_dcov_logical_ids_hash,
              logical_ids_hash) &&
    identical(manifest$qualification_dcov_residual_key_hash,
              residual_key_hash) &&
    identical(manifest$qualification_dcov_rows_hash, rows_hash),
  "qualification dCov summary/manifest claims match recomputed rows"
)
claims_match <- function(claims) all(vapply(
  names(recomputed_summary),
  function(field) identical(claims[[field]], recomputed_summary[[field]]),
  logical(1L)
))
claim_rejections <- vapply(names(recomputed_summary), function(field) {
  tampered <- summary
  value <- tampered[[field]]
  tampered[[field]] <- switch(
    typeof(value),
    integer = value + 1L,
    double = value + 1,
    character = paste0("f", substring(value, 2L)),
    fail("unsupported dCov summary claim type")
  )
  !claims_match(tampered)
}, logical(1L))
assert_true(
  all(claim_rejections),
  "independent dCov gate rejects every supplied summary claim mutation"
)

csv_projection <- dcov
for (field in names(csv_projection)[vapply(
  csv_projection, is.list, logical(1L)
)]) {
  csv_projection[[field]] <- vapply(
    csv_projection[[field]],
    function(value) if (length(value) == 0L) "" else
      paste(value, collapse = ";"),
    character(1L)
  )
}
expected_csv_path <- tempfile("fastkpc_phase3c_dcov_csv_")
on.exit(unlink(expected_csv_path, force = TRUE), add = TRUE)
utils::write.csv(
  csv_projection, expected_csv_path, row.names = FALSE, na = ""
)
read_raw <- function(path) {
  connection <- file(path, open = "rb")
  on.exit(close(connection), add = TRUE)
  readBin(connection, what = "raw", n = file.info(path)$size)
}
assert_identical(
  read_raw(dcov_csv_path), read_raw(expected_csv_path),
  "qualification dCov CSV is exact deterministic RDS projection"
)
assert_true(
  !any(vapply(dcov, function(column) {
    is.list(column) && any(vapply(column, function(value) {
      is.numeric(value) && length(value) == 351L
    }, logical(1L)))
  }, logical(1L))),
  "qualification artifact contains no residual-vector payload"
)

cat("PASS Phase 3C qualification dCov gate\n")
cat(
  "rows/near_alpha=", nrow(dcov), "/", sum(dcov$near_alpha), "\n",
  "max_abs/flips/near_flips/errors/fallbacks=",
  format(max_difference, digits = 17L), "/", decision_flip_count, "/",
  near_alpha_flip_count, "/", backend_error_count, "/",
  spectra_fallback_count, "\n",
  "logical_ids_hash=", logical_ids_hash, "\n",
  "rows_hash=", rows_hash, "\n", sep = ""
)
