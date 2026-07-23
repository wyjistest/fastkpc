source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
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
      (is.null(pattern) || grepl(pattern, conditionMessage(error), fixed = TRUE)),
    message
  )
  invisible(error)
}
assert_sha_columns <- function(value, fields, message) {
  assert_true(
    all(fields %in% names(value)) && all(vapply(fields, function(field) {
      column <- value[[field]]
      is.character(column) && !anyNA(column) &&
        all(grepl("^[0-9a-f]{64}$", column))
    }, logical(1L))),
    message
  )
}

hash_utf8 <- function(value) fastkpc_full_cuda_census_hash_utf8(value)

native_command <- function(role, executable_path, executable_label, argv) {
  list(
    role = role,
    executable_path = executable_path,
    executable_sha256 = hash_utf8(executable_label),
    argv = argv
  )
}

build_environment_names <- sort(c(
  "AR", "AS", "CC", "CFLAGS", "COMPILER_PATH", "CPATH", "CPPFLAGS",
  "CUDAFLAGS", "CUDAHOSTCXX", "CUDA_HOME", "CUDA_PATH", "CXX",
  "CXXFLAGS", "GCC_EXEC_PREFIX", "LANG", "LC_ALL", "LDFLAGS",
  "LD_LIBRARY_PATH", "LIBRARY_PATH", "MAKEFLAGS", "NVCCFLAGS",
  "NVCC_APPEND_FLAGS", "NVCC_CCBIN", "NVCC_PREPEND_FLAGS", "PATH",
  "PKG_CPPFLAGS", "PKG_CXXFLAGS", "PKG_LIBS", "R_ARCH", "R_HOME",
  "R_LIBS", "R_LIBS_SITE", "R_LIBS_USER", "R_MAKEVARS_SITE",
  "R_MAKEVARS_USER", "SHLIB_CXXLD", "SHLIB_CXXLDFLAGS",
  "SOURCE_DATE_EPOCH"
), method = "radix")
build_environment <- data.frame(
  name = build_environment_names,
  is_set = build_environment_names == "LC_ALL",
  value = ifelse(build_environment_names == "LC_ALL", "C", ""),
  stringsAsFactors = FALSE
)
build_commands <- list(
  native_command(
    "cxx_compile", "/usr/bin/c++", "c++ tool",
    c("<EXECUTABLE>", "-O3", "-c", "/worktree/input.cpp",
      "-o", "<OUTPUT>")
  ),
  native_command(
    "cuda_compile", "/usr/local/cuda/bin/nvcc", "nvcc tool",
    c("<EXECUTABLE>", "-O3", "-arch=sm_89", "-c",
      "/worktree/kernel.cu", "-o", "<OUTPUT>")
  ),
  native_command(
    "link", "/usr/bin/c++", "c++ tool",
    c("<EXECUTABLE>", "-shared", "-o", "<OUTPUT>",
      "/worktree/input.o", "/worktree/kernel.o")
  )
)

build_attestation_a <- list(
  schema_version = "full-cuda-ci-native-build-dependencies-v3",
  trace_semantics = "linux-strace-successful-read-exec-evidence-v3",
  trace_invocation = "strace invocation pid=17301",
  build_working_dir = "/worktree",
  trace_path = "/tmp/build-17301.strace",
  trace_sha256 = hash_utf8("raw trace pid=17301"),
  tracer_path = "/usr/bin/strace",
  tracer_sha256 = hash_utf8("strace tool"),
  dependency_count = 4L,
  files = data.frame(
    path = c(
      "/usr/bin/c++", "/usr/bin/strace", "/usr/local/cuda/bin/nvcc",
      "/worktree/input.cpp"
    ),
    sha256 = vapply(
      c(
        "c++ tool", "strace tool", "nvcc tool", "canonical build input"
      ),
      hash_utf8, character(1L)
    ),
    stringsAsFactors = FALSE
  ),
  exclusion_count = 1L,
  exclusions = data.frame(
    path = "/worktree/fastkpc/build/fastkpc_cuda.so.tmp.17301",
    reason = "generated_output", stringsAsFactors = FALSE
  ),
  command_projection_schema_version =
    "full-cuda-ci-native-build-command-projection-v1",
  command_count = as.integer(length(build_commands)),
  commands = build_commands,
  build_environment_schema_version =
    "full-cuda-ci-native-build-environment-v1",
  build_environment = build_environment
)
build_attestation_b <- build_attestation_a
build_attestation_b$trace_invocation <-
  "strace invocation pid=28412 with invocation-only noise"
build_attestation_b$trace_path <- "/tmp/build-28412.strace"
build_attestation_b$trace_sha256 <-
  hash_utf8("raw trace pid=28412 with invocation-only failed probe")
build_attestation_b$exclusions$path <-
  "/worktree/fastkpc/build/fastkpc_cuda.so.tmp.28412"
canonical_attestation_a <-
  fastkpc_full_cuda_fixed_sp_native_build_attestation_hash(
    build_attestation_a
  )
canonical_attestation_b <-
  fastkpc_full_cuda_fixed_sp_native_build_attestation_hash(
    build_attestation_b
  )
assert_identical(
  canonical_attestation_a, canonical_attestation_b,
  "canonical build attestation ignores raw PID/invocation trace noise"
)
for (semantic_index in seq_len(nrow(build_attestation_b$files))) {
  changed_attestation <- build_attestation_b
  changed_attestation$files$sha256[[semantic_index]] <- hash_utf8(
    paste("semantic dependency change", semantic_index)
  )
  changed_path <- changed_attestation$files$path[[semantic_index]]
  for (command_index in seq_along(changed_attestation$commands)) {
    if (identical(
          changed_attestation$commands[[command_index]]$executable_path,
          changed_path
        )) {
      changed_attestation$commands[[command_index]]$executable_sha256 <-
        changed_attestation$files$sha256[[semantic_index]]
    }
  }
  assert_true(
    !identical(
      canonical_attestation_a,
      fastkpc_full_cuda_fixed_sp_native_build_attestation_hash(
        changed_attestation
      )
    ),
    "canonical build attestation binds every tool/input dependency"
  )
}
command_mutations <- list(
  optimization = function(value) {
    value$commands[[1L]]$argv[[2L]] <- "-O0"
    value
  },
  architecture = function(value) {
    value$commands[[2L]]$argv[[3L]] <- "-arch=sm_75"
    value
  },
  definition = function(value) {
    value$commands[[1L]]$argv <- append(
      value$commands[[1L]]$argv, "-DFASTKPC_HOSTILE=1", after = 2L
    )
    value
  },
  command_order = function(value) {
    value$commands <- rev(value$commands)
    value
  },
  nvcc_environment = function(value) {
    row <- match("NVCC_PREPEND_FLAGS", value$build_environment$name)
    value$build_environment$is_set[[row]] <- TRUE
    value$build_environment$value[[row]] <- "--use_fast_math"
    value
  }
)
for (mutation_name in names(command_mutations)) {
  changed_attestation <- command_mutations[[mutation_name]](
    build_attestation_b
  )
  assert_true(
    !identical(
      canonical_attestation_a,
      fastkpc_full_cuda_fixed_sp_native_build_attestation_hash(
        changed_attestation
      )
    ),
    paste("canonical build attestation binds", mutation_name)
  )
}
for (missing_field in c(
  "command_projection_schema_version", "command_count", "commands",
  "build_environment_schema_version", "build_environment"
)) {
  incomplete_attestation <- build_attestation_a
  incomplete_attestation[[missing_field]] <- NULL
  assert_error(
    fastkpc_full_cuda_fixed_sp_native_build_attestation_hash(
      incomplete_attestation
    ),
    paste("incomplete native command projection rejects", missing_field)
  )
}
incomplete_environment <- build_attestation_a
incomplete_environment$build_environment <-
  incomplete_environment$build_environment[-1L, , drop = FALSE]
assert_error(
  fastkpc_full_cuda_fixed_sp_native_build_attestation_hash(
    incomplete_environment
  ),
  "incomplete native build environment projection fails closed"
)

execution_roots <- .fastkpc_full_cuda_phase3_execution_roots()
assert_identical(
  unname(execution_roots[["oracle_runner"]]),
  "fastkpc/tools/run_full_cuda_ci_fixed_sp_oracle_sp.R",
  "executable Phase 3 oracle runner is an authenticated source root"
)
execution_closure <-
  fastkpc_full_cuda_fixed_sp_discover_execution_source_closure(
    execution_roots, project_root = "."
  )
runner_id <- "fastkpc/tools/run_full_cuda_ci_fixed_sp_oracle_sp.R"
assert_true(
  runner_id %in% execution_closure$source_ids &&
    runner_id %in% unname(execution_closure$direct_source_ids),
  "executable Phase 3 oracle runner is in the authenticated closure"
)
closure_hashes <- vapply(
  execution_closure$source_file_paths,
  fastkpc_full_cuda_fixed_sp_sha256_file,
  character(1L)
)
dirty_runner_hashes <- closure_hashes
dirty_runner_hashes[[runner_id]] <- hash_utf8("dirty runner contents")
assert_true(
  !identical(
    fastkpc_full_cuda_fixed_sp_source_closure_hash(
      execution_closure, closure_hashes
    ),
    fastkpc_full_cuda_fixed_sp_source_closure_hash(
      execution_closure, dirty_runner_hashes
    )
  ),
  "dirty executable runner contents change authenticated source identity"
)

synthetic_root <- normalizePath(".", winslash = "/", mustWork = TRUE)
synthetic_cxx <- normalizePath(Sys.which("g++"), winslash = "/",
                               mustWork = TRUE)
synthetic_ld <- normalizePath(Sys.which("ld"), winslash = "/",
                              mustWork = TRUE)
synthetic_object <- file.path(synthetic_root, "fastkpc", "build", "input.o")
synthetic_output <- file.path(
  synthetic_root, "fastkpc", "build", "fastkpc_cuda.so.tmp.17301"
)
synthetic_exec_paths <- c(synthetic_cxx, synthetic_ld)
synthetic_exec_lines <- c(
  paste0(
    "17301 execve(\"", synthetic_cxx, "\", [\"g++\", \"-shared\", ",
    "\"-o\", \"", synthetic_output, "\", \"", synthetic_object,
    "\"], 0x0 /* 40 vars */) = 0"
  ),
  paste0(
    "17302 execve(\"", synthetic_ld, "\", [\"ld\", \"-shared\", ",
    "\"-plugin-opt=-fresolution=/tmp/ccPIDnoise.res\", \"-o\", ",
    "\"", synthetic_output, "\", \"", synthetic_object,
    "\"], 0x0 /* 40 vars */) = 0"
  )
)
synthetic_tools <- sort(unique(synthetic_exec_paths), method = "radix")
synthetic_tool_files <- data.frame(
  path = synthetic_tools,
  sha256 = vapply(
    synthetic_tools, fastkpc_full_cuda_fixed_sp_sha256_file, character(1L)
  ),
  stringsAsFactors = FALSE
)
synthetic_link_projection <-
  fastkpc_full_cuda_fixed_sp_native_build_commands(
    exec_lines = synthetic_exec_lines,
    exec_syscalls = rep("execve", 2L),
    exec_paths = synthetic_exec_paths,
    build_working_dir = synthetic_root,
    files = synthetic_tool_files
  )
assert_true(
  length(synthetic_link_projection) == 1L &&
    identical(synthetic_link_projection[[1L]]$role, "link") &&
    !any(grepl("ccPIDnoise", synthetic_link_projection[[1L]]$argv,
               fixed = TRUE)) &&
    identical(
      synthetic_link_projection[[1L]]$argv[
        match("-o", synthetic_link_projection[[1L]]$argv) + 1L
      ],
      "<OUTPUT>"
    ),
  "canonical command projection excludes nested linker invocation noise"
)

oracle_row_schemas <- fastkpc_full_cuda_fixed_sp_oracle_row_schemas()
for (schema_name in c("setup_results", "resource_metrics")) {
  assert_true(
    all(c(
      "phase2_shard_load_count",
      "phase2_shard_authentication_count"
    ) %in% names(oracle_row_schemas[[schema_name]])),
    paste(schema_name, "exposes authoritative Phase 2 shard load counters")
  )
}

stable_identity_fields <- .fastkpc_full_cuda_phase3_identity_fields()
stable_identity_fixture <- as.list(stats::setNames(
  rep("stable-fixture", length(stable_identity_fields)),
  stable_identity_fields
))
stable_identity_fixture$native_build_attestation_schema_version <-
  "full-cuda-ci-native-build-trace-attestation-v2"
stable_identity_fixture$native_build_attestation_sha256 <-
  canonical_attestation_a
trace_identity_a <- stable_identity_fixture
trace_identity_a$native_build_trace_invocation <-
  "strace invocation pid=17301"
trace_identity_a$native_build_trace_sha256 <- hash_utf8(paste(
  "17301 execve build tool", "17301 open canonical input", sep = "\n"
))
trace_identity_a$native_build_dependencies_sha256 <-
  hash_utf8("raw dependency envelope pid=17301")
trace_identity_a$native_library_sha256 <-
  hash_utf8("native binary with invocation build-id pid=17301")
trace_identity_b <- stable_identity_fixture
trace_identity_b$native_build_trace_invocation <-
  "strace invocation pid=28412 with invocation-only noise"
trace_identity_b$native_build_trace_sha256 <- hash_utf8(paste(
  "[pid 28412] execve build tool", "[pid 28412] open canonical input",
  "[pid 28412] invocation-only failed probe", sep = "\n"
))
trace_identity_b$native_build_dependencies_sha256 <-
  hash_utf8("raw dependency envelope pid=28412 with invocation-only noise")
trace_identity_b$native_library_sha256 <-
  hash_utf8("native binary with invocation build-id pid=28412")
trace_identity_b$native_build_attestation_sha256 <- canonical_attestation_b
assert_true(
  !identical(
    trace_identity_a$native_build_trace_sha256,
    trace_identity_b$native_build_trace_sha256
  ),
  "raw PID-bearing build traces remain distinct diagnostic provenance"
)
assert_true(
  !identical(
    trace_identity_a$native_library_sha256,
    trace_identity_b$native_library_sha256
  ),
  "invocation-specific native binary bytes remain diagnostic provenance"
)
assert_identical(
  .fastkpc_full_cuda_phase3_identity_hash(trace_identity_a),
  .fastkpc_full_cuda_phase3_identity_hash(trace_identity_b),
  paste(
    "stable identity ignores PID-bearing raw trace and linker build-id",
    "invocation noise"
  )
)
for (semantic_field in c(
  "native_build_inputs_sha256", "native_build_tracer_sha256",
  "native_build_attestation_sha256"
)) {
  changed_identity <- trace_identity_b
  changed_identity[[semantic_field]] <- hash_utf8(
    paste("semantically changed", semantic_field)
  )
  assert_true(
    !identical(
      .fastkpc_full_cuda_phase3_identity_hash(trace_identity_a),
      .fastkpc_full_cuda_phase3_identity_hash(changed_identity)
    ),
    paste("stable identity binds", semantic_field)
  )
}

executed_route_config <- fastkpc_full_cuda_phase3_route_config()
assert_identical(
  executed_route_config$svd_rank_tolerance,
  "sigma_max*sqrt(double_epsilon)",
  "route config binds the executed CUDA SVD rank threshold"
)
legacy_svd_route_config <- executed_route_config
legacy_svd_route_config$svd_rank_tolerance <-
  "max(augmented_rows,null_dim)*sigma_max*double_epsilon"
legacy_svd_route_config$sha256 <-
  .fastkpc_full_cuda_phase3_named_hash(
    legacy_svd_route_config[setdiff(names(legacy_svd_route_config), "sha256")]
  )
assert_true(
  !identical(
    executed_route_config$sha256, legacy_svd_route_config$sha256
  ),
  "route hash binds the executed CUDA SVD rank threshold"
)

qualification_callback_schema <-
  fastkpc_full_cuda_fixed_sp_qualification_dcov_schema()
qualification_callback_diagnostic <- list(
  n = 351L, numCol = 35L, index = 1, lowrank_mode = "spectra",
  lowrank_full_eig_count = 0L, lowrank_spectra_count = 2L,
  lowrank_spectra_converged_count = 2L,
  lowrank_spectra_failed_count = 0L,
  lowrank_spectra_fallback_full_eig_count = 0L,
  lowrank_spectra_iterations = 4L, lowrank_spectra_nconv = 70L,
  lowrank_spectra_ncv = 71L, lowrank_spectra_tol = 1e-10,
  lowrank_spectra_matvec_count = 0L
)
qualification_callback_columns <- lapply(
  qualification_callback_schema$names,
  function(field) {
    switch(
      qualification_callback_schema$types[[field]],
      character = "fixture", integer = 1L, double = 0,
      logical = FALSE, list = I(list(character())),
      fail("unsupported qualification callback fixture type")
    )
  }
)
names(qualification_callback_columns) <- qualification_callback_schema$names
qualification_callback_columns$parity_scope <- "qualification"
qualification_callback_columns$index <- 1L
qualification_callback_columns$numCol <- 35L
qualification_callback_columns$selection_reasons <-
  I(list(c("qualification", "near_alpha")))
qualification_callback_columns$diagnostics <-
  I(list(qualification_callback_diagnostic))
qualification_callback_nonempty <- structure(
  qualification_callback_columns,
  class = "data.frame", row.names = 1L
)
qualification_callback_empty <-
  .fastkpc_full_cuda_phase3_empty_full_qualification_dcov()
assert_true(
  fastkpc_full_cuda_fixed_sp_validate_qualification_dcov_frame(
    qualification_callback_nonempty
  ),
  "non-empty callback fixture passes the production qualification schema"
)
assert_true(
  identical(names(qualification_callback_empty),
            qualification_callback_schema$names) &&
    nrow(qualification_callback_empty) == 0L &&
    all(vapply(
      qualification_callback_schema$list_fields,
      function(field) {
        identical(
          attributes(qualification_callback_empty[[field]]),
          list(class = "AsIs")
        )
      }, logical(1L)
    )),
  "empty callback fixture has the production qualification schema"
)

compact_guard_assignments <- list()
collect_compact_guard <- function(node) {
  if (is.call(node) && length(node) >= 3L &&
      identical(node[[1L]], as.name("<-")) &&
      identical(node[[2L]], as.name("compact_callback_result"))) {
    compact_guard_assignments[[length(compact_guard_assignments) + 1L]] <<-
      node[[3L]]
  }
  if (is.call(node) || is.expression(node) || is.pairlist(node)) {
    lapply(as.list(node), collect_compact_guard)
  }
  invisible(NULL)
}
collect_compact_guard(body(
  fastkpc_full_cuda_fixed_sp_execute_oracle_setup
))
assert_true(
  length(compact_guard_assignments) == 1L,
  "oracle setup has exactly one callback compact-result guard"
)
callback_forbidden_fields <- c(
  "coefficients", "fitted", "residuals", "rss", "rhs",
  "cuda_nullspace_rhs"
)
callback_guard_accepts <- function(value) {
  eval(
    compact_guard_assignments[[1L]],
    envir = list2env(
      list(
        shadow_callback_result = value,
        forbidden_callback_fields = callback_forbidden_fields
      ),
      parent = environment(
        fastkpc_full_cuda_fixed_sp_execute_oracle_setup
      )
    )
  )
}
forbidden_matrix_frames <- lapply(callback_forbidden_fields, function(field) {
  structure(
    stats::setNames(list(I(matrix(1, nrow = 1L))), field),
    class = "data.frame", row.names = 1L
  )
})
forbidden_list_frames <- lapply(callback_forbidden_fields, function(field) {
  structure(
    stats::setNames(list(I(list(matrix(1, nrow = 1L)))), field),
    class = "data.frame", row.names = 1L
  )
})
alias_frames <- lapply(c("Re-Siduals", "Xty.null"), function(field) {
  value <- data.frame(injected = 1, stringsAsFactors = FALSE)
  names(value) <- field
  value
})
valid_guard_results <- vapply(
  list(qualification_callback_empty, qualification_callback_nonempty),
  callback_guard_accepts, logical(1L)
)
hostile_guard_results <- vapply(
  c(forbidden_matrix_frames, forbidden_list_frames, alias_frames),
  callback_guard_accepts, logical(1L)
)
assert_true(
  all(valid_guard_results) && !any(hostile_guard_results),
  paste0(
    "callback compact guard accepts exact empty/non-empty qualification ",
    "frames and rejects response injection; valid=",
    paste(valid_guard_results, collapse = ","), "; hostile=",
    paste(hostile_guard_results, collapse = ",")
  )
)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP Phase 3 fixed-sp oracle subset\n")
  quit(save = "no", status = 0L)
}

executor_name <- "fastkpc_full_cuda_phase3_run_oracle_shard"
executor <- get0(executor_name, mode = "function", inherits = TRUE)
if (is.null(executor)) fail(paste(executor_name, "is missing"))
summarize_oracle_rows <- get0(
  "fastkpc_full_cuda_phase3_summarize_oracle_rows",
  mode = "function", inherits = TRUE
)
if (is.null(summarize_oracle_rows)) {
  fail("fastkpc_full_cuda_phase3_summarize_oracle_rows is missing")
}

subset_started <- proc.time()[["elapsed"]]
qualified_native <-
  fastkpc_full_cuda_phase3_discover_qualified_native_evidence()
qualified_dependencies <-
  qualified_native$provenance$native_build_dependencies
qualified_roles <- vapply(
  qualified_dependencies$commands, `[[`, character(1L), "role"
)
assert_true(
  all(c("cxx_compile", "cuda_compile", "link") %in% qualified_roles) &&
    sum(qualified_roles == "link") == 1L &&
    qualified_dependencies$command_count == length(qualified_roles) &&
    !any(vapply(qualified_dependencies$commands, function(command) {
      any(grepl("/tmp/cc", command$argv, fixed = TRUE)) ||
        any(grepl("fastkpc_cuda.so.tmp.", command$argv, fixed = TRUE))
    }, logical(1L))) &&
    "NVCC_PREPEND_FLAGS" %in%
      qualified_dependencies$build_environment$name,
  "real qualified build binds canonical driver argv and relevant env"
)
assert_true(
  runner_id %in% names(qualified_native$provenance$source_file_paths) &&
    runner_id %in% unname(qualified_native$provenance$direct_source_ids),
  "real qualified provenance authenticates the executable oracle runner"
)
qualified_projection <- .fastkpc_full_cuda_phase3_execution_projection(
  qualified_native$provenance
)
assert_identical(
  qualified_projection$native_build_attestation_sha256,
  fastkpc_full_cuda_fixed_sp_native_build_attestation_hash(
    qualified_dependencies
  ),
  "real qualified projection binds the complete command attestation"
)
build_fastkpc_cuda_native(rebuild = FALSE)
assert_true(fastkpc_cuda_available(), "CUDA must be available")

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
iteration <- fastkpc_full_cuda_fixed_sp_scope(catalog, "iteration")
route_levels <- fastkpc_full_cuda_fixed_sp_contract()$route_levels
routes_by_setup <- split(
  iteration$target_rows$planned_route,
  iteration$target_rows$prepared_s_key_sha256
)
setup_keys <- sort(names(routes_by_setup), method = "radix")
pairs <- utils::combn(setup_keys, 2L, simplify = FALSE)
setup_catalog_rank <- match(
  setup_keys, catalog$setup_index$prepared_s_key_sha256
)
assert_true(!anyNA(setup_catalog_rank), "subset setup Phase 2 shard join")
setup_phase2_shard_id <- stats::setNames(
  as.integer(
    (setup_catalog_rank - 1L) %% catalog$catalog_contract$shard_count
  ),
  setup_keys
)
valid_pair <- vapply(pairs, function(pair) {
  routes <- lapply(pair, function(key) unique(routes_by_setup[[key]]))
  all(lengths(routes) < length(route_levels)) &&
    setequal(unique(unlist(routes, use.names = FALSE)), route_levels)
}, logical(1L))
shared_source_shard <- vapply(pairs, function(pair) {
  length(unique(setup_phase2_shard_id[pair])) == 1L
}, logical(1L))
pair_candidates <- pairs[valid_pair & shared_source_shard]
assert_true(
  length(pair_candidates) > 0L,
  "two-setup three-route fixture shares one Phase 2 source shard"
)
pair_target_counts <- vapply(pair_candidates, function(pair) {
  sum(iteration$target_rows$prepared_s_key_sha256 %in% pair)
}, integer(1L))
pair_labels <- vapply(pair_candidates, paste, character(1L), collapse = ":")
selected_pair <- pair_candidates[[order(
  pair_target_counts, pair_labels, method = "radix"
)[[1L]]]]
selected_pair <- sort(selected_pair, method = "radix")
assert_true(
  length(unique(setup_phase2_shard_id[selected_pair])) == 1L,
  "selected setup pair proves shared Phase 2 source-shard reuse"
)
selected_targets <- iteration$target_rows[
  iteration$target_rows$prepared_s_key_sha256 %in% selected_pair,
  , drop = FALSE
]
selected_targets <- selected_targets[order(
  match(selected_targets$prepared_s_key_sha256, selected_pair),
  selected_targets$residual_key_sha256,
  method = "radix"
), , drop = FALSE]
rownames(selected_targets) <- NULL
selected_targets <-
  .fastkpc_full_cuda_phase3_oracle_descriptor_target_rows(
    catalog, selected_targets
  )
subset_plan <- fastkpc_full_cuda_phase3_plan_shards(
  setup_keys = selected_pair,
  target_rows = selected_targets,
  scope = "iteration",
  shard_count = 1L
)
selected_targets <- subset_plan$target_rows
expected_target_count <- as.integer(nrow(selected_targets))
assert_true(
  length(selected_pair) == 2L && expected_target_count > 0L &&
    setequal(selected_targets$planned_route, route_levels) &&
    all(selected_targets$shard_id == 0L),
  "selected setup union covers Cholesky, QR, and SVD"
)

device_id <- suppressWarnings(as.integer(Sys.getenv(
  "FASTKPC_FULL_CUDA_PHASE3_DEVICE", unset = "0"
)))
assert_true(
  length(device_id) == 1L && !is.na(device_id) && device_id >= 0L,
  "test device id"
)
runtime_identity <- fastkpc_full_cuda_phase3_discover_runtime_evidence(
  list(), device_id
)
subset_identity <- list(
  schema_version = "full-cuda-ci-phase3-test-input-identity-v1",
  canonical_setup_corpus_hash = fastkpc_full_cuda_census_key_set_hash(
    selected_pair
  ),
  canonical_target_corpus_hash = fastkpc_full_cuda_census_key_set_hash(
    sort(selected_targets$residual_key_sha256, method = "radix")
  ),
  route_config_hash = executed_route_config$sha256,
  source_commit = fastkpc_full_cuda_source_commit(),
  cuda_toolkit_version = runtime_identity$cuda_toolkit_version,
  cuda_driver_version = runtime_identity$cuda_driver_version,
  gpu_name = runtime_identity$gpu_name,
  gpu_uuid = runtime_identity$gpu_uuid,
  compute_capability_major = runtime_identity$compute_capability_major,
  compute_capability_minor = runtime_identity$compute_capability_minor,
  compute_capability = runtime_identity$compute_capability,
  sm_count = runtime_identity$sm_count,
  device_id = runtime_identity$device_id,
  cusolver_deterministic_mode_required =
    runtime_identity$cusolver_deterministic_mode_required,
  cublas_math_mode_required = runtime_identity$cublas_math_mode_required,
  cublas_atomics_mode_required =
    runtime_identity$cublas_atomics_mode_required,
  cublas_user_workspace_required =
    runtime_identity$cublas_user_workspace_required,
  cublas_workspace_bytes_required =
    runtime_identity$cublas_workspace_bytes_required,
  cublas_workspace_min_alignment_required =
    runtime_identity$cublas_workspace_min_alignment_required
)
subset_identity$sha256 <- .fastkpc_full_cuda_phase3_named_hash(subset_identity)
subset_output_dir <- tempfile("fastkpc-phase3-oracle-subset-resume-")
on.exit(unlink(subset_output_dir, recursive = TRUE, force = TRUE), add = TRUE)
subset_lifecycle <- new.env(parent = emptyenv())
subset_lifecycle$runtime_create_count <- 0L
subset_lifecycle$runtime_destroy_count <- 0L
subset_lifecycle$executor_count <- 0L
capacity <- fastkpc_full_cuda_fixed_sp_contract()$canonical_capacities
callback_setup_key <- selected_pair[[1L]]
callback_target_rows <- selected_targets[
  selected_targets$prepared_s_key_sha256 == callback_setup_key,
  , drop = FALSE
]
rownames(callback_target_rows) <- NULL
callback_selected <- fastkpc_full_cuda_fixed_sp_oracle_setup_batch(
  catalog, callback_setup_key, callback_target_rows
)
callback_runtime <- fixed_sp_cuda_runtime_create(device_id)
callback_runtime_open <- TRUE
on.exit({
  if (isTRUE(callback_runtime_open)) {
    try(fixed_sp_cuda_runtime_free(callback_runtime), silent = TRUE)
  }
}, add = TRUE)
fixed_sp_cuda_runtime_reserve(
  callback_runtime, capacity$n, capacity$null_dim, capacity$target_count,
  capacity$penalty_count, capacity$augmented_rows
)
execute_callback_fixture <- function(shadow_callback = NULL) {
  fastkpc_full_cuda_fixed_sp_execute_oracle_setup(
    context = callback_runtime, catalog = catalog,
    setup_key = callback_setup_key, target_rows = callback_target_rows,
    shard_id = 0L, setup_ordinal = 1L, selected = callback_selected,
    phase2_shard_load_count = 1L,
    phase2_shard_authentication_count = 1L,
    phase2_shard_load_elapsed_ms = 0,
    shadow_callback = shadow_callback
  )
}
callback_lifecycle <- new.env(parent = emptyenv())
callback_lifecycle$normal_count <- 0L
callback_lifecycle$empty_count <- 0L
callback_lifecycle$throw_count <- 0L
callback_lifecycle$forbidden_count <- 0L
normal_callback_result <- execute_callback_fixture(function(
    setup_key, target_keys, residuals) {
  callback_lifecycle$normal_count <- callback_lifecycle$normal_count + 1L
  assert_true(
    identical(setup_key, callback_setup_key) &&
      identical(target_keys, callback_target_rows$residual_key_sha256) &&
      is.matrix(residuals) &&
      identical(
        dim(residuals),
        c(as.integer(capacity$n), as.integer(nrow(callback_target_rows)))
      ) && all(is.finite(residuals)),
    "real callback receives the synchronous explicit residual matrix"
  )
  qualification_callback_nonempty
})
assert_true(
  callback_lifecycle$normal_count == 1L &&
    identical(
      normal_callback_result$shadow_callback_result,
      qualification_callback_nonempty
    ) &&
    all(normal_callback_result$resource_metrics[
      c(
        "prepared_handle_destroy_count", "residual_token_release_count",
        "output_slot_release_count"
      )
    ] == 1L),
  paste(
    "normal callback returns validated qualification evidence and releases",
    "runtime resources"
  )
)
empty_callback_result <- execute_callback_fixture(function(
    setup_key, target_keys, residuals) {
  callback_lifecycle$empty_count <- callback_lifecycle$empty_count + 1L
  qualification_callback_empty
})
assert_true(
  callback_lifecycle$empty_count == 1L &&
    identical(
      empty_callback_result$shadow_callback_result,
      qualification_callback_empty
    ) &&
    all(empty_callback_result$resource_metrics[
      c(
        "prepared_handle_destroy_count", "residual_token_release_count",
        "output_slot_release_count"
      )
    ] == 1L),
  paste(
    "empty production qualification evidence is compact and releases",
    "runtime resources"
  )
)
assert_error(
  execute_callback_fixture(function(setup_key, target_keys, residuals) {
    callback_lifecycle$throw_count <- callback_lifecycle$throw_count + 1L
    stop("intentional callback failure", call. = FALSE)
  }),
  "throwing callback propagates its error",
  "intentional callback failure"
)
assert_error(
  execute_callback_fixture(function(setup_key, target_keys, residuals) {
    callback_lifecycle$forbidden_count <-
      callback_lifecycle$forbidden_count + 1L
    data.frame(residuals = I(list(residuals)))
  }),
  "callback cannot return residual matrices for shard payload injection",
  "not compact"
)
callback_recovery <- execute_callback_fixture()
assert_true(
  callback_lifecycle$throw_count == 1L &&
    callback_lifecycle$forbidden_count == 1L &&
    identical(
      names(callback_recovery),
      c("setup_results", "target_parity", "resource_metrics", "stage_timing")
    ) && all(callback_recovery$resource_metrics[
      c(
        "prepared_handle_destroy_count", "residual_token_release_count",
        "output_slot_release_count"
      )
    ] == 1L),
  "callback errors release token, slot, and handle before the next solve"
)
fixed_sp_cuda_runtime_free(callback_runtime)
callback_runtime_open <- FALSE

subset_runtime_create <- function() {
  subset_lifecycle$runtime_create_count <-
    subset_lifecycle$runtime_create_count + 1L
  runtime <- fixed_sp_cuda_runtime_create(device_id)
  keep <- FALSE
  on.exit({
    if (!keep) try(fixed_sp_cuda_runtime_free(runtime), silent = TRUE)
  }, add = TRUE)
  fixed_sp_cuda_runtime_reserve(
    runtime, capacity$n, capacity$null_dim, capacity$target_count,
    capacity$penalty_count, capacity$augmented_rows
  )
  subset_lifecycle$runtime_reserved <- fixed_sp_cuda_runtime_info(runtime)
  keep <- TRUE
  runtime
}
subset_runtime_destroy <- function(runtime) {
  subset_lifecycle$runtime_after <- fixed_sp_cuda_runtime_info(runtime)
  fixed_sp_cuda_runtime_free(runtime)
  subset_lifecycle$runtime_destroy_count <-
    subset_lifecycle$runtime_destroy_count + 1L
  invisible(NULL)
}
subset_executor <- function(context, shard_id, setup_keys, target_rows) {
  subset_lifecycle$executor_count <- subset_lifecycle$executor_count + 1L
  value <- executor(
    context = context, shard_id = shard_id, setup_keys = setup_keys,
    target_rows = target_rows, catalog = catalog
  )
  subset_lifecycle$executor_result <- value
  value
}
first_run <- fastkpc_full_cuda_phase3_run_shards(
  output_dir = subset_output_dir, kind = "oracle_sp",
  setup_keys = selected_pair, target_rows = selected_targets,
  identity = subset_identity, route_config = executed_route_config,
  executor = subset_executor, runtime_create = subset_runtime_create,
  runtime_destroy = subset_runtime_destroy, scope = "iteration",
  shard_count = 1L
)
assert_true(
  identical(first_run$status, "complete") &&
    identical(first_run$written_shard_ids, 0L) &&
    identical(first_run$reused_shard_ids, integer()) &&
    subset_lifecycle$runtime_create_count == 1L &&
    subset_lifecycle$runtime_destroy_count == 1L &&
    subset_lifecycle$executor_count == 1L,
  "first real subset invocation writes one shard in one closed session"
)
result <- subset_lifecycle$executor_result
runtime_reserved <- subset_lifecycle$runtime_reserved
runtime_after <- subset_lifecycle$runtime_after
merged <- fastkpc_full_cuda_phase3_merge_shards(
  output_dir = subset_output_dir, kind = "oracle_sp",
  setup_keys = selected_pair, target_rows = selected_targets,
  identity = subset_identity, route_config = executed_route_config,
  scope = "iteration", shard_count = 1L
)

assert_identical(
  names(result), c("payload", "resource_counts"),
  "oracle shard executor result schema"
)
payload <- merged$payload
assert_identical(payload, result$payload, "merged payload equals the real shard")
assert_identical(
  names(payload),
  c(
    "setup_results", "target_parity", "resource_metrics",
    "stage_timing", "fallbacks", "failures", "summary"
  ),
  "oracle shard payload schema"
)
assert_true(nrow(payload$setup_results) == 2L, "two setup rows")
assert_true(
  nrow(payload$target_parity) == expected_target_count,
  "all selected target rows"
)
assert_identical(
  payload$setup_results$prepared_s_key_sha256, selected_pair,
  "setup rows retain shard/radix order"
)
assert_identical(
  payload$target_parity$residual_key_sha256,
  selected_targets$residual_key_sha256,
  "target rows retain canonical setup/target order"
)
assert_true(all(payload$target_parity$solver_status %in% c(
  "OK_CHOLESKY_BATCHED", "OK_CHOLESKY_SINGLE",
  "OK_AUGMENTED_QR", "OK_AUGMENTED_SVD"
)), "all subset targets OK")
assert_true(
  setequal(payload$target_parity$planned_route, route_levels) &&
    setequal(payload$target_parity$executed_route, route_levels),
  "subset planned/executed route coverage"
)
assert_true(
  all(is.finite(payload$target_parity$coefficient_max_abs_diff)) &&
    all(is.finite(payload$target_parity$coefficient_relative_l2)),
  "subset coefficient comparison evidence"
)
assert_true(
  max(payload$target_parity$residual_max_abs_diff) < 1e-7,
  "subset residual max-abs parity"
)
assert_true(
  max(payload$target_parity$residual_relative_l2) < 1e-7,
  "subset residual relative-L2 parity"
)
assert_true(
  max(payload$target_parity$fitted_max_abs_diff) < 1e-7,
  "subset fitted max-abs parity"
)
assert_true(
  max(payload$target_parity$fitted_relative_l2) < 1e-7,
  "subset fitted relative-L2 parity"
)
assert_true(
  all(is.finite(payload$target_parity$rss_max_abs_diff)) &&
    all(is.finite(payload$target_parity$rss_relative_l2)),
  "subset RSS comparison evidence"
)
assert_true(
  max(payload$target_parity$rhs_max_abs_diff) < 1e-12 &&
    max(payload$target_parity$rhs_relative_l2) < 1e-12,
  "subset CUDA-built RHS parity"
)
assert_true(
  all(payload$target_parity$output_all_finite),
  "subset finite outputs"
)
assert_sha_columns(
  payload$target_parity,
  c(
    "coefficient_candidate_sha256", "coefficient_oracle_sha256",
    "fitted_candidate_sha256", "fitted_oracle_sha256",
    "residual_candidate_sha256", "residual_oracle_sha256",
    "rss_candidate_sha256", "rss_oracle_sha256",
    "rhs_candidate_sha256", "rhs_oracle_sha256"
  ),
  "compact candidate/oracle hashes"
)
assert_true(
  all(c(
    "residual_key_sha256", "prepared_s_key_sha256", "target",
    "canonical_target_rank", "condition", "condition_bucket",
    "phase1_coefficient_rank", "planned_route", "executed_route",
    "reroute_reason", "solver_status", "target_true_batched",
    "cholesky_to_svd_count", "qr_to_svd_count", "effective_rank",
    "numeric_reference", "fallback_type", "error_message_sha256"
  ) %in% names(payload$target_parity)),
  "target identity, route, rank, fallback, and error evidence"
)
assert_true(
  all(payload$target_parity$numeric_reference == "mgcv-fixed-sp") &&
    all(payload$target_parity$fallback_type == "NONE"),
  "C_magic authority with no fallback"
)
assert_true(sum(payload$fallbacks$count) == 0L, "subset no fallback")
assert_true(nrow(payload$failures) == 0L, "subset no failure rows")
assert_true(
  !any(vapply(payload, function(component) {
    is.data.frame(component) && any(c(
      "coefficients", "fitted", "residuals", "rss"
    ) %in% names(component))
  }, logical(1L))),
  "shard payload stores no fitted/residual/output vectors"
)

resources <- payload$resource_metrics
assert_true(nrow(resources) == 2L, "one resource row per setup")
unique_phase2_shard_count <- as.integer(length(unique(
  payload$setup_results$phase2_shard_id
)))
assert_true(
  sum(payload$setup_results$phase2_shard_load_count) ==
    unique_phase2_shard_count &&
    sum(payload$setup_results$phase2_shard_authentication_count) ==
      unique_phase2_shard_count &&
    sum(resources$phase2_shard_load_count) == unique_phase2_shard_count &&
    sum(resources$phase2_shard_authentication_count) ==
      unique_phase2_shard_count &&
    payload$summary$unique_phase2_shard_count ==
      unique_phase2_shard_count &&
    payload$summary$phase2_shard_load_count == unique_phase2_shard_count &&
    payload$summary$phase2_shard_authentication_count ==
      unique_phase2_shard_count &&
    unique_phase2_shard_count < nrow(payload$setup_results),
  "subset authenticates and loads each unique Phase 2 shard exactly once"
)
assert_true(
  all(resources$prepared_handle_create_count == 1L) &&
    all(resources$prepared_handle_destroy_count == 1L) &&
    all(resources$residual_token_acquire_count == 1L) &&
    all(resources$residual_token_release_count == 1L) &&
    all(resources$output_slot_acquire_count == 1L) &&
    all(resources$output_slot_release_count == 1L) &&
    all(!resources$output_slot_leased_after_release),
  "every setup releases its token, lease, and prepared handle"
)
assert_true(
  all(resources$per_target_allocation_count_after_warmup == 0L) &&
    all(resources$per_target_handle_create_count == 0L) &&
    all(resources$implicit_residual_d2h_count == 0L) &&
    all(resources$cpu_fallback_count == 0L) &&
    all(resources$unknown_fallback_count == 0L) &&
    all(resources$approximate_backend_count == 0L),
  "subset resource and fallback gates"
)
assert_true(
  all(resources$rhs_authority == "cuda-x0-transpose-y") &&
    all(resources$full_cuda_data_plane),
  "GPU-computed RHS remains authoritative"
)
assert_true(
  runtime_after$workspace_grow_count == runtime_reserved$workspace_grow_count &&
    runtime_after$stable_workspace_grow_count ==
      runtime_reserved$stable_workspace_grow_count &&
    runtime_after$cublas_math_mode == "pedantic" &&
    runtime_after$cublas_atomics_mode == "not_allowed" &&
    runtime_after$cusolver_deterministic_mode == "enabled",
  "one reserved deterministic runtime serves both setups without growth"
)

recomputed <- summarize_oracle_rows(
  setup_results = payload$setup_results,
  target_parity = payload$target_parity,
  resource_metrics = payload$resource_metrics,
  stage_timing = payload$stage_timing,
  fallbacks = payload$fallbacks,
  failures = payload$failures
)
assert_identical(payload$summary, recomputed, "summary is recomputed from rows")
mutated_payload <- payload
qr_setup <- which(mutated_payload$setup_results$planned_qr_target_count > 0L)[[1L]]
mutated_payload$resource_metrics$qr_checkpoint_wait_count[[qr_setup]] <-
  as.integer(
    mutated_payload$resource_metrics$qr_checkpoint_wait_count[[qr_setup]] +
      mutated_payload$resource_metrics$target_count[[qr_setup]]
  )
validate_oracle_payload <- get0(
  ".fastkpc_full_cuda_phase3_validate_oracle_payload",
  mode = "function", inherits = TRUE
)
assert_true(!is.null(validate_oracle_payload), "oracle shard validator is missing")
assert_error(
  {
    mutated_payload$summary <- summarize_oracle_rows(
      setup_results = mutated_payload$setup_results,
      target_parity = mutated_payload$target_parity,
      resource_metrics = mutated_payload$resource_metrics,
      stage_timing = mutated_payload$stage_timing,
      fallbacks = mutated_payload$fallbacks,
      failures = mutated_payload$failures
    )
    validate_oracle_payload(mutated_payload)
  },
  "target-level stable-sync mutation must fail shard validation",
  "target-level stable sync"
)
assert_true(
  nrow(recomputed) == 1L && recomputed$setup_count == 2L &&
    recomputed$target_count == expected_target_count &&
    recomputed$target_level_stable_sync_count == 0L &&
    recomputed$cuda_device_synchronize_count == 0L &&
    recomputed$non_ok_solver_status_count == 0L &&
    recomputed$cholesky_to_svd_count == 0L &&
    recomputed$qr_to_svd_count == 0L &&
    recomputed$cpu_fallback_count == 0L &&
    recomputed$unknown_fallback_count == 0L &&
    recomputed$approximate_backend_count == 0L &&
    !"pass" %in% names(recomputed),
  "row-derived subset summary"
)
assert_identical(
  unname(as.integer(result$resource_counts[c(
    "prepared_handle_create_count", "prepared_handle_destroy_count"
  )])),
  c(2L, 2L),
  "executor setup resource totals"
)
assert_identical(
  unname(as.integer(result$resource_counts[c(
    "residual_token_acquire_count", "residual_token_release_count",
    "output_slot_acquire_count", "output_slot_release_count"
  )])),
  rep(nrow(payload$setup_results), 4L),
  "executor lease accounting sums one physical batch row per setup"
)

session_files <- list.files(
  file.path(subset_output_dir, "sessions"),
  pattern = "^session_[A-Za-z0-9_-]+\\.json$", full.names = TRUE
)
assert_true(length(session_files) == 1L, "real subset writes one session row")
subset_session <- jsonlite::read_json(session_files[[1L]], simplifyVector = TRUE)
assert_true(
  as.integer(subset_session$prepared_handle_create_count) == 2L &&
    as.integer(subset_session$prepared_handle_destroy_count) == 2L &&
    as.integer(subset_session$residual_token_acquire_count) == 2L &&
    as.integer(subset_session$residual_token_release_count) == 2L &&
    as.integer(subset_session$output_slot_acquire_count) == 2L &&
    as.integer(subset_session$output_slot_release_count) == 2L &&
    as.integer(subset_session$target_level_stable_sync_count) == 0L,
  "session accounting conserves one token and lease per setup batch"
)

pure_resume_calls <- new.env(parent = emptyenv())
pure_resume_calls$runtime <- 0L
pure_resume_calls$executor <- 0L
pure_resume_calls$destroy <- 0L
unexpected_runtime_create <- function() {
  pure_resume_calls$runtime <- pure_resume_calls$runtime + 1L
  fail("pure resume created a runtime")
}
unexpected_executor <- function(...) {
  pure_resume_calls$executor <- pure_resume_calls$executor + 1L
  fail("pure resume executed setup work")
}
unexpected_runtime_destroy <- function(...) {
  pure_resume_calls$destroy <- pure_resume_calls$destroy + 1L
  fail("pure resume destroyed a runtime that should not exist")
}
pure_resume <- fastkpc_full_cuda_phase3_run_shards(
  output_dir = subset_output_dir, kind = "oracle_sp",
  setup_keys = selected_pair, target_rows = selected_targets,
  identity = subset_identity, route_config = executed_route_config,
  executor = unexpected_executor, runtime_create = unexpected_runtime_create,
  runtime_destroy = unexpected_runtime_destroy, scope = "iteration",
  shard_count = 1L
)
assert_true(
  identical(pure_resume$status, "complete") &&
    identical(pure_resume$reused_shard_ids, 0L) &&
    identical(pure_resume$written_shard_ids, integer()) &&
    pure_resume$runtime_context_create_count == 0L &&
    pure_resume$runtime_context_destroy_count == 0L &&
    pure_resume_calls$runtime == 0L && pure_resume_calls$executor == 0L &&
    pure_resume_calls$destroy == 0L && length(list.files(
      file.path(subset_output_dir, "sessions"),
      pattern = "^session_[A-Za-z0-9_-]+\\.json$"
    )) == 1L,
  "second real subset invocation is a pure shard resume"
)

cat(
  "PASS Phase 3 fixed-sp oracle subset:",
  nrow(payload$setup_results), "setups /",
  nrow(payload$target_parity), "targets /",
  payload$summary$phase2_shard_load_count, "unique Phase 2 loads /",
  payload$summary$phase2_shard_authentication_count,
  "unique Phase 2 authentications /",
  format(proc.time()[["elapsed"]] - subset_started, digits = 8L),
  "seconds\n"
)
