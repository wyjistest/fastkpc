source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/cuda_native.R")

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
assert_error_identical <- function(expression, expected, message) {
  error <- tryCatch({
    force(expression)
    NULL
  }, error = identity)
  assert_true(
    inherits(error, "error") && identical(conditionMessage(error), expected),
    message
  )
}

fixture_dir <- tempfile(
  "fastkpc_native_build_provenance_", tmpdir = tempdir()
)
dir.create(fixture_dir)
on.exit(unlink(fixture_dir, recursive = TRUE, force = TRUE), add = TRUE)

maps_deleted_path <- file.path(fixture_dir, "mapped>fixture.so")
maps_encoded_path <- paste0(
  dirname(maps_deleted_path), "/mapped\\", "076fixture.so"
)
maps_fixture_path <- file.path(fixture_dir, "proc-self-maps")
writeLines(c(
  paste0(
    "7f0000000000-7f0000001000 r--p 00000000 08:01 42 ",
    maps_encoded_path, " (deleted)"
  ),
  "7f0000001000-7f0000002000 rw-p 00000000 00:00 0 [heap]",
  "7f0000002000-7f0000003000 r-xp 00000000 08:01 43 /tmp/plain.so"
), maps_fixture_path, useBytes = TRUE)
assert_identical(
  .fastkpc_cuda_mapped_object_paths(maps_path = maps_fixture_path),
  sort(c(maps_deleted_path, "/tmp/plain.so"), method = "radix"),
  "Linux mapped-object snapshot decodes and retains deleted file mappings"
)

compiler_path <- file.path(fixture_dir, "fixture-compiler")
header_path <- file.path(fixture_dir, "fixture header.hpp")
escaped_header_path <- file.path(fixture_dir, "fixture>returned.hpp")
output_path <- file.path(fixture_dir, "fixture-output.o")
dirfd_include_dir <- file.path(fixture_dir, "pkg", "include")
dir.create(dirfd_include_dir, recursive = TRUE)
dirfd_header_path <- file.path(dirfd_include_dir, "header.h")
writeLines("#!/bin/sh", compiler_path, useBytes = TRUE)
writeLines("// fixture header", header_path, useBytes = TRUE)
writeLines("// escaped fixture header", escaped_header_path, useBytes = TRUE)
writeLines("fixture output", output_path, useBytes = TRUE)
writeLines("// dirfd fixture header", dirfd_header_path, useBytes = TRUE)
compiler_path <- normalizePath(compiler_path, winslash = "/", mustWork = TRUE)
header_path <- normalizePath(header_path, winslash = "/", mustWork = TRUE)
escaped_header_path <- normalizePath(
  escaped_header_path, winslash = "/", mustWork = TRUE
)
output_path <- normalizePath(output_path, winslash = "/", mustWork = TRUE)
dirfd_include_dir <- normalizePath(
  dirfd_include_dir, winslash = "/", mustWork = TRUE
)
dirfd_header_path <- normalizePath(
  dirfd_header_path, winslash = "/", mustWork = TRUE
)
trace_path <- file.path(fixture_dir, "build.strace")
pseudo_missing_path <- "/proc/999999999/fastkpc-missing"
generated_missing_path <- file.path(fixture_dir, "generated-missing.o")
trace_lines <- c(
  paste0(
    "100 execve(\"", compiler_path,
    "\", [\"fixture-compiler\"], 0x0 /* 0 vars */) = 0"
  ),
  paste0(
    "100 openat(AT_FDCWD, \"", header_path,
    "\", O_RDONLY|O_CLOEXEC) = 3<", header_path, ">"
  ),
  paste0(
    "100 openat(AT_FDCWD, \"", header_path,
    "\", O_RDONLY|O_CLOEXEC) = 4<", header_path, ">"
  ),
  paste0(
    "100 openat(AT_FDCWD, \"", output_path,
    "\", O_WRONLY|O_CREAT|O_TRUNC, 0666) = 5<", output_path, ">"
  ),
  paste0(
    "100 openat(AT_FDCWD, \"", output_path,
    "\", O_RDONLY|O_CLOEXEC) = 6<", output_path, ">"
  ),
  paste0(
    "100 openat(AT_FDCWD, \"", generated_missing_path,
    "\", O_WRONLY|O_CREAT|O_TRUNC, 0666) = 12<",
    generated_missing_path, ">"
  ),
  paste0(
    "100 openat(AT_FDCWD, \"", generated_missing_path,
    "\", O_RDONLY|O_CLOEXEC) = 13<", generated_missing_path, ">"
  ),
  paste0(
    "100 openat(5<", dirfd_include_dir,
    ">, \"header.h\", O_RDONLY|O_CLOEXEC) = 7<",
    dirfd_header_path, ">"
  ),
  paste0(
    "100 openat(AT_FDCWD, \"", escaped_header_path,
    "\", O_RDONLY|O_CLOEXEC) = 9<", dirname(escaped_header_path),
    "/fixture\\", "76returned.hpp>"
  ),
  "100 openat(AT_FDCWD, \"/dev/null\", O_RDONLY|O_CLOEXEC) = 8</dev/null>",
  paste0(
    "100 openat(AT_FDCWD, \"/proc/filesystems\", ",
    "O_RDONLY|O_CLOEXEC) = 10</proc/filesystems>"
  ),
  paste0(
    "100 openat(AT_FDCWD, \"", pseudo_missing_path,
    "\", O_RDONLY|O_CLOEXEC) = 11<", pseudo_missing_path, ">"
  ),
  paste0(
    "100 openat(AT_FDCWD, \"", fixture_dir,
    "/missing.hpp\", O_RDONLY|O_CLOEXEC) = -1 ENOENT"
  )
)
writeLines(trace_lines, trace_path, useBytes = TRUE)

strace_path <- Sys.which("strace")
assert_true(nzchar(strace_path), "strace is required for provenance fixtures")
strace_path <- normalizePath(strace_path, winslash = "/", mustWork = TRUE)
assert_error_matching(
  .fastkpc_cuda_resolve_strace(character()),
  "strace is required",
  "qualification build tracing rejects a missing strace executable"
)
trace_invocation <- paste(
  "LC_ALL=C", strace_path, "-f -qq -yy -s 65535",
  "-e trace=%file",
  "-e status=successful -o <trace> bash <build-script>"
)
dependencies <-
  fastkpc_full_cuda_fixed_sp_capture_native_build_dependencies(
    trace_path = trace_path,
    build_working_dir = ".",
    tracer_path = strace_path,
    trace_invocation = trace_invocation
  )
expected_paths <- sort(
  unique(c(
    compiler_path, dirfd_header_path, escaped_header_path, header_path,
    strace_path
  )),
  method = "radix"
)
assert_identical(
  dependencies$files$path, expected_paths,
  "native build trace retains successful read/exec regular files only"
)
assert_true(
  identical(names(dependencies$files), c("path", "sha256")) &&
    nrow(dependencies$files) == 5L &&
    all(grepl("^[0-9a-f]{64}$", dependencies$files$sha256)) &&
    identical(
      dependencies$schema_version,
      "full-cuda-ci-native-build-dependencies-v2"
    ) && identical(
      dependencies$trace_semantics,
      "linux-strace-successful-read-exec-evidence-v2"
    ) && identical(dependencies$trace_path, normalizePath(
      trace_path, winslash = "/", mustWork = TRUE
    )) && identical(
      dependencies$trace_sha256,
      fastkpc_full_cuda_fixed_sp_sha256_file(trace_path)
    ) &&
    identical(dependencies$tracer_path, strace_path) &&
    identical(dependencies$trace_invocation, trace_invocation) &&
    grepl("^[0-9a-f]{64}$", dependencies$tracer_sha256) &&
    grepl("^[0-9a-f]{64}$", dependencies$aggregate_sha256),
  "native build dependency closure has strict inspectable identity"
)
expected_exclusions <- data.frame(
  path = sort(c(
    "/dev/null", "/proc/filesystems", generated_missing_path,
    pseudo_missing_path, output_path
  ), method = "radix"),
  reason = character(5L),
  stringsAsFactors = FALSE
)
expected_exclusions$reason <- c(
  "pseudo_fs", "pseudo_fs", "generated_output", "pseudo_fs",
  "generated_output"
)[match(
  expected_exclusions$path,
  c(
    "/dev/null", "/proc/filesystems", generated_missing_path,
    pseudo_missing_path, output_path
  )
)]
rownames(expected_exclusions) <- as.character(seq_len(nrow(expected_exclusions)))
assert_true(
  identical(dependencies$exclusion_count, 5L) &&
    identical(dependencies$exclusions, expected_exclusions),
  "native build dependency capture records every evidence-backed exclusion"
)
trace_hash_mutation <- dependencies
trace_hash_mutation$trace_sha256 <- strrep("a", 64L)
assert_true(
  !identical(
    fastkpc_full_cuda_fixed_sp_native_build_dependency_hash(
      trace_hash_mutation
    ),
    dependencies$aggregate_sha256
  ),
  "native build dependency aggregate binds the exact raw trace hash"
)
exclusion_hash_mutation <- dependencies
exclusion_hash_mutation$exclusions$reason[[1L]] <- "non_regular"
assert_true(
  !identical(
    fastkpc_full_cuda_fixed_sp_native_build_dependency_hash(
      exclusion_hash_mutation
    ),
    dependencies$aggregate_sha256
  ),
  "native build dependency aggregate binds every exclusion reason"
)
assert_true(
  isTRUE(fastkpc_full_cuda_fixed_sp_validate_native_build_dependencies(
    dependencies
  )),
  "native build dependency closure validates"
)

capture_ordered_fixture <- function(name, fixture_lines) {
  fixture_trace_path <- file.path(fixture_dir, paste0(name, ".strace"))
  baseline_read <- paste0(
    "100 openat(AT_FDCWD, \"", header_path,
    "\", O_RDONLY|O_CLOEXEC) = 20<", header_path, ">"
  )
  writeLines(c(baseline_read, fixture_lines), fixture_trace_path,
             useBytes = TRUE)
  fastkpc_full_cuda_fixed_sp_capture_native_build_dependencies(
    trace_path = fixture_trace_path,
    build_working_dir = ".",
    tracer_path = strace_path,
    trace_invocation = trace_invocation
  )
}

late_truncate_path <- file.path(fixture_dir, "read-before-truncate.hpp")
writeLines("// read before truncate", late_truncate_path, useBytes = TRUE)
late_truncate_path <- normalizePath(
  late_truncate_path, winslash = "/", mustWork = TRUE
)
late_truncate <- capture_ordered_fixture("read-before-truncate", c(
  paste0(
    "100 openat(AT_FDCWD, \"", late_truncate_path,
    "\", O_RDONLY|O_CLOEXEC) = 21<", late_truncate_path, ">"
  ),
  paste0(
    "100 openat(AT_FDCWD, \"", late_truncate_path,
    "\", O_WRONLY|O_TRUNC) = 22<", late_truncate_path, ">"
  )
))
late_truncate_is_dependency <-
  late_truncate_path %in% late_truncate$files$path &&
  !late_truncate_path %in% late_truncate$exclusions$path

late_deleted_path <- file.path(fixture_dir, "deleted-read-before-truncate.hpp")
late_deleted_error <- tryCatch({
  capture_ordered_fixture("deleted-read-before-truncate", c(
    paste0(
      "100 openat(AT_FDCWD, \"", late_deleted_path,
      "\", O_RDONLY|O_CLOEXEC) = 23<", late_deleted_path, ">"
    ),
    paste0(
      "100 openat(AT_FDCWD, \"", late_deleted_path,
      "\", O_WRONLY|O_TRUNC) = 24<", late_deleted_path, ">"
    )
  ))
  NULL
}, error = identity)
late_deleted_fails_closed <- inherits(late_deleted_error, "error") &&
  identical(
    conditionMessage(late_deleted_error),
    paste0(
      "native build successful read/exec path disappeared before hash ",
      "capture: ", late_deleted_path
    )
  )

rdwr_input_path <- file.path(fixture_dir, "rdwr-input.hpp")
writeLines("// O_RDWR input", rdwr_input_path, useBytes = TRUE)
rdwr_input_path <- normalizePath(
  rdwr_input_path, winslash = "/", mustWork = TRUE
)
rdwr_input <- capture_ordered_fixture("rdwr-input", paste0(
  "100 openat(AT_FDCWD, \"", rdwr_input_path,
  "\", O_RDWR|O_CLOEXEC) = 25<", rdwr_input_path, ">"
))
rdwr_is_dependency <- rdwr_input_path %in% rdwr_input$files$path &&
  !rdwr_input_path %in% rdwr_input$exclusions$path

prior_truncate_path <- file.path(fixture_dir, "truncate-before-read.o")
writeLines("fixture generated bytes", prior_truncate_path, useBytes = TRUE)
prior_truncate_path <- normalizePath(
  prior_truncate_path, winslash = "/", mustWork = TRUE
)
prior_truncate <- capture_ordered_fixture("truncate-before-read", c(
  paste0(
    "100 openat(AT_FDCWD, \"", prior_truncate_path,
    "\", O_WRONLY|O_TRUNC) = 26<", prior_truncate_path, ">"
  ),
  paste0(
    "100 openat(AT_FDCWD, \"", prior_truncate_path,
    "\", O_RDONLY|O_CLOEXEC) = 27<", prior_truncate_path, ">"
  )
))
prior_truncate_is_generated <- identical(
  prior_truncate$exclusions[
    prior_truncate$exclusions$path == prior_truncate_path, , drop = FALSE
  ],
  data.frame(
    path = prior_truncate_path,
    reason = "generated_output",
    stringsAsFactors = FALSE,
    row.names = "1"
  )
) && !prior_truncate_path %in% prior_truncate$files$path

ordered_generation_results <- c(
  late_truncate_is_dependency = late_truncate_is_dependency,
  late_deleted_fails_closed = late_deleted_fails_closed,
  rdwr_is_dependency = rdwr_is_dependency,
  prior_truncate_is_generated = prior_truncate_is_generated
)
assert_true(
  all(ordered_generation_results),
  paste0(
    "native build generation requires prior unambiguous evidence; ",
    paste(
      names(ordered_generation_results), ordered_generation_results,
      sep = "=", collapse = ","
    )
  )
)

actual_build_script <- file.path(fixture_dir, "actual-build.sh")
actual_trace_path <- file.path(fixture_dir, "actual-build.strace")
writeLines(c(
  "#!/bin/sh",
  "cksum \"$1\" >/dev/null"
), actual_build_script, useBytes = TRUE)
actual_trace_status <- system2(
  strace_path,
  c(
    "-f", "-qq", "-yy", "-s", "65535",
    "-e", "trace=%file",
    "-e", "status=successful", "-o", shQuote(actual_trace_path),
    "bash", shQuote(actual_build_script), shQuote(header_path)
  ),
  stdout = FALSE, stderr = FALSE, env = "LC_ALL=C"
)
assert_identical(
  actual_trace_status, 0L,
  "actual no-CUDA strace build fixture succeeds"
)
actual_dependencies <-
  fastkpc_full_cuda_fixed_sp_capture_native_build_dependencies(
    trace_path = actual_trace_path,
    build_working_dir = ".",
    tracer_path = strace_path,
    trace_invocation = trace_invocation
  )
assert_true(
  all(c(header_path, strace_path) %in% actual_dependencies$files$path),
  "actual strace output captures the read header and tracer bytes"
)
assert_true(
  identical(
    actual_dependencies$trace_sha256,
    fastkpc_full_cuda_fixed_sp_sha256_file(actual_trace_path)
  ) && identical(
    names(actual_dependencies$exclusions), c("path", "reason")
  ),
  "actual no-CUDA trace retains raw-byte and exclusion provenance"
)

duplicate_path_dependencies <- dependencies
duplicate_path_dependencies$files <- rbind(
  duplicate_path_dependencies$files,
  transform(
    duplicate_path_dependencies$files[1L, , drop = FALSE],
    sha256 = strrep("f", 64L)
  )
)
duplicate_path_dependencies$dependency_count <-
  as.integer(nrow(duplicate_path_dependencies$files))
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_validate_native_build_dependencies(
    duplicate_path_dependencies
  ),
  "native build dependency identity is malformed",
  "native build dependency identity rejects path/hash collisions"
)

empty_trace_path <- file.path(fixture_dir, "empty.strace")
writeLines(character(), empty_trace_path, useBytes = TRUE)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_capture_native_build_dependencies(
    trace_path = empty_trace_path,
    build_working_dir = ".",
    tracer_path = strace_path,
    trace_invocation = trace_invocation
  ),
  "native build trace is empty",
  "native build dependency capture rejects empty traces"
)
malformed_trace_path <- file.path(fixture_dir, "malformed.strace")
writeLines(
  "100 openat(AT_FDCWD, unterminated, O_RDONLY) = 3",
  malformed_trace_path, useBytes = TRUE
)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_capture_native_build_dependencies(
    trace_path = malformed_trace_path,
    build_working_dir = ".",
    tracer_path = strace_path,
    trace_invocation = trace_invocation
  ),
  "native build trace is malformed",
  "native build dependency capture rejects malformed traced syscalls"
)
unresolved_relative_trace_path <- file.path(
  fixture_dir, "unresolved-relative.strace"
)
writeLines(
  paste0(
    "100 openat(5<", dirfd_include_dir,
    ">, \"header.h\", O_RDONLY|O_CLOEXEC) = 7"
  ),
  unresolved_relative_trace_path, useBytes = TRUE
)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_capture_native_build_dependencies(
    trace_path = unresolved_relative_trace_path,
    build_working_dir = ".",
    tracer_path = strace_path,
    trace_invocation = trace_invocation
  ),
  "relative native build path lacks resolved strace identity",
  "relative dirfd open without a resolved fd annotation fails closed"
)

disappeared_path <- file.path(fixture_dir, "disappeared-after-read.hpp")
disappeared_trace_path <- file.path(fixture_dir, "disappeared.strace")
writeLines(c(
  paste0(
    "100 openat(AT_FDCWD, \"", header_path,
    "\", O_RDONLY|O_CLOEXEC) = 3<", header_path, ">"
  ),
  paste0(
    "100 openat(AT_FDCWD, \"", disappeared_path,
    "\", O_RDONLY|O_CLOEXEC) = 4<", disappeared_path, ">"
  )
), disappeared_trace_path, useBytes = TRUE)
assert_error_identical(
  fastkpc_full_cuda_fixed_sp_capture_native_build_dependencies(
    trace_path = disappeared_trace_path,
    build_working_dir = ".",
    tracer_path = strace_path,
    trace_invocation = trace_invocation
  ),
  paste0(
    "native build successful read/exec path disappeared before hash capture: ",
    disappeared_path
  ),
  "missing successful non-pseudo reads fail closed with exact evidence"
)

writeLines("// dependency mutation", header_path, useBytes = TRUE)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_verify_native_build_dependencies(dependencies),
  "native build dependency changed",
  "native build dependency verification rejects byte mutation"
)
writeLines("// fixture header", header_path, useBytes = TRUE)
writeLines(c(trace_lines, "100 +++ exited with 0 +++"), trace_path,
           useBytes = TRUE)
assert_error_identical(
  fastkpc_full_cuda_fixed_sp_verify_native_build_dependencies(dependencies),
  "native build trace changed during qualification",
  "native build dependency verification rejects raw trace mutation"
)
writeLines(trace_lines, trace_path, useBytes = TRUE)

loaded_state <- compiler_path
loaded_paths <- function() loaded_state
unload_without_effect <- function(path) invisible(NULL)
assert_error_matching(
  .fastkpc_cuda_unload_exact_for_rebuild(
    compiler_path,
    loaded_paths = loaded_paths,
    unload = unload_without_effect
  ),
  "remains loaded after rebuild unload",
  "rebuild fails closed when the exact old DLL remains registered"
)

assert_error_matching(
  .fastkpc_cuda_unload_exact_for_rebuild(
    compiler_path,
    loaded_paths = function() character(),
    mapped_paths = function() compiler_path,
    unload = function(path) fail("stale unregistered mapping was unloaded")
  ),
  "fresh R process",
  "rebuild rejects a registered-absent but mapped stale DLL"
)

expected_sha256 <- fastkpc_full_cuda_fixed_sp_sha256_file(compiler_path)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_verify_loaded_native_library(
    compiler_path,
    expected_sha256,
    loaded_paths = function() compiler_path,
    mapped_paths = function() character()
  ),
  "not mapped",
  "runtime provenance rejects registered but unmapped native identity"
)
assert_true(
  fastkpc_full_cuda_fixed_sp_verify_loaded_native_library(
    compiler_path,
    expected_sha256,
    loaded_paths = function() compiler_path,
    mapped_paths = function() compiler_path
  ),
  "runtime provenance accepts exact registered and mapped native identity"
)
load_state <- new.env(parent = emptyenv())
load_state$paths <- character()
load_state$hash <- expected_sha256
hash_file <- function(path) load_state$hash
load_with_hash_drift <- function(path) {
  load_state$paths <- path
  load_state$hash <- strrep("a", 64L)
  invisible(NULL)
}
assert_error_matching(
  .fastkpc_cuda_load_built_library_exact(
    compiler_path,
    expected_sha256 = expected_sha256,
    hash_file = hash_file,
    loaded_paths = function() load_state$paths,
    mapped_paths = function() load_state$paths,
    load = load_with_hash_drift
  ),
  "changed while loading",
  "qualified load rejects native bytes changed during dyn.load"
)

load_state <- new.env(parent = emptyenv())
load_state$paths <- character()
load_state$hash <- expected_sha256
assert_error_matching(
  .fastkpc_cuda_load_built_library_exact(
    compiler_path,
    expected_sha256 = expected_sha256,
    hash_file = function(path) load_state$hash,
    loaded_paths = function() load_state$paths,
    mapped_paths = function() load_state$paths,
    load = function(path) {
      load_state$paths <- paste0(path, ".other")
      invisible(NULL)
    }
  ),
  "exact built DLL path is not loaded",
  "qualified load rejects a loaded-path mismatch"
)

load_state <- new.env(parent = emptyenv())
load_state$called <- FALSE
assert_error_matching(
  .fastkpc_cuda_load_built_library_exact(
    compiler_path,
    expected_sha256 = expected_sha256,
    hash_file = function(path) expected_sha256,
    loaded_paths = function() character(),
    mapped_paths = function() compiler_path,
    load = function(path) load_state$called <- TRUE
  ),
  "fresh R process",
  "qualified load rejects a stale exact mapping before dyn.load"
)
assert_true(
  !load_state$called,
  "qualified stale-mapping rejection happens before dyn.load"
)

load_state <- new.env(parent = emptyenv())
load_state$paths <- character()
assert_error_matching(
  .fastkpc_cuda_load_built_library_exact(
    compiler_path,
    expected_sha256 = expected_sha256,
    hash_file = function(path) expected_sha256,
    loaded_paths = function() load_state$paths,
    mapped_paths = function() character(),
    load = function(path) load_state$paths <- path
  ),
  "not mapped after dyn.load",
  "qualified load requires exact mapped-object identity after dyn.load"
)

mapping_source <- file.path(fixture_dir, "mapping_fixture.cpp")
mapping_library <- file.path(
  fixture_dir, paste0("mapping_fixture", .Platform$dynlib.ext)
)
writeLines(c(
  "#include <R.h>",
  "#include <Rinternals.h>",
  "template <typename T> struct Holder { static T value; };",
  "template <typename T> T Holder<T>::value = 1;",
  paste(
    "extern \"C\" SEXP mapping_fixture(void) {",
    "return ScalarInteger(Holder<int>::value); }"
  )
), mapping_source, useBytes = TRUE)
mapping_build_status <- system2(
  file.path(R.home("bin"), "R"),
  c(
    "CMD", "SHLIB", shQuote(mapping_source),
    "-o", shQuote(mapping_library)
  ),
  stdout = FALSE, stderr = FALSE
)
assert_identical(
  mapping_build_status, 0L,
  "no-CUDA mapped-object reproduction library builds"
)
mapping_library <- normalizePath(
  mapping_library, winslash = "/", mustWork = TRUE
)
dyn.load(mapping_library)
dyn.unload(mapping_library)
assert_identical(
  unlink(mapping_library, force = TRUE), 0L,
  "no-CUDA mapped-object reproduction unlinks the unloaded pathname"
)
assert_true(
  !mapping_library %in% .fastkpc_cuda_loaded_paths() &&
    mapping_library %in% .fastkpc_cuda_mapped_object_paths(),
  "real dyn.unload removes registration while GNU-unique ELF stays mapped"
)
assert_error_matching(
  .fastkpc_cuda_unload_exact_for_rebuild(mapping_library),
  "fresh R process",
  "real stale same-path ELF mapping requires a fresh R process"
)

cat("PASS Phase 3C native build provenance no-CUDA gate\n")
