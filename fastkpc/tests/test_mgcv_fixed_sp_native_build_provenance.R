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

fixture_dir <- tempfile(
  "fastkpc_native_build_provenance_", tmpdir = tempdir()
)
dir.create(fixture_dir)
on.exit(unlink(fixture_dir, recursive = TRUE, force = TRUE), add = TRUE)

compiler_path <- file.path(fixture_dir, "fixture-compiler")
header_path <- file.path(fixture_dir, "fixture header.hpp")
output_path <- file.path(fixture_dir, "fixture-output.o")
dirfd_include_dir <- file.path(fixture_dir, "pkg", "include")
dir.create(dirfd_include_dir, recursive = TRUE)
dirfd_header_path <- file.path(dirfd_include_dir, "header.h")
writeLines("#!/bin/sh", compiler_path, useBytes = TRUE)
writeLines("// fixture header", header_path, useBytes = TRUE)
writeLines("fixture output", output_path, useBytes = TRUE)
writeLines("// dirfd fixture header", dirfd_header_path, useBytes = TRUE)
compiler_path <- normalizePath(compiler_path, winslash = "/", mustWork = TRUE)
header_path <- normalizePath(header_path, winslash = "/", mustWork = TRUE)
output_path <- normalizePath(output_path, winslash = "/", mustWork = TRUE)
dirfd_include_dir <- normalizePath(
  dirfd_include_dir, winslash = "/", mustWork = TRUE
)
dirfd_header_path <- normalizePath(
  dirfd_header_path, winslash = "/", mustWork = TRUE
)
trace_path <- file.path(fixture_dir, "build.strace")
writeLines(c(
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
    "100 openat(5<", dirfd_include_dir,
    ">, \"header.h\", O_RDONLY|O_CLOEXEC) = 7<",
    dirfd_header_path, ">"
  ),
  "100 openat(AT_FDCWD, \"/dev/null\", O_RDONLY|O_CLOEXEC) = 8</dev/null>",
  paste0(
    "100 openat(AT_FDCWD, \"", fixture_dir,
    "/missing.hpp\", O_RDONLY|O_CLOEXEC) = -1 ENOENT"
  )
), trace_path, useBytes = TRUE)

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
    compiler_path, dirfd_header_path, header_path, output_path, strace_path
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
    identical(dependencies$tracer_path, strace_path) &&
    identical(dependencies$trace_invocation, trace_invocation) &&
    grepl("^[0-9a-f]{64}$", dependencies$tracer_sha256) &&
    grepl("^[0-9a-f]{64}$", dependencies$aggregate_sha256),
  "native build dependency closure has strict inspectable identity"
)
assert_true(
  isTRUE(fastkpc_full_cuda_fixed_sp_validate_native_build_dependencies(
    dependencies
  )),
  "native build dependency closure validates"
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

writeLines("// dependency mutation", header_path, useBytes = TRUE)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_verify_native_build_dependencies(dependencies),
  "native build dependency changed",
  "native build dependency verification rejects byte mutation"
)

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

expected_sha256 <- fastkpc_full_cuda_fixed_sp_sha256_file(compiler_path)
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
    load = function(path) {
      load_state$paths <- paste0(path, ".other")
      invisible(NULL)
    }
  ),
  "exact built DLL path is not loaded",
  "qualified load rejects a loaded-path mismatch"
)

cat("PASS Phase 3C native build provenance no-CUDA gate\n")
