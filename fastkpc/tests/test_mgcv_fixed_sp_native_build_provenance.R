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
maps_records <- .fastkpc_cuda_mapped_object_records(
  maps_path = maps_fixture_path
)
assert_true(
  identical(
    names(maps_records),
    c(
      "path", "live_path", "deleted", "device_major_hex",
      "device_minor_hex", "inode"
    )
  ) &&
    identical(
      maps_records$path,
      c(paste0(maps_deleted_path, " (deleted)"), "/tmp/plain.so")
    ) &&
    identical(maps_records$live_path, c(NA_character_, "/tmp/plain.so")) &&
    identical(maps_records$deleted, c(TRUE, FALSE)) &&
    identical(maps_records$inode, c("42", "43")),
  "Linux mapped-object records preserve deleted path and inode identity"
)
assert_identical(
  .fastkpc_cuda_mapped_object_paths(maps_path = maps_fixture_path),
  "/tmp/plain.so",
  "live-path snapshot excludes deleted mappings instead of normalizing them"
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
      "full-cuda-ci-native-build-dependencies-v3"
    ) && identical(
      dependencies$trace_semantics,
      "linux-strace-successful-read-exec-evidence-v3"
    ) && identical(
      dependencies$build_working_dir,
      normalizePath(".", winslash = "/", mustWork = TRUE)
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
working_dir_hash_mutation <- dependencies
working_dir_hash_mutation$build_working_dir <- fixture_dir
assert_true(
  !identical(
    fastkpc_full_cuda_fixed_sp_native_build_dependency_hash(
      working_dir_hash_mutation
    ),
    dependencies$aggregate_sha256
  ),
  "native build dependency aggregate binds the normalized build directory"
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

pathname_token_paths <- file.path(
  fixture_dir, c("O_WRONLY_header.hpp", "O_TRUNC_header.hpp")
)
for (path in pathname_token_paths) {
  writeLines("// pathname token fixture", path, useBytes = TRUE)
}
pathname_token_paths <- normalizePath(
  pathname_token_paths, winslash = "/", mustWork = TRUE
)
pathname_token_dependencies <- capture_ordered_fixture(
  "quoted-pathname-flag-tokens",
  vapply(
    seq_along(pathname_token_paths),
    function(index) paste0(
      "100 openat(AT_FDCWD, \"", pathname_token_paths[[index]],
      "\", O_RDONLY|O_CLOEXEC) = ", 30L + index, "<",
      pathname_token_paths[[index]], ">"
    ),
    character(1L)
  )
)
assert_true(
  all(pathname_token_paths %in% pathname_token_dependencies$files$path) &&
    !any(pathname_token_paths %in%
         pathname_token_dependencies$exclusions$path),
  "quoted pathname flag tokens do not alter open flag classification"
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

assert_true(
  fastkpc_full_cuda_fixed_sp_verify_native_build_dependencies(dependencies),
  "valid native build evidence reconstructs from its retained raw trace"
)
forged_dependencies <- dependencies
forged_dependencies$files <- rbind(
  forged_dependencies$files,
  data.frame(
    path = output_path,
    sha256 = fastkpc_full_cuda_fixed_sp_sha256_file(output_path),
    stringsAsFactors = FALSE
  )
)
forged_dependencies$files <- forged_dependencies$files[
  order(forged_dependencies$files$path, method = "radix"), , drop = FALSE
]
rownames(forged_dependencies$files) <- as.character(seq_len(nrow(
  forged_dependencies$files
)))
forged_dependencies$dependency_count <- as.integer(nrow(
  forged_dependencies$files
))
forged_dependencies$exclusions <- forged_dependencies$exclusions[
  forged_dependencies$exclusions$path != output_path, , drop = FALSE
]
rownames(forged_dependencies$exclusions) <- as.character(seq_len(nrow(
  forged_dependencies$exclusions
)))
forged_dependencies$exclusion_count <- as.integer(nrow(
  forged_dependencies$exclusions
))
forged_dependencies$aggregate_sha256 <-
  fastkpc_full_cuda_fixed_sp_native_build_dependency_hash(
    forged_dependencies
  )
assert_true(
  fastkpc_full_cuda_fixed_sp_validate_native_build_dependencies(
    forged_dependencies
  ),
  "forged native build tables remain internally self-consistent"
)
assert_error_identical(
  fastkpc_full_cuda_fixed_sp_verify_native_build_dependencies(
    forged_dependencies
  ),
  "native build dependency evidence does not reconstruct from retained trace",
  paste0(
    "verification rejects self-consistent forged dependency/exclusion tables ",
    "by reparsing the retained trace"
  )
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

pin_build_dir <- file.path(fixture_dir, "pin-build")
dir.create(pin_build_dir)
canonical_native_path <- file.path(pin_build_dir, "fastkpc_cuda.so")
original_native_bytes <- charToRaw("qualified native bytes")
replacement_native_bytes <- charToRaw("replacement native bytes")
writeBin(original_native_bytes, canonical_native_path)
canonical_native_path <- normalizePath(
  canonical_native_path, winslash = "/", mustWork = TRUE
)
canonical_native_sha256 <-
  fastkpc_full_cuda_fixed_sp_sha256_file(canonical_native_path)
live_owner_snapshot_original <- if (exists(
  ".fastkpc_cuda_live_owner_snapshot", envir = .GlobalEnv, inherits = FALSE
)) get(".fastkpc_cuda_live_owner_snapshot", envir = .GlobalEnv) else NULL
assign(
  ".fastkpc_cuda_live_owner_snapshot",
  function() list(
    runtime = 1L, prepared = 0L, residual = 0L, total = 1L
  ),
  envir = .GlobalEnv
)
on.exit({
  if (is.null(live_owner_snapshot_original)) {
    if (exists(".fastkpc_cuda_live_owner_snapshot", envir = .GlobalEnv,
               inherits = FALSE)) {
      rm(".fastkpc_cuda_live_owner_snapshot", envir = .GlobalEnv)
    }
  } else {
    assign(".fastkpc_cuda_live_owner_snapshot", live_owner_snapshot_original,
           envir = .GlobalEnv)
  }
}, add = TRUE)
assert_error_matching(
  .fastkpc_cuda_unload_exact_for_rebuild(
    canonical_native_path,
    loaded_paths = function() character(),
    mapped_paths = function() character(),
    unload = function(path) fail("live-owner guard must run before unload")
  ),
  "live fixed-SP CUDA external pointers",
  "qualified rebuild unload fails closed while fixed-SP extptr owners are live"
)
assign(
  ".fastkpc_cuda_live_owner_snapshot",
  function() list(
    runtime = 0L, prepared = 0L, residual = 0L, total = 0L
  ),
  envir = .GlobalEnv
)
pin_load_state <- new.env(parent = emptyenv())
pin_load_state$loaded_paths <- character()
pin_load_state$mapped_paths <- character()
pin_load_state$loaded_bytes <- raw()
pin_load_state$load_path <- ""
pinned_native_path <- .fastkpc_cuda_pin_and_load_built_library(
  canonical_native_path,
  expected_sha256 = canonical_native_sha256,
  loaded_paths = function() pin_load_state$loaded_paths,
  mapped_paths = function() pin_load_state$mapped_paths,
  load = function(path) {
    replacement_path <- file.path(pin_build_dir, "replacement.so")
    writeBin(replacement_native_bytes, replacement_path)
    assert_true(
      file.rename(replacement_path, canonical_native_path),
      "canonical native fixture replacement succeeds"
    )
    pin_load_state$load_path <- path
    pin_load_state$loaded_bytes <- readBin(
      path, what = "raw", n = file.info(path)$size
    )
    pin_load_state$loaded_paths <- path
    pin_load_state$mapped_paths <- path
    invisible(NULL)
  },
  unload = function(path) fail("successful pinned load was unloaded")
)
assert_true(
  !identical(pinned_native_path, canonical_native_path) &&
    identical(pin_load_state$load_path, pinned_native_path) &&
    identical(basename(pinned_native_path), "fastkpc_cuda.so") &&
    identical(
      dirname(dirname(pinned_native_path)),
      normalizePath(pin_build_dir, winslash = "/", mustWork = TRUE)
    ) && file.exists(pinned_native_path) &&
    identical(pin_load_state$loaded_bytes, original_native_bytes) &&
    identical(
      readBin(
        canonical_native_path, what = "raw",
        n = file.info(canonical_native_path)$size
      ),
      replacement_native_bytes
    ),
  paste0(
    "qualified load uses a unique exact-basename hard-link snapshot whose ",
    "bytes survive canonical replacement"
  )
)
pin_record_identity <- .fastkpc_cuda_posix_file_identity(pinned_native_path)
pin_records <- function(paths) {
  if (length(paths) == 0L) {
    return(data.frame(
      path = character(), live_path = character(), deleted = logical(),
      device_major_hex = character(), device_minor_hex = character(),
      inode = character(), stringsAsFactors = FALSE
    ))
  }
  data.frame(
    path = paths,
    live_path = paths,
    deleted = rep(FALSE, length(paths)),
    device_major_hex = rep(
      pin_record_identity$device_major_hex, length(paths)
    ),
    device_minor_hex = rep(
      pin_record_identity$device_minor_hex, length(paths)
    ),
    inode = rep(pin_record_identity$inode, length(paths)),
    stringsAsFactors = FALSE
  )
}
protected_prune <- .fastkpc_cuda_prune_qualified_pins(
  roots = pin_build_dir,
  loaded_paths = function() pinned_native_path,
  mapped_records = function() pin_records(pinned_native_path)
)
assert_true(
  dir.exists(dirname(pinned_native_path)) &&
    !dirname(pinned_native_path) %in% protected_prune,
  "qualified pin cleanup preserves active loaded/mapped pins"
)
unmapped_prune <- .fastkpc_cuda_prune_qualified_pins(
  roots = pin_build_dir,
  loaded_paths = function() character(),
  mapped_records = function() pin_records(character())
)
assert_true(
  !dir.exists(dirname(pinned_native_path)) &&
    dirname(pinned_native_path) %in% unmapped_prune,
  "qualified pin cleanup removes stale unregistered/unmapped pins"
)

pin_drift_state <- new.env(parent = emptyenv())
pin_drift_state$load_called <- FALSE
pin_snapshot_dirs <- function() sort(list.files(
  pin_build_dir,
  pattern = "^\\.fastkpc_cuda-qualified-",
  all.files = TRUE,
  full.names = TRUE
), method = "radix")
pin_drift_dirs_before <- pin_snapshot_dirs()
assert_error_matching(
  .fastkpc_cuda_pin_and_load_built_library(
    canonical_native_path,
    expected_sha256 = fastkpc_full_cuda_fixed_sp_sha256_file(
      canonical_native_path
    ),
    hash_file = function(path) strrep("a", 64L),
    loaded_paths = function() character(),
    mapped_paths = function() character(),
    load = function(path) pin_drift_state$load_called <- TRUE
  ),
  "does not match qualified build hash",
  "pinned native hash drift fails closed"
)
assert_true(
  !pin_drift_state$load_called &&
    identical(pin_snapshot_dirs(), pin_drift_dirs_before),
  "pinned native hash drift fails before load and cleans its private directory"
)

retry_load_state <- new.env(parent = emptyenv())
retry_load_state$attempt_paths <- character()
retry_load_state$loaded_paths <- character()
retry_load_state$mapped_paths <- character()
retry_load_state$unloaded_paths <- character()
retry_load <- function(path) {
  retry_load_state$attempt_paths <- c(
    retry_load_state$attempt_paths, path
  )
  retry_load_state$loaded_paths <- path
  if (length(retry_load_state$attempt_paths) > 1L) {
    retry_load_state$mapped_paths <- c(
      retry_load_state$mapped_paths, path
    )
  }
  invisible(NULL)
}
retry_unload <- function(path) {
  retry_load_state$unloaded_paths <- c(
    retry_load_state$unloaded_paths, path
  )
  retry_load_state$loaded_paths <- setdiff(
    retry_load_state$loaded_paths, path
  )
  retry_load_state$mapped_paths <- unique(c(
    retry_load_state$mapped_paths, path
  ))
  invisible(NULL)
}
retry_native_sha256 <- fastkpc_full_cuda_fixed_sp_sha256_file(
  canonical_native_path
)
assert_error_matching(
  .fastkpc_cuda_pin_and_load_built_library(
    canonical_native_path,
    expected_sha256 = retry_native_sha256,
    loaded_paths = function() retry_load_state$loaded_paths,
    mapped_paths = function() retry_load_state$mapped_paths,
    load = retry_load,
    unload = retry_unload
  ),
  "not mapped after dyn.load",
  "post-load pinned native verification failure is rethrown"
)
first_retry_path <- retry_load_state$attempt_paths[[1L]]
assert_true(
  identical(retry_load_state$unloaded_paths, first_retry_path) &&
    !dir.exists(dirname(first_retry_path)) &&
    first_retry_path %in% retry_load_state$mapped_paths,
  "post-load pinned native failure unloads the exact path and cleans staging"
)
second_retry_path <- .fastkpc_cuda_pin_and_load_built_library(
  canonical_native_path,
  expected_sha256 = retry_native_sha256,
  loaded_paths = function() retry_load_state$loaded_paths,
  mapped_paths = function() retry_load_state$mapped_paths,
  load = retry_load,
  unload = retry_unload
)
assert_true(
  length(retry_load_state$attempt_paths) == 2L &&
    !identical(second_retry_path, first_retry_path) &&
    identical(retry_load_state$loaded_paths, second_retry_path) &&
    all(c(first_retry_path, second_retry_path) %in%
        retry_load_state$mapped_paths),
  "a unique pinned pathname permits retry after residual failed-load mapping"
)
rollback_loaded <- second_retry_path
rollback_state <- new.env(parent = emptyenv())
rollback_state$loaded_paths <- rollback_loaded
rollback_state$mapped_paths <- rollback_loaded
rollback_state$unloaded <- character()
.fastkpc_cuda_rollback_qualified_native_load(
  list(native_library_path = rollback_loaded,
       native_library_sha256 = retry_native_sha256),
  loaded_paths = function() rollback_state$loaded_paths,
  mapped_records = function() pin_records(rollback_state$mapped_paths),
  unload = function(path) {
    rollback_state$unloaded <- c(rollback_state$unloaded, path)
    rollback_state$loaded_paths <- setdiff(rollback_state$loaded_paths, path)
    rollback_state$mapped_paths <- setdiff(rollback_state$mapped_paths, path)
    invisible(NULL)
  }
)
assert_true(
  identical(rollback_state$unloaded, rollback_loaded) &&
    !dir.exists(dirname(rollback_loaded)),
  "post-load qualified provenance failure rollback unloads and removes its pin"
)
assert_true(
  identical(pin_snapshot_dirs(), pin_drift_dirs_before),
  "qualified load failure and rollback leave no stale pin directory growth"
)
exit_cleanup_dir <- file.path(
  pin_build_dir, ".fastkpc_cuda-qualified-exit-fixture"
)
dir.create(exit_cleanup_dir)
exit_cleanup_path <- file.path(exit_cleanup_dir, "fastkpc_cuda.so")
writeBin(original_native_bytes, exit_cleanup_path)
exit_cleanup_path <- normalizePath(
  exit_cleanup_path, winslash = "/", mustWork = TRUE
)
exit_cleanup_sha256 <- fastkpc_full_cuda_fixed_sp_sha256_file(
  exit_cleanup_path
)
.fastkpc_cuda_remember_pinned_library(exit_cleanup_path, exit_cleanup_sha256)
exit_cleanup_removed <- .fastkpc_cuda_cleanup_remembered_pins_at_exit()
assert_true(
  !dir.exists(exit_cleanup_dir) &&
    normalizePath(exit_cleanup_dir, winslash = "/", mustWork = FALSE) %in%
      exit_cleanup_removed,
  "qualified pin process-exit cleanup helper removes remembered pin dirs"
)

qualified_loader_fixture <- function() {
  function_names <- c(
    "build_fastkpc_cuda_native", ".fastkpc_cuda_sha256_file",
    ".fastkpc_cuda_load_built_library_exact",
    ".fastkpc_cuda_pin_and_load_built_library",
    ".fastkpc_cuda_verify_registered_library_identity"
  )
  originals <- lapply(
    function_names, get, envir = .GlobalEnv, inherits = FALSE
  )
  names(originals) <- function_names
  on.exit({
    for (name in function_names) {
      assign(name, originals[[name]], envir = .GlobalEnv)
    }
  }, add = TRUE)
  state <- new.env(parent = emptyenv())
  state$legacy_load_path <- ""
  state$pin_input_path <- ""
  state$full_verification_count <- 0L
  mock_pinned_dir <- file.path(pin_build_dir, ".mock-qualified-snapshot")
  dir.create(mock_pinned_dir)
  mock_pinned_path <- file.path(mock_pinned_dir, "fastkpc_cuda.so")
  writeBin(original_native_bytes, mock_pinned_path)
  mock_pinned_path <- normalizePath(
    mock_pinned_path, winslash = "/", mustWork = TRUE
  )
  assign(
    "build_fastkpc_cuda_native",
    function(rebuild, trace_path, tracer_path) canonical_native_path,
    envir = .GlobalEnv
  )
  assign(
    ".fastkpc_cuda_sha256_file",
    function(path) canonical_native_sha256,
    envir = .GlobalEnv
  )
  assign(
    ".fastkpc_cuda_load_built_library_exact",
    function(so, ...) {
      state$legacy_load_path <- so
      invisible(so)
    },
    envir = .GlobalEnv
  )
  assign(
    ".fastkpc_cuda_pin_and_load_built_library",
    function(so, expected_sha256, ...) {
      state$pin_input_path <- so
      invisible(mock_pinned_path)
    },
    envir = .GlobalEnv
  )
  assign(
    ".fastkpc_cuda_verify_registered_library_identity",
    function(path, expected_sha256, ...) {
      state$full_verification_count <- state$full_verification_count + 1L
      TRUE
    },
    envir = .GlobalEnv
  )
  result <- load_fastkpc_cuda_native_qualified(
    trace_path = trace_path, tracer_path = strace_path
  )
  list(
    result = result,
    state = state,
    mock_pinned_path = mock_pinned_path
  )
}
qualified_loader_result <- qualified_loader_fixture()
assert_true(
  identical(
    qualified_loader_result$result$native_library_path,
    qualified_loader_result$mock_pinned_path
  ) && identical(
    qualified_loader_result$state$pin_input_path,
    canonical_native_path
  ) && !nzchar(qualified_loader_result$state$legacy_load_path) &&
    identical(qualified_loader_result$state$full_verification_count, 1L),
  paste(
    "qualified loader fully verifies the pinned registered identity exactly",
    "once before publishing it"
  )
)

qualified_loader_verification_failure_fixture <- function() {
  function_names <- c(
    "build_fastkpc_cuda_native", ".fastkpc_cuda_sha256_file",
    ".fastkpc_cuda_pin_and_load_built_library",
    ".fastkpc_cuda_verify_registered_library_identity",
    ".fastkpc_cuda_rollback_qualified_native_load",
    ".fastkpc_cuda_remember_identity"
  )
  originals <- lapply(
    function_names, get, envir = .GlobalEnv, inherits = FALSE
  )
  names(originals) <- function_names
  on.exit({
    for (name in function_names) {
      assign(name, originals[[name]], envir = .GlobalEnv)
    }
  }, add = TRUE)
  state <- new.env(parent = emptyenv())
  state$rollback_count <- 0L
  state$remember_count <- 0L
  assign(
    "build_fastkpc_cuda_native",
    function(rebuild, trace_path, tracer_path) canonical_native_path,
    envir = .GlobalEnv
  )
  assign(
    ".fastkpc_cuda_sha256_file",
    function(path) canonical_native_sha256,
    envir = .GlobalEnv
  )
  assign(
    ".fastkpc_cuda_pin_and_load_built_library",
    function(so, expected_sha256, ...) {
      invisible(qualified_loader_result$mock_pinned_path)
    },
    envir = .GlobalEnv
  )
  assign(
    ".fastkpc_cuda_verify_registered_library_identity",
    function(path, expected_sha256, ...) {
      stop("fixture full verification failure", call. = FALSE)
    },
    envir = .GlobalEnv
  )
  assign(
    ".fastkpc_cuda_rollback_qualified_native_load",
    function(native_load, ...) {
      state$rollback_count <- state$rollback_count + 1L
      invisible(native_load$native_library_path)
    },
    envir = .GlobalEnv
  )
  assign(
    ".fastkpc_cuda_remember_identity",
    function(path, sha256) {
      state$remember_count <- state$remember_count + 1L
      invisible(path)
    },
    envir = .GlobalEnv
  )
  assert_error_matching(
    load_fastkpc_cuda_native_qualified(
      trace_path = trace_path, tracer_path = strace_path
    ),
    "fixture full verification failure",
    "qualified loader propagates full identity verification failure"
  )
  state
}
qualified_loader_failure_state <-
  qualified_loader_verification_failure_fixture()
assert_true(
  identical(qualified_loader_failure_state$rollback_count, 1L) &&
    identical(qualified_loader_failure_state$remember_count, 0L),
  paste(
    "qualified loader rolls back a loaded pin and does not cache identity",
    "when full verification fails"
  )
)

registered_loader_fixture <- function() {
  function_names <- c(
    "build_fastkpc_cuda_native", "getLoadedDLLs",
    ".fastkpc_cuda_verify_registered_library_identity",
    ".fastkpc_cuda_sha256_file", "dyn.load"
  )
  existed <- vapply(
    function_names, exists, logical(1L), envir = .GlobalEnv,
    inherits = FALSE
  )
  originals <- lapply(function_names[existed], function(name) {
    get(name, envir = .GlobalEnv, inherits = FALSE)
  })
  names(originals) <- function_names[existed]
  on.exit({
    for (name in function_names) {
      if (existed[[name]]) {
        assign(name, originals[[name]], envir = .GlobalEnv)
      } else if (exists(name, envir = .GlobalEnv, inherits = FALSE)) {
        rm(list = name, envir = .GlobalEnv)
      }
    }
  }, add = TRUE)
  state <- new.env(parent = emptyenv())
  state$build_called <- FALSE
  state$load_path <- ""
  state$loaded_dll_query_count <- 0L
  state$full_verification_count <- 0L
  state$hash_count <- 0L
  assign(
    "build_fastkpc_cuda_native",
    function(rebuild = FALSE, ...) {
      state$build_called <- TRUE
      canonical_native_path
    },
    envir = .GlobalEnv
  )
  assign(
    "getLoadedDLLs",
    function() {
      state$loaded_dll_query_count <- state$loaded_dll_query_count + 1L
      list(fastkpc_cuda = list(path =
        qualified_loader_result$mock_pinned_path))
    },
    envir = .GlobalEnv
  )
  assign(
    ".fastkpc_cuda_sha256_file",
    function(path) {
      state$hash_count <- state$hash_count + 1L
      canonical_native_sha256
    },
    envir = .GlobalEnv
  )
  assign(
    ".fastkpc_cuda_verify_registered_library_identity",
    function(path, expected_sha256, ...) {
      state$full_verification_count <- state$full_verification_count + 1L
      .fastkpc_cuda_sha256_file(path)
      TRUE
    },
    envir = .GlobalEnv
  )
  assign(
    "dyn.load",
    function(path) {
      state$load_path <- path
      invisible(NULL)
    },
    envir = .GlobalEnv
  )
  results <- lapply(seq_len(8L), function(index) {
    load_fastkpc_cuda_native(rebuild = FALSE)
  })
  list(results = results, state = state)
}
registered_loader_result <- registered_loader_fixture()
assert_true(
  all(vapply(
    registered_loader_result$results,
    identical,
    logical(1L),
    qualified_loader_result$mock_pinned_path
  )) && !registered_loader_result$state$build_called &&
    !nzchar(registered_loader_result$state$load_path) &&
    identical(registered_loader_result$state$loaded_dll_query_count, 8L),
  "generic CUDA access cheaply guards and reuses the registered pinned package identity"
)
assert_true(
  identical(registered_loader_result$state$full_verification_count, 0L) &&
    identical(registered_loader_result$state$hash_count, 0L),
  paste(
    "cached CUDA access must not repeat full native verification or SHA-256",
    "on each wrapper call"
  )
)
wrong_registered_loader_fixture <- function() {
  function_names <- c(
    "build_fastkpc_cuda_native", "getLoadedDLLs", "getNativeSymbolInfo",
    ".fastkpc_cuda_sha256_file", "dyn.load"
  )
  existed <- vapply(
    function_names, exists, logical(1L), envir = .GlobalEnv,
    inherits = FALSE
  )
  originals <- lapply(function_names[existed], function(name) {
    get(name, envir = .GlobalEnv, inherits = FALSE)
  })
  names(originals) <- function_names[existed]
  on.exit({
    for (name in function_names) {
      if (existed[[name]]) {
        assign(name, originals[[name]], envir = .GlobalEnv)
      } else if (exists(name, envir = .GlobalEnv, inherits = FALSE)) {
        rm(list = name, envir = .GlobalEnv)
      }
    }
  }, add = TRUE)
  wrong_path <- file.path(pin_build_dir, "wrong-fastkpc_cuda.so")
  writeBin(charToRaw("wrong-fastkpc_cuda-so"), wrong_path)
  wrong_path <- normalizePath(wrong_path, winslash = "/", mustWork = TRUE)
  assign(
    "build_fastkpc_cuda_native",
    function(rebuild = FALSE, ...) canonical_native_path,
    envir = .GlobalEnv
  )
  assign(
    "getLoadedDLLs",
    function() list(fastkpc_cuda = list(path = wrong_path, name = "fastkpc_cuda")),
    envir = .GlobalEnv
  )
  assign(
    "getNativeSymbolInfo",
    function(name, PACKAGE, withRegistrationInfo = FALSE) {
      list(name = name, dll = list(name = "fastkpc_cuda", path = wrong_path))
    },
    envir = .GlobalEnv
  )
  assign(
    ".fastkpc_cuda_sha256_file",
    function(path) {
      path <- normalizePath(path, winslash = "/", mustWork = FALSE)
      if (identical(path, canonical_native_path)) {
        canonical_native_sha256
      } else {
        fastkpc_full_cuda_fixed_sp_sha256_file(path)
      }
    },
    envir = .GlobalEnv
  )
  assign(
    "dyn.load",
    function(path) fail("wrong registered package must fail before dyn.load"),
    envir = .GlobalEnv
  )
  assert_error_matching(
    load_fastkpc_cuda_native(rebuild = FALSE),
    "registered",
    "generic loader rejects a wrong same-name registered package image"
  )
}
wrong_registered_loader_fixture()

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
mock_dllinfo <- function(name, path) {
  structure(list(name = name, path = path), class = "DLLInfo")
}
mock_symbol_info <- function(name, path, dll_name = "fastkpc_cuda") {
  list(name = name, dll = mock_dllinfo(dll_name, path))
}
mock_file_identity <- function(path, device_major_hex = "08",
                               device_minor_hex = "01", inode = "42") {
  list(
    path = normalizePath(path, winslash = "/", mustWork = FALSE),
    device_major_hex = device_major_hex,
    device_minor_hex = device_minor_hex,
    inode = inode
  )
}
mock_mapped_record <- function(path, device_major_hex = "08",
                               device_minor_hex = "01", inode = "42") {
  normalized <- normalizePath(path, winslash = "/", mustWork = FALSE)
  data.frame(
    path = normalized,
    live_path = normalized,
    deleted = FALSE,
    device_major_hex = device_major_hex,
    device_minor_hex = device_minor_hex,
    inode = inode,
    stringsAsFactors = FALSE
  )
}
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_verify_loaded_native_library(
    compiler_path,
    expected_sha256,
    loaded_dlls = function() list(fastkpc_cuda = mock_dllinfo(
      "fastkpc_cuda", compiler_path
    )),
    mapped_records = function() data.frame(
      path = character(),
      live_path = character(),
      deleted = logical(),
      device_major_hex = character(),
      device_minor_hex = character(),
      inode = character(),
      stringsAsFactors = FALSE
    ),
    file_identity = function(path) mock_file_identity(path),
    symbol_info = function(name, PACKAGE, withRegistrationInfo = FALSE) {
      mock_symbol_info(name, compiler_path)
    }
  ),
  "not mapped",
  "runtime provenance rejects registered but unmapped native identity"
)
assert_true(
  fastkpc_full_cuda_fixed_sp_verify_loaded_native_library(
    compiler_path,
    expected_sha256,
    loaded_dlls = function() list(fastkpc_cuda = mock_dllinfo(
      "fastkpc_cuda", compiler_path
    )),
    mapped_records = function() data.frame(
      path = compiler_path,
      live_path = compiler_path,
      deleted = FALSE,
      device_major_hex = "08",
      device_minor_hex = "01",
      inode = "42",
      stringsAsFactors = FALSE
    ),
    file_identity = function(path) mock_file_identity(path),
    symbol_info = function(name, PACKAGE, withRegistrationInfo = FALSE) {
      mock_symbol_info(name, compiler_path)
    }
  ),
  "runtime provenance accepts exact registered, mapped, and symbol-bound native identity"
)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_verify_loaded_native_library(
    compiler_path,
    expected_sha256,
    loaded_dlls = function() list(fastkpc_cuda = mock_dllinfo(
      "fastkpc_cuda", compiler_path
    )),
    mapped_records = function() mock_mapped_record(
      compiler_path,
      device_major_hex = NA_character_,
      device_minor_hex = NA_character_,
      inode = NA_character_
    ),
    file_identity = function(path) mock_file_identity(path),
    symbol_info = function(name, PACKAGE, withRegistrationInfo = FALSE) {
      mock_symbol_info(name, compiler_path)
    }
  ),
  "mapped inode identity is missing",
  "runtime provenance rejects mapped records without inode identity"
)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_verify_loaded_native_library(
    compiler_path,
    expected_sha256,
    loaded_dlls = function() list(fastkpc_cuda = mock_dllinfo(
      "fastkpc_cuda", compiler_path
    )),
    mapped_records = function() rbind(
      mock_mapped_record(compiler_path),
      mock_mapped_record(
        compiler_path,
        device_major_hex = NA_character_,
        device_minor_hex = NA_character_,
        inode = NA_character_
      )
    ),
    file_identity = function(path) mock_file_identity(path),
    symbol_info = function(name, PACKAGE, withRegistrationInfo = FALSE) {
      mock_symbol_info(name, compiler_path)
    }
  ),
  "mapped inode identity is missing",
  "runtime provenance rejects partially missing mapped inode identity"
)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_verify_loaded_native_library(
    compiler_path,
    expected_sha256,
    loaded_dlls = function() list(fastkpc_cuda = mock_dllinfo(
      "fastkpc_cuda", compiler_path
    )),
    mapped_records = function() data.frame(
      path = paste0(compiler_path, " (deleted)"),
      live_path = NA_character_,
      deleted = TRUE,
      device_major_hex = "08",
      device_minor_hex = "01",
      inode = "42",
      stringsAsFactors = FALSE
    ),
    file_identity = function(path) mock_file_identity(path),
    symbol_info = function(name, PACKAGE, withRegistrationInfo = FALSE) {
      mock_symbol_info(name, compiler_path)
    }
  ),
  "deleted",
  "runtime provenance rejects deleted mapped-object records as live identity"
)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_verify_loaded_native_library(
    compiler_path,
    expected_sha256,
    loaded_dlls = function() list(fastkpc_cuda = mock_dllinfo(
      "fastkpc_cuda", compiler_path
    )),
    mapped_records = function() data.frame(
      path = compiler_path,
      live_path = compiler_path,
      deleted = FALSE,
      device_major_hex = "08",
      device_minor_hex = "01",
      inode = "999",
      stringsAsFactors = FALSE
    ),
    file_identity = function(path) mock_file_identity(path, inode = "42"),
    symbol_info = function(name, PACKAGE, withRegistrationInfo = FALSE) {
      mock_symbol_info(name, compiler_path)
    }
  ),
  "inode",
  "runtime provenance rejects mapped inode mismatches"
)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_verify_loaded_native_library(
    compiler_path,
    expected_sha256,
    loaded_dlls = function() list(fastkpc_cuda = mock_dllinfo(
      "fastkpc_cuda", compiler_path
    )),
    mapped_records = function() data.frame(
      path = compiler_path,
      live_path = compiler_path,
      deleted = FALSE,
      device_major_hex = "08",
      device_minor_hex = "01",
      inode = "42",
      stringsAsFactors = FALSE
    ),
    file_identity = function(path) mock_file_identity(path),
    symbol_info = function(name, PACKAGE, withRegistrationInfo = FALSE) {
      mock_symbol_info(name, paste0(compiler_path, ".wrong"))
    }
  ),
  "symbol",
  "runtime provenance rejects phase3 symbols bound to a different DLL"
)

default_provenance_dir <- tempfile(
  "fastkpc_production_mapped_records_", tmpdir = "fastkpc/tests"
)
dir.create(default_provenance_dir)
on.exit(unlink(default_provenance_dir, recursive = TRUE, force = TRUE),
        add = TRUE)
default_provenance_source_path <- file.path(
  default_provenance_dir, "runner.R"
)
writeLines("default_provenance_value <- 1L", default_provenance_source_path,
           useBytes = TRUE)
default_provenance_closure <-
  fastkpc_full_cuda_fixed_sp_discover_execution_source_closure(
    root_sources = c(default_runner = default_provenance_source_path),
    project_root = "."
  )
default_provenance_source_hashes <- vapply(
  default_provenance_closure$source_file_paths,
  fastkpc_full_cuda_fixed_sp_sha256_file,
  character(1L)
)
default_provenance_native_path <- file.path(
  default_provenance_dir, "fastkpc_cuda.so"
)
writeBin(charToRaw("strict mapped native fixture"),
         default_provenance_native_path)
default_provenance_native_path <- normalizePath(
  default_provenance_native_path, winslash = "/", mustWork = TRUE
)
default_provenance_native_inputs <- file.path(
  default_provenance_dir, c("build.sh", "native.cu")
)
names(default_provenance_native_inputs) <- c("build", "native")
writeLines("#!/bin/sh\nexit 0", default_provenance_native_inputs[["build"]],
           useBytes = TRUE)
writeLines("// strict mapped native input",
           default_provenance_native_inputs[["native"]], useBytes = TRUE)
default_provenance_native_inputs[] <- vapply(
  default_provenance_native_inputs,
  normalizePath, character(1L), winslash = "/", mustWork = TRUE
)
default_provenance_native_input_hashes <- vapply(
  default_provenance_native_inputs,
  fastkpc_full_cuda_fixed_sp_sha256_file,
  character(1L)
)
default_provenance_native_sha256 <-
  fastkpc_full_cuda_fixed_sp_sha256_file(default_provenance_native_path)
default_provenance_native_identity <-
  .fastkpc_cuda_posix_file_identity(default_provenance_native_path)
default_provenance_mapped_records <- function(
    device_major_hex = default_provenance_native_identity$device_major_hex,
    device_minor_hex = default_provenance_native_identity$device_minor_hex,
    inode = default_provenance_native_identity$inode) {
  force(device_major_hex)
  force(device_minor_hex)
  force(inode)
  function(maps_path = "/proc/self/maps") {
    data.frame(
      path = default_provenance_native_path,
      live_path = default_provenance_native_path,
      deleted = FALSE,
      device_major_hex = device_major_hex,
      device_minor_hex = device_minor_hex,
      inode = inode,
      stringsAsFactors = FALSE
    )
  }
}
default_provenance_loaded_paths <- function() default_provenance_native_path
with_default_native_observers <- function(mapped_records, expression) {
  function_names <- c(".fastkpc_cuda_mapped_object_records",
                      "getNativeSymbolInfo")
  existed <- vapply(
    function_names, exists, logical(1L), envir = .GlobalEnv,
    inherits = FALSE
  )
  originals <- lapply(function_names[existed], function(name) {
    get(name, envir = .GlobalEnv, inherits = FALSE)
  })
  names(originals) <- function_names[existed]
  on.exit({
    for (name in function_names) {
      if (existed[[name]]) {
        assign(name, originals[[name]], envir = .GlobalEnv)
      } else if (exists(name, envir = .GlobalEnv, inherits = FALSE)) {
        rm(list = name, envir = .GlobalEnv)
      }
    }
  }, add = TRUE)
  assign(".fastkpc_cuda_mapped_object_records", mapped_records,
         envir = .GlobalEnv)
  assign(
    "getNativeSymbolInfo",
    function(name, PACKAGE, withRegistrationInfo = FALSE) {
      if (!identical(PACKAGE, "fastkpc_cuda")) {
        fail("unexpected native symbol PACKAGE lookup")
      }
      mock_symbol_info(name, default_provenance_native_path)
    },
    envir = .GlobalEnv
  )
  force(expression)
}
capture_default_execution_provenance <- function() {
  fastkpc_full_cuda_fixed_sp_capture_execution_provenance(
    source_closure = default_provenance_closure,
    expected_source_sha256 = default_provenance_source_hashes,
    native_library_path = default_provenance_native_path,
    native_build_input_paths = default_provenance_native_inputs,
    expected_native_build_input_sha256 =
      default_provenance_native_input_hashes,
    native_build_dependencies = dependencies,
    expected_native_library_sha256 = default_provenance_native_sha256,
    loaded_paths = default_provenance_loaded_paths
  )
}
execution_provenance_formals <- list(
  capture = names(formals(
    fastkpc_full_cuda_fixed_sp_capture_execution_provenance
  )),
  verify = names(formals(
    fastkpc_full_cuda_fixed_sp_verify_execution_provenance
  ))
)
assert_true(
  "mapped_records" %in% execution_provenance_formals$capture &&
    "mapped_records" %in% execution_provenance_formals$verify &&
    !"mapped_paths" %in% execution_provenance_formals$capture &&
    !"mapped_paths" %in% execution_provenance_formals$verify,
  "execution provenance production API exposes strict mapped records, not path snapshots"
)
default_execution_provenance <- with_default_native_observers(
  default_provenance_mapped_records(),
  capture_default_execution_provenance()
)
assert_true(
  identical(default_execution_provenance$native_library_path,
            default_provenance_native_path) &&
    identical(
      default_execution_provenance$native_library_identity,
      "qualified-pinned-inode-sha-exact-registered-mapped-path-v3"
    ) && !isTRUE(
      default_execution_provenance$execution_sources_unchanged_after_run
    ),
  "execution provenance production default accepts strict mapped-object identity"
)
assert_error_matching(
  with_default_native_observers(
    default_provenance_mapped_records(
      device_major_hex = NA_character_,
      device_minor_hex = NA_character_,
      inode = NA_character_
    ),
    capture_default_execution_provenance()
  ),
  "mapped inode identity is missing",
  "execution provenance production default rejects missing mapped inode identity"
)
verified_default_execution_provenance <- with_default_native_observers(
  default_provenance_mapped_records(),
  fastkpc_full_cuda_fixed_sp_verify_execution_provenance(
    default_execution_provenance,
    loaded_paths = default_provenance_loaded_paths
  )
)
assert_true(
  isTRUE(
    verified_default_execution_provenance$execution_sources_unchanged_after_run
  ),
  "execution provenance verification default accepts strict mapped-object identity"
)
assert_error_matching(
  with_default_native_observers(
    default_provenance_mapped_records(
      device_major_hex = NA_character_,
      device_minor_hex = NA_character_,
      inode = NA_character_
    ),
    fastkpc_full_cuda_fixed_sp_verify_execution_provenance(
      default_execution_provenance,
      loaded_paths = default_provenance_loaded_paths
    )
  ),
  "execution source snapshot changed during qualification",
  "execution provenance verification default rejects missing mapped inode identity"
)
unlink(default_provenance_dir, recursive = TRUE, force = TRUE)
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
mapping_records_after_unload <- .fastkpc_cuda_mapped_object_records()
assert_true(
  !mapping_library %in% .fastkpc_cuda_loaded_paths() &&
    paste0(mapping_library, " (deleted)") %in%
      mapping_records_after_unload$path &&
    !mapping_library %in% .fastkpc_cuda_mapped_object_paths(),
  paste(
    "real dyn.unload removes registration while GNU-unique ELF remains",
    "mapped only as a deleted record"
  )
)
assert_error_matching(
  .fastkpc_cuda_unload_exact_for_rebuild(mapping_library),
  "fresh R process",
  "real stale same-path ELF mapping requires a fresh R process"
)

cat("PASS Phase 3C native build provenance no-CUDA gate\n")
