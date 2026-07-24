source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
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
assert_sha_vector <- function(value, size, message) {
  assert_true(
    is.character(value) && length(value) == size && !anyNA(value) &&
      all(grepl("^[0-9a-f]{64}$", value)),
    message
  )
}
assert_bare_vector <- function(value, type, size, message,
                               allow_na = FALSE) {
  assert_true(
    typeof(value) == type && length(value) == size && !is.object(value) &&
      is.null(attributes(value)) && (allow_na || !anyNA(value)),
    message
  )
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

independent_open_flag_tokens <- function(lines, syscalls = NULL) {
  if (is.null(syscalls)) {
    syscall_matches <- regmatches(
      lines,
      regexec(
        "^[[:space:]]*[0-9]+[[:space:]]+(openat2|openat|open)\\(",
        lines, perl = TRUE
      )
    )
    assert_true(
      all(lengths(syscall_matches) == 2L),
      "independent open syscall extraction is complete"
    )
    syscalls <- vapply(syscall_matches, `[[`, character(1L), 2L)
  }
  assert_true(
    typeof(lines) == "character" && typeof(syscalls) == "character" &&
      identical(length(lines), length(syscalls)) &&
      all(syscalls %in% c("open", "openat", "openat2")),
    "independent open flag inputs are aligned"
  )
  quoted_path <- '"(?:\\\\.|[^"\\\\])*"'
  expressions <- vapply(seq_along(lines), function(index) {
    prefix <- switch(
      syscalls[[index]],
      open = paste0("open\\([[:space:]]*", quoted_path),
      openat = paste0(
        "openat\\([^,]+,[[:space:]]*", quoted_path
      ),
      openat2 = paste0(
        "openat2\\([^,]+,[[:space:]]*", quoted_path
      )
    )
    suffix <- if (identical(syscalls[[index]], "openat2")) {
      paste0(
        "[[:space:]]*,[[:space:]]*\\{[[:space:]]*",
        "flags[[:space:]]*=[[:space:]]*([^,}]*)"
      )
    } else {
      "[[:space:]]*,[[:space:]]*([^,)]*)"
    }
    matched <- regmatches(
      lines[[index]],
      regexec(paste0(prefix, suffix), lines[[index]], perl = TRUE)
    )[[1L]]
    assert_true(
      length(matched) == 2L,
      "independent open flag extraction is complete"
    )
    trimws(matched[[2L]])
  }, character(1L))
  unname(lapply(expressions, function(expression) {
    pieces <- strsplit(expression, "[^A-Z0-9_]+", perl = TRUE)[[1L]]
    unname(pieces[grepl("^O_[A-Z0-9_]+$", pieces)])
  }))
}

independent_open_events <- function(lines, syscalls = NULL) {
  tokens <- independent_open_flag_tokens(lines, syscalls)
  has_flag <- function(flag) {
    vapply(tokens, function(value) flag %in% value, logical(1L))
  }
  read <- (has_flag("O_RDONLY") | has_flag("O_RDWR")) &
    !has_flag("O_WRONLY")
  generation <- has_flag("O_TRUNC") | has_flag("O_TMPFILE") |
    (has_flag("O_CREAT") & has_flag("O_EXCL"))
  list(
    read = read,
    generation = generation,
    directory = read & has_flag("O_DIRECTORY")
  )
}

independent_generation_open <- function(lines, syscalls = NULL) {
  independent_open_events(lines, syscalls)$generation
}

independent_ordered_generated_paths <- function(
    event_paths, access_event, generation_event) {
  assert_true(
    typeof(event_paths) == "character" &&
      identical(length(event_paths), length(access_event)) &&
      identical(length(event_paths), length(generation_event)),
    "independent ordered generation inputs are aligned"
  )
  paths <- unique(event_paths[access_event])
  access_paths <- event_paths
  access_paths[!access_event] <- NA_character_
  generation_paths <- event_paths
  generation_paths[!generation_event] <- NA_character_
  first_access <- match(paths, access_paths)
  first_generation <- match(paths, generation_paths)
  setNames(
    !is.na(first_generation) & first_generation <= first_access,
    paths
  )
}

independent_csv_semantic_frame <- function(value) {
  rownames(value) <- NULL
  value
}

independent_native_build_command_lines <- function(value) {
  lines <- c(
    paste0(
      "command_projection.schema=",
      value$command_projection_schema_version
    ),
    paste0("command_count=", value$command_count)
  )
  for (index in seq_along(value$commands)) {
    command <- value$commands[[index]]
    lines <- c(
      lines,
      paste0("command.", index, ".role=", command$role),
      paste0(
        "command.", index, ".executable_path=", command$executable_path
      ),
      paste0(
        "command.", index, ".executable_sha256=",
        command$executable_sha256
      ),
      paste0("command.", index, ".argc=", length(command$argv))
    )
    for (argument_index in seq_along(command$argv)) {
      argument <- command$argv[[argument_index]]
      lines <- c(lines, paste0(
        "command.", index, ".argv.", argument_index, "=",
        nchar(argument, type = "bytes"), ":", argument
      ))
    }
  }
  environment <- value$build_environment
  lines <- c(
    lines,
    paste0(
      "build_environment.schema=",
      value$build_environment_schema_version
    ),
    paste0("build_environment.count=", nrow(environment))
  )
  for (index in seq_len(nrow(environment))) {
    lines <- c(
      lines,
      paste0(
        "build_environment.", index, ".name=", environment$name[[index]]
      ),
      paste0(
        "build_environment.", index, ".is_set=",
        if (environment$is_set[[index]]) "true" else "false"
      ),
      paste0(
        "build_environment.", index, ".value=",
        nchar(environment$value[[index]], type = "bytes"), ":",
        environment$value[[index]]
      )
    )
  }
  lines
}

independent_native_build_environment_names <- sort(c(
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

synthetic_multibyte_value <- intToUtf8(0x03bb)
synthetic_native_build_projection <- list(
  command_projection_schema_version =
    "full-cuda-ci-native-build-command-projection-v1",
  command_count = 1L,
  commands = list(list(
    role = "cuda_compile",
    executable_path = "/opt/cuda/bin/nvcc",
    executable_sha256 = strrep("a", 64L),
    argv = c("<EXECUTABLE>", "-c", "src.cu", "-o", "<OUTPUT>")
  )),
  build_environment_schema_version =
    "full-cuda-ci-native-build-environment-v1",
  build_environment = data.frame(
    name = c("LC_ALL", "PATH"),
    is_set = c(TRUE, FALSE),
    value = c(synthetic_multibyte_value, ""),
    stringsAsFactors = FALSE
  )
)
assert_identical(
  independent_native_build_command_lines(synthetic_native_build_projection),
  c(
    "command_projection.schema=full-cuda-ci-native-build-command-projection-v1",
    "command_count=1",
    "command.1.role=cuda_compile",
    "command.1.executable_path=/opt/cuda/bin/nvcc",
    paste0("command.1.executable_sha256=", strrep("a", 64L)),
    "command.1.argc=5",
    "command.1.argv.1=12:<EXECUTABLE>",
    "command.1.argv.2=2:-c",
    "command.1.argv.3=6:src.cu",
    "command.1.argv.4=2:-o",
    "command.1.argv.5=8:<OUTPUT>",
    "build_environment.schema=full-cuda-ci-native-build-environment-v1",
    "build_environment.count=2",
    "build_environment.1.name=LC_ALL",
    "build_environment.1.is_set=true",
    paste0("build_environment.1.value=2:", synthetic_multibyte_value),
    "build_environment.2.name=PATH",
    "build_environment.2.is_set=false",
    "build_environment.2.value=0:"
  ),
  "independent v3 encoder length-prefixes command and environment values"
)

assert_identical(
  independent_generation_open(c(
    "100 openat(AT_FDCWD, \"/tmp/input\", O_RDWR|O_CLOEXEC) = 3</tmp/input>",
    paste0(
      "100 openat(AT_FDCWD, \"/tmp/ambiguous\", ",
      "O_WRONLY|O_CREAT) = 4</tmp/ambiguous>"
    ),
    paste0(
      "100 openat(AT_FDCWD, \"/tmp/truncated\", ",
      "O_WRONLY|O_TRUNC) = 5</tmp/truncated>"
    ),
    paste0(
      "100 openat(AT_FDCWD, \"/tmp/#1\", ",
      "O_RDWR|O_TMPFILE) = 6</tmp/#1>"
    ),
    paste0(
      "100 openat(AT_FDCWD, \"/tmp/exclusive\", ",
      "O_RDWR|O_CREAT|O_EXCL) = 7</tmp/exclusive>"
    )
  )),
  c(FALSE, FALSE, TRUE, TRUE, TRUE),
  "independent generation flags reject O_RDWR and ambiguous O_CREAT"
)
pathname_token_open_lines <- c(
  paste0(
    "100 openat(AT_FDCWD, \"/tmp/O_WRONLY_header.hpp\", ",
    "O_RDONLY|O_CLOEXEC) = 31</tmp/O_WRONLY_header.hpp>"
  ),
  paste0(
    "100 openat(AT_FDCWD, \"/tmp/O_TRUNC_header.hpp\", ",
    "O_RDONLY|O_CLOEXEC) = 32</tmp/O_TRUNC_header.hpp>"
  ),
  paste0(
    "100 openat2(AT_FDCWD, ",
    "\"/tmp/escaped_\\\"O_TRUNC\\\"_header.hpp\", ",
    "{flags=O_RDONLY|O_CLOEXEC, mode=0, ",
    "resolve=RESOLVE_BENEATH}, 24) = 33",
    "</tmp/escaped_\\\"O_TRUNC\\\"_header.hpp>"
  ),
  paste0(
    "100 openat(AT_FDCWD, \"/tmp/exact-token.hpp\", ",
    "O_RDONLY_HEADER|O_TRUNCATED|O_DIRECTORYISH|",
    "O_CREATING|O_EXCLUSIVE) = 34</tmp/exact-token.hpp>"
  )
)
pathname_token_open_syscalls <- c("openat", "openat", "openat2", "openat")
expected_pathname_token_flags <- c(
  rep(list(c("O_RDONLY", "O_CLOEXEC")), 3L),
  list(c(
    "O_RDONLY_HEADER", "O_TRUNCATED", "O_DIRECTORYISH", "O_CREATING",
    "O_EXCLUSIVE"
  ))
)
assert_identical(
  independent_generation_open(pathname_token_open_lines),
  rep(FALSE, length(pathname_token_open_lines)),
  "independent generation ignores flag tokens in quoted pathnames"
)
production_pathname_token_flags <- tryCatch(
  fastkpc_full_cuda_fixed_sp_open_flag_tokens(
    pathname_token_open_lines, pathname_token_open_syscalls
  ),
  error = identity
)
independent_pathname_token_flags <- tryCatch(
  independent_open_flag_tokens(
    pathname_token_open_lines, pathname_token_open_syscalls
  ),
  error = identity
)
assert_true(
  identical(production_pathname_token_flags, expected_pathname_token_flags) &&
    identical(independent_pathname_token_flags,
              expected_pathname_token_flags) &&
    identical(production_pathname_token_flags,
              independent_pathname_token_flags),
  "raw trace and independent parsers agree on pathname flag tokens"
)
assert_identical(
  unname(independent_ordered_generated_paths(
    event_paths = c("/tmp/late", "/tmp/late"),
    access_event = c(TRUE, FALSE),
    generation_event = c(FALSE, TRUE)
  )),
  FALSE,
  "independent generation cannot use truncation after first access"
)
assert_identical(
  unname(independent_ordered_generated_paths(
    event_paths = c("/tmp/prior", "/tmp/prior"),
    access_event = c(FALSE, TRUE),
    generation_event = c(TRUE, FALSE)
  )),
  TRUE,
  "independent generation accepts truncation before first access"
)

csv_semantic_fixture <- data.frame(
  path = c("/tmp/second", "/tmp/first"),
  reason = c("generated_output", "pseudo_fs"),
  stringsAsFactors = FALSE,
  row.names = c("trace-2", "trace-1")
)
csv_semantic_path <- tempfile("fastkpc_phase3c_csv_semantics_", fileext = ".csv")
utils::write.csv(csv_semantic_fixture, csv_semantic_path, row.names = FALSE)
published_csv_semantic_fixture <- utils::read.csv(
  csv_semantic_path, stringsAsFactors = FALSE, check.names = FALSE
)
unlink(csv_semantic_path, force = TRUE)
canonical_csv_semantic_fixture <-
  independent_csv_semantic_frame(csv_semantic_fixture)
assert_true(
  identical(canonical_csv_semantic_fixture, published_csv_semantic_fixture) &&
    identical(
      .row_names_info(canonical_csv_semantic_fixture, type = 0L),
      c(NA_integer_, -2L)
    ),
  "independent CSV semantic frames use automatic row names"
)

pass_fixture_output <- tempfile("fastkpc_phase3c_supplied_pass_")
assert_error_matching(
  fastkpc_full_cuda_write_fixed_sp_qualification_artifact(
    result = list(summary = list(pass = TRUE)),
    output_dir = pass_fixture_output
  ),
  "caller-supplied pass",
  "qualification writer rejects a caller-supplied pass boolean first"
)
assert_true(
  !file.exists(pass_fixture_output) && !dir.exists(pass_fixture_output),
  "supplied-pass rejection happens before artifact publication"
)

publication_fixture_parent <- tempfile("fastkpc_phase3c_publication_")
dir.create(publication_fixture_parent)
publication_order_fixture <- c(
  "payload.txt", "manifest.json", "summary.json"
)
make_publication_staging <- function(name) {
  staging <- file.path(publication_fixture_parent, name)
  dir.create(staging)
  writeLines("payload", file.path(staging, "payload.txt"), useBytes = TRUE)
  writeLines("manifest", file.path(staging, "manifest.json"), useBytes = TRUE)
  writeLines("summary", file.path(staging, "summary.json"), useBytes = TRUE)
  staging
}
publication_staging <- make_publication_staging("staging-race")
publication_hashes <- setNames(vapply(
  publication_order_fixture,
  function(name) fastkpc_full_cuda_fixed_sp_sha256_file(
    file.path(publication_staging, name)
  ),
  character(1L)
), publication_order_fixture)

concurrent_empty_destination <- file.path(
  publication_fixture_parent, "concurrent-empty-output"
)
dir.create(concurrent_empty_destination)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_publish_qualification_staging(
    staging_dir = publication_staging,
    output_dir = concurrent_empty_destination,
    publication_order = publication_order_fixture,
    expected_sha256 = publication_hashes
  ),
  "exclusive qualification output reservation failed",
  "concurrently-created empty destination is never replaced"
)
assert_true(
  dir.exists(concurrent_empty_destination) &&
    length(list.files(concurrent_empty_destination, all.files = FALSE)) == 0L &&
    identical(
      sort(list.files(publication_staging), method = "radix"),
      sort(publication_order_fixture, method = "radix")
    ),
  "failed exclusive reservation preserves destination and staging identity"
)

concurrent_sentinel_destination <- file.path(
  publication_fixture_parent, "concurrent-sentinel-output"
)
dir.create(concurrent_sentinel_destination)
sentinel_path <- file.path(concurrent_sentinel_destination, "sentinel.txt")
writeLines("concurrent-owner", sentinel_path, useBytes = TRUE)
sentinel_sha256 <- fastkpc_full_cuda_fixed_sp_sha256_file(sentinel_path)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_publish_qualification_staging(
    staging_dir = publication_staging,
    output_dir = concurrent_sentinel_destination,
    publication_order = publication_order_fixture,
    expected_sha256 = publication_hashes
  ),
  "exclusive qualification output reservation failed",
  "preexisting destination with sentinel is never replaced"
)
assert_true(
  identical(
    fastkpc_full_cuda_fixed_sp_sha256_file(sentinel_path), sentinel_sha256
  ) && identical(list.files(concurrent_sentinel_destination), "sentinel.txt"),
  "failed publication leaves concurrent sentinel bytes untouched"
)

post_move_failure_results <- lapply(
  c("payload.txt", "summary.json"),
  function(failure_name) {
    staging <- make_publication_staging(paste0(
      "staging-post-move-", sub("\\.json$", "", failure_name)
    ))
    hashes <- setNames(vapply(
      publication_order_fixture,
      function(name) fastkpc_full_cuda_fixed_sp_sha256_file(
        file.path(staging, name)
      ),
      character(1L)
    ), publication_order_fixture)
    destination <- file.path(
      publication_fixture_parent,
      paste0("post-move-output-", sub("\\.json$", "", failure_name))
    )
    concurrent_sentinel <- file.path(destination, "concurrent-sentinel.txt")
    sentinel_contents <- paste0("concurrent-owner-after-", failure_name)
    move_then_error <- function(from, to) {
      moved <- file.rename(from, to)
      if (isTRUE(moved) && identical(basename(to), failure_name)) {
        writeLines(sentinel_contents, concurrent_sentinel, useBytes = TRUE)
        stop("injected post-rename failure for ", failure_name,
             call. = FALSE)
      }
      moved
    }
    error <- tryCatch({
      fastkpc_full_cuda_fixed_sp_publish_qualification_staging(
        staging_dir = staging,
        output_dir = destination,
        publication_order = publication_order_fixture,
        expected_sha256 = hashes,
        move_file = move_then_error
      )
      NULL
    }, error = identity)
    list(
      expected_error = inherits(error, "error") && grepl(
        paste0("injected post-rename failure for ", failure_name),
        conditionMessage(error), fixed = TRUE
      ),
      completion_marker_absent =
        !file.exists(file.path(destination, "summary.json")),
      owned_residue_absent = !any(file.exists(file.path(
        destination, publication_order_fixture
      ))),
      concurrent_sentinel_preserved =
        file.exists(concurrent_sentinel) && identical(
          readLines(concurrent_sentinel, warn = FALSE), sentinel_contents
        )
    )
  }
)
assert_true(
  all(vapply(post_move_failure_results, function(result) {
    all(unlist(result, use.names = FALSE))
  }, logical(1L))),
  paste(
    "post-rename payload and summary failures remove only writer-owned",
    "files and preserve concurrent sentinels"
  )
)

successful_staging <- make_publication_staging("staging-success")
successful_hashes <- setNames(vapply(
  publication_order_fixture,
  function(name) fastkpc_full_cuda_fixed_sp_sha256_file(
    file.path(successful_staging, name)
  ),
  character(1L)
), publication_order_fixture)
successful_destination <- file.path(publication_fixture_parent, "output")
assert_identical(
  fastkpc_full_cuda_fixed_sp_publish_qualification_staging(
    staging_dir = successful_staging,
    output_dir = successful_destination,
    publication_order = publication_order_fixture,
    expected_sha256 = successful_hashes
  ),
  normalizePath(successful_destination, winslash = "/", mustWork = TRUE),
  "qualification staging publishes into one exclusively reserved directory"
)
assert_true(
  file.exists(file.path(successful_destination, "summary.json")) &&
    !dir.exists(successful_staging) &&
    setequal(list.files(successful_destination), publication_order_fixture),
  "summary presence is the qualification publication completion marker"
)
unlink(publication_fixture_parent, recursive = TRUE, force = TRUE)
assert_true(
  !dir.exists(publication_fixture_parent),
  "publication fixture cleans its owned paths"
)

catalog_artifact_fixture_dir <- file.path(
  "fastkpc", "artifacts", "full_cuda_ci",
  "fixed_sp_cuda_qualification_v1"
)
if (!dir.exists(catalog_artifact_fixture_dir)) {
  catalog_artifact_fixture_dir <- file.path(
    "fastkpc", "artifacts", "full_cuda_ci",
    "fixed_sp_cuda_qualification_v1.pre_task8_4692f79_20260719"
  )
}
catalog_target_records_fixture <- readRDS(file.path(
  catalog_artifact_fixture_dir, "target_parity.rds"
))
catalog_batch_records_fixture <- readRDS(file.path(
  catalog_artifact_fixture_dir, "batch_metrics.rds"
))
catalog_setup_records_fixture <- readRDS(file.path(
  catalog_artifact_fixture_dir, "setup_metrics.rds"
))

catalog_fixture <- data.frame(
  scope = "qualification",
  authenticated = TRUE,
  catalog_open_count = 1L,
  setup_count = 2061L,
  target_count = 6143L,
  scope_subset_hash =
    "0adea2bac7b31615421f180b6caa5aeef5567bafa0e45a319b358136bf429c61",
  ordered_setup_key_digest =
    "0286e36adf55ce4536711d015eb428442914b0c2d753dfc3148ff40f6b2f158e",
  ordered_target_key_digest =
    "51210ecaec11e1da5623841404bef0f9f8558a959bf9de531d9438839ee321ee",
  stringsAsFactors = FALSE
)
catalog_summary_fixture <- list(
  scope_subset_hash = catalog_fixture$scope_subset_hash,
  qualification_subset_hash = catalog_fixture$scope_subset_hash,
  ordered_setup_key_digest =
    "0286e36adf55ce4536711d015eb428442914b0c2d753dfc3148ff40f6b2f158e",
  ordered_target_key_digest =
    "51210ecaec11e1da5623841404bef0f9f8558a959bf9de531d9438839ee321ee",
  route_status_hash = fastkpc_full_cuda_census_metadata_hash(list(
    catalog_target_records_fixture$residual_key_sha256,
    catalog_target_records_fixture$planned_route,
    catalog_target_records_fixture$executed_route,
    catalog_target_records_fixture$reroute_reason,
    catalog_target_records_fixture$solver_status
  )),
  numeric_hash = fastkpc_full_cuda_census_metadata_hash(list(
    catalog_target_records_fixture$residual_key_sha256,
    catalog_target_records_fixture$fitted_numeric_hash,
    catalog_target_records_fixture$residual_numeric_hash
  )),
  setup_count = 2061L,
  target_count = 6143L
)
authoritative_catalog_fixture <- list(
  qualification_subset_hash =
    "0adea2bac7b31615421f180b6caa5aeef5567bafa0e45a319b358136bf429c61",
  ordered_setup_key_digest = catalog_summary_fixture$ordered_setup_key_digest,
  ordered_target_key_digest = catalog_summary_fixture$ordered_target_key_digest,
  setup_count = 2061L,
  target_count = 6143L
)
authoritative_record_fixture <- c(authoritative_catalog_fixture, list(
  setup_keys = catalog_setup_records_fixture$prepared_s_key_sha256,
  target_keys = catalog_target_records_fixture$residual_key_sha256,
  target_prepared_s_keys =
    catalog_target_records_fixture$prepared_s_key_sha256
))
validated_record_fixture <-
  fastkpc_full_cuda_fixed_sp_qualification_validate_record_identity(
    catalog_records = catalog_fixture,
    summary = catalog_summary_fixture,
    authoritative = authoritative_record_fixture,
    setup_records = catalog_setup_records_fixture,
    batch_records = catalog_batch_records_fixture,
    target_records = catalog_target_records_fixture
  )
assert_true(
  identical(validated_record_fixture$catalog_authenticated, TRUE) &&
    identical(
      validated_record_fixture$ordered_setup_key_digest,
      catalog_summary_fixture$ordered_setup_key_digest
    ) && identical(
      validated_record_fixture$ordered_target_key_digest,
      catalog_summary_fixture$ordered_target_key_digest
    ) && identical(
      validated_record_fixture$route_status_hash,
      catalog_summary_fixture$route_status_hash
    ) && identical(
      validated_record_fixture$numeric_hash,
      catalog_summary_fixture$numeric_hash
    ),
  "actual qualification rows reproduce the authenticated catalog identity"
)
assert_identical(
  fastkpc_full_cuda_fixed_sp_qualification_validate_catalog_evidence(
    catalog_records = catalog_fixture,
    summary = catalog_summary_fixture,
    authoritative = authoritative_catalog_fixture
  ),
  TRUE,
  "qualification catalog evidence matches canonical authenticated identity"
)
forged_catalog_fixture <- catalog_fixture
forged_catalog_fixture$scope_subset_hash <- strrep("a", 64L)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_qualification_validate_catalog_evidence(
    catalog_records = forged_catalog_fixture,
    summary = catalog_summary_fixture,
    authoritative = authoritative_catalog_fixture
  ),
  "catalog evidence does not match canonical authentication",
  "arbitrary SHA cannot self-attest qualification catalog authentication"
)
forged_catalog_digest_fixture <- catalog_fixture
forged_catalog_digest_fixture$ordered_target_key_digest <- strrep("b", 64L)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_qualification_validate_catalog_evidence(
    catalog_records = forged_catalog_digest_fixture,
    summary = catalog_summary_fixture,
    authoritative = authoritative_catalog_fixture
  ),
  "catalog evidence does not match canonical authentication",
  "catalog record digests cannot self-attest canonical key identity"
)

forged_target_key_records <- catalog_target_records_fixture
forged_target_key_records$residual_key_sha256 <- rep.int(
  strrep("a", 64L), nrow(forged_target_key_records)
)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_qualification_validate_record_identity(
    catalog_records = catalog_fixture,
    summary = catalog_summary_fixture,
    authoritative = authoritative_record_fixture,
    setup_records = catalog_setup_records_fixture,
    batch_records = catalog_batch_records_fixture,
    target_records = forged_target_key_records
  ),
  "qualification published rows do not match canonical catalog identity",
  paste(
    "syntactically valid repeated target keys are rejected before",
    "qualification publication"
  )
)

expected_prepublication_summary_fields <- c(
  "scope", "scope_subset_hash", "ordered_setup_key_digest",
  "ordered_target_key_digest", "route_status_hash", "numeric_hash",
  "setup_count", "target_count", "penalty_root_matrix_count",
  "penalty_root_row_count", "H_root_matrix_count", "planned_cholesky_count",
  "planned_qr_count", "planned_svd_count", "executed_cholesky_count",
  "executed_qr_count", "executed_svd_count", "cholesky_to_svd_count",
  "qr_to_svd_count", "svd_finite_high_count", "svd_nonfinite_count",
  "all_safe_batch_count", "mixed_batch_count", "all_stable_batch_count",
  "true_batched_subgroup_count", "true_batched_target_count",
  "cholesky_single_target_count", "whole_batch_true_batched_count",
  "stable_not_implemented_count", "stable_reroute_count",
  "non_ok_status_count", "root_rank_mismatch_count",
  "aggregate_penalty_factor_count", "aggregate_svd_b_build_count",
  "aggregate_penalty_root_rank_mismatch_count",
  "aggregate_penalty_root_pivot_mismatch_count",
  "aggregate_penalty_root_d2h_count", "aggregate_penalty_root_d2h_bytes",
  "workspace_grow_count_after_warmup",
  "stable_workspace_grow_count_after_warmup",
  "per_target_allocation_count_after_warmup",
  "per_target_handle_create_count", "cuda_device_synchronize_count",
  "target_level_stable_sync_count", "implicit_residual_d2h_count",
  "all_output_slot_leases_released", "invalid_output_init_count",
  "cpu_fallback_count", "unknown_fallback_count", "approximate_backend_count",
  "qr_checkpoint_record_count", "qr_checkpoint_wait_count",
  "svd_checkpoint_record_count", "svd_checkpoint_wait_count",
  "max_residual_abs_diff", "max_residual_relative_l2_diff",
  "max_fitted_abs_diff", "max_fitted_relative_l2_diff",
  "qualification_subset_hash"
)
expected_dcov_summary_fields <- c(
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
)
expected_publication_summary_fields <- c(
  "artifact_schema_version", "catalog_authenticated", "provenance_mode",
  "head_base_commit", "source_closure_schema_version",
  "source_closure_count", "source_closure_sha256",
  "execution_snapshot_sha256", "relevant_sources_dirty_or_untracked",
  "native_build_inputs_sha256", "native_build_dependency_count",
  "native_build_dependencies_sha256", "native_build_trace_sha256",
  "native_build_tracer_sha256", "native_library_sha256",
  "execution_sources_unchanged_after_run",
  "elapsed_seconds", "stage_timing_total_seconds", "payload_file_count"
)
published_summary_namespace_fixture <- jsonlite::fromJSON(
  file.path(catalog_artifact_fixture_dir, "summary.json"),
  simplifyVector = FALSE
)
if (length(setdiff(
      c(
        expected_dcov_summary_fields, "native_build_inputs_sha256",
        "native_build_dependency_count", "native_build_dependencies_sha256",
        "native_build_trace_sha256", "native_build_tracer_sha256"
      ),
      names(published_summary_namespace_fixture)
    )) > 0L || !identical(
      published_summary_namespace_fixture$artifact_schema_version,
      "full-cuda-ci-fixed-sp-qualification-v6"
    )) {
  synthetic_dcov_summary_fixture <- list(
    qualification_dcov_logical_test_count = 3808L,
    qualification_dcov_near_alpha_count = 1478L,
    qualification_dcov_unique_residual_key_count = 6143L,
    qualification_dcov_max_absolute_p_value_difference = 0,
    qualification_dcov_decision_flip_count = 0L,
    qualification_dcov_near_alpha_decision_flip_count = 0L,
    qualification_dcov_backend_error_count = 0L,
    qualification_dcov_spectra_fallback_count = 0L,
    qualification_dcov_logical_ids_hash = strrep("a", 64L),
    qualification_dcov_residual_key_hash = strrep("b", 64L),
    qualification_dcov_rows_hash = strrep("c", 64L)
  )
  synthetic_publication_summary_fixture <-
    published_summary_namespace_fixture[
      intersect(
        expected_publication_summary_fields,
        names(published_summary_namespace_fixture)
      )
    ]
  synthetic_publication_summary_fixture$artifact_schema_version <-
    "full-cuda-ci-fixed-sp-qualification-v6"
  synthetic_publication_summary_fixture$native_build_inputs_sha256 <-
    strrep("d", 64L)
  synthetic_publication_summary_fixture$native_build_dependency_count <- 3L
  synthetic_publication_summary_fixture$native_build_dependencies_sha256 <-
    strrep("e", 64L)
  synthetic_publication_summary_fixture$native_build_trace_sha256 <-
    strrep("1", 64L)
  synthetic_publication_summary_fixture$native_build_tracer_sha256 <-
    strrep("f", 64L)
  synthetic_publication_summary_fixture$payload_file_count <- 17L
  synthetic_publication_summary_fixture <-
    synthetic_publication_summary_fixture[expected_publication_summary_fields]
  published_summary_namespace_fixture <- c(
    published_summary_namespace_fixture[
      expected_prepublication_summary_fields
    ],
    synthetic_dcov_summary_fixture,
    synthetic_publication_summary_fixture
  )
}
canonical_input_summary_fixture <- published_summary_namespace_fixture[
  expected_prepublication_summary_fields
]
assert_identical(
  names(canonical_input_summary_fixture),
  expected_prepublication_summary_fields,
  "pre-publication summary fixture has the frozen canonical namespace"
)

wrong_scope_summary_fixture <- canonical_input_summary_fixture
wrong_scope_summary_fixture$scope <- "iteration"
input_summary_failure_cases <- c(
  list(scope_iteration = wrong_scope_summary_fixture),
  setNames(lapply(c(
    expected_dcov_summary_fields, expected_publication_summary_fields
  ), function(field) {
    value <- canonical_input_summary_fixture
    value[[field]] <- published_summary_namespace_fixture[[field]]
    value
  }), paste0("reserved_", c(
    expected_dcov_summary_fields, expected_publication_summary_fields
  ))),
  list(duplicate_scope = c(
    canonical_input_summary_fixture, list(scope = "qualification")
  ))
)
input_summary_failure_results <- lapply(
  input_summary_failure_cases,
  function(summary_fixture) {
    error <- tryCatch({
      fastkpc_full_cuda_fixed_sp_qualification_validate_input_summary(
        summary_fixture
      )
      NULL
    }, error = identity)
    inherits(error, "error") && grepl(
      "qualification input summary schema is invalid",
      conditionMessage(error), fixed = TRUE
    )
  }
)
assert_true(
  all(unlist(input_summary_failure_results, use.names = FALSE)),
  paste(
    "qualification input summary rejects wrong scope, every publication-owned",
    "field, and duplicate names"
  )
)

duplicate_final_summary_fixture <- c(
  published_summary_namespace_fixture,
  list(elapsed_seconds = published_summary_namespace_fixture$elapsed_seconds)
)
final_type_drift_summary_fixture <- published_summary_namespace_fixture
final_type_drift_summary_fixture$payload_file_count <- as.double(
  final_type_drift_summary_fixture$payload_file_count
)
final_count_drift_summary_fixture <- published_summary_namespace_fixture
final_count_drift_summary_fixture$payload_file_count <- 16L
final_summary_failure_results <- lapply(list(
  duplicate = duplicate_final_summary_fixture,
  type_drift = final_type_drift_summary_fixture,
  count_drift = final_count_drift_summary_fixture
), function(summary_fixture) {
  error <- tryCatch({
    fastkpc_full_cuda_fixed_sp_qualification_validate_published_summary(
      summary_fixture
    )
    NULL
  }, error = identity)
  inherits(error, "error") && grepl(
    "qualification published summary schema is invalid",
    conditionMessage(error), fixed = TRUE
  )
})
assert_true(
  all(unlist(final_summary_failure_results, use.names = FALSE)),
  "published summary rejects duplicate names, type drift, and count drift"
)
assert_identical(
  fastkpc_full_cuda_fixed_sp_qualification_validate_input_summary(
    canonical_input_summary_fixture
  ),
  TRUE,
  "canonical pre-publication summary namespace and types are accepted"
)
assert_identical(
  fastkpc_full_cuda_fixed_sp_qualification_validate_published_summary(
    published_summary_namespace_fixture
  ),
  TRUE,
  "canonical published summary namespace and types are accepted"
)

summary_type_fixture <-
  fastkpc_full_cuda_fixed_sp_phase3c_expected_counts("qualification")
assert_identical(
  fastkpc_full_cuda_fixed_sp_qualification_validate_summary_counts(
    summary_type_fixture
  ),
  TRUE,
  "qualification summary accepts exact bare count scalars"
)
summary_double_drift <- summary_type_fixture
summary_double_drift$setup_count <- 2061
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_qualification_validate_summary_counts(
    summary_double_drift
  ),
  "summary count types",
  "qualification summary rejects integer-to-double drift"
)
summary_character_drift <- summary_type_fixture
summary_character_drift$setup_count <- "2061"
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_qualification_validate_summary_counts(
    summary_character_drift
  ),
  "summary count types",
  "qualification summary rejects character count drift"
)
summary_attribute_drift <- summary_type_fixture
summary_attribute_drift$all_output_slot_leases_released <- structure(
  TRUE, fixture_attribute = "not-bare"
)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_qualification_validate_summary_counts(
    summary_attribute_drift
  ),
  "summary count types",
  "qualification summary rejects scalar attribute drift"
)
summary_json_roundtrip_path <- tempfile(
  "fastkpc_phase3c_summary_type_", fileext = ".json"
)
fastkpc_full_cuda_fixed_sp_write_qualification_json(
  summary_type_fixture, summary_json_roundtrip_path
)
summary_json_roundtrip <- jsonlite::fromJSON(
  summary_json_roundtrip_path, simplifyVector = FALSE
)
unlink(summary_json_roundtrip_path, force = TRUE)
assert_identical(
  summary_json_roundtrip[names(summary_type_fixture)],
  summary_type_fixture,
  "qualification JSON preserves exact integer/logical/double scalar types"
)

schema_fixture_dir <- catalog_artifact_fixture_dir
schema_fixture <- list(
  target_records = readRDS(file.path(schema_fixture_dir, "target_parity.rds")),
  batch_records = readRDS(file.path(schema_fixture_dir, "batch_metrics.rds")),
  setup_records = readRDS(file.path(schema_fixture_dir, "setup_metrics.rds")),
  runtime_records = utils::read.csv(
    file.path(schema_fixture_dir, "runtime_metrics.csv"),
    stringsAsFactors = FALSE, check.names = FALSE
  )
)
for (field in c(
  "cublas_workspace_alignment", "creator_pid", "generation",
  "workspace_bytes", "cublas_workspace_bytes", "eigen_workspace_bytes",
  "qr_workspace_bytes", "svd_workspace_bytes", "augmented_workspace_bytes",
  "aggregate_factor_workspace_bytes"
)) {
  schema_fixture$runtime_records[[field]] <-
    as.double(schema_fixture$runtime_records[[field]])
}
expected_target_rds_names <- c(
  "prepared_s_key_sha256", "batch_ordinal", "target_ordinal",
  "residual_key_sha256", "target", "null_dim", "phase1_condition",
  "condition_bucket", "phase1_coefficient_rank", "planned_route",
  "authenticated_planned_route", "executed_route", "reroute_reason",
  "solver_status", "target_true_batched", "qr_rank", "geqrf_info",
  "ormqr_info", "effective_rank", "sigma_max", "smallest_retained_sigma",
  "svd_info", "aggregate_penalty_root_rank",
  "aggregate_penalty_root_pivot", "aggregate_factor_call_count",
  "aggregate_b_build_count", "aggregate_dstop",
  "cpu_aggregate_penalty_root_rank", "cpu_aggregate_penalty_root_pivot",
  "cpu_aggregate_effective_rank",
  "cpu_aggregate_effective_rank_threshold", "cpu_aggregate_sigma_max",
  "aggregate_penalty_root_rank_exact",
  "aggregate_penalty_root_pivot_exact", "aggregate_effective_rank_exact",
  "numeric_reference", "outputs_all_finite", "residual_max_abs_diff",
  "residual_relative_l2_diff", "fitted_max_abs_diff",
  "fitted_relative_l2_diff", "fitted_numeric_hash",
  "residual_numeric_hash", "oracle_call_count", "oracle_fitted_hash",
  "oracle_residual_hash", "authenticated_fitted_hash",
  "authenticated_residual_hash", "oracle_fitted_hash_exact",
  "oracle_residual_hash_exact", "approximate_backend"
)
expected_batch_rds_names <- c(
  "prepared_s_key_sha256", "batch_ordinal", "target_count",
  "batch_call_count", "native_batch_call", "true_batched_kernel",
  "true_batched_subgroup_count", "true_batched_attempted_target_count",
  "true_batched_target_count", "cholesky_single_target_count",
  "potrf_batched_call_count", "potrs_batched_call_count",
  "planned_cholesky_target_count", "planned_qr_target_count",
  "planned_svd_target_count", "executed_cholesky_target_count",
  "executed_qr_target_count", "executed_svd_target_count",
  "stable_reroute_count", "cholesky_to_svd_count", "qr_to_svd_count",
  "aggregate_penalty_factor_count", "aggregate_svd_b_build_count",
  "aggregate_penalty_root_d2h_count", "aggregate_penalty_root_d2h_bytes",
  "target_batch_h2d_call_count", "target_h2d_copy_count",
  "target_h2d_bytes", "rhs_device_build_count", "full_cuda_data_plane",
  "invalid_output_init_count", "coefficient_batch_finalize_call_count",
  "fitted_batch_finalize_call_count",
  "residual_rss_batch_finalize_call_count",
  "per_target_output_finalize_call_count",
  "batch_output_finalized_target_count", "canonical_output_order_exact",
  "target_keys_exact", "route_status_conservation_exact",
  "resource_snapshot_captured", "resource_instrumentation_version",
  "resource_allocation_count_before_solve",
  "resource_allocation_count_after_solve",
  "resource_handle_create_count_before_solve",
  "resource_handle_create_count_after_solve",
  "cuda_device_allocation_count_during_solve",
  "cuda_host_allocation_count_during_solve",
  "stream_create_count_during_solve", "event_create_count_during_solve",
  "cublas_handle_create_count_during_solve",
  "cusolver_handle_create_count_during_solve",
  "per_target_allocation_count_after_warmup",
  "per_target_handle_create_count", "workspace_grow_count_after_warmup",
  "stable_workspace_grow_count_after_warmup",
  "cuda_device_synchronize_count",
  "cholesky_factor_checkpoint_record_count",
  "cholesky_factor_checkpoint_wait_count",
  "cholesky_solve_checkpoint_record_count",
  "cholesky_solve_checkpoint_wait_count", "qr_checkpoint_record_count",
  "qr_checkpoint_wait_count", "svd_checkpoint_record_count",
  "svd_checkpoint_wait_count", "implicit_residual_d2h_count",
  "cpu_fallback_count", "unknown_fallback_count", "solve_elapsed_ms",
  "pre_shadow_materialize_call_count",
  "pre_shadow_materialize_target_count", "pre_shadow_d2h_bytes",
  "shadow_materialize_elapsed_ms", "post_shadow_materialize_call_count",
  "post_shadow_materialize_target_count", "post_shadow_d2h_bytes",
  "output_slot_release_count", "output_slot_leased_after_release"
)
expected_setup_rds_names <- c(
  "prepared_s_key_sha256", "batch_ordinal", "prepared_handle_create_count",
  "prepared_handle_free_count", "setup_h2d_upload_count", "setup_h2d_bytes",
  "penalty_root_build_count", "penalty_root_rank_mismatch_count",
  "penalty_root_bytes", "penalty_root_build_ms", "penalty_root_matrix_count",
  "penalty_root_row_count", "H_root_matrix_count", "H_root_rank",
  "rank_reference_materialize_call_count",
  "rank_reference_materialize_elapsed_ms", "setup_shadow_d2h_count",
  "setup_shadow_d2h_bytes", "coefficient_output_capacity",
  "prepared_generation", "output_slot_state_before_solve",
  "output_slot_state_after_solve", "output_slot_state_after_release",
  "output_slot_leased_after_release", "output_slot_poison_reason_empty"
)
expected_runtime_names <- c(
  "stage", "device_id", "gpu_name", "runtime_context_create_count",
  "cuda_device_allocation_count", "cuda_host_allocation_count",
  "stream_create_count", "event_create_count", "cublas_handle_create_count",
  "cusolver_handle_create_count", "workspace_grow_count",
  "cuda_device_synchronize_count", "compute_capability_major",
  "compute_capability_minor", "sm_count", "cuda_toolkit_version",
  "cuda_driver_version", "cusolver_deterministic_mode", "cublas_math_mode",
  "cublas_atomics_mode", "cublas_user_workspace_installed",
  "cublas_workspace_alignment", "workspace_reserve_count", "freed",
  "creator_pid", "generation", "gesvdj_info_create_count",
  "gesvdj_info_destroy_count", "stable_workspace_grow_count",
  "cholesky_factor_checkpoint_record_count",
  "cholesky_factor_checkpoint_wait_count",
  "cholesky_solve_checkpoint_record_count",
  "cholesky_solve_checkpoint_wait_count", "qr_checkpoint_record_count",
  "qr_checkpoint_wait_count", "svd_checkpoint_record_count",
  "svd_checkpoint_wait_count", "workspace_bytes", "cublas_workspace_bytes",
  "eigen_workspace_bytes", "qr_workspace_bytes", "svd_workspace_bytes",
  "augmented_workspace_bytes", "aggregate_factor_workspace_bytes"
)
independent_schema_types <- function(
    names, character = character(), integer = character(),
    double = character(), logical = character(), list = character()) {
  types <- setNames(rep.int(NA_character_, length(names)), names)
  for (type in c("character", "integer", "double", "logical", "list")) {
    types[get(type, inherits = FALSE)] <- type
  }
  assert_true(!anyNA(types), "independent frozen schema covers every column")
  types
}
expected_target_rds_types <- independent_schema_types(
  expected_target_rds_names,
  character = c(
    "prepared_s_key_sha256", "residual_key_sha256", "condition_bucket",
    "planned_route", "authenticated_planned_route", "executed_route",
    "reroute_reason", "solver_status", "numeric_reference",
    "fitted_numeric_hash", "residual_numeric_hash", "oracle_fitted_hash",
    "oracle_residual_hash", "authenticated_fitted_hash",
    "authenticated_residual_hash"
  ),
  integer = c(
    "batch_ordinal", "target_ordinal", "target", "null_dim",
    "phase1_coefficient_rank", "qr_rank", "geqrf_info", "ormqr_info",
    "effective_rank", "svd_info", "aggregate_penalty_root_rank",
    "aggregate_factor_call_count", "aggregate_b_build_count",
    "cpu_aggregate_penalty_root_rank", "cpu_aggregate_effective_rank",
    "oracle_call_count"
  ),
  double = c(
    "phase1_condition", "sigma_max", "smallest_retained_sigma",
    "aggregate_dstop", "cpu_aggregate_effective_rank_threshold",
    "cpu_aggregate_sigma_max", "residual_max_abs_diff",
    "residual_relative_l2_diff", "fitted_max_abs_diff",
    "fitted_relative_l2_diff"
  ),
  logical = c(
    "target_true_batched", "aggregate_penalty_root_rank_exact",
    "aggregate_penalty_root_pivot_exact", "aggregate_effective_rank_exact",
    "outputs_all_finite", "oracle_fitted_hash_exact",
    "oracle_residual_hash_exact", "approximate_backend"
  ),
  list = c(
    "aggregate_penalty_root_pivot", "cpu_aggregate_penalty_root_pivot"
  )
)
expected_batch_logical <- c(
  "native_batch_call", "true_batched_kernel", "full_cuda_data_plane",
  "canonical_output_order_exact", "target_keys_exact",
  "route_status_conservation_exact", "resource_snapshot_captured",
  "output_slot_leased_after_release"
)
expected_batch_double <- c(
  "aggregate_penalty_root_d2h_bytes", "target_h2d_bytes",
  "solve_elapsed_ms", "pre_shadow_d2h_bytes",
  "shadow_materialize_elapsed_ms", "post_shadow_d2h_bytes"
)
expected_batch_rds_types <- independent_schema_types(
  expected_batch_rds_names,
  character = "prepared_s_key_sha256",
  integer = setdiff(
    expected_batch_rds_names,
    c("prepared_s_key_sha256", expected_batch_logical, expected_batch_double)
  ),
  double = expected_batch_double,
  logical = expected_batch_logical
)
expected_setup_character <- c(
  "prepared_s_key_sha256", "output_slot_state_before_solve",
  "output_slot_state_after_solve", "output_slot_state_after_release"
)
expected_setup_logical <- c(
  "output_slot_leased_after_release", "output_slot_poison_reason_empty"
)
expected_setup_double <- c(
  "setup_h2d_bytes", "penalty_root_bytes", "penalty_root_build_ms",
  "rank_reference_materialize_elapsed_ms", "setup_shadow_d2h_bytes",
  "coefficient_output_capacity", "prepared_generation"
)
expected_setup_rds_types <- independent_schema_types(
  expected_setup_rds_names,
  character = expected_setup_character,
  integer = setdiff(
    expected_setup_rds_names,
    c(expected_setup_character, expected_setup_logical, expected_setup_double)
  ),
  double = expected_setup_double,
  logical = expected_setup_logical
)
expected_runtime_character <- c(
  "stage", "gpu_name", "cusolver_deterministic_mode", "cublas_math_mode",
  "cublas_atomics_mode"
)
expected_runtime_logical <- c("cublas_user_workspace_installed", "freed")
expected_runtime_double <- c(
  "cublas_workspace_alignment", "creator_pid", "generation",
  "workspace_bytes", "cublas_workspace_bytes", "eigen_workspace_bytes",
  "qr_workspace_bytes", "svd_workspace_bytes", "augmented_workspace_bytes",
  "aggregate_factor_workspace_bytes"
)
expected_runtime_types <- independent_schema_types(
  expected_runtime_names,
  character = expected_runtime_character,
  integer = setdiff(
    expected_runtime_names,
    c(expected_runtime_character, expected_runtime_logical,
      expected_runtime_double)
  ),
  double = expected_runtime_double,
  logical = expected_runtime_logical
)
assert_independent_frame_schema <- function(
    frame, expected_names, expected_types, expected_rows,
    list_fields = character()) {
  assert_true(
    is.data.frame(frame) && nrow(frame) == expected_rows &&
      identical(names(frame), expected_names),
    "independent artifact frame names/order and row count are exact"
  )
  for (field in expected_names) {
    value <- frame[[field]]
    if (field %in% list_fields) {
      assert_true(
        typeof(value) == "list" && length(value) == expected_rows &&
          is.object(value) && identical(attributes(value), list(class = "AsIs")) &&
          all(vapply(value, function(element) {
            typeof(element) == "integer" && !is.object(element) &&
              is.null(attributes(element)) && !anyNA(element)
          }, logical(1L))),
        paste("independent artifact list column schema is exact", field)
      )
    } else {
      assert_bare_vector(
        value, expected_types[[field]], expected_rows,
        paste("independent artifact column schema is exact", field),
        allow_na = TRUE
      )
    }
  }
  invisible(TRUE)
}
assert_independent_frame_schema(
  schema_fixture$target_records, expected_target_rds_names,
  expected_target_rds_types, 6143L,
  c("aggregate_penalty_root_pivot", "cpu_aggregate_penalty_root_pivot")
)
assert_independent_frame_schema(
  schema_fixture$batch_records, expected_batch_rds_names,
  expected_batch_rds_types, 2061L
)
assert_independent_frame_schema(
  schema_fixture$setup_records, expected_setup_rds_names,
  expected_setup_rds_types, 2061L
)
assert_independent_frame_schema(
  schema_fixture$runtime_records, expected_runtime_names,
  expected_runtime_types, 3L
)

schema_target_csv <- utils::read.csv(
  file.path(schema_fixture_dir, "target_parity.csv"),
  stringsAsFactors = FALSE, check.names = FALSE
)
schema_batch_csv <- utils::read.csv(
  file.path(schema_fixture_dir, "batch_metrics.csv"),
  stringsAsFactors = FALSE, check.names = FALSE
)
schema_setup_csv <- utils::read.csv(
  file.path(schema_fixture_dir, "setup_metrics.csv"),
  stringsAsFactors = FALSE, check.names = FALSE
)
schema_fallbacks <- utils::read.csv(
  file.path(schema_fixture_dir, "fallbacks.csv"),
  stringsAsFactors = FALSE, check.names = FALSE
)
schema_failures <- utils::read.csv(
  file.path(schema_fixture_dir, "failures.csv"),
  stringsAsFactors = FALSE, check.names = FALSE
)
assert_true(
  identical(names(schema_target_csv), expected_target_rds_names) &&
    nrow(schema_target_csv) == 6143L &&
    identical(names(schema_batch_csv), expected_batch_rds_names) &&
    nrow(schema_batch_csv) == 2061L &&
    identical(names(schema_setup_csv), expected_setup_rds_names) &&
    nrow(schema_setup_csv) == 2061L &&
    identical(
      schema_target_csv$prepared_s_key_sha256,
      schema_fixture$target_records$prepared_s_key_sha256
    ) && identical(
      schema_target_csv$residual_key_sha256,
      schema_fixture$target_records$residual_key_sha256
    ) && identical(
      schema_batch_csv$prepared_s_key_sha256,
      schema_fixture$batch_records$prepared_s_key_sha256
    ) && identical(
      schema_setup_csv$prepared_s_key_sha256,
      schema_fixture$setup_records$prepared_s_key_sha256
    ),
  "qualification CSV headers, row counts, and canonical key order are frozen"
)
for (field in c(
  "aggregate_penalty_root_pivot", "cpu_aggregate_penalty_root_pivot"
)) {
  expected_projection <- vapply(
    schema_fixture$target_records[[field]],
    function(value) if (length(value) == 0L) "" else paste(value, collapse = ";"),
    character(1L)
  )
  assert_identical(
    schema_target_csv[[field]], expected_projection,
    paste("qualification CSV list projection is deterministic", field)
  )
}
assert_true(
  identical(
    names(schema_fallbacks),
    c("residual_key_sha256", "fallback_type", "reason")
  ) && nrow(schema_fallbacks) == 0L && identical(
    names(schema_failures),
    c("stage", "prepared_s_key_sha256", "residual_key_sha256",
      "error_class", "error_message")
  ) && nrow(schema_failures) == 0L,
  "fallback and failure CSV schemas are exact and empty"
)
assert_identical(
  fastkpc_full_cuda_fixed_sp_qualification_validate_result_schema(
    target_records = schema_fixture$target_records,
    batch_records = schema_fixture$batch_records,
    setup_records = schema_fixture$setup_records,
    runtime_records = schema_fixture$runtime_records
  ),
  TRUE,
  "qualification production result schema accepts the frozen artifact"
)
target_order_drift <- schema_fixture$target_records
target_order_drift <- target_order_drift[c(2L, 1L, seq.int(3L, ncol(
  target_order_drift
)))]
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_qualification_validate_result_schema(
    target_records = target_order_drift,
    batch_records = schema_fixture$batch_records,
    setup_records = schema_fixture$setup_records,
    runtime_records = schema_fixture$runtime_records
  ),
  "qualification result schema mismatch",
  "qualification result schema rejects target column-order drift"
)
target_type_drift <- schema_fixture$target_records
target_type_drift$qr_rank <- as.double(target_type_drift$qr_rank)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_qualification_validate_result_schema(
    target_records = target_type_drift,
    batch_records = schema_fixture$batch_records,
    setup_records = schema_fixture$setup_records,
    runtime_records = schema_fixture$runtime_records
  ),
  "qualification result schema mismatch",
  "qualification result schema rejects omitted route-field type drift"
)

provenance_fixture_dir <- tempfile(
  "fastkpc_phase3c_provenance_", tmpdir = "fastkpc/tests"
)
dir.create(provenance_fixture_dir)
provenance_source_paths <- file.path(
  provenance_fixture_dir, c("runner.R", "direct.R", "transitive.R")
)
names(provenance_source_paths) <- c("runner", "direct", "transitive")
writeLines(
  paste0('source("', provenance_source_paths[["direct"]], '")'),
  provenance_source_paths[["runner"]], useBytes = TRUE
)
writeLines(
  paste0('source("', provenance_source_paths[["transitive"]], '")'),
  provenance_source_paths[["direct"]], useBytes = TRUE
)
writeLines(
  "fixture_transitive_numeric <- function(x) x + 1",
  provenance_source_paths[["transitive"]], useBytes = TRUE
)
provenance_closure <-
  fastkpc_full_cuda_fixed_sp_discover_execution_source_closure(
    root_sources = c(qualification_runner = provenance_source_paths[["runner"]]),
    project_root = "."
  )
expected_provenance_source_ids <- sort(
  unname(provenance_source_paths), method = "radix"
)
assert_true(
  identical(
    provenance_closure$source_closure_schema_version,
    "full-cuda-ci-execution-source-closure-v1"
  ) && identical(provenance_closure$source_closure_count, 3L) &&
    identical(provenance_closure$source_ids, expected_provenance_source_ids) &&
    identical(names(provenance_closure$source_file_paths),
              expected_provenance_source_ids),
  "execution source discovery closes transitive literal sources in radix order"
)
provenance_native_dir <- file.path(
  provenance_fixture_dir, ".qualified-native-fixture"
)
dir.create(provenance_native_dir)
provenance_native_path <- file.path(
  provenance_native_dir, "fastkpc_cuda.so"
)
writeBin(charToRaw("fixture-native-library"), provenance_native_path)
provenance_native_path <- normalizePath(
  provenance_native_path, winslash = "/", mustWork = TRUE
)
provenance_native_build_input_paths <- file.path(
  provenance_fixture_dir, c("build.sh", "native.cu")
)
names(provenance_native_build_input_paths) <- c("build", "native")
writeLines("#!/bin/sh\nexit 0", provenance_native_build_input_paths[["build"]],
           useBytes = TRUE)
writeLines("// fixture native source",
           provenance_native_build_input_paths[["native"]], useBytes = TRUE)
provenance_native_build_input_paths[] <- vapply(
  provenance_native_build_input_paths,
  normalizePath, character(1L), winslash = "/", mustWork = TRUE
)
provenance_preload_hashes <- vapply(
  provenance_closure$source_file_paths,
  fastkpc_full_cuda_fixed_sp_sha256_file,
  character(1L)
)
provenance_native_build_input_hashes <- vapply(
  provenance_native_build_input_paths,
  fastkpc_full_cuda_fixed_sp_sha256_file,
  character(1L)
)
provenance_trace_path <- file.path(provenance_fixture_dir, "build.strace")
provenance_strace_path <- .fastkpc_cuda_resolve_strace(Sys.which("strace"))
provenance_trace_invocation <-
  .fastkpc_cuda_trace_invocation(provenance_strace_path)
provenance_external_dependency_path <- file.path(
  provenance_fixture_dir, "external-header.hpp"
)
writeLines(
  "// external fixture header", provenance_external_dependency_path,
  useBytes = TRUE
)
provenance_external_dependency_path <- normalizePath(
  provenance_external_dependency_path, winslash = "/", mustWork = TRUE
)
writeLines(c(
  paste0(
    "100 execve(\"", provenance_native_build_input_paths[["build"]],
    "\", [\"build.sh\"], 0x0 /* 0 vars */) = 0"
  ),
  paste0(
    "100 openat(AT_FDCWD, \"",
    provenance_native_build_input_paths[["native"]],
    "\", O_RDONLY|O_CLOEXEC) = 3<",
    provenance_native_build_input_paths[["native"]], ">"
  ),
  paste0(
    "100 openat(AT_FDCWD, \"", provenance_external_dependency_path,
    "\", O_RDONLY|O_CLOEXEC) = 4<",
    provenance_external_dependency_path, ">"
  )
), provenance_trace_path, useBytes = TRUE)
provenance_native_build_dependencies <-
  fastkpc_full_cuda_fixed_sp_capture_native_build_dependencies(
    trace_path = provenance_trace_path,
    build_working_dir = ".",
    tracer_path = provenance_strace_path,
    trace_invocation = provenance_trace_invocation
  )
provenance_native_sha256 <-
  fastkpc_full_cuda_fixed_sp_sha256_file(provenance_native_path)
provenance_native_identity <-
  .fastkpc_cuda_posix_file_identity(provenance_native_path)
provenance_loaded_paths <- function() provenance_native_path
provenance_mapped_records <- function() data.frame(
  path = provenance_native_path,
  live_path = provenance_native_path,
  deleted = FALSE,
  device_major_hex = provenance_native_identity$device_major_hex,
  device_minor_hex = provenance_native_identity$device_minor_hex,
  inode = provenance_native_identity$inode,
  stringsAsFactors = FALSE
)
with_provenance_symbol_binding <- function(expression) {
  existed <- exists(
    "getNativeSymbolInfo", envir = .GlobalEnv, inherits = FALSE
  )
  if (existed) {
    original <- get(
      "getNativeSymbolInfo", envir = .GlobalEnv, inherits = FALSE
    )
  }
  assign(
    "getNativeSymbolInfo",
    function(name, PACKAGE, withRegistrationInfo = FALSE) {
      list(name = name, dll = list(path = provenance_native_path))
    },
    envir = .GlobalEnv
  )
  on.exit({
    if (existed) {
      assign("getNativeSymbolInfo", original, envir = .GlobalEnv)
    } else {
      rm("getNativeSymbolInfo", envir = .GlobalEnv)
    }
  }, add = TRUE)
  force(expression)
}
provenance_fixture <- with_provenance_symbol_binding(
  fastkpc_full_cuda_fixed_sp_capture_execution_provenance(
    source_closure = provenance_closure,
    expected_source_sha256 = provenance_preload_hashes,
    native_library_path = provenance_native_path,
    native_build_input_paths = provenance_native_build_input_paths,
    expected_native_build_input_sha256 =
      provenance_native_build_input_hashes,
    native_build_dependencies = provenance_native_build_dependencies,
    expected_native_library_sha256 = provenance_native_sha256,
    loaded_paths = provenance_loaded_paths,
    mapped_records = provenance_mapped_records
  )
)
assert_true(
  identical(provenance_fixture$source_closure_count, 3L) &&
    identical(names(provenance_fixture$source_file_sha256),
              expected_provenance_source_ids) &&
    identical(names(provenance_fixture$source_file_git_state),
              expected_provenance_source_ids) &&
    identical(
      provenance_fixture$native_build_input_paths,
      provenance_native_build_input_paths
    ) && identical(
      provenance_fixture$native_build_input_sha256,
      provenance_native_build_input_hashes
    ) && identical(
      names(provenance_fixture$native_build_input_git_state),
      names(provenance_native_build_input_paths)
    ) && identical(
      provenance_fixture$native_build_dependencies,
      provenance_native_build_dependencies
    ) && identical(
      provenance_fixture$native_library_sha256,
      provenance_native_sha256
    ) && identical(
      provenance_fixture$provenance_schema_version,
      "full-cuda-ci-execution-source-snapshot-v6"
    ) && identical(
      provenance_fixture$native_library_identity,
      "qualified-pinned-inode-sha-exact-registered-mapped-path-v3"
    ) && identical(
    provenance_fixture$provenance_mode,
    "working-tree-execution-snapshot-v1"
  ) && isTRUE(provenance_fixture$relevant_sources_dirty_or_untracked) &&
    !isTRUE(provenance_fixture$execution_sources_unchanged_after_run),
  "execution provenance captures fixed-order dirty source identity"
)
verify_provenance_fixture <- function(value) {
  with_provenance_symbol_binding(
    fastkpc_full_cuda_fixed_sp_verify_execution_provenance(
      value, loaded_paths = provenance_loaded_paths,
      mapped_records = provenance_mapped_records
    )
  )
}
verified_provenance_fixture <-
  verify_provenance_fixture(provenance_fixture)
assert_true(
  isTRUE(verified_provenance_fixture$execution_sources_unchanged_after_run),
  "execution provenance verifies unchanged source/native bytes"
)
outside_source_path <- file.path(provenance_fixture_dir, "outside.R")
writeLines("outside_value <- 1L", outside_source_path, useBytes = TRUE)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_guarded_source(
    file = outside_source_path, source_closure = provenance_closure
  ),
  "escaped the authenticated execution closure",
  "runtime source guard rejects an un-prehashed lazy dependency"
)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_discover_execution_source_closure(
    root_sources = c(
      first = provenance_source_paths[["runner"]],
      duplicate = provenance_source_paths[["runner"]]
    ),
    project_root = "."
  ),
  "root identity is ambiguous",
  "execution source discovery rejects duplicate root identity ambiguity"
)
missing_source_path <- file.path(provenance_fixture_dir, "missing-parent.R")
writeLines(
  paste0('source("', file.path(provenance_fixture_dir, "missing.R"), '")'),
  missing_source_path, useBytes = TRUE
)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_discover_execution_source_closure(
    root_sources = c(missing = missing_source_path), project_root = "."
  ),
  "does not exist",
  "execution source discovery rejects missing literal dependencies"
)
writeLines(
  "fixture_transitive_numeric <- function(x) x + 2",
  provenance_source_paths[["transitive"]],
  useBytes = TRUE, sep = "\n"
)
assert_error_matching(
  verify_provenance_fixture(provenance_fixture),
  "execution source snapshot changed",
  "execution provenance fails closed on source mutation"
)
writeLines(
  "fixture_transitive_numeric <- function(x) x + 1",
  provenance_source_paths[["transitive"]], useBytes = TRUE
)
writeLines(
  "// mutated fixture native source",
  provenance_native_build_input_paths[["native"]], useBytes = TRUE
)
assert_error_matching(
  verify_provenance_fixture(provenance_fixture),
  "native build input snapshot changed",
  "execution provenance fails closed on native build input mutation"
)
writeLines(
  "// fixture native source",
  provenance_native_build_input_paths[["native"]], useBytes = TRUE
)
writeLines(
  "// external dependency mutation",
  provenance_external_dependency_path, useBytes = TRUE
)
assert_error_matching(
  verify_provenance_fixture(provenance_fixture),
  "native build dependency changed",
  "execution provenance fails closed on traced dependency mutation"
)

dynamic_source_path <- file.path(provenance_fixture_dir, "dynamic.R")
writeLines(c(
  paste0('dependency <- "', provenance_source_paths[["transitive"]], '"'),
  "source(dependency)"
), dynamic_source_path, useBytes = TRUE)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_discover_execution_source_closure(
    root_sources = c(dynamic = dynamic_source_path), project_root = "."
  ),
  "dynamic source()",
  "execution source discovery rejects dynamic source calls"
)

cycle_a_path <- file.path(provenance_fixture_dir, "cycle_a.R")
cycle_b_path <- file.path(provenance_fixture_dir, "cycle_b.R")
writeLines(
  paste0('source("', cycle_b_path, '")'), cycle_a_path, useBytes = TRUE
)
writeLines(
  paste0('source("', cycle_a_path, '")'), cycle_b_path, useBytes = TRUE
)
assert_error_matching(
  fastkpc_full_cuda_fixed_sp_discover_execution_source_closure(
    root_sources = c(cycle = cycle_a_path), project_root = "."
  ),
  "execution source cycle",
  "execution source discovery rejects load-time cycles"
)
unlink(provenance_fixture_dir, recursive = TRUE, force = TRUE)
assert_true(
  !dir.exists(provenance_fixture_dir),
  "execution provenance fixture is removed after the mutation gate"
)

qualification_runner_source <- paste(
  readLines(
    "fastkpc/tools/run_full_cuda_ci_fixed_sp_qualification.R",
    warn = FALSE
  ),
  collapse = "\n"
)
assert_true(
  grepl(
    "load_fastkpc_cuda_native_qualified(",
    qualification_runner_source, fixed = TRUE
  ) && grepl(
    "capture_native_build_dependencies(",
    qualification_runner_source, fixed = TRUE
  ),
  "qualification runner traces the clean build and certifies exact bytes"
)
assert_true(
  all(c(
    "native_build_dependencies.csv", "native_build_exclusions.csv",
    "native_build_trace.txt"
  ) %in% fastkpc_full_cuda_fixed_sp_qualification_payload_names()),
  "qualification payload publishes trace, dependency, and exclusion evidence"
)
qualification_test_lines <- readLines(
  "fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3c_qualification.R",
  warn = FALSE
)
independent_parser_start <- grep(
  "^independent_trace_tables <- function", qualification_test_lines
)
independent_parser_end <- grep(
  "^reparsed_native_build <-", qualification_test_lines
)
independent_parser_source <- if (
    length(independent_parser_start) == 1L &&
      length(independent_parser_end) == 1L &&
      independent_parser_start < independent_parser_end) {
  paste(
    qualification_test_lines[
      seq.int(independent_parser_start, independent_parser_end - 1L)
    ],
    collapse = "\n"
  )
} else {
  ""
}
assert_true(
  nzchar(independent_parser_source) && grepl(
    "independent_ordered_generated_paths(",
    independent_parser_source, fixed = TRUE
  ) && grepl(
    "independent_open_events(open_lines, open_syscalls)",
    independent_parser_source, fixed = TRUE
  ) && !grepl(
    "fastkpc_full_cuda_fixed_sp_open_flag_tokens",
    independent_parser_source, fixed = TRUE
  ) && !grepl("write_open <-", independent_parser_source, fixed = TRUE),
  paste0(
    "independent qualification reparse uses isolated open flags and ",
    "ordered generation evidence"
  )
)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP Phase 3C fixed-sp qualification gate\n")
  quit(save = "no", status = 0L)
}

runner_path <- file.path(
  "fastkpc", "tools", "run_full_cuda_ci_fixed_sp_qualification.R"
)
assert_true(file.exists(runner_path), "Phase 3C qualification runner is missing")
assert_true(requireNamespace("jsonlite", quietly = TRUE), "jsonlite is required")
assert_true(requireNamespace("digest", quietly = TRUE), "digest is required")

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
output_dir <- tempfile("fastkpc_phase3c_qualification_")
on.exit(unlink(output_dir, recursive = TRUE, force = TRUE), add = TRUE)

runner_output <- suppressWarnings(system2(
  "Rscript", runner_path,
  stdout = TRUE, stderr = TRUE,
  env = c(
    "FASTKPC_RUN_CUDA_TESTS=1",
    "FASTKPC_FULL_CUDA_PHASE3_DEVICE=0",
    paste0("FASTKPC_FULL_CUDA_PHASE3_PHASE0_DIR=", phase0_dir),
    paste0("FASTKPC_FULL_CUDA_PHASE3_PHASE1_DIR=", phase1_dir),
    paste0("FASTKPC_FULL_CUDA_PHASE3_PHASE2_DIR=", phase2_dir),
    paste0("FASTKPC_FULL_CUDA_PHASE3_DATA=", data_path),
    paste0("FASTKPC_FULL_CUDA_PHASE3_OUTPUT=", output_dir)
  )
))
runner_status <- attr(runner_output, "status")
if (is.null(runner_status)) runner_status <- 0L
if (runner_status != 0L) {
  fail(paste0(
    "Phase 3C qualification runner failed with status ", runner_status,
    ":\n", paste(runner_output, collapse = "\n")
  ))
}

expected_files <- c(
  "target_parity.rds", "target_parity.csv",
  "batch_metrics.rds", "batch_metrics.csv",
  "setup_metrics.rds", "setup_metrics.csv",
  "qualification_dcov_parity.rds", "qualification_dcov_parity.csv",
  "runtime_metrics.csv", "stage_timing.csv", "fallbacks.csv",
  "failures.csv", "native_build_dependencies.csv",
  "native_build_exclusions.csv", "native_build_trace.txt",
  "commands.txt", "environment.txt", "manifest.json", "summary.json"
)
actual_files <- sort(list.files(output_dir, all.files = FALSE))
assert_identical(
  actual_files, sort(expected_files),
  "qualification artifact has the exact published file surface"
)

target_records <- readRDS(file.path(output_dir, "target_parity.rds"))
batch_records <- readRDS(file.path(output_dir, "batch_metrics.rds"))
setup_records <- readRDS(file.path(output_dir, "setup_metrics.rds"))
dcov_records <- readRDS(file.path(
  output_dir, "qualification_dcov_parity.rds"
))
target_csv <- utils::read.csv(
  file.path(output_dir, "target_parity.csv"),
  stringsAsFactors = FALSE, check.names = FALSE
)
batch_csv <- utils::read.csv(
  file.path(output_dir, "batch_metrics.csv"),
  stringsAsFactors = FALSE, check.names = FALSE
)
setup_csv <- utils::read.csv(
  file.path(output_dir, "setup_metrics.csv"),
  stringsAsFactors = FALSE, check.names = FALSE
)
dcov_csv <- utils::read.csv(
  file.path(output_dir, "qualification_dcov_parity.csv"),
  stringsAsFactors = FALSE, check.names = FALSE
)
runtime_records <- utils::read.csv(
  file.path(output_dir, "runtime_metrics.csv"),
  stringsAsFactors = FALSE, check.names = FALSE
)
stage_timing <- utils::read.csv(
  file.path(output_dir, "stage_timing.csv"),
  stringsAsFactors = FALSE, check.names = FALSE
)
environment_lines <- readLines(
  file.path(output_dir, "environment.txt"), warn = FALSE
)
fallbacks <- utils::read.csv(
  file.path(output_dir, "fallbacks.csv"),
  stringsAsFactors = FALSE, check.names = FALSE
)
failures <- utils::read.csv(
  file.path(output_dir, "failures.csv"),
  stringsAsFactors = FALSE, check.names = FALSE
)
manifest <- jsonlite::fromJSON(
  file.path(output_dir, "manifest.json"), simplifyVector = FALSE
)
summary <- jsonlite::fromJSON(
  file.path(output_dir, "summary.json"), simplifyVector = FALSE
)

assert_true(
  is.data.frame(target_records) && nrow(target_records) == 6143L &&
    is.data.frame(batch_records) && nrow(batch_records) == 2061L &&
    is.data.frame(setup_records) && nrow(setup_records) == 2061L &&
    is.data.frame(dcov_records) && nrow(dcov_records) == 3808L &&
    is.data.frame(runtime_records) && nrow(runtime_records) == 3L &&
    identical(names(runtime_records), expected_runtime_names) &&
    identical(runtime_records$stage,
              c("runtime-created", "workspace-reserved", "final")),
  "qualification evidence row counts are exact"
)
assert_independent_frame_schema(
  target_records, expected_target_rds_names, expected_target_rds_types, 6143L,
  c("aggregate_penalty_root_pivot", "cpu_aggregate_penalty_root_pivot")
)
assert_independent_frame_schema(
  batch_records, expected_batch_rds_names, expected_batch_rds_types, 2061L
)
assert_independent_frame_schema(
  setup_records, expected_setup_rds_names, expected_setup_rds_types, 2061L
)
assert_true(
  identical(names(target_csv), expected_target_rds_names) &&
    nrow(target_csv) == 6143L &&
    identical(names(batch_csv), expected_batch_rds_names) &&
    nrow(batch_csv) == 2061L &&
    identical(names(setup_csv), expected_setup_rds_names) &&
    nrow(setup_csv) == 2061L &&
    identical(names(dcov_csv), names(dcov_records)) &&
    nrow(dcov_csv) == 3808L &&
    identical(target_csv$prepared_s_key_sha256,
              target_records$prepared_s_key_sha256) &&
    identical(target_csv$residual_key_sha256,
              target_records$residual_key_sha256) &&
    identical(batch_csv$prepared_s_key_sha256,
              batch_records$prepared_s_key_sha256) &&
    identical(setup_csv$prepared_s_key_sha256,
              setup_records$prepared_s_key_sha256),
  "published CSV headers, row counts, and canonical key order are exact"
)
for (field in c(
  "aggregate_penalty_root_pivot", "cpu_aggregate_penalty_root_pivot"
)) {
  expected_projection <- vapply(
    target_records[[field]],
    function(value) if (length(value) == 0L) "" else paste(value, collapse = ";"),
    character(1L)
  )
  assert_identical(
    target_csv[[field]], expected_projection,
    paste("published CSV list projection is deterministic", field)
  )
}
assert_true(
  identical(
    names(fallbacks), c("residual_key_sha256", "fallback_type", "reason")
  ) && nrow(fallbacks) == 0L && identical(
    names(failures),
    c("stage", "prepared_s_key_sha256", "residual_key_sha256",
      "error_class", "error_message")
  ) && nrow(failures) == 0L,
  "qualification publishes no fallback or failure rows"
)
assert_true(
  is.data.frame(stage_timing) && nrow(stage_timing) == 5L &&
    identical(names(stage_timing), c("stage", "elapsed_seconds")) &&
    identical(stage_timing$stage, c(
      "cuda_initialize", "capture_execution_provenance",
      "qualification_numeric_lifecycle", "qualification_dcov_parity",
      "verify_execution_provenance"
    )) && typeof(stage_timing$stage) == "character" &&
    is.null(attributes(stage_timing$stage)) &&
    typeof(stage_timing$elapsed_seconds) == "double" &&
    is.null(attributes(stage_timing$elapsed_seconds)) &&
    all(is.finite(stage_timing$elapsed_seconds)) &&
    all(stage_timing$elapsed_seconds >= 0),
  "qualification stage timing is complete"
)
assert_true(
  !"pass" %in% names(manifest) && !"pass" %in% names(summary),
  "qualification gate rejects supplied pass booleans"
)
expected_summary_fields <- c(
  "scope", "scope_subset_hash", "ordered_setup_key_digest",
  "ordered_target_key_digest", "route_status_hash", "numeric_hash",
  "setup_count", "target_count", "penalty_root_matrix_count",
  "penalty_root_row_count", "H_root_matrix_count", "planned_cholesky_count",
  "planned_qr_count", "planned_svd_count", "executed_cholesky_count",
  "executed_qr_count", "executed_svd_count", "cholesky_to_svd_count",
  "qr_to_svd_count", "svd_finite_high_count", "svd_nonfinite_count",
  "all_safe_batch_count", "mixed_batch_count", "all_stable_batch_count",
  "true_batched_subgroup_count", "true_batched_target_count",
  "cholesky_single_target_count", "whole_batch_true_batched_count",
  "stable_not_implemented_count", "stable_reroute_count",
  "non_ok_status_count", "root_rank_mismatch_count",
  "aggregate_penalty_factor_count", "aggregate_svd_b_build_count",
  "aggregate_penalty_root_rank_mismatch_count",
  "aggregate_penalty_root_pivot_mismatch_count",
  "aggregate_penalty_root_d2h_count", "aggregate_penalty_root_d2h_bytes",
  "workspace_grow_count_after_warmup",
  "stable_workspace_grow_count_after_warmup",
  "per_target_allocation_count_after_warmup",
  "per_target_handle_create_count", "cuda_device_synchronize_count",
  "target_level_stable_sync_count", "implicit_residual_d2h_count",
  "all_output_slot_leases_released", "invalid_output_init_count",
  "cpu_fallback_count", "unknown_fallback_count", "approximate_backend_count",
  "qr_checkpoint_record_count", "qr_checkpoint_wait_count",
  "svd_checkpoint_record_count", "svd_checkpoint_wait_count",
  "max_residual_abs_diff", "max_residual_relative_l2_diff",
  "max_fitted_abs_diff", "max_fitted_relative_l2_diff",
  "qualification_subset_hash", expected_dcov_summary_fields,
  "artifact_schema_version",
  "catalog_authenticated", "provenance_mode", "head_base_commit",
  "source_closure_schema_version", "source_closure_count",
  "source_closure_sha256", "execution_snapshot_sha256",
  "relevant_sources_dirty_or_untracked", "native_build_inputs_sha256",
  "native_build_dependency_count", "native_build_dependencies_sha256",
  "native_build_trace_sha256", "native_build_tracer_sha256",
  "native_library_sha256",
  "execution_sources_unchanged_after_run", "elapsed_seconds",
  "stage_timing_total_seconds", "payload_file_count"
)
assert_identical(
  names(summary), expected_summary_fields,
  "summary field order is frozen"
)
expected_manifest_fields <- c(
  "schema_version", "scope", "catalog_authenticated",
  "provenance_schema_version", "provenance_mode", "head_base_commit",
  "source_closure_schema_version", "source_discovery_semantics",
  "source_project_root", "source_closure_count", "source_closure_sha256",
  "direct_source_ids", "source_dependency_map",
  "source_file_paths", "source_file_sha256", "source_file_git_state",
  "native_build_input_paths", "native_build_input_sha256",
  "native_build_input_git_state", "native_build_inputs_sha256",
  "native_build_dependencies_schema_version",
  "native_build_dependency_trace_semantics",
  "native_build_dependency_trace_invocation", "native_build_working_dir",
  "native_build_trace_path",
  "native_build_trace_sha256", "native_build_tracer_path",
  "native_build_tracer_sha256", "native_build_dependency_count",
  "native_build_exclusion_count", "native_build_dependencies_sha256",
  "relevant_sources_dirty_or_untracked", "native_library_identity",
  "native_library_path", "native_library_sha256",
  "execution_snapshot_sha256", "execution_sources_unchanged_after_run",
  "device_id", "phase0_dir", "phase1_dir", "phase2_dir", "data_path",
  "qualification_subset_hash", "ordered_setup_key_digest",
  "ordered_target_key_digest", "route_status_hash", "numeric_hash",
  "qualification_logical_tests_sha256",
  "qualification_dcov_logical_ids_hash",
  "qualification_dcov_residual_key_hash",
  "qualification_dcov_rows_hash",
  "payload_file_sha256", "publication_order"
)
assert_identical(
  names(manifest), expected_manifest_fields,
  "manifest field order is frozen"
)
assert_json_scalar <- function(value, type, message) {
  assert_true(
    typeof(value) == type && length(value) == 1L && !is.object(value) &&
      is.null(attributes(value)) && !anyNA(value),
    message
  )
}
for (field in c(
  "schema_version", "scope", "provenance_schema_version",
  "provenance_mode", "head_base_commit", "source_closure_schema_version",
  "source_discovery_semantics", "source_project_root",
  "source_closure_sha256", "native_library_identity", "native_library_path",
  "native_build_inputs_sha256", "native_build_dependencies_schema_version",
  "native_build_dependency_trace_semantics",
  "native_build_dependency_trace_invocation", "native_build_working_dir",
  "native_build_trace_path",
  "native_build_trace_sha256", "native_build_tracer_path",
  "native_build_tracer_sha256", "native_build_dependencies_sha256",
  "native_library_sha256",
  "execution_snapshot_sha256", "phase0_dir", "phase1_dir", "phase2_dir",
  "data_path", "qualification_subset_hash", "ordered_setup_key_digest",
  "ordered_target_key_digest", "route_status_hash", "numeric_hash",
  "qualification_logical_tests_sha256",
  "qualification_dcov_logical_ids_hash",
  "qualification_dcov_residual_key_hash",
  "qualification_dcov_rows_hash"
)) {
  assert_json_scalar(
    manifest[[field]], "character",
    paste("manifest character scalar type is exact", field)
  )
}
for (field in c(
  "catalog_authenticated", "relevant_sources_dirty_or_untracked",
  "execution_sources_unchanged_after_run"
)) {
  assert_json_scalar(
    manifest[[field]], "logical",
    paste("manifest logical scalar type is exact", field)
  )
}
assert_json_scalar(
  manifest$device_id, "integer", "manifest device id is one exact integer"
)
assert_json_scalar(
  manifest$source_closure_count, "integer",
  "manifest source closure count is one exact integer"
)
assert_json_scalar(
  manifest$native_build_dependency_count, "integer",
  "manifest native build dependency count is one exact integer"
)
assert_json_scalar(
  manifest$native_build_exclusion_count, "integer",
  "manifest native build exclusion count is one exact integer"
)

expected_execution_source_ids <- c(
  "fastkpc/R/cuda_native.R",
  "fastkpc/R/full_cuda_ci_fixed_sp_runtime.R",
  "fastkpc/R/full_cuda_ci_gate.R",
  "fastkpc/R/full_cuda_ci_oracle_contract.R",
  "fastkpc/R/full_cuda_ci_prepared_s_contract.R",
  "fastkpc/R/full_cuda_ci_workload_census.R",
  "fastkpc/R/mgcv_compat_contract.R",
  "fastkpc/R/mgcv_extract_oracle.R",
  "fastkpc/R/native.R",
  "fastkpc/tools/run_full_cuda_ci_fixed_sp_qualification.R"
)
expected_direct_source_ids <- c(
  qualification_runner = runner_path,
  workload_census = "fastkpc/R/full_cuda_ci_workload_census.R",
  prepared_s_contract = "fastkpc/R/full_cuda_ci_prepared_s_contract.R",
  fixed_sp_runtime = "fastkpc/R/full_cuda_ci_fixed_sp_runtime.R",
  cuda_native = "fastkpc/R/cuda_native.R"
)
expected_source_dependency_map <- list(
  "fastkpc/R/cuda_native.R" = character(),
  "fastkpc/R/full_cuda_ci_fixed_sp_runtime.R" = character(),
  "fastkpc/R/full_cuda_ci_gate.R" = character(),
  "fastkpc/R/full_cuda_ci_oracle_contract.R" =
    "fastkpc/R/full_cuda_ci_gate.R",
  "fastkpc/R/full_cuda_ci_prepared_s_contract.R" = c(
    "fastkpc/R/full_cuda_ci_workload_census.R",
    "fastkpc/R/mgcv_compat_contract.R",
    "fastkpc/R/mgcv_extract_oracle.R",
    "fastkpc/R/native.R"
  ),
  "fastkpc/R/full_cuda_ci_workload_census.R" = c(
    "fastkpc/R/full_cuda_ci_oracle_contract.R",
    "fastkpc/R/mgcv_compat_contract.R"
  ),
  "fastkpc/R/mgcv_compat_contract.R" = character(),
  "fastkpc/R/mgcv_extract_oracle.R" =
    "fastkpc/R/mgcv_compat_contract.R",
  "fastkpc/R/native.R" = character(),
  "fastkpc/tools/run_full_cuda_ci_fixed_sp_qualification.R" = c(
    "fastkpc/R/cuda_native.R",
    "fastkpc/R/full_cuda_ci_fixed_sp_runtime.R",
    "fastkpc/R/full_cuda_ci_prepared_s_contract.R",
    "fastkpc/R/full_cuda_ci_workload_census.R"
  )
)
expected_execution_source_paths <- setNames(vapply(
  expected_execution_source_ids,
  function(path) normalizePath(path, winslash = "/", mustWork = TRUE),
  character(1L)
), expected_execution_source_ids)
expected_native_build_input_ids <- sort(c(
  "fastkpc/tools/build_cuda_native.sh",
  list.files(
    "fastkpc/src", pattern = "\\.(c|cc|cpp|cu|cuh|h|hpp)$",
    recursive = TRUE, full.names = TRUE
  )
), method = "radix")
expected_native_build_input_paths <- setNames(vapply(
  expected_native_build_input_ids,
  normalizePath, character(1L), winslash = "/", mustWork = TRUE
), expected_native_build_input_ids)
assert_json_character_map <- function(value, expected_names, message) {
  assert_true(
    is.list(value) && !is.object(value) &&
      identical(names(value), expected_names) &&
      all(vapply(value, function(element) {
        typeof(element) == "character" && length(element) == 1L &&
          !is.object(element) && is.null(attributes(element)) &&
          !anyNA(element) && nzchar(element)
      }, logical(1L))),
    message
  )
}
assert_json_character_map(
  manifest$direct_source_ids, names(expected_direct_source_ids),
  "manifest direct-source role map has fixed names and scalar types"
)
assert_json_character_map(
  manifest$source_file_paths, expected_execution_source_ids,
  "manifest source path map has fixed names and exact scalar types"
)
assert_json_character_map(
  manifest$source_file_sha256, expected_execution_source_ids,
  "manifest source hash map has fixed names and exact scalar types"
)
assert_json_character_map(
  manifest$source_file_git_state, expected_execution_source_ids,
  "manifest source git-state map has fixed names and exact scalar types"
)
assert_json_character_map(
  manifest$native_build_input_paths, expected_native_build_input_ids,
  "manifest native build-input path map has fixed names and scalar types"
)
assert_json_character_map(
  manifest$native_build_input_sha256, expected_native_build_input_ids,
  "manifest native build-input hash map has fixed names and scalar types"
)
assert_json_character_map(
  manifest$native_build_input_git_state, expected_native_build_input_ids,
  "manifest native build-input git-state map has fixed names and scalar types"
)
manifest_direct_source_ids <- unlist(
  manifest$direct_source_ids, use.names = TRUE
)
manifest_source_dependencies <- lapply(
  manifest$source_dependency_map,
  function(value) as.character(unlist(value, use.names = FALSE))
)
assert_identical(
  manifest_direct_source_ids, expected_direct_source_ids,
  "manifest identifies the exact runner direct source roots"
)
assert_identical(
  manifest_source_dependencies, expected_source_dependency_map,
  "manifest records the complete canonical source dependency graph"
)
manifest_source_paths <- unlist(
  manifest$source_file_paths, use.names = TRUE
)
manifest_source_hashes <- unlist(
  manifest$source_file_sha256, use.names = TRUE
)
manifest_source_states <- unlist(
  manifest$source_file_git_state, use.names = TRUE
)
manifest_native_build_input_paths <- unlist(
  manifest$native_build_input_paths, use.names = TRUE
)
manifest_native_build_input_hashes <- unlist(
  manifest$native_build_input_sha256, use.names = TRUE
)
manifest_native_build_input_states <- unlist(
  manifest$native_build_input_git_state, use.names = TRUE
)
assert_identical(
  manifest_source_paths, expected_execution_source_paths,
  "manifest paths identify the exact execution R sources"
)
actual_execution_source_hashes <- vapply(
  manifest_source_paths, function(path) {
    digest::digest(file = path, algo = "sha256", serialize = FALSE)
  }, character(1L)
)
assert_identical(
  unname(manifest_source_hashes),
  unname(actual_execution_source_hashes),
  "execution R source hashes remain unchanged after the long run"
)
independent_git_state <- function(path) {
  status <- suppressWarnings(base::system2(
    "git",
    c("status", "--porcelain=v1", "--untracked-files=all", "--", path),
    stdout = TRUE, stderr = TRUE
  ))
  exit_status <- attr(status, "status")
  assert_true(
    is.null(exit_status) || exit_status == 0L,
    "independent source git-state command succeeds"
  )
  if (length(status) == 0L) return("clean")
  if (any(startsWith(status, "??"))) return("untracked")
  "tracked-dirty"
}
actual_execution_source_states <- vapply(
  expected_execution_source_paths,
  independent_git_state,
  character(1L)
)
assert_identical(
  unname(manifest_source_states),
  unname(actual_execution_source_states),
  "manifest records current dirty/untracked state per execution source"
)
assert_identical(
  manifest_native_build_input_paths, expected_native_build_input_paths,
  "manifest paths identify every native build input"
)
actual_native_build_input_hashes <- vapply(
  expected_native_build_input_paths,
  function(path) digest::digest(
    file = path, algo = "sha256", serialize = FALSE
  ),
  character(1L)
)
assert_identical(
  unname(manifest_native_build_input_hashes),
  unname(actual_native_build_input_hashes),
  "native build-input hashes remain unchanged after the long run"
)
actual_native_build_input_states <- vapply(
  expected_native_build_input_paths,
  independent_git_state,
  character(1L)
)
assert_identical(
  unname(manifest_native_build_input_states),
  unname(actual_native_build_input_states),
  "manifest records native build-input git state"
)
head_base_commit <- base::system2(
  "git", c("rev-parse", "HEAD"), stdout = TRUE, stderr = FALSE
)
assert_true(
  length(head_base_commit) == 1L &&
    grepl("^[0-9a-f]{40}$", head_base_commit) &&
    identical(manifest$head_base_commit, unname(head_base_commit)) &&
    !"source_commit" %in% names(manifest),
  "manifest distinguishes HEAD/base commit from working-tree execution"
)
assert_true(
  identical(
    manifest$provenance_schema_version,
    "full-cuda-ci-execution-source-snapshot-v6"
  ) && identical(
    manifest$provenance_mode,
    "working-tree-execution-snapshot-v1"
  ) && identical(
    manifest$source_closure_schema_version,
    "full-cuda-ci-execution-source-closure-v1"
  ) && identical(
    manifest$source_discovery_semantics,
    "parsed-r-ast-load-time-literal-source-v1"
  ) && identical(manifest$source_closure_count, 10L) && identical(
    manifest$source_project_root,
    normalizePath(".", winslash = "/", mustWork = TRUE)
  ) && identical(
    manifest$relevant_sources_dirty_or_untracked,
    any(c(
      actual_execution_source_states,
      actual_native_build_input_states
    ) != "clean")
  ) && isTRUE(manifest$execution_sources_unchanged_after_run),
  "manifest provenance mode and dirty-source gate are exact"
)
canonical_native_library_path <- file.path(
  normalizePath("fastkpc/build", winslash = "/", mustWork = TRUE),
  "fastkpc_cuda.so"
)
qualified_native_library_dir <- dirname(manifest$native_library_path)
assert_true(
  identical(
    manifest$native_library_identity,
    "qualified-pinned-inode-sha-exact-registered-mapped-path-v3"
  ) && is.character(manifest$native_library_path) &&
    length(manifest$native_library_path) == 1L &&
    !file.exists(manifest$native_library_path) &&
    !dir.exists(qualified_native_library_dir) &&
    identical(basename(manifest$native_library_path), "fastkpc_cuda.so") &&
    identical(
      dirname(dirname(manifest$native_library_path)),
      normalizePath("fastkpc/build", winslash = "/", mustWork = TRUE)
    ) && startsWith(
      basename(dirname(manifest$native_library_path)),
      ".fastkpc_cuda-qualified-"
    ) &&
    grepl("^[0-9a-f]{64}$", manifest$native_library_sha256) &&
    file.exists(canonical_native_library_path) &&
    identical(
      manifest$native_library_sha256,
      digest::digest(
        file = canonical_native_library_path, algo = "sha256",
        serialize = FALSE
      )
    ),
  paste0(
    "manifest records the execution-time pinned native path, post-exit ",
    "cleanup, and canonical published bytes"
  )
)
native_build_input_identity_lines <- unlist(lapply(
  expected_native_build_input_ids,
  function(input_id) c(
    paste0(
      "native_build_input.", input_id, ".path=",
      manifest_native_build_input_paths[[input_id]]
    ),
    paste0(
      "native_build_input.", input_id, ".sha256=",
      manifest_native_build_input_hashes[[input_id]]
    )
  )
), use.names = FALSE)
independent_native_build_inputs_sha256 <- unname(digest::digest(
  charToRaw(enc2utf8(paste0(
    paste(native_build_input_identity_lines, collapse = "\n"), "\n"
  ))),
  algo = "sha256", serialize = FALSE
))
assert_identical(
  manifest$native_build_inputs_sha256,
  independent_native_build_inputs_sha256,
  "manifest binds native source and build-script bytes as one identity"
)
native_build_dependencies <- utils::read.csv(
  file.path(output_dir, "native_build_dependencies.csv"),
  stringsAsFactors = FALSE, check.names = FALSE
)
native_build_exclusions <- utils::read.csv(
  file.path(output_dir, "native_build_exclusions.csv"),
  stringsAsFactors = FALSE, check.names = FALSE
)
assert_true(
  identical(names(native_build_dependencies), c("path", "sha256")) &&
    nrow(native_build_dependencies) == manifest$native_build_dependency_count &&
    nrow(native_build_dependencies) > 0L &&
    identical(
      native_build_dependencies$path,
      sort(native_build_dependencies$path, method = "radix")
    ) && !anyDuplicated(native_build_dependencies$path) &&
    all(fastkpc_full_cuda_fixed_sp_regular_file_mask(
      native_build_dependencies$path
    )) &&
    all(grepl("^[0-9a-f]{64}$", native_build_dependencies$sha256)),
  "native build dependency payload is canonical and complete"
)
assert_true(
  identical(names(native_build_exclusions), c("path", "reason")) &&
    nrow(native_build_exclusions) == manifest$native_build_exclusion_count &&
    nrow(native_build_exclusions) > 0L &&
    identical(
      native_build_exclusions$path,
      sort(native_build_exclusions$path, method = "radix")
    ) && !anyDuplicated(native_build_exclusions$path) &&
    all(native_build_exclusions$reason %in% c(
      "generated_output", "pseudo_fs", "non_regular"
    )) && !any(
      native_build_exclusions$path %in% native_build_dependencies$path
    ),
  "native build exclusion payload is canonical and complete"
)
trace_payload_path <- file.path(output_dir, manifest$native_build_trace_path)
assert_true(
  identical(manifest$native_build_trace_path, "native_build_trace.txt") &&
    file.exists(trace_payload_path) && !dir.exists(trace_payload_path) &&
    identical(
      digest::digest(
        file = trace_payload_path, algo = "sha256", serialize = FALSE
      ),
      manifest$native_build_trace_sha256
    ),
  "manifest binds the independently inspectable raw native build trace"
)
actual_native_build_dependency_hashes <- vapply(
  native_build_dependencies$path,
  function(path) digest::digest(
    file = path, algo = "sha256", serialize = FALSE
  ),
  character(1L)
)
assert_identical(
  unname(native_build_dependencies$sha256),
  unname(actual_native_build_dependency_hashes),
  "native build dependency bytes remain unchanged after qualification"
)
tracer_index <- match(
  manifest$native_build_tracer_path, native_build_dependencies$path
)
assert_true(
  identical(
    manifest$native_build_dependencies_schema_version,
    "full-cuda-ci-native-build-dependencies-v3"
  ) && identical(
    manifest$native_build_dependency_trace_semantics,
    "linux-strace-successful-read-exec-evidence-v3"
  ) && identical(
    manifest$native_build_working_dir,
    normalizePath(".", winslash = "/", mustWork = TRUE)
  ) && !is.na(tracer_index) && identical(
    native_build_dependencies$sha256[[tracer_index]],
    manifest$native_build_tracer_sha256
  ),
  "manifest records the traced dependency semantics and tracer identity"
)
dependency_identity_lines <- c(
  paste0(
    "schema_version=", manifest$native_build_dependencies_schema_version
  ),
  paste0(
    "trace_semantics=", manifest$native_build_dependency_trace_semantics
  ),
  paste0(
    "trace_invocation=", manifest$native_build_dependency_trace_invocation
  ),
  paste0("build_working_dir=", manifest$native_build_working_dir),
  paste0("trace.sha256=", manifest$native_build_trace_sha256),
  paste0("tracer.path=", manifest$native_build_tracer_path),
  paste0("tracer.sha256=", manifest$native_build_tracer_sha256),
  paste0("dependency_count=", manifest$native_build_dependency_count)
)
for (index in seq_len(nrow(native_build_dependencies))) {
  dependency_identity_lines <- c(
    dependency_identity_lines,
    paste0(
      "dependency.", index, ".path=",
      native_build_dependencies$path[[index]]
    ),
    paste0(
      "dependency.", index, ".sha256=",
      native_build_dependencies$sha256[[index]]
    )
  )
}
dependency_identity_lines <- c(
  dependency_identity_lines,
  paste0("exclusion_count=", manifest$native_build_exclusion_count)
)
for (index in seq_len(nrow(native_build_exclusions))) {
  dependency_identity_lines <- c(
    dependency_identity_lines,
    paste0(
      "exclusion.", index, ".path=",
      native_build_exclusions$path[[index]]
    ),
    paste0(
      "exclusion.", index, ".reason=",
      native_build_exclusions$reason[[index]]
    )
  )
}
independent_trace_tables <- function(trace_path, tracer_path) {
  lines <- readLines(trace_path, warn = FALSE, encoding = "bytes")
  call_pattern <- paste0(
    "^[[:space:]]*(?:[0-9]+|\\[pid[[:space:]]+[0-9]+\\])",
    "[[:space:]]+(open|openat|openat2|execve|execveat)\\("
  )
  traced <- grepl(call_pattern, lines, perl = TRUE)
  calls <- lines[traced]
  syscalls <- sub(paste0(call_pattern, ".*$"), "\\1", calls, perl = TRUE)
  succeeded <- grepl("[[:space:]]= (0|[1-9][0-9]*)", calls)
  failed <- grepl("[[:space:]]= -[0-9]+", calls)
  assert_true(
    length(calls) > 0L && all(succeeded | failed),
    "independent native build trace parser accepts every traced call"
  )
  calls <- calls[succeeded]
  syscalls <- syscalls[succeeded]
  quoted <- '"((?:[^"\\\\]|\\\\.)*)"'
  extract_group <- function(line, pattern) {
    match <- regmatches(line, regexec(pattern, line, perl = TRUE))[[1L]]
    assert_true(
      length(match) == 2L,
      "independent native build trace path extraction is complete"
    )
    fastkpc_full_cuda_fixed_sp_decode_strace_path(match[[2L]])
  }
  open_path <- function(line, syscall) {
    resolved <- regmatches(
      line,
      regexec("^.*[[:space:]]= [0-9]+<(/[^>]*)>[[:space:]]*$",
              line, perl = TRUE)
    )[[1L]]
    if (length(resolved) == 2L) {
      return(fastkpc_full_cuda_fixed_sp_decode_strace_path(resolved[[2L]]))
    }
    if (identical(syscall, "open")) {
      return(extract_group(line, paste0("open\\(", quoted)))
    }
    extract_group(
      line, paste0("(?:openat|openat2)\\([^,]+,[[:space:]]*", quoted)
    )
  }
  exec_path <- function(line, syscall) {
    if (identical(syscall, "execve")) {
      return(extract_group(line, paste0("execve\\(", quoted)))
    }
    extract_group(
      line, paste0("execveat\\([^,]+,[[:space:]]*", quoted)
    )
  }
  is_open <- syscalls %in% c("open", "openat", "openat2")
  open_lines <- calls[is_open]
  open_syscalls <- syscalls[is_open]
  open_paths <- unname(vapply(
    seq_along(open_lines),
    function(index) open_path(open_lines[[index]], open_syscalls[[index]]),
    character(1L)
  ))
  exec_lines <- calls[!is_open]
  exec_syscalls <- syscalls[!is_open]
  exec_paths <- unname(vapply(
    seq_along(exec_lines),
    function(index) exec_path(exec_lines[[index]], exec_syscalls[[index]]),
    character(1L)
  ))
  open_events <- independent_open_events(open_lines, open_syscalls)
  read_open <- open_events$read
  generation_open <- open_events$generation
  directory_open <- open_events$directory
  event_paths <- character(length(calls))
  event_paths[is_open] <- open_paths
  event_paths[!is_open] <- exec_paths
  access_event <- !is_open
  access_event[is_open] <- read_open
  generation_event <- rep.int(FALSE, length(calls))
  generation_event[is_open] <- generation_open
  directory_event <- rep.int(FALSE, length(calls))
  directory_event[is_open] <- directory_open
  generated_by_path <- independent_ordered_generated_paths(
    event_paths, access_event, generation_event
  )
  candidates <- names(generated_by_path)
  access_paths <- event_paths
  access_paths[!access_event] <- NA_character_
  first_access <- match(candidates, access_paths)
  assert_true(
    all(startsWith(event_paths, "/")),
    "independent native build trace paths are absolute"
  )
  reasons <- rep.int(NA_character_, length(candidates))
  reasons[unname(generated_by_path)] <- "generated_output"
  reasons[grepl(
    "^/(?:proc|sys|dev)(?:/|$)", candidates, perl = TRUE
  )] <- "pseudo_fs"
  reasons[is.na(reasons) & directory_event[first_access]] <- "non_regular"
  remaining <- which(is.na(reasons))
  exists <- file.exists(candidates[remaining])
  assert_true(
    all(exists),
    "independent trace parse finds no unexplained missing dependency"
  )
  surviving <- remaining[exists]
  regular <- vapply(
    candidates[surviving],
    function(path) file_test("-f", path),
    logical(1L)
  )
  reasons[surviving[!regular]] <- "non_regular"
  dependency_paths <- vapply(
    candidates[surviving[regular]], normalizePath, character(1L),
    winslash = "/", mustWork = TRUE
  )
  dependency_paths <- sort(
    unique(c(unname(dependency_paths), tracer_path)), method = "radix"
  )
  files <- data.frame(
    path = dependency_paths,
    sha256 = unname(vapply(
      dependency_paths,
      function(path) digest::digest(
        file = path, algo = "sha256", serialize = FALSE
      ),
      character(1L)
    )),
    stringsAsFactors = FALSE
  )
  exclusions <- data.frame(
    path = candidates[!is.na(reasons)],
    reason = reasons[!is.na(reasons)],
    stringsAsFactors = FALSE
  )
  exclusions <- exclusions[
    order(exclusions$path, method = "radix"), , drop = FALSE
  ]
  files <- independent_csv_semantic_frame(files)
  exclusions <- independent_csv_semantic_frame(exclusions)
  list(files = files, exclusions = exclusions)
}
reparsed_native_build <- independent_trace_tables(
  trace_payload_path, manifest$native_build_tracer_path
)
replayed_native_build <-
  fastkpc_full_cuda_fixed_sp_reconstruct_native_build_dependencies(
    trace_path = trace_payload_path,
    build_working_dir = manifest$native_build_working_dir,
    tracer_path = manifest$native_build_tracer_path,
    trace_invocation = manifest$native_build_dependency_trace_invocation
  )
replayed_native_build_files <- independent_csv_semantic_frame(
  replayed_native_build$files
)
replayed_native_build_exclusions <- independent_csv_semantic_frame(
  replayed_native_build$exclusions
)
replayed_command_projection_clean <- tryCatch({
  commands <- replayed_native_build$commands
  command_rows_clean <- vapply(commands, function(command) {
    executable_index <- match(
      command$executable_path, replayed_native_build_files$path
    )
    output_flags <- which(command$argv == "-o")
    output_markers <- which(command$argv == "<OUTPUT>")
    is.list(command) && !is.object(command) && identical(
      names(command),
      c("role", "executable_path", "executable_sha256", "argv")
    ) && typeof(command$role) == "character" &&
      length(command$role) == 1L && !anyNA(command$role) &&
      command$role %in% c("cxx_compile", "cuda_compile", "link") &&
      typeof(command$executable_path) == "character" &&
      length(command$executable_path) == 1L &&
      !anyNA(command$executable_path) &&
      startsWith(command$executable_path, "/") &&
      typeof(command$executable_sha256) == "character" &&
      length(command$executable_sha256) == 1L &&
      grepl("^[0-9a-f]{64}$", command$executable_sha256) &&
      typeof(command$argv) == "character" && !is.object(command$argv) &&
      is.null(attributes(command$argv)) && length(command$argv) >= 4L &&
      !anyNA(command$argv) && all(nzchar(command$argv)) &&
      !any(grepl("[\r\n]", command$argv)) &&
      identical(command$argv[[1L]], "<EXECUTABLE>") &&
      length(output_flags) == 1L &&
      output_flags[[1L]] < length(command$argv) &&
      identical(output_markers, output_flags + 1L) &&
      !is.na(executable_index) && identical(
        command$executable_sha256,
        replayed_native_build_files$sha256[[executable_index]]
      )
  }, logical(1L))
  command_roles <- vapply(commands, `[[`, character(1L), "role")
  identical(
    replayed_native_build$command_projection_schema_version,
    "full-cuda-ci-native-build-command-projection-v1"
  ) && typeof(replayed_native_build$command_count) == "integer" &&
    length(replayed_native_build$command_count) == 1L &&
    !is.na(replayed_native_build$command_count) &&
    replayed_native_build$command_count >= 3L && is.list(commands) &&
    !is.object(commands) && is.null(attributes(commands)) &&
    length(commands) == replayed_native_build$command_count &&
    all(command_rows_clean) && identical(
      sort(unique(command_roles), method = "radix"),
      c("cuda_compile", "cxx_compile", "link")
    )
}, error = function(error) FALSE)
replayed_environment <- replayed_native_build$build_environment
replayed_build_environment_clean <- tryCatch(
  identical(
    replayed_native_build$build_environment_schema_version,
    "full-cuda-ci-native-build-environment-v1"
  ) && is.data.frame(replayed_environment) && identical(
    names(replayed_environment), c("name", "is_set", "value")
  ) && identical(
    replayed_environment$name, independent_native_build_environment_names
  ) && nrow(replayed_environment) ==
    length(independent_native_build_environment_names) && identical(
    rownames(replayed_environment),
    as.character(seq_len(nrow(replayed_environment)))
  ) && typeof(replayed_environment$is_set) == "logical" &&
    !is.object(replayed_environment$is_set) &&
    is.null(attributes(replayed_environment$is_set)) &&
    !anyNA(replayed_environment$is_set) &&
    typeof(replayed_environment$value) == "character" &&
    !is.object(replayed_environment$value) &&
    is.null(attributes(replayed_environment$value)) &&
    !anyNA(replayed_environment$value) &&
    !any(grepl("[\r\n]", replayed_environment$value)) &&
    all(replayed_environment$is_set | replayed_environment$value == ""),
  error = function(error) FALSE
)
assert_true(
  isTRUE(replayed_command_projection_clean) &&
    isTRUE(replayed_build_environment_clean),
  paste0(
    "raw trace replay reconstructs the complete native build command and ",
    "environment projection"
  )
)
assert_true(
  identical(
    replayed_native_build$schema_version,
    manifest$native_build_dependencies_schema_version
  ) && identical(
    replayed_native_build$trace_semantics,
    manifest$native_build_dependency_trace_semantics
  ) && identical(
    replayed_native_build$trace_invocation,
    manifest$native_build_dependency_trace_invocation
  ) && identical(
    replayed_native_build$build_working_dir,
    manifest$native_build_working_dir
  ) && identical(
    replayed_native_build$trace_sha256,
    manifest$native_build_trace_sha256
  ) && identical(
    replayed_native_build$tracer_path,
    manifest$native_build_tracer_path
  ) && identical(
    replayed_native_build$tracer_sha256,
    manifest$native_build_tracer_sha256
  ) && identical(
    replayed_native_build$dependency_count,
    manifest$native_build_dependency_count
  ) && identical(
    replayed_native_build$exclusion_count,
    manifest$native_build_exclusion_count
  ),
  "raw trace replay binds the manifest native build trace metadata"
)
assert_true(
  identical(reparsed_native_build$files, native_build_dependencies) &&
    identical(
      reparsed_native_build$exclusions, native_build_exclusions
    ) && identical(
      replayed_native_build_files, reparsed_native_build$files
    ) && identical(
      replayed_native_build_exclusions, reparsed_native_build$exclusions
    ),
  paste0(
    "published raw trace independently reconstructs the published and ",
    "replayed dependency tables"
  )
)
dependency_identity_lines <- c(
  dependency_identity_lines,
  independent_native_build_command_lines(replayed_native_build)
)
independent_native_build_dependencies_sha256 <- unname(digest::digest(
  charToRaw(enc2utf8(paste0(
    paste(dependency_identity_lines, collapse = "\n"), "\n"
  ))),
  algo = "sha256", serialize = FALSE
))
assert_identical(
  manifest$native_build_dependencies_sha256,
  independent_native_build_dependencies_sha256,
  paste0(
    "manifest binds the actual traced native build dependency closure, ",
    "command projection, and frozen environment"
  )
)
assert_true(
  identical(
    replayed_native_build$aggregate_sha256,
    independent_native_build_dependencies_sha256
  ),
  "production raw-trace replay agrees with the independent full-v3 encoder"
)
closure_lines <- c(
  paste0("closure.schema=", manifest$source_closure_schema_version),
  paste0("closure.discovery=", manifest$source_discovery_semantics),
  paste0("closure.project_root=", manifest$source_project_root),
  paste0("closure.count=", manifest$source_closure_count)
)
for (role in names(expected_direct_source_ids)) {
  closure_lines <- c(closure_lines, paste0(
    "closure.root.", role, "=", manifest_direct_source_ids[[role]]
  ))
}
for (source_id in expected_execution_source_ids) {
  closure_lines <- c(
    closure_lines,
    paste0("closure.source.", source_id, ".path=",
           manifest_source_paths[[source_id]]),
    paste0("closure.source.", source_id, ".sha256=",
           manifest_source_hashes[[source_id]]),
    paste0("closure.source.", source_id, ".dependencies=",
           paste(manifest_source_dependencies[[source_id]], collapse = ","))
  )
}
independent_source_closure_hash <- unname(digest::digest(
  charToRaw(enc2utf8(paste0(paste(closure_lines, collapse = "\n"), "\n"))),
  algo = "sha256", serialize = FALSE
))
assert_identical(
  manifest$source_closure_sha256, independent_source_closure_hash,
  "source closure hash uses the frozen locale-independent graph encoding"
)
snapshot_lines <- c(
  paste0("schema_version=", manifest$provenance_schema_version),
  paste0("provenance_mode=", manifest$provenance_mode),
  paste0("head_base_commit=", manifest$head_base_commit),
  paste0("source_closure.schema=", manifest$source_closure_schema_version),
  paste0("source_closure.discovery=", manifest$source_discovery_semantics),
  paste0("source_closure.project_root=", manifest$source_project_root),
  paste0("source_closure.count=", manifest$source_closure_count),
  paste0("source_closure.sha256=", manifest$source_closure_sha256)
)
for (role in names(expected_direct_source_ids)) {
  snapshot_lines <- c(snapshot_lines, paste0(
    "source_closure.root.", role, "=", manifest_direct_source_ids[[role]]
  ))
}
for (source_id in expected_execution_source_ids) {
  snapshot_lines <- c(
    snapshot_lines,
    paste0("source.", source_id, ".path=",
           manifest_source_paths[[source_id]]),
    paste0("source.", source_id, ".sha256=",
           manifest_source_hashes[[source_id]]),
    paste0("source.", source_id, ".git_state=",
           manifest_source_states[[source_id]]),
    paste0("source.", source_id, ".dependencies=",
           paste(manifest_source_dependencies[[source_id]], collapse = ","))
  )
}
for (input_id in expected_native_build_input_ids) {
  snapshot_lines <- c(
    snapshot_lines,
    paste0("native_build_input.", input_id, ".path=",
           manifest_native_build_input_paths[[input_id]]),
    paste0("native_build_input.", input_id, ".sha256=",
           manifest_native_build_input_hashes[[input_id]]),
    paste0("native_build_input.", input_id, ".git_state=",
           manifest_native_build_input_states[[input_id]])
  )
}
snapshot_lines <- c(
  snapshot_lines,
  paste0(
    "native_build_inputs.sha256=",
    manifest$native_build_inputs_sha256
  ),
  paste0(
    "native_build_dependencies.schema=",
    manifest$native_build_dependencies_schema_version
  ),
  paste0(
    "native_build_dependencies.trace_semantics=",
    manifest$native_build_dependency_trace_semantics
  ),
  paste0(
    "native_build_dependencies.trace_invocation=",
    manifest$native_build_dependency_trace_invocation
  ),
  paste0(
    "native_build_dependencies.build_working_dir=",
    manifest$native_build_working_dir
  ),
  paste0(
    "native_build_dependencies.trace_sha256=",
    manifest$native_build_trace_sha256
  ),
  paste0(
    "native_build_dependencies.tracer_path=",
    manifest$native_build_tracer_path
  ),
  paste0(
    "native_build_dependencies.tracer_sha256=",
    manifest$native_build_tracer_sha256
  ),
  paste0(
    "native_build_dependencies.count=",
    manifest$native_build_dependency_count
  ),
  paste0(
    "native_build_dependencies.exclusion_count=",
    manifest$native_build_exclusion_count
  ),
  paste0(
    "native_build_dependencies.sha256=",
    manifest$native_build_dependencies_sha256
  ),
  paste0(
    "relevant_sources_dirty_or_untracked=",
    if (manifest$relevant_sources_dirty_or_untracked) "true" else "false"
  ),
  paste0("native.identity=", manifest$native_library_identity),
  paste0("native.path=", manifest$native_library_path),
  paste0("native.sha256=", manifest$native_library_sha256)
)
independent_snapshot_hash <- unname(digest::digest(
  charToRaw(enc2utf8(paste0(
    paste(snapshot_lines, collapse = "\n"), "\n"
  ))),
  algo = "sha256", serialize = FALSE
))
assert_identical(
  manifest$execution_snapshot_sha256,
  independent_snapshot_hash,
  "execution provenance uses the frozen locale-independent text encoding"
)
assert_true(
  identical(summary$provenance_mode, manifest$provenance_mode) &&
    identical(summary$head_base_commit, manifest$head_base_commit) &&
    identical(
      summary$source_closure_schema_version,
      manifest$source_closure_schema_version
    ) && identical(
      summary$source_closure_count, manifest$source_closure_count
    ) && identical(
      summary$source_closure_sha256, manifest$source_closure_sha256
    ) &&
    identical(
      summary$execution_snapshot_sha256,
      manifest$execution_snapshot_sha256
    ) && identical(
      summary$relevant_sources_dirty_or_untracked,
      manifest$relevant_sources_dirty_or_untracked
    ) && identical(
      summary$native_build_inputs_sha256,
      manifest$native_build_inputs_sha256
    ) && identical(
      summary$native_build_dependency_count,
      manifest$native_build_dependency_count
    ) && identical(
      summary$native_build_dependencies_sha256,
      manifest$native_build_dependencies_sha256
    ) && identical(
      summary$native_build_trace_sha256,
      manifest$native_build_trace_sha256
    ) && identical(
      summary$native_build_tracer_sha256,
      manifest$native_build_tracer_sha256
    ) && identical(
      summary$native_library_sha256,
      manifest$native_library_sha256
    ) && isTRUE(summary$execution_sources_unchanged_after_run),
  "summary repeats only validated execution provenance evidence"
)

independent_catalog <- fastkpc_full_cuda_open_fixed_sp_catalog(
  phase0_dir = phase0_dir,
  phase1_dir = phase1_dir,
  phase2_dir = phase2_dir,
  data_path = data_path
)
independent_qualification <- fastkpc_full_cuda_fixed_sp_scope(
  independent_catalog, "qualification"
)
expected_setup_keys <- as.character(
  independent_qualification$setup_rows$prepared_s_key_sha256
)
expected_target_keys <- as.character(
  independent_qualification$target_rows$residual_key_sha256
)
expected_target_prepared_s_keys <- as.character(
  independent_qualification$target_rows$prepared_s_key_sha256
)
independent_setup_digest <- fastkpc_full_cuda_census_key_set_hash(
  expected_setup_keys
)
independent_target_digest <- fastkpc_full_cuda_census_key_set_hash(
  expected_target_keys
)
assert_true(
  identical(
    independent_catalog$catalog_contract$qualification_subset_hash,
    "0adea2bac7b31615421f180b6caa5aeef5567bafa0e45a319b358136bf429c61"
  ) && identical(
    manifest$qualification_subset_hash,
    independent_catalog$catalog_contract$qualification_subset_hash
  ) && identical(
    summary$qualification_subset_hash,
    independent_catalog$catalog_contract$qualification_subset_hash
  ) && identical(summary$scope_subset_hash,
                  manifest$qualification_subset_hash) &&
    identical(manifest$ordered_setup_key_digest, independent_setup_digest) &&
    identical(summary$ordered_setup_key_digest, independent_setup_digest) &&
    identical(manifest$ordered_target_key_digest, independent_target_digest) &&
    identical(summary$ordered_target_key_digest, independent_target_digest),
  "writer catalog identity matches an independently reopened canonical catalog"
)
assert_true(
  length(expected_setup_keys) == 2061L && !anyNA(expected_setup_keys) &&
    length(expected_target_keys) == 6143L && !anyNA(expected_target_keys),
  "authenticated qualification keys map completely"
)
assert_identical(
  setup_records$prepared_s_key_sha256, expected_setup_keys,
  "setup rows preserve PreparedSKey radix order"
)
assert_identical(
  batch_records$prepared_s_key_sha256, expected_setup_keys,
  "batch rows preserve PreparedSKey radix order"
)
assert_identical(
  target_records$residual_key_sha256, expected_target_keys,
  "target rows preserve canonical per-setup residual-key order"
)
assert_identical(
  target_records$prepared_s_key_sha256, expected_target_prepared_s_keys,
  "target rows preserve canonical PreparedS ownership"
)
assert_true(
  !anyDuplicated(expected_setup_keys) && !anyDuplicated(expected_target_keys) &&
    all(expected_target_prepared_s_keys %in% expected_setup_keys) &&
    identical(
      order(
        expected_target_prepared_s_keys, expected_target_keys,
        method = "radix"
      ),
      seq_along(expected_target_keys)
    ),
  "qualification setup/target keys and ownership order are canonical"
)

required_target_fields <- c(
  "prepared_s_key_sha256", "residual_key_sha256", "condition_bucket",
  "phase1_condition", "null_dim", "planned_route",
  "authenticated_planned_route", "executed_route", "reroute_reason",
  "solver_status", "target_true_batched", "geqrf_info", "ormqr_info",
  "svd_info", "effective_rank", "sigma_max", "smallest_retained_sigma",
  "aggregate_penalty_root_rank", "aggregate_penalty_root_pivot",
  "aggregate_factor_call_count", "aggregate_b_build_count",
  "cpu_aggregate_penalty_root_rank", "cpu_aggregate_penalty_root_pivot",
  "cpu_aggregate_effective_rank", "cpu_aggregate_effective_rank_threshold",
  "cpu_aggregate_sigma_max", "aggregate_penalty_root_rank_exact",
  "aggregate_penalty_root_pivot_exact", "aggregate_effective_rank_exact",
  "numeric_reference", "outputs_all_finite", "residual_max_abs_diff",
  "residual_relative_l2_diff", "fitted_max_abs_diff",
  "fitted_relative_l2_diff", "fitted_numeric_hash",
  "residual_numeric_hash", "oracle_call_count", "oracle_fitted_hash",
  "oracle_residual_hash", "authenticated_fitted_hash",
  "authenticated_residual_hash", "oracle_fitted_hash_exact",
  "oracle_residual_hash_exact", "approximate_backend"
)
required_batch_fields <- c(
  "target_count", "true_batched_subgroup_count",
  "planned_cholesky_target_count", "planned_qr_target_count",
  "planned_svd_target_count", "executed_cholesky_target_count",
  "executed_qr_target_count", "executed_svd_target_count",
  "cholesky_to_svd_count", "qr_to_svd_count", "stable_reroute_count",
  "aggregate_penalty_factor_count", "aggregate_svd_b_build_count",
  "aggregate_penalty_root_d2h_count", "workspace_grow_count_after_warmup",
  "per_target_allocation_count_after_warmup",
  "per_target_handle_create_count", "cuda_device_synchronize_count",
  "qr_checkpoint_wait_count", "svd_checkpoint_wait_count",
  "implicit_residual_d2h_count", "invalid_output_init_count",
  "cpu_fallback_count", "unknown_fallback_count",
  "output_slot_release_count", "output_slot_leased_after_release"
)
required_setup_fields <- c(
  "penalty_root_matrix_count", "penalty_root_row_count",
  "H_root_matrix_count", "penalty_root_rank_mismatch_count",
  "output_slot_state_after_release", "output_slot_leased_after_release"
)
required_runtime_fields <- c(
  "stage", "workspace_grow_count", "stable_workspace_grow_count",
  "cuda_device_synchronize_count"
)
assert_true(
  length(setdiff(required_target_fields, names(target_records))) == 0L &&
    length(setdiff(required_batch_fields, names(batch_records))) == 0L &&
    length(setdiff(required_setup_fields, names(setup_records))) == 0L &&
    length(setdiff(required_runtime_fields, names(runtime_records))) == 0L,
  "qualification row-level numerical and lifecycle schema is complete"
)
target_type_fields <- list(
  character = c(
    "prepared_s_key_sha256", "residual_key_sha256", "condition_bucket",
    "planned_route", "authenticated_planned_route", "executed_route",
    "reroute_reason", "solver_status", "numeric_reference",
    "fitted_numeric_hash", "residual_numeric_hash", "oracle_fitted_hash",
    "oracle_residual_hash", "authenticated_fitted_hash",
    "authenticated_residual_hash"
  ),
  integer = c(
    "null_dim", "target_ordinal", "geqrf_info", "ormqr_info", "svd_info",
    "aggregate_factor_call_count", "aggregate_b_build_count",
    "oracle_call_count"
  ),
  logical = c(
    "target_true_batched", "outputs_all_finite",
    "oracle_fitted_hash_exact", "oracle_residual_hash_exact",
    "approximate_backend"
  ),
  double = c(
    "phase1_condition", "residual_max_abs_diff",
    "residual_relative_l2_diff", "fitted_max_abs_diff",
    "fitted_relative_l2_diff"
  )
)
for (type in names(target_type_fields)) {
  for (field in target_type_fields[[type]]) {
    assert_bare_vector(
      target_records[[field]], type, 6143L,
      paste("qualification target column is exact bare", type, field)
    )
  }
}
for (field in c(
  "aggregate_penalty_root_rank", "cpu_aggregate_penalty_root_rank",
  "cpu_aggregate_effective_rank"
)) {
  assert_bare_vector(
    target_records[[field]], "integer", 6143L,
    paste("qualification target nullable integer schema", field),
    allow_na = TRUE
  )
}
for (field in c(
  "aggregate_penalty_root_rank_exact",
  "aggregate_penalty_root_pivot_exact", "aggregate_effective_rank_exact"
)) {
  assert_bare_vector(
    target_records[[field]], "logical", 6143L,
    paste("qualification target nullable logical schema", field),
    allow_na = TRUE
  )
}
for (field in c(
  "cpu_aggregate_effective_rank_threshold", "cpu_aggregate_sigma_max"
)) {
  assert_bare_vector(
    target_records[[field]], "double", 6143L,
    paste("qualification target nullable double schema", field),
    allow_na = TRUE
  )
}
for (field in c(
  "aggregate_penalty_root_pivot", "cpu_aggregate_penalty_root_pivot"
)) {
  value <- target_records[[field]]
  assert_true(
    typeof(value) == "list" && length(value) == 6143L &&
      identical(attributes(value), list(class = "AsIs")),
    paste("qualification pivot list schema is exact", field)
  )
}
for (field in c(
  "target_count", "true_batched_subgroup_count",
  "aggregate_penalty_factor_count", "aggregate_svd_b_build_count",
  "aggregate_penalty_root_d2h_count",
  "workspace_grow_count_after_warmup",
  "stable_workspace_grow_count_after_warmup",
  "per_target_allocation_count_after_warmup",
  "per_target_handle_create_count", "cuda_device_synchronize_count",
  "implicit_residual_d2h_count", "invalid_output_init_count",
  "cpu_fallback_count", "unknown_fallback_count",
  "output_slot_release_count"
)) {
  assert_bare_vector(
    batch_records[[field]], "integer", 2061L,
    paste("qualification batch integer schema is exact", field)
  )
}
assert_bare_vector(
  batch_records$aggregate_penalty_root_d2h_bytes,
  "double", 2061L,
  "qualification batch aggregate-root D2H byte schema is exact"
)
for (field in c(
  "output_slot_leased_after_release", "native_batch_call"
)) {
  assert_bare_vector(
    batch_records[[field]], "logical", 2061L,
    paste("qualification batch logical schema is exact", field)
  )
}
for (field in c(
  "penalty_root_matrix_count", "penalty_root_row_count",
  "H_root_matrix_count", "penalty_root_rank_mismatch_count"
)) {
  assert_bare_vector(
    setup_records[[field]], "integer", 2061L,
    paste("qualification setup integer schema is exact", field)
  )
}
assert_bare_vector(
  setup_records$output_slot_state_after_release,
  "character", 2061L,
  "qualification setup release-state schema is exact"
)
assert_bare_vector(
  setup_records$output_slot_leased_after_release,
  "logical", 2061L,
  "qualification setup lease schema is exact"
)

route_counts <- function(rows) {
  planned_cholesky <- sum(rows$planned_route == "CHOLESKY_BATCHED")
  planned_qr <- sum(rows$planned_route == "AUGMENTED_QR")
  planned_svd <- sum(rows$planned_route == "AUGMENTED_SVD")
  executed_cholesky <- sum(rows$executed_route == "CHOLESKY_BATCHED")
  executed_qr <- sum(rows$executed_route == "AUGMENTED_QR")
  executed_svd <- sum(rows$executed_route == "AUGMENTED_SVD")
  cholesky_to_svd <- sum(
    rows$planned_route == "CHOLESKY_BATCHED" &
      rows$executed_route == "AUGMENTED_SVD"
  )
  qr_to_svd <- sum(
    rows$planned_route == "AUGMENTED_QR" &
      rows$executed_route == "AUGMENTED_SVD"
  )
  stable_reroute <- sum(rows$planned_route != rows$executed_route)
  list(
    planned_cholesky = as.integer(planned_cholesky),
    planned_qr = as.integer(planned_qr),
    planned_svd = as.integer(planned_svd),
    executed_cholesky = as.integer(executed_cholesky),
    executed_qr = as.integer(executed_qr),
    executed_svd = as.integer(executed_svd),
    cholesky_to_svd = as.integer(cholesky_to_svd),
    qr_to_svd = as.integer(qr_to_svd),
    stable_reroute = as.integer(stable_reroute),
    conserved =
      planned_cholesky == executed_cholesky + cholesky_to_svd &&
      planned_qr == executed_qr + qr_to_svd &&
      executed_svd == planned_svd + cholesky_to_svd + qr_to_svd &&
      stable_reroute == cholesky_to_svd + qr_to_svd
  )
}

target_indices <- lapply(expected_setup_keys, function(setup_key) {
  which(target_records$prepared_s_key_sha256 == setup_key)
})
target_count_by_setup <- vapply(target_indices, length, integer(1L))
safe_count_by_setup <- vapply(target_indices, function(indices) {
  as.integer(sum(target_records$planned_route[indices] == "CHOLESKY_BATCHED"))
}, integer(1L))
all_safe <- safe_count_by_setup == target_count_by_setup
all_stable <- safe_count_by_setup == 0L
mixed <- !all_safe & !all_stable
route_totals <- route_counts(target_records)
assert_true(
  isTRUE(route_totals$conserved) && all(vapply(target_indices, function(indices) {
    isTRUE(route_counts(target_records[indices, , drop = FALSE])$conserved)
  }, logical(1L))),
  "route conservation holds globally and for every setup batch"
)

reserved_runtime <- match("workspace-reserved", runtime_records$stage)
final_runtime <- match("final", runtime_records$stage)
assert_true(
  !is.na(reserved_runtime) && !is.na(final_runtime),
  "runtime reserve and final snapshots are present"
)
environment_blank <- match("", environment_lines)
assert_true(
  !is.na(environment_blank) && environment_blank > 1L,
  "environment metadata has one nonempty fixed header"
)
environment_header <- environment_lines[seq_len(environment_blank - 1L)]
environment_keys <- sub("=.*$", "", environment_header)
environment_values <- sub("^[^=]*=", "", environment_header)
assert_true(
  !anyDuplicated(environment_keys) && all(nzchar(environment_keys)),
  "environment metadata keys are unique and nonempty"
)
environment_metadata <- setNames(environment_values, environment_keys)
assert_true(
  nzchar(environment_metadata[["cuda_toolkit_version"]]) &&
    nzchar(environment_metadata[["cuda_driver_version"]]) &&
    identical(
      environment_metadata[["cuda_toolkit_version"]],
      as.character(runtime_records$cuda_toolkit_version[[final_runtime]])
    ) && identical(
      environment_metadata[["cuda_driver_version"]],
      as.character(runtime_records$cuda_driver_version[[final_runtime]])
    ) && identical(
      environment_metadata[["qualification_dcov_backend"]], "cpp"
    ) && identical(
      environment_metadata[["qualification_dcov_low_rank_backend"]],
      "spectra"
    ),
  "environment CUDA toolkit/driver versions match final runtime evidence"
)
assert_true(
  identical(
    environment_metadata[["provenance_mode"]],
    manifest$provenance_mode
  ) && identical(
    environment_metadata[["head_base_commit"]],
    manifest$head_base_commit
  ) && identical(
    environment_metadata[["source_closure_schema_version"]],
    manifest$source_closure_schema_version
  ) && identical(
    environment_metadata[["source_discovery_semantics"]],
    manifest$source_discovery_semantics
  ) && identical(
    environment_metadata[["source_closure_count"]],
    as.character(manifest$source_closure_count)
  ) && identical(
    environment_metadata[["source_closure_sha256"]],
    manifest$source_closure_sha256
  ) && identical(
    environment_metadata[["execution_snapshot_sha256"]],
    manifest$execution_snapshot_sha256
  ) && identical(
    environment_metadata[["native_build_inputs_sha256"]],
    manifest$native_build_inputs_sha256
  ) && identical(
    environment_metadata[["native_build_dependencies_schema_version"]],
    manifest$native_build_dependencies_schema_version
  ) && identical(
    environment_metadata[["native_build_dependency_trace_semantics"]],
    manifest$native_build_dependency_trace_semantics
  ) && identical(
    environment_metadata[["native_build_dependency_trace_invocation"]],
    manifest$native_build_dependency_trace_invocation
  ) && identical(
    environment_metadata[["native_build_working_dir"]],
    manifest$native_build_working_dir
  ) && identical(
    environment_metadata[["native_build_trace_path"]],
    manifest$native_build_trace_path
  ) && identical(
    environment_metadata[["native_build_trace_sha256"]],
    manifest$native_build_trace_sha256
  ) && identical(
    environment_metadata[["native_build_tracer_path"]],
    manifest$native_build_tracer_path
  ) && identical(
    environment_metadata[["native_build_tracer_sha256"]],
    manifest$native_build_tracer_sha256
  ) && identical(
    environment_metadata[["native_build_dependency_count"]],
    as.character(manifest$native_build_dependency_count)
  ) && identical(
    environment_metadata[["native_build_exclusion_count"]],
    as.character(manifest$native_build_exclusion_count)
  ) && identical(
    environment_metadata[["native_build_dependencies_sha256"]],
    manifest$native_build_dependencies_sha256
  ) && identical(
    environment_metadata[["native_library_path"]],
    manifest$native_library_path
  ) && identical(
    environment_metadata[["native_library_sha256"]],
    manifest$native_library_sha256
  ),
  "environment provenance header matches the authenticated manifest"
)
for (source_id in expected_execution_source_ids) {
  assert_true(
    identical(
      environment_metadata[[paste0("source.", source_id, ".path")]],
      manifest_source_paths[[source_id]]
    ) && identical(
      environment_metadata[[paste0("source.", source_id, ".sha256")]],
      manifest_source_hashes[[source_id]]
    ) && identical(
      environment_metadata[[paste0("source.", source_id, ".git_state")]],
      manifest_source_states[[source_id]]
    ),
    paste("environment source provenance matches manifest", source_id)
  )
}
for (input_id in expected_native_build_input_ids) {
  assert_true(
    identical(
      environment_metadata[[paste0(
        "native_build_input.", input_id, ".path"
      )]],
      manifest_native_build_input_paths[[input_id]]
    ) && identical(
      environment_metadata[[paste0(
        "native_build_input.", input_id, ".sha256"
      )]],
      manifest_native_build_input_hashes[[input_id]]
    ) && identical(
      environment_metadata[[paste0(
        "native_build_input.", input_id, ".git_state"
      )]],
      manifest_native_build_input_states[[input_id]]
    ),
    paste("environment native build provenance matches manifest", input_id)
  )
}
runtime_workspace_delta <- as.integer(
  runtime_records$workspace_grow_count[[final_runtime]] -
    runtime_records$workspace_grow_count[[reserved_runtime]]
)
runtime_stable_workspace_delta <- as.integer(
  runtime_records$stable_workspace_grow_count[[final_runtime]] -
    runtime_records$stable_workspace_grow_count[[reserved_runtime]]
)
runtime_sync_delta <- as.integer(
  runtime_records$cuda_device_synchronize_count[[final_runtime]] -
    runtime_records$cuda_device_synchronize_count[[reserved_runtime]]
)
qr_affected <- batch_records$planned_qr_target_count > 0L
svd_affected <- batch_records$executed_svd_target_count > 0L
target_level_stable_sync_count <- as.integer(sum(
  pmax(
    batch_records$qr_checkpoint_wait_count - as.integer(qr_affected), 0L
  ) + pmax(
    batch_records$svd_checkpoint_wait_count - as.integer(svd_affected), 0L
  )
))
whole_batch_true_batched <- as.integer(sum(vapply(
  target_indices, function(indices) {
    length(indices) >= 2L && all(target_records$target_true_batched[indices])
  }, logical(1L)
)))
svd <- target_records$executed_route == "AUGMENTED_SVD"
non_svd <- !svd

recomputed <- list(
  setup_count = as.integer(nrow(setup_records)),
  target_count = as.integer(nrow(target_records)),
  penalty_root_matrix_count = as.integer(sum(
    setup_records$penalty_root_matrix_count
  )),
  penalty_root_row_count = as.integer(sum(
    setup_records$penalty_root_row_count
  )),
  H_root_matrix_count = as.integer(sum(setup_records$H_root_matrix_count)),
  planned_cholesky_count = route_totals$planned_cholesky,
  planned_qr_count = route_totals$planned_qr,
  planned_svd_count = route_totals$planned_svd,
  executed_cholesky_count = route_totals$executed_cholesky,
  executed_qr_count = route_totals$executed_qr,
  executed_svd_count = route_totals$executed_svd,
  cholesky_to_svd_count = route_totals$cholesky_to_svd,
  qr_to_svd_count = route_totals$qr_to_svd,
  svd_finite_high_count = as.integer(sum(
    svd & is.finite(target_records$phase1_condition)
  )),
  svd_nonfinite_count = as.integer(sum(
    svd & !is.finite(target_records$phase1_condition)
  )),
  all_safe_batch_count = as.integer(sum(all_safe)),
  mixed_batch_count = as.integer(sum(mixed)),
  all_stable_batch_count = as.integer(sum(all_stable)),
  true_batched_subgroup_count = as.integer(sum(
    batch_records$true_batched_subgroup_count
  )),
  true_batched_target_count = as.integer(sum(
    target_records$target_true_batched
  )),
  cholesky_single_target_count = as.integer(sum(
    target_records$solver_status == "OK_CHOLESKY_SINGLE"
  )),
  whole_batch_true_batched_count = whole_batch_true_batched,
  non_ok_status_count = as.integer(sum(
    !startsWith(target_records$solver_status, "OK_")
  )),
  stable_not_implemented_count = as.integer(sum(
    target_records$solver_status == "ERR_STABLE_PATH_NOT_IMPLEMENTED"
  )),
  stable_reroute_count = route_totals$stable_reroute,
  root_rank_mismatch_count = as.integer(sum(
    setup_records$penalty_root_rank_mismatch_count
  )),
  aggregate_penalty_factor_count = as.integer(sum(
    target_records$aggregate_factor_call_count
  )),
  aggregate_svd_b_build_count = as.integer(sum(
    target_records$aggregate_b_build_count
  )),
  aggregate_penalty_root_rank_mismatch_count = as.integer(sum(
    !target_records$aggregate_penalty_root_rank_exact[svd]
  )),
  aggregate_penalty_root_pivot_mismatch_count = as.integer(sum(
    !target_records$aggregate_penalty_root_pivot_exact[svd]
  )),
  aggregate_penalty_root_d2h_count = as.integer(sum(
    batch_records$aggregate_penalty_root_d2h_count
  )),
  aggregate_penalty_root_d2h_bytes = as.double(sum(
    batch_records$aggregate_penalty_root_d2h_bytes
  )),
  workspace_grow_count_after_warmup = runtime_workspace_delta,
  stable_workspace_grow_count_after_warmup =
    runtime_stable_workspace_delta,
  per_target_allocation_count_after_warmup = as.integer(sum(
    batch_records$per_target_allocation_count_after_warmup
  )),
  per_target_handle_create_count = as.integer(sum(
    batch_records$per_target_handle_create_count
  )),
  cuda_device_synchronize_count = runtime_sync_delta,
  target_level_stable_sync_count = target_level_stable_sync_count,
  implicit_residual_d2h_count = as.integer(sum(
    batch_records$implicit_residual_d2h_count
  )),
  all_output_slot_leases_released = all(
    setup_records$output_slot_state_after_release == "free" &
      !setup_records$output_slot_leased_after_release &
      batch_records$output_slot_release_count == 1L &
      !batch_records$output_slot_leased_after_release
  ),
  invalid_output_init_count = as.integer(sum(
    batch_records$invalid_output_init_count
  )),
  cpu_fallback_count = as.integer(sum(batch_records$cpu_fallback_count)),
  unknown_fallback_count = as.integer(sum(
    batch_records$unknown_fallback_count
  )),
  approximate_backend_count = as.integer(sum(
    target_records$approximate_backend
  ))
)
expected <- list(
  setup_count = 2061L,
  target_count = 6143L,
  penalty_root_matrix_count = 6272L,
  penalty_root_row_count = 63552L,
  H_root_matrix_count = 0L,
  planned_cholesky_count = 3889L,
  planned_qr_count = 190L,
  planned_svd_count = 2064L,
  executed_cholesky_count = 3889L,
  executed_qr_count = 190L,
  executed_svd_count = 2064L,
  cholesky_to_svd_count = 0L,
  qr_to_svd_count = 0L,
  svd_finite_high_count = 902L,
  svd_nonfinite_count = 1162L,
  all_safe_batch_count = 723L,
  mixed_batch_count = 652L,
  all_stable_batch_count = 686L,
  true_batched_subgroup_count = 806L,
  true_batched_target_count = 3320L,
  cholesky_single_target_count = 569L,
  whole_batch_true_batched_count = 723L,
  non_ok_status_count = 0L,
  stable_not_implemented_count = 0L,
  stable_reroute_count = 0L,
  root_rank_mismatch_count = 0L,
  aggregate_penalty_factor_count = 2064L,
  aggregate_svd_b_build_count = 4128L,
  aggregate_penalty_root_rank_mismatch_count = 0L,
  aggregate_penalty_root_pivot_mismatch_count = 0L,
  aggregate_penalty_root_d2h_count = 0L,
  aggregate_penalty_root_d2h_bytes = 0,
  workspace_grow_count_after_warmup = 0L,
  stable_workspace_grow_count_after_warmup = 0L,
  per_target_allocation_count_after_warmup = 0L,
  per_target_handle_create_count = 0L,
  cuda_device_synchronize_count = 0L,
  target_level_stable_sync_count = 0L,
  implicit_residual_d2h_count = 0L,
  all_output_slot_leases_released = TRUE,
  invalid_output_init_count = 2061L,
  cpu_fallback_count = 0L,
  unknown_fallback_count = 0L,
  approximate_backend_count = 0L
)
assert_identical(
  recomputed, expected,
  "independently recomputed qualification counts are exact"
)

assert_true(
  is.list(summary) && !is.object(summary) && !anyDuplicated(names(summary)) &&
    all(names(expected) %in% names(summary)) &&
    all(vapply(names(expected), function(field) {
      value <- summary[[field]]
      reference <- expected[[field]]
      typeof(value) == typeof(reference) && length(value) == 1L &&
        !is.object(value) && is.null(attributes(value)) && !anyNA(value)
    }, logical(1L))),
  "summary JSON count fields have exact bare scalar types"
)
for (field in c(
  "scope", "scope_subset_hash", "qualification_subset_hash",
  "ordered_setup_key_digest", "ordered_target_key_digest",
  "route_status_hash", "numeric_hash", "artifact_schema_version",
  "provenance_mode", "head_base_commit", "source_closure_schema_version",
  "source_closure_sha256", "execution_snapshot_sha256",
  "native_build_inputs_sha256", "native_build_dependencies_sha256",
  "native_build_trace_sha256", "native_build_tracer_sha256",
  "native_library_sha256",
  "qualification_dcov_logical_ids_hash",
  "qualification_dcov_residual_key_hash", "qualification_dcov_rows_hash"
)) {
  assert_json_scalar(
    summary[[field]], "character",
    paste("summary character scalar type is exact", field)
  )
}
assert_json_scalar(
  summary$native_build_dependency_count, "integer",
  "summary native build dependency count is one exact integer"
)
for (field in c(
  "catalog_authenticated", "relevant_sources_dirty_or_untracked",
  "execution_sources_unchanged_after_run"
)) {
  assert_json_scalar(
    summary[[field]], "logical",
    paste("summary logical scalar type is exact", field)
  )
}
for (field in c(
  "max_residual_abs_diff", "max_residual_relative_l2_diff",
  "max_fitted_abs_diff", "max_fitted_relative_l2_diff",
  "qualification_dcov_max_absolute_p_value_difference",
  "elapsed_seconds", "stage_timing_total_seconds"
)) {
  assert_json_scalar(
    summary[[field]], "double",
    paste("summary double scalar type is exact", field)
  )
}
assert_json_scalar(
  summary$payload_file_count, "integer",
  "summary payload file count is one exact integer"
)
assert_json_scalar(
  summary$source_closure_count, "integer",
  "summary source closure count is one exact integer"
)
for (field in setdiff(
  expected_dcov_summary_fields,
  c(
    "qualification_dcov_max_absolute_p_value_difference",
    "qualification_dcov_logical_ids_hash",
    "qualification_dcov_residual_key_hash",
    "qualification_dcov_rows_hash"
  )
)) {
  assert_json_scalar(
    summary[[field]], "integer",
    paste("summary dCov integer scalar type is exact", field)
  )
}
assert_identical(
  summary[names(expected)], expected,
  "summary counts match independently recomputed RDS evidence"
)

error_fields <- c(
  "residual_max_abs_diff", "residual_relative_l2_diff",
  "fitted_max_abs_diff", "fitted_relative_l2_diff"
)
assert_true(
  all(vapply(error_fields, function(field) {
    value <- target_records[[field]]
    is.double(value) && length(value) == 6143L &&
      all(is.finite(value)) && all(value >= 0) && all(value < 1e-7)
  }, logical(1L))),
  "every qualification target satisfies the numeric ceiling"
)
route_numeric_max <- do.call(rbind, lapply(
  split(seq_len(nrow(target_records)), target_records$executed_route),
  function(indices) vapply(error_fields, function(field) {
    max(target_records[[field]][indices])
  }, double(1L))
))
condition_numeric_max <- do.call(rbind, lapply(
  split(seq_len(nrow(target_records)), target_records$condition_bucket),
  function(indices) vapply(error_fields, function(field) {
    max(target_records[[field]][indices])
  }, double(1L))
))
assert_true(
  all(is.finite(route_numeric_max)) && all(route_numeric_max < 1e-7) &&
    all(is.finite(condition_numeric_max)) &&
    all(condition_numeric_max < 1e-7),
  "route and condition-bucket numeric maxima satisfy the frozen ceiling"
)
assert_true(
  all(target_records$outputs_all_finite) &&
    identical(
      target_records$numeric_reference, rep("mgcv-fixed-sp", 6143L)
    ) && identical(target_records$oracle_call_count, rep(1L, 6143L)) &&
    all(target_records$oracle_fitted_hash_exact) &&
    all(target_records$oracle_residual_hash_exact) &&
    !any(target_records$approximate_backend),
  "all targets use finite authoritative mgcv-fixed-sp output"
)
for (field in c(
  "fitted_numeric_hash", "residual_numeric_hash", "oracle_fitted_hash",
  "oracle_residual_hash", "authenticated_fitted_hash",
  "authenticated_residual_hash"
)) {
  assert_sha_vector(
    target_records[[field]], 6143L,
    paste("qualification target hash is valid for", field)
  )
}
assert_true(
  identical(
    target_records$oracle_fitted_hash,
    target_records$authenticated_fitted_hash
  ) && identical(
    target_records$oracle_residual_hash,
    target_records$authenticated_residual_hash
  ),
  "C_magic hashes match the authenticated Phase 2 oracle"
)

expected_factor_calls <- as.integer(svd)
expected_b_builds <- 2L * expected_factor_calls
assert_true(
  identical(target_records$aggregate_factor_call_count,
            expected_factor_calls) &&
    identical(target_records$aggregate_b_build_count, expected_b_builds) &&
    identical(as.integer(sum(expected_factor_calls)), 2064L) &&
    identical(as.integer(sum(expected_b_builds)), 4128L),
  "per-target aggregate SVD factor/build lifecycle is exact"
)
assert_true(
  all(target_records$aggregate_penalty_root_rank_exact[svd]) &&
    all(target_records$aggregate_penalty_root_pivot_exact[svd]) &&
    all(target_records$aggregate_effective_rank_exact[svd]) &&
    identical(
      target_records$aggregate_penalty_root_rank[svd],
      target_records$cpu_aggregate_penalty_root_rank[svd]
    ) && all(vapply(which(svd), function(index) {
      identical(
        target_records$aggregate_penalty_root_pivot[[index]],
        target_records$cpu_aggregate_penalty_root_pivot[[index]]
      )
    }, logical(1L))) && identical(
      target_records$effective_rank[svd],
      target_records$cpu_aggregate_effective_rank[svd]
    ) && all(
      target_records$cpu_aggregate_effective_rank_threshold[svd] ==
        target_records$cpu_aggregate_sigma_max[svd] *
          sqrt(.Machine$double.eps)
    ) && all(is.na(target_records$aggregate_penalty_root_rank_exact[non_svd])) &&
    all(is.na(target_records$aggregate_penalty_root_pivot_exact[non_svd])) &&
    all(is.na(target_records$aggregate_effective_rank_exact[non_svd])),
  paste0(
    "device aggregate-root rank/pivot and padded augmented-SVD rank match ",
    "the CPU LAPACK/C_magic threshold oracle"
  )
)

qr <- target_records$executed_route == "AUGMENTED_QR"
assert_true(
  all(target_records$geqrf_info[qr] == 0L) &&
    all(target_records$ormqr_info[qr] == 0L) &&
    all(target_records$svd_info[svd] == 0L) &&
    all(startsWith(target_records$solver_status, "OK_")) &&
    all(target_records$planned_route == target_records$executed_route) &&
    all(target_records$reroute_reason == ""),
  "all cuSOLVER calls succeed without stable rerouting"
)
assert_true(
  runtime_workspace_delta == 0L && runtime_stable_workspace_delta == 0L &&
    sum(batch_records$aggregate_penalty_root_d2h_bytes) == 0 &&
    sum(batch_records$aggregate_penalty_root_d2h_count) == 0L &&
    runtime_sync_delta == 0L &&
    target_level_stable_sync_count == 0L &&
    sum(batch_records$per_target_allocation_count_after_warmup) == 0L &&
    sum(batch_records$per_target_handle_create_count) == 0L &&
    sum(batch_records$implicit_residual_d2h_count) == 0L,
  "qualification has no post-warmup growth or target-level synchronization"
)

route_status_hash <- fastkpc_full_cuda_census_metadata_hash(list(
  target_records$residual_key_sha256,
  target_records$planned_route,
  target_records$executed_route,
  target_records$reroute_reason,
  target_records$solver_status
))
numeric_hash <- fastkpc_full_cuda_census_metadata_hash(list(
  target_records$residual_key_sha256,
  target_records$fitted_numeric_hash,
  target_records$residual_numeric_hash
))
assert_identical(
  summary$route_status_hash, route_status_hash,
  "qualification route/status hash is independently reproducible"
)
assert_identical(
  manifest$route_status_hash, route_status_hash,
  "manifest route/status hash is recomputed from published target rows"
)
assert_identical(
  summary$numeric_hash, numeric_hash,
  "qualification numeric hash is independently reproducible"
)
assert_identical(
  manifest$numeric_hash, numeric_hash,
  "manifest numeric hash is recomputed from published target rows"
)

qualification_logical_tests <- readRDS(file.path(
  phase2_dir, "qualification_logical_tests.rds"
))
recomputed_dcov <-
  fastkpc_full_cuda_fixed_sp_summarize_qualification_dcov_records(
    dcov_records, qualification_logical_tests,
    target_records$residual_key_sha256
  )
assert_true(
  all(vapply(names(recomputed_dcov), function(field) {
    identical(summary[[field]], recomputed_dcov[[field]])
  }, logical(1L))) && identical(
    manifest$qualification_logical_tests_sha256,
    digest::digest(
      file = file.path(phase2_dir, "qualification_logical_tests.rds"),
      algo = "sha256", serialize = FALSE
    )
  ) && identical(
    manifest$qualification_dcov_logical_ids_hash,
    recomputed_dcov$qualification_dcov_logical_ids_hash
  ) && identical(
    manifest$qualification_dcov_residual_key_hash,
    recomputed_dcov$qualification_dcov_residual_key_hash
  ) && identical(
    manifest$qualification_dcov_rows_hash,
    recomputed_dcov$qualification_dcov_rows_hash
  ),
  "qualification dCov summary and manifest are recomputed from RDS rows"
)

payload_hashes <- unlist(manifest$payload_file_sha256, use.names = TRUE)
payload_names <- setdiff(expected_files, c("manifest.json", "summary.json"))
assert_identical(
  names(payload_hashes), payload_names,
  "manifest authenticates every payload in publication order"
)
actual_hashes <- vapply(payload_names, function(name) {
  digest::digest(
    file = file.path(output_dir, name), algo = "sha256", serialize = FALSE
  )
}, character(1L))
assert_identical(
  unname(payload_hashes), unname(actual_hashes),
  "manifest payload hashes match published bytes"
)
assert_identical(
  unname(payload_hashes[["native_build_trace.txt"]]),
  manifest$native_build_trace_sha256,
  "native build trace provenance hash matches its payload hash"
)
assert_true(
  identical(manifest$schema_version,
            "full-cuda-ci-fixed-sp-qualification-v6") &&
    identical(manifest$scope, "qualification") &&
    identical(unname(unlist(manifest$publication_order, use.names = FALSE)),
              c(payload_names, "manifest.json", "summary.json")) &&
    isTRUE(manifest$catalog_authenticated),
  "manifest binds qualification scope and staged publication order"
)

max_errors <- vapply(error_fields, function(field) {
  max(target_records[[field]])
}, double(1L))
cat("PASS Phase 3C fixed-sp qualification gate\n")
cat(
  "counts setups/targets/roots/root_rows/H_roots=",
  recomputed$setup_count, "/", recomputed$target_count, "/",
  recomputed$penalty_root_matrix_count, "/",
  recomputed$penalty_root_row_count, "/",
  recomputed$H_root_matrix_count, "\n", sep = ""
)
cat(
  "planned/executed cholesky/qr/svd=",
  recomputed$planned_cholesky_count, "/", recomputed$planned_qr_count, "/",
  recomputed$planned_svd_count, " ", recomputed$executed_cholesky_count, "/",
  recomputed$executed_qr_count, "/", recomputed$executed_svd_count,
  "\n", sep = ""
)
cat(
  "batch classes safe/mixed/stable=", recomputed$all_safe_batch_count, "/",
  recomputed$mixed_batch_count, "/", recomputed$all_stable_batch_count,
  "\n", sep = ""
)
cat(
  "aggregate factors/builds=", recomputed$aggregate_penalty_factor_count,
  "/", recomputed$aggregate_svd_b_build_count, "\n", sep = ""
)
cat(
  "max errors residual_abs/residual_rel/fitted_abs/fitted_rel=",
  paste(format(max_errors, digits = 17L), collapse = "/"), "\n", sep = ""
)
cat("route_status_hash=", route_status_hash, "\n", sep = "")
cat("numeric_hash=", numeric_hash, "\n", sep = "")
cat(
  "runner_elapsed_seconds=", format(as.numeric(summary$elapsed_seconds),
                                    digits = 17L), "\n", sep = ""
)
