.fastkpc_cuda_root <- function() {
  if (file.exists("fastkpc/src/r_api_cuda.cpp")) return(normalizePath("fastkpc"))
  stop("Cannot find fastkpc/src/r_api_cuda.cpp from current working directory",
       call. = FALSE)
}

fastkpc_resolve_max_conditioning_size <- function(
    max_conditioning_size = Inf, p) {
  if (!is.numeric(max_conditioning_size) ||
      length(max_conditioning_size) != 1L ||
      is.object(max_conditioning_size) ||
      !is.null(attributes(max_conditioning_size)) ||
      is.na(max_conditioning_size) || max_conditioning_size < 0 ||
      !is.numeric(p) || length(p) != 1L || is.na(p) ||
      p < 2 || p != as.integer(p)) {
    stop("max_conditioning_size must be one non-negative number and p must be an integer >= 2",
         call. = FALSE)
  }
  maximum <- as.integer(p) - 2L
  if (is.infinite(max_conditioning_size)) {
    if (max_conditioning_size < 0) {
      stop("max_conditioning_size must be non-negative", call. = FALSE)
    }
    return(maximum)
  }
  as.integer(min(floor(max_conditioning_size), maximum))
}

.fastkpc_cuda_so <- function() {
  file.path(.fastkpc_cuda_root(), "build", "fastkpc_cuda.so")
}

.fastkpc_cuda_loaded_paths <- function() {
  unname(vapply(getLoadedDLLs(), function(dll) {
    normalizePath(
      dll[["path"]], winslash = "/", mustWork = FALSE
    )
  }, character(1L)))
}

.fastkpc_cuda_decode_proc_maps_path <- function(value) {
  if (typeof(value) != "character" || length(value) != 1L ||
      is.object(value) || !is.null(attributes(value)) || anyNA(value)) {
    stop("Linux mapped-object path is malformed", call. = FALSE)
  }
  input <- charToRaw(value)
  output <- raw()
  index <- 1L
  while (index <= length(input)) {
    if (input[[index]] == as.raw(92L) && index + 3L <= length(input)) {
      digits <- as.integer(input[seq.int(index + 1L, index + 3L)]) - 48L
      if (all(digits >= 0L & digits <= 7L)) {
        output <- c(output, as.raw(sum(digits * c(64L, 8L, 1L))))
        index <- index + 4L
        next
      }
    }
    output <- c(output, input[[index]])
    index <- index + 1L
  }
  rawToChar(output)
}

.fastkpc_cuda_mapped_object_records <- function(maps_path = "/proc/self/maps") {
  if (typeof(maps_path) != "character" || length(maps_path) != 1L ||
      is.object(maps_path) || !is.null(attributes(maps_path)) ||
      anyNA(maps_path) || !nzchar(maps_path) || !file.exists(maps_path) ||
      dir.exists(maps_path)) {
    stop("Linux mapped-object snapshot is unavailable", call. = FALSE)
  }
  lines <- tryCatch(
    readLines(maps_path, warn = FALSE, encoding = "bytes"),
    error = function(error) stop(
      "Linux mapped-object snapshot is unavailable: ",
      conditionMessage(error), call. = FALSE
    )
  )
  pattern <- paste0(
    "^[[:xdigit:]]+-[[:xdigit:]]+[[:space:]]+[-rwxsp]{4}",
    "[[:space:]]+[[:xdigit:]]+[[:space:]]+([[:xdigit:]]+):",
    "([[:xdigit:]]+)[[:space:]]+([0-9]+)(?:[[:space:]]+(.*))?$"
  )
  if (length(lines) == 0L || any(!grepl(pattern, lines, perl = TRUE))) {
    stop("Linux mapped-object snapshot is malformed", call. = FALSE)
  }
  encoded <- sub(pattern, "\\4", lines, perl = TRUE)
  paths <- vapply(
    encoded, .fastkpc_cuda_decode_proc_maps_path, character(1L)
  )
  keep <- startsWith(paths, "/")
  if (!any(keep)) {
    return(data.frame(
      path = character(),
      live_path = character(),
      deleted = logical(),
      device_major_hex = character(),
      device_minor_hex = character(),
      inode = character(),
      stringsAsFactors = FALSE
    ))
  }
  paths <- unname(paths[keep])
  deleted <- endsWith(paths, " (deleted)")
  live_source <- sub(" \\(deleted\\)$", "", paths)
  live_path <- rep.int(NA_character_, length(paths))
  live_path[!deleted] <- vapply(
    live_source[!deleted], normalizePath, character(1L),
    winslash = "/", mustWork = FALSE
  )
  records <- data.frame(
    path = paths,
    live_path = live_path,
    deleted = deleted,
    device_major_hex = tolower(sub(pattern, "\\1", lines[keep], perl = TRUE)),
    device_minor_hex = tolower(sub(pattern, "\\2", lines[keep], perl = TRUE)),
    inode = sub(pattern, "\\3", lines[keep], perl = TRUE),
    stringsAsFactors = FALSE
  )
  records <- unique(records)
  records[order(records$path, records$inode, method = "radix"), , drop = FALSE]
}

.fastkpc_cuda_mapped_object_paths <- function(maps_path = "/proc/self/maps") {
  records <- .fastkpc_cuda_mapped_object_records(maps_path = maps_path)
  paths <- records$live_path[!records$deleted]
  paths <- paths[!is.na(paths)]
  if (length(paths) == 0L) return(character())
  sort(unique(unname(paths)), method = "radix")
}

.fastkpc_cuda_normalize_path_snapshot <- function(value, label) {
  if (typeof(value) != "character" || anyNA(value)) {
    stop(label, " is malformed", call. = FALSE)
  }
  unname(vapply(
    value, normalizePath, character(1L),
    winslash = "/", mustWork = FALSE
  ))
}

.fastkpc_cuda_normalize_hex_identity <- function(value) {
  if (typeof(value) != "character" || anyNA(value)) {
    stop("native library device identity is malformed", call. = FALSE)
  }
  normalized <- tolower(value)
  normalized <- sub("^0+([0-9a-f]+)$", "\\1", normalized)
  normalized[normalized == ""] <- "0"
  unname(normalized)
}

.fastkpc_cuda_posix_file_identity <- function(
    path, stat_binary = Sys.which("stat")) {
  if (typeof(path) != "character" || length(path) != 1L ||
      is.object(path) || !is.null(attributes(path)) || anyNA(path) ||
      !nzchar(path) || !file.exists(path) || dir.exists(path)) {
    stop("native library file identity path is invalid", call. = FALSE)
  }
  stat_binary <- unname(stat_binary)
  if (typeof(stat_binary) != "character" || length(stat_binary) != 1L ||
      is.object(stat_binary) || !is.null(attributes(stat_binary)) ||
      anyNA(stat_binary) || !nzchar(stat_binary) || !file.exists(stat_binary) ||
      dir.exists(stat_binary)) {
    stop("POSIX stat helper is unavailable", call. = FALSE)
  }
  normalized <- normalizePath(path, winslash = "/", mustWork = TRUE)
  bash_binary <- unname(Sys.which("bash"))
  if (typeof(bash_binary) != "character" || length(bash_binary) != 1L ||
      is.object(bash_binary) || !is.null(attributes(bash_binary)) ||
      anyNA(bash_binary) || !nzchar(bash_binary) || !file.exists(bash_binary) ||
      dir.exists(bash_binary)) {
    stop("POSIX stat helper shell is unavailable", call. = FALSE)
  }
  stat_script <- paste(
    sprintf("hex=$(%s -Lc %%D %s) || exit 1",
            shQuote(stat_binary), shQuote(normalized)),
    sprintf("ino=$(%s -Lc %%i %s) || exit 1",
            shQuote(stat_binary), shQuote(normalized)),
    "dev=$((16#$hex))",
    "maj=$((((dev >> 8) & 0xfff) | ((dev >> 32) & 0xfffff000)))",
    "min=$(((dev & 0xff) | ((dev >> 12) & 0xffffff00)))",
    sprintf("printf '%%x,%%x,%%s,%%s\\n' \"$maj\" \"$min\" \"$ino\" %s",
            shQuote(normalized)),
    sep = "; "
  )
  output <- tryCatch(
    system2(
      bash_binary,
      c("-lc", shQuote(stat_script)),
      stdout = TRUE, stderr = TRUE
    ),
    error = function(error) stop(
      "native library file identity stat failed: ",
      conditionMessage(error), call. = FALSE
    )
  )
  if (!identical(attr(output, "status", exact = TRUE), NULL) ||
      length(output) != 1L || is.na(output[[1L]]) ||
      !grepl("^[0-9a-fA-F]+,[0-9a-fA-F]+,[0-9]+,", output[[1L]])) {
    stop("native library file identity stat output is malformed",
         call. = FALSE)
  }
  fields <- strsplit(output[[1L]], ",", fixed = TRUE)[[1L]]
  if (length(fields) < 4L) {
    stop("native library file identity stat output is malformed",
         call. = FALSE)
  }
  list(
    path = normalized,
    device_major_hex = .fastkpc_cuda_normalize_hex_identity(fields[[1L]]),
    device_minor_hex = .fastkpc_cuda_normalize_hex_identity(fields[[2L]]),
    inode = fields[[3L]]
  )
}

.fastkpc_cuda_registered_identity_cache <- new.env(parent = emptyenv())
.fastkpc_cuda_qualified_pin_cache <- new.env(parent = emptyenv())

.fastkpc_cuda_live_owner_snapshot <- function(dlls = getLoadedDLLs,
                                              call = .Call) {
  if (!is.function(dlls) || !is.function(call)) {
    stop("fixed-SP CUDA owner snapshot callbacks are malformed",
         call. = FALSE)
  }
  loaded <- dlls()
  if (is.null(loaded[["fastkpc_cuda"]])) {
    return(list(runtime = 0L, prepared = 0L, residual = 0L, total = 0L))
  }
  snapshot <- tryCatch(
    call("C_fixed_sp_cuda_live_owner_snapshot", PACKAGE = "fastkpc_cuda"),
    error = function(error) stop(
      "fixed-SP CUDA owner snapshot is unavailable: ",
      conditionMessage(error), call. = FALSE
    )
  )
  required <- c("runtime", "prepared", "residual", "total")
  if (!is.list(snapshot) || is.object(snapshot) ||
      !identical(names(snapshot), required)) {
    stop("fixed-SP CUDA owner snapshot is malformed", call. = FALSE)
  }
  clean <- lapply(required, function(name) {
    value <- snapshot[[name]]
    if (typeof(value) == "double" && length(value) == 1L &&
        !is.object(value) && is.null(attributes(value)) && !anyNA(value) &&
        is.finite(value) && value >= 0 && value <= .Machine$integer.max &&
        identical(value, floor(value))) {
      return(as.integer(value))
    }
    if (typeof(value) == "integer" && length(value) == 1L &&
        !is.object(value) && is.null(attributes(value)) && !anyNA(value) &&
        value >= 0L) {
      return(value)
    }
    stop("fixed-SP CUDA owner snapshot is malformed", call. = FALSE)
  })
  names(clean) <- required
  if (!identical(
        clean$total,
        as.integer(clean$runtime + clean$prepared + clean$residual)
      )) {
    stop("fixed-SP CUDA owner snapshot total is inconsistent",
         call. = FALSE)
  }
  clean
}

.fastkpc_cuda_assert_no_live_fixed_sp_owners <- function(
    snapshot = .fastkpc_cuda_live_owner_snapshot()) {
  required <- c("runtime", "prepared", "residual", "total")
  if (!is.list(snapshot) || !identical(names(snapshot), required)) {
    stop("fixed-SP CUDA owner snapshot is malformed", call. = FALSE)
  }
  total <- snapshot$total
  if (typeof(total) != "integer" || length(total) != 1L ||
      is.object(total) || !is.null(attributes(total)) || anyNA(total) ||
      total < 0L) {
    stop("fixed-SP CUDA owner snapshot is malformed", call. = FALSE)
  }
  if (total != 0L) {
    stop(
      "live fixed-SP CUDA external pointers block qualified native ",
      "unload/rebuild: runtime=", snapshot$runtime,
      ", prepared=", snapshot$prepared,
      ", residual=", snapshot$residual,
      call. = FALSE
    )
  }
  invisible(TRUE)
}

.fastkpc_cuda_empty_mapped_records <- function() {
  data.frame(
    path = character(), live_path = character(), deleted = logical(),
    device_major_hex = character(), device_minor_hex = character(),
    inode = character(), stringsAsFactors = FALSE
  )
}

.fastkpc_cuda_pin_root_dirs <- function(
    roots = dirname(.fastkpc_cuda_so())) {
  if (typeof(roots) != "character" || is.object(roots) || anyNA(roots) ||
      length(roots) == 0L) {
    stop("qualified CUDA pin cleanup roots are malformed", call. = FALSE)
  }
  roots <- unique(unname(vapply(
    roots, normalizePath, character(1L), winslash = "/", mustWork = TRUE
  )))
  roots[!grepl("(^|/)artifacts($|/)", roots)]
}

.fastkpc_cuda_qualified_pin_dirs <- function(
    roots = dirname(.fastkpc_cuda_so())) {
  roots <- .fastkpc_cuda_pin_root_dirs(roots)
  dirs <- unlist(lapply(roots, function(root) {
    list.files(
      root, pattern = "^\\.fastkpc_cuda-qualified-", all.files = TRUE,
      full.names = TRUE, recursive = FALSE, no.. = TRUE
    )
  }), use.names = FALSE)
  if (length(dirs) == 0L) return(character())
  dirs <- dirs[dir.exists(dirs)]
  sort(unique(unname(vapply(
    dirs, normalizePath, character(1L), winslash = "/", mustWork = TRUE
  ))), method = "radix")
}

.fastkpc_cuda_validate_pin_path <- function(path) {
  if (typeof(path) != "character" || length(path) != 1L ||
      is.object(path) || !is.null(attributes(path)) || anyNA(path) ||
      !nzchar(path) || !file.exists(path) || dir.exists(path)) {
    stop("qualified CUDA pin path is malformed", call. = FALSE)
  }
  path <- normalizePath(path, winslash = "/", mustWork = TRUE)
  parent <- basename(dirname(path))
  if (!identical(basename(path), "fastkpc_cuda.so") ||
      !grepl("^\\.fastkpc_cuda-qualified-", parent) ||
      grepl("(^|/)artifacts($|/)", path)) {
    stop("qualified CUDA pin path is outside the cleanup boundary",
         call. = FALSE)
  }
  path
}

.fastkpc_cuda_register_pin_cleanup_finalizer <- function() {
  cache <- .fastkpc_cuda_qualified_pin_cache
  if (isTRUE(cache$finalizer_registered)) return(invisible(TRUE))
  reg.finalizer(
    cache,
    function(env) {
      try(.fastkpc_cuda_cleanup_remembered_pins_at_exit(), silent = TRUE)
      invisible(NULL)
    },
    onexit = TRUE
  )
  cache$finalizer_registered <- TRUE
  invisible(TRUE)
}

.fastkpc_cuda_remember_pinned_library <- function(path, sha256) {
  path <- .fastkpc_cuda_validate_pin_path(path)
  if (typeof(sha256) != "character" || length(sha256) != 1L ||
      is.object(sha256) || !is.null(attributes(sha256)) || anyNA(sha256) ||
      !grepl("^[0-9a-f]{64}$", sha256)) {
    stop("qualified CUDA pin hash is malformed", call. = FALSE)
  }
  .fastkpc_cuda_register_pin_cleanup_finalizer()
  cache <- .fastkpc_cuda_qualified_pin_cache
  paths <- if (exists("paths", envir = cache, inherits = FALSE)) {
    cache$paths
  } else character()
  cache$paths <- sort(unique(c(paths, path)), method = "radix")
  cache$sha256 <- c(
    if (exists("sha256", envir = cache, inherits = FALSE)) cache$sha256
    else setNames(character(), character()),
    setNames(sha256, path)
  )
  invisible(path)
}

.fastkpc_cuda_forget_pinned_library <- function(path) {
  path <- normalizePath(path, winslash = "/", mustWork = FALSE)
  cache <- .fastkpc_cuda_qualified_pin_cache
  if (exists("paths", envir = cache, inherits = FALSE)) {
    cache$paths <- setdiff(cache$paths, path)
  }
  if (exists("sha256", envir = cache, inherits = FALSE)) {
    cache$sha256 <- cache$sha256[names(cache$sha256) != path]
  }
  invisible(path)
}

.fastkpc_cuda_pin_cleanup_dir <- function(path) {
  if (typeof(path) != "character" || length(path) != 1L ||
      is.object(path) || !is.null(attributes(path)) || anyNA(path) ||
      !nzchar(path)) {
    stop("qualified CUDA pin cleanup path is malformed", call. = FALSE)
  }
  path <- normalizePath(path, winslash = "/", mustWork = FALSE)
  parent <- basename(dirname(path))
  if (!identical(basename(path), "fastkpc_cuda.so") ||
      !grepl("^\\.fastkpc_cuda-qualified-", parent) ||
      grepl("(^|/)artifacts($|/)", path)) {
    stop("qualified CUDA pin cleanup path is outside the cleanup boundary",
         call. = FALSE)
  }
  dirname(path)
}

.fastkpc_cuda_cleanup_remembered_pins_at_exit <- function() {
  cache <- .fastkpc_cuda_qualified_pin_cache
  paths <- if (exists("paths", envir = cache, inherits = FALSE)) {
    cache$paths
  } else character()
  if (length(paths) == 0L) return(character())
  dirs <- sort(unique(unname(vapply(
    paths, .fastkpc_cuda_pin_cleanup_dir, character(1L)
  ))), method = "radix")
  removed <- character()
  for (dir in dirs) {
    if (!dir.exists(dir)) next
    unlink(dir, recursive = TRUE, force = TRUE)
    if (!dir.exists(dir)) removed <- c(removed, dir)
  }
  if (length(removed) > 0L) {
    for (dir in removed) {
      prefix <- paste0(dir, "/")
      if (exists("paths", envir = cache, inherits = FALSE)) {
        cache$paths <- cache$paths[!startsWith(cache$paths, prefix)]
      }
      if (exists("sha256", envir = cache, inherits = FALSE)) {
        cache$sha256 <- cache$sha256[!startsWith(names(cache$sha256), prefix)]
      }
    }
  }
  sort(unique(removed), method = "radix")
}

.fastkpc_cuda_clear_registered_identity_if_path <- function(path) {
  path <- normalizePath(path, winslash = "/", mustWork = FALSE)
  cache <- .fastkpc_cuda_registered_identity_cache
  if (exists("path", envir = cache, inherits = FALSE) &&
      identical(normalizePath(cache$path, winslash = "/", mustWork = FALSE),
                path)) {
    rm(list = intersect(c("path", "sha256"), ls(cache, all.names = TRUE)),
       envir = cache)
  }
  invisible(path)
}

.fastkpc_cuda_prune_qualified_pins <- function(
    roots = dirname(.fastkpc_cuda_so()),
    loaded_paths = .fastkpc_cuda_loaded_paths,
    mapped_records = .fastkpc_cuda_mapped_object_records) {
  if (!is.function(loaded_paths) || !is.function(mapped_records)) {
    stop("qualified CUDA pin cleanup callbacks are malformed", call. = FALSE)
  }
  roots <- .fastkpc_cuda_pin_root_dirs(roots)
  candidates <- .fastkpc_cuda_qualified_pin_dirs(roots)
  if (length(candidates) == 0L) return(character())
  loaded <- .fastkpc_cuda_normalize_path_snapshot(
    loaded_paths(), "loaded CUDA DLL path snapshot"
  )
  records <- mapped_records()
  required_record_fields <- c(
    "path", "live_path", "deleted", "device_major_hex",
    "device_minor_hex", "inode"
  )
  if (!is.data.frame(records) ||
      !all(required_record_fields %in% names(records))) {
    stop("mapped CUDA object record snapshot is malformed", call. = FALSE)
  }
  mapped_live <- records$live_path[!is.na(records$live_path)]
  deleted_rows <- !is.na(records$deleted) & records$deleted
  mapped_deleted <- sub(
    " \\(deleted\\)$", "", records$path[deleted_rows]
  )
  protected <- unique(c(loaded, mapped_live, mapped_deleted))
  protected <- protected[!is.na(protected) & nzchar(protected)]
  protected <- normalizePath(protected, winslash = "/", mustWork = FALSE)
  removed <- character()
  for (dir in candidates) {
    if (!grepl("^\\.fastkpc_cuda-qualified-", basename(dir))) {
      next
    }
    parent <- normalizePath(dirname(dir), winslash = "/", mustWork = TRUE)
    if (!parent %in% roots) next
    prefix <- paste0(dir, "/")
    active <- any(protected == dir | startsWith(protected, prefix))
    if (active) next
    unlink(dir, recursive = TRUE, force = TRUE)
    if (!dir.exists(dir)) {
      removed <- c(removed, dir)
      cache <- .fastkpc_cuda_qualified_pin_cache
      if (exists("paths", envir = cache, inherits = FALSE)) {
        stale <- startsWith(cache$paths, prefix)
        cache$paths <- cache$paths[!stale]
      }
      if (exists("sha256", envir = cache, inherits = FALSE)) {
        stale <- startsWith(names(cache$sha256), prefix)
        cache$sha256 <- cache$sha256[!stale]
      }
    }
  }
  sort(unique(removed), method = "radix")
}

.fastkpc_cuda_phase3_symbol_names <- function() {
  c(
    "C_fastkpc_cuda_phase3_environment_identity",
    "C_fixed_sp_cuda_runtime_create",
    "C_fixed_sp_cuda_runtime_info"
  )
}

.fastkpc_cuda_verify_registered_library_identity <- function(
    path, expected_sha256, loaded_dlls = getLoadedDLLs,
    mapped_records = .fastkpc_cuda_mapped_object_records,
    symbol_info = getNativeSymbolInfo,
    file_identity = .fastkpc_cuda_posix_file_identity,
    hash_file = .fastkpc_cuda_sha256_file) {
  if (!is.function(loaded_dlls) || !is.function(mapped_records) ||
      !is.function(symbol_info) || !is.function(file_identity) ||
      !is.function(hash_file) ||
      typeof(expected_sha256) != "character" ||
      length(expected_sha256) != 1L || is.object(expected_sha256) ||
      !is.null(attributes(expected_sha256)) || anyNA(expected_sha256) ||
      !grepl("^[0-9a-f]{64}$", expected_sha256)) {
    stop("loaded native library identity is malformed", call. = FALSE)
  }
  native_identity <- file_identity(path)
  current_sha256 <- hash_file(native_identity$path)
  dlls <- loaded_dlls()
  registered <- dlls[["fastkpc_cuda"]]
  if (is.null(registered) || is.null(registered[["path"]])) {
    stop("registered fastkpc_cuda package entry is missing", call. = FALSE)
  }
  registered_identity <- file_identity(registered[["path"]])
  if (!identical(registered_identity$path, native_identity$path)) {
    stop("registered fastkpc_cuda package path mismatch", call. = FALSE)
  }
  same_file_identity <- function(left, right) {
    identical(left$path, right$path) &&
      identical(
        .fastkpc_cuda_normalize_hex_identity(left$device_major_hex),
        .fastkpc_cuda_normalize_hex_identity(right$device_major_hex)
      ) &&
      identical(
        .fastkpc_cuda_normalize_hex_identity(left$device_minor_hex),
        .fastkpc_cuda_normalize_hex_identity(right$device_minor_hex)
      ) &&
      identical(as.character(left$inode), as.character(right$inode))
  }
  if (!same_file_identity(registered_identity, native_identity)) {
    stop("registered fastkpc_cuda package inode mismatch", call. = FALSE)
  }
  records <- mapped_records()
  required_record_fields <- c(
    "path", "live_path", "deleted", "device_major_hex",
    "device_minor_hex", "inode"
  )
  if (!is.data.frame(records) || !all(required_record_fields %in% names(records))) {
    stop("mapped native library record snapshot is malformed", call. = FALSE)
  }
  deleted_record <- paste0(native_identity$path, " (deleted)")
  if (deleted_record %in% records$path) {
    stop("exact qualified native library path is mapped only as deleted record",
         call. = FALSE)
  }
  live_rows <- records$live_path == native_identity$path
  live_rows[is.na(live_rows)] <- FALSE
  if (!any(live_rows)) {
    stop("exact qualified native library path is not mapped", call. = FALSE)
  }
  live_exact_rows <- live_rows & !is.na(records$deleted) & !records$deleted
  missing_identity_rows <- live_exact_rows & (
    is.na(records$device_major_hex) |
      is.na(records$device_minor_hex) |
      is.na(records$inode)
  )
  if (any(missing_identity_rows) || !any(live_exact_rows)) {
    stop("exact qualified native library mapped inode identity is missing",
         call. = FALSE)
  }
  record_major <- .fastkpc_cuda_normalize_hex_identity(
    records$device_major_hex
  )
  record_minor <- .fastkpc_cuda_normalize_hex_identity(
    records$device_minor_hex
  )
  native_major <- .fastkpc_cuda_normalize_hex_identity(
    native_identity$device_major_hex
  )
  native_minor <- .fastkpc_cuda_normalize_hex_identity(
    native_identity$device_minor_hex
  )
  identity_rows <- live_exact_rows &
    record_major == native_major &
    record_minor == native_minor &
    as.character(records$inode) == as.character(native_identity$inode)
  if (!any(identity_rows)) {
    stop("exact qualified native library mapped inode mismatch",
         call. = FALSE)
  }
  for (symbol_name in .fastkpc_cuda_phase3_symbol_names()) {
    info <- symbol_info(
      symbol_name, PACKAGE = "fastkpc_cuda", withRegistrationInfo = TRUE
    )
    if (!is.list(info) || is.null(info$dll) || is.null(info$dll[["path"]])) {
      stop("phase3 native symbol binding is malformed", call. = FALSE)
    }
    symbol_identity <- tryCatch(
      file_identity(info$dll[["path"]]),
      error = function(error) stop(
        "phase3 native symbol DLL binding mismatch: ",
        conditionMessage(error), call. = FALSE
      )
    )
    if (!same_file_identity(symbol_identity, native_identity)) {
      stop("phase3 native symbol DLL binding mismatch", call. = FALSE)
    }
  }
  if (!identical(current_sha256, expected_sha256)) {
    stop("qualified native library bytes changed", call. = FALSE)
  }
  TRUE
}

.fastkpc_cuda_remember_identity <- function(path, sha256) {
  cache <- .fastkpc_cuda_registered_identity_cache
  cache$path <- normalizePath(path, winslash = "/", mustWork = TRUE)
  cache$sha256 <- sha256
  invisible(cache$path)
}

.fastkpc_cuda_unload_exact_for_rebuild <- function(
    so, loaded_paths = .fastkpc_cuda_loaded_paths,
    mapped_paths = NULL,
    mapped_records = .fastkpc_cuda_mapped_object_records,
    unload = dyn.unload) {
  if (!is.function(loaded_paths) || !is.function(unload) ||
      (!is.null(mapped_paths) && !is.function(mapped_paths)) ||
      !is.function(mapped_records)) {
    stop("CUDA rebuild unload callbacks are malformed", call. = FALSE)
  }
  .fastkpc_cuda_assert_no_live_fixed_sp_owners()
  path <- normalizePath(so, winslash = "/", mustWork = FALSE)
  before <- .fastkpc_cuda_normalize_path_snapshot(
    loaded_paths(), "loaded CUDA DLL path snapshot"
  )
  if (is.null(mapped_paths)) {
    mapped_before_records <- mapped_records()
    mapped_before <- mapped_before_records$live_path[
      !mapped_before_records$deleted
    ]
    mapped_before <- mapped_before[!is.na(mapped_before)]
    deleted_before <- paste0(path, " (deleted)") %in%
      mapped_before_records$path
  } else {
    mapped_before <- .fastkpc_cuda_normalize_path_snapshot(
      mapped_paths(), "mapped CUDA object path snapshot"
    )
    deleted_before <- FALSE
  }
  if (path %in% mapped_before && !path %in% before) {
    stop(
      "exact old CUDA DLL remains mapped without R registration; ",
      "start a fresh R process", call. = FALSE
    )
  }
  if (deleted_before && !path %in% before) {
    stop(
      "exact old CUDA DLL remains mapped without R registration; ",
      "start a fresh R process", call. = FALSE
    )
  }
  if (path %in% before) {
    .fastkpc_cuda_assert_no_live_fixed_sp_owners()
    tryCatch(
      unload(path),
      error = function(error) stop(
        "failed to unload exact CUDA DLL before rebuild: ",
        conditionMessage(error), call. = FALSE
      )
    )
  }
  after <- .fastkpc_cuda_normalize_path_snapshot(
    loaded_paths(), "loaded CUDA DLL path snapshot"
  )
  if (is.null(mapped_paths)) {
    mapped_after_records <- mapped_records()
    mapped_after <- mapped_after_records$live_path[
      !mapped_after_records$deleted
    ]
    mapped_after <- mapped_after[!is.na(mapped_after)]
    deleted_after <- paste0(path, " (deleted)") %in%
      mapped_after_records$path
  } else {
    mapped_after <- .fastkpc_cuda_normalize_path_snapshot(
      mapped_paths(), "mapped CUDA object path snapshot"
    )
    deleted_after <- FALSE
  }
  if (path %in% after) {
    stop("exact old CUDA DLL remains loaded after rebuild unload",
         call. = FALSE)
  }
  if (path %in% mapped_after || deleted_after) {
    stop(
      "exact old CUDA DLL remains mapped after rebuild unload; ",
      "start a fresh R process", call. = FALSE
    )
  }
  invisible(path)
}

.fastkpc_cuda_resolve_strace <- function(candidate = Sys.which("strace")) {
  if (typeof(candidate) != "character" || length(candidate) != 1L ||
      is.object(candidate) || anyNA(candidate)) {
    stop("strace is required for qualified CUDA native rebuild",
         call. = FALSE)
  }
  candidate <- unname(candidate)
  if (!nzchar(candidate) || !file.exists(candidate) || dir.exists(candidate)) {
    stop("strace is required for qualified CUDA native rebuild",
         call. = FALSE)
  }
  normalizePath(candidate, winslash = "/", mustWork = TRUE)
}

.fastkpc_cuda_trace_invocation <- function(tracer_path) {
  paste(
    "LC_ALL=C", tracer_path, "-f -qq -yy -s 65535",
    "-e trace=%file",
    "-e status=successful -o <trace> bash <build-script>"
  )
}

.fastkpc_cuda_sha256_file <- function(path) {
  if (!requireNamespace("digest", quietly = TRUE)) {
    stop("digest is required for qualified CUDA native loading",
         call. = FALSE)
  }
  if (!is.character(path) || length(path) != 1L || is.na(path) ||
      !nzchar(path) || !file.exists(path) || dir.exists(path)) {
    stop("qualified CUDA native hash path is invalid", call. = FALSE)
  }
  unname(digest::digest(
    file = path, algo = "sha256", serialize = FALSE
  ))
}

.fastkpc_cuda_pin_built_library <- function(
    so, expected_sha256, hash_file = .fastkpc_cuda_sha256_file) {
  path <- normalizePath(so, winslash = "/", mustWork = TRUE)
  staging_dir <- tempfile(
    pattern = ".fastkpc_cuda-qualified-", tmpdir = dirname(path)
  )
  if (!dir.create(
        staging_dir, recursive = FALSE, showWarnings = FALSE, mode = "0700"
      )) {
    stop("failed to create qualified CUDA native snapshot directory",
         call. = FALSE)
  }
  pinned <- FALSE
  on.exit({
    if (!pinned && dir.exists(staging_dir)) {
      unlink(staging_dir, recursive = TRUE, force = TRUE)
    }
  }, add = TRUE)
  pinned_path <- file.path(staging_dir, "fastkpc_cuda.so")
  if (!identical(file.link(path, pinned_path), TRUE)) {
    stop("failed to pin qualified CUDA native library", call. = FALSE)
  }
  pinned_path <- normalizePath(
    pinned_path, winslash = "/", mustWork = TRUE
  )
  if (!identical(hash_file(pinned_path), expected_sha256)) {
    stop("pinned CUDA DLL does not match qualified build hash",
         call. = FALSE)
  }
  pinned <- TRUE
  pinned_path
}

.fastkpc_cuda_load_built_library_exact <- function(
    so, expected_sha256, hash_file = .fastkpc_cuda_sha256_file,
    loaded_paths = .fastkpc_cuda_loaded_paths,
    mapped_paths = NULL,
    mapped_records = .fastkpc_cuda_mapped_object_records,
    load = dyn.load, unload = dyn.unload) {
  if (!is.function(hash_file) || !is.function(loaded_paths) ||
      (!is.null(mapped_paths) && !is.function(mapped_paths)) ||
      !is.function(mapped_records) || !is.function(load) || !is.function(unload) ||
      typeof(expected_sha256) != "character" ||
      length(expected_sha256) != 1L || is.object(expected_sha256) ||
      !is.null(attributes(expected_sha256)) || anyNA(expected_sha256) ||
      !grepl("^[0-9a-f]{64}$", expected_sha256)) {
    stop("qualified CUDA native load identity is malformed", call. = FALSE)
  }
  path <- normalizePath(so, winslash = "/", mustWork = TRUE)
  if (!identical(hash_file(path), expected_sha256)) {
    stop("built CUDA DLL changed before dyn.load", call. = FALSE)
  }
  before <- .fastkpc_cuda_normalize_path_snapshot(
    loaded_paths(), "loaded CUDA DLL path snapshot"
  )
  if (is.null(mapped_paths)) {
    mapped_before_records <- mapped_records()
    mapped_before <- mapped_before_records$live_path[
      !mapped_before_records$deleted
    ]
    mapped_before <- mapped_before[!is.na(mapped_before)]
    deleted_before <- paste0(path, " (deleted)") %in%
      mapped_before_records$path
  } else {
    mapped_before <- .fastkpc_cuda_normalize_path_snapshot(
      mapped_paths(), "mapped CUDA object path snapshot"
    )
    deleted_before <- FALSE
  }
  if (path %in% before) {
    stop("exact built CUDA DLL path was already loaded", call. = FALSE)
  }
  if (path %in% mapped_before || deleted_before) {
    stop(
      "exact built CUDA DLL path was already mapped; start a fresh R process",
      call. = FALSE
    )
  }
  loaded <- FALSE
  tryCatch({
    load(path)
    loaded <- TRUE
    after <- .fastkpc_cuda_normalize_path_snapshot(
      loaded_paths(), "loaded CUDA DLL path snapshot"
    )
    if (is.null(mapped_paths)) {
      mapped_after_records <- mapped_records()
      mapped_after <- mapped_after_records$live_path[
        !mapped_after_records$deleted
      ]
      mapped_after <- mapped_after[!is.na(mapped_after)]
    } else {
      mapped_after <- .fastkpc_cuda_normalize_path_snapshot(
        mapped_paths(), "mapped CUDA object path snapshot"
      )
    }
    if (!path %in% after) {
      stop("exact built DLL path is not loaded after dyn.load", call. = FALSE)
    }
    if (!path %in% mapped_after) {
      stop("exact built DLL path is not mapped after dyn.load", call. = FALSE)
    }
    if (!identical(hash_file(path), expected_sha256)) {
      stop("built CUDA DLL changed while loading", call. = FALSE)
    }
    invisible(path)
  }, error = function(error) {
    if (loaded) {
      tryCatch(unload(path), error = function(unload_error) NULL)
    }
    stop(conditionMessage(error), call. = FALSE)
  })
}

.fastkpc_cuda_pin_and_load_built_library <- function(
    so, expected_sha256, hash_file = .fastkpc_cuda_sha256_file,
    loaded_paths = .fastkpc_cuda_loaded_paths,
    mapped_paths = NULL,
    mapped_records = .fastkpc_cuda_mapped_object_records,
    load = dyn.load, unload = dyn.unload) {
  pinned_path <- .fastkpc_cuda_pin_built_library(
    so, expected_sha256 = expected_sha256, hash_file = hash_file
  )
  loaded <- FALSE
  on.exit({
    if (!loaded && dir.exists(dirname(pinned_path))) {
      unlink(dirname(pinned_path), recursive = TRUE, force = TRUE)
    }
  }, add = TRUE)
  .fastkpc_cuda_load_built_library_exact(
    pinned_path,
    expected_sha256 = expected_sha256,
    hash_file = hash_file,
    loaded_paths = loaded_paths,
    mapped_paths = mapped_paths,
    mapped_records = mapped_records,
    load = load,
    unload = unload
  )
  loaded <- TRUE
  .fastkpc_cuda_remember_pinned_library(pinned_path, expected_sha256)
  invisible(pinned_path)
}

.fastkpc_cuda_rollback_qualified_native_load <- function(
    native_load, loaded_paths = .fastkpc_cuda_loaded_paths,
    mapped_records = .fastkpc_cuda_mapped_object_records,
    unload = dyn.unload) {
  path <- if (is.list(native_load)) {
    native_load$native_library_path
  } else native_load
  path <- .fastkpc_cuda_validate_pin_path(path)
  .fastkpc_cuda_unload_exact_for_rebuild(
    path,
    loaded_paths = loaded_paths,
    mapped_records = mapped_records,
    unload = unload
  )
  .fastkpc_cuda_clear_registered_identity_if_path(path)
  removed <- .fastkpc_cuda_prune_qualified_pins(
    roots = dirname(dirname(path)),
    loaded_paths = loaded_paths,
    mapped_records = mapped_records
  )
  .fastkpc_cuda_forget_pinned_library(path)
  invisible(removed)
}

build_fastkpc_cuda_native <- function(
    rebuild = FALSE, trace_path = NULL, tracer_path = NULL) {
  root <- .fastkpc_cuda_root()
  so <- .fastkpc_cuda_so()
  if (rebuild) {
    .fastkpc_cuda_unload_exact_for_rebuild(so)
  }
  if (rebuild || !file.exists(so)) {
    script <- file.path(root, "tools", "build_cuda_native.sh")
    if (!file.exists(script)) {
      stop("Cannot find CUDA build script: ", script, call. = FALSE)
    }
    if (is.null(trace_path)) {
      if (!is.null(tracer_path)) {
        stop("CUDA native tracer requires a trace output path",
             call. = FALSE)
      }
      status <- system2("bash", script)
    } else {
      if (!rebuild || !is.character(trace_path) ||
          length(trace_path) != 1L || is.na(trace_path) ||
          !nzchar(trace_path) || dir.exists(trace_path)) {
        stop("qualified CUDA native trace path is malformed",
             call. = FALSE)
      }
      trace_parent <- normalizePath(
        dirname(trace_path), winslash = "/", mustWork = TRUE
      )
      trace_path <- file.path(trace_parent, basename(trace_path))
      tracer_path <- .fastkpc_cuda_resolve_strace(tracer_path)
      status <- system2(
        tracer_path,
        c(
          "-f", "-qq", "-yy", "-s", "65535",
          "-e", "trace=%file",
          "-e", "status=successful", "-o", shQuote(trace_path),
          "bash", shQuote(script)
        ),
        env = "LC_ALL=C"
      )
    }
    if (!identical(status, 0L)) {
      stop("CUDA native build failed with status ", status, call. = FALSE)
    }
    if (!is.null(trace_path) &&
        (!file.exists(trace_path) || dir.exists(trace_path))) {
      stop("qualified CUDA native build trace was not produced",
           call. = FALSE)
    }
  }
  normalizePath(so, winslash = "/", mustWork = TRUE)
}

load_fastkpc_cuda_native <- function(rebuild = FALSE) {
  if (!requireNamespace("Rcpp", quietly = TRUE)) {
    stop("Rcpp is required to load fastkpc CUDA native code", call. = FALSE)
  }
  if (!rebuild) {
    registered <- getLoadedDLLs()[["fastkpc_cuda"]]
    if (!is.null(registered)) {
      registered_path <- normalizePath(
        registered[["path"]], winslash = "/", mustWork = TRUE
      )
      cache <- .fastkpc_cuda_registered_identity_cache
      if (exists("path", envir = cache, inherits = FALSE) &&
          exists("sha256", envir = cache, inherits = FALSE) &&
          identical(registered_path, cache$path) &&
          typeof(cache$sha256) == "character" &&
          length(cache$sha256) == 1L && !is.object(cache$sha256) &&
          is.null(attributes(cache$sha256)) && !anyNA(cache$sha256) &&
          grepl("^[0-9a-f]{64}$", cache$sha256)) {
        return(invisible(registered_path))
      }
      canonical_so <- normalizePath(
        .fastkpc_cuda_so(), winslash = "/", mustWork = FALSE
      )
      if (file.exists(canonical_so) && identical(registered_path, canonical_so)) {
        canonical_sha256 <- .fastkpc_cuda_sha256_file(canonical_so)
        .fastkpc_cuda_verify_registered_library_identity(
          registered_path, canonical_sha256
        )
        .fastkpc_cuda_remember_identity(registered_path, canonical_sha256)
        return(invisible(registered_path))
      }
      stop("registered fastkpc_cuda package does not match current native identity",
           call. = FALSE)
    }
  }
  so <- build_fastkpc_cuda_native(rebuild = rebuild)
  loaded <- vapply(getLoadedDLLs(), function(dll) normalizePath(dll[["path"]],
                                                               mustWork = FALSE),
                   character(1))
  if (!normalizePath(so, mustWork = FALSE) %in% loaded) {
    dyn.load(so)
  }
  .fastkpc_cuda_remember_identity(so, .fastkpc_cuda_sha256_file(so))
  invisible(so)
}

load_fastkpc_cuda_native_qualified <- function(
    trace_path, tracer_path = Sys.which("strace")) {
  if (!requireNamespace("Rcpp", quietly = TRUE)) {
    stop("Rcpp is required to load fastkpc CUDA native code", call. = FALSE)
  }
  tracer_path <- .fastkpc_cuda_resolve_strace(tracer_path)
  so <- build_fastkpc_cuda_native(
    rebuild = TRUE, trace_path = trace_path, tracer_path = tracer_path
  )
  built_sha256 <- .fastkpc_cuda_sha256_file(so)
  pinned_so <- NULL
  tryCatch({
    pinned_so <- .fastkpc_cuda_pin_and_load_built_library(
      so, expected_sha256 = built_sha256
    )
    .fastkpc_cuda_verify_registered_library_identity(
      pinned_so, built_sha256
    )
    .fastkpc_cuda_remember_identity(pinned_so, built_sha256)
    list(
      native_library_path = pinned_so,
      native_library_sha256 = built_sha256,
      trace_path = normalizePath(
        trace_path, winslash = "/", mustWork = TRUE
      ),
      tracer_path = tracer_path,
      trace_invocation = .fastkpc_cuda_trace_invocation(tracer_path)
    )
  }, error = function(error) {
    if (!is.null(pinned_so)) {
      rollback <- tryCatch(
        .fastkpc_cuda_rollback_qualified_native_load(list(
          native_library_path = pinned_so,
          native_library_sha256 = built_sha256
        )),
        error = identity
      )
      if (inherits(rollback, "error")) {
        stop(
          conditionMessage(error),
          "; qualified native rollback failed: ",
          conditionMessage(rollback),
          call. = FALSE
        )
      }
    }
    stop(conditionMessage(error), call. = FALSE)
  })
}

fastkpc_cuda_available <- function() {
  load_fastkpc_cuda_native()
  isTRUE(.Call("C_fastkpc_cuda_available", PACKAGE = "fastkpc_cuda"))
}

fastkpc_cuda_device_info <- function() {
  load_fastkpc_cuda_native()
  .Call("C_fastkpc_cuda_device_info", PACKAGE = "fastkpc_cuda")
}

fastkpc_cuda_phase3_environment_identity <- function(device_id) {
  if (typeof(device_id) != "integer" || length(device_id) != 1L ||
      is.object(device_id) || !is.null(attributes(device_id)) ||
      is.na(device_id) || device_id < 0L) {
    stop("device_id must be one non-negative integer", call. = FALSE)
  }
  load_fastkpc_cuda_native()
  .Call(
    "C_fastkpc_cuda_phase3_environment_identity", device_id,
    PACKAGE = "fastkpc_cuda"
  )
}

fastkpc_cuda_legacy_dcov_gamma_cpp_oracle <- function(
    x, y, numCol = as.integer(floor(length(x) / 10)), index = 1) {
  bare_double_vector <- function(value) {
    typeof(value) == "double" && length(value) > 5L &&
      !is.object(value) && is.null(attributes(value)) &&
      all(is.finite(value))
  }
  num_col_clean <- typeof(numCol) == "integer" && length(numCol) == 1L &&
    !is.object(numCol) && is.null(attributes(numCol)) &&
    !is.na(numCol) && numCol > 0L && numCol < length(x)
  index_clean <- typeof(index) %in% c("integer", "double") &&
    length(index) == 1L &&
    !is.object(index) && is.null(attributes(index)) &&
    is.finite(index) && index >= 0 && index <= 2
  if (!bare_double_vector(x) || !bare_double_vector(y) ||
      !identical(length(x), length(y)) || !isTRUE(num_col_clean) ||
      !isTRUE(index_clean)) {
    stop("registered legacy dCov gamma inputs are malformed", call. = FALSE)
  }
  index <- as.double(index)
  load_fastkpc_cuda_native()
  .Call(
    "C_fastkpc_cuda_legacy_dcov_gamma_cpp_oracle",
    x, y, numCol, index, PACKAGE = "fastkpc_cuda"
  )
}

legacy_dcov_spectra_matvec_cuda <- function(a, rhs) {
  load_fastkpc_cuda_native()
  a <- as.matrix(a)
  storage.mode(a) <- "double"
  rhs_was_vector <- is.null(dim(rhs))
  rhs <- if (rhs_was_vector) {
    matrix(as.numeric(rhs), ncol = 1L)
  } else {
    as.matrix(rhs)
  }
  storage.mode(rhs) <- "double"
  if (nrow(a) != ncol(a)) {
    stop("matrix must be square", call. = FALSE)
  }
  if (nrow(rhs) != nrow(a)) {
    stop("rhs row count must match matrix dimension", call. = FALSE)
  }
  if (!all(is.finite(a)) || !all(is.finite(rhs))) {
    stop("Data contains missing or infinite values", call. = FALSE)
  }
  result <- .Call("C_legacy_dcov_spectra_matvec_cuda", a, rhs,
                  PACKAGE = "fastkpc_cuda")
  if (rhs_was_vector) {
    result$values <- as.numeric(result$values[, 1L])
  }
  result
}

legacy_dcov_spectra_matvec_cuda_handle <- function(a) {
  load_fastkpc_cuda_native()
  a <- as.matrix(a)
  storage.mode(a) <- "double"
  if (nrow(a) != ncol(a)) {
    stop("matrix must be square", call. = FALSE)
  }
  if (!all(is.finite(a))) {
    stop("Data contains missing or infinite values", call. = FALSE)
  }
  handle <- .Call("C_legacy_dcov_spectra_matvec_cuda_handle_create", a,
                  PACKAGE = "fastkpc_cuda")
  class(handle) <- c("fastkpc_cuda_matvec_handle", class(handle))
  handle
}

legacy_dcov_spectra_matvec_cuda_handle_apply <- function(handle, rhs) {
  load_fastkpc_cuda_native()
  if (!is.list(handle) || is.null(handle$ptr)) {
    stop("CUDA matvec handle must be a handle object", call. = FALSE)
  }
  rhs_was_vector <- is.null(dim(rhs))
  rhs <- if (rhs_was_vector) {
    matrix(as.numeric(rhs), ncol = 1L)
  } else {
    as.matrix(rhs)
  }
  storage.mode(rhs) <- "double"
  if (nrow(rhs) != as.integer(handle$n)) {
    stop("rhs row count must match matrix dimension", call. = FALSE)
  }
  if (!all(is.finite(rhs))) {
    stop("Data contains missing or infinite values", call. = FALSE)
  }
  result <- .Call("C_legacy_dcov_spectra_matvec_cuda_handle_apply",
                  handle$ptr, rhs, PACKAGE = "fastkpc_cuda")
  if (rhs_was_vector) {
    result$values <- as.numeric(result$values[, 1L])
  }
  result
}

legacy_dcov_spectra_matvec_cuda_handle_project <- function(handle, basis) {
  load_fastkpc_cuda_native()
  if (!is.list(handle) || is.null(handle$ptr)) {
    stop("CUDA matvec handle must be a handle object", call. = FALSE)
  }
  basis <- if (is.null(dim(basis))) {
    matrix(as.numeric(basis), ncol = 1L)
  } else {
    as.matrix(basis)
  }
  storage.mode(basis) <- "double"
  if (nrow(basis) != as.integer(handle$n)) {
    stop("basis row count must match matrix dimension", call. = FALSE)
  }
  if (ncol(basis) < 1L) {
    stop("basis must have at least one column", call. = FALSE)
  }
  if (!all(is.finite(basis))) {
    stop("Data contains missing or infinite values", call. = FALSE)
  }
  .Call("C_legacy_dcov_spectra_matvec_cuda_handle_project",
        handle$ptr, basis, PACKAGE = "fastkpc_cuda")
}

legacy_dcov_spectra_matvec_cuda_handle_apply_sequence <- function(handle, rhs) {
  load_fastkpc_cuda_native()
  if (!is.list(handle) || is.null(handle$ptr)) {
    stop("CUDA matvec handle must be a handle object", call. = FALSE)
  }
  rhs_was_vector <- is.null(dim(rhs))
  rhs <- if (rhs_was_vector) {
    matrix(as.numeric(rhs), ncol = 1L)
  } else {
    as.matrix(rhs)
  }
  storage.mode(rhs) <- "double"
  if (nrow(rhs) != as.integer(handle$n)) {
    stop("rhs row count must match matrix dimension", call. = FALSE)
  }
  if (!all(is.finite(rhs))) {
    stop("Data contains missing or infinite values", call. = FALSE)
  }
  result <- .Call("C_legacy_dcov_spectra_matvec_cuda_handle_apply_sequence",
                  handle$ptr, rhs, PACKAGE = "fastkpc_cuda")
  if (rhs_was_vector) {
    result$values <- as.numeric(result$values[, 1L])
  }
  result
}

legacy_dcov_spectra_matvec_cuda_operator_eigs <- function(a, nev,
                                                          ncv = NULL,
                                                          tol = 1e-10,
                                                          maxitr = 1000L) {
  load_fastkpc_cuda_native()
  a <- as.matrix(a)
  storage.mode(a) <- "double"
  if (nrow(a) != ncol(a)) {
    stop("matrix must be square", call. = FALSE)
  }
  if (!all(is.finite(a))) {
    stop("Data contains missing or infinite values", call. = FALSE)
  }
  nev <- as.integer(nev)
  if (length(nev) != 1L || is.na(nev) || nev < 1L || nev >= nrow(a)) {
    stop("nev must be positive and smaller than matrix dimension",
         call. = FALSE)
  }
  ncv <- if (is.null(ncv)) {
    min(nrow(a), max(2L * nev + 1L, 20L))
  } else {
    as.integer(ncv)
  }
  if (length(ncv) != 1L || is.na(ncv) || ncv <= nev || ncv > nrow(a)) {
    stop("ncv must be greater than nev and no larger than matrix dimension",
         call. = FALSE)
  }
  .Call("C_legacy_dcov_spectra_matvec_cuda_operator_eigs",
        a, nev, as.integer(ncv), as.numeric(tol), as.integer(maxitr),
        PACKAGE = "fastkpc_cuda")
}

legacy_dcov_spectra_matvec_cuda_lowrank_shadow <- function(x, y, numCol,
                                                           ncv = NULL,
                                                           tol = 1e-10,
                                                           maxitr = 1000L) {
  load_fastkpc_cuda_native()
  x <- as.numeric(x)
  y <- as.numeric(y)
  if (length(x) != length(y)) {
    stop("Sample sizes must agree", call. = FALSE)
  }
  if (length(x) <= 5L) {
    stop("legacy dCov gamma lowrank shadow requires n > 5", call. = FALSE)
  }
  if (!all(is.finite(x)) || !all(is.finite(y))) {
    stop("Data contains missing or infinite values", call. = FALSE)
  }
  numCol <- as.integer(numCol)
  if (length(numCol) != 1L || is.na(numCol) ||
      numCol < 1L || numCol >= length(x)) {
    stop("numCol must be positive and less than sample size",
         call. = FALSE)
  }
  ncv <- if (is.null(ncv)) {
    min(length(x), max(2L * numCol + 1L, 20L))
  } else {
    as.integer(ncv)
  }
  if (length(ncv) != 1L || is.na(ncv) || ncv <= numCol || ncv > length(x)) {
    stop("ncv must be greater than numCol and no larger than sample size",
         call. = FALSE)
  }
  if (!is.finite(tol) || tol <= 0) {
    stop("tol must be a positive finite value", call. = FALSE)
  }
  maxitr <- as.integer(maxitr)
  if (length(maxitr) != 1L || is.na(maxitr) || maxitr <= 0L) {
    stop("maxitr must be positive", call. = FALSE)
  }
  .Call("C_legacy_dcov_spectra_matvec_cuda_lowrank_shadow",
        x, y, numCol, ncv, as.numeric(tol), maxitr,
        PACKAGE = "fastkpc_cuda")
}

legacy_dcov_spectra_matvec_cuda_lowrank_gamma <- function(x, y, numCol,
                                                          index = 1,
                                                          ncv = NULL,
                                                          tol = 1e-10,
                                                          maxitr = 1000L) {
  load_fastkpc_cuda_native()
  x <- as.numeric(x)
  y <- as.numeric(y)
  if (length(x) != length(y)) {
    stop("Sample sizes must agree", call. = FALSE)
  }
  if (length(x) <= 5L) {
    stop("legacy dCov gamma lowrank CUDA requires n > 5", call. = FALSE)
  }
  if (!all(is.finite(x)) || !all(is.finite(y))) {
    stop("Data contains missing or infinite values", call. = FALSE)
  }
  numCol <- as.integer(numCol)
  if (length(numCol) != 1L || is.na(numCol) ||
      numCol < 1L || numCol >= length(x)) {
    stop("numCol must be positive and less than sample size",
         call. = FALSE)
  }
  index <- as.numeric(index)
  if (length(index) != 1L || is.na(index) || index < 0 || index > 2) {
    index <- 1
  }
  ncv <- if (is.null(ncv)) {
    min(length(x), max(2L * numCol + 1L, 20L))
  } else {
    as.integer(ncv)
  }
  if (length(ncv) != 1L || is.na(ncv) || ncv <= numCol || ncv > length(x)) {
    stop("ncv must be greater than numCol and no larger than sample size",
         call. = FALSE)
  }
  if (!is.finite(tol) || tol <= 0) {
    stop("tol must be a positive finite value", call. = FALSE)
  }
  maxitr <- as.integer(maxitr)
  if (length(maxitr) != 1L || is.na(maxitr) || maxitr <= 0L) {
    stop("maxitr must be positive", call. = FALSE)
  }
  .Call("C_legacy_dcov_spectra_matvec_cuda_lowrank_gamma",
        x, y, numCol, index, ncv, as.numeric(tol), maxitr,
        PACKAGE = "fastkpc_cuda")
}

legacy_dcov_spectra_matvec_cuda_lowrank_gamma_batch <- function(x, y, numCol,
                                                                index = 1,
                                                                ncv = NULL,
                                                                tol = 1e-10,
                                                                maxitr = 1000L) {
  load_fastkpc_cuda_native()
  x <- as.matrix(x)
  y <- as.matrix(y)
  storage.mode(x) <- "double"
  storage.mode(y) <- "double"
  if (!identical(dim(x), dim(y))) {
    stop("x and y must have identical dimensions", call. = FALSE)
  }
  if (nrow(x) <= 5L) {
    stop("legacy dCov gamma lowrank CUDA requires n > 5", call. = FALSE)
  }
  if (ncol(x) < 1L) {
    stop("batch must contain at least one pair", call. = FALSE)
  }
  if (!all(is.finite(x)) || !all(is.finite(y))) {
    stop("Data contains missing or infinite values", call. = FALSE)
  }
  numCol <- as.integer(numCol)
  if (length(numCol) != 1L || is.na(numCol) ||
      numCol < 1L || numCol >= nrow(x)) {
    stop("numCol must be positive and less than sample size",
         call. = FALSE)
  }
  index <- as.numeric(index)
  if (length(index) != 1L || is.na(index) || index < 0 || index > 2) {
    index <- 1
  }
  ncv <- if (is.null(ncv)) {
    min(nrow(x), max(2L * numCol + 1L, 20L))
  } else {
    as.integer(ncv)
  }
  if (length(ncv) != 1L || is.na(ncv) || ncv <= numCol || ncv > nrow(x)) {
    stop("ncv must be greater than numCol and no larger than sample size",
         call. = FALSE)
  }
  if (!is.finite(tol) || tol <= 0) {
    stop("tol must be a positive finite value", call. = FALSE)
  }
  maxitr <- as.integer(maxitr)
  if (length(maxitr) != 1L || is.na(maxitr) || maxitr <= 0L) {
    stop("maxitr must be positive", call. = FALSE)
  }
  .Call("C_legacy_dcov_spectra_matvec_cuda_lowrank_gamma_batch",
        x, y, numCol, index, ncv, as.numeric(tol), maxitr,
        PACKAGE = "fastkpc_cuda")
}

legacy_dcov_gamma_cpp_component_cache_batch_native <- function(
    residuals, left_columns, right_columns, numCol = 35L, index = 1) {
  load_fastkpc_cuda_native()
  residuals <- as.matrix(residuals)
  storage.mode(residuals) <- "double"
  .Call(
    "C_legacy_dcov_gamma_cpp_component_cache_batch",
    residuals, as.integer(left_columns), as.integer(right_columns),
    as.integer(numCol), as.numeric(index), PACKAGE = "fastkpc_cuda"
  )
}

legacy_dcov_spectra_matvec_cuda_lowrank_shadow_grid <- function(cases,
                                                                case_indices =
                                                                  NULL,
                                                                sample_sizes =
                                                                  NULL,
                                                                numCol,
                                                                ncv = NULL,
                                                                tol = 1e-10,
                                                                maxitr = 1000L) {
  if (!is.list(cases) || length(cases) < 1L) {
    stop("cases must be a non-empty list", call. = FALSE)
  }
  if (missing(numCol)) {
    stop("numCol is required", call. = FALSE)
  }
  if (is.null(case_indices)) {
    case_indices <- seq_along(cases)
  }
  case_indices <- as.integer(case_indices)
  if (length(case_indices) < 1L || anyNA(case_indices) ||
      any(case_indices < 1L) || any(case_indices > length(cases))) {
    stop("case_indices must select entries from cases", call. = FALSE)
  }
  use_full_sample <- is.null(sample_sizes)
  sample_sizes <- if (use_full_sample) {
    NA_integer_
  } else {
    as.integer(sample_sizes)
  }
  if (length(sample_sizes) < 1L || (!use_full_sample && anyNA(sample_sizes)) ||
      any(sample_sizes <= 5L, na.rm = TRUE)) {
    stop("sample_sizes must be greater than 5", call. = FALSE)
  }

  grid <- expand.grid(case_index = case_indices,
                      sample_size = sample_sizes,
                      KEEP.OUT.ATTRS = FALSE)
  rows <- vector("list", nrow(grid))
  for (row_index in seq_len(nrow(grid))) {
    case_index <- grid$case_index[row_index]
    case <- cases[[case_index]]
    if (!is.null(case$residuals$rx) && !is.null(case$residuals$ry)) {
      x <- case$residuals$rx
      y <- case$residuals$ry
    } else if (!is.null(case$rx) && !is.null(case$ry)) {
      x <- case$rx
      y <- case$ry
    } else if (!is.null(case$x) && !is.null(case$y)) {
      x <- case$x
      y <- case$y
    } else {
      stop("case ", case_index,
           " must contain residuals$rx/residuals$ry, rx/ry, or x/y",
           call. = FALSE)
    }
    sample_size <- grid$sample_size[row_index]
    n <- length(x)
    if (length(y) != n) {
      stop("case ", case_index, " sample sizes must agree", call. = FALSE)
    }
    if (is.na(sample_size)) {
      sample_size <- n
    }
    if (sample_size > n) {
      stop("sample_size exceeds case ", case_index, " sample size",
           call. = FALSE)
    }
    result <- legacy_dcov_spectra_matvec_cuda_lowrank_shadow(
      x[seq_len(sample_size)],
      y[seq_len(sample_size)],
      numCol = numCol,
      ncv = ncv,
      tol = tol,
      maxitr = maxitr
    )
    rows[[row_index]] <- data.frame(
      case_index = case_index,
      sample_size = sample_size,
      backend = as.character(result$backend),
      n = as.integer(result$n),
      numCol = as.integer(result$numCol),
      ncv = as.integer(result$ncv),
      cpu_converged_x = isTRUE(result$cpu_converged_x),
      cpu_converged_y = isTRUE(result$cpu_converged_y),
      cuda_converged_x = isTRUE(result$cuda_converged_x),
      cuda_converged_y = isTRUE(result$cuda_converged_y),
      max_abs_eigenvalue_diff_x =
        as.numeric(result$max_abs_eigenvalue_diff_x),
      max_abs_eigenvalue_diff_y =
        as.numeric(result$max_abs_eigenvalue_diff_y),
      min_centered_abs_corr_x =
        as.numeric(result$min_centered_abs_corr_x),
      min_centered_abs_corr_y =
        as.numeric(result$min_centered_abs_corr_y),
      nV2_abs_diff = as.numeric(result$nV2_abs_diff),
      x_moment_abs_diff = as.numeric(result$x_moment_abs_diff),
      y_moment_abs_diff = as.numeric(result$y_moment_abs_diff),
      statistic_input_max_abs_diff =
        as.numeric(result$statistic_input_max_abs_diff),
      spectra_matvec_count = as.integer(result$spectra_matvec_count),
      matrix_h2d_ms_during_compute =
        as.numeric(result$matrix_h2d_ms_during_compute),
      matrix_bytes = as.numeric(result$matrix_bytes),
      workspace_realloc_count = as.integer(result$workspace_realloc_count),
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, rows)
}

legacy_dcov_spectra_matvec_cuda_handle_free <- function(handle) {
  load_fastkpc_cuda_native()
  if (!is.list(handle) || is.null(handle$ptr)) {
    stop("CUDA matvec handle must be a handle object", call. = FALSE)
  }
  invisible(.Call("C_legacy_dcov_spectra_matvec_cuda_handle_free",
                  handle$ptr, PACKAGE = "fastkpc_cuda"))
}

fast_dcov_batch_cuda <- function(x, y, index = 1, legacy_index = TRUE) {
  load_fastkpc_cuda_native()
  x <- if (is.matrix(x)) x else matrix(as.numeric(x), ncol = 1)
  y <- if (is.matrix(y)) y else matrix(as.numeric(y), ncol = 1)
  storage.mode(x) <- "double"
  storage.mode(y) <- "double"
  .Call("C_fast_dcov_batch_cuda", x, y, as.numeric(index), isTRUE(legacy_index),
        PACKAGE = "fastkpc_cuda")
}

fast_hsic_gamma_cuda <- function(x, y, sig = 1) {
  load_fastkpc_cuda_native()
  .Call("C_fast_hsic_gamma_cuda", as.numeric(x), as.numeric(y),
        as.numeric(sig), PACKAGE = "fastkpc_cuda")
}

fast_hsic_perm_cuda <- function(x, y, sig = 1, replicates = 100L,
                                seed, include_observed = TRUE) {
  if (missing(seed) || is.null(seed)) {
    stop("CUDA HSIC permutation requires explicit seed in this stage",
         call. = FALSE)
  }
  load_fastkpc_cuda_native()
  .Call("C_fast_hsic_perm_cuda", as.numeric(x), as.numeric(y),
        as.numeric(sig), as.integer(replicates), as.integer(seed),
        isTRUE(include_observed), PACKAGE = "fastkpc_cuda")
}

fastspline_residual_cuda <- function(y, S, fastspline_params = list(),
                                     fallback = TRUE) {
  load_fastkpc_cuda_native()
  S <- as.matrix(S)
  storage.mode(S) <- "double"
  .Call("C_fastspline_residual_cuda", as.numeric(y), S, fastspline_params,
        isTRUE(fallback), PACKAGE = "fastkpc_cuda")
}

fastspline_residual_batch_cuda <- function(data, targets, conditioning_sets,
                                           fastspline_params = list(),
                                           fallback = TRUE) {
  load_fastkpc_cuda_native()
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  .Call("C_fastspline_residual_batch_cuda", data, as.integer(targets),
        conditioning_sets, fastspline_params, isTRUE(fallback),
        PACKAGE = "fastkpc_cuda")
}

mgcv_extract_gpu_solve_handle_fixed_sp_cuda <- function(handle) {
  load_fastkpc_cuda_native()
  X <- as.matrix(handle$X)
  y <- as.numeric(handle$y)
  Z <- as.matrix(handle$Z)
  XtX_null <- as.matrix(handle$XtX_null)
  penalty_null <- as.matrix(handle$penalty_null)
  Xty_null <- as.numeric(handle$Xty_null)
  storage.mode(X) <- "double"
  storage.mode(y) <- "double"
  storage.mode(Z) <- "double"
  storage.mode(XtX_null) <- "double"
  storage.mode(penalty_null) <- "double"
  storage.mode(Xty_null) <- "double"
  .Call("C_mgcv_extract_gpu_solve_handle_fixed_sp",
        X, y, Z, XtX_null, penalty_null, Xty_null,
        PACKAGE = "fastkpc_cuda")
}

full_cuda_ci_single_penalty_mroot_cuda <- function(
    penalty_matrix, penalty_rank, log_sp) {
  load_fastkpc_cuda_native()
  penalty_matrix <- as.matrix(penalty_matrix)
  storage.mode(penalty_matrix) <- "double"
  .Call(
    "C_full_cuda_ci_single_penalty_mroot_cuda",
    penalty_matrix, as.integer(penalty_rank), as.double(log_sp),
    PACKAGE = "fastkpc_cuda"
  )
}

full_cuda_ci_native_setup_native <- function(conditioning) {
  load_fastkpc_cuda_native()
  conditioning <- as.matrix(conditioning)
  storage.mode(conditioning) <- "double"
  .Call(
    "C_full_cuda_ci_native_setup", conditioning,
    PACKAGE = "fastkpc_cuda"
  )
}

full_cuda_ci_multi_penalty_capacity_qualify_native <- function(
    setup, geometry, Y) {
  load_fastkpc_cuda_native()
  Y <- as.matrix(Y)
  storage.mode(Y) <- "double"
  .Call(
    "C_full_cuda_ci_multi_penalty_capacity_qualify",
    setup$X,
    geometry$magic_qr_packed,
    geometry$magic_tau,
    geometry$magic_r,
    geometry$magic_pivot,
    geometry$penalty_roots,
    geometry$penalty_matrices,
    as.integer(setup$penalty_ranks),
    geometry$initial_log_sp,
    Y,
    PACKAGE = "fastkpc_cuda"
  )
}

full_cuda_ci_native_geometry_prepare_native <- function(
    X, penalty_blocks, penalty_offsets, penalty_ranks) {
  load_fastkpc_cuda_native()
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  penalty_blocks <- lapply(penalty_blocks, function(block) {
    block <- as.matrix(block)
    storage.mode(block) <- "double"
    block
  })
  .Call(
    "C_full_cuda_ci_native_geometry_prepare",
    X, penalty_blocks, as.integer(penalty_offsets),
    as.integer(penalty_ranks), PACKAGE = "fastkpc_cuda"
  )
}

full_cuda_ci_multi_penalty_gcv_evaluate_cpp_native <- function(
    X, y, penalty_blocks, penalty_offsets, penalty_ranks, log_sp,
    H = NULL, constraint = NULL,
    rank_tolerance = sqrt(.Machine$double.eps)) {
  load_fastkpc_cuda_native()
  X <- as.matrix(X)
  y <- as.numeric(y)
  penalty_blocks <- lapply(penalty_blocks, function(value) {
    value <- as.matrix(value)
    storage.mode(value) <- "double"
    value
  })
  if (!is.null(H)) {
    H <- as.matrix(H)
    storage.mode(H) <- "double"
  }
  if (!is.null(constraint)) {
    constraint <- as.matrix(constraint)
    storage.mode(constraint) <- "double"
  }
  storage.mode(X) <- "double"
  .Call(
    "C_full_cuda_ci_multi_penalty_gcv_evaluate_cpp",
    X, y, penalty_blocks, as.integer(penalty_offsets),
    as.integer(penalty_ranks), as.double(log_sp), H, constraint,
    as.double(rank_tolerance), PACKAGE = "fastkpc_cuda"
  )
}

full_cuda_ci_multi_penalty_gcv_optimize_cpp_native <- function(
    X, y, penalty_blocks, penalty_offsets, penalty_ranks,
    H = NULL, constraint = NULL, control = list()) {
  load_fastkpc_cuda_native()
  X <- as.matrix(X)
  y <- as.numeric(y)
  penalty_blocks <- lapply(penalty_blocks, function(value) {
    value <- as.matrix(value)
    storage.mode(value) <- "double"
    value
  })
  if (!is.null(H)) {
    H <- as.matrix(H)
    storage.mode(H) <- "double"
  }
  if (!is.null(constraint)) {
    constraint <- as.matrix(constraint)
    storage.mode(constraint) <- "double"
  }
  storage.mode(X) <- "double"
  .Call(
    "C_full_cuda_ci_multi_penalty_gcv_optimize_cpp",
    X, y, penalty_blocks, as.integer(penalty_offsets),
    as.integer(penalty_ranks), H, constraint, control,
    PACKAGE = "fastkpc_cuda"
  )
}

full_cuda_ci_multi_penalty_gcv_evaluate_cuda_native <- function(
    X, Y, magic_qr_packed, magic_tau, magic_r, magic_pivot,
    penalty_roots, penalty_matrices, penalty_ranks, log_sp,
    rank_tolerance = sqrt(.Machine$double.eps)) {
  load_fastkpc_cuda_native()
  X <- as.matrix(X)
  Y <- as.matrix(Y)
  magic_qr_packed <- as.matrix(magic_qr_packed)
  magic_r <- as.matrix(magic_r)
  log_sp <- as.matrix(log_sp)
  penalty_roots <- lapply(penalty_roots, function(value) {
    value <- as.matrix(value)
    storage.mode(value) <- "double"
    value
  })
  penalty_matrices <- lapply(penalty_matrices, function(value) {
    value <- as.matrix(value)
    storage.mode(value) <- "double"
    value
  })
  storage.mode(X) <- "double"
  storage.mode(Y) <- "double"
  storage.mode(magic_qr_packed) <- "double"
  storage.mode(magic_r) <- "double"
  storage.mode(log_sp) <- "double"
  .Call(
    "C_full_cuda_ci_multi_penalty_gcv_evaluate_cuda",
    X, Y, magic_qr_packed, as.double(magic_tau), magic_r,
    as.integer(magic_pivot), penalty_roots, penalty_matrices,
    as.integer(penalty_ranks), log_sp, as.double(rank_tolerance),
    PACKAGE = "fastkpc_cuda"
  )
}

full_cuda_ci_multi_penalty_gcv_grouped_prototype_native <- function(
    prepared, Y, log_sp, force_stable_svd, rank_tolerance,
    grouped_warps_per_block, timing_repetitions) {
  load_fastkpc_cuda_native()
  prepared <- as.list(prepared)
  prepared$X <- as.matrix(prepared$X)
  prepared$magic_qr_packed <- as.matrix(prepared$magic_qr_packed)
  prepared$magic_r <- as.matrix(prepared$magic_r)
  prepared$magic_tau <- as.double(prepared$magic_tau)
  prepared$magic_pivot <- as.integer(prepared$magic_pivot)
  prepared$penalty_ranks <- as.integer(prepared$penalty_ranks)
  prepared$penalty_roots <- lapply(prepared$penalty_roots, function(value) {
    value <- as.matrix(value)
    storage.mode(value) <- "double"
    value
  })
  prepared$penalty_matrices <- lapply(
    prepared$penalty_matrices,
    function(value) {
      value <- as.matrix(value)
      storage.mode(value) <- "double"
      value
    }
  )
  storage.mode(prepared$X) <- "double"
  storage.mode(prepared$magic_qr_packed) <- "double"
  storage.mode(prepared$magic_r) <- "double"
  Y <- as.matrix(Y)
  log_sp <- as.matrix(log_sp)
  storage.mode(Y) <- "double"
  storage.mode(log_sp) <- "double"
  .Call(
    "C_full_cuda_ci_multi_penalty_gcv_grouped_prototype",
    prepared, Y, log_sp, as.integer(force_stable_svd),
    as.double(rank_tolerance), as.integer(grouped_warps_per_block),
    as.integer(timing_repetitions), PACKAGE = "fastkpc_cuda"
  )
}

full_cuda_ci_multi_penalty_gcv_optimize_cuda_native <- function(
    X, Y, magic_qr_packed, magic_tau, magic_r, magic_pivot,
    penalty_roots, penalty_matrices, penalty_ranks, initial_log_sp,
    control = list()) {
  load_fastkpc_cuda_native()
  X <- as.matrix(X)
  Y <- as.matrix(Y)
  magic_qr_packed <- as.matrix(magic_qr_packed)
  magic_r <- as.matrix(magic_r)
  penalty_roots <- lapply(penalty_roots, function(value) {
    value <- as.matrix(value)
    storage.mode(value) <- "double"
    value
  })
  penalty_matrices <- lapply(penalty_matrices, function(value) {
    value <- as.matrix(value)
    storage.mode(value) <- "double"
    value
  })
  storage.mode(X) <- "double"
  storage.mode(Y) <- "double"
  storage.mode(magic_qr_packed) <- "double"
  storage.mode(magic_r) <- "double"
  .Call(
    "C_full_cuda_ci_multi_penalty_gcv_optimize_cuda",
    X, Y, magic_qr_packed, as.double(magic_tau), magic_r,
    as.integer(magic_pivot), penalty_roots, penalty_matrices,
    as.integer(penalty_ranks), as.double(initial_log_sp), control,
    PACKAGE = "fastkpc_cuda"
  )
}

full_cuda_ci_multi_penalty_gcv_prepared_create_native <- function(
    prepared, target_capacity, device_id = 0L) {
  load_fastkpc_cuda_native()
  .Call(
    "C_full_cuda_ci_multi_penalty_gcv_prepared_create",
    prepared$X, prepared$magic_qr_packed, as.double(prepared$magic_tau),
    prepared$magic_r, as.integer(prepared$magic_pivot),
    prepared$penalty_roots, prepared$penalty_matrices,
    as.integer(prepared$penalty_ranks),
    as.double(prepared$initial_log_sp), as.integer(target_capacity),
    as.integer(device_id), PACKAGE = "fastkpc_cuda"
  )
}

full_cuda_ci_multi_penalty_gcv_prepared_info_native <- function(handle) {
  load_fastkpc_cuda_native()
  .Call(
    "C_full_cuda_ci_multi_penalty_gcv_prepared_info", handle,
    PACKAGE = "fastkpc_cuda"
  )
}

full_cuda_ci_multi_penalty_gcv_prepared_free_native <- function(handle) {
  load_fastkpc_cuda_native()
  invisible(.Call(
    "C_full_cuda_ci_multi_penalty_gcv_prepared_free", handle,
    PACKAGE = "fastkpc_cuda"
  ))
}

full_cuda_ci_multi_penalty_gcv_optimize_batch_native <- function(
    handle, Y, target_keys, control = list()) {
  load_fastkpc_cuda_native()
  Y <- as.matrix(Y)
  storage.mode(Y) <- "double"
  .Call(
    "C_full_cuda_ci_multi_penalty_gcv_optimize_batch",
    handle, Y, as.character(target_keys), control,
    PACKAGE = "fastkpc_cuda"
  )
}

full_cuda_ci_multi_penalty_gcv_optimize_multi_native <- function(
    requests, concurrency, control = list()) {
  load_fastkpc_cuda_native()
  .Call(
    "C_full_cuda_ci_multi_penalty_gcv_optimize_multi",
    requests, as.integer(concurrency), control,
    PACKAGE = "fastkpc_cuda"
  )
}

full_cuda_ci_multi_penalty_gcv_residual_info_native <- function(token) {
  load_fastkpc_cuda_native()
  .Call(
    "C_full_cuda_ci_multi_penalty_gcv_residual_info", token,
    PACKAGE = "fastkpc_cuda"
  )
}

full_cuda_ci_multi_penalty_gcv_residual_shadow_native <- function(token) {
  load_fastkpc_cuda_native()
  .Call(
    "C_full_cuda_ci_multi_penalty_gcv_residual_shadow", token,
    PACKAGE = "fastkpc_cuda"
  )
}

full_cuda_ci_multi_penalty_gcv_residual_release_native <- function(token) {
  load_fastkpc_cuda_native()
  invisible(.Call(
    "C_full_cuda_ci_multi_penalty_gcv_residual_release", token,
    PACKAGE = "fastkpc_cuda"
  ))
}

full_cuda_ci_multi_penalty_gcv_residual_free_native <- function(token) {
  load_fastkpc_cuda_native()
  invisible(.Call(
    "C_full_cuda_ci_multi_penalty_gcv_residual_free", token,
    PACKAGE = "fastkpc_cuda"
  ))
}

full_cuda_ci_phase10_optimizer_residual_parity_native <- function(
    optimizer_residual, fixed_residual) {
  load_fastkpc_cuda_native()
  .Call(
    "C_full_cuda_ci_phase10_optimizer_residual_parity",
    optimizer_residual, fixed_residual,
    PACKAGE = "fastkpc_cuda"
  )
}

full_cuda_ci_phase10_fixed_residual_identity_native <- function(
    left_residual, right_residual) {
  load_fastkpc_cuda_native()
  .Call(
    "C_full_cuda_ci_phase10_fixed_residual_identity",
    left_residual, right_residual,
    PACKAGE = "fastkpc_cuda"
  )
}

full_cuda_ci_single_penalty_gcv_cuda <- function(
    X, Y, rhs_transform, eigenvalues, magic_qr_packed, magic_tau, magic_r,
    magic_penalty_root, magic_penalty_matrix, target_ids, penalty_rank, initial_sp,
    sp_grid = numeric(), materialize_grid = length(sp_grid) > 0L,
    keep_transcript = FALSE) {
  load_fastkpc_cuda_native()
  X <- as.matrix(X)
  Y <- as.matrix(Y)
  rhs_transform <- as.matrix(rhs_transform)
  eigenvalues <- as.numeric(eigenvalues)
  magic_qr_packed <- as.matrix(magic_qr_packed)
  magic_tau <- as.numeric(magic_tau)
  magic_r <- as.matrix(magic_r)
  magic_penalty_root <- as.matrix(magic_penalty_root)
  magic_penalty_matrix <- as.matrix(magic_penalty_matrix)
  sp_grid <- as.numeric(sp_grid)
  storage.mode(X) <- "double"
  storage.mode(Y) <- "double"
  storage.mode(rhs_transform) <- "double"
  storage.mode(magic_qr_packed) <- "double"
  storage.mode(magic_r) <- "double"
  storage.mode(magic_penalty_root) <- "double"
  storage.mode(magic_penalty_matrix) <- "double"
  .Call(
    "C_full_cuda_ci_single_penalty_gcv_cuda",
    X, Y, rhs_transform, eigenvalues, magic_qr_packed, magic_tau, magic_r,
    magic_penalty_root, magic_penalty_matrix, as.integer(target_ids),
    as.integer(penalty_rank),
    as.numeric(initial_sp), sp_grid, isTRUE(materialize_grid),
    isTRUE(keep_transcript),
    PACKAGE = "fastkpc_cuda"
  )
}

full_cuda_ci_single_penalty_gcv_multi_cuda <- function(
    setups, concurrency = 1L) {
  load_fastkpc_cuda_native()
  if (!is.list(setups) || length(setups) == 0L) {
    stop("setups must be a non-empty list", call. = FALSE)
  }
  concurrency <- as.integer(concurrency)
  if (length(concurrency) != 1L || is.na(concurrency) ||
      concurrency < 1L || concurrency > 16L) {
    stop("concurrency must be one integer in [1, 16]", call. = FALSE)
  }
  .Call(
    "C_full_cuda_ci_single_penalty_gcv_multi_cuda",
    setups, concurrency,
    PACKAGE = "fastkpc_cuda"
  )
}

full_cuda_ci_single_penalty_gcv_fixed_sp_cuda <- function(
    handle, X, Y, rhs_transform, eigenvalues, magic_qr_packed, magic_tau,
    magic_r, magic_penalty_root, magic_penalty_matrix, target_ids, penalty_rank,
    initial_sp, planned_route,
    target_keys, outputs = c("residuals")) {
  load_fastkpc_cuda_native()
  .Call(
    "C_full_cuda_ci_single_penalty_gcv_fixed_sp_cuda",
    handle, X, Y, rhs_transform, as.double(eigenvalues), magic_qr_packed,
    as.double(magic_tau), magic_r,
    magic_penalty_root, magic_penalty_matrix, as.integer(target_ids),
    as.integer(penalty_rank),
    as.double(initial_sp), as.character(planned_route),
    as.character(target_keys),
    as.character(outputs), PACKAGE = "fastkpc_cuda"
  )
}

mgcv_extract_gpu_solve_same_setup_batch_fixed_sp_cuda <- function(handles) {
  load_fastkpc_cuda_native()
  if (!is.list(handles) || length(handles) == 0L) {
    stop("handles must be a non-empty list", call. = FALSE)
  }
  first <- handles[[1L]]
  X <- as.matrix(first$X)
  Z <- as.matrix(first$Z)
  XtX_null <- as.matrix(first$XtX_null)
  Y <- do.call(cbind, lapply(handles, function(handle) as.numeric(handle$y)))
  Xty_null <- do.call(cbind, lapply(handles, function(handle) as.numeric(handle$Xty_null)))
  penalty_null_list <- lapply(handles, function(handle) {
    penalty <- as.matrix(handle$penalty_null)
    storage.mode(penalty) <- "double"
    penalty
  })
  storage.mode(X) <- "double"
  storage.mode(Y) <- "double"
  storage.mode(Z) <- "double"
  storage.mode(XtX_null) <- "double"
  storage.mode(Xty_null) <- "double"
  .Call("C_mgcv_extract_gpu_solve_same_setup_batch_fixed_sp",
        X, Y, Z, XtX_null, penalty_null_list, Xty_null,
        PACKAGE = "fastkpc_cuda")
}

fast_skeleton_cuda <- function(data, alpha, max_conditioning_size,
                               index = 1, legacy_index = TRUE,
                               batch_size = 0) {
  load_fastkpc_cuda_native()
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  .Call("C_fast_skeleton_cuda", data, as.numeric(alpha),
        as.integer(max_conditioning_size), as.numeric(index),
        isTRUE(legacy_index), as.integer(batch_size),
        PACKAGE = "fastkpc_cuda")
}

fast_skeleton_cuda_cached <- function(data, alpha, max_conditioning_size,
                                      index = 1, legacy_index = TRUE,
                                      batch_size = 0,
                                      residual_cache = TRUE) {
  load_fastkpc_cuda_native()
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  .Call("C_fast_skeleton_cuda_cached", data, as.numeric(alpha),
        as.integer(max_conditioning_size), as.numeric(index),
        isTRUE(legacy_index), as.integer(batch_size),
        isTRUE(residual_cache),
        PACKAGE = "fastkpc_cuda")
}

fast_skeleton_cuda_backend <- function(data, alpha, max_conditioning_size,
                                       residual_backend = "linear",
                                       residual_device = c("auto", "cpu", "cuda"),
                                       residual_cache = TRUE,
                                       index = 1,
                                       legacy_index = TRUE,
                                       batch_size = 0,
                                       residual_batch_size = 0,
                                       scheduler = c("auto", "layer", "legacy"),
                                       scheduler_diagnostics = TRUE,
                                       fastspline_params = list(),
                                       cuda_residual_fallback = TRUE,
                                       ci_method = "dcc.gamma",
                                       hsic_params = list(),
                                       permutation_params = list(),
                                       ci_diagnostics = TRUE) {
  load_fastkpc_cuda_native()
  residual_device <- match.arg(residual_device)
  scheduler <- match.arg(scheduler)
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  .Call("C_fast_skeleton_cuda_backend", data, as.numeric(alpha),
        as.integer(max_conditioning_size), as.numeric(index),
        isTRUE(legacy_index), as.integer(batch_size),
        isTRUE(residual_cache), as.character(residual_backend),
        as.character(residual_device), as.integer(residual_batch_size),
        as.character(scheduler), isTRUE(scheduler_diagnostics),
        fastspline_params,
        isTRUE(cuda_residual_fallback), as.character(ci_method),
        hsic_params, permutation_params, isTRUE(ci_diagnostics),
        PACKAGE = "fastkpc_cuda")
}

precision_replay_layer_native <- function(adjacency, edge_x, edge_y, x, y,
                                          conditioning_sets, p_values, alpha,
                                          pmax = NULL, trace_level = c("summary", "full", "none")) {
  load_fastkpc_cuda_native()
  trace_level <- match.arg(trace_level)
  adjacency <- as.matrix(adjacency)
  storage.mode(adjacency) <- "integer"
  if (is.null(pmax)) {
    pmax <- matrix(-Inf, nrow(adjacency), ncol(adjacency))
    diag(pmax) <- 1
  }
  pmax <- as.matrix(pmax)
  storage.mode(pmax) <- "double"
  .Call("C_precision_replay_layer_native",
        adjacency,
        pmax,
        as.integer(edge_x),
        as.integer(edge_y),
        as.integer(x),
        as.integer(y),
        conditioning_sets,
        as.numeric(p_values),
        as.numeric(alpha),
        as.character(trace_level),
        PACKAGE = "fastkpc_cuda")
}

precision_make_layer_plan_native <- function(adjacency, level) {
  load_fastkpc_cuda_native()
  adjacency <- as.matrix(adjacency)
  storage.mode(adjacency) <- "integer"
  .Call("C_precision_make_layer_plan_native",
        adjacency,
        as.integer(level),
        PACKAGE = "fastkpc_cuda")
}

precision_run_skeleton_ptable_native <- function(
    p = 6L, alpha = 0.05, max_conditioning_size = 2L,
    trace_level = c("summary", "full", "none")) {
  load_fastkpc_cuda_native()
  trace_level <- match.arg(trace_level)
  .Call("C_precision_run_skeleton_ptable_native",
        as.integer(p),
        as.numeric(alpha),
        as.integer(max_conditioning_size),
        as.character(trace_level),
        PACKAGE = "fastkpc_cuda")
}

precision_run_skeleton_provider_native <- function(
    p, alpha, max_conditioning_size, provider,
    trace_level = c("summary", "full", "none")) {
  load_fastkpc_cuda_native()
  if (!is.function(provider)) {
    stop("provider must be a function", call. = FALSE)
  }
  trace_level <- match.arg(trace_level)
  .Call("C_precision_run_skeleton_provider_native",
        as.integer(p),
        as.numeric(alpha),
        as.integer(max_conditioning_size),
        provider,
        as.character(trace_level),
        PACKAGE = "fastkpc_cuda")
}

precision_run_skeleton_dcov0_native <- function(
    data, alpha, index = 1, legacy_index = TRUE,
    trace_level = c("summary", "full", "none")) {
  load_fastkpc_cuda_native()
  trace_level <- match.arg(trace_level)
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  .Call("C_precision_run_skeleton_dcov0_native",
        data,
        as.numeric(alpha),
        as.numeric(index),
        isTRUE(legacy_index),
        as.character(trace_level),
        PACKAGE = "fastkpc_cuda")
}

precision_run_skeleton_exact_ci_native <- function(
    data, alpha, max_conditioning_size, index = 1, legacy_index = TRUE,
    trace_level = c("summary", "full", "none")) {
  load_fastkpc_cuda_native()
  trace_level <- match.arg(trace_level)
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  .Call("C_precision_run_skeleton_exact_ci_native",
        data,
        as.numeric(alpha),
        as.integer(max_conditioning_size),
        as.numeric(index),
        isTRUE(legacy_index),
        as.character(trace_level),
        PACKAGE = "fastkpc_cuda")
}

precision_run_skeleton_residual_provider_native <- function(
    data, alpha, max_conditioning_size, residual_provider,
    index = 1, legacy_index = TRUE,
    trace_level = c("summary", "full", "none")) {
  load_fastkpc_cuda_native()
  if (!is.function(residual_provider)) {
    stop("residual_provider must be a function", call. = FALSE)
  }
  trace_level <- match.arg(trace_level)
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  .Call("C_precision_run_skeleton_residual_provider_native",
        data,
        as.numeric(alpha),
        as.integer(max_conditioning_size),
        residual_provider,
        as.numeric(index),
        isTRUE(legacy_index),
        as.character(trace_level),
        PACKAGE = "fastkpc_cuda")
}

precision_run_skeleton_residual_provider_legacy_dcov_native <- function(
    data, alpha, max_conditioning_size, residual_provider,
    index = 1, numCol = floor(nrow(as.matrix(data)) / 10),
    trace_level = c("summary", "full", "logical", "none")) {
  load_fastkpc_cuda_native()
  if (!is.function(residual_provider)) {
    stop("residual_provider must be a function", call. = FALSE)
  }
  trace_level <- match.arg(trace_level)
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  .Call("C_precision_run_skeleton_residual_provider_legacy_dcov_native",
        data,
        as.numeric(alpha),
        as.integer(max_conditioning_size),
        residual_provider,
        as.numeric(index),
        as.integer(numCol),
        as.character(trace_level),
        PACKAGE = "fastkpc_cuda")
}

precision_run_skeleton_full_cuda_native <- function(
    data, alpha, max_conditioning_size = Inf,
    index = 1, numCol = floor(nrow(as.matrix(data)) / 10),
    trace_level = c("summary", "logical", "full", "none"),
    compatible_cuda_strict = TRUE,
    ci_method = c("dcc.gamma", "dcc.perm", "hsic.gamma", "hsic.perm"),
    hsic_params = list(sig = 1),
    permutation_params = list(replicates = 100L, seed = NULL,
                              include_observed = TRUE)) {
  load_fastkpc_cuda_native()
  trace_level <- match.arg(trace_level)
  ci_method <- match.arg(ci_method)
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  max_conditioning_size <- fastkpc_resolve_max_conditioning_size(
    max_conditioning_size, ncol(data)
  )
  if (identical(ci_method, "dcc.gamma")) {
    return(.Call(
      "C_full_cuda_ci_one_call_skeleton",
      data,
      as.numeric(alpha),
      max_conditioning_size,
      as.numeric(index),
      as.integer(numCol),
      as.character(trace_level),
      as.logical(compatible_cuda_strict),
      PACKAGE = "fastkpc_cuda"
    ))
  }
  hsic_sig <- as.numeric(if (is.null(hsic_params$sig)) 1 else
    hsic_params$sig)[1L]
  permutation_replicates <- as.integer(
    if (is.null(permutation_params$replicates)) 100L else
      permutation_params$replicates
  )[1L]
  permutation_seed <- if (is.null(permutation_params$seed)) NULL else
    permutation_params$seed
  permutation_has_seed <- !is.null(permutation_seed)
  if (ci_method %in% c("dcc.perm", "hsic.perm") &&
      !isTRUE(permutation_has_seed)) {
    stop("strict compatible CUDA permutation methods require an explicit seed",
         call. = FALSE)
  }
  permutation_seed_value <- if (isTRUE(permutation_has_seed)) {
    as.integer(permutation_seed)[1L]
  } else {
    0L
  }
  if (!is.finite(hsic_sig) || hsic_sig <= 0 ||
      is.na(permutation_replicates) || permutation_replicates < 1L ||
      is.na(permutation_seed_value) || permutation_seed_value < 0L) {
    stop("invalid strict compatible CUDA CI method parameters", call. = FALSE)
  }
  if (ci_method %in% c("dcc.perm", "hsic.perm")) {
    set.seed(permutation_seed_value)
  }
  .Call(
    "C_full_cuda_ci_one_call_skeleton_method",
    data,
    as.numeric(alpha),
    max_conditioning_size,
    as.numeric(index),
    as.integer(numCol),
    as.character(trace_level),
    as.logical(compatible_cuda_strict),
    as.character(ci_method),
    hsic_sig,
    permutation_replicates,
    as.logical(if (is.null(permutation_params$include_observed)) TRUE else
      permutation_params$include_observed),
    as.logical(permutation_has_seed),
    permutation_seed_value,
    PACKAGE = "fastkpc_cuda"
  )
}

full_cuda_ci_one_call_cache_control_native <- function(
    action = c("info", "reset", "configure", "configure_target"),
    capacity = NULL) {
  load_fastkpc_cuda_native()
  action <- match.arg(action)
  capacity_value <- if (is.null(capacity)) -1L else as.integer(capacity)
  .Call(
    "C_full_cuda_ci_one_call_cache_control",
    as.character(action), capacity_value,
    PACKAGE = "fastkpc_cuda"
  )
}

full_cuda_ci_one_call_cache_state_native <- function(data) {
  load_fastkpc_cuda_native()
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  .Call(
    "C_full_cuda_ci_one_call_cache_state", data,
    PACKAGE = "fastkpc_cuda"
  )
}

full_cuda_ci_method_seeded_permutations_native <- function(
    ci_method = c("dcc.perm", "hsic.perm"), n, replicates, seeds) {
  load_fastkpc_cuda_native()
  ci_method <- match.arg(ci_method)
  .Call(
    "C_full_cuda_ci_method_seeded_permutations",
    as.character(ci_method),
    as.integer(n),
    as.integer(replicates),
    as.integer(seeds),
    PACKAGE = "fastkpc_cuda"
  )
}

fastkpc_native_legacy_mgcv_residual_backend <- function() {
  raw <- tolower(Sys.getenv("FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND",
                            unset = "r"))
  if (raw %in% c("", "legacy", "r")) {
    "r"
  } else if (identical(raw, "cpp_guarded")) {
    "cpp_guarded"
  } else {
    "r"
  }
}

fastkpc_native_legacy_mgcv_backend_condition_threshold <- function() {
  value <- suppressWarnings(as.numeric(Sys.getenv(
    "FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_CONDITION_THRESHOLD",
    unset = "1e12"
  )))
  if (length(value) != 1L || !is.finite(value) || value < 0) 1e12 else value
}

fastkpc_native_legacy_mgcv_backend_native_s_size_limit <- function() {
  raw <- Sys.getenv("FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_NATIVE_S_SIZE_LIMIT",
                    unset = "Inf")
  value <- suppressWarnings(as.numeric(raw))
  if (length(value) != 1L || is.na(value) || value < 0) Inf else value
}

fastkpc_native_legacy_mgcv_provider_cores <- function() {
  raw <- Sys.getenv("FASTKPC_NATIVE_LEGACY_MGCV_PROVIDER_CORES", unset = "1")
  value <- suppressWarnings(as.integer(raw))
  if (length(value) != 1L || is.na(value) || value < 1L) 1L else value
}

fastkpc_native_prepare_legacy_mgcv_cpp_backend <- function() {
  if (!exists("fastkpc_legacy_mgcv_residual_cpp_backend_target",
              mode = "function")) {
    source("fastkpc/R/legacy_runner.R")
  }
  if (exists("fastkpc_legacy_prepare_mgcv_cpp_shadow",
             mode = "function")) {
    fastkpc_legacy_prepare_mgcv_cpp_shadow(TRUE)
  }
  required <- c(
    "fastkpc_legacy_runtime_zero",
    "fastkpc_legacy_mgcv_residual_cpp_backend_target",
    "fastkpc_legacy_env"
  )
  missing <- required[!vapply(required, exists, logical(1), mode = "function")]
  if (length(missing) > 0L) {
    stop("legacy mgcv C++ residual backend missing helper: ",
         paste(missing, collapse = ","), call. = FALSE)
  }
  invisible(TRUE)
}

fastkpc_native_counter_value <- function(counter_env, name, default) {
  if (is.null(counter_env)) return(default)
  value <- counter_env[[name]]
  if (is.null(value)) default else value
}

fastkpc_native_provider_backend_label <- function(backend) {
  if (identical(backend, "cpp_guarded")) {
    "legacy-mgcv-cpp-guarded-level-batch"
  } else {
    "legacy-mgcv-regrXonS-level-batch"
  }
}

fastkpc_legacy_mgcv_residual_provider_matrix <- function(
    data, requests, counter_env = NULL,
    backend = fastkpc_native_legacy_mgcv_residual_backend(),
    condition_threshold =
      fastkpc_native_legacy_mgcv_backend_condition_threshold(),
    native_s_size_limit =
      fastkpc_native_legacy_mgcv_backend_native_s_size_limit()) {
  backend <- match.arg(backend, c("r", "cpp_guarded"))
  required <- c("request_index", "target", "conditioning_sets",
                "S_key", "conditioning_size")
  missing_fields <- setdiff(required, names(requests))
  if (length(missing_fields) > 0L) {
    stop("residual provider request table missing fields: ",
         paste(missing_fields, collapse = ","), call. = FALSE)
  }
  if (!is.null(counter_env)) {
    level_calls <- if (is.null(counter_env$level_calls)) {
      0L
    } else {
      counter_env$level_calls
    }
    request_count <- if (is.null(counter_env$request_count)) {
      0L
    } else {
      counter_env$request_count
    }
    counter_env$level_calls <- level_calls + 1L
    counter_env$request_count <- request_count + nrow(requests)
    counter_env$mgcv_backend <- backend
  }

  if (identical(backend, "cpp_guarded")) {
    fastkpc_native_prepare_legacy_mgcv_cpp_backend()
    if (is.null(counter_env)) {
      metrics <- fastkpc_legacy_runtime_zero()
      legacy_env <- fastkpc_legacy_env()
    } else {
      if (is.null(counter_env$mgcv_cpp_metrics)) {
        counter_env$mgcv_cpp_metrics <- fastkpc_legacy_runtime_zero()
      }
      if (is.null(counter_env$mgcv_legacy_env)) {
        counter_env$mgcv_legacy_env <- fastkpc_legacy_env()
      }
      metrics <- counter_env$mgcv_cpp_metrics
      legacy_env <- counter_env$mgcv_legacy_env
    }
  }

  out <- matrix(NA_real_, nrow(data), nrow(requests))
  provider_cores_requested <- fastkpc_native_legacy_mgcv_provider_cores()
  provider_cores_used <- min(provider_cores_requested, nrow(requests))
  provider_parallel_enabled <- identical(backend, "r") &&
    identical(.Platform$OS.type, "unix") &&
    provider_cores_used > 1L
  if (!is.null(counter_env)) {
    parallel_cores_seen <- counter_env$provider_parallel_cores
    if (is.null(parallel_cores_seen)) parallel_cores_seen <- 0L
    parallel_level_count <- counter_env$provider_parallel_level_count
    if (is.null(parallel_level_count)) parallel_level_count <- 0L
    parallel_request_count <- counter_env$provider_parallel_request_count
    if (is.null(parallel_request_count)) parallel_request_count <- 0L
    counter_env$provider_parallel_enabled <-
      isTRUE(counter_env$provider_parallel_enabled) ||
      isTRUE(provider_parallel_enabled)
    counter_env$provider_parallel_cores <- max(
      as.integer(parallel_cores_seen),
      if (isTRUE(provider_parallel_enabled)) provider_cores_used else 0L
    )
    counter_env$provider_parallel_level_count <-
      as.integer(parallel_level_count) +
      as.integer(isTRUE(provider_parallel_enabled))
    counter_env$provider_parallel_request_count <-
      as.integer(parallel_request_count) +
      if (isTRUE(provider_parallel_enabled)) nrow(requests) else 0L
  }
  if (isTRUE(provider_parallel_enabled)) {
    residual_list <- parallel::mclapply(
      seq_len(nrow(requests)),
      function(i) {
        S <- as.integer(requests$conditioning_sets[[i]])
        if (length(S) == 0L) {
          stop("legacy mgcv residual provider received unconditional request",
               call. = FALSE)
        }
        target <- as.integer(requests$target[[i]])
        fastkpc_legacy_mgcv_residual(
          data = data,
          target = target,
          S = S
        )
      },
      mc.cores = provider_cores_used,
      mc.preschedule = TRUE
    )
    for (i in seq_along(residual_list)) {
      out[, i] <- residual_list[[i]]
    }
    return(out)
  }
  for (i in seq_len(nrow(requests))) {
    S <- as.integer(requests$conditioning_sets[[i]])
    if (length(S) == 0L) {
      stop("legacy mgcv residual provider received unconditional request",
           call. = FALSE)
    }
    target <- as.integer(requests$target[[i]])
    if (identical(backend, "cpp_guarded")) {
      backend_result <- fastkpc_legacy_mgcv_residual_cpp_backend_target(
        metrics = metrics,
        target_data = data[, target, drop = FALSE],
        s_data = data[, S, drop = FALSE],
        env = legacy_env,
        condition_threshold = condition_threshold,
        native_s_size_limit = native_s_size_limit,
        target = target,
        S = S
      )
      metrics <- backend_result$metrics
      if (!is.null(counter_env)) counter_env$mgcv_cpp_metrics <- metrics
      out[, i] <- backend_result$residual
    } else {
      out[, i] <- fastkpc_legacy_mgcv_residual(
        data = data,
        target = target,
        S = S
      )
    }
  }
  out
}

fastkpc_legacy_mgcv_residual_provider <- function(
    data, counter_env = NULL,
    backend = fastkpc_native_legacy_mgcv_residual_backend()) {
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  backend <- match.arg(backend, c("r", "cpp_guarded"))
  force(counter_env)
  function(requests, level) {
    fastkpc_legacy_mgcv_residual_provider_matrix(
      data = data,
      requests = requests,
      counter_env = counter_env,
      backend = backend
    )
  }
}

fastkpc_legacy_mgcv_residual_batch_provider <- function(data,
                                                        counter_env = NULL,
                                                        backend =
                                                          fastkpc_native_legacy_mgcv_residual_backend()) {
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  backend <- match.arg(backend, c("r", "cpp_guarded"))
  force(counter_env)
  function(requests, level) {
    residuals <- fastkpc_legacy_mgcv_residual_provider_matrix(
      data = data,
      requests = requests,
      counter_env = counter_env,
      backend = backend
    )
    list(
      residuals = residuals,
      contract = "level-residual-matrix-v1",
      backend = fastkpc_native_provider_backend_label(backend),
      mgcv_backend = backend,
      level = as.integer(level),
      request_count = as.integer(nrow(requests)),
      n = as.integer(nrow(data))
    )
  }
}

fastkpc_native_attach_mgcv_provider_summary <- function(result, counter_env,
                                                       backend) {
  backend <- match.arg(backend, c("r", "cpp_guarded"))
  result$summary$residual_provider_mgcv_backend <- backend
  enabled <- identical(backend, "cpp_guarded")
  result$summary$residual_provider_mgcv_cpp_backend_enabled <- enabled
  metrics <- fastkpc_native_counter_value(counter_env, "mgcv_cpp_metrics", NULL)
  metric_value <- function(name, default = 0) {
    if (is.null(metrics) || is.null(metrics[[name]])) return(default)
    metrics[[name]]
  }
  result$summary$residual_provider_mgcv_cpp_backend_count <-
    as.integer(metric_value("mgcv_cpp_backend_count", 0L))
  result$summary$residual_provider_mgcv_cpp_backend_native_count <-
    as.integer(metric_value("mgcv_cpp_backend_native_count", 0L))
  result$summary$residual_provider_mgcv_cpp_backend_fallback_count <-
    as.integer(metric_value("mgcv_cpp_backend_fallback_count", 0L))
  result$summary$residual_provider_mgcv_cpp_backend_error_count <-
    as.integer(metric_value("mgcv_cpp_backend_error_count", 0L))
  result$summary$residual_provider_mgcv_cpp_backend_high_condition_fallback_count <-
    as.integer(metric_value("mgcv_cpp_backend_high_condition_fallback_count", 0L))
  result$summary$residual_provider_mgcv_cpp_backend_outside_envelope_fallback_count <-
    as.integer(metric_value("mgcv_cpp_backend_outside_envelope_fallback_count", 0L))
  result$summary$residual_provider_mgcv_cpp_backend_ms <-
    as.numeric(metric_value("mgcv_cpp_backend_ms", 0))
  result$summary$residual_provider_mgcv_cpp_backend_native_solve_ms <-
    as.numeric(metric_value("mgcv_cpp_backend_native_solve_ms", 0))
  result$summary$residual_provider_parallel_enabled <-
    isTRUE(fastkpc_native_counter_value(
      counter_env, "provider_parallel_enabled", FALSE
    ))
  result$summary$residual_provider_parallel_cores <-
    as.integer(fastkpc_native_counter_value(
      counter_env, "provider_parallel_cores", 0L
    ))
  result$summary$residual_provider_parallel_level_count <-
    as.integer(fastkpc_native_counter_value(
      counter_env, "provider_parallel_level_count", 0L
    ))
  result$summary$residual_provider_parallel_request_count <-
    as.integer(fastkpc_native_counter_value(
      counter_env, "provider_parallel_request_count", 0L
    ))
  result
}

precision_run_skeleton_legacy_mgcv_legacy_dcov_native <- function(
    data, alpha, max_conditioning_size,
    index = 1, numCol = floor(nrow(as.matrix(data)) / 10),
    trace_level = c("summary", "full", "logical", "none"),
    dcov_batch = c("env", "none", "level", "canonical", "round")) {
  dcov_batch <- match.arg(dcov_batch)
  old_dcov_batch <- Sys.getenv("FASTKPC_NATIVE_LEGACY_DCOV_BATCH",
                               unset = NA_character_)
  if (dcov_batch %in% c("level", "canonical", "round")) {
    Sys.setenv(FASTKPC_NATIVE_LEGACY_DCOV_BATCH = dcov_batch)
    on.exit({
      if (is.na(old_dcov_batch)) {
        Sys.unsetenv("FASTKPC_NATIVE_LEGACY_DCOV_BATCH")
      } else {
        Sys.setenv(FASTKPC_NATIVE_LEGACY_DCOV_BATCH = old_dcov_batch)
      }
    }, add = TRUE)
  } else if (identical(dcov_batch, "none")) {
    Sys.unsetenv("FASTKPC_NATIVE_LEGACY_DCOV_BATCH")
    on.exit({
      if (is.na(old_dcov_batch)) {
        Sys.unsetenv("FASTKPC_NATIVE_LEGACY_DCOV_BATCH")
      } else {
        Sys.setenv(FASTKPC_NATIVE_LEGACY_DCOV_BATCH = old_dcov_batch)
      }
    }, add = TRUE)
  }
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  provider_backend <- fastkpc_native_legacy_mgcv_residual_backend()
  provider_counter <- new.env(parent = emptyenv())
  result <- precision_run_skeleton_residual_provider_legacy_dcov_native(
    data = data,
    alpha = alpha,
    max_conditioning_size = max_conditioning_size,
    residual_provider = fastkpc_legacy_mgcv_residual_batch_provider(
      data,
      counter_env = provider_counter,
      backend = provider_backend
    ),
    index = index,
    numCol = numCol,
    trace_level = trace_level
  )
  result <- fastkpc_native_attach_mgcv_provider_summary(
    result = result,
    counter_env = provider_counter,
    backend = provider_backend
  )
  result$summary$entrypoint <- "legacy-mgcv-legacy-dcov-native"
  result$summary$residual_provider_hidden <- TRUE
  result
}

fast_orient_wanpdag_cuda <- function(
    skeleton_result, data, alpha = 0.2,
    residual_backend = "fastSpline",
    orientation_residual_device = c("auto", "cpu", "cuda"),
    residual_cache = TRUE, index = 1, legacy_index = TRUE,
    orientation_batch_size = 0, orientation_diagnostics = TRUE,
    orient_collider = TRUE, solve_confl = FALSE,
    rules = c(TRUE, TRUE, TRUE), fastspline_params = list(),
    cuda_residual_fallback = TRUE, ci_method = "dcc.gamma",
    hsic_params = list(), permutation_params = list(),
    ci_diagnostics = TRUE) {
  load_fastkpc_cuda_native()
  orientation_residual_device <- match.arg(orientation_residual_device)
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  .Call(
    "C_fast_orient_wanpdag_cuda",
    data,
    as.matrix(skeleton_result$adjacency),
    skeleton_result$sepsets,
    as.numeric(alpha),
    as.numeric(index),
    isTRUE(legacy_index),
    isTRUE(residual_cache),
    as.character(residual_backend),
    as.character(orientation_residual_device),
    as.integer(orientation_batch_size),
    isTRUE(orientation_diagnostics),
    fastspline_params,
    isTRUE(cuda_residual_fallback),
    isTRUE(orient_collider),
    isTRUE(solve_confl),
    as.logical(rules),
    as.character(ci_method),
    hsic_params,
    permutation_params,
    isTRUE(ci_diagnostics),
    PACKAGE = "fastkpc_cuda"
  )
}

fast_kpc_wanpdag_cuda <- function(data, alpha, max_conditioning_size,
                                  residual_backend = "fastSpline",
                                  residual_device = c("auto", "cpu", "cuda"),
                                  orientation_residual_device = c("auto", "cpu", "cuda"),
                                  residual_cache = TRUE,
                                  index = 1,
                                  legacy_index = TRUE,
                                  batch_size = 0,
                                  residual_batch_size = 0,
                                  orientation_batch_size = 0,
                                  scheduler = c("auto", "layer", "legacy"),
                                  scheduler_diagnostics = TRUE,
                                  orientation_diagnostics = TRUE,
                                  orient_collider = TRUE,
                                  solve_confl = FALSE,
                                  rules = c(TRUE, TRUE, TRUE),
                                  fastspline_params = list(),
                                  cuda_residual_fallback = TRUE,
                                  ci_method = "dcc.gamma",
                                  hsic_params = list(),
                                  permutation_params = list(),
                                  ci_diagnostics = TRUE) {
  load_fastkpc_cuda_native()
  residual_device <- match.arg(residual_device)
  orientation_residual_device <- match.arg(orientation_residual_device)
  scheduler <- match.arg(scheduler)
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  .Call("C_fast_kpc_wanpdag_cuda", data, as.numeric(alpha),
        as.integer(max_conditioning_size), as.numeric(index),
        isTRUE(legacy_index), as.integer(batch_size),
        isTRUE(residual_cache), as.character(residual_backend),
        as.character(residual_device),
        as.character(orientation_residual_device),
        as.integer(residual_batch_size),
        as.integer(orientation_batch_size),
        as.character(scheduler), isTRUE(scheduler_diagnostics),
        isTRUE(orientation_diagnostics),
        fastspline_params,
        isTRUE(cuda_residual_fallback), isTRUE(orient_collider),
        isTRUE(solve_confl), as.logical(rules), as.character(ci_method),
        hsic_params, permutation_params, isTRUE(ci_diagnostics),
        PACKAGE = "fastkpc_cuda")
}

fixed_sp_cuda_runtime_create <- function(device_id = 0L) {
  load_fastkpc_cuda_native()
  .Call("C_fixed_sp_cuda_runtime_create", as.integer(device_id),
        PACKAGE = "fastkpc_cuda")
}

fixed_sp_cuda_live_owner_snapshot <- function() {
  load_fastkpc_cuda_native()
  .fastkpc_cuda_live_owner_snapshot()
}

fixed_sp_cuda_runtime_reserve <- function(
    runtime, n, null_dim, target_count, penalty_count, augmented_rows) {
  invisible(.Call(
    "C_fixed_sp_cuda_runtime_reserve", runtime, as.integer(n),
    as.integer(null_dim), as.integer(target_count), as.integer(penalty_count),
    as.integer(augmented_rows), PACKAGE = "fastkpc_cuda"
  ))
}

fixed_sp_cuda_runtime_info <- function(runtime) {
  load_fastkpc_cuda_native()
  .Call("C_fixed_sp_cuda_runtime_info", runtime, PACKAGE = "fastkpc_cuda")
}

fixed_sp_cuda_runtime_free <- function(runtime) {
  load_fastkpc_cuda_native()
  invisible(.Call("C_fixed_sp_cuda_runtime_free", runtime,
                  PACKAGE = "fastkpc_cuda"))
}

fixed_sp_cuda_prepared_create <- function(runtime, dto) {
  load_fastkpc_cuda_native()
  .Call("C_fixed_sp_cuda_prepared_create", runtime, dto,
        PACKAGE = "fastkpc_cuda")
}

fixed_sp_cuda_prepared_info <- function(handle) {
  load_fastkpc_cuda_native()
  .Call("C_fixed_sp_cuda_prepared_info", handle, PACKAGE = "fastkpc_cuda")
}

fixed_sp_cuda_prepared_materialize_roots_for_test <- function(handle) {
  load_fastkpc_cuda_native()
  .Call("C_fixed_sp_cuda_prepared_materialize_roots_for_test", handle,
        PACKAGE = "fastkpc_cuda")
}

fixed_sp_cuda_build_augmented_for_test <- function(
    handle, Y, SP, target_index) {
  load_fastkpc_cuda_native()
  .Call(
    "C_fixed_sp_cuda_build_augmented_for_test", handle,
    as.double(Y), as.double(SP), as.integer(target_index),
    PACKAGE = "fastkpc_cuda"
  )
}

fixed_sp_cuda_prepared_free <- function(handle) {
  load_fastkpc_cuda_native()
  invisible(.Call("C_fixed_sp_cuda_prepared_free", handle,
                  PACKAGE = "fastkpc_cuda"))
}

fixed_sp_cuda_solve_batch <- function(
    handle, Y, SP, planned_route, target_keys,
    outputs = c("residuals")) {
  load_fastkpc_cuda_native()
  .Call(
    "C_fixed_sp_cuda_solve_batch", handle, Y, SP,
    as.character(planned_route), as.character(target_keys),
    as.character(outputs), PACKAGE = "fastkpc_cuda"
  )
}

fixed_sp_cuda_residual_info <- function(token) {
  load_fastkpc_cuda_native()
  .Call("C_fixed_sp_cuda_residual_info", token, PACKAGE = "fastkpc_cuda")
}

fixed_sp_cuda_materialize_shadow <- function(
    token, outputs = c("residuals")) {
  load_fastkpc_cuda_native()
  .Call(
    "C_fixed_sp_cuda_materialize_shadow", token, as.character(outputs),
    PACKAGE = "fastkpc_cuda"
  )
}

fixed_sp_cuda_residual_release <- function(token) {
  load_fastkpc_cuda_native()
  invisible(.Call("C_fixed_sp_cuda_residual_release", token,
                  PACKAGE = "fastkpc_cuda"))
}

fixed_sp_cuda_residual_free <- function(token) {
  load_fastkpc_cuda_native()
  invisible(.Call("C_fixed_sp_cuda_residual_free", token,
                  PACKAGE = "fastkpc_cuda"))
}
