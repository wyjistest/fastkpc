if (!exists("fastkpc_full_cuda_census_logical_hash", mode = "function")) {
  source("fastkpc/R/full_cuda_ci_workload_census.R")
}

fastkpc_full_cuda_shadow_logical_contract <- function(logical_tests) {
  if (!is.data.frame(logical_tests) || nrow(logical_tests) == 0L) {
    stop("logical contract source must be a nonempty data frame",
         call. = FALSE)
  }
  list(
    schema_version = "full-cuda-ci-shadow-logical-corpus-contract-v1",
    logical_test_count = as.integer(nrow(logical_tests)),
    canonical_logical_census_hash =
      fastkpc_full_cuda_census_logical_hash(logical_tests)
  )
}

fastkpc_full_cuda_shadow_canonical_logical_contract <- function() {
  phase0_contract <- fastkpc_full_cuda_canonical_contract()
  phase1_contract <- fastkpc_full_cuda_census_input_contract()
  list(
    schema_version = "full-cuda-ci-shadow-logical-corpus-contract-v1",
    logical_test_count = as.integer(sum(phase0_contract$n_edgetests)),
    canonical_logical_census_hash =
      as.character(phase1_contract$canonical_logical_census_hash)
  )
}

fastkpc_full_cuda_shadow_validate_logical_contract <- function(
    logical_tests, expected_logical_contract) {
  required <- c(
    "schema_version", "logical_test_count",
    "canonical_logical_census_hash"
  )
  if (!is.list(expected_logical_contract) ||
      length(setdiff(required, names(expected_logical_contract))) > 0L ||
      !identical(
        expected_logical_contract$schema_version,
        "full-cuda-ci-shadow-logical-corpus-contract-v1"
      )) {
    stop("expected logical corpus identity contract is invalid",
         call. = FALSE)
  }
  expected_count <- expected_logical_contract$logical_test_count
  expected_hash <- expected_logical_contract$canonical_logical_census_hash
  if (!is.integer(expected_count) || length(expected_count) != 1L ||
      is.na(expected_count) || expected_count < 1L ||
      !is.character(expected_hash) || length(expected_hash) != 1L ||
      is.na(expected_hash) ||
      !grepl("^[0-9a-f]{64}$", expected_hash)) {
    stop("expected logical corpus identity contract is invalid",
         call. = FALSE)
  }
  actual_count <- as.integer(nrow(logical_tests))
  if (!identical(actual_count, expected_count)) {
    stop(
      "logical corpus identity count mismatch: expected ", expected_count,
      "; observed ", actual_count,
      call. = FALSE
    )
  }
  actual_hash <- fastkpc_full_cuda_census_logical_hash(logical_tests)
  if (!identical(actual_hash, expected_hash)) {
    stop(
      "logical corpus identity hash mismatch: expected ", expected_hash,
      "; observed ", actual_hash,
      call. = FALSE
    )
  }
  invisible(TRUE)
}

fastkpc_full_cuda_shadow_authenticated_phase2_records <- function(catalog) {
  dependencies <- c(
    "fastkpc_full_cuda_fixed_sp_catalog_contract",
    "fastkpc_full_cuda_fixed_sp_capture_phase2_files",
    "fastkpc_full_cuda_fixed_sp_parse_phase2_files",
    "fastkpc_full_cuda_prepared_s_validate_target_state_frame_schema"
  )
  if (any(!vapply(dependencies, exists, logical(1L), mode = "function",
                  inherits = TRUE))) {
    stop("authenticated Phase 2 association evidence helpers are unavailable",
         call. = FALSE)
  }
  contract <- fastkpc_full_cuda_fixed_sp_catalog_contract()
  target_state_file <- "target_state_index.rds"
  capture_contract <- contract
  capture_contract$phase2_file_sha256 <- c(
    contract$phase2_file_sha256,
    setNames(contract$target_state_index_rds_sha256, target_state_file)
  )
  captured <- fastkpc_full_cuda_fixed_sp_capture_phase2_files(
    catalog$phase2_dir, capture_contract
  )
  phase2_objects <- fastkpc_full_cuda_fixed_sp_parse_phase2_files(captured)
  setup_index <- phase2_objects$setup_index
  setup_fields <- c(
    "same_S_group_id", "prepared_s_key_sha256",
    "phase1_setup_fingerprint"
  )
  setup_records <- setup_index[, setup_fields, drop = FALSE]
  setup_sha256_columns <- all(vapply(setup_records, function(column) {
    typeof(column) == "character" && !is.object(column) &&
      is.null(attributes(column)) && !anyNA(column) &&
      all(grepl("^[0-9a-f]{64}$", column))
  }, logical(1L)))
  setup_clean <- is.data.frame(setup_index) &&
    nrow(setup_records) == 8634L &&
    length(setdiff(
      c("schema_version", setup_fields), names(setup_index)
    )) == 0L &&
    all(setup_index$schema_version ==
          "full-cuda-ci-prepared-s-setup-v1") && setup_sha256_columns &&
    !anyDuplicated(setup_records$same_S_group_id) &&
    !anyDuplicated(setup_records$prepared_s_key_sha256) &&
    identical(
      as.character(captured$hashes[["prepared_s_setup_index.csv"]]),
      as.character(
        contract$phase2_file_sha256[["prepared_s_setup_index.csv"]]
      )
    )
  if (!isTRUE(setup_clean)) {
    stop("authenticated Phase 2 setup association evidence is malformed",
         call. = FALSE)
  }
  setup_records <- setup_records[order(
    setup_records$prepared_s_key_sha256, method = "radix"
  ), , drop = FALSE]
  rownames(setup_records) <- NULL
  setup_evidence <- list(
    schema_version = "full-cuda-ci-shadow-setup-association-v1",
    phase2_setup_index_csv_sha256 = unname(
      captured$hashes[["prepared_s_setup_index.csv"]]
    ),
    association_sha256 =
      fastkpc_full_cuda_census_frame_hash(setup_records),
    records = setup_records
  )

  target_connection <- gzcon(rawConnection(
    captured$raw[[target_state_file]], open = "rb"
  ))
  on.exit(close(target_connection), add = TRUE)
  target_state_index <- tryCatch(
    readRDS(target_connection),
    error = function(error) {
      stop("authenticated Phase 2 target association evidence is malformed",
           call. = FALSE)
    }
  )
  tryCatch(
    fastkpc_full_cuda_prepared_s_validate_target_state_frame_schema(
      target_state_index
    ),
    error = function(error) {
      stop("authenticated Phase 2 target association evidence is malformed",
           call. = FALSE)
    }
  )
  target_fields <- c(
    "residual_key_sha256", "target", "same_S_group_id",
    "prepared_s_key_sha256", "phase1_setup_fingerprint"
  )
  target_records <- target_state_index[, target_fields, drop = FALSE]
  target_sha256_fields <- setdiff(target_fields, "target")
  target_sha256_columns <- all(vapply(
    target_records[target_sha256_fields], function(column) {
      typeof(column) == "character" && !is.object(column) &&
        is.null(attributes(column)) && !anyNA(column) &&
        all(grepl("^[0-9a-f]{64}$", column))
    }, logical(1L)
  ))
  target_setup_match <- match(
    target_records$same_S_group_id, setup_records$same_S_group_id
  )
  target_clean <- nrow(target_records) == contract$target_state_count &&
    target_sha256_columns &&
    typeof(target_records$target) == "integer" &&
    !is.object(target_records$target) &&
    is.null(attributes(target_records$target)) &&
    !anyNA(target_records$target) &&
    all(target_records$target >= 1L & target_records$target <= 48L) &&
    !anyDuplicated(target_records$residual_key_sha256) &&
    !anyNA(target_setup_match) &&
    identical(
      target_records$prepared_s_key_sha256,
      setup_records$prepared_s_key_sha256[target_setup_match]
    ) && identical(
      target_records$phase1_setup_fingerprint,
      setup_records$phase1_setup_fingerprint[target_setup_match]
    ) && identical(
      unname(captured$hashes[[target_state_file]]),
      unname(contract$target_state_index_rds_sha256)
    ) && identical(
      unname(contract$target_state_index_rds_sha256),
      unname(contract$phase2_semantic_file_sha256[[
        "target_state_index_rds"
      ]])
    )
  if (!isTRUE(target_clean)) {
    stop("authenticated Phase 2 target association evidence is malformed",
         call. = FALSE)
  }
  target_records <- target_records[order(
    target_records$residual_key_sha256, method = "radix"
  ), , drop = FALSE]
  rownames(target_records) <- NULL
  target_evidence <- list(
    schema_version = "full-cuda-ci-shadow-target-association-v1",
    phase2_target_state_index_rds_sha256 = unname(
      captured$hashes[[target_state_file]]
    ),
    association_sha256 =
      fastkpc_full_cuda_census_frame_hash(target_records),
    records = target_records
  )
  list(setup = setup_evidence, target = target_evidence)
}

fastkpc_full_cuda_shadow_authenticated_setup_records <- function(catalog) {
  fastkpc_full_cuda_shadow_authenticated_phase2_records(catalog)$setup
}

fastkpc_full_cuda_shadow_map_execution_units <- function(
    logical_tests, setup_rows, target_rows, setup_index,
    expected_logical_contract, authenticated_setup_evidence,
    authenticated_target_evidence,
    shard_count = 64L) {
  required_logical <- c(
    "logical_sequence_id", "level", "x", "y", "S_key",
    "formula_class", "residual_key_x", "residual_key_y"
  )
  required_setup <- c(
    "same_S_group_id", "setup_fingerprint", "prepared_s_key_sha256"
  )
  required_target <- c(
    "residual_key_sha256", "same_S_group_id", "setup_fingerprint",
    "prepared_s_key_sha256", "target"
  )
  required_setup_index <- c(
    "same_S_group_id", "prepared_s_key_sha256"
  )
  if (!is.data.frame(logical_tests) ||
      length(setdiff(required_logical, names(logical_tests))) > 0L) {
    stop("shadow logical-test mapping input is malformed", call. = FALSE)
  }
  logical_sequence_id <- fastkpc_full_cuda_shadow_integer(
    logical_tests$logical_sequence_id, "logical_sequence_id"
  )
  if (anyDuplicated(logical_sequence_id)) {
    stop("logical_tests contains duplicate logical_sequence_id",
         call. = FALSE)
  }
  if (!identical(logical_sequence_id, seq_len(nrow(logical_tests)))) {
    stop("logical_tests must be in complete canonical logical_sequence_id order",
         call. = FALSE)
  }
  fastkpc_full_cuda_shadow_validate_logical_contract(
    logical_tests, expected_logical_contract
  )
  if (!is.data.frame(setup_rows) ||
      length(setdiff(required_setup, names(setup_rows))) > 0L ||
      !is.data.frame(target_rows) ||
      length(setdiff(required_target, names(target_rows))) > 0L ||
      !is.data.frame(setup_index) ||
      length(setdiff(required_setup_index, names(setup_index))) > 0L ||
      typeof(shard_count) != "integer" || length(shard_count) != 1L ||
      is.object(shard_count) || !is.null(attributes(shard_count)) ||
      is.na(shard_count) || shard_count != 64L) {
    stop("shadow target/setup mapping indexes are malformed", call. = FALSE)
  }
  authenticated_fields <- c(
    "schema_version", "phase2_setup_index_csv_sha256",
    "association_sha256", "records"
  )
  authenticated_record_fields <- c(
    "same_S_group_id", "prepared_s_key_sha256",
    "phase1_setup_fingerprint"
  )
  authenticated_clean <- is.list(authenticated_setup_evidence) &&
    !is.object(authenticated_setup_evidence) && identical(
      names(authenticated_setup_evidence), authenticated_fields
    ) && identical(
      authenticated_setup_evidence$schema_version,
      "full-cuda-ci-shadow-setup-association-v1"
    ) && is.data.frame(authenticated_setup_evidence$records) &&
    identical(
      names(authenticated_setup_evidence$records),
      authenticated_record_fields
    ) && nrow(authenticated_setup_evidence$records) == 8634L &&
    grepl(
      "^[0-9a-f]{64}$",
      authenticated_setup_evidence$phase2_setup_index_csv_sha256
    ) && identical(
      authenticated_setup_evidence$association_sha256,
      fastkpc_full_cuda_census_frame_hash(
        authenticated_setup_evidence$records
      )
    )
  if (!isTRUE(authenticated_clean)) {
    stop("authenticated Phase 2 setup association evidence is malformed",
         call. = FALSE)
  }
  authenticated_target_fields <- c(
    "schema_version", "phase2_target_state_index_rds_sha256",
    "association_sha256", "records"
  )
  authenticated_target_record_fields <- c(
    "residual_key_sha256", "target", "same_S_group_id",
    "prepared_s_key_sha256", "phase1_setup_fingerprint"
  )
  target_contract <- fastkpc_full_cuda_fixed_sp_catalog_contract()
  authenticated_target_records <- authenticated_target_evidence$records
  authenticated_target_clean <-
    is.list(authenticated_target_evidence) &&
    !is.object(authenticated_target_evidence) && identical(
      names(authenticated_target_evidence), authenticated_target_fields
    ) && identical(
      authenticated_target_evidence$schema_version,
      "full-cuda-ci-shadow-target-association-v1"
    ) && is.data.frame(authenticated_target_records) && identical(
      names(authenticated_target_records), authenticated_target_record_fields
    ) && nrow(authenticated_target_records) == 110617L &&
    identical(
      authenticated_target_evidence$phase2_target_state_index_rds_sha256,
      unname(target_contract$target_state_index_rds_sha256)
    ) && identical(
      authenticated_target_evidence$association_sha256,
      fastkpc_full_cuda_census_frame_hash(authenticated_target_records)
    ) && identical(
      order(
        authenticated_target_records$residual_key_sha256, method = "radix"
      ),
      seq_len(nrow(authenticated_target_records))
    )
  if (!isTRUE(authenticated_target_clean)) {
    stop("authenticated Phase 2 target association evidence is malformed",
         call. = FALSE)
  }

  sha256_vector <- function(value) {
    typeof(value) == "character" && !is.object(value) &&
      is.null(attributes(value)) && !anyNA(value) &&
      all(grepl("^[0-9a-f]{64}$", value))
  }
  setup_group <- as.character(setup_rows$same_S_group_id)
  setup_key <- as.character(setup_rows$prepared_s_key_sha256)
  setup_fingerprint <- as.character(setup_rows$setup_fingerprint)
  target_key <- as.character(target_rows$residual_key_sha256)
  target_group <- as.character(target_rows$same_S_group_id)
  target_setup_key <- as.character(target_rows$prepared_s_key_sha256)
  target_setup_fingerprint <- as.character(target_rows$setup_fingerprint)
  index_group <- as.character(setup_index$same_S_group_id)
  index_key <- as.character(setup_index$prepared_s_key_sha256)
  authenticated_records <- authenticated_setup_evidence$records
  authenticated_group <- as.character(
    authenticated_records$same_S_group_id
  )
  authenticated_key <- as.character(
    authenticated_records$prepared_s_key_sha256
  )
  authenticated_fingerprint <- as.character(
    authenticated_records$phase1_setup_fingerprint
  )
  authenticated_target_key <- as.character(
    authenticated_target_records$residual_key_sha256
  )
  authenticated_target_group <- as.character(
    authenticated_target_records$same_S_group_id
  )
  authenticated_target_setup_key <- as.character(
    authenticated_target_records$prepared_s_key_sha256
  )
  authenticated_target_setup_fingerprint <- as.character(
    authenticated_target_records$phase1_setup_fingerprint
  )
  if (anyDuplicated(target_key)) {
    stop("shadow target index contains duplicate residual key",
         call. = FALSE)
  }
  clean_indexes <-
    nrow(setup_rows) == 8634L && nrow(target_rows) == 110617L &&
    nrow(setup_index) == 8634L &&
    all(vapply(
      list(
        setup_group, setup_key, setup_fingerprint, target_key,
        target_group, target_setup_key, target_setup_fingerprint,
        index_group, index_key, authenticated_group, authenticated_key,
        authenticated_fingerprint, authenticated_target_key,
        authenticated_target_group, authenticated_target_setup_key,
        authenticated_target_setup_fingerprint
      ),
      sha256_vector, logical(1L)
    )) &&
    !anyDuplicated(setup_group) && !anyDuplicated(setup_key) &&
    !anyDuplicated(index_group) &&
    !anyDuplicated(index_key)
  if (!isTRUE(clean_indexes)) {
    stop("shadow target/setup mapping indexes are malformed", call. = FALSE)
  }
  setup_index_match <- match(setup_group, index_group)
  target_setup_match <- match(target_group, setup_group)
  authenticated_setup_match <- match(setup_group, authenticated_group)
  if (anyNA(authenticated_setup_match) ||
      !identical(setup_key, authenticated_key[authenticated_setup_match]) ||
      !identical(
        setup_fingerprint,
        authenticated_fingerprint[authenticated_setup_match]
      )) {
    stop("authenticated Phase 2 setup association mismatch",
         call. = FALSE)
  }
  authenticated_target_match <- match(target_key, authenticated_target_key)
  if (anyNA(authenticated_target_match) ||
      !identical(
        target_rows$target,
        authenticated_target_records$target[authenticated_target_match]
      ) || !identical(
        target_group,
        authenticated_target_group[authenticated_target_match]
      ) || !identical(
        target_setup_key,
        authenticated_target_setup_key[authenticated_target_match]
      ) || !identical(
        target_setup_fingerprint,
        authenticated_target_setup_fingerprint[
          authenticated_target_match
        ]
      )) {
    stop("authenticated Phase 2 target association mismatch",
         call. = FALSE)
  }
  if (anyNA(setup_index_match) || anyNA(target_setup_match) ||
      !identical(setup_key, index_key[setup_index_match]) ||
      !identical(target_setup_key, setup_key[target_setup_match]) ||
      !identical(
        target_setup_fingerprint,
        setup_fingerprint[target_setup_match]
      )) {
    stop("shadow target/setup fingerprint mapping conflict",
         call. = FALSE)
  }

  level <- fastkpc_full_cuda_shadow_integer(
    logical_tests$level, "level", minimum = 0L
  )
  x <- fastkpc_full_cuda_shadow_integer(logical_tests$x, "x")
  y <- fastkpc_full_cuda_shadow_integer(logical_tests$y, "y")
  formula_class <- as.character(logical_tests$formula_class)
  S_key <- as.character(logical_tests$S_key)
  residual_key_x <- as.character(logical_tests$residual_key_x)
  residual_key_y <- as.character(logical_tests$residual_key_y)
  direct <- level == 0L
  conditional <- !direct
  direct_clean <- all(S_key[direct] == "") &&
    all(formula_class[direct] == "direct-ci") &&
    all(is.na(residual_key_x[direct])) &&
    all(is.na(residual_key_y[direct]))
  conditional_clean <- all(nzchar(S_key[conditional])) &&
    all(formula_class[conditional] %in% c(
      "full-smooth", "additive-smooth"
    )) &&
    !anyNA(residual_key_x[conditional]) &&
    !anyNA(residual_key_y[conditional]) &&
    all(nzchar(residual_key_x[conditional])) &&
    all(nzchar(residual_key_y[conditional]))
  if (!isTRUE(direct_clean)) {
    stop("direct logical rows must not carry residual keys", call. = FALSE)
  }
  if (!isTRUE(conditional_clean)) {
    stop("conditional logical rows require two residual keys",
         call. = FALSE)
  }

  conditional_rows <- which(conditional)
  endpoint_x <- match(residual_key_x[conditional_rows], target_key)
  endpoint_y <- match(residual_key_y[conditional_rows], target_key)
  if (anyNA(endpoint_x) || anyNA(endpoint_y)) {
    stop("conditional logical endpoint residual key is missing",
         call. = FALSE)
  }
  if (any(target_rows$target[endpoint_x] != x[conditional_rows]) ||
      any(target_rows$target[endpoint_y] != y[conditional_rows])) {
    stop("conditional residual key target endpoint conflict",
         call. = FALSE)
  }
  prepared_s_key_x <- target_setup_key[endpoint_x]
  prepared_s_key_y <- target_setup_key[endpoint_y]
  if (any(prepared_s_key_x != prepared_s_key_y)) {
    stop("conditional residual endpoints map to different PreparedSKeys",
         call. = FALSE)
  }
  canonical_setup_keys <- sort(index_key, method = "radix")
  setup_rank <- match(prepared_s_key_x, canonical_setup_keys)
  if (anyNA(setup_rank)) {
    stop("conditional PreparedSKey shard mapping is incomplete",
         call. = FALSE)
  }
  shard_id <- as.integer((setup_rank - 1L) %% shard_count)

  direct_tests <- logical_tests[direct, , drop = FALSE]
  direct_tests$prepared_s_key_x <- NA_character_
  direct_tests$prepared_s_key_y <- NA_character_
  direct_tests$shard_id <- NA_integer_
  conditional_tests <- logical_tests[conditional_rows, , drop = FALSE]
  conditional_tests$prepared_s_key_x <- prepared_s_key_x
  conditional_tests$prepared_s_key_y <- prepared_s_key_y
  conditional_tests$shard_id <- shard_id
  rownames(direct_tests) <- rownames(conditional_tests) <- NULL

  if (nrow(direct_tests) != 2213L ||
      nrow(conditional_tests) != 238276L ||
      length(unique(prepared_s_key_x)) != 8634L ||
      !identical(
        sort(c(
          direct_tests$logical_sequence_id,
          conditional_tests$logical_sequence_id
        )),
        seq_len(240489L)
      )) {
    stop("shadow execution-unit canonical coverage mismatch",
         call. = FALSE)
  }
  list(
    schema_version = "full-cuda-ci-shadow-plan-v1",
    logical_contract = expected_logical_contract,
    phase2_setup_index_csv_sha256 =
      authenticated_setup_evidence$phase2_setup_index_csv_sha256,
    setup_association_sha256 =
      authenticated_setup_evidence$association_sha256,
    phase2_target_state_index_rds_sha256 =
      authenticated_target_evidence$phase2_target_state_index_rds_sha256,
    target_association_sha256 =
      authenticated_target_evidence$association_sha256,
    shard_count = shard_count,
    direct_tests = direct_tests,
    conditional_tests = conditional_tests
  )
}

.fastkpc_full_cuda_shadow_plan_registry <-
  new.env(hash = TRUE, parent = emptyenv())

.fastkpc_full_cuda_shadow_plan_registry_max_entries <- function() {
  32L
}

.fastkpc_full_cuda_shadow_plan_metadata_fields <- function() {
  c(
    "schema_version", "logical_contract", "catalog_authority_sha256",
    "catalog_lineage_sha256", "route_config_sha256",
    "phase2_setup_index_csv_sha256", "setup_association_sha256",
    "phase2_target_state_index_rds_sha256", "target_association_sha256",
    "shard_count", "direct_tests_sha256", "conditional_tests_sha256"
  )
}

.fastkpc_full_cuda_shadow_plan_fields <- function() {
  c(
    .fastkpc_full_cuda_shadow_plan_metadata_fields(),
    "plan_identity_sha256", "authentication_token", "direct_tests",
    "conditional_tests"
  )
}

.fastkpc_full_cuda_shadow_bare_sha256 <- function(value) {
  typeof(value) == "character" && length(value) == 1L &&
    !is.object(value) && is.null(attributes(value)) && !is.na(value) &&
    grepl("^[0-9a-f]{64}$", value)
}

.fastkpc_full_cuda_shadow_plan_metadata_hash <- function(plan) {
  fields <- .fastkpc_full_cuda_shadow_plan_metadata_fields()
  if (!is.list(plan) || length(setdiff(fields, names(plan))) > 0L) {
    stop("shadow plan metadata is malformed", call. = FALSE)
  }
  fastkpc_full_cuda_census_named_metadata_hash(plan[fields])
}

.fastkpc_full_cuda_shadow_phase2_file_records <- function(
    catalog, setup_index_sha256, target_state_sha256) {
  if (!.fastkpc_full_cuda_shadow_bare_sha256(setup_index_sha256) ||
      !.fastkpc_full_cuda_shadow_bare_sha256(target_state_sha256) ||
      !is.list(catalog) || is.null(catalog$phase2_dir)) {
    stop("shadow plan Phase 2 file identity is malformed", call. = FALSE)
  }
  logical_path <- c(
    "prepared_s_setup_index.csv", "target_state_index.rds"
  )
  phase2_dir <- normalizePath(
    catalog$phase2_dir, winslash = "/", mustWork = TRUE
  )
  paths <- file.path(phase2_dir, logical_path)
  paths <- unname(vapply(
    paths, normalizePath, character(1L), winslash = "/", mustWork = TRUE
  ))
  if (!all(file.exists(paths)) || any(dir.exists(paths))) {
    stop("shadow plan Phase 2 file identity is unavailable", call. = FALSE)
  }
  info <- file.info(paths, extra_cols = FALSE)
  if (!is.data.frame(info) || nrow(info) != length(paths) ||
      anyNA(info$size) || anyNA(info$mtime) || anyNA(info$ctime)) {
    stop("shadow plan Phase 2 file identity is malformed", call. = FALSE)
  }
  records <- data.frame(
    logical_path = logical_path,
    path = paths,
    size = as.character(info$size),
    mtime = sprintf("%.6f", as.numeric(info$mtime)),
    ctime = sprintf("%.6f", as.numeric(info$ctime)),
    sha256 = c(setup_index_sha256, target_state_sha256),
    stringsAsFactors = FALSE
  )
  rownames(records) <- NULL
  records
}

.fastkpc_full_cuda_shadow_prune_plan_registry <- function(
    protect = character()) {
  registry <- .fastkpc_full_cuda_shadow_plan_registry
  limit <- .fastkpc_full_cuda_shadow_plan_registry_max_entries()
  keys <- ls(registry, all.names = TRUE)
  if (length(keys) <= limit) return(invisible(keys))
  ages <- vapply(keys, function(key) {
    entry <- get(key, envir = registry, inherits = FALSE)
    if (is.list(entry) && is.double(entry$registered_at) &&
        length(entry$registered_at) == 1L &&
        is.finite(entry$registered_at)) {
      entry$registered_at
    } else {
      -Inf
    }
  }, numeric(1L))
  removable <- keys[order(ages, method = "radix")]
  removable <- removable[!removable %in% protect]
  while (length(ls(registry, all.names = TRUE)) > limit &&
         length(removable) > 0L) {
    key <- removable[[1L]]
    if (bindingIsLocked(key, registry)) unlockBinding(key, registry)
    rm(list = key, envir = registry)
    removable <- removable[-1L]
  }
  invisible(ls(registry, all.names = TRUE))
}

.fastkpc_full_cuda_shadow_register_plan_identity <- function(
    metadata, plan_identity_sha256, phase2_file_records) {
  token <- new.env(parent = emptyenv())
  registry_id <- fastkpc_full_cuda_census_named_metadata_hash(list(
    schema_version = "full-cuda-ci-shadow-plan-registry-key-v1",
    process_id = as.integer(Sys.getpid()),
    token_environment = format(token),
    catalog_authority_sha256 = metadata$catalog_authority_sha256,
    plan_identity_sha256 = plan_identity_sha256
  ))
  registry <- .fastkpc_full_cuda_shadow_plan_registry
  if (exists(registry_id, envir = registry, inherits = FALSE)) {
    stop("shadow plan registry token collision", call. = FALSE)
  }
  assign(
    "schema_version", "full-cuda-ci-shadow-plan-token-v1", envir = token
  )
  assign("registry_id", registry_id, envir = token)
  assign(
    "catalog_authority_sha256", metadata$catalog_authority_sha256,
    envir = token
  )
  assign("plan_identity_sha256", plan_identity_sha256, envir = token)
  lockEnvironment(token, bindings = TRUE)

  entry <- list(
    schema_version = "full-cuda-ci-shadow-plan-registry-entry-v1",
    registered_at = as.double(Sys.time()),
    token = token,
    catalog_authority_sha256 = metadata$catalog_authority_sha256,
    catalog_lineage_sha256 = metadata$catalog_lineage_sha256,
    route_config_sha256 = metadata$route_config_sha256,
    plan_identity_sha256 = plan_identity_sha256,
    direct_tests_sha256 = metadata$direct_tests_sha256,
    conditional_tests_sha256 = metadata$conditional_tests_sha256,
    phase2_file_records = phase2_file_records
  )
  assign(registry_id, entry, envir = registry)
  lockBinding(registry_id, registry)
  .fastkpc_full_cuda_shadow_prune_plan_registry(protect = registry_id)
  token
}

.fastkpc_full_cuda_shadow_authenticate_plan <- function(
    mapped_plan, catalog, authority) {
  mapped_fields <- c(
    "schema_version", "logical_contract",
    "phase2_setup_index_csv_sha256", "setup_association_sha256",
    "phase2_target_state_index_rds_sha256", "target_association_sha256",
    "shard_count", "direct_tests", "conditional_tests"
  )
  clean <- is.list(mapped_plan) && !is.object(mapped_plan) &&
    identical(names(mapped_plan), mapped_fields) &&
    identical(mapped_plan$schema_version, "full-cuda-ci-shadow-plan-v1") &&
    is.data.frame(mapped_plan$direct_tests) &&
    is.data.frame(mapped_plan$conditional_tests) &&
    is.list(authority) &&
    .fastkpc_full_cuda_shadow_bare_sha256(authority$authority_sha256) &&
    is.list(authority$lineage) && isTRUE(authority$lineage$authenticated) &&
    identical(
      catalog$phase3_catalog_authority_sha256,
      authority$authority_sha256
    )
  if (!isTRUE(clean)) {
    stop("shadow planner authentication inputs are malformed", call. = FALSE)
  }
  expected_logical_contract <-
    fastkpc_full_cuda_shadow_canonical_logical_contract()
  if (!identical(mapped_plan$logical_contract, expected_logical_contract)) {
    stop("shadow planner logical contract authentication failed",
         call. = FALSE)
  }
  contract <- fastkpc_full_cuda_fixed_sp_catalog_contract()
  immutable_phase2 <- identical(
    mapped_plan$phase2_setup_index_csv_sha256,
    unname(contract$phase2_file_sha256[["prepared_s_setup_index.csv"]])
  ) && identical(
    mapped_plan$phase2_target_state_index_rds_sha256,
    unname(contract$target_state_index_rds_sha256)
  )
  if (!isTRUE(immutable_phase2)) {
    stop("shadow planner immutable Phase 2 identity mismatch",
         call. = FALSE)
  }
  route <- fastkpc_full_cuda_phase3_route_config()
  route_hash <- fastkpc_full_cuda_phase3_route_config_hash(route)
  if (!identical(route$dcov_backend, "legacy-cpp-spectra") ||
      !identical(route$cpu_fallback_allowed, FALSE) ||
      !identical(route$approximate_backend_allowed, FALSE)) {
    stop("shadow planner route authentication failed", call. = FALSE)
  }
  metadata <- list(
    schema_version = "full-cuda-ci-shadow-plan-v2",
    logical_contract = expected_logical_contract,
    catalog_authority_sha256 = authority$authority_sha256,
    catalog_lineage_sha256 =
      fastkpc_full_cuda_census_named_metadata_hash(authority$lineage),
    route_config_sha256 = route_hash,
    phase2_setup_index_csv_sha256 =
      mapped_plan$phase2_setup_index_csv_sha256,
    setup_association_sha256 = mapped_plan$setup_association_sha256,
    phase2_target_state_index_rds_sha256 =
      mapped_plan$phase2_target_state_index_rds_sha256,
    target_association_sha256 = mapped_plan$target_association_sha256,
    shard_count = mapped_plan$shard_count,
    direct_tests_sha256 =
      fastkpc_full_cuda_census_frame_hash(mapped_plan$direct_tests),
    conditional_tests_sha256 =
      fastkpc_full_cuda_census_frame_hash(mapped_plan$conditional_tests)
  )
  plan_identity_sha256 <-
    fastkpc_full_cuda_census_named_metadata_hash(metadata)
  phase2_file_records <- .fastkpc_full_cuda_shadow_phase2_file_records(
    catalog,
    metadata$phase2_setup_index_csv_sha256,
    metadata$phase2_target_state_index_rds_sha256
  )
  authentication_token <-
    .fastkpc_full_cuda_shadow_register_plan_identity(
      metadata, plan_identity_sha256, phase2_file_records
    )
  c(
    metadata,
    list(
      plan_identity_sha256 = plan_identity_sha256,
      authentication_token = authentication_token,
      direct_tests = mapped_plan$direct_tests,
      conditional_tests = mapped_plan$conditional_tests
    )
  )
}

.fastkpc_full_cuda_shadow_validate_supplied_plan_impl <- function(
    catalog, plan) {
  expected_fields <- .fastkpc_full_cuda_shadow_plan_fields()
  sha256_fields <- c(
    "catalog_authority_sha256", "catalog_lineage_sha256",
    "route_config_sha256", "phase2_setup_index_csv_sha256",
    "setup_association_sha256", "phase2_target_state_index_rds_sha256",
    "target_association_sha256", "direct_tests_sha256",
    "conditional_tests_sha256", "plan_identity_sha256"
  )
  clean_plan <- is.list(plan) && !is.object(plan) &&
    identical(names(plan), expected_fields) &&
    identical(plan$schema_version, "full-cuda-ci-shadow-plan-v2") &&
    all(vapply(
      plan[sha256_fields], .fastkpc_full_cuda_shadow_bare_sha256, logical(1L)
    )) && is.integer(plan$shard_count) && length(plan$shard_count) == 1L &&
    !is.object(plan$shard_count) && is.null(attributes(plan$shard_count)) &&
    identical(plan$shard_count, 64L) && is.data.frame(plan$direct_tests) &&
    is.data.frame(plan$conditional_tests)
  if (!isTRUE(clean_plan)) stop("malformed supplied shadow plan")

  token <- plan$authentication_token
  token_fields <- c(
    "catalog_authority_sha256", "plan_identity_sha256", "registry_id",
    "schema_version"
  )
  clean_token <- is.environment(token) && environmentIsLocked(token) &&
    identical(ls(token, all.names = TRUE), token_fields) &&
    all(vapply(token_fields, bindingIsLocked, logical(1L), env = token)) &&
    identical(token$schema_version, "full-cuda-ci-shadow-plan-token-v1") &&
    .fastkpc_full_cuda_shadow_bare_sha256(token$registry_id) &&
    identical(token$catalog_authority_sha256,
              plan$catalog_authority_sha256) &&
    identical(token$plan_identity_sha256, plan$plan_identity_sha256)
  if (!isTRUE(clean_token)) stop("invalid supplied shadow plan token")

  registry <- .fastkpc_full_cuda_shadow_plan_registry
  if (!exists(token$registry_id, envir = registry, inherits = FALSE) ||
      !bindingIsLocked(token$registry_id, registry)) {
    stop("missing supplied shadow plan registry entry")
  }
  entry <- get(token$registry_id, envir = registry, inherits = FALSE)
  entry_fields <- c(
    "schema_version", "registered_at", "token",
    "catalog_authority_sha256", "catalog_lineage_sha256",
    "route_config_sha256", "plan_identity_sha256",
    "direct_tests_sha256", "conditional_tests_sha256",
    "phase2_file_records"
  )
  clean_entry <- is.list(entry) && !is.object(entry) &&
    identical(names(entry), entry_fields) &&
    identical(
      entry$schema_version,
      "full-cuda-ci-shadow-plan-registry-entry-v1"
    ) && is.double(entry$registered_at) &&
    length(entry$registered_at) == 1L && is.finite(entry$registered_at) &&
    identical(entry$token, token) &&
    identical(entry$catalog_authority_sha256,
              plan$catalog_authority_sha256) &&
    identical(entry$catalog_lineage_sha256,
              plan$catalog_lineage_sha256) &&
    identical(entry$route_config_sha256, plan$route_config_sha256) &&
    identical(entry$plan_identity_sha256, plan$plan_identity_sha256) &&
    identical(entry$direct_tests_sha256, plan$direct_tests_sha256) &&
    identical(entry$conditional_tests_sha256,
              plan$conditional_tests_sha256)
  if (!isTRUE(clean_entry)) stop("stale supplied shadow plan registry entry")

  metadata_hash <- .fastkpc_full_cuda_shadow_plan_metadata_hash(plan)
  if (!identical(metadata_hash, plan$plan_identity_sha256) ||
      !identical(metadata_hash, entry$plan_identity_sha256)) {
    stop("supplied shadow plan metadata identity mismatch")
  }

  authority <- fastkpc_full_cuda_phase3_discover_catalog_authority(catalog)
  lineage_hash <-
    fastkpc_full_cuda_census_named_metadata_hash(authority$lineage)
  route <- fastkpc_full_cuda_phase3_route_config()
  route_hash <- fastkpc_full_cuda_phase3_route_config_hash(route)
  current_binding <-
    is.list(authority) && is.list(authority$lineage) &&
    isTRUE(authority$lineage$authenticated) &&
    identical(catalog$phase3_catalog_authority_sha256,
              authority$authority_sha256) &&
    identical(plan$catalog_authority_sha256,
              authority$authority_sha256) &&
    identical(plan$catalog_lineage_sha256, lineage_hash) &&
    identical(plan$route_config_sha256, route_hash) &&
    identical(route$dcov_backend, "legacy-cpp-spectra") &&
    identical(route$cpu_fallback_allowed, FALSE) &&
    identical(route$approximate_backend_allowed, FALSE)
  if (!isTRUE(current_binding)) {
    stop("supplied shadow plan catalog or route binding mismatch")
  }

  contract <- fastkpc_full_cuda_fixed_sp_catalog_contract()
  immutable_phase2 <- identical(
    plan$phase2_setup_index_csv_sha256,
    unname(contract$phase2_file_sha256[["prepared_s_setup_index.csv"]])
  ) && identical(
    plan$phase2_target_state_index_rds_sha256,
    unname(contract$target_state_index_rds_sha256)
  )
  current_phase2_file_records <-
    .fastkpc_full_cuda_shadow_phase2_file_records(
      catalog,
      entry$phase2_file_records$sha256[[1L]],
      entry$phase2_file_records$sha256[[2L]]
    )
  if (!isTRUE(immutable_phase2) ||
      !identical(current_phase2_file_records, entry$phase2_file_records)) {
    stop("supplied shadow plan Phase 2 file identity mismatch")
  }

  direct_tests_sha256 <-
    fastkpc_full_cuda_census_frame_hash(plan$direct_tests)
  conditional_tests_sha256 <-
    fastkpc_full_cuda_census_frame_hash(plan$conditional_tests)
  if (!identical(direct_tests_sha256, plan$direct_tests_sha256) ||
      !identical(direct_tests_sha256, entry$direct_tests_sha256) ||
      !identical(conditional_tests_sha256,
                 plan$conditional_tests_sha256) ||
      !identical(conditional_tests_sha256,
                 entry$conditional_tests_sha256)) {
    stop("supplied shadow plan frame identity mismatch")
  }

  expected_logical_contract <-
    fastkpc_full_cuda_shadow_canonical_logical_contract()
  logical_tests <- catalog$inputs$logical_tests
  if (!identical(plan$logical_contract, expected_logical_contract) ||
      !identical(
        as.character(catalog$inputs$manifest$canonical_logical_census_hash),
        expected_logical_contract$canonical_logical_census_hash
      )) {
    stop("supplied shadow plan logical contract mismatch")
  }
  fastkpc_full_cuda_shadow_validate_logical_contract(
    logical_tests, expected_logical_contract
  )
  expected_direct <- logical_tests[logical_tests$level == 0L, , drop = FALSE]
  rownames(expected_direct) <- NULL
  logical_fields <- names(logical_tests)
  plan_row_fields <- c(
    logical_fields, "prepared_s_key_x", "prepared_s_key_y", "shard_id"
  )
  exact_direct <- nrow(expected_direct) == 2213L &&
    nrow(plan$direct_tests) == 2213L &&
    nrow(plan$conditional_tests) == 238276L &&
    identical(names(plan$direct_tests), plan_row_fields) &&
    identical(names(plan$conditional_tests), plan_row_fields) &&
    all(vapply(logical_fields, function(field) {
      identical(plan$direct_tests[[field]], expected_direct[[field]])
    }, logical(1L))) &&
    all(is.na(plan$direct_tests$prepared_s_key_x)) &&
    all(is.na(plan$direct_tests$prepared_s_key_y)) &&
    all(is.na(plan$direct_tests$shard_id))
  if (!isTRUE(exact_direct)) {
    stop("supplied shadow plan direct logical lineage mismatch")
  }
  invisible(plan)
}

.fastkpc_full_cuda_shadow_validate_supplied_plan <- function(catalog, plan) {
  tryCatch(
    .fastkpc_full_cuda_shadow_validate_supplied_plan_impl(catalog, plan),
    error = function(error) {
      stop(
        "direct-CI execution plan does not match authenticated catalog",
        call. = FALSE
      )
    }
  )
}

fastkpc_full_cuda_shadow_plan <- function(catalog) {
  dependencies <- c(
    "fastkpc_full_cuda_phase3_discover_catalog_authority",
    "fastkpc_full_cuda_fixed_sp_scope"
  )
  if (any(!vapply(dependencies, exists, logical(1L), mode = "function",
                  inherits = TRUE))) {
    stop("full CUDA fixed-sp catalog helpers are unavailable",
         call. = FALSE)
  }
  authority <- fastkpc_full_cuda_phase3_discover_catalog_authority(catalog)
  if (!is.list(authority) || !is.list(authority$lineage) ||
      !isTRUE(authority$lineage$authenticated)) {
    stop("shadow planner requires an authenticated catalog",
         call. = FALSE)
  }
  logical_tests <- catalog$inputs$logical_tests
  expected_contract <- fastkpc_full_cuda_shadow_canonical_logical_contract()
  if (!identical(
        as.character(catalog$inputs$manifest$canonical_logical_census_hash),
        expected_contract$canonical_logical_census_hash
      )) {
    stop("shadow planner Phase 1 logical corpus lineage mismatch",
         call. = FALSE)
  }
  full_scope <- fastkpc_full_cuda_fixed_sp_scope(catalog, "full")
  authenticated_phase2_evidence <-
    fastkpc_full_cuda_shadow_authenticated_phase2_records(catalog)
  mapped_plan <- fastkpc_full_cuda_shadow_map_execution_units(
    logical_tests = logical_tests,
    setup_rows = full_scope$setup_rows,
    target_rows = full_scope$target_rows,
    setup_index = catalog$setup_index,
    expected_logical_contract = expected_contract,
    authenticated_setup_evidence = authenticated_phase2_evidence$setup,
    authenticated_target_evidence = authenticated_phase2_evidence$target,
    shard_count = as.integer(catalog$catalog_contract$shard_count)
  )
  .fastkpc_full_cuda_shadow_authenticate_plan(
    mapped_plan, catalog, authority
  )
}

fastkpc_full_cuda_shadow_integer <- function(value, field, minimum = 1L) {
  if (!is.numeric(value)) {
    stop("logical_tests contains invalid ", field, call. = FALSE)
  }
  numeric_value <- suppressWarnings(as.numeric(value))
  integer_value <- suppressWarnings(as.integer(value))
  if (any(!is.finite(numeric_value)) || anyNA(integer_value) ||
      any(numeric_value != integer_value) || any(integer_value < minimum)) {
    stop("logical_tests contains invalid ", field, call. = FALSE)
  }
  integer_value
}

fastkpc_full_cuda_shadow_labels <- function(labels) {
  if (!is.character(labels) || length(labels) < 2L || anyNA(labels) ||
      any(!nzchar(labels)) || anyDuplicated(labels)) {
    stop("labels must be unique nonempty character values", call. = FALSE)
  }
  labels
}

fastkpc_full_cuda_replay_logical_ci <- function(
    logical_tests, candidate_p_value, labels,
    expected_logical_contract =
      fastkpc_full_cuda_shadow_canonical_logical_contract()) {
  if (!is.data.frame(logical_tests) || nrow(logical_tests) == 0L) {
    stop("logical_tests must be a nonempty data frame", call. = FALSE)
  }
  required <- c(
    "logical_sequence_id", "source_sequence_id", "source_task_index",
    "level", "x", "y", "S_key", "alpha", "reference_decision"
  )
  missing <- setdiff(required, names(logical_tests))
  if (length(missing) > 0L) {
    stop("logical_tests missing fields: ", paste(missing, collapse = ","),
         call. = FALSE)
  }
  labels <- fastkpc_full_cuda_shadow_labels(labels)
  row_count <- nrow(logical_tests)
  if (!is.numeric(candidate_p_value) ||
      length(candidate_p_value) != row_count) {
    stop("candidate_p_value must be numeric with one value per logical row",
         call. = FALSE)
  }
  candidate_p_value <- as.numeric(candidate_p_value)
  if (any(!is.finite(candidate_p_value))) {
    stop("candidate_p_value must contain only finite values", call. = FALSE)
  }

  logical_sequence_id <- fastkpc_full_cuda_shadow_integer(
    logical_tests$logical_sequence_id, "logical_sequence_id"
  )
  if (anyDuplicated(logical_sequence_id)) {
    stop("logical_tests contains duplicate logical_sequence_id",
         call. = FALSE)
  }
  if (!identical(logical_sequence_id, seq_len(row_count))) {
    stop("logical_tests must be in complete canonical logical_sequence_id order",
         call. = FALSE)
  }
  fastkpc_full_cuda_shadow_validate_logical_contract(
    logical_tests, expected_logical_contract
  )
  source_sequence_id <- fastkpc_full_cuda_shadow_integer(
    logical_tests$source_sequence_id, "source_sequence_id"
  )
  source_task_index <- fastkpc_full_cuda_shadow_integer(
    logical_tests$source_task_index, "source_task_index"
  )
  level <- fastkpc_full_cuda_shadow_integer(
    logical_tests$level, "level", minimum = 0L
  )
  x <- fastkpc_full_cuda_shadow_integer(logical_tests$x, "x")
  y <- fastkpc_full_cuda_shadow_integer(logical_tests$y, "y")
  node_count <- length(labels)
  if (any(x > node_count) || any(y > node_count) || any(x == y)) {
    stop("logical_tests contains invalid node indexes", call. = FALSE)
  }
  source_keys <- paste(level, source_task_index, sep = "|")
  if (anyDuplicated(source_keys)) {
    stop("logical_tests contains duplicate source task indexes", call. = FALSE)
  }
  S_key <- as.character(logical_tests$S_key)
  if (length(S_key) != row_count || anyNA(S_key)) {
    stop("logical_tests contains invalid S_key", call. = FALSE)
  }
  alpha <- suppressWarnings(as.numeric(logical_tests$alpha))
  if (length(alpha) != row_count || any(!is.finite(alpha))) {
    stop("logical_tests contains invalid alpha", call. = FALSE)
  }
  reference_decision <- as.character(logical_tests$reference_decision)
  if (length(reference_decision) != row_count || anyNA(reference_decision) ||
      any(!reference_decision %in% c("dependent", "independent"))) {
    stop("logical_tests contains invalid reference_decision", call. = FALSE)
  }

  candidate_independent <- candidate_p_value > alpha
  candidate_decision <- ifelse(
    candidate_independent, "independent", "dependent"
  )
  decision_flip <- candidate_decision != reference_decision
  deletes_edge <- rep(FALSE, row_count)

  adjacency <- matrix(
    TRUE, node_count, node_count, dimnames = list(labels, labels)
  )
  diag(adjacency) <- FALSE
  pMax <- matrix(0, node_count, node_count, dimnames = list(labels, labels))
  diag(pMax) <- 1
  sepsets <- lapply(seq_len(node_count), function(index) {
    row <- lapply(seq_len(node_count), function(column) integer())
    names(row) <- labels
    row
  })
  names(sepsets) <- labels

  for (row_index in seq_len(row_count)) {
    tested_x <- x[[row_index]]
    tested_y <- y[[row_index]]
    if (!isTRUE(adjacency[tested_x, tested_y])) next
    if (candidate_p_value[[row_index]] > pMax[tested_x, tested_y]) {
      pMax[tested_x, tested_y] <- candidate_p_value[[row_index]]
      pMax[tested_y, tested_x] <- candidate_p_value[[row_index]]
    }
    if (!candidate_independent[[row_index]]) next
    S <- fastkpc_full_cuda_parse_s_key(S_key[[row_index]])
    canonical_S_key <- fastkpc_full_cuda_s_key(S)
    if (!identical(canonical_S_key, S_key[[row_index]]) ||
        length(S) != level[[row_index]] || any(S < 1L) ||
        any(S > node_count) || any(S %in% c(tested_x, tested_y))) {
      stop("logical_tests contains invalid canonical S_key at row ",
           row_index, call. = FALSE)
    }
    adjacency[tested_x, tested_y] <- FALSE
    adjacency[tested_y, tested_x] <- FALSE
    deletes_edge[[row_index]] <- TRUE
    sepsets[[tested_x]][[tested_y]] <- S
  }

  logical_trace <- data.frame(
    logical_sequence_id = logical_sequence_id,
    source_sequence_id = source_sequence_id,
    source_task_index = source_task_index,
    level = level,
    x = x,
    y = y,
    S_key = S_key,
    p_value = candidate_p_value,
    candidate_p_value = candidate_p_value,
    alpha = alpha,
    reference_decision = reference_decision,
    candidate_independent = candidate_independent,
    candidate_decision = candidate_decision,
    decision_flip = decision_flip,
    deletes_edge = deletes_edge,
    stringsAsFactors = FALSE
  )
  tasks <- data.frame(
    logical_sequence_id = logical_sequence_id,
    canonical_test_order_id = source_sequence_id,
    source_sequence_id = source_sequence_id,
    task_index = source_task_index,
    source_task_index = source_task_index,
    level = level,
    edge_x = pmin(x, y),
    edge_y = pmax(x, y),
    x = x,
    y = y,
    S_key = S_key,
    p_candidate = candidate_p_value,
    candidate_independent = candidate_independent,
    candidate_decision = candidate_decision,
    decision_flip = decision_flip,
    native_edge_deleted = deletes_edge,
    native_edge_ignored = FALSE,
    stringsAsFactors = FALSE
  )
  n.edgetests <- as.integer(tabulate(
    level + 1L, nbins = max(level) + 1L
  ))
  skeleton <- list(
    adjacency = adjacency,
    sepsets = sepsets,
    pMax = pMax,
    n.edgetests = n.edgetests,
    tasks = tasks,
    summary = list(
      unknown_fallback_count = 0L,
      approximate_backend_count = 0L
    )
  )
  list(skeleton = skeleton, logical_trace = logical_trace)
}

.fastkpc_full_cuda_shadow_direct_ci_rows <- function(catalog, plan = NULL) {
  execution_plan <- if (is.null(plan)) {
    fastkpc_full_cuda_shadow_plan(catalog)
  } else {
    plan
  }
  .fastkpc_full_cuda_shadow_validate_supplied_plan(catalog, execution_plan)
  direct <- execution_plan$direct_tests
  data <- catalog$inputs$data
  phase0_manifest <- catalog$phase0$manifest
  clean_data <- is.matrix(data) && typeof(data) == "double" &&
    identical(dim(data), c(351L, 48L)) && all(is.finite(data)) &&
    identical(as.integer(phase0_manifest$index), 1L) &&
    identical(as.integer(phase0_manifest$numCol), 35L)
  if (!isTRUE(clean_data)) {
    stop("direct-CI canonical data or route is malformed", call. = FALSE)
  }
  if (!exists("fastkpc_full_cuda_data_hash", mode = "function",
              inherits = TRUE)) {
    stop("direct-CI canonical data hash helper is unavailable",
         call. = FALSE)
  }
  catalog_authority <-
    fastkpc_full_cuda_phase3_discover_catalog_authority(catalog)
  actual_data_hash <- fastkpc_full_cuda_data_hash(data)
  if (!identical(
        actual_data_hash,
        catalog_authority$lineage$dataset_matrix_sha256
      )) {
    stop("direct-CI canonical data matrix hash mismatch", call. = FALSE)
  }
  x_matrix <- data[, as.integer(direct$x), drop = FALSE]
  y_matrix <- data[, as.integer(direct$y), drop = FALSE]
  oracle <- tryCatch(
    fastkpc_full_cuda_fixed_sp_with_legacy_dcov_backend(function() {
      fastkpc_legacy_dcov_gamma_cpp_oracle_batch(
        x = x_matrix,
        y = y_matrix,
        numCol = 35L,
        index = 1
      )
    }),
    error = function(error) {
      stop("direct-CI pinned legacy C++ Spectra backend error: ",
           conditionMessage(error), call. = FALSE)
    }
  )
  p_value <- as.double(oracle$p.value)
  diagnostics <- oracle$diagnostics
  diagnostic_fields <- c(
    "n", "batch_count", "numCol", "index", "lowrank_mode",
    "lowrank_full_eig_count", "lowrank_spectra_count",
    "lowrank_spectra_converged_count", "lowrank_spectra_failed_count",
    "lowrank_spectra_fallback_full_eig_count"
  )
  clean_diagnostics <- is.list(diagnostics) &&
    length(setdiff(diagnostic_fields, names(diagnostics))) == 0L &&
    identical(diagnostics$n, 351L) &&
    identical(diagnostics$batch_count, 2213L) &&
    identical(diagnostics$numCol, 35L) &&
    identical(as.double(diagnostics$index), 1) &&
    identical(diagnostics$lowrank_mode, "spectra") &&
    identical(diagnostics$lowrank_full_eig_count, 0L) &&
    identical(diagnostics$lowrank_spectra_count, 4426L) &&
    identical(diagnostics$lowrank_spectra_converged_count, 4426L) &&
    identical(diagnostics$lowrank_spectra_failed_count, 0L) &&
    identical(
      diagnostics$lowrank_spectra_fallback_full_eig_count, 0L
    ) && length(p_value) == 2213L && all(is.finite(p_value))
  if (!isTRUE(clean_diagnostics)) {
    stop("direct-CI pinned legacy C++ Spectra diagnostics failed",
         call. = FALSE)
  }

  reference_p_value <- as.double(direct$reference_p_value)
  alpha <- as.double(direct$alpha)
  candidate_decision <- ifelse(
    p_value > alpha, "independent", "dependent"
  )
  rows <- data.frame(
    logical_sequence_id = as.integer(direct$logical_sequence_id),
    source_sequence_id = as.integer(direct$source_sequence_id),
    source_task_index = as.integer(direct$source_task_index),
    level = as.integer(direct$level),
    x = as.integer(direct$x),
    y = as.integer(direct$y),
    S_key = as.character(direct$S_key),
    residual_key_x = as.character(direct$residual_key_x),
    residual_key_y = as.character(direct$residual_key_y),
    reference_p_value = reference_p_value,
    candidate_p_value = p_value,
    absolute_p_value_difference = abs(p_value - reference_p_value),
    alpha = alpha,
    reference_decision = as.character(direct$reference_decision),
    candidate_decision = candidate_decision,
    decision_flip = candidate_decision !=
      as.character(direct$reference_decision),
    backend = rep.int("legacy-cpp", nrow(direct)),
    low_rank_backend = rep.int("spectra", nrow(direct)),
    backend_error = rep.int(FALSE, nrow(direct)),
    spectra_fallback = rep.int(FALSE, nrow(direct)),
    stringsAsFactors = FALSE
  )
  rownames(rows) <- NULL
  .fastkpc_full_cuda_phase3_validate_direct_ci_rows(rows)
  rows
}

fastkpc_full_cuda_shadow_write_direct_ci <- function(
    catalog, output_dir, plan = NULL) {
  rows <- .fastkpc_full_cuda_shadow_direct_ci_rows(catalog, plan)
  validated <- fastkpc_full_cuda_phase3_publish_direct_ci_payload(
    rows = rows, catalog = catalog, output_dir = output_dir
  )
  validated$payload
}
