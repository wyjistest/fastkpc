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

.fastkpc_full_cuda_shadow_plan_registry_option_name <- function() {
  "fastkpc.full_cuda.shadow_plan_registry.v1"
}

.fastkpc_full_cuda_shadow_plan_registry_marker_option_name <- function() {
  "fastkpc.full_cuda.shadow_plan_registry.initialized.v1"
}

.fastkpc_full_cuda_shadow_plan_registry_schema_version <- function() {
  "full-cuda-ci-shadow-plan-registry-v1"
}

.fastkpc_full_cuda_shadow_valid_plan_registry <- function(registry) {
  is.environment(registry) && identical(
    attributes(registry),
    list(
      schema_version =
        .fastkpc_full_cuda_shadow_plan_registry_schema_version()
    )
  )
}

.fastkpc_full_cuda_shadow_initialize_plan_registry <- function() {
  option_name <- .fastkpc_full_cuda_shadow_plan_registry_option_name()
  marker_name <-
    .fastkpc_full_cuda_shadow_plan_registry_marker_option_name()
  schema_version <-
    .fastkpc_full_cuda_shadow_plan_registry_schema_version()
  registry <- getOption(option_name, NULL)
  marker <- getOption(marker_name, NULL)
  definition_environment <- environment(
    .fastkpc_full_cuda_shadow_initialize_plan_registry
  )
  previous <- get0(
    ".fastkpc_full_cuda_shadow_plan_registry_singleton",
    envir = definition_environment, inherits = FALSE, ifnotfound = NULL
  )

  if (is.null(marker)) {
    if (!is.null(registry) || !is.null(previous)) {
      stop("shadow plan registry singleton initialization is corrupt",
           call. = FALSE)
    }
    registry <- new.env(hash = TRUE, parent = emptyenv())
    attr(registry, "schema_version") <- schema_version
    options(structure(
      list(registry, schema_version), names = c(option_name, marker_name)
    ))
    return(registry)
  }
  if (!identical(marker, schema_version) ||
      !.fastkpc_full_cuda_shadow_valid_plan_registry(registry) ||
      (!is.null(previous) && !identical(previous, registry))) {
    stop("shadow plan registry singleton is corrupt", call. = FALSE)
  }
  registry
}

.fastkpc_full_cuda_shadow_plan_registry_singleton <-
  .fastkpc_full_cuda_shadow_initialize_plan_registry()

.fastkpc_full_cuda_shadow_plan_registry <-
  .fastkpc_full_cuda_shadow_plan_registry_singleton

.fastkpc_full_cuda_shadow_current_plan_registry <- function() {
  option_name <- .fastkpc_full_cuda_shadow_plan_registry_option_name()
  marker_name <-
    .fastkpc_full_cuda_shadow_plan_registry_marker_option_name()
  registry <- getOption(option_name, NULL)
  definition_environment <- environment(
    .fastkpc_full_cuda_shadow_current_plan_registry
  )
  bound_registry <- get0(
    ".fastkpc_full_cuda_shadow_plan_registry",
    envir = definition_environment, inherits = FALSE, ifnotfound = NULL
  )
  if (!identical(
        getOption(marker_name, NULL),
        .fastkpc_full_cuda_shadow_plan_registry_schema_version()
      ) || !.fastkpc_full_cuda_shadow_valid_plan_registry(registry) ||
      !identical(
        registry, .fastkpc_full_cuda_shadow_plan_registry_singleton
      ) || !identical(registry, bound_registry)) {
    stop("shadow plan registry singleton is corrupt", call. = FALSE)
  }
  registry
}

.fastkpc_full_cuda_shadow_plan_metadata_fields <- function() {
  c(
    "schema_version", "logical_contract", "catalog_authority_sha256",
    "catalog_lineage_sha256", "route_config_sha256",
    "phase2_setup_index_csv_sha256", "setup_association_sha256",
    "phase2_target_state_index_rds_sha256", "target_association_sha256",
    "shard_count", "direct_tests_schema_sha256",
    "conditional_tests_schema_sha256", "direct_tests_sha256",
    "conditional_tests_sha256"
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

.fastkpc_full_cuda_shadow_bare_sha256_vector <- function(
    value, expected_count) {
  typeof(value) == "character" && length(value) == expected_count &&
    !is.object(value) && is.null(attributes(value)) && !anyNA(value) &&
    all(grepl("^[0-9a-f]{64}$", value))
}

.fastkpc_full_cuda_shadow_bare_pid <- function(value) {
  typeof(value) == "integer" && length(value) == 1L &&
    !is.object(value) && is.null(attributes(value)) && !is.na(value) &&
    value > 0L
}

.fastkpc_full_cuda_shadow_plan_frame_schema_hashes <- function(
    catalog, direct_tests, conditional_tests) {
  logical_tests <- catalog$inputs$logical_tests
  logical_attributes <- attributes(logical_tests)
  logical_clean <- is.data.frame(logical_tests) &&
    identical(class(logical_tests), "data.frame") &&
    nrow(logical_tests) == 240489L && !anyDuplicated(names(logical_tests)) &&
    !anyDuplicated(names(logical_attributes)) &&
    all(vapply(logical_tests, function(column) {
      !is.object(column) && is.null(attributes(column)) &&
        length(column) == nrow(logical_tests)
    }, logical(1L)))
  if (!isTRUE(logical_clean)) {
    stop("shadow plan logical frame schema is malformed", call. = FALSE)
  }
  fields <- c(
    names(logical_tests), "prepared_s_key_x", "prepared_s_key_y", "shard_id"
  )
  types <- c(
    unname(vapply(logical_tests, typeof, character(1L))),
    "character", "character", "integer"
  )
  inherited_attribute_names <-
    sort(setdiff(names(logical_attributes), c("names", "row.names")))
  inherited_attribute_types <- unname(vapply(
    inherited_attribute_names,
    function(name) typeof(logical_attributes[[name]]),
    character(1L)
  ))
  inherited_attribute_objects <- unname(vapply(
    inherited_attribute_names,
    function(name) is.object(logical_attributes[[name]]),
    logical(1L)
  ))
  schema_hash <- function(frame, row_count, label) {
    frame_attributes <- attributes(frame)
    clean <- is.data.frame(frame) && identical(class(frame), "data.frame") &&
      nrow(frame) == row_count && identical(names(frame), fields) &&
      identical(
        sort(names(frame_attributes)), sort(names(logical_attributes))
      ) &&
      !anyDuplicated(names(frame_attributes)) &&
      identical(.row_names_info(frame, 0L), c(NA_integer_, -row_count)) &&
      all(vapply(inherited_attribute_names, function(name) {
        identical(frame_attributes[[name]], logical_attributes[[name]])
      }, logical(1L))) &&
      all(vapply(seq_along(fields), function(index) {
        column <- frame[[index]]
        identical(typeof(column), types[[index]]) &&
          length(column) == row_count && !is.object(column) &&
          is.null(attributes(column))
      }, logical(1L)))
    if (!isTRUE(clean)) {
      stop("shadow plan ", label, " exact frame schema mismatch",
           call. = FALSE)
    }
    fastkpc_full_cuda_census_named_metadata_hash(list(
      schema_version = "full-cuda-ci-shadow-frame-schema-v1",
      row_count = as.integer(row_count),
      field_order = fields,
      column_typeof = types,
      column_object_policy = rep.int(FALSE, length(fields)),
      column_attributes_policy = rep.int("none", length(fields)),
      frame_class = "data.frame",
      frame_object_policy = TRUE,
      frame_attribute_names = sort(names(logical_attributes)),
      inherited_attribute_names = inherited_attribute_names,
      inherited_attribute_typeof = inherited_attribute_types,
      inherited_attribute_object_policy = inherited_attribute_objects,
      inherited_attribute_value_policy = "identical-to-logical-contract",
      row_names_policy = "automatic"
    ))
  }
  list(
    direct_tests_schema_sha256 =
      schema_hash(direct_tests, 2213L, "direct"),
    conditional_tests_schema_sha256 =
      schema_hash(conditional_tests, 238276L, "conditional")
  )
}

.fastkpc_full_cuda_shadow_catalog_association_hashes <- function(catalog) {
  if (!is.list(catalog) || !is.list(catalog$inputs)) {
    stop("shadow plan catalog association inputs are malformed",
         call. = FALSE)
  }
  setup_index <- catalog$setup_index
  setup_metadata <- catalog$inputs$same_s_setup_metadata
  target_metadata <- catalog$inputs$target_fit_metadata
  setup_fields <- c(
    "same_S_group_id", "prepared_s_key_sha256",
    "phase1_setup_fingerprint"
  )
  setup_metadata_fields <- c("same_S_group_id", "setup_fingerprint")
  target_metadata_fields <- c(
    "residual_key_sha256", "target", "same_S_group_id",
    "setup_fingerprint"
  )
  contract <- fastkpc_full_cuda_fixed_sp_catalog_contract()
  setup_count <- as.integer(contract$selected_group_count)
  target_count <- as.integer(contract$target_state_count)
  schema_clean <- is.data.frame(setup_index) &&
    is.data.frame(setup_metadata) && is.data.frame(target_metadata) &&
    nrow(setup_index) == setup_count &&
    nrow(setup_metadata) == setup_count &&
    nrow(target_metadata) == target_count &&
    length(setdiff(c("schema_version", setup_fields),
                   names(setup_index))) == 0L &&
    length(setdiff(setup_metadata_fields, names(setup_metadata))) == 0L &&
    length(setdiff(target_metadata_fields, names(target_metadata))) == 0L &&
    all(setup_index$schema_version ==
          "full-cuda-ci-prepared-s-setup-v1")
  if (!isTRUE(schema_clean)) {
    stop("shadow plan catalog association schemas are malformed",
         call. = FALSE)
  }

  setup_records <- setup_index[, setup_fields, drop = FALSE]
  setup_sha256_clean <- all(vapply(setup_records, function(value) {
    .fastkpc_full_cuda_shadow_bare_sha256_vector(value, setup_count)
  }, logical(1L))) &&
    .fastkpc_full_cuda_shadow_bare_sha256_vector(
      setup_metadata$same_S_group_id, setup_count
    ) && .fastkpc_full_cuda_shadow_bare_sha256_vector(
      setup_metadata$setup_fingerprint, setup_count
    )
  target_sha256_clean <- all(vapply(
    target_metadata[c(
      "residual_key_sha256", "same_S_group_id", "setup_fingerprint"
    )],
    function(value) {
      .fastkpc_full_cuda_shadow_bare_sha256_vector(value, target_count)
    },
    logical(1L)
  ))
  target_value_clean <- typeof(target_metadata$target) == "integer" &&
    length(target_metadata$target) == target_count &&
    !is.object(target_metadata$target) &&
    is.null(attributes(target_metadata$target)) &&
    !anyNA(target_metadata$target) &&
    all(target_metadata$target >= 1L & target_metadata$target <= 48L)
  unique_clean <- !anyDuplicated(setup_records$same_S_group_id) &&
    !anyDuplicated(setup_records$prepared_s_key_sha256) &&
    !anyDuplicated(setup_metadata$same_S_group_id) &&
    !anyDuplicated(target_metadata$residual_key_sha256)
  if (!isTRUE(setup_sha256_clean) || !isTRUE(target_sha256_clean) ||
      !isTRUE(target_value_clean) || !isTRUE(unique_clean)) {
    stop("shadow plan catalog association values are malformed",
         call. = FALSE)
  }

  setup_metadata_match <- match(
    setup_records$same_S_group_id, setup_metadata$same_S_group_id
  )
  target_setup_match <- match(
    target_metadata$same_S_group_id, setup_records$same_S_group_id
  )
  association_clean <- !anyNA(setup_metadata_match) &&
    !anyNA(target_setup_match) &&
    setequal(
      setup_records$same_S_group_id, setup_metadata$same_S_group_id
    ) && setequal(
      setup_records$same_S_group_id,
      unique(target_metadata$same_S_group_id)
    ) && identical(
      setup_records$phase1_setup_fingerprint,
      setup_metadata$setup_fingerprint[setup_metadata_match]
    ) && identical(
      target_metadata$setup_fingerprint,
      setup_records$phase1_setup_fingerprint[target_setup_match]
    )
  if (!isTRUE(association_clean)) {
    stop("shadow plan catalog setup/target associations are inconsistent",
         call. = FALSE)
  }

  target_records <- data.frame(
    residual_key_sha256 = target_metadata$residual_key_sha256,
    target = target_metadata$target,
    same_S_group_id = target_metadata$same_S_group_id,
    prepared_s_key_sha256 =
      setup_records$prepared_s_key_sha256[target_setup_match],
    phase1_setup_fingerprint = target_metadata$setup_fingerprint,
    stringsAsFactors = FALSE
  )
  setup_records <- setup_records[order(
    setup_records$prepared_s_key_sha256, method = "radix"
  ), , drop = FALSE]
  target_records <- target_records[order(
    target_records$residual_key_sha256, method = "radix"
  ), , drop = FALSE]
  rownames(setup_records) <- rownames(target_records) <- NULL
  if (!identical(names(setup_records), setup_fields) ||
      !identical(names(target_records), c(
        "residual_key_sha256", "target", "same_S_group_id",
        "prepared_s_key_sha256", "phase1_setup_fingerprint"
      )) || anyDuplicated(setup_records$prepared_s_key_sha256) ||
      anyDuplicated(target_records$residual_key_sha256)) {
    stop("shadow plan canonical association projections are malformed",
         call. = FALSE)
  }
  list(
    setup_association_sha256 =
      fastkpc_full_cuda_census_frame_hash(setup_records),
    target_association_sha256 =
      fastkpc_full_cuda_census_frame_hash(target_records)
  )
}

.fastkpc_full_cuda_shadow_plan_metadata_hash <- function(plan) {
  fields <- .fastkpc_full_cuda_shadow_plan_metadata_fields()
  if (!is.list(plan) || length(setdiff(fields, names(plan))) > 0L) {
    stop("shadow plan metadata is malformed", call. = FALSE)
  }
  fastkpc_full_cuda_census_named_metadata_hash(plan[fields])
}

.fastkpc_full_cuda_shadow_phase2_file_records <- function(
    catalog, setup_index_sha256, target_state_sha256,
    .transition_hook = NULL) {
  if (!.fastkpc_full_cuda_shadow_bare_sha256(setup_index_sha256) ||
      !.fastkpc_full_cuda_shadow_bare_sha256(target_state_sha256) ||
      !is.list(catalog) || is.null(catalog$phase2_dir) ||
      (!is.null(.transition_hook) && !is.function(.transition_hook))) {
    stop("shadow plan Phase 2 file identity is malformed", call. = FALSE)
  }
  logical_path <- c(
    "prepared_s_setup_index.csv", "target_state_index.rds"
  )
  phase2_dir <- normalizePath(
    catalog$phase2_dir, winslash = "/", mustWork = TRUE
  )
  paths <- file.path(phase2_dir, logical_path)
  require_regular_paths <- function() {
    links <- Sys.readlink(paths)
    clean <- length(paths) == length(logical_path) &&
      length(links) == length(paths) && !anyNA(links) &&
      !any(nzchar(links)) && all(file.exists(paths)) &&
      !any(dir.exists(paths)) && all(file_test("-f", paths))
    if (!isTRUE(clean)) {
      stop("shadow plan Phase 2 paths must be regular non-symlink files",
           call. = FALSE)
    }
    invisible(TRUE)
  }
  require_regular_paths()
  canonical_paths <- unname(vapply(
    paths, normalizePath, character(1L), winslash = "/", mustWork = TRUE
  ))
  if (!exists("fastkpc_full_cuda_fixed_sp_sha256_file", mode = "function",
              inherits = TRUE)) {
    stop("shadow plan Phase 2 file hash helper is unavailable",
         call. = FALSE)
  }
  snapshot_metadata <- function() {
    require_regular_paths()
    info <- file.info(paths, extra_cols = FALSE)
    current_paths <- unname(vapply(
      paths, normalizePath, character(1L), winslash = "/",
      mustWork = TRUE
    ))
    require_regular_paths()
    clean <- is.data.frame(info) && nrow(info) == length(paths) &&
      !anyNA(info$size) && !anyNA(info$mode) && !anyNA(info$mtime) &&
      !anyNA(info$ctime) && all(info$size >= 0) &&
      identical(current_paths, canonical_paths)
    if (!isTRUE(clean)) {
      stop("shadow plan Phase 2 file identity is malformed", call. = FALSE)
    }
    data.frame(
      path = current_paths,
      size = as.numeric(info$size),
      mode = as.integer(info$mode),
      mtime = as.numeric(info$mtime),
      ctime = as.numeric(info$ctime),
      stringsAsFactors = FALSE
    )
  }
  hash_files <- function() {
    require_regular_paths()
    hashes <- unname(vapply(
      paths, fastkpc_full_cuda_fixed_sp_sha256_file, character(1L)
    ))
    require_regular_paths()
    if (!all(vapply(
          hashes, .fastkpc_full_cuda_shadow_bare_sha256, logical(1L)
        ))) {
      stop("shadow plan Phase 2 current file hash mismatch", call. = FALSE)
    }
    hashes
  }

  pre_metadata <- snapshot_metadata()
  first_sha256 <- hash_files()
  post_metadata <- snapshot_metadata()
  if (is.function(.transition_hook)) {
    .transition_hook(paths = paths, stage = "between_hash_passes")
  }
  second_sha256 <- hash_files()
  final_metadata <- snapshot_metadata()
  if (!identical(pre_metadata, post_metadata) ||
      !identical(post_metadata, final_metadata) ||
      !identical(first_sha256, second_sha256)) {
    stop("shadow plan Phase 2 files changed while snapshotting",
         call. = FALSE)
  }
  expected_sha256 <- c(setup_index_sha256, target_state_sha256)
  if (!identical(second_sha256, expected_sha256)) {
    stop("shadow plan Phase 2 current file hash mismatch", call. = FALSE)
  }
  records <- data.frame(
    logical_path = logical_path,
    path = final_metadata$path,
    size = as.character(final_metadata$size),
    mode = as.character(final_metadata$mode),
    mtime = sprintf("%.6f", final_metadata$mtime),
    ctime = sprintf("%.6f", final_metadata$ctime),
    sha256 = second_sha256,
    stringsAsFactors = FALSE
  )
  rownames(records) <- NULL
  records
}

.fastkpc_full_cuda_shadow_register_plan_identity <- function(
    metadata, plan_identity_sha256, phase2_file_records) {
  token <- new.env(parent = emptyenv())
  creator_pid <- as.integer(Sys.getpid())
  if (!.fastkpc_full_cuda_shadow_bare_pid(creator_pid)) {
    stop("shadow plan creator process identity is malformed", call. = FALSE)
  }
  registry_id <- fastkpc_full_cuda_census_named_metadata_hash(list(
    schema_version = "full-cuda-ci-shadow-plan-registry-key-v1",
    process_id = creator_pid,
    token_environment = format(token),
    catalog_authority_sha256 = metadata$catalog_authority_sha256,
    plan_identity_sha256 = plan_identity_sha256
  ))
  registry <- .fastkpc_full_cuda_shadow_current_plan_registry()
  if (exists(registry_id, envir = registry, inherits = FALSE)) {
    stop("shadow plan registry token collision", call. = FALSE)
  }
  assign(
    "schema_version", "full-cuda-ci-shadow-plan-token-v2", envir = token
  )
  assign("registry_id", registry_id, envir = token)
  assign("creator_pid", creator_pid, envir = token)
  assign(
    "catalog_authority_sha256", metadata$catalog_authority_sha256,
    envir = token
  )
  assign("plan_identity_sha256", plan_identity_sha256, envir = token)
  lockEnvironment(token, bindings = TRUE)

  entry <- list(
    schema_version = "full-cuda-ci-shadow-plan-registry-entry-v4",
    registered_at = as.double(Sys.time()),
    creator_pid = creator_pid,
    token = token,
    catalog_authority_sha256 = metadata$catalog_authority_sha256,
    catalog_lineage_sha256 = metadata$catalog_lineage_sha256,
    route_config_sha256 = metadata$route_config_sha256,
    setup_association_sha256 = metadata$setup_association_sha256,
    target_association_sha256 = metadata$target_association_sha256,
    plan_identity_sha256 = plan_identity_sha256,
    direct_tests_schema_sha256 = metadata$direct_tests_schema_sha256,
    conditional_tests_schema_sha256 =
      metadata$conditional_tests_schema_sha256,
    direct_tests_sha256 = metadata$direct_tests_sha256,
    conditional_tests_sha256 = metadata$conditional_tests_sha256,
    phase2_file_records = phase2_file_records
  )
  assign(registry_id, entry, envir = registry)
  lockBinding(registry_id, registry)
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
  frame_schema_hashes <- .fastkpc_full_cuda_shadow_plan_frame_schema_hashes(
    catalog, mapped_plan$direct_tests, mapped_plan$conditional_tests
  )
  metadata <- list(
    schema_version = "full-cuda-ci-shadow-plan-v3",
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
    direct_tests_schema_sha256 =
      frame_schema_hashes$direct_tests_schema_sha256,
    conditional_tests_schema_sha256 =
      frame_schema_hashes$conditional_tests_schema_sha256,
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
    "target_association_sha256", "direct_tests_schema_sha256",
    "conditional_tests_schema_sha256", "direct_tests_sha256",
    "conditional_tests_sha256", "plan_identity_sha256"
  )
  clean_plan <- is.list(plan) && !is.object(plan) &&
    identical(names(plan), expected_fields) &&
    identical(plan$schema_version, "full-cuda-ci-shadow-plan-v3") &&
    all(vapply(
      plan[sha256_fields], .fastkpc_full_cuda_shadow_bare_sha256, logical(1L)
    )) && is.integer(plan$shard_count) && length(plan$shard_count) == 1L &&
    !is.object(plan$shard_count) && is.null(attributes(plan$shard_count)) &&
    identical(plan$shard_count, 64L) && is.data.frame(plan$direct_tests) &&
    is.data.frame(plan$conditional_tests)
  if (!isTRUE(clean_plan)) stop("malformed supplied shadow plan")

  token <- plan$authentication_token
  token_fields <- c(
    "catalog_authority_sha256", "creator_pid", "plan_identity_sha256",
    "registry_id", "schema_version"
  )
  current_pid <- as.integer(Sys.getpid())
  clean_token <- is.environment(token) && environmentIsLocked(token) &&
    identical(ls(token, all.names = TRUE), token_fields) &&
    all(vapply(token_fields, bindingIsLocked, logical(1L), env = token)) &&
    identical(token$schema_version, "full-cuda-ci-shadow-plan-token-v2") &&
    .fastkpc_full_cuda_shadow_bare_sha256(token$registry_id) &&
    .fastkpc_full_cuda_shadow_bare_pid(token$creator_pid) &&
    .fastkpc_full_cuda_shadow_bare_pid(current_pid) &&
    identical(token$creator_pid, current_pid) &&
    identical(token$catalog_authority_sha256,
              plan$catalog_authority_sha256) &&
    identical(token$plan_identity_sha256, plan$plan_identity_sha256)
  if (!isTRUE(clean_token)) stop("invalid supplied shadow plan token")

  registry <- .fastkpc_full_cuda_shadow_current_plan_registry()
  if (!exists(token$registry_id, envir = registry, inherits = FALSE) ||
      !bindingIsLocked(token$registry_id, registry)) {
    stop("missing supplied shadow plan registry entry")
  }
  entry <- get(token$registry_id, envir = registry, inherits = FALSE)
  entry_fields <- c(
    "schema_version", "registered_at", "creator_pid", "token",
    "catalog_authority_sha256", "catalog_lineage_sha256",
    "route_config_sha256", "setup_association_sha256",
    "target_association_sha256", "plan_identity_sha256",
    "direct_tests_schema_sha256", "conditional_tests_schema_sha256",
    "direct_tests_sha256", "conditional_tests_sha256",
    "phase2_file_records"
  )
  clean_entry <- is.list(entry) && !is.object(entry) &&
    identical(names(entry), entry_fields) &&
    identical(
      entry$schema_version,
      "full-cuda-ci-shadow-plan-registry-entry-v4"
    ) && is.double(entry$registered_at) &&
    length(entry$registered_at) == 1L && is.finite(entry$registered_at) &&
    .fastkpc_full_cuda_shadow_bare_pid(entry$creator_pid) &&
    identical(entry$creator_pid, current_pid) &&
    identical(entry$creator_pid, token$creator_pid) &&
    identical(entry$token, token) &&
    identical(entry$catalog_authority_sha256,
              plan$catalog_authority_sha256) &&
    identical(entry$catalog_lineage_sha256,
              plan$catalog_lineage_sha256) &&
    identical(entry$route_config_sha256, plan$route_config_sha256) &&
    identical(entry$setup_association_sha256,
              plan$setup_association_sha256) &&
    identical(entry$target_association_sha256,
              plan$target_association_sha256) &&
    identical(entry$plan_identity_sha256, plan$plan_identity_sha256) &&
    identical(entry$direct_tests_schema_sha256,
              plan$direct_tests_schema_sha256) &&
    identical(entry$conditional_tests_schema_sha256,
              plan$conditional_tests_schema_sha256) &&
    identical(entry$direct_tests_sha256, plan$direct_tests_sha256) &&
    identical(entry$conditional_tests_sha256,
              plan$conditional_tests_sha256)
  if (!isTRUE(clean_entry)) stop("stale supplied shadow plan registry entry")

  metadata_hash <- .fastkpc_full_cuda_shadow_plan_metadata_hash(plan)
  if (!identical(metadata_hash, plan$plan_identity_sha256) ||
      !identical(metadata_hash, entry$plan_identity_sha256)) {
    stop("supplied shadow plan metadata identity mismatch")
  }

  current_frame_schemas <-
    .fastkpc_full_cuda_shadow_plan_frame_schema_hashes(
      catalog, plan$direct_tests, plan$conditional_tests
    )
  schema_binding <- identical(
    current_frame_schemas$direct_tests_schema_sha256,
    plan$direct_tests_schema_sha256
  ) && identical(
    current_frame_schemas$direct_tests_schema_sha256,
    entry$direct_tests_schema_sha256
  ) && identical(
    current_frame_schemas$conditional_tests_schema_sha256,
    plan$conditional_tests_schema_sha256
  ) && identical(
    current_frame_schemas$conditional_tests_schema_sha256,
    entry$conditional_tests_schema_sha256
  )
  if (!isTRUE(schema_binding)) {
    stop("supplied shadow plan exact frame schema identity mismatch")
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

  current_associations <-
    .fastkpc_full_cuda_shadow_catalog_association_hashes(catalog)
  association_binding <- identical(
    current_associations$setup_association_sha256,
    plan$setup_association_sha256
  ) && identical(
    current_associations$setup_association_sha256,
    entry$setup_association_sha256
  ) && identical(
    current_associations$target_association_sha256,
    plan$target_association_sha256
  ) && identical(
    current_associations$target_association_sha256,
    entry$target_association_sha256
  )
  if (!isTRUE(association_binding)) {
    stop("supplied shadow plan catalog association identity mismatch")
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

fastkpc_full_cuda_shadow_conditional_row_schema <- function() {
  c(
    logical_sequence_id = "integer",
    source_sequence_id = "integer",
    source_task_index = "integer",
    level = "integer",
    x = "integer",
    y = "integer",
    S_key = "character",
    residual_key_x = "character",
    residual_key_y = "character",
    prepared_s_key_sha256 = "character",
    shard_id = "integer",
    reference_p_value = "double",
    candidate_p_value = "double",
    absolute_p_value_difference = "double",
    alpha = "double",
    reference_decision = "character",
    candidate_decision = "character",
    decision_flip = "logical",
    near_alpha = "logical",
    near_alpha_bucket = "character",
    backend = "character",
    low_rank_backend = "character",
    backend_error = "logical",
    spectra_fallback = "logical",
    planned_route_x = "character",
    executed_route_x = "character",
    reroute_reason_x = "character",
    solver_status_x = "character",
    planned_route_y = "character",
    executed_route_y = "character",
    reroute_reason_y = "character",
    solver_status_y = "character"
  )
}

.fastkpc_full_cuda_shadow_conditional_base_schema <- function() {
  schema <- fastkpc_full_cuda_shadow_conditional_row_schema()
  schema[!names(schema) %in% c(
    "x", "y",
    "planned_route_x", "executed_route_x", "reroute_reason_x",
    "solver_status_x", "planned_route_y", "executed_route_y",
    "reroute_reason_y", "solver_status_y"
  )]
}

.fastkpc_full_cuda_shadow_empty_frame <- function(schema) {
  columns <- lapply(unname(schema), function(type) switch(
    type,
    character = character(),
    integer = integer(),
    double = double(),
    logical = logical(),
    stop("unsupported conditional shadow field type", call. = FALSE)
  ))
  names(columns) <- names(schema)
  as.data.frame(columns, stringsAsFactors = FALSE, optional = TRUE)
}

fastkpc_full_cuda_shadow_empty_conditional_rows <- function() {
  .fastkpc_full_cuda_shadow_empty_frame(
    fastkpc_full_cuda_shadow_conditional_row_schema()
  )
}

.fastkpc_full_cuda_shadow_validate_atomic_frame <- function(
    value, schema, label, allow_empty = FALSE) {
  clean <- is.data.frame(value) && identical(names(value), names(schema)) &&
    (nrow(value) > 0L || isTRUE(allow_empty)) &&
    all(vapply(names(schema), function(field) {
      column <- value[[field]]
      typeof(column) == schema[[field]] && length(column) == nrow(value) &&
        !is.object(column) && is.null(attributes(column)) && !anyNA(column)
    }, logical(1L)))
  if (!isTRUE(clean)) {
    stop(label, " schema or types are malformed", call. = FALSE)
  }
  invisible(TRUE)
}

.fastkpc_full_cuda_shadow_whole_scalar <- function(
    value, minimum, label) {
  clean <- typeof(value) %in% c("integer", "double") &&
    length(value) == 1L && !is.object(value) &&
    is.null(attributes(value)) && !is.na(value) && is.finite(value) &&
    value == floor(value) && value >= minimum &&
    value <= .Machine$integer.max
  if (!isTRUE(clean)) {
    stop(label, " must be one whole scalar >= ", minimum, call. = FALSE)
  }
  as.integer(value)
}

fastkpc_full_cuda_shadow_runtime_counters <- function(resource_metrics) {
  fields <- c(
    "implicit_residual_d2h_count",
    "shadow_materialize_call_count",
    "shadow_materialize_target_count"
  )
  clean <- is.data.frame(resource_metrics) &&
    all(fields %in% names(resource_metrics)) &&
    all(vapply(fields, function(field) {
      value <- resource_metrics[[field]]
      typeof(value) == "integer" && !is.object(value) &&
        is.null(attributes(value)) && !anyNA(value) && all(value >= 0L)
    }, logical(1L)))
  if (!isTRUE(clean)) {
    stop("conditional shadow runtime counter rows are malformed",
         call. = FALSE)
  }
  totals <- lapply(fields, function(field) {
    total <- sum(as.double(resource_metrics[[field]]))
    if (!is.finite(total) || total > .Machine$integer.max) {
      stop("conditional shadow runtime counter total is invalid: ", field,
           call. = FALSE)
    }
    as.integer(total)
  })
  names(totals) <- fields
  totals
}

.fastkpc_full_cuda_shadow_key_vector <- function(
    value, label, allow_empty = FALSE) {
  clean <- typeof(value) == "character" && !is.object(value) &&
    is.null(attributes(value)) &&
    (length(value) > 0L || isTRUE(allow_empty)) && !anyNA(value) &&
    all(grepl("^[0-9a-f]{64}$", value)) && !anyDuplicated(value)
  if (!isTRUE(clean)) {
    stop(label, " contains a missing, malformed, or duplicate key",
         call. = FALSE)
  }
  value
}

.fastkpc_full_cuda_shadow_conditional_logical_fields <- function() {
  c(
    "logical_sequence_id", "source_sequence_id", "source_task_index",
    "level", "x", "y", "S_key", "residual_key_x", "residual_key_y",
    "prepared_s_key_x", "prepared_s_key_y", "shard_id",
    "reference_p_value", "alpha", "reference_decision",
    "absolute_log_distance_from_alpha"
  )
}

.fastkpc_full_cuda_shadow_validate_setup_logical <- function(
    logical_tests, setup_key, shard_id) {
  required <- .fastkpc_full_cuda_shadow_conditional_logical_fields()
  if (!is.data.frame(logical_tests) || nrow(logical_tests) == 0L ||
      length(setdiff(required, names(logical_tests))) > 0L ||
      anyDuplicated(names(logical_tests))) {
    stop("conditional shadow setup logical rows are malformed",
         call. = FALSE)
  }
  setup_clean <- .fastkpc_full_cuda_shadow_bare_sha256(setup_key)
  shard_id <- .fastkpc_full_cuda_shadow_whole_scalar(
    shard_id, 0L, "conditional shadow shard_id"
  )
  if (!isTRUE(setup_clean) || shard_id >= 64L) {
    stop("conditional shadow setup identity is malformed", call. = FALSE)
  }

  integer_fields <- c(
    "logical_sequence_id", "source_sequence_id", "source_task_index",
    "level", "x", "y", "shard_id"
  )
  integer_values <- lapply(integer_fields, function(field) {
    fastkpc_full_cuda_shadow_integer(
      logical_tests[[field]], field,
      minimum = if (field == "shard_id") 0L else 1L
    )
  })
  names(integer_values) <- integer_fields
  logical_ids <- integer_values$logical_sequence_id
  if (anyDuplicated(logical_ids) || !identical(
        order(logical_ids, method = "radix"), seq_along(logical_ids)
      )) {
    stop("conditional shadow logical order is not canonical",
         call. = FALSE)
  }
  if (any(integer_values$level < 1L) ||
      any(integer_values$x > 48L) || any(integer_values$y > 48L) ||
      any(integer_values$x == integer_values$y)) {
    stop("conditional shadow logical endpoint values are malformed",
         call. = FALSE)
  }

  character_fields <- c(
    "S_key", "residual_key_x", "residual_key_y",
    "prepared_s_key_x", "prepared_s_key_y", "reference_decision"
  )
  character_clean <- all(vapply(character_fields, function(field) {
    value <- logical_tests[[field]]
    typeof(value) == "character" && length(value) == nrow(logical_tests) &&
      !is.object(value) && is.null(attributes(value)) && !anyNA(value)
  }, logical(1L)))
  key_clean <- isTRUE(character_clean) && all(vapply(
    c(
      "residual_key_x", "residual_key_y", "prepared_s_key_x",
      "prepared_s_key_y"
    ),
    function(field) all(grepl("^[0-9a-f]{64}$", logical_tests[[field]])),
    logical(1L)
  ))
  ownership_clean <- isTRUE(key_clean) &&
    all(logical_tests$prepared_s_key_x == setup_key) &&
    all(logical_tests$prepared_s_key_y == setup_key) &&
    all(integer_values$shard_id == shard_id)
  if (!isTRUE(ownership_clean)) {
    stop("conditional shadow logical setup ownership is invalid",
         call. = FALSE)
  }
  if (any(!nzchar(logical_tests$S_key)) ||
      any(logical_tests$residual_key_x == logical_tests$residual_key_y)) {
    stop("conditional shadow logical endpoint keys are malformed",
         call. = FALSE)
  }

  reference_p_value <- suppressWarnings(as.double(
    logical_tests$reference_p_value
  ))
  alpha <- suppressWarnings(as.double(logical_tests$alpha))
  log_distance <- suppressWarnings(as.double(
    logical_tests$absolute_log_distance_from_alpha
  ))
  numeric_clean <- length(reference_p_value) == nrow(logical_tests) &&
    length(alpha) == nrow(logical_tests) &&
    length(log_distance) == nrow(logical_tests) &&
    all(is.finite(reference_p_value)) &&
    all(reference_p_value >= 0 & reference_p_value <= 1) &&
    all(is.finite(alpha) & alpha > 0 & alpha < 1) &&
    all(!is.na(log_distance) & log_distance >= 0) &&
    all(logical_tests$reference_decision %in%
          c("dependent", "independent")) && identical(
      logical_tests$reference_decision,
      ifelse(reference_p_value > alpha, "independent", "dependent")
    )
  if (!isTRUE(numeric_clean)) {
    stop("conditional shadow logical decision inputs are malformed",
         call. = FALSE)
  }
  derived_near <- is.finite(log_distance) & log_distance <= log(2)
  if ("near_alpha" %in% names(logical_tests)) {
    near <- logical_tests$near_alpha
    if (typeof(near) != "logical" || is.object(near) ||
        !is.null(attributes(near)) || anyNA(near) ||
        !identical(near, derived_near)) {
      stop("conditional shadow logical near-alpha evidence is malformed",
           call. = FALSE)
    }
  }
  list(
    integer = integer_values,
    reference_p_value = reference_p_value,
    alpha = alpha,
    log_distance = log_distance,
    near_alpha = derived_near,
    shard_id = shard_id
  )
}

fastkpc_full_cuda_shadow_compute_setup_rows <- function(
    logical_tests, setup_key, shard_id, target_keys, residuals,
    .dcov_batch_fun = fastkpc_legacy_dcov_gamma_cpp_oracle_batch,
    .backend_scope = fastkpc_full_cuda_fixed_sp_with_legacy_dcov_backend) {
  if (!is.function(.dcov_batch_fun) || !is.function(.backend_scope)) {
    stop("conditional shadow dCov backend callbacks are malformed",
         call. = FALSE)
  }
  logical_info <- .fastkpc_full_cuda_shadow_validate_setup_logical(
    logical_tests, setup_key, shard_id
  )
  target_keys <- .fastkpc_full_cuda_shadow_key_vector(
    target_keys, "conditional shadow target keys"
  )
  if (!identical(target_keys, sort(target_keys, method = "radix"))) {
    stop("conditional shadow target keys are not canonical", call. = FALSE)
  }
  matrix_clean <- is.matrix(residuals) && typeof(residuals) == "double" &&
    !is.object(residuals) && nrow(residuals) >= 2L &&
    ncol(residuals) == length(target_keys) && all(is.finite(residuals))
  residual_colnames <- if (is.matrix(residuals)) colnames(residuals) else NULL
  names_clean <- is.null(residual_colnames) ||
    (identical(residual_colnames, target_keys) &&
       !anyDuplicated(residual_colnames))
  if (!isTRUE(matrix_clean) || !isTRUE(names_clean)) {
    stop("conditional shadow residual matrix/columns are malformed",
         call. = FALSE)
  }

  endpoint_x <- match(logical_tests$residual_key_x, target_keys)
  endpoint_y <- match(logical_tests$residual_key_y, target_keys)
  if (anyNA(endpoint_x) || anyNA(endpoint_y)) {
    stop("conditional shadow endpoint residual key is missing from setup",
         call. = FALSE)
  }
  target_index <- setNames(seq_along(target_keys), target_keys)
  x_index <- unname(target_index[logical_tests$residual_key_x])
  y_index <- unname(target_index[logical_tests$residual_key_y])
  pair_count <- nrow(logical_tests)
  oracle <- tryCatch(
    .backend_scope(function() {
      .dcov_batch_fun(
        x = residuals[, x_index, drop = FALSE],
        y = residuals[, y_index, drop = FALSE],
        numCol = 35L, index = 1
      )
    }),
    error = function(error) {
      stop("conditional shadow dCov backend error: ",
           conditionMessage(error), call. = FALSE)
    }
  )
  output_clean <- is.list(oracle) && !is.object(oracle) &&
    all(c("p.value", "diagnostics") %in% names(oracle)) &&
    is.numeric(oracle$p.value) && length(oracle$p.value) == pair_count
  if (!isTRUE(output_clean)) {
    stop("conditional shadow dCov output is malformed", call. = FALSE)
  }
  p_value <- as.double(oracle$p.value)
  if (any(!is.finite(p_value)) || any(p_value < 0 | p_value > 1)) {
    stop("conditional shadow dCov output p-values are malformed",
         call. = FALSE)
  }
  diagnostic <- oracle$diagnostics
  diagnostic_fields <- c(
    "n", "batch_count", "numCol", "index", "lowrank_mode",
    "lowrank_full_eig_count", "lowrank_spectra_count",
    "lowrank_spectra_converged_count", "lowrank_spectra_failed_count",
    "lowrank_spectra_fallback_full_eig_count"
  )
  diagnostic_clean <- is.list(diagnostic) && !is.object(diagnostic) &&
    length(setdiff(diagnostic_fields, names(diagnostic))) == 0L &&
    identical(diagnostic$n, as.integer(nrow(residuals))) &&
    identical(diagnostic$batch_count, as.integer(pair_count)) &&
    identical(diagnostic$numCol, 35L) &&
    identical(as.double(diagnostic$index), 1) &&
    identical(diagnostic$lowrank_mode, "spectra") &&
    identical(diagnostic$lowrank_full_eig_count, 0L) &&
    identical(diagnostic$lowrank_spectra_count,
              as.integer(2L * pair_count)) &&
    identical(diagnostic$lowrank_spectra_converged_count,
              as.integer(2L * pair_count)) &&
    identical(diagnostic$lowrank_spectra_failed_count, 0L) &&
    identical(diagnostic$lowrank_spectra_fallback_full_eig_count, 0L)
  if (!isTRUE(diagnostic_clean)) {
    stop("conditional shadow pinned Spectra diagnostics are malformed",
         call. = FALSE)
  }

  alpha <- logical_info$alpha
  reference <- logical_info$reference_p_value
  candidate_decision <- ifelse(
    p_value > alpha, "independent", "dependent"
  )
  rows <- data.frame(
    logical_sequence_id = logical_info$integer$logical_sequence_id,
    source_sequence_id = logical_info$integer$source_sequence_id,
    source_task_index = logical_info$integer$source_task_index,
    level = logical_info$integer$level,
    S_key = as.character(logical_tests$S_key),
    residual_key_x = as.character(logical_tests$residual_key_x),
    residual_key_y = as.character(logical_tests$residual_key_y),
    prepared_s_key_sha256 = rep.int(setup_key, pair_count),
    shard_id = rep.int(logical_info$shard_id, pair_count),
    reference_p_value = reference,
    candidate_p_value = p_value,
    absolute_p_value_difference = abs(p_value - reference),
    alpha = alpha,
    reference_decision = as.character(logical_tests$reference_decision),
    candidate_decision = candidate_decision,
    decision_flip = candidate_decision != logical_tests$reference_decision,
    near_alpha = logical_info$near_alpha,
    near_alpha_bucket = vapply(
      logical_info$log_distance,
      fastkpc_full_cuda_census_near_alpha_bucket, character(1L)
    ),
    backend = rep.int("legacy-cpp", pair_count),
    low_rank_backend = rep.int("spectra", pair_count),
    backend_error = rep.int(FALSE, pair_count),
    spectra_fallback = rep.int(FALSE, pair_count),
    stringsAsFactors = FALSE
  )
  rownames(rows) <- NULL
  .fastkpc_full_cuda_shadow_validate_atomic_frame(
    rows, .fastkpc_full_cuda_shadow_conditional_base_schema(),
    "conditional shadow setup rows"
  )
  rows
}

.fastkpc_full_cuda_shadow_route_fields <- function() {
  c(
    "prepared_s_key_sha256", "residual_key_sha256", "planned_route",
    "executed_route", "reroute_reason", "solver_status"
  )
}

.fastkpc_full_cuda_shadow_route_reason_exact <- function(
    planned_route, executed_route, reroute_reason) {
  same_route <- planned_route == executed_route
  cholesky_to_svd <- planned_route == "CHOLESKY_BATCHED" &
    executed_route == "AUGMENTED_SVD"
  qr_to_svd <- planned_route == "AUGMENTED_QR" &
    executed_route == "AUGMENTED_SVD"
  all(
    (same_route & reroute_reason == "") |
      (cholesky_to_svd &
         reroute_reason == "CHOLESKY_NON_POSITIVE_PIVOT") |
      (qr_to_svd & reroute_reason == "QR_RANK_GUARD_REJECTED")
  )
}

.fastkpc_full_cuda_shadow_route_status_exact <- function(
    executed_route, solver_status) {
  all(
    (executed_route == "CHOLESKY_BATCHED" &
       solver_status %in% c(
         "OK_CHOLESKY_BATCHED", "OK_CHOLESKY_SINGLE"
       )) |
      (executed_route == "AUGMENTED_QR" &
         solver_status == "OK_AUGMENTED_QR") |
      (executed_route == "AUGMENTED_SVD" &
         solver_status == "OK_AUGMENTED_SVD")
  )
}

.fastkpc_full_cuda_shadow_validate_target_routes <- function(
    target_rows, expected_setup_key = NULL) {
  required <- .fastkpc_full_cuda_shadow_route_fields()
  clean <- is.data.frame(target_rows) && nrow(target_rows) > 0L &&
    length(setdiff(required, names(target_rows))) == 0L &&
    all(vapply(required, function(field) {
      value <- target_rows[[field]]
      typeof(value) == "character" && length(value) == nrow(target_rows) &&
        !is.object(value) && is.null(attributes(value)) && !anyNA(value) &&
        (field == "reroute_reason" || all(nzchar(value)))
    }, logical(1L))) &&
    all(grepl("^[0-9a-f]{64}$", target_rows$prepared_s_key_sha256)) &&
    all(grepl("^[0-9a-f]{64}$", target_rows$residual_key_sha256)) &&
    !anyDuplicated(target_rows$residual_key_sha256) && identical(
      order(target_rows$residual_key_sha256, method = "radix"),
      seq_len(nrow(target_rows))
    )
  if (!isTRUE(clean)) {
    stop("conditional shadow target routes contain malformed or duplicate keys",
         call. = FALSE)
  }
  setup_keys <- unique(target_rows$prepared_s_key_sha256)
  ownership_clean <- length(setup_keys) == 1L &&
    (is.null(expected_setup_key) || identical(setup_keys, expected_setup_key))
  route_levels <- fastkpc_full_cuda_fixed_sp_contract()$route_levels
  status_levels <- fastkpc_full_cuda_fixed_sp_contract()$target_status_levels
  route_clean <- all(target_rows$planned_route %in% route_levels) &&
    all(target_rows$executed_route %in% route_levels) &&
    all(target_rows$solver_status %in% status_levels) &&
    .fastkpc_full_cuda_shadow_route_reason_exact(
      target_rows$planned_route, target_rows$executed_route,
      target_rows$reroute_reason
    ) && .fastkpc_full_cuda_shadow_route_status_exact(
      target_rows$executed_route, target_rows$solver_status
    )
  if (!isTRUE(ownership_clean)) {
    stop("conditional shadow target route ownership is invalid",
         call. = FALSE)
  }
  if (!isTRUE(route_clean)) {
    stop("conditional shadow target route/status values are invalid",
         call. = FALSE)
  }
  invisible(TRUE)
}

fastkpc_full_cuda_shadow_attach_target_routes <- function(
    rows, target_rows, logical_tests) {
  .fastkpc_full_cuda_shadow_validate_atomic_frame(
    rows, .fastkpc_full_cuda_shadow_conditional_base_schema(),
    "conditional shadow setup rows"
  )
  setup_key <- unique(rows$prepared_s_key_sha256)
  if (length(setup_key) != 1L) {
    stop("conditional shadow setup row ownership is invalid",
         call. = FALSE)
  }
  logical_clean <- is.data.frame(logical_tests) &&
    all(c("logical_sequence_id", "x", "y") %in% names(logical_tests)) &&
    nrow(logical_tests) == nrow(rows) &&
    !anyDuplicated(logical_tests$logical_sequence_id)
  logical_match <- if (isTRUE(logical_clean)) {
    match(rows$logical_sequence_id, logical_tests$logical_sequence_id)
  } else {
    rep.int(NA_integer_, nrow(rows))
  }
  x <- suppressWarnings(as.integer(logical_tests$x[logical_match]))
  y <- suppressWarnings(as.integer(logical_tests$y[logical_match]))
  logical_exact <- isTRUE(logical_clean) && !anyNA(logical_match) &&
    identical(
      as.integer(logical_tests$logical_sequence_id[logical_match]),
      rows$logical_sequence_id
    ) && length(x) == nrow(rows) && length(y) == nrow(rows) &&
    !anyNA(x) && !anyNA(y) && all(x >= 1L & x <= 48L) &&
    all(y >= 1L & y <= 48L) && all(x != y)
  if (!isTRUE(logical_exact)) {
    stop("conditional shadow endpoint identity attachment is malformed",
         call. = FALSE)
  }
  rows$x <- x
  rows$y <- y
  .fastkpc_full_cuda_shadow_validate_target_routes(
    target_rows, expected_setup_key = setup_key
  )
  endpoint_x <- match(rows$residual_key_x, target_rows$residual_key_sha256)
  endpoint_y <- match(rows$residual_key_y, target_rows$residual_key_sha256)
  if (anyNA(endpoint_x) || anyNA(endpoint_y)) {
    stop("conditional shadow endpoint route key is missing", call. = FALSE)
  }
  for (endpoint in c("x", "y")) {
    index <- if (endpoint == "x") endpoint_x else endpoint_y
    rows[[paste0("planned_route_", endpoint)]] <-
      target_rows$planned_route[index]
    rows[[paste0("executed_route_", endpoint)]] <-
      target_rows$executed_route[index]
    rows[[paste0("reroute_reason_", endpoint)]] <-
      target_rows$reroute_reason[index]
    rows[[paste0("solver_status_", endpoint)]] <-
      target_rows$solver_status[index]
  }
  schema <- fastkpc_full_cuda_shadow_conditional_row_schema()
  rows <- rows[, names(schema), drop = FALSE]
  rownames(rows) <- NULL
  fastkpc_full_cuda_shadow_validate_conditional_rows(
    rows,
    expected_logical_tests = logical_tests[logical_match, , drop = FALSE],
    expected_setup_key = setup_key,
    expected_shard_id = unique(rows$shard_id),
    expected_target_rows = target_rows
  )
  rows
}

fastkpc_full_cuda_shadow_validate_conditional_rows <- function(
    rows, expected_logical_tests = NULL, expected_setup_key = NULL,
    expected_shard_id = NULL, expected_target_rows = NULL,
    allow_empty = FALSE) {
  schema <- fastkpc_full_cuda_shadow_conditional_row_schema()
  .fastkpc_full_cuda_shadow_validate_atomic_frame(
    rows, schema, "conditional shadow rows", allow_empty = allow_empty
  )
  if (nrow(rows) == 0L) return(TRUE)
  canonical <- !anyDuplicated(rows$logical_sequence_id) && identical(
    order(rows$logical_sequence_id, method = "radix"),
    seq_len(nrow(rows))
  )
  p_value_clean <- all(is.finite(rows$reference_p_value)) &&
    all(rows$reference_p_value >= 0 & rows$reference_p_value <= 1) &&
    all(is.finite(rows$candidate_p_value)) &&
    all(rows$candidate_p_value >= 0 & rows$candidate_p_value <= 1) &&
    all(is.finite(rows$alpha) & rows$alpha > 0 & rows$alpha < 1) &&
    identical(
      rows$absolute_p_value_difference,
      abs(rows$candidate_p_value - rows$reference_p_value)
    )
  reference_decision <- ifelse(
    rows$reference_p_value > rows$alpha, "independent", "dependent"
  )
  candidate_decision <- ifelse(
    rows$candidate_p_value > rows$alpha, "independent", "dependent"
  )
  decision_clean <- identical(rows$reference_decision, reference_decision) &&
    identical(rows$candidate_decision, candidate_decision) && identical(
      rows$decision_flip,
      candidate_decision != reference_decision
    )
  log_distance <- abs(log(
    pmax(rows$reference_p_value, .Machine$double.xmin) / rows$alpha
  ))
  near_alpha <- is.finite(log_distance) & log_distance <= log(2)
  near_bucket <- vapply(
    log_distance, fastkpc_full_cuda_census_near_alpha_bucket, character(1L)
  )
  identity_clean <- all(rows$level > 0L) &&
    all(rows$x >= 1L & rows$x <= 48L) &&
    all(rows$y >= 1L & rows$y <= 48L) && all(rows$x != rows$y) &&
    all(nzchar(rows$S_key)) &&
    all(grepl("^[0-9a-f]{64}$", rows$residual_key_x)) &&
    all(grepl("^[0-9a-f]{64}$", rows$residual_key_y)) &&
    all(grepl("^[0-9a-f]{64}$", rows$prepared_s_key_sha256)) &&
    all(rows$shard_id >= 0L & rows$shard_id < 64L) &&
    identical(rows$near_alpha, near_alpha) &&
    identical(rows$near_alpha_bucket, near_bucket)
  backend_clean <- all(rows$backend == "legacy-cpp") &&
    all(rows$low_rank_backend == "spectra") &&
    !any(rows$backend_error) && !any(rows$spectra_fallback) &&
    !any(rows$decision_flip)
  route_levels <- fastkpc_full_cuda_fixed_sp_contract()$route_levels
  status_levels <- fastkpc_full_cuda_fixed_sp_contract()$target_status_levels
  route_clean <- all(rows$planned_route_x %in% route_levels) &&
    all(rows$planned_route_y %in% route_levels) &&
    all(rows$executed_route_x %in% route_levels) &&
    all(rows$executed_route_y %in% route_levels) &&
    all(rows$solver_status_x %in% status_levels) &&
    all(rows$solver_status_y %in% status_levels) &&
    .fastkpc_full_cuda_shadow_route_reason_exact(
      rows$planned_route_x, rows$executed_route_x,
      rows$reroute_reason_x
    ) && .fastkpc_full_cuda_shadow_route_reason_exact(
      rows$planned_route_y, rows$executed_route_y,
      rows$reroute_reason_y
    ) && .fastkpc_full_cuda_shadow_route_status_exact(
      rows$executed_route_x, rows$solver_status_x
    ) && .fastkpc_full_cuda_shadow_route_status_exact(
      rows$executed_route_y, rows$solver_status_y
    )
  if (!isTRUE(canonical) || !isTRUE(p_value_clean) ||
      !isTRUE(decision_clean) || !isTRUE(identity_clean) ||
      !isTRUE(backend_clean) || !isTRUE(route_clean)) {
    stop("conditional shadow canonical row contract failed",
         call. = FALSE)
  }

  if (!is.null(expected_setup_key)) {
    if (!.fastkpc_full_cuda_shadow_bare_sha256(expected_setup_key) ||
        any(rows$prepared_s_key_sha256 != expected_setup_key)) {
      stop("conditional shadow expected setup ownership mismatch",
           call. = FALSE)
    }
  }
  if (!is.null(expected_shard_id)) {
    expected_shard_id <- .fastkpc_full_cuda_shadow_whole_scalar(
      expected_shard_id, 0L, "expected conditional shadow shard_id"
    )
    if (expected_shard_id >= 64L || any(rows$shard_id != expected_shard_id)) {
      stop("conditional shadow expected shard ownership mismatch",
           call. = FALSE)
    }
  }
  if (!is.null(expected_logical_tests)) {
    required <- .fastkpc_full_cuda_shadow_conditional_logical_fields()
    exact <- is.data.frame(expected_logical_tests) &&
      nrow(expected_logical_tests) == nrow(rows) &&
      length(setdiff(required, names(expected_logical_tests))) == 0L &&
      identical(
        as.integer(expected_logical_tests$logical_sequence_id),
        rows$logical_sequence_id
      ) &&
      all(vapply(c(
        "source_sequence_id", "source_task_index", "level", "x", "y"
      ), function(field) {
        identical(as.integer(expected_logical_tests[[field]]), rows[[field]])
      }, logical(1L))) &&
      all(vapply(c(
        "S_key", "residual_key_x", "residual_key_y",
        "reference_decision"
      ), function(field) {
        identical(as.character(expected_logical_tests[[field]]),
                  rows[[field]])
      }, logical(1L))) &&
      identical(as.double(expected_logical_tests$reference_p_value),
                rows$reference_p_value) &&
      identical(as.double(expected_logical_tests$alpha), rows$alpha) &&
      identical(as.character(expected_logical_tests$prepared_s_key_x),
                rows$prepared_s_key_sha256) &&
      identical(as.character(expected_logical_tests$prepared_s_key_y),
                rows$prepared_s_key_sha256) &&
      identical(as.integer(expected_logical_tests$shard_id), rows$shard_id)
    if (!isTRUE(exact)) {
      stop("conditional shadow logical lineage mismatch", call. = FALSE)
    }
  }
  if (!is.null(expected_target_rows)) {
    target_fields <- .fastkpc_full_cuda_shadow_route_fields()
    if (length(setdiff(target_fields, names(expected_target_rows))) > 0L) {
      stop("conditional shadow expected target routes are malformed",
           call. = FALSE)
    }
    target_routes <- expected_target_rows[, target_fields, drop = FALSE]
    target_routes <- target_routes[order(
      target_routes$residual_key_sha256, method = "radix"
    ), , drop = FALSE]
    rownames(target_routes) <- NULL
    .fastkpc_full_cuda_shadow_validate_target_routes(
      target_routes,
      expected_setup_key = unique(rows$prepared_s_key_sha256)
    )
    endpoint_x <- match(
      rows$residual_key_x, target_routes$residual_key_sha256
    )
    endpoint_y <- match(
      rows$residual_key_y, target_routes$residual_key_sha256
    )
    exact_routes <- !anyNA(endpoint_x) && !anyNA(endpoint_y) &&
      all(vapply(c(
        "planned_route", "executed_route", "reroute_reason",
        "solver_status"
      ), function(field) {
        identical(rows[[paste0(field, "_x")]],
                  target_routes[[field]][endpoint_x]) &&
          identical(rows[[paste0(field, "_y")]],
                    target_routes[[field]][endpoint_y])
      }, logical(1L)))
    if (!isTRUE(exact_routes)) {
      stop("conditional shadow endpoint route lineage mismatch",
           call. = FALSE)
    }
  }
  TRUE
}

fastkpc_full_cuda_shadow_scope <- function(catalog, plan, scope) {
  required_helpers <- c(
    "fastkpc_full_cuda_fixed_sp_scope",
    ".fastkpc_full_cuda_phase3_oracle_descriptor_target_rows",
    "fastkpc_full_cuda_census_file_hash"
  )
  missing <- required_helpers[!vapply(
    required_helpers, exists, logical(1L), mode = "function",
    inherits = TRUE
  )]
  if (length(missing) > 0L) {
    stop("conditional shadow scope helpers are unavailable: ",
         paste(missing, collapse = ","), call. = FALSE)
  }
  .fastkpc_full_cuda_shadow_validate_supplied_plan(catalog, plan)
  if (typeof(scope) != "character" || length(scope) != 1L ||
      is.object(scope) || !is.null(attributes(scope)) || is.na(scope) ||
      !scope %in% c("iteration", "qualification", "full")) {
    stop("conditional shadow scope is invalid", call. = FALSE)
  }
  selected <- fastkpc_full_cuda_fixed_sp_scope(catalog, scope)
  setup_keys <- as.character(selected$setup_rows$prepared_s_key_sha256)
  setup_keys <- .fastkpc_full_cuda_shadow_key_vector(
    setup_keys, "conditional shadow scope setup keys"
  )
  if (!identical(setup_keys, sort(setup_keys, method = "radix"))) {
    stop("conditional shadow scope setup order is not canonical",
         call. = FALSE)
  }
  target_rows <- .fastkpc_full_cuda_phase3_oracle_descriptor_target_rows(
    catalog, selected$target_rows
  )
  target_rows$shard_id <- as.integer(target_rows$phase2_shard_id)
  descriptor_fields <-
    .fastkpc_full_cuda_phase3_oracle_descriptor_target_fields()
  target_rows <- target_rows[, descriptor_fields, drop = FALSE]
  setup_match <- match(
    setup_keys, catalog$setup_index$prepared_s_key_sha256
  )
  if (anyNA(setup_match)) {
    stop("conditional shadow scope setup association is incomplete",
         call. = FALSE)
  }
  setup_assignments <- data.frame(
    prepared_s_key_sha256 = setup_keys,
    shard_id = as.integer(
      (setup_match - 1L) %% catalog$catalog_contract$shard_count
    ),
    stringsAsFactors = FALSE
  )
  target_setup_match <- match(
    target_rows$prepared_s_key_sha256,
    setup_assignments$prepared_s_key_sha256
  )
  if (anyNA(target_setup_match) || !identical(
        target_rows$shard_id,
        setup_assignments$shard_id[target_setup_match]
      )) {
    stop("conditional shadow target shard ownership is inconsistent",
         call. = FALSE)
  }

  if (identical(scope, "full")) {
    logical_tests <- plan$conditional_tests
  } else {
    semantic_name <- paste0(scope, "_logical_tests_rds")
    logical_path <- file.path(
      catalog$phase2_dir, paste0(scope, "_logical_tests.rds")
    )
    expected_hash <- unname(
      fastkpc_full_cuda_fixed_sp_catalog_contract()$
        phase2_semantic_file_sha256[[semantic_name]]
    )
    if (!file.exists(logical_path) || dir.exists(logical_path) ||
        !identical(
          fastkpc_full_cuda_census_file_hash(logical_path), expected_hash
        )) {
      stop("conditional shadow logical selection hash mismatch",
           call. = FALSE)
    }
    logical_tests <- tryCatch(
      readRDS(logical_path),
      error = function(error) {
        stop("conditional shadow logical selection is unreadable",
             call. = FALSE)
      }
    )
    expected_count <- if (identical(scope, "iteration")) 44L else 3808L
    logical_clean <- is.data.frame(logical_tests) &&
      nrow(logical_tests) == expected_count &&
      !anyDuplicated(names(logical_tests)) &&
      "logical_sequence_id" %in% names(logical_tests) &&
      typeof(logical_tests$logical_sequence_id) == "integer" &&
      !anyNA(logical_tests$logical_sequence_id) &&
      !anyDuplicated(logical_tests$logical_sequence_id) && identical(
        order(logical_tests$logical_sequence_id, method = "radix"),
        seq_len(nrow(logical_tests))
      )
    if (!isTRUE(logical_clean)) {
      stop("conditional shadow logical selection is malformed",
           call. = FALSE)
    }
    plan_match <- match(
      logical_tests$logical_sequence_id,
      plan$conditional_tests$logical_sequence_id
    )
    if (anyNA(plan_match)) {
      stop("conditional shadow logical selection is outside the plan",
           call. = FALSE)
    }
    mapped <- plan$conditional_tests[plan_match, , drop = FALSE]
    shared_fields <- intersect(names(logical_tests), names(mapped))
    exact_lineage <- all(vapply(shared_fields, function(field) {
      identical(logical_tests[[field]], mapped[[field]])
    }, logical(1L)))
    if (!isTRUE(exact_lineage)) {
      stop("conditional shadow logical selection lineage mismatch",
           call. = FALSE)
    }
    logical_tests$prepared_s_key_x <- mapped$prepared_s_key_x
    logical_tests$prepared_s_key_y <- mapped$prepared_s_key_y
    logical_tests$shard_id <- mapped$shard_id
  }
  if (!"near_alpha" %in% names(logical_tests)) {
    logical_tests$near_alpha <-
      is.finite(logical_tests$absolute_log_distance_from_alpha) &
      logical_tests$absolute_log_distance_from_alpha <= log(2)
  }

  logical_ids <- as.integer(logical_tests$logical_sequence_id)
  endpoint_x <- match(
    logical_tests$residual_key_x, target_rows$residual_key_sha256
  )
  endpoint_y <- match(
    logical_tests$residual_key_y, target_rows$residual_key_sha256
  )
  setup_x <- if (anyNA(endpoint_x)) character() else {
    target_rows$prepared_s_key_sha256[endpoint_x]
  }
  setup_y <- if (anyNA(endpoint_y)) character() else {
    target_rows$prepared_s_key_sha256[endpoint_y]
  }
  expected_setup_set <- sort(unique(
    as.character(logical_tests$prepared_s_key_x)
  ), method = "radix")
  exact <- !anyDuplicated(logical_ids) && identical(
    order(logical_ids, method = "radix"), seq_along(logical_ids)
  ) && !anyNA(endpoint_x) && !anyNA(endpoint_y) && identical(
    setup_x, as.character(logical_tests$prepared_s_key_x)
  ) && identical(
    setup_y, as.character(logical_tests$prepared_s_key_y)
  ) && identical(
    as.integer(logical_tests$shard_id),
    target_rows$shard_id[endpoint_x]
  ) && identical(expected_setup_set, setup_keys)
  if (!isTRUE(exact)) {
    stop("conditional shadow scope logical/target ownership mismatch",
         call. = FALSE)
  }
  rownames(target_rows) <- rownames(logical_tests) <- NULL
  list(
    scope = scope,
    setup_keys = setup_keys,
    setup_assignments = setup_assignments,
    target_rows = target_rows,
    logical_tests = logical_tests
  )
}
