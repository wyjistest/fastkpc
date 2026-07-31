fastkpc_full_cuda_phase7_require <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
}

fastkpc_full_cuda_phase7_source_paths <- function() {
  native <- list.files(
    "fastkpc/src", recursive = TRUE, full.names = TRUE,
    include.dirs = FALSE
  )
  native <- native[grepl(
    "\\.(c|cc|cpp|cxx|cu|h|hh|hpp|hxx|cuh|inc)$", native
  )]
  phase7 <- c(
    "fastkpc/R/cuda_native.R",
    "fastkpc/R/full_cuda_ci_gate.R",
    "fastkpc/R/full_cuda_ci_oracle_contract.R",
    "fastkpc/R/full_cuda_ci_workload_census.R",
    "fastkpc/R/full_cuda_ci_prepared_s_contract.R",
    "fastkpc/R/full_cuda_ci_fixed_sp_runtime.R",
    "fastkpc/R/full_cuda_ci_fixed_sp_shadow.R",
    "fastkpc/R/full_cuda_ci_phase3_artifacts.R",
    "fastkpc/R/full_cuda_ci_phase35_contracts.R",
    "fastkpc/R/full_cuda_ci_single_penalty_gcv.R",
    "fastkpc/R/full_cuda_ci_phase4_artifacts.R",
    "fastkpc/R/full_cuda_ci_multi_penalty_cpp.R",
    "fastkpc/R/full_cuda_ci_phase5_artifacts.R",
    "fastkpc/R/full_cuda_ci_multi_penalty_cuda.R",
    "fastkpc/R/full_cuda_ci_phase6_artifacts.R",
    "fastkpc/R/full_cuda_ci_native_setup.R",
    "fastkpc/R/full_cuda_ci_phase7_artifacts.R",
    "fastkpc/R/full_cuda_ci_phase7_publication.R",
    "fastkpc/tools/build_cuda_native.sh",
    "fastkpc/tools/run_full_cuda_ci_native_setup_corpus.R",
    "fastkpc/tools/merge_full_cuda_ci_native_setup_evidence.R",
    "fastkpc/tools/run_full_cuda_ci_native_setup_gate.sh",
    "fastkpc/tools/run_full_cuda_ci_native_setup_artifacts.R",
    "fastkpc/tools/run_full_cuda_ci_single_penalty_gcv_shadow.R",
    "fastkpc/tools/run_full_cuda_ci_single_penalty_gcv_partitions.sh",
    "fastkpc/tools/run_full_cuda_ci_multi_penalty_cuda_partition.R"
  )
  paths <- sort(unique(c(native, phase7)), method = "radix")
  fastkpc_full_cuda_phase7_require(
    length(paths) > 0L && all(file.exists(paths) & !dir.exists(paths)),
    "Phase 7 source closure contains a missing file"
  )
  paths
}

fastkpc_full_cuda_phase7_source_closure <- function() {
  paths <- fastkpc_full_cuda_phase7_source_paths()
  hashes <- setNames(as.list(vapply(
    paths, fastkpc_full_cuda_census_file_hash, character(1L)
  )), paths)
  list(
    table = data.frame(
      path = names(hashes),
      sha256 = unlist(hashes, use.names = FALSE),
      stringsAsFactors = FALSE
    ),
    hashes = hashes,
    sha256 = fastkpc_full_cuda_phase35_sha256_utf8(
      fastkpc_full_cuda_phase35_canonical_json(hashes)
    )
  )
}

fastkpc_full_cuda_phase7_native_identity <- function() {
  load_fastkpc_cuda_native()
  dll <- getLoadedDLLs()[["fastkpc_cuda"]]
  fastkpc_full_cuda_phase7_require(
    !is.null(dll), "Phase 7 native DLL is not loaded"
  )
  path <- normalizePath(dll[["path"]], winslash = "/", mustWork = TRUE)
  list(path = path, sha256 = fastkpc_full_cuda_census_file_hash(path))
}

fastkpc_full_cuda_phase7_execution_identity <- function(catalog, mode) {
  mode <- match.arg(mode, c(
    "native-setup-oracle", "native-setup-shadow",
    "native-setup-backend"
  ))
  source <- fastkpc_full_cuda_phase7_source_closure()
  native <- fastkpc_full_cuda_phase7_native_identity()
  inputs <- c(
    phase0_manifest = file.path(catalog$phase0_dir, "manifest.json"),
    phase1_manifest = file.path(catalog$phase1_dir, "manifest.json"),
    phase2_manifest = file.path(catalog$phase2_dir, "manifest.json")
  )
  input_hashes <- setNames(as.list(vapply(
    inputs, fastkpc_full_cuda_census_file_hash, character(1L)
  )), names(inputs))
  value <- list(
    schema_version = "full-cuda-ci-phase7-execution-identity-v1",
    source_commit = fastkpc_full_cuda_source_commit(),
    producer_source_closure_sha256 = source$sha256,
    native_binary_sha256 = native$sha256,
    route_semantic_version = paste0(
      "full-cuda-ci-phase7-", mode, "-v1"
    ),
    native_setup_semantic_version =
      "mgcv-1.9-1-tprs-native-setup-v1",
    dataset_sha256 = catalog$inputs$dataset_sha256,
    input_file_sha256 = input_hashes,
    setup_count = 8634L,
    target_count = 110617L,
    logical_test_count = 240489L
  )
  value$identity_sha256 <- fastkpc_full_cuda_census_named_metadata_hash(value)
  value
}

fastkpc_full_cuda_phase7_scalar <- function(row, field, mode = NULL) {
  fastkpc_full_cuda_phase7_require(
    is.data.frame(row) && nrow(row) == 1L && field %in% names(row),
    paste0("Phase 7 setup row is missing ", field)
  )
  value <- row[[field]][[1L]]
  if (!is.null(mode)) {
    value <- switch(
      mode,
      character = as.character(value),
      integer = as.integer(value),
      numeric = as.numeric(value),
      logical = as.logical(value),
      stop("unsupported Phase 7 scalar mode", call. = FALSE)
    )
  }
  value
}

fastkpc_full_cuda_phase7_setup_scope <- function(catalog) {
  scope <- fastkpc_full_cuda_fixed_sp_scope(catalog, "full")
  setup_rank <- match(
    scope$setup_rows$prepared_s_key_sha256,
    catalog$setup_index$prepared_s_key_sha256
  )
  shard_count <- as.integer(catalog$catalog_contract$shard_count)
  shard_id <- as.integer((setup_rank - 1L) %% shard_count)
  fastkpc_full_cuda_phase7_require(
    nrow(scope$setup_rows) == 8634L && !anyNA(setup_rank) &&
      !anyDuplicated(scope$setup_rows$prepared_s_key_sha256) &&
      all(shard_id >= 0L & shard_id < shard_count),
    "Phase 7 canonical setup scope is malformed"
  )
  list(
    setup_rows = scope$setup_rows,
    setup_rank = as.integer(setup_rank),
    shard_id = shard_id,
    shard_ids = sort(unique(shard_id))
  )
}

fastkpc_full_cuda_phase7_native_candidate <- function(catalog, setup_row) {
  data <- as.matrix(catalog$inputs$data)
  storage.mode(data) <- "double"
  sorted_S <- fastkpc_full_cuda_census_parse_s(
    fastkpc_full_cuda_phase7_scalar(setup_row, "S_key", "character")
  )
  conditioning <- data[, sorted_S, drop = FALSE]
  candidate <- full_cuda_ci_native_setup_native(conditioning)
  expected_key <- fastkpc_full_cuda_phase7_scalar(
    setup_row, "prepared_s_key_sha256", "character"
  )
  expected_penalty_ranks <- unname(as.integer(
    fastkpc_full_cuda_phase7_scalar(setup_row, "penalty_ranks")
  ))
  expected_penalty_offsets <- unname(as.integer(
    fastkpc_full_cuda_phase7_scalar(setup_row, "penalty_offsets")
  ))
  expected_basis_dimensions <- unname(as.integer(
    fastkpc_full_cuda_phase7_scalar(setup_row, "basis_dimensions")
  ))
  expected_penalty_hashes <- unname(as.character(
    fastkpc_full_cuda_phase7_scalar(setup_row, "penalty_hashes")
  ))
  actual_penalty_hashes <- unname(vapply(
    candidate$penalty_blocks,
    fastkpc_full_cuda_census_metadata_hash,
    character(1L)
  ))
  expected_dimensions <- c(
    fastkpc_full_cuda_phase7_scalar(
      setup_row, "model_matrix_nrow", "integer"
    ),
    fastkpc_full_cuda_phase7_scalar(
      setup_row, "model_matrix_ncol", "integer"
    )
  )
  constraint_dimensions <- unname(as.integer(
    fastkpc_full_cuda_phase7_scalar(setup_row, "constraint_dimensions")
  ))
  clean <-
    identical(candidate$schema_version, "full-cuda-ci-native-setup-v1") &&
    identical(candidate$semantic_version,
              "mgcv-1.9-1-tprs-native-setup-v1") &&
    identical(candidate$formula_class,
              fastkpc_full_cuda_phase7_scalar(
                setup_row, "formula_class", "character"
              )) &&
    identical(candidate$S_size, as.integer(length(sorted_S))) &&
    identical(dim(candidate$X), expected_dimensions) &&
    identical(candidate$penalty_offsets, expected_penalty_offsets) &&
    identical(candidate$penalty_ranks, expected_penalty_ranks) &&
    identical(candidate$basis_dimensions, expected_basis_dimensions) &&
    identical(actual_penalty_hashes, expected_penalty_hashes) &&
    identical(constraint_dimensions, c(0L, ncol(candidate$X))) &&
    identical(
      fastkpc_full_cuda_phase7_scalar(
        setup_row, "constraint_nullspace_dimension", "integer"
      ),
      as.integer(ncol(candidate$X))
    ) &&
    identical(
      fastkpc_full_cuda_phase7_scalar(setup_row, "H_hash", "character"),
      "NONE"
    ) &&
    identical(
      fastkpc_full_cuda_phase7_scalar(
        setup_row, "weights_policy", "character"
      ),
      "none"
    ) &&
    identical(
      fastkpc_full_cuda_phase7_scalar(
        setup_row, "offset_policy", "character"
      ),
      "none"
    ) &&
    identical(candidate$diagnostics$legacy_mgcv_setup_count, 0L) &&
    identical(candidate$diagnostics$r_callback_count, 0L) &&
    identical(candidate$diagnostics$unsupported_count, 0L) &&
    is.character(expected_key) && grepl("^[0-9a-f]{64}$", expected_key)
  fastkpc_full_cuda_phase7_require(
    clean,
    paste0("Phase 7 native setup metadata mismatch: ", expected_key)
  )
  list(candidate = candidate, sorted_S = unname(as.integer(sorted_S)))
}

fastkpc_full_cuda_phase7_runtime_setup <- function(
    catalog, setup_row, target_states, setup_key) {
  setup_key <- as.character(setup_key)
  fastkpc_full_cuda_phase7_require(
    length(setup_key) == 1L && !is.na(setup_key) &&
      grepl("^[0-9a-f]{64}$", setup_key) &&
      identical(
        setup_key,
        fastkpc_full_cuda_phase7_scalar(
          setup_row, "prepared_s_key_sha256", "character"
        )
      ) && is.data.frame(target_states) && nrow(target_states) > 0L &&
      all(target_states$prepared_s_key_sha256 == setup_key),
    "Phase 7 runtime setup lineage is malformed"
  )
  built <- fastkpc_full_cuda_phase7_native_candidate(catalog, setup_row)
  candidate <- built$candidate
  label_values <- target_states$selected_sp_names
  labels <- unname(as.character(label_values[[1L]]))
  labels_clean <- length(labels) == length(candidate$penalty_blocks) &&
    all(nzchar(labels)) && all(vapply(
      label_values,
      function(value) identical(unname(as.character(value)), labels),
      logical(1L)
    ))
  fastkpc_full_cuda_phase7_require(
    labels_clean, "Phase 7 smoothing-parameter labels are inconsistent"
  )
  p <- ncol(candidate$X)
  diagnostics <- list(
    schema_version = "full-cuda-ci-native-setup-authority-v1",
    semantic_version = candidate$semantic_version,
    semantic_fingerprint = candidate$semantic_fingerprint,
    native_setup_count = 1L,
    native_geometry_count = 0L,
    oracle_setup_count = 0L,
    legacy_mgcv_setup_count = 0L,
    legacy_mgcv_fit_count = 0L,
    r_callback_count = 0L,
    unsupported_count = 0L,
    authority = "native-cpp"
  )
  provider_fingerprint <- fastkpc_full_cuda_census_named_metadata_hash(list(
    schema_version = "full-cuda-ci-native-setup-provider-v1",
    semantic_version = candidate$semantic_version
  ))
  semantic_fingerprint <- fastkpc_full_cuda_census_named_metadata_hash(list(
    schema_version = "full-cuda-ci-native-runtime-semantic-v1",
    prepared_s_key_sha256 = setup_key,
    native_semantic_fingerprint = candidate$semantic_fingerprint
  ))
  representation_fingerprint <-
    fastkpc_full_cuda_census_named_metadata_hash(list(
      schema_version = "full-cuda-ci-native-runtime-representation-v1",
      semantic_fingerprint = semantic_fingerprint,
      model_matrix_sha256 =
        fastkpc_full_cuda_census_metadata_hash(candidate$X),
      gram_matrix_sha256 =
        fastkpc_full_cuda_census_metadata_hash(candidate$gram_matrix),
      penalty_sha256 = unname(vapply(
        candidate$penalty_blocks,
        fastkpc_full_cuda_census_metadata_hash,
        character(1L)
      ))
    ))
  list(
    schema_version = "full-cuda-ci-native-runtime-setup-v1",
    dataset_sha256 = as.character(catalog$inputs$dataset_sha256),
    prepared_s_key_sha256 = setup_key,
    same_S_group_id = fastkpc_full_cuda_phase7_scalar(
      setup_row, "same_S_group_id", "character"
    ),
    phase1_setup_fingerprint = fastkpc_full_cuda_phase7_scalar(
      setup_row, "setup_fingerprint", "character"
    ),
    phase1_model_matrix_hash = fastkpc_full_cuda_phase7_scalar(
      setup_row, "model_matrix_hash", "character"
    ),
    sorted_S = built$sorted_S,
    formula_class = candidate$formula_class,
    formula_semantics_version = "kpcalg_regrXonS_v1",
    native_semantics_version = candidate$semantic_version,
    native_semantic_fingerprint = candidate$semantic_fingerprint,
    provider_fingerprint = provider_fingerprint,
    semantic_fingerprint = semantic_fingerprint,
    representation_fingerprint = representation_fingerprint,
    X = candidate$X,
    gram_matrix = candidate$gram_matrix,
    penalty_blocks = candidate$penalty_blocks,
    penalty_offsets = unname(as.integer(candidate$penalty_offsets)),
    penalty_ranks = unname(as.integer(candidate$penalty_ranks)),
    mgcv_penalty_rank_metadata =
      unname(as.integer(candidate$penalty_ranks)),
    penalty_sp_indices = seq_along(candidate$penalty_blocks),
    penalty_sp_labels = labels,
    constraint = matrix(numeric(), nrow = 0L, ncol = p),
    constraint_mode = "identity",
    constraint_nullspace = NULL,
    constraint_nullspace_dimension = as.integer(p),
    H = NULL,
    weights = NULL,
    weights_policy = "none-or-unit",
    offset_policy = "none-or-zero",
    nullspace_gram_matrix = NULL,
    basis_dimensions = candidate$basis_dimensions,
    smooth_null_space_dimensions = candidate$null_space_dimensions,
    smooth_S_scale = candidate$smooth_S_scale,
    smooth_shift = candidate$shifts,
    native_setup_diagnostics = diagnostics
  )
}

fastkpc_full_cuda_phase7_fixed_sp_dto <- function(setup) {
  fastkpc_full_cuda_phase7_require(
    is.list(setup) && identical(
      setup$schema_version, "full-cuda-ci-native-runtime-setup-v1"
    ) && is.matrix(setup$X) && is.matrix(setup$gram_matrix) &&
      identical(dim(setup$gram_matrix), rep.int(ncol(setup$X), 2L)) &&
      identical(setup$constraint_mode, "identity") &&
      is.null(setup$constraint_nullspace) &&
      is.null(setup$nullspace_gram_matrix) && is.null(setup$H),
    "Phase 7 native fixed-sp DTO setup is malformed"
  )
  schema <- fastkpc_full_cuda_fixed_sp_contract()$native_dto_schema_version
  value <- list(
    schema_version = schema,
    dataset_sha256 = setup$dataset_sha256,
    prepared_s_key_sha256 = setup$prepared_s_key_sha256,
    same_S_group_id = setup$same_S_group_id,
    phase1_setup_fingerprint = setup$phase1_setup_fingerprint,
    provider_fingerprint = setup$provider_fingerprint,
    semantic_fingerprint = setup$semantic_fingerprint,
    representation_fingerprint = setup$representation_fingerprint,
    prepared_s_setup_schema_version = "full-cuda-ci-prepared-s-setup-v1",
    native_dto_schema_version = schema,
    data_p = 48L,
    n = as.integer(nrow(setup$X)),
    coefficient_dim = as.integer(ncol(setup$X)),
    null_dim = as.integer(setup$constraint_nullspace_dimension),
    penalty_count = as.integer(length(setup$penalty_blocks)),
    X = setup$X,
    constraint_mode = setup$constraint_mode,
    constraint_nullspace = setup$constraint_nullspace,
    gram_matrix = setup$gram_matrix,
    nullspace_gram_matrix = setup$nullspace_gram_matrix,
    penalty_blocks = setup$penalty_blocks,
    penalty_offsets_zero_based = setup$penalty_offsets - 1L,
    penalty_ranks = setup$penalty_ranks,
    penalty_sp_indices_zero_based = setup$penalty_sp_indices - 1L,
    penalty_sp_labels = setup$penalty_sp_labels,
    H = setup$H,
    weights_policy = setup$weights_policy,
    offset_policy = setup$offset_policy
  )
  fastkpc_full_cuda_phase7_require(
    identical(names(value), fastkpc_full_cuda_fixed_sp_native_dto_fields()),
    "Phase 7 native fixed-sp DTO fields drifted"
  )
  value
}

fastkpc_full_cuda_phase7_setup_builder <- function(
    catalog, setup_row, target_states, setup_key) {
  fastkpc_full_cuda_phase7_runtime_setup(
    catalog = catalog, setup_row = setup_row,
    target_states = target_states, setup_key = setup_key
  )
}

fastkpc_full_cuda_phase7_max_list_error <- function(candidate, oracle) {
  fastkpc_full_cuda_phase7_require(
    is.list(candidate) && is.list(oracle) &&
      length(candidate) == length(oracle),
    "Phase 7 list comparison dimensions differ"
  )
  max(vapply(seq_along(candidate), function(index) {
    left <- as.numeric(candidate[[index]])
    right <- as.numeric(oracle[[index]])
    if (length(left) != length(right)) return(Inf)
    if (length(left) == 0L) return(0)
    max(abs(left - right))
  }, numeric(1L)))
}

fastkpc_full_cuda_phase7_compare_setup <- function(
    catalog, setup_row, oracle, shard_id, rebuild_oracle = FALSE) {
  setup_key <- fastkpc_full_cuda_phase7_scalar(
    setup_row, "prepared_s_key_sha256", "character"
  )
  if (isTRUE(rebuild_oracle)) {
    canonical_fields <- names(catalog$inputs$same_s_setup_metadata)
    rebuilt <- fastkpc_full_cuda_build_prepared_s_setup(
      catalog$inputs, setup_row[, canonical_fields, drop = FALSE]
    )
    fastkpc_full_cuda_phase7_require(
      identical(rebuilt$prepared_s_key_sha256, setup_key) &&
        identical(rebuilt$semantic_fingerprint,
                  oracle$semantic_fingerprint) &&
        identical(rebuilt$representation_fingerprint,
                  oracle$representation_fingerprint),
      paste0("Phase 7 rebuilt oracle drift: ", setup_key)
    )
    oracle <- rebuilt
  }
  native <- fastkpc_full_cuda_phase7_native_candidate(
    catalog, setup_row
  )$candidate
  native_geometry <- full_cuda_ci_native_geometry_prepare_native(
    native$X, native$penalty_blocks, native$penalty_offsets,
    native$penalty_ranks
  )
  oracle_qr <- qr(oracle$X, LAPACK = TRUE)
  oracle_mroot <- get("mroot", envir = asNamespace("mgcv"))
  oracle_roots <- lapply(seq_along(oracle$penalty_blocks), function(index) {
    local <- oracle_mroot(
      oracle$penalty_blocks[[index]],
      rank = oracle$mgcv_penalty_rank_metadata[[index]],
      method = "chol"
    )
    full <- matrix(
      0, ncol(oracle$X), oracle$mgcv_penalty_rank_metadata[[index]]
    )
    rows <- oracle$penalty_offsets[[index]] + seq_len(nrow(local)) - 1L
    full[rows, ] <- local
    full[oracle_qr$pivot, , drop = FALSE]
  })
  oracle_penalty_matrices <- lapply(oracle_roots, tcrossprod)
  oracle_initial_sp <- get(
    "initial.sp", envir = asNamespace("mgcv")
  )(oracle$X, oracle$penalty_blocks, oracle$penalty_offsets)
  x_identical <- identical(native$X, oracle$X)
  penalty_error <- fastkpc_full_cuda_phase7_max_list_error(
    native$penalty_blocks, oracle$penalty_blocks
  )
  projector_error <- if (x_identical) {
    0
  } else {
    native_q <- qr.Q(qr(native$X, LAPACK = TRUE), complete = FALSE)
    oracle_q <- qr.Q(oracle_qr, complete = FALSE)
    max(abs(tcrossprod(native_q) - tcrossprod(oracle_q)))
  }
  smooth_columns <- if (ncol(native$X) > 1L) 2:ncol(native$X) else integer()
  constraint_error <- if (length(smooth_columns) == 0L) 0 else max(abs(
    colMeans(native$X[, smooth_columns, drop = FALSE]) -
      colMeans(oracle$X[, smooth_columns, drop = FALSE])
  ))
  data.frame(
    prepared_s_key_sha256 = setup_key,
    same_S_group_id = fastkpc_full_cuda_phase7_scalar(
      setup_row, "same_S_group_id", "character"
    ),
    shard_id = as.integer(shard_id),
    S_size = fastkpc_full_cuda_phase7_scalar(
      setup_row, "S_size", "integer"
    ),
    formula_class = native$formula_class,
    coefficient_count = ncol(native$X),
    penalty_count = length(native$penalty_blocks),
    native_semantic_fingerprint = native$semantic_fingerprint,
    x_max_absolute_error = max(abs(native$X - oracle$X)),
    model_space_projector_max_absolute_error = projector_error,
    constraint_max_absolute_error = constraint_error,
    penalty_operator_max_absolute_error = penalty_error,
    shift_max_absolute_error = fastkpc_full_cuda_phase7_max_list_error(
      native$shifts, oracle$smooth_shift
    ),
    penalty_scale_max_absolute_error = max(abs(
      as.numeric(native$smooth_S_scale) -
        as.numeric(unlist(oracle$smooth_S_scale, use.names = FALSE))
    )),
    rank_mismatch_count = sum(
      native$penalty_ranks != oracle$mgcv_penalty_rank_metadata
    ),
    null_space_mismatch_count = sum(
      native$null_space_dimensions !=
        oracle$smooth_null_space_dimensions
    ),
    basis_dimension_mismatch_count = sum(
      native$basis_dimensions != oracle$basis_dimensions
    ),
    qr_q_max_absolute_error = max(abs(
      native_geometry$magic_q - qr.Q(oracle_qr, complete = FALSE)
    )),
    qr_packed_max_absolute_error = max(abs(
      native_geometry$magic_qr_packed - unclass(oracle_qr)$qr
    )),
    qr_tau_max_absolute_error = max(abs(
      native_geometry$magic_tau - as.numeric(oracle_qr$qraux)
    )),
    qr_r_max_absolute_error = max(abs(
      native_geometry$magic_r - qr.R(oracle_qr, complete = FALSE)
    )),
    qr_pivot_mismatch_count = sum(
      native_geometry$magic_pivot != as.integer(oracle_qr$pivot)
    ),
    mroot_max_absolute_error = fastkpc_full_cuda_phase7_max_list_error(
      native_geometry$penalty_roots, oracle_roots
    ),
    penalty_matrix_max_absolute_error =
      fastkpc_full_cuda_phase7_max_list_error(
        native_geometry$penalty_matrices, oracle_penalty_matrices
      ),
    initial_sp_max_absolute_error = max(abs(
      native_geometry$initial_sp - as.numeric(oracle_initial_sp)
    )),
    fixed_sp_operator_equivalent =
      x_identical && penalty_error == 0,
    native_setup_count = 1L,
    oracle_setup_count = 1L,
    legacy_mgcv_setup_count = if (isTRUE(rebuild_oracle)) 1L else 0L,
    r_callback_count = 0L,
    unsupported_count = 0L,
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_phase7_scan_setup_shard <- function(
    catalog, shard_id, rebuild_oracle = TRUE, progress = interactive(),
    preparation = NULL, execution_identity = NULL) {
  shard_id <- as.integer(shard_id)
  scope <- fastkpc_full_cuda_phase7_setup_scope(catalog)
  selected <- scope$shard_id == shard_id
  setup_rows <- scope$setup_rows[selected, , drop = FALSE]
  fastkpc_full_cuda_phase7_require(
    length(shard_id) == 1L && !is.na(shard_id) &&
      shard_id %in% scope$shard_ids && nrow(setup_rows) > 0L,
    "Phase 7 setup shard is outside the canonical scope"
  )
  if (!is.null(execution_identity)) {
    fastkpc_full_cuda_phase7_require(
      is.list(execution_identity) && identical(
        execution_identity$schema_version,
        "full-cuda-ci-phase7-execution-identity-v1"
      ) && identical(
        execution_identity$route_semantic_version,
        "full-cuda-ci-phase7-native-setup-oracle-v1"
      ),
      "Phase 7 supplied setup-shard execution identity is malformed"
    )
  }
  if (is.null(preparation)) {
    preparation <- .fastkpc_full_cuda_prepared_s_prepare_shard_read(
      inputs = catalog$inputs,
      shard_count = catalog$catalog_contract$shard_count,
      expected_source_commit = catalog$phase2_manifest$source_commit
    )
  }
  completed <- .fastkpc_full_cuda_prepared_s_read_one_shard(
    shard_dir = file.path(catalog$phase2_dir, "shards"),
    shard_id = shard_id, inputs = catalog$inputs,
    preparation = preparation
  )
  payload <- completed$payload
  setup_keys <- as.character(setup_rows$prepared_s_key_sha256)
  setup_match <- match(setup_keys, payload$ordered_setup_keys)
  fastkpc_full_cuda_phase7_require(
    !anyNA(setup_match), "Phase 7 oracle shard is missing a setup"
  )
  rows <- vector("list", length(setup_keys))
  started <- proc.time()[["elapsed"]]
  for (index in seq_along(setup_keys)) {
    rows[[index]] <- fastkpc_full_cuda_phase7_compare_setup(
      catalog = catalog,
      setup_row = setup_rows[index, , drop = FALSE],
      oracle = payload$prepared_s_setups[[setup_match[[index]]]],
      shard_id = shard_id,
      rebuild_oracle = rebuild_oracle
    )
    if (isTRUE(progress) && index %% 25L == 0L) {
      cat(
        "Phase 7 setup shard ", shard_id, ": ", index, "/",
        length(setup_keys), "\n", sep = ""
      )
      flush.console()
    }
  }
  rows <- do.call(rbind, rows)
  rownames(rows) <- NULL
  list(
    schema_version = "full-cuda-ci-native-setup-shard-evidence-v1",
    shard_id = shard_id,
    rebuild_oracle = isTRUE(rebuild_oracle),
    setup_keys = setup_keys,
    elapsed_seconds = proc.time()[["elapsed"]] - started,
    execution_identity = if (is.null(execution_identity)) {
      fastkpc_full_cuda_phase7_execution_identity(
        catalog, "native-setup-oracle"
      )
    } else {
      execution_identity
    },
    rows = rows
  )
}

fastkpc_full_cuda_phase7_merge_setup_shards <- function(catalog, paths) {
  paths <- as.character(paths)
  fastkpc_full_cuda_phase7_require(
    length(paths) == 64L && !anyDuplicated(paths) &&
      all(file.exists(paths) & !dir.exists(paths)),
    "Phase 7 requires all 64 setup shard files"
  )
  parts <- lapply(paths, readRDS)
  shard_ids <- vapply(parts, `[[`, integer(1L), "shard_id")
  clean_parts <- identical(sort(shard_ids), 0:63) && all(vapply(
    parts, function(value) {
      identical(
        value$schema_version,
        "full-cuda-ci-native-setup-shard-evidence-v1"
      ) && isTRUE(value$rebuild_oracle) && is.data.frame(value$rows) &&
        nrow(value$rows) == length(value$setup_keys) &&
        identical(value$rows$prepared_s_key_sha256, value$setup_keys)
    }, logical(1L)
  ))
  fastkpc_full_cuda_phase7_require(
    clean_parts, "Phase 7 setup shard evidence is malformed"
  )
  identity_json <- vapply(parts, function(value) {
    fastkpc_full_cuda_phase35_canonical_json(value$execution_identity)
  }, character(1L))
  current_identity <- fastkpc_full_cuda_phase7_execution_identity(
    catalog, "native-setup-oracle"
  )
  fastkpc_full_cuda_phase7_require(
    length(unique(identity_json)) == 1L && identical(
      identity_json[[1L]],
      fastkpc_full_cuda_phase35_canonical_json(current_identity)
    ),
    "Phase 7 setup shard execution identities differ"
  )
  rows <- do.call(rbind, lapply(parts, `[[`, "rows"))
  scope <- fastkpc_full_cuda_phase7_setup_scope(catalog)
  expected_keys <- as.character(scope$setup_rows$prepared_s_key_sha256)
  index <- match(expected_keys, rows$prepared_s_key_sha256)
  fastkpc_full_cuda_phase7_require(
    nrow(rows) == 8634L && !anyNA(index) &&
      !anyDuplicated(rows$prepared_s_key_sha256),
    "Phase 7 setup shards do not exactly cover the canonical corpus"
  )
  rows <- rows[index, , drop = FALSE]
  rownames(rows) <- NULL
  error_fields <- c(
    "x_max_absolute_error",
    "model_space_projector_max_absolute_error",
    "constraint_max_absolute_error",
    "penalty_operator_max_absolute_error",
    "shift_max_absolute_error", "penalty_scale_max_absolute_error",
    "qr_q_max_absolute_error", "qr_packed_max_absolute_error",
    "qr_tau_max_absolute_error", "qr_r_max_absolute_error",
    "mroot_max_absolute_error", "penalty_matrix_max_absolute_error",
    "initial_sp_max_absolute_error"
  )
  mismatch_fields <- c(
    "rank_mismatch_count", "null_space_mismatch_count",
    "basis_dimension_mismatch_count", "qr_pivot_mismatch_count"
  )
  counts_by_s <- tabulate(rows$S_size, nbins = 7L)
  pass <- identical(counts_by_s, c(48L, 1126L, 4064L, 2152L,
                                    955L, 245L, 44L)) &&
    all(vapply(error_fields, function(field) {
      all(is.finite(rows[[field]])) && max(rows[[field]]) == 0
    }, logical(1L))) &&
    all(vapply(mismatch_fields, function(field) {
      sum(rows[[field]]) == 0L
    }, logical(1L))) &&
    all(rows$fixed_sp_operator_equivalent) &&
    sum(rows$native_setup_count) == 8634L &&
    sum(rows$oracle_setup_count) == 8634L &&
    sum(rows$legacy_mgcv_setup_count) == 8634L &&
    sum(rows$r_callback_count) == 0L &&
    sum(rows$unsupported_count) == 0L
  summary <- list(
    schema_version = "full-cuda-ci-native-setup-corpus-summary-v1",
    setup_count = nrow(rows),
    counts_by_S_size = as.integer(counts_by_s),
    native_setup_count = sum(rows$native_setup_count),
    oracle_setup_count = sum(rows$oracle_setup_count),
    legacy_mgcv_setup_count = sum(rows$legacy_mgcv_setup_count),
    r_callback_count = sum(rows$r_callback_count),
    unsupported_count = sum(rows$unsupported_count),
    maximum_error_by_field = setNames(as.list(vapply(
      error_fields, function(field) max(rows[[field]]), numeric(1L)
    )), error_fields),
    mismatch_count_by_field = setNames(as.list(vapply(
      mismatch_fields, function(field) sum(rows[[field]]), integer(1L)
    )), mismatch_fields),
    fixed_sp_operator_equivalence_count =
      sum(rows$fixed_sp_operator_equivalent),
    pass = pass
  )
  fastkpc_full_cuda_phase7_require(
    pass, "Phase 7 full setup corpus gate failed"
  )
  list(
    schema_version = "full-cuda-ci-native-setup-corpus-evidence-v1",
    summary = summary,
    execution_identity = current_identity,
    shard_file_sha256 = setNames(as.list(vapply(
      paths, fastkpc_full_cuda_census_file_hash, character(1L)
    )), basename(paths)),
    rows = rows
  )
}
