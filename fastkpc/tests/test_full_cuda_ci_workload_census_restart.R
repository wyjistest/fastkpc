fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_error <- function(expression, pattern, message) {
  error <- tryCatch({ force(expression); NULL }, error = identity)
  assert_true(inherits(error, "error") &&
                grepl(pattern, conditionMessage(error), fixed = TRUE),
              message)
}

source("fastkpc/R/full_cuda_ci_workload_census.R")

runtime_identity <- fastkpc_full_cuda_census_runtime_identity()
assert_true(all(vapply(runtime_identity[c(
  "source_commit", "R_version", "mgcv_version", "BLAS_identity",
  "LAPACK_identity"
)], function(value) length(value) == 1L && !is.na(value) && nzchar(value),
logical(1L))) &&
              length(runtime_identity$BLAS_thread_count) == 1L,
            "runtime identity must contain one scalar value for every field")

named_frame <- data.frame(
  convergence_fields = I(list(list(
    converged = list(source = "fit$converged", value = TRUE)
  )))
)
renamed_frame <- named_frame
names(renamed_frame$convergence_fields[[1L]])[[1L]] <- "renamed"
assert_true(
  !identical(fastkpc_full_cuda_census_frame_hash(named_frame),
             fastkpc_full_cuda_census_frame_hash(renamed_frame)),
  "frame authentication must preserve semantic nested names"
)

hex64 <- function(value) sprintf("%064d", as.integer(value))
key_values <- c(12L, 1L, 9L, 3L, 11L, 5L, 7L, 2L, 10L, 4L, 8L, 6L)
group_index <- rep(seq_len(6L), each = 2L)
requests <- data.frame(
  residual_key_sha256 = hex64(key_values),
  target = rep(1:2, 6L),
  S_key = as.character(group_index),
  S_size = 1L,
  formula_class = "full-smooth",
  same_S_group_id = hex64(100L + group_index),
  stringsAsFactors = FALSE
)
shard_count <- 3L
assigned <- fastkpc_full_cuda_census_assign_shards(requests, shard_count)
sorted_keys <- sort(requests$residual_key_sha256, method = "radix")
assert_true(identical(assigned$residual_key_sha256, sorted_keys) &&
              identical(assigned$sorted_rank, seq_len(12L)) &&
              identical(assigned$shard_id, rep(0:2, 4L)),
            "shards must use lexicographic sorted-rank modulo assignment")

duplicate_requests <- requests
duplicate_requests$residual_key_sha256[[2L]] <-
  duplicate_requests$residual_key_sha256[[1L]]
assert_error(
  fastkpc_full_cuda_census_assign_shards(duplicate_requests, shard_count),
  "duplicate residual key",
  "shard assignment must reject duplicate canonical keys"
)

corpus_payload <- paste0(paste(sorted_keys, collapse = "\n"), "\n")
risk_config <- fastkpc_full_cuda_census_risk_config()
context <- list(
  canonical_key_corpus_hash =
    fastkpc_full_cuda_census_hash_utf8(corpus_payload),
  canonical_logical_census_hash = strrep("d", 64L),
  dataset_sha256 = strrep("a", 64L),
  oracle_input_bundle_sha256 = strrep("b", 64L),
  source_commit = strrep("c", 40L),
  R_version = "R fixture 4.4.1",
  mgcv_version = "1.9-1-fixture",
  BLAS_identity = "fixture-blas",
  LAPACK_identity = "fixture-lapack",
  BLAS_thread_count = 1L,
  formula_semantics_version = "kpcalg_regrXonS_v1",
  mgcv_semantics_version = "legacy-mgcv-gam-default-selection-v1",
  risk_threshold_config_hash =
    fastkpc_full_cuda_census_metadata_hash(risk_config),
  metadata_schema_version = fastkpc_full_cuda_census_metadata_schema_version(),
  data = matrix(0, nrow = 2L, ncol = 2L),
  risk_config = risk_config,
  logical_tests = data.frame(
    logical_sequence_id = integer(),
    absolute_log_distance_from_alpha = numeric(),
    stringsAsFactors = FALSE
  )
)

invalid_context <- context
invalid_context$BLAS_thread_count <- NA_integer_
assert_error(
  fastkpc_full_cuda_census_shard_manifest(assigned, 0L, invalid_context),
  "invalid shard context field",
  "manifest must reject an unresumable unknown BLAS thread count"
)

sparse_assigned <- fastkpc_full_cuda_census_assign_shards(
  requests[1:2, , drop = FALSE], 4L
)
sparse_context <- context
sparse_context$canonical_key_corpus_hash <-
  fastkpc_full_cuda_census_key_set_hash(
    sparse_assigned$residual_key_sha256
  )
sparse_manifest <- fastkpc_full_cuda_census_shard_manifest(
  sparse_assigned, 3L, sparse_context
)
assert_true(sparse_manifest$shard_count == 4L &&
              sparse_manifest$expected_key_count_for_shard == 0L &&
              identical(sparse_manifest$expected_key_hash_for_shard,
                        fastkpc_full_cuda_census_key_set_hash(character())),
            "manifest must retain explicit empty trailing shards")

for (shard_id in 0:(shard_count - 1L)) {
  manifest <- fastkpc_full_cuda_census_shard_manifest(
    assigned, shard_id, context
  )
  shard_keys <- assigned$residual_key_sha256[
    assigned$shard_id == shard_id
  ]
  expected_hash <- fastkpc_full_cuda_census_hash_utf8(
    paste0(paste(shard_keys, collapse = "\n"), "\n")
  )
  assert_true(manifest$expected_key_count_for_shard == 4L &&
                identical(manifest$expected_key_hash_for_shard,
                          expected_hash) &&
                identical(manifest$shard_id, as.integer(shard_id)) &&
                identical(manifest$shard_count, shard_count),
              "shard manifest must freeze expected key count and hash")
}

fixture_setup <- function(request_row) {
  group_id <- request_row$same_S_group_id[[1L]]
  setup_fingerprint <- fastkpc_full_cuda_census_hash_utf8(
    paste0("setup:", group_id)
  )
  data.frame(
    same_S_group_id = group_id,
    S_key = request_row$S_key[[1L]],
    S_size = 1L,
    formula_class = "full-smooth",
    representative_residual_key_sha256 =
      request_row$residual_key_sha256[[1L]],
    formula_semantics_version = "kpcalg_regrXonS_v1",
    model_matrix_nrow = 2L,
    model_matrix_ncol = 2L,
    model_matrix_hash = fastkpc_full_cuda_census_hash_utf8(
      paste0("model:", group_id)
    ),
    model_matrix_rank = 2L,
    model_matrix_condition = 1,
    penalty_count = 1L,
    penalty_block_dimensions = I(list("1x1")),
    penalty_ranks = I(list(1L)),
    penalty_offsets = I(list(2L)),
    penalty_hashes = I(list(fastkpc_full_cuda_census_hash_utf8(
      paste0("penalty:", group_id)
    ))),
    penalty_nullity = 1L,
    constraint_dimensions = I(list(c(0L, 2L))),
    constraint_rank = 0L,
    constraint_nullspace_dimension = 2L,
    constraint_hash = fastkpc_full_cuda_census_hash_utf8(
      paste0("constraint:", group_id)
    ),
    H_dimensions = I(list(integer())),
    H_hash = "NONE",
    weights_policy = "none",
    offset_policy = "none",
    smooth_classes = I(list("fixture.smooth")),
    basis_dimensions = I(list(2L)),
    conditioning_rank = 1L,
    conditioning_condition = 1,
    near_constant_conditioning_count = 0L,
    setup_fingerprint = setup_fingerprint,
    mgcv_version = "1.9-1-fixture",
    R_version = "R fixture 4.4.1",
    stringsAsFactors = FALSE
  )
}

fixture_target <- function(request_row, setup) {
  key <- request_row$residual_key_sha256[[1L]]
  data.frame(
    residual_key_sha256 = key,
    same_S_group_id = request_row$same_S_group_id[[1L]],
    setup_fingerprint = setup$setup_fingerprint[[1L]],
    shard_id = as.integer(request_row$shard_id[[1L]]),
    target = as.integer(request_row$target[[1L]]),
    fit_status = "success",
    fit_error = "NONE",
    fit_time_ms = 1,
    formula = "x1 ~ s(x2)",
    method = "GCV.Cp",
    optimizer = "mgcv-default",
    family = "gaussian",
    link = "identity",
    selected_sp = I(list(1)),
    selected_sp_names = I(list("s(x2)")),
    selected_sp_hash = fastkpc_full_cuda_census_hash_utf8(
      paste0("sp:", key)
    ),
    GCV_Cp_score = 1,
    EDF = 1,
    convergence_fields = I(list(list(
      converged = list(source = "fit$converged", value = TRUE)
    ))),
    warning_classes = I(list(list())),
    warning_messages = I(list(character())),
    coefficient_rank = 2L,
    coefficient_all_finite = TRUE,
    fitted_all_finite = TRUE,
    residual_all_finite = TRUE,
    penalized_system_condition_at_selected_sp = 1,
    target_sd = 1,
    target_near_constant = FALSE,
    coefficient_hash = fastkpc_full_cuda_census_hash_utf8(
      paste0("coef:", key)
    ),
    fitted_hash = fastkpc_full_cuda_census_hash_utf8(
      paste0("fitted:", key)
    ),
    residual_hash = fastkpc_full_cuda_census_hash_utf8(
      paste0("residual:", key)
    ),
    target_fit_fingerprint = fastkpc_full_cuda_census_hash_utf8(
      paste0("target:", key)
    ),
    stringsAsFactors = FALSE
  )
}

fit_calls <- new.env(parent = emptyenv())
fit_calls$count <- 0L
fixture_fit <- function(data, request_row, risk_config) {
  fit_calls$count <- fit_calls$count + 1L
  setup <- fixture_setup(request_row)
  target <- fixture_target(request_row, setup)
  risk <- data.frame(
    case_type = "target_key",
    residual_key_sha256 = request_row$residual_key_sha256[[1L]],
    logical_sequence_id = NA_integer_,
    same_S_group_id = request_row$same_S_group_id[[1L]],
    high_condition = FALSE,
    rank_deficient = FALSE,
    near_constant_target = FALSE,
    near_constant_conditioner = FALSE,
    multi_penalty = FALSE,
    near_alpha = FALSE,
    mgcv_warning = FALSE,
    mgcv_nonconverged = FALSE,
    nonfinite_metadata = FALSE,
    condition_bucket = "finite_lt_1e4",
    near_alpha_bucket = NA_character_,
    stringsAsFactors = FALSE
  )
  list(setup_observation = setup, target_fit = target, risk_cases = risk)
}

wrong_risk_fit <- function(data, request_row, risk_config) {
  value <- fixture_fit(data, request_row, risk_config)
  value$risk_cases$residual_key_sha256[[1L]] <- hex64(999L)
  value
}
assert_error(
  fastkpc_full_cuda_census_run_shard(
    assigned, 0L, context, tempfile("full-cuda-ci-wrong-risk-key-"),
    fit_fun = wrong_risk_fit
  ),
  "fit_fun returned the wrong risk residual key",
  "shard execution must reject a risk row joined to another key"
)

missing_setup_fit <- function(data, request_row, risk_config) {
  value <- fixture_fit(data, request_row, risk_config)
  value$setup_observation <- NULL
  value
}
assert_error(
  fastkpc_full_cuda_census_run_shard(
    assigned, 0L, context, tempfile("full-cuda-ci-missing-setup-"),
    fit_fun = missing_setup_fit
  ),
  "successful fit is missing its setup observation",
  "successful target rows must retain one setup observation"
)

wrong_setup_fit <- function(data, request_row, risk_config) {
  value <- fixture_fit(data, request_row, risk_config)
  value$setup_observation$setup_fingerprint[[1L]] <- hex64(998L)
  value
}
assert_error(
  fastkpc_full_cuda_census_run_shard(
    assigned, 0L, context, tempfile("full-cuda-ci-wrong-setup-"),
    fit_fun = wrong_setup_fit
  ),
  "fit_fun target/setup lineage mismatch",
  "target rows must reference the setup observation from the same key"
)
fit_calls$count <- 0L

sparse_calls_before <- fit_calls$count
sparse_empty <- fastkpc_full_cuda_census_run_shard(
  assigned_requests = sparse_assigned,
  shard_id = 3L,
  context = sparse_context,
  output_dir = tempfile("full-cuda-ci-empty-shard-"),
  fit_fun = fixture_fit
)
assert_true(sparse_empty$status == "written" &&
              fit_calls$count == sparse_calls_before &&
              length(sparse_empty$payload$request_keys) == 0L &&
              nrow(sparse_empty$payload$target_fits) == 0L,
            "zero-key shards must complete without invoking fit_fun")

wrong_initial_context <- context
wrong_initial_context$canonical_key_corpus_hash <- strrep("f", 64L)
assert_error(
  fastkpc_full_cuda_census_run_shard(
    assigned_requests = assigned,
    shard_id = 0L,
    context = wrong_initial_context,
    output_dir = tempfile("full-cuda-ci-wrong-corpus-"),
    fit_fun = fixture_fit
  ),
  "canonical key corpus hash mismatch",
  "initial shard execution must reject a wrong corpus hash before fitting"
)
wrong_initial_risk <- context
wrong_initial_risk$risk_threshold_config_hash <- strrep("e", 64L)
assert_error(
  fastkpc_full_cuda_census_run_shard(
    assigned_requests = assigned,
    shard_id = 0L,
    context = wrong_initial_risk,
    output_dir = tempfile("full-cuda-ci-wrong-risk-"),
    fit_fun = fixture_fit
  ),
  "risk threshold config hash mismatch",
  "initial shard execution must derive the risk-config hash"
)

output_dir <- tempfile("full-cuda-ci-restart-")
dir.create(output_dir, recursive = TRUE)
first <- fastkpc_full_cuda_census_run_shard(
  assigned_requests = assigned,
  shard_id = 0L,
  context = context,
  output_dir = output_dir,
  fit_fun = fixture_fit
)
assert_true(first$status == "written" && fit_calls$count == 4L &&
              file.exists(first$paths$rds) &&
              file.exists(first$paths$summary_json) &&
              length(list.files(output_dir, pattern = "\\.tmp")) == 0L,
            "shard writer must atomically publish RDS then completion JSON")

first_summary <- jsonlite::read_json(
  first$paths$summary_json, simplifyVector = TRUE
)
assert_true(all(c(
  "setup_observations_hash", "target_fits_hash", "target_risks_hash",
  "payload_hash"
) %in% names(first_summary)),
"completed shard summary must authenticate every persisted metadata table")

tampered_payload <- readRDS(first$paths$rds)
tampered_payload$target_fits$GCV_Cp_score[[1L]] <-
  tampered_payload$target_fits$GCV_Cp_score[[1L]] + 1
saveRDS(tampered_payload, first$paths$rds, version = 2)
assert_error(
  fastkpc_full_cuda_census_run_shard(
    assigned_requests = assigned,
    shard_id = 0L,
    context = context,
    output_dir = output_dir,
    fit_fun = fixture_fit
  ),
  "payload hash mismatch",
  "resume must reject finite persisted metadata corruption"
)
saveRDS(first$payload, first$paths$rds, version = 2)

nested_name_tamper <- readRDS(first$paths$rds)
names(nested_name_tamper$target_fits$convergence_fields[[1L]])[[1L]] <-
  "renamed"
saveRDS(nested_name_tamper, first$paths$rds, version = 2)
assert_error(
  fastkpc_full_cuda_census_run_shard(
    assigned_requests = assigned,
    shard_id = 0L,
    context = context,
    output_dir = output_dir,
    fit_fun = fixture_fit
  ),
  "payload hash mismatch",
  "resume must reject semantic nested-name corruption"
)
saveRDS(first$payload, first$paths$rds, version = 2)

reused <- fastkpc_full_cuda_census_run_shard(
  assigned_requests = assigned,
  shard_id = 0L,
  context = context,
  output_dir = output_dir,
  fit_fun = fixture_fit
)
assert_true(reused$status == "reused" && fit_calls$count == 4L,
            "a completed exact-manifest shard must be reused without fitting")

identity_fields <- c(
  "canonical_key_corpus_hash", "canonical_logical_census_hash",
  "risk_threshold_config_hash", "R_version",
  "mgcv_version", "source_commit", "BLAS_identity", "LAPACK_identity",
  "BLAS_thread_count"
)
for (field in identity_fields) {
  wrong <- context
  wrong[[field]] <- if (field == "BLAS_thread_count") {
    2L
  } else if (field == "canonical_key_corpus_hash") {
    strrep("d", 64L)
  } else if (field == "canonical_logical_census_hash") {
    strrep("c", 64L)
  } else if (field == "risk_threshold_config_hash") {
    strrep("e", 64L)
  } else if (field == "source_commit") {
    strrep("d", 40L)
  } else {
    paste0(context[[field]], "-wrong")
  }
  assert_error(
    fastkpc_full_cuda_census_run_shard(
      assigned_requests = assigned,
      shard_id = 0L,
      context = wrong,
      output_dir = output_dir,
      fit_fun = fixture_fit
    ),
    if (field == "canonical_key_corpus_hash") {
      "canonical key corpus hash mismatch"
    } else if (field == "risk_threshold_config_hash") {
      "risk threshold config hash mismatch"
    } else {
      "shard manifest mismatch"
    },
    paste("resume must reject changed identity field", field)
  )
}

interrupt_dir <- tempfile("full-cuda-ci-interrupt-")
interrupt_count <- 0L
interrupt_fit <- function(data, request_row, risk_config) {
  interrupt_count <<- interrupt_count + 1L
  if (interrupt_count == 2L) stop("injected interruption", call. = FALSE)
  fixture_fit(data, request_row, risk_config)
}
assert_error(
  fastkpc_full_cuda_census_run_shard(
    assigned_requests = assigned,
    shard_id = 1L,
    context = context,
    output_dir = interrupt_dir,
    fit_fun = interrupt_fit
  ),
  "injected interruption",
  "injected shard interruption must propagate"
)
interrupted_paths <- fastkpc_full_cuda_census_shard_paths(interrupt_dir, 1L)
assert_true(!file.exists(interrupted_paths$rds) &&
              !file.exists(interrupted_paths$summary_json),
            "interrupted execution must not leave a completed partial shard")

for (shard_id in c(2L, 1L)) {
  fastkpc_full_cuda_census_run_shard(
    assigned_requests = assigned,
    shard_id = shard_id,
    context = context,
    output_dir = output_dir,
    fit_fun = fixture_fit
  )
}

merged <- fastkpc_full_cuda_census_merge_shards(
  requests = requests,
  shard_count = shard_count,
  context = context,
  shard_dir = output_dir
)
assert_true(identical(merged$target_fit_metadata$residual_key_sha256,
                      sorted_keys) &&
              nrow(merged$target_fit_metadata) == 12L &&
              nrow(merged$same_s_setup_metadata) == 6L &&
              nrow(merged$target_risk_metadata) == 12L &&
              all(merged$field_coverage$coverage_ratio[
                merged$field_coverage$required
              ] == 1),
            "merge must retain exact target, setup, and risk lineage")

valid_gate <- fastkpc_full_cuda_census_metadata_gate(merged, requests)
assert_true(isTRUE(valid_gate$pass) &&
              isTRUE(valid_gate$exact_target_risk_key_set) &&
              isTRUE(valid_gate$exact_setup_observation_key_set) &&
              isTRUE(valid_gate$exact_target_setup_lineage) &&
              isTRUE(valid_gate$exact_authenticated_metadata) &&
              isTRUE(valid_gate$exact_risk_semantics),
            "metadata gate must expose exact per-request lineage")

finite_metadata_tamper <- merged
finite_metadata_tamper$target_fit_metadata$GCV_Cp_score[[1L]] <-
  finite_metadata_tamper$target_fit_metadata$GCV_Cp_score[[1L]] + 1
finite_metadata_tamper_gate <- fastkpc_full_cuda_census_metadata_gate(
  finite_metadata_tamper, requests
)
assert_true(!isTRUE(finite_metadata_tamper_gate$pass) &&
              !isTRUE(finite_metadata_tamper_gate$exact_authenticated_metadata),
            "metadata gate must reject in-memory finite metadata drift")

nested_metadata_tamper <- merged
names(nested_metadata_tamper$target_fit_metadata$
        convergence_fields[[1L]])[[1L]] <- "renamed"
nested_metadata_tamper_gate <- fastkpc_full_cuda_census_metadata_gate(
  nested_metadata_tamper, requests
)
assert_true(!isTRUE(nested_metadata_tamper_gate$pass) &&
              !isTRUE(
                nested_metadata_tamper_gate$exact_authenticated_metadata
              ),
            "metadata gate must reject in-memory nested-name drift")

near_constant_semantic_tamper <- merged
near_constant_semantic_tamper$target_fit_metadata$target_sd[[1L]] <- 0
near_constant_semantic_tamper$target_fit_metadata$
  target_near_constant[[1L]] <- FALSE
near_constant_semantic_tamper$target_risk_metadata$
  near_constant_target[[1L]] <- FALSE
near_constant_semantic_tamper$risk_cases <-
  fastkpc_full_cuda_census_risk_cases(
    near_constant_semantic_tamper$target_risk_metadata,
    context$logical_tests
  )
near_constant_semantic_tamper$authenticated_metadata_hashes <-
  fastkpc_full_cuda_census_authenticated_metadata_hashes(
    near_constant_semantic_tamper
  )
near_constant_semantic_tamper_gate <-
  fastkpc_full_cuda_census_metadata_gate(
    near_constant_semantic_tamper, requests
  )
assert_true(
  !isTRUE(near_constant_semantic_tamper_gate$pass) &&
    isTRUE(near_constant_semantic_tamper_gate$exact_authenticated_metadata) &&
    !isTRUE(
      near_constant_semantic_tamper_gate$exact_target_near_constant_semantics
    ) &&
    near_constant_semantic_tamper_gate$
      target_near_constant_mismatch_count > 0L,
  "metadata gate must derive near-constant target status from target_sd"
)

risk_semantic_tamper <- merged
risk_semantic_tamper$target_risk_metadata$high_condition[[1L]] <- TRUE
risk_semantic_tamper$target_risk_metadata$rank_deficient[[1L]] <- TRUE
risk_semantic_tamper$target_risk_metadata$near_constant_target[[1L]] <- TRUE
risk_semantic_tamper$target_risk_metadata$near_constant_conditioner[[1L]] <- TRUE
risk_semantic_tamper$target_risk_metadata$multi_penalty[[1L]] <- TRUE
risk_semantic_tamper_gate <- fastkpc_full_cuda_census_metadata_gate(
  risk_semantic_tamper, requests
)
assert_true(!isTRUE(risk_semantic_tamper_gate$pass) &&
              !isTRUE(risk_semantic_tamper_gate$exact_risk_semantics) &&
              risk_semantic_tamper_gate$risk_semantic_mismatch_count > 0L,
            "metadata gate must recompute every target risk classification")

missing_risk <- merged
missing_risk$target_risk_metadata <-
  missing_risk$target_risk_metadata[-1L, , drop = FALSE]
missing_risk_gate <- fastkpc_full_cuda_census_metadata_gate(
  missing_risk, requests
)
assert_true(!isTRUE(missing_risk_gate$pass) &&
              !isTRUE(missing_risk_gate$exact_target_risk_key_set),
            "metadata gate must reject an omitted target risk row")

wrong_lineage <- merged
wrong_lineage$target_fit_metadata$setup_fingerprint[[1L]] <- hex64(997L)
wrong_lineage_gate <- fastkpc_full_cuda_census_metadata_gate(
  wrong_lineage, requests
)
assert_true(!isTRUE(wrong_lineage_gate$pass) &&
              !isTRUE(wrong_lineage_gate$exact_target_setup_lineage),
            "metadata gate must reject a target joined to another setup")

nonfinite_fit_time <- merged
nonfinite_fit_time$target_fit_metadata$fit_time_ms[[1L]] <- Inf
nonfinite_fit_time_gate <- fastkpc_full_cuda_census_metadata_gate(
  nonfinite_fit_time, requests
)
assert_true(!isTRUE(nonfinite_fit_time_gate$pass) &&
              nonfinite_fit_time_gate$unclassified_nonfinite_count == 1L,
            "non-finite fit timing must be classified or fail closed")

nonfinite_output <- merged
nonfinite_output$target_fit_metadata$residual_all_finite[[1L]] <- FALSE
nonfinite_output_gate <- fastkpc_full_cuda_census_metadata_gate(
  nonfinite_output, requests
)
assert_true(!isTRUE(nonfinite_output_gate$pass) &&
              nonfinite_output_gate$unclassified_nonfinite_count == 1L,
            "non-finite residual output evidence must fail closed")

nonfinite_setup <- merged
nonfinite_setup$same_s_setup_metadata$model_matrix_condition[[1L]] <- Inf
nonfinite_setup_gate <- fastkpc_full_cuda_census_metadata_gate(
  nonfinite_setup, requests
)
assert_true(!isTRUE(nonfinite_setup_gate$pass) &&
              nonfinite_setup_gate$unclassified_nonfinite_count > 0L,
            "non-finite setup diagnostics must be classified or fail closed")

error_requests <- requests[1:2, , drop = FALSE]
error_assigned <- fastkpc_full_cuda_census_assign_shards(error_requests, 1L)
error_context <- context
error_context$canonical_key_corpus_hash <-
  fastkpc_full_cuda_census_key_set_hash(
    error_assigned$residual_key_sha256
  )
error_fit <- function(data, request_row, risk_config) {
  target <- fastkpc_full_cuda_census_target_error_row(
    request_row, simpleError("injected fit failure")
  )
  risk <- data.frame(
    case_type = "target_key",
    residual_key_sha256 = request_row$residual_key_sha256[[1L]],
    logical_sequence_id = NA_integer_,
    same_S_group_id = request_row$same_S_group_id[[1L]],
    high_condition = FALSE,
    rank_deficient = FALSE,
    near_constant_target = TRUE,
    near_constant_conditioner = FALSE,
    multi_penalty = FALSE,
    near_alpha = FALSE,
    mgcv_warning = FALSE,
    mgcv_nonconverged = FALSE,
    nonfinite_metadata = TRUE,
    condition_bucket = "nonfinite_unknown",
    near_alpha_bucket = NA_character_,
    stringsAsFactors = FALSE
  )
  list(setup_observation = NULL, target_fit = target, risk_cases = risk)
}
error_dir <- tempfile("full-cuda-ci-error-row-")
invisible(fastkpc_full_cuda_census_run_shard(
  error_assigned, 0L, error_context, error_dir, fit_fun = error_fit
))
error_merged <- fastkpc_full_cuda_census_merge_shards(
  error_requests, 1L, error_context, error_dir
)
assert_true(nrow(error_merged$target_fit_metadata) == 2L &&
              sum(error_merged$target_fit_metadata$fit_status == "error") ==
                2L &&
              "shard_id" %in% names(error_merged$target_fit_metadata) &&
              all(error_merged$target_fit_metadata$shard_id == 0L) &&
              any(error_merged$field_coverage$required &
                    error_merged$field_coverage$coverage_ratio < 1),
            "merge must retain error rows, shard lineage, and an incomplete gate")

summary_tamper_paths <- fastkpc_full_cuda_census_shard_paths(output_dir, 2L)
summary_tamper <- jsonlite::read_json(
  summary_tamper_paths$summary_json, simplifyVector = TRUE
)
summary_tamper$request_key_count <- summary_tamper$request_key_count + 1L
fastkpc_full_cuda_write_json(summary_tamper,
                             summary_tamper_paths$summary_json)
assert_error(
  fastkpc_full_cuda_census_merge_shards(
    requests, shard_count, context, output_dir
  ),
  "shard summary count mismatch",
  "merge must reject a completion summary with false counts"
)
summary_tamper$request_key_count <- summary_tamper$request_key_count - 1L
fastkpc_full_cuda_write_json(summary_tamper,
                             summary_tamper_paths$summary_json)

missing_paths <- fastkpc_full_cuda_census_shard_paths(output_dir, 2L)
missing_backup <- paste0(missing_paths$summary_json, ".backup")
assert_true(file.rename(missing_paths$summary_json, missing_backup),
            "test fixture must hide one shard summary")
assert_error(
  fastkpc_full_cuda_census_merge_shards(
    requests, shard_count, context, output_dir
  ),
  "missing shard",
  "merge must reject a missing completion summary"
)
assert_true(file.rename(missing_backup, missing_paths$summary_json),
            "test fixture must restore the missing shard summary")

shard_zero <- fastkpc_full_cuda_census_shard_paths(output_dir, 0L)
shard_one <- fastkpc_full_cuda_census_shard_paths(output_dir, 1L)
one_rds_backup <- paste0(shard_one$rds, ".backup")
one_json_backup <- paste0(shard_one$summary_json, ".backup")
assert_true(file.copy(shard_one$rds, one_rds_backup, overwrite = TRUE) &&
              file.copy(shard_one$summary_json, one_json_backup,
                        overwrite = TRUE) &&
              file.copy(shard_zero$rds, shard_one$rds, overwrite = TRUE) &&
              file.copy(shard_zero$summary_json, shard_one$summary_json,
                        overwrite = TRUE),
            "test fixture must install a duplicate declared shard")
assert_error(
  fastkpc_full_cuda_census_merge_shards(
    requests, shard_count, context, output_dir
  ),
  "duplicate shard",
  "merge must reject two files declaring the same shard id"
)
assert_true(file.copy(one_rds_backup, shard_one$rds, overwrite = TRUE) &&
              file.copy(one_json_backup, shard_one$summary_json,
                        overwrite = TRUE),
            "test fixture must restore shard one")

duplicate_payload <- readRDS(shard_one$rds)
duplicate_payload$target_fits$residual_key_sha256[[2L]] <-
  duplicate_payload$target_fits$residual_key_sha256[[1L]]
saveRDS(duplicate_payload, shard_one$rds, version = 2)
assert_error(
  fastkpc_full_cuda_census_merge_shards(
    requests, shard_count, context, output_dir
  ),
  "duplicate residual key",
  "merge must reject duplicate target-fit keys"
)
assert_true(file.copy(one_rds_backup, shard_one$rds, overwrite = TRUE),
            "test fixture must restore the duplicate-key shard")

wrong_risk_payload <- readRDS(shard_one$rds)
wrong_risk_payload$target_risks$residual_key_sha256[[1L]] <- hex64(996L)
saveRDS(wrong_risk_payload, shard_one$rds, version = 2)
assert_error(
  fastkpc_full_cuda_census_merge_shards(
    requests, shard_count, context, output_dir
  ),
  "shard risk key set mismatch",
  "merge must reject a persisted risk row joined to another key"
)
assert_true(file.copy(one_rds_backup, shard_one$rds, overwrite = TRUE),
            "test fixture must restore the wrong-risk shard")

wrong_setup_payload <- readRDS(shard_one$rds)
wrong_setup_payload$setup_observations$setup_fingerprint[[1L]] <- hex64(995L)
saveRDS(wrong_setup_payload, shard_one$rds, version = 2)
assert_error(
  fastkpc_full_cuda_census_merge_shards(
    requests, shard_count, context, output_dir
  ),
  "shard target/setup lineage mismatch",
  "merge must reject persisted target/setup fingerprint drift"
)
assert_true(file.copy(one_rds_backup, shard_one$rds, overwrite = TRUE),
            "test fixture must restore the wrong-setup shard")
unlink(c(one_rds_backup, one_json_backup))

cat("PASS full CUDA CI census restart qualification\n")
