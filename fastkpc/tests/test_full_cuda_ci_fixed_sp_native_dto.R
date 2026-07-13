source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")

started_at <- proc.time()[["elapsed"]]

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_error <- function(expression, message, pattern = NULL) {
  error <- tryCatch({ force(expression); NULL }, error = identity)
  assert_true(inherits(error, "error"), message)
  if (!is.null(pattern)) {
    assert_true(
      grepl(pattern, conditionMessage(error), fixed = TRUE),
      paste0(message, ": unexpected error: ", conditionMessage(error))
    )
  }
  invisible(error)
}

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
  phase0_dir, phase1_dir, phase2_dir, data_path, require_full = TRUE
)
iteration <- fastkpc_full_cuda_fixed_sp_scope(catalog, "iteration")
setup_index <- which(iteration$setup_rows$penalty_count > 1L)[[1L]]
selected_scope <- iteration
selected_scope$setup_rows <- iteration$setup_rows[
  setup_index, , drop = FALSE
]
selected_key <- selected_scope$setup_rows$prepared_s_key_sha256[[1L]]
selected_scope$target_rows <- iteration$target_rows[
  iteration$target_rows$prepared_s_key_sha256 == selected_key,
  , drop = FALSE
]
selected_rank <- match(
  selected_key, catalog$setup_index$prepared_s_key_sha256
)
selected_scope$shard_ids <- as.integer(
  (selected_rank - 1L) %% catalog$catalog_contract$shard_count
)
batches <- fastkpc_full_cuda_fixed_sp_batches(catalog, selected_scope)
assert_true(length(batches) == 1L, "focused scope materializes one batch")
batch <- batches[[1L]]

dto <- fastkpc_full_cuda_fixed_sp_native_dto(batch$setup)
native_batch <- fastkpc_full_cuda_fixed_sp_native_batch(batch, dto)

expected_dto_fields <- c(
  "schema_version", "dataset_sha256", "prepared_s_key_sha256",
  "same_S_group_id", "phase1_setup_fingerprint", "provider_fingerprint",
  "semantic_fingerprint", "representation_fingerprint",
  "prepared_s_setup_schema_version", "native_dto_schema_version",
  "data_p", "n", "coefficient_dim", "null_dim", "penalty_count", "X",
  "constraint_mode", "constraint_nullspace", "gram_matrix",
  "nullspace_gram_matrix", "penalty_blocks",
  "penalty_offsets_zero_based", "penalty_ranks",
  "penalty_sp_indices_zero_based", "penalty_sp_labels", "H",
  "weights_policy", "offset_policy"
)
assert_true(identical(names(dto), expected_dto_fields),
            "native DTO exact schema")
assert_true(identical(
  names(native_batch),
  c("Y", "SP", "planned_route", "target_keys", "target_count")
), "native batch exact schema")
assert_true(identical(
  dto$schema_version, "full-cuda-ci-prepared-s-native-dto-v1"
), "native DTO schema version")
assert_true(identical(
  dto$prepared_s_setup_schema_version, batch$setup$schema_version
), "native DTO Prepared-S setup schema identity")
assert_true(identical(
  dto$native_dto_schema_version,
  fastkpc_full_cuda_fixed_sp_contract()$native_dto_schema_version
), "native DTO Task 1 schema identity")
assert_true(identical(dto$data_p, 48L) && identical(
  dto$data_p, fastkpc_full_cuda_canonical_contract()$p
), "native DTO canonical dataset width identity")
assert_true(identical(dto$prepared_s_key_sha256, selected_key),
            "native DTO PreparedSKey")
assert_true(identical(dto$n, as.integer(nrow(batch$setup$X))) &&
              identical(dto$coefficient_dim,
                        as.integer(ncol(batch$setup$X))) &&
              identical(dto$null_dim, as.integer(
                batch$setup$constraint_nullspace_dimension
              )), "native DTO dimensions")
assert_true(dto$penalty_count > 1L && identical(
  dto$penalty_count, as.integer(length(batch$setup$penalty_blocks))
), "canonical multi-penalty DTO")
assert_true(typeof(dto$X) == "double" && is.matrix(dto$X) &&
              all(vapply(dto$penalty_blocks, function(block) {
                typeof(block) == "double" && is.matrix(block)
              }, logical(1L))), "DTO matrices remain double matrices")
assert_true(identical(
  dto$penalty_sp_indices_zero_based,
  seq.int(0L, dto$penalty_count - 1L)
), "v1 penalty-to-SP mapping is zero-based identity")
assert_true(identical(native_batch$target_count,
                      as.integer(ncol(batch$Y))) &&
              identical(dim(native_batch$SP),
                        c(dto$penalty_count, native_batch$target_count)),
            "native batch dimensions")
assert_true(is.null(native_batch$nullspace_rhs) &&
              is.null(native_batch$oracle_rhs) &&
              is.null(native_batch$oracle_nullspace_rhs),
            "production native batch carries no CPU RHS")
assert_true(identical(names(attributes(native_batch$Y)), "dim") &&
              identical(names(attributes(native_batch$SP)), "dim"),
            "native numeric payload matrices carry only dim")

reconstruct_penalty <- function(block, zero_offset, coefficient_dim) {
  out <- matrix(0, coefficient_dim, coefficient_dim)
  indices <- zero_offset + seq_len(nrow(block))
  out[indices, indices] <- block
  out
}
for (index in unique(c(1L, dto$penalty_count))) {
  expected <- matrix(0, dto$coefficient_dim, dto$coefficient_dim)
  phase2_indices <- batch$setup$penalty_offsets[[index]] +
    seq_len(nrow(batch$setup$penalty_blocks[[index]])) - 1L
  expected[phase2_indices, phase2_indices] <-
    batch$setup$penalty_blocks[[index]]
  actual <- reconstruct_penalty(
    dto$penalty_blocks[[index]],
    dto$penalty_offsets_zero_based[[index]], dto$coefficient_dim
  )
  assert_true(identical(actual, expected),
              "first/last zero-based penalty reconstruction")
}

zero_sp <- native_batch$SP
zero_sp[1L, 1L] <- 0
fastkpc_full_cuda_fixed_sp_validate_numeric_inputs(native_batch$Y, zero_sp)
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_numeric_inputs(
    native_batch$Y, replace(zero_sp, 1L, -1)
  ), "negative SP must fail closed", "numeric inputs are malformed"
)
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_numeric_inputs(
    native_batch$Y, replace(zero_sp, 1L, Inf)
  ), "nonfinite SP must fail closed", "numeric inputs are malformed"
)
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_numeric_inputs(
    replace(native_batch$Y, 1L, NA_real_), zero_sp
  ), "nonfinite Y must fail closed", "numeric inputs are malformed"
)
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_numeric_inputs(
    matrix(0L, nrow = 1L, ncol = 1L), zero_sp
  ), "integer Y matrix must fail closed", "numeric inputs are malformed"
)
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_numeric_inputs(
    native_batch$Y, matrix(0L, nrow = 1L, ncol = 1L)
  ), "integer SP matrix must fail closed", "numeric inputs are malformed"
)
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_numeric_inputs(
    matrix("0", nrow = 1L, ncol = 1L), zero_sp
  ), "character Y matrix must fail closed", "numeric inputs are malformed"
)
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_numeric_inputs(
    native_batch$Y, matrix("0", nrow = 1L, ncol = 1L)
  ), "character SP matrix must fail closed", "numeric inputs are malformed"
)
factor_y <- structure(factor("0"), dim = c(1L, 1L))
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_numeric_inputs(factor_y, zero_sp),
  "factor Y matrix must fail closed", "numeric inputs are malformed"
)
factor_sp <- structure(factor("0"), dim = c(1L, 1L))
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_numeric_inputs(
    native_batch$Y, factor_sp
  ), "factor SP matrix must fail closed", "numeric inputs are malformed"
)
attributed_y <- native_batch$Y
attr(attributed_y, "forged") <- TRUE
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_numeric_inputs(attributed_y, zero_sp),
  "arbitrary Y attributes must fail closed", "numeric inputs are malformed"
)
attributed_sp <- zero_sp
attr(attributed_sp, "forged") <- TRUE
assert_error(
  fastkpc_full_cuda_fixed_sp_validate_numeric_inputs(
    native_batch$Y, attributed_sp
  ), "arbitrary SP attributes must fail closed", "numeric inputs are malformed"
)

bad_dto <- dto[rev(seq_along(dto))]
assert_error(
  fastkpc_full_cuda_fixed_sp_native_batch(batch, bad_dto),
  "reordered DTO fields must fail closed", "native DTO is malformed"
)
bad_dto <- dto
bad_dto$prepared_s_setup_schema_version <- "forged-prepared-s-schema"
assert_error(
  fastkpc_full_cuda_fixed_sp_native_batch(batch, bad_dto),
  "Prepared-S schema identity tamper must fail closed", "lineage mismatch"
)
bad_dto <- dto
bad_dto$native_dto_schema_version <- "forged-native-dto-schema"
assert_error(
  fastkpc_full_cuda_fixed_sp_native_batch(batch, bad_dto),
  "native DTO schema identity tamper must fail closed", "lineage mismatch"
)
bad_dto <- dto
bad_dto$data_p <- dto$data_p + 1L
assert_error(
  fastkpc_full_cuda_fixed_sp_native_batch(batch, bad_dto),
  "native DTO data_p tamper must fail closed", "lineage mismatch"
)

bad <- batch
bad$Y[1L, 1L] <- bad$Y[1L, 1L] + 1e-6
assert_error(
  fastkpc_full_cuda_fixed_sp_native_batch(bad, dto),
  "modified Y must fail closed", "Y hash mismatch"
)
bad <- batch
bad$SP[1L, 1L] <- bad$SP[1L, 1L] + 1e-6
assert_error(
  fastkpc_full_cuda_fixed_sp_native_batch(bad, dto),
  "modified SP must fail closed", "selected-SP hash mismatch"
)
bad <- batch
bad$target_rows$selected_sp_names[[1L]] <-
  rev(bad$target_rows$selected_sp_names[[1L]])
assert_error(
  fastkpc_full_cuda_fixed_sp_native_batch(bad, dto),
  "SP name order must fail closed", "SP name order mismatch"
)
bad <- batch
class(bad$target_rows$selected_sp_names[[1L]]) <- "forged_sp_names"
assert_error(
  fastkpc_full_cuda_fixed_sp_native_batch(bad, dto),
  "classed SP names must fail closed", "SP name order mismatch"
)
bad <- batch
bad$prepared_s_key_sha256 <- strrep("0", 64L)
assert_error(
  fastkpc_full_cuda_fixed_sp_native_batch(bad, dto),
  "mismatched PreparedSKey must fail closed", "lineage mismatch"
)
bad <- batch
bad$planned_route <- bad$planned_route[-1L]
assert_error(
  fastkpc_full_cuda_fixed_sp_native_batch(bad, dto),
  "route length must fail closed", "route metadata is malformed"
)
bad <- batch
bad$planned_route[[1L]] <- "NOT_A_ROUTE"
assert_error(
  fastkpc_full_cuda_fixed_sp_native_batch(bad, dto),
  "route level must fail closed", "route metadata is malformed"
)
bad <- batch
route_levels <- fastkpc_full_cuda_fixed_sp_contract()$route_levels
bad$planned_route[[1L]] <- setdiff(
  route_levels, bad$planned_route[[1L]]
)[[1L]]
assert_error(
  fastkpc_full_cuda_fixed_sp_native_batch(bad, dto),
  "different valid route must fail closed", "route metadata is malformed"
)
bad <- batch
full_rank_indices <- which(
  bad$target_rows$coefficient_rank == dto$null_dim
)
assert_true(length(full_rank_indices) > 0L,
            "route tamper fixture requires one full-rank target")
full_rank_index <- full_rank_indices[[1L]]
bad$condition[[full_rank_index]] <- if (identical(
  bad$planned_route[[full_rank_index]], "CHOLESKY_BATCHED"
)) Inf else 1
assert_error(
  fastkpc_full_cuda_fixed_sp_native_batch(bad, dto),
  "condition tamper without route update must fail closed",
  "route metadata is malformed"
)
bad <- batch
class(bad$target_rows$residual_key_sha256) <- "forged_target_keys"
assert_error(
  fastkpc_full_cuda_fixed_sp_native_batch(bad, dto),
  "classed target keys must fail closed", "target keys are malformed"
)
bad <- batch
forged_key <- strrep("0", 64L)
bad$target_rows$residual_key_sha256[[1L]] <- forged_key
colnames(bad$Y)[[1L]] <- forged_key
colnames(bad$SP)[[1L]] <- forged_key
assert_error(
  fastkpc_full_cuda_fixed_sp_native_batch(bad, dto),
  "forged residual key must fail payload semantics",
  "residual key serialization mismatch"
)
bad <- batch
tamper_index <- 1L
payload <- bad$target_rows$residual_key_payload[[tamper_index]]
payload_p <- fastkpc_full_cuda_prepared_s_payload_integer(payload, "p")
target <- bad$target_rows$target[[tamper_index]]
tampered_target <- if (target < payload_p) target + 1L else target - 1L
tampered_payload <- fastkpc_full_cuda_census_residual_payload(
  target = tampered_target,
  S = bad$setup$sorted_S,
  formula_class = bad$setup$formula_class,
  data_hash = bad$setup$dataset_sha256,
  n = nrow(bad$setup$X),
  p = payload_p
)
tampered_key <- fastkpc_full_cuda_census_hash_utf8(tampered_payload)
bad$target_rows$residual_key_payload[[tamper_index]] <- tampered_payload
bad$target_rows$residual_key_sha256[[tamper_index]] <- tampered_key
colnames(bad$Y)[[tamper_index]] <- tampered_key
colnames(bad$SP)[[tamper_index]] <- tampered_key
bad$target_rows$target_state_fingerprint[[tamper_index]] <-
  fastkpc_full_cuda_target_state_fingerprint(
    bad$target_rows[tamper_index, , drop = FALSE]
  )
assert_error(
  fastkpc_full_cuda_fixed_sp_native_batch(bad, dto),
  "coherent residual payload tamper must fail setup semantics",
  "residual key serialization mismatch"
)
bad <- batch
tamper_index <- 1L
tampered_payload <- fastkpc_full_cuda_census_residual_payload(
  target = bad$target_rows$target[[tamper_index]],
  S = bad$setup$sorted_S,
  formula_class = bad$setup$formula_class,
  data_hash = bad$setup$dataset_sha256,
  n = nrow(bad$setup$X),
  p = dto$data_p + 1L
)
tampered_key <- fastkpc_full_cuda_census_hash_utf8(tampered_payload)
bad$target_rows$residual_key_payload[[tamper_index]] <- tampered_payload
bad$target_rows$residual_key_sha256[[tamper_index]] <- tampered_key
colnames(bad$Y)[[tamper_index]] <- tampered_key
colnames(bad$SP)[[tamper_index]] <- tampered_key
bad$target_rows$target_state_fingerprint[[tamper_index]] <-
  fastkpc_full_cuda_target_state_fingerprint(
    bad$target_rows[tamper_index, , drop = FALSE]
  )
assert_error(
  fastkpc_full_cuda_fixed_sp_native_batch(bad, dto),
  "coherent p+1 residual payload tamper must fail canonical data_p",
  "residual key serialization mismatch"
)
bad <- batch
bad$target_rows <- bad$target_rows[rev(seq_len(nrow(bad$target_rows))),
                                   , drop = FALSE]
assert_error(
  fastkpc_full_cuda_fixed_sp_native_batch(bad, dto),
  "target row order must fail closed", "target order mismatch"
)
bad <- batch
reverse_order <- rev(seq_len(nrow(bad$target_rows)))
bad$target_rows <- bad$target_rows[reverse_order, , drop = FALSE]
bad$Y <- bad$Y[, reverse_order, drop = FALSE]
bad$SP <- bad$SP[, reverse_order, drop = FALSE]
bad$planned_route <- bad$planned_route[reverse_order]
bad$condition <- bad$condition[reverse_order]
assert_error(
  fastkpc_full_cuda_fixed_sp_native_batch(bad, dto),
  "coherently reordered batch must fail canonical order",
  "canonical target order"
)
bad <- batch
duplicate_source <- 1L
duplicate_destination <- 2L
bad$target_rows[duplicate_destination, ] <-
  bad$target_rows[duplicate_source, ]
bad$Y[, duplicate_destination] <- bad$Y[, duplicate_source]
bad$SP[, duplicate_destination] <- bad$SP[, duplicate_source]
colnames(bad$Y)[[duplicate_destination]] <-
  colnames(bad$Y)[[duplicate_source]]
colnames(bad$SP)[[duplicate_destination]] <-
  colnames(bad$SP)[[duplicate_source]]
bad$condition[[duplicate_destination]] <- bad$condition[[duplicate_source]]
bad$planned_route[[duplicate_destination]] <-
  bad$planned_route[[duplicate_source]]
assert_true(identical(
  bad$target_rows$residual_key_sha256,
  sort(bad$target_rows$residual_key_sha256, method = "radix")
), "duplicate target fixture remains radix sorted")
assert_error(
  fastkpc_full_cuda_fixed_sp_native_batch(bad, dto),
  "coherent sorted duplicate target must fail identity uniqueness",
  "duplicate target identity"
)
bad <- batch
bad$target_rows <- bad$target_rows[-1L, , drop = FALSE]
assert_error(
  fastkpc_full_cuda_fixed_sp_native_batch(bad, dto),
  "target row count must fail closed", "target batch is malformed"
)
bad <- batch
class(bad$target_rows$y_hash) <- "forged_y_hashes"
assert_error(
  fastkpc_full_cuda_fixed_sp_native_batch(bad, dto),
  "classed Y hashes must fail closed", "Y hash metadata is malformed"
)
bad <- batch
class(bad$target_rows$selected_sp_hash) <- "forged_sp_hashes"
assert_error(
  fastkpc_full_cuda_fixed_sp_native_batch(bad, dto),
  "classed SP hashes must fail closed",
  "selected-SP hash metadata is malformed"
)
bad <- batch
bad$Y <- bad$Y[-1L, , drop = FALSE]
assert_error(
  fastkpc_full_cuda_fixed_sp_native_batch(bad, dto),
  "wrong Y dimensions must fail closed", "target batch is malformed"
)
bad <- batch
bad$SP <- bad$SP[-1L, , drop = FALSE]
assert_error(
  fastkpc_full_cuda_fixed_sp_native_batch(bad, dto),
  "wrong SP dimensions must fail closed", "target batch is malformed"
)
bad <- batch
bad$Y <- matrix(
  0L, nrow = nrow(bad$Y), ncol = ncol(bad$Y),
  dimnames = dimnames(bad$Y)
)
assert_error(
  fastkpc_full_cuda_fixed_sp_native_batch(bad, dto),
  "integer batch Y must fail closed", "numeric inputs are malformed"
)
bad <- batch
bad$SP <- matrix(
  0L, nrow = nrow(bad$SP), ncol = ncol(bad$SP),
  dimnames = dimnames(bad$SP)
)
assert_error(
  fastkpc_full_cuda_fixed_sp_native_batch(bad, dto),
  "integer batch SP must fail closed", "numeric inputs are malformed"
)

assert_error(
  fastkpc_full_cuda_fixed_sp_native_dto(
    structure(batch$setup, class = "forged_prepared_s")
  ), "classed setup must fail closed", "PreparedSSetup is malformed"
)
bad <- batch$setup
bad$X <- NULL
assert_error(
  fastkpc_full_cuda_fixed_sp_native_dto(bad),
  "missing setup field must fail closed", "PreparedSSetup is malformed"
)
bad <- batch$setup
storage.mode(bad$X) <- "integer"
assert_error(
  fastkpc_full_cuda_fixed_sp_native_dto(bad),
  "integer X must fail closed", "X must be a finite double matrix"
)
bad <- batch$setup
bad$X[1L, 1L] <- Inf
assert_error(
  fastkpc_full_cuda_fixed_sp_native_dto(bad),
  "nonfinite X must fail closed", "X must be a finite double matrix"
)
bad <- batch$setup
storage.mode(bad$penalty_blocks[[1L]]) <- "integer"
assert_error(
  fastkpc_full_cuda_fixed_sp_native_dto(bad),
  "integer penalty block must fail closed",
  "penalty blocks must be finite square double matrices"
)
bad <- batch$setup
bad$penalty_blocks[[1L]][1L, 1L] <- Inf
assert_error(
  fastkpc_full_cuda_fixed_sp_native_dto(bad),
  "nonfinite penalty block must fail closed",
  "penalty blocks must be finite square double matrices"
)
bad <- batch$setup
bad$penalty_blocks[[1L]] <- bad$penalty_blocks[[1L]][-1L, , drop = FALSE]
assert_error(
  fastkpc_full_cuda_fixed_sp_native_dto(bad),
  "nonsquare penalty block must fail closed",
  "penalty blocks must be finite square double matrices"
)
bad <- batch$setup
bad$weights_policy <- "unsupported-weights"
assert_error(
  fastkpc_full_cuda_fixed_sp_native_dto(bad),
  "unsupported weights policy must fail closed", "weights policy"
)
bad <- batch$setup
bad$offset_policy <- "unsupported-offset"
assert_error(
  fastkpc_full_cuda_fixed_sp_native_dto(bad),
  "unsupported offset policy must fail closed", "offset policy"
)
bad <- batch$setup
bad$H <- diag(ncol(bad$X))
assert_error(
  fastkpc_full_cuda_fixed_sp_native_dto(bad),
  "non-null H must fail closed", "non-null H"
)
bad <- batch$setup
bad$sp_mapping <- diag(length(bad$penalty_blocks))
assert_error(
  fastkpc_full_cuda_fixed_sp_native_dto(bad),
  "SP mapping must fail closed", "smoothing mapping"
)
bad <- batch$setup
bad$min_sp <- rep(0, length(bad$penalty_blocks))
assert_error(
  fastkpc_full_cuda_fixed_sp_native_dto(bad),
  "min.sp mapping must fail closed", "smoothing mapping"
)
for (bad_offset in c(-1L, 0L)) {
  bad <- batch$setup
  bad$penalty_offsets[[1L]] <- bad_offset
  assert_error(
    fastkpc_full_cuda_fixed_sp_native_dto(bad),
    "nonpositive Phase 2 offset must fail closed", "penalty offset"
  )
}
bad <- batch$setup
bad$penalty_offsets[[1L]] <- as.integer(
  ncol(bad$X) - nrow(bad$penalty_blocks[[1L]]) + 2L
)
assert_error(
  fastkpc_full_cuda_fixed_sp_native_dto(bad),
  "past-end penalty block must fail closed", "penalty offset"
)
bad <- batch$setup
bad$penalty_offsets <- as.double(bad$penalty_offsets)
bad$penalty_offsets[[1L]] <- 1.5
assert_error(
  fastkpc_full_cuda_fixed_sp_native_dto(bad),
  "fractional offsets must fail closed", "bare integer vector"
)
bad <- batch$setup
attr(bad$penalty_offsets, "forged") <- TRUE
assert_error(
  fastkpc_full_cuda_fixed_sp_native_dto(bad),
  "attributed offsets must fail closed", "bare integer vector"
)
for (bad_index in c(0L, length(batch$setup$penalty_blocks) + 1L)) {
  bad <- batch$setup
  bad$penalty_sp_indices[[1L]] <- bad_index
  assert_error(
    fastkpc_full_cuda_fixed_sp_native_dto(bad),
    "out-of-range SP index must fail closed", "penalty SP index"
  )
}
bad <- batch$setup
bad$penalty_sp_indices[1:2] <- rev(bad$penalty_sp_indices[1:2])
assert_error(
  fastkpc_full_cuda_fixed_sp_native_dto(bad),
  "reordered SP indices must fail closed", "identity penalty-to-SP mapping"
)
bad <- batch$setup
bad$penalty_sp_indices <- as.double(bad$penalty_sp_indices)
bad$penalty_sp_indices[[1L]] <- 1.5
assert_error(
  fastkpc_full_cuda_fixed_sp_native_dto(bad),
  "fractional SP indices must fail closed", "bare integer vector"
)
bad <- batch$setup
attr(bad$penalty_sp_indices, "forged") <- TRUE
assert_error(
  fastkpc_full_cuda_fixed_sp_native_dto(bad),
  "attributed SP indices must fail closed", "bare integer vector"
)

task4_apis <- c(
  "fixed_sp_cuda_runtime_create", "fixed_sp_cuda_runtime_reserve",
  "fixed_sp_cuda_prepared_create", "fixed_sp_cuda_solve_batch"
)
assert_true(!any(vapply(task4_apis, exists, logical(1L),
                        mode = "function", inherits = TRUE)),
            "Task 3 must not introduce CUDA runtime APIs")

elapsed <- proc.time()[["elapsed"]] - started_at
cat(sprintf(
  "PASS Phase 3 native DTO (1 setup, %d targets, %.3f seconds)\n",
  native_batch$target_count, elapsed
))
