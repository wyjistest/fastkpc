source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
expect_error_contains <- function(expr, text) {
  error <- tryCatch(force(expr), error = identity)
  assert_true(
    inherits(error, "error") &&
      grepl(text, conditionMessage(error), fixed = TRUE),
    paste0("expected error containing '", text, "'")
  )
  invisible(error)
}

required_helpers <- c(
  "fastkpc_full_cuda_fixed_sp_phase3a_relative_l2_diff",
  "fastkpc_full_cuda_fixed_sp_phase3a_validate_parity",
  "fastkpc_full_cuda_fixed_sp_phase3a_benchmark_identity",
  "fastkpc_full_cuda_fixed_sp_phase3a_benchmark_path_identities"
)
missing_helpers <- required_helpers[!vapply(
  required_helpers, exists, logical(1L), mode = "function", inherits = TRUE
)]
assert_true(
  length(missing_helpers) == 0L,
  paste("missing Phase 3A iteration helpers:",
        paste(missing_helpers, collapse = ", "))
)

actual <- c(1e-150, -2e-150)
reference <- c(0, 0)
numerator <- sqrt(sum((actual - reference)^2))
frozen_expected <- numerator / 1e-300
old_result <- numerator / max(
  sqrt(sum(reference^2)), .Machine$double.xmin
)
observed <- fastkpc_full_cuda_fixed_sp_phase3a_relative_l2_diff(
  actual, reference
)
assert_true(
  isTRUE(all.equal(observed, frozen_expected, tolerance = 1e-15)),
  "relative-L2 uses the frozen 1e-300 denominator floor"
)
assert_true(
  !isTRUE(all.equal(observed, old_result, tolerance = 1e-15)),
  "relative-L2 must not use .Machine$double.xmin"
)

safe_count <- 172L
stable_count <- 98L
target_keys <- sprintf("%064x", seq_len(safe_count + stable_count))
stable_routes <- rep(c("AUGMENTED_QR", "AUGMENTED_SVD"),
                     length.out = stable_count)
good_records <- data.frame(
  residual_key_sha256 = target_keys,
  planned_route = c(
    rep("CHOLESKY_BATCHED", safe_count),
    stable_routes
  ),
  authenticated_planned_route = c(
    rep("CHOLESKY_BATCHED", safe_count),
    stable_routes
  ),
  solver_status = c(
    rep("OK_CHOLESKY_SINGLE", safe_count),
    rep("ERR_STABLE_PATH_NOT_IMPLEMENTED", stable_count)
  ),
  residual_max_abs_diff = c(rep(1e-12, safe_count), rep(NA_real_, stable_count)),
  residual_relative_l2_diff = c(
    rep(1e-12, safe_count), rep(NA_real_, stable_count)
  ),
  fitted_max_abs_diff = c(rep(1e-12, safe_count), rep(NA_real_, stable_count)),
  fitted_relative_l2_diff = c(
    rep(1e-12, safe_count), rep(NA_real_, stable_count)
  ),
  stringsAsFactors = FALSE
)

benchmark_call_count <- 0L
validate_then_benchmark <- function(records) {
  parity <- fastkpc_full_cuda_fixed_sp_phase3a_validate_parity(records)
  benchmark_call_count <<- benchmark_call_count + 1L
  parity
}

bad_status <- good_records
bad_status$solver_status[[safe_count + 1L]] <- "OK_QR_SINGLE"
expect_error_contains(
  validate_then_benchmark(bad_status),
  "Phase 3A iteration status/numerical parity failed"
)
assert_true(
  benchmark_call_count == 0L,
  "bad stable status fails before benchmark work"
)

bad_error <- good_records
bad_error$residual_relative_l2_diff[[1L]] <- 1e-7
expect_error_contains(
  validate_then_benchmark(bad_error),
  "Phase 3A iteration status/numerical parity failed"
)
assert_true(
  benchmark_call_count == 0L,
  "bad numerical parity fails before benchmark work"
)

bad_authenticated_route <- good_records
bad_authenticated_route$authenticated_planned_route[[safe_count + 1L]] <-
  "AUGMENTED_SVD"
expect_error_contains(
  validate_then_benchmark(bad_authenticated_route),
  "Phase 3A iteration status/numerical parity failed"
)
assert_true(
  benchmark_call_count == 0L,
  "wrong per-key route fails before benchmark work"
)

bad_stable_route <- good_records
bad_stable_route$planned_route[[safe_count + 1L]] <- "FORGED_STABLE"
bad_stable_route$authenticated_planned_route[[safe_count + 1L]] <-
  "FORGED_STABLE"
expect_error_contains(
  validate_then_benchmark(bad_stable_route),
  "Phase 3A iteration status/numerical parity failed"
)
assert_true(
  benchmark_call_count == 0L,
  "stable attribution accepts only authenticated QR or SVD routes"
)

bad_stable_nan <- good_records
bad_stable_nan$residual_max_abs_diff[[safe_count + 1L]] <- NaN
expect_error_contains(
  validate_then_benchmark(bad_stable_nan),
  "Phase 3A iteration status/numerical parity failed"
)
assert_true(
  benchmark_call_count == 0L,
  "stable numerical evidence rejects NaN masquerading as NA"
)

parity <- validate_then_benchmark(good_records)
assert_true(
  benchmark_call_count == 1L &&
    identical(names(parity), c("safe", "stable")) &&
    sum(parity$safe) == safe_count && sum(parity$stable) == stable_count,
  "valid parity reaches benchmark work with frozen target partitions"
)

safe_keys <- good_records$residual_key_sha256[parity$safe]
safe_descriptors <- lapply(safe_keys, function(key) {
  list(native = list(target_keys = key))
})
prototype_dto <- list(
  prepared_s_key_sha256 = sprintf("%064x", 1L),
  X = matrix(c(1, 0, 1, 0, 1, 1), nrow = 3L),
  constraint_mode = "identity",
  coefficient_dim = 2L,
  null_dim = 2L,
  gram_matrix = crossprod(matrix(c(1, 0, 1, 0, 1, 1), nrow = 3L)),
  nullspace_gram_matrix = NULL,
  penalty_count = 1L,
  penalty_blocks = list(diag(2L)),
  penalty_offsets_zero_based = list(0L),
  penalty_sp_labels = "sp1"
)
canonical_y <- lapply(seq_along(safe_keys), function(index) {
  c(as.numeric(index), as.numeric(index + 1L), as.numeric(index + 2L))
})
canonical_sp <- lapply(seq_along(safe_keys), function(index) {
  as.numeric(index)
})
canonical_rhs <- lapply(canonical_y, function(y) {
  as.numeric(crossprod(prototype_dto$X, y))
})
canonical_target_rows <- Map(function(key, index, y, sp) {
  data.frame(
    residual_key_sha256 = key,
    target = as.integer(index),
    y_hash = fastkpc_full_cuda_census_metadata_hash(y),
    selected_sp_hash = fastkpc_full_cuda_census_metadata_hash(
      stats::setNames(sp, prototype_dto$penalty_sp_labels)
    ),
    stringsAsFactors = FALSE
  )
}, safe_keys, seq_along(safe_keys), canonical_y, canonical_sp)
wrong_sp_source_error <- tryCatch(
  fastkpc_full_cuda_fixed_sp_phase3a_prototype_expected_identity(
    dto = prototype_dto,
    target_key = safe_keys[[1L]],
    target_row = canonical_target_rows[[1L]],
    y = canonical_y[[1L]],
    sp = canonical_sp[[2L]],
    canonical_nullspace_rhs = canonical_rhs[[1L]],
    planned_route = "CHOLESKY_BATCHED"
  ),
  error = identity
)
assert_true(
  inherits(wrong_sp_source_error, "error") &&
    grepl("Phase 3A prototype expected source is malformed",
          conditionMessage(wrong_sp_source_error), fixed = TRUE),
  "selected SP must match the authenticated target-row hash"
)
wrong_y_nullspace <- canonical_y[[1L]] + c(-1, -1, 1)
assert_true(
  identical(
    as.numeric(crossprod(prototype_dto$X, wrong_y_nullspace)),
    canonical_rhs[[1L]]
  ),
  "wrong-Y adversary preserves the canonical nullspace RHS"
)
wrong_y_source_error <- tryCatch(
  fastkpc_full_cuda_fixed_sp_phase3a_prototype_expected_identity(
    dto = prototype_dto,
    target_key = safe_keys[[1L]],
    target_row = canonical_target_rows[[1L]],
    y = wrong_y_nullspace,
    sp = canonical_sp[[1L]],
    canonical_nullspace_rhs = canonical_rhs[[1L]],
    planned_route = "CHOLESKY_BATCHED"
  ),
  error = identity
)
assert_true(
  inherits(wrong_y_source_error, "error") &&
    grepl("Phase 3A prototype expected source is malformed",
          conditionMessage(wrong_y_source_error), fixed = TRUE),
  "target-row Y hash rejects a wrong response with the same RHS"
)
wrong_row_hash <- canonical_target_rows[[1L]]
wrong_row_hash$y_hash <- sprintf("%064x", safe_count + 1L)
wrong_row_hash_error <- tryCatch(
  fastkpc_full_cuda_fixed_sp_phase3a_prototype_expected_identity(
    dto = prototype_dto,
    target_key = safe_keys[[1L]],
    target_row = wrong_row_hash,
    y = canonical_y[[1L]],
    sp = canonical_sp[[1L]],
    canonical_nullspace_rhs = canonical_rhs[[1L]],
    planned_route = "CHOLESKY_BATCHED"
  ),
  error = identity
)
assert_true(
  inherits(wrong_row_hash_error, "error") &&
    grepl("Phase 3A prototype expected source is malformed",
          conditionMessage(wrong_row_hash_error), fixed = TRUE),
  "forged authenticated row hashes are rejected"
)
prototype_payloads <- Map(function(key, y, sp) {
  c(
    list(target_key = key),
    fastkpc_full_cuda_fixed_sp_phase3a_prototype_handle(
      prototype_dto, y, sp
    )
  )
}, safe_keys, canonical_y, canonical_sp)
prototype_descriptors <- Map(function(native, prototype, key, target_row,
                                    y, sp, rhs) {
  list(
    native = c(
      native$native,
      list(
        Y = matrix(y, ncol = 1L),
        SP = matrix(sp, ncol = 1L),
        planned_route = "CHOLESKY_BATCHED"
      )
    ),
    prototype = prototype,
    prototype_expected =
      fastkpc_full_cuda_fixed_sp_phase3a_prototype_expected_identity(
        dto = prototype_dto,
        target_key = key,
        target_row = target_row,
        y = y,
        sp = sp,
        canonical_nullspace_rhs = rhs,
        planned_route = "CHOLESKY_BATCHED"
      )
  )
}, safe_descriptors, prototype_payloads, safe_keys, canonical_target_rows,
   canonical_y, canonical_sp, canonical_rhs)
all_copied_actual_descriptors <- prototype_descriptors
for (index in seq_along(all_copied_actual_descriptors)) {
  all_copied_actual_descriptors[[index]]$prototype <- prototype_payloads[[1L]]
}
expect_error_contains(
  fastkpc_full_cuda_fixed_sp_phase3a_benchmark_path_identities(
    safe_descriptors, all_copied_actual_descriptors, safe_keys
  ),
  "Phase 3A prototype payload identity mismatch"
)
wrong_y_descriptors <- prototype_descriptors
wrong_y_descriptors[[1L]]$prototype <- c(
  list(target_key = safe_keys[[1L]]),
  fastkpc_full_cuda_fixed_sp_phase3a_prototype_handle(
    prototype_dto, canonical_y[[2L]], canonical_sp[[1L]]
  )
)
expect_error_contains(
  fastkpc_full_cuda_fixed_sp_phase3a_benchmark_path_identities(
    safe_descriptors, wrong_y_descriptors, safe_keys
  ),
  "Phase 3A prototype payload identity mismatch"
)
wrong_sp_descriptors <- prototype_descriptors
wrong_sp_descriptors[[1L]]$prototype <- c(
  list(target_key = safe_keys[[1L]]),
  fastkpc_full_cuda_fixed_sp_phase3a_prototype_handle(
    prototype_dto, canonical_y[[1L]], canonical_sp[[2L]]
  )
)
expect_error_contains(
  fastkpc_full_cuda_fixed_sp_phase3a_benchmark_path_identities(
    safe_descriptors, wrong_sp_descriptors, safe_keys
  ),
  "Phase 3A prototype payload identity mismatch"
)
benchmark_identity <-
  fastkpc_full_cuda_fixed_sp_phase3a_benchmark_identity(
    safe_descriptors, safe_keys
  )
assert_true(
  identical(names(benchmark_identity), c(
    "benchmark_target_count", "ordered_target_keys",
    "target_key_corpus_hash"
  )) &&
    identical(benchmark_identity$benchmark_target_count, safe_count) &&
    identical(benchmark_identity$ordered_target_keys, safe_keys) &&
    identical(
      benchmark_identity$target_key_corpus_hash,
      fastkpc_full_cuda_census_key_set_hash(safe_keys)
    ),
  "benchmark identity freezes the exact ordered 172-target corpus"
)
expect_error_contains(
  fastkpc_full_cuda_fixed_sp_phase3a_benchmark_identity(
    safe_descriptors[-length(safe_descriptors)], safe_keys
  ),
  "Phase 3A benchmark target identity mismatch"
)
expect_error_contains(
  fastkpc_full_cuda_fixed_sp_phase3a_benchmark_identity(
    safe_descriptors[c(2L, 1L, 3:length(safe_descriptors))], safe_keys
  ),
  "Phase 3A benchmark target identity mismatch"
)

path_identities <-
  fastkpc_full_cuda_fixed_sp_phase3a_benchmark_path_identities(
    safe_descriptors, prototype_descriptors, safe_keys
  )
assert_true(
  identical(names(path_identities), c("persistent", "prototype")) &&
    identical(path_identities$persistent, benchmark_identity) &&
    identical(path_identities$prototype$benchmark_target_count, safe_count) &&
    identical(path_identities$prototype$ordered_target_keys, safe_keys) &&
    identical(path_identities$prototype$ordered_payload_hashes,
              unname(vapply(prototype_payloads,
                            fastkpc_full_cuda_census_named_metadata_hash,
                            character(1L)))) &&
    identical(
      path_identities$prototype$payload_corpus_hash,
      fastkpc_full_cuda_census_key_set_hash(
        path_identities$prototype$ordered_payload_hashes
      )
    ),
  "persistent and prototype paths authenticate the same keys and payloads"
)
expect_error_contains(
  fastkpc_full_cuda_fixed_sp_phase3a_benchmark_path_identities(
    safe_descriptors[-length(safe_descriptors)], prototype_descriptors, safe_keys
  ),
  "Phase 3A benchmark target identity mismatch"
)
expect_error_contains(
  fastkpc_full_cuda_fixed_sp_phase3a_benchmark_path_identities(
    safe_descriptors, prototype_descriptors[-length(prototype_descriptors)],
    safe_keys
  ),
  "Phase 3A benchmark target identity mismatch"
)
reordered_descriptors <-
  safe_descriptors[c(2L, 1L, 3:length(safe_descriptors))]
expect_error_contains(
  fastkpc_full_cuda_fixed_sp_phase3a_benchmark_path_identities(
    reordered_descriptors, prototype_descriptors, safe_keys
  ),
  "Phase 3A benchmark target identity mismatch"
)
swapped_prototype_descriptors <- prototype_descriptors
swapped_prototypes <- lapply(
  prototype_descriptors[c(2L, 1L, 3:length(prototype_descriptors))],
  `[[`, "prototype"
)
for (index in seq_along(swapped_prototype_descriptors)) {
  swapped_prototype_descriptors[[index]]$prototype <- swapped_prototypes[[index]]
}
expect_error_contains(
  fastkpc_full_cuda_fixed_sp_phase3a_benchmark_path_identities(
    safe_descriptors, swapped_prototype_descriptors, safe_keys
  ),
  "Phase 3A prototype payload identity mismatch"
)
duplicated_prototype_descriptors <- prototype_descriptors
duplicated_prototype_descriptors[[2L]]$prototype <-
  duplicated_prototype_descriptors[[1L]]$prototype
expect_error_contains(
  fastkpc_full_cuda_fixed_sp_phase3a_benchmark_path_identities(
    safe_descriptors, duplicated_prototype_descriptors, safe_keys
  ),
  "Phase 3A prototype payload identity mismatch"
)
mutated_prototype_descriptors <- prototype_descriptors
mutated_prototype_descriptors[[1L]]$prototype$X[[1L]] <-
  mutated_prototype_descriptors[[1L]]$prototype$X[[1L]] + 1
expect_error_contains(
  fastkpc_full_cuda_fixed_sp_phase3a_benchmark_path_identities(
    safe_descriptors, mutated_prototype_descriptors, safe_keys
  ),
  "Phase 3A prototype payload identity mismatch"
)

runner_body <- paste(
  deparse(body(fastkpc_run_full_cuda_fixed_sp_phase3a_iteration)),
  collapse = "\n"
)
parity_position <- regexpr(
  "fastkpc_full_cuda_fixed_sp_phase3a_validate_parity",
  runner_body, fixed = TRUE
)[[1L]]
benchmark_setup_position <- regexpr(
  "safe_descriptors <- list()",
  runner_body, fixed = TRUE
)[[1L]]
warmup_position <- regexpr(
  "persistent_warmup <- run_persistent_corpus",
  runner_body, fixed = TRUE
)[[1L]]
expected_identity_position <- regexpr(
  "prototype_expected <-", runner_body, fixed = TRUE
)[[1L]]
prototype_payload_position <- regexpr(
  "prototype <- fastkpc_full_cuda_fixed_sp_phase3a_prototype_handle",
  runner_body, fixed = TRUE
)[[1L]]
assert_true(
  parity_position > 0L && benchmark_setup_position > parity_position &&
    warmup_position > benchmark_setup_position &&
    expected_identity_position > benchmark_setup_position &&
    prototype_payload_position > expected_identity_position,
  "production authenticates canonical prototype inputs before payload construction"
)

cat("PASS Phase 3A iteration helper contracts\n")
