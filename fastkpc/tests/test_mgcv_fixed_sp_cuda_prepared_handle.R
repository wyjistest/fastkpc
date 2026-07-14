source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_error <- function(expression, pattern, message) {
  error <- tryCatch(force(expression), error = identity)
  assert_true(
    inherits(error, "error") &&
      grepl(pattern, conditionMessage(error), fixed = TRUE),
    paste0(message, if (inherits(error, "error")) {
      paste0(": ", conditionMessage(error))
    } else {
      ": no error"
    })
  )
  invisible(error)
}
assert_close <- function(actual, expected, tolerance, message) {
  assert_true(identical(dim(actual), dim(expected)),
              paste0(message, ": dimensions"))
  error <- max(abs(actual - expected))
  assert_true(is.finite(error) && error <= tolerance,
              paste0(message, ": max error ", format(error, digits = 17L)))
}

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP Phase 3 prepared handle\n")
  quit(save = "no", status = 0)
}

build_fastkpc_cuda_native(rebuild = TRUE)
catalog <- fastkpc_full_cuda_open_fixed_sp_catalog(
  file.path("fastkpc", "artifacts", "full_cuda_ci", "oracle_351x48_v1"),
  file.path("fastkpc", "artifacts", "full_cuda_ci",
            "workload_census_351x48_v1"),
  file.path("fastkpc", "artifacts", "full_cuda_ci",
            "prepared_s_contract_v1"),
  file.path("fastkpc", "artifacts", "kpc_tprs_real_zhu",
            "cancer_RD-causalDiscoveryInput.rds")
)
iteration <- fastkpc_full_cuda_fixed_sp_scope(catalog, "iteration")
setup_index <- which(
  iteration$setup_rows$penalty_count > 1L &
    iteration$setup_rows$constraint_rank == 0L
)[[1L]]
scope <- iteration
scope$setup_rows <- iteration$setup_rows[setup_index, , drop = FALSE]
selected_key <- scope$setup_rows$prepared_s_key_sha256[[1L]]
scope$target_rows <- iteration$target_rows[
  iteration$target_rows$prepared_s_key_sha256 == selected_key,
  , drop = FALSE
]
selected_rank <- match(
  selected_key, catalog$setup_index$prepared_s_key_sha256
)
scope$shard_ids <- as.integer(
  (selected_rank - 1L) %% catalog$catalog_contract$shard_count
)
batch <- fastkpc_full_cuda_fixed_sp_batches(catalog, scope)[[1L]]
dto <- fastkpc_full_cuda_fixed_sp_native_dto(batch$setup)
assert_true(identical(dto$constraint_mode, "identity") &&
              identical(dto$null_dim, dto$coefficient_dim),
            "focused setup uses canonical identity constraint")
explicit_dto <- dto
explicit_q <- dto$coefficient_dim - 1L
householder_v <- seq_len(dto$coefficient_dim)
householder_v <- householder_v / sqrt(sum(householder_v^2))
householder <- diag(dto$coefficient_dim) -
  2 * tcrossprod(householder_v)
explicit_Z <- householder[
  , seq_len(explicit_q), drop = FALSE
]
explicit_X_null <- dto$X %*% explicit_Z
assert_close(crossprod(explicit_Z), diag(explicit_q), 1e-12,
             "dense explicit Z is orthonormal")
explicit_dto$constraint_mode <- "explicit"
explicit_dto$constraint_nullspace <- explicit_Z
explicit_dto$null_dim <- explicit_q
explicit_dto$nullspace_gram_matrix <- crossprod(explicit_X_null)

unreserved <- fixed_sp_cuda_runtime_create(0L)
on.exit(try(fixed_sp_cuda_runtime_free(unreserved), silent = TRUE), add = TRUE)
assert_error(
  fixed_sp_cuda_prepared_create(unreserved, dto),
  "reserve", "unreserved runtime must fail closed"
)
fixed_sp_cuda_runtime_free(unreserved)

overflow_dto <- dto
overflow_dto$X <- matrix(
  .Machine$double.xmax, nrow = dto$n, ncol = dto$coefficient_dim
)
overflow_dto$constraint_mode <- "explicit"
overflow_dto$constraint_nullspace <- matrix(
  1, nrow = dto$coefficient_dim, ncol = 1L
)
overflow_dto$null_dim <- 1L
overflow_dto$nullspace_gram_matrix <- matrix(1, nrow = 1L, ncol = 1L)
overflow_runtime <- fixed_sp_cuda_runtime_create(0L)
on.exit(try(fixed_sp_cuda_runtime_free(overflow_runtime), silent = TRUE),
        add = TRUE)
fixed_sp_cuda_runtime_reserve(
  overflow_runtime, dto$n, 1L, 1L, dto$penalty_count, 407L
)
assert_error(
  fixed_sp_cuda_prepared_create(overflow_runtime, overflow_dto),
  "derived", "nonfinite derived explicit setup must fail closed"
)
fixed_sp_cuda_runtime_free(overflow_runtime)

growth_runtime <- fixed_sp_cuda_runtime_create(0L)
on.exit(try(fixed_sp_cuda_runtime_free(growth_runtime), silent = TRUE),
        add = TRUE)
fixed_sp_cuda_runtime_reserve(
  growth_runtime, dto$n, dto$null_dim, 1L, dto$penalty_count, 407L
)
growth_handle <- fixed_sp_cuda_prepared_create(growth_runtime, dto)
on.exit(try(fixed_sp_cuda_prepared_free(growth_handle), silent = TRUE),
        add = TRUE)
fixed_sp_cuda_runtime_reserve(
  growth_runtime, dto$n, dto$null_dim, 1L, dto$penalty_count, 407L
)
fixed_sp_cuda_runtime_reserve(
  growth_runtime, dto$n - 1L, dto$null_dim, 1L,
  dto$penalty_count, 406L
)
assert_error(
  fixed_sp_cuda_runtime_reserve(
    growth_runtime, dto$n, dto$null_dim, 2L, dto$penalty_count, 407L
  ),
  "active prepared", "runtime growth with an active handle must fail closed"
)
fixed_sp_cuda_prepared_free(growth_handle)
fixed_sp_cuda_runtime_reserve(
  growth_runtime, dto$n, dto$null_dim, 2L, dto$penalty_count, 407L
)
fixed_sp_cuda_runtime_free(growth_runtime)

runtime <- fixed_sp_cuda_runtime_create(0L)
on.exit(try(fixed_sp_cuda_runtime_free(runtime), silent = TRUE), add = TRUE)
fixed_sp_cuda_runtime_reserve(runtime, 351L, 64L, 47L, 7L, 407L)

n_limited_runtime <- fixed_sp_cuda_runtime_create(0L)
on.exit(try(fixed_sp_cuda_runtime_free(n_limited_runtime), silent = TRUE),
        add = TRUE)
fixed_sp_cuda_runtime_reserve(
  n_limited_runtime, dto$n - 1L, dto$null_dim, 1L,
  dto$penalty_count, 1L
)
assert_error(
  fixed_sp_cuda_prepared_create(n_limited_runtime, dto),
  "exceed reserved", "n above reserved capacity must fail closed"
)
fixed_sp_cuda_runtime_free(n_limited_runtime)

q_limited_runtime <- fixed_sp_cuda_runtime_create(0L)
on.exit(try(fixed_sp_cuda_runtime_free(q_limited_runtime), silent = TRUE),
        add = TRUE)
fixed_sp_cuda_runtime_reserve(
  q_limited_runtime, dto$n, dto$null_dim - 1L, 1L,
  dto$penalty_count, 1L
)
assert_error(
  fixed_sp_cuda_prepared_create(q_limited_runtime, dto),
  "exceed reserved", "null_dim above reserved capacity must fail closed"
)
fixed_sp_cuda_runtime_free(q_limited_runtime)

penalty_limited_runtime <- fixed_sp_cuda_runtime_create(0L)
on.exit(try(fixed_sp_cuda_runtime_free(penalty_limited_runtime),
            silent = TRUE), add = TRUE)
fixed_sp_cuda_runtime_reserve(
  penalty_limited_runtime, dto$n, dto$null_dim, 1L,
  dto$penalty_count - 1L, 1L
)
assert_error(
  fixed_sp_cuda_prepared_create(penalty_limited_runtime, dto),
  "exceed reserved", "penalty_count above reserved capacity must fail closed"
)
fixed_sp_cuda_runtime_free(penalty_limited_runtime)

bad_dto <- dto[rev(seq_along(dto))]
assert_error(
  fixed_sp_cuda_prepared_create(runtime, bad_dto),
  "exact fields", "reordered DTO fields must fail closed"
)
bad_dto <- dto[-1L]
assert_error(
  fixed_sp_cuda_prepared_create(runtime, bad_dto),
  "exact fields", "dropped DTO field must fail closed"
)
bad_dto <- dto
names(bad_dto)[[1L]] <- "forged_schema_version"
assert_error(
  fixed_sp_cuda_prepared_create(runtime, bad_dto),
  "exact fields", "renamed DTO field must fail closed"
)
bad_dto <- dto
bad_dto$data_p <- 47L
assert_error(
  fixed_sp_cuda_prepared_create(runtime, bad_dto),
  "data_p", "canonical data_p tamper must fail closed"
)
bad_dto <- dto
bad_dto$n <- structure(dto$n, forged = TRUE)
assert_error(
  fixed_sp_cuda_prepared_create(runtime, bad_dto),
  "scalar integer", "attributed scalar integer must fail closed"
)
bad_dto <- dto
bad_dto$null_dim <- structure(dto$null_dim, class = "forged_integer")
assert_error(
  fixed_sp_cuda_prepared_create(runtime, bad_dto),
  "scalar integer", "classed scalar integer must fail closed"
)
bad_dto <- dto
bad_dto$schema_version <- "forged-native-dto-schema"
assert_error(
  fixed_sp_cuda_prepared_create(runtime, bad_dto),
  "schema lineage", "schema version tamper must fail closed"
)
bad_dto <- dto
bad_dto$dataset_sha256 <- "not-a-sha256"
assert_error(
  fixed_sp_cuda_prepared_create(runtime, bad_dto),
  "SHA-256", "lineage SHA tamper must fail closed"
)
bad_dto <- dto
bad_dto$gram_matrix <- bad_dto$gram_matrix[-1L, , drop = FALSE]
assert_error(
  fixed_sp_cuda_prepared_create(runtime, bad_dto),
  "shape", "malformed Gram shape must fail closed"
)
bad_dto <- dto
bad_dto$X[[1L]] <- Inf
assert_error(
  fixed_sp_cuda_prepared_create(runtime, bad_dto),
  "finite", "nonfinite setup matrix must fail closed"
)
bad_dto <- dto
bad_dto$penalty_blocks[[1L]] <-
  bad_dto$penalty_blocks[[1L]][-1L, , drop = FALSE]
assert_error(
  fixed_sp_cuda_prepared_create(runtime, bad_dto),
  "shape", "non-square penalty block must fail closed"
)
bad_dto <- dto
attr(bad_dto$penalty_blocks, "forged") <- TRUE
assert_error(
  fixed_sp_cuda_prepared_create(runtime, bad_dto),
  "canonical named list", "attributed penalty_blocks list must fail closed"
)
bad_dto <- dto
names(bad_dto$penalty_blocks)[[1L]] <- "forged_penalty"
assert_error(
  fixed_sp_cuda_prepared_create(runtime, bad_dto),
  "canonical named list", "penalty block names must fail closed"
)
bad_dto <- dto
bad_dto$penalty_blocks[[1L]][[1L]] <- Inf
assert_error(
  fixed_sp_cuda_prepared_create(runtime, bad_dto),
  "finite", "nonfinite penalty block must fail closed"
)
bad_dto <- dto
bad_dto$penalty_offsets_zero_based[[1L]] <- dto$coefficient_dim
assert_error(
  fixed_sp_cuda_prepared_create(runtime, bad_dto),
  "offset", "out-of-range penalty offset must fail closed"
)
bad_dto <- dto
bad_dto$penalty_ranks[[1L]] <- nrow(dto$penalty_blocks[[1L]]) + 1L
assert_error(
  fixed_sp_cuda_prepared_create(runtime, bad_dto),
  "rank", "out-of-range penalty rank must fail closed"
)
bad_dto <- dto
bad_dto$penalty_sp_indices_zero_based[c(1L, 2L)] <-
  rev(bad_dto$penalty_sp_indices_zero_based[c(1L, 2L)])
assert_error(
  fixed_sp_cuda_prepared_create(runtime, bad_dto),
  "identity", "non-identity penalty-to-SP mapping must fail closed"
)
bad_dto <- explicit_dto
bad_dto$constraint_nullspace <- explicit_Z[-1L, , drop = FALSE]
assert_error(
  fixed_sp_cuda_prepared_create(runtime, bad_dto),
  "shape", "explicit constraint shape must fail closed"
)
bad_dto <- explicit_dto
bad_dto["constraint_nullspace"] <- list(NULL)
assert_error(
  fixed_sp_cuda_prepared_create(runtime, bad_dto),
  "explicit constraint", "missing explicit constraint must fail closed"
)
bad_dto <- explicit_dto
bad_dto$nullspace_gram_matrix <-
  bad_dto$nullspace_gram_matrix[-1L, , drop = FALSE]
assert_error(
  fixed_sp_cuda_prepared_create(runtime, bad_dto),
  "shape", "explicit nullspace Gram shape must fail closed"
)
bad_dto <- explicit_dto
bad_dto["nullspace_gram_matrix"] <- list(NULL)
assert_error(
  fixed_sp_cuda_prepared_create(runtime, bad_dto),
  "explicit constraint", "missing nullspace Gram must fail closed"
)
bad_dto <- dto
bad_dto$H <- diag(dto$coefficient_dim)
assert_error(
  fixed_sp_cuda_prepared_create(runtime, bad_dto),
  "non-null H", "non-null H must fail closed"
)
bad_dto <- dto
bad_dto$weights_policy <- "non-unit-weights"
assert_error(
  fixed_sp_cuda_prepared_create(runtime, bad_dto),
  "weights policy", "unsupported weights policy must fail closed"
)
bad_dto <- dto
bad_dto$offset_policy <- "non-zero-offset"
assert_error(
  fixed_sp_cuda_prepared_create(runtime, bad_dto),
  "offset policy", "unsupported offset policy must fail closed"
)
assert_error(
  fixed_sp_cuda_prepared_info(runtime),
  "tagged external pointer", "wrong prepared pointer tag must fail closed"
)

runtime_before_prepared <- fixed_sp_cuda_runtime_info(runtime)
handle <- fixed_sp_cuda_prepared_create(runtime, dto)
on.exit(try(fixed_sp_cuda_prepared_free(handle), silent = TRUE), add = TRUE)
info <- fixed_sp_cuda_prepared_info(handle)
runtime_after_prepared <- fixed_sp_cuda_runtime_info(runtime)

assert_true(identical(info$prepared_s_key_sha256,
                      dto$prepared_s_key_sha256), "PreparedSKey")
assert_true(info$setup_h2d_upload_count == 1L, "one setup upload")
expected_setup_h2d_bytes <- 8 * as.numeric(
  length(dto$X) + length(dto$gram_matrix) +
    dto$penalty_count * dto$null_dim * dto$null_dim
)
assert_true(identical(info$setup_h2d_bytes, expected_setup_h2d_bytes),
            "identity setup uploads X, Gram, and projected penalties once")
assert_true(info$n == dto$n && info$null_dim == dto$null_dim,
            "setup dimensions")
assert_true(!isTRUE(info$output_slot_leased),
            "new prepared handle has a free output slot")
repeat_info <- fixed_sp_cuda_prepared_info(handle)
assert_true(repeat_info$setup_h2d_upload_count == 1L &&
              repeat_info$setup_h2d_bytes == info$setup_h2d_bytes,
            "prepared info does not repeat setup upload")
assert_true(
  identical(runtime_after_prepared$stream_create_count,
            runtime_before_prepared$stream_create_count) &&
    identical(runtime_after_prepared$cublas_handle_create_count,
              runtime_before_prepared$cublas_handle_create_count) &&
    identical(runtime_after_prepared$cusolver_handle_create_count,
              runtime_before_prepared$cusolver_handle_create_count),
  "prepared create reuses runtime stream and library handles"
)
assert_error(
  fixed_sp_cuda_prepared_free(runtime),
  "tagged external pointer", "wrong prepared free tag must fail closed"
)

expected_projected_penalties <- array(
  0, dim = c(explicit_q, explicit_q, dto$penalty_count)
)
truncated_projected_penalties <- expected_projected_penalties
for (index in seq_len(dto$penalty_count)) {
  full_penalty <- matrix(0, dto$coefficient_dim, dto$coefficient_dim)
  block <- dto$penalty_blocks[[index]]
  block_indices <- dto$penalty_offsets_zero_based[[index]] +
    seq_len(nrow(block))
  full_penalty[block_indices, block_indices] <- block
  expected_projected_penalties[, , index] <-
    crossprod(explicit_Z, full_penalty %*% explicit_Z)
  truncated_projected_penalties[, , index] <- full_penalty[
    seq_len(explicit_q), seq_len(explicit_q), drop = FALSE
  ]
}
assert_true(
  max(abs(explicit_X_null -
            dto$X[, seq_len(explicit_q), drop = FALSE])) > 1e-6,
  "dense explicit fixture distinguishes XZ from column truncation"
)
assert_true(
  max(abs(expected_projected_penalties -
            truncated_projected_penalties)) > 1e-6,
  "dense explicit fixture distinguishes Z'PZ from leading submatrices"
)
explicit_handle <- fixed_sp_cuda_prepared_create(runtime, explicit_dto)
on.exit(try(fixed_sp_cuda_prepared_free(explicit_handle), silent = TRUE),
        add = TRUE)
explicit_info <- fixed_sp_cuda_prepared_info(explicit_handle)
expected_explicit_h2d_bytes <- 8 * as.numeric(
  length(explicit_dto$X) + length(explicit_Z) +
    length(explicit_X_null) + length(explicit_dto$nullspace_gram_matrix) +
    explicit_dto$penalty_count * explicit_q * explicit_q
)
assert_true(identical(
  explicit_info$setup_h2d_bytes, expected_explicit_h2d_bytes
), "explicit setup uploads X, Z, X_null, Gram, and projected penalties")
runtime_before_shadow <- fixed_sp_cuda_runtime_info(runtime)
prepared_before_shadow <- fixed_sp_cuda_prepared_info(explicit_handle)
explicit_shadow <- .Call(
  "C_fixed_sp_cuda_test_prepared_static_shadow", explicit_handle,
  PACKAGE = "fastkpc_cuda"
)
assert_true(identical(fixed_sp_cuda_runtime_info(runtime),
                      runtime_before_shadow),
            "test shadow does not change runtime diagnostics")
assert_true(identical(fixed_sp_cuda_prepared_info(explicit_handle),
                      prepared_before_shadow),
            "test shadow does not change prepared diagnostics")
assert_true(identical(
  names(explicit_shadow),
  c("X_null", "gram", "projected_penalties")
), "explicit static shadow schema")
assert_close(explicit_shadow$X_null, explicit_X_null, 1e-12,
             "explicit X_null projection")
assert_close(explicit_shadow$gram,
             explicit_dto$nullspace_gram_matrix, 0,
             "explicit nullspace Gram")
assert_close(explicit_shadow$projected_penalties,
             expected_projected_penalties, 1e-12,
             "explicit projected penalties")
fixed_sp_cuda_prepared_free(explicit_handle)

fixed_sp_cuda_runtime_free(runtime)
retained_info <- fixed_sp_cuda_prepared_info(handle)
assert_true(retained_info$setup_h2d_upload_count == 1L,
            "prepared handle retains runtime context")

fixed_sp_cuda_prepared_free(handle)
fixed_sp_cuda_prepared_free(handle)
assert_error(
  fixed_sp_cuda_prepared_info(handle),
  "freed", "freed prepared handle rejects use"
)

cat("PASS Phase 3 prepared handle\n")
