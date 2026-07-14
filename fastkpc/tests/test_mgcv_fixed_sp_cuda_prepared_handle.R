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

unreserved <- fixed_sp_cuda_runtime_create(0L)
on.exit(try(fixed_sp_cuda_runtime_free(unreserved), silent = TRUE), add = TRUE)
assert_error(
  fixed_sp_cuda_prepared_create(unreserved, dto),
  "reserve", "unreserved runtime must fail closed"
)
fixed_sp_cuda_runtime_free(unreserved)

runtime <- fixed_sp_cuda_runtime_create(0L)
on.exit(try(fixed_sp_cuda_runtime_free(runtime), silent = TRUE), add = TRUE)
fixed_sp_cuda_runtime_reserve(runtime, 351L, 64L, 47L, 7L, 407L)

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
bad_dto$penalty_offsets_zero_based[[1L]] <- dto$coefficient_dim
assert_error(
  fixed_sp_cuda_prepared_create(runtime, bad_dto),
  "offset", "out-of-range penalty offset must fail closed"
)
bad_dto <- dto
bad_dto$penalty_sp_indices_zero_based[c(1L, 2L)] <-
  rev(bad_dto$penalty_sp_indices_zero_based[c(1L, 2L)])
assert_error(
  fixed_sp_cuda_prepared_create(runtime, bad_dto),
  "identity", "non-identity penalty-to-SP mapping must fail closed"
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
