source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP Phase 3A nonfinite fixed-sp output\n")
  quit(save = "no", status = 0)
}

build_fastkpc_cuda_native(rebuild = TRUE)

sha <- function(character) strrep(character, 64L)
dto <- list(
  schema_version = "full-cuda-ci-prepared-s-native-dto-v1",
  dataset_sha256 = sha("1"),
  prepared_s_key_sha256 = sha("2"),
  same_S_group_id = sha("3"),
  phase1_setup_fingerprint = sha("4"),
  provider_fingerprint = sha("5"),
  semantic_fingerprint = sha("6"),
  representation_fingerprint = sha("7"),
  prepared_s_setup_schema_version = "full-cuda-ci-prepared-s-setup-v1",
  native_dto_schema_version = "full-cuda-ci-prepared-s-native-dto-v1",
  data_p = 48L,
  n = 2L,
  coefficient_dim = 1L,
  null_dim = 1L,
  penalty_count = 1L,
  X = matrix(.Machine$double.xmax, nrow = 2L, ncol = 1L),
  constraint_mode = "identity",
  constraint_nullspace = NULL,
  gram_matrix = matrix(1, nrow = 1L, ncol = 1L),
  nullspace_gram_matrix = NULL,
  penalty_blocks = list(penalty_1 = matrix(0, nrow = 1L, ncol = 1L)),
  penalty_offsets_zero_based = 0L,
  penalty_ranks = 0L,
  penalty_sp_indices_zero_based = 0L,
  penalty_sp_labels = "sp1",
  H = NULL,
  weights_policy = "none-or-unit",
  offset_policy = "none-or-zero"
)

runtime <- fixed_sp_cuda_runtime_create(0L)
on.exit(try(fixed_sp_cuda_runtime_free(runtime), silent = TRUE), add = TRUE)
fixed_sp_cuda_runtime_reserve(runtime, 2L, 1L, 1L, 1L, 3L)
handle <- fixed_sp_cuda_prepared_create(runtime, dto)
on.exit(try(fixed_sp_cuda_prepared_free(handle), silent = TRUE), add = TRUE)

outputs <- c("coefficients", "fitted", "residuals", "rss", "rhs")
result_names <- c(
  "coefficients", "fitted", "residuals", "rss", "cuda_nullspace_rhs"
)
solve_overflow <- function() {
  fixed_sp_cuda_solve_batch(
    handle,
    matrix(2, nrow = 2L, ncol = 1L),
    matrix(0, nrow = 1L, ncol = 1L),
    "CHOLESKY_BATCHED", sha("8"), outputs = outputs
  )
}

token <- solve_overflow()
on.exit(try(fixed_sp_cuda_residual_free(token), silent = TRUE), add = TRUE)
info_before_shadow <- fixed_sp_cuda_residual_info(token)
shadow <- fixed_sp_cuda_materialize_shadow(token, outputs = outputs)
info <- fixed_sp_cuda_residual_info(token)
all_shadow_na <- identical(names(shadow), result_names) &&
  all(vapply(shadow, function(value) all(is.na(value)), logical(1L)))
assert_true(
  identical(info_before_shadow$planned_route, "CHOLESKY_BATCHED") &&
    identical(info_before_shadow$executed_route, "CHOLESKY_BATCHED") &&
    identical(info_before_shadow$reroute_reason, "") &&
    identical(info_before_shadow$solver_status, "ERR_NONFINITE_OUTPUT") &&
    identical(info_before_shadow$nonfinite_output_count, 1L) &&
    identical(info$solver_status, "ERR_NONFINITE_OUTPUT") &&
    identical(info$nonfinite_output_count, 1L) &&
    all_shadow_na &&
    info$shadow_materialize_call_count == 1L &&
    info$shadow_materialize_target_count == 1L &&
    info$shadow_d2h_bytes == 0 &&
    info$implicit_residual_d2h_count == 0L,
  paste0(
    "nonfinite CUDA outputs must fail closed without numeric D2H; status=",
    info$solver_status, "; count=",
    if (is.null(info$nonfinite_output_count)) "MISSING" else
      info$nonfinite_output_count,
    "; all_na=", all_shadow_na, "; shadow_d2h_bytes=",
    info$shadow_d2h_bytes
  )
)
assert_true(
  identical(dim(shadow$coefficients), c(1L, 1L)) &&
    identical(dim(shadow$fitted), c(2L, 1L)) &&
    identical(dim(shadow$residuals), c(2L, 1L)) &&
    identical(length(shadow$rss), 1L) &&
    identical(dim(shadow$cuda_nullspace_rhs), c(1L, 1L)),
  "nonfinite shadow retains canonical output shapes"
)
fixed_sp_cuda_residual_release(token)
fixed_sp_cuda_residual_free(token)

released_token <- solve_overflow()
fixed_sp_cuda_residual_release(released_token)
replacement_token <- solve_overflow()
released_info <- fixed_sp_cuda_residual_info(released_token)
assert_true(
  identical(released_info$solver_status, "ERR_NONFINITE_OUTPUT") &&
    identical(released_info$nonfinite_output_count, 1L),
  "release resolves and retains compact nonfinite status before slot reuse"
)
fixed_sp_cuda_residual_free(released_token)
fixed_sp_cuda_residual_free(replacement_token)
assert_true(
  !isTRUE(fixed_sp_cuda_prepared_info(handle)$output_slot_leased),
  "free before explicit status observation resolves and releases safely"
)

fixed_sp_cuda_prepared_free(handle)
fixed_sp_cuda_runtime_free(runtime)
cat("PASS Phase 3A nonfinite fixed-sp output\n")
