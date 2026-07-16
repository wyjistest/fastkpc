source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP Phase 3C fixed-sp stable geometry\n")
  quit(save = "no", status = 0L)
}

build_fastkpc_cuda_native(rebuild = TRUE)
assert_true(fastkpc_cuda_available(), "CUDA must be available")

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
  n = 1L,
  coefficient_dim = 2L,
  null_dim = 2L,
  penalty_count = 1L,
  X = matrix(c(1, 0), nrow = 1L, ncol = 2L),
  constraint_mode = "identity",
  constraint_nullspace = NULL,
  gram_matrix = matrix(c(1, 0, 0, 0), nrow = 2L, ncol = 2L),
  nullspace_gram_matrix = NULL,
  penalty_blocks = list(penalty_1 = matrix(0, nrow = 2L, ncol = 2L)),
  penalty_offsets_zero_based = 0L,
  penalty_ranks = 0L,
  penalty_sp_indices_zero_based = 0L,
  penalty_sp_labels = "sp1",
  H = NULL,
  weights_policy = "none-or-unit",
  offset_policy = "none-or-zero"
)

runtime <- fixed_sp_cuda_runtime_create(0L)
handle <- NULL
token <- NULL
on.exit({
  if (!is.null(token)) {
    try(fixed_sp_cuda_residual_release(token), silent = TRUE)
    try(fixed_sp_cuda_residual_free(token), silent = TRUE)
  }
  if (!is.null(handle)) {
    try(fixed_sp_cuda_prepared_free(handle), silent = TRUE)
  }
  if (!is.null(runtime)) {
    try(fixed_sp_cuda_runtime_free(runtime), silent = TRUE)
  }
}, add = TRUE)

fixed_sp_cuda_runtime_reserve(
  runtime, n = 1L, null_dim = 2L, target_count = 1L,
  penalty_count = 1L, augmented_rows = 1L
)
handle <- fixed_sp_cuda_prepared_create(runtime, dto)
routes <- c("AUGMENTED_QR", "AUGMENTED_SVD")
errors <- lapply(seq_along(routes), function(index) {
  result <- tryCatch(
    fixed_sp_cuda_solve_batch(
      handle, matrix(1, nrow = 1L, ncol = 1L),
      matrix(0, nrow = 1L, ncol = 1L), routes[[index]],
      sha(as.character(7L + index)), outputs = "residuals"
    ),
    error = identity
  )
  if (!inherits(result, "error")) {
    token <<- result
    fail(paste(routes[[index]], "accepted an augmented system with rows < q"))
  }
  result
})
messages <- vapply(errors, conditionMessage, character(1L))
expected_message <- "fixed-sp augmented system requires rows >= q"
assert_true(
  identical(messages, rep(expected_message, length(routes))),
  paste0(
    "stable routes must reject rows < q before cuSOLVER; observed=",
    paste(paste(routes, messages, sep = ":"), collapse = ";")
  )
)
prepared_info <- fixed_sp_cuda_prepared_info(handle)
assert_true(
  identical(prepared_info$output_slot_leased, FALSE) &&
    identical(prepared_info$output_slot_state, "free"),
  "geometry rejection restores the prepared output slot"
)

fixed_sp_cuda_prepared_free(handle)
handle <- NULL
fixed_sp_cuda_runtime_free(runtime)
runtime <- NULL

cat("PASS Phase 3C fixed-sp stable geometry; routes:",
    paste(routes, collapse = ","), "\n")
