source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_error <- function(expression, pattern, message) {
  error <- tryCatch(force(expression), error = identity)
  assert_true(
    inherits(error, "error") &&
      grepl(pattern, conditionMessage(error), fixed = TRUE),
    message
  )
}

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP exact GPU fixed-SP evaluator\n")
  quit(save = "no", status = 0L)
}

load_fastkpc_cuda_native()
assert_true(fastkpc_cuda_available(), "CUDA must be available")

setup_index_path <- file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1",
  "prepared_s_setup_index.rds"
)
target_index_path <- file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1",
  "target_state_index.rds"
)
data_path <- file.path(
  "fastkpc", "artifacts", "kpc_tprs_real_zhu",
  "cancer_RD-causalDiscoveryInput.rds"
)
assert_true(
  all(file.exists(c(setup_index_path, target_index_path, data_path))),
  "exact fixed-SP evaluator witnesses are missing"
)

setup_key <-
  "0185fd19057086a6c573cb43343e509a19ddd9ce040de5516fc34cc02f8caa0e"
target_keys <- c(
  "exact-fixed-evaluator-target-1",
  "exact-fixed-evaluator-target-2",
  "exact-fixed-evaluator-target-3"
)
residual_keys <- c(
  "f18995bafb14ad113e6c2a1d06960db2701f28e3734bde43f6cdeab75dd9d7f9",
  "0c718a26c53a1ab297ee11c658d09e96b3927f0e80f6899dc35f448f1c889ba9",
  "01be165cdeda37e52d372ae714346973f0b73cc64e829ab31425ed3a1148644e"
)
setup <- readRDS(setup_index_path)[[setup_key]]
target_index <- readRDS(target_index_path)
target_rows <- target_index[
  match(residual_keys, target_index$residual_key_sha256), , drop = FALSE
]
assert_true(
  !anyNA(target_rows$residual_key_sha256) &&
    all(target_rows$prepared_s_key_sha256 == setup_key),
  "exact fixed-SP evaluator target identity changed"
)

data <- as.matrix(readRDS(data_path))
storage.mode(data) <- "double"
Y <- unname(data[, target_rows$target, drop = FALSE])
SP <- matrix(
  vapply(target_rows$selected_sp, function(value) {
    as.double(value[[1L]])
  }, numeric(1L)),
  nrow = length(setup$penalty_blocks)
)
oracle <- lapply(seq_len(ncol(Y)), function(index) {
  fastkpc_mgcv_magic_fixed_sp_from_prepared(
    prepared_setup = setup,
    target_state = list(
      row = target_rows[index, , drop = FALSE],
      y = Y[, index]
    )
  )
})
oracle_residuals <- do.call(cbind, lapply(oracle, `[[`, "residuals"))

geometry <- full_cuda_ci_native_geometry_prepare_native(
  setup$X, setup$penalty_blocks, as.integer(setup$penalty_offsets),
  as.integer(setup$penalty_ranks)
)
prepared <- list(
  X = unname(setup$X),
  magic_qr_packed = unname(geometry$magic_qr_packed),
  magic_tau = unname(geometry$magic_tau),
  magic_r = unname(geometry$magic_r),
  magic_pivot = as.integer(geometry$magic_pivot),
  penalty_roots = lapply(geometry$penalty_roots, unname),
  penalty_matrices = lapply(geometry$penalty_matrices, unname),
  penalty_ranks = as.integer(setup$penalty_ranks),
  initial_log_sp = unname(geometry$initial_log_sp)
)

handle <- full_cuda_ci_multi_penalty_gcv_prepared_create_native(
  prepared, ncol(Y)
)
active_token <- NULL
small_handle <- NULL
small_token <- NULL
on.exit({
  if (!is.null(small_token)) {
    try(full_cuda_ci_multi_penalty_gcv_residual_release_native(small_token),
        silent = TRUE)
    try(full_cuda_ci_multi_penalty_gcv_residual_free_native(small_token),
        silent = TRUE)
  }
  if (!is.null(small_handle)) {
    try(full_cuda_ci_multi_penalty_gcv_prepared_free_native(small_handle),
        silent = TRUE)
  }
  if (!is.null(active_token)) {
    try(full_cuda_ci_multi_penalty_gcv_residual_release_native(active_token),
        silent = TRUE)
    try(full_cuda_ci_multi_penalty_gcv_residual_free_native(active_token),
        silent = TRUE)
  }
  if (!is.null(handle)) {
    try(full_cuda_ci_multi_penalty_gcv_prepared_free_native(handle),
        silent = TRUE)
  }
}, add = TRUE)

p_value <- function(residuals) {
  fastkpc_cuda_legacy_dcov_gamma_cpp_oracle(
    residuals[, 1L], residuals[, 2L], numCol = 35L, index = 1
  )$p.value
}
oracle_p <- p_value(oracle_residuals)

run_exact <- function(columns = seq_len(ncol(Y))) {
  active_token <<-
    full_cuda_ci_multi_penalty_gcv_evaluate_fixed_batch_native(
      handle, Y[, columns, drop = FALSE], log(SP[, columns, drop = FALSE]),
      target_keys[columns]
    )
  info <- full_cuda_ci_multi_penalty_gcv_residual_info_native(active_token)
  shadow <- full_cuda_ci_multi_penalty_gcv_residual_shadow_native(active_token)
  full_cuda_ci_multi_penalty_gcv_residual_release_native(active_token)
  full_cuda_ci_multi_penalty_gcv_residual_free_native(active_token)
  active_token <<- NULL
  list(info = info, shadow = shadow)
}

first <- run_exact()
second <- run_exact()
assert_true(
  isTRUE(first$info$fixed_sp_evaluation) &&
    isTRUE(first$info$device_resident) &&
    all(first$info$solver_info == 0L) &&
    all(first$info$decomposition_route == 2L) &&
    all(first$info$numerical_rank > 0L),
  "exact fixed-SP evaluator did not use a successful stable GPU SVD"
)
assert_true(
  max(abs(first$shadow$residuals - oracle_residuals)) <= 1e-10 &&
    abs(p_value(first$shadow$residuals) - oracle_p) <= 1e-10,
  "exact fixed-SP evaluator does not reproduce the mgcv contract"
)
assert_true(
  identical(first$shadow$coefficients, second$shadow$coefficients) &&
    identical(first$shadow$residuals, second$shadow$residuals) &&
    identical(first$info$solver_info, second$info$solver_info) &&
    identical(first$info$decomposition_route,
              second$info$decomposition_route),
  "exact fixed-SP evaluator is not repeat-deterministic"
)

cohort_12 <- run_exact(c(1L, 2L))
cohort_13 <- run_exact(c(1L, 3L))
cohort_21 <- run_exact(c(2L, 1L))
assert_true(
  identical(first$shadow$coefficients[, 1:2, drop = FALSE],
            cohort_12$shadow$coefficients) &&
    identical(first$shadow$residuals[, 1:2, drop = FALSE],
              cohort_12$shadow$residuals) &&
    identical(first$shadow$coefficients[, c(1L, 3L), drop = FALSE],
              cohort_13$shadow$coefficients) &&
    identical(first$shadow$residuals[, c(1L, 3L), drop = FALSE],
              cohort_13$shadow$residuals) &&
    identical(first$shadow$coefficients[, 2:1, drop = FALSE],
              cohort_21$shadow$coefficients) &&
    identical(first$shadow$residuals[, 2:1, drop = FALSE],
              cohort_21$shadow$residuals),
  "exact fixed-SP evaluator changed with cohort composition or target order"
)

set.seed(6001)
small_n <- 90L
small_p <- 8L
small_data <- matrix(stats::rnorm(small_n * small_p), small_n, small_p)
for (column in 2:small_p) {
  parents <- which(stats::runif(column - 1L) < 0.65)
  if (length(parents) > 0L) {
    small_data[, column] <- small_data[, column] +
      small_data[, parents, drop = FALSE] %*%
        stats::runif(length(parents), 0.35, 0.9)
  }
}
small_conditioning <- 3L
small_targets <- setdiff(seq_len(small_p), small_conditioning)
small_setup <- full_cuda_ci_native_setup_native(
  small_data[, small_conditioning, drop = FALSE]
)
assert_true(
  ncol(small_setup$X) == 10L && length(small_setup$penalty_blocks) == 1L,
  "small exact fixed-SP evaluator fixture changed shape"
)
small_geometry <- full_cuda_ci_native_geometry_prepare_native(
  small_setup$X, small_setup$penalty_blocks,
  as.integer(small_setup$penalty_offsets),
  as.integer(small_setup$penalty_ranks)
)
small_prepared <- list(
  X = unname(small_setup$X),
  magic_qr_packed = unname(small_geometry$magic_qr_packed),
  magic_tau = unname(small_geometry$magic_tau),
  magic_r = unname(small_geometry$magic_r),
  magic_pivot = as.integer(small_geometry$magic_pivot),
  penalty_roots = lapply(small_geometry$penalty_roots, unname),
  penalty_matrices = lapply(small_geometry$penalty_matrices, unname),
  penalty_ranks = as.integer(small_setup$penalty_ranks),
  initial_log_sp = unname(small_geometry$initial_log_sp)
)
small_Y <- unname(small_data[, small_targets, drop = FALSE])
small_log_sp <- matrix(c(
  22.8160194402389678, -2.2863757002192449, 21.1505315768868911,
  24.9261800565194562, 1.4402542868075707, -5.1760493856035419,
  2.3906587973334368
), nrow = 1L)
small_target_keys <- sprintf("%064x", small_targets)
small_handle <- full_cuda_ci_multi_penalty_gcv_prepared_create_native(
  small_prepared, ncol(small_Y)
)
run_small_exact <- function(columns = seq_len(ncol(small_Y))) {
  small_token <<-
    full_cuda_ci_multi_penalty_gcv_evaluate_fixed_batch_native(
      small_handle, small_Y[, columns, drop = FALSE],
      small_log_sp[, columns, drop = FALSE], small_target_keys[columns]
    )
  shadow <- full_cuda_ci_multi_penalty_gcv_residual_shadow_native(small_token)
  full_cuda_ci_multi_penalty_gcv_residual_release_native(small_token)
  full_cuda_ci_multi_penalty_gcv_residual_free_native(small_token)
  small_token <<- NULL
  shadow
}
small_first <- run_small_exact()
small_repeats <- lapply(seq_len(4L), function(index) run_small_exact())
small_123 <- run_small_exact(c(1L, 2L, 3L))
small_135 <- run_small_exact(c(1L, 3L, 5L))
small_312 <- run_small_exact(c(3L, 1L, 2L))
assert_true(
  all(vapply(small_repeats, function(value) {
    identical(value$coefficients, small_first$coefficients) &&
      identical(value$residuals, small_first$residuals)
  }, logical(1L))) &&
    identical(small_123$coefficients,
              small_first$coefficients[, c(1L, 2L, 3L), drop = FALSE]) &&
    identical(small_123$residuals,
              small_first$residuals[, c(1L, 2L, 3L), drop = FALSE]) &&
    identical(small_135$coefficients,
              small_first$coefficients[, c(1L, 3L, 5L), drop = FALSE]) &&
    identical(small_135$residuals,
              small_first$residuals[, c(1L, 3L, 5L), drop = FALSE]) &&
    identical(small_312$coefficients,
              small_first$coefficients[, c(3L, 1L, 2L), drop = FALSE]) &&
    identical(small_312$residuals,
              small_first$residuals[, c(3L, 1L, 2L), drop = FALSE]),
  "small exact fixed-SP evaluator changed across repeated or reordered cohorts"
)
full_cuda_ci_multi_penalty_gcv_prepared_free_native(small_handle)
small_handle <- NULL
assert_true(
  !isTRUE(full_cuda_ci_multi_penalty_gcv_prepared_info_native(
    handle
  )$residual_slot_leased),
  "exact fixed-SP evaluator did not release its output slot"
)

bad_log_sp <- log(SP)
bad_log_sp[1L] <- NaN
assert_error(
  full_cuda_ci_multi_penalty_gcv_evaluate_fixed_batch_native(
    handle, Y, bad_log_sp, target_keys
  ),
  "dimensions are inconsistent",
  "exact fixed-SP evaluator did not fail closed on non-finite log-SP"
)
assert_true(
  !isTRUE(full_cuda_ci_multi_penalty_gcv_prepared_info_native(
    handle
  )$residual_slot_leased),
  "failed exact fixed-SP evaluation leaked its output slot"
)

full_cuda_ci_multi_penalty_gcv_prepared_free_native(handle)
handle <- NULL
cat("PASS exact GPU fixed-SP evaluator\n")
