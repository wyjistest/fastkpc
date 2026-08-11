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
    message
  )
  invisible(error)
}

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP fixed-sp CUDA residual certificate\n")
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
  "fixed-sp certificate witnesses are missing"
)

resource_snapshot <- function() {
  .Call("C_fixed_sp_cuda_test_resource_snapshot", PACKAGE = "fastkpc_cuda")
}
resource_before <- resource_snapshot()

setup_key <-
  "0185fd19057086a6c573cb43343e509a19ddd9ce040de5516fc34cc02f8caa0e"
target_keys <- c(
  "f18995bafb14ad113e6c2a1d06960db2701f28e3734bde43f6cdeab75dd9d7f9",
  "0c718a26c53a1ab297ee11c658d09e96b3927f0e80f6899dc35f448f1c889ba9"
)
setup <- readRDS(setup_index_path)[[setup_key]]
target_index <- readRDS(target_index_path)
target_rows <- target_index[
  match(target_keys, target_index$residual_key_sha256), , drop = FALSE
]
assert_true(
  !anyNA(target_rows$residual_key_sha256) &&
    identical(target_rows$residual_key_sha256, target_keys) &&
    all(target_rows$prepared_s_key_sha256 == setup_key),
  "fixed-sp certificate target identity changed"
)

data <- as.matrix(readRDS(data_path))
storage.mode(data) <- "double"
Y <- unname(data[, target_rows$target, drop = FALSE])
SP <- matrix(
  vapply(target_rows$selected_sp, function(value) {
    as.double(value[[1L]])
  }, numeric(1L)),
  nrow = 1L
)
oracle <- lapply(seq_along(target_keys), function(index) {
  fastkpc_mgcv_magic_fixed_sp_from_prepared(
    prepared_setup = setup,
    target_state = list(
      row = target_rows[index, , drop = FALSE],
      y = Y[, index]
    )
  )
})
assert_true(
  all(vapply(oracle, function(value) {
    isTRUE(value$authoritative) &&
      identical(value$solve_source, "mgcv-C-magic-from-prepared-s")
  }, logical(1L))),
  "fixed-sp certificate oracle is not authoritative mgcv C_magic"
)
oracle_coefficients <- do.call(cbind, lapply(oracle, `[[`, "coefficients"))
oracle_residuals <- do.call(cbind, lapply(oracle, `[[`, "residuals"))

dto <- fastkpc_full_cuda_fixed_sp_native_dto(setup)
runtime <- fixed_sp_cuda_runtime_create(0L)
handle <- NULL
active_token <- NULL
on.exit({
  if (!is.null(active_token)) {
    try(fixed_sp_cuda_residual_release(active_token), silent = TRUE)
    try(fixed_sp_cuda_residual_free(active_token), silent = TRUE)
  }
  if (!is.null(handle)) {
    try(fixed_sp_cuda_prepared_free(handle), silent = TRUE)
  }
  if (!is.null(runtime)) {
    try(fixed_sp_cuda_runtime_free(runtime), silent = TRUE)
  }
}, add = TRUE)

fixed_sp_cuda_runtime_reserve(
  runtime, nrow(Y), dto$null_dim, ncol(Y), dto$penalty_count,
  nrow(Y) + dto$null_dim
)
handle <- fixed_sp_cuda_prepared_create(runtime, dto)

p_value <- function(residuals) {
  fastkpc_cuda_legacy_dcov_gamma_cpp_oracle(
    residuals[, 1L], residuals[, 2L], numCol = 35L, index = 1
  )$p.value
}
oracle_p <- p_value(oracle_residuals)

run_certified <- function(route) {
  active_token <<- fixed_sp_cuda_solve_certified_batch_for_test(
    handle, Y, SP, rep(route, ncol(Y)), target_keys,
    outputs = c("coefficients", "residuals"),
    refinement_max_iterations = 4L,
    backward_error_tolerance = 1e-15,
    fail_closed = TRUE
  )
  info <- fixed_sp_cuda_residual_info(active_token)
  shadow <- fixed_sp_cuda_materialize_shadow(
    active_token, c("coefficients", "residuals")
  )
  fixed_sp_cuda_residual_release(active_token)
  fixed_sp_cuda_residual_free(active_token)
  active_token <<- NULL
  list(info = info, shadow = shadow)
}

routes <- c("CHOLESKY_BATCHED", "AUGMENTED_QR", "AUGMENTED_SVD")
expected_status <- c(
  CHOLESKY_BATCHED = "OK_CHOLESKY_BATCHED",
  AUGMENTED_QR = "OK_AUGMENTED_QR",
  AUGMENTED_SVD = "OK_AUGMENTED_SVD"
)
expected_refinement <- c(
  CHOLESKY_BATCHED = "PRIMARY_CHOLESKY",
  AUGMENTED_QR = "PRIMARY_AUGMENTED_QR",
  AUGMENTED_SVD = "INDEPENDENT_AUGMENTED_QR"
)

for (route in routes) {
  first <- run_certified(route)
  second <- run_certified(route)
  info <- first$info
  shadow <- first$shadow

  assert_true(
    identical(info$planned_route, rep(route, ncol(Y))) &&
      identical(info$executed_route, rep(route, ncol(Y))) &&
      identical(info$solver_status,
                rep(unname(expected_status[[route]]), ncol(Y))) &&
      isTRUE(info$residual_certificate_enabled) &&
      identical(info$refinement_iterations, rep(4L, ncol(Y))) &&
      identical(info$refinement_solver,
                rep(unname(expected_refinement[[route]]), ncol(Y))) &&
      identical(info$refinement_solver_info, rep(0L, ncol(Y))) &&
      all(info$residual_certificate_passed) &&
      identical(info$residual_certificate_failure_count, 0L) &&
      all(is.finite(info$initial_kkt_backward_error)) &&
      all(is.finite(info$final_kkt_backward_error)) &&
      max(info$final_kkt_backward_error) <= 1e-15,
    paste(route, "does not satisfy its certified GPU route contract")
  )
  assert_true(
    identical(info$cpu_fallback_count, 0L) &&
      identical(info$unknown_fallback_count, 0L) &&
      identical(info$cuda_device_allocation_count_during_solve, 0L) &&
      identical(info$cuda_host_allocation_count_during_solve, 0L) &&
      identical(info$stream_create_count_during_solve, 0L) &&
      identical(info$event_create_count_during_solve, 0L) &&
      identical(info$cublas_handle_create_count_during_solve, 0L) &&
      identical(info$cusolver_handle_create_count_during_solve, 0L) &&
      identical(info$implicit_residual_d2h_count, 0L),
    paste(route, "used fallback, allocation, handle creation, or implicit D2H")
  )
  assert_true(
    max(abs(shadow$coefficients - oracle_coefficients)) <= 1e-10 &&
      max(abs(shadow$residuals - oracle_residuals)) <= 1e-10 &&
      abs(p_value(shadow$residuals) - oracle_p) <= 1e-10,
    paste(route, "does not reproduce the strict mgcv/p-value contract")
  )
  assert_true(
    identical(first$shadow$coefficients, second$shadow$coefficients) &&
      identical(first$shadow$residuals, second$shadow$residuals) &&
      identical(first$info$initial_kkt_backward_error,
                second$info$initial_kkt_backward_error) &&
      identical(first$info$final_kkt_backward_error,
                second$info$final_kkt_backward_error) &&
      identical(first$info$refinement_solver,
                second$info$refinement_solver) &&
      identical(first$info$refinement_solver_info,
                second$info$refinement_solver_info),
    paste(route, "certificate execution is not repeat-deterministic")
  )
}

active_token <- fixed_sp_cuda_solve_certified_batch_for_test(
  handle, Y[, 1L, drop = FALSE], SP[, 1L, drop = FALSE],
  "AUGMENTED_SVD", target_keys[[1L]], outputs = "residuals",
  refinement_max_iterations = 4L,
  backward_error_tolerance = 1e-20,
  fail_closed = TRUE
)
failed_info <- fixed_sp_cuda_residual_info(active_token)
assert_true(
  identical(failed_info$solver_status, "ERR_CERTIFICATE_FAILED") &&
    identical(failed_info$residual_certificate_passed, FALSE) &&
    identical(failed_info$residual_certificate_failure_count, 1L),
  "injected certificate failure did not fail closed"
)
assert_error(
  fixed_sp_cuda_materialize_shadow(active_token, "residuals"),
  "ERR_CERTIFICATE_FAILED",
  "certificate failure published a partial shadow output"
)
failed_after_materialize <- fixed_sp_cuda_residual_info(active_token)
assert_true(
  identical(failed_after_materialize$shadow_materialize_call_count, 0L) &&
    identical(failed_after_materialize$shadow_materialize_target_count, 0L) &&
    identical(failed_after_materialize$shadow_d2h_bytes, 0),
  "certificate failure changed shadow publication diagnostics"
)
fixed_sp_cuda_residual_release(active_token)
fixed_sp_cuda_residual_free(active_token)
active_token <- NULL

fixed_sp_cuda_prepared_free(handle)
handle <- NULL
fixed_sp_cuda_runtime_free(runtime)
runtime <- NULL
invisible(gc())

resource_after <- resource_snapshot()
resource_classes <- c(
  "cuda_device", "cuda_host", "stream", "event",
  "cublas_handle", "cusolver_handle", "gesvdj_info"
)
assert_true(
  all(vapply(resource_classes, function(resource) {
    field <- paste0(resource, "_active_count")
    identical(resource_after[[field]], resource_before[[field]])
  }, logical(1L))) &&
    identical(resource_after$cleanup_error_count,
              resource_before$cleanup_error_count),
  "fixed-sp certificate test leaked a CUDA resource"
)

cat("PASS fixed-sp CUDA KKT refinement and fail-closed mechanism\n")
