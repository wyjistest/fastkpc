source("fastkpc/R/mgcv_extract_oracle.R")
source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_error <- function(expr, pattern, message) {
  error <- tryCatch({
    force(expr)
    NULL
  }, error = identity)
  assert_true(
    inherits(error, "error") && grepl(pattern, conditionMessage(error), fixed = TRUE),
    message
  )
}

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP mgcvExtractGPU native fixed-sp solve: FASTKPC_RUN_CUDA_TESTS != 1\n")
  quit(save = "no", status = 0)
}

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP mgcvExtractGPU native fixed-sp solve: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

build_fastkpc_cuda_native(rebuild = TRUE)
if (!fastkpc_cuda_available()) {
  cat("SKIP mgcvExtractGPU native fixed-sp solve: CUDA unavailable\n")
  quit(save = "no", status = 0)
}

set.seed(245)
n <- 64
s1 <- stats::runif(n, -2, 2)
s2 <- stats::runif(n, -2, 2)
y <- sin(s1) + cos(s2) + stats::rnorm(n, sd = 0.05)
data <- data.frame(y = y, s1 = s1, s2 = s2)
formula <- y ~ s(s1, s2, k = 12, bs = "tp")
sp <- 0.55

setup <- fastkpc_mgcv_extract_setup(
  formula = formula,
  data = data,
  sp = sp,
  target = 1L,
  S = c(2L, 3L),
  k = 12L,
  bs = "tp"
)
handle <- fastkpc_mgcv_extract_gpu_setup_handle(setup)
cpu <- fastkpc_mgcv_extract_gpu_solve_handle_fixed_sp(handle)
resource_snapshot <- function() {
  .Call("C_fixed_sp_cuda_test_resource_snapshot", PACKAGE = "fastkpc_cuda")
}
success_resources_before <- resource_snapshot()
gpu <- fastkpc_mgcv_extract_gpu_solve_handle_fixed_sp_cuda(handle)
success_resources_after <- resource_snapshot()
resource_lifecycle_verbs <- list(
  cuda_device = c("allocate", "free"),
  cuda_host = c("allocate", "free"),
  stream = c("create", "destroy"),
  event = c("create", "destroy"),
  cublas_handle = c("create", "destroy"),
  cusolver_handle = c("create", "destroy"),
  gesvdj_info = c("create", "destroy")
)
assert_balanced_resource_lifecycle <- function(before, after, message) {
  assert_true(
    all(vapply(names(resource_lifecycle_verbs), function(resource) {
      verbs <- resource_lifecycle_verbs[[resource]]
      acquire_success <- paste(resource, verbs[[1L]], "success_count", sep = "_")
      teardown_success <- paste(resource, verbs[[2L]], "success_count", sep = "_")
      active <- paste(resource, "active_count", sep = "_")
      ownership_indeterminate <- paste(
        resource, "ownership_indeterminate_count", sep = "_"
      )
      acquired <- after[[acquire_success]] - before[[acquire_success]]
      torn_down <- after[[teardown_success]] - before[[teardown_success]]
      acquired > 0L &&
        acquired == torn_down &&
        after[[active]] == before[[active]] &&
        after[[ownership_indeterminate]] == before[[ownership_indeterminate]]
    }, logical(1L))) &&
      after$cleanup_error_count == before$cleanup_error_count,
    message
  )
}
assert_balanced_resource_lifecycle(
  success_resources_before,
  success_resources_after,
  "legacy transient adapter releases every CUDA resource"
)
xtx_resources_before <- resource_snapshot()
drifted_xtx_handle <- handle
drifted_xtx_handle$XtX_null[1L, 1L] <-
  drifted_xtx_handle$XtX_null[1L, 1L] + 1e-6
assert_error(
  fastkpc_mgcv_extract_gpu_solve_handle_fixed_sp_cuda(drifted_xtx_handle),
  "XtX_null does not match independently recomputed crossprod(X_null)",
  "legacy adapter rejects caller XtX_null drift"
)
assert_true(
  identical(resource_snapshot(), xtx_resources_before),
  "XtX_null drift fails before acquiring CUDA resources"
)
xty_resources_before <- resource_snapshot()
drifted_xty_handle <- handle
drifted_xty_handle$Xty_null[1L] <-
  drifted_xty_handle$Xty_null[1L] + 1e-6
assert_error(
  fastkpc_mgcv_extract_gpu_solve_handle_fixed_sp_cuda(drifted_xty_handle),
  "Xty_null does not match independently recomputed crossprod(X_null, y)",
  "legacy adapter rejects caller Xty_null drift"
)
assert_true(
  identical(resource_snapshot(), xty_resources_before),
  "Xty_null drift fails before acquiring CUDA resources"
)
overflow_resources_before <- resource_snapshot()
overflow_handle <- handle
overflow_scale <- sqrt(.Machine$double.xmax) * 2
overflow_handle$X[] <- 0
overflow_handle$Z[] <- 0
overflow_handle$X[1L, 1L] <- overflow_scale
overflow_handle$Z[1L, 1L] <- overflow_scale
assert_error(
  fastkpc_mgcv_extract_gpu_solve_handle_fixed_sp_cuda(overflow_handle),
  "recomputed X_null contains missing or infinite values",
  "legacy adapter rejects nonfinite derived X_null"
)
assert_true(
  identical(resource_snapshot(), overflow_resources_before),
  "nonfinite derived X_null fails before acquiring CUDA resources"
)
coefficient_overflow_scale <- .Machine$double.xmax / 2
coefficient_overflow_handle <- list(
  X = matrix(rep(0.25 / coefficient_overflow_scale, 2L), nrow = 1L),
  y = 2,
  Z = matrix(rep(coefficient_overflow_scale, 2L), ncol = 1L),
  XtX_null = matrix(0.25, nrow = 1L),
  penalty_null = matrix(0, nrow = 1L),
  Xty_null = 1
)
coefficient_overflow_resources_before <- resource_snapshot()
assert_error(
  mgcv_extract_gpu_solve_handle_fixed_sp_cuda(coefficient_overflow_handle),
  "reconstructed coefficients contains missing or infinite values",
  "legacy adapter rejects nonfinite reconstructed p-space coefficients"
)
coefficient_overflow_resources_after <- resource_snapshot()
assert_balanced_resource_lifecycle(
  coefficient_overflow_resources_before,
  coefficient_overflow_resources_after,
  "coefficient reconstruction failure releases every CUDA resource"
)
device_count <- .Call(
  "C_fixed_sp_cuda_test_device_count", PACKAGE = "fastkpc_cuda"
)
if (device_count >= 2L) {
  .Call("C_fixed_sp_cuda_test_set_device", 1L, PACKAGE = "fastkpc_cuda")
  on.exit(
    try(.Call("C_fixed_sp_cuda_test_set_device", 0L,
              PACKAGE = "fastkpc_cuda"), silent = TRUE),
    add = TRUE
  )
  device_one_gpu <- fastkpc_mgcv_extract_gpu_solve_handle_fixed_sp_cuda(handle)
  current_device <- .Call(
    "C_fixed_sp_cuda_test_get_device", PACKAGE = "fastkpc_cuda"
  )
  assert_true(
    identical(current_device, 1L),
    "legacy adapter preserves the caller current CUDA device"
  )
  assert_true(
    max(abs(device_one_gpu$residuals - cpu$residuals)) < 1e-7,
    "legacy adapter solves correctly on the caller current CUDA device"
  )
  .Call("C_fixed_sp_cuda_test_set_device", 0L, PACKAGE = "fastkpc_cuda")
}
api_gpu <- fastkpc_mgcv_extract_gpu_fixed_sp(
  formula = formula,
  data = data,
  sp = sp,
  target = 1L,
  S = c(2L, 3L),
  k = 12L,
  bs = "tp",
  device = "cuda",
  allow_cpu_fallback = FALSE,
  solve_strategy = "handle"
)

assert_true(identical(gpu$backend_family, "mgcvExtractGPU"),
            "native solve backend should identify mgcvExtractGPU")
assert_true(identical(gpu$mode, "fixed-sp-native-gpu-solve"),
            "native solve mode should identify CUDA solve")
assert_true(identical(gpu$solve_source, "mgcvExtractGPU-native-fixed-sp"),
            "native solve source should be explicit")
assert_true(identical(gpu$used_device, "cuda"),
            "native solve should report cuda")
assert_true(isTRUE(gpu$native_gpu_solve_used),
            "native solve should report native GPU usage")
assert_true(identical(gpu$diagnostics$runtime_version,
                      "full-cuda-ci-fixed-sp-runtime-v1"),
            "legacy single delegates to stable runtime")
assert_true(isTRUE(gpu$diagnostics$compatibility_transient_context),
            "legacy adapter declares transient context")
assert_true(identical(gpu$diagnostics$planned_route, "AUGMENTED_SVD") &&
              identical(gpu$diagnostics$executed_route, "AUGMENTED_SVD"),
            "unclassified compatibility call conservatively uses SVD")
assert_true(identical(gpu$diagnostics$solver_status, "OK_AUGMENTED_SVD"),
            "compatibility adapter reports stable SVD success")
assert_true(gpu$diagnostics$cpu_fallback_count == 0L,
            "compatibility adapter has no CPU fallback")
assert_true(gpu$diagnostics$rhs_device_build_count == 1L &&
              identical(gpu$diagnostics$rhs_authority,
                        "cuda-x0-transpose-y") &&
              isTRUE(gpu$diagnostics$full_cuda_data_plane),
            "compatibility adapter builds the production RHS on CUDA")
assert_true(length(gpu$coefficients) == length(cpu$coefficients),
            "native solve coefficient length")
assert_true(max(abs(gpu$theta - cpu$theta)) < 1e-7,
            "native theta should match CPU handle solve")
assert_true(max(abs(gpu$coefficients - cpu$coefficients)) < 1e-7,
            "native coefficients should match CPU handle solve")
assert_true(max(abs(gpu$fitted - cpu$fitted)) < 1e-7,
            "native fitted values should match CPU handle solve")
assert_true(max(abs(gpu$residuals - cpu$residuals)) < 1e-7,
            "native residuals should match CPU handle solve")
assert_true(abs(gpu$rss - cpu$rss) < 1e-7,
            "native RSS should match CPU handle solve")
assert_true(identical(api_gpu$used_device, "cuda"),
            "top-level API should use CUDA handle solve")
assert_true(isTRUE(api_gpu$native_gpu_solve_used),
            "top-level API should report native GPU solve usage")
assert_true(!isTRUE(api_gpu$fallback_used),
            "top-level API should not report fallback when CUDA solve succeeds")
assert_true(identical(api_gpu$solve_source, "mgcvExtractGPU-native-fixed-sp"),
            "top-level API should preserve native solve source")
assert_true(max(abs(api_gpu$residuals - cpu$residuals)) < 1e-7,
            "top-level CUDA residuals should match CPU handle solve")

cat("PASS mgcvExtractGPU native fixed-sp solve\n")
