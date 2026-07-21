source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) {
  if (!isTRUE(value)) fail(message)
}

load_fastkpc_cuda_native(rebuild = TRUE)
if (!isTRUE(fastkpc_cuda_available())) {
  cat("SKIP Phase 3 static environment identity: CUDA unavailable\n")
  quit(save = "no", status = 0L)
}

before <- .Call(
  "C_fixed_sp_cuda_test_resource_snapshot", PACKAGE = "fastkpc_cuda"
)
identity <- fastkpc_cuda_phase3_environment_identity(0L)
after <- .Call(
  "C_fixed_sp_cuda_test_resource_snapshot", PACKAGE = "fastkpc_cuda"
)

assert_true(identical(before, after),
            "static environment query creates no fixed-SP resources")
assert_true(identical(
  names(identity),
  c(
    "schema_version", "runtime_abi_schema_version",
    "configuration_schema_version", "device_id", "cuda_toolkit_version",
    "cuda_driver_version", "gpu_name", "gpu_uuid",
    "compute_capability_major", "compute_capability_minor", "sm_count",
    "cusolver_deterministic_mode_required", "cublas_math_mode_required",
    "cublas_atomics_mode_required", "cublas_user_workspace_required",
    "cublas_workspace_bytes_required",
    "cublas_workspace_min_alignment_required"
  )
), "static environment policy query exact schema")
assert_true(
  identical(identity$schema_version,
            "full-cuda-ci-phase3-environment-policy-v1") &&
    identical(identity$runtime_abi_schema_version,
              "full-cuda-ci-fixed-sp-runtime-v1") &&
    identical(identity$configuration_schema_version,
              "full-cuda-ci-fixed-sp-environment-policy-v1") &&
    identical(identity$device_id, 0L) &&
    identical(identity$cusolver_deterministic_mode_required, "enabled") &&
    identical(identity$cublas_math_mode_required, "pedantic") &&
    identical(identity$cublas_atomics_mode_required, "not_allowed") &&
    isTRUE(identity$cublas_user_workspace_required) &&
    identical(identity$cublas_workspace_bytes_required, 16777216) &&
    identical(identity$cublas_workspace_min_alignment_required, 256),
  "static environment query reports immutable runtime policy"
)
assert_true(grepl("^GPU-[0-9a-f]{32}$", identity$gpu_uuid),
            "static environment query reports canonical GPU UUID")

cat("PASS Phase 3 static environment identity (zero fixed-SP resources)\n")
