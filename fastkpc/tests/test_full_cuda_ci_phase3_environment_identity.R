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
    "cusolver_deterministic_mode", "cublas_math_mode",
    "cublas_atomics_mode", "cublas_user_workspace_installed",
    "cublas_workspace_bytes", "cublas_workspace_alignment"
  )
), "static environment query exact schema")
assert_true(
  identical(identity$schema_version,
            "full-cuda-ci-phase3-environment-identity-v1") &&
    identical(identity$runtime_abi_schema_version,
              "full-cuda-ci-fixed-sp-runtime-v1") &&
    identical(identity$configuration_schema_version,
              "full-cuda-ci-fixed-sp-environment-config-v1") &&
    identical(identity$device_id, 0L) &&
    identical(identity$cusolver_deterministic_mode, "enabled") &&
    identical(identity$cublas_math_mode, "pedantic") &&
    identical(identity$cublas_atomics_mode, "not_allowed") &&
    isTRUE(identity$cublas_user_workspace_installed) &&
    identical(identity$cublas_workspace_bytes, 16777216) &&
    identical(identity$cublas_workspace_alignment, 256),
  "static environment query reports authenticated runtime configuration"
)
assert_true(grepl("^GPU-[0-9a-f]{32}$", identity$gpu_uuid),
            "static environment query reports canonical GPU UUID")

cat("PASS Phase 3 static environment identity (zero fixed-SP resources)\n")
