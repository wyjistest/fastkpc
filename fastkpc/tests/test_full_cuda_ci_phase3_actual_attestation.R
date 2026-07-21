source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_phase3_artifacts.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) {
  if (!isTRUE(value)) fail(message)
}
assert_error <- function(expression, message) {
  error <- tryCatch({
    force(expression)
    NULL
  }, error = function(error) error)
  assert_true(inherits(error, "error"), message)
}

fastkpc_full_cuda_phase3_discover_qualified_native_evidence()
if (!isTRUE(.Call("C_fastkpc_cuda_available", PACKAGE = "fastkpc_cuda"))) {
  cat("SKIP Phase 3 actual runtime attestation: CUDA unavailable\n")
  quit(save = "no", status = 0L)
}

catalog <- fastkpc_full_cuda_open_fixed_sp_catalog(
  file.path("fastkpc", "artifacts", "full_cuda_ci", "oracle_351x48_v1"),
  file.path("fastkpc", "artifacts", "full_cuda_ci", "workload_census_351x48_v1"),
  file.path("fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1"),
  file.path("fastkpc", "artifacts", "kpc_tprs_real_zhu",
            "cancer_RD-causalDiscoveryInput.rds"),
  require_full = TRUE
)
identity <- fastkpc_full_cuda_phase3_input_identity(catalog, 0L)

runtime <- fixed_sp_cuda_runtime_create(0L)
on.exit({
  try(fixed_sp_cuda_runtime_free(runtime), silent = TRUE)
}, add = TRUE)
capacities <- fastkpc_full_cuda_fixed_sp_contract()$canonical_capacities
fixed_sp_cuda_runtime_reserve(
  runtime,
  capacities$n,
  capacities$null_dim,
  1L,
  capacities$penalty_count,
  capacities$augmented_rows
)

attestation <- .fastkpc_full_cuda_phase3_validate_runtime_attestation(
  runtime, identity
)
assert_true(
  is.list(attestation) &&
    isTRUE(attestation$authenticated) &&
    identical(attestation$runtime_info$device_id, 0L) &&
    identical(
      attestation$runtime_info$cublas_workspace_bytes,
      identity$cublas_workspace_bytes_required
    ) &&
    attestation$runtime_info$cublas_workspace_alignment >=
      identity$cublas_workspace_min_alignment_required &&
    identical(
      attestation$execution_evidence$native_library_sha256,
      identity$native_library_sha256
    ) &&
    identical(
      attestation$device_evidence$gpu_uuid,
      identity$gpu_uuid
    ),
  "actual runtime attestation binds static policy, runtime facts, and native binary"
)

forged_workspace_identity <- identity
forged_workspace_identity$cublas_workspace_bytes_required <-
  forged_workspace_identity$cublas_workspace_bytes_required + 1
forged_workspace_identity$sha256 <-
  .fastkpc_full_cuda_phase3_identity_hash(forged_workspace_identity)
assert_error(
  .fastkpc_full_cuda_phase3_validate_runtime_attestation(
    runtime, forged_workspace_identity
  ),
  "workspace bytes mismatch must fail closed"
)

forged_native_identity <- identity
forged_native_identity$native_library_sha256 <- strrep("0", 64L)
forged_native_identity$sha256 <-
  .fastkpc_full_cuda_phase3_identity_hash(forged_native_identity)
assert_error(
  .fastkpc_full_cuda_phase3_validate_runtime_attestation(
    runtime, forged_native_identity
  ),
  "native library mismatch must fail closed"
)

forged_gpu_identity <- identity
forged_gpu_identity$gpu_uuid <- paste0("GPU-", strrep("f", 32L))
forged_gpu_identity$sha256 <-
  .fastkpc_full_cuda_phase3_identity_hash(forged_gpu_identity)
assert_error(
  .fastkpc_full_cuda_phase3_validate_runtime_attestation(
    runtime, forged_gpu_identity
  ),
  "static GPU identity mismatch must fail closed"
)

fixed_sp_cuda_runtime_free(runtime)

cat("PASS Phase 3 actual runtime attestation (device 0)\n")
