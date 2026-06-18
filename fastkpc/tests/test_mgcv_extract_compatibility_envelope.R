fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

source("fastkpc/R/mgcv_extract_compatibility_envelope.R")

env <- fastkpc_mgcv_extract_gpu_capabilities(
  observed_R_version = "4.5.0",
  observed_mgcv_version = "1.9-4",
  observed_cuda_backend_version = "mgcvExtractGPU-v1"
)

required <- c(
  "backend", "role", "supported_R_versions", "supported_mgcv_versions",
  "observed_R_version", "observed_mgcv_version",
  "supported_family", "supported_formula_classes",
  "supported_single_penalty_modes", "setup_fingerprint_schema_version",
  "cuda_backend_version", "native_same_setup_batch_version",
  "spectral_gcv_version", "compatibility_status", "compatibility_action"
)
missing <- setdiff(required, names(env))
assert_true(length(missing) == 0L,
            paste("missing capability fields:", paste(missing, collapse = ", ")))
assert_true(env$backend == "mgcvExtractGPU",
            "backend should be mgcvExtractGPU")
assert_true(env$role == "version-pinned compatibility bridge",
            "role should describe bridge boundary")
assert_true(env$supported_family == "gaussian_identity",
            "supported family should be restricted")

cat("PASS mgcvExtractGPU compatibility envelope\n")
