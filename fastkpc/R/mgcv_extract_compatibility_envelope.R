fastkpc_check_mgcv_extract_gpu_compatibility <- function(
  observed_R_version,
  observed_mgcv_version,
  supported_R_versions,
  supported_mgcv_versions,
  allow_canary = FALSE
) {
  r_ok <- as.character(observed_R_version) %in% as.character(supported_R_versions)
  mgcv_ok <- as.character(observed_mgcv_version) %in%
    as.character(supported_mgcv_versions)
  if (r_ok && mgcv_ok) {
    status <- "supported"
    action <- "run-mgcvExtractGPU"
  } else if (isTRUE(allow_canary)) {
    status <- "canary"
    action <- "warn-and-run"
  } else {
    status <- "unsupported"
    action <- "fallback"
  }
  warning_message <- if (identical(status, "supported")) "" else paste(
    "mgcvExtractGPU compatibility envelope mismatch:",
    "observed R", observed_R_version,
    "observed mgcv", observed_mgcv_version,
    "supported R", paste(supported_R_versions, collapse = ","),
    "supported mgcv", paste(supported_mgcv_versions, collapse = ","),
    "action", action
  )
  list(
    compatibility_status = status,
    compatibility_action = action,
    warning_message = warning_message
  )
}

fastkpc_mgcv_extract_gpu_capabilities <- function(
  observed_R_version = paste(R.version$major, R.version$minor, sep = "."),
  observed_mgcv_version = if (requireNamespace("mgcv", quietly = TRUE)) {
    as.character(utils::packageVersion("mgcv"))
  } else {
    NA_character_
  },
  observed_cuda_backend_version = "mgcvExtractGPU-v1",
  supported_R_versions = c("4.5.0"),
  supported_mgcv_versions = c("1.9-4")
) {
  check <- fastkpc_check_mgcv_extract_gpu_compatibility(
    observed_R_version = observed_R_version,
    observed_mgcv_version = observed_mgcv_version,
    supported_R_versions = supported_R_versions,
    supported_mgcv_versions = supported_mgcv_versions,
    allow_canary = FALSE
  )
  list(
    backend = "mgcvExtractGPU",
    role = "version-pinned compatibility bridge",
    supported_R_versions = supported_R_versions,
    supported_mgcv_versions = supported_mgcv_versions,
    observed_R_version = as.character(observed_R_version),
    observed_mgcv_version = as.character(observed_mgcv_version),
    supported_family = "gaussian_identity",
    supported_formula_classes = c("full-smooth", "additive-smooth"),
    supported_single_penalty_modes = c("|S| = 1", "|S| = 2"),
    setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1",
    cuda_backend_version = observed_cuda_backend_version,
    native_same_setup_batch_version = "native-same-setup-repeated-cuda-solve-v1",
    spectral_gcv_version = "single-penalty-spectral-gcv-v1",
    compatibility_status = check$compatibility_status,
    compatibility_action = check$compatibility_action,
    warning_message = check$warning_message
  )
}
