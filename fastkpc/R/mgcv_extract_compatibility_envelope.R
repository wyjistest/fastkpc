fastkpc_check_mgcv_extract_gpu_compatibility <- function(
  observed_R_version,
  observed_mgcv_version,
  supported_R_versions = c("4.5.0"),
  supported_mgcv_versions = c("1.9-4"),
  family = "gaussian",
  link = "identity",
  formula_class = "full-smooth",
  S_size = 1L,
  penalty_count = 1L,
  setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1",
  supported_setup_fingerprint_schema_versions = "mgcvExtractGPU-setup-v1",
  cuda_available = TRUE,
  mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
  supported_mgcvExtractGPU_backend_versions = "mgcvExtractGPU-v1",
  spectral_gcv_version = "single-penalty-spectral-gcv-v1",
  supported_spectral_gcv_versions = "single-penalty-spectral-gcv-v1",
  allow_canary = FALSE
) {
  checks <- c(
    R_version = as.character(observed_R_version) %in%
      as.character(supported_R_versions),
    mgcv_version = as.character(observed_mgcv_version) %in%
      as.character(supported_mgcv_versions),
    family = identical(as.character(family), "gaussian"),
    link = identical(as.character(link), "identity"),
    formula_class = as.character(formula_class) %in%
      c("full-smooth", "additive-smooth"),
    S_size = as.integer(S_size) <= 2L,
    penalty_count = as.integer(penalty_count) == 1L,
    setup_fingerprint_schema_version =
      as.character(setup_fingerprint_schema_version) %in%
        as.character(supported_setup_fingerprint_schema_versions),
    cuda_available = isTRUE(cuda_available),
    mgcvExtractGPU_backend_version =
      as.character(mgcvExtractGPU_backend_version) %in%
        as.character(supported_mgcvExtractGPU_backend_versions),
    spectral_gcv_version = as.character(spectral_gcv_version) %in%
      as.character(supported_spectral_gcv_versions)
  )
  unsupported <- names(checks)[!checks]
  if (length(unsupported) == 0L) {
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
    "unsupported", paste(unsupported, collapse = ","),
    "action", action
  )
  list(
    compatibility_status = status,
    compatibility_action = action,
    warning_message = warning_message,
    supported_checks = names(checks)[checks],
    unsupported_checks = unsupported
  )
}

fastkpc_check_mgcv_extract_cpu_compatibility <- function(
  observed_R_version,
  observed_mgcv_version,
  supported_R_versions = c("4.5.0"),
  supported_mgcv_versions = c("1.9-4"),
  family = "gaussian",
  link = "identity",
  formula_class = "full-smooth",
  S_size = 1L,
  penalty_count = 1L,
  setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1",
  supported_setup_fingerprint_schema_versions = "mgcvExtractGPU-setup-v1",
  spectral_gcv_version = "single-penalty-spectral-gcv-v1",
  supported_spectral_gcv_versions = "single-penalty-spectral-gcv-v1",
  allow_canary = FALSE
) {
  checks <- c(
    R_version = as.character(observed_R_version) %in%
      as.character(supported_R_versions),
    mgcv_version = as.character(observed_mgcv_version) %in%
      as.character(supported_mgcv_versions),
    family = identical(as.character(family), "gaussian"),
    link = identical(as.character(link), "identity"),
    formula_class = as.character(formula_class) %in%
      c("full-smooth", "additive-smooth"),
    S_size = as.integer(S_size) <= 2L,
    penalty_count = as.integer(penalty_count) == 1L,
    setup_fingerprint_schema_version =
      as.character(setup_fingerprint_schema_version) %in%
        as.character(supported_setup_fingerprint_schema_versions),
    spectral_gcv_version = as.character(spectral_gcv_version) %in%
      as.character(supported_spectral_gcv_versions)
  )
  unsupported <- names(checks)[!checks]
  if (length(unsupported) == 0L) {
    status <- "supported"
    action <- "run-mgcvExtractCPU"
  } else if (isTRUE(allow_canary)) {
    status <- "canary"
    action <- "warn-and-run"
  } else {
    status <- "unsupported"
    action <- "fallback"
  }
  warning_message <- if (identical(status, "supported")) "" else paste(
    "mgcvExtractCPU compatibility envelope mismatch:",
    "observed R", observed_R_version,
    "observed mgcv", observed_mgcv_version,
    "supported R", paste(supported_R_versions, collapse = ","),
    "supported mgcv", paste(supported_mgcv_versions, collapse = ","),
    "unsupported", paste(unsupported, collapse = ","),
    "action", action
  )
  list(
    compatibility_status = status,
    compatibility_action = action,
    warning_message = warning_message,
    supported_checks = names(checks)[checks],
    unsupported_checks = unsupported
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
