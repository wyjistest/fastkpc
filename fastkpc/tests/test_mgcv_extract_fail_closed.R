fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

source("fastkpc/R/mgcv_extract_compatibility_envelope.R")

supported <- fastkpc_check_mgcv_extract_gpu_compatibility(
  observed_R_version = "4.5.0",
  observed_mgcv_version = "1.9-4",
  supported_R_versions = "4.5.0",
  supported_mgcv_versions = "1.9-4",
  allow_canary = FALSE
)
assert_true(supported$compatibility_status == "supported",
            "supported versions should be supported")
assert_true(supported$compatibility_action == "run-mgcvExtractGPU",
            "supported versions should run GPU bridge")

unsupported <- fastkpc_check_mgcv_extract_gpu_compatibility(
  observed_R_version = "4.6.0",
  observed_mgcv_version = "1.10-0",
  supported_R_versions = "4.5.0",
  supported_mgcv_versions = "1.9-4",
  allow_canary = FALSE
)
assert_true(unsupported$compatibility_status == "unsupported",
            "unsupported versions should be unsupported")
assert_true(unsupported$compatibility_action == "fallback",
            "unsupported versions should fall back")
assert_true(grepl("4.6.0", unsupported$warning_message, fixed = TRUE),
            "warning should include observed R version")
assert_true(grepl("1.10-0", unsupported$warning_message, fixed = TRUE),
            "warning should include observed mgcv version")

canary <- fastkpc_check_mgcv_extract_gpu_compatibility(
  observed_R_version = "4.6.0",
  observed_mgcv_version = "1.10-0",
  supported_R_versions = "4.5.0",
  supported_mgcv_versions = "1.9-4",
  allow_canary = TRUE
)
assert_true(canary$compatibility_status == "canary",
            "allow_canary should mark unverified versions canary")
assert_true(canary$compatibility_action == "warn-and-run",
            "canary should warn and run")

cat("PASS mgcvExtractGPU fail-closed compatibility\n")
