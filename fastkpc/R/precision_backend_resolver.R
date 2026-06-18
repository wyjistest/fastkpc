source("fastkpc/R/backend_routing_policy.R")
source("fastkpc/R/mgcv_extract_compatibility_envelope.R")

fastkpc_precision_runtime_capabilities <- function() {
  list(
    R_version = paste(R.version$major, R.version$minor, sep = "."),
    mgcv_version = if (requireNamespace("mgcv", quietly = TRUE)) {
      as.character(utils::packageVersion("mgcv"))
    } else {
      NA_character_
    },
    cuda_available = tryCatch(
      exists("fastkpc_cuda_available", mode = "function") &&
        isTRUE(fastkpc_cuda_available()),
      error = function(e) FALSE
    ),
    cuda_device_capability = NA_character_,
    mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
    spectral_gcv_version = "single-penalty-spectral-gcv-v1",
    setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
  )
}

fastkpc_resolve_backend_request <- function(
  precision = c("fast", "compatible", "hybrid"),
  alpha,
  tau,
  S,
  formula_class,
  penalty_count,
  family,
  link,
  setup_fingerprint,
  runtime_capabilities,
  fallback_backend = "legacy-mgcv",
  allow_canary = FALSE
) {
  precision <- match.arg(precision)
  runtime_capabilities <- runtime_capabilities %||%
    fastkpc_precision_runtime_capabilities()
  checks <- fastkpc_check_mgcv_extract_gpu_compatibility(
    observed_R_version = runtime_capabilities$R_version %||% NA_character_,
    observed_mgcv_version = runtime_capabilities$mgcv_version %||% NA_character_,
    family = family,
    link = link,
    formula_class = formula_class,
    S_size = length(S),
    penalty_count = penalty_count,
    setup_fingerprint_schema_version =
      runtime_capabilities$setup_fingerprint_schema_version %||% NA_character_,
    cuda_available = runtime_capabilities$cuda_available %||% FALSE,
    mgcvExtractGPU_backend_version =
      runtime_capabilities$mgcvExtractGPU_backend_version %||% NA_character_,
    spectral_gcv_version =
      runtime_capabilities$spectral_gcv_version %||% NA_character_,
    allow_canary = allow_canary
  )
  supported <- identical(checks$compatibility_status, "supported")
  route <- fastkpc_select_backend_route(
    precision = precision,
    S_size = length(S),
    single_penalty = as.integer(penalty_count) == 1L,
    mgcv_extract_gpu_supported = supported,
    tau = tau,
    fallback_backend = fallback_backend
  )
  if (identical(precision, "compatible") && !supported) {
    route$primary_backend <- fallback_backend
  }
  if (identical(precision, "hybrid") && !supported) {
    route$verifier_backend <- fallback_backend
  }
  if (identical(precision, "fast")) {
    route$compatibility_status <- "approximate"
    route$compatibility_action <- "run-fastSpline"
    route$fallback_reason <- ""
    route$supported_checks <- character()
    route$unsupported_checks <- character()
  } else {
    route$compatibility_status <- checks$compatibility_status
    route$compatibility_action <- checks$compatibility_action
    route$fallback_reason <- if (supported) "" else checks$warning_message
    route$supported_checks <- checks$supported_checks
    route$unsupported_checks <- checks$unsupported_checks
  }
  route$fallback_backend <- fallback_backend
  route$near_alpha_policy <- list(alpha = as.numeric(alpha), tau = as.numeric(tau))
  route$setup_fingerprint <- as.character(setup_fingerprint)
  route$runtime_capabilities <- runtime_capabilities
  route
}
