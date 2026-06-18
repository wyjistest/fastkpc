fastkpc_select_backend_route <- function(precision = c("fast", "compatible", "hybrid"),
                                         S_size,
                                         single_penalty,
                                         mgcv_extract_supported = NULL,
                                         mgcv_extract_gpu_supported = NULL,
                                         tau = log(2),
                                         fallback_backend = "legacy-mgcv",
                                         compatible_backend = "mgcvExtractGPUGCV") {
  precision <- match.arg(precision)
  if (is.null(mgcv_extract_supported)) {
    mgcv_extract_supported <- mgcv_extract_gpu_supported
  }
  supported_extract <- isTRUE(mgcv_extract_supported) &&
    isTRUE(single_penalty) && as.integer(S_size) <= 2L

  if (identical(precision, "fast")) {
    return(list(
      precision = precision,
      primary_backend = "fastSplineCUDA",
      verifier_backend = NA_character_,
      compatibility_claim = "approximate",
      canonical_replay_required = FALSE,
      tau = as.numeric(tau),
      fallback_reason = ""
    ))
  }

  if (identical(precision, "compatible")) {
    if (supported_extract) {
      primary <- compatible_backend
      reason <- ""
    } else {
      primary <- fallback_backend
      reason <- "mgcvExtract unsupported for requested S/formula/envelope"
    }
    return(list(
      precision = precision,
      primary_backend = primary,
      verifier_backend = NA_character_,
      compatibility_claim = "mgcv-setup-anchored",
      canonical_replay_required = TRUE,
      tau = as.numeric(tau),
      fallback_reason = reason
    ))
  }

  verifier <- if (supported_extract) compatible_backend else fallback_backend
  reason <- if (supported_extract) "" else
    "mgcvExtract verifier unsupported; using fallback verifier"
  list(
    precision = precision,
    primary_backend = "fastSplineCUDA",
    verifier_backend = verifier,
    compatibility_claim = "hybrid-primary-approximate-verifier-compatible",
    canonical_replay_required = TRUE,
    tau = as.numeric(tau),
    fallback_reason = reason
  )
}

fastkpc_near_alpha_trigger <- function(primary_p, alpha, tau) {
  primary_p <- pmax(as.numeric(primary_p), .Machine$double.xmin)
  alpha <- max(as.numeric(alpha), .Machine$double.xmin)
  abs(log(primary_p / alpha)) <= as.numeric(tau)
}
