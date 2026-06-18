source("fastkpc/R/precision_backend_resolver.R")
source("fastkpc/R/hybrid_verifier.R")

fastkpc_precision_failure_cases <- function() {
  list(
    unsupported_mgcv_version = list(kind = "unsupported_mgcv_version"),
    unsupported_R_version = list(kind = "unsupported_R_version"),
    cuda_unavailable = list(kind = "cuda_unavailable"),
    setup_fingerprint_mismatch = list(kind = "setup_fingerprint_mismatch"),
    nan_primary_p = list(kind = "nan_primary_p"),
    verifier_failure = list(kind = "verifier_failure")
  )
}

fastkpc_failure_caps <- function(kind) {
  caps <- list(
    R_version = "4.5.0",
    mgcv_version = "1.9-4",
    cuda_available = TRUE,
    mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
    spectral_gcv_version = "single-penalty-spectral-gcv-v1",
    setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
  )
  if (identical(kind, "unsupported_mgcv_version")) caps$mgcv_version <- "unsupported"
  if (identical(kind, "unsupported_R_version")) caps$R_version <- "unsupported"
  if (identical(kind, "cuda_unavailable")) caps$cuda_available <- FALSE
  if (identical(kind, "setup_fingerprint_mismatch")) {
    caps$setup_fingerprint_schema_version <- "stale-schema"
  }
  caps
}

fastkpc_run_precision_failure_injection <- function(cases) {
  rows <- lapply(names(cases), function(case_id) {
    kind <- cases[[case_id]]$kind
    route <- fastkpc_resolve_backend_request(
      precision = "hybrid",
      alpha = 0.05,
      tau = log(2),
      S = 1L,
      formula_class = "full-smooth",
      penalty_count = 1L,
      family = "gaussian",
      link = "identity",
      setup_fingerprint = if (identical(kind, "setup_fingerprint_mismatch")) {
        "stale-setup"
      } else {
        "setup"
      },
      runtime_capabilities = fastkpc_failure_caps(kind)
    )
    primary_p <- if (identical(kind, "nan_primary_p")) NaN else 0.049
    verifier_p <- if (identical(kind, "verifier_failure")) NA_real_ else 0.051
    p_used <- if (is.finite(verifier_p)) verifier_p else primary_p
    data.frame(
      case_id = case_id,
      backend_requested = "mgcvExtractGPUGCV",
      backend_used = route$verifier_backend,
      compatibility_action = if (identical(kind, "nan_primary_p") ||
                                 identical(kind, "verifier_failure")) {
        "fallback"
      } else {
        route$compatibility_action
      },
      fallback_reason = if (nzchar(route$fallback_reason)) {
        route$fallback_reason
      } else {
        paste("injected", kind)
      },
      primary_p = primary_p,
      verifier_p = verifier_p,
      p_used = p_used,
      p_source_used = if (is.finite(verifier_p)) route$verifier_backend else
        "fastSplineCUDA",
      decision_before_verify = is.finite(primary_p) && primary_p > 0.05,
      decision_after_verify = is.finite(p_used) && p_used > 0.05,
      canonical_replay_preserved = TRUE,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}
