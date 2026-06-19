fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

source("fastkpc/R/precision_backend_resolver.R")

caps <- list(
  R_version = "4.5.0",
  mgcv_version = "1.9-4",
  cuda_available = TRUE,
  cuda_device_capability = "8.9",
  mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
  spectral_gcv_version = "single-penalty-spectral-gcv-v1",
  setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
)

fast_route <- fastkpc_resolve_backend_request(
  precision = "fast", alpha = 0.05, tau = log(2), S = c(1L, 2L),
  formula_class = "full-smooth", penalty_count = 1L,
  family = "gaussian", link = "identity",
  setup_fingerprint = "setup-a", runtime_capabilities = caps
)
assert_true(fast_route$primary_backend == "fastSplineCUDA",
            "fast mode must use fastSplineCUDA")
assert_true(is.na(fast_route$verifier_backend),
            "fast mode must not select a verifier")
assert_true(fast_route$compatibility_claim == "approximate",
            "fast mode must be approximate")

compatible <- fastkpc_resolve_backend_request(
  precision = "compatible", alpha = 0.05, tau = log(2), S = c(1L, 2L),
  formula_class = "full-smooth", penalty_count = 1L,
  family = "gaussian", link = "identity",
  setup_fingerprint = "setup-a", runtime_capabilities = caps
)
assert_true(compatible$primary_backend == "mgcvExtractGPUGCV",
            "supported compatible mode should select mgcvExtractGPUGCV")
assert_true(compatible$compatibility_status == "supported",
            "supported compatible mode should be supported")

bad_family <- fastkpc_resolve_backend_request(
  precision = "compatible", alpha = 0.05, tau = log(2), S = 1L,
  formula_class = "full-smooth", penalty_count = 1L,
  family = "binomial", link = "logit",
  setup_fingerprint = "setup-b", runtime_capabilities = caps
)
assert_true(bad_family$primary_backend == "legacy-mgcv",
            "unsupported compatible mode must fall back")
assert_true(bad_family$compatibility_action == "fallback",
            "unsupported compatible mode must fail closed")
assert_true(grepl("family", bad_family$fallback_reason, fixed = TRUE),
            "fallback reason should name family")

hybrid_bad_cuda <- fastkpc_resolve_backend_request(
  precision = "hybrid", alpha = 0.05, tau = log(2), S = 1L,
  formula_class = "full-smooth", penalty_count = 1L,
  family = "gaussian", link = "identity",
  setup_fingerprint = "setup-c",
  runtime_capabilities = modifyList(caps, list(cuda_available = FALSE))
)
assert_true(hybrid_bad_cuda$primary_backend == "fastSplineCUDA",
            "hybrid primary remains fastSplineCUDA")
assert_true(hybrid_bad_cuda$verifier_backend == "legacy-mgcv",
            "unsupported hybrid verifier should fall back")
assert_true(hybrid_bad_cuda$canonical_replay_required,
            "hybrid must require canonical replay")

canary_gpu <- fastkpc_resolve_backend_request(
  precision = "compatible", alpha = 0.05, tau = log(2), S = 1L,
  formula_class = "full-smooth", penalty_count = 1L,
  family = "gaussian", link = "identity",
  setup_fingerprint = "setup-canary",
  runtime_capabilities = modifyList(caps, list(R_version = "4.4.1")),
  allow_canary = TRUE,
  execution_engine = "cuda"
)
assert_true(canary_gpu$compatibility_status == "canary",
            "canary route should preserve canary status")
assert_true(canary_gpu$compatibility_action == "warn-and-run",
            "canary route should warn and run")
assert_true(canary_gpu$primary_backend == "mgcvExtractGPUGCV",
            "canary compatible route should execute mgcvExtractGPUGCV")

canary_hybrid <- fastkpc_resolve_backend_request(
  precision = "hybrid", alpha = 0.05, tau = log(2), S = 1L,
  formula_class = "full-smooth", penalty_count = 1L,
  family = "gaussian", link = "identity",
  setup_fingerprint = "setup-canary-hybrid",
  runtime_capabilities = modifyList(caps, list(mgcv_version = "1.9.1")),
  allow_canary = TRUE,
  execution_engine = "cuda"
)
assert_true(canary_hybrid$primary_backend == "fastSplineCUDA",
            "canary hybrid primary remains fastSplineCUDA")
assert_true(canary_hybrid$verifier_backend == "mgcvExtractGPUGCV",
            "canary hybrid verifier should execute mgcvExtractGPUGCV")

cat("PASS precision backend resolver\n")
