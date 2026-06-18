fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

source("fastkpc/R/backend_routing_policy.R")

fast_route <- fastkpc_select_backend_route(
  precision = "fast",
  S_size = 2L,
  single_penalty = TRUE,
  mgcv_extract_gpu_supported = TRUE
)
assert_true(fast_route$primary_backend == "fastSplineCUDA",
            "fast mode should use fastSplineCUDA")
assert_true(is.na(fast_route$verifier_backend),
            "fast mode should not select verifier")
assert_true(fast_route$compatibility_claim == "approximate",
            "fast mode should be approximate only")

compatible_route <- fastkpc_select_backend_route(
  precision = "compatible",
  S_size = 2L,
  single_penalty = TRUE,
  mgcv_extract_gpu_supported = TRUE
)
assert_true(compatible_route$primary_backend == "mgcvExtractGPUGCV",
            "compatible supported route should use mgcvExtractGPUGCV")
assert_true(compatible_route$compatibility_claim == "mgcv-setup-anchored",
            "compatible claim should be mgcv setup anchored")
assert_true(compatible_route$canonical_replay_required == TRUE,
            "compatible route should preserve canonical replay")

fallback_route <- fastkpc_select_backend_route(
  precision = "compatible",
  S_size = 4L,
  single_penalty = FALSE,
  mgcv_extract_gpu_supported = FALSE
)
assert_true(fallback_route$primary_backend %in% c("mgcvExtractCPU", "legacy-mgcv"),
            "unsupported compatible route should fall back to mgcvExtractCPU/legacy")
assert_true(nzchar(fallback_route$fallback_reason),
            "fallback route should explain reason")

hybrid_route <- fastkpc_select_backend_route(
  precision = "hybrid",
  S_size = 1L,
  single_penalty = TRUE,
  mgcv_extract_gpu_supported = TRUE,
  tau = log(2)
)
assert_true(hybrid_route$primary_backend == "fastSplineCUDA",
            "hybrid primary should be fastSplineCUDA")
assert_true(hybrid_route$verifier_backend == "mgcvExtractGPUGCV",
            "hybrid verifier should be mgcvExtractGPUGCV when supported")
assert_true(hybrid_route$canonical_replay_required == TRUE,
            "hybrid must require canonical replay")
assert_true(all(fastkpc_near_alpha_trigger(c(0.04, 0.20), 0.05, log(2)) ==
                  c(TRUE, FALSE)),
            "near-alpha trigger should use log band around alpha")

cat("PASS backend routing policy\n")
