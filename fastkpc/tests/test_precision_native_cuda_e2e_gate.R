source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP precision native CUDA E2E gate: FASTKPC_RUN_CUDA_TESTS != 1\n")
  quit(save = "no", status = 0)
}

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP precision native CUDA E2E gate: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

build_fastkpc_cuda_native(rebuild = FALSE)
if (!fastkpc_cuda_available()) {
  cat("SKIP precision native CUDA E2E gate: CUDA unavailable\n")
  quit(save = "no", status = 0)
}

set.seed(2102)
n <- 72
z <- stats::runif(n, -2, 2)
x <- sin(z) + stats::rnorm(n, sd = 0.08)
y <- cos(z) + stats::rnorm(n, sd = 0.08)
w <- z + stats::rnorm(n, sd = 0.08)
data <- cbind(x, y, w)

assert_gpu_trace <- function(result, verifier = FALSE) {
  trace <- result$diagnostics$precision_trace
  nonempty <- trace[nzchar(trace$S_key), , drop = FALSE]
  assert_true(nrow(nonempty) > 0L,
              "native CUDA gate requires at least one non-empty S test")
  assert_true(all(grepl("mgcvExtractGPU", nonempty$attempt_backend_sequence,
                        fixed = TRUE)),
              "non-empty S attempts should include mgcvExtractGPU execution")
  assert_true(!any(grepl("mgcvExtractCPUGCVBridge", nonempty$attempt_backend_sequence,
                         fixed = TRUE)),
              "successful native CUDA attempts should not call CPU bridge")
  assert_true(!any(grepl("legacy-mgcv", nonempty$attempt_backend_sequence,
                         fixed = TRUE)),
              "successful native CUDA attempts should not call legacy mgcv")

  if (isTRUE(verifier)) {
    verified <- nonempty[nonempty$near_alpha_triggered, , drop = FALSE]
    assert_true(nrow(verified) > 0L,
                "hybrid tau=Inf should trigger native CUDA verifier")
    assert_true(all(verified$verifier_planned == "mgcvExtractGPUGCV"),
                "hybrid verifier should plan mgcvExtractGPUGCV")
    assert_true(all(verified$verifier_executed == "mgcvExtractGPU"),
                "hybrid verifier should execute mgcvExtractGPU")
  } else {
    receipt <- result$skeleton$precision_receipt
    assert_true(identical(receipt$used_device_x, "cuda") &&
                  identical(receipt$used_device_y, "cuda"),
                "receipt should report CUDA target devices")
    assert_true(isTRUE(receipt$native_gpu_solve_used_x) &&
                  isTRUE(receipt$native_gpu_solve_used_y),
                "receipt should report native GPU solves for x/y")
    assert_true(identical(receipt$selected_solve_backend_executed_x, "cuda") &&
                  identical(receipt$selected_solve_backend_executed_y, "cuda"),
                "receipt should report CUDA selected solves")
    assert_true(nzchar(receipt$shared_setup_fingerprint) &&
                  identical(receipt$setup_fingerprint_x,
                            receipt$shared_setup_fingerprint) &&
                  identical(receipt$setup_fingerprint_y,
                            receipt$shared_setup_fingerprint),
                "receipt should expose one shared setup fingerprint")
    assert_true(is.finite(receipt$timings$total_ms) &&
                  is.finite(receipt$timings$residualization_total_ms) &&
                  is.finite(receipt$timings$ci_test_ms) &&
                  receipt$timings$total_ms >= receipt$timings$ci_test_ms,
                "receipt should expose total/residualization/CI timings")
    assert_true(all(nonempty$backend_planned == "mgcvExtractGPUGCV"),
                "compatible non-empty S rows should plan mgcvExtractGPUGCV")
    assert_true(all(nonempty$backend_executed == "mgcvExtractGPU"),
                "compatible non-empty S rows should execute mgcvExtractGPU")
  }
}

compatible <- fast_kpc(
  data,
  alpha = 0.05,
  max_conditioning_size = 1,
  engine = "cuda",
  precision = "compatible",
  graph_stage = "skeleton",
  allow_canary_mgcv_extract = TRUE
)
assert_true(compatible$config$precision_execution_status == "data-plane-executed",
            "compatible CUDA should execute precision data plane")
assert_gpu_trace(compatible, verifier = FALSE)

hybrid <- fast_kpc(
  data,
  alpha = 0.05,
  max_conditioning_size = 1,
  engine = "cuda",
  precision = "hybrid",
  tau = Inf,
  graph_stage = "skeleton",
  precision_trace_level = "full",
  allow_canary_mgcv_extract = TRUE
)
assert_true(hybrid$config$precision_execution_status == "batched-primary-data-plane",
            "hybrid CUDA should execute batched primary precision data plane")
assert_gpu_trace(hybrid, verifier = TRUE)

cat("PASS precision native CUDA E2E gate\n")
