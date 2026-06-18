source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

make_spy <- function(p_value, backend_name) {
  force(p_value)
  force(backend_name)
  function(data, x, y, S, ci_method, index, legacy_index, hsic_params,
           permutation_params, route, role) {
    list(
      p.value = p_value,
      residual_backend_executed = backend_name,
      ci_backend_executed = "spy-ci",
      setup_fingerprint = paste0("spy:", paste(S, collapse = "|")),
      p_source_used = paste0(role, ":", backend_name, "+spy-ci"),
      timings = list(ci_test_ms = 0)
    )
  }
}

set.seed(103)
data <- matrix(rnorm(90 * 4), 90, 4)
caps <- list(
  R_version = "4.5.0",
  mgcv_version = "1.9-4",
  cuda_available = TRUE,
  mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
  spectral_gcv_version = "single-penalty-spectral-gcv-v1",
  setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
)
executors <- list(
  `direct-ci` = make_spy(0.001, "direct-ci-spy"),
  fastSplineCPU = make_spy(0.049, "fastSplineCPU-spy"),
  mgcvExtractGPUGCV = make_spy(0.049, "mgcvExtractCPU-spy")
)

primary <- fast_kpc(
  data, alpha = 0.05, max_conditioning_size = 1,
  engine = "cpu", precision = "fast",
  graph_stage = "skeleton", seed = 103,
  precision_executors = executors,
  runtime_capabilities = caps
)
hybrid <- fast_kpc(
  data, alpha = 0.05, max_conditioning_size = 1,
  engine = "cpu", precision = "hybrid", tau = log(2),
  graph_stage = "skeleton", seed = 103,
  precision_executors = executors,
  runtime_capabilities = caps
)

assert_true(hybrid$config$precision == "hybrid",
            "config should record hybrid precision")
assert_true(isTRUE(hybrid$config$canonical_replay_required),
            "hybrid must require canonical replay")
assert_true(hybrid$config$precision_execution_status == "data-plane-executed",
            "hybrid replay should execute CPU data plane")
assert_true(is.data.frame(hybrid$diagnostics$precision_trace),
            "hybrid result should include precision trace")

trace <- hybrid$diagnostics$precision_trace
required <- c("run_id", "canonical_test_order_id", "backend_requested",
              "backend_used", "p_source_used", "fallback_reason",
              "near_alpha_triggered", "verifier_executed",
              "decision_before_verify", "decision_after_verify")
assert_true(all(required %in% names(trace)),
            "precision trace should expose p-value source, verifier, and fallback")
assert_true(all(order(trace$canonical_test_order_id) ==
                  seq_along(trace$canonical_test_order_id)),
            "hybrid trace must preserve canonical ordering")
assert_true(any(trace$near_alpha_triggered),
            "spy primary p should force at least one verifier replay")
assert_true(any(trace$verifier_executed == "mgcvExtractCPU-spy"),
            "hybrid replay should execute verifier")
assert_true(identical(hybrid$skeleton$adjacency, primary$skeleton$adjacency),
            "equal verifier p should preserve primary adjacency")

cat("PASS precision hybrid e2e replay\n")
