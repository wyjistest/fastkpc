source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

make_spy <- function(p_empty, p_one, p_two, backend_name, calls_env) {
  force(p_empty)
  force(p_one)
  force(p_two)
  force(backend_name)
  force(calls_env)
  function(data, x, y, S, ci_method, index, legacy_index, hsic_params,
           permutation_params, route, role) {
    calls_env$count <- calls_env$count + 1L
    calls_env$rows[[length(calls_env$rows) + 1L]] <- list(
      x = x, y = y, S = S, role = role, backend = backend_name
    )
    p_value <- switch(as.character(length(S)),
                      `0` = p_empty,
                      `1` = p_one,
                      `2` = p_two,
                      p_two)
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

caps_cpu_only <- list(
  R_version = "4.5.0",
  mgcv_version = "1.9-4",
  cuda_available = FALSE,
  mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
  spectral_gcv_version = "single-penalty-spectral-gcv-v1",
  setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
)

set.seed(1102)
data <- matrix(rnorm(60 * 4), 60, 4)
compatible_calls <- new.env(parent = emptyenv())
compatible_calls$count <- 0L
compatible_calls$rows <- list()

compatible <- fast_kpc(
  data,
  alpha = 0.05,
  max_conditioning_size = 2,
  engine = "cpu",
  precision = "compatible",
  graph_stage = "skeleton",
  precision_executors = list(
    `direct-ci` = make_spy(0.001, 0.001, 0.001, "direct-ci-spy",
                           new.env(parent = emptyenv())),
    mgcvExtractCPUGCVBridge = make_spy(0.001, 0.001, 0.051,
                                       "mgcvExtractCPU-spy",
                                       compatible_calls),
    `legacy-mgcv` = make_spy(0.9, 0.9, 0.9, "legacy-mgcv-spy",
                             new.env(parent = emptyenv()))
  ),
  runtime_capabilities = caps_cpu_only
)

assert_true(any(vapply(compatible_calls$rows, function(row) length(row$S) == 2L,
                       logical(1))),
            "compatible |S|=2 should call CPU mgcvExtract executor")
assert_true(any(compatible$diagnostics$precision_trace$S_key == "3|4" &
                  compatible$diagnostics$precision_trace$backend_planned ==
                    "mgcvExtractCPUGCVBridge"),
            "trace should record planned CPU mgcvExtract backend for |S|=2")

primary_calls <- new.env(parent = emptyenv())
primary_calls$count <- 0L
primary_calls$rows <- list()
verifier_calls <- new.env(parent = emptyenv())
verifier_calls$count <- 0L
verifier_calls$rows <- list()

hybrid <- fast_kpc(
  data,
  alpha = 0.05,
  max_conditioning_size = 2,
  engine = "cpu",
  precision = "hybrid",
  tau = log(2),
  graph_stage = "skeleton",
  precision_executors = list(
    `direct-ci` = make_spy(0.001, 0.001, 0.001, "direct-ci-spy",
                           new.env(parent = emptyenv())),
    fastSplineCPU = make_spy(0.001, 0.001, 0.049,
                             "fastSplineCPU-spy", primary_calls),
    mgcvExtractCPUGCVBridge = make_spy(0.001, 0.001, 0.051,
                                       "mgcvExtractCPU-spy",
                                       verifier_calls),
    `legacy-mgcv` = make_spy(0.9, 0.9, 0.9, "legacy-mgcv-spy",
                             new.env(parent = emptyenv()))
  ),
  runtime_capabilities = caps_cpu_only
)
verified_two <- hybrid$diagnostics$precision_trace[
  hybrid$diagnostics$precision_trace$near_alpha_triggered &
    grepl("\\|", hybrid$diagnostics$precision_trace$S_key),
  , drop = FALSE
]
assert_true(nrow(verified_two) > 0L,
            "hybrid |S|=2 should trigger verifier under near-alpha primary")
assert_true(any(verified_two$verifier_executed == "mgcvExtractCPU-spy" &
                  verified_two$p_used == 0.051),
            "hybrid |S|=2 should use CPU mgcvExtract verifier p-value")

cat("PASS precision CPU |S|=2 data plane\n")
