source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

make_spy <- function(p_value, backend_name, calls_env,
                     p_value_nonempty = p_value) {
  force(p_value)
  force(p_value_nonempty)
  force(backend_name)
  force(calls_env)
  function(data, x, y, S, ci_method, index, legacy_index, hsic_params,
           permutation_params, route, role) {
    calls_env$count <- calls_env$count + 1L
    list(
      p.value = if (length(S) == 0L) p_value else p_value_nonempty,
      residual_backend_executed = backend_name,
      ci_backend_executed = "spy-ci",
      setup_fingerprint = paste0("spy:", paste(S, collapse = "|")),
      p_source_used = paste0(role, ":", backend_name, "+spy-ci"),
      timings = list(ci_test_ms = 0)
    )
  }
}

caps <- list(
  R_version = "4.5.0",
  mgcv_version = "1.9-4",
  cuda_available = TRUE,
  mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
  spectral_gcv_version = "single-penalty-spectral-gcv-v1",
  setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
)

set.seed(1002)
data <- matrix(rnorm(48 * 3), 48, 3)

direct_calls <- new.env(parent = emptyenv())
direct_calls$count <- 0L
primary_calls <- new.env(parent = emptyenv())
primary_calls$count <- 0L
verifier_calls <- new.env(parent = emptyenv())
verifier_calls$count <- 0L
legacy_calls <- new.env(parent = emptyenv())
legacy_calls$count <- 0L

primary_na <- fast_kpc(
  data,
  alpha = 0.05,
  max_conditioning_size = 1,
  engine = "cpu",
  precision = "hybrid",
  tau = log(2),
  graph_stage = "skeleton",
  precision_executors = list(
    `direct-ci` = make_spy(0.001, "direct-ci-spy", direct_calls),
    fastSplineCPU = make_spy(0.001, "fastSplineCPU-spy", primary_calls,
                             p_value_nonempty = NA_real_),
    mgcvExtractGPUGCV = make_spy(0.051, "mgcvExtractCPU-spy",
                                 verifier_calls),
    `legacy-mgcv` = make_spy(0.9, "legacy-mgcv-spy", legacy_calls)
  ),
  runtime_capabilities = caps
)

trace <- primary_na$diagnostics$precision_trace
triggered <- trace[trace$near_alpha_triggered, , drop = FALSE]
assert_true(primary_calls$count > 0L,
            "primary should execute non-empty residual tests")
assert_true(verifier_calls$count > 0L,
            "non-finite primary p should trigger verifier")
assert_true(legacy_calls$count == 0L,
            "finite mgcvExtract verifier should not call legacy fallback")
assert_true(any(is.na(triggered$primary_p_raw) &
                  triggered$verifier_p_raw == 0.051 &
                  triggered$p_used == 0.051),
            "trace should use finite verifier p after primary NA")

direct_calls$count <- 0L
primary_calls$count <- 0L
verifier_calls$count <- 0L
legacy_calls$count <- 0L

verifier_na <- fast_kpc(
  data,
  alpha = 0.05,
  max_conditioning_size = 1,
  engine = "cpu",
  precision = "hybrid",
  tau = log(2),
  graph_stage = "skeleton",
  precision_executors = list(
    `direct-ci` = make_spy(0.001, "direct-ci-spy", direct_calls),
    fastSplineCPU = make_spy(0.049, "fastSplineCPU-spy", primary_calls),
    mgcvExtractGPUGCV = make_spy(NA_real_, "mgcvExtractCPU-spy",
                                 verifier_calls),
    `legacy-mgcv` = make_spy(0.051, "legacy-mgcv-spy", legacy_calls)
  ),
  runtime_capabilities = caps
)

trace <- verifier_na$diagnostics$precision_trace
fallback_rows <- trace[trace$fallback_triggered, , drop = FALSE]
assert_true(verifier_calls$count > 0L,
            "near-alpha test should attempt mgcvExtract verifier")
assert_true(legacy_calls$count > 0L,
            "non-finite mgcvExtract verifier should call legacy fallback")
assert_true(any(fallback_rows$verifier_executed == "legacy-mgcv-spy" &
                  fallback_rows$p_used == 0.051),
            "trace should use finite legacy verifier p after verifier NA")
assert_true(any(fallback_rows$fallback_reason ==
                  "verifier returned non-finite p-value"),
            "trace should record non-finite verifier fallback reason")

direct_calls$count <- 0L
primary_calls$count <- 0L
verifier_calls$count <- 0L
legacy_calls$count <- 0L

legacy_na <- fast_kpc(
  data,
  alpha = 0.05,
  max_conditioning_size = 1,
  engine = "cpu",
  precision = "hybrid",
  tau = log(2),
  graph_stage = "skeleton",
  precision_executors = list(
    `direct-ci` = make_spy(0.001, "direct-ci-spy", direct_calls),
    fastSplineCPU = make_spy(0.049, "fastSplineCPU-spy", primary_calls),
    mgcvExtractGPUGCV = make_spy(NA_real_, "mgcvExtractCPU-spy",
                                 verifier_calls),
    `legacy-mgcv` = make_spy(NA_real_, "legacy-mgcv-spy", legacy_calls)
  ),
  runtime_capabilities = caps
)

trace <- legacy_na$diagnostics$precision_trace
fallback_rows <- trace[trace$fallback_triggered, , drop = FALSE]
assert_true(verifier_calls$count > 0L && legacy_calls$count > 0L,
            "legacy fallback should run when mgcvExtract verifier is NA")
assert_true(any(fallback_rows$verifier_executed == "legacy-mgcv-spy" &
                  is.na(fallback_rows$verifier_p_raw) &
                  fallback_rows$p_used == 1 &
                  fallback_rows$nonfinite_action == "na-delete-use-1"),
            "legacy NA should apply NAdelete only after fallback attempt")

cat("PASS precision hybrid nonfinite verifier policy\n")
