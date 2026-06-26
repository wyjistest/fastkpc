source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

make_spy <- function(p_empty, p_nonempty, backend_name, calls_env) {
  force(p_empty)
  force(p_nonempty)
  force(backend_name)
  force(calls_env)
  function(data, x, y, S, ci_method, index, legacy_index, hsic_params,
           permutation_params, route, role) {
    calls_env$rows[[length(calls_env$rows) + 1L]] <- list(
      x = x, y = y, S = S, role = role,
      seed = permutation_params$seed %||% NA_integer_,
      replicates = permutation_params$replicates %||% NA_integer_
    )
    list(
      p.value = if (length(S) == 0L) p_empty else p_nonempty,
      residual_backend_executed = backend_name,
      ci_backend_executed = "spy-ci",
      setup_fingerprint = paste0("spy:", paste(S, collapse = "|")),
      p_source_used = paste0(role, ":", backend_name, "+spy-ci"),
      timings = list(ci_test_ms = 0)
    )
  }
}

set.seed(1103)
data <- matrix(rnorm(80 * 4), 80, 4)
caps <- list(
  R_version = "4.5.0",
  mgcv_version = "1.9-4",
  cuda_available = FALSE,
  mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
  spectral_gcv_version = "single-penalty-spectral-gcv-v1",
  setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
)

primary_calls_a <- new.env(parent = emptyenv())
primary_calls_a$rows <- list()
verifier_calls_a <- new.env(parent = emptyenv())
verifier_calls_a$rows <- list()
executors_a <- list(
  `direct-ci` = make_spy(0.001, 0.001, "direct-ci-spy",
                         new.env(parent = emptyenv())),
  fastSplineCPU = make_spy(0.001, 0.049, "fastSplineCPU-spy",
                           primary_calls_a),
  mgcvExtractCPUGCVBridge = make_spy(0.001, 0.051, "mgcvExtractCPU-spy",
                                     verifier_calls_a),
  `legacy-mgcv` = make_spy(0.9, 0.9, "legacy-mgcv-spy",
                           new.env(parent = emptyenv()))
)

a <- fast_kpc(
  data,
  alpha = 0.05,
  max_conditioning_size = 1,
  engine = "cpu",
  precision = "hybrid",
  tau = Inf,
  graph_stage = "skeleton",
  ci_method = "hsic.perm",
  permutation_params = list(replicates = 12L, seed = 404L,
                            include_observed = TRUE),
  precision_executors = executors_a,
  runtime_capabilities = caps,
  precision_trace_level = "full"
)

primary_calls_b <- new.env(parent = emptyenv())
primary_calls_b$rows <- list()
verifier_calls_b <- new.env(parent = emptyenv())
verifier_calls_b$rows <- list()
executors_b <- list(
  `direct-ci` = make_spy(0.001, 0.001, "direct-ci-spy",
                         new.env(parent = emptyenv())),
  fastSplineCPU = make_spy(0.001, 0.049, "fastSplineCPU-spy",
                           primary_calls_b),
  mgcvExtractCPUGCVBridge = make_spy(0.001, 0.051, "mgcvExtractCPU-spy",
                                     verifier_calls_b),
  `legacy-mgcv` = make_spy(0.9, 0.9, "legacy-mgcv-spy",
                           new.env(parent = emptyenv()))
)
b <- fast_kpc(
  data,
  alpha = 0.05,
  max_conditioning_size = 1,
  engine = "cpu",
  precision = "hybrid",
  tau = Inf,
  graph_stage = "skeleton",
  ci_method = "hsic.perm",
  permutation_params = list(replicates = 12L, seed = 404L,
                            include_observed = TRUE),
  precision_executors = executors_b,
  runtime_capabilities = caps,
  precision_trace_level = "full"
)

trace_a <- a$diagnostics$precision_trace
trace_b <- b$diagnostics$precision_trace
required <- c("ci_randomness_id", "permutation_seed_effective",
              "permutation_plan_spec_hash", "permutation_plan_hash",
              "permutation_replicates")
assert_true(all(required %in% names(trace_a)),
            "precision trace should expose CI randomness plan fields")
assert_true(identical(trace_a$ci_randomness_id, trace_b$ci_randomness_id),
            "fixed seed should reproduce CI randomness ids")
assert_true(identical(trace_a$permutation_plan_spec_hash,
                      trace_b$permutation_plan_spec_hash),
            "fixed seed should reproduce permutation plan spec hash")
assert_true(identical(trace_a$permutation_plan_spec_hash,
                      trace_a$permutation_plan_hash),
            "deprecated permutation plan hash alias should match spec hash")
assert_true(all(trace_a$permutation_replicates == 12L),
            "trace should record permutation replicate count")

verified <- trace_a[trace_a$near_alpha_triggered, , drop = FALSE]
assert_true(nrow(verified) > 0L,
            "tau=Inf should force verifier rows for non-empty S")
assert_true(all(nzchar(verified$permutation_plan_spec_hash)),
            "verified rows should retain explicit permutation plan spec hash")

primary_nonempty <- Filter(function(row) length(row$S) > 0L,
                           primary_calls_a$rows)
verifier_nonempty <- Filter(function(row) length(row$S) > 0L,
                            verifier_calls_a$rows)
assert_true(length(primary_nonempty) > 0L && length(verifier_nonempty) > 0L,
            "spy executors should observe non-empty primary and verifier calls")
assert_true(identical(primary_nonempty[[1]]$seed, verifier_nonempty[[1]]$seed),
            "primary and verifier should reuse the same effective seed")
assert_true(identical(primary_nonempty[[1]]$replicates,
                      verifier_nonempty[[1]]$replicates),
            "primary and verifier should reuse the same replicate count")

cat("PASS precision HSIC permutation trace plan\n")
