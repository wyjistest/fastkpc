source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

make_spy <- function(p_value, backend_name, calls_env, p_value_nonempty = p_value) {
  force(p_value)
  force(p_value_nonempty)
  force(backend_name)
  force(calls_env)
  function(data, x, y, S, ci_method, index, legacy_index, hsic_params,
           permutation_params, route, role) {
    calls_env$count <- calls_env$count + 1L
    calls_env$rows[[length(calls_env$rows) + 1L]] <- list(
      x = x, y = y, S = S, role = role, backend = backend_name
    )
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

set.seed(993)
data <- matrix(rnorm(50 * 3), 50, 3)
direct_calls <- new.env(parent = emptyenv())
direct_calls$count <- 0L
direct_calls$rows <- list()
primary_calls <- new.env(parent = emptyenv())
primary_calls$count <- 0L
primary_calls$rows <- list()
compatible_calls <- new.env(parent = emptyenv())
compatible_calls$count <- 0L
compatible_calls$rows <- list()

executors <- list(
  `direct-ci` = make_spy(0.049, "direct-ci-spy", direct_calls),
  fastSplineCPU = make_spy(0.049, "fastSplineCPU-spy", primary_calls),
  mgcvExtractGPUGCV = make_spy(0.049, "mgcvExtractCPU-spy",
                               compatible_calls, p_value_nonempty = 0.051)
)

caps <- list(
  R_version = "4.5.0",
  mgcv_version = "1.9-4",
  cuda_available = TRUE,
  mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
  spectral_gcv_version = "single-penalty-spectral-gcv-v1",
  setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
)

fast <- fast_kpc(
  data,
  alpha = 0.05,
  max_conditioning_size = 1,
  engine = "cpu",
  precision = "fast",
  graph_stage = "skeleton",
  precision_executors = executors,
  runtime_capabilities = caps
)

assert_true(primary_calls$count > 0L,
            "fast precision should call primary fastSpline spy")
assert_true(compatible_calls$count == 0L,
            "fast precision must not call compatible spy")
assert_true(fast$config$precision_execution_status == "data-plane-executed",
            "fast spy data plane should be marked executed")

primary_calls$count <- 0L
primary_calls$rows <- list()
compatible_calls$count <- 0L
compatible_calls$rows <- list()
direct_calls$count <- 0L
direct_calls$rows <- list()

compatible <- fast_kpc(
  data,
  alpha = 0.05,
  max_conditioning_size = 1,
  engine = "cpu",
  precision = "compatible",
  graph_stage = "skeleton",
  precision_executors = executors,
  runtime_capabilities = caps
)

assert_true(primary_calls$count == 0L,
            "compatible precision must not execute fastSpline primary spy")
assert_true(direct_calls$count > 0L,
            "compatible precision should execute direct CI for empty S")
assert_true(compatible_calls$count > 0L,
            "compatible precision should execute mgcv-compatible spy")
assert_true(compatible$config$backend_planned == "mgcvExtractGPUGCV",
            "compatible supported route should plan mgcvExtractGPUGCV")
assert_true(compatible$config$backend_executed == "mgcvExtractCPU-spy",
            "backend_executed must come from compatible executor receipt")
assert_true(compatible$config$backend_used == compatible$config$backend_executed,
            "backend_used must match executed compatible backend")
assert_true(compatible$config$precision_execution_status == "data-plane-executed",
            "compatible vertical slice should mark data-plane execution")
assert_true(!isTRUE(compatible$skeleton$adjacency[1, 2]),
            "compatible p=0.051 at alpha=0.05 should delete edge 1-2")
assert_true(identical(as.integer(compatible$skeleton$sepsets[[1]][[2]]), 3L),
            "compatible decision should record canonical |S|=1 sepset")

trace <- compatible$diagnostics$precision_trace
assert_true(any(trace$p_used == 0.051),
            "real trace should contain compatible p_used")
assert_true(any(trace$p_source_used ==
                  "primary:mgcvExtractCPU-spy+spy-ci"),
            "trace should record composite compatible p source")
assert_true(any(trace$backend_executed == "mgcvExtractCPU-spy"),
            "trace backend_executed should include compatible executor receipts")
assert_true(any(trace$edge_deleted & trace$sepset_recorded == "3"),
            "trace should record edge deletion and sepset from compatible p")

cat("PASS precision compatible data plane spy\n")
