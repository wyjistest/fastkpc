source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

old_pair <- if (exists("fastkpc_mgcv_extract_gpu_gcv_for_pair", mode = "function")) {
  fastkpc_mgcv_extract_gpu_gcv_for_pair
} else {
  NULL
}
on.exit({
  if (is.null(old_pair)) {
    rm("fastkpc_mgcv_extract_gpu_gcv_for_pair", envir = .GlobalEnv)
  } else {
    assign("fastkpc_mgcv_extract_gpu_gcv_for_pair", old_pair,
           envir = .GlobalEnv)
  }
}, add = TRUE)

pair_env <- new.env(parent = emptyenv())
pair_env$count <- 0L
pair_env$calls <- list()
assign("fastkpc_mgcv_extract_gpu_gcv_for_pair", function(data, x, y, S,
                                                         sp_grid = NULL) {
  pair_env$count <- pair_env$count + 1L
  pair_env$calls[[length(pair_env$calls) + 1L]] <- list(x = x, y = y, S = S)
  residual_for <- function(target) {
    as.numeric(data[, target]) - mean(data[, target])
  }
  fitted_for <- function(target) {
    rep(mean(data[, target]), nrow(data))
  }
  setup <- paste0("cached-spy:S:", paste(S, collapse = "|"))
  list(
    residuals = cbind(x = residual_for(x), y = residual_for(y)),
    fitted = cbind(x = fitted_for(x), y = fitted_for(y)),
    setup_fingerprint = setup,
    setup_fingerprint_x = setup,
    setup_fingerprint_y = setup,
    shared_setup_fingerprint = setup,
    sp = c(x = 0.4 + x / 100, y = 0.4 + y / 100),
    score = c(x = 1 + x / 10, y = 1 + y / 10),
    edf = c(x = 3 + x / 10, y = 3 + y / 10),
    selected_grid_index = c(x = x, y = y),
    gcv_grid_points = c(x = 9L, y = 9L),
    grid = list(x = data.frame(sp = 0.4, gcv = 1),
                y = data.frame(sp = 0.5, gcv = 1)),
    fit = list(
      used_device = "cuda",
      native_gpu_solve_used = TRUE,
      used_device_x = "cuda",
      used_device_y = "cuda",
      native_gpu_solve_used_x = TRUE,
      native_gpu_solve_used_y = TRUE,
      setup_fingerprint_x = setup,
      setup_fingerprint_y = setup,
      shared_setup_fingerprint = setup,
      sp_selection_backend_executed_x = "r-cpu-spectral",
      sp_selection_backend_executed_y = "r-cpu-spectral",
      gcv_score_backend_executed_x = "r-cpu-spectral",
      gcv_score_backend_executed_y = "r-cpu-spectral",
      selected_solve_backend_executed_x = "cuda",
      selected_solve_backend_executed_y = "cuda",
      same_setup_pair_batch_used = TRUE
    ),
    timings = list(
      residualization_total_ms = 20,
      mgcv_setup_cpu_ms = 4,
      spectral_prepare_ms = 5,
      gcv_score_ms = 6,
      linear_solve_ms = 7,
      host_to_device_ms = 8,
      device_to_host_ms = 9,
      residual_materialize_ms = 10
    )
  )
}, envir = .GlobalEnv)

caps <- list(
  R_version = "4.5.0",
  mgcv_version = "1.9-4",
  cuda_available = TRUE,
  cuda_device_capability = "8.9",
  mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
  spectral_gcv_version = "single-penalty-spectral-gcv-v1",
  setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
)

set.seed(2201)
data <- matrix(rnorm(72 * 5), 72, 5)

uncached <- fast_kpc(
  data,
  alpha = 1.1,
  max_conditioning_size = 1,
  engine = "cuda",
  precision = "compatible",
  graph_stage = "skeleton",
  residual_cache = FALSE,
  runtime_capabilities = caps
)
uncached_calls <- pair_env$count

pair_env$count <- 0L
pair_env$calls <- list()
cached <- fast_kpc(
  data,
  alpha = 1.1,
  max_conditioning_size = 1,
  engine = "cuda",
  precision = "compatible",
  graph_stage = "skeleton",
  residual_cache = TRUE,
  runtime_capabilities = caps
)
cached_calls <- pair_env$count

assert_true(identical(uncached$skeleton$adjacency, cached$skeleton$adjacency),
            "same-S residual cache must not change adjacency")
assert_true(identical(uncached$skeleton$sepsets, cached$skeleton$sepsets),
            "same-S residual cache must not change sepsets")
assert_true(identical(uncached$diagnostics$precision_trace$canonical_test_order_id,
                      cached$diagnostics$precision_trace$canonical_test_order_id),
            "same-S residual cache must preserve canonical test order")
assert_true(cached_calls < uncached_calls,
            "same-S residual cache should reduce GPU pair residualization calls")
assert_true(isTRUE(cached$skeleton$residual_cache$enabled),
            "precision skeleton should report residual cache enabled")
assert_true(cached$skeleton$residual_cache$hits > 0L,
            "precision skeleton should report same-S residual cache hits")
assert_true(cached$skeleton$residual_cache$computations == cached_calls,
            "residual cache computations should match pair wrapper calls")
assert_true("cache_hit" %in% names(cached$diagnostics$precision_trace),
            "precision trace should expose per-test cache hit status")
assert_true(any(cached$diagnostics$precision_trace$cache_hit),
            "precision trace should mark tests served from residual cache")

cat("PASS precision same-S residual cache\n")
