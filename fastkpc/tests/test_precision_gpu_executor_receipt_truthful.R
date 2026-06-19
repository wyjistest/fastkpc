source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

old_pair <- fastkpc_mgcv_extract_gpu_gcv_for_pair
on.exit(assign("fastkpc_mgcv_extract_gpu_gcv_for_pair", old_pair,
               envir = .GlobalEnv), add = TRUE)

call_env <- new.env(parent = emptyenv())
call_env$count <- 0L
assign("fastkpc_mgcv_extract_gpu_gcv_for_pair", function(data, x, y, S,
                                                         sp_grid = NULL) {
  call_env$count <- call_env$count + 1L
  Sys.sleep(0.01)
  residuals <- cbind(
    as.numeric(data[, x]) - mean(data[, x]),
    as.numeric(data[, y]) - mean(data[, y])
  )
  colnames(residuals) <- c("x", "y")
  list(
    residuals = residuals,
    fitted = cbind(rep(mean(data[, x]), nrow(data)),
                   rep(mean(data[, y]), nrow(data))),
    setup_fingerprint = "shared-setup",
    setup_fingerprint_x = "shared-setup",
    setup_fingerprint_y = "shared-setup",
    shared_setup_fingerprint = "shared-setup",
    sp = c(x = 1.5, y = 2.5),
    score = c(x = 2, y = 3),
    edf = c(x = 4, y = 5),
    selected_grid_index = c(x = 2L, y = 3L),
    gcv_grid_points = c(x = 2L, y = 2L),
    grid = list(x = data.frame(sp = c(0.1, 1), gcv = c(2, 1)),
                y = data.frame(sp = c(0.1, 1), gcv = c(2, 1))),
    fit = list(
      used_device = "cuda",
      native_gpu_solve_used = TRUE,
      used_device_x = "cuda",
      used_device_y = "cuda",
      native_gpu_solve_used_x = TRUE,
      native_gpu_solve_used_y = TRUE,
      setup_fingerprint_x = "shared-setup",
      setup_fingerprint_y = "shared-setup",
      shared_setup_fingerprint = "shared-setup",
      sp_selection_backend_executed_x = "r-cpu-spectral",
      sp_selection_backend_executed_y = "r-cpu-spectral",
      gcv_score_backend_executed_x = "r-cpu-spectral",
      gcv_score_backend_executed_y = "r-cpu-spectral",
      selected_solve_backend_executed_x = "cuda",
      selected_solve_backend_executed_y = "cuda",
      same_setup_pair_batch_used = TRUE
    ),
    timings = list(
      residualization_total_ms = 22,
      mgcv_setup_cpu_ms = 2,
      spectral_prepare_ms = 3,
      gcv_score_ms = 4,
      linear_solve_ms = 5,
      host_to_device_ms = 6,
      device_to_host_ms = 7,
      residual_materialize_ms = 8
    )
  )
}, envir = .GlobalEnv)

set.seed(1501)
data <- matrix(rnorm(40 * 3), 40, 3)
route <- list(
  primary_backend = "mgcvExtractGPUGCV",
  setup_fingerprint = "S:3"
)

receipt <- fastkpc_execute_ci_mgcv_extract_gpu(
  data = data,
  x = 1L,
  y = 2L,
  S = 3L,
  ci_method = "dcc.gamma",
  index = 1,
  legacy_index = TRUE,
  hsic_params = list(),
  permutation_params = list(),
  route = route,
  role = "primary"
)

assert_true(call_env$count == 1L,
            "GPU executor should call pair wrapper once")
assert_true(receipt$used_device == "cuda",
            "used_device should be derived from target fit receipts")
assert_true(isTRUE(receipt$native_gpu_solve_used_x) &&
              isTRUE(receipt$native_gpu_solve_used_y),
            "receipt should expose native GPU solve flags for x and y")
assert_true(receipt$setup_fingerprint_x == "shared-setup" &&
              receipt$setup_fingerprint_y == "shared-setup" &&
              receipt$shared_setup_fingerprint == "shared-setup",
            "receipt should expose x/y/shared setup fingerprints")
assert_true(receipt$sp_selection_backend_executed_x == "r-cpu-spectral" &&
              receipt$sp_selection_backend_executed_y == "r-cpu-spectral",
            "receipt should expose sp selection backend per target")
assert_true(isTRUE(receipt$same_setup_pair_batch_used),
            "receipt should expose same-setup pair batch execution")
assert_true(is.finite(receipt$timings$total_ms) &&
              receipt$timings$total_ms >= receipt$timings$residualization_total_ms,
            "total_ms should include residualization time")
assert_true(identical(receipt$timings$residualization_total_ms, 22),
            "residualization_total_ms should come from pair wrapper")
assert_true(is.finite(receipt$timings$ci_test_ms) &&
              receipt$timings$ci_test_ms < receipt$timings$total_ms,
            "ci_test_ms should not include residualization")

assign("fastkpc_mgcv_extract_gpu_gcv_for_pair", function(data, x, y, S,
                                                         sp_grid = NULL) {
  residuals <- cbind(
    as.numeric(data[, x]) - mean(data[, x]),
    as.numeric(data[, y]) - mean(data[, y])
  )
  colnames(residuals) <- c("x", "y")
  list(
    residuals = residuals,
    fitted = residuals * 0,
    setup_fingerprint = "setup-1",
    setup_fingerprint_x = "setup-1",
    setup_fingerprint_y = "setup-2",
    shared_setup_fingerprint = "",
    sp = c(x = 1.5, y = 2.5),
    score = c(x = 2, y = 3),
    edf = c(x = 4, y = 5),
    selected_grid_index = c(x = 2L, y = 3L),
    gcv_grid_points = c(x = 2L, y = 2L),
    grid = list(x = data.frame(), y = data.frame()),
    fit = list(
      used_device = "cuda",
      native_gpu_solve_used = TRUE,
      used_device_x = "cuda",
      used_device_y = "cuda",
      native_gpu_solve_used_x = TRUE,
      native_gpu_solve_used_y = TRUE,
      setup_fingerprint_x = "setup-1",
      setup_fingerprint_y = "setup-2",
      selected_solve_backend_executed_x = "cuda",
      selected_solve_backend_executed_y = "cuda"
    ),
    timings = list(residualization_total_ms = 1)
  )
}, envir = .GlobalEnv)

mismatch <- tryCatch(
  fastkpc_execute_ci_mgcv_extract_gpu(
    data = data,
    x = 1L,
    y = 2L,
    S = 3L,
    ci_method = "dcc.gamma",
    index = 1,
    legacy_index = TRUE,
    hsic_params = list(),
    permutation_params = list(),
    route = route,
    role = "primary"
  ),
  error = function(e) e
)
assert_true(inherits(mismatch, "error") &&
              grepl("setup fingerprint", conditionMessage(mismatch),
                    fixed = TRUE),
            "GPU executor should reject mismatched x/y setup fingerprints")

cat("PASS precision GPU executor receipt truthful\n")
