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

call_env <- new.env(parent = emptyenv())
call_env$count <- 0L
assign("fastkpc_mgcv_extract_gpu_gcv_for_pair", function(data, x, y, S,
                                                         sp_grid = NULL) {
  call_env$count <- call_env$count + 1L
  residuals <- cbind(
    as.numeric(data[, x]) - mean(data[, x]),
    as.numeric(data[, y]) - mean(data[, y])
  )
  colnames(residuals) <- c("x", "y")
  list(
    residuals = residuals,
    fitted = cbind(rep(mean(data[, x]), nrow(data)),
                   rep(mean(data[, y]), nrow(data))),
    setup_fingerprint = "pair-shared-setup",
    setup_fingerprint_x = "pair-shared-setup",
    setup_fingerprint_y = "pair-shared-setup",
    shared_setup_fingerprint = "pair-shared-setup",
    sp = c(x = 0.4, y = 1.2),
    score = c(x = 2.1, y = 3.2),
    edf = c(x = 4.1, y = 4.2),
    selected_grid_index = c(x = 3L, y = 5L),
    gcv_grid_points = c(x = 9L, y = 9L),
    grid = list(x = data.frame(sp = 0.4, gcv = 2.1),
                y = data.frame(sp = 1.2, gcv = 3.2)),
    fit = list(
      used_device = "cuda",
      native_gpu_solve_used = TRUE,
      used_device_x = "cuda",
      used_device_y = "cuda",
      native_gpu_solve_used_x = TRUE,
      native_gpu_solve_used_y = TRUE,
      setup_fingerprint_x = "pair-shared-setup",
      setup_fingerprint_y = "pair-shared-setup",
      shared_setup_fingerprint = "pair-shared-setup",
      sp_selection_backend_executed_x = "r-cpu-spectral",
      sp_selection_backend_executed_y = "r-cpu-spectral",
      gcv_score_backend_executed_x = "r-cpu-spectral",
      gcv_score_backend_executed_y = "r-cpu-spectral",
      selected_solve_backend_executed_x = "cuda",
      selected_solve_backend_executed_y = "cuda",
      same_setup_pair_batch_used = TRUE
    ),
    timings = list(
      residualization_total_ms = 23,
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

set.seed(2103)
data <- matrix(stats::rnorm(48 * 3), 48, 3)
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
            "GPU executor should call pair GCV wrapper once")
assert_true(isTRUE(receipt$same_setup_pair_batch_used),
            "receipt should report same-setup pair batch use")
assert_true(receipt$shared_setup_fingerprint == "pair-shared-setup",
            "receipt should carry pair shared setup fingerprint")
assert_true(receipt$selected_solve_backend_executed_x == "cuda" &&
              receipt$selected_solve_backend_executed_y == "cuda",
            "receipt should carry per-target selected solve backend")
assert_true(receipt$timings$residualization_total_ms == 23 &&
              receipt$timings$mgcv_setup_cpu_ms == 4 &&
              receipt$timings$linear_solve_ms == 7,
            "receipt should use pair timing fields without double summing")

cat("PASS precision GPU pair wrapper receipt\n")
