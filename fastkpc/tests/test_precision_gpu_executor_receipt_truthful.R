source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

old_fit <- fastkpc_mgcv_extract_gpu_gcv_for_target
on.exit(assign("fastkpc_mgcv_extract_gpu_gcv_for_target", old_fit,
               envir = .GlobalEnv), add = TRUE)

call_env <- new.env(parent = emptyenv())
call_env$targets <- integer()
assign("fastkpc_mgcv_extract_gpu_gcv_for_target", function(data, target, S,
                                                           sp_grid = NULL) {
  call_env$targets <- c(call_env$targets, as.integer(target))
  Sys.sleep(0.01)
  list(
    residuals = as.numeric(data[, target]) - mean(data[, target]),
    fitted = rep(mean(data[, target]), nrow(data)),
    setup_fingerprint = "shared-setup",
    setup_fingerprint_full = list(fingerprint = "shared-setup"),
    sp = 0.5 + target,
    score = 1 + target,
    edf = 2 + target,
    grid = data.frame(sp = c(0.1, 1), gcv = c(2, 1)),
    fit = list(
      used_device = "cuda",
      native_gpu_solve_used = TRUE,
      setup_fingerprint = list(fingerprint = "shared-setup"),
      sp_selection_backend_executed = "r-cpu-spectral",
      gcv_score_backend_executed = "r-cpu-spectral",
      selected_solve_backend_executed = "cuda"
    ),
    timings = list(
      residualization_total_ms = 11,
      setup_cpu_ms = 2,
      spectral_prepare_ms = 3,
      gcv_score_ms = 4,
      linear_solve_ms = 5,
      host_to_device_ms = 6,
      device_to_host_ms = 7
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

assert_true(identical(call_env$targets, c(1L, 2L)),
            "GPU executor should fit x and y targets")
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
assert_true(is.finite(receipt$timings$total_ms) &&
              receipt$timings$total_ms >= receipt$timings$residualization_total_ms,
            "total_ms should include residualization time")
assert_true(is.finite(receipt$timings$residualization_total_ms) &&
              receipt$timings$residualization_total_ms >= 20,
            "residualization_total_ms should include x/y residualization")
assert_true(is.finite(receipt$timings$ci_test_ms) &&
              receipt$timings$ci_test_ms < receipt$timings$total_ms,
            "ci_test_ms should not include residualization")

assign("fastkpc_mgcv_extract_gpu_gcv_for_target", function(data, target, S,
                                                           sp_grid = NULL) {
  setup_fingerprint <- paste0("setup-", target)
  list(
    residuals = as.numeric(data[, target]) - mean(data[, target]),
    fitted = rep(mean(data[, target]), nrow(data)),
    setup_fingerprint = setup_fingerprint,
    setup_fingerprint_full = list(fingerprint = setup_fingerprint),
    sp = 0.5 + target,
    score = 1 + target,
    edf = 2 + target,
    grid = data.frame(sp = c(0.1, 1), gcv = c(2, 1)),
    fit = list(
      used_device = "cuda",
      native_gpu_solve_used = TRUE,
      setup_fingerprint = list(fingerprint = setup_fingerprint),
      sp_selection_backend_executed = "r-cpu-spectral",
      gcv_score_backend_executed = "r-cpu-spectral",
      selected_solve_backend_executed = "cuda"
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
