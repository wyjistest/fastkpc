source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
require_true <- function(value, message) if (!isTRUE(value)) fail(message)

arguments <- commandArgs(trailingOnly = TRUE)
iterations <- if (length(arguments) >= 1L) as.integer(arguments[[1L]]) else 200L
output_path <- if (length(arguments) >= 2L) arguments[[2L]] else ""
requested_q <- if (length(arguments) >= 3L) as.integer(arguments[[3L]]) else 46L
if (length(iterations) != 1L || is.na(iterations) || iterations < 2L) {
  fail("iterations must be an integer >= 2")
}
if (length(requested_q) != 1L || is.na(requested_q)) {
  fail("q must be one of 28, 37, 46, 55, or 64")
}

if (!isTRUE(tryCatch(fastkpc_cuda_available(), error = function(error) FALSE))) {
  fail("CUDA is unavailable")
}
if (!requireNamespace("digest", quietly = TRUE)) {
  fail("digest is required")
}

# These setup and target states are authenticated Phase 2 payloads. Loading
# the owning shard directly avoids re-hashing the full 763 MiB setup index in
# a diagnostic benchmark.
fixtures <- data.frame(
  q = c(28L, 37L, 46L, 55L, 64L),
  shard = c(54L, 17L, 35L, 42L, 22L),
  prepared_key = c(
    "f49fd047bec777e4904a79e2899f16f09334b9a7dd3a996fc4f15f1099908584",
    "cc04945a6a1a8c04694c1c40907963b45cef6eb1768f094b7e64595e2ba6b9fd",
    "d19f0f5bb97e3fb12d4c874299d4db1c6f5f3ff72a73d20b1094f482fca84349",
    "95472b5513beb6c979720a30ef38193718bae855a90c8500fdbe3157e201cc8b",
    "aee91bac0a5e8a411a072dd25639c207f32035fa95aa5bec8081c552f0b17af6"
  ),
  stringsAsFactors = FALSE
)
fixture <- fixtures[fixtures$q == requested_q, , drop = FALSE]
if (nrow(fixture) != 1L) fail("q must be one of 28, 37, 46, 55, or 64")
prepared_key <- fixture$prepared_key[[1L]]
shard_path <- file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1",
  "shards", paste0("shard_", fixture$shard[[1L]], ".rds")
)
data_path <- file.path(
  "fastkpc", "artifacts", "kpc_tprs_real_zhu",
  "cancer_RD-causalDiscoveryInput.rds"
)
shard <- readRDS(shard_path)
setup <- shard$prepared_s_setups[[prepared_key]]
states <- shard$target_states[
  shard$target_states$prepared_s_key_sha256 == prepared_key,
  , drop = FALSE
]
states <- states[order(states$residual_key_sha256, method = "radix"),
                 , drop = FALSE]
require_true(!is.null(setup) && nrow(states) >= 2L,
             "captured stable-SVD setup is unavailable")
states <- states[seq_len(2L), , drop = FALSE]

data <- readRDS(data_path)
dataset_sha256 <- shard$manifest$dataset_matrix_sha256
targets <- lapply(seq_len(nrow(states)), function(index) {
  fastkpc_full_cuda_materialize_target_state(
    states[index, , drop = FALSE], data, dataset_sha256
  )
})
Y <- do.call(cbind, lapply(targets, `[[`, "y"))
SP <- do.call(cbind, lapply(targets, function(target) {
  as.numeric(target$row$selected_sp[[1L]])
}))
storage.mode(Y) <- storage.mode(SP) <- "double"
target_keys <- as.character(states$residual_key_sha256)
routes <- rep("AUGMENTED_SVD", 2L)
outputs <- c("coefficients", "fitted", "residuals", "rss")
dto <- fastkpc_full_cuda_fixed_sp_native_dto(setup)
require_true(
  identical(dim(Y), c(dto$n, 2L)) &&
    identical(dim(SP), c(dto$penalty_count, 2L)) &&
    identical(dto$null_dim, requested_q),
  "captured benchmark dimensions changed"
)

runtime0 <- runtime1 <- NULL
handle0 <- handle1 <- NULL
active_tokens <- list()
cleanup_token <- function(token) {
  if (is.null(token)) return(invisible(NULL))
  try(fixed_sp_cuda_residual_release(token), silent = TRUE)
  try(fixed_sp_cuda_residual_free(token), silent = TRUE)
  invisible(NULL)
}
on.exit({
  for (token in active_tokens) cleanup_token(token)
  if (!is.null(handle1)) try(fixed_sp_cuda_prepared_free(handle1), silent = TRUE)
  if (!is.null(handle0)) try(fixed_sp_cuda_prepared_free(handle0), silent = TRUE)
  if (!is.null(runtime1)) try(fixed_sp_cuda_runtime_free(runtime1), silent = TRUE)
  if (!is.null(runtime0)) try(fixed_sp_cuda_runtime_free(runtime0), silent = TRUE)
}, add = TRUE)

runtime0 <- fixed_sp_cuda_runtime_create(0L)
runtime1 <- fixed_sp_cuda_runtime_create(0L)
for (runtime in list(runtime0, runtime1)) {
  fixed_sp_cuda_runtime_reserve(
    runtime, dto$n, dto$null_dim, 2L, dto$penalty_count,
    as.integer(dto$n + dto$null_dim)
  )
}
handle0 <- fixed_sp_cuda_prepared_create(runtime0, dto)
handle1 <- fixed_sp_cuda_prepared_create(runtime1, dto)

finish_token <- function(token, materialize) {
  info <- fixed_sp_cuda_residual_info(token)
  shadow <- if (materialize) {
    fixed_sp_cuda_materialize_shadow(token, outputs)
  } else {
    NULL
  }
  fixed_sp_cuda_residual_release(token)
  fixed_sp_cuda_residual_free(token)
  list(info = info, shadow = shadow)
}

elapsed_ms <- function(started) {
  as.double((proc.time()[["elapsed"]] - started) * 1000)
}

run_a <- function(materialize = FALSE) {
  started <- proc.time()[["elapsed"]]
  token <- fixed_sp_cuda_submit_deferred_svd_for_test(
    handle0, Y, SP, routes, target_keys, outputs
  )
  submit_ms <- elapsed_ms(started)
  active_tokens <<- list(token)
  finished <- finish_token(token, materialize)
  active_tokens <<- list()
  list(
    submit_ms = submit_ms,
    completion_ms = elapsed_ms(started),
    info = finished$info,
    shadow = finished$shadow
  )
}

run_b <- function(materialize = FALSE) {
  started <- proc.time()[["elapsed"]]
  submit0_started <- proc.time()[["elapsed"]]
  token0 <- fixed_sp_cuda_submit_deferred_svd_for_test(
    handle0, Y[, 1L, drop = FALSE], SP[, 1L, drop = FALSE],
    routes[[1L]], target_keys[[1L]], outputs
  )
  submit0_ms <- elapsed_ms(submit0_started)
  active_tokens <<- list(token0)

  submit1_started <- proc.time()[["elapsed"]]
  token1 <- fixed_sp_cuda_submit_deferred_svd_for_test(
    handle1, Y[, 2L, drop = FALSE], SP[, 2L, drop = FALSE],
    routes[[2L]], target_keys[[2L]], outputs
  )
  submit1_ms <- elapsed_ms(submit1_started)
  submit_total_ms <- elapsed_ms(started)
  active_tokens <<- list(token0, token1)

  finished0 <- finish_token(token0, materialize)
  active_tokens <<- list(token1)
  finished1 <- finish_token(token1, materialize)
  active_tokens <<- list()
  list(
    submit0_ms = submit0_ms,
    submit1_ms = submit1_ms,
    submit_total_ms = submit_total_ms,
    completion_ms = elapsed_ms(started),
    info = list(finished0$info, finished1$info),
    shadow = list(finished0$shadow, finished1$shadow)
  )
}

merge_b_shadow <- function(value) {
  list(
    coefficients = cbind(value[[1L]]$coefficients,
                         value[[2L]]$coefficients),
    fitted = cbind(value[[1L]]$fitted, value[[2L]]$fitted),
    residuals = cbind(value[[1L]]$residuals, value[[2L]]$residuals),
    rss = c(value[[1L]]$rss, value[[2L]]$rss)
  )
}

merge_b_info_field <- function(value, field) {
  unname(c(value[[1L]][[field]], value[[2L]][[field]]))
}

# Warm both contexts and establish the exact serial reference before timing.
invisible(run_a(FALSE))
invisible(run_b(FALSE))
reference <- run_a(TRUE)
candidate <- run_b(TRUE)
merged_candidate_shadow <- merge_b_shadow(candidate$shadow)
shadow_diagnostics <- lapply(names(reference$shadow), function(field) {
  left <- reference$shadow[[field]]
  right <- merged_candidate_shadow[[field]]
  list(
    field = field,
    identical = identical(left, right),
    values_identical = identical(as.numeric(left), as.numeric(right)),
    attributes_identical = identical(attributes(left), attributes(right)),
    maximum_absolute_difference = if (length(left) == length(right)) {
      max(abs(as.numeric(left) - as.numeric(right)))
    } else {
      Inf
    }
  )
})
names(shadow_diagnostics) <- names(reference$shadow)
require_true(
  all(vapply(
    shadow_diagnostics, `[[`, logical(1L), "identical"
  )),
  paste(
    "dual-context SVD values differ bitwise from the serial batch:",
    paste(vapply(shadow_diagnostics, function(value) {
      paste0(value$field, "=", value$maximum_absolute_difference)
    }, character(1L)), collapse = ", ")
  )
)
for (field in c(
  "planned_route", "executed_route", "reroute_reason", "solver_status",
  "effective_rank", "sigma_max", "smallest_retained_sigma", "svd_info",
  "aggregate_penalty_root_rank", "aggregate_penalty_root_pivot",
  "aggregate_factor_call_count", "aggregate_b_build_count", "aggregate_dstop"
)) {
  require_true(
    identical(unname(reference$info[[field]]),
              merge_b_info_field(candidate$info, field)),
    paste("dual-context SVD diagnostic differs for", field)
  )
}

timings <- data.frame(
  iteration = seq_len(iterations),
  a_submit_ms = numeric(iterations),
  a_completion_ms = numeric(iterations),
  b_submit0_ms = numeric(iterations),
  b_submit1_ms = numeric(iterations),
  b_submit_total_ms = numeric(iterations),
  b_completion_ms = numeric(iterations)
)
for (iteration in seq_len(iterations)) {
  if (iteration %% 2L == 1L) {
    a <- run_a(FALSE)
    b <- run_b(FALSE)
  } else {
    b <- run_b(FALSE)
    a <- run_a(FALSE)
  }
  timings$a_submit_ms[[iteration]] <- a$submit_ms
  timings$a_completion_ms[[iteration]] <- a$completion_ms
  timings$b_submit0_ms[[iteration]] <- b$submit0_ms
  timings$b_submit1_ms[[iteration]] <- b$submit1_ms
  timings$b_submit_total_ms[[iteration]] <- b$submit_total_ms
  timings$b_completion_ms[[iteration]] <- b$completion_ms
}

quantiles <- function(value) {
  stats::setNames(
    as.numeric(stats::quantile(value, c(0.5, 0.9, 0.95, 0.99))),
    c("p50", "p90", "p95", "p99")
  )
}
native_library_path <- normalizePath(
  "fastkpc/build/fastkpc_cuda.so", winslash = "/", mustWork = TRUE
)
summary <- list(
  schema_version = "fixed-sp-svd-lane-benchmark-v2",
  device_id = 0L,
  native_library_path = native_library_path,
  native_library_sha256 = unname(digest::digest(
    file = native_library_path, algo = "sha256", serialize = FALSE
  )),
  prepared_s_key_sha256 = prepared_key,
  target_keys = target_keys,
  n = dto$n,
  q = dto$null_dim,
  penalty_count = dto$penalty_count,
  iterations = iterations,
  bitwise_output_exact = TRUE,
  split_after_shared_rhs_required = TRUE,
  shadow_diagnostics = shadow_diagnostics,
  bitwise_diagnostics_exact = TRUE,
  a_submit_ms = quantiles(timings$a_submit_ms),
  a_completion_ms = quantiles(timings$a_completion_ms),
  b_submit_total_ms = quantiles(timings$b_submit_total_ms),
  b_completion_ms = quantiles(timings$b_completion_ms),
  median_submit_speedup = stats::median(timings$a_submit_ms) /
    stats::median(timings$b_submit_total_ms),
  median_completion_speedup = stats::median(timings$a_completion_ms) /
    stats::median(timings$b_completion_ms),
  runtime0 = fixed_sp_cuda_runtime_info(runtime0),
  runtime1 = fixed_sp_cuda_runtime_info(runtime1)
)
result <- list(
  summary = summary,
  timings = timings,
  reference = list(info = reference$info, shadow = reference$shadow)
)
if (nzchar(output_path)) saveRDS(result, output_path, version = 3L)

cat(
  "fixed-SP stable-SVD single-GPU A/B\n",
  "  shape: (", dto$n + dto$null_dim, " x ", dto$null_dim, ")\n",
  "  A completion p50: ", summary$a_completion_ms[["p50"]], " ms\n",
  "  B completion p50: ", summary$b_completion_ms[["p50"]], " ms\n",
  "  completion speedup: ", summary$median_completion_speedup, "x\n",
  "  submit speedup: ", summary$median_submit_speedup, "x\n",
  "  bitwise exact: TRUE\n",
  sep = ""
)
