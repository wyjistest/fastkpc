source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!isTRUE(tryCatch(
  fastkpc_cuda_available(),
  error = function(error) FALSE
))) {
  cat("SKIP strict method permutation overlap: CUDA unavailable\n")
  quit(save = "no", status = 0L)
}

resource_snapshot <- function() {
  value <- .Call(
    "C_fixed_sp_cuda_test_resource_snapshot", PACKAGE = "fastkpc_cuda"
  )
  fields <- grep("_active_count$", names(value), value = TRUE)
  stats::setNames(as.numeric(unlist(value[fields], use.names = FALSE)), fields)
}

set.seed(4817)
n <- 48L
p <- 6L
common <- stats::rnorm(n)
data <- sapply(seq_len(p), function(index) {
  common + 0.18 * stats::rnorm(n) +
    0.05 * sin(common * (index + 1L))
})
colnames(data) <- paste0("x", seq_len(p))

run_method <- function(method, overlap) {
  Sys.setenv(
    FASTKPC_STRICT_METHOD_PERMUTATION_GPU_OVERLAP =
      if (overlap) "1" else "0",
    FASTKPC_STRICT_METHOD_DEFERRED_SVD_SUBMISSION = "1",
    FASTKPC_STRICT_PERMUTATION_RESIDUAL_ROUTE = "stable-svd"
  )
  invisible(full_cuda_ci_one_call_cache_control_native("reset"))
  before <- resource_snapshot()
  set.seed(707L)
  result <- precision_run_skeleton_full_cuda_native(
    data = data,
    alpha = 0.1,
    max_conditioning_size = 3L,
    index = 1,
    numCol = 35L,
    trace_level = "logical",
    compatible_cuda_strict = TRUE,
    ci_method = method,
    hsic_params = list(sig = 1),
    permutation_params = list(
      replicates = 20L,
      seed = 707L,
      include_observed = TRUE
    )
  )
  rng_state <- .Random.seed
  after <- resource_snapshot()
  assert_true(
    identical(before, after),
    paste(method, "permutation overlap leaked CUDA resources")
  )
  list(result = result, rng_state = rng_state)
}

on.exit(Sys.unsetenv(
  "FASTKPC_STRICT_METHOD_PERMUTATION_GPU_OVERLAP",
  "FASTKPC_STRICT_METHOD_DEFERRED_SVD_SUBMISSION",
  "FASTKPC_STRICT_PERMUTATION_RESIDUAL_ROUTE"
), add = TRUE)

for (method in c("dcc.perm", "hsic.perm")) {
  synchronous <- run_method(method, FALSE)
  overlapped <- run_method(method, TRUE)
  left <- synchronous$result
  right <- overlapped$result

  assert_true(
    identical(left$tasks$logical_sequence_id,
              right$tasks$logical_sequence_id) &&
      identical(left$tasks$p_used, right$tasks$p_used) &&
      identical(left$tasks$native_edge_deleted,
                right$tasks$native_edge_deleted) &&
      identical(left$adjacency, right$adjacency) &&
      identical(left$sepsets, right$sepsets) &&
      identical(left$pMax, right$pMax) &&
      identical(left$n.edgetests, right$n.edgetests) &&
      identical(synchronous$rng_state, overlapped$rng_state),
    paste(method, "permutation overlap changed strict method semantics")
  )

  sync_summary <- left$summary
  overlap_summary <- right$summary
  assert_true(
    !isTRUE(sync_summary$method_permutation_gpu_overlap_enabled) &&
      sync_summary$method_preparation_submit_before_rng_count == 0L &&
      sync_summary$method_preparation_ready_at_submit_count == 0L &&
      sync_summary$method_preparation_ready_after_permutation_count == 0L &&
      sync_summary$method_preparation_submit_host_ms == 0 &&
      sync_summary$method_permutation_gpu_overlap_lower_bound_ms == 0 &&
      isTRUE(overlap_summary$method_permutation_gpu_overlap_enabled) &&
      overlap_summary$method_preparation_submit_before_rng_count ==
        overlap_summary$frontier_batch_count &&
      overlap_summary$method_deferred_preparation_error_count == 0L &&
      overlap_summary$method_deferred_svd_submission_count > 0L &&
      overlap_summary$method_nonblocking_preparation_submit_count >=
        overlap_summary$method_deferred_svd_submission_count &&
      overlap_summary$method_submit_hidden_stream_sync_count == 0L &&
      overlap_summary$method_submit_hidden_device_sync_count == 0L &&
      overlap_summary$method_submit_completion_event_wait_count == 0L &&
      overlap_summary$method_in_flight_peak == 1L &&
      overlap_summary$method_intermediate_host_event_wait_count == 0L &&
      overlap_summary$method_final_result_host_event_wait_count ==
        overlap_summary$frontier_batch_count &&
      overlap_summary$method_permutation_gpu_overlap_lower_bound_ms <=
        overlap_summary$method_permutation_gpu_overlap_upper_bound_ms &&
      overlap_summary$method_permutation_gpu_overlap_upper_bound_ms <=
        overlap_summary$method_permutation_table_host_ms,
    paste(method, "permutation overlap structural diagnostics changed")
  )
}

cat("test_full_cuda_ci_method_permutation_overlap.R: PASS\n")
