source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase35_contracts.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_error <- function(expression, pattern, message) {
  error <- tryCatch({
    force(expression)
    NULL
  }, error = identity)
  assert_true(inherits(error, "error"), message)
  assert_true(
    grepl(pattern, conditionMessage(error), fixed = TRUE),
    paste0(message, ": unexpected error: ", conditionMessage(error))
  )
  invisible(error)
}

if (!isTRUE(tryCatch(
  fastkpc_cuda_available(),
  error = function(error) FALSE
))) {
  cat("SKIP Phase 10 one-call hardening: CUDA unavailable\n")
  quit(save = "no", status = 0L)
}

resource_snapshot <- function() {
  .Call("C_fixed_sp_cuda_test_resource_snapshot", PACKAGE = "fastkpc_cuda")
}

active_resource_counts <- function(value) {
  fields <- grep("_active_count$", names(value), value = TRUE)
  stats::setNames(as.numeric(unlist(value[fields], use.names = FALSE)), fields)
}

inject_acquire_failure <- function(resource) {
  invisible(.Call(
    "C_fixed_sp_cuda_test_inject_next_resource_acquire_failure",
    as.character(resource), PACKAGE = "fastkpc_cuda"
  ))
}

normalize_sepsets <- function(value) {
  unname(lapply(unname(value), function(row) {
    unname(lapply(unname(row), function(entry) sort(as.integer(entry))))
  }))
}

same_semantics <- function(left, right) {
  identical(left$adjacency, right$adjacency) &&
    identical(normalize_sepsets(left$sepsets), normalize_sepsets(right$sepsets)) &&
    identical(as.integer(left$n.edgetests), as.integer(right$n.edgetests)) &&
    identical(
      left$tasks[, c(
        "level", "task_index", "edge_x", "edge_y", "x", "y", "S_key",
        "native_edge_deleted"
      )],
      right$tasks[, c(
        "level", "task_index", "edge_x", "edge_y", "x", "y", "S_key",
        "native_edge_deleted"
      )]
    ) &&
    identical(left$tasks$p_used, right$tasks$p_used)
}

run_candidate <- function(data, max_conditioning_size = 1L) {
  result <- precision_run_skeleton_full_cuda_native(
    data = data,
    alpha = 0.1,
    max_conditioning_size = max_conditioning_size,
    index = 1,
    numCol = 35L,
    trace_level = "logical",
    compatible_cuda_strict = TRUE
  )
  labels <- colnames(data)
  dimnames(result$adjacency) <- list(labels, labels)
  dimnames(result$pMax) <- list(labels, labels)
  names(result$sepsets) <- labels
  result$sepsets <- lapply(result$sepsets, function(row) {
    names(row) <- labels
    row
  })
  result
}

cache_control <- function(action, capacity = NULL) {
  invisible(full_cuda_ci_one_call_cache_control_native(action, capacity))
}

set.seed(6011)
n <- 72L
z <- stats::runif(n, -2, 2)
data <- cbind(
  x1 = sin(z) + stats::rnorm(n, sd = 0.08),
  x2 = cos(z) + stats::rnorm(n, sd = 0.08),
  x3 = z + stats::rnorm(n, sd = 0.08),
  x4 = z^2 + stats::rnorm(n, sd = 0.08)
)

on.exit({
  try(full_cuda_ci_one_call_cache_control_native("configure", 262144L),
      silent = TRUE)
  try(full_cuda_ci_one_call_cache_control_native(
    "configure_target", 131072L
  ), silent = TRUE)
  try(full_cuda_ci_one_call_cache_control_native("reset"), silent = TRUE)
}, add = TRUE)

# Unsupported public semantics and invalid inputs must fail before replay.
unsupported_errors <- list()
unsupported_errors$alpha <- assert_error(
  precision_run_skeleton_full_cuda_native(
    data, 0.05, 1L, 1, 35L, "logical", TRUE
  ),
  "alpha is outside the qualified value 0.1",
  "unsupported alpha must fail closed"
)
unsupported_errors$index <- assert_error(
  precision_run_skeleton_full_cuda_native(
    data, 0.1, 1L, 2, 35L, "logical", TRUE
  ),
  "index is outside the qualified value 1",
  "unsupported index must fail closed"
)
unsupported_errors$num_col <- assert_error(
  precision_run_skeleton_full_cuda_native(
    data, 0.1, 1L, 1, 34L, "logical", TRUE
  ),
  "numCol is outside the qualified value 35",
  "unsupported numCol must fail closed"
)
unsupported_errors$strict <- assert_error(
  precision_run_skeleton_full_cuda_native(
    data, 0.1, 1L, 1, 35L, "logical", FALSE
  ),
  "requires strict mode",
  "non-strict one-call execution must fail closed"
)
nonfinite <- data
nonfinite[1L, 1L] <- NA_real_
unsupported_errors$na_input <- assert_error(
  run_candidate(nonfinite),
  "data must contain finite doubles",
  "NA input must fail closed"
)
nonfinite[1L, 1L] <- Inf
unsupported_errors$infinite_input <- assert_error(
  run_candidate(nonfinite),
  "data must contain finite doubles",
  "infinite input must fail closed"
)

# Four capacity points must reconstruct the same canonical result.
cache_control("configure", 4096L)
cache_control("configure_target", 4096L)
cache_control("reset")
reference <- run_candidate(data)
capacity_rows <- lapply(c(1L, 2L, 4L, 4096L), function(capacity) {
  cache_control("configure", capacity)
  cache_control("configure_target", capacity)
  cache_control("reset")
  result <- run_candidate(data)
  assert_true(
    same_semantics(reference, result) &&
      result$summary$result_cache_capacity == capacity &&
      result$summary$target_cache_capacity == capacity &&
      result$summary$authority_gate_pass,
    paste0("cache capacity ", capacity, " changed canonical semantics")
  )
  data.frame(
    capacity = capacity,
    result_evictions = result$summary$result_cache_eviction_count,
    target_evictions = result$summary$target_cache_eviction_count,
    pass = TRUE,
    stringsAsFactors = FALSE
  )
})
capacity_rows <- do.call(rbind, capacity_rows)
assert_true(
  all(capacity_rows$pass) &&
    all(capacity_rows$result_evictions[capacity_rows$capacity < 4096L] > 0L) &&
    any(capacity_rows$target_evictions > 0L),
  "low-capacity sweeps did not exercise deterministic reconstruction"
)

# Collinear conditioning variables and near constants must remain finite.
set.seed(6012)
rank_z <- stats::runif(n, -1.5, 1.5)
rank_data <- cbind(
  x1 = rank_z,
  x2 = rank_z,
  x3 = rank_z + stats::rnorm(n, sd = 1e-6),
  x4 = sin(rank_z) + stats::rnorm(n, sd = 1e-5),
  x5 = rank_z^2 + stats::rnorm(n, sd = 1e-5)
)
cache_control("configure", 4096L)
cache_control("configure_target", 4096L)
cache_control("reset")
rank_first <- run_candidate(rank_data, 2L)
cache_control("reset")
rank_second <- run_candidate(rank_data, 2L)
assert_true(
  same_semantics(rank_first, rank_second) &&
    all(is.finite(rank_first$tasks$p_used)) &&
    rank_first$summary$authority_gate_pass,
  "rank-deficient complete-route replay is non-deterministic or non-finite"
)

set.seed(6013)
near_constant_data <- cbind(
  x1 = stats::rnorm(n),
  x2 = stats::rnorm(n),
  x3 = stats::rnorm(n),
  x4 = 1 + stats::rnorm(n, sd = 1e-10)
)
cache_control("reset")
near_constant <- run_candidate(near_constant_data, 1L)
assert_true(
  all(is.finite(near_constant$tasks$p_used)) &&
    near_constant$summary$authority_gate_pass,
  "near-constant complete-route replay is non-finite"
)

# Injected CUDA allocation failures must leave no live tracked resources.
cache_control("reset")
before_oom <- resource_snapshot()
inject_acquire_failure("cuda_device")
oom_error <- assert_error(
  run_candidate(data),
  "injected tracked fixed-sp resource acquire failure: cuda_device",
  "injected CUDA OOM must fail closed"
)
after_oom <- resource_snapshot()
assert_true(
  identical(active_resource_counts(before_oom),
            active_resource_counts(after_oom)) &&
    after_oom$cuda_device_allocate_failure_count ==
      before_oom$cuda_device_allocate_failure_count + 1,
  "injected CUDA OOM leaked a tracked resource"
)

before_stream <- resource_snapshot()
inject_acquire_failure("stream")
stream_error <- assert_error(
  run_candidate(data),
  "injected tracked fixed-sp resource acquire failure: stream",
  "injected CUDA stream error must fail closed"
)
after_stream <- resource_snapshot()
assert_true(
  identical(active_resource_counts(before_stream),
            active_resource_counts(after_stream)) &&
    after_stream$stream_create_failure_count ==
      before_stream$stream_create_failure_count + 1,
  "injected CUDA stream failure leaked a tracked resource"
)

# ABI and capability mismatches must never execute an unqualified route.
abi <- fastkpc_full_cuda_phase35_semantic_abi_info()
abi_major_error <- assert_error(
  fastkpc_full_cuda_phase35_negotiate_semantic_abi(
    abi, required_major = 2L
  ),
  "semantic ABI major mismatch",
  "semantic ABI major mismatch must fail closed"
)
abi_capability_error <- assert_error(
  fastkpc_full_cuda_phase35_negotiate_semantic_abi(
    abi, required_capabilities = "unknown-phase10-capability-v1"
  ),
  "required semantic capability is unknown",
  "unknown required capability must fail closed"
)

# Repeated complete calls must return the tracked CUDA ledger to its baseline.
cache_control("configure", 4096L)
cache_control("configure_target", 4096L)
cache_control("reset")
leak_before <- resource_snapshot()
repeat_results <- lapply(seq_len(12L), function(index) run_candidate(data))
leak_after <- resource_snapshot()
assert_true(
  all(vapply(repeat_results, function(value) {
    same_semantics(reference, value) && value$summary$authority_gate_pass
  }, logical(1L))) &&
    identical(active_resource_counts(leak_before),
              active_resource_counts(leak_after)),
  "repeated one-call execution leaked tracked CUDA resources"
)

evidence_path <- Sys.getenv(
  "FASTKPC_PHASE10_HARDENING_EVIDENCE_RDS", unset = ""
)
if (nzchar(evidence_path)) {
  failure_errors <- c(
    unsupported_errors,
    list(
      cuda_device_oom = oom_error,
      cuda_stream = stream_error,
      abi_major = abi_major_error,
      abi_capability = abi_capability_error
    )
  )
  failure_cases <- data.frame(
    case_id = names(failure_errors),
    error_message = vapply(
      failure_errors, conditionMessage, character(1L)
    ),
    fail_closed = TRUE,
    partial_graph_published = FALSE,
    stringsAsFactors = FALSE
  )
  pathology_cases <- data.frame(
    case_id = c("rank-deficient", "near-constant"),
    logical_test_count = c(
      nrow(rank_first$tasks), nrow(near_constant$tasks)
    ),
    all_finite = c(
      all(is.finite(rank_first$tasks$p_used)),
      all(is.finite(near_constant$tasks$p_used))
    ),
    deterministic = c(same_semantics(rank_first, rank_second), TRUE),
    authority_gate_pass = c(
      rank_first$summary$authority_gate_pass,
      near_constant$summary$authority_gate_pass
    ),
    stringsAsFactors = FALSE
  )
  resource_rows <- do.call(rbind, lapply(list(
    cuda_oom_before = before_oom,
    cuda_oom_after = after_oom,
    stream_before = before_stream,
    stream_after = after_stream,
    repeated_before = leak_before,
    repeated_after = leak_after
  ), function(value) {
    data.frame(
      resource = names(active_resource_counts(value)),
      active_count = unname(active_resource_counts(value)),
      stringsAsFactors = FALSE
    )
  }))
  resource_rows$snapshot <- rep(
    names(list(
      cuda_oom_before = before_oom,
      cuda_oom_after = after_oom,
      stream_before = before_stream,
      stream_after = after_stream,
      repeated_before = leak_before,
      repeated_after = leak_after
    )),
    each = length(active_resource_counts(before_oom))
  )
  resource_rows <- resource_rows[, c(
    "snapshot", "resource", "active_count"
  )]
  saveRDS(list(
    schema_version = "full-cuda-ci-phase10-hardening-evidence-v1",
    captured_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
    failure_cases = failure_cases,
    capacity_sweep = capacity_rows,
    pathology_cases = pathology_cases,
    resource_snapshots = resource_rows,
    repeated_run_count = length(repeat_results),
    representative_result = reference,
    pass = TRUE
  ), evidence_path, compress = "xz")
}

cat(
  "PASS Phase 10 one-call hardening; capacity_points=",
  nrow(capacity_rows),
  " repeated_runs=", length(repeat_results),
  "\n", sep = ""
)
