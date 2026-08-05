source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/legacy_runner.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!isTRUE(tryCatch(
  fastkpc_cuda_available(),
  error = function(error) FALSE
))) {
  cat("SKIP strict method residual cache: CUDA unavailable\n")
  quit(save = "no", status = 0L)
}

set.seed(9231)
n <- 80L
p <- 6L
common <- stats::rnorm(n)
data <- sapply(seq_len(p), function(index) {
  common + 0.4 * stats::rnorm(n)
})
colnames(data) <- paste0("x", seq_len(p))

run_cpu <- function(method) {
  env <- fastkpc_legacy_env()
  suff_stat <- list(
    data = data,
    ic.method = method,
    index = 1,
    numCol = 35L,
    sig = 1,
    p = 100L
  )
  set.seed(707)
  skeleton <- pcalg::skeleton(
    suffStat = suff_stat,
    indepTest = env$kernelCItest,
    alpha = 0.1,
    labels = colnames(data),
    m.max = 3L,
    method = "stable"
  )
  list(
    adjacency = methods::as(skeleton@graph, "matrix") != 0,
    sepsets = skeleton@sepset,
    pMax = skeleton@pMax,
    n.edgetests = as.integer(skeleton@n.edgetests),
    rng_state = .Random.seed
  )
}

run_cuda <- function(method) {
  invisible(full_cuda_ci_one_call_cache_control_native("reset"))
  set.seed(707)
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
      replicates = 100L,
      seed = 707L,
      include_observed = TRUE
    )
  )
  list(result = result, rng_state = .Random.seed)
}

for (method in c("dcc.perm", "hsic.perm")) {
  oracle <- run_cpu(method)
  first <- run_cuda(method)
  second <- run_cuda(method)
  candidate <- first$result
  replay <- second$result
  summary <- candidate$summary

  assert_true(
    identical(unname(candidate$adjacency != 0), unname(oracle$adjacency)) &&
      identical(unname(candidate$pMax), unname(oracle$pMax)) &&
      identical(as.integer(candidate$n.edgetests), oracle$n.edgetests),
    paste(method, "cache route changed the CPU skeleton authority")
  )
  assert_true(
    identical(first$rng_state, oracle$rng_state) &&
      identical(second$rng_state, oracle$rng_state),
    paste(method, "cache route changed the R RNG terminal state")
  )
  assert_true(
    identical(candidate$tasks$p_used, replay$tasks$p_used) &&
      identical(candidate$adjacency, replay$adjacency) &&
      identical(candidate$sepsets, replay$sepsets) &&
      identical(candidate$pMax, replay$pMax) &&
      identical(candidate$n.edgetests, replay$n.edgetests),
    paste(method, "cache route is not bitwise repeatable")
  )

  assert_true(
    summary$method_residual_cache_capacity_entries >= 2 &&
      summary$method_residual_cache_all_hit_batch_count > 0L &&
      summary$method_residual_cache_bypassed_target_count > 0L &&
      summary$physical_residual_fits < summary$logical_residual_requests,
    paste(method, "fixture did not exercise physical residual reuse")
  )
  assert_true(
    summary$method_residual_cache_lookup_count ==
        summary$logical_residual_requests &&
      summary$method_residual_cache_lookup_count ==
        summary$method_residual_cache_hit_count +
          summary$method_residual_cache_insert_count &&
      summary$method_residual_cache_bypassed_target_count ==
        2L * summary$method_residual_cache_all_hit_batch_count &&
      summary$physical_residual_fits ==
        summary$logical_residual_requests -
          summary$method_residual_cache_bypassed_target_count &&
      summary$cuda_exact_screen_residual_target_count ==
        summary$physical_residual_fits &&
      2L * summary$cuda_exact_screen_residual_batch_count ==
        summary$physical_residual_fits &&
      summary$method_residual_cache_gather_d2d_bytes ==
        summary$method_residual_cache_bypassed_target_count * n * 8,
    paste(method, "residual cache physical-work accounting changed")
  )
  assert_true(
    summary$residual_d2h_bytes == 0 &&
      summary$component_d2h_bytes == 0 &&
      summary$cpu_residual_solve_count == 0L &&
      summary$unknown_fallback_count == 0L,
    paste(method, "residual cache authority failed closed")
  )
}

oracle <- run_cpu("hsic.gamma")
first <- run_cuda("hsic.gamma")
second <- run_cuda("hsic.gamma")
candidate <- first$result
replay <- second$result
summary <- candidate$summary

assert_true(
  identical(unname(candidate$adjacency != 0), unname(oracle$adjacency)) &&
    isTRUE(all.equal(
      unname(candidate$pMax), unname(oracle$pMax),
      tolerance = 1e-10, check.attributes = FALSE
    )) &&
    identical(as.integer(candidate$n.edgetests), oracle$n.edgetests),
  "hsic.gamma cache route changed the CPU skeleton authority"
)
assert_true(
  identical(first$rng_state, oracle$rng_state) &&
    identical(second$rng_state, oracle$rng_state),
  "hsic.gamma cache route unexpectedly consumed R RNG state"
)
assert_true(
  identical(candidate$tasks$p_used, replay$tasks$p_used) &&
    identical(candidate$adjacency, replay$adjacency) &&
    identical(candidate$sepsets, replay$sepsets) &&
    identical(candidate$pMax, replay$pMax) &&
    identical(candidate$n.edgetests, replay$n.edgetests),
  "hsic.gamma cache route is not bitwise repeatable"
)
assert_true(
  summary$method_residual_cache_all_hit_batch_count > 0L &&
    summary$method_residual_cache_bypassed_target_count >
      2L * summary$method_residual_cache_all_hit_batch_count &&
    summary$physical_residual_fits < summary$logical_residual_requests,
  "hsic.gamma fixture did not exercise variable-width residual reuse"
)
assert_true(
  summary$method_residual_cache_lookup_count ==
      summary$logical_residual_requests &&
    summary$method_residual_cache_lookup_count ==
      summary$method_residual_cache_hit_count +
        summary$method_residual_cache_insert_count &&
    summary$physical_residual_fits ==
      summary$logical_residual_requests -
        summary$method_residual_cache_bypassed_target_count &&
    summary$cuda_exact_screen_residual_target_count ==
      summary$physical_residual_fits &&
    summary$cuda_exact_screen_residual_batch_count +
        summary$method_residual_cache_all_hit_batch_count ==
      summary$frontier_batch_count &&
    summary$method_residual_cache_gather_d2d_bytes ==
      summary$method_residual_cache_bypassed_target_count * n * 8,
  "hsic.gamma residual cache physical-work accounting changed"
)
assert_true(
  summary$residual_d2h_bytes == 0 &&
    summary$component_d2h_bytes == 0 &&
    summary$cpu_residual_solve_count == 0L &&
    summary$unknown_fallback_count == 0L,
  "hsic.gamma residual cache authority failed closed"
)

cat("test_full_cuda_ci_method_residual_cache.R: PASS\n")
