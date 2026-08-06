source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!isTRUE(tryCatch(
  fastkpc_cuda_available(),
  error = function(error) FALSE
))) {
  cat("SKIP strict method failure order: CUDA unavailable\n")
  quit(save = "no", status = 0L)
}

resource_snapshot <- function() {
  .Call("C_fixed_sp_cuda_test_resource_snapshot", PACKAGE = "fastkpc_cuda")
}
active_resource_counts <- function(value) {
  fields <- grep("_active_count$", names(value), value = TRUE)
  stats::setNames(as.numeric(unlist(value[fields], use.names = FALSE)), fields)
}
arm_failure <- function(stage) {
  invisible(.Call(
    "C_full_cuda_ci_test_arm_method_failure",
    as.character(stage), PACKAGE = "fastkpc_cuda"
  ))
}
failure_snapshot <- function() {
  .Call(
    "C_full_cuda_ci_test_method_failure_snapshot",
    PACKAGE = "fastkpc_cuda"
  )
}

set.seed(4173)
n <- 48L
p <- 3L
common <- stats::rnorm(n)
data <- sapply(seq_len(p), function(index) {
  common + 0.3 * stats::rnorm(n)
})
colnames(data) <- paste0("x", seq_len(p))

run_method <- function(method) {
  precision_run_skeleton_full_cuda_native(
    data = data,
    alpha = 0.1,
    max_conditioning_size = 0L,
    index = 1,
    numCol = 35L,
    trace_level = "logical",
    compatible_cuda_strict = TRUE,
    ci_method = method,
    permutation_params = list(
      replicates = 10L,
      seed = 707L,
      include_observed = TRUE
    )
  )
}

run_failure <- function(method, stage) {
  invisible(full_cuda_ci_one_call_cache_control_native("reset"))
  set.seed(707)
  rng_before <- .Random.seed
  resources_before <- active_resource_counts(resource_snapshot())
  arm_failure(stage)
  error <- tryCatch({
    run_method(method)
    NULL
  }, error = identity)
  rng_after <- .Random.seed
  resources_after <- active_resource_counts(resource_snapshot())
  snapshot <- failure_snapshot()
  assert_true(
    inherits(error, "error") &&
      identical(
        conditionMessage(error),
        paste0("injected strict method failure: ", stage)
      ) &&
      identical(snapshot$schema_version,
                "strict-method-failure-injection-snapshot-v1") &&
      identical(snapshot$armed_stage, stage) &&
      identical(snapshot$observed_stage, stage) &&
      isTRUE(snapshot$triggered) && snapshot$checkpoint_count >= 1L,
    paste(stage, "failure injection did not reach its checkpoint")
  )
  assert_true(
    identical(resources_before, resources_after),
    paste(stage, "failure injection leaked a tracked CUDA resource")
  )
  list(before = rng_before, after = rng_after, error_class = class(error))
}

post_generation_stages <- c(
  "after_permutation_generation",
  "after_permutation_seal",
  "after_request_identity_build",
  "before_synchronous_method_call",
  "before_validate_request",
  "after_validate_request",
  "after_execution_context_validate",
  "after_prepared_identity_validate",
  "before_residual_solve",
  "after_residual_acquire",
  "after_component_wait",
  "after_pair_wait",
  "after_compact_wait",
  "after_consumer_wait",
  "after_synchronous_method_call"
)

for (method in c("dcc.perm", "hsic.perm")) {
  before_generation <- run_failure(method, "before_permutation_generation")
  assert_true(
    identical(before_generation$before, before_generation$after),
    paste(method, "pre-permutation failure advanced R RNG state")
  )

  post_generation <- lapply(
    post_generation_stages,
    function(stage) run_failure(method, stage)
  )
  reference_rng <- post_generation[[1L]]$after
  reference_error_class <- post_generation[[1L]]$error_class
  rng_matches <- vapply(post_generation, function(value) {
    identical(value$after, reference_rng)
  }, logical(1L))
  assert_true(
    !identical(post_generation[[1L]]$before, reference_rng) &&
      all(rng_matches),
    paste(
      method,
      "post-permutation failure changed R RNG consumption order at",
      paste(post_generation_stages[!rng_matches], collapse = ", ")
    )
  )
  assert_true(
    all(vapply(post_generation, function(value) {
      identical(value$error_class, reference_error_class)
    }, logical(1L))),
    paste(method, "failure checkpoints changed the observable error class")
  )
}

cat("test_full_cuda_ci_method_failure_order.R: PASS\n")
