source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!isTRUE(tryCatch(
  fastkpc_cuda_available(),
  error = function(error) FALSE
))) {
  cat("SKIP strict method layered identity: CUDA unavailable\n")
  quit(save = "no", status = 0L)
}

resource_snapshot <- function() {
  value <- .Call(
    "C_fixed_sp_cuda_test_resource_snapshot", PACKAGE = "fastkpc_cuda"
  )
  fields <- grep("_active_count$", names(value), value = TRUE)
  stats::setNames(as.numeric(unlist(value[fields], use.names = FALSE)), fields)
}
arm_tamper <- function(layer) {
  invisible(.Call(
    "C_full_cuda_ci_test_arm_method_identity_tamper",
    layer, PACKAGE = "fastkpc_cuda"
  ))
}
rng_state_hash <- function(state) {
  payload <- paste0(
    "schema=fastkpc-r-rng-state-v1\n",
    "length=", length(state), "\n",
    paste0(as.integer(state), "\n", collapse = "")
  )
  .Call("C_full_cuda_ci_sha256_utf8", payload, PACKAGE = "fastkpc_cuda")
}

set.seed(8401)
n <- 48L
p <- 3L
common <- stats::rnorm(n)
data <- sapply(seq_len(p), function(index) {
  common + 0.25 * stats::rnorm(n)
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
    hsic_params = list(sig = 1),
    permutation_params = list(
      replicates = 10L,
      seed = 707L,
      include_observed = TRUE
    )
  )
}

for (method in c("hsic.gamma", "dcc.perm", "hsic.perm")) {
  invisible(full_cuda_ci_one_call_cache_control_native("reset"))
  set.seed(707)
  rng_initial_sha256 <- rng_state_hash(.Random.seed)
  result <- run_method(method)
  rng_final_sha256 <- rng_state_hash(.Random.seed)
  summary <- result$summary
  permutation <- method %in% c("dcc.perm", "hsic.perm")
  rng_receipt_ok <- if (permutation) {
    summary$method_permutation_rng_receipt_count ==
        summary$frontier_batch_count &&
      summary$method_permutation_rng_exact_receipt_count ==
        summary$frontier_batch_count &&
      grepl("^[0-9a-f]{64}$",
            summary$method_permutation_rng_initial_state_sha256) &&
      grepl("^[0-9a-f]{64}$",
            summary$method_permutation_rng_final_state_sha256) &&
      identical(summary$method_permutation_rng_initial_state_sha256,
                rng_initial_sha256) &&
      identical(summary$method_permutation_rng_final_state_sha256,
                rng_final_sha256)
  } else {
    summary$method_permutation_rng_receipt_count == 0L &&
      summary$method_permutation_rng_exact_receipt_count == 0L &&
      identical(summary$method_permutation_rng_initial_state_sha256, "") &&
      identical(summary$method_permutation_rng_final_state_sha256, "")
  }
  identity_layers_ok <-
    summary$method_static_identity_authentication_count ==
        summary$frontier_batch_count &&
      summary$method_permutation_attestation_count ==
        summary$frontier_batch_count &&
      summary$method_combined_identity_authentication_count ==
        summary$frontier_batch_count
  ticket_ok <-
    summary$method_preparation_submit_count ==
        summary$frontier_batch_count &&
      summary$method_finalization_count == summary$frontier_batch_count &&
      summary$method_preparation_ticket_consumed_count ==
        summary$frontier_batch_count
  event_ok <- summary$method_in_flight_peak == 1L &&
    summary$method_intermediate_host_event_wait_count == 0L &&
    summary$method_final_result_host_event_wait_count ==
      summary$frontier_batch_count &&
    summary$method_submit_hidden_stream_sync_count == 0L &&
    summary$method_submit_hidden_device_sync_count == 0L &&
    summary$method_submit_completion_event_wait_count == 0L
  timing_ok <- summary$method_static_identity_validation_host_ms >= 0 &&
    summary$method_permutation_attestation_validation_host_ms >= 0 &&
    summary$method_combined_identity_validation_host_ms >= 0 &&
    summary$method_request_identity_validation_host_ms + 1e-9 >=
      summary$method_static_identity_validation_host_ms +
        summary$method_permutation_attestation_validation_host_ms +
        summary$method_combined_identity_validation_host_ms
  assert_true(
    identity_layers_ok && ticket_ok && event_ok && timing_ok && rng_receipt_ok,
    paste(
      method, "did not authenticate all three identity layers:",
      paste(c(
        batches = summary$frontier_batch_count,
        static = summary$method_static_identity_authentication_count,
        permutation = summary$method_permutation_attestation_count,
        combined = summary$method_combined_identity_authentication_count,
        rng_receipts = summary$method_permutation_rng_receipt_count,
        rng_exact = summary$method_permutation_rng_exact_receipt_count,
        identity_layers_ok = identity_layers_ok,
        ticket_ok = ticket_ok,
        event_ok = event_ok,
        timing_ok = timing_ok,
        rng_receipt_ok = rng_receipt_ok
      ), collapse = ", ")
    )
  )
}

expected_messages <- c(
  static = "strict CI method static identity mismatch",
  permutation = "strict CI method permutation attestation mismatch",
  combined = "strict CI method combined identity mismatch"
)
tamper_results <- lapply(names(expected_messages), function(layer) {
  invisible(full_cuda_ci_one_call_cache_control_native("reset"))
  set.seed(707)
  rng_before <- .Random.seed
  resources_before <- resource_snapshot()
  arm_tamper(layer)
  error <- tryCatch({
    run_method("dcc.perm")
    NULL
  }, error = identity)
  resources_after <- resource_snapshot()
  assert_true(
    inherits(error, "error") &&
      identical(conditionMessage(error), expected_messages[[layer]]) &&
      identical(resources_before, resources_after),
    paste(layer, "identity tamper did not fail closed")
  )
  list(rng_before = rng_before, rng_after = .Random.seed,
       error_class = class(error))
})
reference_rng <- tamper_results[[1L]]$rng_after
reference_error_class <- tamper_results[[1L]]$error_class
assert_true(
  !identical(tamper_results[[1L]]$rng_before, reference_rng) &&
    all(vapply(tamper_results, function(value) {
      identical(value$rng_after, reference_rng) &&
        identical(value$error_class, reference_error_class)
    }, logical(1L))),
  "identity tamper layers changed RNG consumption or observable error class"
)

header <- paste(
  readLines("fastkpc/src/cuda/full_cuda_ci_method_batch.hpp", warn = FALSE),
  collapse = "\n"
)
implementation_lines <- readLines(
  "fastkpc/src/cuda/full_cuda_ci_method_batch.cu", warn = FALSE
)
implementation <- paste(implementation_lines, collapse = "\n")
submit_start <- grep(
  "^MethodPreparationTicket submit_method_preparation\\(",
  implementation_lines
)
finalize_start <- grep(
  "^FullCudaCiMethodBatchResult finalize_method_from_permutation\\(",
  implementation_lines
)
submit_body <- if (length(submit_start) == 1L &&
                   length(finalize_start) == 1L &&
                   submit_start < finalize_start) {
  paste(implementation_lines[submit_start:(finalize_start - 1L)],
        collapse = "\n")
} else {
  ""
}
assert_true(
  grepl("full-cuda-ci-method-static-request-identity-v1", header,
        fixed = TRUE) &&
    grepl("full-cuda-ci-method-permutation-attestation-v2", header,
          fixed = TRUE) &&
    grepl("full-cuda-ci-method-combined-request-identity-v2", header,
          fixed = TRUE) &&
    grepl("SealedPermutationArtifact(const SealedPermutationArtifact&) = delete",
          header, fixed = TRUE) &&
    grepl("MethodPreparationTicket(const MethodPreparationTicket&) = delete",
          header, fixed = TRUE) &&
    grepl("const FullCudaCiMethodStaticRequest& request", header,
          fixed = TRUE) &&
    grepl("finalize_method_from_permutation", header, fixed = TRUE) &&
    nzchar(submit_body) &&
    !grepl("permutation_table", submit_body, fixed = TRUE) &&
    !grepl("PermutationAttestation", submit_body, fixed = TRUE) &&
    !grepl("CombinedRequestIdentity", submit_body, fixed = TRUE) &&
    grepl("planned_route[", implementation, fixed = TRUE),
  "layered identity schemas or planned-route binding changed"
)

cat("test_full_cuda_ci_method_layered_identity.R: PASS\n")
