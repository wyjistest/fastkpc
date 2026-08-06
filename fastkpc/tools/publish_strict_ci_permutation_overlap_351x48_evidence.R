#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args)) args[[1L]] else paste0(
  "fastkpc/artifacts/strict_ci_methods_351x48_permutation_overlap_v1"
)

fail <- function(message) stop(message, call. = FALSE)
require_true <- function(value, message) if (!isTRUE(value)) fail(message)
sha256_file <- function(path) {
  require_true(file.exists(path), paste("missing file:", path))
  unname(digest::digest(file = path, algo = "sha256", serialize = FALSE))
}
git_output <- function(arguments) {
  value <- suppressWarnings(system2(
    "git", arguments, stdout = TRUE, stderr = TRUE
  ))
  status <- attr(value, "status")
  require_true(is.null(status) || identical(status, 0L), paste(
    "git command failed:", paste(arguments, collapse = " ")
  ))
  value
}
git_state <- function(path) {
  value <- git_output(c("status", "--porcelain", "--", path))
  if (!length(value)) return("clean")
  if (substr(value[[1L]], 1L, 2L) == "??") "untracked" else "tracked-dirty"
}
payload_entry <- function(path, root) {
  info <- file.info(path)
  list(
    relative_path = substring(
      normalizePath(path, winslash = "/"), nchar(root) + 2L
    ),
    bytes = unname(as.numeric(info$size)),
    sha256 = sha256_file(path)
  )
}
atomic_write_json <- function(value, path) {
  temporary <- tempfile(".permutation-overlap-manifest-", tmpdir = dirname(path))
  on.exit(unlink(temporary, force = TRUE), add = TRUE)
  jsonlite::write_json(
    value, temporary, auto_unbox = TRUE, pretty = TRUE, digits = 17,
    null = "null"
  )
  require_true(file.rename(temporary, path), "failed to publish manifest")
}

require_true(requireNamespace("digest", quietly = TRUE) &&
               requireNamespace("jsonlite", quietly = TRUE),
             "digest and jsonlite are required")
require_true(dir.exists(output_dir), "overlap evidence directory is missing")
output_dir <- normalizePath(output_dir, winslash = "/")

previous_manifest_path <- paste0(
  "fastkpc/artifacts/strict_ci_methods_351x48_optimized_v2/manifest.json"
)
previous_manifest <- jsonlite::read_json(
  previous_manifest_path, simplifyVector = FALSE
)
expected <- list(
  "dcc.perm" = c(skeleton = 229675L, orientation = 11L),
  "hsic.perm" = c(skeleton = 282113L, orientation = 2L)
)

methods <- vector("list", length(expected))
names(methods) <- names(expected)
for (method in names(expected)) {
  method_dir <- file.path(output_dir, "payload", method)
  candidate_path <- file.path(method_dir, "candidate.rds")
  receipt_path <- file.path(method_dir, "wanpdag_receipt.rds")
  require_true(all(file.exists(c(candidate_path, receipt_path))),
               paste(method, "overlap payload is incomplete"))

  candidate_payload <- readRDS(candidate_path)
  candidate <- candidate_payload$result
  summary <- candidate$summary
  validation <- candidate_payload$validation
  receipt <- readRDS(receipt_path)
  prior <- previous_manifest$methods[[method]]
  elapsed <- as.numeric(candidate_payload$elapsed)
  previous_elapsed <- as.numeric(
    prior$performance$optimized_skeleton_elapsed_sec
  )
  batches <- as.integer(summary$frontier_batch_count)

  structure_ok <-
    isTRUE(summary$method_permutation_gpu_overlap_enabled) &&
    identical(as.integer(summary$method_preparation_submit_before_rng_count),
              batches) &&
    identical(as.integer(summary$method_deferred_preparation_error_count), 0L) &&
    as.integer(summary$method_preparation_ready_at_submit_count) <= batches &&
    as.integer(summary$method_preparation_ready_after_permutation_count) <=
      batches &&
    summary$method_permutation_gpu_overlap_lower_bound_ms >= 0 &&
    summary$method_permutation_gpu_overlap_lower_bound_ms <=
      summary$method_permutation_gpu_overlap_upper_bound_ms &&
    summary$method_permutation_gpu_overlap_upper_bound_ms <=
      summary$method_permutation_table_host_ms &&
    identical(as.integer(summary$method_submit_hidden_stream_sync_count), 0L) &&
    identical(as.integer(summary$method_submit_hidden_device_sync_count), 0L) &&
    identical(as.integer(summary$method_submit_completion_event_wait_count), 0L) &&
    identical(as.integer(summary$method_intermediate_host_event_wait_count), 0L) &&
    identical(as.integer(summary$method_final_result_host_event_wait_count),
              batches) &&
    identical(as.integer(summary$method_in_flight_peak), 1L) &&
    identical(as.integer(summary$method_permutation_payload_validation_scan_count),
              0L) &&
    identical(as.numeric(summary$method_permutation_payload_validation_scan_bytes),
              0) &&
    identical(as.integer(summary$method_static_identity_authentication_count),
              batches) &&
    identical(as.integer(summary$method_permutation_attestation_count), batches) &&
    identical(as.integer(summary$method_combined_identity_authentication_count),
              batches) &&
    identical(as.integer(summary$method_permutation_rng_receipt_count), batches) &&
    identical(as.integer(summary$method_permutation_rng_exact_receipt_count),
              batches)

  parity_ok <- isTRUE(validation$pass) && isTRUE(receipt$pass) &&
    identical(receipt$method, method) &&
    identical(as.integer(summary$logical_tests_consumed),
              expected[[method]][["skeleton"]]) &&
    identical(as.integer(receipt$kpcalg_authority_ci_test_count),
              expected[[method]][["orientation"]]) &&
    isTRUE(receipt$kpcalg_authority_route_pass) &&
    isTRUE(receipt$kpcalg_authority_trace_structure_identical) &&
    isTRUE(receipt$kpcalg_authority_p_within_tolerance) &&
    isTRUE(receipt$kpcalg_authority_decision_trace_identical) &&
    isTRUE(receipt$kpcalg_authority_pdag_identical) &&
    isTRUE(receipt$kpcalg_authority_rng_state_identical) &&
    summary$residual_d2h_bytes == 0 && summary$component_d2h_bytes == 0

  require_true(
    identical(summary$ci_method, method) && structure_ok && parity_ok &&
      elapsed < previous_elapsed,
    paste(method, "permutation-overlap evidence gate failed")
  )

  methods[[method]] <- list(
    ci_method = method,
    config = list(
      n = 351L, p = 48L, alpha = 0.1,
      max_conditioning_size = "Inf", index = 1, numCol = 35L,
      hsic_sig = 1, permutation_replicates = 100L,
      permutation_seed = 707L, permutation_include_observed = TRUE,
      setup_optimizer_pipeline = TRUE,
      setup_optimizer_producer_delay_us = 5000L,
      in_flight_limit = 1L
    ),
    authority = list(
      skeleton_authority = "full_cuda",
      orientation_authority = "kpcalg_cpu_wanpdag",
      native_cuda_orientation_status = "experimental"
    ),
    performance = list(
      previous_elapsed_sec = previous_elapsed,
      overlap_elapsed_sec = elapsed,
      elapsed_saved_sec = previous_elapsed - elapsed,
      reduction_fraction = (previous_elapsed - elapsed) / previous_elapsed,
      speedup = previous_elapsed / elapsed,
      classification = if (method == "dcc.perm")
        "conditional-small-improvement" else "go-material-improvement",
      previous_residual_solve_sec =
        as.numeric(prior$performance$residual_solve_optimized_sec),
      overlap_residual_solve_sec =
        as.numeric(summary$cuda_residual_solve_host_ms) / 1000,
      previous_component_build_sec =
        as.numeric(prior$performance$component_build_optimized_sec),
      overlap_component_build_sec =
        as.numeric(summary$cuda_dcov_component_build_ms) / 1000,
      previous_permutation_table_sec =
        as.numeric(prior$performance$permutation_table_sec),
      overlap_permutation_table_sec =
        as.numeric(summary$method_permutation_table_host_ms) / 1000,
      native_setup_sec = as.numeric(summary$native_setup_ms) / 1000,
      optimizer_sec = as.numeric(summary$cuda_optimizer_host_ms) / 1000
    ),
    overlap = list(
      enabled = TRUE,
      preparation_submit_before_rng_count =
        as.integer(summary$method_preparation_submit_before_rng_count),
      preparation_ready_at_submit_count =
        as.integer(summary$method_preparation_ready_at_submit_count),
      preparation_ready_after_permutation_count =
        as.integer(summary$method_preparation_ready_after_permutation_count),
      deferred_preparation_error_count =
        as.integer(summary$method_deferred_preparation_error_count),
      preparation_submit_host_sec =
        as.numeric(summary$method_preparation_submit_host_ms) / 1000,
      measured_lower_bound_sec =
        as.numeric(summary$method_permutation_gpu_overlap_lower_bound_ms) / 1000,
      modeled_upper_bound_sec =
        as.numeric(summary$method_permutation_gpu_overlap_upper_bound_ms) / 1000,
      hidden_stream_sync_count =
        as.integer(summary$method_submit_hidden_stream_sync_count),
      hidden_device_sync_count =
        as.integer(summary$method_submit_hidden_device_sync_count),
      submit_completion_event_wait_count =
        as.integer(summary$method_submit_completion_event_wait_count),
      intermediate_host_wait_count =
        as.integer(summary$method_intermediate_host_event_wait_count),
      final_host_wait_count =
        as.integer(summary$method_final_result_host_event_wait_count),
      in_flight_peak = as.integer(summary$method_in_flight_peak)
    ),
    identity_and_rng = list(
      static_authentication_count =
        as.integer(summary$method_static_identity_authentication_count),
      permutation_attestation_count =
        as.integer(summary$method_permutation_attestation_count),
      combined_authentication_count =
        as.integer(summary$method_combined_identity_authentication_count),
      exact_rng_receipt_count =
        as.integer(summary$method_permutation_rng_exact_receipt_count),
      trusted_payload_rescan_count =
        as.integer(summary$method_permutation_payload_validation_scan_count),
      rng_state_identical = validation$rng_state_identical
    ),
    parity = list(
      skeleton_ci_test_count = expected[[method]][["skeleton"]],
      orientation_ci_test_count = expected[[method]][["orientation"]],
      max_abs_p_diff = validation$max_abs_p_diff,
      p_values_bitwise_exact = validation$p_values_within_tolerance,
      decision_flip_count = validation$decision_flip_count,
      logical_trace_identical = validation$structural_trace_identical,
      adjacency_identical = validation$adjacency_identical,
      sepsets_identical = validation$sepsets_identical,
      pmax_bitwise_exact = validation$pmax_within_tolerance,
      n_edgetests_identical = validation$n_edgetests_identical,
      authority_clean = validation$authority_clean,
      cache_accounted = validation$cache_accounted,
      orientation_trace_identical =
        receipt$kpcalg_authority_trace_structure_identical,
      orientation_decisions_identical =
        receipt$kpcalg_authority_decision_trace_identical,
      orientation_p_values_bitwise_exact =
        receipt$kpcalg_authority_p_within_tolerance,
      final_pdag_identical = receipt$kpcalg_authority_pdag_identical,
      pass = TRUE
    ),
    payloads = list(
      candidate = payload_entry(candidate_path, output_dir),
      wanpdag_receipt = payload_entry(receipt_path, output_dir)
    )
  )
}

runtime_paths <- c(
  contract = "fastkpc/src/full_cuda_ci_contract.cpp",
  contract_api = "fastkpc/src/full_cuda_ci_contract.hpp",
  r_api = "fastkpc/src/r_api_cuda.cpp",
  method_batch_cuda = "fastkpc/src/cuda/full_cuda_ci_method_batch.cu",
  method_batch_api = "fastkpc/src/cuda/full_cuda_ci_method_batch.hpp",
  fixed_sp_runtime = "fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu",
  fixed_sp_runtime_types =
    "fastkpc/src/cuda/mgcv_fixed_sp_runtime_types.hpp",
  one_call = "fastkpc/src/full_cuda_ci_one_call.cpp",
  candidate_runner =
    "fastkpc/tools/run_strict_ci_method_351x48_candidate.R",
  wanpdag_validator = "fastkpc/tools/verify_strict_wanpdag_trace.R",
  overlap_regression =
    "fastkpc/tests/test_full_cuda_ci_method_permutation_overlap.R",
  failure_order_regression =
    "fastkpc/tests/test_full_cuda_ci_method_failure_order.R",
  critical_path_regression =
    "fastkpc/tests/test_full_cuda_ci_method_critical_path_diagnostics.R",
  layered_identity_regression =
    "fastkpc/tests/test_full_cuda_ci_method_layered_identity.R",
  cuda_builder = "fastkpc/tools/build_cuda_native.sh"
)
require_true(all(file.exists(runtime_paths)), "runtime source closure is incomplete")
runtime_states <- vapply(runtime_paths, git_state, character(1L))
require_true(all(runtime_states == "clean"),
             "runtime source closure is not clean")
runtime_hashes <- vapply(runtime_paths, sha256_file, character(1L))
evidence_paths <- c(
  publisher =
    "fastkpc/tools/publish_strict_ci_permutation_overlap_351x48_evidence.R",
  manifest_regression =
    "fastkpc/tests/test_strict_ci_permutation_overlap_351x48_manifest.R",
  evidence_readme = paste0(
    "fastkpc/artifacts/strict_ci_methods_351x48_permutation_overlap_v1/",
    "README.md"
  )
)
require_true(all(file.exists(evidence_paths)),
             "evidence tooling closure is incomplete")
evidence_hashes <- vapply(evidence_paths, sha256_file, character(1L))
head_commit <- git_output(c("rev-parse", "HEAD"))[[1L]]
origin_main <- git_output(c("rev-parse", "origin/main"))[[1L]]
native_path <- normalizePath(
  "fastkpc/build/fastkpc_cuda.so", winslash = "/", mustWork = TRUE
)
dataset_path <- normalizePath(
  "fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds",
  winslash = "/", mustWork = TRUE
)
previous_total <- sum(vapply(
  methods, function(value) value$performance$previous_elapsed_sec, numeric(1L)
))
overlap_total <- sum(vapply(
  methods, function(value) value$performance$overlap_elapsed_sec, numeric(1L)
))

manifest <- list(
  schema_version =
    "fastkpc-strict-ci-permutation-overlap-351x48-evidence-v1",
  created_at = format(Sys.time(), tz = "UTC", usetz = TRUE),
  scope = "canonical-development-351x48-default-Inf-permutation-methods",
  phase10_status = "ACTIVE",
  promotion_status = "NOT_PROMOTED",
  sealed_holdout_status = "SEALED_NOT_RELEASED",
  recommended_route_changed = FALSE,
  runtime_producer_commit = head_commit,
  architecture = list(
    skeleton = "full-CUDA",
    orientation = "kpcalg CPU WAN-PDAG authority",
    full_cuda_wanpdag_claim = FALSE
  ),
  methods = methods,
  aggregate = list(
    method_count = length(methods),
    skeleton_ci_test_count = sum(vapply(
      expected, `[[`, integer(1L), "skeleton"
    )),
    orientation_ci_test_count = sum(vapply(
      expected, `[[`, integer(1L), "orientation"
    )),
    previous_elapsed_sec = previous_total,
    overlap_elapsed_sec = overlap_total,
    elapsed_saved_sec = previous_total - overlap_total,
    reduction_fraction = (previous_total - overlap_total) / previous_total,
    speedup = previous_total / overlap_total,
    all_process_gates_pass = TRUE
  ),
  unchanged_method_reference = list(
    ci_method = "hsic.gamma",
    evidence_manifest =
      "../strict_ci_methods_351x48_optimized_v2/manifest.json",
    elapsed_sec = as.numeric(
      previous_manifest$methods$hsic.gamma$performance$optimized_skeleton_elapsed_sec
    ),
    rerun_in_this_evidence = FALSE
  ),
  provenance = list(
    head_commit = head_commit,
    origin_main = origin_main,
    head_equals_origin_main = identical(head_commit, origin_main),
    runtime_sources_dirty_or_untracked = any(runtime_states != "clean"),
    runtime_source_relative_paths = as.list(runtime_paths),
    runtime_source_sha256 = as.list(runtime_hashes),
    runtime_source_git_state = as.list(runtime_states),
    evidence_tool_relative_paths = as.list(evidence_paths),
    evidence_tool_sha256 = as.list(evidence_hashes),
    native_library_path = native_path,
    native_library_bytes = unname(as.numeric(file.info(native_path)$size)),
    native_library_sha256 = sha256_file(native_path),
    previous_manifest_relative_path =
      "../strict_ci_methods_351x48_optimized_v2/manifest.json",
    previous_manifest_sha256 = sha256_file(previous_manifest_path),
    dataset_path = dataset_path,
    dataset_bytes = unname(as.numeric(file.info(dataset_path)$size)),
    dataset_sha256 = sha256_file(dataset_path)
  ),
  rebuild_commands = list(
    native = "bash fastkpc/tools/build_cuda_native.sh",
    candidate = paste(
      "FASTKPC_PHASE10_SETUP_OPTIMIZER_PIPELINE=1",
      "FASTKPC_PHASE10_SETUP_OPTIMIZER_PRODUCER_DELAY_US=5000",
      "FASTKPC_STRICT_METHOD_PERMUTATION_GPU_OVERLAP=1",
      "Rscript fastkpc/tools/run_strict_ci_method_351x48_candidate.R",
      "<dcc.perm|hsic.perm> <candidate-output.rds>"
    ),
    wanpdag = paste(
      "Rscript fastkpc/tools/verify_strict_wanpdag_trace.R",
      "<candidate> <skeleton-oracle> <method> <receipt>",
      "<data> 707 0 kpcalg_authority"
    ),
    publish = paste(
      "Rscript",
      "fastkpc/tools/publish_strict_ci_permutation_overlap_351x48_evidence.R",
      "fastkpc/artifacts/strict_ci_methods_351x48_permutation_overlap_v1"
    )
  )
)
require_true(isTRUE(manifest$aggregate$all_process_gates_pass),
             "aggregate overlap evidence gate failed")
atomic_write_json(manifest, file.path(output_dir, "manifest.json"))
cat("published strict permutation overlap evidence:", output_dir, "\n")
