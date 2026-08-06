#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args)) args[[1L]] else
  "fastkpc/artifacts/strict_ci_methods_351x48_optimized_v1"

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
    path = normalizePath(path, winslash = "/"),
    relative_path = substring(
      normalizePath(path, winslash = "/"), nchar(root) + 2L
    ),
    bytes = unname(as.numeric(info$size)),
    sha256 = sha256_file(path)
  )
}
atomic_write_json <- function(value, path) {
  temporary <- tempfile(".optimized-method-manifest-", tmpdir = dirname(path))
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
require_true(dir.exists(output_dir), "optimized evidence directory is missing")
output_dir <- normalizePath(output_dir, winslash = "/")

baseline_root <- normalizePath(
  "fastkpc/artifacts/strict_ci_methods_351x48_v1", winslash = "/"
)
baseline_manifest <- jsonlite::read_json(
  file.path(baseline_root, "manifest.json"), simplifyVector = FALSE
)
expected <- list(
  "hsic.gamma" = c(skeleton = 322679L, orientation = 6L),
  "dcc.perm" = c(skeleton = 229675L, orientation = 11L),
  "hsic.perm" = c(skeleton = 282113L, orientation = 2L)
)

methods <- vector("list", length(expected))
names(methods) <- names(expected)
for (method in names(expected)) {
  method_dir <- file.path(output_dir, "payload", method)
  candidate_path <- file.path(method_dir, "candidate.rds")
  receipt_path <- file.path(method_dir, "wanpdag_receipt.rds")
  baseline_path <- file.path(
    baseline_root, "payload", method, "candidate.rds"
  )
  require_true(all(file.exists(c(
    candidate_path, receipt_path, baseline_path
  ))), paste(method, "payload is incomplete"))

  candidate_payload <- readRDS(candidate_path)
  candidate <- candidate_payload$result
  baseline_payload <- readRDS(baseline_path)
  baseline <- baseline_payload$result
  receipt <- readRDS(receipt_path)
  summary <- candidate$summary
  old_summary <- baseline$summary
  validation <- candidate_payload$validation
  residual_route <- if (method == "hsic.gamma") {
    summary$strict_hsic_gamma_residual_route
  } else {
    summary$strict_permutation_residual_route
  }
  permutation_accounted <- if (method == "hsic.perm") {
    isTRUE(summary$method_permutation_inline_r_index_requested) &&
      isTRUE(summary$method_permutation_inline_r_index_active) &&
      summary$method_permutation_inline_r_index_count ==
        summary$method_permutation_table_value_count &&
      summary$method_permutation_inline_r_draw_count >=
        summary$method_permutation_inline_r_index_count
  } else {
    !isTRUE(summary$method_permutation_inline_r_index_requested) &&
      !isTRUE(summary$method_permutation_inline_r_index_active) &&
      summary$method_permutation_inline_r_index_count == 0
  }
  require_true(
    identical(summary$ci_method, method) && isTRUE(validation$pass) &&
      identical(receipt$method, method) && isTRUE(receipt$pass) &&
      isTRUE(receipt$kpcalg_authority_route_pass) &&
      identical(as.integer(summary$logical_tests_consumed),
                expected[[method]][["skeleton"]]) &&
      identical(as.integer(receipt$kpcalg_authority_ci_test_count),
                expected[[method]][["orientation"]]) &&
      summary$residual_d2h_bytes == 0 &&
      summary$component_d2h_bytes == 0 &&
      summary$method_residual_cache_eviction_count == 0L &&
      identical(residual_route, "qr-through-2") &&
      identical(summary$sha256_backend, "openssl-sha256") &&
      summary$method_request_identity_build_host_ms >= 0 &&
      summary$method_request_identity_validation_host_ms >= 0 &&
      isTRUE(permutation_accounted),
    paste(method, "optimized evidence gate failed")
  )

  old_elapsed <- as.numeric(baseline_payload$elapsed)
  elapsed <- as.numeric(candidate_payload$elapsed)
  methods[[method]] <- list(
    ci_method = method,
    config = list(
      n = 351L, p = 48L, alpha = 0.1,
      max_conditioning_size = "Inf", index = 1, numCol = 35L,
      hsic_sig = 1,
      permutation_replicates = if (method == "hsic.gamma") NULL else 100L,
      permutation_seed = if (method == "hsic.gamma") NULL else 707L,
      permutation_include_observed = if (method == "hsic.gamma") NULL else TRUE
    ),
    authority = list(
      skeleton_authority = "full_cuda",
      orientation_authority = "kpcalg_cpu_wanpdag",
      native_cuda_orientation_status = "experimental"
    ),
    numerical_contract = list(
      skeleton_p_value = if (method == "hsic.gamma")
        "absolute-tolerance-1e-10" else "bitwise-exact",
      skeleton_decision_trace = "exact",
      graph_and_process = "exact",
      orientation_ci_and_pdag = "bitwise-exact"
    ),
    optimization = list(
      residual_route = residual_route,
      sha256_backend = summary$sha256_backend,
      persistent_execution_context = TRUE,
      hsic_permutation_inline_r_index =
        isTRUE(summary$method_permutation_inline_r_index_active),
      hsic_component_cache = method == "hsic.perm"
    ),
    performance = list(
      previous_skeleton_elapsed_sec = old_elapsed,
      optimized_skeleton_elapsed_sec = elapsed,
      elapsed_saved_sec = old_elapsed - elapsed,
      speedup = old_elapsed / elapsed,
      residual_solve_previous_sec =
        as.numeric(old_summary$cuda_residual_solve_host_ms) / 1000,
      residual_solve_optimized_sec =
        as.numeric(summary$cuda_residual_solve_host_ms) / 1000,
      component_build_previous_sec =
        as.numeric(old_summary$cuda_dcov_component_build_ms) / 1000,
      component_build_optimized_sec =
        as.numeric(summary$cuda_dcov_component_build_ms) / 1000,
      permutation_table_sec =
        as.numeric(summary$method_permutation_table_host_ms) / 1000,
      request_identity_build_sec =
        as.numeric(summary$method_request_identity_build_host_ms) / 1000,
      request_identity_validation_sec =
        as.numeric(summary$method_request_identity_validation_host_ms) / 1000
    ),
    physical_work = list(
      logical_residual_requests =
        as.integer(summary$logical_residual_requests),
      previous_physical_residual_fits =
        as.integer(old_summary$physical_residual_fits),
      optimized_physical_residual_fits =
        as.integer(summary$physical_residual_fits),
      unique_residual_keys = as.integer(summary$unique_residual_key_count),
      all_hit_batch_count =
        as.integer(summary$method_residual_cache_all_hit_batch_count),
      bypassed_target_count =
        as.integer(summary$method_residual_cache_bypassed_target_count),
      cache_hit_count =
        as.integer(summary$method_residual_cache_hit_count),
      cache_insert_count =
        as.integer(summary$method_residual_cache_insert_count),
      cache_eviction_count =
        as.integer(summary$method_residual_cache_eviction_count),
      cache_device_bytes =
        as.numeric(summary$method_residual_cache_device_bytes),
      gather_d2d_bytes =
        as.numeric(summary$method_residual_cache_gather_d2d_bytes),
      execution_context_call_count =
        as.integer(summary$method_execution_context_call_count),
      execution_context_reuse_count =
        as.integer(summary$method_execution_context_reuse_count),
      component_cache_hit_count =
        as.integer(summary$method_component_cache_hit_count),
      component_cache_miss_count =
        as.integer(summary$method_component_cache_miss_count),
      component_cache_eviction_count =
        as.integer(summary$method_component_cache_eviction_count),
      component_cache_device_bytes =
        as.numeric(summary$method_component_cache_device_bytes),
      permutation_table_value_count =
        as.numeric(summary$method_permutation_table_value_count),
      permutation_inline_index_count =
        as.numeric(summary$method_permutation_inline_r_index_count),
      permutation_inline_draw_count =
        as.numeric(summary$method_permutation_inline_r_draw_count),
      residual_d2h_bytes = as.numeric(summary$residual_d2h_bytes),
      component_d2h_bytes = as.numeric(summary$component_d2h_bytes)
    ),
    parity = list(
      skeleton_ci_test_count = expected[[method]][["skeleton"]],
      orientation_ci_test_count = expected[[method]][["orientation"]],
      max_abs_p_diff_from_previous = validation$max_abs_p_diff,
      p_values_within_contract = validation$p_values_within_tolerance,
      decision_flip_count = validation$decision_flip_count,
      logical_trace_identical = validation$structural_trace_identical,
      adjacency_identical = validation$adjacency_identical,
      sepsets_identical = validation$sepsets_identical,
      pmax_within_tolerance = validation$pmax_within_tolerance,
      n_edgetests_identical = validation$n_edgetests_identical,
      rng_state_identical = validation$rng_state_identical,
      orientation_trace_identical =
        receipt$kpcalg_authority_trace_structure_identical,
      orientation_decisions_identical =
        receipt$kpcalg_authority_decision_trace_identical,
      final_pdag_identical = receipt$kpcalg_authority_pdag_identical,
      authority_clean = validation$authority_clean,
      cache_accounted = validation$cache_accounted,
      pass = isTRUE(validation$pass) && isTRUE(receipt$pass)
    ),
    payloads = list(
      candidate = payload_entry(candidate_path, output_dir),
      wanpdag_receipt = payload_entry(receipt_path, output_dir),
      baseline_candidate = list(
        relative_path = file.path(
          "..", "strict_ci_methods_351x48_v1", "payload", method,
          "candidate.rds"
        ),
        sha256 = sha256_file(baseline_path)
      )
    )
  )
}

source_paths <- c(
  contract = "fastkpc/src/full_cuda_ci_contract.cpp",
  contract_api = "fastkpc/src/full_cuda_ci_contract.hpp",
  method_batch_cuda = "fastkpc/src/cuda/full_cuda_ci_method_batch.cu",
  method_batch_api = "fastkpc/src/cuda/full_cuda_ci_method_batch.hpp",
  fixed_sp_runtime = "fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu",
  fixed_sp_runtime_types =
    "fastkpc/src/cuda/mgcv_fixed_sp_runtime_types.hpp",
  one_call = "fastkpc/src/full_cuda_ci_one_call.cpp",
  candidate_runner =
    "fastkpc/tools/run_strict_ci_method_351x48_candidate.R",
  wanpdag_validator = "fastkpc/tools/verify_strict_wanpdag_trace.R",
  evidence_publisher =
    "fastkpc/tools/publish_strict_ci_methods_351x48_optimized_evidence.R",
  cache_regression =
    "fastkpc/tests/test_full_cuda_ci_method_residual_cache.R",
  strict_method_regression =
    "fastkpc/tests/test_full_cuda_ci_strict_methods.R",
  manifest_regression =
    "fastkpc/tests/test_strict_ci_methods_351x48_optimized_manifest.R",
  cuda_builder = "fastkpc/tools/build_cuda_native.sh"
)
require_true(all(file.exists(source_paths)), "source closure is incomplete")
source_states <- vapply(source_paths, git_state, character(1L))
source_hashes <- vapply(source_paths, sha256_file, character(1L))
head_commit <- git_output(c("rev-parse", "HEAD"))[[1L]]
origin_main <- git_output(c("rev-parse", "origin/main"))[[1L]]
native_path <- normalizePath(
  "fastkpc/build/fastkpc_cuda.so", winslash = "/", mustWork = TRUE
)
dataset_path <- normalizePath(
  "fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds",
  winslash = "/", mustWork = TRUE
)

manifest <- list(
  schema_version = "fastkpc-strict-ci-methods-351x48-optimized-evidence-v1",
  created_at = format(Sys.time(), tz = "UTC", usetz = TRUE),
  scope = "canonical-development-351x48-default-Inf",
  phase10_status = "ACTIVE",
  promotion_status = "NOT_PROMOTED",
  sealed_holdout_status = "SEALED_NOT_RELEASED",
  recommended_route_changed = FALSE,
  architecture = list(
    skeleton = "full-CUDA",
    orientation = "kpcalg CPU WAN-PDAG authority",
    full_cuda_wanpdag_claim = FALSE
  ),
  optimization = list(
    residual_cache_policy = "all-hit-complete-original-cohort-only",
    residual_cache_budget_bytes = 384 * 1024^2,
    residual_payload_transfer = "device-to-device-gather",
    hsic_component_policy = "parallel-rows-serial-numerical-order",
    fixed_sp_residual_route = "augmented-QR-through-2-then-stable-SVD",
    hsic_permutation_component_cache_budget_bytes = 384 * 1024^2,
    hsic_permutation_index_policy = "inline-R-Rejection-through-n32768",
    request_authentication = "OpenSSL-SHA256-no-intermediate-payload-copy"
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
    previous_elapsed_sec = sum(vapply(
      methods, function(x) x$performance$previous_skeleton_elapsed_sec,
      numeric(1L)
    )),
    optimized_elapsed_sec = sum(vapply(
      methods, function(x) x$performance$optimized_skeleton_elapsed_sec,
      numeric(1L)
    )),
    all_process_gates_pass = all(vapply(
      methods, function(x) isTRUE(x$parity$pass), logical(1L)
    ))
  ),
  provenance = list(
    head_base_commit = head_commit,
    origin_main = origin_main,
    head_equals_origin_main = identical(head_commit, origin_main),
    relevant_sources_dirty_or_untracked = any(source_states != "clean"),
    source_file_relative_paths = as.list(source_paths),
    source_file_sha256 = as.list(source_hashes),
    source_file_git_state = as.list(source_states),
    native_library_path = native_path,
    native_library_bytes = unname(as.numeric(file.info(native_path)$size)),
    native_library_sha256 = sha256_file(native_path),
    sha256_backend = "openssl-sha256",
    openssl_version = paste(system2(
      "openssl", "version", stdout = TRUE, stderr = TRUE
    ), collapse = " "),
    dataset_path = dataset_path,
    dataset_bytes = unname(as.numeric(file.info(dataset_path)$size)),
    dataset_sha256 = sha256_file(dataset_path)
  ),
  rebuild_commands = list(
    native = "sh fastkpc/tools/build_cuda_native.sh",
    candidate = paste(
      "Rscript fastkpc/tools/run_strict_ci_method_351x48_candidate.R",
      "<method> <candidate-output.rds>"
    ),
    wanpdag = paste(
      "Rscript fastkpc/tools/verify_strict_wanpdag_trace.R",
      "<candidate> <oracle> <method> <receipt> <data> 707 <tol>",
      "kpcalg_authority"
    ),
    publish = paste(
      "Rscript",
      "fastkpc/tools/publish_strict_ci_methods_351x48_optimized_evidence.R",
      "fastkpc/artifacts/strict_ci_methods_351x48_optimized_v1"
    )
  )
)
require_true(isTRUE(manifest$aggregate$all_process_gates_pass),
             "aggregate optimized evidence gate failed")
atomic_write_json(manifest, file.path(output_dir, "manifest.json"))
cat("published optimized strict-method evidence:", output_dir, "\n")
