#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args)) args[[1L]] else
  "fastkpc/artifacts/strict_ci_methods_351x48_v1"

fail <- function(message) stop(message, call. = FALSE)
require_true <- function(value, message) if (!isTRUE(value)) fail(message)

sha256_file <- function(path) {
  require_true(file.exists(path), paste("missing file:", path))
  unname(digest::digest(
    file = path, algo = "sha256", serialize = FALSE
  ))
}

sha256_object <- function(value) {
  unname(digest::digest(value, algo = "sha256", serialize = TRUE))
}

git_output <- function(arguments) {
  output <- suppressWarnings(system2("git", arguments, stdout = TRUE,
                                     stderr = TRUE))
  status <- attr(output, "status")
  require_true(is.null(status) || identical(status, 0L), paste(
    "git command failed:", paste(arguments, collapse = " ")
  ))
  output
}

git_state <- function(path) {
  value <- git_output(c("status", "--porcelain", "--", path))
  if (!length(value)) return("clean")
  code <- substr(value[[1L]], 1L, 2L)
  if (identical(code, "??")) "untracked" else "tracked-dirty"
}

copy_authenticated <- function(source, destination) {
  dir.create(dirname(destination), recursive = TRUE, showWarnings = FALSE)
  source_hash <- sha256_file(source)
  if (file.exists(destination)) {
    require_true(identical(sha256_file(destination), source_hash), paste(
      "existing persistent payload differs:", destination
    ))
  } else {
    require_true(file.copy(source, destination, overwrite = FALSE,
                           copy.mode = TRUE, copy.date = TRUE),
                 paste("failed to persist payload:", destination))
    require_true(identical(sha256_file(destination), source_hash), paste(
      "persisted payload hash changed:", destination
    ))
  }
  info <- file.info(destination)
  list(
    path = normalizePath(destination, winslash = "/"),
    relative_path = file.path(
      "payload", basename(dirname(destination)), basename(destination)
    ),
    bytes = unname(as.numeric(info$size)),
    sha256 = source_hash
  )
}

required_receipt_gates <- c(
  "logical_trace_identical",
  "skeleton_adjacency_identical",
  "skeleton_sepsets_identical",
  "skeleton_pmax_within_tolerance",
  "skeleton_n_edgetests_identical",
  "kpcalg_authority_trace_structure_identical",
  "kpcalg_authority_p_within_tolerance",
  "kpcalg_authority_decision_trace_identical",
  "kpcalg_authority_pdag_identical",
  "kpcalg_authority_rng_state_identical",
  "kpcalg_authority_route_pass",
  "pass"
)

inputs <- list(
  "hsic.gamma" = list(
    candidate = "/tmp/fastkpc-strict-hsic-gamma-351x48-inf-svd.rds",
    oracle = "/tmp/fastkpc-strict-hsic-gamma-351x48-inf-svd-oracle.rds",
    receipt = paste0(
      "/tmp/fastkpc-strict-hsic-gamma-351x48-",
      "wanpdag-production-authority-v3.rds"
    )
  ),
  "dcc.perm" = list(
    candidate = paste0(
      "/tmp/fastkpc-strict-dcc-perm-351x48-",
      "inf-svd-2dcta-seed707.rds"
    ),
    oracle = "/tmp/fastkpc-strict-dcc-perm-351x48-inf-full-oracle-v1.rds",
    receipt = paste0(
      "/tmp/fastkpc-strict-dcc-perm-351x48-",
      "wanpdag-production-authority-v3.rds"
    )
  ),
  "hsic.perm" = list(
    candidate = paste0(
      "/tmp/fastkpc-strict-hsic-perm-351x48-",
      "inf-svd-2dcta-seed707.rds"
    ),
    oracle = "/tmp/fastkpc-strict-hsic-perm-351x48-inf-full-oracle-v1.rds",
    receipt = paste0(
      "/tmp/fastkpc-strict-hsic-perm-351x48-",
      "wanpdag-production-authority-v3.rds"
    )
  )
)

required_packages <- c("digest", "jsonlite")
missing_packages <- required_packages[!vapply(
  required_packages, requireNamespace, logical(1L), quietly = TRUE
)]
require_true(!length(missing_packages), paste(
  "missing evidence publication package(s):",
  paste(missing_packages, collapse = ", ")
))
require_true(file.exists("fastkpc/build/fastkpc_cuda.so"),
             "current CUDA native binary is missing")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
require_true(dir.exists(output_dir), "failed to create evidence directory")
output_dir <- normalizePath(output_dir, winslash = "/")

source_paths <- c(
  strict_api = "fastkpc/R/fast_kpc.R",
  wanpdag_validator = "fastkpc/tools/verify_strict_wanpdag_trace.R",
  evidence_publisher =
    "fastkpc/tools/publish_strict_ci_methods_351x48_evidence.R",
  authority_regression =
    "fastkpc/tests/test_compatible_cuda_wanpdag_authority.R",
  strict_method_regression =
    "fastkpc/tests/test_full_cuda_ci_strict_methods.R",
  manifest_regression =
    "fastkpc/tests/test_strict_ci_methods_351x48_manifest.R",
  cuda_builder = "fastkpc/tools/build_cuda_native.sh"
)
require_true(all(file.exists(source_paths)), "evidence source closure is missing")

head_commit <- git_output(c("rev-parse", "HEAD"))[[1L]]
origin_main <- git_output(c("rev-parse", "origin/main"))[[1L]]
source_states <- vapply(source_paths, git_state, character(1L))
source_hashes <- vapply(source_paths, sha256_file, character(1L))
relevant_dirty <- any(source_states != "clean")

dataset_path <- paste0(
  "fastkpc/artifacts/kpc_tprs_real_zhu/",
  "cancer_RD-causalDiscoveryInput.rds"
)
require_true(file.exists(dataset_path), "canonical dataset is missing")
native_path <- normalizePath("fastkpc/build/fastkpc_cuda.so", winslash = "/")

set.seed(707L)
permutation_initial_rng_state <- .Random.seed
methods <- vector("list", length(inputs))
names(methods) <- names(inputs)

for (method in names(inputs)) {
  paths <- inputs[[method]]
  require_true(all(file.exists(unlist(paths, use.names = FALSE))), paste(
    method, "evidence input is missing"
  ))
  candidate_payload <- readRDS(paths$candidate)
  candidate <- if (is.list(candidate_payload) &&
                   !is.null(candidate_payload$result)) {
    candidate_payload$result
  } else {
    candidate_payload
  }
  oracle <- readRDS(paths$oracle)
  receipt <- readRDS(paths$receipt)
  require_true(
    identical(candidate$summary$ci_method, method) &&
      isTRUE(oracle$pass) && identical(receipt$method, method) &&
      identical(receipt$skeleton_SHD, 0L) &&
      all(vapply(required_receipt_gates, function(field) {
        isTRUE(receipt[[field]])
      }, logical(1L))) &&
      identical(
        receipt$kpcalg_authority_diagnostics$orientation_authority,
        "kpcalg"
      ) &&
      !isTRUE(
        receipt$kpcalg_authority_diagnostics$native_orientation_executed
      ),
    paste(method, "strict receipt gate failed")
  )

  method_dir <- file.path(output_dir, "payload", method)
  payloads <- list(
    candidate = copy_authenticated(
      paths$candidate, file.path(method_dir, "candidate.rds")
    ),
    oracle = copy_authenticated(
      paths$oracle, file.path(method_dir, "skeleton_oracle.rds")
    ),
    wanpdag_receipt = copy_authenticated(
      paths$receipt, file.path(method_dir, "wanpdag_receipt.rds")
    )
  )
  rng_start <- receipt$kpcalg_authority_diagnostics$rng_start_state
  rng_end <- receipt$kpcalg_authority_diagnostics$rng_end_state
  permutation <- method %in% c("dcc.perm", "hsic.perm")
  candidate_elapsed <- if (is.list(candidate_payload) &&
                           !is.null(candidate_payload$elapsed)) {
    as.numeric(candidate_payload$elapsed)
  } else {
    NA_real_
  }
  methods[[method]] <- list(
    ci_method = method,
    config = list(
      n = 351L,
      p = 48L,
      alpha = 0.1,
      max_conditioning_size = "Inf",
      index = 1,
      numCol = 35L,
      hsic_sig = if (startsWith(method, "hsic")) 1 else NULL,
      permutation_replicates = if (permutation) 100L else NULL,
      permutation_seed = if (permutation) 707L else NULL,
      permutation_include_observed = if (permutation) TRUE else NULL
    ),
    authority = list(
      skeleton_authority = "full_cuda",
      orientation_authority = "kpcalg_cpu_wanpdag",
      native_cuda_orientation_status = "experimental"
    ),
    numerical_contract = list(
      skeleton_p_value = if (method == "hsic.gamma") {
        "absolute-tolerance-1e-10"
      } else {
        "bitwise-exact"
      },
      skeleton_decision_trace = "exact",
      graph_and_process = "exact",
      orientation_ci_and_pdag = "bitwise-exact"
    ),
    permutation_rng_contract = list(
      applicable = permutation,
      contract = if (permutation) {
        "legacy-r-global-stream-exact"
      } else {
        "not-applicable"
      },
      initial_state_sha256 = if (permutation) {
        sha256_object(permutation_initial_rng_state)
      } else {
        NULL
      },
      skeleton_end_orientation_start_state_sha256 =
        if (permutation) sha256_object(rng_start) else NULL,
      final_state_sha256 = if (permutation) sha256_object(rng_end) else NULL,
      final_state_identical_to_oracle =
        isTRUE(receipt$kpcalg_authority_rng_state_identical)
    ),
    performance = list(
      skeleton_elapsed_sec = candidate_elapsed,
      orientation_elapsed_sec =
        as.numeric(receipt$kpcalg_authority_orientation_elapsed_sec)
    ),
    parity = list(
      skeleton_ci_test_count = as.integer(receipt$skeleton_task_count),
      orientation_ci_test_count =
        as.integer(receipt$kpcalg_authority_ci_test_count),
      skeleton_SHD = as.integer(receipt$skeleton_SHD),
      skeleton_max_abs_p_diff =
        as.numeric(receipt$skeleton_max_abs_pmax_diff),
      orientation_max_abs_p_diff =
        as.numeric(receipt$kpcalg_authority_max_abs_p_diff),
      logical_trace_identical = isTRUE(receipt$logical_trace_identical),
      adjacency_identical = isTRUE(receipt$skeleton_adjacency_identical),
      sepsets_identical = isTRUE(receipt$skeleton_sepsets_identical),
      pmax_within_tolerance =
        isTRUE(receipt$skeleton_pmax_within_tolerance),
      n_edgetests_identical =
        isTRUE(receipt$skeleton_n_edgetests_identical),
      orientation_trace_identical =
        isTRUE(receipt$kpcalg_authority_trace_structure_identical),
      orientation_decisions_identical =
        isTRUE(receipt$kpcalg_authority_decision_trace_identical),
      final_pdag_identical =
        isTRUE(receipt$kpcalg_authority_pdag_identical),
      pass = isTRUE(receipt$pass)
    ),
    payloads = payloads
  )
}

manifest <- list(
  schema_version = "fastkpc-strict-ci-methods-351x48-evidence-v1",
  scope = "canonical-development-qualification",
  created_at = format(Sys.time(), tz = "UTC", usetz = TRUE),
  promotion_status = "NOT_PROMOTED",
  phase10_status = "ACTIVE",
  sealed_holdout_status = "SEALED_NOT_RELEASED",
  recommended_route_changed = FALSE,
  architecture = list(
    input = "R binary64 matrix",
    skeleton = "full-CUDA",
    orientation = "kpcalg CPU WAN-PDAG authority",
    full_cuda_wanpdag_claim = FALSE
  ),
  provenance = list(
    mode = "working-tree-execution-snapshot-v1",
    head_base_commit = head_commit,
    origin_main = origin_main,
    head_equals_origin_main = identical(head_commit, origin_main),
    producer_commit = if (relevant_dirty) NULL else head_commit,
    producer_commit_status = if (relevant_dirty) {
      "UNCOMMITTED_WORKTREE"
    } else {
      "COMMITTED"
    },
    relevant_sources_dirty_or_untracked = relevant_dirty,
    source_file_paths = as.list(normalizePath(source_paths, winslash = "/")),
    source_file_relative_paths = as.list(source_paths),
    source_file_sha256 = as.list(source_hashes),
    source_file_git_state = as.list(source_states),
    native_library_path = native_path,
    native_library_bytes = unname(as.numeric(file.info(native_path)$size)),
    native_library_sha256 = sha256_file(native_path),
    dataset_path = normalizePath(dataset_path, winslash = "/"),
    dataset_bytes = unname(as.numeric(file.info(dataset_path)$size)),
    dataset_sha256 = sha256_file(dataset_path)
  ),
  validator = list(
    identity = "verify_strict_wanpdag_trace.R:kpcalg_authority",
    schema_version = "fastkpc-strict-wanpdag-trace-oracle-v2",
    source_path = normalizePath(
      "fastkpc/tools/verify_strict_wanpdag_trace.R", winslash = "/"
    ),
    source_sha256 = sha256_file(
      "fastkpc/tools/verify_strict_wanpdag_trace.R"
    ),
    authority_regression =
      "fastkpc/tests/test_compatible_cuda_wanpdag_authority.R",
    strict_method_regression =
      "fastkpc/tests/test_full_cuda_ci_strict_methods.R"
  ),
  methods = methods,
  aggregate = list(
    method_count = length(methods),
    skeleton_ci_test_count = sum(vapply(
      methods, function(value) value$parity$skeleton_ci_test_count,
      integer(1L)
    )),
    orientation_ci_test_count = sum(vapply(
      methods, function(value) value$parity$orientation_ci_test_count,
      integer(1L)
    )),
    all_SHD_zero = all(vapply(
      methods, function(value) identical(value$parity$skeleton_SHD, 0L),
      logical(1L)
    )),
    all_process_gates_pass = all(vapply(
      methods, function(value) isTRUE(value$parity$pass), logical(1L)
    ))
  ),
  rebuild_commands = list(
    native = "bash fastkpc/tools/build_cuda_native.sh",
    strict_method_regression = paste(
      "CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1",
      "Rscript fastkpc/tests/test_full_cuda_ci_strict_methods.R"
    ),
    authority_regression = paste(
      "CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1",
      "Rscript fastkpc/tests/test_compatible_cuda_wanpdag_authority.R"
    ),
    publish = paste(
      "Rscript fastkpc/tools/publish_strict_ci_methods_351x48_evidence.R",
      "fastkpc/artifacts/strict_ci_methods_351x48_v1"
    )
  )
)

require_true(identical(manifest$aggregate$skeleton_ci_test_count, 834467L),
             "aggregate skeleton CI count changed")
require_true(identical(manifest$aggregate$orientation_ci_test_count, 19L),
             "aggregate orientation CI count changed")
require_true(isTRUE(manifest$aggregate$all_SHD_zero) &&
               isTRUE(manifest$aggregate$all_process_gates_pass),
             "aggregate strict gate failed")

manifest_path <- file.path(output_dir, "manifest.json")
temporary <- tempfile(".strict-ci-manifest-", tmpdir = output_dir)
on.exit(unlink(temporary, force = TRUE), add = TRUE)
jsonlite::write_json(
  manifest, temporary, auto_unbox = TRUE, pretty = TRUE,
  null = "null", na = "null", digits = NA
)
require_true(file.rename(temporary, manifest_path),
             "failed to publish evidence manifest")
cat(sprintf(
  paste0(
    "strict CI evidence published: methods=%d skeleton_tests=%d ",
    "orientation_tests=%d SHD0=%s process_pass=%s path=%s\n"
  ),
  manifest$aggregate$method_count,
  manifest$aggregate$skeleton_ci_test_count,
  manifest$aggregate$orientation_ci_test_count,
  manifest$aggregate$all_SHD_zero,
  manifest$aggregate$all_process_gates_pass,
  normalizePath(manifest_path, winslash = "/")
))
