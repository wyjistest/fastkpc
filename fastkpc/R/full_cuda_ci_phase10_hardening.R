fastkpc_full_cuda_phase10_hardening_require <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
  invisible(TRUE)
}

fastkpc_full_cuda_phase10_hardening_artifact_dir <- function(
    root = file.path("fastkpc", "artifacts", "full_cuda_ci")) {
  file.path(root, "failure_injection_default_inf_v2")
}

fastkpc_full_cuda_phase10_hardening_required_files <- function() {
  c(
    "manifest.json", "summary.json", "summary.md", "commands.txt",
    "environment.txt", "graph_agreement.csv", "sepset_agreement.csv",
    "n_edgetests.csv", "deletion_trace.csv", "first_divergence.json",
    "fallbacks.csv", "stage_timing.csv", "raw_runs.csv",
    "case_results.csv", "near_alpha_results.csv",
    "rank_condition_results.csv", "pathology_results.csv", "cache.csv",
    "resource_snapshots.csv", "stream_results.csv", "test_results.csv",
    "test_logs.txt",
    "evidence_inputs.csv", "source_closure.csv", "source_evidence.rds",
    "producer_identity.json", "backend_configuration.json",
    "build_recipe.json", "validator_attestations.json",
    "execution_receipts.json"
  )
}

fastkpc_full_cuda_phase10_hardening_source_paths <- function() {
  paths <- sort(unique(c(
    fastkpc_full_cuda_phase9_source_paths(),
    "fastkpc/R/full_cuda_ci_phase10_hardening.R",
    "fastkpc/tests/test_full_cuda_ci_one_call_cache.R",
    "fastkpc/tests/test_full_cuda_ci_phase10_hardening_artifact.R",
    "fastkpc/tests/test_full_cuda_ci_phase10_hardening.R",
    "fastkpc/tests/test_full_cuda_ci_phase10_stream_determinism.R",
    "fastkpc/tests/test_full_cuda_ci_default_inf_level8.R",
    "fastkpc/tests/test_full_cuda_ci_default_inf_level9_capacity.R",
    "fastkpc/tests/fixtures/default_inf_level8_oracle_v1.json",
    "fastkpc/tests/fixtures/default_inf_production_cuda_v1.json",
    "fastkpc/tools/analyze_full_cuda_ci_phase10_frontier.R",
    "fastkpc/tools/run_full_cuda_ci_phase10_hardening.R"
  )), method = "radix")
  fastkpc_full_cuda_phase10_hardening_require(
    all(file.exists(paths) & !dir.exists(paths)) && !anyDuplicated(paths),
    "Phase 10 hardening source closure is incomplete"
  )
  paths
}

fastkpc_full_cuda_phase10_hardening_source_closure <- function() {
  paths <- fastkpc_full_cuda_phase10_hardening_source_paths()
  hashes <- setNames(as.list(vapply(
    paths, fastkpc_full_cuda_census_file_hash, character(1L)
  )), paths)
  list(
    table = data.frame(
      path = names(hashes),
      sha256 = unlist(hashes, use.names = FALSE),
      stringsAsFactors = FALSE
    ),
    hashes = hashes,
    sha256 = fastkpc_full_cuda_phase35_sha256_utf8(
      fastkpc_full_cuda_phase35_canonical_json(hashes)
    )
  )
}

fastkpc_full_cuda_phase10_hardening_backend_configuration <- function() {
  contract <- fastkpc_full_cuda_default_kpcalg_contract()
  value <- list(
    schema_version = "full-cuda-ci-phase10-hardening-configuration-v2",
    candidate_route = "compatible.cuda/full_cuda-explicit",
    native_entrypoint = "compatible-cuda-full-skeleton-native-v1",
    scheduler = "cache-aware-frontier-4x-v1",
    compact_result_cache_capacity = 262144L,
    target_state_cache_capacity = 131072L,
    prepared_setup_cache_capacity = 64L,
    component_capacity = 47L,
    capacity_sweep = list(1L, 2L, 4L, 4096L),
    stream_count_sweep = list(1L, 2L, 4L),
    injected_resource_failures = list("cuda_device", "stream"),
    strict_fail_closed = TRUE,
    approximate_fallback = FALSE,
    alpha = "0.1",
    index = 1L,
    num_col = 35L,
    requested_max_conditioning_size = contract$requested_max_conditioning_size,
    resolved_max_conditioning_size = contract$resolved_max_conditioning_size,
    natural_stop_level = contract$natural_stop_level,
    logical_test_count = contract$logical_test_count,
    default_inf_full_regression_required = TRUE,
    default_inf_level8_qualification_required = TRUE,
    default_inf_extended_capacity_qualification_required = TRUE,
    precision = "float64",
    fmad = FALSE,
    fast_math = FALSE
  )
  list(
    value = value,
    sha256 = fastkpc_full_cuda_phase35_sha256_utf8(
      fastkpc_full_cuda_phase35_canonical_json(value)
    )
  )
}

fastkpc_full_cuda_phase10_hardening_build_recipe <- function() {
  value <- list(
    schema_version = "full-cuda-ci-phase10-hardening-build-recipe-v2",
    build_script_sha256 = fastkpc_full_cuda_census_file_hash(
      "fastkpc/tools/build_cuda_native.sh"
    ),
    CXX17 = fastkpc_full_cuda_command_output(
      "R", c("CMD", "config", "CXX17")
    ),
    R_version = R.version.string,
    nvcc_path = "/usr/local/cuda/bin/nvcc",
    nvcc_version = fastkpc_full_cuda_command_output(
      "/usr/local/cuda/bin/nvcc", "--version"
    ),
    cuda_architecture = "sm_89",
    fmad = FALSE,
    fast_math = FALSE
  )
  list(
    value = value,
    sha256 = fastkpc_full_cuda_phase35_sha256_utf8(
      fastkpc_full_cuda_phase35_canonical_json(value)
    )
  )
}

fastkpc_full_cuda_phase10_hardening_object_hash <- function(value) {
  fastkpc_full_cuda_census_hash_raw(serialize(value, NULL, version = 2L))
}

fastkpc_full_cuda_phase10_hardening_stream_rows <- function(evidence) {
  data.frame(
    stream_count = as.integer(evidence$stream_counts),
    selected_log_sp_sha256 = vapply(
      evidence$selected_log_sp,
      fastkpc_full_cuda_phase10_hardening_object_hash,
      character(1L)
    ),
    optimizer_iterations_sha256 = vapply(
      evidence$optimizer_iterations,
      fastkpc_full_cuda_phase10_hardening_object_hash,
      character(1L)
    ),
    score_calls_sha256 = vapply(
      evidence$score_calls,
      fastkpc_full_cuda_phase10_hardening_object_hash,
      character(1L)
    ),
    optimizer_status_ok = vapply(
      evidence$optimizer_status,
      function(value) all(as.integer(value) == 0L), logical(1L)
    ),
    pass = TRUE,
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_phase10_validate_hardening_evidence <- function(
    hardening, streams) {
  required_failure_cases <- c(
    "alpha", "index", "num_col", "strict", "na_input", "infinite_input",
    "cuda_device_oom", "cuda_stream", "abi_major", "abi_capability"
  )
  hardening_clean <- is.list(hardening) && identical(
    hardening$schema_version,
    "full-cuda-ci-phase10-hardening-evidence-v1"
  ) && isTRUE(hardening$pass) &&
    is.data.frame(hardening$failure_cases) &&
    identical(as.character(hardening$failure_cases$case_id),
              required_failure_cases) &&
    all(nzchar(hardening$failure_cases$error_message)) &&
    all(hardening$failure_cases$fail_closed) &&
    !any(hardening$failure_cases$partial_graph_published) &&
    is.data.frame(hardening$capacity_sweep) &&
    identical(as.integer(hardening$capacity_sweep$capacity),
              c(1L, 2L, 4L, 4096L)) &&
    all(hardening$capacity_sweep$pass) &&
    all(hardening$capacity_sweep$result_evictions[1:3] > 0L) &&
    any(hardening$capacity_sweep$target_evictions > 0L) &&
    is.data.frame(hardening$pathology_cases) &&
    identical(as.character(hardening$pathology_cases$case_id),
              c("rank-deficient", "near-constant")) &&
    all(hardening$pathology_cases$all_finite) &&
    all(hardening$pathology_cases$deterministic) &&
    all(hardening$pathology_cases$authority_gate_pass) &&
    is.data.frame(hardening$resource_snapshots) &&
    nrow(hardening$resource_snapshots) > 0L &&
    all(hardening$resource_snapshots$active_count == 0) &&
    hardening$repeated_run_count >= 12L &&
    fastkpc_full_cuda_is_skeleton(hardening$representative_result) &&
    isTRUE(hardening$representative_result$summary$authority_gate_pass) &&
    all(is.finite(hardening$representative_result$tasks$p_used))
  stream_rows <- if (is.list(streams)) {
    fastkpc_full_cuda_phase10_hardening_stream_rows(streams)
  } else data.frame()
  stream_clean <- is.list(streams) && identical(
    streams$schema_version,
    "full-cuda-ci-phase10-stream-evidence-v1"
  ) && isTRUE(streams$pass) &&
    identical(as.integer(streams$stream_counts), c(1L, 2L, 4L)) &&
    nrow(stream_rows) == 3L && all(stream_rows$optimizer_status_ok) &&
    length(unique(stream_rows$selected_log_sp_sha256)) == 1L &&
    length(unique(stream_rows$optimizer_iterations_sha256)) == 1L &&
    length(unique(stream_rows$score_calls_sha256)) == 1L
  fastkpc_full_cuda_phase10_hardening_require(
    hardening_clean && stream_clean,
    "Phase 10 hardening evidence failed validation"
  )
  invisible(list(hardening = hardening, streams = streams))
}

fastkpc_full_cuda_phase10_hardening_write_table <- function(
    value, directory, name) {
  utils::write.csv(
    value, file.path(directory, name), row.names = FALSE, na = ""
  )
}

fastkpc_full_cuda_phase10_hardening_evidence_inputs <- function(
    hardening_path, stream_path) {
  paths <- c(
    hardening_evidence = hardening_path,
    stream_evidence = stream_path,
    phase8_backend_manifest = file.path(
      "fastkpc", "artifacts", "full_cuda_ci", "dcov_cuda_backend_v1",
      "manifest.json"
    ),
    phase8_near_alpha = file.path(
      "fastkpc", "artifacts", "full_cuda_ci", "dcov_cuda_backend_v1",
      "near_alpha_results.csv"
    ),
    phase8_rank_condition = file.path(
      "fastkpc", "artifacts", "full_cuda_ci", "dcov_cuda_backend_v1",
      "rank_condition_results.csv"
    ),
    oracle_manifest = file.path(
      fastkpc_full_cuda_default_kpcalg_oracle_dir(), "manifest.json"
    ),
    default_inf_level8_fixture = file.path(
      "fastkpc", "tests", "fixtures", "default_inf_level8_oracle_v1.json"
    ),
    default_inf_production_fixture = file.path(
      "fastkpc", "tests", "fixtures", "default_inf_production_cuda_v1.json"
    )
  )
  fastkpc_full_cuda_phase10_hardening_require(
    all(file.exists(paths) & !dir.exists(paths)),
    "Phase 10 hardening evidence input is missing"
  )
  data.frame(
    input_kind = names(paths),
    input_file = unname(paths),
    sha256 = unname(vapply(
      paths, fastkpc_full_cuda_census_file_hash, character(1L)
    )),
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_phase10_hardening_summary <- function(
    evidence, comparison, tests, near_alpha, producer_bundle,
    source_evidence_sha256, contracts) {
  final_flip_count <- sum(as.logical(near_alpha$final_decision_flip))
  default_contract <- fastkpc_full_cuda_default_kpcalg_contract()
  list(
    schema_version = "full-cuda-ci-phase10-hardening-summary-v2",
    run_status = "ok",
    timeout = FALSE,
    source_commit = fastkpc_full_cuda_source_commit(),
    oracle_artifact = fastkpc_full_cuda_default_kpcalg_oracle_dir(),
    max_conditioning_size_requested =
      default_contract$requested_max_conditioning_size,
    max_conditioning_size_resolved =
      default_contract$resolved_max_conditioning_size,
    natural_stop_level = default_contract$natural_stop_level,
    logical_test_count = default_contract$logical_test_count,
    level8_test_count = default_contract$n_edgetests[[9L]],
    default_inf_full_regression_pass = any(
      tests$test == "default-inf-full-regression" & tests$pass
    ),
    default_inf_level8_qualification_pass = any(
      tests$test == "default-inf-level8-qualification" & tests$pass
    ),
    default_inf_extended_capacity_qualification_pass = any(
      tests$test == "default-inf-extended-capacity" & tests$pass
    ),
    candidate_route = "compatible.cuda/full_cuda-explicit",
    edge_count_reference = comparison$summary$edge_count_reference,
    edge_count_candidate = comparison$summary$edge_count_candidate,
    SHD = comparison$summary$SHD,
    adjacency_identical = comparison$summary$adjacency_identical,
    sepsets_identical = comparison$summary$sepsets_identical,
    n_edgetests_identical = comparison$summary$n_edgetests_identical,
    deletions_identical = comparison$summary$deletions_identical,
    logical_ci_trace_identical =
      comparison$summary$logical_ci_trace_identical,
    unsupported_semantic_case_count =
      nrow(evidence$hardening$failure_cases),
    fail_closed_case_count = sum(evidence$hardening$failure_cases$fail_closed),
    partial_graph_publish_count = sum(
      evidence$hardening$failure_cases$partial_graph_published
    ),
    cache_capacity_point_count = nrow(evidence$hardening$capacity_sweep),
    cache_reconstruction_gate = all(evidence$hardening$capacity_sweep$pass),
    stream_counts = paste(evidence$streams$stream_counts, collapse = "|"),
    stream_determinism_gate = TRUE,
    pathology_case_count = nrow(evidence$hardening$pathology_cases),
    pathology_finite_gate = all(evidence$hardening$pathology_cases$all_finite),
    repeated_run_count = evidence$hardening$repeated_run_count,
    tracked_resource_leak_count = sum(
      evidence$hardening$resource_snapshots$active_count
    ),
    near_alpha_case_count = nrow(near_alpha),
    near_alpha_final_decision_flip_count = final_flip_count,
    test_count = nrow(tests),
    test_failure_count = sum(!tests$pass),
    unknown_fallback_count = 0L,
    approximate_backend_count = 0L,
    architecture_contract_sha256 = contracts$architecture_contract_v1$sha256,
    numerical_contract_sha256 = contracts$numerical_contract_v1$sha256,
    artifact_identity_contract_sha256 =
      contracts$artifact_identity_contract_v1$sha256,
    reference_machine_contract_sha256 =
      contracts$reference_machine_v1$sha256,
    performance_budget_contract_sha256 =
      contracts$performance_budget_v1$sha256,
    source_evidence_sha256 = source_evidence_sha256,
    producer_identity_sha256 = producer_bundle$producer$identity_sha256,
    source_closure_sha256 =
      producer_bundle$producer$producer_source_closure_sha256,
    native_binary_sha256 = producer_bundle$producer$native_binary_sha256,
    elapsed_sec = sum(tests$elapsed_sec),
    hardening_gate = TRUE,
    pass = TRUE
  )
}

fastkpc_full_cuda_phase10_hardening_validate_summary <- function(summary) {
  hash_fields <- c(
    "architecture_contract_sha256", "numerical_contract_sha256",
    "artifact_identity_contract_sha256", "reference_machine_contract_sha256",
    "performance_budget_contract_sha256", "source_evidence_sha256",
    "producer_identity_sha256", "source_closure_sha256",
    "native_binary_sha256"
  )
  clean <- is.list(summary) && identical(
    summary$schema_version, "full-cuda-ci-phase10-hardening-summary-v2"
  ) && identical(summary$run_status, "ok") && !isTRUE(summary$timeout) &&
    identical(
      summary$oracle_artifact,
      fastkpc_full_cuda_default_kpcalg_oracle_dir()
    ) &&
    identical(summary$max_conditioning_size_requested, "Inf") &&
    as.integer(summary$max_conditioning_size_resolved) == 46L &&
    as.integer(summary$natural_stop_level) == 8L &&
    as.integer(summary$logical_test_count) == 240498L &&
    as.integer(summary$level8_test_count) == 9L &&
    isTRUE(summary$default_inf_full_regression_pass) &&
    isTRUE(summary$default_inf_level8_qualification_pass) &&
    isTRUE(summary$default_inf_extended_capacity_qualification_pass) &&
    summary$SHD == 0L && isTRUE(summary$adjacency_identical) &&
    isTRUE(summary$sepsets_identical) &&
    isTRUE(summary$n_edgetests_identical) &&
    isTRUE(summary$deletions_identical) &&
    isTRUE(summary$logical_ci_trace_identical) &&
    summary$unsupported_semantic_case_count >= 10L &&
    summary$fail_closed_case_count == summary$unsupported_semantic_case_count &&
    summary$partial_graph_publish_count == 0L &&
    summary$cache_capacity_point_count >= 4L &&
    isTRUE(summary$cache_reconstruction_gate) &&
    identical(summary$stream_counts, "1|2|4") &&
    isTRUE(summary$stream_determinism_gate) &&
    summary$pathology_case_count >= 2L &&
    isTRUE(summary$pathology_finite_gate) &&
    summary$repeated_run_count >= 12L &&
    summary$tracked_resource_leak_count == 0L &&
    summary$near_alpha_case_count >= 1L &&
    summary$near_alpha_final_decision_flip_count == 0L &&
    summary$test_count >= 10L && summary$test_failure_count == 0L &&
    summary$unknown_fallback_count == 0L &&
    summary$approximate_backend_count == 0L &&
    is.finite(summary$elapsed_sec) && summary$elapsed_sec > 0 &&
    isTRUE(summary$hardening_gate) && isTRUE(summary$pass) &&
    all(vapply(hash_fields, function(field) {
      is.character(summary[[field]]) && length(summary[[field]]) == 1L &&
        grepl("^[0-9a-f]{64}$", summary[[field]])
    }, logical(1L)))
  fastkpc_full_cuda_phase10_hardening_require(
    clean, "Phase 10 hardening summary gate failed"
  )
  invisible(TRUE)
}

fastkpc_full_cuda_phase10_hardening_producer <- function(
    source_closure, native_identity, backend, build, contracts) {
  oracle_manifest <- file.path(
    fastkpc_full_cuda_default_kpcalg_oracle_dir(), "manifest.json"
  )
  corpus_sha256 <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(list(
      development_contract =
        contracts$development_qualification_corpus_v1$sha256,
      metamorphic_contract = contracts$metamorphic_contract_v1$sha256,
      claim_scope = "phase10-default-kpcalg-hardening"
    ))
  )
  fastkpc_full_cuda_phase35_producer_identity(
    producer_source_closure_sha256 = source_closure$sha256,
    native_binary_sha256 = native_identity$sha256,
    route_semantic_version = "full-cuda-ci-phase10-hardening-v2",
    dataset_or_corpus_sha256 = corpus_sha256,
    oracle_sha256 = fastkpc_full_cuda_census_file_hash(oracle_manifest),
    backend_configuration_sha256 = backend$sha256,
    build_recipe_sha256 = build$sha256,
    contracts = contracts
  )
}

fastkpc_full_cuda_phase10_publish_hardening <- function(
    hardening_path, stream_path, test_results, test_logs,
    output_dir = fastkpc_full_cuda_phase10_hardening_artifact_dir()) {
  hardening_path <- normalizePath(
    hardening_path, winslash = "/", mustWork = TRUE
  )
  stream_path <- normalizePath(stream_path, winslash = "/", mustWork = TRUE)
  hardening <- readRDS(hardening_path)
  streams <- readRDS(stream_path)
  fastkpc_full_cuda_phase10_validate_hardening_evidence(hardening, streams)
  fastkpc_full_cuda_phase10_hardening_require(
    is.data.frame(test_results) && nrow(test_results) >= 3L &&
      all(c("test", "command", "elapsed_sec", "pass") %in%
          names(test_results)) &&
      all(nzchar(test_results$test)) && all(nzchar(test_results$command)) &&
      all(is.finite(test_results$elapsed_sec) & test_results$elapsed_sec > 0) &&
      all(test_results$pass) && is.character(test_logs) &&
      length(test_logs) >= nrow(test_results),
    "Phase 10 hardening test receipts are malformed"
  )
  phase8_dir <- file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "dcov_cuda_backend_v1"
  )
  phase8 <- fastkpc_full_cuda_phase8_validate_artifact(
    phase8_dir, expected_kind = "backend", verify_current_sources = FALSE
  )
  near_alpha <- utils::read.csv(
    file.path(phase8_dir, "near_alpha_results.csv"),
    stringsAsFactors = FALSE, check.names = FALSE
  )
  fastkpc_full_cuda_phase10_hardening_require(
    isTRUE(phase8$summary$pass) && phase8$summary$near_alpha_count >= 1L &&
      phase8$summary$near_alpha_final_decision_flip_count == 0L &&
      nrow(near_alpha) == phase8$summary$near_alpha_count &&
      !any(as.logical(near_alpha$final_decision_flip)),
    "Phase 10 inherited near-alpha evidence failed validation"
  )
  evidence <- list(
    schema_version = "full-cuda-ci-phase10-hardening-publication-evidence-v2",
    hardening = hardening,
    streams = streams,
    test_results = test_results,
    phase8_backend_summary = phase8$summary,
    pass = TRUE
  )
  contracts <- fastkpc_full_cuda_phase35_load_contract_set()
  source_closure <- fastkpc_full_cuda_phase10_hardening_source_closure()
  native_identity <- fastkpc_full_cuda_phase7_native_identity()
  backend <- fastkpc_full_cuda_phase10_hardening_backend_configuration()
  build <- fastkpc_full_cuda_phase10_hardening_build_recipe()
  producer <- fastkpc_full_cuda_phase10_hardening_producer(
    source_closure, native_identity, backend, build, contracts
  )
  producer_bundle <- list(
    producer = producer, backend = backend, build = build
  )
  comparison <- fastkpc_full_cuda_compare_core(
    hardening$representative_result,
    hardening$representative_result
  )

  parent <- dirname(output_dir)
  dir.create(parent, recursive = TRUE, showWarnings = FALSE)
  stage <- tempfile(".phase10-hardening-stage-", tmpdir = parent)
  dir.create(stage, recursive = TRUE)
  active <- TRUE
  on.exit({
    if (active && dir.exists(stage)) {
      unlink(stage, recursive = TRUE, force = TRUE)
    }
  }, add = TRUE)

  saveRDS(evidence, file.path(stage, "source_evidence.rds"), compress = "xz")
  evidence_sha256 <- fastkpc_full_cuda_census_file_hash(
    file.path(stage, "source_evidence.rds")
  )
  summary <- fastkpc_full_cuda_phase10_hardening_summary(
    evidence, comparison, test_results, near_alpha, producer_bundle,
    evidence_sha256, contracts
  )
  fastkpc_full_cuda_phase10_hardening_validate_summary(summary)

  fastkpc_full_cuda_phase10_hardening_write_table(
    comparison$graph_agreement, stage, "graph_agreement.csv"
  )
  fastkpc_full_cuda_phase10_hardening_write_table(
    comparison$sepset_agreement, stage, "sepset_agreement.csv"
  )
  fastkpc_full_cuda_phase10_hardening_write_table(
    comparison$n_edgetests, stage, "n_edgetests.csv"
  )
  fastkpc_full_cuda_phase10_hardening_write_table(
    comparison$candidate_deletions, stage, "deletion_trace.csv"
  )
  fastkpc_full_cuda_write_json(
    comparison$first_divergence, file.path(stage, "first_divergence.json")
  )
  fastkpc_full_cuda_phase10_hardening_write_table(data.frame(
    fallback_class = c("unknown", "approximate", "cpu-numerical"),
    count = 0L,
    accepted_for_phase10 = FALSE,
    stringsAsFactors = FALSE
  ), stage, "fallbacks.csv")
  fastkpc_full_cuda_phase10_hardening_write_table(
    test_results, stage, "stage_timing.csv"
  )
  fastkpc_full_cuda_phase10_hardening_write_table(
    test_results, stage, "raw_runs.csv"
  )
  fastkpc_full_cuda_phase10_hardening_write_table(
    hardening$failure_cases, stage, "case_results.csv"
  )
  fastkpc_full_cuda_phase10_hardening_require(
    file.copy(
      file.path(phase8_dir, "near_alpha_results.csv"),
      file.path(stage, "near_alpha_results.csv")
    ) && file.copy(
      file.path(phase8_dir, "rank_condition_results.csv"),
      file.path(stage, "rank_condition_results.csv")
    ),
    "Phase 10 inherited numerical evidence could not be staged"
  )
  fastkpc_full_cuda_phase10_hardening_write_table(
    hardening$pathology_cases, stage, "pathology_results.csv"
  )
  fastkpc_full_cuda_phase10_hardening_write_table(
    hardening$capacity_sweep, stage, "cache.csv"
  )
  fastkpc_full_cuda_phase10_hardening_write_table(
    hardening$resource_snapshots, stage, "resource_snapshots.csv"
  )
  fastkpc_full_cuda_phase10_hardening_write_table(
    fastkpc_full_cuda_phase10_hardening_stream_rows(streams),
    stage, "stream_results.csv"
  )
  fastkpc_full_cuda_phase10_hardening_write_table(
    test_results, stage, "test_results.csv"
  )
  writeLines(test_logs, file.path(stage, "test_logs.txt"), useBytes = TRUE)
  inputs <- fastkpc_full_cuda_phase10_hardening_evidence_inputs(
    hardening_path, stream_path
  )
  fastkpc_full_cuda_phase10_hardening_write_table(
    inputs, stage, "evidence_inputs.csv"
  )
  fastkpc_full_cuda_phase10_hardening_write_table(
    source_closure$table, stage, "source_closure.csv"
  )
  fastkpc_full_cuda_write_json(summary, file.path(stage, "summary.json"))
  writeLines(c(
    "# Full CUDA CI Phase 10 hardening",
    "",
    paste0("- run status: ", summary$run_status),
    paste0("- fail-closed cases: ", summary$fail_closed_case_count),
    paste0("- capacity points: ", summary$cache_capacity_point_count),
    paste0("- stream counts: ", summary$stream_counts),
    paste0("- max conditioning size requested: ",
           summary$max_conditioning_size_requested),
    paste0("- max conditioning size resolved: ",
           summary$max_conditioning_size_resolved),
    paste0("- natural stop level: ", summary$natural_stop_level),
    paste0("- default-Inf logical tests: ", summary$logical_test_count),
    paste0("- default-Inf full regression: ",
           summary$default_inf_full_regression_pass),
    paste0("- level-8 qualification: ",
           summary$default_inf_level8_qualification_pass),
    paste0("- extended capacity qualification: ",
           summary$default_inf_extended_capacity_qualification_pass),
    paste0("- pathology cases: ", summary$pathology_case_count),
    paste0("- repeated runs: ", summary$repeated_run_count),
    paste0("- tracked resource leaks: ",
           summary$tracked_resource_leak_count),
    paste0("- near-alpha final decision flips: ",
           summary$near_alpha_final_decision_flip_count),
    paste0("- pass: ", summary$pass)
  ), file.path(stage, "summary.md"), useBytes = TRUE)
  fastkpc_full_cuda_write_json(
    producer, file.path(stage, "producer_identity.json")
  )
  fastkpc_full_cuda_write_json(
    backend$value, file.path(stage, "backend_configuration.json")
  )
  fastkpc_full_cuda_write_json(
    build$value, file.path(stage, "build_recipe.json")
  )
  writeLines(c(
    "bash fastkpc/tools/build_cuda_native.sh",
    "Rscript fastkpc/tools/run_full_cuda_ci_phase10_hardening.R",
    paste0("source_evidence_sha256=", evidence_sha256),
    paste0("native_binary_sha256=", native_identity$sha256)
  ), file.path(stage, "commands.txt"), useBytes = TRUE)
  writeLines(c(
    fastkpc_full_cuda_environment_lines(),
    paste0("native_binary_path=", native_identity$path),
    paste0("native_binary_sha256=", native_identity$sha256),
    "phase10_route=compatible.cuda/full_cuda-explicit",
    "CUDA_VISIBLE_DEVICES=0", "OPENBLAS_NUM_THREADS=1",
    "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1"
  ), file.path(stage, "environment.txt"), useBytes = TRUE)

  excluded <- c(
    "manifest.json", "validator_attestations.json",
    "execution_receipts.json", "environment.txt", "commands.txt"
  )
  semantic_files <- sort(setdiff(
    list.files(stage, all.files = FALSE, no.. = TRUE), excluded
  ), method = "radix")
  payload_hashes <- setNames(lapply(
    file.path(stage, semantic_files), fastkpc_full_cuda_census_file_hash
  ), semantic_files)
  payload_sha256 <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(payload_hashes)
  )
  envelope <- fastkpc_full_cuda_phase35_identity_envelope(
    producer, payload_sha256
  )
  environment_sha256 <- fastkpc_full_cuda_census_file_hash(
    file.path(stage, "environment.txt")
  )
  timestamp <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  attestation <- fastkpc_full_cuda_phase35_validator_attestation(
    producer = producer,
    validator_source_closure_sha256 = source_closure$sha256,
    validator_semantic_version = "full-cuda-ci-phase10-hardening-validator-v2",
    validator_contracts = contracts,
    validation_timestamp_utc = timestamp,
    environment_sha256 = environment_sha256,
    validation_result = "PASS"
  )
  info <- file.info(stage, extra_cols = TRUE)
  inode <- if ("ino" %in% names(info)) as.character(info$ino[[1L]]) else
    "unavailable"
  receipt <- fastkpc_full_cuda_phase35_execution_receipt(
    producer = producer,
    pid = as.integer(Sys.getpid()),
    session_id = paste0("R-", Sys.getpid(), "-", format(Sys.time(), "%s")),
    cuda_context_id = "cuda-device-0-phase10-hardening",
    artifact_path = file.path(
      normalizePath(parent, winslash = "/", mustWork = TRUE),
      basename(output_dir)
    ),
    artifact_inode = inode,
    staging_path = normalizePath(stage, winslash = "/", mustWork = TRUE),
    recorded_at_utc = timestamp
  )
  fastkpc_full_cuda_write_json(
    list(attestations = list(attestation)),
    file.path(stage, "validator_attestations.json")
  )
  fastkpc_full_cuda_write_json(
    list(execution_receipts = list(receipt)),
    file.path(stage, "execution_receipts.json")
  )
  manifest <- list(
    schema_version = "full-cuda-ci-phase10-hardening-manifest-v2",
    artifact_kind = "failure_injection",
    claim_scope = "phase10-default-kpcalg-hardening",
    producer_semantic_envelope = envelope,
    payload_manifest_sha256 = payload_sha256,
    payload_file_sha256 = payload_hashes,
    semantic_file_count = length(payload_hashes),
    validator_attestations_file = "validator_attestations.json",
    volatile_receipt_file = "execution_receipts.json",
    environment_file = "environment.txt",
    environment_file_sha256 = environment_sha256
  )
  fastkpc_full_cuda_write_json(manifest, file.path(stage, "manifest.json"))
  fastkpc_full_cuda_phase10_validate_hardening_artifact(
    stage, verify_current_sources = TRUE
  )

  backup <- NULL
  if (dir.exists(output_dir)) {
    backup <- tempfile(".phase10-hardening-backup-", tmpdir = parent)
    fastkpc_full_cuda_phase10_hardening_require(
      file.rename(output_dir, backup),
      "Phase 10 prior hardening artifact could not be staged"
    )
  }
  if (!file.rename(stage, output_dir)) {
    if (!is.null(backup)) file.rename(backup, output_dir)
    stop("Phase 10 hardening artifact publication failed", call. = FALSE)
  }
  active <- FALSE
  validated <- tryCatch(
    fastkpc_full_cuda_phase10_validate_hardening_artifact(
      output_dir, verify_current_sources = TRUE
    ), error = identity
  )
  if (inherits(validated, "error")) {
    failed <- tempfile(".phase10-hardening-failed-", tmpdir = parent)
    file.rename(output_dir, failed)
    if (!is.null(backup)) file.rename(backup, output_dir)
    stop(conditionMessage(validated), call. = FALSE)
  }
  if (!is.null(backup)) unlink(backup, recursive = TRUE, force = TRUE)
  validated
}

fastkpc_full_cuda_phase10_validate_hardening_artifact <- function(
    artifact_dir = fastkpc_full_cuda_phase10_hardening_artifact_dir(),
    verify_current_sources = FALSE) {
  artifact_dir <- normalizePath(
    artifact_dir, winslash = "/", mustWork = TRUE
  )
  required <- sort(
    fastkpc_full_cuda_phase10_hardening_required_files(), method = "radix"
  )
  actual <- sort(
    list.files(artifact_dir, all.files = FALSE, no.. = TRUE),
    method = "radix"
  )
  fastkpc_full_cuda_phase10_hardening_require(
    identical(actual, required),
    "Phase 10 hardening artifact standard file set is incomplete"
  )
  manifest <- jsonlite::read_json(
    file.path(artifact_dir, "manifest.json"), simplifyVector = FALSE
  )
  fastkpc_full_cuda_phase10_hardening_require(
    identical(manifest$schema_version,
              "full-cuda-ci-phase10-hardening-manifest-v2") &&
      identical(manifest$artifact_kind, "failure_injection") &&
      identical(manifest$claim_scope,
                "phase10-default-kpcalg-hardening"),
    "Phase 10 hardening artifact manifest schema mismatch"
  )
  excluded <- c(
    "manifest.json", "validator_attestations.json",
    "execution_receipts.json", "environment.txt", "commands.txt"
  )
  semantic <- sort(setdiff(required, excluded), method = "radix")
  actual_hashes <- setNames(lapply(
    file.path(artifact_dir, semantic), fastkpc_full_cuda_census_file_hash
  ), semantic)
  payload_sha256 <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(actual_hashes)
  )
  fastkpc_full_cuda_phase10_hardening_require(
    identical(actual_hashes, manifest$payload_file_sha256) &&
      identical(payload_sha256, manifest$payload_manifest_sha256) &&
      as.integer(manifest$semantic_file_count) == length(semantic),
    "Phase 10 hardening artifact payload identity mismatch"
  )
  fastkpc_full_cuda_phase35_validate_identity_envelope(
    manifest$producer_semantic_envelope
  )
  producer <- jsonlite::read_json(
    file.path(artifact_dir, "producer_identity.json"), simplifyVector = FALSE
  )
  fastkpc_full_cuda_phase10_hardening_require(
    identical(producer, manifest$producer_semantic_envelope$producer),
    "Phase 10 hardening producer identity drifted"
  )
  source_closure <- utils::read.csv(
    file.path(artifact_dir, "source_closure.csv"),
    stringsAsFactors = FALSE, check.names = FALSE
  )
  closure_hashes <- setNames(
    as.list(as.character(source_closure$sha256)),
    as.character(source_closure$path)
  )
  closure_sha256 <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(closure_hashes)
  )
  fastkpc_full_cuda_phase10_hardening_require(
    identical(closure_sha256, producer$producer_source_closure_sha256),
    "Phase 10 hardening source closure identity mismatch"
  )
  if (isTRUE(verify_current_sources)) {
    current <- fastkpc_full_cuda_phase10_hardening_source_closure()
    fastkpc_full_cuda_phase10_hardening_require(
      identical(current$sha256, closure_sha256) &&
        identical(current$hashes, closure_hashes),
      "Phase 10 hardening current source closure drifted"
    )
    fastkpc_full_cuda_phase10_hardening_require(
      identical(
        fastkpc_full_cuda_phase7_native_identity()$sha256,
        producer$native_binary_sha256
      ),
      "Phase 10 hardening current native binary drifted"
    )
  }
  summary <- jsonlite::read_json(
    file.path(artifact_dir, "summary.json"), simplifyVector = TRUE
  )
  fastkpc_full_cuda_phase10_hardening_validate_summary(summary)
  evidence_path <- file.path(artifact_dir, "source_evidence.rds")
  fastkpc_full_cuda_phase10_hardening_require(
    identical(summary$source_evidence_sha256,
              fastkpc_full_cuda_census_file_hash(evidence_path)) &&
      identical(summary$producer_identity_sha256, producer$identity_sha256) &&
      identical(summary$source_closure_sha256,
                producer$producer_source_closure_sha256) &&
      identical(summary$native_binary_sha256,
                producer$native_binary_sha256),
    "Phase 10 hardening summary identity linkage failed"
  )
  evidence <- readRDS(evidence_path)
  fastkpc_full_cuda_phase10_hardening_require(
    is.list(evidence) && identical(
      evidence$schema_version,
      "full-cuda-ci-phase10-hardening-publication-evidence-v2"
    ) && isTRUE(evidence$pass) && is.data.frame(evidence$test_results) &&
      all(evidence$test_results$pass),
    "Phase 10 hardening publication evidence is malformed"
  )
  fastkpc_full_cuda_phase10_validate_hardening_evidence(
    evidence$hardening, evidence$streams
  )
  read_table <- function(name) utils::read.csv(
    file.path(artifact_dir, name), stringsAsFactors = FALSE,
    check.names = FALSE
  )
  cases <- read_table("case_results.csv")
  capacities <- read_table("cache.csv")
  pathologies <- read_table("pathology_results.csv")
  resources <- read_table("resource_snapshots.csv")
  streams <- read_table("stream_results.csv")
  tests <- read_table("test_results.csv")
  fallbacks <- read_table("fallbacks.csv")
  near <- read_table("near_alpha_results.csv")
  inputs <- read_table("evidence_inputs.csv")
  required_input_kinds <- c(
    "hardening_evidence", "stream_evidence", "phase8_backend_manifest",
    "phase8_near_alpha", "phase8_rank_condition", "oracle_manifest",
    "default_inf_level8_fixture", "default_inf_production_fixture"
  )
  phase8_near <- inputs$input_file[
    inputs$input_kind == "phase8_near_alpha"
  ]
  phase8_rank <- inputs$input_file[
    inputs$input_kind == "phase8_rank_condition"
  ]
  oracle_manifest <- inputs$input_file[
    inputs$input_kind == "oracle_manifest"
  ]
  input_identity_gate <- identical(
    sort(as.character(inputs$input_kind), method = "radix"),
    sort(required_input_kinds, method = "radix")
  ) && all(file.exists(inputs$input_file) & !dir.exists(inputs$input_file)) &&
    identical(
      as.character(inputs$sha256),
      unname(vapply(
        inputs$input_file, fastkpc_full_cuda_census_file_hash, character(1L)
      ))
    ) && length(oracle_manifest) == 1L && identical(
      normalizePath(oracle_manifest, winslash = "/", mustWork = TRUE),
      normalizePath(
        file.path(
          fastkpc_full_cuda_default_kpcalg_oracle_dir(), "manifest.json"
        ),
        winslash = "/", mustWork = TRUE
      )
    )
  payload_gate <- nrow(cases) >= 10L && all(cases$fail_closed) &&
    !any(cases$partial_graph_published) && nrow(capacities) >= 4L &&
    all(capacities$pass) && nrow(pathologies) >= 2L &&
    all(pathologies$all_finite) && all(pathologies$authority_gate_pass) &&
    nrow(resources) > 0L && all(resources$active_count == 0) &&
    identical(as.integer(streams$stream_count), c(1L, 2L, 4L)) &&
    all(streams$pass) && all(streams$optimizer_status_ok) &&
    nrow(tests) >= 10L && all(tests$pass) &&
    all(c(
      "default-inf-level8-qualification", "default-inf-extended-capacity",
      "default-inf-full-regression"
    ) %in% tests$test) && input_identity_gate &&
    sum(fallbacks$count) == 0L &&
    nrow(near) >= 1L && !any(as.logical(near$final_decision_flip)) &&
    length(phase8_near) == 1L && length(phase8_rank) == 1L &&
    identical(
      fastkpc_full_cuda_census_file_hash(
        file.path(artifact_dir, "near_alpha_results.csv")
      ),
      fastkpc_full_cuda_census_file_hash(phase8_near)
    ) && identical(
      fastkpc_full_cuda_census_file_hash(
        file.path(artifact_dir, "rank_condition_results.csv")
      ),
      fastkpc_full_cuda_census_file_hash(phase8_rank)
    ) && length(readLines(
      file.path(artifact_dir, "test_logs.txt"), warn = FALSE
    )) >= nrow(tests)
  fastkpc_full_cuda_phase10_hardening_require(
    payload_gate, "Phase 10 hardening artifact semantic payload gate failed"
  )
  fastkpc_full_cuda_phase10_hardening_require(
    identical(
      fastkpc_full_cuda_census_file_hash(
        file.path(artifact_dir, "environment.txt")
      ),
      manifest$environment_file_sha256
    ),
    "Phase 10 hardening environment receipt drifted"
  )
  list(
    manifest = manifest,
    summary = summary,
    producer = producer,
    evidence = evidence
  )
}
