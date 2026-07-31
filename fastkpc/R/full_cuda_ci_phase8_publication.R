fastkpc_full_cuda_phase8_publication_require <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
  invisible(TRUE)
}

fastkpc_full_cuda_phase8_artifact_kinds <- function() {
  c("component_oracle", "full_shadow", "backend")
}

fastkpc_full_cuda_phase8_artifact_directories <- function() {
  c(
    component_oracle = "dcov_cuda_component_oracle_v1",
    full_shadow = "dcov_cuda_full_shadow_v1",
    backend = "dcov_cuda_backend_v1"
  )
}

fastkpc_full_cuda_phase8_required_files <- function() {
  c(
    "manifest.json", "summary.json", "summary.md", "commands.txt",
    "environment.txt", "graph_agreement.csv", "sepset_agreement.csv",
    "n_edgetests.csv", "deletion_trace.csv", "first_divergence.json",
    "fallbacks.csv", "stage_timing.csv", "raw_runs.csv",
    "case_results.csv", "near_alpha_results.csv",
    "rank_condition_results.csv", "cache.csv", "pair_results.csv",
    "evidence_inputs.csv", "source_closure.csv", "source_evidence.rds",
    "producer_identity.json", "backend_configuration.json",
    "build_recipe.json", "validator_attestations.json",
    "execution_receipts.json"
  )
}

fastkpc_full_cuda_phase8_publication_source_paths <- function() {
  paths <- sort(unique(c(
    fastkpc_full_cuda_phase7_source_paths(),
    "fastkpc/R/full_cuda_ci_phase35_bakeoff.R",
    "fastkpc/R/full_cuda_ci_phase8_dcov.R",
    "fastkpc/R/full_cuda_ci_phase8_publication.R",
    "fastkpc/tools/run_full_cuda_ci_phase8_dcov.R",
    "fastkpc/tools/run_full_cuda_ci_phase8_artifacts.R"
  )), method = "radix")
  fastkpc_full_cuda_phase8_publication_require(
    all(file.exists(paths) & !dir.exists(paths)) && !anyDuplicated(paths),
    "Phase 8 publication source closure is incomplete"
  )
  paths
}

fastkpc_full_cuda_phase8_publication_source_closure <- function() {
  paths <- fastkpc_full_cuda_phase8_publication_source_paths()
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

fastkpc_full_cuda_phase8_backend_configuration <- function(kind) {
  kind <- match.arg(kind, fastkpc_full_cuda_phase8_artifact_kinds())
  value <- list(
    schema_version = "full-cuda-ci-phase8-backend-configuration-v1",
    artifact_kind = kind,
    selected_architecture =
      "exact-cuda-screen-plus-guarded-legacy-full-eig-cuda",
    exact_screen_role =
      "internal-decision-screen-outside-closed-refinement-guard",
    guard_lower_inclusive = "0.05",
    guard_upper_inclusive = "0.15",
    alpha = "0.1",
    index = 1L,
    num_col = 35L,
    direct_adapter = "distance-invariant-intercept-shift",
    conditional_setup_backend = "phase7-native-cpp",
    residual_backend = "phase3-fixed-sp-cuda-device-handle",
    component_capacity_direct = 48L,
    component_capacity_conditional = 47L,
    precision = "float64",
    fmad = FALSE,
    fast_math = FALSE,
    backend_authoritative = identical(kind, "backend"),
    shadow_comparison = !identical(kind, "backend"),
    cpu_numerical_dcov_authority = FALSE,
    large_payload_d2h = FALSE
  )
  list(
    value = value,
    sha256 = fastkpc_full_cuda_phase35_sha256_utf8(
      fastkpc_full_cuda_phase35_canonical_json(value)
    )
  )
}

fastkpc_full_cuda_phase8_build_recipe <- function() {
  value <- list(
    schema_version = "full-cuda-ci-phase8-build-recipe-v1",
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
    eigensolver = "cusolverDnDsyevd",
    gamma_tail = "cuda-device-double",
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

fastkpc_full_cuda_phase8_producer <- function(
    kind, evidence, source_closure, native_identity, contracts) {
  backend <- fastkpc_full_cuda_phase8_backend_configuration(kind)
  build <- fastkpc_full_cuda_phase8_build_recipe()
  corpus_sha256 <- fastkpc_full_cuda_census_named_metadata_hash(list(
    schema_version = "full-cuda-ci-phase8-canonical-corpus-v1",
    dataset_sha256 = evidence$execution_identity$dataset_sha256,
    logical_test_count = 240489L,
    direct_logical_test_count = 2213L,
    conditional_logical_test_count = 238276L,
    component_count = 110665L
  ))
  oracle_sha256 <- fastkpc_full_cuda_census_named_metadata_hash(list(
    schema_version = "full-cuda-ci-phase8-oracle-lineage-v1",
    phase7_evidence_sha256 = evidence$phase7_evidence_sha256,
    phase35_feasibility_manifest_sha256 =
      fastkpc_full_cuda_census_file_hash(file.path(
        "fastkpc", "artifacts", "full_cuda_ci",
        "phase35_feasibility_v1", "manifest.json"
      ))
  ))
  producer <- fastkpc_full_cuda_phase35_producer_identity(
    producer_source_closure_sha256 = source_closure$sha256,
    native_binary_sha256 = native_identity$sha256,
    route_semantic_version = paste0(
      "full-cuda-ci-phase8-guarded-dcov-", kind, "-v1"
    ),
    dataset_or_corpus_sha256 = corpus_sha256,
    oracle_sha256 = oracle_sha256,
    backend_configuration_sha256 = backend$sha256,
    build_recipe_sha256 = build$sha256,
    contracts = contracts
  )
  list(producer = producer, backend = backend, build = build)
}

fastkpc_full_cuda_phase8_artifact_summary <- function(
    kind, evidence, producer_bundle, evidence_sha256, contracts) {
  source <- evidence$summary
  list(
    schema_version = "full-cuda-ci-phase8-artifact-summary-v1",
    artifact_kind = kind,
    run_status = "ok",
    timeout = FALSE,
    source_commit = evidence$execution_identity$source_commit,
    oracle_artifact =
      "fastkpc/artifacts/full_cuda_ci/oracle_351x48_v1",
    candidate_route =
      "phase7-native-residual+guarded-exact-screen-full-eig-cuda",
    logical_test_count = source$logical_test_count,
    direct_logical_test_count = source$direct_logical_test_count,
    conditional_logical_test_count = source$conditional_logical_test_count,
    unique_residual_component_count =
      source$unique_residual_component_count,
    exact_screen_component_count = source$exact_screen_component_count,
    guarded_pair_count = source$guarded_pair_count,
    refined_component_count = source$refined_component_count,
    screen_decision_flip_count = source$screen_decision_flip_count,
    final_decision_flip_count = source$final_decision_flip_count,
    near_alpha_count = source$near_alpha_count,
    near_alpha_final_decision_flip_count =
      source$near_alpha_final_decision_flip_count,
    maximum_refined_p_value_absolute_error =
      source$maximum_refined_p_value_absolute_error,
    cuda_dcov_host_boundary_ms = source$cuda_dcov_host_boundary_ms,
    legacy_cpu_spectra_dcov_ms = source$legacy_cpu_spectra_dcov_ms,
    dcov_performance_ratio = source$dcov_performance_ratio,
    dcov_budget_ms = source$dcov_budget_ms,
    dcov_budget_pass = source$dcov_budget_pass,
    dcov_same_machine_speed_pass = source$dcov_same_machine_speed_pass,
    matrix_h2d_bytes = source$matrix_h2d_bytes,
    residual_d2h_bytes = source$residual_d2h_bytes,
    component_d2h_bytes = source$component_d2h_bytes,
    host_synchronization_count = source$host_synchronization_count,
    component_cache_request_count = source$component_cache_request_count,
    component_cache_hit_count = source$component_cache_hit_count,
    component_cache_miss_count = source$component_cache_miss_count,
    component_cache_eviction_count = source$component_cache_eviction_count,
    cpu_dcov_component_count = source$cpu_dcov_component_count,
    cpu_dcov_eigen_or_lowrank_count =
      source$cpu_dcov_eigen_or_lowrank_count,
    cpu_dcov_pair_statistic_count = source$cpu_dcov_pair_statistic_count,
    cpu_gamma_p_value_count = source$cpu_gamma_p_value_count,
    cpu_spectra_count = source$cpu_spectra_count,
    cuda_dcov_pair_count = source$cuda_dcov_pair_count,
    cuda_gamma_p_value_count = source$cuda_gamma_p_value_count,
    r_callback_count = 0L,
    legacy_mgcv_fit_count = 0L,
    legacy_mgcv_setup_count = 0L,
    cpu_residual_solve_count = 0L,
    unknown_fallback_count = source$unknown_fallback_count,
    approximate_backend_count = source$approximate_backend_count,
    edge_count_reference = source$edge_count_reference,
    edge_count_candidate = source$edge_count_candidate,
    SHD = source$SHD,
    adjacency_identical = source$adjacency_identical,
    sepsets_identical = source$sepsets_identical,
    n_edgetests_identical = source$n_edgetests_identical,
    deletions_identical = source$deletions_identical,
    backend_authoritative = identical(kind, "backend"),
    shadow_comparison = !identical(kind, "backend"),
    architecture_contract_sha256 =
      contracts$architecture_contract_v1$sha256,
    numerical_contract_sha256 = contracts$numerical_contract_v1$sha256,
    artifact_identity_contract_sha256 =
      contracts$artifact_identity_contract_v1$sha256,
    reference_machine_contract_sha256 =
      contracts$reference_machine_v1$sha256,
    performance_budget_contract_sha256 =
      contracts$performance_budget_v1$sha256,
    elapsed_sec = source$elapsed_sec,
    source_evidence_sha256 = evidence_sha256,
    execution_identity_sha256 = evidence$execution_identity_sha256,
    producer_identity_sha256 =
      producer_bundle$producer$identity_sha256,
    source_closure_sha256 =
      producer_bundle$producer$producer_source_closure_sha256,
    native_binary_sha256 =
      producer_bundle$producer$native_binary_sha256,
    pass = TRUE
  )
}

fastkpc_full_cuda_phase8_validate_summary <- function(summary, kind) {
  expected_hash_fields <- c(
    "source_evidence_sha256", "execution_identity_sha256",
    "producer_identity_sha256", "source_closure_sha256",
    "native_binary_sha256", "architecture_contract_sha256",
    "numerical_contract_sha256", "artifact_identity_contract_sha256",
    "reference_machine_contract_sha256",
    "performance_budget_contract_sha256"
  )
  clean <- is.list(summary) && identical(
    summary$schema_version, "full-cuda-ci-phase8-artifact-summary-v1"
  ) && identical(summary$artifact_kind, kind) &&
    identical(summary$run_status, "ok") && !isTRUE(summary$timeout) &&
    summary$logical_test_count == 240489L &&
    summary$direct_logical_test_count == 2213L &&
    summary$conditional_logical_test_count == 238276L &&
    summary$unique_residual_component_count == 110665L &&
    summary$exact_screen_component_count == 110665L &&
    summary$guarded_pair_count == 1188L &&
    summary$refined_component_count == 1559L &&
    summary$screen_decision_flip_count == 92L &&
    summary$final_decision_flip_count == 0L &&
    summary$near_alpha_count == 1529L &&
    summary$near_alpha_final_decision_flip_count == 0L &&
    summary$maximum_refined_p_value_absolute_error <= 1e-10 &&
    summary$cuda_dcov_host_boundary_ms > 0 &&
    summary$cuda_dcov_host_boundary_ms <= 47000 &&
    summary$legacy_cpu_spectra_dcov_ms > summary$cuda_dcov_host_boundary_ms &&
    summary$dcov_performance_ratio < 1 &&
    summary$dcov_budget_ms == 47000 &&
    isTRUE(summary$dcov_budget_pass) &&
    isTRUE(summary$dcov_same_machine_speed_pass) &&
    summary$matrix_h2d_bytes > 0 &&
    summary$residual_d2h_bytes == 0 && summary$component_d2h_bytes == 0 &&
    summary$host_synchronization_count > 0L &&
    summary$component_cache_request_count ==
      summary$component_cache_hit_count +
        summary$component_cache_miss_count &&
    summary$component_cache_eviction_count == 0L &&
    summary$cpu_dcov_component_count == 0L &&
    summary$cpu_dcov_eigen_or_lowrank_count == 0L &&
    summary$cpu_dcov_pair_statistic_count == 0L &&
    summary$cpu_gamma_p_value_count == 0L && summary$cpu_spectra_count == 0L &&
    summary$cuda_dcov_pair_count == 241677L &&
    summary$cuda_gamma_p_value_count == 241677L &&
    summary$r_callback_count == 0L && summary$legacy_mgcv_fit_count == 0L &&
    summary$legacy_mgcv_setup_count == 0L &&
    summary$cpu_residual_solve_count == 0L &&
    summary$unknown_fallback_count == 0L &&
    summary$approximate_backend_count == 0L &&
    summary$edge_count_reference == 110L &&
    summary$edge_count_candidate == 110L && summary$SHD == 0L &&
    isTRUE(summary$adjacency_identical) &&
    isTRUE(summary$sepsets_identical) &&
    isTRUE(summary$n_edgetests_identical) &&
    isTRUE(summary$deletions_identical) &&
    identical(isTRUE(summary$backend_authoritative),
              identical(kind, "backend")) &&
    identical(isTRUE(summary$shadow_comparison),
              !identical(kind, "backend")) &&
    is.numeric(summary$elapsed_sec) && is.finite(summary$elapsed_sec) &&
    summary$elapsed_sec > 0 && isTRUE(summary$pass) &&
    all(vapply(expected_hash_fields, function(field) {
      is.character(summary[[field]]) && length(summary[[field]]) == 1L &&
        grepl("^[0-9a-f]{64}$", summary[[field]])
    }, logical(1L)))
  fastkpc_full_cuda_phase8_publication_require(
    clean, "Phase 8 artifact summary gate failed"
  )
  invisible(TRUE)
}

fastkpc_full_cuda_phase8_write_table <- function(value, directory, name) {
  utils::write.csv(
    value, file.path(directory, name), row.names = FALSE, na = ""
  )
  invisible(file.path(directory, name))
}

fastkpc_full_cuda_phase8_cache_table <- function(evidence) {
  d <- evidence$diagnostics
  exact_requests <- 2L * evidence$summary$logical_test_count
  exact_misses <- evidence$summary$exact_screen_component_count
  refine_requests <- 2L * evidence$summary$guarded_pair_count
  refine_misses <- evidence$summary$refined_component_count
  data.frame(
    cache = c("exact-centered-distance", "legacy-full-eig-component"),
    capacity = c(48L, 48L),
    requests = c(exact_requests, refine_requests),
    hits = c(exact_requests - exact_misses,
             refine_requests - refine_misses),
    misses = c(exact_misses, refine_misses),
    evictions = c(0L, 0L),
    component_build_cuda_ms = c(
      sum(d$screen_component_cuda_ms),
      sum(d$refinement_component_cuda_ms)
    ),
    semantic_eviction_effect = c("none", "none"),
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_phase8_evidence_inputs <- function(evidence) {
  paths <- c(
    phase7_native_setup_backend = file.path(
      "fastkpc", "artifacts", "full_cuda_ci", "native_setup_backend_v1",
      "source_evidence.rds"
    ),
    phase35_feasibility = file.path(
      "fastkpc", "artifacts", "full_cuda_ci", "phase35_feasibility_v1",
      "manifest.json"
    ),
    canonical_logical = file.path(
      "fastkpc", "artifacts", "full_cuda_ci",
      "workload_census_351x48_v1", "logical_ci_tests.rds"
    ),
    canonical_dataset = file.path(
      "fastkpc", "artifacts", "kpc_tprs_real_zhu",
      "cancer_RD-causalDiscoveryInput.rds"
    )
  )
  hashes <- vapply(paths, fastkpc_full_cuda_census_file_hash, character(1L))
  fastkpc_full_cuda_phase8_publication_require(
    identical(unname(hashes[["phase7_native_setup_backend"]]),
              evidence$phase7_evidence_sha256),
    "Phase 8 evidence input Phase 7 hash drifted"
  )
  data.frame(
    input_kind = names(paths),
    input_file = unname(paths),
    sha256 = unname(hashes),
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_phase8_write_payload <- function(
    directory, evidence, summary, evidence_sha256) {
  comparison <- evidence$graph$comparison
  fastkpc_full_cuda_phase8_write_table(
    comparison$graph_agreement, directory, "graph_agreement.csv"
  )
  fastkpc_full_cuda_phase8_write_table(
    comparison$sepset_agreement, directory, "sepset_agreement.csv"
  )
  fastkpc_full_cuda_phase8_write_table(
    comparison$n_edgetests, directory, "n_edgetests.csv"
  )
  fastkpc_full_cuda_phase8_write_table(
    comparison$candidate_deletions, directory, "deletion_trace.csv"
  )
  fastkpc_full_cuda_write_json(
    comparison$first_divergence,
    file.path(directory, "first_divergence.json")
  )
  fallbacks <- data.frame(
    fallback_class = c(
      "unknown", "approximate", "cpu-dcov-component", "cpu-dcov-eigen",
      "cpu-dcov-pair", "cpu-gamma-pvalue", "cpu-spectra"
    ),
    supported_scope = "none",
    count = 0L,
    accepted_for_phase8 = FALSE,
    stringsAsFactors = FALSE
  )
  fastkpc_full_cuda_phase8_write_table(
    fallbacks, directory, "fallbacks.csv"
  )
  fastkpc_full_cuda_phase8_write_table(
    evidence$diagnostics, directory, "stage_timing.csv"
  )
  raw_runs <- data.frame(
    run = 1:3,
    stage = c("phase8-full", "phase7-residual-authority",
              "phase35-architecture-selection"),
    route = c(
      "guarded-exact-screen-full-eig-cuda",
      "native-setup-cuda-residual", "guarded-architecture-go"
    ),
    evidence_file = c(
      "source_evidence.rds", "native_setup_backend_v1/source_evidence.rds",
      "phase35_feasibility_v1/manifest.json"
    ),
    sha256 = c(
      evidence_sha256, evidence$phase7_evidence_sha256,
      fastkpc_full_cuda_census_file_hash(file.path(
        "fastkpc", "artifacts", "full_cuda_ci",
        "phase35_feasibility_v1", "manifest.json"
      ))
    ),
    elapsed_sec = c(evidence$summary$elapsed_sec, NA_real_, NA_real_),
    pass = TRUE,
    stringsAsFactors = FALSE
  )
  fastkpc_full_cuda_phase8_write_table(
    raw_runs, directory, "raw_runs.csv"
  )
  fastkpc_full_cuda_phase8_write_table(
    evidence$pairs, directory, "case_results.csv"
  )
  fastkpc_full_cuda_phase8_write_table(
    evidence$pairs[evidence$pairs$near_alpha, , drop = FALSE],
    directory, "near_alpha_results.csv"
  )
  fastkpc_full_cuda_phase8_write_table(
    evidence$diagnostics, directory, "rank_condition_results.csv"
  )
  fastkpc_full_cuda_phase8_write_table(
    fastkpc_full_cuda_phase8_cache_table(evidence), directory, "cache.csv"
  )
  pair_results <- evidence$pairs[
    evidence$pairs$refined | evidence$pairs$screen_decision_flip,
    , drop = FALSE
  ]
  fastkpc_full_cuda_phase8_write_table(
    pair_results, directory, "pair_results.csv"
  )
  fastkpc_full_cuda_phase8_write_table(
    fastkpc_full_cuda_phase8_evidence_inputs(evidence),
    directory, "evidence_inputs.csv"
  )
  fastkpc_full_cuda_write_summary_md(
    summary, file.path(directory, "summary.md"),
    paste0("Full CUDA CI Phase 8 ", summary$artifact_kind)
  )
  invisible(directory)
}

fastkpc_full_cuda_phase8_validate_payload <- function(
    artifact_dir, evidence) {
  read_table <- function(name) utils::read.csv(
    file.path(artifact_dir, name), stringsAsFactors = FALSE,
    check.names = FALSE
  )
  comparison <- evidence$graph$comparison
  graph <- read_table("graph_agreement.csv")
  sepsets <- read_table("sepset_agreement.csv")
  n_edgetests <- read_table("n_edgetests.csv")
  deletions <- read_table("deletion_trace.csv")
  first <- jsonlite::read_json(
    file.path(artifact_dir, "first_divergence.json"), simplifyVector = TRUE
  )
  fallbacks <- read_table("fallbacks.csv")
  timing <- read_table("stage_timing.csv")
  raw_runs <- read_table("raw_runs.csv")
  cases <- read_table("case_results.csv")
  near_alpha <- read_table("near_alpha_results.csv")
  rank_condition <- read_table("rank_condition_results.csv")
  cache <- read_table("cache.csv")
  pairs <- read_table("pair_results.csv")
  inputs <- read_table("evidence_inputs.csv")
  expected_n_edgetests <- c(
    2213L, 52659L, 125293L, 40694L, 13293L, 5422L, 835L, 80L
  )
  graph_gate <- nrow(graph) == 1L &&
    graph$edge_count_reference[[1L]] == 110L &&
    graph$edge_count_candidate[[1L]] == 110L && graph$SHD[[1L]] == 0L &&
    isTRUE(as.logical(graph$adjacency_identical[[1L]])) &&
    nrow(sepsets) == nrow(comparison$sepset_agreement) &&
    all(as.logical(sepsets$identical)) && nrow(n_edgetests) == 8L &&
    identical(as.integer(n_edgetests$reference), expected_n_edgetests) &&
    identical(as.integer(n_edgetests$candidate), expected_n_edgetests) &&
    all(as.logical(n_edgetests$identical)) &&
    nrow(deletions) == nrow(comparison$candidate_deletions) &&
    !isTRUE(first$first_divergence_found) && nrow(fallbacks) == 7L &&
    sum(fallbacks$count) == 0L &&
    !any(as.logical(fallbacks$accepted_for_phase8))
  fastkpc_full_cuda_phase8_publication_require(
    graph_gate, "Phase 8 standard graph payload is malformed"
  )
  timing_gate <- nrow(timing) == 8635L &&
    sum(timing$pair_count) == 240489L &&
    sum(timing$screen_component_count) == 110665L &&
    sum(timing$refined_pair_count) == 1188L &&
    sum(timing$refined_component_count) == 1559L &&
    sum(timing$residual_d2h_bytes) == 0 &&
    sum(timing$component_d2h_bytes) == 0 &&
    sum(timing$cpu_dcov_component_count) == 0L &&
    sum(timing$cpu_dcov_eigen_or_lowrank_count) == 0L &&
    sum(timing$cpu_dcov_pair_statistic_count) == 0L &&
    sum(timing$cpu_gamma_p_value_count) == 0L &&
    sum(timing$cpu_spectra_count) == 0L &&
    sum(timing$solver_failure_count) == 0L &&
    all(as.logical(timing$leak_free))
  case_gate <- nrow(cases) == 240489L &&
    identical(as.integer(cases$logical_sequence_id), 1:240489) &&
    sum(as.logical(cases$refined)) == 1188L &&
    sum(as.logical(cases$screen_decision_flip)) == 92L &&
    !any(as.logical(cases$final_decision_flip)) &&
    all(!as.logical(cases$screen_decision_flip) |
          as.logical(cases$refined)) &&
    all(is.finite(cases$screen_p_value)) &&
    all(is.finite(cases$final_p_value)) &&
    nrow(near_alpha) == 1529L &&
    all(as.logical(near_alpha$near_alpha)) &&
    !any(as.logical(near_alpha$final_decision_flip)) &&
    nrow(pairs) == 1188L && all(as.logical(pairs$refined)) &&
    sum(as.logical(pairs$screen_decision_flip)) == 92L &&
    !any(as.logical(pairs$final_decision_flip))
  supplemental_gate <- nrow(rank_condition) == 8635L &&
    nrow(cache) == 2L &&
    sum(cache$requests) == evidence$summary$component_cache_request_count &&
    sum(cache$hits) == evidence$summary$component_cache_hit_count &&
    sum(cache$misses) == evidence$summary$component_cache_miss_count &&
    all(cache$hits + cache$misses == cache$requests) &&
    sum(cache$evictions) == 0L && nrow(raw_runs) == 3L &&
    all(as.logical(raw_runs$pass)) &&
    all(grepl("^[0-9a-f]{64}$", raw_runs$sha256)) &&
    nrow(inputs) == 4L &&
    all(grepl("^[0-9a-f]{64}$", inputs$sha256)) &&
    file.info(file.path(artifact_dir, "summary.md"))$size[[1L]] > 0L
  fastkpc_full_cuda_phase8_publication_require(
    timing_gate && case_gate && supplemental_gate,
    "Phase 8 standard numerical payload is malformed"
  )
  invisible(TRUE)
}

fastkpc_full_cuda_phase8_manifest_fields <- function() {
  c(
    "schema_version", "artifact_kind", "claim_scope",
    "producer_semantic_envelope", "payload_manifest_sha256",
    "payload_file_sha256", "semantic_file_count",
    "validator_attestations_file", "volatile_receipt_file",
    "environment_file", "environment_file_sha256"
  )
}

fastkpc_full_cuda_phase8_validate_artifact <- function(
    artifact_dir, expected_kind = NULL, verify_current_sources = FALSE) {
  artifact_dir <- normalizePath(
    artifact_dir, winslash = "/", mustWork = TRUE
  )
  manifest <- jsonlite::read_json(
    file.path(artifact_dir, "manifest.json"), simplifyVector = FALSE
  )
  fastkpc_full_cuda_phase8_publication_require(
    is.list(manifest) &&
      identical(names(manifest), fastkpc_full_cuda_phase8_manifest_fields()) &&
      identical(
        manifest$schema_version,
        "full-cuda-ci-phase8-dcov-artifact-manifest-v1"
      ) && identical(
        manifest$claim_scope,
        "phase8-guarded-device-resident-dcov-complete-canonical"
      ),
    "Phase 8 artifact manifest schema mismatch"
  )
  kind <- as.character(manifest$artifact_kind)
  fastkpc_full_cuda_phase8_publication_require(
    kind %in% fastkpc_full_cuda_phase8_artifact_kinds() &&
      (is.null(expected_kind) || identical(kind, expected_kind)),
    "Phase 8 artifact kind mismatch"
  )
  required <- fastkpc_full_cuda_phase8_required_files()
  actual_files <- sort(
    list.files(artifact_dir, all.files = FALSE, no.. = TRUE),
    method = "radix"
  )
  fastkpc_full_cuda_phase8_publication_require(
    identical(actual_files, sort(required, method = "radix")),
    "Phase 8 artifact standard file set is incomplete"
  )
  excluded <- c(
    "manifest.json", "validator_attestations.json",
    "execution_receipts.json", "environment.txt", "commands.txt"
  )
  expected_semantic <- sort(setdiff(required, excluded), method = "radix")
  payload_hashes <- manifest$payload_file_sha256
  fastkpc_full_cuda_phase8_publication_require(
    is.list(payload_hashes) &&
      identical(sort(names(payload_hashes), method = "radix"),
                expected_semantic),
    "Phase 8 payload manifest is malformed"
  )
  actual_hashes <- setNames(lapply(names(payload_hashes), function(name) {
    fastkpc_full_cuda_census_file_hash(file.path(artifact_dir, name))
  }), names(payload_hashes))
  fastkpc_full_cuda_phase8_publication_require(
    all(vapply(names(payload_hashes), function(name) {
      identical(payload_hashes[[name]], actual_hashes[[name]])
    }, logical(1L))),
    "Phase 8 payload file hash mismatch"
  )
  payload_sha256 <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(actual_hashes)
  )
  fastkpc_full_cuda_phase8_publication_require(
    identical(payload_sha256, manifest$payload_manifest_sha256) &&
      as.integer(manifest$semantic_file_count) == length(actual_hashes),
    "Phase 8 payload manifest identity mismatch"
  )
  envelope <- manifest$producer_semantic_envelope
  fastkpc_full_cuda_phase35_validate_identity_envelope(envelope)
  producer <- jsonlite::read_json(
    file.path(artifact_dir, "producer_identity.json"), simplifyVector = FALSE
  )
  .fastkpc_full_cuda_phase35_validate_producer(producer)
  fastkpc_full_cuda_phase8_publication_require(
    identical(envelope$payload_manifest_sha256, payload_sha256) &&
      identical(
        fastkpc_full_cuda_phase35_canonical_json(producer),
        fastkpc_full_cuda_phase35_canonical_json(envelope$producer)
      ),
    "Phase 8 producer envelope mismatch"
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
  fastkpc_full_cuda_phase8_publication_require(
    identical(closure_sha256, producer$producer_source_closure_sha256),
    "Phase 8 producer source closure mismatch"
  )
  if (isTRUE(verify_current_sources)) {
    current <- fastkpc_full_cuda_phase8_publication_source_closure()
    fastkpc_full_cuda_phase8_publication_require(
      identical(current$sha256, closure_sha256) &&
        identical(current$hashes, closure_hashes),
      "Phase 8 artifact source closure is stale"
    )
  }
  backend <- jsonlite::read_json(
    file.path(artifact_dir, "backend_configuration.json"),
    simplifyVector = FALSE
  )
  build <- jsonlite::read_json(
    file.path(artifact_dir, "build_recipe.json"), simplifyVector = FALSE
  )
  fastkpc_full_cuda_phase8_publication_require(
    identical(
      fastkpc_full_cuda_phase35_sha256_utf8(
        fastkpc_full_cuda_phase35_canonical_json(backend)
      ), producer$backend_configuration_sha256
    ) && identical(
      fastkpc_full_cuda_phase35_sha256_utf8(
        fastkpc_full_cuda_phase35_canonical_json(build)
      ), producer$build_recipe_sha256
    ),
    "Phase 8 backend/build identity mismatch"
  )
  environment_sha256 <- fastkpc_full_cuda_census_file_hash(
    file.path(artifact_dir, manifest$environment_file)
  )
  fastkpc_full_cuda_phase8_publication_require(
    identical(environment_sha256, manifest$environment_file_sha256),
    "Phase 8 environment evidence hash mismatch"
  )
  attestations <- jsonlite::read_json(
    file.path(artifact_dir, manifest$validator_attestations_file),
    simplifyVector = FALSE
  )$attestations
  receipts <- jsonlite::read_json(
    file.path(artifact_dir, manifest$volatile_receipt_file),
    simplifyVector = FALSE
  )$execution_receipts
  fastkpc_full_cuda_phase8_publication_require(
    is.list(attestations) && length(attestations) > 0L &&
      is.list(receipts) && length(receipts) > 0L,
    "Phase 8 attestation or receipt is missing"
  )
  for (attestation in attestations) {
    .fastkpc_full_cuda_phase35_validate_attestation(attestation)
    fastkpc_full_cuda_phase8_publication_require(
      identical(attestation$attested_producer_sha256,
                producer$identity_sha256) &&
        identical(attestation$environment_sha256, environment_sha256) &&
        identical(attestation$validation_result, "PASS"),
      "Phase 8 validator attestation mismatch"
    )
  }
  for (receipt in receipts) {
    .fastkpc_full_cuda_phase35_validate_receipt(receipt)
    fastkpc_full_cuda_phase8_publication_require(
      identical(receipt$producer_sha256, producer$identity_sha256),
      "Phase 8 execution receipt mismatch"
    )
  }
  summary <- jsonlite::read_json(
    file.path(artifact_dir, "summary.json"), simplifyVector = TRUE
  )
  fastkpc_full_cuda_phase8_validate_summary(summary, kind)
  fastkpc_full_cuda_phase8_publication_require(
    identical(summary$producer_identity_sha256, producer$identity_sha256) &&
      identical(summary$source_closure_sha256,
                producer$producer_source_closure_sha256) &&
      identical(summary$native_binary_sha256,
                producer$native_binary_sha256) &&
      identical(summary$source_evidence_sha256,
                actual_hashes$source_evidence.rds),
    "Phase 8 artifact summary identity mismatch"
  )
  evidence <- readRDS(file.path(artifact_dir, "source_evidence.rds"))
  fastkpc_full_cuda_phase8_validate_full(
    evidence, verify_current_identity = TRUE
  )
  fastkpc_full_cuda_phase8_publication_require(
    identical(summary$execution_identity_sha256,
              evidence$execution_identity_sha256),
    "Phase 8 execution identity mismatch"
  )
  fastkpc_full_cuda_phase8_validate_payload(artifact_dir, evidence)
  list(
    kind = kind, manifest = manifest, summary = summary,
    producer = producer, source_closure = source_closure,
    evidence = evidence
  )
}

fastkpc_full_cuda_phase8_publish_one <- function(
    kind, output_dir, evidence, evidence_path, evidence_sha256,
    contracts, source_closure, native_identity) {
  producer_bundle <- fastkpc_full_cuda_phase8_producer(
    kind, evidence, source_closure, native_identity, contracts
  )
  summary <- fastkpc_full_cuda_phase8_artifact_summary(
    kind, evidence, producer_bundle, evidence_sha256, contracts
  )
  fastkpc_full_cuda_phase8_validate_summary(summary, kind)
  output_parent <- dirname(output_dir)
  dir.create(output_parent, recursive = TRUE, showWarnings = FALSE)
  stage_dir <- tempfile(
    paste0(".phase8-", kind, "-stage-"), tmpdir = output_parent
  )
  dir.create(stage_dir, recursive = TRUE)
  stage_active <- TRUE
  on.exit({
    if (stage_active && dir.exists(stage_dir)) {
      unlink(stage_dir, recursive = TRUE, force = TRUE)
    }
  }, add = TRUE)
  fastkpc_full_cuda_phase8_publication_require(
    file.copy(
      evidence_path, file.path(stage_dir, "source_evidence.rds"),
      overwrite = TRUE, copy.mode = FALSE, copy.date = FALSE
    ),
    "Phase 8 source evidence copy failed"
  )
  fastkpc_full_cuda_phase8_write_payload(
    stage_dir, evidence, summary, evidence_sha256
  )
  fastkpc_full_cuda_phase8_write_table(
    source_closure$table, stage_dir, "source_closure.csv"
  )
  fastkpc_full_cuda_write_json(summary, file.path(stage_dir, "summary.json"))
  fastkpc_full_cuda_write_json(
    producer_bundle$producer,
    file.path(stage_dir, "producer_identity.json")
  )
  fastkpc_full_cuda_write_json(
    producer_bundle$backend$value,
    file.path(stage_dir, "backend_configuration.json")
  )
  fastkpc_full_cuda_write_json(
    producer_bundle$build$value, file.path(stage_dir, "build_recipe.json")
  )
  writeLines(c(
    paste0("source evidence SHA-256: ", evidence_sha256),
    paste0("producer source closure SHA-256: ", source_closure$sha256),
    paste0("native binary SHA-256: ", native_identity$sha256),
    "Rscript fastkpc/tools/run_full_cuda_ci_phase8_dcov.R",
    "Rscript fastkpc/tools/run_full_cuda_ci_phase8_artifacts.R"
  ), file.path(stage_dir, "commands.txt"), useBytes = TRUE)
  writeLines(c(
    fastkpc_full_cuda_environment_lines(),
    paste0("native_binary_path=", native_identity$path),
    paste0("native_binary_sha256=", native_identity$sha256),
    "phase8_dcov_execution=guarded-exact-screen-full-eig-cuda",
    "CUDA_VISIBLE_DEVICES=0", "OPENBLAS_NUM_THREADS=1",
    "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1"
  ), file.path(stage_dir, "environment.txt"), useBytes = TRUE)
  excluded <- c(
    "manifest.json", "validator_attestations.json",
    "execution_receipts.json", "environment.txt", "commands.txt"
  )
  semantic_files <- sort(setdiff(
    list.files(stage_dir, all.files = FALSE, no.. = TRUE), excluded
  ), method = "radix")
  payload_hashes <- setNames(lapply(
    file.path(stage_dir, semantic_files),
    fastkpc_full_cuda_census_file_hash
  ), semantic_files)
  payload_sha256 <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(payload_hashes)
  )
  envelope <- fastkpc_full_cuda_phase35_identity_envelope(
    producer_bundle$producer, payload_sha256
  )
  environment_sha256 <- fastkpc_full_cuda_census_file_hash(
    file.path(stage_dir, "environment.txt")
  )
  timestamp <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  attestation <- fastkpc_full_cuda_phase35_validator_attestation(
    producer = producer_bundle$producer,
    validator_source_closure_sha256 = source_closure$sha256,
    validator_semantic_version = "full-cuda-ci-phase8-validator-v1",
    validator_contracts = contracts,
    validation_timestamp_utc = timestamp,
    environment_sha256 = environment_sha256,
    validation_result = "PASS"
  )
  stage_info <- file.info(stage_dir, extra_cols = TRUE)
  stage_inode <- if ("ino" %in% names(stage_info)) {
    as.character(stage_info$ino[[1L]])
  } else "unavailable"
  final_path <- file.path(
    normalizePath(output_parent, winslash = "/", mustWork = TRUE),
    basename(output_dir)
  )
  receipt <- fastkpc_full_cuda_phase35_execution_receipt(
    producer = producer_bundle$producer,
    pid = as.integer(Sys.getpid()),
    session_id = paste0("R-", Sys.getpid(), "-", format(Sys.time(), "%s")),
    cuda_context_id = "cuda-device-0-phase8-dcov",
    artifact_path = final_path,
    artifact_inode = stage_inode,
    staging_path = normalizePath(stage_dir, winslash = "/", mustWork = TRUE),
    recorded_at_utc = timestamp
  )
  fastkpc_full_cuda_write_json(
    list(attestations = list(attestation)),
    file.path(stage_dir, "validator_attestations.json")
  )
  fastkpc_full_cuda_write_json(
    list(execution_receipts = list(receipt)),
    file.path(stage_dir, "execution_receipts.json")
  )
  manifest <- list(
    schema_version = "full-cuda-ci-phase8-dcov-artifact-manifest-v1",
    artifact_kind = kind,
    claim_scope =
      "phase8-guarded-device-resident-dcov-complete-canonical",
    producer_semantic_envelope = envelope,
    payload_manifest_sha256 = payload_sha256,
    payload_file_sha256 = payload_hashes,
    semantic_file_count = length(payload_hashes),
    validator_attestations_file = "validator_attestations.json",
    volatile_receipt_file = "execution_receipts.json",
    environment_file = "environment.txt",
    environment_file_sha256 = environment_sha256
  )
  fastkpc_full_cuda_write_json(manifest, file.path(stage_dir, "manifest.json"))
  fastkpc_full_cuda_phase8_validate_artifact(
    stage_dir, expected_kind = kind, verify_current_sources = TRUE
  )
  backup <- NULL
  if (dir.exists(output_dir)) {
    backup <- tempfile(
      paste0(".phase8-", kind, "-backup-"), tmpdir = output_parent
    )
    fastkpc_full_cuda_phase8_publication_require(
      file.rename(output_dir, backup),
      "Phase 8 prior artifact could not be staged"
    )
  }
  if (!file.rename(stage_dir, output_dir)) {
    if (!is.null(backup)) file.rename(backup, output_dir)
    stop("Phase 8 artifact publication failed", call. = FALSE)
  }
  stage_active <- FALSE
  validated <- tryCatch(
    fastkpc_full_cuda_phase8_validate_artifact(
      output_dir, expected_kind = kind, verify_current_sources = TRUE
    ), error = identity
  )
  if (inherits(validated, "error")) {
    failed <- tempfile(
      paste0(".phase8-", kind, "-failed-"), tmpdir = output_parent
    )
    file.rename(output_dir, failed)
    if (!is.null(backup)) file.rename(backup, output_dir)
    stop(conditionMessage(validated), call. = FALSE)
  }
  if (!is.null(backup)) unlink(backup, recursive = TRUE, force = TRUE)
  validated
}

fastkpc_full_cuda_phase8_publish_artifacts <- function(
    evidence_path,
    output_root = fastkpc_full_cuda_phase8_artifact_root()) {
  evidence_path <- normalizePath(
    evidence_path, winslash = "/", mustWork = TRUE
  )
  evidence <- readRDS(evidence_path)
  fastkpc_full_cuda_phase8_validate_full(
    evidence, verify_current_identity = TRUE
  )
  contracts <- fastkpc_full_cuda_phase35_load_contract_set()
  source_closure <- fastkpc_full_cuda_phase8_publication_source_closure()
  native_identity <- fastkpc_full_cuda_phase7_native_identity()
  evidence_sha256 <- fastkpc_full_cuda_census_file_hash(evidence_path)
  directories <- fastkpc_full_cuda_phase8_artifact_directories()
  results <- lapply(
    fastkpc_full_cuda_phase8_artifact_kinds(), function(kind) {
      fastkpc_full_cuda_phase8_publish_one(
        kind = kind,
        output_dir = file.path(output_root, directories[[kind]]),
        evidence = evidence, evidence_path = evidence_path,
        evidence_sha256 = evidence_sha256, contracts = contracts,
        source_closure = source_closure,
        native_identity = native_identity
      )
    }
  )
  names(results) <- fastkpc_full_cuda_phase8_artifact_kinds()
  results
}
