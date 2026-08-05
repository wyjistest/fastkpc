source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_shadow.R")
source("fastkpc/R/full_cuda_ci_phase3_artifacts.R")
source("fastkpc/R/full_cuda_ci_phase35_contracts.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_phase35_bakeoff.R")
source("fastkpc/R/full_cuda_ci_single_penalty_gcv.R")
source("fastkpc/R/full_cuda_ci_phase4_artifacts.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cpp.R")
source("fastkpc/R/full_cuda_ci_phase5_artifacts.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cuda.R")
source("fastkpc/R/full_cuda_ci_phase6_artifacts.R")
source("fastkpc/R/full_cuda_ci_native_setup.R")
source("fastkpc/R/full_cuda_ci_phase7_artifacts.R")
source("fastkpc/R/full_cuda_ci_phase7_publication.R")
source("fastkpc/R/full_cuda_ci_phase8_dcov.R")
source("fastkpc/R/full_cuda_ci_phase8_publication.R")
source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase9_artifact.R")
source("fastkpc/R/full_cuda_ci_phase10_hardening.R")
source("fastkpc/R/full_cuda_ci_phase10_campaign.R")
source("fastkpc/R/full_cuda_ci_phase10_holdout.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

campaign_dir <- fastkpc_full_cuda_phase10_campaign_artifact_dir()
hardening_dir <- fastkpc_full_cuda_phase10_hardening_artifact_dir()
holdout_dir <- fastkpc_full_cuda_phase10_holdout_artifact_dir()

assert_true(
  file.exists(file.path(holdout_dir, "manifest.json")),
  paste(
    "Phase 10 completion audit requires the externally released sealed",
    "holdout artifact"
  )
)

campaign <- fastkpc_full_cuda_phase10_validate_campaign_artifact(
  campaign_dir, verify_current_sources = TRUE
)
hardening <- fastkpc_full_cuda_phase10_validate_hardening_artifact(
  hardening_dir, verify_current_sources = TRUE
)
holdout <- fastkpc_full_cuda_phase10_validate_holdout_artifact(
  holdout_dir, verify_current_sources = TRUE
)
contracts <- fastkpc_full_cuda_phase35_load_contract_set()

campaign_summary <- campaign$summary
hardening_summary <- hardening$summary
holdout_summary <- holdout$summary
expected_n_edgetests <- c(
  2213L, 52659L, 125293L, 40694L, 13293L, 5422L, 835L, 80L, 9L
)
authority_fields <- fastkpc_full_cuda_phase10_campaign_authority_zero_fields()

assert_true(
  isTRUE(campaign_summary$pass) &&
    identical(campaign_summary$run_status, "ok") &&
    identical(campaign_summary$candidate_route,
              "compatible.cuda/full_cuda-explicit") &&
    campaign_summary$edge_count_reference == 110L &&
    campaign_summary$edge_count_candidate == 110L &&
    campaign_summary$SHD == 0L &&
    isTRUE(campaign_summary$adjacency_identical) &&
    isTRUE(campaign_summary$sepsets_identical) &&
    isTRUE(campaign_summary$n_edgetests_identical) &&
    isTRUE(campaign_summary$deletions_identical) &&
    isTRUE(campaign_summary$logical_ci_trace_identical) &&
    identical(campaign_summary$max_conditioning_size_requested, "Inf") &&
    campaign_summary$max_conditioning_size_resolved == 46L &&
    campaign_summary$natural_stop_level == 8L &&
    campaign_summary$logical_tests_consumed == 240498L &&
    campaign_summary$candidate_cold_repetitions == 5L &&
    campaign_summary$candidate_warm_repetitions == 5L &&
    campaign_summary$correct_baseline_repetitions == 5L &&
    campaign_summary$complete_warmup_repetitions == 5L &&
    isTRUE(campaign_summary$repeatability_gate) &&
    isTRUE(campaign_summary$every_run_correctness_gate) &&
    isTRUE(campaign_summary$every_candidate_authority_gate),
  "Phase 10 completion audit canonical correctness or repeatability failed"
)

assert_true(
  campaign_summary$warm_median_sec <= 120 &&
    campaign_summary$candidate_to_baseline_ratio <= 0.80 &&
    isTRUE(campaign_summary$absolute_performance_gate) &&
    isTRUE(campaign_summary$relative_performance_gate) &&
    isTRUE(campaign_summary$warm_stretch_gate),
  "Phase 10 completion audit performance gate failed"
)

assert_true(
  all(vapply(authority_fields, function(field) {
    identical(as.numeric(campaign_summary[[field]]), 0)
  }, logical(1L))) &&
    campaign_summary$result_cache_capacity == 262144L &&
    campaign_summary$target_cache_capacity == 131072L &&
    campaign_summary$native_setup_cache_capacity == 64L &&
    campaign_summary$component_cache_capacity == 47L,
  "Phase 10 completion audit CUDA authority or bounded-cache gate failed"
)

campaign_n_edgetests <- utils::read.csv(
  file.path(campaign_dir, "n_edgetests.csv"), stringsAsFactors = FALSE
)
campaign_raw <- utils::read.csv(
  file.path(campaign_dir, "raw_runs.csv"), stringsAsFactors = FALSE
)
assert_true(
  identical(as.integer(campaign_n_edgetests$reference),
            expected_n_edgetests) &&
    identical(as.integer(campaign_n_edgetests$candidate),
              expected_n_edgetests) &&
    nrow(campaign_raw) == 15L && all(campaign_raw$SHD == 0L) &&
    all(as.logical(campaign_raw$adjacency_identical)) &&
    all(as.logical(campaign_raw$sepsets_identical)) &&
    all(as.logical(campaign_raw$n_edgetests_identical)) &&
    all(as.logical(campaign_raw$deletions_identical)) &&
    all(as.logical(campaign_raw$logical_ci_trace_identical)) &&
    all(as.logical(campaign_raw$pass)),
  "Phase 10 completion audit raw canonical campaign evidence failed"
)

assert_true(
  isTRUE(hardening_summary$pass) &&
    isTRUE(hardening_summary$hardening_gate) &&
    hardening_summary$unsupported_semantic_case_count >= 10L &&
    hardening_summary$fail_closed_case_count ==
      hardening_summary$unsupported_semantic_case_count &&
    hardening_summary$partial_graph_publish_count == 0L &&
    isTRUE(hardening_summary$cache_reconstruction_gate) &&
    identical(hardening_summary$stream_counts, "1|2|4") &&
    isTRUE(hardening_summary$stream_determinism_gate) &&
    isTRUE(hardening_summary$pathology_finite_gate) &&
    hardening_summary$repeated_run_count >= 12L &&
    hardening_summary$tracked_resource_leak_count == 0L &&
    hardening_summary$near_alpha_final_decision_flip_count == 0L &&
    hardening_summary$test_failure_count == 0L,
  "Phase 10 completion audit hardening or fail-closed gate failed"
)

required_coverage <- c(
  "different_n", "different_p", "s_size_1", "s_size_2",
  "s_size_gt_2", "collinearity", "near_constants",
  "multiple_penalty_counts", "near_alpha_decisions"
)
assert_true(
  isTRUE(holdout_summary$pass) &&
    identical(holdout_summary$holdout_state, "OPENED_VALIDATED") &&
    isTRUE(holdout_summary$release_event_persisted_before_payload_read) &&
    isTRUE(holdout_summary$run_once_gate) &&
    holdout_summary$maximum_SHD == 0L &&
    isTRUE(holdout_summary$adjacency_identical) &&
    isTRUE(holdout_summary$sepsets_identical) &&
    isTRUE(holdout_summary$n_edgetests_identical) &&
    isTRUE(holdout_summary$deletions_identical) &&
    isTRUE(holdout_summary$logical_ci_trace_identical) &&
    holdout_summary$decision_flip_count == 0L &&
    holdout_summary$unknown_fallback_count == 0L &&
    holdout_summary$approximate_backend_count == 0L &&
    holdout_summary$cpu_numerical_fallback_count == 0L &&
    holdout_summary$nonfinite_result_count == 0L &&
    identical(names(holdout_summary$coverage), required_coverage) &&
    all(unlist(holdout_summary$coverage, use.names = FALSE)) &&
    isTRUE(holdout_summary$coverage_gate) &&
    isTRUE(holdout_summary$canonical_campaign_gate) &&
    isTRUE(holdout_summary$phase10_promotion_claim) &&
    isTRUE(holdout_summary$recommended_compatible_cuda) &&
    isTRUE(holdout_summary$possible_default_requires_explicit_approval),
  "Phase 10 completion audit sealed-holdout promotion gate failed"
)

contract_hash_fields <- c(
  architecture_contract_v1 = "architecture_contract_sha256",
  numerical_contract_v1 = "numerical_contract_sha256",
  artifact_identity_contract_v1 = "artifact_identity_contract_sha256",
  reference_machine_v1 = "reference_machine_contract_sha256",
  performance_budget_v1 = "performance_budget_contract_sha256"
)
contract_names <- fastkpc_full_cuda_phase35_contract_names()
assert_true(
  all(vapply(names(contract_hash_fields), function(name) {
    identical(
      campaign_summary[[contract_hash_fields[[name]]]],
      contracts[[name]]$sha256
    )
  }, logical(1L))) && identical(
    holdout_summary$promotion_holdout_contract_sha256,
    contracts$promotion_holdout_manifest_v1$sha256
  ) && identical(names(campaign$producer$contract_snapshots),
                 contract_names) && all(vapply(contract_names, function(name) {
    identical(
      campaign$producer$contract_snapshots[[name]]$sha256,
      contracts[[name]]$sha256
    )
  }, logical(1L))) && identical(
    names(holdout$producer$contract_snapshots), contract_names
  ) && all(vapply(contract_names, function(name) {
    identical(
      holdout$producer$contract_snapshots[[name]]$sha256,
      contracts[[name]]$sha256
    )
  }, logical(1L))) && identical(
    contracts$promotion_holdout_manifest_v1$payload$state,
    "SEALED_NOT_RELEASED"
  ),
  "Phase 10 completion audit tracked contract authority failed"
)

read_document <- function(path) paste(readLines(path, warn = FALSE),
                                      collapse = "\n")
root_readme <- read_document("README.md")
package_readme <- read_document("fastkpc/README.md")
goal <- read_document("goal-5.6.md")
documentation <- paste(root_readme, package_readme, sep = "\n")
required_documentation <- c(
  "fastkpc_compatible_cuda_skeleton",
  "route = \"full_cuda\"",
  "compatible_cuda_strict = TRUE",
  "bash fastkpc/tools/build_cuda_native.sh",
  "bash fastkpc/tools/run_full_cuda_ci_gate.sh",
  "FASTKPC_PROMOTION_HOLDOUT_RELEASE_DIR",
  "FASTKPC_PROMOTION_HOLDOUT_RELEASE_TOKEN",
  "Gaussian",
  "identity",
  "fail",
  "mgcv clone"
)
assert_true(
  all(vapply(required_documentation, grepl, logical(1L),
             x = documentation, fixed = TRUE)) &&
    grepl(
      "| 10 | Full gate, hardening, and promotion | COMPLETE",
      goal, fixed = TRUE
    ),
  "Phase 10 completion audit documentation or roadmap closure failed"
)

cat(
  "PASS Phase 10 Section 13 completion audit; campaign=",
  campaign$producer$identity_sha256,
  " holdout=", holdout$producer$identity_sha256,
  "\n", sep = ""
)
