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

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_error <- function(expression, pattern, message) {
  error <- tryCatch(force(expression), error = identity)
  assert_true(
    inherits(error, "error") &&
      grepl(pattern, conditionMessage(error), fixed = TRUE),
    paste0(message, if (inherits(error, "error")) {
      paste0(": ", conditionMessage(error))
    } else ": no error")
  )
}

order <- fastkpc_full_cuda_phase10_campaign_order()
assert_true(
  nrow(order) == 15L &&
    identical(order$sequence, seq_len(15L)) &&
    identical(as.character(order$mode[1:5]),
              rep("candidate_cold", 5L)) &&
    identical(as.character(order$mode[6:15]), c(
      "candidate_warm", "correct_baseline",
      "correct_baseline", "candidate_warm",
      "candidate_warm", "correct_baseline",
      "correct_baseline", "candidate_warm",
      "candidate_warm", "correct_baseline"
    )) && all(table(order$mode) == 5L),
  "Phase 10 fixed campaign order drifted"
)
assert_true(
  identical(
    fastkpc_full_cuda_phase10_campaign_run_key("candidate_warm", 3L),
    "candidate_warm-03"
  ),
  "Phase 10 campaign run key is not canonical"
)
assert_error(
  fastkpc_full_cuda_phase10_campaign_run_key("candidate", 1L),
  "Phase 10 campaign run identity is invalid",
  "Phase 10 invalid campaign mode must fail closed"
)

statistics <- fastkpc_full_cuda_phase10_campaign_statistics(c(5, 1, 3, 2, 4))
assert_true(
  statistics$repetitions[[1L]] == 5L &&
    statistics$median_sec[[1L]] == 3 &&
    statistics$min_sec[[1L]] == 1 &&
    statistics$max_sec[[1L]] == 5 &&
    statistics$iqr_sec[[1L]] == 2,
  "Phase 10 campaign timing statistics are incorrect"
)
assert_error(
  fastkpc_full_cuda_phase10_campaign_statistics(c(1, NA_real_)),
  "Phase 10 campaign timing vector is invalid",
  "Phase 10 non-finite timing must fail closed"
)

hash <- strrep("a", 64L)
freeze <- list(
  schema_version = "full-cuda-ci-phase10-campaign-freeze-v1",
  source_commit = "synthetic",
  source_closure_sha256 = hash,
  native_binary_path = "/synthetic/fastkpc_cuda.so",
  native_binary_sha256 = hash,
  backend_configuration_sha256 = hash,
  build_recipe_sha256 = hash,
  contract_sha256 = list(contract = hash),
  canonical_input_sha256 = list(input = hash),
  hardening_producer_identity_sha256 = hash,
  hardening_source_closure_sha256 = hash,
  machine_identity = list(hostname = "synthetic"),
  campaign_order = fastkpc_full_cuda_phase10_campaign_order_records(),
  candidate_cold_repetitions = 5L,
  candidate_warm_repetitions = 5L,
  correct_baseline_repetitions = 5L,
  holdout_state = "SEALED_NOT_RELEASED",
  holdout_opened = FALSE
)
freeze$freeze_identity_sha256 <- fastkpc_full_cuda_phase35_sha256_utf8(
  fastkpc_full_cuda_phase35_canonical_json(freeze)
)
fastkpc_full_cuda_phase10_campaign_validate_freeze(freeze)
tampered_freeze <- freeze
tampered_freeze$native_binary_sha256 <- strrep("b", 64L)
assert_error(
  fastkpc_full_cuda_phase10_campaign_validate_freeze(tampered_freeze),
  "Phase 10 campaign freeze identity mismatch",
  "Phase 10 campaign freeze tampering must fail closed"
)

summary <- list(
  schema_version = "full-cuda-ci-phase10-campaign-summary-v1",
  run_status = "ok", timeout = FALSE,
  edge_count_reference = 110L, edge_count_candidate = 110L, SHD = 0L,
  adjacency_identical = TRUE, sepsets_identical = TRUE,
  n_edgetests_identical = TRUE, deletions_identical = TRUE,
  logical_ci_trace_identical = TRUE,
  max_conditioning_size_requested = "Inf",
  max_conditioning_size_resolved = 46L,
  natural_stop_level = 8L,
  logical_tests_consumed = 240498L,
  candidate_cold_repetitions = 5L,
  candidate_warm_repetitions = 5L,
  correct_baseline_repetitions = 5L,
  complete_warmup_repetitions = 5L,
  cold_median_sec = 90, cold_min_sec = 80, cold_max_sec = 100,
  cold_mad_sec = 5, cold_iqr_sec = 10,
  warm_median_sec = 1, warm_min_sec = 0.9, warm_max_sec = 1.1,
  warm_mad_sec = 0.05, warm_iqr_sec = 0.1,
  baseline_median_sec = 600, baseline_min_sec = 590,
  baseline_max_sec = 610, baseline_mad_sec = 5,
  baseline_iqr_sec = 10, candidate_to_baseline_ratio = 1 / 600,
  absolute_performance_gate = TRUE, relative_performance_gate = TRUE,
  repeatability_gate = TRUE, every_run_correctness_gate = TRUE,
  every_candidate_authority_gate = TRUE,
  result_cache_capacity = 262144L, target_cache_capacity = 131072L,
  native_setup_cache_capacity = 64L, component_cache_capacity = 47L,
  hardening_gate = TRUE, holdout_state = "SEALED_NOT_RELEASED",
  holdout_gate = FALSE, phase10_canonical_campaign_claim = TRUE,
  phase10_promotion_claim = FALSE, recommended_route = FALSE,
  elapsed_sec = 1000, pass = TRUE
)
for (field in fastkpc_full_cuda_phase10_campaign_authority_zero_fields()) {
  summary[[field]] <- 0L
}
for (field in c(
  "freeze_identity_sha256", "architecture_contract_sha256",
  "numerical_contract_sha256", "artifact_identity_contract_sha256",
  "reference_machine_contract_sha256",
  "performance_budget_contract_sha256", "source_evidence_sha256",
  "producer_identity_sha256", "source_closure_sha256",
  "native_binary_sha256", "hardening_producer_identity_sha256"
)) {
  summary[[field]] <- hash
}
fastkpc_full_cuda_phase10_campaign_validate_summary(summary)
bad_summary <- summary
bad_summary$warm_median_sec <- 121
bad_summary$absolute_performance_gate <- FALSE
assert_error(
  fastkpc_full_cuda_phase10_campaign_validate_summary(bad_summary),
  "Phase 10 canonical campaign summary gate failed",
  "Phase 10 absolute performance regression must fail closed"
)

closure <- fastkpc_full_cuda_phase10_campaign_source_closure()
assert_true(
  nrow(closure$table) == length(closure$hashes) &&
    all(grepl("^[0-9a-f]{64}$", closure$table$sha256)) &&
    grepl("^[0-9a-f]{64}$", closure$sha256) &&
    "fastkpc/tools/run_full_cuda_ci_phase10_worker.R" %in%
      closure$table$path,
  "Phase 10 campaign source closure is incomplete"
)

cold_warm_path <- Sys.getenv(
  "FASTKPC_PHASE10_COLD_WARM_DIAGNOSTIC_RDS",
  unset = ""
)
baseline_path <- Sys.getenv(
  "FASTKPC_PHASE10_BASELINE_DIAGNOSTIC_RDS",
  unset = ""
)
if (nzchar(cold_warm_path) && nzchar(baseline_path) &&
    file.exists(cold_warm_path) && file.exists(baseline_path)) {
  cold_warm <- readRDS(cold_warm_path)
  baseline <- readRDS(baseline_path)
  fastkpc_full_cuda_phase10_validate_candidate_result(
    cold_warm$cold$result, boundary = "cold"
  )
  fastkpc_full_cuda_phase10_validate_candidate_result(
    cold_warm$warm$result, boundary = "warm"
  )
  fastkpc_full_cuda_phase10_validate_baseline_result(baseline$result)
  cases <- fastkpc_full_cuda_phase10_campaign_cases(cold_warm$cold$result)
  assert_true(
    sum(is.finite(cases$absolute_log_distance_from_alpha) &
          cases$absolute_log_distance_from_alpha <= log(2)) == 1529L,
    "Phase 10 near-alpha corpus does not use the frozen log-ratio rule"
  )
}

cat("PASS Phase 10 campaign helper, freeze, and boundary gates\n")
