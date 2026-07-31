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

contracts <- fastkpc_full_cuda_phase35_load_contract_set()
contract <- fastkpc_full_cuda_phase10_holdout_contract(contracts)
assert_true(
  identical(contract$state, "SEALED_NOT_RELEASED") &&
    !isTRUE(contract$payload_present_in_repository) &&
    identical(
      contract$custody$release_token_environment_variable,
      "FASTKPC_PROMOTION_HOLDOUT_RELEASE_TOKEN"
    ),
  "Phase 10 holdout tracked custody contract drifted"
)

token <- paste(rep("release-token", 4L), collapse = "-")
attestation <- list(
  schema_version =
    "full-cuda-ci-promotion-holdout-release-attestation-v1",
  custody_authority = contract$custody$authority,
  holdout_id = contract$holdout_id,
  contract_manifest_identity_sha256 =
    contract$commitment$manifest_identity_sha256,
  payload_file = "payload.rds",
  payload_sha256 = strrep("a", 64L),
  release_token_sha256 = fastkpc_full_cuda_phase35_sha256_utf8(token),
  released_at_utc = "2026-07-31T00:00:00Z"
)
attestation$attestation_identity_sha256 <-
  fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(attestation)
  )
fastkpc_full_cuda_phase10_holdout_validate_attestation(
  attestation, token, contracts
)
bad_attestation <- attestation
bad_attestation$payload_file <- "../payload.rds"
assert_error(
  fastkpc_full_cuda_phase10_holdout_validate_attestation(
    bad_attestation, token, contracts
  ),
  "Phase 10 holdout custodian attestation is malformed",
  "Phase 10 holdout payload traversal must fail closed"
)
assert_error(
  fastkpc_full_cuda_phase10_holdout_validate_attestation(
    attestation, paste0(token, "-wrong"), contracts
  ),
  "Phase 10 holdout release token is invalid",
  "Phase 10 holdout wrong token must fail closed"
)

campaign <- list(
  producer = list(identity_sha256 = strrep("b", 64L)),
  evidence = list(freeze = list(
    freeze_identity_sha256 = strrep("c", 64L)
  ))
)
event <- fastkpc_full_cuda_phase10_holdout_open_event(
  campaign, attestation, attestation$release_token_sha256,
  "2026-07-31T00:01:00Z"
)
fastkpc_full_cuda_phase10_holdout_validate_open_event(
  event, campaign, attestation
)
bad_event <- event
bad_event$state <- "PAYLOAD_READ_BEFORE_RECORD"
assert_error(
  fastkpc_full_cuda_phase10_holdout_validate_open_event(
    bad_event, campaign, attestation
  ),
  "Phase 10 holdout open event is malformed",
  "Phase 10 holdout invalid open state must fail closed"
)

assert_error(
  fastkpc_full_cuda_phase10_validate_holdout_payload(list()),
  "Phase 10 holdout payload is malformed",
  "Phase 10 malformed holdout payload must fail closed"
)

coverage <- stats::setNames(as.list(rep(TRUE, 9L)), c(
  "different_n", "different_p", "s_size_1", "s_size_2",
  "s_size_gt_2", "collinearity", "near_constants",
  "multiple_penalty_counts", "near_alpha_decisions"
))
summary <- list(
  schema_version = "full-cuda-ci-phase10-holdout-summary-v1",
  run_status = "ok", timeout = FALSE,
  holdout_id = "full-cuda-ci-promotion-holdout-v1",
  holdout_state = "OPENED_VALIDATED",
  release_event_persisted_before_payload_read = TRUE,
  run_once_gate = TRUE, case_count = 2L, logical_test_count = 100L,
  maximum_SHD = 0L, adjacency_identical = TRUE,
  sepsets_identical = TRUE, n_edgetests_identical = TRUE,
  deletions_identical = TRUE, logical_ci_trace_identical = TRUE,
  decision_flip_count = 0L, unknown_fallback_count = 0L,
  approximate_backend_count = 0L, cpu_numerical_fallback_count = 0L,
  nonfinite_result_count = 0L, coverage = coverage,
  coverage_gate = TRUE, canonical_campaign_gate = TRUE,
  elapsed_sec = 10, phase10_promotion_claim = TRUE,
  recommended_compatible_cuda = TRUE,
  possible_default_requires_explicit_approval = TRUE,
  pass = TRUE
)
for (field in c(
  "canonical_campaign_producer_identity_sha256",
  "canonical_campaign_freeze_identity_sha256", "payload_sha256",
  "release_attestation_identity_sha256", "open_event_identity_sha256",
  "architecture_contract_sha256", "numerical_contract_sha256",
  "artifact_identity_contract_sha256",
  "promotion_holdout_contract_sha256", "source_evidence_sha256",
  "producer_identity_sha256", "source_closure_sha256",
  "native_binary_sha256"
)) summary[[field]] <- strrep("d", 64L)
fastkpc_full_cuda_phase10_holdout_validate_summary(summary)
bad_summary <- summary
bad_summary$coverage$near_alpha_decisions <- FALSE
bad_summary$coverage_gate <- FALSE
assert_error(
  fastkpc_full_cuda_phase10_holdout_validate_summary(bad_summary),
  "Phase 10 holdout summary gate failed",
  "Phase 10 incomplete holdout coverage must fail closed"
)

temporary <- tempfile("phase10-holdout-unreleased-")
assert_error(
  fastkpc_full_cuda_phase10_release_holdout(
    release_dir = "", token = "", staging_dir = paste0(temporary, "-s"),
    output_dir = paste0(temporary, "-o")
  ),
  "Phase 10 holdout release requires a new custodian envelope",
  "Phase 10 holdout release without custody authority must fail closed"
)

closure <- fastkpc_full_cuda_phase10_holdout_source_closure()
assert_true(
  grepl("^[0-9a-f]{64}$", closure$sha256) &&
    all(grepl("^[0-9a-f]{64}$", closure$table$sha256)) &&
    "fastkpc/tests/test_full_cuda_ci_phase10_completion_audit.R" %in%
      closure$table$path &&
    "fastkpc/tools/run_full_cuda_ci_phase10_holdout.R" %in%
      closure$table$path,
  "Phase 10 holdout source closure is incomplete"
)

cat("PASS Phase 10 sealed-holdout custody and fail-closed helpers\n")
