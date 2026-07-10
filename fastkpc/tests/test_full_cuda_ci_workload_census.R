fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_error <- function(expression, pattern, message) {
  error <- tryCatch({
    force(expression)
    NULL
  }, error = identity)
  assert_true(
    inherits(error, "error") && grepl(pattern, conditionMessage(error)),
    message
  )
}

source("fastkpc/R/full_cuda_ci_workload_census.R")

trace <- data.frame(
  logical_sequence_id = 1:5,
  source_sequence_id = c(11L, 12L, 13L, 14L, 15L),
  source_task_index = c(1L, 4L, 9L, 15L, 22L),
  level = c(0L, 1L, 2L, 3L, 1L),
  x = c(1L, 1L, 2L, 3L, 4L),
  y = c(2L, 3L, 4L, 5L, 5L),
  S_key = c("", "4", "1|5", "1|2|6", "2"),
  p_value = c(0.02, 0.1, 0.08, 0.20, 0),
  deletes_edge = c(FALSE, TRUE, FALSE, TRUE, FALSE),
  stringsAsFactors = FALSE
)
deletions <- data.frame(
  level = c(1L, 3L),
  edge_x = c(3L, 3L),
  edge_y = c(1L, 5L),
  S_key = c("4", "1|2|6"),
  stringsAsFactors = FALSE
)

rows <- fastkpc_full_cuda_census_logical_tests(
  trace = trace,
  deletions = deletions,
  alpha = 0.1,
  data_hash = paste(rep("a", 64L), collapse = ""),
  n = 20L,
  p = 6L
)

expected_fields <- c(
  "logical_sequence_id", "source_sequence_id", "source_task_index",
  "level", "x", "y", "S_key", "S_size", "formula_class",
  "reference_p_value", "alpha", "reference_decision",
  "reference_independent", "deletes_edge", "selected_sepset",
  "signed_distance_from_alpha", "absolute_distance_from_alpha",
  "signed_log_ratio_from_alpha", "absolute_log_distance_from_alpha",
  "residual_key_x", "residual_key_y"
)
assert_true(identical(names(rows), expected_fields),
            "logical census fields must match the approved schema")
assert_true(identical(rows$source_sequence_id, trace$source_sequence_id),
            "logical census must retain source_sequence_id")
assert_true(identical(rows$formula_class,
                      c("direct-ci", "full-smooth", "full-smooth",
                        "additive-smooth", "full-smooth")),
            "logical census must reuse the compatibility formula enum")
assert_true(all(is.na(rows[rows$level == 0L, c("residual_key_x",
                                               "residual_key_y")])),
            "level 0 must not create GAM residual keys")
assert_true(all(nzchar(rows$residual_key_x[rows$level > 0L])) &&
              all(nzchar(rows$residual_key_y[rows$level > 0L])),
            "conditional rows must reference canonical residual hashes")
assert_true(identical(rows$reference_independent, trace$deletes_edge),
            "reference decisions must be explicit")
assert_true(identical(rows$selected_sepset, trace$deletes_edge),
            "selected sepsets must follow deleting logical rows")
assert_true(rows$signed_distance_from_alpha[[2L]] == 0,
            "signed alpha distance must retain the boundary")
assert_true(rows$reference_decision[[2L]] == "independent",
            "legacy replay uses p >= alpha")
assert_true(is.finite(rows$signed_log_ratio_from_alpha[[5L]]),
            "zero p-values must use the manifest p floor")

bad_decision <- trace
bad_decision$deletes_edge[[2L]] <- FALSE
assert_error(
  fastkpc_full_cuda_census_logical_tests(
    bad_decision, deletions, 0.1,
    paste(rep("a", 64L), collapse = ""), 20L, 6L
  ),
  "alpha decision",
  "p-value and deletes_edge disagreement must fail closed"
)

bad_s <- trace
bad_s$S_key[[3L]] <- "5|1"
assert_error(
  fastkpc_full_cuda_census_logical_tests(
    bad_s, deletions, 0.1,
    paste(rep("a", 64L), collapse = ""), 20L, 6L
  ),
  "canonical sorted unique",
  "unsorted conditioning sets must fail closed"
)

bad_p <- trace
bad_p$p_value[[1L]] <- NA_real_
assert_error(
  fastkpc_full_cuda_census_logical_tests(
    bad_p, deletions, 0.1,
    paste(rep("a", 64L), collapse = ""), 20L, 6L
  ),
  "non-finite",
  "non-finite canonical p-values must fail closed"
)

bad_deletions <- deletions
bad_deletions$S_key[[1L]] <- "5"
assert_error(
  fastkpc_full_cuda_census_logical_tests(
    trace, bad_deletions, 0.1,
    paste(rep("a", 64L), collapse = ""), 20L, 6L
  ),
  "deletion trace",
  "deleting rows must match the deletion trace one to one"
)

duplicate_row <- rows[2L, , drop = FALSE]
duplicate_row$logical_sequence_id <- 6L
duplicate_row$source_sequence_id <- 16L
duplicate_row$source_task_index <- 30L
request_rows <- rbind(rows, duplicate_row)
requests <- fastkpc_full_cuda_census_residual_requests(request_rows)

expected_request_fields <- c(
  "residual_key_payload", "residual_key_sha256", "target", "S_key",
  "S_size", "formula_class", "same_S_group_id", "same_S_group_size",
  "request_multiplicity", "first_logical_sequence_id",
  "last_logical_sequence_id", "first_level", "last_level"
)
assert_true(identical(names(requests), expected_request_fields),
            "residual request fields must match the approved schema")
assert_true(sum(requests$request_multiplicity) == 10L,
            "five conditional logical rows must expand to ten requests")
assert_true(nrow(requests) == 8L,
            "duplicate target|S requests must deduplicate globally")
assert_true(sum(requests$request_multiplicity == 2L) == 2L,
            "the duplicated CI row must increase both residual multiplicities")
assert_true(all(requests$same_S_group_size == 2L),
            "same-S group size must count unique target keys")
assert_true(!anyDuplicated(requests$residual_key_payload) &&
              !anyDuplicated(requests$residual_key_sha256),
            "canonical payloads and hashes must be one-to-one")
assert_true(all(grepl("\n$", requests$residual_key_payload)) &&
              !any(grepl("\r", requests$residual_key_payload)),
            "canonical payloads must use one final LF and no CR")
assert_true(all(nchar(requests$residual_key_sha256) == 64L),
            "residual keys must use lowercase SHA-256")
assert_true(identical(requests$residual_key_sha256,
                      sort(requests$residual_key_sha256, method = "radix")),
            "residual request rows must be sorted by SHA-256")
corpus_hash <- attr(requests, "canonical_key_corpus_hash", exact = TRUE)
assert_true(is.character(corpus_hash) && length(corpus_hash) == 1L &&
              nchar(corpus_hash) == 64L,
            "request table must carry a canonical corpus hash")

assert_error(
  fastkpc_full_cuda_census_residual_requests(
    request_rows,
    hash_fun = function(value) paste(rep("0", 64L), collapse = "")
  ),
  "collision",
  "one hash for multiple payloads must fail closed"
)

oracle_dir <- "fastkpc/artifacts/full_cuda_ci/oracle_351x48_v1"
data_path <- paste0(
  "fastkpc/artifacts/kpc_tprs_real_zhu/",
  "cancer_RD-causalDiscoveryInput.rds"
)
contract <- fastkpc_full_cuda_census_input_contract()
assert_true(length(contract$file_hashes) == 13L,
            "Phase 1 input contract must pin all oracle and dataset files")
assert_true(
  identical(
    contract$oracle_input_bundle_sha256,
    "7700bc78240984c36f8ae5ca281362a0afb8d7dedd34a5711ce4ab76a2ebee0e"
  ),
  "Phase 1 input contract must pin the reviewed bundle hash"
)

inputs <- fastkpc_full_cuda_census_load_inputs(
  oracle_dir = oracle_dir,
  data_path = data_path,
  contract = contract
)
assert_true(isTRUE(inputs$oracle_inherited_graph_gate),
            "Phase 1 must independently validate the inherited graph gate")
assert_true(identical(inputs$new_candidate_graph_gate, "NOT_APPLICABLE"),
            "Phase 1 must not claim a new candidate graph gate")
assert_true(nrow(inputs$oracle_input_hashes) == 13L &&
              all(inputs$oracle_input_hashes$identical),
            "Phase 1 must expose all validated input hashes")
assert_true(identical(inputs$oracle_input_bundle_sha256,
                      contract$oracle_input_bundle_sha256),
            "loaded input bundle hash must match the frozen contract")
assert_true(identical(dim(inputs$data), c(351L, 48L)),
            "loaded canonical data must retain the 351x48 dimensions")

bad_manifest_oracle <- inputs$oracle
bad_manifest_oracle$manifest$alpha <- 0.2
assert_error(
  fastkpc_full_cuda_census_validate_semantic_inputs(
    inputs$data, bad_manifest_oracle, oracle_dir, contract
  ),
  "manifest",
  "manifest alpha/config drift must fail independent semantic validation"
)

bad_summary_oracle <- inputs$oracle
bad_summary_oracle$summary$unknown_fallback_count <- 1L
assert_error(
  fastkpc_full_cuda_census_validate_inherited_evidence(
    bad_summary_oracle, oracle_dir, contract
  ),
  "fallback",
  "summary fallback counters must be checked independently of pass=true"
)

bad_edges_oracle <- inputs$oracle
bad_edges_oracle$summary$edge_count_candidate <- 109L
assert_error(
  fastkpc_full_cuda_census_validate_inherited_evidence(
    bad_edges_oracle, oracle_dir, contract
  ),
  "edge count",
  "summary edge counts must be checked independently of pass=true"
)

structural <- fastkpc_full_cuda_census_structural(inputs)
assert_true(nrow(structural$logical_tests) == 240489L,
            "canonical structural census must include every logical test")
assert_true(sum(structural$logical_tests$S_size > 0L) == 238276L,
            "canonical structural census must separate conditional tests")
assert_true(
  sum(structural$residual_requests$request_multiplicity) == 476552L,
  "canonical structural census must preserve every residual request"
)
assert_true(nrow(structural$residual_requests) == 110617L,
            "canonical structural census must deduplicate target|S globally")
assert_true(
  length(unique(structural$residual_requests$same_S_group_id)) == 8634L,
  "canonical structural census must identify every nonempty S group"
)
assert_true(
  identical(
    structural$canonical_key_corpus_hash,
    "b843630969f116da63f7fad095c54de2ff471540159ff97ca56c3871d6b2e1fa"
  ),
  "canonical structural census must reproduce the frozen corpus hash"
)
assert_true(
  identical(structural$historical_route_metric$value, 273284L) &&
    identical(structural$historical_route_metric$metric_role,
              "historical_route_metric") &&
    !isTRUE(structural$historical_route_metric$hard_gate),
  "S-affinity fit count must remain a non-gating historical metric"
)
assert_true(isTRUE(fastkpc_full_cuda_census_validate_structural(
  structural, canonical = TRUE
)), "canonical structural hard gates must pass")

tampered_dir <- tempfile("phase1-oracle-tampered-")
dir.create(tampered_dir, recursive = TRUE, showWarnings = FALSE)
oracle_files <- list.files(oracle_dir, full.names = TRUE)
invisible(file.copy(oracle_files, tampered_dir, overwrite = TRUE))
cat("\n", file = file.path(tampered_dir, "summary.json"), append = TRUE)
assert_error(
  fastkpc_full_cuda_census_load_inputs(
    oracle_dir = tampered_dir,
    data_path = data_path,
    contract = contract
  ),
  "input hash mismatch",
  "one-byte oracle input drift must fail before parsing"
)

cat("PASS full CUDA CI workload census structural helpers\n")
