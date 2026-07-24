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

source("fastkpc/R/full_cuda_ci_gate.R")
shadow_path <- "fastkpc/R/full_cuda_ci_fixed_sp_shadow.R"
if (file.exists(shadow_path)) source(shadow_path)

phase0_dir <- "fastkpc/artifacts/full_cuda_ci/oracle_351x48_v1"
phase1_dir <- "fastkpc/artifacts/full_cuda_ci/workload_census_351x48_v1"
data_path <- paste0(
  "fastkpc/artifacts/kpc_tprs_real_zhu/",
  "cancer_RD-causalDiscoveryInput.rds"
)

oracle <- fastkpc_load_full_cuda_ci_oracle(phase0_dir)
phase1 <- list(
  logical_ci_tests = readRDS(file.path(phase1_dir, "logical_ci_tests.rds"))
)
canonical_data <- readRDS(data_path)

assert_true(
  exists("fastkpc_full_cuda_replay_logical_ci", mode = "function"),
  "full CUDA fixed-sp shadow logical-CI replayer should exist"
)
assert_true(
  exists("fastkpc_full_cuda_compare_candidate_skeleton", mode = "function"),
  "in-memory full CUDA candidate skeleton comparator should exist"
)
canonical_logical_contract <-
  fastkpc_full_cuda_shadow_canonical_logical_contract()
assert_true(
  identical(canonical_logical_contract$logical_test_count, 240489L) &&
    identical(
      canonical_logical_contract$canonical_logical_census_hash,
      "c9b48074dd59a439fceb9d5e64806adda5620cc4abe32095371abc447ef98634"
    ),
  "default replay contract should bind the authenticated canonical corpus"
)

expect_corpus_identity_error <- function(expression, pattern, case_name) {
  error <- tryCatch({
    force(expression)
    NULL
  }, error = identity)
  if (inherits(error, "error") && grepl(pattern, conditionMessage(error))) {
    return(character())
  }
  case_name
}

suffix_dropped <- phase1$logical_ci_tests[-nrow(phase1$logical_ci_tests),
                                          , drop = FALSE]
reordered_renumbered <- phase1$logical_ci_tests
reordered_renumbered[c(1L, 2L), ] <-
  reordered_renumbered[c(2L, 1L), ]
reordered_renumbered$logical_sequence_id <- seq_len(nrow(reordered_renumbered))
corpus_identity_failures <- c(
  expect_corpus_identity_error(
    fastkpc_full_cuda_replay_logical_ci(
      suffix_dropped,
      suffix_dropped$reference_p_value,
      colnames(canonical_data)
    ),
    "logical corpus identity count mismatch",
    "suffix-drop"
  ),
  expect_corpus_identity_error(
    fastkpc_full_cuda_replay_logical_ci(
      reordered_renumbered,
      reordered_renumbered$reference_p_value,
      colnames(canonical_data)
    ),
    "logical corpus identity hash mismatch",
    "reorder-then-renumber"
  )
)
assert_true(
  length(corpus_identity_failures) == 0L,
  paste(
    "logical corpus identity rejection missing for",
    paste(corpus_identity_failures, collapse = ", ")
  )
)

replayed <- fastkpc_full_cuda_replay_logical_ci(
  logical_tests = phase1$logical_ci_tests,
  candidate_p_value = phase1$logical_ci_tests$reference_p_value,
  labels = colnames(canonical_data)
)
comparison <- fastkpc_full_cuda_compare_candidate_skeleton(
  oracle, replayed$skeleton
)

assert_true(nrow(replayed$logical_trace) == 240489L,
            "full replay logical count")
assert_true(sum(replayed$logical_trace$deletes_edge) == 1018L,
            "full replay deletion count")
assert_true(identical(
  replayed$skeleton$n.edgetests,
  c(2213L, 52659L, 125293L, 40694L, 13293L, 5422L, 835L, 80L)
), "full replay n.edgetests")
assert_true(comparison$summary$edge_count_candidate == 110L,
            "oracle replay edge count")
assert_true(comparison$summary$SHD == 0L,
            "oracle replay SHD")
assert_true(
  isTRUE(comparison$summary$pass) &&
    comparison$summary$adjacency_identical &&
    comparison$summary$sepsets_identical &&
    comparison$summary$n_edgetests_identical &&
    comparison$summary$deletions_identical,
  "oracle replay graph evidence"
)
assert_true(
  identical(replayed$skeleton$pMax, oracle$reference$pMax),
  "oracle replay pMax"
)
assert_true(
  identical(replayed$logical_trace$logical_sequence_id,
            phase1$logical_ci_tests$logical_sequence_id) &&
    identical(replayed$logical_trace$source_sequence_id,
              phase1$logical_ci_tests$source_sequence_id) &&
    identical(replayed$logical_trace$source_task_index,
              phase1$logical_ci_tests$source_task_index) &&
    identical(replayed$logical_trace$candidate_p_value,
              phase1$logical_ci_tests$reference_p_value) &&
    identical(replayed$skeleton$tasks$source_sequence_id,
              phase1$logical_ci_tests$source_sequence_id) &&
    identical(replayed$skeleton$tasks$source_task_index,
              phase1$logical_ci_tests$source_task_index) &&
    identical(replayed$skeleton$tasks$p_candidate,
              phase1$logical_ci_tests$reference_p_value) &&
    !any(replayed$logical_trace$decision_flip),
  "reference replay should preserve lineage and decisions"
)

boundary_tests <- phase1$logical_ci_tests[1:2, , drop = FALSE]
boundary_tests$logical_sequence_id <- 1:2
boundary_tests$source_sequence_id <- 1:2
boundary_tests$source_task_index <- 1:2
boundary_tests$x <- c(1L, 1L)
boundary_tests$y <- c(2L, 3L)
boundary_tests$S_key <- c("", "")
boundary_tests$S_size <- c(0L, 0L)
boundary_tests$level <- c(0L, 0L)
boundary_tests$alpha <- c(0.1, 0.1)
boundary_tests$reference_decision <- c("dependent", "independent")
boundary_tests$reference_independent <- c(FALSE, TRUE)
boundary_tests$deletes_edge <- c(FALSE, TRUE)
boundary_contract <- fastkpc_full_cuda_shadow_logical_contract(boundary_tests)
boundary <- fastkpc_full_cuda_replay_logical_ci(
  logical_tests = boundary_tests,
  candidate_p_value = c(0.1, 0.1000000001),
  labels = c("A", "B", "C"),
  expected_logical_contract = boundary_contract
)
assert_true(
  identical(boundary$logical_trace$candidate_independent, c(FALSE, TRUE)) &&
    identical(boundary$logical_trace$candidate_decision,
              c("dependent", "independent")),
  "candidate independence must use strict candidate p-value > alpha"
)

nonfinite_p <- phase1$logical_ci_tests$reference_p_value
nonfinite_p[[1L]] <- Inf
assert_error(
  fastkpc_full_cuda_replay_logical_ci(
    phase1$logical_ci_tests, nonfinite_p, colnames(canonical_data)
  ),
  "candidate_p_value.*finite",
  "nonfinite candidate p-values must fail closed"
)

final_adjacency <- oracle$reference$adjacency
dependent_rows <- which(
  phase1$logical_ci_tests$reference_decision == "dependent" &
    final_adjacency[cbind(phase1$logical_ci_tests$x,
                          phase1$logical_ci_tests$y)]
)
mutation_index <- dependent_rows[[1L]]
mutated_p <- phase1$logical_ci_tests$reference_p_value
mutated_p[[mutation_index]] <- phase1$logical_ci_tests$alpha[[mutation_index]] * 2
mutated <- fastkpc_full_cuda_replay_logical_ci(
  phase1$logical_ci_tests, mutated_p, colnames(canonical_data)
)
mutated_comparison <- fastkpc_full_cuda_compare_candidate_skeleton(
  oracle, mutated$skeleton
)
assert_true(
  identical(which(mutated$logical_trace$decision_flip), mutation_index) &&
    isTRUE(mutated$logical_trace$deletes_edge[[mutation_index]]) &&
    isTRUE(mutated_comparison$first_divergence$first_divergence_found) &&
    identical(mutated_comparison$first_divergence$type,
              "logical_ci_trace") &&
    identical(mutated_comparison$first_divergence$logical_sequence_id,
              phase1$logical_ci_tests$logical_sequence_id[[mutation_index]]) &&
    identical(mutated_comparison$first_divergence$candidate_p,
              mutated_p[[mutation_index]]),
  "one decision mutation should produce deterministic first-divergence evidence"
)

reordered <- phase1$logical_ci_tests
reordered[c(1L, 2L), ] <- reordered[c(2L, 1L), ]
assert_error(
  fastkpc_full_cuda_replay_logical_ci(
    reordered, reordered$reference_p_value, colnames(canonical_data)
  ),
  "canonical logical_sequence_id order",
  "reordered logical rows must fail closed"
)

dropped <- phase1$logical_ci_tests[-2L, , drop = FALSE]
assert_error(
  fastkpc_full_cuda_replay_logical_ci(
    dropped, dropped$reference_p_value, colnames(canonical_data)
  ),
  "canonical logical_sequence_id order",
  "dropped logical rows must fail closed"
)

duplicated <- phase1$logical_ci_tests
duplicated$logical_sequence_id[[2L]] <-
  duplicated$logical_sequence_id[[1L]]
assert_error(
  fastkpc_full_cuda_replay_logical_ci(
    duplicated, duplicated$reference_p_value, colnames(canonical_data)
  ),
  "duplicate logical_sequence_id",
  "duplicate logical sequence IDs must fail closed"
)

cat("full CUDA CI fixed-sp shadow replay: PASS\n")
