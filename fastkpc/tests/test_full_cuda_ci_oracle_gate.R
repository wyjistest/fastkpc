fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_error <- function(expression, pattern, message) {
  error <- tryCatch({
    force(expression)
    NULL
  }, error = identity)
  assert_true(inherits(error, "error") && grepl(pattern,
                                                conditionMessage(error)),
              message)
}

gate_path <- "fastkpc/R/full_cuda_ci_gate.R"
if (file.exists(gate_path)) source(gate_path)
runner_path <- "fastkpc/tools/run_full_cuda_ci_oracle.R"

assert_true(
  exists("fastkpc_write_full_cuda_ci_oracle", mode = "function"),
  "full CUDA CI oracle writer should exist"
)
assert_true(
  exists("fastkpc_compare_full_cuda_ci_candidate", mode = "function"),
  "full CUDA CI candidate comparator should exist"
)
assert_true(
  exists("fastkpc_full_cuda_compare_candidate_skeleton", mode = "function"),
  "in-memory full CUDA CI candidate skeleton comparator should exist"
)
assert_true(file.exists(runner_path),
            "full CUDA CI oracle runner should exist")

runner_failure_root <- tempfile("full-cuda-ci-runner-failure-")
runner_stale_dir <- file.path(
  runner_failure_root, "current_correct_route_351x48_v1"
)
dir.create(runner_stale_dir, recursive = TRUE, showWarnings = FALSE)
writeLines('{"pass":true}', file.path(runner_stale_dir, "summary.json"))
runner_failure <- suppressWarnings(system2(
  "Rscript",
  c(runner_path, "missing-data.rds", "missing-oracle.rds",
    "missing-candidate.rds", runner_failure_root),
  stdout = TRUE,
  stderr = TRUE
))
assert_true(!is.null(attr(runner_failure, "status")) &&
              attr(runner_failure, "status") != 0L,
            "runner should fail when required inputs are missing")
assert_true(!file.exists(file.path(runner_stale_dir, "summary.json")),
            "runner input failure should clear stale comparison output")

environment_warnings <- character()
invisible(withCallingHandlers(
  fastkpc_full_cuda_environment_lines(),
  warning = function(warning) {
    environment_warnings <<- c(environment_warnings,
                               conditionMessage(warning))
    invokeRestart("muffleWarning")
  }
))
assert_true(length(environment_warnings) == 0L,
            "full CUDA CI environment collection should not warn")

make_sepsets <- function(p) {
  lapply(seq_len(p), function(i) {
    lapply(seq_len(p), function(j) integer())
  })
}

labels <- c("A", "B", "C", "D")
p <- length(labels)
adjacency <- matrix(TRUE, p, p, dimnames = list(labels, labels))
diag(adjacency) <- FALSE
adjacency[1L, 3L] <- adjacency[3L, 1L] <- FALSE
adjacency[2L, 4L] <- adjacency[4L, 2L] <- FALSE

reference_sepsets <- make_sepsets(p)
reference_sepsets[[2L]][[4L]] <- c(1L, 3L)
candidate_sepsets <- make_sepsets(p)
candidate_sepsets[[4L]][[2L]] <- c(3L, 1L)

pmax <- matrix(0, p, p, dimnames = list(labels, labels))
diag(pmax) <- 1
pmax[1L, 3L] <- pmax[3L, 1L] <- 0.2
pmax[2L, 4L] <- pmax[4L, 2L] <- 0.3

reference <- list(
  adjacency = adjacency,
  sepsets = reference_sepsets,
  pMax = pmax,
  n.edgetests = c(1L, 1L, 1L),
  per.level.log = list(
    list(list(x = 1L, y = 3L, S_xy = integer(), S_yx = NULL)),
    list(),
    list(list(x = 2L, y = 4L, S_xy = c(1L, 3L), S_yx = NULL))
  )
)

candidate_tasks <- data.frame(
  canonical_test_order_id = c(1L, 2L, 3L),
  task_index = c(3L, 1L, 4L),
  level = c(0L, 1L, 2L),
  edge_x = c(1L, 1L, 2L),
  edge_y = c(3L, 2L, 4L),
  x = c(1L, 1L, 2L),
  y = c(3L, 2L, 4L),
  S_key = c("", "4", "1|3"),
  p_candidate = c(0.2, 0.01, 0.3),
  native_edge_deleted = c(TRUE, FALSE, TRUE),
  native_edge_ignored = c(FALSE, FALSE, FALSE),
  stringsAsFactors = FALSE
)

candidate <- list(
  adjacency = adjacency,
  sepsets = candidate_sepsets,
  pMax = pmax,
  n.edgetests = c(1L, 1L, 1L),
  tasks = candidate_tasks,
  summary = list(
    elapsed_sec = 1.25,
    unknown_fallback_count = 0L,
    approximate_backend_count = 0L
  )
)

data <- matrix(seq_len(24), nrow = 6L, ncol = p,
               dimnames = list(NULL, labels))
canonical_contract <- fastkpc_full_cuda_canonical_contract()
assert_true(identical(canonical_contract$n, 351L) &&
              identical(canonical_contract$p, 48L) &&
              identical(canonical_contract$edge_count, 110L) &&
              nzchar(canonical_contract$source_result_hash),
            "canonical contract should identify the 351x48 oracle fixture")
synthetic_graph_hashes <- fastkpc_full_cuda_graph_hashes(reference)
synthetic_contract <- list(
  n = nrow(data),
  p = ncol(data),
  data_hash = fastkpc_full_cuda_data_hash(data),
  column_order = colnames(data),
  alpha = 0.1,
  edge_count = 4L,
  adjacency_hash = synthetic_graph_hashes$adjacency_hash,
  sepset_hash = synthetic_graph_hashes$sepset_hash,
  deletion_trace_hash = synthetic_graph_hashes$deletion_trace_hash,
  source_result_hash = NULL,
  n_edgetests = c(1L, 1L, 1L),
  index = 1L,
  numCol = 1L,
  max_conditioning_size = 2L
)
assert_true(isTRUE(fastkpc_full_cuda_validate_canonical_fixture(
  data = data,
  skeleton = reference,
  alpha = 0.1,
  index = 1L,
  numCol = 1L,
  max_conditioning_size = 2L,
  contract = synthetic_contract
)), "matching fixture should satisfy an explicit canonical contract")
changed_data <- data
changed_data[1L, 1L] <- changed_data[1L, 1L] + 1
assert_error(
  fastkpc_full_cuda_validate_canonical_fixture(
    data = changed_data,
    skeleton = reference,
    alpha = 0.1,
    index = 1L,
    numCol = 1L,
    max_conditioning_size = 2L,
    contract = synthetic_contract
  ),
  "data hash",
  "canonical fixture validation should reject different data"
)
swapped_graph <- reference
swapped_graph$adjacency[1L, 3L] <- TRUE
swapped_graph$adjacency[3L, 1L] <- TRUE
swapped_graph$adjacency[1L, 2L] <- FALSE
swapped_graph$adjacency[2L, 1L] <- FALSE
assert_error(
  fastkpc_full_cuda_validate_canonical_fixture(
    data = data,
    skeleton = swapped_graph,
    alpha = 0.1,
    index = 1L,
    numCol = 1L,
    max_conditioning_size = 2L,
    contract = synthetic_contract
  ),
  "adjacency hash",
  "canonical fixture validation should reject same-count graph drift"
)
root <- tempfile("full-cuda-ci-gate-")
oracle_dir <- file.path(root, "oracle")
comparison_dir <- file.path(root, "comparison")

oracle <- fastkpc_write_full_cuda_ci_oracle(
  reference = reference,
  logical_trace_source = candidate,
  data = data,
  output_dir = oracle_dir,
  alpha = 0.1,
  index = 1L,
  numCol = 1L,
  max_conditioning_size = 2L,
  source_result_path = "synthetic-reference.rds",
  oracle_route_environment = c(TEST_ROUTE = "legacy")
)

assert_error(
  fastkpc_write_full_cuda_ci_oracle(
    reference = reference,
    logical_trace_source = NULL,
    data = data,
    output_dir = file.path(root, "oracle-missing-trace"),
    alpha = 0.1,
    index = 1L,
    numCol = 1L,
    max_conditioning_size = 2L
  ),
  "logical trace",
  "oracle writer should reject a missing logical trace"
)

truncated_trace <- candidate
truncated_trace$tasks <- truncated_trace$tasks[-2L, , drop = FALSE]
assert_error(
  fastkpc_write_full_cuda_ci_oracle(
    reference = reference,
    logical_trace_source = truncated_trace,
    data = data,
    output_dir = file.path(root, "oracle-truncated-trace"),
    alpha = 0.1,
    index = 1L,
    numCol = 1L,
    max_conditioning_size = 2L
  ),
  "n.edgetests",
  "oracle writer should reject a trace that is incomplete by level"
)

self_renumbered_trace <- candidate
self_renumbered_trace$tasks$task_index[1L] <- 1L
assert_error(
  fastkpc_write_full_cuda_ci_oracle(
    reference = reference,
    logical_trace_source = self_renumbered_trace,
    data = data,
    output_dir = file.path(root, "oracle-self-renumbered-trace"),
    alpha = 0.1,
    index = 1L,
    numCol = 1L,
    max_conditioning_size = 2L
  ),
  "canonical layer plan",
  "oracle writer should reject self-renumbered logical task order"
)

fallback_reference <- reference
fallback_reference$summary <- list(
  legacy_dcov_native_cuda_lowrank_backend_fallback_count = 1L
)
assert_error(
  fastkpc_write_full_cuda_ci_oracle(
    reference = fallback_reference,
    logical_trace_source = candidate,
    data = data,
    output_dir = file.path(root, "oracle-reference-fallback"),
    alpha = 0.1,
    index = 1L,
    numCol = 1L,
    max_conditioning_size = 2L
  ),
  "oracle fallback",
  "oracle writer should reject nonzero reference fallback counters"
)

required_oracle_files <- c(
  "manifest.json", "summary.json", "summary.md", "adjacency.rds",
  "adjacency.csv", "sepsets.rds", "n_edgetests.csv",
  "logical_ci_trace.rds", "deletion_trace.csv", "pmax.rds",
  "near_alpha_cases.csv", "environment.txt", "commands.txt",
  "graph_agreement.csv", "sepset_agreement.csv",
  "first_divergence.json", "fallbacks.csv", "stage_timing.csv",
  "raw_runs.csv"
)
assert_true(
  all(file.exists(file.path(oracle_dir, required_oracle_files))),
  "full CUDA CI oracle artifact should contain the standard schema"
)
assert_true(isTRUE(oracle$summary$pass),
            "oracle self-comparison should pass")
manifest <- jsonlite::read_json(file.path(oracle_dir, "manifest.json"),
                                simplifyVector = TRUE)
required_manifest_fields <- c(
  "data_hash", "data_dimensions", "column_order", "alpha",
  "max_conditioning_size", "index", "numCol", "R_version",
  "mgcv_version", "package_session_information", "compiler_versions",
  "cuda_driver_version", "cuda_runtime_version", "gpu_model",
  "cpu_model", "thread_counts", "source_commit",
  "oracle_route_environment", "source_result_hash",
  "logical_ci_trace_source_hash"
)
assert_true(length(setdiff(required_manifest_fields, names(manifest))) == 0L,
            "oracle manifest should contain all Phase 0 environment fields")
assert_true(isTRUE(manifest$logical_ci_trace_available),
            "qualified trace source should populate oracle logical CI trace")
assert_true(identical(as.integer(manifest$logical_ci_trace_count), 3L),
            "oracle manifest should count logical CI trace rows")
assert_true("source_task_index" %in% names(oracle$logical_trace) &&
              identical(as.integer(oracle$logical_trace$source_task_index),
                        c(3L, 1L, 4L)),
            "oracle logical trace should retain canonical source task indices")
duplicate_task_trace <- oracle$logical_trace[c(1L, 1L, 2L), , drop = FALSE]
duplicate_task_trace$level <- c(0L, 0L, 1L)
duplicate_task_trace$logical_sequence_id <- seq_len(nrow(duplicate_task_trace))
assert_error(
  fastkpc_full_cuda_validate_logical_trace(
    duplicate_task_trace, c(2L, 1L), role = "test"
  ),
  "duplicate",
  "logical trace validation should reject duplicate source task indices"
)

tampered_oracle_dir <- file.path(root, "tampered-oracle")
dir.create(tampered_oracle_dir, recursive = TRUE, showWarnings = FALSE)
invisible(file.copy(list.files(oracle_dir, full.names = TRUE),
                    tampered_oracle_dir, recursive = TRUE))
tampered_trace_path <- file.path(tampered_oracle_dir,
                                 "logical_ci_trace.rds")
tampered_trace <- readRDS(tampered_trace_path)
saveRDS(tampered_trace[-1L, , drop = FALSE], tampered_trace_path)
assert_error(
  fastkpc_load_full_cuda_ci_oracle(tampered_oracle_dir),
  "n.edgetests",
  "oracle loader should reject a truncated logical trace artifact"
)

in_memory_comparison <- fastkpc_full_cuda_compare_candidate_skeleton(
  oracle, candidate
)
assert_true(
  isTRUE(in_memory_comparison$summary$pass) &&
    isTRUE(in_memory_comparison$summary$adjacency_identical) &&
    isTRUE(in_memory_comparison$summary$sepsets_identical) &&
    isTRUE(in_memory_comparison$summary$n_edgetests_identical) &&
    isTRUE(in_memory_comparison$summary$deletions_identical),
  "in-memory candidate skeleton comparison should use the oracle core"
)

comparison <- fastkpc_compare_full_cuda_ci_candidate(
  oracle = oracle_dir,
  candidate = candidate,
  output_dir = comparison_dir,
  candidate_route = "synthetic-candidate"
)
assert_true(isTRUE(comparison$summary$pass),
            "semantically identical candidate should pass")
assert_true(isTRUE(comparison$summary$adjacency_identical),
            "candidate adjacency should match")
assert_true(isTRUE(comparison$summary$sepsets_identical),
            "opposite-direction sepset storage should normalize equally")
assert_true(isTRUE(comparison$summary$n_edgetests_identical),
            "candidate n.edgetests should match")
assert_true(isTRUE(comparison$summary$deletions_identical),
            "candidate deletion trace should match")
required_comparison_fields <- c(
  "timeout", "source_commit",
  "deleting_test_identical", "first_divergence_found",
  "first_divergence_level", "first_divergence_edge",
  "first_divergence_S", "reference_p", "candidate_p",
  "reference_decision", "candidate_decision"
)
assert_true(
  length(setdiff(required_comparison_fields,
                 names(comparison$summary))) == 0L,
  "candidate summary should expose first-divergence fields"
)
assert_true(!isTRUE(comparison$first_divergence$first_divergence_found),
            "passing candidate should not report a first divergence")
comparison_manifest <- jsonlite::read_json(
  file.path(comparison_dir, "manifest.json"), simplifyVector = TRUE
)
assert_true(identical(comparison_manifest$oracle_data_hash,
                      manifest$data_hash) &&
              identical(as.numeric(comparison_manifest$alpha), 0.1),
            "comparison manifest should retain oracle data/config identity")

missing_fallback_counters <- candidate
missing_fallback_counters$summary$unknown_fallback_count <- NULL
missing_fallback_counters$summary$approximate_backend_count <- NULL
missing_fallback_counters$summary$compatible_cuda_route <-
  "legacy-mgcv-provider-native-legacy-dcov"
missing_fallback_result <- fastkpc_compare_full_cuda_ci_candidate(
  oracle = oracle_dir,
  candidate = missing_fallback_counters,
  output_dir = file.path(root, "missing-fallback-counters"),
  candidate_route = "missing-fallback-counters"
)
assert_true(!isTRUE(missing_fallback_result$summary$pass),
            "route text should not replace explicit fallback counters")
assert_true(identical(missing_fallback_result$first_divergence$type,
                      "fallback"),
            "missing fallback counters should emit a fallback divergence")

backend_fallback <- candidate
backend_fallback$summary$legacy_dcov_native_cuda_lowrank_backend_fallback_count <-
  7L
backend_fallback$summary$legacy_dcov_native_lowrank_spectra_fallback_full_eig_count <-
  3L
backend_fallback_result <- fastkpc_compare_full_cuda_ci_candidate(
  oracle = oracle_dir,
  candidate = backend_fallback,
  output_dir = file.path(root, "backend-fallback"),
  candidate_route = "backend-fallback"
)
assert_true(!isTRUE(backend_fallback_result$summary$pass),
            "nonzero backend fallback counters should fail")
assert_true(identical(backend_fallback_result$first_divergence$type,
                      "fallback"),
            "backend fallback should emit a fallback divergence")
assert_true(identical(
  as.integer(backend_fallback_result$summary$backend_fallback_error_count),
  10L
), "fallback total should include reason-qualified counter names")
backend_fallback_rows <- utils::read.csv(
  file.path(root, "backend-fallback", "fallbacks.csv"),
  stringsAsFactors = FALSE
)
assert_true(any(
  backend_fallback_rows$key ==
    "legacy_dcov_native_cuda_lowrank_backend_fallback_count" &
    backend_fallback_rows$count == 7L
), "fallback artifact should retain concrete backend counters")
assert_true(any(
  backend_fallback_rows$key ==
    "legacy_dcov_native_lowrank_spectra_fallback_full_eig_count" &
    backend_fallback_rows$count == 3L
), "fallback artifact should retain reason-qualified fallback counters")

shuffled_candidate <- candidate
shuffled_candidate$tasks <- shuffled_candidate$tasks[c(3L, 1L, 2L), ,
                                                     drop = FALSE]
shuffled_result <- fastkpc_compare_full_cuda_ci_candidate(
  oracle = oracle_dir,
  candidate = shuffled_candidate,
  output_dir = file.path(root, "shuffled-candidate"),
  candidate_route = "shuffled-candidate"
)
assert_true(isTRUE(shuffled_result$summary$pass),
            "task rows should normalize to canonical per-level task order")

wrapped_comparison <- fastkpc_compare_full_cuda_ci_candidate(
  oracle = oracle_dir,
  candidate = list(
    summary = data.frame(elapsed_sec = 2.5),
    facade = candidate
  ),
  output_dir = file.path(root, "wrapped-candidate"),
  candidate_route = "wrapped-candidate"
)
assert_true(identical(wrapped_comparison$summary$elapsed_sec, 2.5),
            "candidate comparator should preserve outer artifact timing")

adjacency_mismatch <- candidate
adjacency_mismatch$adjacency[1L, 2L] <- FALSE
adjacency_mismatch$adjacency[2L, 1L] <- FALSE
adjacency_result <- fastkpc_compare_full_cuda_ci_candidate(
  oracle = oracle_dir,
  candidate = adjacency_mismatch,
  output_dir = file.path(root, "adjacency-mismatch"),
  candidate_route = "adjacency-mismatch"
)
assert_true(!isTRUE(adjacency_result$summary$pass),
            "adjacency mismatch should fail")
assert_true(identical(adjacency_result$first_divergence$type, "adjacency"),
            "adjacency mismatch should be the first divergence")

logical_graph_mismatch <- adjacency_mismatch
logical_graph_mismatch$tasks$S_key[2L] <- "2"
logical_graph_result <- fastkpc_compare_full_cuda_ci_candidate(
  oracle = oracle_dir,
  candidate = logical_graph_mismatch,
  output_dir = file.path(root, "logical-graph-mismatch"),
  candidate_route = "logical-graph-mismatch"
)
assert_true(identical(logical_graph_result$first_divergence$type,
                      "logical_ci_trace"),
            "logical divergence should precede its final graph drift")

wrong_labels <- candidate
wrong <- paste0("X", seq_len(p))
dimnames(wrong_labels$adjacency) <- list(wrong, wrong)
dimnames(wrong_labels$pMax) <- list(wrong, wrong)
wrong_label_result <- fastkpc_compare_full_cuda_ci_candidate(
  oracle = oracle_dir,
  candidate = wrong_labels,
  output_dir = file.path(root, "wrong-labels"),
  candidate_route = "wrong-labels"
)
assert_true(!isTRUE(wrong_label_result$summary$pass),
            "candidate with incompatible matrix labels should fail")
assert_true(identical(wrong_label_result$first_divergence$type, "adjacency"),
            "matrix label mismatch should be reported as adjacency")

missing_labels <- candidate
dimnames(missing_labels$adjacency) <- NULL
dimnames(missing_labels$pMax) <- NULL
missing_label_result <- fastkpc_compare_full_cuda_ci_candidate(
  oracle = oracle_dir,
  candidate = missing_labels,
  output_dir = file.path(root, "missing-labels"),
  candidate_route = "missing-labels"
)
assert_true(!isTRUE(missing_label_result$summary$pass),
            "candidate without matrix labels should fail column-order gate")
assert_true(identical(missing_label_result$first_divergence$type,
                      "adjacency"),
            "missing matrix labels should be reported as adjacency")

sepset_mismatch <- candidate
sepset_mismatch$sepsets[[4L]][[2L]] <- 1L
sepset_result <- fastkpc_compare_full_cuda_ci_candidate(
  oracle = oracle_dir,
  candidate = sepset_mismatch,
  output_dir = file.path(root, "sepset-mismatch"),
  candidate_route = "sepset-mismatch"
)
assert_true(!isTRUE(sepset_result$summary$pass),
            "sepset mismatch should fail")
assert_true(identical(sepset_result$first_divergence$type, "sepset"),
            "sepset mismatch should be the first divergence")

tests_mismatch <- candidate
tests_mismatch$n.edgetests[2L] <- tests_mismatch$n.edgetests[2L] + 1L
tests_result <- fastkpc_compare_full_cuda_ci_candidate(
  oracle = oracle_dir,
  candidate = tests_mismatch,
  output_dir = file.path(root, "tests-mismatch"),
  candidate_route = "tests-mismatch"
)
assert_true(!isTRUE(tests_result$summary$pass),
            "n.edgetests mismatch should fail")
assert_true(identical(tests_result$first_divergence$type, "n_edgetests"),
            "n.edgetests mismatch should be reported")

deletion_mismatch <- candidate
deletion_mismatch$tasks$level[
  deletion_mismatch$tasks$native_edge_deleted &
    deletion_mismatch$tasks$level == 2L
] <- 1L
deletion_result <- fastkpc_compare_full_cuda_ci_candidate(
  oracle = oracle_dir,
  candidate = deletion_mismatch,
  output_dir = file.path(root, "deletion-mismatch"),
  candidate_route = "deletion-mismatch"
)
assert_true(!isTRUE(deletion_result$summary$pass),
            "deletion trace mismatch should fail")
assert_true(identical(deletion_result$first_divergence$type,
                      "logical_ci_trace"),
            "logical trace should localize a deletion mismatch first")
deletion_only_result <- fastkpc_full_cuda_compare_core(
  reference = reference,
  candidate = deletion_mismatch,
  reference_logical = fastkpc_full_cuda_empty_logical_trace()
)
assert_true(identical(deletion_only_result$first_divergence$type,
                      "deletion_trace"),
            "deletion trace mismatch should remain available without a trace")

missing_logical_trace <- candidate
missing_logical_trace$tasks <- NULL
missing_logical_result <- fastkpc_compare_full_cuda_ci_candidate(
  oracle = oracle_dir,
  candidate = missing_logical_trace,
  output_dir = file.path(root, "missing-logical-trace"),
  candidate_route = "missing-logical-trace"
)
assert_true(!isTRUE(missing_logical_result$summary$pass),
            "candidate missing oracle logical trace should fail closed")
assert_true(identical(missing_logical_result$first_divergence$type,
                      "logical_ci_trace"),
            "missing logical trace should be reported")

missing_dir <- file.path(root, "missing-candidate")
dir.create(missing_dir, recursive = TRUE, showWarnings = FALSE)
saveRDS(adjacency, file.path(missing_dir, "adjacency.rds"))
missing_result <- fastkpc_compare_full_cuda_ci_candidate(
  oracle = oracle_dir,
  candidate = NULL,
  output_dir = missing_dir,
  candidate_route = "missing-candidate"
)
assert_true(!isTRUE(missing_result$summary$pass),
            "missing candidate graph should fail closed")
assert_true(isTRUE(missing_result$first_divergence$first_divergence_found),
            "missing candidate should emit first divergence")
assert_true(!file.exists(file.path(missing_dir, "adjacency.rds")),
            "failed comparison should not retain a stale candidate graph")

cat("PASS full CUDA CI oracle gate\n")
