fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

source("fastkpc/R/full_cuda_ci_workload_census.R")

output_dir <- tempfile("full-cuda-ci-census-real-subset-")
runner <- "fastkpc/tools/run_full_cuda_ci_workload_census.R"
runner_env <- c(
  paste0("FASTKPC_FULL_CUDA_CENSUS_ORACLE_DIR=",
         "fastkpc/artifacts/full_cuda_ci/oracle_351x48_v1"),
  paste0("FASTKPC_FULL_CUDA_CENSUS_DATA_PATH=",
         "fastkpc/artifacts/kpc_tprs_real_zhu/",
         "cancer_RD-causalDiscoveryInput.rds"),
  paste0("FASTKPC_FULL_CUDA_CENSUS_OUTPUT_DIR=", output_dir),
  "FASTKPC_FULL_CUDA_CENSUS_MODE=metadata",
  "FASTKPC_FULL_CUDA_CENSUS_MAX_KEYS=8",
  "FASTKPC_FULL_CUDA_CENSUS_WORKERS=1",
  "FASTKPC_FULL_CUDA_CENSUS_SHARD_COUNT=2",
  "FASTKPC_FULL_CUDA_CENSUS_RESUME=1"
)

run_runner <- function(environment = runner_env) {
  output <- system2(
    "Rscript", runner, env = environment,
    stdout = TRUE, stderr = TRUE
  )
  status <- attr(output, "status")
  if (is.null(status)) status <- 0L
  assert_true(status == 0L,
              paste(c("workload census runner failed", output),
                    collapse = "\n"))
  output
}

invisible(run_runner())
paths <- fastkpc_full_cuda_census_artifact_paths(output_dir)
required_paths <- unlist(paths[setdiff(names(paths), "shards_dir")],
                         use.names = FALSE)
assert_true(all(file.exists(required_paths)) && dir.exists(paths$shards_dir),
            "metadata runner must write every standard Phase 1 artifact")

summary <- jsonlite::read_json(paths$summary_json, simplifyVector = TRUE)
manifest <- jsonlite::read_json(paths$manifest_json, simplifyVector = TRUE)
target_fits <- readRDS(paths$target_fit_metadata_rds)
same_s_setups <- readRDS(paths$same_s_setup_metadata_rds)
requests <- readRDS(paths$residual_requests_rds)
parity <- utils::read.csv(paths$legacy_layout_parity_results_csv,
                          stringsAsFactors = FALSE)

expected_keys <- head(
  sort(requests$residual_key_sha256, method = "radix"), 8L
)
assert_true(isTRUE(summary$pass) &&
              identical(summary$run_scope, "scaled_prefix") &&
              identical(summary$phase1_complete, FALSE) &&
              summary$selected_key_count == 8L &&
              summary$canonical_key_count == 110617L,
            "scaled metadata summary must pass without claiming Phase 1 closure")
assert_true(summary$logical_test_count == 240489L &&
              summary$conditional_logical_test_count == 238276L &&
              summary$conditional_residual_request_count == 476552L &&
              summary$canonical_global_unique_conditional_target_s_count ==
                110617L &&
              summary$unique_conditional_S_count == 8634L &&
              summary$same_s_setup_metadata_rows == nrow(same_s_setups) &&
              summary$target_fit_metadata_rows == 8L &&
              summary$setup_observation_metadata_rows == 8L &&
              summary$target_risk_metadata_rows == 8L &&
              summary$mgcv_fit_error_count == 0L &&
              summary$same_s_invariant_violation_count == 0L &&
              summary$required_field_coverage == 1 &&
              isTRUE(summary$exact_target_request_lineage) &&
              isTRUE(summary$exact_target_risk_key_set) &&
              isTRUE(summary$exact_target_risk_lineage) &&
              isTRUE(summary$exact_setup_observation_key_set) &&
              isTRUE(summary$exact_target_setup_lineage) &&
              isTRUE(summary$exact_warning_classification) &&
              isTRUE(summary$exact_nonfinite_classification) &&
              summary$misclassified_warning_count == 0L &&
              summary$misclassified_nonfinite_count == 0L &&
              isTRUE(summary$legacy_layout_parity_pass) &&
              identical(summary$canonical_key_corpus_hash,
                        paste0(
                          "b843630969f116da63f7fad095c54de2ff471540159ff97ca5",
                          "6c3871d6b2e1fa"
                        )),
            "summary must expose the exact canonical Phase 1 closure schema")
assert_true(identical(manifest$metadata_schema_version,
                      "full-cuda-ci-metadata-v2"),
            "artifact must record the lineage-hardened metadata schema")
assert_true(all(c(
  "counts_by_s_size", "counts_by_penalty_count",
  "counts_by_model_dimension", "counts_by_condition_bucket",
  "same_s_group_size_distribution", "near_alpha_bucket_counts",
  "fit_time_by_s_size", "fit_time_by_penalty_count"
) %in% names(summary)),
"summary must embed every approved structural and fit-time distribution")
assert_true(nrow(target_fits) == 8L &&
              identical(target_fits$residual_key_sha256, expected_keys) &&
              nrow(same_s_setups) > 0L,
            "metadata runner must fit the fixed first eight SHA keys")
assert_true(nrow(parity) == 7L && all(parity$pass),
            "metadata mode must carry passing legacy-layout parity evidence")
assert_true(isTRUE(summary$oracle_inherited_graph_gate) &&
              identical(summary$new_candidate_graph_gate,
                        "NOT_APPLICABLE") &&
              isTRUE(manifest$oracle_inherited_graph_gate) &&
              identical(manifest$new_candidate_graph_gate,
                        "NOT_APPLICABLE"),
            "runner must preserve inherited Phase 0 gate scope")
assert_true(manifest$requested_workers == 1L &&
              manifest$actual_workers == 1L &&
              manifest$shard_count == 2L &&
              summary$executed_key_count == 8L &&
              summary$written_shard_count == 2L &&
              summary$reused_shard_count == 0L,
            "manifest must record requested and actual execution shape")

merged_rds <- c(
  paths$same_s_setup_metadata_rds,
  paths$target_fit_metadata_rds,
  paths$risk_cases_rds
)
hash_before <- vapply(
  merged_rds, fastkpc_full_cuda_census_file_hash, character(1L)
)
invisible(run_runner())
hash_after <- vapply(
  merged_rds, fastkpc_full_cuda_census_file_hash, character(1L)
)
assert_true(identical(hash_before, hash_after),
            "resume must produce byte-identical merged RDS artifacts")
resumed_summary <- jsonlite::read_json(paths$summary_json,
                                       simplifyVector = TRUE)
assert_true(resumed_summary$executed_key_count == 0L &&
              resumed_summary$written_shard_count == 0L &&
              resumed_summary$reused_shard_count == 2L &&
              identical(resumed_summary$keys_per_sec,
                        summary$keys_per_sec) &&
              identical(resumed_summary$estimated_full_elapsed_sec,
                        summary$estimated_full_elapsed_sec),
            "pure resume must preserve first-run performance estimates")

mode_environment <- function(selected_mode, selected_output) {
  value <- runner_env
  value[grepl("^FASTKPC_FULL_CUDA_CENSUS_MODE=", value)] <-
    paste0("FASTKPC_FULL_CUDA_CENSUS_MODE=", selected_mode)
  value[grepl("^FASTKPC_FULL_CUDA_CENSUS_OUTPUT_DIR=", value)] <-
    paste0("FASTKPC_FULL_CUDA_CENSUS_OUTPUT_DIR=", selected_output)
  value
}

structural_dir <- tempfile("full-cuda-ci-census-structural-")
invisible(run_runner(mode_environment("structural", structural_dir)))
structural_paths <- fastkpc_full_cuda_census_artifact_paths(structural_dir)
structural_summary <- jsonlite::read_json(
  structural_paths$summary_json, simplifyVector = TRUE
)
assert_true(isTRUE(structural_summary$pass) &&
              identical(structural_summary$phase1_complete, FALSE) &&
              file.exists(structural_paths$logical_tests_rds) &&
              file.exists(structural_paths$residual_requests_rds),
            "structural mode must write canonical tables without Phase 1 closure")

parity_dir <- tempfile("full-cuda-ci-census-parity-")
invisible(run_runner(mode_environment("parity", parity_dir)))
parity_paths <- fastkpc_full_cuda_census_artifact_paths(parity_dir)
parity_summary <- jsonlite::read_json(
  parity_paths$summary_json, simplifyVector = TRUE
)
parity_only <- utils::read.csv(
  parity_paths$legacy_layout_parity_results_csv,
  stringsAsFactors = FALSE
)
assert_true(isTRUE(parity_summary$pass) &&
              identical(parity_summary$phase1_complete, FALSE) &&
              nrow(parity_only) == 7L && all(parity_only$pass),
            "parity mode must write exact evidence without Phase 1 closure")

cat("PASS full CUDA CI census real subset\n")
