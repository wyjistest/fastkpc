source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_shadow.R")
source("fastkpc/R/full_cuda_ci_phase3_artifacts.R")
source("fastkpc/R/full_cuda_ci_phase35_contracts.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_single_penalty_gcv.R")
source("fastkpc/R/full_cuda_ci_phase4_artifacts.R")
source("fastkpc/R/full_cuda_ci_phase4_backend.R")
source("fastkpc/R/full_cuda_ci_phase4_publication.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cpp.R")
source("fastkpc/R/full_cuda_ci_phase5_artifacts.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cuda.R")
source("fastkpc/R/full_cuda_ci_phase6_artifacts.R")

partition_dir <- Sys.getenv("FASTKPC_PHASE6_PARTITION_DIR", unset = "")
output_path <- Sys.getenv("FASTKPC_PHASE6_MERGED_OUTPUT", unset = "")
phase5_evidence_path <- Sys.getenv(
  "FASTKPC_PHASE6_PHASE5_EVIDENCE",
  unset = file.path(
    "fastkpc", "artifacts", "full_cuda_ci",
    "multi_penalty_cpp_full_shadow_v1", "source_evidence.rds"
  )
)
if (!nzchar(partition_dir) || !dir.exists(partition_dir)) {
  stop("FASTKPC_PHASE6_PARTITION_DIR must be an existing directory",
       call. = FALSE)
}
if (!nzchar(output_path)) {
  stop("FASTKPC_PHASE6_MERGED_OUTPUT is required", call. = FALSE)
}
if (!file.exists(phase5_evidence_path)) {
  stop("FASTKPC_PHASE6_PHASE5_EVIDENCE is missing", call. = FALSE)
}
paths <- list.files(
  partition_dir, pattern = "^partition_[0-9]+\\.rds$", full.names = TRUE
)
paths <- sort(paths, method = "radix")
if (length(paths) == 0L) {
  stop("Phase 6 partition directory is empty", call. = FALSE)
}

catalog <- fastkpc_full_cuda_open_fixed_sp_catalog(
  file.path("fastkpc", "artifacts", "full_cuda_ci", "oracle_351x48_v1"),
  file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "workload_census_351x48_v1"
  ),
  file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1"
  ),
  file.path(
    "fastkpc", "artifacts", "kpc_tprs_real_zhu",
    "cancer_RD-causalDiscoveryInput.rds"
  ),
  require_full = TRUE
)
merged <- fastkpc_full_cuda_phase6_merge_partitions(
  catalog, paths, phase5_evidence_path = phase5_evidence_path
)
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
temporary <- paste0(output_path, ".tmp-", Sys.getpid())
on.exit(unlink(temporary, force = TRUE), add = TRUE)
saveRDS(merged, temporary, compress = "gzip", version = 3L)
if (!file.rename(temporary, output_path)) {
  stop("Phase 6 merged evidence publication rename failed", call. = FALSE)
}
cat(
  "PASS Phase 6 merged shadow setups=", merged$summary$setup_count,
  " targets=", merged$summary$target_count,
  " logical=", merged$summary$logical_test_count,
  " concurrency=", merged$configured_concurrency,
  " SHD=", merged$mixed_graph$summary$SHD,
  " fallback=", merged$mixed_graph$summary$residual_numerical_fallback_count,
  " output=", output_path, "\n", sep = ""
)
