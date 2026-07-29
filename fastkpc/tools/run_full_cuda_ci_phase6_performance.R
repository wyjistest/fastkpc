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
source("fastkpc/R/full_cuda_ci_phase6_performance.R")

merged_path <- Sys.getenv("FASTKPC_PHASE6_MERGED_EVIDENCE", unset = "")
output_path <- Sys.getenv("FASTKPC_PHASE6_PERFORMANCE_OUTPUT", unset = "")
if (!nzchar(merged_path) || !file.exists(merged_path)) {
  stop("FASTKPC_PHASE6_MERGED_EVIDENCE must name the merged RDS",
       call. = FALSE)
}
if (!nzchar(output_path)) {
  stop("FASTKPC_PHASE6_PERFORMANCE_OUTPUT is required", call. = FALSE)
}
merged <- readRDS(merged_path)
evidence <- fastkpc_full_cuda_phase6_build_performance_evidence(
  merged, merged_evidence_path = merged_path
)
fastkpc_full_cuda_phase6_validate_performance_evidence(
  evidence, merged, merged_evidence_path = merged_path,
  verify_current_inputs = TRUE
)
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
temporary <- paste0(output_path, ".tmp-", Sys.getpid())
on.exit(unlink(temporary, force = TRUE), add = TRUE)
saveRDS(evidence, temporary, compress = "gzip", version = 3L)
if (!file.rename(temporary, output_path)) {
  stop("Phase 6 performance evidence publication rename failed",
       call. = FALSE)
}
cat(
  "PASS Phase 6 full residual performance candidate_ms=",
  format(evidence$summary$candidate_residual_wall_ms, digits = 8L),
  " baseline_ms=",
  format(evidence$summary$baseline_residual_wall_ms, digits = 8L),
  " ratio=",
  format(evidence$summary$candidate_to_baseline_ratio, digits = 8L),
  " output=", output_path, "\n", sep = ""
)
