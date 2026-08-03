source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase10_coalescing_opportunity.R")

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) > 2L) {
  stop(
    paste(
      "usage: profile_full_cuda_ci_phase10_coalescing_opportunity.R",
      "[V5_EVIDENCE_RDS] [OUTPUT_RDS]"
    ),
    call. = FALSE
  )
}
evidence_path <- if (length(arguments) >= 1L) arguments[[1L]] else
  fastkpc_full_cuda_phase10_coalescing_default_evidence()
output_path <- if (length(arguments) >= 2L) arguments[[2L]] else
  fastkpc_full_cuda_phase10_coalescing_default_output()

evidence <- readRDS(evidence_path)
opportunity <- fastkpc_full_cuda_phase10_coalescing_opportunity(evidence)
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
saveRDS(opportunity, output_path, compress = "xz")
fastkpc_full_cuda_phase10_validate_coalescing_opportunity(
  readRDS(output_path), evidence$result$summary
)

cat(
  "PASS Phase 10 setup-aware coalescing opportunity; decision=",
  opportunity$decision, " output=", output_path, "\n", sep = ""
)
print(opportunity$totals, row.names = FALSE)
print(opportunity$by_level, row.names = FALSE)
print(opportunity$level3_checkpoint, row.names = FALSE)
print(opportunity$frontier_density, row.names = FALSE)
