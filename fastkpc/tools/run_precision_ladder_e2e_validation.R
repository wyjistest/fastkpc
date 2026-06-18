source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/precision_execution_trace.R")
source("fastkpc/R/workload_structure_stats.R")
source("fastkpc/R/hybrid_heldout_validation.R")
source("fastkpc/R/true_batched_kernel_decision.R")

output_dir <- Sys.getenv(
  "FASTKPC_PRECISION_E2E_DIR",
  file.path("fastkpc", "artifacts", "precision_ladder_e2e")
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

set.seed(501)
data <- matrix(rnorm(80 * 6), 80, 6)
fast <- fast_kpc(data, alpha = 0.2, max_conditioning_size = 1,
                 precision = "fast", engine = "cpu", graph_stage = "skeleton")
compatible <- fast_kpc(data, alpha = 0.2, max_conditioning_size = 1,
                       precision = "compatible", engine = "cpu",
                       graph_stage = "skeleton")
hybrid <- fast_kpc(data, alpha = 0.2, max_conditioning_size = 1,
                   precision = "hybrid", engine = "cpu",
                   graph_stage = "skeleton")

trace <- do.call(rbind, list(
  fast$diagnostics$precision_trace,
  compatible$diagnostics$precision_trace,
  hybrid$diagnostics$precision_trace
))
utils::write.csv(trace, file.path(output_dir, "precision_execution_trace.csv"),
                 row.names = FALSE)

summary <- data.frame(
  precision = c("fast", "compatible", "hybrid"),
  backend_used = c(fast$config$backend_used,
                   compatible$config$backend_used,
                   hybrid$config$backend_used),
  compatibility_action = c(fast$config$compatibility_action,
                           compatible$config$compatibility_action,
                           hybrid$config$compatibility_action),
  fallback_reason = c(fast$config$fallback_reason,
                      compatible$config$fallback_reason,
                      hybrid$config$fallback_reason),
  skeleton_edges = c(fast$metrics$skeleton_edge_count,
                     compatible$metrics$skeleton_edge_count,
                     hybrid$metrics$skeleton_edge_count)
)
utils::write.csv(summary, file.path(output_dir, "precision_e2e_summary.csv"),
                 row.names = FALSE)
cat("precision ladder e2e artifacts:", output_dir, "\n")
