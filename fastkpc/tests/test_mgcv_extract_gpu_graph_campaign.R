source("fastkpc/R/mgcv_extract_gpu_graph_campaign.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

out_dir <- tempfile("fastkpc-mgcv-extract-gpu-graph-")
campaign <- fastkpc_run_mgcv_extract_gpu_graph_campaign(
  output_dir = out_dir,
  seeds = 21L,
  n_values = 120L,
  p_values = 8L,
  alpha_values = 0.05,
  tau_values = log(c(2, 3)),
  max_conditioning_levels = c(1L, 2L)
)

graph_file <- file.path(out_dir, "mgcv_extract_gpu_graph_comparison.csv")
diagnostics_file <- file.path(out_dir, "mgcv_extract_gpu_hybrid_diagnostics.csv")
summary_file <- file.path(out_dir, "mgcv_extract_gpu_graph_summary.csv")

assert_true(file.exists(graph_file), "graph comparison CSV should exist")
assert_true(file.exists(diagnostics_file), "hybrid diagnostics CSV should exist")
assert_true(file.exists(summary_file), "graph summary CSV should exist")

required_graph <- c(
  "scenario_id", "seed", "n", "p", "alpha", "tau",
  "max_conditioning_level", "backend", "backend_role",
  "skeleton_shd", "skeleton_precision", "skeleton_recall",
  "skeleton_f1", "edge_deletion_mismatch",
  "first_separating_set_mismatch", "sepset_mismatch_rate",
  "wanpdag_orientation_mismatch", "arrowhead_agreement",
  "near_alpha_verifier_calls", "verifier_induced_decision_changes",
  "runtime_sec"
)
missing_graph <- setdiff(required_graph, names(campaign$graph))
assert_true(length(missing_graph) == 0L,
            paste("missing graph fields:", paste(missing_graph, collapse = ", ")))

expected_backends <- c(
  "legacy-mgcv",
  "fastSplineCUDA",
  "mgcvExtractGPUFixedSP",
  "mgcvExtractGPUGCV",
  "hybrid-fastSplineCUDA-mgcvExtractGPU"
)
assert_true(all(expected_backends %in% unique(campaign$graph$backend)),
            "graph comparison should include the precision ladder backends")

hybrid_rows <- campaign$graph[
  campaign$graph$backend == "hybrid-fastSplineCUDA-mgcvExtractGPU",
  , drop = FALSE
]
primary_rows <- campaign$graph[campaign$graph$backend == "fastSplineCUDA",
                               , drop = FALSE]
assert_true(nrow(hybrid_rows) == nrow(primary_rows),
            "hybrid and primary rows should cover the same scenarios")
assert_true(all(hybrid_rows$skeleton_shd <= primary_rows$skeleton_shd),
            "hybrid verifier should not increase skeleton SHD in deterministic campaign")
assert_true(any(hybrid_rows$skeleton_shd < primary_rows$skeleton_shd),
            "hybrid verifier should improve skeleton SHD for at least one scenario")

required_diag <- c(
  "scenario_id", "seed", "n", "p", "alpha", "tau",
  "max_conditioning_level", "canonical_test_order_id",
  "x", "y", "S_key", "primary_p", "verifier_p", "p_used",
  "p_source_used", "near_alpha_triggered",
  "decision_before_verify", "decision_after_verify",
  "verifier_backend", "verification_reason"
)
missing_diag <- setdiff(required_diag, names(campaign$diagnostics))
assert_true(length(missing_diag) == 0L,
            paste("missing diagnostics fields:", paste(missing_diag, collapse = ", ")))

diagnostics <- campaign$diagnostics
split_key <- paste(diagnostics$scenario_id, diagnostics$seed, diagnostics$n,
                   diagnostics$p, diagnostics$alpha, diagnostics$tau,
                   diagnostics$max_conditioning_level, sep = "::")
for (key in unique(split_key)) {
  ids <- diagnostics$canonical_test_order_id[split_key == key]
  assert_true(identical(ids, sort(ids)),
              "hybrid diagnostics must be written in canonical order")
}

verified <- diagnostics$p_source_used == "mgcvExtractGPU"
assert_true(any(verified), "campaign should trigger mgcvExtractGPU verification")
assert_true(all(diagnostics$p_used[verified] == diagnostics$verifier_p[verified]),
            "verified rows should use verifier p-values")
assert_true(all(diagnostics$p_used[!verified] == diagnostics$primary_p[!verified]),
            "unverified rows should keep primary p-values")

required_summary <- c(
  "scenario_id", "seed", "n", "p", "alpha",
  "max_conditioning_level", "recommended_tau",
  "primary_skeleton_shd", "hybrid_skeleton_shd",
  "skeleton_shd_reduction", "verifier_call_rate",
  "hybrid_runtime_sec", "legacy_runtime_sec", "speedup_vs_legacy"
)
missing_summary <- setdiff(required_summary, names(campaign$summary))
assert_true(length(missing_summary) == 0L,
            paste("missing summary fields:", paste(missing_summary, collapse = ", ")))
assert_true(all(campaign$summary$skeleton_shd_reduction >= 0),
            "recommended hybrid rows should control graph drift")

cat("PASS mgcvExtractGPU graph campaign\n")
