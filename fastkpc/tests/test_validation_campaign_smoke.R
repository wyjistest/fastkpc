source("fastkpc/R/validation_campaign.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

campaign <- run_fastkpc_validation_campaign(
  seeds = c(21, 22),
  n_values = c(70),
  scenarios = c("chain", "independent"),
  engines = c("cpu", "cuda"),
  residual_backends = c("linear", "fastSpline"),
  alpha = 0.2,
  max_conditioning_size = 1L,
  legacy = TRUE,
  benchmark = TRUE,
  output_dir = NULL
)

required <- c("config", "runs", "graph_metrics", "pairwise_diffs",
              "cpu_cuda", "linear_fastspline", "legacy", "timings", "cache",
              "orientation_counts", "errors", "artifacts", "summary")
missing <- setdiff(required, names(campaign))
assert_true(length(missing) == 0L,
            paste("campaign missing fields:", paste(missing, collapse = ", ")))

assert_true(is.data.frame(campaign$runs), "runs should be data.frame")
assert_true(nrow(campaign$runs) == 2L * 1L * 2L * 2L * 2L,
            "runs row count should equal seeds*n_values*scenarios*engines*backends")
assert_true(all(campaign$runs$status == "ok"), "all smoke campaign runs should be ok")
assert_true(is.data.frame(campaign$cpu_cuda), "cpu_cuda should be data.frame")
assert_true(all(campaign$cpu_cuda$max_abs_pmax_diff < 1e-8, na.rm = TRUE),
            "CPU-vs-CUDA pMax diff should be tiny")
assert_true(any(campaign$cpu_cuda$pdag_identical), "at least one CPU-vs-CUDA pdag should match")
assert_true(is.data.frame(campaign$legacy), "legacy should be data.frame")
if (!all(campaign$legacy$available)) {
  assert_true(any(grepl("pcalg|graph", campaign$legacy$reason_if_unavailable)),
              "legacy unavailable rows should mention pcalg/graph")
}
assert_true(is.list(campaign$summary), "summary should exist")
assert_true(campaign$summary$total_runs == nrow(campaign$runs),
            "summary total_runs should match runs")

cat("test_validation_campaign_smoke.R: PASS\n")
