source("fastkpc/R/mgcv_residual_oracle_trace.R")
source("fastkpc/R/mgcv_residual_replay_spec.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

missing <- c("mgcv", "RSpectra")[
  !vapply(c("mgcv", "RSpectra"), requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP mgcv residual replay spec: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

set.seed(93477)
n <- 70L
z1 <- stats::runif(n, -2, 2)
z2 <- stats::rnorm(n)
z3 <- stats::runif(n, -1, 1)
data <- cbind(
  x1 = sin(z1) + stats::rnorm(n, sd = 0.06),
  x2 = cos(z1) + stats::rnorm(n, sd = 0.06),
  x3 = z1 + stats::rnorm(n, sd = 0.04),
  x4 = z2 + 0.2 * z1,
  x5 = z3 + stats::rnorm(n, sd = 0.08),
  x6 = z1 * z3 + stats::rnorm(n, sd = 0.12)
)
cases <- data.frame(
  case_id = c("replay_s1", "replay_s2", "replay_gt2"),
  x = c(1L, 1L, 2L),
  y = c(2L, 6L, 5L),
  S = c("3", "3,5", "3,4,5"),
  role = c("|S|=1", "|S|=2", "|S|>2"),
  stringsAsFactors = FALSE
)
oracle_dir <- tempfile("fastkpc-mgcv-oracle-for-replay-")
oracle_artifact <- fastkpc_run_mgcv_residual_oracle_trace(
  data = data,
  cases = cases,
  alpha = 0.1,
  output_dir = oracle_dir
)
invisible(oracle_artifact)

out_dir <- tempfile("fastkpc-mgcv-replay-spec-")
artifact <- fastkpc_run_mgcv_residual_replay_spec(
  data = data,
  oracle_dir = oracle_dir,
  output_dir = out_dir,
  alpha = 0.1
)

assert_true(file.exists(file.path(out_dir, "summary.csv")),
            "replay summary.csv should be written")
assert_true(file.exists(file.path(out_dir, "cases.csv")),
            "replay cases.csv should be written")
assert_true(file.exists(file.path(out_dir, "summary.md")),
            "replay summary.md should be written")

required <- c(
  "case_id", "S_size", "formula_route", "residual_x_match",
  "residual_y_match", "residual_pair_match", "dcov_p_oracle",
  "dcov_p_replay", "dcov_p_abs_diff", "decision_oracle",
  "decision_replay", "decision_match",
  "dcov_alpha_oracle", "dcov_alpha_replay",
  "dcov_log_alpha_distance_oracle", "dcov_log_alpha_distance_replay",
  "conditioning_rank_oracle", "conditioning_rank_replay",
  "conditioning_rank_deficient_oracle",
  "conditioning_rank_deficient_replay",
  "near_constant_column_count_oracle",
  "near_constant_column_count_replay",
  "target_near_constant_count_oracle",
  "target_near_constant_count_replay",
  "lpmatrix_rank_x_oracle", "lpmatrix_rank_x_replay",
  "lpmatrix_rank_y_oracle", "lpmatrix_rank_y_replay",
  "smooth_labels_x_oracle", "smooth_labels_x_replay",
  "replay_status"
)
missing_fields <- setdiff(required, names(artifact$cases))
assert_true(length(missing_fields) == 0L,
            paste("replay cases missing", missing_fields[[1L]]))

assert_true(nrow(artifact$cases) == nrow(cases),
            "replay spec should evaluate every oracle case")
assert_true(all(artifact$cases$residual_pair_match),
            "replay spec should exactly match oracle residual hashes")
assert_true(all(artifact$cases$decision_match),
            "replay spec should not flip oracle decisions")
assert_true(max(artifact$cases$dcov_p_abs_diff) <= 1e-12,
            "replay spec should reproduce oracle dCov p-values")
assert_true(all(artifact$cases$dcov_alpha_oracle == 0.1),
            "replay spec should preserve oracle alpha")
assert_true(all(artifact$cases$dcov_alpha_replay == 0.1),
            "replay spec should record replay alpha")
assert_true(all(is.finite(artifact$cases$dcov_log_alpha_distance_oracle)),
            "replay spec should preserve oracle log-alpha distance")
assert_true(all(artifact$cases$conditioning_rank_oracle ==
                  artifact$cases$conditioning_rank_replay),
            "replay spec should preserve conditioning rank diagnostics")
assert_true(all(artifact$cases$lpmatrix_rank_x_replay > 0L),
            "replay spec should record replay lpmatrix rank diagnostics")

summary <- artifact$summary
required_summary <- c(
  "rank_deficient_case_count",
  "near_constant_case_count",
  "min_dcov_log_alpha_distance_oracle",
  "min_dcov_log_alpha_distance_replay"
)
missing_summary <- setdiff(required_summary, names(summary))
assert_true(length(missing_summary) == 0L,
            paste("replay summary missing", missing_summary[[1L]]))
assert_true(summary$case_count[[1L]] == nrow(cases),
            "replay summary should count cases")
assert_true(summary$residual_pair_match_count[[1L]] == nrow(cases),
            "replay summary should count residual matches")
assert_true(summary$decision_flip_count[[1L]] == 0L,
            "replay summary should report no decision flips")
assert_true(summary$pass[[1L]],
            "replay summary should pass when residual and decisions match")
assert_true(summary$min_dcov_log_alpha_distance_oracle[[1L]] >= 0,
            "replay summary should record oracle alpha-distance risk")

cat("PASS mgcv residual replay spec\n")
