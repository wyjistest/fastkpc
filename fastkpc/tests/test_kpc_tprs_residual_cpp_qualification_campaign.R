source("fastkpc/R/kpc_tprs_residual_cpp_qualification.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_has_names <- function(x, required, message) {
  missing <- setdiff(required, names(x))
  if (length(missing) > 0L) {
    fail(paste0(message, ": missing ", paste(missing, collapse = ", ")))
  }
}

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP kpcTprsResidualCPP qualification campaign: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

output_dir <- tempfile("kpc-tprs-qualification-")
real_path <- tempfile(fileext = ".csv")
set.seed(62401)
real_s <- stats::runif(58, -2, 2)
utils::write.csv(
  data.frame(
    r1 = sin(real_s) + stats::rnorm(58, sd = 0.05),
    r2 = cos(real_s) + stats::rnorm(58, sd = 0.05),
    r3 = real_s + stats::rnorm(58, sd = 0.03),
    r4 = 0.35 * sin(real_s) + stats::rnorm(58, sd = 0.06)
  ),
  real_path,
  row.names = FALSE
)

campaign <- fastkpc_run_kpc_tprs_residual_cpp_qualification(
  output_dir = output_dir,
  repeats = 1L,
  alpha = 0.05,
  max_conditioning_size = 2L,
  real_data_path = real_path,
  no_oracle_check = TRUE
)

assert_has_names(
  campaign,
  c("runs", "graph_agreement", "trace_summary", "qualification_summary",
    "no_oracle", "summary", "paths", "output_dir"),
  "qualification campaign"
)
assert_true(is.data.frame(campaign$runs), "runs should be a data frame")
assert_true(is.data.frame(campaign$graph_agreement),
            "graph agreement should be a data frame")
assert_true(is.data.frame(campaign$trace_summary),
            "trace summary should be a data frame")
assert_true(is.data.frame(campaign$qualification_summary),
            "qualification summary should be a data frame")
assert_true(is.data.frame(campaign$no_oracle),
            "no-oracle summary should be a data frame")

required_run_cols <- c("scenario_id", "repeat", "mode", "status",
                       "wall_time_sec", "backend_planned",
                       "backend_executed", "conditional_tests")
assert_true(all(required_run_cols %in% names(campaign$runs)),
            "runs should include mode/backend/timing fields")
assert_true(all(c("reference_mgcv", "candidate_kpc") %in%
                  unique(campaign$runs$mode)),
            "campaign should run reference and candidate modes")
assert_true(any(grepl("^real-", campaign$runs$scenario_id)),
            "campaign should include real-data scenario when provided")
assert_true(all(campaign$runs$status == "ok"),
            paste(campaign$runs$error_message[campaign$runs$status != "ok"],
                  collapse = "; "))

assert_true(all(c("scenario_id", "repeat", "adjacency_identical",
                  "n_edgetests_identical", "pmax_max_abs_diff",
                  "first_sepset_mismatch_rate", "all_sepset_mismatch_rate",
                  "passed") %in% names(campaign$graph_agreement)),
            "graph agreement should expose graph and sepset fields")
assert_true(all(campaign$graph_agreement$passed),
            paste(campaign$graph_agreement$scenario_id[
              !campaign$graph_agreement$passed], collapse = ", "))
assert_true(max(campaign$graph_agreement$pmax_max_abs_diff, na.rm = TRUE) < 1e-4,
            "pMax drift should remain bounded")

candidate_trace <- campaign$trace_summary[
  campaign$trace_summary$mode == "candidate_kpc", , drop = FALSE]
assert_true(nrow(candidate_trace) > 0L, "candidate trace rows should exist")
assert_true(sum(candidate_trace$kpc_rows, na.rm = TRUE) > 0L,
            "candidate should execute kpcTprsResidualCPP rows")
assert_true(sum(candidate_trace$two_d_kpc_rows, na.rm = TRUE) > 0L,
            "candidate should exercise |S|=2 kpcTprsResidualCPP rows")
assert_true(sum(candidate_trace$fallback_rows, na.rm = TRUE) == 0L,
            "candidate stress campaign should not need fallback")

assert_true(nrow(campaign$no_oracle) == 1L,
            "no-oracle summary should contain one row")
assert_true(isTRUE(campaign$no_oracle$passed[[1L]]),
            campaign$no_oracle$failure_reason[[1L]])
assert_true(campaign$no_oracle$forbidden_calls[[1L]] == 0L,
            "no-oracle guard should record zero forbidden mgcv calls")

assert_true(nrow(campaign$qualification_summary) == 1L,
            "qualification summary should contain one row")
assert_true(isTRUE(campaign$qualification_summary$passed[[1L]]),
            campaign$qualification_summary$failure_reason[[1L]])

for (path in unlist(campaign$paths, use.names = FALSE)) {
  assert_true(file.exists(path), paste("missing artifact:", path))
}
assert_true(file.exists(file.path(output_dir, "summary.md")),
            "summary markdown should be written")

cat("PASS kpcTprsResidualCPP qualification campaign\n")
