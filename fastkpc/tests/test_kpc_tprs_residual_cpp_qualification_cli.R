assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP kpcTprsResidualCPP qualification CLI: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

output_dir <- tempfile("kpc-tprs-qualification-cli-")
status <- system2(
  "Rscript",
  c("fastkpc/tools/run_kpc_tprs_residual_cpp_qualification.R", output_dir),
  env = c("FASTKPC_KPC_TPRS_REPEATS=1",
          "FASTKPC_KPC_TPRS_NO_ORACLE=TRUE")
)
assert_true(identical(status, 0L),
            "qualification CLI should exit 0")

required <- c("runs.csv", "graph_agreement.csv", "trace_summary.csv",
              "backend_comparison.csv", "pvalue_drift.csv",
              "qualification_summary.csv", "promotion_summary.csv",
              "no_oracle.csv", "summary.md")
for (name in required) {
  assert_true(file.exists(file.path(output_dir, name)),
              paste("missing artifact:", name))
}
summary <- utils::read.csv(file.path(output_dir, "qualification_summary.csv"))
assert_true(nrow(summary) == 1L, "summary should have one row")
assert_true(isTRUE(summary$passed[[1L]]),
            summary$failure_reason[[1L]])

cat("PASS kpcTprsResidualCPP qualification CLI\n")
