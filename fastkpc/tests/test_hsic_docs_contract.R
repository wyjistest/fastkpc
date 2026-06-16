readme <- paste(readLines("fastkpc/README.md", warn = FALSE), collapse = "\n")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

assert_true(grepl("ci_method = \"hsic.gamma\"", readme, fixed = TRUE),
            "README should document hsic.gamma")
assert_true(grepl("ci_diagnostics", readme, fixed = TRUE),
            "README should document ci_diagnostics")
assert_true(grepl("ci_method = \"hsic.perm\"", readme, fixed = TRUE),
            "README should document hsic.perm")
assert_true(grepl("ci_backend = \"cuda-hsic\"", readme, fixed = TRUE),
            "README should document CUDA HSIC backend")
assert_true(grepl("native-cpu fallback", readme, fixed = TRUE),
            "README should document CUDA HSIC CPU fallback")
assert_true(grepl("ci_method_diagnostics.csv", readme, fixed = TRUE),
            "README should document CI method diagnostics artifact")
assert_true(!grepl("- HSIC or permutation tests", readme, fixed = TRUE),
            "README should not list HSIC/permutation tests as unimplemented")

reports_readme <- paste(readLines("fastkpc/reports/README.md", warn = FALSE),
                        collapse = "\n")
assert_true(grepl("ci_method_diffs.csv", reports_readme, fixed = TRUE),
            "reports README should document ci_method_diffs.csv")
assert_true(grepl("ci_method_diagnostics.csv", reports_readme, fixed = TRUE),
            "reports README should document ci_method_diagnostics.csv")

cat("test_hsic_docs_contract.R: PASS\n")
