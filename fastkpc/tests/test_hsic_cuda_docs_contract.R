readme <- paste(readLines("fastkpc/README.md", warn = FALSE), collapse = "\n")
reports_readme <- paste(readLines("fastkpc/reports/README.md", warn = FALSE),
                        collapse = "\n")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

assert_true(grepl("CUDA HSIC kernels", readme, fixed = TRUE),
            "README should mention CUDA HSIC kernels")
assert_true(grepl("cuda-hsic", readme, fixed = TRUE),
            "README should mention cuda-hsic")
assert_true(grepl("native-cpu fallback", readme, fixed = TRUE),
            "README should mention native-cpu fallback")
assert_true(grepl("explicit permutation seed", readme, fixed = TRUE),
            "README should mention explicit permutation seed")
assert_true(grepl("CUDA HSIC permutation requires explicit seed", readme, fixed = TRUE),
            "README should mention seed requirement")
assert_true(grepl("ci_backend", readme, fixed = TRUE),
            "README should mention ci_backend")
assert_true(grepl("cuda_hsic_used", readme, fixed = TRUE),
            "README should mention cuda_hsic_used")
assert_true(grepl("kpcalg/R/*.R files are not modified", readme, fixed = TRUE),
            "README should mention kpcalg/R is unchanged")

assert_true(grepl("hsic_cuda_backend_diagnostics.csv", reports_readme, fixed = TRUE),
            "reports README should mention hsic_cuda_backend_diagnostics.csv")
assert_true(grepl("hsic_cuda_cpu_fallbacks.csv", reports_readme, fixed = TRUE),
            "reports README should mention hsic_cuda_cpu_fallbacks.csv")
assert_true(grepl("hsic_cuda_perf.csv", reports_readme, fixed = TRUE),
            "reports README should mention hsic_cuda_perf.csv")

cat("test_hsic_cuda_docs_contract.R: PASS\n")
