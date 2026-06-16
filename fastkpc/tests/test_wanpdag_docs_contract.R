readme <- readLines("fastkpc/README.md", warn = FALSE)
text <- paste(readme, collapse = "\n")

assert_contains <- function(pattern) {
  if (!grepl(pattern, text, fixed = TRUE)) {
    stop("README missing required text: ", pattern, call. = FALSE)
  }
}

required <- c(
  "WAN-PDAG Orientation Scope",
  "WAN-PDAG API",
  "WAN-PDAG Validation",
  "WAN-PDAG Benchmark",
  "WAN-PDAG Known Limits",
  "fast_kpc_wanpdag_cpp",
  "fast_kpc_wanpdag_cuda",
  "kpcalg::kpc() is not replaced",
  "kpcalg/R/*.R files are not modified"
)

for (pattern in required) assert_contains(pattern)

cat("test_wanpdag_docs_contract.R: PASS\n")
