text <- paste(readLines("fastkpc/README.md", warn = FALSE), collapse = "\n")

fail <- function(message) stop(message, call. = FALSE)
assert_grepl <- function(pattern, message) {
  if (!grepl(pattern, text, fixed = TRUE)) fail(message)
}

assert_grepl("fastSpline CUDA is a high-throughput approximate backend",
             "README must describe fastSpline CUDA as approximate")
assert_grepl("mgcvExtractCPU is a version-pinned extraction oracle",
             "README must describe mgcvExtractCPU as oracle")
assert_grepl("No full mgcv clone",
             "README must state full mgcv clone non-goal")
assert_grepl("No bamGPU",
             "README must state bamGPU non-goal")
assert_grepl("default s(s1, s2) is not a tensor-product smooth",
             "README must state smooth semantics")
assert_grepl("near-alpha verifier",
             "README must document hybrid verifier")

cat("PASS mgcv compatibility docs contract\n")
