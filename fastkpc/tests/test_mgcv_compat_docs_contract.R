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
assert_grepl("Gate B campaign",
             "README must mention Gate B campaign")
assert_grepl("canonical hybrid replay",
             "README must mention canonical hybrid replay")
assert_grepl("verifier may replace p-values but not replay order",
             "README must state verifier replay invariant")
assert_grepl("Baseline: mgcv Gate B fixed-sp self-solve + hybrid canonical replay",
             "README must document Gate B baseline")
assert_grepl("Commit: 5da2313",
             "README must document baseline commit")
assert_grepl("Tag: mgcv-gate-b-v1",
             "README must document baseline tag")
assert_grepl("fastkpc_mgcv_extract_capabilities()",
             "README must document mgcv capability object")
assert_grepl("hybrid calibration campaign",
             "README must document hybrid calibration")
assert_grepl("hybrid_calibration_summary.csv",
             "README must document calibration artifact")
assert_grepl("graph-level golden snapshots",
             "README must document golden snapshots")

cat("PASS mgcv compatibility docs contract\n")
