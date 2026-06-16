source("fastkpc/R/fast_kpc.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

set.seed(102)
data <- cbind(
  a = seq(-2, 2, length.out = 100),
  b = sin(seq(-2, 2, length.out = 100)),
  c = rnorm(100),
  d = cos(seq(-2, 2, length.out = 100))
)

result <- fast_kpc(data, alpha = 0.2, max_conditioning_size = 1L,
                   engine = "cpu", residual_backend = "fastSpline")

required_top <- c("config", "data_info", "engine", "skeleton", "orientation",
                  "metrics", "timings", "cache", "validation", "benchmark",
                  "diagnostics")
missing_top <- setdiff(required_top, names(result))
assert_true(length(missing_top) == 0L,
            paste("missing result fields:", paste(missing_top, collapse = ", ")))

required_config <- c("alpha", "max_conditioning_size", "engine_requested",
                     "engine_used", "residual_backend",
                     "residual_device_requested", "residual_device_used",
                     "graph_stage",
                     "residual_cache", "index", "legacy_index", "batch_size",
                     "orient_collider", "solve_confl", "rules",
                     "fastspline_params", "cuda_residual_fallback",
                     "validate", "benchmark", "legacy",
                     "seed")
assert_true(all(required_config %in% names(result$config)),
            "config should include all required fields")

required_metrics <- c("skeleton_edge_count", "directed_edge_count",
                      "undirected_edge_count", "bidirected_edge_count",
                      "orientation_event_count", "generalized_orientation_count",
                      "max_pmax", "min_nonzero_pmax")
assert_true(all(required_metrics %in% names(result$metrics)),
            "metrics should include all required fields")

assert_true(validate_fastkpc_result(result), "validate_fastkpc_result should return TRUE")
assert_true(is.character(capture.output(print(result))), "print method should produce text")
summary_value <- summary(result)
assert_true(is.list(summary_value), "summary should return a list")
assert_true(identical(fastkpc_extract_skeleton(result), result$skeleton),
            "skeleton extractor should return skeleton")
assert_true(identical(fastkpc_extract_pdag(result), result$orientation$pdag),
            "pdag extractor should return pdag")

bad <- result
bad$config <- NULL
err <- tryCatch(validate_fastkpc_result(bad), error = conditionMessage)
assert_true(grepl("missing config", err), "missing config should be rejected")

cat("test_fastkpc_result_contract.R: PASS\n")
