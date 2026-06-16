source("fastkpc/R/wanpdag_validation.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

validation <- validate_wanpdag_against_legacy(
  seed = 81,
  n = 120,
  alpha = 0.2,
  max_conditioning_size = 1L
)

required <- c(
  "available",
  "reason_if_unavailable",
  "native",
  "legacy",
  "diff",
  "event_counts",
  "cache_stats",
  "metrics",
  "fixture"
)
missing <- setdiff(required, names(validation))
assert_true(length(missing) == 0L,
            paste("legacy validation missing fields:",
                  paste(missing, collapse = ", ")))

if (!validation$available) {
  assert_true(grepl("pcalg|graph", validation$reason_if_unavailable),
              "unavailable legacy validation should mention missing pcalg/graph")
} else {
  assert_true(is.list(validation$diff$directed),
              "available legacy diff should contain directed section")
  assert_true(is.list(validation$diff$undirected),
              "available legacy diff should contain undirected section")
  assert_true(is.list(validation$diff$bidirected),
              "available legacy diff should contain bidirected section")
}

counts <- unlist(validation$event_counts, use.names = FALSE)
assert_true(length(counts) > 0L && all(is.finite(counts)) && all(counts >= 0),
            "native event counts should be non-negative")
assert_true(validation$cache_stats$computations <= validation$cache_stats$requests,
            "cache computations should not exceed requests")
assert_true(isTRUE(validation$fixture$pdag_exact),
            "hand-written fixture should match expected pdag exactly")

cat("test_wanpdag_legacy_validation.R: PASS\n")
