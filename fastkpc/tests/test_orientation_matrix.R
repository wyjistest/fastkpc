source("fastkpc/R/native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

self <- orientation_matrix_selftest()

required <- c(
  "empty_has_no_edges",
  "undirected_roundtrip",
  "directed_roundtrip",
  "conflict_roundtrip",
  "edge_predicates_correct",
  "from_skeleton_symmetric",
  "diff_counts_correct",
  "invalid_indices_rejected"
)

missing <- setdiff(required, names(self))
assert_true(length(missing) == 0L,
            paste("orientation matrix selftest missing fields:",
                  paste(missing, collapse = ", ")))

for (name in required) {
  assert_true(self[[name]], paste(name, "should be TRUE"))
}

cat("test_orientation_matrix.R: PASS\n")
