source("fastkpc/R/skeleton_ptable_parity.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

compare_sepsets <- function(left, right) {
  if (length(left) != length(right)) return(FALSE)
  for (i in seq_along(left)) {
    if (length(left[[i]]) != length(right[[i]])) return(FALSE)
    for (j in seq_along(left[[i]])) {
      lhs <- sort(as.integer(left[[i]][[j]]))
      rhs <- sort(as.integer(right[[i]][[j]]))
      if (!identical(lhs, rhs)) return(FALSE)
    }
  }
  TRUE
}

p <- 6L
alpha <- 0.05
max_conditioning_size <- 2L

reference <- fastkpc_run_skeleton_ptable_parity(
  output_dir = tempfile("skeleton-ptable-reference-"),
  p = p,
  alpha = alpha,
  max_conditioning_size = max_conditioning_size
)
native <- precision_run_skeleton_ptable_native(
  p = p,
  alpha = alpha,
  max_conditioning_size = max_conditioning_size,
  trace_level = "full"
)

assert_true(identical(native$adjacency, reference$native$adjacency),
            "native one-call p-table adjacency should match R/native reference")
assert_true(max(abs(native$pMax - reference$native$pmax)) < 1e-12,
            "native one-call p-table pMax should match R/native reference")
assert_true(identical(as.integer(native$n.edgetests),
                      as.integer(reference$native$n_edge_tests)),
            "native one-call p-table n.edgetests should match reference")
assert_true(compare_sepsets(native$sepsets, reference$native$sepsets),
            "native one-call p-table sepsets should match reference")

summary <- native$summary
assert_true(identical(as.integer(summary$p), p),
            "native one-call summary should record p")
assert_true(identical(as.integer(summary$max_conditioning_size),
                      max_conditioning_size),
            "native one-call summary should record max conditioning size")
assert_true(identical(as.integer(summary$levels),
                      max_conditioning_size + 1L),
            "native one-call summary should record level count")
assert_true(identical(as.integer(summary$tasks_planned),
                      as.integer(reference$summary$tasks_planned[[1L]])),
            "native one-call summary should record planned task count")
assert_true(identical(as.integer(summary$tests_replayed),
                      as.integer(reference$summary$tests_replayed[[1L]])),
            "native one-call summary should record replayed test count")
assert_true(identical(as.integer(summary$deletions),
                      as.integer(reference$summary$deletions[[1L]])),
            "native one-call summary should record deletion count")

tasks <- native$tasks
levels <- native$levels
required_task_fields <- c(
  "canonical_test_order_id", "level", "task_index", "edge_x", "edge_y",
  "x", "y", "S_key", "conditioning_size", "p_candidate",
  "native_edge_deleted", "native_edge_ignored"
)
assert_true(length(setdiff(required_task_fields, names(tasks))) == 0L,
            "native one-call task trace should expose canonical fields")
assert_true(identical(tasks$canonical_test_order_id, seq_len(nrow(tasks))),
            "native one-call task trace should preserve global canonical order")
assert_true(any(tasks$level == 2L & tasks$native_edge_deleted),
            "native one-call p-table should include a level-2 deletion")
assert_true(any(tasks$level == 2L & tasks$native_edge_ignored),
            "native one-call p-table should include level-2 ignored tasks")
assert_true(any(levels$level == 2L & levels$tasks_ignored_after_delete > 0L),
            "native one-call level trace should record level-2 ignored tasks")

cat("PASS skeleton p-table native one-call\n")
