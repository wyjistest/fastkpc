source("fastkpc/R/skeleton_ptable_parity.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

out_dir <- tempfile("skeleton-ptable-parity-")
artifact <- fastkpc_run_skeleton_ptable_parity(output_dir = out_dir)
summary <- artifact$summary[1L, , drop = FALSE]
tasks <- artifact$tasks
levels <- artifact$levels

required_task_fields <- c(
  "canonical_test_order_id", "level", "task_index", "edge_x", "edge_y",
  "x", "y", "S_key", "p_used", "native_edge_deleted",
  "native_edge_ignored"
)
missing_task_fields <- setdiff(required_task_fields, names(tasks))
assert_true(length(missing_task_fields) == 0L,
            paste("missing task fields:",
                  paste(missing_task_fields, collapse = ",")))
assert_true(identical(tasks$canonical_test_order_id, seq_len(nrow(tasks))),
            "p-table task artifact should preserve canonical order")
assert_true(any(tasks$native_edge_deleted),
            "p-table scenario should include at least one deletion")
assert_true(any(tasks$native_edge_ignored),
            "p-table scenario should include ignored post-delete tasks")
assert_true(any(tasks$level == 1L & tasks$S_key == "3" &
                  tasks$edge_x == 1L & tasks$edge_y == 2L &
                  tasks$native_edge_deleted),
            "edge 1-2 should be deleted by the S=3 p-table row")

required_level_fields <- c(
  "level", "tasks_planned", "tests_replayed",
  "tasks_ignored_after_delete", "deletions"
)
missing_level_fields <- setdiff(required_level_fields, names(levels))
assert_true(length(missing_level_fields) == 0L,
            paste("missing level fields:",
                  paste(missing_level_fields, collapse = ",")))
assert_true(all(levels$tasks_planned >= levels$tests_replayed),
            "planned tasks should dominate replayed tasks")

assert_true(isTRUE(summary$adjacency_identical),
            "native p-table replay adjacency should match R reference")
assert_true(isTRUE(summary$sepsets_identical),
            "native p-table replay sepsets should match R reference")
assert_true(isTRUE(summary$n_edgetests_identical),
            "native p-table replay n.edgetests should match R reference")
assert_true(summary$pmax_max_abs_diff < 1e-12,
            "native p-table replay pMax should match R reference")

assert_true(file.exists(artifact$paths$tasks_csv),
            "p-table task CSV should be written")
assert_true(file.exists(artifact$paths$levels_csv),
            "p-table level CSV should be written")
assert_true(file.exists(artifact$paths$summary_csv),
            "p-table summary CSV should be written")
assert_true(file.exists(artifact$paths$summary_md),
            "p-table summary Markdown should be written")
summary_md <- paste(readLines(artifact$paths$summary_md, warn = FALSE),
                    collapse = "\n")
assert_true(grepl("Skeleton P-Table Native Replay Parity", summary_md,
                  fixed = TRUE),
            "summary Markdown should name the p-table parity gate")

cat("PASS skeleton p-table parity artifact\n")
