source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

assert_task_equal <- function(native_tasks, r_tasks, level) {
  assert_true(length(native_tasks) == length(r_tasks),
              paste("native task count mismatch at level", level))
  if (length(r_tasks) == 0L) return(invisible(TRUE))

  for (i in seq_along(r_tasks)) {
    native <- native_tasks[[i]]
    ref <- r_tasks[[i]]
    assert_true(identical(as.integer(native$task_id), as.integer(ref$task_id)),
                paste("task_id mismatch at level", level, "row", i))
    assert_true(identical(as.integer(native$edge_x), as.integer(ref$edge_x)),
                paste("edge_x mismatch at level", level, "row", i))
    assert_true(identical(as.integer(native$edge_y), as.integer(ref$edge_y)),
                paste("edge_y mismatch at level", level, "row", i))
    assert_true(identical(as.integer(native$x), as.integer(ref$x)),
                paste("x mismatch at level", level, "row", i))
    assert_true(identical(as.integer(native$y), as.integer(ref$y)),
                paste("y mismatch at level", level, "row", i))
    assert_true(identical(as.integer(native$S), as.integer(ref$S)),
                paste("S mismatch at level", level, "row", i))
    assert_true(identical(native$S_key, ref$S_key),
                paste("S_key mismatch at level", level, "row", i))
    assert_true(identical(as.integer(native$conditioning_size),
                          as.integer(length(ref$S))),
                paste("conditioning_size mismatch at level", level, "row", i))
    assert_true(identical(native$conditioning_target_side,
                          ref$conditioning_target_side),
                paste("conditioning target side mismatch at level", level,
                      "row", i))
  }
  invisible(TRUE)
}

adjacency <- matrix(TRUE, 6L, 6L)
diag(adjacency) <- FALSE
adjacency[1L, 6L] <- adjacency[6L, 1L] <- FALSE
adjacency[2L, 5L] <- adjacency[5L, 2L] <- FALSE

for (level in 0:2) {
  native <- precision_make_layer_plan_native(adjacency, level)
  r_tasks <- fastkpc_batched_precision_make_layer_plan(adjacency, level)
  assert_task_equal(native$tasks, r_tasks, level)

  assert_true(identical(as.integer(native$summary$level), as.integer(level)),
              paste("native summary level mismatch at level", level))
  assert_true(identical(as.integer(native$summary$p), as.integer(ncol(adjacency))),
              paste("native summary p mismatch at level", level))
  assert_true(identical(as.integer(native$summary$tasks_planned),
                        as.integer(length(r_tasks))),
              paste("native summary task count mismatch at level", level))
  assert_true(identical(as.integer(native$summary$unconditional_tasks),
                        sum(vapply(r_tasks, function(task) length(task$S) == 0L,
                                   logical(1L)))),
              paste("native unconditional count mismatch at level", level))
  assert_true(identical(as.integer(native$summary$conditional_tasks),
                        sum(vapply(r_tasks, function(task) length(task$S) > 0L,
                                   logical(1L)))),
              paste("native conditional count mismatch at level", level))
}

cat("PASS skeleton native layer plan parity\n")
