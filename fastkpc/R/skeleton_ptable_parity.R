if (!exists("precision_replay_layer_native", mode = "function")) {
  source("fastkpc/R/cuda_native.R")
}
if (!exists("fastkpc_batched_precision_make_layer_plan", mode = "function")) {
  source("fastkpc/R/fast_kpc.R")
}

fastkpc_ptable_sepsets <- function(p) {
  replicate(p, replicate(p, integer(), simplify = FALSE), simplify = FALSE)
}

fastkpc_ptable_s_key <- function(S) {
  S <- as.integer(S)
  if (length(S) == 0L) return("")
  paste(S, collapse = "|")
}

fastkpc_ptable_p_for_task <- function(task, alpha) {
  if (length(task$S) == 0L) return(alpha / 5)
  if (task$edge_x == 1L && task$edge_y == 2L &&
      identical(as.integer(task$S), 3L)) {
    return(alpha * 1.4)
  }
  if (task$edge_x == 1L && task$edge_y == 3L &&
      identical(as.integer(task$S), 2L)) {
    return(alpha * 1.2)
  }
  if (task$edge_x == 2L && task$edge_y == 4L &&
      length(task$S) == 1L && task$S[[1L]] %in% c(1L, 3L)) {
    return(alpha * 1.1)
  }
  alpha / (3 + task$task_id)
}

fastkpc_ptable_replay_layer_r <- function(adjacency, pmax, tasks, p_values,
                                          alpha) {
  p <- nrow(adjacency)
  delete_edges <- matrix(FALSE, p, p)
  sepsets <- fastkpc_ptable_sepsets(p)
  edge_done <- new.env(parent = emptyenv())
  rows <- vector("list", length(tasks))
  level_log <- list()
  tests_replayed <- 0L
  ignored <- 0L
  deletions <- 0L

  for (i in seq_along(tasks)) {
    task <- tasks[[i]]
    key <- paste(task$edge_x, task$edge_y, sep = "-")
    already_deleted <- isTRUE(edge_done[[key]]) ||
      !isTRUE(adjacency[task$edge_x, task$edge_y])
    if (already_deleted) {
      ignored <- ignored + 1L
      rows[[i]] <- data.frame(
        task_index = as.integer(i),
        edge_x = task$edge_x,
        edge_y = task$edge_y,
        x = task$x,
        y = task$y,
        S_key = task$S_key,
        p_used = NA_real_,
        edge_deleted = FALSE,
        edge_already_deleted = TRUE,
        stringsAsFactors = FALSE
      )
      next
    }

    tests_replayed <- tests_replayed + 1L
    pval <- as.numeric(p_values[[i]])
    if (!is.finite(pval)) pval <- 1.0
    if (pval > pmax[task$edge_x, task$edge_y]) {
      pmax[task$edge_x, task$edge_y] <- pval
      pmax[task$edge_y, task$edge_x] <- pval
    }
    deleted <- pval >= alpha
    if (deleted) {
      deletions <- deletions + 1L
      delete_edges[task$edge_x, task$edge_y] <- TRUE
      delete_edges[task$edge_y, task$edge_x] <- TRUE
      sepsets[[task$edge_x]][[task$edge_y]] <- as.integer(task$S)
      sepsets[[task$edge_y]][[task$edge_x]] <- as.integer(task$S)
      level_log[[length(level_log) + 1L]] <- list(
        x = as.integer(task$edge_x),
        y = as.integer(task$edge_y),
        S = as.integer(task$S),
        p.value = pval
      )
      edge_done[[key]] <- TRUE
    }
    rows[[i]] <- data.frame(
      task_index = as.integer(i),
      edge_x = task$edge_x,
      edge_y = task$edge_y,
      x = task$x,
      y = task$y,
      S_key = task$S_key,
      p_used = pval,
      edge_deleted = deleted,
      edge_already_deleted = FALSE,
      stringsAsFactors = FALSE
    )
  }

  adjacency[delete_edges] <- FALSE
  list(
    adjacency = adjacency,
    sepsets = sepsets,
    pMax = pmax,
    n.edgetests = tests_replayed,
    per.level.log = level_log,
    summary = list(
      tasks_planned = length(tasks),
      tests_replayed = tests_replayed,
      tasks_ignored_after_delete = ignored,
      deletions = deletions
    ),
    replay_rows = do.call(rbind, rows)
  )
}

fastkpc_ptable_merge_sepsets <- function(state_sepsets, level_log) {
  for (entry in level_log) {
    x <- as.integer(entry$x)
    y <- as.integer(entry$y)
    S <- as.integer(entry$S)
    state_sepsets[[x]][[y]] <- S
    state_sepsets[[y]][[x]] <- S
  }
  state_sepsets
}

fastkpc_ptable_native_replay_level <- function(adjacency, pmax, tasks, p_values,
                                               alpha, trace_level = "full") {
  precision_replay_layer_native(
    adjacency = adjacency,
    edge_x = vapply(tasks, `[[`, integer(1L), "edge_x"),
    edge_y = vapply(tasks, `[[`, integer(1L), "edge_y"),
    x = vapply(tasks, `[[`, integer(1L), "x"),
    y = vapply(tasks, `[[`, integer(1L), "y"),
    conditioning_sets = lapply(tasks, `[[`, "S"),
    p_values = as.numeric(p_values),
    alpha = alpha,
    pmax = pmax,
    trace_level = trace_level
  )
}

fastkpc_ptable_state <- function(p) {
  list(
    adjacency = {
      adj <- matrix(TRUE, p, p)
      diag(adj) <- FALSE
      adj
    },
    pmax = {
      mat <- matrix(-Inf, p, p)
      diag(mat) <- 1
      mat
    },
    sepsets = fastkpc_ptable_sepsets(p),
    n_edge_tests = integer()
  )
}

fastkpc_run_skeleton_ptable_parity <- function(
    output_dir = file.path("fastkpc", "artifacts", "skeleton_ptable_parity"),
    p = 4L,
    alpha = 0.05,
    max_conditioning_size = 1L) {
  p <- as.integer(p)
  alpha <- as.numeric(alpha)
  max_conditioning_size <- as.integer(max_conditioning_size)
  if (p < 3L) stop("p-table parity requires p >= 3", call. = FALSE)

  r_state <- fastkpc_ptable_state(p)
  native_state <- fastkpc_ptable_state(p)
  task_rows <- list()
  level_rows <- list()
  global_id <- 0L

  for (level in seq.int(0L, max_conditioning_size)) {
    tasks <- fastkpc_batched_precision_make_layer_plan(
      native_state$adjacency, level
    )
    p_values <- vapply(tasks, fastkpc_ptable_p_for_task, numeric(1L),
                       alpha = alpha)
    native <- fastkpc_ptable_native_replay_level(
      native_state$adjacency, native_state$pmax, tasks, p_values, alpha,
      trace_level = "full"
    )
    ref <- fastkpc_ptable_replay_layer_r(
      r_state$adjacency, r_state$pmax, tasks, p_values, alpha
    )

    if (!identical(native$adjacency, ref$adjacency)) {
      stop("native p-table replay adjacency mismatch", call. = FALSE)
    }
    if (max(abs(native$pMax - ref$pMax)) > 1e-12) {
      stop("native p-table replay pMax mismatch", call. = FALSE)
    }
    if (!identical(as.integer(native$summary$tests_replayed),
                   as.integer(ref$summary$tests_replayed))) {
      stop("native p-table replay test count mismatch", call. = FALSE)
    }
    if (!identical(as.integer(native$summary$deletions),
                   as.integer(ref$summary$deletions))) {
      stop("native p-table replay deletion count mismatch", call. = FALSE)
    }

    native_state$adjacency <- native$adjacency
    native_state$pmax <- native$pMax
    native_state$sepsets <- fastkpc_ptable_merge_sepsets(
      native_state$sepsets, native$per.level.log
    )
    native_state$n_edge_tests <- c(
      native_state$n_edge_tests,
      as.integer(native$summary$tests_replayed)
    )
    r_state$adjacency <- ref$adjacency
    r_state$pmax <- ref$pMax
    r_state$sepsets <- fastkpc_ptable_merge_sepsets(
      r_state$sepsets, ref$per.level.log
    )
    r_state$n_edge_tests <- c(
      r_state$n_edge_tests,
      as.integer(ref$summary$tests_replayed)
    )

    if (length(tasks) > 0L) {
      for (i in seq_along(tasks)) {
        global_id <- global_id + 1L
        task <- tasks[[i]]
        task_rows[[length(task_rows) + 1L]] <- data.frame(
          canonical_test_order_id = global_id,
          level = as.integer(level),
          task_index = as.integer(i),
          edge_x = as.integer(task$edge_x),
          edge_y = as.integer(task$edge_y),
          x = as.integer(task$x),
          y = as.integer(task$y),
          S_key = task$S_key,
          p_used = as.numeric(p_values[[i]]),
          native_edge_deleted =
            i %in% as.integer(native$deleted_task_index %||% integer()),
          native_edge_ignored =
            i %in% as.integer(native$ignored_task_index %||% integer()),
          stringsAsFactors = FALSE
        )
      }
    }
    level_rows[[length(level_rows) + 1L]] <- data.frame(
      level = as.integer(level),
      tasks_planned = as.integer(native$summary$tasks_planned),
      tests_replayed = as.integer(native$summary$tests_replayed),
      tasks_ignored_after_delete =
        as.integer(native$summary$tasks_ignored_after_delete),
      deletions = as.integer(native$summary$deletions),
      stringsAsFactors = FALSE
    )
  }

  sepset_mismatch <- FALSE
  for (i in seq_len(p)) {
    for (j in seq_len(p)) {
      left <- sort(as.integer(native_state$sepsets[[i]][[j]]))
      right <- sort(as.integer(r_state$sepsets[[i]][[j]]))
      if (!identical(left, right)) sepset_mismatch <- TRUE
    }
  }

  summary <- data.frame(
    p = p,
    alpha = alpha,
    max_conditioning_size = max_conditioning_size,
    levels = max_conditioning_size + 1L,
    tasks_planned = sum(vapply(level_rows, function(row) row$tasks_planned,
                               integer(1L))),
    tests_replayed = sum(native_state$n_edge_tests),
    deletions = sum(vapply(level_rows, function(row) row$deletions,
                           integer(1L))),
    adjacency_identical = identical(native_state$adjacency, r_state$adjacency),
    pmax_max_abs_diff = max(abs(native_state$pmax - r_state$pmax)),
    n_edgetests_identical =
      identical(as.integer(native_state$n_edge_tests),
                as.integer(r_state$n_edge_tests)),
    sepsets_identical = !isTRUE(sepset_mismatch),
    stringsAsFactors = FALSE
  )

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  paths <- list(
    tasks_csv = file.path(output_dir, "skeleton_ptable_tasks.csv"),
    levels_csv = file.path(output_dir, "skeleton_ptable_levels.csv"),
    summary_csv = file.path(output_dir, "skeleton_ptable_summary.csv"),
    summary_md = file.path(output_dir, "skeleton_ptable_summary.md")
  )
  utils::write.csv(do.call(rbind, task_rows), paths$tasks_csv, row.names = FALSE)
  utils::write.csv(do.call(rbind, level_rows), paths$levels_csv, row.names = FALSE)
  utils::write.csv(summary, paths$summary_csv, row.names = FALSE)
  md <- c(
    "# Skeleton P-Table Native Replay Parity",
    "",
    paste0("- Tasks: `", basename(paths$tasks_csv), "`"),
    paste0("- Levels: `", basename(paths$levels_csv), "`"),
    paste0("- Summary: `", basename(paths$summary_csv), "`"),
    paste0("- Adjacency identical: ", summary$adjacency_identical[[1L]]),
    paste0("- Sepsets identical: ", summary$sepsets_identical[[1L]]),
    paste0("- pMax max abs diff: ", summary$pmax_max_abs_diff[[1L]])
  )
  writeLines(md, paths$summary_md)

  list(
    summary = summary,
    tasks = do.call(rbind, task_rows),
    levels = do.call(rbind, level_rows),
    native = native_state,
    reference = r_state,
    paths = paths
  )
}
