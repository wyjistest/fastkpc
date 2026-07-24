fastkpc_full_cuda_or <- function(x, y) if (is.null(x)) y else x

fastkpc_full_cuda_require_namespace <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) {
    stop("package required for full CUDA CI artifacts: ", package,
         call. = FALSE)
  }
}

fastkpc_full_cuda_read_result <- function(value) {
  if (is.character(value) && length(value) == 1L) {
    if (!file.exists(value)) {
      stop("result file not found: ", value, call. = FALSE)
    }
    return(readRDS(value))
  }
  value
}

fastkpc_full_cuda_is_skeleton <- function(value) {
  is.list(value) && !is.null(value$adjacency) &&
    !is.null(value$sepsets) && !is.null(value$n.edgetests)
}

fastkpc_full_cuda_extract_skeleton <- function(value,
                                               role = c("auto", "oracle",
                                                        "candidate")) {
  role <- match.arg(role)
  value <- fastkpc_full_cuda_read_result(value)
  if (is.null(value)) return(NULL)
  if (fastkpc_full_cuda_is_skeleton(value)) return(value)

  slots <- switch(
    role,
    oracle = c("skeleton", "reference", "facade", "candidate", "result"),
    candidate = c("facade", "candidate", "skeleton", "result", "reference"),
    auto = c("skeleton", "facade", "candidate", "reference", "result")
  )
  for (slot in slots) {
    candidate <- value[[slot]]
    if (fastkpc_full_cuda_is_skeleton(candidate)) {
      candidate$artifact_summary <- value$summary
      return(candidate)
    }
  }
  NULL
}

fastkpc_full_cuda_matrix_labels <- function(adjacency) {
  labels <- rownames(adjacency)
  if (is.null(labels) || length(labels) != nrow(adjacency)) {
    labels <- paste0("V", seq_len(nrow(adjacency)))
  }
  as.character(labels)
}

fastkpc_full_cuda_align_matrix <- function(value, labels) {
  if (is.null(value)) return(NULL)
  value <- as.matrix(value)
  if (!identical(dim(value), c(length(labels), length(labels)))) {
    return(NULL)
  }
  rows <- rownames(value)
  cols <- colnames(value)
  if (is.null(rows) || is.null(cols)) return(NULL)
  labels_match <- length(unique(rows)) == length(labels) &&
    length(unique(cols)) == length(labels) &&
    setequal(as.character(rows), labels) &&
    setequal(as.character(cols), labels)
  if (!labels_match) return(NULL)
  value <- value[labels, labels, drop = FALSE]
  dimnames(value) <- list(labels, labels)
  value
}

fastkpc_full_cuda_normalize_s <- function(value) {
  if (is.null(value) || length(value) == 0L) return(integer())
  value <- suppressWarnings(as.integer(value))
  sort(unique(value[!is.na(value)]))
}

fastkpc_full_cuda_parse_s_key <- function(value) {
  if (is.null(value) || length(value) == 0L || is.na(value[[1L]]) ||
      !nzchar(trimws(as.character(value[[1L]])))) {
    return(integer())
  }
  pieces <- strsplit(as.character(value[[1L]]), "[|,[:space:]]+")[[1L]]
  fastkpc_full_cuda_normalize_s(pieces[nzchar(pieces)])
}

fastkpc_full_cuda_s_key <- function(value) {
  paste(fastkpc_full_cuda_normalize_s(value), collapse = "|")
}

fastkpc_full_cuda_sepset_cell <- function(sepsets, i, j, labels) {
  if (is.null(sepsets) || length(sepsets) < i) return(integer())
  outer_index <- i
  outer_names <- names(sepsets)
  if (!is.null(outer_names) && labels[[i]] %in% outer_names) {
    outer_index <- match(labels[[i]], outer_names)
  }
  row <- sepsets[[outer_index]]
  if (is.null(row) || length(row) < j) return(integer())
  inner_index <- j
  inner_names <- names(row)
  if (!is.null(inner_names) && labels[[j]] %in% inner_names) {
    inner_index <- match(labels[[j]], inner_names)
  }
  fastkpc_full_cuda_normalize_s(row[[inner_index]])
}

fastkpc_full_cuda_empty_sepset_frame <- function() {
  data.frame(
    edge_x = integer(),
    edge_y = integer(),
    edge_x_label = character(),
    edge_y_label = character(),
    adjacent = logical(),
    S_key = character(),
    S_size = integer(),
    direction_conflict = logical(),
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_normalize_sepsets <- function(skeleton, labels = NULL) {
  if (!fastkpc_full_cuda_is_skeleton(skeleton)) {
    return(fastkpc_full_cuda_empty_sepset_frame())
  }
  adjacency <- as.matrix(skeleton$adjacency)
  if (is.null(labels)) labels <- fastkpc_full_cuda_matrix_labels(adjacency)
  adjacency <- fastkpc_full_cuda_align_matrix(adjacency, labels)
  if (is.null(adjacency)) return(fastkpc_full_cuda_empty_sepset_frame())
  if (length(labels) < 2L) return(fastkpc_full_cuda_empty_sepset_frame())

  rows <- vector("list", choose(length(labels), 2L))
  index <- 0L
  for (i in seq_len(length(labels) - 1L)) {
    for (j in seq.int(i + 1L, length(labels))) {
      forward <- fastkpc_full_cuda_sepset_cell(
        skeleton$sepsets, i, j, labels
      )
      reverse <- fastkpc_full_cuda_sepset_cell(
        skeleton$sepsets, j, i, labels
      )
      forward_key <- fastkpc_full_cuda_s_key(forward)
      reverse_key <- fastkpc_full_cuda_s_key(reverse)
      nonempty <- c(forward_key, reverse_key)
      nonempty <- unique(nonempty[nzchar(nonempty)])
      conflict <- length(nonempty) > 1L
      selected <- if (length(nonempty) == 0L) "" else sort(nonempty)[[1L]]
      index <- index + 1L
      rows[[index]] <- data.frame(
        edge_x = i,
        edge_y = j,
        edge_x_label = labels[[i]],
        edge_y_label = labels[[j]],
        adjacent = isTRUE(adjacency[i, j]),
        S_key = selected,
        S_size = length(fastkpc_full_cuda_parse_s_key(selected)),
        direction_conflict = conflict,
        stringsAsFactors = FALSE
      )
    }
  }
  do.call(rbind, rows)
}

fastkpc_full_cuda_empty_deletion_trace <- function() {
  data.frame(
    canonical_deletion_id = integer(),
    source_sequence_id = integer(),
    level = integer(),
    edge_x = integer(),
    edge_y = integer(),
    tested_x = integer(),
    tested_y = integer(),
    S_key = character(),
    p_value = numeric(),
    trace_source = character(),
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_deletion_trace_from_tasks <- function(tasks) {
  if (!is.data.frame(tasks) || nrow(tasks) == 0L ||
      !"native_edge_deleted" %in% names(tasks)) {
    return(fastkpc_full_cuda_empty_deletion_trace())
  }
  keep <- as.logical(tasks$native_edge_deleted)
  if ("native_edge_ignored" %in% names(tasks)) {
    keep <- keep & !as.logical(tasks$native_edge_ignored)
  }
  rows <- tasks[which(keep), , drop = FALSE]
  if (nrow(rows) == 0L) return(fastkpc_full_cuda_empty_deletion_trace())

  tested_x <- if ("x" %in% names(rows)) rows$x else rows$edge_x
  tested_y <- if ("y" %in% names(rows)) rows$y else rows$edge_y
  source_id <- if ("canonical_test_order_id" %in% names(rows)) {
    rows$canonical_test_order_id
  } else {
    seq_len(nrow(rows))
  }
  p_name <- intersect(c("p_candidate", "p_used", "p_value", "p"),
                      names(rows))
  p_value <- if (length(p_name) == 0L) {
    rep(NA_real_, nrow(rows))
  } else {
    as.numeric(rows[[p_name[[1L]]]])
  }
  s_values <- if ("S_key" %in% names(rows)) {
    vapply(rows$S_key, function(value) {
      fastkpc_full_cuda_s_key(fastkpc_full_cuda_parse_s_key(value))
    }, character(1))
  } else {
    rep("", nrow(rows))
  }
  data.frame(
    canonical_deletion_id = seq_len(nrow(rows)),
    source_sequence_id = as.integer(source_id),
    level = as.integer(rows$level),
    edge_x = pmin(as.integer(tested_x), as.integer(tested_y)),
    edge_y = pmax(as.integer(tested_x), as.integer(tested_y)),
    tested_x = as.integer(tested_x),
    tested_y = as.integer(tested_y),
    S_key = s_values,
    p_value = p_value,
    trace_source = "tasks",
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_deletion_trace_from_level_log <- function(level_log) {
  if (!is.list(level_log) || length(level_log) == 0L) {
    return(fastkpc_full_cuda_empty_deletion_trace())
  }
  rows <- list()
  source_id <- 0L
  for (level_index in seq_along(level_log)) {
    entries <- level_log[[level_index]]
    if (!is.list(entries) || length(entries) == 0L) next
    for (entry in entries) {
      if (!is.list(entry) || is.null(entry$x) || is.null(entry$y)) next
      source_id <- source_id + 1L
      x <- as.integer(entry$x[[1L]])
      y <- as.integer(entry$y[[1L]])
      if (!is.null(entry$S_xy)) {
        tested_x <- x
        tested_y <- y
        s_value <- entry$S_xy
      } else if (!is.null(entry$S_yx)) {
        tested_x <- y
        tested_y <- x
        s_value <- entry$S_yx
      } else {
        tested_x <- x
        tested_y <- y
        s_value <- integer()
      }
      rows[[length(rows) + 1L]] <- data.frame(
        canonical_deletion_id = length(rows) + 1L,
        source_sequence_id = source_id,
        level = as.integer(level_index - 1L),
        edge_x = min(x, y),
        edge_y = max(x, y),
        tested_x = tested_x,
        tested_y = tested_y,
        S_key = fastkpc_full_cuda_s_key(s_value),
        p_value = NA_real_,
        trace_source = "per.level.log",
        stringsAsFactors = FALSE
      )
    }
  }
  if (length(rows) == 0L) return(fastkpc_full_cuda_empty_deletion_trace())
  do.call(rbind, rows)
}

fastkpc_full_cuda_deletion_trace_from_sepsets <- function(skeleton, labels) {
  sepsets <- fastkpc_full_cuda_normalize_sepsets(skeleton, labels)
  sepsets <- sepsets[!sepsets$adjacent, , drop = FALSE]
  if (nrow(sepsets) == 0L) return(fastkpc_full_cuda_empty_deletion_trace())
  data.frame(
    canonical_deletion_id = seq_len(nrow(sepsets)),
    source_sequence_id = NA_integer_,
    level = as.integer(sepsets$S_size),
    edge_x = as.integer(sepsets$edge_x),
    edge_y = as.integer(sepsets$edge_y),
    tested_x = NA_integer_,
    tested_y = NA_integer_,
    S_key = sepsets$S_key,
    p_value = NA_real_,
    trace_source = "sepsets",
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_normalize_deletion_trace <- function(skeleton,
                                                        labels = NULL) {
  if (!fastkpc_full_cuda_is_skeleton(skeleton)) {
    return(fastkpc_full_cuda_empty_deletion_trace())
  }
  if (is.null(labels)) {
    labels <- fastkpc_full_cuda_matrix_labels(skeleton$adjacency)
  }
  trace <- fastkpc_full_cuda_deletion_trace_from_tasks(skeleton$tasks)
  if (nrow(trace) == 0L) {
    trace <- fastkpc_full_cuda_deletion_trace_from_level_log(
      skeleton$per.level.log
    )
  }
  if (nrow(trace) == 0L) {
    trace <- fastkpc_full_cuda_deletion_trace_from_sepsets(skeleton, labels)
  }
  if (nrow(trace) == 0L) return(trace)
  order_index <- order(trace$level, trace$edge_x, trace$edge_y, trace$S_key,
                       na.last = TRUE)
  trace <- trace[order_index, , drop = FALSE]
  rownames(trace) <- NULL
  trace$canonical_deletion_id <- seq_len(nrow(trace))
  trace
}

fastkpc_full_cuda_empty_logical_trace <- function() {
  data.frame(
    logical_sequence_id = integer(),
    source_sequence_id = integer(),
    source_task_index = integer(),
    level = integer(),
    x = integer(),
    y = integer(),
    S_key = character(),
    p_value = numeric(),
    deletes_edge = logical(),
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_normalize_logical_trace <- function(skeleton) {
  if (!fastkpc_full_cuda_is_skeleton(skeleton)) {
    return(fastkpc_full_cuda_empty_logical_trace())
  }
  tasks <- skeleton$tasks
  if (is.data.frame(tasks) && nrow(tasks) > 0L) {
    keep <- rep(TRUE, nrow(tasks))
    if ("native_edge_ignored" %in% names(tasks)) {
      keep <- !as.logical(tasks$native_edge_ignored)
    }
    tasks <- tasks[which(keep), , drop = FALSE]
    if (nrow(tasks) == 0L) return(fastkpc_full_cuda_empty_logical_trace())
    if (all(c("level", "task_index") %in% names(tasks))) {
      tie_breaker <- if ("canonical_test_order_id" %in% names(tasks)) {
        as.integer(tasks$canonical_test_order_id)
      } else {
        seq_len(nrow(tasks))
      }
      tasks <- tasks[
        order(as.integer(tasks$level), as.integer(tasks$task_index),
              tie_breaker, na.last = TRUE),
        , drop = FALSE
      ]
      rownames(tasks) <- NULL
    }
    p_name <- intersect(c("p_candidate", "p_used", "p_value", "p"),
                        names(tasks))
    p_value <- if (length(p_name) == 0L) {
      rep(NA_real_, nrow(tasks))
    } else {
      as.numeric(tasks[[p_name[[1L]]]])
    }
    x <- if ("x" %in% names(tasks)) tasks$x else tasks$edge_x
    y <- if ("y" %in% names(tasks)) tasks$y else tasks$edge_y
    source_id <- if ("canonical_test_order_id" %in% names(tasks)) {
      tasks$canonical_test_order_id
    } else {
      seq_len(nrow(tasks))
    }
    source_task_index <- if ("task_index" %in% names(tasks)) {
      as.integer(tasks$task_index)
    } else {
      rep(NA_integer_, nrow(tasks))
    }
    s_values <- if ("S_key" %in% names(tasks)) {
      vapply(tasks$S_key, function(value) {
        fastkpc_full_cuda_s_key(fastkpc_full_cuda_parse_s_key(value))
      }, character(1))
    } else {
      rep("", nrow(tasks))
    }
    deleted <- if ("native_edge_deleted" %in% names(tasks)) {
      as.logical(tasks$native_edge_deleted)
    } else {
      rep(FALSE, nrow(tasks))
    }
    return(data.frame(
      logical_sequence_id = seq_len(nrow(tasks)),
      source_sequence_id = as.integer(source_id),
      source_task_index = source_task_index,
      level = as.integer(tasks$level),
      x = as.integer(x),
      y = as.integer(y),
      S_key = s_values,
      p_value = p_value,
      deletes_edge = deleted,
      stringsAsFactors = FALSE
    ))
  }

  trace <- skeleton$precision_trace
  if (!is.data.frame(trace) || nrow(trace) == 0L) {
    return(fastkpc_full_cuda_empty_logical_trace())
  }
  id_name <- intersect(c("canonical_test_order_id", "logical_sequence_id"),
                       names(trace))
  task_index_name <- intersect(c("task_index", "source_task_index"),
                               names(trace))
  level_name <- intersect(c("conditioning_level", "level"), names(trace))
  p_name <- intersect(c("p_used", "primary_p", "p_value", "p"),
                      names(trace))
  s_name <- intersect(c("S_key", "conditioning_set"), names(trace))
  delete_name <- intersect(c("edge_deleted", "decision_after_verify",
                             "native_edge_deleted"), names(trace))
  data.frame(
    logical_sequence_id = seq_len(nrow(trace)),
    source_sequence_id = if (length(id_name)) {
      as.integer(trace[[id_name[[1L]]]])
    } else seq_len(nrow(trace)),
    source_task_index = if (length(task_index_name)) {
      as.integer(trace[[task_index_name[[1L]]]])
    } else rep(NA_integer_, nrow(trace)),
    level = if (length(level_name)) {
      as.integer(trace[[level_name[[1L]]]])
    } else NA_integer_,
    x = as.integer(trace$x),
    y = as.integer(trace$y),
    S_key = if (length(s_name)) {
      vapply(trace[[s_name[[1L]]]], function(value) {
        fastkpc_full_cuda_s_key(fastkpc_full_cuda_parse_s_key(value))
      }, character(1))
    } else rep("", nrow(trace)),
    p_value = if (length(p_name)) {
      as.numeric(trace[[p_name[[1L]]]])
    } else NA_real_,
    deletes_edge = if (length(delete_name)) {
      as.logical(trace[[delete_name[[1L]]]])
    } else FALSE,
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_validate_logical_trace <- function(
    logical_trace, n_edgetests, role = "oracle") {
  expected <- as.integer(n_edgetests)
  if (length(expected) == 0L || anyNA(expected) || any(expected < 0L)) {
    stop(role, " n.edgetests are invalid", call. = FALSE)
  }
  if (!is.data.frame(logical_trace) || nrow(logical_trace) == 0L) {
    stop(role, " logical trace is missing", call. = FALSE)
  }
  levels <- as.integer(logical_trace$level)
  if (anyNA(levels) || any(levels < 0L) ||
      any(levels >= length(expected))) {
    stop(role, " logical trace contains an invalid level", call. = FALSE)
  }
  actual <- tabulate(levels + 1L, nbins = length(expected))
  if (!identical(as.integer(actual), expected)) {
    stop(role, " logical trace counts do not match n.edgetests: expected ",
         paste(expected, collapse = ","), "; observed ",
         paste(actual, collapse = ","), call. = FALSE)
  }
  task_indices <- as.integer(logical_trace$source_task_index)
  if (length(task_indices) != nrow(logical_trace) || anyNA(task_indices) ||
      any(task_indices <= 0L)) {
    stop(role, " logical trace is missing canonical source task indices",
         call. = FALSE)
  }
  task_keys <- paste(levels, task_indices, sep = "|")
  if (anyDuplicated(task_keys)) {
    stop(role, " logical trace contains duplicate source task indices",
         call. = FALSE)
  }
  invisible(actual)
}

fastkpc_full_cuda_plan_combinations <- function(values, choose) {
  values <- as.integer(values)
  if (choose == 0L) return(list(integer()))
  if (length(values) < choose) return(list())
  indices <- utils::combn(seq_along(values), choose, simplify = FALSE)
  lapply(indices, function(index) values[as.integer(index)])
}

fastkpc_full_cuda_validate_logical_trace_plan <- function(
    logical_trace, p, level_count, expected_adjacency = NULL,
    role = "oracle") {
  p <- as.integer(p)
  level_count <- as.integer(level_count)
  adjacency <- matrix(TRUE, p, p)
  diag(adjacency) <- FALSE

  for (level in seq_len(level_count) - 1L) {
    rows <- logical_trace[logical_trace$level == level, , drop = FALSE]
    rows <- rows[order(rows$source_task_index), , drop = FALSE]
    wanted <- as.integer(rows$source_task_index)
    wanted_position <- 1L
    task_index <- 0L

    visit_task <- function(x, y, S) {
      task_index <<- task_index + 1L
      if (wanted_position > length(wanted) ||
          task_index != wanted[[wanted_position]]) {
        return(invisible(NULL))
      }
      row <- rows[wanted_position, , drop = FALSE]
      expected_s <- fastkpc_full_cuda_s_key(S)
      if (!identical(as.integer(row$x[[1L]]), as.integer(x)) ||
          !identical(as.integer(row$y[[1L]]), as.integer(y)) ||
          !identical(as.character(row$S_key[[1L]]), expected_s)) {
        stop(role, " logical trace does not match canonical layer plan at ",
             "level ", level, " task ", task_index, call. = FALSE)
      }
      wanted_position <<- wanted_position + 1L
      invisible(NULL)
    }

    if (p >= 2L) {
      for (x in seq_len(p - 1L)) {
        for (y in seq.int(x + 1L, p)) {
          if (!isTRUE(adjacency[x, y])) next
          nx <- which(adjacency[, x] & (seq_len(p) != y))
          for (S in fastkpc_full_cuda_plan_combinations(nx, level)) {
            visit_task(x, y, S)
          }
          ny <- which(adjacency[, y] & (seq_len(p) != x))
          for (S in fastkpc_full_cuda_plan_combinations(ny, level)) {
            visit_task(y, x, S)
          }
        }
      }
    }
    if (wanted_position <= length(wanted)) {
      stop(role, " logical trace source task index exceeds canonical layer ",
           "plan at level ", level, call. = FALSE)
    }

    deleted <- rows[which(rows$deletes_edge), , drop = FALSE]
    if (nrow(deleted) > 0L) {
      for (i in seq_len(nrow(deleted))) {
        x <- as.integer(deleted$x[[i]])
        y <- as.integer(deleted$y[[i]])
        adjacency[x, y] <- FALSE
        adjacency[y, x] <- FALSE
      }
    }
  }

  if (!is.null(expected_adjacency) &&
      !identical(unname(as.logical(adjacency)),
                 unname(as.logical(expected_adjacency)))) {
    stop(role, " logical trace decisions do not reproduce oracle adjacency",
         call. = FALSE)
  }
  invisible(adjacency)
}

fastkpc_full_cuda_edge_count <- function(adjacency) {
  if (is.null(adjacency)) return(NA_integer_)
  as.integer(sum(unname(adjacency)) / 2L)
}

fastkpc_full_cuda_shd <- function(left, right) {
  if (is.null(left) || is.null(right) || !identical(dim(left), dim(right))) {
    return(NA_integer_)
  }
  as.integer(sum(abs(as.integer(left) - as.integer(right))) / 2L)
}

fastkpc_full_cuda_summary_value <- function(skeleton, name, default = NULL) {
  summaries <- list(skeleton$artifact_summary, skeleton$summary)
  for (summary in summaries) {
    if (is.data.frame(summary) && nrow(summary) > 0L &&
        name %in% names(summary)) {
      return(summary[[name]][[1L]])
    }
    if (is.list(summary) && !is.null(summary[[name]])) return(summary[[name]])
  }
  default
}

fastkpc_full_cuda_candidate_fallbacks <- function(skeleton) {
  unknown <- fastkpc_full_cuda_summary_value(
    skeleton, "unknown_fallback_count", NULL
  )
  approximate <- fastkpc_full_cuda_summary_value(
    skeleton, "approximate_backend_count", NULL
  )
  summaries <- list(skeleton$artifact_summary, skeleton$summary)
  summary_names <- unique(unlist(lapply(summaries, names), use.names = FALSE))
  counter_names <- grep("(fallback|error).*count$", summary_names,
                        value = TRUE)
  counter_names <- setdiff(counter_names, c("unknown_fallback_count",
                                             "approximate_backend_count"))
  counter_values <- vapply(counter_names, function(name) {
    value <- fastkpc_full_cuda_summary_value(skeleton, name, NA_real_)
    if (length(value) != 1L) return(NA_real_)
    suppressWarnings(as.numeric(value))
  }, numeric(1))
  details <- data.frame(
    type = ifelse(grepl("error.*count$", counter_names), "error", "fallback"),
    key = counter_names,
    reason = rep("candidate summary counter", length(counter_names)),
    count = unname(counter_values),
    stringsAsFactors = FALSE
  )
  backend_count <- if (length(counter_values) == 0L) {
    0L
  } else if (any(!is.finite(counter_values))) {
    NA_integer_
  } else {
    as.integer(sum(counter_values))
  }
  list(
    unknown_fallback_count = if (is.null(unknown)) NA_integer_ else
      as.integer(unknown),
    approximate_backend_count = if (is.null(approximate)) NA_integer_ else
      as.integer(approximate),
    backend_fallback_error_count = backend_count,
    details = details
  )
}

fastkpc_full_cuda_empty_first_divergence <- function() {
  list(
    first_divergence_found = FALSE,
    type = NA_character_,
    logical_sequence_id = NA_integer_,
    level = NA_integer_,
    edge_x = NA_integer_,
    edge_y = NA_integer_,
    edge_x_label = NA_character_,
    edge_y_label = NA_character_,
    S = NA_character_,
    reference_p = NA_real_,
    candidate_p = NA_real_,
    reference_decision = NA,
    candidate_decision = NA,
    message = NA_character_
  )
}

fastkpc_full_cuda_first_row_difference <- function(left, right, columns) {
  count <- max(nrow(left), nrow(right))
  if (count == 0L) return(NULL)
  for (i in seq_len(count)) {
    if (i > nrow(left) || i > nrow(right)) return(i)
    lhs <- left[i, columns, drop = FALSE]
    rhs <- right[i, columns, drop = FALSE]
    if (!identical(lhs, rhs)) return(i)
  }
  NULL
}

fastkpc_full_cuda_compare_core <- function(reference, candidate,
                                            reference_deletions = NULL,
                                            reference_logical = NULL) {
  if (!fastkpc_full_cuda_is_skeleton(reference)) {
    stop("reference does not contain a skeleton", call. = FALSE)
  }
  labels <- fastkpc_full_cuda_matrix_labels(reference$adjacency)
  ref_adj <- fastkpc_full_cuda_align_matrix(reference$adjacency, labels)
  ref_sepsets <- fastkpc_full_cuda_normalize_sepsets(reference, labels)
  if (is.null(reference_deletions)) {
    reference_deletions <- fastkpc_full_cuda_normalize_deletion_trace(
      reference, labels
    )
  }
  if (is.null(reference_logical)) {
    reference_logical <- fastkpc_full_cuda_normalize_logical_trace(reference)
  }

  if (!fastkpc_full_cuda_is_skeleton(candidate)) {
    first <- fastkpc_full_cuda_empty_first_divergence()
    first$first_divergence_found <- TRUE
    first$type <- "candidate_missing"
    first$message <- "candidate graph is missing"
    summary <- fastkpc_full_cuda_summary_with_divergence(list(
      run_status = "missing",
      timeout = FALSE,
      source_commit = fastkpc_full_cuda_source_commit(),
      edge_count_reference = fastkpc_full_cuda_edge_count(ref_adj),
      edge_count_candidate = NA_integer_,
      SHD = NA_integer_,
      adjacency_identical = FALSE,
      sepsets_identical = FALSE,
      n_edgetests_identical = FALSE,
      deletions_identical = FALSE,
      logical_ci_trace_identical = NA,
      unknown_fallback_count = NA_integer_,
      approximate_backend_count = NA_integer_,
      backend_fallback_error_count = NA_integer_,
      pass = FALSE
    ), first)
    return(list(
      summary = summary,
      first_divergence = first,
      graph_agreement = data.frame(),
      sepset_agreement = data.frame(),
      n_edgetests = data.frame(),
      reference_deletions = reference_deletions,
      candidate_deletions = fastkpc_full_cuda_empty_deletion_trace(),
      candidate_logical = fastkpc_full_cuda_empty_logical_trace(),
      fallbacks = data.frame(
        type = character(), key = character(), reason = character(),
        count = numeric(), stringsAsFactors = FALSE
      )
    ))
  }

  candidate_adj <- fastkpc_full_cuda_align_matrix(candidate$adjacency, labels)
  adjacency_identical <- !is.null(candidate_adj) &&
    identical(unname(candidate_adj), unname(ref_adj))
  shd <- fastkpc_full_cuda_shd(candidate_adj, ref_adj)

  candidate_sepsets <- fastkpc_full_cuda_normalize_sepsets(candidate, labels)
  sepset_columns <- c("edge_x", "edge_y", "S_key", "direction_conflict")
  sepsets_identical <- identical(
    ref_sepsets[, sepset_columns, drop = FALSE],
    candidate_sepsets[, sepset_columns, drop = FALSE]
  )

  ref_tests <- as.integer(reference$n.edgetests)
  candidate_tests <- as.integer(candidate$n.edgetests)
  n_edgetests_identical <- identical(candidate_tests, ref_tests)
  n_count <- max(length(ref_tests), length(candidate_tests))
  n_edgetests <- data.frame(
    level = seq_len(n_count) - 1L,
    reference = c(ref_tests, rep(NA_integer_, n_count - length(ref_tests))),
    candidate = c(candidate_tests,
                  rep(NA_integer_, n_count - length(candidate_tests)))
  )
  n_edgetests$identical <- with(
    n_edgetests,
    !is.na(reference) & !is.na(candidate) & reference == candidate
  )

  candidate_deletions <- fastkpc_full_cuda_normalize_deletion_trace(
    candidate, labels
  )
  deletion_columns <- c("level", "edge_x", "edge_y", "S_key")
  deletions_identical <- identical(
    reference_deletions[, deletion_columns, drop = FALSE],
    candidate_deletions[, deletion_columns, drop = FALSE]
  )

  candidate_logical <- fastkpc_full_cuda_normalize_logical_trace(candidate)
  logical_columns <- c("level", "source_task_index", "x", "y", "S_key",
                       "deletes_edge")
  logical_ci_trace_identical <- if (nrow(reference_logical) == 0L) {
    NA
  } else if (nrow(candidate_logical) == 0L) {
    FALSE
  } else {
    identical(reference_logical[, logical_columns, drop = FALSE],
              candidate_logical[, logical_columns, drop = FALSE])
  }

  first <- fastkpc_full_cuda_empty_first_divergence()
  if (identical(logical_ci_trace_identical, FALSE)) {
    first$first_divergence_found <- TRUE
    first$type <- "logical_ci_trace"
    first$message <- "candidate logical CI trace differs from oracle"
    mismatch <- fastkpc_full_cuda_first_row_difference(
      reference_logical, candidate_logical, logical_columns
    )
    if (!is.null(mismatch)) {
      reference_row <- if (mismatch <= nrow(reference_logical)) {
        reference_logical[mismatch, , drop = FALSE]
      } else NULL
      candidate_row <- if (mismatch <= nrow(candidate_logical)) {
        candidate_logical[mismatch, , drop = FALSE]
      } else NULL
      source <- if (!is.null(reference_row)) reference_row else candidate_row
      first$logical_sequence_id <- source$logical_sequence_id[[1L]]
      first$level <- source$level[[1L]]
      first$edge_x <- min(source$x[[1L]], source$y[[1L]])
      first$edge_y <- max(source$x[[1L]], source$y[[1L]])
      first$S <- source$S_key[[1L]]
      if (!is.null(reference_row)) {
        first$reference_p <- reference_row$p_value[[1L]]
        first$reference_decision <- reference_row$deletes_edge[[1L]]
      }
      if (!is.null(candidate_row)) {
        first$candidate_p <- candidate_row$p_value[[1L]]
        first$candidate_decision <- candidate_row$deletes_edge[[1L]]
      }
    }
  } else if (!adjacency_identical) {
    first$first_divergence_found <- TRUE
    first$type <- "adjacency"
    first$message <- "candidate adjacency differs from oracle"
    if (!is.null(candidate_adj)) {
      mismatch <- which(upper.tri(ref_adj) & candidate_adj != ref_adj,
                        arr.ind = TRUE)
      if (nrow(mismatch) > 0L) {
        first$edge_x <- mismatch[1L, 1L]
        first$edge_y <- mismatch[1L, 2L]
        first$edge_x_label <- labels[[first$edge_x]]
        first$edge_y_label <- labels[[first$edge_y]]
        first$reference_decision <- isTRUE(ref_adj[first$edge_x,
                                                   first$edge_y])
        first$candidate_decision <- isTRUE(candidate_adj[first$edge_x,
                                                         first$edge_y])
      }
    }
  } else if (!sepsets_identical) {
    first$first_divergence_found <- TRUE
    first$type <- "sepset"
    first$message <- "candidate normalized sepsets differ from oracle"
    mismatch <- fastkpc_full_cuda_first_row_difference(
      ref_sepsets, candidate_sepsets, sepset_columns
    )
    if (!is.null(mismatch)) {
      source <- if (mismatch <= nrow(ref_sepsets)) {
        ref_sepsets[mismatch, , drop = FALSE]
      } else {
        candidate_sepsets[mismatch, , drop = FALSE]
      }
      first$edge_x <- source$edge_x[[1L]]
      first$edge_y <- source$edge_y[[1L]]
      first$edge_x_label <- source$edge_x_label[[1L]]
      first$edge_y_label <- source$edge_y_label[[1L]]
      first$S <- if (mismatch <= nrow(ref_sepsets)) {
        ref_sepsets$S_key[[mismatch]]
      } else NA_character_
    }
  } else if (!n_edgetests_identical) {
    first$first_divergence_found <- TRUE
    first$type <- "n_edgetests"
    first$message <- "candidate logical n.edgetests differ from oracle"
    mismatch <- which(!n_edgetests$identical)[[1L]]
    first$level <- n_edgetests$level[[mismatch]]
  } else if (!deletions_identical) {
    first$first_divergence_found <- TRUE
    first$type <- "deletion_trace"
    first$message <- "candidate canonical deletion trace differs from oracle"
    mismatch <- fastkpc_full_cuda_first_row_difference(
      reference_deletions, candidate_deletions, deletion_columns
    )
    if (!is.null(mismatch)) {
      source <- if (mismatch <= nrow(reference_deletions)) {
        reference_deletions[mismatch, , drop = FALSE]
      } else {
        candidate_deletions[mismatch, , drop = FALSE]
      }
      first$logical_sequence_id <- source$source_sequence_id[[1L]]
      first$level <- source$level[[1L]]
      first$edge_x <- source$edge_x[[1L]]
      first$edge_y <- source$edge_y[[1L]]
      first$edge_x_label <- labels[[first$edge_x]]
      first$edge_y_label <- labels[[first$edge_y]]
      first$S <- source$S_key[[1L]]
      if (mismatch <= nrow(reference_deletions)) {
        first$reference_p <- reference_deletions$p_value[[mismatch]]
      }
      if (mismatch <= nrow(candidate_deletions)) {
        first$candidate_p <- candidate_deletions$p_value[[mismatch]]
      }
    }
  }

  fallback <- fastkpc_full_cuda_candidate_fallbacks(candidate)
  fallback_clean <- identical(fallback$unknown_fallback_count, 0L) &&
    identical(fallback$approximate_backend_count, 0L) &&
    identical(fallback$backend_fallback_error_count, 0L)
  if (!isTRUE(first$first_divergence_found) && !fallback_clean) {
    first$first_divergence_found <- TRUE
    first$type <- "fallback"
    first$message <- paste0(
      "candidate fallback evidence is missing or nonzero: unknown=",
      fallback$unknown_fallback_count, ", approximate=",
      fallback$approximate_backend_count, ", backend=",
      fallback$backend_fallback_error_count
    )
  }
  logical_trace_clean <- nrow(reference_logical) == 0L ||
    isTRUE(logical_ci_trace_identical)
  pass <- adjacency_identical && isTRUE(shd == 0L) &&
    sepsets_identical && n_edgetests_identical && deletions_identical &&
    logical_trace_clean && fallback_clean

  sepset_agreement <- merge(
    ref_sepsets,
    candidate_sepsets,
    by = c("edge_x", "edge_y", "edge_x_label", "edge_y_label"),
    suffixes = c("_reference", "_candidate"),
    all = TRUE,
    sort = TRUE
  )
  sepset_agreement$identical <- with(
    sepset_agreement,
    !is.na(S_key_reference) & !is.na(S_key_candidate) &
      S_key_reference == S_key_candidate &
      !direction_conflict_reference & !direction_conflict_candidate
  )

  graph_agreement <- data.frame(
    edge_count_reference = fastkpc_full_cuda_edge_count(ref_adj),
    edge_count_candidate = fastkpc_full_cuda_edge_count(candidate_adj),
    SHD = shd,
    adjacency_identical = adjacency_identical,
    stringsAsFactors = FALSE
  )
  summary <- fastkpc_full_cuda_summary_with_divergence(list(
    run_status = "ok",
    timeout = FALSE,
    source_commit = fastkpc_full_cuda_source_commit(),
    edge_count_reference = graph_agreement$edge_count_reference[[1L]],
    edge_count_candidate = graph_agreement$edge_count_candidate[[1L]],
    SHD = shd,
    adjacency_identical = adjacency_identical,
    sepsets_identical = sepsets_identical,
    n_edgetests_identical = n_edgetests_identical,
    deletions_identical = deletions_identical,
    logical_ci_trace_identical = logical_ci_trace_identical,
    unknown_fallback_count = fallback$unknown_fallback_count,
    approximate_backend_count = fallback$approximate_backend_count,
    backend_fallback_error_count = fallback$backend_fallback_error_count,
    pass = pass
  ), first)
  list(
    summary = summary,
    first_divergence = first,
    graph_agreement = graph_agreement,
    sepset_agreement = sepset_agreement,
    n_edgetests = n_edgetests,
    reference_deletions = reference_deletions,
    candidate_deletions = candidate_deletions,
    candidate_logical = candidate_logical,
    candidate_adjacency = candidate_adj,
    candidate_sepsets = candidate$sepsets,
    candidate_pmax = candidate$pMax,
    fallbacks = fallback$details
  )
}

fastkpc_full_cuda_command_output <- function(command, args = character()) {
  output <- tryCatch(
    suppressWarnings(system2(command, args, stdout = TRUE, stderr = TRUE)),
    error = function(e) paste0("unavailable: ", conditionMessage(e))
  )
  status <- attr(output, "status", exact = TRUE)
  if (!is.null(status) && !identical(as.integer(status), 0L)) {
    return(paste0("unavailable(status=", as.integer(status), "): ",
                  paste(output, collapse = "\n")))
  }
  paste(output, collapse = "\n")
}

fastkpc_full_cuda_cpu_model <- function() {
  lines <- tryCatch(
    readLines("/proc/cpuinfo", warn = FALSE),
    error = function(e) character()
  )
  match <- grep("^model name[[:space:]]*:", lines, value = TRUE)
  if (length(match) == 0L) return("unavailable")
  trimws(sub("^[^:]+:", "", match[[1L]]))
}

fastkpc_full_cuda_cuda_runtime_version <- function() {
  path <- "/usr/local/cuda/version.json"
  if (file.exists(path) && requireNamespace("jsonlite", quietly = TRUE)) {
    value <- tryCatch(
      jsonlite::read_json(path, simplifyVector = TRUE),
      error = function(e) NULL
    )
    version <- value$cuda_cudart$version
    if (!is.null(version)) return(as.character(version))
  }
  output <- fastkpc_full_cuda_command_output(
    "/usr/local/cuda/bin/nvcc", "--version"
  )
  match <- regmatches(output, regexpr("release [0-9.]+", output))
  if (length(match) == 0L || !nzchar(match)) return("unavailable")
  sub("release ", "", match, fixed = TRUE)
}

fastkpc_full_cuda_thread_counts <- function() {
  env_names <- c(
    "OMP_NUM_THREADS",
    "FASTKPC_NATIVE_LEGACY_MGCV_PROVIDER_CORES",
    "FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH_THREADS",
    "FASTKPC_NATIVE_CUDA_LOWRANK_BATCH_THREADS"
  )
  values <- Sys.getenv(env_names, unset = NA_character_)
  list(
    detected_logical_cores = parallel::detectCores(logical = TRUE),
    detected_physical_cores = parallel::detectCores(logical = FALSE),
    environment = as.list(values)
  )
}

fastkpc_full_cuda_source_commit <- function() {
  output <- fastkpc_full_cuda_command_output("git", c("rev-parse", "HEAD"))
  strsplit(output, "\n", fixed = TRUE)[[1L]][[1L]]
}

fastkpc_full_cuda_environment_lines <- function() {
  c(
    paste0("generated_at_utc=", format(Sys.time(), tz = "UTC", usetz = TRUE)),
    paste0("source_commit=", fastkpc_full_cuda_source_commit()),
    paste0("R=", R.version.string),
    paste0("mgcv=", if (requireNamespace("mgcv", quietly = TRUE)) {
      as.character(utils::packageVersion("mgcv"))
    } else "unavailable"),
    paste0("CPU=", fastkpc_full_cuda_cpu_model()),
    paste0("GPU=", fastkpc_full_cuda_command_output(
      "nvidia-smi", c("--query-gpu=name", "--format=csv,noheader")
    )),
    paste0("CUDA_DRIVER=", fastkpc_full_cuda_command_output(
      "nvidia-smi", c("--query-gpu=driver_version", "--format=csv,noheader")
    )),
    paste0("NVCC=", fastkpc_full_cuda_command_output(
      "/usr/local/cuda/bin/nvcc", "--version"
    )),
    paste0("CXX17=", fastkpc_full_cuda_command_output(
      "R", c("CMD", "config", "CXX17")
    )),
    "",
    capture.output(sessionInfo())
  )
}

fastkpc_full_cuda_write_json <- function(value, path) {
  fastkpc_full_cuda_require_namespace("jsonlite")
  jsonlite::write_json(
    value,
    path,
    auto_unbox = TRUE,
    pretty = TRUE,
    null = "null",
    na = "null",
    digits = NA
  )
}

fastkpc_full_cuda_summary_with_divergence <- function(summary, first) {
  summary$deleting_test_identical <- summary$deletions_identical
  summary$first_divergence_found <-
    isTRUE(first$first_divergence_found)
  summary$first_divergence_level <- first$level
  summary$first_divergence_edge <- if (is.na(first$edge_x) ||
                                       is.na(first$edge_y)) {
    NA_character_
  } else {
    paste(first$edge_x, first$edge_y, sep = "|")
  }
  summary$first_divergence_S <- first$S
  summary$reference_p <- first$reference_p
  summary$candidate_p <- first$candidate_p
  summary$reference_decision <- first$reference_decision
  summary$candidate_decision <- first$candidate_decision
  summary
}

fastkpc_full_cuda_write_summary_md <- function(summary, path, title) {
  value <- function(name) {
    item <- summary[[name]]
    if (is.null(item) || length(item) == 0L || is.na(item[[1L]])) "NA" else
      as.character(item[[1L]])
  }
  writeLines(c(
    paste0("# ", title),
    "",
    paste0("- run status: ", value("run_status")),
    paste0("- timeout: ", value("timeout")),
    paste0("- source commit: ", value("source_commit")),
    paste0("- edge count: ", value("edge_count_candidate"), " / ",
           value("edge_count_reference")),
    paste0("- SHD: ", value("SHD")),
    paste0("- adjacency identical: ", value("adjacency_identical")),
    paste0("- normalized sepsets identical: ", value("sepsets_identical")),
    paste0("- n.edgetests identical: ", value("n_edgetests_identical")),
    paste0("- deletion trace identical: ", value("deletions_identical")),
    paste0("- logical CI trace identical: ",
           value("logical_ci_trace_identical")),
    paste0("- unknown fallback count: ", value("unknown_fallback_count")),
    paste0("- approximate backend count: ",
           value("approximate_backend_count")),
    paste0("- backend fallback/error count: ",
           value("backend_fallback_error_count")),
    paste0("- pass: ", value("pass"))
  ), path)
}

fastkpc_full_cuda_artifact_paths <- function(output_dir) {
  list(
    manifest_json = file.path(output_dir, "manifest.json"),
    summary_json = file.path(output_dir, "summary.json"),
    summary_csv = file.path(output_dir, "summary.csv"),
    summary_md = file.path(output_dir, "summary.md"),
    adjacency_rds = file.path(output_dir, "adjacency.rds"),
    adjacency_csv = file.path(output_dir, "adjacency.csv"),
    sepsets_rds = file.path(output_dir, "sepsets.rds"),
    n_edgetests_csv = file.path(output_dir, "n_edgetests.csv"),
    logical_ci_trace_rds = file.path(output_dir, "logical_ci_trace.rds"),
    deletion_trace_csv = file.path(output_dir, "deletion_trace.csv"),
    pmax_rds = file.path(output_dir, "pmax.rds"),
    near_alpha_csv = file.path(output_dir, "near_alpha_cases.csv"),
    environment_txt = file.path(output_dir, "environment.txt"),
    commands_txt = file.path(output_dir, "commands.txt"),
    graph_agreement_csv = file.path(output_dir, "graph_agreement.csv"),
    sepset_agreement_csv = file.path(output_dir, "sepset_agreement.csv"),
    first_divergence_json = file.path(output_dir, "first_divergence.json"),
    fallbacks_csv = file.path(output_dir, "fallbacks.csv"),
    stage_timing_csv = file.path(output_dir, "stage_timing.csv"),
    raw_runs_csv = file.path(output_dir, "raw_runs.csv")
  )
}

fastkpc_full_cuda_reset_output_dir <- function(output_dir) {
  if (!is.character(output_dir) || length(output_dir) != 1L ||
      is.na(output_dir) || !nzchar(output_dir)) {
    stop("artifact output directory must be a non-empty path",
         call. = FALSE)
  }
  normalized <- normalizePath(output_dir, mustWork = FALSE)
  if (identical(normalized, "/") ||
      identical(normalized, normalizePath(".", mustWork = TRUE))) {
    stop("refusing to reset unsafe artifact output directory: ", output_dir,
         call. = FALSE)
  }
  if (dir.exists(output_dir) || file.exists(output_dir)) {
    unlink(output_dir, recursive = TRUE, force = TRUE)
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  invisible(output_dir)
}

fastkpc_full_cuda_data_hash <- function(data) {
  fastkpc_full_cuda_require_namespace("digest")
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  digest::digest(data, algo = "sha256", serialize = TRUE)
}

fastkpc_full_cuda_file_hash <- function(path) {
  if (!is.character(path) || length(path) != 1L || is.na(path) ||
      !nzchar(path) || !file.exists(path)) {
    return(NA_character_)
  }
  fastkpc_full_cuda_require_namespace("digest")
  digest::digest(file = path, algo = "sha256", serialize = FALSE)
}

fastkpc_full_cuda_graph_hashes <- function(skeleton) {
  if (!fastkpc_full_cuda_is_skeleton(skeleton)) {
    stop("graph hash input does not contain a skeleton", call. = FALSE)
  }
  fastkpc_full_cuda_require_namespace("digest")
  labels <- fastkpc_full_cuda_matrix_labels(skeleton$adjacency)
  adjacency <- fastkpc_full_cuda_align_matrix(skeleton$adjacency, labels)
  if (is.null(adjacency)) {
    stop("graph hash input has invalid adjacency labels", call. = FALSE)
  }
  normalized_sepsets <- fastkpc_full_cuda_normalize_sepsets(skeleton, labels)
  deletion_trace <- fastkpc_full_cuda_normalize_deletion_trace(
    skeleton, labels
  )
  list(
    adjacency_hash = digest::digest(adjacency, algo = "sha256",
                                    serialize = TRUE),
    sepset_hash = digest::digest(normalized_sepsets, algo = "sha256",
                                serialize = TRUE),
    deletion_trace_hash = digest::digest(deletion_trace, algo = "sha256",
                                         serialize = TRUE)
  )
}

fastkpc_full_cuda_canonical_contract <- function() {
  list(
    n = 351L,
    p = 48L,
    data_hash =
      "971f1e0784817c6febfc84154f19540be02a92f6e4acad784dc7f92b979f1df7",
    column_order = c(
      "FOS", "JUN", "SFTPB", "JUNB", "RGS16", "IER2", "C16ORF89",
      "EGR1", "GADD45B", "C4BPA", "SFTPD", "HLA.DPB1", "SELENBP1",
      "SUSD2", "FOSB", "HLA.DRA", "SFTA3", "ATF3", "GDF15",
      "HLA.DRB5", "HOPX", "IGFBP2", "SCGB3A2", "AQP4", "CYP4B1",
      "EFNA1", "TMEM125", "RNASE1", "HLA.DPA1", "SCPEP1", "NAPSA",
      "CD74", "MYLIP", "ALDH2", "SLC22A31", "CTSH", "LGMN",
      "HLA.DRB1", "ALDH3A2", "MIR614", "SLC34A2", "SRSF7", "TXNIP",
      "ANXA1", "PPP1CB", "CTSD", "CXCL17", "HLA.DMA"
    ),
    alpha = 0.1,
    edge_count = 110L,
    adjacency_hash =
      "74c265d3a59878aa144812b94d6df71d5a6f8133b346dcb6d98bc5cb00a80905",
    sepset_hash =
      "e00ee8187612399aa2984f21b2e8ea1ae9db081129357bef4f931085fafc6630",
    deletion_trace_hash =
      "1206443d5aeffb5bc187ac43f0b45072333de56ae0f61328bff6ca45dd49e779",
    source_result_hash =
      "259762d78eb5a0378b312d51040f3c5c3484e0b72dc2afecf9fdc61bcdf81a59",
    n_edgetests = c(2213L, 52659L, 125293L, 40694L, 13293L, 5422L,
                    835L, 80L),
    index = 1L,
    numCol = 35L,
    max_conditioning_size = 46L
  )
}

fastkpc_full_cuda_validate_canonical_fixture <- function(
    data, skeleton, alpha, index, numCol, max_conditioning_size,
    source_result_path = NULL,
    contract = fastkpc_full_cuda_canonical_contract()) {
  if (!fastkpc_full_cuda_is_skeleton(skeleton)) {
    stop("canonical fixture result does not contain a skeleton",
         call. = FALSE)
  }
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  if (!identical(c(nrow(data), ncol(data)),
                 c(as.integer(contract$n), as.integer(contract$p)))) {
    stop("canonical fixture data dimensions do not match contract",
         call. = FALSE)
  }
  if (is.null(colnames(data)) ||
      !identical(as.character(colnames(data)),
                 as.character(contract$column_order))) {
    stop("canonical fixture column order does not match contract",
         call. = FALSE)
  }
  if (!identical(fastkpc_full_cuda_data_hash(data),
                 as.character(contract$data_hash))) {
    stop("canonical fixture data hash does not match contract",
         call. = FALSE)
  }
  adjacency <- fastkpc_full_cuda_align_matrix(
    skeleton$adjacency, as.character(contract$column_order)
  )
  if (is.null(adjacency) ||
      !identical(fastkpc_full_cuda_edge_count(adjacency),
                 as.integer(contract$edge_count))) {
    stop("canonical fixture edge count does not match contract",
         call. = FALSE)
  }
  graph_hashes <- fastkpc_full_cuda_graph_hashes(skeleton)
  graph_contract_fields <- c("adjacency_hash", "sepset_hash",
                             "deletion_trace_hash")
  for (name in graph_contract_fields) {
    if (!identical(graph_hashes[[name]], as.character(contract[[name]]))) {
      stop("canonical fixture ", sub("_", " ", name, fixed = TRUE),
           " does not match contract", call. = FALSE)
    }
  }
  if (!is.null(contract$source_result_hash)) {
    source_hash <- fastkpc_full_cuda_file_hash(source_result_path)
    if (!identical(source_hash, as.character(contract$source_result_hash))) {
      stop("canonical fixture source result hash does not match contract",
           call. = FALSE)
    }
  }
  if (!identical(as.integer(skeleton$n.edgetests),
                 as.integer(contract$n_edgetests))) {
    stop("canonical fixture n.edgetests do not match contract",
         call. = FALSE)
  }
  scalar_checks <- list(
    alpha = c(as.numeric(alpha), as.numeric(contract$alpha)),
    index = c(as.integer(index), as.integer(contract$index)),
    numCol = c(as.integer(numCol), as.integer(contract$numCol)),
    max_conditioning_size = c(as.integer(max_conditioning_size),
                              as.integer(contract$max_conditioning_size))
  )
  for (name in names(scalar_checks)) {
    values <- scalar_checks[[name]]
    if (length(values) != 2L || anyNA(values) || values[[1L]] != values[[2L]]) {
      stop("canonical fixture ", name, " does not match contract",
           call. = FALSE)
    }
  }
  TRUE
}

fastkpc_write_full_cuda_ci_oracle <- function(
    reference, data, output_dir, alpha, index, numCol,
    max_conditioning_size, source_result_path = NA_character_,
    logical_trace_source = NULL,
    logical_trace_source_path = NA_character_,
    oracle_route_environment = character(), commands = character(),
    near_alpha_threshold = 1e-6) {
  reference <- fastkpc_full_cuda_extract_skeleton(reference, role = "oracle")
  if (!fastkpc_full_cuda_is_skeleton(reference)) {
    stop("oracle input does not contain a skeleton", call. = FALSE)
  }
  reference_fallback <- fastkpc_full_cuda_candidate_fallbacks(reference)
  oracle_unknown_fallback_count <-
    if (is.na(reference_fallback$unknown_fallback_count)) 0L else
      reference_fallback$unknown_fallback_count
  oracle_approximate_backend_count <-
    if (is.na(reference_fallback$approximate_backend_count)) 0L else
      reference_fallback$approximate_backend_count
  oracle_backend_fallback_error_count <-
    reference_fallback$backend_fallback_error_count
  if (!identical(oracle_unknown_fallback_count, 0L) ||
      !identical(oracle_approximate_backend_count, 0L) ||
      !identical(oracle_backend_fallback_error_count, 0L)) {
    stop("oracle fallback evidence is missing or nonzero: unknown=",
         oracle_unknown_fallback_count, ", approximate=",
         oracle_approximate_backend_count, ", backend=",
         oracle_backend_fallback_error_count, call. = FALSE)
  }
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  labels <- fastkpc_full_cuda_matrix_labels(reference$adjacency)
  if (ncol(data) != length(labels)) {
    stop("oracle data column count does not match adjacency", call. = FALSE)
  }
  if (is.null(colnames(data))) colnames(data) <- labels
  data <- data[, labels, drop = FALSE]
  fastkpc_full_cuda_reset_output_dir(output_dir)
  paths <- fastkpc_full_cuda_artifact_paths(output_dir)

  adjacency <- fastkpc_full_cuda_align_matrix(reference$adjacency, labels)
  normalized_sepsets <- fastkpc_full_cuda_normalize_sepsets(reference, labels)
  deletion_trace <- fastkpc_full_cuda_normalize_deletion_trace(reference,
                                                               labels)
  logical_trace <- fastkpc_full_cuda_normalize_logical_trace(reference)
  if (!is.null(logical_trace_source)) {
    trace_skeleton <- fastkpc_full_cuda_extract_skeleton(
      logical_trace_source,
      role = "candidate"
    )
    trace_qualification <- fastkpc_full_cuda_compare_core(
      reference = reference,
      candidate = trace_skeleton,
      reference_deletions = deletion_trace,
      reference_logical = fastkpc_full_cuda_empty_logical_trace()
    )
    if (!isTRUE(trace_qualification$summary$pass)) {
      stop("logical trace source failed the oracle graph gate: ",
           trace_qualification$first_divergence$type, call. = FALSE)
    }
    qualified_trace <- fastkpc_full_cuda_normalize_logical_trace(
      trace_skeleton
    )
    if (nrow(qualified_trace) == 0L) {
      stop("logical trace source contains no consumed CI tasks",
           call. = FALSE)
    }
    logical_trace <- qualified_trace
  }
  fastkpc_full_cuda_validate_logical_trace(
    logical_trace, reference$n.edgetests, role = "oracle"
  )
  fastkpc_full_cuda_validate_logical_trace_plan(
    logical_trace,
    p = length(labels),
    level_count = length(reference$n.edgetests),
    expected_adjacency = adjacency,
    role = "oracle"
  )
  n_edgetests <- data.frame(
    level = seq_along(reference$n.edgetests) - 1L,
    n_edgetests = as.integer(reference$n.edgetests)
  )
  near_alpha <- logical_trace[
    is.finite(logical_trace$p_value) &
      abs(logical_trace$p_value - alpha) <= near_alpha_threshold,
    , drop = FALSE
  ]
  if (nrow(near_alpha) > 0L) {
    near_alpha$distance_from_alpha <- abs(near_alpha$p_value - alpha)
  } else {
    near_alpha$distance_from_alpha <- numeric()
  }

  environment_lines <- fastkpc_full_cuda_environment_lines()
  manifest <- list(
    schema_version = "full-cuda-ci-oracle-v1",
    generated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
    data_hash = fastkpc_full_cuda_data_hash(data),
    data_dimensions = list(n = nrow(data), p = ncol(data)),
    column_order = colnames(data),
    alpha = as.numeric(alpha),
    max_conditioning_size = as.integer(max_conditioning_size),
    index = as.integer(index),
    numCol = as.integer(numCol),
    R_version = R.version.string,
    mgcv_version = if (requireNamespace("mgcv", quietly = TRUE)) {
      as.character(utils::packageVersion("mgcv"))
    } else NA_character_,
    package_session_information = paste(capture.output(sessionInfo()),
                                        collapse = "\n"),
    compiler_versions = list(
      CXX17 = fastkpc_full_cuda_command_output(
        "R", c("CMD", "config", "CXX17")
      ),
      CXX17_version = fastkpc_full_cuda_command_output(
        fastkpc_full_cuda_command_output(
          "R", c("CMD", "config", "CXX17")
        ),
        "--version"
      ),
      NVCC = fastkpc_full_cuda_command_output(
        "/usr/local/cuda/bin/nvcc", "--version"
      )
    ),
    cuda_driver_version = strsplit(
      fastkpc_full_cuda_command_output(
        "nvidia-smi",
        c("--query-gpu=driver_version", "--format=csv,noheader")
      ),
      "\n",
      fixed = TRUE
    )[[1L]],
    cuda_runtime_version = fastkpc_full_cuda_cuda_runtime_version(),
    gpu_model = strsplit(
      fastkpc_full_cuda_command_output(
        "nvidia-smi", c("--query-gpu=name", "--format=csv,noheader")
      ),
      "\n",
      fixed = TRUE
    )[[1L]],
    cpu_model = fastkpc_full_cuda_cpu_model(),
    thread_counts = fastkpc_full_cuda_thread_counts(),
    source_commit = fastkpc_full_cuda_source_commit(),
    source_result_path = source_result_path,
    source_result_hash = fastkpc_full_cuda_file_hash(source_result_path),
    logical_ci_trace_source_path = logical_trace_source_path,
    logical_ci_trace_source_hash = fastkpc_full_cuda_file_hash(
      logical_trace_source_path
    ),
    oracle_route_environment = as.list(oracle_route_environment),
    logical_ci_trace_available = nrow(logical_trace) > 0L,
    logical_ci_trace_count = nrow(logical_trace),
    deletion_trace_count = nrow(deletion_trace),
    environment_file = basename(paths$environment_txt)
  )
  summary <- list(
    run_status = "ok",
    timeout = FALSE,
    source_commit = manifest$source_commit,
    oracle_artifact = normalizePath(output_dir, mustWork = FALSE),
    candidate_route = "oracle-self",
    edge_count_reference = fastkpc_full_cuda_edge_count(adjacency),
    edge_count_candidate = fastkpc_full_cuda_edge_count(adjacency),
    SHD = 0L,
    adjacency_identical = TRUE,
    sepsets_identical = !any(normalized_sepsets$direction_conflict),
    n_edgetests_identical = TRUE,
    deletions_identical = TRUE,
    logical_ci_trace_identical = if (nrow(logical_trace) > 0L) TRUE else NA,
    unknown_fallback_count = oracle_unknown_fallback_count,
    approximate_backend_count = oracle_approximate_backend_count,
    backend_fallback_error_count = oracle_backend_fallback_error_count,
    elapsed_sec = NA_real_,
    pass = !any(normalized_sepsets$direction_conflict)
  )
  first <- fastkpc_full_cuda_empty_first_divergence()
  summary <- fastkpc_full_cuda_summary_with_divergence(summary, first)

  saveRDS(adjacency, paths$adjacency_rds)
  utils::write.csv(adjacency, paths$adjacency_csv, row.names = TRUE)
  saveRDS(reference$sepsets, paths$sepsets_rds)
  utils::write.csv(n_edgetests, paths$n_edgetests_csv, row.names = FALSE)
  saveRDS(logical_trace, paths$logical_ci_trace_rds)
  utils::write.csv(deletion_trace, paths$deletion_trace_csv,
                   row.names = FALSE)
  saveRDS(reference$pMax, paths$pmax_rds)
  utils::write.csv(near_alpha, paths$near_alpha_csv, row.names = FALSE)
  writeLines(environment_lines, paths$environment_txt)
  writeLines(commands, paths$commands_txt)
  fastkpc_full_cuda_write_json(manifest, paths$manifest_json)
  fastkpc_full_cuda_write_json(summary, paths$summary_json)
  utils::write.csv(as.data.frame(summary, stringsAsFactors = FALSE),
                   paths$summary_csv, row.names = FALSE)
  fastkpc_full_cuda_write_summary_md(summary, paths$summary_md,
                                     "Full CUDA CI oracle 351x48 v1")
  utils::write.csv(data.frame(
    edge_count_reference = summary$edge_count_reference,
    edge_count_candidate = summary$edge_count_candidate,
    SHD = 0L,
    adjacency_identical = TRUE
  ), paths$graph_agreement_csv, row.names = FALSE)
  sepset_self <- normalized_sepsets
  sepset_self$identical <- !sepset_self$direction_conflict
  utils::write.csv(sepset_self, paths$sepset_agreement_csv,
                   row.names = FALSE)
  fastkpc_full_cuda_write_json(first, paths$first_divergence_json)
  oracle_fallbacks <- rbind(data.frame(
    type = c("unknown", "approximate"),
    key = c("unknown_fallback_count", "approximate_backend_count"),
    reason = c("oracle summary counter", "oracle summary counter"),
    count = c(oracle_unknown_fallback_count,
              oracle_approximate_backend_count),
    stringsAsFactors = FALSE
  ), reference_fallback$details)
  utils::write.csv(oracle_fallbacks, paths$fallbacks_csv, row.names = FALSE)
  utils::write.csv(data.frame(
    stage = character(), elapsed_ms = numeric(), stringsAsFactors = FALSE
  ), paths$stage_timing_csv, row.names = FALSE)
  utils::write.csv(data.frame(
    run = integer(), elapsed_sec = numeric(), pass = logical(),
    stringsAsFactors = FALSE
  ), paths$raw_runs_csv, row.names = FALSE)

  list(
    summary = summary,
    manifest = manifest,
    reference = reference,
    deletion_trace = deletion_trace,
    logical_trace = logical_trace,
    paths = paths,
    output_dir = output_dir
  )
}

fastkpc_load_full_cuda_ci_oracle <- function(output_dir) {
  paths <- fastkpc_full_cuda_artifact_paths(output_dir)
  required <- c("manifest_json", "summary_json", "adjacency_rds",
                "sepsets_rds", "n_edgetests_csv", "deletion_trace_csv",
                "logical_ci_trace_rds", "pmax_rds")
  missing <- required[!vapply(paths[required], file.exists, logical(1))]
  if (length(missing) > 0L) {
    stop("oracle artifact is incomplete: ", paste(missing, collapse = ","),
         call. = FALSE)
  }
  fastkpc_full_cuda_require_namespace("jsonlite")
  n_edgetests <- utils::read.csv(paths$n_edgetests_csv,
                                 stringsAsFactors = FALSE)
  reference <- list(
    adjacency = readRDS(paths$adjacency_rds),
    sepsets = readRDS(paths$sepsets_rds),
    n.edgetests = as.integer(n_edgetests$n_edgetests),
    pMax = readRDS(paths$pmax_rds)
  )
  logical_trace <- readRDS(paths$logical_ci_trace_rds)
  fastkpc_full_cuda_validate_logical_trace(
    logical_trace, reference$n.edgetests, role = "loaded oracle"
  )
  fastkpc_full_cuda_validate_logical_trace_plan(
    logical_trace,
    p = nrow(reference$adjacency),
    level_count = length(reference$n.edgetests),
    expected_adjacency = reference$adjacency,
    role = "loaded oracle"
  )
  list(
    manifest = jsonlite::read_json(paths$manifest_json,
                                   simplifyVector = TRUE),
    summary = jsonlite::read_json(paths$summary_json,
                                  simplifyVector = TRUE),
    reference = reference,
    deletion_trace = utils::read.csv(paths$deletion_trace_csv,
                                     stringsAsFactors = FALSE),
    logical_trace = logical_trace,
    paths = paths,
    output_dir = output_dir
  )
}

fastkpc_full_cuda_compare_candidate_skeleton <- function(oracle, candidate) {
  fastkpc_full_cuda_compare_core(
    reference = oracle$reference,
    candidate = candidate,
    reference_deletions = oracle$deletion_trace,
    reference_logical = oracle$logical_trace
  )
}

fastkpc_compare_full_cuda_ci_candidate <- function(
    oracle, candidate, output_dir, candidate_route = "candidate",
    commands = character()) {
  oracle <- if (is.character(oracle) && length(oracle) == 1L) {
    fastkpc_load_full_cuda_ci_oracle(oracle)
  } else {
    oracle
  }
  if (is.null(oracle$reference)) {
    stop("oracle object is missing reference skeleton", call. = FALSE)
  }
  candidate <- fastkpc_full_cuda_extract_skeleton(candidate,
                                                  role = "candidate")
  comparison <- fastkpc_full_cuda_compare_candidate_skeleton(
    oracle = oracle, candidate = candidate
  )
  elapsed <- if (fastkpc_full_cuda_is_skeleton(candidate)) {
    fastkpc_full_cuda_summary_value(candidate, "elapsed_sec", NA_real_)
  } else NA_real_
  comparison$summary$oracle_artifact <- oracle$output_dir
  comparison$summary$candidate_route <- candidate_route
  comparison$summary$elapsed_sec <- as.numeric(elapsed)

  fastkpc_full_cuda_reset_output_dir(output_dir)
  paths <- fastkpc_full_cuda_artifact_paths(output_dir)
  writeLines(fastkpc_full_cuda_environment_lines(), paths$environment_txt)
  writeLines(commands, paths$commands_txt)
  fastkpc_full_cuda_write_json(comparison$summary, paths$summary_json)
  utils::write.csv(as.data.frame(comparison$summary,
                                 stringsAsFactors = FALSE),
                   paths$summary_csv, row.names = FALSE)
  fastkpc_full_cuda_write_summary_md(
    comparison$summary, paths$summary_md,
    paste0("Full CUDA CI comparison: ", candidate_route)
  )
  fastkpc_full_cuda_write_json(comparison$first_divergence,
                               paths$first_divergence_json)
  utils::write.csv(comparison$graph_agreement,
                   paths$graph_agreement_csv, row.names = FALSE)
  utils::write.csv(comparison$sepset_agreement,
                   paths$sepset_agreement_csv, row.names = FALSE)
  utils::write.csv(comparison$n_edgetests,
                   paths$n_edgetests_csv, row.names = FALSE)
  utils::write.csv(comparison$candidate_deletions,
                   paths$deletion_trace_csv, row.names = FALSE)
  saveRDS(comparison$candidate_logical, paths$logical_ci_trace_rds)
  alpha <- suppressWarnings(as.numeric(oracle$manifest$alpha))
  near_alpha <- comparison$candidate_logical[0, , drop = FALSE]
  if (length(alpha) == 1L && is.finite(alpha) &&
      nrow(comparison$candidate_logical) > 0L) {
    distance <- abs(comparison$candidate_logical$p_value - alpha)
    keep <- is.finite(distance) & distance <= 1e-6
    near_alpha <- comparison$candidate_logical[keep, , drop = FALSE]
    near_alpha$distance_from_alpha <- distance[keep]
  } else {
    near_alpha$distance_from_alpha <- numeric()
  }
  utils::write.csv(near_alpha, paths$near_alpha_csv, row.names = FALSE)
  if (!is.null(comparison$candidate_adjacency)) {
    saveRDS(comparison$candidate_adjacency, paths$adjacency_rds)
    utils::write.csv(comparison$candidate_adjacency,
                     paths$adjacency_csv, row.names = TRUE)
  }
  if (!is.null(comparison$candidate_sepsets)) {
    saveRDS(comparison$candidate_sepsets, paths$sepsets_rds)
  }
  if (!is.null(comparison$candidate_pmax)) {
    saveRDS(comparison$candidate_pmax, paths$pmax_rds)
  }
  fallbacks <- rbind(data.frame(
    type = c("unknown", "approximate"),
    key = c("unknown_fallback_count", "approximate_backend_count"),
    reason = c("summary counter", "summary counter"),
    count = c(comparison$summary$unknown_fallback_count,
              comparison$summary$approximate_backend_count),
    stringsAsFactors = FALSE
  ), comparison$fallbacks)
  utils::write.csv(fallbacks, paths$fallbacks_csv, row.names = FALSE)
  utils::write.csv(data.frame(
    stage = character(), elapsed_ms = numeric(), stringsAsFactors = FALSE
  ), paths$stage_timing_csv, row.names = FALSE)
  utils::write.csv(data.frame(
    run = 1L,
    elapsed_sec = comparison$summary$elapsed_sec,
    pass = comparison$summary$pass
  ), paths$raw_runs_csv, row.names = FALSE)
  manifest <- list(
    schema_version = "full-cuda-ci-comparison-v1",
    generated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
    source_commit = fastkpc_full_cuda_source_commit(),
    oracle_artifact = oracle$output_dir,
    candidate_route = candidate_route,
    oracle_data_hash = oracle$manifest$data_hash,
    data_dimensions = oracle$manifest$data_dimensions,
    column_order = oracle$manifest$column_order,
    alpha = oracle$manifest$alpha,
    max_conditioning_size = oracle$manifest$max_conditioning_size,
    index = oracle$manifest$index,
    numCol = oracle$manifest$numCol,
    oracle_source_commit = oracle$manifest$source_commit
  )
  fastkpc_full_cuda_write_json(manifest, paths$manifest_json)

  comparison$paths <- paths
  comparison$output_dir <- output_dir
  comparison
}
