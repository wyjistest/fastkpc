fastkpc_full_cuda_phase10_coalescing_require <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
  invisible(TRUE)
}

fastkpc_full_cuda_phase10_coalescing_default_evidence <- function() {
  file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "phase10_profile_v2",
    "fresh-data-level-07-prefill-attribution-profile-v5-development.rds"
  )
}

fastkpc_full_cuda_phase10_coalescing_default_output <- function() {
  file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "phase10_profile_v2",
    "setup-aware-coalescing-opportunity-v1.rds"
  )
}

fastkpc_full_cuda_phase10_coalescing_cohort_key <- function(level, s_key) {
  paste0("level=", as.integer(level), "\nS=", as.character(s_key), "\n")
}

fastkpc_full_cuda_phase10_coalescing_q <- function(conditioning_size) {
  conditioning_size <- as.integer(conditioning_size)
  ifelse(
    conditioning_size == 1L, 10L,
    ifelse(conditioning_size == 2L, 30L, 1L + 9L * conditioning_size)
  )
}

fastkpc_full_cuda_phase10_coalescing_plan_frame <- function(plan) {
  tasks <- plan$tasks
  count <- length(tasks)
  data.frame(
    task_index = seq_len(count),
    edge_x = vapply(tasks, `[[`, integer(1L), "edge_x"),
    edge_y = vapply(tasks, `[[`, integer(1L), "edge_y"),
    x = vapply(tasks, `[[`, integer(1L), "x"),
    y = vapply(tasks, `[[`, integer(1L), "y"),
    S_key = vapply(tasks, `[[`, character(1L), "S_key"),
    conditioning_size = vapply(
      tasks, `[[`, integer(1L), "conditioning_size"
    ),
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_phase10_coalescing_consumed_targets <- function(tasks) {
  tasks <- tasks[tasks$level > 0L, , drop = FALSE]
  cohort <- fastkpc_full_cuda_phase10_coalescing_cohort_key(
    tasks$level, tasks$S_key
  )
  split(
    c(as.integer(tasks$x), as.integer(tasks$y)),
    c(cohort, cohort),
    drop = TRUE
  ) |>
    lapply(function(targets) sort(unique(as.integer(targets))))
}

fastkpc_full_cuda_phase10_coalescing_level_cohorts <- function(
    plan_tasks, level, consumed_targets) {
  if (level == 0L || nrow(plan_tasks) == 0L) return(data.frame())
  s_keys <- sort(unique(plan_tasks$S_key), method = "radix")
  split_targets <- split(
    c(plan_tasks$x, plan_tasks$y),
    c(plan_tasks$S_key, plan_tasks$S_key),
    drop = TRUE
  )
  target_vectors <- lapply(s_keys, function(s_key) {
    sort(unique(as.integer(split_targets[[s_key]])))
  })
  conditioning_sizes <- vapply(s_keys, function(s_key) {
    unique_size <- unique(plan_tasks$conditioning_size[
      plan_tasks$S_key == s_key
    ])
    fastkpc_full_cuda_phase10_coalescing_require(
      length(unique_size) == 1L,
      "Phase 10 coalescing cohort mixed conditioning sizes"
    )
    as.integer(unique_size)
  }, integer(1L))
  cohort_keys <- fastkpc_full_cuda_phase10_coalescing_cohort_key(
    level, s_keys
  )
  consumed_vectors <- lapply(cohort_keys, function(key) {
    value <- consumed_targets[[key]]
    if (is.null(value)) integer() else as.integer(value)
  })
  for (index in seq_along(target_vectors)) {
    fastkpc_full_cuda_phase10_coalescing_require(
      all(consumed_vectors[[index]] %in% target_vectors[[index]]),
      "Phase 10 consumed TargetKey is outside its v5 setup cohort"
    )
  }
  cohort_index <- seq_along(s_keys)
  value <- data.frame(
    level = rep.int(as.integer(level), length(s_keys)),
    penalty_class = if (level <= 2L) "single" else "multi",
    cohort_index = cohort_index,
    window_id = as.integer((cohort_index - 1L) %/% 64L + 1L),
    setup_position_in_window = as.integer((cohort_index - 1L) %% 64L + 1L),
    setup_cohort_key = cohort_keys,
    S_key = s_keys,
    conditioning_size = conditioning_sizes,
    coefficient_dim = fastkpc_full_cuda_phase10_coalescing_q(
      conditioning_sizes
    ),
    penalty_count = ifelse(conditioning_sizes <= 2L, 1L, conditioning_sizes),
    target_count = lengths(target_vectors),
    consumed_target_count = lengths(consumed_vectors),
    stringsAsFactors = FALSE
  )
  value$unconsumed_target_count <-
    value$target_count - value$consumed_target_count
  value$ever_demanded <- value$consumed_target_count > 0L
  value$target_vector <- I(target_vectors)
  value$consumed_target_vector <- I(consumed_vectors)
  value$unconsumed_target_vector <- I(Map(
    setdiff, target_vectors, consumed_vectors
  ))
  value$static_target_key_vector <- I(Map(function(key, targets) {
    paste0(key, "target=", targets, "\n")
  }, cohort_keys, target_vectors))
  value$static_optimizer_state_key_vector <- I(lapply(
    value$static_target_key_vector, function(keys) {
      paste0("schema=static-v5-optimizer-state-opportunity-v1\n", keys)
    }
  ))
  value$first_ready_epoch <- NA_integer_
  value$first_demand_epoch <- NA_integer_
  value
}

fastkpc_full_cuda_phase10_coalescing_validate_trace_level <- function(
    observed, plan_tasks, level, alpha) {
  fastkpc_full_cuda_phase10_coalescing_require(
    length(unique(observed$task_index)) == nrow(observed) &&
      all(observed$task_index >= 1L) &&
      all(observed$task_index <= nrow(plan_tasks)),
    "Phase 10 coalescing trace task indices are malformed"
  )
  expected <- plan_tasks[observed$task_index, , drop = FALSE]
  structural <- c("edge_x", "edge_y", "x", "y", "S_key",
                  "conditioning_size")
  fastkpc_full_cuda_phase10_coalescing_require(
    all(vapply(structural, function(field) {
      identical(expected[[field]], observed[[field]])
    }, logical(1L))) &&
      identical(
        as.logical(observed$native_edge_deleted),
        as.logical(observed$p_used >= alpha)
      ) && all(is.finite(observed$p_used)),
    paste0("Phase 10 coalescing trace changed at level ", level)
  )
  invisible(TRUE)
}

fastkpc_full_cuda_phase10_coalescing_replay_level <- function(
    plan_tasks, observed, adjacency, cohorts, level, alpha) {
  p <- nrow(adjacency)
  task_count <- nrow(plan_tasks)
  fastkpc_full_cuda_phase10_coalescing_validate_trace_level(
    observed, plan_tasks, level, alpha
  )
  p_values <- rep.int(NA_real_, task_count)
  p_values[observed$task_index] <- observed$p_used
  edge_id <- (plan_tasks$edge_x - 1L) * p + (plan_tasks$edge_y - 1L)
  edge_tasks <- split(plan_tasks$task_index, edge_id, drop = TRUE)
  edge_ids <- sort(unique(edge_id))
  edge_positions <- integer(p * p)
  direct_key <- "\001direct"
  scheduler_key <- function(s_key) ifelse(nzchar(s_key), s_key, direct_key)
  ready <- list()
  append_ready <- function(key, edge) {
    prior <- ready[[key]]
    ready[[key]] <<- if (is.null(prior)) edge else c(prior, edge)
  }
  current_task <- function(edge) {
    tasks <- edge_tasks[[as.character(edge)]]
    tasks[[edge_positions[[edge + 1L]] + 1L]]
  }
  for (edge in edge_ids) {
    task_index <- current_task(edge)
    append_ready(scheduler_key(plan_tasks$S_key[[task_index]]), edge)
  }

  cohort_by_s <- if (nrow(cohorts) == 0L) integer() else
    stats::setNames(seq_len(nrow(cohorts)), cohorts$S_key)
  window_by_s <- if (nrow(cohorts) == 0L) integer() else
    stats::setNames(cohorts$window_id, cohorts$S_key)
  first_ready <- rep.int(NA_integer_, nrow(cohorts))
  first_demand <- rep.int(NA_integer_, nrow(cohorts))
  seen_ready <- character()
  demand_activated_windows <- rep.int(
    FALSE, if (nrow(cohorts) == 0L) 0L else max(cohorts$window_id)
  )
  watermark <- 0L
  epoch_rows <- list()
  epoch <- 0L
  consumed_task_indices <- integer()
  delete_edges <- matrix(FALSE, p, p)

  # Replay v5's ordered largest-bucket frontier using only stored p-values.
  while (length(ready) > 0L) {
    epoch <- epoch + 1L
    ready_keys <- sort(names(ready), method = "radix")
    ready_sizes <- vapply(ready[ready_keys], length, integer(1L))
    selected_key <- ready_keys[[which(ready_sizes == max(ready_sizes))[[1L]]]]
    selected_edges <- ready[[selected_key]]
    ready[[selected_key]] <- NULL
    selected_s_key <- if (identical(selected_key, direct_key)) "" else
      selected_key

    ready_s_keys <- setdiff(ready_keys, direct_key)
    newly_ready <- setdiff(ready_s_keys, seen_ready)
    if (length(newly_ready) > 0L) {
      indices <- unname(cohort_by_s[newly_ready])
      fastkpc_full_cuda_phase10_coalescing_require(
        all(!is.na(indices)),
        "Phase 10 frontier exposed a setup outside the v5 cohort plan"
      )
      first_ready[indices[is.na(first_ready[indices])]] <- epoch
      seen_ready <- union(seen_ready, newly_ready)
    }

    selected_cohort <- if (nzchar(selected_s_key))
      unname(cohort_by_s[[selected_s_key]]) else NA_integer_
    selected_window <- if (nzchar(selected_s_key))
      unname(window_by_s[[selected_s_key]]) else NA_integer_
    first_setup_demand <- !is.na(selected_cohort) &&
      is.na(first_demand[[selected_cohort]])
    if (first_setup_demand) first_demand[[selected_cohort]] <- epoch

    ready_indices <- if (length(ready_s_keys) == 0L) integer() else
      unname(cohort_by_s[ready_s_keys])
    ready_in_active <- if (length(ready_indices) == 0L) 0L else sum(
      demand_activated_windows[cohorts$window_id[ready_indices]]
    )
    ready_inactive <- length(ready_indices) - ready_in_active
    activated_setup_count <- if (nrow(cohorts) == 0L) 0L else sum(
      demand_activated_windows[cohorts$window_id]
    )
    pending_setup_count <- nrow(cohorts) - activated_setup_count
    pending_window_count <- sum(!demand_activated_windows)
    demand_new_window <- !is.na(selected_window) &&
      !demand_activated_windows[[selected_window]]
    demand_new_setup_count <- 0L
    demand_new_target_count <- 0L
    if (demand_new_window) {
      in_window <- cohorts$window_id == selected_window
      demand_new_setup_count <- sum(in_window)
      demand_new_target_count <- sum(cohorts$target_count[in_window])
      demand_activated_windows[[selected_window]] <- TRUE
    }
    watermark_new_windows <- if (is.na(selected_window) ||
        selected_window <= watermark) 0L else selected_window - watermark
    if (!is.na(selected_window)) watermark <- max(watermark, selected_window)

    selected_tasks <- vapply(selected_edges, current_task, integer(1L))
    selected_p <- p_values[selected_tasks]
    fastkpc_full_cuda_phase10_coalescing_require(
      all(is.finite(selected_p)),
      paste0("Phase 10 frontier replay reached an unrecorded task at level ",
             level)
    )
    consumed_task_indices <- c(consumed_task_indices, selected_tasks)
    for (position in seq_along(selected_edges)) {
      edge <- selected_edges[[position]]
      task_index <- selected_tasks[[position]]
      edge_positions[[edge + 1L]] <- edge_positions[[edge + 1L]] + 1L
      deleted <- selected_p[[position]] >= alpha
      if (deleted) {
        x <- plan_tasks$edge_x[[task_index]]
        y <- plan_tasks$edge_y[[task_index]]
        delete_edges[x, y] <- TRUE
        delete_edges[y, x] <- TRUE
        edge_positions[[edge + 1L]] <- length(
          edge_tasks[[as.character(edge)]]
        )
      } else if (edge_positions[[edge + 1L]] < length(
          edge_tasks[[as.character(edge)]])) {
        next_index <- current_task(edge)
        append_ready(scheduler_key(plan_tasks$S_key[[next_index]]), edge)
      }
    }

    epoch_rows[[length(epoch_rows) + 1L]] <- data.frame(
      level = as.integer(level),
      epoch = epoch,
      ready_setup_count = length(ready_s_keys),
      newly_ready_setup_count = length(newly_ready),
      previously_ready_setup_count = length(ready_s_keys) -
        length(newly_ready),
      activated_setup_cohort_count = activated_setup_count,
      pending_setup_cohort_count = pending_setup_count,
      pending_window_count = pending_window_count,
      ready_in_activated_window_count = ready_in_active,
      ready_in_inactive_window_count = ready_inactive,
      coalescible_full_window_count = ready_inactive %/% 64L,
      coalescible_tail_setup_count = ready_inactive %% 64L,
      selected_S_key = selected_s_key,
      selected_setup_first_demand = first_setup_demand,
      selected_window_id = selected_window,
      selected_group_task_count = length(selected_tasks),
      demand_order_window_activation_count = as.integer(demand_new_window),
      demand_order_activated_setup_count = demand_new_setup_count,
      demand_order_activated_target_count = demand_new_target_count,
      original_order_watermark_activation_count = watermark_new_windows,
      stringsAsFactors = FALSE
    )
  }

  fastkpc_full_cuda_phase10_coalescing_require(
    identical(sort(consumed_task_indices), sort(observed$task_index)),
    paste0("Phase 10 frontier replay coverage changed at level ", level)
  )
  adjacency[delete_edges] <- 0L
  if (nrow(cohorts) > 0L) {
    cohorts$first_ready_epoch <- first_ready
    cohorts$first_demand_epoch <- first_demand
    fastkpc_full_cuda_phase10_coalescing_require(
      identical(cohorts$ever_demanded, !is.na(cohorts$first_demand_epoch)),
      paste0("Phase 10 setup demand identity changed at level ", level)
    )
  }
  list(
    adjacency = adjacency,
    cohorts = cohorts,
    epochs = if (length(epoch_rows) == 0L) data.frame() else
      do.call(rbind, epoch_rows)
  )
}

fastkpc_full_cuda_phase10_coalescing_window_frame <- function(cohorts) {
  keys <- unique(paste(cohorts$level, cohorts$window_id, sep = ":"))
  rows <- lapply(keys, function(key) {
    fields <- strsplit(key, ":", fixed = TRUE)[[1L]]
    level <- as.integer(fields[[1L]])
    window_id <- as.integer(fields[[2L]])
    selected <- cohorts$level == level & cohorts$window_id == window_id
    demanded_epochs <- cohorts$first_demand_epoch[selected]
    value <- data.frame(
      level = level,
      penalty_class = unique(cohorts$penalty_class[selected]),
      window_id = window_id,
      conditioning_group_count = sum(selected),
      optimizer_setup_count = sum(selected),
      target_optimization_count = sum(cohorts$target_count[selected]),
      unique_target_key_count = sum(cohorts$target_count[selected]),
      consumed_unique_target_key_count = sum(
        cohorts$consumed_target_count[selected]
      ),
      unconsumed_unique_target_key_count = sum(
        cohorts$unconsumed_target_count[selected]
      ),
      ever_demanded = any(cohorts$ever_demanded[selected]),
      first_demand_epoch = if (all(is.na(demanded_epochs))) NA_integer_ else
        min(demanded_epochs, na.rm = TRUE),
      singleton_skipped_request_count = 0L,
      singleton_skipped_target_count = 0L,
      stringsAsFactors = FALSE
    )
    value$setup_cohort_vector <- I(list(
      cohorts$setup_cohort_key[selected]
    ))
    value$setup_target_count_vector <- I(list(
      cohorts$target_count[selected]
    ))
    value$static_target_key_vector <- I(list(unlist(
      cohorts$static_target_key_vector[selected], use.names = FALSE
    )))
    value
  })
  do.call(rbind, rows)
}

fastkpc_full_cuda_phase10_coalescing_quantiles <- function(
    values, level, unit) {
  values <- values[is.finite(values)]
  probabilities <- c(0, .25, .5, .75, .95, 1)
  data.frame(
    level = as.integer(level),
    unit = unit,
    probability = probabilities,
    epoch = if (length(values) == 0L) rep.int(NA_real_, 6L) else
      as.numeric(stats::quantile(
        values, probabilities, names = FALSE, type = 1
      )),
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_phase10_coalescing_opportunity <- function(evidence) {
  summary <- evidence$result$summary
  tasks <- evidence$result$tasks
  receipt <- summary$prefill_batches
  fastkpc_full_cuda_phase10_coalescing_require(
    is.list(evidence) &&
      identical(
        evidence$schema_version,
        "full-cuda-ci-phase10-compute-warm-evidence-v2"
      ) && !isTRUE(evidence$formal_canonical) && is.data.frame(tasks) &&
      identical(summary$scheduler, "cuda-level-target-prefill-host-v5") &&
      as.integer(summary$p) == 48L && as.integer(summary$levels) == 8L &&
      as.integer(summary$logical_tests_consumed) == 240489L &&
      is.data.frame(receipt) && nrow(receipt) == 137L,
    "Phase 10 coalescing opportunity requires the canonical v5 profile"
  )
  required_task_fields <- c(
    "level", "task_index", "edge_x", "edge_y", "x", "y", "S_key",
    "conditioning_size", "p_used", "native_edge_deleted"
  )
  fastkpc_full_cuda_phase10_coalescing_require(
    all(required_task_fields %in% names(tasks)),
    "Phase 10 coalescing source trace is incomplete"
  )

  consumed_targets <-
    fastkpc_full_cuda_phase10_coalescing_consumed_targets(tasks)
  p <- as.integer(summary$p)
  adjacency <- matrix(1L, p, p)
  diag(adjacency) <- 0L
  cohorts_by_level <- list()
  epochs_by_level <- list()
  for (level in 0:7) {
    plan <- precision_make_layer_plan_native(adjacency, level)
    plan_tasks <- fastkpc_full_cuda_phase10_coalescing_plan_frame(plan)
    observed <- tasks[tasks$level == level, , drop = FALSE]
    observed <- observed[order(observed$task_index), , drop = FALSE]
    cohorts <- fastkpc_full_cuda_phase10_coalescing_level_cohorts(
      plan_tasks, level, consumed_targets
    )
    replay <- fastkpc_full_cuda_phase10_coalescing_replay_level(
      plan_tasks, observed, adjacency, cohorts, level,
      as.numeric(summary$alpha)
    )
    adjacency <- replay$adjacency
    if (level > 0L) cohorts_by_level[[as.character(level)]] <- replay$cohorts
    epochs_by_level[[as.character(level)]] <- replay$epochs
  }
  cohorts <- do.call(rbind, cohorts_by_level)
  rownames(cohorts) <- NULL
  epochs <- do.call(rbind, epochs_by_level)
  rownames(epochs) <- NULL
  windows <- fastkpc_full_cuda_phase10_coalescing_window_frame(cohorts)
  receipt_key <- paste(receipt$level, receipt$window_id, sep = ":")
  window_key <- paste(windows$level, windows$window_id, sep = ":")
  receipt_index <- match(window_key, receipt_key)
  fastkpc_full_cuda_phase10_coalescing_require(
    !anyNA(receipt_index) && length(unique(receipt_index)) == nrow(receipt),
    "Phase 10 v5 window identity could not be rebuilt"
  )
  matched_receipt <- receipt[receipt_index, , drop = FALSE]
  count_fields <- c(
    "level", "penalty_class", "window_id", "conditioning_group_count",
    "optimizer_setup_count", "target_optimization_count",
    "unique_target_key_count",
    "consumed_unique_target_key_count",
    "unconsumed_unique_target_key_count",
    "singleton_skipped_request_count", "singleton_skipped_target_count"
  )
  fastkpc_full_cuda_phase10_coalescing_require(
    all(vapply(count_fields, function(field) {
      identical(windows[[field]], matched_receipt[[field]])
    }, logical(1L))),
    "Phase 10 rebuilt v5 window receipt changed"
  )
  windows$optimizer_host_ms <- matched_receipt$optimizer_host_ms
  windows$batch_wall_ms <- matched_receipt$batch_wall_ms

  max_demanded_window <- vapply(split(
    windows$window_id[windows$ever_demanded],
    windows$level[windows$ever_demanded]
  ), max, integer(1L))
  windows$executed_by_original_order_watermark <- vapply(
    seq_len(nrow(windows)), function(index) {
      windows$window_id[[index]] <=
        max_demanded_window[[as.character(windows$level[[index]])]]
    }, logical(1L)
  )

  level_rows <- lapply(sort(unique(windows$level)), function(level) {
    selected_windows <- windows$level == level
    selected_cohorts <- cohorts$level == level
    data.frame(
      level = level,
      total_v5_windows = sum(selected_windows),
      ever_demanded_windows = sum(
        selected_windows & windows$ever_demanded
      ),
      never_demanded_windows = sum(
        selected_windows & !windows$ever_demanded
      ),
      watermark_executed_windows = sum(
        selected_windows & windows$executed_by_original_order_watermark
      ),
      watermark_skipped_windows = sum(
        selected_windows & !windows$executed_by_original_order_watermark
      ),
      total_setup_cohorts = sum(selected_cohorts),
      ever_demanded_setup_cohorts = sum(
        selected_cohorts & cohorts$ever_demanded
      ),
      never_demanded_setup_cohorts = sum(
        selected_cohorts & !cohorts$ever_demanded
      ),
      total_target_optimizations = sum(cohorts$target_count[selected_cohorts]),
      consumed_target_keys = sum(
        cohorts$consumed_target_count[selected_cohorts]
      ),
      unconsumed_targets_inside_demanded_cohorts = sum(
        cohorts$unconsumed_target_count[
          selected_cohorts & cohorts$ever_demanded
        ]
      ),
      targets_in_never_demanded_cohorts = sum(
        cohorts$target_count[selected_cohorts & !cohorts$ever_demanded]
      ),
      targets_in_never_demanded_windows = sum(
        windows$target_optimization_count[
          selected_windows & !windows$ever_demanded
        ]
      ),
      recorded_wall_ms_of_never_demanded_windows = sum(
        windows$batch_wall_ms[selected_windows & !windows$ever_demanded]
      ),
      watermark_skipped_recorded_wall_ms = sum(
        windows$batch_wall_ms[
          selected_windows & !windows$executed_by_original_order_watermark
        ]
      ),
      stringsAsFactors = FALSE
    )
  })
  by_level <- do.call(rbind, level_rows)
  totals <- data.frame(
    metric = setdiff(names(by_level), "level"),
    value = vapply(
      by_level[setdiff(names(by_level), "level")], sum, numeric(1L)
    ),
    stringsAsFactors = FALSE
  )
  total_value <- stats::setNames(totals$value, totals$metric)
  through_level3 <- by_level$level <= 3L
  level3_checkpoint <- data.frame(
    metric = c(
      "v5_optimizer_boundaries", "lazy_demand_order_boundaries",
      "v5_setup_submissions", "lazy_demand_order_setup_submissions",
      "v5_target_optimizations", "lazy_demand_order_target_optimizations",
      "skippable_recorded_batch_wall_ms"
    ),
    value = c(
      sum(by_level$total_v5_windows[through_level3]),
      sum(by_level$ever_demanded_windows[through_level3]),
      sum(by_level$total_setup_cohorts[through_level3]),
      sum(windows$optimizer_setup_count[
        windows$level <= 3L & windows$ever_demanded
      ]),
      sum(by_level$total_target_optimizations[through_level3]),
      sum(windows$target_optimization_count[
        windows$level <= 3L & windows$ever_demanded
      ]),
      sum(by_level$recorded_wall_ms_of_never_demanded_windows[through_level3])
    ),
    stringsAsFactors = FALSE
  )

  first_demand_distribution <- do.call(rbind, lapply(
    sort(unique(cohorts$level)), function(level) {
      rbind(
        fastkpc_full_cuda_phase10_coalescing_quantiles(
          cohorts$first_demand_epoch[cohorts$level == level], level,
          "setup_cohort"
        ),
        fastkpc_full_cuda_phase10_coalescing_quantiles(
          windows$first_demand_epoch[windows$level == level], level,
          "v5_window"
        )
      )
    }
  ))
  newly_ready <- cohorts[is.finite(cohorts$first_ready_epoch), c(
    "level", "first_ready_epoch", "penalty_class", "coefficient_dim",
    "penalty_count", "target_count"
  )]
  frontier_epoch_shape <- stats::aggregate(
    rep.int(1L, nrow(newly_ready)),
    newly_ready,
    sum
  )
  names(frontier_epoch_shape)[[ncol(frontier_epoch_shape)]] <- "setup_count"
  frontier_density <- do.call(rbind, lapply(
    split(epochs, epochs$level), function(level_epochs) {
      data.frame(
        level = unique(level_epochs$level),
        frontier_epoch_count = nrow(level_epochs),
        newly_ready_setup_p50 = unname(stats::quantile(
          level_epochs$newly_ready_setup_count, .5, type = 1
        )),
        newly_ready_setup_p95 = unname(stats::quantile(
          level_epochs$newly_ready_setup_count, .95, type = 1
        )),
        newly_ready_setup_max = max(level_epochs$newly_ready_setup_count),
        inactive_ready_setup_p50 = unname(stats::quantile(
          level_epochs$ready_in_inactive_window_count, .5, type = 1
        )),
        inactive_ready_setup_p95 = unname(stats::quantile(
          level_epochs$ready_in_inactive_window_count, .95, type = 1
        )),
        inactive_ready_setup_max = max(
          level_epochs$ready_in_inactive_window_count
        ),
        epochs_with_full_64_setup_window = sum(
          level_epochs$coalescible_full_window_count > 0L
        ),
        stringsAsFactors = FALSE
      )
    }
  ))

  decision <- if (
    total_value[["recorded_wall_ms_of_never_demanded_windows"]] >= 15000
  ) {
    "GO_LAZY_ORIGINAL_WINDOW_PROTOTYPE"
  } else if (
    total_value[["targets_in_never_demanded_cohorts"]] >=
      0.5 * as.numeric(summary$prefill_unconsumed_unique_target_key_count)
  ) {
    "HOLD_FOR_SETUP_COHORT_PROTOTYPE"
  } else {
    "STOP_SCHEDULER_OPPORTUNITY_TOO_SMALL"
  }
  result <- list(
    schema_version =
      "full-cuda-ci-phase10-setup-aware-coalescing-opportunity-v1",
    source = list(
      evidence_schema_version = evidence$schema_version,
      dataset_key = summary$dataset_key,
      scheduler = summary$scheduler,
      elapsed_sec = evidence$elapsed_sec,
      max_conditioning_size = 7L,
      analysis_method =
        "existing-p-used-native-layer-plan-frontier-replay-no-cuda-ci",
      numerical_cuda_call_count = 0L,
      identity_scope = paste(
        "static setup/target keys are interpreted within the source DatasetKey",
        "and use conditioning identity;",
        "they do not claim the native PreparedS fingerprint"
      )
    ),
    receipt_rebuild = list(
      window_count = nrow(windows),
      setup_cohort_count = nrow(cohorts),
      target_optimization_count = sum(cohorts$target_count),
      consumed_target_key_count = sum(cohorts$consumed_target_count),
      exact = TRUE
    ),
    decision = decision,
    totals = totals,
    by_level = by_level,
    level3_checkpoint = level3_checkpoint,
    setup_cohorts = cohorts,
    v5_windows = windows,
    frontier_epochs = epochs,
    frontier_epoch_shape = frontier_epoch_shape,
    frontier_density = frontier_density,
    first_demand_distribution = first_demand_distribution
  )
  fastkpc_full_cuda_phase10_validate_coalescing_opportunity(result, summary)
  result
}

fastkpc_full_cuda_phase10_validate_coalescing_opportunity <- function(
    value, source_summary = NULL) {
  fastkpc_full_cuda_phase10_coalescing_require(
    is.list(value) && identical(
      value$schema_version,
      "full-cuda-ci-phase10-setup-aware-coalescing-opportunity-v1"
    ) && isTRUE(value$receipt_rebuild$exact) &&
      nrow(value$v5_windows) == value$receipt_rebuild$window_count &&
      nrow(value$setup_cohorts) == value$receipt_rebuild$setup_cohort_count &&
      sum(value$setup_cohorts$target_count) ==
        value$receipt_rebuild$target_optimization_count &&
      sum(value$setup_cohorts$consumed_target_count) ==
        value$receipt_rebuild$consumed_target_key_count &&
      all(value$setup_cohorts$target_count ==
            value$setup_cohorts$consumed_target_count +
              value$setup_cohorts$unconsumed_target_count) &&
      all(value$v5_windows$target_optimization_count ==
            value$v5_windows$consumed_unique_target_key_count +
              value$v5_windows$unconsumed_unique_target_key_count) &&
      all(value$v5_windows$conditioning_group_count ==
            value$v5_windows$optimizer_setup_count) &&
      all(value$v5_windows$optimizer_host_ms >= 0) &&
      all(value$v5_windows$batch_wall_ms >= 0) &&
      all(lengths(value$v5_windows$setup_cohort_vector) ==
            value$v5_windows$optimizer_setup_count) &&
      all(lengths(value$v5_windows$static_target_key_vector) ==
            value$v5_windows$target_optimization_count) &&
      all(lengths(value$setup_cohorts$target_vector) ==
            value$setup_cohorts$target_count) &&
      all(lengths(value$setup_cohorts$static_optimizer_state_key_vector) ==
            value$setup_cohorts$target_count) &&
      all(value$frontier_epochs$demand_order_window_activation_count %in%
            0:1) &&
      identical(value$source$numerical_cuda_call_count, 0L) &&
      identical(
        value$source$analysis_method,
        "existing-p-used-native-layer-plan-frontier-replay-no-cuda-ci"
      ) &&
      value$decision %in% c(
        "GO_LAZY_ORIGINAL_WINDOW_PROTOTYPE",
        "HOLD_FOR_SETUP_COHORT_PROTOTYPE",
        "STOP_SCHEDULER_OPPORTUNITY_TOO_SMALL"
      ),
    "Phase 10 setup-aware coalescing opportunity is malformed"
  )
  if (!is.null(source_summary)) {
    fastkpc_full_cuda_phase10_coalescing_require(
      value$receipt_rebuild$window_count ==
        as.integer(source_summary$prefill_window_count) &&
        value$receipt_rebuild$setup_cohort_count ==
          as.integer(source_summary$prefill_optimizer_setup_count) &&
        value$receipt_rebuild$target_optimization_count ==
          as.integer(source_summary$prefill_target_optimization_count) &&
        value$receipt_rebuild$consumed_target_key_count ==
          as.integer(source_summary$prefill_consumed_unique_target_key_count),
      "Phase 10 coalescing opportunity disagrees with source counters"
    )
  }
  invisible(value)
}
