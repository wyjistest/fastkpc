source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/legacy_runner.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!isTRUE(tryCatch(
  fastkpc_cuda_available(),
  error = function(error) FALSE
))) {
  cat("SKIP strict full-CUDA CI methods: CUDA unavailable\n")
  quit(save = "no", status = 0L)
}

normalize_sepsets <- function(value) {
  unname(lapply(unname(value), function(row) {
    unname(lapply(unname(row), function(entry) sort(as.integer(entry))))
  }))
}

task_permutation_seed <- function(method, base_seed, level, edge_x, edge_y,
                                  x, y, S_key) {
  payload <- paste0(
    "schema=full-cuda-ci-task-permutation-seed-v1\n",
    "method=", method, "\n",
    "level=", level, "\n",
    "edge=", edge_x, "|", edge_y, "\n",
    "orientation=", x, "|", y, "\n",
    "S=", S_key, "\n"
  )
  hash <- unname(digest::digest(
    enc2utf8(payload), algo = "sha256", serialize = FALSE
  ))
  prefix <- strtoi(substr(hash, 1L, 4L), base = 16L) * 65536 +
    strtoi(substr(hash, 5L, 8L), base = 16L)
  as.integer((as.numeric(base_seed) + prefix %% 2147483647) %% 2147483647)
}

center_distance <- function(values) {
  distance <- abs(outer(values, values, FUN = "-"))
  centered <- sweep(distance, 1L, rowMeans(distance), FUN = "-")
  centered <- sweep(centered, 2L, colMeans(distance), FUN = "-")
  centered + mean(distance)
}

center_hsic_factor <- function(values, sig, num_col) {
  factor <- kernlab::inchol(
    as.matrix(values),
    kernel = kernlab::rbfdot(sigma = 1 / sig),
    maxiter = num_col
  )@.Data
  sweep(factor, 2L, colMeans(factor), FUN = "-")
}

hsic_factor_statistic <- function(left, right, permutation = NULL) {
  if (!is.null(permutation)) {
    right <- right[permutation, , drop = FALSE]
  }
  sum(crossprod(left, right)^2) / nrow(left)^2
}

make_canonical_provider <- function(data, method, sig, num_col, replicates,
                                    seed) {
  env <- fastkpc_legacy_env()

  function(tasks, level) {
    if (nrow(tasks) == 0L) return(numeric())
    seeds <- vapply(seq_len(nrow(tasks)), function(index) {
      task_permutation_seed(
        method, seed, level,
        tasks$edge_x[[index]], tasks$edge_y[[index]],
        tasks$x[[index]], tasks$y[[index]], tasks$S_key[[index]]
      )
    }, integer(1L))
    permutations <- if (method %in% c("dcc.perm", "hsic.perm")) {
      full_cuda_ci_method_seeded_permutations_native(
        method, nrow(data), replicates, seeds
      )
    } else {
      NULL
    }

    vapply(seq_len(nrow(tasks)), function(index) {
      left_target <- tasks$x[[index]]
      right_target <- tasks$y[[index]]
      conditioning_set <- as.integer(tasks$conditioning_sets[[index]])
      values <- data[, c(left_target, right_target), drop = FALSE]
      if (length(conditioning_set) > 0L) {
        values <- env$regrXonS(
          values, data[, conditioning_set, drop = FALSE]
        )
      }
      if (method == "dcc.perm") {
        left <- center_distance(values[, 1L])
        right <- center_distance(values[, 2L])
        observed <- sum(left * right)
        replicate_values <- vapply(seq_len(replicates), function(replicate) {
          permutation <- permutations[, replicate, index] + 1L
          sum(left * right[permutation, permutation])
        }, numeric(1L))
        return((1 + sum(replicate_values >= observed)) / (replicates + 1))
      }
      if (method == "hsic.perm") {
        left <- center_hsic_factor(
          values[, 1L], sig = sig, num_col = num_col
        )
        right <- center_hsic_factor(
          values[, 2L], sig = sig, num_col = num_col
        )
        observed <- hsic_factor_statistic(left, right)
        replicate_values <- vapply(seq_len(replicates), function(replicate) {
          permutation <- permutations[, replicate, index] + 1L
          hsic_factor_statistic(left, right, permutation)
        }, numeric(1L))
        return((1 + sum(replicate_values >= observed)) / (replicates + 1))
      }
      env$hsic.gamma(
        values[, 1L], values[, 2L],
        sig = sig, numCol = num_col
      )$p.value
    }, numeric(1L))
  }
}

run_canonical_oracle <- function(data, method, alpha, max_level, num_col,
                                 sig, replicates, seed) {
  precision_run_skeleton_provider_native(
    p = ncol(data),
    alpha = alpha,
    max_conditioning_size = max_level,
    provider = make_canonical_provider(
      data, method, sig, num_col, replicates, seed
    ),
    trace_level = "full"
  )
}

run_recorded_canonical_oracle <- function(data, recorded_tasks, alpha,
                                          max_level) {
  task_keys <- paste(
    recorded_tasks$level, recorded_tasks$x, recorded_tasks$y,
    recorded_tasks$S_key, sep = "|"
  )
  if (anyDuplicated(task_keys)) {
    stop("recorded kpcalg task identity is not unique", call. = FALSE)
  }
  p_values <- setNames(as.numeric(recorded_tasks$p_used), task_keys)
  provider <- function(tasks, level) {
    if (nrow(tasks) == 0L) return(numeric())
    S_keys <- vapply(tasks$conditioning_sets, function(S) {
      paste(sort(as.integer(S)), collapse = "|")
    }, character(1L))
    keys <- paste(level, tasks$x, tasks$y, S_keys, sep = "|")
    values <- unname(p_values[keys])
    # The provider planner may ask for trailing rows that canonical replay
    # later ignores after an earlier separating set deletes the same edge.
    values[is.na(values)] <- 0
    if (length(values) != nrow(tasks)) {
      stop(sprintf(
        "recorded provider alignment changed: tasks=%d keys=%d values=%d",
        nrow(tasks), length(keys), length(values)
      ), call. = FALSE)
    }
    values
  }
  precision_run_skeleton_provider_native(
    p = ncol(data),
    alpha = alpha,
    max_conditioning_size = max_level,
    provider = provider,
    trace_level = "full"
  )
}

run_kpcalg_graph_oracle <- function(data, method, alpha, max_level, num_col,
                                    sig, replicates, seed) {
  env <- fastkpc_legacy_env()
  suff_stat <- list(
    data = data,
    ic.method = method,
    index = 1,
    numCol = as.integer(num_col),
    sig = as.numeric(sig),
    p = as.integer(replicates)
  )
  set.seed(seed)
  skeleton <- pcalg::skeleton(
    suffStat = suff_stat,
    indepTest = env$kernelCItest,
    alpha = alpha,
    labels = colnames(data),
    m.max = max_level,
    method = "stable"
  )
  list(
    adjacency = methods::as(skeleton@graph, "matrix") != 0,
    sepsets = skeleton@sepset,
    pMax = skeleton@pMax,
    n.edgetests = as.integer(skeleton@n.edgetests)
  )
}

run_kpcalg_trace_oracle <- function(data, method, alpha, max_level, num_col,
                                    sig, replicates, seed) {
  env <- fastkpc_legacy_env()
  calls <- list()
  traced_test <- function(x, y, S, suffStat) {
    p_value <- env$kernelCItest(x, y, S, suffStat)
    calls[[length(calls) + 1L]] <<- data.frame(
      level = length(S),
      x = as.integer(x),
      y = as.integer(y),
      S_key = paste(sort(as.integer(S)), collapse = "|"),
      p_used = as.numeric(p_value),
      native_edge_deleted = as.numeric(p_value) >= alpha,
      stringsAsFactors = FALSE
    )
    p_value
  }
  suff_stat <- list(
    data = data,
    ic.method = method,
    index = 1,
    numCol = as.integer(num_col),
    sig = as.numeric(sig),
    p = as.integer(replicates)
  )
  set.seed(seed)
  skeleton <- pcalg::skeleton(
    suffStat = suff_stat,
    indepTest = traced_test,
    alpha = alpha,
    labels = colnames(data),
    m.max = max_level,
    method = "stable"
  )
  trace <- if (length(calls) == 0L) {
    data.frame(
      level = integer(), x = integer(), y = integer(), S_key = character(),
      p_used = numeric(), native_edge_deleted = logical()
    )
  } else {
    do.call(rbind, calls)
  }
  rownames(trace) <- NULL
  list(
    adjacency = methods::as(skeleton@graph, "matrix") != 0,
    sepsets = skeleton@sepset,
    pMax = skeleton@pMax,
    n.edgetests = as.integer(skeleton@n.edgetests),
    tasks = trace
  )
}

run_cuda_candidate <- function(data, method, alpha, max_level, num_col,
                               sig, replicates, seed) {
  invisible(full_cuda_ci_one_call_cache_control_native("reset"))
  precision_run_skeleton_full_cuda_native(
    data = data,
    alpha = alpha,
    max_conditioning_size = max_level,
    index = 1,
    numCol = num_col,
    trace_level = "logical",
    compatible_cuda_strict = TRUE,
    ci_method = method,
    hsic_params = list(sig = sig),
    permutation_params = list(
      replicates = replicates,
      seed = seed,
      include_observed = TRUE
    )
  )
}

structural_trace <- function(value) {
  value <- value[!value$native_edge_ignored, c(
    "level", "task_index", "edge_x", "edge_y", "x", "y", "S_key",
    "conditioning_size", "native_edge_deleted"
  ), drop = FALSE]
  rownames(value) <- NULL
  value
}

consumed_p_values <- function(value) {
  as.numeric(value$p_used[!value$native_edge_ignored])
}

canonical_task_identity <- function(value) {
  value <- value[!value$native_edge_ignored, c(
    "level", "x", "y", "S_key", "p_used", "native_edge_deleted"
  ), drop = FALSE]
  value$task_key <- paste(
    value$level, value$x, value$y, value$S_key, sep = "|"
  )
  value
}

compare_kpcalg_process <- function(candidate_tasks, kpcalg_tasks,
                                   tolerance) {
  candidate <- canonical_task_identity(candidate_tasks)
  legacy <- kpcalg_tasks
  legacy$task_key <- paste(
    legacy$level, legacy$x, legacy$y, legacy$S_key, sep = "|"
  )
  unique_keys <- !anyDuplicated(candidate$task_key) &&
    !anyDuplicated(legacy$task_key)
  same_keys <- unique_keys && setequal(candidate$task_key, legacy$task_key)
  if (!same_keys) {
    return(list(
      same_task_keys = FALSE,
      p_values_within_tolerance = FALSE,
      decision_trace_identical = FALSE,
      max_abs_p_diff = Inf
    ))
  }
  legacy <- legacy[match(candidate$task_key, legacy$task_key), , drop = FALSE]
  differences <- abs(candidate$p_used - legacy$p_used)
  list(
    same_task_keys = TRUE,
    p_values_within_tolerance = all(differences <= tolerance),
    decision_trace_identical = identical(
      as.logical(candidate$native_edge_deleted),
      as.logical(legacy$native_edge_deleted)
    ),
    max_abs_p_diff = max(differences)
  )
}

trim_trailing_empty_levels <- function(value) {
  value <- as.integer(value)
  while (length(value) > 1L && tail(value, 1L) == 0L) {
    value <- head(value, -1L)
  }
  value
}

compare_method <- function(candidate, canonical, legacy, method, max_level,
                           fixture, tolerance = 1e-10) {
  canonical_adjacency <- canonical$adjacency != 0
  legacy_adjacency <- legacy$adjacency != 0
  candidate_adjacency <- candidate$adjacency != 0
  canonical_shd <- sum(abs(
    as.integer(unname(candidate_adjacency)) -
      as.integer(unname(canonical_adjacency))
  )) / 2L
  legacy_shd <- sum(abs(
    as.integer(unname(candidate_adjacency)) -
      as.integer(unname(legacy_adjacency))
  )) / 2L
  candidate_p <- consumed_p_values(candidate$tasks)
  canonical_p <- consumed_p_values(canonical$tasks)
  p_differences <- if (length(candidate_p) == length(canonical_p)) {
    abs(candidate_p - canonical_p)
  } else {
    Inf
  }

  data.frame(
    method = method,
    fixture = fixture,
    max_level = max_level,
    canonical_SHD = as.integer(canonical_shd),
    legacy_SHD = as.integer(legacy_shd),
    adjacency_identical = identical(
      unname(candidate_adjacency), unname(canonical_adjacency)
    ),
    sepsets_identical = identical(
      normalize_sepsets(candidate$sepsets),
      normalize_sepsets(canonical$sepsets)
    ),
    pmax_identical = identical(
      unname(candidate$pMax), unname(canonical$pMax)
    ),
    pmax_within_tolerance = isTRUE(all.equal(
      unname(candidate$pMax), unname(canonical$pMax),
      tolerance = tolerance, check.attributes = FALSE
    )),
    n_edgetests_identical = identical(
      trim_trailing_empty_levels(candidate$n.edgetests),
      trim_trailing_empty_levels(canonical$n.edgetests)
    ),
    trace_identical = identical(
      structural_trace(candidate$tasks), structural_trace(canonical$tasks)
    ),
    p_values_identical = identical(candidate_p, canonical_p),
    p_values_within_tolerance = all(p_differences <= tolerance),
    candidate_tests = length(candidate_p),
    highest_consumed_level = if (nrow(candidate$tasks) == 0L) {
      -1L
    } else {
      max(as.integer(candidate$tasks$level))
    },
    decision_flips = if (length(candidate_p) == length(canonical_p)) {
      sum((candidate_p >= 0.1) != (canonical_p >= 0.1))
    } else {
      NA_integer_
    },
    max_abs_p_diff = if (length(candidate_p) == length(canonical_p)) {
      max(p_differences)
    } else {
      Inf
    },
    stringsAsFactors = FALSE
  )
}

set.seed(9127)
n <- 56L
z <- stats::runif(n, -2, 2)
data <- cbind(
  x1 = sin(z) + stats::rnorm(n, sd = 0.08),
  x2 = cos(z) + stats::rnorm(n, sd = 0.08),
  x3 = z + stats::rnorm(n, sd = 0.08),
  x4 = z^2 + stats::rnorm(n, sd = 0.08),
  x5 = z^3 + stats::rnorm(n, sd = 0.08)
)
set.seed(9231)
dense_n <- 80L
dense_p <- 6L
common <- stats::rnorm(dense_n)
dense_data <- sapply(seq_len(dense_p), function(index) {
  common + 0.4 * stats::rnorm(dense_n)
})
colnames(dense_data) <- paste0("x", seq_len(dense_p))

methods <- c("dcc.perm", "hsic.gamma", "hsic.perm")
scenarios <- list(
  nonlinear = list(data = data, max_levels = c(0L, 1L, 2L)),
  dense = list(data = dense_data, max_levels = 3L)
)
comparisons <- lapply(names(scenarios), function(fixture) {
  scenario <- scenarios[[fixture]]
  do.call(rbind, lapply(scenario$max_levels, function(max_level) {
    do.call(rbind, lapply(methods, function(method) {
      legacy_trace <- run_kpcalg_trace_oracle(
        scenario$data, method, alpha = 0.1,
        max_level = max_level, num_col = 35L,
        sig = 1, replicates = 100L, seed = 707L
      )
      candidate <- run_cuda_candidate(
        scenario$data, method, alpha = 0.1,
        max_level = max_level, num_col = 35L,
        sig = 1, replicates = 100L, seed = 707L
      )
      canonical <- if (method %in% c("dcc.perm", "hsic.perm")) {
        run_recorded_canonical_oracle(
          scenario$data, legacy_trace$tasks, alpha = 0.1,
          max_level = max_level
        )
      } else {
        run_canonical_oracle(
          scenario$data, method, alpha = 0.1,
          max_level = max_level, num_col = 35L,
          sig = 1, replicates = 100L, seed = 707L
        )
      }
      process <- compare_kpcalg_process(
        candidate$tasks, legacy_trace$tasks,
        tolerance = if (method == "hsic.gamma") 1e-10 else 0
      )
      comparison <- compare_method(
        candidate, canonical, legacy_trace, method, max_level, fixture
      )
      comparison$kpcalg_task_keys_identical <- process$same_task_keys
      comparison$kpcalg_p_values_within_tolerance <-
        process$p_values_within_tolerance
      comparison$kpcalg_decision_trace_identical <-
        process$decision_trace_identical
      comparison$kpcalg_max_abs_p_diff <- process$max_abs_p_diff
      comparison
    }))
  }))
})
comparisons <- do.call(rbind, comparisons)
print(comparisons, row.names = FALSE)

required <- c(
  "canonical_SHD", "legacy_SHD", "adjacency_identical",
  "sepsets_identical", "pmax_within_tolerance", "n_edgetests_identical",
  "trace_identical", "p_values_within_tolerance",
  "kpcalg_task_keys_identical", "kpcalg_p_values_within_tolerance",
  "kpcalg_decision_trace_identical"
)
failed <- comparisons$canonical_SHD != 0L | comparisons$legacy_SHD != 0L |
  !apply(comparisons[, setdiff(required, c("canonical_SHD", "legacy_SHD")),
                     drop = FALSE], 1L, all)
assert_true(!any(failed), paste0(
  "strict full-CUDA CI method parity failed: ",
  paste(comparisons$method[failed], collapse = ", ")
))

cat("PASS strict full-CUDA CI method parity\n")
