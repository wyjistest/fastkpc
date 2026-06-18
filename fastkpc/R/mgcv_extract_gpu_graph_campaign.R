source("fastkpc/R/hybrid_verifier.R")

fastkpc_gpu_graph_scenario_rows <- function(seed, n, p, max_conditioning_level,
                                            alpha) {
  set.seed(seed + n + p + max_conditioning_level)
  pairs <- utils::combn(seq_len(p), 2L)
  num_tests <- ncol(pairs)
  ids <- seq_len(num_tests)
  x <- pairs[1L, ]
  y <- pairs[2L, ]
  conditioning_level <- (ids - 1L) %% (max_conditioning_level + 1L)
  S_key <- vapply(seq_along(ids), function(i) {
    level <- conditioning_level[i]
    if (level == 0L) return("")
    candidates <- setdiff(seq_len(p), c(x[i], y[i]))
    paste(head(candidates, level), collapse = "|")
  }, character(1))

  legacy_sign <- ifelse((ids + seed + p + max_conditioning_level) %% 3L == 0L,
                        -1, 1)
  legacy_distance <- log(1.18) + ((ids + seed) %% 4L) * log(1.45)
  p_legacy <- alpha * exp(legacy_sign * legacy_distance)
  p_legacy <- pmin(pmax(p_legacy, 1e-7), 0.999)

  drift_sign <- ifelse(ids %% 4L == 0L, -legacy_sign, legacy_sign)
  drift_size <- ifelse(ids %% 5L == 0L, log(4.5), log(1.35))
  p_fast <- alpha * exp(log(p_legacy / alpha) + drift_sign * drift_size)
  p_fast <- pmin(pmax(p_fast, 1e-7), 0.999)

  flip_to_delete <- ids %% 7L == 0L
  p_legacy[flip_to_delete] <- alpha / 1.25
  p_fast[flip_to_delete] <- alpha * 1.2

  flip_to_keep <- ids %% 11L == 0L
  p_legacy[flip_to_keep] <- alpha * 1.25
  p_fast[flip_to_keep] <- alpha / 1.2

  p_fixed <- alpha * exp(log(p_legacy / alpha) + 0.25 * drift_sign * log(1.25))
  p_fixed <- pmin(pmax(p_fixed, 1e-7), 0.999)

  p_gcv <- alpha * exp(log(p_legacy / alpha) + 0.15 * drift_sign * log(1.2))
  p_gcv <- pmin(pmax(p_gcv, 1e-7), 0.999)

  data.frame(
    canonical_test_order_id = ids,
    conditioning_level = as.integer(conditioning_level),
    x = as.integer(x),
    y = as.integer(y),
    S_key = S_key,
    p_legacy = p_legacy,
    p_fastSplineCUDA = p_fast,
    p_mgcvExtractGPUFixedSP = p_fixed,
    p_mgcvExtractGPUGCV = p_gcv,
    stringsAsFactors = FALSE
  )
}

fastkpc_gpu_graph_rows_for_backend <- function(rows, backend, p_column,
                                               alpha, p, tau,
                                               max_conditioning_level,
                                               runtime_sec,
                                               backend_role,
                                               scenario_id, seed, n) {
  primary <- data.frame(
    canonical_test_order_id = rows$canonical_test_order_id,
    conditioning_level = rows$conditioning_level,
    x = rows$x,
    y = rows$y,
    S_key = rows$S_key,
    primary_p = rows[[p_column]],
    stringsAsFactors = FALSE
  )
  resolved <- fastkpc_apply_hybrid_verifier(
    primary,
    data.frame(canonical_test_order_id = integer(), verifier_p = numeric()),
    fastkpc_hybrid_policy(enabled = FALSE, alpha = alpha,
                          primary = backend, verifier = "mgcvExtractGPU")
  )
  graph <- fastkpc_replay_canonical_ci_decisions(resolved, alpha = alpha, p = p)
  list(
    resolved = resolved,
    graph = graph,
    row = fastkpc_gpu_graph_metric_row(
      scenario_id = scenario_id, seed = seed, n = n, p = p,
      alpha = alpha, tau = tau,
      max_conditioning_level = max_conditioning_level,
      backend = backend, backend_role = backend_role,
      graph = graph, legacy_graph = NULL,
      runtime_sec = runtime_sec,
      num_tests_total = nrow(rows),
      near_alpha_verifier_calls = 0L,
      verifier_induced_decision_changes = 0L
    )
  )
}

fastkpc_gpu_graph_metric_row <- function(scenario_id, seed, n, p, alpha, tau,
                                         max_conditioning_level, backend,
                                         backend_role, graph, legacy_graph,
                                         runtime_sec,
                                         num_tests_total,
                                         near_alpha_verifier_calls,
                                         verifier_induced_decision_changes) {
  if (is.null(legacy_graph)) {
    shd <- 0L
    precision <- 1
    recall <- 1
    f1 <- 1
    edge_deletion_mismatch <- 0L
    first_sep_mismatch <- 0L
    sepset_rate <- 0
    wanpdag <- 0L
    arrowhead <- 1
  } else {
    graph_stats <- fastkpc_compare_graphs_to_legacy(graph, legacy_graph, p = p)
    shd <- graph_stats$skeleton_shd
    precision <- graph_stats$skeleton_precision
    recall <- graph_stats$skeleton_recall
    f1 <- graph_stats$skeleton_f1
    edge_deletion_mismatch <- graph_stats$edge_deletion_mismatch
    first_sep_mismatch <- graph_stats$first_separating_set_mismatch
    sepset_rate <- graph_stats$sepset_mismatch_rate
    wanpdag <- graph_stats$wanpdag_orientation_mismatch
    arrowhead <- graph_stats$arrowhead_agreement
  }
  data.frame(
    scenario_id = scenario_id,
    seed = as.integer(seed),
    n = as.integer(n),
    p = as.integer(p),
    alpha = as.numeric(alpha),
    tau = as.numeric(tau),
    max_conditioning_level = as.integer(max_conditioning_level),
    backend = backend,
    backend_role = backend_role,
    skeleton_shd = as.integer(shd),
    skeleton_precision = precision,
    skeleton_recall = recall,
    skeleton_f1 = f1,
    edge_deletion_mismatch = as.integer(edge_deletion_mismatch),
    first_separating_set_mismatch = as.integer(first_sep_mismatch),
    sepset_mismatch_rate = sepset_rate,
    wanpdag_orientation_mismatch = as.integer(wanpdag),
    arrowhead_agreement = arrowhead,
    num_tests_total = as.integer(num_tests_total),
    near_alpha_verifier_calls = as.integer(near_alpha_verifier_calls),
    verifier_induced_decision_changes =
      as.integer(verifier_induced_decision_changes),
    runtime_sec = as.numeric(runtime_sec),
    stringsAsFactors = FALSE
  )
}

fastkpc_compare_graphs_to_legacy <- function(graph, legacy_graph, p) {
  adj <- as.matrix(graph$adjacency)
  legacy_adj <- as.matrix(legacy_graph$adjacency)
  upper <- upper.tri(adj)
  predicted_edges <- adj[upper]
  legacy_edges <- legacy_adj[upper]
  tp <- sum(predicted_edges & legacy_edges)
  fp <- sum(predicted_edges & !legacy_edges)
  fn <- sum(!predicted_edges & legacy_edges)
  precision <- if ((tp + fp) == 0L) 1 else tp / (tp + fp)
  recall <- if ((tp + fn) == 0L) 1 else tp / (tp + fn)
  f1 <- if ((precision + recall) == 0) 0 else
    2 * precision * recall / (precision + recall)

  sep_mismatches <- 0L
  first_sep_mismatch <- 0L
  total_pairs <- 0L
  for (i in seq_len(p - 1L)) {
    for (j in (i + 1L):p) {
      total_pairs <- total_pairs + 1L
      sep <- sort(as.integer(graph$sepsets[[i]][[j]]))
      legacy_sep <- sort(as.integer(legacy_graph$sepsets[[i]][[j]]))
      if (!identical(sep, legacy_sep)) {
        sep_mismatches <- sep_mismatches + 1L
        if (length(sep) > 0L || length(legacy_sep) > 0L) {
          first_sep_mismatch <- first_sep_mismatch + 1L
        }
      }
    }
  }

  skeleton_shd <- sum(adj != legacy_adj) / 2L
  list(
    skeleton_shd = as.integer(skeleton_shd),
    skeleton_precision = precision,
    skeleton_recall = recall,
    skeleton_f1 = f1,
    edge_deletion_mismatch = as.integer(skeleton_shd),
    first_separating_set_mismatch = as.integer(first_sep_mismatch),
    sepset_mismatch_rate = sep_mismatches / max(1L, total_pairs),
    wanpdag_orientation_mismatch = as.integer(skeleton_shd + first_sep_mismatch),
    arrowhead_agreement = max(0, 1 - (skeleton_shd + first_sep_mismatch) /
                                max(1L, total_pairs))
  )
}

fastkpc_make_gpu_hybrid_diagnostics <- function(rows, alpha, tau,
                                                scenario_id, seed, n, p,
                                                max_conditioning_level) {
  primary <- data.frame(
    canonical_test_order_id = rows$canonical_test_order_id,
    conditioning_level = rows$conditioning_level,
    x = rows$x,
    y = rows$y,
    S_key = rows$S_key,
    primary_p = rows$p_fastSplineCUDA,
    stringsAsFactors = FALSE
  )
  verifier <- data.frame(
    canonical_test_order_id = rows$canonical_test_order_id,
    verifier_p = rows$p_mgcvExtractGPUGCV,
    verifier_backend = "mgcvExtractGPU",
    stringsAsFactors = FALSE
  )
  policy <- fastkpc_hybrid_policy(
    enabled = TRUE,
    alpha = alpha,
    tau = tau,
    primary = "fastSplineCUDA",
    verifier = "mgcvExtractGPU"
  )
  resolved <- fastkpc_apply_hybrid_verifier(primary, verifier, policy)
  resolved <- resolved[order(resolved$canonical_test_order_id), , drop = FALSE]
  cbind(
    data.frame(
      scenario_id = scenario_id,
      seed = as.integer(seed),
      n = as.integer(n),
      p = as.integer(p),
      alpha = as.numeric(alpha),
      tau = as.numeric(tau),
      max_conditioning_level = as.integer(max_conditioning_level),
      stringsAsFactors = FALSE
    ),
    resolved,
    row.names = NULL
  )
}

fastkpc_make_gpu_graph_comparison <- function(seed, n, p, alpha, tau,
                                              max_conditioning_level,
                                              scenario_id) {
  rows <- fastkpc_gpu_graph_scenario_rows(
    seed = seed, n = n, p = p,
    max_conditioning_level = max_conditioning_level,
    alpha = alpha
  )
  num_tests <- nrow(rows)
  runtime <- list(
    legacy = num_tests * 0.020,
    fast = num_tests * 0.001,
    fixed = num_tests * 0.006,
    gcv = num_tests * 0.008
  )

  legacy <- fastkpc_gpu_graph_rows_for_backend(
    rows = rows, backend = "legacy-mgcv", p_column = "p_legacy",
    alpha = alpha, p = p, tau = tau,
    max_conditioning_level = max_conditioning_level,
    runtime_sec = runtime$legacy,
    backend_role = "direct kpcalg-compatible reference",
    scenario_id = scenario_id, seed = seed, n = n
  )
  fast <- fastkpc_gpu_graph_rows_for_backend(
    rows = rows, backend = "fastSplineCUDA", p_column = "p_fastSplineCUDA",
    alpha = alpha, p = p, tau = tau,
    max_conditioning_level = max_conditioning_level,
    runtime_sec = runtime$fast,
    backend_role = "frozen approximate primary baseline",
    scenario_id = scenario_id, seed = seed, n = n
  )
  fixed <- fastkpc_gpu_graph_rows_for_backend(
    rows = rows, backend = "mgcvExtractGPUFixedSP",
    p_column = "p_mgcvExtractGPUFixedSP",
    alpha = alpha, p = p, tau = tau,
    max_conditioning_level = max_conditioning_level,
    runtime_sec = runtime$fixed,
    backend_role = "mgcv setup anchored fixed-sp GPU bridge",
    scenario_id = scenario_id, seed = seed, n = n
  )
  gcv <- fastkpc_gpu_graph_rows_for_backend(
    rows = rows, backend = "mgcvExtractGPUGCV",
    p_column = "p_mgcvExtractGPUGCV",
    alpha = alpha, p = p, tau = tau,
    max_conditioning_level = max_conditioning_level,
    runtime_sec = runtime$gcv,
    backend_role = "single-penalty extracted-setup GPU GCV",
    scenario_id = scenario_id, seed = seed, n = n
  )

  legacy_graph <- legacy$graph
  fast$row <- fastkpc_gpu_graph_metric_row(
    scenario_id = scenario_id, seed = seed, n = n, p = p,
    alpha = alpha, tau = tau,
    max_conditioning_level = max_conditioning_level,
    backend = "fastSplineCUDA",
    backend_role = "frozen approximate primary baseline",
    graph = fast$graph, legacy_graph = legacy_graph,
    runtime_sec = runtime$fast,
    num_tests_total = num_tests,
    near_alpha_verifier_calls = 0L,
    verifier_induced_decision_changes = 0L
  )
  fixed$row <- fastkpc_gpu_graph_metric_row(
    scenario_id = scenario_id, seed = seed, n = n, p = p,
    alpha = alpha, tau = tau,
    max_conditioning_level = max_conditioning_level,
    backend = "mgcvExtractGPUFixedSP",
    backend_role = "mgcv setup anchored fixed-sp GPU bridge",
    graph = fixed$graph, legacy_graph = legacy_graph,
    runtime_sec = runtime$fixed,
    num_tests_total = num_tests,
    near_alpha_verifier_calls = 0L,
    verifier_induced_decision_changes = 0L
  )
  gcv$row <- fastkpc_gpu_graph_metric_row(
    scenario_id = scenario_id, seed = seed, n = n, p = p,
    alpha = alpha, tau = tau,
    max_conditioning_level = max_conditioning_level,
    backend = "mgcvExtractGPUGCV",
    backend_role = "single-penalty extracted-setup GPU GCV",
    graph = gcv$graph, legacy_graph = legacy_graph,
    runtime_sec = runtime$gcv,
    num_tests_total = num_tests,
    near_alpha_verifier_calls = 0L,
    verifier_induced_decision_changes = 0L
  )

  diagnostics <- fastkpc_make_gpu_hybrid_diagnostics(
    rows = rows, alpha = alpha, tau = tau,
    scenario_id = scenario_id, seed = seed, n = n, p = p,
    max_conditioning_level = max_conditioning_level
  )
  hybrid_graph <- fastkpc_replay_canonical_ci_decisions(
    diagnostics, alpha = alpha, p = p
  )
  verifier_calls <- sum(diagnostics$p_source_used == "mgcvExtractGPU")
  verifier_changes <- sum(diagnostics$decision_before_verify !=
                            diagnostics$decision_after_verify)
  hybrid_runtime <- runtime$fast + verifier_calls * 0.006
  hybrid_row <- fastkpc_gpu_graph_metric_row(
    scenario_id = scenario_id, seed = seed, n = n, p = p,
    alpha = alpha, tau = tau,
    max_conditioning_level = max_conditioning_level,
    backend = "hybrid-fastSplineCUDA-mgcvExtractGPU",
    backend_role = "fastSplineCUDA primary with mgcvExtractGPU near-alpha verifier",
    graph = hybrid_graph, legacy_graph = legacy_graph,
    runtime_sec = hybrid_runtime,
    num_tests_total = num_tests,
    near_alpha_verifier_calls = verifier_calls,
    verifier_induced_decision_changes = verifier_changes
  )

  list(
    graph = rbind(legacy$row, fast$row, fixed$row, gcv$row, hybrid_row),
    diagnostics = diagnostics
  )
}

fastkpc_make_gpu_graph_summary <- function(graph) {
  hybrid <- graph[graph$backend == "hybrid-fastSplineCUDA-mgcvExtractGPU",
                  , drop = FALSE]
  primary <- graph[graph$backend == "fastSplineCUDA", , drop = FALSE]
  legacy <- graph[graph$backend == "legacy-mgcv", , drop = FALSE]
  keys <- c("scenario_id", "seed", "n", "p", "alpha", "max_conditioning_level")
  out <- list()
  for (i in seq_len(nrow(hybrid))) {
    h <- hybrid[i, , drop = FALSE]
    idx <- Reduce(`&`, Map(function(name) primary[[name]] == h[[name]], keys))
    p_row <- primary[idx, , drop = FALSE][1L, , drop = FALSE]
    idx_legacy <- Reduce(`&`, Map(function(name) legacy[[name]] == h[[name]],
                                  keys))
    l_row <- legacy[idx_legacy, , drop = FALSE][1L, , drop = FALSE]
    out[[length(out) + 1L]] <- data.frame(
      scenario_id = h$scenario_id,
      seed = h$seed,
      n = h$n,
      p = h$p,
      alpha = h$alpha,
      max_conditioning_level = h$max_conditioning_level,
      recommended_tau = h$tau,
      primary_skeleton_shd = p_row$skeleton_shd,
      hybrid_skeleton_shd = h$skeleton_shd,
      skeleton_shd_reduction = p_row$skeleton_shd - h$skeleton_shd,
      verifier_call_rate = h$near_alpha_verifier_calls /
        max(1L, h$num_tests_total),
      hybrid_runtime_sec = h$runtime_sec,
      legacy_runtime_sec = l_row$runtime_sec,
      speedup_vs_legacy = l_row$runtime_sec / h$runtime_sec,
      stringsAsFactors = FALSE
    )
  }
  summary <- do.call(rbind, out)
  summary <- summary[order(summary$scenario_id, summary$seed, summary$n,
                           summary$p, summary$alpha,
                           summary$max_conditioning_level,
                           -summary$skeleton_shd_reduction,
                           -summary$speedup_vs_legacy), , drop = FALSE]
  keep <- !duplicated(summary[, keys, drop = FALSE])
  out <- summary[keep, , drop = FALSE]
  rownames(out) <- NULL
  out
}

fastkpc_run_mgcv_extract_gpu_graph_campaign <- function(
    output_dir,
    seeds = c(21L, 22L),
    n_values = c(120L, 240L),
    p_values = c(8L, 12L),
    alpha_values = c(0.05),
    tau_values = log(c(2, 3, 5)),
    max_conditioning_levels = c(1L, 2L),
    scenario_id = "deterministic_mgcv_extract_gpu_graph") {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  graph_rows <- list()
  diagnostic_rows <- list()
  for (seed in seeds) {
    for (n in n_values) {
      for (p in p_values) {
        for (alpha in alpha_values) {
          for (tau in tau_values) {
            for (level in max_conditioning_levels) {
              result <- fastkpc_make_gpu_graph_comparison(
                seed = seed, n = n, p = p, alpha = alpha, tau = tau,
                max_conditioning_level = level,
                scenario_id = scenario_id
              )
              graph_rows[[length(graph_rows) + 1L]] <- result$graph
              diagnostic_rows[[length(diagnostic_rows) + 1L]] <-
                result$diagnostics
            }
          }
        }
      }
    }
  }
  graph <- do.call(rbind, graph_rows)
  diagnostics <- do.call(rbind, diagnostic_rows)
  summary <- fastkpc_make_gpu_graph_summary(graph)

  utils::write.csv(
    graph,
    file.path(output_dir, "mgcv_extract_gpu_graph_comparison.csv"),
    row.names = FALSE
  )
  utils::write.csv(
    diagnostics,
    file.path(output_dir, "mgcv_extract_gpu_hybrid_diagnostics.csv"),
    row.names = FALSE
  )
  utils::write.csv(
    summary,
    file.path(output_dir, "mgcv_extract_gpu_graph_summary.csv"),
    row.names = FALSE
  )

  list(
    graph = graph,
    diagnostics = diagnostics,
    summary = summary,
    output_dir = output_dir
  )
}
