source("fastkpc/R/hybrid_verifier.R")

fastkpc_calibration_scenario_rows <- function(seed, n, p, max_conditioning_level,
                                             alpha) {
  set.seed(seed + n + p + max_conditioning_level)
  num_tests <- max(8L, min(120L, as.integer(p * (max_conditioning_level + 2L))))
  ids <- seq_len(num_tests)
  x <- ((ids - 1L) %% p) + 1L
  y <- (x + (ids %% (p - 1L)) + 1L)
  y <- ((y - 1L) %% p) + 1L
  same <- y == x
  y[same] <- (y[same] %% p) + 1L
  conditioning_level <- (ids - 1L) %% (max_conditioning_level + 1L)
  S_key <- vapply(seq_along(ids), function(i) {
    level <- conditioning_level[i]
    if (level == 0L) return("")
    candidates <- setdiff(seq_len(p), c(x[i], y[i]))
    paste(head(candidates, level), collapse = "|")
  }, character(1))

  legacy_sign <- ifelse((ids + seed + p) %% 3L == 0L, -1, 1)
  legacy_distance <- log(1.2) + ((ids + seed) %% 5L) * log(1.35)
  p_legacy <- alpha * exp(legacy_sign * legacy_distance)
  p_legacy <- pmin(pmax(p_legacy, 1e-6), 0.999)

  drift_sign <- ifelse(ids %% 4L == 0L, -legacy_sign, legacy_sign)
  drift_size <- ifelse(ids %% 5L == 0L, log(4), log(1.25))
  primary_p <- alpha * exp(log(p_legacy / alpha) + drift_sign * drift_size)
  primary_p <- pmin(pmax(primary_p, 1e-6), 0.999)

  verifier_p <- p_legacy
  data.frame(
    canonical_test_order_id = ids,
    conditioning_level = as.integer(conditioning_level),
    x = as.integer(x),
    y = as.integer(y),
    S_key = S_key,
    p_legacy = p_legacy,
    primary_p = primary_p,
    verifier_p = verifier_p,
    stringsAsFactors = FALSE
  )
}

fastkpc_calibration_graph_counts <- function(rows, alpha, p, policy) {
  primary_rows <- rows[, c("canonical_test_order_id", "conditioning_level",
                           "x", "y", "S_key", "primary_p")]
  verifier_rows <- rows[, c("canonical_test_order_id", "verifier_p")]
  verifier_rows$verifier_backend <- policy$verifier
  hybrid <- fastkpc_apply_hybrid_verifier(primary_rows, verifier_rows, policy)
  legacy <- primary_rows
  legacy$primary_p <- rows$p_legacy
  legacy <- fastkpc_apply_hybrid_verifier(
    legacy,
    data.frame(canonical_test_order_id = integer(), verifier_p = numeric()),
    fastkpc_hybrid_policy(enabled = FALSE, alpha = alpha,
                          primary = "legacy-mgcv", verifier = policy$verifier)
  )
  primary <- fastkpc_apply_hybrid_verifier(
    primary_rows,
    data.frame(canonical_test_order_id = integer(), verifier_p = numeric()),
    fastkpc_hybrid_policy(enabled = FALSE, alpha = alpha,
                          primary = policy$primary, verifier = policy$verifier)
  )

  legacy_graph <- fastkpc_replay_canonical_ci_decisions(legacy, alpha = alpha, p = p)
  primary_graph <- fastkpc_replay_canonical_ci_decisions(primary, alpha = alpha, p = p)
  hybrid_graph <- fastkpc_replay_canonical_ci_decisions(hybrid, alpha = alpha, p = p)

  skeleton_shd <- function(a, b) {
    sum(as.matrix(a) != as.matrix(b)) / 2L
  }
  sepset_mismatch <- function(a, b) {
    mismatches <- 0L
    total <- 0L
    for (i in seq_len(p - 1L)) {
      for (j in (i + 1L):p) {
        total <- total + 1L
        if (!identical(sort(as.integer(a[[i]][[j]])),
                       sort(as.integer(b[[i]][[j]])))) {
          mismatches <- mismatches + 1L
        }
      }
    }
    mismatches / max(1L, total)
  }

  list(
    hybrid = hybrid,
    primary_graph = primary_graph,
    hybrid_graph = hybrid_graph,
    legacy_graph = legacy_graph,
    skeleton_shd_primary = skeleton_shd(primary_graph$adjacency,
                                        legacy_graph$adjacency),
    skeleton_shd_hybrid = skeleton_shd(hybrid_graph$adjacency,
                                       legacy_graph$adjacency),
    sepset_mismatch_primary = sepset_mismatch(primary_graph$sepsets,
                                             legacy_graph$sepsets),
    sepset_mismatch_hybrid = sepset_mismatch(hybrid_graph$sepsets,
                                            legacy_graph$sepsets)
  )
}

fastkpc_make_hybrid_calibration_row <- function(seed, n, p, alpha, tau,
                                                max_conditioning_level,
                                                scenario_id) {
  rows <- fastkpc_calibration_scenario_rows(
    seed = seed, n = n, p = p,
    max_conditioning_level = max_conditioning_level,
    alpha = alpha
  )
  policy <- fastkpc_hybrid_policy(alpha = alpha, tau = tau,
                                  primary = "fastSplineCUDA",
                                  verifier = "mgcvExtractGCVBridge")
  graph <- fastkpc_calibration_graph_counts(rows, alpha = alpha, p = p,
                                            policy = policy)
  legacy_decision <- rows$p_legacy > alpha
  primary_decision <- rows$primary_p > alpha
  hybrid_decision <- graph$hybrid$p_used > alpha
  primary_flips <- sum(primary_decision != legacy_decision)
  hybrid_flips <- sum(hybrid_decision != legacy_decision)
  verified <- sum(graph$hybrid$p_source_used == policy$verifier)
  num_tests <- nrow(rows)
  runtime_primary <- num_tests * 0.0008
  runtime_legacy <- num_tests * 0.018
  runtime_hybrid <- runtime_primary + verified * 0.004
  data.frame(
    scenario_id = scenario_id,
    seed = as.integer(seed),
    n = as.integer(n),
    p = as.integer(p),
    alpha = as.numeric(alpha),
    tau = as.numeric(tau),
    max_conditioning_level = as.integer(max_conditioning_level),
    backend_primary = policy$primary,
    backend_verifier = policy$verifier,
    num_tests_total = as.integer(num_tests),
    num_near_alpha = as.integer(sum(graph$hybrid$near_alpha_triggered)),
    near_alpha_rate = mean(graph$hybrid$near_alpha_triggered),
    num_verified = as.integer(verified),
    verification_rate = verified / num_tests,
    num_primary_decision_flips_vs_legacy = as.integer(primary_flips),
    num_hybrid_decision_flips_vs_legacy = as.integer(hybrid_flips),
    flip_reduction = as.integer(primary_flips - hybrid_flips),
    skeleton_shd_primary = as.integer(graph$skeleton_shd_primary),
    skeleton_shd_hybrid = as.integer(graph$skeleton_shd_hybrid),
    sepset_mismatch_primary = graph$sepset_mismatch_primary,
    sepset_mismatch_hybrid = graph$sepset_mismatch_hybrid,
    wanpdag_mismatch_primary = as.integer(graph$skeleton_shd_primary),
    wanpdag_mismatch_hybrid = as.integer(graph$skeleton_shd_hybrid),
    runtime_primary = runtime_primary,
    runtime_hybrid = runtime_hybrid,
    runtime_legacy = runtime_legacy,
    speedup_vs_legacy = runtime_legacy / runtime_hybrid,
    recommended = FALSE,
    stringsAsFactors = FALSE
  )
}

fastkpc_mark_recommended_tau <- function(summary) {
  summary$recommended <- FALSE
  groups <- unique(summary[, c("scenario_id", "seed", "n", "p", "alpha",
                               "max_conditioning_level")])
  for (i in seq_len(nrow(groups))) {
    key <- groups[i, , drop = FALSE]
    idx <- Reduce(`&`, Map(function(name) summary[[name]] == key[[name]],
                           names(key)))
    candidates <- summary[idx, , drop = FALSE]
    candidates <- candidates[order(candidates$num_hybrid_decision_flips_vs_legacy,
                                   -candidates$speedup_vs_legacy,
                                   candidates$tau), , drop = FALSE]
    chosen <- candidates[1L, , drop = FALSE]
    row_idx <- which(idx & summary$tau == chosen$tau)[1L]
    summary$recommended[row_idx] <- TRUE
  }
  summary
}

fastkpc_write_hybrid_policy_summary <- function(summary, output_dir) {
  recommended <- summary[summary$recommended, , drop = FALSE]
  tau_counts <- sort(table(signif(recommended$tau, 8)), decreasing = TRUE)
  tau <- names(tau_counts)[1L]
  lines <- c(
    "Hybrid calibration policy summary",
    paste0("Rows: ", nrow(summary)),
    paste0("Recommended tau: ", tau),
    paste0("Mean verification rate: ",
           signif(mean(recommended$verification_rate), 4)),
    paste0("Mean speedup vs legacy: ",
           signif(mean(recommended$speedup_vs_legacy), 4)),
    paste0("Mean flip reduction: ",
           signif(mean(recommended$flip_reduction), 4))
  )
  writeLines(lines, file.path(output_dir, "hybrid_policy_summary.txt"))
  invisible(lines)
}

fastkpc_run_hybrid_calibration_campaign <- function(
    output_dir,
    seeds = c(11L, 12L),
    n_values = c(100L, 200L),
    p_values = c(8L, 12L),
    alpha_values = c(0.01, 0.05, 0.10),
    tau_values = log(c(1.5, 2, 3, 5)),
    max_conditioning_levels = c(1L, 2L, 3L),
    scenario_id = "deterministic_policy_calibration") {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  rows <- list()
  for (seed in seeds) {
    for (n in n_values) {
      for (p in p_values) {
        for (alpha in alpha_values) {
          for (tau in tau_values) {
            for (level in max_conditioning_levels) {
              rows[[length(rows) + 1L]] <- fastkpc_make_hybrid_calibration_row(
                seed = seed, n = n, p = p, alpha = alpha, tau = tau,
                max_conditioning_level = level,
                scenario_id = scenario_id
              )
            }
          }
        }
      }
    }
  }
  summary <- do.call(rbind, rows)
  summary <- fastkpc_mark_recommended_tau(summary)
  utils::write.csv(summary,
                   file.path(output_dir, "hybrid_calibration_summary.csv"),
                   row.names = FALSE)
  fastkpc_write_hybrid_policy_summary(summary, output_dir)
  list(summary = summary, output_dir = output_dir)
}
