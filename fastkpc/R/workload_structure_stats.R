fastkpc_wss_count_by_size <- function(values, S_size) {
  pieces <- tapply(values, S_size, sum, na.rm = TRUE)
  if (length(pieces) == 0L) return("")
  paste(paste0("|S|=", names(pieces), ":", as.integer(pieces)), collapse = ";")
}

fastkpc_workload_structure_stats <- function(test_plan,
                                             dataset_id,
                                             n,
                                             p,
                                             alpha,
                                             max_conditioning_level) {
  required <- c("S_key", "S_size", "conditioning_level")
  missing <- setdiff(required, names(test_plan))
  if (length(missing) > 0L) {
    stop("test_plan missing columns: ", paste(missing, collapse = ", "),
         call. = FALSE)
  }
  levels <- sort(unique(as.integer(test_plan$conditioning_level)))
  rows <- lapply(levels, function(level) {
    subset <- test_plan[as.integer(test_plan$conditioning_level) == level,
                        , drop = FALSE]
    group_counts <- as.integer(table(subset$S_key))
    if (length(group_counts) == 0L) group_counts <- 0L
    near <- if ("near_alpha" %in% names(subset)) as.integer(subset$near_alpha) else
      rep(0L, nrow(subset))
    verifier <- if ("verifier_called" %in% names(subset)) {
      as.integer(subset$verifier_called)
    } else {
      rep(0L, nrow(subset))
    }
    supported <- if ("mgcvExtractGPU_supported" %in% names(subset)) {
      as.logical(subset$mgcvExtractGPU_supported)
    } else {
      rep(NA, nrow(subset))
    }
    data.frame(
      dataset_id = as.character(dataset_id),
      n = as.integer(n),
      p = as.integer(p),
      alpha = as.numeric(alpha),
      max_conditioning_level = as.integer(max_conditioning_level),
      conditioning_level = as.integer(level),
      num_ci_tests = as.integer(nrow(subset)),
      num_unique_S = as.integer(length(unique(subset$S_key))),
      num_same_setup_groups = as.integer(sum(group_counts >= 2L)),
      targets_per_setup_p50 = as.numeric(stats::median(group_counts)),
      targets_per_setup_p95 = as.numeric(stats::quantile(group_counts, 0.95,
                                                         names = FALSE)),
      targets_per_setup_max = as.integer(max(group_counts)),
      num_tests_by_S_size = fastkpc_wss_count_by_size(rep(1L, nrow(subset)),
                                                      subset$S_size),
      runtime_by_S_size = NA_real_,
      near_alpha_tests_by_S_size = fastkpc_wss_count_by_size(near, subset$S_size),
      verifier_calls_by_S_size = fastkpc_wss_count_by_size(verifier, subset$S_size),
      mgcvExtractGPU_supported_tests = as.integer(sum(supported %in% TRUE,
                                                      na.rm = TRUE)),
      mgcvExtractGPU_unsupported_tests = as.integer(sum(supported %in% FALSE,
                                                        na.rm = TRUE)),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

fastkpc_write_workload_structure_stats <- function(
  stats,
  output_dir = file.path("fastkpc", "artifacts", "workload_structure_stats")
) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  csv_path <- file.path(output_dir, "workload_structure_stats.csv")
  report_path <- file.path(output_dir, "workload_structure_stats_report.md")
  utils::write.csv(stats, csv_path, row.names = FALSE, na = "")
  total_tests <- sum(as.integer(stats$num_ci_tests), na.rm = TRUE)
  same_setup <- sum(as.integer(stats$num_same_setup_groups), na.rm = TRUE)
  unsupported <- sum(as.integer(stats$mgcvExtractGPU_unsupported_tests), na.rm = TRUE)
  lines <- c(
    "# fastkpc Workload Structure Stats",
    "",
    paste0("- Total CI tests summarized: ", total_tests),
    paste0("- same-setup multiplicity groups: ", same_setup),
    paste0("- mgcvExtractGPU unsupported tests: ", unsupported),
    "- |S| > 2 cases indicate high-order additive smooth workload.",
    "- Runtime by |S| is NA when full expensive CI was not run."
  )
  writeLines(lines, report_path)
  list(csv_path = csv_path, report_path = report_path, stats = stats)
}

fastkpc_cache_aware_workload_stats <- function(residual_requests,
                                               dataset_id,
                                               n,
                                               p,
                                               alpha,
                                               max_conditioning_level) {
  required <- c("setup_fingerprint", "canonical_test_order_id", "target_id",
                "cache_hit", "device_solve_called", "S_size",
                "conditioning_level")
  missing <- setdiff(required, names(residual_requests))
  if (length(missing) > 0L) {
    stop("residual_requests missing columns: ", paste(missing, collapse = ", "),
         call. = FALSE)
  }
  groups <- split(residual_requests, residual_requests$setup_fingerprint)
  rows <- lapply(names(groups), function(key) {
    subset <- groups[[key]]
    data.frame(
      dataset_id = as.character(dataset_id),
      n = as.integer(n),
      p = as.integer(p),
      alpha = as.numeric(alpha),
      max_conditioning_level = as.integer(max_conditioning_level),
      setup_fingerprint = as.character(key),
      conditioning_level = as.integer(subset$conditioning_level[1L]),
      S_size = as.integer(subset$S_size[1L]),
      ci_tests_per_setup =
        as.integer(length(unique(subset$canonical_test_order_id))),
      raw_residual_requests_per_setup = as.integer(nrow(subset)),
      unique_targets_per_setup =
        as.integer(length(unique(as.character(subset$target_id)))),
      uncached_targets_per_setup =
        as.integer(sum(!as.logical(subset$cache_hit), na.rm = TRUE)),
      device_solve_calls_per_setup =
        as.integer(sum(as.logical(subset$device_solve_called), na.rm = TRUE)),
      cache_hit_rate = mean(as.logical(subset$cache_hit), na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}
