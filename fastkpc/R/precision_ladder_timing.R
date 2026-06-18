fastkpc_timing_component_names <- function() {
  c(
    "mgcv_setup_cpu_ms", "setup_cache_lookup_ms", "host_to_device_ms",
    "spectral_prepare_ms", "gcv_score_ms", "linear_solve_ms",
    "residual_materialize_ms", "device_to_host_ms", "ci_test_ms",
    "canonical_replay_ms"
  )
}

fastkpc_precision_ladder_timing_row <- function(
  backend,
  mode,
  solve_source,
  native_gpu_solve_used = FALSE,
  true_batched_kernel = FALSE,
  targets_per_setup = 1L,
  setup_reuse_count = NA_integer_,
  mgcv_setup_cpu_ms = NA_real_,
  setup_cache_lookup_ms = NA_real_,
  setup_cache_hit = NA,
  host_to_device_ms = NA_real_,
  spectral_prepare_ms = NA_real_,
  gcv_score_ms = NA_real_,
  linear_solve_ms = NA_real_,
  residual_materialize_ms = NA_real_,
  device_to_host_ms = NA_real_,
  ci_test_ms = NA_real_,
  canonical_replay_ms = NA_real_,
  total_ms = NA_real_,
  gcv_grid_points = NA_integer_,
  gcv_grid_boundary_hit = NA,
  condition_estimate = NA_real_,
  fallback_reason = NA_character_,
  setup_fingerprint = NA_character_,
  target_fingerprint = NA_character_
) {
  components <- c(
    mgcv_setup_cpu_ms, setup_cache_lookup_ms, host_to_device_ms,
    spectral_prepare_ms, gcv_score_ms, linear_solve_ms,
    residual_materialize_ms, device_to_host_ms, ci_test_ms,
    canonical_replay_ms
  )
  known_sum <- sum(as.numeric(components), na.rm = TRUE)
  if (!is.finite(total_ms)) total_ms <- known_sum
  note <- if (is.finite(total_ms) && total_ms + 1e-9 < known_sum) {
    "total_ms less than known component sum"
  } else if (any(is.na(components))) {
    "partial timing components"
  } else {
    ""
  }
  data.frame(
    backend = as.character(backend),
    mode = as.character(mode),
    solve_source = as.character(solve_source),
    native_gpu_solve_used = isTRUE(native_gpu_solve_used),
    true_batched_kernel = isTRUE(true_batched_kernel),
    targets_per_setup = as.integer(targets_per_setup),
    setup_reuse_count = as.integer(setup_reuse_count),
    mgcv_setup_cpu_ms = as.numeric(mgcv_setup_cpu_ms),
    setup_cache_lookup_ms = as.numeric(setup_cache_lookup_ms),
    setup_cache_hit = setup_cache_hit,
    host_to_device_ms = as.numeric(host_to_device_ms),
    spectral_prepare_ms = as.numeric(spectral_prepare_ms),
    gcv_score_ms = as.numeric(gcv_score_ms),
    linear_solve_ms = as.numeric(linear_solve_ms),
    residual_materialize_ms = as.numeric(residual_materialize_ms),
    device_to_host_ms = as.numeric(device_to_host_ms),
    ci_test_ms = as.numeric(ci_test_ms),
    canonical_replay_ms = as.numeric(canonical_replay_ms),
    total_ms = as.numeric(total_ms),
    gcv_grid_points = as.integer(gcv_grid_points),
    gcv_grid_boundary_hit = gcv_grid_boundary_hit,
    condition_estimate = as.numeric(condition_estimate),
    fallback_reason = as.character(fallback_reason),
    setup_fingerprint = as.character(setup_fingerprint),
    target_fingerprint = as.character(target_fingerprint),
    timing_accounting_note = note,
    stringsAsFactors = FALSE
  )
}

fastkpc_classify_timing_bottleneck <- function(row) {
  if (is.data.frame(row)) row <- row[1L, , drop = FALSE]
  values <- c(
    mgcv_setup_dominated = as.numeric(row$mgcv_setup_cpu_ms),
    gcv_dominated = sum(c(as.numeric(row$spectral_prepare_ms),
                          as.numeric(row$gcv_score_ms)), na.rm = TRUE),
    linear_solve_dominated = as.numeric(row$linear_solve_ms),
    ci_dominated = as.numeric(row$ci_test_ms),
    replay_dominated = as.numeric(row$canonical_replay_ms)
  )
  values[!is.finite(values)] <- 0
  if (max(values) <= 0) return("unclassified")
  names(values)[which.max(values)]
}

fastkpc_timing_markdown_table <- function(df) {
  shown <- df
  shown[] <- lapply(shown, function(x) {
    if (is.numeric(x)) x <- signif(x, 6)
    x <- as.character(x)
    x[is.na(x)] <- ""
    x
  })
  header <- paste(names(shown), collapse = " | ")
  divider <- paste(rep("---", ncol(shown)), collapse = " | ")
  rows <- if (nrow(shown) == 0L) "" else
    paste(apply(shown, 1L, paste, collapse = " | "), collapse = " |\n| ")
  paste0("| ", header, " |\n| ", divider, " |\n",
         if (nzchar(rows)) paste0("| ", rows, " |\n") else "")
}

fastkpc_write_precision_ladder_timing_report <- function(
  rows,
  output_dir = file.path("fastkpc", "artifacts", "precision_ladder_timing")
) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  rows$bottleneck <- vapply(seq_len(nrow(rows)), function(i) {
    fastkpc_classify_timing_bottleneck(rows[i, , drop = FALSE])
  }, character(1))
  csv_path <- file.path(output_dir, "precision_ladder_timing.csv")
  report_path <- file.path(output_dir, "precision_ladder_timing_report.md")
  utils::write.csv(rows, csv_path, row.names = FALSE, na = "")
  lines <- c(
    "# fastkpc Precision Ladder Timing Report",
    "",
    "This report distinguishes same-setup batch overhead reduction from a true batched solve kernel.",
    "same-setup batch diagnostics must keep true_batched_kernel = false until a fused kernel exists.",
    "",
    "## Bottlenecks",
    "",
    fastkpc_timing_markdown_table(rows),
    "",
    "## Kernel Decision Signal",
    "",
    "A true batched solve kernel is likely useful only when linear_solve_dominated rows also have high targets_per_setup."
  )
  writeLines(lines, report_path)
  list(csv_path = csv_path, report_path = report_path, rows = rows)
}
