fail <- function(message) stop(message, call. = FALSE)

arguments <- commandArgs(trailingOnly = TRUE)
if (!length(arguments) %in% c(2L, 3L, 4L)) {
  fail(paste(
    "usage: analyze_phase10_legacy_component_wave_trace.R",
    "WAVE_RECEIPT BASELINE_RECEIPT [OUTPUT] [TWO_BATCH_BENCHMARK]"
  ))
}

wave_path <- arguments[[1L]]
baseline_path <- arguments[[2L]]
output_path <- if (length(arguments) == 3L) arguments[[3L]] else NULL
if (length(arguments) == 4L) output_path <- arguments[[3L]]
benchmark_path <- if (length(arguments) == 4L) arguments[[4L]] else NULL
for (path in c(wave_path, baseline_path)) {
  if (!file.exists(path) || dir.exists(path)) {
    fail(paste("receipt does not exist:", path))
  }
}
if (!is.null(output_path) && (file.exists(output_path) || dir.exists(output_path))) {
  fail("analysis output already exists")
}
if (!is.null(benchmark_path) &&
    (!file.exists(benchmark_path) || dir.exists(benchmark_path))) {
  fail("two-batch benchmark does not exist")
}

wave_receipt <- readRDS(wave_path)
baseline_receipt <- readRDS(baseline_path)
if (!isTRUE(wave_receipt$audit$pass) ||
    !isTRUE(baseline_receipt$audit$pass)) {
  fail("wave analysis requires passing contract receipts")
}

trace <- wave_receipt$result$legacy_component_wave_group_trace
wave_summary <- wave_receipt$result$summary
baseline_summary <- baseline_receipt$result$summary
histogram <- baseline_receipt$result$legacy_component_eig_miss_histogram
required_trace_fields <- c(
  "level", "wave_index", "group_position", "ready_bucket_count",
  "selected_task_count", "executed_pair_count", "component_request_count",
  "component_hit_count", "component_miss_count", "residual_host_ms",
  "full_eig_cuda_ms", "component_build_cuda_ms", "dcov_host_ms"
)
required_histogram_fields <- c(
  "miss_count", "batch_count", "component_count", "full_eig_cuda_ms"
)
missing_trace_fields <- setdiff(required_trace_fields, names(trace))
missing_histogram_fields <- setdiff(required_histogram_fields, names(histogram))
if (!is.data.frame(trace) || length(missing_trace_fields)) {
  fail(paste(
    "wave trace is missing fields:", paste(missing_trace_fields, collapse = ",")
  ))
}
if (!is.data.frame(histogram) || length(missing_histogram_fields)) {
  fail(paste(
    "baseline histogram is missing fields:",
    paste(missing_histogram_fields, collapse = ",")
  ))
}

summary_value <- function(summary, field) {
  value <- summary[[field]]
  if (is.null(value) || length(value) != 1L || is.na(value)) {
    fail(paste("summary field is malformed:", field))
  }
  as.numeric(value)
}

accounting <- c(
  group_count = nrow(trace),
  request_count = sum(trace$component_request_count),
  hit_count = sum(trace$component_hit_count),
  miss_count = sum(trace$component_miss_count),
  pair_count = sum(trace$executed_pair_count)
)
expected_accounting <- c(
  group_count = summary_value(wave_summary, "legacy_component_wave_group_count"),
  request_count = summary_value(wave_summary, "legacy_component_cache_request_count"),
  hit_count = summary_value(wave_summary, "legacy_component_cache_hit_count"),
  miss_count = summary_value(wave_summary, "legacy_component_cache_miss_count"),
  pair_count = summary_value(wave_summary, "cuda_legacy_compatible_pair_count")
)
if (length(accounting) != length(expected_accounting) ||
    any(unname(accounting) != unname(expected_accounting))) {
  fail("wave trace request/hit/miss/pair accounting does not close")
}
if (any(trace$component_request_count !=
        trace$component_hit_count + trace$component_miss_count)) {
  fail("one or more wave groups have malformed cache accounting")
}

wave_capacity <- as.integer(summary_value(
  wave_summary, "legacy_component_wave_capacity_requested"
))
if (wave_capacity < 2L) fail("wave receipt does not contain a multi-group wave")
wave_key <- paste(trace$level, trace$wave_index, sep = "|")
wave_rows <- split(seq_len(nrow(trace)), wave_key)
if (any(lengths(wave_rows) > wave_capacity)) {
  fail("wave trace exceeds its configured capacity")
}
if (any(vapply(wave_rows, function(rows) {
  identical(sort(trace$group_position[rows]), seq_along(rows))
}, logical(1L)) == FALSE)) {
  fail("wave group positions are not contiguous")
}

summarize_wave <- function(rows) {
  rows <- rows[order(trace$group_position[rows])]
  first <- rows[[1L]]
  second <- if (length(rows) >= 2L) rows[[2L]] else NA_integer_
  second_value <- function(field) {
    if (is.na(second)) 0 else as.numeric(trace[[field]][[second]])
  }
  eig_a <- as.numeric(trace$full_eig_cuda_ms[[first]])
  eig_b <- second_value("full_eig_cuda_ms")
  component_a <- as.numeric(trace$component_build_cuda_ms[[first]])
  component_b <- second_value("component_build_cuda_ms")
  data.frame(
    level = trace$level[[first]],
    wave_index = trace$wave_index[[first]],
    group_count = length(rows),
    miss_a = trace$component_miss_count[[first]],
    miss_b = second_value("component_miss_count"),
    eig_a_ms = eig_a,
    eig_b_ms = eig_b,
    sequential_eig_ms = eig_a + eig_b,
    perfect_eig_overlap_ms = min(eig_a, eig_b),
    perfect_eig_makespan_ms = max(eig_a, eig_b),
    component_a_ms = component_a,
    component_b_ms = component_b,
    sequential_component_ms = component_a + component_b,
    perfect_component_overlap_ms = min(component_a, component_b),
    perfect_component_makespan_ms = max(component_a, component_b),
    stringsAsFactors = FALSE
  )
}
waves <- do.call(rbind, lapply(wave_rows, summarize_wave))
rownames(waves) <- NULL

pair_rows <- waves$group_count == 2L
normalized_miss_a <- pmin(waves$miss_a[pair_rows], waves$miss_b[pair_rows])
normalized_miss_b <- pmax(waves$miss_a[pair_rows], waves$miss_b[pair_rows])
pair_frequency <- aggregate(
  rep.int(1L, sum(pair_rows)),
  by = list(miss_a = normalized_miss_a, miss_b = normalized_miss_b),
  FUN = sum
)
names(pair_frequency)[[3L]] <- "wave_count"
pair_frequency <- pair_frequency[order(
  -pair_frequency$wave_count, pair_frequency$miss_a, pair_frequency$miss_b
), , drop = FALSE]
rownames(pair_frequency) <- NULL

saturated <- histogram$miss_count >= 24L & histogram$component_count > 0
if (!any(saturated)) fail("baseline histogram has no saturated eig rows")
saturated_ms_per_component <-
  sum(histogram$full_eig_cuda_ms[saturated]) /
  sum(histogram$component_count[saturated])
global_saturated_eig_ms <-
  sum(trace$component_miss_count) * saturated_ms_per_component

wave_elapsed_sec <- as.numeric(wave_receipt$audit$elapsed_sec)
baseline_elapsed_sec <- as.numeric(baseline_receipt$audit$elapsed_sec)
wave_regression_sec <- wave_elapsed_sec - baseline_elapsed_sec
perfect_eig_overlap_sec <- sum(waves$perfect_eig_overlap_ms) / 1000
perfect_component_overlap_sec <-
  sum(waves$perfect_component_overlap_ms) / 1000

metrics <- list(
  schema_version = "phase10-legacy-component-wave-opportunity-v1",
  wave_receipt = normalizePath(wave_path, winslash = "/", mustWork = TRUE),
  baseline_receipt = normalizePath(
    baseline_path, winslash = "/", mustWork = TRUE
  ),
  wave_capacity = wave_capacity,
  wave_count = nrow(waves),
  paired_wave_count = sum(pair_rows),
  singleton_wave_count = sum(!pair_rows),
  group_count = nrow(trace),
  component_miss_count = sum(trace$component_miss_count),
  traced_full_eig_sec = sum(trace$full_eig_cuda_ms) / 1000,
  summary_full_eig_sec =
    summary_value(wave_summary, "cuda_dcov_full_eig_ms") / 1000,
  perfect_eig_overlap_upper_bound_sec = perfect_eig_overlap_sec,
  perfect_component_overlap_upper_bound_sec = perfect_component_overlap_sec,
  saturated_ms_per_component = saturated_ms_per_component,
  global_saturated_eig_sec = global_saturated_eig_ms / 1000,
  wave_elapsed_sec = wave_elapsed_sec,
  baseline_elapsed_sec = baseline_elapsed_sec,
  wave_regression_sec = wave_regression_sec,
  perfect_eig_net_upper_bound_sec =
    perfect_eig_overlap_sec - wave_regression_sec,
  perfect_component_net_upper_bound_sec =
    perfect_component_overlap_sec - wave_regression_sec
)
if (abs(metrics$traced_full_eig_sec - metrics$summary_full_eig_sec) > 1e-6) {
  fail("wave trace full-eig time does not match its summary")
}

if (!is.null(benchmark_path)) {
  if (!requireNamespace("digest", quietly = TRUE)) fail("digest is required")
  benchmark_receipt <- readRDS(benchmark_path)
  benchmark <- benchmark_receipt$benchmark
  if (!isTRUE(benchmark_receipt$summary$parity_pass) ||
      !is.data.frame(benchmark) ||
      !all(c(
        "first_components", "second_components",
        "projected_saving_fraction"
      ) %in% names(benchmark))) {
    fail("two-batch benchmark receipt is malformed")
  }
  benchmark_key <- paste(
    pmin(benchmark$first_components, benchmark$second_components),
    pmax(benchmark$first_components, benchmark$second_components), sep = "|"
  )
  if (anyDuplicated(benchmark_key)) {
    fail("two-batch benchmark contains duplicate miss shapes")
  }
  wave_miss_key <- paste(
    pmin(waves$miss_a, waves$miss_b),
    pmax(waves$miss_a, waves$miss_b), sep = "|"
  )
  matched_benchmark <- match(wave_miss_key, benchmark_key)
  benchmark_fraction <- benchmark$projected_saving_fraction[matched_benchmark]
  positive_pair <- waves$group_count == 2L &
    waves$miss_a > 0L & waves$miss_b > 0L
  if (any(positive_pair & is.na(benchmark_fraction))) {
    fail("two-batch benchmark does not cover every positive-miss wave")
  }
  waves$benchmark_saving_fraction <- ifelse(
    is.na(benchmark_fraction), 0, benchmark_fraction
  )
  waves$benchmark_projected_saving_ms <-
    waves$sequential_eig_ms * waves$benchmark_saving_fraction
  projected_saving_sec <- sum(waves$benchmark_projected_saving_ms) / 1000
  metrics$two_batch_benchmark_path <- normalizePath(
    benchmark_path, winslash = "/", mustWork = TRUE
  )
  metrics$two_batch_benchmark_sha256 <- unname(digest::digest(
    file = benchmark_path, algo = "sha256", serialize = FALSE
  ))
  metrics$two_batch_benchmark_positive_pair_coverage <-
    sum(positive_pair & !is.na(benchmark_fraction)) / sum(positive_pair)
  metrics$two_batch_projected_eig_saving_sec <- projected_saving_sec
  metrics$two_batch_projected_net_saving_sec <-
    projected_saving_sec - wave_regression_sec
}

analysis <- list(
  schema_version = metrics$schema_version,
  metrics = metrics,
  accounting = accounting,
  pair_frequency = pair_frequency,
  waves = waves
)
if (!is.null(output_path)) {
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  temporary_path <- paste0(output_path, ".tmp")
  on.exit(unlink(temporary_path, force = TRUE), add = TRUE)
  saveRDS(analysis, temporary_path, compress = "xz", version = 3L)
  if (!file.rename(temporary_path, output_path)) fail("failed to publish analysis")
}

cat(sprintf(
  paste0(
    "PASS wave trace analysis; groups=%d; waves=%d; paired=%d; ",
    "eig=%.3fs; perfect_eig_overlap=%.3fs; wave_regression=%.3fs; ",
    "perfect_eig_net_upper_bound=%.3fs; saturated_global_eig=%.3fs\n"
  ),
  metrics$group_count, metrics$wave_count, metrics$paired_wave_count,
  metrics$traced_full_eig_sec,
  metrics$perfect_eig_overlap_upper_bound_sec,
  metrics$wave_regression_sec,
  metrics$perfect_eig_net_upper_bound_sec,
  metrics$global_saturated_eig_sec
))
if (!is.null(benchmark_path)) {
  cat(sprintf(
    "two_batch_projected_eig_saving=%.3fs; projected_net=%.3fs\n",
    metrics$two_batch_projected_eig_saving_sec,
    metrics$two_batch_projected_net_saving_sec
  ))
}
cat("Most frequent normalized miss pairs:\n")
print(utils::head(pair_frequency, 20L), row.names = FALSE)
