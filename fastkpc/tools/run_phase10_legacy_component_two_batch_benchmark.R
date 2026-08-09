fail <- function(message) stop(message, call. = FALSE)

arguments <- commandArgs(trailingOnly = TRUE)
if (!length(arguments) %in% c(3L, 4L, 5L)) {
  fail(paste(
    "usage: run_phase10_legacy_component_two_batch_benchmark.R",
    "OPPORTUNITY BINARY OUTPUT [TOP_PAIR_COUNT] [TRIALS]"
  ))
}
if (!requireNamespace("digest", quietly = TRUE)) fail("digest is required")

opportunity_path <- arguments[[1L]]
binary_path <- arguments[[2L]]
output_path <- arguments[[3L]]
top_pair_count <- if (length(arguments) >= 4L) {
  as.integer(arguments[[4L]])
} else {
  75L
}
trials <- if (length(arguments) >= 5L) as.integer(arguments[[5L]]) else 5L
if (!file.exists(opportunity_path) || dir.exists(opportunity_path) ||
    !file.exists(binary_path) || dir.exists(binary_path)) {
  fail("benchmark input or binary does not exist")
}
if (file.exists(output_path) || dir.exists(output_path)) {
  fail("benchmark output already exists")
}
if (is.na(top_pair_count) || top_pair_count < 1L ||
    is.na(trials) || trials < 3L) {
  fail("benchmark count or trial count is invalid")
}

opportunity <- readRDS(opportunity_path)
waves <- opportunity$waves
required_wave_fields <- c(
  "group_count", "miss_a", "miss_b", "sequential_eig_ms",
  "perfect_eig_overlap_ms"
)
if (!is.data.frame(waves) || length(setdiff(required_wave_fields, names(waves)))) {
  fail("opportunity artifact has a malformed wave table")
}
positive <- waves[
  waves$group_count == 2L & waves$miss_a > 0L & waves$miss_b > 0L,
  , drop = FALSE
]
positive$normalized_a <- pmin(positive$miss_a, positive$miss_b)
positive$normalized_b <- pmax(positive$miss_a, positive$miss_b)
pair_opportunity <- aggregate(
  cbind(
    wave_count = rep.int(1L, nrow(positive)),
    sequential_eig_ms = positive$sequential_eig_ms,
    perfect_eig_overlap_ms = positive$perfect_eig_overlap_ms
  ),
  by = list(
    first_components = positive$normalized_a,
    second_components = positive$normalized_b
  ),
  FUN = sum
)
pair_opportunity <- pair_opportunity[order(
  -pair_opportunity$perfect_eig_overlap_ms,
  pair_opportunity$first_components,
  pair_opportunity$second_components
), , drop = FALSE]
rownames(pair_opportunity) <- NULL
selected <- utils::head(pair_opportunity, top_pair_count)
stress <- data.frame(
  first_components = c(16L, 24L, 47L),
  second_components = c(16L, 24L, 47L)
)
selected <- unique(rbind(
  selected[, c("first_components", "second_components")], stress
))

parse_output <- function(output) {
  line <- output[[length(output)]]
  tokens <- strsplit(line, " +")[[1L]]
  key_value <- tokens[grepl("^[^=]+=.+$", tokens)]
  keys <- sub("=.*$", "", key_value)
  values <- sub("^[^=]+=", "", key_value)
  if (!length(keys) || anyDuplicated(keys)) fail("benchmark output is malformed")
  parsed <- as.list(values)
  names(parsed) <- keys
  parsed
}

rows <- vector("list", nrow(selected))
raw_output <- vector("list", nrow(selected))
binary <- normalizePath(binary_path, winslash = "/", mustWork = TRUE)
for (index in seq_len(nrow(selected))) {
  first <- as.integer(selected$first_components[[index]])
  second <- as.integer(selected$second_components[[index]])
  output <- system2(
    binary,
    args = c(351L, first, 8L, trials, second),
    stdout = TRUE,
    stderr = TRUE,
    env = "CUDA_VISIBLE_DEVICES=0"
  )
  status <- attr(output, "status")
  if (!is.null(status) && status != 0L) {
    fail(paste(
      "two-batch benchmark failed for", first, second,
      paste(output, collapse = "\n")
    ))
  }
  parsed <- parse_output(output)
  required <- c(
    "first_components", "second_components", "lanes_per_batch", "trials",
    "alternate_mode", "separate_device_median_ms",
    "alternate_device_median_ms", "separate_to_alternate_device_speedup",
    "alternate_batch_overlap_median_ms", "eigenvalue_bit_mismatch_count",
    "eigenvector_bit_mismatch_count", "serial_failure_count",
    "concurrent_failure_count"
  )
  if (length(setdiff(required, names(parsed)))) {
    fail("two-batch benchmark output is missing required fields")
  }
  numeric_names <- setdiff(names(parsed), "alternate_mode")
  for (name in numeric_names) parsed[[name]] <- as.numeric(parsed[[name]])
  if (parsed$first_components != first || parsed$second_components != second ||
      parsed$lanes_per_batch != 8 || parsed$trials != trials ||
      !identical(parsed$alternate_mode, "split") ||
      parsed$eigenvalue_bit_mismatch_count != 0 ||
      parsed$eigenvector_bit_mismatch_count != 0 ||
      parsed$serial_failure_count != 0 ||
      parsed$concurrent_failure_count != 0) {
    fail("two-batch benchmark parity or identity gate failed")
  }
  rows[[index]] <- parsed
  raw_output[[index]] <- output
  cat(sprintf(
    "PASS pair %d/%d (%d,%d): sequential=%.3fms split=%.3fms speedup=%.4fx\n",
    index, nrow(selected), first, second,
    parsed$separate_device_median_ms,
    parsed$alternate_device_median_ms,
    parsed$separate_to_alternate_device_speedup
  ))
}

all_fields <- unique(unlist(lapply(rows, names), use.names = FALSE))
benchmark <- as.data.frame(do.call(rbind, lapply(rows, function(row) {
  values <- rep(NA_character_, length(all_fields))
  names(values) <- all_fields
  values[names(row)] <- vapply(row, as.character, character(1L))
  values
})), stringsAsFactors = FALSE)
numeric_fields <- setdiff(names(benchmark), "alternate_mode")
benchmark[numeric_fields] <- lapply(benchmark[numeric_fields], as.numeric)

benchmark_key <- paste(
  pmin(benchmark$first_components, benchmark$second_components),
  pmax(benchmark$first_components, benchmark$second_components), sep = "|"
)
opportunity_key <- paste(
  pair_opportunity$first_components,
  pair_opportunity$second_components, sep = "|"
)
matched <- match(benchmark_key, opportunity_key)
benchmark$canonical_wave_count <- pair_opportunity$wave_count[matched]
benchmark$canonical_sequential_eig_ms <-
  pair_opportunity$sequential_eig_ms[matched]
benchmark$canonical_perfect_overlap_ms <-
  pair_opportunity$perfect_eig_overlap_ms[matched]
benchmark$projected_saving_fraction <- pmax(
  0,
  1 - benchmark$alternate_device_median_ms /
    benchmark$separate_device_median_ms
)
benchmark$projected_canonical_saving_ms <-
  benchmark$canonical_sequential_eig_ms * benchmark$projected_saving_fraction

covered <- !is.na(benchmark$canonical_perfect_overlap_ms)
total_perfect_overlap_ms <- sum(pair_opportunity$perfect_eig_overlap_ms)
covered_perfect_overlap_ms <-
  sum(benchmark$canonical_perfect_overlap_ms[covered])
projected_covered_saving_ms <-
  sum(benchmark$projected_canonical_saving_ms[covered])
uncovered_perfect_overlap_ms <-
  total_perfect_overlap_ms - covered_perfect_overlap_ms
wave_regression_ms <- opportunity$metrics$wave_regression_sec * 1000

gpu_identity <- system2(
  "nvidia-smi",
  args = c(
    "--query-gpu=index,uuid,name,driver_version,pci.bus_id",
    "--format=csv,noheader"
  ),
  stdout = TRUE,
  stderr = TRUE
)
receipt <- list(
  schema_version = "phase10-legacy-component-two-batch-benchmark-v1",
  generated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  opportunity_path = normalizePath(
    opportunity_path, winslash = "/", mustWork = TRUE
  ),
  opportunity_sha256 = unname(digest::digest(
    file = opportunity_path, algo = "sha256", serialize = FALSE
  )),
  binary_path = binary,
  binary_sha256 = unname(digest::digest(
    file = binary, algo = "sha256", serialize = FALSE
  )),
  source_path = normalizePath(
    "fastkpc/tools/phase10_cuda_full_eig_lane_benchmark.cu",
    winslash = "/", mustWork = TRUE
  ),
  source_sha256 = unname(digest::digest(
    file = "fastkpc/tools/phase10_cuda_full_eig_lane_benchmark.cu",
    algo = "sha256", serialize = FALSE
  )),
  gpu_identity = gpu_identity,
  top_pair_count_requested = top_pair_count,
  trials = trials,
  summary = list(
    benchmark_pair_count = nrow(benchmark),
    canonical_pair_shape_count = nrow(pair_opportunity),
    total_perfect_overlap_sec = total_perfect_overlap_ms / 1000,
    covered_perfect_overlap_sec = covered_perfect_overlap_ms / 1000,
    opportunity_coverage = covered_perfect_overlap_ms / total_perfect_overlap_ms,
    projected_covered_saving_sec = projected_covered_saving_ms / 1000,
    uncovered_perfect_overlap_upper_bound_sec =
      uncovered_perfect_overlap_ms / 1000,
    wave_regression_sec = wave_regression_ms / 1000,
    projected_plus_uncovered_upper_bound_net_sec =
      (projected_covered_saving_ms + uncovered_perfect_overlap_ms -
        wave_regression_ms) / 1000,
    minimum_speedup = min(benchmark$separate_to_alternate_device_speedup),
    median_speedup = stats::median(
      benchmark$separate_to_alternate_device_speedup
    ),
    maximum_speedup = max(benchmark$separate_to_alternate_device_speedup),
    parity_pass = all(
      benchmark$eigenvalue_bit_mismatch_count == 0 &
      benchmark$eigenvector_bit_mismatch_count == 0 &
      benchmark$serial_failure_count == 0 &
      benchmark$concurrent_failure_count == 0
    )
  ),
  pair_opportunity = pair_opportunity,
  benchmark = benchmark,
  raw_output = raw_output
)

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
temporary_path <- paste0(output_path, ".tmp")
on.exit(unlink(temporary_path, force = TRUE), add = TRUE)
saveRDS(receipt, temporary_path, compress = "xz", version = 3L)
if (!file.rename(temporary_path, output_path)) fail("failed to publish receipt")
cat(sprintf(
  paste0(
    "PASS two-batch benchmark; pairs=%d; coverage=%.2f%%; ",
    "projected_covered_saving=%.3fs; regression=%.3fs; ",
    "projected_plus_uncovered_net_upper_bound=%.3fs; output=%s\n"
  ),
  receipt$summary$benchmark_pair_count,
  100 * receipt$summary$opportunity_coverage,
  receipt$summary$projected_covered_saving_sec,
  receipt$summary$wave_regression_sec,
  receipt$summary$projected_plus_uncovered_upper_bound_net_sec,
  output_path
))
