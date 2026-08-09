fail <- function(message) stop(message, call. = FALSE)

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) != 5L) {
  fail(paste(
    "usage: finalize_phase10_full_eig_component_closure.R",
    "LANE_BINARY SOLVER_BINARY BASELINE_RECEIPT",
    "SELECTIVE_OPPORTUNITY OUTPUT"
  ))
}
if (!requireNamespace("digest", quietly = TRUE)) fail("digest is required")

lane_binary <- arguments[[1L]]
solver_binary <- arguments[[2L]]
baseline_path <- arguments[[3L]]
opportunity_path <- arguments[[4L]]
output_path <- arguments[[5L]]
inputs <- c(lane_binary, solver_binary, baseline_path, opportunity_path)
if (any(!file.exists(inputs)) || any(dir.exists(inputs))) {
  fail("one or more full-eig closure inputs do not exist")
}
if (file.exists(output_path) || dir.exists(output_path)) {
  fail("full-eig closure output already exists")
}

parse_key_values <- function(output) {
  if (!length(output)) fail("benchmark produced no output")
  tokens <- strsplit(output[[length(output)]], " +")[[1L]]
  fields <- tokens[grepl("^[^=]+=.+$", tokens)]
  keys <- sub("=.*$", "", fields)
  values <- sub("^[^=]+=", "", fields)
  if (!length(keys) || anyDuplicated(keys)) {
    fail("benchmark output is malformed")
  }
  result <- as.list(values)
  names(result) <- keys
  result
}

run_benchmark <- function(binary, args) {
  output <- system2(
    normalizePath(binary, winslash = "/", mustWork = TRUE),
    args = args,
    stdout = TRUE,
    stderr = TRUE,
    env = "CUDA_VISIBLE_DEVICES=0"
  )
  status <- attr(output, "status")
  if (!is.null(status) && status != 0L) {
    fail(paste("benchmark failed:", paste(output, collapse = "\n")))
  }
  list(parsed = parse_key_values(output), raw = output)
}

numeric_fields <- function(value, excluded = character()) {
  for (name in setdiff(names(value), excluded)) {
    converted <- suppressWarnings(as.numeric(value[[name]]))
    if (is.na(converted)) fail(paste("non-numeric benchmark field:", name))
    value[[name]] <- converted
  }
  value
}

lane_shapes <- data.frame(
  components = c(2L, 8L, 9L, 9L, 16L, 16L, 48L, 48L),
  lanes = c(8L, 8L, 8L, 9L, 8L, 16L, 8L, 16L),
  trials = c(15L, 15L, 11L, 11L, 11L, 11L, 9L, 9L)
)
lane_rows <- vector("list", nrow(lane_shapes))
lane_raw <- vector("list", nrow(lane_shapes))
for (index in seq_len(nrow(lane_shapes))) {
  shape <- lane_shapes[index, , drop = FALSE]
  run <- run_benchmark(lane_binary, c(
    351L, shape$components, shape$lanes, shape$trials
  ))
  parsed <- numeric_fields(run$parsed, excluded = "alternate_mode")
  required <- c(
    "n", "components", "lanes_per_batch", "trials", "alternate_mode",
    "concurrent_device_median_ms", "alternate_device_median_ms",
    "separate_to_alternate_device_speedup",
    "eigenvalue_bit_mismatch_count", "eigenvector_bit_mismatch_count",
    "serial_failure_count", "concurrent_failure_count"
  )
  if (length(setdiff(required, names(parsed))) ||
      parsed$n != 351 || parsed$components != shape$components ||
      parsed$lanes_per_batch != shape$lanes ||
      parsed$trials != shape$trials ||
      !identical(parsed$alternate_mode, "threaded") ||
      parsed$eigenvalue_bit_mismatch_count != 0 ||
      parsed$eigenvector_bit_mismatch_count != 0 ||
      parsed$serial_failure_count != 0 ||
      parsed$concurrent_failure_count != 0) {
    fail("lane benchmark identity or parity gate failed")
  }
  lane_rows[[index]] <- parsed
  lane_raw[[index]] <- run$raw
  cat(sprintf(
    "PASS lane shape %d/%d: components=%d lanes=%d device=%.3fms threaded_ratio=%.4fx\n",
    index, nrow(lane_shapes), shape$components, shape$lanes,
    parsed$concurrent_device_median_ms,
    parsed$separate_to_alternate_device_speedup
  ))
}

all_lane_fields <- unique(unlist(lapply(lane_rows, names), use.names = FALSE))
lane_benchmark <- as.data.frame(do.call(rbind, lapply(lane_rows, function(row) {
  values <- rep(NA_character_, length(all_lane_fields))
  names(values) <- all_lane_fields
  values[names(row)] <- vapply(row, as.character, character(1L))
  values
})), stringsAsFactors = FALSE)
lane_numeric <- setdiff(names(lane_benchmark), "alternate_mode")
lane_benchmark[lane_numeric] <- lapply(lane_benchmark[lane_numeric], as.numeric)

lane_comparisons <- do.call(rbind, lapply(c(9L, 16L, 48L), function(count) {
  rows <- lane_benchmark[lane_benchmark$components == count, , drop = FALSE]
  rows <- rows[order(rows$lanes_per_batch), , drop = FALSE]
  if (nrow(rows) != 2L) fail("lane comparison shape is incomplete")
  data.frame(
    components = count,
    baseline_lanes = rows$lanes_per_batch[[1L]],
    expanded_lanes = rows$lanes_per_batch[[2L]],
    baseline_device_ms = rows$concurrent_device_median_ms[[1L]],
    expanded_device_ms = rows$concurrent_device_median_ms[[2L]],
    expanded_lane_speedup =
      rows$concurrent_device_median_ms[[1L]] /
        rows$concurrent_device_median_ms[[2L]]
  )
}))
rownames(lane_comparisons) <- NULL

solver_run <- run_benchmark(solver_binary, c(351L, 20L))
solver <- numeric_fields(solver_run$parsed)
solver_required <- c(
  "n", "repeats", "divide_info", "jacobi_info",
  "divide_per_solve_ms", "jacobi_per_solve_ms", "jacobi_speedup",
  "maximum_eigenvalue_error", "maximum_sign_aligned_vector_error"
)
if (length(setdiff(solver_required, names(solver))) ||
    solver$n != 351 || solver$repeats != 20 ||
    solver$divide_info != 0 || solver$jacobi_info != 0) {
  fail("solver benchmark identity or status gate failed")
}

baseline <- readRDS(baseline_path)
opportunity <- readRDS(opportunity_path)
if (!isTRUE(baseline$audit$pass) ||
    !identical(
      opportunity$schema_version,
      "phase10-legacy-component-wave-opportunity-v1"
    )) {
  fail("full-eig closure receipt inputs are malformed")
}
opportunity_metrics <- opportunity$metrics
required_opportunity <- c(
  "two_batch_benchmark_positive_pair_coverage",
  "two_batch_projected_eig_saving_sec",
  "wave_regression_sec", "two_batch_projected_net_saving_sec"
)
if (length(setdiff(required_opportunity, names(opportunity_metrics))) ||
    opportunity_metrics$two_batch_benchmark_positive_pair_coverage != 1) {
  fail("selective-wave opportunity coverage is incomplete")
}

parity_pass <- all(
  lane_benchmark$eigenvalue_bit_mismatch_count == 0 &
    lane_benchmark$eigenvector_bit_mismatch_count == 0 &
    lane_benchmark$serial_failure_count == 0 &
    lane_benchmark$concurrent_failure_count == 0
)
expanded_lane_max_speedup <- max(lane_comparisons$expanded_lane_speedup)
threaded_max_speedup <- max(
  lane_benchmark$separate_to_alternate_device_speedup[
    lane_benchmark$lanes_per_batch == 8L
  ]
)
closure_pass <- parity_pass &&
  expanded_lane_max_speedup < 1.01 &&
  threaded_max_speedup < 1.01 &&
  solver$jacobi_speedup < 1.0 &&
  opportunity_metrics$two_batch_projected_net_saving_sec < 10.0

receipt <- list(
  schema_version = "phase10-full-eig-component-closure-v1",
  generated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  tool = list(
    path = normalizePath(
      "fastkpc/tools/finalize_phase10_full_eig_component_closure.R",
      winslash = "/", mustWork = TRUE
    ),
    sha256 = unname(digest::digest(
      file = "fastkpc/tools/finalize_phase10_full_eig_component_closure.R",
      algo = "sha256", serialize = FALSE
    ))
  ),
  baseline = list(
    path = normalizePath(baseline_path, winslash = "/", mustWork = TRUE),
    sha256 = unname(digest::digest(
      file = baseline_path, algo = "sha256", serialize = FALSE
    )),
    native_sha256 = baseline$native_sha256,
    elapsed_sec = baseline$audit$elapsed_sec,
    audit_pass = baseline$audit$pass
  ),
  selective_opportunity = list(
    path = normalizePath(opportunity_path, winslash = "/", mustWork = TRUE),
    sha256 = unname(digest::digest(
      file = opportunity_path, algo = "sha256", serialize = FALSE
    )),
    metrics = opportunity_metrics
  ),
  lane_binary = list(
    path = normalizePath(lane_binary, winslash = "/", mustWork = TRUE),
    sha256 = unname(digest::digest(
      file = lane_binary, algo = "sha256", serialize = FALSE
    )),
    source_path = normalizePath(
      "fastkpc/tools/phase10_cuda_full_eig_lane_benchmark.cu",
      winslash = "/", mustWork = TRUE
    ),
    source_sha256 = unname(digest::digest(
      file = "fastkpc/tools/phase10_cuda_full_eig_lane_benchmark.cu",
      algo = "sha256", serialize = FALSE
    ))
  ),
  solver_binary = list(
    path = normalizePath(solver_binary, winslash = "/", mustWork = TRUE),
    sha256 = unname(digest::digest(
      file = solver_binary, algo = "sha256", serialize = FALSE
    )),
    source_path = normalizePath(
      "fastkpc/tools/phase10_cuda_full_eig_solver_benchmark.cu",
      winslash = "/", mustWork = TRUE
    ),
    source_sha256 = unname(digest::digest(
      file = "fastkpc/tools/phase10_cuda_full_eig_solver_benchmark.cu",
      algo = "sha256", serialize = FALSE
    ))
  ),
  gpu_identity = system2(
    "nvidia-smi",
    args = c(
      "--query-gpu=index,uuid,name,driver_version,pci.bus_id",
      "--format=csv,noheader"
    ),
    stdout = TRUE,
    stderr = TRUE
  ),
  lane_benchmark = lane_benchmark,
  lane_comparisons = lane_comparisons,
  lane_raw_output = lane_raw,
  solver_benchmark = solver,
  solver_raw_output = solver_run$raw,
  summary = list(
    lane_and_thread_parity_pass = parity_pass,
    expanded_lane_max_speedup = expanded_lane_max_speedup,
    threaded_max_speedup = threaded_max_speedup,
    jacobi_speedup = solver$jacobi_speedup,
    jacobi_maximum_eigenvalue_error = solver$maximum_eigenvalue_error,
    jacobi_maximum_sign_aligned_vector_error =
      solver$maximum_sign_aligned_vector_error,
    selective_wave_projected_net_saving_sec =
      opportunity_metrics$two_batch_projected_net_saving_sec,
    decision = "STOP_CURRENT_STRICT_SINGLE_GPU_FULL_EIG_COMPONENT",
    pass = closure_pass
  )
)

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
temporary_path <- paste0(output_path, ".tmp")
on.exit(unlink(temporary_path, force = TRUE), add = TRUE)
saveRDS(receipt, temporary_path, compress = "xz", version = 3L)
if (!file.rename(temporary_path, output_path)) {
  fail("failed to publish full-eig closure receipt")
}
cat(sprintf(
  paste0(
    "%s full-eig component closure; expanded_lane_max=%.4fx; ",
    "threaded_max=%.4fx; jacobi=%.4fx; selective_net=%.3fs; output=%s\n"
  ),
  if (closure_pass) "PASS" else "FAIL",
  expanded_lane_max_speedup, threaded_max_speedup,
  solver$jacobi_speedup,
  opportunity_metrics$two_batch_projected_net_saving_sec,
  output_path
))
if (!closure_pass) fail("full-eig component closure gate failed")
