source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(
      Sys.getenv("FASTKPC_RUN_FULL_DEFAULT_INF_CUDA_TEST", unset = "0"),
      "1")) {
  cat("SKIP full production default-Inf CUDA regression\n")
  quit(save = "no", status = 0L)
}
if (!requireNamespace("digest", quietly = TRUE) ||
    !requireNamespace("jsonlite", quietly = TRUE) ||
    !isTRUE(tryCatch(fastkpc_cuda_available(), error = function(error) FALSE))) {
  cat("SKIP full production default-Inf CUDA regression: dependency unavailable\n")
  quit(save = "no", status = 0L)
}
required_environment <- c(
  CUDA_VISIBLE_DEVICES = "0", OPENBLAS_NUM_THREADS = "1",
  OMP_NUM_THREADS = "1", MKL_NUM_THREADS = "1", BLIS_NUM_THREADS = "1",
  VECLIB_MAXIMUM_THREADS = "1"
)
assert_true(
  identical(
    Sys.getenv(names(required_environment), unset = ""),
    required_environment
  ),
  "full production default-Inf CUDA regression requires device 0 and single-thread BLAS"
)

fixture <- jsonlite::read_json(
  "fastkpc/tests/fixtures/default_inf_production_cuda_v1.json",
  simplifyVector = TRUE
)
data <- as.matrix(readRDS(file.path(
  "fastkpc", "artifacts", "kpc_tprs_real_zhu",
  "cancer_RD-causalDiscoveryInput.rds"
)))
storage.mode(data) <- "double"

raw_double_hash <- function(value) {
  payload <- writeBin(
    as.double(value), raw(), size = 8L, endian = "little"
  )
  digest::digest(payload, algo = "sha256", serialize = FALSE)
}
raw_integer_hash <- function(value) {
  payload <- writeBin(
    as.integer(value), raw(), size = 4L, endian = "little"
  )
  digest::digest(payload, algo = "sha256", serialize = FALSE)
}
text_hash <- function(value) {
  digest::digest(
    charToRaw(paste(value, collapse = "\n")),
    algo = "sha256", serialize = FALSE
  )
}
double_bits <- function(value) {
  vapply(value, function(element) {
    payload <- writeBin(
      as.double(element), raw(), size = 8L, endian = "little"
    )
    paste(sprintf("%02x", as.integer(payload)), collapse = "")
  }, character(1L))
}

assert_true(
  identical(dim(data), c(fixture$n, fixture$p)) &&
    identical(
      digest::digest(data, algo = "sha256", serialize = TRUE),
      fixture$data_sha256
    ),
  "default-Inf production input identity changed"
)

invisible(full_cuda_ci_one_call_cache_control_native("configure", 262144L))
invisible(full_cuda_ci_one_call_cache_control_native(
  "configure_target", 131072L
))
invisible(full_cuda_ci_one_call_cache_control_native("reset"))

# Omit max_conditioning_size deliberately: this tests the public Inf default.
result <- fastkpc_compatible_cuda_skeleton(
  data,
  alpha = fixture$alpha,
  labels = colnames(data),
  options = list(
    route = "full_cuda",
    compatible_cuda_strict = TRUE,
    index = 1,
    numCol = 35L,
    trace_level = "logical"
  )
)

summary <- result$summary
assert_true(
  identical(summary$max_conditioning_size_requested,
            fixture$requested_max_conditioning_size) &&
    identical(as.integer(summary$max_conditioning_size_resolved),
              fixture$resolved_max_conditioning_size) &&
    identical(as.integer(result$levels$level),
              seq.int(0L, fixture$natural_stop_level)) &&
    identical(as.integer(result$n.edgetests), fixture$n_edgetests) &&
    identical(nrow(result$tasks), fixture$logical_test_count) &&
    identical(as.integer(summary$logical_tests_consumed),
              fixture$logical_test_count) &&
    identical(as.integer(summary$physical_tests_evaluated),
              fixture$physical_test_count) &&
    identical(as.integer(summary$physical_target_optimization_count),
              fixture$physical_target_optimization_count) &&
    identical(as.integer(summary$unique_target_key_count),
              fixture$unique_target_key_count) &&
    identical(as.integer(summary$unique_residual_key_count),
              fixture$unique_residual_key_count) &&
    identical(as.integer(summary$native_setup_count),
              fixture$native_setup_count) &&
    identical(as.integer(summary$cuda_optimizer_host_boundary_count),
              fixture$optimizer_boundary_count),
  "default-Inf production counts or natural-stop semantics changed"
)

task_structure_fields <- c(
  "canonical_test_order_id", "level", "task_index", "edge_x", "edge_y",
  "x", "y", "S_key", "conditioning_size", "native_edge_deleted",
  "native_edge_ignored"
)
task_rows <- do.call(
  paste,
  c(unname(result$tasks[task_structure_fields]), list(sep = "\t"))
)
sepset_rows <- unlist(lapply(seq_along(result$sepsets), function(i) {
  lapply(seq_along(result$sepsets[[i]]), function(j) {
    paste(
      i, j, paste(as.integer(result$sepsets[[i]][[j]]), collapse = "|"),
      sep = "\t"
    )
  })
}), recursive = TRUE, use.names = FALSE)

actual_hashes <- c(
  adjacency_int32_le = raw_integer_hash(result$adjacency),
  sepsets_canonical_text = text_hash(sepset_rows),
  pmax_binary64_le = raw_double_hash(result$pMax),
  task_structure_canonical_text = text_hash(task_rows),
  task_p_used_binary64_le = raw_double_hash(result$tasks$p_used)
)
expected_hashes <- unlist(fixture$hashes, use.names = TRUE)
assert_true(
  identical(actual_hashes, expected_hashes),
  "default-Inf production graph, trace, or p-value bits changed"
)

level8 <- result$tasks[result$tasks$level == fixture$natural_stop_level,
                       , drop = FALSE]
assert_true(
  nrow(level8) == 9L &&
    identical(double_bits(level8$p_used),
              fixture$level8_production_p_value_bits_le) &&
    !any(level8$native_edge_deleted) &&
    !any(level8$native_edge_ignored),
  "default-Inf production level-8 payload changed"
)

authority_zero_fields <- c(
  "r_callback_count", "legacy_mgcv_fit_count", "legacy_mgcv_setup_count",
  "cpu_residual_solve_count", "cpu_dcov_component_count",
  "cpu_dcov_eigen_or_lowrank_count", "cpu_dcov_pair_stat_count",
  "cpu_gamma_pvalue_count", "cpu_spectra_count", "residual_d2h_bytes",
  "component_d2h_bytes", "unknown_fallback_count",
  "approximate_backend_count"
)
assert_true(
  isTRUE(summary$authority_gate_pass) &&
    all(vapply(authority_zero_fields, function(field) {
      identical(as.numeric(summary[[field]]), 0)
    }, logical(1L))),
  "default-Inf production CUDA authority gate failed"
)

cat(
  "PASS full production default-Inf CUDA regression; levels=0:8; tasks=",
  nrow(result$tasks), "; p_values_bitwise=TRUE\n", sep = ""
)
