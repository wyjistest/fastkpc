source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!isTRUE(tryCatch(
  fastkpc_cuda_available(),
  error = function(error) FALSE
))) {
  cat("SKIP Phase 9 one-call compatible CUDA skeleton: CUDA unavailable\n")
  quit(save = "no", status = 0L)
}

assert_true(
  exists("precision_run_skeleton_full_cuda_native", mode = "function"),
  "Phase 9 native one-call wrapper is missing"
)

set.seed(5901)
n <- 72L
z <- stats::runif(n, -2, 2)
data <- cbind(
  x1 = sin(z) + stats::rnorm(n, sd = 0.08),
  x2 = cos(z) + stats::rnorm(n, sd = 0.08),
  x3 = z + stats::rnorm(n, sd = 0.08),
  x4 = z^2 + stats::rnorm(n, sd = 0.08)
)

reference <- precision_run_skeleton_legacy_mgcv_legacy_dcov_native(
  data = data,
  alpha = 0.1,
  max_conditioning_size = 1L,
  index = 1,
  numCol = 35L,
  trace_level = "logical",
  dcov_batch = "round"
)
candidate <- precision_run_skeleton_full_cuda_native(
  data = data,
  alpha = 0.1,
  max_conditioning_size = 1L,
  index = 1,
  numCol = 35L,
  trace_level = "logical",
  compatible_cuda_strict = TRUE
)
facade <- fastkpc_compatible_cuda_skeleton(
  data = data,
  alpha = 0.1,
  labels = colnames(data),
  options = list(
    route = "full_cuda",
    compatible_cuda_strict = TRUE,
    max_conditioning_size = 1L,
    index = 1,
    numCol = 35L,
    trace_level = "logical"
  )
)

normalize_sepsets <- function(value) {
  unname(lapply(unname(value), function(row) {
    unname(lapply(unname(row), function(entry) sort(as.integer(entry))))
  }))
}

canonicalize_tasks <- function(value) {
  value[order(value$level, value$task_index), , drop = FALSE]
}

assert_canonical_task_order <- function(candidate_tasks, reference_tasks,
                                        label) {
  expected <- canonicalize_tasks(reference_tasks)
  structural_fields <- c(
    "level", "task_index", "edge_x", "edge_y", "x", "y", "S_key",
    "conditioning_size"
  )
  assert_true(
    identical(seq_len(nrow(candidate_tasks)),
              order(candidate_tasks$level, candidate_tasks$task_index)) &&
      all(vapply(structural_fields, function(field) {
        identical(candidate_tasks[[field]], expected[[field]])
      }, logical(1L))) &&
      identical(candidate_tasks$native_edge_deleted,
                expected$native_edge_deleted) &&
      !any((candidate_tasks$p_used >= 0.1) !=
             (expected$p_used >= 0.1)),
    paste0("Phase 9 ", label,
           " trace does not follow canonical layer-plan replay")
  )
}

assert_true(
  identical(candidate$adjacency, reference$adjacency),
  "Phase 9 focused adjacency differs from the legacy oracle"
)
assert_true(
  identical(unname(facade$adjacency), candidate$adjacency) &&
    identical(normalize_sepsets(facade$sepsets),
              normalize_sepsets(candidate$sepsets)) &&
    identical(as.integer(facade$n.edgetests),
              as.integer(candidate$n.edgetests)) &&
    isTRUE(facade$summary$compatible_cuda_facade) &&
    identical(facade$summary$compatible_cuda_route, "compatible.cuda"),
  "Phase 9 explicit compatible CUDA facade does not use the one-call route"
)
assert_true(
  identical(normalize_sepsets(candidate$sepsets),
            normalize_sepsets(reference$sepsets)),
  "Phase 9 focused sepsets differ from the legacy oracle"
)
assert_true(
  identical(as.integer(candidate$n.edgetests),
            as.integer(reference$n.edgetests)),
  "Phase 9 focused logical n.edgetests differ from the legacy oracle"
)

summary <- candidate$summary
required_zero <- c(
  "r_callback_count",
  "legacy_mgcv_fit_count",
  "legacy_mgcv_setup_count",
  "cpu_residual_solve_count",
  "cpu_dcov_component_count",
  "cpu_dcov_eigen_or_lowrank_count",
  "cpu_dcov_pair_stat_count",
  "cpu_gamma_pvalue_count",
  "cpu_spectra_count",
  "residual_d2h_bytes",
  "unknown_fallback_count",
  "approximate_backend_count"
)
assert_true(
  all(required_zero %in% names(summary)) &&
    all(vapply(required_zero, function(name) {
      identical(as.numeric(summary[[name]]), 0)
    }, logical(1L))),
  "Phase 9 focused authority counters are not all zero"
)
assert_true(
  identical(summary$entrypoint,
            "compatible-cuda-full-skeleton-native-v1") &&
    identical(summary$compatible_cuda_route, "compatible.cuda") &&
    isTRUE(summary$compatible_cuda_strict) &&
    identical(as.integer(summary$native_call_count), 1L) &&
    summary$cuda_single_penalty_target_count > 0L &&
    summary$cuda_residual_batch_count > 0L &&
    summary$cuda_dcov_component_count > 0L &&
    summary$cuda_dcov_pair_count == summary$physical_tests_evaluated &&
    summary$cuda_gamma_pvalue_count == summary$physical_tests_evaluated &&
    summary$logical_tests_consumed == sum(candidate$n.edgetests) &&
    summary$speculative_tests_ignored >= 0L,
  "Phase 9 focused native diagnostics are incomplete"
)
assert_true(
  identical(candidate$tasks$canonical_test_order_id,
            seq_len(nrow(candidate$tasks))) &&
    all(is.finite(candidate$tasks$p_used)),
  "Phase 9 focused logical replay trace is malformed"
)
assert_canonical_task_order(candidate$tasks, reference$tasks, "focused")

set.seed(6001)
multi_n <- 90L
multi_p <- 8L
multi_data <- matrix(stats::rnorm(multi_n * multi_p), multi_n, multi_p)
for (column in 2:multi_p) {
  parents <- which(stats::runif(column - 1L) < 0.65)
  if (length(parents) > 0L) {
    multi_data[, column] <- multi_data[, column] +
      multi_data[, parents, drop = FALSE] %*%
        stats::runif(length(parents), 0.35, 0.9)
  }
}
colnames(multi_data) <- paste0("v", seq_len(multi_p))
multi_reference <- precision_run_skeleton_legacy_mgcv_legacy_dcov_native(
  data = multi_data,
  alpha = 0.1,
  max_conditioning_size = 3L,
  index = 1,
  numCol = 35L,
  trace_level = "logical",
  dcov_batch = "round"
)
multi_candidate <- precision_run_skeleton_full_cuda_native(
  data = multi_data,
  alpha = 0.1,
  max_conditioning_size = 3L,
  index = 1,
  numCol = 35L,
  trace_level = "logical",
  compatible_cuda_strict = TRUE
)
assert_true(
  length(multi_candidate$n.edgetests) == 4L &&
    multi_candidate$summary$cuda_multi_penalty_target_count > 0L &&
    identical(multi_candidate$adjacency, multi_reference$adjacency) &&
    identical(normalize_sepsets(multi_candidate$sepsets),
              normalize_sepsets(multi_reference$sepsets)) &&
    identical(as.integer(multi_candidate$n.edgetests),
              as.integer(multi_reference$n.edgetests)),
  "Phase 9 focused multi-penalty live CUDA replay differs from legacy"
)
assert_canonical_task_order(
  multi_candidate$tasks, multi_reference$tasks, "multi-penalty"
)

invalid_alpha <- tryCatch(
  precision_run_skeleton_full_cuda_native(
    data = data,
    alpha = 0.05,
    max_conditioning_size = 0L,
    index = 1,
    numCol = 35L,
    compatible_cuda_strict = TRUE
  ),
  error = function(error) error
)
assert_true(
  inherits(invalid_alpha, "error") &&
    grepl("alpha", conditionMessage(invalid_alpha), fixed = TRUE),
  "Phase 9 strict route must fail closed outside the qualified alpha"
)

invalid_num_col <- tryCatch(
  precision_run_skeleton_full_cuda_native(
    data = data,
    alpha = 0.1,
    max_conditioning_size = 0L,
    index = 1,
    numCol = 7L,
    compatible_cuda_strict = TRUE
  ),
  error = function(error) error
)
assert_true(
  inherits(invalid_num_col, "error") &&
    grepl("numCol", conditionMessage(invalid_num_col), fixed = TRUE),
  "Phase 9 strict route must fail closed outside qualified numCol=35"
)

cat(
  "PASS Phase 9 focused one-call compatible CUDA skeleton; logical=",
  summary$logical_tests_consumed,
  " physical=", summary$physical_tests_evaluated,
  "\n", sep = ""
)
