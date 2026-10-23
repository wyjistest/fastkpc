# Offline audit, also callable on the .pending snapshot if a later check fails.
qualify_pinned_optimizer_receipt <- function(receipt, baseline_path = NULL) {
  result <- receipt$result
  resources_before <- receipt$resources_before
  resources_after <- receipt$resources_after
  seed_before <- receipt$seed_before
  seed_after <- receipt$seed_after
  stopifnot(is.data.frame(result$tasks), nrow(result$tasks) > 0L,
            is.matrix(result$pMax), is.matrix(result$adjacency),
            is.list(result$sepsets), length(result$n.edgetests) > 0L)
  summary <- result$summary
  live_fields <- c("live_device_allocations", "live_device_bytes", "live_streams", "live_events")
  authority_fields <- c(
    "r_callback_count", "legacy_mgcv_fit_count", "legacy_mgcv_setup_count",
    "cpu_residual_solve_count", "cpu_dcov_component_count", "cpu_dcov_eigen_or_lowrank_count",
    "cpu_dcov_pair_stat_count", "cpu_gamma_pvalue_count", "cpu_spectra_count",
    "residual_d2h_bytes", "component_d2h_bytes", "unknown_fallback_count",
    "approximate_backend_count"
  )
  gates <- list(
    authority_zero = all(vapply(authority_fields, function(nm) {
      !is.null(summary[[nm]]) && identical(as.numeric(summary[[nm]]), 0)
    }, logical(1L))),
    resource_closed = identical(resources_before[live_fields], resources_after[live_fields]),
    rng_unchanged = identical(seed_before, seed_after),
    all_p_finite = all(is.finite(result$tasks$p_used))
  )
  oracle_errors <- NULL
  if (receipt$workload == "canonical") {
    oracle_dir <- "fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2"
    oracle <- readRDS(file.path(oracle_dir, "logical_ci_trace.rds"))
    reference_pmax <- readRDS(file.path(oracle_dir, "pmax.rds"))
    gates$oracle_task_identity_exact <- nrow(result$tasks) == nrow(oracle) &&
      identical(as.integer(result$tasks$canonical_test_order_id), as.integer(oracle$logical_sequence_id)) &&
      identical(as.integer(result$tasks$level), as.integer(oracle$level)) &&
      identical(as.integer(result$tasks$task_index), as.integer(oracle$source_task_index)) &&
      identical(as.integer(result$tasks$x), as.integer(oracle$x)) &&
      identical(as.integer(result$tasks$y), as.integer(oracle$y)) &&
      identical(as.character(result$tasks$S_key), as.character(oracle$S_key)) &&
      identical(as.logical(result$tasks$native_edge_deleted), as.logical(oracle$deletes_edge))
    finite <- is.finite(reference_pmax)
    oracle_errors <- c(
      ci = if (gates$oracle_task_identity_exact) max(abs(result$tasks$p_used - oracle$p_value)) else Inf,
      pMax = max(abs(result$pMax[finite] - reference_pmax[finite]))
    )
    gates$oracle_ci_contract <- isTRUE(oracle_errors[["ci"]] <= 1e-10)
    gates$oracle_pmax_contract <- identical(is.na(result$pMax), is.na(reference_pmax)) &&
      identical(is.infinite(result$pMax), is.infinite(reference_pmax)) &&
      isTRUE(oracle_errors[["pMax"]] <= 1e-10)
    gates$oracle_adjacency_exact <- identical(unname(result$adjacency),
      unname(readRDS(file.path(oracle_dir, "adjacency.rds"))))
    gates$oracle_n_edgetests_exact <- identical(as.integer(result$n.edgetests),
      c(2213L,52659L,125293L,40694L,13293L,5422L,835L,80L,9L))
    gates$exact_fixed_route <- identical(summary$strict_gamma_residual_route, "exact-fixed-stable-svd")
    gates$authority_gate_pass <- isTRUE(summary$authority_gate_pass)
  }
  if (!is.null(baseline_path)) {
    baseline <- readRDS(baseline_path)
    stopifnot(identical(baseline$workload, receipt$workload))
    task_fields <- c("level", "task_index", "x", "y", "S_key", "canonical_test_order_id",
                     "native_edge_deleted", "native_edge_ignored")
    gates$task_identity_exact <- identical(result$tasks[task_fields], baseline$result$tasks[task_fields])
    gates$p_value_bitwise_exact <- identical(result$tasks$p_used, baseline$result$tasks$p_used,
                                             num.eq = FALSE, single.NA = FALSE)
    gates$pmax_bitwise_exact <- identical(result$pMax, baseline$result$pMax,
                                          num.eq = FALSE, single.NA = FALSE)
    gates$adjacency_exact <- identical(result$adjacency, baseline$result$adjacency)
    gates$sepsets_exact <- identical(result$sepsets, baseline$result$sepsets)
    gates$n_edgetests_exact <- identical(result$n.edgetests, baseline$result$n.edgetests)
    gates$rng_endpoint_exact <- identical(seed_after, baseline$seed_after)
  }
  
  receipt$gates <- gates
  receipt$oracle_errors <- oracle_errors
  receipt
}
