.fastkpc_full_cuda_phase35_feasibility_require <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
  invisible(TRUE)
}

.fastkpc_full_cuda_phase35_feasibility_hash_file <- function(path, expected,
                                                              label) {
  .fastkpc_full_cuda_phase35_feasibility_require(
    is.character(path) && length(path) == 1L && !is.na(path) &&
      file.exists(path) && !dir.exists(path),
    paste0(label, " is missing")
  )
  actual <- fastkpc_full_cuda_census_file_hash(path)
  .fastkpc_full_cuda_phase35_feasibility_require(
    identical(actual, expected), paste0(label, " SHA-256 mismatch")
  )
  actual
}

fastkpc_full_cuda_phase35_load_evidence_bundle <- function(
    evidence_manifest_path, require_current_execution_source = TRUE) {
  evidence <- readRDS(evidence_manifest_path)
  required <- c(
    "schema_version", "execution_source_file_sha256",
    "execution_source_closure_sha256", "native_binary_sha256",
    "two_scale_rds", "two_scale_rds_sha256", "full_exact_rds",
    "full_exact_rds_sha256", "full_hybrid_rds",
    "full_hybrid_rds_sha256"
  )
  .fastkpc_full_cuda_phase35_feasibility_require(
    is.list(evidence) && identical(names(evidence), required) &&
      identical(evidence$schema_version,
                "full-cuda-ci-phase35-prepublication-evidence-v1"),
    "Phase 3.5 evidence manifest schema mismatch"
  )
  source_hashes <- evidence$execution_source_file_sha256
  .fastkpc_full_cuda_phase35_feasibility_require(
    is.list(source_hashes) && length(source_hashes) >= 1L &&
      !is.null(names(source_hashes)) && !anyNA(names(source_hashes)) &&
      !anyDuplicated(names(source_hashes)) &&
      all(vapply(source_hashes, function(value) {
        is.character(value) && length(value) == 1L && !is.na(value) &&
          grepl("^[0-9a-f]{64}$", value)
      }, logical(1L))),
    "Phase 3.5 evidence execution source closure is malformed"
  )
  closure_sha256 <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(source_hashes)
  )
  .fastkpc_full_cuda_phase35_feasibility_require(
    identical(closure_sha256, evidence$execution_source_closure_sha256),
    "Phase 3.5 evidence execution source closure hash mismatch"
  )
  if (isTRUE(require_current_execution_source)) {
    for (path in names(source_hashes)) {
      .fastkpc_full_cuda_phase35_feasibility_hash_file(
        path, source_hashes[[path]], paste0("execution source ", path)
      )
    }
    .fastkpc_full_cuda_phase35_feasibility_hash_file(
      file.path("fastkpc", "build", "fastkpc_cuda.so"),
      evidence$native_binary_sha256, "executed native binary"
    )
  }
  .fastkpc_full_cuda_phase35_feasibility_hash_file(
    evidence$two_scale_rds, evidence$two_scale_rds_sha256,
    "two-scale evidence"
  )
  .fastkpc_full_cuda_phase35_feasibility_hash_file(
    evidence$full_exact_rds, evidence$full_exact_rds_sha256,
    "full exact-screen evidence"
  )
  .fastkpc_full_cuda_phase35_feasibility_hash_file(
    evidence$full_hybrid_rds, evidence$full_hybrid_rds_sha256,
    "full guarded-hybrid evidence"
  )

  two_scale <- readRDS(evidence$two_scale_rds)
  full_exact <- readRDS(evidence$full_exact_rds)
  full_hybrid <- readRDS(evidence$full_hybrid_rds)
  scale_gate <- function(value, scale_id, pair_count, screen_flips,
                         refined_pairs, refined_components,
                         refinement_groups) {
    is.list(value) &&
      identical(value$schema_version,
                "full-cuda-ci-phase35-guarded-hybrid-scale-result-v1") &&
      identical(value$scale_id, scale_id) && all(value$gates) &&
      nrow(value$pairs) == pair_count &&
      sum(value$pairs$screen_decision_flip) == screen_flips &&
      !any(value$pairs$decision_flip) &&
      sum(value$pairs$refined) == refined_pairs &&
      sum(value$refinement_groups$component_count) == refined_components &&
      nrow(value$refinement_groups) == refinement_groups &&
      max(abs(value$refinement_pairs$
                refined_p_value_difference_from_legacy)) <= 1e-10
  }
  evidence_valid <-
    is.list(two_scale) &&
    identical(two_scale$schema_version,
              "full-cuda-ci-phase35-two-scale-guarded-hybrid-run-v1") &&
    scale_gate(
      two_scale$hybrid_a, "A_qualification_complete", 3808L, 92L,
      1142L, 1532L, 385L
    ) &&
    scale_gate(
      two_scale$hybrid_b, "B_level2_reuse_quartiles_192", 21380L, 9L,
      78L, 124L, 29L
    ) &&
    nrow(two_scale$scale_b$group_table) == 192L &&
    all(table(two_scale$scale_b$group_table$reuse_quartile) == 48L) &&
    is.list(full_exact) &&
    identical(full_exact$schema_version,
              "full-cuda-ci-phase35-exact-scale-result-v1") &&
    identical(full_exact$scale_id,
              "FULL_CONDITIONAL_LEVELS_1_TO_7") &&
    nrow(full_exact$pairs) == 238276L &&
    nrow(full_exact$groups) == 8634L &&
    sum(full_exact$groups$component_count) == 110617L &&
    sum(full_exact$pairs$decision_flip) == 92L &&
    is.list(full_hybrid) &&
    identical(full_hybrid$scale_id,
              "FULL_CONDITIONAL_LEVELS_1_TO_7") &&
    all(full_hybrid$gates) &&
    nrow(full_hybrid$pairs) == 238276L &&
    sum(full_hybrid$pairs$screen_decision_flip) == 92L &&
    !any(full_hybrid$pairs$decision_flip) &&
    sum(full_hybrid$pairs$refined) == 1142L &&
    nrow(full_hybrid$refinement_groups) == 385L &&
    sum(full_hybrid$refinement_groups$component_count) == 1532L &&
    max(abs(full_hybrid$refinement_pairs$
              refined_p_value_difference_from_legacy)) <= 1e-10
  .fastkpc_full_cuda_phase35_feasibility_require(
    evidence_valid, "Phase 3.5 evidence numerical or structural gate failed"
  )
  list(
    manifest = evidence,
    two_scale = two_scale,
    full_exact = full_exact,
    full_hybrid = full_hybrid
  )
}

.fastkpc_full_cuda_phase35_cache_simulate <- function(groups, pairs,
                                                       capacity) {
  hits <- misses <- evictions <- 0L
  for (indices in groups) {
    indices <- indices[order(pairs$logical_sequence_id[indices],
                             method = "radix")]
    references <- as.vector(t(cbind(
      pairs$residual_key_x[indices], pairs$residual_key_y[indices]
    )))
    cache <- character()
    last_use <- integer()
    tick <- 0L
    for (key in references) {
      tick <- tick + 1L
      location <- match(key, cache)
      if (!is.na(location)) {
        hits <- hits + 1L
        last_use[[location]] <- tick
      } else {
        misses <- misses + 1L
        if (length(cache) >= capacity) {
          victim <- which.min(last_use)
          cache <- cache[-victim]
          last_use <- last_use[-victim]
          evictions <- evictions + 1L
        }
        cache <- c(cache, key)
        last_use <- c(last_use, tick)
      }
    }
  }
  data.frame(
    component_capacity = as.integer(capacity),
    component_lookups = as.integer(2L * nrow(pairs)),
    component_hits = as.integer(hits),
    component_misses = as.integer(misses),
    component_evictions = as.integer(evictions),
    hit_ratio = hits / (hits + misses),
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_phase35_build_cache_memory_model <- function(
    evidence_bundle,
    setup_results_path = file.path(
      "fastkpc", "artifacts", "full_cuda_ci",
      "fixed_sp_cuda_oracle_sp_v1", "setup_results.rds"
    ),
    gpu_memory_bytes = 25757220864) {
  full <- evidence_bundle$full_hybrid
  pairs <- full$pairs
  groups <- split(
    seq_len(nrow(pairs)),
    factor(
      pairs$prepared_s_key_sha256,
      levels = sort(unique(pairs$prepared_s_key_sha256), method = "radix")
    )
  )
  capacity_points <- c(2L, 8L, 16L, 47L)
  cache_curve <- do.call(rbind, lapply(
    capacity_points,
    function(capacity) .fastkpc_full_cuda_phase35_cache_simulate(
      groups, pairs, capacity
    )
  ))
  rownames(cache_curve) <- NULL

  reuse_distances <- integer()
  future_distances <- integer()
  for (indices in groups) {
    indices <- indices[order(pairs$logical_sequence_id[indices],
                             method = "radix")]
    references <- as.vector(t(cbind(
      pairs$residual_key_x[indices], pairs$residual_key_y[indices]
    )))
    positions <- split(seq_along(references), references)
    group_distances <- unlist(lapply(
      positions,
      function(value) if (length(value) > 1L) diff(value) else integer()
    ), use.names = FALSE)
    reuse_distances <- c(reuse_distances, group_distances)
    future_distances <- c(future_distances, group_distances)
  }
  probabilities <- c(0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1)
  reuse_summary <- data.frame(
    probability = probabilities,
    endpoint_reference_distance = as.numeric(stats::quantile(
      reuse_distances, probabilities, names = FALSE, type = 8
    )),
    future_endpoint_reference_distance = as.numeric(stats::quantile(
      future_distances, probabilities, names = FALSE, type = 8
    )),
    stringsAsFactors = FALSE
  )

  setup_results <- readRDS(setup_results_path)
  runtime <- evidence_bundle$two_scale$runtime_info
  exact_peak <- max(full$exact_groups$peak_live_device_bytes)
  refinement_peak <- max(full$refinement_groups$peak_live_device_bytes)
  exact_component_pool <- max(full$exact_groups$peak_component_bytes)
  refinement_component_pool <-
    max(full$refinement_groups$persistent_component_bytes)
  refinement_eig_workspace <-
    max(full$refinement_groups$eig_workspace_bytes)
  refinement_pair_workspace <-
    max(full$refinement_groups$pair_workspace_bytes)
  residual_cache <- 47 * 351 * 8
  runtime_workspace <- runtime$workspace_bytes
  cublas_workspace <- runtime$cublas_workspace_bytes
  prepared_setup <- max(setup_results$setup_h2d_bytes)
  measured_peak <- runtime_workspace + cublas_workspace + prepared_setup +
    max(exact_peak, refinement_peak)
  concurrent_allocation_count <- runtime$cuda_device_allocation_count +
    max(full$refinement_groups$device_allocation_count) +
    max(setup_results$setup_h2d_upload_count)
  alignment_slack <- concurrent_allocation_count * 255
  allocator_metadata_reserve <- 1024^2
  stream_event_state_reserve <- 1024^2
  fragmentation_reserve <- ceiling(0.25 * measured_peak)
  opaque_context_library_reserve <- 512 * 1024^2
  declared_peak <- measured_peak + alignment_slack +
    allocator_metadata_reserve + stream_event_state_reserve +
    fragmentation_reserve +
    opaque_context_library_reserve
  memory <- data.frame(
    item = c(
      "fixed_sp_runtime_workspace", "fixed_sp_cublas_workspace",
      "largest_prepared_setup_upload", "exact_component_call_peak",
      "legacy_refinement_call_peak", "measured_concurrent_peak",
      "reference_residual_cache_47x351", "exact_component_pool_peak",
      "legacy_refinement_persistent_component_pool_peak",
      "legacy_refinement_eig_workspace_peak",
      "legacy_refinement_pair_workspace_peak",
      "alignment_slack", "allocator_metadata_reserve",
      "stream_event_state_reserve",
      "fragmentation_reserve_25pct", "opaque_context_library_reserve",
      "declared_capacity_bound", "reference_gpu_memory",
      "declared_headroom"
    ),
    bytes = c(
      runtime_workspace, cublas_workspace, prepared_setup, exact_peak,
      refinement_peak, measured_peak,
      residual_cache, exact_component_pool, refinement_component_pool,
      refinement_eig_workspace, refinement_pair_workspace,
      alignment_slack, allocator_metadata_reserve,
      stream_event_state_reserve, fragmentation_reserve,
      opaque_context_library_reserve, declared_peak, gpu_memory_bytes,
      gpu_memory_bytes - declared_peak
    ),
    measured = c(
      TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, TRUE, TRUE, TRUE, TRUE,
      FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE
    ),
    stringsAsFactors = FALSE
  )
  policy <- list(
    schema_version = "full-cuda-ci-phase35-cache-memory-model-v1",
    scheduling_unit = "complete-same-S-group-with-compact-logical-replay",
    replacement_policy = "deterministic-LRU-with-canonical-key-tie-break",
    reference_component_capacity = 47L,
    minimum_component_capacity = 2L,
    capacity_points = capacity_points,
    full_conditional_pair_count = nrow(pairs),
    full_conditional_group_count = length(groups),
    unique_component_count = 110617L,
    repeated_endpoint_reference_count = length(reuse_distances),
    residual_component_split =
      "runtime-residual-slot-is-bounded-by-47x351-binary64;component-pool-dominates",
    eviction_semantics =
      "reconstruct-from-authenticated-device-residual-with-no-result-change",
    oom_policy =
      "fail-closed-before-replay-and-never-publish-partial-graph",
    reference_capacity_zero_eviction =
      cache_curve$component_evictions[
        cache_curve$component_capacity == 47L
      ] == 0L,
    declared_peak_bytes = declared_peak,
    declared_headroom_bytes = gpu_memory_bytes - declared_peak,
    headroom_ratio = (gpu_memory_bytes - declared_peak) / gpu_memory_bytes,
    pass = declared_peak < 0.25 * gpu_memory_bytes &&
      cache_curve$component_misses[
        cache_curve$component_capacity == 47L
      ] == 110617L &&
      cache_curve$component_evictions[
        cache_curve$component_capacity == 47L
      ] == 0L
  )
  .fastkpc_full_cuda_phase35_feasibility_require(
    isTRUE(policy$pass), "Phase 3.5 cache and memory model gate failed"
  )
  level_values <- data.frame(
    level = pairs$level[match(
      full$exact_groups$prepared_s_key_sha256,
      pairs$prepared_s_key_sha256
    )],
    pair_count = full$exact_groups$pair_count,
    component_count = full$exact_groups$component_count,
    peak_live_device_bytes = full$exact_groups$peak_live_device_bytes,
    stringsAsFactors = FALSE
  )
  dense_counts <- stats::aggregate(
    cbind(pair_count, component_count) ~ level, level_values, sum
  )
  dense_peaks <- stats::aggregate(
    cbind(component_count, peak_live_device_bytes) ~ level,
    level_values, max
  )
  names(dense_peaks)[names(dense_peaks) == "component_count"] <-
    "maximum_group_component_count"
  dense_groups <- stats::aggregate(
    pair_count ~ level, level_values, length
  )
  names(dense_groups)[names(dense_groups) == "pair_count"] <- "group_count"
  dense_level_pressure <- Reduce(
    function(left, right) merge(left, right, by = "level", sort = TRUE),
    list(dense_counts, dense_peaks, dense_groups)
  )
  list(
    policy = policy,
    cache_curve = cache_curve,
    reuse_summary = reuse_summary,
    memory = memory,
    dense_level_pressure = dense_level_pressure
  )
}

.fastkpc_full_cuda_phase35_solver_class <- function(routes) {
  ifelse(grepl("SVD", routes, fixed = TRUE), "SVD",
         ifelse(grepl("QR", routes, fixed = TRUE), "QR", "CHOLESKY"))
}

.fastkpc_full_cuda_phase35_class_p95_bound <- function(
    groups, pairs, count_field, elapsed_field, breaks, class_kind) {
  routes <- tapply(
    pairs$solver_route, pairs$prepared_s_key_sha256,
    function(value) paste(unique(value), collapse = "|")
  )
  counts <- groups[[count_field]]
  elapsed <- groups[[elapsed_field]]
  classes <- if (identical(class_kind, "solver-and-size")) {
    paste(
      .fastkpc_full_cuda_phase35_solver_class(
        routes[groups$prepared_s_key_sha256]
      ),
      cut(counts, breaks = breaks), sep = "|"
    )
  } else {
    as.character(cut(counts, breaks = breaks))
  }
  values <- data.frame(
    class = classes,
    count = counts,
    elapsed_ms = elapsed,
    rate_ms = elapsed / counts,
    stringsAsFactors = FALSE
  )
  class_count <- stats::aggregate(count ~ class, values, sum)
  class_p95 <- stats::aggregate(
    rate_ms ~ class, values,
    function(value) as.numeric(stats::quantile(
      value, 0.95, names = FALSE, type = 8
    ))
  )
  class_groups <- stats::aggregate(rate_ms ~ class, values, length)
  names(class_groups)[[2L]] <- "group_count"
  result <- Reduce(
    function(left, right) merge(left, right, by = "class", sort = TRUE),
    list(class_count, class_p95, class_groups)
  )
  result$bound_ms <- result$count * result$rate_ms
  result
}

fastkpc_full_cuda_phase35_build_performance_model <- function(
    evidence_bundle, candidate_measurements, contracts) {
  full <- evidence_bundle$full_hybrid
  exact_component <- .fastkpc_full_cuda_phase35_class_p95_bound(
    full$exact_groups, full$pairs, "component_count",
    "component_build_cuda_ms", c(0, 2, 4, 8, 16, 32, Inf),
    "solver-and-size"
  )
  refine_component <- .fastkpc_full_cuda_phase35_class_p95_bound(
    full$refinement_groups, full$refinement_pairs, "component_count",
    "component_build_cuda_ms", c(0, 2, 4, 8, 16, 32, Inf),
    "solver-and-size"
  )
  exact_pair <- .fastkpc_full_cuda_phase35_class_p95_bound(
    full$exact_groups, full$pairs, "pair_count",
    "pair_evaluation_cuda_ms",
    c(0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, Inf), "size"
  )
  refine_pair <- .fastkpc_full_cuda_phase35_class_p95_bound(
    full$refinement_groups, full$refinement_pairs, "pair_count",
    "pair_evaluation_cuda_ms",
    c(0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, Inf), "size"
  )
  exact_overhead <- full$exact_groups$dcov_host_boundary_ms -
    full$exact_groups$component_build_cuda_ms -
    full$exact_groups$pair_evaluation_cuda_ms
  refine_overhead <- full$refinement_groups$dcov_host_boundary_ms -
    full$refinement_groups$component_build_cuda_ms -
    full$refinement_groups$pair_evaluation_cuda_ms

  component_class_bound <- sum(exact_component$bound_ms) +
    sum(refine_component$bound_ms)
  level_zero_component_bound <- 48 * max(c(
    candidate_measurements$global_full_eig_per_component_ms,
    max(full$refinement_groups$component_build_cuda_ms /
          full$refinement_groups$component_count)
  ))
  component_pre_contingency <- component_class_bound +
    level_zero_component_bound
  component_bound <- 1000 + 1.25 * component_pre_contingency

  pair_class_bound <- sum(exact_pair$bound_ms) + sum(refine_pair$bound_ms)
  synchronization_bound <-
    as.numeric(stats::quantile(exact_overhead, 0.95, type = 8)) *
      nrow(full$exact_groups) +
    as.numeric(stats::quantile(refine_overhead, 0.95, type = 8)) *
      nrow(full$refinement_groups)
  level_zero_pair_sync_bound <- 250
  pair_pre_contingency <- pair_class_bound + synchronization_bound +
    level_zero_pair_sync_bound
  pair_bound <- 1000 + 1.25 * pair_pre_contingency

  budget <- contracts$performance_budget_v1$payload
  allocations <- vapply(
    budget$component_budgets, `[[`, integer(1L), "warm_upper_bound_ms"
  )
  feasibility <- allocations
  feasibility[["dcov_component"]] <- ceiling(component_bound)
  feasibility[["dcov_pair_and_gamma"]] <- ceiling(pair_bound)
  budget_table <- data.frame(
    component = names(allocations),
    allocated_upper_bound_ms = as.integer(allocations),
    feasibility_used_or_reserved_ms = as.numeric(feasibility),
    implemented_in_phase35 = names(allocations) %in%
      c("dcov_component", "dcov_pair_and_gamma"),
    stringsAsFactors = FALSE
  )
  model <- list(
    schema_version = "full-cuda-ci-phase35-performance-model-v1",
    bound_method = budget$feasibility$bound_method,
    measured_full_conditional_component_cuda_ms =
      full$timing$total_component_cuda_ms[[1L]],
    measured_full_conditional_pair_gamma_cuda_ms =
      full$timing$total_pair_gamma_cuda_ms[[1L]],
    measured_full_conditional_dcov_boundary_ms =
      full$timing$total_dcov_host_boundary_ms[[1L]],
    component_class_p95_bound_ms = component_class_bound,
    pair_class_p95_bound_ms = pair_class_bound,
    synchronization_class_p95_bound_ms = synchronization_bound,
    level_zero_component_bound_ms = level_zero_component_bound,
    level_zero_pair_sync_bound_ms = level_zero_pair_sync_bound,
    fixed_overhead_per_budget_ms = 1000,
    throughput_and_variance_contingency_ratio = 0.25,
    dcov_component_upper_bound_ms = component_bound,
    dcov_pair_gamma_upper_bound_ms = pair_bound,
    dcov_total_upper_bound_ms = component_bound + pair_bound,
    dcov_allocated_upper_bound_ms =
      budget$feasibility$dcov_total_upper_bound_ms,
    full_campaign_used_or_reserved_ms = sum(feasibility),
    full_campaign_allocated_upper_bound_ms =
      budget$feasibility$total_upper_bound_ms,
    unimplemented_work_retains_full_allocation = TRUE,
    best_microbenchmark_global_extrapolation_used = FALSE,
    pass = component_bound <= allocations[["dcov_component"]] &&
      pair_bound <= allocations[["dcov_pair_and_gamma"]] &&
      component_bound + pair_bound <=
        budget$feasibility$dcov_total_upper_bound_ms &&
      sum(feasibility) <= budget$feasibility$total_upper_bound_ms
  )
  .fastkpc_full_cuda_phase35_feasibility_require(
    isTRUE(model$pass), "Phase 3.5 performance feasibility bound failed"
  )
  list(
    model = model,
    budget = budget_table,
    exact_component_classes = exact_component,
    refinement_component_classes = refine_component,
    exact_pair_classes = exact_pair,
    refinement_pair_classes = refine_pair
  )
}

.fastkpc_full_cuda_phase35_parse_measurement_line <- function(
    output, required_fields, label) {
  .fastkpc_full_cuda_phase35_feasibility_require(
    is.character(output) && length(output) >= 1L,
    paste0(label, " output is missing")
  )
  lines <- trimws(output[nzchar(trimws(output))])
  candidates <- lines[grepl("(^|[[:space:]])n=[0-9]+", lines)]
  .fastkpc_full_cuda_phase35_feasibility_require(
    length(candidates) == 1L,
    paste0(label, " must emit exactly one measurement line")
  )
  tokens <- strsplit(candidates[[1L]], "[[:space:]]+")[[1L]]
  fields <- strsplit(tokens, "=", fixed = TRUE)
  .fastkpc_full_cuda_phase35_feasibility_require(
    all(lengths(fields) == 2L), paste0(label, " output is malformed")
  )
  keys <- vapply(fields, `[[`, character(1L), 1L)
  values <- vapply(fields, `[[`, character(1L), 2L)
  .fastkpc_full_cuda_phase35_feasibility_require(
    !anyDuplicated(keys) && identical(keys, required_fields),
    paste0(label, " output field set or order mismatch")
  )
  numeric_values <- suppressWarnings(as.numeric(values))
  .fastkpc_full_cuda_phase35_feasibility_require(
    !anyNA(numeric_values) && all(is.finite(numeric_values)) &&
      all(numeric_values >= 0),
    paste0(label, " output contains an invalid measurement")
  )
  setNames(as.list(numeric_values), keys)
}

fastkpc_full_cuda_phase35_parse_candidate_measurements <- function(
    eig_output, block_output) {
  eig_fields <- c(
    "n", "repeats", "selected", "info", "full_lwork", "partial_lwork",
    "full_total_ms", "full_per_component_ms",
    "two_sided_partial_total_ms", "two_sided_partial_per_component_ms",
    "full_110617_bound_ms", "partial_110617_bound_ms"
  )
  block_fields <- c(
    "n", "block", "batch", "iterations", "repeats", "sequence_ms",
    "per_component_ms", "bound_110617_ms"
  )
  eig <- .fastkpc_full_cuda_phase35_parse_measurement_line(
    eig_output, eig_fields, "eigensolver benchmark"
  )
  block <- .fastkpc_full_cuda_phase35_parse_measurement_line(
    block_output, block_fields, "block benchmark"
  )
  valid <- identical(eig$n, 351) && identical(eig$selected, 35) &&
    identical(eig$info, 0) && eig$repeats >= 1 &&
    identical(block$n, 351) && identical(block$block, 62) &&
    identical(block$batch, 47) && identical(block$iterations, 12) &&
    block$repeats >= 1 &&
    abs(eig$full_110617_bound_ms -
          eig$full_per_component_ms * 110617) <= 1 &&
    abs(eig$partial_110617_bound_ms -
          eig$two_sided_partial_per_component_ms * 110617) <= 1 &&
    abs(block$bound_110617_ms -
          block$per_component_ms * 110617) <= 1
  .fastkpc_full_cuda_phase35_feasibility_require(
    valid, "candidate benchmark dimensions or arithmetic gate failed"
  )
  table <- rbind(
    data.frame(
      candidate_id = "candidate-a-full-eig-global",
      measurement = c(
        "per_component_ms", "diagnostic_global_110617_bound_ms",
        "workspace_elements"
      ),
      value = c(
        eig$full_per_component_ms, eig$full_110617_bound_ms,
        eig$full_lwork
      ),
      unit = c("ms", "ms", "binary64-elements"),
      feasibility_model_use = c(
        "level-zero-only", "rejection-evidence-only", "memory-diagnostic"
      ),
      stringsAsFactors = FALSE
    ),
    data.frame(
      candidate_id = "candidate-a-two-sided-partial-eig-global",
      measurement = c(
        "per_component_ms", "diagnostic_global_110617_bound_ms",
        "workspace_elements"
      ),
      value = c(
        eig$two_sided_partial_per_component_ms,
        eig$partial_110617_bound_ms, eig$partial_lwork
      ),
      unit = c("ms", "ms", "binary64-elements"),
      feasibility_model_use = c(
        "none", "rejection-evidence-only", "memory-diagnostic"
      ),
      stringsAsFactors = FALSE
    ),
    data.frame(
      candidate_id = "candidate-b-block-krylov-core",
      measurement = c(
        "per_component_ms", "diagnostic_global_110617_core_bound_ms",
        "block_size", "batch_size", "iteration_count"
      ),
      value = c(
        block$per_component_ms, block$bound_110617_ms, block$block,
        block$batch, block$iterations
      ),
      unit = c("ms", "ms", "columns", "components", "iterations"),
      feasibility_model_use = c(
        "none", "rejection-evidence-only", "configuration",
        "configuration", "configuration"
      ),
      stringsAsFactors = FALSE
    )
  )
  rownames(table) <- NULL
  list(
    eig = eig,
    block = block,
    global_full_eig_per_component_ms = eig$full_per_component_ms,
    global_full_eig_bound_ms = eig$full_110617_bound_ms,
    global_partial_eig_bound_ms = eig$partial_110617_bound_ms,
    block_core_bound_ms = block$bound_110617_ms,
    table = table
  )
}

.fastkpc_full_cuda_phase35_sum_field <- function(values, field) {
  if (is.null(values) || nrow(values) == 0L || !field %in% names(values)) {
    return(0)
  }
  sum(values[[field]])
}

.fastkpc_full_cuda_phase35_scale_row <- function(value, scale_kind) {
  exact <- value$exact_groups
  refinement <- value$refinement_groups
  pairs <- value$pairs
  timing <- value$timing[1L, , drop = FALSE]
  max_error <- if (nrow(value$refinement_pairs) == 0L) 0 else {
    max(abs(value$refinement_pairs$
              refined_p_value_difference_from_legacy))
  }
  cpu_count <- sum(vapply(
    c("cpu_dcov_component_count", "cpu_dcov_eigen_count",
      "cpu_dcov_pair_statistic_count", "cpu_gamma_p_value_count"),
    function(field) {
      .fastkpc_full_cuda_phase35_sum_field(exact, field) +
        .fastkpc_full_cuda_phase35_sum_field(refinement, field)
    }, numeric(1L)
  ))
  data.frame(
    scale_id = value$scale_id,
    scale_kind = scale_kind,
    pair_count = nrow(pairs),
    exact_group_count = nrow(exact),
    exact_component_count = sum(exact$component_count),
    screen_decision_flip_count = sum(pairs$screen_decision_flip),
    guarded_pair_count = sum(pairs$refined),
    refinement_group_count = nrow(refinement),
    refinement_component_count = sum(refinement$component_count),
    final_decision_flip_count = sum(pairs$decision_flip),
    near_alpha_pair_count = sum(pairs$near_alpha),
    near_alpha_final_decision_flip_count =
      sum(pairs$near_alpha & pairs$decision_flip),
    maximum_refined_p_value_absolute_error = max_error,
    component_cuda_ms = timing$total_component_cuda_ms,
    pair_gamma_cuda_ms = timing$total_pair_gamma_cuda_ms,
    dcov_host_boundary_ms = timing$total_dcov_host_boundary_ms,
    residual_d2h_bytes =
      .fastkpc_full_cuda_phase35_sum_field(exact, "residual_d2h_bytes") +
      .fastkpc_full_cuda_phase35_sum_field(
        refinement, "residual_d2h_bytes"
      ),
    component_d2h_bytes =
      .fastkpc_full_cuda_phase35_sum_field(exact, "component_d2h_bytes") +
      .fastkpc_full_cuda_phase35_sum_field(
        refinement, "component_d2h_bytes"
      ),
    cpu_numerical_dcov_count = cpu_count,
    simulated_backend_timing = FALSE,
    universal_p_value_parity_claim = FALSE,
    all_screen_flips_guarded =
      all(!pairs$screen_decision_flip | pairs$refined),
    pass = all(value$gates) && sum(pairs$decision_flip) == 0L &&
      max_error <= 1e-10 && cpu_count == 0 &&
      .fastkpc_full_cuda_phase35_sum_field(
        exact, "residual_d2h_bytes"
      ) == 0 &&
      .fastkpc_full_cuda_phase35_sum_field(
        refinement, "residual_d2h_bytes"
      ) == 0 &&
      .fastkpc_full_cuda_phase35_sum_field(
        exact, "component_d2h_bytes"
      ) == 0 &&
      .fastkpc_full_cuda_phase35_sum_field(
        refinement, "component_d2h_bytes"
      ) == 0,
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_phase35_build_scale_results <- function(evidence_bundle) {
  rows <- rbind(
    .fastkpc_full_cuda_phase35_scale_row(
      evidence_bundle$two_scale$hybrid_a, "qualification-complete"
    ),
    .fastkpc_full_cuda_phase35_scale_row(
      evidence_bundle$two_scale$hybrid_b,
      "campaign-slice-level2-reuse-quartiles"
    ),
    .fastkpc_full_cuda_phase35_scale_row(
      evidence_bundle$full_hybrid, "complete-conditional-levels-1-to-7"
    )
  )
  rownames(rows) <- NULL
  .fastkpc_full_cuda_phase35_feasibility_require(
    all(rows$pass), "Phase 3.5 measured scale gate failed"
  )
  rows
}

fastkpc_full_cuda_phase35_build_full_pair_decisions <- function(
    evidence_bundle) {
  pairs <- evidence_bundle$full_hybrid$pairs
  order <- order(pairs$logical_sequence_id, method = "radix")
  pairs <- pairs[order, , drop = FALSE]
  refined_error <- rep(NA_real_, nrow(pairs))
  refined_error[pairs$refined] <- abs(
    pairs$candidate_p_value[pairs$refined] -
      pairs$legacy_reference_p_value[pairs$refined]
  )
  result <- data.frame(
    logical_sequence_id = pairs$logical_sequence_id,
    level = pairs$level,
    prepared_s_key_sha256 = pairs$prepared_s_key_sha256,
    x = pairs$x,
    y = pairs$y,
    S_key = pairs$S_key,
    alpha = pairs$alpha,
    legacy_reference_p_value = pairs$legacy_reference_p_value,
    exact_screen_p_value = pairs$screen_p_value,
    final_route_p_value = pairs$candidate_p_value,
    refined = pairs$refined,
    final_backend = ifelse(
      pairs$refined, "guarded-legacy-full-eig-cuda",
      "exact-cuda-internal-decision-screen-outside-guard"
    ),
    legacy_reference_independent = pairs$legacy_reference_independent,
    exact_screen_independent = pairs$screen_independent,
    final_route_independent = pairs$candidate_independent,
    screen_decision_flip = pairs$screen_decision_flip,
    final_decision_flip = pairs$decision_flip,
    near_alpha = pairs$near_alpha,
    deletes_edge = pairs$deletes_edge,
    refined_p_value_absolute_error = refined_error,
    stringsAsFactors = FALSE
  )
  guard <- evidence_bundle$full_hybrid$guard
  selected <- result$exact_screen_p_value >= guard$lower_inclusive &
    result$exact_screen_p_value <= guard$upper_inclusive
  valid <- nrow(result) == 238276L &&
    !anyDuplicated(result$logical_sequence_id) &&
    identical(result$refined, selected) &&
    sum(result$screen_decision_flip) == 92L &&
    all(!result$screen_decision_flip | result$refined) &&
    !any(result$final_decision_flip) &&
    sum(result$refined) == 1142L &&
    sum(result$near_alpha) == 1478L &&
    !any(result$near_alpha & result$final_decision_flip) &&
    max(result$refined_p_value_absolute_error, na.rm = TRUE) <= 1e-10
  .fastkpc_full_cuda_phase35_feasibility_require(
    valid, "Phase 3.5 full conditional decision table gate failed"
  )
  result
}

fastkpc_full_cuda_phase35_build_candidate_decisions <- function(
    evidence_bundle, candidate_measurements, performance_model,
    scale_results) {
  hybrid_a <- evidence_bundle$two_scale$hybrid_a
  hybrid_b <- evidence_bundle$two_scale$hybrid_b
  full <- evidence_bundle$full_hybrid
  exact_full <- evidence_bundle$full_exact
  hybrid_cpu_count <- sum(scale_results$cpu_numerical_dcov_count)
  hybrid_residual_d2h <- sum(scale_results$residual_d2h_bytes)
  hybrid_component_d2h <- sum(scale_results$component_d2h_bytes)
  full_refined_error <- max(abs(
    full$refinement_pairs$refined_p_value_difference_from_legacy
  ))
  rows <- data.frame(
    candidate_id = c(
      "candidate-a-full-eig-global",
      "candidate-a-two-sided-partial-eig-global",
      "candidate-b-block-krylov-core",
      "candidate-c-exact-cuda",
      "candidate-c-screen-plus-guarded-a-full-eig"
    ),
    architecture = c(
      "sequential-cusolver-full-eig-per-component",
      "two-cusolver-selected-eig-solves-per-component",
      "device-resident-dense-block-krylov-core",
      "exact-centered-distance-cuda-gamma",
      "exact-cuda-screen-with-closed-0.05-to-0.15-full-eig-refinement"
    ),
    evaluated_scope = c(
      "diagnostic-global-extrapolation-and-guarded-refinement",
      "diagnostic-global-extrapolation",
      "component-core-microbenchmark-only",
      "complete-conditional-screen-and-two-measured-scales",
      "two-measured-scales-complete-risk-near-alpha-and-full-conditional"
    ),
    component_representation_parity = c(
      "PASS_REPRESENTATIVE_LEGACY_REFINEMENT",
      "NOT_ESTABLISHED",
      "NOT_ESTABLISHED",
      "NOT_LEGACY_REPRESENTATION",
      "PASS_WHERE_REFINED"
    ),
    pair_moment_p_value_parity = c(
      "PASS_REFINED_PAIRS_MAX_ABS_ERROR_LE_1E-10",
      "NOT_ESTABLISHED",
      "NOT_ESTABLISHED",
      "FAIL_UNIVERSAL_LEGACY_P_VALUE_PARITY",
      "PASS_REFINED_PAIRS_ONLY_DECISION_PARITY_GLOBAL"
    ),
    complete_near_alpha_decision_flip_count = c(
      NA_integer_, NA_integer_, NA_integer_,
      sum(exact_full$pairs$near_alpha & exact_full$pairs$decision_flip),
      sum(hybrid_a$pairs$near_alpha & hybrid_a$pairs$decision_flip)
    ),
    complete_declared_risk_decision_flip_count = c(
      NA_integer_, NA_integer_, NA_integer_,
      sum(evidence_bundle$two_scale$exact_a$pairs$decision_flip),
      sum(hybrid_a$pairs$decision_flip)
    ),
    scale_a_measured_ms = c(
      hybrid_a$timing$refinement_dcov_host_boundary_ms,
      NA_real_, NA_real_,
      hybrid_a$timing$screen_dcov_host_boundary_ms,
      hybrid_a$timing$total_dcov_host_boundary_ms
    ),
    scale_b_measured_ms = c(
      hybrid_b$timing$refinement_dcov_host_boundary_ms,
      NA_real_, NA_real_,
      hybrid_b$timing$screen_dcov_host_boundary_ms,
      hybrid_b$timing$total_dcov_host_boundary_ms
    ),
    diagnostic_global_bound_ms = c(
      candidate_measurements$global_full_eig_bound_ms,
      candidate_measurements$global_partial_eig_bound_ms,
      candidate_measurements$block_core_bound_ms,
      NA_real_, NA_real_
    ),
    conservative_dcov_bound_ms = c(
      NA_real_, NA_real_, NA_real_, NA_real_,
      performance_model$model$dcov_total_upper_bound_ms
    ),
    residual_d2h_bytes = c(
      0, NA, NA,
      .fastkpc_full_cuda_phase35_sum_field(
        exact_full$groups, "residual_d2h_bytes"
      ),
      hybrid_residual_d2h
    ),
    component_d2h_bytes = c(
      0, NA, NA,
      .fastkpc_full_cuda_phase35_sum_field(
        exact_full$groups, "component_d2h_bytes"
      ),
      hybrid_component_d2h
    ),
    cpu_numerical_dcov_count = c(
      0, NA, NA,
      sum(vapply(
        c("cpu_dcov_component_count", "cpu_dcov_pair_statistic_count",
          "cpu_gamma_p_value_count"),
        function(field) .fastkpc_full_cuda_phase35_sum_field(
          exact_full$groups, field
        ), numeric(1L)
      )),
      hybrid_cpu_count
    ),
    failure_behavior = c(
      "fail-closed-solver-status-returned",
      "microbenchmark-only-not-qualified",
      "no-finalization-or-convergence-contract",
      "fail-closed-native-status",
      "fail-closed-before-replay-no-partial-graph"
    ),
    decision = c(
      "REJECT_GLOBAL_RETAIN_GUARDED_REFINEMENT",
      "REJECT_GLOBAL",
      "REJECT_NOT_GO",
      "REJECT_COMPATIBLE_AUTHORITY",
      "GO_FOR_PHASE8_IMPLEMENTATION_AND_FULL_QUALIFICATION"
    ),
    phase8_go = c(FALSE, FALSE, FALSE, FALSE, TRUE),
    production_promoted = rep(FALSE, 5L),
    universal_p_value_parity_claim = rep(FALSE, 5L),
    reason = c(
      "global component bound exceeds 35-second allocation; guarded measured use is feasible",
      "global component bound exceeds 35-second allocation and parity is unproved",
      "core timing omits finalization convergence pair margin and semantic parity",
      "92 complete-conditional decision flips reject exact CUDA as compatible authority",
      paste0(
        "zero final flips; all screen flips guarded; refined max p error ",
        format(full_refined_error, digits = 8L),
        "; conservative dCov bound fits 47 seconds"
      )
    ),
    stringsAsFactors = FALSE
  )
  selected <- rows[rows$phase8_go, , drop = FALSE]
  valid <- nrow(selected) == 1L &&
    identical(
      selected$candidate_id,
      "candidate-c-screen-plus-guarded-a-full-eig"
    ) &&
    selected$complete_near_alpha_decision_flip_count == 0L &&
    selected$complete_declared_risk_decision_flip_count == 0L &&
    selected$scale_a_measured_ms <= 47000 &&
    selected$scale_b_measured_ms <= 47000 &&
    selected$conservative_dcov_bound_ms <= 47000 &&
    selected$residual_d2h_bytes == 0 &&
    selected$component_d2h_bytes == 0 &&
    selected$cpu_numerical_dcov_count == 0 &&
    !selected$production_promoted &&
    !selected$universal_p_value_parity_claim
  .fastkpc_full_cuda_phase35_feasibility_require(
    valid, "Phase 3.5 candidate architecture decision gate failed"
  )
  rows
}

fastkpc_full_cuda_phase35_build_corpus_policy <- function(
    contracts, evidence_bundle) {
  development <- contracts$development_qualification_corpus_v1$payload
  metamorphic <- contracts$metamorphic_contract_v1$payload
  holdout <- contracts$promotion_holdout_manifest_v1$payload
  scale_a <- evidence_bundle$two_scale$hybrid_a
  rows <- data.frame(
    corpus_id = c(
      "development-qualification-v1", "metamorphic-v1",
      holdout$holdout_id
    ),
    tracked_contract = c(
      "development_qualification_corpus_v1", "metamorphic_contract_v1",
      "promotion_holdout_manifest_v1"
    ),
    role = c(
      "reusable-development-correctness-and-risk-qualification",
      "reusable-representation-scheduling-and-cache-invariance",
      "sealed-phase10-promotion-holdout"
    ),
    state = c(
      "FROZEN_AND_MEASURED_PHASE35_SCALE_A",
      "FROZEN_PROTOCOL_EXECUTION_DEFERRED_TO_AFFECTED_IMPLEMENTATIONS",
      holdout$state
    ),
    ordinary_development_access = c(TRUE, TRUE, FALSE),
    payload_present_in_repository = c(TRUE, TRUE, FALSE),
    setup_count = c(
      nrow(scale_a$exact_groups), NA_integer_, NA_integer_
    ),
    target_or_component_count = c(
      sum(scale_a$exact_groups$component_count), NA_integer_, NA_integer_
    ),
    pair_count = c(nrow(scale_a$pairs), NA_integer_, NA_integer_),
    near_alpha_pair_count = c(
      sum(scale_a$pairs$near_alpha), NA_integer_, NA_integer_
    ),
    required_class_or_transformation_count = c(
      length(development$required_risk_classes),
      length(metamorphic$transformations),
      length(holdout$custody$release_requires)
    ),
    execution_claim = c(
      "complete-declared-risk-and-near-alpha-development-scale",
      "policy-frozen-no-claim-of-complete-phase35-transformation-execution",
      "not-opened-and-not-accessed"
    ),
    promotion_claim = c(FALSE, FALSE, FALSE),
    pass = c(
      nrow(scale_a$exact_groups) ==
        development$canonical_counts$qualification_setup_count &&
        sum(scale_a$exact_groups$component_count) ==
          development$canonical_counts$target_count &&
        nrow(scale_a$pairs) ==
          development$canonical_counts$dcov_pair_count &&
        sum(scale_a$pairs$near_alpha) ==
          development$canonical_counts$near_alpha_pair_count &&
        !any(scale_a$pairs$decision_flip),
      identical(metamorphic$base_corpus,
                "development_qualification_corpus_v1") &&
        length(metamorphic$transformations) == 7L &&
        identical(metamorphic$failure_policy,
                  "persist-minimal-reproducer-and-stop-promotion"),
      identical(holdout$state, "SEALED_NOT_RELEASED") &&
        identical(holdout$ordinary_development_access, "forbidden") &&
        !isTRUE(holdout$payload_present_in_repository)
    ),
    stringsAsFactors = FALSE
  )
  .fastkpc_full_cuda_phase35_feasibility_require(
    all(rows$pass) && !any(rows$promotion_claim),
    "Phase 3.5 corpus policy gate failed"
  )
  rows
}

fastkpc_full_cuda_phase35_build_performance_tables <- function(performance) {
  model <- performance$model
  bound <- data.frame(
    metric = c(
      "dcov_component", "dcov_pair_and_gamma", "dcov_total",
      "full_campaign_used_or_reserved"
    ),
    measured_ms = c(
      model$measured_full_conditional_component_cuda_ms,
      model$measured_full_conditional_pair_gamma_cuda_ms,
      model$measured_full_conditional_dcov_boundary_ms, NA_real_
    ),
    class_stratified_p95_ms = c(
      model$component_class_p95_bound_ms,
      model$pair_class_p95_bound_ms,
      model$component_class_p95_bound_ms + model$pair_class_p95_bound_ms,
      NA_real_
    ),
    synchronization_p95_ms = c(
      0, model$synchronization_class_p95_bound_ms,
      model$synchronization_class_p95_bound_ms, NA_real_
    ),
    level_zero_ms = c(
      model$level_zero_component_bound_ms,
      model$level_zero_pair_sync_bound_ms,
      model$level_zero_component_bound_ms +
        model$level_zero_pair_sync_bound_ms,
      NA_real_
    ),
    fixed_overhead_ms = c(
      model$fixed_overhead_per_budget_ms,
      model$fixed_overhead_per_budget_ms,
      2 * model$fixed_overhead_per_budget_ms, NA_real_
    ),
    contingency_ratio = c(
      model$throughput_and_variance_contingency_ratio,
      model$throughput_and_variance_contingency_ratio,
      model$throughput_and_variance_contingency_ratio, NA_real_
    ),
    feasibility_bound_ms = c(
      model$dcov_component_upper_bound_ms,
      model$dcov_pair_gamma_upper_bound_ms,
      model$dcov_total_upper_bound_ms,
      model$full_campaign_used_or_reserved_ms
    ),
    allocated_upper_bound_ms = c(
      performance$budget$allocated_upper_bound_ms[
        performance$budget$component == "dcov_component"
      ],
      performance$budget$allocated_upper_bound_ms[
        performance$budget$component == "dcov_pair_and_gamma"
      ],
      model$dcov_allocated_upper_bound_ms,
      model$full_campaign_allocated_upper_bound_ms
    ),
    best_microbenchmark_global_extrapolation_used = FALSE,
    pass = c(
      model$dcov_component_upper_bound_ms <=
        performance$budget$allocated_upper_bound_ms[
          performance$budget$component == "dcov_component"
        ],
      model$dcov_pair_gamma_upper_bound_ms <=
        performance$budget$allocated_upper_bound_ms[
          performance$budget$component == "dcov_pair_and_gamma"
        ],
      model$dcov_total_upper_bound_ms <= model$dcov_allocated_upper_bound_ms,
      model$full_campaign_used_or_reserved_ms <=
        model$full_campaign_allocated_upper_bound_ms
    ),
    stringsAsFactors = FALSE
  )
  classes <- do.call(rbind, Map(
    function(values, route, work) {
      data.frame(
        route = route,
        work = work,
        values,
        stringsAsFactors = FALSE,
        check.names = FALSE
      )
    },
    list(
      performance$exact_component_classes,
      performance$refinement_component_classes,
      performance$exact_pair_classes,
      performance$refinement_pair_classes
    ),
    c("exact-screen", "legacy-refinement", "exact-screen",
      "legacy-refinement"),
    c("component", "component", "pair-and-gamma", "pair-and-gamma")
  ))
  rownames(classes) <- NULL
  .fastkpc_full_cuda_phase35_feasibility_require(
    all(bound$pass) && !any(bound$best_microbenchmark_global_extrapolation_used),
    "Phase 3.5 published performance table gate failed"
  )
  list(bound = bound, budget = performance$budget, classes = classes)
}

.fastkpc_full_cuda_phase35_read_artifact_csv <- function(
    output_dir, name, expected_columns) {
  value <- read.csv(
    file.path(output_dir, name), stringsAsFactors = FALSE,
    check.names = FALSE, na.strings = ""
  )
  .fastkpc_full_cuda_phase35_feasibility_require(
    identical(names(value), expected_columns),
    paste0("feasibility artifact schema mismatch: ", name)
  )
  value
}

fastkpc_full_cuda_phase35_validate_feasibility_artifact <- function(
    output_dir, verify_current_sources = FALSE) {
  required_functions <- c(
    "fastkpc_full_cuda_phase35_validate_identity_envelope",
    "fastkpc_full_cuda_phase35_canonical_json",
    "fastkpc_full_cuda_phase35_sha256_utf8",
    "fastkpc_full_cuda_census_file_hash"
  )
  missing <- required_functions[!vapply(
    required_functions, exists, logical(1L), mode = "function",
    inherits = TRUE
  )]
  .fastkpc_full_cuda_phase35_feasibility_require(
    length(missing) == 0L,
    paste0(
      "feasibility artifact validator dependency is missing: ",
      if (length(missing) == 0L) "unknown" else missing[[1L]]
    )
  )
  .fastkpc_full_cuda_phase35_feasibility_require(
    requireNamespace("jsonlite", quietly = TRUE),
    "jsonlite is required to validate the feasibility artifact"
  )
  expected_files <- c(
    "benchmark_builds.csv", "benchmark_measurements.csv", "cache.csv",
    "candidate_decisions.csv", "commands.txt", "corpus_policy.csv",
    "dense_level_pressure.csv", "environment.txt", "evidence_inputs.csv",
    "execution_receipts.json", "full_pair_decisions.csv", "manifest.json",
    "memory_model.csv", "performance_bound.csv", "performance_budget.csv",
    "performance_classes.csv", "producer_identity.json",
    "reuse_distance.csv", "scale_results.csv", "source_closure.csv",
    "summary.json", "summary.md", "validator_attestations.json"
  )
  .fastkpc_full_cuda_phase35_feasibility_require(
    dir.exists(output_dir) &&
      identical(
        sort(list.files(output_dir), method = "radix"),
        sort(expected_files, method = "radix")
      ),
    "feasibility artifact file set is incomplete"
  )
  read_json <- function(name) jsonlite::read_json(
    file.path(output_dir, name), simplifyVector = FALSE
  )
  manifest <- read_json("manifest.json")
  producer <- read_json("producer_identity.json")
  summary <- read_json("summary.json")
  attestations <- read_json("validator_attestations.json")$attestations
  receipts <- read_json("execution_receipts.json")$execution_receipts
  expected_manifest_fields <- c(
    "schema_version", "claim_scope", "producer_semantic_envelope",
    "payload_manifest_sha256", "payload_file_sha256",
    "semantic_file_count", "validator_attestations_file",
    "volatile_receipt_file"
  )
  .fastkpc_full_cuda_phase35_feasibility_require(
    is.list(manifest) && identical(names(manifest), expected_manifest_fields) &&
      identical(
        manifest$schema_version,
        "full-cuda-ci-phase35-feasibility-manifest-v1"
      ) &&
      identical(
        manifest$claim_scope, "phase3.5-architecture-feasibility-only"
      ),
    "feasibility artifact manifest schema mismatch"
  )
  .fastkpc_full_cuda_phase35_validate_producer(producer)
  .fastkpc_full_cuda_phase35_feasibility_require(
    fastkpc_full_cuda_phase35_validate_identity_envelope(
      manifest$producer_semantic_envelope
    ) &&
      identical(manifest$producer_semantic_envelope$producer, producer) &&
      identical(
        manifest$producer_semantic_envelope$payload_manifest_sha256,
        manifest$payload_manifest_sha256
      ) &&
      length(manifest$producer_semantic_envelope$attestations) == 0L &&
      length(manifest$producer_semantic_envelope$execution_receipts) == 0L,
    "feasibility producer semantic envelope mismatch"
  )
  .fastkpc_full_cuda_phase35_feasibility_require(
    is.list(attestations) && length(attestations) >= 1L &&
      is.list(receipts) && length(receipts) >= 1L,
    "feasibility attestation or receipt namespace is empty"
  )
  for (attestation in attestations) {
    .fastkpc_full_cuda_phase35_validate_attestation(attestation)
    .fastkpc_full_cuda_phase35_feasibility_require(
      identical(
        attestation$attested_producer_sha256, producer$identity_sha256
      ) && identical(attestation$validation_result, "PASS"),
      "feasibility validator attestation linkage mismatch"
    )
  }
  for (receipt in receipts) {
    .fastkpc_full_cuda_phase35_validate_receipt(receipt)
    .fastkpc_full_cuda_phase35_feasibility_require(
      identical(receipt$producer_sha256, producer$identity_sha256),
      "feasibility execution receipt linkage mismatch"
    )
  }

  payload_hashes <- manifest$payload_file_sha256
  .fastkpc_full_cuda_phase35_feasibility_require(
    is.list(payload_hashes) && !is.null(names(payload_hashes)) &&
      !anyNA(names(payload_hashes)) && !anyDuplicated(names(payload_hashes)) &&
      length(payload_hashes) == as.integer(manifest$semantic_file_count) &&
      all(vapply(payload_hashes, function(value) {
        is.character(value) && length(value) == 1L &&
          grepl("^[0-9a-f]{64}$", value)
      }, logical(1L))),
    "feasibility payload file hash manifest is malformed"
  )
  for (name in names(payload_hashes)) {
    path <- file.path(output_dir, name)
    .fastkpc_full_cuda_phase35_feasibility_require(
      file.exists(path) &&
        identical(
          payload_hashes[[name]], fastkpc_full_cuda_census_file_hash(path)
        ),
      paste0("feasibility payload file hash mismatch: ", name)
    )
  }
  recomputed_payload_manifest_sha256 <-
    fastkpc_full_cuda_phase35_sha256_utf8(
      fastkpc_full_cuda_phase35_canonical_json(payload_hashes)
    )
  .fastkpc_full_cuda_phase35_feasibility_require(
    identical(
      recomputed_payload_manifest_sha256,
      manifest$payload_manifest_sha256
    ),
    "feasibility payload manifest identity mismatch"
  )

  source_closure <- .fastkpc_full_cuda_phase35_read_artifact_csv(
    output_dir, "source_closure.csv", c("path", "sha256")
  )
  source_valid <- nrow(source_closure) >= 1L && !anyNA(source_closure) &&
    !anyDuplicated(source_closure$path) &&
    identical(
      source_closure$path, sort(source_closure$path, method = "radix")
    ) && all(grepl("^[0-9a-f]{64}$", source_closure$sha256))
  .fastkpc_full_cuda_phase35_feasibility_require(
    source_valid, "feasibility source closure schema mismatch"
  )
  closure_hashes <- setNames(
    as.list(source_closure$sha256), source_closure$path
  )
  closure_sha256 <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(closure_hashes)
  )
  if (isTRUE(verify_current_sources)) {
    for (index in seq_len(nrow(source_closure))) {
      path <- source_closure$path[[index]]
      .fastkpc_full_cuda_phase35_feasibility_require(
        file.exists(path) &&
          identical(
            fastkpc_full_cuda_census_file_hash(path),
            source_closure$sha256[[index]]
          ),
        paste0("feasibility current source hash mismatch: ", path)
      )
    }
  }
  .fastkpc_full_cuda_phase35_feasibility_require(
    identical(closure_sha256, producer$producer_source_closure_sha256) &&
      identical(summary$source_closure_sha256, closure_sha256) &&
      identical(summary$producer_identity_sha256,
                producer$identity_sha256) &&
      identical(summary$native_binary_sha256,
                producer$native_binary_sha256),
    "feasibility producer closure identity mismatch"
  )
  native_path <- file.path("fastkpc", "build", "fastkpc_cuda.so")
  if (isTRUE(verify_current_sources)) {
    .fastkpc_full_cuda_phase35_feasibility_require(
      file.exists(native_path) &&
        identical(
          fastkpc_full_cuda_census_file_hash(native_path),
          producer$native_binary_sha256
        ),
      "feasibility current native binary SHA-256 mismatch"
    )
  }

  benchmark_builds <- .fastkpc_full_cuda_phase35_read_artifact_csv(
    output_dir, "benchmark_builds.csv",
    c(
      "benchmark", "source_path", "source_sha256", "binary_path",
      "binary_sha256", "compiler_path", "compiler_sha256",
      "compile_arguments", "run_arguments", "raw_output",
      "sequential_idle_gpu_measurement"
    )
  )
  benchmark_measurements <- .fastkpc_full_cuda_phase35_read_artifact_csv(
    output_dir, "benchmark_measurements.csv",
    c(
      "candidate_id", "measurement", "value", "unit",
      "feasibility_model_use"
    )
  )
  benchmark_valid <- nrow(benchmark_builds) == 2L &&
    identical(benchmark_builds$benchmark, c("eig", "block")) &&
    all(benchmark_builds$sequential_idle_gpu_measurement) &&
    all(grepl("^[0-9a-f]{64}$", benchmark_builds$source_sha256)) &&
    all(grepl("^[0-9a-f]{64}$", benchmark_builds$binary_sha256)) &&
    all(grepl("^[0-9a-f]{64}$", benchmark_builds$compiler_sha256))
  .fastkpc_full_cuda_phase35_feasibility_require(
    benchmark_valid, "feasibility benchmark build evidence is malformed"
  )
  if (isTRUE(verify_current_sources)) {
    for (index in seq_len(nrow(benchmark_builds))) {
      for (kind in c("source", "binary", "compiler")) {
        path <- benchmark_builds[[paste0(kind, "_path")]][[index]]
        expected <- benchmark_builds[[paste0(kind, "_sha256")]][[index]]
        .fastkpc_full_cuda_phase35_feasibility_require(
          file.exists(path) &&
            identical(fastkpc_full_cuda_census_file_hash(path), expected),
          paste0("feasibility benchmark ", kind, " hash mismatch")
        )
      }
    }
  }
  reparsed_measurements <-
    fastkpc_full_cuda_phase35_parse_candidate_measurements(
      benchmark_builds$raw_output[
        benchmark_builds$benchmark == "eig"
      ],
      benchmark_builds$raw_output[
        benchmark_builds$benchmark == "block"
      ]
    )
  measurement_key <- paste(
    benchmark_measurements$candidate_id,
    benchmark_measurements$measurement, sep = "|"
  )
  reparsed_key <- paste(
    reparsed_measurements$table$candidate_id,
    reparsed_measurements$table$measurement, sep = "|"
  )
  measurement_valid <- identical(measurement_key, reparsed_key) &&
    max(abs(
      benchmark_measurements$value - reparsed_measurements$table$value
    )) <= 1e-9 &&
    identical(benchmark_measurements$unit, reparsed_measurements$table$unit) &&
    identical(
      benchmark_measurements$feasibility_model_use,
      reparsed_measurements$table$feasibility_model_use
    )
  .fastkpc_full_cuda_phase35_feasibility_require(
    measurement_valid, "feasibility benchmark measurement replay mismatch"
  )

  candidates <- .fastkpc_full_cuda_phase35_read_artifact_csv(
    output_dir, "candidate_decisions.csv",
    c(
      "candidate_id", "architecture", "evaluated_scope",
      "component_representation_parity", "pair_moment_p_value_parity",
      "complete_near_alpha_decision_flip_count",
      "complete_declared_risk_decision_flip_count", "scale_a_measured_ms",
      "scale_b_measured_ms", "diagnostic_global_bound_ms",
      "conservative_dcov_bound_ms", "residual_d2h_bytes",
      "component_d2h_bytes", "cpu_numerical_dcov_count",
      "failure_behavior", "decision", "phase8_go", "production_promoted",
      "universal_p_value_parity_claim", "reason"
    )
  )
  selected <- candidates[candidates$phase8_go, , drop = FALSE]
  candidate_valid <- nrow(candidates) == 5L && nrow(selected) == 1L &&
    identical(
      selected$candidate_id,
      "candidate-c-screen-plus-guarded-a-full-eig"
    ) &&
    identical(
      selected$decision,
      "GO_FOR_PHASE8_IMPLEMENTATION_AND_FULL_QUALIFICATION"
    ) &&
    selected$complete_near_alpha_decision_flip_count == 0L &&
    selected$complete_declared_risk_decision_flip_count == 0L &&
    selected$scale_a_measured_ms <= 47000 &&
    selected$scale_b_measured_ms <= 47000 &&
    selected$conservative_dcov_bound_ms <= 47000 &&
    selected$residual_d2h_bytes == 0 &&
    selected$component_d2h_bytes == 0 &&
    selected$cpu_numerical_dcov_count == 0 &&
    !any(candidates$production_promoted) &&
    !any(candidates$universal_p_value_parity_claim) &&
    candidates$diagnostic_global_bound_ms[
      candidates$candidate_id == "candidate-a-full-eig-global"
    ] > 35000 &&
    candidates$diagnostic_global_bound_ms[
      candidates$candidate_id ==
        "candidate-a-two-sided-partial-eig-global"
    ] > 35000
  .fastkpc_full_cuda_phase35_feasibility_require(
    candidate_valid, "feasibility candidate decision gate failed"
  )

  scales <- .fastkpc_full_cuda_phase35_read_artifact_csv(
    output_dir, "scale_results.csv",
    c(
      "scale_id", "scale_kind", "pair_count", "exact_group_count",
      "exact_component_count", "screen_decision_flip_count",
      "guarded_pair_count", "refinement_group_count",
      "refinement_component_count", "final_decision_flip_count",
      "near_alpha_pair_count", "near_alpha_final_decision_flip_count",
      "maximum_refined_p_value_absolute_error", "component_cuda_ms",
      "pair_gamma_cuda_ms", "dcov_host_boundary_ms", "residual_d2h_bytes",
      "component_d2h_bytes", "cpu_numerical_dcov_count",
      "simulated_backend_timing", "universal_p_value_parity_claim",
      "all_screen_flips_guarded", "pass"
    )
  )
  scale_valid <- nrow(scales) == 3L &&
    identical(
      scales$scale_id,
      c(
        "A_qualification_complete", "B_level2_reuse_quartiles_192",
        "FULL_CONDITIONAL_LEVELS_1_TO_7"
      )
    ) && identical(scales$pair_count, c(3808L, 21380L, 238276L)) &&
    identical(scales$exact_group_count, c(2061L, 192L, 8634L)) &&
    identical(scales$exact_component_count, c(6143L, 7236L, 110617L)) &&
    identical(scales$screen_decision_flip_count, c(92L, 9L, 92L)) &&
    identical(scales$guarded_pair_count, c(1142L, 78L, 1142L)) &&
    identical(scales$refinement_group_count, c(385L, 29L, 385L)) &&
    identical(scales$refinement_component_count, c(1532L, 124L, 1532L)) &&
    !any(scales$final_decision_flip_count) &&
    identical(scales$near_alpha_pair_count, c(1478L, 98L, 1478L)) &&
    !any(scales$near_alpha_final_decision_flip_count) &&
    all(scales$maximum_refined_p_value_absolute_error <= 1e-10) &&
    !any(scales$simulated_backend_timing) &&
    !any(scales$universal_p_value_parity_claim) &&
    all(scales$all_screen_flips_guarded) && all(scales$pass) &&
    all(scales$residual_d2h_bytes == 0) &&
    all(scales$component_d2h_bytes == 0) &&
    all(scales$cpu_numerical_dcov_count == 0)
  .fastkpc_full_cuda_phase35_feasibility_require(
    scale_valid, "feasibility measured scale gate failed"
  )

  pairs <- .fastkpc_full_cuda_phase35_read_artifact_csv(
    output_dir, "full_pair_decisions.csv",
    c(
      "logical_sequence_id", "level", "prepared_s_key_sha256", "x", "y",
      "S_key", "alpha", "legacy_reference_p_value",
      "exact_screen_p_value", "final_route_p_value", "refined",
      "final_backend", "legacy_reference_independent",
      "exact_screen_independent", "final_route_independent",
      "screen_decision_flip", "final_decision_flip", "near_alpha",
      "deletes_edge", "refined_p_value_absolute_error"
    )
  )
  selected_by_guard <- pairs$exact_screen_p_value >= 0.05 &
    pairs$exact_screen_p_value <= 0.15
  level_counts <- table(factor(pairs$level, levels = 1:7))
  pair_valid <- nrow(pairs) == 238276L &&
    !anyDuplicated(pairs$logical_sequence_id) &&
    identical(
      pairs$logical_sequence_id,
      sort(pairs$logical_sequence_id, method = "radix")
    ) && all(grepl("^[0-9a-f]{64}$", pairs$prepared_s_key_sha256)) &&
    identical(as.integer(level_counts),
              c(52659L, 125293L, 40694L, 13293L, 5422L, 835L, 80L)) &&
    all(is.finite(pairs$legacy_reference_p_value)) &&
    all(is.finite(pairs$exact_screen_p_value)) &&
    all(is.finite(pairs$final_route_p_value)) &&
    all(pairs$legacy_reference_p_value >= 0 &
          pairs$legacy_reference_p_value <= 1) &&
    all(pairs$exact_screen_p_value >= 0 & pairs$exact_screen_p_value <= 1) &&
    all(pairs$final_route_p_value >= 0 & pairs$final_route_p_value <= 1) &&
    identical(pairs$refined, selected_by_guard) &&
    sum(pairs$screen_decision_flip) == 92L &&
    all(!pairs$screen_decision_flip | pairs$refined) &&
    !any(pairs$final_decision_flip) && sum(pairs$refined) == 1142L &&
    sum(pairs$near_alpha) == 1478L &&
    !any(pairs$near_alpha & pairs$final_decision_flip) &&
    all(is.na(pairs$refined_p_value_absolute_error[!pairs$refined])) &&
    !anyNA(pairs$refined_p_value_absolute_error[pairs$refined]) &&
    max(pairs$refined_p_value_absolute_error[pairs$refined]) <= 1e-10 &&
    max(abs(
      pairs$refined_p_value_absolute_error[pairs$refined] -
        abs(
          pairs$final_route_p_value[pairs$refined] -
            pairs$legacy_reference_p_value[pairs$refined]
        )
    )) <= 1e-14 &&
    all(
      pairs$final_route_p_value[!pairs$refined] ==
        pairs$exact_screen_p_value[!pairs$refined]
    ) &&
    all(pairs$final_backend[pairs$refined] ==
          "guarded-legacy-full-eig-cuda") &&
    all(pairs$final_backend[!pairs$refined] ==
          "exact-cuda-internal-decision-screen-outside-guard")
  .fastkpc_full_cuda_phase35_feasibility_require(
    pair_valid, "feasibility full conditional decision gate failed"
  )

  cache <- .fastkpc_full_cuda_phase35_read_artifact_csv(
    output_dir, "cache.csv",
    c(
      "component_capacity", "component_lookups", "component_hits",
      "component_misses", "component_evictions", "hit_ratio"
    )
  )
  expected_cache <- data.frame(
    component_capacity = c(2L, 8L, 16L, 47L),
    component_lookups = rep(476552L, 4L),
    component_hits = c(251933L, 275097L, 309758L, 365935L),
    component_misses = c(224619L, 201455L, 166794L, 110617L),
    component_evictions = c(207351L, 141879L, 83736L, 0L)
  )
  cache_valid <- nrow(cache) == 4L && all(vapply(
    names(expected_cache),
    function(name) identical(cache[[name]], expected_cache[[name]]),
    logical(1L)
  )) && max(abs(
    cache$hit_ratio - cache$component_hits / cache$component_lookups
  )) <= 1e-12
  .fastkpc_full_cuda_phase35_feasibility_require(
    cache_valid, "feasibility cache model gate failed"
  )

  reuse <- .fastkpc_full_cuda_phase35_read_artifact_csv(
    output_dir, "reuse_distance.csv",
    c(
      "probability", "endpoint_reference_distance",
      "future_endpoint_reference_distance"
    )
  )
  reuse_valid <- nrow(reuse) == 8L &&
    max(abs(reuse$probability -
              c(0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1))) <= 1e-15 &&
    identical(
      as.numeric(reuse$endpoint_reference_distance),
      c(1, 2, 2, 16, 66, 103, 203, 719)
    ) &&
    identical(
      reuse$future_endpoint_reference_distance,
      reuse$endpoint_reference_distance
    )
  .fastkpc_full_cuda_phase35_feasibility_require(
    reuse_valid, "feasibility reuse-distance model gate failed"
  )

  memory <- .fastkpc_full_cuda_phase35_read_artifact_csv(
    output_dir, "memory_model.csv", c("item", "bytes", "measured")
  )
  memory_value <- function(item) memory$bytes[memory$item == item]
  memory_valid <- nrow(memory) == 19L && !anyDuplicated(memory$item) &&
    all(is.finite(memory$bytes)) && all(memory$bytes >= 0) &&
    memory_value("fixed_sp_runtime_workspace") == 5859776 &&
    memory_value("fixed_sp_cublas_workspace") == 16777216 &&
    memory_value("largest_prepared_setup_upload") == 441856 &&
    memory_value("exact_component_call_peak") == 46592540 &&
    memory_value("legacy_refinement_call_peak") == 51398896 &&
    memory_value("reference_residual_cache_47x351") == 131976 &&
    memory_value("declared_capacity_bound") <
      0.25 * memory_value("reference_gpu_memory") &&
    memory_value("declared_headroom") ==
      memory_value("reference_gpu_memory") -
        memory_value("declared_capacity_bound")
  .fastkpc_full_cuda_phase35_feasibility_require(
    memory_valid, "feasibility memory model gate failed"
  )

  pressure <- .fastkpc_full_cuda_phase35_read_artifact_csv(
    output_dir, "dense_level_pressure.csv",
    c(
      "level", "pair_count", "component_count",
      "maximum_group_component_count", "peak_live_device_bytes",
      "group_count"
    )
  )
  pressure_valid <- nrow(pressure) == 7L &&
    identical(pressure$level, 1:7) &&
    identical(
      pressure$pair_count,
      c(52659L, 125293L, 40694L, 13293L, 5422L, 835L, 80L)
    ) && sum(pressure$component_count) == 110617L &&
    sum(pressure$group_count) == 8634L &&
    max(pressure$maximum_group_component_count) == 47L &&
    max(pressure$peak_live_device_bytes) == 46592540
  .fastkpc_full_cuda_phase35_feasibility_require(
    pressure_valid, "feasibility dense-level pressure gate failed"
  )

  bound <- .fastkpc_full_cuda_phase35_read_artifact_csv(
    output_dir, "performance_bound.csv",
    c(
      "metric", "measured_ms", "class_stratified_p95_ms",
      "synchronization_p95_ms", "level_zero_ms", "fixed_overhead_ms",
      "contingency_ratio", "feasibility_bound_ms",
      "allocated_upper_bound_ms",
      "best_microbenchmark_global_extrapolation_used", "pass"
    )
  )
  budget <- .fastkpc_full_cuda_phase35_read_artifact_csv(
    output_dir, "performance_budget.csv",
    c(
      "component", "allocated_upper_bound_ms",
      "feasibility_used_or_reserved_ms", "implemented_in_phase35"
    )
  )
  classes <- .fastkpc_full_cuda_phase35_read_artifact_csv(
    output_dir, "performance_classes.csv",
    c("route", "work", "class", "count", "rate_ms", "group_count",
      "bound_ms")
  )
  component_bound <- bound$feasibility_bound_ms[
    bound$metric == "dcov_component"
  ]
  pair_bound <- bound$feasibility_bound_ms[
    bound$metric == "dcov_pair_and_gamma"
  ]
  dcov_bound <- bound$feasibility_bound_ms[bound$metric == "dcov_total"]
  campaign_bound <- bound$feasibility_bound_ms[
    bound$metric == "full_campaign_used_or_reserved"
  ]
  performance_valid <- nrow(bound) == 4L && all(bound$pass) &&
    !any(bound$best_microbenchmark_global_extrapolation_used) &&
    component_bound <= 35000 && pair_bound <= 12000 &&
    dcov_bound <= 47000 && campaign_bound <= 120000 &&
    nrow(budget) == 8L &&
    identical(
      budget$component,
      c(
        "input_and_h2d", "native_setup", "gcv_selection",
        "fixed_sp_solve_and_residual", "dcov_component",
        "dcov_pair_and_gamma", "control_replay_and_packaging",
        "contingency"
      )
    ) && sum(budget$allocated_upper_bound_ms) == 120000L &&
    abs(sum(budget$feasibility_used_or_reserved_ms) - campaign_bound) <= 1 &&
    all(
      budget$feasibility_used_or_reserved_ms[
        !budget$implemented_in_phase35
      ] == budget$allocated_upper_bound_ms[!budget$implemented_in_phase35]
    ) && nrow(classes) >= 4L && all(classes$count > 0) &&
    all(classes$group_count > 0) && all(classes$rate_ms >= 0) &&
    all(classes$bound_ms >= 0) &&
    sum(classes$count[
      classes$route == "exact-screen" & classes$work == "component"
    ]) == 110617L &&
    sum(classes$count[
      classes$route == "legacy-refinement" &
        classes$work == "component"
    ]) == 1532L &&
    sum(classes$count[
      classes$route == "exact-screen" &
        classes$work == "pair-and-gamma"
    ]) == 238276L &&
    sum(classes$count[
      classes$route == "legacy-refinement" &
        classes$work == "pair-and-gamma"
    ]) == 1142L
  .fastkpc_full_cuda_phase35_feasibility_require(
    performance_valid, "feasibility conservative performance gate failed"
  )

  corpus <- .fastkpc_full_cuda_phase35_read_artifact_csv(
    output_dir, "corpus_policy.csv",
    c(
      "corpus_id", "tracked_contract", "role", "state",
      "ordinary_development_access", "payload_present_in_repository",
      "setup_count", "target_or_component_count", "pair_count",
      "near_alpha_pair_count", "required_class_or_transformation_count",
      "execution_claim", "promotion_claim", "pass"
    )
  )
  corpus_valid <- nrow(corpus) == 3L && all(corpus$pass) &&
    !any(corpus$promotion_claim) &&
    identical(
      corpus$tracked_contract,
      c(
        "development_qualification_corpus_v1", "metamorphic_contract_v1",
        "promotion_holdout_manifest_v1"
      )
    ) && corpus$setup_count[[1L]] == 2061L &&
    corpus$target_or_component_count[[1L]] == 6143L &&
    corpus$pair_count[[1L]] == 3808L &&
    corpus$near_alpha_pair_count[[1L]] == 1478L &&
    !corpus$ordinary_development_access[[3L]] &&
    !corpus$payload_present_in_repository[[3L]] &&
    identical(corpus$state[[3L]], "SEALED_NOT_RELEASED") &&
    grepl("no-claim-of-complete", corpus$execution_claim[[2L]],
          fixed = TRUE)
  .fastkpc_full_cuda_phase35_feasibility_require(
    corpus_valid, "feasibility corpus policy gate failed"
  )

  evidence_inputs <- .fastkpc_full_cuda_phase35_read_artifact_csv(
    output_dir, "evidence_inputs.csv",
    c(
      "evidence_id", "path", "sha256",
      "embedded_summary_sufficient_for_on_disk_validation"
    )
  )
  evidence_valid <- nrow(evidence_inputs) == 6L &&
    !anyDuplicated(evidence_inputs$evidence_id) &&
    all(grepl("^[0-9a-f]{64}$", evidence_inputs$sha256)) &&
    all(evidence_inputs$embedded_summary_sufficient_for_on_disk_validation) &&
    identical(
      evidence_inputs$sha256[
        evidence_inputs$evidence_id == "executed-native-binary"
      ], producer$native_binary_sha256
    )
  .fastkpc_full_cuda_phase35_feasibility_require(
    evidence_valid, "feasibility evidence input manifest gate failed"
  )

  tracked <- fastkpc_full_cuda_phase35_load_contract_set(
    verify_source_artifacts = FALSE
  )
  contract_summary_fields <- c(
    architecture_contract_v1 = "architecture_contract_sha256",
    numerical_contract_v1 = "numerical_contract_sha256",
    artifact_identity_contract_v1 =
      "artifact_identity_contract_sha256",
    reference_machine_v1 = "reference_machine_contract_sha256",
    performance_budget_v1 = "performance_budget_contract_sha256",
    development_qualification_corpus_v1 =
      "development_corpus_contract_sha256",
    metamorphic_contract_v1 = "metamorphic_contract_sha256",
    promotion_holdout_manifest_v1 =
      "promotion_holdout_contract_sha256"
  )
  contract_summary_valid <- all(vapply(
    names(contract_summary_fields),
    function(name) identical(
      summary[[contract_summary_fields[[name]]]], tracked[[name]]$sha256
    ),
    logical(1L)
  ))
  full_scale <- scales[
    scales$scale_id == "FULL_CONDITIONAL_LEVELS_1_TO_7", , drop = FALSE
  ]
  summary_valid <- is.list(summary) &&
    identical(
      summary$schema_version,
      "full-cuda-ci-phase35-feasibility-summary-v1"
    ) &&
    identical(
      summary$claim_scope, "phase3.5-architecture-feasibility-only"
    ) && identical(summary$run_status, "COMPLETE") &&
    identical(summary$selected_candidate_id, selected$candidate_id) &&
    identical(summary$selected_decision, selected$decision) &&
    isTRUE(summary$phase8_go) &&
    !isTRUE(summary$production_backend_promoted) &&
    !isTRUE(summary$phase10_promotion_claim) &&
    !isTRUE(summary$full_graph_claim) &&
    !isTRUE(summary$universal_p_value_parity_claim) &&
    !isTRUE(summary$unguarded_exact_values_are_legacy_p_values) &&
    identical(
      summary$unguarded_exact_role,
      "internal-decision-screen-outside-closed-refinement-guard"
    ) && summary$guard_lower_inclusive == 0.05 &&
    summary$guard_upper_inclusive == 0.15 && summary$alpha == 0.1 &&
    summary$qualification_pair_count == 3808L &&
    summary$qualification_screen_flip_count == 92L &&
    summary$qualification_final_flip_count == 0L &&
    summary$campaign_slice_pair_count == 21380L &&
    summary$campaign_slice_final_flip_count == 0L &&
    summary$full_conditional_pair_count == 238276L &&
    summary$full_conditional_component_count == 110617L &&
    summary$full_conditional_screen_flip_count == 92L &&
    summary$full_conditional_guarded_pair_count == 1142L &&
    summary$full_conditional_final_flip_count == 0L &&
    summary$maximum_refined_p_value_absolute_error <= 1e-10 &&
    abs(summary$measured_full_conditional_dcov_ms -
          full_scale$dcov_host_boundary_ms) <= 1e-9 &&
    abs(summary$conservative_dcov_upper_bound_ms - dcov_bound) <= 1e-9 &&
    summary$allocated_dcov_upper_bound_ms == 47000L &&
    abs(summary$full_campaign_used_or_reserved_ms - campaign_bound) <= 1e-9 &&
    summary$full_campaign_allocated_upper_bound_ms == 120000L &&
    summary$reference_component_capacity == 47L &&
    summary$reference_capacity_miss_count == 110617L &&
    summary$reference_capacity_eviction_count == 0L &&
    summary$declared_peak_device_bytes ==
      memory_value("declared_capacity_bound") &&
    summary$declared_device_headroom_bytes ==
      memory_value("declared_headroom") &&
    summary$residual_d2h_bytes == 0 &&
    summary$component_d2h_bytes == 0 &&
    summary$cpu_numerical_dcov_count == 0L &&
    isTRUE(summary$corpus_policy_pass) &&
    identical(summary$holdout_state, "SEALED_NOT_RELEASED") &&
    isTRUE(summary$pass) && contract_summary_valid
  .fastkpc_full_cuda_phase35_feasibility_require(
    summary_valid, "feasibility artifact summary hard gate failed"
  )
  invisible(list(
    manifest = manifest,
    producer = producer,
    summary = summary,
    candidate_decisions = candidates,
    scale_results = scales,
    performance_bound = bound,
    cache = cache,
    corpus_policy = corpus
  ))
}
