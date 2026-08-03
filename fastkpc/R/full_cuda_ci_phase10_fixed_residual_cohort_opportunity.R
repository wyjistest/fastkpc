fastkpc_full_cuda_phase10_fixed_cohort_require <- function(
    condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
  invisible(TRUE)
}

fastkpc_full_cuda_phase10_fixed_cohort_default_output <- function() {
  file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "phase10_profile_v2",
    "fixed-residual-cohort-opportunity-v1.rds"
  )
}

fastkpc_full_cuda_phase10_fixed_cohort_categories <- function() {
  c(
    "all_miss",
    "mixed",
    "all_hit_new_cohort",
    "all_hit_repeated_cohort"
  )
}

fastkpc_full_cuda_phase10_fixed_cohort_close <- function(left, right) {
  length(left) == 1L && length(right) == 1L &&
    is.finite(left) && is.finite(right) &&
    abs(left - right) <= 1e-9 * max(1, abs(right))
}

fastkpc_full_cuda_phase10_fixed_cohort_decision <- function(
    qualified_upper_bound_ms) {
  if (qualified_upper_bound_ms >= 8000) {
    "QUALIFY_WHOLE_COHORT_CACHE_PROTOTYPE"
  } else if (qualified_upper_bound_ms >= 5000) {
    "CONDITIONAL_WHOLE_COHORT_CACHE_PROTOTYPE"
  } else {
    "STOP_CROSS_BATCH_FIXED_RESIDUAL_CACHE_OPPORTUNITY"
  }
}

fastkpc_full_cuda_phase10_fixed_residual_cohort_opportunity <- function(
    summary) {
  categories <- fastkpc_full_cuda_phase10_fixed_cohort_categories()
  batch_fields <- paste0("exact_residual_", categories, "_batch_count")
  target_fields <- paste0("exact_residual_", categories, "_target_count")
  solve_fields <- paste0(
    "exact_residual_", categories, "_solve_host_ms"
  )
  component_fields <- paste0(
    "exact_residual_", categories, "_component_build_ms"
  )
  scalar_fields <- c(
    "n", "p", "elapsed_sec", "cuda_exact_screen_residual_batch_count",
    "cuda_exact_screen_residual_target_count", "unique_residual_key_count",
    "physical_residual_fits", "excess_residual_fit_count",
    "cuda_exact_screen_residual_solve_host_ms",
    "cuda_exact_screen_component_build_ms", "unknown_fallback_count",
    "approximate_backend_count", "cpu_residual_solve_count",
    "cpu_dcov_component_count", "residual_d2h_bytes",
    "component_d2h_bytes", batch_fields, target_fields, solve_fields,
    component_fields
  )
  missing <- scalar_fields[!scalar_fields %in% names(summary)]
  fastkpc_full_cuda_phase10_fixed_cohort_require(
    is.list(summary) && length(missing) == 0L,
    paste(
      "Phase 10 fixed residual cohort summary is missing fields:",
      paste(missing, collapse = ", ")
    )
  )
  values <- vapply(scalar_fields, function(field) {
    value <- summary[[field]]
    if (length(value) != 1L) NA_real_ else as.numeric(value)
  }, numeric(1L))
  count_fields <- c(
    "n", "p", "cuda_exact_screen_residual_batch_count",
    "cuda_exact_screen_residual_target_count", "unique_residual_key_count",
    "physical_residual_fits", "excess_residual_fit_count",
    "unknown_fallback_count", "approximate_backend_count",
    "cpu_residual_solve_count", "cpu_dcov_component_count",
    "residual_d2h_bytes", "component_d2h_bytes", batch_fields,
    target_fields
  )
  fastkpc_full_cuda_phase10_fixed_cohort_require(
    all(is.finite(values) & values >= 0) &&
      all(values[count_fields] == floor(values[count_fields])) &&
      is.character(summary$dataset_key) &&
      length(summary$dataset_key) == 1L &&
      nzchar(summary$dataset_key) &&
      isTRUE(summary$authority_gate_pass),
    "Phase 10 fixed residual cohort summary is malformed"
  )

  batch_count <- unname(values[batch_fields])
  target_count <- unname(values[target_fields])
  solve_ms <- unname(values[solve_fields])
  component_ms <- unname(values[component_fields])
  exact_batch_count <- values[["cuda_exact_screen_residual_batch_count"]]
  exact_target_count <- values[["cuda_exact_screen_residual_target_count"]]
  exact_solve_ms <-
    values[["cuda_exact_screen_residual_solve_host_ms"]]
  exact_component_ms <- values[["cuda_exact_screen_component_build_ms"]]
  authority_gate <- all(values[c(
    "unknown_fallback_count", "approximate_backend_count",
    "cpu_residual_solve_count", "cpu_dcov_component_count",
    "residual_d2h_bytes", "component_d2h_bytes"
  )] == 0)
  accounting_gate <-
    sum(batch_count) == exact_batch_count &&
      sum(target_count) == exact_target_count &&
      fastkpc_full_cuda_phase10_fixed_cohort_close(
        sum(solve_ms), exact_solve_ms
      ) &&
      fastkpc_full_cuda_phase10_fixed_cohort_close(
        sum(component_ms), exact_component_ms
      ) &&
      values[["physical_residual_fits"]] == exact_target_count &&
      values[["unique_residual_key_count"]] +
        values[["excess_residual_fit_count"]] == exact_target_count &&
      sum(target_count[3:4]) <= values[["excess_residual_fit_count"]]
  fastkpc_full_cuda_phase10_fixed_cohort_require(
    authority_gate && accounting_gate && exact_batch_count > 0 &&
      exact_target_count > 0,
    "Phase 10 fixed residual cohort accounting is malformed"
  )

  category_table <- data.frame(
    category = categories,
    qualified_for_reuse = categories == "all_hit_repeated_cohort",
    batch_count = as.integer(batch_count),
    target_count = as.integer(target_count),
    residual_solve_host_ms = solve_ms,
    exact_component_build_ms = component_ms,
    combined_observed_ms = solve_ms + component_ms,
    batch_fraction = batch_count / exact_batch_count,
    target_fraction = target_count / exact_target_count,
    stringsAsFactors = FALSE
  )
  qualified <- category_table[
    category_table$qualified_for_reuse, , drop = FALSE
  ]
  qualified_upper_bound_ms <- qualified$combined_observed_ms[[1L]]
  result <- list(
    schema_version =
      "full-cuda-ci-phase10-fixed-residual-cohort-opportunity-v1",
    claim_scope = paste(
      "diagnostic upper bound for exact-screen whole-cohort reuse;",
      "not a production speedup claim"
    ),
    decision = fastkpc_full_cuda_phase10_fixed_cohort_decision(
      qualified_upper_bound_ms
    ),
    source = list(
      dataset_key = summary$dataset_key,
      n = as.integer(values[["n"]]),
      p = as.integer(values[["p"]]),
      elapsed_ms = 1000 * values[["elapsed_sec"]],
      exact_batch_count = as.integer(exact_batch_count),
      exact_target_count = as.integer(exact_target_count),
      unique_residual_key_count =
        as.integer(values[["unique_residual_key_count"]]),
      excess_residual_fit_count =
        as.integer(values[["excess_residual_fit_count"]]),
      exact_residual_solve_host_ms = exact_solve_ms,
      exact_component_build_ms = exact_component_ms
    ),
    categories = category_table,
    qualified_opportunity = list(
      category = "all_hit_repeated_cohort",
      batch_count = qualified$batch_count[[1L]],
      target_count = qualified$target_count[[1L]],
      residual_solve_upper_bound_ms =
        qualified$residual_solve_host_ms[[1L]],
      exact_component_build_upper_bound_ms =
        qualified$exact_component_build_ms[[1L]],
      combined_upper_bound_ms = qualified_upper_bound_ms,
      batch_fraction = qualified$batch_fraction[[1L]],
      target_fraction = qualified$target_fraction[[1L]]
    ),
    thresholds = list(
      prototype_qualify_ms = 8000,
      conditional_lower_bound_ms = 5000
    ),
    limitations = list(
      target_granular_cache_qualified = FALSE,
      mixed_batch_target_elision_qualified = FALSE,
      all_hit_new_cohort_qualified = FALSE,
      capacity_sweep_available = FALSE,
      capacity_sweep_reason = paste(
        "aggregate v1 does not retain ResidualKey reuse distances;",
        "a prototype is allowed only if this upper bound clears its gate"
      )
    ),
    gates = list(
      category_accounting = accounting_gate,
      strict_authority = authority_gate,
      qualified_category_only = TRUE
    )
  )
  fastkpc_full_cuda_phase10_validate_fixed_residual_cohort_opportunity(result)
  result
}

fastkpc_full_cuda_phase10_validate_fixed_residual_cohort_opportunity <-
function(value) {
  categories <- fastkpc_full_cuda_phase10_fixed_cohort_categories()
  table <- value$categories
  qualified <- if (is.data.frame(table)) {
    table[table$category == "all_hit_repeated_cohort", , drop = FALSE]
  } else NULL
  expected_columns <- c(
    "category", "qualified_for_reuse", "batch_count", "target_count",
    "residual_solve_host_ms", "exact_component_build_ms",
    "combined_observed_ms", "batch_fraction", "target_fraction"
  )
  numeric_columns <- setdiff(
    expected_columns, c("category", "qualified_for_reuse")
  )
  finite_table <- is.data.frame(table) &&
    identical(names(table), expected_columns) &&
    all(vapply(table[numeric_columns], function(column) {
      is.numeric(column) && all(is.finite(column)) && all(column >= 0)
    }, logical(1L)))
  source <- value$source
  close <- fastkpc_full_cuda_phase10_fixed_cohort_close
  source_numeric_fields <- c(
    "n", "p", "elapsed_ms", "exact_batch_count", "exact_target_count",
    "unique_residual_key_count", "excess_residual_fit_count",
    "exact_residual_solve_host_ms", "exact_component_build_ms"
  )
  finite_source <- is.list(source) &&
    all(source_numeric_fields %in% names(source)) &&
    all(vapply(source[source_numeric_fields], function(item) {
      is.numeric(item) && length(item) == 1L && is.finite(item) && item >= 0
    }, logical(1L))) &&
    all(vapply(source[c(
      "n", "p", "exact_batch_count", "exact_target_count",
      "unique_residual_key_count", "excess_residual_fit_count"
    )], function(item) item == floor(item), logical(1L))) &&
    source$n > 0 && source$p > 0 && source$elapsed_ms > 0 &&
    source$exact_batch_count > 0 && source$exact_target_count > 0 &&
    is.character(source$dataset_key) &&
    length(source$dataset_key) == 1L && nzchar(source$dataset_key)
  valid <- is.list(value) && identical(
    value$schema_version,
    "full-cuda-ci-phase10-fixed-residual-cohort-opportunity-v1"
  ) && identical(
    value$claim_scope,
    paste(
      "diagnostic upper bound for exact-screen whole-cohort reuse;",
      "not a production speedup claim"
    )
  ) && finite_table && nrow(table) == length(categories) &&
    identical(table$category, categories) &&
    identical(
      table$qualified_for_reuse,
      categories == "all_hit_repeated_cohort"
    ) && all(table$batch_count == floor(table$batch_count)) &&
    all(table$target_count == floor(table$target_count)) &&
    all(vapply(seq_len(nrow(table)), function(index) {
      close(
        table$combined_observed_ms[[index]],
        table$residual_solve_host_ms[[index]] +
          table$exact_component_build_ms[[index]]
      )
    }, logical(1L))) && finite_source &&
    all(vapply(seq_len(nrow(table)), function(index) {
      close(
        table$batch_fraction[[index]],
        table$batch_count[[index]] / source$exact_batch_count
      ) && close(
        table$target_fraction[[index]],
        table$target_count[[index]] / source$exact_target_count
      )
    }, logical(1L))) &&
    sum(table$batch_count) == source$exact_batch_count &&
    sum(table$target_count) == source$exact_target_count &&
    close(
      sum(table$residual_solve_host_ms),
      source$exact_residual_solve_host_ms
    ) && close(
      sum(table$exact_component_build_ms),
      source$exact_component_build_ms
    ) && source$unique_residual_key_count +
      source$excess_residual_fit_count == source$exact_target_count &&
    sum(table$target_count[3:4]) <= source$excess_residual_fit_count &&
    nrow(qualified) == 1L &&
    identical(
      value$qualified_opportunity$category,
      "all_hit_repeated_cohort"
    ) && value$qualified_opportunity$batch_count ==
      qualified$batch_count[[1L]] &&
    value$qualified_opportunity$target_count ==
      qualified$target_count[[1L]] && close(
        value$qualified_opportunity$residual_solve_upper_bound_ms,
        qualified$residual_solve_host_ms[[1L]]
      ) && close(
        value$qualified_opportunity$exact_component_build_upper_bound_ms,
        qualified$exact_component_build_ms[[1L]]
      ) && close(
        value$qualified_opportunity$combined_upper_bound_ms,
        qualified$combined_observed_ms[[1L]]
      ) && close(
        value$qualified_opportunity$batch_fraction,
        qualified$batch_fraction[[1L]]
      ) && close(
        value$qualified_opportunity$target_fraction,
        qualified$target_fraction[[1L]]
      ) && identical(
        value$decision,
        fastkpc_full_cuda_phase10_fixed_cohort_decision(
          qualified$combined_observed_ms[[1L]]
        )
      ) && identical(value$thresholds$prototype_qualify_ms, 8000) &&
    identical(value$thresholds$conditional_lower_bound_ms, 5000) &&
    identical(value$limitations$target_granular_cache_qualified, FALSE) &&
    identical(value$limitations$mixed_batch_target_elision_qualified, FALSE) &&
    identical(value$limitations$all_hit_new_cohort_qualified, FALSE) &&
    identical(value$limitations$capacity_sweep_available, FALSE) &&
    isTRUE(value$gates$category_accounting) &&
    isTRUE(value$gates$strict_authority) &&
    isTRUE(value$gates$qualified_category_only)
  fastkpc_full_cuda_phase10_fixed_cohort_require(
    valid, "Phase 10 fixed residual cohort opportunity is malformed"
  )
  invisible(TRUE)
}

fastkpc_full_cuda_phase10_write_fixed_residual_cohort_opportunity <- function(
    value, path =
      fastkpc_full_cuda_phase10_fixed_cohort_default_output()) {
  fastkpc_full_cuda_phase10_validate_fixed_residual_cohort_opportunity(value)
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  saveRDS(value, path, compress = "xz")
  invisible(path)
}
