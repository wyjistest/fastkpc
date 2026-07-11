fail <- function(message) stop(message, call. = FALSE)

assert_true <- function(value, message) {
  if (!isTRUE(value)) fail(message)
}

assert_identical <- function(actual, expected, message) {
  if (!identical(actual, expected)) fail(message)
}

assert_error <- function(expression, pattern, message) {
  error <- tryCatch({
    force(expression)
    NULL
  }, error = identity)
  assert_true(
    inherits(error, "error") &&
      grepl(pattern, conditionMessage(error), fixed = TRUE),
    message
  )
}

assert_sha256 <- function(value, message) {
  assert_true(
    is.character(value) && length(value) == 1L && !is.na(value) &&
      grepl("^[0-9a-f]{64}$", value),
    message
  )
}

assert_sorted_reasons <- function(table, message) {
  assert_true(
    is.data.frame(table) &&
      "selection_reasons" %in% names(table) &&
      is.list(table$selection_reasons) &&
      all(vapply(table$selection_reasons, function(value) {
        is.character(value) && length(value) > 0L &&
          identical(value, sort(unique(value), method = "radix"))
      }, logical(1L))),
    message
  )
}

rows_with_reason <- function(table, reason) {
  which(vapply(table$selection_reasons, function(value) {
    reason %in% value
  }, logical(1L)))
}

scramble_selection_inputs <- function(inputs) {
  result <- inputs
  table_names <- c(
    "logical_tests", "residual_requests", "same_s_setup_metadata",
    "target_fit_metadata", "target_risks", "risk_cases"
  )
  set.seed(20260711L)
  for (table_name in table_names) {
    value <- result[[table_name]]
    value <- value[sample.int(nrow(value)), , drop = FALSE]
    rownames(value) <- NULL
    character_fields <- names(value)[vapply(
      value, is.character, logical(1L)
    )]
    factor_fields <- intersect(
      character_fields,
      c(
        "case_type", "residual_key_sha256", "same_S_group_id",
        "condition_bucket", "near_alpha_bucket", "formula_class",
        "residual_key_x", "residual_key_y"
      )
    )
    for (field in factor_fields) {
      levels <- rev(sort(unique(value[[field]]), method = "radix"))
      value[[field]] <- factor(value[[field]], levels = levels)
    }
    result[[table_name]] <- value
  }
  result
}

source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")

run_spectra_fallback_injection <- function() {
  selector_name <-
    "fastkpc_full_cuda_select_prepared_s_iteration_subset"
  original_selector <- get(
    selector_name, envir = .GlobalEnv, inherits = FALSE
  )
  environment_name <- "FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK"
  original_environment <- Sys.getenv(
    environment_name, unset = NA_character_
  )
  on.exit({
    assign(
      selector_name, original_selector,
      envir = .GlobalEnv
    )
    if (is.na(original_environment)) {
      Sys.unsetenv(environment_name)
    } else {
      Sys.setenv(
        FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = original_environment
      )
    }
  }, add = TRUE)

  sample_count <- 40L
  data <- matrix(
    seq_len(sample_count * 2L) / (sample_count * 2L),
    nrow = sample_count,
    ncol = 2L
  )
  storage.mode(data) <- "double"
  keys <- sprintf("%064x", seq_len(88L))
  logical_tests <- data.frame(
    logical_sequence_id = seq_len(44L),
    S_size = rep(1L, 44L),
    reference_p_value = rep(0.2, 44L),
    alpha = rep(0.1, 44L),
    reference_decision = rep("independent", 44L),
    reference_independent = rep(TRUE, 44L),
    residual_key_x = keys[seq_len(44L)],
    residual_key_y = keys[44L + seq_len(44L)],
    stringsAsFactors = FALSE
  )
  assign(
    selector_name,
    function(inputs) list(logical_tests = logical_tests),
    envir = .GlobalEnv
  )
  residuals <- new.env(hash = TRUE, parent = emptyenv())
  for (index in seq_along(keys)) {
    assign(
      keys[[index]],
      as.numeric(seq_len(sample_count) + index),
      envir = residuals
    )
  }
  valid_diagnostics <- list(
    lowrank_mode = "spectra",
    lowrank_full_eig_count = 0L,
    lowrank_spectra_count = 2L,
    lowrank_spectra_converged_count = 2L,
    lowrank_spectra_failed_count = 0L,
    lowrank_spectra_fallback_full_eig_count = 0L,
    lowrank_spectra_nconv = 2L * 35L
  )
  fastkpc_full_cuda_prepared_s_validate_dcov_spectra_diagnostics(
    valid_diagnostics, numCol = 35L
  )
  fallback_diagnostics <- valid_diagnostics
  fallback_diagnostics$lowrank_spectra_fallback_full_eig_count <- 1L
  fallback_oracle <- function(x, y, numCol, index) {
    list(
      p.value = 0.2,
      diagnostics = fallback_diagnostics
    )
  }
  Sys.setenv(
    FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK =
      "prepared-s-focused-fallback-sentinel"
  )
  assert_error(
    fastkpc_full_cuda_run_prepared_s_dcov_parity(
      inputs = list(
        data = data,
        dataset_sha256 = fastkpc_full_cuda_data_hash(data)
      ),
      logical_tests = logical_tests,
      residuals = residuals,
      oracle_fun = fallback_oracle
    ),
    "Spectra diagnostics mismatch",
    paste0(
      "matching p-value parity must reject a reported full-eigen fallback"
    )
  )
  assert_identical(
    Sys.getenv(environment_name, unset = NA_character_),
    "prepared-s-focused-fallback-sentinel",
    "fallback rejection must restore the prior Spectra environment"
  )
}

reason_store <- fastkpc_full_cuda_prepared_s_selection_reason_store()
fastkpc_full_cuda_prepared_s_add_selection_reason(
  reason_store, "group-a", c("reason-z", "reason-a", "reason-z")
)
fastkpc_full_cuda_prepared_s_add_selection_reason(
  reason_store,
  c("group-a", "group-b", "group-b"),
  c("reason-b", "reason-z", "reason-a")
)
assert_identical(
  fastkpc_full_cuda_prepared_s_selection_reasons(
    reason_store, c("group-a", "group-b")
  ),
  list(
    c("reason-a", "reason-b", "reason-z"),
    c("reason-a", "reason-z")
  ),
  "selection reasons must support one-to-many and pairwise aggregation"
)
run_spectra_fallback_injection()

if (identical(
      Sys.getenv("FASTKPC_PREPARED_S_TEST_SCOPE", unset = "full"),
      "focused"
    )) {
  cat("PASS focused Prepared-S reason and Spectra fail-closed tests\n")
  quit(save = "no", status = 0L)
}

required_packages <- c("mgcv", "jsonlite", "Rcpp")
missing_packages <- required_packages[!vapply(
  required_packages, requireNamespace, logical(1L), quietly = TRUE
)]
if (length(missing_packages) > 0L) {
  cat(
    "SKIP Prepared-S real subset: required packages are unavailable: ",
    paste(missing_packages, collapse = ","), "\n",
    sep = ""
  )
  quit(save = "no", status = 0L)
}

census_dir <- Sys.getenv(
  "FASTKPC_PHASE1_CENSUS_DIR",
  unset = "fastkpc/artifacts/full_cuda_ci/workload_census_351x48_v1"
)
data_path <- Sys.getenv(
  "FASTKPC_PHASE2_DATA_PATH",
  unset = paste0(
    "fastkpc/artifacts/kpc_tprs_real_zhu/",
    "cancer_RD-causalDiscoveryInput.rds"
  )
)
if (!dir.exists(census_dir) || !file.exists(data_path)) {
  cat(
    "SKIP Prepared-S real subset: authenticated artifacts are unavailable\n"
  )
  quit(save = "no", status = 0L)
}

inputs <- fastkpc_full_cuda_prepared_s_load_inputs(
  census_dir = census_dir,
  data_path = data_path
)

iteration <- fastkpc_full_cuda_select_prepared_s_iteration_subset(inputs)
assert_true(nrow(iteration$setup_groups) == 44L, "iteration setup count")
assert_true(nrow(iteration$target_keys) == 270L, "iteration target count")
assert_true(nrow(iteration$logical_tests) == 44L, "iteration dCov count")
assert_identical(
  as.character(iteration$setup_groups$same_S_group_id),
  sort(as.character(iteration$setup_groups$same_S_group_id),
       method = "radix"),
  "iteration setup order"
)
assert_identical(
  as.character(iteration$target_keys$residual_key_sha256),
  sort(as.character(iteration$target_keys$residual_key_sha256),
       method = "radix"),
  "iteration target order"
)
assert_identical(
  as.integer(iteration$logical_tests$logical_sequence_id),
  sort(as.integer(iteration$logical_tests$logical_sequence_id)),
  "iteration logical-test order"
)
assert_sorted_reasons(
  iteration$setup_groups, "iteration setup reasons must be canonical"
)
assert_sorted_reasons(
  iteration$target_keys, "iteration target reasons must be canonical"
)
assert_sorted_reasons(
  iteration$logical_tests, "iteration logical reasons must be canonical"
)
assert_sha256(
  iteration$iteration_subset_hash,
  "iteration subset hash must be versioned and deterministic"
)
assert_identical(
  iteration$iteration_subset_hash,
  "d69956655e2ee0186aceb4bf2d545b831051c389aed139769d9e5902f433fb96",
  "iteration subset hash must retain the frozen Task 5 identity"
)

risk_table <-
  fastkpc_full_cuda_reconstruct_prepared_s_target_risk_table(inputs)
assert_true(
  nrow(risk_table) == 110617L &&
    !anyDuplicated(as.character(risk_table$residual_key_sha256)) &&
    identical(
      as.character(risk_table$residual_key_sha256),
      sort(as.character(risk_table$residual_key_sha256), method = "radix")
    ),
  "reconstructed target-risk table must preserve canonical key order"
)
assert_true(
  all(c(
    "penalty_count", "condition", "condition_bucket", "S_size",
    "request_multiplicity", "setup_target_count",
    "setup_logical_request_count", "convergence_signature",
    "optimizer_iterations"
  ) %in% names(risk_table)),
  "reconstructed target-risk table must expose selection metadata"
)
rank_deficient <- risk_table$rank_deficient %in% TRUE
assert_true(
  any(rank_deficient) &&
    all(is.infinite(risk_table$condition[rank_deficient])) &&
    all(risk_table$condition_bucket[rank_deficient] ==
          "rank_deficient_inf") &&
    all(risk_table$nonfinite_metadata[rank_deficient]) &&
    all(risk_table$coefficient_all_finite[rank_deficient]) &&
    all(risk_table$fitted_all_finite[rank_deficient]) &&
    all(risk_table$residual_all_finite[rank_deficient]),
  "rank-deficient targets must retain Inf metadata and finite outputs"
)

all_logical <- as.data.frame(
  inputs$logical_tests, stringsAsFactors = FALSE
)
tight <- all_logical$S_size > 0L &
  is.finite(all_logical$absolute_log_distance_from_alpha) &
  all_logical$absolute_log_distance_from_alpha <= 1e-3
tight_ids <- as.integer(all_logical$logical_sequence_id[tight])
assert_true(
  length(tight_ids) == 6L &&
    all(tight_ids %in% iteration$logical_tests$logical_sequence_id),
  "iteration must include all six canonical tight-alpha tests"
)

risk_key_index_x <- match(
  as.character(all_logical$residual_key_x),
  as.character(risk_table$residual_key_sha256)
)
risk_key_index_y <- match(
  as.character(all_logical$residual_key_y),
  as.character(risk_table$residual_key_sha256)
)
numerical_or_convergence_risk <- with(
  risk_table,
  high_condition | rank_deficient | near_constant_target |
    near_constant_conditioner | mgcv_warning | mgcv_nonconverged |
    nonfinite_metadata
)
finite_ordinary_target <- with(
  risk_table,
  coefficient_all_finite & fitted_all_finite & residual_all_finite &
    is.finite(condition) & condition < 1e8 &
    !numerical_or_convergence_risk
)
ordinary_eligible <- all_logical$S_size > 0L &
  !is.na(risk_key_index_x) & !is.na(risk_key_index_y) &
  finite_ordinary_target[risk_key_index_x] &
  finite_ordinary_target[risk_key_index_y] &
  is.finite(all_logical$absolute_log_distance_from_alpha) &
  all_logical$absolute_log_distance_from_alpha > log(2)
key_x <- as.character(all_logical$residual_key_x)
key_y <- as.character(all_logical$residual_key_y)
x_first <- key_x <= key_y
canonical_pair_key <- paste0(
  ifelse(x_first, key_x, key_y), "|",
  ifelse(x_first, key_y, key_x)
)
assert_true(
  all(risk_table$multi_penalty[risk_table$S_size %in% 3:7]),
  paste0(
    "canonical S_size 3..7 targets must retain unavoidable structural ",
    "multi-penalty classification"
  )
)
for (S_size in seq_len(7L)) {
  reason <- paste0("ordinary_lower_median:S_size=", S_size)
  selected_index <- rows_with_reason(iteration$logical_tests, reason)
  assert_true(
    length(selected_index) == 1L,
    paste0("iteration must select one ordinary witness for S_size=", S_size)
  )
  selected <- iteration$logical_tests[selected_index, , drop = FALSE]
  endpoints <- match(
    c(selected$residual_key_x[[1L]], selected$residual_key_y[[1L]]),
    risk_table$residual_key_sha256
  )
  assert_true(
    !anyNA(endpoints) && all(finite_ordinary_target[endpoints]) &&
      selected$absolute_log_distance_from_alpha[[1L]] > log(2),
    paste0(
      "ordinary witness endpoints must satisfy frozen numerical and ",
      "convergence risk rules for S_size=", S_size
    )
  )
  if (S_size >= 3L) {
    assert_true(
      all(risk_table$multi_penalty[endpoints]),
      paste0(
        "ordinary S_size=", S_size,
        " witness must allow structural multi_penalty"
      )
    )
  }
  candidates <- which(
    ordinary_eligible & all_logical$S_size == S_size
  )
  candidates <- candidates[order(
    canonical_pair_key[candidates],
    all_logical$logical_sequence_id[candidates],
    method = "radix"
  )]
  expected <- candidates[[(length(candidates) + 1L) %/% 2L]]
  assert_identical(
    selected$logical_sequence_id[[1L]],
    as.integer(all_logical$logical_sequence_id[[expected]]),
    paste0("ordinary lower-median identity mismatch for S_size=", S_size)
  )
}

nonconverged <- risk_table$mgcv_nonconverged %in% TRUE
convergence_strata <- paste(
  risk_table$convergence_signature[nonconverged],
  risk_table$S_size[nonconverged],
  risk_table$condition_bucket[nonconverged],
  sep = "|"
)
observed_convergence_strata <- sort(
  unique(convergence_strata), method = "radix"
)
assert_true(
  length(observed_convergence_strata) == 3L,
  "canonical frozen mgcv_nonconverged target strata must equal three"
)
for (stratum in observed_convergence_strata) {
  reason <- paste0("convergence_risk:", stratum)
  selected_index <- rows_with_reason(iteration$target_keys, reason)
  assert_true(
    length(selected_index) == 1L,
    paste0("iteration must select one frozen convergence stratum: ", stratum)
  )
  candidates <- which(
    nonconverged & paste(
      risk_table$convergence_signature,
      risk_table$S_size,
      risk_table$condition_bucket,
      sep = "|"
    ) == stratum
  )
  expected <- candidates[order(
    -risk_table$optimizer_iterations[candidates],
    risk_table$residual_key_sha256[candidates],
    method = "radix"
  )[[1L]]]
  assert_identical(
    iteration$target_keys$residual_key_sha256[[selected_index]],
    risk_table$residual_key_sha256[[expected]],
    paste0("frozen convergence stratum ordering mismatch: ", stratum)
  )
}
convergence_reason_count <- sum(vapply(
  iteration$target_keys$selection_reasons,
  function(value) sum(startsWith(value, "convergence_risk:")),
  integer(1L)
))
assert_identical(
  as.integer(convergence_reason_count), 3L,
  "iteration must record exactly three frozen convergence reasons"
)

setup_group_ids <- sort(unique(
  as.character(risk_table$same_S_group_id)
), method = "radix")
first_target_by_group <- match(
  setup_group_ids, risk_table$same_S_group_id
)
setup_metrics <- data.frame(
  same_S_group_id = setup_group_ids,
  setup_target_count =
    risk_table$setup_target_count[first_target_by_group],
  setup_logical_request_count =
    risk_table$setup_logical_request_count[first_target_by_group],
  stringsAsFactors = FALSE
)
anchor_specs <- list(
  list("setup_target_fanout_anchor", "setup_target_count", 2L),
  list("setup_target_fanout_anchor", "setup_target_count", 9L),
  list("setup_target_fanout_anchor", "setup_target_count", 47L),
  list(
    "setup_logical_request_load_anchor",
    "setup_logical_request_count", 2L
  ),
  list(
    "setup_logical_request_load_anchor",
    "setup_logical_request_count", 16L
  ),
  list(
    "setup_logical_request_load_anchor",
    "setup_logical_request_count", 3092L
  )
)
for (spec in anchor_specs) {
  reason <- paste0(spec[[1L]], "=", spec[[3L]])
  selected_index <- rows_with_reason(iteration$setup_groups, reason)
  assert_true(
    length(selected_index) == 1L,
    paste0("iteration must record one anchor reason: ", reason)
  )
  metric <- setup_metrics[[spec[[2L]]]]
  distance <- abs(metric - spec[[3L]])
  candidates <- setup_metrics$same_S_group_id[
    distance == min(distance)
  ]
  expected_group <- sort(candidates, method = "radix")[[1L]]
  selected_group <-
    iteration$setup_groups$same_S_group_id[[selected_index]]
  assert_identical(
    selected_group, expected_group,
    paste0("anchor nearest/tie semantics mismatch: ", reason)
  )
  canonical_group_keys <- risk_table$residual_key_sha256[
    risk_table$same_S_group_id == selected_group
  ]
  assert_true(
    all(canonical_group_keys %in%
          iteration$target_keys$residual_key_sha256),
    paste0("anchor must include every target in group: ", reason)
  )
}

for (index in seq_len(nrow(iteration$logical_tests))) {
  logical_row <- iteration$logical_tests[index, , drop = FALSE]
  logical_id <- logical_row$logical_sequence_id[[1L]]
  endpoints <- c(
    logical_row$residual_key_x[[1L]],
    logical_row$residual_key_y[[1L]]
  )
  endpoint_index <- match(
    endpoints, iteration$target_keys$residual_key_sha256
  )
  assert_true(
    !anyNA(endpoint_index) && all(vapply(
      iteration$target_keys$selection_reasons[endpoint_index],
      function(value) paste0("logical_endpoint:", logical_id) %in% value,
      logical(1L)
    )),
    paste0("logical endpoint closure mismatch: ", logical_id)
  )
}

canonical_setups <- as.data.frame(
  inputs$same_s_setup_metadata, stringsAsFactors = FALSE
)
for (group_id in iteration$setup_groups$same_S_group_id) {
  group_keys <- sort(
    risk_table$residual_key_sha256[
      risk_table$same_S_group_id == group_id
    ],
    method = "radix"
  )
  setup_index <- match(group_id, canonical_setups$same_S_group_id)
  representative <-
    canonical_setups$representative_residual_key_sha256[[setup_index]]
  lower_median <- group_keys[[(length(group_keys) + 1L) %/% 2L]]
  maximum <- group_keys[[length(group_keys)]]
  closure <- c(representative, lower_median, maximum)
  closure_reasons <- c(
    "setup_representative", "setup_lower_median_target",
    "setup_maximum_target"
  )
  closure_index <- match(
    closure, iteration$target_keys$residual_key_sha256
  )
  assert_true(
    !anyNA(closure_index) && all(vapply(seq_along(closure), function(i) {
      closure_reasons[[i]] %in%
        iteration$target_keys$selection_reasons[[closure_index[[i]]]]
    }, logical(1L))),
    paste0("setup three-target closure mismatch: ", group_id)
  )
}

qualification <-
  fastkpc_full_cuda_select_prepared_s_qualification_subset(inputs)
assert_true(
  nrow(qualification$seed_target_keys) == 2356L,
  "qualification seed count"
)
assert_true(
  nrow(qualification$target_keys) == 6143L,
  "qualification expanded target count"
)
assert_true(
  nrow(qualification$logical_tests) == 3808L,
  "qualification logical-test count"
)
assert_true(
  nrow(qualification$setup_groups) == 2061L,
  "qualification same-S group count"
)
assert_true(
  sum(qualification$logical_tests$near_alpha) == 1478L,
  "all conditional near-alpha tests must be selected"
)

penalty_distribution <- table(factor(
  qualification$target_keys$penalty_count,
  levels = c(1L, 3L, 4L, 5L, 6L, 7L)
))
assert_identical(
  unname(as.integer(penalty_distribution)),
  c(3327L, 872L, 837L, 730L, 312L, 65L),
  "qualification penalty distribution"
)

rare <- with(
  risk_table,
  rank_deficient | nonfinite_metadata | mgcv_nonconverged |
    (high_condition & penalty_count == 1L)
)
rare_keys <- as.character(risk_table$residual_key_sha256[rare])
seed_keys <- as.character(
  qualification$seed_target_keys$residual_key_sha256
)
assert_true(
  all(rare_keys %in% seed_keys),
  "qualification seed must include every canonical rare-risk key"
)

conditional_near_alpha <- with(
  inputs$logical_tests,
  S_size > 0L & is.finite(absolute_log_distance_from_alpha) &
    absolute_log_distance_from_alpha <= log(2)
)
near_alpha_ids <- as.integer(
  inputs$logical_tests$logical_sequence_id[conditional_near_alpha]
)
assert_true(
  length(near_alpha_ids) == 1478L &&
    all(near_alpha_ids %in%
          qualification$logical_tests$logical_sequence_id),
  "qualification must include every conditional near-alpha test"
)

assert_sorted_reasons(
  qualification$seed_target_keys,
  "qualification seed reasons must be canonical"
)
assert_sorted_reasons(
  qualification$setup_groups,
  "qualification setup reasons must be canonical"
)
assert_sorted_reasons(
  qualification$target_keys,
  "qualification target reasons must be canonical"
)
assert_sorted_reasons(
  qualification$logical_tests,
  "qualification logical reasons must be canonical"
)
assert_sha256(
  qualification$qualification_subset_hash,
  "qualification subset hash must be versioned and deterministic"
)

coverage <- qualification$coverage
assert_true(
  is.data.frame(coverage) &&
    all(c(
      "coverage_type", "coverage_value", "canonical_count",
      "selected_count", "coverage_claimed"
    ) %in% names(coverage)) &&
    all(c(
      "risk_class", "condition_bucket", "penalty_count", "S_size",
      "reference_decision", "setup_target_count_quantile"
    ) %in% coverage$coverage_type),
  "qualification coverage must report every required dimension"
)
absent_risks <- c(
  "near_constant_target", "near_constant_conditioner", "mgcv_warning"
)
absent_rows <- coverage$coverage_type == "risk_class" &
  coverage$coverage_value %in% absent_risks
assert_true(
  sum(absent_rows) == length(absent_risks) &&
    all(coverage$canonical_count[absent_rows] == 0L) &&
    all(coverage$selected_count[absent_rows] == 0L) &&
    !any(coverage$coverage_claimed[absent_rows]),
  "canonically absent risk classes must remain zero and unclaimed"
)

scrambled <- scramble_selection_inputs(inputs)
iteration_scrambled <-
  fastkpc_full_cuda_select_prepared_s_iteration_subset(scrambled)
qualification_scrambled <-
  fastkpc_full_cuda_select_prepared_s_qualification_subset(scrambled)
assert_identical(
  iteration_scrambled$iteration_subset_hash,
  iteration$iteration_subset_hash,
  "iteration selection must ignore input row and factor order"
)
assert_identical(
  qualification_scrambled$qualification_subset_hash,
  qualification$qualification_subset_hash,
  "qualification selection must ignore input row and factor order"
)
assert_identical(
  iteration_scrambled$target_keys$residual_key_sha256,
  iteration$target_keys$residual_key_sha256,
  "iteration selected keys must be stable under scrambling"
)
assert_identical(
  qualification_scrambled$target_keys$residual_key_sha256,
  qualification$target_keys$residual_key_sha256,
  "qualification selected keys must be stable under scrambling"
)

if (identical(
      Sys.getenv("FASTKPC_PREPARED_S_TEST_SCOPE", unset = "full"),
      "selection"
    )) {
  cat("PASS exact Prepared-S iteration and qualification selection tests\n")
  quit(save = "no", status = 0L)
}

prepared_by_group <- setNames(lapply(
  seq_len(nrow(iteration$setup_groups)),
  function(index) {
    group_id <- iteration$setup_groups$same_S_group_id[[index]]
    setup_index <- match(
      group_id, inputs$same_s_setup_metadata$same_S_group_id
    )
    assert_true(
      !is.na(setup_index),
      "selected group must resolve to canonical same-S metadata"
    )
    fastkpc_full_cuda_build_prepared_s_setup(
      inputs,
      inputs$same_s_setup_metadata[setup_index, , drop = FALSE]
    )
  }
), as.character(iteration$setup_groups$same_S_group_id))
assert_true(
  length(prepared_by_group) == 44L &&
    identical(
      names(prepared_by_group),
      as.character(iteration$setup_groups$same_S_group_id)
    ),
  "iteration must build every selected PreparedSSetup exactly once"
)

assert_error(
  fastkpc_full_cuda_run_prepared_s_target_parity(
    inputs = inputs,
    prepared_by_group = prepared_by_group,
    target_keys = iteration$target_keys[
      rev(seq_len(nrow(iteration$target_keys))), , drop = FALSE
    ]
  ),
  "target key set/order mismatch",
  "target parity must reject a noncanonical selected-key order"
)
target_parity <- fastkpc_full_cuda_run_prepared_s_target_parity(
  inputs = inputs,
  prepared_by_group = prepared_by_group,
  target_keys = iteration$target_keys
)
assert_true(
  is.list(target_parity) && is.data.frame(target_parity$rows) &&
    nrow(target_parity$rows) == 270L &&
    identical(
      target_parity$rows$residual_key_sha256,
      iteration$target_keys$residual_key_sha256
    ),
  "target parity must preserve the exact iteration target order"
)
assert_true(
  all(target_parity$rows$coefficient_hash_exact) &&
    all(target_parity$rows$fitted_hash_exact) &&
    all(target_parity$rows$residual_hash_exact),
  "all iteration targets must reproduce exact Phase 1 hashes"
)
assert_true(
  is.environment(target_parity$residuals) &&
    identical(target_parity$target_state_build_count, 44L) &&
    identical(target_parity$target_state_cache_group_count, 44L),
  "target parity must cache one TargetState table per same-S group"
)

dcov_env_name <- "FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK"
original_dcov_env <- Sys.getenv(dcov_env_name, unset = NA_character_)
Sys.setenv(FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK =
             "prepared-s-restoration-sentinel")
assert_error(
  fastkpc_full_cuda_run_prepared_s_dcov_parity(
    inputs = inputs,
    logical_tests = iteration$logical_tests[
      rev(seq_len(nrow(iteration$logical_tests))), , drop = FALSE
    ],
    residuals = target_parity$residuals
  ),
  "logical test set/order mismatch",
  "dCov parity must reject a noncanonical logical-test order"
)
assert_identical(
  Sys.getenv(dcov_env_name, unset = NA_character_),
  "prepared-s-restoration-sentinel",
  "dCov order validation must restore the prior environment"
)
dcov_parity <- fastkpc_full_cuda_run_prepared_s_dcov_parity(
  inputs = inputs,
  logical_tests = iteration$logical_tests,
  residuals = target_parity$residuals
)
assert_identical(
  Sys.getenv(dcov_env_name, unset = NA_character_),
  "prepared-s-restoration-sentinel",
  "dCov parity must restore the prior Spectra route environment"
)
assert_true(
  is.data.frame(dcov_parity$rows) && nrow(dcov_parity$rows) == 44L &&
    identical(
      dcov_parity$rows$logical_sequence_id,
      iteration$logical_tests$logical_sequence_id
    ),
  "dCov parity must preserve the exact iteration logical-test order"
)
assert_true(
  max(dcov_parity$rows$absolute_p_value_drift) <= 1e-12 &&
    all(dcov_parity$rows$decision_identical) &&
    identical(dcov_parity$decision_flip_count, 0L),
  "dCov parity must remain within 1e-12 with zero decision flips"
)
assert_true(
  is.list(dcov_parity$rows$diagnostics) &&
    all(vapply(
      dcov_parity$rows$diagnostics,
      function(value) is.list(value) && length(value) > 0L,
      logical(1L)
    )),
  "dCov parity must retain native diagnostics without fallbacks"
)
assert_true(
  all(vapply(dcov_parity$rows$diagnostics, function(diagnostics) {
    identical(diagnostics$lowrank_mode, "spectra") &&
      identical(as.integer(diagnostics$lowrank_full_eig_count), 0L) &&
      identical(as.integer(diagnostics$lowrank_spectra_count), 2L) &&
      identical(
        as.integer(diagnostics$lowrank_spectra_converged_count), 2L
      ) &&
      identical(
        as.integer(diagnostics$lowrank_spectra_failed_count), 0L
      ) &&
      identical(
        as.integer(
          diagnostics$lowrank_spectra_fallback_full_eig_count
        ),
        0L
      ) &&
      as.integer(diagnostics$lowrank_spectra_nconv) >= 2L * 35L
  }, logical(1L))),
  "all dCov rows must prove the exact no-fallback Spectra route"
)

missing_residuals <- list2env(
  as.list(target_parity$residuals, all.names = TRUE),
  envir = new.env(hash = TRUE, parent = emptyenv())
)
missing_key <- iteration$logical_tests$residual_key_x[[1L]]
rm(list = missing_key, envir = missing_residuals)
Sys.setenv(FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK =
             "prepared-s-missing-sentinel")
assert_error(
  fastkpc_full_cuda_run_prepared_s_dcov_parity(
    inputs = inputs,
    logical_tests = iteration$logical_tests,
    residuals = missing_residuals
  ),
  "dCov parity residual is missing",
  "dCov parity must fail closed when an endpoint residual is missing"
)
assert_identical(
  Sys.getenv(dcov_env_name, unset = NA_character_),
  "prepared-s-missing-sentinel",
  "dCov parity must restore the environment after endpoint failure"
)
if (is.na(original_dcov_env)) {
  Sys.unsetenv(dcov_env_name)
} else {
  Sys.setenv(FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = original_dcov_env)
}

cat(
  "METRICS iteration_hash=", iteration$iteration_subset_hash,
  " setup_groups=", nrow(iteration$setup_groups),
  " target_keys=", nrow(iteration$target_keys),
  " logical_tests=", nrow(iteration$logical_tests),
  " qualification_seed=", nrow(qualification$seed_target_keys),
  " qualification_targets=", nrow(qualification$target_keys),
  " qualification_logical=", nrow(qualification$logical_tests),
  " qualification_groups=", nrow(qualification$setup_groups),
  " target_exact=", sum(
    target_parity$rows$coefficient_hash_exact &
      target_parity$rows$fitted_hash_exact &
      target_parity$rows$residual_hash_exact
  ),
  " dcov_max_drift=", format(
    max(dcov_parity$rows$absolute_p_value_drift), scientific = TRUE
  ),
  " dcov_exact=", sum(dcov_parity$rows$p_value_exact),
  " dcov_flips=", dcov_parity$decision_flip_count,
  "\n",
  sep = ""
)
cat("PASS Prepared-S iteration target and dCov parity\n")
