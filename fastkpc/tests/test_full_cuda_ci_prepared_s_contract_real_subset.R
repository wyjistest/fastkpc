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
