fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_error <- function(expression, pattern, message) {
  error <- tryCatch({ force(expression); NULL }, error = identity)
  assert_true(inherits(error, "error") &&
                grepl(pattern, conditionMessage(error), fixed = TRUE),
              message)
}

source("fastkpc/R/full_cuda_ci_workload_census.R")

risk_config_contract <- fastkpc_full_cuda_census_risk_config()
assert_true(all(c(
  "risk_schema_version", "near_constant_threshold", "rank_estimator",
  "rank_tolerance_formula", "condition_estimator", "condition_buckets",
  "high_condition_threshold", "near_alpha_tau", "near_alpha_buckets",
  "p_floor", "nonfinite_policy", "selected_sp_policy",
  "warning_capture_policy", "convergence_field_policy"
) %in% names(risk_config_contract)),
"risk config must freeze every threshold, bucket, and metadata policy")

outer_failure <- list(
  converged = TRUE,
  outer.info = list(conv = "iteration limit reached")
)
outer_success <- list(
  converged = TRUE,
  outer.info = list(conv = "full convergence")
)
assert_true(fastkpc_full_cuda_census_nonconverged(outer_failure) &&
              !fastkpc_full_cuda_census_nonconverged(outer_success),
            "nonconvergence risk must interpret an explicit outer.info signal")
outer_fields <- fastkpc_full_cuda_census_convergence_fields(outer_failure)
assert_true(identical(outer_fields$outer.info$source, "fit$outer.info") &&
              identical(outer_fields$outer.info$value$conv,
                        "iteration limit reached"),
            "convergence metadata must retain raw slot provenance")

empty <- fastkpc_full_cuda_census_svd_diagnostics(
  matrix(numeric(), nrow = 0L, ncol = 0L)
)
zero <- fastkpc_full_cuda_census_svd_diagnostics(matrix(0, 2L, 2L))
rank_deficient <- fastkpc_full_cuda_census_svd_diagnostics(
  matrix(c(1, 2, 2, 4), nrow = 2L), expected_rank = 2L
)
high_condition <- fastkpc_full_cuda_census_svd_diagnostics(
  diag(c(1, 1e-13)), expected_rank = 2L
)
nonfinite <- fastkpc_full_cuda_census_svd_diagnostics(
  matrix(c(1, Inf), nrow = 1L), expected_rank = 1L
)
assert_true(empty$rank == 0L && is.na(empty$condition) &&
              empty$bucket == "not_applicable_empty",
            "empty SVD diagnostics must be explicitly not applicable")
assert_true(zero$rank == 0L && is.infinite(zero$condition) &&
              zero$bucket == "rank_deficient_inf",
            "all-zero matrices must be rank deficient")
assert_true(rank_deficient$rank == 1L &&
              is.infinite(rank_deficient$condition) &&
              rank_deficient$bucket == "rank_deficient_inf",
            "rank-deficient matrices must have infinite condition")
assert_true(high_condition$rank == 2L &&
              is.finite(high_condition$condition) &&
              high_condition$bucket == "finite_ge_1e12",
            "finite high-condition matrices must retain their finite bucket")
assert_true(is.na(nonfinite$rank) && is.na(nonfinite$condition) &&
              nonfinite$bucket == "nonfinite_unknown",
            "non-finite matrices must retain an explicit unknown bucket")

near_alpha_distances <- c(
  0, 1e-12, 1e-9, 1e-6, 1e-3,
  log(1.01), log(1.1), log(2), log(2) + 1e-8
)
near_alpha_expected <- c(
  "exact_boundary", "le_1e_minus_12", "le_1e_minus_9",
  "le_1e_minus_6", "le_1e_minus_3", "le_log_1_01",
  "le_log_1_1", "le_log_2", "farther"
)
assert_true(identical(
  vapply(near_alpha_distances,
         fastkpc_full_cuda_census_near_alpha_bucket,
         character(1L)),
  near_alpha_expected
), "near-alpha buckets must use the frozen inclusive boundaries")

near_constant_cases <- list(
  numeric(), 1, rep(1, 3L), 1 + c(0, 1e-10, -1e-10), c(1, 2, 3)
)
near_constant_expected <- c(TRUE, TRUE, TRUE, TRUE, FALSE)
near_constant_actual <- vapply(
  near_constant_cases,
  function(value) fastkpc_full_cuda_census_near_constant(value)$near_constant,
  logical(1L)
)
assert_true(identical(near_constant_actual, near_constant_expected),
            "near-constant classification must cover empty and scalar vectors")

constraint <- matrix(c(1, 0), nrow = 1L)
nullspace <- fastkpc_full_cuda_census_right_nullspace(constraint)
assert_true(identical(dim(nullspace), c(2L, 1L)) &&
              max(abs(constraint %*% nullspace)) < 1e-14,
            "constraint nullspace must be deterministic and orthogonal")

sp_two <- structure(list(
  S = list(diag(2)), first.para = 2L, last.para = 3L,
  first.sp = 2L, last.sp = 2L, bs.dim = 2L
), class = "fastkpc_test_smooth")
sp_one <- structure(list(
  S = list(diag(2) * 3), first.para = 4L, last.para = 5L,
  first.sp = 1L, last.sp = 1L, bs.dim = 2L
), class = "fastkpc_test_smooth")
penalty_order <- fastkpc_full_cuda_census_penalty_components(
  list(smooth = list(sp_two, sp_one)), coefficient_count = 5L
)
assert_true(identical(penalty_order$offsets, c(4L, 2L)) &&
              identical(
                penalty_order$hashes,
                vapply(list(diag(2) * 3, diag(2)),
                       fastkpc_full_cuda_census_metadata_hash,
                       character(1L))
              ),
            "penalty blocks must follow first.sp order, not smooth list order")

inputs <- fastkpc_full_cuda_census_load_inputs(
  oracle_dir = "fastkpc/artifacts/full_cuda_ci/oracle_351x48_v1",
  data_path = paste0(
    "fastkpc/artifacts/kpc_tprs_real_zhu/",
    "cancer_RD-causalDiscoveryInput.rds"
  )
)
structural <- fastkpc_full_cuda_census_structural(inputs)
requests <- structural$residual_requests

selected_groups <- unlist(lapply(c(1L, 2L, 3L, max(requests$S_size)),
                                 function(size) {
  candidates <- requests[requests$S_size == size, , drop = FALSE]
  group_ids <- unique(candidates$same_S_group_id)
  group_ids[[1L]]
}), use.names = FALSE)
subset_requests <- do.call(rbind, lapply(selected_groups, function(group_id) {
  rows <- requests[requests$same_S_group_id == group_id, , drop = FALSE]
  rows[seq_len(min(2L, nrow(rows))), , drop = FALSE]
}))
rownames(subset_requests) <- NULL
subset_requests$shard_id <- 0L
assert_true(nrow(subset_requests) == 8L &&
              length(unique(subset_requests$same_S_group_id)) == 4L,
            "metadata subset must contain two targets from four S groups")

risk_config <- risk_config_contract
fits <- lapply(seq_len(nrow(subset_requests)), function(i) {
  fastkpc_full_cuda_census_fit_key(
    data = inputs$data,
    request_row = subset_requests[i, , drop = FALSE],
    risk_config = risk_config
  )
})
assert_true(all(vapply(fits, function(value) {
  identical(value$target_fit$fit_status[[1L]], "success")
}, logical(1L))), "representative canonical metadata fits must succeed")

setup_observations <- do.call(rbind, lapply(fits, `[[`, "setup_observation"))
target_fits <- do.call(rbind, lapply(fits, `[[`, "target_fit"))
target_risks <- do.call(rbind, lapply(fits, `[[`, "risk_cases"))

required_setup_fields <- c(
  "same_S_group_id", "S_key", "S_size", "formula_class",
  "representative_residual_key_sha256", "formula_semantics_version",
  "model_matrix_nrow", "model_matrix_ncol", "model_matrix_hash",
  "model_matrix_rank", "model_matrix_condition", "penalty_count",
  "penalty_block_dimensions", "penalty_ranks", "penalty_offsets",
  "penalty_hashes", "penalty_nullity", "constraint_dimensions",
  "constraint_rank", "constraint_nullspace_dimension", "constraint_hash",
  "H_dimensions", "H_hash", "weights_policy", "offset_policy",
  "smooth_classes", "basis_dimensions", "conditioning_rank",
  "conditioning_condition", "near_constant_conditioning_count",
  "setup_fingerprint", "mgcv_version", "R_version"
)
required_target_fields <- c(
  "residual_key_sha256", "same_S_group_id", "setup_fingerprint", "shard_id",
  "target", "fit_status", "fit_error", "fit_time_ms", "formula",
  "method", "optimizer", "family", "link", "selected_sp",
  "selected_sp_names", "selected_sp_hash", "GCV_Cp_score", "EDF",
  "convergence_fields", "warning_classes", "warning_messages",
  "coefficient_rank", "coefficient_all_finite", "fitted_all_finite",
  "residual_all_finite", "penalized_system_condition_at_selected_sp",
  "target_sd", "target_near_constant", "coefficient_hash",
  "fitted_hash", "residual_hash", "target_fit_fingerprint"
)
assert_true(identical(names(setup_observations), required_setup_fields),
            "setup observations must expose the approved schema")
assert_true(identical(names(target_fits), required_target_fields),
            "target fits must expose the approved schema")
assert_true(all(target_fits$fit_status == "success") &&
              all(is.finite(target_fits$fit_time_ms)) &&
              all(target_fits$fit_time_ms >= 0) &&
              all(target_fits$coefficient_all_finite) &&
              all(target_fits$fitted_all_finite) &&
              all(target_fits$residual_all_finite),
            "successful target rows must retain fit status and timing")
assert_true(all(vapply(target_fits$selected_sp, function(value) {
  is.numeric(value) && length(value) > 0L && all(is.finite(value))
}, logical(1L))) &&
              all(vapply(target_fits$selected_sp_names, function(value) {
                is.character(value) && length(value) > 0L
              }, logical(1L))),
            "selected sp values and original names must remain separate")

for (group_id in unique(setup_observations$same_S_group_id)) {
  rows <- setup_observations[
    setup_observations$same_S_group_id == group_id, , drop = FALSE
  ]
  assert_true(length(unique(rows$model_matrix_hash)) == 1L &&
                length(unique(rows$constraint_hash)) == 1L &&
                length(unique(rows$setup_fingerprint)) == 1L &&
                length(unique(vapply(rows$penalty_hashes,
                                     fastkpc_full_cuda_census_metadata_hash,
                                     character(1L)))) == 1L,
              "same-S setup observations must be response-independent")
}

same_s_setups <- fastkpc_full_cuda_census_compress_setups(setup_observations)
assert_true(nrow(same_s_setups) == 4L &&
              identical(names(same_s_setups), required_setup_fields),
            "same-S compression must emit one setup row per group")

broken_setups <- setup_observations
broken_setups$model_matrix_hash[[1L]] <- paste0(
  "f", substring(broken_setups$model_matrix_hash[[1L]], 2L)
)
assert_error(
  fastkpc_full_cuda_census_compress_setups(broken_setups),
  "same-S setup invariant violation",
  "setup compression must fail closed on response-dependent metadata"
)

coverage <- fastkpc_full_cuda_census_field_coverage(
  same_s_setup_metadata = same_s_setups,
  target_fit_metadata = target_fits
)
assert_true(all(c("table", "field", "total", "present", "finite",
                  "required", "coverage_ratio") %in% names(coverage)) &&
              all(coverage$coverage_ratio[coverage$required] == 1),
            "required metadata fields must have complete subset coverage")
assert_error(
  fastkpc_full_cuda_census_field_coverage(
    same_s_setup_metadata = same_s_setups,
    target_fit_metadata = target_fits[
      , setdiff(names(target_fits), "residual_hash"), drop = FALSE
    ]
  ),
  "metadata table missing required fields",
  "field coverage must reject a silently omitted required field"
)

risk_cases <- fastkpc_full_cuda_census_risk_cases(
  target_risks = target_risks,
  logical_tests = structural$logical_tests
)
risk_flags <- c(
  "high_condition", "rank_deficient", "near_constant_target",
  "near_constant_conditioner", "multi_penalty", "near_alpha",
  "mgcv_warning", "mgcv_nonconverged", "nonfinite_metadata"
)
assert_true(all(risk_flags %in% names(risk_cases)) &&
              nrow(risk_cases) > 0L &&
              any(risk_cases$multi_penalty) && any(risk_cases$near_alpha),
            "row-level risks must retain multi-penalty and near-alpha cases")

bad_request <- subset_requests[1L, , drop = FALSE]
bad_request$target <- ncol(inputs$data) + 1L
failed_fit <- fastkpc_full_cuda_census_fit_key(
  data = inputs$data,
  request_row = bad_request,
  risk_config = risk_config
)
assert_true(is.null(failed_fit$setup_observation) &&
              failed_fit$target_fit$fit_status[[1L]] == "error" &&
              nzchar(failed_fit$target_fit$fit_error[[1L]]) &&
              identical(failed_fit$target_fit$shard_id[[1L]], 0L),
            "failed mgcv keys must remain explicit error rows")

cat("PASS full CUDA CI workload census metadata\n")
