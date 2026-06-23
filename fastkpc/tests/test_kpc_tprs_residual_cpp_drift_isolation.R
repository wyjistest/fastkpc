source("fastkpc/R/kpc_tprs_residual_cpp.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_has_names <- function(x, required, message) {
  missing <- setdiff(required, names(x))
  if (length(missing) > 0L) {
    fail(paste0(message, ": missing ", paste(missing, collapse = ", ")))
  }
}

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP kpcTprsResidualCPP drift isolation: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

set.seed(62306)
n <- 100L
x <- as.numeric(scale(seq(-2, 2, length.out = n)))
y <- sin(x) + stats::rnorm(n, sd = 0.03)
S <- matrix(x, ncol = 1)

iso <- fastkpc_kpc_tprs_fixed_sp_drift_isolation(
  y = y,
  S = S,
  sp = 1,
  edf_search_grid = exp(seq(log(1e-6), log(1e6), length.out = 41L))
)

assert_has_names(
  iso,
  c("backend_family", "mode", "authoritative", "conditioning_size",
    "uz_projector_distance",
    "raw_projector_distance", "absorbed_projector_distance",
    "penalty_shape_distance", "penalty_scale_ratio",
    "same_raw_sp", "scale_corrected", "edf_matched",
    "classification", "diagnostics"),
  "drift isolation result"
)
assert_true(identical(iso$backend_family, "kpcTprsResidualCPP"),
            "backend family")
assert_true(isFALSE(iso$authoritative),
            "drift isolation must be non-authoritative")
assert_true(is.finite(iso$uz_projector_distance),
            "UZ/full-parameter projector distance finite")
assert_true(is.finite(iso$raw_projector_distance),
            "raw projector distance finite")
assert_true(is.finite(iso$absorbed_projector_distance),
            "absorbed projector distance finite")
assert_true(is.finite(iso$penalty_shape_distance),
            "penalty shape distance finite")
assert_true(is.finite(iso$penalty_scale_ratio),
            "penalty scale ratio finite")
assert_has_names(iso$same_raw_sp,
                 c("candidate_sp", "oracle_sp", "residual_rel_l2",
                   "fitted_rel_l2", "edf_candidate", "edf_oracle"),
                 "same raw sp row")
assert_has_names(iso$scale_corrected,
                 c("candidate_sp", "oracle_sp", "residual_rel_l2",
                   "fitted_rel_l2", "edf_candidate", "edf_oracle"),
                 "scale corrected row")
assert_has_names(iso$edf_matched,
                 c("candidate_sp", "oracle_sp", "residual_rel_l2",
                   "fitted_rel_l2", "edf_candidate", "edf_oracle",
                   "edf_abs_diff"),
                 "EDF matched row")
assert_true(iso$edf_matched$edf_abs_diff <= iso$same_raw_sp$edf_abs_diff,
            "EDF-matched path should not worsen EDF match")
assert_true(iso$classification %in%
              c("function-space", "constraint-intercept", "penalty-shape",
                "penalty-scale", "unclassified"),
            "classification")
assert_has_names(iso$diagnostics,
                 c("selected_eigenvalues", "truncation_eigengap",
                   "rank_T", "rank_TU", "Z_orthogonality_error",
                   "TPS_constraint_error", "pre_rms_column_norms",
                   "post_rms_column_norms"),
                 "1D truncation diagnostics")
assert_true(length(iso$diagnostics$selected_eigenvalues) == 10L,
            "1D should retain k eigen directions before TPS constraint")
assert_true(iso$diagnostics$rank_T == 2L, "1D polynomial rank")
assert_true(iso$diagnostics$rank_TU == 2L, "1D T'U rank")
assert_true(is.finite(iso$diagnostics$truncation_eigengap),
            "truncation eigengap finite")
assert_true(iso$diagnostics$Z_orthogonality_error < 1e-8,
            "TPS null-space basis should be orthonormal")
assert_true(iso$diagnostics$TPS_constraint_error < 1e-8,
            "TPS side constraint should be enforced before identifiability")
assert_true(max(abs(iso$diagnostics$post_rms_column_norms - 1)) < 1e-8,
            "final 1D X columns should be RMS scaled")

campaign <- fastkpc_run_kpc_tprs_drift_isolation_campaign(
  scenarios = fastkpc_kpc_tprs_1d_first_drift_scenarios(seed = 62307, n = 100L),
  sp = 1
)
assert_true(is.data.frame(campaign$summary), "campaign summary")
assert_true(nrow(campaign$summary) >= 4L, "1D-first scenario count")
assert_has_names(campaign$summary,
                 c("scenario", "conditioning_size", "raw_projector_distance",
                   "absorbed_projector_distance", "penalty_shape_distance",
                   "penalty_scale_ratio", "edf_matched_residual_rel_l2",
                   "scale_corrected_residual_rel_l2", "classification",
                   "failure_reason"),
                 "campaign summary")
assert_true(any(nzchar(campaign$summary$failure_reason)) ||
              all(is.finite(campaign$summary$raw_projector_distance)),
            "failed drift scenarios should record a failure reason")

cat("PASS kpcTprsResidualCPP drift isolation\n")
