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
  cat("SKIP kpcTprsResidualCPP penalty geometry: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

set.seed(62308)
n <- 100L
x <- as.numeric(scale(seq(-2, 2, length.out = n)))
y <- cos(1.7 * x) + stats::rnorm(n, sd = 0.02)
S <- matrix(x, ncol = 1)

geom <- fastkpc_kpc_tprs_penalty_geometry_isolation(
  y = y,
  S = S,
  lambda_values = c(1e-6, 1e-3, 1, 1e3, 1e6)
)

assert_has_names(
  geom,
  c("backend_family", "mode", "authoritative", "conditioning_size",
    "signed_eigenvalues", "candidate_stage_identities",
    "oracle_stage_identities", "generalized_spectrum",
    "smoother_comparison", "classification", "diagnostics"),
  "penalty geometry result"
)
assert_true(identical(geom$backend_family, "kpcTprsResidualCPP"),
            "backend family")
assert_true(isFALSE(geom$authoritative),
            "penalty geometry diagnostics must be non-authoritative")
assert_true(length(geom$signed_eigenvalues) == 10L,
            "1D retained signed eigenvalues")
assert_true(any(geom$signed_eigenvalues < 0) &&
              any(geom$signed_eigenvalues > 0),
            "retained eigenvalue signs should be preserved")

assert_has_names(geom$candidate_stage_identities,
                 c("uz_penalty_identity_rel_error",
                   "pre_rms_congruence_rel_error",
                   "post_rms_congruence_rel_error",
                   "ident_absorb_congruence_rel_error"),
                 "candidate stage identities")
assert_has_names(geom$oracle_stage_identities,
                 c("uz_penalty_identity_rel_error"),
                 "oracle stage identities")
assert_true(is.finite(geom$candidate_stage_identities$uz_penalty_identity_rel_error),
            "candidate UZ penalty identity finite")
assert_true(is.finite(geom$oracle_stage_identities$uz_penalty_identity_rel_error),
            "oracle UZ penalty identity finite")
assert_true(is.finite(geom$candidate_stage_identities$pre_rms_congruence_rel_error),
            "pre-RMS congruence finite")
assert_true(is.finite(geom$candidate_stage_identities$post_rms_congruence_rel_error),
            "post-RMS congruence finite")
assert_true(is.finite(geom$candidate_stage_identities$ident_absorb_congruence_rel_error),
            "identifiability congruence finite")
assert_true(geom$candidate_stage_identities$uz_penalty_identity_rel_error < 1e-8,
            "candidate S == UZ_delta' E UZ_delta")
assert_true(geom$oracle_stage_identities$uz_penalty_identity_rel_error < 1e-8,
            "oracle S.scale-adjusted UZ penalty identity")
assert_true(geom$candidate_stage_identities$pre_rms_congruence_rel_error < 1e-12,
            "candidate pre-RMS penalty stage")
assert_true(geom$candidate_stage_identities$post_rms_congruence_rel_error < 1e-12,
            "candidate post-RMS congruence")
assert_true(geom$candidate_stage_identities$ident_absorb_congruence_rel_error < 1e-12,
            "candidate identifiability congruence")

assert_has_names(geom$generalized_spectrum,
                 c("generalized_penalty_rank",
                   "generalized_eigenvalues_positive",
                   "log_spectrum_scale_offset",
                   "log_spectrum_shape_rmse"),
                 "generalized spectrum")
assert_true(geom$generalized_spectrum$generalized_penalty_rank == 8L,
            "1D centered smooth penalty rank")
assert_true(length(geom$generalized_spectrum$generalized_eigenvalues_positive$candidate) == 8L,
            "candidate positive generalized spectrum length")
assert_true(length(geom$generalized_spectrum$generalized_eigenvalues_positive$oracle) == 8L,
            "oracle positive generalized spectrum length")
assert_true(is.finite(geom$generalized_spectrum$log_spectrum_shape_rmse),
            "log spectrum shape RMSE finite")
assert_true(geom$generalized_spectrum$log_spectrum_shape_rmse < 1e-8,
            "generalized spectrum should be shape-aligned after scale offset")

assert_true(is.data.frame(geom$smoother_comparison),
            "smoother comparison table")
assert_true(nrow(geom$smoother_comparison) == 5L,
            "one smoother row per lambda")
assert_has_names(geom$smoother_comparison,
                 c("lambda", "best_scale", "smoother_rel_frobenius"),
                 "smoother comparison")
assert_true(all(is.finite(geom$smoother_comparison$smoother_rel_frobenius)),
            "smoother distances finite")
assert_true(max(geom$smoother_comparison$global_scale_smoother_rel_frobenius) < 1e-6,
            "one global scale should align all smoother matrices")
assert_true(identical(geom$classification, "penalty-scale"),
            "Phase 1e should classify standard 1D drift as penalty scaling")
assert_true(geom$classification %in%
              c("penalty-assembly", "penalty-shape", "penalty-scale",
                "metric-false-positive", "stage-mismatch", "unclassified"),
            "penalty geometry classification")

cat("PASS kpcTprsResidualCPP penalty geometry\n")
