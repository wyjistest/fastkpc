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
  cat("SKIP kpcTprsResidualCPP fixed-sp parity: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

set.seed(62304)
n <- 88L
s1 <- stats::runif(n, -2, 2)
s2 <- stats::runif(n, -1, 1)
y1 <- sin(s1) + stats::rnorm(n, sd = 0.04)
y2 <- sin(s1) + 0.2 * s2 + stats::rnorm(n, sd = 0.04)

one <- fastkpc_kpc_tprs_fixed_sp_parity(
  y = y1,
  S = matrix(s1, ncol = 1),
  sp_values = c(1e-3, 1, 1e3),
  fitted_rel_l2_tol = 1e-5,
  residual_rel_l2_tol = 1e-5
)
two <- fastkpc_kpc_tprs_fixed_sp_parity(
  y = y2,
  S = cbind(s1 = s1, s2 = s2),
  sp_values = c(1e-3, 1, 1e3),
  fitted_rel_l2_tol = 1e-5,
  residual_rel_l2_tol = 1e-5
)

required <- c(
  "backend_family", "mode", "authoritative", "conditioning_size",
  "oracle_setup_fingerprint", "candidate_setup_fingerprint",
  "projector_distance", "penalty_spectrum_distance",
  "fixed_sp", "fixed_sp_residual_rel_l2", "fixed_sp_fitted_rel_l2",
  "edf_abs_diff", "gate_b1_passed", "gate_b2_passed", "passed",
  "diagnostics"
)
assert_has_names(one, required, "1D parity result")
assert_has_names(two, required, "2D parity result")
assert_true(identical(one$backend_family, "kpcTprsResidualCPP"),
            "1D backend family")
assert_true(identical(two$backend_family, "kpcTprsResidualCPP"),
            "2D backend family")
assert_true(isFALSE(one$authoritative), "1D parity must be non-authoritative")
assert_true(isFALSE(two$authoritative), "2D parity must be non-authoritative")
assert_true(nrow(one$fixed_sp) == 3L, "1D fixed-sp rows")
assert_true(nrow(two$fixed_sp) == 3L, "2D fixed-sp rows")
assert_true(all(is.finite(one$fixed_sp$residual_rel_l2)),
            "1D residual rel-L2 finite")
assert_true(all(is.finite(two$fixed_sp$residual_rel_l2)),
            "2D residual rel-L2 finite")
assert_true(isFALSE(one$passed) || isFALSE(two$passed),
            "candidate must not silently qualify before fixed-sp parity passes")

campaign <- fastkpc_run_kpc_tprs_fixed_sp_parity_campaign(
  scenarios = list(
    one_d = list(y = y1, S = matrix(s1, ncol = 1)),
    two_d = list(y = y2, S = cbind(s1 = s1, s2 = s2))
  ),
  sp_values = c(1e-3, 1, 1e3)
)
assert_true(is.data.frame(campaign$summary), "campaign summary data frame")
assert_true(nrow(campaign$summary) == 2L, "campaign summary rows")
assert_has_names(campaign$summary,
                 c("scenario", "conditioning_size", "projector_distance",
                   "penalty_spectrum_distance", "max_residual_rel_l2",
                   "max_fitted_rel_l2", "passed"),
                 "campaign summary")
assert_true(isFALSE(all(campaign$summary$passed)),
            "campaign must not report complete fixed-sp parity prematurely")

cat("PASS kpcTprsResidualCPP fixed-sp parity\n")
