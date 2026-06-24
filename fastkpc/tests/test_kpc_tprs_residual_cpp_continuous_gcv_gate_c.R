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
  cat("SKIP kpcTprsResidualCPP continuous GCV Gate C: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

check_case <- function(S, y, z, label) {
  oracle <- mgcv::gam(
    formula = fastkpc_kpc_tprs_formula(S),
    data = fastkpc_kpc_tprs_data_frame(y, S),
    method = "GCV.Cp"
  )
  setup <- kpc_tprs_residual_cpp_setup(S)
  oracle_absorbed <- fastkpc_kpc_tprs_absorbed_oracle_setup(y, S)
  mapped <- fastkpc_kpc_tprs_map_mgcv_sp_to_canonical(
    sp = as.numeric(oracle$sp),
    setup = setup,
    oracle_absorbed = oracle_absorbed
  )
  lambda_grid <- exp(seq(log(mapped$canonical_lambda / 20),
                         log(mapped$canonical_lambda * 20),
                         length.out = 17L))

  fit <- fastkpc_kpc_tprs_gcv_candidate(
    y = y,
    S = S,
    lambda_grid = lambda_grid,
    refine = TRUE
  )
  assert_has_names(
    fit,
    c("backend_family", "mode", "authoritative", "selected_sp",
      "score", "edf", "fitted", "residuals", "basis_rank",
      "null_space_rank", "setup_fingerprint", "grid", "diagnostics"),
    paste(label, "GCV candidate")
  )
  assert_true(identical(fit$backend_family, "kpcTprsResidualCPP"),
              paste(label, "backend family"))
  assert_true(identical(fit$mode, "continuous-gcv-candidate-shadow"),
              paste(label, "mode"))
  assert_true(isFALSE(fit$authoritative), paste(label, "non-authoritative"))
  assert_true(isTRUE(fit$diagnostics$brent_refined),
              paste(label, "Brent refinement should run"))
  assert_true(identical(fit$diagnostics$gcv_selection, "mgcv-local"),
              paste(label, "default GCV selection should be mgcv-local"))
  assert_true(is.data.frame(fit$grid) && nrow(fit$grid) >= 3L,
              paste(label, "local bracket diagnostics"))
  assert_true(all(c("lambda", "rss", "edf", "gcv", "valid") %in% names(fit$grid)),
              paste(label, "grid columns"))
  assert_true(is.finite(fit$selected_sp) && fit$selected_sp > 0,
              paste(label, "finite selected lambda"))
  assert_true(is.finite(fit$score) && is.finite(fit$edf),
              paste(label, "finite score/EDF"))
  assert_true(fit$score <= min(fit$grid$gcv[fit$grid$valid]) + 1e-8,
              paste(label, "refined score should not be worse than grid minimum"))
  assert_true(abs(log(fit$selected_sp / mapped$canonical_lambda)) < log(2),
              paste(label, "selected smoothing should be equivalent to mgcv"))
  assert_true(fastkpc_kpc_tprs_rel_l2(fit$fitted, stats::fitted(oracle)) < 1e-3,
              paste(label, "GCV fitted values close to mgcv"))
  assert_true(fastkpc_kpc_tprs_rel_l2(fit$residuals, stats::residuals(oracle)) < 1e-3,
              paste(label, "GCV residuals close to mgcv"))
  assert_true(abs(fit$edf - sum(oracle$edf)) < 1e-3,
              paste(label, "GCV EDF close to mgcv"))
  global_fit <- fastkpc_kpc_tprs_gcv_candidate(
    y = y,
    S = S,
    lambda_grid = lambda_grid,
    refine = TRUE,
    selection = "global-grid"
  )
  assert_true(identical(global_fit$diagnostics$gcv_selection, "global-grid"),
              paste(label, "global-grid diagnostic selection"))
  assert_true(is.data.frame(global_fit$grid) &&
                nrow(global_fit$grid) == length(lambda_grid),
              paste(label, "global grid diagnostics"))

  oracle_z <- mgcv::gam(
    formula = fastkpc_kpc_tprs_formula(S),
    data = fastkpc_kpc_tprs_data_frame(z, S),
    method = "GCV.Cp"
  )
  oracle_z_setup <- kpc_tprs_residual_cpp_setup(S)
  oracle_z_absorbed <- fastkpc_kpc_tprs_absorbed_oracle_setup(z, S)
  mapped_z <- fastkpc_kpc_tprs_map_mgcv_sp_to_canonical(
    sp = as.numeric(oracle_z$sp),
    setup = oracle_z_setup,
    oracle_absorbed = oracle_z_absorbed
  )
  lambda_grid_z <- exp(seq(log(mapped_z$canonical_lambda / 20),
                           log(mapped_z$canonical_lambda * 20),
                           length.out = 17L))
  fit_z <- fastkpc_kpc_tprs_gcv_candidate(
    y = z,
    S = S,
    lambda_grid = lambda_grid_z,
    refine = TRUE
  )
  candidate_p <- dcov_gamma_exact(fit$residuals, fit_z$residuals)$p.value
  oracle_p <- dcov_gamma_exact(stats::residuals(oracle),
                               stats::residuals(oracle_z))$p.value
  assert_true(abs(candidate_p - oracle_p) < 1e-4,
              paste(label, "GCV CI p-value close to mgcv"))
}

set.seed(62314)
n <- 95L
s1 <- stats::runif(n, -2, 2)
y1 <- sin(1.2 * s1) + 0.2 * s1 + stats::rnorm(n, sd = 0.03)
z1 <- cos(0.8 * s1) - 0.1 * s1 + stats::rnorm(n, sd = 0.03)
check_case(matrix(s1, ncol = 1), y1, z1, "|S|=1")

s2 <- stats::runif(n, -1.5, 1.5)
S2 <- cbind(s1 = s1, s2 = s2)
y2 <- sin(s1) + cos(1.3 * s2) + stats::rnorm(n, sd = 0.03)
z2 <- cos(0.7 * s1) - sin(s2) + stats::rnorm(n, sd = 0.03)
check_case(S2, y2, z2, "|S|=2")

cat("PASS kpcTprsResidualCPP continuous GCV Gate C\n")
