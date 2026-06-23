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
  cat("SKIP kpcTprsResidualCPP mapped-sp Gate B-R: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

set.seed(62310)
n <- 100L
x <- as.numeric(scale(seq(-2, 2, length.out = n)))
y <- sin(1.3 * x) + 0.2 * x + stats::rnorm(n, sd = 0.02)
z <- cos(0.7 * x) - 0.1 * x + stats::rnorm(n, sd = 0.02)

scenarios <- list(
  standard = list(y = y, z = z, S = matrix(x, ncol = 1)),
  translation = list(y = y, z = z, S = matrix(x + 5.25, ncol = 1)),
  rescale = list(y = y, z = z, S = matrix(2.5 * x, ncol = 1)),
  row_permutation = {
    ord <- rev(seq_len(n))
    list(y = y[ord], z = z[ord], S = matrix(x[ord], ncol = 1))
  },
  duplicates = list(
    y = c(y, y[seq_len(8L)]),
    z = c(z, z[seq_len(8L)]),
    S = matrix(c(x, x[seq_len(8L)]), ncol = 1)
  )
)

for (name in names(scenarios)) {
  scenario <- scenarios[[name]]
  parity <- fastkpc_kpc_tprs_mapped_sp_residual_parity(
    y = scenario$y,
    S = scenario$S,
    sp_values = c(1e-6, 1e-3, 1, 1e3, 1e6),
    fitted_rel_l2_tol = 1e-6,
    residual_rel_l2_tol = 1e-6,
    edf_abs_tol = 1e-6
  )
  assert_has_names(
    parity,
    c("backend_family", "mode", "mapped_sp", "fixed_sp",
      "gate_b_r_passed", "passed", "diagnostics"),
    paste(name, "mapped-sp parity")
  )
  assert_true(identical(parity$mode, "mapped-sp-residual-parity-shadow"),
              paste(name, "mode"))
  assert_true(isTRUE(parity$gate_b_r_passed),
              paste(name, "mapped-sp Gate B-R should pass"))
  assert_true(isTRUE(parity$passed), paste(name, "mapped-sp parity passed"))
  assert_true(max(parity$fixed_sp$fitted_rel_l2) <= 1e-6,
              paste(name, "mapped fitted relative L2"))
  assert_true(max(parity$fixed_sp$residual_rel_l2) <= 1e-6,
              paste(name, "mapped residual relative L2"))
  assert_true(max(parity$fixed_sp$edf_abs_diff) <= 1e-6,
              paste(name, "mapped EDF absolute difference"))
  assert_true(identical(parity$mapped_sp$source_sp_semantics,
                        "mgcv-smoothCon-scaled-penalty"),
              paste(name, "source sp semantics"))
  assert_true(identical(parity$mapped_sp$target_lambda_semantics,
                        "kpc-canonical-penalty"),
              paste(name, "target lambda semantics"))

  ci <- fastkpc_kpc_tprs_mapped_sp_ci_parity(
    x = scenario$y,
    y = scenario$z,
    S = scenario$S,
    sp = 1,
    p_abs_tol = 1e-8
  )
  assert_has_names(ci,
                   c("candidate_p", "oracle_p", "p_abs_diff", "passed"),
                   paste(name, "mapped-sp CI parity"))
  assert_true(isTRUE(ci$passed), paste(name, "CI p-value parity"))
  assert_true(ci$p_abs_diff <= 1e-8,
              paste(name, "CI p-value absolute difference"))
}

cat("PASS kpcTprsResidualCPP 1D mapped-sp Gate B-R\n")
