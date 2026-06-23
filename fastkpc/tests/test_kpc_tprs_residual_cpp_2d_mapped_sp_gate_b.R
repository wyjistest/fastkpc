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
  cat("SKIP kpcTprsResidualCPP 2D mapped-sp Gate B: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

set.seed(62313)
n <- 90L
s1 <- stats::runif(n, -2, 2)
s2 <- stats::runif(n, -1.5, 1.5)
base_S <- cbind(s1 = s1, s2 = s2)
y <- sin(s1) + cos(1.4 * s2) + 0.15 * s1 * s2 +
  stats::rnorm(n, sd = 0.03)
z <- cos(0.9 * s1) - sin(s2) + stats::rnorm(n, sd = 0.03)
theta <- pi / 6
rotation <- matrix(c(cos(theta), -sin(theta), sin(theta), cos(theta)), 2, 2)

scenarios <- list(
  standard = list(y = y, z = z, S = base_S),
  translation = list(y = y, z = z, S = sweep(base_S, 2, c(4.5, -2.75), "+")),
  common_scale = list(y = y, z = z, S = 2.25 * base_S),
  rotation = list(y = y, z = z, S = base_S %*% rotation),
  row_permutation = {
    ord <- rev(seq_len(n))
    list(y = y[ord], z = z[ord], S = base_S[ord, , drop = FALSE])
  }
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
    c("conditioning_size", "mapped_sp", "fixed_sp",
      "gate_b_r_passed", "passed"),
    paste(name, "2D mapped-sp parity")
  )
  assert_true(identical(parity$conditioning_size, 2L),
              paste(name, "2D conditioning size"))
  assert_true(isTRUE(parity$gate_b_r_passed),
              paste(name, "2D mapped-sp Gate B should pass"))
  assert_true(max(parity$fixed_sp$fitted_rel_l2) <= 1e-6,
              paste(name, "2D mapped fitted relative L2"))
  assert_true(max(parity$fixed_sp$residual_rel_l2) <= 1e-6,
              paste(name, "2D mapped residual relative L2"))
  assert_true(max(parity$fixed_sp$edf_abs_diff) <= 1e-6,
              paste(name, "2D mapped EDF absolute difference"))

  ci <- fastkpc_kpc_tprs_mapped_sp_ci_parity(
    x = scenario$y,
    y = scenario$z,
    S = scenario$S,
    sp = 1,
    p_abs_tol = 1e-8
  )
  assert_true(isTRUE(ci$passed), paste(name, "2D CI p-value parity"))
  assert_true(ci$p_abs_diff <= 1e-8,
              paste(name, "2D CI p-value absolute difference"))
}

cat("PASS kpcTprsResidualCPP 2D mapped-sp Gate B\n")
