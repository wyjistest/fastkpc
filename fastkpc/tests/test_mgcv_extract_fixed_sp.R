source("fastkpc/R/mgcv_compat_contract.R")
source("fastkpc/R/mgcv_extract_oracle.R")

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP mgcv not installed\n")
  quit(save = "no", status = 0)
}

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
relative_l2 <- function(a, b) {
  denom <- sqrt(sum(as.numeric(b)^2))
  if (denom == 0) return(sqrt(sum(as.numeric(a - b)^2)))
  sqrt(sum(as.numeric(a - b)^2)) / denom
}

run_case <- function(name, formula, data, S) {
  legacy <- mgcv::gam(formula, data = data, method = "GCV.Cp")
  ref <- fastkpc_mgcv_gam_fixed_sp_reference(
    formula = formula,
    data = data,
    sp = legacy$sp,
    method = "GCV.Cp",
    target = 1L,
    S = S
  )
  solved <- fastkpc_mgcv_extract_fixed_sp_solve(
    formula = formula,
    data = data,
    sp = legacy$sp,
    method = "GCV.Cp",
    target = 1L,
    S = S
  )

  assert_true(identical(ref$mode, "mgcv-gam-fixed-sp-reference"),
              paste(name, "reference mode"))
  assert_true(identical(ref$solve_source, "mgcv"),
              paste(name, "reference solve source"))
  assert_true(identical(solved$mode, "fixed-sp-self-solve"),
              paste(name, "self-solve mode"))
  assert_true(identical(solved$solve_source, "fastkpc-fixed-sp"),
              paste(name, "solve source"))
  assert_true(identical(solved$sp_source, "fixed-input"),
              paste(name, "sp source"))
  assert_true(identical(solved$gcv_source, "none"),
              paste(name, "gcv source"))
  assert_true(isFALSE(solved$is_self_contained_gcv),
              paste(name, "not self-contained gcv"))

  max_fit <- max(abs(solved$fitted - ref$fitted))
  max_res <- max(abs(solved$residuals - ref$residuals))
  rel_fit <- relative_l2(solved$fitted, ref$fitted)
  rel_res <- relative_l2(solved$residuals, ref$residuals)

  assert_true(max_fit < 1e-5,
              paste(name, "fitted max abs diff too large:", max_fit))
  assert_true(max_res < 1e-5,
              paste(name, "residual max abs diff too large:", max_res))
  assert_true(rel_fit < 1e-5,
              paste(name, "fitted relative L2 too large:", rel_fit))
  assert_true(rel_res < 1e-5,
              paste(name, "residual relative L2 too large:", rel_res))
  assert_true(nchar(solved$setup_fingerprint$fingerprint) > 0,
              paste(name, "setup fingerprint required"))
  assert_true(nchar(solved$target_fingerprint$fingerprint) > 0,
              paste(name, "target fingerprint required"))
}

set.seed(2101)
n <- 90
s1 <- stats::runif(n, -2, 2)
s2 <- stats::runif(n, -2, 2)
s3 <- stats::runif(n, -2, 2)
y <- sin(s1) + cos(s2) + 0.2 * s3 + stats::rnorm(n, sd = 0.1)
data <- data.frame(y = y, s1 = s1, s2 = s2, s3 = s3)

run_case("|S|=1", y ~ s(s1), data, S = 2L)
run_case("|S|=2", y ~ s(s1, s2), data, S = c(2L, 3L))
run_case("|S|=3 additive", y ~ s(s1) + s(s2) + s(s3), data, S = c(2L, 3L, 4L))

wrapper <- fastkpc_mgcv_extract_fixed_sp(
  formula = y ~ s(s1),
  data = data,
  sp = mgcv::gam(y ~ s(s1), data = data, method = "GCV.Cp")$sp,
  method = "GCV.Cp",
  target = 1L,
  S = 2L
)
assert_true(identical(wrapper$mode, "fixed-sp-self-solve"),
            "compatibility wrapper should now use fixed-sp self-solve")
assert_true(identical(wrapper$compatibility_alias_for,
                      "fastkpc_mgcv_extract_fixed_sp_solve"),
            "wrapper must name the self-solve alias target")

cat("PASS mgcv extract fixed-sp self-solve\n")
