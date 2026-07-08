source("fastkpc/R/mgcv_compat_contract.R")
source("fastkpc/R/mgcv_extract_oracle.R")
source("fastkpc/R/native.R")

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP mgcv fixed-sp C++ solver: mgcv unavailable\n")
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
  setup <- fastkpc_mgcv_extract_setup(
    formula = formula,
    data = data,
    sp = legacy$sp,
    method = "GCV.Cp",
    target = 1L,
    S = S
  )
  reference <- fastkpc_mgcv_solve_setup_fixed_sp(setup)
  cpp <- fastkpc_mgcv_solve_setup_fixed_sp_cpp(setup)

  assert_true(identical(cpp$backend_family, "mgcvExtractCPP"),
              paste(name, "backend family"))
  assert_true(identical(cpp$mode, "fixed-sp-native-cpp-solve"),
              paste(name, "mode"))
  assert_true(identical(cpp$solve_source, "fastkpc-native-cpp-fixed-sp"),
              paste(name, "solve source"))
  assert_true(!isTRUE(cpp$authoritative),
              paste(name, "C++ solver should be shadow-only"))
  assert_true(length(cpp$residuals) == nrow(data),
              paste(name, "residual length"))
  assert_true(length(cpp$coefficients) == ncol(setup$X),
              paste(name, "coefficient length"))

  max_fit <- max(abs(cpp$fitted - reference$fitted))
  max_res <- max(abs(cpp$residuals - reference$residuals))
  rel_fit <- relative_l2(cpp$fitted, reference$fitted)
  rel_res <- relative_l2(cpp$residuals, reference$residuals)
  assert_true(max_fit < 1e-5,
              paste(name, "fitted max abs diff too large:", max_fit))
  assert_true(max_res < 1e-5,
              paste(name, "residual max abs diff too large:", max_res))
  assert_true(rel_fit < 1e-5,
              paste(name, "fitted relative L2 too large:", rel_fit))
  assert_true(rel_res < 1e-5,
              paste(name, "residual relative L2 too large:", rel_res))
}

set.seed(43103)
n <- 86L
s1 <- stats::runif(n, -2, 2)
s2 <- stats::runif(n, -2, 2)
s3 <- stats::runif(n, -1, 1)
y <- sin(s1) + cos(s2) + 0.25 * s3 + stats::rnorm(n, sd = 0.08)
data <- data.frame(y = y, s1 = s1, s2 = s2, s3 = s3)

assert_true(exists("fastkpc_mgcv_solve_setup_fixed_sp_cpp"),
            "C++ fixed-sp setup solver wrapper must exist")

run_case("|S|=1", y ~ s(s1), data, S = 2L)
run_case("|S|=2", y ~ s(s1, s2), data, S = c(2L, 3L))
run_case("|S|=3 additive", y ~ s(s1) + s(s2) + s(s3),
         data, S = c(2L, 3L, 4L))

cat("PASS mgcv fixed-sp C++ solver\n")
