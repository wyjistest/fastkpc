source("fastkpc/R/mgcv_compat_contract.R")
source("fastkpc/R/mgcv_extract_oracle.R")

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP mgcv not installed\n")
  quit(save = "no", status = 0)
}

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

set.seed(41001)
n <- 80
data <- data.frame(
  y = sin(seq_len(n) / 9) + stats::rnorm(n, sd = 0.08),
  s1 = stats::runif(n, -2, 2)
)
legacy <- mgcv::gam(y ~ s(s1), data = data, method = "GCV.Cp")
setup <- fastkpc_mgcv_extract_setup(
  formula = y ~ s(s1),
  data = data,
  sp = legacy$sp,
  method = "GCV.Cp",
  target = 1L,
  S = 2L
)

assert_true(exists("fastkpc_mgcv_solve_setup_fixed_sp"),
            "pure setup self-solve function must exist")
assert_true(exists("fastkpc_mgcv_magic_kernel_fixed_sp_coefficients"),
            "fixed-sp kernel helper must exist")
solve_body_text <- paste(deparse(body(fastkpc_mgcv_solve_setup_fixed_sp)), collapse = "\n")
kernel_body_text <- paste(
  deparse(body(fastkpc_mgcv_magic_kernel_fixed_sp_coefficients)),
  collapse = "\n"
)
body_text <- paste(solve_body_text, kernel_body_text, sep = "\n")
assert_true(!grepl("mgcv::gam", body_text, fixed = TRUE),
            "pure setup solve must not call mgcv::gam")
assert_true(!grepl("mgcv::magic", body_text, fixed = TRUE),
            "pure setup solve must not call mgcv::magic")
assert_true(!grepl("fastkpc_mgcv_gam_fixed_sp_reference", body_text, fixed = TRUE),
            "pure setup solve must not call the mgcv reference path")

solution <- fastkpc_mgcv_solve_setup_fixed_sp(setup)
assert_true(identical(solution$mode, "fixed-sp-setup-self-solve"),
            "pure setup solve mode")
assert_true(identical(solution$solve_source, "fastkpc-fixed-sp"),
            "pure setup solve source")
assert_true(identical(solution$setup_diagnostics$solver_kernel,
                      "mgcv-C-magic-fixed-sp"),
            "pure setup solve must disclose the fixed-sp kernel")
assert_true(length(solution$residuals) == n, "residual length")
assert_true(length(solution$fitted) == n, "fitted length")
assert_true(length(solution$coefficients) == ncol(setup$X), "coefficient length")

cat("PASS mgcv self-solve purity\n")
