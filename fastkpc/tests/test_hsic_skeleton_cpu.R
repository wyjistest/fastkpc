source("fastkpc/R/native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

max_abs_diff <- function(a, b) {
  max(abs(as.numeric(a) - as.numeric(b)))
}

off_diag_values <- function(mat) {
  mat[row(mat) != col(mat)]
}

set.seed(213)
n <- 54
z <- runif(n, -2, 2)
data <- cbind(
  x1 = sin(z) + rnorm(n, sd = 0.08),
  x2 = cos(z) + rnorm(n, sd = 0.08),
  x3 = z^2 + rnorm(n, sd = 0.08),
  x4 = rnorm(n)
)

build_fastkpc_native(rebuild = TRUE)

gamma_a <- fast_skeleton_cpp_backend(
  data, alpha = 0.2, max_conditioning_size = 1,
  residual_backend = "linear", residual_cache = TRUE,
  ci_method = "hsic.gamma", hsic_params = list(sig = 1)
)
gamma_b <- fast_skeleton_cpp_backend(
  data, alpha = 0.2, max_conditioning_size = 1,
  residual_backend = "linear", residual_cache = TRUE,
  ci_method = "hsic.gamma", hsic_params = list(sig = 1)
)

assert_true(gamma_a$ci_method == "hsic.gamma",
            "skeleton should record hsic.gamma method")
assert_true(gamma_a$ci_backend == "native-cpu",
            "CPU HSIC skeleton should record native-cpu CI backend")
assert_true(all(is.finite(off_diag_values(gamma_a$pMax))),
            "HSIC gamma off-diagonal pMax values should be finite")
assert_true(identical(gamma_a$adjacency, t(gamma_a$adjacency)),
            "HSIC gamma adjacency should be symmetric")
assert_true(identical(gamma_a$adjacency, gamma_b$adjacency),
            "HSIC gamma repeated adjacency should match")
assert_true(max_abs_diff(gamma_a$pMax, gamma_b$pMax) < 1e-12,
            "HSIC gamma repeated pMax should match")

perm_a <- fast_skeleton_cpp_backend(
  data, alpha = 0.2, max_conditioning_size = 1,
  residual_backend = "linear", residual_cache = TRUE,
  ci_method = "hsic.perm", hsic_params = list(sig = 1),
  permutation_params = list(replicates = 25L, seed = 901L,
                            include_observed = TRUE)
)
perm_b <- fast_skeleton_cpp_backend(
  data, alpha = 0.2, max_conditioning_size = 1,
  residual_backend = "linear", residual_cache = TRUE,
  ci_method = "hsic.perm", hsic_params = list(sig = 1),
  permutation_params = list(replicates = 25L, seed = 901L,
                            include_observed = TRUE)
)
perm_c <- fast_skeleton_cpp_backend(
  data, alpha = 0.2, max_conditioning_size = 1,
  residual_backend = "linear", residual_cache = TRUE,
  ci_method = "hsic.perm", hsic_params = list(sig = 1),
  permutation_params = list(replicates = 25L, seed = 902L,
                            include_observed = TRUE)
)

assert_true(perm_a$ci_method == "hsic.perm",
            "skeleton should record hsic.perm method")
assert_true(max_abs_diff(perm_a$pMax, perm_b$pMax) < 1e-12,
            "HSIC permutation fixed seed pMax should repeat")
assert_true(perm_a$ci_diagnostics$ci_hsic_perm_tests > 0,
            "HSIC permutation diagnostics should record test count")
assert_true(max_abs_diff(perm_a$pMax, perm_c$pMax) > 0 ||
              !identical(perm_a$ci_diagnostics, perm_c$ci_diagnostics),
            "different HSIC permutation seed should change pMax or diagnostics")

default <- fast_skeleton_cpp_backend(
  data, alpha = 0.2, max_conditioning_size = 1,
  residual_backend = "linear", residual_cache = TRUE
)
assert_true(default$ci_method == "dcc.gamma",
            "default skeleton method should remain dcc.gamma")

cat("test_hsic_skeleton_cpu.R: PASS\n")
