source("fastkpc/R/native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

max_abs_diff <- function(a, b) {
  max(abs(as.numeric(a) - as.numeric(b)))
}

off_diagonal_values <- function(mat) {
  mat[row(mat) != col(mat)]
}

set.seed(321)
n <- 42L
z <- runif(n, -2, 2)
x <- sin(z) + rnorm(n, sd = 0.06)
y <- z^2 + rnorm(n, sd = 0.06)

build_fastkpc_native(rebuild = TRUE)

a <- fast_dcov_perm_cpp(
  x, y, index = 1, replicates = 31L, seed = 700L,
  include_observed = TRUE
)
b <- fast_dcov_perm_cpp(
  x, y, index = 1, replicates = 31L, seed = 700L,
  include_observed = TRUE
)
c <- fast_dcov_perm_cpp(
  x, y, index = 1, replicates = 31L, seed = 701L,
  include_observed = TRUE
)

energy_reference <- energy::dcov.test(x, y, index = 1, R = 1L)

assert_true(a$method == "dcc.perm",
            "native dCov permutation should report dcc.perm")
assert_true(abs(a$statistic - unname(energy_reference$statistic)) < 1e-12,
            "native observed dCov statistic should match energy::dcov.test")
assert_true(is.finite(a$p.value) && a$p.value >= 0 && a$p.value <= 1,
            "native dCov permutation p-value should be in [0, 1]")
assert_true(length(a$replicates) == 31L,
            "native dCov permutation should return requested replicates")
assert_true(identical(a$replicates, b$replicates) &&
              identical(a$p.value, b$p.value),
            "native dCov permutation fixed seed should repeat exactly")
assert_true(!identical(a$replicates, c$replicates),
            "native dCov permutation different seed should change replicates")
assert_true(a$diagnostics$replicates == 31L &&
              isTRUE(a$diagnostics$used_seed) &&
              a$diagnostics$seed == 700L,
            "native dCov permutation diagnostics should record its plan")

data <- cbind(x1 = x, x2 = y, x3 = z, x4 = rnorm(n))
perm_a <- fast_skeleton_cpp_backend(
  data, alpha = 0.2, max_conditioning_size = 1L,
  residual_backend = "linear", residual_cache = TRUE,
  ci_method = "dcc.perm",
  permutation_params = list(replicates = 17L, seed = 88L,
                            include_observed = TRUE)
)
perm_b <- fast_skeleton_cpp_backend(
  data, alpha = 0.2, max_conditioning_size = 1L,
  residual_backend = "linear", residual_cache = TRUE,
  ci_method = "dcc.perm",
  permutation_params = list(replicates = 17L, seed = 88L,
                            include_observed = TRUE)
)

assert_true(perm_a$ci_method == "dcc.perm" &&
              perm_a$ci_backend == "native-cpu",
            "CPU skeleton should expose native dcc.perm routing")
assert_true(perm_a$ci_diagnostics$ci_dcc_perm_tests > 0L,
            "CPU skeleton should record dcc.perm test count")
assert_true(perm_a$ci_diagnostics$ci_dcc_permutation_replicates ==
              perm_a$ci_diagnostics$ci_dcc_perm_tests * 17L,
            "CPU skeleton should account for every permutation replicate")
assert_true(identical(perm_a$adjacency, perm_b$adjacency),
            "CPU dcc.perm fixed-seed adjacency should repeat")
assert_true(max_abs_diff(perm_a$pMax, perm_b$pMax) == 0,
            "CPU dcc.perm fixed-seed pMax should repeat exactly")
assert_true(all(is.finite(off_diagonal_values(perm_a$pMax))),
            "CPU dcc.perm off-diagonal pMax should be finite")

cat("test_dcc_permutation_native.R: PASS\n")
