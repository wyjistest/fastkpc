source("fastkpc/R/fast_kpc.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

max_abs_diff <- function(a, b) {
  max(abs(as.numeric(a) - as.numeric(b)))
}

set.seed(141)
n <- 96L
z <- seq(-2.5, 2.5, length.out = n)
data <- cbind(
  x1 = z,
  x2 = sin(z) + rnorm(n, sd = 0.08),
  x3 = cos(0.5 * z) + rnorm(n, sd = 0.08),
  x4 = z^2 + rnorm(n, sd = 0.08)
)

params <- list(knots = 7L, lambda_count = 13L, ridge = 1e-8)
permutation_params <- list(
  replicates = 19L,
  seed = 505L,
  include_observed = TRUE
)

load_fastkpc_cuda_native(rebuild = FALSE)

cpu <- fast_kpc(
  data,
  alpha = 0.18,
  max_conditioning_size = 1L,
  engine = "cpu",
  graph_stage = "wanpdag",
  residual_backend = "fastSpline",
  scheduler = "legacy",
  fastspline_params = params,
  ci_method = "dcc.perm",
  permutation_params = permutation_params
)
a <- fast_kpc(
  data,
  alpha = 0.18,
  max_conditioning_size = 1L,
  engine = "cuda",
  graph_stage = "wanpdag",
  residual_backend = "fastSpline",
  residual_device = "cuda",
  orientation_residual_device = "cuda",
  scheduler = "legacy",
  fastspline_params = params,
  ci_method = "dcc.perm",
  permutation_params = permutation_params
)
b <- fast_kpc(
  data,
  alpha = 0.18,
  max_conditioning_size = 1L,
  engine = "cuda",
  graph_stage = "wanpdag",
  residual_backend = "fastSpline",
  residual_device = "cuda",
  orientation_residual_device = "cuda",
  scheduler = "legacy",
  fastspline_params = params,
  ci_method = "dcc.perm",
  permutation_params = permutation_params
)

diag <- a$orientation$diagnostics
assert_true(a$skeleton$ci_backend == "cuda-dcov" &&
              a$orientation$ci_method == "dcc.perm",
            "WAN-PDAG should carry dcc.perm through both stages")
assert_true(a$skeleton$ci_diagnostics$ci_dcc_perm_cuda_tests > 0L,
            "WAN-PDAG skeleton should execute CUDA dcc.perm tests")
assert_true(diag$regrvonps_calls > 0L && diag$regrvonps_cuda_calls > 0L,
            "WAN-PDAG fixture should exercise CUDA generalized orientation")
assert_true(diag$regrvonps_dcc_perm_cuda_tests > 0L,
            "WAN-PDAG orientation should execute CUDA dcc.perm tests")
assert_true(diag$regrvonps_dcc_permutation_replicates ==
              diag$regrvonps_dcc_perm_cuda_tests * 19L,
            "WAN-PDAG should account for every orientation replicate")
assert_true(identical(a$skeleton$adjacency, b$skeleton$adjacency) &&
              identical(a$orientation$pdag, b$orientation$pdag),
            "WAN-PDAG CUDA dcc.perm fixed seed should repeat exactly")
assert_true(identical(cpu$skeleton$adjacency, a$skeleton$adjacency) &&
              max_abs_diff(cpu$skeleton$pMax, a$skeleton$pMax) == 0,
            "CPU and CUDA WAN-PDAG dcc.perm skeletons should match")
assert_true(identical(cpu$orientation$pdag, a$orientation$pdag) &&
              identical(cpu$orientation$counts, a$orientation$counts),
            "CPU and CUDA WAN-PDAG dcc.perm orientation should match")

cat("test_dcc_permutation_wanpdag.R: PASS\n")
