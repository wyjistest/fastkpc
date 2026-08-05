source("fastkpc/R/fast_kpc.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

max_abs_diff <- function(a, b) {
  max(abs(as.numeric(a) - as.numeric(b)))
}

set.seed(322)
n <- 48L
z <- runif(n, -2, 2)
data <- cbind(
  x1 = sin(z) + rnorm(n, sd = 0.07),
  x2 = z + rnorm(n, sd = 0.07),
  x3 = z^2 + rnorm(n, sd = 0.07),
  x4 = rnorm(n)
)

load_fastkpc_cuda_native(rebuild = FALSE)

permutation_params <- list(
  replicates = 19L,
  seed = 404L,
  include_observed = TRUE
)

cpu <- fast_skeleton_cpp_backend(
  data, alpha = 0.2, max_conditioning_size = 1L,
  residual_backend = "linear", residual_cache = TRUE,
  ci_method = "dcc.perm", permutation_params = permutation_params
)
cuda_a <- fast_skeleton_cuda_backend(
  data, alpha = 0.2, max_conditioning_size = 1L,
  residual_backend = "linear", residual_device = "cpu",
  residual_cache = TRUE, batch_size = 8L, scheduler = "legacy",
  ci_method = "dcc.perm", permutation_params = permutation_params
)
cuda_b <- fast_skeleton_cuda_backend(
  data, alpha = 0.2, max_conditioning_size = 1L,
  residual_backend = "linear", residual_device = "cpu",
  residual_cache = TRUE, batch_size = 8L, scheduler = "legacy",
  ci_method = "dcc.perm", permutation_params = permutation_params
)
cuda_layer <- fast_skeleton_cuda_backend(
  data, alpha = 0.2, max_conditioning_size = 1L,
  residual_backend = "linear", residual_device = "cpu",
  residual_cache = TRUE, batch_size = 8L, scheduler = "layer",
  ci_method = "dcc.perm", permutation_params = permutation_params
)

assert_true(cuda_a$ci_method == "dcc.perm",
            "CUDA skeleton should record dcc.perm")
assert_true(cuda_a$ci_backend == "cuda-dcov" &&
              identical(cuda_a$ci_backend_reason, ""),
            "fixed-seed CUDA dcc.perm should execute on cuda-dcov")
assert_true(cuda_a$ci_diagnostics$ci_dcc_perm_cuda_tests > 0L,
            "CUDA dcc.perm should record physical CUDA tests")
assert_true(cuda_a$ci_diagnostics$ci_dcc_permutation_replicates ==
              cuda_a$ci_diagnostics$ci_dcc_perm_cuda_tests * 19L,
            "CUDA dcc.perm should account for every GPU replicate")
assert_true(identical(cuda_a$adjacency, cuda_b$adjacency) &&
              max_abs_diff(cuda_a$pMax, cuda_b$pMax) == 0,
            "CUDA dcc.perm fixed seed should repeat exactly")
assert_true(identical(cpu$adjacency, cuda_a$adjacency),
            "CPU and CUDA dcc.perm skeleton adjacency should match")
assert_true(max_abs_diff(cpu$pMax, cuda_a$pMax) == 0,
            "CPU and CUDA dcc.perm pMax should match")
assert_true(identical(cuda_a$adjacency, cuda_layer$adjacency) &&
              max_abs_diff(cuda_a$pMax, cuda_layer$pMax) == 0,
            "CUDA dcc.perm layer and legacy schedulers should match")

fallback <- fast_skeleton_cuda_backend(
  data, alpha = 0.2, max_conditioning_size = 1L,
  residual_backend = "linear", residual_device = "cpu",
  residual_cache = TRUE, batch_size = 8L, scheduler = "legacy",
  ci_method = "dcc.perm",
  permutation_params = list(replicates = 19L,
                            include_observed = TRUE)
)

assert_true(fallback$ci_backend == "native-cpu",
            "seedless CUDA dcc.perm should fail closed to native CPU")
assert_true(fallback$ci_backend_reason ==
              "CUDA dCov permutation requires explicit seed in this stage",
            "seedless CUDA dcc.perm should expose its fallback reason")
assert_true(fallback$ci_diagnostics$ci_dcc_cuda_fallback_tests > 0L,
            "seedless CUDA dcc.perm should account for fallback tests")

public <- fast_kpc(
  data,
  alpha = 0.2,
  max_conditioning_size = 1L,
  engine = "cuda",
  graph_stage = "skeleton",
  residual_backend = "linear",
  residual_device = "cpu",
  ci_method = "dcc.perm",
  permutation_params = permutation_params
)

assert_true(public$config$ci_method == "dcc.perm" &&
              public$config$ci_backend == "cuda-dcov",
            "fast_kpc should expose CUDA dcc.perm routing")
assert_true(isTRUE(public$config$cuda_dcov_permutation_requested) &&
              isTRUE(public$config$cuda_dcov_permutation_used),
            "fast_kpc should truthfully record CUDA dcc.perm use")

cat("test_dcc_permutation_cuda.R: PASS\n")
