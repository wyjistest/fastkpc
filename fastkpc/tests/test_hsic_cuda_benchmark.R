source("fastkpc/R/hsic_cuda_validation.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

gamma <- validate_hsic_cuda_gamma_kernel(seed = 401, n = 64, sig = 1)
assert_true(gamma$metrics$statistic_abs_diff < 1e-8,
            "CUDA HSIC gamma statistic should match CPU")
assert_true(gamma$metrics$pvalue_abs_diff < 1e-7,
            "CUDA HSIC gamma p-value should match CPU")
assert_true(gamma$metrics$ci_backend == "cuda-hsic",
            "CUDA HSIC gamma validation should report cuda-hsic")

perm <- validate_hsic_cuda_permutation_kernel(seed = 402, n = 56,
                                              replicates = 24L)
assert_true(perm$metrics$fixed_seed_repeats,
            "CUDA HSIC permutation validation should repeat fixed seeds")
assert_true(perm$metrics$ci_backend == "cuda-hsic",
            "CUDA HSIC permutation validation should report cuda-hsic")

comparison <- compare_hsic_cuda_cpu_skeleton(seed = 403, n = 64)
assert_true(comparison$metrics$adjacency_identical,
            "CUDA HSIC skeleton should match CPU adjacency")
assert_true(comparison$metrics$max_abs_pmax_diff < 1e-7,
            "CUDA HSIC skeleton pMax should be close to CPU")
assert_true(comparison$metrics$ci_backend == "cuda-hsic",
            "CUDA HSIC skeleton comparison should report cuda-hsic")

bench <- benchmark_hsic_cuda_backends(seed = 404,
                                      n_values = c(48),
                                      methods = c("hsic.gamma", "hsic.perm"),
                                      repeats = 1L)
assert_true(is.data.frame(bench$timings),
            "CUDA HSIC benchmark should return timings data frame")
assert_true(all(c("cpu", "cuda") %in% bench$timings$backend),
            "CUDA HSIC benchmark should include CPU and CUDA rows")
assert_true("speedup" %in% names(bench$summary),
            "CUDA HSIC benchmark summary should include speedup")
assert_true(all(is.finite(bench$summary$speedup)),
            "CUDA HSIC benchmark speedup values should be finite")

cat("test_hsic_cuda_benchmark.R: PASS\n")
