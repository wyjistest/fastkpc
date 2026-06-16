source("fastkpc/R/hsic_validation.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

gamma <- validate_hsic_native_gamma(seed = 221, n = 44, sig = 1)
assert_true(is.finite(gamma$metrics$native_p_value),
            "HSIC gamma validation should return finite native p-value")

perm <- validate_hsic_native_permutation(seed = 222, n = 40,
                                         replicates = 20L)
assert_true(perm$metrics$replicates_identical,
            "HSIC permutation validation should record deterministic replicates")

resolution <- compare_hsic_cpu_cuda_resolution(seed = 224, n = 44)
assert_true(resolution$metrics$adjacency_identical,
            "HSIC CPU/CUDA resolution comparison should match adjacency")

bench <- benchmark_hsic_backends(seed = 225, n = 48, repeats = 1)
assert_true(all(c("hsic.gamma", "hsic.perm") %in% bench$timings$ci_method),
            "HSIC benchmark should include gamma and permutation timings")
assert_true(is.data.frame(bench$summary),
            "HSIC benchmark should include summary data frame")

cat("test_hsic_benchmark.R: PASS\n")
