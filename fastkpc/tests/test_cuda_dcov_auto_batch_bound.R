source("fastkpc/R/cuda_native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

build_fastkpc_cuda_native(rebuild = TRUE)

set.seed(812)
n <- 12
p <- 100
data <- matrix(rnorm(n * p), n, p)
colnames(data) <- paste0("V", seq_len(p))

auto <- fast_skeleton_cuda_backend(
  data,
  alpha = 0.01,
  max_conditioning_size = 0,
  residual_backend = "linear",
  residual_device = "cpu",
  scheduler = "layer",
  batch_size = 0,
  residual_cache = TRUE
)

used <- auto$scheduler_diagnostics$summary$dcov_batch_size_used
assert_true(used > 0L, "auto dCov batch size should be recorded")
assert_true(used <= 512L,
            "auto dCov batch size should be bounded for large task layers")
assert_true(auto$scheduler_diagnostics$summary$dcov_batches >= 2L,
            "large task layers should be split into multiple dCov batches")

cat("test_cuda_dcov_auto_batch_bound.R: PASS\n")
