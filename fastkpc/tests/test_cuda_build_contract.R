assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

source("fastkpc/R/cuda_native.R")

built <- build_fastkpc_cuda_native(rebuild = TRUE)
assert_true(file.exists(built), "CUDA shared object should exist after build")
assert_true(normalizePath(built) == normalizePath("fastkpc/build/fastkpc_cuda.so"),
            "CUDA build should produce fastkpc/build/fastkpc_cuda.so")

load_fastkpc_cuda_native()
assert_true(fastkpc_cuda_available(), "CUDA should be available")

info <- fastkpc_cuda_device_info()
required <- c("device_id", "name", "compute_capability", "total_global_mem")
assert_true(all(required %in% names(info)), "device info should include required fields")
assert_true(nchar(info$name) > 0, "device name should be non-empty")
assert_true(grepl("^[0-9]+\\.[0-9]+$", info$compute_capability),
            "compute capability should be major.minor")
assert_true(info$total_global_mem > 0, "device memory should be positive")

source("fastkpc/R/native.R")
build_fastkpc_native(rebuild = TRUE)
set.seed(11)
x <- rnorm(30)
y <- rnorm(30)
p <- fast_dcov_exact_cpp(x, y)
assert_true(is.finite(p) && p >= 0 && p <= 1,
            "CPU wrappers should still run after CUDA build")

cat("test_cuda_build_contract.R: PASS\n")
