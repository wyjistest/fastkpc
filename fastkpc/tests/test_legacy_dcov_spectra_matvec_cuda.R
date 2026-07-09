source("fastkpc/R/cuda_native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

assert_error <- function(expr, pattern) {
  msg <- tryCatch({
    force(expr)
    NA_character_
  }, error = function(e) conditionMessage(e))
  if (is.na(msg)) stop("Expected error matching: ", pattern, call. = FALSE)
  if (!grepl(pattern, msg, fixed = TRUE)) {
    stop("Error message did not match. Got: ", msg, call. = FALSE)
  }
}

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP legacy dCov Spectra CUDA matvec: FASTKPC_RUN_CUDA_TESTS != 1\n")
  quit(save = "no", status = 0)
}

build_fastkpc_cuda_native(rebuild = TRUE)
if (!fastkpc_cuda_available()) {
  cat("SKIP legacy dCov Spectra CUDA matvec: CUDA unavailable\n")
  quit(save = "no", status = 0)
}

set.seed(7901)
n <- 37L
rhs_count <- 5L
raw <- matrix(rnorm(n * n), n, n)
matrix_a <- 0.5 * (raw + t(raw))
rhs <- matrix(rnorm(n * rhs_count), n, rhs_count)

gpu <- legacy_dcov_spectra_matvec_cuda(matrix_a, rhs)
cpu <- matrix_a %*% rhs

assert_true(is.matrix(gpu$values), "CUDA matvec should return a matrix")
assert_true(identical(dim(gpu$values), dim(rhs)),
            "CUDA matvec result should preserve rhs dimensions")
assert_true(max(abs(gpu$values - cpu)) < 1e-10,
            "CUDA matvec should match host dense matrix multiplication")
assert_true(identical(gpu$backend, "cuda-dense-sym-matvec"),
            "CUDA matvec should report backend")
assert_true(identical(as.integer(gpu$n), n),
            "CUDA matvec should report matrix dimension")
assert_true(identical(as.integer(gpu$rhs_count), rhs_count),
            "CUDA matvec should report rhs count")
assert_true(identical(as.integer(gpu$kernel_launch_count), rhs_count),
            "CUDA matvec should launch once per rhs in the first prototype")
assert_true(as.numeric(gpu$kernel_ms) >= 0,
            "CUDA matvec should report kernel timing")
assert_true(as.numeric(gpu$total_ms) >= as.numeric(gpu$kernel_ms),
            "CUDA matvec total timing should include kernel timing")

vector_rhs <- rnorm(n)
gpu_vec <- legacy_dcov_spectra_matvec_cuda(matrix_a, vector_rhs)
cpu_vec <- as.numeric(matrix_a %*% vector_rhs)
assert_true(is.numeric(gpu_vec$values),
            "CUDA matvec should preserve vector rhs shape")
assert_true(length(gpu_vec$values) == n,
            "CUDA vector matvec should return n values")
assert_true(max(abs(gpu_vec$values - cpu_vec)) < 1e-10,
            "CUDA vector matvec should match host dense matrix multiplication")

assert_error(
  legacy_dcov_spectra_matvec_cuda(matrix_a[-1L, ], rhs[-1L, ]),
  "matrix must be square"
)
bad_rhs <- rhs[-1L, , drop = FALSE]
assert_error(
  legacy_dcov_spectra_matvec_cuda(matrix_a, bad_rhs),
  "rhs row count must match matrix dimension"
)
bad_matrix <- matrix_a
bad_matrix[1L, 1L] <- Inf
assert_error(
  legacy_dcov_spectra_matvec_cuda(bad_matrix, rhs),
  "Data contains missing or infinite values"
)

cat("PASS legacy dCov Spectra CUDA matvec\n")
