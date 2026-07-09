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

handle <- legacy_dcov_spectra_matvec_cuda_handle(matrix_a)
on.exit(try(legacy_dcov_spectra_matvec_cuda_handle_free(handle), silent = TRUE),
        add = TRUE)
assert_true(identical(handle$backend, "cuda-dense-sym-matvec-handle"),
            "CUDA matvec handle should report backend")
assert_true(identical(as.integer(handle$n), n),
            "CUDA matvec handle should report matrix dimension")
assert_true(as.numeric(handle$matrix_h2d_ms) >= 0,
            "CUDA matvec handle should report matrix upload timing")
assert_true(as.numeric(handle$matrix_bytes) == n * n * 8,
            "CUDA matvec handle should report matrix bytes")

reuse <- legacy_dcov_spectra_matvec_cuda_handle_apply(handle, rhs)
assert_true(is.matrix(reuse$values), "CUDA handle matvec should return a matrix")
assert_true(max(abs(reuse$values - cpu)) < 1e-10,
            "CUDA handle matvec should match host dense matrix multiplication")
assert_true(identical(reuse$backend, "cuda-dense-sym-matvec-handle"),
            "CUDA handle matvec should report backend")
assert_true(identical(as.integer(reuse$device_matrix_reuse_count), 1L),
            "CUDA handle matvec should reuse the resident device matrix")
assert_true(identical(as.numeric(reuse$matrix_h2d_ms), 0),
            "CUDA handle matvec should not re-upload the matrix during apply")
assert_true(identical(as.integer(reuse$kernel_launch_count), rhs_count),
            "CUDA handle matvec should launch once per rhs in this prototype")

reuse_again <- legacy_dcov_spectra_matvec_cuda_handle_apply(handle, rhs)
assert_true(max(abs(reuse_again$values - cpu)) < 1e-10,
            "CUDA handle repeated matvec should match host dense multiplication")
assert_true(identical(as.integer(reuse_again$device_workspace_reuse_count), 1L),
            "CUDA handle repeated matvec should reuse resident RHS/output workspace")
assert_true(identical(as.integer(reuse_again$workspace_realloc_count), 0L),
            "CUDA handle repeated matvec should not reallocate RHS/output workspace")
assert_true(identical(as.numeric(reuse_again$workspace_alloc_ms), 0),
            "CUDA handle repeated matvec should not spend time allocating reused workspace")
assert_true(as.numeric(reuse_again$workspace_bytes) >= n * rhs_count * 2 * 8,
            "CUDA handle should report resident RHS/output workspace bytes")

wide_rhs <- cbind(rhs, rhs[, 1:2, drop = FALSE])
wide_cpu <- matrix_a %*% wide_rhs
wide_reuse <- legacy_dcov_spectra_matvec_cuda_handle_apply(handle, wide_rhs)
assert_true(max(abs(wide_reuse$values - wide_cpu)) < 1e-10,
            "CUDA handle wider matvec should match host dense multiplication")
assert_true(identical(as.integer(wide_reuse$workspace_realloc_count), 1L),
            "CUDA handle wider matvec should grow RHS/output workspace once")
assert_true(as.numeric(wide_reuse$workspace_bytes) >= n * ncol(wide_rhs) * 2 * 8,
            "CUDA handle should report grown RHS/output workspace bytes")

reuse_vec <- legacy_dcov_spectra_matvec_cuda_handle_apply(handle, vector_rhs)
assert_true(is.numeric(reuse_vec$values),
            "CUDA handle matvec should preserve vector rhs shape")
assert_true(max(abs(reuse_vec$values - cpu_vec)) < 1e-10,
            "CUDA handle vector matvec should match host dense multiplication")
assert_true(identical(as.integer(reuse_vec$device_workspace_reuse_count), 1L),
            "CUDA handle vector matvec should reuse the grown RHS/output workspace")
assert_true(identical(as.integer(reuse_vec$workspace_realloc_count), 0L),
            "CUDA handle vector matvec should not shrink/reallocate workspace")

sequence_rhs <- rhs[, 1:4, drop = FALSE]
sequence_cpu <- matrix_a %*% sequence_rhs
sequence_reuse <- legacy_dcov_spectra_matvec_cuda_handle_apply_sequence(
  handle, sequence_rhs
)
assert_true(is.matrix(sequence_reuse$values),
            "CUDA handle sequence matvec should return a matrix")
assert_true(identical(dim(sequence_reuse$values), dim(sequence_rhs)),
            "CUDA handle sequence matvec should preserve rhs dimensions")
assert_true(max(abs(sequence_reuse$values - sequence_cpu)) < 1e-10,
            "CUDA handle sequence matvec should match host dense multiplication")
assert_true(
  identical(sequence_reuse$backend, "cuda-dense-sym-matvec-handle-sequence"),
  "CUDA handle sequence matvec should report backend"
)
assert_true(identical(as.integer(sequence_reuse$matvec_call_count), 4L),
            "CUDA handle sequence matvec should report one call per rhs column")
assert_true(identical(as.integer(sequence_reuse$kernel_launch_count), 4L),
            "CUDA handle sequence matvec should launch once per rhs column")
assert_true(identical(as.integer(sequence_reuse$device_matrix_reuse_count), 4L),
            "CUDA handle sequence matvec should reuse the device matrix per call")
assert_true(identical(as.integer(sequence_reuse$device_workspace_reuse_count), 4L),
            "CUDA handle sequence matvec should reuse warm device workspace per call")
assert_true(identical(as.integer(sequence_reuse$workspace_realloc_count), 0L),
            "CUDA handle sequence matvec should not reallocate warm workspace")
assert_true(identical(as.numeric(sequence_reuse$matrix_h2d_ms), 0),
            "CUDA handle sequence matvec should not re-upload the matrix")
assert_true(as.numeric(sequence_reuse$workspace_bytes) >= n * ncol(wide_rhs) * 2 * 8,
            "CUDA handle sequence matvec should report retained workspace bytes")

legacy_dcov_spectra_matvec_cuda_handle_free(handle)
assert_error(
  legacy_dcov_spectra_matvec_cuda_handle_apply(handle, rhs),
  "CUDA matvec handle has been freed"
)

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
