source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

assert_true(
  isTRUE(tryCatch(fastkpc_cuda_available(), error = function(error) FALSE)),
  "CUDA native library is unavailable"
)

values <- c(
  "",
  "a",
  "strict-permutation-attestation",
  paste(rep.int("0123456789abcdef", 8192L), collapse = "")
)
for (value in values) {
  expected <- .Call(
    "C_full_cuda_ci_sha256_utf8", value, PACKAGE = "fastkpc_cuda"
  )
  for (chunk_size in c(1L, 3L, 64L, 351L, 4096L, 65536L)) {
    actual <- .Call(
      "C_full_cuda_ci_test_sha256_incremental_utf8",
      value, chunk_size, PACKAGE = "fastkpc_cuda"
    )
    assert_true(
      identical(actual, expected),
      paste("incremental SHA-256 changed at chunk size", chunk_size)
    )
  }
}

bad_chunk_rejected <- inherits(try(
  .Call(
    "C_full_cuda_ci_test_sha256_incremental_utf8",
    "payload", 0L, PACKAGE = "fastkpc_cuda"
  ),
  silent = TRUE
), "try-error")
assert_true(bad_chunk_rejected, "incremental SHA-256 accepted an invalid chunk")

header <- paste(
  readLines("fastkpc/src/cuda/full_cuda_ci_method_batch.hpp", warn = FALSE),
  collapse = "\n"
)
assert_true(
  grepl("SealedPermutationTableHandle(const SealedPermutationTableHandle&) = delete",
        header, fixed = TRUE) &&
    grepl("friend class PermutationTableBuilder", header, fixed = TRUE) &&
    grepl("const int* data() const noexcept", header, fixed = TRUE) &&
    !grepl("int* data() noexcept", header, fixed = TRUE) &&
    !grepl("seal_full_cuda_ci_permutation_table", header, fixed = TRUE),
  "sealed permutation handle copy or mutable-data contract changed"
)

cat("test_full_cuda_ci_sealed_permutation_attestation.R: PASS\n")
