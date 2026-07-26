source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP reproducible CUDA native build gate\n")
  quit(save = "no", status = 0L)
}

first_path <- build_fastkpc_cuda_native(rebuild = TRUE)
first_sha256 <- .fastkpc_cuda_sha256_file(first_path)

second_path <- build_fastkpc_cuda_native(rebuild = TRUE)
second_sha256 <- .fastkpc_cuda_sha256_file(second_path)

assert_true(
  identical(first_sha256, second_sha256),
  paste0(
    "consecutive CUDA native builds must be byte-identical; first=",
    first_sha256, "; second=", second_sha256
  )
)

cat("PASS reproducible CUDA native build gate\n")
cat("native_sha256=", second_sha256, "\n", sep = "")
