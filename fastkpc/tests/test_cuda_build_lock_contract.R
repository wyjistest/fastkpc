script <- readLines("fastkpc/tools/build_cuda_native.sh", warn = FALSE)
text <- paste(script, collapse = "\n")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

assert_true(grepl("fastkpc_cuda.lock", text, fixed = TRUE),
            "CUDA build script should use a shared lock path")
assert_true(grepl("flock", text, fixed = TRUE),
            "CUDA build script should serialize rebuilds with flock")
assert_true(grepl('9>"$LOCK"', text, fixed = TRUE),
            "CUDA build script should hold the lock for the build subshell")
assert_true(grepl("fastkpc_cuda.so.tmp.", text, fixed = TRUE),
            "CUDA build script should link to a process-local temporary shared object")
assert_true(grepl("mv -f \"$TMP_SO\" \"$SO\"", text, fixed = TRUE),
            "CUDA build script should atomically replace the final shared object")

cat("PASS CUDA build lock contract\n")
