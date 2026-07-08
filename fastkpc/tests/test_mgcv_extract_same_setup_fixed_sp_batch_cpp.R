source("fastkpc/R/mgcv_extract_oracle.R")
source("fastkpc/R/native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP same-setup fixed-sp cpp batch: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

set.seed(249)
n <- 56
s1 <- stats::runif(n, -2, 2)
Y <- cbind(
  y1 = sin(s1) + stats::rnorm(n, sd = 0.04),
  y2 = cos(s1) + stats::rnorm(n, sd = 0.04),
  y3 = sin(2 * s1) + stats::rnorm(n, sd = 0.04)
)
S_data <- data.frame(s1 = s1)
sp <- c(0.25, 0.9, 1.4)
target_ids <- c(21L, 22L, 23L)

batch <- fastkpc_mgcv_extract_same_setup_batch_fixed_sp_cpp(
  Y = Y,
  S_data = S_data,
  S = 1L,
  sp = sp,
  k = 8L,
  bs = "tp",
  target_ids = target_ids
)

single_setups <- lapply(seq_len(ncol(Y)), function(j) {
  fastkpc_mgcv_extract_setup(
    formula = y ~ s(s1, k = 8, bs = "tp"),
    data = data.frame(y = Y[, j], S_data),
    sp = sp[j],
    target = target_ids[j],
    S = 1L,
    k = 8L,
    bs = "tp"
  )
})
single <- lapply(single_setups, fastkpc_mgcv_solve_setup_fixed_sp_cpp)

assert_true(identical(batch$backend_family, "mgcvExtractCPU"),
            "batch backend should identify mgcvExtractCPU")
assert_true(identical(batch$mode, "fixed-sp-same-setup-native-cpp-batch-prototype"),
            "batch mode should identify same-setup cpp prototype")
assert_true(identical(batch$solve_source,
                      "fastkpc-native-same-setup-fixed-sp-batch-prototype"),
            "batch solve source should be explicit")
assert_true(identical(batch$used_device, "cpu"),
            "same-setup cpp batch should report cpu")
assert_true(isTRUE(batch$native_cpp_solve_used),
            "same-setup batch should report native C++ usage")
assert_true(isFALSE(batch$is_self_contained_gcv),
            "same-setup fixed-sp batch must not claim self-contained GCV")
assert_true(all(batch$target_ids == target_ids),
            "same-setup batch preserves target ids")
assert_true(all(abs(batch$sp - sp) < 1e-12),
            "same-setup batch preserves per-target fixed sp")
assert_true(all(dim(batch$residuals) == c(n, ncol(Y))),
            "same-setup residual dimensions")
assert_true(all(dim(batch$fitted) == c(n, ncol(Y))),
            "same-setup fitted dimensions")
for (j in seq_len(ncol(Y))) {
  assert_true(max(abs(batch$residuals[, j] - single[[j]]$residuals)) < 1e-8,
              paste("batch residual column", j, "matches independent fixed-sp solve"))
  assert_true(max(abs(batch$fitted[, j] - single[[j]]$fitted)) < 1e-8,
              paste("batch fitted column", j, "matches independent fixed-sp solve"))
}
assert_true(length(unique(batch$setup_fingerprints)) == 1L,
            "same-setup batch should reuse one setup fingerprint")
assert_true(identical(batch$diagnostics$targets, as.integer(ncol(Y))),
            "diagnostics should record target count")
assert_true(identical(batch$diagnostics$setup_reused, TRUE),
            "diagnostics should record setup reuse")
assert_true(identical(batch$diagnostics$true_batched_kernel, FALSE),
            "diagnostics should avoid claiming a true batched kernel")
assert_true(identical(batch$diagnostics$batch_stage,
                      "same-setup-repeated-cpp-solve-prototype"),
            "diagnostics should identify the prototype batch stage")
assert_true(!is.null(batch$template_setup),
            "batch should expose the reused template setup for diagnostics")

cat("PASS same-setup fixed-sp cpp batch\n")
