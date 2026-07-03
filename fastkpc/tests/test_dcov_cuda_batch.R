source("fastkpc/R/dcov_exact.R")
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

check_column_matches_cpu <- function(x, y, index = 1, legacy_index = TRUE, tol = 1e-10) {
  cuda <- fast_dcov_batch_cuda(x, y, index = index, legacy_index = legacy_index)
  xmat <- if (is.matrix(x)) x else matrix(as.numeric(x), ncol = 1)
  ymat <- if (is.matrix(y)) y else matrix(as.numeric(y), ncol = 1)
  for (k in seq_len(ncol(xmat))) {
    cpu <- dcov_gamma_exact(xmat[, k], ymat[, k], index = index,
                            legacy_index = legacy_index)
    assert_true(abs(cuda$p.value[k] - cpu$p.value) < tol,
                sprintf("p.value mismatch for column %d", k))
    assert_true(abs(cuda$nV2[k] - unname(cpu$statistic)) < tol,
                sprintf("nV2 mismatch for column %d", k))
    assert_true(abs(cuda$mean[k] - unname(cpu$estimates[2])) < tol,
                sprintf("mean mismatch for column %d", k))
    assert_true(abs(cuda$variance[k] - unname(cpu$estimates[3])) < tol,
                sprintf("variance mismatch for column %d", k))
  }
}

build_fastkpc_cuda_native(rebuild = TRUE)

set.seed(21)
x <- rnorm(80)
y <- 0.2 * x + rnorm(80)
check_column_matches_cpu(x, y)

set.seed(22)
xb <- matrix(rnorm(120 * 7), 120, 7)
yb <- xb * rep(seq(0.05, 0.35, length.out = 7), each = 120) +
  matrix(rnorm(120 * 7), 120, 7)
check_column_matches_cpu(xb, yb)

set.seed(23)
large_x <- matrix(rnorm(300 * 64), 300, 64)
large_y <- matrix(rnorm(300 * 64), 300, 64)
large <- fast_dcov_batch_cuda(large_x, large_y)
assert_true(length(large$p.value) == 64, "batch output should have 64 p-values")
assert_true(all(is.finite(large$p.value)), "batch p-values should be finite")
assert_true(all(large$p.value >= 0 & large$p.value <= 1),
            "batch p-values should be in [0, 1]")
assert_true(identical(as.integer(large$diagnostics$workspace_reuse_count), 0L),
            "direct dCov batch API should use the no-workspace overload")
assert_true(identical(as.integer(large$diagnostics$workspace_grow_count), 0L),
            "direct dCov batch API should not report workspace growth")
assert_true(as.integer(large$diagnostics$raw_aggregate_fused_count) > 0L,
            "direct dCov batch API should use fused raw aggregate rowsum")
assert_true(as.integer(large$diagnostics$rowsum_kernel_launch_count) > 0L,
            "direct dCov batch API should count rowsum launches")
assert_true(as.integer(large$diagnostics$rowsum_chunk_count) > 0L,
            "direct dCov batch API should count rowsum chunks")
assert_true(as.numeric(large$diagnostics$rowsum_total_blocks) > 0,
            "direct dCov batch API should count rowsum blocks")
assert_true(as.numeric(large$diagnostics$rowsum_pair_count) > 0,
            "direct dCov batch API should count rowsum pair work")
assert_true(as.integer(large$diagnostics$rowsum_abs_fast_count) > 0L,
            "default dCov rowsum should use the abs fast path")
assert_true(identical(as.integer(large$diagnostics$rowsum_pow_generic_count),
                      0L),
            "default dCov rowsum should avoid the generic pow path")
assert_true(as.numeric(large$diagnostics$rowsum_abs_pair_count) > 0,
            "default dCov rowsum should count abs fast pair work")
assert_true(identical(as.numeric(large$diagnostics$rowsum_generic_pair_count),
                      0),
            "default dCov rowsum should avoid generic pair work")
assert_true(as.integer(large$diagnostics$rowsum_threads) > 0L,
            "direct dCov batch API should report rowsum threads")
assert_true(identical(as.integer(large$diagnostics$rowsum_threads), 64L),
            "default abs-fast dCov rowsum should use tuned 64-thread blocks")
assert_true(as.integer(large$diagnostics$rowsum_n_max) == nrow(large_x),
            "direct dCov batch API should report rowsum n")
assert_true(as.integer(large$diagnostics$rowsum_batch_total) == ncol(large_x),
            "direct dCov batch API should report rowsum batch")
assert_true(as.integer(large$diagnostics$rowsum_max_chunk_batch) > 0L,
            "direct dCov batch API should report max rowsum chunk batch")
assert_true(as.numeric(large$diagnostics$rowsum_max_chunk_sec) >= 0,
            "direct dCov batch API should report max rowsum chunk time")
assert_true(as.integer(large$diagnostics$rowsum_max_chunk_n) == nrow(large_x),
            "direct dCov batch API should report max rowsum chunk n")
old_rowsum_block <- Sys.getenv("FASTKPC_DCOV_ROWSUM_BLOCK", unset = NA)
on.exit({
  if (is.na(old_rowsum_block)) {
    Sys.unsetenv("FASTKPC_DCOV_ROWSUM_BLOCK")
  } else {
    Sys.setenv(FASTKPC_DCOV_ROWSUM_BLOCK = old_rowsum_block)
  }
}, add = TRUE)
set.seed(2300)
block_x <- matrix(rnorm(160 * 4), 160, 4)
block_y <- matrix(rnorm(160 * 4), 160, 4)
Sys.setenv(FASTKPC_DCOV_ROWSUM_BLOCK = "64")
block64 <- fast_dcov_batch_cuda(block_x, block_y)
assert_true(identical(as.integer(block64$diagnostics$rowsum_threads), 64L),
            "dCov rowsum block override should select 64 threads")
assert_true(as.integer(block64$diagnostics$rowsum_abs_fast_count) > 0L,
            "64-thread dCov rowsum override should use abs fast path")
Sys.setenv(FASTKPC_DCOV_ROWSUM_BLOCK = "128")
block128 <- fast_dcov_batch_cuda(block_x, block_y)
assert_true(identical(as.integer(block128$diagnostics$rowsum_threads), 128L),
            "dCov rowsum block override should select 128 threads")
assert_true(as.integer(block128$diagnostics$rowsum_abs_fast_count) > 0L,
            "128-thread dCov rowsum override should use abs fast path")
assert_true(as.integer(large$diagnostics$row_product_reduce_count) > 0L,
            "direct dCov batch API should use row-product reduce")
assert_true(identical(as.integer(large$diagnostics$pvalue_only_count), 0L),
            "direct dCov batch API should not use pvalue-only output")
assert_true(as.integer(large$diagnostics$full_result_materialize_count) > 0L,
            "direct dCov batch API should materialize the full result")
assert_true(as.integer(large$diagnostics$grid_limit_query_count) +
              as.integer(large$diagnostics$grid_limit_process_cache_hit_count) >= 1L,
            "direct dCov batch API should report grid-limit lookup accounting")
assert_true(as.numeric(large$diagnostics$top_level_wall_sec) > 0,
            "direct dCov batch API should report top-level wall time")
large_warm <- fast_dcov_batch_cuda(large_x, large_y)
assert_true(identical(as.integer(large_warm$diagnostics$grid_limit_query_count), 0L),
            "warm direct dCov batch API should use process grid-limit cache")
assert_true(as.integer(large_warm$diagnostics$grid_limit_process_cache_hit_count) >= 1L,
            "warm direct dCov batch API should report process grid-limit cache hit")

set.seed(2301)
wide_batch <- 70000L
wide_x <- matrix(rnorm(12 * wide_batch), 12, wide_batch)
wide_y <- matrix(rnorm(12 * wide_batch), 12, wide_batch)
wide <- fast_dcov_batch_cuda(wide_x, wide_y)
assert_true(length(wide$p.value) == wide_batch,
            "wide batch output should keep all p-values")
assert_true(all(is.finite(wide$p.value)),
            "wide batch p-values should be finite")
for (k in c(1L, 35000L, wide_batch)) {
  cpu <- dcov_gamma_exact(wide_x[, k], wide_y[, k])
  assert_true(abs(wide$p.value[k] - cpu$p.value) < 1e-10,
              sprintf("wide batch p.value mismatch for column %d", k))
}

set.seed(24)
xi <- matrix(rnorm(90 * 3), 90, 3)
yi <- matrix(rnorm(90 * 3), 90, 3)
legacy <- fast_dcov_batch_cuda(xi, yi, index = 1.5, legacy_index = TRUE)
semantic <- fast_dcov_batch_cuda(xi, yi, index = 1.5, legacy_index = FALSE)
assert_true(max(abs(legacy$nV2 - semantic$nV2)) > 1e-8,
            "legacy_index should change nV2 when index != 1")
assert_true(as.integer(legacy$diagnostics$rowsum_abs_fast_count) > 0L,
            "legacy index mode should keep the abs fast rowsum path")
assert_true(identical(as.integer(legacy$diagnostics$rowsum_pow_generic_count),
                      0L),
            "legacy index mode should avoid generic pow rowsum")
assert_true(as.integer(semantic$diagnostics$rowsum_pow_generic_count) > 0L,
            "semantic index != 1 should use generic pow rowsum")
assert_true(as.numeric(semantic$diagnostics$rowsum_generic_pair_count) > 0,
            "semantic index != 1 should count generic pair work")
assert_true(identical(as.integer(semantic$diagnostics$rowsum_abs_fast_count),
                      0L),
            "semantic index != 1 should not use abs fast rowsum")
assert_true(identical(as.integer(semantic$diagnostics$rowsum_threads), 256L),
            "semantic index != 1 should keep the generic 256-thread rowsum")
check_column_matches_cpu(xi, yi, index = 1.5, legacy_index = FALSE)

assert_error(fast_dcov_batch_cuda(1:5, 1:5), "gamma approximation requires n > 5")
bad <- matrix(rnorm(80), 40, 2)
bad[3, 1] <- Inf
assert_error(fast_dcov_batch_cuda(bad, bad), "Data contains missing or infinite values")

cat("test_dcov_cuda_batch.R: PASS\n")
