# dcov.gamma.gpu — drop-in GPU replacement for kpcalg::dcov.gamma
#
# Computes the exact (full-rank) statistic instead of the original's truncated
# RSpectra::eigs approximation, on GPU with O(n) memory.  Differences from the
# original, all deliberate:
#   * no numCol argument: the eigendecomposition truncation is mathematically
#     unnecessary (exact identity nV^2 = sum(A o B)/n); the original's default
#     numCol = n/10 introduces ~4% statistic error under H0
#   * `index` is honored as documented (the original silently ignores it)
#   * p-value uses pgamma(lower.tail = FALSE), which does not underflow to 0
#     below ~1e-16 the way the original's 1 - pgamma(...) does
# With index = 1 and away from the underflow region, results agree with the
# exact CPU reference to floating-point roundoff.

local({
  ofile <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  dir <- if (is.null(ofile)) "." else dirname(normalizePath(ofile))
  so <- file.path(dir, "dcov_gpu.so")
  if (!file.exists(so)) so <- "dcov_gpu.so"
  dyn.load(so)
})

dcov.gpu.warmup <- function() invisible(.Call("C_dcov_gpu_warmup"))

# Returns the raw scalars; useful for validation.
# c(sum(AoB), sum(AoA), sum(BoB), sum(K), sum(L))
dcov.gpu.stats <- function(x, y, index = 1) {
  x <- if (is.matrix(x)) x else as.numeric(x)
  y <- if (is.matrix(y)) y else as.numeric(y)
  storage.mode(x) <- "double"
  storage.mode(y) <- "double"
  .Call("C_dcov_gpu_stats", x, y, as.numeric(index))
}

dcov.gamma.gpu <- function(x, y, index = 1) {
  n <- if (is.matrix(x)) nrow(x) else length(x)
  m <- if (is.matrix(y)) nrow(y) else length(y)
  if (index < 0 || index > 2) {
    warning("index must be in [0,2), using default index=1")
    index <- 1
  }
  if (n != m) stop("Sample sizes must agree")
  if (!(all(is.finite(c(x, y)))))
    stop("Data contains missing or infinite values")

  s <- dcov.gpu.stats(x, y, index)
  Sab <- s[1]; Saa <- s[2]; Sbb <- s[3]; sumK <- s[4]; sumL <- s[5]

  nV2 <- Sab / n
  nV2Mean <- (sumK / n^2) * (sumL / n^2)
  nV2Variance <- 2 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3) *
    Saa * Sbb / n^2
  alpha <- nV2Mean^2 / nV2Variance
  beta <- nV2Variance / nV2Mean
  pval <- pgamma(q = nV2, shape = alpha, rate = 1 / beta, lower.tail = FALSE)
  dCov <- sqrt(nV2 / n)

  names(dCov) <- "dCov"
  names(nV2) <- "nV^2"
  names(nV2Mean) <- "nV^2 mean"
  names(nV2Variance) <- "nV^2 variance"
  e <- list(method = "dCov test of independence (GPU, exact)",
            statistic = nV2,
            estimate = dCov,
            estimates = c(nV2, nV2Mean, nV2Variance),
            p.value = pval,
            replicates = NULL,
            data.name = sprintf("index %g, Gamma approximation", index))
  class(e) <- "htest"
  e
}
