# Benchmark: original dcov.gamma vs exact CPU direct vs GPU
suppressMessages(library(RSpectra))
eigs <- RSpectra::eigs
source("/data/wenyujianData/kpcalg/kpcalg/R/dcovgamma.R")
source("/data/wenyujianData/kpcalg/gpu-dcov/dcov_gamma_gpu.R")

dcov.gamma.exact <- function(x, y) {
  n <- length(x)
  K <- as.matrix(dist(x)); L <- as.matrix(dist(y))
  ctr <- function(M) {
    rm <- rowMeans(M)
    M - outer(rm, rep(1, n)) - outer(rep(1, n), rm) + mean(M)
  }
  A <- ctr(K); B <- ctr(L)
  nV2 <- sum(A * B) / n
  nV2Mean <- mean(K) * mean(L)
  nV2Var <- 2*(n-4)*(n-5)/n/(n-1)/(n-2)/(n-3) * sum(A*A) * sum(B*B) / n^2
  pgamma(nV2, shape = nV2Mean^2/nV2Var, rate = nV2Mean/nV2Var, lower.tail = FALSE)
}

tmin <- function(f, reps = 3) {
  ts <- replicate(reps, system.time(f())[["elapsed"]])
  min(ts)
}

t_warm <- system.time(dcov.gpu.warmup())[["elapsed"]]
cat(sprintf("GPU context warmup (once per R session): %.2f s\n\n", t_warm))

cat(sprintf("%-8s %12s %12s %12s | %9s %9s\n",
            "n", "orig(s)", "cpu.exact(s)", "gpu(s)", "gpu.vs.orig", "gpu.vs.cpu"))
set.seed(123)
for (n in c(1000, 3000, 10000, 30000, 100000)) {
  x <- rnorm(n); y <- 0.3 * x^2 + rnorm(n)
  t_gpu <- tmin(function() dcov.gamma.gpu(x, y), reps = if (n <= 30000) 3 else 2)
  t_orig <- if (n <= 10000) tmin(function() dcov.gamma(x, y, numCol = 100), reps = 1) else NA
  t_cpu  <- if (n <= 10000) tmin(function() dcov.gamma.exact(x, y), reps = 1) else NA
  cat(sprintf("%-8d %12s %12s %12.4f | %9s %9s\n", n,
              ifelse(is.na(t_orig), "-", sprintf("%.3f", t_orig)),
              ifelse(is.na(t_cpu),  "-", sprintf("%.3f", t_cpu)),
              t_gpu,
              ifelse(is.na(t_orig), "-", sprintf("%.0fx", t_orig / t_gpu)),
              ifelse(is.na(t_cpu),  "-", sprintf("%.0fx", t_cpu / t_gpu))))
}
