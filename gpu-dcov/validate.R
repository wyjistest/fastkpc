# Validation: GPU implementation vs exact CPU reference vs original kpcalg dcov.gamma
suppressMessages(library(RSpectra))
eigs <- RSpectra::eigs
source("/data/wenyujianData/kpcalg/kpcalg/R/dcovgamma.R")   # original (needs `eigs` in scope)
source("/data/wenyujianData/kpcalg/gpu-dcov/dcov_gamma_gpu.R")

# Exact CPU reference: direct algorithm, no eigendecomposition
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
  nV2Variance <- 2 * (n-4) * (n-5) / n / (n-1) / (n-2) / (n-3) *
    sum(A * A) * sum(B * B) / n^2
  alpha <- nV2Mean^2 / nV2Variance
  beta <- nV2Variance / nV2Mean
  list(nV2 = nV2, mean = nV2Mean, var = nV2Variance,
       p = pgamma(nV2, shape = alpha, rate = 1/beta, lower.tail = FALSE))
}

dcov.gpu.warmup()
relerr <- function(a, b) abs(a - b) / pmax(abs(b), .Machine$double.xmin)

cases <- list()
set.seed(42)
for (n in c(300, 1000, 3000)) {
  x0 <- rnorm(n); y0 <- rnorm(n)                       # H0: independent
  x1 <- rnorm(n); y1 <- 0.3 * x1^2 + rnorm(n, sd = 1)  # H1: dependent
  cases[[paste0("n", n, "_H0")]] <- list(x = x0, y = y0, n = n)
  cases[[paste0("n", n, "_H1")]] <- list(x = x1, y = y1, n = n)
}

cat(sprintf("%-12s %12s %12s %12s | %10s %10s | %10s %10s\n",
            "case", "nV2.gpu", "p.gpu", "p.orig",
            "re.nV2", "re.var", "re.p", "re.p.orig"))
ok <- TRUE
for (nm in names(cases)) {
  cs <- cases[[nm]]
  ex <- dcov.gamma.exact(cs$x, cs$y)
  g  <- dcov.gamma.gpu(cs$x, cs$y)
  o  <- dcov.gamma(cs$x, cs$y, numCol = min(100, cs$n %/% 10))
  re_nv2 <- relerr(unname(g$statistic), ex$nV2)
  re_var <- relerr(unname(g$estimates[3]), ex$var)
  re_p   <- relerr(g$p.value, ex$p)
  re_po  <- relerr(o$p.value, ex$p)   # original's truncation error, for reference
  cat(sprintf("%-12s %12.6g %12.6g %12.6g | %10.2e %10.2e | %10.2e %10.2e\n",
              nm, unname(g$statistic), g$p.value, o$p.value,
              re_nv2, re_var, re_p, re_po))
  if (re_nv2 > 1e-10 || re_var > 1e-10 || re_p > 1e-8) {
    ok <- FALSE
    cat("  ^^ GPU vs exact-CPU mismatch above tolerance!\n")
  }
}

# multivariate input (d=3) — original handles only vectors, compare GPU vs exact via dist()
n <- 500; set.seed(7)
X <- matrix(rnorm(n * 3), n, 3); Y <- matrix(rnorm(n * 3), n, 3)
K <- as.matrix(dist(X)); L <- as.matrix(dist(Y))
rmK <- rowMeans(K); rmL <- rowMeans(L)
A <- K - outer(rmK, rep(1, n)) - outer(rep(1, n), rmK) + mean(K)
B <- L - outer(rmL, rep(1, n)) - outer(rep(1, n), rmL) + mean(L)
s <- dcov.gpu.stats(X, Y)
re <- max(relerr(s[1], sum(A*B)), relerr(s[2], sum(A*A)),
          relerr(s[3], sum(B*B)), relerr(s[4], sum(K)), relerr(s[5], sum(L)))
cat(sprintf("multivariate d=3 (n=500): max relerr of 5 scalars = %.2e\n", re))
if (re > 1e-10) { ok <- FALSE; cat("  ^^ mismatch!\n") }

# index honored: index=1.5 vs CPU reference with K^1.5
Ki <- K^1.5; Li <- L^1.5
rmK <- rowMeans(Ki); rmL <- rowMeans(Li)
A <- Ki - outer(rmK, rep(1, n)) - outer(rep(1, n), rmK) + mean(Ki)
B <- Li - outer(rmL, rep(1, n)) - outer(rep(1, n), rmL) + mean(Li)
s <- dcov.gpu.stats(X, Y, index = 1.5)
re <- max(relerr(s[1], sum(A*B)), relerr(s[2], sum(A*A)), relerr(s[3], sum(B*B)))
cat(sprintf("index=1.5 (n=500, d=3):  max relerr of 3 sums    = %.2e\n", re))
if (re > 1e-10) { ok <- FALSE; cat("  ^^ mismatch!\n") }

cat(if (ok) "\nALL CHECKS PASSED\n" else "\nFAILURES PRESENT\n")
