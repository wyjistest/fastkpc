# Real-data check of the structured 1-D distance-matrix eigen route against
# legacy kpcalg::dcov.gamma.  Components: the 48 raw columns of the canonical
# 351x48 input (many exact ties) plus 60 mgcv GAM residuals with |S| = 1..3.
# Uses the robust formulation U = (H F L_Q) P; dense LAPACK svd/eigen stand in
# for the production bidiagonal QR and secular solver (validation only).
# Run from the repository root:
#   Rscript fastkpc/tools/research/exact_speedup_20261020/verify_structured_dcov_real_data.R
suppressMessages({library(kpcalg); library(RSpectra); library(mgcv)})

dense_topk <- function(x, k) {
  D <- as.matrix(dist(x))
  e <- eigen(D, symmetric = TRUE)
  idx <- order(abs(e$values), decreasing = TRUE)[seq_len(k)]
  list(values = e$values[idx], vectors = e$vectors[, idx, drop = FALSE],
       Dmean = mean(D))
}

structured_topk <- function(x, k) {
  n <- length(x)
  o <- order(x)
  xs <- x[o]
  r <- rle(xs)
  cnt <- r$lengths
  m <- length(r$values)
  stopifnot(m >= 3)
  h <- diff(r$values)
  tt <- cumsum(cnt)
  Ld <- sqrt(tt[-1] / (tt[-m] * cnt[-1]))                       # diag(L_Q)
  Le <- -sqrt(tt[1:(m - 2)] / (tt[2:(m - 1)] * cnt[2:(m - 1)]))  # subdiag(L_Q)
  gi <- 1 / sqrt(h)
  B <- diag(gi * Ld, m - 1)
  B[cbind(2:(m - 1), 1:(m - 2))] <- gi[-1] * Le
  sv <- svd(B)
  a <- -2 / sv$d^2                                  # eigenvalues of HDH
  LP <- Ld * sv$v
  LP[2:(m - 1), ] <- LP[2:(m - 1), ] + Le * sv$v[1:(m - 2), ]
  RC <- rbind(apply(LP, 2, function(col) rev(cumsum(rev(col)))), 0)
  U <- RC[rep(seq_len(m), cnt), , drop = FALSE]
  U <- sweep(U, 2, colMeans(U))                     # U = (H F L_Q) P
  g <- diff(xs)
  kpos <- seq_len(n - 1)
  rs <- c(0, cumsum(kpos * g)) + c(rev(cumsum(rev((n - kpos) * g))), 0)
  alpha <- sum(rs) / n
  bt <- drop(crossprod(U, rs / sqrt(n)))
  Ar <- rbind(c(alpha, bt), cbind(bt, diag(a, m - 1)))
  ea <- eigen(Ar, symmetric = TRUE)
  kk <- min(k, m)
  idx <- order(abs(ea$values), decreasing = TRUE)[seq_len(kk)]
  Vs <- cbind(rep(1 / sqrt(n), n), U) %*% ea$vectors[, idx, drop = FALSE]
  V <- matrix(0, n, k)
  V[o, seq_len(kk)] <- Vs
  list(values = c(ea$values[idx], rep(0, k - kk)), vectors = V,
       Dmean = alpha / n)
}

center_parts <- function(p) {
  list(lambda = p$values, W = sweep(p$vectors, 2, colMeans(p$vectors)),
       Dmean = p$Dmean)
}

pvalue_from_parts <- function(px, py, n) {
  C <- crossprod(px$W, py$W)
  nV2 <- sum(outer(px$lambda, py$lambda) * C^2) / n
  tx <- sum(outer(px$lambda, px$lambda) * crossprod(px$W)^2)
  ty <- sum(outer(py$lambda, py$lambda) * crossprod(py$W)^2)
  mu <- px$Dmean * py$Dmean
  va <- 2 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3) * tx * ty /
    n^4 * n^2
  1 - pgamma(q = nV2, shape = mu^2 / va, rate = 1 / (va / mu))
}

X <- as.matrix(readRDS(
  "fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"))
n <- nrow(X)
k <- 35L
set.seed(20261020)
vars <- lapply(seq_len(ncol(X)), function(j) X[, j])
names(vars) <- paste0("col", seq_len(ncol(X)))
for (r in seq_len(60)) {
  sz <- c(1L, 2L, 3L)[(r - 1) %% 3 + 1]
  idx <- sample(ncol(X), sz + 1L)
  df <- data.frame(y = X[, idx[1]], X[, idx[-1], drop = FALSE])
  names(df) <- c("y", paste0("s", seq_len(sz)))
  f <- if (sz <= 2) {
    as.formula(paste0("y ~ s(", paste(names(df)[-1], collapse = ","), ")"))
  } else {
    as.formula(paste0("y ~ ", paste0("s(", names(df)[-1], ")",
                                     collapse = "+")))
  }
  vars[[paste0("res", r, "_S", sz)]] <- residuals(gam(f, data = df))
}
ties <- sapply(vars[1:48], function(v) length(v) - length(unique(v)))
cat(sprintf("components: %d (48 raw columns, ties per column %d..%d; %d GAM residuals), n=%d k=%d\n",
            length(vars), min(ties), max(ties), length(vars) - 48L, n, k))

dense <- lapply(lapply(vars, dense_topk, k = k), center_parts)
struct <- lapply(lapply(vars, structured_topk, k = k), center_parts)
op_err <- mapply(function(d, s) {
  Hd <- d$W %*% (d$lambda * t(d$W))
  Hs <- s$W %*% (s$lambda * t(s$W))
  norm(Hd - Hs, "F") / norm(Hd, "F")
}, dense, struct)
cat(sprintf("max centered rank-k operator rel err (structured vs dense eigen): %.3e\n",
            max(op_err)))

pairs <- t(combn(length(vars), 2))
pairs <- pairs[sample(nrow(pairs), 3000L), , drop = FALSE]
p_dense <- p_struct <- numeric(nrow(pairs))
for (t in seq_len(nrow(pairs))) {
  i <- pairs[t, 1]; j <- pairs[t, 2]
  p_dense[t] <- pvalue_from_parts(dense[[i]], dense[[j]], n)
  p_struct[t] <- pvalue_from_parts(struct[[i]], struct[[j]], n)
}
chk <- seq_len(300L)
p_kpcalg <- vapply(chk, function(t) {
  kpcalg::dcov.gamma(vars[[pairs[t, 1]]], vars[[pairs[t, 2]]], index = 1,
                     numCol = k)$p.value
}, numeric(1))
cat(sprintf("pairs: %d; kpcalg::dcov.gamma reference on the first %d\n",
            nrow(pairs), length(chk)))
cat(sprintf("max |p_structured - p_dense|   = %.3e\n", max(abs(p_struct - p_dense))))
cat(sprintf("max |p_structured - p_kpcalg|  = %.3e\n",
            max(abs(p_struct[chk] - p_kpcalg))))
cat(sprintf("max |p_dense - p_kpcalg|       = %.3e\n",
            max(abs(p_dense[chk] - p_kpcalg))))
cat(sprintf("alpha=0.1 decisions identical (structured vs kpcalg): %s\n",
            identical(p_struct[chk] >= 0.1, p_kpcalg >= 0.1)))
