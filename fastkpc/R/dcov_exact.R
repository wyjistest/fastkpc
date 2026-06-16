dcov_gamma_exact <- function(x, y, index = 1, legacy_index = TRUE) {
  if (is.matrix(x)) {
    n <- nrow(x)
    x_mat <- x
  } else {
    n <- length(x)
    x_mat <- as.numeric(x)
  }
  if (is.matrix(y)) {
    m <- nrow(y)
    y_mat <- y
  } else {
    m <- length(y)
    y_mat <- as.numeric(y)
  }

  if (index < 0 || index > 2) {
    warning("index must be in [0,2), using default index=1")
    index <- 1
  }
  if (n != m) stop("Sample sizes must agree")
  if (n <= 5) stop("gamma approximation requires n > 5")
  if (!(all(is.finite(c(x_mat, y_mat))))) {
    stop("Data contains missing or infinite values")
  }

  K <- as.matrix(stats::dist(x_mat))
  L <- as.matrix(stats::dist(y_mat))
  if (!isTRUE(legacy_index)) {
    K <- K^index
    L <- L^index
  }

  center_distance <- function(D) {
    row_mean <- rowMeans(D)
    grand_mean <- mean(D)
    sweep(sweep(D, 1, row_mean, "-"), 2, row_mean, "-") + grand_mean
  }

  A <- center_distance(K)
  B <- center_distance(L)

  nV2 <- sum(A * B) / n
  nV2Mean <- mean(K) * mean(L)
  nV2Variance <- 2 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3) *
    sum(A * A) * sum(B * B) / n^2

  alpha <- nV2Mean^2 / nV2Variance
  beta <- nV2Variance / nV2Mean
  pval <- stats::pgamma(q = nV2, shape = alpha, rate = 1 / beta,
                        lower.tail = FALSE)
  dCov <- sqrt(nV2 / n)

  names(dCov) <- "dCov"
  names(nV2) <- "nV^2"
  names(nV2Mean) <- "nV^2 mean"
  names(nV2Variance) <- "nV^2 variance"

  data_name <- if (isTRUE(legacy_index)) {
    sprintf("index %g ignored for legacy compatibility, Gamma approximation", index)
  } else {
    sprintf("index %g, Gamma approximation", index)
  }

  result <- list(
    method = "dCov test of independence (CPU, exact)",
    statistic = nV2,
    estimate = dCov,
    estimates = c(nV2, nV2Mean, nV2Variance),
    p.value = pval,
    replicates = NULL,
    data.name = data_name
  )
  class(result) <- "htest"
  result
}
