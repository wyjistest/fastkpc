source("fastkpc/R/dcov_exact.R")
source("fastkpc/R/native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

set_key <- function(values) paste(sort(as.integer(values)), collapse = ",")

combinations <- function(values, k) {
  values <- as.integer(values)
  if (k == 0) return(list(integer(0)))
  if (length(values) < k) return(list())
  raw <- utils::combn(seq_along(values), k, simplify = FALSE)
  lapply(raw, function(i) as.integer(values[i]))
}

r_exact_ci <- function(data, x, y, S, alpha, index = 1, legacy_index = TRUE) {
  if (length(S) == 0) {
    return(dcov_gamma_exact(data[, x], data[, y], index, legacy_index)$p.value)
  }
  residuals <- lm(data[, c(x, y)] ~ as.matrix(data[, S, drop = FALSE]))$residuals
  dcov_gamma_exact(residuals[, 1], residuals[, 2], index, legacy_index)$p.value
}

r_skeleton_reference <- function(data, alpha, max_conditioning_size,
                                 index = 1, legacy_index = TRUE) {
  p <- ncol(data)
  adj <- matrix(TRUE, p, p)
  diag(adj) <- FALSE
  pmax <- matrix(-Inf, p, p)
  diag(pmax) <- 1
  sepsets <- vector("list", p)
  for (i in seq_len(p)) sepsets[[i]] <- vector("list", p)
  n_edge_tests <- integer()
  per_level_log <- list()

  for (ord in 0:max_conditioning_size) {
    snapshot <- adj
    deletions <- matrix(FALSE, p, p)
    level_tests <- 0L
    level_log <- list()
    for (x in seq_len(p - 1)) {
      for (y in (x + 1):p) {
      if (!snapshot[x, y]) next
      edge_done <- FALSE
      neighbors_x <- setdiff(which(snapshot[, x]), y)
      for (S in combinations(neighbors_x, ord)) {
        level_tests <- level_tests + 1L
        pval <- r_exact_ci(data, x, y, S, alpha, index, legacy_index)
        pmax[x, y] <- max(pmax[x, y], pval)
        pmax[y, x] <- pmax[x, y]
        if (pval >= alpha) {
          deletions[x, y] <- TRUE
          deletions[y, x] <- TRUE
          sepsets[[x]][[y]] <- S
          sepsets[[y]][[x]] <- S
          level_log[[length(level_log) + 1L]] <- list(x = x, y = y, S = S, p.value = pval)
          edge_done <- TRUE
          break
        }
      }
      if (edge_done) next
      neighbors_y <- setdiff(which(snapshot[, y]), x)
      for (S in combinations(neighbors_y, ord)) {
        level_tests <- level_tests + 1L
        pval <- r_exact_ci(data, y, x, S, alpha, index, legacy_index)
        pmax[x, y] <- max(pmax[x, y], pval)
        pmax[y, x] <- pmax[x, y]
        if (pval >= alpha) {
          deletions[x, y] <- TRUE
          deletions[y, x] <- TRUE
          sepsets[[x]][[y]] <- S
          sepsets[[y]][[x]] <- S
          level_log[[length(level_log) + 1L]] <- list(x = x, y = y, S = S, p.value = pval)
          break
        }
      }
      }
    }
    adj[deletions] <- FALSE
    n_edge_tests <- c(n_edge_tests, level_tests)
    per_level_log[[length(per_level_log) + 1L]] <- level_log
  }

  list(adjacency = adj, sepsets = sepsets, pMax = pmax,
       n.edgetests = n_edge_tests, per.level.log = per_level_log)
}

set.seed(4)
n <- 80
z <- runif(n)
data <- cbind(
  x1 = z + rnorm(n, sd = 0.2),
  x2 = z^2 + rnorm(n, sd = 0.2),
  x3 = z,
  x4 = rnorm(n)
)

cpp_p <- fast_dcov_exact_cpp(data[, 1], data[, 2])
r_p <- dcov_gamma_exact(data[, 1], data[, 2])$p.value
assert_true(abs(cpp_p - r_p) < 1e-10, "C++ dCov p-value should match R exact")

cpp <- fast_skeleton_cpp(data, alpha = 0.2, max_conditioning_size = 1)
ref <- r_skeleton_reference(data, alpha = 0.2, max_conditioning_size = 1)

assert_true(is.matrix(cpp$adjacency), "adjacency should be a matrix")
assert_true(identical(dim(cpp$adjacency), c(ncol(data), ncol(data))),
            "adjacency dimensions should match variable count")
assert_true(identical(cpp$adjacency, t(cpp$adjacency)),
            "adjacency should be symmetric")
assert_true(all(diag(cpp$adjacency) == FALSE), "adjacency diagonal should be FALSE")

assert_true(max(abs(cpp$pMax - t(cpp$pMax))) < 1e-12, "pMax should be symmetric")
assert_true(all(diag(cpp$pMax) == 1), "pMax diagonal should be 1")
assert_true(length(cpp$n.edgetests) == 2, "n.edgetests should have one count per level")

assert_true(identical(cpp$adjacency, ref$adjacency),
            "C++ skeleton adjacency should match R exact reference")
assert_true(max(abs(cpp$pMax - ref$pMax)) < 1e-10,
            "C++ skeleton pMax should match R exact reference")
assert_true(identical(cpp$n.edgetests, ref$n.edgetests),
            "C++ skeleton n.edgetests should match R exact reference")

for (i in seq_along(ref$sepsets)) {
  for (j in seq_along(ref$sepsets[[i]])) {
    assert_true(identical(set_key(cpp$sepsets[[i]][[j]]), set_key(ref$sepsets[[i]][[j]])),
                sprintf("sepset mismatch for pair %d,%d", i, j))
  }
}

cat("test_skeleton_mvp.R: PASS\n")
