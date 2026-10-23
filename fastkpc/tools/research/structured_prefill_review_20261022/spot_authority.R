# Independent spot check of newly added 22-col authority values using the installed kpcalg package.
suppressMessages(library(kpcalg))
A <- readRDS("fastkpc/artifacts/full_cuda_ci/structured_prefill_20261022/authority.rds")
oracle <- readRDS("fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2/logical_ci_trace.rds")
tr <- A$trace
key <- function(z) paste(pmin(z$x, z$y), pmax(z$x, z$y), z$S_key, sep = "|")
new <- which(is.na(match(key(tr), key(oracle))))
new <- new[!duplicated(key(tr)[new])]
cat("new unique keys:", length(new), "\n")
data <- as.matrix(readRDS("fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"))[, 1:22, drop = FALSE]
suff <- list(data = data, ic.method = "dcc.gamma", index = 1, numCol = 35L)
set.seed(7)
pick <- c(sample(new[tr$level[new] == 2], 2), sample(new[tr$level[new] == 3], 2), sample(new[tr$level[new] >= 4], 2))
for (i in pick) {
  S <- if (nzchar(tr$S_key[i])) as.integer(strsplit(tr$S_key[i], "|", fixed = TRUE)[[1]]) else integer()
  p <- kpcalg::kernelCItest(x = tr$x[i], y = tr$y[i], S = S, suffStat = suff)
  cat(sprintf("level %d x=%2d y=%2d S={%s}  authority %.17g  kpcalg-pkg %.17g  |diff| %.3g\n",
      tr$level[i], tr$x[i], tr$y[i], tr$S_key[i], tr$p_value[i], p, abs(p - tr$p_value[i])))
}
