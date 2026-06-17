fastkpc_legacy_env <- function(path = "kpcalg/R") {
  if (!dir.exists(path)) {
    stop("Cannot find legacy kpcalg R directory: ", path, call. = FALSE)
  }

  env <- new.env(parent = globalenv())

  if (requireNamespace("energy", quietly = TRUE)) env$dcov.test <- energy::dcov.test
  if (requireNamespace("RSpectra", quietly = TRUE)) env$eigs <- RSpectra::eigs
  if (requireNamespace("kernlab", quietly = TRUE)) {
    env$inchol <- kernlab::inchol
    env$rbfdot <- kernlab::rbfdot
  }
  if (requireNamespace("mgcv", quietly = TRUE)) env$gam <- mgcv::gam
  if (requireNamespace("graph", quietly = TRUE)) env$numEdges <- graph::numEdges
  if (requireNamespace("pcalg", quietly = TRUE)) env$triple2numb <- pcalg::triple2numb
  env$as <- methods::as
  env$combn <- utils::combn
  env$makeCluster <- parallel::makeCluster
  env$clusterEvalQ <- parallel::clusterEvalQ
  env$parLapply <- parallel::parLapply
  env$stopCluster <- parallel::stopCluster

  files <- c(
    "dcovgamma.R",
    "frmladditivesmooth.R",
    "frmlfullsmooth.R",
    "hsicgamma.R",
    "hsicperm.R",
    "hsicclust.R",
    "hsictest.R",
    "regrXonS.R",
    "kernelCItest.R",
    "regrvonps.R",
    "udag2wanpdag.R",
    "kpc.R"
  )

  for (file in file.path(path, files)) {
    if (file.exists(file)) sys.source(file, envir = env)
  }

  env
}

fastkpc_require_legacy_packages <- function(packages) {
  missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    stop("Missing legacy kpcalg dependency: ", paste(missing, collapse = ", "),
         call. = FALSE)
  }
  invisible(TRUE)
}

fastkpc_legacy_packages_for_method <- function(ic.method, conditional = FALSE) {
  packages <- character()
  if (ic.method == "dcc.gamma") packages <- c(packages, "RSpectra")
  if (ic.method == "dcc.perm") packages <- c(packages, "energy")
  if (ic.method %in% c("hsic.gamma", "hsic.perm", "hsic.clust")) {
    packages <- c(packages, "kernlab")
  }
  if (ic.method == "hsic.clust") packages <- c(packages, "parallel")
  if (conditional && ic.method != "hsic.clust") packages <- c(packages, "mgcv")
  unique(packages)
}

fastkpc_legacy_dcov_gamma <- function(x, y, index = 1, numCol = NULL,
                                      env = fastkpc_legacy_env()) {
  fastkpc_require_legacy_packages("RSpectra")
  if (is.null(numCol)) numCol <- floor(length(x) / 10)
  env$dcov.gamma(x = x, y = y, index = index, numCol = numCol)
}

fastkpc_legacy_kernel_ci <- function(data, x, y, S = integer(),
                                     ic.method = "dcc.gamma",
                                     index = 1,
                                     numCol = floor(nrow(data) / 10),
                                     env = fastkpc_legacy_env(),
                                     ...) {
  fastkpc_require_legacy_packages(
    fastkpc_legacy_packages_for_method(ic.method, conditional = length(S) > 0)
  )
  suffStat <- list(
    data = as.matrix(data),
    ic.method = ic.method,
    index = index,
    numCol = numCol
  )
  env$kernelCItest(x = x, y = y, S = S, suffStat = suffStat, ...)
}

fastkpc_legacy_skeleton <- function(data, alpha, max_conditioning_size,
                                    method = "stable",
                                    ic.method = "dcc.gamma",
                                    index = 1,
                                    numCol = floor(nrow(data) / 10),
                                    env = fastkpc_legacy_env(),
                                    ...) {
  if (!requireNamespace("pcalg", quietly = TRUE)) {
    stop("pcalg is required for legacy skeleton baselines", call. = FALSE)
  }
  fastkpc_require_legacy_packages(
    fastkpc_legacy_packages_for_method(ic.method, conditional = max_conditioning_size > 0)
  )
  data <- as.matrix(data)
  labels <- colnames(data)
  if (is.null(labels)) labels <- paste0("V", seq_len(ncol(data)))
  suffStat <- list(
    data = data,
    ic.method = ic.method,
    index = index,
    numCol = numCol
  )
  pcalg::skeleton(
    suffStat = suffStat,
    indepTest = env$kernelCItest,
    alpha = alpha,
    labels = labels,
    m.max = max_conditioning_size,
    method = method,
    ...
  )
}

fastkpc_fixed_scenario <- function(seed = 4, n = 80) {
  set.seed(seed)
  z <- stats::runif(n)
  data <- cbind(
    x1 = z + stats::rnorm(n, sd = 0.2),
    x2 = z^2 + stats::rnorm(n, sd = 0.2),
    x3 = z,
    x4 = stats::rnorm(n)
  )
  list(
    data = data,
    alpha = 0.2,
    max_conditioning_size = 1L,
    description = "Fixed four-variable nonlinear scenario used by fastkpc MVP tests"
  )
}
