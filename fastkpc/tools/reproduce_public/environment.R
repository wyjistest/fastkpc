args <- commandArgs(TRUE)
stopifnot(length(args) == 1L, !file.exists(args[[1L]]))
packages <- c("Rcpp", "RcppArmadillo", "RcppEigen", "RSpectra", "pcalg", "graph",
  "RBGL", "mgcv", "Matrix", "energy", "kernlab", "jsonlite", "digest")
missing <- packages[!vapply(packages, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1L))]
if (length(missing)) stop("Missing R packages: ", paste(missing, collapse = ", "))
jsonlite::write_json(list(R = R.version.string, platform = R.version$platform,
  R_home = R.home(), package_versions = setNames(lapply(packages,
    function(p) as.character(packageVersion(p))), packages),
  session_info = capture.output(sessionInfo()),
  BLAS = unname(extSoftVersion()["BLAS"])), args[[1L]],
  pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat("PASS R dependencies\n")
