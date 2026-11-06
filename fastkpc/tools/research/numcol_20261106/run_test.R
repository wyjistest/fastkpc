source("fastkpc/tools/research/numcol_20261106/common.R")
args <- commandArgs(TRUE)
stopifnot(length(args) == 2L)
native <- normalizePath(args[[1L]])
.fastkpc_cuda_so <- local({ path <- native; function() path })
stopifnot(fastkpc_cuda_available())
rank_mode("off")
Sys.setenv(FASTKPC_RUN_CUDA_TESTS = "1")
expressions <- parse(args[[2L]])
skip <- vapply(expressions, function(e) {
  identical(e, quote(source("fastkpc/R/fast_kpc.R"))) ||
      identical(e, quote(source("fastkpc/R/cuda_native.R"))) ||
      identical(e, quote(source("fastkpc/tools/research/numcol_20261106/common.R")))
}, logical(1L))
# One eval retains top-level on.exit handlers until the entire test completes.
eval(expressions[!skip], envir = .GlobalEnv)
stopifnot(normalizePath(getLoadedDLLs()[["fastkpc_cuda"]][["path"]]) == native)
