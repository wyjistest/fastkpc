# Run an existing regression on an explicitly identified research native.
args <- commandArgs(TRUE)
stopifnot(length(args)==2L)
source("fastkpc/R/fast_kpc.R")
native <- normalizePath(args[[1L]],mustWork=TRUE)
.fastkpc_cuda_so <- local({path<-native;function()path})
stopifnot(fastkpc_cuda_available()) # CUDA absence is FAIL in this campaign.
expressions <- parse(args[[2L]])
stopifnot(identical(expressions[[1L]],quote(source("fastkpc/R/fast_kpc.R"))))
eval(expressions[-1L],envir=.GlobalEnv)
stopifnot(normalizePath(getLoadedDLLs()[["fastkpc_cuda"]][["path"]])==native)
