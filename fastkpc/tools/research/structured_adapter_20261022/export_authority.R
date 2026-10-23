# Read the archived oracle only. No CPU numerical solve or fresh oracle run.
args<-commandArgs(TRUE); stopifnot(length(args)==1L)
root<-"fastkpc/artifacts/full_cuda_ci/level_component_prefill_canonical_20261021"
m<-jsonlite::read_json(file.path(root,"manifest.json"))
p<-"fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2/logical_ci_trace.rds"
stopifnot(identical(digest::digest(file=p,algo="sha256",serialize=FALSE),m$inputs[[p]]))
x<-readRDS(p); options(digits=17)
write.table(x[,c("S_key","x","y","p_value")],args[[1]],sep="\t",quote=FALSE,row.names=FALSE)
