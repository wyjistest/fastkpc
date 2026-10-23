# Independent OFFLINE validation authority. Never loaded by a CUDA producer,
# never used to repair/replace a GPU result, and excluded from all timing runs.
# Reuse archived canonical values; evaluate only previously uncovered CI keys.
args<-commandArgs(TRUE);stopifnot(length(args)==1L)
out<-args[[1L]];dir.create(out,recursive=TRUE,showWarnings=FALSE)
stopifnot(!file.exists(file.path(out,"authority.rds")))
source("fastkpc/R/legacy_runner.R")
sha<-function(p)digest::digest(file=p,algo="sha256",serialize=FALSE)
reference_path<-"fastkpc/artifacts/full_cuda_ci/level_component_prefill_canonical_20261021/subset-off/receipt.rds.gz"
oracle_path<-"fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2/logical_ci_trace.rds"
dataset<-"fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"
reference<-readRDS(gzfile(reference_path));tasks<-reference$result$tasks
oracle<-readRDS(oracle_path)
key<-function(z)paste(pmin(z$x,z$y),pmax(z$x,z$y),z$S_key,sep="|")
ids<-key(tasks); matched<-match(ids,key(oracle)); values<-oracle$p_value[matched]
missing<-which(is.na(matched)); unique_missing<-missing[!duplicated(ids[missing])]
env<-fastkpc_legacy_env();data<-as.matrix(readRDS(dataset))[,1:22,drop=FALSE]
suff<-list(data=data,ic.method="dcc.gamma",index=1,numCol=35L)
started<-proc.time()[["elapsed"]]
for(j in seq_along(unique_missing)) {
  i<-unique_missing[[j]];S<-if(nzchar(tasks$S_key[i]))as.integer(strsplit(tasks$S_key[i],"|",fixed=TRUE)[[1L]])else integer()
  set.seed(20261022L+j)
  p<-env$kernelCItest(x=tasks$x[i],y=tasks$y[i],S=S,suffStat=suff)
  stopifnot(length(p)==1L,is.finite(p),p>=0,p<=1)
  values[ids==ids[i]]<-as.numeric(p)
  if(j%%100L==0L || j==1L) {cat("offline authority",j,"/",length(unique_missing),"\n");flush.console()}
}
stopifnot(all(is.finite(values)))
trace<-tasks[,c("canonical_test_order_id","level","task_index","x","y","S_key")]
trace$p_value<-values
authority<-list(schema="subset22-offline-authority-v1",trace=trace,
  archived_logical_count=sum(!is.na(matched)),new_logical_count=length(missing),
  new_unique_ci_count=length(unique_missing),elapsed_sec=proc.time()[["elapsed"]]-started,
  policy="independent kpcalg validation only; never consumed by numerical production",
  inputs=setNames(lapply(c(reference_path,oracle_path,dataset),sha),c(reference_path,oracle_path,dataset)),
  source_sha256=setNames(lapply(c("fastkpc/R/legacy_runner.R",list.files("kpcalg/R",full.names=TRUE),
    "fastkpc/tools/research/structured_prefill_20261022/authority.R"),sha),
    c("fastkpc/R/legacy_runner.R",list.files("kpcalg/R",full.names=TRUE),
    "fastkpc/tools/research/structured_prefill_20261022/authority.R")),
  session=capture.output(sessionInfo()))
saveRDS(authority,file.path(out,"authority.rds"),compress=FALSE)
jsonlite::write_json(list(logical_ci=nrow(trace),archived=authority$archived_logical_count,
  new_logical=length(missing),new_unique=length(unique_missing),
  max_reference_error=max(abs(tasks$p_used-values)),
  reference_alpha_flips=sum((tasks$p_used>=.1)!=(values>=.1)),
  sha256=sha(file.path(out,"authority.rds")),inputs=authority$inputs),
  file.path(out,"authority.json"),pretty=TRUE,auto_unbox=TRUE,digits=NA)
cat("PASS offline authority coverage",nrow(trace),"\n")
