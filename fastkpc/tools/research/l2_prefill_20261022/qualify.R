# Independently join physical GPU pair outputs to the real logical trace.
# Arithmetic here compares results and replays graph/pMax control only.
args<-commandArgs(TRUE);stopifnot(length(args)==1L);out<-args[[1L]]
capture<-readRDS(file.path(out,"capture/receipt.rds"))
reference<-readRDS("fastkpc/artifacts/full_cuda_ci/structured_prefill_canonical_20261022/structured1/receipt.rds")
t<-reference$result$tasks;keep<-t$level==2L
oracle<-readRDS("fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2/logical_ci_trace.rds")
oracle_pmax<-readRDS("fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2/pmax.rds")
stopifnot(all(unlist(capture$gates)),sum(keep)==125293L,nrow(t)==nrow(oracle))
bits<-function(a,b) identical(as.double(a),as.double(b),num.eq=FALSE,single.NA=FALSE)
changed<-function(a,b)colSums(matrix(writeBin(as.double(a),raw(),size=8)!=writeBin(as.double(b),raw(),size=8),nrow=8))>0
identity<-function(x,y,s)paste(x,y,s,sep=":")
logical_key<-identity(t$x[keep],t$y[keep],t$S_key[keep])
mapping<-read.delim(file.path(out,"capture/physical-map.tsv"),stringsAsFactors=FALSE,
 colClasses=c("integer","integer","character","numeric","integer","integer"))
physical_key<-identity(mapping$x,mapping$y,mapping$S_key)
stopifnot(nrow(mapping)==125293L,!anyDuplicated(physical_key),!anyDuplicated(logical_key),!anyDuplicated(mapping$sequence))
positions<-match(logical_key,physical_key);stopifnot(!anyNA(positions))
make_pmax<-function(p){m<-matrix(-Inf,48,48);diag(m)<-1;for(i in seq_along(p)){
 x<-t$edge_x[i];y<-t$edge_y[i];if(p[i]>m[x,y])m[x,y]<-m[y,x]<-p[i]};m}
stopifnot(bits(make_pmax(t$p_used),reference$result$pMax))
names_run<-c("original","dense1","structured1","structured2","dense2")
reports<-list();values<-list()
for(name in names_run){
 p<-read.delim(file.path(out,paste0(name,".tsv")),stringsAsFactors=FALSE,colClasses=c("integer","integer","character","numeric","numeric","numeric"))
 # The TSV stores unordered endpoints for presentation; direction is recovered
 # from the original consumer, not inferred by folding reciprocal CI tests.
 stopifnot(identical(p$sequence,mapping$sequence),identical(p$S_key,mapping$S_key),
  identical(p$x,pmin(mapping$x,mapping$y)),identical(p$y,pmax(mapping$x,mapping$y)))
 original<-p$original_p[positions];new<-p$p[positions]
 stopifnot(bits(original,t$p_used[keep]))
 if(name=="original")stopifnot(bits(new,original))
 full<-t$p_used;full[keep]<-new;pm<-make_pmax(full)
 finite<-is.finite(oracle_pmax)
 error<-abs(new-oracle$p_value[keep]);near<-oracle$p_value[keep]>=.05&oracle$p_value[keep]<=.15
 gates<-list(logical_identity=length(new)==125293L,ci_contract=max(error)<=1e-10,
  pmax_contract=identical(dim(pm),dim(oracle_pmax))&&
   identical(unname(is.finite(pm)),unname(is.finite(oracle_pmax)))&&
   bits(pm[!finite],oracle_pmax[!finite])&&max(abs(pm[finite]-oracle_pmax[finite]))<=1e-10,
  all_decisions=identical(full>=.1,oracle$p_value>=.1),
  original_gpu_anchor=bits(original,t$p_used[keep]),
  resources=jsonlite::read_json(file.path(out,paste0(name,".json")))$resources_zero)
 reports[[name]]<-list(gates=gates,physical_pairs=nrow(p),logical_ci_count=length(new),
  changed_vs_current=sum(changed(new,original)),max_difference_vs_current=max(abs(new-original)),
  max_authority_error=max(error),near_alpha=list(count=sum(near),changed=sum(changed(new,original)[near]),max_authority_error=max(error[near])),
  frozen_full_trace_pmax_changed=sum(changed(pm,reference$result$pMax)),
  frozen_full_trace_pmax_max_authority_error=max(abs(pm[finite]-oracle_pmax[finite])))
 values[[name]]<-new
 if(!all(unlist(gates)))print(list(run=name,report=reports[[name]]))
 stopifnot(all(unlist(gates)))
}
repeat_gates<-list(dense_bitwise=bits(values$dense1,values$dense2),structured_bitwise=bits(values$structured1,values$structured2))
stopifnot(all(unlist(repeat_gates)))
structured_vs_dense<-list(changed=sum(changed(values$structured1,values$dense1)),maximum=max(abs(values$structured1-values$dense1)))
result<-list(status="PASS_FROZEN_L2_QUALIFICATION",runs=reports,repeat_gates=repeat_gates,
 structured_vs_dense=structured_vs_dense,
 scope="L2 producer/component/pair GPU replay; full pMax uses frozen higher-level trace, not a new full-runtime run",
 runtime_integration="NOT_RUN",phase10="ACTIVE",candidate="NOT_FROZEN",holdout="SEALED",recommended=FALSE)
jsonlite::write_json(result,file.path(out,"qualification.json"),pretty=TRUE,auto_unbox=TRUE,digits=NA)
saveRDS(list(report=result,values=values),file.path(out,"qualification.rds"),compress=FALSE)
print(result)
