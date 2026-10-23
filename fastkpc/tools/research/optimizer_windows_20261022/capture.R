# Freeze real L3/L5 windows while reproducing the qualified S2 prefix.
args<-commandArgs(TRUE);stopifnot(length(args)==1L)
out<-normalizePath(args[[1L]]);stopifnot(!file.exists(file.path(out,'capture.rds')))
source('fastkpc/R/fast_kpc.R');source('fastkpc/R/full_cuda_ci_phase35_vertical.R')
sha<-function(p)digest::digest(file=p,algo='sha256',serialize=FALSE)
native<-'fastkpc/build/fastkpc_cuda.so'
stopifnot(sha(native)=='8e457e811eaa01ae85326b8ff3cc0bcfa56f5b907ad6840a302a8205370d3874')
Sys.setenv(FASTKPC_DCC_GAMMA_LEVEL_COMPONENT_PREFILL='1',
 FASTKPC_DCC_GAMMA_STRUCTURED_COMPONENT_PREFILL='1',FASTKPC_DCC_GAMMA_L2_COMPONENT_PREFILL='1',
 FASTKPC_DIAG_LEVEL_COMPONENT_ORDER='1')
data<-as.matrix(readRDS('fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds'))
invisible(full_cuda_ci_one_call_cache_control_native('configure',262144L))
invisible(full_cuda_ci_one_call_cache_control_native('configure_target',131072L))
invisible(full_cuda_ci_one_call_cache_control_native('reset'))
set.seed(20261020L);seed<-.Random.seed
before<-fastkpc_full_cuda_phase35_vertical_resource_snapshot()
result<-fastkpc_compatible_cuda_skeleton(data,alpha=.1,options=list(route='full_cuda',
 compatible_cuda_strict=TRUE,max_conditioning_size=5L,index=1,numCol=35L,
 trace_level='logical',ci_method='dcc.gamma'))
after<-fastkpc_full_cuda_phase35_vertical_resource_snapshot()
reference_path<-'fastkpc/artifacts/full_cuda_ci/l2_prefill_runtime_20261022/s2b/receipt.rds'
reference<-readRDS(reference_path);rt<-subset(reference$result$tasks,level<=5L);row.names(rt)<-NULL
t<-result$tasks
fields<-c('level','task_index','x','y','S_key','canonical_test_order_id','native_edge_deleted','native_edge_ignored')
graph<-matrix(TRUE,48,48);diag(graph)<-FALSE
pmax<-matrix(-Inf,48,48);diag(pmax)<-1
for(i in seq_len(nrow(rt))) {
 x<-rt$x[i];y<-rt$y[i];pmax[x,y]<-pmax[y,x]<-max(pmax[x,y],rt$p_used[i])
 if(rt$native_edge_deleted[i])graph[x,y]<-graph[y,x]<-FALSE
}
live<-c('live_device_allocations','live_device_bytes','live_streams','live_events')
fixtures<-file.path(out,'raw',c('L3-W31.bin','L3-W32.bin','L5-W7.bin','L5-W8.bin'))
gates<-list(task_identity=identical(t[fields],rt[fields]),
 ci_bits=identical(t$p_used,rt$p_used,num.eq=FALSE),
 pmax_bits=identical(unname(result$pMax),pmax,num.eq=FALSE),
 graph=identical(unname(result$adjacency),graph),
 counts=identical(as.integer(result$n.edgetests),as.integer(reference$result$n.edgetests[1:6])),
 rng=identical(seed,.Random.seed),resources=all(unlist(after[live])==0),
 fixtures=all(file.exists(fixtures)))
receipt<-list(schema='optimizer-window-capture-v1',result=result,gates=gates,
 native_sha256=sha(native),reference_sha256=sha(reference_path),
 fixture_sha256=setNames(vapply(fixtures,sha,''),basename(fixtures)),
 seed_before=seed,seed_after=.Random.seed,resources_before=before,resources_after=after,
 evidence_class='diagnostic capture; shadows are diagnostic only; elapsed is not performance evidence')
saveRDS(receipt,file.path(out,'capture.rds'),compress=FALSE)
jsonlite::write_json(receipt[c('schema','gates','native_sha256','reference_sha256','fixture_sha256','evidence_class')],
 file.path(out,'capture.json'),pretty=TRUE,auto_unbox=TRUE)
print(gates);stopifnot(all(unlist(gates)));cat('PASS captured canonical prefix',nrow(t),'CI\n')
