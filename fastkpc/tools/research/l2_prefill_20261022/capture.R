# Actual canonical L0-L2 only. Capture time is not performance evidence.
args <- commandArgs(TRUE); stopifnot(length(args)==1L)
out <- normalizePath(args[[1L]],mustWork=FALSE)
dir.create(out,recursive=TRUE,showWarnings=FALSE)
stopifnot(!file.exists(file.path(out,"receipt.rds")))
dir.create(file.path(out,"raw"),showWarnings=FALSE)
stopifnot(normalizePath(Sys.getenv("FASTKPC_L2_CAPTURE"))==normalizePath(file.path(out,"raw")))
source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
sha <- function(p) digest::digest(file=p,algo="sha256",serialize=FALSE)
native <- "fastkpc/build/fastkpc_cuda.so"
native_sha <- sha(native)
stopifnot(native_sha=="89a2fab0bd02b695878556a74694fc522cddcbe2c5c2f22f74c7e4da6d5d56c3")
dataset <- "fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"
data <- as.matrix(readRDS(dataset));storage.mode(data)<-"double"
con<-file(file.path(out,"raw","dataset.f64"),"wb");writeBin(as.double(data),con,size=8L,endian="little");close(con)
Sys.setenv(FASTKPC_DCC_GAMMA_LEVEL_COMPONENT_PREFILL="0",
 FASTKPC_DCC_GAMMA_STRUCTURED_COMPONENT_PREFILL="0",FASTKPC_DIAG_LEVEL_COMPONENT_ORDER="1")
invisible(full_cuda_ci_one_call_cache_control_native("configure",262144L))
invisible(full_cuda_ci_one_call_cache_control_native("configure_target",131072L))
invisible(full_cuda_ci_one_call_cache_control_native("reset"))
set.seed(20261020L); seed <- .Random.seed
before<-fastkpc_full_cuda_phase35_vertical_resource_snapshot()
result<-fastkpc_compatible_cuda_skeleton(data,alpha=.1,options=list(route="full_cuda",
 compatible_cuda_strict=TRUE,max_conditioning_size=2L,index=1,numCol=35L,
 trace_level="logical",ci_method="dcc.gamma"))
after<-fastkpc_full_cuda_phase35_vertical_resource_snapshot()
reference_path<-"fastkpc/artifacts/full_cuda_ci/structured_prefill_canonical_20261022/off/receipt.rds"
reference<-readRDS(reference_path); t<-result$tasks; rt<-subset(reference$result$tasks,level<=2)
row.names(rt)<-NULL
fields<-c("level","task_index","x","y","S_key","canonical_test_order_id","native_edge_deleted","native_edge_ignored")
prefix_graph<-matrix(TRUE,48,48);diag(prefix_graph)<-FALSE
prefix_pmax<-matrix(-Inf,48,48);diag(prefix_pmax)<-1
for(i in seq_len(nrow(rt))) {
 x<-rt$x[i];y<-rt$y[i]
 prefix_pmax[x,y]<-prefix_pmax[y,x]<-max(prefix_pmax[x,y],rt$p_used[i])
 if(rt$native_edge_deleted[i])prefix_graph[x,y]<-prefix_graph[y,x]<-FALSE
}
live<-c("live_device_allocations","live_device_bytes","live_streams","live_events")
authority_fields<-c("r_callback_count","legacy_mgcv_fit_count","legacy_mgcv_setup_count",
 "cpu_residual_solve_count","cpu_dcov_component_count","cpu_dcov_eigen_or_lowrank_count",
 "cpu_dcov_pair_stat_count","cpu_gamma_pvalue_count","cpu_spectra_count","residual_d2h_bytes",
 "component_d2h_bytes","unknown_fallback_count","approximate_backend_count")
oracle<-readRDS("fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2/logical_ci_trace.rds")
oracle<-subset(oracle,level<=2)
gates<-list(task_identity=identical(t[fields],rt[fields]),
 ci_bits=identical(t$p_used,rt$p_used,num.eq=FALSE),
 pmax_bits=identical(unname(result$pMax),prefix_pmax,num.eq=FALSE),
 adjacency=identical(unname(result$adjacency),prefix_graph),
 counts=identical(as.integer(result$n.edgetests),c(2213L,52659L,125293L)),
 authority=max(abs(t$p_used-oracle$p_value))<=1e-10,
 gpu_only=all(vapply(authority_fields,function(k)!is.null(result$summary[[k]])&&result$summary[[k]]==0,logical(1))),
 resources=all(unlist(after[live])==0)&&identical(before[live],after[live]),
 rng=identical(seed,.Random.seed),native_unchanged=sha(native)==native_sha)
receipt<-list(schema="l2-prefill-capture-v1",native_sha256=native_sha,result=result,
 seed_before=seed,seed_after=.Random.seed,resources_before=before,resources_after=after,gates=gates,
 evidence_class="external observation only; all shadows diagnostic; no capture performance claim")
saveRDS(receipt,file.path(out,"receipt.rds"),compress=FALSE)
write.csv(t,file.path(out,"tasks.csv"),row.names=FALSE)
write.csv(oracle,file.path(out,"authority.csv"),row.names=FALSE)
jsonlite::write_json(list(gates=gates,native_sha256=native_sha,
 reference_sha256=sha(reference_path),dataset_sha256=sha(dataset),
 receipt_sha256=sha(file.path(out,"receipt.rds")),max_error=max(abs(t$p_used-oracle$p_value))),
 file.path(out,"capture.json"),pretty=TRUE,auto_unbox=TRUE,digits=NA)
print(gates);stopifnot(all(unlist(gates)))
cat("PASS actual L2 capture",nrow(t),"logical CI\n")
