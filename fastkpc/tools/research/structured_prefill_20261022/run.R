# Fresh-process same-binary off/dense/structured complete 22-column skeleton.
# Numerical results are produced on CUDA. Authority is an offline comparison.
args<-commandArgs(TRUE)
stopifnot(length(args)==4L,args[[2]]%in%c("off","dense","structured"))
out<-args[[1]];mode<-args[[2]];reference_path<-args[[3]];authority_path<-args[[4]]
stopifnot(!file.exists(file.path(out,"receipt.rds")))
dir.create(out,recursive=TRUE,showWarnings=FALSE)
Sys.setenv(FASTKPC_DCC_GAMMA_LEVEL_COMPONENT_PREFILL=if(mode=="off")"0"else"1",
  FASTKPC_DCC_GAMMA_STRUCTURED_COMPONENT_PREFILL=if(mode=="structured")"1"else"0",
  FASTKPC_DIAG_LEVEL_COMPONENT_ORDER="1")
source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
source("fastkpc/tools/research/exact_speedup_20261020/pinned_optimizer_gates.R")
sha<-function(p)digest::digest(file=p,algo="sha256",serialize=FALSE)
sources<-c("fastkpc/src/full_cuda_ci_one_call.cpp","fastkpc/src/cuda/full_cuda_ci_vertical.cu",
 "fastkpc/src/cuda/full_cuda_ci_vertical.hpp","fastkpc/src/cuda/full_cuda_ci_structured_eig.cu",
 "fastkpc/src/cuda/full_cuda_ci_structured_eig.hpp","fastkpc/tools/build_cuda_native.sh")
source_sha<-setNames(vapply(sources,sha,""),sources)
native<-"fastkpc/build/fastkpc_cuda.so";native_sha<-sha(native)
dataset<-"fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"
data<-as.matrix(readRDS(dataset))[,1:22,drop=FALSE];storage.mode(data)<-"double"
invisible(full_cuda_ci_one_call_cache_control_native("configure",262144L))
invisible(full_cuda_ci_one_call_cache_control_native("configure_target",131072L))
invisible(full_cuda_ci_one_call_cache_control_native("reset"))
set.seed(20261020L);seed<-.Random.seed
before<-fastkpc_full_cuda_phase35_vertical_resource_snapshot()
started<-proc.time()[["elapsed"]]
result<-fastkpc_compatible_cuda_skeleton(data,alpha=.1,options=list(
 route="full_cuda",compatible_cuda_strict=TRUE,max_conditioning_size=Inf,
 index=1,numCol=35L,trace_level="logical",ci_method="dcc.gamma"))
elapsed<-unname(proc.time()[["elapsed"]]-started)
receipt<-list(schema="structured-prefill-subset22-v1",mode=mode,workload="subset22-default-inf",
 native_sha256=native_sha,source_sha256=source_sha,dataset_sha256=sha(dataset),elapsed_sec=elapsed,
 seed_before=seed,seed_after=.Random.seed,resources_before=before,
 resources_after=fastkpc_full_cuda_phase35_vertical_resource_snapshot(),result=result,
 runtime_environment=as.list(Sys.getenv(c("CUDA_VISIBLE_DEVICES","CUDA_DEVICE_MAX_CONNECTIONS",
 "OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS",
 "FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES","FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS",
 "FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES"))),
 evidence_class="source-closure development; no promotion")
saveRDS(receipt,file.path(out,"pending.rds"),compress=FALSE)
read_receipt<-function(p)readRDS(if(grepl("\\.gz$",p))gzfile(p)else p)
reference<-read_receipt(reference_path)
# Common audit helper takes an RDS path; gzfile auto-detection is not universal.
receipt<-qualify_pinned_optimizer_receipt(receipt,NULL)
g<-receipt$gates;s<-result$summary;t<-result$tasks;ref<-reference$result
task_fields<-c("level","task_index","x","y","S_key","canonical_test_order_id",
 "native_edge_deleted","native_edge_ignored")
g$task_identity_and_deletion_exact<-identical(t[task_fields],ref$tasks[task_fields])
g$adjacency_exact<-identical(result$adjacency,ref$adjacency)
g$sepsets_exact<-identical(result$sepsets,ref$sepsets)
g$n_edgetests_exact<-identical(result$n.edgetests,ref$n.edgetests)
g$rng_endpoint_exact<-identical(receipt$seed_after,reference$seed_after)
comparisons<-list(ci_bitwise=identical(t$p_used,ref$tasks$p_used,num.eq=FALSE,single.NA=FALSE),
 pmax_bitwise=identical(result$pMax,ref$pMax,num.eq=FALSE,single.NA=FALSE),
 max_ci_difference=max(abs(t$p_used-ref$tasks$p_used)),
 max_pmax_difference=max(abs(result$pMax-ref$pMax)),
 alpha_flips=sum((t$p_used>=.1)!=(ref$tasks$p_used>=.1)))
g$reference_ci_contract<-comparisons$max_ci_difference<=1e-10 && comparisons$alpha_flips==0
g$reference_pmax_contract<-comparisons$max_pmax_difference<=1e-10
if(mode!="structured") {g$ci_bits_exact<-comparisons$ci_bitwise;g$pmax_bits_exact<-comparisons$pmax_bitwise}
authority<-readRDS(authority_path);a<-authority$trace
g$authority_task_identity<-identical(t[names(a)[names(a)!="p_value"]],a[names(a)[names(a)!="p_value"]])
g$authority_ci_contract<-nrow(a)==nrow(t) && all(abs(t$p_used-a$p_value)<=1e-10)
g$authority_decisions<-identical(t$p_used>=.1,a$p_value>=.1)
pmax_from_trace<-function(values) {
 m<-matrix(-Inf,22,22,dimnames=dimnames(result$pMax));diag(m)<-1
 for(i in seq_along(values)) {x<-t$edge_x[i];y<-t$edge_y[i];m[x,y]<-m[y,x]<-max(m[x,y],values[i])}
 m
}
g$pmax_reconstruction<-identical(pmax_from_trace(t$p_used),result$pMax,num.eq=FALSE,single.NA=FALSE)
authority_pmax<-pmax_from_trace(a$p_value)
g$authority_pmax_contract<-all(abs(result$pMax-authority_pmax)<=1e-10)
g$native_unchanged<-identical(sha(native),native_sha)
g$source_unchanged<-identical(source_sha,setNames(vapply(sources,sha,""),sources))
g$complete_trace<-nrow(t)==24264L
g$requested_mode<-identical(s$level_component_prefill_enabled,mode!="off") &&
 identical(s$structured_component_prefill_enabled,mode=="structured")
g$authority_gate<-isTRUE(s$authority_gate_pass)
g$exact_primary<-s$cuda_multi_penalty_target_count==s$cuda_multi_penalty_exact_primary_target_count &&
 s$cuda_multi_penalty_stability_replay_target_count==0 && s$cuda_multi_penalty_terminal_confirmation_count==0
g$stable_svd<-s$cuda_exact_fixed_evaluator_target_count==s$cuda_exact_fixed_evaluator_stable_svd_target_count
g$no_approximate_gamma<-s$guarded_pair_count==0 && s$cuda_exact_screen_pair_count==0 &&
 s$cuda_guard_refinement_pair_count==0 && s$cuda_legacy_compatible_pair_count==s$physical_tests_evaluated
g$cache_metadata<-s$legacy_component_cache_route_mismatch_count==0 && s$legacy_component_cache_status_mismatch_count==0
g$arena_closed<-s$exact_primary_fixed_reuse_allocation_count==s$exact_primary_fixed_reuse_free_count &&
 s$exact_primary_fixed_reuse_live_device_bytes_after==0 && s$exact_primary_fixed_reuse_fallback_target_count==0
if(mode!="off") {
 g$same_native<-identical(native_sha,reference$native_sha256)
 g$physical_groups_exact<-identical(s$physical_group_sequence_sha256,ref$summary$physical_group_sequence_sha256)
 g$gram_keys_and_order_exact<-identical(s$gram_resolution_sequence_sha256,ref$summary$gram_resolution_sequence_sha256)
 rows<-s$level_component_prefill
 g$level_ledgers<-length(rows)==4 && all(vapply(rows,function(r)
   r$planned==r$produced && r$planned==r$residual_targets && r$resolved+r$pending_at_end==r$planned,logical(1)))
 g$full_coverage<-s$level_component_produced==7270 && s$level_component_resolved==6632 &&
   s$exact_primary_fixed_reuse_hit_target_count==0 && sum(vapply(rows,`[[`,numeric(1),"resolve_batches"))==2592
 if(mode=="structured") g$structured_ledger<-all(vapply(rows,function(r)
   r$structured$accepted+r$structured$dense_fallback==r$planned &&
   r$structured$contexts<=10 && r$structured$width==64 && r$structured$peak_inflight<=640,logical(1)))
}
receipt$gates<-g;receipt$comparisons<-comparisons
receipt$oracle_errors<-list(ci=max(abs(t$p_used-a$p_value)),pMax=max(abs(result$pMax-authority_pmax)))
receipt$authority_sha256<-sha(authority_path)
saveRDS(receipt,file.path(out,"receipt.rds"),compress=FALSE)
jsonlite::write_json(list(schema=receipt$schema,mode=mode,native_sha256=native_sha,
 source_sha256=as.list(source_sha),receipt_sha256=sha(file.path(out,"receipt.rds")),
 reference_sha256=sha(reference_path),authority_sha256=sha(authority_path),
 runner_sha256=sha("fastkpc/tools/research/structured_prefill_20261022/run.R"),
 elapsed_sec=elapsed,gates=g,comparisons=comparisons,oracle_errors=receipt$oracle_errors,
 level_component_prefill=s$level_component_prefill,resources_before=before,
 resources_after=receipt$resources_after,evidence_class=receipt$evidence_class),
 file.path(out,"report.json"),pretty=TRUE,auto_unbox=TRUE,digits=NA)
print(g);print(comparisons);print(receipt$oracle_errors)
cat("elapsed",elapsed,"seconds; logical CI",nrow(t),"\n")
stopifnot(all(unlist(g)))
