# Complete canonical skeleton. CPU authority is read only after GPU production.
# Rscript run.R OUTPUT s3|s2 REFERENCE REPEAT_REFERENCE_OR_DASH
args <- commandArgs(TRUE)
stopifnot(length(args)==4L, args[[2L]] %in% c("s3","s2"))
out <- args[[1L]]; mode <- args[[2L]]; reference_path <- args[[3L]]
stopifnot(!file.exists(file.path(out,"receipt.rds")))
dir.create(out,recursive=TRUE,showWarnings=FALSE)
Sys.setenv(FASTKPC_DCC_GAMMA_LEVEL_COMPONENT_PREFILL="1",
  FASTKPC_DCC_GAMMA_STRUCTURED_COMPONENT_PREFILL="1",
  FASTKPC_DCC_GAMMA_L2_COMPONENT_PREFILL=if(mode=="s2")"1"else"0",
  FASTKPC_DIAG_LEVEL_COMPONENT_ORDER="1")
stopifnot(!nzchar(Sys.getenv("FASTKPC_LSCP_CANONICAL_AUDIT_DIR")))
source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
source("fastkpc/tools/research/exact_speedup_20261020/pinned_optimizer_gates.R")
sha <- function(p) digest::digest(file=p,algo="sha256",serialize=FALSE)
build <- jsonlite::read_json(file.path(dirname(out),"build.json"),simplifyVector=TRUE)
sources <- names(build$source_sha256)
source_sha <- setNames(vapply(sources,sha,""),sources)
native <- "fastkpc/build/fastkpc_cuda.so"; native_sha <- sha(native)
stopifnot(identical(native_sha,build$native_sha256),
  identical(source_sha,unlist(build$source_sha256)),build$status=="PASS")
dataset <- "fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"
data <- as.matrix(readRDS(dataset)); storage.mode(data) <- "double"
stopifnot(identical(dim(data),c(351L,48L)))
invisible(full_cuda_ci_one_call_cache_control_native("configure",262144L))
invisible(full_cuda_ci_one_call_cache_control_native("configure_target",131072L))
invisible(full_cuda_ci_one_call_cache_control_native("reset"))
set.seed(20261020L); seed <- .Random.seed
before <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
started <- proc.time()[["elapsed"]]
result <- fastkpc_compatible_cuda_skeleton(data,alpha=.1,options=list(
  route="full_cuda",compatible_cuda_strict=TRUE,max_conditioning_size=Inf,
  index=1,numCol=35L,trace_level="logical",ci_method="dcc.gamma"))
elapsed <- unname(proc.time()[["elapsed"]]-started)
receipt <- list(schema="l2-prefill-runtime-canonical-v1",mode=mode,workload="canonical",
  native_sha256=native_sha,source_sha256=source_sha,dataset_sha256=sha(dataset),elapsed_sec=elapsed,
  seed_before=seed,seed_after=.Random.seed,resources_before=before,
  resources_after=fastkpc_full_cuda_phase35_vertical_resource_snapshot(),result=result,
  runtime_environment=as.list(Sys.getenv(c("CUDA_VISIBLE_DEVICES","CUDA_DEVICE_MAX_CONNECTIONS",
    "OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS",
    "FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES",
    "FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS",
    "FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES"))),
  evidence_class="full-build source-closure development; not a clean producer or promotion")
saveRDS(receipt,file.path(out,"pending.rds"),compress=FALSE)
receipt <- qualify_pinned_optimizer_receipt(receipt,reference_path)
reference <- readRDS(reference_path); ref <- reference$result
s <- result$summary; t <- result$tasks; g <- receipt$gates
comparisons <- g[c("p_value_bitwise_exact","pmax_bitwise_exact")]
if(mode=="s2") g[c("p_value_bitwise_exact","pmax_bitwise_exact")] <- NULL
expressions <- parse("fastkpc/tools/run_phase10_pvalue_contract_audit.R")
for(e in expressions) if(is.call(e) && identical(e[[1L]],as.name("<-")) &&
  is.symbol(e[[2L]]) && as.character(e[[2L]]) %in%
  c("fail","numeric_comparison","canonical_deletion_sepsets")) eval(e)
oracle_dir <- "fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2"
oracle <- readRDS(file.path(oracle_dir,"logical_ci_trace.rds"))
authority_pmax <- numeric_comparison(result$pMax,readRDS(file.path(oracle_dir,"pmax.rds")),1e-10)
g$authority_pmax_all_values <- authority_pmax$pass
g$authority_semantic_sepsets <- identical(canonical_deletion_sepsets(t,"native_edge_deleted"),
  canonical_deletion_sepsets(oracle,"deletes_edge"))
g$authority_decisions <- identical(t$p_used>=.1,oracle$p_value>=.1)
g$native_unchanged <- identical(sha(native),native_sha)
g$source_unchanged <- identical(source_sha,setNames(vapply(sources,sha,""),sources))
g$requested_mode <- isTRUE(s$level_component_prefill_enabled) &&
  isTRUE(s$structured_component_prefill_enabled) &&
  identical(s$l2_component_prefill_enabled,mode=="s2")
g$complete_trace <- nrow(t)==240498L
g$exact_primary <- s$cuda_multi_penalty_target_count==80751 &&
  s$cuda_multi_penalty_exact_primary_target_count==80751 &&
  s$cuda_multi_penalty_stability_replay_target_count==0 &&
  s$cuda_multi_penalty_stability_replay_selected_count==0 &&
  s$cuda_multi_penalty_terminal_confirmation_count==0
g$stable_svd <- s$cuda_exact_fixed_evaluator_target_count==s$cuda_exact_fixed_evaluator_stable_svd_target_count
g$no_approximate_gamma <- s$guarded_pair_count==0 && s$cuda_exact_screen_pair_count==0 &&
  s$cuda_guard_refinement_pair_count==0 && s$cuda_legacy_compatible_pair_count==s$physical_tests_evaluated
g$cache_metadata <- s$legacy_component_cache_route_mismatch_count==0 &&
  s$legacy_component_cache_status_mismatch_count==0
g$arena_closed <- s$exact_primary_fixed_reuse_allocation_count==s$exact_primary_fixed_reuse_free_count &&
  s$exact_primary_fixed_reuse_live_device_bytes_after==0 && s$exact_primary_fixed_reuse_fallback_target_count==0
live <- c("live_device_allocations","live_device_bytes","live_streams","live_events")
g$resources_zero <- all(unlist(before[live])==0) && all(unlist(receipt$resources_after[live])==0)
if(mode=="s2") g$same_native <- identical(native_sha,reference$native_sha256)
g$physical_groups_exact <- identical(s$physical_group_sequence_sha256,ref$summary$physical_group_sequence_sha256)
g$gram_keys_and_order_exact <- identical(s$gram_resolution_sequence_sha256,ref$summary$gram_resolution_sequence_sha256)
rows <- s$level_component_prefill
g$level_ledgers <- length(rows)==if(mode=="s2")7L else 6L
g$level_ledgers <- g$level_ledgers && all(vapply(rows,function(r)
  r$planned==r$produced && r$produced==r$residual_targets &&
  r$resolved+r$pending_at_end==r$planned,logical(1L)))
g$full_coverage <- s$level_component_produced==if(mode=="s2")130670 else 80751
g$full_coverage <- g$full_coverage &&
  s$level_component_resolved==if(mode=="s2")108379 else 65694
g$full_coverage <- g$full_coverage && s$exact_primary_fixed_reuse_hit_target_count==0 &&
  sum(vapply(rows,`[[`,numeric(1L),"resolve_batches"))==if(mode=="s2")24932 else 15264
g$structured_ledger <- all(vapply(rows,function(r)
  r$structured$accepted+r$structured$dense_fallback==r$planned &&
  r$structured$contexts<=10 && r$structured$width==64 && r$structured$peak_inflight<=640,
  logical(1L)))
consumers <- s$level_component_consumers
g$consumer_ledgers <- all(vapply(consumers,function(r)
  r$batches==r$all_hit_batches+r$mixed_batches+r$all_miss_batches &&
  r$targets==r$bypassed_targets+r$residual_targets,logical(1L)))
l2consumer <- Filter(function(r)r$level==2,consumers)[[1L]]
g$l2_consumer <- l2consumer$batches==22720 && l2consumer$targets==147176
if(mode=="s2") {
  l2 <- Filter(function(r)r$level==2,rows)[[1L]]
  g$l2_producer <- l2$planned==49919 && l2$produced==49919 && l2$resolved==42685 &&
    l2$pending_at_end==7234 && l2$residual_targets==49919 &&
    s$l2_component_prepared_count==1126 && s$l2_component_residual_batches==3356 &&
    s$l2_component_residual_targets==49919
  g$l2_consumer <- g$l2_consumer && l2consumer$all_hit_batches==22720 &&
    l2consumer$mixed_batches==0 && l2consumer$all_miss_batches==0 &&
    l2consumer$bypassed_targets==147176 && l2consumer$residual_targets==0 &&
    l2consumer$residual_host_ms==0 && l2consumer$full_eig_cuda_ms==0
} else {
  g$l2_disabled <- s$l2_component_prepared_count==0 && s$l2_component_residual_batches==0 &&
    s$l2_component_residual_targets==0
  g$l2_consumer <- g$l2_consumer && l2consumer$all_hit_batches==13052 &&
    l2consumer$mixed_batches==8454 && l2consumer$all_miss_batches==1214 &&
    l2consumer$bypassed_targets==60066 && l2consumer$residual_targets==87110
}
prefilled_consumers <- Filter(function(r)r$level>=if(mode=="s2")2 else 3,consumers)
g$prefilled_consumers_all_bypass <- all(vapply(prefilled_consumers,function(r)
  r$batches==r$all_hit_batches && r$targets==r$bypassed_targets &&
  r$residual_targets==0 && r$residual_host_ms==0 && r$full_eig_cuda_ms==0,logical(1L)))
bits_differ <- function(a,b) {
  stopifnot(length(a)==length(b))
  colSums(matrix(writeBin(as.double(a),raw(),size=8)!=writeBin(as.double(b),raw(),size=8),nrow=8))>0
}
different <- bits_differ(t$p_used,ref$tasks$p_used)
g$only_l2_p_changed <- !any(different[t$level!=2L])
if(mode=="s2") {
  frozen_path <- "fastkpc/artifacts/full_cuda_ci/l2_prefill_20261022/qualification.rds"
  frozen <- readRDS(frozen_path)$values$structured1
  current <- t$p_used[t$level==2L]
  stopifnot(length(frozen)==length(current))
  comparisons$frozen_l2 <- list(receipt_sha256=sha(frozen_path),
    bitwise=identical(current,frozen,num.eq=FALSE,single.NA=FALSE),
    changed=sum(bits_differ(current,frozen)),maximum=max(abs(current-frozen)))
  g$frozen_l2_continuous_contract <- comparisons$frozen_l2$maximum<=1e-10
}
comparisons$changed_ci_count <- sum(different)
comparisons$max_ci_difference <- max(abs(t$p_used-ref$tasks$p_used))
comparisons$changed_pmax_count <- sum(bits_differ(result$pMax,ref$pMax))
comparisons$pmax <- numeric_comparison(result$pMax,ref$pMax,1e-10)
comparisons$alpha_flips <- sum((t$p_used>=.1)!=(ref$tasks$p_used>=.1))
g$reference_continuous_contract <- comparisons$max_ci_difference<=1e-10 && comparisons$pmax$pass
g$reference_decisions <- comparisons$alpha_flips==0
if(args[[4L]]!="-") {
  repeated <- readRDS(args[[4L]])
  g$repeat_ci_bitwise <- identical(t$p_used,repeated$result$tasks$p_used,num.eq=FALSE,single.NA=FALSE)
  g$repeat_pmax_bitwise <- identical(result$pMax,repeated$result$pMax,num.eq=FALSE,single.NA=FALSE)
  g$repeat_native <- identical(native_sha,repeated$native_sha256)
}
changed <- t[different,,drop=FALSE]
changed$reference_p <- ref$tasks$p_used[different]
changed$absolute_difference <- abs(changed$p_used-changed$reference_p)
saveRDS(changed,file.path(out,"changed-ci.rds"),compress=FALSE)
near <- oracle$p_value>=.05 & oracle$p_value<=.15
comparisons$authority_near_alpha <- list(interval=c(.05,.15),count=sum(near),
  maximum_absolute_error=max(abs(t$p_used[near]-oracle$p_value[near])),
  changed_vs_reference=sum(different[near]))
comparisons$by_level <- lapply(sort(unique(t$level)),function(level) {
  keep<-t$level==level
  list(level=level,count=sum(keep),changed=sum(different[keep]),
    max_difference=max(abs(t$p_used[keep]-ref$tasks$p_used[keep])),
    max_authority_error=max(abs(t$p_used[keep]-oracle$p_value[keep])),
    near_alpha_count=sum(keep & near),
    near_alpha_max_authority_error=if(any(keep & near))
      max(abs(t$p_used[keep & near]-oracle$p_value[keep & near])) else 0)
})
sum_stage <- function(field) sum(vapply(s$level_component_prefill,`[[`,numeric(1),field))/1000
l2rows <- Filter(function(r)r$level==2,rows)
l2field <- function(field) if(length(l2rows)) l2rows[[1L]][[field]]/1000 else 0
stages <- list(elapsed_sec=elapsed,optimizer_sec=s$cuda_optimizer_host_ms/1000,
  l2_elapsed_sec=result$levels$elapsed_ms[result$levels$level==2]/1000,
  stage_a_sec=sum_stage("stage_a_host_ms"),
  begin_sec=sum_stage("begin_host_ms"),finish_sec=sum_stage("finish_host_ms"),
  l2_stage_a_sec=l2field("stage_a_host_ms"),l2_stage_b_sec=l2field("stage_b_host_ms"),
  l2_begin_sec=l2field("begin_host_ms"),l2_finish_sec=l2field("finish_host_ms"),
  l2_input_sec=s$l2_component_input_host_ms/1000,
  l2_prepared_sec=s$l2_component_prepare_host_ms/1000,
  l2_evaluate_sec=s$l2_component_evaluate_host_ms/1000,
  l2_release_sec=s$l2_component_release_host_ms/1000,
  l2_consumer_residual_sec=l2consumer$residual_host_ms/1000,
  l2_consumer_eig_sec=l2consumer$full_eig_cuda_ms/1000,
  l2_consumer_dcov_sec=l2consumer$dcov_host_ms/1000,
  stage_b_sec=sum_stage("stage_b_host_ms"),consumer_dcov_sec=s$cuda_dcov_host_ms/1000,
  prepared_build_sec=s$cuda_multi_penalty_prepared_build_ms/1000,
  native_setup_sec=s$native_setup_ms/1000,consumer_eig_sec=s$cuda_dcov_full_eig_ms/1000,
  optimizer_iterations=s$cuda_multi_penalty_optimizer_iteration_sum,
  optimizer_max_iterations=s$cuda_multi_penalty_optimizer_iteration_max,
  optimizer_by_level=lapply(sort(unique(s$prefill_batches$level)),function(level) {
    keep<-s$prefill_batches$level==level
    list(level=level,host_sec=sum(s$prefill_batches$optimizer_host_ms[keep])/1000,
      targets=sum(s$prefill_batches$target_optimization_count[keep]))
  }))
receipt$gates<-g; receipt$comparisons<-comparisons; receipt$stages<-stages
receipt$authority_pmax<-authority_pmax
receipt$authority_sha256<-setNames(vapply(file.path(oracle_dir,c("logical_ci_trace.rds","pmax.rds","adjacency.rds")),sha,""),
  c("logical_ci_trace","pmax","adjacency"))
saveRDS(receipt,file.path(out,"receipt.rds"),compress=FALSE)
jsonlite::write_json(list(schema=receipt$schema,mode=mode,native_sha256=native_sha,
  source_sha256=as.list(source_sha),runner_sha256=sha("fastkpc/tools/research/l2_prefill_runtime_20261022/run.R"),
  reference_sha256=sha(reference_path),receipt_sha256=sha(file.path(out,"receipt.rds")),
  authority_sha256=as.list(receipt$authority_sha256),elapsed_sec=elapsed,gates=g,
  oracle_errors=receipt$oracle_errors,authority_pmax=authority_pmax,comparisons=comparisons,
  stages=stages,evidence_class=receipt$evidence_class,level_component_prefill=rows,
  level_component_consumers=consumers,
  l2_prepared_count=s$l2_component_prepared_count,l2_residual_batches=s$l2_component_residual_batches,
  level_trace=result$levels,
  resources_before=before,resources_after=receipt$resources_after),
  file.path(out,"report.json"),pretty=TRUE,auto_unbox=TRUE,digits=NA)
print(g);print(comparisons);print(stages)
cat("elapsed",elapsed,"seconds; logical CI",nrow(t),"\n")
stopifnot(all(unlist(g)))
