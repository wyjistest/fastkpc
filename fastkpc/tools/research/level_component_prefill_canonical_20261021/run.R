# Canonical same-binary dense prefill qualification. CPU only audits GPU results.
# Rscript run.R OUTPUT off|on REFERENCE [TIMING_REFERENCE]
args <- commandArgs(TRUE)
stopifnot(length(args) %in% c(3L,4L), args[[2L]] %in% c("off","on"))
out <- args[[1L]]; mode <- args[[2L]]; baseline_path <- args[[3L]]
stopifnot(!file.exists(file.path(out,"receipt.rds")))
dir.create(out,recursive=TRUE,showWarnings=FALSE)
Sys.setenv(FASTKPC_DCC_GAMMA_LEVEL_COMPONENT_PREFILL=if(mode=="on") "1" else "0",
           FASTKPC_DIAG_LEVEL_COMPONENT_ORDER="1")
source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
source("fastkpc/tools/research/exact_speedup_20261020/pinned_optimizer_gates.R")
sha <- function(p) digest::digest(file=p,algo="sha256",serialize=FALSE)
native <- "fastkpc/build/fastkpc_cuda.so"
sources <- c("fastkpc/src/full_cuda_ci_one_call.cpp","fastkpc/src/cuda/full_cuda_ci_vertical.cu",
             "fastkpc/src/cuda/full_cuda_ci_vertical.hpp")
native_sha <- sha(native); source_sha <- setNames(vapply(sources,sha,""),sources)
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
receipt <- list(schema="level-component-prefill-canonical-v1",mode=mode,workload="canonical",
  native_sha256=native_sha,source_sha256=source_sha,dataset_sha256=sha(dataset),elapsed_sec=elapsed,
  seed_before=seed,seed_after=.Random.seed,resources_before=before,
  resources_after=fastkpc_full_cuda_phase35_vertical_resource_snapshot(),result=result,
  runtime_environment=as.list(Sys.getenv(c("CUDA_VISIBLE_DEVICES","CUDA_DEVICE_MAX_CONNECTIONS",
    "OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS",
    "FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES",
    "FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS",
    "FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES"))),
  evidence_class=if(nzchar(Sys.getenv("FASTKPC_LSCP_CANONICAL_AUDIT_DIR")))
    "diagnostic D2H observer; excluded from timing evidence" else "single source-closure development run")
saveRDS(receipt,file.path(out,"pending.rds"),compress=FALSE)
receipt <- qualify_pinned_optimizer_receipt(receipt,baseline_path)
s <- result$summary; reference <- readRDS(baseline_path)
receipt$comparisons <- receipt$gates[c("p_value_bitwise_exact","pmax_bitwise_exact")]
# Canonical on is qualified against the existing authority tolerance. Its dense
# baseline differences remain explicit and must separately pass payload attribution.
if(mode=="on") receipt$gates[c("p_value_bitwise_exact","pmax_bitwise_exact")] <- NULL
expressions <- parse("fastkpc/tools/run_phase10_pvalue_contract_audit.R")
for(e in expressions) if(is.call(e) && identical(e[[1L]],as.name("<-")) &&
  is.symbol(e[[2L]]) && as.character(e[[2L]]) %in%
  c("fail","numeric_comparison","canonical_deletion_sepsets")) eval(e)
oracle_dir <- "fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2"
oracle <- readRDS(file.path(oracle_dir,"logical_ci_trace.rds"))
receipt$authority_pmax <- numeric_comparison(result$pMax,readRDS(file.path(oracle_dir,"pmax.rds")),1e-10)
receipt$gates$authority_pmax_all_values <- receipt$authority_pmax$pass
receipt$gates$authority_semantic_sepsets <- identical(
  canonical_deletion_sepsets(result$tasks,"native_edge_deleted"),
  canonical_deletion_sepsets(oracle,"deletes_edge"))
receipt$gates$native_unchanged <- identical(sha(native),native_sha)
receipt$gates$source_unchanged <- identical(source_sha,setNames(vapply(sources,sha,""),sources))
receipt$gates$requested_mode <- identical(s$level_component_prefill_enabled,mode=="on")
receipt$gates$complete_trace <- nrow(result$tasks)==240498L
receipt$gates$exact_primary <- s$cuda_multi_penalty_target_count==80751 &&
  s$cuda_multi_penalty_exact_primary_target_count==80751 &&
  s$cuda_multi_penalty_stability_replay_target_count==0 &&
  s$cuda_multi_penalty_stability_replay_selected_count==0 &&
  s$cuda_multi_penalty_terminal_confirmation_count==0
receipt$gates$stable_svd <- s$cuda_exact_fixed_evaluator_target_count==
  s$cuda_exact_fixed_evaluator_stable_svd_target_count
receipt$gates$no_approximate_gamma <- s$guarded_pair_count==0 &&
  s$cuda_exact_screen_pair_count==0 && s$cuda_guard_refinement_pair_count==0 &&
  s$cuda_legacy_compatible_pair_count==s$physical_tests_evaluated
receipt$gates$cache_metadata <- s$legacy_component_cache_route_mismatch_count==0 &&
  s$legacy_component_cache_status_mismatch_count==0
receipt$gates$arena_closed <- s$exact_primary_fixed_reuse_allocation_count==
  s$exact_primary_fixed_reuse_free_count && s$exact_primary_fixed_reuse_live_device_bytes_after==0 &&
  s$exact_primary_fixed_reuse_fallback_target_count==0
if(mode=="on") {
  receipt$gates$same_native <- identical(native_sha,reference$native_sha256)
  receipt$gates$physical_groups_exact <- identical(s$physical_group_sequence_sha256,
    reference$result$summary$physical_group_sequence_sha256)
  receipt$gates$gram_keys_and_order_exact <- identical(s$gram_resolution_sequence_sha256,
    reference$result$summary$gram_resolution_sequence_sha256)
  rows <- s$level_component_prefill
  receipt$gates$level_ledgers <- length(rows)==6L && all(vapply(rows,function(r)
    r$planned==r$produced && r$produced==r$residual_targets &&
    r$resolved+r$pending_at_end==r$planned,logical(1L)))
  receipt$gates$full_coverage <- s$level_component_produced==80751 &&
    s$level_component_resolved==65694 && s$exact_primary_fixed_reuse_hit_target_count==0 &&
    sum(vapply(rows,`[[`,numeric(1L),"resolve_batches"))==15264
}
bits_differ <- function(a,b) {
  stopifnot(length(a)==length(b))
  colSums(matrix(writeBin(as.double(a),raw(),size=8)!=writeBin(as.double(b),raw(),size=8),nrow=8))>0
}
different <- bits_differ(result$tasks$p_used,reference$result$tasks$p_used)
changed <- result$tasks[different,,drop=FALSE]
changed$baseline_p <- reference$result$tasks$p_used[different]
changed$absolute_difference <- abs(changed$p_used-changed$baseline_p)
saveRDS(changed,file.path(out,"changed-ci.rds"),compress=FALSE)
write.csv(changed,file.path(out,"changed-ci.csv"),row.names=FALSE)
receipt$comparisons$changed_ci_count <- sum(different)
receipt$comparisons$max_ci_difference <- max(abs(result$tasks$p_used-reference$result$tasks$p_used))
receipt$comparisons$changed_pmax_count <- sum(bits_differ(result$pMax,reference$result$pMax))
if(length(args)==4L) {
  timing_reference <- readRDS(args[[4L]])
  receipt$gates$observer_matches_timing_p <- identical(result$tasks$p_used,
    timing_reference$result$tasks$p_used,num.eq=FALSE,single.NA=FALSE)
  receipt$gates$observer_matches_timing_pmax <- identical(result$pMax,
    timing_reference$result$pMax,num.eq=FALSE,single.NA=FALSE)
  receipt$gates$observer_matches_timing_groups <- identical(s$physical_group_sequence_sha256,
    timing_reference$result$summary$physical_group_sequence_sha256)
}
saveRDS(receipt,file.path(out,"receipt.rds"),compress=FALSE)
jsonlite::write_json(list(schema=receipt$schema,mode=mode,native_sha256=native_sha,
  source_sha256=as.list(source_sha),runner_sha256=sha("fastkpc/tools/research/level_component_prefill_canonical_20261021/run.R"),
  reference_sha256=sha(baseline_path),receipt_sha256=sha(file.path(out,"receipt.rds")),
  elapsed_sec=elapsed,gates=receipt$gates,oracle_errors=receipt$oracle_errors,
  authority_pmax=receipt$authority_pmax,comparisons=receipt$comparisons,
  evidence_class=receipt$evidence_class,level_component_prefill=s$level_component_prefill),
  file.path(out,"report.json"),pretty=TRUE,auto_unbox=TRUE,digits=NA)
print(receipt$gates); print(receipt$comparisons); print(receipt$oracle_errors)
cat("elapsed",elapsed,"seconds; logical CI",nrow(result$tasks),"\n")
stopifnot(all(unlist(receipt$gates)))
