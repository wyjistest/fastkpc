# Actual 351x48 L2 producer/consumer smoke; GPU S3 is the comparison producer.
args <- commandArgs(TRUE)
stopifnot(length(args)==1L)
source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
source("fastkpc/tools/research/exact_speedup_20261020/pinned_optimizer_gates.R")
data <- as.matrix(readRDS("fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"))
storage.mode(data) <- "double"
set.seed(20261022L)
for(j in 9:48) data[,j] <- data[sample.int(nrow(data)),j]
Sys.setenv(FASTKPC_DCC_GAMMA_LEVEL_COMPONENT_PREFILL="1",
  FASTKPC_DCC_GAMMA_STRUCTURED_COMPONENT_PREFILL="1",FASTKPC_DIAG_LEVEL_COMPONENT_ORDER="1")
live <- c("live_device_allocations","live_device_bytes","live_streams","live_events")
run <- function(l2) {
  cat("START runtime smoke L2=",l2,"\n",sep=""); flush.console()
  Sys.setenv(FASTKPC_DCC_GAMMA_L2_COMPONENT_PREFILL=as.character(l2))
  invisible(full_cuda_ci_one_call_cache_control_native("configure",262144L))
  invisible(full_cuda_ci_one_call_cache_control_native("configure_target",131072L))
  invisible(full_cuda_ci_one_call_cache_control_native("reset"))
  set.seed(20261020L); seed <- .Random.seed
  before <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
  started <- proc.time()[["elapsed"]]
  result <- fastkpc_compatible_cuda_skeleton(data,alpha=.1,options=list(
    route="full_cuda",compatible_cuda_strict=TRUE,max_conditioning_size=2L,
    index=1,numCol=35L,trace_level="logical",ci_method="dcc.gamma"))
  receipt <- list(workload="l2-smoke",result=result,seed_before=seed,seed_after=.Random.seed,
    elapsed_sec=proc.time()[["elapsed"]]-started,resources_before=before,
    resources_after=fastkpc_full_cuda_phase35_vertical_resource_snapshot())
  receipt <- qualify_pinned_optimizer_receipt(receipt)
  stopifnot(all(unlist(receipt$gates)),all(unlist(receipt$resources_after[live])==0))
  cat("PASS runtime smoke L2=",l2," numerics/resources\n",sep=""); flush.console()
  receipt
}
s3 <- run(0L); s2 <- run(1L)
a <- s3$result; b <- s2$result
fields <- c("level","task_index","x","y","S_key","canonical_test_order_id",
  "native_edge_deleted","native_edge_ignored")
stopifnot(identical(a$tasks[fields],b$tasks[fields]),identical(a$adjacency,b$adjacency),
  identical(a$sepsets,b$sepsets),identical(a$n.edgetests,b$n.edgetests),
  identical(s3$seed_after,s2$seed_after),
  identical(a$summary$physical_group_sequence_sha256,b$summary$physical_group_sequence_sha256),
  identical(a$summary$gram_resolution_sequence_sha256,b$summary$gram_resolution_sequence_sha256),
  max(abs(a$tasks$p_used-b$tasks$p_used))<=1e-10,
  identical(is.finite(a$pMax),is.finite(b$pMax)),
  max(abs(a$pMax[is.finite(a$pMax)]-b$pMax[is.finite(b$pMax)]))<=1e-10)
rows <- b$summary$level_component_prefill
stopifnot(length(rows)==1L,rows[[1L]]$level==2L,rows[[1L]]$planned>0,
  rows[[1L]]$planned==rows[[1L]]$produced,
  rows[[1L]]$produced==rows[[1L]]$residual_targets,
  b$summary$l2_component_prepared_count>0,b$summary$l2_component_residual_batches>0,
  b$summary$l2_component_residual_targets==rows[[1L]]$produced,
  b$summary$exact_primary_fixed_reuse_coefficient_gather_copy_count==0)
l2 <- Filter(function(r)r$level==2,b$summary$level_component_consumers)[[1L]]
stopifnot(l2$batches==l2$all_hit_batches,l2$mixed_batches==0,l2$all_miss_batches==0,
  l2$residual_targets==0,l2$residual_host_ms==0,l2$full_eig_cuda_ms==0,
  l2$targets==l2$bypassed_targets)
saveRDS(list(s3=s3,s2=s2),file.path(args[[1L]],"smoke.rds"),compress=FALSE)
jsonlite::write_json(list(status="PASS",logical_ci=nrow(b$tasks),
  max_p_difference=max(abs(a$tasks$p_used-b$tasks$p_used)),
  elapsed_sec=c(s3=s3$elapsed_sec,s2=s2$elapsed_sec),l2=l2,
  prefill=rows,resource_closed=TRUE),file.path(args[[1L]],"smoke.json"),
  pretty=TRUE,auto_unbox=TRUE,digits=NA)
cat("PASS runtime smoke:",nrow(b$tasks),"CI; L2",rows[[1L]]$produced,"components\n")
