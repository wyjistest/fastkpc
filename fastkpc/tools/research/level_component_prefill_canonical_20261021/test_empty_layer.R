args <- commandArgs(TRUE)
stopifnot(length(args)%in%c(2L,3L),args[[2L]]%in%c("off","on"),!file.exists(args[[1L]]))
Sys.setenv(FASTKPC_DCC_GAMMA_LEVEL_COMPONENT_PREFILL=if(args[[2L]]=="on") "1" else "0")
source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
source("fastkpc/tools/research/exact_speedup_20261020/pinned_optimizer_gates.R")
data <- as.matrix(readRDS("fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"))[,1:22]
invisible(full_cuda_ci_one_call_cache_control_native("configure",262144L))
invisible(full_cuda_ci_one_call_cache_control_native("configure_target",131072L))
invisible(full_cuda_ci_one_call_cache_control_native("reset"))
set.seed(20261020L); seed <- .Random.seed
before <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
result <- fastkpc_compatible_cuda_skeleton(data,alpha=.1,options=list(route="full_cuda",
  compatible_cuda_strict=TRUE,max_conditioning_size=Inf,index=1,numCol=35L,
  trace_level="logical",ci_method="dcc.gamma"))
receipt <- list(workload="injected-empty-level3",seed_before=seed,seed_after=.Random.seed,
  resources_before=before,resources_after=fastkpc_full_cuda_phase35_vertical_resource_snapshot(),result=result)
receipt <- qualify_pinned_optimizer_receipt(receipt,if(length(args)==3L) args[[3L]] else NULL)
print(result$levels)
stopifnot(tail(result$levels$level,1)==3L,tail(result$n.edgetests,1)==0L,
  result$summary$level_component_produced==0,length(result$summary$level_component_prefill)==0L,
  all(unlist(receipt$gates)))
saveRDS(receipt,args[[1L]],compress=FALSE)
cat("PASS injected empty level 3, zero prefill lifecycle, graph parity and resources\n")
