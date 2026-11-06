source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
data<-as.matrix(readRDS("fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"))
storage.mode(data)<-"double"
fields<-c("live_device_allocations","live_device_bytes","live_streams","live_events")
reject<-function(pattern,p=22L,k=35L) {
  invisible(full_cuda_ci_one_call_cache_control_native("reset"))
  before<-fastkpc_full_cuda_phase35_vertical_resource_snapshot()[fields]
  message<-tryCatch({
    fastkpc_compatible_cuda_skeleton(data[,seq_len(p),drop=FALSE],alpha=.1,
      options=list(route="full_cuda",compatible_cuda_strict=TRUE,max_conditioning_size=0L,
        index=1,numCol=k,trace_level="logical",ci_method="dcc.gamma"))
    "unexpected success"
  },error=conditionMessage)
  stopifnot(grepl(pattern,message,fixed=TRUE),
    identical(before,fastkpc_full_cuda_phase35_vertical_resource_snapshot()[fields]))
}
Sys.setenv(FASTKPC_DCC_GAMMA_STRUCTURED_COMPONENT_PREFILL="2")
reject("must be 0 or 1")
Sys.setenv(FASTKPC_DCC_GAMMA_STRUCTURED_COMPONENT_PREFILL="1",
           FASTKPC_DCC_GAMMA_LEVEL_COMPONENT_PREFILL="0")
reject("requires level prefill and dcc.gamma")
Sys.setenv(FASTKPC_DCC_GAMMA_LEVEL_COMPONENT_PREFILL="1")
reject("numCol must be one integer, positive and less than sample size",22L,0L)
Sys.setenv(FASTKPC_DCC_GAMMA_EXACT_PRIMARY_FIXED_REUSE="0")
reject("requires fresh caches and exact-primary reuse")
reject("requires fresh caches and exact-primary reuse",48L)
Sys.setenv(FASTKPC_DCC_GAMMA_EXACT_PRIMARY_FIXED_REUSE="1",
  FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES="1048576",
  FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS="1",
  FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES="8")
invisible(full_cuda_ci_one_call_cache_control_native("reset"))
value <- fastkpc_compatible_cuda_skeleton(data[1:350,1:23],alpha=.1,
  options=list(route="full_cuda",compatible_cuda_strict=TRUE,max_conditioning_size=0L,
    index=1,numCol=35L,trace_level="logical",ci_method="dcc.gamma"))
stopifnot(isTRUE(value$summary$structured_component_prefill_enabled),
  all(unlist(fastkpc_full_cuda_phase35_vertical_resource_snapshot()[fields]) == 0))
cat("PASS structured opt-in configuration rejection and resource closure\n")
