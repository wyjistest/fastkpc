# Narrow configuration and fail-closed gates, with no numerical CPU producer.
source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
data<-as.matrix(readRDS("fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"))
storage.mode(data)<-"double"
fields<-c("live_device_allocations","live_device_bytes","live_streams","live_events")
reject<-function(pattern,p=48L,n=351L,method="dcc.gamma") {
  invisible(full_cuda_ci_one_call_cache_control_native("reset"))
  before<-fastkpc_full_cuda_phase35_vertical_resource_snapshot()[fields]
  message<-tryCatch({
    fastkpc_compatible_cuda_skeleton(data[seq_len(n),seq_len(p),drop=FALSE],alpha=.1,
      options=list(route="full_cuda",compatible_cuda_strict=TRUE,max_conditioning_size=0L,
        index=1,numCol=35L,trace_level="logical",ci_method=method))
    "unexpected success"
  },error=conditionMessage)
  stopifnot(grepl(pattern,message,fixed=TRUE),
    identical(before,fastkpc_full_cuda_phase35_vertical_resource_snapshot()[fields]))
}
Sys.setenv(FASTKPC_DCC_GAMMA_L2_COMPONENT_PREFILL="2")
reject("must be 0 or 1")
Sys.setenv(FASTKPC_DCC_GAMMA_L2_COMPONENT_PREFILL="1",
  FASTKPC_DCC_GAMMA_LEVEL_COMPONENT_PREFILL="1",FASTKPC_DCC_GAMMA_STRUCTURED_COMPONENT_PREFILL="0")
reject("L2 component prefill requires structured level prefill")
Sys.setenv(FASTKPC_DCC_GAMMA_STRUCTURED_COMPONENT_PREFILL="1")
reject("L2 component prefill requires structured level prefill",p=22L)
reject("L2 component prefill requires structured level prefill",n=350L)
reject("L2 component prefill requires structured level prefill",method="hsic.gamma")
Sys.setenv(FASTKPC_DCC_GAMMA_EXACT_PRIMARY_FIXED_REUSE="0")
reject("requires fresh caches and exact-primary reuse")
Sys.setenv(FASTKPC_DCC_GAMMA_EXACT_PRIMARY_FIXED_REUSE="1",
  FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS="0")
reject("requires cache/bypass/8 lanes and original scheduler")
cat("PASS L2 configuration rejection and resource closure\n")
