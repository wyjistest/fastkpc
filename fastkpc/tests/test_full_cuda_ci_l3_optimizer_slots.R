# Reject unsupported L3 slot opt-ins before numerical work and without leaks.
source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
data <- as.matrix(readRDS("fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"))
fields <- c("live_device_allocations","live_device_bytes","live_streams","live_events")
base <- c(FASTKPC_DCC_GAMMA_L3_OPTIMIZER_SLOTS="1",
  FASTKPC_DCC_GAMMA_L6_OPTIMIZER_SLOTS="1",
  FASTKPC_CUDA_DEVICE_PROPERTIES_CACHE="1",
  FASTKPC_DCC_GAMMA_L4_OPTIMIZER_SLOTS="1",
  FASTKPC_DCC_GAMMA_L4_OPTIMIZER_WORKERS32="1",
  FASTKPC_DCC_GAMMA_L5_OPTIMIZER_SLOTS="1",
  FASTKPC_DCC_GAMMA_LEVEL_COMPONENT_PREFILL="1",
  FASTKPC_DCC_GAMMA_STRUCTURED_COMPONENT_PREFILL="1",
  FASTKPC_DCC_GAMMA_L2_COMPONENT_PREFILL="1",
  FASTKPC_DCC_GAMMA_EXACT_PRIMARY_FIXED_REUSE="1",
  FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H="1",
  FASTKPC_PHASE10_SETUP_OPTIMIZER_PIPELINE="0")
reject <- function(changes,pattern="L3 optimizer slots require canonical S2",input=data) {
  do.call(Sys.setenv,as.list(base));do.call(Sys.setenv,as.list(changes))
  invisible(full_cuda_ci_one_call_cache_control_native("reset"))
  before <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()[fields]
  message <- tryCatch({
    fastkpc_compatible_cuda_skeleton(input,alpha=.1,options=list(route="full_cuda",
      compatible_cuda_strict=TRUE,max_conditioning_size=0L,index=1,numCol=35L,
      trace_level="logical",ci_method="dcc.gamma"))
    "unexpected success"
  },error=conditionMessage)
  stopifnot(grepl(pattern,message,fixed=TRUE),
    identical(before,fastkpc_full_cuda_phase35_vertical_resource_snapshot()[fields]))
}
reject(c(FASTKPC_DCC_GAMMA_L3_OPTIMIZER_SLOTS="invalid"),"must be 0 or 1")
reject(c(FASTKPC_CUDA_DEVICE_PROPERTIES_CACHE="0"))
reject(c(FASTKPC_DCC_GAMMA_L4_OPTIMIZER_SLOTS="0"))
reject(c(FASTKPC_DCC_GAMMA_L5_OPTIMIZER_SLOTS="0"))
reject(c(FASTKPC_DCC_GAMMA_L6_OPTIMIZER_SLOTS="0"))
reject(c(FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H="0"))
reject(c(FASTKPC_DCC_GAMMA_EXACT_PRIMARY_FIXED_REUSE="0"))
reject(c(FASTKPC_DCC_GAMMA_L2_COMPONENT_PREFILL="0"))
reject(c(FASTKPC_PHASE10_SETUP_OPTIMIZER_PIPELINE="1"))
reject(c(FASTKPC_DCC_GAMMA_LEVEL_COMPONENT_PREFILL="0",
  FASTKPC_DCC_GAMMA_STRUCTURED_COMPONENT_PREFILL="0",
  FASTKPC_DCC_GAMMA_L2_COMPONENT_PREFILL="0"),input=data[-1L,])
do.call(Sys.setenv,as.list(base))
Sys.unsetenv("FASTKPC_DCC_GAMMA_L3_OPTIMIZER_SLOTS")
invisible(full_cuda_ci_one_call_cache_control_native("reset"))
before <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()[fields]
default <- fastkpc_compatible_cuda_skeleton(data,alpha=.1,options=list(route="full_cuda",
  compatible_cuda_strict=TRUE,max_conditioning_size=0L,index=1,numCol=35L,
  trace_level="logical",ci_method="dcc.gamma"))
stopifnot(identical(default$summary$l3_optimizer_slots$enabled,FALSE),
  default$summary$l3_optimizer_slots$created==0,
  identical(before,fastkpc_full_cuda_phase35_vertical_resource_snapshot()[fields]))
cat("PASS ten L3 slot configuration rejection cases, default-off and resource closure\n")
