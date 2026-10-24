# Configuration failures must leave the original GPU resource state intact.
source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
data <- as.matrix(readRDS("fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"))
fields <- c("live_device_allocations","live_device_bytes","live_streams","live_events")
reject <- function(pattern) {
  invisible(full_cuda_ci_one_call_cache_control_native("reset"))
  before <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()[fields]
  message <- tryCatch({
    fastkpc_compatible_cuda_skeleton(data,alpha=.1,options=list(route="full_cuda",
      compatible_cuda_strict=TRUE,max_conditioning_size=0L,index=1,numCol=35L,
      trace_level="logical",ci_method="dcc.gamma"))
    "unexpected success"
  },error=conditionMessage)
  stopifnot(grepl(pattern,message,fixed=TRUE),
    identical(before,fastkpc_full_cuda_phase35_vertical_resource_snapshot()[fields]))
}
Sys.setenv(FASTKPC_DCC_GAMMA_L5_OPTIMIZER_SLOTS="invalid")
reject("must be 0 or 1")
Sys.setenv(FASTKPC_DCC_GAMMA_L5_OPTIMIZER_SLOTS="1",
  FASTKPC_DCC_GAMMA_LEVEL_COMPONENT_PREFILL="1",
  FASTKPC_DCC_GAMMA_STRUCTURED_COMPONENT_PREFILL="1",
  FASTKPC_DCC_GAMMA_L2_COMPONENT_PREFILL="1",
  FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H="0")
reject("L5 optimizer slots require canonical S2")
Sys.setenv(FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H="1",
  FASTKPC_DCC_GAMMA_EXACT_PRIMARY_FIXED_REUSE="0")
reject("L5 optimizer slots require canonical S2")
cat("PASS L5 slot configuration rejection and resource closure\n")
