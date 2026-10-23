# Reject unsupported prototype configurations before numerical production and
# prove that each rejection releases all native CUDA resources.
source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
data <- as.matrix(readRDS("fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"))[,1:22,drop=FALSE]
storage.mode(data) <- "double"
live <- c("live_device_allocations", "live_device_bytes", "live_streams", "live_events")
reject <- function(pattern, p=22L) {
  invisible(full_cuda_ci_one_call_cache_control_native("reset"))
  before <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()[live]
  message <- tryCatch({
    fastkpc_compatible_cuda_skeleton(data[,seq_len(p),drop=FALSE], alpha=.1,
      options=list(route="full_cuda", compatible_cuda_strict=TRUE,
        max_conditioning_size=0L,index=1,numCol=35L,trace_level="logical",ci_method="dcc.gamma"))
    "unexpected success"
  }, error=function(e) conditionMessage(e))
  stopifnot(grepl(pattern,message,fixed=TRUE),
    identical(before,fastkpc_full_cuda_phase35_vertical_resource_snapshot()[live]))
}
Sys.setenv(FASTKPC_DCC_GAMMA_LEVEL_COMPONENT_PREFILL="2")
reject("must be 0 or 1")
Sys.setenv(FASTKPC_DCC_GAMMA_LEVEL_COMPONENT_PREFILL="1")
reject("restricted to dcc.gamma 351x22",21L)
Sys.setenv(FASTKPC_DCC_GAMMA_EXACT_PRIMARY_FIXED_REUSE="0")
reject("requires fresh caches and exact-primary reuse")
Sys.setenv(FASTKPC_DCC_GAMMA_EXACT_PRIMARY_FIXED_REUSE="1",
  FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES="0",
  FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS="0",
  FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES="0")
reject("requires cache/bypass/8 lanes and original scheduler")
Sys.setenv(FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES="1048576",
  FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS="1")
reject("requires cache/bypass/8 lanes and original scheduler")
cat("PASS level component prefill configuration rejection and resource closure\n")
