# Strict full-CUDA dcc.gamma skeleton on the public 500x50 development
# fixture, printing the largest stage timers.  Uses the retained eig
# configuration (11 GiB component cache, residual bypass, 8 eig lanes) unless
# those variables are already set or FASTKPC_PROFILE_DEFAULT_ROUTE=1.
# Run from the repository root:
#   CUDA_VISIBLE_DEVICES=0 Rscript fastkpc/tools/research/exact_speedup_20261020/run_development_500x50.R
retained <- c(FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES = "11811160064",
              FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS = "1",
              FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES = "8")
if (!identical(Sys.getenv("FASTKPC_PROFILE_DEFAULT_ROUTE"), "1")) {
  unset <- retained[!nzchar(Sys.getenv(names(retained)))]
  if (length(unset)) do.call(Sys.setenv, as.list(unset))
}
cat("eig configuration:",
    paste0(names(retained), "=", Sys.getenv(names(retained)), collapse = " "),
    "\n")

suppressMessages(source("fastkpc/R/fast_kpc.R"))
x <- readRDS("fastkpc/tests/fixtures/full_cuda_ci_development_500x50_v2.rds")
opts <- list(route = "full_cuda", compatible_cuda_strict = TRUE,
             max_conditioning_size = Inf, index = 1, numCol = 35L,
             trace_level = "logical")
t0 <- Sys.time()
sk <- fastkpc_compatible_cuda_skeleton(x$data, alpha = 0.1, options = opts)
el <- as.numeric(Sys.time() - t0, units = "secs")
s <- sk$summary
cat(sprintf("ELAPSED %.3f s (fixture 20-core CPU baseline %.1f s)\n", el,
            x$baseline_elapsed_sec))
for (nm in c("physical_tests_evaluated", "cuda_multi_penalty_target_count",
             "cuda_single_penalty_target_count")) {
  cat(sprintf("%-40s %s\n", nm, format(s[[nm]])))
}
for (nm in grep("eig_lane|legacy_component_cache_(lookup|hit|miss)_count$",
                names(s), value = TRUE))
  cat(sprintf("%-40s %s\n", nm, format(s[[nm]])))
ms <- unlist(s[grep("_ms$", names(s))])
print(round(head(sort(ms[ms > 100], decreasing = TRUE), 12), 1))
