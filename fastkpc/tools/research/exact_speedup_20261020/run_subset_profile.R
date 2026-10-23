# Strict full-CUDA dcc.gamma skeleton on the first SUBSET_P columns of the
# canonical 351x48 input, max_conditioning_size = 3.  Uses the retained eig
# configuration of the canonical Phase 10 evidence (11 GiB component cache,
# residual bypass, 8 eig lanes) unless those variables are already set or
# FASTKPC_PROFILE_DEFAULT_ROUTE=1.  Run from the repository root:
#   CUDA_VISIBLE_DEVICES=0 SUBSET_P=22 Rscript fastkpc/tools/research/exact_speedup_20261020/run_subset_profile.R
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
X <- as.matrix(readRDS("fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"))
p <- as.integer(Sys.getenv("SUBSET_P", "30"))
opts <- list(route = "full_cuda", compatible_cuda_strict = TRUE,
             max_conditioning_size = 3, index = 1, numCol = 35L,
             trace_level = "logical")
t0 <- Sys.time()
sk <- fastkpc_compatible_cuda_skeleton(X[, seq_len(p)], alpha = 0.1, options = opts)
el <- as.numeric(Sys.time() - t0, units = "secs")
s <- sk$summary
cat(sprintf("ELAPSED %.3f s  p=%d\n", el, p))
for (nm in c("physical_tests_evaluated", "cuda_multi_penalty_target_count",
             "cuda_single_penalty_target_count", "cuda_multi_penalty_optimizer_setup_count",
             "cuda_multi_penalty_optimizer_host_ms", "cuda_multi_penalty_optimizer_summed_setup_host_ms",
             "cuda_multi_penalty_optimizer_max_setup_host_ms", "cuda_multi_penalty_qr_svd_cycles",
             "cuda_multi_penalty_derivative_hessian_cycles", "cuda_multi_penalty_penalty_factor_cycles",
             "cuda_multi_penalty_score_construction_cycles", "cuda_multi_penalty_optimizer_iteration_sum",
             "cuda_multi_penalty_optimizer_iteration_max", "cuda_multi_penalty_complete_evaluation_count",
             "cuda_multi_penalty_score_only_evaluation_count", "cuda_multi_penalty_stable_svd_evaluation_count",
             "cuda_dcov_full_eig_ms", "cuda_dcov_host_ms", "native_setup_ms", "cuda_residual_solve_host_ms",
             "cuda_optimizer_host_ms"))
  if (!is.null(s[[nm]])) cat(sprintf("%-52s %s\n", nm, format(s[[nm]])))
for (nm in grep("eig_lane|legacy_component_cache_(lookup|hit|miss)_count$",
                names(s), value = TRUE))
  cat(sprintf("%-52s %s\n", nm, format(s[[nm]])))
