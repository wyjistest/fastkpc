source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_phase10_development_500x50.R")
for (file in c("fastkpc/tests/test_compatible_cuda_wanpdag_authority.R",
               "fastkpc/tools/research/real_data_parity_20261102/run.R"))
  for (e in parse(file))
    if (is.call(e) && identical(e[[1L]], as.name("<-")) && is.symbol(e[[2L]]) &&
        as.character(e[[2L]]) %in% c("normalize_sepsets", "trace_frame", "numeric_check",
          "strict_p", "orientation_reference", "compare_orientation")) eval(e)

rank_flags <- c("FASTKPC_DCC_GAMMA_LEVEL_COMPONENT_PREFILL", "FASTKPC_DCC_GAMMA_STRUCTURED_COMPONENT_PREFILL",
  "FASTKPC_DCC_GAMMA_L2_COMPONENT_PREFILL", "FASTKPC_DCC_GAMMA_L3_OPTIMIZER_SLOTS",
  "FASTKPC_DCC_GAMMA_L4_OPTIMIZER_WORKERS32", "FASTKPC_DCC_GAMMA_L4_OPTIMIZER_SLOTS",
  "FASTKPC_DCC_GAMMA_L5_OPTIMIZER_SLOTS", "FASTKPC_DCC_GAMMA_L6_OPTIMIZER_SLOTS",
  "FASTKPC_DCC_GAMMA_VECTOR_POSTBACK", "FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H",
  "FASTKPC_DCC_GAMMA_STRUCTURED_WORK_PAIR")
rank_mode <- function(mode) {
  do.call(Sys.setenv, as.list(setNames(rep(if (mode == "on") "1" else "0", length(rank_flags)), rank_flags)))
  Sys.setenv(FASTKPC_DCC_GAMMA_EXACT_PRIMARY_OPTIMIZER = "1",
    FASTKPC_DCC_GAMMA_EXACT_PRIMARY_FIXED_REUSE = "1",
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES = "11811160064",
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS = "1",
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES = "8")
  if (mode == "uncached") Sys.setenv(FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES = "0",
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS = "0",
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES = "0")
}

rank_data <- function(case) {
  real_paths <- c(
    real179 = "causalDiscovery/20221022181603/CD8.c11.Tex.PDCD1-causalDiscoveryInput.RData",
    real243 = "causalDiscovery/20220418113957/logged-liver-exhausted-allData-causalDiscoveryInput.RData",
    real300 = "causalDiscovery/20240315113222/CD8.c11.Tex.PDCD1-causalDiscoveryInput.RData")
  if (case %in% names(real_paths)) {
    payload <- new.env(parent = emptyenv()); loaded <- load(real_paths[[case]], envir = payload)
    stopifnot(length(loaded) == 1L)
    data <- as.matrix(payload[[loaded]])
  } else if (case == "public500") {
    data <- fastkpc_full_cuda_phase10_development_500x50_data()
  } else {
    settings <- list(dense500 = c(500L, 8L), large1025 = c(1025L, 4L),
      dense140 = c(140L, 6L), dense133 = c(133L, 6L), dense134 = c(134L, 6L))
    stopifnot(case %in% names(settings))
    n <- settings[[case]][[1L]]; p <- settings[[case]][[2L]]
    set.seed(n * 100L + p)
    z <- rnorm(n)
    data <- matrix(z, n, p) + .2 * matrix(rnorm(n * p), n, p)
    colnames(data) <- paste0("rank_v", seq_len(p))
  }
  storage.mode(data) <- "double"
  data
}

rank_live <- c("live_device_allocations", "live_device_bytes", "live_streams", "live_events")
rank_authority <- c("r_callback_count", "legacy_mgcv_fit_count", "legacy_mgcv_setup_count",
  "cpu_residual_solve_count", "cpu_dcov_component_count", "cpu_dcov_eigen_or_lowrank_count",
  "cpu_dcov_pair_stat_count", "cpu_gamma_pvalue_count", "cpu_spectra_count",
  "residual_d2h_bytes", "component_d2h_bytes", "unknown_fallback_count", "approximate_backend_count")
rank_resources <- function() {
  r <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
  stopifnot(all(rank_live %in% names(r)), all(unlist(r[rank_live]) == 0),
    fixed_sp_cuda_live_owner_snapshot()$total == 0)
  r
}
