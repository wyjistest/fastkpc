# Pin the cohort-width class boundary: widths 16..20 on a few cases, comparing
# the first sixteen targets' fitted digests with the pair cohort (class A).
source("fastkpc/tools/research/guarded_qr_20261020/common.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
args <- commandArgs(TRUE)
stopifnot(length(args) == 1L)
out <- args[[1L]]
native <- "fastkpc/build/fastkpc_cuda.so"
native_sha <- file_sha(native)
stopifnot(identical(native_sha, "fe8286f7e6c6114ef66e040b16b693f5abcb23333dc9c0fd767c042c509c3c50"))
data48 <- as.matrix(readRDS("fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"))
storage.mode(data48) <- "double"
datasets <- list(`22` = data48[, seq_len(22L), drop = FALSE], `48` = data48)
case_files <- file.path(out, "cases", sprintf("%03d.rds", c(1L, 2L, 3L, 25L, 31L, 35L)))
Sys.setenv(FASTKPC_TEST_MULTI_PENALTY_EXACT_ONLY = "1")
load_fastkpc_cuda_native()
widths <- 16:20
rows <- list()
for (path in case_files) {
  d <- readRDS(path)
  data <- datasets[[d$case$data]]
  S <- d$case$S
  setup <- full_cuda_ci_native_setup_native(data[, S, drop = FALSE])
  p <- fastkpc_full_cuda_phase6_prepare(make_setup(setup))
  stopifnot(identical(sha(p), d$geometry_sha256))
  base <- d$shapes$prefix_16
  base <- c(base, setdiff(d$shapes$all_natural, base))
  pair_digest <- d$digests[d$digests$cohort == "pair" & d$digests$quantity == "optimizer_fitted", c("target", "sha256")]
  wide_digest <- d$digests[d$digests$cohort == "all_natural" & d$digests$quantity == "optimizer_fitted", c("target", "sha256")]
  for (w in widths) {
    if (w > length(base)) next
    targets <- base[seq_len(w)]
    Y <- unname(data[, targets, drop = FALSE])
    z <- .Call("C_full_cuda_ci_multi_penalty_capacity_qualify",
      p$X, p$magic_qr_packed, as.double(p$magic_tau), p$magic_r,
      as.integer(p$magic_pivot), p$penalty_roots, p$penalty_matrices,
      as.integer(p$penalty_ranks), as.double(p$initial_log_sp), Y, NULL,
      PACKAGE = "fastkpc_cuda")
    stopifnot(all(z$optimizer_solver_info == 0L), all(z$fixed_solver_info == 0L))
    digests <- vapply(seq_len(w), function(j) sha(bits(z$optimizer_fitted[, j])), "")
    first_two <- targets[1:2]
    same_as_pair <- digests[1:2] == pair_digest$sha256[match(first_two, pair_digest$target)]
    same_as_wide <- digests[1:2] == wide_digest$sha256[match(first_two, wide_digest$target)]
    rows[[length(rows) + 1L]] <- data.frame(case = basename(path), level = d$case$level, q = ncol(p$X), width = w,
      pair_targets_match_narrow_class = all(same_as_pair), pair_targets_match_wide_class = all(same_as_wide),
      stringsAsFactors = FALSE)
    cat(sprintf("%s level %d q %d width %2d: matches narrow(pair) %s, matches wide(all) %s\n", basename(path), d$case$level, ncol(p$X), w, all(same_as_pair), all(same_as_wide)))
  }
}
res <- do.call(rbind, rows)
write.csv(res, file.path(out, "width_threshold.csv"), row.names = FALSE)
saveRDS(list(native_sha256 = native_sha, rows = res), file.path(out, "width_threshold.rds"))
