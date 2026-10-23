# Gate 0 (p-level): with one fixed residual token per S, does the legacy eig
# batch shape (component batch width, pair count, pair order) change the
# p-value / statistic / mean / variance bits of a pair?  The component build
# forms Gram matrices with cublasDgemmStridedBatched(batchCount = miss_count)
# and the pair stage uses cublasDgemmBatched(batchCount = pair_count), so this
# is a necessary condition for a level-wide component prefill to reproduce
# the current route bitwise.  Only accepted-native public entries are used.
source("fastkpc/tools/research/guarded_qr_20261020/common.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
args <- commandArgs(TRUE)
stopifnot(length(args) == 1L)
out <- args[[1L]]
dir.create(out, recursive = TRUE, showWarnings = FALSE)
native <- "fastkpc/build/fastkpc_cuda.so"
native_sha <- file_sha(native)
accepted_native <- "fe8286f7e6c6114ef66e040b16b693f5abcb23333dc9c0fd767c042c509c3c50"
stopifnot(identical(native_sha, accepted_native))
data48 <- as.matrix(readRDS("fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"))
storage.mode(data48) <- "double"
data <- data48[, seq_len(22L), drop = FALSE]
archive_dir <- "fastkpc/artifacts/full_cuda_ci/structured_real_20261020/cohorts"
archive_files <- sort(list.files(archive_dir, pattern = "\\.rds$", full.names = TRUE))
n_cases <- 12L
Sys.setenv(FASTKPC_TEST_MULTI_PENALTY_EXACT_ONLY = "1")
load_fastkpc_cuda_native()
before <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()

digest_cols <- function(z, j) sha(bits(c(z$records$p_value[j], z$numerical$statistic[j],
                                         z$numerical$mean[j], z$numerical$variance[j])))
record_row <- function(z, j) data.frame(p_value = z$records$p_value[j],
  statistic = z$numerical$statistic[j], mean = z$numerical$mean[j],
  variance = z$numerical$variance[j], stringsAsFactors = FALSE)

all_rows <- list()
case_summaries <- list()
for (ci in seq_len(n_cases)) {
  d <- readRDS(archive_files[[ci]])
  stopifnot(identical(d$native_sha256, accepted_native))
  S <- as.integer(strsplit(d$task$S_key, "|", fixed = TRUE)[[1L]])
  targets <- setdiff(seq_len(ncol(data)), S)          # 19 targets
  nt <- length(targets)
  setup <- full_cuda_ci_native_setup_native(data[, S, drop = FALSE])
  p <- fastkpc_full_cuda_phase6_prepare(make_setup(setup))
  stopifnot(identical(sha(p), d$geometry_sha256))
  Y <- unname(data[, targets, drop = FALSE])
  # selected log-SP for every target (coefficient/log-SP bits are cohort independent)
  z <- .Call("C_full_cuda_ci_multi_penalty_capacity_qualify",
    p$X, p$magic_qr_packed, as.double(p$magic_tau), p$magic_r,
    as.integer(p$magic_pivot), p$penalty_roots, p$penalty_matrices,
    as.integer(p$penalty_ranks), as.double(p$initial_log_sp), Y, NULL, PACKAGE = "fastkpc_cuda")
  stopifnot(all(z$optimizer_solver_info == 0L))
  selected <- z$selected_log_sp
  fixture <- list(id = sprintf("component-batch-shape-%03d", ci), setup = make_setup(setup), Y = Y, data_p = 48L)
  key <- sha(list(fixture$id, fixture$setup))
  keys <- vapply(seq_len(nt), function(t) sha(list(key, t, bits(selected[, t]))), "")
  dto <- make_dto(fixture$setup, key, fixture$data_p)
  multi <- fastkpc_full_cuda_phase6_prepared_create(p, nt)
  runtime <- fixed_sp_cuda_runtime_create(0L)
  fixed_sp_cuda_runtime_reserve(runtime, dto$n, dto$null_dim, nt, dto$penalty_count, dto$n + sum(dto$penalty_ranks))
  handle <- fixed_sp_cuda_prepared_create(runtime, dto)
  token <- full_cuda_ci_multi_penalty_gcv_evaluate_fixed_batch_native(multi, Y, selected, keys)
  info <- full_cuda_ci_multi_penalty_gcv_residual_info_native(token)
  stopifnot(isTRUE(info$fixed_sp_evaluation), all(info$solver_info == 0L), all(info$decomposition_route == 2L))
  residual_sha <- sha(bits(full_cuda_ci_multi_penalty_gcv_residual_shadow_native(token)$residuals))
  # pair sets: 9 disjoint pairs over the 19 targets (ordinals), plus all 171 pairs
  disjoint <- cbind(seq(1L, 17L, by = 2L), seq(2L, 18L, by = 2L))
  all_pairs <- t(combn(nt, 2L))
  shapes <- list(
    single_pairs = lapply(seq_len(nrow(disjoint)), function(i) disjoint[i, , drop = FALSE]),
    width4_pairs2 = list(disjoint[1:2, ], disjoint[3:4, ], disjoint[5:6, ], disjoint[7:8, ]),
    width8_pairs4 = list(disjoint[1:4, ], disjoint[5:8, ]),
    width16_pairs8 = list(disjoint[1:8, ]),
    width19_pairs9 = list(rbind(disjoint, c(19L, 1L))),
    width19_pairs171 = list(all_pairs),
    width19_pairs171_reversed = list(all_pairs[nrow(all_pairs):1, , drop = FALSE]),
    width19_pairs10_offset = list(rbind(disjoint[2:9, ], c(19L, 18L), c(17L, 16L)))
  )
  run_request <- function(pairs, label) {
    P <- nrow(pairs)
    referenced <- unique(c(pairs[, 1L], pairs[, 2L]))
    request <- fastkpc_full_cuda_phase35_legacy_eig_batch_request(
      key, keys, as.numeric(seq_len(P)), as.integer(pairs[, 1L]), as.integer(pairs[, 2L]),
      0.1, length(referenced), 35L)
    r <- fastkpc_full_cuda_phase35_legacy_eig_batch_from_optimizer_residual(handle, token, request)
    stopifnot(nrow(r$records) == P, all(r$records$status == "OK"), r$diagnostics$component_build_count == length(referenced),
      r$diagnostics$cuda_full_eig_count == length(referenced),
      r$diagnostics$component_d2h_bytes == 0, r$diagnostics$residual_d2h_bytes == 0)
    do.call(rbind, lapply(seq_len(P), function(j) {
      cbind(data.frame(case = ci, S_key = d$task$S_key, shape = label, width = length(referenced), pair_count = P,
        position = j, left = targets[pairs[j, 1L]], right = targets[pairs[j, 2L]],
        digest = digest_cols(r, j), stringsAsFactors = FALSE), record_row(r, j))
    }))
  }
  rows <- list()
  for (shape_name in names(shapes)) {
    for (k in seq_along(shapes[[shape_name]])) {
      label <- if (length(shapes[[shape_name]]) > 1L) sprintf("%s_%d", shape_name, k) else shape_name
      rows[[length(rows) + 1L]] <- run_request(shapes[[shape_name]][[k]], label)
    }
  }
  # repeat the single pairs once more to check run-to-run determinism on the same token
  for (i in seq_len(nrow(disjoint))) rows[[length(rows) + 1L]] <- run_request(disjoint[i, , drop = FALSE], sprintf("single_pairs_repeat_%d", i))
  full_cuda_ci_multi_penalty_gcv_residual_free_native(token)
  fixed_sp_cuda_prepared_free(handle)
  fixed_sp_cuda_runtime_free(runtime)
  full_cuda_ci_multi_penalty_gcv_prepared_free_native(multi)
  tab <- do.call(rbind, rows)
  tab$pair_key <- paste(pmin(tab$left, tab$right), pmax(tab$left, tab$right))
  per_pair <- aggregate(digest ~ pair_key, data = tab, FUN = function(x) length(unique(x)))
  multi_digest <- sum(per_pair$digest > 1L)
  # reference = single_pairs shape where available
  ref <- tab[grepl("^single_pairs_[0-9]+$", tab$shape), c("pair_key", "digest")]
  names(ref)[2] <- "ref_digest"
  cmp <- merge(tab, ref, by = "pair_key")
  mism <- cmp[cmp$digest != cmp$ref_digest, ]
  max_p_diff <- if (nrow(mism)) max(abs(mism$p_value - tab$p_value[match(mism$pair_key, tab$pair_key)])) else 0
  case_summaries[[ci]] <- data.frame(case = ci, S_key = d$task$S_key, q = ncol(p$X), targets = nt,
    requests = length(rows), pair_rows = nrow(tab), distinct_pairs = nrow(per_pair),
    pairs_with_multiple_digests = multi_digest,
    mismatching_rows_vs_single = nrow(mism), max_p_abs_diff_vs_single = max_p_diff,
    residual_sha256 = residual_sha, stringsAsFactors = FALSE)
  all_rows[[ci]] <- tab
  cat(sprintf("case %02d S=%s: requests %d, pairs %d, pairs with >1 digest %d, rows != single-pair reference %d (max |dp| %.3g)\n",
    ci, d$task$S_key, length(rows), nrow(per_pair), multi_digest, nrow(mism), max_p_diff))
  flush.console()
}
after <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
fields <- c("live_device_allocations", "live_device_bytes", "live_streams", "live_events")
stopifnot(identical(before[fields], after[fields]), identical(file_sha(native), native_sha))
summary <- do.call(rbind, case_summaries)
rows <- do.call(rbind, all_rows)
write.csv(summary, file.path(out, "summary.csv"), row.names = FALSE)
write.csv(rows, file.path(out, "pair_rows.csv"), row.names = FALSE)
saveRDS(list(schema = "fastkpc-component-batch-shape-v1", native_sha256 = native_sha,
  environment = Sys.getenv(c("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_MAX_CONNECTIONS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "FASTKPC_TEST_MULTI_PENALTY_EXACT_ONLY")),
  resources_before = before, resources_after = after, summary = summary, rows = rows), file.path(out, "results.rds"), compress = FALSE)
cat("done\n")
