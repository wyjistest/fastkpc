# Residual cohort bit-identity experiment (multi-penalty route, levels >= 3).
#
# Question: with the same prepared-S geometry (X bits), the same Y columns and
# the same exact-primary coefficient bits, does the cohort shape (target count
# and column position) of the residual materialization change the fitted or
# residual bytes?  The production finalizer passes target_count straight into
# cublasDgemm(n, target_count, q), so this is a GEMM shape question.
#
# Only the accepted native and its public capacity-qualification entry are
# used.  No production source, object or library is modified.  Every cohort is
# an independent call: optimizer (one block per target) + cohort-wide GEMM
# finalization + exact-fixed stable-SVD evaluation, with full host shadows.
source("fastkpc/tools/research/guarded_qr_20261020/common.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
args <- commandArgs(TRUE)
stopifnot(length(args) == 1L)
out <- args[[1L]]
dir.create(out, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out, "cases"), showWarnings = FALSE)
native <- "fastkpc/build/fastkpc_cuda.so"
native_sha <- file_sha(native)
accepted_native <- "fe8286f7e6c6114ef66e040b16b693f5abcb23333dc9c0fd767c042c509c3c50"
stopifnot(identical(native_sha, accepted_native))
data_path <- "fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"
data48 <- as.matrix(readRDS(data_path))
storage.mode(data48) <- "double"
stopifnot(identical(dim(data48), c(351L, 48L)))
datasets <- list(`22` = data48[, seq_len(22L), drop = FALSE], `48` = data48)

# ---- case selection: real S identities ---------------------------------
archive_dir <- "fastkpc/artifacts/full_cuda_ci/structured_real_20261020/cohorts"
archive_files <- sort(list.files(archive_dir, pattern = "\\.rds$", full.names = TRUE))
stopifnot(length(archive_files) == 320L)
n_archive <- 24L
cases <- lapply(archive_files[seq_len(n_archive)], function(path) {
  d <- readRDS(path)
  stopifnot(identical(d$native_sha256, accepted_native), d$task$level == 3L)
  list(source = basename(path), data = "22", level = 3L,
       S = as.integer(strsplit(d$task$S_key, "|", fixed = TRUE)[[1L]]),
       pair = c(d$task$x, d$task$y), archived = d)
})
canonical_path <- "fastkpc/artifacts/full_cuda_ci/exact_speedup_20261020/canonical-queues32.rds"
canon <- readRDS(canonical_path)
canon_sha <- file_sha(canonical_path)
tasks <- canon$result$tasks
set.seed(20261021L)
deep_counts <- c(`4` = 6L, `5` = 4L, `6` = 3L, `7` = 2L, `8` = 1L)
for (L in names(deep_counts)) {
  tl <- tasks[tasks$level == as.integer(L), ]
  tl <- tl[!duplicated(tl$S_key), ]
  pick <- tl[sort(sample.int(nrow(tl), min(nrow(tl), deep_counts[[L]]))), ]
  for (i in seq_len(nrow(pick))) {
    cases[[length(cases) + 1L]] <- list(
      source = sprintf("canonical-queues32 test %d", pick$canonical_test_order_id[i]),
      data = "48", level = as.integer(L),
      S = as.integer(strsplit(pick$S_key[i], "|", fixed = TRUE)[[1L]]),
      pair = c(pick$x[i], pick$y[i]), archived = NULL)
  }
}
cat(sprintf("%d cases (%d archived level-3 pairs, %d deeper canonical S)\n",
            length(cases), n_archive, length(cases) - n_archive))

# ---- cohort shapes ------------------------------------------------------
prefix_widths <- c(3L, 5L, 7L, 10L, 16L, 24L, 32L)
cohort_shapes <- function(all_targets, pair, seed) {
  base <- c(pair, setdiff(all_targets, pair))      # pair first, then the rest
  shapes <- list(pair = pair, pair_repeat = pair, pair_swapped = rev(pair))
  for (k in prefix_widths) if (k <= length(base)) shapes[[sprintf("prefix_%02d", k)]] <- base[seq_len(k)]
  if (length(base) >= 8L) shapes[["window_5_at_4"]] <- base[4:8]
  if (length(base) >= 13L) shapes[["window_5_at_9"]] <- base[9:13]
  if (length(base) >= 6L) shapes[["prefix_3_shifted"]] <- base[c(3L, 1L, 2L)]
  shapes[["all_natural"]] <- all_targets
  shapes[["all_reversed"]] <- rev(all_targets)
  set.seed(seed)
  shapes[["all_permuted"]] <- sample(all_targets)
  shapes
}

column_digests <- function(z, quantity) {
  m <- z[[quantity]]
  vapply(seq_len(ncol(m)), function(j) sha(bits(m[, j])), "")
}
quantities <- c("optimizer_coefficients", "optimizer_fitted", "optimizer_residuals",
                "fixed_coefficients", "fixed_fitted", "fixed_residuals",
                "selected_log_sp", "fixed_log_sp")

Sys.setenv(FASTKPC_TEST_MULTI_PENALTY_EXACT_ONLY = "1")
load_fastkpc_cuda_native()
before <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
call_count <- 0L
summary_rows <- list()
for (ci in seq_along(cases)) {
  case <- cases[[ci]]
  path <- file.path(out, "cases", sprintf("%03d.rds", ci))
  if (file.exists(path)) { summary_rows[[ci]] <- readRDS(path)$summary; next }
  data <- datasets[[case$data]]
  S <- case$S
  setup <- full_cuda_ci_native_setup_native(data[, S, drop = FALSE])
  p <- fastkpc_full_cuda_phase6_prepare(make_setup(setup))
  geometry_sha <- sha(p)
  if (!is.null(case$archived)) stopifnot(identical(geometry_sha, case$archived$geometry_sha256))
  all_targets <- setdiff(seq_len(ncol(data)), S)
  shapes <- cohort_shapes(all_targets, case$pair, 20261021L + ci)
  values <- list()      # quantity -> target -> list(cohort -> column)
  rows <- list()
  for (shape_name in names(shapes)) {
    targets <- shapes[[shape_name]]
    Y <- unname(data[, targets, drop = FALSE])
    t0 <- proc.time()[["elapsed"]]
    z <- .Call("C_full_cuda_ci_multi_penalty_capacity_qualify",
      p$X, p$magic_qr_packed, as.double(p$magic_tau), p$magic_r,
      as.integer(p$magic_pivot), p$penalty_roots, p$penalty_matrices,
      as.integer(p$penalty_ranks), as.double(p$initial_log_sp), Y, NULL,
      PACKAGE = "fastkpc_cuda")
    elapsed <- proc.time()[["elapsed"]] - t0
    call_count <- call_count + 1L
    stopifnot(z$target_count == length(targets),
      all(z$optimizer_solver_info == 0L), all(z$fixed_solver_info == 0L),
      all(z$fixed_decomposition_route == 2L), isTRUE(z$fixed_sp_evaluation),
      z$diagnostics$cuda_exact_primary_target_count == length(targets),
      all(unlist(z$diagnostics[c("cpu_objective_count", "cpu_optimizer_count",
        "cpu_multi_penalty_solve_count", "fallback_count")]) == 0L))
    for (quantity in quantities) {
      digests <- column_digests(z, quantity)
      for (j in seq_along(targets)) {
        tg <- as.character(targets[j])
        values[[quantity]][[tg]][[shape_name]] <- z[[quantity]][, j]
        rows[[length(rows) + 1L]] <- data.frame(case = ci, cohort = shape_name,
          width = length(targets), position = j, target = targets[j],
          quantity = quantity, sha256 = digests[j], stringsAsFactors = FALSE)
      }
    }
    rows[[length(rows) + 1L]] <- NULL
    attr(rows, "elapsed") <- c(attr(rows, "elapsed"), setNames(elapsed, shape_name))
  }
  table <- do.call(rbind, rows)
  # per (quantity, target): distinct digests across cohorts and max abs diff vs the pair cohort
  per_target <- do.call(rbind, lapply(quantities, function(quantity) {
    do.call(rbind, lapply(names(values[[quantity]]), function(tg) {
      cols <- values[[quantity]][[tg]]
      ref_name <- if ("pair" %in% names(cols)) "pair" else names(cols)[1L]
      ref <- cols[[ref_name]]
      digs <- vapply(cols, function(v) sha(bits(v)), "")
      diffs <- vapply(cols, function(v) max(abs(v - ref)), 0)
      data.frame(case = ci, quantity = quantity, target = as.integer(tg),
        cohorts = length(cols), distinct = length(unique(digs)),
        reference = ref_name, max_abs_diff = max(diffs),
        mismatching_cohorts = paste(names(cols)[digs != digs[[ref_name]]], collapse = ","),
        stringsAsFactors = FALSE)
    }))
  }))
  # anchors: archived production-dispatcher outputs for the same pair cohort
  anchor <- NULL
  if (!is.null(case$archived)) {
    a <- case$archived$result
    anchor <- data.frame(quantity = c("optimizer_coefficients", "fixed_residuals", "fixed_coefficients", "selected_log_sp"),
      identical = c(identical(bits(values$optimizer_coefficients[[as.character(case$pair[1])]]$pair), bits(a$optimizer_coefficients[, 1L])) &&
                    identical(bits(values$optimizer_coefficients[[as.character(case$pair[2])]]$pair), bits(a$optimizer_coefficients[, 2L])),
                  identical(bits(values$fixed_residuals[[as.character(case$pair[1])]]$pair), bits(a$fixed_residuals[, 1L])) &&
                    identical(bits(values$fixed_residuals[[as.character(case$pair[2])]]$pair), bits(a$fixed_residuals[, 2L])),
                  identical(bits(values$fixed_coefficients[[as.character(case$pair[1])]]$pair), bits(a$fixed_coefficients[, 1L])) &&
                    identical(bits(values$fixed_coefficients[[as.character(case$pair[2])]]$pair), bits(a$fixed_coefficients[, 2L])),
                  identical(bits(values$selected_log_sp[[as.character(case$pair[1])]]$pair), bits(a$selected_log_sp[, 1L])) &&
                    identical(bits(values$selected_log_sp[[as.character(case$pair[2])]]$pair), bits(a$selected_log_sp[, 2L]))),
      stringsAsFactors = FALSE)
  }
  # optimizer-route residual vs exact-fixed residual for the same cohort/column
  route_match <- table[table$quantity == "optimizer_residuals", ]
  fixed_tab <- table[table$quantity == "fixed_residuals", ]
  key <- function(t) paste(t$cohort, t$target)
  route_identical <- sum(route_match$sha256 == fixed_tab$sha256[match(key(route_match), key(fixed_tab))])
  summary <- data.frame(case = ci, source = case$source, data = case$data, level = case$level,
    S_key = paste(S, collapse = "|"), q = ncol(p$X), targets = length(all_targets),
    cohorts = length(shapes), calls = length(shapes),
    max_width = max(vapply(shapes, length, 1L)),
    coefficient_targets_with_multiple_digests = sum(per_target$distinct[per_target$quantity == "optimizer_coefficients"] > 1L),
    fitted_targets_with_multiple_digests = sum(per_target$distinct[per_target$quantity == "optimizer_fitted"] > 1L),
    residual_targets_with_multiple_digests = sum(per_target$distinct[per_target$quantity == "optimizer_residuals"] > 1L),
    fixed_residual_targets_with_multiple_digests = sum(per_target$distinct[per_target$quantity == "fixed_residuals"] > 1L),
    log_sp_targets_with_multiple_digests = sum(per_target$distinct[per_target$quantity == "selected_log_sp"] > 1L),
    max_fitted_abs_diff = max(per_target$max_abs_diff[per_target$quantity == "optimizer_fitted"]),
    max_residual_abs_diff = max(per_target$max_abs_diff[per_target$quantity == "optimizer_residuals"]),
    max_fixed_residual_abs_diff = max(per_target$max_abs_diff[per_target$quantity == "fixed_residuals"]),
    optimizer_vs_fixed_residual_identical_columns = route_identical,
    optimizer_vs_fixed_residual_columns = nrow(route_match),
    anchor_all_identical = if (is.null(anchor)) NA else all(anchor$identical),
    geometry_sha256 = geometry_sha, stringsAsFactors = FALSE)
  saveRDS(list(case = case[c("source", "data", "level", "S", "pair")], geometry_sha256 = geometry_sha,
    shapes = shapes, digests = table, per_target = per_target, anchor = anchor,
    elapsed = attr(rows, "elapsed"), summary = summary, native_sha256 = native_sha,
    pair_shadow = list(optimizer_coefficients = values$optimizer_coefficients, fixed_residuals = values$fixed_residuals)[0L]),
    path, compress = FALSE)
  summary_rows[[ci]] <- summary
  cat(sprintf("case %02d level %d q %d targets %d: coef multi %d fitted multi %d resid multi %d fixed multi %d | max fitted diff %.3g resid %.3g | opt==fixed %d/%d | anchor %s\n",
    ci, case$level, ncol(p$X), length(all_targets),
    summary$coefficient_targets_with_multiple_digests, summary$fitted_targets_with_multiple_digests,
    summary$residual_targets_with_multiple_digests, summary$fixed_residual_targets_with_multiple_digests,
    summary$max_fitted_abs_diff, summary$max_residual_abs_diff,
    route_identical, nrow(route_match), as.character(summary$anchor_all_identical)))
  flush.console()
}
after <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
fields <- c("live_device_allocations", "live_device_bytes", "live_streams", "live_events")
stopifnot(identical(before[fields], after[fields]), identical(file_sha(native), native_sha))
summary <- do.call(rbind, summary_rows)
saveRDS(list(schema = "fastkpc-residual-cohort-identity-v1", native_sha256 = native_sha,
  canonical_receipt_sha256 = canon_sha, data_sha256 = file_sha(data_path),
  environment = Sys.getenv(c("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_MAX_CONNECTIONS",
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "FASTKPC_TEST_MULTI_PENALTY_EXACT_ONLY")),
  resources_before = before, resources_after = after, summary = summary),
  file.path(out, "summary.rds"), compress = FALSE)
write.csv(summary, file.path(out, "summary.csv"), row.names = FALSE)
cat("done; cases", nrow(summary), "\n")
