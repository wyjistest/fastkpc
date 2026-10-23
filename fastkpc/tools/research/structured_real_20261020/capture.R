# Offline capture through the accepted GPU capacity dispatcher. No CPU solver
# supplies residuals; kpcalg below is an explicitly separate diagnostic oracle.
source("fastkpc/tools/research/guarded_qr_20261020/common.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
args <- commandArgs(TRUE)
stopifnot(length(args) == 1L)
out <- args[[1L]]
dir.create(out, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out, "cohorts"), showWarnings = FALSE)
native <- "fastkpc/build/fastkpc_cuda.so"
data_path <- "fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"
task_path <- "/tmp/fastkpc-exact-speedup-20261020/subset-restored-queues32.rds"
receipt <- readRDS(task_path)
native_sha <- file_sha(native)
stopifnot(identical(native_sha, receipt$native_sha256))
data <- as.matrix(readRDS(data_path))[, seq_len(22L), drop = FALSE]
storage.mode(data) <- "double"
tasks <- receipt$result$tasks
tasks <- tasks[tasks$level == 3L, ]
# One pair per S guarantees 640 distinct (target,S) identities. Include
# near-alpha development witnesses, then spread across the remaining S keys.
tasks <- tasks[order(abs(tasks$p_used - 0.1), tasks$canonical_test_order_id), ]
tasks <- tasks[!duplicated(tasks$S_key), ]
near <- head(tasks, 64L)
rest <- tasks[-seq_len(64L), ]
rest <- rest[order(rest$canonical_test_order_id), ]
chosen <- rbind(near, rest[round(seq(1, nrow(rest), length.out = 256L)), ])
stopifnot(nrow(chosen) == 320L, !anyDuplicated(chosen$S_key))
Sys.setenv(FASTKPC_TEST_MULTI_PENALTY_EXACT_ONLY = "1")
load_fastkpc_cuda_native()
before <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
details <- list()
for (i in seq_len(nrow(chosen))) {
  row <- chosen[i, ]
  path <- file.path(out, "cohorts", sprintf("%03d.rds", i))
  if (file.exists(path)) {
    d <- readRDS(path)
    stopifnot(identical(d$native_sha256, native_sha), identical(d$task, row))
  } else {
    S <- as.integer(strsplit(row$S_key, "|", fixed = TRUE)[[1L]])
    setup <- full_cuda_ci_native_setup_native(data[, S, drop = FALSE])
    p <- fastkpc_full_cuda_phase6_prepare(make_setup(setup))
    Y <- unname(data[, c(row$x, row$y), drop = FALSE])
    z <- .Call("C_full_cuda_ci_multi_penalty_capacity_qualify",
      p$X, p$magic_qr_packed, as.double(p$magic_tau), p$magic_r,
      as.integer(p$magic_pivot), p$penalty_roots, p$penalty_matrices,
      as.integer(p$penalty_ranks), as.double(p$initial_log_sp), Y, NULL,
      PACKAGE = "fastkpc_cuda")
    stopifnot(all(z$fixed_solver_info == 0L),
      all(z$fixed_decomposition_route == 2L),
      z$fixed_sp_evaluation, z$fixed_device_resident_before_release,
      z$fixed_residual_slot_released, z$optimizer_residual_slot_released,
      z$diagnostics$cuda_exact_primary_target_count == 2L,
      all(unlist(z$diagnostics[c("cpu_objective_count", "cpu_optimizer_count",
        "cpu_multi_penalty_solve_count", "fallback_count")]) == 0L))
    set.seed(20261020L + i)
    authority <- kpcalg::dcov.gamma(z$fixed_residuals[, 1L],
      z$fixed_residuals[, 2L], index = 1, numCol = 35L)
    stopifnot(is.finite(authority$p.value),
      abs(authority$p.value - row$p_used) <= 1e-10,
      (authority$p.value >= 0.1) == (row$p_used >= 0.1))
    d <- list(task = row, native_sha256 = native_sha,
      geometry_sha256 = sha(p), result = z, authority = authority)
    saveRDS(d, path, compress = FALSE)
  }
  details[[i]] <- d
  if (i %% 10L == 0L || i == 1L) {
    cat(sprintf("captured %d/320 pairs; task=%d p=%.9g\n",
      i, row$canonical_test_order_id, d$authority$p.value))
    flush.console()
  }
}
after <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
fields <- c("live_device_allocations", "live_device_bytes", "live_streams", "live_events")
stopifnot(identical(before[fields], after[fields]), identical(file_sha(native), native_sha))
residuals <- do.call(cbind, lapply(details, function(d) d$result$fixed_residuals))
stopifnot(identical(dim(residuals), c(351L, 640L)), all(is.finite(residuals)))
binary <- file.path(out, "residuals.f64")
con <- file(binary, "wb")
writeBin(as.double(residuals), con, size = 8L, endian = "little")
close(con)
pairs <- do.call(rbind, lapply(seq_along(details), function(i) {
  d <- details[[i]]
  data.frame(pair = i, component_x = 2L*i-2L, component_y = 2L*i-1L,
    task = d$task$canonical_test_order_id, S_key = d$task$S_key,
    x = d$task$x, y = d$task$y, original_gpu_p = d$task$p_used,
    authority_p = d$authority$p.value,
    statistic = unname(d$authority$estimates[1L]),
    mean = unname(d$authority$estimates[2L]),
    variance = unname(d$authority$estimates[3L]))
}))
write.csv(pairs, file.path(out, "pairs.csv"), row.names = FALSE)
meta <- list(schema = "structured-real-residual-capture-v1",
  producer = system2("git", c("rev-parse", "HEAD"), stdout = TRUE),
  native_sha256 = native_sha, n = 351L, k = 35L, components = 640L,
  pairs = 320L, dataset_columns = 22L, conditioning_level = 3L,
  selection = "64 nearest-alpha unique-S pairs, 256 evenly spaced remaining unique-S pairs",
  capture = "GPU exact-primary optimizer plus GPU exact-fixed evaluator, independent two-target replay",
  authority = "kpcalg diagnostic oracle on the identical captured residuals; not a CPU fallback",
  original_trace_max_p_error = max(abs(pairs$authority_p - pairs$original_gpu_p)),
  resource_counters_closed = TRUE, resources_before = before, resources_after = after,
  source_files = setNames(lapply(c(data_path, task_path, native,
    "fastkpc/tools/research/structured_real_20261020/capture.R",
    "fastkpc/tools/research/guarded_qr_20261020/common.R"), file_sha),
    c(data_path, task_path, native,
      "fastkpc/tools/research/structured_real_20261020/capture.R",
      "fastkpc/tools/research/guarded_qr_20261020/common.R")),
  residual_sha256 = file_sha(binary))
jsonlite::write_json(meta, file.path(out, "capture.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
cat("PASS capture; max p error against original GPU task", meta$original_trace_max_p_error, "\n")
