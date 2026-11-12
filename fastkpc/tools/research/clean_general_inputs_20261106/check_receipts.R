# Read saved results only; no native loading or numerical recomputation.
args <- commandArgs(TRUE)
stopifnot(length(args) == 3L, !file.exists(args[[3L]]))
reference_root <- args[[1L]]; qualification_root <- args[[2L]]
summary <- jsonlite::read_json(file.path(qualification_root, "summary.json"), simplifyVector = FALSE)
stopifnot(isTRUE(summary$pass), requireNamespace("pcalg", quietly = TRUE))
key <- function(tasks) paste(tasks$level, tasks$x, tasks$y, tasks$S_key, sep = "\037")
numeric_error <- function(a, b) {
  stopifnot(identical(dim(a), dim(b)), length(a) == length(b))
  a <- as.double(a); b <- as.double(b)
  stopifnot(identical(is.na(a), is.na(b)), identical(is.infinite(a), is.infinite(b)),
    identical(a[is.infinite(a)], b[is.infinite(b)]))
  keep <- is.finite(a) & is.finite(b)
  error <- if (any(keep)) max(abs(a[keep] - b[keep])) else 0
  stopifnot(error <= 1e-10)
  error
}
rows <- list()
for (case in names(summary$reports)) {
  original <- readRDS(file.path(reference_root, case, "receipt.rds"))
  replay <- readRDS(file.path(qualification_root, paste0(case, "-replay"), "receipt.rds"))
  values <- replay$all_cpu_values
  modes <- list(); saved <- list()
  for (mode in c("off", "on")) {
    receipt <- readRDS(file.path(qualification_root, paste0(case, "-", mode), "receipt.rds"))
    result <- receipt$candidate$skeleton; s <- result$summary
    positions <- match(key(result$tasks), key(values))
    stopifnot(!anyNA(positions), identical(receipt$input_sha256, original$input_sha256),
      s$fixed_sp_root_cache_identity_rejection_count == 0,
      s$exact_primary_fixed_reuse_fallback_target_count == 0,
      s$exact_primary_fixed_reuse_live_device_bytes_after == 0,
      s$exact_primary_fixed_reuse_allocation_count == s$exact_primary_fixed_reuse_free_count)
    pools <- s[paste0("l", 3:6, "_optimizer_slots")]
    for (pool in pools) {
      stopifnot(pool$created == pool$destroyed, pool$created <= 32)
      if (pool$created > 0) stopifnot(pool$minimum_free_bytes >= 2 * 1024^3)
    }
    modes[[mode]] <- list(
      ci_count = nrow(result$tasks),
      ci_error = numeric_error(result$tasks$p_used, values$p_value[positions]),
      pmax_error = numeric_error(result$pMax, original$skeleton@pMax),
      component_produced = s$level_component_produced,
      structured_enabled = s$structured_component_prefill_enabled,
      root_identity_rejections = s$fixed_sp_root_cache_identity_rejection_count,
      arena_allocations = s$exact_primary_fixed_reuse_allocation_count,
      arena_frees = s$exact_primary_fixed_reuse_free_count,
      pools = pools)
    saved[[mode]] <- result
  }
  fields <- c("level", "x", "y", "S_key", "native_edge_deleted", "native_edge_ignored")
  stopifnot(identical(saved$off$tasks[fields], saved$on$tasks[fields]))
  rows[[case]] <- list(modes = modes,
    off_on_ci_error = numeric_error(saved$off$tasks$p_used, saved$on$tasks$p_used),
    off_on_pmax_error = numeric_error(saved$off$pMax, saved$on$pMax),
    off_on_ci_bits_equal = identical(saved$off$tasks$p_used, saved$on$tasks$p_used, num.eq = FALSE),
    off_on_pmax_bits_equal = identical(saved$off$pMax, saved$on$pMax, num.eq = FALSE))
}
jsonlite::write_json(list(pass = TRUE, cases = rows), args[[3L]], pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat("PASS independent receipt audit", length(rows), "inputs\n")
