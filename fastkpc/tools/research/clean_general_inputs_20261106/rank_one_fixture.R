# The upstream rank-one workaround exists only in this isolated test environment.
args <- commandArgs(TRUE)
stopifnot(length(args) == 2L, !file.exists(args[[2L]]))
patched_snapshot <- args[[1L]]; out <- args[[2L]]; dir.create(out)
source("fastkpc/tools/research/numcol_20261106/common.R")
sha <- function(path) digest::digest(file = path, algo = "sha256", serialize = FALSE)
original_sha <- sha("kpcalg/R/dcovgamma.R")
stopifnot(original_sha == "d0620dd9b06f006170b5e142b892c202619296797b2bf9ca842a7c2b5b3b715f",
  sha(patched_snapshot) == "05a30eb59a7404ab24e3137380862d2840b389fc3b17415b9c6045a1fa0e4a69",
  fastkpc_cuda_available())
original <- fastkpc_legacy_env()
fixture <- fastkpc_legacy_env()
sys.source(patched_snapshot, envir = fixture)
rank_mode("off")
Sys.setenv(FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = "full")
set.seed(607L); n <- 140L; x <- rnorm(n)
data <- cbind(x = x, y = x + .05 * rnorm(n), z = x^2 + .2 * rnorm(n), w = rnorm(n))
checks <- list()
for (j in 2:4) {
  error <- tryCatch(original$dcov.gamma(data[, 1], data[, j], numCol = 1L), error = identity)
  stopifnot(inherits(error, "error"), grepl("non-conformable", conditionMessage(error)))
  a <- fixture$dcov.gamma(data[, 1], data[, j], numCol = 1L)
  b <- .Call("C_fastkpc_cuda_legacy_dcov_gamma_cpp_oracle", as.double(data[, 1]), as.double(data[, j]), 1L, 1)
  checks[[length(checks)+1L]] <- list(original_error = conditionMessage(error),
    pvalue = numeric_check(a$p.value, b$p.value), statistic = numeric_check(as.numeric(a$statistic), b$statistic),
    moments = numeric_check(as.numeric(a$estimates), as.numeric(b$estimates)))
}
invisible(full_cuda_ci_one_call_cache_control_native("reset"))
set.seed(707L)
candidate <- fastkpc_compatible_cuda_skeleton(data, .1,
  options = list(route = "full_cuda", compatible_cuda_strict = TRUE, numCol = 1L, trace_level = "logical"))
tasks <- candidate$tasks; expected <- numeric(nrow(tasks))
for (j in seq_len(nrow(tasks))) {
  subset <- if (nzchar(tasks$S_key[[j]])) as.integer(strsplit(tasks$S_key[[j]], "|", fixed = TRUE)[[1]]) else integer()
  residual <- if (length(subset)) original$regrXonS(data[, c(tasks$x[[j]], tasks$y[[j]])], data[, subset]) else
    data[, c(tasks$x[[j]], tasks$y[[j]])]
  expected[[j]] <- .Call("C_fastkpc_cuda_legacy_dcov_gamma_cpp_oracle", as.double(residual[, 1]),
    as.double(residual[, 2]), 1L, 1)$p.value
}
ci <- numeric_check(tasks$p_used, expected)
gates <- list(original_file_unchanged = identical(original_sha, sha("kpcalg/R/dcovgamma.R")),
  isolated_patch_matches_full_eig = all(vapply(checks, function(x) x$pvalue$pass && x$statistic$pass && x$moments$pass, logical(1))),
  gpu_matches_independent_full_eig = ci$pass, resources = !is.null(rank_resources()),
  authority = all(vapply(rank_authority, function(field) !is.null(candidate$summary[[field]]) && candidate$summary[[field]] == 0, logical(1))))
saveRDS(list(candidate = candidate, cpu_p_values = expected, checks = checks), file.path(out, "receipt.rds"), compress = FALSE)
jsonlite::write_json(list(pass = all(vapply(gates, isTRUE, logical(1))), gates = gates, checks = checks,
  ci_count = nrow(tasks), ci = ci, native_sha256 = sha("fastkpc/build/fastkpc_cuda.so"), original_r_sha256 = original_sha,
  patched_snapshot_sha256 = sha(patched_snapshot), original_r_rank_one_orientation_qualified = FALSE,
  evidence_class = "separate fixture plus independent full eig; original rank-one R error remains documented"),
  file.path(out, "report.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
print(gates); stopifnot(all(vapply(gates, isTRUE, logical(1))))
cat("PASS isolated rank-one fixture; original reference retained\n")
