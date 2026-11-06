source("fastkpc/tools/research/numcol_20261106/common.R")
args <- commandArgs(TRUE)
stopifnot(length(args) == 3L, !file.exists(args[[3L]]))
native <- args[[1L]]; parent_path <- args[[2L]]; out <- args[[3L]]
dir.create(out)
.fastkpc_cuda_so <- local({ path <- native; function() path })
stopifnot(fastkpc_cuda_available())
rank_mode("off")
Sys.setenv(FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = "full")
old <- new.env(parent = globalenv()); old$eigs <- RSpectra::eigs
sys.source(parent_path, envir = old)
current <- fastkpc_legacy_env()
set.seed(607L)
n <- 140L; x <- rnorm(n)
data <- cbind(x = x, y = x + .05 * rnorm(n), z = x^2 + .2 * rnorm(n), w = rnorm(n))
same <- list(); scalar <- list()
for (j in 2:4) {
  for (k in c(2L, 7L, 35L, 50L, 64L, 65L, 139L)) {
    set.seed(707L); a <- suppressWarnings(old$dcov.gamma(data[, 1L], data[, j], numCol = k))
    set.seed(707L); b <- suppressWarnings(current$dcov.gamma(data[, 1L], data[, j], numCol = k))
    stopifnot(identical(a, b))
    same[[length(same) + 1L]] <- list(pair = j, k = k, bitwise = TRUE)
  }
  err <- tryCatch(old$dcov.gamma(data[, 1L], data[, j], numCol = 1L), error = identity)
  stopifnot(inherits(err, "error"), grepl("non-conformable", conditionMessage(err)))
  a <- current$dcov.gamma(data[, 1L], data[, j], numCol = 1L)
  b <- .Call("C_fastkpc_cuda_legacy_dcov_gamma_cpp_oracle",
    as.double(data[, 1L]), as.double(data[, j]), 1L, 1)
  scalar[[length(scalar) + 1L]] <- list(p_value = numeric_check(a$p.value, b$p.value),
    estimates = numeric_check(as.numeric(a$estimates), as.numeric(b$estimates)),
    statistic = numeric_check(as.numeric(a$statistic), b$statistic))
  stopifnot(all(vapply(tail(scalar, 1L)[[1L]], function(x) x$pass, logical(1L))))
}
for (e in parse("fastkpc/tools/research/real_data_parity_20261102/run.R"))
  if (is.call(e) && identical(e[[1L]], as.name("<-")) && is.symbol(e[[2L]]) &&
      as.character(e[[2L]]) %in% c("key", "cpu_schedule_reference")) eval(e)
suff_stat <- function(data) list(data = data, ic.method = "dcc.gamma", index = 1, numCol = 1L)
cases <- list(empty_after_level0 = data, empty_after_level1 = rank_data("dense140"))
reports <- list(); receipts <- list()
for (name in names(cases)) {
  data <- cases[[name]]
  calls <- list()
  traced <- function(x, y, S, suffStat) {
    value <- strict_p(current$kernelCItest(x, y, S, suffStat))
    calls[[length(calls) + 1L]] <<- data.frame(level = length(S), x = x, y = y,
      S_key = paste(sort(as.integer(S)), collapse = "|"), p_value = value)
    value
  }
  set.seed(707L)
  cpu <- pcalg::skeleton(suffStat = suff_stat(data), indepTest = traced, alpha = .1,
    labels = colnames(data), m.max = Inf, method = "stable")
  cpu_trace <- do.call(rbind, calls)
  replay <- cpu_schedule_reference(data, cpu_trace)$result
  dimnames(replay$adjacency) <- dimnames(replay$pMax) <- list(colnames(data), colnames(data))
  stopifnot(identical(unname(replay$adjacency != 0), unname(methods::as(cpu@graph, "matrix") != 0)),
    numeric_check(unname(replay$pMax), unname(cpu@pMax))$pass,
    identical(normalize_sepsets(replay$sepsets), normalize_sepsets(cpu@sepset)),
    identical(as.integer(replay$n.edgetests), as.integer(cpu@n.edgetests)))
  invisible(full_cuda_ci_one_call_cache_control_native("reset"))
  set.seed(707L)
  candidate <- fastkpc_compatible_cuda_wanpdag(data, options = list(numCol = 1L, trace_level = "logical"))
  rng <- .Random.seed
  oracle <- fastkpc_full_cuda_phase10_development_oracle(replay)
  comparison <- fastkpc_full_cuda_compare_candidate_skeleton(oracle, candidate$skeleton)
  trace <- fastkpc_full_cuda_normalize_logical_trace(candidate$skeleton)
  ci <- numeric_check(trace$p_value, oracle$logical_trace$p_value)
  pmax <- numeric_check(candidate$skeleton$pMax, replay$pMax)
  set.seed(707L)
  orientation <- orientation_reference(cpu, data)
  s <- candidate$skeleton$summary
  gates <- c(list(skeleton = isTRUE(comparison$summary$pass), ci = ci$pass, pmax = pmax$pass,
    rank = s$numCol == 1L,
    counts_exact = identical(as.integer(candidate$skeleton$n.edgetests), as.integer(cpu@n.edgetests)),
    expected_stop = max(trace$level) == if (name == "empty_after_level0") 0L else 1L,
    empty_graph = !any(candidate$skeleton$adjacency),
    authority = all(vapply(rank_authority, function(key) !is.null(s[[key]]) && s[[key]] == 0, logical(1L))),
    rng = identical(rng, orientation$rng_end), resources = !is.null(rank_resources())),
    compare_orientation(candidate$orientation, orientation))
  receipts[[name]] <- list(candidate = candidate, cpu = cpu, replay = replay, orientation = orientation)
  reports[[name]] <- list(gates = gates, comparison = comparison$summary, ci = ci, pmax = pmax,
    ci_count = nrow(trace), orientation_ci_count = nrow(orientation$trace),
    n_edgetests = candidate$skeleton$n.edgetests)
  print(gates)
}
saveRDS(receipts, file.path(out, "receipt.rds"), compress = FALSE)
pass <- all(vapply(reports, function(r) all(vapply(r$gates, isTRUE, logical(1L))), logical(1L)))
jsonlite::write_json(list(status = if (pass) "PASS" else "FAIL",
  cases = reports, preserved_cases = same, scalar_cases = scalar,
  parent_source_sha256 = digest::digest(file = parent_path, algo = "sha256", serialize = FALSE),
  current_source_sha256 = digest::digest(file = "kpcalg/R/dcovgamma.R", algo = "sha256", serialize = FALSE),
  native_sha256 = digest::digest(file = native, algo = "sha256", serialize = FALSE)),
  file.path(out, "report.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
stopifnot(pass)
cat("PASS rank-one R boundary and complete public GPU WAN-PDAG\n")
