source("fastkpc/tools/research/numcol_20261106/common.R")
args <- commandArgs(TRUE)
stopifnot(length(args) == 3L)
native <- args[[1L]]; out <- args[[2L]]; which <- args[[3L]]
stopifnot(which %in% c("all", "35"), !file.exists(out))
dir.create(out)
.fastkpc_cuda_so <- local({ path <- native; function() path })
rank_mode("off")
stopifnot(fastkpc_cuda_available())
for (e in parse("fastkpc/tests/test_full_cuda_ci_strict_methods.R"))
  if (is.call(e) && identical(e[[1L]], as.name("<-")) && is.symbol(e[[2L]]) &&
      is.call(e[[3L]]) && identical(e[[3L]][[1L]], as.name("function"))) eval(e)
set.seed(9127L)
n <- 140L; z <- runif(n, -2, 2)
data <- cbind(x1 = sin(z) + rnorm(n, sd = .08), x2 = cos(z) + rnorm(n, sd = .08),
  x3 = z + rnorm(n, sd = .08), x4 = z^2 + rnorm(n, sd = .08), x5 = rnorm(n))
reports <- list(); receipts <- list()
for (method in c("dcc.perm", "hsic.gamma", "hsic.perm")) {
  grid <- data.frame(k = if (which == "35") 35L else c(1L, 7L, 35L, 50L, 64L, 65L, 102L, 139L), sig = 1)
  if (which == "all" && method != "dcc.perm")
    grid <- rbind(grid, data.frame(k = c(65L, 102L), sig = 1e-5))
  for (index in seq_len(nrow(grid))) {
    k <- grid$k[[index]]; sig <- grid$sig[[index]]
    cat("START", method, k, "sig", sig, "\n"); flush.console()
    actual_cpu_rank <- if (method == "dcc.perm") NA_integer_ else max(vapply(seq_len(ncol(data)),
      function(j) ncol(kernlab::inchol(as.matrix(data[, j]),
        kernel = kernlab::rbfdot(sigma = 1 / sig), maxiter = k)@.Data), integer(1L)))
    if (sig < 1) stopifnot(actual_cpu_rank > 64L)
    reference <- run_kpcalg_trace_oracle(data, method, .1, 2L, k, sig, 100L, 707L)
    reference_rng <- .Random.seed
    replay <- run_recorded_canonical_oracle(data, reference$tasks, .1, 2L)
    set.seed(707L)
    candidate <- run_cuda_candidate(data, method, .1, 2L, k, sig, 100L, 707L)
    rng <- .Random.seed
    comparison <- compare_method(candidate, replay, reference, method, 2L, "numCol")
    process <- compare_kpcalg_process(candidate$tasks, reference$tasks,
      if (method == "hsic.gamma") 1e-10 else 0)
    pmax <- numeric_check(unname(candidate$pMax), unname(reference$pMax))
    s <- candidate$summary
    gates <- list(numerical = isTRUE(process$p_values_within_tolerance),
      identities = isTRUE(process$same_task_keys), decisions = isTRUE(process$decision_trace_identical),
      pmax = pmax$pass, graph = comparison$canonical_SHD == 0 && comparison$legacy_SHD == 0,
      task_order = comparison$trace_identical, sepsets = comparison$sepsets_identical,
      counts = comparison$n_edgetests_identical,
      counts_exact_pcalg = identical(as.integer(candidate$n.edgetests), as.integer(reference$n.edgetests)),
      rank = s$numCol == k,
      rng = identical(rng, reference_rng), resources = !is.null(rank_resources()),
      authority = all(vapply(rank_authority, function(key) !is.null(s[[key]]) && s[[key]] == 0, logical(1L))))
    name <- paste(method, k, sig, sep = "-")
    receipts[[name]] <- list(candidate = candidate, reference = reference, replay = replay,
      seed_after = rng, reference_seed_after = reference_rng)
    reports[[name]] <- list(method = method, numCol = k, sig = sig, actual_cpu_rank = actual_cpu_rank, gates = gates,
      pass = all(vapply(gates, isTRUE, logical(1L))), ci_count = nrow(reference$tasks),
      ci_error = process$max_abs_p_diff, pmax = pmax)
    saveRDS(receipts, file.path(out, "receipts.rds"), compress = FALSE)
    jsonlite::write_json(reports, file.path(out, "cases.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
    print(gates); stopifnot(reports[[name]]$pass)
  }
}
jsonlite::write_json(list(status = "PASS", native_sha256 = digest::digest(file = native,
  algo = "sha256", serialize = FALSE), cases = reports), file.path(out, "report.json"),
  pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat("PASS", length(reports), "method/rank cases\n")
