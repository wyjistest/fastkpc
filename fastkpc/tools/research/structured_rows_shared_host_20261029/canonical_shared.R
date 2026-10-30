# Keep all canonical arithmetic and gates; label the supplement and timestamp
# outside the existing compute timer. Evaluate the entire vector together.
stopifnot(Sys.getenv("FASTKPC_DIAG_SHARED_HOST_TIMING") == "1",
          Sys.getenv("FASTKPC_DIAG_SHARED_HOST_QUALIFICATION") == "0")
shared_parent_runner <- "fastkpc/tools/research/build_consistency_20261029/canonical.R"
shared_original <- parse(shared_parent_runner)
shared_expressions <- list()
shared_counts <- c(label=0L, start=0L, end=0L)
shared_label <- paste("shared-host structured row-only development timing; GPU1 exclusive,",
                      "GPU0 background allowed; not isolated acceptance or promotion")
for (shared_expr in shared_original) {
  shared_lhs <- if (is.call(shared_expr) &&
    identical(shared_expr[[1L]], as.name("<-")) && is.symbol(shared_expr[[2L]]))
      as.character(shared_expr[[2L]]) else ""
  if (shared_lhs == "evidence_class") {
    shared_expr <- substitute(evidence_class <- LABEL, list(LABEL=shared_label))
    shared_counts[["label"]] <- shared_counts[["label"]] + 1L
  }
  if (shared_lhs == "started") {
    shared_expressions <- c(shared_expressions,
      list(quote(shared_compute_start_epoch <- as.numeric(Sys.time()))))
    shared_counts[["start"]] <- shared_counts[["start"]] + 1L
  }
  shared_expressions <- c(shared_expressions, list(shared_expr))
  if (shared_lhs == "elapsed") {
    shared_expressions <- c(shared_expressions, list(quote({
      shared_compute_end_epoch <- as.numeric(Sys.time())
      jsonlite::write_json(list(start_epoch=shared_compute_start_epoch,
        end_epoch=shared_compute_end_epoch, elapsed_sec=elapsed,
        evidence_class=shared_label, timestamps_outside_timer=TRUE),
        file.path(out,"compute-window.json"), auto_unbox=TRUE, pretty=TRUE, digits=NA)
    })))
    shared_counts[["end"]] <- shared_counts[["end"]] + 1L
  }
}
stopifnot(all(shared_counts == 1L))
shared_out <- commandArgs(TRUE)[[1L]]
stopifnot(dir.exists(shared_out),
          !file.exists(file.path(shared_out,"runner-transformation.json")))
saveRDS(as.expression(shared_expressions), file.path(shared_out,"effective-runner.rds"),
        compress=FALSE)
jsonlite::write_json(list(parent_runner=shared_parent_runner,
  parent_sha256=digest::digest(file=shared_parent_runner,algo="sha256",serialize=FALSE),
  effective_sha256=digest::digest(file=file.path(shared_out,"effective-runner.rds"),
                                algo="sha256",serialize=FALSE),
  changes=as.list(shared_counts), arithmetic_and_gates_unchanged=TRUE),
  file.path(shared_out,"runner-transformation.json"), auto_unbox=TRUE, pretty=TRUE)
eval(as.expression(shared_expressions), envir=.GlobalEnv)
