# Materialize frozen inputs outside the clean producer checkout.
args <- commandArgs(TRUE)
stopifnot(length(args) == 2L, !dir.exists(args[[2L]]))
original_root <- args[[1L]]; out <- args[[2L]]
dir.create(out, recursive = TRUE)
source("fastkpc/R/full_cuda_ci_phase10_development_500x50.R")
paths <- c(
  real179 = "causalDiscovery/20221022181603/CD8.c11.Tex.PDCD1-causalDiscoveryInput.RData",
  real243 = "causalDiscovery/20220418113957/logged-liver-exhausted-allData-causalDiscoveryInput.RData",
  real300 = "causalDiscovery/20240315113222/CD8.c11.Tex.PDCD1-causalDiscoveryInput.RData")
reports <- list()
for (name in c(names(paths), "public500", "dense500", "blocks256", "dense1025")) {
  raw_path <- NULL
  if (name %in% names(paths)) {
    raw_path <- file.path(original_root, paths[[name]])
    payload <- new.env(parent = emptyenv()); loaded <- load(raw_path, envir = payload)
    stopifnot(length(loaded) == 1L)
    data <- as.matrix(payload[[loaded]])
  } else if (name == "public500") {
    data <- fastkpc_full_cuda_phase10_development_500x50_data()
  } else {
    shape <- switch(name, dense500 = c(500L, 8L), blocks256 = c(256L, 12L), dense1025 = c(1025L, 4L))
    n <- shape[[1L]]; p <- shape[[2L]]
    set.seed(n * 100L + p)
    if (name != "blocks256") {
      z <- rnorm(n)
      data <- matrix(z, n, p) + .2 * matrix(rnorm(n * p), n, p)
    } else {
      data <- matrix(rnorm(n * p), n, p)
      for (start in seq.int(1L, p, by = 3L)) {
        if (start + 1L <= p) data[, start + 1L] <- .8 * data[, start] + .6 * data[, start + 1L]
        if (start + 2L <= p) data[, start + 2L] <- .6 * data[, start + 1L] + .8 * data[, start + 2L]
      }
    }
    colnames(data) <- paste0("clean_v", seq_len(p))
  }
  storage.mode(data) <- "double"
  stopifnot(all(is.finite(data)), !is.null(colnames(data)))
  path <- file.path(out, paste0(name, ".rds"))
  saveRDS(data, path, compress = FALSE)
  reports[[name]] <- list(n = nrow(data), p = ncol(data), default_numCol = floor(nrow(data)/10),
    serialized_data_sha256 = digest::digest(data, algo = "sha256"),
    path = path, file_sha256 = digest::digest(file = path, algo = "sha256", serialize = FALSE),
    raw_path = raw_path, raw_file_sha256 = if (!is.null(raw_path))
      digest::digest(file = raw_path, algo = "sha256", serialize = FALSE) else NULL)
}
jsonlite::write_json(reports, file.path(out, "manifest.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat("PASS seven frozen full inputs\n")
