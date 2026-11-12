# Public synthetic inputs only. No private datasets or previous artifacts.
args <- commandArgs(TRUE)
stopifnot(length(args) == 2L, !dir.exists(args[[1L]]))
out <- args[[1L]]; cases <- strsplit(args[[2L]], ",", fixed = TRUE)[[1L]]
stopifnot(length(cases) > 0L, !anyDuplicated(cases),
  all(cases %in% c("public500", "blocks256", "dense500", "dense1025")))
dir.create(out, recursive = TRUE)
source("fastkpc/R/full_cuda_ci_phase10_development_500x50.R")
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
manifest <- list()
for (case in cases) {
  if (case == "public500") {
    data <- fastkpc_full_cuda_phase10_development_500x50_data()
  } else {
    shape <- switch(case, dense500 = c(500L, 8L), blocks256 = c(256L, 12L), dense1025 = c(1025L, 4L))
    n <- shape[[1L]]; p <- shape[[2L]]
    set.seed(n * 100L + p)
    if (case == "blocks256") {
      data <- matrix(rnorm(n * p), n, p)
      for (start in seq.int(1L, p, by = 3L)) {
        if (start + 1L <= p) data[, start + 1L] <- .8 * data[, start] + .6 * data[, start + 1L]
        if (start + 2L <= p) data[, start + 2L] <- .6 * data[, start + 1L] + .8 * data[, start + 2L]
      }
    } else {
      z <- rnorm(n)
      data <- matrix(z, n, p) + .2 * matrix(rnorm(n * p), n, p)
    }
    colnames(data) <- paste0("clean_v", seq_len(p))
  }
  storage.mode(data) <- "double"
  stopifnot(all(is.finite(data)), !is.null(colnames(data)))
  path <- file.path(out, paste0(case, ".rds"))
  saveRDS(data, path, compress = FALSE)
  manifest[[case]] <- list(n = nrow(data), p = ncol(data), default_numCol = floor(nrow(data) / 10),
    path = paste0("inputs/", case, ".rds"),
    file_sha256 = digest::digest(file = path, algo = "sha256", serialize = FALSE),
    serialized_data_sha256 = digest::digest(data, algo = "sha256"),
    fp64_column_major_sha256 = digest::digest(writeBin(as.double(data), raw(), size = 8L,
      endian = "little"), algo = "sha256", serialize = FALSE))
}
jsonlite::write_json(manifest, file.path(out, "manifest.json"), pretty = TRUE,
  auto_unbox = TRUE, digits = NA)
cat("PASS public input generation", paste(cases, collapse = ","), "\n")
