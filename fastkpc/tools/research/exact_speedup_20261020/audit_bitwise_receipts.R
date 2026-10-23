# Explicit IEEE representation checks, including signed zero and NaN payloads.
# Rscript .../audit_bitwise_receipts.R INPUT_DIRECTORY NEW_JSON
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L, dir.exists(args[[1L]]), !file.exists(args[[2L]]))
pairs <- list(
  c("canonical-default.rds", "canonical-queues32.rds"),
  c("canonical-queues32.rds", "canonical-pinned-queues32.rds"),
  c("subset-pageable-connections32.rds", "subset-restored-queues32.rds")
)
sha <- function(path) digest::digest(file = path, algo = "sha256", serialize = FALSE)
checks <- lapply(pairs, function(pair) {
  paths <- file.path(args[[1L]], pair)
  a <- readRDS(paths[[1L]])
  b <- readRDS(paths[[2L]])
  stopifnot(identical(a$workload, b$workload))
  gates <- list(
    p = identical(a$result$tasks$p_used, b$result$tasks$p_used,
                  num.eq = FALSE, single.NA = FALSE),
    pMax = identical(a$result$pMax, b$result$pMax, num.eq = FALSE, single.NA = FALSE)
  )
  stopifnot(all(unlist(gates)))
  list(reference = pair[[1L]], candidate = pair[[2L]],
       reference_sha256 = sha(paths[[1L]]), candidate_sha256 = sha(paths[[2L]]),
       count = nrow(b$result$tasks), gates = gates)
})
jsonlite::write_json(list(schema = "explicit-ieee-bitwise-receipt-audit-v1", checks = checks),
                     args[[2L]], auto_unbox = TRUE, pretty = TRUE)
cat("Explicit IEEE bitwise audit PASS:", length(checks), "receipt pairs\n")
