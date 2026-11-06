args <- commandArgs(TRUE)
stopifnot(length(args) == 2L)
parent <- readRDS(file.path(args[[1L]], "receipts.rds"))
candidate <- readRDS(file.path(args[[2L]], "receipts.rds"))
stopifnot(length(parent) == 3L)
for (name in names(parent)) {
  a <- parent[[name]]; b <- candidate[[name]]
  stopifnot(!is.null(b), identical(a$candidate$tasks, b$candidate$tasks),
    identical(a$candidate$pMax, b$candidate$pMax),
    identical(a$candidate$adjacency, b$candidate$adjacency),
    identical(a$candidate$sepsets, b$candidate$sepsets),
    identical(a$candidate$n.edgetests, b$candidate$n.edgetests),
    identical(a$seed_after, b$seed_after))
}
cat("PASS three-method parent/candidate k35 bitwise task/pMax/graph/RNG regression\n")
