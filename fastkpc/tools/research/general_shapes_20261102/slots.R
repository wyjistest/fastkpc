args <- commandArgs(TRUE)
stopifnot(length(args) == 3L)
source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
native <- normalizePath(args[[1L]])
.fastkpc_cuda_so <- local({ path <- native; function() path })
stopifnot(fastkpc_cuda_available())
dyn.load(normalizePath(args[[2L]]))
out <- args[[3L]]; stopifnot(!file.exists(out)); dir.create(out)
fields <- c("live_device_allocations", "live_device_bytes", "live_streams", "live_events")
reports <- list()
for (n in c(128L, 500L)) for (level in 3:6) {
  set.seed(1000L * n + level)
  inputs <- lapply(1:2, function(i) {
    setup <- full_cuda_ci_native_setup_native(matrix(rnorm(n * level), n, level))
    list(X = setup$X, ranks = setup$penalty_ranks,
      geometry = full_cuda_ci_native_geometry_prepare_native(setup$X,
        setup$penalty_blocks, setup$penalty_offsets, setup$penalty_ranks))
  })
  for (count in c(2L, 9L, 19L, 34L, 58L)) for (postback in c("0", "1")) {
    Sys.setenv(FASTKPC_DCC_GAMMA_VECTOR_POSTBACK = postback,
      FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H = "1")
    Y <- matrix(rnorm(n * count), n, count)
    before <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()[fields]
    result <- .Call("C_general_shape_slots", as.integer(level), inputs, Y)
    after <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()[fields]
    stopifnot(isTRUE(result$pass), identical(before, after), all(unlist(after) == 0))
    row <- c(list(n = n, level = level, targets = count, postback = postback), result)
    reports[[length(reports) + 1L]] <- row
    cat("PASS", n, level, count, postback, "\n"); flush.console()
  }
}
jsonlite::write_json(list(status = "PASS", native_sha256 = digest::digest(file = native,
  algo = "sha256", serialize = FALSE), cases = reports,
  targets_compared = sum(vapply(reports, `[[`, 0L, "targets_compared")),
  negative_cases = sum(vapply(reports, `[[`, 0L, "negative_cases"))),
  file.path(out, "report.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
