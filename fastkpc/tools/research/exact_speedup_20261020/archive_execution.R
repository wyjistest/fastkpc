# Run only after all timed jobs finish. Compression/copying is outside timing.
# Rscript .../archive_execution.R INPUT_DIRECTORY NEW_OUTPUT_DIRECTORY
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L, dir.exists(args[[1L]]), !dir.exists(args[[2L]]),
          requireNamespace("digest", quietly = TRUE), requireNamespace("jsonlite", quietly = TRUE))
dir.create(args[[2L]], recursive = TRUE)
sha <- function(path) unname(digest::digest(file = path, algo = "sha256", serialize = FALSE))
files <- list.files(args[[1L]], pattern = "^(canonical|subset)-.*[.]rds$", full.names = TRUE)
stopifnot(length(files) >= 2L)
records <- lapply(files, function(path) {
  x <- readRDS(path)
  stopifnot(identical(x$schema, "exact-speedup-pinned-optimizer-ab-v1"))
  destination <- file.path(args[[2L]], basename(path))
  saveRDS(x, destination, compress = "gzip")
  selected <- c("cuda_optimizer_host_ms", "cuda_multi_penalty_optimizer_host_ms",
                "cuda_dcov_host_ms", "cuda_dcov_full_eig_ms", "cuda_residual_solve_host_ms",
                "native_setup_ms", "physical_tests_evaluated", "logical_tests_evaluated")
  selected <- selected[selected %in% names(x$result$summary)]
  list(file = basename(destination), sha256 = sha(destination),
       original_uncompressed_sha256 = sha(path), workload = x$workload, pinned = x$pinned,
       native_sha256 = x$native_sha256, elapsed_sec = x$elapsed_sec,
       runtime_environment = x$runtime_environment, gates = x$gates,
       oracle_errors = x$oracle_errors, logical_ci_count = nrow(x$result$tasks),
       stages = x$result$summary[selected])
})
native <- "fastkpc/build/fastkpc_cuda.so"
native_sha <- sha(native)
stopifnot(any(vapply(records, function(x) identical(x$native_sha256, native_sha), logical(1L))))
stopifnot(file.copy(native, file.path(args[[2L]], paste0("native-", native_sha, ".so"))))
evidence <- list.files(args[[1L]], pattern = "[.](json|txt|log|png|nsys-rep|so)$", full.names = TRUE)
evidence <- evidence[!grepl("build-", basename(evidence))]
stopifnot(all(file.copy(evidence, file.path(args[[2L]], basename(evidence)))))
evidence_index <- lapply(file.path(args[[2L]], basename(evidence)), function(path) {
  list(file = basename(path), sha256 = sha(path), bytes = file.info(path)$size)
})
source_paths <- sort(c(list.files("fastkpc/src", recursive = TRUE, full.names = TRUE),
                       list.files("fastkpc/R", recursive = TRUE, full.names = TRUE),
                       list.files("fastkpc/tools/research/exact_speedup_20261020",
                                  recursive = TRUE, full.names = TRUE)))
source_paths <- source_paths[!file.info(source_paths)$isdir]
source_paths <- source_paths[!grepl("(__pycache__|[.](pyc|o|so)$)", source_paths)]
sources <- setNames(vapply(source_paths, sha, character(1L)), source_paths)
manifest <- list(
  schema = "phase10-exact-speedup-execution-v1",
  generated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  evidence_class = "source-closure-development-not-promotion",
  source_head = system2("git", c("rev-parse", "HEAD"), stdout = TRUE),
  source_closure_sha256 = digest::digest(sources, algo = "sha256"),
  source_files = as.list(sources), native_sha256 = native_sha,
  canonical_input_sha256 = sha("fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"),
  receipts = records, evidence = evidence_index,
  phase10 = "ACTIVE", candidate_frozen = FALSE, holdout_opened = FALSE,
  recommended = FALSE
)
jsonlite::write_json(manifest, file.path(args[[2L]], "manifest.json"),
                     auto_unbox = TRUE, pretty = TRUE, digits = NA, null = "null")
cat("Archived receipts and native to", normalizePath(args[[2L]]), "\n")
cat("manifest_sha256", sha(file.path(args[[2L]], "manifest.json")), "\n")
