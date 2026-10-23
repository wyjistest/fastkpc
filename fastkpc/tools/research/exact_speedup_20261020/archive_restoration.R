# Supplement the immutable campaign archive after restoring/rebuilding runtime.
# Rscript .../archive_restoration.R INPUT_DIRECTORY CAMPAIGN_ARCHIVE_DIRECTORY
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L, dir.exists(args[[1L]]), dir.exists(args[[2L]]),
          requireNamespace("digest", quietly = TRUE), requireNamespace("jsonlite", quietly = TRUE))
sha <- function(path) unname(digest::digest(file = path, algo = "sha256", serialize = FALSE))
original_manifest <- file.path(args[[2L]], "manifest.json")
stopifnot(file.exists(original_manifest))
destination <- file.path(args[[2L]], "restoration")
stopifnot(!dir.exists(destination))
runtime_source_clean <- identical(system2("git", c(
  "diff", "--quiet", "HEAD", "--", "fastkpc/src", "fastkpc/R"
)), 0L)
stopifnot(runtime_source_clean)
native <- "fastkpc/build/fastkpc_cuda.so"
native_sha <- sha(native)
receipt_path <- file.path(args[[1L]], "subset-restored-queues32.rds")
receipt <- readRDS(receipt_path)
stopifnot(identical(receipt$workload, "subset22"), all(unlist(receipt$gates)),
          identical(receipt$native_sha256, native_sha))
memcheck <- readLines(file.path(args[[1L]], "structured-memcheck.log"), warn = FALSE)
stopifnot(any(grepl("ERROR SUMMARY: 0 errors", memcheck, fixed = TRUE)),
          any(grepl("LEAK SUMMARY: 0 bytes leaked in 0 allocations", memcheck, fixed = TRUE)))
dir.create(destination)
saveRDS(receipt, file.path(destination, basename(receipt_path)), compress = "gzip")
evidence <- list.files(args[[1L]], full.names = TRUE)
evidence <- evidence[grepl(
  "^(restored-.*[.](json|log|txt|nsys-rep)|subset-restored-queues32[.]log|structured-memcheck[.](json|log)|canonical-runner-frozen[.]R)$",
  basename(evidence)
)]
stopifnot(all(file.copy(evidence, file.path(destination, basename(evidence)))))
tool_paths <- sort(c(
  list.files("fastkpc/tools/research/exact_speedup_20261020", recursive = TRUE, full.names = TRUE),
  "fastkpc/tools/with_phase10_exact_development_env.sh",
  "fastkpc/tools/phase10_cuda_full_eig_lane_benchmark.cu"
))
tool_paths <- tool_paths[!file.info(tool_paths)$isdir]
tool_paths <- tool_paths[!grepl("(__pycache__|[.](pyc|o|so)$)", tool_paths)]
objects <- paste0("fastkpc/build/mgcv_multi_penalty_gcv",
                  c("", "_small64", "_ext192", "_ext384", "_ext559"), ".o")
stopifnot(all(file.exists(objects)))
copied <- list.files(destination, full.names = TRUE)
manifest <- list(
  schema = "phase10-exact-speedup-restoration-v1",
  generated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  source_head = system2("git", c("rev-parse", "HEAD"), stdout = TRUE),
  original_campaign_manifest_sha256 = sha(original_manifest),
  runtime_sources_match_head = runtime_source_clean,
  restored_native_sha256 = native_sha,
  matches_original_accepted_native = identical(native_sha,
    sha(file.path(args[[1L]], "baseline-fe8286f7.so"))),
  restored_optimizer_object_sha256 = as.list(setNames(vapply(objects, sha, character(1L)), objects)),
  final_tool_source_sha256 = as.list(setNames(vapply(tool_paths, sha, character(1L)), tool_paths)),
  subset_elapsed_sec = receipt$elapsed_sec,
  subset_gates = receipt$gates,
  subset_runtime_environment = receipt$runtime_environment,
  structured_memcheck_zero_errors_and_leaks = TRUE,
  full_canonical_on_restored_native_run_this_campaign = FALSE,
  evidence = lapply(copied, function(path) list(
    file = basename(path), sha256 = sha(path), bytes = file.info(path)$size
  )),
  evidence_class = "development-restoration-and-subset-parity-not-promotion",
  phase10 = "ACTIVE", candidate_frozen = FALSE, holdout_opened = FALSE, recommended = FALSE
)
jsonlite::write_json(manifest, file.path(destination, "manifest.json"),
                     auto_unbox = TRUE, pretty = TRUE, digits = NA, null = "null")
cat("Restoration manifest SHA-256:", sha(file.path(destination, "manifest.json")), "\n")
