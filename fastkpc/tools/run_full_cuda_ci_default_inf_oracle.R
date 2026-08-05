source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/legacy_runner.R")

args <- commandArgs(trailingOnly = TRUE)
arg_or_default <- function(index, default) {
  if (length(args) >= index && nzchar(args[[index]])) args[[index]] else default
}

data_path <- arg_or_default(
  1L,
  file.path("fastkpc", "artifacts", "kpc_tprs_real_zhu",
            "cancer_RD-causalDiscoveryInput.rds")
)
source_oracle_dir <- arg_or_default(
  2L,
  file.path("fastkpc", "artifacts", "full_cuda_ci", "oracle_351x48_v1")
)
output_dir <- arg_or_default(
  3L,
  file.path("fastkpc", "artifacts", "full_cuda_ci",
            "oracle_351x48_default_inf_v2")
)

required <- c(
  data_path,
  file.path(source_oracle_dir, "manifest.json"),
  file.path(source_oracle_dir, "adjacency.rds"),
  file.path(source_oracle_dir, "sepsets.rds"),
  file.path(source_oracle_dir, "pmax.rds"),
  file.path(source_oracle_dir, "n_edgetests.csv"),
  file.path(source_oracle_dir, "logical_ci_trace.rds"),
  file.path(source_oracle_dir, "deletion_trace.csv")
)
missing <- required[!file.exists(required)]
if (length(missing) > 0L) {
  stop("default-Inf oracle input is missing: ",
       paste(missing, collapse = ", "), call. = FALSE)
}
if (!requireNamespace("digest", quietly = TRUE) ||
    !requireNamespace("jsonlite", quietly = TRUE)) {
  stop("digest and jsonlite are required", call. = FALSE)
}

data <- as.matrix(readRDS(data_path))
storage.mode(data) <- "double"
adjacency <- readRDS(file.path(source_oracle_dir, "adjacency.rds")) != 0
sepsets <- readRDS(file.path(source_oracle_dir, "sepsets.rds"))
pmax <- readRDS(file.path(source_oracle_dir, "pmax.rds"))
n_edgetests <- utils::read.csv(
  file.path(source_oracle_dir, "n_edgetests.csv"),
  stringsAsFactors = FALSE
)
logical_trace <- readRDS(
  file.path(source_oracle_dir, "logical_ci_trace.rds")
)
deletion_trace <- utils::read.csv(
  file.path(source_oracle_dir, "deletion_trace.csv"),
  stringsAsFactors = FALSE
)
source_manifest <- jsonlite::read_json(
  file.path(source_oracle_dir, "manifest.json"), simplifyVector = TRUE
)

expected_data_hash <-
  "971f1e0784817c6febfc84154f19540be02a92f6e4acad784dc7f92b979f1df7"
data_hash <- digest::digest(data, algo = "sha256", serialize = TRUE)
if (!identical(data_hash, expected_data_hash) ||
    !identical(dim(data), c(351L, 48L)) ||
    !identical(dim(adjacency), c(48L, 48L))) {
  stop("default-Inf oracle canonical input identity changed", call. = FALSE)
}
if (!identical(as.integer(n_edgetests$level), 0:7) ||
    !identical(as.integer(n_edgetests$n_edgetests),
               c(2213L, 52659L, 125293L, 40694L,
                 13293L, 5422L, 835L, 80L))) {
  stop("default-Inf oracle level-7 source counts changed", call. = FALSE)
}

level <- 8L
alpha <- 0.1
index <- 1
num_col <- 35L
p <- ncol(data)
snapshot <- adjacency
live_adjacency <- adjacency
ind <- which(snapshot, arr.ind = TRUE)
ind <- ind[order(ind[, 1L]), , drop = FALSE]
env <- fastkpc_legacy_env()
suff_stat <- list(
  data = data,
  ic.method = "dcc.gamma",
  index = index,
  numCol = num_col
)

double_bits <- function(value) {
  raw <- writeBin(as.double(value), raw(), size = 8L, endian = "little")
  paste(sprintf("%02x", as.integer(raw)), collapse = "")
}

task_rows <- list()
pcalg_done <- TRUE
for (row in seq_len(nrow(ind))) {
  x <- ind[row, 1L]
  y <- ind[row, 2L]
  if (!isTRUE(live_adjacency[y, x])) next
  neighbor_mask <- snapshot[, x]
  neighbor_mask[y] <- FALSE
  neighbors <- seq_len(p)[neighbor_mask]
  if (length(neighbors) < level) next
  if (length(neighbors) > level) pcalg_done <- FALSE
  conditioning_sets <- utils::combn(
    neighbors, level, simplify = FALSE
  )
  for (S in conditioning_sets) {
    p_value <- env$kernelCItest(
      x = x, y = y, S = S, suffStat = suff_stat
    )
    if (length(p_value) == 0L || !is.numeric(p_value) || is.na(p_value[[1L]])) {
      p_value <- 1
    } else {
      p_value <- as.numeric(p_value[[1L]])
    }
    if (pmax[x, y] < p_value) pmax[x, y] <- p_value
    deletes_edge <- p_value >= alpha
    task_rows[[length(task_rows) + 1L]] <- data.frame(
      source_task_index = length(task_rows) + 1L,
      level = level,
      x = x,
      y = y,
      x_label = colnames(data)[x],
      y_label = colnames(data)[y],
      S_key = paste(S, collapse = "|"),
      S_labels = paste(colnames(data)[S], collapse = "|"),
      p_value = p_value,
      p_value_bits_le = double_bits(p_value),
      deletes_edge = deletes_edge,
      stringsAsFactors = FALSE
    )
    if (deletes_edge) {
      live_adjacency[x, y] <- live_adjacency[y, x] <- FALSE
      sepsets[[x]][[y]] <- as.integer(S)
      break
    }
  }
}
tasks <- do.call(rbind, task_rows)
if (is.null(tasks)) {
  stop("default-Inf oracle produced no level-8 tasks", call. = FALSE)
}

for (i in seq_len(p - 1L)) {
  for (j in seq.int(i + 1L, p)) {
    pmax[i, j] <- pmax[j, i] <- max(pmax[i, j], pmax[j, i])
  }
}

expected_pairs <- c(2L, 11L, 13L, 31L, 32L, 35L, 39L, 41L, 45L)
if (!identical(nrow(tasks), 9L) ||
    !identical(as.integer(tasks$x), rep.int(3L, 9L)) ||
    !identical(as.integer(tasks$y), expected_pairs) ||
    any(tasks$deletes_edge) || !isTRUE(pcalg_done)) {
  stop("default-Inf oracle level-8 canonical structure changed",
       call. = FALSE)
}
remaining_degrees <- colSums(live_adjacency)
if (as.integer(max(remaining_degrees)) != 9L ||
    any(remaining_degrees > level + 1L)) {
  stop("default-Inf oracle natural-stop proof failed", call. = FALSE)
}

new_trace <- data.frame(
  logical_sequence_id = max(logical_trace$logical_sequence_id) +
    seq_len(nrow(tasks)),
  source_sequence_id = max(logical_trace$source_sequence_id) +
    seq_len(nrow(tasks)),
  source_task_index = as.integer(tasks$source_task_index),
  level = rep.int(level, nrow(tasks)),
  x = as.integer(tasks$x),
  y = as.integer(tasks$y),
  S_key = tasks$S_key,
  p_value = tasks$p_value,
  deletes_edge = tasks$deletes_edge,
  stringsAsFactors = FALSE
)
logical_trace <- rbind(logical_trace, new_trace)
n_edgetests <- rbind(
  n_edgetests,
  data.frame(level = level, n_edgetests = nrow(tasks))
)

reference_tasks <- data.frame(
  canonical_test_order_id = as.integer(logical_trace$logical_sequence_id),
  level = as.integer(logical_trace$level),
  task_index = as.integer(logical_trace$source_task_index),
  x = as.integer(logical_trace$x),
  y = as.integer(logical_trace$y),
  S_key = as.character(logical_trace$S_key),
  p_used = as.numeric(logical_trace$p_value),
  native_edge_deleted = as.logical(logical_trace$deletes_edge),
  native_edge_ignored = FALSE,
  stringsAsFactors = FALSE
)
reference <- list(
  adjacency = live_adjacency,
  sepsets = sepsets,
  pMax = pmax,
  n.edgetests = as.integer(n_edgetests$n_edgetests),
  tasks = reference_tasks,
  summary = list(
    unknown_fallback_count = 0L,
    approximate_backend_count = 0L
  )
)
command <- paste(
  "Rscript fastkpc/tools/run_full_cuda_ci_default_inf_oracle.R",
  shQuote(data_path), shQuote(source_oracle_dir), shQuote(output_dir)
)
written <- fastkpc_write_full_cuda_ci_oracle(
  reference = reference,
  data = data,
  output_dir = output_dir,
  alpha = alpha,
  index = index,
  numCol = num_col,
  max_conditioning_size = p - 2L,
  logical_trace_source = reference,
  oracle_route_environment = c(m.max = "Inf", method = "stable"),
  commands = command
)
saveRDS(tasks, file.path(output_dir, "level8_tasks.rds"))
utils::write.csv(tasks, file.path(output_dir, "level8_tasks.csv"),
                 row.names = FALSE)

manifest <- written$manifest
manifest$semantics <- "pcalg::skeleton(method='stable', m.max=Inf)"
manifest$requested_max_conditioning_size <- "Inf"
manifest$resolved_max_conditioning_size <- p - 2L
manifest$natural_stop_level <- level
manifest$n_edgetests <- as.integer(n_edgetests$n_edgetests)
manifest$level8_task_count <- nrow(tasks)
manifest$level8_deletion_count <- sum(tasks$deletes_edge)
manifest$maximum_degree_before_level8 <- max(colSums(snapshot))
manifest$maximum_degree_after_level8 <- max(remaining_degrees)
manifest$level9_task_possible <- any(remaining_degrees > level + 1L)
manifest$source_oracle_schema <- source_manifest$schema_version
manifest$source_oracle_dir <- source_oracle_dir
manifest$source_adjacency_sha256 <- digest::digest(
  file = file.path(source_oracle_dir, "adjacency.rds"), algo = "sha256"
)
manifest$pcalg_version <- as.character(utils::packageVersion("pcalg"))
manifest$RSpectra_version <- as.character(utils::packageVersion("RSpectra"))
fastkpc_full_cuda_write_json(
  manifest, file.path(output_dir, "manifest.json")
)

cat("default-Inf oracle: ", output_dir, "\n", sep = "")
cat("n.edgetests: ", paste(n_edgetests$n_edgetests, collapse = ","), "\n",
    sep = "")
cat("level 8 tasks/deletions: ", nrow(tasks), "/",
    sum(tasks$deletes_edge), "\n", sep = "")
