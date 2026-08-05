source("fastkpc/R/fast_kpc.R")

args <- commandArgs(trailingOnly = TRUE)
arg_or_default <- function(index, default) {
  if (length(args) >= index && nzchar(args[[index]])) args[[index]] else default
}

candidate_path <- arg_or_default(1L, "")
oracle_path <- arg_or_default(2L, "")
method <- arg_or_default(3L, "")
output_path <- arg_or_default(4L, "")
data_path <- arg_or_default(
  5L,
  "fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"
)
seed <- as.integer(arg_or_default(6L, "707"))
tolerance <- as.numeric(arg_or_default(
  7L, if (method == "hsic.gamma") "1e-10" else "0"
))
gate_mode <- arg_or_default(8L, "native_cuda")

fail <- function(message) stop(message, call. = FALSE)
require_true <- function(value, message) if (!isTRUE(value)) fail(message)
parse_conditioning_set <- function(key) {
  if (!nzchar(key)) return(integer())
  as.integer(strsplit(key, "|", fixed = TRUE)[[1L]])
}
conditioning_order_key <- function(key) {
  values <- parse_conditioning_set(key)
  if (!length(values)) return("")
  paste(sprintf("%010d", values), collapse = "|")
}
task_identity <- function(level, x, y, S_key) {
  paste(as.integer(level), as.integer(x), as.integer(y),
        as.character(S_key), sep = "\037")
}
normalize_undirected_sepsets <- function(value, p) {
  output <- vector("list", p * (p - 1L) / 2L)
  position <- 0L
  for (left in seq_len(p - 1L)) {
    for (right in seq.int(left + 1L, p)) {
      position <- position + 1L
      output[[position]] <- sort(unique(c(
        as.integer(value[[left]][[right]]),
        as.integer(value[[right]][[left]])
      )))
    }
  }
  output
}
trace_rows <- function(value) {
  if (!length(value)) {
    return(data.frame(
      target = integer(), other = integer(), subset_key = character(),
      residual_conditioning_key = character(), p_value = numeric(),
      rejected = logical(), stringsAsFactors = FALSE
    ))
  }
  do.call(rbind, lapply(value, function(entry) {
    data.frame(
      target = as.integer(entry$target),
      other = as.integer(entry$other),
      subset_key = paste(sort(as.integer(entry$subset)), collapse = "|"),
      residual_conditioning_key = paste(
        sort(as.integer(entry$residual_conditioning_set)), collapse = "|"
      ),
      p_value = as.numeric(entry$p.value),
      rejected = isTRUE(entry$rejected),
      stringsAsFactors = FALSE
    )
  }))
}
atomic_save_rds <- function(value, path) {
  temporary <- tempfile(".strict-wanpdag-", tmpdir = dirname(path))
  on.exit(unlink(temporary, force = TRUE), add = TRUE)
  saveRDS(value, temporary, compress = "xz")
  require_true(file.rename(temporary, path),
               "failed to publish WAN-PDAG receipt")
}

require_true(method %in% c(
  "dcc.gamma", "dcc.perm", "hsic.gamma", "hsic.perm"
), "unsupported CI method")
require_true(file.exists(candidate_path), "candidate result is missing")
require_true(file.exists(oracle_path), "trace oracle is missing")
require_true(file.exists(data_path), "canonical data is missing")
require_true(nzchar(output_path) && !file.exists(output_path),
             "output path is invalid or already exists")
require_true(is.finite(tolerance) && tolerance >= 0,
             "tolerance must be finite and non-negative")
require_true(gate_mode %in% c("native_cuda", "kpcalg_authority"),
             "gate mode must be native_cuda or kpcalg_authority")

payload <- readRDS(candidate_path)
candidate <- if (is.list(payload) && !is.null(payload$result)) {
  payload$result
} else {
  payload
}
oracle <- readRDS(oracle_path)
data <- as.matrix(readRDS(data_path))
storage.mode(data) <- "double"
tasks <- candidate$tasks
oracle_method <- if (is.null(oracle$method)) method else
  as.character(oracle$method)
require_true(
  identical(dim(data), c(351L, 48L)) && is.data.frame(tasks) &&
    nrow(tasks) > 0L && identical(candidate$summary$ci_method, method) &&
    is.list(oracle) && isTRUE(oracle$pass) &&
    identical(oracle_method, method) &&
    length(oracle$cpu_p_values) == nrow(tasks),
  "candidate and full trace oracle contracts differ"
)

cpu_p_values <- as.numeric(oracle$cpu_p_values)
require_true(all(is.finite(cpu_p_values)),
             "full trace oracle contains non-finite p-values")
task_keys <- task_identity(tasks$level, tasks$x, tasks$y, tasks$S_key)
require_true(!anyDuplicated(task_keys), "candidate task keys are not unique")
p_value_by_key <- new.env(
  hash = TRUE, parent = emptyenv(), size = max(29L, length(task_keys))
)
invisible(list2env(
  setNames(as.list(cpu_p_values), task_keys), envir = p_value_by_key
))

# Ask pcalg itself to reconstruct the stable skeleton and its physical test
# order from the full CPU p-value oracle.
provider_calls <- vector("list", nrow(tasks))
provider_count <- 0L
oracle_provider <- function(x, y, S, suffStat) {
  key <- task_identity(length(S), x, y, paste(sort(as.integer(S)),
                                              collapse = "|"))
  p_value <- get0(key, envir = p_value_by_key, inherits = FALSE)
  if (is.null(p_value) || length(p_value) != 1L || !is.finite(p_value)) {
    fail(paste0("pcalg replay requested unknown task: ", key))
  }
  provider_count <<- provider_count + 1L
  provider_calls[[provider_count]] <<- key
  p_value
}

suff_stat <- list(
  data = data,
  ic.method = method,
  index = 1,
  numCol = 35L,
  sig = 1,
  p = 100L
)
replayed_skeleton <- pcalg::skeleton(
  suffStat = suff_stat,
  indepTest = oracle_provider,
  alpha = 0.1,
  labels = colnames(data),
  m.max = ncol(data) - 2L,
  method = "stable",
  verbose = FALSE
)
provider_calls <- unlist(provider_calls[seq_len(provider_count)],
                         use.names = FALSE)
conditioning_keys <- vapply(
  as.character(tasks$S_key), conditioning_order_key, character(1L)
)
physical_order <- order(
  tasks$level, tasks$x, tasks$y, conditioning_keys, method = "radix"
)
candidate_physical_keys <- task_keys[physical_order]
logical_trace_identical <- identical(provider_calls, candidate_physical_keys)

replayed_adjacency <- methods::as(replayed_skeleton@graph, "matrix") != 0
candidate_adjacency <- candidate$adjacency != 0
skeleton_adjacency_identical <- identical(
  unname(replayed_adjacency), unname(candidate_adjacency)
)
skeleton_shd <- sum(abs(
  as.integer(replayed_adjacency) - as.integer(candidate_adjacency)
)) / 2L
skeleton_sepsets_identical <- identical(
  normalize_undirected_sepsets(replayed_skeleton@sepset, ncol(data)),
  normalize_undirected_sepsets(candidate$sepsets, ncol(data))
)
skeleton_pmax_difference <- max(abs(
  unname(replayed_skeleton@pMax) - unname(candidate$pMax)
))
skeleton_n_edgetests_identical <- identical(
  as.integer(replayed_skeleton@n.edgetests),
  as.integer(candidate$n.edgetests)
)

env <- fastkpc_legacy_env()
legacy_ci_trace <- list()
legacy_ci_count <- 0L
original_regrvonps <- env$regrVonPS
env$regrVonPS <- function(G, V, S, suffStat, indepTest = env$kernelCItest,
                          alpha = 0.2) {
  parents <- which(G[, V] == 1 & G[V, ] == 0, arr.ind = TRUE)
  conditioning <- sort(unique(c(as.integer(S), as.integer(parents))))
  call_position <- 0L
  traced_indep_test <- function(x, y, S = NULL, suffStat) {
    call_position <<- call_position + 1L
    p_value <- indepTest(x = x, y = y, S = S, suffStat = suffStat)
    legacy_ci_count <<- legacy_ci_count + 1L
    legacy_ci_trace[[legacy_ci_count]] <<- list(
      target = as.integer(V),
      other = as.integer(S_outer[[call_position]]),
      subset = as.integer(S_outer),
      residual_conditioning_set = conditioning,
      p.value = as.numeric(p_value),
      rejected = as.numeric(p_value) < alpha
    )
    p_value
  }
  S_outer <- as.integer(S)
  original_regrvonps(
    G = G, V = V, S = S_outer, suffStat = suffStat,
    indepTest = traced_indep_test, alpha = alpha
  )
}

orientation_rng_start_source <- "not-applicable"
if (method %in% c("dcc.perm", "hsic.perm")) {
  shard_dir <- paste0(oracle_path, ".shards")
  shard_paths <- if (dir.exists(shard_dir)) {
    sort(list.files(
      shard_dir,
      pattern = "^shard-[0-9]+-of-[0-9]+\\.rds$",
      full.names = TRUE
    ))
  } else {
    character()
  }
  if (length(shard_paths) == as.integer(oracle$shard_count)) {
    final_shard <- readRDS(shard_paths[[length(shard_paths)]])
    require_true(
      identical(final_shard$schema_version,
                "fastkpc-strict-permutation-trace-shard-v1") &&
        identical(final_shard$method, method) &&
        identical(final_shard$seed, seed) &&
        identical(final_shard$candidate_md5, oracle$candidate_md5) &&
        identical(final_shard$shard, as.integer(oracle$shard_count)) &&
        identical(final_shard$shard_count, as.integer(oracle$shard_count)) &&
        isTRUE(final_shard$rng_state_identical) &&
        is.integer(final_shard$rng_end_state),
      "final permutation oracle shard RNG receipt is invalid"
    )
    skeleton_end_rng_state <- final_shard$rng_end_state
    orientation_rng_start_source <- "authenticated-final-oracle-shard"
  } else {
    set.seed(seed)
    fastkpc_consume_legacy_permutation_tasks_export(
      method, nrow(data), 100L, nrow(tasks)
    )
    skeleton_end_rng_state <- .Random.seed
    orientation_rng_start_source <- "full-rng-replay"
  }
} else {
  set.seed(seed)
  skeleton_end_rng_state <- .Random.seed
}
assign(".Random.seed", skeleton_end_rng_state, envir = .GlobalEnv)
legacy_started <- proc.time()
legacy_orientation <- env$udag2wanpdag(
  gInput = replayed_skeleton,
  suffStat = suff_stat,
  indepTest = env$kernelCItest,
  alpha = 0.1,
  verbose = FALSE,
  solve.confl = FALSE,
  orientCollider = TRUE,
  rules = c(TRUE, TRUE, TRUE)
)
legacy_elapsed <- unname((proc.time() - legacy_started)[["elapsed"]])
legacy_end_rng_state <- .Random.seed
legacy_pdag <- methods::as(legacy_orientation@graph, "matrix")
storage.mode(legacy_pdag) <- "integer"
legacy_trace <- trace_rows(legacy_ci_trace)

assign(".Random.seed", skeleton_end_rng_state, envir = .GlobalEnv)
authority_started <- proc.time()
authority_orientation <- fastkpc_orient_wanpdag_kpcalg_authority(
  skeleton = candidate,
  data = data,
  alpha = 0.1,
  index = 1,
  numCol = 35L,
  ci_method = method,
  hsic_params = list(sig = 1),
  permutation_params = if (method %in% c("dcc.perm", "hsic.perm")) {
    list(replicates = 100L, seed = seed, include_observed = TRUE)
  } else {
    list()
  },
  orient_collider = TRUE,
  solve_confl = FALSE,
  rules = c(TRUE, TRUE, TRUE),
  rng_state = skeleton_end_rng_state
)
authority_elapsed <- unname((proc.time() - authority_started)[["elapsed"]])
authority_end_rng_state <- .Random.seed
authority_pdag <- authority_orientation$pdag
authority_trace <- trace_rows(authority_orientation$ci_trace)

assign(".Random.seed", skeleton_end_rng_state, envir = .GlobalEnv)
native_started <- proc.time()
native_orientation <- fast_orient_wanpdag_cuda(
  candidate,
  data,
  alpha = 0.1,
  residual_backend = "fastSpline",
  orientation_residual_device = "cuda",
  residual_cache = TRUE,
  orientation_batch_size = 0L,
  orientation_diagnostics = TRUE,
  orient_collider = TRUE,
  solve_confl = FALSE,
  rules = c(TRUE, TRUE, TRUE),
  cuda_residual_fallback = FALSE,
  ci_method = method,
  hsic_params = list(sig = 1),
  permutation_params = if (method %in% c("dcc.perm", "hsic.perm")) {
    list(replicates = 100L, seed = seed, include_observed = TRUE)
  } else {
    list()
  },
  ci_diagnostics = TRUE
)
native_elapsed <- unname((proc.time() - native_started)[["elapsed"]])
native_end_rng_state <- .Random.seed
native_trace <- trace_rows(native_orientation$ci_trace)

trace_structure_fields <- c(
  "target", "other", "subset_key", "residual_conditioning_key"
)
authority_trace_structure_identical <- identical(
  legacy_trace[, trace_structure_fields, drop = FALSE],
  authority_trace[, trace_structure_fields, drop = FALSE]
)
authority_p_differences <- if (nrow(legacy_trace) == nrow(authority_trace)) {
  abs(legacy_trace$p_value - authority_trace$p_value)
} else {
  Inf
}
authority_p_within_tolerance <-
  nrow(legacy_trace) == nrow(authority_trace) &&
  all(authority_p_differences <= tolerance)
authority_decision_trace_identical <-
  nrow(legacy_trace) == nrow(authority_trace) &&
  identical(legacy_trace$rejected, authority_trace$rejected)
authority_pdag_identical <- identical(
  unname(legacy_pdag), unname(authority_pdag)
)
authority_matrix_diff <- max(abs(
  as.numeric(legacy_pdag) - as.numeric(authority_pdag)
))

orientation_trace_structure_identical <- identical(
  legacy_trace[, trace_structure_fields, drop = FALSE],
  native_trace[, trace_structure_fields, drop = FALSE]
)
orientation_p_differences <- if (nrow(legacy_trace) == nrow(native_trace)) {
  abs(legacy_trace$p_value - native_trace$p_value)
} else {
  Inf
}
orientation_p_within_tolerance <-
  nrow(legacy_trace) == nrow(native_trace) &&
  all(orientation_p_differences <= tolerance)
orientation_decision_trace_identical <-
  nrow(legacy_trace) == nrow(native_trace) &&
  identical(legacy_trace$rejected, native_trace$rejected)
orientation_pdag_identical <- identical(
  unname(legacy_pdag), unname(as.matrix(native_orientation$pdag))
)
orientation_matrix_diff <- max(abs(
  as.numeric(legacy_pdag) - as.numeric(native_orientation$pdag)
))

receipt <- list(
  schema_version = "fastkpc-strict-wanpdag-trace-oracle-v2",
  candidate_path = normalizePath(candidate_path),
  oracle_path = normalizePath(oracle_path),
  data_path = normalizePath(data_path),
  method = method,
  alpha = 0.1,
  seed = seed,
  tolerance = tolerance,
  gate_mode = gate_mode,
  orientation_rng_start_source = orientation_rng_start_source,
  skeleton_task_count = nrow(tasks),
  logical_trace_identical = logical_trace_identical,
  skeleton_adjacency_identical = skeleton_adjacency_identical,
  skeleton_SHD = as.integer(skeleton_shd),
  skeleton_sepsets_identical = skeleton_sepsets_identical,
  skeleton_pmax_within_tolerance =
    skeleton_pmax_difference <= tolerance,
  skeleton_max_abs_pmax_diff = skeleton_pmax_difference,
  skeleton_n_edgetests_identical = skeleton_n_edgetests_identical,
  legacy_orientation_elapsed_sec = legacy_elapsed,
  kpcalg_authority_orientation_elapsed_sec = authority_elapsed,
  native_orientation_elapsed_sec = native_elapsed,
  legacy_ci_test_count = nrow(legacy_trace),
  kpcalg_authority_ci_test_count = nrow(authority_trace),
  native_ci_test_count = nrow(native_trace),
  kpcalg_authority_trace_structure_identical =
    authority_trace_structure_identical,
  kpcalg_authority_max_abs_p_diff = max(authority_p_differences),
  kpcalg_authority_p_within_tolerance = authority_p_within_tolerance,
  kpcalg_authority_decision_trace_identical =
    authority_decision_trace_identical,
  kpcalg_authority_pdag_identical = authority_pdag_identical,
  kpcalg_authority_max_abs_matrix_diff = authority_matrix_diff,
  kpcalg_authority_rng_state_identical = identical(
    legacy_end_rng_state, authority_end_rng_state
  ),
  orientation_trace_structure_identical =
    orientation_trace_structure_identical,
  orientation_max_abs_p_diff = max(orientation_p_differences),
  orientation_p_within_tolerance = orientation_p_within_tolerance,
  orientation_decision_trace_identical =
    orientation_decision_trace_identical,
  orientation_pdag_identical = orientation_pdag_identical,
  orientation_max_abs_matrix_diff = orientation_matrix_diff,
  orientation_rng_state_identical = identical(
    legacy_end_rng_state, native_end_rng_state
  ),
  legacy_pdag = legacy_pdag,
  kpcalg_authority_pdag = authority_pdag,
  native_pdag = native_orientation$pdag,
  legacy_ci_trace = legacy_trace,
  kpcalg_authority_ci_trace = authority_trace,
  kpcalg_authority_diagnostics = authority_orientation$diagnostics,
  native_ci_trace = native_trace,
  native_diagnostics = native_orientation$diagnostics
)
receipt$kpcalg_authority_pass <-
  isTRUE(receipt$kpcalg_authority_trace_structure_identical) &&
  isTRUE(receipt$kpcalg_authority_p_within_tolerance) &&
  isTRUE(receipt$kpcalg_authority_decision_trace_identical) &&
  isTRUE(receipt$kpcalg_authority_pdag_identical) &&
  isTRUE(receipt$kpcalg_authority_rng_state_identical)
receipt$skeleton_pass <- isTRUE(receipt$logical_trace_identical) &&
  isTRUE(receipt$skeleton_adjacency_identical) &&
  identical(receipt$skeleton_SHD, 0L) &&
  isTRUE(receipt$skeleton_sepsets_identical) &&
  isTRUE(receipt$skeleton_pmax_within_tolerance) &&
  isTRUE(receipt$skeleton_n_edgetests_identical)
receipt$native_cuda_pass <-
  isTRUE(receipt$orientation_trace_structure_identical) &&
  isTRUE(receipt$orientation_p_within_tolerance) &&
  isTRUE(receipt$orientation_decision_trace_identical) &&
  isTRUE(receipt$orientation_pdag_identical) &&
  isTRUE(receipt$orientation_rng_state_identical)
receipt$kpcalg_authority_route_pass <-
  isTRUE(receipt$skeleton_pass) &&
  isTRUE(receipt$kpcalg_authority_pass)
receipt$native_cuda_route_pass <-
  isTRUE(receipt$skeleton_pass) &&
  isTRUE(receipt$kpcalg_authority_pass) &&
  isTRUE(receipt$native_cuda_pass)
receipt$pass <- if (gate_mode == "kpcalg_authority") {
  receipt$kpcalg_authority_route_pass
} else {
  receipt$native_cuda_route_pass
}

atomic_save_rds(receipt, output_path)
cat(sprintf(
  paste0(
    paste0(
      "strict WAN-PDAG oracle: method=%s gate=%s skeleton_tasks=%d ",
      "orientation_tests=%d "
    ),
    paste0(
      "authority_max_abs_p_diff=%.17g authority_pass=%s ",
      "native_max_abs_p_diff=%.17g native_pdag_diff=%d pass=%s\n"
    )
  ),
  method, gate_mode, nrow(tasks), nrow(legacy_trace),
  receipt$kpcalg_authority_max_abs_p_diff,
  receipt$kpcalg_authority_pass,
  receipt$orientation_max_abs_p_diff,
  as.integer(receipt$orientation_max_abs_matrix_diff), receipt$pass
))
require_true(receipt$pass, paste0(
  "strict WAN-PDAG trace oracle gate failed: ", gate_mode
))
