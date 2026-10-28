# Production canonical S2 with status-wait, L5 slots, L4 workers32 and L4 slots ON
# (committed 4a1ff6b, native 037b92f1...). Read-only review profile: the elapsed
# time under nsys is NOT performance evidence. Bitwise check vs Codex's ON1 receipt.
# Rscript run_profile.R OUTPUT_DIR
# FASTKPC_REVIEW_NATIVE_SHA256 overrides the expected native (e.g. a candidate build).
args <- commandArgs(TRUE); stopifnot(length(args) == 1L)
out <- args[[1L]]; dir.create(out, recursive = TRUE, showWarnings = FALSE)
stopifnot(!file.exists(file.path(out, "profile-result.rds")))
Sys.setenv(FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H = "1",
  FASTKPC_DCC_GAMMA_L5_OPTIMIZER_SLOTS = "1",
  FASTKPC_DCC_GAMMA_L4_OPTIMIZER_WORKERS32 = "1",
  FASTKPC_DCC_GAMMA_L4_OPTIMIZER_SLOTS = "1",
  FASTKPC_PHASE10_SETUP_OPTIMIZER_PIPELINE = "0",
  FASTKPC_DCC_GAMMA_LEVEL_COMPONENT_PREFILL = "1",
  FASTKPC_DCC_GAMMA_STRUCTURED_COMPONENT_PREFILL = "1",
  FASTKPC_DCC_GAMMA_L2_COMPONENT_PREFILL = "1",
  FASTKPC_DIAG_LEVEL_COMPONENT_ORDER = "1")
source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
sha <- function(p) digest::digest(file = p, algo = "sha256", serialize = FALSE)
native <- "fastkpc/build/fastkpc_cuda.so"
expected_native <- Sys.getenv("FASTKPC_REVIEW_NATIVE_SHA256",
  "037b92f109a727ea2c50c737fb1c2b70048221bcffe41f215a532622398ffb58")
stopifnot(identical(sha(native), expected_native))
dataset <- "fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"
data <- as.matrix(readRDS(dataset)); storage.mode(data) <- "double"
stopifnot(identical(dim(data), c(351L, 48L)))
invisible(full_cuda_ci_one_call_cache_control_native("configure", 262144L))
invisible(full_cuda_ci_one_call_cache_control_native("configure_target", 131072L))
invisible(full_cuda_ci_one_call_cache_control_native("reset"))
set.seed(20261020L)
epoch_start <- as.numeric(Sys.time())
started <- proc.time()[["elapsed"]]
result <- fastkpc_compatible_cuda_skeleton(data, alpha = .1, options = list(
  route = "full_cuda", compatible_cuda_strict = TRUE, max_conditioning_size = Inf,
  index = 1, numCol = 35L, trace_level = "logical", ci_method = "dcc.gamma"))
elapsed <- unname(proc.time()[["elapsed"]] - started)
epoch_end <- as.numeric(Sys.time())
ref <- readRDS("fastkpc/artifacts/full_cuda_ci/optimizer_l4_slot_runtime_20261024/on1/receipt.rds")$result
bitwise <- identical(result$tasks$p_used, ref$tasks$p_used) &&
  identical(result$pMax, ref$pMax) && identical(result$adjacency, ref$adjacency) &&
  identical(result$n.edgetests, ref$n.edgetests)
s <- result$summary
saveRDS(list(elapsed = elapsed, epoch_start = epoch_start, epoch_end = epoch_end,
  native_sha256 = expected_native,
  device_properties_cache = Sys.getenv("FASTKPC_CUDA_DEVICE_PROPERTIES_CACHE"),
  bitwise_vs_codex_on1 = bitwise, levels = result$levels,
  prefill_batches = s$prefill_batches, l4 = s$l4_optimizer_slots, l5 = s$l5_optimizer_slots),
  file.path(out, "profile-result.rds"))
jsonlite::write_json(list(elapsed = elapsed, epoch_start = epoch_start, epoch_end = epoch_end,
  native_sha256 = expected_native,
  device_properties_cache = Sys.getenv("FASTKPC_CUDA_DEVICE_PROPERTIES_CACHE"),
  bitwise_vs_codex_on1 = bitwise, levels = result$levels,
  prefill_batches = s$prefill_batches), file.path(out, "profile-result.json"),
  auto_unbox = TRUE, digits = NA, pretty = TRUE)
cat("elapsed", elapsed, "bitwise", bitwise, "\n")
stopifnot(bitwise)
