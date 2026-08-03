source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cuda.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS", unset = "0"), "1")) {
  cat("SKIP Phase 10 grouped QR prototype: FASTKPC_RUN_CUDA_TESTS != 1\n")
  quit(save = "no", status = 0)
}
if (!isTRUE(tryCatch(fastkpc_cuda_available(), error = function(error) FALSE))) {
  cat("SKIP Phase 10 grouped QR prototype: CUDA unavailable\n")
  quit(save = "no", status = 0)
}

cases <- data.frame(
  label = c("q28-p3", "q37-p4", "q46-p5", "q55-p6", "q64-p7"),
  shard_id = c(36L, 28L, 56L, 31L, 1L),
  prepared_s_key_sha256 = c(
    "18b367b21d9ab1f391df7f790af330b1e0770e0db7a83c032cfbeb73645b5b9a",
    "c8522edfd10de23093ae68eed6e2196fffec6daa944ec04c687c3c8e2ba7840e",
    "e24dde52f78eab81021dc092fb072111f0d27733d6056b5490162039f8e87094",
    "e931fdb62d19fe6739f096548cf013ddacd60a2d4f4d5d582ba609102625a605",
    "05b18f09b303d481e84ec14c6a3d11da92db2fcc0f9f213aa8319015f7369ae4"
  ),
  expected_q = c(28L, 37L, 46L, 55L, 64L),
  expected_penalties = 3:7,
  stringsAsFactors = FALSE
)
target_count <- 512L
timing_repetitions <- 7L
warp_counts <- c(2L, 4L, 8L)
comparison_fields <- c(
  "rss", "edf", "score", "condition", "aggregate_penalty_rank",
  "numerical_rank", "solver_info", "gradient", "hessian", "coefficients"
)
data <- readRDS(file.path(
  "fastkpc", "artifacts", "kpc_tprs_real_zhu",
  "cancer_RD-causalDiscoveryInput.rds"
))

rows <- list()
route_totals <- c(guarded = 0L, stable = 0L)
for (case_index in seq_len(nrow(cases))) {
  case <- cases[case_index, , drop = FALSE]
  shard <- readRDS(file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1",
    "shards", paste0("shard_", case$shard_id, ".rds")
  ))
  setup <- shard$prepared_s_setups[[case$prepared_s_key_sha256]]
  state_index <- which(
    shard$target_states$prepared_s_key_sha256 ==
      case$prepared_s_key_sha256
  )[[1L]]
  assert_true(
    !is.null(setup) && !is.na(state_index) &&
      ncol(setup$X) == case$expected_q &&
      length(setup$penalty_blocks) == case$expected_penalties,
    paste0("grouped QR prototype fixture is malformed: ", case$label)
  )
  context <- fastkpc_full_cuda_validate_materialized_target_for_prepared(
    setup,
    fastkpc_full_cuda_materialize_target_state(
      shard$target_states[state_index, , drop = FALSE],
      data, setup$dataset_sha256
    )
  )
  prepared <- fastkpc_full_cuda_phase6_prepare(setup)
  Y <- matrix(context$y, nrow = length(context$y), ncol = target_count)
  log_sp <- matrix(
    log(context$sp), nrow = length(context$sp), ncol = target_count
  )
  baseline <- fastkpc_full_cuda_phase6_evaluate_cuda(prepared, Y, log_sp)

  for (warps in warp_counts) {
    candidate <- fastkpc_full_cuda_phase10_grouped_evaluate_prototype(
      prepared, Y, log_sp,
      grouped_warps_per_block = warps,
      timing_repetitions = timing_repetitions
    )
    diagnostics <- candidate$prototype_diagnostics
    assert_true(
      identical(
        candidate$schema_version,
        "full-cuda-ci-grouped-evaluator-prototype-result-v1"
      ) && identical(
        candidate$evaluation$schema_version,
        "full-cuda-ci-multi-penalty-grouped-evaluation-prototype-v1"
      ) && identical(
        diagnostics$execution_strategy,
        "development-only-warp-grouped-qr-stable-svd-queue"
      ),
      paste0("grouped QR prototype schema drifted: ", case$label)
    )
    mismatch_fields <- grep(
      "mismatch_count$", names(diagnostics), value = TRUE
    )
    assert_true(
      isTRUE(diagnostics$exact_parity) &&
        length(mismatch_fields) > 0L &&
        all(vapply(
          diagnostics[mismatch_fields], identical, logical(1L), 0
        )),
      paste0("grouped QR internal bitwise parity failed: ", case$label)
    )
    assert_true(
      diagnostics$grouped_guarded_qr_count ==
        diagnostics$baseline_guarded_qr_count &&
        diagnostics$grouped_stable_svd_count ==
          diagnostics$baseline_stable_svd_count &&
        diagnostics$grouped_failure_queue_count ==
          diagnostics$grouped_stable_svd_count,
      paste0("grouped QR route/failure queue drifted: ", case$label)
    )
    assert_true(
      all(vapply(comparison_fields, function(field) {
        identical(baseline[[field]], candidate$evaluation[[field]])
      }, logical(1L))),
      paste0("grouped QR public evaluator parity failed: ", case$label)
    )
    rows[[length(rows) + 1L]] <- data.frame(
      label = case$label,
      q = case$expected_q,
      penalties = case$expected_penalties,
      warps_per_block = warps,
      guarded = diagnostics$grouped_guarded_qr_count,
      stable = diagnostics$grouped_stable_svd_count,
      baseline_qr_ms = diagnostics$baseline_qr_median_ms,
      grouped_qr_ms = diagnostics$grouped_qr_median_ms,
      throughput_speedup = diagnostics$qr_throughput_speedup,
      exact_parity = diagnostics$exact_parity,
      stringsAsFactors = FALSE
    )
    if (warps == warp_counts[[1L]]) {
      route_totals <- route_totals + c(
        guarded = diagnostics$grouped_guarded_qr_count,
        stable = diagnostics$grouped_stable_svd_count
      )
    }
  }
  if (identical(case$label[[1L]], "q37-p4")) {
    forced <- fastkpc_full_cuda_phase10_grouped_evaluate_prototype(
      prepared, Y, log_sp,
      force_stable_svd = TRUE,
      grouped_warps_per_block = 2L,
      timing_repetitions = 1L
    )
    forced_diagnostics <- forced$prototype_diagnostics
    assert_true(
      isTRUE(forced_diagnostics$exact_parity) &&
        forced_diagnostics$grouped_guarded_qr_count == 0L &&
        forced_diagnostics$grouped_stable_svd_count == target_count &&
        forced_diagnostics$grouped_failure_queue_count == target_count,
      "grouped QR explicit stable-SVD route drifted"
    )
  }
}

rows <- do.call(rbind, rows)
best_speedup <- max(rows$throughput_speedup)
decision <- if (best_speedup >= 1.5) {
  "PASS_TO_OPTIMIZER_INTEGRATION"
} else if (best_speedup < 1.3) {
  "STOP_BEFORE_OPTIMIZER_INTEGRATION"
} else {
  "HOLD_FOR_ADDITIONAL_PROFILING"
}
rows$decision <- decision
print(rows, row.names = FALSE, digits = 8)
assert_true(
  route_totals[["guarded"]] > 0L && route_totals[["stable"]] > 0L,
  "grouped QR campaign must cover guarded QR and stable SVD routes"
)
production_source <- paste(
  readLines("fastkpc/src/full_cuda_ci_one_call.cpp", warn = FALSE),
  collapse = "\n"
)
assert_true(
  !grepl("grouped_evaluate_prototype", production_source, fixed = TRUE) &&
    !grepl("grouped_prototype", production_source, fixed = TRUE),
  "development grouped QR prototype leaked into production one-call"
)
if (identical(
      Sys.getenv("FASTKPC_GROUPED_QR_REQUIRE_GO", unset = "0"), "1"
    )) {
  assert_true(
    identical(decision, "PASS_TO_OPTIMIZER_INTEGRATION"),
    paste0(
      "grouped QR throughput gate did not pass: best speedup=",
      format(best_speedup, digits = 17), ", decision=", decision
    )
  )
}
cat(
  "PASS Phase 10 grouped QR prototype exactness; throughput decision=",
  decision, ", best_speedup=", format(best_speedup, digits = 8), "\n",
  sep = ""
)
