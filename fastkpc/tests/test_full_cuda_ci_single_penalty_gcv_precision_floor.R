source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_shadow.R")
source("fastkpc/R/full_cuda_ci_phase3_artifacts.R")
source("fastkpc/R/full_cuda_ci_phase35_contracts.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_single_penalty_gcv.R")
source("fastkpc/R/full_cuda_ci_phase4_artifacts.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS", unset = "0"), "1")) {
  cat("SKIP Phase 4 precision-floor replay: FASTKPC_RUN_CUDA_TESTS != 1\n")
  quit(save = "no", status = 0)
}
if (!isTRUE(tryCatch(fastkpc_cuda_available(), error = function(error) FALSE))) {
  cat("SKIP Phase 4 precision-floor replay: CUDA unavailable\n")
  quit(save = "no", status = 0)
}

regression_cases <- data.frame(
  prepared_s_key_sha256 = c(
    "0df3599d3b6c6ea6e3c99f5b80fbbb51b70797494699116f33d88fa3eca808d1",
    "45b718aaf9dab33661f5ed6ca512bcfafca0371036381be0f3045d05d3a8bebb",
    "dbd7c0cce091b2d0337397f83aa9504ff0e1c9e69eaa2a1666306b92f343d5c8"
  ),
  target = c(2L, 2L, 3L),
  stringsAsFactors = FALSE
)

catalog <- fastkpc_full_cuda_open_fixed_sp_catalog(
  file.path("fastkpc", "artifacts", "full_cuda_ci", "oracle_351x48_v1"),
  file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "workload_census_351x48_v1"
  ),
  file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1"
  ),
  file.path(
    "fastkpc", "artifacts", "kpc_tprs_real_zhu",
    "cancer_RD-causalDiscoveryInput.rds"
  ),
  require_full = TRUE
)
scope <- fastkpc_full_cuda_phase4_single_penalty_scope(catalog)
numerical <- fastkpc_full_cuda_phase35_load_contract("numerical_contract_v1")
residual_max_absolute <- as.numeric(
  numerical$payload$tolerances$residual$max_absolute
)
residual_relative_l2 <- as.numeric(
  numerical$payload$tolerances$residual$relative_l2
)

runtime <- fixed_sp_cuda_runtime_create(0L)
runtime_freed <- FALSE
on.exit({
  if (!runtime_freed) try(fixed_sp_cuda_runtime_free(runtime), silent = TRUE)
}, add = TRUE)
capacity <- fastkpc_full_cuda_fixed_sp_contract()$canonical_capacities
fixed_sp_cuda_runtime_reserve(
  runtime, capacity$n, capacity$null_dim, capacity$target_count,
  capacity$penalty_count, capacity$augmented_rows
)

subset_batch_target <- function(batch, target) {
  index <- which(batch$target_ids == target)
  assert_true(
    length(index) == 1L,
    "Phase 4 precision-floor fixture target is missing or duplicated"
  )
  batch$states <- batch$states[index, , drop = FALSE]
  batch$metadata <- batch$metadata[index, , drop = FALSE]
  batch$Y <- batch$Y[, index, drop = FALSE]
  batch$oracle_sp <- batch$oracle_sp[index]
  batch$target_keys <- batch$target_keys[index]
  batch$target_ids <- batch$target_ids[index]
  batch$planned_route <- batch$planned_route[index]
  batch
}

result_rows <- lapply(seq_len(nrow(regression_cases)), function(case_index) {
  setup_key <- regression_cases$prepared_s_key_sha256[[case_index]]
  target <- regression_cases$target[[case_index]]
  setup_position <- match(
    setup_key, as.character(scope$setup_rows$prepared_s_key_sha256)
  )
  assert_true(
    !is.na(setup_position),
    "Phase 4 precision-floor fixture setup is outside the canonical scope"
  )
  cat(
    "Phase 4 precision-floor case", case_index, "/",
    nrow(regression_cases), "\n"
  )
  flush.console()

  shard <- fastkpc_full_cuda_phase4_read_shard(
    catalog, scope, scope$shard_id[[setup_position]]
  )
  batch <- subset_batch_target(
    fastkpc_full_cuda_phase4_batch_from_shard(catalog, shard, setup_key),
    target
  )
  direct <- fastkpc_full_cuda_phase4_cuda_batch(
    batch$setup, batch$Y, target_ids = batch$target_ids
  )
  direct_diagnostics <- direct$diagnostics
  assert_true(
    direct_diagnostics$exact_replay_target_count == 1L &&
      direct_diagnostics$exact_replay_numerical_risk_count == 1L,
    "Phase 4 precision-floor target must use numerical-risk exact replay"
  )
  assert_true(
    direct_diagnostics$legacy_mgcv_target_calls == 0L &&
      direct_diagnostics$cpu_score_count == 0L &&
      direct_diagnostics$cpu_optimizer_count == 0L &&
      direct_diagnostics$fallback_count == 0L,
    "Phase 4 precision-floor replay must remain CUDA-only"
  )

  integrated <- fastkpc_full_cuda_phase4_execute_integrated_setup(
    runtime, batch
  )
  integrated_diagnostics <- integrated$diagnostics
  assert_true(
    integrated_diagnostics$exact_replay_target_count == 1L &&
      integrated_diagnostics$exact_replay_numerical_risk_count == 1L &&
      integrated_diagnostics$legacy_mgcv_target_calls == 0L &&
      integrated_diagnostics$fallback_count == 0L,
    "Phase 4 integrated precision-floor replay must remain CUDA-only"
  )
  assert_true(
    identical(
      as.numeric(integrated$targets$sp), as.numeric(direct$targets$sp)
    ),
    "Phase 4 direct and integrated precision-floor selections must agree"
  )

  oracle <- fastkpc_full_cuda_phase4_reference_fit(
    batch$setup, batch$Y, batch$oracle_sp
  )
  residual_error <- fastkpc_full_cuda_phase4_column_errors(
    integrated$shadow$residuals, oracle$residuals
  )
  assert_true(
    residual_error$max_absolute[[1L]] <= residual_max_absolute &&
      residual_error$relative_l2[[1L]] <= residual_relative_l2,
    "Phase 4 precision-floor replay must satisfy the residual contract"
  )

  data.frame(
    prepared_s_key_sha256 = setup_key,
    target = target,
    reported_rms_gradient = direct$targets$reported_rms_gradient[[1L]],
    log_sp_error = direct$targets$log_sp[[1L]] - log(batch$oracle_sp[[1L]]),
    residual_max_absolute = residual_error$max_absolute[[1L]],
    residual_relative_l2 = residual_error$relative_l2[[1L]],
    stringsAsFactors = FALSE
  )
})

fixed_sp_cuda_runtime_free(runtime)
runtime_freed <- TRUE
results <- do.call(rbind, result_rows)
print(results, row.names = FALSE, digits = 16)
cat("PASS Phase 4 precision-floor exact replay regression\n")
