source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_single_penalty_gcv.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cuda.R")
source("fastkpc/R/full_cuda_ci_native_setup.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

assert_true(
  !"mgcv" %in% loadedNamespaces(),
  "mgcv must not be loaded before the native setup test"
)

single_shard <- readRDS(file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1",
  "shards", "shard_53.rds"
))
single_key <-
  "038625cf986592962b731c79bf4c53c0415c820c7175a56b374a1f4ede5d5e55"
single_oracle <- single_shard$prepared_s_setups[[single_key]]
multi_shard <- readRDS(file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1",
  "shards", "shard_1.rds"
))
multi_key <-
  "001245052f571033286b2dc7526c24dbe5ec5c221660c094a8b9f052376b91da"
multi_oracle <- multi_shard$prepared_s_setups[[multi_key]]

data <- as.matrix(readRDS(file.path(
  "fastkpc", "artifacts", "kpc_tprs_real_zhu",
  "cancer_RD-causalDiscoveryInput.rds"
)))
storage.mode(data) <- "double"
metadata <- readRDS(file.path(
  "fastkpc", "artifacts", "full_cuda_ci",
  "workload_census_351x48_v1", "same_s_setup_metadata.rds"
))
build_native <- function(shard, oracle, key) {
  row <- metadata[
    metadata$same_S_group_id == oracle$same_S_group_id, , drop = FALSE
  ]
  row$prepared_s_key_sha256 <- key
  states <- shard$target_states[
    shard$target_states$prepared_s_key_sha256 == key, , drop = FALSE
  ]
  catalog <- list(inputs = list(
    data = data, dataset_sha256 = oracle$dataset_sha256
  ))
  setup <- fastkpc_full_cuda_phase7_runtime_setup(
    catalog, row, states, key
  )
  target <- fastkpc_full_cuda_materialize_target_state(
    states[1L, , drop = FALSE], data, oracle$dataset_sha256
  )
  fastkpc_full_cuda_validate_materialized_target_for_prepared(setup, target)
  setup
}

single <- build_native(single_shard, single_oracle, single_key)
multi <- build_native(multi_shard, multi_oracle, multi_key)

single_geometry <- fastkpc_full_cuda_phase4_spectral_prepare(single)
multi_geometry <- fastkpc_full_cuda_phase6_prepare(multi)

assert_true(
  !"mgcv" %in% loadedNamespaces(),
  "native Phase 4/6 setup preparation loaded mgcv"
)
assert_true(
  identical(
    single_geometry$native_geometry_diagnostics$legacy_mgcv_mroot_count,
    0L
  ) && identical(
    single_geometry$native_geometry_diagnostics$legacy_mgcv_initial_sp_count,
    0L
  ) && identical(
    single_geometry$native_geometry_diagnostics$r_qr_count, 0L
  ) && identical(
    multi_geometry$native_geometry_diagnostics$legacy_mgcv_mroot_count,
    0L
  ) && identical(
    multi_geometry$native_geometry_diagnostics$legacy_mgcv_initial_sp_count,
    0L
  ) && identical(
    multi_geometry$native_geometry_diagnostics$r_qr_count, 0L
  ) && identical(
    single$native_setup_diagnostics$legacy_mgcv_setup_count, 0L
  ) && identical(
    single$native_setup_diagnostics$r_callback_count, 0L
  ) && identical(
    multi$native_setup_diagnostics$legacy_mgcv_setup_count, 0L
  ) && identical(
    multi$native_setup_diagnostics$r_callback_count, 0L
  ),
  "native geometry diagnostics report a legacy setup dependency"
)

cat("PASS Phase 7 native setup without mgcv namespace\n")
