source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_phase3_artifacts.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) {
  if (!isTRUE(value)) fail(message)
}

load_fastkpc_cuda_native()
if (!isTRUE(fastkpc_cuda_available())) {
  cat("SKIP Phase 3 real identity: CUDA unavailable\n")
  quit(save = "no", status = 0L)
}

catalog <- fastkpc_full_cuda_open_fixed_sp_catalog(
  file.path("fastkpc", "artifacts", "full_cuda_ci", "oracle_351x48_v1"),
  file.path("fastkpc", "artifacts", "full_cuda_ci", "workload_census_351x48_v1"),
  file.path("fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1"),
  file.path("fastkpc", "artifacts", "kpc_tprs_real_zhu",
            "cancer_RD-causalDiscoveryInput.rds"),
  require_full = TRUE
)
before <- .Call(
  "C_fixed_sp_cuda_test_resource_snapshot", PACKAGE = "fastkpc_cuda"
)
identity <- fastkpc_full_cuda_phase3_input_identity(catalog, 0L)
after <- .Call(
  "C_fixed_sp_cuda_test_resource_snapshot", PACKAGE = "fastkpc_cuda"
)

assert_true(identical(before, after),
            "real identity discovery creates no fixed-SP resources")
assert_true(
  identical(identity$device_id, 0L) &&
    identical(identity$phase2_source_commit,
              "42ef3efa08327056ffe5c9aad7a8953ff6864c7e") &&
    identical(identity$execution_provenance_state, "pre-run-capture") &&
    identical(identity$execution_sources_unchanged_after_run, FALSE) &&
    identical(identity$R_version, R.version.string) &&
    identical(identity$mgcv_version,
              as.character(utils::packageVersion("mgcv"))) &&
    grepl("^[0-9a-f]{64}$", identity$sha256),
  "real catalog identity retains lineage and current pre-run evidence"
)

cat("PASS Phase 3 real catalog identity (device 0, zero fixed-SP resources)\n")
