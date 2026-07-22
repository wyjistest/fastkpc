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
assert_error <- function(expression, message) {
  error <- tryCatch({
    force(expression)
    NULL
  }, error = function(error) error)
  assert_true(inherits(error, "error"), message)
}

catalog <- fastkpc_full_cuda_open_fixed_sp_catalog(
  file.path("fastkpc", "artifacts", "full_cuda_ci", "oracle_351x48_v1"),
  file.path("fastkpc", "artifacts", "full_cuda_ci", "workload_census_351x48_v1"),
  file.path("fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1"),
  file.path("fastkpc", "artifacts", "kpc_tprs_real_zhu",
            "cancer_RD-causalDiscoveryInput.rds"),
  require_full = TRUE
)
authority <- fastkpc_full_cuda_phase3_discover_catalog_authority(catalog)
assert_true(
  identical(
    fastkpc_full_cuda_phase3_discover_catalog_evidence(catalog),
    authority$lineage
  ),
  "real catalog default authority discovery returns authenticated lineage"
)
catalog_authority_registry <- get(
  ".fastkpc_full_cuda_phase3_catalog_authority_registry", envir = .GlobalEnv
)
catalog_authority_token <- catalog$phase3_catalog_authority_token
catalog_authority_key <- catalog_authority_token$authority_sha256
catalog_authority_record <- get(
  catalog_authority_key, envir = catalog_authority_registry, inherits = FALSE
)
mutated_catalog_with_copied_token <- catalog
mutated_catalog_with_copied_token$phase2_manifest$source_commit <-
  strrep("e", 40L)
assert_error(
  fastkpc_full_cuda_phase3_discover_catalog_evidence(
    mutated_catalog_with_copied_token
  ),
  "copied authority token into a mutated catalog must fail"
)
source_hash_mutated_catalog <- catalog
source_hash_mutated_catalog$phase0_manifest_hash <-
  paste0("0", substr(catalog$phase0_manifest_hash, 2L, 64L))
assert_error(
  fastkpc_full_cuda_phase3_discover_catalog_evidence(
    source_hash_mutated_catalog
  ),
  "catalog authority must reject source hash mutation"
)
rm(list = catalog_authority_key, envir = catalog_authority_registry)
assert_error(
  fastkpc_full_cuda_phase3_discover_catalog_evidence(catalog),
  "missing authority registry entry must fail"
)
assign(
  catalog_authority_key,
  catalog_authority_record,
  envir = catalog_authority_registry
)
stale_catalog_authority_record <- catalog_authority_record
stale_catalog_authority_record$phase2_source_commit <- strrep("d", 40L)
stale_catalog_authority_record$sha256 <-
  .fastkpc_full_cuda_phase3_catalog_authority_hash(
    stale_catalog_authority_record
  )
assign(
  catalog_authority_key,
  stale_catalog_authority_record,
  envir = catalog_authority_registry
)
assert_error(
  fastkpc_full_cuda_phase3_discover_catalog_evidence(catalog),
  "stale authority registry entry must fail"
)
assign(
  catalog_authority_key,
  catalog_authority_record,
  envir = catalog_authority_registry
)

if (identical(
  Sys.getenv("FASTKPC_PHASE3_REAL_IDENTITY_MODE"),
  "catalog-authority-only"
)) {
  cat("PASS Phase 3 real catalog authority identity\n")
  quit(save = "no", status = 0L)
}

fastkpc_full_cuda_phase3_discover_qualified_native_evidence()
if (!isTRUE(.Call("C_fastkpc_cuda_available", PACKAGE = "fastkpc_cuda"))) {
  cat("SKIP Phase 3 real identity: CUDA unavailable\n")
  quit(save = "no", status = 0L)
}
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
