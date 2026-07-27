source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_shadow.R")
source("fastkpc/R/full_cuda_ci_phase3_artifacts.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

assert_true(
  exists("fastkpc_full_cuda_phase3_publish_shadow_artifact",
         mode = "function"),
  "authenticated fixed-sp shadow artifact publisher should exist"
)
assert_true(
  exists("fastkpc_validate_full_cuda_fixed_sp_shadow_artifact",
         mode = "function"),
  "independent fixed-sp shadow artifact validator should exist"
)

assert_true(
  exists("fastkpc_full_cuda_shadow_merge_logical_rows", mode = "function"),
  "fixed-sp shadow direct/conditional normalizer should exist"
)

assert_true(
  all(vapply(c(
    ".fastkpc_full_cuda_phase3_shadow_phase0_authority",
    ".fastkpc_full_cuda_phase3_load_shadow_phase0_oracle",
    ".fastkpc_full_cuda_phase3_revalidate_shadow_phase0_authority"
  ), exists, logical(1L), mode = "function")),
  "shadow replay exposes fresh canonical Phase 0 loader authority"
)
assert_true(
  all(vapply(c(
    ".fastkpc_full_cuda_phase3_shadow_work_dir_prefix",
    ".fastkpc_full_cuda_phase3_shadow_work_bootstrap_prefix",
    ".fastkpc_full_cuda_phase3_acquire_shadow_work_dir",
    ".fastkpc_full_cuda_phase3_write_shadow_work_dir_marker",
    ".fastkpc_full_cuda_phase3_cleanup_shadow_work_dirs"
  ), exists, logical(1L), mode = "function")),
  paste(
    "shadow publication exposes atomic marked bootstrap acquisition and exact",
    "owned staging/rollback cleanup"
  )
)

focused_only <- identical(
  Sys.getenv("FASTKPC_TASK9_FOCUSED_ONLY", unset = "0"), "1"
)

focused_work_output <- tempfile("phase3-shadow-work-output-")
dir.create(focused_work_output, recursive = TRUE, showWarnings = FALSE)
on.exit(unlink(focused_work_output, recursive = TRUE, force = TRUE),
        add = TRUE)
focused_work_lock <-
  .fastkpc_full_cuda_phase3_acquire_shadow_artifact_lock(
    focused_work_output
  )
focused_staging <- .fastkpc_full_cuda_phase3_acquire_shadow_work_dir(
  focused_work_output, "staging", focused_work_lock
)
writeLines("partial payload", file.path(focused_staging, "payload.rds"),
           useBytes = TRUE)
.fastkpc_full_cuda_phase3_cleanup_shadow_work_dirs(
  focused_work_output, c("staging", "rollback"), focused_work_lock
)
assert_true(
  !file.exists(focused_staging),
  "owned stale shadow staging is removed under the publication OS lock"
)
focused_malformed_staging <- tempfile(
  .fastkpc_full_cuda_phase3_shadow_work_dir_prefix(
    focused_work_output, "staging"
  ),
  tmpdir = dirname(focused_work_output)
)
dir.create(focused_malformed_staging, showWarnings = FALSE)
writeLines("not an ownership marker", file.path(
  focused_malformed_staging, ".fastkpc-shadow-work-owner.json"
), useBytes = TRUE)
malformed_staging_error <- tryCatch({
  .fastkpc_full_cuda_phase3_cleanup_shadow_work_dirs(
    focused_work_output, "staging", focused_work_lock
  )
  NULL
}, error = function(error) error)
assert_true(
  inherits(malformed_staging_error, "error") &&
    dir.exists(focused_malformed_staging),
  "malformed shadow staging ownership fails closed without deletion"
)
unlink(focused_malformed_staging, recursive = TRUE, force = TRUE)
focused_symlink_target <- tempfile("phase3-shadow-work-target-")
dir.create(focused_symlink_target, showWarnings = FALSE)
focused_symlink_staging <- tempfile(
  .fastkpc_full_cuda_phase3_shadow_work_dir_prefix(
    focused_work_output, "staging"
  ),
  tmpdir = dirname(focused_work_output)
)
assert_true(
  file.symlink(focused_symlink_target, focused_symlink_staging),
  "focused staging fixture creates a sibling symlink"
)
symlink_staging_error <- tryCatch({
  .fastkpc_full_cuda_phase3_cleanup_shadow_work_dirs(
    focused_work_output, "staging", focused_work_lock
  )
  NULL
}, error = function(error) error)
assert_true(
  inherits(symlink_staging_error, "error") &&
    dir.exists(focused_symlink_target),
  "symlink shadow staging fails closed without deleting its target"
)
unlink(focused_symlink_staging, force = TRUE)
unlink(focused_symlink_target, recursive = TRUE, force = TRUE)
.fastkpc_full_cuda_phase3_release_shadow_artifact_lock(focused_work_lock)
focused_work_lock <- NULL

focused_prior_paths <- file.path(
  focused_work_output, c("payload.rds", "manifest.json", "summary.json")
)
writeLines("prior payload", focused_prior_paths[[1L]], useBytes = TRUE)
writeLines("prior manifest", focused_prior_paths[[2L]], useBytes = TRUE)
writeLines("prior summary", focused_prior_paths[[3L]], useBytes = TRUE)
focused_prior_hashes <- vapply(
  focused_prior_paths, fastkpc_full_cuda_census_file_hash, character(1L)
)
bootstrap_crash_script <- tempfile(
  "phase3-shadow-bootstrap-crash-", fileext = ".R"
)
writeLines(c(
  'source("fastkpc/R/full_cuda_ci_gate.R")',
  'source("fastkpc/R/full_cuda_ci_oracle_contract.R")',
  'source("fastkpc/R/full_cuda_ci_workload_census.R")',
  'source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")',
  'source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")',
  'source("fastkpc/R/full_cuda_ci_fixed_sp_shadow.R")',
  'source("fastkpc/R/full_cuda_ci_phase3_artifacts.R")',
  'args <- commandArgs(trailingOnly = TRUE)',
  'output_dir <- args[[1L]]',
  'purpose <- args[[2L]]',
  'event <- args[[3L]]',
  'lock <- .fastkpc_full_cuda_phase3_acquire_shadow_artifact_lock(output_dir)',
  paste0(
    '.fastkpc_full_cuda_phase3_acquire_shadow_work_dir(',
    'output_dir,purpose,lock,.publication_hook=function(observed) {',
    'if (identical(observed,event)) tools::pskill(Sys.getpid(),9L)})'
  )
), bootstrap_crash_script, useBytes = TRUE)
on.exit(unlink(bootstrap_crash_script, force = TRUE), add = TRUE)
focused_work_siblings <- function(output_dir) {
  entries <- list.files(
    dirname(output_dir), full.names = TRUE, all.files = TRUE, no.. = TRUE
  )
  names <- basename(entries)
  reserved <- vapply(c("staging", "rollback"), function(purpose) {
    startsWith(
      names,
      .fastkpc_full_cuda_phase3_shadow_work_dir_prefix(output_dir, purpose)
    )
  }, logical(length(entries)))
  bootstrap <- startsWith(
    names,
    .fastkpc_full_cuda_phase3_shadow_work_bootstrap_prefix(output_dir)
  )
  entries[bootstrap | rowSums(reserved) > 0L]
}
for (purpose in c("staging", "rollback")) {
  for (event in c(
    paste0("after_", purpose, "_bootstrap_create"),
    paste0("during_", purpose, "_marker_write"),
    paste0("after_", purpose, "_marker_validation")
  )) {
    crash_output <- suppressWarnings(system2(
      file.path(R.home("bin"), "Rscript"),
      args = c(
        bootstrap_crash_script, focused_work_output, purpose, event
      ),
      stdout = TRUE, stderr = TRUE
    ))
    assert_true(
      !is.null(attr(crash_output, "status")) &&
        attr(crash_output, "status") != 0L &&
        length(focused_work_siblings(focused_work_output)) == 1L,
      paste(
        "hard exit leaves one exact dead-owner work resource at", event,
        "status=", attr(crash_output, "status"),
        "output=", paste(crash_output, collapse = " | "),
        "siblings=", paste(
          basename(focused_work_siblings(focused_work_output)),
          collapse = ","
        )
      )
    )
    recovery_lock <-
      .fastkpc_full_cuda_phase3_acquire_shadow_artifact_lock(
        focused_work_output
      )
    .fastkpc_full_cuda_phase3_cleanup_shadow_work_dirs(
      focused_work_output, c("staging", "rollback"), recovery_lock
    )
    .fastkpc_full_cuda_phase3_release_shadow_artifact_lock(recovery_lock)
    assert_true(
      length(focused_work_siblings(focused_work_output)) == 0L &&
        identical(
          vapply(
            focused_prior_paths, fastkpc_full_cuda_census_file_hash,
            character(1L)
          ),
          focused_prior_hashes
        ),
      paste(
        "next locked publication cleanup removes dead-owner bootstrap state",
        "without poisoning the prior generation at", event
      )
    )
  }
}
publication_only <- identical(
  Sys.getenv("FASTKPC_TASK9_PUBLICATION_ONLY", unset = "0"), "1"
)
assert_true(
  exists(".fastkpc_full_cuda_phase3_expected_native_sha256",
         mode = "function"),
  "Phase 3 execution identity exposes explicit expected native SHA semantics"
)
production_native <- strrep("1", 64L)
test_native <- strrep("2", 64L)
assert_true(
  identical(
    .fastkpc_full_cuda_phase3_expected_native_sha256(
      list(
        schema_version = "full-cuda-ci-phase3-input-identity-v1",
        native_library_sha256 = production_native
      )
    ), production_native
  ) && identical(
    .fastkpc_full_cuda_phase3_expected_native_sha256(
      list(schema_version =
             "full-cuda-ci-phase3-test-input-identity-v1"),
      claimed_sha256 = test_native
    ), test_native
  ),
  "production native SHA comes from identity while tests use explicit claims"
)
assert_true(
  exists(".fastkpc_full_cuda_phase3_shadow_manifest_dispatch",
         mode = "function"),
  "shadow manifest dispatch distinguishes absent, null, and versioned schema"
)
focused_json_value <- function(type) switch(
  type, character = "fixture", logical = TRUE, integer = 1L,
  object = list(fixture = "value")
)
v2_dispatch <- lapply(
  .fastkpc_full_cuda_phase3_shadow_manifest_types(), focused_json_value
)
v2_dispatch$manifest_schema_version <-
  "full-cuda-ci-fixed-sp-shadow-manifest-v2"
v2_summary <- lapply(
  .fastkpc_full_cuda_phase3_shadow_summary_types(), focused_json_value
)
v2_summary$summary_schema_version <-
  "full-cuda-ci-fixed-sp-shadow-summary-v1"
focused_legacy_identity_fields <- setdiff(
  .fastkpc_full_cuda_phase3_identity_fields(), "schema_version"
)
focused_legacy_integer_fields <- c(
  "source_closure_count", "native_build_dependency_count",
  "native_build_exclusion_count", "cuda_toolkit_version",
  "cuda_driver_version", "compute_capability_major",
  "compute_capability_minor", "sm_count", "device_id", "shard_count"
)
focused_legacy_logical_fields <- c(
  "relevant_sources_dirty_or_untracked",
  "execution_sources_unchanged_after_run",
  "cublas_user_workspace_required", "authenticated"
)
focused_legacy_numeric_fields <- c(
  "cublas_workspace_bytes_required",
  "cublas_workspace_min_alignment_required"
)
focused_legacy_identity <- setNames(lapply(
  focused_legacy_identity_fields, function(field) {
    if (field %in% focused_legacy_integer_fields) return(1L)
    if (field %in% focused_legacy_logical_fields) return(TRUE)
    if (field %in% focused_legacy_numeric_fields) return(1)
    "fixture"
  }
), focused_legacy_identity_fields)
legacy_dispatch <- c(list(
  artifact_schema_version = fastkpc_full_cuda_phase3_shadow_schema_version(),
  artifact_kind = "full_shadow",
  input_identity_schema_version = "full-cuda-ci-phase3-input-identity-v1",
  input_identity_sha256 = strrep("a", 64L),
  payload_names = "logical_ci_parity.rds",
  publication_order = c(
    "logical_ci_parity.rds", "manifest.json", "summary.json"
  ),
  payload_file_sha256 = list(
    "logical_ci_parity.rds" = strrep("b", 64L)
  )
), focused_legacy_identity)
legacy_summary <- list(
  artifact_schema_version = fastkpc_full_cuda_phase3_shadow_schema_version(),
  artifact_kind = "full_shadow", manifest_sha256 = strrep("c", 64L),
  shard_count = 1L, payload_count = 1L, pass = TRUE
)
assert_true(
  identical(
    .fastkpc_full_cuda_phase3_shadow_manifest_dispatch(
      legacy_dispatch, legacy_summary
    ),
    "legacy"
  ) && identical(
    .fastkpc_full_cuda_phase3_shadow_manifest_dispatch(
      v2_dispatch, v2_summary
    ),
    "task9_v2"
  ),
  "only exact historical and exact v2 namespaces dispatch"
)
for (bad_manifest in list(
  list(manifest_schema_version = NULL),
  list(manifest_schema_version = "unknown-shadow-version"),
  list(
    manifest_schema_version = NULL,
    artifact_schema_version = fastkpc_full_cuda_phase3_shadow_schema_version(),
    artifact_kind = "full_shadow"
  )
)) {
  dispatch_error <- tryCatch({
    .fastkpc_full_cuda_phase3_shadow_manifest_dispatch(
      bad_manifest, legacy_summary
    )
    NULL
  }, error = function(error) error)
  assert_true(
    inherits(dispatch_error, "error"),
    "null, unknown, and hybrid shadow schema claims fail closed"
  )
}
for (hybrid in list(
  c(legacy_dispatch, list(source_inputs = list())),
  c(v2_dispatch, list(legacy_discriminator = "full_shadow"))
)) {
  hybrid_error <- tryCatch({
    .fastkpc_full_cuda_phase3_shadow_manifest_dispatch(
      hybrid, legacy_summary
    )
    NULL
  }, error = function(error) error)
  assert_true(
    inherits(hybrid_error, "error"),
    "legacy/v2 extra-field and hybrid schema claims fail closed"
  )
}
missing_v2_summary_error <- tryCatch({
  .fastkpc_full_cuda_phase3_shadow_manifest_dispatch(v2_dispatch, NULL)
  NULL
}, error = function(error) error)
assert_true(
  inherits(missing_v2_summary_error, "error"),
  "v2 dispatch requires its exact summary namespace"
)
legacy_summary_extra <- c(legacy_summary, list(v2_pass = TRUE))
legacy_summary_error <- tryCatch({
  .fastkpc_full_cuda_phase3_shadow_manifest_dispatch(
    legacy_dispatch, legacy_summary_extra
  )
  NULL
}, error = function(error) error)
assert_true(
  inherits(legacy_summary_error, "error"),
  "legacy dispatch requires the exact historical summary namespace"
)
assert_true(
  exists(".fastkpc_full_cuda_phase3_exact_json_namespace",
         mode = "function"),
  "Task 9 metadata validation exposes exact ordered namespace checking"
)
namespace_value <- list(schema_version = "v1", pass = TRUE)
invisible(.fastkpc_full_cuda_phase3_exact_json_namespace(
  namespace_value, c("schema_version", "pass"), "focused metadata"
))
for (bad_namespace in list(
  c(namespace_value, list(extra = 1L)),
  namespace_value["schema_version"],
  structure(
    list("v1", TRUE),
    names = c("schema_version", "schema_version")
  )
)) {
  namespace_error <- tryCatch({
    .fastkpc_full_cuda_phase3_exact_json_namespace(
      bad_namespace, c("schema_version", "pass"), "focused metadata"
    )
    NULL
  }, error = function(error) error)
  assert_true(
    inherits(namespace_error, "error"),
    "extra, missing, and duplicate metadata fields fail closed"
  )
}
assert_true(
  exists(".fastkpc_full_cuda_phase3_validate_native_claim",
         mode = "function"),
  "loaded Phase 3 evidence validates native SHA against identity authority"
)
production_identity_info <- list(
  identity_info_schema_version =
    "full-cuda-ci-phase3-validated-execution-identity-v2",
  production_identity = TRUE,
  expected_native_library_sha256 = production_native
)
test_identity_info <- list(
  production_identity = FALSE,
  expected_native_library_sha256 = NULL
)
invisible(.fastkpc_full_cuda_phase3_validate_native_claim(
  production_native, production_identity_info, "focused production claim"
))
invisible(.fastkpc_full_cuda_phase3_validate_native_claim(
  test_native, test_identity_info, "focused test claim"
))
native_rehash_error <- tryCatch({
  .fastkpc_full_cuda_phase3_validate_native_claim(
    test_native, production_identity_info, "coherent wrong production claim"
  )
  NULL
}, error = function(error) error)
assert_true(
  inherits(native_rehash_error, "error"),
  "coherently rehashed production native claims fail identity authority"
)
coherent_wrong_native_claims <- c(
  oracle_manifest = test_native, shadow_manifest = test_native,
  oracle_shard = test_native, shadow_shard = test_native
)
coherent_native_errors <- vapply(
  names(coherent_wrong_native_claims), function(label) {
    inherits(tryCatch({
      .fastkpc_full_cuda_phase3_validate_native_claim(
        coherent_wrong_native_claims[[label]], production_identity_info,
        paste("coherent", label)
      )
      NULL
    }, error = function(error) error), "error")
  }, logical(1L)
)
assert_true(
  all(coherent_native_errors),
  "coherently rehashed oracle/shadow shard and manifest claims all fail"
)
assert_true(
  exists(".fastkpc_full_cuda_phase3_validate_shadow_native_linkage",
         mode = "function"),
  "shadow validation exposes explicit three-way native identity linkage"
)
assert_true(
  exists(
    ".fastkpc_full_cuda_phase3_revalidate_completed_oracle_native",
    mode = "function"
  ),
  "completed oracle validation exposes a fresh end-of-use native gate"
)
focused_native_path <- tempfile("phase3-shadow-native-")
writeLines("authenticated native fixture", focused_native_path,
           useBytes = TRUE)
on.exit(unlink(focused_native_path, force = TRUE), add = TRUE)
focused_native_sha <- fastkpc_full_cuda_census_file_hash(focused_native_path)
focused_native_identity <- list(
  schema_version = "full-cuda-ci-phase3-input-identity-v1",
  native_library_path = normalizePath(focused_native_path, mustWork = TRUE),
  native_library_sha256 = focused_native_sha
)
invisible(.fastkpc_full_cuda_phase3_validate_shadow_native_linkage(
  focused_native_identity, focused_native_identity, focused_native_sha,
  "focused native linkage"
))
focused_embedded_wrong <- focused_native_identity
focused_embedded_wrong$native_library_sha256 <- strrep("d", 64L)
embedded_native_error <- tryCatch({
  .fastkpc_full_cuda_phase3_validate_shadow_native_linkage(
    focused_native_identity, focused_embedded_wrong, focused_native_sha,
    "focused embedded native mutation"
  )
  NULL
}, error = function(error) error)
assert_true(
  inherits(embedded_native_error, "error"),
  "refreshed embedded production native mutation fails three-way linkage"
)
focused_top_level_wrong <- tryCatch({
  .fastkpc_full_cuda_phase3_validate_shadow_native_linkage(
    focused_native_identity, focused_native_identity, strrep("e", 64L),
    "focused top-level native mutation"
  )
  NULL
}, error = function(error) error)
assert_true(
  inherits(focused_top_level_wrong, "error"),
  "top-level production native mutation fails three-way linkage"
)
focused_native_before <- readBin(
  focused_native_path, "raw", n = file.info(focused_native_path)$size
)
focused_native_replacement <- tempfile(
  "phase3-shadow-native-replacement-", tmpdir = dirname(focused_native_path)
)
writeLines("replacement native fixture", focused_native_replacement,
           useBytes = TRUE)
assert_true(
  file.rename(focused_native_replacement, focused_native_path),
  "focused native fixture is atomically replaced after its initial gate"
)
focused_final_native_error <- tryCatch({
  .fastkpc_full_cuda_phase3_revalidate_completed_oracle_native(
    focused_native_identity, focused_native_identity, focused_native_sha,
    "focused completed oracle final native gate"
  )
  NULL
}, error = function(error) error)
focused_native_connection <- file(focused_native_path, open = "wb")
writeBin(focused_native_before, focused_native_connection)
close(focused_native_connection)
assert_true(
  inherits(focused_final_native_error, "error") && grepl(
    "native", conditionMessage(focused_final_native_error),
    ignore.case = TRUE
  ) && grepl(
    "mismatch", conditionMessage(focused_final_native_error),
    ignore.case = TRUE
  ),
  "end-of-use native gate freshly hashes and rejects a replaced library"
)
assert_true(
  exists(
    ".fastkpc_full_cuda_phase3_with_owned_shadow_execution_snapshot",
    mode = "function"
  ),
  "helper-owned snapshot mint and release share one lifecycle boundary"
)
focused_snapshot_create <- 0L
focused_snapshot_release <- 0L
original_snapshot_create <- get(
  "fastkpc_full_cuda_phase3_create_shadow_execution_snapshot",
  envir = globalenv()
)
original_snapshot_release <- get(
  "fastkpc_full_cuda_phase3_release_shadow_execution_snapshot",
  envir = globalenv()
)
assign(
  "fastkpc_full_cuda_phase3_create_shadow_execution_snapshot",
  function(catalog, shadow_plan, scope) {
    focused_snapshot_create <<- focused_snapshot_create + 1L
    structure(list(id = focused_snapshot_create), class = "focused_snapshot")
  },
  envir = globalenv()
)
assign(
  "fastkpc_full_cuda_phase3_release_shadow_execution_snapshot",
  function(token) {
    focused_snapshot_release <<- focused_snapshot_release + 1L
    invisible(TRUE)
  },
  envir = globalenv()
)
focused_snapshot_value <-
  .fastkpc_full_cuda_phase3_with_owned_shadow_execution_snapshot(
    catalog = list(), shadow_plan = list(), scope = "iteration",
    callback = function(token) token$id
  )
focused_snapshot_failure <- tryCatch({
  .fastkpc_full_cuda_phase3_with_owned_shadow_execution_snapshot(
    catalog = list(), shadow_plan = list(), scope = "iteration",
    callback = function(token) stop(
      "focused helper failure after mint", call. = FALSE
    )
  )
  NULL
}, error = function(error) error)
assign(
  "fastkpc_full_cuda_phase3_create_shadow_execution_snapshot",
  original_snapshot_create, envir = globalenv()
)
assign(
  "fastkpc_full_cuda_phase3_release_shadow_execution_snapshot",
  original_snapshot_release, envir = globalenv()
)
assert_true(
  identical(focused_snapshot_value, 1L) &&
    inherits(focused_snapshot_failure, "error") &&
    identical(
      c(create = focused_snapshot_create, release = focused_snapshot_release),
      c(create = 2L, release = 2L)
    ),
  "helper-owned snapshots release exactly once on success and failure"
)
assert_true(
  all(vapply(c(
    ".fastkpc_full_cuda_phase3_acquire_shadow_artifact_lock",
    ".fastkpc_full_cuda_phase3_release_shadow_artifact_lock"
  ), exists, logical(1L), mode = "function")),
  "shadow publication exposes a dedicated artifact lock"
)
focused_lock_dir <- tempfile("phase3-shadow-focused-lock-")
dir.create(focused_lock_dir, recursive = TRUE, showWarnings = FALSE)
on.exit(unlink(focused_lock_dir, recursive = TRUE, force = TRUE), add = TRUE)
focused_lock <-
  .fastkpc_full_cuda_phase3_acquire_shadow_artifact_lock(focused_lock_dir)
lock_contention_error <- tryCatch({
  .fastkpc_full_cuda_phase3_acquire_shadow_artifact_lock(focused_lock_dir)
  NULL
}, error = function(error) error)
assert_true(
  inherits(lock_contention_error, "error"),
  "concurrent shadow publication lock acquisition fails closed"
)
.fastkpc_full_cuda_phase3_release_shadow_artifact_lock(focused_lock)
focused_lock <-
  .fastkpc_full_cuda_phase3_acquire_shadow_artifact_lock(focused_lock_dir)
.fastkpc_full_cuda_phase3_release_shadow_artifact_lock(focused_lock)
cross_process_lock_script <- tempfile(
  "phase3-shadow-lock-child-", fileext = ".R"
)
writeLines(c(
  'source("fastkpc/R/full_cuda_ci_gate.R")',
  'source("fastkpc/R/full_cuda_ci_oracle_contract.R")',
  'source("fastkpc/R/full_cuda_ci_workload_census.R")',
  'source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")',
  'source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")',
  'source("fastkpc/R/full_cuda_ci_fixed_sp_shadow.R")',
  'source("fastkpc/R/full_cuda_ci_phase3_artifacts.R")',
  sprintf(
    '.fastkpc_full_cuda_phase3_acquire_shadow_artifact_lock(%s)',
    encodeString(focused_lock_dir, quote = '"')
  )
), cross_process_lock_script, useBytes = TRUE)
on.exit(unlink(cross_process_lock_script, force = TRUE), add = TRUE)
focused_lock <-
  .fastkpc_full_cuda_phase3_acquire_shadow_artifact_lock(focused_lock_dir)
cross_process_lock_output <- suppressWarnings(system2(
  file.path(R.home("bin"), "Rscript"), cross_process_lock_script,
  stdout = TRUE, stderr = TRUE
))
assert_true(
  !is.null(attr(cross_process_lock_output, "status")) &&
    attr(cross_process_lock_output, "status") != 0L,
  "cross-process OS shadow artifact lock contention fails closed"
)
.fastkpc_full_cuda_phase3_release_shadow_artifact_lock(focused_lock)
focused_lock <- NULL
assert_true(
  exists(".fastkpc_full_cuda_phase3_commit_shadow_generation",
         mode = "function"),
  "shadow publication exposes an atomic rollback generation commit"
)
assert_true(
  all(vapply(c(
    ".fastkpc_full_cuda_phase3_shadow_rollback_state_path",
    ".fastkpc_full_cuda_phase3_write_shadow_rollback_state",
    ".fastkpc_full_cuda_phase3_recover_shadow_rollback"
  ), exists, logical(1L), mode = "function")),
  paste(
    "shadow rollback exposes authenticated recovery state and startup",
    "recovery before stale cleanup"
  )
)
focused_generation_dir <- tempfile("phase3-shadow-focused-generation-")
dir.create(focused_generation_dir, recursive = TRUE, showWarnings = FALSE)
on.exit(unlink(focused_generation_dir, recursive = TRUE, force = TRUE),
        add = TRUE)
focused_final <- setNames(
  file.path(focused_generation_dir, c(
    "payload.dat", "manifest.json", "summary.json"
  )), c("payload", "manifest", "summary")
)
writeLines("old-payload", focused_final[["payload"]], useBytes = TRUE)
writeLines("old-manifest", focused_final[["manifest"]], useBytes = TRUE)
writeLines("old-summary", focused_final[["summary"]], useBytes = TRUE)
focused_old_hashes <- vapply(
  focused_final, fastkpc_full_cuda_census_file_hash, character(1L)
)
focused_transaction_lock <-
  .fastkpc_full_cuda_phase3_acquire_shadow_artifact_lock(
    focused_generation_dir
  )
on.exit(.fastkpc_full_cuda_phase3_release_shadow_artifact_lock(
  focused_transaction_lock
), add = TRUE)
marker_stage <- tempfile("shadow-generation-marker-stage-")
dir.create(marker_stage, recursive = TRUE, showWarnings = FALSE)
marker_staged <- setNames(
  file.path(marker_stage, basename(focused_final)), names(focused_final)
)
writeLines("new-payload", marker_staged[["payload"]], useBytes = TRUE)
writeLines("new-manifest", marker_staged[["manifest"]], useBytes = TRUE)
writeLines("new-summary", marker_staged[["summary"]], useBytes = TRUE)
markers_hidden_before_payload <- FALSE
.fastkpc_full_cuda_phase3_commit_shadow_generation(
  staged_paths = marker_staged, final_paths = focused_final,
  payload_keys = "payload", manifest_key = "manifest",
  summary_key = "summary", output_dir = focused_generation_dir,
  artifact_lock = focused_transaction_lock,
  validate_function = function() invisible(TRUE),
  .publication_hook = function(event) {
    if (identical(event, "before_payload_commit")) {
      markers_hidden_before_payload <<-
        !file.exists(focused_final[["manifest"]]) &&
        !file.exists(focused_final[["summary"]])
    }
  }
)
assert_true(
  isTRUE(markers_hidden_before_payload),
  "both completion markers are unpublished before the first payload rename"
)
writeLines("old-payload", focused_final[["payload"]], useBytes = TRUE)
writeLines("old-manifest", focused_final[["manifest"]], useBytes = TRUE)
writeLines("old-summary", focused_final[["summary"]], useBytes = TRUE)
unlink(marker_stage, recursive = TRUE, force = TRUE)
for (failure_event in c(
  "payload_write", "before_manifest_commit",
  "after_manifest_commit", "validation_failure"
)) {
  stage <- tempfile("shadow-generation-stage-")
  dir.create(stage, recursive = TRUE, showWarnings = FALSE)
  staged <- setNames(file.path(stage, basename(focused_final)),
                     names(focused_final))
  writeLines("new-payload", staged[["payload"]], useBytes = TRUE)
  writeLines("new-manifest", staged[["manifest"]], useBytes = TRUE)
  writeLines("new-summary", staged[["summary"]], useBytes = TRUE)
  transaction_error <- tryCatch({
    .fastkpc_full_cuda_phase3_commit_shadow_generation(
      staged_paths = staged, final_paths = focused_final,
      payload_keys = "payload", manifest_key = "manifest",
      summary_key = "summary", output_dir = focused_generation_dir,
      artifact_lock = focused_transaction_lock,
      validate_function = function() {
        if (identical(failure_event, "validation_failure")) {
          stop("injected focused validation failure", call. = FALSE)
        }
        invisible(TRUE)
      },
      .publication_hook = function(event) {
        if (failure_event %in% c(
              "before_manifest_commit", "after_manifest_commit"
            ) && identical(event, failure_event)) {
          stop("injected focused publication failure", call. = FALSE)
        }
      },
      .rename_function = function(from, to) {
        if (identical(failure_event, "payload_write") &&
            identical(from, staged[["payload"]])) return(FALSE)
        file.rename(from, to)
      }
    )
    NULL
  }, error = function(error) error)
  assert_true(
    inherits(transaction_error, "error") && all(file.exists(focused_final)) &&
      identical(
      vapply(focused_final, fastkpc_full_cuda_census_file_hash,
             character(1L)), focused_old_hashes
    ),
    paste(
      "shadow generation rollback preserves prior bytes at", failure_event,
      "observed:", if (inherits(transaction_error, "error")) {
        conditionMessage(transaction_error)
      } else {
        "no error"
      }, "present=", paste(file.exists(focused_final), collapse = ",")
    )
  )
  unlink(stage, recursive = TRUE, force = TRUE)
}
focused_rollback_siblings <- function() {
  entries <- list.files(
    dirname(focused_generation_dir), full.names = TRUE,
    all.files = TRUE, no.. = TRUE
  )
  prefix <- .fastkpc_full_cuda_phase3_shadow_work_dir_prefix(
    focused_generation_dir, "rollback"
  )
  entries[startsWith(basename(entries), prefix)]
}
make_focused_stage <- function(label) {
  stage <- tempfile(paste0("shadow-restore-", label, "-stage-"))
  dir.create(stage, recursive = TRUE, showWarnings = FALSE)
  paths <- setNames(
    file.path(stage, basename(focused_final)), names(focused_final)
  )
  for (key in names(paths)) {
    writeLines(paste0(label, "-", key), paths[[key]], useBytes = TRUE)
  }
  list(dir = stage, paths = paths)
}
for (failed_restore_ordinal in seq_along(focused_final)) {
  stage <- make_focused_stage(
    paste0("restore-failure-", failed_restore_ordinal)
  )
  restore_ordinal <- 0L
  restore_error <- tryCatch({
    .fastkpc_full_cuda_phase3_commit_shadow_generation(
      staged_paths = stage$paths, final_paths = focused_final,
      payload_keys = "payload", manifest_key = "manifest",
      summary_key = "summary", output_dir = focused_generation_dir,
      artifact_lock = focused_transaction_lock,
      validate_function = function() stop(
        "injected validation failure before focused restore", call. = FALSE
      ),
      .rename_function = function(from, to) {
        if (startsWith(
              basename(from), ".fastkpc-shadow-restore-"
            )) {
          restore_ordinal <<- restore_ordinal + 1L
          if (restore_ordinal == failed_restore_ordinal) return(FALSE)
        }
        file.rename(from, to)
      }
    )
    NULL
  }, error = function(error) error)
  rollback <- focused_rollback_siblings()
  assert_true(
    inherits(restore_error, "error") && length(rollback) == 1L,
    paste(
      "failed restore rename preserves one rollback generation at ordinal",
      failed_restore_ordinal
    )
  )
  rollback_state <- .fastkpc_full_cuda_phase3_read_json(
    .fastkpc_full_cuda_phase3_shadow_rollback_state_path(rollback[[1L]]),
    "focused rollback recovery state"
  )
  backup_paths <- file.path(rollback[[1L]], basename(focused_final))
  assert_true(
    identical(
      rollback_state$schema_version,
      "full-cuda-ci-shadow-rollback-state-v1"
    ) && identical(rollback_state$state, "recovery_required") &&
      identical(
        rollback_state$output_dir,
        normalizePath(focused_generation_dir, mustWork = TRUE)
      ) && .fastkpc_full_cuda_phase3_bare_integer(
        rollback_state$owner_pid, 1L
      ) && .fastkpc_full_cuda_phase3_sha256(
        rollback_state$failed_restore_error_sha256
      ) &&
      identical(
        as.integer(rollback_state$failed_restore_ordinal),
        as.integer(failed_restore_ordinal)
      ) &&
      file.exists(.fastkpc_full_cuda_phase3_shadow_work_dir_marker_path(
        rollback[[1L]]
      )) && all(file.exists(backup_paths)) && identical(
        unname(vapply(
          backup_paths, fastkpc_full_cuda_census_file_hash, character(1L)
        )),
        unname(focused_old_hashes)
      ),
    paste(
      "recovery_required marker and exact prior backup hashes survive restore",
      "failure at ordinal", failed_restore_ordinal
    )
  )
  recovered <- tryCatch(
    .fastkpc_full_cuda_phase3_recover_shadow_rollback(
      focused_generation_dir, focused_final, focused_transaction_lock,
      validate_function = function() {
        assert_true(
          all(file.exists(focused_final)) && identical(
            vapply(
              focused_final, fastkpc_full_cuda_census_file_hash,
              character(1L)
            ), focused_old_hashes
          ),
          paste(
            "startup rollback validation observes exact restored prior bytes;",
            "present=", paste(file.exists(focused_final), collapse = ",")
          )
        )
        invisible(TRUE)
      }
    ),
    error = function(error) error
  )
  assert_true(
    !inherits(recovered, "error") && isTRUE(recovered$recovered) &&
      length(focused_rollback_siblings()) == 0L &&
      all(file.exists(focused_final)) && identical(
        vapply(
          focused_final, fastkpc_full_cuda_census_file_hash, character(1L)
        ), focused_old_hashes
      ),
    paste(
      "next locked startup restores and cleans failed rollback ordinal",
      failed_restore_ordinal, "observed:",
      if (inherits(recovered, "error")) conditionMessage(recovered) else "ok",
      "present=", paste(file.exists(focused_final), collapse = ",")
    )
  )
  unlink(stage$dir, recursive = TRUE, force = TRUE)
}

marker_failure_stage <- make_focused_stage("marker-update-failure")
marker_restore_ordinal <- 0L
marker_update_error <- tryCatch({
  .fastkpc_full_cuda_phase3_commit_shadow_generation(
    staged_paths = marker_failure_stage$paths, final_paths = focused_final,
    payload_keys = "payload", manifest_key = "manifest",
    summary_key = "summary", output_dir = focused_generation_dir,
    artifact_lock = focused_transaction_lock,
    validate_function = function() stop(
      "injected validation failure before marker update", call. = FALSE
    ),
    .publication_hook = function(event) {
      if (identical(event, "before_recovery_marker_commit")) {
        stop("injected recovery marker update failure", call. = FALSE)
      }
    },
    .rename_function = function(from, to) {
      if (startsWith(basename(from), ".fastkpc-shadow-restore-")) {
        marker_restore_ordinal <<- marker_restore_ordinal + 1L
        if (marker_restore_ordinal == 1L) return(FALSE)
      }
      file.rename(from, to)
    }
  )
  NULL
}, error = function(error) error)
marker_rollback <- focused_rollback_siblings()
marker_state <- .fastkpc_full_cuda_phase3_read_json(
  .fastkpc_full_cuda_phase3_shadow_rollback_state_path(
    marker_rollback[[1L]]
  ),
  "focused rollback ready state"
)
marker_backup_paths <- file.path(
  marker_rollback[[1L]], basename(focused_final)
)
assert_true(
  inherits(marker_update_error, "error") &&
    length(marker_rollback) == 1L &&
    identical(marker_state$state, "backup_ready") &&
    all(file.exists(marker_backup_paths)) && identical(
      unname(vapply(
        marker_backup_paths, fastkpc_full_cuda_census_file_hash, character(1L)
      )),
      unname(focused_old_hashes)
    ),
  paste(
    "failed recovery marker update preserves authenticated backup_ready state",
    "and every prior byte hash"
  )
)
unlink(marker_failure_stage$dir, recursive = TRUE, force = TRUE)
next_stage <- make_focused_stage("next-publication")
next_hashes <- vapply(
  next_stage$paths, fastkpc_full_cuda_census_file_hash, character(1L)
)
prior_restored_before_new_commit <- FALSE
.fastkpc_full_cuda_phase3_commit_shadow_generation(
  staged_paths = next_stage$paths, final_paths = focused_final,
  payload_keys = "payload", manifest_key = "manifest",
  summary_key = "summary", output_dir = focused_generation_dir,
  artifact_lock = focused_transaction_lock,
  validate_function = function() invisible(TRUE),
  .publication_hook = function(event) {
    if (identical(event, "after_rollback_recovery")) {
      prior_restored_before_new_commit <<- identical(
        vapply(
          focused_final, fastkpc_full_cuda_census_file_hash, character(1L)
        ), focused_old_hashes
      )
    }
  }
)
assert_true(
  isTRUE(prior_restored_before_new_commit) &&
    length(focused_rollback_siblings()) == 0L && identical(
      vapply(
        focused_final, fastkpc_full_cuda_census_file_hash, character(1L)
      ), next_hashes
    ),
  paste(
    "next publication restores prior generation before replacement, commits",
    "the new generation, and leaves no rollback sibling"
  )
)
unlink(next_stage$dir, recursive = TRUE, force = TRUE)
writeLines("old-payload", focused_final[["payload"]], useBytes = TRUE)
writeLines("old-manifest", focused_final[["manifest"]], useBytes = TRUE)
writeLines("old-summary", focused_final[["summary"]], useBytes = TRUE)

create_focused_recovery_backup <- function(label) {
  stage <- make_focused_stage(label)
  restore_ordinal <- 0L
  error <- tryCatch({
    .fastkpc_full_cuda_phase3_commit_shadow_generation(
      staged_paths = stage$paths, final_paths = focused_final,
      payload_keys = "payload", manifest_key = "manifest",
      summary_key = "summary", output_dir = focused_generation_dir,
      artifact_lock = focused_transaction_lock,
      validate_function = function() stop(
        "injected validation failure for backup mutation", call. = FALSE
      ),
      .rename_function = function(from, to) {
        if (startsWith(basename(from), ".fastkpc-shadow-restore-")) {
          restore_ordinal <<- restore_ordinal + 1L
          if (restore_ordinal == 1L) return(FALSE)
        }
        file.rename(from, to)
      }
    )
    NULL
  }, error = function(error) error)
  unlink(stage$dir, recursive = TRUE, force = TRUE)
  rollback <- focused_rollback_siblings()
  assert_true(
    inherits(error, "error") && length(rollback) == 1L,
    paste("focused backup mutation fixture is recoverable:", label)
  )
  rollback[[1L]]
}
for (backup_mutation in c("missing", "mismatched", "symlinked")) {
  rollback <- create_focused_recovery_backup(
    paste0("backup-", backup_mutation)
  )
  backup_payload <- file.path(rollback, basename(focused_final[["payload"]]))
  backup_bytes <- readBin(
    backup_payload, "raw", n = file.info(backup_payload)$size
  )
  symlink_target <- NULL
  if (identical(backup_mutation, "missing")) {
    unlink(backup_payload, force = TRUE)
  } else if (identical(backup_mutation, "mismatched")) {
    writeLines("forged rollback backup", backup_payload, useBytes = TRUE)
  } else {
    symlink_target <- tempfile("shadow-rollback-symlink-target-")
    writeLines("symlink target", symlink_target, useBytes = TRUE)
    unlink(backup_payload, force = TRUE)
    assert_true(
      file.symlink(symlink_target, backup_payload),
      "focused rollback backup symlink fixture is created"
    )
  }
  recovery_error <- tryCatch({
    .fastkpc_full_cuda_phase3_recover_shadow_rollback(
      focused_generation_dir, focused_final, focused_transaction_lock
    )
    NULL
  }, error = function(error) error)
  assert_true(
    inherits(recovery_error, "error") && dir.exists(rollback) &&
      file.exists(.fastkpc_full_cuda_phase3_shadow_work_dir_marker_path(
        rollback
      )) && file.exists(
        .fastkpc_full_cuda_phase3_shadow_rollback_state_path(rollback)
      ) && switch(
        backup_mutation,
        missing = !file.exists(backup_payload),
        mismatched = !identical(
          fastkpc_full_cuda_census_file_hash(backup_payload),
          focused_old_hashes[["payload"]]
        ),
        symlinked = nzchar(Sys.readlink(backup_payload))
      ),
    paste(
      backup_mutation,
      "rollback backup fails closed and preserves ownership evidence"
    )
  )
  unlink(backup_payload, force = TRUE)
  backup_connection <- file(backup_payload, open = "wb")
  writeBin(backup_bytes, backup_connection)
  close(backup_connection)
  if (!is.null(symlink_target)) unlink(symlink_target, force = TRUE)
  repaired <- .fastkpc_full_cuda_phase3_recover_shadow_rollback(
    focused_generation_dir, focused_final, focused_transaction_lock
  )
  assert_true(
    isTRUE(repaired$recovered) &&
      length(focused_rollback_siblings()) == 0L &&
      all(file.exists(focused_final)) && identical(
        vapply(
          focused_final, fastkpc_full_cuda_census_file_hash, character(1L)
        ), focused_old_hashes
      ),
    paste("repaired", backup_mutation, "backup recovers exact prior bytes")
  )
}
unlink(focused_final[["summary"]], force = TRUE)
partial_stage <- tempfile("shadow-generation-partial-stage-")
dir.create(partial_stage, recursive = TRUE, showWarnings = FALSE)
partial_staged <- setNames(
  file.path(partial_stage, basename(focused_final)), names(focused_final)
)
writeLines("new-payload", partial_staged[["payload"]], useBytes = TRUE)
writeLines("new-manifest", partial_staged[["manifest"]], useBytes = TRUE)
writeLines("new-summary", partial_staged[["summary"]], useBytes = TRUE)
partial_error <- tryCatch({
  .fastkpc_full_cuda_phase3_commit_shadow_generation(
    staged_paths = partial_staged, final_paths = focused_final,
    payload_keys = "payload", manifest_key = "manifest",
    summary_key = "summary", output_dir = focused_generation_dir,
    artifact_lock = focused_transaction_lock,
    validate_function = function() stop(
      "injected partial-generation validation failure", call. = FALSE
    )
  )
  NULL
}, error = function(error) error)
assert_true(
  inherits(partial_error, "error") &&
    !file.exists(focused_final[["manifest"]]) &&
    !file.exists(focused_final[["summary"]]) && identical(
      fastkpc_full_cuda_census_file_hash(focused_final[["payload"]]),
      focused_old_hashes[["payload"]]
    ),
  "failed replacement of a partial generation removes both completion markers"
)
unlink(partial_stage, recursive = TRUE, force = TRUE)
.fastkpc_full_cuda_phase3_release_shadow_artifact_lock(
  focused_transaction_lock
)
focused_transaction_lock <- NULL

crash_stage <- tempfile("shadow-generation-crash-stage-")
dir.create(crash_stage, recursive = TRUE, showWarnings = FALSE)
crash_staged <- setNames(
  file.path(crash_stage, basename(focused_final)), names(focused_final)
)
writeLines("crash-payload", crash_staged[["payload"]], useBytes = TRUE)
writeLines("crash-manifest", crash_staged[["manifest"]], useBytes = TRUE)
writeLines("crash-summary", crash_staged[["summary"]], useBytes = TRUE)
writeLines("old-payload", focused_final[["payload"]], useBytes = TRUE)
writeLines("old-manifest", focused_final[["manifest"]], useBytes = TRUE)
writeLines("old-summary", focused_final[["summary"]], useBytes = TRUE)
crash_script <- tempfile("phase3-shadow-crash-child-", fileext = ".R")
writeLines(c(
  'source("fastkpc/R/full_cuda_ci_gate.R")',
  'source("fastkpc/R/full_cuda_ci_oracle_contract.R")',
  'source("fastkpc/R/full_cuda_ci_workload_census.R")',
  'source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")',
  'source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")',
  'source("fastkpc/R/full_cuda_ci_fixed_sp_shadow.R")',
  'source("fastkpc/R/full_cuda_ci_phase3_artifacts.R")',
  sprintf(
    'lock <- .fastkpc_full_cuda_phase3_acquire_shadow_artifact_lock(%s)',
    encodeString(focused_generation_dir, quote = '"')
  ),
  sprintf(paste0(
    '.fastkpc_full_cuda_phase3_commit_shadow_generation(',
    'staged_paths=setNames(c(%s,%s,%s),c("payload","manifest","summary")),',
    'final_paths=setNames(c(%s,%s,%s),c("payload","manifest","summary")),',
    'payload_keys="payload",manifest_key="manifest",summary_key="summary",',
    'output_dir=%s,artifact_lock=lock,validate_function=function() TRUE,',
    '.publication_hook=function(event) { if (identical(event, ',
    '"after_payload_commit")) tools::pskill(Sys.getpid(), 9L) })'
  ),
  encodeString(crash_staged[["payload"]], quote = '"'),
  encodeString(crash_staged[["manifest"]], quote = '"'),
  encodeString(crash_staged[["summary"]], quote = '"'),
  encodeString(focused_final[["payload"]], quote = '"'),
  encodeString(focused_final[["manifest"]], quote = '"'),
  encodeString(focused_final[["summary"]], quote = '"'),
  encodeString(focused_generation_dir, quote = '"'))
), crash_script, useBytes = TRUE)
on.exit(unlink(c(crash_stage, crash_script), recursive = TRUE, force = TRUE),
        add = TRUE)
crash_output <- suppressWarnings(system2(
  file.path(R.home("bin"), "Rscript"), crash_script,
  stdout = TRUE, stderr = TRUE
))
assert_true(
  !is.null(attr(crash_output, "status")) &&
    attr(crash_output, "status") != 0L &&
    !file.exists(focused_final[["manifest"]]) &&
    !file.exists(focused_final[["summary"]]),
  "hard exit after first payload replacement leaves no completion markers"
)
recovery_stage <- tempfile("shadow-generation-recovery-stage-")
dir.create(recovery_stage, recursive = TRUE, showWarnings = FALSE)
recovery_staged <- setNames(
  file.path(recovery_stage, basename(focused_final)), names(focused_final)
)
writeLines("recovered-payload", recovery_staged[["payload"]], useBytes = TRUE)
writeLines("recovered-manifest", recovery_staged[["manifest"]], useBytes = TRUE)
writeLines("recovered-summary", recovery_staged[["summary"]], useBytes = TRUE)
recovery_lock <-
  .fastkpc_full_cuda_phase3_acquire_shadow_artifact_lock(
    focused_generation_dir
  )
.fastkpc_full_cuda_phase3_commit_shadow_generation(
  staged_paths = recovery_staged, final_paths = focused_final,
  payload_keys = "payload", manifest_key = "manifest",
  summary_key = "summary", output_dir = focused_generation_dir,
  artifact_lock = recovery_lock,
  validate_function = function() invisible(TRUE)
)
.fastkpc_full_cuda_phase3_release_shadow_artifact_lock(recovery_lock)
assert_true(
  identical(readLines(focused_final[["payload"]]), "recovered-payload") &&
    identical(readLines(focused_final[["manifest"]]), "recovered-manifest") &&
    identical(readLines(focused_final[["summary"]]), "recovered-summary") &&
    !any(startsWith(
      list.files(
        dirname(focused_generation_dir), all.files = TRUE,
        no.. = TRUE, full.names = FALSE
      ),
      paste0(".", basename(focused_generation_dir),
             "-phase3-shadow-rollback-")
    )),
  "publication after a hard exit recomputes and completes a clean generation"
)
unlink(recovery_stage, recursive = TRUE, force = TRUE)
assert_true(
  exists(".fastkpc_full_cuda_phase3_shadow_command_lines",
         mode = "function"),
  "shadow publication validates command evidence before mutation"
)
assert_true(
  identical(
    .fastkpc_full_cuda_phase3_shadow_command_lines("Rscript focused.R"),
    "Rscript focused.R"
  ),
  "nonempty exact command evidence is preserved"
)
for (bad_commands in list(
  character(), "", NA_character_, c("ok", "bad\ncommand"), factor("bad")
)) {
  command_error <- tryCatch({
    .fastkpc_full_cuda_phase3_shadow_command_lines(bad_commands)
    NULL
  }, error = function(error) error)
  assert_true(
    inherits(command_error, "error"),
    "empty, malformed, attributed, or multiline commands fail closed"
  )
}
assert_true(
  exists(".fastkpc_full_cuda_phase3_validate_shadow_constraints",
         mode = "function"),
  "v2 public validator exposes exact caller-constraint matching"
)
focused_expected_constraints <- list(
  identity = list(schema_version = "test", sha256 = strrep("3", 64L)),
  catalog_authority_sha256 = strrep("4", 64L), device_id = 0L,
  scope = "iteration", phase0_dir = "/phase0", oracle_sp_dir = "/oracle",
  shard_count = 64L, canonical_setup_shards = FALSE,
  shadow_plan_identity_sha256 = strrep("5", 64L),
  direct_logical_sequence_id = 1L
)
focused_matching_constraints <- focused_expected_constraints
invisible(.fastkpc_full_cuda_phase3_validate_shadow_constraints(
  focused_matching_constraints, focused_expected_constraints
))
for (field in names(focused_matching_constraints)) {
  mismatched <- focused_matching_constraints
  mismatched[[field]] <- switch(
    field,
    identity = list(schema_version = "wrong", sha256 = strrep("6", 64L)),
    catalog_authority_sha256 = strrep("6", 64L),
    device_id = 1L, scope = "qualification", phase0_dir = "/wrong",
    oracle_sp_dir = "/wrong", shard_count = 63L,
    canonical_setup_shards = TRUE,
    shadow_plan_identity_sha256 = strrep("6", 64L),
    direct_logical_sequence_id = 2L
  )
  constraint_error <- tryCatch({
    .fastkpc_full_cuda_phase3_validate_shadow_constraints(
      mismatched, focused_expected_constraints
    )
    NULL
  }, error = function(error) error)
  assert_true(
    inherits(constraint_error, "error"),
    paste("mismatched public constraint fails closed:", field)
  )
}
if (focused_only) {
  cat("full CUDA CI fixed-sp shadow artifact focused probes: PASS\n")
  quit(save = "no", status = 0L)
}

phase1_rows <- readRDS(file.path(
  "fastkpc", "artifacts", "full_cuda_ci",
  "workload_census_351x48_v1", "logical_ci_tests.rds"
))
direct_source <- phase1_rows[phase1_rows$logical_sequence_id == 1L,
                             , drop = FALSE]
conditional_source <- phase1_rows[
  phase1_rows$logical_sequence_id == 2214L, , drop = FALSE
]
direct <- data.frame(
  logical_sequence_id = as.integer(direct_source$logical_sequence_id),
  source_sequence_id = as.integer(direct_source$source_sequence_id),
  source_task_index = as.integer(direct_source$source_task_index),
  level = as.integer(direct_source$level),
  x = as.integer(direct_source$x), y = as.integer(direct_source$y),
  S_key = as.character(direct_source$S_key),
  residual_key_x = NA_character_, residual_key_y = NA_character_,
  reference_p_value = as.double(direct_source$reference_p_value),
  candidate_p_value = as.double(direct_source$reference_p_value),
  absolute_p_value_difference = 0,
  alpha = as.double(direct_source$alpha),
  reference_decision = as.character(direct_source$reference_decision),
  candidate_decision = as.character(direct_source$reference_decision),
  decision_flip = FALSE, backend = "legacy-cpp",
  low_rank_backend = "spectra", backend_error = FALSE,
  spectra_fallback = FALSE, stringsAsFactors = FALSE
)
conditional <- as.data.frame(lapply(
  fastkpc_full_cuda_shadow_conditional_row_schema(),
  function(type) switch(type, integer = 0L, double = 0,
                        logical = FALSE, character = "")
), stringsAsFactors = FALSE, optional = TRUE)
for (field in intersect(names(conditional_source), names(conditional))) {
  conditional[[field]] <- conditional_source[[field]]
}
conditional$logical_sequence_id <- 2214L
conditional$candidate_p_value <- conditional$reference_p_value
conditional$absolute_p_value_difference <- 0
conditional$candidate_decision <- conditional$reference_decision
conditional$decision_flip <- FALSE
conditional$near_alpha <- is.finite(
  conditional_source$absolute_log_distance_from_alpha
) && conditional_source$absolute_log_distance_from_alpha <= log(2)
conditional$near_alpha_bucket <- fastkpc_full_cuda_census_near_alpha_bucket(
  abs(log(pmax(conditional$reference_p_value, .Machine$double.xmin) /
            conditional$alpha))
)
conditional$prepared_s_key_sha256 <- strrep("a", 64L)
conditional$shard_id <- 0L
conditional$backend <- "cpp"
conditional$backend_version <- fastkpc_full_cuda_shadow_dcov_backend_version()
conditional$low_rank_backend <- "spectra"
for (endpoint in c("x", "y")) {
  conditional[[paste0("planned_route_", endpoint)]] <- "AUGMENTED_SVD"
  conditional[[paste0("executed_route_", endpoint)]] <- "AUGMENTED_SVD"
  conditional[[paste0("reroute_reason_", endpoint)]] <- ""
  conditional[[paste0("solver_status_", endpoint)]] <- "OK_AUGMENTED_SVD"
}
merged_rows <- fastkpc_full_cuda_shadow_merge_logical_rows(
  direct_rows = direct, conditional_rows = conditional,
  expected_logical_sequence_id = c(1L, 2214L)
)
assert_true(
  identical(merged_rows$logical_sequence_id, c(1L, 2214L)) &&
    identical(merged_rows$scope_sequence_ordinal, 1:2) &&
    identical(merged_rows$source_type, c("direct", "conditional")) &&
    !any(merged_rows$decision_flip),
  "merged logical rows preserve selected numeric order and derive decisions"
)

assert_true(
  exists("fastkpc_full_cuda_shadow_reconstruct_target_routes",
         mode = "function"),
  "fixed-sp shadow target route reconstructor should exist"
)
expected_target_keys <- sort(c(
  conditional$residual_key_x, conditional$residual_key_y
), method = "radix")
route_evidence <- fastkpc_full_cuda_shadow_reconstruct_target_routes(
  merged_rows, expected_target_keys = expected_target_keys,
  expected_target_corpus_hash =
    fastkpc_full_cuda_census_key_set_hash(expected_target_keys)
)
assert_true(
  nrow(route_evidence$target_routes) == 2L &&
    route_evidence$summary$planned_svd_target_count == 2L &&
    route_evidence$summary$executed_svd_target_count == 2L &&
    route_evidence$summary$stable_reroute_count == 0L,
  "target route evidence is reconstructed and conserved"
)
conflicting_rows <- rbind(merged_rows, merged_rows[2L, , drop = FALSE])
conflicting_rows$logical_sequence_id[[3L]] <- 2215L
conflicting_rows$planned_route_x[[3L]] <- "AUGMENTED_QR"
conflict <- tryCatch({
  fastkpc_full_cuda_shadow_reconstruct_target_routes(conflicting_rows)
  NULL
}, error = function(error) error)
assert_true(
  inherits(conflict, "error") && grepl(
    "conflicting repeated target route evidence",
    conditionMessage(conflict), fixed = TRUE
  ),
  "conflicting repeated target route observations fail closed"
)

phase0_source_dir <- file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "oracle_351x48_v1"
)
phase0_dir <- tempfile("phase3-shadow-phase0-authority-")
dir.create(phase0_dir, recursive = TRUE, showWarnings = FALSE)
phase0_source_files <- list.files(
  phase0_source_dir, full.names = TRUE, all.files = TRUE, no.. = TRUE
)
assert_true(
  all(file.copy(
    phase0_source_files, phase0_dir, recursive = TRUE,
    copy.mode = TRUE, copy.date = TRUE
  )),
  "fixture copies Phase 0 authority without mutating repository artifacts"
)
on.exit(unlink(phase0_dir, recursive = TRUE, force = TRUE), add = TRUE)
phase1_dir <- file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "workload_census_351x48_v1"
)
phase2_dir <- file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1"
)
data_path <- file.path(
  "fastkpc", "artifacts", "kpc_tprs_real_zhu",
  "cancer_RD-causalDiscoveryInput.rds"
)
catalog <- fastkpc_full_cuda_open_fixed_sp_catalog(
  phase0_dir, phase1_dir, phase2_dir, data_path
)
plan <- fastkpc_full_cuda_shadow_plan(catalog)
execution_snapshot <-
  fastkpc_full_cuda_phase3_create_shadow_execution_snapshot(
    catalog = catalog, plan = plan, scope = "iteration"
  )
on.exit(
  fastkpc_full_cuda_phase3_release_shadow_execution_snapshot(
    execution_snapshot
  ), add = TRUE
)
selected <- .fastkpc_full_cuda_phase3_resolve_shadow_execution_snapshot(
  execution_snapshot, expected_scope = "iteration"
)
phase0_capture <- .fastkpc_full_cuda_phase3_shadow_phase0_authority(
  catalog, phase0_dir
)
phase0_expected <- .fastkpc_full_cuda_phase3_load_shadow_phase0_oracle(
  catalog, phase0_capture
)
mutated_phase0_catalog <- catalog
mutated_phase0_catalog$phase0$reference$adjacency[1L, 2L] <-
  !mutated_phase0_catalog$phase0$reference$adjacency[1L, 2L]
mutated_phase0_catalog$phase0$logical_trace$p_value[[1L]] <-
  mutated_phase0_catalog$phase0$logical_trace$p_value[[1L]] + 0.25
mutated_phase0_catalog$phase0$deletion_trace$p_value[[1L]] <-
  mutated_phase0_catalog$phase0$deletion_trace$p_value[[1L]] + 0.25
mutated_phase0_catalog$phase0$reference$sepsets <- list(forged = TRUE)
mutated_phase0_catalog$phase0$reference$n.edgetests[[1L]] <-
  mutated_phase0_catalog$phase0$reference$n.edgetests[[1L]] + 1L
phase0_reloaded <- .fastkpc_full_cuda_phase3_load_shadow_phase0_oracle(
  mutated_phase0_catalog,
  .fastkpc_full_cuda_phase3_shadow_phase0_authority(
    mutated_phase0_catalog, phase0_dir
  )
)
assert_true(
  identical(
    phase0_reloaded$reference$adjacency,
    phase0_expected$reference$adjacency
  ) && identical(
    phase0_reloaded$logical_trace, phase0_expected$logical_trace
  ) && identical(
    phase0_reloaded$deletion_trace, phase0_expected$deletion_trace
  ) && identical(
    phase0_reloaded$reference$sepsets,
    phase0_expected$reference$sepsets
  ) && identical(
    phase0_reloaded$reference$n.edgetests,
    phase0_expected$reference$n.edgetests
  ),
  paste(
    "fresh canonical Phase 0 reload ignores caller catalog adjacency,",
    "logical, deletion, sepset, and n.edgetests mutations"
  )
)
rm(mutated_phase0_catalog, phase0_reloaded, phase0_expected)
logical_rows <- selected$logical_rows
target_rows <- selected$target_rows
setup_keys <- selected$setup_keys
setup_authority <- selected$setup_authority
fixture_assignments <- fastkpc_full_cuda_phase3_assign_setup_shards(
  setup_keys, 64L
)
setup_shard_match <- match(
  setup_keys, fixture_assignments$prepared_s_key_sha256
)
setup_authority$shard_id <-
  fixture_assignments$shard_id[setup_shard_match]
target_rows$shard_id <- fixture_assignments$shard_id[match(
  target_rows$prepared_s_key_sha256,
  fixture_assignments$prepared_s_key_sha256
)]
logical_rows$shard_id <- fixture_assignments$shard_id[match(
  logical_rows$prepared_s_key_x,
  fixture_assignments$prepared_s_key_sha256
)]
endpoint_keys <- sort(unique(c(
  logical_rows$residual_key_x, logical_rows$residual_key_y
)), method = "radix")
assert_true(
  nrow(logical_rows) == 44L && nrow(target_rows) == 270L &&
    length(setup_keys) == 44L &&
    identical(setup_authority$prepared_s_key_sha256, setup_keys),
  "fixture uses authenticated iteration logical/setup/target authority"
)

route_status <- function(route, target_index) {
  if (identical(route, "CHOLESKY_BATCHED")) {
    setup_key <- target_rows$prepared_s_key_sha256[[target_index]]
    batched <- sum(
      target_rows$prepared_s_key_sha256 == setup_key &
        target_rows$planned_route == "CHOLESKY_BATCHED"
    ) >= 2L
    return(if (batched) "OK_CHOLESKY_BATCHED" else "OK_CHOLESKY_SINGLE")
  }
  switch(route, AUGMENTED_QR = "OK_AUGMENTED_QR",
         AUGMENTED_SVD = "OK_AUGMENTED_SVD")
}
conditional_fixture <- as.data.frame(lapply(
  fastkpc_full_cuda_shadow_conditional_row_schema(),
  function(type) switch(type,
    integer = rep.int(0L, nrow(logical_rows)),
    double = rep.int(0, nrow(logical_rows)),
    logical = rep.int(FALSE, nrow(logical_rows)),
    character = rep.int("", nrow(logical_rows))
  )
), stringsAsFactors = FALSE)
for (field in intersect(names(logical_rows), names(conditional_fixture))) {
  conditional_fixture[[field]] <- logical_rows[[field]]
}
conditional_fixture$prepared_s_key_sha256 <-
  logical_rows$prepared_s_key_x
conditional_fixture$shard_id <- logical_rows$shard_id
conditional_fixture$candidate_p_value <- logical_rows$reference_p_value
conditional_fixture$absolute_p_value_difference <- rep.int(
  0, nrow(logical_rows)
)
conditional_fixture$candidate_decision <- logical_rows$reference_decision
conditional_fixture$decision_flip <- rep.int(FALSE, nrow(logical_rows))
distance <- abs(log(pmax(
  logical_rows$reference_p_value, .Machine$double.xmin
) / logical_rows$alpha))
conditional_fixture$near_alpha <- distance <= log(2)
conditional_fixture$near_alpha_bucket <- vapply(
  distance, fastkpc_full_cuda_census_near_alpha_bucket, character(1L)
)
conditional_fixture$backend <- rep.int("cpp", nrow(logical_rows))
conditional_fixture$backend_version <-
  rep.int(fastkpc_full_cuda_shadow_dcov_backend_version(),
          nrow(logical_rows))
conditional_fixture$low_rank_backend <- rep.int(
  "spectra", nrow(logical_rows)
)
for (endpoint in c("x", "y")) {
  key <- logical_rows[[paste0("residual_key_", endpoint)]]
  route_index <- match(key, target_rows$residual_key_sha256)
  route <- target_rows$planned_route[route_index]
  conditional_fixture[[paste0("planned_route_", endpoint)]] <- route
  conditional_fixture[[paste0("executed_route_", endpoint)]] <- route
  conditional_fixture[[paste0("reroute_reason_", endpoint)]] <-
    rep.int("", nrow(logical_rows))
  conditional_fixture[[paste0("solver_status_", endpoint)]] <-
    vapply(seq_along(route), function(index) {
      route_status(route[[index]], route_index[[index]])
    }, character(1L))
}
invisible(fastkpc_full_cuda_shadow_validate_conditional_rows(
  conditional_fixture, expected_logical_tests = logical_rows
))

oracle_frame <- function(name, row_count) {
  schema <- fastkpc_full_cuda_fixed_sp_oracle_row_schemas()[[name]]
  columns <- lapply(unname(schema), function(type) switch(
    type,
    character = rep.int("", row_count),
    integer = rep.int(0L, row_count),
    double = rep.int(0, row_count),
    logical = rep.int(FALSE, row_count)
  ))
  names(columns) <- names(schema)
  as.data.frame(columns, stringsAsFactors = FALSE, optional = TRUE)
}
target_setup <- match(
  target_rows$prepared_s_key_sha256, setup_keys
)
target_count <- as.integer(tabulate(
  target_setup, nbins = length(setup_keys)
))
setup_shard <- setup_authority$shard_id
setup_ordinal <- as.integer(ave(
  seq_along(setup_keys), setup_shard, FUN = seq_along
))
route_count <- function(route) as.integer(tabulate(
  target_setup[target_rows$planned_route == route],
  nbins = length(setup_keys)
))
planned_cholesky <- route_count("CHOLESKY_BATCHED")
planned_qr <- route_count("AUGMENTED_QR")
planned_svd <- route_count("AUGMENTED_SVD")
resource_fixture <- oracle_frame("resource_metrics", length(setup_keys))
resource_fixture$prepared_s_key_sha256 <- setup_keys
resource_fixture$shard_id <- setup_shard
resource_fixture$setup_ordinal <- setup_ordinal
resource_fixture$canonical_setup_rank <-
  setup_authority$canonical_setup_rank
resource_fixture$target_count <- target_count
resource_fixture$phase2_shard_load_count <- as.integer(!duplicated(setup_shard))
resource_fixture$phase2_shard_authentication_count <-
  resource_fixture$phase2_shard_load_count
resource_fixture$planned_cholesky_target_count <- planned_cholesky
resource_fixture$planned_qr_target_count <- planned_qr
resource_fixture$planned_svd_target_count <- planned_svd
resource_fixture$executed_cholesky_target_count <- planned_cholesky
resource_fixture$executed_qr_target_count <- planned_qr
resource_fixture$executed_svd_target_count <- planned_svd
resource_fixture$true_batched_subgroup_count <-
  as.integer(planned_cholesky >= 2L)
resource_fixture$true_batched_attempted_target_count <- as.integer(ifelse(
  planned_cholesky >= 2L, planned_cholesky, 0L
))
resource_fixture$true_batched_target_count <- as.integer(ifelse(
  planned_cholesky >= 2L, planned_cholesky, 0L
))
resource_fixture$aggregate_penalty_factor_count <- planned_svd
resource_fixture$aggregate_svd_b_build_count <- 2L * planned_svd
resource_fixture$cholesky_factor_checkpoint_record_count <-
  as.integer(planned_cholesky > 0L)
resource_fixture$cholesky_factor_checkpoint_wait_count <-
  resource_fixture$cholesky_factor_checkpoint_record_count
resource_fixture$cholesky_solve_checkpoint_record_count <-
  as.integer(planned_cholesky > 0L)
resource_fixture$cholesky_solve_checkpoint_wait_count <-
  resource_fixture$cholesky_solve_checkpoint_record_count
resource_fixture$qr_checkpoint_record_count <- as.integer(planned_qr > 0L)
resource_fixture$qr_checkpoint_wait_count <-
  resource_fixture$qr_checkpoint_record_count
resource_fixture$svd_checkpoint_record_count <- as.integer(planned_svd > 0L)
resource_fixture$svd_checkpoint_wait_count <-
  resource_fixture$svd_checkpoint_record_count
resource_fixture$coefficient_batch_finalize_call_count <-
  as.integer(planned_cholesky > 0L)
resource_fixture$fitted_batch_finalize_call_count <-
  as.integer(planned_cholesky > 0L)
resource_fixture$residual_rss_batch_finalize_call_count <-
  as.integer(planned_cholesky > 0L)
resource_fixture$per_target_output_finalize_call_count <-
  planned_qr + planned_svd
resource_fixture$batch_output_finalized_target_count <- target_count
for (field in c(
  "prepared_handle_create_count", "prepared_handle_destroy_count",
  "residual_token_acquire_count", "residual_token_release_count",
  "output_slot_acquire_count", "output_slot_release_count",
  "setup_h2d_upload_count", "target_batch_h2d_call_count",
  "rhs_device_build_count", "shadow_materialize_call_count",
  "invalid_output_init_count"
)) resource_fixture[[field]] <- rep.int(1L, length(setup_keys))
explicit_constraint <- setup_authority$constraint_mode == "explicit"
resource_fixture$setup_h2d_bytes <- 8 * (
  as.double(setup_authority$n) * as.double(setup_authority$coefficient_dim) +
    as.double(explicit_constraint) * (
      as.double(setup_authority$coefficient_dim) *
        as.double(setup_authority$null_dim) +
      as.double(setup_authority$n) * as.double(setup_authority$null_dim)
    ) + as.double(setup_authority$null_dim)^2 +
    as.double(setup_authority$penalty_count) *
      as.double(setup_authority$null_dim)^2 +
    as.double(setup_authority$has_H) *
      as.double(setup_authority$null_dim)^2
)
resource_fixture$target_h2d_copy_count <- rep.int(2L, length(setup_keys))
resource_fixture$target_h2d_bytes <-
  8 * as.double(target_count) * as.double(
    setup_authority$n + setup_authority$penalty_count
  )
resource_fixture$rhs_authority <- rep.int(
  "cuda-x0-transpose-y", length(setup_keys)
)
resource_fixture$full_cuda_data_plane <- rep.int(TRUE, length(setup_keys))
resource_fixture$shadow_materialize_target_count <- target_count
resource_fixture$shadow_d2h_bytes <-
  8 * as.double(target_count) * as.double(
    setup_authority$coefficient_dim + 2L * setup_authority$n + 1L +
      setup_authority$null_dim
  )
resource_fixture$cusolver_deterministic_mode <- rep.int(
  "enabled", length(setup_keys)
)
resource_fixture$cublas_math_mode <- rep.int(
  "pedantic", length(setup_keys)
)
resource_fixture$cublas_atomics_mode <- rep.int(
  "not_allowed", length(setup_keys)
)
resource_fixture$cublas_user_workspace_installed <- rep.int(
  TRUE, length(setup_keys)
)
resource_fixture$cublas_workspace_bytes <- rep.int(
  16777216, length(setup_keys)
)
resource_fixture$cublas_workspace_alignment <- rep.int(
  256, length(setup_keys)
)
stage_fixture <- oracle_frame("stage_timing", 6L * length(setup_keys))
stage_fixture$prepared_s_key_sha256 <- rep(setup_keys, each = 6L)
stage_fixture$shard_id <- rep(setup_shard, each = 6L)
stage_fixture$setup_ordinal <- rep(setup_ordinal, each = 6L)
stage_fixture$stage <- rep(c(
  "phase2_shard_load", "prepared_handle_create", "solve",
  "shadow_materialize", "cmagic_oracle", "release_and_free"
), length(setup_keys))
stage_fixture$elapsed_ms <- rep.int(0, nrow(stage_fixture))
payload_fixture <- list(
  logical_ci_parity = conditional_fixture,
  resource_metrics = resource_fixture,
  stage_timing = stage_fixture
)
invisible(fastkpc_full_cuda_phase3_validate_shadow_payload(
  payload_fixture, expected_setup_keys = setup_keys,
  expected_target_rows = target_rows,
  expected_logical_tests = logical_rows,
  require_logical_authority = TRUE,
  expected_setup_rows = setup_authority
))
assert_true(
  all(c(
    "planned_cholesky_target_count", "planned_qr_target_count",
    "planned_svd_target_count", "executed_cholesky_target_count",
    "executed_qr_target_count", "executed_svd_target_count",
    "cholesky_to_svd_count", "qr_to_svd_count", "stable_reroute_count"
  ) %in% names(fastkpc_full_cuda_shadow_runtime_counters(
    resource_fixture
  ))),
  "runtime aggregate exposes every independently checked route counter"
)

sha <- fastkpc_full_cuda_census_hash_utf8
route_config <- fastkpc_full_cuda_phase3_route_config()
identity <- list(
  schema_version = "full-cuda-ci-phase3-test-input-identity-v1",
  canonical_setup_corpus_hash =
    fastkpc_full_cuda_census_key_set_hash(setup_keys),
  canonical_target_corpus_hash =
    fastkpc_full_cuda_census_key_set_hash(
      sort(target_rows$residual_key_sha256, method = "radix")
    ),
  route_config_hash = route_config$sha256,
  source_commit = strrep("1", 40L), cuda_toolkit_version = 12040L,
  cuda_driver_version = 55054L, gpu_name = "Synthetic GPU",
  gpu_uuid = paste0("GPU-", strrep("a", 32L)),
  compute_capability_major = 8L, compute_capability_minor = 0L,
  compute_capability = "8.0", sm_count = 108L, device_id = 0L,
  cusolver_deterministic_mode_required = "enabled",
  cublas_math_mode_required = "pedantic",
  cublas_atomics_mode_required = "not_allowed",
  cublas_user_workspace_required = TRUE,
  cublas_workspace_bytes_required = 16777216,
  cublas_workspace_min_alignment_required = 256
)
identity$sha256 <- .fastkpc_full_cuda_phase3_named_hash(identity)

production_native_path <- tempfile(
  "phase3-shadow-production-native-", fileext = .Platform$dynlib.ext
)
writeBin(charToRaw("Task 9 production native identity fixture"),
         production_native_path)
on.exit(unlink(production_native_path, force = TRUE), add = TRUE)
production_native_file <- list(
  path = normalizePath(production_native_path, mustWork = TRUE),
  device_major_hex = "1", device_minor_hex = "1", inode = "1"
)
production_native_sha256 <- fastkpc_full_cuda_census_file_hash(
  production_native_path
)
production_catalog_authority <-
  .fastkpc_full_cuda_phase3_catalog_authority_snapshot(catalog)
production_policy <- .fastkpc_full_cuda_phase3_policy_contract()
production_abi <- fastkpc_full_cuda_fixed_sp_runtime_abi()
production_route <- fastkpc_full_cuda_phase3_route_config()
production_source_commit <- strrep("7", 40L)
production_identity <- list(
  schema_version = "full-cuda-ci-phase3-input-identity-v1",
  phase0_manifest_hash = production_catalog_authority$phase0_manifest_hash,
  phase1_manifest_hash = production_catalog_authority$phase1_manifest_hash,
  phase2_manifest_hash = production_catalog_authority$phase2_manifest_hash,
  dataset_file_sha256 = production_catalog_authority$dataset_file_sha256,
  dataset_matrix_sha256 = production_catalog_authority$dataset_matrix_sha256,
  canonical_setup_corpus_hash =
    production_catalog_authority$canonical_setup_corpus_hash,
  canonical_target_corpus_hash =
    production_catalog_authority$canonical_target_corpus_hash,
  phase0_source_commit = production_catalog_authority$phase0_source_commit,
  phase1_source_commit = production_catalog_authority$phase1_source_commit,
  phase2_source_commit = production_catalog_authority$phase2_source_commit,
  phase2_R_version = production_catalog_authority$phase2_R_version,
  phase2_mgcv_version = production_catalog_authority$phase2_mgcv_version,
  runtime_abi = production_abi$schema_version,
  runtime_abi_hash = production_abi$sha256,
  runtime_policy_schema_version =
    production_policy$configuration_schema_version,
  route_config_hash = production_route$sha256,
  source_commit = production_source_commit,
  phase3_source_commit = production_source_commit,
  R_version = R.version.string,
  phase3_R_version = R.version.string,
  mgcv_version = as.character(utils::packageVersion("mgcv")),
  phase3_mgcv_version = as.character(utils::packageVersion("mgcv")),
  provenance_schema_version =
    "full-cuda-ci-execution-source-snapshot-v6",
  provenance_mode = "working-tree-execution-snapshot-v1",
  source_closure_schema_version =
    "full-cuda-ci-execution-source-closure-v1",
  source_discovery_semantics =
    "parsed-r-ast-load-time-literal-source-v1",
  source_closure_count = 1L,
  source_closure_sha256 = sha("production-source-closure"),
  execution_snapshot_sha256 = sha("production-execution-snapshot"),
  relevant_sources_dirty_or_untracked = FALSE,
  execution_sources_unchanged_after_run = TRUE,
  execution_provenance_state = "post-run-verified",
  native_library_identity =
    "qualified-pinned-inode-sha-exact-registered-mapped-path-v3",
  native_library_path = production_native_file$path,
  native_library_device_major_hex = production_native_file$device_major_hex,
  native_library_device_minor_hex = production_native_file$device_minor_hex,
  native_library_inode = production_native_file$inode,
  native_library_sha256 = production_native_sha256,
  native_build_inputs_sha256 = sha("production-native-build-inputs"),
  native_build_dependencies_schema_version =
    "full-cuda-ci-native-build-dependencies-v3",
  native_build_attestation_schema_version =
    "full-cuda-ci-native-build-trace-attestation-v2",
  native_build_attestation_sha256 =
    sha("production-native-build-attestation"),
  native_build_trace_semantics =
    "linux-strace-successful-read-exec-evidence-v3",
  native_build_trace_invocation = "Task 9 fixture native trace",
  native_build_tracer_path = "/usr/bin/strace",
  native_build_dependency_count = 1L,
  native_build_exclusion_count = 0L,
  native_build_dependencies_sha256 =
    sha("production-native-build-dependencies"),
  native_build_trace_sha256 = sha("production-native-build-trace"),
  native_build_tracer_sha256 = sha("production-native-build-tracer"),
  cuda_toolkit_version = 12040L, cuda_driver_version = 55054L,
  gpu_name = "Task 9 Production Fixture GPU",
  gpu_uuid = paste0("GPU-", strrep("b", 32L)),
  compute_capability_major = 8L, compute_capability_minor = 0L,
  compute_capability = "8.0", sm_count = 108L, device_id = 0L,
  cusolver_deterministic_mode_required =
    production_policy$cusolver_deterministic_mode_required,
  cublas_math_mode_required = production_policy$cublas_math_mode_required,
  cublas_atomics_mode_required =
    production_policy$cublas_atomics_mode_required,
  cublas_user_workspace_required =
    production_policy$cublas_user_workspace_required,
  cublas_workspace_bytes_required =
    production_policy$cublas_workspace_bytes_required,
  cublas_workspace_min_alignment_required =
    production_policy$cublas_workspace_min_alignment_required,
  shard_count = production_route$shard_count, authenticated = TRUE
)
production_identity$sha256 <-
  .fastkpc_full_cuda_phase3_identity_hash(production_identity)
invisible(fastkpc_full_cuda_phase3_validate_input_identity(
  production_identity
))

output_dir <- tempfile("phase3-shadow-artifact-")
oracle_sp_dir <- tempfile("phase3-shadow-oracle-sp-")
on.exit(unlink(c(output_dir, oracle_sp_dir), recursive = TRUE, force = TRUE),
        add = TRUE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

direct_authority <- phase1_rows[phase1_rows$level == 0L, , drop = FALSE]
direct_fixture <- data.frame(
  logical_sequence_id = as.integer(direct_authority$logical_sequence_id),
  source_sequence_id = as.integer(direct_authority$source_sequence_id),
  source_task_index = as.integer(direct_authority$source_task_index),
  level = as.integer(direct_authority$level),
  x = as.integer(direct_authority$x), y = as.integer(direct_authority$y),
  S_key = as.character(direct_authority$S_key),
  residual_key_x = as.character(direct_authority$residual_key_x),
  residual_key_y = as.character(direct_authority$residual_key_y),
  reference_p_value = as.double(direct_authority$reference_p_value),
  candidate_p_value = as.double(direct_authority$reference_p_value),
  absolute_p_value_difference = rep.int(0, nrow(direct_authority)),
  alpha = as.double(direct_authority$alpha),
  reference_decision = as.character(direct_authority$reference_decision),
  candidate_decision = as.character(direct_authority$reference_decision),
  decision_flip = rep.int(FALSE, nrow(direct_authority)),
  backend = rep.int("legacy-cpp", nrow(direct_authority)),
  low_rank_backend = rep.int("spectra", nrow(direct_authority)),
  backend_error = rep.int(FALSE, nrow(direct_authority)),
  spectra_fallback = rep.int(FALSE, nrow(direct_authority)),
  stringsAsFactors = FALSE
)
direct_payload <- .fastkpc_full_cuda_phase3_direct_ci_payload(
  direct_fixture, catalog
)
artifact_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  output_dir, "full_shadow"
)
saveRDS(direct_payload, artifact_paths$direct_ci_rds,
        version = 2, compress = FALSE)
direct_summary <- .fastkpc_full_cuda_phase3_direct_ci_summary(
  direct_payload,
  fastkpc_full_cuda_census_file_hash(artifact_paths$direct_ci_rds)
)
.fastkpc_full_cuda_phase3_write_json_exact(
  direct_summary, artifact_paths$direct_ci_summary_json
)
invisible(fastkpc_full_cuda_phase3_validate_direct_ci_payload(
  output_dir, catalog
))

context_count <- 0L
run <- fastkpc_full_cuda_phase3_run_shards(
  output_dir = output_dir, kind = "full_shadow",
  setup_keys = setup_keys, target_rows = target_rows,
  identity = identity, route_config = route_config,
  executor = function(context, shard_id, setup_keys, target_rows) {
    rows <- conditional_fixture[
      conditional_fixture$shard_id == shard_id, , drop = FALSE
    ]
    resources <- resource_fixture[
      resource_fixture$shard_id == shard_id, , drop = FALSE
    ]
    stages <- stage_fixture[
      stage_fixture$shard_id == shard_id, , drop = FALSE
    ]
    rownames(rows) <- rownames(resources) <- rownames(stages) <- NULL
    count <- as.integer(length(setup_keys))
    list(
      payload = list(
        logical_ci_parity = rows,
        resource_metrics = resources,
        stage_timing = stages
      ),
      resource_counts = list(
        prepared_handle_create_count = count,
        prepared_handle_destroy_count = count,
        residual_token_acquire_count = count,
        residual_token_release_count = count,
        output_slot_acquire_count = count,
        output_slot_release_count = count
      )
    )
  },
  runtime_create = function() {
    context_count <<- context_count + 1L
    new.env(parent = emptyenv())
  },
  runtime_destroy = function(context) invisible(NULL),
  scope = "iteration", shard_count = 64L
)
assert_true(identical(run$status, "complete") && context_count == 1L,
            "fixture shard source is complete")

make_oracle_payload <- function(shard_id, shard_setup_keys, shard_targets) {
  setup_count <- length(shard_setup_keys)
  target_count_value <- nrow(shard_targets)
  if (setup_count == 0L) {
    setup_results <- oracle_frame("setup_results", 0L)
    target_parity <- oracle_frame("target_parity", 0L)
    resource_metrics <- oracle_frame("resource_metrics", 0L)
    stage_timing <- oracle_frame("stage_timing", 0L)
    fallbacks <- .fastkpc_full_cuda_phase3_oracle_fallback_rows(
      target_parity, resource_metrics
    )
    failures <- .fastkpc_full_cuda_phase3_oracle_failure_rows(target_parity)
    summary <- fastkpc_full_cuda_phase3_summarize_oracle_rows(
      setup_results, target_parity, resource_metrics, stage_timing,
      fallbacks, failures
    )
    return(list(
      setup_results = setup_results, target_parity = target_parity,
      resource_metrics = resource_metrics, stage_timing = stage_timing,
      fallbacks = fallbacks, failures = failures, summary = summary
    ))
  }
  setup_rank <- match(
    shard_targets$prepared_s_key_sha256, shard_setup_keys
  )
  target_parity <- oracle_frame("target_parity", target_count_value)
  target_parity$prepared_s_key_sha256 <-
    shard_targets$prepared_s_key_sha256
  target_parity$shard_id <- rep.int(as.integer(shard_id), target_count_value)
  target_parity$setup_ordinal <- as.integer(setup_rank)
  target_parity$canonical_setup_rank <-
    shard_targets$canonical_setup_rank
  target_parity$target_ordinal <- as.integer(ave(
    seq_len(target_count_value), shard_targets$prepared_s_key_sha256,
    FUN = seq_along
  ))
  target_parity$canonical_target_rank <-
    shard_targets$canonical_target_rank
  target_parity$residual_key_sha256 <-
    shard_targets$residual_key_sha256
  target_parity$target <- shard_targets$target
  target_parity$null_dim <- shard_targets$null_dim
  target_parity$condition <- shard_targets$condition
  target_parity$condition_bucket <- vapply(
    seq_len(target_count_value), function(index) {
      fastkpc_full_cuda_census_condition_bucket(
        target_parity$condition[[index]],
        shard_targets$coefficient_rank[[index]],
        target_parity$null_dim[[index]]
      )
    }, character(1L)
  )
  target_parity$phase1_coefficient_rank <-
    shard_targets$coefficient_rank
  target_parity$planned_route <- shard_targets$planned_route
  target_parity$authenticated_planned_route <- shard_targets$planned_route
  target_parity$executed_route <- shard_targets$planned_route
  target_parity$reroute_reason <- rep.int("", target_count_value)
  cholesky <- target_parity$planned_route == "CHOLESKY_BATCHED"
  qr <- target_parity$planned_route == "AUGMENTED_QR"
  svd <- target_parity$planned_route == "AUGMENTED_SVD"
  cholesky_count <- vapply(shard_setup_keys, function(key) {
    sum(cholesky[target_parity$prepared_s_key_sha256 == key])
  }, integer(1L))
  true_batched <- cholesky & cholesky_count[setup_rank] >= 2L
  target_parity$solver_status <- ifelse(
    cholesky,
    ifelse(true_batched, "OK_CHOLESKY_BATCHED", "OK_CHOLESKY_SINGLE"),
    ifelse(qr, "OK_AUGMENTED_QR", "OK_AUGMENTED_SVD")
  )
  target_parity$target_true_batched <- true_batched
  target_parity$true_batched_kernel <- vapply(
    seq_len(target_count_value), function(index) {
      selected_target <- setup_rank == setup_rank[[index]]
      sum(selected_target) >= 2L && all(true_batched[selected_target])
    }, logical(1L)
  )
  target_parity$true_batched_target_count <-
    as.integer(ifelse(
      cholesky_count[setup_rank] >= 2L,
      cholesky_count[setup_rank], 0L
    ))
  target_parity$qr_rank <- as.integer(ifelse(
    qr, target_parity$null_dim, -1L
  ))
  target_parity$geqrf_info <- as.integer(ifelse(qr, 0L, -1L))
  target_parity$ormqr_info <- as.integer(ifelse(qr, 0L, -1L))
  target_parity$effective_rank <- as.integer(ifelse(
    svd, pmax(1L, pmin(
      shard_targets$coefficient_rank, target_parity$null_dim
    )), -1L
  ))
  target_parity$sigma_max <- ifelse(svd, 10, NaN)
  target_parity$smallest_retained_sigma <- ifelse(svd, 1, NaN)
  target_parity$svd_info <- as.integer(ifelse(svd, 0L, -1L))
  target_parity$aggregate_penalty_root_rank <- as.integer(ifelse(
    svd, pmax(1L, pmin(
      shard_targets$coefficient_rank, target_parity$null_dim
    )), NA_integer_
  ))
  target_parity$aggregate_penalty_root_pivot_sha256 <- vapply(
    shard_targets$residual_key_sha256,
    function(key) sha(paste0("pivot-", key)), character(1L)
  )
  target_parity$aggregate_factor_call_count <- as.integer(svd)
  target_parity$aggregate_b_build_count <- 2L * as.integer(svd)
  target_parity$aggregate_dstop <- ifelse(svd, 1e-12, NA_real_)
  target_parity$numeric_reference <- rep.int(
    "mgcv-fixed-sp", target_count_value
  )
  for (field in c(
    "coefficient_all_finite", "fitted_all_finite",
    "residual_all_finite", "rss_all_finite", "rhs_all_finite",
    "output_all_finite", "coefficient_oracle_phase2_exact",
    "fitted_oracle_phase2_exact", "residual_oracle_phase2_exact",
    "full_cuda_data_plane"
  )) target_parity[[field]] <- rep.int(TRUE, target_count_value)
  for (field in grep(
    "_(max_abs_diff|relative_l2)$", names(target_parity), value = TRUE
  )) target_parity[[field]] <- rep.int(0, target_count_value)
  for (field in grep(
    "(sha256|fingerprint)$", names(target_parity), value = TRUE
  )) {
    if (!field %in% c(
      "prepared_s_key_sha256", "residual_key_sha256",
      "aggregate_penalty_root_pivot_sha256"
    )) {
      target_parity[[field]] <- vapply(
        shard_targets$residual_key_sha256,
        function(key) sha(paste0(field, "-", key)), character(1L)
      )
    }
  }
  target_parity$selected_sp_sha256 <- shard_targets$selected_sp_hash
  target_parity$coefficient_phase2_sha256 <- shard_targets$coefficient_hash
  target_parity$fitted_phase2_sha256 <- shard_targets$fitted_hash
  target_parity$residual_phase2_sha256 <- shard_targets$residual_hash
  target_parity$target_fit_fingerprint <-
    shard_targets$target_fit_fingerprint
  target_parity$oracle_call_count <- rep.int(1L, target_count_value)
  target_parity$rhs_authority <- rep.int(
    "cuda-x0-transpose-y", target_count_value
  )
  target_parity$approximate_backend <- rep.int(FALSE, target_count_value)
  target_parity$fallback_type <- rep.int("NONE", target_count_value)
  target_parity$error_code <- rep.int("NONE", target_count_value)
  target_parity$error_message_sha256 <- rep.int(
    sha(""), target_count_value
  )

  authority_match <- match(
    shard_setup_keys, setup_authority$prepared_s_key_sha256
  )
  shard_setup_authority <- setup_authority[authority_match, , drop = FALSE]
  setup_results <- oracle_frame("setup_results", setup_count)
  setup_results$prepared_s_key_sha256 <- shard_setup_keys
  setup_results$shard_id <- rep.int(as.integer(shard_id), setup_count)
  setup_results$setup_ordinal <- seq_len(setup_count)
  setup_results$canonical_setup_rank <-
    shard_setup_authority$canonical_setup_rank
  setup_results$phase2_shard_id <- as.integer(vapply(
    shard_setup_keys, function(key) unique(
      shard_targets$phase2_shard_id[
        shard_targets$prepared_s_key_sha256 == key
      ]
    ), integer(1L)
  ))
  oracle_phase2_unit <- as.integer(!duplicated(data.frame(
    shard_id = setup_results$shard_id,
    phase2_shard_id = setup_results$phase2_shard_id
  )))
  setup_results$phase2_shard_load_count <- oracle_phase2_unit
  setup_results$phase2_shard_authentication_count <- oracle_phase2_unit
  setup_results$n <- shard_setup_authority$n
  setup_results$coefficient_dim <- shard_setup_authority$coefficient_dim
  setup_results$null_dim <- shard_setup_authority$null_dim
  setup_results$penalty_count <- shard_setup_authority$penalty_count
  setup_results$target_count <- as.integer(vapply(
    shard_setup_keys, function(key) {
      sum(target_parity$prepared_s_key_sha256 == key)
    }, integer(1L)
  ))
  setup_results$target_key_set_sha256 <- vapply(
    shard_setup_keys, function(key) {
      fastkpc_full_cuda_census_key_set_hash(
        target_parity$residual_key_sha256[
          target_parity$prepared_s_key_sha256 == key
        ]
      )
    }, character(1L)
  )
  setup_results$prepared_handle_create_count <- rep.int(1L, setup_count)
  setup_results$prepared_handle_destroy_count <- rep.int(1L, setup_count)
  setup_results$setup_h2d_upload_count <- rep.int(1L, setup_count)
  setup_results$setup_h2d_bytes <-
    resource_fixture$setup_h2d_bytes[match(shard_setup_keys, setup_keys)]
  setup_results$penalty_root_build_count <-
    shard_setup_authority$penalty_count
  setup_results$penalty_root_matrix_count <-
    shard_setup_authority$penalty_count
  setup_results$penalty_root_row_count <-
    shard_setup_authority$penalty_count * shard_setup_authority$null_dim
  route_count_for_setup <- function(key, route) sum(
    target_parity$prepared_s_key_sha256 == key &
      target_parity$planned_route == route
  )
  for (prefix in c("planned", "executed")) {
    for (route_name in c("cholesky", "qr", "svd")) {
      route <- c(
        cholesky = "CHOLESKY_BATCHED", qr = "AUGMENTED_QR",
        svd = "AUGMENTED_SVD"
      )[[route_name]]
      setup_results[[paste0(prefix, "_", route_name, "_target_count")]] <-
        as.integer(vapply(
          shard_setup_keys, route_count_for_setup, integer(1L), route = route
        ))
    }
  }
  setup_results$true_batched_target_count <- as.integer(ifelse(
    cholesky_count >= 2L, cholesky_count, 0L
  ))
  setup_results$setup_load_elapsed_ms <- rep.int(0, setup_count)
  setup_results$total_elapsed_ms <- rep.int(0, setup_count)

  resource_metrics <- resource_fixture[
    match(shard_setup_keys, resource_fixture$prepared_s_key_sha256),
    , drop = FALSE
  ]
  resource_metrics$shard_id <- rep.int(as.integer(shard_id), setup_count)
  resource_metrics$setup_ordinal <- seq_len(setup_count)
  resource_metrics$phase2_shard_load_count <- oracle_phase2_unit
  resource_metrics$phase2_shard_authentication_count <- oracle_phase2_unit
  rownames(resource_metrics) <- NULL
  stage_timing <- do.call(rbind, lapply(seq_along(shard_setup_keys),
    function(index) {
      rows <- stage_fixture[
        stage_fixture$prepared_s_key_sha256 == shard_setup_keys[[index]],
        , drop = FALSE
      ]
      rows$shard_id <- rep.int(as.integer(shard_id), nrow(rows))
      rows$setup_ordinal <- rep.int(as.integer(index), nrow(rows))
      rows
    }
  ))
  rownames(stage_timing) <- NULL
  fallbacks <- .fastkpc_full_cuda_phase3_oracle_fallback_rows(
    target_parity, resource_metrics
  )
  failures <- .fastkpc_full_cuda_phase3_oracle_failure_rows(target_parity)
  summary <- fastkpc_full_cuda_phase3_summarize_oracle_rows(
    setup_results, target_parity, resource_metrics, stage_timing,
    fallbacks, failures
  )
  list(
    setup_results = setup_results, target_parity = target_parity,
    resource_metrics = resource_metrics, stage_timing = stage_timing,
    fallbacks = fallbacks, failures = failures, summary = summary
  )
}

oracle_context_count <- 0L
oracle_target_rows <- target_rows
oracle_target_rows$shard_id <- rep.int(0L, nrow(oracle_target_rows))
oracle_run <- fastkpc_full_cuda_phase3_run_shards(
  output_dir = oracle_sp_dir, kind = "oracle_sp",
  setup_keys = setup_keys, target_rows = oracle_target_rows,
  identity = identity, route_config = route_config,
  executor = function(context, shard_id, setup_keys, target_rows) {
    payload <- make_oracle_payload(
      shard_id, setup_keys, target_rows
    )
    tryCatch(
      .fastkpc_full_cuda_phase3_validate_oracle_payload(
        payload, expected_setup_keys = setup_keys,
        expected_target_rows = target_rows
      ),
      error = function(error) {
        field_map <- c(
          prepared_s_key_sha256 = "prepared_s_key_sha256",
          residual_key_sha256 = "residual_key_sha256",
          shard_id = "shard_id",
          canonical_setup_rank = "canonical_setup_rank",
          canonical_target_rank = "canonical_target_rank",
          target = "target", null_dim = "null_dim",
          condition = "condition",
          coefficient_rank = "phase1_coefficient_rank",
          planned_route = "planned_route",
          selected_sp_hash = "selected_sp_sha256",
          coefficient_hash = "coefficient_phase2_sha256",
          fitted_hash = "fitted_phase2_sha256",
          residual_hash = "residual_phase2_sha256",
          target_fit_fingerprint = "target_fit_fingerprint"
        )
        mismatched <- names(field_map)[!vapply(
          names(field_map), function(field) identical(
            payload$target_parity[[field_map[[field]]]],
            target_rows[[field]]
          ), logical(1L)
        )]
        stop(
          "oracle fixture shard ", shard_id, ": ",
          conditionMessage(error), "; target fields=",
          paste(mismatched, collapse = ","), call. = FALSE
        )
      }
    )
    count <- as.integer(length(setup_keys))
    list(
      payload = payload,
      resource_counts = list(
        prepared_handle_create_count = count,
        prepared_handle_destroy_count = count,
        residual_token_acquire_count = count,
        residual_token_release_count = count,
        output_slot_acquire_count = count,
        output_slot_release_count = count
      )
    )
  },
  runtime_create = function() {
    oracle_context_count <<- oracle_context_count + 1L
    new.env(parent = emptyenv())
  },
  runtime_destroy = function(context) invisible(NULL),
  scope = "iteration", shard_count = 1L
)
assert_true(
  identical(oracle_run$status, "complete") && oracle_context_count == 1L,
  "oracle fixture writes authenticated Phase 3 shard/session evidence"
)
oracle_merged <- fastkpc_full_cuda_phase3_merge_shards(
  output_dir = oracle_sp_dir, kind = "oracle_sp",
  setup_keys = setup_keys, target_rows = oracle_target_rows,
  identity = identity, route_config = route_config,
  scope = "iteration", shard_count = 1L
)
assert_true(
  identical(
    fastkpc_full_cuda_census_key_set_hash(
      sort(
        oracle_merged$payload$target_parity$residual_key_sha256,
        method = "radix"
      )
    ), identity$canonical_target_corpus_hash
  ),
  "merged oracle shard corpus matches the test identity"
)
risk_rows <- data.frame(
  residual_key_sha256 = target_rows$residual_key_sha256,
  high_condition = rep.int(FALSE, nrow(target_rows)),
  rank_deficient = rep.int(FALSE, nrow(target_rows)),
  nonfinite_metadata = rep.int(FALSE, nrow(target_rows)),
  near_constant_target = rep.int(FALSE, nrow(target_rows)),
  near_constant_conditioner = rep.int(FALSE, nrow(target_rows)),
  mgcv_warning = rep.int(FALSE, nrow(target_rows)),
  mgcv_nonconverged = rep.int(FALSE, nrow(target_rows)),
  near_alpha = rep.int(FALSE, nrow(target_rows)),
  stringsAsFactors = FALSE
)
oracle_publication <- tryCatch(
  fastkpc_full_cuda_phase3_publish_oracle_artifact(
    output_dir = oracle_sp_dir, setup_keys = setup_keys,
    target_rows = oracle_target_rows, identity = identity,
    route_config = route_config, scope = "iteration", shard_count = 1L,
    risk_rows = risk_rows,
    command_lines =
      "Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_shadow_artifact.R"
  ),
  error = function(error) {
    manifest_path <- file.path(oracle_sp_dir, "manifest.json")
    manifest <- if (file.exists(manifest_path)) {
      jsonlite::read_json(manifest_path, simplifyVector = TRUE)
    } else list(expected_target_hash = "missing")
    target_path <- file.path(oracle_sp_dir, "target_parity.rds")
    published_target_hash <- if (file.exists(target_path)) {
      fastkpc_full_cuda_census_key_set_hash(
        readRDS(target_path)$residual_key_sha256
      )
    } else "missing"
    stop(
      conditionMessage(error), "; identity target=",
      identity$canonical_target_corpus_hash, "; manifest target=",
      manifest$expected_target_hash, "; payload target=",
      published_target_hash, call. = FALSE
    )
  }
)
assert_true(
  identical(oracle_publication$status, "published") &&
    isTRUE(oracle_publication$validation$authenticated),
  "oracle fixture is completed through the Phase 3 publisher"
)
oracle_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  oracle_sp_dir, "oracle_sp"
)

production_oracle_dir <- tempfile("phase3-shadow-production-oracle-")
on.exit(unlink(production_oracle_dir, recursive = TRUE, force = TRUE),
        add = TRUE)
production_oracle_context_count <- 0L
production_oracle_run <- fastkpc_full_cuda_phase3_run_shards(
  output_dir = production_oracle_dir, kind = "oracle_sp",
  setup_keys = setup_keys, target_rows = oracle_target_rows,
  identity = production_identity, route_config = production_route,
  executor = function(context, shard_id, setup_keys, target_rows) {
    payload <- make_oracle_payload(shard_id, setup_keys, target_rows)
    count <- as.integer(length(setup_keys))
    list(
      payload = payload,
      resource_counts = list(
        prepared_handle_create_count = count,
        prepared_handle_destroy_count = count,
        residual_token_acquire_count = count,
        residual_token_release_count = count,
        output_slot_acquire_count = count,
        output_slot_release_count = count
      )
    )
  },
  runtime_create = function() {
    production_oracle_context_count <<-
      production_oracle_context_count + 1L
    new.env(parent = emptyenv())
  },
  runtime_destroy = function(context) invisible(NULL),
  scope = "iteration", shard_count = 1L,
  executed_native_library_sha256 = production_native_sha256
)
assert_true(
  identical(production_oracle_run$status, "complete") &&
    production_oracle_context_count == 1L,
  "production-shaped oracle fixture writes authenticated shard evidence"
)
original_open_canonical_oracle_catalog <- get(
  ".fastkpc_full_cuda_phase3_open_canonical_oracle_catalog",
  envir = globalenv()
)
original_phase3_input_identity <- get(
  "fastkpc_full_cuda_phase3_input_identity", envir = globalenv()
)
assign(
  ".fastkpc_full_cuda_phase3_open_canonical_oracle_catalog",
  function() catalog, envir = globalenv()
)
assign(
  "fastkpc_full_cuda_phase3_input_identity",
  function(catalog, device_id) production_identity,
  envir = globalenv()
)
on.exit({
  assign(
    ".fastkpc_full_cuda_phase3_open_canonical_oracle_catalog",
    original_open_canonical_oracle_catalog, envir = globalenv()
  )
  assign(
    "fastkpc_full_cuda_phase3_input_identity",
    original_phase3_input_identity, envir = globalenv()
  )
}, add = TRUE)
production_oracle_publication <-
  fastkpc_full_cuda_phase3_publish_oracle_artifact(
    output_dir = production_oracle_dir, setup_keys = setup_keys,
    target_rows = oracle_target_rows, identity = production_identity,
    route_config = production_route, scope = "iteration", shard_count = 1L,
    risk_rows = NULL, catalog = catalog, device_id = 0L,
    command_lines =
      "Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_shadow_artifact.R"
  )
assert_true(
  identical(production_oracle_publication$status, "published") &&
    isTRUE(fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
      production_oracle_dir, expected_identity = production_identity,
      require_full = FALSE
    )$authenticated),
  "production-shaped completed oracle fixture passes public validation"
)
volatile_native_path <- tempfile(
  "phase3-shadow-volatile-native-", fileext = .Platform$dynlib.ext
)
assert_true(
  file.copy(production_native_path, volatile_native_path),
  "volatile-session fixture copies the identical native image"
)
on.exit(unlink(volatile_native_path, force = TRUE), add = TRUE)
volatile_production_identity <- production_identity
volatile_production_identity$execution_snapshot_sha256 <-
  sha("different-production-execution-snapshot")
volatile_production_identity$native_library_path <-
  normalizePath(volatile_native_path, mustWork = TRUE)
volatile_production_identity$native_library_inode <- "2"
volatile_production_identity$native_build_dependencies_sha256 <-
  sha("different-production-native-build-dependencies")
volatile_production_identity$native_build_trace_sha256 <-
  sha("different-production-native-build-trace")
volatile_production_identity$sha256 <-
  .fastkpc_full_cuda_phase3_identity_hash(volatile_production_identity)
assert_true(
  identical(volatile_production_identity$sha256,
            production_identity$sha256),
  "volatile-session fixture preserves the stable production identity hash"
)
volatile_oracle_linkage <- tryCatch(
  .fastkpc_full_cuda_phase3_shadow_oracle_sp_evidence(
    production_oracle_dir,
    identity = volatile_production_identity,
    catalog = catalog,
    scope = "iteration",
    executed_native_library_sha256 = production_native_sha256
  ),
  error = function(error) error
)
assert_true(
  !inherits(volatile_oracle_linkage, "error") &&
    identical(
      volatile_oracle_linkage$manifest_sha256,
      fastkpc_full_cuda_census_file_hash(
        file.path(production_oracle_dir, "manifest.json")
      )
    ),
  paste(
    "completed-oracle linkage permits volatile native-session and trace",
    "identity fields while retaining stable/native gates:",
    if (inherits(volatile_oracle_linkage, "error")) {
      conditionMessage(volatile_oracle_linkage)
    } else {
      "accepted"
    }
  )
)
.task9_final_native_recheck_count <- 0L
.task9_final_native_replacement_path <- production_native_path
.task9_final_native_replacement_bytes <- readBin(
  production_native_path, "raw", n = file.info(production_native_path)$size
)
invisible(trace(
  ".fastkpc_full_cuda_phase3_oracle_immutable_file_hashes",
  exit = quote({
    .task9_final_native_recheck_count <<-
      .task9_final_native_recheck_count + 1L
    if (.task9_final_native_recheck_count == 2L) {
      replacement <- tempfile(
        "phase3-oracle-native-late-replacement-",
        tmpdir = dirname(.task9_final_native_replacement_path)
      )
      writeBin(charToRaw("late replacement native"), replacement)
      if (!file.rename(
            replacement, .task9_final_native_replacement_path
          )) {
        stop("failed to atomically replace late native fixture",
             call. = FALSE)
      }
    }
  }), print = FALSE, where = globalenv()
))
late_native_error <- tryCatch({
  fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
    production_oracle_dir, expected_identity = production_identity,
    require_full = FALSE
  )
  NULL
}, error = function(error) error)
untrace(
  ".fastkpc_full_cuda_phase3_oracle_immutable_file_hashes",
  where = globalenv()
)
production_native_connection <- file(production_native_path, open = "wb")
writeBin(
  .task9_final_native_replacement_bytes, production_native_connection
)
close(production_native_connection)
rm(
  .task9_final_native_recheck_count,
  .task9_final_native_replacement_path,
  .task9_final_native_replacement_bytes,
  envir = globalenv()
)
assert_true(
  inherits(late_native_error, "error") && grepl(
    "final native recheck", conditionMessage(late_native_error),
    fixed = TRUE
  ) && grepl(
    "mismatch", conditionMessage(late_native_error), ignore.case = TRUE
  ),
  paste(
    "public completed-oracle validation rejects a native file atomically",
    "replaced after initial validation and immutable artifact rechecks;",
    "observed:", if (inherits(late_native_error, "error")) {
      conditionMessage(late_native_error)
    } else {
      "no error"
    }
  )
)
production_oracle_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  production_oracle_dir, "oracle_sp"
)
production_manifest_before <- readBin(
  production_oracle_paths$manifest_json, "raw",
  n = file.info(production_oracle_paths$manifest_json)$size
)
production_summary_before <- readBin(
  production_oracle_paths$summary_json, "raw",
  n = file.info(production_oracle_paths$summary_json)$size
)
production_manifest <- jsonlite::read_json(
  production_oracle_paths$manifest_json, simplifyVector = FALSE
)
production_manifest$input_identity$native_library_sha256 <- strrep("0", 64L)
.fastkpc_full_cuda_phase3_write_json_exact(
  production_manifest, production_oracle_paths$manifest_json
)
production_summary <- jsonlite::read_json(
  production_oracle_paths$summary_json, simplifyVector = FALSE
)
production_summary$manifest_sha256 <- fastkpc_full_cuda_census_file_hash(
  production_oracle_paths$manifest_json
)
.fastkpc_full_cuda_phase3_write_json_exact(
  production_summary, production_oracle_paths$summary_json
)
production_native_error <- tryCatch({
  fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
    production_oracle_dir, expected_identity = production_identity,
    require_full = FALSE
  )
  NULL
}, error = function(error) error)
assert_true(
  inherits(production_native_error, "error") && grepl(
    "native", conditionMessage(production_native_error), ignore.case = TRUE
  ) && grepl(
    "mismatch", conditionMessage(production_native_error), ignore.case = TRUE
  ),
  paste(
    "public completed-oracle validator rejects an embedded-only native SHA",
    "mutation through the production four-way gate"
  )
)
manifest_connection <- file(
  production_oracle_paths$manifest_json, open = "wb"
)
writeBin(production_manifest_before, manifest_connection)
close(manifest_connection)
summary_connection <- file(
  production_oracle_paths$summary_json, open = "wb"
)
writeBin(production_summary_before, summary_connection)
close(summary_connection)
assign(
  ".fastkpc_full_cuda_phase3_open_canonical_oracle_catalog",
  original_open_canonical_oracle_catalog, envir = globalenv()
)
assign(
  "fastkpc_full_cuda_phase3_input_identity",
  original_phase3_input_identity, envir = globalenv()
)

publish_args <- list(
  output_dir = output_dir, catalog = catalog, setup_keys = setup_keys,
  target_rows = target_rows, identity = identity,
  route_config = route_config, scope = "iteration",
  phase0_dir = phase0_dir, oracle_sp_dir = oracle_sp_dir,
  shard_count = 64L, direct_logical_sequence_id = 1L,
  command_lines =
    "Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_shadow_artifact.R"
)
.task9_heavy_probe <- new.env(parent = emptyenv())
.task9_heavy_probe$merge <- 0L
.task9_heavy_probe$route <- 0L
.task9_heavy_probe$replay <- 0L
.task9_heavy_probe$plan <- 0L
.task9_heavy_probe$scope_authority <- 0L
.task9_heavy_probe$snapshot_create <- 0L
.task9_heavy_probe$snapshot_release <- 0L
invisible(trace(
  "fastkpc_full_cuda_phase3_merge_shards",
  tracer = quote(if (identical(kind, "full_shadow")) {
    .task9_heavy_probe$merge <- .task9_heavy_probe$merge + 1L
  }), print = FALSE, where = globalenv()
))
invisible(trace(
  "fastkpc_full_cuda_shadow_reconstruct_target_routes",
  tracer = quote({
    .task9_heavy_probe$route <- .task9_heavy_probe$route + 1L
  }), print = FALSE, where = globalenv()
))
invisible(trace(
  "fastkpc_full_cuda_replay_logical_ci",
  tracer = quote({
    .task9_heavy_probe$replay <- .task9_heavy_probe$replay + 1L
  }), print = FALSE, where = globalenv()
))
invisible(trace(
  "fastkpc_full_cuda_shadow_plan",
  tracer = quote({
    .task9_heavy_probe$plan <- .task9_heavy_probe$plan + 1L
  }), print = FALSE, where = globalenv()
))
invisible(trace(
  ".fastkpc_full_cuda_phase3_shadow_scope_authority",
  tracer = quote({
    .task9_heavy_probe$scope_authority <-
      .task9_heavy_probe$scope_authority + 1L
  }), print = FALSE, where = globalenv()
))
invisible(trace(
  "fastkpc_full_cuda_phase3_create_shadow_execution_snapshot",
  tracer = quote({
    .task9_heavy_probe$snapshot_create <-
      .task9_heavy_probe$snapshot_create + 1L
  }), print = FALSE, where = globalenv()
))
invisible(trace(
  "fastkpc_full_cuda_phase3_release_shadow_execution_snapshot",
  tracer = quote({
    .task9_heavy_probe$snapshot_release <-
      .task9_heavy_probe$snapshot_release + 1L
  }), print = FALSE, where = globalenv()
))
on.exit({
  if (exists(".task9_heavy_probe", envir = globalenv(), inherits = FALSE)) {
    for (name in c(
      "fastkpc_full_cuda_phase3_merge_shards",
      "fastkpc_full_cuda_shadow_reconstruct_target_routes",
      "fastkpc_full_cuda_replay_logical_ci",
      "fastkpc_full_cuda_shadow_plan",
      ".fastkpc_full_cuda_phase3_shadow_scope_authority",
      "fastkpc_full_cuda_phase3_create_shadow_execution_snapshot",
      "fastkpc_full_cuda_phase3_release_shadow_execution_snapshot"
    )) untrace(name, where = globalenv())
    rm(.task9_heavy_probe, envir = globalenv())
  }
}, add = TRUE)
publisher_child_identity <- tempfile(
  "phase3-shadow-publisher-child-identity-", fileext = ".rds"
)
publisher_child_setup <- tempfile(
  "phase3-shadow-publisher-child-setup-", fileext = ".rds"
)
publisher_child_targets <- tempfile(
  "phase3-shadow-publisher-child-targets-", fileext = ".rds"
)
saveRDS(identity, publisher_child_identity, version = 2)
saveRDS(setup_keys, publisher_child_setup, version = 2)
saveRDS(target_rows, publisher_child_targets, version = 2)
publisher_crash_script <- tempfile(
  "phase3-shadow-real-publisher-crash-", fileext = ".R"
)
writeLines(c(
  'source("fastkpc/R/full_cuda_ci_gate.R")',
  'source("fastkpc/R/full_cuda_ci_oracle_contract.R")',
  'source("fastkpc/R/full_cuda_ci_workload_census.R")',
  'source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")',
  'source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")',
  'source("fastkpc/R/full_cuda_ci_fixed_sp_shadow.R")',
  'source("fastkpc/R/full_cuda_ci_phase3_artifacts.R")',
  sprintf(
    'catalog <- fastkpc_full_cuda_open_fixed_sp_catalog(%s,%s,%s,%s)',
    encodeString(phase0_dir, quote = '"'),
    encodeString(phase1_dir, quote = '"'),
    encodeString(phase2_dir, quote = '"'),
    encodeString(data_path, quote = '"')
  ),
  sprintf('identity <- readRDS(%s)',
          encodeString(publisher_child_identity, quote = '"')),
  sprintf('setup_keys <- readRDS(%s)',
          encodeString(publisher_child_setup, quote = '"')),
  sprintf('target_rows <- readRDS(%s)',
          encodeString(publisher_child_targets, quote = '"')),
  sprintf(paste0(
    'fastkpc_full_cuda_phase3_publish_shadow_artifact(',
    'output_dir=%s,catalog=catalog,setup_keys=setup_keys,',
    'target_rows=target_rows,identity=identity,',
    'route_config=fastkpc_full_cuda_phase3_route_config(),',
    'scope="iteration",phase0_dir=%s,oracle_sp_dir=%s,',
    'shard_count=64L,direct_logical_sequence_id=1L,',
    'command_lines="Task 9 real publisher crash fixture",',
    '.publication_hook=function(event) { if (identical(event,',
    '"after_staged_payload_write")) tools::pskill(Sys.getpid(),9L) })'
  ),
  encodeString(output_dir, quote = '"'),
  encodeString(phase0_dir, quote = '"'),
  encodeString(oracle_sp_dir, quote = '"'))
), publisher_crash_script, useBytes = TRUE)
on.exit(unlink(c(
  publisher_child_identity, publisher_child_setup,
  publisher_child_targets, publisher_crash_script
), force = TRUE), add = TRUE)
publisher_crash_output <- suppressWarnings(system2(
  file.path(R.home("bin"), "Rscript"), publisher_crash_script,
  stdout = TRUE, stderr = TRUE
))
shadow_work_siblings <- function(purpose) {
  entries <- list.files(
    dirname(output_dir), full.names = TRUE, all.files = TRUE, no.. = TRUE
  )
  prefix <- .fastkpc_full_cuda_phase3_shadow_work_dir_prefix(
    output_dir, purpose
  )
  entries[startsWith(basename(entries), prefix)]
}
crashed_staging <- shadow_work_siblings("staging")
assert_true(
  !is.null(attr(publisher_crash_output, "status")) &&
    attr(publisher_crash_output, "status") != 0L &&
    !file.exists(artifact_paths$manifest_json) &&
    !file.exists(artifact_paths$summary_json) &&
    length(crashed_staging) == 1L &&
    file.exists(.fastkpc_full_cuda_phase3_shadow_work_dir_marker_path(
      crashed_staging[[1L]]
    )) && file.exists(file.path(
      crashed_staging[[1L]], "logical_ci_parity.rds"
    )),
  paste(
    "real publisher hard kill after its first staged payload leaves no",
    "completion markers and one authenticated stale staging directory;",
    "status=", attr(publisher_crash_output, "status"),
    "manifest=", file.exists(artifact_paths$manifest_json),
    "summary=", file.exists(artifact_paths$summary_json),
    "staging_count=", length(crashed_staging),
    "marker=", length(crashed_staging) == 1L && file.exists(
      .fastkpc_full_cuda_phase3_shadow_work_dir_marker_path(
        crashed_staging[[1L]]
      )
    ),
    "payload=", length(crashed_staging) == 1L && file.exists(file.path(
      crashed_staging[[1L]], "logical_ci_parity.rds"
    )),
    "output=", paste(publisher_crash_output, collapse = " | ")
  )
)
stale_rollback_lock <-
  .fastkpc_full_cuda_phase3_acquire_shadow_artifact_lock(output_dir)
stale_rollback <- .fastkpc_full_cuda_phase3_acquire_shadow_work_dir(
  output_dir, "rollback", stale_rollback_lock
)
.fastkpc_full_cuda_phase3_release_shadow_artifact_lock(stale_rollback_lock)
published <- do.call(
  fastkpc_full_cuda_phase3_publish_shadow_artifact, publish_args
)
publisher_heavy_counts <- c(
  plan = .task9_heavy_probe$plan,
  scope = .task9_heavy_probe$scope_authority,
  snapshot = .task9_heavy_probe$snapshot_create
)
assert_true(
  identical(publisher_heavy_counts, c(plan = 1L, scope = 1L, snapshot = 1L)),
  paste(
    "publisher constructs and selects authenticated scope exactly once:",
    paste(names(publisher_heavy_counts), publisher_heavy_counts,
          sep = "=", collapse = ",")
  )
)
assert_true(
  identical(published$status, "published") &&
    isTRUE(published$validation$authenticated) &&
    !isTRUE(published$summary$full_scope) &&
    identical(published$summary$first_divergence, "NOT_APPLICABLE"),
  "authenticated selected-scope shadow artifact publishes and validates"
)
assert_true(
  length(shadow_work_siblings("staging")) == 0L &&
    length(shadow_work_siblings("rollback")) == 0L,
  "next real publication removes all owned stale staging and rollback siblings"
)
matching_public_validation <-
  fastkpc_validate_full_cuda_fixed_sp_shadow_artifact(
    output_dir, expected_identity = identity, catalog = catalog,
    device_id = identity$device_id, setup_keys = setup_keys,
    target_rows = target_rows, route_config = route_config,
    scope = "iteration", phase0_dir = phase0_dir,
    oracle_sp_dir = oracle_sp_dir, shard_count = 64L,
    canonical_setup_shards = FALSE,
    shadow_plan_identity_sha256 = plan$plan_identity_sha256,
    direct_logical_sequence_id = 1L, require_full = FALSE
  )
assert_true(
  isTRUE(matching_public_validation$authenticated) &&
    identical(
      c(
        merge = .task9_heavy_probe$merge,
        route = .task9_heavy_probe$route,
        replay = .task9_heavy_probe$replay
      ),
      c(merge = 2L, route = 2L, replay = 2L)
    ) && .task9_heavy_probe$snapshot_create == 2L &&
    .task9_heavy_probe$snapshot_release == 2L && identical(
      c(
        plan = .task9_heavy_probe$plan,
        scope = .task9_heavy_probe$scope_authority,
        snapshot = .task9_heavy_probe$snapshot_create
      ) - publisher_heavy_counts,
      c(plan = 1L, scope = 1L, snapshot = 1L)
    ),
  paste(
    "fresh publication plus standalone validation performs exactly two",
    "heavy passes with one plan/scope/snapshot build per pass"
  )
)
expect_standalone_snapshot_release <- function(
    function_name, tracer, label) {
  before <- c(
    create = .task9_heavy_probe$snapshot_create,
    release = .task9_heavy_probe$snapshot_release
  )
  invisible(trace(
    function_name, tracer = tracer, print = FALSE, where = globalenv()
  ))
  error <- tryCatch({
    fastkpc_validate_full_cuda_fixed_sp_shadow_artifact(
      output_dir, require_full = FALSE
    )
    NULL
  }, error = function(error) error)
  untrace(function_name, where = globalenv())
  after <- c(
    create = .task9_heavy_probe$snapshot_create,
    release = .task9_heavy_probe$snapshot_release
  )
  assert_true(
    inherits(error, "error") && identical(
      after - before, c(create = 1L, release = 1L)
    ),
    paste(label, "releases exactly one standalone execution snapshot")
  )
  invisible(error)
}
expect_standalone_snapshot_release(
  ".fastkpc_full_cuda_phase3_shadow_oracle_sp_evidence",
  quote(stop("injected completed-oracle native mismatch", call. = FALSE)),
  "completed-oracle native mismatch"
)
expect_standalone_snapshot_release(
  "fastkpc_full_cuda_compare_candidate_skeleton",
  quote(stop("injected graph replay comparison failure", call. = FALSE)),
  "graph replay comparison failure"
)
.task9_phase0_reload_calls <- 0L
expect_standalone_snapshot_release(
  "fastkpc_full_cuda_census_load_inputs",
  quote({
    .task9_phase0_reload_calls <<- .task9_phase0_reload_calls + 1L
    if (.task9_phase0_reload_calls == 2L) {
      stop("injected fresh Phase 0 reload failure", call. = FALSE)
    }
  }),
  "fresh Phase 0 reload failure"
)
rm(.task9_phase0_reload_calls, envir = globalenv())
if (publication_only) {
  cat("full CUDA CI fixed-sp shadow artifact publication probes: PASS\n")
  quit(save = "no", status = 0L)
}
wrong_identity_constraint <- identity
wrong_identity_constraint$source_commit <- strrep("9", 40L)
wrong_phase0_dir <- tempfile("phase3-shadow-wrong-phase0-")
dir.create(wrong_phase0_dir, recursive = TRUE, showWarnings = FALSE)
assert_true(
  all(file.copy(
    phase0_source_files, wrong_phase0_dir, recursive = TRUE,
    copy.mode = TRUE, copy.date = TRUE
  )),
  "fixture creates a separate valid-looking Phase 0 directory"
)
wrong_phase0_link <- tempfile("phase3-shadow-wrong-phase0-link-")
assert_true(
  file.symlink(wrong_phase0_dir, wrong_phase0_link),
  "fixture creates a Phase 0 symlink alias"
)
on.exit(unlink(
  c(wrong_phase0_dir, wrong_phase0_link), recursive = TRUE, force = TRUE
), add = TRUE)
for (constraint_case in list(
  list(
    label = "mismatched expected identity",
    args = list(expected_identity = wrong_identity_constraint)
  ),
  list(label = "malformed supplied catalog", args = list(catalog = list())),
  list(
    label = "mismatched source path",
    args = list(phase0_dir = dirname(phase0_dir))
  ),
  list(
    label = "different valid-looking Phase 0 authority",
    args = list(phase0_dir = wrong_phase0_dir)
  ),
  list(
    label = "symlink alias to different Phase 0 authority",
    args = list(phase0_dir = wrong_phase0_link)
  ),
  list(
    label = "unsupported caller snapshot",
    args = list(execution_snapshot = list(forged = TRUE))
  )
)) {
  constraint_error <- tryCatch({
    do.call(
      fastkpc_validate_full_cuda_fixed_sp_shadow_artifact,
      c(list(output_dir = output_dir, require_full = FALSE),
        constraint_case$args)
    )
    NULL
  }, error = function(error) error)
  assert_true(
    inherits(constraint_error, "error"),
    paste(constraint_case$label, "fails v2 public validation")
  )
}
subprocess_sources <- c(
  "fastkpc/R/full_cuda_ci_gate.R",
  "fastkpc/R/full_cuda_ci_oracle_contract.R",
  "fastkpc/R/full_cuda_ci_workload_census.R",
  "fastkpc/R/full_cuda_ci_prepared_s_contract.R",
  "fastkpc/R/full_cuda_ci_fixed_sp_runtime.R",
  "fastkpc/R/full_cuda_ci_fixed_sp_shadow.R",
  "fastkpc/R/full_cuda_ci_phase3_artifacts.R"
)
subprocess_expression <- paste0(
  paste(sprintf(
    "source(%s)", encodeString(subprocess_sources, quote = "\"")
  ), collapse = ";"),
  ";value <- fastkpc_validate_full_cuda_fixed_sp_shadow_artifact(",
  encodeString(
    normalizePath(output_dir, winslash = "/", mustWork = TRUE),
    quote = "\""
  ),
  ", require_full=FALSE);stopifnot(isTRUE(value$authenticated),",
  "!is.null(value$recomputed_graph));cat('TASK9_RECOMPUTED\\n')"
)
subprocess_output <- system2(
  "Rscript", c("--vanilla", "-e", shQuote(subprocess_expression)),
  stdout = TRUE, stderr = TRUE
)
assert_true(
  is.null(attr(subprocess_output, "status")) &&
    any(subprocess_output == "TASK9_RECOMPUTED"),
  paste(
    "clean-process documented public validator call performs Task 9",
    "source/oracle/graph recomputation"
  )
)
full_error <- tryCatch({
  fastkpc_validate_full_cuda_fixed_sp_shadow_artifact(
    output_dir, require_full = TRUE
  )
  NULL
}, error = function(error) error)
assert_true(
  inherits(full_error, "error") && grepl(
    "non-full shadow artifact", conditionMessage(full_error), fixed = TRUE
  ),
  "non-full artifact cannot satisfy require_full=TRUE"
)

shadow_shard_paths <- sort(list.files(
  artifact_paths$shards_dir,
  pattern = "^shard_[0-9]+\\.(rds|summary\\.json)$",
  full.names = TRUE
), method = "radix")
shadow_session_paths <- sort(list.files(
  artifact_paths$sessions_dir, full.names = TRUE
), method = "radix")
resume_byte_paths <- c(
  artifact_paths$direct_ci_rds, artifact_paths$direct_ci_summary_json,
  shadow_shard_paths, shadow_session_paths,
  unlist(artifact_paths[.fastkpc_full_cuda_phase3_shadow_publication_keys()],
         use.names = FALSE)
)
resume_byte_hashes <- vapply(
  resume_byte_paths, fastkpc_full_cuda_census_file_hash, character(1L)
)
resume_metadata <- file.info(resume_byte_paths)[
  , c("size", "mode", "mtime"), drop = FALSE
]
semantic_snapshot <- function() {
  shard_rds <- shadow_shard_paths[grepl("\\.rds$", shadow_shard_paths)]
  list(
    direct_rows = fastkpc_full_cuda_census_frame_hash(
      readRDS(artifact_paths$direct_ci_rds)$rows
    ),
    direct_summary = .fastkpc_full_cuda_phase3_named_hash(
      jsonlite::read_json(
        artifact_paths$direct_ci_summary_json, simplifyVector = FALSE
      )
    ),
    shard_payload = unname(vapply(shard_rds, function(path) {
      readRDS(path)$payload_semantic_hash
    }, character(1L))),
    logical_rows = fastkpc_full_cuda_census_frame_hash(
      readRDS(artifact_paths$logical_ci_parity_rds)
    ),
    adjacency = digest::digest(
      readRDS(artifact_paths$adjacency_rds),
      algo = "sha256", serialize = TRUE
    ),
    deletion = digest::digest(
      read.csv(artifact_paths$deletion_trace_csv,
               stringsAsFactors = FALSE, check.names = FALSE),
      algo = "sha256", serialize = TRUE
    ),
    sepset = digest::digest(
      read.csv(artifact_paths$sepset_agreement_csv,
               stringsAsFactors = FALSE, check.names = FALSE),
      algo = "sha256", serialize = TRUE
    ),
    n_edgetests = digest::digest(
      read.csv(artifact_paths$n_edgetests_csv,
               stringsAsFactors = FALSE, check.names = FALSE),
      algo = "sha256", serialize = TRUE
    ),
    first_divergence = .fastkpc_full_cuda_phase3_named_hash(
      jsonlite::read_json(
        artifact_paths$first_divergence_json, simplifyVector = FALSE
      )
    ),
    manifest = .fastkpc_full_cuda_phase3_named_hash(
      jsonlite::read_json(
        artifact_paths$manifest_json, simplifyVector = FALSE
      )
    ),
    summary = .fastkpc_full_cuda_phase3_named_hash(
      jsonlite::read_json(
        artifact_paths$summary_json, simplifyVector = FALSE
      )
    )
  )
}
resume_semantic_hashes <- semantic_snapshot()
resume_context_create <- 0L
resume_context_destroy <- 0L
pure_resume <- fastkpc_full_cuda_phase3_run_shards(
  output_dir = output_dir, kind = "full_shadow",
  setup_keys = setup_keys, target_rows = target_rows,
  identity = identity, route_config = route_config,
  executor = function(...) stop("pure resume executed a shard", call. = FALSE),
  runtime_create = function() {
    resume_context_create <<- resume_context_create + 1L
    stop("pure resume created a runtime", call. = FALSE)
  },
  runtime_destroy = function(context) {
    resume_context_destroy <<- resume_context_destroy + 1L
    invisible(NULL)
  },
  scope = "iteration", shard_count = 64L
)
heavy_before_reuse <- c(
  merge = .task9_heavy_probe$merge,
  route = .task9_heavy_probe$route,
  replay = .task9_heavy_probe$replay
)
reused_publication <- do.call(
  fastkpc_full_cuda_phase3_publish_shadow_artifact, publish_args
)
assert_true(
  identical(pure_resume$reused_shard_ids, 0:63) &&
    length(pure_resume$written_shard_ids) == 0L &&
    is.null(pure_resume$session_id) &&
    pure_resume$runtime_context_create_count == 0L &&
    pure_resume$runtime_context_destroy_count == 0L &&
    resume_context_create == 0L && resume_context_destroy == 0L &&
    identical(reused_publication$status, "reused") && identical(
      vapply(resume_byte_paths, fastkpc_full_cuda_census_file_hash,
             character(1L)), resume_byte_hashes
    ) && identical(semantic_snapshot(), resume_semantic_hashes) &&
    identical(
      file.info(resume_byte_paths)[
        , c("size", "mode", "mtime"), drop = FALSE
      ], resume_metadata
    ) && identical(
      c(
        merge = .task9_heavy_probe$merge,
        route = .task9_heavy_probe$route,
        replay = .task9_heavy_probe$replay
      ) - heavy_before_reuse,
      c(merge = 1L, route = 1L, replay = 1L)
    ),
  paste(
    "pure resume reuses direct, all 64 shards, sessions, and publication",
    paste(
      "with zero CUDA contexts/writes/churn, identical byte/semantic",
      "hashes/metadata, and exactly one reuse validation pass"
    )
  )
)

incomplete_dir <- tempfile("phase3-shadow-incomplete-session-")
dir.create(incomplete_dir, recursive = TRUE, showWarnings = FALSE)
on.exit(unlink(incomplete_dir, recursive = TRUE, force = TRUE), add = TRUE)
copied <- file.copy(
  list.files(output_dir, all.files = TRUE, no.. = TRUE, full.names = TRUE),
  incomplete_dir, recursive = TRUE, copy.mode = TRUE, copy.date = TRUE
)
assert_true(all(copied), "incomplete-session fixture clones publication")
incomplete_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  incomplete_dir, "full_shadow"
)
incomplete_session_path <- list.files(
  incomplete_paths$sessions_dir, pattern = "^session_.*\\.json$",
  full.names = TRUE
)[[1L]]
incomplete_session <- jsonlite::read_json(
  incomplete_session_path, simplifyVector = FALSE
)
incomplete_session$status <- "running"
.fastkpc_full_cuda_phase3_write_json_exact(
  incomplete_session, incomplete_session_path
)
incomplete_direct_hashes <- vapply(
  c(incomplete_paths$direct_ci_rds,
    incomplete_paths$direct_ci_summary_json),
  fastkpc_full_cuda_census_file_hash, character(1L)
)
incomplete_context_create <- 0L
incomplete_context_destroy <- 0L
incomplete_resume <- fastkpc_full_cuda_phase3_run_shards(
  output_dir = incomplete_dir, kind = "full_shadow",
  setup_keys = setup_keys, target_rows = target_rows,
  identity = identity, route_config = route_config,
  executor = function(context, shard_id, setup_keys, target_rows) {
    rows <- conditional_fixture[
      conditional_fixture$shard_id == shard_id, , drop = FALSE
    ]
    resources <- resource_fixture[
      resource_fixture$shard_id == shard_id, , drop = FALSE
    ]
    stages <- stage_fixture[
      stage_fixture$shard_id == shard_id, , drop = FALSE
    ]
    rownames(rows) <- rownames(resources) <- rownames(stages) <- NULL
    count <- as.integer(length(setup_keys))
    list(
      payload = list(
        logical_ci_parity = rows,
        resource_metrics = resources,
        stage_timing = stages
      ),
      resource_counts = list(
        prepared_handle_create_count = count,
        prepared_handle_destroy_count = count,
        residual_token_acquire_count = count,
        residual_token_release_count = count,
        output_slot_acquire_count = count,
        output_slot_release_count = count
      )
    )
  },
  runtime_create = function() {
    incomplete_context_create <<- incomplete_context_create + 1L
    new.env(parent = emptyenv())
  },
  runtime_destroy = function(context) {
    incomplete_context_destroy <<- incomplete_context_destroy + 1L
    invisible(NULL)
  },
  scope = "iteration", shard_count = 64L
)
assert_true(
  identical(incomplete_resume$written_shard_ids, 0:63) &&
    length(incomplete_resume$reused_shard_ids) == 0L &&
    incomplete_resume$runtime_context_create_count == 1L &&
    incomplete_resume$runtime_context_destroy_count == 1L &&
    incomplete_context_create == 1L && incomplete_context_destroy == 1L &&
    identical(
      vapply(
        c(incomplete_paths$direct_ci_rds,
          incomplete_paths$direct_ci_summary_json),
        fastkpc_full_cuda_census_file_hash, character(1L)
      ), incomplete_direct_hashes
    ),
  paste(
    "all 64 shards from an incomplete session are recomputed once while",
    "the authenticated direct pair is reused byte-for-byte"
  )
)

validation_args <- c(
  publish_args[setdiff(names(publish_args), "command_lines")],
  list(require_full = FALSE)
)
snapshot_bytes <- function(paths) {
  setNames(lapply(paths, function(path) {
    readBin(path, what = "raw", n = file.info(path)$size)
  }), paths)
}
restore_bytes <- function(snapshot) {
  for (path in names(snapshot)) {
    connection <- file(path, open = "wb")
    on.exit(close(connection), add = TRUE)
    writeBin(snapshot[[path]], connection)
    close(connection)
    on.exit(NULL, add = FALSE)
  }
}
expect_rejected <- function(paths, mutate, label) {
  snapshot <- snapshot_bytes(paths)
  on.exit(restore_bytes(snapshot), add = TRUE)
  mutate()
  error <- tryCatch({
    do.call(
      fastkpc_validate_full_cuda_fixed_sp_shadow_artifact,
      validation_args
    )
    NULL
  }, error = function(error) error)
  assert_true(inherits(error, "error"), paste(label, "must fail closed"))
  restore_bytes(snapshot)
  on.exit(NULL, add = FALSE)
  invisible(error)
}

refresh_shadow_summary_manifest_hash <- function() {
  summary <- jsonlite::read_json(
    artifact_paths$summary_json, simplifyVector = FALSE
  )
  summary$manifest_sha256 <- fastkpc_full_cuda_census_file_hash(
    artifact_paths$manifest_json
  )
  .fastkpc_full_cuda_phase3_write_json_exact(
    summary, artifact_paths$summary_json
  )
}

manifest_schema_mutations <- list(
  null_schema = function(manifest) {
    manifest["manifest_schema_version"] <- list(NULL)
    manifest
  },
  unknown_schema = function(manifest) {
    manifest$manifest_schema_version <- "unknown-shadow-version"
    manifest
  },
  extra_manifest_field = function(manifest) {
    manifest$extra_claim <- TRUE
    manifest
  },
  missing_manifest_field = function(manifest) {
    manifest$source_commit <- NULL
    manifest
  },
  hybrid_null_legacy_claim = function(manifest) {
    manifest["manifest_schema_version"] <- list(NULL)
    manifest$artifact_kind <- "full_shadow"
    manifest$artifact_schema_version <-
      fastkpc_full_cuda_phase3_shadow_schema_version()
    manifest
  }
)
for (case_name in names(manifest_schema_mutations)) {
  mutate_manifest <- manifest_schema_mutations[[case_name]]
  expect_rejected(
    c(artifact_paths$manifest_json, artifact_paths$summary_json),
    function() {
      manifest <- jsonlite::read_json(
        artifact_paths$manifest_json, simplifyVector = FALSE
      )
      .fastkpc_full_cuda_phase3_write_json_exact(
        mutate_manifest(manifest), artifact_paths$manifest_json
      )
      refresh_shadow_summary_manifest_hash()
    }, case_name
  )
}
for (case_name in c("extra_summary_field", "missing_summary_field")) {
  expect_rejected(artifact_paths$summary_json, function() {
    summary <- jsonlite::read_json(
      artifact_paths$summary_json, simplifyVector = FALSE
    )
    if (identical(case_name, "extra_summary_field")) {
      summary$extra_claim <- TRUE
    } else {
      summary$source_artifact_sha256 <- NULL
    }
    .fastkpc_full_cuda_phase3_write_json_exact(
      summary, artifact_paths$summary_json
    )
  }, case_name)
}

expect_rejected(
  c(
    artifact_paths$logical_ci_parity_rds,
    artifact_paths$manifest_json, artifact_paths$summary_json
  ),
  function() {
    rows <- readRDS(artifact_paths$logical_ci_parity_rds)
    rows$candidate_p_value[[1L]] <- if (
      rows$reference_decision[[1L]] == "dependent"
    ) rows$alpha[[1L]] * 2 else rows$alpha[[1L]] / 2
    rows$candidate_decision[[1L]] <- ifelse(
      rows$candidate_p_value[[1L]] > rows$alpha[[1L]],
      "independent", "dependent"
    )
    rows$decision_flip[[1L]] <- TRUE
    rows$absolute_p_value_difference[[1L]] <- abs(
      rows$candidate_p_value[[1L]] - rows$reference_p_value[[1L]]
    )
    saveRDS(rows, artifact_paths$logical_ci_parity_rds,
            version = 2, compress = FALSE)
    manifest <- jsonlite::read_json(
      artifact_paths$manifest_json, simplifyVector = FALSE
    )
    manifest$merged_logical_rows_sha256 <-
      fastkpc_full_cuda_census_frame_hash(rows)
    manifest$file_sha256$logical_ci_parity_rds <-
      fastkpc_full_cuda_census_file_hash(
        artifact_paths$logical_ci_parity_rds
      )
    .fastkpc_full_cuda_phase3_write_json_exact(
      manifest, artifact_paths$manifest_json
    )
    summary <- jsonlite::read_json(
      artifact_paths$summary_json, simplifyVector = FALSE
    )
    summary$pass <- TRUE
    summary$decision_flip_count <- 0L
    summary$manifest_sha256 <- fastkpc_full_cuda_census_file_hash(
      artifact_paths$manifest_json
    )
    .fastkpc_full_cuda_phase3_write_json_exact(
      summary, artifact_paths$summary_json
    )
  },
  "candidate p-value with forged hashes and pass claims"
)

sequence_mutations <- list(
  missing_logical_sequence_id = function(rows) {
    rows$logical_sequence_id <- NULL
    rows
  },
  duplicate_logical_sequence_id = function(rows) {
    rows$logical_sequence_id[[2L]] <- rows$logical_sequence_id[[1L]]
    rows
  },
  reordered_logical_sequence_id = function(rows) rows[2:1, , drop = FALSE],
  noncontiguous_logical_sequence_id = function(rows) {
    rows$logical_sequence_id[[2L]] <- rows$logical_sequence_id[[2L]] + 1L
    rows
  },
  missing_scope_sequence_ordinal = function(rows) {
    rows$scope_sequence_ordinal <- NULL
    rows
  },
  duplicate_scope_sequence_ordinal = function(rows) {
    rows$scope_sequence_ordinal[[2L]] <- rows$scope_sequence_ordinal[[1L]]
    rows
  },
  reordered_scope_sequence_ordinal = function(rows) {
    rows$scope_sequence_ordinal[1:2] <- rows$scope_sequence_ordinal[2:1]
    rows
  },
  noncontiguous_scope_sequence_ordinal = function(rows) {
    rows$scope_sequence_ordinal[[2L]] <- 3L
    rows
  },
  wrong_sparse_global_logical_sequence_id = function(rows) {
    rows$logical_sequence_id[[2L]] <- 2215L
    rows
  }
)
for (case_name in names(sequence_mutations)) {
  mutate <- sequence_mutations[[case_name]]
  expect_rejected(
    artifact_paths$logical_ci_parity_rds,
    function() {
      rows <- mutate(readRDS(artifact_paths$logical_ci_parity_rds))
      rownames(rows) <- NULL
      saveRDS(rows, artifact_paths$logical_ci_parity_rds,
              version = 2, compress = FALSE)
    }, case_name
  )
}

nonempty_shard_rds <- shadow_shard_paths[grepl(
  "\\.rds$", shadow_shard_paths
)]
nonempty_shard_rds <- nonempty_shard_rds[vapply(
  nonempty_shard_rds, function(path) {
    nrow(readRDS(path)$payload$resource_metrics) > 0L
  }, logical(1L)
)]
assert_true(
  length(nonempty_shard_rds) == 44L &&
    length(shadow_shard_paths[grepl("\\.rds$", shadow_shard_paths)]) == 64L,
  "fixture contains 64 complete shard pairs and 44 executed setup shards"
)
mutated_shard_rds <- nonempty_shard_rds[[1L]]
mutated_shard_summary <- sub(
  "\\.rds$", ".summary.json", mutated_shard_rds
)
forge_shadow_shard_payload <- function(mutate_payload) {
  envelope <- readRDS(mutated_shard_rds)
  envelope$payload <- mutate_payload(envelope$payload)
  hashes <- .fastkpc_full_cuda_phase3_payload_semantic_hashes(
    envelope$payload
  )
  envelope$payload_semantic_hashes <- as.list(hashes)
  envelope$payload_semantic_hash <-
    .fastkpc_full_cuda_phase3_payload_semantic_hash(hashes)
  saveRDS(envelope, mutated_shard_rds, version = 2, compress = FALSE)
  summary <- jsonlite::read_json(
    mutated_shard_summary, simplifyVector = FALSE
  )
  summary$payload_semantic_hashes <- as.list(hashes)
  summary$payload_semantic_hash <- envelope$payload_semantic_hash
  summary$rds_file_sha256 <- fastkpc_full_cuda_census_file_hash(
    mutated_shard_rds
  )
  .fastkpc_full_cuda_phase3_write_json_exact(
    summary, mutated_shard_summary
  )
}
source_payload_mutations <- list(
  missing_resource_column = function(payload) {
    payload$resource_metrics$cpu_fallback_count <- NULL
    payload
  },
  missing_zero_counter = function(payload) {
    payload$resource_metrics$unknown_fallback_count <- NULL
    payload
  },
  junk_stage = function(payload) {
    payload$stage_timing$stage[[1L]] <- "junk_stage"
    payload
  },
  fabricated_fallback = function(payload) {
    payload$resource_metrics$cpu_fallback_count[[1L]] <- 1L
    payload
  }
)
for (case_name in names(source_payload_mutations)) {
  mutate_payload <- source_payload_mutations[[case_name]]
  pair_snapshot <- snapshot_bytes(c(
    mutated_shard_rds, mutated_shard_summary
  ))
  forge_shadow_shard_payload(mutate_payload)
  error <- tryCatch({
    forged_merged <- fastkpc_full_cuda_phase3_merge_shards(
      output_dir = output_dir, kind = "full_shadow",
      setup_keys = setup_keys, target_rows = target_rows,
      identity = identity, route_config = route_config,
      scope = "iteration", shard_count = 64L
    )
    fastkpc_full_cuda_phase3_validate_shadow_payload(
      forged_merged$payload,
      expected_setup_keys = setup_keys,
      expected_target_rows = target_rows,
      expected_logical_tests = logical_rows,
      require_logical_authority = TRUE,
      expected_setup_rows = setup_authority
    )
    NULL
  }, error = function(error) error)
  restore_bytes(pair_snapshot)
  assert_true(
    inherits(error, "error"),
    paste(case_name, "authenticated merged payload must fail closed")
  )
}

expect_rejected(
  c(
    artifact_paths$resource_metrics_csv,
    artifact_paths$manifest_json, artifact_paths$summary_json
  ),
  function() {
    resources <- read.csv(
      artifact_paths$resource_metrics_csv,
      stringsAsFactors = FALSE, check.names = FALSE
    )
    resources <- resources[rev(seq_len(nrow(resources))), , drop = FALSE]
    write.csv(resources, artifact_paths$resource_metrics_csv,
              row.names = FALSE, na = "NA", quote = TRUE)
    manifest <- jsonlite::read_json(
      artifact_paths$manifest_json, simplifyVector = FALSE
    )
    manifest$file_sha256$resource_metrics_csv <-
      fastkpc_full_cuda_census_file_hash(
        artifact_paths$resource_metrics_csv
      )
    .fastkpc_full_cuda_phase3_write_json_exact(
      manifest, artifact_paths$manifest_json
    )
    summary <- jsonlite::read_json(
      artifact_paths$summary_json, simplifyVector = FALSE
    )
    summary$manifest_sha256 <- fastkpc_full_cuda_census_file_hash(
      artifact_paths$manifest_json
    )
    .fastkpc_full_cuda_phase3_write_json_exact(
      summary, artifact_paths$summary_json
    )
  }, "reordered published resource evidence"
)

expect_graph_derivative_rejected <- function(key, path, mutate, label) {
  snapshot <- snapshot_bytes(c(
    path, artifact_paths$manifest_json, artifact_paths$summary_json
  ))
  on.exit(restore_bytes(snapshot), add = TRUE)
  mutate()
  manifest <- jsonlite::read_json(
    artifact_paths$manifest_json, simplifyVector = FALSE
  )
  manifest$file_sha256[[key]] <- fastkpc_full_cuda_census_file_hash(path)
  .fastkpc_full_cuda_phase3_write_json_exact(
    manifest, artifact_paths$manifest_json
  )
  refresh_shadow_summary_manifest_hash()
  forged_manifest <- jsonlite::read_json(
    artifact_paths$manifest_json, simplifyVector = TRUE
  )
  forged_summary <- jsonlite::read_json(
    artifact_paths$summary_json, simplifyVector = TRUE
  )
  assert_true(
    identical(
      unname(forged_manifest$file_sha256[[key]]),
      fastkpc_full_cuda_census_file_hash(path)
    ) && identical(
      forged_summary$manifest_sha256,
      fastkpc_full_cuda_census_file_hash(artifact_paths$manifest_json)
    ),
    paste(label, "refreshes payload, manifest, and summary hash linkage")
  )
  error <- tryCatch({
    .fastkpc_full_cuda_phase3_validate_shadow_graph_derivatives(
      artifact_paths, matching_public_validation$recomputed_graph,
      matching_public_validation$recomputed_first_divergence
    )
    NULL
  }, error = function(error) error)
  assert_true(
    inherits(error, "error"),
    paste(label, "must fail independent graph semantic validation")
  )
  restore_bytes(snapshot)
  on.exit(NULL, add = FALSE)
}

expect_graph_derivative_rejected(
  "adjacency_rds", artifact_paths$adjacency_rds, function() {
  adjacency <- readRDS(artifact_paths$adjacency_rds)
  adjacency[1L, 2L] <- !adjacency[1L, 2L]
  adjacency[2L, 1L] <- adjacency[1L, 2L]
  saveRDS(adjacency, artifact_paths$adjacency_rds,
          version = 2, compress = FALSE)
}, "forged adjacency")

for (case in list(
  list(key = "deletion_trace_csv", path = artifact_paths$deletion_trace_csv,
       label = "forged deletion trace"),
  list(key = "sepset_agreement_csv", path = artifact_paths$sepset_agreement_csv,
       label = "forged sepset agreement"),
  list(key = "n_edgetests_csv", path = artifact_paths$n_edgetests_csv,
       label = "forged n.edgetests")
)) {
  expect_graph_derivative_rejected(case$key, case$path, function() {
    writeLines("forged", case$path, useBytes = TRUE)
  }, case$label)
}

expect_graph_derivative_rejected(
  "first_divergence_json", artifact_paths$first_divergence_json, function() {
  first <- jsonlite::read_json(
    artifact_paths$first_divergence_json, simplifyVector = FALSE
  )
  first$first_divergence_found <- TRUE
  first$type <- "adjacency"
  first$message <- "forged"
  .fastkpc_full_cuda_phase3_write_json_exact(
    first, artifact_paths$first_divergence_json
  )
}, "forged first divergence")

expect_rejected(artifact_paths$summary_json, function() {
  summary <- jsonlite::read_json(
    artifact_paths$summary_json, simplifyVector = FALSE
  )
  summary$pass <- TRUE
  summary$logical_test_count <- 240489L
  summary$candidate_graph_gate <- TRUE
  .fastkpc_full_cuda_phase3_write_json_exact(
    summary, artifact_paths$summary_json
  )
}, "forged pass and summary counters")

expect_rejected(
  c(oracle_paths$manifest_json, oracle_paths$summary_json),
  function() {
    manifest <- jsonlite::read_json(
      oracle_paths$manifest_json, simplifyVector = FALSE
    )
    manifest$fixture_nonce <- "wrong-oracle-sp"
    .fastkpc_full_cuda_phase3_write_json_exact(
      manifest, oracle_paths$manifest_json
    )
    summary <- jsonlite::read_json(
      oracle_paths$summary_json, simplifyVector = FALSE
    )
    summary$manifest_sha256 <- fastkpc_full_cuda_census_file_hash(
      oracle_paths$manifest_json
    )
    .fastkpc_full_cuda_phase3_write_json_exact(
      summary, oracle_paths$summary_json
    )
  }, "wrong oracle-sp manifest hash"
)

expect_oracle_rejected <- function(paths, mutate, label) {
  snapshot <- snapshot_bytes(paths)
  on.exit(restore_bytes(snapshot), add = TRUE)
  mutate()
  error <- tryCatch({
    fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
      oracle_sp_dir, expected_identity = identity, require_full = FALSE
    )
    NULL
  }, error = function(error) error)
  assert_true(inherits(error, "error"), paste(label, "must fail closed"))
  restore_bytes(snapshot)
  on.exit(NULL, add = FALSE)
  invisible(error)
}

expect_oracle_rejected(oracle_paths$summary_json, function() {
  summary <- jsonlite::read_json(
    oracle_paths$summary_json, simplifyVector = FALSE
  )
  summary$pass <- TRUE
  summary$target_count <- 0L
  summary$manifest_sha256 <- fastkpc_full_cuda_census_file_hash(
    oracle_paths$manifest_json
  )
  .fastkpc_full_cuda_phase3_write_json_exact(
    summary, oracle_paths$summary_json
  )
}, "forged oracle summary pass/hash/counters")

expect_oracle_rejected(
  c(
    oracle_paths$target_parity_rds, oracle_paths$manifest_json,
    oracle_paths$summary_json
  ),
  function() {
    targets <- readRDS(oracle_paths$target_parity_rds)
    targets$planned_route[[1L]] <- if (
      targets$planned_route[[1L]] == "AUGMENTED_SVD"
    ) "AUGMENTED_QR" else "AUGMENTED_SVD"
    saveRDS(targets, oracle_paths$target_parity_rds,
            version = 2, compress = FALSE)
    manifest <- jsonlite::read_json(
      oracle_paths$manifest_json, simplifyVector = FALSE
    )
    manifest$payload_file_sha256[["target_parity.rds"]] <-
      fastkpc_full_cuda_census_file_hash(oracle_paths$target_parity_rds)
    .fastkpc_full_cuda_phase3_write_json_exact(
      manifest, oracle_paths$manifest_json
    )
    oracle_summary <- jsonlite::read_json(
      oracle_paths$summary_json, simplifyVector = FALSE
    )
    oracle_summary$manifest_sha256 <- fastkpc_full_cuda_census_file_hash(
      oracle_paths$manifest_json
    )
    .fastkpc_full_cuda_phase3_write_json_exact(
      oracle_summary, oracle_paths$summary_json
    )
  }, "forged oracle target route payload with refreshed hashes"
)

for (linkage_case in c("native", "embedded_native", "route")) {
  force(linkage_case)
  expect_oracle_rejected(
    c(
      oracle_paths$manifest_json, oracle_paths$summary_json
    ),
    function() {
      manifest <- jsonlite::read_json(
        oracle_paths$manifest_json, simplifyVector = FALSE
      )
      if (identical(linkage_case, "native")) {
        manifest$executed_native_library_sha256 <- strrep("0", 64L)
      } else if (identical(linkage_case, "embedded_native")) {
        manifest$input_identity$native_library_sha256 <- strrep("0", 64L)
      } else {
        manifest$input_identity$route_config_hash <- strrep("0", 64L)
      }
      .fastkpc_full_cuda_phase3_write_json_exact(
        manifest, oracle_paths$manifest_json
      )
      summary <- jsonlite::read_json(
        oracle_paths$summary_json, simplifyVector = FALSE
      )
      summary$manifest_sha256 <- fastkpc_full_cuda_census_file_hash(
        oracle_paths$manifest_json
      )
      .fastkpc_full_cuda_phase3_write_json_exact(
        summary, oracle_paths$summary_json
      )
    }, paste("forged oracle", linkage_case, "linkage")
  )
}

expect_rejected(
  c(artifact_paths$manifest_json, artifact_paths$summary_json),
  function() {
    manifest <- jsonlite::read_json(
      artifact_paths$manifest_json, simplifyVector = FALSE
    )
    manifest$source_inputs$input_identity$native_library_sha256 <-
      strrep("0", 64L)
    .fastkpc_full_cuda_phase3_write_json_exact(
      manifest, artifact_paths$manifest_json
    )
    refresh_shadow_summary_manifest_hash()
  }, "refreshed embedded shadow native identity"
)

expect_rejected(
  c(artifact_paths$manifest_json, artifact_paths$summary_json),
  function() {
    manifest <- jsonlite::read_json(
      artifact_paths$manifest_json, simplifyVector = FALSE
    )
    manifest$source_inputs$catalog_authority$phase1_dir <-
      paste0(manifest$source_inputs$catalog_authority$phase1_dir, "-moved")
    .fastkpc_full_cuda_phase3_write_json_exact(
      manifest, artifact_paths$manifest_json
    )
    summary <- jsonlite::read_json(
      artifact_paths$summary_json, simplifyVector = FALSE
    )
    summary$manifest_sha256 <- fastkpc_full_cuda_census_file_hash(
      artifact_paths$manifest_json
    )
    .fastkpc_full_cuda_phase3_write_json_exact(
      summary, artifact_paths$summary_json
    )
  }, "moved authenticated catalog input path"
)

expect_rejected(artifact_paths$failures_csv, function() {
  unlink(artifact_paths$failures_csv, force = TRUE)
}, "missing required output file")

.task9_phase0_replace_path <- file.path(phase0_dir, "manifest.json")
.task9_phase0_replace_once <- TRUE
phase0_manifest_before <- readBin(
  .task9_phase0_replace_path, "raw",
  n = file.info(.task9_phase0_replace_path)$size
)
phase0_replacement_snapshot_before <- c(
  create = .task9_heavy_probe$snapshot_create,
  release = .task9_heavy_probe$snapshot_release
)
invisible(trace(
  "fastkpc_full_cuda_compare_candidate_skeleton",
  tracer = quote({
    if (isTRUE(.task9_phase0_replace_once)) {
      .task9_phase0_replace_once <- FALSE
      writeLines(
        c(readLines(.task9_phase0_replace_path, warn = FALSE), " "),
        .task9_phase0_replace_path, useBytes = TRUE
      )
    }
  }), print = FALSE, where = globalenv()
))
phase0_replacement_error <- tryCatch({
  fastkpc_validate_full_cuda_fixed_sp_shadow_artifact(
    output_dir, require_full = FALSE
  )
  NULL
}, error = function(error) error)
phase0_replacement_snapshot_after <- c(
  create = .task9_heavy_probe$snapshot_create,
  release = .task9_heavy_probe$snapshot_release
)
untrace("fastkpc_full_cuda_compare_candidate_skeleton", where = globalenv())
phase0_manifest_connection <- file(.task9_phase0_replace_path, open = "wb")
writeBin(phase0_manifest_before, phase0_manifest_connection)
close(phase0_manifest_connection)
rm(.task9_phase0_replace_path, .task9_phase0_replace_once,
   envir = globalenv())
assert_true(
  inherits(phase0_replacement_error, "error") && grepl(
    "authority", conditionMessage(phase0_replacement_error),
    ignore.case = TRUE
  ) && identical(
    phase0_replacement_snapshot_after - phase0_replacement_snapshot_before,
    c(create = 1L, release = 1L)
  ),
  "Phase 0 replacement during graph replay fails post-use authority recheck"
)

assert_true(
  .task9_heavy_probe$snapshot_create ==
    .task9_heavy_probe$snapshot_release,
  paste(
    "standalone validation releases every minted execution snapshot across",
    "success, restart, and mutation paths"
  )
)
for (name in c(
  "fastkpc_full_cuda_phase3_merge_shards",
  "fastkpc_full_cuda_shadow_reconstruct_target_routes",
  "fastkpc_full_cuda_replay_logical_ci",
  "fastkpc_full_cuda_shadow_plan",
  ".fastkpc_full_cuda_phase3_shadow_scope_authority",
  "fastkpc_full_cuda_phase3_create_shadow_execution_snapshot",
  "fastkpc_full_cuda_phase3_release_shadow_execution_snapshot"
)) untrace(name, where = globalenv())
rm(.task9_heavy_probe, envir = globalenv())

cat("full CUDA CI fixed-sp shadow artifact: PASS\n")
