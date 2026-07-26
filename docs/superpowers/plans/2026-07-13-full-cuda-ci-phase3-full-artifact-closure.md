# Full-CUDA CI Phase 3 Artifact Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and independently validate the complete 110,617-target fixed-sp CUDA oracle artifact and the 240,489-test logical-CI shadow artifact, then close Phase 3 only after exact graph, sepset, deletion-trace, and `n.edgetests` gates pass.

**Architecture:** Extend the qualified Phase 3C runtime with two restartable artifact pipelines. The oracle-sp pipeline streams canonical PreparedSKey batches through one persistent CUDA context per execution session, compares every target with the Phase 2 `C_magic` oracle, and publishes compact 64-shard evidence. The shadow pipeline solves the same conditional target corpus again, consumes each setup's materialized residuals immediately with the pinned legacy C++ Spectra dCov comparator, merges those rows with an authenticated level-0 direct-CI payload, and replays candidate decisions in canonical logical order before using the existing full-CUDA graph comparator.

**Tech Stack:** R 4.4.1, mgcv 1.9.1 Phase 0/1/2 artifacts, Phase 3A/3B/3C C++17/CUDA runtime, Rcpp `.Call`, legacy C++ Spectra dCov, RDS/CSV/JSON SHA-256 artifacts, deterministic 64-shard resume, existing `full_cuda_ci_gate.R` graph comparator.

---

## Preconditions

Start only after every Phase 3C iteration and qualification command passes on
the declared GPU. Do not use a full run to debug runtime code.

The implementation branch must contain `main` at or after `37ce594`, the
Phase 3 review-amendment commit, and the accepted Phase 3A/3B/3C commits. The
pre-plan production-code baseline remains `42ef3ef` for provenance only.

Frozen canonical counts:

```text
PreparedSSetup groups                         = 8,634
TargetState rows                              = 110,617
smooth penalty root matrices                  = 28,527
smooth penalty root rows                      = 249,610
explicit constraints / non-null H             = 0 / 0
planned Cholesky / QR / SVD targets            = 73,158 / 4,210 / 33,249

logical CI tests                              = 240,489
direct level-0 tests                          = 2,213
conditional tests                             = 238,276
conditional residual-key pairs                = 238,276
canonical deletions                           = 1,018
near-alpha tests, all / conditional            = 1,529 / 1,478
n.edgetests by level                          =
  2,213 / 52,659 / 125,293 / 40,694 /
  13,293 / 5,422 / 835 / 80
```

Full closure uses the safe-reroute policy. The planned counts above remain
exact. Declared Cholesky-to-SVD or QR-to-SVD execution reroutes are allowed
only when every target records `planned_route`, `executed_route`,
`reroute_reason`, and `solver_status`, the conservation equations hold, all
numeric and decision gates pass, and SHD remains zero.

Required output directories:

```text
fastkpc/artifacts/full_cuda_ci/fixed_sp_cuda_oracle_sp_v1/
fastkpc/artifacts/full_cuda_ci/fixed_sp_cuda_full_shadow_v1/
```

Artifacts are generated evidence and are never staged through the shared
`fastkpc/artifacts` symlink.

## Files

Create:

```text
fastkpc/R/full_cuda_ci_phase3_artifacts.R
  Versioned artifact paths, route/environment identity, shard/session
  manifests, atomic publication, merge, and independent validators.

fastkpc/R/full_cuda_ci_fixed_sp_shadow.R
  Logical-test mapping, direct-CI execution, conditional residual consumption,
  canonical decision replay, and candidate skeleton construction.

fastkpc/tools/run_full_cuda_ci_fixed_sp_oracle_sp.R
fastkpc/tools/run_full_cuda_ci_fixed_sp_shadow.R

fastkpc/tests/test_full_cuda_ci_phase3_artifact_contract.R
fastkpc/tests/test_full_cuda_ci_phase3_shard_resume.R
fastkpc/tests/test_full_cuda_ci_fixed_sp_oracle_subset.R
fastkpc/tests/test_full_cuda_ci_fixed_sp_oracle_artifact.R
fastkpc/tests/test_full_cuda_ci_fixed_sp_shadow_replay.R
fastkpc/tests/test_full_cuda_ci_fixed_sp_shadow_mapping.R
fastkpc/tests/test_full_cuda_ci_fixed_sp_shadow_subset.R
fastkpc/tests/test_full_cuda_ci_fixed_sp_shadow_artifact.R
```

Modify:

```text
fastkpc/R/full_cuda_ci_fixed_sp_runtime.R
fastkpc/R/full_cuda_ci_gate.R
fastkpc/tests/test_full_cuda_ci_oracle_gate.R
docs/superpowers/specs/2026-07-13-full-cuda-ci-phase3-fixed-sp-runtime-design.md
goal-5.6.md
```

`full_cuda_ci_gate.R` changes are limited to a public candidate-skeleton
comparison helper needed by the new offline replayer. Do not modify Phase 0
oracle semantics.

## Spec Coverage for This Subplan

```text
artifact schemas and authenticated lineage                 Task 1
64-shard assignment, sessions, resume, atomic write        Task 2
oracle-sp target execution and small-subset parity         Task 3
oracle-sp merge, validator, qualification preflight        Task 4
oracle-sp full-run preflight; final run after source freeze Task 5 / Task 10
canonical logical decision/graph replayer                  Task 6
direct-CI and conditional setup mapping                    Task 7
conditional shadow shard execution and subset gate         Task 8
shadow merge, graph validator, restart/publication         Task 9
full 240,489-test shadow and graph artifact                Task 10
clean verification, Phase 3 exit, roadmap record           Task 11
```

## Task 1: Freeze Phase 3 Artifact Schemas and Lineage

**Files:**
- Create: `fastkpc/R/full_cuda_ci_phase3_artifacts.R`
- Create: `fastkpc/tests/test_full_cuda_ci_phase3_artifact_contract.R`
- Modify: `fastkpc/R/full_cuda_ci_fixed_sp_runtime.R`

- [ ] **Step 1: Write failing artifact-path/schema assertions**

Create a non-CUDA test that sources the Phase 0/1/2/runtime files and requires:

```r
oracle_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  tempfile("phase3-oracle-"), kind = "oracle_sp"
)
shadow_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  tempfile("phase3-shadow-"), kind = "full_shadow"
)

assert_true(identical(
  fastkpc_full_cuda_phase3_oracle_schema_version(),
  "full-cuda-ci-fixed-sp-oracle-sp-artifact-v1"
), "oracle artifact schema")
assert_true(identical(
  fastkpc_full_cuda_phase3_shadow_schema_version(),
  "full-cuda-ci-fixed-sp-shadow-artifact-v1"
), "shadow artifact schema")
assert_true(all(c(
  "manifest_json", "summary_json", "commands_txt", "environment_txt",
  "input_hashes_csv", "route_config_json", "runtime_lifecycle_csv",
  "resource_metrics_csv", "stage_timing_csv", "fallbacks_csv",
  "failures_csv", "shards_dir", "sessions_dir"
) %in% names(oracle_paths)), "common Phase 3 paths")
assert_true(all(c(
  "setup_results_csv", "setup_results_rds", "target_parity_csv",
  "target_parity_rds", "risk_cases_csv", "risk_cases_rds",
  "qualification_dcov_csv", "qualification_dcov_rds"
) %in% names(oracle_paths)), "oracle payload paths")
assert_true(all(c(
  "logical_ci_parity_csv", "logical_ci_parity_rds",
  "deletion_trace_csv", "sepset_agreement_csv", "n_edgetests_csv",
  "adjacency_rds", "first_divergence_json", "direct_ci_rds",
  "direct_ci_summary_json"
) %in% names(shadow_paths)), "shadow payload paths")
```

- [ ] **Step 2: Run and verify missing schema functions**

```bash
Rscript fastkpc/tests/test_full_cuda_ci_phase3_artifact_contract.R
```

Expected: FAIL because `full_cuda_ci_phase3_artifacts.R` and its schema/path
functions do not exist.

- [ ] **Step 3: Implement fixed route and environment identity**

Add:

```r
fastkpc_full_cuda_phase3_route_config <- function() {
  list(
    schema_version = "full-cuda-ci-fixed-sp-route-v1",
    condition_lt_1e8 = "CHOLESKY_BATCHED",
    condition_1e8_to_lt_1e12 = "AUGMENTED_QR",
    condition_ge_1e12 = "AUGMENTED_SVD",
    rank_deficient = "AUGMENTED_SVD",
    unauthenticated = "AUGMENTED_SVD",
    svd_rank_tolerance =
      "max(augmented_rows,null_dim)*sigma_max*double_epsilon",
    residual_tolerance = 1e-7,
    fitted_tolerance = 1e-7,
    qualification_dcov_p_tolerance = 1e-10,
    dcov_backend = "legacy-cpp-spectra",
    reroute_policy = "declared-cuda-svd-reroute-with-conservation",
    cpu_fallback_allowed = FALSE,
    approximate_backend_allowed = FALSE,
    shard_count = 64L
  )
}
```

Build `fastkpc_full_cuda_phase3_input_identity(catalog, device_id)` from the
already validated Phase 0/1/2 manifest hashes, dataset/corpus hashes, Phase 3
runtime ABI, route-config hash, source commit, R/mgcv versions, CUDA toolkit,
driver, GPU name/UUID/compute capability/SM count, device id, queried
cuSOLVER deterministic mode, cuBLAS math/atomics modes, and user-workspace
size/alignment. Missing or mismatched execution identity fails before a shard
can be reused or written.

- [ ] **Step 4: Implement artifact paths and required payload sets**

Define separate path constructors for `oracle_sp` and `full_shadow`. The
required payload lists exactly match the Phase 3 design; only `shards/` and
`sessions/` may exist before top-level publication. Treat `summary.json` as
the completion marker and require `manifest.json` whenever it exists.

- [ ] **Step 5: Add fail-closed manifest validation tests**

In the test, mutate each of these fields independently and require an error:

```text
Phase 0/1/2 manifest hash
dataset/canonical setup/canonical target corpus hash
route-config hash
runtime ABI
source commit
CUDA toolkit/driver/GPU UUID/compute capability
deterministic/math/atomics modes or cuBLAS workspace identity
artifact schema version
shard count
```

Also write a forged `summary.json` with `pass=true` but no payload and require
the validator to reject it.

- [ ] **Step 6: Run the contract test**

```bash
Rscript fastkpc/tests/test_full_cuda_ci_phase3_artifact_contract.R
```

Expected: PASS without initializing CUDA.

- [ ] **Step 7: Commit artifact contracts**

```bash
git add fastkpc/R/full_cuda_ci_phase3_artifacts.R \
  fastkpc/R/full_cuda_ci_fixed_sp_runtime.R \
  fastkpc/tests/test_full_cuda_ci_phase3_artifact_contract.R
git commit -m "feat: define full CUDA CI Phase 3 artifact contracts"
```

## Task 2: Implement Deterministic Shards, Sessions, and Resume

**Files:**
- Modify: `fastkpc/R/full_cuda_ci_phase3_artifacts.R`
- Create: `fastkpc/tests/test_full_cuda_ci_phase3_shard_resume.R`

- [ ] **Step 1: Write failing deterministic-assignment tests**

Load the authenticated 8,634-key setup index and require:

```r
assigned <- fastkpc_full_cuda_phase3_assign_setup_shards(setup_keys, 64L)
expected_order <- sort(unique(setup_keys), method = "radix")
assert_true(identical(assigned$prepared_s_key_sha256, expected_order),
            "radix setup order")
assert_true(identical(
  assigned$shard_id,
  as.integer((seq_along(expected_order) - 1L) %% 64L)
), "rank-modulo shard assignment")
assert_true(length(unique(assigned$prepared_s_key_sha256)) == 8634L,
            "one row per canonical setup")
```

Join all TargetState rows and require every target inherits exactly one setup
shard and the merged target-key hash equals the Phase 2 canonical target hash.

Production `scope = full` hardcodes `64` and rejects every shard-count
override. Tests and non-full iteration/qualification dry runs may pass an
explicit function argument or
`FASTKPC_FULL_CUDA_PHASE3_TEST_SHARD_COUNT`; the runner must reject that
variable when `scope = full`.

- [ ] **Step 2: Run and verify missing shard planner**

```bash
Rscript fastkpc/tests/test_full_cuda_ci_phase3_shard_resume.R
```

Expected: FAIL on the missing assignment/session functions.

- [ ] **Step 3: Implement session manifests**

Use one execution session for all missing shards in a runner invocation:

```r
session <- list(
  schema_version = "full-cuda-ci-phase3-session-v1",
  session_id = fastkpc_full_cuda_phase3_session_id(),
  input_identity_hash = identity$sha256,
  route_config_hash = route$sha256,
  requested_shard_ids = as.integer(missing_shards),
  completed_shard_ids = integer(),
  runtime_context_create_count = 0L,
  runtime_context_destroy_count = 0L,
  prepared_handle_create_count = 0L,
  prepared_handle_destroy_count = 0L,
  residual_token_release_count = 0L,
  output_slot_acquire_count = 0L,
  output_slot_release_count = 0L,
  status = "running"
)
```

Write `sessions/session_<id>.json` atomically after every completed shard.
Before marking a session `complete`, release every token/handle, destroy the
single context, require create/destroy `1/1`, require output-slot
acquire/release equality, and write the final session JSON.
A pure resume with no missing shards creates no session and no CUDA context.

- [ ] **Step 4: Implement shard manifests and atomic pairs**

Each `shards/shard_<id>.rds` plus `.summary.json` pair binds:

```text
artifact kind and schema
session id and immutable session_identity_hash
input identity and route-config hashes
expected setup keys/count/hash
expected target keys/count/hash
source commit and GPU environment
payload semantic hashes
```

Write the RDS temporary first, validate it in memory and from disk, rename it,
then rename the summary JSON last. A summary without RDS, RDS without summary,
duplicate shard id, unexpected shard id, payload mismatch, or incomplete
session reference is not reusable. On reuse, load the referenced session JSON,
require `status = complete`, and require its recomputed immutable identity hash
to equal the shard field; mutable completed-shard/counter fields are not part
of that identity hash.

- [ ] **Step 5: Add graceful-stop and hostile-resume tests**

Use a synthetic eight-setup corpus, four shards, and a fake executor. Test:

```text
first run stops cleanly after two shards
session closes with context counters 1/1
resume reuses two and writes two
pure resume writes zero and creates zero contexts
wrong route/GPU/corpus identity rejects reuse
corrupt RDS or summary rejects reuse
shard from status=running session is recomputed
missing or duplicate shard fails merge
repeated merge is byte-identical for merged RDS payloads
```

- [ ] **Step 6: Run shard/resume tests**

```bash
Rscript fastkpc/tests/test_full_cuda_ci_phase3_shard_resume.R
```

Expected: PASS without CUDA.

- [ ] **Step 7: Commit restartable sharding**

```bash
git add fastkpc/R/full_cuda_ci_phase3_artifacts.R \
  fastkpc/tests/test_full_cuda_ci_phase3_shard_resume.R
git commit -m "feat: add restartable full CUDA CI Phase 3 shards"
```

## Task 3: Execute Oracle-SP Shards on a Real Subset

**Files:**
- Modify: `fastkpc/R/full_cuda_ci_fixed_sp_runtime.R`
- Modify: `fastkpc/R/full_cuda_ci_phase3_artifacts.R`
- Create: `fastkpc/tools/run_full_cuda_ci_fixed_sp_oracle_sp.R`
- Create: `fastkpc/tests/test_full_cuda_ci_fixed_sp_oracle_subset.R`

- [ ] **Step 1: Write a failing two-setup CUDA subset test**

Select two iteration setups so the union contains Cholesky, QR, and SVD
targets. Execute through one reserved runtime and require one shard payload
with:

```r
assert_true(nrow(payload$setup_results) == 2L, "two setup rows")
assert_true(nrow(payload$target_parity) == expected_target_count,
            "all selected target rows")
assert_true(all(payload$target_parity$solver_status %in% c(
  "OK_CHOLESKY_BATCHED", "OK_CHOLESKY_SINGLE",
  "OK_AUGMENTED_QR", "OK_AUGMENTED_SVD"
)), "all subset targets OK")
assert_true(max(payload$target_parity$residual_max_abs_diff) < 1e-7,
            "subset residual max-abs parity")
assert_true(max(payload$target_parity$residual_relative_l2) < 1e-7,
            "subset residual relative-L2 parity")
assert_true(max(payload$target_parity$fitted_max_abs_diff) < 1e-7,
            "subset fitted max-abs parity")
assert_true(max(payload$target_parity$fitted_relative_l2) < 1e-7,
            "subset fitted relative-L2 parity")
assert_true(all(payload$target_parity$output_all_finite),
            "subset finite outputs")
assert_true(sum(payload$fallbacks$count) == 0L, "subset no fallback")
```

- [ ] **Step 2: Run and verify missing oracle shard executor**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_oracle_subset.R
```

Expected: FAIL on missing `fastkpc_full_cuda_phase3_run_oracle_shard`.

- [ ] **Step 3: Implement one-setup oracle execution**

For each setup in shard/radix order:

```text
load and authenticate only the Phase 2 shard containing the setup
materialize the complete canonical target batch
create one PreparedSGpuHandle
solve the complete batch once
explicitly materialize shadow coefficients/fitted/residual/RSS
run the Phase 2 C_magic adapter for each target
compute compact numeric errors and hashes
release token and prepared handle before the next setup
```

The target row includes at least:

```text
residual_key_sha256
prepared_s_key_sha256
target and canonical target rank
condition value/bucket and coefficient_rank
planned_route, executed_route, reroute_reason, solver_status
true-batched fields and Cholesky-to-SVD / QR-to-SVD counters
GPU augmented effective rank and Phase 1 coefficient rank
coefficient/fitted/residual/RSS oracle and candidate hashes
max-absolute and relative-L2 errors
finite flags, solver info, fallback/error fields
```

No residual/fitted vector is stored in the shard.

- [ ] **Step 4: Add setup and resource rows**

Record setup upload/root counts, dimensions, target counts, planned/executed
route counts and declared reroutes,
root-rank mismatches, post-warm-up allocations, handle creates, event/checkpoint
counts, implicit/shadow D2H bytes, and stage timings. Recompute shard summaries
from these rows; never accept a caller-provided `pass` flag.

- [ ] **Step 5: Run the two-setup test, then the 44-setup iteration corpus**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_oracle_subset.R
FASTKPC_RUN_CUDA_TESTS=1 \
FASTKPC_FULL_CUDA_PHASE3_SCOPE=iteration \
FASTKPC_FULL_CUDA_PHASE3_TEST_SHARD_COUNT=4 \
FASTKPC_FULL_CUDA_PHASE3_OUTPUT=/tmp/fastkpc_phase3_oracle_iteration \
  Rscript fastkpc/tools/run_full_cuda_ci_fixed_sp_oracle_sp.R
```

Expected: subset PASS; iteration solves `270/270`, planned and executed route
counts are both `172/31/67`, declared reroutes are `0/0`, and fallback/non-OK
rows are zero.

- [ ] **Step 6: Commit the oracle shard executor**

```bash
git add fastkpc/R/full_cuda_ci_fixed_sp_runtime.R \
  fastkpc/R/full_cuda_ci_phase3_artifacts.R \
  fastkpc/tools/run_full_cuda_ci_fixed_sp_oracle_sp.R \
  fastkpc/tests/test_full_cuda_ci_fixed_sp_oracle_subset.R
git commit -m "feat: execute fixed-sp CUDA oracle shards"
```

## Task 4: Merge, Publish, and Validate the Oracle-SP Artifact

**Files:**
- Modify: `fastkpc/R/full_cuda_ci_phase3_artifacts.R`
- Modify: `fastkpc/tools/run_full_cuda_ci_fixed_sp_oracle_sp.R`
- Create: `fastkpc/tests/test_full_cuda_ci_fixed_sp_oracle_artifact.R`

- [ ] **Step 1: Write a failing artifact merge/validator test**

Use the iteration corpus with four test shards and require the published
artifact to contain every oracle-sp payload named in the design. Independently
reload shard pairs and recompute:

```r
validated <- fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact(
  output_dir, require_full = FALSE
)
assert_true(validated$setup_count == 44L, "iteration setup merge")
assert_true(validated$target_count == 270L, "iteration target merge")
assert_true(validated$non_ok_solver_status_count == 0L,
            "iteration solver-status gate")
assert_true(identical(validated$planned_route_counts,
                      c(172L, 31L, 67L)),
            "iteration planned route counts")
assert_true(identical(validated$executed_route_counts,
                      c(172L, 31L, 67L)) &&
              validated$cholesky_to_svd_count == 0L &&
              validated$qr_to_svd_count == 0L,
            "iteration executed route conservation")
assert_true(validated$cpu_fallback_count == 0L &&
              validated$unknown_fallback_count == 0L &&
              validated$approximate_backend_count == 0L,
            "iteration fallback gate")
assert_true(validated$summary_recomputed, "summary not trusted")
```

Mutate one target row after forging `summary.json$pass = true`; validation must
still fail on the shard semantic hash/numeric gate.

- [ ] **Step 2: Run and verify missing merge validator**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_oracle_artifact.R
```

Expected: FAIL on missing oracle merge/publication functions.

- [ ] **Step 3: Merge compact oracle evidence**

Read every expected shard through its validator, reject duplicate setup/target
keys, and sort:

```text
setup_results: prepared_s_key_sha256 radix order
target_parity: canonical target-key radix order
qualification_dcov_parity: logical_sequence_id numeric order
```

Build `risk_cases` by joining Phase 1 risk rows to target parity. Preserve all
high-condition, rank-deficient, nonfinite-metadata, near-constant, warning,
nonconverged, and near-alpha selectors even when they overlap.

The validator recomputes planned counts from authenticated condition metadata,
recomputes executed counts from per-target rows, requires every changed route
to have the declared Cholesky-pivot or QR-rank-guard reason, and applies the
three route-conservation equations. It must not require full-run reroute counts
to be zero; numerical, decision, and graph gates determine whether a declared
CUDA stability reroute is acceptable.

- [ ] **Step 4: Compute qualification dCov evidence during merge input runs**

The full oracle run includes all 6,143 qualification keys. While executing a
setup that owns qualification logical tests, retain only that setup's explicit
shadow residual matrix, run all associated qualification pairs through the
pinned legacy C++ Spectra backend, append compact parity rows, then release the
matrix before the next setup. Require:

```text
qualification dCov rows              = 3,808
conditional near-alpha rows          = 1,478
max absolute p-value difference      < 1e-10
decision flips                       = 0
backend errors / Spectra fallbacks   = 0 / 0
```

- [ ] **Step 5: Publish top-level payload atomically**

At runner start, if both top-level markers exist, first run the complete
validator and return unchanged on an exact pure resume. Otherwise remove any
stale/partial `summary.json` or `manifest.json` while retaining valid
shard/session pairs. After all shards validate:

```text
write merged RDS/CSV and resource/timing/fallback/failure payloads to temps
read them back and validate counts/hashes
rename payloads into place
write manifest.json penultimate; it hashes every payload except summary.json
write summary.json last; it binds the completed manifest SHA-256
run the completed-artifact validator once more from disk
```

A complete exact artifact may return immediately on pure resume after full
validation and must create zero CUDA contexts.

- [ ] **Step 6: Run iteration publication, corruption, and resume tests**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_oracle_artifact.R
```

Expected: PASS for first publication, partial graceful resume, pure resume,
and every hostile corruption rejection.

- [ ] **Step 7: Re-run the Phase 3C qualification gate before full scale**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3c_qualification.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3c_dcov.R
```

Expected: `6,143/6,143` targets OK, planned/executed route counts both
`3,889/190/2,064`, declared reroutes `0/0`, `3,808` dCov rows, and zero
decision flips/fallbacks/errors.

- [ ] **Step 8: Commit oracle publication and validation**

```bash
git add fastkpc/R/full_cuda_ci_phase3_artifacts.R \
  fastkpc/tools/run_full_cuda_ci_fixed_sp_oracle_sp.R \
  fastkpc/tests/test_full_cuda_ci_fixed_sp_oracle_artifact.R
git commit -m "feat: publish authenticated fixed-sp CUDA oracle artifacts"
```

## Task 5: Generate the Full 110,617-Target Oracle-SP Artifact

**Files:**
- No source changes expected.
- Output: `fastkpc/artifacts/full_cuda_ci/fixed_sp_cuda_oracle_sp_v1/`

> **Execution-order amendment (2026-07-24):** the production identity binds
> both the current Git `HEAD` and the complete oracle execution-source
> closure. Tasks 6-9 necessarily change that closure, so an artifact produced
> here cannot pass the independent validator after those commits. Run the
> short gates in Step 1 now, but defer Steps 2-5 until Task 9 implementation
> and review are complete. Execute those steps immediately before Task 10's
> full shadow run, with no intervening source commit.
>
> A pre-freeze run at `7ee938f` completed and authenticated all 64 shard pairs
> but failed during top-level publication because the risk selector read
> `target_fit_metadata` instead of the authenticated full `target_risks`
> table. Preserve those shards as historical execution evidence. Do not adopt
> them under a different source identity; a truthful adoption would require a
> new dual execution/publication identity schema. The final artifact must be
> rerun under the frozen post-Task-9 source.

- [ ] **Step 1: Clean-build and run the short gates once more**

```bash
rm -f fastkpc/build/*.o fastkpc/build/fastkpc_cuda.so
bash fastkpc/tools/build_cuda_native.sh
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_cuda_native_reproducible_build.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3c_iteration.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3c_qualification.R
```

Expected: clean build, reproducible-build gate, and both numeric gates PASS.
Stop here on any failure.

- [ ] **Step 2: Run the full oracle-sp artifact**

```bash
FASTKPC_FULL_CUDA_PHASE3_DEVICE=0 \
FASTKPC_FULL_CUDA_PHASE3_PHASE0_DIR=fastkpc/artifacts/full_cuda_ci/oracle_351x48_v1 \
FASTKPC_FULL_CUDA_PHASE3_PHASE1_DIR=fastkpc/artifacts/full_cuda_ci/workload_census_351x48_v1 \
FASTKPC_FULL_CUDA_PHASE3_PHASE2_DIR=fastkpc/artifacts/full_cuda_ci/prepared_s_contract_v1 \
FASTKPC_FULL_CUDA_PHASE3_DATA=fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds \
FASTKPC_FULL_CUDA_PHASE3_OUTPUT=fastkpc/artifacts/full_cuda_ci/fixed_sp_cuda_oracle_sp_v1 \
FASTKPC_FULL_CUDA_PHASE3_SCOPE=full \
FASTKPC_FULL_CUDA_PHASE3_RESUME=1 \
  Rscript fastkpc/tools/run_full_cuda_ci_fixed_sp_oracle_sp.R
```

Do not set any tolerance, route threshold, fallback, or backend environment
override. The runner owns the frozen route config.

- [ ] **Step 3: Run the independent full validator**

```bash
Rscript -e 'source("fastkpc/R/full_cuda_ci_gate.R"); source("fastkpc/R/full_cuda_ci_oracle_contract.R"); source("fastkpc/R/full_cuda_ci_workload_census.R"); source("fastkpc/R/full_cuda_ci_prepared_s_contract.R"); source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R"); source("fastkpc/R/cuda_native.R"); source("fastkpc/R/full_cuda_ci_phase3_artifacts.R"); fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact("fastkpc/artifacts/full_cuda_ci/fixed_sp_cuda_oracle_sp_v1", require_full = TRUE)'
```

Expected hard gates:

```text
completed shard pairs                         = 64 / 64
PreparedSSetup / TargetState                  = 8,634 / 110,617
planned route counts                          = 73,158 / 4,210 / 33,249
executed Cholesky = 73,158 - cholesky_to_svd_count
executed QR       = 4,210 - qr_to_svd_count
executed SVD      = 33,249 + both reroute counts
reroute reason/status complete                = TRUE
smooth root matrices / rows                   = 28,527 / 249,610
setup uploads / prepared creates              = 8,634 / 8,634
output-slot acquires / releases                = equal
invalid-output initializations                 = 8,634
non-OK / nonfinite output rows                = 0 / 0
residual/fitted max-abs and relative-L2        < 1e-7
post-warm-up target allocations/handles       = 0 / 0
implicit residual D2H / cudaDeviceSynchronize = 0 / 0
target-level stable sync                      = 0
RHS authority / full CUDA data plane          = cuda-x0-transpose-y / TRUE
deterministic / pedantic / no-atomics modes   = enabled / TRUE / TRUE
unknown / CPU / approximate fallback          = 0 / 0 / 0
qualification dCov rows / decision flips      = 3,808 / 0
qualification backend errors / Spectra fallbacks = 0 / 0
```

For an uninterrupted run, execution sessions/context creates equal `1/1`.
For a resumed run, each accepted session must be complete with context
create/destroy `1/1`; aggregate context creates equal accepted session count.

- [ ] **Step 4: Run pure resume**

Run the command from Step 2 again. Expected: all 64 shards reused, zero CUDA
contexts created, merged semantic hashes unchanged, and validator PASS.

- [ ] **Step 5: Record the artifact hash for the shadow lineage**

Capture the completed oracle artifact manifest and summary SHA-256 in the
shadow runner's input identity. Do not edit `goal-5.6.md` yet; Phase 3 remains
open until the full logical-CI graph artifact passes.

## Task 6: Build a Canonical Offline Logical-CI Replayer

**Files:**
- Create: `fastkpc/R/full_cuda_ci_fixed_sp_shadow.R`
- Create: `fastkpc/tests/test_full_cuda_ci_fixed_sp_shadow_replay.R`
- Modify: `fastkpc/R/full_cuda_ci_gate.R`
- Modify: `fastkpc/tests/test_full_cuda_ci_oracle_gate.R`

- [ ] **Step 1: Write a failing replay-of-oracle test**

Load Phase 0 and Phase 1, pass Phase 1's `reference_p_value` column as the
candidate p-values, and require:

```r
replayed <- fastkpc_full_cuda_replay_logical_ci(
  logical_tests = phase1$logical_ci_tests,
  candidate_p_value = phase1$logical_ci_tests$reference_p_value,
  labels = colnames(canonical_data)
)
comparison <- fastkpc_full_cuda_compare_candidate_skeleton(
  oracle, replayed$skeleton
)
assert_true(nrow(replayed$logical_trace) == 240489L,
            "full replay logical count")
assert_true(sum(replayed$logical_trace$deletes_edge) == 1018L,
            "full replay deletion count")
assert_true(identical(replayed$skeleton$n.edgetests,
  c(2213L, 52659L, 125293L, 40694L, 13293L, 5422L, 835L, 80L)),
  "full replay n.edgetests")
assert_true(comparison$summary$edge_count_candidate == 110L,
            "oracle replay edge count")
assert_true(comparison$summary$SHD == 0L,
            "oracle replay SHD")
assert_true(comparison$summary$sepsets_identical &&
              comparison$summary$n_edgetests_identical &&
              comparison$summary$deletions_identical,
            "oracle replay graph evidence")
```

- [ ] **Step 2: Run and verify missing replayer**

```bash
Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_shadow_replay.R
```

Expected: FAIL on missing replay/comparison helpers.

- [ ] **Step 3: Implement the frozen decision convention**

For every row sorted by `logical_sequence_id`:

```r
candidate_independent <- is.finite(candidate_p_value) &
  candidate_p_value > alpha
candidate_decision <- ifelse(candidate_independent,
                             "independent", "dependent")
decision_flip <- candidate_decision != reference_decision
```

Reject nonfinite candidate p-values instead of converting them to dependent.
Do not infer the boundary convention from proximity to alpha; it is strict
`>` and must reproduce every Phase 1 reference decision when fed reference
p-values.

- [ ] **Step 4: Implement deterministic graph/sepset replay**

Start from the complete undirected 48-node graph. Traverse the frozen logical
rows in canonical order. When a candidate-independent row finds its edge still
adjacent:

```text
delete both adjacency directions
set deletes_edge = TRUE for that logical row
store sorted S in the tested x->y sepset cell
update symmetric pMax with the candidate p-value
```

Otherwise set `deletes_edge = FALSE`. Preserve `source_sequence_id`,
`source_task_index`, `x`, `y`, `S_key`, level, and candidate p-value in the
candidate task/logical trace. Derive `n.edgetests` by level from all frozen
rows, not from deletion count.

- [ ] **Step 5: Expose comparison without writing an artifact**

Refactor the existing comparator only enough to add:

```r
fastkpc_full_cuda_compare_candidate_skeleton <- function(oracle, candidate) {
  fastkpc_full_cuda_compare_core(
    reference = oracle$reference,
    candidate = candidate,
    reference_deletions = oracle$deletion_trace,
    reference_logical = oracle$logical_trace
  )
}
```

Keep `fastkpc_compare_full_cuda_ci_candidate()` behavior unchanged.

- [ ] **Step 6: Add mutation and ordering tests**

Flip one dependent reference p-value to `alpha * 2`, reorder two logical rows,
drop one row, and duplicate one `logical_sequence_id` in separate cases.
Require deterministic first-divergence evidence or fail-closed input errors.

- [ ] **Step 7: Run replay and existing oracle-gate tests**

```bash
Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_shadow_replay.R
Rscript fastkpc/tests/test_full_cuda_ci_oracle_gate.R
```

Expected: both PASS; reference replay independently reconstructs the Phase 0
graph evidence.

- [ ] **Step 8: Commit the canonical replayer**

```bash
git add fastkpc/R/full_cuda_ci_fixed_sp_shadow.R \
  fastkpc/R/full_cuda_ci_gate.R \
  fastkpc/tests/test_full_cuda_ci_fixed_sp_shadow_replay.R \
  fastkpc/tests/test_full_cuda_ci_oracle_gate.R
git commit -m "feat: replay full CUDA CI decisions from logical p-values"
```

## Task 7: Map Direct and Conditional Logical Tests to Execution Units

**Files:**
- Modify: `fastkpc/R/full_cuda_ci_fixed_sp_shadow.R`
- Modify: `fastkpc/R/full_cuda_ci_phase3_artifacts.R`
- Create: `fastkpc/tests/test_full_cuda_ci_fixed_sp_shadow_mapping.R`

- [ ] **Step 1: Write failing canonical mapping assertions**

Build a shadow work plan from authenticated Phase 1 logical rows and Phase 2
target/setup indexes. Require:

```r
plan <- fastkpc_full_cuda_shadow_plan(catalog)
assert_true(nrow(plan$direct_tests) == 2213L, "direct level-0 count")
assert_true(nrow(plan$conditional_tests) == 238276L,
            "conditional logical count")
assert_true(all(is.na(plan$direct_tests$residual_key_x)) &&
              all(is.na(plan$direct_tests$residual_key_y)),
            "direct tests have no residual keys")
assert_true(all(nzchar(plan$conditional_tests$residual_key_x)) &&
              all(nzchar(plan$conditional_tests$residual_key_y)),
            "conditional tests have two residual keys")
assert_true(all(plan$conditional_tests$prepared_s_key_x ==
                  plan$conditional_tests$prepared_s_key_y),
            "conditional endpoints share PreparedSKey")
assert_true(length(unique(plan$conditional_tests$prepared_s_key_x)) == 8634L,
            "all canonical setups consumed")
assert_true(identical(sort(c(plan$direct_tests$logical_sequence_id,
                             plan$conditional_tests$logical_sequence_id)),
                      seq_len(240489L)),
            "logical sequence coverage")
```

- [ ] **Step 2: Run and verify missing shadow planner**

```bash
Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_shadow_mapping.R
```

Expected: FAIL on missing `fastkpc_full_cuda_shadow_plan`.

- [ ] **Step 3: Join residual keys to setup shards fail-closed**

Join each conditional endpoint through the canonical TargetState index. Reject
missing/duplicate keys, endpoint keys mapped to different PreparedSKeys,
target/setup fingerprint conflict, direct rows carrying residual keys, or
conditional rows lacking either key. Assign each conditional logical row to
the setup's deterministic 64-shard id.

- [ ] **Step 4: Implement the level-0 direct-CI payload**

For the 2,213 direct rows, take raw canonical data columns `x` and `y`, call
the pinned legacy C++ Spectra dCov gamma backend, and write one atomic pair:

```text
direct_ci.rds
direct_ci.summary.json
```

Each row stores sequence/source ids, x/y, empty S, reference/candidate p-value,
absolute difference, reference/candidate decision, decision flip, backend
error, and Spectra fallback. The manifest binds Phase 0/1/data hashes and the
dCov route config. Require `2,213` rows, zero backend errors/fallbacks, and
strict decision-convention parity. This is a correctness comparator, not a
CPU residual fallback.

- [ ] **Step 5: Add hostile mapping/direct-payload tests**

Test one missing endpoint key, cross-setup pair, duplicate logical id, corrupt
direct payload hash, wrong data hash, and backend fallback row. Every case
must fail before shadow merge.

- [ ] **Step 6: Run mapping tests**

```bash
Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_shadow_mapping.R
```

Expected: PASS with exact `2,213 / 238,276 / 240,489` coverage.

- [ ] **Step 7: Commit shadow planning and direct CI**

```bash
git add fastkpc/R/full_cuda_ci_fixed_sp_shadow.R \
  fastkpc/R/full_cuda_ci_phase3_artifacts.R \
  fastkpc/tests/test_full_cuda_ci_fixed_sp_shadow_mapping.R
git commit -m "feat: map full CUDA CI shadow execution units"
```

## Task 8: Execute Conditional Shadow Shards on Subsets

**Files:**
- Modify: `fastkpc/R/full_cuda_ci_fixed_sp_shadow.R`
- Modify: `fastkpc/R/full_cuda_ci_phase3_artifacts.R`
- Create: `fastkpc/tools/run_full_cuda_ci_fixed_sp_shadow.R`
- Create: `fastkpc/tests/test_full_cuda_ci_fixed_sp_shadow_subset.R`

- [ ] **Step 1: Write a failing iteration-shadow CUDA test**

Use the authenticated 44-test Phase 2 iteration logical corpus. For each
selected setup, solve its complete selected target batch once, materialize
residuals explicitly, compute associated dCov rows, and require:

```r
assert_true(nrow(rows) == 44L, "iteration shadow logical rows")
assert_true(all(order(rows$logical_sequence_id) == seq_len(nrow(rows))),
            "iteration shadow canonical order")
assert_true(sum(rows$decision_flip) == 0L,
            "iteration shadow decision flips")
assert_true(sum(rows$backend_error) == 0L,
            "iteration shadow backend errors")
assert_true(sum(rows$spectra_fallback) == 0L,
            "iteration shadow Spectra fallbacks")
assert_true(runtime$implicit_residual_d2h_count == 0L,
            "iteration no implicit residual D2H")
assert_true(runtime$shadow_materialize_target_count == 270L,
            "iteration explicit shadow materialization")
```

- [ ] **Step 2: Run and verify missing conditional executor**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_shadow_subset.R
```

Expected: FAIL on missing conditional shadow shard execution.

- [ ] **Step 3: Implement setup-local residual consumption**

For each setup in a shadow shard:

```text
solve the complete canonical target batch once
explicitly materialize the residual matrix
build a residual_key_sha256 -> column index map
verify exact target-key coverage and no duplicate columns
compute every logical pair assigned to the setup with C++ Spectra
append compact logical rows
drop the residual matrix before releasing the token/handle
```

No residual vector crosses setup boundaries or enters the shard payload. Every
conditional pair must find both keys in the current matrix; lookup failure is
a shard failure, never a request for CPU mgcv residuals.

- [ ] **Step 4: Freeze conditional logical row fields**

Store:

```text
logical_sequence_id, source_sequence_id, source_task_index
level, x, y, S_key, residual_key_x, residual_key_y
prepared_s_key_sha256, shard_id
reference_p_value, candidate_p_value, absolute_p_value_difference
alpha, reference_decision, candidate_decision, decision_flip
near_alpha and near_alpha_bucket
dCov backend/version, backend_error, Spectra fallback
declared target routes/statuses for x and y
```

Candidate decision uses the Task 6 strict convention. The row does not derive
`deletes_edge`; that is a global canonical replay result after all rows merge.

- [ ] **Step 5: Run iteration then 3,808-pair qualification shadow**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_shadow_subset.R
FASTKPC_RUN_CUDA_TESTS=1 \
FASTKPC_FULL_CUDA_PHASE3_SCOPE=qualification \
FASTKPC_FULL_CUDA_PHASE3_OUTPUT=/tmp/fastkpc_phase3_shadow_qualification \
  Rscript fastkpc/tools/run_full_cuda_ci_fixed_sp_shadow.R
```

Expected: iteration `44/44` and qualification `3,808/3,808` rows, conditional
near-alpha `1,478`, zero flips/errors/fallbacks, and exact resource gates.

- [ ] **Step 6: Commit conditional shadow execution**

```bash
git add fastkpc/R/full_cuda_ci_fixed_sp_shadow.R \
  fastkpc/R/full_cuda_ci_phase3_artifacts.R \
  fastkpc/tools/run_full_cuda_ci_fixed_sp_shadow.R \
  fastkpc/tests/test_full_cuda_ci_fixed_sp_shadow_subset.R
git commit -m "feat: execute fixed-sp CUDA logical shadow shards"
```

## Task 9: Merge, Replay, Publish, and Validate the Shadow Artifact

**Files:**
- Modify: `fastkpc/R/full_cuda_ci_fixed_sp_shadow.R`
- Modify: `fastkpc/R/full_cuda_ci_phase3_artifacts.R`
- Modify: `fastkpc/tools/run_full_cuda_ci_fixed_sp_shadow.R`
- Create: `fastkpc/tests/test_full_cuda_ci_fixed_sp_shadow_artifact.R`

- [ ] **Step 1: Write a failing shadow merge/publication test**

Use a deterministic real subset plus direct rows selected by logical id.
Publish a non-full artifact and independently validate row/key/order coverage,
decision derivation, fallback evidence, and graph-replay output. Forge summary
booleans after mutating one candidate p-value; the validator must recompute the
flip and reject the artifact.

- [ ] **Step 2: Run and verify missing shadow publisher**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_shadow_artifact.R
```

Expected: FAIL on missing shadow merge/validator functions.

- [ ] **Step 3: Merge direct and conditional logical rows**

Validate the direct pair and all expected conditional shard pairs, concatenate,
sort by integer `logical_sequence_id`, and require a unique contiguous sequence.
For the full scope require:

```text
direct / conditional / total rows = 2,213 / 238,276 / 240,489
near-alpha all / conditional       = 1,529 / 1,478
backend errors / Spectra fallback = 0 / 0
decision flips                    = 0
```

Do not run graph comparison before row coverage and decision gates pass.
From the two endpoint planned/executed route, reroute-reason, and solver-status
columns, reconstruct one row per residual key and require every repeated
observation of a key to agree. Require the resulting 110,617-key corpus hash
and planned counts `73,158 / 4,210 / 33,249` to match the oracle-sp artifact,
then independently recompute both reroute counts and the executed-route
conservation equations. This proves the shadow consumed the complete target
corpus without storing a second target-parity table.

- [ ] **Step 4: Replay candidate graph and invoke the existing comparator**

Pass merged candidate p-values to Task 6's replayer, then call
`fastkpc_full_cuda_compare_candidate_skeleton()`. Write:

```text
logical_ci_parity.rds/.csv
deletion_trace.csv
sepset_agreement.csv
n_edgetests.csv
adjacency.rds
first_divergence.json
runtime_lifecycle.csv
resource_metrics.csv
stage_timing.csv
fallbacks.csv
failures.csv
```

The full validator recomputes graph comparison from merged logical rows and
does not trust saved adjacency, deletion trace, agreement CSVs, or summary.

- [ ] **Step 5: Add exact full graph gates**

Require:

```text
candidate_graph_gate             = TRUE
logical_test_count               = 240,489
edge_count                       = 110
oracle_edge_count                = 110
SHD                              = 0
normalized_sepsets_identical     = TRUE
n_edgetests_exact                = TRUE
deletion_trace_exact             = TRUE
decision_flip_count              = 0
first_divergence                 = NOT_APPLICABLE
```

Also require all runtime route/resource/fallback gates from the oracle-sp
artifact and exact linkage to its completed manifest hash.

- [ ] **Step 6: Test interruption, resume, and independent validation**

Cover graceful stop after conditional shards, direct-payload reuse, pure
resume, wrong oracle-sp artifact hash, incomplete session, corrupt logical
row, missing/duplicate sequence id, forged adjacency, forged deletion trace,
and forged `pass=true`. Incomplete-session shards must be recomputed.

- [ ] **Step 7: Run the shadow artifact tests**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_shadow_artifact.R
Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_shadow_replay.R
Rscript fastkpc/tests/test_full_cuda_ci_oracle_gate.R
```

Expected: all PASS.

- [ ] **Step 8: Commit shadow publication and validation**

```bash
git add fastkpc/R/full_cuda_ci_fixed_sp_shadow.R \
  fastkpc/R/full_cuda_ci_phase3_artifacts.R \
  fastkpc/tools/run_full_cuda_ci_fixed_sp_shadow.R \
  fastkpc/tests/test_full_cuda_ci_fixed_sp_shadow_artifact.R
git commit -m "feat: publish authenticated full CUDA CI shadow artifacts"
```

## Task 10: Generate the Final Oracle and Full 240,489-Test Shadow Artifacts

**Files:**
- No source changes expected.
- Output: `fastkpc/artifacts/full_cuda_ci/fixed_sp_cuda_full_shadow_v1/`

- [ ] **Step 1: Generate and revalidate the final oracle-sp prerequisite**

After Task 9 is committed and both reviews pass, run Task 5 Steps 2-5 under
that exact source identity. Do not make another source commit between the
oracle run, its pure-resume validation, and the full shadow run below.

If `fixed_sp_cuda_oracle_sp_v1` contains a completed artifact from an older
execution-source or native-build identity, preserve it under a uniquely named
historical directory before rerunning Task 5 Step 2. Do not ask resume to
adopt stale shards under the new identity. For the reproducible-native-build
repair discovered after the `ef2a116` run, use:

```bash
mv fastkpc/artifacts/full_cuda_ci/fixed_sp_cuda_oracle_sp_v1 \
  fastkpc/artifacts/full_cuda_ci/fixed_sp_cuda_oracle_sp_v1.pre_reproducible_native_ef2a116_20260727
mkdir -p fastkpc/artifacts/full_cuda_ci/fixed_sp_cuda_oracle_sp_v1
```

Commit the native-build repair before generating the replacement oracle.
After that commit, make no source change through oracle generation, oracle
validation and pure resume, full shadow generation, shadow validation, and
shadow pure resume.

```bash
Rscript -e 'source("fastkpc/R/full_cuda_ci_gate.R"); source("fastkpc/R/full_cuda_ci_oracle_contract.R"); source("fastkpc/R/full_cuda_ci_workload_census.R"); source("fastkpc/R/full_cuda_ci_prepared_s_contract.R"); source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R"); source("fastkpc/R/cuda_native.R"); source("fastkpc/R/full_cuda_ci_phase3_artifacts.R"); fastkpc_validate_full_cuda_fixed_sp_oracle_sp_artifact("fastkpc/artifacts/full_cuda_ci/fixed_sp_cuda_oracle_sp_v1", require_full = TRUE)'
```

Expected: PASS. The shadow runner refuses an incomplete or mismatched oracle-sp
artifact.

- [ ] **Step 2: Run the full shadow artifact**

```bash
FASTKPC_FULL_CUDA_PHASE3_DEVICE=0 \
FASTKPC_FULL_CUDA_PHASE3_PHASE0_DIR=fastkpc/artifacts/full_cuda_ci/oracle_351x48_v1 \
FASTKPC_FULL_CUDA_PHASE3_PHASE1_DIR=fastkpc/artifacts/full_cuda_ci/workload_census_351x48_v1 \
FASTKPC_FULL_CUDA_PHASE3_PHASE2_DIR=fastkpc/artifacts/full_cuda_ci/prepared_s_contract_v1 \
FASTKPC_FULL_CUDA_PHASE3_ORACLE_SP_DIR=fastkpc/artifacts/full_cuda_ci/fixed_sp_cuda_oracle_sp_v1 \
FASTKPC_FULL_CUDA_PHASE3_DATA=fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds \
FASTKPC_FULL_CUDA_PHASE3_OUTPUT=fastkpc/artifacts/full_cuda_ci/fixed_sp_cuda_full_shadow_v1 \
FASTKPC_FULL_CUDA_PHASE3_SCOPE=full \
FASTKPC_FULL_CUDA_PHASE3_RESUME=1 \
  Rscript fastkpc/tools/run_full_cuda_ci_fixed_sp_shadow.R
```

The runner pins C++ Spectra internally. Do not supply backend, tolerance,
decision, route, or fallback overrides.

- [ ] **Step 3: Run the independent full shadow validator**

```bash
Rscript -e 'source("fastkpc/R/full_cuda_ci_gate.R"); source("fastkpc/R/full_cuda_ci_oracle_contract.R"); source("fastkpc/R/full_cuda_ci_workload_census.R"); source("fastkpc/R/full_cuda_ci_prepared_s_contract.R"); source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R"); source("fastkpc/R/full_cuda_ci_fixed_sp_shadow.R"); source("fastkpc/R/cuda_native.R"); source("fastkpc/R/full_cuda_ci_phase3_artifacts.R"); fastkpc_validate_full_cuda_fixed_sp_shadow_artifact("fastkpc/artifacts/full_cuda_ci/fixed_sp_cuda_full_shadow_v1", require_full = TRUE)'
```

Expected: all Task 9 graph gates, exact route/resource gates, `64/64`
conditional shard pairs, one valid direct payload pair, and zero
fallback/error/nonfinite rows.

- [ ] **Step 4: Run pure resume and compare semantic hashes**

Run Step 2 again. Expected: direct payload and all 64 shards reused, zero CUDA
contexts, unchanged logical/adjacency/sepset/deletion semantic hashes, and
validator PASS.

- [ ] **Step 5: Preserve first-divergence evidence even on success**

Require `first_divergence.json` to contain the versioned empty-divergence
record with `first_divergence_found=false` and type/message
`NOT_APPLICABLE`; absence of the file is not accepted as success. Normalize
the successful full-shadow artifact record at publication time without
changing the existing Phase 0 comparator's generic NA-valued empty helper.

## Task 11: Verify Phase 3 and Record Closure

**Files:**
- Modify: `goal-5.6.md`
- Verify: `fastkpc/tests/test_cuda_native_reproducible_build.R`

- [ ] **Step 1: Clean-build CUDA**

```bash
rm -f fastkpc/build/*.o fastkpc/build/fastkpc_cuda.so
bash fastkpc/tools/build_cuda_native.sh
```

Expected: exit zero.

- [ ] **Step 2: Run non-CUDA contract/replay/resume tests**

```bash
Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_catalog.R
Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_native_dto.R
Rscript fastkpc/tests/test_full_cuda_ci_phase3_artifact_contract.R
Rscript fastkpc/tests/test_full_cuda_ci_phase3_shard_resume.R
Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_shadow_replay.R
Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_shadow_mapping.R
Rscript fastkpc/tests/test_full_cuda_ci_oracle_gate.R
```

Expected: all PASS.

- [ ] **Step 3: Run CUDA resource/numeric/subset tests**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_runtime_lifecycle.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_runtime_misuse.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_cuda_native_reproducible_build.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3c_iteration.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3c_qualification.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3c_dcov.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_oracle_subset.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_oracle_artifact.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_shadow_subset.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_shadow_artifact.R
```

Expected: all PASS.

- [ ] **Step 4: Validate both completed full artifacts from disk**

Run the exact validators from Tasks 5 and 10. Expected: both PASS with fresh
recomputation; do not rely on their saved summaries.

- [ ] **Step 5: Run existing CUDA regressions and hygiene**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_extract_gpu_native_fixed_sp.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_extract_gpu_native_batch_bridge.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_extract_gpu_same_setup_batch.R
git diff --check
git status --short
```

Expected: tests PASS, no whitespace errors, and the shared artifact symlink is
not staged.

- [ ] **Step 6: Update the active roadmap only after both validators pass**

Record exact artifact paths, manifest hashes, elapsed/resource summaries, and:

```text
Phase 3 complete:
  PreparedSSetup / TargetState solved = 8,634 / 110,617
  planned routes                      = 73,158 / 4,210 / 33,249
  executed routes                     = planned counts adjusted by validated reroutes
  Cholesky->SVD / QR->SVD reroutes    = exact artifact counts with reasons/statuses
  numeric tolerance failures          = 0
  full logical CI rows                = 240,489
  decision flips                      = 0
  edge_count                          = 110 / 110
  SHD                                 = 0
  normalized sepsets identical        = TRUE
  n.edgetests exact                   = TRUE
  deletion trace exact                = TRUE
  unknown / CPU / approximate fallback= 0 / 0 / 0

Active next task:
  Phase 4 CUDA smoothing-parameter selection using the accepted Phase 3
  persistent residual runtime.
```

- [ ] **Step 7: Commit the Phase 3 closure record**

```bash
git add goal-5.6.md
git commit -m "docs: record full CUDA CI Phase 3 closure"
```

- [ ] **Step 8: Request final code/spec/artifact review**

Review all Phase 3A/3B/3C/closure changes against the design, inspect both
artifact validators and manifest hashes, resolve every finding, and rerun
Tasks 11.1 through 11.5 before pushing the closure record.

## Exit Condition

Phase 3 is complete only after both independently validated full artifacts
exist and every Task 11 gate passes. Qualification-only evidence, inherited
Phase 0 graph evidence, a saved `pass=true`, or a full target solve without
the logical-CI replay cannot close the phase.
