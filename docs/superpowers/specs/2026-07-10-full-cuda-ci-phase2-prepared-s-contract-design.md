# Full-CUDA CI Phase 2 Prepared-S Contract Design

## Status

Amended after independent contract and corpus review. Phase 1 is complete at
artifact source commit
`1560068ba8d635e806612554e11bbed92c0b8843`. This design freezes the Phase 2
contract before implementation planning.

## Decision

Build an explicit, normalized, response-independent `PreparedSSetup` for all
8,634 canonical nonempty conditioning-set groups. Construct each setup with a
deterministic zero response, no selected smoothing parameter, and
`mgcv::gam(..., fit = FALSE)`; whitelist-project the raw object into plain
data and discard the raw mgcv object. Build a lightweight `TargetState` index
for all 110,617 canonical conditional `target|S` keys.

Use two deterministic numerical gates:

~~~text
iteration gate:
  PreparedSSetup groups = 44
  target|S keys         = 270
  logical dCov pairs    = 44

extended qualification:
  PreparedSSetup groups = 2,061
  target|S keys         = 6,143
  logical dCov pairs    = 3,808
~~~

The iteration gate is class-complete and intended for normal development. The
extended qualification remains the Phase 2 artifact gate. Both are derived
from the authenticated Phase 1 corpus, not maintained as hand-written case
lists.

Use the version-pinned mgcv `C_magic` fixed-sp kernel as the Phase 2 reference
adapter. Existing C++ normal-equation and CUDA Cholesky solvers remain shadow
implementations. Stable persistent CUDA solving is Phase 3 work.

Do not persist a stripped raw mgcv `G` object as the public contract. Do not
persist only a reconstruction recipe that requires a later `mgcv::gam` call.

## Why This Architecture

The completed census established:

```text
canonical conditional target|S keys = 110,617
same-S groups                        =   8,634
single-penalty groups                =   1,174
multi-penalty groups                 =   7,460
```

The corpus is dominated by multi-penalty additive setups:

```text
penalty count 1 = 1,174 groups
penalty count 3 = 4,064 groups
penalty count 4 = 2,152 groups
penalty count 5 =   955 groups
penalty count 6 =   245 groups
penalty count 7 =    44 groups
```

The current `fastkpc_mgcv_extract_retarget_setup()` replaces only top-level
`y`, `sp`, and `target`. Its copied `setup$G` still contains template-response
state and template `lsp0`. The current `fastkpc_prepare_gpu_setup_state()`
strips `G/y`, but rejects multi-penalty setups. Neither object is the final
Phase 2 contract.

The current stripping helper also leaves formula environments capable of
retaining the complete input data and template target. A public contract
therefore cannot retain formulas, calls, environments, closures, smooth
objects, fitted gam objects, or raw `G`.

Canonical exploratory checks showed that a zero-response, no-fixed-sp
`fit = FALSE` setup plus a new
target `y` and target-specific selected `sp` reproduces independently built
fixed-sp mgcv coefficient, fitted, and residual hashes exactly for penalty
counts 1, 3, 4, 5, 6, and 7, including a rank-deficient case. This makes an
explicit normalized contract both feasible and testable without weakening
numeric gates.

The provider `fit = FALSE` model matrix and Phase 1 fitted lpmatrix differ by
approximately `1e-10` in observed probes while defining the same model and
producing exact fixed-sp fitted/residual hashes. Phase 2 must therefore retain
both representation and semantic checks; it must not require those two raw
matrix hashes to be identical.

## Phase Boundary

Phase 2 owns:

- authenticated Phase 1 and dataset loading;
- one response-independent setup build per same-S group;
- the versioned `PreparedSSetup` and `TargetState` schemas;
- all-group structural and lineage validation;
- deterministic qualification-subset selection;
- fixed-sp mgcv reference retargeting without per-target `mgcv::gam` setup;
- exact residual, fitted, and coefficient parity plus the pinned dCov oracle
  tolerance and exact decision parity;
- restartable, hash-protected setup shards.

Phase 2 does not own:

- persistent CUDA streams, handles, or device buffers;
- a true fused multi-target CUDA kernel;
- a stable CUDA QR/SVD path;
- CUDA smoothing-parameter selection;
- production skeleton integration;
- replacement of mgcv basis construction.

Those are Phase 3 and later roadmap tasks.

## Input Contract

The standard runner accepts:

```text
fastkpc/artifacts/full_cuda_ci/workload_census_351x48_v1/
fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds
```

It must validate these exact file SHA-256 values before building or reusing a
shard:

```text
manifest.json
  b0990cfc932a5fcabc09ad25e352e7babb67fc8a127f11c7d2b88887c4940574
summary.json
  4f71d1bbbbdd2436e3576b728363120bb9b911897b9dee3ecf6f8a5d3379eb24
logical_ci_tests.rds
  17781421df868ae7822c022ac58ad6322292ad51053d5da686a3d6f79b40d7c8
residual_requests.rds
  d7a995f12f6bc118a39009b0b685cb5c28068d418c5a569d552cf26e8748ec8b
same_s_setup_metadata.rds
  8b35a463b17a64512d653da949f5ac74f7cc21223f346a304ac52fdfe8434a3f
target_fit_metadata.rds
  af09b5dc4c6a34d7ec126e1fe7f3f1f9c3d7fcb6316ada759a293abe76d8323c
risk_cases.rds
  1e0951e9856bea3c9a1b7ba83ec03b79a678e7aa60464d7f6808397ab8d9a7bc
canonical dataset RDS
  e03cbfafed3336f3da7725878e92af24556b5ebdc6227b94ea84469e62e94036
```

The loader must also require:

```text
phase1_complete                       = TRUE
metadata_schema_version               = full-cuda-ci-metadata-v4
canonical logical test count          = 240,489
conditional logical test count        = 238,276
conditional residual request count    = 476,552
canonical target|S key count          = 110,617
same-S setup count                     = 8,634
fit error count                        = 0
canonical logical census hash          =
  c9b48074dd59a439fceb9d5e64806adda5620cc4abe32095371abc447ef98634
canonical key corpus hash              =
  b843630969f116da63f7fad095c54de2ff471540159ff97ca56c3871d6b2e1fa
dataset matrix SHA-256                 =
  971f1e0784817c6febfc84154f19540be02a92f6e4acad784dc7f92b979f1df7
```

Named metadata authentication must be recomputed and match:

```text
setup observation metadata =
  5282820451b2658c636132e579859c8c2c8e6497a926b8b6d9c393e0043e667a
same-S setup metadata =
  07830db88c62aa7658d44373e86d897b254e453773b4c0070460dc20fce91113
target fit metadata =
  361672b87cd056a689f578a5eb7660a55d056395ac270d3e28dbbe24738bab40
target risk metadata =
  95eba27f5ea7904761ae4afbc203c58a84566fc3cc308d1b9c90acca69cc96f2
risk cases =
  4a2748ba469e039143c482fd4cf0367324886cc526552e9529117cab7c596d91
```

The loader also requires all 64 Phase 1 shard RDS/summary pairs. It validates
their context and payload hashes through the Phase 1 shard validator, merges
the 110,617 setup-observation and target-risk rows in canonical key order, and
recomputes the two hashes above. It then verifies every non-lineage setup
metadata field is invariant within each same-S group, not only model-matrix,
penalty, constraint, and combined setup hashes.

The Phase 2 loader must not trust a copied `pass` flag. It recomputes file
hashes, named metadata hashes, key-to-group lineage, selected-sp lengths, and
dataset identity. Any mismatch fails before shard reuse.

## PreparedSKey

Each setup has a canonical UTF-8 payload and SHA-256. The payload uses fixed
field order, decimal integers, lowercase hex hashes, comma-separated sorted S,
and a final LF:

```text
schema_version=full-cuda-ci-prepared-s-key-v1
dataset_sha256=<canonical normalized matrix SHA-256>
same_S_group_id=<Phase 1 group SHA-256>
sorted_S=<comma-separated one-based indexes>
formula_class=<full-smooth|additive-smooth>
formula_semantics_version=kpcalg_regrXonS_v1
mgcv_semantics_version=mgcv-gam-gcv-cp-v1
family=gaussian
link=identity
method=GCV.Cp
optimizer=mgcv-default
bs=tp
k=mgcv-default
select=false
scale=mgcv-default
na_action=na.fail
R_version=<exact Phase 1 value>
mgcv_version=<exact Phase 1 value>
```

Required fields:

```text
prepared_s_key_payload
prepared_s_key_sha256
```

The implementation fails closed if one hash maps to two payloads or one
payload maps to two hashes.

## PreparedSSetup Contract

Schema version:

```text
full-cuda-ci-prepared-s-setup-v1
```

Required identity and semantic fields:

```text
schema_version
dataset_sha256
prepared_s_key_payload
prepared_s_key_sha256
same_S_group_id
phase1_setup_fingerprint
sorted_S
formula_class
formula_semantics_version
mgcv_semantics_version
R_version
mgcv_version
family
link
method
optimizer
provider_fingerprint
```

Required response-independent numeric state:

```text
X
coefficient_labels
intercept
assign
cmX
penalty_blocks
penalty_offsets
penalty_ranks
penalty_order
penalty_sp_indices
penalty_sp_labels
sp_mapping
sp_mapping_offset
min_sp
constraint
constraint_mode
constraint_nullspace
constraint_rank
constraint_nullspace_dimension
H
weights
weights_policy
offset
offset_policy
mgcv_penalty_rank_metadata
smooth_classes
smooth_labels
smooth_terms
smooth_by
basis_dimensions
smooth_p_order
smooth_null_space_dimensions
smooth_ranks
smooth_side_constraints
smooth_reparameterized
smooth_parameter_ranges
smooth_sp_ranges
smooth_S_scale
smooth_shift
model_matrix_rank
model_matrix_condition
conditioning_rank
conditioning_condition
penalty_nullity
scoring_n
scoring_n_true
scoring_min_edf
scoring_pearson_extra
scoring_deviance_extra
```

Required static algebra:

```text
weighted_X_policy
gram_matrix
nullspace_gram_policy
nullspace_gram_matrix
```

Canonical neutral values use compact encodings:

```text
empty constraint  -> constraint_mode = identity, no explicit identity Z
identity nullspace -> nullspace_gram_policy = alias-gram,
                      nullspace_gram_matrix = NULL
unit weights      -> weights = NULL
zero offset       -> offset = NULL
absent H          -> H = NULL
```

Do not store a second `X_null` copy when the nullspace is identity. Consumers
interpret `X_null` as `X` under `constraint_mode = identity`.

The object must not contain any field named or semantically equivalent to:

```text
G
y
target
sp
lsp0
Xty
Xty_null
target_fit_fingerprint
residual_hash
fitted_hash
```

The object must also contain no environment, function, formula, call,
external pointer, fitted `gam`, or raw mgcv smooth object. Smooth metadata is
whitelist-projected into plain atomic vectors, matrices, and lists. The
validator recursively scans names, values, classes, and attributes and rejects
response-bearing or executable objects.

Fingerprints are separated:

```text
representation_fingerprint:
  exact hash of the canonical plain-data representation and PreparedSKey

semantic_fingerprint:
  excludes PreparedSKey, target identity, backend identity, and representative
  construction inputs; covers model space, constraints, ordered penalty
  action, provider semantics, and fixed-sp probe behavior
```

The provider fingerprint covers exact R and mgcv versions, hashes of
`kpcalg::regrXonS` and both pinned formula-helper bodies, contrasts, NA
policy, family/link/method defaults, and the Prepared-S extractor schema. It
does not include the target, selected sp, execution backend, or shard identity.

## TargetState Contract

Schema version:

```text
full-cuda-ci-target-state-v1
```

Required fields:

```text
schema_version
residual_key_payload
residual_key_sha256
prepared_s_key_sha256
same_S_group_id
phase1_setup_fingerprint
target
y_source
y_hash
projected_rhs
nullspace_projected_rhs
selected_sp
selected_sp_names
selected_sp_hash
GCV_Cp_score
EDF
convergence_fields
warning_classes
warning_messages
coefficient_rank
coefficient_hash
fitted_hash
residual_hash
target_fit_fingerprint
target_state_fingerprint
```

`y_source` is the canonical dataset SHA-256 plus one-based target column. The
persistent artifact does not duplicate 110,617 copies of a 351-element target
vector. The runtime materializer loads `y` from the authenticated canonical
matrix, verifies `y_hash`, and returns an in-memory TargetState containing the
numeric vector.

`projected_rhs` and `nullspace_projected_rhs` are computed in same-S matrix
batches. The selected smoothing parameter vector remains target-specific and
must have the same length and order as `penalty_blocks`.

## Building One Prepared Setup

For each Phase 1 same-S row:

1. Parse sorted S and construct the exact `x1, x2, ...` conditioning layout
   with `x1 = rep(0, n)`.
2. Build the canonical regrXonS formula through the pinned formula helper.
3. Call `mgcv::gam(..., method = "GCV.Cp", fit = FALSE)` without a fixed
   smoothing parameter.
4. Whitelist-project response-independent matrices, penalties, constraints,
   design metadata, plain smooth descriptors, mapping metadata, and scoring
   constants.
5. Discard the formula, formula environment, call, family closure, smooth
   objects, raw `G`, zero response, and every non-whitelisted attribute.
6. Build static Gram/nullspace state and provider/representation/semantic
   fingerprints.
7. Validate dimensions, ranks, penalty ordering, policies, column-space
   semantics, and Phase 1 lineage.
8. Write the normalized object to its deterministic shard.

`representative_residual_key_sha256` remains Phase 1 lineage metadata only.
It is not used to construct the setup and is not part of the semantic
fingerprint.

## Fixed-Sp Reference Adapter

Add a reference-only adapter with this logical API:

```text
fastkpc_mgcv_magic_fixed_sp_from_prepared(
  prepared_setup,
  target_state
)
```

It invokes the version-pinned mgcv `C_magic` fixed-sp kernel from normalized
PreparedSSetup fields. It supplies target `y` and `sp` from TargetState and
constructs the all-fixed `L/lsp0` values transiently. It does not call
`mgcv::gam`, and it does not mutate or retain a raw template `G`.

The adapter returns:

```text
coefficients
fitted
residuals
selected_sp
setup fingerprint
target fingerprint
solver identity
```

The canonical qualification gate requires exact Phase 1 coefficient, fitted,
and residual hashes. A numeric tolerance is not a substitute for these hashes.

## Semantic Equivalence

The provider representation hash is required for restart and artifact
authentication. The Phase 1 fitted-lpmatrix hash remains lineage evidence, but
it is not required to equal the `fit = FALSE` provider-X hash. Their model
space and behavior must be equivalent.

For selected groups the comparator validates:

1. exact formula, family, method, S order, dimensions, and policy identity;
2. equal model-matrix rank and column-space principal angles within the
   versioned tolerance formula;
3. equal constraint-nullspace rank and projector action;
4. equal penalty action in canonical selected-sp order;
5. exact fixed-sp coefficient, fitted, and residual hashes for selected
   targets;
6. downstream Phase 0 C++ Spectra-route dCov p-value drift at most `1e-12`
   and exact decisions.

The principal-angle tolerance is diagnostic, not an escape from the exact
behavior gates:

```text
semantic_angle_tolerance =
  64 * .Machine$double.eps * max(nrow(X), ncol(X))
```

It is recorded in the manifest. Exact fixed-sp fitted/residual hashes remain
the authoritative behavior gate.

## Deterministic Iteration Subset

The default development gate is class-complete and small enough for repeated
execution:

```text
PreparedSSetup groups = 44
target|S keys         = 270
logical dCov pairs    = 44
Phase 1 fit-time sum  = 12.745 seconds
expected iteration    = approximately 30-60 seconds
```

Selection is deterministic:

1. Include all six conditional tests with
   `absolute_log_distance_from_alpha <= 1e-3`.
2. Within conditional tests at most `log(2)` from alpha, select the closest
   test for every observed `(S_size, reference_decision)`, breaking ties by
   `logical_sequence_id`.
3. Select one ordinary logical case for each `S_size = 1..7`; both targets
   have finite outputs, condition below `1e8`, no numerical or convergence
   risk flag, and alpha distance > `log(2)`; the structural `multi_penalty`
   flag is allowed because canonical `S_size = 3..7` additive smooths
   necessarily have multiple penalties. Choose the lower-median canonical
   residual-pair key.
4. Select one numerical-risk target for every observed
   `(penalty_count, condition_bucket)` in
   `finite_1e8_to_lt_1e12`, `finite_ge_1e12`, and
   `rank_deficient_inf`. Finite buckets choose maximum condition then
   residual key; rank-deficient buckets choose the smallest residual key.
5. Select one target for every observed
   `(convergence_signature,S_size,condition_bucket)` stratum among targets
   where the frozen Phase1 `mgcv_nonconverged` risk flag is `TRUE`, ordered by
   descending optimizer iteration count then residual key.
6. Anchor same-S multiplicity at unique-target fan-out `2/9/47` and logical
   request load `2/16/3092`, choosing the exact or nearest value then the
   smallest group key. Include every target from those groups.
7. Attach each risk target and multiplicity group to its closest-alpha
   canonical consumer, add both residual endpoints, and add representative,
   lower-median, and maximum targets for any remaining selected setup.

The selector records all reasons and hashes sorted setup IDs, target keys,
logical IDs, and reason rows into an iteration-subset identity. It must produce
exactly 44/270/44 on the authenticated corpus.

## Deterministic Qualification Subset

The selector operates on authenticated Phase 1 tables and bytewise-radix key
ordering.

### Seed target keys

Select all target keys with any of:

```text
rank_deficient
nonfinite_metadata
mgcv_nonconverged
high_condition and penalty_count == 1
```

This contributes all rare canonical cases, not a sample.

For every observed `(penalty_count, condition_bucket)` stratum, add:

```text
minimum condition key
lower-median condition key
maximum condition key
lexicographically smallest residual key
```

For each selected-sp component within each penalty count, add keys at:

```text
minimum sp
lower-median sp
maximum sp
```

For each penalty count and empirical setup multiplicity probability
`0, .25, .5, .75, 1`, select the group whose target count is closest to the
type-1 empirical quantile. Break ties by `same_S_group_id`. From that group add
the lexicographic minimum, lower-median, and maximum target keys.

The authenticated corpus produces exactly:

```text
seed target keys = 2,356
```

### Logical-test expansion

Add every conditional near-alpha logical test. Direct-CI level 0 is outside
the residual corpus and is excluded.

For each seed target key, add the first canonical conditional logical test
that consumes it. Add both residual keys required by every selected logical
test. Finally add the first test for every observed `(S_size,
reference_decision)` combination.

The authenticated corpus produces exactly:

```text
conditional near-alpha logical tests = 1,478
selected logical tests               = 3,808
expanded target keys                 = 6,143
selected same-S groups               = 2,061
```

Expanded target-key counts by penalty count are:

```text
penalty count 1 = 3,327
penalty count 3 =   872
penalty count 4 =   837
penalty count 5 =   730
penalty count 6 =   312
penalty count 7 =    65
```

The selector writes every reason attached to each row. It fails if any
canonical rare-risk key, conditional near-alpha test, penalty count, condition
bucket, S size, or reference decision lacks coverage.

The canonical corpus contains zero near-constant target cases, zero
near-constant conditioner cases, and zero mgcv-warning cases. The artifact
records these canonical counts as zero. Synthetic unit tests cover those
absent classes; the real subset must not claim that it contains them.

## dCov Parity

Materialize and cache the 6,143 qualified residual vectors after fixed-sp hash
validation. For each of the 3,808 selected logical tests:

1. resolve its two canonical residual keys;
2. run the pinned legacy `dcov.gamma` implementation with Phase 0 `index` and
   `numCol`;
3. compare the p-value to `reference_p_value` from the authenticated logical
   trace through the same C++ Spectra route with a Phase 2 tolerance of
   `1e-12`;
4. replay the frozen `p >= alpha` decision;
5. record p-value identity, signed alpha distance, and decision identity.

Hard gates:

```text
legacy dCov max p-value drift   <= 1e-12
legacy dCov decision flip count = 0
```

The artifact also reports the exact-p-value match count. Bitwise p-value
identity is informative but is not a hard gate. Phase 2 tightens the existing
source-controlled `1e-8` dCov oracle envelope because it replays the same
C++ Spectra route with exact residual vectors.

Parent skeleton replay is not run in Phase 2. Graph evidence remains inherited
from Phase 0/1 and is reported with the same scope distinction.

## Sharding and Restart

Use 64 deterministic setup shards. Sort `prepared_s_key_sha256` bytewise and
assign:

```text
shard_id = (sorted_rank - 1) %% 64
```

Each shard stores:

```text
schema and context manifest
ordered prepared setup keys
PreparedSSetup objects
ordered TargetState rows for those groups
expected key count
expected key-set hash
payload hash
build timing
```

Shard context includes:

```text
Phase 1 input bundle hash
dataset file and matrix hashes
canonical logical and key corpus hashes
PreparedSKey corpus hash
source commit
R and mgcv versions
BLAS/LAPACK identity
BLAS thread count
schema versions
semantic tolerance config hash
qualification-selection config hash
```

Write a temporary file, validate it by reading it back, then atomically rename
it. Resume validates context, expected keys, object schemas, response-leakage
scan, and payload hash. A stale or mismatched shard is rejected, never silently
reused.

## Artifact Layout

Required directory:

```text
fastkpc/artifacts/full_cuda_ci/prepared_s_contract_v1/
```

Required files:

```text
manifest.json
summary.json
summary.md
commands.txt
environment.txt
input_hashes.csv
prepared_s_setup_index.rds
prepared_s_setup_index.csv
target_state_index.rds
target_state_index.csv
iteration_setup_groups.rds
iteration_setup_groups.csv
iteration_target_keys.rds
iteration_target_keys.csv
iteration_logical_tests.rds
iteration_logical_tests.csv
iteration_coverage.csv
qualification_setup_groups.rds
qualification_setup_groups.csv
qualification_target_keys.rds
qualification_target_keys.csv
qualification_logical_tests.rds
qualification_logical_tests.csv
qualification_coverage.csv
setup_semantic_parity.csv
target_retarget_parity.csv
dcov_parity.csv
unsupported_envelope.csv
fallbacks.csv
stage_timing.csv
shards/shard_<id>.rds
shards/shard_<id>.summary.json
```

CSV TargetState rows contain RHS hashes and lengths. Numeric projected RHS
vectors remain in RDS to avoid lossy text serialization.

Expected uncompressed model-matrix payload is approximately 771 MiB. Penalty
blocks add approximately 24 MiB. The implementation reports compressed
artifact size, maximum resident memory, per-shard size, and stage timings, but
Phase 2 has no wall-time performance gate.

## Hard Gates

Full structural artifact:

```text
phase2_input_authenticated                = TRUE
prepared_s_setup_count                    = 8,634
target_state_count                        = 110,617
prepared_s_key_corpus_exact               = TRUE
target_key_corpus_exact                   = TRUE
setup_lineage_exact                       = TRUE
target_lineage_exact                      = TRUE
response_leakage_count                    = 0
prepared_setup_fingerprint_collision_count = 0
target_state_fingerprint_collision_count   = 0
unsupported_canonical_setup_count          = 0
unknown_fallback_count                     = 0
approximate_backend_count                  = 0
```

Qualification subset:

```text
iteration_setup_group_count              = 44
iteration_target_key_count               = 270
iteration_logical_test_count             = 44
seed_target_key_count                  = 2,356
qualification_target_key_count         = 6,143
qualification_logical_test_count       = 3,808
qualification_same_S_group_count       = 2,061
conditional_near_alpha_test_count      = 1,478
fixed_sp_coefficient_hash_exact        = TRUE
fixed_sp_fitted_hash_exact             = TRUE
fixed_sp_residual_hash_exact           = TRUE
legacy_dcov_max_abs_p_value_diff        <= 1e-12
legacy_dcov_decision_flip_count        = 0
```

Inherited graph evidence:

```text
oracle_inherited_graph_gate = TRUE
new_candidate_graph_gate    = NOT_APPLICABLE
```

## Failure Policy

Fail closed on:

- Phase 1 file, schema, named-metadata, or lineage mismatch;
- dataset identity mismatch;
- PreparedSKey collision or serialization ambiguity;
- target-specific data found in PreparedSSetup;
- selected-sp length/order mismatch;
- non-finite structural matrices;
- response materialization hash mismatch;
- Phase 1 setup metadata mismatch;
- missing rare-risk or near-alpha coverage;
- fixed-sp coefficient, fitted, or residual hash mismatch;
- dCov p-value drift above `1e-12` or any decision mismatch;
- stale, missing, duplicate, or corrupt shard;
- any fallback, approximation, or unsupported canonical setup.

No tolerance widening, warning-only downgrade, or silent CPU/GPU substitution
is allowed.

## Code Boundaries

Create:

```text
fastkpc/R/full_cuda_ci_prepared_s_contract.R
fastkpc/tools/run_full_cuda_ci_prepared_s_contract.R
fastkpc/tests/test_full_cuda_ci_prepared_s_contract.R
fastkpc/tests/test_full_cuda_ci_target_retarget.R
fastkpc/tests/test_full_cuda_ci_prepared_s_contract_restart.R
fastkpc/tests/test_full_cuda_ci_prepared_s_contract_real_subset.R
```

Reuse without changing production routing:

```text
fastkpc/R/full_cuda_ci_workload_census.R
fastkpc/R/mgcv_compat_contract.R
fastkpc/R/mgcv_extract_oracle.R
fastkpc/R/native.R
```

Do not add Phase 2 logic to `legacy_runner.R` or the live skeleton path.

## API Shape

The module exposes focused functions:

```text
fastkpc_full_cuda_prepared_s_input_contract()
fastkpc_full_cuda_prepared_s_load_inputs()
fastkpc_full_cuda_prepared_s_key()
fastkpc_full_cuda_build_prepared_s_setup()
fastkpc_full_cuda_validate_prepared_s_setup()
fastkpc_full_cuda_build_target_states()
fastkpc_full_cuda_validate_target_states()
fastkpc_full_cuda_materialize_target_state()
fastkpc_mgcv_magic_fixed_sp_from_prepared()
fastkpc_full_cuda_select_prepared_s_iteration_subset()
fastkpc_full_cuda_select_prepared_s_qualification_subset()
fastkpc_full_cuda_run_prepared_s_target_parity()
fastkpc_full_cuda_run_prepared_s_dcov_parity()
fastkpc_full_cuda_run_prepared_s_shard()
fastkpc_full_cuda_merge_prepared_s_shards()
fastkpc_full_cuda_write_prepared_s_artifact()
```

## Tests

### Contract unit test

`test_full_cuda_ci_prepared_s_contract.R` verifies:

- canonical key serialization and collision checks;
- response-bearing fields are rejected recursively;
- neutral state uses compact encodings;
- penalty order and selected-sp order are identical;
- target state references the canonical dataset column;
- malformed dimensions, hashes, and fingerprints fail closed;
- synthetic nonunit weights, nonzero offsets, near-constant responses, and
  warning/non-finite cases are explicitly classified.

### Target retarget test

`test_full_cuda_ci_target_retarget.R` covers canonical real cases for penalty
counts 1, 3, 4, 5, 6, and 7, including high-condition, rank-deficient,
nonconverged, and near-alpha-linked targets. It compares independent fixed-sp
mgcv, PreparedSSetup retargeting, and Phase 1 hashes.

### Restart test

`test_full_cuda_ci_prepared_s_contract_restart.R` injects:

- interruption before atomic rename;
- duplicate and missing shard;
- wrong PreparedSKey corpus;
- wrong Phase 1 input hash;
- wrong schema/tolerance/selection config;
- corrupted payload;
- response-bearing data inserted into a shard.

Every case must fail or rebuild deterministically as specified.

### Real-subset test

`test_full_cuda_ci_prepared_s_contract_real_subset.R` has three layers:

1. exact selection and numerical execution of the complete 44/270/44
   iteration corpus;
2. a focused unit smoke over one representative per penalty count;
3. selection-only validation of the complete 2,356/6,143/3,808/2,061
   qualification corpus.

The full 6,143-key parity run is performed by the artifact runner, not by every
default test invocation.

## Standard Commands

Local tests:

```bash
Rscript fastkpc/tests/test_full_cuda_ci_prepared_s_contract.R
Rscript fastkpc/tests/test_full_cuda_ci_target_retarget.R
Rscript fastkpc/tests/test_full_cuda_ci_prepared_s_contract_restart.R
Rscript fastkpc/tests/test_full_cuda_ci_prepared_s_contract_real_subset.R
git diff --check
```

Full Phase 2 artifact:

```bash
FASTKPC_FULL_CUDA_PREPARED_S_WORKERS=20 \
FASTKPC_FULL_CUDA_PREPARED_S_SHARD_COUNT=64 \
FASTKPC_FULL_CUDA_PREPARED_S_RESUME=1 \
Rscript fastkpc/tools/run_full_cuda_ci_prepared_s_contract.R
```

## Exit Condition

Phase 2 is complete only when the authenticated artifact contains all 8,634
PreparedSSetup objects and all 110,617 TargetState rows, the deterministic
qualification subset passes exact fixed-sp and dCov gates, unsupported and
fallback counts are zero, restart tests pass, and the artifact can be consumed
by Phase 3 without calling `mgcv::gam` once per target.
