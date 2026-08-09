# Phase 10 strict full-eig component closure

## Decision

The current single-GPU `dcc.gamma` full-eig component optimization is closed
under `numerical_contract_v1`:

```text
retained implementation:  GPU-only strict legacy-compatible full eig
retained scheduling:      one frontier group at a time
retained eig lanes:       8
final development time:   928.720 s
decision:                 STOP_CURRENT_STRICT_SINGLE_GPU_FULL_EIG_COMPONENT
promotion status:         NOT_PROMOTED
candidate:                NOT_FROZEN
sealed holdout:           SEALED_NOT_RELEASED
recommended route:        false
```

This STOP is scoped to the full-eig component on the current RTX 4090,
current CUDA/cuSOLVER stack, and current strict numerical contract. It is not a
claim that a different library, GPU generation, or numerical-contract epoch
cannot improve the route.

## Hard contract

Every retained result satisfies all of the following:

```text
all canonical CI absolute errors       <= 1e-10
all finite pMax absolute errors         <= 1e-10
adjacency / semantic sepsets            exact
n.edgetests / task and deletion identity exact
CPU residual/component/eig/gamma        0
unknown or approximate fallback         0
residual/component payload D2H           0
CUDA resource counters after call        0
```

The production numerical authority remains deterministic
`cusolverDnDsyevd`. The invalid historical conditional-screen route and its
`219.513`-second timing are not eligible evidence.

## Retained implementation

The retained strict development configuration is:

```text
legacy component cache budget       11 GiB
component-cache residual bypass     enabled
persistent full-eig lanes           8
frontier wave capacity              1
positive-miss wave policy           disabled
```

Its canonical cache accounting closes exactly:

```text
component requests                  228,033
component cache hits                117,350
unique component misses             110,683
cache inserts                       110,683
cache bypasses                            0
all-hit residual-bypass batches      13,641
component batches                    38,622
active eig-lane peak                      8
```

The cache capacity is 119,817 entries, so every unique canonical component is
materialized once and retained. Increasing this cache cannot remove another
canonical eigensolve.

## Performance progression

The accepted changes reduce the repaired strict GPU-only route as follows:

| Configuration | Outer elapsed | Full eig | Component build | Residual |
| --- | ---: | ---: | ---: | ---: |
| uncached correctness repair | 2293.965 s | - | 1701.543 s | 340.900 s |
| persistent component cache | 1425.015 s | 809.982 s | 833.281 s | 340.504 s |
| cached pair-only path | 1406.854 s | 809.897 s | 833.132 s | 340.514 s |
| all-hit residual bypass | 1378.152 s | 810.520 s | 833.766 s | 306.897 s |
| persistent 8-lane eig | 931.290 s | 407.850 s | 431.023 s | 307.051 s |
| final current-binary audit | **928.720 s** | **407.716 s** | **430.865 s** | **306.573 s** |

Relative to the corrected 2293.965-second baseline, the retained route is
`2.470x` faster and saves 1365.245 seconds (`59.51%`). The final 928.720 and
earlier 930.100-second canonical runs are treated as the same performance band,
not as evidence for a separate 1.380-second optimization.

The final component build is:

```text
distance construction                 0.360 s
deterministic full eig               407.716 s
component selection/finalization      22.789 s
total component build                430.865 s
```

Even deleting all non-eigensolver component work would save only 23.149
seconds (`2.49%` of outer elapsed). The remaining dominant cost is therefore
the required cuSOLVER eigensolver itself.

## Packing and overlap closure

The 8-lane canonical histogram has a theoretical globally saturated full-eig
time of 323.616 seconds. Against the 930.100-second diagnostic run, this leaves
84.329 seconds of ideal packing headroom. That number is not a realizable
performance budget because online skeleton replay determines which next
frontier groups exist and how they coalesce.

An unselective two-group synchronous wave demonstrated the cost of changing
that frontier shape:

```text
outer elapsed                        972.204 s
paired waves                          22,022
full eig                             426.298 s
scheduler regression                  42.104 s
all-shape measured eig saving         51.707 s
projected net upper bound              9.603 s
```

A stricter policy paired only groups that both had predicted positive misses
and disjoint missing component keys. Its complete canonical audit produced:

```text
outer elapsed                        967.114 s
selective pairs                        8,935
prediction-incomplete count                0
missing-key overlap count                  0
prediction host time                   0.816 s
full eig                             423.237 s
perfect overlap upper bound          111.195 s
scheduler regression                  37.014 s
```

All 235 canonical positive-miss pair shapes were then measured with two
independent 8-lane contexts. The benchmark contains 237 rows including two
stress shapes, has 100% canonical opportunity coverage, and has zero
eigenvalue/eigenvector bit mismatches. Frequency-weighted device saving is
only 38.483 seconds, leaving a projected net upper bound of **1.469 seconds**
after the observed scheduling regression. This is the same scale as canonical
run-to-run variation and does not qualify an asynchronous ticket refactor.

Capacity-eight synchronous waves also regressed to 1070.153 seconds and
59,091 component batches. More in-flight frontier groups are therefore
rejected.

## Lane, submission, and solver closure

The machine closure receipt reruns representative shapes without competing GPU
processes.

Increasing lanes beyond eight is flat:

| Components | Baseline lanes | Expanded lanes | Device speedup |
| ---: | ---: | ---: | ---: |
| 9 | 8 | 9 | 1.0013x |
| 16 | 8 | 16 | 1.0024x |
| 48 | 8 | 16 | 0.9986x |

The maximum observed improvement is `1.0024x`, below measurement and
integration significance. Per-lane host-thread submission also has no material
gain: its best representative ratio is `0.9979x`, and several shapes regress
more substantially. The existing single-host-thread submission is retained.

`cusolverDnDsyevj` is not an alternative for this contract. On the canonical
351x351 shape it is only `0.0860x` as fast as `cusolverDnDsyevd`, while its
maximum eigenvalue difference is `1.413127e-10` and its maximum sign-aligned
eigenvector difference is `1.858878e-06`. It is both slower and numerically
ineligible.

## Final canonical evidence

The final current-binary default-`Inf` audit records:

```text
outer elapsed                         928.720 s
CI tests                              240,498
maximum CI absolute error             6.9555444737190442e-11
CI mismatches above 1e-10                    0
maximum pMax absolute error           6.9555444737190442e-11
pMax mismatches above 1e-10                  0
task / deletion identity              exact
adjacency / semantic sepsets          exact
n.edgetests                           exact
CPU authority / fallback / large D2H       0
vertical resource closure             PASS
```

Evidence identities:

All files below are under
`fastkpc/artifacts/full_cuda_ci/phase10_legacy_component_cache_v1/`.

```text
native binary SHA-256
  e3ee4a97624bb64d69e8de55fd86f78748e3375670282b0b5b8cf0a19d9ffc9d

final canonical receipt SHA-256
  canonical_wave1_final_audit.rds
  bf46ea77ac0fcb724263b4f6b7533ae2efde59e4f3a7edefa1d57ae3a95d35a8

final semantic revalidation SHA-256
  canonical_wave1_final_revalidation.rds
  7cc53131d6f3526d4124c7ea223a57f10f6f9bcaacfea9991c69c9300696e621

selective-wave canonical receipt SHA-256
  canonical_wave2_positive_miss_sync_audit.rds
  3a7c02225189e082e41db83e8d2b577844896eade6c063675a3bd9c06893bddd

selective-wave all-shape benchmark SHA-256
  canonical_wave2_positive_miss_two_batch_benchmark_all_pairs.rds
  0ae3fd937f8681b3c575ae9b7b13308e431582db6686c3e68a7b1d05424fb7be

benchmarked opportunity SHA-256
  canonical_wave2_positive_miss_benchmarked_opportunity.rds
  54105a327186ab218af6d86071d86dfacaad69a236fb7eff85da333c0445ad19

machine closure receipt SHA-256
  canonical_full_eig_component_closure.rds
  e1380520aecf1bb14b80bb852bb81a8c0cc9880b98eb818db3fcf8256ffac840
```

These are local, native-hash-bound development artifacts from the current
worktree. The runtime source is not a frozen clean producer, so they are not
candidate-freeze or promotion evidence.

## Reopen conditions

Reopen this component only when at least one of the following is true:

```text
- a new CUDA/cuSOLVER release or GPU generation demonstrates a material
  complete-route gain while preserving the current all-CI 1e-10 contract;
- a scheduler design removes online frontier fragmentation and demonstrates
  a material repeated full-run gain, not only ideal or microbenchmark overlap;
- a new numerical-contract major version explicitly permits a different
  eigensolver, subset/low-rank method, or altered component semantics;
- a separate multi-GPU execution goal is authorized.
```

Until then, further single-GPU full-eig component work is stopped. Phase 10
promotion remains active only as a broader project status; this component no
longer has a qualified material optimization route.
