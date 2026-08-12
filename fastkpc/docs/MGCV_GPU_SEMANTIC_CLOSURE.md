# GPU mgcv selected-SP and fixed-SP semantic closure

## Status

The GPU-only semantic repair is complete for the qualified `dcc.gamma`
production route.

```text
semantic closure commit:       377676f
exact-primary commit:          305d7a7
current runtime commit:        dcb1749
audited native SHA-256:        fe8286f7e6c6114ef66e040b16b693f5abcb23333dc9c0fd767c042c509c3c50
selected-SP finalization:      GO_EXACT_GPU_PRIMARY
fixed-SP execution:            GO_EXACT_GPU_EVALUATOR
exact-primary/fixed reuse:     GO_COEFFICIENT_ONLY
CPU numerical authority:       0
CPU/unknown fallback:          0
KKT refinement authority:      STOP_NOT_MGCV_SEMANTIC_AUTHORITY
Phase 10:                      ACTIVE
candidate:                     NOT_FROZEN
recommended route:             false
sealed holdout:                SEALED_NOT_RELEASED
```

This is a correctness and implementation closure, not promotion evidence. The
historical `219.513`-second conditional-screen route remains invalid. The later
`928.720`-second full-eig optimization evidence predates this exact selected-SP
and fixed-SP closure and is not a performance result for producer `377676f`.

## Production shape

The one-call route now keeps both smoothing-parameter selection and the final
fixed-SP residual execution on the GPU:

```text
CUDA stable-SVD exact-primary selected-SP optimizer
  -> selected log-SP retained in native ownership
  -> final selected-SP stable-SVD coefficient identity
  -> level-scoped device coefficient arena
  -> original cohort-ordered CUDA fitted/residual materialization
     (independent exact fixed-SP evaluator when no arena identity exists)
  -> device-resident residual/component/full-eig/gamma path
  -> compact p-value result only
```

The exact selected-SP implementation ports the required LAPACK 3.12 numerical
building blocks into deterministic CUDA device code. Capacity-specific
multi-penalty evaluators cover the qualified `q=64/80/192/384/559` buckets.
The single-penalty spectral path sends endpoints at or below
`2.0 * convergence_tolerance` to exact replay; this includes the real witness
that exposed the former precision-floor gap.

The production exact fixed-SP evaluator is distinct from the experimental KKT
refinement API. Every one-call exact-evaluator target uses the stable-SVD route,
and no CPU residual, optimizer, dCov component, eigensolver, gamma p-value, or
fallback result is accepted.

Producer `305d7a7` makes the exact-primary optimizer the `dcc.gamma` one-call
default. `FASTKPC_DCC_GAMMA_EXACT_PRIMARY_OPTIMIZER=0` retains the former
primary-plus-replay shape only as an explicit regression escape hatch. Other CI
methods do not opt into exact-primary execution.

Producer `dcb1749` additionally makes exact-primary coefficient reuse the
`dcc.gamma` default. Targets whose accepted optimizer state previously reused
an earlier evaluation first receive one deterministic stable-SVD refresh at
the final selected log-SP. The runtime detaches only the resulting
coefficients, route, rank, and solver status into a level-scoped device arena.
At CI consumption it gathers coefficients in the original exact-fixed cohort
order and reruns the same `X beta` and `Y - X beta` materialization. It does not
reuse the old optimizer residual. `FASTKPC_DCC_GAMMA_EXACT_PRIMARY_FIXED_REUSE=0`
disables this path; disabling exact-primary also disables reuse and restores
the primary-plus-replay route without requiring a second setting.

## Exact-primary and retained full-eig requalification

The exact producer was first qualified with the retained full-eig stack:

```text
FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES=11811160064
FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS=1
FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES=8
```

The isolated exact baseline took `2960.106s`. Three retained-stack repeats took
`1512.117s`, `1540.782s`, and `1538.940s`, for a `1538.940s` median. Profiling
then showed that every multi-penalty target executed the stability replay and
selected its result; a captured primary and replay consumed `13.946s` and
`13.226s`, respectively, while merge cost was only `0.443ms`.

The production optimization therefore runs that same deterministic stable-SVD
optimizer once as the primary and omits the redundant replay and merge. Its
`305d7a7` source-closure canonical audit produced:

| Reference route | Elapsed (s) | Current reduction from reference |
| --- | ---: | ---: |
| exact baseline A | `2960.106` | `57.02%` |
| retained stack C, three-run median | `1538.940` | `17.32%` |
| exact-primary production | **`1272.360`** | reference |

That value is one isolated source-closure development run, not a clean
checkout or repeated promotion median. Its hard gates passed:

```text
logical CI tests                         240498 / 240498
maximum CI absolute error               3.0547508966805026e-12
maximum pMax absolute error             3.0547508966805026e-12
exact-primary multi-penalty targets     80751 / 80751
stability replay targets / selected     0 / 0
terminal confirmations                  0
exact fixed stable-SVD targets          164386 / 164386
component requests / hits / misses      228033 / 117350 / 110683
all-hit residual-bypass batches         13641
eig lanes requested / active peak       8 / 8
cache route/status mismatches            0 / 0
CPU residual authority / fallback        0 / 0
resource closure                         PASS
```

The v8 compute profile now accounts for residual bypass explicitly:

```text
physical residual fits + bypass targets = legacy residual targets
164434 + 63599                           = 228033
```

Its top-level attribution was:

| Stage | Host boundary (s) |
| --- | ---: |
| selected-SP optimizer | `585.585` |
| dCov/full-eig | `437.221` |
| exact residual solve | `177.935` |
| native setup | `22.663` |
| control/packaging, unattributed | `48.615` |

That canonical run's tasks, `pMax`, adjacency, and sepsets are bitwise
identical to the earlier `c700c1b1` exact-primary prototype. The final
12-witness run covers `q=28,37,55,64,73,82`; all 12 exact-primary identities,
stable-SVD finalizations, and numerical contracts pass. Its exact-primary
evidence identities are:

```text
runtime commit                         305d7a7
native SHA-256                         20ab56e7de5a88982c46140df4a90ee8ab544e3ee020bae6a67ef6d73c0bb5eb
canonical receipt SHA-256              cc8fec635c91df0ad925a3b3c34afa8048d5ee64ffd81b910a960403328d5662
12-witness receipt SHA-256             384a3b2b694d0df69b391b2757c536da5653974c06a02b798e20e327113e9fe4
```

The test-only prototype also passed all three larger real inputs in
`1251.553s`, `1051.098s`, and `628.140s`, with the same maximum p-value errors
already listed below. Their receipt SHA-256 values are, respectively,
`30a21a70b828701262195b5cf3119f56cad11b52c45b7b4786f181bc573bf069`,
`b63a1e64c9edbebffe8c44b625df229d4a3d33773172826644dd8971330d3d94`,
and `bdf090a0cefff4efa8c1c25803ca7a789a23243c9a9c2d605bd86e6988bba4a2`.
Those three receipts use native SHA
`c700c1b16127791c8d922e4029922f48f5f408fc605ded7c68c13d2b0887a928`;
they are not relabelled as current-runtime runs. Production cleanup is attested
by that canonical bitwise identity and the 12-shape witness
campaign. A candidate freeze would still require fresh repeated runs from a
clean checkout of the frozen producer.

## Exact-primary coefficient reuse closure

The prior `STOP_OPTIMIZER_RESIDUAL_NUMERICAL_PARITY` remains valid. That
experiment attempted to consume the optimizer's existing residual, whose
rank-deficient coefficient representative did not match exact fixed-SP
finalization. The accepted route instead forces the final selected-SP
stable-SVD evaluation when needed, proves coefficient identity, retains only
those coefficients, and reproduces fitted values and residuals through the
original finalization operations.

Three clean exact-primary baseline runs took `1266.594s`, `1284.209s`, and
`1283.663s`, for a median of `1283.663s`. Two coefficient-only candidate runs
took `1149.219s` and `1141.963s`, for a two-run median of `1145.591s`. The
median reduction is `138.072s` (`10.76%`). The work ledger explains the change:

```text
final selected-SP refresh targets             35059
coefficient-reuse hit targets                 75020
reuse fallback targets                            0
baseline exact-fixed stable-SVD targets      164386
candidate exact-fixed stable-SVD targets      89366
optimizer stable-SVD target delta            +35059
exact-fixed stable-SVD target delta          -75020

baseline median optimizer boundary          596.616 s
candidate median optimizer boundary         600.311 s
baseline median exact-fixed boundary        177.958 s
candidate median exact-fixed boundary        34.679 s
```

After default-on policy and regression escape-hatch hardening, a complete
native rebuild produced SHA-256
`fe8286f7e6c6114ef66e040b16b693f5abcb23333dc9c0fd767c042c509c3c50`.
With the reuse environment variable unset, the final source-closure canonical
audit took `1140.851s` and passed:

```text
logical CI tests                         240498 / 240498
maximum CI / pMax absolute error         3.0547508966805026e-12
exact-primary targets                    80751 / 80751
selected-SP refresh targets              35059
coefficient arena stores / reuse hits    80751 / 75020
reuse hit batches / fallback targets     15264 / 0
remaining exact-fixed stable-SVD targets 89366 / 89366
arena peak device bytes                  16991872
arena allocations / frees / live bytes   9 / 9 / 0
component requests / hits / misses       228033 / 117350 / 110683
all-hit residual-bypass batches          13641
eig lanes requested / active peak        8 / 8
CPU numerical authority / fallback       0 / 0
task, p-value, pMax and graph identity    PASS
resource closure                         PASS
```

The final default-bound result is bitwise identical to the pre-default
candidate for tasks, p-values, `pMax`, adjacency, sepsets, and `n.edgetests`.
Its receipt SHA-256 is
`b3414ca690bde1301ee6393aa5aed6208a60628e4e91d54de09b423441c67af4`.
The two candidate receipt SHA-256 values are
`81407bf94afbdae0e7360b8c9e09c7be53330b2e44fa271727480189c4b5eb83`
and `1dbdb8c915c576c65eb89c377c95a25177d7ee69a9813aa2bdcd6b144dcee08c`.

The final default-bound single-run attribution is:

| Stage | Host boundary (s) |
| --- | ---: |
| selected-SP optimizer | `596.810` |
| dCov/full eig | `437.082` |
| exact-fixed residual evaluation | `34.420` |
| native setup | `22.814` |
| control/packaging, unattributed | `49.725` |

Exact-fixed finalization is therefore no longer the third large stage. The
remaining dominant measured boundaries are selected-SP exact-primary itself
and the retained dCov/full-eig stack. This is attribution, not authorization to
reopen an already stopped optimizer or eig prototype without new evidence.

A 12-witness campaign spanning `q=28,37,55,64,73,82` reports bitwise-identical
selected log-SP, route, rank, solver info, coefficients, fitted values,
residuals, and derived p-values between refreshed exact-primary output and
independent exact-fixed finalization. Its receipt SHA-256 is
`fc9f802461f31d5ddf17506381c19cfb8e93c165f20410546118d76e3dbf42a9`.
The candidate campaign also passed maximum-capacity q192/q384/q559
qualification. The generic optimizer, persistent residual-token, Phase 10
hardening, and focused one-call regression suites pass on the final full-build
binary.

Three final-binary real-input fixtures independently compare reuse disabled
and enabled:

| Shape | Baseline (s) | Reuse (s) | Reuse hits | p-value / pMax / graph |
| --- | ---: | ---: | ---: | --- |
| `179x8` | `2.186` | `1.965` | 8 | bitwise exact |
| `243x9` | `3.842` | `3.665` | 121 | bitwise exact |
| `300x10` | `6.222` | `5.946` | 104 | bitwise exact |

All three have zero fallback and closed resources. Their combined receipt
SHA-256 is
`9cb38dfa1937846bf0a0996dcaf7b7332b1a7821d0662cad6f926b060ba42b86`.

The `1140.851s` value is one isolated source-closure development run. The
`1145.591s` value is a two-run development median generated before the final
default-policy patch. Neither is a clean-checkout repeated producer median or
promotion evidence.

## Semantic-closure native gates

The original semantic-closure full rebuild and focused tests passed under
native SHA `6f417bce...b5697`:

```text
PASS Phase 4 single-penalty CUDA GCV
PASS Phase 4 single-penalty GCV reference
PASS Phase 4 precision-floor exact replay regression
PASS exact GPU fixed-SP evaluator
PASS Phase 9 focused one-call compatible CUDA skeleton
PASS fixed-sp CUDA KKT refinement and fail-closed mechanism
```

The canonical default-`Inf` production test then passed all 240,498 logical CI
tests:

```text
maximum CI absolute error       3.0547508966805026e-12
maximum pMax absolute error     3.0547508966805026e-12
CI errors above 1e-10           0
pMax errors above 1e-10         0
task/deletion/graph semantics   exact
CPU authority/fallback          0
native SHA-256                  6f417bce...b5697
```

## Real-input qualification

Three larger `causalDiscoveryInput` datasets were replayed against their CPU
mgcv p-value oracles. All task identities, logical/deletion traces, adjacency,
sepsets, `pMax`, `n.edgetests`, orientation events and CI identities, final
PDAG, and R RNG endpoints passed. All CUDA active-resource and external-owner
counters returned to zero.

| Dataset | Tasks | Maximum p error | Maximum CPU-pMax error | Elapsed (s) |
| --- | ---: | ---: | ---: | ---: |
| `20230331145508` | 63,249 | `1.6710660633023622e-12` | `7.8631545719076712e-13` | `1794.974` |
| `20240102191119` | 116,911 | `3.6932956692936614e-12` | `2.3558932582545822e-13` | `1704.821` |
| `20241215091815` | 43,057 | `4.8635392588258775e-12` | `1.1435297153639112e-13` | `959.197` |

The local receipt SHA-256 values are:

```text
20230331145508  c45c7964e176da795b86f7cfbd1ceb360b9fb7d7eff9da5ae8abcaea0bfc1380
20240102191119  199d56829fb88ab8aaf748a4ad40f15d4586ac99ba5a6d2492c79ea9fc2df69f
20241215091815  605fc31cb5a1c5d2c0f0003cdd8bb9ce26494232dd7cb4eb38b1492cd31f17f5
```

These receipts are local development evidence. The external input directory
and archive are not repository artifacts and are not committed.

## Selected-SP and fixed-finalization witnesses

The 12-witness campaign covers `q=28,37,55,64,73,82`, multiple penalty counts,
and both base and extended capacity evaluators:

```text
cases passed                    12 / 12
stable-SVD finalizations        12 / 12
maximum log-SP error            1.1102230246251565e-16
maximum SP relative error       0
maximum residual error          1.7763568394002505e-15
maximum p-value error           2.55351295663786e-15
CPU authority/fallback          0 / 0
receipt SHA-256                 84cf17da6f5e17630f0e8fe4a5f3ac32099713b6bf87694f1e71a32d5aa90f6f
```

The former failing task `14591`, `8 _||_ 21 | {3,38}`, is now exact within the
contract:

```text
maximum GPU/mgcv log-SP error       1.3633538742396922e-13
maximum GPU/CPU-fixed residual error 6.2172489379008766e-15
GPU full-group p-value               0.025497028574845593
CPU oracle p-value                   0.025497028574845926
solver route/rank/info                exact stable result
receipt SHA-256                       119237cbe8d0a844f654dc96a2c3de88859c7902e39442d6a41ffca407354b3c
```

## KKT/refinement STOP

The requested FP64 KKT/backward-error and iterative-refinement route was also
implemented and tested as an explicitly test-only API. It is not production
authority.

On 14 real fixed-SP witnesses:

```text
KKT certificate passed                         3
KKT certificate failed                        11
expected fail-closed cases                    11
observed fail-closed with zero shadow publish 11
certificate passes that still failed p<=1e-10  1
CPU/unknown fallback                           0 / 0
resources released                             true
```

The largest refined p-value error was `0.014580781407526389`. More
importantly, one QR witness met the `1e-15` KKT certificate while its p-value
error was `1.454857345706273e-10`. A small KKT residual proves that a solution
satisfies the penalized equations; it does not prove that it selected the same
rank-deficient pseudoinverse representative and downstream p-value as mgcv.

The resulting decision is therefore:

```text
STOP_KKT_CERTIFICATE_NOT_SUFFICIENT_FOR_MGCV_SEMANTICS
```

The v2 STOP diagnostic itself passes because every certificate failure fails
closed without publishing a partial result, and because it detects the
semantic false positive. Its receipt SHA-256 is
`e7fad831a47d8cb001fac82f344c7d15a824b4e4d95d3d7043c0e8e8d0b8675d`.

## Route decision

The production conclusion is narrow:

```text
GPU exact selected-SP + exact fixed-SP evaluator:          GO
refreshed exact-primary coefficient reuse:                 GO
unrefreshed optimizer residual reuse:                      STOP
generic KKT/refinement as mgcv authority:                  STOP
CPU fallback:                                              FORBIDDEN
all-task p-value tolerance:                                1e-10
```

This completes the requested GPU semantic replication on the qualified route.
It does not pass Phase 10 performance gates, freeze a candidate, recommend the
route, or authorize opening the sealed holdout.
