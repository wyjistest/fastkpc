# GPU mgcv selected-SP and fixed-SP semantic closure

## Status

The GPU-only semantic repair is complete for the qualified `dcc.gamma`
production route.

```text
runtime producer commit:       377676f
audited native SHA-256:        6f417bce389fa1d852e6aba8d35fb2a836e9f5a9ba30cabdbbb4c5ba789b5697
selected-SP finalization:      GO_EXACT_GPU_REPLAY
fixed-SP execution:            GO_EXACT_GPU_EVALUATOR
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
CUDA selected-SP optimizer
  -> risk-gated CUDA exact objective replay
  -> selected log-SP retained in native ownership
  -> CUDA exact fixed-SP evaluator at that log-SP
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

## Final native gates

The final full rebuild and focused tests passed under the audited native SHA:

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
GPU exact selected-SP replay + exact fixed-SP evaluator: GO
generic KKT/refinement as mgcv authority:                 STOP
CPU fallback:                                             FORBIDDEN
all-task p-value tolerance:                               1e-10
```

This completes the requested GPU semantic replication on the qualified route.
It does not pass Phase 10 performance gates, freeze a candidate, recommend the
route, or authorize opening the sealed holdout.
