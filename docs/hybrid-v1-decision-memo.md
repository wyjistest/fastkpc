# hybrid-v1 decision memo

Date: 2026-06-23
Commit: bd63aad
Environment: R 4.4.1, mgcv 1.9.1, CUDA driver 555.42.06, 2x NVIDIA GeForce RTX 4090

## Decision

Freeze hybrid-v1 as an accuracy-oriented opt-in mode.

Product positioning:

- `precision = "fast"` remains the default high-throughput mode.
- `precision = "hybrid"` is the recommended accuracy-oriented opt-in mode.
- `precision = "compatible"` remains the reference / validation mode.

Do not restart CUDA kernel, replay, cache, or verifier optimization for hybrid-v1 unless there is a correctness failure, crash, invalid cache reuse, or native CUDA fixed-sp parity regression.

## Evidence

Artifacts:

- Tail latency campaign: `fastkpc/artifacts/precision_tail_30`
- Graph-value campaign: `fastkpc/artifacts/precision_graph_value`
- Orientation-risk validation: `fastkpc/artifacts/hybrid_v1_orientation_validation`

Tail campaign:

- Modes: `primary_only_cuda`, `hybrid_cuda`
- Repeats: 30
- Runs: 180 / 180 OK
- Warm-up: enabled
- Mode order: randomized
- Tau: `log(2)`

Hybrid / primary-only tail results:

| scenario | median ratio | p90 ratio | p90 overhead ms |
| --- | ---: | ---: | ---: |
| chain-nonlinear-p5 | 1.393 | 2.253 | 29.0 |
| fork-additive-p6 | 1.677 | 2.205 | 58.1 |
| scale-nonlinear-p8 | 1.623 | 2.029 | 84.1 |
| pooled | 1.652 | 2.152 | 66.4 |

The p90 ratio misses the earlier nominal `<= 1.75` ratio gate, but the absolute p90 overhead is only 29-90 ms across scenarios. The slow-run attribution is mainly higher verifier count, with one small-workload row attributed to GCV scoring time. This is not enough evidence to justify another kernel or scheduler optimization pass.

Graph-value campaign:

- Modes: `legacy_mgcv`, `primary_only_cuda`, `hybrid_cuda`
- Repeats: 10
- Runs: 90 / 90 OK
- Warm-up: enabled
- Mode order: randomized

Pooled graph-value result:

- Corrected flips: 6
- Introduced flips: 1
- Median primary-vs-legacy `pMax` diff: 0.4738
- Median hybrid-vs-legacy `pMax` diff: 0.4430
- Median hybrid runtime / legacy runtime: 0.0686, about 14.6x faster than legacy
- First sepset mismatch rate: primary 0.233, hybrid 0.133

By scenario:

| scenario | corrected | introduced | median primary pMax diff | median hybrid pMax diff | median hybrid / legacy runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| chain-nonlinear-p5 | 1 | 1 | 0.2184 | 0.1827 | 0.0879 |
| fork-additive-p6 | 4 | 0 | 0.4935 | 0.4588 | 0.0874 |
| scale-nonlinear-p8 | 1 | 0 | 0.6340 | 0.5356 | 0.0127 |

Interpretation: hybrid has positive graph-value signal, but it is not universally monotone. `chain-nonlinear-p5` has one corrected and one introduced flip, so hybrid should remain opt-in rather than become an unconditional default.

## Orientation-Risk Validation

The current precision data plane covers skeleton execution. It does not yet route precision hybrid skeletons through the public `graph_stage = "wanpdag"` path. To validate whether the observed sepset and `pMax` changes are orientation-relevant, the orientation-risk validation re-ran the same synthetic scenarios and fed the legacy, primary-only, and hybrid skeleton/sepset objects into the same native WAN-PDAG orientation core.

Orientation-risk summary:

| scenario | primary PDAG identical rate | hybrid PDAG identical rate | median primary PDAG cell diff | median hybrid PDAG cell diff |
| --- | ---: | ---: | ---: | ---: |
| chain-nonlinear-p5 | 0.8 | 0.8 | 0.0 | 0.0 |
| fork-additive-p6 | 0.4 | 0.5 | 1.0 | 0.5 |
| scale-nonlinear-p8 | 0.3 | 0.3 | 4.5 | 3.5 |

Hybrid is neutral on `chain-nonlinear-p5` and improves median PDAG cell difference on `fork-additive-p6` and `scale-nonlinear-p8`. This supports the hypothesis that sepset and `pMax` improvements can matter for WAN-PDAG orientation, but it is not a production WAN-PDAG precision data-plane validation.

## Tail Gate Policy

Use a two-part tail gate:

- Small workloads: prioritize absolute p90 overhead. Current hybrid-v1 overhead of 29-90 ms is acceptable.
- Larger workloads: prioritize p90 ratio and total throughput.

Do not optimize solely because a small workload has a high ratio with a small denominator.

## Remaining Validation

Before changing defaults:

1. Run the same graph-value campaign on one or more real datasets.
2. Add stress cases known to produce near-alpha fastSpline-vs-mgcv flips.
3. Validate WAN-PDAG end-to-end once the precision data plane is explicitly connected to the WAN-PDAG stage.

Until then, hybrid remains accuracy-oriented opt-in.
