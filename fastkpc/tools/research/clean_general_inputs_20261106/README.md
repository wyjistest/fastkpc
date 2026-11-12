# Clean checkout and original-kpcalg qualification, 2026-11-06

Freeze producer `2be6a816d3039a5f9340ba7018166d737ab11d9a` in an independent
detached worktree. Restore the main workspace's original `kpcalg/R/dcovgamma.R`
byte for byte; retain both earlier snapshots and receipts. This campaign never
patches the reference function. Historical patched-R receipts remain historical.

Build every one of the 44 objects through the full build entry, with no copied
parent objects or incremental flags. Record all commands, sources, dependencies,
tool versions and the new native identity. Do not replace the main installation.

Seven full inputs: real 179x8, 243x9, 300x10; public 500x50; dense 500x8;
block-correlated 256x12; dense 1025x4. Defaults are genuinely omitted, resolving
to ranks 17/24/30/50/50/25/102. All searches use default Inf/natural stopping.
The dense 500x8 case must reach level 6 and the dense 1025x4 case must exercise
the producer outside structured capacity. Seeds and input construction are
frozen in `prepare.R` before any results are inspected.

The CPU authority is fresh pcalg stable search with the unmodified R kpcalg,
mgcv and RSpectra functions, followed by original WAN-PDAG orientation. Native
host scheduling replays independently obtained R p-values to check native task
and deletion order. Raw pcalg visits nodes in a different order; do not label
that raw order exact. Speculative native tasks receive independent R checks.

Every logical CI and finite pMax must agree within 1e-10, with no decision flips.
Require task identity/order, deletion trace, graph, semantic sepset unions, raw
n.edgetests, orientation CI identity/order/values, final PDAG and RNG endpoints.
GPU skeleton CPU/unknown authority and residual/component large D2H remain zero;
stable-SVD, cache metadata and all CUDA resource closure checks must pass.

First run OFF and ON correctness jobs on all seven inputs. Only after all pass,
run fresh-process timing in order A1/C1/C2/A2/A3/C3 for each input. A is ordinary
exact-primary/coefficient-reuse/retained GPU; C additionally enables the current
prefill/structured/slot/work-pair/postback combination. The properties cache is
enabled in both. Retain 11 GiB cache, bypass, eight requested eig lanes, physical
GPU1, CPUs 0-19, one math-library thread and 32 CUDA connections.

Timing processes warm only an unrelated 80x3 input, then reset dataset caches
and the seed before the measured input. Primary timing is the whole public
WAN-PDAG call; skeleton and CPU orientation are also reported separately.
Artifact reads/writes, hashes and comparisons stay outside the timed call.
Both GPUs must be free of external compute for accepted performance runs.
Correctness only requires an exclusive selected GPU; record other-GPU use.

Report each input independently. Predeclared outer non-regression: C/A median
<=1.05. Speed is proven only if C/A<=0.95 and the slowest C is faster than the
fastest A. Otherwise report NOT_PROVEN or REGRESSION; never change thresholds
after results. The three-run per-mode development comparison is not promotion.
No source changes, new optimization, candidate freeze or holdout access is part
of this campaign. Native/data/large receipts remain local.

## Completed correctness, timing and final closure

`references1`, `qualification1` and `supplement4` pass on clean full-build native
`d8793173b00c89d971f9210ae4f7020f7bef786dab5e48ec43dfc8fc310d7350`.
`correctness_closure1` seals their independent audit. The streaming driver
`qualify_ready.py` processed completed references first, then dense500, with
each original-R receipt hashed before replay or GPU execution. It uses the
same comparisons and OFF/ON settings as the original campaign driver.

The November 6 correctness closure records an external GPU0 task that prevented
timing at that date. It remains an unchanged historical receipt. `timing1`
subsequently completed all 42 fresh processes on November 10, 09:51–15:06
Asia/Shanghai. All process records pass the isolation check. All seven inputs
meet outer non-regression; only real300 meets the speed gate (3.353/2.893 seconds,
13.72% less elapsed time). The other six remain `NOT_PROVEN`.

The dense500 reference performs 18,000 CPU orientation CIs. Its timed whole-call
OFF/ON medians are 2798.063/2797.599 seconds, while skeleton medians are
14.362/11.146 seconds. CPU orientation dominates this input. Per-stage medians
are not an additive account of one run; these are two GPU configurations, not
a CPU-vs-GPU speed comparison.

Before the first timing run, the warmup was corrected to use the full-CUDA
WAN-PDAG entry on the unrelated 80x3 data. The skeleton facade defaults to the
legacy route when no route is supplied. Before that fix, no timing had executed
the old branch; all runs completed at that point used `phase=correctness`. The runner also checks the
loaded native path. Earlier stages retain their exact harness snapshots and
hashes; the final audit verifies those snapshots instead of relabeling them.

The following commands produced the completed timing and final closure:

```sh
python3 fastkpc/tools/research/clean_general_inputs_20261106/campaign.py timing timing1 qualification1
python3 fastkpc/tools/research/clean_general_inputs_20261106/audit.py references1 qualification1 timing1 supplement4 closure1
```

`closure1` completed on November 12. The auditor rechecks frozen source/object/
generated/native identities, prior closure hashes, saved per-stage harnesses,
process order/isolation, and all 56 raw qualification/timing receipts. The new
`check_final_receipts.R` only reads saved RDS files and pure comparison helpers;
it checks original-R p-values, graph/order/RNG, orientation, capacity, resources
and raw timing against the reports without loading or running the native.
All 63,056 CI and 146,576 orientation comparisons pass. The final manifest
contains 505 evidence files, including snapshots of the final audit tools.

To independently audit existing results again, use a new output tag (the tool
refuses to overwrite an existing directory):

```sh
python3 fastkpc/tools/research/clean_general_inputs_20261106/audit.py references1 qualification1 timing1 supplement4 closure_review
```

No timing rerun is needed for that audit. A new timing campaign requires fresh
tags and the original isolation/order/thresholds. See the
[report](../../../docs/PHASE10_CLEAN_GENERAL_INPUTS_20261106.md) and
[final evidence index](../../../docs/PHASE10_CLEAN_GENERAL_INPUTS_ATTESTATION_20261112.json).
The original correctness attestation and each run's harness snapshot preserve
their original identities; the final audit does not relabel them.
