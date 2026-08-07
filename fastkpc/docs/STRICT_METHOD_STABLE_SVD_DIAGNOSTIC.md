# Strict-method stable-SVD diagnostic

## Scope

This development diagnostic evaluates exact single-GPU optimizations for the
authoritative fixed-SP stable-SVD route used by `hsic.gamma`, `dcc.perm`, and
`hsic.perm`. It does not evaluate `dcc.gamma` or change its Phase 10,
recommended-route, candidate, or sealed-holdout status.

The receipts below were produced from a dirty development worktree based on
`32cc75d`. They are local development evidence, not clean promotion evidence.
The retained runtime and reproduction-tool source closure was subsequently
committed, without further source changes, as
`41decffc5c210f08f4d50082aa98db9d57fd65ae`. This records the exact retained
implementation, but does not retroactively make the performance receipts a
clean-producer campaign.

## Dual-context SVD lanes: STOP

`tools/benchmark_fixed_sp_svd_lanes.R` uses authenticated real-data cohorts and
compares:

- A: two targets submitted serially through one CUDA runtime;
- B: one target per independent CUDA runtime, stream, cuSOLVER handle, cuBLAS
  handle, `gesvdjInfo`, and workspace, submitted by one host thread.

All coefficients, fitted values, residuals, RSS values, routes, statuses,
ranks, singular diagnostics, and aggregate-root diagnostics were bitwise
identical. Performance regressed:

| q | A completion p50 | B completion p50 | A / B |
| -: | ---------------: | ---------------: | ----: |
| 28 | 12 ms | 18 ms | 0.667x |
| 37 | 17 ms | 22 ms | 0.773x |
| 46 | 21 ms | 27 ms | 0.778x |

Nsight Systems found no overlapping kernel pairs between the two streams for
the q=46 cohort. Two host threads cannot repair absent device overlap, so the
dual-lane and two-host-thread variants stop here. A single cohort must remain
on one GPU.

Local receipts:

```text
/tmp/fastkpc_svd_lane_q28_100.rds
/tmp/fastkpc_svd_lane_q37_100.rds
/tmp/fastkpc_svd_lane_q46_200.rds
/tmp/fastkpc_svd_lane_q46_nsys.nsys-rep
```

## Cooperative aggregate factor: GO

The original `factor_fixed_sp_aggregate_penalty_kernel` used one CUDA thread
for penalty assembly, pivoted Cholesky, and root-factor construction. The new
single-block implementation parallelizes independent matrix elements and
columns while retaining the exact numerical order within each element:

- penalty terms are accumulated in their original order;
- pivot scans, pivot ties, and the outer `j` loop remain serial;
- each dot product retains its original row order;
- matrix swaps operate on disjoint elements and join at block barriers.

Five captured shapes (`q=28,37,46,55,64`) were saved from the old native
binary and compared across rebuilds. Coefficients, fitted values, residuals,
RSS, route/status, rank, singular diagnostics, pivots, factor-call counts,
B-build counts, and `dstop` were all bitwise identical.

For q=46, Nsight measured:

| kernel | old average | new average | speedup | old GPU share | new GPU share |
| ------ | ----------: | ----------: | ------: | ------------: | ------------: |
| aggregate factor | 1.249 ms | 0.113 ms | 11.1x | 16.8% | 1.8% |

`gesvdj` duration remained effectively unchanged. The fixed-SP augmented-SVD
qualification passed all 67 planned SVD cases, and the mixed-route test passed
its canonical Cholesky/SVD batch and resource-accounting gates.

The full 351 x 48 development runs were:

| method | previous best | cooperative factor | reduction |
| ------ | ------------: | -----------------: | --------: |
| `hsic.gamma` | 593.814 s | 556.256 s | 37.558 s (6.32%) |
| `dcc.perm` | 839.726 s | 792.100 s | 47.626 s (5.67%) |
| `hsic.perm` | 950.008 s | 905.157 s | 44.851 s (4.72%) |
| total | 2383.548 s | 2253.513 s | 130.035 s (5.46%) |

The corresponding residual host boundaries fell from
`383.800/452.323/429.871` seconds to `339.434/400.496/382.922` seconds.
All three candidate runners passed graph, sepset, `pMax`, count, trace,
authority, fallback, and cache-accounting gates. Both permutation methods kept
bitwise p-values and final `.Random.seed` states.

The production `kpcalg_authority` WAN-PDAG reruns also passed. They covered
6/11/2 orientation CI tests for `hsic.gamma`/`dcc.perm`/`hsic.perm`, with zero
authority p-value difference and identical final PDAGs. The experimental
native-CUDA orientation diagnostics retained their historical p-value
differences (`1.03e-4`, `0.0297`, and `0`, respectively) while also producing
identical final PDAGs; they remain non-authoritative.

The performance native SHA-256 was:

```text
00afa154dbbaae393ebbb54ac88c8ef198b917b62320514cbeb85d6b6a5711e2
```

These are single-run development values. They do not replace the existing
clean evidence or constitute repeated-median promotion evidence.

The final machine-readable completion audit binds the native binary, all three
method gates, WAN-PDAG authority receipts, dual-lane STOP, cache-capacity STOP,
and final q=46 cross-binary parity:

```text
path    /tmp/fastkpc_strict_single_gpu_completion_audit_20260807.rds
sha256  60a477751039906611cb82a89627a7fe6b5ff0eea634778e189391dc87a2e2de
pass    TRUE
```

## HSIC component-cache capacity: STOP

A temporary fail-closed diagnostic control swept the component-cache budget
while retaining the one-eighth-of-free-memory allocation cap. The production
source keeps its existing fixed 384 MiB budget; the unsuccessful sweep control
is not retained.

Increasing the requested budget to 1536 MiB produced 16,387 slots and reduced
component misses and cumulative component time, but did not reduce the outer
critical path:

| metric | 384 MiB | 1536 MiB |
| ------ | -------: | --------: |
| cache slots | 4,096 | 16,387 |
| misses | 323,541 | 253,963 |
| evictions | 234,199 | 152,330 |
| component time | 120.576 s | 90.548 s |
| outer elapsed | 905.157 s | 908.144 s |

A repeat of the default budget on the final diagnostic binary took 908.258
seconds. The two capacities are therefore equivalent within observed run
variation even though the larger cache saves about 30 seconds of cumulative
component work. That work is already hidden behind the residual/permutation
critical path. The 3072 MiB sweep and a larger production default are not
qualified.

The temporary cache-sweep binary SHA-256 was:

```text
8728d520fd02b9a99eb161652409c2a73e4946309162d113532dede825a195e1
```

## Remaining boundary

After the aggregate-factor change, cuSOLVER SVD kernels and their projections
dominate the direct-SVD device work. Ordinary dual streams do not overlap them,
`gesvdjBatched` is inapplicable to the augmented matrix shapes, and replacing
the solver or changing its tolerance would violate the strict numerical
contract. B rebuild, root emission, and CUDA launch overhead are too small to
justify further production refactoring from the current evidence.

CUDA Graph capture is not promoted to an implementation task. The profile
shows that the large host boundaries are chiefly queued-work boundaries in
`cudaMemcpyAsync`/cuSOLVER calls, while raw kernel-launch API time is small.
Graph capture would therefore add substantial library-capture and pointer
lifetime complexity without a demonstrated material opportunity.

For two RTX 4090 devices, use one complete method per GPU. Do not split one
cohort across devices. For a three-method campaign, the balanced schedule is:

```text
GPU 0: hsic.perm
GPU 1: dcc.perm, then hsic.gamma
```

This improves campaign throughput, not individual method latency, and should
be timed separately because both processes still share CPU and memory
bandwidth.

## Route closure

The strict-method single-GPU optimization goal is complete under the current
bitwise numerical contract:

```text
strict_method_single_gpu_optimization:
  status: COMPLETE
  termination: STOP_CURRENT_BITWISE_CONTRACT_CUSOLVER_GESVDJ_BOUND

accepted_current_best:
  hsic.gamma: 556.256 s
  dcc.perm:   792.100 s
  hsic.perm:  905.157 s
  total:     2253.513 s
```

The route may be reopened only when one of these conditions holds:

- a new CUDA/cuSOLVER release or GPU generation demonstrates a material
  full-route gain while preserving the current strict semantics;
- a new numerical-contract major version is explicitly authorized; or
- a separate multi-GPU execution goal is opened for complete-method task
  parallelism.

This closure does not change the `dcc.gamma` Phase 10 state, freeze a
candidate, alter the recommended route, or release the sealed holdout. A
future promotion candidate still requires a clean checkout, clean native
rebuild, and complete repeated campaign after the candidate commit is frozen.
