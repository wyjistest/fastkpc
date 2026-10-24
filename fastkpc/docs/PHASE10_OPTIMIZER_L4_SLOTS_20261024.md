# L4 32-slot bounded-refill isolated prototype — 2026-10-24

Historical experiment boundary: this report retains the isolated prototype's
original runtime identity and results. Subsequent cache/arena coexistence and
default-off integration are qualified in the [L4 runtime report](PHASE10_OPTIMIZER_L4_SLOT_RUNTIME_20261024.md),
with same-native full-method OFF/ON medians of 463.288/407.531 seconds.

Decision: **`GO_L4_BOUNDED_REFILL_ISOLATED_WARM_CONTEXT`**. Six alternating paired
runs reduce the captured L4 full-lifecycle median from **84.740 to 27.345 seconds**
(`3.099x`, **67.73%**), with six paired wins and all numerical/resource gates
passing. This is a research-only GPU1 result under the explicitly amended warm-
context capacity protocol below, not runtime integration or promotion evidence.
The original cold-context capacity STOP is preserved. No production source or
default changes; current canonical development median remains **461.663 seconds**.

## Compared execution and identity

Runtime remains `33e2d3e941eb25529b350474de713f7fbf741cc9`; native SHA-256:

```
2c8dd8af4ea021d8a784588717476f091a0132b9415067e4a3e115ca65bcd0ee
```

The existing source attestation binds all 98 production inputs. The research DSO
compiles the same small64 numerical implementation with hidden host entrypoints;
all **14 GPU functions' normalized SASS instruction words** match the production
object `24fd038c…414452`. Production numerical symbols are not exported from the
research DSO.

Input is the original L4 capture: **34 windows, 2,154 setups, 17,767 targets**,
`n=351`, `q=37`, four penalties, total root rank 32. Actual cohort sizes remain
`2,3,4,5,6,7,8,10,11,12,14,16,17,18`; no padding, splitting or sorting.

- **W:** the actual production `multi_penalty_gcv_capacity_optimize_multi`,
  32 workers and the original 34 windows. Only one window's prepared objects
  is resident. A host observer copies coefficients, fitted and residual outputs
  D2D before each original production batch returns.
- **S:** 32 independent slots of capacity 18, original-order atomic dispatch,
  geometry rebind/upload on each owning stream, then the unchanged optimizer.
  The output lease is released only after all three output arrays are saved.

Both arms retain the full **105,038,504-byte** output arena and defer diagnostic
shadow downloads/bitwise comparisons until the entire layer ends. Thus W is an
instrumented production baseline; neither timing is a full-method canonical time.
Lifecycle timing charges packing, pool/arena/events, request/result preparation,
uploads, execution, output retention, lease release and destruction. Fixture
reading, diagnostic shadow validation and CSV reporting are excluded.

The machine settings are GPU1, CPU affinity 0–19, CUDA connections 32,
status-wait enabled and one BLAS/OMP/MKL thread. Both GPU process lists and
200 ms memory/utilization/power/clock/temperature samples are retained. No other
GPU compute process was observed; GPU0 utilization stayed at zero. This is one
fresh C++ process with repeated captured-layer pairs, not fresh-process canonical
repeats or a new clean-checkout full-method campaign.

## Numerical and failure qualification

Ordinary reduced qualification covers the first captured request of all 14
capacities plus the layer endpoint: 15 setups / 136 targets, exercised through
W, sequential S and two-worker S. It compares all 14 floating vectors and 19
integer vectors in each capture, including selected SP, complete optimizer
trajectory counters, coefficients, fitted, residual, rank, route and solver info.

Busy/invalid rebind preserves generation and identity. Released tokens stay
stale across reuse, and an old token's destructor cannot release a new output.
Injected exceptions after partial H2D upload and after the first D2D output copy
poison/drain the slot. Parallel cancellation closes every lease and resource.
These are injected software failures, not simulated GPU hardware faults.

Reduced compute-sanitizer repeats all 14 capacities and the lifetime/failure/
cancellation checks: **zero access errors and zero bytes leaked**. No CUDA
allocation/free, stream/event create/destroy or BLAS-handle mutation is allowed
inside a worker interval. GPU-only authority, zero fallback, stable-SVD status
and final resource closure are mandatory throughout.

Final qualification includes **16 fully verified full-layer passes** (14 campaign
passes including warmups, plus two capacity attribution passes). With reduced
checks and the native warm control, **284,878 target-output comparison instances**
pass. Each full pass retains exactly **216,194 optimizer iterations**. The failed
original capacity attempt is separate and is not counted as a verified pass.
Diagnostic full-output D2H is confined to post-run research verification; no
production D2H behavior or downstream p-value/graph contract is newly qualified.

## Capacity attribution and the explicit protocol amendment

The original protocol set a <=1 GiB physical increment and >=2 GiB free gate.
Its first full-layer attempt **failed after computation** because it compared
against a CUDA context before the first optimizer execution. That failed attempt
and original protocol remain immutable; it did not complete shadow verification
and is not counted as a successful full-layer numerical receipt.

Two fresh-process attribution follow-ups complete full-layer output verification:

| Phase | Cold-context increment | After production optimizer warmup |
| --- | ---: | ---: |
| 32 slots created | 780 MiB | 778 MiB |
| Slots plus full output arena | 882 MiB | 880 MiB |
| Complete L4 computation | 2,032 MiB | **886 MiB** |
| After all tracked resources are released | 1,156 MiB | 10 MiB |

The warm control first executes the **actual production optimizer** on the
captured 18-target request, checks all output bits, and destroys its resources.
That alone leaves **1,152 MiB** resident in the CUDA context. Application device
allocation for the slot pool and arena is unchanged at **590,676,136 bytes**;
resource counters at execution are 33 allocations, 32 streams, 96 events and
32 BLAS handles, and all close to zero afterwards. The exact driver/library
allocation responsible for the retained context memory is not identified.

This establishes that the original cold measurement included a common first-use
cost. Before any balanced timings, a separate amended protocol was declared:
measure incremental pool cost relative to a compute-warm production-native
context, retaining the **same 1 GiB and 2 GiB limits**. The original replay binary
is unchanged and checks the increment both after pool creation and after work
on every S pass. Original cold STOP remains part of the evidence; it is not
retroactively marked PASS.

See [the pre-timing amendment](../tools/research/optimizer_l4_slots_20261024/CAPACITY_AMENDMENT.md).
This is **standalone warm-context capacity only**. Production component-cache,
downstream-arena and other-context coexistence remains unqualified. Common
cold-start memory and absolute process telemetry must still be budgeted.

## Balanced performance gate

One complete W and S warmup, then six alternating W/S and S/W pairs. The unchanged
performance gate requires median full-lifecycle S/W <=0.90 and six paired S wins,
alongside every numerical/resource gate. All gates pass under the amended
standalone capacity basis.

| Six-run median | W: original windows | S: bounded refill |
| --- | ---: | ---: |
| Execution boundary, including output retention | 80.413 s | 27.239 s |
| Prepared/pool and result preparation | 3.395 s | 0.049 s |
| Destruction/release | 0.916 s | 0.021 s |
| Candidate geometry packing | 0 s | 0.036 s |
| **Full lifecycle** | **84.740 s** | **27.345 s** |

The lifecycle S/W ratio is **0.322687**, a median difference of **57.396 seconds**.
All six paired savings are **57.184–57.843 seconds**. W lifecycle range is
84.568–85.012 seconds; S range is 27.169–27.384 seconds. Work-boundary S/W is
0.338744. These are measured captured-layer gains; the earlier fixed-service-time
tail model is not being substituted for a measurement.

The median summed per-setup optimizer host boundaries are 846.234 worker-seconds
for W and 839.470 for S. Numerical work and iteration counts remain identical;
the benefit is chiefly in scheduling the existing work and reducing repeated
prepared-object lifecycles. Summed host boundaries are not device time and must
not be added to the wall-time table.

Every timed S pass measures **876 MiB** physical increment above its warm context
(the independent fresh-process warm attribution measured 886 MiB), with tracked
application allocation fixed at 590,676,136 bytes. W's warm per-pass increment is
1,264 MiB. The whole campaign samples a maximum GPU1 usage of **3,254 MiB**;
this includes the baseline, context and diagnostic environment and is not a
production-cache coexistence measurement.

## Evidence and continuation boundary

Tools: [`optimizer_l4_slots_20261024`](../tools/research/optimizer_l4_slots_20261024/).
Local receipts: `fastkpc/artifacts/full_cuda_ci/optimizer_l4_slots_20261024/`.
The original protocol/build/checks/memcheck/failed capacity receipts are retained,
as are both capacity-attribution probes and the amended protocol declared before
the balanced campaign. `closure.json` and `manifest.json` bind all of them:

```
closure SHA256
bc3b0523f1b091a62e95aaaa932eb8c6086c145e9e7f5d9d85e75d308e1ce239

manifest SHA256
7b5c8a288bf7285c01da3d28d3a3cfdccc3d627ed4bb3b847d8334f787f0ebd6
```

The initial workflow in the immutable README reproduces the cold capacity stop.
Its follow-up commands are, in order, `build_capacity_diagnostic.py`,
`run_capacity_diagnostic.py`, `build_capacity_warm_diagnostic.py`,
`run_capacity_warm_diagnostic.py`, `run_revised.py`, then `archive_revised.py`,
each with the artifact directory as its argument. The original replay binary,
capacity guard and performance threshold remain unchanged.

The isolated performance gate passes. The next gate is production
cache/arena coexistence with >=2 GiB headroom, followed by a separate default-off
runtime qualification with complete canonical p-value/task/graph/RNG/resource
checks and repeated outer timings. Captured optimizer equality alone does not
qualify downstream consumers or graph decisions.

Current full-method development median remains **461.663 seconds**. L4 full-layer
prepared-residency STOP, production defaults, Phase 10 ACTIVE, candidate
NOT_FROZEN, holdout SEALED and recommended=false remain unchanged. No commit or
push is performed by this experiment.
