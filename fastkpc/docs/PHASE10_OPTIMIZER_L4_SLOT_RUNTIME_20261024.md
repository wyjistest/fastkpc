# L4 bounded-slot runtime qualification — 2026-10-24

Status: **GO_L4_BOUNDED_SLOTS_RUNTIME_DEVELOPMENT**. Capacity, native parity,
lifetime and all six full-canonical runs pass. Same-native OFF/ON three-run
outer medians are **463.288/407.531 seconds**, saving **55.757 seconds (12.04%)**,
with three paired wins. The option remains off by default. This is isolated
GPU1 source-closure development evidence from an audited incremental build,
not a clean-producer or promotion result. Phase 10 ACTIVE, candidate NOT_FROZEN,
holdout SEALED and recommended=false are unchanged.

## Capacity gate before runtime changes

Both capacity probes use the unchanged parent runtime `33e2d3e` and native
SHA-256 `2c8dd8af4ea021d8a784588717476f091a0132b9415067e4a3e115ca65bcd0ee`,
with canonical S2, L5 slots, L4 workers32, 11 GiB component cache, bypass and
eight eig lanes on GPU1. GPU0 and other compute processes remain idle. The
parent's 98-input source attestation is unchanged.

The first complete probe deliberately keeps the original L4 window handles
while adding 32 capacity-18 native prepared objects, detach events and another
5,259,032-byte coefficient arena. It passes all 55 canonical gates and checks
576 extra warm targets, but observes only **1,791,819,776 free bytes**. This is
a failed conservative capacity probe, retained as such; it double-counts the
handles the candidate is meant to replace. Its pool increment is 822,083,584
bytes. It does not establish replacement capacity or a candidate STOP.

The separately declared replacement probe runs at the entry to
`build_full_cuda_ci_level_components` for L4. All original multi-penalty window
handles have been released by `complete_multi_penalty_prefill_window`, while
real setup geometry, the coefficient arena and complete Stage A residual
storage remain resident. Level residual storage is allocated in `begin_level`;
`store_residual` copies into that allocation without growing it. The candidate
pool also ends before Stage B.

At that actual cache/arena boundary, the already qualified research DSO replays
the entire captured L4 layer through 32 capacity-18 slots. This includes the
larger, **105,038,504-byte** coefficients/fitted/residual diagnostic arena and
concurrent work, with allocation-boundary and 10 ms physical-memory observations.
All **2,154 setups / 17,767 targets / 216,194 iterations** and all output bits
match the capture. Minimum free memory is **2,586,640,384 bytes** (2.409 GiB),
and warm physical increment is **922,746,880 bytes** (880 MiB). Both unchanged
gates, >=2 GiB free and <=1 GiB increment, pass. The subsequent complete
canonical also passes all 55 gates, with no changed CI/pMax bits and maximum
authority error **3.05425129631942e-12**.

These are instrumented capacity diagnostics, not performance timings. Two
earlier harness attempts failed before slot measurement (R loader path and an
incorrect assumption that W1 contained an 18-target request); their logs are
preserved as well. See the [replacement capacity protocol](../tools/research/optimizer_l4_slot_runtime_20261024/REPLACEMENT_CAPACITY.md).

## Runtime implementation and build identity

`FASTKPC_DCC_GAMMA_L4_OPTIMIZER_SLOTS=1` selects the opt-in. It requires canonical
351x48 S2, L4 workers32, exact-primary coefficient reuse, status-wait enabled,
no optimizer trace and the original setup scheduling policy. The default is
off. L4 accepts only q37, four penalties, total root rank32 and the captured
target counts `2,3,4,5,6,7,8,10,11,12,14,16,17,18`.

There are at most 32 prepared slots, each with capacity18, and one original-order
setup queue. Each worker rebinds owned geometry, runs the existing optimizer,
detaches coefficients into the pre-reserved level arena, waits for its detach
event and releases its output lease before reuse. Stage A rebinds slot0 and
uses the unchanged <=16-target cohort split and original component order.
The pool is released before Stage B. L4 reports one actual optimizer boundary;
it does not fabricate the former 34 windows. L4 and L5 share scheduling and
cleanup machinery but retain separate geometry validators and native APIs.

Creation, post-work, every Stage A setup boundary and final pool close check
joint headroom. Pool-only warm increment is measured at creation and after
optimizer work. A failed rebind must be drained and discarded. Unsupported
configuration, live leases, capacity failure and CUDA errors fail closed.

New native SHA-256:

```
037b92f109a727ea2c50c737fb1c2b70048221bcffe41f215a532622398ffb58
```

The audited incremental build snapshots 99 source inputs, retains the parent's
verified objects and rebuilds seven dependent objects. The five changed inputs
are the L4 geometry header, capacity header/implementation, small64 host-entry
wrapper and one-call scheduling. No mathematical kernel source changed. This
is uncommitted source-closure development, not a full rebuild or clean-producer
receipt. The previous native and both build snapshots remain archived.

## Correct Stage A comparison boundary

The initial native test compared Stage A with unsplit optimizer shadows and
failed. A separate executable linked to the frozen parent binary demonstrates
the same difference: 17/18-target Stage A uses existing 9+8 and 9+9 cohorts,
changing 18,847 values relative to the unsplit optimizer, by at most
4.440892098500626e-15. The other 12 capacities and endpoint match their optimizer
shadows. This difference predates the candidate.

The corrected test compares optimizer outputs with all original capture bits
and Stage A with the parent's actual Stage A for identical cohorts. The ordinary
run passes **544 target-output comparisons**, **122 negative/lifetime checks**,
partial-upload and partial-detach cleanup, two-worker cancellation and resource
closure, with zero worker allocation/handle mutations. Memcheck repeats those
checks with **0 errors and 0 bytes leaked**. L4 slot, L5 slot and original L4
worker configuration rejection tests pass. Normalized instruction words match
for every GPU function in all three rebuilt CUDA objects; all retained object
hashes remain exact. No numerical tolerance was introduced. The completed
full-canonical campaign below also preserves every CI/pMax bit.

## Full-method protocol

Same binary and CPU affinity0–19, fresh R processes and dataset-specific caches;
sequence OFF/ON/ON/OFF/OFF/ON. Both arms retain L5 slots and L4 workers32. CPU
BLAS/OMP/MKL threads=1 and CUDA connections=32. Persistence and offline authority
comparison are outside the method timing boundary. Both GPUs' compute-process
lists and 200 ms telemetry are retained.

Hard gates require all CI/pMax bits equal the accepted reference and same-native
OFF, all authority errors <=1e-10, exact task/deletion/skeleton/sepset/count/RNG,
physical-group and Gram resolution order, unchanged other-layer windows/workers,
GPU-only authority, zero fallback/large production D2H, closed resources and
the declared joint L4 capacity. GO requires median outer ON/OFF <=0.98,
L4 optimizer <=0.90 and three paired outer wins. This is not a new real-data,
final-PDAG, holdout or promotion qualification.

## Full-method results

All six runs finish successfully on the same frozen source/native/tools. GPU0
has no compute job, and GPU1 has no unexpected compute process. The run order
is retained below; pair numbers identify the predeclared OFF/ON comparisons.

| Run order | Mode / pair | Outer seconds | L4 optimizer host seconds |
| --- | --- | ---: | ---: |
| 1 | OFF1 | 467.485 | 80.286883 |
| 2 | ON1 | 407.531 | 27.200660 |
| 3 | ON2 | 400.933 | 27.251857 |
| 4 | OFF2 | 463.288 | 80.323084 |
| 5 | OFF3 | 457.818 | 80.233663 |
| 6 | ON3 | 408.793 | 27.245752 |
| | OFF median | **463.288** | **80.286883** |
| | ON median | **407.531** | **27.245752** |

The outer ON/OFF ratio is **0.879649**, below the predeclared 0.98 limit.
The L4 optimizer ratio is **0.339355**, below 0.90: its measured host boundary
falls by **53.041 seconds (66.06%, 2.947x)**. Paired outer savings are
**59.954/62.355/49.025 seconds**, so all three pairs win. The historical
461.663-second median belongs to the previous runtime campaign; it is not
this experiment's OFF baseline.

| Recorded host boundary, median seconds | OFF | ON |
| --- | ---: | ---: |
| Total optimizer | 260.043729 | 203.983182 |
| Component Stage A | 16.555650 | 16.560536 |
| Component Stage B | 144.438083 | 144.442685 |
| Native setup | 22.861899 | 22.619050 |
| Multi-penalty prepared build | 12.526790 | 9.267360 |

The total optimizer boundary falls by **56.061 seconds**, consistent with the
**55.757-second** outer saving, while component stages remain stable. These
are recorded host boundaries, not isolated device time; stage medians and
nested prepared-build timing must not be added into an exact savings identity.
Other-layer timing variation is retained in the receipts. The qualified gain
is the complete bounded-refill/slot-reuse execution shape, not a claim that
allocation alone or all 53 seconds of L4 time is eliminated device work.

Each run preserves all **240,498 CI p-values and pMax bits** against the accepted
reference and same-native OFF, together with task identity/order, deletion
trace, skeleton, sepsets, counts and RNG endpoint. Physical cohorts, Gram
resolution order and other-layer window/worker settings remain exact. Maximum
CI and pMax error against the pinned authority is **3.05425129631942e-12**,
below 1e-10; the same maximum covers the 1,337 CI values in [0.05, 0.15],
with zero changed bits versus the reference. Exact-primary, stable-SVD,
cache metadata and level ledgers pass. CPU numerical authority, fallback and
large production residual/component D2H remain zero; tracked resources close.

Every ON run creates/destroys **32 slots**, executes **2,154 optimizer binds,
detaches and Stage A binds**, and covers **17,767 targets**. Across all ON runs,
minimum observed free memory during L4 pool residency is **2,699,296,768 bytes
(2.514 GiB)**, above 2 GiB, and maximum pool increment is **809,500,672 bytes
(772 MiB)**, below 1 GiB. All six sampled whole-run GPU1 peaks are **22,366 MiB**.
That whole-run peak is not the L4-active headroom measurement; the pool closes
before Stage B. The earlier full-layer prepared-residency STOP remains binding.

## Closure and evidence limits

The default-off L4 runtime option is retained as development GO for the
qualified canonical 351x48 S2 envelope, with L5 slots and L4 workers32 enabled.
This completes the capacity-to-runtime qualification step. It does not qualify
arbitrary datasets/shapes, newly run final PDAGs, a default change or promotion.
The build is an audited incremental build on uncommitted source over HEAD
`9bacecf78cd99a3683702d305248e4588475f279`; the receipts must not be relabeled as
coming from a future clean producer commit. The parent attestation and previous
receipts retain their original identities. Three ON repetitions are development
evidence, not the formal five-run performance gate.

The archiver verifies all 99 source inputs, frozen tools/dependencies, native,
run references, authorities and receipt hashes before issuing GO. Local evidence
is under `fastkpc/artifacts/full_cuda_ci/optimizer_l4_slot_runtime_20261024/`:

| Artifact | SHA-256 |
| --- | --- |
| `closure.json` | `e22309a9c76871a4872078a20e523b93a76cc3a6ed36544fb8464a956759a34f` |
| `manifest.json` | `71be9feb2326273ee531fb8f4263f6c1e7b78e74509e11b05d73bfcf8b8f41b0` |
| `build.json` | `53f7091d0fe94d71d7d79b8aae63ccb4506f4ece72555ec68dfb675be5066b3d` |
| `checks.json` | `209d390860099f2bef2715a85f35a2453d2d6e23395aeb6f279412988eb3731c` |

The manifest retains both capacity-probe outcomes, original failed test logs,
the corrected parent Stage A reference, ordinary/sanitizer checks, six receipts,
process isolation records and GPU telemetry. Nothing is promoted or enabled
by default; Phase 10 ACTIVE, candidate NOT_FROZEN, holdout SEALED and
recommended=false remain unchanged.

Final repository checks pass for tracked diffs, and all frozen source/native/
tool/reference hashes still match. The additional untracked-file whitespace
check reports an extra final blank line in the L4 geometry header and four
frozen research files (`campaign.py`, `coexistence_replacement.cpp`,
`coexistence_replacement.py`, `run.R`). Those bytes are retained with the tested
hashes; any formatting cleanup belongs to a separately attested revision.

## Repository handoff

Runtime commit `4a1ff6b19dce5d16845292c981a224f518d0de34` contains the qualified
default-off implementation and configuration tests. Research commit
`1c27d6f1ae0bf4c34ee447611bb2ae740994527e` preserves all three L4 diagnostic,
isolated-prototype and runtime toolsets byte for byte.

The [Git source attestation](PHASE10_L4_SLOT_SOURCE_ATTESTATION_20261024.json)
verifies all **99 runtime source inputs**, **44 archived tool entries** and
**six dependencies** against those commits. It binds the native SHA and original
manifest/closure/build/checks hashes without modifying the previous attestation
or any receipt. No build or GPU campaign was rerun for this handoff. References
to uncommitted source above describe the original run, not the current Git
state; the evidence remains precommit source-closure development.

Tools: [`optimizer_l4_slot_runtime_20261024`](../tools/research/optimizer_l4_slot_runtime_20261024/).
Local receipts: `fastkpc/artifacts/full_cuda_ci/optimizer_l4_slot_runtime_20261024/`.
The qualification campaign itself did not commit or push; this repository
handoff follows the separately authorized publication request.
