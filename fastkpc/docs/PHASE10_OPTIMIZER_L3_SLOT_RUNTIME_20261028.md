# L3 bounded-slot runtime qualification — 2026-10-28

Native qualification and all six isolated canonical runs **PASS**. The
predeclared gate gives **GO_L3_BOUNDED_SLOTS_RUNTIME_DEVELOPMENT**.
Same-native OFF/ON medians are **370.553/329.078 seconds**, saving
**41.475 seconds (11.19%)**, with three paired outer wins. L3 optimizer medians
fall **50.160702/15.249473 seconds (69.60%)**.

The accepted opt-in development median becomes **329.078 seconds**; the prior
L6 result of 368.851 seconds is historical. Use this campaign's 370.553-second
OFF median for the L3 comparison. This remains audited incremental source
closure, not a clean producer or promotion. Phase 10 ACTIVE, candidate
NOT_FROZEN, holdout SEALED and recommended=false remain unchanged.

## Runtime and capacity scope

The default-off `FASTKPC_DCC_GAMMA_L3_OPTIMIZER_SLOTS` requires canonical S2,
n=351/q=28/three penalties/root rank=24, exact-primary selected-coefficient refresh,
no decomposition trace, status wait, properties cache and L4/L5/L6 slots.
All original cohorts of 2–33 targets remain unpadded. Thirty-two capacity-33
slots serve one original-index queue; completed selected coefficients detach
into the actual reserved coefficient arena before the slot is reused. Stage A
uses the original target/cohort order and closes the pool before Stage B.
Unsupported explicit opt-ins fail closed. There is no new GPU warmup, preload,
stack-limit change, full-layer diagnostic arena or numerical algorithm change.

The [v2 coexistence qualification](PHASE10_OPTIMIZER_L3_COEXISTENCE_V2_20261027.md)
is a prerequisite, with its split raw capacity contract retained:

| Checkpoint | Hard gate |
| --- | --- |
| After actual coefficient arena reserve, before pool creation | Original raw baseline |
| Creation complete, before any L3 launch | Increment <=1.5 GiB |
| Optimizer complete, relative to creation complete | Growth <=1.25 GiB |
| Optimizer complete, relative to original baseline | Total increment <=2.75 GiB |
| Creation, optimizer completion, each Stage A setup entry and before close | Free >=2 GiB |
| Stage A raw total increment | Record only; no 2.75 GiB ceiling |

No model subtraction or baseline reset is allowed. The runtime records these
checkpoints only, without the diagnostic 10 ms sampler; this does not prove
continuous peak headroom. Existing L4/L6 <=1 GiB pool-increment and >=2 GiB
free-memory guards are unchanged. Downstream free-memory comparisons are
descriptive, not a new noninferiority gate.

Pool creation has a separate `pool_create_ms` timer within total `prepare_ms`.
The old `prepared_build_sec` counter covers only original per-request handle
construction and excludes slot pools, so its decline cannot establish net
creation savings. No optimizer shadow is included in the performance campaign.

## Source closure and native qualification

Frozen parent native:
`313d9f3cc6dc8512b5624aae84cc459c6083ff04fd7883c32812fdb081510a1b`.
New native:
`e8c39703f01e09260d330167b49267c295aa2221167d5248d0980a97c80ca045`.

The audited incremental build binds **103 production inputs**: four changed
files and two new L3 headers relative to the frozen L6 source closure. Seven
objects are rebuilt and relinked; other objects remain identical. All **36 GPU
functions** in the three rebuilt CUDA objects retain identical normalized
instruction words. This is source-closure evidence, not a clean checkout or
full rebuild claim.

Captured qualification covers every actual cohort size 2–33 and the final
request: **33 requests**, **2,272 optimizer/Stage A target comparisons** and
**267 negative/lifecycle checks** per run. Optimizer outputs match capture;
coefficients/fitted/residual/rank/route/info match the frozen parent's actual
Stage A finalization. The original Stage A cohort splitting is preserved.

Initial allocation rejection, failure during creation after arena allocation,
partial rebind uploads, coefficient detach failure, stale generations, old
token destruction, busy/invalid rebind and worker cancellation/drain pass.
Worker resource mutations are zero; tracked allocations, bytes, streams,
events and BLAS handles close. Memcheck repeats the qualification with
**zero errors and zero leaked bytes**. Ten L3 configuration rejections,
unset-option default-off behavior and existing L4/L5/L6 configuration tests pass.

The first check attempt stopped at a test assertion: a 200,000-byte
limit rejected the prepared object's single arena before allocation, so it did
not simulate partial creation. The corrected test injects failure on the third
setup H2D after allocation. Original logs, binaries and test source remain in
`checks/`; successful checks are in `checks-attempt2/`. Production source,
native and numerical thresholds were unchanged by this test correction.

## Predeclared isolated performance protocol

Same native, six fresh R processes, fixed order
**OFF1/ON1/ON2/OFF2/OFF3/ON3**. Only L3 slots change; retained S2 settings,
L4/L5/L6 slots, properties cache, status wait and coefficient reuse stay on.
GPU1 exclusive, both GPUs otherwise idle; CPU affinity 0–19, one BLAS/OMP/MKL
thread, CUDA connections 32. Process monitoring covers both GPUs; any external
interference preserves the interrupted attempt and signals only our process.
Completed unfavorable timings are never discarded. Persistence and offline
saved CPU-authority comparison stay outside outer elapsed; numerical execution
remains GPU-only. No diagnostic shadow or decomposition trace is timed.

Required: all 240,498 CI/pMax bits, task/deletion/skeleton/sepset/count/RNG,
physical cohorts/Gram order, optimizer counters and other-layer scheduling
unchanged; all finite authority errors <=1e-10; zero CPU numerical authority,
fallback or large production D2H; all resource and capacity gates pass.

Performance GO requires ON/OFF outer median <=0.98, L3 optimizer median ratio
<=0.90 and all three paired outer wins. Smaller positive/inconsistent gains are
CONDITIONAL; no median gain or any hard-gate failure is STOP. Tools and these
thresholds were frozen before the first run.

## Six isolated runs and correctness

All six runs completed in the declared order without interruption or discarded
timings. GPU1 had no other compute process and GPU0 had no compute task in the
recorded monitoring. OFF1 compares against the frozen L6 producer; subsequent
runs compare against same-native OFF1, with repeated-mode bitwise checks added.

| Run | Outer s | L3 optimizer s | L3 layer s | L3 remainder s | Outside L3 s | Hard gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OFF1 | 370.553 | 50.362108 | 123.322068 | 72.959961 | 247.230932 | 65/65 |
| ON1 | 328.165 | 15.248548 | 84.512943 | 69.264396 | 243.652057 | 68/68 |
| ON2 | 331.547 | 15.264530 | 84.691295 | 69.426766 | 246.855705 | 71/71 |
| OFF2 | 369.124 | 49.319922 | 122.201844 | 72.881922 | 246.922156 | 68/68 |
| OFF3 | 377.543 | 50.160702 | 123.354361 | 73.193659 | 254.188639 | 68/68 |
| ON3 | 329.078 | 15.249473 | 84.815384 | 69.565910 | 244.262616 | 71/71 |

Paired outer savings are **42.388 / 37.577 / 48.465 seconds**. Outer median
ratio **0.8880726914 <=0.98**, L3 optimizer median ratio
**0.3040123594 <=0.90**, and **3/3 paired wins** all pass the original gate.
OFF3's slower elapsed remains in the complete record and the comparison.

Every run preserves all **240,498 CI p-value and pMax bits** with zero changed
values or alpha flips. Task/deletion/skeleton/sepset/count/RNG, physical cohort
and Gram order, optimizer iteration counters and other-layer scheduling pass.
Maximum CI and pMax error against saved authority is
**3.05425129631942e-12**, below 1e-10. CPU numerical authority, unknown/CPU
fallback and residual/component large D2H remain zero; tracked live resources
return to zero. These checks cover the complete skeleton, without claiming a
new final-PDAG or real-data qualification.

Each ON uses **32 created / 32 destroyed slots**, 4,065 optimizer binds,
coefficient detaches and Stage A setup binds, and **54,878 targets**. Peak
workers are 32. OFF retains all 64 original windows. There is no diagnostic
shadow asymmetry in this campaign.

| Boundary, median s | OFF | ON |
| --- | ---: | ---: |
| Full method | 370.553 | 329.078 |
| L3 layer | 123.322068 | 84.691295 |
| L3 optimizer boundary | 50.160702 | 15.249473 |
| L3 remainder | 72.959961 | 69.426766 |
| Other layers and outside all layers | 247.230932 | 244.262616 |
| Total optimizer host boundary | 168.961419 | 128.915171 |
| L8 optimizer, unchanged code | 20.224763 | 16.903389 |

Each run's L3 remainder is its layer elapsed minus optimizer elapsed; outside
L3 is outer minus the layer. Medians of these separate quantities need not
sum. The optimizer boundary is elapsed host work, not pure device kernel time;
ON includes worker rebind and coefficient detach. L3 remainder changes also
include preparation/completion scheduling. Other-layer differences, including
the displayed L8 variation, remain in outer and are not attributed to L3 slots.
This result does not reopen the existing L8 STOP.

Actual ON pool creation is **8.617417 / 8.836075 / 14.139400 ms**, within total
slot preparation **498.528557 / 500.656468 / 507.313599 ms**. The median old
whole-method prepared-build counter falls 1.405002→0.285866 seconds but excludes
slot construction; it is not reported as net creation savings.

## Runtime capacity and downstream headroom

All three ON runs have identical raw byte observations:

| Measurement | Bytes | Gate |
| --- | ---: | --- |
| Baseline after actual arena reserve | 5,829,165,056 free | Original baseline |
| Creation complete | 4,610,719,744 free | Before any L3 launch |
| Optimizer complete | 3,404,857,344 free | Original baseline retained |
| Creation increment | 1,218,445,312 | <=1,610,612,736 |
| Optimizer growth since creation | 1,205,862,400 | <=1,342,177,280 |
| Optimizer total increment | 2,424,307,712 | <=2,952,790,016 |
| Stage A maximum recorded raw increment | 2,424,307,712 | Record only |
| Minimum L3 checkpoint free | **3,404,857,344 (3.171021 GiB)** | >=2,147,483,648 |

Each ON records **4,068 capacity checkpoints**: creation, optimizer completion,
4,065 Stage A setup entries and before close. The latter 4,066 form the Stage A
checkpoint group. They are not a continuous peak proof. The three increment
measurements match the prerequisite v2 values without subtraction or reset;
absolute free readings belong to this new native/run and are not substituted
with the older receipt's free values.

| Downstream measurement | Every OFF, bytes | Every ON, bytes |
| --- | ---: | ---: |
| L4 minimum checkpoint free | 2,685,534,208 | 2,689,728,512 |
| L5 free after pool creation | 2,358,378,496 | 2,362,572,800 |
| L6 minimum checkpoint free | 2,253,520,896 | **2,257,715,200** |

L4/L6 increments remain 809,500,672 / 473,956,352 bytes, within their unchanged
1 GiB gates. ON is 4 MiB higher at each displayed downstream checkpoint; this
is descriptive, with no post-hoc relative gate or causal attribution. The
tightest L6 ON checkpoint has **2.102661 GiB free**, only **105.125 MiB** above
its hard floor (OFF 101.125 MiB). This does not qualify a larger envelope.
External 200 ms telemetry observes whole-GPU peaks 22,083 MiB OFF /22,081 MiB ON;
these coarse observations are separate from the runtime raw-capacity checks.

## Evidence and limits

Tools: `fastkpc/tools/research/optimizer_l3_slot_runtime_20261028/`.
Evidence: `fastkpc/artifacts/full_cuda_ci/optimizer_l3_slot_runtime_20261028/`.
`build.json`, `checks.json`, `protocol.json`, `closure.json`, per-run receipts
and `manifest.json` bind source, native, tests, references and environment.
The manifest contains 243 archived files, 103 production inputs, 11 tools,
six direct dependencies and 12 references; native-check fixture/audit dependency
hashes are also bound in `checks.json`. Failed pre-campaign test evidence is
retained alongside the successful attempt.

| Receipt | SHA-256 |
| --- | --- |
| build.json | `12258db0431627ec6cf689f03cbf582f5a343d8d14dc8580ce28245b7afedfdd` |
| checks.json | `e3490444de5e1f3ea369ca9ba29f0b2736b62f587666afe01b1349eee2cb074c` |
| protocol.json | `2296e27fc7512ef21db14f9dc2dee34f77b8d4ab6a2474c7a791d4dc81773a51` |
| closure.json | `082abc12aeaf919ffc2113d66c49a0b1edceaf19b4fb0044eb3c25a1e82671ad` |
| manifest.json | `08926be1d07ceecbfe64d30ccb4eff83591634adc20573d78dbc7a5ccbb06ded` |

The current decision is runtime integration/qualification complete and
**development GO**, with accepted opt-in median 329.078 seconds. No new
real-data/final-PDAG campaign, clean-producer rerun or promotion is claimed.
The option stays default-off; Phase 10 ACTIVE, candidate NOT_FROZEN, sealed
holdout and recommended=false remain unchanged. Keep the original
STOP_L3_REPLACEMENT_POOL_INCREMENT under its old contract and the captured/v2
coexistence GOs as historical evidence. Further work must follow a review of
the new complete profile; this result does not authorize a new L8 experiment.
All own benchmark/monitor processes have ended. No commit or push was performed.
