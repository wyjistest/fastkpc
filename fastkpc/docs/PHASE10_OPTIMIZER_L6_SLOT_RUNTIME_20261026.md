# L6 bounded-slot runtime qualification — 2026-10-26, completed 2026-10-27

Replacement coexistence: **PASS_REAL_CACHE_ARENA_REPLACEMENT_COEXISTENCE**.
Native runtime checks and all **six isolated canonical runs**: **PASS**.
The unchanged predeclared performance gate gives
**GO_L6_BOUNDED_SLOTS_RUNTIME_DEVELOPMENT**. Same-native OFF/ON medians are
**395.741/368.851 seconds**, saving **26.890 seconds (6.79%)**, with three paired
outer wins. L6 optimizer medians are **42.173691/18.006620 seconds**, down
**57.30%**. The accepted opt-in development median is now **368.851 seconds**;
the preceding device-properties result of **392.416 seconds** is historical.
Use this campaign's 395.741-second OFF median for the L6 comparison.
The October 26 interruption and original evidence are preserved; the October 27
continuation completed only OFF3/ON3, with zero completed timings discarded.
This is incremental source-closure development evidence. Phase 10 ACTIVE,
candidate NOT_FROZEN, holdout SEALED and recommended=false remain unchanged.

## Replacement capacity before runtime changes

The frozen parent is cache native
`994c43d386c859f23c34509b857db22064fb9a009b8fe871c719dd608f820df9`.
The probe runs canonical 351x48 S2 with properties cache, L4/L5 slots, status
wait, 11 GiB component cache, all-hit bypass and eight eig lanes on exclusive
GPU1; GPU0 is idle. It does not change production sources or numerical code.

At actual L6 `build_full_cuda_ci_level_components` entry, weak references prove
that all **245 original prepared handles have expired**. Real setup geometry,
the completed coefficient arena and Stage A residual storage remain resident.
The already qualified captured DSO replays all **245 setups / 1,148 targets /
17,062 iterations** through 32 capacity-5 slots in original request order.
Its 14 floating and 19 integer output fields are bitwise exact. Original
two/three/five-target cohorts are preserved.

The extra diagnostic coefficient/fitted/residual arena is **6,952,288 bytes**,
larger than the eventual production coefficient payload. Diagnostic shadow
D2H is accounted separately and is never numerical authority. Memory is sampled
every 10 ms and at allocation/free boundaries through pool destruction.

| Measurement | Bytes |
| --- | ---: |
| Free before the probe | 2,734,489,600 |
| Free after pool creation | 2,252,144,640 |
| Minimum free, including work and destruction | **2,245,853,184** |
| Warm physical increment | **488,636,416 (466 MiB)** |
| Free after close | 2,728,198,144 |

The unchanged >=2 GiB / <=1 GiB gates pass, with minimum headroom **2.092 GiB**,
only **93.8125 MiB above the free-memory threshold**. The earlier approximate
2.6 GiB prediction is not used. The after-close physical difference is 6 MiB;
it is retained in the accounting, with no claim that every physical byte
returns. Application allocations, streams, events and BLAS handles return
exactly to their pre-probe counts; worker resource mutations and live leases
are zero. There are 1,941 physical-memory observations.

The subsequent full canonical passes all original hard gates: 240,498 CI/pMax
bits, task/deletion/skeleton/sepset/count/RNG, authority and resource closure.
Maximum CI and pMax authority error remains `3.05425129631942e-12`. The instrumented
417.523-second outer boundary is diagnostic only, not performance evidence.

Protocol: [REPLACEMENT_CAPACITY.md](../tools/research/optimizer_l6_slot_runtime_20261026/REPLACEMENT_CAPACITY.md).
Evidence: `fastkpc/artifacts/full_cuda_ci/optimizer_l6_slot_runtime_20261026/coexistence-replacement/`.

## Runtime scope and predeclared gate

The default-off `FASTKPC_DCC_GAMMA_L6_OPTIMIZER_SLOTS` uses the qualified canonical
q55/six-penalty/root-rank48 envelope, 32 capacity-5 slots and one original-order
queue. L4/L5, properties cache, exact-primary coefficient reuse and status wait
remain enabled. Unsupported configurations fail closed. No L8 preload or
general GEMM warmup is introduced.

The incremental runtime native is
`313d9f3cc6dc8512b5624aae84cc459c6083ff04fd7883c32812fdb081510a1b`.
It binds 101 production source inputs on HEAD `d27e59b`, with four changed
inputs and one new L6 header relative to the frozen cache parent. Seven affected
objects were rebuilt; retained objects are exact. All **36 device functions**
in the three rebuilt CUDA objects retain identical normalized instruction words.
This establishes device-code identity, not a claim of a clean/full rebuild.

Each slot retains its own geometry/workspace and completion event. A worker
rebinds, runs the unchanged optimizer, detaches final selected coefficients into
the reserved level arena, waits for detach and releases the output lease before
reuse. Stage A consumes the original cohorts and original target order. The
pool closes before Stage B. Runtime checks joint free memory and pool increment
through its actual optimizer and Stage A lifetime; the probe does not replace
those checks.

Native qualification passes all three target capacities and the final request:
**52 optimizer/Stage A target comparisons and 34 negative/lifetime checks per
run**, both normally and under memcheck. Stage A is compared to actual frozen-
parent finalization. Stale generations, old-token destruction, invalid geometry,
partial-upload/detach failures and cancellation/drain pass; worker resource
mutations are zero and tracked resources close. Memcheck reports **zero errors
and zero leaked bytes**. The new L6 configuration tests and existing L4/L5 slots
and L4-worker configuration tests all pass.

GPU0 became occupied by an unrelated training process after native checks began.
Before either supplemental run, the protocol added one separate OFF/ON pair on
exclusive GPU1 for complete correctness/resource qualification. Its `shared-*`
artifacts explicitly identify the shared host and cannot supply an accepted
performance median or contribute timings to the isolated campaign. The isolated
six-run gate below is unchanged.

## Shared-host canonical correctness and capacity

Both fresh-process runs use the same `313d9f3c…510a1b` binary, GPU1, CPU affinity
0–19 and one BLAS/OMP/MKL thread. Only L6 slots change; properties cache, L4/L5
slots, status wait, coefficient reuse and retained S2 component settings stay on.
The OFF run passes **60 gates**, including comparison with the frozen parent.
The ON run passes **61 gates**, including same-native identity with OFF.

- All **240,498 CI p-values and pMax bits** are identical, with zero changed
  values and zero decision flips. Maximum CI/pMax authority error remains
  `3.05425129631942e-12`, below `1e-10`.
- Task/deletion/skeleton/sepset/count/RNG, physical cohorts, Gram order,
  optimizer iteration totals and other-layer scheduling remain exact.
- CPU numerical authority, fallback and large production D2H remain zero;
  tracked live allocations/bytes/streams/events return to zero.
- ON uses **32 created / 32 destroyed slots**, 245 optimizer rebinds, 245
  coefficient detaches and 245 original-cohort Stage A rebinds, covering all
  **1,148 L6 targets**. Peak workers are 32; original OFF uses four windows
  containing 64/64/64/53 setups.

Minimum checkpoint-observed L6-active free memory is **2,260,533,248 bytes (2.105 GiB)**,
**107.8125 MiB above** the 2 GiB gate. Maximum measured pool increment is
**473,956,352 bytes (452 MiB)**, below 1 GiB. These runtime optimizer/Stage A
checkpoints are distinct from the sampled replacement probe above; they do not
provide continuous peak monitoring. The margin remains
narrow; the result does not qualify a larger or more general dataset envelope.

The following single-pair timings are retained only as shared-host diagnostics:

| Boundary | OFF seconds | ON seconds |
| --- | ---: | ---: |
| Complete method | 391.423 | 376.372 |
| L6 optimizer | 39.405033 | 18.082080 |
| L8 optimizer, unchanged implementation | 16.694126 | 22.054300 |

The observed outer reduction is 15.051 seconds; the L6 reduction is 21.323
seconds. Other stages differ, including the displayed L8 change. This pair
neither establishes a repeated median nor attributes those differences. GPU1
remains exclusive throughout both runs; GPU0's external process is explicitly
recorded. These timings will **not** be pooled with the isolated campaign.

## Isolated performance: six complete, original gate passed

Six same-native fresh-process runs were declared in order
OFF1/ON1/ON2/OFF2/OFF3/ON3, both GPUs otherwise idle. Required: all original
canonical gates, outer ON/OFF median <=0.98, L6 optimizer median ratio <=0.90,
and three paired outer wins. Timings include the complete method; artifact
persistence and offline authority checks remain outside the timer. The
[protocol](../tools/research/optimizer_l6_slot_runtime_20261026/README.md) keeps
all attempts and declares GO/CONDITIONAL/STOP before performance runs.

Both GPUs became idle and the campaign started on GPU1 at 15:46 on October 26.
The first four runs completed in the original order, with both GPUs otherwise
free of compute jobs throughout each run. All numerical/resource gates pass:
60/61/64/63 gates respectively, including repeated CI/pMax bit identity where
applicable. On October 27, the audited continuation completed OFF3/ON3 with
**63/63 and 64/64 gates** passing, respectively. All six runs preserve every
one of the 240,498 CI p-values and all pMax bits,
task/deletion/skeleton/sepset/count/RNG, original cohorts and Gram order,
optimizer trajectory counters, authority and resource closure. CPU numerical
authority, fallback and large production D2H remain zero. Maximum authority
error remains `3.05425129631942e-12`.

| Run | Outer s | L0–L5 s | L6 s | L7 s | L8 s | Outside levels s | L6 optimizer s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OFF1 | 400.084 | 317.053 | 46.018 | 9.994 | 25.212 | 1.807 | 43.538 |
| ON1 | 375.336 | 318.520 | 20.390 | 17.588 | 16.997 | 1.842 | 18.007 |
| ON2 | 368.851 | 317.563 | 20.417 | 11.972 | 17.060 | 1.839 | 18.055 |
| OFF2 | 395.741 | 317.424 | 44.677 | 14.844 | 16.927 | 1.869 | 42.174 |
| OFF3 | 395.684 | 318.106 | 43.721 | 11.382 | 20.610 | 1.865 | 41.268 |
| ON3 | 368.352 | 317.134 | 20.304 | 12.117 | 16.954 | 1.843 | 17.985 |

L6 optimizer is a sub-boundary of L6, not another term to add to outer. Outside
levels is measured as outer minus the sum of level boundaries. Values above are
rounded; receipts retain the full precision. All three paired changes are
accounted for below; positive values mean ON is faster:

| Pair | Outer saving s | L0–L5 s | L6 s | L7 s | L8 s | Outside levels s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OFF1/ON1 | 24.748 | -1.467 | 25.628 | -7.594 | 8.215 | -0.035 |
| OFF2/ON2 | 26.890 | -0.139 | 24.260 | 2.873 | -0.134 | 0.031 |
| OFF3/ON3 | 27.332 | 0.972 | 23.417 | -0.735 | 3.656 | 0.022 |

L7 and L8 differences remain included in outer; no cause is assigned without a
trace and no completed result is excluded as noise. The original acceptance
gate passes: outer median ratio **0.932052 <= 0.98**, L6 optimizer median ratio
**0.426963 <= 0.90**, and **3/3 paired outer wins**. No thresholds changed.

| Boundary | OFF median s | ON median s |
| --- | ---: | ---: |
| Complete method | 395.741 | 368.851 |
| L6 optimizer | 42.173691 | 18.006620 |
| Total optimizer | 194.440527 | 167.215938 |
| Stage A | 14.505112 | 14.547472 |
| Stage B | 144.228853 | 144.833722 |
| Native setup | 22.692468 | 22.668788 |

These are independently calculated boundary medians, not additive terms. The
accepted opt-in canonical development median is **368.851 seconds**. It is not
a clean-producer repeated median or performance-promotion result.

All three isolated ON runs record minimum checkpoint-observed L6-active free memory of
**2,260,533,248 bytes (2.105 GiB)** and maximum pool increment
**473,956,352 bytes (452 MiB)**. All 32 slots close; all 245 optimizer binds,
detaches and original-cohort Stage A binds complete, and tracked resources
return to zero. These identical observations demonstrate repeatability in the
measured configuration, not deterministic future headroom or continuous peak
coverage. The margin above the 2 GiB gate remains **107.8125 MiB**; any future
increase in resident memory must be requalified at L6. Whole-run 200 ms GPU
samples peak at **22,373 MiB OFF / 22,077 MiB ON**, a different boundary from
the L6 pool measurements.

On October 26 at **16:16:44**, after about 239 seconds of OFF3, the monitor observed external
GPU0 Python process `1722370`. It terminated only this campaign's R process
(exit -15). That interrupted attempt contains no `pending.rds`, `receipt.rds`
or `report.json`; it did not produce a completed outer timing. ON3 was not
launched in that attempt. This is an isolation interruption, not a numerical
or performance failure.

The completed continuation directory is
`fastkpc/artifacts/full_cuda_ci/optimizer_l6_slot_runtime_20261026_isolated_completion/`.
Preparation verifies and copies the original inventory, retains the four
completed receipts byte-for-byte, and archives the copied incomplete OFF3 in
`invalidated/off3-attempt1/`. The original tree is unchanged. The initial pending
audit is retained. The final recovery audit is **PASS**, with six valid complete
runs, no remaining runs and **zero completed timings discarded**.
The shared-host pair remains separate and does not contribute extra observations.

On October 27 both GPUs were free of compute jobs. The
[recovery tools](../tools/research/optimizer_l6_slot_runtime_recovery_20261026/README.md)
first rechecked the frozen inputs and four receipts, then ran only OFF3/ON3 on
GPU1, beginning at 09:43:16 and 09:50:02 respectively. GPU UUIDs, driver version
555.42.06, 450 W power limits, CPU affinity 0–19 and single-thread BLAS/OMP/MKL
settings matched the original campaign. Both GPUs remained otherwise free of
compute jobs throughout each run. Final `audit.py` and the original frozen
`archive.py` completed successfully; the manifest binds 269 files and 101
production source inputs.

The continuation executed the frozen original setup and run loop with the same
binary, numerical runner, configuration, original OFF3/ON3 order and performance
gates. Frozen hashes were verified before running; the archived recovery runner
refuses to retry completed numerical failures or discard completed timings.
Do not restart `campaign.py`, `resume.py` or the closure scripts over the
completed evidence directories.
No benchmark or polling job is left running at this handoff. L8 retains
`STOP_L8_PRELOAD_PERFORMANCE`. Review this completed profile before declaring a
separate L7/L8 diagnostic; L7 has one original window and does not automatically
inherit L6's four-window refill opportunity. No new preload or warmup is added.

The build and receipts remain audited incremental source-closure
development evidence, not clean-producer, new real-data/final-PDAG or promotion
qualification. All options stay off by default.

Evidence root: `fastkpc/artifacts/full_cuda_ci/optimizer_l6_slot_runtime_20261026/`.
The qualification inventory was added after the two runs; it does not modify
their frozen runner, protocol or gates.

| Artifact | SHA-256 |
| --- | --- |
| `build.json` | `5c795487ffea16967d54c2aee79e7f871ba32aad805ce422b9d533d29f8f5c67` |
| `checks.json` | `acb4cabf5ecee5cb9ec98ffdf6290a255d43e20253864deba2206cd55b5c32b8` |
| `shared-qualification-closure.json` | `090fa55ced0777bba7bc2f8963f4cc9b66897347d4b4f87e8d6571ff989e9c19` |
| `qualification-manifest.json` | `30b0e9263f0b89d504e7434911d426b48cc2520b76659669fc29fd999fa95b0d` |

Continuation artifacts (in the separate `_isolated_completion` directory):

| Artifact | SHA-256 |
| --- | --- |
| `recovery-plan.json` | `5ee3ec3f2c4b1eb3019be42472b160e1f66033a2820b85736ac39c86a442515a` |
| `pending-audit-1.json` | `5ceab5301d3444655f8a52ea33d71e6c1535bbb058365d306754750d2407d06c` |
| `pending-audit-2.json` | `2dddeec4029b900000ce44e468c2e8e0808f3570d204933ff90bdb507e561a42` |
| `continuation-1.json` | `e9ad35d6a81476d7313c8fef7d9a8835e7c8c99165354795e0ef523b851532a6` |
| `recovery-audit.json` | `f4f807551b185ffdcc3198a9f8616b42cba97ae5e0cfa36d3c49138a7f32e160` |
| `closure.json` | `f358aa31b7ac1f1350b207e572046f6cf677808ccf6609ed566c8edc7d1413ba` |
| `manifest.json` | `593b1c59d2b3c7b3a50e93cfb3101691310a754c03c85f0c482696e5f7d8aeb6` |
