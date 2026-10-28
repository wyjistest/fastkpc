# Phase 10 device-properties cache qualification — 2026-10-25

Status: **GO_DEVICE_PROPERTIES_CACHE_DEVELOPMENT**, completed 2026-10-26.
The original six-run isolated campaign now passes its unchanged performance
gate: OFF/ON outer medians **408.882/392.416 seconds**, saving **16.466 seconds
(4.03%)**, with three paired wins. Every canonical hard gate passes. The earlier
shared-host OFF3/ON3 supplement remains separate and is not pooled into these
medians. Evidence is audited incremental source closure; the cache remains
default-off. Phase 10 ACTIVE, candidate NOT_FROZEN, holdout SEALED and
recommended=false remain unchanged.

## Scope and motivation

The revised optimizer review measured 6,777 `cudaGetDeviceProperties` callbacks,
20.972625024 seconds of summed host duration, a 10.712397397-second interval
union, and 7.940001302 seconds on the main thread. Those overlapping, instrumented
durations are not prospective outer-time savings. In particular, L3 prepared
construction's approximately 5.24 seconds overlaps any future L3 slot work.

This experiment only caches successful static device-properties queries. All
nine first-party call sites share the implementation in `cuda_status.cpp`,
including all five multi-penalty capacity translation units. No optimizer,
kernel, Stage A/B, cohort, math-policy, preload or slot-scheduler change is made.

The process-local flag `FASTKPC_CUDA_DEVICE_PROPERTIES_CACHE` defaults to zero;
set it to `1` before first CUDA status/helper use to opt in. The mode is fixed at
first use; values other than empty/0/1 fail. The cache keys CUDA logical device
IDs, assuming visibility stays fixed after CUDA initialization. A mutex covers
first query and publication, so simultaneous misses cannot produce duplicate
successful queries. Failed queries are not cached, and callers retain their
CUDA status checks and error text. Availability/current-device/count/set-device,
version and stream checks are unchanged. Cached metadata is not a device-health
probe. Cache hits do not repeat the properties API; independent live CUDA checks
remain responsible for runtime/device failures.

The cache owns a small host map and no GPU allocations, streams, events, handles
or workspaces. Diagnostic counters describe this helper only, excluding queries
inside CUDA libraries. A research-only R bridge snapshots counters outside the
full-method timer.

## Protocol fixed before performance runs

The accepted parent is runtime `4a1ff6b`, repository HEAD `d27e59b`, native
`037b92f109a727ea2c50c737fb1c2b70048221bcffe41f215a532622398ffb58`.
The reference is the previous L4-ON canonical receipt, not the L4-OFF result.

Both modes keep S2, L4/L5 slots, L4 workers32, status wait, 11 GiB component cache,
all-hit bypass and eight eig lanes enabled. Only device-properties caching
changes. Six fresh R processes were planned as OFF1/ON1/ON2/OFF2/OFF3/ON3 on GPU1, with both
GPUs otherwise free, CPU affinity 0–19, one BLAS/OMP thread, one source/native
closure and no profiling injection. Artifact persistence and offline authority
checks are outside the timer.

Hard gates retain 240,498 CI and all pMax bits, task order/deletion/skeleton/
sepsets/count/RNG, physical cohorts, Gram order, optimizer counters and resource
closure. Authority error must remain within `1e-10`; CPU authority/fallback and
large production D2H remain zero. ON must issue one successful first-party query
for the single visible device per process; OFF must query on each request.

Development GO requires all three paired wins and median savings of at least
the maximum of 2 seconds, 0.5% of OFF median, and either mode's elapsed range.
Positive savings below that gate are CONDITIONAL; no median improvement is
STOP. The flag stays default-off. This is a small optimization with its own
predeclared gate; the earlier L4 2% threshold is not reused.

## Evidence locations

- Tools: `fastkpc/tools/research/device_properties_cache_20261025/`.
- Campaign: `fastkpc/artifacts/full_cuda_ci/device_properties_cache_20261025/`.
- Shared-host supplement:
  `fastkpc/artifacts/full_cuda_ci/device_properties_cache_20261025_shared_host/`.
- Completed isolated recovery:
  `fastkpc/artifacts/full_cuda_ci/device_properties_cache_20261025_isolated_completion/`.
- Old reports, tools and receipts are unchanged.
- The build is an audited incremental source closure: nine affected objects
  rebuilt, all remaining parent objects verified unchanged. It is not a full
  rebuild or a clean-producer performance receipt.

## Native qualification

The successful build binds 100 source inputs and produces native
`994c43d386c859f23c34509b857db22064fb9a009b8fe871c719dd608f820df9`.
All **107 GPU functions** in the eight affected CUDA objects have exactly the
same instruction words as their accepted parent objects, including every
multi-penalty capacity. The ninth rebuilt object is the shared host helper.

Mocked-property tests exercise concurrent first access, device-key isolation,
failure/retry, invalid arguments, live availability/version failures, unchanged
caller error messages, default/empty/OFF/ON/invalid settings and mode freezing.
These mocks do not implement any numerical computation.

Real GPU1 metadata matches direct CUDA properties byte-for-byte. A fresh-process
32-thread check (512 concurrent calls plus two public diagnostic calls) records
514 helper requests in each mode: OFF makes 514 queries, ON makes one. This is
a mechanism check, not a whole-method performance result.

The frozen L4 fixture covers 15 requests across 14 target-capacity shapes. Both
modes pass 544 optimizer/Stage A comparisons and 122 negative checks per run,
including partial upload/detach cancellation, stale token handling and resource
closure. Native and captured-fixture memcheck runs report **0 errors and 0 bytes
leaked**. No production GPU resource is added by the host metadata cache.

## Original isolated campaign before interruption

Four isolated full canonical runs have completed with every hard gate passing:

| Run | Cache | Outer seconds | First-party requests / queries |
| --- | --- | ---: | ---: |
| OFF1 | disabled | 410.266 | 6,777 / 6,777 |
| ON1 | enabled | 392.416 | 6,777 / 1 |
| ON2 | enabled | 391.504 | 6,777 / 1 |
| OFF2 | disabled | 406.673 | 6,777 / 6,777 |

Paired savings are 17.850 and 15.169 seconds. All 240,498 CI and pMax bits,
skeleton semantics, RNG, numerical counters, cohorts and resource gates pass.
The maximum authority error remains `3.05425129631942e-12`. These four runs do
not yet satisfy the predeclared three-pair performance gate.

OFF3 was interrupted after approximately 117 seconds when another user's job
started on GPU0. The isolation monitor terminated only this campaign's R
process. No completed timing or numerical receipt was discarded. The partial
log, GPU/process samples and explicit interruption reason are retained; this
attempt cannot contribute to performance statistics.

The original isolated OFF3/ON3 runs required both GPUs to be free. `resume.py`
executes the frozen campaign setup and loop, preserves the original protocol,
archives only incomplete interference attempts, and refuses to retry a completed
numerical failure. It adds a separate continuation receipt and copies its own
source into the artifact directory. `audit_continuation.py` binds those recovery
records before the unchanged original `archive.py` applies the original gates.

## User-authorized shared-host continuation

After GPU0 occupancy and GPU1 availability were explained, the user requested
continuing the tests on GPU1. The added `shared_host/campaign.py` executes the
original frozen setup and run loop, allowing foreign compute only on the other
GPU. It uses the same binary, sources, numerical configuration, OFF/ON order,
reference receipts and hard gates. A small R wrapper changes only the original
runner's evidence-class string before evaluation; both the receipt and JSON
report explicitly identify shared-host supplemental evidence. Both wrappers
and all original tools are bound before execution.

The original protocol, four complete isolated runs and incomplete interrupted
attempt are unchanged. Supplemental results live in a separate artifact tree:

| Run | Cache | Outer seconds | First-party requests / queries |
| --- | --- | ---: | ---: |
| OFF3 supplemental | disabled | 404.665 | 6,777 / 6,777 |
| ON3 supplemental | enabled | 389.392 | 6,777 / 1 |

Both runs pass every hard gate (61 OFF, 62 ON): all 240,498 CI and all pMax bits match the
accepted GPU reference and the earlier same-mode run; task/deletion/skeleton/
sepsets/count/RNG, optimizer counters, cohorts and Gram order remain exact.
Maximum CI and pMax authority error is `3.05425129631942e-12`. CPU numerical
authority, fallback and large production D2H remain zero. Both runs close all
tracked resources, and the sampled GPU1 peak is 22,366 MiB for each. Cache query
failures are zero; ON records 6,776 hits and one cache entry.

The supplemental pair saves 15.273 seconds (3.774%). Optimizer host time falls
from 199.904446 to 187.262607 seconds, and prepared-build time from 9.398398 to
1.512589 seconds. Stage B is 144.384943/144.210406 seconds. These are the existing
instrumentation boundaries; overlapping boundaries must not be summed into a
new attribution.

The process monitor records no foreign GPU1 compute. GPU0 has PID 1374309 during
OFF3 and PIDs 1374309/1390432 during ON3. Its sampled average utilization is
15.528%/12.120%, respectively. Accordingly, both process records have
`selected_gpu_isolated=true`, `shared_host_allowed=true`, and `isolated=false`.
The positive paired timing is supplemental development evidence, not the third
isolated pair or a replacement for the predeclared three-pair gate. No pooled
supplemental/isolated pooled median is published.

Machine audit: `PASS_SHARED_HOST_SUPPLEMENT`. The manifest binds the two new
receipts, tools, native/source identity, authority references, GPU/process logs,
and hashes of the original evidence as observed during this continuation.

- Closure SHA256:
  `6822416272902c2a49f98c0853b0366b33678a5b6b816b3c7b078d4c88e1e990`.
- Manifest SHA256:
  `717d603822063bbe6c78aa12a2d570102c94de87e881f25f6ac56c252012b126`.

No campaign process remains running. The opt-in code stays default-off; no
commit, push, new real-data/final-PDAG qualification or promotion is claimed.

## Completed isolation recovery — 2026-10-26

Both GPUs became idle. The pre-existing continuation copy was resumed with the
unchanged `resume.py`, then checked by `audit_continuation.py` and the original
`archive.py`. Original and supplemental artifact trees remain unchanged. The
interrupted incomplete attempt is preserved under the copy's `invalidated/`;
the recovery audit confirms six isolated complete runs and zero discarded
completed timings. The sequence, binary, configuration and gates are unchanged.

| Run | Cache | Outer seconds | First-party requests / queries |
| --- | --- | ---: | ---: |
| OFF3 isolated | disabled | 408.882 | 6,777 / 6,777 |
| ON3 isolated | enabled | 397.792 | 6,777 / 1 |

All six isolated runs preserve every CI/pMax bit and all semantic, authority
and resource gates; maximum CI and pMax authority error remains
`3.05425129631942e-12`. Paired savings are **17.850/15.169/11.090 seconds**.
OFF/ON ranges are 3.593/6.288 seconds, so the predeclared minimum median saving
is 6.288 seconds. Actual saving is **16.466 seconds**, passing with three wins.

Prepared-build medians are 9.215638/1.441541 seconds; complete optimizer host
medians are 205.699023/190.824856 seconds. These overlapping boundaries and
late-layer variations do not independently attribute the entire outer saving
to properties calls. The newly accepted opt-in development median is
**392.416 seconds**; 407.531 seconds remains the preceding L4-slot result.

Recovery manifest SHA256:
`9bccd7ea7542d27e3b81ebbda10a7a585316a76ab5cb11f4f168a22ae0b926be`.
The producer/native identity remains the uncommitted incremental cache closure
above. No clean-checkout, new real-data/final-PDAG or promotion claim is added.
