# Prepared-owned optimizer status wait

Default-off `FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H=1` creates one
blocking-sync event per prepared object, on that object's device. It waits
after the existing producer event and before the optimizer's first status
D2H. The original download, final synchronization and downstream producer
event remain. No fixed-evaluator, kernel, cohort, scheduler or window change.

The policy is read at prepared creation. No event is created in a worker,
shared through thread-local storage, or recreated for later trace downloads.
Invalid values fail closed. The current 64-request/64-worker window remains.

Run from the repository root into a new artifact directory:

```sh
python3 -I fastkpc/tools/research/structured_prefill_canonical_20261022/build.py OUTPUT
python3 -I fastkpc/tools/research/optimizer_status_wait_20261023/checks.py OUTPUT
python3 -I fastkpc/tools/research/optimizer_status_wait_20261023/campaign.py OUTPUT
python3 -I fastkpc/tools/research/optimizer_status_wait_20261023/archive.py OUTPUT
```

The full build starts from an empty production-only snapshot, compiles all
capacity buckets and installs only the completed native. `qualify.cpp` uses
test-only CUDA interposition to check event flags, allocation/reuse/destruction,
record/wait failure propagation, invalid configuration, trace download and
multi-worker cancellation. It compares complete output bits on actual q28/q46
captures and synthetic q73/q82/q193/q385 capacity cases. Synthetic cases test
host behavior and capacity dispatch; full canonical covers actual larger shapes.
It also verifies that the independent fixed evaluator retains its original
wait/download path and output bits. CUDA memcheck covers the real q28/q46 full
paths, allocation/failure/destruction in all five buckets, and cancellation.
The interrupted wider memcheck is preserved in `validation-v1` and is not a pass.
No test hook enters production code. `device_code.py` compares all five capacity
objects' SASS with the archived S2 full build, normalizing only anonymous
translation-unit hash names; `checks.py` validates its object hashes and result.
Correctness checks may use an idle GPU selected by
`FASTKPC_STATUS_WAIT_CHECK_DEVICE=0|1`; their timing is not performance evidence.
The canonical campaign defaults to GPU0 with both GPUs otherwise idle. For this
execution the user authorized GPU1 while GPU0 has another user's training job:

```sh
FASTKPC_STATUS_WAIT_PERF_DEVICE=1 FASTKPC_STATUS_WAIT_SHARED_HOST=1 \
  python3 -I fastkpc/tools/research/optimizer_status_wait_20261023/campaign.py OUTPUT
```

This configuration requires GPU1 alone to be free of other compute processes;
all GPU process samples are preserved. Its results are shared-host development
evidence, not isolated performance evidence. The initial `semantic.py` pair
is retained separately and excluded from the four timed comparison runs.

The canonical campaign runs S2 off/on/on/off, two timings per setting, on the
selected GPU, CPU affinity 0-19, 32 CUDA connections,
one BLAS/OMP/MKL thread, retained cache/bypass/eig lanes, and fresh R/cache state.
GPU processes are sampled during each run. An unexpected compute process on the
selected GPU (or either GPU in isolated mode) invalidates the run and stops only
this benchmark, preserving its evidence.
It reuses the complete L2 runtime numerical, cohort, Gram, authority and
resource gates, additionally requiring every CI/pMax bit to stay unchanged.
The first off run also matches the historical S2 receipt. Snapshot persistence
and offline comparisons are outside elapsed. This is development evidence.

Predeclared acceptance: all hard gates pass; enabled two-run outer median is
at most 1.03 times disabled median. This bounded no-material-regression check
is not a confidence interval or promotion gate. Defaults remain off regardless.
The experiment does not qualify runtime rolling, sorted scheduling, whole-layer
residency, other numerical routes, new real inputs or final PDAG.
