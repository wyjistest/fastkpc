# L4/L5 optimizer attribution, 2026-10-29

Diagnostic only. Production source, native, numerical kernels, 32 slots, request
order and cohorts are unchanged. No optimizer or scheduling optimization is made.

The historical traces have no setup keys. `legacy.py` identifies anonymous slow
bursts, without equating GPU start order with canonical request order. A separate
current-native run attaches canonical fixture identities to native optimizer
calls through CUPTI external correlation. The observer reads the **existing**
completed status D2H buffer; it adds no device allocation, copy or synchronization.
Its host-only layout is mechanically extracted from the authenticated small64
source, then checked against every returned target and the aggregate counters.

The captured L4/L5 fixtures supply original request indices and all expected
optimizer outputs. The observer compares all 11 floating-vector and 16 integer-
vector optimizer fields bitwise. Full canonical gates are reused from the
work-pair qualification runner. Outputs and additional analysis are development
diagnostics, not new outer-performance or promotion evidence.

Canonical request index is zero-based fixture order. The rebind observer captures
the address of the original geometry reference; all addresses must equal the
index-zero address plus index times the authenticated geometry struct size. The
native passes `geometry.at(request)` by reference from its unchanged ordered
vector, proving current request-array identity without guessing from time order.
Host timestamps cover rebind and batch entry; they do not measure atomic queue
fetch. GPU kernel timestamps are independently correlated.
Per-target phase cycles include scheduling stalls and are overlapping work across
targets. Their sum is not wall time. The target with most accounted phase cycles
is a candidate critical target, not a measured final block completion.

Run from the repository root, in a fresh artifact directory:

```
python3 fastkpc/tools/research/optimizer_l45_attribution_20261029/build.py ARTIFACT_DIR
python3 fastkpc/tools/research/optimizer_l45_attribution_20261029/legacy.py ARTIFACT_DIR
python3 fastkpc/tools/research/optimizer_l45_attribution_20261029/run.py ARTIFACT_DIR
python3 fastkpc/tools/research/optimizer_l45_attribution_20261029/analyze.py ARTIFACT_DIR
python3 fastkpc/tools/research/optimizer_l45_attribution_20261029/archive.py ARTIFACT_DIR
```

The current-native run uses GPU1, affinity 0-19, one math thread, 32 CUDA
connections and all accepted opt-in settings. It requires no competing GPU job.
Keep failed attempts and sealed evidence unchanged. No CPU numerical authority,
fallback, new decomposition trace, hardware performance counter or extra payload
D2H is introduced.

Completed evidence is in
`fastkpc/artifacts/full_cuda_ci/optimizer_l45_attribution_20261029/attempt3/`.
All 3,109 setups / 24,583 targets and 240,498 canonical CI gates pass. Existing
SVD cycles account for 96.389% / 97.383% of L4/L5 recorded phase cycles; vector
postback (including deferred right-vector completion) accounts for 58.928% /
71.698% of SVD cycles. These overlapping counters are not elapsed-time savings.

The successful GPU run's original tools are retained in `protocol-tools/`.
The offline analyzer subsequently corrected two assumptions: CUPTI/Linux thread
IDs require a per-level bijection, and 11 kernel ends exceed host-return
timestamps by up to 70.837 us. Those discrepancies remain in `analysis.json`;
cross-domain timestamp differences do not establish waits or GPU gaps. No GPU
rerun or numerical change was needed for these analysis corrections.

`archive.py` also binds shared fixture/runner/R sources to their current Git
blobs, and the dataset/oracle files to saved receipt/report hashes. The additional
shared-source closure is explicitly post-run, not a retroactive protocol freeze.
Before editing any protocol tool after a run, use `archive.py ARTIFACT_DIR
--snapshot` while all hashes still match. Archive only after analysis succeeds;
never overwrite an earlier attempt or its sealed outputs.
