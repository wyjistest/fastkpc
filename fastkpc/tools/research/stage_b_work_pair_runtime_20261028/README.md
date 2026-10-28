# Fixed-VT work-pair runtime qualification

Default-off production integration, conditional on the missing L4–L8 GPU
bitwise qualification. Existing L2/L3 and synthetic evidence remains sealed.
No CPU numerical authority, fallback, algorithm, rotation order, cache policy,
lane count, wave shape, or optimizer change is permitted.

Execution order:

1. On the unchanged e8c39703 native, capture all 25,873 L4–L8 components,
   retaining original cohorts, keys, solver metadata and payload hashes. The
   canonical capture must pass the existing authority, bitwise, capacity and
   resource gates. Observer D2H is diagnostic, not production performance.
2. Compare the original and fixed-VT diagnostic kernels on these exact inputs:
   d/e/q, rotation records/order/count/overflow, metadata, roots, recovered
   outputs, and shared end canaries. Reuse the sealed libraries; all byte
   comparisons run on GPU. L4/5/6 use ten contexts, L7 two, L8 one.
3. Only after success, generate a separate fixed-VT production entry. An
   immutable plan mode binds the entry to 7N or 5N shared storage. Reject
   mismatched configurations on the host before *any* launch. Never launch
   an undersized original kernel. The production option defaults off.
4. Run configuration/lifecycle qualification, then fresh-process canonical
   OFF1/ON1/ON2/OFF2/OFF3/ON3 on the same native, GPU1, CPU affinity 0–19,
   one math thread, 32 CUDA connections, and no other GPU compute tasks.
   Preserve failed attempts. Full CI/pMax must be bitwise equal to the saved
   accepted source closure and satisfy the independent <=1e-10 authority.
   All existing graph/trace/order/RNG, GPU authority, capacity and resource
   gates remain. Capture and qualification instrumentation is excluded from
   the performance runs.
5. Predeclared full-method GO: all correctness/capacity gates pass, ON outer
   median improves >=2%, every ON outer is below every OFF outer, and total
   Stage B median improves >=5%. Report L2–L8 separately; do not extrapolate
   replay savings or claim actual occupancy from the occupancy API.

The accepted full-method opt-in development median remains 329.078 s until
this integrated campaign passes. Phase10 ACTIVE, candidate NOT_FROZEN,
holdout SEALED, recommended=false, default-off policy and L8 optimizer STOP
are unchanged. No clean-producer or promotion claim, commit or push.

Commands are run from the repository root. Build/campaign scripts require a
fresh artifact directory and authenticate their inputs before and after work.

    python3 fastkpc/tools/research/stage_b_work_pair_runtime_20261028/build_capture.py ARTIFACT/deep
    python3 fastkpc/tools/research/stage_b_work_pair_runtime_20261028/capture.py ARTIFACT/deep

After deep qualification, apply the production integration and run:

    python3 fastkpc/tools/research/stage_b_work_pair_runtime_20261028/build_runtime.py ARTIFACT/runtime ARTIFACT/deep
    python3 fastkpc/tools/research/stage_b_work_pair_runtime_20261028/checks.py ARTIFACT/runtime ARTIFACT/deep
    python3 fastkpc/tools/research/stage_b_work_pair_runtime_20261028/campaign.py ARTIFACT/runtime
    python3 fastkpc/tools/research/stage_b_work_pair_runtime_20261028/archive.py ARTIFACT/runtime ARTIFACT/deep

The build preserves all parent objects except structured eig and the one-call
host wrapper. It checks the original generated include byte for byte and the
new include against the sealed candidate (entry name is the only difference).
Runtime checks cover default OFF, invalid options, plan immutability across
environment changes, 7N/5N entry/storage mismatches before launch, invalid
submissions, GPU output parity and allocation closure. Per-mode host counters
authenticate all canonical launches. `destroyed_before_return` is sampled
while the local cache still owns its plans; final resource checks occur after
the one-call returns and its cache has been destroyed.

The observer and qualifier are mechanically adapted copies of the sealed
stage_b_attribution_20261028 and stage_b_work_pair_20261028 tools. Their
original files and evidence are never edited.
