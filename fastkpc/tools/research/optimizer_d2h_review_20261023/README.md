# Optimizer double-window review: completion-path wait

Review of the [double-window diagnostic](../../../docs/PHASE10_OPTIMIZER_DOUBLE_WINDOW_DIAGNOSTIC_20261023.md).
Production source, native, and defaults are unchanged. Research natives are built
outside `fastkpc/build`.

Trace analyses (read-only, any Nsight sqlite export of the double-window replay):

| Script | Question |
| --- | --- |
| `d2h_wait_cause.py` | What ends just before each delayed result D2H starts? Do D2H intervals overlap? |
| `d2h_lock_chain.py` | Does D2H k start at max(previous D2H API return, own last kernel end)? |
| `d2h_api_after_kernel.py` | Is the D2H API entered while the setup's own kernel is still running? |
| `queue_pairs.py` | Which stream ends just before each delayed H2D (queue-sharing partner)? |
| `coincidence_source.py` | Were kernel ends coinciding with delayed D2H starts launched by the just-returned worker? |
| `profile_summary.py` | Codex's shape metrics (wall, peak, delayed H2D, D2H wait, cross-window overlap) |
| `schedule_model.py` | Idealized schedule from per-setup service times in a timeline CSV |

Standalone driver probes (`run_probes.sh OUT`, idle GPU0, 32 connections):
`pageable_d2h_convoy.cu`, `queue_policy_probe.cu`, `queue_refill_probe.cu`,
`queue_mapping_probe.cu`.

Research natives and replay:

```sh
W=/tmp/d2h_native; T=fastkpc/tools/research/optimizer_d2h_review_20261023
A=fastkpc/artifacts/full_cuda_ci/optimizer_d2h_review_20261023
mkdir -p $W/control/src/cuda $W/patched/src/cuda $W/relink
for v in control patched; do cp fastkpc/src/cuda/mgcv_multi_penalty_gcv.cu \
  fastkpc/src/cuda/mgcv_multi_penalty_gcv_small64.cu $W/$v/src/cuda/; done
python3 -I $T/apply_wait_before_d2h.py $W/patched/src/cuda/mgcv_multi_penalty_gcv.cu
sh $T/build_research_natives.sh $W          # relink must reproduce production SHA
python3 -I $T/build_replay.py $A/control $W/control/fastkpc_cuda.so
python3 -I $T/build_replay.py $A/patched $W/patched/fastkpc_cuda.so
python3 -I $T/campaign.py $A/patched campaign L5 L3
python3 -I $T/campaign.py $A/control campaign L5 L3
python3 -I $T/campaign.py $A/patched profile L5 L3
python3 -I $T/compare.py production=fastkpc/artifacts/full_cuda_ci/optimizer_double_windows_20261022 \
  control=$A/control patched=$A/patched
python3 -I $T/archive.py
```

The patch (`wait_before_d2h.diff`) only adds a host wait on a blocking-sync event
before the pageable status download in `multi_penalty_gcv_cuda_optimize_batch`;
device SASS of the Small64 object is identical in production, control, and patched.
Only the Small64 bucket (L3/L5 captured windows) is rebuilt. The replay, fixture,
observer, comparator, and captured inputs are Codex's, unchanged and SHA-checked.
Natives live in `/tmp`; their SHA-256 values are bound in the manifest.
