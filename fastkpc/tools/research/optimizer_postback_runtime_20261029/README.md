# Default-off q37/q46 vector postback runtime qualification

This campaign compiles a separate `postback64` backend. Original small64 objects,
kernel stacks and default execution remain intact. The environment flag is
`FASTKPC_DCC_GAMMA_VECTOR_POSTBACK=0|1`; unset is OFF. A prepared object captures
its backend at creation. Only q37/q46 select ON, and mixed backend multi-setup
requests fail before submission. The one-call opt-in scope is `dcc.gamma`.

The baseline snapshot was authenticated before production edits. Historical
source manifests remain bound to that snapshot; they are not rewritten to
describe the new source as an old clean producer.

Run from the repository root, with both GPUs free of external compute:

```sh
python3 -I fastkpc/tools/research/optimizer_postback_runtime_20261029/build.py device attempt1
python3 -I fastkpc/tools/research/optimizer_postback_runtime_20261029/build.py link attempt1
python3 -I fastkpc/tools/research/optimizer_postback_runtime_20261029/qualify.py build attempt1 qualification2
python3 -I fastkpc/tools/research/optimizer_postback_runtime_20261029/qualify.py run attempt1 qualification2
python3 -I fastkpc/tools/research/optimizer_postback_runtime_20261029/run_outer.py attempt1 qualification2
python3 -I fastkpc/tools/research/optimizer_postback_runtime_20261029/archive.py attempt1 qualification2 audit
python3 -I fastkpc/tools/research/optimizer_postback_runtime_20261029/archive.py attempt1 qualification2 install
python3 -I fastkpc/tools/research/optimizer_postback_runtime_20261029/archive.py attempt1 qualification2 seal
```

Existing output paths are exclusive: use a new attempt for a rerun. The original
baseline freeze is a one-time pre-edit operation, not a command to repeat after
source edits. All new numerical evaluation is GPU-only; host comparisons use
authenticated captured GPU results and the existing authority receipts.

`qualification` preserves an initial test-harness compile failure (missing
`void**` cast for host `cudaMalloc`); it ran no GPU code. `qualification2` is the
corrected frozen harness. Its tests include full C/WORK bits, immutable backend
selection, invalid flags, mixed-mode rejection, fixed evaluator, coefficient
finalization, lease/rebind, multi-setup calls and non-target q28. GPU unit and
runtime contracts have memcheck/synccheck campaigns. Complete original-order L4
and L5 cohorts use the original 32 slots.

The canonical sequence is OFF1, ON1, ON2, OFF2, OFF3, ON3, all on the same native.
It retains the previous complete numerical, semantic, ordering and capacity
gates, and adds actual backend counts plus exact original raw capacity checks
for OFF. The performance gate is median ratio <=0.98 and 3/3 paired wins; L8
variation is retained. Data caches start fresh, persistence is outside the outer
compute boundary, and GPU1/CPU affinity 0-19/single math threads remain fixed.

This is incremental source-closure development evidence. It is not a full clean
rebuild, new real-data/final-PDAG qualification, candidate freeze or promotion.
