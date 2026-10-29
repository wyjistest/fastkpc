# Postback runtime review: embedded cubin comparison (2026-10-29)

Review tool for [PHASE10_OPTIMIZER_POSTBACK_RUNTIME_REVIEW_20261029.md](../../../docs/PHASE10_OPTIMIZER_POSTBACK_RUNTIME_REVIEW_20261029.md).
Host only; no GPU, no production edits. Extracts every embedded cubin from the preserved baseline native and the
installed runtime native with `cuobjdump -xelf all`, compares them as SHA-256 multisets, and records register/stack
usage of every function in cubins that exist only in the new native. Fails closed on native identity mismatch,
empty extraction, any missing baseline cubin or an unexpected number of new cubins; refuses to overwrite its output.

This shows which device objects the *installed* binary reused unchanged. It does not show that a clean rebuild from
current sources would reproduce them: for example `r_api_cuda.cpp` includes `mgcv_multi_penalty_gcv_extended.hpp`,
whose new `postback64` declarations change that translation unit's preprocessed source.

```sh
R=fastkpc/artifacts/full_cuda_ci/optimizer_postback_runtime_20261029
python3 -I fastkpc/tools/research/optimizer_postback_runtime_review_20261029/cubin_compare.py \
  $R/baseline/fastkpc_cuda.so 12acc3cd01cce2dcac5f2bfb5ec02f27056e92154cbab2534c7f03bdc22caa7d \
  fastkpc/build/fastkpc_cuda.so a1e16a9c0c8089ea90491426c56475394449efb03317be4a87f1925c7b73bd34 \
  1 NEW_OUTPUT.json                      # expect: PASS common 17 old_only 0 new_only 1
```
