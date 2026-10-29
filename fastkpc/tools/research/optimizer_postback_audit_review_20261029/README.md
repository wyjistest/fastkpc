# Vector-postback audit review: host slice cross-check (2026-10-29)

Review tool for [PHASE10_OPTIMIZER_POSTBACK_AUDIT_REVIEW_20261029.md](../../../docs/PHASE10_OPTIMIZER_POSTBACK_AUDIT_REVIEW_20261029.md).
Host only; no GPU, no production edits. It compiles the sealed, unchanged extracted reference DCOPY/DGEMM/DTRMM/DLARFB
bodies (`symbolic_vendor.inc` from the audit evidence) with IEEE `double` and compares one full DLARFB call with
per-vector calls (Q: one C column each, P: one C row each; WORK row r, ldwork = q; lanes in reverse order) for every
block in the audit inventory, bitwise. Inputs cover dense values, signed zeros, 0/1 T diagonals and Inf/NaN in C;
unreferenced V1/T parts and initial WORK are NaN sentinels. This is an equivalence cross-check, not GPU qualification
and not numerical authority.

```sh
A=fastkpc/artifacts/full_cuda_ci/optimizer_postback_audit_20261029/attempt1
T=fastkpc/tools/research/optimizer_postback_audit_review_20261029
OUT=$(mktemp -d)
python3 -I $T/export_blocks.py $A/inventory.json $OUT/blocks.tsv
g++ -O2 -std=c++17 -ffp-contract=off -Wno-write-strings -DVENDOR_INC="\"$PWD/$A/symbolic_vendor.inc\"" \
  $T/slice_check.cpp -lcrypto -o $OUT/slice_check
$OUT/slice_check $OUT/blocks.tsv --validate-only
$OUT/slice_check $OUT/blocks.tsv        # expect: cases 18636 failures 0 finite_output_nan_count 0 vector_calls 709404
g++ -O2 -std=c++17 -ffp-contract=off -Wno-write-strings -DVENDOR_INC="\"$PWD/$A/symbolic_vendor.inc\"" \
  -DSENSITIVITY_CONTROL $T/slice_check.cpp -lcrypto -o $OUT/slice_control
$OUT/slice_control $OUT/blocks.tsv      # expect failures > 0 (one input's lowest bit flipped)
```

The revised input gate authenticates the inventory JSON and exact TSV bytes,
validates all 1,553 rows and requires 18,636 cases / 709,404 vector calls.
Missing, empty, truncated or altered input must exit 2, including with
`--validate-only` (which performs no floating-point evaluation). The NaN count
reports observed propagation into C; it is not a memory-access detector.
