# Exact optimizer vector-postback implementation audit

Source-specific audit for q=37/46. No production edits, CUDA compilation, GPU
execution, floating-point numerical evaluator, performance claim or runtime
integration. The accepted 311.440 s development median and promotion states stay
unchanged.

Run from the repository root:

```
python3 fastkpc/tools/research/optimizer_postback_audit_20261029/audit.py NEW_OUTPUT_DIRECTORY
```

The tool authenticates the unchanged native and 104 production inputs against the
sealed L4/L5 attribution. It inventories the actual Q/L/N/F/C and P/R/T/F/R
DLARFB branches, enumerates the producer's legal augmented shapes and fixed
reflector blocks, and checks workspace/address ownership with integers.

A host-only symbolic harness compiles the **unchanged extracted** reference
DCOPY/DGEMM/DTRMM/DLARFB function bodies with `doublereal` replaced by expression
tokens. It compares full DLARFB calls with independent single-vector calls of
the same reference function. Ordered expression DAGs use exact interning keys,
not floating-point results or probabilistic expression hashes. Zero/one tags
exercise the relevant DTRMM input predicates; this is not a numerical solver or
GPU parity qualification. Guard cells and undefined symbolic inputs expose
incorrect writes/reads. A triangular overwrite counterexample rejects arbitrary
column reordering.

Q slices own C columns; P slices own C rows. Both own WORK rows, with original
leading dimensions. DLARFT, reflector order, NB, LWORK, T and query/status writes
remain serial. Slicing DORMBR itself is excluded: it changes NW and can change
the L4 block size. Source-specific reasoning and integer/token checks do not
prove compiled CUDA behavior, synchronization correctness or performance.

Outputs preserve source/tool snapshots, commands, call/block inventories,
symbolic results, failure logs if any, and a manifest. The tool refuses to
overwrite an existing output directory. Future GPU prototyping remains a
separate task with bitwise, resource, whole-layer and outer-time gates.
