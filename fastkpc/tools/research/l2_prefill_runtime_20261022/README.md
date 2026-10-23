# Bounded L2 runtime qualification

The default-off `FASTKPC_DCC_GAMMA_L2_COMPONENT_PREFILL=1` extends the existing
structured level prefill from L>=3 (S3) to L>=2 (S2), only for dcc.gamma,
n=351, p=48, numCol=35. It requires the existing structured/level flags,
fresh caches, exact-primary reuse, 11 GiB component cache, residual bypass,
eight eig lanes, and the original scheduler. No numerical kernel is replaced.

Stage A reuses the actual single-penalty optimizer window's contexts and exact
selected log-SP bits. For each S it creates one fixed evaluator, evaluates
balanced cohorts of at most 16 targets, detaches device residuals before
workspace reuse, and releases the evaluator. Stage B and downstream Gram
resolution use the already qualified structured implementation.

From the repository root, using a new output directory:

```sh
python3 -I fastkpc/tools/research/l2_prefill_runtime_20261022/build.py OUTPUT
python3 -I fastkpc/tools/research/l2_prefill_runtime_20261022/checks.py OUTPUT
python3 -I fastkpc/tools/research/l2_prefill_runtime_20261022/campaign.py OUTPUT
python3 -I fastkpc/tools/research/l2_prefill_runtime_20261022/archive.py OUTPUT
```

`build.py` reuses the production-only full snapshot recipe. `checks.py` runs
configuration rejection, retained regression, 640-component lifecycle and
mixed fallback checks, compute-sanitizer, and a small actual runtime smoke.
The smoke retains eight canonical columns and independently shuffles the
other 40; its GPU S3/S2 comparison is not CPU-authority qualification.

The first integration smoke caught an obsolete accounting equality: L2's
independent fixed evaluations were being counted as coefficient-arena gathers.
The source now counts L2 fixed targets separately. The historical
`rebuild_ledger.py` archives that failed native/source/log and recompiles only
the corrected C++ unit in this campaign's new full-build snapshot, requiring
the other 42 object hashes to remain unchanged. Fresh reproduction uses the
corrected sources directly through `build.py`; it does not need this repair.

`campaign.py` runs S3 -> S2 -> S2 -> S3 sequentially on GPU0 in fresh R
processes, requiring both GPUs to have no compute jobs before each run.
The first S3 must reproduce historical S3 bits. S2 must satisfy all 240,498
archived authority CI values and all pMax values at 1e-10, and preserve task,
deletion, graph, sepset, count, RNG, physical cohort, and Gram ordering.
Only L2 may change p-value bits; each mode must repeat bitwise.
CPU numerical authority, fallback, large residual/component D2H, cache
metadata mismatches, and leaked CUDA resources must remain zero.

Full method elapsed excludes artifact persistence and offline authority
comparisons. Reports separate Stage A (including retained-Y copy, creation,
exact solve, D2D detach/registration, release), Stage B, begin/reset, finish,
L2 consumer work and elapsed, and complete outer elapsed. The prepared-build
counter overlaps Stage A and must not be added again to the total.

This is a source-closure development campaign. It neither enables the flag by
default nor qualifies new real inputs, final PDAG, a candidate, or promotion.
Phase 10 remains ACTIVE and the holdout remains sealed.
