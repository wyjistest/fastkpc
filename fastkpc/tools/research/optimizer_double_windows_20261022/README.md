# Bounded adjacent optimizer windows

Compare the already captured L3 W31/W32 and L5 W7/W8 without changing the
production native, its numerical kernels, target cohorts, or runtime defaults.
The original fixture parser, complete-state comparator, and observer are reused
unchanged and bound to their archived SHA-256 hashes.

All variants use the same 128 prepared handles, created before warmup:

| Variant | Native calls | Workers |
| --- | --- | --- |
| A | 64 requests followed by 64 requests | 64 per call |
| B | 128 requests in original order | 64 |
| C | 128 requests in original order | 32 |

Worker count bounds concurrent host calls; it does not establish exclusive
hardware-queue mappings. No sorting experiment or production rolling scheduler
is implemented. Setup creation/release and arena/cache registration are outside
the measured experiment. Both A outputs stay leased until the second call has
returned, matching the combined call's output lifetime and resident inputs.

Warm A/B/C once each. Then run all six permutations of A/B/C in fixed order:
ABC, BCA, CAB, CBA, BAC, ACB. This gives six primary timings per variant and
balanced positions. Input request construction, diagnostic shadow D2H, bitwise
comparison, and file output occur outside the complete-pair timer. The timer
includes original native uploads, optimizer, fitted/residual materialization,
ordinary metadata downloads, scheduler calls, and return packaging. Separate
observed and Nsight runs provide per-setup timelines and do not enter medians.

Every result is compared to the original captured output at its original setup
index, including all exported optimizer floating/integral state, coefficients,
fitted/residual, route/rank/info/status and trajectory counters. Target key order,
GPU-only exact-primary authority, and release of every output lease are checked.
Memcheck uses two unchanged setups from each window (four total); it is a reduced
resource check, not complete 128-setup capacity or runtime qualification.

Run from the repository root; the output directory must not exist at build time.
The campaign pins affinity to CPUs 0-19, requires both GPUs free of compute jobs,
and uses only GPU0 with 32 CUDA connections and BLAS/OMP/MKL threads set to one.

```sh
python3 -I fastkpc/tools/research/optimizer_double_windows_20261022/build.py OUTPUT
python3 -I fastkpc/tools/research/optimizer_double_windows_20261022/campaign.py OUTPUT smoke
python3 -I fastkpc/tools/research/optimizer_double_windows_20261022/campaign.py OUTPUT campaign
python3 -I fastkpc/tools/research/optimizer_double_windows_20261022/campaign.py OUTPUT profile
python3 -I fastkpc/tools/research/optimizer_double_windows_20261022/campaign.py OUTPUT memcheck
python3 -I fastkpc/tools/research/optimizer_double_windows_20261022/analyze.py OUTPUT
python3 -I fastkpc/tools/research/optimizer_double_windows_20261022/plot.py OUTPUT
python3 -I fastkpc/tools/research/optimizer_double_windows_20261022/archive.py OUTPUT
```

Evidence remains an isolated captured-cohort development diagnostic. Even a
successful combined call does not qualify prepared-handle reuse/allocation
during GPU execution, shared arena registration, full method semantics, or
integrated capacity. Phase 10, candidate, holdout, and recommendation stay unchanged.

The completed campaign finds A fastest in all six cycles for both pairs. Its
closure validator explicitly checks that result before recording
`STOP_FOR_CAPTURED_PAIRS`; a different future result requires a new evaluation
instead of silently inheriting this decision. See the
[result report](../../../docs/PHASE10_OPTIMIZER_DOUBLE_WINDOW_DIAGNOSTIC_20261023.md).
