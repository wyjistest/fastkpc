# Canonical structured prefill qualification

This is a default-off, n=351, p=48, numCol=35 extension of the qualified
22-column structured producer. Selected-SP, residuals, structured arithmetic,
original delayed Gram grouping, pair/gamma and scheduler remain unchanged.
The only runtime change is the restricted shape gate. The production build
uses `fastkpc/tools/generate_recorded_dbdsqr.py`, without a research dependency.

Run sequentially from the repository root, with both GPUs free of compute jobs:

```sh
python3 -I fastkpc/tools/research/structured_prefill_canonical_20261022/build.py OUTPUT
python3 -I fastkpc/tools/research/structured_prefill_canonical_20261022/checks.py OUTPUT
python3 -I fastkpc/tools/research/structured_prefill_canonical_20261022/campaign.py OUTPUT
python3 -I fastkpc/tools/research/structured_prefill_canonical_20261022/archive.py OUTPUT
```

The build copies all production sources, the generator and build recipe into
`OUTPUT/build-source`, builds every object from an empty build directory, and
installs the resulting native and objects. The research directory is absent
from that snapshot. This proves full source closure; it is not a committed
clean-producer performance run. The prior native and generator bytes are bound
by SHA. `build.py OUTPUT --finish` only validates and installs an already
successfully completed snapshot build; it never skips an incomplete build.

The checks cover unsupported configuration rejection, retained cache parity,
640 archived structured payloads, original Gram lifecycle, 56 true small-gap
fallbacks among 70 components split 48+8, failure without publication, rotation
overflow, abandoned submission and resource closure. Sanitizer covers four
archived payloads plus the lifecycle and mixed-input cases, not the full R run.

The campaign uses GPU0, 32 CUDA connections, one thread per math library, an
11-GiB component cache, bypass and eight retained dense eig lanes. Each method
run uses a fresh R process and dataset caches. It first checks off against the
previous canonical off receipt, including p-value and pMax bits. It then runs
dense/structured/structured/dense, stopping immediately on any gate failure.
All four comparisons use the same native and fixed configuration.

Each complete default-Inf skeleton must preserve all 240,498 logical CI,
task/deletion/graph/sepset/count/RNG, physical cohorts and Gram key/order.
All CI and every pMax value must pass the archived independent kpcalg authority
at absolute tolerance 1e-10. Structured-versus-dense low-bit changes are counted,
not called bitwise equivalence; repeated runs within a mode must be bitwise
identical. No CPU numerical producer or fallback, residual/component D2H or
unclosed tracked CUDA resources is permitted. The original info/rotation/gap
policy may select GPU dense fallback and every such component is reported.

Artifact persistence and authority comparison occur outside the skeleton
timer. External 200-ms GPU telemetry includes both GPUs. Stage B is completed
host wall time; overlapping per-lane event durations are never summed into a
device makespan. Memory telemetry is a sampled peak, not an instantaneous bound.
Two repetitions per mode are development evidence, not a promotion median.
Phase 10, candidate, holdout, recommendation and the default remain unchanged.
The campaign qualifies the skeleton; no new WAN-PDAG/orientation run is claimed.
