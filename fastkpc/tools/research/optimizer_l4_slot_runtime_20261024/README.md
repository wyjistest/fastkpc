# L4 slot runtime qualification (2026-10-24)

This is a separate continuation of the immutable isolated L4 slot experiment.
Production defaults, candidate, holdout and promotion are unchanged. GPU1 only;
no CPU numerical fallback. No commit or push is part of this experiment.

## Gate 1: actual cache / arena coexistence

Run `python3 coexistence.py ARTIFACT_DIR/coexistence` before runtime edits.
The unchanged accepted native runs the complete canonical S2 workload with L5
slots and L4 workers32 enabled. A research interposer adds 32 real native
small64 prepared objects (351/37/4, capacity 18), their detach events, and an
additional complete L4 coefficient arena (5,259,032 bytes). It executes and
checks the captured 18-target request on every extra slot, retaining the pool
from the first L4 optimizer call through the end of L4 Stage A.

This deliberately retains the baseline's original window handles and its real
coefficient arena as well. A PASS is conservative coexistence evidence; a
failure is inconclusive for replacement capacity and must not be mislabeled a
candidate STOP. Allocation-boundary and 10 ms cudaMemGetInfo observations plus
all optimizer/component-store boundaries cover the relevant lifecycle. The
pool is released before Stage B, matching the proposed runtime lifetime.

Required: <=1 GiB warm incremental pool cost, >=2 GiB free throughout pool
residency, 34 original windows / 2,154 setups / 17,767 targets and exact Stage A
coverage; all 32 warm outputs exact; full canonical numerical, scheduling,
RNG and resource gates from the immutable prior runner. Diagnostic shadow D2H
only checks the extra research executions; it never supplies production output.
The instrumented elapsed time is not performance evidence.

The conservative probe completed all 55 canonical gates but measured only
1,791,819,776 free bytes while double-counting window handles. The follow-up
declared in [REPLACEMENT_CAPACITY.md](REPLACEMENT_CAPACITY.md) runs the complete
captured layer against the real completed L4 cache/arena after those handles
are released. `coexistence_replacement.py` passes: 2,586,640,384 minimum free
bytes, 922,746,880 incremental bytes, 17,767 exact targets / 216,194 iterations,
and all 55 complete canonical gates. Both probe results are retained. Neither
probe's elapsed time is performance evidence.

Probe reproduction requires the unchanged attested parent source/native
`33e2d3e` / `2c8dd8af…bcd0ee` and the immutable captured slot DSO. Its local
accepted receipt directory is `coexistence-replacement`; `coexistence-retry2`
holds the completed conservative failure. The two preceding attempts failed
before slot measurement (R library loader path, then selecting only W1 which
has no 18-target cohort). These are harness failures, not capacity results.

## Conditional Gate 2: default-off runtime

Only after coexistence passes: a narrow L4 option, 32 slots of capacity 18,
original-order setup queue, unchanged target cohorts and mathematical kernels.
Slot reuse must wait for coefficient detach; Stage A must use the original
cohorts and component order. Verify all 14 captured target capacities, stale
leases, failures/cancellation, resource closure and normalized device code.

Full canonical sequence: OFF/ON/ON/OFF/OFF/ON, three fresh R processes per mode,
same binary, GPU1, CPU affinity 0-19, one BLAS/OMP/MKL thread, CUDA connections32.
Both modes retain L5 slots, L4 workers32, S2 and the existing retained full-eig
stack. Fresh dataset-specific caches; output persistence outside outer time.
Require exact p-values/pMax versus same-native OFF and the accepted reference,
all authority errors <=1e-10, task/deletion/skeleton/sepsets/count/RNG exact,
physical cohorts and Gram resolution order exact, no CPU authority/fallback,
and closed resources. L4 ON reports one real layer boundary, not 34 fictitious
windows. Other-layer window/worker identities must remain exact.

Performance GO requires ON/OFF median outer <=0.98, L4 optimizer <=0.90 and
three paired outer wins, with all hard gates and >=2 GiB joint L4 headroom.
No performance threshold is inferred from the isolated 3.099x speedup.
This remains source-closure development evidence, not a clean producer or
promotion campaign. Freeze source, native, tools and dependencies before runs.

After the replacement gate, `build.py ARTIFACT_DIR` snapshots the new source
and reuses the parent's SHA-verified objects, rebuilding seven dependent
objects. `checks.py ARTIFACT_DIR` checks native API parity, the original Stage A
cohort split, injected failure cleanup, reduced memcheck, opt-in rejection and
normalized device instructions. Then run `campaign.py ARTIFACT_DIR` and
`archive.py ARTIFACT_DIR`. Do not modify frozen inputs during the campaign.

Stage A must be compared with the parent's actual Stage A, not an unsplit
optimizer shadow. The first native check used that wrong boundary and failed;
its `qualify.log` is preserved. The frozen parent binary independently shows
18,847 changed fitted/residual values for the 17/18-target requests when Stage A
uses its existing 9/8 and 9/9 cohorts (maximum 4.440892098500626e-15); the other
12 widths and endpoint do not change. `stage_a_reference.cpp` generates exact
references using that old binary and original cohorts. `checks-v2` compares the
candidate against those references with no tolerance, while continuing to
compare optimizer outputs against every captured bit. This corrects the
comparison boundary and does not alter runtime math, cohorts or acceptance
tolerance. Original checks and both parent/candidate binaries remain archived.
