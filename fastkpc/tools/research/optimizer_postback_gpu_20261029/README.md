# Isolated q37/q46 exact vector postback, 2026-10-29

Research only. Keep the accepted production native and all 104 source inputs
unchanged. The candidate preserves original LWORK/NB, descending block order,
serial reference DLARFT, and unchanged reference DLARFB/BLAS bodies. Warp 0
owns independent C columns for Q and C rows for P; each owns a WORK row with
the original leading dimension. All 32 lanes rendezvous around every block.
L5 tail blocks remain Q=14 and P=13, followed by a full 32-column block.

`build.py` authenticates the sealed static audit and creates a separate source
overlay. `candidate.inc` is injected only there. No production file, default,
native library or sealed prior evidence is rewritten. Unsupported q uses the
original implementation. The candidate is not a CPU numerical fallback.

The input-gate repair in the preceding review tool is checked with one valid
input and five negative inputs, using `--validate-only`; no host floating-point
reference is run. `unit.cu` generates input bits on the host and performs all
reflector arithmetic and reference/candidate comparison on GPU. It covers
1,553 DLARFB block shapes and 166 complete Q/P calls across all 83 augmented
shapes, with four patterns and three seeds: 20,628 cases. Patterns include
signed zero, zero/unit T diagonals and Inf/NaN in C. Memcheck/synccheck cover
one pattern/seed of all 1,719 shapes. NaN checks observe output propagation;
they are not proof of absence of reads.

The full optimizer candidate rebuilds only the small64 CUDA translation unit
and relinks a research native with the other 42 authenticated object files.
The accepted native remains the uninstrumented reference. Separate profile
libraries use the same blocked loop with serial or sliced DLARFB and collect
Q/immediate-P/deferred-P, DLARFT and DLARFB counts/cycles. Profile libraries
must independently reproduce the captured outputs; their cycles include
stalls and are not elapsed-time savings or instrumentation-free performance.

The cohort harness uses the original 32 slots and original ordered atomic
request assignment, with unchanged cohorts: L4 2,154 setups / 17,767 targets;
L5 955 / 6,816. Full qualification compares optimizer state, all recorded
trajectory counters, coefficients, fitted values and residuals against the
authenticated captured fixtures. A separate timing mode detaches coefficients
and residuals D2D, verifies after the timed work, and excludes diagnostic
payload D2H from that boundary. Its diagnostic arena and creation/destruction
are reported separately; it does not qualify runtime co-residency by itself.

One warmup per variant precedes three alternating pairs. Enter full outer
qualification only if combined work and charged lifecycle medians improve
at least 5%, neither layer regresses over 3%, and all three pairs win.
Then run three canonical pairs in fresh R processes, preserving the complete
existing authority/bitwise/task/graph/RNG/cohort/resource gates. Require at
least 2% outer median improvement and three paired wins. Both stages run on
GPU1 with affinity 0–19, one math thread and 32 CUDA connections. No competing
GPU job is allowed. Performance runs must also wait for our CUDA compilers.

From repository root, use a new directory for each build attempt:

```sh
T=fastkpc/tools/research/optimizer_postback_gpu_20261029
O=fastkpc/artifacts/full_cuda_ci/optimizer_postback_gpu_20261029/attempt2
python3 -I "$T/build.py" "$O"
python3 -I "$T/run_unit.py" "$O"
python3 -I "$T/build_backend.py" "$O" candidate
python3 -I "$T/build_cohorts.py" "$O" reference candidate
python3 -I "$T/run_cohorts.py" "$O" qualification
# These profile builds are independent, but finish all compilation before timing.
python3 -I "$T/build_backend.py" "$O" reference-profile
python3 -I "$T/build_backend.py" "$O" candidate-profile
python3 -I "$T/build_cohorts.py" "$O" reference-profile candidate-profile
python3 -I "$T/run_cohorts.py" "$O" profile
python3 -I "$T/run_cohorts.py" "$O" timing
# Conditional on the declared layer gate:
python3 -I "$T/run_outer.py" "$O"
python3 -I "$T/archive.py" "$O"
```

Keep failed attempts and their logs. Do not rerun into an existing protocol or
alter its frozen inputs. This experiment does not authorize production
integration, default changes, candidate freezing or holdout access. Accepted
311.440-second opt-in development evidence remains unchanged until a separate
runtime integration is qualified.
