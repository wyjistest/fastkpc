# Default-off structured component prefill, 351 × 22

This campaign first corrects the timing completion boundary, then integrates
the qualified three-field producer into the existing level>=3 prefill. It
does not enable a default or qualify canonical 351×48. Selected-SP, residual
finalization, original delayed Gram cohorts and pair/gamma remain unchanged.

`FASTKPC_DCC_GAMMA_STRUCTURED_COMPONENT_PREFILL=1` additionally requires the
existing level-prefill flag, dcc.gamma, 351×22, numCol=35, fresh caches,
exact-primary reuse, component-cache bypass, eight dense lanes and the original
scheduler. The new flag defaults to zero. Unsupported configurations fail closed.

One host thread submits up to ten independent 64-component contexts before
reading compact status. Each context owns only tracked CUDA workspace, stream
and events. The 11 GiB cache and original dense/Gram/pair context are shared.
The structured solver has its own translation unit compiled with `--fmad=false`
and a private vendored LAPACK namespace. Its arithmetic kernels and the original
centering statements are checked against the qualified research sources.

The fixed info/rotation/gap<1e-8 rule chooses GPU dense fallback per component.
Fallback indices are chunked to the original dense capacity (22 here). Every
structured and fallback store joins the original context stream before a wave
is committed; consumers cannot read an unfinished or failed level. The final
join precedes residual-arena release. Eig-solver identity is stored separately
from the residual's stable-SVD route/status. No CPU numerical fallback exists.

Run from the repository root:

```sh
python3 fastkpc/tools/research/structured_prefill_20261022/export_fixture.py WORK/fixtures
FASTKPC_INCREMENTAL_EIG_ONLY=1 sh fastkpc/tools/build_cuda_native.sh
```

The build recompiles one-call, vertical CUDA, R API and the new structured TU;
other objects are retained. Evidence is source-closure development, not a clean
producer build or promotion. The qualified historical adapter remains archived
with its old native and input hashes.

`completion_timing.py OUT.json` uses the archived adapter and the same 640 inputs
for every shape. Run with CUDA_VISIBLE_DEVICES=1, CUDA_DEVICE_MAX_CONNECTIONS=32,
all math-library threads=1, and LD_LIBRARY_PATH including R/CUDA lib directories.
It reports host return and final completion separately, joining all streams
inside the timer. It does not include context creation or shadow D2H.

Compile `lifecycle.cu` with nvcc -O3 -arch=sm_89 -std=c++17,
`-Ifastkpc/src/cuda`, the new `fastkpc/build/full_cuda_ci_structured_eig.o`,
`-Xlinker=-Bsymbolic`, the native .so, and R/cudart/cublas/cusolver libraries.
Arguments are `WORK/fixtures/residuals.f64 WORK/fixtures/three-fields.f64`.
The normal run checks all 640 archived payloads byte-for-byte. Append `smoke`
under compute-sanitizer memcheck/leak-check: it checks four archived payloads,
70 mixed components including 30 real gap fallbacks split 22+8, failure without
publication, pending reset, actual rotation overflow and abandoned submission.
The injected dense error exists only in this source-shared test build.

Also run these in separate processes on an idle GPU:

```sh
sh fastkpc/tools/with_phase10_exact_development_env.sh Rscript fastkpc/tests/test_full_cuda_ci_structured_prefill.R
sh fastkpc/tools/with_phase10_exact_development_env.sh Rscript fastkpc/tests/test_full_cuda_ci_level_component_prefill.R
sh fastkpc/tools/with_phase10_exact_development_env.sh Rscript fastkpc/tests/test_full_cuda_ci_legacy_component_cache.R
```

`authority.R OUTPUT` is an independent offline TEST oracle, not a production
path: 19,623 logical CI reuse archived canonical kpcalg values; the remaining
4,641 logical CI (4,518 unique keys) are evaluated separately with the original
kpcalg/mgcv functions. It never supplies or repairs a GPU result. Finish this
CPU comparison campaign before starting timing runs. Its provenance, versions,
source hashes and complete p-value trace are retained.

`campaign.py OUTPUT` then runs fresh off/dense/structured R processes sequentially
on GPU1, collecting an external 200-ms device-memory sample. It requires the
offline authority at OUTPUT/authority.rds. All result persistence and numerical
comparison happen after the complete skeleton timer. `run.R` checks all 24,264
CI and pMax against authority at <=1e-10, exact task/deletion/graph/sepset/count/
RNG/group/Gram identities, GPU-only production and resource closure. The off and
dense paths additionally require bitwise p/pMax identity. Structured-vs-dense
bit changes are reported, not misrepresented as bitwise output equivalence.

The scope is complete 22-column skeleton qualification. It does not include a
new WAN-PDAG campaign or a full canonical run. Phase 10, holdout, candidate and
recommendation state do not change. Structured lane event durations overlap;
their sum must not be substituted for Stage B host wall time.
