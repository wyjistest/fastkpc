# Build-entry consistency qualification

This campaign fixes only the method-batch TU's numerical flags. Other CUDA
translation units retain their existing FMA policy. It does not relabel the
accepted postback receipts or install a new native before qualification.

Run from the repository root, in order:

```sh
python3 fastkpc/tools/research/build_consistency_20261029/build.py freeze
# Apply the shared method-batch flag fix before the next command.
python3 fastkpc/tools/research/build_consistency_20261029/compare_objects.py declare
python3 fastkpc/tools/research/build_consistency_20261029/build.py build
python3 fastkpc/tools/research/build_consistency_20261029/compare_objects.py
python3 fastkpc/tools/research/build_consistency_20261029/qualify.py parent
python3 fastkpc/tools/research/build_consistency_20261029/qualify.py full
python3 fastkpc/tools/research/build_consistency_20261029/qualify.py incremental
python3 fastkpc/tools/research/build_consistency_20261029/canonical.py
```

The full build starts with an empty build directory in a production source
snapshot. The actual fixed-SP incremental entry then rebuilds in the same
directory. Both commands, every linked object and final native are compared.
NVCC embeds process-dependent temporary filenames as local STT_FILE symbols:
keep the raw object hashes and separately compare diagnostic copies made with
`objcopy --strip-debug --discard-all`. Those copies are never linked or installed.
Compile commands, all object payloads and the actual linked native must match.
The first running build driver predates this distinction and is archived as
`build-driver-original.py`; its raw-object assertion can fail solely for this
metadata. The separate payload/native gate must pass before GPU qualification.
All products stay under a new artifact directory; historical and installed
objects are not overwritten. This is full source-closure evidence, not a
committed clean-checkout performance campaign.

GPU-only cross-native tests cover `dcc.perm`, `hsic.gamma`, and `hsic.perm`
on five fixtures per native (nonlinear levels 0/1/2, dense level 3 and a real
351x6 subset). They require bitwise p/pMax, task/deletion/graph/sepset/count,
route/status, permutation RNG and resource/authority parity. Existing identity,
permutation-overlap, deferred-SVD and failure-order regressions also run on the
full build. These are regressions against the retained qualified GPU native,
not new CPU-authority campaigns for those methods. Missing CUDA is FAIL;
SKIP never counts as PASS.

The initial existing-test wrapper exposed invalid multi-argument
`Sys.unsetenv` cleanup in the overlap/deferred tests. The two tests now pass a
character vector; the wrapper evaluates the entire expression vector so cleanup
runs after the test. `regressions.py` preserves the first failed harness log and
reruns only the four existing regressions, retaining the already-passing method
receipt. Tested earlier tool versions are archived with their original hashes.

Finally one full canonical numerical run bridges the full rebuild to the
accepted postback result, including the archived 1e-10 authority gates. That
single timing is not performance evidence. GPU1 must be exclusive; unrelated
GPU0 work is recorded and permitted only for these numerical checks. The initial
incremental qualification was not started because GPU0 became busy; its original
protocol is preserved separately. Performance experiments continue to require
both GPUs free of external compute. Only `PASS_BUILD_FIX_PARENT` may
be used as the fixed parent for the row-only capacity experiment.
