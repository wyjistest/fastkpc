# Public-input reproduction

This entry rebuilds a frozen producer and creates new R references, GPU
correctness receipts and interleaved A/C timings using only public synthetic
inputs. It needs neither the local `causalDiscovery` directory nor any previous
native, capture or campaign artifact. The original numerical sources remain
unchanged. It does not install the resulting native in the caller's checkout.

The default producer is `2be6a816d3039a5f9340ba7018166d737ab11d9a`, also tested
in the [seven-input campaign](../../docs/PHASE10_CLEAN_GENERAL_INPUTS_20261106.md).
The new native receives its own SHA; it need not have the historical native's
hash. Tool snapshots and the invoking checkout's Git HEAD are recorded
separately from the producer identity.

The [local validation report](../../docs/PHASE10_PUBLIC_REPRODUCTION_20261112.md)
records a complete run of the two defaults from a new path containing spaces.

## Prerequisites

Use Linux, Python >=3.9, Git, util-linux (`taskset`, `flock`), R with its C++17
compiler, and CUDA at `/usr/local/cuda`. The unchanged producer build targets
`sm_89`; this entry accepts compute capability 8.9. Current validation uses an
RTX 4090 with 24 GiB, CUDA 12.5.82, driver 555.42.06, GCC 9.4.0, and R 4.4.1.
Other architectures and smaller-memory GPUs have not been qualified here.

Required R namespaces and the tested versions:

| Package | Version | Package | Version |
|---|---|---|---|
| Rcpp | 1.0.13 | RcppArmadillo | 14.0.2.1 |
| RcppEigen | 0.3.4.0.2 | RSpectra | 0.16.2 |
| pcalg | 2.7.12 | graph | 1.84.1 |
| RBGL | 1.82.0 | mgcv | 1.9.1 |
| Matrix | 1.7.0 | energy | 1.7.12 |
| kernlab | 0.9.33 | jsonlite | 1.8.9 |
| digest | 0.6.37 | | |

Install these packages and their dependencies in the selected R library before
running. `doctor` checks availability and records versions, compiler, CUDA,
BLAS, full R session information and GPU identity. It does not install packages
or claim that a different dependency version is equivalent. A new environment
must pass its own numerical gates; resume rejects changes in the recorded
environment. This is an environment record, not a container or dependency
installer.

Run from a repository checkout containing this directory and the producer
commit above. Use a full clone or fetch the required history before starting.

```bash
python3 fastkpc/tools/reproduce_public/reproduce.py doctor --gpu 1 --cpus 0-19
python3 fastkpc/tools/reproduce_public/reproduce.py run \
  --work-dir '/path/to/new public reproduction' --gpu 1 --cpus 0-19
```

Choose an available physical GPU index or full UUID. Omit `--cpus` to use the
current process affinity; an explicit list must be a subset of that affinity.
The work directory must not exist. Paths containing spaces are supported;
the frozen build is invoked through a temporary path without spaces because
its include flags otherwise cannot handle them. Commands and that alias are
preserved in the build log.

The full build compiles 44 objects from a separate detached clone, without
parent objects or incremental flags. On the development machine this costs
about 50 minutes and produces about 1.3 GiB of build artifacts. Allow several
GiB for sources, receipts and an optional second copy of the evidence bundle.
Default public-input runs take additional time to compute fresh R references.

## Input and execution protocol

| Case | Complete matrix | Default numCol | Purpose |
|---|---|---:|---|
| public500 (default) | 500 x 50 | 50 | Public multi-variable workload |
| blocks256 (default) | 256 x 12 | 25 | Small block-correlated workload |
| dense500 (optional) | 500 x 8 | 50 | Natural search must reach L6 |
| dense1025 (optional) | 1025 x 4 | 102 | Must exercise dense GPU producer outside structured capacity |

Select additional cases before starting, for example:

```bash
python3 fastkpc/tools/reproduce_public/reproduce.py run \
  --work-dir /path/to/new-four-case-run --gpu 1 \
  --cases public500,blocks256,dense500,dense1025
```

The dense cases are expensive because the original CPU orientation dominates
their complete calls. The previous campaign recorded about 2,800 seconds per
dense500 GPU-plus-orientation call, and about 230 seconds for dense1025. Eight
GPU calls per case plus fresh CPU reference/replay work are required. Selecting
only the two defaults does not qualify the optional deep/dense cases.

All input rows and columns are retained. Generation fixes seeds and RNG kinds,
and records both RDS-file and FP64 column-major data hashes. The public calls
omit `numCol` and the conditioning limit, using `floor(n/10)` and `Inf` with
natural stopping. Method is `dcc.gamma`, alpha is 0.1 and index is 1.

Compare generated input identities with the published validation record when
claiming the same workload. A different R serialization version may change a
file hash; the FP64 data hash distinguishes this from changed numerical input.

For each input the entry performs:

1. A fresh, unmodified R `kpcalg`/mgcv/RSpectra reference with `pcalg` search.
2. Native host scheduling replay of those independent R values, checking the
   scheduling semantics separately from the original pcalg node-major order.
3. OFF and ON correctness calls.
4. Six fresh-process timings, in fixed order `A1 C1 C2 A2 A3 C3`.
5. An independent reread of all eight GPU receipts, with no native loaded.

A/OFF and C/ON are both GPU configurations. A retains exact-primary and fixed
coefficient reuse; C also requests the existing prefill, structured, slots,
work-pair and postback switches. Their precise settings are in the frozen
`source/fastkpc/tools/research/numcol_20261106/common.R`. Shape/capacity dispatch
still controls which mechanisms actually execute. The current bounded cache
budget and eight eigen lanes are retained.

Each timing process warms unrelated 80 x 3 data and clears dataset-specific
caches before the complete WAN-PDAG call. The boundary includes GPU skeleton
and original CPU orientation; input generation, reference construction, receipt
I/O and comparisons are outside it. Stage medians are reported separately and
must not be added as if they describe one run.

Correctness needs the selected GPU free of other compute. Timing needs all
GPUs free of external compute, matching the existing campaign's host-isolation
policy. The entry samples compute PIDs about every three seconds and records
them. Detected interference rejects the attempt and terminates only this
entry's process group. These are sampled checks, not proof that no short-lived
external process ever ran. They also do not monitor every unrelated CPU job;
reserve the host for comparable timings.

## Results, stopping and resuming

`--until prepare`, `--until build` or `--until qualify` stops at a completed
boundary. Continue using the frozen driver:

```bash
python3 '/path/to/new public reproduction/tools/reproduce.py' resume \
  --work-dir '/path/to/new public reproduction'
```

Resume verifies source/tool, input, native/object, environment and completed
receipt identities. It never silently reuses an incomplete or rejected job.
A failed build/job retains its logs and requires a fresh work directory;
successful earlier boundaries are resumable. Once sealed, use `verify` rather
than `resume`. To change cases, affinity, producer or GPU, start a new run.

Failures, missing prerequisites and missing evidence exit nonzero. There is
no artifact-dependent `SKIP` path. `summary.json` separates correctness from
performance:

- `PASS_PUBLIC_INPUT_QUALIFICATION` means the raw numerical/behavior audit
  passed: CI and pMax absolute error <=1e-10; task/deletion/graph/sepset/count,
  complete orientation/PDAG/RNG, GPU skeleton authority and resource gates.
- `all_inputs_non_regression` requires each input's ON/OFF outer median <=1.05.
- Speed `PROVEN` requires ON/OFF median <=0.95 **and** slowest ON < fastest OFF.
  Otherwise it is `NOT_PROVEN`, or `REGRESSION` if the median ratio exceeds 1.05.

A numerical PASS is retained even when timing regresses; exit zero alone is
not a performance PASS. CPU reference timings include tracing and are not a
CPU performance baseline. This protocol proves neither general GPU-vs-CPU
speedup nor promotion. Phase 10 remains ACTIVE, candidate NOT_FROZEN, holdout
SEALED and recommended=false.

## Portable evidence and independent checks

The output contains frozen sources/tools, protocol/environment, all compiler
commands, every object/native hash, public input identities, R references,
raw GPU receipts, timing/process logs, an independent receipt audit and a
relative-path manifest. The manifest excludes `.git`, Python bytecode and the
run lock. It preserves diagnostic absolute paths as historical metadata;
verification uses only relative manifest entries.

```bash
python3 fastkpc/tools/reproduce_public/reproduce.py bundle \
  --work-dir '/path/to/new public reproduction' \
  --output /path/outside/run/public-reproduction.tar.gz
tar -xzf /path/outside/run/public-reproduction.tar.gz -C /path/to/empty/destination
python3 fastkpc/tools/reproduce_public/reproduce.py verify \
  --work-dir /path/to/empty/destination/public-reproduction \
  --manifest /path/to/empty/destination/public-reproduction/manifest.json \
  --expected-sha256 TRUSTED_MANIFEST_SHA256
```

`bundle` creates a local archive and prints its SHA; it sends nothing. Share
the manifest SHA through the review handoff separately. Without a trusted
manifest hash, `verify` establishes internal file consistency only. Verification
uses Python's standard library and needs no R, CUDA or GPU. Absolute paths,
parent traversal, escaping symlinks, missing files and hash mismatches fail.

An engineer with the R prerequisites can independently recheck raw receipts
after relocation, without loading the native:

```bash
cd /path/to/empty/destination/public-reproduction/source
Rscript --vanilla ../tools/audit_receipts.R \
  ../references ../qualification ../timing /path/outside/bundle/reaudit.json
```

The same verifier also accepts the historical seven-input closure's `evidence`
map: point `--work-dir` at its complete artifact root and provide that closure's
manifest and trusted SHA. That checks supplied files, not missing private
datasets or numerical claims by itself. The new public bundle does not include
the seven-input campaign's private records, and cannot replace its 56-receipt
audit. A fresh public run, a hash check and a raw-receipt reaudit are distinct
levels of evidence; none by itself is a third-party rerun on another machine.

Host-only failure-path tests:

```bash
python3 -m unittest discover -s fastkpc/tools/reproduce_public -p 'test_*.py' -v
```
