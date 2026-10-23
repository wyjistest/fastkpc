# Isolated guarded-QR diagnostic

This is a development rejection/qualification experiment for
`PHASE10_MAGIC_GPU_DESIGN_PROPOSAL.md`, not a production backend. It leaves
`fastkpc/src`, `fastkpc/R`, the production binary, and all defaults unchanged.

The namespaced diagnostic library compiles a mechanically instrumented copy
of the current production optimizer. The builder refuses ambiguous anchors
and records the input/generated/native SHA-256 identities. Its driver launches
the existing primary kernel directly: exact SVD or guarded QR, with no whole
trajectory stability replay or merge. The QR experiment also disables the
old boundary exact-shadow confirmation. Existing per-evaluation QR-to-SVD
fallback stays visible and marks the whole target for exact replay.

Both paths refresh the selected state. Both feed their selected log-SP into
the unchanged **production GPU exact fixed evaluator and GPU full eig/gamma**.
The diagnostic bridge has an old `optimizer_residual` name/returned route
label; the saved input metadata explicitly authenticates
`fixed_sp_evaluation=true`, route 2, and solver info 0. Residual D2H is used
only for diagnostic comparisons; the full-eig consumer uses the device view.
No CPU numerical fallback is enabled. Native setup/geometry preparation and
host comparison/bookkeeping retain their existing roles.
The old diagnostic PreparedS DTO requires a protocol `data_p=48`; the adapter
uses that constant even for real inputs with another p. Actual input p is
retained separately. All numerical dimensions/arrays use the actual n and q.

From the repository root, with both GPUs otherwise idle:

```sh
python3 fastkpc/tools/research/guarded_qr_20261020/build_probe.py /tmp/fastkpc-guarded-qr-20261020
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=32
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
Rscript fastkpc/tools/research/guarded_qr_20261020/prepare.R /tmp/fastkpc-guarded-qr-20261020/fixtures.rds
Rscript fastkpc/tools/research/guarded_qr_20261020/run.R /tmp/fastkpc-guarded-qr-20261020 smoke
Rscript fastkpc/tools/research/guarded_qr_20261020/run.R /tmp/fastkpc-guarded-qr-20261020 calibration
Rscript fastkpc/tools/research/guarded_qr_20261020/run.R /tmp/fastkpc-guarded-qr-20261020 validation
Rscript fastkpc/tools/research/guarded_qr_20261020/check_capacity_reference.R /tmp/fastkpc-guarded-qr-20261020
```

Do not overlap compilation with timed runs. Output files are never silently
replaced across fixture identities. The runner can resume completed cohort
receipts with the same fixture and diagnostic binary; the frozen guard cannot
be overwritten. The fixture builder currently requires the archived canonical
connections32 receipt at its original `/tmp` path; a captured `fixtures.rds`
can instead be used directly to reproduce the numeric experiment.

The declared list contains 72 two-target cohorts: 20 calibration and 52
independent development validation, six q shapes (28 through 73), historical
hard cases, canonical task identities, and three large real inputs. No
conditioning set crosses the split. Historical real-input task receipts
provide identities only; their invalid old p-values are never an authority
or a selection criterion. No sealed holdout is read. q=82 and larger shapes
are outside this first bounded experiment.

Trace fields are kind/iteration/probe index/decision, route/ranks, score,
condition estimate, score delta, a heuristic rounding scale, log-SP, and
decision margins. Kinds 0–5 are initial, trial acceptance, convergence,
Hessian sign, Newton cap, and boundary acceptance. Same-SP GPU QR/SVD
evaluations compare score, gradient, Hessian and coefficients before interpreting
independent-trajectory differences. Exact probe selected-SP must match the
installed producer bitwise; its refreshed coefficients must match production
exact finalization, and traced/untraced repeated outputs must agree.

The frozen guard is empirical. It includes rank/root/status/mixed-SVD risks
and a predeclared grid of score/convergence/Hessian/Newton/condition margins.
There is no QR-only singular-value distance certificate. Calibration chooses
the lowest measured paired-kernel cost with no calibration misses, requiring
the existing residual/EDF/score/RSS/statistic/moment contracts and a tighter
1e-11 p margin against the exact GPU reference. Validation cannot retune it.
This is **not** a proof of an error bound or a replacement for mgcv authority.

Timing consists of one traced warmup and three untraced full optimizer kernel
runs per path. Order alternates between cohorts. Reported guarded cost is
QR kernel time plus measured from-initial-state exact kernel time for flagged
two-target cohorts. It is a conservative cohort replay granularity, while
excluding guard/dispatch/downstream overhead. It is not an integrated guarded
runtime measurement, a full-method estimate, or a promotion result. The
production cross-setup scheduler has different concurrency. A later target-only
compacted replay queue would need its own measurement and numerical gates.
The probe is compiled in the base80 bucket. The untimed capacity check compares
all captured exact selected states and exact fixed outputs against production
small64/base80 dispatch. Numeric equality does not make the probe's isolated
timings representative of the production scheduler or capacity buckets.
