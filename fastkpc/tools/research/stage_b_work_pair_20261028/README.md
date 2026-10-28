# Isolated fixed-VT DBDSQR work-pair qualification

This is an isolated GPU experiment, not a production change or promotion.
Bind native e8c39703f01e09260d330167b49267c295aa2221167d5248d0980a97c80ca045,
the 103 production inputs in the accepted L3 build, generated DBDSQR
be76c5e06ce8bce4013a9759a3a54fc63feed9e421a1d3d44af77e11f32f53b9,
and the sealed Stage B attempt3 captures (49,919 + 54,878 components).

Only for UPLO=L, NCVT=1, NRU=NCC=0: remove unused forward W2/W3 stores;
remove unused backward W0/W1 stores and map retained W2/W3 values and consumers
to W0/W1. Keep lower-to-upper stores, all scalar arithmetic, signs, helpers,
sorting, convergence, d/e/q layout, recover, lanes and synchronization.
The isolated entry rejects unsupported configurations. Dynamic shared changes
from 7*N doubles to 5*N doubles; no production generator/file is overwritten.

Qualification uses two separately compiled diagnostic libraries, with the same
--fmad=false flags and identical instrumentation. GPU comparison checks d/e/q,
full rotation records in order, count/overflow/info/branch coverage, a/b,
scale/live metadata, candidates, vectors/values/means and boundary status.
Diagnostic exports and shared canaries are absent from timing libraries.
All 104,797 captured inputs run in their original order, ten lanes of width64.
Saved native three-field hashes authenticate the uninstrumented candidate in
every complete Stage B replay. Synthetic GPU cases cover r=1..350, repeated
values, splits, directions, zero/nonzero shifts, sign/tie/signed-zero inputs,
rotation-capacity boundaries, precheck errors and injected nonconvergence.
Any branch not naturally exercised is reported, never counted as natural
coverage. CPU code constructs inputs/compares bytes; it never solves QR/SVD.

Before timing, require all parity, boundary, canary and allocation-closure gates.
Record compiled REG/STACK/LOCAL and CUDA occupancy API results for blockSize32,
baseline dynamic 19656 and candidate 14040, on the same GPU. Occupancy remains a
theoretical upper bound, not measured hardware utilization. No NCU permission
changes. No cooperative QR, recover modification, extra lanes or CPU fallback.

Use GPU1, both GPUs otherwise free of compute jobs, CPU affinity0-19, one math
thread, CUDA connections32. Stop only our job if external compute appears and
retain that attempt. Predeclare six fresh full-level timing runs in order
A1,C1,C2,A2,A3,C3 (A=unchanged native, C=uninstrumented candidate interposer).
Use the sealed observer: final payload hashing is after each Stage B boundary;
the measured boundary includes recover, status, wave joins and cache stores.
Each run checks all native component hashes, zero fallback, original cohorts,
interposer counts and tracked resource closure. No profiler timing in medians.

Stage GO requires correctness, both L2/L3 medians below A, and combined medians
at least 5% lower with all three C totals below all three A totals. Otherwise
report CONDITIONAL for positive but insufficient/noisy benefit, or STOP for
parity/resource failure or nonpositive combined benefit. This threshold only
decides whether a later production experiment is warranted; it does not change
accepted 329.078s, defaults, L8 STOP, Phase10 ACTIVE, candidate NOT_FROZEN,
holdout SEALED or recommended=false. No full-method speedup claim.

Run build.py OUTPUT, then campaign.py OUTPUT. Source/build/protocol hashes are
recorded before GPU execution. All failures and amendments remain in evidence.
