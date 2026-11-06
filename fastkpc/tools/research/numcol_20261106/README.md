# numCol generalization, 2026-11-06

Preserve caller numCol or the existing floor(n/10) default. The statistical
parameter is never replaced to fit an optimized kernel. Start from the installed
general-shape native 476ad65e...5a37b. Keep its archive and all earlier evidence.

The dense dcc.gamma selector uses global index/mask scratch above the small shared
selector's capacity. Structured support keeps its actual kernel envelope; other
shapes use dense GPU. HSIC products use bounded shared tiles with the original
dot-product and final scalar accumulation order. Mathematical compiler flags,
alpha/index and the other public contracts are unchanged.

`build.py TAG` snapshots sources and recompiles the four affected translation
units with authenticated full-build commands, including method --fmad=false.
All other linked objects retain their parent hashes. This is an incremental
development build, not clean-checkout promotion evidence.

`campaign.py BUILD OUTPUT` freezes a protocol and runs fresh CPU references,
then ordinary and optional optimized GPU paths, including
three complete real inputs at their defaults, public 500x50 at default 50,
1025x4 at default 102, explicit 1/2/7/35/50/64/65/n-1, structured boundary
133/134 at k=64, and cache-disabled runs at k=1/65. Each complete search uses
m.max=Inf. Real/default500 cases also qualify CPU-authority final PDAG and RNG.

Every CI and finite pMax must agree with CPU within 1e-10; task/deletion order,
graph, sepsets, counts, RNG, GPU authority, cache metadata, and resource closure
must pass. Do not weaken gates after observing results. Timings are single-run
observations and establish no performance baseline. Preserve failed attempts.

Additional checks cover invalid rank rejection before truncation, cross-rank
result-cache isolation, three other methods (including permutation RNG), and
the historical 351x48/k35 regression. Local real data and bulky receipts are
not distributed; missing inputs/CUDA are failures for this campaign.

Attempt1 is retained: k=1 on dense140 passes every numerical/graph/resource
gate but naturally stops at level 1. The initial harness incorrectly required
level-2 prefill on every named dense input. The corrected harness requires and
records prefill when search reaches level 2; it does not force deeper search or
change numCol. Production source, native and numerical tolerances are unchanged.

Supplement1 exposed an existing all-result-hit accounting error with retained
eig lanes: it required a positive active lane count even when no components
missed. Candidate2 requires zero active lanes for zero misses, and a positive
count otherwise. The new cross-rank cache test covers this real runtime fix.

Supplement2 passes that cache test, then stops on an older one-call test's
exact column-name assertion. It omitted three existing worker/concurrency
diagnostic columns. The test's expected schema is updated explicitly; all
numerical and accounting assertions remain in place.

Supplement3 passes numCol and the one-call test, then exposes the new runner's
per-expression eval lifetime: a fixture's on.exit freed its handle before use.
The runner now evaluates the filtered expression vector once, as the existing
build-consistency runner does. No native change is involved.

Attempt2 was stopped by the original whole-machine monitor when another user's
GPU0 task started. Its completed receipts are preserved with their identities;
attempt3 authenticates twelve reused successes and completes on exclusive GPU1.
Other-GPU use is now recorded without stopping correctness checks. None of
these single observations establish an isolated performance median.

The separate rank-one check found the legacy R `diag(single_value)` construction
error and an empty-graph stopping mismatch. The R correction gives the diagonal
an explicit dimension. Candidate3 matches pcalg's `any(G)` condition in both the
CUDA scheduler and the CPU reference scheduler; no empty level is appended
after deleting the last edge. Attempt4 uses this final native for fresh CPU
and GPU runs. CPU mgcv and CPU Spectra numerical implementations are unchanged
and remain independent of the CUDA numerical implementation. Old receipts
retain their original native and R source identities.

`rank_one.py BUILD OUTPUT` checks 21 original/patched R results bitwise, compares
rank-one statistics and p-values with CPU full eig, and qualifies full WAN-PDAG
against pcalg after deletion of all edges at levels 0 and 1. `public.py BUILD
CAMPAIGN OUTPUT` calls the public 500x50 WAN-PDAG entry with numCol omitted.
`supplement.py BUILD OUTPUT` checks the three other methods including raw
pcalg n.edgetests, cache/errors, CUDA memcheck and full canonical regression.
`close.py BUILD CAMPAIGN SUPPLEMENT PUBLIC RANK_ONE audit|install` verifies
the final receipts and atomically installs the qualified native under its build
lock, preserving the parent archive. It does not commit or push the repository.

## Git handoff boundary

The Git handoff excludes the local `kpcalg/R/dcovgamma.R` patch. The recorded
November 6 receipts retain patched-R SHA
`05a30eb59a7404ab24e3137380862d2840b389fc3b17415b9c6045a1fa0e4a69`;
the original Git reference has SHA
`d0620dd9b06f006170b5e142b892c202619296797b2bf9ca842a7c2b5b3b715f`.
Both snapshots remain in the local artifact tree (`legacy-r-source` and
`rank-one-parent`). The archived `sealed-tools` directory preserves the exact
tools used at closure; this README additionally describes the later handoff.

`rank_one.py` documents that local patched-reference experiment. It does not
pass with the original R function, which errors at rank one; do not remove its
assertions or silently patch the Git reference to rerun it. Likewise `close.py`
authenticates the recorded local dependencies and is not a fresh-clone
qualification shortcut. Independent native CPU full eig provides the separate
rank-one GPU numerical comparison. The two patched-R WAN-PDAG fixtures end in
empty graphs and do not qualify nonempty rank-one orientation with original R.
