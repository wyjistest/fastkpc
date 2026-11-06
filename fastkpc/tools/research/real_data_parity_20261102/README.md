# Real-data correctness rerun, 2026-11-02

Use the installed `b2b79fc6...e135b12b` native and verify the 107 production
source hashes from the October 30 attestation. Do not rebuild or change
production options globally. No sealed holdout is used.

Run one fresh process on physical GPU1 at a time, CPU affinity 0-19 and
single-threaded math libraries. Reset dataset caches. Save inputs' identities,
commands, GPU/process samples, full receipts, and failures outside timed calls.
Elapsed times are observations from correctness runs, not a promotion campaign.

1. `canonical.py`: rerun the accepted structured-row/work-pair/slots/postback
   configuration on the complete 351x48 real cancer input, reusing the unchanged
   canonical qualification runner and authenticated full CPU skeleton oracle.
2. `cases.py attempt1`: independently recompute complete CPU and GPU WAN-PDAG
   results for the 179x8, 243x9 and 300x10 historical real inputs. Use
   `dcc.gamma`, alpha 0.1, index 1, numCol 35, and m.max=Inf. The CPU authority
   is pcalg stable skeleton with the repository's kpcalg/mgcv CI implementation;
   invalid p-values fail the run. General GPU settings are exact-primary,
   coefficient reuse, 11 GiB component cache, residual bypass and 8 eig lanes.
   Canonical-only structured/slots/postback options stay disabled for these
   other shapes. These are complete input matrices, not cropped samples.
3. The final case independently runs CPU-authority WAN-PDAG orientation on
   the new canonical GPU skeleton and on the archived CPU skeleton. It does
   not recompute the large canonical CPU skeleton or claim GPU orientation.

Predeclared gates: every CI and finite pMax absolute error <=1e-10, special
pMax values exact, logical CI identity/order and decisions exact, zero decision
flips, skeleton adjacency, undirected sepset unions, n.edgetests, orientation
CI order/decisions, final PDAG and RNG endpoint equal. Require zero CPU
numerical fallback in the GPU skeleton, zero large residual/component D2H,
stable-SVD route, cache metadata agreement, and resources released. Preserve
failures and investigate; never weaken a numerical gate after seeing output.

The historical Shiny application's result files are not numerical authority.
The three small CPU references are generated afresh. Canonical CPU skeleton
evidence retains its original identity. Phase 10/promotion/holdout status is
unchanged by this campaign.

## Comparator correction before attempt2

Attempt1 is retained as FAIL: on 179x8 all 338 p-values, pMax, graph, orientation,
RNG and resource checks passed, but the new harness incorrectly required raw
pcalg node-major call order to equal native undirected-edge-major call order.
The existing WAN-PDAG authority test already compares pcalg by CI identity;
the established logical-order contract uses the native CPU scheduler.

Attempt2 keeps the raw pcalg order observation (false), requires identical
CI identities and all original numerical/graph checks, and additionally replays
the independently computed CPU p-values through the existing CPU scheduler.
Missing speculative CPU evaluations are computed with the same CPU oracle,
never filled from GPU values. The unchanged logical-trace normalizer is used
to require exact native task order and deletion trace. The CPU replay itself
must also reproduce the independently searched pcalg graph, pMax, sepsets and
test counts. No production code, numerical tolerance or input has changed.

Attempt2 exposed a harness-only empty-terminal-layer bug in that new CPU
provider: R's `paste()` recycled the scalar level against zero-length columns.
Attempt3 explicitly returns `numeric()` for a zero-row plan. Attempt2 logs and
its complete GPU/CPU result files are retained; no production change is made.

Attempt3 then exposed the generic CPU replay API's zero-count padding through
its requested conditioning ceiling (53,172,109,4,0,0,0 versus 53,172,109,4).
Its 338 logical rows/order/decisions and pMax were already exact. Attempt4
replays through the independently completed CPU search's last observed level;
both original GPU and pcalg searches still use m.max=Inf. This changes only the
auxiliary replay's empty tail, not the searches or their strict count check.
