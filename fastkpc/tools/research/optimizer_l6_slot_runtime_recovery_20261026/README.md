# L6 isolated campaign continuation

The original six-run protocol and runtime are frozen. OFF1/ON1/ON2/OFF2
completed with all gates passing. An external GPU0 job interrupted OFF3 before
any complete result or timing was written; ON3 has not started.

`prepare.py ORIGINAL CONTINUATION` verifies the four completed receipts and the
interruption, copies the evidence into a new directory, and moves only the
copied incomplete OFF3 attempt into `invalidated/`. The original tree is kept
unchanged. This preparation does not start a GPU workload.

When both GPUs are idle, `resume.py CONTINUATION` executes the frozen original
campaign setup and run loop, skipping only already-completed valid runs. It
keeps the original numerical runner, native, source closure, settings, remaining
OFF3/ON3 order and performance gates. Any further interference is preserved as
another incomplete attempt. A completed numerical failure or completed timing
must never be discarded or retried by this continuation.

`audit.py CONTINUATION --pending` binds a partial handoff. After all six runs,
use `audit.py CONTINUATION`, then the original runtime campaign's `archive.py`
on the continuation directory. The original and continuation timings are not
pooled as extra observations: there are exactly six valid planned executions.
The earlier shared-host pair remains diagnostic-only.

Evidence remains incremental source closure, not a clean producer or promotion.
L8 stays stopped pending a complete isolated L6 result. Phase 10, candidate,
holdout, recommended route and default options are unchanged.
