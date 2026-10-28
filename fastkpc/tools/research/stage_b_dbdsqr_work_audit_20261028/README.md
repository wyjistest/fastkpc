# Fixed-VT DBDSQR work-lifetime audit

Read-only audit of the accepted structured Stage B numerical implementation.
No CUDA execution, CPU numerical evaluator, production edit, compilation,
prototype implementation, performance measurement or driver-policy change.

The scope is the canonical n=351 caller with UPLO=L, NCVT=1, NRU=NCC=0.
Distinguish dataset rows N, distinct residual values G, DBDSQR dimension
r=G-1 and work-segment length p=r-1. Do not substitute optimizer q for r.

`audit.py OUTPUT_DIRECTORY`:

1. Verifies the accepted native, all 103 production inputs, the generated
   DBDSQR text and its exact mechanical derivation from the vendored source.
2. Inventories every work store and pointer consumer in that exact function.
   This is a source-specific inventory, not a general C++ data-flow analyzer.
3. Checks integer address bounds and symbolic value/order equivalence for a
   proposed two-segment layout. It never evaluates a floating-point QR/SVD.
   It includes direction switches and a counterexample to blind aliasing.
4. Computes shared-memory bounds with the recorded hardware parameters and
   allocation granularity. These are hypothetical bounds, not measured
   occupancy, generated-code resources or performance.
5. Writes a new evidence directory with input snapshots, access inventory,
   results and a manifest. Refuses to overwrite an existing directory.

Human review supplies the control-flow/lifetime argument: the model validates
that argument's indices and operand ordering, not the compiled GPU program.
Every future prototype still needs GPU bitwise, error-path, resource and full
captured Stage B qualification. Production integration is not authorized by
this static result. The accepted 329.078 s median and Phase 10 states remain.
