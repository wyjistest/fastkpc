# Replacement capacity probe, declared before execution

The conservative coexistence probe retained both the original prepared windows
and 32 extra capacity-18 slots. It observed 1,791,819,776 free bytes, below the
unchanged 2 GiB gate. Its full 34 windows / 2,154 setups / 17,767 targets and
576 extra warm targets are preserved. This is a failed conservative probe, not
a capacity PASS or a proven replacement STOP. No production source has changed.

The next probe runs the unchanged complete canonical producer. At the L4
`build_full_cuda_ci_level_components` entry, the original optimizer windows
have completed and their multi-penalty handles have been released. All L4 setup
geometry, the actual completed coefficient arena and the complete Stage A
component/residual storage are still resident. This is the proposed pool's
last live boundary, immediately before Stage B.

At that boundary, replay the full captured L4 layer using the already validated
research slot DSO: 32 slots, capacity 18, original-order queue, all 2,154 setups /
17,767 targets, unchanged numerical instructions. Preserve the full 105,038,504
byte coefficients/fitted/residual arena, which is larger than the runtime's
5,259,032 byte coefficient payload, and compare all captured outputs after
completion. Stop and release the research pool before resuming real Stage B.

Required, unchanged: <=1 GiB incremental physical memory over this warm native
state, >=2 GiB free throughout creation, concurrent work and destruction;
full captured output equality; full canonical gates and native resource
closure. Observe every CUDA allocation and sample cudaMemGetInfo every 10 ms.
This is a cache-resident captured replay, not runtime integration or a timing
result. Candidate runtime must separately retain the same headroom checks
through its own actual lifecycle and pass full OFF/ON qualification.

This replacement measurement removes the explicitly identified double-counted
window handles. It does not lower the capacity requirement. If it fails, do
not integrate the 32-slot runtime under the existing memory policy.
