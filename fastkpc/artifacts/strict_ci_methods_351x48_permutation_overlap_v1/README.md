# Strict permutation CI 351x48 overlap evidence v1

This directory indexes clean development receipts produced by runtime commit
`6c01d71` for the canonical default-`Inf` `dcc.perm` and `hsic.perm` full-CUDA
skeleton routes. The native library was rebuilt from that clean commit and has
SHA-256 `88eb8939759e6beb119126b06a2a110d2f7ca44ccda4333c8fef0109bc5fa59b`.
Execution used one RTX 4090, the existing setup/optimizer pipeline, its 5 ms
producer delay, and one in-flight strict-method preparation ticket.

## Performance

Measured outer elapsed time on the same canonical workload:

```text
method       optimized v2   overlap v1   saved      reduction
dcc.perm         849.693 s    839.726 s    9.967 s      1.17%
hsic.perm       1042.358 s    950.008 s   92.350 s      8.86%
total           1892.051 s   1789.734 s  102.317 s      5.41%
```

`dcc.perm` is a small improvement rather than the theoretical 117-second
overlap opportunity. Every preparation was ready by the end of its permutation
build, while preparation submission itself accumulated 459.429 seconds of host
time because QR cohorts retain internal waits. Concurrent permutation build
time rose from 117.275 to 139.876 seconds.

`hsic.perm` realizes a material wall-time improvement. Concurrent permutation
build time rose from 255.772 to 278.271 seconds, but the preparation overlap
still reduced outer elapsed by 92.350 seconds. The event-query lower bound is
intentionally conservative and does not count preparations that complete part
way through a permutation build.

## Structural gates

```text
metric                                      dcc.perm   hsic.perm
preparation submitted before RNG              229675      282113
ready immediately after submit                     0          68
ready after permutation                       229675      281890
deferred preparation errors                         0           0
hidden stream/device synchronization              0/0         0/0
submit completion-event waits                       0           0
intermediate host event waits                       0           0
final compact-result host waits                 229675      282113
in-flight peak                                       1           1
trusted payload rescans                              0           0
```

All established process gates pass:

```text
skeleton CI tests                         511,788
kpcalg-authority orientation CI tests          13
permutation p-value mismatches                  0
permutation final RNG-state mismatches          0
graph / sepset / pMax / n.edgetests mismatches  0
final kpcalg WAN-PDAG mismatches                 0
CPU skeleton authority / fallback               0 / 0
residual / component D2H bytes                  0 / 0
```

Skeleton authority remains full CUDA and production orientation authority
remains kpcalg CPU WAN-PDAG. Native CUDA orientation remains experimental.
This is development evidence: Phase 10 remains active, no candidate is frozen,
the recommended route is unchanged, and the sealed holdout remains closed.

Dense RDS payloads are retained locally under `payload/` and ignored by Git.
`manifest.json` records their sizes and SHA-256 values, runtime source closure,
native/dataset identity, exact parity gates, and rebuild commands.
