# Strict CI methods 351x48 optimized evidence v2

This directory indexes clean development receipts produced from commit
`830cfbe12559ee5b2be1e861b7be20defe69cbdb` for the default-`Inf` 351x48
`hsic.gamma`, `dcc.perm`, and `hsic.perm` full-CUDA skeleton routes.

The native library was rebuilt from clean runtime sources and has SHA-256
`443e2538e6f2030bbd071fa769a53b8e397c3cb73bc5b38c55fc6393c5eafbd3`.
All runs enabled the opt-in original-window setup/optimizer pipeline with a
5 ms producer delay. Execution used one RTX 4090.

## Retained changes

```text
authoritative fixed-SP root cache:
  enabled for all supported full-CUDA CI methods
  roots come only from the authoritative fixed-SP eigensolver
  every hit reauthenticates shape and penalty-block bits

setup/optimizer pipeline:
  original windows, cohorts, target count, and target order are unchanged
  R and Rcpp objects remain on the main thread
  one owned optimizer input may be pending

permutation authentication:
  C++17 move-only SealedPermutationTableHandle
  private owned payload with no mutable consumer view
  incremental OpenSSL SHA-256 during row generation
  trusted one-call consumer uses the sealed attestation without rescanning
  untrusted boundaries still require a full payload rehash
```

The GPU residual/component preparation and R permutation generation are still
serial in this version. No asynchronous permutation/GPU overlap is claimed.

## Performance

Measured outer elapsed time on the same canonical workload:

```text
method       optimized v1   optimized v2   saved       reduction
hsic.gamma      631.522 s      593.814 s    37.708 s      5.97%
dcc.perm        945.359 s      849.693 s    95.666 s     10.12%
hsic.perm      1152.905 s     1042.358 s   110.547 s      9.59%
total         2729.786 s     2485.865 s   243.921 s      8.94%
```

The current stage boundaries are:

```text
method       setup      optimizer   residual    component   permutation+SHA
hsic.gamma    25.540 s    161.927 s   383.800 s    31.272 s       0.000 s
dcc.perm      23.194 s    159.501 s   455.173 s    41.944 s     117.275 s
hsic.perm     23.035 s    140.283 s   433.361 s   120.492 s     255.772 s
```

For `dcc.perm`, trusted request-identity build and validation now total about
1.469 seconds, versus 108.851 seconds in optimized v1. For `hsic.perm`, they
now total about 1.764 seconds, versus 135.129 seconds. The larger
permutation-table boundary above includes the new incremental digest work, so
only complete outer elapsed time is used to judge the net improvement.

## Correctness and authority

All established development gates pass:

```text
skeleton CI tests                         834,467
kpcalg-authority orientation CI tests          19
permutation p-value mismatches                  0
permutation final RNG-state mismatches          0
hsic.gamma decision flips                       0
graph / sepset / pMax / n.edgetests mismatches  0
final kpcalg WAN-PDAG mismatches                 0
CPU skeleton authority / fallback               0 / 0
residual / component D2H bytes                  0 / 0
```

`hsic.gamma` retains its existing absolute `1e-10` p-value contract; its
maximum observed difference is `8.271161533457416e-14`. Both permutation
methods remain bitwise exact, including the final R RNG state. Skeleton
authority is full CUDA; production orientation authority remains kpcalg CPU
WAN-PDAG. Native CUDA orientation remains experimental.

`manifest.json` records payload sizes and SHA-256 values, producer source
closure, native-library identity, performance details, and parity receipts.
Dense RDS payloads remain local and ignored by Git.

This is development evidence, not promotion evidence. No candidate is frozen,
the recommended route is unchanged, and the sealed holdout remains
`SEALED_NOT_RELEASED`.
