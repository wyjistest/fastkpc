# Strict CI methods 351x48 evidence v1

This directory is the persistent local evidence index for the canonical
351x48 qualification of `hsic.gamma`, `dcc.perm`, and `hsic.perm`.

The production architecture is:

```text
R binary64 matrix
  -> full-CUDA skeleton
  -> adjacency / sepsets / pMax / n.edgetests
  -> kpcalg CPU WAN-PDAG authority
  -> final PDAG
```

`manifest.json` is intended for Git tracking. Dense RDS payloads live below
`payload/` and remain ignored by Git; the manifest records each payload's
absolute and relative path, byte size, and SHA-256. It also records source and
native-binary identities, method parameters, parity results, orientation RNG
state hashes, and rebuild commands.

This is canonical development qualification, not promotion evidence. Phase 10
remains active, the candidate is not frozen, the recommended route is
unchanged, and the external holdout remains `SEALED_NOT_RELEASED`.

Rebuild the index without recomputing the large numerical runs:

```bash
Rscript fastkpc/tools/publish_strict_ci_methods_351x48_evidence.R \
  fastkpc/artifacts/strict_ci_methods_351x48_v1
```

The publisher fails closed if an input receipt fails, a persistent payload has
a different hash, aggregate counts change, or any method no longer has SHD 0
and complete process parity.
