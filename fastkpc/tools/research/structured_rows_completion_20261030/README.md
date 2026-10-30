# Structured row-only isolated completion

This tool adds descriptive paired stage accounting after the unchanged original
six-run isolated campaign. It does not modify gates, drop runs, replace outer
elapsed with adjusted elapsed, or establish a cause for L7/L8 timing differences.
The existing qualified natives and production source are preserved.

```sh
python3 fastkpc/tools/research/structured_rows_20261029/campaign.py
python3 fastkpc/tools/research/structured_rows_completion_20261030/completion.py analyze
python3 fastkpc/tools/research/structured_rows_20261029/archive.py audit
python3 fastkpc/tools/research/structured_rows_20261029/archive.py install
python3 fastkpc/tools/research/structured_rows_completion_20261030/completion.py verify-history
python3 fastkpc/tools/research/structured_rows_20261029/archive.py seal
```

Only the original capacity and non-regression GO permits installation. A speed
GO is not required. The original campaign, qualification, pending and shared-host
receipts keep their original bytes and identities; default flags stay unchanged.

The shared-host manifest records the old installed native at its live build
path. After authorized installation, historical verification checks that single
dependency against the already frozen original native in
`build_consistency_20261029/parent/fastkpc_cuda.so`, with exactly the original SHA.
The explicit mapping and both native identities are recorded separately. Every
other historical dependency must still match its original path/hash. This is a
recorded installation transition, not a rewrite of the shared-host evidence.

Outputs are new files under
`structured_rows_20261029/completion_20261030/`, included by the original final
seal. Source/receipts are local development evidence, not a clean committed
producer, promotion, new real-input qualification or final-PDAG campaign.
