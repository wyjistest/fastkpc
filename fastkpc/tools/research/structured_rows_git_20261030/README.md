# Structured row-only Git source closure

`verify.py` reads the sealed local build, isolated and shared-host evidence.
It rehashes the complete final manifest, checks the installed native, 44 linked
objects and generated includes, and compares qualification tools and regression
fixes to their recorded bytes. The historical shared-host live-native dependency
has exactly one explicit mapping to the frozen original native with the same SHA.
No archive, installed binary or source is rewritten; JSON is emitted on stdout.

Before committing, run from the repository root:

```sh
python3 fastkpc/tools/research/structured_rows_git_20261030/verify.py
```

After separate build-fix and runtime/tool commits, supply their actual hashes:

```sh
python3 fastkpc/tools/research/structured_rows_git_20261030/verify.py \
  --build-commit BUILD_FIX_COMMIT --runtime-commit RUNTIME_COMMIT
```

The second mode also compares every measured production input and qualification
file to the corresponding Git blob. Store the resulting JSON as the handoff's
source attestation. This is a post-run identity attestation; the original receipts
remain development source-closure evidence and are not relabeled as a clean
checkout campaign. No CUDA build or GPU test is launched. Syntax checks parse
Python/R and the build shell script without executing their numerical work.

Both modes require the authenticated local artifact trees. Missing files or
identity mismatches fail the command; a fresh clone alone does not supply these
datasets, natives or receipts. The verifier is a new handoff tool, separate from
the 25 original qualification files whose sealed hashes remain unchanged.
