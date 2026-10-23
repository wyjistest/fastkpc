"""Export authenticated archived bytes for the production source-shared test."""
import gzip
import io
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, 'fastkpc/tools/research/structured_adapter_20261022')
import run as R

out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
base=Path('fastkpc/artifacts/full_cuda_ci/structured_adapter_20261022')
manifest=json.loads((base/'manifest.json').read_text())
p=base/'real640.npz.gz'
entry=manifest['files'][p.name]
assert R.sha(p)==(entry['sha256'] if isinstance(entry,dict) else entry)
x,keys,_=R.real_fixture(False)
z=np.load(io.BytesIO(gzip.decompress(p.read_bytes())))
x.tofile(out/'residuals.f64')
np.ascontiguousarray(z['structured_payload'][:,:-1]).tofile(out/'three-fields.f64')
(out/'fixtures.json').write_text(json.dumps(dict(
    source_manifest_sha256=R.sha(base/'manifest.json'),
    source_payload_sha256=R.sha(p),
    files={n:R.sha(out/n) for n in ('residuals.f64','three-fields.f64')}),indent=2)+'\n')
