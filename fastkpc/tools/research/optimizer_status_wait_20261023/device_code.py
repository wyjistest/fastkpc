"""Compare all five capacity objects' SASS, ignoring anonymous TU hash names."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

root=Path(__file__).resolve().parents[4]
out=Path(sys.argv[1]).resolve()
old=root/'fastkpc/artifacts/full_cuda_ci/l2_prefill_runtime_20261022'
old_build=json.loads((old/'build.json').read_text())
new_build=json.loads((out/'build.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def sass(path):
    digest=hashlib.sha256();functions=0
    process=subprocess.Popen(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(path)],
                             stdout=subprocess.PIPE,text=True)
    for line in process.stdout:
        # CUDA embeds a translation-unit hash in anonymous-namespace symbols.
        # Keep the capacity file name, full function name and every instruction.
        line=re.sub(r'_GLOBAL__N__[0-9a-f]{8}_(\d+_mgcv_multi_penalty_gcv\w*_cu_)[0-9a-f]{8}',
                    r'_GLOBAL__N__SOURCE_\1SOURCE',line)
        functions+='Function :' in line
        digest.update(line.encode())
    assert process.wait()==0 and functions>0
    return dict(sha256=digest.hexdigest(),functions=functions)
rows={}
for suffix in ['','_small64','_ext192','_ext384','_ext559']:
    name='mgcv_multi_penalty_gcv'+suffix+'.o'
    a=old/'build-source/fastkpc/build'/name
    b=out/'build-source/fastkpc/build'/name
    assert sha(a)==old_build['objects'][name] and sha(b)==new_build['objects'][name]
    before,after=sass(a),sass(b)
    rows[name]=dict(old_object_sha256=sha(a),new_object_sha256=sha(b),
                    old_sass=before,new_sass=after,exact=before==after)
    print(name,'PASS' if before==after else 'FAIL',flush=True)
(out/'device-code.json').write_text(json.dumps(dict(objects=rows,
    status='PASS' if all(r['exact'] for r in rows.values()) else 'FAIL'),indent=2)+'\n')
assert all(r['exact'] for r in rows.values()),'device instructions changed'
