"""Export signed scalar bits from the previously qualified GPU payload probe."""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('old',type=Path)
parser.add_argument('out',type=Path)
parser.add_argument('--prepare-only',action='store_true')
args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prior=json.loads((args.old/'native-full/report.json').read_text())
keys=(args.old/'native/keys.txt').read_text().splitlines()
changed=prior['unique_changed_keys']['self_moment']
assert len(keys)==6632 and len(changed)==1002
(args.out/'changed-keys.txt').write_text('\n'.join(changed)+'\n')
if args.prepare_only:
    print('prepared 1002 changed keys',flush=True)
    sys.exit(0)
binding=json.loads((args.old/'build/build.json').read_text())
assert sha(args.old/'build/probe.so')==binding['probe_sha256']
assert sha(Path('fastkpc/build/fastkpc_cuda.so'))==binding['native_sha256']==prior['native_sha256']
if '/usr/local/lib/R/lib' not in os.getenv('LD_LIBRARY_PATH','').split(':'):
    env=dict(os.environ); env['LD_LIBRARY_PATH']='/usr/local/lib/R/lib:'+env.get('LD_LIBRARY_PATH','')
    os.execve(sys.executable,[sys.executable,*sys.argv],env)
ct.CDLL('/usr/local/lib/R/lib/libR.so',mode=ct.RTLD_GLOBAL)
ct.CDLL(str(Path('fastkpc/build/fastkpc_cuda.so').resolve()),mode=ct.RTLD_GLOBAL)
lib=ct.CDLL(str((args.old/'build/probe.so').resolve()))
ptr=ct.POINTER(ct.c_double)
lib.payload_probe_create.argtypes=[ct.c_int]*4+[ct.c_ulonglong]
lib.payload_probe_create.restype=ct.c_void_p
lib.payload_probe_run.argtypes=[ct.c_void_p,ptr,ct.c_int,ptr,ct.POINTER(ct.c_int)]
lib.payload_probe_destroy.argtypes=[ct.c_void_p]
lib.payload_probe_snapshot.argtypes=[ct.POINTER(ct.c_longlong)]
def snapshot():
    s=(ct.c_longlong*4)(); assert lib.payload_probe_snapshot(s)==0; return list(s)
before=snapshot(); probe=lib.payload_probe_create(351,35,22,8,11811160064); assert probe
entry=351*35+35+2
original=np.empty(len(keys),dtype='<f8'); candidate=np.empty_like(original)
id_by_key={key:i for i,key in enumerate(keys)}
raw=args.old/'native/raw'
comparisons=0
try:
    for repeat,order in enumerate((keys,keys[::-1],keys)):
        for offset in range(0,len(order),22):
            group=order[offset:offset+22]
            x=np.asfortranarray(np.column_stack([np.fromfile(raw/(key+'.residual.f64'),dtype='<f8') for key in group]))
            output=np.empty((len(group),entry),dtype='<f8'); lanes=ct.c_int()
            assert lib.payload_probe_run(probe,x.ctypes.data_as(ptr),len(group),output.ctypes.data_as(ptr),ct.byref(lanes))==0
            assert lanes.value==min(8,len(group)) and np.isfinite(output).all()
            for j,key in enumerate(group):
                reference=np.fromfile(raw/(key+'.f64'),dtype='<f8')
                assert np.array_equal(reference[:-1].view('u8'),output[j,:-1].view('u8'))
                i=id_by_key[key]
                if repeat==0:
                    original[i]=reference[-1]; candidate[i]=output[j,-1]
                else:
                    assert candidate[i:i+1].view('u8')[0]==output[j,-1:].view('u8')[0],key
                comparisons+=1
        print('candidate pass',repeat+1,'comparisons',comparisons,flush=True)
finally:
    lib.payload_probe_destroy(probe)
after=snapshot(); assert before==after
actual=[key for i,key in enumerate(keys) if original[i:i+1].view('u8')[0]!=candidate[i:i+1].view('u8')[0]]
assert set(actual)==set(changed)
np.savez(args.out/'scalars.npz',original=original,candidate=candidate)
(args.out/'keys.json').write_text(json.dumps(keys)+'\n')
report=dict(schema='singleton-scalar-export-v1',status='PASS_EXPORT',keys=len(keys),changed=len(actual),
 comparisons=comparisons,native_sha256=binding['native_sha256'],
 original_report_sha256=sha(args.old/'native-full/report.json'),
 probe_sha256=binding['probe_sha256'],build_sha256=sha(args.old/'build/build.json'),
 runner_sha256=sha(Path(__file__)),scalars_sha256=sha(args.out/'scalars.npz'),
 keys_sha256=sha(args.out/'keys.json'),resources_before=before,resources_after=after,
 max_abs=float(np.max(np.abs(candidate-original))),
 max_relative=float(np.max(np.abs(candidate-original)/np.abs(original))),
 policy='GPU produces all candidates; CPU only compares/serializes; natural/reverse/repeat22 exact')
(args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
