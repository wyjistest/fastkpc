"""Bind the completed bounded diagnostic to immutable native/source/fixture hashes."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[4];here=Path(__file__).resolve().parent
out=Path(sys.argv[1]).resolve()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
analysis=json.loads((out/'analysis.json').read_text())
capture=json.loads((out/'capture.json').read_text())
assert all(capture['gates'].values())
authority=json.loads(subprocess.check_output(['Rscript','--vanilla',str(here/'qualify.R'),str(out)],text=True))
assert all(authority['gates'].values())
(out/'capture-authority.json').write_text(json.dumps(authority,indent=2)+'\n')
builds=[json.loads((out/(name+'-build.json')).read_text()) for name in ('capture','replay')]
for b in builds:
    assert sha(root/'fastkpc/build/fastkpc_cuda.so')==b['native_sha256']
    assert all(sha(root/p)==v for p,v in b['source_closure'].items())
    assert all(sha(root/p)==v for p,v in b['sources'].items())
for name in ('L3-W31','L5-W7'):
    text=(out/('memcheck-'+name+'.log')).read_text()
    assert 'ERROR SUMMARY: 0 errors' in text and '0 bytes leaked' in text
    assert (out/('profile-'+name+'.sqlite')).exists()
assert len(analysis['windows'])==4 and len(analysis['profiles'])==2
assert all(w['ratio_to64']['16']>1 and w['ratio_to64']['32']>1 for w in analysis['windows'].values())
telemetry={}
for path in out.glob('*-gpu.csv'):
    with path.open() as f:data=list(csv.reader(f))
    g0=[r for r in data if len(r)>=7 and int(r[1])==0]
    g1=[r for r in data if len(r)>=7 and int(r[1])==1]
    assert g0 and g1
    telemetry[path.stem]=dict(gpu0_sampled_peak_mib=max(float(r[2]) for r in g0),
      gpu0_sm_clock_min_mhz=min(float(r[5]) for r in g0),gpu0_sm_clock_max_mhz=max(float(r[5]) for r in g0),
      gpu1_utilization_max=max(float(r[3]) for r in g1),gpu1_sampled_peak_mib=max(float(r[2]) for r in g1))
summary=dict(schema='optimizer-window-closure-v1',status='PASS_CAPTURED_WINDOW_DIAGNOSTIC',
 native_sha256=capture['native_sha256'],capture_gates=capture['gates'],
 all_replay_bitwise=True,memcheck_errors=0,memcheck_leaked_bytes=0,
 capture_cpu_authority_fallback_zero=True,
 telemetry=telemetry,analysis_sha256=sha(out/'analysis.json'),
 scope='L3 W31/W32 and L5 W7/W8, 64 unchanged setups per window, workers 16/32/64',
 production_sources_unchanged=True,production_defaults_unchanged=True,
 decisions=dict(reduce_workers='STOP_FOR_CAPTURED_WINDOWS',
   bounded_two_window_replay='GO_AS_NEXT_DIAGNOSTIC',runtime_rolling_integration='NOT_YET_GO'),
 phase10='ACTIVE',candidate='NOT_FROZEN',holdout='SEALED',recommended=False,
 evidence_class='captured-cohort development diagnostic; not full-method speedup or promotion')
(out/'closure.json').write_text(json.dumps(summary,indent=2)+'\n')
environment=dict(affinity=sorted(os.sched_getaffinity(0)),cuda_visible_devices='0',
 cuda_device_max_connections='32',openblas_threads=1,omp_threads=1,mkl_threads=1,
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total,power.limit',
   '--format=csv'],text=True),
 nvcc=subprocess.check_output(['/usr/local/cuda/bin/nvcc','--version'],text=True),
 nsys=subprocess.check_output(['nsys','--version'],text=True))
(out/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
files=sorted(p for p in out.rglob('*') if p.is_file() and p.name!='manifest.json')
manifest=dict(schema='optimizer-window-manifest-v1',files={str(p.relative_to(out)):sha(p) for p in files},
 tools={str(p.relative_to(root)):sha(p) for p in sorted(here.iterdir()) if p.is_file()},
 native_sha256=capture['native_sha256'],source_closure=builds[0]['source_closure'])
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('PASS closure; manifest SHA256',sha(out/'manifest.json'))
