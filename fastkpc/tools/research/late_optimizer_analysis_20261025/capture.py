"""Bind the captured windows to the unchanged native and complete canonical gates."""
import hashlib,json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[4];out=Path(sys.argv[1]).resolve()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
r=read(out/'capture/report.json');h=read(out/'capture/hook.json');p=read(out/'capture-process.json')
assert all(r['gates'].values()) and all(h['gates'].values()) and p['exit_code']==0 and not p['unexpected_gpu_pids']
assert h['capture'] and not h['preload'] and h['after']['capture_windows']==5
assert sha(out/'capture/receipt.rds')==r['receipt_sha256']==h['receipt_sha256']
inspect=root/'fastkpc/artifacts/full_cuda_ci/optimizer_l6_slots_20261025/inspect'
text=subprocess.check_output([str(inspect),str(out/'capture-raw')],text=True)
assert 'L6 windows=4 setups=245 targets=1148 iterations=17062 capacities=2:7,3:28,5:210, total_root_rank=48:245,' in text
assert 'L8 windows=1 setups=9 targets=18 iterations=378 capacities=2:9, total_root_rank=64:9,' in text
files=[out/'capture/report.json',out/'capture/hook.json',out/'capture/receipt.rds',out/'capture-process.json',
       out/'capture-protocol.json',out/'capture-processes.json',out/'capture.log',*sorted((out/'capture-raw').glob('*.bin'))]
assert len(list((out/'capture-raw').glob('*.bin')))==5
result=dict(schema='late-optimizer-capture-closure-v1',status='PASS',native_sha256=r['native_sha256'],
  evidence_class='GPU1-exclusive shared-host diagnostic with shadow D2H, not performance evidence',
  numerical_gates=r['gates'],hook_gates=h['gates'],oracle_errors=r['oracle_errors'],process=p,
  fixture_summary=text,inspector_sha256=sha(inspect),
  inspector_source_sha256=sha(root/'fastkpc/tools/research/optimizer_l6_slots_20261025/inspect.cpp'),
  files={str(f.relative_to(out)):sha(f) for f in files})
with (out/'capture-closure.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
print(text,end='');print('PASS capture closure',sha(out/'capture-closure.json'))
